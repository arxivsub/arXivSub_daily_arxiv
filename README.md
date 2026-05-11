# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-05-11 | 今日论文总数: 836

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Learning and Reusing Policy Decompositions for Hierarchical Generalized Planning with LLM Agents

**arXiv ID:** 2605.06957 | [PDF](https://arxiv.org/pdf/2605.06957v1)

**作者:** Shirin Sohrabi `[一作]` (IBM), Michael Katz `[通讯]` (IBM)

**通讯引用:** 3375 | [OpenAlex ID](https://openalex.org/A5103010384)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于LLM的动态策略学习框架，融合通用规划与层级任务分解，能够从成功执行中自动提取、验证并泛化可重用的可执行策略组件；

**💡 创新点**

创新点在于实现无符号域模型的通用规划与HTN分解，利用执行反馈动态生成可重用的参数化政策组件，并通过语义检索和聚类实现跨域知识迁移；

**🔧 技术方法**

核心技术包括自然语言抽象化、语义检索(BAAI嵌入)、LLM代码生成、组件拆解与验证、聚类归纳与去重、以及多轮调试与评估；

**📊 数据集**

在AppWorld基准（含训练、普通测试与挑战测试集）上进行评估，挑战集包含未见应用如Gmail、Amazon；

**📈 对比分析**

与多种基线（Claude Opus、ReAct、LOOP等）对比，使用Claude Sonnet 4.6时在普通/挑战测试中分别达到98.2%/97.8%和98.3%/97.8%的场景完成率，显著优于无重用版与公开模型，且在调试次数与成本上更高效；

**⚠️ 局限性**

主要局限包括对已定义元域与场景边界的依赖、组件库规模扩展导致的token/计算开销、以及对未知域或无明确情境分组的适应性有限。

---

## 2. Equivalence of Coarse and Fine-Grained Models for Learning with Distribution Shift

**arXiv ID:** 2605.07005 | [PDF](https://arxiv.org/pdf/2605.07005v1)

**作者:** Adam R. Klivans `[一作]` (University of Texas at Austin), Arsen Vasilyan `[通讯]` (University of Texas at Austin)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5030920630)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `f7dab867-23a8-4241-85e9-4ba79c6402f9`

**🎯 论文内容**

本文证明了分布无关下的 PQ 学习（按点拒绝）与 TDS 学习（整体拒绝）等价，并给出了从 PQ 学习到 TDS 学习的黑箱化简；随后利用此等价得到半空间类的 TDS 学习硬性结果；最后提出了利用成员查询实现高效 PQ / TDS 学习半空间的算法。

**💡 创新点**

核心创新在于构造了一个分支程序（branching program）与马尔科夫提升（martingale boosting）相结合的减化过程，能够将能拒绝整体分布的 TDS 学习器转化为能够按点拒绝的 PQ 学习器；并首次展示了成员查询在 TDS 学习中的价值，突破了先前关于查询无效的结论。

**🔧 技术方法**

主要技术包括：1) 通过混合论证得到单样本弱区分器；2) 采用分支程序与平衡（balance）函数实现弱区分器的提升；3) 使用 Forster 变换将半空间映射为大边距半空间；4) 通过决策列表和 VC 维度分析控制拒绝率与误差。

**📊 数据集**

该工作全部为理论分析，未使用实测数据集；所有实验和比较均基于理论复杂度和假设下的抽象分布。

**📈 对比分析**

由于是理论论文，未与其他方法进行实验对比；主要通过归纳推导得到硬性上界（如半空间 TDS 学习的 NP-hardness）和算法时间复杂度（多项式阶），而成员查询算法则证明了在分布无关情况下可实现多项式时间 PQ 学习。

**⚠️ 局限性**

限制包括：1) 仅适用于分布无关场景，分布特定情况下的等价性尚未证明；2) 对于非半空间概念类，TDS 学习仍缺乏多项式时间算法；3) 需要成员查询操作，实际部署可能受限；4) 对训练分布的假设（如存在非零边距）是实现算法的前置条件。

---

## 3. SHARP: A Self-Evolving Human-Auditable Rubric Policy for Financial Trading Agents

**arXiv ID:** 2605.06822 | [PDF](https://arxiv.org/pdf/2605.06822v1)

**作者:** Xiwen Chen `[一作]` (Morgan Stanley), Huayu Li `[通讯]` (University of Arizona)

**通讯引用:** 1948 | [OpenAlex ID](https://openalex.org/A5100721211)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于规则表的自适应LLM交易框架（SHARP），将无约束的文本优化改为结构化的条件‑动作规则演化；

**💡 创新点**

创新点在于通过可审计的规则表与三代理演化循环（归因、演化、验证）解决信用分配和策略漂移问题，实现精准、可解释的规则更新；

**🔧 技术方法**

核心技术包括GPT‑4o‑mini/GPT‑4.1‑mini及开源模型、结构化规则表、跨样本归因推理、原子规则变异、走‑前验证门控和符号化信用分配；

**📊 数据集**

使用了三类16只股票的公开数据（AI Tech、Biotech、Consumer Discretionary），结合1‑3天新闻、价格特征和宏观变量，在2025‑2026年进行走‑前回测；

**📈 对比分析**

与非LLM基准（随机、动量、均值回复）、静态规则LLM以及Lopez‑Lira、FinCon、FinHEAR等先进LLM系统对比，SHARP在GPT‑4o‑mini上实现+33%总回报、SR 2.45、最大回撤7.3%，比静态规则提升10‑20%，并在开源模型上从负回报提升至正回报；

**⚠️ 局限性**

局限性包括回测未考虑滑点、市场冲击、信号衰减与长周期制度变化，依赖单一公共新闻来源，缺乏实盘验证，以及LLM本身不具备真实金融推理能力。

---

## 4. SpecBlock: Block-Iterative Speculative Decoding with Dynamic Tree Drafting

**arXiv ID:** 2605.07243 | [PDF](https://arxiv.org/pdf/2605.07243v1)

**作者:** Weijie Shi `[一作]` (Hong Kong University of Science and Technology), Xiaofang Zhou `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 24295 | [OpenAlex ID](https://openalex.org/A5011384237)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种块迭代式的草稿生成器 SpecBlock，用于加速大语言模型推理。

**💡 创新点**

在单次前向中生成 K 个依赖的候选位置，并通过层级隐藏状态移位与可训练的 rank 头动态控制树宽度，同时使用有效前缀学习和成本感知的在线更新。

**🔧 技术方法**

SpecBlock 结合了 Transformer 解码器、层级隐藏状态移位、可训练的 rank 头、有效前缀掩码、验证器树、以及基于 verifier 反馈的成本感知 bandit 更新。

**📊 数据集**

在 UltraChat-200K、ShareGPT 以及多种评测基准（MT‑Bench、HumanEval、MATH‑500、Alpaca、Natural Questions、WMT‑23）上进行训练和评测。

**📈 对比分析**

与标准的 Speculative Sampling、Medusa、ParallelSpec、Falcon、EAGLE‑3 等对比，SpecBlock 在三款目标模型上平均提升 8–13% 的速度，并通过成本感知适配进一步提升到 11–19%。

**⚠️ 局限性**

仍然受限于模型规模和提示分布漂移；当输入长度很短或在低温设置下，接受率提升有限，且在线更新仍需额外计算开销。

---

## 5. A Finite-Iteration Theory for Asynchronous Categorical Distributional Temporal-Difference Learning

**arXiv ID:** 2605.06866 | [PDF](https://arxiv.org/pdf/2605.06866v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 6. CARMEN: CORDIC-Accelerated Resource-Efficient Multi-Precision Inference Engine for Deep Learning

**arXiv ID:** 2605.06878 | [PDF](https://arxiv.org/pdf/2605.06878v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 7. Predictive but Not Plannable: RC-aux for Latent World Models

**arXiv ID:** 2605.07278 | [PDF](https://arxiv.org/pdf/2605.07278v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 8. SplatWeaver: Learning to Allocate Gaussian Primitives for Generalizable Novel View Synthesis

**arXiv ID:** 2605.07287 | [PDF](https://arxiv.org/pdf/2605.07287v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 9. EgoPro-Bench: Benchmarking Personalized Proactive Interaction in Egocentric Video Streams

**arXiv ID:** 2605.07299 | [PDF](https://arxiv.org/pdf/2605.07299v1)

**作者:** Dongchuan Ran `[一作]` (Sensetime Reasearch), Lewei Lu `[通讯]` (Sensetime Reasearch)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 EgoPro‑Bench，一个面向个性化主动交互的第一人称视角视频基准，并基于该基准训练了 ProAct‑Stream 模型；

**💡 创新点**

创新点在于：①整合事件驱动与意图驱动两大分支，实现精细化时间标注与用户画像驱动的主动响应；②提出“短思考，优交互”原则，通过限制推理 token 提升低延迟下的主动性；③设计统一的评估协议与多维度指标，客观衡量响应时机与质量；

**🔧 技术方法**

使用的技术包括：多模态大型语言模型（如 Qwen3‑VL‑8B‑Instruct）、两阶段训练（SFT + RL，采用 GSPO）、流式视频处理、用户画像生成、LLM‑based 评判器与奖励函数；

**📊 数据集**

数据集为 EgoPro‑Bench，包含 12,000+ 训练样本和 2,400 测试样本，覆盖 12 个领域（事件驱动：对象、动作；意图驱动：工作、旅行、运动、艺术、导航等）；

**📈 对比分析**

与公开的事件驱动基准（OmniMMI、StreamingBench、OvO‑Bench）及意图驱动子集进行对比，ProAct‑Stream 在绝大多数指标上超过同类模型，即使参数仅 8B；在精度、召回、F1 以及 GHA 等方面显著优于基线；

**⚠️ 局限性**

局限性包括：①对极端视觉噪声与快速运动场景仍易出现误判；②模型在低资源场景下仍需进一步压缩与加速；③评估主要基于合成用户画像，真实用户实验尚待验证。

---

## 10. Toward Quantum-Safe 6G: Experimental Evaluation of Post-Quantum Cryptography Techniques

**arXiv ID:** 2605.06881 | [PDF](https://arxiv.org/pdf/2605.06881v1)

**作者:** Ananya Kudaloor `[一作]` (Toshiba Europe Ltd), Adnan Aijaz `[通讯]` (Toshiba Europe Ltd)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

对NIST标准化的后量子密码学（ML‑KEM、ML‑DSA、Falcon）在6G核心、边缘和IoT环境中的原语性能、TLS1.3握手效率以及信令负载进行了系统性实证评估。

**💡 创新点**

提出了跨层次评估框架，将算法级吞吐、握手吞吐（CPS）、Ciphertext扩展率（CER）和TCP payload量级统一量化，首次揭示PQC在多种硬件平台与网络条件下的系统级权衡。

**🔧 技术方法**

使用liboqs、OpenSSL 3.5.2与OQS provider在Intel i7和Raspberry Pi 4上实现原语基准；通过TLS1.3握手工具测量CPS；采用自定义脚本计算CER和TCP payload；结合实验数据进行对比分析。

**📊 数据集**

实验采用自建实验平台（CPU、ARM）与模拟网络流量，未使用公开数据集，重点在不同硬件与网络延迟/丢包情景下的握手数据。

**📈 对比分析**

通过原语吞吐、握手CPS、TCP payload量化比较，结果显示在高性能平台上PQC握手吞吐与经典相近，但在边缘/IoT上因Ciphertext扩展导致payload大幅增加、能耗提升；在引入人工延迟/丢包时握手吞吐急剧下降。

**⚠️ 局限性**

局限性包括：未在真实无线链路（衰落、拥塞）上完成端到端评估；压缩参数对解密失败率与可靠性的影响尚未充分验证；缺乏针对6G服务等级的标准化部署配置；未覆盖硬件加速实现与侧信道安全性。

---

## 11. Hybrid Multiport Receivers for Slow Fluid Antenna Multiple Access

**arXiv ID:** 2605.06958 | [PDF](https://arxiv.org/pdf/2605.06958v1)

**作者:** José P. González-Coma `[一作]`, F. Javier López-Martínez `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种流体天线混合多端口接收机架构（FAHM），在仅使用极少 RF 链（如 2 链）的情况下实现多端口选择与混合组合的解耦。

**💡 创新点**

创新点在于：①将端口选择与 RF 链解耦，采用低复杂度 GEPort（基于 rank‑1 结构的递推逆矩阵更新）实现端口选择；②引入有效端口指标 P_eff，用以自适应确定需要激活的端口数量；③证明传统 CUMA 接收机是 FAHM 的特殊实例，形成统一框架。

**🔧 技术方法**

主要技术包括 GEPort 端口选择、递推矩阵更新（O(N²) 复杂度）、有效端口指标（P_eff）、混合模拟/数字组合设计（2 RF 链实现）、以及基于 Rayleigh/Rice 多径模型的 Monte Carlo 仿真。

**📊 数据集**

采用仿真生成的 Rayleigh / Rice 随机多径信道数据集；未使用公开实验数据集。

**📈 对比分析**

通过与慢 FAMA、DC、CUMA 等基准方案对比，实验表明 FAHM‑GEPort 在平均谱效率和掉线概率上均优于其他方案，尤其在中低 K 值、散射路径数中等时表现突出；在高 SNR 及多用户场景中仍保持较好的性能。

**⚠️ 局限性**

局限性包括：GEPort 迭代仍存在较高计算开销，尤其在端口数大时；在高 Rice 因子（强直射波）或极端多用户干扰环境下性能下降；以及需要进一步验证硬件实现的实际收益与功耗。

---

## 12. STDA-Net: Spectrogram-Based Domain Adaptation for cross-dataset Sleep Stage Classification

**arXiv ID:** 2605.06736 | [PDF](https://arxiv.org/pdf/2605.06736v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 13. Towards multi-modal forgery representation learning for AI-generated video detection and localization

**arXiv ID:** 2605.07232 | [PDF](https://arxiv.org/pdf/2605.07232v1)

**作者:** Dat Le `[一作]`, Shu Hu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一个三分支多模态框架，联合视觉、音频与LMM语义分支，实现AI生成视频的检测与细粒度时序定位。

**💡 创新点**

核心创新在于将大型多模态模型的语义判别与专门的空间-时间视觉分支及多尺度局部伪造音频分支融合，既实现检测又可精确定位局部伪造。

**🔧 技术方法**

采用CLIP视觉编码+VQ‑VAE重建+IAFA注意力的视觉分支；Wav2Vec2.0+PartialSpoof+gMLP的音频分支；以及LLaVA式的LMM语义分支和动态多头注意力融合模块。

**📊 数据集**

在AV‑Deepfake1M++大规模语音视觉深伪数据集上训练并评估，零样本推理在FakeAVCeleb上验证跨域泛化。

**📈 对比分析**

与MM‑Det、UMMAFormer、BA‑TFD等SOTA基线对比，在AV‑Deepfake1M++上视频AUC达96.66%，段级AUC 98.23%，在FakeAVCeleb上AUC 83.03%，在时序定位上AP@0.5提升约34%，显著优于对手。

**⚠️ 局限性**

目前采用两阶段训练，视觉与音频分支先分别训练后融合，缺乏端到端联合优化与统一的音视频潜在空间，导致模型复杂度高且训练流程不够简洁。

---

## 14. How Well Do LLMs Perform on the Simplest Long-Chain Reasoning Tasks: An Empirical Study on the Equivalence Class Problem

**arXiv ID:** 2605.06882 | [PDF](https://arxiv.org/pdf/2605.06882v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 15. Sat3R: Satellite DSM Reconstruction via RPC-Aware Depth Fine-tuning

**arXiv ID:** 2605.07264 | [PDF](https://arxiv.org/pdf/2605.07264v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 16. A Reproducible Multi-Architecture Baseline for Token-Level Chinese Metaphor Identification under the MIPVU Framework

**arXiv ID:** 2605.07170 | [PDF](https://arxiv.org/pdf/2605.07170v1)

**作者:** Yufeng Wu `[一作]` (City University of Hong Kong), Yufeng Wu `[通讯]` (City University of Hong Kong)

**通讯引用:** 5223 | [OpenAlex ID](https://openalex.org/A5026608819)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了可复现的多架构基线，用于PSU CMC上基于MIPVU的token级隐喻识别，并首次在中文中使用MelBERT的基本意义融合。

**💡 创新点**

创新在于用MCD7构建中文基本意义资源，首次在中文隐喻识别中实现MelBERT的词义融合，并系统比较Encoder、MelBERT及LLM在同一数据集上的表现。

**🔧 技术方法**

采用RoBERTa‑wwm‑ext‑large微调、MelBERT（含MIP/ SPV通道）和Qwen3.5‑9B+QLoRA指令微调生成式方法，并实现多种任务格式。

**📊 数据集**

使用PSU Chinese Metaphor Corpus（PSU CMC）以及从《现代汉语词典》第七版提取的MCD7基础意义向量资源。

**📈 对比分析**

在固定5个seed下比较正类F1，MelBERT MIP‑only最高达0.7281±0.0050，MelBERT Full为0.7270±0.0069，RoBERTa为0.7142±0.0121，QLoRA生成式仅为0.6157±0.0113，显示Encoder显著优于LLM。

**⚠️ 局限性**

局限在于MelBERT仅使用第一义项忽略多义词；LLM任务格式选择有限，未能与Encoder竞争；评测仅覆盖PSU CMC，未验证跨语料通用性。

---

## 17. Retrieve, Integrate, and Synthesize: Spatial-Semantic Grounded Latent Visual Reasoning

**arXiv ID:** 2605.07106 | [PDF](https://arxiv.org/pdf/2605.07106v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 18. Star Elastic: Many-in-One Reasoning LLMs with Efficient Budget Control

**arXiv ID:** 2605.07182 | [PDF](https://arxiv.org/pdf/2605.07182v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 19. Why DDIM Hallucinates More than DDPM: A Theoretical Analysis of Reverse Dynamics

**arXiv ID:** 2605.06831 | [PDF](https://arxiv.org/pdf/2605.06831v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 20. From Map-and-Encap to BIER: Observations on Network Routing Scalability

**arXiv ID:** 2605.07071 | [PDF](https://arxiv.org/pdf/2605.07071v1)

**作者:** Tianyuan Yu `[一作]` (UCLA), Lixia Zhang `[通讯]` (UCLA)

**通讯引用:** 30051 | [OpenAlex ID](https://openalex.org/A5116294214)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了从早期统一 IP 地址到后来的 Identifier/Locator 分离（Map‑and‑Encap）等多种路由可扩展性方案的发展，并提出了四条关键观察，指出了多播与单播路由扩展性共通的根源。

**💡 创新点**

创新点在于：①将 Map‑and‑Encap 视为跨多播/单播的统一架构；②强调可扩展性应以可控的网络拓扑为基准而非外部应用动态；③揭示 BGP 缺乏类似单域出口路由器的拓扑抽象是全球规模部署的根本障碍；④通过 BIER 的理论分析验证了在单域内实现多播可扩展性的可行性。

**🔧 技术方法**

所用技术包括：Map‑and‑Encap 原理（RFC 1955）、LISP（Endpoint Identifiers/EID 与 Routing Locators/RLOC）、MPLS、BIER（Bit‑Index Explicit Replication）、BGP、iBGP、MVPN、DNS、DDT、ILNP、IPv6 等；同时结合了路由协议（PIM、MOSPF、DVMRP）和标签/位串编码的路由表结构（FIB、BIFT）。

**📊 数据集**

论文主要基于文献综述和理论分析，没有使用实验或测量数据集；对比主要来源于已有工作（LISP、MPLS、BIER 的部署报告与规范）。

**📈 对比分析**

比较方法：①从资源需求的增长变量入手，将单播 DFZ 前缀增长、状态化多播（S,G）树增长、BIER 的 BIFT 规模三者对比；②结合实际部署案例（如 MPLS 在大型 ISP 内部的成功、LISP 的有限推广、BIER 在实验域中的可行性），说明基于拓扑的设计能够保持有限且可控的状态；⑥性能上指出 BIER 的 BIFT 与网络拓扑匹配，理论上可扩展；MPLS 已经在实际网络中实现大规模部署；LISP 仍受跨域协调成本限制。

**⚠️ 局限性**

局限性：①全球范围缺乏出口路由器抽象，导致 BIER 与 LISP 等方案难以跨 AS 部署；②映射系统（如 LISP 的 DDT）缺乏统一治理模型，跨域协调成本高；③多播仍受应用层动态控制，状态增长不可预期；④本文为综述性工作，未提供实验验证或量化性能指标。

---

## 21. Closed-Form Linear-Probe Dataset Distillation for Pre-trained Vision Models

**arXiv ID:** 2605.07194 | [PDF](https://arxiv.org/pdf/2605.07194v1)

**作者:** Bincheng Peng `[一作]` (Hokkaido University), Miki Haseyama `[通讯]` (Hokkaido University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5040047120)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向冻结预训练编码器的闭式线性探针数据蒸馏方法CLP-DD，利用闭式岭回归和样本空间核化实现无内循环的蒸馏；

**💡 创新点**

创新点在于：①将冻结特征下的线性探针闭式求解与样本空间核等价化，消除轨迹展开和NTK近似；②采用温度缩放的类锚点交叉熵作为外部目标，使优化直接对真实样本的判别性能产生梯度；

**🔧 技术方法**

主要技术包括闭式岭回归、样本空间核化、温度缩放的类锚点交叉熵、对齐梯度的自动微分；

**📊 数据集**

使用ImageNet‑100、ImageNet‑1K、Spawrious、WaterBirds四个数据集进行实验；

**📈 对比分析**

与随机、中心点、邻居等真实图像基线以及LGM（含与不含DSA）比较，CLP-DD在IPC=1时相较于无DSA LGM提升约15‑20%，与含DSA LGM接近但显著降低14倍的计算时间和1/8的显存使用；

**⚠️ 局限性**

局限性包括：在极端背景偏移任务上仍低于完整数据集基线；对超参数（温度τ、岭正则λ）敏感；在更大IPC或更深网络下的可扩展性仍待验证。

---

## 22. Theoretical Limits of Language Model Alignment

**arXiv ID:** 2605.07105 | [PDF](https://arxiv.org/pdf/2605.07105v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 23. HARMONY: Bridging the Personalization-Generalization Gap by Mitigating Representation Skew in Heterogeneous Split Federated Learning

**arXiv ID:** 2605.07211 | [PDF](https://arxiv.org/pdf/2605.07211v1)

**作者:** Jiseok Youn `[一作]` (Seoul National University), Saewoong Bahk `[通讯]` (Seoul National University)

**通讯引用:** 3322 | [OpenAlex ID](https://openalex.org/A5040786910)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50`

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

## 24. FlashMol: High-Quality Molecule Generation in as Few as Four Steps

**arXiv ID:** 2605.07020 | [PDF](https://arxiv.org/pdf/2605.07020v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 25. Computing Short SAT Implicants via Ising/QUBO Encodings

**arXiv ID:** 2605.07017 | [PDF](https://arxiv.org/pdf/2605.07017v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 26. Estimating Correlation Clustering Cost in Node-Arrival Stream

**arXiv ID:** 2605.07091 | [PDF](https://arxiv.org/pdf/2605.07091v1)

**作者:** Kaiwen Liu `[一作]` (Indiana University), Qin Zhang `[通讯]` (Indiana University)

**通讯引用:** 7222 | [OpenAlex ID](https://openalex.org/A5100418221)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了在节点到达流模型下，用子线性空间估计相关聚类成本的算法 C^4Approx。

**💡 创新点**

创新点是结合参考集与高低度分解，在节点流中实现无缝估计并给出下界证明。

**🔧 技术方法**

采用随机排列、改造的 Pivot 算法、参考集查询、采样估计和流式递归搜索等技术。

**📊 数据集**

在维基百科、LiveJournal 以及 ImageNet‑21K 图数据集上实验验证。

**📈 对比分析**

与经典 Pivot 及稀疏‑密集分解算法对比，C^4Approx 在占用空间仅 2% 时即可与 Pivot 相当，且误差更低、波动更小。

**⚠️ 局限性**

局限在需多次扫描（k+4 次）且对 k 的取值敏感，且在极稀疏图中加性项影响显著。

---

## 27. Securing Computer-Use Agents: A Unified Architecture-Lifecycle Framework for Deployment-Grounded Reliability

**arXiv ID:** 2605.07110 | [PDF](https://arxiv.org/pdf/2605.07110v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 28. Accelerated Relax-and-Round for Concave Coverage Problems

**arXiv ID:** 2605.06900 | [PDF](https://arxiv.org/pdf/2605.06900v1)

**作者:** Matthew Fahrbach `[一作]` (Google Research), Morteza Zadimoghaddam `[通讯]` (Google Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种加速的 relax‑and‑round 算法，用于求解凸合覆盖（concave coverage）问题，能够在多项式时间内获得接近最优的近似解。

**💡 创新点**

创新点包括：① 用 Nesterov 平滑把离散 LP 约束转为可微光滑目标，从而使用加速投影梯度法大幅降低求解 LP 的时间；② 结合 Carathéodory 分解与随机交换放缩（swap rounding）实现快速高质量的取整；③ 证明了多种奖励函数（如 log、分段线性、isoelastic）的 Poisson concavity ratio，并给出相应的紧逼近比例。

**🔧 技术方法**

核心技术包括 Nesterov 平滑、加速投影梯度（FISTA）求解光滑覆盖目标、Carathéodory 组合分解、swap rounding、以及对多项式时间约束的投影算法。

**📊 数据集**

实验数据集涵盖：① 通过理论构造的困难合成图；② SNAP 社交网络图（如包含 4039 个顶点、180507 条边的图，和 317080 个顶点、2416812 条边的图）。

**📈 对比分析**

与传统的 LP + pipage rounding 方法（GLOP、HiGHS、SCIP）相比，本算法在 600 秒 LP 求解限制下，能够在 10 倍以上更短的时间完成整个 relax‑and‑round 流程，并得到更高的目标函数值；尤其在大规模实例中，pipage rounding 甚至超时，而本方法仍保持可行且高效。

**⚠️ 局限性**

主要限制：算法假设只有统一基元（cardinality 约束）；对非基元约束的泛化尚未讨论；需要先验估计最优目标值以设置平滑参数，且在极端稠密图或非常大的 n、m 时仍可能面临内存或计算瓶颈。

---

## 29. Gradient-Based LoRA Rank Allocation Under GRPO: An Empirical Study

**arXiv ID:** 2605.07366 | [PDF](https://arxiv.org/pdf/2605.07366v1)

**作者:** Yash Ganpat Sawant `[一作]` `[通讯]` (Independent Researcher), Yash Ganpat Sawant (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在RL对齐方法GRPO下，梯度驱动的LoRA层级rank分配是否有效，使用Qwen2.5-1.5B与GSM8K数据对不同rank分配策略进行训练和评估。

**💡 创新点**

发现RL梯度景观比SFT更平坦，梯度放大效应导致非均匀分配会进一步拉大梯度重要性分布，从而削弱模型性能，证明统一rank分配在RL中更优。

**🔧 技术方法**

采用LoRA适配器、奖励敏感性梯度剖面、GRPO训练、梯度重要性评分、比例/随机/减少rank分配策略，并对梯度分布与性能进行系统分析。

**📊 数据集**

使用Qwen/Qwen2.5-1.5B-Instruct模型和GSM8K算术推理数据集（结构化XML输出）。

**📈 对比分析**

对比方法：在相同参数预算下，测量GSM8K测试集（200样本）的准确率；结果显示统一分配74.5%、比例分配70.0%、随机分配67.5%、减少分配65.0%；训练奖励曲线相近但评估性能差异明显。

**⚠️ 局限性**

局限性：仅在单一模型、单一数据集和单一RL算法（GRPO）下实验；测试样本有限且仅使用单一随机种子；未实现动态rank调整；关于低梯度层承担结构功能的假设未得到验证。

---

## 30. Generalising Travel Time Prediction To Varying Route Choices In Urban Networks

**arXiv ID:** 2605.06918 | [PDF](https://arxiv.org/pdf/2605.06918v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 31. PAMPOS: Causal Transformer-based Trajectory Prediction for Attack-Agnostic Misbehavior Detection in V2X Networks

**arXiv ID:** 2605.06833 | [PDF](https://arxiv.org/pdf/2605.06833v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 32. When the Ruler is Broken: Parsing-Induced Suppression in LLM-Based Security Log Evaluation

**arXiv ID:** 2605.07293 | [PDF](https://arxiv.org/pdf/2605.07293v1)

**作者:** Chaitanya Vilas Garware `[一作]` (University of Alabama at Birmingham), Sharif Noor Zisad `[通讯]` (University of Alabama at Birmingham)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5095853206)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对LLM SOC日志分类评估流程进行方法学审计，发现正则解析导致的“解析抑制”误差，并提出SOC‑Bench v0评测框架。

**💡 创新点**

量化并首次公开“解析抑制”问题，展示严格与模糊解析对准确率的巨大影响，并提出标准化的模糊提取规范与基准。

**🔧 技术方法**

使用LoRA微调的TinyLlama‑1.1B模型、正则与模糊字段提取器、宏平均准确率评估以及对Claude Sonnet零样本基准进行对照。

**📊 数据集**

采用OpenSOC‑AI的50条保留评测集（13类SOC标签），并在SOC‑Bench框架下定义13类MITRE对齐标签。

**📈 对比分析**

在相同模型输出下，用严格解析得到0%威胁准确率，模糊解析恢复76%，与Claude Sonnet零样本的88%相比，差距仅12个百分点。

**⚠️ 局限性**

评测样本数不足导致单类置信区间宽，模糊解析的假阳性未系统测评，且仅针对TinyLlama，未验证其他模型族。

---

## 33. High-Fidelity Surface Splatting-Based 3D Reconstruction from Multi-View Images

**arXiv ID:** 2605.07254 | [PDF](https://arxiv.org/pdf/2605.07254v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 34. BioProVLA-Agent: An Affordable, Protocol-Driven, Vision-Enhanced VLA-Enabled Embodied Multi-Agent System with Closed-Loop-Capable Reasoning for Biological Laboratory Manipulation

**arXiv ID:** 2605.07306 | [PDF](https://arxiv.org/pdf/2605.07306v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 35. Dr-BA: Separable Optimization for Direct Radar Bundle Adjustment & Localization

**arXiv ID:** 2605.07041 | [PDF](https://arxiv.org/pdf/2605.07041v1)

**作者:** Daniil Lisus `[一作]` (University of Toronto), Timothy D. Barfoot `[通讯]` (University of Toronto)

**通讯引用:** 8099 | [OpenAlex ID](https://openalex.org/A5004788089)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一种基于2D旋转雷达的直接束调整框架（Dr-BA），用于构建稠密且局部与全局一致的雷达地图，并在此基础上实现直接雷达定位（DRL）。

**💡 创新点**

创新点包括：①首次提出直接雷达束调整方法；②将问题表述为可分离优化，利用VarPro将姿态与地图独立优化；③结合自适应高斯模糊、累计图像处理和局部地图提升稀疏环境的可用信息；④构建直接雷达定位（DRL）框架，实现对已建地图的高精度定位。

**🔧 技术方法**

采用可分离优化（VarPro）、稠密雷达强度图像处理、基于SE(2)的姿态优化、关键帧策略、DRO去畸变、Gaussian模糊、累计图像等技术；通过高斯-牛顿迭代求解。

**📊 数据集**

使用Boreas Road Trip数据集，包含Navtech RAS6雷达、Silicon Sensing DMU41 IMU以及RTK-GNSS/INS地面真值；共20条轨迹，5种不同环境，累计约206.4公里。

**📈 对比分析**

与DRO（单一里程计）、TBV-SLAM（闭环SLAM）以及Dr-PoGO（姿态图优化）等基线在ATE、EPE、self-consistency和定位RMSE等指标上进行对比。Dr-BA在大多数路段获得最低ATE和EPE，显著提升局部一致性；DRL-Dr-BA在定位上比RTR更优，且在多数测试场景下无失效。

**⚠️ 局限性**

局限性包括：在稀疏或高度遮蔽的环境中性能下降；未显式建模雷达强度对视角的依赖；需要良好的初始轨迹以保证收敛；单帧雷达信息不足时仍可能失败。

---

## 36. EDA-Schema-V2: A Multimodal Schema, Open Datasets, and Benchmarks for Machine Learning in Digital Physical Design

**arXiv ID:** 2605.06952 | [PDF](https://arxiv.org/pdf/2605.06952v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 37. MultiSoc-4D: A Benchmark for Diagnosing Instruction-Induced Label Collapse in Closed-Set LLM Annotation of Bengali Social Media

**arXiv ID:** 2605.06940 | [PDF](https://arxiv.org/pdf/2605.06940v1)

**作者:** Souvik Pramanik `[一作]` (North South University), Md. Shahriar Hussain `[通讯]` (North South University)

**通讯引用:** 86 | [OpenAlex ID](https://openalex.org/A5108260951)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一个面向孟加拉语社交媒体的多维度标注数据集 MultiSoc-4D，并系统研究了在闭集指令下 LLM 的标签崩塌现象。

**💡 创新点**

发现并量化了“指令诱导标签崩塌”以及“同意幻觉”，并提出 Bias Ratio 等度量来描述注释偏差。

**🔧 技术方法**

采用 ChatGPT、Gemini、Claude、Grok 等 LLM 进行注释，并通过 Fleiss κ、Bias Ratio、FN 率等指标进行行为分析。

**📊 数据集**

使用来自 Facebook、Twitter、YouTube、TikTok、Likee、Instagram 的 58K+ 评论文本构成的孟加拉语数据集。

**📈 对比分析**

与 40+ 传统 ML、Transformer 及指令调优 LLM 的宏观 F1 进行对比，结果显示 LLM 在多数任务上表现良好，但在仇恨与讽刺等少数类的 F1 远低于人类标注。

**⚠️ 局限性**

局限在于闭集标注框架导致标签空间压缩、模型安全对抗使其回避仇恨/讽刺标签，且未实现开放集标注以缓解标签崩塌。

---

## 38. UniV2D: Bridging Visual Restoration and Semantic Perception for Underwater Salient Object Detection

**arXiv ID:** 2605.07146 | [PDF](https://arxiv.org/pdf/2605.07146v1)

**作者:** Laibin Chang `[一作]` (Wuhan University), Bo Du `[通讯]` (Wuhan University)

**通讯引用:** 31149 | [OpenAlex ID](https://openalex.org/A5060042752)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种统一的水下视觉与检测网络UniV2D，能够在单一模型中同时完成水下图像增强与显著目标检测；

**💡 创新点**

创新点在于：①利用语义驱动的双分支初始化（SCSM+MACR）实现视觉恢复与显著性先验的互助；②设计跨层特征调制（CLFM）实现任务级的双向细粒度对齐；③在联合损失中同时优化恢复质量与检测精度；

**🔧 技术方法**

使用的关键技术包括：U‑Net型骨干网络、Self‑Calibrated Saliency Masking、Mask‑Aware Content Restoration、Cross‑Level Feature Modulation、BCE+IoU、SSIM/PSNR、VGG感知损失以及联合多任务训练；

**📊 数据集**

实验数据集：UIE任务使用UIEB、LSUI、UWscene；USOD任务使用USOD10K、USOD、SUIM；

**📈 对比分析**

与多种单任务和多任务SOTA方法（如WaterDiffusion、U‑Shape、TCTL‑Net、U2‑Net等）比较，UniV2D在SSIM、PSNR、PCQI、UCIQE、S‑α、F‑β^w、E‑ϕ^m等指标均获得第一或第二名，推理速度仅0.016s/图，参数13.29M，FLOPs31.23G，显著优于WaterDiffusion等同任务模型；

**⚠️ 局限性**

局限性：仍对极端低照度或高浑浊场景的泛化能力有限；依赖标注的显著性与增强配对数据，数据集规模相对有限；在极低算力设备上仍可能面临内存或计算瓶颈。

---

## 39. GAD in the Wild: Benchmarking Graph Anomaly Detection under Realistic Deployment Challenges

**arXiv ID:** 2605.07133 | [PDF](https://arxiv.org/pdf/2605.07133v1)

**作者:** Jingjing Zhou `[一作]` (Zhejiang Gongshang University), Ivan Lee `[通讯]` (Adelaide University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个多维度的图异常检测基准，系统评估模型在百万级规模、极端类不平衡、属性缺失等实际部署场景下的表现。

**💡 创新点**

创新点在于同时关注规模、异常稀缺和属性不完整三大现实缺口，并对九种代表性模型进行统一实验，揭示了现有方法在可扩展性、极端稀缺性和缺失处理上的缺陷。

**🔧 技术方法**

采用了规模扩展生成、异常比例调节、属性缺失模拟及多种填充策略的实验框架，并对比了基于自编码器、对比学习、邻域重建等九种GAD方法。

**📊 数据集**

使用了五个公开图数据集（Twitter、PubMed、Credit、DGraph‑Fin、T‑Social）以及人工扩展与改写版本。

**📈 对比分析**

通过统一的实验协议对检测效果（AUC‑ROC、AUC‑PR、召回率）和部署效率（训练时间、峰值GPU内存）进行评估，结果显示大多数GNN方法在百万节点图上会OOM，极端稀缺情况下召回率接近零，属性缺失对重建模型影响显著。

**⚠️ 局限性**

局限性包括：只对三维因素独立实验，未研究它们交互效应；实验在单机实验室硬件上进行，缺乏大规模分布式环境验证；只评估了当前九种方法，未考虑更新或混合模型。

---

## 40. CLIPer: Tailoring Diverse User Preference via Classifier-Guided Inference-Time Personalization

**arXiv ID:** 2605.07162 | [PDF](https://arxiv.org/pdf/2605.07162v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 41. From Standard English to Singlish: A Retrieval-Augmented Approach for Code-Switched Creole Generation in Large Language Models

**arXiv ID:** 2605.07132 | [PDF](https://arxiv.org/pdf/2605.07132v1)

**作者:** Foong Ming Lai `[一作]` (National University of Singapore), Yi-Chieh Lee `[通讯]` (National University of Singapore)

**通讯引用:** 2073 | [OpenAlex ID](https://openalex.org/A5054435118)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

该研究提出了一种检索增强生成（RAG）框架，通过外部化Singlish词典实现无参数微调的代码切换生成，并与零样本提示进行对比。

**💡 创新点**

创新点在于将代码切换视为可检索的词汇控制，将词典外部化以实现可审计的稀疏词汇替换，而非依赖模型参数调整。

**🔧 技术方法**

使用了检索增强生成、语义搜索、依赖句法分析、OpenAI句子嵌入、HNSW近似最近邻检索以及GPT‑5两轮生成技术。

**📊 数据集**

采用了来自Singlish Dictionary的198条词典条目作为外部词典，并使用164名新加坡受试者的交互数据进行人工评估。

**📈 对比分析**

与零样本提示对比，RAG在编辑距离极小（中位数1）且语义相似度高（0.978）时，用户感知的自然度和适当性与零样本提示相当。

**⚠️ 局限性**

局限性在于仅覆盖词汇层面的代码切换，未考虑语法和语用特征；并且基于英语矩阵语言，可能难以推广至非英语矩阵体系。

---

## 42. XiYOLO: Energy-Aware Object Detection via Iterative Architecture Search and Scaling

**arXiv ID:** 2605.06927 | [PDF](https://arxiv.org/pdf/2605.06927v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 43. Knowledge Transfer Scaling Laws for 3D Medical Imaging

**arXiv ID:** 2605.06859 | [PDF](https://arxiv.org/pdf/2605.06859v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 44. CASCADE: Case-Based Continual Adaptation for Large Language Models During Deployment

**arXiv ID:** 2605.06702 | [PDF](https://arxiv.org/pdf/2605.06702v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 45. The Text Uncanny Valley: Non-Monotonic Performance Degradation in LLM Information Retrieval

**arXiv ID:** 2605.07186 | [PDF](https://arxiv.org/pdf/2605.07186v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 46. A Theory of Online Learning with Autoregressive Chain-of-Thought Reasoning

**arXiv ID:** 2605.06819 | [PDF](https://arxiv.org/pdf/2605.06819v1)

**作者:** Ilan Doron-Arad `[一作]` (MIT), Elchanan Mossel `[通讯]` (MIT)

**通讯引用:** 9969 | [OpenAlex ID](https://openalex.org/A5013467728)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

本文提出并分析了自回归链式思考（Chain‑of‑Thought, CoT）在在线学习中的理论框架，研究了在仅观察最终输出（End‑to‑End）与完整生成轨迹（CoT）两种监督模式下，学习隐含的下一步生成器的误差界和对生成长度的依赖性；同时对自回归线性阈值类给出了最优误差与样本复杂度上界与下界，并揭示了在无监督与CoT监督下的计算复杂度差异。

**💡 创新点**

创新点主要包括：①构造了完整的在线误差阶层分类，证明在CoT监督下误差不随生成长度增长；②给出了在End‑to‑End监督下，误差随生成长度的所有可实现增长率（常数到对数级），并证明对数上界不可改进；③针对自回归线性阈值类证明最优误差为Θ(d²)，并提供了与统计学习下的下界相匹配的新证明；④展示了在无监督与CoT监督下的计算复杂度分离；⑤在随机生成器模型下给出了误差可指数级放大的例子。

**🔧 技术方法**

主要技术手段包括：Littlestone维度与其树形表征；SOA（Standard Optimal Algorithm）与版本空间更新；树覆盖与Sauer‑Shelah‑Perles引理的树版；子对数增长率构造与诱导子类；概率方法与大数定律构造硬类；latch 机制将线性阈值类的输出“拖延”至整个生成过程；以及在随机模型中利用马尔可夫链、KL 散度与Bretagnolle–Huber 不等式。

**📊 数据集**

本工作为纯理论研究，无使用具体数据集；所有结果均在符号层面证明。

**📈 对比分析**

与现有工作对比：对比统计PAC学习中已知的对数上界，本文证明在线误差界在CoT监督下不随生成长度增加，且在End‑to‑End监督下实现了从常数到对数的完整误差阶层；在线学习下的计算复杂度与批处理学习下的可行性之间也给出了新的关系。

**⚠️ 局限性**

局限性：所有误差与维度结果均假设基类具有有限Littlestone维度；对非Littlestone类仅给出极端例子；随机生成器模型中的指数级误差是特殊构造，未给出一般情况；在多分类、部分反馈等更广泛场景下的结果尚未完整探讨。

---

## 47. Bi3: A Biplatform, Bicultural, Biperson Dataset for Social Robot Navigation

**arXiv ID:** 2605.06863 | [PDF](https://arxiv.org/pdf/2605.06863v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 48. Instruction Tuning Changes How Upstream State Conditions Late Readout: A Cross-Patching Diagnostic

**arXiv ID:** 2605.07284 | [PDF](https://arxiv.org/pdf/2605.07284v1)

**作者:** Yifan Zhou `[一作]` (University of California, Los Angeles), Yifan Zhou `[通讯]` (University of California, Los Angeles)

**通讯引用:** 1819 | [OpenAlex ID](https://openalex.org/A5101700900)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了指令微调如何通过早期计算与后期堆叠协同影响下一词预测，使用第一分歧交叉补丁技术对 PT 与 IT 检查点进行对比。

**💡 创新点**

创新点在于引入了“first-divergence cross-patching”四格因果分解，明确区分了后期堆叠的直接效应与对早期状态的依赖，并展示了不同训练方案（指令微调 vs. 领域特定微调）对这一机制的差异。

**🔧 技术方法**

使用了激活补丁、跨模型特征跨编码器（crosscoders）、PCA 边界状态投影、受限续写等多种技术来探测层级耦合与稀疏特征。

**📊 数据集**

数据集主要为多模态提示集合（Prompt-bootstrap），涵盖五个 4B‑32B 稠密模型的发布检查点和同基底的控制实验；此外还使用了 OpenMath2、Tulu‑3、OLMo‑2 等领域特定数据。

**📈 对比分析**

对比方法是计算 PT/IT 在第一分歧位置的 logit 差异，并在四格交叉补丁下分解为 PT‑上游、IT‑上游以及交互效应；结果显示交互效应平均约 +1.68 logits，正向一致且在各模型族中稳健，表明后期堆叠与早期状态存在显著耦合。

**⚠️ 局限性**

局限性包括：仅关注单一 token（第一分歧）且局部 token‑margin，未能泛化到整体生成行为；实验仅覆盖稠密模型，缺乏对 MoE 或更大规模模型的验证；跨模型补丁可能受模型对齐和对话模板影响。

---

## 49. Sub-Gaussian Concentration and Entropic Normality of the Maximum Likelihood Estimator

**arXiv ID:** 2605.07107 | [PDF](https://arxiv.org/pdf/2605.07107v1)

**作者:** Leighton P. Barnes `[一作]` (Carnegie Mellon University), Alex Dytso `[通讯]` (Qualcomm Flarion Technology, Inc.)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文在最大似然估计（MLE）的经典中心极限定理基础上，提出了更强的渐近正态性结果，证明了归一化MLE在更高阶层面（如子高斯尾、矩收敛、相对熵收敛）下的收敛性；

**💡 创新点**

创新点在于：①引入对分数函数及其导数的子高斯、次指数约束，从而实现子高斯尾界和矩收敛；②通过平滑技术和信息熵控制，证明了MLE的相对熵（KL散度）趋于零，进一步在额外的Fisher信息或导数有界条件下去掉平滑实现无平滑的相对熵收敛；

**🔧 技术方法**

主要技术包括：高阶矩估计与子高斯/次指数尾控制、统一大数定律、切比雪夫与马尔可夫不等式、信息熵不等式、傅里叶变换与特征函数解析、熵增不等式、Pinsker与Talagrand不等式以及对MLE密度的Fisher信息或导数有界性分析；

**📊 数据集**

本文未使用具体实验数据集，而是在理论框架下给出若干典型分布（如指数族、Pearson IV 位置族、Logistic、Cauchy）作为满足假设的例子；

**📈 对比分析**

方法的性能未通过实验评估，而是通过数学证明表明在满足假设时，MLE的相对熵收敛速度可比传统CLT更强，且在平滑后可直接收敛到标准正态分布；

**⚠️ 局限性**

局限性包括：①需要分数函数及其导数满足子高斯/次指数尾约束，某些模型难以验证；②去除平滑时需额外Fisher信息或导数有界假设，限制了适用范围；③仅讨论单参数情况，尚未扩展到多维MLE；

---

## 50. Social Understanding, Placeness, and Identity Alignment: A Design Framework for Friendship-Supportive Youth Social Media

**arXiv ID:** 2605.07025 | [PDF](https://arxiv.org/pdf/2605.07025v1)

**作者:** JaeWon Kim `[一作]` (University of Washington), Alexis Hiniker `[通讯]` (University of Washington)

**通讯引用:** 3440 | [OpenAlex ID](https://openalex.org/A5074077266)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对5项涉及331名13–25岁青少年的访谈、共创、问卷、日记研究和一次跨国现场部署的结果进行跨研究综合，提出了一个包含三大支柱（社会理解、空间感知、身份对齐）和九个设计空间的青少年友谊支持社交媒体设计框架。

**💡 创新点**

将社会渗透、社会透明度、空间缺失、身份发展等社会心理学与人机交互理论转化为可操作的设计范畴，提供统一的术语与评估维度，揭示当前平台设计中被忽视的关键空间，并指出未来研究的空白。

**🔧 技术方法**

主要采用质性研究方法：访谈、共创工作坊、调查问卷、日记记录和现场部署；对209个设计相关数据点进行主题编码、迭代归类，形成框架；在场景部署中利用行为日志与访谈进行定性比较。

**📊 数据集**

研究使用的原始数据来自331名青少年（13–25岁）的访谈稿、共创记录、问卷结果、日记条目以及一次为期4周的跨国场景部署日志；共提炼209条设计相关数据点。

**📈 对比分析**

在场景部署（Study E）中将包含关系支撑功能的定制平台与传统基准平台进行对比，使用行为日志和访谈来评估功能的影响；论文未给出具体数值性能指标，而是通过用户行为与感知差异阐释设计效益。

**⚠️ 局限性**

框架的实证基础主要来自美国和韩国的青少年，样本有限；缺乏对更年轻儿童、不同文化背景或以即时通讯为主的系统的验证；部分维度（如身份货币、共享存在）仍处于理论化阶段，缺乏深入的量化评估；未来工作需在更广泛情境下进一步测试与细化。

---

## 51. Regulating Branch Parallelism in LLM Serving

**arXiv ID:** 2605.06914 | [PDF](https://arxiv.org/pdf/2605.06914v1)

**作者:** Swapnil Gandhi `[一作]` (Stanford University), Christos Kozyrakis `[通讯]` (Stanford University)

**通讯引用:** 21206 | [OpenAlex ID](https://openalex.org/A5042148531)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型（LLM）内部并行解码的调度问题，提出按步宽度动态控制的 TAPER 控制器

**💡 创新点**

创新点在于将分支级调度与内存管理解耦，利用可逆的每步宽度调节和 branch‑externality 预测，解决了 eager 与固定宽度策略导致的 throughput trap

**🔧 技术方法**

使用了分支外部性预测模型、基于剩余 SLO slack 的预算约束、贪心规划算法，以及 kv‑cache 结构的 branch‑level 调度

**📊 数据集**

在 Qwen3‑32B 上使用混合工作负载（ShareGPT、Multiverse 等）以及 Azure 真实请求轨迹进行评估

**📈 对比分析**

与 IRP‑Off、IRP‑C2、IRP‑C5、IRP‑Eager 对比，TAPER 在不同负载下实现 1.77×（对 IRP‑Off）和 1.48×（对 IRP‑Eager）的 goodput，SLO 达成率始终 >95%，而其他方案在高负载下 SLO 低于 50% 或吞吐受限

**⚠️ 局限性**

限制包括假设分支完全独立、线性延迟预测在极大批量下可能失准、仅在单节点环境下验证，且未考虑跨节点调度和多租户场景

---

## 52. Can LLMs Take Retrieved Information with a Grain of Salt?

**arXiv ID:** 2605.06919 | [PDF](https://arxiv.org/pdf/2605.06919v1)

**作者:** Behzad Shayegh `[一作]` (RBC Borealis), Leo Feng `[通讯]` (RBC Borealis)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出检索增强问答中对检索置信度的“上下文置信度服从度”评估框架，并设计基于先前回答提醒、置信度重校准和上下文简化的交互策略以提升LLM的适应性。

**💡 创新点**

创新点在于①从概率边缘化出发定义理想输出分布并量化服从误差；②提出三步交互（prior reminder、certainty recalibration、context simplification）在不修改模型权重的前提下显著提升服从度；③一次性重校准映射可在多任务间迁移。

**🔧 技术方法**

使用总变差距离、概率边缘化、对数线性组合构造理想分布、重校准映射、上下文简化、先前回答提醒等技术，并在八种开放权重LLM上实施。

**📊 数据集**

以ClashEval数据集为测试基准，包含多种QA类别的正负检索上下文。

**📈 对比分析**

在八种不同规模、架构（LLaMA、Qwen、Gemma）的LLM上进行基线对比；交互策略将上下文置信度服从误差平均从0.52降至0.39，提升约25%，在大模型上显著提升。

**⚠️ 局限性**

局限性包括：仅适用于已提供置信度信号的场景；重校准映射需一次性实验且可能随领域变化；对极长或极复杂上下文仍存在误判；未考虑训练阶段的自我不确定性处理。

---

## 53. When Are Experts Misrouted? Counterfactual Routing Analysis in Mixture-of-Experts Language Models

**arXiv ID:** 2605.07260 | [PDF](https://arxiv.org/pdf/2605.07260v1)

**作者:** Youngsik Yoon `[一作]` (POSTECH), Jungseul Ok `[通讯]` (POSTECH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对已训练的稀疏 MoE 语言模型的路由决策进行因果对比评估，并提出仅更新路由器的最小化方法来改进下游推理性能

**💡 创新点**

揭示标准 top‑k 训练的“对抗性盲点”，即模型仅对已执行路由产生梯度，而对同等计算量的未执行路由缺乏 token‑级信号；同时证明只更新路由器即可提升推理成功率

**🔧 技术方法**

对比标准路由与采样的等计算路由；使用 Gumbel‑top‑k 采样、门控权重重计算；提出 Expert Preference Optimization (EPO) 的路由器更新策略；使用前向干预测量 next‑token 交叉熵差异

**📊 数据集**

Qwen3‑30B‑A3B、GPT‑OSS‑20B、DeepSeek‑V2‑Lite、OLMoE‑1B‑7B 四大开源 MoE 模型；验证推理路径（MATH Level‑5、AIME、HMMT、GPQA‑Diamond）

**📈 对比分析**

通过比较标准路由与等计算路由在不同 token 难度（Confident/Ambiguous/Fragile）下的 next‑token 概率、Top‑K 归属率以及 best‑route 与标准路由的概率差距；结果显示：在 Confident tokens 上对齐度高，Fragile tokens 上差距显著；在 AIME/HMMT 上仅更新路由器即可提升 pass@K（提升幅度小但显著），表明路由失配可通过路由器更新可补偿

**⚠️ 局限性**

仅评估单层路由，未考虑跨层协同效应；仅在已通过验证的推理路径上评估，可能高估改进空间；对抗性盲点修复需在预训练阶段引入 token‑级对抗信号，尚未实现大规模可行方案

---

## 54. Modern column generation for estimating single- and multi-purchase ranked list choice models

**arXiv ID:** 2605.06948 | [PDF](https://arxiv.org/pdf/2605.06948v1)

**作者:** Luciano Costa `[一作]` (Federal University of Paraíba), Jean-François Cordeau `[通讯]` (HEC Montréal)

**通讯引用:** 17241 | [OpenAlex ID](https://openalex.org/A5016828932)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于动态规划的列生成框架，用于高效估计单购与多购的非参数排序列表离散选择模型；

**💡 创新点**

创新点在于设计了首个动态规划算法求解通用线性排序问题的定价子问题，并结合加速技巧显著降低计算复杂度；

**🔧 技术方法**

技术包括列生成、动态规划、主支配规则、完成界限、不可达商品过滤、启发式定价以及EM与L1误差估计方法；

**📊 数据集**

实验使用合成数据（包含随机、结构化、多购、考虑集限制等四类）与真实/半合成数据（酒店预订、寿司偏好）；

**📈 对比分析**

与基于MILP的定价子问题以及已有多购模型比较，动态规划实现速度提升40–70倍、误差更低、估计得到的组合优化收益更高；

**⚠️ 局限性**

局限在于仍基于传统效用最大化框架，无法捕捉系统性非理性行为，并且L1误差在样本稀疏时表现差。

---

## 55. PersonaGest: Personalized Co-Speech Gesture Generation with Semantic-Guided Hierarchical Motion Representation

**arXiv ID:** 2605.07252 | [PDF](https://arxiv.org/pdf/2605.07252v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 56. FAME: Forecasting Academic Impact via Continuous-Time Manifold Evolution

**arXiv ID:** 2605.07208 | [PDF](https://arxiv.org/pdf/2605.07208v1)

**作者:** Jianrong Ding `[一作]` (Chinese University of Hong Kong), Qiang Xu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 14741 | [OpenAlex ID](https://openalex.org/A5088556682)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文针对使用大型语言模型（LLM）进行科研成果影响力评估的局限性，提出一种基于连续时间流形演化的时空建模框架 FAME（Forecasting Academic Impact via Continuous-Time Manifold Evolution）。通过构建高质量的灵感图（inspiration graph）、将论文映射到连续流形并学习动态主题脊线（topic spines），利用几何约束将高影响力论文与其所属领域的前进动量对齐，从而实现对未来论文影响力的预测。

**💡 创新点**

创新点包括：
1) 将科研影响力预测转化为可验证的代理任务，即通过历史论文的实际影响力来评估预测模型。
2) 通过 retrieve‑and‑verify 管道利用 LLM 只做知识流验证，构造高质量、去噪的灵感图。
3) 引入连续时间流形学习与动态主题脊线，实现论文在时空连续空间中的几何对齐，从而捕捉领域演化的动量。
4) 将流形得分嵌入传统 LLM 评估器，显著提升其预测性能。

**🔧 技术方法**

技术手段包括：
- LLM 进行检索与验证以构造灵感图；
- 文本嵌入与 K‑means 聚类确定基底主题；
- 双分支神经网络（文本分支与图分支）联合学习连续流形表示；
- 通过几何约束（位置对齐、动量对齐）训练流形；
- Spearman 相关系数作为评价指标进行滑动窗口验证。

**📊 数据集**

数据集：3,200 篇来自 arXiv 的论文，涵盖三个快速演进的子领域。每篇论文的影响力通过四个指标（GitHub stars、citation counts、influential citations、Altmetric scores）归一化后汇总得到。

**📈 对比分析**

与现有 LLM 评估器（包括通用 LLM 和领域专用模型 SciJudge）进行对比。FAME 在 Spearman 相关系数上超过 0.5，显著高于所有 LLM 基线；在多维影响力预测任务中保持稳定且显著的优越性。将 FAME 的流形分数融入 LLM 评估后，其预测准确性进一步提升。

**⚠️ 局限性**

局限性：
- 依赖于 LLM 的检索与验证过程，可能引入偏差或误检；
- 构造灵感图时仍需手工设定阈值与约束，限制了自动化程度；
- 仅在 arXiv 论文上验证，未在其他学科或大规模真实引用网络中检验；
- 流形模型对参数设置和训练过程敏感，需要进一步优化稳定性与可解释性。

---

## 57. UniD-Shift: Towards Unified Semantic Segmentation via Interpretable Share-Private Multimodal Decomposition

**arXiv ID:** 2605.07356 | [PDF](https://arxiv.org/pdf/2605.07356v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 58. Cryptographic and Information-theoretic Security Capacities for General Arbitrarily Varying Wiretap Channels

**arXiv ID:** 2605.06751 | [PDF](https://arxiv.org/pdf/2605.06751v1)

**作者:** Holger Boche `[一作]` (Technical University Of Munich), Marc Geitz `[通讯]` (Deutsche Telekom Ag)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文研究了通用任意变化窃听信道（GAVWC）和常规任意变化窃听信道（AVWC）的强保密容量与语义保密容量，证明在子双指数级窜扰者策略下两者相等，给出了语义保密容量与强保密容量之间的上界，并构造了反例表明在某些系统中语义保密容量可能为零，而强保密容量正好。

**💡 创新点**

创新点包括：①首次将信息理论中的强保密与密码学中的语义保密（及其等价的不可区分性）在任意变化信道上进行统一分析；②给出在子双指数级窜扰者策略下两种容量相等的充分条件；③构造了一个系统的反例，证明语义保密容量可能严格小于强保密容量；④给出了两者容量差距的精确上界，并指出该界在某些实例中可达最优。

**🔧 技术方法**

采用的技术主要是随机编码与信息谱分析、对称化（symmetrizability）理论、信息半径（information radius）与Pinsker不等式、随机编码码本提炼、以及从可区分性攻击到语义保密的转换证明；同时使用了典型集合论证、Chernoff界、以及与可逆/不可逆的通道状态集合大小的对数估计。

**📊 数据集**

无具体数据集，所有结果均为理论推导与信息理论极限分析。

**📈 对比分析**

比较方法：先给出强保密容量与平均错误率/最大错误率的等价性，再将其与语义保密/不可区分性等价性结合，利用随机编码与典型集合技术证明容量相等或给出差距；在构造反例时直接计算相应容量；在给出上界时利用通道集合大小的对数与双指数增长的关系进行上界估计。性能方面，若窜扰者策略数目为子双指数级，则两种容量完全相同；若为双指数级或更大，则语义保密容量可小于强保密容量，且其差距受通道集合大小对数控制。

**⚠️ 局限性**

局限性：1）结果仅在窜扰者策略数目为子双指数级时成立，双指数级及以上时需进一步分析；2）对称化条件下的可达性尚未完全揭示，尤其是GAVWC的正性条件；3）论文主要针对经典信道，量子版本的对应结果尚待扩展；4）在实际通信系统中，假设窜扰者和窃听者完全了解编码方案与通道模型可能不现实；5）对于更强攻击模型（如主动攻击、可逆/不可逆混合攻击）仍未给出完整理论。

---

## 59. Repeated Deceptive Path Planning against Learnable Observer

**arXiv ID:** 2605.07174 | [PDF](https://arxiv.org/pdf/2605.07174v1)

**作者:** Shiyue Cao `[一作]` (University of Chinese Academy of Sciences), Kaiqi Huang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 14985 | [OpenAlex ID](https://openalex.org/A5028693655)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了在可学习观察者面前的重复欺骗路径规划（RDPP）任务，并提出了双层优化框架 Deceptive Meta Planning（DeMP），通过episode层面快速适应和meta层面更新来维持长期欺骗效果。

**💡 创新点**

核心创新在于将元学习与欺骗规划结合，利用meta梯度提前预判观察者学习动态，从而缓解单纯反应式策略的适应滞后问题；同时提出了可与观察者预测直接关联的代理目标函数。

**🔧 技术方法**

采用Soft Actor-Critic（SAC）实现代理策略；观察者采用双层LSTM进行目标识别；代理策略通过episode级别的强化学习损失和meta级别的高阶梯度更新进行优化。

**📊 数据集**

实验使用标准49×49与100×100格子地图，设置五个候选目标和大规模障碍；此外在海盗追踪场景中加入动态追踪者进行进一步验证。

**📈 对比分析**

与Honest Agent、AM、DEAM以及仅进行episode级更新的Naïve Agent对比。DeMP在400轮交互中保持最低的目标识别概率（最高欺骗率），且路径成本仅略高于基线；在海盗追踪实验中捕获率最低，证明其在主动对抗环境中的优势。

**⚠️ 局限性**

实验仅限于离散格子环境，未覆盖更高维度或真实世界动态场景，且meta更新频率与计算成本之间仍需平衡。

---

## 60. Query-efficient model evaluation using cached responses

**arXiv ID:** 2605.07096 | [PDF](https://arxiv.org/pdf/2605.07096v1)

**作者:** Hayden Helm `[一作]` (Helivan), Carey Priebe `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种利用已缓存模型响应通过数据核视角空间（DKPS）预测新模型在多任务基准上的分数，从而显著减少所需查询数量；

**💡 创新点**

创新点在于证明在黑盒设置下，基于DKPS的最近邻回归具有查询效率，并结合样本分数的自适应集成方法；

**🔧 技术方法**

核心技术包括数据核视角空间构建、多维尺度映射、线性回归/最近邻回归、离线查询集选择（基于R²）以及离群度量；

**📊 数据集**

实验使用HELM‑Lite基准的四个子任务：MATH、LegalBench、MedQA、WMT‑14，利用已评估模型的响应缓存；

**📈 对比分析**

与无DKPS的基准（样本分数、总体均值）以及DKPS线性回归相比，集成方法在几乎所有查询预算和任务上取得最低MAE，尤其在低查询预算（m≈8–10）时可达10倍查询成本节约；

**⚠️ 局限性**

局限包括：需所有参考模型在同一查询集上评估；对随机或不完备查询集敏感；理论假设对噪声分数和非欧氏距离不一定成立；对不同模态或复杂评分函数的适用性仍需验证。

---

## 61. Bridging Textual Profiles and Latent User Embeddings for Personalization

**arXiv ID:** 2605.06981 | [PDF](https://arxiv.org/pdf/2605.06981v1)

**作者:** Zhaoxuan Tan `[一作]` (University of Notre Dame), Mohamed Hammad `[通讯]` (Google)

**通讯引用:** 16 | [OpenAlex ID](https://openalex.org/A5045567803)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种利用强化学习将文本用户画像与潜在用户嵌入相结合的个性化框架，学习可解释且能提升检索效果的用户文本摘要；

**💡 创新点**

创新点在于：①把用户画像生成视为策略学习任务，使用嵌入空间的对比奖励和文本空间的下一项预测奖励共同驱动生成；②采用Group Relative Policy Optimization在多候选生成上实现高效、稳定的梯度更新；③通过冻结或微调嵌入模型验证文本摘要在检索与问答任务中的双重有效性；

**🔧 技术方法**

使用的大模型：Gemma3‑1B‑it 作为生成器；Qwen3‑Embedding‑0.6B 作为嵌入模型；强化学习技术：GRPO、InfoNCE 对比奖励、下一项多选文本奖励；还用到监督对比学习 fine‑tune 嵌入模型；

**📊 数据集**

数据集：Amazon Reviews 2023（Clothing、Books、Electronics、Sports）与 Google Local Reviews（纽约州）；

**📈 对比分析**

与多类基线（原始历史、手工生成画像、RLAIF、RLPF、LangPTune 等）以及传统序列推荐模型（SASRec、BERT4Rec、UniSRec）比较；在冻结嵌入模型下取得 NDCG@10/Recall@10 最高，跨域与在域训练均表现更好；在个性化问答任务中，生成画像提升了下一项预测准确率和评分预测 MAE；

**⚠️ 局限性**

局限性包括：①缺乏统一的“真实”用户画像作为监督，RL奖励可能偏离人类期望；②依赖大型 LLM 与嵌入模型，计算与推理成本较高；③仅在电商与本地点评数据上验证，未检验对其他领域或多模态用户行为的泛化；④生成的画像可读性虽提升但仍受模型预训练偏差影响；

---

## 62. Why Does Agentic Safety Fail to Generalize Across Tasks?

**arXiv ID:** 2605.06992 | [PDF](https://arxiv.org/pdf/2605.06992v1)

**作者:** Yonatan Slutzky `[一作]` (Tel Aviv University), Nadav Cohen `[通讯]` (Tel Aviv University)

**通讯引用:** 1030 | [OpenAlex ID](https://openalex.org/A5104108669)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了多任务AI代理在安全性上的泛化能力，先用理论证明安全约束导致任务到控制器映射的Lipschitz常数更大，从而安全性更难泛化；随后在线性二次控制、四旋翼导航和LLM CRM任务等三个实验设置中验证这一结论。

**💡 创新点**

创新点在于首次从理论角度证明安全要求会显著增加任务-控制器映射的Lipschitz常数，进而导致跨任务泛化难度提升；并给出了LQR映射的上界，构成了对安全泛化难度的量化描述。

**🔧 技术方法**

技术方法包括：线性二次调节（LQR）与鲁棒控制理论、Lipschitz常数分析；仿真实验采用神经网络四旋翼导航、LLM（LLaMA-3.2）CRM任务，以及线性控制仿真；训练采用仿真教师（安全/不安全）和Imitation Learning。

**📊 数据集**

实验数据集：线性二次控制中的随机Q矩阵；四旋翼仿真中的随机目标位置与随机障碍物集合；CRM benchmark中的自然语言任务模板；教师策略使用GPT‑5.2，学生策略使用LLaMA‑3.2（或三层全连接网络）。

**📈 对比分析**

对比方法是将安全教师与不安全教师在训练任务和未见任务上的模仿误差进行对比。实验结果显示，在训练任务上两者误差相近，但在未见任务上安全教师的误差显著更高，说明安全泛化效果差；在所有三种设置中均出现类似趋势。

**⚠️ 局限性**

局限性包括：只考虑通过仿真学习安全，未探讨交互式学习；理论仅适用于线性二次控制且仅关注风险处理；依赖稳定性、对称互易和对齐正则等技术条件；使用Lipschitz常数作为泛化指标，未直接分析有限样本效应；在多任务RL/LLM环境中未给出完整理论支持。

---

## 63. The Cost of Consensus: Malignant Epistemic Herding and Adaptive Gating in Distributed Multi-Agent Search

**arXiv ID:** 2605.06988 | [PDF](https://arxiv.org/pdf/2605.06988v1)

**作者:** David Farr `[一作]` (University of Washington), Jevin West `[通讯]` (University of Washington)

**通讯引用:** 6567 | [OpenAlex ID](https://openalex.org/A5046879461)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究分布式多智能体搜索任务中通信内容和频率如何影响任务成功与集体认知一致性；

**💡 创新点**

提出恶性认知集体（Malignant Epistemic Herding, MEH）概念，并设计熵增量门控（entropy‑delta gating）作为自适应通信机制，既抑制频繁信任融合导致的错误收敛，又保持高效信息传递；

**🔧 技术方法**

使用离散网格搜索仿真、离散概率分布、Shannon 熵、逆熵加权融合、事件触发门控以及完整因子实验设计；

**📊 数据集**

在 50×50 网格上随机放置单一目标，使用 1,000 条情节对每个 108 条件进行仿真，另外对 25×25、50×50、75×75、100×100 等不同规模网格进行扩展实验；

**📈 对比分析**

与无通信、点估计（C1）、完整分布频繁传输（C2）和门控分布（C3）四种协议比较，结果显示 C3 在任务成功率、真值对齐度和 MEH 率方面优于其它方案，且传输量减少 98%；

**⚠️ 局限性**

局限包括使用贪婪移动策略、同质智能体、单目标静态环境、理想传感器、同步执行和基于广播的网络模型，未来需扩展到异构、动态、多目标、噪声感知和异步系统等更复杂场景。

---

## 64. From 0-Order Selection to 2-Order Judgment: Combinatorial Hardening Exposes Compositional Failures in Frontier LLMs

**arXiv ID:** 2605.07268 | [PDF](https://arxiv.org/pdf/2605.07268v1)

**作者:** Hanmeng Liu `[一作]` (Hainan University), Xiaozhang Liu `[通讯]` (Hainan University)

**通讯引用:** 606 | [OpenAlex ID](https://openalex.org/A5039994407)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出LogiHard框架，将多选题通过逻辑组合合成转化为2阶逻辑判断，并使用IRT-CAT进行动态自适应评估；

**💡 创新点**

创新点在于：①通过确定性逻辑组合硬化保证验证性与可扩展性；②将Item Response Theory与自适应测试结合，实时生成难度；③揭示LLM在多选组合推理中的“组合映射缺失”瓶颈；

**🔧 技术方法**

采用逻辑组合合成、9维认知评分、基于3PL的IRT-CAT、自动定理验证、LLM链式思考与自适应更新等技术；

**📊 数据集**

使用由6,235道中英高考、LSAT、GMAT等构成的LogiHard-2k数据集，并在MMLU上做零转移评估；

**📈 对比分析**

通过12个顶尖LLM在原始、NOTA、shuffle、Hard-Base、Hard-Comb等多模式下比较，Hard-Comb平均降幅31%–56%，人类79.5%；在MMLU零转移下降46.99%，显示跨域适用性；

**⚠️ 局限性**

局限性在于：①数据集仅来自中英考试，缺乏跨文化覆盖；②Gold Score仅基于单一模型的思考轨迹；③使用固定3PL参数，未考虑共享上下文的测试单元效应；

---

## 65. Low-code and no-code with BESSER to create and deploy smart web applications

**arXiv ID:** 2605.07376 | [PDF](https://arxiv.org/pdf/2605.07376v1)

**作者:** Iván Alfonso `[一作]` (Luxembourg Institute of Science and Technology), Jordi Cabot `[通讯]` (Luxembourg Institute of Science and Technology)

**通讯引用:** 8907 | [OpenAlex ID](https://openalex.org/A5074872542)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并演示了开源低代码框架BESSER，用于设计、生成和部署含AI代理的智能Web应用。

**💡 创新点**

提供B-UML建模语言、无代码GUI编辑器以及全链路自动代码生成和一键云部署，解决商业LCDP的厂商锁定和可扩展性缺陷。

**🔧 技术方法**

使用Python（FastAPI、Pydantic、SQLAlchemy）、React+TypeScript、WebSocket、LLM交互、GitHub+Render云部署。

**📊 数据集**

未使用公开数据集，仅以图书馆FAQ场景作为示例。

**📈 对比分析**

未给出定量对比，仅通过视频演示展示生成代码可直接部署，部署时间仅数分钟。

**⚠️ 局限性**

缺乏性能评估和大规模实验，当前示例仅限于单一场景，未来需验证在更复杂应用中的可扩展性。

---

## 66. Masks Can Talk: Extracting Structured Text Information from Single-Modal Images for Remote Sensing Change Detection

**arXiv ID:** 2605.07178 | [PDF](https://arxiv.org/pdf/2605.07178v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 67. Discovering Ordinary Differential Equations with LLM-Based Qualitative and Quantitative Evaluation

**arXiv ID:** 2605.07323 | [PDF](https://arxiv.org/pdf/2605.07323v1)

**作者:** Sum Kyun Song `[一作]` (Chung-Ang University), Jae Yong Lee `[通讯]` (Chung-Ang University)

**通讯引用:** 9160 | [OpenAlex ID](https://openalex.org/A5100369464)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a8e75ba4-7a2d-4153-b003-06c94533add0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出DoLQ框架，通过LLM采样器、参数优化器和科学家代理实现基于定性物理推理与定量误差评估的迭代式ODE发现。

**💡 创新点**

创新点在于将LLM的定性物理推理与定量误差评估融合到多代理循环中，并引入科学家代理进行语义评估以指导候选项生成；同时采用混合全局-局部参数优化。

**🔧 技术方法**

使用Gemini 2.5 Flash Lite等LLM进行候选项生成与评估，差分进化+BFGS混合优化，残差与积分NMSE等定量误差指标，定性定量交互评估以及多代理协同搜索。

**📊 数据集**

利用ODEbench的七个标准多维ODE任务（SIR、CDIMA、Glider等）以及扩展的ID-Ext测试集，总共八个数据集。

**📈 对比分析**

与ICSR、LASR、LLM‑SR、EDL等基线在积分NMSE、残差NMSE及成功率等指标上对比，DoLQ在大多数任务中实现最高成功率（≈80%+），且NMSE显著低于基线。

**⚠️ 局限性**

局限性包括：依赖数值微分导致对噪声敏感；定性推理可能导致过早收敛到错误假设；目前仅针对ODE，推广到PDE仍面临方法与求解器的挑战。

---

## 68. In-Context Credit Assignment via the Core

**arXiv ID:** 2605.06920 | [PDF](https://arxiv.org/pdf/2605.06920v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 69. Unlocking High-Fidelity Molecular Generation from Mass Spectra via Dual-Stream Line Graph Diffusion

**arXiv ID:** 2605.07048 | [PDF](https://arxiv.org/pdf/2605.07048v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 70. LoHGNet: Infrared Small Target Detection through Lorentz Geometric Encoding with High-Order Relation Learning

**arXiv ID:** 2605.07213 | [PDF](https://arxiv.org/pdf/2605.07213v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 71. MIND: Monge Inception Distance for Generative Models Evaluation

**arXiv ID:** 2605.06797 | [PDF](https://arxiv.org/pdf/2605.06797v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 72. RateQuant: Optimal Mixed-Precision KV Cache Quantization via Rate-Distortion Theory

**arXiv ID:** 2605.06675 | [PDF](https://arxiv.org/pdf/2605.06675v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 73. Movable Subarray-Aided Hybrid Beamforming for Near-Field Multiuser Communications

**arXiv ID:** 2605.07261 | [PDF](https://arxiv.org/pdf/2605.07261v1)

**作者:** Xiangqian Xu `[一作]` (University of Electronic Science and Technology of China), Arumugam Nallanathan `[通讯]` (Queen Mary University of London)

**通讯引用:** 32500 | [OpenAlex ID](https://openalex.org/A5002265731)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在近场多用户MISO系统中提出基于可移动子阵列（MSA）的混合波束成形（HBF）框架，并通过交替优化（AO）联合设计数字、模拟波束以及MSA位置以最大化系统总速率。

**💡 创新点**

创新点在于：① 将MSA的空间可变度与近场距离相关自由度相结合，实现了3D精准波束聚焦；② 针对XL‑MIMO的硬件与能耗限制，引入可移动子阵列与子连HBF的组合，兼顾可行性与性能；③ 采用分数规划、ADMM及投影梯度上升的组合，显著降低优化复杂度。

**🔧 技术方法**

主要技术：混合波束成形、近场波束成形模型、分数规划（FP）、交替方向乘子法（ADMM）、投影梯度上升、梯度投影、模拟仿真。

**📊 数据集**

实验基于仿真数据：64‑天线基站（30 GHz），16个用户，子阵列4‑天线，BS阵列尺寸20λ，MSA可移动区域为A/4×A/4，传输功率10 dBm，噪声-80 dBm，Monte‑Carlo 1000次。

**📈 对比分析**

与三种基线比较：1）离散网格的穷举搜索；2）稀疏UPA；3）密集UPA。结果显示，MSA‑HBF在约30次迭代内收敛，且在不同区域大小与功率下均能显著优于稀疏/密集UPA，逼近穷举搜索的性能。

**⚠️ 局限性**

局限性：梯度上升方法可能陷入局部最优，尤其在大区域尺寸下；仿真未考虑硬件非理想、运动延迟与能耗；模型仅为单频单用户仿真，未验证多频/多用户更复杂场景。

---

## 74. Learning Agent Routing From Early Experience

**arXiv ID:** 2605.07180 | [PDF](https://arxiv.org/pdf/2605.07180v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 75. SmellBench: Evaluating LLM Agents on Architectural Code Smell Repair

**arXiv ID:** 2605.07001 | [PDF](https://arxiv.org/pdf/2605.07001v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 76. Latent Order Bandits

**arXiv ID:** 2605.07304 | [PDF](https://arxiv.org/pdf/2605.07304v1)

**作者:** Emil Carlsson `[一作]` (Sleep Cycle AB), Fredrik D. Johansson `[通讯]` (Chalmers University of Technology)

**通讯引用:** 5037 | [OpenAlex ID](https://openalex.org/A5049528659)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了latent order bandits（LOB）模型，利用隐状态下的部分排序约束提升多臂赌博机的个性化性能。

**💡 创新点**

核心创新在于仅要求每个隐状态提供动作的部分优先顺序，而非完整奖励分布，从而允许同一状态下实例具有不同绝对奖励尺度。

**🔧 技术方法**

设计了两种算法：基于UCB的LOB-UCB与基于后验采样的LOB-TS，并通过投影到满足部分排序的参数集合实现。

**📊 数据集**

实验使用合成环境与MovieLens用户评分数据（k=19种电影类型），将用户聚类为m个隐状态。

**📈 对比分析**

与无结构MAB、传统完整-参数隐状态bandit以及Thompson采样基线相比，LOB-UCB/TS在有足够排序约束时表现优于或相当于完整模型，在同一隐状态奖励尺度不一致时更优。

**⚠️ 局限性**

局限性包括理论上UB仅为O(k√(mT))，未达到MAB最优O(√(kT))；算法在每轮投影时计算量大，且目前仅适用于非上下文型bandit。

---

## 77. On the Divergence of Differential Temporal Difference Learning without Local Clocks

**arXiv ID:** 2605.06874 | [PDF](https://arxiv.org/pdf/2605.06874v1)

**作者:** David Antrobius `[一作]` (University of Virginia), Shangtong Zhang `[通讯]` (University of Virginia)

**通讯引用:** 716 | [OpenAlex ID](https://openalex.org/A5033834190)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在平均报酬强化学习中，研究了差分 TD（Differential TD）算法的学习率时钟选择，证明当使用全局时钟（仅依赖时间步）时，算法在某些超参数 η 下会发散，而使用局部时钟（依赖状态访问次数）时则收敛，给出了一个具体的反例。

**💡 创新点**

创新点在于揭示平均报酬 RL 与折扣 RL 的差异：在平均报酬情形下，局部时钟与全局时钟的收敛性不再等价；并通过解析正稳定性和秩一次扰动理论构造了一个极具洞察力的反例，证明了此前关于正稳定性的猜想是错误的。

**🔧 技术方法**

主要技术包括：ODE 方法与全局渐近稳定性分析、秩一次扰动理论、Routh‑Hurwitz 判别法、谱分析以及对正稳定性阈值 η_* 的解析推导。

**📊 数据集**

使用的“数据集”是人工构造的 25 状态（m=23）马尔可夫决策过程，状态转移矩阵和学习率参数按理论构造得到；没有使用公开 RL 基准数据集。

**📈 对比分析**

实验通过在构造的 MDP 上同时实现差分 TD 的全局时钟与局部时钟版本，采用相同的学习率序列和初始估计；结果显示在 η = 2α（处于不稳定区间）时，全局时钟递推发散，而局部时钟递推收敛至 0，验证了理论结论。

**⚠️ 局限性**

局限性在于：仅对差分 TD 及其特定学习率形式给出结论；未证明全局时钟在所有 η > 0 下几乎必然发散；对更通用的平均报酬算法的推广仍为开放问题；实验仅在极小规模手工构造 MDP 上进行，缺乏对真实任务的验证。

---

## 78. Qwen3-VL-Seg: Unlocking Open-World Referring Segmentation with Vision-Language Grounding

**arXiv ID:** 2605.07141 | [PDF](https://arxiv.org/pdf/2605.07141v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 79. Reformulating KV Cache Eviction Problem for Long-Context LLM Inference

**arXiv ID:** 2605.07234 | [PDF](https://arxiv.org/pdf/2605.07234v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 80. A Unified Open-Set Framework for Scalable PUF-Based Authentication of Heterogeneous IoT Devices

**arXiv ID:** 2605.07340 | [PDF](https://arxiv.org/pdf/2605.07340v1)

**作者:** Xin Wang `[一作]` (Chinese University of Hong Kong), Yue Zheng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 13519 | [OpenAlex ID](https://openalex.org/A5100636088)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个统一的开集物理不可克隆函数（PUF）身份认证框架，能够在异构物联网设备网络中对设备进行可靠识别和伪装拒绝。

**💡 创新点**

通过使用 OpenGAN 的生成器-判别器结构在不收集真实异常样本的情况下学习已注册设备与未知设备的边界，并将不同类型 PUF 的原始响应统一映射为灰度图像，突破了 PUF 架构兼容性瓶颈。

**🔧 技术方法**

采用 LFSR 扩展挑战、灰度图像编码、ResNet‑18 CNN 与 OpenGAN 训练、混合加密（RSA+AES）以及 Bloom filter 进行重放检测，最终在 Raspberry Pi 上实现快速原型。

**📊 数据集**

实验使用了四组公开数据集：Arbiter、SRAM、噪声 DRAM 以及混合（Arbiter+SRAM）数据，涉及 45 台设备以上，涵盖强弱混合多种 PUF。

**📈 对比分析**

与现有分类方案（如 Mexis、Millwood 等）以及传统 CRP 数据库/模型方法对比，取得 100% 关闭集准确率、FAR<0.5%、FRR<1%，并在单次鉴权周期内完成 0.67 s，速度提升约 30 倍。

**⚠️ 局限性**

目前在极端环境噪声条件下的鲁棒性仍待进一步验证，且在大规模部署时 Bloom filter 的内存占用与阈值调优可能成为性能瓶颈。

---

## 81. Dr. Post-Training: A Data Regularization Perspective on LLM Post-Training

**arXiv ID:** 2605.07063 | [PDF](https://arxiv.org/pdf/2605.07063v1)

**作者:** Pingbang Hu `[一作]` (University of Illinois Urbana--Champaign), Jiaqi W. Ma `[通讯]` (University of Illinois Urbana--Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 Dr. Post-Training 的框架，将一般训练数据视为对稀缺目标数据的正则化器，用于 LLM 的后期训练，避免传统的数据选择问题。

**💡 创新点**

创新点在于将一般训练数据从“样本池”转化为“数据诱导正则化”，构造可调的可行集实现 bias–variance 平衡，并引入 Group‑Wise Subset Update 以进一步细化参数分组正则化。

**🔧 技术方法**

技术方法包括：基于大前向后向一次计算的自定义张量生命周期调度、压缩打分（Compressed Scoring）、分组子集投影、LoRA/MeSO 参数高效化、激活检查点兼容性以及软权重化扩展。

**📊 数据集**

实验使用 Llama‑3.2‑1B、GPT‑Neo‑2.7B、Qwen3‑1.7B 等模型，在 SFT（对话摘要、多语言 QA、闭卷问答、阅读理解 QA）和 RLHF / RLVR（去毒、可验证奖励）任务中，并在 SmolLM2‑360M、TinyLlama‑1.1B、Llama‑3.2‑3B 上进行系统基准。

**📈 对比分析**

与标准训练和现有最佳数据选择基线（如 GREATS、offline/global subset 等）对比，Dr. Post-Training 在所有任务上均取得显著提升（如 QA F1 提升至 27.4%，RLHF 收敛更快且毒性更低，RLVR 准确率更高），且仅增加 3–34% 的运行时间和极小的内存占用。

**⚠️ 局限性**

局限性包括：对非常小的模型时开销不小；需要精细划分分组以兼容激活检查点；近似压缩打分可能影响样本排序的准确性；在多源域适配场景下仍需进一步验证。

---

## 82. Tyche: One Step Flow for Efficient Probabilistic Weather Forecasting

**arXiv ID:** 2605.06916 | [PDF](https://arxiv.org/pdf/2605.06916v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 83. Learning Material-Aware Hamiltonian Risk Fields for Safe Navigation

**arXiv ID:** 2605.07038 | [PDF](https://arxiv.org/pdf/2605.07038v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 84. SAGE: Hierarchical LLM-Based Literary Evaluation through Ontology-Grounded Interpretive Dimensions

**arXiv ID:** 2605.07102 | [PDF](https://arxiv.org/pdf/2605.07102v1)

**作者:** Tianyu Wang `[一作]` (Mercy University), Nianjun Zhou `[通讯]` (IBM)

**通讯引用:** 663 | [OpenAlex ID](https://openalex.org/A5020632441)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SAGE 体系，将文学质量分解为六个层级，并通过 LLM 进行分层、结构化评估。

**💡 创新点**

创新点在于：①基于本体论的解释维度拆分；②多轮自反推理与独立验证双轨架构；③内容与标题两种模式的对比评估。

**🔧 技术方法**

使用 GPT‑5‑mini 大语言模型，配合定制化提示工程与五轮迭代推理。

**📊 数据集**

采用 100 条短篇小说数据集：50 经典文学、30 垃圾推理小说、20 LLM 生成文本，按 2,000–8,000 词长度筛选。

**📈 对比分析**

对 600 次评估（3 层 × 2 模式 × 100 文本）进行统计：收敛率 98.8%，评审一致率 >94%，内容/标题模式差异 ≤0.05；按类别比较显示经典>通俗>LLM，所有层均显著，文化与哲学维度差距更大。

**⚠️ 局限性**

局限性：仅测试单一 LLM；仅包含英文短篇；缺乏人类专家对照；未验证跨语言或更大文本形式的适用性。

---

## 85. When Does a Language Model Commit? A Finite-Answer Theory of Pre-Verbalization Commitment

**arXiv ID:** 2605.06723 | [PDF](https://arxiv.org/pdf/2605.06723v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 86. Evidence-Tracked Tape Semantics for Probabilistic Computation

**arXiv ID:** 2605.07259 | [PDF](https://arxiv.org/pdf/2605.07259v1)

**作者:** Liron Cohen `[一作]` (Ben-Gurion University), Tomer Samara `[通讯]` (Ben-Gurion University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出一种基于随机磁带的证据追踪语义模型，用以在高阶逻辑中证明概率程序的属性，并通过后置提取实现数值概率推理。

**💡 创新点**

在现有磁带语义中引入证据追踪与可证据变换，提出可实现的磁带重排和拆分（独立抽样）原理，并给出从磁带层到概率律层的桥接定理。

**🔧 技术方法**

采用单子核心→证据框架→三重（tripos）的通用构造，结合测度论中的随机磁带、可证据映射与拆分映射，以及期望提取与几乎必然商。

**📊 数据集**

无（本文为理论性研究，无实验数据集）。

**📈 对比分析**

无实验对比，主要提供理论证明与抽象性证明技术；若有实现，理论上可支持独立采样的误差降低等。

**⚠️ 局限性**

局限性：仅适用于可数结果类型，未考虑连续空间；实现依赖于可证据磁带映射的可计算性；实验评估缺失。

---

## 87. Self Driving Datasets: From 20 Million Papers to Nuanced Biomedical Knowledge at Scale

**arXiv ID:** 2605.07022 | [PDF](https://arxiv.org/pdf/2605.07022v1)

**作者:** Haydn Jones `[一作]` (University of Pennsylvania), Jacob R. Gardner `[通讯]` (University of Pennsylvania)

**通讯引用:** 1973 | [OpenAlex ID](https://openalex.org/A5072585411)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套基于 PubMed 文献的自动化数据集生成系统，利用实体标注、混合稀疏‑稠密检索与多智能体深度研究实现从文本到结构化记录的全流程抽取；

**💡 创新点**

创新点在于①全谱医学实体标注覆盖19类，②基于实体过滤的检索策略显著降低搜索成本，③多智能体自动化制定检索与抽取方案，并在抽取后进行多轴人工评估，生成包含上下文细节的高质量数据；

**🔧 技术方法**

核心技术包括大型语言模型（gpt‑oss‑120b、GPT‑5、Qwen3.5‑9B）进行实体标注与抽取，基于 UMLS/OPSIN/ChEBI/UniProt 的命名实体归一化，混合稀疏‑稠密向量检索，和多智能体协作框架实现自驱动抽取；

**📊 数据集**

使用规模达 22.5M 篇 PubMed 论文、2.5T tokens 的语料，生成 6 个任务数据集（血脑屏障通透性、口服生物利用度、LD50、基因‑疾病关联、蛋白亚细胞定位、化学反应产物），共计 91K–3M 条记录；

**📈 对比分析**

与现有手工整理基准（TDC、B3DB、ClinVar、UniProt、ORD）对比，错误率降至 0.6–7.7%（低于手工数据 5–16%），数据规模相当或更大，且抽取成本约 0.001 美元/条；

**⚠️ 局限性**

局限在未处理图表信息、仅依赖 PubMed 许可文本、对新型实体类别或实验细节的覆盖仍有限，以及模型在跨语料迁移时的适用性需进一步验证。

---

## 88. Gradient Extrapolation-Based Policy Optimization

**arXiv ID:** 2605.06755 | [PDF](https://arxiv.org/pdf/2605.06755v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 89. Group of Skills: Group-Structured Skill Retrieval for Agent Skill Libraries

**arXiv ID:** 2605.06978 | [PDF](https://arxiv.org/pdf/2605.06978v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 90. TraXion: Rethinking Pre-training Frameworks for Mobility and Beyond

**arXiv ID:** 2605.06906 | [PDF](https://arxiv.org/pdf/2605.06906v1)

**作者:** Shang-Ling Hsu `[一作]` (University of Southern California), Khurram Shafique `[通讯]` (Novateur Research Solutions)

**通讯引用:** 2121 | [OpenAlex ID](https://openalex.org/A5008076815)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了针对多实体时空事件流（MESES）的自监督预训练框架TraXion，利用噪声检测与实体原型对齐的双重目标进行预训练。

**💡 创新点**

首先将MESES的三条结构性公理（联合属性密度、实体耦合表示、共享上下文）形式化，随后设计满足这些公理的预训练目标与多轴注意力架构，实现跨领域事件流的统一建模。

**🔧 技术方法**

采用噪声检测（类似ELECTRA）、实体原型对比损失、因式化多轴注意力（序列、特征、共现轴）以及共现子层的Transformer骨干网络。

**📊 数据集**

使用六个公共移动数据集（NUMOSIM-LA、UA-Berlin/Atlanta、Foursquare-Tokyo、Gowalla-Stockholm/Austin）、LANL企业认证日志和PhysioNet eICU-CRD临床事件作为实验数据。

**📈 对比分析**

在异常检测、下一POI/访问预测、社交链接推断、ICU死亡预测等四类任务上与多种基线（UniTraj、LogGPT、CEHR-BERT等）对比，TraXion在所有任务与数据集上均取得或逼近最佳性能，异常检测AP提升高达22个百分点。

**⚠️ 局限性**

局限性包括：仅为判别式模型，无法生成事件；共现子层需手工切换；未训练跨实例统一的MESES基线；在时间预测任务上性能低于部分基线模型。

---

## 91. Improved Model-based Reinforcement Learning with Smooth Kernels

**arXiv ID:** 2605.07218 | [PDF](https://arxiv.org/pdf/2605.07218v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 92. Direct-to-Event Spiking Neural Network Transfer

**arXiv ID:** 2605.07207 | [PDF](https://arxiv.org/pdf/2605.07207v1)

**作者:** Nhan Trong Luu `[一作]` (Can Tho University), Truong Cong Thang `[通讯]` (University Of Aizu)

**通讯引用:** 2553 | [OpenAlex ID](https://openalex.org/A5078708126)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出直接编码到事件编码的SNN迁移问题，并给出了自我知识蒸馏（SKD）方法实现能效高效迁移

**💡 创新点**

首次定义D2E迁移问题，结合自蒸馏理论，提出针对事件编码的迁移优化框架

**🔧 技术方法**

自蒸馏、事件编码（TTFS）、LIF神经元、BPTT、SpikingJelly与PyTorch框架

**📊 数据集**

CIFAR-10、CIFAR-100、ImageNet以及基于DVS的模拟CIFAR数据集

**📈 对比分析**

与传统的直接微调TSF对比，SKD在多种网络（ResNet、VGG、WRN、SEW-RN）上提升30–50%的验证准确率，并在TTFS编码下将能耗降低约45%

**⚠️ 局限性**

额外的训练开销（≈33%）和对预训练模型质量的依赖，若教师模型性能低则蒸馏效果减弱

---

## 93. Almost Sure Convergence Rates of Stochastic Approximation and Reinforcement Learning via a Poisson-Moreau Drift

**arXiv ID:** 2605.07104 | [PDF](https://arxiv.org/pdf/2605.07104v1)

**作者:** Xinyu Liu `[一作]` (University of Virginia), Shangtong Zhang `[通讯]` (University of Virginia)

**通讯引用:** 716 | [OpenAlex ID](https://openalex.org/A5033834190)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究了在马尔可夫噪声下，随机逼近和强化学习算法的几乎确定收敛速率，特别是针对一类期望更新为收缩映射的随机逼近算法。

**💡 创新点**

创新点在于提出了一种新的Poisson方程修正的Lyapunov漂移构造，能够在马尔可夫噪声下获得更精确的几乎确定收敛速率。

**🔧 技术方法**

使用了Moreau-envelope平滑技术和Poisson方程修正的方法。

**📊 数据集**

未具体提及使用的数据集，但讨论了与Q学习和线性时间差学习等强化学习算法相关的设置。

**📈 对比分析**

与现有文献相比，本文的收敛速率在学习率范围内更广泛且更快，尤其在η∈(1/2, 1)的情况下，收敛速率接近o(n^(1-2η))，而对于η=1的情况，接近o(n^-1)。

**⚠️ 局限性**

限制在于虽然提出的收敛速率比现有结果更快，但仍未达到最优速率o(n^-1loglog n)，并且尚不清楚该方法是否可以扩展到η∈(0, 1)的全范围。

---

## 94. VITA-QinYu: Expressive Spoken Language Model for Role-Playing and Singing

**arXiv ID:** 2605.06765 | [PDF](https://arxiv.org/pdf/2605.06765v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 95. From Assistance to Agency: Rethinking Autonomy and Control in CI/CD Pipelines

**arXiv ID:** 2605.07062 | [PDF](https://arxiv.org/pdf/2605.07062v1)

**作者:** Marcus Emmanuel Barnes `[一作]` (University of Toronto), Safwat Hassan `[通讯]` (University of Toronto)

**通讯引用:** 494 | [OpenAlex ID](https://openalex.org/A5022060601)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过梳理研究原型与工业平台，提出了“权力转移（authority transfer）”的概念，并用数据平面（data-plane）/控制平面（control-plane）区分来重新定义 CI/CD 中的代理（agent）自治水平；同时归纳了当前实践中的三大模式（受限自治、治理为安全支撑、评估滞后），并基于此提出了四项研究议程。

**💡 创新点**

创新点在于：① 把 CI/CD 中代理自治视为权力转移问题，而非单纯提升任务性能；② 引入数据平面/控制平面区分，为讨论自治边界与治理机制提供了新的术语；③ 系统性总结当前实践的三大模式并提出有序的研究优先级；④ 指出缺失的评估框架并提出未来评估维度。

**🔧 技术方法**

主要技术/概念包括：大语言模型驱动的代理（LLM agents）、MAPE‑K 反馈循环模型、基于策略的代理（policy‑aware agents）、运行时监控与边界验证、GitHub/仓库权限控制与 PR 工作流。

**📊 数据集**

本文并未在实验上使用特定数据集；它依赖公开的工业平台文档、研究原型描述以及对 GitHub 相关公开数据（如 AIDev）进行的概念性讨论。

**📈 对比分析**

由于缺乏实证评估，本文没有给出性能数值或对比方法；它指出现有 CI/CD 评估指标（构建时长、部署频率、故障率）无法衡量代理决策质量、治理影响等系统级指标，并呼吁建立新的评估框架与基准。

**⚠️ 局限性**

局限性包括：① 依赖公开工业文档，可能不完整或有营销倾向；② 当前技术发展迅速，实证数据稀缺，缺乏长期效果评估；③ 本文为概念性/系统性综述，未给出量化实验结果。

---

## 96. Benchmarking Large Language Models for IoC Recovery under Adversarial Code Obfuscation and Encryption

**arXiv ID:** 2605.06910 | [PDF](https://arxiv.org/pdf/2605.06910v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 97. Target-Aware Data Augmentation for SAT Prediction

**arXiv ID:** 2605.06931 | [PDF](https://arxiv.org/pdf/2605.06931v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 98. Deciding DFA-Primality is NP-Hard

**arXiv ID:** 2605.07031 | [PDF](https://arxiv.org/pdf/2605.07031v1)

**作者:** Daniel Alexander Spenner `[一作]` `[通讯]` (Technische Universität Dortmund), Daniel Alexander Spenner (Technische Universität Dortmund)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

通过构造特殊的 DFA 并基于最大访问词的泵送性质，证明了判定 DFA 是否为质数（不可分解）的 NP 难度，从而突破了之前仅有 NL 难度与 ExpSpace 上限的结果。

**💡 创新点**

首次利用“最大访问词”与“pumping”特性为最小线性安全 DFA 的可分解性提供了完整的判定标准，并在此基础上给出从 SAT 到 DFA 质数判定的多项式时间规约，完成了 NP 难度证明。

**🔧 技术方法**

核心技术包括：① 针对 CNF 公式构造特定的线性安全 DFA（A_Φ）以编码公式与赋值；② 证明该 DFA 的最大访问词形如 udc^，并通过泵送分析判定是否满足公式；③ 通过对最大访问词的泵送性质（p-条件）得到 DFA 可分解性的充要条件。

**📊 数据集**

无真实数据集，论文完全是理论性证明，所有实验和验证均在抽象的 DFA 构造与逻辑推理基础上完成。

**📈 对比分析**

与之前的 NL‑完备性结果相比，本工作把复杂度上限从 NL 提升到 NP，证明了在加入接受 sink 的“几乎无环” DFA（最小线性安全 DFA）下，质数判定问题成为 NP‑完备；同时表明在真正无环 DFA（有限语言 DFA）下仍保持 NL‑完备。

**⚠️ 局限性**

限制在于：① 只适用于 DFA，未考虑 NFA 或更一般自动机；② 仅给出理论上的 NP 难度与 NP 完备性，没有提供多项式时间的判定算法；③ 对于实际大规模 DFA 的可分解性判定仍缺乏高效实现与评估。

---

## 99. Same Signal, Opposite Meaning: Direction-Informed Adaptive Learning for LLM Agents

**arXiv ID:** 2605.06908 | [PDF](https://arxiv.org/pdf/2605.06908v1)

**作者:** Ziming Li `[一作]` (University of Connecticut), Chuxu Zhang `[通讯]` (University of Connecticut)

**通讯引用:** 5650 | [OpenAlex ID](https://openalex.org/A5022275632)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对LLM代理的自适应测试时计算问题，提出一种基于实测计算效益的门控方法DIAL，以避免传统固定方向门控导致的计算浪费甚至性能下降；

**💡 创新点**

核心创新在于：①从“计算需求”到“计算适用性”两源模型解释信号方向逆转；②采用无信号偏倚的随机探索收集对比数据；③使用稀疏线性门结合多源特征实现环境与模型特定的方向学习；

**🔧 技术方法**

主要技术包括对抗式对比实测收益（paired counterfactual rollout）、稀疏L1正则化逻辑回归门、LLM生成任务特定特征池、以及对信号-效益相关性的Spearman/Simpson分析；

**📊 数据集**

在六大多样化环境（HotpotQA、FEVER、WebShop、APPS、TWExpress、Plancraft）和三种LLM骨干（Qwen3、Phi-3.5、Llama-3.1）上进行实验；

**📈 对比分析**

与多种固定方向或固定预算基线（CaTS、SEAG、CoRefine、CATTS、AUQ、s1_budget）对比，DIAL在大多数环境/骨干组合下实现了更高的成功率与更低的计算成本，整体Pareto优势显著；

**⚠️ 局限性**

主要局限包括：对信息稀缺任务（如FEVER）探索覆盖不足、一次性探索成本需在多次推理中摊销，以及对极端计算适用性环境（如Plancraft）仍需进一步的自适应策略或奖励调优。

---

## 100. PaT: Planning-after-Trial for Efficient Test-Time Code Generation

**arXiv ID:** 2605.07248 | [PDF](https://arxiv.org/pdf/2605.07248v1)

**作者:** Youngsik Yoon `[一作]` (POSTECH), Jungseul Ok `[通讯]` (POSTECH)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种先试后规划（PaT）自适应策略，并将其与异构模型配置相结合，用于提升代码生成的推理成本与性能。

**💡 创新点**

创新点在于将规划过程推迟到生成失败后触发，实现推理成本自适应控制；以及通过将小模型负责生成、强模型负责规划的异构部署，显著降低整体成本。

**🔧 技术方法**

采用生成器‑规划器‑执行器三阶段管道，利用测试集验证作为失败信号；使用自生成测试用例、最佳‑N 采样、平稳阈值停机等技术；在 Qwen3 系列、Llama‑3.1、DeepSeek‑Coder 等 LLM 上实现。

**📊 数据集**

使用 HumanEval、MBPP 及其 EvalPlus 扩展版，以及 xCodeEval（Easy、Mid、Hard、Expert 四类）等公开代码生成基准数据集。

**📈 对比分析**

与 Standard、Best‑of‑N、CodeT、FunCoder 等基线对比，PaT 在所有模型规模和基准上均取得更高 Pass@1，平均通过率可与大模型相当，同时平均推理成本下降约 60%，异构配置更进一步将成本降低约 69%。

**⚠️ 局限性**

局限性包括：依赖自生成测试用例会产生噪声，需要平稳停机启发式；异构配置会增加静态内存占用；并未完全解决所有复杂任务的规划与生成需求。

---

## 101. An Embarrassingly Simple Graph Heuristic Reveals Shortcut-Solvable Benchmarks for Sequential Recommendation

**arXiv ID:** 2605.07125 | [PDF](https://arxiv.org/pdf/2605.07125v1)

**作者:** Haoyu Han `[一作]` (Michigan State University), Jiliang Tang `[通讯]` (Michigan State University)

**通讯引用:** 26033 | [OpenAlex ID](https://openalex.org/A5040639891)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一个仅基于最近交互项的图检索启发式方法，对顺序推荐进行基准审计

**💡 创新点**

发现并归纳了三种数据集“shortcut”结构——低分支局部转移、特征平滑转移和对长历史依赖有限，使得简单方法可逼近甚至超越复杂模型

**🔧 技术方法**

构建物品转移图、利用L2归一化特征相似度与边权加权、无训练的检索+排序

**📊 数据集**

在14个包含丰富物品侧信息的数据集上评测，包括Amazon Review(Beauty,Sports,Toys,CDs)、Delicious,LastFM,MovieLens-1M,Yelp,MIND,Goodreads-Comics,Goodreads-Children,STEAM,H&M,Amazon-M2-UK

**📈 对比分析**

与LightGCN、SR‑GNN、SASRec、HSTU、TIGER、LETTER、CoFiRec等多类别基线对比，单方法在10/14数据集达到最佳或次优；在Amazon Review中相对NDCG@10提升高达44%

**⚠️ 局限性**

所提出的启发式仅捕捉局部结构和特征相似性，无法利用长序列依赖或复杂语义，因而在长序列或非平滑特征数据集上表现不足

---

## 102. IntentGrasp: A Comprehensive Benchmark for Intent Understanding

**arXiv ID:** 2605.06832 | [PDF](https://arxiv.org/pdf/2605.06832v1)

**作者:** Yuwei Yin `[一作]` (University of British Columbia), Giuseppe Carenini `[通讯]` (University of British Columbia)

**通讯引用:** 6113 | [OpenAlex ID](https://openalex.org/A5049259877)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个名为 Intent Understanding Benchmark（意图理解基准）的多领域、多实例类型的统一多选问答评测体系，并在此基础上提出了 Intentional Fine‑Tuning（IFT）训练方法，用于提升大型语言模型的意图理解能力。

**💡 创新点**

创新点在于①通过三阶段处理（数据集挑选、意图标签语境化、任务格式统一）将 49 个开放许可证、跨 12 个领域的多样数据集整合为统一的评测框架；②构建了规模庞大的训练集（262,759 条）及两套评测集（All Set 12,909 条、Gem Set 470 条）；③提出了 IFT 方法，直接在该训练集上微调 LLM，并通过留一域实验验证了跨域泛化能力。

**🔧 技术方法**

主要技术包括：多选问答格式设计、意图标签上下文化处理、基于多选答案的 F1 评估、随机选项置换、意图提示（IA）与链式推理（CoT）对比、使用 HuggingFace Transformers 进行大模型微调、实验对比方法（随机猜测、人类基准）。

**📊 数据集**

使用了 49 个公开数据集，涵盖 12 个领域（日常生活、智能助手、毒性言论、写作、通用、电子商务、教学、同理回应、新闻、客服、冠状病毒大流行、政策制定），文本形式包括 Query、Dialogue、Monologue，标签类型为单/多意图，注释来源为人工/人工合成。

**📈 对比分析**

通过在 All Set 和 Gem Set 上对 20 个 LLM（7 个家族，包含 GPT‑5、Gemini‑3、Claude‑4 等前沿模型）进行评测，发现所有模型在 All Set 上 F1 均低于 60%，在 Gem Set 上低于 25%，且 17/20 模型甚至低于 15.2% 的随机猜测基线；采用 IFT 后，Qwen3‑4B 在 All Set 上提升至约 70% F1，在 Gem Set 上提升至约 32% F1，显著超过原始模型和提示基线。

**⚠️ 局限性**

局限性包括：①评测集仍然偏向单一选项，难以充分体现多意图推理；②在新闻、政策制定等领域即使使用 IFT 仍表现差，说明对长文本和专业语境的意图识别仍不足；③微调仅在公开数据上完成，模型在真实对话场景中的安全性与稳健性尚待进一步验证；④评测中可能存在对训练数据泄漏的担忧，虽然实验表明影响有限，但仍需持续监测。

---

## 103. Edge Deep Learning in Computer Vision and Medical Diagnostics: A Comprehensive Survey

**arXiv ID:** 2605.06714 | [PDF](https://arxiv.org/pdf/2605.06714v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 104. Echo: KV-Cache-Free Associative Recall with Spectral Koopman Operators

**arXiv ID:** 2605.06997 | [PDF](https://arxiv.org/pdf/2605.06997v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 105. Hierarchical Perfusion Graphs for Tumor Heterogeneity Modeling in Glioma Molecular Subtyping

**arXiv ID:** 2605.07156 | [PDF](https://arxiv.org/pdf/2605.07156v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 106. TriDE: Triangle-Consistent Translation Directions for Global Camera Pose Estimation

**arXiv ID:** 2605.06889 | [PDF](https://arxiv.org/pdf/2605.06889v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 107. Guidelines for Cultivating a Sense of Belonging to Reduce Developer Burnout

**arXiv ID:** 2605.06827 | [PDF](https://arxiv.org/pdf/2605.06827v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 108. Switchcraft: AI Model Router for Agentic Tool Calling

**arXiv ID:** 2605.07112 | [PDF](https://arxiv.org/pdf/2605.07112v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 109. Attention Transfer Is Not Universally Effective for Vision Transformers

**arXiv ID:** 2605.07191 | [PDF](https://arxiv.org/pdf/2605.07191v1)

**作者:** Huaiyuan Qin `[一作]` (Institute for Infocomm Research (I²R), A*STAR), Hongyuan Zhu `[通讯]` (Institute for Infocomm Research (I²R), A*STAR)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

评估并系统验证了在不同ViT教师模型上使用Attention Transfer（注意力复制与蒸馏）在标准学生架构下的效果，并发现并非所有教师模型都能从注意力迁移中获益。

**💡 创新点**

首次揭示了Attention Transfer在不同ViT家族中的有效性边界，证明其成功与否取决于教师与学生之间的架构匹配；通过在学生中添加教师专有组件即可完全消除失败，厘清了架构不匹配是失败的根本原因。

**🔧 技术方法**

利用注意力复制、注意力蒸馏、Q/K/V分解、层级与组件级消融、KL散度测量等技术，结合对不同损失函数与预训练策略的系统对比实验。

**📊 数据集**

主要使用ImageNet‑1K作为训练和评估基准，进一步在iNaturalist细粒度子集和四个分布偏移基准上验证了结果的稳健性。

**📈 对比分析**

与从零随机初始化的No‑Transfer基线对比，七个成功家族的注意力迁移可提升最高13.9%，而四个失败家族在标准架构下甚至比基线低5.1%；在加入教师专有组件后，所有失败家族的迁移表现全部反转为正向提升。

**⚠️ 局限性**

实验仅覆盖标准ViT学生架构，未探讨其他模型结构或跨模态的迁移；虽然加入教师专有组件能解决问题，但对如何在更大规模或不同任务中系统化地调整架构仍缺乏指导；此外，研究聚焦于注意力通路，可能忽略了其他潜在可迁移机制。

---

## 110. Simple KNN-Based Outlier Detection Achieves Robust Clustering

**arXiv ID:** 2605.07130 | [PDF](https://arxiv.org/pdf/2605.07130v1)

**作者:** Tianle Jiang `[一作]` (Duke University), Yufa Zhou `[通讯]` (Duke University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过将KNN最近邻距离作为异常点判别指标，对鲁棒k‑means问题提出并实现了两种基于KNN的异常点剔除算法OKMeans和OKMeans2，并给出了常数因子近似保证。

**💡 创新点**

创新点在于首次为经典KNN异常检测方法提供严谨理论分析，证明在每个最优聚类至少包含cz个点（c>1）的实际假设下，KNN剔除z个点即可得到9-近似（c=3）或≈5.98-近似（c=3）到标准k‑means的常数因子近似，并在此基础上改进至更优近似。

**🔧 技术方法**

技术手段包括：最优中心的近似求解、核心点（coreset）构造、距离矩阵快速计算（FAISS/Scikit‑Learn）、以及对KNN距离的统计分析与证明。

**📊 数据集**

实验使用四个真实数据集：SKIN、SUSY、SHUTTLE和KDDFULL，分别在其中加入合成异常点或将少数类视为异常点进行评测。

**📈 对比分析**

与TIKMeans、IKMeans、RobustKMeans++、NKMeans、KMeans++等六个基线算法对比，结果显示OKMeans/OKMeans2在聚类代价、异常点召回率和运行时间上均能匹配或优于大多数基线，尤其在SUSY、KDDFULL等大规模数据集上实现了显著加速。

**⚠️ 局限性**

局限性包括：需满足每个最优簇至少cz个点的假设；KNN参数K必须线性依赖于z，单一固定常数K在某些数据上表现不佳；理论近似常数相对较大；核心点构造和KNN计算在极大数据集上仍有一定开销。

---

## 111. Inductive Power Grid Cascading Failure Analysis with GRU-Gated Graph Attention

**arXiv ID:** 2605.07010 | [PDF](https://arxiv.org/pdf/2605.07010v1)

**作者:** Tianxin Zhou `[一作]` (Santa Clara University), Haibing Lu `[通讯]` (Santa Clara University)

**通讯引用:** 1499 | [OpenAlex ID](https://openalex.org/A5060287641)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种可在不同电网间零样本迁移的自监督图注意力框架CG-CAE，用于识别电网线路在级联故障中的脆弱性。

**💡 创新点**

创新点在于：① 引入GRU门控的图注意力网络，使节点状态可在级联迭代中递归更新；② 设计级联深度掩码过滤仅包含因果相关源节点的注意力；③ 通过跨网格训练让单一模型在结构完全不同的电网上实现零样本迁移。

**🔧 技术方法**

核心技术包括GRU‑门控图注意力网络（GAT）、级联深度掩码、基于级联曝光得分的脆弱性排序，以及自监督级联迭代标签重构。

**📊 数据集**

使用PyPSA‑EUR数据集：2013年三张欧洲电网（葡萄牙、德国、法国）作为训练集；2019年六张电网（葡萄牙、瑞士、英国、德国、西班牙、法国）作为评估集。

**📈 对比分析**

与静态基线（电学介数EB和PageRank PR）比较；CG‑CAE在所有评估电网上无论阈值（top‑τ%）都能获得更高的平均真实脆弱性，且在大约30个级联样本下即可稳定；在深度级联中优势更明显。

**⚠️ 局限性**

局限性包括：① 仅与静态基线对比，未与已针对每网格训练的监督模型比较；② 对极浅级联或极小网格时结果噪声较大；③ 训练仍需要级联仿真生成大量样本，虽无后续微调但仍存在模拟成本。

---

## 112. Hidden Coalitions in Multi-Agent AI: A Spectral Diagnostic from Internal Representations

**arXiv ID:** 2605.06696 | [PDF](https://arxiv.org/pdf/2605.06696v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 113. Topology-Enhanced Alignment for Large Language Models: Trajectory Topology Loss and Topological Preference Optimization

**arXiv ID:** 2605.07172 | [PDF](https://arxiv.org/pdf/2605.07172v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 114. AirBender: Adaptive Transportation of Bendable Objects Using Dual UAVs

**arXiv ID:** 2605.07003 | [PDF](https://arxiv.org/pdf/2605.07003v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 115. Leveraging fNIRS to Evaluate Workload for Adaptive Training in Virtual Reality

**arXiv ID:** 2605.06909 | [PDF](https://arxiv.org/pdf/2605.06909v1)

**作者:** Cara A. Spencer `[一作]` (University of Colorado Boulder), Leanne Hirshfield `[通讯]` (University of Colorado Boulder)

**通讯引用:** 2410 | [OpenAlex ID](https://openalex.org/A5009602263)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在虚拟现实环境中使用功能近红外光谱（fNIRS）测量认知负荷，验证内在负荷与外在负荷对大脑激活的差异，并探讨其在自适应训练中的应用。

**💡 创新点**

首次在高保真VR训练任务中同时采集并分析内在和外在负荷的fNIRS脑信号，揭示不同负荷对应的特定脑区激活模式，为基于神经信号的自适应训练提供诊断依据。

**🔧 技术方法**

采用fNIRS连续波光谱测量、NASA‑TLX主观负荷量表、行为指标（完成时间、错误率），并使用GLM、AR‑IRLS等预处理和统计方法分析脑血氧变化。

**📊 数据集**

36名受试者在VR形状组装任务中的fNIRS时间序列数据（80通道脑血氧信号、短分离通道及运动传感器），以及对应的行为和主观负荷记录。

**📈 对比分析**

通过二因素重复测量ANOVA和t检验比较高低负荷条件下的HbR/HbO激活；结果显示高内在负荷显著激活左侧前额叶和角回，外在负荷仅激活右角回，行为与主观负荷与fNIRS激活一致，验证方法有效。

**⚠️ 局限性**

样本量有限、个体差异未完全控制、fNIRS易受运动噪声影响、仅能测量皮层表层，难以捕捉更深层结构。

---

## 116. Uncovering and Shaping the Latent Representation of 3D Scene Topology in Vision-Language Models

**arXiv ID:** 2605.07148 | [PDF](https://arxiv.org/pdf/2605.07148v1)

**作者:** Haoming Wang `[一作]` (University of Pittsburgh), Wei Gao `[通讯]` (University of Pittsburgh)

**通讯引用:** 13513 | [OpenAlex ID](https://openalex.org/A5075969794)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

对VLM内部空间进行线性特征提取，分离3D拓扑表示并用Dirichlet能量正则化进行微调。

**💡 创新点**

通过跨场景平均线性分离空间子空间，证明其对应拉普拉斯特征，并提出单项Dirichlet正则化。

**🔧 技术方法**

线性投影、SVD、PCA、Dirichlet能量计算与LoRA微调。

**📊 数据集**

SynSpat3D合成数据，以及VSI-Bench和MindCube真实评测。

**📈 对比分析**

与普通LoRA、2D正则、文本域认知图三基准对比，单步500步微调后在VSI-Bench/MindCube 上提升约12% 以上。

**⚠️ 局限性**

仅关注静态拓扑，忽略时间动态与非欧几里得场景，过度正则化会导致性能退化。

---

## 117. TAS-LoRA: Transformer Architecture Search with Mixture-of-LoRA Experts

**arXiv ID:** 2605.07256 | [PDF](https://arxiv.org/pdf/2605.07256v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 118. Three-in-One World Model: Energy-Based Consistency, Prediction, and Counterfactual Inference for Marketing Intervention

**arXiv ID:** 2605.07199 | [PDF](https://arxiv.org/pdf/2605.07199v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 119. Zero-Shot Neural Network Evaluation with Sample-Wise Activation Patterns

**arXiv ID:** 2605.07378 | [PDF](https://arxiv.org/pdf/2605.07378v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 120. Traffic Scenario Orchestration from Language via Constraint Satisfaction

**arXiv ID:** 2605.06966 | [PDF](https://arxiv.org/pdf/2605.06966v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 121. MELD: Multi-Task Equilibrated Learning Detector for AI-Generated Text

**arXiv ID:** 2605.06903 | [PDF](https://arxiv.org/pdf/2605.06903v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 122. Sparse Attention as a Range Searching Problem: Towards an Inference-Efficient Index for KV Cache

**arXiv ID:** 2605.06763 | [PDF](https://arxiv.org/pdf/2605.06763v1)

**作者:** Mohsen Dehghankar `[一作]` (University of Illinois), Abolfazl Asudeh `[通讯]` (University of Illinois)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种名为 Louver 的 KV 缓存稀疏注意力索引，能够在保持零误检（zero false negatives）的前提下显著减少稀疏注意力的计算量。

**💡 创新点**

创新点在于将稀疏注意力建模为半空间范围搜索问题，并设计了轻量化、可动态阈值的索引结构。该索引兼顾了高剪枝率、动态适应性和与现有 LLM 推理流程的无缝集成，同时提供理论与实践层面的零误检保证。

**🔧 技术方法**

采用的核心技术包括：半空间范围搜索、PCA 归一化子空间划分、固定大小聚类与球形包围盒过滤、阈值算法（TA）以及 GPU/CPU 级别的并行核加速。实现上通过缓冲区批量更新与 GPU 并行流融合，进一步降低延迟。

**📊 数据集**

使用的数据集包括 LongBench、RULER、AIME 2024、MATH-500、以及多款开源 LLM（Llama-3.1-8B、DeepSeek-R1-Distill-Llama-8B、Qwen2.5-7B/14B）。

**📈 对比分析**

与固定预算方法（如 H_2O、StreamingLLM）、自适应预算方法（Twilight）、检索式 KV offloading 方法（RetrievalAttention、InfLLM、MagicPIG）以及 FlashAttention‑2 等密集注意力进行对比。结果显示，Louver 在长上下文和长推理任务中保持或超过密集注意力的准确率；GPU 端速度提升至 15.3×、CPU 端提升至 10.3×，在 KV offloading 场景下 F1 提升至 38.9%。

**⚠️ 局限性**

局限性包括：对极低 KV 保留比例时剪枝效果下降；阈值推断对性能影响显著，若阈值设置不当可能导致检索过多或过少；在极大模型或多 GPU 并行部署时仍需进一步评估和优化。

---

## 123. Bridging the Last Mile of Circuit Design: PostEDA-Bench, a Hierarchical Benchmark for PPA Convergence and DRC Fixing

**arXiv ID:** 2605.06936 | [PDF](https://arxiv.org/pdf/2605.06936v1)

**作者:** Pengju Liu `[一作]` (University of Minnesota), Caiwen Ding `[通讯]` (University of Minnesota)

**通讯引用:** 3243 | [OpenAlex ID](https://openalex.org/A5030060072)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PostEDA-Bench，一个包含 145 个任务的分层基准，用于评估 LLM 代理在 EDA 后期的 DRC 修复和 PPA 收敛。

**💡 创新点**

创新点在于：①同时覆盖 DRC 与 PPA 两大真实后端任务；②分层设计（DRC-Essential、DRC-Reasoning、PPA-Mono、PPA-Multi），可细粒度评估不同技能；③支持视觉输入、机器可验证评估，兼容开源与商业工具链。

**🔧 技术方法**

使用 LLM 代理框架（ReAct、Proposer–Critic、ORFS‑Agent），并结合视觉模态、迭代预算和思考模式（链式/树式）进行实验。

**📊 数据集**

数据集基于 3 个公开 RTL 源（RTLLM、VerilogEval‑Human、OpenCores），映射到 ASAP7，并生成合成与真实后流 DRC 及 PPA 任务。

**📈 对比分析**

对比 8 种商业与开源 LLM 在不同代理框架下的 SR/VRR/NIS。结果显示：DRC‑Essential 最佳 SR≈85%，DRC‑Reasoning 仅 36%，PPA‑Multi 仅 20%；视觉增强可显著提升 DRC 任务；PPA‑Multi 主要瓶颈是多目标权衡推理。

**⚠️ 局限性**

局限性包括：①从合成到真实任务的性能显著下降；②缺乏对几何规划与 RTL 结构重构的深入支持；③不同代理框架的优势并不统一，需进一步探索更通用的探索策略。

---

## 124. A Generalized Singular Value Theory for Neural Networks

**arXiv ID:** 2605.06938 | [PDF](https://arxiv.org/pdf/2605.06938v1)

**作者:** Brian Charles Brown `[一作]`, Sean Warnick `[通讯]` (Brigham Young University)

**通讯引用:** 1253 | [OpenAlex ID](https://openalex.org/A5010055271)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

基于广义奇异值分解（GSVD）理论，证明了大多数现代神经网络架构在最终线性层之前可以接受广义SVD表示，并且在输入输出行为上没有变化。

**💡 创新点**

提出了一种数据驱动的算法来估计这种表示，并提出了一种自然促进分解的模型架构。通过学习的表示可以识别对模型输入的对抗扰动，并为未来在模型偏差和可逆性等领域的应用开发了必要的理论。

**🔧 技术方法**

使用了广义奇异值分解（GSVD）理论，结合了数据驱动算法和模型架构设计。

**📊 数据集**

使用了MNIST和Fashion-MNIST数据集进行实验，验证了GSVD构造的有效性。

**📈 对比分析**

与传统的对抗攻击方法（如C&W L^2攻击）相比，基于GSVD的攻击在多个数据集上表现出更高的成功率，并且在某些情况下查询次数更少。

**⚠️ 局限性**

GSVD的构造在某些情况下可能会放宽线性SVD的性质，导致最大的奇异值不一定是严格的上界，并且在非凸情况下，最小化Σ的元素需要非凸优化，这对实际实现构成挑战。

---

## 125. On the Role of Strain and Vorticity in Numerical Integration Error for Flow Matching

**arXiv ID:** 2605.06680 | [PDF](https://arxiv.org/pdf/2605.06680v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 126. On Privacy Leakage in Tabular Diffusion Models: Influential Factors, Attacker Knowledge, and Metrics

**arXiv ID:** 2605.06835 | [PDF](https://arxiv.org/pdf/2605.06835v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 127. On Similarity of Computational Kernels in our Codes and Proxies

**arXiv ID:** 2605.06968 | [PDF](https://arxiv.org/pdf/2605.06968v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 128. A Closed-Form Upper Bound for Admissible Learning-Rate Steps in Belief-Space Dynamics

**arXiv ID:** 2605.06741 | [PDF](https://arxiv.org/pdf/2605.06741v1)

**作者:** Zixi Li `[一作]` (Datawhale), Youzhen Li `[通讯]` (Datawhale)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了在概率单纯形上使用KL/贝尔曼几何的局部收敛性分析，给出了交叉熵损失下的学习率上界；

**💡 创新点**

给出了闭式可计算的可接受步长上界，并结合熵门限实现自适应衰减；

**🔧 技术方法**

利用KL散度的三点恒等式、Bregman几何、收敛性证明与对数熵门限；

**📊 数据集**

主要在3分类的模拟信念追踪实验中验证；

**📈 对比分析**

与固定步长和ADS自适应步长进行对比，实验表明遵循上界可避免信念崩溃；

**⚠️ 局限性**

仅在单纯形内部局部近似，未考虑全局收敛、梯度噪声和参数空间映射等问题。

---

## 129. The Position Curse: LLMs Struggle to Locate the Last Few Items in a List

**arXiv ID:** 2605.07127 | [PDF](https://arxiv.org/pdf/2605.07127v1)

**作者:** Zhanqi Zhang `[一作]` (UC San Diego), Li Ji-An `[通讯]` (New York University)

**通讯引用:** 3392 | [OpenAlex ID](https://openalex.org/A5100402500)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并系统评估LLM在短列表中基于位置的检索能力，发现存在“位置诅咒”——后向检索显著弱于前向检索；构建PosBench训练集并通过LoRA微调提升该能力，进一步验证在PyIndex代码理解基准上的迁移效果。

**💡 创新点**

首次明确定义并量化“位置诅咒”，提出四维任务框架（检索方向、锚点类型、索引方向、项目类型）评估位置检索；设计PosBench与PyIndex数据集，展示LoRA微调可部分弥补缺陷并跨任务迁移。

**🔧 技术方法**

基于Transformer LLM的前后检索任务设计；使用LoRA低秩适配微调；在无思考与思考（Chain-of-Thought）两种条件下对模型进行评估；结合位置编码理论分析模型瓶颈。

**📊 数据集**

PosBench（包含字母、词、代码行序列）与PyIndex（Python列表索引表达式）两大数据集；使用BigCode、Open-Orca、OpenHermes等公开数据生成训练样本。

**📈 对比分析**

采用三-shot示例，评估Qwen3.5、Llama3.2、Gemma4、Ministral3等模型在前向/后向检索任务上的准确率；LoRA微调后，L=20字母序列前向检索从~80%提升至>90%，后向检索从~10%提升至~30%；在PyIndex基准上基线32%提升至70%（前向几乎满分），仍未达到饱和。

**⚠️ 局限性**

仅评估短列表位置检索，缺乏更广泛场景；LoRA提升有限，后向检索仍显弱；需要新的预训练数据或模型架构以彻底弥补“位置诅咒”。

---

## 130. Conservative Flows: A New Paradigm of Generative Models

**arXiv ID:** 2605.06905 | [PDF](https://arxiv.org/pdf/2605.06905v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 131. The Single-File Test: A Longitudinal Public-Interface Evaluation of First-Output LLM Web Generation with Social Reach Tracking

**arXiv ID:** 2605.06707 | [PDF](https://arxiv.org/pdf/2605.06707v1)

**作者:** Diego Cabezas Palacios `[一作]` `[通讯]`, Diego Cabezas Palacios

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在公开聊天界面下，观察性比较四大LLM（GPT、Gemini、Grok、Claude）生成单文件HTML的表现，涵盖代码质量、延迟与社交媒体曝光；

**💡 创新点**

创新点在于采用严格的一次输出协议、将实验结果以可视化视频+AI歌曲形式发布至X/TikTok/YouTube，并结合LLM-as-a-judge与社交影响的监督预测；

**🔧 技术方法**

主要技术包括自然语言提示生成HTML、视频录制评估、加权性能评分、推理效率度量、Gemini视频评测以及Lasso‑Ridge回归预测X曝光和HTML行数；

**📊 数据集**

使用了自建的“HTML AI Battle”数据集，包含68个单文件HTML输出、17个实验记录、评分、时序、视频、社交指标等；

**📈 对比分析**

比较方法通过人工评分与Gemini评测三维指标，并统计平均性能、推理效率、模型占优次数；Claude在平均得分、稳定性和占优次数上领先；推理时间与质量无正相关，Gemini评测更宽松；

**⚠️ 局限性**

局限性包括样本量小（仅17个实验）、公共接口版本漂移、单一人工评测者、未完全随机化、以及社交曝光受账号历史和平台因素影响

---

## 132. Less Random, More Private: What is the Optimal Subsampling Scheme for DP-SGD?

**arXiv ID:** 2605.07072 | [PDF](https://arxiv.org/pdf/2605.07072v1)

**作者:** Andy Dong `[一作]` (Stanford University), Ayfer Özgür `[通讯]` (Stanford University)

**通讯引用:** 1690 | [OpenAlex ID](https://openalex.org/A5003322209)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

研究了差分隐私随机梯度下降（DP‑SGD）中采样方案的最优性，提出了 Balanced Iteration Subsampling（BIS）并实现了一种近似精确的 Monte Carlo 会计方法。

**💡 创新点**

创新点包括：① 在低噪声和高噪声极限下证明 BIS 是独立样本采样族中的最优方案，揭示参与方差是主导隐私放大因素；② 设计了屏蔽+动态规划的“screen‑then‑exact” Monte Carlo 会计，消除 RDP 与 PLD 组合分析的松弛；③ 在实验中显示 BIS 在低噪声、高效能场景下可将噪声乘子降低最高 9.6%。

**🔧 技术方法**

采用的技术手段包括 Rényi 差分隐私（RDP）、隐私损失分布（PLD）、Estimate‑Verify‑Release（EVR）框架、基于动态规划的精确似然比计算、O(T) 上界筛选、以及大规模并行 Monte Carlo 采样。

**📊 数据集**

未使用具体公开数据集，而是基于多种真实 DP‑SGD 配置（T、k、ε、δ 等）在 Stanford Sherlock 计算集群上进行实验。

**📈 对比分析**

与 Poisson、BIS‑RDP、RA‑PLD 等方法对比，BIS‑MC 在低噪声时可显著降低噪声倍数（最多 9.6%），在高噪声时与 Poisson 收敛。实验表明 BIS‑MC 在大多数配置下保持或超过 Poisson 的隐私效能，且计算时间从数分钟到数小时不等。

**⚠️ 局限性**

局限性：① 需要大量 Monte Carlo 采样，计算成本仍较高（12 分钟–12 小时）；② 证明仅覆盖独立样本采样族，对更一般的采样策略未给出最优结论；③ 在高噪声极限下优势几乎消失；④ 对实际大规模模型训练的实测效果尚待进一步验证。

---

## 133. How Big Should a Wireless Foundation Model Be?

**arXiv ID:** 2605.07266 | [PDF](https://arxiv.org/pdf/2605.07266v1)

**作者:** Wei-Lun Cheng `[一作]` (National Taiwan University), Wanjiun Liao `[通讯]` (National Taiwan University)

**通讯引用:** 4456 | [OpenAlex ID](https://openalex.org/A5062720793)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究无线基础模型的规模上限，证明通道的非线性内在维数决定模型容量，并提出基于标准OFDM导频的测试时训练（TTT）来在推理时自适应。

**💡 创新点**

创新点在于：①使用非线性内在维数（d_NL）量化无线通道的有效自由度；②给出经验缩放法则，说明无线模型快速饱和；③提出pilot‑aided TTT，让小模型在部署时即可实现显著性能提升。

**🔧 技术方法**

主要技术包括非线性内在维数估计（Two‑NN、MLE）、遮掩自编码器网络、梯度更新的TTT、跨域迁移实验以及基准性能评估（NMSE、MCM等）。

**📊 数据集**

使用的数据集涵盖：NTN卫星仿真、3GPP CDL-A～E、KU Leuven室内大规模MIMO OTA、DeepMIMO毫米波、DICHASUS室内Ka波测量等真实与仿真混合。

**📈 对比分析**

通过与大型静态模型（85M/96M）和传统LS+插值方法比较，发现12M+TTT在SNR>10dB可匹敌或超越96M模型，且推理成本下降约9倍；TTT在SNR=20dB提升NMSE达7.2dB，低SNR下传统方法更稳健。

**⚠️ 局限性**

局限性包括：TTT需要梯度计算和额外算力，受硬件支持限制；适用于场景变化慢的通道，无法在高速衰落下即时跟踪；跨域迁移仍需大量导频；目前验证集中在通道估计，其他下游任务仍待研究。

---

## 134. PRISM: Refracting the Entangled User Behavior Space for E-Commerce Search

**arXiv ID:** 2605.07296 | [PDF](https://arxiv.org/pdf/2605.07296v1)

**作者:** Haoqian Zhang `[一作]` (Sichuan University), Yi Zhang `[通讯]` (Sichuan University)

**通讯引用:** 97825 | [OpenAlex ID](https://openalex.org/A5100388089)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 PRISM 框架，用于在电商搜索中联合建模用户偏好与物品相关性，以提高行为预测和排名效果。

**💡 创新点**

创新点在于三大模块：①偏好纠正（Preference Rectification）抑制相关性干扰；②LLM 驱动的语义锚定（Semantic Anchoring）利用大语言模型产生正负原型校准相关性向量；③偏好条件的证据路由（Evidence Routing）在修正后的偏好状态下自适应聚合多源证据。

**🔧 技术方法**

技术包括注意力交互、正交投影、语义原型对齐、路由注意力、可学习交互函数、BCE 目标与多项辅助损失（方向、幅度、原型对齐、路由熵）等。

**📊 数据集**

实验使用 KuaiSAR（短视频搜索日志）和 JDSearch（京东电商搜索）两个公开数据集。

**📈 对比分析**

与多种基线（DSSM、QEM、HEM、CLK、NISE、DCMT、PRINT、DRP 等）及多种背骨组合比较，PRISM 在 AUC、LogLoss、NDCG@10、HR@10 上均显著优于所有对手，且在热点与长尾子集上表现尤为突出。

**⚠️ 局限性**

局限在于仍需在更复杂多行为场景下验证，概率校准与多任务适配待进一步改进。

---

## 135. A Unified Measure-Theoretic View of Diffusion, Score-Based, and Flow Matching Generative Models

**arXiv ID:** 2605.06829 | [PDF](https://arxiv.org/pdf/2605.06829v1)

**作者:** Aditya Ranganath `[一作]` (Lawrence Livermore National Laboratory), Mukesh Singhal `[通讯]` (University of California Merced)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了连续时间生成模型的统一框架，阐明了扩散模型、基于分数的模型、概率流ODE与流匹配在概率传输中的对应关系；

**💡 创新点**

创新点在于把这些方法统一到学习时间相关向量场的通用视角，明确了路径、学习对象、采样动态与目标的对应关系，并系统比较了它们的误差来源和理论局限；

**🔧 技术方法**

使用的技术包括随机微分方程、连续归一化流、Fokker–Planck方程、概率流ODE、流匹配回归、分数匹配、SDE/ODE数值求解器、Schrödinger桥与最优传输等；

**📊 数据集**

该论文为综述性质，并未使用特定实验数据集；

**📈 对比分析**

通过统一符号和理论推导，比较了各方法在采样效率、数值稳定性、可解释性以及对条件生成的适配性；实验结果表明，尽管所有方法在理论上可生成相同边缘分布，但在采样步骤、数值误差和对逆问题的鲁棒性上存在显著差异；

**⚠️ 局限性**

局限在于缺乏统一的实验评估，理论分析仍停留在高层抽象，未能给出具体的路径设计准则、学习与求解器协同误差定量界定，且对非高斯、非欧氏空间的推广仍有待深入研究。

---

## 136. AT-VLA: Adaptive Tactile Injection for Enhanced Feedback Reaction in Vision-Language-Action Models

**arXiv ID:** 2605.07308 | [PDF](https://arxiv.org/pdf/2605.07308v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 137. TTF: Temporal Token Fusion for Efficient Video-Language Model

**arXiv ID:** 2605.07355 | [PDF](https://arxiv.org/pdf/2605.07355v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 138. ProtoSSL: Interpretable Prototype Learning from Unlabeled Time-Series Data

**arXiv ID:** 2605.06943 | [PDF](https://arxiv.org/pdf/2605.06943v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 139. Revisiting Adam for Streaming Reinforcement Learning

**arXiv ID:** 2605.06764 | [PDF](https://arxiv.org/pdf/2605.06764v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 140. Adaptive auditing of AI systems with anytime-valid guarantees

**arXiv ID:** 2605.07002 | [PDF](https://arxiv.org/pdf/2605.07002v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 141. Learning Visual Feature-Based World Models via Residual Latent Action

**arXiv ID:** 2605.07079 | [PDF](https://arxiv.org/pdf/2605.07079v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 142. Enabling Unsupervised Training of Deep EEG Denoisers With Intelligent Partitioning

**arXiv ID:** 2605.06724 | [PDF](https://arxiv.org/pdf/2605.06724v1)

**作者:** Qiyu Rao `[一作]` (Imperial College London), Danilo Mandic `[通讯]` (Imperial College London)

**通讯引用:** 24915 | [OpenAlex ID](https://openalex.org/A5103001848)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于智能划分的自监督EEG去噪方法 iPSD，可在无清晰参考信号的情况下恢复高质量脑电。

**💡 创新点**

创新点在于学习一个划分模块，将单一噪声EEG分成共享相同干净信号但噪声互相独立的两条子信号，从而实现 Noise2Noise 样式的自监督训练，并推出零样本版本 iPSD-Zero。

**🔧 技术方法**

核心技术包括使用强化学习（PPO）优化划分策略、卷积神经网络做去噪、以及对称正则化保证划分后与去噪前的可逆性。

**📊 数据集**

使用 CHB‑MIT 头皮EEG 数据生成的合成噪声数据（WGN、EMG）和真实可穿戴耳内EEG 数据（37 位老年人共 684 小时）。

**📈 对比分析**

与多种基线（wavelet、VMD、EMD、WPT‑ICA、Optimal‑WT 等）比较，iPSD 在 SNR、PSNR 和谱均方误差上均优于基线，最大可提升 3.3 dB SNR、谱 MSE 降至 <60，并使睡眠阶段分类准确率提升至 87%（接近临床头皮EEG 级别）。

**⚠️ 局限性**

局限在于假设噪声在两条子信号中相互独立且与信号无关，若噪声呈相关或非独立特性，性能可能受限。

---

## 143. Mean-Pooled Cosine Similarity is Not Length-Invariant: Theory and Cross-Domain Evidence for a Length-Invariant Alternative

**arXiv ID:** 2605.07345 | [PDF](https://arxiv.org/pdf/2605.07345v1)

**作者:** Sibayan Mitra `[一作]` (Birla Institute of Technology and Science), Dhruv Kumar `[通讯]` (Birla Institute of Technology and Science)

**通讯引用:** 6802 | [OpenAlex ID](https://openalex.org/A5027859418)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

分析并实证证明均值池化余弦相似度在变长输入时会产生长度偏差，揭示其在代码、自然语言和视觉任务中的误导性。

**💡 创新点**

提出使用长度不变的线性CKA（以及RV系数）作为替代指标，给出理论推导并跨领域验证，揭示之前的跨语言“近似”结论往往是长度偏差造成的。

**🔧 技术方法**

采用均值池化、1/√n噪声收缩理论、线性CKA、RV系数、回归分析等技术手段。

**📊 数据集**

使用 HumanEvalPack（代码）、WMT14/16（自然语言）、CLIP synthetic captions（视觉）等公开数据集。

**📈 对比分析**

对比均值池化余弦与CKA的结果显示，CKA将长度解释的方差从 52–75% 降至 <1%，并使长度系数符号逆转，说明真实跨语言收敛仍存在但无长度偏差。

**⚠️ 局限性**

局限性包括仅使用线性CKA（对齐与尺度可能产生偏差）、对模型覆盖有限、未验证非线性CKA或其他长度不变指标、以及对真实图像/字幕对的实验不足。

---

## 144. A New Interaction Concept for Interactive and Autoactive Program Verification

**arXiv ID:** 2605.06972 | [PDF](https://arxiv.org/pdf/2605.06972v1)

**作者:** Wolfram Pfeifer `[一作]` (Karlsruhe Institute of Technology), Daniel Drodt `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种新的交互式与自动化验证相结合的交互概念，在源代码层面展示和操作证明状态。

**💡 创新点**

创新点在于将证明目标以 JML 的假设和断言形式直接插入源代码，追踪公式来源并在逻辑层与代码层之间实现自动映射，从而大幅降低验证者与证明者之间的认知鸿沟。

**🔧 技术方法**

使用的技术包括 KeY 验证器、JavaDL 逻辑、JML 规范语言、程序状态追踪与反向翻译算法、以及插件化的 GUI 交互层。

**📊 数据集**

验证所用数据集为两份约 40 行 Java 程序（凯撒密码实现）及其 50 行 JML 规范，实验任务是定位并修复源代码或规范中的小缺陷。

**📈 对比分析**

比较方法为用户研究：六名经验丰富的验证专家分别在新视图和传统序列视图下完成相同任务，并记录时间与反馈。新视图平均完成时间略低（9.8 分钟 vs 11.0 分钟），标准差略大；整体表明新交互方式可提升理解与修复效率。

**⚠️ 局限性**

主要局限包括：只实现了单一循环规则；无法在 JML 中表示 assignable 子句和方法/循环的 framing 子句；当公式与原始输入相差过大或缺失来源信息时，需采用最佳尝试翻译，功能尚不完整；并且尚未实现证明脚本的自动记录与重放。

---

## 145. EULER-ADAS: Energy-Efficient & SIMD-Unified Logarithmic-Posit Engine for Precision-Reconfigurable Approximate ADAS Acceleration

**arXiv ID:** 2605.06875 | [PDF](https://arxiv.org/pdf/2605.06875v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 146. A Simple Method for School Choice Lotteries

**arXiv ID:** 2605.06721 | [PDF](https://arxiv.org/pdf/2605.06721v1)

**作者:** Yasunori Okumura `[一作]` (Tokyo University of Marine Science and Technology), Yasunori Okumura `[通讯]` (Tokyo University of Marine Science and Technology)

**通讯引用:** 126 | [OpenAlex ID](https://openalex.org/A5081675055)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种多项式时间的简单方法，用以构造满足等价学生均等对待且在所有 ex ante 稳定彩票中不被支配的学校分配方案。

**💡 创新点**

创新点在于将等价学生均等对待（ETE）的再分配操作直接应用于已得到的受限高效稳定匹配，从而避免了原先复杂算法的循环与终止问题。

**🔧 技术方法**

主要技术包括：稳定匹配理论、Pareto 效率与约束效率概念、以及 ETE 重新分配的概率重排方法。

**📊 数据集**

论文不使用实证数据，而是通过理论构造与数学证明完成。

**📈 对比分析**

通过对比已有的 FDA、FDAT 等机制，证明该方法在多项式时间内实现 ex ante 稳定且不被任何其他 ex ante 稳定彩票支配；对比结果主要以证明可行性与非支配性为准。

**⚠️ 局限性**

局限在于：尽管得到的彩票在 ex ante 稳定下不被支配，但并不能保证其在 ex post 稳定下也不被支配；此外，理论假设下的等价学生群组划分可能在实际场景中较难满足。

---

## 147. From Model to Data (M2D): Shifting Complexity from GNNs to Graphs for Transparent Graph Learning

**arXiv ID:** 2605.06814 | [PDF](https://arxiv.org/pdf/2605.06814v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 148. Neurosymbolic Imitation Learning with Human Guidance: A Privileged Information Approach

**arXiv ID:** 2605.07166 | [PDF](https://arxiv.org/pdf/2605.07166v1)

**作者:** Nikhilesh Prabhakar `[一作]`, Sriraam Natarajan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了神经符号模仿学习框架GRAIL，利用人类视线等特权信息对Atari游戏进行学习与决策

**💡 创新点**

创新点在于将人类眼动预测器与模糊逻辑规则前向链路结合，实现可解释的动作策略，并通过最小化规则复杂度提升泛化性

**🔧 技术方法**

核心技术包括OCAtari对象抽象、神经谓词的模糊原子化、Łukasiewicz模糊逻辑、T=2步可微前向链路、HumanGazeNet眼动编码及权重重标定

**📊 数据集**

使用的公开Atari数据集为Seaquest（49物体）、Asterix（25物体）与Freeway（12物体），每个环境提供数十万帧样本

**📈 对比分析**

与BC、AGIL、BC+Mask等纯神经基线相比，GRAIL在Seaquest和Asterix上取得更高平均分，且学习到的规则权重高度可解释，能够展示对氧气管理或躲避敌人的策略优先级

**⚠️ 局限性**

局限性主要在于规则集需人工或经验化设计，过于复杂的规则会影响可解释性与可维护性，同时当前框架对新游戏的迁移仍依赖较多规则工程

---

## 149. Zombies in Alternate Realities: The Afterlife of Domain Names in DNS Integrations

**arXiv ID:** 2605.06880 | [PDF](https://arxiv.org/pdf/2605.06880v1)

**作者:** Sulyab Thottungal Valapu `[一作]` (USC), Raffaele Sommese `[通讯]` (University of Twente)

**通讯引用:** 289 | [OpenAlex ID](https://openalex.org/A5002961985)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对DNS与其他生态系统整合导致的僵尸链接问题进行系统性研究，定义威胁模型，测量Web PKI、ENS、Maven Central中僵尸链接的比例、持续时间，并检索已被利用的攻击实例。

**💡 创新点**

提出统一的DNS整合僵尸链接威胁模型，量化不同生态系统的僵尸比例与生命周期，识别已被利用的攻击，并给出针对设计的改进建议。

**🔧 技术方法**

结合域名注册历史推断、CT日志、区块链事件、DNS查询和Maven镜像抓取，使用算法推断DNS拥有权区间；统计分析与可视化；利用证书吊销、DNSSEC等机制。

**📊 数据集**

RDAP查询、dns.coffee区块委托数据、CT日志（OpenINTEL）、Etherscan区块链交易日志、Maven Central镜像元数据、DNS TXT记录扫描、TLS服务器响应等。

**📈 对比分析**

通过计算僵尸比例、持续时间分布、攻击可利用率，使用时间序列与分位数分析；发现Web PKI僵尸约2.7%，ENS On‑chain 24%，Maven 15%；僵尸寿命在Web PKI为90天，ENS On‑chain与Maven为多年；性能主要体现在数据收集周期为数月到数年。

**⚠️ 局限性**

数据覆盖有限（RDAP不全、zone文件缺失、扫描间隔1天），只能推断注册区间而非所有所有权变更，僵尸识别阈值80天可能漏检，攻击确认缺乏直接证据，且无法覆盖全部域名/生态系统。

---

## 150. TUANDROMD-X: Advanced Entropy and Visual Analytics Dataset for Enhanced Malware Detection and Classification

**arXiv ID:** 2605.06718 | [PDF](https://arxiv.org/pdf/2605.06718v1)

**作者:** Parthajit Borah `[一作]` (Tezpur University), J. K. Kalita `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个包含30,000条Android恶意与良好软件样本的双模态数据集（熵图与灰度图），并在该数据集上训练CNN进行分类。

**💡 创新点**

首次提供同时包含熵分析和灰度图表示的公开数据集，并展示熵图在恶意软件检测中优于传统灰度图的效果。

**🔧 技术方法**

使用了静态分析（熵滑动窗口、灰度映射）、Python、PIL、卷积神经网络（ResNet、DenseNet）等技术。

**📊 数据集**

TUANDROMD-X数据集：20,000条恶意软件样本（71类）+ 10,000条良好软件样本（1类）。

**📈 对比分析**

通过将数据分为70%训练、10%验证、20%测试，比较多种CNN模型的准确率、精确率、召回率和F1分数；熵图模型在整体上达85–88%准确率，灰度图模型约76–80%，表明熵表示更具区分度。

**⚠️ 局限性**

数据集存在类别不平衡，尤其是少数恶意类别与良好软件的样本不足，且不做增强，限制了模型泛化能力。

---

## 151. Can Agents Price a Reaction? Evaluating LLMs on Chemical Cost Reasoning

**arXiv ID:** 2605.07251 | [PDF](https://arxiv.org/pdf/2605.07251v1)

**作者:** Yuyang Wu `[一作]` (Carnegie Mellon University), Olexandr Isayev `[通讯]` (Carnegie Mellon University)

**通讯引用:** 18530 | [OpenAlex ID](https://openalex.org/A5011932992)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了ChemCost基准，评估LLM工具使用代理在化学采购成本估算任务中的表现。

**💡 创新点**

创新点在于提供了可判定的采购成本标签、四阶段噪声注入、以及基于工具调用轨迹的阶段级诊断评估。

**🔧 技术方法**

采用ReAct框架、四种确定性工具（命名解析、报价检索、分子量计算、算术运算），并对多种前沿、开源与化学专门化LLM进行评测。

**📊 数据集**

使用来自Open Reaction Database、教材、Organic Syntheses、PaRoutes、ChemPU等共1427条反应，和230,775个供应商报价的数据集。

**📈 对比分析**

与人类参考和多种模型比较，最好的模型在干净输入下CTA@25仅达50.6%，在噪声条件下准确率显著下降，显示出工具调用不足或不稳定。

**⚠️ 局限性**

主要限制是对格式噪声的鲁棒性差、对多步路线的性能急剧下降，以及工具交互的稳定性和解释性不足。

---

## 152. Connectivity Oracle Under Vertex Failures by Shortcutting Unbreakable Decomposition

**arXiv ID:** 2605.07168 | [PDF](https://arxiv.org/pdf/2605.07168v1)

**作者:** Xizhe Li `[一作]` (University of Michigan), Benyu Wang `[通讯]` (University of Michigan)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种在顶点失效情况下的连通性预言机，更新时间仅依赖失效数k，并且查询时间为最优O(k)。

**💡 创新点**

创新点在于通过树分解上的快捷路将空间从二次下降到近线性；利用n相关预言机进行自举以降低预处理时间；以及引入补丁集机制实现O(k)查询。

**🔧 技术方法**

核心技术包括不可分解（unbreakable）分解框架、树快捷路构造、限制袋图、张量/扭矩（torsos）与配置图（profiles）以及多级补丁集维护。

**📊 数据集**

本文为理论工作，未使用具体实验数据集，所有结果均为算法复杂度分析。

**📈 对比分析**

与现有预言机相比，本文在空间和预处理时间上实现近线性，更新时间完全独立于n；查询时间达到O(k)的条件最优，显著优于之前需要Ω(n²)空间或2^{2^{O(k)}}时间的预言机。

**⚠️ 局限性**

主要局限是更新时间仍为O(k⁶)，并且空间、预处理时间含有2^{O(k²)}的指数因子；α_c(n)因子仍存在，且未能达到O(k²)的更新时间。

---

## 153. Analyzing the Adoption of Database Management Systems Throughout the History of Open Source Projects

**arXiv ID:** 2605.06817 | [PDF](https://arxiv.org/pdf/2605.06817v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 154. Sparse Random-Feature Neural Networks with Krylov-Based SVD for Singularly Perturbed ODE

**arXiv ID:** 2605.07286 | [PDF](https://arxiv.org/pdf/2605.07286v1)

**作者:** Kevin Kurian Thomas Vaidyan `[一作]` (University of British Columbia), Siddharth Rout `[通讯]` (University of British Columbia)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5031763340)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种稀疏化的随机特征神经网络（PIELM）框架，通过高斯编码提升激活矩阵秩，并利用稀疏奇异值分解（sSVD）求解最小二乘问题；

**💡 创新点**

创新点在于将结构化稀疏性引入隐藏层激活，采用高斯编码提高秩并保持稀疏；利用Golub‑Kahan双边分解与完整正交化实现高效稳定的稀疏SVD；

**🔧 技术方法**

使用高斯编码（Gaussian kernel）、Golub‑Kahan bidiagonalization、Lanczos迭代、稀疏SVD、LSQR/LSMR对比、Gram‑Schmidt正交化等技术；

**📊 数据集**

主要数据集为一维稳态对流扩散方程（Pe=-10^4）以及一个真实的极度病态稀疏矩阵Xingo3012（20,944×20,944），并使用随机生成的高斯稀疏矩阵做验证；

**📈 对比分析**

与传统密集RFNN/PIELM相比，稀疏方法在相同或更高精度下显著降低了内存占用和训练时间；在Pe=10^4、10,000节点下成功求解，而传统方法仅能处理Pe≤10^3；

**⚠️ 局限性**

局限在于稀疏激活矩阵仍可能低秩导致表达能力受限；正交化过程成本高，稀疏SVD需要较多迭代；在高秩/高精度需求下，稀疏矩阵最终会变为稠密，导致内存压力。

---

## 155. Exploring the "Banality" of Deception in Generative AI

**arXiv ID:** 2605.07012 | [PDF](https://arxiv.org/pdf/2605.07012v1)

**作者:** Ishitaa Narwane `[一作]` (Maastricht University), Konrad Kollnig `[通讯]` (Maastricht University)

**通讯引用:** 248 | [OpenAlex ID](https://openalex.org/A5059625515)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了生成式 AI（如聊天机器人）中的欺骗设计，聚焦“平庸欺骗”概念，分析其如何通过无形、日常交互方式影响用户并将用户视为共制者；

**💡 创新点**

创新点在于将“平庸欺骗”视角应用于生成式 AI，强调用户的主动参与与共创角色，并提出以意识提升、干预工具和监管机制来增强用户自主性的思路；

**🔧 技术方法**

主要采用理论框架与案例分析方法（如Raine v OpenAI案）阐述观点，并未引入新的实验技术；

**📊 数据集**

未使用机器学习数据集，主要参考文献、法律案例和先前关于暗模式的研究；

**📈 对比分析**

本文并未进行实验比较或性能评估，而是以概念性讨论和案例说明为主；

**⚠️ 局限性**

局限性包括缺乏实证验证、对不同用户群体与应用场景的普适性未知，以及对监管与技术实现细节的细化不足。

---

## 156. Adaptive Negative Reinforcement for LLM Reasoning:Dynamically Balancing Correction and Diversity in RLVR

**arXiv ID:** 2605.07137 | [PDF](https://arxiv.org/pdf/2605.07137v1)

**作者:** Yash Ingle `[一作]` (Sardar Vallabhbhai National Institute of Technology), Sudhakar Mishra `[通讯]` (Sardar Vallabhbhai National Institute of Technology)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5122770754)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 RLVR 框架下提出了两种负样本强化策略：Adaptive Negative Sample Reinforcement (A-NSR) 与 Confidence-Weighted Negative Reinforcement (CW-NSR)，用于动态调节负奖励的强度和样本权重。

**💡 创新点**

创新点在于：①引入时间调度函数实现训练进度的自适应负奖励；②基于模型生成序列的置信度给错误样本分配不同权重，从而实现更细粒度的负样本挖掘。

**🔧 技术方法**

技术方法包括：RLVR 的 PSR–NSR 分解、指数/余弦/性能驱动的负奖励调度、基于序列几何均值的置信度评估、token‑level 梯度分析，以及使用 TRL‑PPO 对 Qwen2.5‑Math‑1.5B 进行 fine‑tuning。

**📊 数据集**

实验数据集为数学推理集 MATH、AIME 2025 以及 AMC23，并在 Qwen2.5‑Math‑1.5B 模型上进行训练与评估。

**📈 对比分析**

与固定权重的 W‑REINFORCE 基线相比，A‑NSR 在低采样预算（Pass@1–32）上显著提升；CW‑NSR 在中高采样预算（Pass@32–256）下对 AIME 与 AMC23 提升更明显；但在 MATH 高采样预算（Pass@128–256）时，W‑REINFORCE 仍保持领先。

**⚠️ 局限性**

局限性包括：训练在长周期下的稳定性尚未充分验证；置信度仅按序列级估计，无法捕捉 token‑级错误；方法目前针对稀疏可验证奖励，难以直接推广至密集或过程‑级奖励场景；实验仅限于数学推理任务，未验证跨领域效果。

---

## 157. Advancing Reliable Synthetic Video Detection: Insights from the SAFE Challenge

**arXiv ID:** 2605.06912 | [PDF](https://arxiv.org/pdf/2605.06912v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 158. Don't Learn the Shape: Forecasting Periodic Time Series by Rank-1 Decomposition

**arXiv ID:** 2605.07222 | [PDF](https://arxiv.org/pdf/2605.07222v1)

**作者:** Takato Honda `[一作]` `[通讯]` (Mellon Inc), Takato Honda (Mellon Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了基于周期序列近似为秩‑1矩阵的思路，提出一种只学习水平（Level）而冻结形状（Shape）的周期预测方法FLAIR，并在多周期多配置基准上实现了可比的性能；

**💡 创新点**

创新点在于①证明周期序列可近似秩‑1，②负实验表明学习形状无益，③设计了闭式先验中心岭回归预测水平并与固定形状相乘；

**🔧 技术方法**

主要技术包括周期重塑、SVD降秩、BIC 选周期、Box‑Cox 变换、先验中心岭回归、GCV 加权平均、以及基于训练窗口的自适应回退；

**📊 数据集**

使用了 GIFT‑Eval（97 个配置、23 个数据集，时间粒度 5 min–年）和 Chronos 0‑shot 25 个数据集；

**📈 对比分析**

与多基线（PatchTST、DLinear、AutoARIMA 等）对比，FLAIR 在 GIFT‑Eval 的 relMASE 为 0.838，几乎等同 PatchTST，但参数仅 P＋p＋1（小时 28 个标量），在 CPU 单核上仅 22 min ；

**⚠️ 局限性**

局限在于当周期能量 r₁ 低于约 0.7、周期数 n_c 较少或形状随周期漂移时失效，需要使用 Seasonal Naive 或 STL 等回退策略。

---

## 159. ImplantMamba: Long-range Sequential Modeling Mamba For Dental Implant Position Prediction

**arXiv ID:** 2605.07082 | [PDF](https://arxiv.org/pdf/2605.07082v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 160. Uneven Evolution of Cognition Across Generations of Generative AI Models

**arXiv ID:** 2605.06815 | [PDF](https://arxiv.org/pdf/2605.06815v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 161. Agentick: A Unified Benchmark for General Sequential Decision-Making Agents

**arXiv ID:** 2605.06869 | [PDF](https://arxiv.org/pdf/2605.06869v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 162. An audio-to-analysis pipeline with certified transcription for information-theoretic profiling of the piano repertoire

**arXiv ID:** 2605.06685 | [PDF](https://arxiv.org/pdf/2605.06685v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 163. TeamBench: Evaluating Agent Coordination under Enforced Role Separation

**arXiv ID:** 2605.07073 | [PDF](https://arxiv.org/pdf/2605.07073v1)

**作者:** Yubin Kim `[一作]` (MIT), Daniel McDuff `[通讯]` (Google Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 TeamBench 基准，利用操作系统沙箱强制实现 Planner、Executor、Verifier 三个角色的访问权限分离，并在此框架下评估多智能体协作的真实效果。

**💡 创新点**

创新点在于：①将角色分离从提示层级提升到系统层级，避免模型角色“崩溃”；②定义团队价值指标 TNI、规划价值与验证价值，量化不同角色的边际贡献；③构建跨提供者（Anthropic、Google、OpenAI）角色混合实验，揭示成本-性能 Pareto 前沿；④结合 40 组人类实验，验证评测指标与人类协作行为的一致性。

**🔧 技术方法**

技术手段包括：容器化沙箱隔离、文件权限控制、工具调用循环、LLM 角色化（Planner、Executor、Verifier）、确定性脚本评分器、Bootstrap 置信区间、McNemar 等统计检验，以及基于日志的角色违规率计算。

**📊 数据集**

使用的数据集为 851 个任务模板（931 个实例），涵盖 19 类任务（安全修补、数据流水线、分布式调试等），任务来源包括 650 条 GitHub bug 报告、30 条数据科学公开数据任务、10 条事件响应案例，所有实例均可通过种子生成器重现。

**📈 对比分析**

对比方法：在 90 任务子集上对 Solo、Restricted、Full Team、Team‑No‑Plan、Team‑No‑Verify 等设置进行配对 bootstrap 评估；在 25 任务子集上做跨提供者 27 组角色混合、prompt‑only vs enforced 的 McNemar 检验；结果显示：团队在 Solo 得分低的 20% 任务中提升约 15–20 点，但在高 Solo 得分任务中会下降 6–10 点；总体团队 pass 率约 40%，Verifier 误接受率高达 49%。

**⚠️ 局限性**

局限性：Verifier 误接受率高，导致团队价值评估不稳定；实验仅覆盖单轮文件交互，未测试多轮对话与动态角色分配；未深入探讨大模型规模对角色分离的影响；评测仅依赖 deterministic grader，缺少更细粒度的错误分析与人类审计；因此结果在更复杂、开放式任务或多轮协作环境下可能不具代表性。

---

## 164. Bifurcation Models: Learning Set-Valued Solution Maps with Weight-Tied Dynamics

**arXiv ID:** 2605.07277 | [PDF](https://arxiv.org/pdf/2605.07277v1)

**作者:** Caleb Jore `[一作]` (University of Central Florida), Jialin Liu `[通讯]` (University of Central Florida)

**通讯引用:** 365 | [OpenAlex ID](https://openalex.org/A5069045947)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了通过权重共享的动态网络（bifurcation model）学习多解问题，将传统的单一标签学习转化为学习集合值映射，并证明其能以正则、可学习的方式表示广泛的集合值函数。

**💡 创新点**

创新点包括：①证明局部 Lipschitz 分支可由全局输入 Lipschitz 的权重共享动态实现；②展示由动态产生的选择器几乎处处光滑，而人工选取的单解选择器可能极不连续；③在无完整分支标签的情况下，通过能量最小化自发发现多解；④在 Allen–Cahn 实验中发现多解不是自动的，需要显式多样性正则化，揭示准确性与多样性之间的权衡。

**🔧 技术方法**

采用权重共享的神经动力学（如 GNN 与 FNO 作为迭代映射），无监督能量损失、分支匹配损失以及多样性正则化（对终点能量与解间差异的平均平方和），并使用理论证明与数值实验相结合。

**📊 数据集**

实验数据集主要包括：
- 带有几何冲突的反铁磁 Ising 图（随机加权图，节点数从数十到数万），
- Allen–Cahn PDE 的随机外迫场（二维周期域上的连续函数）。

**📈 对比分析**

与传统的单标签监督（如 Gurobi 选取的解或 IMEX 预测）以及无监督能量训练的比较：
  • 在 Ising 任务中，能量训练的循环网络在保持低能量的同时发现平均约 13–14 个不同解；
  • 在 Allen–Cahn 任务中，单标签监督导致单一解且物理残差高，而仅能量训练得到最低残差但仅发现 1 个解；
  • 引入多样性正则化后，解的多样性显著提升（从 1 到 18+），但残差和能量均有所下降，呈现明确的准确性–多样性权衡。

**⚠️ 局限性**

局限性包括：
  1) 多样性不是自动产生的，需要手动正则化且需调参；
  2) 在复杂高维实例中，收敛到所有分支仍存在困难，尤其当能量函数缺乏足够区分度时；
  3) 对于需要特定解分支的实际工程任务，单解监督仍可能更简单；
  4) 需要多次随机初始化和较长迭代，计算成本相对较高。

---

## 165. When Descent Is Too Stable: Event-Triggered Hamiltonian Learning to Optimize

**arXiv ID:** 2605.06868 | [PDF](https://arxiv.org/pdf/2605.06868v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 166. CSR: Infinite-Horizon Real-Time Policies with Massive Cached State Representations

**arXiv ID:** 2605.07325 | [PDF](https://arxiv.org/pdf/2605.07325v1)

**作者:** Robin Karlsson `[一作]` (GODOT Inc), Go Suzui `[通讯]` (GODOT Inc)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Cached State Representation (CSR) 与 Asynchronous State Reconciliation (ASR) 框架，显著降低大规模 LLM 在机器人连续推理中的时间‑到‑首标记 (TTFT) 延迟。

**💡 创新点**

创新点在于通过理论证明前缀稳定性、增量可扩展性与异步状态重构三条必要条件，构建最优 KV‑cache 重用策略，并在无限时域内实现无延迟峰值的连续高频推理。

**🔧 技术方法**

使用 KV‑cache 复用、动态状态分块、异步计算资源预热、vLLM 推理引擎、GPU 边缘计算等技术，并结合 Qwen 2.5‑235B 等大模型。

**📊 数据集**

实验数据来自物理机器人平台（Raspberry Pi 5 + 4×H100 GPU）和 AMA‑bench 融合评测数据集。

**📈 对比分析**

与无序基线、StreamingLLM、RAG、AMA‑Agent 等方法对比，CSR 在 120K token 上实现 26‑倍 TTFT 降低（14.67 s → 0.56 s），ASR 在连续 10 次回收周期内保持无峰值；在 AMA‑bench recall 0.836，明显优于其他方案。

**⚠️ 局限性**

局限性包括对大规模 GPU 与高速网络的高度依赖，动态后缀长度受限，模型规模与推理速度之间仍存在权衡，且实现复杂度较高。

---

## 167. McNdroid: A Longitudinal Multimodal Benchmark for Robust Drift Detection in Android Malware

**arXiv ID:** 2605.06894 | [PDF](https://arxiv.org/pdf/2605.06894v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 168. Deterministic Monotone Min-Plus Product and Convolution

**arXiv ID:** 2605.07150 | [PDF](https://arxiv.org/pdf/2605.07150v1)

**作者:** Ce Jin `[一作]` (University of California Berkeley), Yinzhan Xu `[通讯]` (University of California San Diego)

**通讯引用:** 413 | [OpenAlex ID](https://openalex.org/A5005932872)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种确定性算法来解决单调最小加法乘积问题，该算法的时间复杂度为 n^(ω+3)/2+o(1) = (n^2.686)，与之前的随机算法相同。

**💡 创新点**

创新点在于提供了一种确定性算法，克服了之前算法需要随机化的缺点，并且可以推广到其他相关问题和变体。

**🔧 技术方法**

使用了多项式矩阵乘法和标准化的随机化去除技术。

**📊 数据集**

使用了多种数据集，包括整数矩阵和单调数组，具体取决于问题的不同变体。

**📈 对比分析**

与之前的随机算法相比，性能相当，但提供了确定性保证，避免了随机化带来的不确定性。

**⚠️ 局限性**

算法的局限性在于仍然依赖于矩阵的结构特性，可能无法有效处理所有类型的输入矩阵。

---

## 169. OneViewAll: Semantic Prior Guided One-View 6D Pose Estimation for Novel Objects

**arXiv ID:** 2605.07023 | [PDF](https://arxiv.org/pdf/2605.07023v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 170. Pomegranate: A Lightweight Compartmentalization Architecture using Virtualization Extensions

**arXiv ID:** 2605.07008 | [PDF](https://arxiv.org/pdf/2605.07008v1)

**作者:** Shriram Raja `[一作]` (Boston University), Richard West `[通讯]` (Boston University)

**通讯引用:** 21334 | [OpenAlex ID](https://openalex.org/A5015039212)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于硬件虚拟化的轻量级内核隔离框架，利用 EPT、vmfunc 及 #VE 等技术，在不修改 Linux 内核源码的情况下，实现对内核功能模块的细粒度隔离。

**💡 创新点**

创新点在于：①通过外部 hypervisor 的 EPT 机制与通用 “sentry” 代码，自动处理跨隔区调用，无需对内核进行代码注释或重编译；②结合 HLAT 防御重映射攻击，提升安全性；③通过对 igc 驱动的边界细化与批量化调用，显著降低隔区切换开销。

**🔧 技术方法**

使用的主要技术包括 Intel VT‑x（EPT、vmfunc、Virtualization Exceptions、HLAT）、Quest‑V 分区 hypervisor 以及 Linux 6.8.1 内核；此外采用 C/C++ 编写的 sentry 代码和自定义 policy 文件实现访问控制。

**📊 数据集**

实验数据集为 2.5 Gbps Intel I225‑V NIC 驱动 igc 的网络流量，使用 netperf UDP_STREAM 测试不同报文尺寸（64‑1472 B）并与未隔离基线内核进行对比。

**📈 对比分析**

通过比较基线、未优化隔离和优化后隔离三种配置的吞吐量，发现 MTU 大小包下优化后吞吐量与基线相近；小包时存在 70‑90 % 的下降；多隔区情况下（2‑32 个隔区）吞吐量保持在线速，差距不足 2 Mbps。

**⚠️ 局限性**

主要局限包括：仅支持静态隔区，无法动态创建或修改策略；缺乏子页面级隔离和堆对象隔离；实验仅在单核上进行，未评估多核迁移成本；对 VM‑func、HLAT 等硬件功能的依赖。

---

## 171. EMRGF: A Practitioner Framework for Governance-Driven Enterprise Technology Modernization

**arXiv ID:** 2605.06703 | [PDF](https://arxiv.org/pdf/2605.06703v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 172. Palm-sized Omnidirectional Vision-Based UAV Exploration with Sparse Topological Map Guidance

**arXiv ID:** 2605.07275 | [PDF](https://arxiv.org/pdf/2605.07275v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 173. On Training in Imagination

**arXiv ID:** 2605.06732 | [PDF](https://arxiv.org/pdf/2605.06732v1)

**作者:** Nadav Timor `[一作]` (Weizmann Institute of Science), David Harel `[通讯]` (Weizmann Institute of Science)

**通讯引用:** 30846 | [OpenAlex ID](https://openalex.org/A5088778612)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对“在想象中训练”框架进行理论分析，提出了将奖励模型视为独立误差源，给出了返回误差的Lipschitz分解，推导了在功率律误差假设下动态样本与奖励样本的最优分配比例，并对奖励噪声的方差与系统性偏差给出了梯度估计的上界与最优预算分配策略。

**💡 创新点**

创新点在于：①将奖励误差与动力学误差分离并给出独立可控的系数；②揭示了Lipschitz常数越小越能紧化误差上界，并与时间平滑化目标相关；③在功率律错误缩放下求得动态样本与奖励样本的闭式最优比例；④在奖励噪声情形下给出噪声方差与采样成本的折衷函数，说明何时应优先投入更高精度奖励或更多样本；⑤阐明系统性奖励偏差无法通过增大样本数消除。

**🔧 技术方法**

主要技术包括：基于Lipschitz理论的误差传播分析、模拟引理（simulation lemma）推广、功率律误差缩放假设、方差上界推导、单变量最优化求解、以及对非平稳奖励偏差的梯度偏差分析。

**📊 数据集**

实验使用了两个基准：①基于随机初始化的2层ReLU MLP 的连续控制环境（synthetic continuous-control），②线性二次高斯（LQG）控制任务。两者均在离线数据集上估计动力学与奖励误差，并拟合功率律曲线。

**📈 对比分析**

与方法的比较主要是通过误差上界与实际误差的比值（R = LHS/RHS）进行检验，结果显示上界在所有525个配置下成立，但典型情况下上界相对松散（合计约65×）。在样本分配实验中，理论给出的最优比例与实际样本比例在使用真实值函数敏感度时高度一致，但使用全局Lipschitz常数时误差高达数千倍；在奖励噪声分配实验中，证明了Φ(c) 的形状决定了是否优先提升奖励精度或增加采样数量。

**⚠️ 局限性**

局限性包括：①假设环境动力学确定性、γL_f(1+L_π)<1 的收敛条件；②依赖全局Lipschitz常数，实际中可能过于保守；③未考虑随机或高维观测下的动态/奖励模型误差；④对系统性奖励偏差的处理仅给出梯度偏差理论，缺乏实用的补偿方法；⑤实验范围仅限于合成控制任务，缺乏在真实机器人或复杂游戏环境中的验证。

---

## 174. EnvSimBench: A Benchmark for Evaluating and Improving LLM-Based Environment Simulation

**arXiv ID:** 2605.07247 | [PDF](https://arxiv.org/pdf/2605.07247v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 175. PSK@EEUCA 2026: Fine-Tuning Large Language Models with Synthetic Data Augmentation for Multi-Class Toxicity Detection in Gaming Chat

**arXiv ID:** 2605.07201 | [PDF](https://arxiv.org/pdf/2605.07201v1)

**作者:** Srikar Kashyap Pulipaka `[一作]` `[通讯]`, Srikar Kashyap Pulipaka

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对World of Tanks聊天文本构建毒性行为分类系统，利用Llama 3.1 8B、LoRA微调与5%合成数据实现对六类毒性标签的识别。

**💡 创新点**

发现并解决“验证陷阱”现象，确定5%合成数据为最佳比例，并通过结构化提示提升模型区分细粒度毒性类别的能力。

**🔧 技术方法**

采用指令调优的大语言模型Llama 3.1 8B，结合LoRA参数高效微调、4‑bit NF4量化、合成数据增强与prompt工程等技术。

**📊 数据集**

使用GameTox语料库中World of Tanks聊天文本，包含约42,959条样本，涵盖非毒性、侮辱/挑衅、其他攻击、仇恨/骚扰、威胁与极端主义六类标签。

**📈 对比分析**

与XLM‑RoBERTa、Gemma 2B/12B、两阶段分类等多种模型对比，官方指标宏F1，最佳系统宏F1为0.6234，排名第4/35。

**⚠️ 局限性**

仅在该特定数据集上验证，验证陷阱现象可能不具普适性；合成数据依赖商业LLM；算力限制未尝试更大模型或更长训练。

---

## 176. LensVLM: Selective Context Expansion for Compressed Visual Representation of Text

**arXiv ID:** 2605.07019 | [PDF](https://arxiv.org/pdf/2605.07019v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 177. Privacy Perceptions in Sensor-Powered Smart Vehicle Cabins

**arXiv ID:** 2605.06847 | [PDF](https://arxiv.org/pdf/2605.06847v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 178. AdpSplit: Error-Driven Adaptive Splitting for Faster Geometry Discovery in 3D Gaussian Splatting

**arXiv ID:** 2605.06876 | [PDF](https://arxiv.org/pdf/2605.06876v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 179. GSM-SEM: Benchmark and Framework for Generating Semantically Variant Augmentations

**arXiv ID:** 2605.07053 | [PDF](https://arxiv.org/pdf/2605.07053v1)

**作者:** Jyotika Singh `[一作]` (Oracle), Dan Roth `[通讯]` (Oracle)

**通讯引用:** 30534 | [OpenAlex ID](https://openalex.org/A5023802054)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出GSM-SEM框架，用以生成保持原答案且计算不变但语义多样化的题目变体。

**💡 创新点**

创新点在于：可重用的随机化语义扰动、答案与推理保持不变、基于多模型一致性和语义相似度的严格过滤，以及可与已有变体套件叠加。

**🔧 技术方法**

技术手段包括：LLM反向工程生成新问题、规则级验证保证数值一致、余弦相似度冗余裁剪、跨模型一致性得分过滤以及多种扰动策略（实体替换、属性修改、关系变更）。

**📊 数据集**

使用的数据集为GSM8K、GSM-Symbolic、GSM-Plus、BigBench-Hard、LogicBench 和 NLR-BIRD，并在这些基准上生成SEM变体。

**📈 对比分析**

评估方法：在14个SOTA LLM上进行零-shot（及少量-shot）测试，计算准确率差和性能下降率；结果表明SEM变体平均导致13.5%性能下降，某些组合达到约28%的最大降幅。

**⚠️ 局限性**

局限性：只适用于包含明确答案与可解析推理路径的问答数据集；需要人工审核保证质量；过滤与评估过程可能引入系统偏差；并非覆盖所有可能的语义扰动类型。

---

## 180. FATE: Future-State-Aware Scheduling for Heterogeneous LLM Workflows

**arXiv ID:** 2605.07238 | [PDF](https://arxiv.org/pdf/2605.07238v1)

**作者:** Zirui Huang `[一作]` (University of Science and Technology of China), Xiangyang Li `[通讯]` (University of Science and Technology of China)

**通讯引用:** 18532 | [OpenAlex ID](https://openalex.org/A5100341802)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对异构多阶段LLM工作流的调度框架FATE，强调未来执行状态的保护来提升端到端延迟

**💡 创新点**

将调度目标从单纯的即时成本扩展到未来状态的优化，结合模型驻留、父子依赖、前缀重用与设备可达性等多维度状态；采用CP‑SAT前沿规划+状态条件评分实现

**🔧 技术方法**

使用CP‑SAT求解器做前沿规划，水平感知候选评分，状态条件成本估算，有限下游视野的前沿计划与增量提交，支持有界多设备分片执行

**📊 数据集**

基于WfCommons/WfInstances构建的工作流DAG基准，以及控制前缀重用的人工合成基准

**📈 对比分析**

与RoundRobin、KVFlow、Helix、HEFT、Halo等基线在同一框架下比较；在工作流DAG基准上FATE实现归一化完成时间与P95延迟分别下降约32%和33%，在前缀重用基准中也持续领先

**⚠️ 局限性**

实验仅覆盖结构化工作流与受控前缀重用，未考虑高峰到达、动态漂移；基线为在统一框架中的适配实现；诊断指标为代理指标，未测量真实硬件传输或缓存命中

---

## 181. ARMOR: An Agentic Framework for Reaction Feasibility Prediction via Adaptive Utility-aware Multi-tool Reasoning

**arXiv ID:** 2605.07103 | [PDF](https://arxiv.org/pdf/2605.07103v1)

**作者:** Ye Liu `[一作]` (Ohio State University), Xia Ning `[通讯]` (Ohio State University)

**通讯引用:** 122888 | [OpenAlex ID](https://openalex.org/A5086524838)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一个Agentic框架，利用工具层级、工具优先级与冲突解决三大模块实现多工具协同进行化学反应可行性预测。

**💡 创新点**

创新点在于显式建模工具的具体效用与模式，采用层级化工具选择并通过记忆增强的对比推理解决工具冲突，显著提升预测精度。

**🔧 技术方法**

主要技术包括大语言模型（LLM）用于模式抽取与冲突解析、DRFP指纹检索相似反应、工具优先级排序以及记忆增强推理机制。

**📊 数据集**

使用FREA数据集（基于USPTO的12,000条可行/不可行反应，按6,000/6,000划分为验证集和测试集）进行实验。

**📈 对比分析**

通过与单工具、统计聚合、动态选择及LLM驱动的基线对比，实验显示在整体ACC 91.62%、F1 91.57%和MCC 0.8323的成绩上显著优于HarderMoE和其他方法。

**⚠️ 局限性**

局限性包括对工具模式和对比示例的依赖，工具数量和多样性受限，且在极端工具预测不一致的极端场景下仍可能存在性能瓶颈。

---

## 182. MedExAgent: Training LLM Agents to Ask, Examine, and Diagnose in Noisy Clinical Environments

**arXiv ID:** 2605.07058 | [PDF](https://arxiv.org/pdf/2605.07058v1)

**作者:** Yicheng Gao `[一作]` (University of Southern California), Ruishan Liu `[通讯]` (University of Southern California)

**通讯引用:** 1269 | [OpenAlex ID](https://openalex.org/A5080351422)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了基于POMDP的临床诊断环境，并训练了MedExAgent以交互式提问、检验调用和诊断输出，兼顾噪声与成本。

**💡 创新点**

创新点包括：①将提问、检验和诊断统一为单一代理的三种动作；②引入系统化噪声模型和真实费用/不适度量化的检验成本；③采用两阶段训练（SFT+DAPO）并设计多项奖励函数。

**🔧 技术方法**

技术手段主要是LLM（基于Meditron3-8B）+POMDP建模+自制噪声注入+工具调用奖励+强化学习算法DAPO。

**📊 数据集**

使用了DDxPlus（1.3M合成病例）、PMC-Patients-v2（250k真实病例）以及AgentClinic-MedQA（201个USMLE样本）作为训练与评估数据集。

**📈 对比分析**

通过与7个基线（包括大规模医用LLM和通用LLM）对比，MedExAgent-8B在三大测试集上的Jaccard、精确率和Cosine相似度均显著优于所有基线，尤其在OOD AgentClinic-MedQA上达到与70B模型相当的性能。

**⚠️ 局限性**

主要局限包括：假设患者病情随时间不变；训练全部基于LLM模拟患者，可能存在分布偏差；仅使用英文数据，缺乏多语言适配。

---

## 183. Agentic Coding Needs Proactivity, Not Just Autonomy

**arXiv ID:** 2605.06717 | [PDF](https://arxiv.org/pdf/2605.06717v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 184. A Wasserstein GAN-based climate scenario generator for risk management and insurance: the case of soil subsidence

**arXiv ID:** 2605.06678 | [PDF](https://arxiv.org/pdf/2605.06678v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 185. A UEFI System with SPDM to Protect Against Unauthorized Device Connections

**arXiv ID:** 2605.06744 | [PDF](https://arxiv.org/pdf/2605.06744v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 186. LKV: End-to-End Learning of Head-wise Budgets and Token Selection for LLM KV Cache Eviction

**arXiv ID:** 2605.06676 | [PDF](https://arxiv.org/pdf/2605.06676v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 187. NSMQ Riddles: A Benchmark of Scientific and Mathematical Riddles for Quizzing Large Language Models

**arXiv ID:** 2605.07051 | [PDF](https://arxiv.org/pdf/2605.07051v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 188. MORPH-U: Multi-Objective Resilient Motion Planning for V2X-Enabled Autonomous Driving in High-Uncertainty Environments via Simulation

**arXiv ID:** 2605.07370 | [PDF](https://arxiv.org/pdf/2605.07370v1)

**作者:** Shih-Yu Lai `[一作]` (National Taiwan University), Shih-Yu Lai `[通讯]` (National Taiwan University)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5103006974)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发并评估了 MORPH‑U，一套基于 CARLA 的闭环车载堆栈，融合 LiDAR/雷达/摄像头与 V2X CAM/DENM，利用 Hybrid‑A* 事件驱动再规划，并通过 Pareto 前沿选择控制参数。

**💡 创新点**

创新点在于将 V2X 作为 LDM 的唯一接口实现概率融合、明确事件驱动再规划的触发与 Byzantine‑inspired 容错门，以及在同一闭环堆栈中公开多目标 Pareto 前沿与鲁棒性评估。

**🔧 技术方法**

技术包括 CARLA 仿真、时间同步融合、Bayesian 权重融合、Hybrid‑A* 搜索、Pure Pursuit + PID 控制、基于分数的多目标优化与阈值门控。

**📊 数据集**

数据集为 CARLA 生成的合成城市交叉口与单车道路径场景，包含 30 种随机种子，并在 S4 场景注入由 10 台站点（3 台 Byzantine）产生的 DENM 攻击。

**📈 对比分析**

通过与 CARLA Autopilot、Sensors‑Only、Sensors+V2X（无门控）基线对比，使用追踪误差、TTC、碰撞数、响应延迟、平滑度、LDM MOTA/MOTP 以及 FPR/FNR 等指标评估；结果显示 V2X 降低追踪误差 16.7%，提高 TTC 46%，完成率提升 24.3pp，门控实现 FPR 0 且 FNR 0。

**⚠️ 局限性**

限制在于评估仅在仿真下完成，未涉及真实无线网络延迟、数据包丢失、协调攻击等；门控仅为经验性的 Byzantine‑inspired 方案，缺乏正式容错证明，且在不同感知/定位条件下性能可能变化。

---

## 189. Kurtosis-Guided Denoising Score Matching for Tabular Anomaly Detection

**arXiv ID:** 2605.06955 | [PDF](https://arxiv.org/pdf/2605.06955v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 190. Asymmetric Phase Coding Audio Watermarking

**arXiv ID:** 2605.07241 | [PDF](https://arxiv.org/pdf/2605.07241v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 191. Cognitive Agent Compilation for Explicit Problem Solver Modeling

**arXiv ID:** 2605.07040 | [PDF](https://arxiv.org/pdf/2605.07040v1)

**作者:** Hyeongdon Moon `[一作]` (Carnegie Mellon University), John Stamper `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2869 | [OpenAlex ID](https://openalex.org/A5060576109)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了将教师LLM的解题能力编译为可解释、可编辑的目标智能体的框架；

**💡 创新点**

首次将教师LLM的诊断与知识编译过程结合，形成可检查的显式知识库，弥补传统LLM缺乏可解释性与可控性的问题；

**🔧 技术方法**

采用小型语言模型（Qwen2.5-2B）作为执行器，Gemma-3-27B 作为教师LLM，使用检索式记忆、QKV机制以及日志概率驱动的动作选择；

**📊 数据集**

使用CMU DataShop公开的OLI_Biology多选题数据集；

**📈 对比分析**

在27道题上通过平均3.1次编译迭代成功完成，累计188条声明性记忆；在第28题达到150次迭代仍未收敛，说明模型在大规模知识库时存在检索退化；

**⚠️ 局限性**

受限于模型先验知识、检索退化（fan effect）、小模型的不可解释性以及对教师LLM的高度依赖，导致可扩展性与可解释性之间的权衡尚未解决。

---

## 192. TriP: A Triangle Puzzle Approach to Robust Translation Averaging

**arXiv ID:** 2605.07143 | [PDF](https://arxiv.org/pdf/2605.07143v1)

**作者:** Zhekai Fan `[一作]` (UC Davis), Yunpeng Shi `[通讯]` (UC Davis)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 TriP，一种基于三角形的鲁棒平移平均框架，用三角形几何推断相邻相对尺度并在对数域同步尺度，从而恢复相机位置。

**💡 创新点**

创新点在于：①使用三角形局部尺度信息而非仅靠边缘或循环一致性；②在对数域进行尺度同步，天然避免零尺度崩塌；③给出针对任意摄像机位置和攻击性畸变的确定性完全恢复保证，突破传统方法对腐败度消失的限制。

**🔧 技术方法**

核心技术包括三角形预筛选（闭合残差与共面检测）、对数尺度同步（Cauchy 损失下的 IRLS 解决）、距离估计与中位数聚合、以及最终的 Cauchy 位移平均求解；实现时提供并行的 PCG 与 Weiszfeld 类型局部平均解算器。

**📊 数据集**

实验使用：①合成网格与环面几何场景，加入统一/聚类一致性腐败；②十二个真实 SfM 数据集（如室内外城市、建筑等多样场景）。

**📈 对比分析**

与 Cycle‑Sync、LUD、BATA、ShapeFit、FusedTA 等基线对比，TriP 在合成数据上在更高腐败比例下保持近零中位误差；在真实数据上在各覆盖率下均实现最低的中位/平均误差，并显著降低尾部误差，速度上可比最快基线快数百倍。

**⚠️ 局限性**

局限性：依赖足够的三角形结构，对极度稀疏图可能失效；对极端攻击性噪声的鲁棒性仍有限，需进一步改进三角形选择与同步策略。

---

## 193. Where to Spend Rollouts: Hit-Utility Optimal Rollout Allocation for Group-Based RLVR

**arXiv ID:** 2605.07114 | [PDF](https://arxiv.org/pdf/2605.07114v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 194. Conformal Agent Error Attribution

**arXiv ID:** 2605.06788 | [PDF](https://arxiv.org/pdf/2605.06788v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 195. A$^2$RD: Agentic Autoregressive Diffusion for Long Video Consistency

**arXiv ID:** 2605.06924 | [PDF](https://arxiv.org/pdf/2605.06924v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 196. PicoEyes: Unified Gaze Estimation Framework for Mixed Reality with a Large-Scale Multi-View Dataset

**arXiv ID:** 2605.07188 | [PDF](https://arxiv.org/pdf/2605.07188v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 197. See Tomorrow, Act Today: Foresight-Driven Autonomous Driving

**arXiv ID:** 2605.07195 | [PDF](https://arxiv.org/pdf/2605.07195v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 198. Information-theoretic Limits of Learning and Estimation

**arXiv ID:** 2605.06710 | [PDF](https://arxiv.org/pdf/2605.06710v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 199. A Systematic Investigation of The RL-Jailbreaker in LLMs

**arXiv ID:** 2605.07032 | [PDF](https://arxiv.org/pdf/2605.07032v1)

**作者:** Montaser Mohammedalamen `[一作]` (Alberta Machine Intelligence Institute), Alyssa Lefaivre Škopac `[通讯]` (Alberta Machine Intelligence Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 RL 机制下的 LLM 破防（jailbreak）进行了系统化拆解与实验验证，评估了不同环境与算法配置对攻击成功率的影响。

**💡 创新点**

首次从 RL 视角而非模型视角拆解 Jailbreak 框架，明确奖励函数、episode 长度、动作空间等结构因素是攻击成功的关键驱动，并验证了 dense reward 与较长 episode 的重要性。

**🔧 技术方法**

使用 PPO 与 DDQN 两种强化学习算法，构建 POMDP 环境，采用 cosine 相似度奖励，结合 LLM helper、prompt 与 response safeguard，进行多轮交互式攻击。

**📊 数据集**

采用 AdvBench（20 条有害提示）作为训练与评估数据，目标模型为 Llama‑3.2‑1B、Llama‑3.2‑3B、Qwen‑3‑4B、Tiny‑aya‑global，配合 Llama‑Guard 与 ShieldGemma 等 safeguard。

**📈 对比分析**

通过对比 baseline、稀疏奖励、密集奖励、动作空间扩展、episode 长度、奖励加成、训练样本量及 RL 算法等设置，发现 dense reward 与更长 episode 在大多数模型上能显著提升 ASR（最高可达 63%），而 DDQN 与 PPO 的表现相近。

**⚠️ 局限性**

局限性包括仅针对小型开源 LLM 进行实验，未涉及闭源或多模态模型；实验规模受计算限制，未能覆盖多语言与高风险专业域；奖励函数与安全模型仅为基础版本，未探索更复杂或更强的防御与多目标奖励。

---

## 200. A Behavioral Framework for Data-Driven Modeling of Nonlinear Systems in Vector-Valued Reproducing Kernel Hilbert Spaces

**arXiv ID:** 2605.07052 | [PDF](https://arxiv.org/pdf/2605.07052v1)

**作者:** Boya Hou `[一作]` (University of Illinois Urbana Champaign), Maxim Raginsky `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

论文提出了一种将Jan Willems的行为学方法推广到离散时间非线性系统的框架，并将其与在向量值核 RKHS 中的两种数据驱动建模方法（最小范数插值和子空间识别）相结合，形成了一套完整的行为建模与预测工具。

**💡 创新点**

创新点在于：
• 将行为学视角与核方法结合，构造了一个可覆盖 LTI、Volterra 系列、Hammerstein 等非线性模型的通用 RKHS 表示；
• 推导了“行为表示定理”（Behavior Representer Theorem），给出最小范数插值在 RKHS 中的精确误差公式；
• 将子空间识别技术转化为在 RKHS 中的 Hankel 结构，得到非线性系统的“基本引理”对应形式；
• 明确了结构假设（系统可控性/可观性、输入的持久激励、RKHS 选取等）对建模与预测的影响。

**🔧 技术方法**

使用技术包括：
• 向量值核 RKHS 与重现定理；
• 最小范数插值与核矩阵求逆；
• 子空间识别的偏移算子、投影与奇异值分解；
• 结构化的 Hankel 矩阵与不完全矩阵的伪逆；
• 线性系统理论（可控性/可观性）与非线性系统的状态空间展开。

**📊 数据集**

该工作主要是理论推导，没有使用公开数据集进行实验；若有实验，数据来源未在论文中给出。

**📈 对比分析**

方法比较：
• 与传统 LTI 的基本引理相似，但在 RKHS 中实现；
• 对非线性系统，最小范数插值提供了精确的预测误差表达式；
• 子空间识别能够直接从训练数据恢复状态序列和系统矩阵；
• 性能表现主要体现在理论性质（可恢复性、误差上界）上，论文未给出数值性能对比。

**⚠️ 局限性**

局限性包括：
• 需要对输入序列满足持久激励且 RKHS 选取合适；
• 计算复杂度随样本数和 RKHS 维度指数增长；
• 仅适用于可观测/可控的系统结构，且假设系统参数可写成 RKHS 中的核函数形式；
• 目前仅在理论层面验证，缺乏实测数据验证与实现细节。

---

## 201. Response Time Enhances Alignment with Heterogeneous Preferences

**arXiv ID:** 2605.06987 | [PDF](https://arxiv.org/pdf/2605.06987v1)

**作者:** Federico Echenique `[一作]` (University of California, Berkeley), Michael I. Jordan `[通讯]` (University of California, Berkeley)

**通讯引用:** 180049 | [OpenAlex ID](https://openalex.org/A5049812527)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出利用匿名标签者的响应时间与漂移扩散模型（DDM）结合，恢复异质偏好环境下的总体平均偏好。

**💡 创新点**

创新点在于证明响应时间可以恢复由单一奖励模型失去的可识别性，构造了无偏估计器和一致的边界估计器（使用 Richardson 外推），并在异质标签者池中实现对平均偏好的一致估计。

**🔧 技术方法**

采用漂移扩散模型、拉普拉斯变换、Richardson 外推、加权平均法和线性回归（OLS）等技术；在基于响应时间的伪目标下对均值漂移和平均偏好进行估计。

**📊 数据集**

在合成数据（两种先验分布的均值漂移和多维线性偏好）以及公开的跨期决策（intertemporal-choice）真实数据集上进行实验。

**📈 对比分析**

与仅使用二元选择的最大似然（MLE）基线以及使用oracle边界的DDM进行对比；结果显示DDM在MSE、余弦相似度等指标上显著优于基线，误差随样本量递减，几乎达到oracle性能。

**⚠️ 局限性**

局限性包括：假设所有标签者共享相同的决策边界；边界估计在有限样本下收敛速度慢；模型对实际决策行为的适配可能受限，若标签者的温度或非决策时间随时间或个体变化，DDM假设可能被违背。

---

## 202. Near-field Channel Estimation for XL-RIS-aided mmWave MIMO Systems

**arXiv ID:** 2605.06719 | [PDF](https://arxiv.org/pdf/2605.06719v1)

**作者:** Erkang Dong `[一作]` (Southeast University), Jiangzhou Wang `[通讯]` (Southeast University)

**通讯引用:** 16991 | [OpenAlex ID](https://openalex.org/A5029002143)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

针对极大规模 RIS 辅助毫米波 MIMO 系统中的近场链路估计，提出了低开销的两阶段估计框架。

**💡 创新点**

创新点在于联合利用 BS‑RIS 共享远场通道结构与 RIS‑用户近场极化域稀疏性，并通过典型虚拟用户提取公共参数，随后使用 ALS 迭代精细化全局通道。

**🔧 技术方法**

采用稀疏恢复（OMP、LAOMP）、极化域字典、近场近似、ALS 交替最小二乘、虚拟单天线用户分解等技术。

**📊 数据集**

仿真使用 30 GHz 系统，BS 128 天线、RIS 256 元素、4 个用户、每个用户 32 天线，随机生成路径数、距离与路径损耗模型。

**📈 对比分析**

与现有近场基准及传统远场估计方法比较，NMSE 在 10 dB SNR 下优于单步初始化，ALS 收敛快，所需 pilot 大幅低于基准（对数级 vs 线性级），在路径数增大时仍保持竞争性能。

**⚠️ 局限性**

局限性包括对 RIS 训练矩阵设计的假设、对远场 BS‑RIS 的严格假设、以及路径数极多时仍需更高迭代次数或字典尺寸；对多用户间干扰建模未深度讨论。

---

## 203. Toward Individual Fairness Without Centralized Data: Selective Counterfactual Consistency for Vertical Federated Learning

**arXiv ID:** 2605.07117 | [PDF](https://arxiv.org/pdf/2605.07117v1)

**作者:** Dawood Wasif `[一作]` (Virginia Tech), Jin-Hee Cho `[通讯]` (Virginia Tech)

**通讯引用:** 5869 | [OpenAlex ID](https://openalex.org/A5011649304)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了在垂直联邦学习中实现个体层级反事实一致性的框架SCC‑VFL，保证在敏感属性干预下的预测稳定性。

**💡 创新点**

创新点在于：①无图谱、差分隐私的特征角色分离；②仅编辑许可的中介特征的受控生成；③服务器端的个体反事实一致性损失，实现对代理路径的精确控制。

**🔧 技术方法**

采用了图自由的差分隐私特征分组、HSIC与风险差异判定、条件变分自编码器生成器、服务器侧一致性正则、分裂学习等技术。

**📊 数据集**

在德国信用、UCI心脏病和COMPAS暴力子集等真实业务领域的表格数据集上进行实验。

**📈 对比分析**

与四个基线（无掩码对抗、均匀反事实、无策略掩码、仅服务器一致性）对比，SCC‑VFL在准确率和对数损失几乎不下降的前提下，选择性一致性间隙与翻转率显著下降（最高可降98%），且属性推断和PGD攻击下的泄漏率更低。

**⚠️ 局限性**

局限性包括：仅验证离散敏感属性和低维表格特征；依赖先验的政策定义；DP保护仅限于特征筛选阶段，未提供完整训练过程的端到端隐私证明。

---

## 204. Convergence and Emergence of In-Context Reinforcement Learning with Chain of Thought

**arXiv ID:** 2605.07123 | [PDF](https://arxiv.org/pdf/2605.07123v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 205. Neurosymbolic Framework for Concept-Driven Logical Reasoning in Skeleton-Based Human Action Recognition

**arXiv ID:** 2605.07140 | [PDF](https://arxiv.org/pdf/2605.07140v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 206. Multi-Objective Constraint Inference using Inverse reinforcement learning

**arXiv ID:** 2605.06951 | [PDF](https://arxiv.org/pdf/2605.06951v1)

**作者:** Syed Ihtesham Hussain Shah `[一作]` (Vrije Universiteit Amsterdam), Annette ten Teije `[通讯]` (Vrije Universiteit Amsterdam)

**通讯引用:** 2447 | [OpenAlex ID](https://openalex.org/A5000424011)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种多目标约束推断框架（MOCI），可从异质专家演示中联合学习共享硬约束和个体偏好。

**💡 创新点**

创新点在于将约束推断与偏好学习结合，使用EM+最大熵IRL迭代聚类与权重更新，支持异质数据且对约束进行可解释的贪婪搜索。

**🔧 技术方法**

使用EM算法、最大熵逆强化学习、梯度上升、贪婪约束搜索及KL阈值正则化。

**📊 数据集**

在合成的Gridworld环境（N×N格子）上测试，含有水域硬约束，生成两类专家（Grass-Lover、Rock-Lover）共20条轨迹。

**📈 对比分析**

与MLCI、ICRL等基线对比，MOCI在CMSE上显著下降（0.027 vs 0.25/0.36），运行时间为0.69s，既保持较高精度又具备竞争力。

**⚠️ 局限性**

局限性：需预先指定专家类型数K；仅处理硬约束，无法推断软约束；在稀疏轨迹覆盖时可能误判未访问状态为约束。

---

## 207. When Routine Chats Turn Toxic: Unintended Long-Term State Poisoning in Personalized Agents

**arXiv ID:** 2605.06731 | [PDF](https://arxiv.org/pdf/2605.06731v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 208. A Comprehensive Survey on Agent Skills: Taxonomy, Techniques, and Applications

**arXiv ID:** 2605.07358 | [PDF](https://arxiv.org/pdf/2605.07358v1)

**作者:** Yingli Zhou `[一作]` (Chinese University of Hong Kong), Xuemin Lin `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 99321 | [OpenAlex ID](https://openalex.org/A5100773343)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了基于大语言模型的代理（agent）系统中可重用的“技能”概念，阐述了技能的生命周期（表示、获取、检索与选择、演化）并对相关技术与生态系统进行系统分类与评述。

**💡 创新点**

创新点在于将技能视为代理系统的核心执行层，提出以生命周期视角统一归纳技能研究，并在此基础上构建了“技能生命周期”四阶段框架与对应的技术与方法分支；同时收集并整理了丰富的资源与项目（如Awesome-Agent-Skills 代码库）。

**🔧 技术方法**

技术主要包括：文本/代码/混合技能表示、经验抽象（Selection、Abstraction、Memory、Procedural Packaging）、任务/语料/人类导向的技能获取、基于稠密/稀疏/结构化检索、上下文感知、成本/效用/反馈驱动的选择策略、以及技能修订、验证、策略耦合、仓库演化与运行时治理等一系列机制。

**📊 数据集**

由于是综述，未使用单一实验数据集；而是基于近三年内（2023‑2026）公开发表的论文、公开技能库、示例系统（如OpenClaw、Claude Code、SkillCraft 等）以及 GitHub 等开源资源进行整理与案例分析。

**📈 对比分析**

对方法的比较主要采用“技术族分类”与“生命周期模块”视角：在每个生命周期阶段列举代表性工作，并通过表格展示其核心设计、结构先验、决策输入与发表年份；未给出统一的数值性能评测，但通过对各类方法的功能、适用场景与优缺点进行系统对比，形成了对现有技术景观的宏观评估。

**⚠️ 局限性**

局限性包括：1) 综述性工作缺乏统一的实验基准与量化评测，难以直接比较方法性能；2) 由于快速演进，部分最新工作可能在文献收录后出现；3) 生态系统与标准化程度不高，导致跨平台迁移与互操作性仍有待解决；4) 对技能质量控制、可信治理等安全性与伦理问题的深入分析仍不足。

---

## 209. Dual-Agent Co-Training for Health Coaching via Implicit Adversarial Preference Optimization

**arXiv ID:** 2605.07011 | [PDF](https://arxiv.org/pdf/2605.07011v1)

**作者:** Da Long `[一作]` (University of Utah), Shandian Zhe `[通讯]` (University of Utah)

**通讯引用:** 1614 | [OpenAlex ID](https://openalex.org/A5024663093)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `a2602d71-93ab-4bad-974b-672788df8193` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

暂无具体研究内容

**💡 创新点**

无法确定创新点

**🔧 技术方法**

未知

**📊 数据集**

未知

**📈 对比分析**

未提供比较与性能信息

**⚠️ 局限性**

缺乏详细描述

---

## 210. A Framework of Variable-Length Source Encryption using Mutual Information Security Criterion: Universal Coding, Strong Converse Theorem

**arXiv ID:** 2605.06802 | [PDF](https://arxiv.org/pdf/2605.06802v1)

**作者:** Yasutada Oohama `[一作]` (University of Electro-Communications), Bagus Santoso `[通讯]` (University of Electro-Communications)

**通讯引用:** 176 | [OpenAlex ID](https://openalex.org/A5031357176)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种基于相互信息安全判据的可变长度无损源加密框架，并给出了在给定信息泄漏上界δ下的安全通信可实现性与最优码率的必要充分条件；

**💡 创新点**

创新点在于将相互信息作为信息泄露度量与可变长度编码结合，证明了与固定长度框架对应的强逆定理，并实现了对任意源分布与密钥分布的通用加密/解密方案；

**🔧 技术方法**

主要技术包括信息论编码理论、相互信息分析、对码字长度的可逆映射构造、指数随机变量的典型集分析和单一解码映射的构造；

**📊 数据集**

未使用具体实验数据集，研究完全基于理论分析与数理证明；

**📈 对比分析**

通过对比理论极限（H(X) ≤ R ≤ H(K)）与构造方案的实现误差和泄漏指数，证明了所提出方案在满足平均码字长度上界R且信息泄漏指数随n指数衰减的同时，实现了与理论极限一致的最优性能；

**⚠️ 局限性**

局限性包括仅适用于离散记忆无关源、密钥与源独立、无噪声信道等理想化假设，且未考虑实际编码复杂度与实现细节，未来工作需扩展到更一般的源模型与可失真编码情形。

---

## 211. TENNOR: Trustworthy Execution for Neural Networks through Obliviousness and Retrievals

**arXiv ID:** 2605.07160 | [PDF](https://arxiv.org/pdf/2605.07160v1)

**作者:** Zifan Qu `[一作]` (George Mason University), Evgenios M. Kornaropoulos `[通讯]` (George Mason University)

**通讯引用:** 458 | [OpenAlex ID](https://openalex.org/A5030196581)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练宽层神经网络时实现双重隐匿，消除访问模式泄露，保证模型和数据在可信执行环境中的安全训练

**💡 创新点**

提出MP‑WTA多探测Winner‑Take‑All局部敏感哈希以及专门的双重隐匿训练管线，显著降低LSH存储需求并提升安全性

**🔧 技术方法**

采用局部敏感哈希、OHT（可观测哈希表）、并行隐匿排序、Deferred Dummy Payload、Oblivious Parallelization、OHT Layout Reuse等技术，并在Intel TDX可信域内实现

**📊 数据集**

在极端多标签分类数据集 Wiki10‑31K、Amazon‑Sub、Wiki‑325K（以及 Amazon‑670K）上进行实验

**📈 对比分析**

与基于Path ORAM的 SLIDE+ORAM 基线对比，速度提升达13×–470×，将数百小时训练压缩至数十分钟，显著降低训练时间

**⚠️ 局限性**

受限于排序与哈希表构造的开销，需较多CPU资源；对GPU或更大模型的并行化支持有限，且在不同TEE实现下仍需进一步适配

---

## 212. SatSurfGS: Generalizable 2D Gaussian Splatting for Sparse-View Satellite Surface Reconstruction

**arXiv ID:** 2605.07181 | [PDF](https://arxiv.org/pdf/2605.07181v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 213. Learned Lagrangian Models of PDEs via Euler-Lagrange Residual Minimization

**arXiv ID:** 2605.07157 | [PDF](https://arxiv.org/pdf/2605.07157v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 214. Language Models Can Autonomously Hack and Self-Replicate

**arXiv ID:** 2605.06760 | [PDF](https://arxiv.org/pdf/2605.06760v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 215. Problem Space Attunement in Youth Social Media Design

**arXiv ID:** 2605.07018 | [PDF](https://arxiv.org/pdf/2605.07018v1)

**作者:** JaeWon Kim `[一作]` (University of Washington), JaeWon Kim `[通讯]` (University of Washington)

**通讯引用:** 64 | [OpenAlex ID](https://openalex.org/A5086265851)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在青少年社交媒体设计研究中，采用了虚构探究、异步远程社区（ARC）参与式协作与基于LLM的模拟沙盒三种方法，系统地探索并制定了以青少年为中心的关系支持性社交媒体设计框架与评估标准。

**💡 创新点**

创新点包括提出“问题空间细致调谐”框架，打破传统以成人视角为主的设计偏见；利用虚构探究扩展社交媒体的概念边界；通过ARC让青少年自主定义“更好”设计的目标与评判标准；构建LLM代理模拟沙盒，模拟真实社交网络中的动态交互，帮助评估设计的社会动力学。

**🔧 技术方法**

技术手段涵盖：虚构探究（Fictional Inquiry）工作坊、Discord异步讨论机器人、基于大型语言模型（LLM）的代理生成与交互模拟、以参与者描述为基础的个性化代理配置。

**📊 数据集**

数据来源为研究参与者本人提供的社交网络信息（包括亲密好友、熟人、家庭成员等的描述），未使用公开大规模数据集；虚构探究收集23名15-24岁青少年的构想，ARC收集数周的讨论记录与反馈。

**📈 对比分析**

在缺乏量化指标的情况下，主要通过定性分析比较：对比传统平台设计与提出的关系支持性设计在青少年表达的满足度、关系维护与隐私感知上的差异；研究未给出具体性能数值，强调的是设计思路与用户体验的可行性。

**⚠️ 局限性**

局限性包括样本规模有限（尤其ARC阶段），自述网络描述的主观性与完整性不确定；LLM代理模拟可能未完全捕捉真实用户行为与群体动态；异步讨论易受参与度波动影响；缺乏跨文化与长期跟踪验证。

---

## 216. Pan-FM: A Pan-Organ Foundation Model with Saliency-Guided Masking for Missing Robustness

**arXiv ID:** 2605.07055 | [PDF](https://arxiv.org/pdf/2605.07055v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 217. Randomness is sometimes necessary for coordination

**arXiv ID:** 2605.06825 | [PDF](https://arxiv.org/pdf/2605.06825v1)

**作者:** Rohan Patil `[一作]` (University of California San Diego), Henrik I. Christensen `[通讯]` (University of California San Diego)

**通讯引用:** 13117 | [OpenAlex ID](https://openalex.org/A5066237365)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

论文提出一种新的跨注意力架构——Diamond Attention，利用每个代理在每个时步采样的随机标量来生成动态的注意力掩码，从而打破同质代理的对称性，实现角色分化并支持全参数共享。

**💡 创新点**

创新点在于将结构化随机掩码嵌入跨注意力中，单轮广播即可产生临时的代理层级关系；该方法在保持参数共享优势的同时，实现了零样本可扩展到不同规模团队，并通过理论证明了随机比特共享是对称破裂的必要条件。

**🔧 技术方法**

技术包括：基于 Transformer 的跨注意力机制、随机标量采样与广播、结构化掩码实现代理对称性破裂、全参数共享与集中训练/去中心化执行 (CTDE)、PPO 训练、以及针对变量行动空间的输出层适配。

**📊 数据集**

使用的数据集主要有：极简 XOR 游戏（离散对称奖励），VMAS（连续空间的 Simple Spread 与 Food Collection 任务），以及 SMACLite（StarCraft II 2s3z 与 3s5z 对抗场景）。

**📈 对比分析**

与 MAPPO、QMIX、IPPO、MASAC、pH-MARL、GSA、MAT 等基线相比，Diamond Attention 在 XOR 游戏中取得 1.0 的成功率（基线仅 0.5），在 VMAS 上实现 N=2~8 的零样本泛化且保持接近最优奖励；在 SMACLite 上零样本转移到更难场景时获得 49.7% 的胜率，而所有基线在此任务中均降为 0%。

**⚠️ 局限性**

主要局限包括：对变量动作空间的支持需通过额外的线性投影层，导致在训练场景下可能略逊；在 XOR 等高维对称任务中，随着代理数增大，PPO 训练因奖励稀疏而难以收敛；假设代理能在单步内广播标量，若换为点对点通信则需要额外的共识机制。

---

## 218. Beyond LoRA vs. Full Fine-Tuning: Gradient-Guided Optimizer Routing for LLM Adaptation

**arXiv ID:** 2605.07111 | [PDF](https://arxiv.org/pdf/2605.07111v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 219. Towards Autonomous Business Intelligence via Data-to-Insight Discovery Agent

**arXiv ID:** 2605.07202 | [PDF](https://arxiv.org/pdf/2605.07202v1)

**作者:** Dongming Wu `[一作]` (Rajax Network Technology(Taobao Shangou of Alibaba)), Ting Chen `[通讯]` (Rajax Network Technology(Taobao Shangou of Alibaba))

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了 AIDA 框架，一种端到端的自治业务分析代理，能够在包含 200+ 指标和 100+ 维度的即时零售环境中，通过专用 DSL 与 LLM 交互，实现数据到洞察的自动发现。

**💡 创新点**

将结构化状态建模、专用 DSL、基于 Pareto 原则的奖励分解、奖励掩码策略以及强化学习相结合，首次实现了在复杂业务环境下的完全自治业务洞察发现，并通过多层奖励机制抑制幻觉与结构错误。

**🔧 技术方法**

采用 Qwen3‑8B 作为后端 LLM，结合 DSL + Python 沙盒工具，使用 REINFORCE + 返回批标准化进行强化学习，设计了中间奖励与累计奖励的分解以及 Schema Masking、Logical Consistency Masking 两种掩码。

**📊 数据集**

在自研即时零售环境中收集 200+ 指标与 100+ 维度的数据，构建训练集 𝒟_s（清洗后的专家轨迹）和 𝒟_rl；评估集为 82 个真实业务案例，涵盖交通、交易、互动、营销与通用五个分析域。

**📈 对比分析**

以 82 个测试案例的累计洞察评分为指标，对比 ReAct、State‑ReAct、AIDA‑SFT；AIDA‑RL 在 50 步内显著高于基线，累计得分提升约 30%，幻觉率降低 70%，在 15 步后已超过 32B 级模型，且在探测维度与边界违规率上表现更佳。

**⚠️ 局限性**

受限于专有 DSL 与企业环境的通用性不足；奖励设计对模型稳定性要求高，需手工调参；RL 训练资源消耗大，模型仍可能在未见过的数据上产生幻觉或逻辑错误；目前仅在零售场景验证，跨域迁移需进一步研究。

---

## 220. Test-Time Compositional Generalization in Diffusion Models via Concept Discovery

**arXiv ID:** 2605.07078 | [PDF](https://arxiv.org/pdf/2605.07078v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 221. Membership Inference Attacks on Vision-Language-Action Models

**arXiv ID:** 2605.07088 | [PDF](https://arxiv.org/pdf/2605.07088v1)

**作者:** Yuefeng Peng `[一作]` (University of Massachusetts Amherst), Amir Houmansadr `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 5497 | [OpenAlex ID](https://openalex.org/A5018588864)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统研究了视觉-语言-动作（VLA）模型的成员推断攻击，提出样本级和轨迹级两种攻击框架，并在不同访问权限下实现多种攻击方法。

**💡 创新点**

创新点在于首次针对VLA模型量化成员推断风险，利用动作误差、时间平滑、曲率等VLA特有信号，构建强大的黑盒攻击，并揭示轨迹级推断可达到近乎完美的泄露率。

**🔧 技术方法**

采用了基于token likelihood、生成置信度、动作误差（L1/MSE）以及轨迹平滑/曲率的攻击技术；同时对比白盒和严格黑盒两种访问模式。

**📊 数据集**

使用了四个LIBERO基准数据集（Spatial、Object、Goal、Long）以及OpenVLA、π_0-fast等代表性VLA模型进行实验。

**📈 对比分析**

通过AUC和TPR@FPR评估，样本级黑盒攻击平均AUC≥0.92；轨迹级攻击平均AUC≈0.999，尤其是时间平滑/曲率攻击在0.1% FPR下TPR接近97%，表明VLA模型极易被识别。

**⚠️ 局限性**

局限性包括：实验仅覆盖有限的VLA模型和基准，缺乏对更大规模或不同动作空间的验证；防御实验仍属初步，未提出针对VLA的成熟隐私防御方案。

---

## 222. Actor-Critic with Active Importance Sampling

**arXiv ID:** 2605.07094 | [PDF](https://arxiv.org/pdf/2605.07094v1)

**作者:** Majid Molaei `[一作]` (Politecnico di Milano), Marcello Restelli `[通讯]` (Politecnico di Milano)

**通讯引用:** 3380 | [OpenAlex ID](https://openalex.org/A5017130830)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了Active-Importance-Sampling Actor-Critic（AISAC）算法，通过主动重要性采样降低策略梯度估计的方差；

**💡 创新点**

创新点在于设计了与目标策略梯度方向和幅值相匹配的行为策略，使数据采集更有效；

**🔧 技术方法**

使用了重要性采样、Actor-Critic框架、Gaussian策略、交叉熵最小化等技术；

**📊 数据集**

在Mujoco的Inverted Pendulum和Half Cheetah连续控制任务上进行实验；

**📈 对比分析**

与传统on‑policy Actor-Critic比较，AISAC在样本效率、学习速度和训练稳定性上均优于基线；

**⚠️ 局限性**

局限性包括仅在连续控制任务上验证，未在高维或离散环境中测试，且对超参数敏感，未与SAC或TD3等高级算法结合。

---

## 223. Don't Retrain, Align: Adapting Autoregressive LMs to Diffusion LMs via Representation Alignment

**arXiv ID:** 2605.06885 | [PDF](https://arxiv.org/pdf/2605.06885v1)

**作者:** Fred Zhangzhi Peng `[一作]` (Duke University), Alexander Tong `[通讯]` (AITHYRA)

**通讯引用:** 1296 | [OpenAlex ID](https://openalex.org/A5053568123)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在保持原有 AR 模型内部表示几何结构的前提下，将其转化为掩码扩散语言模型（DLM）的简洁方法。

**💡 创新点**

创新点在于引入层级余弦对齐损失，直接将 DLM 的隐藏状态与冻结的 AR 教师对齐，从而把 DLM 训练转化为机制适配问题，显著提升训练速度与样本效率。

**🔧 技术方法**

采用掩码扩散训练目标、层级余弦对齐、BERT 风格双向注意力、以及传统的 PAPL 损失；实现过程仅改动注意力掩码，无需额外适配器。

**📊 数据集**

使用 50B 代币的 Nemotron‑SFT‑Code 代码指令集进行训练，并在 HumanEval、MBPP 及其 EvalPlus 版本上评估生成效果。

**📈 对比分析**

相较于仅做掩码扩散训练的基线，加入对齐后在 0.6B、1.7B、4B 模型上 HumanEval pass@10 提升 6–9 点；在 4B 规模下与公开 DLM（如 Dream、LLaDA）相比，参数更少、训练成本更低且取得相当或更优的 pass@10 表现。

**⚠️ 局限性**

局限性包括：需要先有规模可观的 AR 检查点；对齐权重需手工调参；方法在极大规模模型或不同架构下的适用性尚未充分验证；在极端低数据或极短序列下的效果仍需进一步研究。

---

## 224. Weblica: Scalable and Reproducible Training Environments for Visual Web Agents

**arXiv ID:** 2605.06761 | [PDF](https://arxiv.org/pdf/2605.06761v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 225. SREGym: A Live Benchmark for AI SRE Agents with High-Fidelity Failure Scenarios

**arXiv ID:** 2605.07161 | [PDF](https://arxiv.org/pdf/2605.07161v1)

**作者:** Jackson Clark `[一作]` (University of Illinois Urbana Champaign), Tianyin Xu `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了高保真、可扩展的 SREGym 基准，用于评估 AI 代理在真实云原生环境中诊断和缓解生产故障的能力；

**💡 创新点**

创新点包括：①模块化故障/噪声注入框架，支持多层次故障、噪声与复杂失败模式；②统一的 Agent 接口（MCP），可与任意 LLM/代理交互；③对 90 个真实、挑战性 SRE 问题的系统化定义与评估；

**🔧 技术方法**

主要技术为基于 Kubernetes 的云原生系统环境、Python API 的故障/噪声注入器、Prometheus/Loki/Jaeger 等观测接口，以及自定义或现成的 LLM（Sonnet‑4.6、Kimi‑K2.5、GPT‑5.4）驱动的代理；

**📊 数据集**

数据集：90 个动态 SRE 问题构成的基准集，涵盖应用层到硬件层的多种故障、噪声与失败模式；

**📈 对比分析**

评估方法：诊断成功率、缓解成功率、端到端成功率、TTD/TTM、token 计数；实验表明诊断成功率 38.9%–72.6%，缓解 57.3%–78.5%，端到端 63%–60%，噪声可显著降低诊断率，代理在多重失败场景中表现更差；

**⚠️ 局限性**

局限性：基准仅涵盖 90 题，虽然可扩展，但仍未覆盖所有生产故障类型；缺乏强化学习训练环境；部分低层次故障与噪声建模仍不够丰富；

---

## 226. MathlibPR: Pull Request Merge-Readiness Benchmark for Formal Mathematical Libraries

**arXiv ID:** 2605.07147 | [PDF](https://arxiv.org/pdf/2605.07147v1)

**作者:** Zixuan Xie `[一作]` (University of Virginia), Shangtong Zhang `[通讯]` (University of Virginia)

**通讯引用:** 716 | [OpenAlex ID](https://openalex.org/A5033834190)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并发布 MathlibPR 基准，评估大语言模型和代理在审阅 Lean4 Mathlib PR 的“合并准备”判断能力。

**💡 创新点**

①从真实 Mathlib4 PR 历史中自动抽取构建通过但不一定合并的快照，形成正负样本；②设计分阶段评估协议（仅差异、增添诊断、再加 PR 说明），使模型可逐步获取更多审阅上下文；③通过结构化 JSON 输出和多指标（MR/NMR 召回、平衡准确率、有效率、AUROC）量化模型表现。

**🔧 技术方法**

使用多种 LLM 模型（DeepSeek、Qwen、Goedel、Kimina）和 LLM 代理（Codex、Claude Code），对 snapshot 进行文本+代码诊断推理；采用三阶段提示与结构化评审模板；通过内部检索与工具调用（仅代理可访问本地仓库）进一步辅助判断。

**📊 数据集**

MathlibPR 数据集：15,895 条构建通过的 PR 快照，11,409 条已合并正样本，4,486 条未合并或被修改负样本；包含 3,687 对同一 PR 的早期与最终快照用于对比分析；数据源为 2026 年 4 月前 Mathlib4 代码库的关闭 PR。

**📈 对比分析**

对全基准评估 LLM 模型，对 500/500 负正样本子集评估代理；使用 MR Recall、NMR Recall、平衡准确率、有效率、AUROC 作为指标。结果显示：最佳平衡准确率仅 36%（DeepSeek Stage 3），NMR Recall 低至 2%；代理在 MR Recall 上表现相对更好，但 NMR Recall 仍不理想；增加上下文或仓库访问均未显著提升识别负样本能力。

**⚠️ 局限性**

局限性包括：①数据集仅覆盖可恢复构建检查记录的 PR，未涵盖所有 PR；②正负标签来源于历史维护者决定，可能受非技术因素影响；③代理评估受成本限制，只用 500/500 样本，无法与模型结果直接对比；④模型与评审接口匹配度低，导致大量无效输出。

---

## 227. When to Use Wireless Challenge-Response Physical Layer Authentication: Design of a Measurable Guideline for OFDM

**arXiv ID:** 2605.06750 | [PDF](https://arxiv.org/pdf/2605.06750v1)

**作者:** Haiyun Liu `[一作]` (University of South Florida), Zhuo Lu `[通讯]` (University of South Florida)

**通讯引用:** 3395 | [OpenAlex ID](https://openalex.org/A5019612445)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种新的对无线挑战-响应物理层认证（PLA）的攻击模型MDLG，并基于随机性测试给出可量化的安全使用准则

**💡 创新点**

首次将信道相关性与攻击概率定量关联，定义MDLG攻击模型，并用随机性测试构造可实现的安全阈值

**🔧 技术方法**

利用OFDM信道估计、相位差分量、二值化、MDLG逐位生成候选方案、贝塔分布概率分析、NIST随机性测试

**📊 数据集**

在室内使用Atheros AR5822/AR9580芯片和TP‑Link WDR4300 AP 采集约5×10⁵个OFDM信道响应数据，涵盖S‑LoS、S‑NLoS、L‑LoS、L‑NLoS等四种环境

**📈 对比分析**

与随机猜测对比，MDLG在各种信道相关性和攻击者计算能力下成功率可高达10¹²倍；随机性测试可显著降低成功率并通过优化阈值平衡效率与安全；实验验证准则可将攻击概率降至10⁻⁴以下

**⚠️ 局限性**

对MDLG的分析假设信道相关性为i.i.d.，未考虑多跳、干扰或多天线系统；随机性测试仅针对特定NIST测试，可能在极端环境下误判；未给出对实时实现的复杂度估计

---

## 228. The E$Δ$-MHC-Geo Transformer: Adaptive Geodesic Operations with Guaranteed Orthogonality

**arXiv ID:** 2605.06729 | [PDF](https://arxiv.org/pdf/2605.06729v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 229. Modular Lie Algebraic PDE Control of Multibody Flexible Manipulators

**arXiv ID:** 2605.06709 | [PDF](https://arxiv.org/pdf/2605.06709v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 230. Domain-level metacognitive monitoring in frontier LLMs: A 33-model atlas

**arXiv ID:** 2605.06673 | [PDF](https://arxiv.org/pdf/2605.06673v1)

**作者:** Jon-Paul Cacioli `[一作]` `[通讯]` (Independent Researcher), Jon-Paul Cacioli (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在MMLU基准上，对33个前沿LLM进行1,500道题（按六个先验域分层）的持续性置信度评估，计算每个模型-域单元的Type-2 AUROC，并通过bootstrap CI、分半稳定性、家族内相似性和代际变化等统计手段构建了LLM元认知监控的“域级档案”。

**💡 创新点**

①首次在标准化大样本上量化LLM在不同认知域的元认知监控差异；②揭示了Applied/Professional域始终最易监控，Formal Reasoning和Natural Science域最难；③发现部分模型家族（Anthropic、Google‑Gemini、Qwen）在域形状上高度一致；④展示了Gemma 3→Gemma 4跨代显著提升的监控质量；⑤证明置信度的有效性与测量方式（二元probe vs 连续口头置信）相关。

**🔧 技术方法**

采用Type-2 AUROC评估置信度与正确率的关联；使用bootstrap 95% CI估计单元不确定性；进行分半相关检验验证聚合与域形稳定性；计算IPSAT（ipsative）域向量的Pearson相关检验家族内聚类；利用Friedman检验与Kendall’s W确定域难度等级；通过线性回归和Spearman相关排除准确率偏倚；使用多种统计软件包实现完整的可复现分析流程。

**📊 数据集**

MMLU 1,500题（每域250题），来自HuggingFace datasets，采用先验域映射（Applied/Professional, Factual, Formal, Humanities, Natural Science, Social）。33个LLM（来自Anthropic、DeepSeek、Google‑Gemini、Gemma、OpenAI、Qwen、Zhipu、GLM-5），共47,151条观察记录。数据、代码及图表公开于GitHub。

**📈 对比分析**

对每个模型-域单元计算Type-2 AUROC，使用1,000次bootstrap获得CI；采用分半检验验证聚合AUROC稳定性（r=0.893）；对IPSAT域向量计算Pearson相关，评估家族内相似性（Anthropic r≈0.455，Gemini r≈0.511，Qwen r≈0.472，其他家族不显著）。生成域难度层级（Applied最高，Formal/Science最低，三中间域差异不显著）。模型平均AUROC范围0.498–0.806，应用/专业域平均0.742，Formal 0.658，Science 0.652。表现表明域级监控差异显著且可检验，但聚合指标掩盖了大部分信息。

**⚠️ 局限性**

• 仅使用单一基准（MMLU）且域划分为先验、缺乏因子验证；• 置信度仅采用连续口头置信（0–100），未探讨二元或概率日志等其它形式；• 采用贪婪解码（temperature=0），可能影响置信分布；• bootstrap CI宽度在高精度模型中仍大于0.25，导致细域差异不显著；• 未直接分析隐藏状态或机制，仅提供假设；• 仅评估一次，未检验测试-重测稳定性；• 部分模型数据缺失或API中断导致样本不均匀。

---

## 231. MAGIQ: A Post-Quantum Multi-Agentic AI Governance System with Provable Security

**arXiv ID:** 2605.06933 | [PDF](https://arxiv.org/pdf/2605.06933v1)

**作者:** Sepideh Avizeh `[一作]` (University of Calgary), Reihaneh Safavi-Naini `[通讯]` (University of Calgary)

**通讯引用:** 7938 | [OpenAlex ID](https://openalex.org/A5010447902)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 MAGIQ，一个面向多代理 AI 系统的后量子安全治理框架，支持用户定义丰富的通信与访问控制预算、在会话层面强制执行这些策略，并通过可证明的协议为代理行为提供可追溯的责任归属。

**💡 创新点**

创新点包括：① 在代理级别引入 A‑session 与 C‑session 两种通信抽象，允许按任务语义定义预算；② 采用 hash 链与 Merkle 树实现高效的计数与预算绑定，完全基于后量子安全原语；③ 在 UC 框架下给出完整的安全证明，并对多代理场景的可扩展性与成本进行系统评估。

**🔧 技术方法**

核心技术：后量子签名（XMSS_SHA2_16_256）、后量子 TLS（PQ‑TLS）与 ML‑DSA、ML‑KEM、HMAC、PRF、hash 链、Merkle 树；协议层使用数字签名、消息认证码和共享密钥；系统模型采用 UC 形式化与全球随机预言机/时钟假设。

**📊 数据集**

实验评估以三类 LLM 任务（日历、邮件、写作）为基础，使用 GPT‑4.1、GPT‑5.4、Qwen2.5‑72B、Qwen3.5‑30B 等公开模型；不涉及传统机器学习数据集，重点测量协议计算/通信开销与任务完成时间。

**📈 对比分析**

与传统基于 Curve25519 的基线系统相比，MAGIQ 的用户/代理注册、会话握手与数据传输均保持在毫秒级，协议总开销仅比基线高约 1–2×；在不同地理位置与 Q_max 规模下，协议延迟与基线几乎无差异；任务执行时间中协议开销占比小于 2%。

**⚠️ 局限性**

局限性：① 仅支持静态策略与静态工作计划，无法动态适应任务变化；② 需要全局同步时钟与可信 CA；③ 对大规模代理网络的可扩展性（如数千个并发会话）未在实验中验证；④ 只评估了少量 LLM 后端，对更复杂的多代理协作流程仍需进一步研究。

---

## 232. Self-Programmed Execution for Language-Model Agents

**arXiv ID:** 2605.06898 | [PDF](https://arxiv.org/pdf/2605.06898v1)

**作者:** Luke J. O'Connor `[一作]` `[通讯]` (Harvard Medical School), Luke J. O'Connor (Harvard Medical School)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种无固定 orchestrator 的自程序化执行（SPE）架构，让语言模型直接执行其自身生成的程序完成多轮任务。

**💡 创新点**

创新点在于将 orchestrator 完全交给模型编写的程序，提出 SPE 状态的形式化定义，并构建了支持自我编辑、无副作用重评估的 Spell Lisp‑style 语言。

**🔧 技术方法**

技术方案包括 CEK 机器的模型调用包装器、Lisp/Clojure 语法、尾表达式模式、quine 自引用以及 Spell 语言实现自程序化执行。

**📊 数据集**

实验使用公开基准：Terminal‑Bench 1.1、SWE‑bench Lite（32 题子集）、LongBench v2、AppWorld 等。

**📈 对比分析**

与 Codex CLI、Claude Code 等传统 harness 对比，Spell 代理在编码基准上表现相当或略逊但成本更低；在长上下文问答和 AppWorld 上效果不如传统 harness，体现出成本–准确度的权衡。

**⚠️ 局限性**

限制包括模型需要先学习 Spell 语言，当前模型易产生无效程序且缺乏专门训练，且对复杂多代理或长任务的适配尚未成熟。

---

## 233. MedAction: Towards Active Multi-turn Clinical Diagnostic LLMs

**arXiv ID:** 2605.07305 | [PDF](https://arxiv.org/pdf/2605.07305v1)

**作者:** Hsin-Ling Hsu `[一作]` (National Chengchi University), Liyue Shen `[通讯]` (University Of Michigan)

**通讯引用:** 6484 | [OpenAlex ID](https://openalex.org/A5072483985)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一个面向主动诊断的多轮训练数据生成框架MedAction，解决了LLM在逐步获取证据并更新诊断的能力缺失问题。

**💡 创新点**

创新点包括：①将医学案例转化为交互式环境，实现模型与环境的多轮对话；②提出两种基于知识图谱的质量指标Disease Trajectory Consistency (DTC) 与 Reasoning–Action Consistency (RAC) 用于筛选高质量诊断轨迹；③结合多教师、多分支采样与过滤策略生成多样化、可解释的训练数据。

**🔧 技术方法**

采用LLM交互式推理、树结构轨迹采样、知识图谱匹配与最短路径距离计算、监督微调（SFT）等技术。

**📊 数据集**

使用公开的PMC病例报告构建交互环境，生成MedAction-32K（32,681条轨迹）和MedAction-300-Hard（300条稀有病案例）数据集。

**📈 对比分析**

在MedR‑Bench和MedAction-300‑Hard上进行评估，8B模型在开放源代码中达到最高诊断准确率和测试推荐F1，相比同尺寸模型提升约10–20%（例如诊断准确率从0.51升至0.69）。

**⚠️ 局限性**

局限性包括：依赖于知识图谱的映射准确性，过滤策略可能丢弃部分有价值的中途轨迹；缺乏真实临床验证，模型在复杂症例下仍可能出现非专业建议。

---

## 234. Decentralized Diffusion Policy Learning for Enhanced Exploration in Cooperative Multi-agent Reinforcement Learning

**arXiv ID:** 2605.07101 | [PDF](https://arxiv.org/pdf/2605.07101v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 235. When Does Critique Improve AI-Assisted Theoretical Physics? SCALAR: Structured Critic--Actor Loop for Agentic Reasoning

**arXiv ID:** 2605.06772 | [PDF](https://arxiv.org/pdf/2605.06772v1)

**作者:** Vasilis Niarchos `[一作]` (University of Crete), Sokratis Trifinopoulos `[通讯]` (CERN)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了SCALAR框架，测试LLM代理在量子场论和弦论问题中的批判–行动循环。

**💡 创新点**

首次系统评估不同演员角色、批判反馈策略和模型规模对多轮交互性能的影响。

**🔧 技术方法**

使用多模型LLM（DeepSeek、Claude Sonnet/Haiku、QWQ）、预设角色与反馈提示、Actor–Critic–Judge管道。

**📊 数据集**

采用三道经典教材问题（Peskin 2.3/4.2、Polchinski 2.7）作为测试集。

**📈 对比分析**

通过平均得分、增益、收敛率等指标对比，发现多轮对话普遍提升，Actor–Critic配对与反馈策略对结果影响显著，模型规模提升对简单问题有利但难题瓶颈不消失。

**⚠️ 局限性**

缺乏无参考验证、对硬题仍易卡住、批判策略效应在不同模型间不一致、样本覆盖有限。

---

## 236. On the Complexity of the Matching Problem of Regular Expressions with Backreferences

**arXiv ID:** 2605.07289 | [PDF](https://arxiv.org/pdf/2605.07289v1)

**作者:** Soh Kumabe `[一作]` (CyberAgent, Inc.), Yuya Uezato `[通讯]` (CyberAgent, Inc.)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并解决了正则表达式中带回引用的字符串匹配问题，尤其是形如 A (B)_x C x D 的 ABCBD 问题，并给出了高效的求解算法。

**💡 创新点**

创新点在于将 ABCBD 问题拆解为 XYYZ 子问题和分支 ABCBD 子问题，利用最大局部幂、因子森林和后缀树的重力路径枚举，实现了近线性时间复杂度的解决方案；同时给出了基于 SETH 和 k‑OV 假设的下界证明，表明一般情况难以突破此复杂度。

**🔧 技术方法**

使用的核心技术包括 Crochemore 的最大局部幂算法、因子森林（factorization forest）数据结构、后缀树的重力路径分解、周期性集合（ultimately periodic set）分析、Minkowski 和与 2‑倍数运算，以及枚举问题的组合策略。

**📊 数据集**

本研究主要为理论算法，没有使用公开的数据集，所有验证均在构造的合成字符串上完成。

**📈 对比分析**

相较于传统的指数级或 O(n²) 的正则表达式匹配方法，提出的算法在满足约束条件下实现了 O(n log n) 或 O(n log |w|) 的时间复杂度，显著提升了性能。

**⚠️ 局限性**

局限性包括：仅适用于二进制字母表且满足特定前缀约束的正则表达式；对更一般的带回引用表达式全局求解仍保持指数级难度，且算法中的常数因子可能较大。

---

## 237. Delulu: A Verified Multi-Lingual Benchmark for Code Hallucination Detection in Fill-in-the-Middle Tasks

**arXiv ID:** 2605.07024 | [PDF](https://arxiv.org/pdf/2605.07024v1)

**作者:** Mahdi Erfanian `[一作]` (University of Illinois Chicago), Shengyu Fu `[通讯]` (Microsoft)

**通讯引用:** 1746 | [OpenAlex ID](https://openalex.org/A5031659418)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个经过多轮对抗筛选、执行验证和人工审核的 1,951 条 FIM (Fill‑in‑the‑Middle) 代码补全样本的多语言、四类幻觉（方法、参数、未定义变量、导入）基准，并对 11 个开源 LLM 进行评测；同时评估了 LLM 作为代码审阅器的幻觉检测能力。

**💡 创新点**

① 将幻觉类型细分为可执行错误的四类，保证可通过 Docker 执行检测；② 采用五阶段对抗生成、判别、难度挖掘、筛选和容器验证的管线，消除标签噪声和生成器偏差；③ 将基准同时用于生成性能评估和判别任务，展示模型在生成与检测两侧均存在显著差距。

**🔧 技术方法**

对抗生成与判别（使用 GPT‑4o‑mini、Claude‑4.5‑Opus 等前沿 LLM 进行生成和判别）；多语言 Docker 环境用于编译与运行错误捕获；Embedding‑based 聚类与难度筛选；链式思考提示 + 二元评分用于判别；对比度量包括 Pass@1、Exact Match、Edit Similarity、CodeBLEU 与 Hallucination Rate。

**📊 数据集**

基于公开 GitHub 代码的 319 个仓库，采样 7 种语言（Python、TypeScript、Go、Java、C#、Rust、C++），按四类幻觉生成 1,951 条已执行验证的样本；与现有 HumanEval、SAFIM 等基准对比。

**📈 对比分析**

结果显示：Qwen2.5‑Coder 32B 最高 Pass@1 84.5%，但仍有 15% 的 FIM 上下文失败；跨家族模型 (CodeLlama、DeepSeek、StarCoder2) 最高 82.7%，说明难度与模型族无关；幻觉检测方面，Claude‑4.5‑Opus 达到 92.1% 两者正确率，仍存在 8% 的误判，显示判别任务仍具挑战。

**⚠️ 局限性**

① 样本数量在某些 (语言、幻觉类型) 细胞 <50 只作参考；② 仅支持单文件、无跨文件依赖的 FIM 上下文；③ 只覆盖产生可执行错误的四类幻觉，逻辑、类型、过时 API 等其他幻觉不在内，导致整体幻觉覆盖率为下限；④ 数据集基于 2025 年 Oct 的仓库快照，强调仅作测试使用，禁止用于微调。

---

## 238. Fast and Effective Redistricting Optimization via Composite-Move Tabu Search

**arXiv ID:** 2605.06682 | [PDF](https://arxiv.org/pdf/2605.06682v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 239. DiffRetriever: Parallel Representative Tokens for Retrieval with Diffusion Language Models

**arXiv ID:** 2605.07210 | [PDF](https://arxiv.org/pdf/2605.07210v1)

**作者:** Shuai Wang `[一作]` (University of Queensland), Guido Zuccon `[通讯]` (University of Queensland)

**通讯引用:** 4964 | [OpenAlex ID](https://openalex.org/A5076031002)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于扩散语言模型的代表性词检索器（DiffRetriever），通过在提示中插入 K 个掩码位置一次性生成多条查询/文档表示，并结合稠密与稀疏两种检索信号。

**💡 创新点**

创新点在于：①利用扩散模型的并行掩码预测实现多代表词检索，解决了自回归模型多词检索的顺序生成瓶颈；②证明多词检索在扩散模型上既有效又低延迟；③通过预算选择和每查询最优预算的上界分析，展示了可进一步提升的空间。

**🔧 技术方法**

主要技术包括：扩散语言模型（Dream、LLaDA）、代表性词提示、ColBERT 风格的稠密 MaxSim 与稀疏词典匹配、对比学习微调（LoRA）、稠密与稀疏融合得分、对比分析与可解释性可视化。

**📊 数据集**

使用的数据集有：MS MARCO passage 排序（dev）、TREC DL 2019/2020（NDCG@10）、BEIR-7（七个多领域检索任务，NDCG@10）。

**📈 对比分析**

通过与 BM25、原始代表词检索、DiffEmbed（均匀池化）以及 RepLLaMA 等基线进行对比。零样本阶段，DiffRetriever 在 LLaDA 上获得最高 BEIR-7 分数；微调后，Dream‑DiffRetriever 在 BEIR-7 上达到了最强表现（平均 NDCG@10≈0.671），且比对比学习微调的单向自回归模型更快（单前向传递而非多步）。

**⚠️ 局限性**

局限性包括：仅在 7–8B 规模的两种扩散模型上验证；未覆盖完整 BEIR 族；实验成本高（数千 GPU 时、20TB 存储）；未实现可部署的预算路由器，缺乏对每查询动态预算选择的实际实现。

---

## 240. PRIMED: Adaptive Modality Suppression for Referring Audio-Visual Segmentation via Biased Competition

**arXiv ID:** 2605.07154 | [PDF](https://arxiv.org/pdf/2605.07154v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 241. A Reproducible Optimisation Protocol for Calibrating Prompt-Based Large Language Model Workflows in Evidence Synthesis

**arXiv ID:** 2605.06937 | [PDF](https://arxiv.org/pdf/2605.06937v1)

**作者:** Teo Susnjak `[一作]` `[通讯]`, Teo Susnjak

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套可复现的提示基准化工作流，使用DSPy和GEPA自动优化结构化证据综合任务（如标题/摘要筛选）的提示，并将校准后的程序保存为可复用工件。

**💡 创新点**

创新点在于将科学任务合同与可变提示分离，构建可执行指标作为优化目标，自动化提示校准并将校准结果打包为完整的、可追溯的科研工件。

**🔧 技术方法**

技术包括DSPy（声明式LLM编程）、GEPA（基于指标的提示优化）、可执行度量函数、JSON工件保存和复现、以及对学生模型与反思模型的温度调节。

**📊 数据集**

使用了SESR‑Eval 41研究的标题/摘要筛选数据集，包含1194条候选记录作为验证基准。

**📈 对比分析**

通过在不同的max_eval预算（2、6、12、24）下运行GEPA，对比未优化的DSPy程序，最佳预算（12）在保持召回率约0.905的同时，使准确率提升至0.797、F1提升至0.848（约+0.8%）。低预算侧重召回，高预算则出现轻微性能下降。

**⚠️ 局限性**

局限性包括：仅验证单一二元筛选任务、单一学生模型、样本量有限，易导致过拟合；提示校准不替代人工判断，需专家监督；不同任务、模型和数据集需要重新设计提示、指标和验证集。

---

## 242. PISTO: Proximal Inference for Stochastic Trajectory Optimization

**arXiv ID:** 2605.07215 | [PDF](https://arxiv.org/pdf/2605.07215v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 243. PerCaM-Health: Personalized Dynamic Causal Graphs for Healthcare Reasoning

**arXiv ID:** 2605.07267 | [PDF](https://arxiv.org/pdf/2605.07267v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 244. From Pixels to Primitives: Scene Change Detection in 3D Gaussian Splatting

**arXiv ID:** 2605.07203 | [PDF](https://arxiv.org/pdf/2605.07203v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 245. Beyond the Wrapper: Identifying Artifact Reliance in Static Malware Classifiers using TRUSTEE

**arXiv ID:** 2605.07034 | [PDF](https://arxiv.org/pdf/2605.07034v1)

**作者:** Riyazuddin Mohammed `[一作]` (Northern Arizona University), Lan Zhang `[通讯]` (Northern Arizona University)

**通讯引用:** 360 | [OpenAlex ID](https://openalex.org/A5100322329)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过 TRUSTEE 解释器结合手工字节级分析，系统性评估静态恶意软件分类器是否仅依赖打包、编译器或证书等非语义特征。

**💡 创新点**

提出可复现的两步框架：① 在多比例、最小重叠的数据集上训练模型并使用 TRUSTEE 提取稳定特征；② 对提取特征进行手工字节级验证，揭示模型对数据偏差的敏感性。

**🔧 技术方法**

使用 TRUSTEE 解释器、TabNet 表格深度学习模型、XGBoost 传统树模型，以及手工字节级 PE 文件分析。

**📊 数据集**

主要数据集为 50,724 例自定义数据集（包含 UB、PB、UM、PM 四类），并使用 EMBER 标准数据集进行交叉验证。

**📈 对比分析**

在不同数据集比例下多次训练模型，利用 TRUSTEE 提取重要特征并手工分类，发现 85-90% 的重要特征为非语义；相同结论在 TabNet 与 XGBoost、以及自定义与 EMBER 数据集上均成立，验证了结果的稳健性。

**⚠️ 局限性**

局限性包括仅关注静态特征，手工分析耗时且难以大规模自动化；TRUSTEE 在极端噪声或新型打包技术下的解释稳定性仍需改进；未验证对动态行为或更复杂攻击的泛化能力。

---

## 246. R$^3$L: Reasoning 3D Layouts from Relative Spatial Relations

**arXiv ID:** 2605.06758 | [PDF](https://arxiv.org/pdf/2605.06758v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 247. Geometric Kolmogorov--Arnold Network (GeoKAN)

**arXiv ID:** 2605.06740 | [PDF](https://arxiv.org/pdf/2605.06740v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 248. Big AI's Regulatory Capture: Mapping Industry Interference and Government Complicity

**arXiv ID:** 2605.06806 | [PDF](https://arxiv.org/pdf/2605.06806v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 249. From Surface Learning to Deep Understanding: A Grounded AI Tutoring System for Moodle

**arXiv ID:** 2605.06963 | [PDF](https://arxiv.org/pdf/2605.06963v1)

**作者:** Anna Ostrowska `[一作]` (Warsaw University of Technology), Anna Wróblewska `[通讯]` (Warsaw University of Technology)

**通讯引用:** 651 | [OpenAlex ID](https://openalex.org/A5031984813)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究开发了一个模块化的 Moodle 插件——AI 教学与学习助手，利用检索增强生成（RAG）实现教师监督下的高质量对话式教学与自动化评测。

**💡 创新点**

创新点在于将 RAG 与“人机循环”工作流相结合，系统将大语言模型的回答严格锚定于教师提供的教材，同时提供 Socratic 方法和布卢姆认知层级的评测框架。

**🔧 技术方法**

采用了 FastAPI + Python 后端、Next.js + React 前端、PHP Moodle 活动模块、PostgreSQL/ChromaDB/MinIO 三层架构，RAG 流程使用 Gemini Flash 2.0、MiniLM-L6-v2 语义检索及向量数据库。

**📊 数据集**

主要数据来源为教师上传的课程资料（PDF、DOCX、PPTX 等）以及 18 名学生和教师的使用日志和反馈；评估以 NLP 与 ML 课程笔记为例进行实验。

**📈 对比分析**

通过 RAGAS 评估，系统在 STEM 领域达 0.97 的信仰度，检索 Top‑K=10；与 GPT‑4o‑mini 与 Llama 3.1 对比，Gemini Flash 2.0 在速度/成本比和信仰度上更优，用户满意度平均 4.00/5。

**⚠️ 局限性**

局限性包括需教师人工审核后方可发布、对多模态内容支持不足、对非 Moodle LMS 的集成仍需扩展以及在极大规模课程时检索性能可能下降。

---

## 250. The Translation Tax Is Not a Scalar: A Counterfactual Audit of English-Source Cue Inheritance in Chinese Multilingual Benchmarks

**arXiv ID:** 2605.07093 | [PDF](https://arxiv.org/pdf/2605.07093v1)

**作者:** Zezheng Lin `[一作]`, Handi Li `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在英文到中文的多语言基准上，作者通过四种代理估计器（回译差距、双语注释校准、原生控制对比、LLM自然化压测）系统性审计了翻译税的存在与规模，得出翻译税并非单一标量，而是与估计器和题目残余度相关的有效性风险。

**💡 创新点**

提出了“翻译税”概念、四个互补代理估计器、基于翻译线索可识别性的报告清单，以及基于LLM自然化的压测方法，首次实现对翻译引发的评估偏差进行多维度三角验证，并揭示了高残余度题目在自然化时显著受益的剂量响应。

**🔧 技术方法**

使用商业LLM进行回译并以BLEU做质量控制；采用LLM重写与LLM验证器构建自然化对照；采用5点Likert法的LLM注释校准；利用Bootstrap项集聚类进行统计推断；进行解析器有效率审计；通过配对比较和差异分析评估模型表现。

**📊 数据集**

MMMLU中文子集（228条样本，覆盖14k题），Belebele中文子集（100条样本），INCLUDE中文原生非翻译对照（545条题）以及完整的MMMLU/Belebele原始数据。

**📈 对比分析**

通过回译差距、原生控制差距、注释相关性以及自然化压测等多种对照，发现大部分估计器给出的翻译税效应为正但幅度极小（0–0.05分），且高残余度题目在自然化后得分提升显著；整体而言模型对翻译引入的短路效果有限，未出现显著的模型族交互。

**⚠️ 局限性**

局限性包括：样本规模有限、仅覆盖英→中单一语言对、代理估计器无法直接测得真实翻译税、LLM自然化和验证依赖机器而非人类、注释校准仅使用单一LLM评审、部分模型在自然化条件下解析器失败、缺乏完整的项级双语对照，导致结论主要揭示趋势而非精确定量。

---

## 251. Relating the Computational and Logical Difficulty of Solving ODEs: From Polynomial to Discontinuous Right-Hand Sides

**arXiv ID:** 2605.07128 | [PDF](https://arxiv.org/pdf/2605.07128v1)

**作者:** Olivier Bournez `[一作]` (École polytechnique), Alonso Núñez `[通讯]` (École polytechnique)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

对常微分方程的求解难度进行理论分类，建立了从计算和逻辑层面对应的阶层。

**💡 创新点**

首次将逆数学中的 Big Five 层级与 ODE 存在性、唯一性及可定义性命题直接对应，揭示正则性假设为算法难度的内在不变量。

**🔧 技术方法**

采用逆数学、可计算分析、连续时间计算模型与复杂度理论相结合的证明技术。

**📊 数据集**

无实测数据集，全部为理论构造与数学证明。

**📈 对比分析**

通过逻辑等价与可计算性界定，将不同表示方式下的求解器性能映射到对应的逻辑系统，表明在低层次可实现多项式时间求解，而在高层次则需要超图像递归或可算分析所需的更强归纳原理。

**⚠️ 局限性**

仅针对初值问题的常微分方程，尚未扩展到偏微分方程、延迟或微分代数系统；理论结果对实际符号求解器的细节仍不直接可量化。

---

## 252. Teaching Language Models to Think in Code

**arXiv ID:** 2605.07237 | [PDF](https://arxiv.org/pdf/2605.07237v1)

**作者:** Hyeon Hwang `[一作]` (Korea University), Jaewoo Kang `[通讯]` (Korea University)

**通讯引用:** 16225 | [OpenAlex ID](https://openalex.org/A5076917278)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了ThinC框架，使模型在一次简短的自然语言规划后通过一系列代码块进行推理，训练出了ThinC-1.7B和ThinC-4B两种模型。

**💡 创新点**

创新点在于将代码本身作为核心推理器，拆分自然语言与代码的角色，解决了传统工具集成推理（TIR）的三大局限，并通过轨迹蒸馏+监督微调+强化学习实现了代码中心化推理。

**🔧 技术方法**

主要技术包括代码轨迹蒸馏、监督微调（SFT）、基于可验证奖励的GRPO强化学习、Python解释器及其科学计算库、以及Qwen系列LLM作为基础模型。

**📊 数据集**

使用了12.2k条代码中心化推理轨迹（ThinC-SFT）采自Skywork-OR1与OpenMathReasoning，RL阶段使用DAPO-Math-17k；评测集为AIME 2024–2026、HMMT 2025和BeyondAIME。

**📈 对比分析**

通过在每个基准上采样16条推理轨迹并报告avg@16，对比NL-only与TIR基线；ThinC-4B在5个竞赛级数学基准上平均达到78.1%准确率，超过所有TIR基线以及更大模型Qwen3-235B-A22B-Thinking，ThinC-1.7B亦显著优于基线。

**⚠️ 局限性**

局限性在于实验仅限于1.7B和4B参数规模，评估领域仅限竞赛级数学，尚未验证在更大模型或其他工具集成推理场景的可扩展性。

---

## 253. Bounding Fixed Points of Non-Monotone Processes: Theory to Practice

**arXiv ID:** 2605.06803 | [PDF](https://arxiv.org/pdf/2605.06803v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 254. Extracting Search Trees from LLM Reasoning Traces Reveals Myopic Planning

**arXiv ID:** 2605.06840 | [PDF](https://arxiv.org/pdf/2605.06840v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 255. DINO-MVR: Multi-View Readout of Frozen DINOv3 for Annotation-Efficient Medical Segmentation

**arXiv ID:** 2605.07221 | [PDF](https://arxiv.org/pdf/2605.07221v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 256. The Context Gathering Decision Process: A POMDP Framework for Agentic Search

**arXiv ID:** 2605.07042 | [PDF](https://arxiv.org/pdf/2605.07042v1)

**作者:** Chinmaya Kausik `[一作]` (University of Michigan), Nathan Kallus `[通讯]` (Netflix)

**通讯引用:** 3275 | [OpenAlex ID](https://openalex.org/A5036921114)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将LLM代理的交互循环正式建模为Context Gathering Decision Process (CGDP)，并提出基于谓词的信念状态与程序化枯竭门两种可插拔改进方法。

**💡 创新点**

创新点在于将迭代搜索抽象为CGDP，并通过分离控制流程与表示实现可插拔的信念状态管理与停止判定，解决传统代理隐式记忆导致的“目标漂移”和“无效循环”。

**🔧 技术方法**

技术手段包括近似Thompson Sampling的LLM策略、Predicate‑Based Adaptive Identification (PBAI) 算法、结构化/自由文本提取器、基于相似度与新颖度的程序化门控，以及多任务检索增强生成框架。

**📊 数据集**

使用的公开数据集有 LoCoMo (对话式问答)、MuSiQue (多跳推理) 与 SWE‑QA‑Pro (代码仓库问答)。

**📈 对比分析**

在四种主流代理框架（IRCoT、ReAct、MemGPT、Iter‑RetGen）上与基线、孤立、PBBS 结构化、PBBS 自由文本等四种内存设置对比，实验显示 PBBS 能提升多跳推理 11.4% 并将 Token 消耗降低 39%，枯竭门则将准确率提升至 +6.9%，验证了方法的有效性。

**⚠️ 局限性**

局限性包括：仅在单一 LLM 模型上验证，跨模型推广未探究；聚焦长时序检索任务，对代码/工具执行等其他场景的适用性待验证；CGDP 作为经验框架，缺乏理论上界与最优性证明。

---

## 257. MEMOREPAIR: Barrier-First Cascade Repair in Agentic Memory

**arXiv ID:** 2605.07242 | [PDF](https://arxiv.org/pdf/2605.07242v1)

**作者:** Yang Zhao `[一作]`, Yue Xiu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 MemoRepair 机制，解决代理记忆中因源节点删除、修正或迁移导致的派生状态级联失效问题。

**💡 创新点**

创新点在于定义了“撤销-先行”级联修复契约，并将验证成功的派生出版问题转化为最大权前闭包，利用单次 s–t 最小割高效求解。

**🔧 技术方法**

使用了有向因果图（影响边与语义边）、预检验证、修复操作符以及基于 min‑cut 的前闭包选择器技术。

**📊 数据集**

在 ToolBench 与 MemoryArena 两个基准上进行实验，利用工具调用轨迹、缓存输出、可执行过程等数据集。

**📈 对比分析**

与多种内部修复策略（如 Remove all、Repair all）以及六个外部记忆系统对比，MemoRepair 在保留约 90–95% 的修复成功率的同时，将修复成本降低至约 0.6，较 Repair all 降低约 40% 的修复操作量。

**⚠️ 局限性**

局限在于对完整影响因果记录和验证覆盖的强依赖；缺失影响边或验证不足会导致泄露或误修复。

---

## 258. The Causally Emergent Alignment Hypothesis: Causal Emergence Aligns with and Predicts Final Reward in Reinforcement Learning Agents

**arXiv ID:** 2605.06746 | [PDF](https://arxiv.org/pdf/2605.06746v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 259. Multi-Objective Multi-Agent Bandits: From Learning Efficiency to Fairness Optimization

**arXiv ID:** 2605.06864 | [PDF](https://arxiv.org/pdf/2605.06864v1)

**作者:** John Wang `[一作]` (University of Massachusetts Amherst), Mengfan Xu `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5083293925)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究了多目标多智能体多臂赌博机(MO-MA-MAB)，提出了Pareto UCB1 Gossip和Simulated NSW UCB Gossip两种去中心化算法，既实现效率学习也兼顾公平性；

**💡 创新点**

创新点在于：① 将Pareto UCB1扩展到异质奖励与时间变化的去中心化网络，并设计了将采样误差与共识误差分离的探索半径；② 将Nash Social Welfare引入MO-MA-MAB，结合奖励模拟与Gossip实现公平学习；

**🔧 技术方法**

采用的技术包括Gossip通信协议、UCB探索机制、贝叶斯假设的子高斯奖励分布、Erdos–Renyi随机图、偏好向量仿真以及分布式统计估计；

**📊 数据集**

实验使用了合成数据：多智能体N、K臂、D维向量奖励，偏好向量从Dirichlet分布采样，奖励为Bernoulli分布，并对均值添加随机扰动；

**📈 对比分析**

与两类基线（独立Pareto UCB1、Gossip Successive Elimination、无Gossip或无奖励模拟版本）进行对比，实验表明Pareto UCB1 Gossip在Pareto regret上达到对数级别收敛，Simulated NSW UCB Gossip在公平（NSW）指标上显著优于基线，效率提升约100%/50%；

**⚠️ 局限性**

局限性包括：① 在公平约束下，Simulated NSW UCB Gossip的理论收敛率为T^{3/4}，无法达到√T；② 网络稀疏或收敛慢会显著增加共识误差；③ 只在理想化假设下给出理论分析，实际鲁棒性和对参数敏感性仍待进一步验证。

---

## 260. Cost-Ordered Feasibility for Multi-Armed Bandits with Cost Subsidy

**arXiv ID:** 2605.07171 | [PDF](https://arxiv.org/pdf/2605.07171v1)

**作者:** Ishank Juneja `[一作]` (Carnegie Mellon University), Osman Yağan `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2384 | [OpenAlex ID](https://openalex.org/A5064066193)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了成本补贴多臂老虎机（MAB-CS），目标是最小化成本同时满足相对未知最优奖励的质量阈值。

**💡 创新点**

提出了实例依赖的下界和联合样本下界，并设计了Cost‑Ordered Feasibility (COF) 算法，利用门控臂和置信区间实现最优样本分配。

**🔧 技术方法**

使用信息理论下界推导、置信区间聚合、最佳臂识别过滤、可组合样本与排他采样等技术，并给出成本与质量遗憾的上界。

**📊 数据集**

在MovieLens和Goodreads推荐系统数据集以及人工合成实例上进行实验。

**📈 对比分析**

与ETC‑CS、PE‑CS、UCB‑CS、TS‑CS等基线对比，COF 在两类数据集和多种补贴因子下均取得更低的累计成本和质量遗憾，尤其在大α时表现最优；在小α时仅略逊于PE‑CS。

**⚠️ 局限性**

局限在于对昂贵臂的样本下界仍有改进空间，且COF对成本分布不均的情形（如LLM路由）尚未优化，未来需研究成本偏置变体和上下文化改进。

---

## 261. LLMs are not (consistently) Bayesian: Quantifying internal (in)consistencies of LLMs' probabilistic beliefs

**arXiv ID:** 2605.06915 | [PDF](https://arxiv.org/pdf/2605.06915v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 262. Benchmarked Yet Not Measured -- Generative AI Should be Evaluated Against Real-World Utility

**arXiv ID:** 2605.06856 | [PDF](https://arxiv.org/pdf/2605.06856v1)

**作者:** Ishani Mondal `[一作]` (University of Maryland), Shweta Bhardwaj `[通讯]` (University of Maryland)

**通讯引用:** 407 | [OpenAlex ID](https://openalex.org/A5042484945)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种面向生成式 AI 的效用评估框架 SCU‑GenEval，基于真实部署案例探究基准与实用性之间的差距；

**💡 创新点**

创新点在于把评估焦点从静态基准指标转向以人类能力提升为核心的可持续效用评估，并构建四阶段流程与三种实用工具（结构化部署协议、用户模拟器、人格‑目标‑条件化代理指标），以系统化、分层地捕捉代理置换、时效坍塌与分布隐蔽等失效模式；

**🔧 技术方法**

采用构建利益相关者-目标映射、构造‑指标矩阵、机制建模、纵向效用测量等方法；引入大规模语言模型、人工与实验用户数据、可解释性日志、模拟器与对照实验；

**📊 数据集**

利用28个跨领域（教育、医疗、软件工程、法律）部署案例与现有基准（ImageNet、GLUE、HumanEval、VBench 等）进行对比；并基于这些案例设计结构化部署数据收集协议；

**📈 对比分析**

与传统基准评估相比，SCU‑GenEval 能揭示基准分数掩盖的长期效用缺失与群体差异；通过对照实验与模拟器预估，框架显示部分模型在短期指标表现良好但长期效用下降或对特定群体产生不利影响；

**⚠️ 局限性**

局限性包括评估成本高、需要多轮长期跟踪与大规模部署数据、对预注册与子组划分的依赖、模拟器与真实用户差异、以及方法论的主观性与可复现性挑战。

---

## 263. Robustness of Refugee-Matching Gains to Off-Policy Evaluation Choices

**arXiv ID:** 2605.06686 | [PDF](https://arxiv.org/pdf/2605.06686v1)

**作者:** Kirk Bansak `[一作]` (Stanford University), Michael Hotard `[通讯]` (Stanford University)

**通讯引用:** 391 | [OpenAlex ID](https://openalex.org/A5014497036)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估算法化难民分配对就业结果的反事实影响，并通过多种离线评估方法验证其稳健性。

**💡 创新点**

证明不同评估方法下的结果一致，排除了“赢家诅咒”偏差，并揭示模型正则化对预测结果的影响。

**🔧 技术方法**

使用IPW、AIPW、AIPW-l等离线评估技术，并结合SGBT和BART预测模型来估计潜在就业率。

**📊 数据集**

基于美国最大难民安置机构的历史分配和就业结果数据，包含2015年四季度至2016年三季度的原始数据和完整的2016年数据。

**📈 对比分析**

与原始模型基础评估结果对比，所有方法均给出相近且统计显著的就业率提升，未出现过度乐观估计。

**⚠️ 局限性**

受小倾向概率导致方差放大、池化小地区的潜在偏差、对SUTVA和无干扰假设的依赖以及仅在美国背景下验证的局限性。

---

## 264. Structured Role-Aware Policy Optimization for Multimodal Reasoning

**arXiv ID:** 2605.07274 | [PDF](https://arxiv.org/pdf/2605.07274v1)

**作者:** Bingqing Jiang `[一作]` (University of Hong Kong), Difan Zou `[通讯]` (University of Hong Kong)

**通讯引用:** 2647 | [OpenAlex ID](https://openalex.org/A5085848346)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 Structured Role-aware Policy Optimization（SRPO），在多模态强化学习框架中将生成的响应划分为感知阶段和推理阶段，并通过自蒸馏对比计算为每个 token 分配角色感知的权重，从而实现细粒度的 credit assignment；

**💡 创新点**

创新点在于：①在保持 GRPO 奖励和优化方向不变的前提下，首次将响应拆分为感知与推理两种功能角色；②利用视觉依赖对感知 token 计分、感知一致性对推理 token 计分，并统一轨迹级基线将这些角色特定的分数映射为正权重，实现更可信、更高效的多模态推理；

**🔧 技术方法**

技术实现包括：GRPO 基础策略梯度、PPO 风格的 clipped surrogate 目标、self-distilled on-policy 对比（视觉损坏 vs 原图、感知衰减 vs 无感知），token-level 权重调制、统一轨迹级基线与正权重映射，以及响应‑token 不确定性惩罚来提升训练稳定性；

**📊 数据集**

使用的训练数据为 ViRL39K（约 39K 可验证的多模态推理任务），在九个多模态推理基准上评估：MathVista、MathVision、We‑Math、MathVerse、DynaMath、MMMU‑Pro、MM‑Vet、LogicVista、NaturalBench；

**📈 对比分析**

与 GRPO、DAPO、Vision‑SR1、ThinkLite‑VL、PAPO、Perception‑R1、VPPO 等公开模型对比，SRPO 在所有基准上平均提升约 3‑5% 甚至在 7B 基础上显著提升，训练曲线更稳健，Pass@k 也表现出更高的成功率；

**⚠️ 局限性**

局限性：①对 token‑level 更新幅度的重新分配可能导致训练不稳定，需要额外的不确定性惩罚；②依赖于预定义的结构化输出（感知+推理两段），对更自由生成的任务适应性有限；③在极长或复杂视觉场景中感知与推理的边界仍可能模糊，影响 credit 分配准确性；

---

## 265. FastOmniTMAE: Parallel Clause Learning for Scalable and Hardware-Efficient Tsetlin Embeddings

**arXiv ID:** 2605.06982 | [PDF](https://arxiv.org/pdf/2605.06982v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 266. Enhancing Eye Movement Biometrics for User Authentication via Continuous Gaze Offset Score Fusion

**arXiv ID:** 2605.06810 | [PDF](https://arxiv.org/pdf/2605.06810v1)

**作者:** Hashim Aziz `[一作]` (Texas State University), Oleg V. Komogortsev `[通讯]` (Texas State University)

**通讯引用:** 3761 | [OpenAlex ID](https://openalex.org/A5035152487)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在现有的深度学习眼动生物识别系统EKYT上，探究并实现了连续注视偏移量与时间特征的分数级融合，以提升身份验证性能。

**💡 创新点**

首次证明连续注视偏移量可作为辅助特征在低质量VR环境下显著提升性能，并通过非线性树模型实现多任务与多模态融合。

**🔧 技术方法**

采用EKYT DenseNet网络提取时间嵌入，计算连续注视偏移角度，使用加权线性融合、树基随机森林、交叉任务与三任务融合等分数级融合策略。

**📊 数据集**

在两大公开数据集GazeBase（实验室眼动仪）和GazeBaseVR（VR头显）上进行实验，分别使用RAN和TEX任务。

**📈 对比分析**

通过5–40秒采样长度的EER和FRR评估，发现非线性融合在高质量数据中跨任务融合可将EER降至0.2%，在VR数据中三任务融合将EER降至4.7%/5.3%，均优于单一EKYT基线。

**⚠️ 局限性**

研究仅采用两数据集，且仅使用简单分数级融合，未探讨更复杂的融合算法或其他硬件环境，限制了结论的泛化。

---

## 267. From Specification to Deployment: Empirical Evidence from a W3C VC + DID Trust Infrastructure for Autonomous Agents

**arXiv ID:** 2605.06738 | [PDF](https://arxiv.org/pdf/2605.06738v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 268. Aquaman: A Transparent Proxy Architecture for Quantum Resilient Key Establishment

**arXiv ID:** 2605.06932 | [PDF](https://arxiv.org/pdf/2605.06932v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 269. Pricing, Matching, and Bundling: an Equilibrium Analysis of Online Platforms

**arXiv ID:** 2605.06711 | [PDF](https://arxiv.org/pdf/2605.06711v1)

**作者:** Gary Qiurui Ma `[一作]` `[通讯]` (Harvard University), Gary Qiurui Ma (Harvard University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究在线平台通过定价、匹配与捆绑三大设计杠杆影响市场参与者决策与整体福利的均衡机制。

**💡 创新点**

创新点在于（1）将平台费、交易费与分配结构统一在一个多面相交的均衡框架中；（2）对匹配问题的计算复杂性给出正式证明并给出近似收益保证；（3）在捆绑情境下提供对卖家选择范围的完整特征化，揭示信息租金与平台自制产品的相互作用。

**🔧 技术方法**

使用了算法博弈论、线性规划与凸优化技术进行均衡与最优性证明；在匹配分析中引入可扩展的仿真框架来处理外部性与动态冲击；在捆绑章节则结合机制设计与信息经济学中的激励约束。

**📊 数据集**

主要采用仿真生成的数据来验证匹配与冲击下的政策效果；论文未使用公开现实平台的数据集，而是基于假设参数构造合成实验。

**📈 对比分析**

方法上与传统单侧平台模型相比，本文通过理论证明显示即便在最坏情形下平台收益可获得至多对数因子最优福利；在仿真实验中，固定预冲击佣金并允许匹配灵活能在市场崩溃后提升 10–20% 的社会福利。

**⚠️ 局限性**

局限性包括：模型假设参与者完全理性且信息完备；仅考虑静态均衡，未对动态决策与学习行为做深入建模；仿真数据为合成，缺乏对真实平台的实证检验。

---

## 270. PACEvolve++: Improving Test-time Learning for Evolutionary Search Agents

**arXiv ID:** 2605.07039 | [PDF](https://arxiv.org/pdf/2605.07039v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 271. HyperEyes: Dual-Grained Efficiency-Aware Reinforcement Learning for Parallel Multimodal Search Agents

**arXiv ID:** 2605.07177 | [PDF](https://arxiv.org/pdf/2605.07177v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 272. Towards Fairness under Label Bias in Image Segmentation: Impact, Measurement and Mitigation

**arXiv ID:** 2605.06891 | [PDF](https://arxiv.org/pdf/2605.06891v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 273. Not All Tokens Need 40 Steps: Heterogeneous Step Allocation in Diffusion Transformers for Efficient Video Generation

**arXiv ID:** 2605.06892 | [PDF](https://arxiv.org/pdf/2605.06892v1)

**作者:** Ernie Chu `[一作]` (Johns Hopkins University), Vishal M. Patel `[通讯]` (Johns Hopkins University)

**通讯引用:** 23002 | [OpenAlex ID](https://openalex.org/A5004716468)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在视频生成中提出了 Heterogeneous Step Allocation（HSA）推理算法，按视频中 spatiotemporal token 的速度动态分配不同的去噪步数。

**💡 创新点**

创新点在于：① 为每个 token 预设异步去噪步数，而非统一步数；② 通过 KV‑cache 同步机制让活跃 token 能够访问完整序列的注意力；③ 使用缓存 Euler 步更新被跳过 token 的潜在状态，无需额外模型前向计算。

**🔧 技术方法**

核心技术包括：流匹配（Flow Matching）框架、KV‑cache 同步、缓存 Euler 更新、分组策略（如动态选择、均匀/随机分配）以及阶段性缓存窗口。

**📊 数据集**

在 Wan‑2.1‑1.3B、Wan‑2.2‑A14B、LTX‑2 等公开的 Diffusion Transformer 视频生成模型上进行实验，并通过 VBench 评测（包含多维度质量与语义分数）以及 PSNR/LPIPS 对比低预算样本。

**📈 对比分析**

与统一 Flow Matching（FM）和 TeaCache（需要离线 profiling）等基线对比，HSA 在 50% 和 25% 运行时预算下显著提升 VBench 分数，保持更高的质量‑效率 Pareto 前沿，且无需昂贵的离线分析。

**⚠️ 局限性**

局限性包括：① 需要预先设定分组与步数分配策略，缺乏自适应性；② 对极端高加速（如低于 25%）时，缓存误差可能导致细节损失；③ 当前仅在视频生成任务验证，尚未在音频或混合模态中充分评估。

---

## 274. Towards Differentially Private Reinforcement Learning with General Function Approximation

**arXiv ID:** 2605.07049 | [PDF](https://arxiv.org/pdf/2605.07049v1)

**作者:** Yi He `[一作]` (Wayne State University), Xingyu Zhou `[通讯]` (Wayne State University)

**通讯引用:** 968 | [OpenAlex ID](https://openalex.org/A5082563145)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了在通用函数逼近框架下，满足联合差分隐私的在线强化学习算法，并证明其子线性回退。

**💡 创新点**

创新点在于将批处理策略更新与无约束探索目标结合，使用指数机制实现私密化，并在一般MDP上首次给出$O(K^{3/5})$回退上界，消除了对贝尔曼完备性和最大最小优化的需求。

**🔧 技术方法**

采用批量更新、指数机制、贝尔曼残差损失、覆盖率（coverability）等技术。

**📊 数据集**

论文以理论实验为主，未使用公开真实数据集，仅提供小规模演示。

**📈 对比分析**

与之前仅限表格或线性模型的私密RL相比，回退保持$O(K^{3/5})$，且在覆盖率条件下实现子线性回退；实验显示批处理与隐私噪声导致回退下降。

**⚠️ 局限性**

局限性包括仍需贝尔曼完备性假设（在一般MDP情形）或对初始状态分布已知；批处理频率和温度参数需手动调节；对无限函数类的扩展仅给出思路。

---

## 275. Streaming Adversarial Robustness in Fuzzy ARTMAP: Mechanism-Aligned Evaluation, Progressive Training, and Interpretable Diagnostics

**arXiv ID:** 2605.06902 | [PDF](https://arxiv.org/pdf/2605.06902v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 276. Towards Security-Auditable LLM Agents: A Unified Graph Representation

**arXiv ID:** 2605.06812 | [PDF](https://arxiv.org/pdf/2605.06812v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 277. CarCrashNet: A Large-Scale Dataset and Hierarchical Neural Solver for Data-Driven Structural Crash Simulation

**arXiv ID:** 2605.07098 | [PDF](https://arxiv.org/pdf/2605.07098v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 278. Visual Text Compression as Measure Transport

**arXiv ID:** 2605.06708 | [PDF](https://arxiv.org/pdf/2605.06708v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 279. Sword: Style-Robust World Models as Simulators via Dynamic Latent Bootstrapping for VLA Policy Post-Training

**arXiv ID:** 2605.07288 | [PDF](https://arxiv.org/pdf/2605.07288v1)

**作者:** Jiaxuan Gao `[一作]` (Tianjin University), Sheng Wen `[通讯]` (Swinburne University of Technology)

**通讯引用:** 7639 | [OpenAlex ID](https://openalex.org/A5076576641)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种鲁棒的 Vision‑Language‑Action (VLA) 世界模型框架 Sword，用于在模拟环境中进行长时段的自回归视频生成和强化学习后训练。

**💡 创新点**

创新点包括①基于结构引导的风格增广（Structure‑Guided Style Augmentation），在不改变几何和任务语义的前提下大幅提升 OOD 泛化；②动态潜在自引导（Dynamic Latent Bootstrapping, DLB），通过在 VAE 潜在空间维护动态缓存，逐步将训练条件从 Teacher‑Forcing 转变为自回归，显著降低曝光偏差；③结合上述两项技术的整体体系实现对传统模型 WoVR 的显著提升。

**🔧 技术方法**

技术手段包括：Cosmos‑Transfer 2.5 风格增广、DepthAnything、GroundingDINO、SAM2 等结构辅助编码、T5 文本编码器、Wan 2.2 TI2V Transformer、Diffusion‑based VAE 生成器、动态潜在缓存机制与逐步自回归训练策略。

**📊 数据集**

主要使用 LIBERO‑Spatial 数据集，包含 1,600 条 512 帧的回放轨迹，训练集 1,500 条，评估集 100 条（包括原始与 50% 风格增广的混合测试）。

**📈 对比分析**

与 WoVR 进行对比，评估指标包括 LPIPS、FID、FVD、FloLPIPS 以及 VLA 策略的成功率。实验表明：在 OOD、混合数据集上 Sword 的 LPIPS/FID/FVD/FloLPIPS 均显著优于 WoVR；强化学习后训练策略的成功率提升约 10–15%。

**⚠️ 局限性**

限制主要体现在：①实验仅与 WoVR 进行对比，缺乏与其他世界模型方法的广泛验证；②训练成本高（约 13,000 A100 GPU‑h，约 1.5 年 GPU 计算），难以在资源受限环境复制；③对风格增广和缓存设计的超参数敏感性尚未系统评估。

---

## 280. Reflections and New Directions for Human-Centered Large Language Models

**arXiv ID:** 2605.06901 | [PDF](https://arxiv.org/pdf/2605.06901v1)

**作者:** Caleb Ziems `[一作]`, Diyi Yang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了以人为中心的大型语言模型（HCLLMs）的设计、数据、技术、评估与责任部署，并以未来工作场景为案例探讨其社会影响。

**💡 创新点**

创新点在于将人机交互（HCI）框架与LLM工程融合，系统阐述了利益相关者分析、数据来源与价值偏见、可解释性/可调控性/安全性三大责任属性，并提出多层次评估与协同工作场景的设计思路。

**🔧 技术方法**

使用的技术包括：基于HCI的用户研究方法、监督微调（SFT）、强化学习与人类反馈（RLHF）、数据审计与可解释性工具、以及协同决策与委派模型等。

**📊 数据集**

数据方面强调了LLM预训练所需的大规模文本语料库，但未给出具体数据集，主要讨论了数据的来源、质量、偏见与所有权等问题。

**📈 对比分析**

方法比较主要通过文献综述与案例分析进行，未给出数值性能指标；评估框架强调模型输出、用户体验与社会层面的量化与质性度量，并提出了生态有效性基准（如GDPval）的重要性。

**⚠️ 局限性**

局限性包括：缺乏实证实验与量化评估；对真实工作场景的任务与数据仍不充分；跨文化、跨语言与多样性利益相关者的研究仍待深入；以及责任属性间的张力与调和策略尚不完整。

---

## 281. Causal-Aware Foundation-Model for Bilevel Optimization in Discrete Choice Settings

**arXiv ID:** 2605.06941 | [PDF](https://arxiv.org/pdf/2605.06941v1)

**作者:** Shivaram Subramanian `[一作]` (IBM T.J. Watson Research Center), Jayant Kalagnanam `[通讯]` (IBM T.J. Watson Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种因果感知的基础模型框架（FMDM）与C3PO三头网络，用于在离散选择环境中实时生成最优定价决策。

**💡 创新点**

创新点在于将模仿学习、收入多任务学习与上下文学习结合，并通过推理时弹性先验实现对新产品的因果自适应，无需重新训练。

**🔧 技术方法**

采用基于Transformer的层次编码器、因果遮蔽的上下文学习模块、可微约束层，以及价格-收入曲线的多任务学习，整体实现对复杂双层优化的近似。

**📊 数据集**

训练数据来自多种经典离散选择模型（MNL、Nested Logit、混合MNL、Iso‑elastic、线性模型）生成的13,000个模拟数据集；评估数据包括医疗设备招投标、航空附加费、亚马逊DVD销售、酒店房价、瑞士地铁客流和酸奶等六个真实领域。

**📈 对比分析**

与TabPFN、Fine‑tuned TabPFN及经典随机优化策略对比，C3PO在PDR/PIR平衡指标上取得最高约61%，并在多领域实验中表现出更高的收益增益和更低的误差。

**⚠️ 局限性**

局限在于对低弹性环境的过度降价倾向、对监管与公平约束的敏感性，以及在缺乏足够弹性先验或上下文示例时性能下降。

---

## 282. Adaptive Memory Decay for Log-Linear Attention

**arXiv ID:** 2605.06946 | [PDF](https://arxiv.org/pdf/2605.06946v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 283. MLAIRE: Multilingual Language-Aware Information Retrieval Evaluation Protocal

**arXiv ID:** 2605.07249 | [PDF](https://arxiv.org/pdf/2605.07249v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 284. The Convergence Gap: Instruction-Tuned Language Models Stabilize Later in the Forward Pass

**arXiv ID:** 2605.07282 | [PDF](https://arxiv.org/pdf/2605.07282v1)

**作者:** Yifan Zhou `[一作]` (University of California, Los Angeles), Yifan Zhou `[通讯]` (University of California, Los Angeles)

**通讯引用:** 1819 | [OpenAlex ID](https://openalex.org/A5101700900)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了一种层级预测动态诊断指标——收敛间隙（convergence gap），用来量化不同模型检查点在前向传播过程中何时稳定到最终预测，并在六组预训练/指令调优（PT/IT）检查点对上进行对比分析。

**💡 创新点**

创新点在于：①引入收敛间隙作为新的诊断工具，可在保持相同历史的前提下衡量模型内部预测的“稳定时间”；②通过端点匹配、无端点相同历史对照以及固定历史模板回放，排除概率分布、置信度或模板序列导致的偏差；③使用匹配前缀 MLP 捣掠/交换干预，识别后置 MLP 窗口是导致指令调优模型延迟稳定的最大可操作窗口。

**🔧 技术方法**

技术手段包括：
- tuned lens 逐层解码下一词分布并与最终分布做 KL 距离；
- 端点匹配（confidence、entropy、top‑1 margin）与无端点相同历史（JS、top‑1 flips）控制；
- 匹配前缀 MLP 捣掠/交换干预以及随机残差投影对照实验；
- Gemma 案例的自由生成行为评估。

**📊 数据集**

数据集：六组公开 PT/IT 检查点（Gemma 3‑4B、Llama 3.1‑8B、Qwen 3‑4B、Mistral 7B、OLMo 2‑7B、DeepSeek‑V2‑Lite），每组使用数千个提示样本，生成长度约 512 令牌；Gemma 额外使用 600 个提示进行自由生成的行为评估。

**📈 对比分析**

比较方法：对每层计算收敛间隙，并在端点匹配与无端点相同历史两种控制下绘制 IT 与 PT 的曲线。结果显示 IT 模型在后置层的收敛间隙显著更大；匹配前缀 MLP 捣掠/交换实验验证后置 MLP 窗口对收敛间隙影响最大；随机残差对照实验表明非特定的后置扰动无法复制这一效应。Gemma 行为案例显示，后置 PT MLP 交换导致助手注册提示的偏好率下降约 5%，验证了收敛间隙与生成行为的关联。

**⚠️ 局限性**

局限性：
- 诊断仅针对 PT/IT 公开检查点，缺乏跨模型族的一般性验证；
- MLP 捣掠/交换干预是人工构造的匹配前缀实验，未必代表自然部署流程；
- 端点匹配与模板回放虽排除大部分偏差，但仍可能受模型内部未被控制的动态影响；
- Gemma 案例仅为单一族群的行为验证，无法推断收敛间隙对多任务或安全性等更广泛行为的预测力。

---

## 285. DPG-CD: Depth-Prior-Guided Cross-Modal Joint 2D-3D Change Detection

**arXiv ID:** 2605.07151 | [PDF](https://arxiv.org/pdf/2605.07151v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 286. Beyond the Black Box: Interpretability of Agentic AI Tool Use

**arXiv ID:** 2605.06890 | [PDF](https://arxiv.org/pdf/2605.06890v1)

**作者:** Hariom Tatsat `[一作]` (Barclays), Ariye Shater `[通讯]` (Barclays)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5117805084)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于稀疏自编码器和线性探针的内部监测框架，用于在多步代理决策前判断是否需要调用工具以及工具调用的风险级别。

**💡 创新点**

创新点在于将稀疏自编码器解码的稀疏特征与线性探针相结合，实时捕获并定位模型内部的工具调用信号，并通过特征消融验证其因果重要性。

**🔧 技术方法**

使用稀疏自编码器（SAE）对Transformer层的隐藏状态进行稀疏分解，随后训练线性探针（Tool‑Need Probe和Tool‑Risk Probe）读取稀疏特征。

**📊 数据集**

主要数据集为NVIDIA Nemotron函数调用数据集，用于训练探针；评估时使用Held‑out Nemotron测试集以及BFCL（Berkeley Function Calling Leaderboard）进行零样本迁移。

**📈 对比分析**

在Nemotron held‑out 测试中，GPT‑OSS 20B的Tool‑Need准确率为75.3%，Gemma 3 27B为71.4%；在BFCL零样本迁移中，两模型的内部信号与实际工具调用的一致性均超过77%，但误报率和漏报率存在差异。

**⚠️ 局限性**

局限性包括：Tool‑Risk探针依赖于任务特定的风险层级定义，跨数据集迁移受工具集和指令匹配的影响；仅评估了两种开源模型，未验证不同架构或规模下的可迁移性；稀疏特征可能随检查点和SAE训练方式漂移。

---

## 287. Decoupling Semantics and Fingerprints: A Universal Representation for AI-Generated Image Detection

**arXiv ID:** 2605.07074 | [PDF](https://arxiv.org/pdf/2605.07074v1)

**作者:** Zhiyuan Wang `[一作]` (Hefei University of Technology), Yunfeng Diao `[通讯]` (Hefei University of Technology)

**通讯引用:** 324 | [OpenAlex ID](https://openalex.org/A5026338785)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种Orthogonal Decomposition and Purification Network (ODP-Net)，通过结构化分离生成器特征、语义内容与普适篡改痕迹，实现AI图像生成检测。

**💡 创新点**

创新点在于利用频谱正交性引入实例感知硬阈值分解、对抗性纯化和流形对齐三步，显式构造三种互相正交的子空间，打破传统特征混叠问题。

**🔧 技术方法**

采用CLIP ViT‑L/14预训练特征提取器，结合实例感知正交分解模块、交叉样本纯化损失和流形对齐正则化，整体通过多项式损失共同训练。

**📊 数据集**

主要使用AIGIBench数据集，其中训练集包含8种生成器（如ProGAN、DALLE‑3等），测试集包含8种未见生成器（如SD3、Midjourney等）以及In‑the‑Wild子集。

**📈 对比分析**

与AIDE、FreqNet、RINE、NRR、VIB、FatFormer、SAFE等7种最先进方法对比，ODP-Net在未见生成器（如Stable Diffusion 3）上达到99.37%准确率，跨域平均准确率和NLL均显著优于基线，表现出最优的泛化与概率校准。

**⚠️ 局限性**

局限性包括：依赖频谱正交性假设，可能对极端压缩或伪影混合场景适应性不足；硬阈值分解在小样本或多模态数据上可能受限；整体模型复杂度较高，需较大训练数据和算力。

---

## 288. A Rod Flow Model for Adam at the Edge of Stability

**arXiv ID:** 2605.06821 | [PDF](https://arxiv.org/pdf/2605.06821v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 289. Christoffel-DPS: Optimal sensor placement in diffusion posterior sampling for arbitrary distributions

**arXiv ID:** 2605.06861 | [PDF](https://arxiv.org/pdf/2605.06861v1)

**作者:** James Rowbottom `[一作]` (University of Cambridge), Ben Adcock `[通讯]` (Simon Fraser University)

**通讯引用:** 4047 | [OpenAlex ID](https://openalex.org/A5065352983)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于Christoffel函数的分布无关式传感器布置框架，并将其与后验采样（Diffusion Posterior Sampling）相结合，推出Christoffel‑DPS（含离线和在线变体）以实现稀疏传感下的高精度状态重建。

**💡 创新点**

创新点在于：①突破传统高斯假设，利用Christoffel函数衡量任意分布下的可辨识度，从而得到理论最优的随机传感分布；②在后验采样框架中实现了基于Christoffel函数的自适应传感器选取；③提出离线估计与在线动态更新两种实用策略。

**🔧 技术方法**

采用Christoffel函数理论、Diffusion Posterior Sampling（DPS）以及基于样本集的secant集估计；实现了离线Christoffel‑DPS（基于最大化empirical Christoffel值的随机/贪婪采样）和在线ensemble Christoffel‑DPS（基于动态重采样的自适应传感）。

**📊 数据集**

在三种基准数据集上验证：Pinball（GRIFDIR）、Darcy流（DiffusionPDE）和Kolmogorov流（physics‑constrained masked diffusion）。

**📈 对比分析**

与随机、A‑、D‑、E‑最优POD、SSPOR等经典OED方法对比，Christoffel‑DPS在低传感器预算下实现了约一半的预算即可达到同样误差，整体误差显著低于基线；在Pinball与Kolmogorov流实验中表现尤为突出。

**⚠️ 局限性**

局限性包括：仅针对点测量的状态重建；secant集估计与贪婪算法的计算开销较大；在线ensemble版本需要多条DPS链，推理成本高；在近似高斯或线性问题时优势不明显；未对梯度增强或非线性PDE逆问题进行实验。

---

## 290. AGWM: Affordance-Grounded World Models for Environments with Compositional Prerequisites

**arXiv ID:** 2605.06841 | [PDF](https://arxiv.org/pdf/2605.06841v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 291. From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms

**arXiv ID:** 2605.06716 | [PDF](https://arxiv.org/pdf/2605.06716v1)

**作者:** Jinghao Luo `[一作]` (South China Normal University), Jing Ma `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 23356 | [OpenAlex ID](https://openalex.org/A5020347295)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

该论文综述了LLM代理的记忆机制，提出了演化框架（存储–反思–经验），系统梳理了驱动演化的需求，并对经验阶段的主动探索与跨轨迹抽象进行深入讨论。

**💡 创新点**

创新点在于将记忆机制从静态存储、错误纠正转化为高层抽象的演化视角，提出三阶段框架与“Why‑How‑What”研究问题，定义了主动探索与跨轨迹抽象的技术要点。

**🔧 技术方法**

主要技术包括轨迹存储（线性、向量、结构化）、反思模块（自省、环境、协同）和经验抽象（显式、隐式、混合），以及抽象操作（对比归纳、层次化、参数内化）。

**📊 数据集**

作为综述论文，未引入专门的数据集，讨论基于已有公开或内部实验数据，但未建立统一 benchmark。

**📈 对比分析**

由于缺乏统一基准，本文未提供定量比较；对已有方法的讨论多基于文献评价，性能差异依赖模型、环境与提示设置。

**⚠️ 局限性**

局限包括缺乏定量对比、缺少统一评估基准、经验阶段与现有学习范式重叠、时间覆盖与新颖性偏差、未能系统化所有技术。

---

## 292. Can You Break RLVER? Probing Adversarial Robustness of RL-Trained Empathetic Agents

**arXiv ID:** 2605.07138 | [PDF](https://arxiv.org/pdf/2605.07138v1)

**作者:** Deeraj S K `[一作]` (Sardar Vallabhbhai National Institute of Technology), Sudhakar Mishra `[通讯]` (Sardar Vallabhbhai National Institute of Technology)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5122770754)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了Adversarial Empathy Benchmark和Emotional Consistency Score，用于评估在对抗性情绪交流中强化学习训练的语言模型的同理心鲁棒性。

**💡 创新点**

创新点在于：①设计六种基于临床心理学的对抗性对话轨迹并引入判别式奖励规则；②提出情绪一致性得分，分离情绪状态追踪与情绪提升两维度；③在对抗性环境下开展受控实验，验证RL训练对同理心的提升。

**🔧 技术方法**

使用了SAGE模拟器生成的情绪对话、PPO/GRPO强化学习框架、Think-Then-Say推理结构与Chain-of-Thought提示，评估器采用Mistral‑7B作为情绪判别者。

**📊 数据集**

数据集由480条情境匹配的对话组成，涵盖六种对抗性轨迹（Escalation、Mood Reversal、Gaslighting、Fact‑Emotion Contradiction、Emotional Flooding、Validation Manipulation），每轨迹10条实例。

**📈 对比分析**

与未调优的基线（Base‑7B）以及同尺寸基线（Base‑1.5B）进行对比，-PPO‑Think条件在最终得分（FS）上从0.761提升至0.963（p<0.001，r=0.688），隐藏意图检测率提升47%，且无对话崩溃。相比之下，状态可读性指标（ECS）变化不显著，显示情绪响应与状态追踪存在分离。

**⚠️ 局限性**

主要局限包括：①所有用户和评判均为同一Mistral‑7B模拟器，存在同族循环；②情绪一致性得分范围窄，可能低估真实跟踪能力；③缺乏人类评估和多语言验证，实验仅基于英文、西方心理学范式。

---

## 293. Do Joint Audio-Video Generation Models Understand Physics?

**arXiv ID:** 2605.07061 | [PDF](https://arxiv.org/pdf/2605.07061v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 294. Real-IAD MVN: A Multi-View Normal Vector Dataset and Benchmark for High-Fidelity Industrial Anomaly Detection

**arXiv ID:** 2605.07149 | [PDF](https://arxiv.org/pdf/2605.07149v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 295. Beyond Reasoning: Reinforcement Learning Unlocks Parametric Knowledge in LLMs

**arXiv ID:** 2605.07153 | [PDF](https://arxiv.org/pdf/2605.07153v1)

**作者:** Wanli Yang `[一作]` (State Key Laboratory Of Ai Safety Institute Of Computing Technology Chinese Academy Of Sciences), Fei Sun `[通讯]` (State Key Laboratory Of Ai Safety Institute Of Computing Technology Chinese Academy Of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究强化学习在大型语言模型中直接提升事实召回的效果，设计了零样本、单跳、闭卷问答的控制实验，并使用二元正确性奖励对模型进行训练。

**💡 创新点**

证明RL不仅能提升推理能力，还能通过概率质量重新分布，拉近隐藏事实与输出分布的距离，从而显著提升直接事实召回；同时发现低可访问性训练样本是最有效的学习信号。

**🔧 技术方法**

采用Group Relative Policy Optimization（GRPO）为主的RL框架，使用LLM-as-a-Judge进行语义验证奖励，结合pass@k分析、数据归因实验等技术。

**📊 数据集**

在Qwen2.5-7B-Instruct、Llama-3.1-8B-Instruct、OLMo-2-7B-Instruct等模型上评估，使用Natural Questions、TriviaQA、PopQA、SimpleQA四个闭卷事实问答数据集，并对训练集做事实级去重。

**📈 对比分析**

与离线监督微调、DPO、拒绝采样微调以及推理时缩放策略（多数投票、链式推理）比较；RL在三模型三数据集平均提升约27%相对准确率，单数据集最高可达53%（NQ），在更大模型（72B）和MoE 也保持显著增益。

**⚠️ 局限性**

要求初始模型在任务上已有非零召回率，极难数据集（如SimpleQA）几乎无提升；奖励稀疏导致RL效果下降；对模型已不存在的知识无法学习，受模型潜在记忆深度限制。

---

## 296. Hard to Read, Easy to Jailbreak: How Visual Degradation Bypasses MLLM Safety Alignment

**arXiv ID:** 2605.07250 | [PDF](https://arxiv.org/pdf/2605.07250v1)

**作者:** Zhixue Song `[一作]` (Westlake University), Chi Zhang `[通讯]` (Westlake University)

**通讯引用:** 26803 | [OpenAlex ID](https://openalex.org/A5100458183)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探究视觉上下文压缩导致的MLLM漏洞，发现图像分辨率降低会引发“认知过载”从而触发越狱攻击，并提出结构化认知卸载的防御方案。

**💡 创新点**

首次提出“攻击舒适区”（ACZ）与认知过载假设，并通过分阶段序列推理实现对视觉输入的安全审计解耦。

**🔧 技术方法**

利用视觉文本压缩、OCR、层级安全探针、prompt工程以及串行推理管线技术。

**📊 数据集**

使用770条聚合自标准基准的有害查询，涵盖英文与中文，结合多种视觉扰动（噪声、模糊、畸变等）进行评测。

**📈 对比分析**

在不同DPI、扰动和模型（Qwen3-VL、GPT‑4.1、Claude‑4.5等）上比较ASR，结构化认知卸载将攻击成功率从约70%降至≈4%，同时保持高质量回复且无误拒。

**⚠️ 局限性**

防御方法导致输出长度平均增加102%，对提示语的依赖较高，且尚未验证对抽象视觉推理攻击的普适性。

---

## 297. InfoGeo: Information-Theoretic Object-Centric Learning for Cross-View Generalizable UAV Geo-Localization

**arXiv ID:** 2605.07099 | [PDF](https://arxiv.org/pdf/2605.07099v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 298. PLOT: Progressive Localization via Optimal Transport in Neural Causal Abstraction

**arXiv ID:** 2605.06979 | [PDF](https://arxiv.org/pdf/2605.06979v1)

**作者:** Jonathn Chang `[一作]` (Cornell University), Ziv Goldfeld `[通讯]` (Cornell University)

**通讯引用:** 503 | [OpenAlex ID](https://openalex.org/A5071112095)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了PLOT（Progressive Localization via Optimal Transport）框架，利用最优传输逐层定位因果变量，并生成干预handle或指导DAS搜索，实现神经网络与高层因果模型的对齐；

**💡 创新点**

创新点在于使用全局OT耦合构建软对应关系，并通过逐步细化（从层/token到坐标/主成分）实现高效定位，既可直接得到干预handle，又能显著加速DAS搜索；

**🔧 技术方法**

核心技术包括最优传输（EOT/UOT）及Sinkhorn算法、PCA特征提取、神经swap、对齐旋转（DAS）和多阶段调优；

**📊 数据集**

实验数据集涵盖HEQ（简单判定任务）、4-bit二进制加法（GRU）以及大型Gemma-2-2B的MCQA（多选问答）；

**📈 对比分析**

与传统DAS对比，OT‑only PLOT在准确率上几乎相同（如HEQ平均0.991 vs 0.995），但运行时间从数十秒降至几秒；PLOT‑guided DAS在保持准确率的同时，将Full‑DAS的时间缩短10–100倍；

**⚠️ 局限性**

局限性包括对层级划分的手工设定、对大模型批处理与并行计算的需求、对PCA或坐标分解在某些任务中的效果不稳定，以及对参数设置（如ε、β、top‑K）的敏感性。

---

## 299. $f$-Divergence Regularized RLHF: Two Tales of Sampling and Unified Analyses

**arXiv ID:** 2605.06977 | [PDF](https://arxiv.org/pdf/2605.06977v1)

**作者:** Di Wu `[一作]` (University of Virginia), Cong Shen `[通讯]` (University of Virginia)

**通讯引用:** 3268 | [OpenAlex ID](https://openalex.org/A5016749653)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了两种针对一般 f‑divergence 正则化的 RLHF（强化学习从人类反馈）在线学习算法，分别基于乐观探索和导数作为不确定性来设计采样策略。

**💡 创新点**

创新点在于首次统一分析 f‑divergence 正则化下的 RLHF，证明两种算法分别可获得 O(log T) 的 regret 与 O(1/T) 的子最优误差，并引入“导数即不确定性”的全新探索理念。

**🔧 技术方法**

技术手段包括：基于 Bradley‑Terry 模型的偏好反馈、f‑divergence 的凸解析、乐观上界构造、Eluder 维度估计、线性奖励模型以及基于梯度与 Hessian 对齐的采样策略。

**📊 数据集**

实验使用了线性上下文-动作模拟数据（随机上下文向量、10 个离散动作）并在 BT 模型下验证，未使用真实 LLM 生成或人类偏好数据集。

**📈 对比分析**

与贪婪采样、均匀采样等基线相比，实验显示两种算法均实现了快速收敛，且导数驱动的采样在 chi‑squared‑mixed KL 与 x log x−log x 正则化下优于标准 KL，符合理论预测。

**⚠️ 局限性**

局限性在于仅在合成实验环境中验证，理论假设如可实现性、有限奖励类和正则化参数等可能不完全适用于大规模 LLM 真实场景，且未对大动作空间下的采样效率做深入分析。

---

## 300. Mitigating Cognitive Bias in RLHF by Altering Rationality

**arXiv ID:** 2605.06895 | [PDF](https://arxiv.org/pdf/2605.06895v1)

**作者:** Tiffany Horter `[一作]` (University of Oxford), Serena Booth `[通讯]` (Brown University)

**通讯引用:** 430 | [OpenAlex ID](https://openalex.org/A5084731369)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种在 RLHF 训练中动态调整合理性参数 β 的方法，以降低模型对人类偏见的复制。

**💡 创新点**

创新点在于将 β 视为上下文和注释依赖的可变参数，并利用 LLM 判别器估计每条反馈的偏见概率，从而自适应地调节权重。

**🔧 技术方法**

技术包括 Boltzmann 理性模型、LLM-as-judge（ChatGPT、Mistral-7B）对偏见的检测、逻辑斯蒂变换动态 β 计算、奖励模型训练以及 GRPO 细调。

**📊 数据集**

使用了两组专门诱发认知偏差的数据集：BRU（205 个多选题）和 CogBias（约 30,000 条测试案例）。

**📈 对比分析**

通过与固定 β（如 1.0、0.9、0.5）以及随机 β 的基线对比，实验显示动态 β 模型在 CogBias 上提升 25.2% 准确率，在 BRU 上提升约 8.3%；同时在 TruthfulQA 等通用任务中无显著性能下降，并在不同偏差比例、模型架构和噪声判别下保持鲁棒性。

**⚠️ 局限性**

主要限制是对偏见判别器的准确性高度依赖，判别器若对某些偏差类型识别不佳，减偏效果有限；此外，模型对“父权主义”风险的道德考量及在更复杂任务中的泛化尚需进一步验证。

---

## 301. Gated QKAN-FWP: Scalable Quantum-inspired Sequence Learning

**arXiv ID:** 2605.06734 | [PDF](https://arxiv.org/pdf/2605.06734v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 302. Adaptive Subspace Projection for Generative Personalization

**arXiv ID:** 2605.07257 | [PDF](https://arxiv.org/pdf/2605.07257v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 303. TRAJGANR: Trajectory-Centric Urban Multimodal Learning via Geospatially Aligned Neural Representations

**arXiv ID:** 2605.06990 | [PDF](https://arxiv.org/pdf/2605.06990v1)

**作者:** Maria Despoina Siampou `[一作]` (Google), Shushman Choudhury `[通讯]` (Google)

**通讯引用:** 180 | [OpenAlex ID](https://openalex.org/A5000095866)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种轨迹中心的多模态自监督预训练框架，将人类移动轨迹与街景图像及其地理位置进行细粒度对齐；

**💡 创新点**

创新点在于：①将轨迹建模为可查询的神经隐函数，实现对连续路径的局部表示；②提出基于此的三模态对齐预训练目标，显著提升了对道路与城市认知任务的表现；

**🔧 技术方法**

技术包括：ViT+投影层用于街景图像编码，Space2Vec+Time2Vec用于位置与时间编码，Transformer实现轨迹编码，神经隐函数实现轨迹连续查询，InfoNCE对齐损失，MoCo式负样本队列；

**📊 数据集**

使用公开的出租车轨迹数据（Porto和San Francisco），并检索对应的Mapillary街景图像；

**📈 对比分析**

与GeoCLIP、SatCLIP、GAIR、UniTraj等多模态和轨迹基准模型比较，在交通速度预测、道路受欢迎度、兴趣点功能预测及硬刹车事件预测等四项任务中，均取得显著提升（如San Francisco交通速度MAE下降71.6%，硬刹车预测提升89.0%）；

**⚠️ 局限性**

局限性：仅集成了轨迹、街景图像和地理位置三种模态，缺乏对道路网络、建筑或遥感影像等其他重要空间信息的整合；预训练规模仅覆盖区域级；未探索零样本或少样本场景；

---

## 304. Behavior Cue Reasoning: Monitorable Reasoning Improves Efficiency and Safety through Oversight

**arXiv ID:** 2605.07021 | [PDF](https://arxiv.org/pdf/2605.07021v1)

**作者:** Christopher Z. Cui `[一作]` (University of California), Prithviraj Ammanabrolu `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大型语言模型中引入行为提示（Behavior Cues）使其在推理过程中主动输出特定的标记，从而实现推理可监测性与可控制性。

**💡 创新点**

通过训练模型自然产生三类控制标记（回答更新、继续推理、停止推理），并让这些标记可被外部监督模块解析并强制执行，提升推理效率与安全性。

**🔧 技术方法**

采用提取工作答案、嵌入标记、监督微调（SFT）和强化学习（RL）等技术对模型进行行为提示训练，并构建基于规则和非推理监视器。

**📊 数据集**

在 Qwen3‑8B 与 GLM‑Z1‑9B 两大模型上，分别在 AIME（数学问题）、Textworld（文本任务）和 Hazardworld（安全约束）三个领域进行评估。

**📈 对比分析**

与基线模型对比，行为提示模型在准确率、胜率和调整分数上略有提升；外部监督通过行为提示可节约约 50% 的推理令牌，并将安全违规率从 46% 提升至 96%。

**⚠️ 局限性**

对分布漂移的鲁棒性有限，标记遵从度在不同任务中差异显著，且需要对标记与模型权重共同微调，难以实现完全无侵入的方式。

---

## 305. RepoZero: Can LLMs Generate a Code Repository from Scratch?

**arXiv ID:** 2605.07122 | [PDF](https://arxiv.org/pdf/2605.07122v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 306. Dual-Scale Temporal Fusion Reveals Structured Predictability in Subseasonal-to-Seasonal Temperature Prediction

**arXiv ID:** 2605.06911 | [PDF](https://arxiv.org/pdf/2605.06911v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 307. Beyond Single Ground Truth: Reference Monism as Epistemic Injustice in ASR Evaluation

**arXiv ID:** 2605.07084 | [PDF](https://arxiv.org/pdf/2605.07084v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 308. Sample Complexity of Stochastic Optimization with Integer Variables

**arXiv ID:** 2605.07239 | [PDF](https://arxiv.org/pdf/2605.07239v1)

**作者:** Hongyu Cheng `[一作]` (Johns Hopkins University), Amitabh Basu `[通讯]` (Johns Hopkins University)

**通讯引用:** 1142 | [OpenAlex ID](https://openalex.org/A5048617506)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了整数随机优化的样本复杂度，并与连续优化进行了对比

**💡 创新点**

首次给出整数优化在不同范数下（ℓ∞、ℓ2）以及强凸光滑情形下的上界与下界，揭示整数约束在某些情形下可降低样本需求，而在强凸光滑下整数优化需要更高样本量；同时证明了统一收敛、ERM 与任意算法在这些场景下的复杂度等价（或可比）

**🔧 技术方法**

理论分析方法包括覆盖数与链式技术、锚定均匀收敛、ERM 误差分析、Fano 不等式构造下界，以及子高斯增量假设来处理无界域的光滑强凸情形

**📊 数据集**

无实际数据集，全部为纯理论证明与解析

**📈 对比分析**

与已知的连续优化样本复杂度对比：在 ℓ∞ 球下，整数优化与连续完全等价；在 ℓ2 球下，当 R²<d 时整数优化样本复杂度更小；在强凸光滑情形下整数优化样本复杂度为 Θ(1/ε²)，而连续情况为 Θ(1/ε)；结果与已知上界下标常数相匹配，说明分析紧确

**⚠️ 局限性**

仅限于箱型或 ℓ₂ 球形约束、1‑Lipschitz或强凸光滑假设，未覆盖更一般的约束或非凸场景；理论证明未通过实验验证，且对实际算法实现的细节缺乏讨论

---

## 309. Metaphors as Scaffolds: Spatial, Embodied, Fantastical, and Relational Framings for Youth Usable Privacy Design

**arXiv ID:** 2605.07185 | [PDF](https://arxiv.org/pdf/2605.07185v1)

**作者:** JaeWon Kim `[一作]` (University of Washington), Alexis Hiniker `[通讯]` (University of Washington)

**通讯引用:** 3440 | [OpenAlex ID](https://openalex.org/A5074077266)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析了三项关于13–24岁青少年隐私认知的研究，探讨不同隐喻（空间、身体、幻想、关系）如何塑造其隐私决策。

**💡 创新点**

提出隐喻选择是隐私设计的首要伦理决策，而非仅仅是用户界面装饰，揭示了隐喻对隐私推理的深层影响。

**🔧 技术方法**

采用概念隐喻理论、通信隐私管理(CPM)框架和Nissenbaum的情境完整性理论，对研究数据进行诠释性分析。

**📊 数据集**

利用三项原始研究的访谈记录、设计原型与现场笔记作为数据集，涵盖“Project H”、“Project D”和“Project T”。

**📈 对比分析**

通过案例比较的方式呈现不同隐喻对隐私推理的影响；未进行量化性能评估，主要依赖质性描述与对比。

**⚠️ 局限性**

局限在样本为美国青少年，受文化（如哈利波特、游戏）影响；缺乏跨文化验证；未在真实产品中检验隐喻选择的实际效果。

---

## 310. Intention assimilation control for accurate tracking with variable impedance in teleoperation

**arXiv ID:** 2605.07037 | [PDF](https://arxiv.org/pdf/2605.07037v1)

**作者:** Atsushi Takagi `[一作]`, Etienne Burdet `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

开发了一种名为意图同化控制（IAC）的远程操控策略，能在跟随机器人阻尼变化时保持高跟踪精度并提升任务完成率。

**💡 创新点**

创新点在于通过估计操作者施加的力来推断其目标轨迹和阻尼，将虚拟目标传递给跟随机器人，从而在低阻尼环境下仍能精确跟踪并避免传统 TIC 的“摆动”问题。

**🔧 技术方法**

使用了卡尔曼滤波器进行目标与阻尼估计、机器人动力学预补偿、阻尼调节以及在实验中采用七自由度 Franka Emika 机械臂和力反馈传感器。

**📊 数据集**

未使用公开数据集，而是自行构建实验数据：自由跟踪、气球软交互、插孔任务和双向抛光任务，记录位置、速度、阻尼和外力等信息。

**📈 对比分析**

将 IAC 与传统 TIC 在单向 100 ms 延迟和无延迟双向条件下进行比较，评估指标包括跟踪误差、任务完成率、完成时间等；实验表明 IAC 在所有阻尼水平下跟踪误差更低、插孔成功率提升 38% 以上、完成时间缩短 40% 以上。

**⚠️ 局限性**

局限性包括仅在相似动力学的同尺寸机器人间验证，未测试跨尺度或动力学差异大的领导-跟随配置；阻尼估计依赖握力或 EMG 需要额外校准；在极低阻尼下估计误差仍可能影响性能。

---

## 311. Dataset Watermarking for Closed LLMs with Provable Detection

**arXiv ID:** 2605.06865 | [PDF](https://arxiv.org/pdf/2605.06865v1)

**作者:** Pengrun Huang `[一作]` (University of California San Diego), Yu-Xiang Wang `[通讯]` (University of California San Diego)

**通讯引用:** 131 | [OpenAlex ID](https://openalex.org/A5101990526)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在黑盒访问条件下，提出一种通过增大随机词对共现频率实现的数据集水印方案，能够在LLM微调阶段检测是否使用该数据集。

**💡 创新点**

创新点在于仅需API访问即可检测水印，提供分布无关、模型无关的误报率理论上限，并通过词对共现而非下游概率或内部logit实现水印嵌入。

**🔧 技术方法**

技术：词对共现增强的重写式水印、统计假设检验、Hoeffding不等式推导误报率、黑盒文本生成采样。

**📊 数据集**

使用三大LLM基准数据集MMLU、ARC-easy、ARC-challenge，以及两种开源基础模型LLaMA-3-8B-Instruct和Gemma-2-2B-Instruct进行实验。

**📈 对比分析**

与STAMP和Radioactive基线相比，在闭盒访问下本方法实现p<0.01的显著检出率，在混合数据和文本轻微扰动条件下仍保持高检出能力，且对模型性能影响微乎其微。

**⚠️ 局限性**

局限性：只能在数据发布前嵌入水印，无法对已发布数据做后期处理；对强逆向改写不鲁棒；评估仅限微调阶段，未验证大规模预训练时的效果。

---

## 312. LENS: Low-Frequency Eigen Noise Shaping for Efficient Diffusion Sampling

**arXiv ID:** 2605.07253 | [PDF](https://arxiv.org/pdf/2605.07253v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 313. ProtSent: Protein Sentence Transformers

**arXiv ID:** 2605.06830 | [PDF](https://arxiv.org/pdf/2605.06830v1)

**作者:** Dan Ofer `[一作]` (Hebrew University of Jerusalem), Nadav Rappoport `[通讯]` (Ben Gurion University of Negev)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对蛋白质语言模型进行对比学习微调，使得序列级别的嵌入空间更能反映功能、结构和进化相似性。

**💡 创新点**

结合五类生物关系数据（Pfam家族、硬负样本、AlphaFold结构、STRING相互作用、DMS功能）并采用轮询采样的多源对比学习框架ProtSent；通过Contrastive Fine‑Tuning实现全通用嵌入。

**🔧 技术方法**

使用ESM‑2（35M/150M）作为backbone，配合SentenceTransformers框架、MultipleNegativesRankingLoss（MNRL）和CoSENT loss进行对比学习，L2归一化后得到序列级别嵌入，并用k‑NN探针评估。

**📊 数据集**

Pfam家族对、Pfam硬负样本、AlphaFold DB结构聚类、STRING蛋白互作、ProteinGym DMS/临床变异数据。

**📈 对比分析**

采用冻结嵌入+k‑NN探针在23个下游任务（分类、回归、检索）进行评估。ProtSent在35M模型上提升了16/23任务，在150M模型上提升了15/23任务；在远程同源检测上+105%提升，SCOPe‑40结构检索Recall@1提升+19.9%，以及在多项功能和变异预测任务上也获得显著进步。

**⚠️ 局限性**

训练仅覆盖所选的五类关系类型，对未覆盖的蛋白相似性（如酶机制、表达模式等）不具针对性；某些回归任务表现不如预期；缺少多种子与置信区间评估；与专门检索系统（如ProtTucker、ProtSearch）直接对比的结果缺失。

---

## 314. ScarfBench: A Benchmark for Cross-Framework Application Migration in Enterprise Java

**arXiv ID:** 2605.06754 | [PDF](https://arxiv.org/pdf/2605.06754v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 315. A Self-Healing Framework for Reliable LLM-Based Autonomous Agents

**arXiv ID:** 2605.06737 | [PDF](https://arxiv.org/pdf/2605.06737v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 316. Polylogarithmic Approximation for Covering and Connecting Multi-Interface Networks

**arXiv ID:** 2605.06899 | [PDF](https://arxiv.org/pdf/2605.06899v1)

**作者:** Michał Szyfelbein `[一作]` (Gdańsk University of Technology), Camille Richer `[通讯]` (Université Paris-Dauphine)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究多接口无线网络中的覆盖与连通问题，给出最小化顶点最大激活接口成本的理论近似算法；

**💡 创新点**

首次在异构成本模型下提出对覆盖问题的O(log m)近似和对连通问题的O(log² m)近似；

**🔧 技术方法**

构造线性规划松弛（ILP）并引入随机阈值的随机化舍入技术；同时在连通问题中使用基于流的割约束和Karger的最小割估计；

**📊 数据集**

无具体实验数据集，所有结果均为理论证明；

**📈 对比分析**

通过理论分析证明在最坏情况下覆盖问题可达O(log m)逼近（与集合覆盖下界一致），连通问题可达O(log² m)逼近；未与实验或其他算法直接比较；

**⚠️ 局限性**

仍存在O(log² m)与O(log m)逼近之间的差距；对特殊图结构（如星图）的更紧逼近尚未得到；

---

## 317. An Aerial Manipulator for Perception-Driven Flower Targeting Toward Contactless Pollination in Vertical Farming

**arXiv ID:** 2605.06759 | [PDF](https://arxiv.org/pdf/2605.06759v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 318. Understanding Performance Collapse in Layer-Pruned Large Language Models via Decision Representation Transitions

**arXiv ID:** 2605.07271 | [PDF](https://arxiv.org/pdf/2605.07271v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 319. PolarAdamW: Disentangling Spectral Control and Schur Gauge-Equivariance in Matrix Optimisation

**arXiv ID:** 2605.07067 | [PDF](https://arxiv.org/pdf/2605.07067v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 320. Finitary Truly Concurrent Bisimulations

**arXiv ID:** 2605.07373 | [PDF](https://arxiv.org/pdf/2605.07373v1)

**作者:** Yong Wang `[一作]` `[通讯]`, Yong Wang

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文在真正并发语义框架下，提出并严格定义了四种 finitary 真正并发预 bisimulation（pomset、step、hp-prebisimulation 和 hhp-prebisimulation），并证明它们与传统预 bisimulation 及其 finitary 形式之间的包含关系与严格性。

**💡 创新点**

创新点在于：① 将 finitary 约束引入真正并发预 bisimulation，弥补了传统预 bisimulation 只能区分无限观察而无法区分有限观察的缺陷；② 通过构造一系列递归预关系（^P_(⋯,n)）实现对任意有限 pomset 集合的“有限化”，从而得到 finitary 预 bisimulation 的完备性与可判定性；③ 对不同类型的并发预 bisimulation 统一了定义与证明框架，并给出了在有限分支 PLTS 上所有关系等价的判定。

**🔧 技术方法**

主要使用的技术包括：基于 labeled poset（pomset）和 partial order 多态的结构化语义；posetal relation 与 hereditary history-preserving relation 的构造；递归函数 F 用于定义预关系的闭包；利用交集和极限（ω）构造 finitary 预关系；以及在同步树（Synchronization Tree）上的形式化映射。

**📊 数据集**

由于研究是理论性的，本文未使用任何实验数据集，所有结果均通过数学证明得到。

**📈 对比分析**

本文通过理论比较，证明了 finitary 预 bisimulation 与传统预 bisimulation、步骤预 bisimulation 等关系的严格包含；在有限分支 PLTS 上证明了所有定义在行为等价上相等，而在无限分支 PLTS 上则展示了严格包含的例子，说明 finitary 预 bisimulation 更能捕捉有限行为。

**⚠️ 局限性**

局限性主要体现在：① 本研究仅提供理论定义与性质证明，缺乏算法实现与复杂度分析；② 对无限分支 PLTS 的处理仍存在严格包含现象，说明 finitary 预 bisimulation 在此类系统上可能过于粗粒；③ 由于未给出实验验证，尚不清楚在实际并发程序或模型检查工具中的可行性和性能表现。

---

## 321. Mask2Cause: Causal Discovery via Adjacency Constrained Causal Attention

**arXiv ID:** 2605.07280 | [PDF](https://arxiv.org/pdf/2605.07280v1)

**作者:** Omar Muhammad `[一作]` (Indian Institute of Science), Deepak N. Subramani `[通讯]` (Indian Institute of Science)

**通讯引用:** 900 | [OpenAlex ID](https://openalex.org/A5076749495)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 Mask2Cause，一个端到端的 Transformer 框架，用来在一次前向传播中直接恢复多变量时间序列的 Granger‑因果图。

**💡 创新点**

创新点包括：① 逆向变量嵌入（Treat 变量历史为 token），② 全局可学习的邻接矩阵约束注意力，③ 通过方差驱动的 NLL 目标同时识别均值与方差因果，④ 将结构学习与预测目标联合优化，避免后期分离提取，⑤ 通过稀疏 L1 正则实现参数压缩。

**🔧 技术方法**

使用技术：Transformer 编码器、Adjacency‑Constrained Masked Attention、Inverted Variable Embedding、可学习的稀疏邻接矩阵、Gaussian NLL / MSE 损失、Softmax+log‑mask 处理、阈值化得到离散因果图。

**📊 数据集**

数据集：VAR 线性系统、Lorenz‑96 混沌系统、CausalTime（交通、AQI、医疗三类）、DREAM3 基因调控网络、混合物理（Mean‑Var 分离的合成数据）以及在高维 (N=100) 的扩展实验。

**📈 对比分析**

与 cMLP、cLSTM、TCDF、CUTS、CUTS+、Causalformer、SRU、eSRU、PCMCI、NGM、LCCM 等基线在 AUROC、AUPRC、SHD 等指标上进行比较。Mask2Cause 在大多数基准上实现接近 1.0 的 AUROC，尤其在 Mixed Physics 数据上 NLL 版本显著优于均值版与传统方法，显示出对方差因果的优势。

**⚠️ 局限性**

限制：① 仍假设无未观测混淆、可观测平稳序列；② 对超大规模 (N≫1000) 的二次复杂度仍是瓶颈；③ 需要手动阈值化得到离散图，阈值对结果影响较大；④ 对极端非线性或强非平稳过程的适用性尚未彻底验证。

---

## 322. LookWhen? Fast Video Recognition by Learning When, Where, and What to Compute

**arXiv ID:** 2605.06809 | [PDF](https://arxiv.org/pdf/2605.06809v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 323. Learned Lyapunov Shielding for Adaptive Control

**arXiv ID:** 2605.06934 | [PDF](https://arxiv.org/pdf/2605.06934v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 324. RRCM: Ranking-Driven Retrieval over Collaborative and Meta Memories for LLM Recommendation

**arXiv ID:** 2605.07129 | [PDF](https://arxiv.org/pdf/2605.07129v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 325. Beyond Factor Aggregation: Gauge-Aware Low-Rank Server Representations for Federated LoRA

**arXiv ID:** 2605.06733 | [PDF](https://arxiv.org/pdf/2605.06733v1)

**作者:** Jinqian Chen `[一作]` (Xi'an Jiaotong University), Jihua Zhu `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 2891 | [OpenAlex ID](https://openalex.org/A5068185614)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 GLoRA，一种在联邦学习中对 LoRA 更新进行基于子空间的 gauge‑aware 聚合方法，能够在保持低秩通信开销的同时兼容不同客户端的资源约束。

**💡 创新点**

创新点：
1) 将 LoRA 更新拆解为 gauge‑不变的子空间投影与坐标，避免因 gauge‑等价变换导致的聚合错误。
2) 通过对所有客户端投影的投影矩阵求特征值，学习一个共识子空间作为服务器端的低秩表示。
3) 提供“rank‑compatible readout”机制，使同一服务器状态能按不同 rank 分发给不同能力的客户端，无需重构稠密矩阵。

**🔧 技术方法**

核心技术：LoRA 参数化、QR 分解（获取列空间基），子空间投影与投影矩阵求特征值，低秩投影坐标聚合，Spectral/核心‑尾读取策略。

**📊 数据集**

实验数据集：GLUE（MNLI、SST‑2、QQP、QNLI 等）、SuperNI（跨任务 FL）、以及大模型 Qwen2‑7B、Gemma‑2‑9B 的评估。

**📈 对比分析**

与 FedIT、FFA‑LoRA、FedEx‑LoRA、FedSA‑LoRA、HetLoRA、FlexLoRA 等基线比较。GLoRA 在所有数据、资源与任务异质性场景下均显著提升性能，尤其在高异质性（Dirichlet 0.1）和稀疏参与下保持稳定，且在大模型上也保持优势。效率方面，GLoRA 的通信与计算成本低于 dense‑update 聚合方法，接近最优的 factor‑level 方法。

**⚠️ 局限性**

局限性：
1) 共识子空间仅基于客户端列空间几何构建，未利用更新能量或任务标签信息，可能导致信息丢失。
2) 对极端极小 rank 或极大 rank 组合的理论上限和收敛性分析尚不完整。
3) 在非常低的通信预算下，子空间投影可能不足以捕捉所有重要更新。

---

## 326. HMACE: Heterogeneous Multi-Agent Collaborative Evolution for Combinatorial Optimization

**arXiv ID:** 2605.07214 | [PDF](https://arxiv.org/pdf/2605.07214v1)

**作者:** Yuping Yan `[一作]` (Westlake University), Yaochu Jin `[通讯]` (Westlake University)

**通讯引用:** 56428 | [OpenAlex ID](https://openalex.org/A5032314861)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种利用大型语言模型的异质多智能体协作进化框架HMACE，自动化搜索并生成可复用的组合优化启发式程序。

**💡 创新点**

创新点在于将进化搜索拆解为四个专职角色（Proposer、Generator、Evaluator、Reflector），并通过行为感知检索与档案更新实现高效、多样化的探索与记忆驱动的反射。

**🔧 技术方法**

技术包括LLM驱动的自然语言策略提议、代码生成与执行、基于行为向量的档案划分、轻量级过滤与多代迭代的协同流程。

**📊 数据集**

实验数据集涵盖经典组合优化任务：TSP（EUC_2D 50–20k城市）、Online BPP（Weibull‑5k）、MKP（10个随机实例）以及PFSP（Taillard）。

**📈 对比分析**

与单智能体、同质多智能体及传统启发式基线（Concorde、OR‑Tools、FunSearch、EoH、CORAL等）比较，HMACE在TSP与在线BPP上实现了平均最小相对缺口（0.464%/0.223%）且令标记token量仅为0.13M/0.42M，显著优于对手。

**⚠️ 局限性**

局限性包括：流程预设固定，缺乏动态角色或工作流自适应；以及长期记忆规模扩展时需更精细的选择与裁剪机制，否则信息过载会削弱检索效果。

---

## 327. AGA3DNet: Anatomy-Guided Gaussian Priors with Multi-view xLSTM for 3D Brain MRI Subtype Classification

**arXiv ID:** 2605.07142 | [PDF](https://arxiv.org/pdf/2605.07142v1)

**作者:** Peiyu Duan `[一作]` (Yale University), Yoshihisa Shinagawa `[通讯]` (Siemens Healthineers)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于报告提取解剖线索的3D脑MRI子型分类框架AGA3DNet，利用Gaussian加权的解剖先验与轻量化3D CNN和多视角xLSTM联合学习；

**💡 创新点**

创新点在于将放射报告中简短解剖短语映射到脑区掩模生成软空间先验，从而在不使用完整文本的情况下引入解剖知识，同时采用焦点+对比损失实现类不平衡处理和特征分离；

**🔧 技术方法**

采用LLM提取解剖短语、SynthSeg+MedSAM2生成解剖掩模、signed distance + Gaussian平滑生成先验、3D卷积编码器、xLSTM多视角序列建模、联合焦点损失与InfoNCE对比损失；

**📊 数据集**

使用内部216名患者的581份3D T2/FLAIR扫描，按患者拆分为70%训练、15%验证、15%测试，正负类分别为病灶/脱髓鞘与其他脑异常；

**📈 对比分析**

与ResNet、MedMamba、nnMamba、Swin-UNETR、ResNet–mLSTM等基线对比，AGA3DNet在准确率0.825、AUC0.724、召回率0.988等指标均优于基线，表现出最佳的敏感度与特异度平衡；

**⚠️ 局限性**

局限性包括仅在单中心数据上评估、放射报告短语提取质量不稳定、解剖掩模覆盖范围有限、缺乏大规模公开伴随报告的脑MRI数据集等。

---

## 328. Integrating Causal DAGs in Deep RL: Activating Minimal Markovian States with Multi-Order Exposure

**arXiv ID:** 2605.07057 | [PDF](https://arxiv.org/pdf/2605.07057v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 329. The Cost of Quantum Resistance: A Hash-Based Commit-Reveal Alternative for Minimizing Blockchain Infrastructure Overhead

**arXiv ID:** 2605.06853 | [PDF](https://arxiv.org/pdf/2605.06853v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 330. AdaTKG: Adaptive Memory for Temporal Knowledge Graph Reasoning

**arXiv ID:** 2605.07121 | [PDF](https://arxiv.org/pdf/2605.07121v1)

**作者:** Seunghan Lee `[一作]` (LG AI Research), Wonbin Ahn `[通讯]` (LG AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出AdaTKG，一种为每个实体维护可学习的指数移动平均记忆的适应性TKG推理方法。

**💡 创新点**

创新点在于将实体表示视为可自适应的过程，记忆通过共享的EMA更新实现，无需为每个实体存储可学习参数，从而支持训练时未出现实体的推理。

**🔧 技术方法**

使用了Transformer编码交互信号、共享EMA更新、可学习门控融合静态+记忆表示，并基于ConvTransE解码。

**📊 数据集**

在四个公开TKG基准（ICEWS14/18/05-15、GDELT）上进行实验。

**📈 对比分析**

与TransFIR等多种静态+归纳基线对比，AdaTKG在新出现实体和未知实体上均提升5‑25% MRR，整体表现最优。

**⚠️ 局限性**

局限在于需要持久化实体记忆，且未解决长周期分布漂移，未来可结合动态码表。

---

## 331. Optimal Experiments for Partial Causal Effect Identification

**arXiv ID:** 2605.06993 | [PDF](https://arxiv.org/pdf/2605.06993v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 332. WiCER: Wiki-memory Compile, Evaluate, Refine Iterative Knowledge Compilation for LLM Wiki Systems

**arXiv ID:** 2605.07068 | [PDF](https://arxiv.org/pdf/2605.07068v1)

**作者:** Juan M. Huerta `[一作]` `[通讯]` (Zinnia Tech Solutions), Juan M. Huerta (Zinnia Tech Solutions)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了基于 KV 缓存的全上下文推理在大型原始文档上的局限性，并提出了 WiCER（Wiki-memory Compile, Evaluate, Refine）迭代编译-评估-改进流程，以填补所谓的编译缺口。

**💡 创新点**

创新点在于将 Counterexample‑Guided Abstraction Refinement（CEGAR）思路迁移到知识编译领域，通过诊断评估失败的 probe 并强制保留关键事实，显著恢复 80% 的质量并将灾难性失败率下降 55%。

**🔧 技术方法**

使用技术包括 KV 缓存推理、Flash Attention、Llama 3.1 8B 量化模型、RAG 检索、LLM‑as‑judge 评估以及诊断‑保留约束的编译器。

**📊 数据集**

实验数据集为 Policygenius（30 篇策划文章）和 17 个领域的 RepLiQA（1,360 篇文档、6,800 题目答案对）。

**📈 对比分析**

与 RAG 进行全面对比，评估维度包括回答质量（1–5 分）、TTFT、吞吐量等；在 30 篇精编文档上全上下文优于 RAG（+0.3 分、7.3× TTFT），在 80 篇原始文档上 RAG 更好；WiCER 迭代后恢复到 3.24 分，灾难性失败率降至 24.8%，与 RAG 差距缩小至 0.46 分，同时 TTFT 仍保持 12× 更快。

**⚠️ 局限性**

局限性包括：实验仅在 Llama 3.1 8B / Apple M4 Pro 上验证；RAG 使用固定分块且不做重排序；部分主题 WiCER 无提升；诊断与重编译成本与 API 费用较高；编译器可能无法严格遵守压缩目标，导致信息丢失。

---

## 333. Social Theory Should Be a Structural Prior for Agentic AI: A Formal Framework for Multi-Agent Social Systems

**arXiv ID:** 2605.07069 | [PDF](https://arxiv.org/pdf/2605.07069v1)

**作者:** Lynnette Hui Xian Ng `[一作]` (Carnegie Mellon University), Kathleen M. Carley `[通讯]` (Carnegie Mellon University)

**通讯引用:** 28308 | [OpenAlex ID](https://openalex.org/A5085927300)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在论文中作者提出了 Multi-Agent Social Systems（MASS）框架，旨在把社会理论作为结构先验，系统化描述 Agentic AI 在社交环境中的四大先验（战略异质性、网络受限依赖、共演化、分布不稳定）并通过代理仿真实验验证其有效性。

**💡 创新点**

创新点在于将社会学中的角色理论、网络嵌入、共演化和议程设置等概念转化为可操作的结构先验，并在 MASS 这一动态网络模型中形式化表达与推导，首次将 Agentic AI 与社会科学的结构先验紧密结合。

**🔧 技术方法**

主要采用了图结构化建模（定义信息交换函数 f、影响动力学函数 g 与网络结构 G），结合图卷积或网络重连机制的代理仿真，以及理论证明和形式化命题的推导。

**📊 数据集**

实验使用人工构造的异质代理群体（包含放大器代理、普通代理等）和多种图网络结构（小世界、无标度），并未使用公开真实数据集。

**📈 对比分析**

通过对比不同代理配置、网络拓扑与扰动实验，观察群体均值态演化、方差变化及分布漂移等指标，实验结果表明四大先验对系统级结果具有显著影响，虽未给出传统机器学习指标，但展示了明显的差异性。

**⚠️ 局限性**

局限性包括：仅在模拟环境中验证，缺乏真实社交平台的数据支撑；模型的评估标准与性能指标尚未标准化；以及对 MASS 框架在大规模真实系统中的可扩展性与治理机制的进一步探索仍待开展。

---

## 334. More Thinking, More Bias: Length-Driven Position Bias in Reasoning Models

**arXiv ID:** 2605.06672 | [PDF](https://arxiv.org/pdf/2605.06672v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 335. Modulated learning for private and distributed regression with just a single sample per client device

**arXiv ID:** 2605.07233 | [PDF](https://arxiv.org/pdf/2605.07233v1)

**作者:** Praneeth Vepakomma `[一作]` (Massachusetts Institute of Technology), Munther Dahleh `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在极端场景下提出一种联邦学习框架：每个客户端仅持有一条数据样本，采用余弦调制的单向变换并加入高斯噪声，随后在服务器端进行去偏处理，得到无偏梯度估计，从而在保持隐私的同时实现线性回归模型的训练；此外还给出了单次一次性估计器和多向量版本，并提供理论分析和实验验证。

**💡 创新点**

核心创新在于：①只需在每个客户端生成一次隐私化的特征向量（单噪声、单变换），即可在服务器端恢复无偏梯度；②相较于传统DP‑SGD直接对梯度加噪，本文通过发送隐私化样本实现更高效的通信与更低的噪声；③引入多向量余弦调制降低 Lipschitz 常数、减小噪声量，理论上提升了方差与收敛速度。

**🔧 技术方法**

使用技术包括：局部差分隐私（Gaussian 机制）+ Lipschitz‑约束的余弦调制变换；服务器端无偏梯度恢复与投影正则化；zCDP 隐私计数；一次性估计器基于矩估计与岭回归；理论上给出方差、收敛和误差下界。

**📊 数据集**

实验使用五个真实回归数据集：CO₂、Fair（years_married）、ModeChoice、RandHIE（lncoins）和 RandHIE（fmde），每条样本视为单独客户端。

**📈 对比分析**

与对标的 DP‑SGD FedAvg（经过调参）以及无隐私 OLS 基线进行对比；结果显示：在强隐私（ε 较小）下，一次性模糊估计器表现最优；随着 ε 增大，多轮迭代方法逐步赶超；总体在足够隐私预算时，性能接近无隐私 OLS。

**⚠️ 局限性**

局限性包括：仅针对线性回归；假设响应变量 y 为公开/非敏感；需要对超参数（α、λ、ω、步长、岭系数等）精细调节；维度升高会显著提升噪声与方差；多轮迭代受 zCDP 顺序组合限制；若所有客户端不全参与，性能可能下降。

---

## 336. Physics-based Digital Twins for Integrated Thermal Energy Systems Using Active Learning

**arXiv ID:** 2605.06756 | [PDF](https://arxiv.org/pdf/2605.06756v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 337. Narrow Secret Loyalty Dodges Black-Box Audits

**arXiv ID:** 2605.06846 | [PDF](https://arxiv.org/pdf/2605.06846v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 338. Toeplitz MLP Mixers are Low Complexity, Information-Rich Sequence Models

**arXiv ID:** 2605.06683 | [PDF](https://arxiv.org/pdf/2605.06683v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 339. Towards Closing the Autoregressive Gap in Language Modeling via Entropy-Gated Continuous Bitstream Diffusion

**arXiv ID:** 2605.07013 | [PDF](https://arxiv.org/pdf/2605.07013v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 340. Faculty mobility reallocates research capacity within persistent institutional hierarchies

**arXiv ID:** 2605.06935 | [PDF](https://arxiv.org/pdf/2605.06935v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 341. Medical Imaging Classification with Cold-Atom Reservoir Computing using Auto-Encoders and Surrogate-Driven Training

**arXiv ID:** 2605.06727 | [PDF](https://arxiv.org/pdf/2605.06727v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 342. Mutation-Guided Differentiable Quadratic Combinatorial Optimization

**arXiv ID:** 2605.06921 | [PDF](https://arxiv.org/pdf/2605.06921v1)

**作者:** Yongliang Sun `[一作]` (Michigan State University), Rongrong Wang `[通讯]` (Michigan State University)

**通讯引用:** 2376 | [OpenAlex ID](https://openalex.org/A5100733176)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种新的“mutation-based differentiable global reset”框架（mQO），通过梯度优化、可微分重置和离散局部搜索相结合，解决最大独立集（MIS）和最大割（MaxCut）问题。

**💡 创新点**

创新点包括：① 对松弛盒约束 QUBO 的局部极值进行理论分析，揭示梯度方法停滞的根本原因；② 针对 MaxCut 提出新的偏置扰动目标 f_B，消除非二进制驻点并能逃离 1-flip 可修复点；③ 设计基于随机重置的全局搜索策略，显著提升搜索半径并减少局部最优停滞。

**🔧 技术方法**

使用的技术主要有：投影梯度上升（PGA）/Adam 等梯度优化器；基于邻接矩阵的 QUBO 表达式；梯度+随机重置的可微分全局搜索；传统的 (1,2)-swap（MIS）和 1/2-flip（MaxCut）局部搜索；以及对目标函数的 λ‑偏置扰动。

**📊 数据集**

实验数据集包括 Erdős–Rényi 随机图（n=1k–40k，密度多样），DIMACS 基准图，随机满足性（RB）图，Barabasi–Albert（BA）图，随机块模型（SBM）。

**📈 对比分析**

与 CPU 传统启发式（Greedy‑MIS、ReduMIS、CQO、Gurobi）、GPU 并行方法（pCQO、RLSA）以及 MaxCut 启发式（Greedy、BLS、Gurobi）进行对比。结果表明，mQO 在大规模实例（n≥30k）下在 5–10 分钟预算内可获得更大的 MIS 规模或更高的割值，且在 GPU 版 pmQO 上进一步提升性能；在小规模实例上仍略逊于最强启发式。

**⚠️ 局限性**

局限性：① 对局部最优的逃逸仍依赖随机重置参数 ρ 与 λ 的调优；② 在极小规模或结构特殊图（如稀疏度极低）时性能不如成熟启发式；③ 当前仅针对 MIS 和 MaxCut 证明，可扩展到其他 COP 的通用性尚未完全验证；④ 仍存在梯度停滞情况，尤其在高密度图中需更频繁重置。

---

## 343. Conformal-Style Quantile Analyses for Stochastic Bandits

**arXiv ID:** 2605.07115 | [PDF](https://arxiv.org/pdf/2605.07115v1)

**作者:** Chengyu Du `[一作]` (University of Massachusetts Amherst), Mengfan Xu `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5083293925)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种基于自适应置信区间的 ACP-UCB1 算法，用来在多臂赌博机中优化预测区间上限的上尾部性能。

**💡 创新点**

创新点在于将可重计算的 conformal 评分量化与 UCB 探索相结合，构造自适应置信水平更新，并证明在局部密度正则条件下实现对数伪损失上界。

**🔧 技术方法**

使用 conformal 预测、经验分位数估计、UCB 探索奖励、局部自适应置信更新和密度正则化的集中不等式。

**📊 数据集**

在合成数据集上实验，包括正态、学生 t 分布及其偏斜变体（Gaussian、Student‑t、skewed Student‑t），未使用真实工业数据。

**📈 对比分析**

与经典 UCB1（基于均值）以及 UCB1‑NORMAL 进行对比；在上尾目标下 ACP‑UCB1 取得对数级别误差，平均目标下 UCB1 更优；当两目标选择不同臂时，另一方表现出现线性误差。

**⚠️ 局限性**

局限性包括：依赖 i.i.d. 与局部密度正则假设；适用范围受限于上尾预测区间；自适应置信更新计算开销；在非正态或极端重尾分布下性能可能退化；未涵盖自适应奖励变化或对抗性环境。

---

## 344. Pretraining Induces a Reusable Spectral Basis for Downstream Task Adaptation

**arXiv ID:** 2605.07302 | [PDF](https://arxiv.org/pdf/2605.07302v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 345. Variable Aerodynamic Damping via Co-Contraction: A Dynamic Isomorphism with Variable Stiffness Actuators

**arXiv ID:** 2605.07292 | [PDF](https://arxiv.org/pdf/2605.07292v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 346. Incentive Design in Competitive Resource Allocation: Exploiting Valuation Asymmetry in Tullock Contests

**arXiv ID:** 2605.07045 | [PDF](https://arxiv.org/pdf/2605.07045v1)

**作者:** Gilberto Diaz-Garcia `[一作]`, Jason R. Marden `[通讯]` (University of California, Santa Barbara)

**通讯引用:** 5994 | [OpenAlex ID](https://openalex.org/A5003146330)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究多玩家Tullock竞赛中，中央协调者通过误报子代理的价值来影响其行为，从而提升自身在竞赛中的收益。

**💡 创新点**

提出利用子代理的价值设置（保留Tullock结构）来重新设计激励，并给出在两子代理与单一对手、以及任意多子代理情况下的最优价值闭式表达与结构性质。

**🔧 技术方法**

采用Nash均衡分析、最优化求解、闭式解推导以及维度降解技术（将问题简化为仅包含两个变量）。

**📊 数据集**

本文未使用具体数据集，全部以理论模型和数学推导为主。

**📈 对比分析**

文章未提供与其他方法的实验比较或性能数值，只通过解析结果说明在相同模型下最优设计可显著提升协调者收益。

**⚠️ 局限性**

局限性包括仅考虑单一协调者、假设Tullock竞赛形式且缺乏参数不确定性；未探讨多协调者或更复杂信息结构的情况。

---

## 347. Region4Web: Rethinking Observation Space Granularity for Web Agents

**arXiv ID:** 2605.07134 | [PDF](https://arxiv.org/pdf/2605.07134v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 348. TajPersLexon: A Tajik-Persian Lexical Resource and Hybrid Model for Cross-Script Low-Resource NLP

**arXiv ID:** 2605.06886 | [PDF](https://arxiv.org/pdf/2605.06886v1)

**作者:** Mullosharaf K. Arabov `[一作]` (Kazan Federal University), Mullosharaf K. Arabov `[通讯]` (Kazan Federal University)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5099178332)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出TajPersLexon，一套40,112条 Tajik–Persian 并行词表，并在此基础上对跨脚本检索进行了系统评测。

**💡 创新点**

首次构建并公开了规模最大的 Tajik–Persian 机器可读词典，同时展示了轻量级混合模型在 CPU-only 环境下的高效可解释性，并揭示了多语言句子变压器在此任务中的失效。

**🔧 技术方法**

使用 SentencePiece 子词分词、FastText/Word2Vec 分布式嵌入、编辑距离、规则式音译以及 LSTM/Transformer 序列到序列、BM25 检索和多语言句子变压器等多种技术。

**📊 数据集**

核心数据集为自编的 TajPersLexon（约 40k 对，含 POS 与示例句），并与现有数字语料库、传统词典相结合进行精炼。

**📈 对比分析**

在 1,000 个负样本的检索实验中，混合模型 Acc@1 为 4.8%，但 BM25、Transformer 和 LSTM 近 99%；在 OCR 噪声修正任务中，混合模型达 96.4% 纠错准确率，证明其实用性。

**⚠️ 局限性**

限制包括词表 POS 分布偏斜、对动词、专有名词等类别的表现欠佳、缺乏上下文与多义词处理，以及对句子级多语言变压器的不足，未来需进一步扩充数据与改进模型。

---

## 349. RELO: Reinforcement Learning to Localize for Visual Object Tracking

**arXiv ID:** 2605.07379 | [PDF](https://arxiv.org/pdf/2605.07379v1)

**作者:** Xin Chen `[一作]` (City University of Hong Kong), Kede Ma `[通讯]` (City University of Hong Kong)

**通讯引用:** 9972 | [OpenAlex ID](https://openalex.org/A5020029652)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于强化学习的目标定位方法 RELO，取代传统手工设计的空间先验，实现直接从任务指标（IoU、AUC）获得监督；

**💡 创新点**

创新点在于将目标定位建模为马尔可夫决策过程，使用 actor‑critic 强化学习直接优化任务相关奖励；引入回归预热阶段稳定学习，并提出层对齐的时序 token 传播以提升跨帧语义一致性；

**🔧 技术方法**

核心技术包括一流 Transformer（HiViT）骨干、回归/策略/价值头、PPO/GRPO 等 RL 算法、层对齐的时序 token 传播、帧级 IoU 与序列级 AUC 混合奖励；

**📊 数据集**

训练集使用 COCO、LaSOT、GOT‑10k、TrackingNet、VastTrack；评测集涵盖 LaSOT、LaSOT_ext、TrackingNet、GOT‑10k、TNL2K、NFS、UAV123；

**📈 对比分析**

与多种最新跟踪器（OSTrack、SwinTrack、SeqTrack、ARTrack、OneTracker、SUTrack、ARPTrack、MixFormer、SimTrack、ODTrack、ARTrackV2、LoRAT）进行对比；在 LaSOT 上获得 75.1% AUC、LaSOT_ext 57.5% AUC，成为大模型组最佳；在 TrackingNet、GOT‑10k 等短期 benchmark 亦排名第一；轻量版 RELO‑T256 在效率赛道上领先；

**⚠️ 局限性**

局限性包括：奖励仅基于 IoU/AUC，难以覆盖遮挡、目标消失、干扰、突变等复杂场景；仅针对单目标单摄像头，未扩展到多目标、多摄像头；RL 仅用于定位，未覆盖模板更新、搜索区调整等完整跟踪决策；未来需设计更丰富奖励、跨视角多目标策略以及全流程 RL 控制。

---

## 350. Rethinking Dense Sequential Chains: Reasoning Language Models Can Extract Answers from Sparse, Order-Shuffling Chain-of-Thoughts

**arXiv ID:** 2605.07307 | [PDF](https://arxiv.org/pdf/2605.07307v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 351. Solving Max-Cut to Global Optimality via Feasibility-Preserving Graph Neural Networks

**arXiv ID:** 2605.07113 | [PDF](https://arxiv.org/pdf/2605.07113v1)

**作者:** Hao Chen `[一作]` (Purdue University), Can Li `[通讯]` (Purdue University)

**通讯引用:** 11048 | [OpenAlex ID](https://openalex.org/A5100334065)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

做了一个针对Max-Cut SDP的可行性保留图神经网络代理，替代传统 SDP 求解器用于分支定界流程；

**💡 创新点**

创新点在于利用Max-Cut SDP的稀疏结构与对角约束，提出 O(n²+ne) 复杂度的 Max‑Cut Weisfeiler–Leman GNN，并通过径向投影保证预测的原始与对偶解都可行，实现自监督训练；

**🔧 技术方法**

采用高阶表达式的图神经网络（Max‑Cut WL）、径向投影确保对偶可行、Goemans–Williamson 近似、批量推理以及与 Mosek 的对比；

**📊 数据集**

使用 Erdős–Rényi 随机图与公开 Max‑Cut 基准库（如 Gurobi/BiqCrunch 提供的实例）作为训练与测试数据；

**📈 对比分析**

与标准基于 Mosek 的 B&B、混合 GNN+Mosek 以及全神经 B&B 进行对比，神经 B&B 在节点评估速度提升约 5–10.6 倍，整体求解时间显著下降；

**⚠️ 局限性**

局限在于对分布外图的泛化有限，且尚未能完全击败最先进的传统 Max‑Cut 求解器（如 BiqCrunch）。

---

## 352. On the Robustness of Distribution Support under Diffusion Guidance

**arXiv ID:** 2605.07220 | [PDF](https://arxiv.org/pdf/2605.07220v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 353. When Symbol Names Should Not Matter: A Logistic Theory of Fresh-Symbol Classification

**arXiv ID:** 2605.07120 | [PDF](https://arxiv.org/pdf/2605.07120v1)

**作者:** Wenjie Guan `[一作]` (Cornell University), Jelena Bradic `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了Transformer在“新符号”分类任务中的泛化机制，提出了通过碰撞图（collision graph）对核逻辑回归（kernel logistic regression）学习过程中的有限样本扰动进行定量分析，从而给出了保证新符号样本分类正确性的理论界限。

**💡 创新点**

创新点包括：① 用碰撞图精细刻画样本级符号重叠对核矩阵的扰动；② 将传统的标量多样性指标拆解为颜色对、度数、谱范数等几何量，得到更细粒度的泛化保证；③ 对一次抽象提示（abstract prompting）的第一阶影响进行分析，阐明何时抽象能够提升新符号泛化；④ 将二分类推理扩展到多分类软最大化（softmax）框架。

**🔧 技术方法**

技术手段：核逻辑回归、Transformer冷冻特征的随机特征/NTK极限、熵对偶（entropy dual）、曲率-谱分析、块平均/ANOVA、偏差-波动证书、抽象提示的Taylor展开、以及多层变换器中的K-Q、V-O乘子调节。

**📊 数据集**

数据集：主要是人工合成的模板分类任务，包括二分类的αβα→+1 vs αββ→-1，以及多数投票任务{ααα,ααβ,αβα,αββ}，每个任务通过在随机词表上注入活跃wildcard进行多次采样得到训练/测试样本。

**📈 对比分析**

比较方法：在同一层Transformer架构下，实验了不同K-Q、V-O乘子组合对模型性能的影响。实验显示：① 纯粹的Transformer需要大量样本才能实现新符号泛化；② 通过非零乘子显著提升小样本阶段的准确率；③ K-Q乘子对二分类任务更为关键，而V-O乘子在下一词预测任务的小样本阶段更为重要；④ 组合使用两者往往得到最佳性能。

**⚠️ 局限性**

局限性：① 仅在一层Transformer的随机特征/NTK极限下推导，未考虑深层网络的表示学习和训练动态；② 研究聚焦于合成模板任务，缺乏对真实自然语言数据的验证；③ 只分析了冻结特征下的核逻辑回归，未涉及优化收敛、宽度有限等实际训练细节。

---

## 354. Demystifying and Detecting Agentic Workflow Injection Vulnerabilities in GitHub Actions

**arXiv ID:** 2605.07135 | [PDF](https://arxiv.org/pdf/2605.07135v1)

**作者:** Shenao Wang `[一作]` (Huazhong University of Science and Technology), Haoyu Wang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 66226 | [OpenAlex ID](https://openalex.org/A5115602103)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并系统地识别了 GitHub Actions 中因 LLM 代理工作流导致的注入漏洞（AWI），并构建了相应的检测工具。

**💡 创新点**

首次提出 AWI 概念、定义两类攻击模式（P2A 与 P2S），并为此设计了专门的工作流 IR、依赖图与污点传播模型，显著提升了对 agentic 工作流的安全分析能力。

**🔧 技术方法**

核心技术包括：基于 LLM 的动作层污点规格推断、YAML 解析生成工作流中间表示（IR）、构建 Agentic Workflow Dependency Graph（AWDG）、污点传播与可达性分析，以及对脚本与守卫（guards）进行静态检测。

**📊 数据集**

使用 1,033 个 AI 辅助 Action 及其 13,392 个实际工作流（来自 10,792 个仓库）进行评估，并对 496 个确认可利用的 AWI 漏洞进行手工验证。

**📈 对比分析**

与现有基线（Prompt Injection Scanner、OpenGrep、Argus、CodeQL）相比，工具在 496 个真阳性案例上覆盖率 100%，P2S 识别率显著提升（+418 个新案例），误报率低至 4.4%，执行时间仅约 12.3 分钟，吞吐量 18.1 条/秒。

**⚠️ 局限性**

主要局限包括：对 Action 行为的规格推断依赖 LLM 预测、可能漏掉新/长尾 Action；静态分析的过度近似导致部分不可实际利用的假阳性；对脚本的分析仅覆盖常见 Shell/JavaScript，无法处理复杂多语言脚本；无法完全模拟 GitHub 运行时的细粒度权限与执行细节。

---

## 355. Arrow: A Foundation Model for Causal Discovery

**arXiv ID:** 2605.07204 | [PDF](https://arxiv.org/pdf/2605.07204v1)

**作者:** Ryan Thompson `[一作]` (University of Technology Sydney), Edwin V. Bonilla `[通讯]` (CSIRO's Data61)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于预训练的零样本因果结构发现模型，能够在观察性表格数据上直接预测DAG。

**💡 创新点**

主要创新点包括：① 通过Skeleton–Order分解实现对DAG的显式无环保证；② 采用Transformer多阶段嵌入实现变量间全局上下文化；③ 在超过1亿个多样化合成任务上进行监督预训练，实现零样本推断与高效推理。

**🔧 技术方法**

技术手段：Transformer编码器（无位置编码实现可交换性），Skeleton–Order概率模型（product‑Bernoulli + Plackett–Luce），复合似然训练目标，基于节点得分排序的最大概率预测。

**📊 数据集**

使用的数据集包括：① 超过1亿条合成任务（随机图、功能形式、噪声模型与样本/变量规模多样化）；② 半合成数据（来自Bayesian Network Repository的4个真实网络，变量数46–107，边数66–150）；③ 公开的真实实验数据（如流式细胞仪数据）。

**📈 对比分析**

与5个基线（包括两种预训练模型和两种任务特定方法）进行对比；在分布内外、半合成和真实数据上取得或超过nSHD、F1、AP的最优/接近最优成绩；推理速度比任务特定方法快1–3个数量级，且零样本推断不需要额外搜索或优化。

**⚠️ 局限性**

局限性：1）仅适用于无潜在混杂器的纯观察性数据；2）对因果可辨识性依赖传统假设，未覆盖不可辨识情形；3）模型规模与预训练成本较大，尚未在极大样本/变量规模（>10k）上验证；4）缺乏对干预数据或多任务/跨域迁移的深入研究。

---

## 356. Online Allocation with Unknown Shared Supply

**arXiv ID:** 2605.07080 | [PDF](https://arxiv.org/pdf/2605.07080v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 357. CASCADE: Context-Aware Relaxation for Speculative Image Decoding

**arXiv ID:** 2605.07230 | [PDF](https://arxiv.org/pdf/2605.07230v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 358. AsyncEvGS: Asynchronous Event-Assisted Gaussian Splatting for Handheld Motion-Blurred Scenes

**arXiv ID:** 2605.07192 | [PDF](https://arxiv.org/pdf/2605.07192v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 359. HumanNet: Scaling Human-centric Video Learning to One Million Hours

**arXiv ID:** 2605.06747 | [PDF](https://arxiv.org/pdf/2605.06747v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 360. Learning Multi-Relational Graph Representations for DNA Methylation-Based Biological Age Estimation

**arXiv ID:** 2605.07175 | [PDF](https://arxiv.org/pdf/2605.07175v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 361. Rethinking Priority Scheduling for Sequential Multi-Agent Decision Making in Stackelberg Games

**arXiv ID:** 2605.07240 | [PDF](https://arxiv.org/pdf/2605.07240v1)

**作者:** Xiangyu Liu `[一作]` (Dalian University of Technology), Ziqi Wei `[通讯]` (Dalian University of Technology)

**通讯引用:** 1688 | [OpenAlex ID](https://openalex.org/A5079553918)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于层次优先调整(HPA)的多智能体强化学习框架，动态决定执行顺序以提升协作性能

**💡 创新点**

证明了执行顺序对N阶Stackelberg均衡的理论影响，并提出可学习的优先级调度策略

**🔧 技术方法**

结合Stackelberg游戏理论、Spatio‑Temporal Sequential Markov Game、option‑critic、PPO以及HAPPO/ HATRPO等强化学习方法

**📊 数据集**

在MuJoCo多智能体控制任务（HalfCheetah、Walker2d）上进行实验

**📈 对比分析**

与MAPPO、HAPPO、HATRPO、STEP等先进算法对比，HPA在四种复杂协作环境中均实现了更高的平均/最大奖励，证明了动态优先级调度的有效性

**⚠️ 局限性**

模型受限于智能体分组导致的序列可行性受限，以及对超参数k的敏感性，未来需进一步改进序列自适应机制

---

## 362. SOM: Structured Opponent Modeling for LLM-based Agents via Structural Causal Model

**arXiv ID:** 2605.07301 | [PDF](https://arxiv.org/pdf/2605.07301v1)

**作者:** Shiyue Cao `[一作]` (University of Chinese Academy of Sciences), Kaiqi Huang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 14985 | [OpenAlex ID](https://openalex.org/A5028693655)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种两阶段对手建模框架SOM，通过结构化因果模型显式地构建和更新对手决策过程，然后利用该结构进行预测与自适应

**💡 创新点**

核心创新在于将对手建模与预测分离，动态构建结构因果图并利用个性化推理示例填充结构方程，既保持可解释性又实现自适应更新

**🔧 技术方法**

利用大型语言模型进行因果图推理、反思提取中间变量、示例检索实现结构化推理；基于结构因果模型（SCM）与自然语言推断技术

**📊 数据集**

在三类多人游戏环境评测：G0.8A（猜数游戏）、Survival Auction Game（密封竞标游戏）和Undercover Game（社会推理游戏）

**📈 对比分析**

与CoT、ToT、K-R、Reflexion等四种LLM推理基线以及混合对手基线进行对比；SOM在所有环境均取得最高平均胜率/存活回合数，尤其在混合对手场景中表现显著优于对手，示例化和图优化模块显著降低预测误差并提升胜率

**⚠️ 局限性**

所学因果结构仅为功能依赖，未通过因果实验验证；模型依赖LLM推理质量，对低容量模型迁移仍有性能差距，且图结构复杂度需控制

---

## 363. Teacher-Feature Drifting: One-Step Diffusion Distillation with Pretrained Diffusion Representations

**arXiv ID:** 2605.07327 | [PDF](https://arxiv.org/pdf/2605.07327v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 364. Semantic State Abstraction Interfaces for LLM-Augmented Portfolio Decisions: Multi-Axis News Decomposition and RL Diagnostics

**arXiv ID:** 2605.06730 | [PDF](https://arxiv.org/pdf/2605.06730v1)

**作者:** Likhita Yerra `[一作]`, Remi Uttejitha Allam `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种语义状态抽象接口（SSAI），将稀疏新闻文本映射为可审计的多维坐标，并通过共享固定映射评估文本表示与优化算法的独立效果。

**💡 创新点**

创新点在于：①构建可审计、可解释的多轴 LLM 接口；②设计共享表示的评估框架，分离表示与优化误差；③量化解释性与性能之间的成本，并提供可复用的实验模板。

**🔧 技术方法**

使用 OpenAI ChatGPT 进行零样本多轴评分（情绪、风险、置信度、波动率），并在 Ridge 回归、因素组合（SFP/SRF/SCW）与深度强化学习（DP‑PPO、SAC）中共享此表示；对齐训练后使用 FinRL 环境进行交易模拟。

**📊 数据集**

数据集为 30 家 NASDAQ‑100 成分股的 2019‑2023 年测试期价格数据（Yahoo Finance）以及对应的 40,850 条 FNSPID 新闻文章。

**📈 对比分析**

比较方法包括：因素组合对比（SSAI‑SFP、PC1‑SFP、FinBERT‑SFP）、监督 Ridge 预测、RL 代理；性能方面，SSAI‑SFP 在 2019‑2023 年的累计回报 307.2%（Sharpe 1.07），低于 PC1‑SFP（433.6%）和 FinBERT‑SFP（386.3%），且在交易成本 0.2% 以上可逆；SAC 在相同表示下的 Sharpe 1.06 高于 DP‑PPO 0.92，但累计回报差异不显著。

**⚠️ 局限性**

局限性包括：①效果主要受篮子选择和交易成本影响，解释性成本难以彻底消除；②仅在单一 NASDAQ‑100 市场和时间窗口内验证；③RL 对比缺乏相同超参数的充分对齐；④未测试更深层次的 LLM 编码（FinGPT/BloombergGPT）作为 RL 状态向量。

---

## 365. Attribution-Based Neuron Utility for Plasticity Restoration in Deep Networks

**arXiv ID:** 2605.06834 | [PDF](https://arxiv.org/pdf/2605.06834v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 366. Learning to Track Instance from Single Nature Language Description

**arXiv ID:** 2605.07064 | [PDF](https://arxiv.org/pdf/2605.07064v1)

**作者:** Yaozong Zheng `[一作]` (Guangxi Normal University), Shuxiang Song `[通讯]` (Guangxi Normal University)

**通讯引用:** 1628 | [OpenAlex ID](https://openalex.org/A5025660318)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出一种全自监督的视觉‑语言跟踪方法，完全基于自然语言描述，无需边框标注；通过弱‑强数据增强、动态令牌聚合与去噪训练实现跨模态对齐与自适应跟踪；

**💡 创新点**

创新点包括：1) 动态令牌聚合模块（DTA），可在视觉令牌中自适应挑选最有语义信息的目标子令牌并融合至语言令牌；2) 弱‑强一致性框架，利用弱强增强对比提升自监督学习鲁棒性；3) 纯语言驱动的跟踪，无需初始框；4) 去噪训练策略剔除伪标注噪声。

**🔧 技术方法**

技术方法：基于 Transformer 的多模态编码器，利用大型视觉‑语言模型（如 APE 或 LISA）生成伪边框；使用注意力机制实现 DTA；采用弱强数据增强、欧式距离去噪；损失函数包含 Focal、GIoU 与 L1。

**📊 数据集**

使用的数据集：LaSOT、TNL2K、LaSOT_ext、OTB99 的公开 VL 跟踪数据集，以及大规模未标注视频用于自监督训练。

**📈 对比分析**

与多种基准方法对比，尤其是全监督与无监督跟踪器；在自监督设置下，-L256/384 模型在 TNL2K、LaSOT、OTB99 上分别比 ATTracker 提升 1.9%、3.6%、9.9% 的 AUC；在 LaSOT_ext 上亦接近全监督性能；在视觉跟踪任务中，在无框架下击败 Diff‑Tracker、SSTrack 等最先进无监督方法。

**⚠️ 局限性**

局限性：伪边框的质量受大型视觉‑语言模型的推理能力限制，若 LVLM 推断不准确会影响跟踪结果；未来需改进跨模态对齐与伪标签生成的鲁棒性。

---

## 367. From Canopy to Collision: A Hybrid Predictive Framework for Identifying Risk Factors in Tree-Involved Traffic Crashes

**arXiv ID:** 2605.06684 | [PDF](https://arxiv.org/pdf/2605.06684v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 368. Signal Reshaping for GRPO in Weak-Feedback Agentic Code Repair

**arXiv ID:** 2605.07276 | [PDF](https://arxiv.org/pdf/2605.07276v1)

**作者:** Jia Li `[一作]` (Chinese University of Hong Kong), Michael R. Lyu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 41992 | [OpenAlex ID](https://openalex.org/A5069596903)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在弱反馈的代码修复任务中，对GRPO训练信号进行重塑，使其在编译‑语义层次、过程信用和同组可比性上更具意义。

**💡 创新点**

提出三层信号重塑方法：1）分层编译与语义奖励恢复轨迹排序；2）步级过程得分细化轨迹内信用；3）失效原因路由治理保持同组可比性，同时保持GRPO原有结构不变。

**🔧 技术方法**

使用分层奖励、步骤权重、资源解耦的rollout治理与GRPO优化；过程得分由信息增益判别器给出；对比Token‑级KL蒸馏等方法。

**📊 数据集**

基于内部构建的compile‑fix数据集（包含编译错误日志、最小足够修复、环境信息等），共约1110个训练样本与400个评估样本。

**📈 对比分析**

与基础零射击、仅编译奖励、仅层化奖励等对照，实验显示全重塑的GRPO在严格编译‑语义准确率上从0.385提升至0.535；加入步骤权重后进一步提升至0.53，并将平均评估步数从23.5降至17.0。

**⚠️ 局限性**

局限：实验仅针对compile‑fix任务，未验证在其他弱反馈代理任务上的推广；Token‑级分布匹配方法未能替代步骤信用；数据集为内部构建，外部复现困难。

---

## 369. Experience Sharing in Mutual Reinforcement Learning for Heterogeneous Language Models

**arXiv ID:** 2605.07244 | [PDF](https://arxiv.org/pdf/2605.07244v1)

**作者:** Xiaoze Liu `[一作]` (Purdue University), Stefano Soatto `[通讯]` (AWS Agentic AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究提出了 Mutual Reinforcement Learning 框架，使不同 LLM 策略在保持各自参数、目标和 tokenizer 的前提下，能够在训练后并行共享结构化经验。

**💡 创新点**

主要创新在于 (1) 设计了跨 tokenizer 的经验交换层 THL，允许异构模型互转 token 级轨迹； (2) 构建了共享经验、资源分配和 tokenizer 兼容三大组件； (3) 在 GRPO 基础上提出了三种共享级别探针（PRP、XGRPO、SGT），并通过上下文 bandit 理论分析其稳定-支持权衡。

**🔧 技术方法**

采用 GRPO、VERL 训练栈、tokenizer 复原与对齐、重要性比率修正、分组归一化、奖励验证与 gated outcome transfer 等技术。

**📊 数据集**

在数学推理任务（如 MATH、GSM8K 等）以及常识/科学 QA 基准（CommonsenseQA、PubMedQA 等）上进行评估，统一答案提取模板。

**📈 对比分析**

通过在相同计算资源、超参数与确定性解码条件下比较三种共享探针，实验和理论均表明 outcome‑level（SGT）在稳定性与性能上最优，显著提升推理成功率。

**⚠️ 局限性**

限制包括：需要额外的跨 tokenizer 对齐开销；仅在后训练 on‑policy 环境下验证；对极端稀疏奖励场景或大规模多模型系统的扩展性尚待探索；对奖励验证机制的依赖可能限制在无验证器的任务中。

---

## 370. Task Relevance Is Not Local Replaceability: A Two-Axis View of Channel Information

**arXiv ID:** 2605.07086 | [PDF](https://arxiv.org/pdf/2605.07086v1)

**作者:** Houman Safaai `[一作]` (Harvard University), Bernardo L. Sabatini `[通讯]` (Harvard Medical School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了将通道重要性拆分为目标相关性和局部可替换性两个维度，并用这两个维度评估通道可去除性

**💡 创新点**

首次系统性地证明局部可替换性比目标相关性更能预测通道移除后的性能损失，并揭示两维度在训练过程中解耦

**🔧 技术方法**

使用基于高斯近似的互信息指标（I_X、I(T;Y)、peer‑overlap 等），并结合非参数 PID、BROJA、KSG 检验；通过结构分析、梯度几何、单通道消融和 FLOPs‑匹配剪枝实验验证

**📊 数据集**

主要在 CIFAR‑100 上使用 ResNet‑18、VGG‑16、MobileNetV2 训练模型，作为压力测试还使用 CIFAR‑10、Tiny‑ImageNet、ImageNet‑100、ConvNeXt‑T/ImageNet‑100、ViT 等数据集

**📈 对比分析**

与传统目标相关性评分（如 I(T;Y)、目标冗余）以及基于范数的基线（如权重范数、Taylor 重要性）对比，局部轴方法在 ResNet‑18、VGG‑16、MobileNetV2 的 FLOPs‑匹配 AUC 方面分别提升 8.0pp、6.6pp、10.1pp，VGG‑16 仍以范数为最优；在 ConvNeXt‑T 的实验中局部轴同样优于目标轴

**⚠️ 局限性**

局限主要在于：1）实验集中在 CIFAR‑100，其他数据集仅为压力测试；2）互信息代理为高斯近似，缺乏对非线性/确定性通道的精确度量；3）局部可替换性主要考虑同层通道，未覆盖跨层补偿路径；4）不同架构下指标的解释和权重分配仍需进一步研究

---

## 371. Self-Consolidating Language Models: Continual Knowledge Incorporation from Context

**arXiv ID:** 2605.07076 | [PDF](https://arxiv.org/pdf/2605.07076v1)

**作者:** Zekun Wang `[一作]` (Georgia Institute of Technology), Christopher J. MacLellan `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 393 | [OpenAlex ID](https://openalex.org/A5077641166)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究将长文本上下文通过持续自我更新写入LLM权重的持续上下文整合方法。

**💡 创新点**

提出SCoL框架，让LLM生成文本指令决定哪些Transformer层需要更新，并使用元强化学习优化保持已整合信息。

**🔧 技术方法**

使用元强化学习、LoRA适配器、文本生成的更新指令、Fisher信息对齐等技术。

**📊 数据集**

在SQuAD知识整合任务和LongBench v2长文本推理任务上进行实验。

**📈 对比分析**

与提示、摘要、批量测试时训练、顺序微调基线比较，SCoL在获取与保持知识方面显著优于基线，在长文本场景中取得最高分。

**⚠️ 局限性**

计算开销大、候选更新评估耗时、可能出现模式坍塌导致更新多样性不足。

---

## 372. Agentic AI and the Industrialization of Cyber Offense: Forecast, Consequences, and Defensive Priorities for Enterprises and the Mittelstand

**arXiv ID:** 2605.06713 | [PDF](https://arxiv.org/pdf/2605.06713v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 373. Bias and Uncertainty in LLM-as-a-Judge Estimation

**arXiv ID:** 2605.06939 | [PDF](https://arxiv.org/pdf/2605.06939v1)

**作者:** James Fiedler `[一作]` `[通讯]` (Indeed Incorporated), James Fiedler (Indeed Incorporated)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 LLM-as-a-Judge 评估中偏差校正估计的可靠性，分析了共享校准对单模型和模型比较的影响，并给出诊断指标。

**💡 创新点**

提出 J 与 ΔJ 作为判断校准可靠性的诊断，揭示共享校准在低 J 或 ΔJ 下会导致偏差放大和符号逆转，并通过理论分析、仿真和 MMLU-Pro 案例验证。

**🔧 技术方法**

采用 Rogan-Gladen 估计、PPI（概率加权）及其 EIF 对应、bootstrap 置信区间、理论推导、仿真和实际数据案例。

**📊 数据集**

使用 MMLU-Pro benchmark 的 Math 与 Biology 两个科目作为实证案例。

**📈 对比分析**

与未校正的原始判决、RG 共享校准、RG 单模型校准、PPI 及其双预算版本进行比较；在高 J 时性能相近，低 J 或 ΔJ 时共享校准表现差，PPI 更稳健但在极低 J 时仍有覆盖问题。

**⚠️ 局限性**

仅适用于二元输出、有限样本、未覆盖输入依赖或自适应收集等情形；诊断只能提示风险，无法在极低 J 时完全恢复可靠性。

---

## 374. Optimal Learning-Augmented Algorithm for Online Bidding

**arXiv ID:** 2605.07349 | [PDF](https://arxiv.org/pdf/2605.07349v1)

**作者:** Changyeol Lee `[一作]` (Yonsei University), Changki Yun `[通讯]` (Seoul National University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文提出了一种基于“竞价配置文件”（bidding profile）的随机学习增强算法，完整刻画了在线竞价问题的鲁棒-一致性（robustness–consistency）Pareto最优权衡，并将该方法推广到线性搜索问题。

**💡 创新点**

创新点在于：① 引入竞价配置文件概念，将随机策略转化为对数变换后的分布；② 通过求解递延微分方程构造出满足任意鲁棒-一致性参数的最优配置文件；③ 证明任意随机策略可归约为配置文件驱动策略，从而完成最优性证明；④ 将同一框架应用于线性搜索，得到接近最优的搜索策略。

**🔧 技术方法**

核心技术包括：随机化策略的函数表示（$G(x)$ 形式）、对数积分分析、递延微分方程求解、单调性与收敛性证明、以及配置文件与策略之间的映射。

**📊 数据集**

论文为理论研究，没有使用公开数据集，所有结果均为数学证明和理论上限/下限。

**📈 对比分析**

与之前的上界与下界相比，本文在随机化在线竞价中消除了鲁棒-一致性权衡的剩余空隙；在线性搜索中提供了新的近似比，接近已知下界，表现出显著改进。

**⚠️ 局限性**

局限性包括：① 对线性搜索的完整Pareto最优性仍未证实，仍存在上界与下界的微小差距；② 对其他可能需要平滑性的学习增强问题未展开研究；③ 目前方法主要针对理论分析，缺乏对实际机器学习预测误差分布的实验验证。

---

## 375. ModelLens: Finding the Best for Your Task from Myriads of Models

**arXiv ID:** 2605.07075 | [PDF](https://arxiv.org/pdf/2605.07075v1)

**作者:** Rui Cai `[一作]` (University of California, Davis), Zhe Zhao `[通讯]` (University of California, Davis)

**通讯引用:** 3319 | [OpenAlex ID](https://openalex.org/A5100631150)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个大规模模型-数据集-指标交互数据集，提出了ModelLens框架，用无前向推理的方式在开放模型生态中进行模型推荐。

**💡 创新点**

创新点在于将公共leaderboard交互视作隐式能力地图，利用结构化先验与残差交互的分解方式，实现对未见模型和数据集的零样本推荐，并通过ID Dropout实现冷启动泛化。

**🔧 技术方法**

采用多视角特征表示（ID、名称、描述嵌入），任务与指标嵌入，结构化先验（模型规模与族群），残差交互网络，联合使用列表、成对与点估计三种损失，训练时使用ID Dropout，输出兼顾全局与细粒度的评分。

**📊 数据集**

数据集包含约1.62M条评测记录，涵盖47K个模型、9.6K个数据集、2,551个任务和348个模型族群，来源于HuggingFace、Open LLM Leaderboard和PapersWithCode。

**📈 对比分析**

与基于特征的转移性估计（如LogME、PACTran）以及基于元学习或指标相似度的无特征方法（如Task2Vec、ZAP、模型大小/流行度等）对比；在性能补全、冷启动和QA路由等实验中，ModelLens在Kendall's τ_w、Hit@K、NDCG等指标上明显优于所有基线，并在路由任务中提升了约50-80%的精度。

**⚠️ 局限性**

局限性包括：对极端稀缺或新兴指标/任务的泛化尚不充分；在需要精确数值预测时对点估计的依赖较弱；对模型的后训练或微调效果尚未考虑；以及在超大规模模型族群中模型结构先验可能需要进一步细化。

---

## 376. MIPIAD: Multilingual Indirect Prompt Injection Attack Defense with Qwen -- TF-IDF Hybrid and Meta-Ensemble Learning

**arXiv ID:** 2605.07269 | [PDF](https://arxiv.org/pdf/2605.07269v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 377. EPTAS for Hard Graph Cut Problems for Dense Graphs

**arXiv ID:** 2605.07265 | [PDF](https://arxiv.org/pdf/2605.07265v1)

**作者:** Kaisei Deguchi `[一作]` (University of Tokyo), Hiroaki Mori `[通讯]` (University of Tokyo)

**通讯引用:** 1547 | [OpenAlex ID](https://openalex.org/A5104051314)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在稠密图上针对多种 NP‐hard 的图割最小化问题提出了高效多项式时间逼近方案（EPTAS），包括 ConstrainedMinCut、MinQuotientCut、ProductSparsestCut 以及其特殊情形 UniformSparsestCut、EdgeExpansion、Conductance、NormalizedCut、BalancedSeparator、SmallSetExpansion。

**💡 创新点**

创新点在于：① 仅使用弱正则性引理、采样与估计以及精细的平衡重构（rebalancing）技术，避免了之前依赖 Lasserre 层次或整数规划的昂贵工具；② 通过统一的归约实现多种问题的 EPTAS；③ 通过对稠密图结构的充分利用，将传统 PTAS 的时间复杂度 n^f(1/ε) 提升到 f(1/ε)·n^O(1)+2^{O(1/ε^c)}，实现真正的 EPTAS。

**🔧 技术方法**

主要技术包括弱正则性引理（Frieze–Kannan 版）、基于采样的邻接估计、整数规划的线性松弛与离散化、动态规划求解 knapsack 结构、以及对成本约束的精细重新平衡；在实现上也用到了确定性弱正则性分解以及离散化的 DP 复杂度分析。

**📊 数据集**

本工作属于理论算法研究，不依赖具体实验数据集；所有结果均在抽象稠密图模型（everywhere‑δ‑dense graphs）下证明。

**📈 对比分析**

与之前的 PTAS（时间为 n^{O(1/ε)}）相比，提出的 EPTAS 取得了运行时间 f(1/ε)·n^{O(1)}+2^{O(1/ε^c)}，显著降低了对 ε 的指数依赖；在所有考虑的问题上均实现了相同的 1+ε 近似比，且在理论上提供了更优的时间复杂度。

**⚠️ 局限性**

局限性包括：① 仅适用于 everywhere‑δ‑dense 图，无法直接推广到一般稀疏图；② 对于成本约束问题，必须假设每个顶点权重至少是常数倍 n；③ 仍需利用弱正则性分解导致的 2^{O(1/ε^c)} 成本，虽然比 n^{O(1/ε)} 好，但在实际大规模图上可能仍不够高效；④ 对于更复杂的图割形式或约束，归约策略尚未验证。

---

## 378. Structural Rationale Distillation via Reasoning Space Compression

**arXiv ID:** 2605.07139 | [PDF](https://arxiv.org/pdf/2605.07139v1)

**作者:** Jialin Yang `[一作]` (University of Calgary), Steve Drew `[通讯]` (University of Calgary)

**通讯引用:** 2963 | [OpenAlex ID](https://openalex.org/A5016341803)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 D‑RPC 通过构建可压缩的推理路径库并在蒸馏时引导教师沿这些路径生成一致性较高的推理链，从而提升小模型的推理性能。

**💡 创新点**

在蒸馏前对教师推理空间进行动态压缩与检索，并用通用推理路径作为条件来约束教师生成，既保持多样性又显著降低监督噪声。

**🔧 技术方法**

利用意图分类、DBSCAN 聚类、检索与路径条件生成、LoRA 参数高效微调，以及 PAC‑Bayes 理论分析来实现路径压缩与蒸馏。

**📊 数据集**

在 GSM8K、AQUA、StrategyQA、AI2ARC、MATH 等数学与常识推理基准上进行实验。

**📈 对比分析**

与 CoT、Freeform、DCoT、SGFT、SuperCorrect 等基线在 Llama 3.1 8B 与 Qwen 3 1.7B 两款学生模型上对比，D‑RPC 在所有五个基准上均取得最高准确率，尤其在难度较高的 MATH 与 AQUA 上提升明显。

**⚠️ 局限性**

实验仅覆盖单一教师（GPT‑5.1）和两种学生模型，无法验证在其他教师、不同规模模型或非数学领域的迁移性；教师查询成本较高；理论分析仅适用于 LoRA 微调，尚未推广到全微调或其他参数高效方法。

---

## 379. Hallucination Detection via Activations of Open-Weight Proxy Analyzers

**arXiv ID:** 2605.07209 | [PDF](https://arxiv.org/pdf/2605.07209v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 380. Same Brain, Different Prediction: How Preprocessing Choices Undermine EEG Decoding Reliability

**arXiv ID:** 2605.07212 | [PDF](https://arxiv.org/pdf/2605.07212v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 381. Conditional generation of antibody sequences with classifier-guided germline-absorbing discrete diffusion

**arXiv ID:** 2605.06720 | [PDF](https://arxiv.org/pdf/2605.06720v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 382. TransDot: An Area-efficient Reconfigurable Floating-Point Unit for Trans-Precision Dot-Product Accumulation for FPGA AI Engines

**arXiv ID:** 2605.07245 | [PDF](https://arxiv.org/pdf/2605.07245v1)

**作者:** Jiayi Wang `[一作]` (University of Washington), Ang Li `[通讯]` (University of Washington)

**通讯引用:** 9408 | [OpenAlex ID](https://openalex.org/A5100413631)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

设计了一种可重构的浮点单元（TransDot），实现了多精度 SIMD FMA 与跨精度点积累（DPA）在共享数据路径上的统一支持；

**💡 创新点**

创新点在于通过重构桶式移位器、分段乘法器以及专门的 FP4 DP2 单元，将原本独立的 DPA 单元与 FMA 单元进行硬件共享，从而在支持 FP4、FP8、FP16 等新兴低精度格式的同时，显著提升面积效率并降低功耗；

**🔧 技术方法**

采用可重构桶式移位器、分段多模式数组乘法器、FP4 2-term DP2 逻辑，并在 RTL 级别实现可配置的数据路径与流水线；

**📊 数据集**

本研究基于 12 nm ASIC 合成与 Place‑and‑Route 测试，未使用特定机器学习数据集；

**📈 对比分析**

与 FPnew 基线对比，TransDot 在 FP16 DPA 和 FP8 DPA 上分别实现了 1.46× 与 2.92× 的面积效率提升，点积累吞吐量分别为 FP16 2×、FP8 4×、FP4 8×，但整体面积平均增加 37.3%，并额外引入一条流水线阶段；

**⚠️ 局限性**

局限性包括面积和延迟的折中（面积增加 37.3%）、额外流水线导致的时延提升，以及仅针对 DPA 设计的模块在某些特定工作负载下可能无法充分利用；

---

## 383. Coupling Models for One-Step Discrete Generation

**arXiv ID:** 2605.07193 | [PDF](https://arxiv.org/pdf/2605.07193v1)

**作者:** Fred Zhangzhi Peng `[一作]` (Duke University), Alexander Tong `[通讯]` (AITHYRA)

**通讯引用:** 1296 | [OpenAlex ID](https://openalex.org/A5053568123)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种一阶离散生成模型 Coupling-Models，先通过双阶段训练将离散序列与高斯潜在变量耦合，再训练并行解码器实现一次性生成；

**💡 创新点**

创新点在于不依赖轨迹压缩，而是学习可采样的高斯耦合并将全局依赖压缩到潜在变量中，从而实现单步并行解码，并可进一步支持少步精炼与多种引导方式；

**🔧 技术方法**

使用技术包括：Stage‑A 的编码器 + 正则化 + 可逆流（Normalizing Flow）将序列映射到标准正态分布；Stage‑B 的并行Transformer/解码器；以及分类器无监督、分类器指导、奖励微调等一阶指导方法；

**📊 数据集**

实验数据集涵盖 MNIST‑Binary（二值图像）、Fly Brain DNA 增强子序列（长度500）和 LM1B 语言数据（约1M英文句子）；

**📈 对比分析**

与现有最强一阶基线比较：MNIST‑Binary FID 从 204.6 降至 5.50；DNA 增强子 FBD 从 15.8 降至 12.9；LM1B 生成困惑度从 119.34 降至 45.82，且保持较高熵；在少步精炼和引导任务中也实现了更优或更高效的性能；

**⚠️ 局限性**

局限性包括模型规模相对有限，未在大规模语言、长序列或蛋白设计等更具挑战性的任务上验证；评估指标（如生成困惑度）与最终样本质量相关性不一定高。

---

## 384. Transformer-Based Wildlife Species Classification from Daily Movement Trajectories

**arXiv ID:** 2605.06726 | [PDF](https://arxiv.org/pdf/2605.06726v1)

**作者:** Obed Irakoze `[一作]` (Carnegie Mellon University Africa), Prasenjit Mitra `[通讯]` (Carnegie Mellon University Africa)

**通讯引用:** 10247 | [OpenAlex ID](https://openalex.org/A5009542542)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

利用Transformer模型对跨区域GPS轨迹进行每日运动序列分类，以识别野生动物种类。

**💡 创新点**

提出在跨研究holdout设置下评估并证明Transformer在运动轨迹分类中的优越性，并引入多维运动特征和时间分辨率分析。

**🔧 技术方法**

Transformer编码器、LSTM、CNN、TCN等序列模型，结合自注意力、位置编码、掩码处理和时间特征编码。

**📊 数据集**

Movebank平台公开的七种非洲野生动物GPS轨迹（狒狒、水牛、豹猫、斑马、大象、狮子、羚羊）。

**📈 对比分析**

在严格的研究级别留出测试集下，Transformer在平衡准确率、F1和AUC上均优于基线，平均提升约8-22个百分点。

**⚠️ 局限性**

受限于数据稀疏、不同采样频率导致缺失和时间分辨率不一致，以及仅做二分类、仅在7种物种和区域上验证。

---

## 385. Evaluating Prompt Injection Defenses for Educational LLM Tutors: Security-Usability-Latency Trade-offs

**arXiv ID:** 2605.06669 | [PDF](https://arxiv.org/pdf/2605.06669v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 386. Rethinking Experience Utilization in Self-Evolving Language Model Agents

**arXiv ID:** 2605.07164 | [PDF](https://arxiv.org/pdf/2605.07164v1)

**作者:** Weixiang Zhao `[一作]` (Harbin Institute of Technology), Ting Liu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 40543 | [OpenAlex ID](https://openalex.org/A5100418162)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了 ExpWeaver，一个在自进化代理决策过程中动态触发经验检索与使用的轻量级机制，证明其在多框架、多模型、多环境下均能提升性能。

**💡 创新点**

将经验利用从固定注入模式改为推理时可选触发，通过触发词实现按需检索，提升了代理在不确定决策点的经验利用效率。

**🔧 技术方法**

采用 ReAct‑style 推理框架、提示工程、触发词激活经验检索、Token 级熵分析以及 GRPO 强化学习以学习经验使用策略。

**📊 数据集**

在 8 个基准上评估，包括 ALFWorld（具象交互）、WebShop（网页导航）以及 HotpotQA、NQ、TriviaQA、2Wiki、MuSiQue、Bamboogle（知识密集型 QA）。

**📈 对比分析**

与无经验、初始化仅、始终开启三种传统经验利用策略对比；在 7 种 LLM（GPT‑5.2、DeepSeek‑V4‑Pro、Kimi‑K2.5、Qwen3‑4B/14B/32B/397B‑A17B 等）与三种环境中，ExpWeaver 一贯取得最高成功率；RL 训练中亦表现最优；因果消融与熵分析进一步证明其效果。

**⚠️ 局限性**

实验仅聚焦于 LLM 驱动的自进化代理，依赖提示和触发词；对非 LLM 代理、不同经验表示或多任务场景的通用性尚未充分验证；计算资源有限导致实验规模受限，缺乏对长时序经验检索与跨任务泛化的深入探讨。

---

## 387. Distributional Process Reward Models: Calibrated Prediction of Future Rewards via Conditional Optimal Transport

**arXiv ID:** 2605.06785 | [PDF](https://arxiv.org/pdf/2605.06785v1)

**作者:** Rachel Ma `[一作]` (Massachusetts Institute of Technology), Kristjan Greenewald `[通讯]` (IBM)

**通讯引用:** 1330 | [OpenAlex ID](https://openalex.org/A5002525343)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于条件最优传输的 Process Reward Model（PRM）校准方法。

**💡 创新点**

首次将条件最优传输用于学习完整的单调条件分位数函数，实现任意置信水平下无交叉、结构保证的校准。

**🔧 技术方法**

使用条件最优传输（CondOT）+ PICNN 结构、分位数回归对比，以及实例自适应推理时缩放（IAS）框架。

**📊 数据集**

在数学推理基准 MATH‑500 和 AIME24-25 等数据集上进行训练和评估。

**📈 对比分析**

与未校准 PRM 和传统分位数回归相比，Brier、ECE 和 WQL 等指标显著下降，Best‑of‑N 的准确率提升，尤其在 OOD AIME 上效果更明显。

**⚠️ 局限性**

校准质量受原始 PRM 排名质量、数据分布偏移以及有限 roll‑out 估计噪声影响，且仅为离线校准层，未对 PRM 本身进行微调。

---

## 388. SoLAR: Error-Resilient Streamable Long-Horizon Free-Viewpoint Video Reconstruction with Anchor Activation and Latent Recalibration

**arXiv ID:** 2605.07346 | [PDF](https://arxiv.org/pdf/2605.07346v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 389. Bringing Multimodal Large Language Models to Infrared-Visible Image Fusion Quality Assessment

**arXiv ID:** 2605.06969 | [PDF](https://arxiv.org/pdf/2605.06969v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 390. How to Compress KV Cache in RL Post-Training? Shadow Mask Distillation for Memory-Efficient Alignment

**arXiv ID:** 2605.06850 | [PDF](https://arxiv.org/pdf/2605.06850v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 391. AI and Consciousness: Shifting Focus Towards Tractable Questions

**arXiv ID:** 2605.06965 | [PDF](https://arxiv.org/pdf/2605.06965v1)

**作者:** Iulia-Maria Comsa `[一作]` (Google DeepMind), Iulia-Maria Comsa `[通讯]` (Google DeepMind)

**通讯引用:** 403 | [OpenAlex ID](https://openalex.org/A5032013007)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过文献综述和概念分析，探讨了人工智能（AI）意识研究的可行性，认为“AI是否能具有主观经验”的核心问题目前在科学上不可解；提出将研究重点转向“人类对AI意识的感知”这一可操作、影响深远的可追踪议题，并给出相关心理学驱动特征列表和实验框架建议。

**💡 创新点**

创新点在于：①引入可追踪性（tractability）作为评估AI意识研究方向的标准；②将“感知AI意识”定位为可实验的社会心理学问题，并提出八种关键行为特征（自我反思、情感、逻辑推理等）；③提出针对AI回答“你是意识吗？”的透明、科学立场模板，强调避免主观一人称的潜在矛盾。

**🔧 技术方法**

主要使用的方法是理论与概念性分析、系统性文献综述以及对现有实验特征的合成整理；未采用具体机器学习算法或编程实现。

**📊 数据集**

未使用任何公开数据集；论述基于已有的调查研究和实验结果（如对GPT-3、机器人吸尘器等的用户感知调查）。

**📈 对比分析**

本文未进行实验或性能评估；比较方法为对比现有AI意识评估方法（如神经标记、概率聚合）与其局限性，指出它们在核心问题上的不可解性。

**⚠️ 局限性**

局限性：①缺乏经验验证的实验数据，提出的特征列表和实验框架仍需实证检验；②理论层面高度依赖未统一的意识理论，可能导致解释不一致；③对“感知AI意识”所提议的社会影响预测主要基于推测，缺乏量化证据。

---

## 392. When Stored Evidence Stops Being Usable: Scale-Conditioned Evaluation of Agent Memory

**arXiv ID:** 2605.07313 | [PDF](https://arxiv.org/pdf/2605.07313v1)

**作者:** Jiaqi Shao `[一作]` (Hong Kong University of Science and Technology), Bing Luo `[通讯]` (Duke Kunshan University)

**通讯引用:** 1013 | [OpenAlex ID](https://openalex.org/A5010645658)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于可证据保持的规模化评估协议，用于测量智能体在不可用记忆增大时的可用性；

**💡 创新点**

创新点在于将记忆规模扩展与检索预算、交互负载、错误来源分解和可用尺度阈值结合成轨迹层面的诊断指标；

**🔧 技术方法**

使用大型语言模型（Qwen3 系列）与多种记忆接口（平面、层级、平面+层级），并记录智能体的检索调用轨迹；

**📊 数据集**

实验基于 LongMemEval 与 LoCoMo 两个基准，采用共享的规模阶梯在不同级别加入无关会话；

**📈 对比分析**

通过 Pass@B、P90R、p_exh/p_wrong 以及 s*_α 等指标与多模型、多接口进行比较，发现如 HippoRAG 在 2 次调用预算内可靠性下降 16–20%，而 Qwen3-32B 与 Qwen3-235B 在同一规模范围内保持稳定；

**⚠️ 局限性**

局限在于仅评估检索调用预算和可用性，未涵盖存储成本、离线索引开销、内部图结构复杂度或对抗鲁棒性等方面。

---

## 393. Rollback-Free Stable Brick Structures Generation

**arXiv ID:** 2605.06947 | [PDF](https://arxiv.org/pdf/2605.06947v1)

**作者:** Chenhui Xu `[一作]` (University of Buffalo), Jinjun Xiong `[通讯]` (University of Buffalo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研发了一种无回滚的砖块结构生成框架STABLE，利用强化学习从物理奖励中学习，直接在一次推理中生成符合物理约束的稳定砖块装配。

**💡 创新点**

将物理可行性从推理时的外部校正转为训练时的结构层奖励，使模型内部化重力、碰撞、连通性和相互锁定等物理先验，实现首个无回滚、无需物理仿真的稳定砖块生成。

**🔧 技术方法**

采用基于Transformer的指令跟随语言模型，先做监督式形状重建（SFT），随后使用Group Relative Policy Optimization（GRPO）强化学习，结合碰撞、形状、一致性、连通性等四项结构奖励。

**📊 数据集**

使用PointCloud2Brick数据集，基于StableText2Brick，包含42,604训练样本和4,785测试样本，每个样本由点云条件和对应的砖块序列组成。

**📈 对比分析**

与通用LLM、BrickGPT、LegoACE等基线比较，STABLE在不使用回滚的推理下取得碰撞率0.99、连通率0.97、互锁得分0.722、形状IoU0.907，并将推理速度从BrickGPT的895秒降低到58秒，速度提升约94%。

**⚠️ 局限性**

在极其复杂或高度自由形状的目标下仍可能产生不完整或低效的砖块分解；奖励设计对不同构造策略的平衡仍有限；训练需要大规模GPU资源；对非标准砖块库的适配仍需进一步研究。

---

## 394. GraphDC: A Divide-and-Conquer Multi-Agent System for Scalable Graph Algorithm Reasoning

**arXiv ID:** 2605.06671 | [PDF](https://arxiv.org/pdf/2605.06671v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 395. Scalable Active Metamaterials for Shape-Morphing

**arXiv ID:** 2605.07030 | [PDF](https://arxiv.org/pdf/2605.07030v1)

**作者:** Jipeng Cui `[一作]` (Texas A&M University), Wei "Wayne" Chen `[通讯]` (Texas A&M University)

**通讯引用:** 256566 | [OpenAlex ID](https://openalex.org/A5100352881)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了可扩展的主动形变超材料（SAM）设计框架，通过将宏观与微观尺度分离，实现从宏观目标形变到微观单元几何的高效逆设计。

**💡 创新点**

创新点在于：①采用刚柔分离的单元格划分，使刚性骨架与软填料解耦；②宏观层引入数据驱动的约束下拉氏网格编辑（ConLME）；③微观层使用条件扩散模型（cDM）或可调搜索（AS）实现单元几何的逆设计；④整个框架在单个单元数据库上可复用，线性扩展到数千个单元。

**🔧 技术方法**

技术包括：拉氏网格编辑、数据驱动软约束、条件扩散模型（基于U‑Net的DDPM）、可调搜索算法、有限元分析、热力学线性弹性模型。

**📊 数据集**

使用了约60,000个热活化单元格的几何与变形响应数据库（包含八根弯曲梁的尺寸与材料参数），该数据库既用于宏观约束也用于微观逆设计。

**📈 对比分析**

与传统拓扑优化（TO）及其他全局优化方法比较，SAM 的计算成本随单元数线性增长（cDM 方案）且显著低于 TO；在多种形变任务（章鱼、机翼、抓手、棋盘梁）中，MAE/ MRE 均保持在 4–23% 以内，R²_macro 0.95–0.99，表现优于现有的周期性、层次化或集成设计方法。

**⚠️ 局限性**

局限性包括：①宏观拉氏网格编辑在极大形变时易产生局部畸变；②需要手动调节软约束权重和节点位置；③使用规则网格限制边界光滑度；④单元数据库覆盖范围受限，难以实现更大幅度或非热激活的形变；⑤对非常大规模或高精度问题仍需进一步加速。

---

## 396. Continuous First, Discrete Later: VQ-VAEs Without Dimensional Collapse

**arXiv ID:** 2605.06870 | [PDF](https://arxiv.org/pdf/2605.06870v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 397. Topic Is Not Agenda: A Citation-Community Audit of Text Embeddings

**arXiv ID:** 2605.07158 | [PDF](https://arxiv.org/pdf/2605.07158v1)

**作者:** Junseon Yoo `[一作]` `[通讯]` (Pluto Labs), Junseon Yoo (Pluto Labs)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了两层引用社区划分（子领域 L1 与研究议题 L2），对八个学科、1.6M 论文使用四种顶尖文本嵌入模型进行大规模邻居一致率测量，并提出简单的引用计数重排序作为诊断与提升手段。

**💡 创新点**

①首次在科学文献中系统评估文本嵌入在细粒度研究议题层面的失效；②使用增强引用图结合 Leiden CPM 实现层次化社区划分；③发现 L1 与 L2 之间约 30% 的差距并持久；④用引用计数重排序展示检索性能显著提升。

**🔧 技术方法**

Leiden CPM 社区检测、增强引用图（直引、书目耦合、共引）、Gemini/Qwen3/SPECTER2 文本嵌入、FAISS 近邻搜索、BM25 以及 LLM 扩展布尔检索、引用计数重排序与 RRF。

**📊 数据集**

约 3.58M 论文的增强引用图（OpenAlex、PubMed、Semantic Scholar 等来源），8 个科学领域（生物学、生物医学、化学、计算机科学、工程、地球/环境科学、材料科学、物理学）的 1.6M 标准化样本，80 个手工挑选的研究议题查询。

**📈 对比分析**

与 BM25、四个稠密检索器（Gemini、Qwen3‑8B、Qwen3‑0.6B、SPECTER2）及两种重排序器（Graph、BM25+cite）比较。单向检索在 L1 上 top‑10 同一子领域一致率 45–52%；在 L2 上仅 15–21%。重排序后 top‑1 L2 一致率提升至约 58–60%，比 Gemini 最佳的 50.6% 提升 7–10 个百分点。

**⚠️ 局限性**

①引用社区仅为研究议题的代理，可能导致过度计数；②Graph retriever 仅使用 Gemini LLM，需验证不同 LLM 或无 LLM 配置；③查询挑选偏向规模大议题，可能影响基准代表性；④仅评估科学文献，需验证在法律、专利、临床指南等其他参考图上的泛化；⑤重排序为简单诊断，未提供学习型重排序方案。

---

## 398. Regret-Oracle Complexity Tradeoffs in Agnostic Online Learning

**arXiv ID:** 2605.07155 | [PDF](https://arxiv.org/pdf/2605.07155v1)

**作者:** Idan Attias `[一作]` (Institute for Data, Econometrics, Algorithms, and Learning), Arvind Ramaswami `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

提出一种基于弱一致性（weak-consistency）oracle的自适应动态裁剪算法，用于在无先验可实现性假设的情况下实现在线学习。

**💡 创新点**

创新点在于：①用弱一致性oracle代替传统的求最小化风险（ERM）oracle；②通过动态裁剪（realizability pruning + mistake pruning）将原本指数级的专家树压缩到 VC 维度的多项式大小；③系统化地把“专家树”与 Hedge 权重结合，保证期望损失不变，达到与原Reduction相同的近最优期望后悔。

**🔧 技术方法**

核心技术包括：
- 通过 Sauer 定理约束可实现的标签前缀数量；
- 设计基于“未来容量”W_t(u)的加权策略，等价于在全树上运行 Hedge；
- 对弱一致性oracle进行多路查询，动态维护活跃路径集合；
- 采用可调学习率实现第一阶（instance‑dependent）后悔上界；
- 通过潜能函数证明裁剪不影响后悔，并给出查询量-后悔的连续折衷。

**📊 数据集**

无实验数据集，论文主要为理论分析与下界证明。

**📈 对比分析**

与传统的SOA+Reduction相比：
- 查询复杂度从 𝒪(T^{2^{𝒪(d)}}) 减至 𝒪(T^{d+1})（d 为 VC 维度）；
- 记忆量从 𝒪(T^d) 减少到 𝒪(t^d)；
- 期望后悔保持在 𝒪̃(√(T·2^{𝒪(d)}))，与最优值相当。

**⚠️ 局限性**

局限性包括：
- 仍需依赖弱一致性oracle，实际实现难度未知；
- 对于低查询预算仍有较大后悔；
- 上界与下界之间存在残留 gap，尤其在高维或特定概念类上；
- 只考虑二分类、0‑1 损失，未讨论多分类或其他损失；
- 对于不可识别的概念类（Littlestone 维度无限）不适用。

---

## 399. The University AI Didn't Replace -- Rethinking Universities in the AI Era

**arXiv ID:** 2605.07056 | [PDF](https://arxiv.org/pdf/2605.07056v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 400. State Representation and Termination for Recursive Reasoning Systems

**arXiv ID:** 2605.06690 | [PDF](https://arxiv.org/pdf/2605.06690v1)

**作者:** Debashis Guha `[一作]` (S P Jain School of Global Management), Tarun Kumar `[通讯]` (eClerx Services Ltd)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出递归推理系统的显式状态表示与基于顺序差距的终止判据。

**💡 创新点**

创新点在于将推理状态建模为知识图谱并使用顺序差距检测非可交换性以决定停止。

**🔧 技术方法**

采用知识图谱、欧几里得平滑嵌入、P_e 与 Q 两个算子以及线性化对易子Gramian。

**📊 数据集**

尚未在公开数据集上进行实验，理论框架适用于长文本问答等。

**📈 对比分析**

比较方法未给出实验，仅提出可与固定步长或自信阈值基线竞争。

**⚠️ 局限性**

局限在于仅提供局部非退化性条件，缺乏全局收敛保证与实证验证。

---

## 401. TubeCensus: A Transparent, Replicable, and Large-Scale Census of YouTube Channels and their Subscriber Counts Over Time

**arXiv ID:** 2605.06999 | [PDF](https://arxiv.org/pdf/2605.06999v1)

**作者:** Chloe Eggleston `[一作]` (University of Colorado Boulder), Maria Leonor Pacheco `[通讯]` (University of Colorado Boulder)

**通讯引用:** 266 | [OpenAlex ID](https://openalex.org/A5005560875)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过从互联网档案馆的 Wayback Machine 抓取并解析近两十年的 YouTube 频道页面，构建了包含 1.06 亿个频道及其历史订阅者数的纵向数据集，并提供了易用的 Python 包进行查询。

**💡 创新点**

创新点在于：① 采用归档网页而非官方 API，克服了 API 访问限制、数据不透明和实时性问题；② 通过统一不同历史 URL 格式的频道标识，实现跨时间去重与整合；③ 将复杂的抓取和解析流程包装成可重复、可共享的工具，满足 FAIR 原则。

**🔧 技术方法**

主要技术包括：Python web 抓取、正则匹配多种 URL 前缀、HTML 解析提取订阅数、批量批处理与去重、Pip 包开发与分发，以及利用多线程与分布式存储提升抓取效率。

**📊 数据集**

使用数据集为：<https://archive.org> 的 Wayback Machine 对 YouTube 频道页面的历史快照（约 12 亿条快照，覆盖 2006-2023 年），以及从该快照中抽取的 1.06 亿个唯一频道的订阅者时间序列。

**📈 对比分析**

对比方法：与 Social Blade 的公开顶级频道列表进行重叠与相关性验证；在两段时间（2006-2013、2014-2016）内抽取聚焦样本，计算与 Social Blade 的 Spearman/Pearson 相关系数。结果显示重叠率约 90-96%，相关系数接近 0.98-1.0，表明数据集在覆盖顶级频道方面表现优异。

**⚠️ 局限性**

局限性包括：① 对频道 ID 与频道页面的依赖导致对视频层面无法直接映射；② 采样受 Wayback Machine 抓取策略偏差影响，导致低订阅者频道覆盖不足；③ 仅记录订阅者数，缺乏视频内容、观看量等更细粒度指标；④ 潜在的隐私与滥用风险，需在使用时遵守伦理规范。

---

## 402. bispectrum: Selective $G$-Bispectra Made Practical

**arXiv ID:** 2605.07270 | [PDF](https://arxiv.org/pdf/2605.07270v1)

**作者:** Johan Mathe `[一作]` (Atmo, Inc.), Nina Miolane `[通讯]` (UC Santa Barbara)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个开源的 PyTorch 库，实现了七种群作用下的选择性 G‑bispectrum，提供统一可微的池化层；

**💡 创新点**

创新点在于：① 将选择性 bispectrum 泛化到连续群（SO(2)、SO(3)）并提出 Θ(L²) 的 S² 选择性增广 bispectrum；② 通过选择性减少计算量至 O(|G|) 或 Θ(L²)；③ 证明了该池化层在低数据/中等容量场景下的完整性与数据效率优势；

**🔧 技术方法**

技术手段包括：群表示理论、Clebsch–Gordan 变换、选择性三重系数选择、GPU 加速的快速傅里叶变换以及对数非线性映射；

**📊 数据集**

使用了 PatchCamelyon（C₈）、OrganMNIST3D（八面体群）和 Spherical MNIST（SO(3)）三大基准数据集；

**📈 对比分析**

与 norm pooling、gate pooling、Fourier‑ELU、max pooling 以及基于数据增强的标准 CNN 对比，实验显示 G‑bispectrum 池化在 10% 或 1% 数据下的 AUC/准确率显著优于其它方法，且在所有旋转测试中的 σ_rot 接近零；

**⚠️ 局限性**

局限性包括：缺乏正式的完整性证明、对高频率/大 L 的数值稳定性分析不充分、未提供逆向恢复接口，且在模型容量很高时不一定优于传统最大池化。

---

## 403. 2.5-D Decomposition for LLM-Based Spatial Construction

**arXiv ID:** 2605.07066 | [PDF](https://arxiv.org/pdf/2605.07066v1)

**作者:** Paul Whitten `[一作]` (Rockwell Automation), Sharath Baddam `[通讯]` (Rockwell Automation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一套基于2.5‑D分解的神经符号管线，让LLM只负责水平平面规划，垂直放置由确定性执行器完成，显著提升了三维构建任务的准确率。

**💡 创新点**

创新点在于将空间建模中的可确定维度（垂直坐标）从LLM输出空间剔除，采用2.5‑D分解与确定性执行相结合，减少LLM的系统性坐标误差；同时引入小段优化、计划验证、以及不确定性处理等模块。

**🔧 技术方法**

使用的技术包括：LLM（GPT‑4o‑mini）、结构分析器、计划验证器、确定性空间执行器、pseudocode级别的“Peephole Prompt Optimization”规则、决策理论下的模糊问题处理等。

**📊 数据集**

主要数据集为Build What I Mean（BWIM）160轮构建任务；另外在IGLU 500任务上进行迁移实验验证通用性。

**📈 对比分析**

与无分解、纯提示工程、以及使用GPT‑4o的大模型等基线进行对比，GPT‑4o‑mini+2.5‑D分解在BWIM上达94.6%结构准确率，超过GPT‑4o（90.3%）和最优竞争系统（76.3%）；在IGLU上亦提升F1分数至0.798。

**⚠️ 局限性**

局限包括：仅适用于受重力等物理约束的任务；对架构师（clarification）错误的依赖导致性能上限约97.6%；Peephole规则的覆盖范围受任务语义分布限制，迁移到不同指令分布时需重新调整。

---

## 404. Breaking the Illusion: When Positive Meets Negative in Multimodal Decoding

**arXiv ID:** 2605.06679 | [PDF](https://arxiv.org/pdf/2605.06679v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 405. From Clouds to Hallucinations: Atmospheric Retrieval Hijacking in Remote Sensing Vision-Language RAG

**arXiv ID:** 2605.07273 | [PDF](https://arxiv.org/pdf/2605.07273v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 406. Better Protein Function Prediction by Modeling Survivorship Bias

**arXiv ID:** 2605.06879 | [PDF](https://arxiv.org/pdf/2605.06879v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 407. A Hierarchical Ensemble Pipeline for Anomaly Detection in ESA Satellite Telemetry

**arXiv ID:** 2605.06681 | [PDF](https://arxiv.org/pdf/2605.06681v1)

**作者:** Lorenzo Riccardo Allegrini `[一作]` (ContinualIST), Geremia Pompei `[通讯]` (ContinualIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个分层集成管道，对ESA卫星遥测多通道时间序列进行异常检测，融合统计特征、形状子特征、两级掩蔽、内通道堆叠和跨通道聚合。

**💡 创新点**

创新点包括：① 两级掩蔽策略保证信息不泄漏并提升模型多样性；② 将形状子挖掘与统计特征相结合，增强表示能力；③ 三层分层集成架构（基线→堆叠→跨通道）和组权重聚合实现多视角融合。

**🔧 技术方法**

使用的技术有：滑动窗口分段、统计特征提取、形状子挖掘与距离特征、滚动极值池化；基线模型XGBoost、内通道堆叠逻辑回归、跨通道聚合XGBoost/Logistic/LSTM；时间序列交叉验证、贝叶斯搜索调参。

**📊 数据集**

使用数据集：ESA Anomaly Detection Benchmark（ESA‑ADB）Mission 1，约7亿条多通道遥测时间序列，已标注异常与正常事件。

**📈 对比分析**

通过Kaggle公开/私有排行榜与不同分段长度、聚合模型组合对比。最佳配置为分段长度50、形状子特征+XGBoost基线、逻辑回归聚合，公开F0.5 = 0.931（排名第一），私有F0.5 = 0.853（排名第三）。

**⚠️ 局限性**

局限性：计算成本较高（单实验约10小时，CPU）；缺乏实时/大规模部署的可扩展性；对形状子挖掘依赖已知异常，可能对新型异常捕捉不足；仅在ESA‑ADB上验证，需在更广泛的航天遥测数据上进一步评估。

---

## 408. Stabilized neural Hamilton--Jacobi--Bellman solvers: Error analysis and applications in model-based reinforcement learning

**arXiv ID:** 2605.07116 | [PDF](https://arxiv.org/pdf/2605.07116v1)

**作者:** Minseok Kim `[一作]` (Seoul National University of Science and Technology), Yeoneung Kim `[通讯]` (Seoul National University of Science and Technology)

**通讯引用:** 41 | [OpenAlex ID](https://openalex.org/A5028468023)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种半离散的Physics‑Informed Neural Network（PINN）策略迭代方法，用神经网络表示价值函数，并通过在小平移量h处查询网络实现有限差分 HJB 算子，从而在连续时间控制中求解 HJB 方程；

**💡 创新点**

将 HJB 算子视作平移算子，既保持有限差分的单调性与人工粘度稳定性，又保留了 mesh‑free 的可扩展性；给出人口 L² 稳定性理论与误差分解，明确模型识别误差随 1/√ν_h 放大；提供随机取样的误差上界和多步传播条件；

**🔧 技术方法**

Physics‑Informed Neural Networks、半离散有限差分策略迭代、随机取样训练、神经动力学识别（回归）、梯度放大因子分析、贪婪策略改进、能量范数和外壁回溯误差分析；

**📊 数据集**

LQR (4,8,16,32,64 维)、Duffing、Spacecraft、Pendulum、Hopper、3D 量子旋转器（Quad3D）、Allen–Cahn (10D/20D) 等经典控制任务以及标准强化学习基准环境；

**📈 对比分析**

与 SAC、PPO、MBPO 等基线对比；在已知动力学的 LQR 与 Duffing 等任务中与精确 HJB 解做精度对照；在未知动力学或高维任务中，SDPI 在大多数任务上样本效率更高、成本更低（如 64D LQR 成本 34 vs. SAC 94、PPO 201、MBPO 95），在非线性任务中亦保持较好性能；

**⚠️ 局限性**

对接触或强非线性动力学（如 Hopper）仍存在数值抖动与粘度引入的光滑化问题；模型识别误差对最终性能敏感，误差随 1/√ν_h 放大；需要手动设定平移尺度 h 与粘度参数，缺乏自适应机制；

---

## 409. MIST: Multimodal Interactive Speech-based Tool-calling Conversational Assistants for Smart Homes

**arXiv ID:** 2605.06897 | [PDF](https://arxiv.org/pdf/2605.06897v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 410. Temporal Attention for Adaptive Control of Euler-Lagrange Systems with Unobservable Memory

**arXiv ID:** 2605.06877 | [PDF](https://arxiv.org/pdf/2605.06877v1)

**作者:** Giansalvo Cirrincione `[一作]`, Adriano Fagiolini `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于自注意力的元控制器架构，利用窗口化的运动历史动态调节欧拉-拉格朗日系统的控制增益，并在安全强化学习框架下训练；

**💡 创新点**

创新点在于：①将控制增益的自适应视作参数级强化学习任务并使用窗口化自注意力实现；②给出Markovian策略的理论误差下界和窗口满足条件；③构造时间残差算子并通过增量秩追踪（INCRT）求得自注意力头数上界；④采用两阶段搜索‑再训练（NAS）策略实现结构选取与策略训练解耦；

**🔧 技术方法**

主要技术包括自注意力（Transformer）网络、Soft Actor‑Critic强化学习、Lyapunov安全屏障、计算式力矩控制、增量秩追踪算法（INCRT）及理论分析工具；

**📊 数据集**

实验数据集为2自由度平面机械臂，采用Stribeck摩擦模型，随机化负载与运动参考，覆盖三种摩擦记忆时间常数τ_z=1 s、2 s、5 s；

**📈 对比分析**

与固定增益基准及2层Transformer基准进行对比，短记忆与匹配记忆情形下，元控制器实现12–19 %的跟踪RMSE降低，效果量大、p值<0.05；长记忆下无明显优势，且出现多次训练失败；

**⚠️ 局限性**

局限性包括：①静态头数设定在长记忆情形下易导致注意力衰竭；②仅验证Stribeck摩擦，未检验对其他不可观测内存动力学的泛化；③未考虑记忆时间常数随时间变化的非平稳情况；④在极长记忆下参数效率下降。

---

## 411. Disambiguating 2D-3D Correspondences in Gaussian Splatting-based Feature Fields for Visual Localization

**arXiv ID:** 2605.07351 | [PDF](https://arxiv.org/pdf/2605.07351v1)

**作者:** Miso Lee `[一作]` (Sungkyunkwan University), Jae-Pil Heo `[通讯]` (Sungkyunkwan University)

**通讯引用:** 1613 | [OpenAlex ID](https://openalex.org/A5029469141)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 SplitGS-Loc 框架，利用高斯拆分和属性信息构建定位专用的 GSFF，实现高效且稳定的 2D‑3D 匹配；

**💡 创新点**

创新点包括：① Mixture‑of‑Gaussians 拆分减少多对一像素‑高斯对应；② 基于 composition weight 的多视角一致性筛选与特征注册，给每个子高斯分配独特特征；③ 完全消除场景级训练和迭代姿态细化需求；

**🔧 技术方法**

采用 Gaussian Splatting、GSFF、Mixture‑of‑Gaussians 拆分、Composition weight‑based 采样与特征聚合、PnP+RANSAC、SuperPoint 特征提取以及 PoseLib 进行位姿估计；

**📊 数据集**

使用 Cambridge Landmarks（室外）和 7Scenes（室内）两个标准数据集进行评估；

**📈 对比分析**

与 STDLoc 及其他稀疏/稠密重建基线在相同或相近地图容量下对比，SplitGS‑Loc 在所有场景中取得更低的平均定位误差（如 GreatCourt 为 8.7 cm/0.15°，显著优于 STDLoc）且 PnP 收敛速度从 314 ms 降至 215 ms；

**⚠️ 局限性**

局限性在于仍受原始高斯质量限制，难以应对极端光照变化或大规模稀疏轨迹的场景，且对光照变化与稀疏轨迹的鲁棒性有限。

---

## 412. SparseRL-Sync: Lossless Weight Synchronization with ~100x Less Communication

**arXiv ID:** 2605.07330 | [PDF](https://arxiv.org/pdf/2605.07330v1)

**作者:** Lucas Hu `[一作]` (Scitix), Jason Zhao `[通讯]` (Scitix)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文提出了一种无损稀疏同步机制（Sparse Sync），将Trainer→Rollout的完整权重量级广播替换为仅传输更新的索引和值，从而在大模型RL训练中显著降低通信负担。

**💡 创新点**

创新点在于（1）系统性证明并量化了BF16权重增量在多种RL算法和模型规模（8B‑671B）下的极高元素级稀疏度；（2）设计了可直接集成到现有分布式训练框架（Megatron‑LM、FSDP）中的稀疏同步流水线；（3）提出了基于索引差分编码与熵编码的无损压缩方案，使原始32‑54×的压缩比提升至约60‑101×。

**🔧 技术方法**

技术包括混合精度训练（FP32主权重→BF16工作权重），FP16/FP8对比稀疏度分析，索引差分编码、熵编码、按桶分段广播、Ray远程调用、NCCL集体通信、NVLink/RDMA网络、以及在Rollout端的稀疏散列更新。

**📊 数据集**

主要使用了RL微调数据集：Qwen3‑30B‑A3B‑Instruct‑2507、Qwen3‑8B、GLM‑4.5、DeepSeek‑V3.1 等大型预训练模型；实验涵盖GRPO、DAPO、GSPO、Async‑RL、Agentic‑RL等多种RL算法；模型规模从8B扩展至671B。

**📈 对比分析**

与完整重量级广播基线对比，Sparse Sync在500步的RL训练中保持了完全一致的奖励曲线（平均差-8×10⁻⁶，MAE 0.0186，相关系数0.9749）。在通信实验中，106B模型在128 H100 GPU分离部署下，IB‑off（TCP）模式下同步时延从45.6 s降至2.26 s，带来≈20×的速度提升；同样的压缩比在IB‑on（RDMA）下也能实现≈10×。整体吞吐量提升与网络带宽成反比，尤其在低带宽场景下优势更为显著。

**⚠️ 局限性**

局限性包括：依赖BF16 cast导致的高稀疏度并非对所有模型或精度设置都成立；对极小模型或高更新密度参数（如LoRA）不具优势；索引差分和熵编码的额外计算开销在极端低延迟系统中可能不可忽视；以及仅关注权重同步，未涉及梯度或激活压缩等更广泛的通信优化。

---

## 413. Activation Differences Reveal Backdoors: A Comparison of SAE Architectures

**arXiv ID:** 2605.07324 | [PDF](https://arxiv.org/pdf/2605.07324v1)

**作者:** Sachin Kumar `[一作]` `[通讯]` (LexisNexis), Sachin Kumar (LexisNexis)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对比 Crosscoder 与 Diff-SAE 两种稀疏自编码器，系统评估它们在检测 SmolLM2-360M 模型中注入的 SQL 注入后门的效果。

**💡 创新点**

提出 Backdoor Isolation Score (BIS) 评估指标，并首次证明差分 SAE 在后门检测中显著优于 Crosscoder，且单一特征即可实现 100% 精度、0% 假正率。

**🔧 技术方法**

使用差分稀疏自编码器（Diff‑SAE）、稀疏自编码器（Crosscoder）、LoRA 与全秩微调、BIS 指标以及 95% 分位阈值等技术。

**📊 数据集**

构造了大规模语料（约 1.6×10⁹ 种组合）作为训练集，其中包含 40% 的 2024 年触发后门样本，评估集包含 500 个触发样本、1,000 个 2023 年安全样本和 1,000 个无年份上下文样本。

**📈 对比分析**

在 LoRA 与全秩微调、四个 transformer 层（14、18、22、26）和两种扩展因子（4×、32×）下，Diff‑SAE 的 BIS 均维持在 0.39–0.40（精度 1.0，召回 0.25），而 Crosscoder BIS 接近 0，差异显著（≈40×提升）。

**⚠️ 局限性**

局限性包括：仅在 360M 模型与单一 SQL 注入后门上验证；对更大模型、不同类型后门和对抗性后门的鲁棒性未评估；召回率仅为 25%，可能需要多特征或集成方法提升。

---

## 414. Splitting User Stories Into Tasks with AI -- A Foe or an Ally?

**arXiv ID:** 2605.07320 | [PDF](https://arxiv.org/pdf/2605.07320v1)

**作者:** Luka Pavlič `[一作]` (University of Maribor), Christian Ploder `[通讯]` (MCI Management Center Innsbruck)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比传统方法与GitLab Duo生成式AI工具在拆分用户故事为任务的效果，进行控制实验。

**💡 创新点**

首次将生成式AI直接用于敏捷任务拆分，强调人机混合模式的可行性。

**🔧 技术方法**

使用GitLab Duo（LLM生成式AI），结合传统规划手段。

**📊 数据集**

实验基于一套Jakarta EE信息系统的八条用户故事（共56条故事），由42名学生开发团队拆分。

**📈 对比分析**

通过实验测量任务数量、完成率与问卷感知，AI生成任务数量约高43%，完成率约59%；团队普遍偏好混合模式。

**⚠️ 局限性**

AI工具仍缺乏上下文理解，生成无关任务，准确率低；受限于单一工具、学生样本和实验规模，结果可能不具普适性。

---

## 415. Amortized-Precision Quantization for Early-Exit Vision Transformers

**arXiv ID:** 2605.07317 | [PDF](https://arxiv.org/pdf/2605.07317v1)

**作者:** Rui Fang `[一作]` (National Taiwan University), Ming-Syan Chen `[通讯]` (National Taiwan University)

**通讯引用:** 16151 | [OpenAlex ID](https://openalex.org/A5036009069)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Amortized-Precision Quantization（APQ）与其对应的双层优化框架MAQEE，联合优化视觉Transformer（ViT）的早期退出阈值与层级量化精度，以实现低精度下的稳定动态推理。

**💡 创新点**

创新点在于：①利用层级使用率构建可利用性感知的量化成本模型；②引入APQ，显式考虑早期退出导致的量化噪声传播；③设计风险感知的双层优化策略，联合调整退出策略与量化位宽，形成稳健的精度-效率 Pareto 前沿。

**🔧 技术方法**

采用的技术包括：低位量化（FPQ、MPQ）、早期退出（EE）、自蒸馏、双层优化、边缘风险评估（退出边界敏感性、量化诱导漂移等）以及利用率加权的成本计算。

**📊 数据集**

实验数据集涵盖图像分类（CIFAR‑100、ImageNet）、语义分割（SceneParse150）和目标检测（MS COCO），使用 DeiT、ViT 与 Swin 三大Transformer骨干。

**📈 对比分析**

通过与ViT‑EE、LGViT、RepQ、ERQ、MPTQ等基线对比，MAQEE在W4A4设置下在准确率保持不变或仅微降的前提下，将BOP降至95%以上，且在准确率或效率上相较最佳基线提升约20%，在Pareto前沿上表现出更优的准确率‑效率折衷。

**⚠️ 局限性**

局限性包括：检测任务中因早期截断导致的精度下降；方法目前仅在ViT上验证，扩展至其他动态推理方式（如MoE、递归Transformer）仍待研究；双层优化增加了训练复杂度和计算开销。

---

## 416. Implicit Compression Regularization: Concise Reasoning via Internal Shorter Distributions in RL Post-Training

**arXiv ID:** 2605.07316 | [PDF](https://arxiv.org/pdf/2605.07316v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 417. Escaping the Diversity Trap in Robotic Manipulation via Anchor-Centric Adaptation

**arXiv ID:** 2605.07381 | [PDF](https://arxiv.org/pdf/2605.07381v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 418. Combating Organized Platform Abuse: Amplifying Weak Risk Signals with Structural Information

**arXiv ID:** 2605.07383 | [PDF](https://arxiv.org/pdf/2605.07383v1)

**作者:** Meng He `[一作]` (Grab Holdings), Jia Long Loh `[通讯]` (Grab Holdings)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一种无标签、几乎无参数、线性时间复杂度的弱信号结构放大方法，通过在平台交易双向图中聚合弱信号在收敛节点的密度，转化低精度的单个信号为高精度的欺诈检测决策。

**💡 创新点**

创新点：①提出Fraudster's Trilemma理论，揭示组织性欺诈必然产生中央化兑现的结构不变；②基于该不变结构设计的弱信号结构放大框架，能在无监督条件下将全球精度仅为16%的弱信号放大到超过90%的精准率；③方法公开“open‑hand”，攻击者即使知道检测逻辑也难以规避。

**🔧 技术方法**

技术：双向聚合（user→convergence node）计数、经验贝叶斯平滑、比例z检验；增量式实时更新；统计测试而非模型训练；与图模型相比无需种子节点和训练。

**📊 数据集**

数据集：Grab Holdings真实交易数据，包含两起事件——A市场促销滥用（数千Sybil、400+司机）和B市场信用卡欺诈（数百Sybil、<10商户）。真值来源分别为人工审核记录和警方报告。

**📈 对比分析**

比较方法：与启发式规则、传统监督机器学习和GNN等基线进行定性对比。实验中，阈值z=10/40时，精准率>90%，召回率>99%；单个弱信号从全球精度16%提升到91%+；跨模态效果（基础设备层信号检测支付层攻击）也显著。相比基线，方法无需标签、低成本、线性复杂且可实时产生结果。

**⚠️ 局限性**

局限性：只能有效检测大规模爆发型攻击；对低频、分散型欺诈缺乏足够统计聚合；若攻击者能维持多散点兑现（违背低成本约束），结构放大效果下降；方法依赖存在弱信号且满足多对少的聚合结构。

---

## 419. Tools as Continuous Flow for Evolving Agentic Reasoning

**arXiv ID:** 2605.07339 | [PDF](https://arxiv.org/pdf/2605.07339v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 420. QuadNorm: Resolution-Robust Normalization for Neural Operators

**arXiv ID:** 2605.07375 | [PDF](https://arxiv.org/pdf/2605.07375v1)

**作者:** Bum Jun Kim `[一作]` (University of Tokyo), Yutaka Matsuo `[通讯]` (University of Tokyo)

**通讯引用:** 14141 | [OpenAlex ID](https://openalex.org/A5090592819)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并验证了基于梯形积分权重的归一化方法（QuadNorm和BlendQuadNorm），以消除神经算子中的离散化相关误差并提升跨分辨率泛化；

**💡 创新点**

创新点在于将数值积分引入归一化层，得到二阶一致的统计量，并提出混合归一化可兼顾原生精度；同时给出理论的传递误差上界并在多尺度、多模型上实证验证；

**🔧 技术方法**

采用梯形数值积分、神经算子架构（FNO、Galerkin Transformer、Transolver）以及多分辨率训练/评估、统计置信区间等技术；

**📊 数据集**

使用Darcy流动数据集、Poisson–Dirichlet、cavity flow、elasticity等非周期PDE生成数据，以及官方FNO Darcy基准；

**📈 对比分析**

与LayerNorm、InstanceNorm、GroupNorm、RMSNorm、无归一化等基线在2×、4×、8×等跨分辨率任务中对比；QuadNorm在8×分辨率下将误差降低约35%至42%，深层模型误差下降超过4倍；BlendQuadNorm在保持原生精度的同时显著提升跨分辨率鲁棒性；

**⚠️ 局限性**

在周期FFT网格下效果有限，QuadNorm可能牺牲原生精度；需要针对不同架构调节α参数；极端分辨率或更大模型下误差仍有累积；目前实验集中在二维PDE，三维或更复杂域的推广尚待研究。

---

## 421. MISA: Mixture of Indexer Sparse Attention for Long-Context LLM Inference

**arXiv ID:** 2605.07363 | [PDF](https://arxiv.org/pdf/2605.07363v1)

**作者:** Ruijie Zhou `[一作]`, Wenjie Pei `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种将DeepSeek Sparse Attention（DSA）索引器改造为Mixture of Indexer Sparse Attention（MoISA）的技术，通过在头轴上进行Mixture-of-Experts路由来显著降低每个查询的头数；

**💡 创新点**

核心创新在于将索引器的多头视为专家集合，利用块级聚合的轻量级路由器选出少量活跃头（h≪H^I）执行精细的token级打分，既保留了多头的表达能力，又降低了O(H^I L)的计算；

**🔧 技术方法**

使用的技术包括：块级池化路由器、MoE路由策略、稀疏多头潜在注意力（Sparse MLA）、TileLang内核实现以及可选的两阶段层次化重排序；

**📊 数据集**

评估数据集主要为LongBench和Needle-in-a-Haystack（NIAH）检索任务，实验基于DeepSeek‑V3.2和GLM‑5两款长上下文模型；

**📈 对比分析**

与原DSA、Block‑Sparse和HISA等基线相比，MoISA在LongBench上平均分数基本与DSA持平（深度模型误差≤0.2点），在NIAH上保持与DSA相同的绿色热图；在单GPU上，TileLang实现实现了约3.8×的索引器内核加速；

**⚠️ 局限性**

局限性包括：仅评估内核层面延迟，未测量完整模型的端到端速度；索引器的内存访问量未降低；且实验仅在插入MoISA后不再进行额外微调，可能存在进一步提升空间。

---

## 422. GraphReAct: Reasoning and Acting for Multi-step Graph Inference

**arXiv ID:** 2605.07357 | [PDF](https://arxiv.org/pdf/2605.07357v1)

**作者:** Xingtong Yu `[一作]` (Chinese University of Hong Kong), Yuan Fang `[通讯]` (Singapore Management University)

**通讯引用:** 4054 | [OpenAlex ID](https://openalex.org/A5027522861)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向图学习的 reasoning‑acting 框架，利用 LLM 逐步推理并通过图结构检索（拓扑检索、语义检索）以及上下文精炼动作动态扩展与压缩推理上下文，从而完成节点分类任务。

**💡 创新点**

创新点在于：① 将图结构信息纳入可执行动作空间，设计了拓扑检索和语义检索两种补充动作；② 通过上下文精炼动作实现推理过程中上下文的渐进式压缩，解决“扩张‑压缩”冲突；③ 在零样本跨数据集设定下实现图‑LLM 对齐与推理的闭环，显著提升零样本迁移性能。

**🔧 技术方法**

核心技术包括：① 对图编码器进行多模态对比预训练，使其输出对齐到 LLM 词嵌入空间；② 采用 LLM 的链式思考（CoT）生成中间思路并驱动动作；③ 通过指令引导的生成实现检索摘要与上下文精炼；④ 结构化动作序列（检索→精炼）实现多步推理；⑤ 采用固定步骤（K=4）与有限检索规模（N=M=4）保证输入长度可控。

**📊 数据集**

使用八个文本属性图数据集：引用网络（Arxiv、PubMed、Cora）和电商网络（Computer、Photo、Children、History、Sports）。

**📈 对比分析**

与 6 类基线（非图 MLP、监督 GNN、无监督 DGI、知识蒸馏、单纯 LLM 以及图‑LLM 方法）进行零样本跨数据集对比，平均准确率与 Macro‑F1 通常高于或与最先进方法持平；Ablation 结果表明拓扑检索与语义检索均提升性能，完整框架（检索+精炼）表现最佳。

**⚠️ 局限性**

局限性包括：① 预设检索与精炼动作序列不具备自适应选择，可能错失更优动作组合；② 受 LLM 输入长度限制，检索规模与上下文压缩都被迫取较小值；③ 在类别极少（如 PubMed）或高同质性节点的场景下，检索引入噪声导致优势有限；④ 对外部文本检索（Search）效果差，说明仅靠图内信息更为有效。

---

## 423. ShellfishNet: A Domain-Specific Benchmark for Visual Recognition of Marine Molluscs

**arXiv ID:** 2605.07338 | [PDF](https://arxiv.org/pdf/2605.07338v1)

**作者:** Ziheng Zhou `[一作]` (Shanghai Ocean University), Jun Yan `[通讯]` (Shanghai Ocean University)

**通讯引用:** 6464 | [OpenAlex ID](https://openalex.org/A5100625103)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了ShellfishNet数据集并对80种视觉模型进行分类与多模态图像描述评测，旨在提升沿海海底生态监测的自动化与鲁棒性。

**💡 创新点**

创新点包括：①结合实验室和野外采集、人工标注和LLM辅助生成的多模态注释；②引入海底图像失真仿真和长尾类别评估；③对多模态大型语言模型在海洋生态描述中的语义一致性与视觉对齐进行细粒度评估。

**🔧 技术方法**

使用了卷积网络、视觉Transformer、状态空间模型、SSL框架以及多模态LLM（如Qwen、LLaVA、InternVL、Gemini、GPT‑5.4）等多种前沿技术。

**📊 数据集**

使用的主要数据集为ShellfishNet（8691张图，32类），其中500张被人工与LLM联合标注为长文本描述；对比了FishNet等泛水生基准以展示域适配差异。

**📈 对比分析**

评测方法包括传统准确率、长尾准确率、BLEU、METEOR、ROUGE‑L、CIDEr、BERTScore 与 CLIP Score；MambaOut 等高级 Transformer 在分类上达96.47% 最高，Gemini 3.1 Pro 在语义一致性上取得 BERTScore‑F1 0.9006，显示出不同模型在视觉细粒度识别与语义表达上的显著差异。

**⚠️ 局限性**

主要限制在于域知识不足导致的视觉错觉与数值推理弱化，长文本评价受 CLIP token 限制，数据仍偏向北方沿海且缺乏跨域泛化实验，后续需进一步提升模型的多域适应与解释性。

---

## 424. TREA: Low-precision Time-Multiplexed, Resource-Efficient Edge Accelerator for Object Detection and Classification

**arXiv ID:** 2605.07321 | [PDF](https://arxiv.org/pdf/2605.07321v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 425. Rethinking Importance Sampling in LLM Policy Optimization: A Cumulative Token Perspective

**arXiv ID:** 2605.07331 | [PDF](https://arxiv.org/pdf/2605.07331v1)

**作者:** Yuheng Zhang `[一作]` (University of Illinois Urbana-Champaign), Nan Jiang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 6245 | [OpenAlex ID](https://openalex.org/A5008181744)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CTPO 算法，通过累积 token 的重要性采样比率实现无偏前缀校正，解决 RL 训练中 bias‑variance 的矛盾。

**💡 创新点**

创新点在于引入累积 token IS 比率作为无偏且方差更低的纠正因子，并结合位置自适应裁剪，实现对不同 token 位置一致的正则化。

**🔧 技术方法**

使用了基于 token‑级 MDP 的政策梯度框架、重要性采样、位置自适应裁剪以及工具集成推理 (TIR) 交互式生成。

**📊 数据集**

在 DeepScaleR 训练集以及四个竞赛级数学推理基准（AIME 2025/2026、HMMT 2025、BRUMO 2025）上进行实验。

**📈 对比分析**

与 GRPO（token‑级 IS）和 GSPO（长度归一化全序列 IS）对比，CTPO 在 Qwen3-4B 与 Qwen3-14B 上在 avg@32 评估上分别提升约 3.7–5.8% 的平均通过率，显示出更优性能。

**⚠️ 局限性**

局限在于仅验证于工具集成推理场景，尚未探讨更长时序或多模态环境下的泛化与计算开销。

---

## 426. GC-ART: Global Learnable Second-Order Rational Tone Curves for Illumination Robustness

**arXiv ID:** 2605.07329 | [PDF](https://arxiv.org/pdf/2605.07329v1)

**作者:** Wei Huang `[一作]` (Microsoft), Joyce Huang `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并评估了 GC-ART，一种基于直方图的全局 Rational 曲线前端，用于提升图像分类在不同照明失真下的鲁棒性。

**💡 创新点**

创新点在于将可微分的 Rational 曲线与图像直方图结合，使用 643 参数的 MLP 预测全局曲线，实现全局单点映射且保持边缘不被模糊。

**🔧 技术方法**

使用软直方图提取、MLP 预测、Rational 曲线映射以及软单调性正则化；训练仅靠交叉熵，端到端优化。

**📊 数据集**

使用 CIFAR-10（32×32）数据集，并在 CIFAR-10-C 生成的亮度、对比度、暗化三种全局照明失真上进行评估。

**📈 对比分析**

与无增强、Zero‑DCE 风格卷积增强、Histogram Equalization、CLAHE 和 Gamma 校正等方法对比，GC‑ART 在暗化上与 Zero‑DCE 接近，在对比度失真上取得最佳学习型方法（48.45%），且 FLOPs 仅为卷积增强的 1/42。

**⚠️ 局限性**

局限在于仅评估了低分辨率 CIFAR-10、三种合成照明失真、有限种子、未调优超参数，且未验证自然 OOD、噪声、模糊等场景；Rational 曲线可能产生超出 [0,1] 的值，仅使用软单调性约束。

---

## 427. DCGL: Dual-Channel Graph Learning with Large Language Models for Knowledge-Aware Recommendation

**arXiv ID:** 2605.07314 | [PDF](https://arxiv.org/pdf/2605.07314v1)

**作者:** Xinchi Zou `[一作]` (Huazhong University of Science and Technology), Zhiwei Shen `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DCGL 框架，采用双通道图学习解决知识图谱与大语言模型融合时因交互频率异质性导致的语义模糊与稀疏性问题。

**💡 创新点**

创新点在于：1）双通道结构将语义与行为信号分离；2）多级对比学习增强鲁棒性并对齐两通道；3）频率感知门控融合根据交互频率动态平衡两通道。

**🔧 技术方法**

使用了 RGAT、LightGCN、LLM 生成实体描述并通过文本嵌入得到语义向量、InfoNCE 对比学习、门控网络与正则化等技术。

**📊 数据集**

实验数据集包括四个真实推荐数据集：DBbook、Book‑Crossing、MovieLens、Amazon‑Book，均包含知识图谱信息。

**📈 对比分析**

与传统 CF、KG 嵌入、GNN、LLM 增强等 14 种基线对比，DCGL 在 Recall@K/NDCG@K 上均显著提升，尤其在稀疏和冷启动场景表现突出。

**⚠️ 局限性**

局限性包括：依赖离线 LLM 生成的质量受限；门控策略需手动调参；对极度噪声 KG 的鲁棒性仍需进一步验证。

---

## 428. UniISP: A Unified ISP Framework for Both Human and Machine Vision

**arXiv ID:** 2605.07359 | [PDF](https://arxiv.org/pdf/2605.07359v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 429. Weather-Robust Scene Semantics with Vision-Aligned 4D Radar

**arXiv ID:** 2605.07367 | [PDF](https://arxiv.org/pdf/2605.07367v1)

**作者:** Kali Hamilton `[一作]` (University of Colorado Boulder), Christoffer Heckman `[通讯]` (University of Colorado Boulder)

**通讯引用:** 1136 | [OpenAlex ID](https://openalex.org/A5009328815)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

使用冻结的视觉语言模型（VLM）对4D毫米波雷达信号进行对齐编码，并通过该对齐的雷达嵌入生成结构化场景描述，从而评估雷达在恶劣天气下的语义理解能力。

**💡 创新点**

提出了雷达-视觉嵌入的对齐方法，并通过在投影器输出处加入LayerNorm解决token‑norm不匹配的问题，使得冻结的VLM能够有效利用雷达特征，展示雷达在雾、雪等天气下保持语义理解的鲁棒性。

**🔧 技术方法**

采用ResNet‑18雷达编码器（含PETR位置编码）、MLP+LayerNorm投影器、冻结的SigLIP视觉编码器进行对齐、冻结的Qwen2.5‑VL‑3B VLM加LoRA微调、结构化caption生成与解析，评估其在不同天气下的检测性能。

**📊 数据集**

使用K‑RADAR 4D雷达数据集，包含雨、雾、轻雪、重雪等多种天气，约8k帧训练集，按序列拆分以避免时序泄漏。

**📈 对比分析**

通过将生成的caption解析为检测结果，计算F1、精度、召回、范围MAE、方位误差和假阳性率。雷达模型在雾、雪下F1>0.44、假阳性率<0.38，摄像头基线在同一天气下F1≈0且假阳性>0.9；JSON格式下最高F1为0.54。

**⚠️ 局限性**

受限于训练数据极少（仅约8k帧），评价上限受caption格式（最多4个目标）限制，自动回归token化导致空间误差远大于雷达本身分辨率，且缺乏专用检测头，雷达- VLM对齐在更复杂特征空间下难以充分利用。

---

## 430. FlightSense: An End-to-End MLOps Platform for Real-Time Flight Delay Prediction via Rotation-Chain Propagation Features and Agentic Conversational AI

**arXiv ID:** 2605.07364 | [PDF](https://arxiv.org/pdf/2605.07364v1)

**作者:** Aditi J. Shelke `[一作]` (Stevens Institute of Technology), Yash M. Kamerkar `[通讯]` (Stevens Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出 FlightSense，一个端到端的 MLOps 平台，用于实时预测航班延误并提供基于自然语言的交互式决策支持。

**💡 创新点**

创新点在于：① 构建基于机身旋转链的延误传播特征，显著提升预测性能；② 将 NOAA 天气特征与旋转链特征融合；③ 将模型推理与 Amazon Bedrock Nova Micro 代理式对话 AI 结合，形成首个实时部署的 ML 推理 + 对话界面。

**🔧 技术方法**

使用技术包括 XGBoost 二分类器、AWS Lambda、SageMaker 训练与推理、Streamlit 仪表盘、Amazon Bedrock Nova Micro 语言模型与工具调用、乘法概率组合等。

**📊 数据集**

数据集为 2018 年美国国内航班的 BTS On-Time Performance 记录（约 7.07M 条）和 10 家主要机场的 NOAA GHCND 天气观测。

**📈 对比分析**

通过三版逐步 ablation 与内部基线比较，V3（含传播 + 天气特征）在 10% 测试集上达 ROC AUC 0.879，明显优于单一 XGBoost 基线（0.732）和文献中两阶段模型（0.898）。

**⚠️ 局限性**

局限包括：训练-测试未做时间序列验证、HPO 预算有限、仅使用起点天气、未加入目的地天气、部分基线训练样本受限、部分信息泄露风险等。

---

## 431. RCoT-Seg: Reinforced Chain-of-Thought for Video Reasoning and Segmentation

**arXiv ID:** 2605.07334 | [PDF](https://arxiv.org/pdf/2605.07334v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 432. Confidence-Aware Alignment Makes Reasoning LLMs More Reliable

**arXiv ID:** 2605.07353 | [PDF](https://arxiv.org/pdf/2605.07353v1)

**作者:** Kejia Chen `[一作]` (Zhejiang University), Ruoxi Jia `[通讯]` (Virginia Tech)

**通讯引用:** 2828 | [OpenAlex ID](https://openalex.org/A5032275274)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于步骤级别的置信度对齐方法 CASPO，通过迭代 DPO 训练模型在生成时将置信度与逻辑正确性对齐，并在推理时使用置信度驱动的树搜索（CaT）来剪枝不确定路径。

**💡 创新点**

创新点在于：①不依赖外部奖励模型或大规模采样，仅利用模型自身的 token 级熵作为置信度信号；②在训练与推理之间形成统一的置信度对齐框架；③在推理时实现 O(V) 的低延迟路径剪枝。

**🔧 技术方法**

使用的技术包括：迭代 Direct Preference Optimization (DPO)、token 级 Shannon 熵置信度估计、基于熵阈值的思考树剪枝（CaT）以及对比实验中的树搜索与自一致性等方法。

**📊 数据集**

数据集方面：主要在 MATH500、Minerva-Math、OlympiadBench、AIME2024/25 等数学推理基准，同时使用 PRM800K 生成评估数据并发布了带置信度标注的步骤级别数据集。

**📈 对比分析**

与多种基线（GRPO、Simple‑RL‑Zero、PURE‑VR、rStar‑Math、Satori、CoT、Self‑Consistency、DiPT）进行对比，CASPO 在 Qwen2.5‑7B‑Instruct 上平均提升约 4‑5% Pass@1，并在 Qwen3‑8B‑Base 上超越树搜索基线，且在推理时保持 2‑3 倍的速度。

**⚠️ 局限性**

局限性包括：①对非常长推理路径的熵估计可能仍受模型温度影响；②在极端复杂推理任务中仍需要更高阈值调优；③实验主要集中在数学领域，跨模态通用性尚待进一步验证。

---

## 433. Mage: Multi-Axis Evaluation of LLM-Generated Executable Game Scenes Beyond Compile-Pass Rate

**arXiv ID:** 2605.07342 | [PDF](https://arxiv.org/pdf/2605.07342v1)

**作者:** Hugh Xuechen Liu `[一作]` (Chalmers University of Technology and University of Gothenburg), Kıvanç Tatar `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个针对Unity场景代码生成的基准和四轴评估协议，利用大模型在多种条件下生成代码并进行系统实验。

**💡 创新点**

创新点在于揭示编译通过率与功能正确性可能不一致，并提出四轴评估（编译、运行、结构、机制）来全面衡量生成质量；同时发现对IR的细粒度（行为层 vs 全场景）并不会显著影响结果。

**🔧 技术方法**

采用了自然语言到C#的代码生成、自动提取结构化IR、统一的API映射规则、四轴评估框架，并使用McNemar等统计方法进行对比分析。

**📊 数据集**

使用了26个手工设计的Unity 2D小游戏目标模式，自动提取两层IR（行为级和全场景级），共生成858条代码实例作为评测数据。

**📈 对比分析**

对四种开放权重LLM（7B–30B）在无IR、行为IR、全场景IR三种条件下进行比较；无IR条件的运行通过率最高（约43%），但结构和机制F1接近0；IR条件的运行通过率下降至14–21%，但结构和机制F1提升至0.8–1.0；两种IR细粒度之间无显著差异。

**⚠️ 局限性**

局限性包括未评估游戏可玩性、仅使用单一引擎（Unity 2022.2）、样本规模仅适用于评估而非训练、模型覆盖有限、三因子解释性未进行因果验证。

---

## 434. Unconsented Sensing: A Sociotechnical Governance Framework for 6G ISAC

**arXiv ID:** 2605.07328 | [PDF](https://arxiv.org/pdf/2605.07328v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 435. GEM: Generating LiDAR World Model via Deformable Mamba

**arXiv ID:** 2605.07326 | [PDF](https://arxiv.org/pdf/2605.07326v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 436. CellScientist: Dual-Space Hierarchical Orchestration for Closed-Loop Refinement of Virtual Cell Models

**arXiv ID:** 2605.07335 | [PDF](https://arxiv.org/pdf/2605.07335v1)

**作者:** Mengran Li `[一作]` (Hong Kong Institute Of Science And Innovation), Zelin Zang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

CellScientist 提出了一个双空间层次框架，用于在虚拟细胞建模（VCM）中实现可追溯的迭代改进，将假设空间与可执行实现空间耦合并通过闭环反馈路由优化模型。

**💡 创新点**

其创新点在于：① 设计假设–实现双空间闭环，② 通过分层层次拓扑（HRT）实现结构化反馈路由，③ 在实现层面采用逻辑约束对齐（LCA），④ 在性能差异路由（PDR）中将执行反馈定位到局部假设更新，并提供可审计的改进轨迹。

**🔧 技术方法**

使用的技术包括：大语言模型（LLM）生成可执行代码、逻辑约束对齐（LCA）保证实现合法性、分层层次拓扑（HRT）管理模型决策依赖、性能差异路由（PDR）实现执行误差到假设的映射，以及整体闭环调度与审计机制。

**📊 数据集**

实验使用的数据集包括 Cell Painting 的四个公共数据集（BBBC021、BBBC036、BBBC047、CPG0016）、LINCS2020 L1000 转录组（Seven Cell Lines 与 Full Data）以及单细胞扰动数据集（Norman、Schiebinger、Papalexi）。

**📈 对比分析**

在 SMILES/plate split 上与固定预测器、搜索基线和 CellForge 进行对比，CellScientist 的最终可执行模型在 MSE/PCC/R^2 方面均取得最高分；在 LINCS2020 上 RMSE/PCC 领先；单细胞扰动上 PCC_DE 更优；在预算受限的工作流中，它实现了最高的规则合规率 (R‑SR)、平均/最佳 PCC 以及最低的整体运行时间。

**⚠️ 局限性**

局限性包括：对 LLM 代码生成质量的依赖导致推理成本较高、对极端批量或跨实验平台的泛化能力尚未充分验证、以及对生成设计的生物学解释和生物学效用验证仍需进一步加强。

---

## 437. Beyond Linear Attention: Softmax Transformers Implement In-Context Reinforcement Learning

**arXiv ID:** 2605.07333 | [PDF](https://arxiv.org/pdf/2605.07333v1)

**作者:** Zixuan Xie `[一作]` (University of Virginia), Shangtong Zhang `[通讯]` (University of Virginia)

**通讯引用:** 716 | [OpenAlex ID](https://openalex.org/A5033834190)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究Transformer在无参数更新的情境下，如何通过标准softmax注意力在预训练后自适应完成新的价值评估任务。

**💡 创新点**

首次在不使用线性注意力简化的前提下，理论证明Transformer前向传播等价于加权softmax TD迭代，给出收敛保证，并证明在强化学习预训练中该参数结构为全局最优。

**🔧 技术方法**

使用softmax自注意力的Transformer、加权softmax TD算法、核方法、时间平移模块（TSM）以及循环共享参数的Transformer结构。

**📊 数据集**

主要实验数据集为随机生成的Boyan链多任务MRP（多任务随机MRP）。

**📈 对比分析**

与传统TD和线性注意力实现对比，实验显示MSVE随层数和上下文长度递减，证明了收敛性和参数“TD块”出现，性能优于线性注意力模型。

**⚠️ 局限性**

收敛保证依赖于softmax注意力的对角占优假设，且在非平稳轨迹或控制任务上的适用性尚未验证。

---

## 438. Generative Modeling with Flux Matching

**arXiv ID:** 2605.07319 | [PDF](https://arxiv.org/pdf/2605.07319v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 439. Inference-Time Attribute Distribution Alignment for Unconditional Diffusion

**arXiv ID:** 2605.07456 | [PDF](https://arxiv.org/pdf/2605.07456v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 440. SSP-based construction of evaluation-annotated data for fine-grained aspect-based sentiment analysis

**arXiv ID:** 2605.07446 | [PDF](https://arxiv.org/pdf/2605.07446v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 441. Learning Image-Adaptive Scale Fields for Metric Depth Recovery

**arXiv ID:** 2605.07418 | [PDF](https://arxiv.org/pdf/2605.07418v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 442. LaTER: Efficient Test-Time Reasoning via Latent Exploration and Explicit Verification

**arXiv ID:** 2605.07315 | [PDF](https://arxiv.org/pdf/2605.07315v1)

**作者:** Xuan Li `[一作]` (University of Science and Technology of China), Junnan Zhu `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 1232 | [OpenAlex ID](https://openalex.org/A5015809194)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Latent-Then-Explicit Reasoning（LaTER）框架，先在连续隐状态空间进行探索，再切换到显式链式思考进行验证与答案生成。

**💡 创新点**

创新点在于将隐状态回投到输入嵌入并保留 KV 缓存，实现训练‑free 的自适应切换，并通过监督的 Latent‑Switch‑69K 数据进一步提升效率和准确率。

**🔧 技术方法**

使用的技术包括隐状态投射（伪逆或学习投影）、entropy 与终止词探针自适应切换、基于 KL 的自蒸馏以及 latent halting 监督。

**📊 数据集**

训练数据为自蒸馏得到的 Latent‑Switch‑69K（约69k 条）以及公开的数学、代码与知识问答 benchmark（AIME、MATH‑500、GSM8K、GPQA、ARC‑Challenge、HumanEval+、MBPP+）。

**📈 对比分析**

与传统显式 CoT 以及仅做 SFT 的 CoT baseline 对比，训练后的 LaTER 在 AIME 2025 上实现 80.0% 的准确率（比 CoT 提升 10 分），并将平均 token 数减少 33%，其余基准亦维持或提升准确率同时降低 token。

**⚠️ 局限性**

局限性包括手工设计的切换阈值在不同样例上的泛化不足，低熵样本可能导致无效切换，以及训练后模型对不同任务的效果不均衡，需进一步学习更丰富的实例适应停止策略。

---

## 443. GRaSp: Automatic Example Optimization for In-Context Learning in Low-Data Tasks

**arXiv ID:** 2605.07454 | [PDF](https://arxiv.org/pdf/2605.07454v1)

**作者:** Simen Bihaug-Frøyland `[一作]` (University of Agder), Henrik Brådland `[通讯]` (University of Agder)

**通讯引用:** 21 | [OpenAlex ID](https://openalex.org/A5063325405)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一种三阶段框架GRaSp，用于在低数据任务中自动优化上下文学习示例。

**💡 创新点**

创新点在于将大规模LLM生成的合成示例与聚类降维相结合，并在遗传算法中引入多样性自适应变异机制，实现从广泛探索到精细优化的动态转变。

**🔧 技术方法**

使用了LLM（OpenAI GPT‑OSS‑120B）生成示例，Qwen3‑Embedding‑4B进行句子嵌入，UMAP降维，HDBSCAN聚类，以及基于DEAP的（μ+λ）遗传算法，并设计了基于群体多样性的自适应变异概率。

**📊 数据集**

主要实验数据集为金融命名实体识别任务FiNER‑139（以及其公开的Financial‑NER‑NLP版本），并在此基础上生成合成候选池。

**📈 对比分析**

与零样本、随机少样本以及全参数监督模型对比，GRaSp在非合成数据上取得了45.84% μ‑F1（比零样本提升5.61个百分点，比随机提升2.93个百分点），但仍显著落后于监督模型。

**⚠️ 局限性**

局限性包括：合成候选池缺乏足够分布多样性导致泛化差；整体性能与监督基线差距较大；对候选池规模和进化迭代次数的敏感度仍需进一步研究。

---

## 444. InsHuman: Towards Natural and Identity-Preserving Human Insertion

**arXiv ID:** 2605.07402 | [PDF](https://arxiv.org/pdf/2605.07402v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 445. Velocity-Space 3D Asset Editing

**arXiv ID:** 2605.07385 | [PDF](https://arxiv.org/pdf/2605.07385v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 446. GameGen-Verifier: Parallel Keypoint-Based Verification for LLM-Generated Games via Runtime State Injection

**arXiv ID:** 2605.07442 | [PDF](https://arxiv.org/pdf/2605.07442v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 447. LoBoFit: Flexible Garment Refitting via Local Bone Mapping Blending

**arXiv ID:** 2605.07450 | [PDF](https://arxiv.org/pdf/2605.07450v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 448. Generating training datasets for legal chatbots in Korean

**arXiv ID:** 2605.07432 | [PDF](https://arxiv.org/pdf/2605.07432v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 449. Exploring CoCo Challenges in ML Engineering Teams: Insights From the Semiconductor Industry

**arXiv ID:** 2605.07389 | [PDF](https://arxiv.org/pdf/2605.07389v1)

**作者:** A. Azamnouri `[一作]` (Technical University of Munich), S. Wagner `[通讯]` (Technical University of Munich)

**通讯引用:** 18496 | [OpenAlex ID](https://openalex.org/A5073284900)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过对一家全球半导体公司的12名从事机器学习（ML）工程的专业人士进行半结构化访谈，定性分析了在硬件驱动的制造环境中ML工程团队面临的协作与沟通（CoCo）挑战与实践，并提出了多种应对策略。

**💡 创新点**

创新点在于：1）首次在硬件行业（半导体）中系统探讨ML工程的CoCo挑战，揭示了数据治理、制造周期长、物理过程耦合等硬件特有约束如何放大传统软件工程中的CoCo问题；2）识别了16类挑战并与已有研究对比，区分了普遍与硬件特有的挑战；3）提出了19种针对具体挑战的实际对策，体现了行业内自下而上的经验总结。

**🔧 技术方法**

研究方法主要是：1）使用半结构化访谈访谈法，设计访谈协议并经过两次试点改进；2）采用自动转写（Whisper）+人工校正的方式获取文本；3）进行开放编码、协同校准、代码集成，最终归纳出主题与挑战。

**📊 数据集**

数据来源为12名在半导体公司工作的从业者（包含数据科学家、软件/系统工程师、物理学家、过程工程师等），访谈时长约30–45分钟。

**📈 对比分析**

本文未进行数值对比或性能评估，而是采用定性描述和归纳分析，呈现挑战出现频次和实践效果。

**⚠️ 局限性**

局限性包括：1）样本来自单一公司，缺乏跨组织验证；2）访谈数量有限（12人），可能无法覆盖全部多样化场景；3）因保密要求无法公开访谈原文，分析过程缺乏外部复核；4）研究聚焦于CoCo问题，未涉及模型性能或技术实现细节。

---

## 450. Game-Theoretic Analysis of Transaction Selection in DAG-Based Distributed Ledgers

**arXiv ID:** 2605.07387 | [PDF](https://arxiv.org/pdf/2605.07387v1)

**作者:** Sebastian Müller `[一作]` (Aix-Marseille Université), Alexandre Reiffers-Masson `[通讯]` (IMT Atlantique)

**通讯引用:** 117 | [OpenAlex ID](https://openalex.org/A5003560609)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

分析并计算了两种费用分配机制（随机费用分配RFA与协作费用分配CFS）的对称纳什均衡（NE），并通过数值模拟比较其交易吞吐量和费用吞吐量；

**💡 创新点**

首次给出这两种模型的混合对称NE的显式表达，并提出基于KKT条件的凸优化求解方法，实现高效计算；

**🔧 技术方法**

采用单次游戏理论框架、KKT条件、凸优化技术以及数值模拟手段；

**📊 数据集**

使用模拟生成的Zipf分布交易费用数据（最大费用从5到100，交易数从100到10,000），并设定N=10、块容量b=100等参数；

**📈 对比分析**

通过与随机选择（RTS）和比例选择（PTS）等基准策略对比，发现CFS在费用吞吐上最高，RFA最差；但CFS的有效交易吞吐最低；

**⚠️ 局限性**

仅考虑单次决策、静态费用与统一延迟，未涵盖多轮/重复游戏、动态费用、网络延迟异质性及DAG结构对奖励的影响，且计算复杂度在大规模交易时仍较高。

---

## 451. Bounded Fitting for Expressive Description Logics

**arXiv ID:** 2605.07452 | [PDF](https://arxiv.org/pdf/2605.07452v1)

**作者:** Maurice Funk `[一作]` (Leipzig University and ScaDS.AI Center), Tom Voellmer `[通讯]` (TU Dortmund University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 bounded fitting 在更表达力的描述逻辑 ALCHIQF（包含逆角色、限定数限制与特征比较）中的应用，并给出了实现与实验验证。

**💡 创新点**

创新点在于证明了在该扩展逻辑中仍能在多项式时间内求得最优 fitting，提出了可扩展的 bounded fitting 算法（对逆角色、数限制、特征比较进行处理），并给出 Occam 与 PAC 泛化保证的充分条件。

**🔧 技术方法**

采用 SAT 归约实现 bounded fitting，并引入 bisimulation 前处理与多线程并行搜索；还设计了数值上限 g(k) 控制数限制，并使用特征值分段编码。

**📊 数据集**

使用标准的 SML-benchmarks（Carcinogenesis、Hepatitis、Lymphography 等）以及从 YAGO 4.5 构造的专门考察数限制的 benchmark 进行评估。

**📈 对比分析**

与 EvoLearner、TDL 以及基于公式搜索的算法进行比较，实验表明在泛化性能上相当甚至优于现有方法，尤其在数限制任务中取得更高准确率和 F1 分数；在求解时间上通过预处理和并行化显著提升。

**⚠️ 局限性**

主要局限包括：特征比较在无界域/无界度数时不保证 Occam；SAT 编码中数值上限 g(k) 的选择影响性能；对于高度表达式复杂度的概念，搜索空间仍巨大，导致计算时间增长。

---

## 452. SR$^2$-LoRA: Self-Rectifying Inter-layer Relations in Low-Rank Adaptation for Class-Incremental Learning

**arXiv ID:** 2605.07420 | [PDF](https://arxiv.org/pdf/2605.07420v1)

**作者:** Fengqiang Wan `[一作]` (Nanjing University of Science and Technology), Yang Yang `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 39264 | [OpenAlex ID](https://openalex.org/A5100397594)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于低秩适配的自纠正层间关系（SR^2-LoRA）方法，用于解决类增量学习中的灾难性遗忘。

**💡 创新点**

从层间关系漂移的角度重新定义灾难性遗忘，并通过对关系矩阵奇异值进行对齐来抑制漂移，显著提升知识保持。

**🔧 技术方法**

使用低秩 LoRA、奇异值分解（SVD）对齐、预训练 ViT-B/16 模型以及参数高效微调（PEFT）技术。

**📊 数据集**

在四个公开基准上评估：CIFAR-100、ImageNet‑R、CUB‑200 和 ImageNet‑A。

**📈 对比分析**

与 CODA‑Prompt、SLCA、InfLoRA、LoRA‑DRS、MACIL 等 PTM 相关的 CIL 方法进行比较；在 5‑50 任务设置下平均准确率提升约 4‑6%，尤其在长任务序列中优势更明显。

**⚠️ 局限性**

实验主要聚焦于 Vision Transformer，缺乏对更深模型或跨模态任务的验证；对超参数 λ 的选取仍需经验调整。

---

## 453. OrchJail: Jailbreaking Tool-Calling Text-to-Image Agents by Orchestration-Guided Fuzzing

**arXiv ID:** 2605.07414 | [PDF](https://arxiv.org/pdf/2605.07414v1)

**作者:** Jianming Chen `[一作]` (Chinese Academy of Sciences), Fanjiang Xu `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 8850 | [OpenAlex ID](https://openalex.org/A5076909902)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种面向工具调用的文本生成图像（T2I）代理的“orchestration-guided fuzzing”框架，用以在黑盒环境下发现并利用工具调度模式进行安全攻击（jailbreak）。

**💡 创新点**

创新点在于：①首次将工具调度视为新的攻击面；②通过从成功攻击案例中抽象宏观规划、微观调度和工具选择三维调度模式，并用因果推理关联到提示词，形成可解释的提示-调度因果知识；③将该因果知识直接驱动变异和候选评估，实现对多步工具链的定向探索。

**🔧 技术方法**

主要技术包括：黑盒模糊测试循环、工具感知种子生成、基于CLIP的语义对齐评估、基于LLM的围绕因果知识的变异（bypass/semantic两支路）、三维评分（bypass概率、语义漂移、调度可行性）和LLM-as-Judge的多目标排序，以及正则表达式+LLM因果推理实现调度抽象与因果关联。

**📊 数据集**

使用的数据集：VBCDE-100（含100条敏感提示，覆盖暴力、血腥、非法行为、歧视等），以及三种目标代理（GenArtist、CREA、LayerCraft）的工具库；生成对比图像时采用无安全过滤的Stable Diffusion，用于FID评估。

**📈 对比分析**

与DACA、RING、SneakyPrompt、JailFuzzer等基线对比，实验表明本文方法在三大目标代理上均获得最高的一次性成功率、最低FID（图像质量更高）以及最低查询次数；在PPL（提示自然度）上也优于所有基线；对PPL‑based防御几乎无影响，对SmoothLLM系列防御仍保持69%以上的成功率，表明对常见防御具有一定鲁棒性。

**⚠️ 局限性**

局限性：①依赖于可观测的工具调用轨迹，若代理内部调度更为隐蔽或加密则效果受限；②对LLM的推理与变异能力高度依赖，强安全保障的LLM可能无法处理敏感内容；③仅在三种代理与VBCDE‑100类别上验证，无法直接证明对更大规模、多样化工具库或更强防御的适用性；④黑盒查询成本仍存在，虽然已显著降低，但在极度受限的查询预算下仍可能不足。

---

## 454. The Proxy Presumption: From Semantic Embeddings to Valid Social Measures

**arXiv ID:** 2605.07409 | [PDF](https://arxiv.org/pdf/2605.07409v1)

**作者:** Baishi Li `[一作]` (National University of Singapore), Ke-Wei Huang `[通讯]` (National University of Singapore)

**通讯引用:** 1009 | [OpenAlex ID](https://openalex.org/A5061690540)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出构造效度协议（CVP）和对抗性中性化方法，系统化评估NLP嵌入在计算社会科学中测量抽象社会构念的有效性。

**💡 创新点**

将心理测量学、因果表示学习与NLP测量实践融合，阐明“代理假设”导致的混杂风险，并提供可操作的效度验证套件和反事实中性化技术。

**🔧 技术方法**

利用因果表示学习原理、对抗性去除、零空间投影、LLM生成的反事实重写以及多种效度检验（稳定性、收敛、判别、增量、已知组和预测效度）。

**📊 数据集**

主要使用公开情感标注数据集GoEmotions进行示例验证，并在2020–2025年ACL/NAACL/EMNLP论文中进行法医编码评估。

**📈 对比分析**

通过ICC和AUC等指标证明稳定性（ICC≥0.85）和增量效度（AUC提升约0.02）；与传统单一相似度度量相比，展示了更高的判别力和对混杂的抵消。

**⚠️ 局限性**

未在新数据集上完整实现CVP的端到端案例；效度评估主要基于已公布信息，可能低估真实做法；对LLM中性化的通用性和计算成本仍待进一步实验。

---

## 455. From Conceptual Scaffold to Prototype: A Standardized Zonal Architecture for Wi-Fi Security Training

**arXiv ID:** 2605.07400 | [PDF](https://arxiv.org/pdf/2605.07400v1)

**作者:** Vyron Kampourakis `[一作]` (Norwegian University of Science and Technology), Sokratis Katsikas `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 4159 | [OpenAlex ID](https://openalex.org/A5022741687)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了面向Wi‑Fi安全实验的概念化 Cyber Range 架构，并实现了一个基于 Linux 命名空间、ath9k 软件无线电的原型；

**💡 创新点**

核心创新在于把 IEEE 802.11 作为平台核心，采用五个功能区域（核心基础设施、学习管理与支持、监控、管理、访问控制）实现模块化、可扩展的设计；

**🔧 技术方法**

利用 Flask+SQLite 的 Web 前端、Python 脚本与 Bash 自动化、Linux 命名空间隔离、ath9k 软件无线电、DHCP/802.1X/EAP 支持、以及一系列 Wi‑Fi 攻防工具；

**📊 数据集**

主要使用自行定义的情景脚本与配置文件作为数据集，未涉及公开数据集；

**📈 对比分析**

论文未给出量化实验或性能对比，只说明原型实现了情景生成、存储、检索和实例化流程，性能未进行系统评估；

**⚠️ 局限性**

局限包括仅支持 Wi‑Fi（不兼容 5G/Bluetooth 等其他无线技术）、基于软件仿真缺乏真实硬件交互、可扩展性和多用户并发性能尚未验证、缺乏高级监控与自适应教学机制。

---

## 456. ST-Gen4D: Embedding 4D Spatiotemporal Cognition into World Model for 4D Generation

**arXiv ID:** 2605.07390 | [PDF](https://arxiv.org/pdf/2605.07390v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 457. EditTransfer++: Toward Faithful and Efficient Visual-Prompt-Guided Image Editing

**arXiv ID:** 2605.07455 | [PDF](https://arxiv.org/pdf/2605.07455v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 458. VNN-LIB 2.0: Rigorous Foundations for Neural Network Verification

**arXiv ID:** 2605.07451 | [PDF](https://arxiv.org/pdf/2605.07451v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 459. A Flexible Adaptive Stable Clustering Algorithm for Archive-Scale Online Mass Spectrometry

**arXiv ID:** 2605.07424 | [PDF](https://arxiv.org/pdf/2605.07424v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 460. Data Contamination in Neural Hieroglyphic Translation: A Reproducibility Study

**arXiv ID:** 2605.07453 | [PDF](https://arxiv.org/pdf/2605.07453v1)

**作者:** Ammar Toutou `[一作]` (Alamein International University), Christine Basta `[通讯]` (University of Basque Country)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

审计古埃及象形文字到德语的 NMT 数据集中的污染问题，剔除目标泄漏并提供去污染的测试集与基线评估。

**💡 创新点**

首次对古语言 NMT 进行系统的污染审核，量化公式化重复导致的训练‑测试泄漏，并提出去污染协议及公开清洁测试集。

**🔧 技术方法**

采用目标层去重、字符 8‑gram 重叠检测、文档层去污染等技术，评估 M2M‑100 与 mBART‑50 等多模型，并使用 BLEU、chrF++、COMET‑22 等自动指标。

**📊 数据集**

使用 Thesaurus Linguae Aegyptiae（TLA）古埃及象形文字与德语平行语料，过滤后得到 18,669 对训练样本；测试集 50 条，其中 34 条已去污染。

**📈 对比分析**

将模型分别在污染样本、清洁样本和全部样本上评估，发现污染样本 BLEU 提升 29–47 分；去污染后 BLEU 降至 30.9–39.2 分，显示出显著的污染影响。

**⚠️ 局限性**

主要局限包括：测试集规模小导致置信区间宽、软泄漏难以细分、缺乏人工评估、长度差异可能进一步放大 BLEU 差异。

---

## 461. Incentivizing User Data Contributions for LLM Improvement under Withdrawal Rights

**arXiv ID:** 2605.07419 | [PDF](https://arxiv.org/pdf/2605.07419v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 462. UMVUE-Type Estimators under Bregman Losses

**arXiv ID:** 2605.07426 | [PDF](https://arxiv.org/pdf/2605.07426v1)

**作者:** Akira Kamatsuka `[一作]` (Shonan Institute of Technology), Shun Watanabe `[通讯]` (Tokyo University of Agriculture and Technology)

**通讯引用:** 3190 | [OpenAlex ID](https://openalex.org/A5101971899)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了在Bregman损失下的无偏估计，提出了Bregman UMVUE的理论框架并证明了其Rao-Blackwell与Lehmann–Scheffé定理的对应形式；

**💡 创新点**

创新点在于将无偏估计概念推广到Bregman散度，提出两种类型的无偏性（type-I与type-II），其中type-I通过对数变换映射到双线性空间实现全新最小方差无偏估计；

**🔧 技术方法**

主要技术包括Bregman散度的偏差-方差分解、Legendre型映射的双线性空间表示、以及在双线性空间上的Rao-Blackwell化与Lehmann–Scheffé化；

**📊 数据集**

文章使用了理论示例（指数分布与对数正态分布）来构造并展示type-I Bregman UMVUE，并未采用实测数据集；

**📈 对比分析**

通过与经典UMVUE的对比，证明在Bregman风险下type-I Bregman UMVUE在理论上实现了最小方差，但在数值例子中与传统UMVUE估计量不相同，缺乏进一步的性能数值评估；

**⚠️ 局限性**

局限性包括仅针对标量参数、对φ需满足严格凸且可逆映射的假设、未探讨多维参数情形、以及缺乏实证实验验证其实际效能。

---

## 463. Risk-Consistent Multiclass Learning from Random Label-Subset Membership Queries

**arXiv ID:** 2605.07413 | [PDF](https://arxiv.org/pdf/2605.07413v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 464. Effective and Memory-Efficient Alternatives to ECC for Reliable Large-Scale DNNs

**arXiv ID:** 2605.07417 | [PDF](https://arxiv.org/pdf/2605.07417v1)

**作者:** Mohammad Hasan Ahmadilivani `[一作]` (Tallinn University of Technology), Jaan Raik `[通讯]` (Tallinn University of Technology)

**通讯引用:** 1737 | [OpenAlex ID](https://openalex.org/A5010286547)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出两种零空间错误检测纠正方法（MSET和CEP），用于在大规模CNN和ViT参数中提高对内存软错误的鲁棒性，并通过实验验证其性能。

**💡 创新点**

1) MSET利用指数最高位复制到尾部LSB实现轻量化保护；2) CEP将每个参数按3位块嵌入奇偶校验，实现全参数细粒度保护；两者均无存储开销且硬件成本低于SECDED。

**🔧 技术方法**

零空间编码、奇偶校验嵌入、硬件解码器实现、随机位翻转软错误注入、Rust/PyTorch实现FaultForge工具、VHDL合成。

**📊 数据集**

ImageNet‑1K预训练的ResNet‑152、MobileNet‑V2、Inception、ViT‑base、DeiT‑base、Swin‑Tiny等。

**📈 对比分析**

与传统SECDED ECC在64/128位内存行对比；在不同BER下通过模型准确率评估；结果显示MSET/CEP在大多数模型下均超过SECDED，尤其CEP在3×10^-5到10^-4 BER下保持准确率>90%；硬件面积和延迟分别比SECDED低3.5×、7×。

**⚠️ 局限性**

对极高BER时MSET保护不足，需与ECC组合；CEP需要在硬件解码器中实现块级校验；部分CNN如MobileNet‑V2在FP16下精度下降；未针对FP8等新格式；未结合训练过程验证。

---

## 465. Emergent Symbolic Structure in Health Foundation Models: Extraction, Alignment, and Cross-Modal Transfer

**arXiv ID:** 2605.07407 | [PDF](https://arxiv.org/pdf/2605.07407v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 466. GPO-V: Jailbreak Diffusion Vision Language Model by Global Probability Optimization

**arXiv ID:** 2605.07399 | [PDF](https://arxiv.org/pdf/2605.07399v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 467. Have Graph -- Will Lift? The Case for Higher-Order Benchmarks

**arXiv ID:** 2605.07397 | [PDF](https://arxiv.org/pdf/2605.07397v1)

**作者:** Bastian Rieck `[一作]` (University of Fribourg), Bastian Rieck `[通讯]` (University of Fribourg)

**通讯引用:** 2944 | [OpenAlex ID](https://openalex.org/A5003031729)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文讨论了几何深度学习与拓扑深度学习中上阶结构的提升策略，指出现有数据集缺失与提升方式的局限，提出并构建了MANTRA高阶三角化数据集，并对其进行了初步实验评估；

**💡 创新点**

创新点包括：①提出无坐标、无特征的高阶数据集MANTRA，填补了上阶数据集空白；②对提升（lifting）策略进行批判性审视并建议多宇宙分析；③提出数据治理、任务设计与版本化等完整的数据集建设准则；

**🔧 技术方法**

采用了高阶消息传递（GNN/TDL）和图变压器等现有GDL/TDL模型对MANTRA进行实验，同时利用抽象simplicial complex表示与同调特征计算；

**📊 数据集**

使用MANTRA数据集（约43k 2D、250k 3D三角化），与传统图数据集（如GraphBench）进行对照；

**📈 对比分析**

通过对比仅使用图表示的GDL模型与可利用高阶消息传递的TDL模型，评估它们在预测Betti数、扭转系数、可定向性等任务上的性能；实验显示两者均表现不佳，TDL略优，表明当前模型仅做局部聚合，未能捕捉全局拓扑；

**⚠️ 局限性**

局限性包括：数据集缺乏节点/边特征或坐标，任务可由传统同调算法轻松完成；实验规模有限，缺乏多任务/大规模评估；提升策略缺乏可学习性；模型表达能力不足，难以真正利用拓扑信息。

---

## 468. Rubric-based On-policy Distillation

**arXiv ID:** 2605.07396 | [PDF](https://arxiv.org/pdf/2605.07396v1)

**作者:** Junfeng Fang `[一作]` (National University of Singapore), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 61908 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于结构化评卷标准（rubric）的On‑Policy Distillation框架（ROPD），通过教师和学生产出的回答生成任务特定的评卷标准，并用其对学生输出打分，作为强化学习的奖励；

**💡 创新点**

创新点在于将传统OPD中对教师logits的依赖替换为仅利用教师文本输出生成的语义评卷标准，从而实现黑盒教师、跨模型、跨架构的高效蒸馏，且样本效率提升高达10倍；

**🔧 技术方法**

使用技术包括：Rubricator（对教师和学生回答对比生成评卷标准）、Verifier（对学生回答进行二进制判定并加权）、多教师采样、多轮Rubric共享、盲判定、基于GRPO的策略优化；

**📊 数据集**

实验数据集包括：数学类的DAPO‑Math‑17K、AIME 24/25、HMMT 25、GPQA‑Diamond；医学/科学类的RaR‑Science/Medical‑20K；通用对齐测试的HealthBench、IFEval；教师模型涵盖GPT‑5.2‑chat、Qwen3‑30B‑A3B、Qwen3‑30B、Qwen3‑4B、Gemma3‑4B等；

**📈 对比分析**

与多种基线对比（SFT、T‑Judge、OVD、GAD、LOPD、ExOPD），ROPD在黑盒和白盒两种设置下均位列榜首，平均提升约10%–20%，在最难的HMMT‑25（Nov.）中从7.08%跃升至41.67%，样本效率提升近9.6×，训练时间相对比传统logit‑OPD快约6.3倍；

**⚠️ 局限性**

局限性主要体现在：1）评估主要聚焦于形式化推理任务，对主观或创意任务的效果未知；2）对Rubricator与Verifier的指令遵循能力有一定依赖，需在更广泛的模型架构上进一步验证。

---

## 469. Offline Policy Optimization with Posterior Sampling

**arXiv ID:** 2605.07393 | [PDF](https://arxiv.org/pdf/2605.07393v1)

**作者:** Hongqiang Lin `[一作]` (Zhejiang University), Haijun Zhang `[通讯]` (University of Science and Technology Beijing)

**通讯引用:** 27059 | [OpenAlex ID](https://openalex.org/A5100458465)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种无惰性模型基于贝叶斯后验采样的策略优化框架PSPO，能够在离线强化学习中同时实现对 OOD 区域的泛化与对模型利用的鲁棒性；

**💡 创新点**

创新点在于将转移动力学视为随机变量，构建基于经验一致性度量的可变后验分布并进行后验采样，从而无需惰性惩罚；并通过 KL 正则化约束策略更新保证学习稳定；

**🔧 技术方法**

采用贝叶斯强化学习、后验采样、KKT 转化的受限策略优化、神经网络 Q 值与策略网络、目标网络、经验回放、模型集成等技术；

**📊 数据集**

使用 D4RL 基准（HalfCheetah、Hopper、Walker2d 等）以及离线最优清算（offline optimal liquidation）环境；

**📈 对比分析**

与 CQL、DMG、EPQ、MOReL、RAMBO、PMDB、ADM 等 SoTA 方法对比，PSPO 在 14 个数据集上优于或与最佳方法持平，平均提升约 10%（p<0.05），并在高不确定性任务中表现尤为突出；

**⚠️ 局限性**

局限性包括：需要模型集成近似后验，参数 β、α、ε 的选择对性能影响较大；对连续动作空间的大规模环境可能存在计算瓶颈；对离线数据质量依赖较强，若数据分布过度偏离真实环境，后验采样可能失效。

---

## 470. EditRefiner: A Human-Aligned Agentic Framework for Image Editing Refinement

**arXiv ID:** 2605.07457 | [PDF](https://arxiv.org/pdf/2605.07457v1)

**作者:** Zitong Xu `[一作]` (Shanghai Jiao Tong University), Guangtao Zhai `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 22200 | [OpenAlex ID](https://openalex.org/A5064168853)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个多代理框架 EditRefiner，并基于此构建了大规模细粒度人类反馈数据集 EditFHF-15K，旨在实现文本引导图像编辑的自动细粒度自我纠错。

**💡 创新点**

创新点在于（1）首次公开 15K 张编辑图像的细粒度缺陷标注与文本推理，提供了人类感知对齐的评估基准；（2）将感知-推理-动作-评估循环引入图像编辑，形成可解释、可迭代的自我纠错机制；（3）通过 SFT+GRPO 的双阶段训练实现了与人类偏好高度一致的推理与评估。

**🔧 技术方法**

核心技术包括基于 Qwen3‑VL‑8B 的 VLM 作为骨干，使用 LoRA 微调多代理模型；Perception Agent 用 saliency 解码器生成缺陷热力图；Reasoning Agent 采用结构化响应 + 强化学习的 GRPO 训练以获得人类对齐的推理；Action Agent 通过生成局部重编辑指令与图像编辑模型交互；Evaluation Agent 使用多维评分头评估感知质量、指令遵循与视觉一致性。

**📊 数据集**

使用了 EditFHF-15K 数据集（约 15K 张编辑图像，60K+80K 区域标注，45K MOS），并在 EditFHF-15K、GEdit‑Bench、I2I‑Bench、KRIS‑Bench 等公开基准上进行评测。

**📈 对比分析**

在所有基准上，EditRefiner 在 PQ、IF、VC 等指标均优于最新 SOTA，平均提升约 6–8%（整体评分提升 5–7%），并在细粒度缺陷定位和人类评估一致性方面实现了显著优势。

**⚠️ 局限性**

主要局限包括：仍需要多轮迭代，迭代次数越多会产生累计误差；对极端复杂场景（如大范围结构变形）鲁棒性不足；对 VLM 的算力依赖较高；以及数据集构建成本高、标注规模有限。

---

## 471. RcLLM: Accelerating Generative Recommendation via Beyond-Prefix KV Caching

**arXiv ID:** 2605.07443 | [PDF](https://arxiv.org/pdf/2605.07443v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 472. Tracking Large-scale Shared Bikes with Inertial Motion Learning in GNSS Blocked Environments

**arXiv ID:** 2605.07412 | [PDF](https://arxiv.org/pdf/2605.07412v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 473. Boosting Automatic Java-to-Cangjie Translation with Multi-Stage LLM Training and Error Repair

**arXiv ID:** 2605.07403 | [PDF](https://arxiv.org/pdf/2605.07403v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 474. Forensic analysis of video data deletion and recovery in Honeywell surveillance file system

**arXiv ID:** 2605.07430 | [PDF](https://arxiv.org/pdf/2605.07430v1)

**作者:** Jinhee Yoon `[一作]` (Sungkyunkwan University), Sungjae Hwang `[通讯]` (Sungkyunkwan University)

**通讯引用:** 927 | [OpenAlex ID](https://openalex.org/A5019449591)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

分析Honeywell NVR专有文件系统结构，研究其三种删除机制（格式化、过期、覆盖），并验证在不同删除方式下视频数据的可恢复性。

**💡 创新点**

首次公开Honeywell NVR的文件系统内部布局与删除行为，证明即使元数据被清零，视频数据仍可直接从视频数据区提取；采用二进制差分技术揭示隐藏的存储结构。

**🔧 技术方法**

采用字节级二进制差分、低层文件系统逆向、H.264 NAL 单元解析以及自定义视频头解析与提取技术。

**📊 数据集**

使用单台Honeywell NVR的磁盘镜像（约45小时录像），并在不同删除操作后生成镜像（#18、#19、#20、#21）进行实验。

**📈 对比分析**

对三种删除方式分别进行恢复实验，结果显示：格式化后需及时恢复，新录像覆盖已删除区；过期与覆盖方式下恢复率较高。通过视频播放验证恢复质量，但未给出定量性能指标。

**⚠️ 局限性**

仅验证单一Honeywell型号，缺乏跨型号通用性；未做量化评估；恢复效果受录像时长、覆盖情况和固件差异影响。

---

## 475. Prompt Engineering Strategies for LLM-based Qualitative Coding of Psychological Safety in Software Engineering Communities: A Controlled Empirical Study

**arXiv ID:** 2605.07422 | [PDF](https://arxiv.org/pdf/2605.07422v1)

**作者:** Moaath Alshaikh `[一作]` (Federal University of Bahia), Manoel Mendonca `[通讯]` (Federal University of Bahia)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了三种大型语言模型（Claude Haiku、DeepSeek-Chat、Gemini 2.5 Flash）在软件工程社区心理安全话语的定性编码任务中的表现，比较了零样本与多样本提示的效果，并探讨了模型输出的稳定性与偏差；

**💡 创新点**

提出了基于多次重复运行评估LLM编码可靠性的实证框架，并揭示了模型对不同类别的系统性偏差与提示方式的差异性影响，为LLM辅助定性编码提供了可操作的提示与评估准则；

**🔧 技术方法**

使用了大型语言模型API、Cohen’s κ与F1得分评估指标、Wilcoxon符号秩检验与Levene方差检验等统计方法；

**📊 数据集**

使用了由Santana 等人整理的116条来自Stack Exchange项目管理与软件工程社区的心理安全相关语料，包含七个Edmondson框架下的行为标签；

**📈 对比分析**

实验采用每个模型-提示组合10次独立运行，计算平均κ与标准差，并与零样本提示进行Wilcoxon比较；结果显示Claude Haiku在多样本提示下显著提升κ（+0.034，p=0.004），Gemini 2.5 Flash亦呈正向趋势，DeepSeek‑Chat无显著变化；在稳定性方面Gemini 2.5 Flash表现最不稳定；在类别偏差上所有模型均过度预测“Sharing Negative Feedback”，欠预测“Expressing Concerns”；

**⚠️ 局限性**

局限性包括数据集规模有限且来自单一社区，类别分布不平衡导致少数类评估不稳；提示设计可能对模型产生隐式偏差；缺乏人类-人类一致性基准；实验仅覆盖零样本与多样本提示，未探讨更高级提示策略（如链式思考、对比提示）等。

---

## 476. Sparse Autoencoders as Plug-and-Play Firewalls for Adversarial Attack Detection in VLMs

**arXiv ID:** 2605.07447 | [PDF](https://arxiv.org/pdf/2605.07447v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 477. Accelerated and data-efficient flow prediction in stirred tanks via physics-informed learning

**arXiv ID:** 2605.07444 | [PDF](https://arxiv.org/pdf/2605.07444v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 478. Towards Photorealistic and Efficient Bokeh Rendering via Diffusion Framework

**arXiv ID:** 2605.07429 | [PDF](https://arxiv.org/pdf/2605.07429v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 479. ChartREG++: Towards Benchmarking and Improving Chart Referring Expression Grounding under Diverse referring clues and Multi-Target Referring

**arXiv ID:** 2605.07415 | [PDF](https://arxiv.org/pdf/2605.07415v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 480. Exposing and Mitigating Temporal Attack in Deepfake Video Detection

**arXiv ID:** 2605.07398 | [PDF](https://arxiv.org/pdf/2605.07398v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 481. BalCapRL: A Balanced Framework for RL-Based MLLM Image Captioning

**arXiv ID:** 2605.07394 | [PDF](https://arxiv.org/pdf/2605.07394v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 482. Unsolvability Ceiling in Multi-LLM Routing: An Empirical Study of Evaluation Artifacts

**arXiv ID:** 2605.07395 | [PDF](https://arxiv.org/pdf/2605.07395v1)

**作者:** Saloni Garg `[一作]` (Independent Researcher), Amit Sagtani `[通讯]` (San Francisco State University)

**通讯引用:** 8 | [OpenAlex ID](https://openalex.org/A5114944587)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估了多层LLM路由的真实可用性，剖析了报告的未解率背后的评估误差。

**💡 创新点**

提出三元评估误差分解框架、双评审验证和CNA度量，揭示了路由可用性被高估的根本原因。

**🔧 技术方法**

使用LLM‑as‑judge、精确匹配对照、随机特征/标签洗牌控制、成本敏感目标以及多模型推理。

**📊 数据集**

六大基准（MMLU、MedQA、HumanEval、MBPP、Alpaca、ShareGPT）和Gemma 4、Llama 3.1两大模型族。

**📈 对比分析**

与单一评判测评对比，发现误差导致13–17pp的机遇成本；路由器在标签不平衡下崩溃，改进后可提升路由性能。

**⚠️ 局限性**

仅在Gemma4、禁用思考模式、4K上下文、评审模型与被评模型重叠以及单一模型家族环境下实验，需进一步验证。

---

## 483. StreamPhy: Streaming Inference of High-Dimensional Physical Dynamics via State Space Models

**arXiv ID:** 2605.07384 | [PDF](https://arxiv.org/pdf/2605.07384v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 484. A Marine Debris Detection Framework for Ocean Robots via Self-Attention Enhancement and Feature Interaction Optimization

**arXiv ID:** 2605.07388 | [PDF](https://arxiv.org/pdf/2605.07388v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 485. MERBIT: A GPU-Based SpMV Method for Iterative Workloads

**arXiv ID:** 2605.07391 | [PDF](https://arxiv.org/pdf/2605.07391v1)

**作者:** Qi Zhang `[一作]` (Sun Yat-sen University), Zan-Bo Zhang `[通讯]` (Guangdong University of Finance and Economics)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对不规则稀疏矩阵的 GPU SpMV 加速方法 MERBIT，专门为 PageRank 等需要重复 SpMV 的迭代工作负载设计。

**💡 创新点**

创新点在于将 merge‑path 分区与 CSR5 的紧凑位字段编码相结合，先一次性生成可重用的调度元数据，并在内核中融合长行处理、warp 级分段归约与双缓冲写入等多种优化，显著减少了调度与边界处理的重复开销。

**🔧 技术方法**

使用的技术包括 GPU merge‑path 分区、32 位紧凑位字段描述符、共享内存中矩阵值的阶段化加载、warp 级分段归约、双缓冲输出提交、一次性预处理以及对长行的特殊 fast‑tackle 路径。

**📊 数据集**

评测基于 50 个 SuiteSparse 集合中大规模（>10M 非零）不规则稀疏矩阵，并将 MERBIT 应用于 PageRank 与 BiCGSTAB 迭代工作负载进行端到端测试。

**📈 对比分析**

与 cuSPARSE CSR/COO、Ginkgo、Merge‑Based SpMV、HOLA、CSR5 等八个基线进行比较，单次 SpMV 在单精度上平均提升 1.27×、双精度提升 1.25×，在 50 个数据集上最高吞吐率分别达到 36/35 个，速度比 cuSPARSE COO 约 1.27×‑1.28×；在迭代工作负载中仍保持 1.15‑1.18× 的加速。

**⚠️ 局限性**

主要局限在于仍受限于内存访问瓶颈，尤其是输入向量的非连续采样；对平均度较高的矩阵加速效果减弱；以及预处理开销在极大规模矩阵上仍不可忽视。

---

## 486. Convex Optimization with Nested Evolving Feasible Sets

**arXiv ID:** 2605.07386 | [PDF](https://arxiv.org/pdf/2605.07386v1)

**作者:** Karthick Krishna M. `[一作]`, Rahul Vaze `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并研究了 CONES（Convex Optimization with Nested Evolving Feasible Sets）框架，解决在可行域随时间递归收缩时的在线凸优化问题，目标同时最小化 regret 与移动成本。

**💡 创新点**

创新点包括：1）定义并分析了与传统 NCBC 相比的更通用 CONES；2）证明了强凸或 α‑sharp 损失函数下，Frugal 算法可实现零 regret 与 O(log T) 的最优移动成本；3）给出了强凸函数下贪心策略的下界与上界；4）为一般凸函数设计了可调节 regret‑移动成本权衡的 LSP 与 GAP‑Frugal 算法；5）在动态约束下的 1‑维情形中提出 5‑competitive 算法，并给出 α‑sharp 情形的 12/α‑competitive 结果。

**🔧 技术方法**

主要技术手段包括：投影与“懒惰”更新（lazy projection）；阶段化分析与几何递归；利用强凸/α‑sharp 性质建立距离-成本关系；构造对抗性可行集序列来证明下界；把动态约束问题映射到 NCBC 并使用工作函数概念；以及对竞争比的分析。

**📊 数据集**

论文未使用任何公开数据集，全部结果均为理论分析与算法证明。

**📈 对比分析**

对比方法：与贪心策略（零 regret但移动成本为 Θ(√T)）相比，Frugal 在强凸/α‑sharp 情形下实现了 O(log T) 的移动成本；在一般凸情形下，LSP 与 GAP‑Frugal 能实现 (T^{1-β},T^β) 或 (O(1),O(log²T)) 的 regret‑移动成本权衡。对于动态约束，1‑维 5‑competitive 与 α‑sharp 的 12/α‑competitive 结果与现有 5‑或 12/α‑竞争算法相比具有竞争力。

**⚠️ 局限性**

局限性：1）对于一般凸函数，仍未实现零 regret 与 O(log T) 移动成本的最优解；2）Frugal 的维度相关常数尚未完全确定；3）动态基准（可变动作序列）的分析与算法尚未完成；4）在高维动态约束下的竞争比和移动成本分析仍是未解决的开放问题。

---

## 487. Can LLMs Solve Science or Just Write Code? Evaluating Quantum Solver Generation

**arXiv ID:** 2605.07525 | [PDF](https://arxiv.org/pdf/2605.07525v1)

**作者:** Luciano Baresi `[一作]` (Politecnico di Milano), Seung Yeob Shin `[通讯]` (University of Luxembourg)

**通讯引用:** 208 | [OpenAlex ID](https://openalex.org/A5088550214)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文评估了大语言模型（LLM）在自动生成量子求解器代码方面的可行性，提出了一种生成-执行-验证的迭代方法。

**💡 创新点**

创新点在于将LLM生成的量子程序与可信的经典求解器进行数值验证的循环，系统量化了迭代反馈对成功率、错误类型和执行时间的影响。

**🔧 技术方法**

技术实现包括：LLM代码生成、容器化执行、数值结果与经典求解器比较、错误分类规则、以及统计检验（Mann‑Whitney U 与 Vargha‑Delaney 效应量）。

**📊 数据集**

数据集为五类科学问题（费米‑霍布德、TFIM、MaxCut、Schwinger、氢分子）各四个实例，10 次重复，10 轮迭代；使用公开与专有的多种 LLM。

**📈 对比分析**

比较方法：将单轮基线与迭代版在成功率、时间开销、错误分布等维度进行对比；结果显示迭代显著提升成功率但引入明显时间成本，强模型（如 GPT‑5.2）表现最佳。

**⚠️ 局限性**

局限包括：LLM 仍易产生语法/API 错误或数值不准确；迭代收敛缓慢；缺乏领域知识指导；评估受限于可求解的实例范围和参考实现的可用性。

---

## 488. Instance and Universally Optimal Bounds for Imprecise Pareto Fronts

**arXiv ID:** 2605.07523 | [PDF](https://arxiv.org/pdf/2605.07523v1)

**作者:** Sarita de Berg `[一作]` (IT University of Copenhagen), Daniel Rutschmann `[通讯]` (Institute of Science and Technology Austria)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5021826063)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了在不确定几何模型下，利用预处理后的区域信息高效构造点集的 Pareto 前沿的算法，针对矩形和单位正方形实现了实例最优检索与几乎最优的运行时间。

**💡 创新点**

创新点在于：①设计了一种基于“依赖关系”的实例最优检索策略，证明其检索次数与任何算法相同；②通过层划分与主动层技术，将检索过程实现为每次 O(log n) 的操作；③对单位正方形实现了完全的普适最优（universal optimal）恢复程序，突破了此前只能针对圆盘的结果。

**🔧 技术方法**

主要技术包括：不确定几何模型、Pareto 前沿与外部/内部 Pareto 前沿的概念；依赖关系与独立区域判定；层划分、主动层、平衡二叉树与区间合并；正交范围查询、射线投射；网格化、单元格分类（A/B/C）；指针搜索树（finger tree）以及信息论下界证明。

**📊 数据集**

论文仅在理论模型下进行，未使用实际数据集；所有实验均基于合成点集与区域集合的理论分析。

**📈 对比分析**

与之前工作相比，矩形的实例最优检索实现了 O(r log n) 的时间，只有 O(log n) 的额外因子；单位正方形的算法在 O(n log n) 预处理后，恢复时间达到了普适最优，优于此前仅针对不重叠或圆盘情况的算法。

**⚠️ 局限性**

局限性包括：仅处理轴对齐矩形与单位正方形；对更一般形状或高维问题尚未给出扩展；实现复杂度较高，实际部署可能受限；理论下界依赖于信息论假设，实际性能仍需实验验证。

---

## 489. Tessellations of Semi-Discrete Flow Matching

**arXiv ID:** 2605.07513 | [PDF](https://arxiv.org/pdf/2605.07513v1)

**作者:** Emile Pierret `[一作]` (ENS-PSL), Julie Delon `[通讯]` (ENS-PSL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过解析半离散Flow Matching的终点映射，研究其分割单元的拓扑与几何特性，证明单元是开且单连通、可在额外假设下同胚于单位球，但与最优传输的Laguerre细分显著不同，如非凸、无界、边界非仿射。

**💡 创新点**

创新点在于揭示Flow Matching的内在几何偏差：在理论上明确其终点分区既保留全局拓扑良好性，又呈现与OT截然不同的几何结构，且提供具体四点反例证明这些差异；同时证明了单元可收缩并给出中心构造。

**🔧 技术方法**

主要技术为闭式速度场推导、流图分析、拓扑学工具（开集覆盖、同胚性与可收缩性证明）、几何反例构造以及对Gaussian混合目标的极限分析。

**📊 数据集**

实验验证主要以MNIST数据在二维PCA切片上训练的简易神经FM模型为例，展示其产生的分区与理论闭式单元相近；其他实验中仅使用理论示例，无外部公开数据集。

**📈 对比分析**

与OT的比较采用Laguerre细分作为对照，分析单元边界、凸性、可界性、邻接关系等属性；实验显示FM单元在几何上往往更不规则、可变形，未给出数值性能指标。

**⚠️ 局限性**

局限性在于只研究了Exact半离散FM的闭式速度场，未涉及训练后近似模型的几何表现；此外分析仅适用于等权重、不同支点的情形，对非等权重或大规模点集的行为仍未知。

---

## 490. Implicit Multi-Camera System Calibration Using Gaussian Processes

**arXiv ID:** 2605.07491 | [PDF](https://arxiv.org/pdf/2605.07491v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 491. TCMIIES: A Browser-Based LLM-Powered Intelligent Information Extraction System for Academic Literature

**arXiv ID:** 2605.07507 | [PDF](https://arxiv.org/pdf/2605.07507v1)

**作者:** Hanqing Zhao `[一作]` (Hebei University), Hanqing Zhao `[通讯]` (Hebei University)

**通讯引用:** 16196 | [OpenAlex ID](https://openalex.org/A5078194334)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个基于浏览器、零安装、使用商用LLM API的传统中医信息智能提取系统（TCMIIES），实现了结构化知识抽取。

**💡 创新点**

创新点在于自动化的schema-guided prompting框架、无服务器前端架构、智能中文数据库字段映射以及多模型并发批处理引擎。

**🔧 技术方法**

使用Vue.js、SheetJS、Fetch API、OpenAI兼容接口，结合DeepSeek、OpenAI、Qwen、Zhipu AI等LLM模型，采用JSON schema提示与正则解析。

**📊 数据集**

利用从CNKI导出的500篇中医论文（涵盖草药、针灸、方剂、证候、整合医学等子领域）进行评测。

**📈 对比分析**

通过与专家标注对比，结构化输出合规率超过94%，字段抽取平均准确率约81.6%，与现有基准相比性能相当或略优，且成本低。

**⚠️ 局限性**

局限包括依赖商用API、JSON解析失败率5–8%，对全文处理支持不足、上下文窗口限制以及高精度场景仍需人工校验。

---

## 492. AudioFace: Language-Assisted Speech-Driven Facial Animation with Multimodal Language Models

**arXiv ID:** 2605.07478 | [PDF](https://arxiv.org/pdf/2605.07478v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 493. Cross-Modal Backdoors in Multimodal Large Language Models

**arXiv ID:** 2605.07490 | [PDF](https://arxiv.org/pdf/2605.07490v1)

**作者:** Runhe Wang `[一作]` (Southeast University), Songze Li `[通讯]` (Southeast University)

**通讯引用:** 2780 | [OpenAlex ID](https://openalex.org/A5085853632)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在多模态大型语言模型（MLLM）中，针对轻量级连接器（connector）实施跨模态后门攻击；只需用单一模态的少量样本进行连接器污染，即可在其他模态输入上触发预设目标输出；通过输入空间对齐与优化实现跨模态激活。

**💡 创新点**

创新点在于发现并利用连接器作为高杠杆攻击面以及共享潜在空间的可转移性，使得后门可从未被污染的模态触发；提出三阶段攻防流程：连接器污染、恶意质心提取、跨模态对抗激活；并验证此攻击在多模态系统中的轻量级与隐蔽性。

**🔧 技术方法**

技术包括：仅更新连接器的微调、基于语言模型输出的加权交叉熵损失、特征蒸馏与参数漂移正则化以保持对齐；利用对抗优化（PGD）在输入空间最大化与恶意质心的余弦相似度并最小化欧氏距离；计算恶意质心时分离方向与幅度并取均值。

**📊 数据集**

使用公开数据集：MS COCO（图像+文本）、Clotho（音频）等；针对ImageBind、CLIP等预训练编码器；对PandaGPT与NExT-GPT两种典型连接器结构进行实验。

**📈 对比分析**

相较于经典单模态后门（BadNets、Blended、TrojVLM）以及基于输入空间的对抗方法（PGD、UAP），该方法在本地仅污染连接器后，跨模态ASR可达95%以上（如Image→Audio、Audio→Image均>95%），且在单模态激活时接近100%；保持模型实用性，后门泄漏率为0%；参数漂移低于0.1，隐蔽性高。

**⚠️ 局限性**

局限性包括：依赖于共享潜在空间的对齐质量；对不同连接器实现（如查询式、交叉注意力式）尚未充分验证；对抗激活需要较高的预算与优化步骤；在强防御（如连接器微调或裁剪、输入预处理）下仍能在一定程度恢复性能。

---

## 494. Does Your Neural Network Extrapolate? Feature Engineering as Identifiability Bias for OOD Generalization

**arXiv ID:** 2605.07483 | [PDF](https://arxiv.org/pdf/2605.07483v1)

**作者:** Leonel Aguilar `[一作]` (ETH Zürich), Nino Antulov-Fantulin `[通讯]` (Aisot Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了在单窗口训练下深度神经网络如何进行外域（OOD）泛化，并提出通过特征映射、标签映射与模型类的结构承诺（φ, ψ, ℳ）来实现可识别的外域推断。

**💡 创新点**

创新点在于将观测等价性与 DGP 承诺框架化，证明单窗口数据无法识别 OOD 结构，仅靠结构承诺即可实现外域可辨识；并提出近边界验证、SINDy δ_OOD 等可操作的诊断方法。

**🔧 技术方法**

使用的技术包括 Fourier 变换、对数变换、稀疏识别（SINDy）、近边界验证、不同位置嵌入（sinusoidal、RoPE、Fourier PE）等特征工程手段；模型包括 5 层 256 Tanh 单层感知机、Transformer、Mamba、S4D、TabPFN、TimesFM 等。

**📊 数据集**

实验数据集涵盖：sin(x) toy、质量作用化学反应（MAK）、Kepler 第三定律（NASA Exoplanet Archive）、跨物种编码 DNA（5 种生物）以及 264 次 Transformer/Mamba/S4D 的位置编码实验。

**📈 对比分析**

对比相同 ID 损失但不同 φ 的模型，发现 OOD 性能差异可达 520 倍；在真实数据中，正确的 φ + 适配模型将 OOD MAPE/误差从数十% 降到几 % 或 AUROC 接近 1，而错误组合导致误差可达 100%+ 或低于 0.5 的 AUROC。

**⚠️ 局限性**

局限性包括：单窗口训练无法通过数据识别 OOD 结构，需要预先承诺；近边界验证需可获取边界样本；SINDy 诊断仅适用于可稀疏的动态；在非多项式或高噪声环境下效果下降；并且即使 φ 正确，仍需足够的模型容量与样本覆盖才能实现外域泛化。

---

## 495. SHRED: Retain-Set-Free Unlearning via Self-Distillation with Logit Demotion

**arXiv ID:** 2605.07482 | [PDF](https://arxiv.org/pdf/2605.07482v1)

**作者:** Zizhao Hu `[一作]` (University of Southern California), Robin Jia `[通讯]` (University of Southern California)

**通讯引用:** 6702 | [OpenAlex ID](https://openalex.org/A5041906762)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无retain集的大语言模型机器遗忘方法，通过自蒸馏与选择性logit降权来实现对记忆内容的去除。

**💡 创新点**

创新点在于：①只对低概率（高信息量）token进行降权，②利用剩余token自我提供的“隐式retain”信号，无需额外retain数据，③采用top‑K KL自蒸馏目标实现一次性训练。

**🔧 技术方法**

核心技术包括：自蒸馏、logit降权（variant A/B）、top‑K KL损失、前向概率筛选与低信息量位置保留。

**📊 数据集**

实验使用了四个公开遗忘基准：TOFU、MUSE、RWKU 与 Hubble（对应的 Llama 3.2 1B/7B、Llama 2 7B/8B 等模型）。

**📈 对比分析**

与 GA、GradDiff、NPO、SimNPO、DPO、RMU、WHP、TaskVec、CEU 等传统方法对比， 在所有四个基准上实现了Pareto最优的遗忘‑效能权衡，且在隐私（MIA）、再学习与持续遗忘场景下表现稳健。

**⚠️ 局限性**

局限性包括：需手动调参（P、π），需要评估探针与遗忘集共享信息；对全域知识移除的适用性有限；在低秩适配（LoRA）下会略微影响效能。

---

## 496. ReasonEdit: Towards Interpretable Image Editing Evaluation via Reinforcement Learning

**arXiv ID:** 2605.07477 | [PDF](https://arxiv.org/pdf/2605.07477v1)

**作者:** Honghua Chen `[一作]` (University of Electronic Science and Technology of China), Guangtao Zhai `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 22200 | [OpenAlex ID](https://openalex.org/A5064168853)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 ReasonEdit-22K 数据集，设计 RE-Reward 奖励模型并训练 ReasonEdit 解释性评估模型，实现对文本引导图像编辑的可解释性评估与生成；

**💡 创新点**

首创结合大规模人类评判、链式推理（CoT）与多模态奖励学习的评估框架，提供逻辑性、准确性和实用性三维细粒度反馈；

**🔧 技术方法**

使用多模态 LLM（如 Qwen‑3.5‑9B）作为 backbone，结合 AdaLoRA/QLoRA 微调，奖励模型采用 Huber+ranking+PLCC 损失，RL 过程采用 Group Relative Policy Optimization (GRPO)；

**📊 数据集**

主要数据集为 ReasonEdit‑22K（22K 编辑图像 + 113K CoT + 1.3M 人工评分），并在 EBench‑18K 与多项公开零样本评测基准上验证；

**📈 对比分析**

与传统 IQA、VQA 及多种 LLM 基线比较，ReasonEdit 在 EBench‑18K 上 SRCC/PLCC 均达到最高水平；在 GenAI‑Bench、AROURA‑Bench、EditScore‑Bench、ImagenHub、IEQA 等零样本基准上亦实现最高或接近最高分，性能提升幅度从 1.5% 到 8% 以上；

**⚠️ 局限性**

仍依赖大规模预训练 LLM 与昂贵的奖励/RL 训练，适配度在极端新任务或高度专业化编辑场景可能受限，且评估结果仍需人工核对以进一步提升可靠性。

---

## 497. NPMixer: Hierarchical Neighboring Patch Mixing for Time Series Forecasting

**arXiv ID:** 2605.07476 | [PDF](https://arxiv.org/pdf/2605.07476v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 498. A Unified Framework for the Detection and Classification of Fatty Pancreas in Ultrasound Images

**arXiv ID:** 2605.07466 | [PDF](https://arxiv.org/pdf/2605.07466v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 499. Approximation Error Upper and Lower Bounds for Hölder Class with Transformers

**arXiv ID:** 2605.07463 | [PDF](https://arxiv.org/pdf/2605.07463v1)

**作者:** Xin He `[一作]` (Wuhan University), Jerry Zhijian Yang `[通讯]` (Wuhan University)

**通讯引用:** 22717 | [OpenAlex ID](https://openalex.org/A5100404947)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文研究标准Transformer（含Softmax、ReLU与残差连接）在逼近Hölder类函数时的表达能力，给出了逼近误差的上界与下界，并将结果应用于回归任务的泛化分析。

**💡 创新点**

创新点在于：①首次在不改造模型的前提下给出Hölder类的逼近上界（O(ε⁻ᵈ₀/α)层）；②利用VC‑dimension上界首次给出Transformer的逼近下界（Ω(ε⁻ᵈ₀/(4α))层）；③提出完整的量化框架，包括量化、上下文映射与值映射的三模块构造；④将理论结果推广到有限样本回归，给出超参数与样本量的最优平衡。

**🔧 技术方法**

主要技术包括：逼近理论（分层构造、量化与上下文映射）、VC‑dimension 计算与下界推导、统一误差分解与覆盖数估计、以及对Hölder函数的网格逼近。

**📊 数据集**

该工作为理论分析，不使用任何实际数据集。

**📈 对比分析**

与以往的理论研究相比，本文提供了更细致的误差上界与下界，揭示了Transformer在逼近Hölder类函数时的容量上限；由于是理论性，未进行实验比较，性能指标以误差率与层数关系呈现。

**⚠️ 局限性**

主要局限包括：上界与下界之间仍存在阶数差距（上界为O(ε⁻ᵈ₀/α)，下界为Ω(ε⁻ᵈ₀/(4α))）；VC‑dimension上界较宽松（O(D⁴)），导致回归误差率非最优；对高光滑度（α>1）函数的逼近未充分利用；以及对多头自注意力层组合的潜在优势尚未深入探讨。

---

## 500. SEIF: Self-Evolving Reinforcement Learning for Instruction Following

**arXiv ID:** 2605.07465 | [PDF](https://arxiv.org/pdf/2605.07465v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 501. On the Complexity of Discounted Robust MDPs with $L_p$ Uncertainty Sets

**arXiv ID:** 2605.07459 | [PDF](https://arxiv.org/pdf/2605.07459v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

---

## 502. Model-Driven Policy Optimization in Differentiable Simulators via Stochastic Exploration

**arXiv ID:** 2605.07520 | [PDF](https://arxiv.org/pdf/2605.07520v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 503. InterLV-Search: Benchmarking Interleaved Multimodal Agentic Search

**arXiv ID:** 2605.07510 | [PDF](https://arxiv.org/pdf/2605.07510v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 504. Synchronizing Minds through Collective Predictive Coding: A Computational Model of Parent-Infant Homeostatic Co-Regulation

**arXiv ID:** 2605.07524 | [PDF](https://arxiv.org/pdf/2605.07524v1)

**作者:** Yushi Tsubamoto `[一作]` (University of Osaka), Takato Horii `[通讯]` (University of Osaka)

**通讯引用:** 449 | [OpenAlex ID](https://openalex.org/A5031377124)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一个基于 POMDP 的主动内感知推理模型，结合 Metropolis–Hastings Naming Game（MHNG）来模拟亲子协作调节婴儿生理状态的过程，并通过仿真验证了该模型在调节和内部表征对齐方面的有效性。

**💡 创新点**

创新点在于将 CPC（集体预测编码）假说与 MHNG 结合，提供了一种无需深度递归模型、仅靠本地可计算机制即可实现代理之间表征同步的方法；展示了内部表征同步可以早于生成模型的收敛，从而为 IBS（脑间同步）的计算基础提供了最小构造性解释。

**🔧 技术方法**

技术方法包括：POMDP 与主动推理框架、Metropolis–Hastings Naming Game、Dirichlet 分布在线学习、Jensen–Shannon 距离评估、以及对生成矩阵熵的统计分析。

**📊 数据集**

使用的是自行设计的 6×6 离散网格世界作为婴儿的生理状态空间（能量与体温），并在此空间内生成的合成观测数据进行仿真实验，没有使用公开实验数据集。

**📈 对比分析**

通过比较三种交互策略（A-only、B-only、MHNG）来评估模型性能。评价指标包括：C^norm（对偏好区域的调节适应度）、生成矩阵平均熵（学习进展）和 JSD（内部表征相似度）。结果显示，MHNG 条件在 C^norm 上表现最佳，并且内部表征同步（JSD）比单方控制更快、更稳健，生成矩阵熵也在相同迭代内收敛，说明 MHNG 提供了更高效的协作调节与同步机制。

**⚠️ 局限性**

局限性包括：使用的离散 2D 网格过于简化，未考虑连续、多维生理状态；亲子双方的感知通道信息量相等且均为离散，缺乏与真实生物/机器人相匹配的复杂信号；偏好矩阵 C 与解释矩阵 E 预设并保持静态，未模拟符号出现的动态过程；因此模型在现实场景中的泛化和可扩展性仍待进一步研究。

---

## 505. LARAG: Link-Aware Retrieval Strategy for RAG Systems in Hyperlinked Technical Documentation

**arXiv ID:** 2605.07517 | [PDF](https://arxiv.org/pdf/2605.07517v1)

**作者:** Giorgia Bolognesi `[一作]` (Rulex s.r.l.), Luca Oneto `[通讯]` (University of Genoa)

**通讯引用:** 8838 | [OpenAlex ID](https://openalex.org/A5045802198)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在技术文档中提出并实现了一种基于链接的检索增强生成（LARAG）方案，利用HTML内建的超链接结构对检索过程进行局部图形化扩展，从而提升答案的真实性与完整性。

**💡 创新点**

创新点在于：①不需要构建或维护显式图结构，而是直接将超链接信息编码为元数据并在检索时按深度优先方式展开；②在同一预处理管线下实现了与传统RAG的直接对比，验证了链接感知检索的有效性。

**🔧 技术方法**

使用的技术包括：大型语言模型（GPT‑4），向量检索（OpenAI embeddings + Chroma DB），LangChain框架，超链接元数据处理，BERTScore评估。

**📊 数据集**

使用的数据集为Rulex Platform技术手册的HTML版，构成20条专家级查询与其参考答案，另外使用了四种不同的提示模板。

**📈 对比分析**

通过在相同预处理条件下对比RAG（k=5/10）与LARAG（1,1,1）三种配置，评估指标为BERTScore F1、精确率、召回率、总token数、检索块数和响应时间。结果显示，LARAG在保持更少检索块和更低token消耗的同时，F1和精确率均超过传统RAG，并且在长篇查询中表现更为稳健。

**⚠️ 局限性**

局限性包括：仅适用于含有丰富超链接的技术文档，无法处理无链接或损坏链接的文本；未构建全局图结构，可能错过跨章节的隐式语义关联；对文档版本、过时内容的敏感度未得到充分验证；在非技术领域或多语言环境中的通用性待进一步验证。

---

## 506. Excluding the Target Domain Improves Extrapolation: Deconfounded Hierarchical Physics Constraints

**arXiv ID:** 2605.07485 | [PDF](https://arxiv.org/pdf/2605.07485v1)

**作者:** Tsuyoshi Okita `[一作]` (Kyushu Institute of Technology), Tsuyoshi Okita `[通讯]` (Kyushu Institute of Technology)

**通讯引用:** 2608 | [OpenAlex ID](https://openalex.org/A5026921207)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 Deconfounded Hierarchical Gate（DHG）与多级物理约束，提升物理约束生成模型在 OOD 条件下的泛化

**💡 创新点**

将 Pearl 的 do‑operator 与后门调节相结合，实现温度去混淆的门控机制，并采用粗到细的层级约束顺序；同时发现跨域预训练时排除目标域可显著提升外推性能

**🔧 技术方法**

使用 Fourier Neural Operator（FNO(1)）预训练捕获条件不变物理模式；条件流匹配（CFM）生成器；对抗式损失与回归损失相结合；DHG 采用 counterfactual 估计、backdoor 调整和 Sigmoid 门控

**📊 数据集**

主要使用 NASA 电池数据集（LiCoO₂/graphite），并在预训练阶段加入 MICH（NMC）和 SNL（LFP）数据以实现跨材料学习

**📈 对比分析**

相较于无约束 Pure CFM 基线，使用 DHG 与层级约束后 RMSE 下降至 0.215（比 0.397 改进 46%），并在多条件评估中保持更低的 FID 与更高的温度辨别率

**⚠️ 局限性**

计算开销约为基线的 1.5 倍；实验仅在 NASA 数据集上验证，缺乏在其他电池或物理系统上的通用性验证

---

## 507. Vaporizer: Breaking Watermarking Schemes for Large Language Model Outputs

**arXiv ID:** 2605.07481 | [PDF](https://arxiv.org/pdf/2605.07481v1)

**作者:** Jonathan Hong Jin Ng `[一作]` (Nanyang Technological University), Anupam Chattopadhyay `[通讯]` (Nanyang Technological University)

**通讯引用:** 6369 | [OpenAlex ID](https://openalex.org/A5089860351)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型（LLM）输出的水印技术在面对语义保持攻击时的鲁棒性，系统评估三种主流水印方案（Provable Robust Watermarking、Publicly Detectable Watermarking、SynthID）的抗攻击性能。

**💡 创新点**

提出一种完整的攻击与评估框架，融合词义替换、机器翻译与神经改写等多种语义保持攻击，并通过多维度质量指标（BERT分数、复杂度、可读性、语法错误）评判攻击对文本质量的影响，从而揭示水印设计的弱点与改进方向。

**🔧 技术方法**

使用了三种水印实现（概率分布控制、公开可验证签名、锦标赛采样），配合MarianMT翻译、BART和Pegasus改写模型进行攻击；评估指标包括Z-score/g-score、BERT F1、复杂度（Gunning Fog）、可读性（Flesch）、语法错误数等；实验平台为Python+PyTorch，GPU RTX 4070 Super。

**📊 数据集**

基于ChatGPT生成的100条多领域写作提示所产生的输出文本，作为水印生成与攻击的实验数据集。

**📈 对比分析**

通过对每种攻击方法计算成功率、平均Z-score/g-score、语义保留与质量变化，比较三种水印的鲁棒性。结果显示：Publicly Detectable Watermarking对所有攻击均100%易破；SynthID在所有攻击中最稳健，仅Pegasus改写能达到14%成功率；Provable Robust Watermarking最为抗性，只有Pegasus改写能达到23%成功率，其他攻击成功率低于5%。

**⚠️ 局限性**

实验样本量有限（仅100条提示），可能无法覆盖真实使用场景的文本多样性；评估聚焦于语义保持攻击，未探究对抗生成式文本或其他攻击方式；攻击效率虽短，但仍需在更大规模与多模型环境中验证结论。

---

## 508. PathPainter: Transferring the Generalization Ability of Image Generation Models to Embodied Navigation

**arXiv ID:** 2605.07496 | [PDF](https://arxiv.org/pdf/2605.07496v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 509. HBEE: Human Behavioral Entropy Engine -- Pre-Registered Multi-Agent LLM Simulation of Peer-Suspicion-Based Detection Inversion

**arXiv ID:** 2605.07472 | [PDF](https://arxiv.org/pdf/2605.07472v1)

**作者:** Vickson Ferrel `[一作]` `[通讯]` (Universiti Malaysia Sarawak), Vickson Ferrel (Universiti Malaysia Sarawak)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过预注册的 LLM 驱动多智能体模拟器 HBEE，对比同僚怀疑级联与 UEBA 两类检测在适应性内部威胁场景下的表现，发现检测逆转现象。

**💡 创新点**

首次在预注册框架内展示 LLM 适应性内部威胁可导致同僚怀疑图反向检测，并证明 UEBA 排名与同僚怀疑图可解耦，揭示传统检测机制在面临动态 OPSEC 指令时的脆弱性。

**🔧 技术方法**

使用 GLM‑4 Flash INT4 LLM 推理、vLLM 服务、Bootstrap 95% BCa 置信区间、Cliff’s Δ、Wilcoxon 同步检验、Mann‑Whitney U 等统计方法，对多条件实验结果进行严谨分析。

**📊 数据集**

以 HBEE 生成的 100 代理 100 tick（≈9,950 事件）模拟数据为主，利用 SNAP Enron 邮件网络的 Gini 系数作为校准参考。

**📈 对比分析**

在 5 个预注册条件下执行 100 次运行，使用 Bootstrap 置信区间和等价性检验，结果显示检测逆转显著（Cliff’s Δ = -0.694，p < 0.01），且 UEBA 排名未出现显著偏移（等价性区间 ±5 排名），说明两种检测指标在适应性内部威胁下解耦。

**⚠️ 局限性**

限制包括 HBEE 的通信结构过于均匀（与真实组织差距大）、仅使用单一 LLM 模型、单一 mole archetype、未考虑多内部威胁与协作、预注册依赖单一评分者的行为可信度审计等，导致结果的可转化性受限。

---

## 510. The Moltbook Files: A Harmless Slopocalypse or Humanity's Last Experiment

**arXiv ID:** 2605.07462 | [PDF](https://arxiv.org/pdf/2605.07462v1)

**作者:** William Brach `[一作]` (Slovak University of Technology), Lukas Galke Poech `[通讯]` (University of Southern Denmark)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5104621238)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

发布了 Moltbook Files 数据集（232k 帖子、2.2M 评论），并对其进行了 PII 匿名化、垃圾信息过滤及多维度分析；随后在 Qwen2.5‑14B‑Instruct 上进行不同程度的 fine‑tune，评估其对事实性与对齐性的影响。

**💡 创新点**

首次系统地去除 PII 并过滤垃圾内容，提供可复现的 AI 代理生成社交媒体数据；通过与规模匹配的 Reddit 数据集对比，揭示 agent‑generated 内容对模型性能的相对影响与潜在尾部风险。

**🔧 技术方法**

使用了 fastText 语言识别、Microsoft Presidio（自定义识别器）、BERTopic+Qwen3‑Embedding‑8B 进行主题建模、RoBERTa‑GoEmotions 做情感分析、LoRA 训练方案和 DeepSeek‑3.2 作为判定者。

**📊 数据集**

Moltbook Files（232k 帖子、2.2M 评论，覆盖 12 天）和规模匹配的 Reddit 训练样本（232k 条）作为对照。

**📈 对比分析**

通过 TruthfulQA‑MC1/MC2 衡量事实性、通过 DeepSeek‑3.2 评估对齐性与连贯性；结果显示，Moltbook fine‑tune 的事实性从 36.6% 降至 18.7%，对齐性也下降到 70‑80%，但相同规模的 Reddit fine‑tune 产生了几乎相同的衰减，表明问题更多源于社交媒体内容本身而非 agent 特有。

**⚠️ 局限性**

数据仅覆盖前 12 天，缺乏长周期行为；PII 与语言识别可能存在误判；实验仅针对 Qwen2.5‑14B‑Instruct、DeepSeek‑3.2 与 TruthfulQA，未覆盖多模型、多评测与更广泛人类内容来源，且 LoRA rank、epoch 与 warmup 共同调节，难以单独拆解各因素影响。

---

## 511. LiteGUI: Distilling Compact GUI Agents with Reinforcement Learning

**arXiv ID:** 2605.07505 | [PDF](https://arxiv.org/pdf/2605.07505v1)

**作者:** Yubin Wu `[一作]` (Moore Threads AI), Hao Chen `[通讯]` (Moore Threads AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无SFT的轻量化视觉语言GUI代理训练范式，结合引导式同策略蒸馏和多解双层GRPO，显著提升小型模型在长序列任务上的表现。

**💡 创新点**

创新点在于：①把知识蒸馏与强化学习无SFT整合；②引入基于多解轨迹的引导式同策略蒸馏，动态匹配学生行为；③提出多解双层奖励机制，兼顾动作级和子任务规划级；④构建自动化多解数据生成管道和公开Lite-Dataset/Lite-Bench。

**🔧 技术方法**

核心技术包括：引导式同策略蒸馏（Guided-OPD）、多解双层GRPO（MD‑GRPO）、统一GUI动作匹配函数、基于VLM评判器的子任务规划奖励、以及多解动作奖励。

**📊 数据集**

使用了自动生成的Lite-Dataset（30K轨迹，11K多解标注）和Lite-Bench（160个多步任务），并在ScreenSpot‑Pro、OS‑World等公开基准上进行评测。

**📈 对比分析**

在ScreenSpot‑Pro、OS‑World和Lite‑Bench上，Lite‑GUI‑2B以46.86%/13.24%/61.76%的平均成功率领先同规模基准，并在多数指标上超过30B规模模型；Lite‑GUI‑30B‑A3B进一步取得58.95%/22.7%/89.26%的SOTA表现。

**⚠️ 局限性**

主要限制包括：仍需较长的历史窗口导致内存占用较高，模型推理仍相对昂贵；以及在极长或极端状态下的泛化仍待提升。

---

## 512. A Decomposed Retrieval-Edit-Rerank Framework for Chord Generation

**arXiv ID:** 2605.07489 | [PDF](https://arxiv.org/pdf/2605.07489v1)

**作者:** Qiqi He `[一作]` (NetEase Cloud Music), Anqi Huang `[通讯]` (NetEase Cloud Music)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种检索-编辑-重排（RER）框架，用于可控且满足音乐理论约束的和弦生成。

**💡 创新点**

将和弦生成拆分为检索候选、约束投影编辑和软重排三阶段，显式分离多样性与可行性，实现模型可解释、可调节。

**🔧 技术方法**

使用对比学习编码器与FAISS进行检索；利用Viterbi算法实现编辑阶段的最小化编辑成本投影；重排阶段基于检索相似度与编辑成本的加权得分。

**📊 数据集**

采用四套流行歌曲数据集：付费歌曲+AIST标注、Wikifonia测试集、909专业流行曲、内部1558条标注曲，用于检索训练与评估。

**📈 对比分析**

与Transformer‑LM、Bi‑LSTM、HMM等端到端基线在CHE、CC、CTD、PCS、MCTD、CTnCTR等客观指标及人类评估（和声性、创意性、整体偏好）进行对比，RER在多样性与可行性上均优于基线，整体表现更平衡。

**⚠️ 局限性**

仍需手动调节超参数（如λ）；当检索候选与约束集差距大时可能产生保守输出；缺乏全局控制器，整体流程仍需人工干预。

---

## 513. Uncovering Hidden Systematics in Neural Network Models for High Energy Physics

**arXiv ID:** 2605.07470 | [PDF](https://arxiv.org/pdf/2605.07470v1)

**作者:** Lucie Flek `[一作]` (University of Bonn), Ulrich Willemsen `[通讯]` (RWTH Aachen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `6215c339-3735-4be3-8a07-5bbb7004712d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了高能物理中深度学习模型对实验不确定性下的鲁棒性，并提出了在符合实验误差约束的前提下构造对抗扰动的方法，用以评估隐藏的系统误差。

**💡 创新点**

创新点在于将对抗攻击技术与物理可实现的误差约束结合，设计了在一维分布保持不变、满足高斯先验的对抗样本，用来量化神经网络的隐性系统不确定性。

**🔧 技术方法**

采用白盒PGD和C&W混合损失，加入χ²与高斯先验约束，训练多种网络（MLP、GNN、Transformer）并进行对抗训练与评估。

**📊 数据集**

使用Delphes快速模拟的13 TeV pp碰撞数据，分别用于tt̅/WW分类、quark–gluon标签和Missing ET二分类等任务。

**📈 对比分析**

通过对比传统切割基线和深度网络在原始样本、对抗扰动样本以及对抗+原始混合训练下的性能，发现对抗样本可导致DNN精度下降≈1–3%，而切割基线几乎不受影响，对抗训练显著缓解了这一效应。

**⚠️ 局限性**

局限性在于对抗扰动仅覆盖了高阶相关性而未验证真实数据；不同网络结构的泛化能力及对系统误差模型的依赖仍需进一步研究。

---

## 514. WeatherSyn: An Instruction Tuning MLLM For Weather Forecasting Report Generation

**arXiv ID:** 2605.07522 | [PDF](https://arxiv.org/pdf/2605.07522v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 515. Physical Simulators as Do-Operators: Causal Discovery under Latent Confounders for AI-for-Science

**arXiv ID:** 2605.07467 | [PDF](https://arxiv.org/pdf/2605.07467v1)

**作者:** Tsuyoshi Okita `[一作]` (Kyushu Institute of Technology), Tsuyoshi Okita `[通讯]` (Kyushu Institute of Technology)

**通讯引用:** 2608 | [OpenAlex ID](https://openalex.org/A5026921207)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 CFM‑SD，一种利用第一性原理物理模拟器作为真干预操作，结合流匹配与 KDE‑MMD 检测隐含混杂、通过 ATE 判断因果方向，并在 DAG 约束下完成因果结构学习。

**💡 创新点**

①首次在存在潜在混杂的情况下，将物理模拟器作为 Pearl 的 do‑operator；②证明在单变量干预下可用 O(d) 次干预识别 d 变量 DAG；③将流匹配、KDE‑MMD 与 ICP 风格直接边筛选结合，形成完整的因果发现框架。

**🔧 技术方法**

流匹配（Flow Matching）估计条件分布；KDE‑MMD 检测观测与干预分布差异；ATE 估计因果效应；ICP‑style 过滤获得直接因果边；DAG 约束消除环路。

**📊 数据集**

合成数据（含线性、四种非线性结构，γ∈{0.0–0.8}的潜在混杂）；真实科学数据 QSTR（分子毒性）与 SEI（电池电解液添加剂）。

**📈 对比分析**

与观测方法 PC、GES、FCI、LiNGAM、以及干预方法 IGSP、UT‑IGSP 进行对比。在线性 SCM 上，CFM‑SD F1≈0.800；非线性 SCM 上平均 F1≈0.673，均明显优于 IGSP(int)（≈0.545）。在 SEI 与 QSTR 任务中，CFM‑SD 的因果效应估计偏差分别下降 57%–58%，显著优于传统 OLS、IPW、AIPW 等方法。

**⚠️ 局限性**

主要局限：①对模拟器误差敏感，若 ϵ_sim 超过 Δ_min/2 可能导致错误结构；②当变量数 d>20 时，O(d²) 的流匹配计算成本高；③ICP‑style 过滤对多直接子节点可能失效；④在无真干预时需使用代理变量，理论支持不足。

---

## 516. Hierarchical Dual-Subspace Decoupling for Continual Learning in Vision-Language Models

**arXiv ID:** 2605.07512 | [PDF](https://arxiv.org/pdf/2605.07512v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 517. Well-Quasi-Ordering Eulerian Digraphs: Bounded Carving Width

**arXiv ID:** 2605.07468 | [PDF](https://arxiv.org/pdf/2605.07468v1)

**作者:** Dario Cavallaro `[一作]` (Technische Universität Berlin), Stephan Kreutzer `[通讯]` (Technische Universität Berlin)

**通讯引用:** 2136 | [OpenAlex ID](https://openalex.org/A5062011497)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

本文证明了有界 carving 宽度（等价于有界度数和树宽）的欧拉有向图在强吸收关系下是良序的，并给出了相应的元定理框架；

**💡 创新点**

创新点在于提出针对欧拉有向图的强吸收的良序框架，并通过引入Ω‑knitwork等新概念实现结构分解与良序的统一证明；

**🔧 技术方法**

主要使用了图的吸收理论、欧拉切分与 well‑linked 链接技术、Ω‑knitwork 结构以及树宽/Carving 宽度分解方法；

**📊 数据集**

本文为纯理论研究，不使用任何实验数据集；

**📈 对比分析**

因缺乏实验数据，未进行方法比较或性能评估；

**⚠️ 局限性**

局限性包括：仅适用于有界度数/Carving 宽度的图，且对无界度数的欧拉图无法保证良序；强吸收结果难以推广至弱吸收，需要进一步研究。

---

## 518. Diffusion-APO: Trajectory-Aware Direct Preference Alignment for Video Diffusion Transformers

**arXiv ID:** 2605.07503 | [PDF](https://arxiv.org/pdf/2605.07503v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 519. From Feasible to Practical: Pareto-Optimal Synthesis Planning

**arXiv ID:** 2605.07521 | [PDF](https://arxiv.org/pdf/2605.07521v1)

**作者:** Friedrich Hastedt `[一作]` (Imperial College London), Antonio del Rio Chanona `[通讯]` (Imperial College London)

**通讯引用:** 3189 | [OpenAlex ID](https://openalex.org/A5050349202)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种多目标化的回溯合成规划算法MORetro^∗，能够在给定预算内直接生成多目标Pareto前沿路线；

**💡 创新点**

通过权重标量化与贝叶斯优化采样相结合，实现对多维目标空间的高效搜索，并给出了Pareto最优性保证；

**🔧 技术方法**

基于A*搜索、线性标量化、BO采样、AND‑OR图剪枝以及条件预测模块；

**📊 数据集**

使用USPTO‑190、Pistachio‑reachable和ChEMBL药物分子等公开合成数据集；

**📈 对比分析**

与单目标Retro^∗和固定权重Baseline比较，MORetro^∗在超体积（HV）提升5–8%，R2指标下降，成功率保持相近，且化学多样性明显提高；

**⚠️ 局限性**

目前仅支持可加性目标，剪枝依赖可接受的下界，且对权重采样的计算成本较高，未来需扩展至更高维目标和更紧凑的可证明下界。

---

## 520. An Automated Framework for Cybersecurity Policy Compliance Assessment Against Security Control Standards

**arXiv ID:** 2605.07515 | [PDF](https://arxiv.org/pdf/2605.07515v1)

**作者:** Bikash Saha `[一作]` (Indian Institute of Technology Kanpur), Sandeep Kumar Shukla `[通讯]` (International Institute of Information Technology Hyderabad)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

PROPARAG 自动评估组织安全政策与 NIST SP 800-53 控制的符合性，自动检索证据、评估覆盖、识别缺口并给出解释与建议。

**💡 创新点**

创新点在于提出控制级检索增强推理框架，分阶段推理并显式关联证据，生成结构化诊断与建议，显著提升评估准确性。

**🔧 技术方法**

采用检索增强型大型语言模型、意图条件检索、控制知识库、可解释的多阶段推理以及证据链验证等技术。

**📊 数据集**

使用两份真实组织政策语料（共 24/31 份文件，约 80k 词）与 1,007 条 NIST SP 800-53 控制条目，人工构建的专家标注基准。

**📈 对比分析**

与五种基线（单步无检索、检索+单步、BM25、RAG、文档级）对比，PROPARAG 在两组织上 F1 分别为 88.5% 和 82.3%，优于最强基线 11%+ 点，且在不同 LLM 后端表现稳健。

**⚠️ 局限性**

局限性包括对政策结构差异敏感，部分模型在识别“部分覆盖”时误差较大，且缺乏对操作层面或多标准兼容的评估，需进一步提升多模态证据与不确定性估计。

---

## 521. Cloud-top infrared observations reveal the four-dimensional precipitation structure

**arXiv ID:** 2605.07499 | [PDF](https://arxiv.org/pdf/2605.07499v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 522. MASPrism: Lightweight Failure Attribution for Multi-Agent Systems Using Prefill-Stage Signals

**arXiv ID:** 2605.07509 | [PDF](https://arxiv.org/pdf/2605.07509v1)

**作者:** Yang Liu `[一作]` (Sun Yat-sen University), Zhuangbin Chen `[通讯]` (Sun Yat-sen University)

**通讯引用:** 716 | [OpenAlex ID](https://openalex.org/A5090114918)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于小语言模型prefill阶段的轻量级多智能体系统故障归因框架，能够在无oracle、无生成诊断文本、无重放或训练的条件下定位失败源。

**💡 创新点**

创新点包括：①利用prefill阶段的负对数似然（NLL）和注意力权重进行症状定位与候选源筛选；②设计两阶段过滤‑诊断管线，仅需两次prefill无解码；③在长轨迹和无oracle环境下实现高效且高准确率的归因。

**🔧 技术方法**

技术细节：在预填充（prefill）阶段提取token级NLL与注意力；step‑level NLL用于识别症状步骤；注意力聚合用于挑选早期候选源；通过重构提示（保留症状与候选步骤完整内容）进行第二次prefill，最终给出候选失败源的排名；实验采用Qwen3‑0.6B（以及Llama系列）作为小型语言模型。

**📊 数据集**

使用数据集 Who&When（Hand‑Crafted HC 与 Algorithm‑Generated AG 两子集）评估单步根因识别，以及 TRAIL（GAIA 与 SWE‑bench 两子集）评估多点错误定位。

**📈 对比分析**

与基线（A2P、GPT‑4o、OpenAI o3、Claude‑3.7‑Sonnet、Gemini‑2.5‑Pro 等）比较，论文在 Who&When‑HC 上 Top‑1 准确率提升 33.41%，在 TRAIL GAIA 与 SWE‑bench 上均超过大模型，平均每条轨迹推理时长 2.66 s（比 A2P 的 17.82 s 快 6.69×），输入 token 数更少，且不产生输出 token。

**⚠️ 局限性**

局限性：仅适用于可访问prefill内部状态的开放权重模型；归因结果为排名候选而非确定因果关系，仍需人工验证；在混合成功/失败的生产流中误报率和判别阈值尚需进一步校准。

---

## 523. How Far Is Document Parsing from Solved? PureDocBench: A Source-TraceableBenchmark across Clean, Degraded, and Real-World Settings

**arXiv ID:** 2605.07492 | [PDF](https://arxiv.org/pdf/2605.07492v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 524. ExpThink: Experience-Guided Reinforcement Learning for Adaptive Chain-of-Thought Compression

**arXiv ID:** 2605.07501 | [PDF](https://arxiv.org/pdf/2605.07501v1)

**作者:** Tingcheng Bian `[一作]` (Baidu Inc.), Miaohui Wang `[通讯]` (Shenzhen University)

**通讯引用:** 2021 | [OpenAlex ID](https://openalex.org/A5080596209)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于强化学习的链式思考压缩框架 ExpThink，利用经验缓冲记录每个问题最短的正确推理轨迹，并通过难度自适应优势实现准确优先、简洁其次的训练目标。

**💡 创新点**

创新点在于：①自演进的经验缓冲可动态收紧压缩阈值；②将正确计数替代标准差做优势归一化，保留难度信息直至梯度更新；③三阶奖励塑形在保持正确性同时对冗长输出进行分层惩罚。

**🔧 技术方法**

技术手段包括：强化学习（改进的 DAPO）、经验驱动奖励塑形、计数优势归一化、三阶奖励函数、Token 计数与正则化、模型自适应阈值更新。

**📊 数据集**

主要使用 DeepScaleR 数学推理数据集进行训练，并在 AIME24、AMC23、MATH-500、Minerva Math、OlympiadBench 等数学基准上评估；外域测试覆盖 LiveCodeBench、GPQA‑Diamond、MMLU 等多任务基准。

**📈 对比分析**

与 LC‑R1、Laser、AutoThink、AdaptThink、JET 等现有 RL 压缩方法对比，ExpThink 在三种模型规模下实现 65.5%–77.2% 的 token 缩减并同步提升准确率，IPT 提升至 3 倍以上，整体性能优于所有基线。

**⚠️ 局限性**

局限性包括：仅在可验证答案的数学推理任务上验证；依赖可靠答案验证；缺乏与原始 DAPO 匹配的计算对比；经验缓冲、α 与 r_pen 等超参仍为经验性选择；未验证多模态或交互式任务的适用性。

---

## 525. Spying Across Chiplets: Side-Channel Attacks in 2.5/3D Integrated Systems

**arXiv ID:** 2605.07486 | [PDF](https://arxiv.org/pdf/2605.07486v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 526. Broken-symmetry shape discrimination on a driven Duffing ring

**arXiv ID:** 2605.07475 | [PDF](https://arxiv.org/pdf/2605.07475v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 527. Efficient Data Selection for Multimodal Models via Incremental Optimization Utility

**arXiv ID:** 2605.07488 | [PDF](https://arxiv.org/pdf/2605.07488v1)

**作者:** Jinhao Jing `[一作]`, Zhan Su `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 One‑Step‑Train (OST) 框架，对大规模多模态模型（LMM）的合成数据进行高效筛选，将样本选择重新表述为一次性梯度更新的增益排序问题。

**💡 创新点**

创新点在于：① 用轻量代理模型模拟一次梯度步长，直接估计样本对验证集损失的即时下降（Δ），避免了昂贵的 LOO 或影响函数计算；② 通过梯度内积评分实现无 Hessian 的排序，理论证明与完整影响函数高度一致；③ 发现代理模型在轻微 warm‑up 阶段最优，避免过拟合合成噪声；④ 证明在固定计算预算下“小而优”能逆转负迁移，形成 Pareto‑optimal 的数据‑compute 权衡。

**🔧 技术方法**

主要技术包括：梯度内积 (gV · gi) 作为 utility 评分；单步模拟更新；轻量代理（InternVL3‑1B）与目标大模型的梯度子空间对齐；两阶段训练协议（比例步数与固定计算）；以及对照实验框架（Full‑SFT、随机、LLM‑as‑a‑Judge、DEITA）。

**📊 数据集**

使用 351,157 条多模态数学推理题目（含 CoT 生成）作为合成训练集，并在 100 条代表性验证集上估计 utility；在 500 条内部基准及四个公开多模态数学/逻辑 benchmark（MathVision、MathVista、WeMath、LogicVista）进行下游评测。

**📈 对比分析**

与全量 SFT、随机采样、LLM‑as‑a‑Judge、DEITA 等基线对比：在 Top‑50% 选样时 GPU 计算量减少 17%（训练阶段 43%），平均准确提升 1.8 分；在 Top‑20% 选样时在固定计算预算下平均分提升 5.6 分，超过 LLM‑as‑a‑Judge 1.8 分、DEITA 0.4 分、Full‑SFT 8.8 分；在复杂推理任务（LogicVista、WeMath）中成功逆转负迁移。

**⚠️ 局限性**

局限性包括：① 依赖验证集（anchor）的代表性，若偏差会误导 utility 评分；② 仅进行点‑级排序，未考虑样本间冗余与交互，可能导致子集多样性不足；③ 代理到目标模型的梯度对齐理论在极端规模或不同架构（如 SSM）下的上限尚未完全验证。

---

## 528. ForgeVLA: Federated Vision-Language-Action Learning without Language Annotations

**arXiv ID:** 2605.07474 | [PDF](https://arxiv.org/pdf/2605.07474v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 529. Transfer Learning Across Fast- and Full-Simulation Domains in High-Energy Physics

**arXiv ID:** 2605.07471 | [PDF](https://arxiv.org/pdf/2605.07471v1)

**作者:** Matthias Schott `[一作]`, Lucie Flek `[通讯]` (Bonn-Aachen International Center for Information Technology, University of Bonn)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文系统研究了在高能物理中，将在快速仿真（Delphes）上训练的模型迁移到不同的快速仿真以及全仿真（ATLAS Open Data）上的表现；

**💡 创新点**

创新点在于对三类典型任务（信号背景分类、夸克-胶子喷注标记、缺失横向能量回归）使用不同网络架构进行跨域迁移学习，并量化迁移对训练数据需求的减小，提出预训练模型可视为可重复使用的科学资产；

**🔧 技术方法**

采用了密集神经网络、图神经网络（EdgeConv）和Transformer编码器等常用深度学习架构，并实现全参数微调与部分冻结两种迁移策略；

**📊 数据集**

使用四种物理过程（t̄t、W/Z+jets、WW）在三种仿真域（ATLAS Delphes、CMS Delphes、ATLAS全仿真）产生的约1.5×10⁵事件的数据集；

**📈 对比分析**

通过比较预训练模型在不同目标域、不同训练样本量下的ROC曲线、AUC、损失和缺失能量分辨率等指标与从零开始训练的基线模型，发现预训练模型在所有任务中均能以约一半的目标域数据达到相同或更好性能；

**⚠️ 局限性**

局限性包括：仅在模拟数据上验证，未评估对真实实验数据的迁移效果；只探讨了有限的模型规模和任务，未涉及更大模型或更多物理过程；对全仿真域的迁移需要较大数据量，且冻结层策略在大域移时表现不佳。

---

## 530. Think-with-Rubrics: From External Evaluator to Internal Reasoning Guidance

**arXiv ID:** 2605.07461 | [PDF](https://arxiv.org/pdf/2605.07461v1)

**作者:** Jiachen Yu `[一作]` (Tsinghua University), Yujiu Yang `[通讯]` (Tsinghua University)

**通讯引用:** 4132 | [OpenAlex ID](https://openalex.org/A5020953714)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Think-with-Rubrics 框架，将 Rubric 生成过程嵌入 LLM 的推理轨迹，让模型在生成答案前先制定自我 Rubric 以指导后续回答。

**💡 创新点**

创新点在于：①将 Rubric 从仅作为评价工具转化为内部思考步骤；②通过黄金 Rubric 与自生成 Rubric 的双重奖励实现自监督与外部监督的协同；③在 RL 阶段采用 DAPO 与自定义奖励函数，提升模型的自一致性和整体表现。

**🔧 技术方法**

使用技术包括：大规模语言模型（Qwen3-8B/4B）、SFT 预训练、强化学习（DAPO）、专门的 Rubric 验证器、加权奖励（黄金 Rubric、Self‑generated Rubric、格式奖励）。

**📊 数据集**

主要数据集：Openrubrics（用于训练 Rubric 生成与 verifier、黄金 Rubric 监督）；IFEval、IFBench、InfoBench 用于公开评估。

**📈 对比分析**

与传统 Rubric-as-Reward 基线相比，Think-with-Rubrics 在 8B 模型上平均提升 3.87 分（在 IFEval、IFBench、InfoBench 上均表现更好），在 4B 模型上提升 4.16 分；混合奖励（golden + self‑generated）得到最佳表现。

**⚠️ 局限性**

局限性包括：仍需依赖外部黄金 Rubric 数据集；自生成 Rubric 在跨任务泛化能力有限；训练过程复杂，需构建专用 verifier；对较小模型的提升有限，可能受模型自身一致性约束。

---

## 531. Learning Minimal-Deviation Corrections for Multi-Dimensional Mismodelling in HEP Simulations

**arXiv ID:** 2605.07460 | [PDF](https://arxiv.org/pdf/2605.07460v1)

**作者:** Matthias Schott `[一作]`, Lucie Flek `[通讯]` (Bonn-Aachen International Center for Information Technology, University of Bonn)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于残差变换的神经网络方法，利用有限的单维分布信息对MC模拟进行最小偏差校正，保持原始模拟的全维相关结构

**💡 创新点**

创新在于将单维对齐与最小化偏差结合为正则化目标，采用可微分直方图损失和相关矩阵惩罚，实现对多维结构的保留与校正

**🔧 技术方法**

使用全连接神经网络实现全局残差变换与两步残差变换，配合可微分直方图、相关矩阵损失及张量化正则化；训练过程基于梯度下降和早停

**📊 数据集**

在Pythia8生成的tt̅样本（Tevatron→LHC）和不同底部事件调校（AZ→Monash）两个基准数据集上进行验证；全部数据均通过Delphes快速仿真

**📈 对比分析**

通过对比单维分布、派生观测量、相关矩阵以及使用分类器的ROC/AUC，结果显示两步方法在大多数特征上恢复率>90%，相关偏差<0.2，ROC差异<1%

**⚠️ 局限性**

局限在于仅利用单维信息，可能在高维极端相关性不足时无法完全捕获；模型复杂度受样本量限制，未在真实实验数据上验证，系统误差估计仍待完善

---

## 532. Estimation of Motor Unit Parameters from Surface Electromyograms using an Informed Autoencoder

**arXiv ID:** 2605.07458 | [PDF](https://arxiv.org/pdf/2605.07458v1)

**作者:** Kaja Balzereit `[一作]` (Hochschule Bielefeld), Axel Schneider `[通讯]` (Hochschule Bielefeld)

**通讯引用:** 1760 | [OpenAlex ID](https://openalex.org/A5009208640)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

利用信息化自编码器（Encoder‑X 变体）结合前向物理模型，从非侵入式肌电（sEMG）信号中同时估计运动单元的归巢中心与传导速度等参数。

**💡 创新点**

提出将物理前向模型直接嵌入自编码器解码器，形成灰盒架构；该方法无需先验的手工参数化，能一次性估计多参数并保持物理可解释性。

**🔧 技术方法**

采用深度前馈编码器 + 解析前向 EMG 生成模型 + 双差分与归一化 + 组合 MSE 与交叉相关的损失函数；训练使用 AdamW，早停。

**📊 数据集**

使用六个合成肌肉的模拟数据：每肌肉 774 个运动单元，挑选 8 个运动单元（315–3367 条肌纤维），40 电极阵列，5000 Hz 采样，加入 Gaussian 噪声（SNR=1）。

**📈 对比分析**

与基于聚类的传统方法对比：在 8 个运动单元中，信息化自编码器在部分案例误差更低；总体平均绝对误差分别为 innervation‑zone ≈ 2.60、传导速度 ≈ 0.17（误差均低于 3.5 与 0.35）。

**⚠️ 局限性**

仅在合成数据上验证，缺乏真实人体实验；模型对噪声、肌肉几何变化的泛化能力未评估；损失函数组合可能导致训练停留在局部最优。

---

## 533. Is the Future Compatible? Diagnosing Dynamic Consistency in World Action Models

**arXiv ID:** 2605.07514 | [PDF](https://arxiv.org/pdf/2605.07514v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 534. A Log-Domain Approximation of SOCS Decoding for Turbo Product Codes

**arXiv ID:** 2605.07519 | [PDF](https://arxiv.org/pdf/2605.07519v1)

**作者:** Oleg Nesterenkov `[一作]` (Skolkovo Institute of Science and Technology), Pavel Rybin `[通讯]` (Skolkovo Institute of Science and Technology)

**通讯引用:** 164 | [OpenAlex ID](https://openalex.org/A5052831471)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种针对带有 eBCH 组件码的涡轮产品码（TPC）的 Log‑Domain Max‑Log 近似 SOCS 解码器，能够在保持 Chase‑II 列表生成的低复杂度基础上，利用分段线性映射产生软输出。

**💡 创新点**

创新点在于：①将 SOCS 的概率域后处理转换为 Log‑Domain 近似；②用基于列表候选与未列候选之间可靠性差距的分段线性函数（带归一化偏移）来生成 extrinsic LLR，避免了概率域运算和神经网络推理；③通过优化少量系数实现对 SOCS 性能的逼近。

**🔧 技术方法**

技术手段包括：Max‑Log 近似、Chase‑II 列表生成、可靠性差距度量、分段线性映射、Monte Carlo 差分进化优化和通用互信息（GMI）尺度因子调优。

**📊 数据集**

使用了基于 (256,239) eBCH 的 TPC 结构的仿真数据，SNR 范围覆盖 BER 至 10⁻⁶ 级别。

**📈 对比分析**

与 Chase–Pyndiah、SOCS（两种配置）以及神经回滚解码器（以及 MAP 参考）进行比较；实验结果显示该方法在相同列表大小下显著优于 Chase–Pyndiah，基本逼近 SOCS 上限，且性能与神经回滚相当，且仅需简单的 Log‑Domain 计算。

**⚠️ 局限性**

局限性包括：需要针对每个半迭代手动优化系数，缺乏通用的解析参数选择方法；目前仅在 (256,239) eBCH 组件码上验证，尚未证明对其他列表生成算法或更长码字的适用性；硬件量化与实现细节仍需进一步研究。

---

## 535. Lightweight Unpaired Smartphone ISP Transfer with Semantic Pseudo-Pairing

**arXiv ID:** 2605.07495 | [PDF](https://arxiv.org/pdf/2605.07495v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 536. DIMoE-Adapters: Dynamic Expert Evolution for Continual Learning in Vision-Language Models

**arXiv ID:** 2605.07494 | [PDF](https://arxiv.org/pdf/2605.07494v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 537. Bilevel Graph Structure Learning, Revisited: Inner-Channel Origins of the Reported Gain

**arXiv ID:** 2605.07577 | [PDF](https://arxiv.org/pdf/2605.07577v1)

**作者:** Minkyoung Kim `[一作]` (Yonsei University), Beakcheol Jang `[通讯]` (Yonsei University)

**通讯引用:** 3025 | [OpenAlex ID](https://openalex.org/A5067151609)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 frozen-ϕ 控制实验，隔离双层图结构学习（GSL）中的图重构与内循环训练动力学两部分，从而重新评估前者在性能提升中的实际贡献。

**💡 创新点**

创新点在于首次将训练动力学与图重构解耦，证明内循环训练动态往往能解释 78–101% 的流量预测提升以及 37–44% 的节点分类提升，并提出 frozen-ϕ 作为标准诊断工具；同时提供图蒸馏作为对非梯度式 GSL 方法的补充。

**🔧 技术方法**

主要技术包括：双层优化框架、梯度近似（第一阶）、Frozen-ϕ 对比实验、内循环隐式梯度正则化分析、T 步内循环探测、图质量扰动实验、以及与谱指标的对照分析。

**📊 数据集**

使用六个流量预测基准（PeMS04/07/08、METR‑LA、PeMS‑BAY、AirQuality）和两个节点分类基准（Cora、Citeseer），配合 DiffConv、ChebConv、MPGRU、DCRNN、AGCRN、GraphWaveNet 等骨干网络；节点分类采用 LDS 的 Bernoulli 边参数化。

**📈 对比分析**

与普通训练（Vanilla）、Frozen-ϕ 和完整双层 GSL 进行三向比较。结果显示：在流量预测上，完整双层相较于 Vanilla 可提升 3.8–6.4% MAE，而 Frozen-ϕ 贡献了 78–101% 的提升；在节点分类中，Frozen-ϕ 贡献 37–44%；此外，谱指标的提升与任务性能不相关。

**⚠️ 局限性**

限制包括：仅适用于梯度式、增量式图更新的双层 GSL（不适用于单层或 EM 方案、非增量重写图的情况）；在异质图或未验证的 GSL 设定下效果未知；节点分类仅在 Cora/Citeseer 上验证；双层方法训练开销大（≈15×），且未讨论推理成本。

---

## 538. PolarVLM: Bridging the Semantic-Physical Gap in Vision-Language Models

**arXiv ID:** 2605.07574 | [PDF](https://arxiv.org/pdf/2605.07574v1)

**作者:** Yuliang Li `[一作]` (Beijing University of Posts and Telecommunications), Zhanyu Ma `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 8177 | [OpenAlex ID](https://openalex.org/A5039812471)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了PolarVLM，一种能够将偏振光学参数（DoLP和AoLP）融合进视觉‑语言模型的双流框架，并创建了PolarVQA基准数据集，用于在反射与透明物体场景下进行开放式视觉问答。

**💡 创新点**

创新点包括：①使用双流架构将RGB与偏振信息分离处理，并通过token级拼接实现柔性融合；②提出两阶段渐进式训练策略，先对偏振信号进行语义对齐，再进行联合指令微调，防止模态崩溃；③构建了首个基于物理引导的偏振VQA基准，提供了75K高质量的物理描述与指令对。

**🔧 技术方法**

采用了CLIP ViT‑L/14的双分支视觉编码器、LLaVA‑1.5（Vicuna‑13B）语言模型、LoRA/QLoRA微调、物理特征的正弦余弦映射、以及基于Florence‑2‑large与GPT‑4o‑mini的自动化物理‑文本生成流水线。

**📊 数据集**

使用了公开的PolarFree、RGBP‑Glass等偏振图像数据，经过自动化生成得到28.5K物理描述和46.8K指令对，总计75K条样本。

**📈 对比分析**

通过LLM‑as‑Judge（GPT‑4o‑mini）在PolarVQA测试集上评估，PolarVLM在“整体”得分上比RGB‑only VLM提升了25.4%（6.08 vs. 4.85），在玻璃计数、定位、反射识别等物理任务上均超过了大型闭源模型GPT‑4.1和所有基准的级联偏振管道。

**⚠️ 局限性**

局限性在于模型规模仅为13B参数，尚未探索更大基座与更广泛数据多样性的扩展；此外，偏振数据获取受限，未来可通过更大范围的真实偏振采集提升鲁棒性。

---

## 539. Beyond Distribution Estimation: Simplex Anchored Structural Inference Towards Universal Semi-Supervised Learning

**arXiv ID:** 2605.07557 | [PDF](https://arxiv.org/pdf/2605.07557v1)

**作者:** Yaxin Hou `[一作]` (Southeast University), Yuheng Jia `[通讯]` (Southeast University)

**通讯引用:** 1813 | [OpenAlex ID](https://openalex.org/A5013880628)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了SAGE框架，解决极度标签稀缺且无标签分布未知的半监督学习问题。

**💡 创新点**

创新点在于把分布估计替换为基于高阶图关系的结构推理，并采用固定单纯形等角紧框架与分布无关的可靠性加权。

**🔧 技术方法**

技术包括图状态关系推理（GRI）、单纯形等角紧框架、分布无关可靠性优先化（DRP）、辅助分类分支、对比损失等。

**📊 数据集**

使用了CIFAR‑10、CIFAR‑100、SVHN、Food‑101、STL‑10以及ImageNet‑1k等公开数据集。

**📈 对比分析**

与FixMatch、FreeMatch、CGMatch、CPG等SSL/RTSSL基线比较，在所有设置下平均提升约8.52%（如CIFAR‑10在61.24%对比CPG的50.24%），并在多种分布情形下均显著优于最强基线。

**⚠️ 局限性**

局限性在于假设类别数已知、图相似度矩阵随批量平方增长、以及在极大规模或开放世界场景下尚未充分验证。

---

## 540. ProteinJEPA: Latent prediction complements protein language models

**arXiv ID:** 2605.07554 | [PDF](https://arxiv.org/pdf/2605.07554v1)

**作者:** Dan Ofer `[一作]` (Hebrew University of Jerusalem), Michal Linial `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 16215 | [OpenAlex ID](https://openalex.org/A5085924022)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在蛋白质序列预训练中，将掩码位置的 latent-space 预测（JEPA）与传统的 masked‑language‑modeling（MLM）交叉熵结合，并在 16 个下游任务上评估其效果。

**💡 创新点**

创新点在于：① 仅在掩码位置使用 JEPA，保持 MLM 交叉熵；② 这种双重损失在预训练时提升了回归、适应性和结构检索任务；③ 证明全位置 JEPA 或单独 JEPA 对模型无益。

**🔧 技术方法**

技术细节包括：JEPA 与 SIGReg 正则化、两层 SwiGLU 预测器、余弦相似度损失、无 EMA teacher，仅在掩码位置计算 latent 预测。

**📊 数据集**

训练数据：UniRef50 未标记蛋白序列；评估数据：由 TAPE、ProteinBERT、SCOPe‑40 等公开基准构成的 16 任务线性探针集合（包含回归、功能、结构、交互等）。

**📈 对比分析**

方法：在相同 8 小时 wall‑clock 预算下与单一 MLM 进行匹配对照；在预训练后端（ESM2‑35M/150M、AMPLIFY‑120M、随机 ESM2‑35M、ProteinBERT2‑35M）中，masked‑position MLM+JEPA 在 10/3/3、11/2/3 等任务上超过 MLM‑only，尤其在回归、适应性和 SCOPe‑40 检索任务上取得显著提升；单独 JEPA 则普遍下滑。

**⚠️ 局限性**

局限性：① 仅在 35–150M 参数规模上测试，未验证更大模型；② 单独 JEPA 效果差，需与 MLM 同在；③ 从零起步的表现受架构影响不稳定；④ 评估仅使用静态平均池化线性探针，未覆盖微调或每残基任务；⑤ 所有位置 JEPA 与掩码位置 JEPA 的损失形式与目标集混合，难以单独归因；⑥ 仅覆盖序列模型，未探讨多模态或更复杂任务。

---

## 541. Mind the Gap: Geometrically Accurate Generative Reconstruction from Disjoint Views

**arXiv ID:** 2605.07550 | [PDF](https://arxiv.org/pdf/2605.07550v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 542. Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation

**arXiv ID:** 2605.07613 | [PDF](https://arxiv.org/pdf/2605.07613v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 543. Disagreement-Regularized Importance Sampling for Adversarial Label Corruption

**arXiv ID:** 2605.07551 | [PDF](https://arxiv.org/pdf/2605.07551v1)

**作者:** Csongor Horváth `[一作]` (Uppsala University), Prashant Singh `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于代理模型集群的失真检测与重要性采样方法，利用跨代理的损失排名方差来筛选训练样本

**💡 创新点**

创新点在于将传统的基于梯度/损失大小的优先采样转为跨模型的不一致性度量（方差），并给出有限样本浓缩与污染率上界的理论保证

**🔧 技术方法**

使用了K个轻量级代理网络训练得到的损失排名方差作为分数，并结合静态裁剪与在线重采样的IS分布；核心技术包括代理集群、方差聚合、McDiarmid不等式证明

**📊 数据集**

实验数据集包括CIFAR‑10、CIFAR‑100、Food‑101，分别用于不同噪声模型（定向高梯度噪声、均匀对称噪声和自然噪声）

**📈 对比分析**

与随机裁剪、EL2N、Forget、Consensus‑loss、AUM以及标准IS、RHO‑LOSS等基线比较；在定向噪声下在CIFAR上获得比随机+均匀IS高7–9pp的准确率，在线IS在25%噪声下相较均匀SGD提升≈9pp，且保持了对抗性鲁棒性

**⚠️ 局限性**

局限性包括：对代理的“简单性偏置”假设在低复杂度或自然噪声场景下失效；需足够的边界样本且keep‑fraction不能过大；理论上对ε<1/2的限制，且K需足够大以保证分离，实际中K=3足够但理论保证保守

---

## 544. SAM 3D Animal: Promptable Animal 3D Reconstruction from Images in the Wild

**arXiv ID:** 2605.07604 | [PDF](https://arxiv.org/pdf/2605.07604v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 545. Your Language Model is Its Own Critic: Reinforcement Learning with Value Estimation from Actor's Internal States

**arXiv ID:** 2605.07579 | [PDF](https://arxiv.org/pdf/2605.07579v1)

**作者:** Yunho Choi `[一作]` (Seoul National University), Yohan Jo `[通讯]` (Seoul National University)

**通讯引用:** 4644 | [OpenAlex ID](https://openalex.org/A5016844435)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过利用大型推理模型的内部隐藏状态训练轻量级探针，POISE在强化学习可验证奖励（RLVR）任务中实现了即时基线估计，避免了传统critic或多次rollout的高昂计算成本。

**💡 创新点**

核心创新在于跨rollout基线设计，使基线与具体生成轨迹独立，并利用内部状态实现无偏、低方差的价值预测，从而显著降低计算开销。

**🔧 技术方法**

技术手段包括轻量级线性探针、提示/推理隐藏状态与熵特征的提取、跨rollout基线构造、PPO式剪辑目标以及在线滑动缓冲区的联合训练。

**📊 数据集**

实验数据集覆盖数学推理（DAPO‑Math 17K、AMC、AIME、HMMT、BRUMO）、编码（AceCoder）、工具调用（ToolDial）及指令跟随（IF‑RLVR），主要以Qwen3‑4B和DeepSeek‑1.5B模型进行评估。

**📈 对比分析**

与基准DAPL和DAPO相比，POISE在Qwen3‑4B与DeepSeek‑1.5B的Avg@32分数相当或略优，同时训练时间缩短约25–30%，梯度方差更低，说明其更高的计算效率与稳定性。

**⚠️ 局限性**

目前仅在固定计算预算下验证，未探讨长期训练效果、token级信用分配、以及对非数学推理任务的更广泛推广，需进一步研究。

---

## 546. Nürnberg NLP at PsyDefDetect: Multi-Axis Voter Ensembles for Psychological Defence Mechanism Classification

**arXiv ID:** 2605.07606 | [PDF](https://arxiv.org/pdf/2605.07606v1)

**作者:** Philipp Steigerwald `[一作]` (Technische Hochschule Nurnberg Georg Simon Ohm), Jens Albrecht `[通讯]` (Technische Hochschule Nurnberg Georg Simon Ohm)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于九个投票者的交叉模型、训练方法和类别粒度的投票集成，用来判别情感支持对话中的心理防御机制水平。

**💡 创新点**

创新点在于：①将九类分类拆分为“无防御”和“防御”两层，分别用通用与专门投票者；②在投票者集合中引入生成式、判别式和逻辑回归三种训练方法，并选取互相不相关错误的不同基模型；③使用 GPT‑5.2 生成的 738 条合成对话来平衡极少数类别，提升投票者的独立性。

**🔧 技术方法**

使用 QLoRA 对大语言模型进行监督微调（SFT）、在已微调模型上接头部进行判别式微调（ClsHead）以及冻结特征后训练的逻辑回归（LR），并在三类基模型（Ministral‑8B、Phi‑4‑14B、Llama‑3.1‑8B）上交叉投票。

**📊 数据集**

主要数据集为 PSYDEFCONV（与 ESConv 结合的情感支持对话语料），并在 80/20 划分的训练集上增添 GPT‑5.2 生成的合成对话，共 738 条。

**📈 对比分析**

相较于 21 组参赛队伍的基线（F1≈0.315），我们的九投票器在隐藏测试集上取得 F1_test=0.420（比基线提升 33.4%），并在比赛中排名第一。

**⚠️ 局限性**

主要局限包括：样本量仅 1,864 条，导致高阶类别的提升难以泛化；投票者选择和阈值设定缺乏足够统计支持；注释者间一致性仅为 κ=0.639，难以进一步提升宏观 F1；系统仍需蒸馏以实现实时部署。

---

## 547. Stencil Computations on Tenstorrent Wormhole

**arXiv ID:** 2605.07599 | [PDF](https://arxiv.org/pdf/2605.07599v1)

**作者:** Lorenzo Piarulli `[一作]` (Sapienza University of Rome), Daniele De Sensi `[通讯]` (Sapienza University of Rome)

**通讯引用:** 685 | [OpenAlex ID](https://openalex.org/A5056277459)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究将二维 5 点模板算子映射到 Tenstorrent Wormhole AI 加速器，提出两种异构实现（Axpy 与 MatMul）并对其性能、能耗及架构瓶颈进行全面分析

**💡 创新点**

首次将稀疏模板运算拆解为子矩阵操作并通过 Axpy 方式避免 tile 转换，揭示 Wormhole 核心对传统 HPC 内核的潜在竞争力，并给出统一内存与后续 Blackhole 架构的改进建议

**🔧 技术方法**

采用 TT‑Metalium C++ API 对 Tensix 核心进行编程，使用 bfloat16 数据格式，配合 CPU 与加速器间 PCIe 传输、子矩阵提取和矩阵乘法实现

**📊 数据集**

以 2D 拉普拉斯方程的 Jacobi 迭代为测试案例，尺寸范围从 1024×1024 到 30720×30720，迭代次数 100/500/1000，使用 bfloat16 精度

**📈 对比分析**

与多线程 OpenMP CPU 基线以及各自的内核时长对比，Axpy 在主机观察时间上比 CPU 慢约 3 倍，但在核级执行时间与能耗上表现更优；MatMul 在 CPU 侧转换占 90% 时延，性能远逊于 Axpy；在理想统一内存（UVM/UPM）条件下，Axpy 可逼近或超越 CPU 基线

**⚠️ 局限性**

主要局限包括 32×32 固定 tile 大小导致 padding 及内存浪费、对标量与小向量运算支持不足、CPU 侧 tilize/untilize 过度耗时、PCIe 传输带宽瓶颈以及设备初始化延迟。

---

## 548. Optimal Recourse Summaries via Bi-Objective Decision Tree Learning

**arXiv ID:** 2605.07598 | [PDF](https://arxiv.org/pdf/2605.07598v1)

**作者:** Ioannis Chatzis `[一作]` (National Technical University Of Athens), Giorgos Stamou `[通讯]` (National Technical University Of Athens)

**通讯引用:** 3025 | [OpenAlex ID](https://openalex.org/A5085359792)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种通过求解双目标最优决策树来生成全局可操作回溯（Recourse）摘要的方法SOGAR，能够在保证成本与效果之间的Pareto最优平衡的前提下为不同人群分组提供统一行动方案。

**💡 创新点**

创新点在于：①将回溯摘要学习转化为可分离的双目标决策树优化问题；②利用STreeD框架实现全局最优且能完整输出Pareto前沿；③提供多目标（成本、失效率）最优解而非单一加权或阈值解。

**🔧 技术方法**

主要技术包括：动态规划与缓存剪枝的STreeD最优决策树求解；CPU/GPU并行加速叶子动作评估；离散化与二值化处理特征以适配树搜索；使用最大百分位移（MPS）成本度量和基于阈值的动作生成。

**📊 数据集**

在四个公共表格数据集上进行评估：Employee Attrition、German Credit、Bank Marketing、Adult Income。

**📈 对比分析**

与五个基线（CET、AReS、GLOBE-CE、GLANCE、T‑CREx）比较。SOGAR在无效性（成本+失效）指标上始终取得最佳成绩，成本低于大多数基线，失效率与某些基线相当或略次；尽管运行时间略高于部分基线，但一次性输出完整Pareto前沿，避免了多次实验。

**⚠️ 局限性**

局限性包括：相对较高的计算开销，尤其在大规模高维数据时缓存与搜索空间可能超出硬件限制；算法的最优性在时间限制下失效，需采用anytime终止；仅提供模型审计的一种视角，无法捕捉所有类型的偏见或复杂特征交互。

---

## 549. Coordinated Motion Planning is FPT on Discretized Simple Polygons

**arXiv ID:** 2605.07570 | [PDF](https://arxiv.org/pdf/2605.07570v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 550. TraceAV-Bench: Benchmarking Multi-Hop Trajectory Reasoning over Long Audio-Visual Videos

**arXiv ID:** 2605.07593 | [PDF](https://arxiv.org/pdf/2605.07593v1)

**作者:** Hengyi Feng `[一作]` (University of Electronic Science and Technology of China), Wentao Zhang `[通讯]` (Peking University)

**通讯引用:** 15392 | [OpenAlex ID](https://openalex.org/A5100459860)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了首个长时音视频多跳推理与多模态幻觉鲁棒性基准（≈2,200道多跳多选题，覆盖578段长视频，总时长339.5小时），并对多种OmniLLM和MLLM进行系统评测。

**💡 创新点**

创新点在于：①将多跳推理扩展到跨模态、跨时长的长视频情境；②同时评估多模态幻觉鲁棒性；③设计三步半自动构建管线和多阶段质量保证流程，确保每题都有可追溯的多跳证据链。

**🔧 技术方法**

技术手段包括：①分钟级视觉字幕（Qwen3‑VL‑32B‑Instruct）+实体缓存；②音频‑视觉异步融合（Gemini‑2.5‑Flash）；③GPT‑5.1驱动的多阶段问答生成与规则/逻辑校验、盲fold检测、人工审核；评测时使用统一的多模态输入和长视频帧采样策略。

**📊 数据集**

数据集基于从OmniVideoBench、LVBench、VideoMME等公开长视频库中挑选的578段10–140分钟多语种视频，去重后进行视觉动态、音频完整性和多跳潜力三项过滤，最终生成约339.5小时的音视频内容。

**📈 对比分析**

对比方法：将闭源Gemini系列、开源OmniLLM（Ming‑Flash‑Omni‑2.0、Qwen3‑Omni‑30B‑A3B等）以及单模态MLLM进行统一评测。结果显示：整体准确率低，最佳闭源模型Gemini 3.1 Pro仅68.29%，最佳开源Ming‑Flash‑Omni‑2.0为51.70%；幻觉鲁棒性与通用推理能力解耦，Gemini 3.1 Pro在幻觉测试中达84.61%，但整体性能仍不佳。

**⚠️ 局限性**

局限性包括：①多跳推理仍显著受限，尤其长链和视频中后期信息；②模型对视频早期信息偏好明显；③幻觉鲁棒性与推理性能脱节，需要针对性训练；④数据规模和多样性（仅2,200题、英语/中文）有限，可能限制泛化；⑤半自动构建管线仍可能引入偏见与错误。

---

## 551. Beyond GSD-as-Token: Continuous Scale Conditioning for Remote Sensing VLMs

**arXiv ID:** 2605.07562 | [PDF](https://arxiv.org/pdf/2605.07562v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 552. Open-Ended Task Discovery via Bayesian Optimization

**arXiv ID:** 2605.07572 | [PDF](https://arxiv.org/pdf/2605.07572v1)

**作者:** Masaki Adachi `[一作]` (Toyota Motor Corporation), Juliusz Ziomek `[通讯]` (University of Oxford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了生成-选择-细化（Generate‑Select‑Refine，GSR）框架，用于在未知任务空间中在线生成、分配预算并优化任务，目标是找到可在给定实验预算内实现的最优任务及其最优解。

**💡 创新点**

创新点在于：①首次将任务生成视为生成优化问题并给出可实现的理论收益上限（对单任务BO只有对数级的额外后悔）；②采用粗到细的分层任务生成策略，结合置信上界实现任务选择；③将大型语言模型（LLM）与贝叶斯优化无缝集成，提供任务生成与效用估计。

**🔧 技术方法**

使用技术包括：GP‑UCB 贝叶斯优化、任务‑UCB 选择策略、LLM 驱动的进化式任务生成、LLM 委员会式效用估计（Bradley‑Terry 模型）、子高斯噪声分析与后悔理论证明。

**📊 数据集**

实验数据集包括：葡萄酒质量（Wine Quality）、SUMMIT 化学工艺仿真、Materials Project 晶体结构数据库，以及常见的人工基准函数（EI、UCB、MES 等）和合成的未知搜索空间测试。

**📈 对比分析**

与基线（ShinkaEvolve、随机、Successive Halving、Hyperband、LLM 选择器、单任务 Seed 与 Oracle）在四大场景（新产品开发、合成放大、逆优化、专利重定位）以及合成基准函数上对比，GSR 一致优于所有基线，逼近 Oracle 性能并在多种任务上实现显著收益。

**⚠️ 局限性**

局限性包括：①对 LLM 生成质量的依赖，生成策略仍属于经验性；②效用函数的定义和校准需人工或实验验证，缺乏通用自学习机制；③理论中对“可成功细化”与“置信阈值”等假设要求较强，在实际应用中需进一步验证。

---

## 553. On the Invariance and Generality of Neural Scaling Laws

**arXiv ID:** 2605.07546 | [PDF](https://arxiv.org/pdf/2605.07546v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 554. Implicit Preference Alignment for Human Image Animation

**arXiv ID:** 2605.07545 | [PDF](https://arxiv.org/pdf/2605.07545v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 555. From Pixels to Prompts: Vision-Language Models

**arXiv ID:** 2605.07544 | [PDF](https://arxiv.org/pdf/2605.07544v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 556. MemCompiler: Compile, Don't Inject -- State-Conditioned Memory for Embodied Agents

**arXiv ID:** 2605.07594 | [PDF](https://arxiv.org/pdf/2605.07594v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 557. System Test Generation for Virtual Reality Applications using Scenario Models

**arXiv ID:** 2605.07534 | [PDF](https://arxiv.org/pdf/2605.07534v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 558. Dynamic Mode Decomposition along Depth in Vision Transformers

**arXiv ID:** 2605.07556 | [PDF](https://arxiv.org/pdf/2605.07556v1)

**作者:** Nishant Suresh Aswani `[一作]` (New York University Tandon), Saif Eddin Jabari `[通讯]` (New York University Tandon)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了ViT深度层的线性自洽性，使用动态模式分解（DMD）拟合单一线性算子在连续Transformer块中的递归应用，并评估其对重构和下游性能的影响。

**💡 创新点**

将线性替代与块递归两条研究线索统一，提出并验证“自治线性假设”，探讨单一时间不变线性算子能否近似多步ViT块，并系统分析不同起始层、剪切长度、矩阵秩与标定样本对拟合质量的影响。

**🔧 技术方法**

采用动态模式分解（DMD）、最小二乘拟合、低秩截断（PCR/ RRR）、余弦相似度与相对ℓ₂误差等评估指标，并与Identity、ReplaceMe等基线方法进行对比。

**📊 数据集**

使用DINO系列预训练ViT模型（DINOv2 ViT-L/14、ViT-G/14；DINOv3 ViT-L/16、ViT-H/16+）的原始图像，标定样本约1,000张。

**📈 对比分析**

与全局拟合（Full DMD）、锚点拟合（Anchored DMD）、无中间约束的端点拟合（ReplaceMe）以及Identity基线比较；在短剪切长度（p≤4）下，自治线性方法在余弦相似度上仅落后ReplaceMe ≤0.02；随剪切长度增长误差显著增大；局部重构良好但对下游最终表示的预测效果不佳，Identity基线在最终层往往表现更好。

**⚠️ 局限性**

仅在DINO ViT上验证，未探讨其他视觉或语言模型；自治线性假设在较深层或较长剪切长度失效；标定样本规模对下游性能影响未充分考虑；缺乏训练后修复步骤来恢复下游效果；未给出直接减少参数量或自动选择剪切点的机制。

---

## 559. Multi-Environment POMDPs with Finite-Horizon Objectives

**arXiv ID:** 2605.07537 | [PDF](https://arxiv.org/pdf/2605.07537v1)

**作者:** Léonard Brice `[一作]` (Institute of Science and Technology Austria), Stefanie Muroya `[通讯]` (Institute of Science and Technology Austria)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5081572933)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究多环境部分可观测马尔可夫决策过程(MEPOMDP)的有限时域价值计算问题，并提出相应算法；

**💡 创新点**

主要创新在于证明该问题与传统POMDP的复杂度相同（PSPACE‑complete），并给出既空间高效又实用的两种算法；

**🔧 技术方法**

利用贝叶斯更新的信念空间、混合策略的Carathéodory定理、以及凸包/点集剪枝技术实现动态规划与多信念更新；

**📊 数据集**

实验使用三类基准：扩展版Rock Sample、Robot Navigation、Identification (Friend or Foe)，每类均含不同规模的网格、状态数和初始状态数；

**📈 对比分析**

与已有工具（<cite>所提方法）对比，实验结果显示在最多千状态、horizon≤6的实例上，本方法平均快数倍，且能成功解决更多实例；

**⚠️ 局限性**

主要限制是对时间的依赖强（尤其是horizon），最优空间算法在实践中速度慢，实用算法在最坏情况下空间指数增长。

---

## 560. SGD for Variational Inference: Tackling Unbounded Variance via Preconditioning and Dynamic Batching

**arXiv ID:** 2605.07531 | [PDF](https://arxiv.org/pdf/2605.07531v1)

**作者:** Hippolyte Labarrière `[一作]`, Lorenzo Rosasco `[通讯]` (University of Genova)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文证明了在椭圆位置尺度族下负ELBO存在全局最优解，并在Blum-Gladyshev（BG）方差条件下给出了带动态批量与预处理的最小化投影随机梯度下降（PSGD）的有限时间与渐近收敛性保证，实验验证了其在高维贝叶斯逻辑回归中的有效性。

**💡 创新点**

创新点在于：①首次在BBVI中严谨证明负ELBO的最优解存在；②将BG方差条件与动态批量、预处理相结合，获得在椭圆位置尺度族下的完整收敛分析；③提供了对比实验中自适应预处理与批量策略对收敛速度与计算时间的量化提升。

**🔧 技术方法**

核心技术包括：椭圆位置尺度分布建模、BG方差条件下的梯度估计与预处理矩阵设计、投影随机梯度下降的动态批量调度、Lipschitz光滑性与强凸性分析。

**📊 数据集**

实验采用贝叶斯逻辑回归（高维Laplace搜索分布）作为数据集，主要用合成数据验证算法性能。

**📈 对比分析**

与未预处理PSGD、固定批量策略以及自适应Λ的两种实现进行了比较，实验显示自适应预处理与动态批量能显著降低负ELBO收敛误差并减少总梯度评估次数和CPU时间，收敛速度接近理论的O(1/√E)率。

**⚠️ 局限性**

局限性包括：仅关注预处理投影SGD，未涵盖自然梯度或基于评分的估计；收敛分析多依赖于凸/光滑假设，非凸情况仅在序列有界时成立；Lipschitz常数L及其它问题特定参数难以事先知晓，导致实际参数调优复杂。

---

## 561. BrickCraft: Visuomotor Skill Composition with Situated Manual Guidance for Long-Horizon Interlocking Brick Assembly

**arXiv ID:** 2605.07605 | [PDF](https://arxiv.org/pdf/2605.07605v1)

**作者:** Jichuan Yu `[一作]` (Tsinghua University), Changliu Liu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2542 | [OpenAlex ID](https://openalex.org/A5040156274)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了 BrickCraft 框架，能够将数字设计的拼块结构通过相对参考块的抽象与分解，利用可组合的视觉运动技能完成长时限拼块机器人装配。

**💡 创新点**

创新点在于：① 将装配步骤相对参考块表述，抽象为有限的可重用原语技能；② 通过“situated manual”将符号意图映射到实时视觉观测，提供空间导向；③ 将上述技能在执行管道中链式组合，实现可组合长时限装配。

**🔧 技术方法**

技术包括：DFS+assembly‑by‑disassembly 任务分解；YOLOv8‑seg 与 SAM2 的视觉分割与追踪；差分策略（Diffusion Policy）作为可视运动技能；背景 dimming 方式的 situated manual；强化学习细化、外力感知等。

**📊 数据集**

使用了约 692 条专家演示轨迹，涵盖 60+ 种不同结构的 LEGO Duplo 2x2 砖；在 50 张图像上人工标注用于训练 YOLOv8‑seg 与 SAM2；全部实验在 Kinova Gen3 + Robotiq 2F‑85 机器人平台上完成。

**📈 对比分析**

通过与 Goal Image‑Conditioned Diffusion Policy (GI‑DP) 与 Goal Image‑Conditioned π0.5 (GI‑π0.5) 两个基线对比，单步成功率达到 86.25%（240 次试验），在未见结构几乎无退化；长时限任务完成率在多种结构中保持高水平，尤其在完全未见的 Castle 结构上仍能完成。

**⚠️ 局限性**

局限性包括：仅针对 2x2 砖设计，难以直接扩展至更大或不同形状的砖；缺乏主动重定位、碰撞规避及结构变形补偿的自适应纠错机制；对高度悬臂结构的支持有限，未实现多臂或辅助支撑。

---

## 562. Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding

**arXiv ID:** 2605.07575 | [PDF](https://arxiv.org/pdf/2605.07575v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 563. A Syndrome-Space Approach to Proximity Gaps and Correlated Agreement for Random Linear Codes

**arXiv ID:** 2605.07595 | [PDF](https://arxiv.org/pdf/2605.07595v1)

**作者:** Chen Yuan `[一作]`, Ruiqi Zhu `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

在随机校验矩阵模型下，对随机线性码直接证明了接近度缺口（proximity gap）和相关一致性（correlated agreement）定理，避免了传统的列表解码（list decoding）作为主要工具。

**💡 创新点**

创新点在于：①引入syndrome‑space重构，将结构化的接近度问题转化为syndrome空间中的球与线（或空间、曲线）的交集问题；②提出 witness‑matrix 以及秩递减（rank‑reduction）机制，实现对结构化错误集合的直接控制；③通过上述方法获得在大符号场和常数符号场下更优的半径上限（ρ < 1–R–ε）和更小的符号大小要求。

**🔧 技术方法**

核心技术包括：syndrome‑space 视角、witness‑matrix 构造、确定性秩递减 lemma、概率计数与 union bound、以及随机校验矩阵的均匀分布性质。

**📊 数据集**

该工作为纯理论分析，无实验数据集；所研究对象为在随机校验矩阵模型生成的随机线性码。

**📈 对比分析**

与以往依赖列表解码的结果对比，本文在大符号场下把接近度从 ρ < 1–R–2ε 提升到 ρ < 1–R–ε，常数符号场下在更小的 q（仅需 q ≥ (2/ε)^2/ε）即可实现更大可接受半径，整体性能在接近率和符号大小上都有显著改进。

**⚠️ 局限性**

局限性：①只适用于随机校验矩阵模型，缺乏构造性编码方案；②对极小 ε 的极限仍存在 ε 的损失；③结果主要针对随机线性码，尚未证明能否直接推广到特定结构码；④对曲线型结构的证明虽然完成，但其参数上仍有进一步提升空间。

---

## 564. Parallel Lifted Planning via Semi-Naive Datalog Evaluation

**arXiv ID:** 2605.07584 | [PDF](https://arxiv.org/pdf/2605.07584v1)

**作者:** Dominik Drexler `[一作]` (Linköping University), Jendrik Seipp `[通讯]` (Linköping University)

**通讯引用:** 601 | [OpenAlex ID](https://openalex.org/A5031089257)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于半否定Datalog评估的提升规划器，并通过规则级和grounding级两层并行加速。

**💡 创新点**

创新点是使用-边锚定的团枚举来避免重复产生地面规则实例，并实现同步半否定执行与两层并行架构。

**🔧 技术方法**

采用Datalog、半否定评估、k-分区团枚举、同步并行、规则级和grounding级并行等技术。

**📊 数据集**

使用Autoscale Agile (AS) 和 Hard-To-Ground (HTG) 两个经典规划基准集。

**📈 对比分析**

与Fast Downward和Powerlifted在基于FF启发式的贪心最佳优先搜索下对比，Tyr在HTG上覆盖率最高，8核时可达6倍加速，平均加速2–4倍。

**⚠️ 局限性**

局限在于当前的grounding级划分策略效果有限，负载均衡不够细粒度，难以充分利用细粒度的Datalog并行；且在某些域如Blocksworld-Large仍无显著优势。

---

## 565. Probabilistic Object Detection with Conformal Prediction

**arXiv ID:** 2605.07549 | [PDF](https://arxiv.org/pdf/2605.07549v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 566. HexiSeq: Accommodating Long Context Training of LLMs over Heterogeneous Hardware

**arXiv ID:** 2605.07569 | [PDF](https://arxiv.org/pdf/2605.07569v1)

**作者:** Yan Liang `[一作]` (Hong Kong University of Science and Technology), Chuan Wu `[通讯]` (University of Hong Kong)

**通讯引用:** 11298 | [OpenAlex ID](https://openalex.org/A5012597518)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套支持异构 GPU 集群的长上下文 LLM 训练系统，能够在 Context Parallelism（CP）和 Head Parallelism（HP）层面实现非对称调度。

**💡 创新点**

创新点在于：①把异构硬件特性直接映射到 CP/HP 调度；②设计了可变 A2A 分组、非均匀序列切分和头部分配的调度抽象；③提出了基于计算、内存与通信约束的优化模型和三阶段层次调度器。

**🔧 技术方法**

使用的技术包括：FlashAttention、Ring Attention、A2A collectives（Ulysses 方式）、自定义异构 A2A 以及子环 KV 交换；调度器基于分析模型和坐标下降实现；系统运行在 PyTorch + NCCL 上。

**📊 数据集**

使用 OpenWebText2 文档作为训练数据，经过 Megatron‑LM 预处理为固定长度样本，采用 GPT‑style 解码器模型（3B、7B、13B、70B）。

**📈 对比分析**

与 USP、Ulysses、Ring Attention 这三种基线在异构测试床（H100+ A100）和大规模模拟（32–128 GPU）上进行对比；在测试床上平均提升 1.11×（最高 1.19×），在模拟中平均提升 1.36×，对齐同等 FLOP 的同构集群仅差 0.5% 左右。

**⚠️ 局限性**

局限性包括：仅在 GPT‑style 仅解码器模型上验证；调度需要精确的设备性能和网络拓扑信息，适配新硬件可能需要重新校准；对极端异构（如多代 L40S）和大规模（>1024 GPU）生产环境的实验尚未展开。

---

## 567. Ensemble Distributionally Robust Bayesian Optimisation

**arXiv ID:** 2605.07565 | [PDF](https://arxiv.org/pdf/2605.07565v1)

**作者:** Tigran Ramazyan `[一作]` (HSE University), Denis Derkach `[通讯]` (HSE University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在分布不确定性下提出 Ensemble Distributionally Robust Bayesian Optimization (EDRBO)，实现连续上下文下的可计算的分布鲁棒贝叶斯优化。

**💡 创新点**

创新点在于结合多模态 GP 集成、Wasserstein 距离下的鲁棒采集函数、Bures‑Wasserstein 经验半径估计，获得理论子线性 regret 上界，并在连续上下文环境下保持计算可行性。

**🔧 技术方法**

采用 Gaussian Process 集成、Wasserstein DRO、信息增益、Bures‑Wasserstein barycenter、UCB 采集函数以及基于 2‑Wasserstein 距离的半径估计等技术。

**📊 数据集**

实验使用多种合成基准（Ackley、Three‑Hump Camel、Six‑Hump Camel、Hartmann、Newsvendor、Modified Branin）和真实世界金融组合优化数据集。

**📈 对比分析**

与 UCB、SBO‑KDE、DRBO‑KDE、DRBO‑MMD、StableOpt、WDRBO 等方法比较，EDRBO 在高非凸多峰问题上显著降低累计 regret，平均性能在 Branin 与组合优化问题上接近最佳。

**⚠️ 局限性**

局限性包括对光滑目标的假设，处理强非线性/非可分决策‑上下文交互时性能下降，且在非平稳或有限数据场景下可能失效；同时依赖于 Wasserstein 近似与聚类参数的选择。

---

## 568. Why Self-Inconsistency Arises in GNN Explanations and How to Exploit It

**arXiv ID:** 2605.07527 | [PDF](https://arxiv.org/pdf/2605.07527v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 569. FS-I2P:A Hierarchical Focus-Sweep Registration Network with Dynamically Allocated Depth

**arXiv ID:** 2605.07607 | [PDF](https://arxiv.org/pdf/2605.07607v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 570. Multimodal Stepwise Clinically-Guided Attention Learning for Pathological Complete Response Prediction in Breast Cancer

**arXiv ID:** 2605.07561 | [PDF](https://arxiv.org/pdf/2605.07561v1)

**作者:** Alice Natalina Caragliano `[一作]` (Università Campus Bio-Medico di Roma), Paolo Soda `[通讯]` (Università Campus Bio-Medico di Roma)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了多模态逐步临床引导注意力学习框架，用于乳腺DCE‑MRI预测病理完全反应（pCR）

**💡 创新点**

通过三步训练，先学习全局影像特征，再用病灶分割指导注意力，最后融合临床变量实现生物学合理且可解释的模型

**🔧 技术方法**

基于3D Swin UNETR Transformer编码器，早期与晚期注意力模块，局部定位损失和多模态融合

**📊 数据集**

利用多中心MAMA‑MIA数据集，包含DUKE、I‑SPY1、I‑SPY2、NACT 四个公开子集，总计1485例乳腺癌患者

**📈 对比分析**

采用留一数据集外验证，与单模态和传统早晚期融合基线对比；在三组外部测试中，Step3方案在敏感度、平衡准确率和AUC上均显著提升，取得最佳平衡

**⚠️ 局限性**

对极小样本外部数据表现仍有限，且注意力在病灶周围仍有扩散，未完全精准聚焦

---

## 571. CCX: Enabling Unmodified Intel SGX Applications on Arm CCA

**arXiv ID:** 2605.07548 | [PDF](https://arxiv.org/pdf/2605.07548v1)

**作者:** Matti Schulze `[一作]` (FAU), Felix Freiling `[通讯]` (FAU)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

设计并实现了 CCX 框架，允许现有 Intel SGX 应用在 Arm CCA 上运行，无需改动源代码。

**💡 创新点**

创新点在于将 SGX 微指令映射到 Arm CCA firmware，并通过多 GPT 与 AEX‑Notify 机制重现 SGX 的 enclave 生命周期，实现完全兼容的跨平台迁移。

**🔧 技术方法**

采用了 EL3 firmware (TF‑A) 中的 CCX‑Core、PMU register 跳转、重新实现的 SGX SDK/内核模块/编译器，并利用 GPT、CPEM、AEX‑Notify 等 Arm CCA 功能。

**📊 数据集**

使用 QEMU Arm 9.2.3 模拟环境和 Nitrogen8M 开发板，对多款开源 SGX 应用（kmeans、TaLoS、SQLite、TrustFL、secure‑analytics‑sgx）以及 NBench 基准进行评测。

**📈 对比分析**

通过与原生 SGX、NanoZone、HiveTEE 等现有 Arm enclave 方案对比，CCX 在大多数基准下实现近原生性能，平均开销约 1% 左右，显著优于 200%+ 的其他方案。

**⚠️ 局限性**

主要限制包括缺乏公开 Arm CCA 硬件导致评估依赖模拟，远程身份验证支持尚未实现，EL3 firmware 扩展略微增加 TCB，且侧信道攻击防护仍需进一步验证。

---

## 572. Deadline-Driven Hierarchical Agentic Resource Sharing for AI Services and RAN Functions in AI-RAN

**arXiv ID:** 2605.07547 | [PDF](https://arxiv.org/pdf/2605.07547v1)

**作者:** Haiyuan Li `[一作]` (University of Bristol), Dimitra Simeonidou `[通讯]` (University of Bristol)

**通讯引用:** 8759 | [OpenAlex ID](https://openalex.org/A5030580652)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 HAF（Hierarchical Agentic Framework）实现 AI-RAN 中 AI 服务与 RAN 功能的异步、deadline 驱动的资源共享，分层处理慢速迁移与高速 GPU/CPU 分配。

**💡 创新点**

创新点包括：1）将资源分配分为两层——LLM 代理负责长周期实例迁移，闭式凸优化负责瞬时分配；2）引入预测 critic 对迁移延迟与 SLO 效果进行评估，减少不必要迁移；3）将 RAN 的硬实时约束与 AI 需求融合到单一优化框架。

**🔧 技术方法**

技术实现：大语言模型（如 Qwen3:32B）作为慢速迁移决策器；基于凸优化的闭式分配算法；两层监督的 MLP 预测 critic；事件驱动的多资源调度。

**📊 数据集**

使用的数据集：Azure LLM 推理请求轨迹（用于 AI 服务负载），以及合成的 RAN 负载（URLLC、eMBB 难度），并在六节点异构集群上进行离散事件模拟。

**📈 对比分析**

与五个基线（静态、Round‑Robin、Lyapunov、Game Theory、CAORA）对比，HAF 在 ρ≈1.0 时实现 90.0% 的整体 SLO 满足率（相对最佳基线提升 20.5%），AI 请求满足率从 51% 提升至 85.3%；在不同负载下保持优势，直至容量饱和点。

**⚠️ 局限性**

局限性：1）依赖 LLM 与 critic 的训练和推理成本；2）迁移延迟对极端实时场景仍是瓶颈；3）仅在仿真环境验证，缺乏真实网络实验；4）当整体负载超过集群有效容量时，优化空间受限。

---

## 573. Revisiting Transformer Layer Parameterization Through Causal Energy Minimization

**arXiv ID:** 2605.07588 | [PDF](https://arxiv.org/pdf/2605.07588v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 574. Tracing the Arrow of Time: Diagnosing Temporal Information Flow in Video-LLMs

**arXiv ID:** 2605.07568 | [PDF](https://arxiv.org/pdf/2605.07568v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 575. Mathematical Reasoning via Intervention-Based Time-Series Causal Discovery Using LLMs as Concept Mastery Simulators

**arXiv ID:** 2605.07600 | [PDF](https://arxiv.org/pdf/2605.07600v1)

**作者:** Tsuyoshi Okita `[一作]` (Kyushu Institute of Technology), Tsuyoshi Okita `[通讯]` (Kyushu Institute of Technology)

**通讯引用:** 2608 | [OpenAlex ID](https://openalex.org/A5026921207)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将LLM自身视为“do”操作的模拟器，利用概念级prompt干预评估因果效应，构建Interventional Capability Probe（ICP）并实现Causal Knowledge Activation（CKA），从而显著提升LLM在数学推理任务中的准确率。

**💡 创新点**

创新点在于将prompt干预正式化为因果干预，区分“知道”与“能用”两种知识状态，并通过ICP诊断因果相关概念，结合因果bandit实现针对性知识激活。

**🔧 技术方法**

使用的技术包括LLM-as-Interventional-Simulator、SVAR结构方程模型、MCTS+Causal UCB、BM25检索以及多镜头恢复等。

**📊 数据集**

所用数据集主要包括Omni-MATH-Rule、Omni-MATH整体、MATH-500、AIME 2024-2026、MathArena以及MATH训练集中的67道验证题。

**📈 对比分析**

与o1-mini、Qwen-72B、rStar-Math等基准相比，CIKA在Omni-MATH-Rule上取得69.7%（比o1-mini提升约8.2个百分点），在GSM8K上达到97.2%，在MathArena最终答案赛题上达46.2%，表现出显著性能提升。

**⚠️ 局限性**

主要局限在于计算成本高（每题约110次LLM调用）、prompt干预等价于do操作尚未完全证明，以及当前ICP仅为单概念干预，未考虑概念交互效应。

---

## 576. Beyond Defenses: Manifold-Aligned Regularization for Intrinsic 3D Point Cloud Robustness

**arXiv ID:** 2605.07590 | [PDF](https://arxiv.org/pdf/2605.07590v1)

**作者:** Pedro Alonso `[一作]` (Southwest Jiaotong University), Tianrui Li `[通讯]` (Southwest Jiaotong University)

**通讯引用:** 25159 | [OpenAlex ID](https://openalex.org/A5070559820)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并实现了基于流形对齐的点云分类框架 MAPR，显著提升了模型对抗攻击的鲁棒性。

**💡 创新点**

通过将点云的内在几何特征与一致性正则化结合，直接约束潜在空间与输入流形的几何对齐，而非单纯依赖输入增强或对抗训练。

**🔧 技术方法**

利用曲率和多尺度扩散的内在特征增强、对称 KL 互信息一致性损失以及 Lipschitz 正则化，配合 k‑NN 图和随机游走拉普拉斯实现。

**📊 数据集**

在合成 CAD 数据集 ModelNet40 和真实扫描数据集 ScanObjectNN 上进行实验评估。

**📈 对比分析**

与原始模型、对抗训练以及 SOR 等防御方法比较，MAPR 在五大网络上平均提升鲁棒性约 20.02%（ModelNet40）和 8.58%（ScanObjectNN），且不需对抗样本即可获得。

**⚠️ 局限性**

仍对 Add‑100 等点增攻击鲁棒性有限；训练时计算开销约 2–2.5 倍，且在 ScanObjectNN 上部分骨干的清洁准确率略有下降。

---

## 577. How to utilize failure demo data?: Effective data selection for imitation learning using distribution differences in attention mechanism

**arXiv ID:** 2605.07560 | [PDF](https://arxiv.org/pdf/2605.07560v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 578. VIMCAN: Visual-Inertial 3D Human Pose Estimation with Hybrid Mamba-Cross-Attention Network

**arXiv ID:** 2605.07552 | [PDF](https://arxiv.org/pdf/2605.07552v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 579. Why do Large Language Models Fail in Low-resource Translation? Unraveling the Token Dynamics of Large Language Models for Machine Translation

**arXiv ID:** 2605.07533 | [PDF](https://arxiv.org/pdf/2605.07533v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 580. GESR: Graph-Based Edge Semantic Reconstruction for Stealthy Communication Detection with Benign-Only Training

**arXiv ID:** 2605.07536 | [PDF](https://arxiv.org/pdf/2605.07536v1)

**作者:** Henghui Xu `[一作]` (Xi'an Jiaotong University), Xiaobo Ma `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 790 | [OpenAlex ID](https://openalex.org/A5003951925)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对流日志下的隐蔽C2通信，提出基于图的边语义重构框架，在善意训练环境下实现主机级异常检测。

**💡 创新点**

创新点在于：① 用结构条件的边语义重构代替传统节点或全局重构；② 通过MAD校准提升重构误差的可比性；③ 设计边到主机的加权聚合机制，使得边异常信息能有效转化为主机风险评分。

**🔧 技术方法**

技术实现包括：GINEConv图神经网络进行边感知编码；多任务边语义重构（连续属性回归 + 离散端口分类）；MAD校准与鲁棒阈值化；以及多种边到主机聚合算子（mean、q90、max、top‑k mean）。

**📊 数据集**

使用公开流日志数据集CTU‑13和CICIDS2017进行实验，按时间顺序划分善意训练与测试集。

**📈 对比分析**

与Isolation Forest、Autoencoder、GraphSAGE+IF、Kitsune、Anomal‑E等基线比较，本文方法在两数据集均获得最高ROC‑AUC、PR‑AUC，并在5%FPR下TPR最高（CTU‑13 0.9647，CICIDS2017 0.8569），整体性能显著优于现有方法。

**⚠️ 局限性**

局限性包括：仅在离线批量评估，缺乏实时在线推理与增量更新；主机级评价主导，未给出边级检测结果；聚合策略对不同数据集有差异；对跨窗口慢速C2信号检测不足；需进一步研究概念漂移下的自适应校准。

---

## 581. Resilience of IEC 61850 Sampled Values-Based Protection Systems Under Coordinated False Data Injections

**arXiv ID:** 2605.07535 | [PDF](https://arxiv.org/pdf/2605.07535v1)

**作者:** Denys Mishchenko `[一作]` (Norwegian University of Science and Technology), Laszlo Erdodi `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文在工业级 PHIL 测试平台上，针对 IEC 61850 采样值（SV）协议的贝叶级（bay-level）防护系统，设计并验证了多向、物理一致的伪造数据注入（FDIA）攻击，评估了其对距离保护等关键功能的影响。

**💡 创新点**

创新点包括：① 采用多参数协调、保持电气物理一致的 FDIA 模式；② 在 PHIL 环境中对真实 IED（如 Siemens SIPROTEC）进行闭环实验验证；③ 提出基于独立二进制信道和遥测保护交叉验证的抗攻击防护扩展，增强贝叶级的韧性。

**🔧 技术方法**

使用技术包括：PHIL 仿真与硬件互连、工业 IED（MU 与保护继电器）、自研 C++ 包捕获与伪造程序、SV 与 PTP 协议分析、基于算法的多向攻击实现、以及在保护继电器固件中嵌入跨信道决策逻辑。

**📊 数据集**

数据集主要来源于 IEEE 9 节点系统中 6‑7 节点的仿真电流/电压波形（由 Simulink 生成并通过电压/电流放大器注入物理电路），以及在实验过程中实时捕获的 SV 流和 PTP 交换帧；未使用公开公共数据集。

**📈 对比分析**

评估方法为对四种攻击场景（伪造短路强制跳闸、伪装正常、并行重放、PTP 伪造）进行实验，并通过保护继电器的跳闸状态、误报/漏报计数以及是否触发安全报警等指标比较。实验表明，攻击在保持时间和帧完整性的前提下能够在不产生明显异常的情况下成功导致错误跳闸、隐藏真实故障或阻止保护动作；提出的韧性方案在大多数场景下恢复了正确的跳闸或至少防止了错误动作，显示出显著的性能优势。

**⚠️ 局限性**

局限性包括：① 仅在单个贝叶、单一子站级别进行实验，缺乏跨站点或完整电网的验证；② 试验基于 Siemens SIPROTEC 设备，其他厂商或混合设备的适用性尚未验证；③ 对于更复杂的多向攻击（如多点同步、跨 VLAN 的隐蔽攻击）实验未覆盖；④ 防护扩展主要依赖硬件层的独立信道，未对软件级检测方法进行深入评估。

---

## 582. Search-based Robustness Testing of Laptop Refurbishing Robotic Software

**arXiv ID:** 2605.07530 | [PDF](https://arxiv.org/pdf/2605.07530v1)

**作者:** Erblin Isaku `[一作]` (Simula Research Laboratory), Francois Picard `[通讯]` (Danish Technological Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于搜索的鲁棒性测试方法（PROBE），用于在笔记本电脑翻新机器人软件中发现并分析目标检测模型的最小局部扰动导致的失败情况。

**💡 创新点**

创新点在于：①将多目标优化（NSGA-II）与结构化、最小化的局部扰动结合，以系统性探索鲁棒性弱点；②通过对失败类型（误检、误分、误定位、模糊）进行细粒度分类与统计；③利用变形关系（MR）评估非失败扰动下模型稳定性；④展示扰动跨模型可迁移性。

**🔧 技术方法**

技术包括：多目标进化算法NSGA-II、局部高斯加色扰动编码、目标检测模型评估（YOLOv11）、Hypervolume评价指标、统计检验（Wilcoxon、Mann‑Whitney）以及变形关系分析。

**📊 数据集**

使用DTI提供的三种工业级YOLOv11模型（origSCDM、noscrewSCDM、fixtureSCDM）和相应的训练/评估图像集（共802张），以及一套独立的25张测试图像用于跨模型迁移评估。

**📈 对比分析**

与随机搜索（RS）比较时，PROBE在Hypervolume、失败率和扰动幅度三项指标上均显著优于RS（p<0.001，效应量大），失败率提升约3–7倍，产生的扰动幅度更小；在跨模型迁移实验中，PROBE的失败率也远高于RS，表明扰动更具通用性。

**⚠️ 局限性**

局限性包括：①实验仅覆盖三种YOLOv11模型，未验证对其他架构或任务的适用性；②扰动空间受预设参数范围限制，可能漏掉更大幅度或非局部扰动导致的失败；③仅评估了单一工业场景，外部有效性未知；④实现中对搜索参数的选择依赖实验调优，可能影响结果的一致性。

---

## 583. SimCT: Recovering Lost Supervision for Cross-Tokenizer On-Policy Distillation

**arXiv ID:** 2605.07711 | [PDF](https://arxiv.org/pdf/2605.07711v1)

**作者:** Jie Sun `[一作]` (University of Science and Technology of China), Xiang Wang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 26710 | [OpenAlex ID](https://openalex.org/A5100732436)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种跨分词器的 on‑policy distillation 方法 SimCT，通过构造最小可对齐单位的监督空间，恢复教师与学生在不同 tokenizer 下丢失的监督信号；

**💡 创新点**

核心创新在于将监督空间从仅共享词表扩展到包含最小可对齐的多 token 单位，既保持 OPD 的损失形式不变，又能显著提升分词器不一致时的教师反馈利用率；

**🔧 技术方法**

采用了对齐 tokenizer 的最小可对齐单位（minimal aligned units）构造监督空间，并对其进行长度归一化（1/k log‑likelihood）得到分数，再将两侧的概率分布映射到该空间做逆 KL 损失；实验中使用 SFT 预热 + OPD 循环，并与 SimpleOPD、DSKD、ALM、GOLD 等基线对比；

**📊 数据集**

实验数据集包括数学推理任务（GSM8K、MATH‑500）和代码生成任务（MBPP、LiveCodeBench‑v6），SFT 训练采用教师生成的多来源数据（GSM8K、Orca‑Math、OpenMathInstruct‑1、MATH、OpenCodeInstruct、KodCode、TACO、CodeContests 等）；

**📈 对比分析**

在所有教师‑学生对（Qwen2.5‑7B→Phi‑4‑mini、Qwen→Gemma‑2‑2B‑IT、Phi→Gemma）和四大基准上，SimCT 对比 SimpleOPD 和其他跨分词器基线均取得显著提升，平均 Pass@1 提升约 6%（相对 SFT），计算成本略高于 SimpleOPD，但仍低于更复杂的基线；

**⚠️ 局限性**

局限性包括：仅在数学和代码两类任务、三对教师‑学生组合上评估；方法依赖 BPE‑style 分词器，非同类分词器可能需要改造；1/k 长度归一化仅为经验选择，缺乏系统评估；目前仅适用于白盒教师（可获取 next‑token 分布），不适用于仅返回采样或标量反馈的黑盒教师。

---

## 584. Fortifying Time Series: DTW-Certified Robust Anomaly Detection

**arXiv ID:** 2605.07690 | [PDF](https://arxiv.org/pdf/2605.07690v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 585. GASim: A Graph-Accelerated Hybrid Framework for Social Simulation

**arXiv ID:** 2605.07692 | [PDF](https://arxiv.org/pdf/2605.07692v1)

**作者:** Xuan Zhou `[一作]` (University of Science and Technology of China), Wu Liu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 471450 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了GASim框架，结合图加速的核心记忆检索（GOM）、并行意见更新（GMP）以及熵驱动的核心划分（EDG）实现大规模社交模拟

**💡 创新点**

创新点在于用图结构替代传统LLM检索，使用图注意力网络并行处理普通代理，并通过信息熵动态识别意见领袖

**🔧 技术方法**

采用图优化记忆、图注意力网络、熵驱动划分和LLM生成/评分等技术

**📊 数据集**

使用Twitter/Weibo公开数据的政治、商业、教育三大主题数据集，共计约35k用户与约167k推文

**📈 对比分析**

与HiSim及非混合基线对比，GASim在核心/普通代理执行时间分别提升约16.4×和27.5×，总体速度提升9.94×，token消耗降低至20%以内，并在四项趋势对齐指标上均优于所有对照方法

**⚠️ 局限性**

局限在于LLM生成的文本缺乏真实性、可能带偏差，以及仅关注文本交互，未纳入多模态信息

---

## 586. Structured Coupling for Flow Matching

**arXiv ID:** 2605.07676 | [PDF](https://arxiv.org/pdf/2605.07676v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 587. Characterizing and Mitigating False-Positive Bug Reports in the Linux Kernel

**arXiv ID:** 2605.07678 | [PDF](https://arxiv.org/pdf/2605.07678v1)

**作者:** Jiashuo Tian `[一作]` (Tianjin University), Junjie Chen `[通讯]` (Tianjin University)

**通讯引用:** 6102 | [OpenAlex ID](https://openalex.org/A5100365536)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对 Linux 内核的误报 bug 进行了系统实验，构建并手工标注了 2006 条闭合报告（1,509 条真实 bug 与 497 条误报），并对其工时成本、组件分布和根因进行量化分析。

**💡 创新点**

创新点在于首次全面量化内核误报的成本与分布，提出了针对 Linux 生态的误报根因分类，并验证了检索增强生成（RAG） LLM 在误报识别与解释中的可行性。

**🔧 技术方法**

主要技术包括深度预标注+人工校验的两阶段标注流程、基于 DeepSeek‑V3 的多种提示策略（zero‑shot、few‑shot、CoT、RAG）以及对 LLM 生成解释的可解释性评估。

**📊 数据集**

使用的数据集为 2006 条 Bugzilla 与 Syzkaller 的闭合 bug 报告，其中 497 条被判定为误报，覆盖了 2020‑2025 年的内核版本。

**📈 对比分析**

与传统逻辑回归与 kNN 基线相比，检索增强生成 RAG 在 81% 准确率、91% 召回率、88% F1 上实现最佳效果；其它提示策略（zero‑shot、few‑shot、CoT）则表现略逊；所有方法均在 1.8 s 内完成一次推理。

**⚠️ 局限性**

主要限制包括：仅涵盖 Bugzilla 与 Syzkaller 的报告，可能不代表其他内核或系统的误报情况；标注仍有主观偏差；LLM 的解释偶尔表层化，缺乏深度验证。

---

## 588. Multi-Dimensional Evaluation of LLMs for Grammatical Error Correction

**arXiv ID:** 2605.07635 | [PDF](https://arxiv.org/pdf/2605.07635v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 589. Direction-Preserving Number Representations

**arXiv ID:** 2605.07662 | [PDF](https://arxiv.org/pdf/2605.07662v1)

**作者:** Bardia Zadeh `[一作]` (Imperial College London), George A. Constantinides `[通讯]` (Imperial College London)

**通讯引用:** 6210 | [OpenAlex ID](https://openalex.org/A5029829952)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究并量化了产品结构向量码（即由标量字母集构成的向量量化）的方向覆盖性能，证明其在高维下与球面码相比存在严格的子最优性，并对常见的浮点、定点及补码格式进行了类似评估。

**💡 创新点**

创新点在于：①建立了一个几何框架和一组严格的上、下界，利用谐波证人和 Wyner 定理证明了产品码在任何固定字母集大小下都无法与球面码匹配；②证明了标准格式在方向保持方面相较于最优产品码存在量化的子最优性；③通过数值优化得到了一种全新 4 位标量字母表，且实验证明其在多维度下的方向误差最小。

**🔧 技术方法**

技术手段包括：几何分析与极限证明（谐波证人、球面覆盖半径下界、Wyner 上界）；符号化证明（Lean 形式化证明）；实验评估（通过 10⁶ 随机单位向量采样近似最坏角误差）；数值优化（差分进化 + Powell 线搜索）来寻找最优标量字母表。

**📊 数据集**

使用的是合成数据：在单位球面上均匀采样的随机向量，采用 10⁶ 个样本对 4 位字母表在各维度（4,8,16,32,64）下的最坏角误差进行评估；没有使用公开机器学习数据集。

**📈 对比分析**

比较方法：对每个字母表（优化表、E2M1、INT/E1M2、E3M0）在相同维度和字母集下，采样 10⁶ 单位向量，计算其最近向量码点的角误差并取最大值；实验结果显示：优化表在所有维度下均表现最佳，E2M1 的误差仅略高于优化表，INT/E1M2 与 E3M0 的误差显著更大；在 16 维下，E2M1 与优化表在对数空间中呈现几乎等距的线性关系。

**⚠️ 局限性**

限制：①产品结构向量码在高维下不可避免地比球面码差，无法达到理论上最优的方向覆盖；②标准格式虽然实用但在方向保持方面已被证明为子最优；③实验评估仅基于随机采样，无法保证绝对最坏误差；④数值优化只在 4 位字母集上完成，扩展到更宽字母集时计算成本显著上升。

---

## 590. Faster Deterministic Streaming Vertex Coloring

**arXiv ID:** 2605.07644 | [PDF](https://arxiv.org/pdf/2605.07644v1)

**作者:** Shiri Chechik `[一作]` (Tel Aviv University), Tianyi Zhang `[通讯]` (Nanjing University)

**通讯引用:** 8641 | [OpenAlex ID](https://openalex.org/A5100437457)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种确定性半流式图顶点着色算法，在 O(√(logΔ)) 次遍历内实现 O(Δ) 颜色。

**💡 创新点**

首次突破了之前确定性流式算法在子对数次遍历内得到线性 Δ 颜色的限制，获得了更低的遍历复杂度。

**🔧 技术方法**

采用几乎 k‑wise 独立哈希、频繁元素检测、顶点与颜色分区以及线性估计等技术。

**📊 数据集**

未使用实验数据集，纯理论分析与证明。

**📈 对比分析**

通过理论分析与之前随机算法对比，证明在 Õ(n) 位空间下，确定性算法在 O(√(logΔ)) 次遍历内即可得到 (1+η)Δ 颜色，显著优于之前 O(logΔ·loglogΔ) 或 O(logΔ) 次遍历的结果。

**⚠️ 局限性**

仍受限于 √(logΔ) 的遍历次数，尚未实现 O(loglogΔ) 或更低；对 (Δ+1)‑着色和最大独立集等相关问题的确定性流式处理仍是未解决的挑战。

---

## 591. EggHand: A Multimodal Foundation Model for Egocentric Hand Pose Forecasting

**arXiv ID:** 2605.07642 | [PDF](https://arxiv.org/pdf/2605.07642v1)

**作者:** Jaeyoung Choi `[一作]` (Daegu Gyeongbuk Institute of Science and Technology), Daehee Park `[通讯]` (Daegu Gyeongbuk Institute of Science and Technology)

**通讯引用:** 383 | [OpenAlex ID](https://openalex.org/A5101688558)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出EggHand框架，利用Egocentric Video-Text Encoder与Vision‑Language‑Action（VLA）解码器联合预测第一人称视角下的3D手姿并支持语言条件控制。

**💡 创新点**

将专门为第一人称视频预训练的EgoVideo编码器与动作结构化的GR00T解码器跨模态融合，并通过几何感知的绝对、相对、配对损失提升姿态稳定性与可控性。

**🔧 技术方法**

使用EgoVideo视觉‑文本编码器、GR00T VLA解码器、跨模态融合层、轻量化适配器、绝对/相对/配对损失以及AdamW优化器。

**📊 数据集**

在真实世界的EgoExo4D数据集上进行训练与评估，该数据集提供10fps RGB视频与同步的3D手关节标注。

**📈 对比分析**

与Static、CVM、EgoH4等基线在ADE、FDE、MPJPE、MPJPE‑F四项指标下对比，EggHand在MPJPE与FDE上明显优于EgoH4（MPJPE↓34.5%，FDE↓18.6%），ADE略高但仍保持竞争水平。

**⚠️ 局限性**

模型仍需依赖离线手姿估计，缺乏端到端训练；在极端遮挡或高噪声场景下预测质量下降；预测时长有限，未来可扩展至更长时间序列。

---

## 592. Is She Even Relevant? When BERT Ignores Explicit Gender Cues

**arXiv ID:** 2605.07622 | [PDF](https://arxiv.org/pdf/2605.07622v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 593. Safe, or Simply Incapable? Rethinking Safety Evaluation for Phone-Use Agents

**arXiv ID:** 2605.07630 | [PDF](https://arxiv.org/pdf/2605.07630v1)

**作者:** Zhengyang Tang `[一作]` (Tencent Hunyuan), Han Hu `[通讯]` (Tencent Hunyuan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过构建 PhoneSafety 基准，在 700 个真实手机交互中的安全关键时刻，对手机使用代理在下一步动作的选择进行评估，区分安全行动、危险行动与无用行动三种结果。

**💡 创新点**

创新点在于揭示“无害结果”可能源于安全判断或失能两种根本原因，并提出三分类评估框架与“失能信号”概念，突破传统任务/episode 级别的安全评估盲点。

**🔧 技术方法**

主要技术包括：收集真实 Android 轨迹、人工标注安全与危险决策边界、基于协议的动作匹配规则以及统计分析（Spearman、Pearson 等）验证失能信号与能力的关联。

**📊 数据集**

使用了基于 130+ 中国手机应用的 4,512 条真实轨迹数据，最终筛选出 700 个安全关键时刻作为评估集，另外包含 7,168 步的普通电话使用评估集。

**📈 对比分析**

对八个代表性手机使用代理（Gemini 3.1 Pro、Seed 2.0 Pro、Claude Opus 4.6 等）进行比较，发现普通能力与安全选择相关性仅为 Spearman ρ=0.515，且失能率在不同场景与协议下保持稳定，说明安全评估不能仅凭整体失误率评判。

**⚠️ 局限性**

局限性包括：评估仅基于中文 Android 生态的离线样本，未覆盖跨平台和多语言情况；评估侧重单步决策，未考虑长序列中的安全策略演化；以及对“失能”分类的定义仍是操作层面的标签，未深入模型内部机制。

---

## 594. Post-training makes large language models less human-like

**arXiv ID:** 2605.07632 | [PDF](https://arxiv.org/pdf/2605.07632v1)

**作者:** Marcel Binz `[一作]` (Helmholtz Munich), Eric Schulz `[通讯]` (Helmholtz Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建大规模的行为实验转录数据集 Psych-201，系统评估大型语言模型（LLM）对人类行为的模拟能力；

**💡 创新点**

创新点在于：①首次以 3.5 倍规模的 Psych-201 数据集覆盖多种实验范式与跨文化人群；②揭示了“后训练”（instruction‑tuning、reasoning、vision 等）普遍降低 LLM 与人类行为的一致性，且该误差随模型代数递增；③验证了“人格诱导”（persona‑induction）在个体层面几乎无效；

**🔧 技术方法**

技术手段包括：使用负对数似然（NLL）衡量模型预测与真实人类回应的贴合度；对不同后训练目标和模型家族计算 Cohen’s d 以量化误差；对实验领域进行聚类分析以探究误差结构；使用对比实验设计评估人格诱导效果；

**📊 数据集**

使用的数据集为 Psych-201，包含 208,021 名参与者、25,906,599 条行为回应以及数百种实验，覆盖语言、心理学、发展与跨文化研究；

**📈 对比分析**

比较方法为基线（预训练基模型）与后训练模型在同一实验序列上的 NLL 进行对比，计算效应大小；结果显示：基模型整体与人类行为更贴合，后训练模型在所有家族和规模上均出现更高 NLL，误差幅度在最新版 Qwen3.x 与 Llama3.x 中显著扩大；

**⚠️ 局限性**

局限性包括：后训练误差的根源尚不完全明了，可能与当前 RLHF 或指令调优技术缺失；人格诱导在个体层面效果微弱，说明仅靠元数据无法捕捉个体差异；数据集仍为语言转录，缺乏多模态或长期互动场景；

---

## 595. Future Validity is the Missing Statistic: From Impossibility to $Φ$-Estimation for Grammar-Faithful Speculative Decoding

**arXiv ID:** 2605.07698 | [PDF](https://arxiv.org/pdf/2605.07698v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 596. The AI-Native Large-Scale Agile Software Development Manifesto

**arXiv ID:** 2605.07717 | [PDF](https://arxiv.org/pdf/2605.07717v1)

**作者:** Ricardo Britto `[一作]` (Ericsson), Marcus Ohlin `[通讯]` (Ericsson)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了AI本土化的大规模敏捷软件开发宣言，阐述了在AI成为核心参与者的情境下，如何通过并行工作、意图驱动团队、活跃知识体系、提前验证、协调的代理工作力和可重用蓝图实现真正的规模化敏捷。

**💡 创新点**

核心创新在于将人工流程从线性、会议驱动、文档为主转向以AI代理为主体的并行、意图驱动、自动化验证与知识共享的新流程；并通过语义层和蓝图生态实现跨团队的可重用与自适应。

**🔧 技术方法**

采用大语言模型驱动的多智能体系统、语义知识图谱（架构、代码、安全、Dev‑Intelligence KG）、行为驱动开发（BDD）、持续验证与测试生成、以及平台工程的“黄金路径”自动化。

**📊 数据集**

文中未使用传统数据集；核心素材为组织内部的需求、规格、代码、测试与安全数据，构成知识图谱的语义层。

**📈 对比分析**

论文为概念与方法论性工作，没有实验或量化对比；通过案例与示例说明预期的速度提升、质量改进和协作效率提升，但未给出具体性能指标。

**⚠️ 局限性**

局限包括：AI模型可能产生幻觉、缺乏跨领域普适的蓝图；多智能体协同的上下文管理与安全控制仍待完善；组织治理与人机边界定义的挑战；以及在极大规模系统中的可扩展性和治理成本。

---

## 597. Learning Tree Automata with Term Rewriting

**arXiv ID:** 2605.07710 | [PDF](https://arxiv.org/pdf/2605.07710v1)

**作者:** Jakub Kopystiański `[一作]` (University of Wrocław), Jan Otop `[通讯]` (University of Wrocław)

**通讯引用:** 222 | [OpenAlex ID](https://openalex.org/A5089075636)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出在树自动机的主动学习中引入术语重写系统（TRS）作为附加信息，利用TRS推断等价查询答案，从而显著减少学习过程中的等价查询次数。

**💡 创新点**

创新点在于：①将TRS集成到树自动机学习框架；②提供TRS与最小化DTFA一致性判定的理论依据；③识别可有效推断的一类TRS子集；④实验验证该机制能大幅降低等价查询；⑤将TRS合成问题与 DFA 同步词问题联系起来，揭示合成难度。

**🔧 技术方法**

使用技术包括：树自动机（DTFA）与 Angluin 学习算法、Myhill–Nerode 定理、术语重写系统与一致性判定、终结性/合并性判定的复杂度分析、线性规则启发式一致性检查、同步词归约、实验实现与计数统计。

**📊 数据集**

实验数据集为随机生成的满足关联性（associativity）约束的正则树语言对应的 DTFA，规模在 23–114 状态之间，全部为合成数据而非真实世界树。

**📈 对比分析**

比较方法：将使用 TRS 辅助的学习与传统无 TRS 的 Angluin 树学习进行对比，评估指标为等价查询次数。实验显示，关联性 TRS 在不同规模的 DTFA 上将等价查询次数平均减少 52%（从 11.3 次降至 5.3 次），最优时可达 87% 的减少。

**⚠️ 局限性**

局限性：
• 一致性判定对一般 TRS 复杂度高，尤其线性规则的判定复杂度仍未确定；
• 终止性要求限制了可使用的 TRS，例如交换规则不可直接使用；
• 该机制主要减少等价查询，对成员查询的改进有限；
• TRS 合成问题尚未给出有效算法，仍是开放研究方向。

---

## 598. Finite-Time Analysis of MCTS in Continuous POMDP Planning

**arXiv ID:** 2605.07703 | [PDF](https://arxiv.org/pdf/2605.07703v1)

**作者:** Da Kong `[一作]` (Technion Israel Institute Of Technology), Vadim Indelman `[通讯]` (Technion Israel Institute Of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

论文提供了 Monte Carlo 树搜索（MCTS）在离散和连续观测空间 POMDP 下的有限时间分析，并提出了具有概率有限时间保证的 Corrected-POMCP 与 Voro-POMCPOW 两种求解器。

**💡 创新点**

创新点在于首次给出 POMDP 下 MCTS 的概率有限时间集中度界，并设计了自适应抽象分区框架及基于 Voronoi 分区的连续观测空间有限时间保证。

**🔧 技术方法**

使用了改进的多项式 UCB 探索奖金、抽象分区与覆盖数上界、Wasserstein 距离 Hölder 连续性分析、进化加宽（progressive widening）以及 Voronoi 分区等技术。

**📊 数据集**

在 Modified LightDark 1D 环境上进行实验，使用 100 次回合的 10 步模拟作为数据集。

**📈 对比分析**

与基线 POMCPOW 进行平均回报对比，Voro-POMCPOW 在保证有限时间误差的前提下，平均回报与 POMCPOW 相当，并在不同置信水平下验证了理论界的实用性。

**⚠️ 局限性**

局限性包括仅适用于有限时限规划、需要观测空间满足 Hölder 连续性假设，无法直接推广到无限期望或更复杂的连续动作/状态空间，且实验验证相对有限。

---

## 599. Gradient Starvation in Binary-Reward GRPO: Why Group-Mean Centering Fails and Why the Simplest Fix Works

**arXiv ID:** 2605.07689 | [PDF](https://arxiv.org/pdf/2605.07689v1)

**作者:** Wenhua Nie `[一作]` (National Taiwan University), Jyh-Shing Roger Jang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在二元奖励下GRPO（Group Relative Policy Optimization）训练中出现的梯度饥饿问题，并提出用固定参考优势（Sign）替代组均值中心化来解决该问题。

**💡 创新点**

创新点在于：①从理论上证明梯度饥饿导致大比例组无梯度；②证明固定参考优势在所有组中都能提供梯度，并能提升Pass@G；③通过实验证明在小组尺寸下，Sign优势显著优于传统DrGRPO。

**🔧 技术方法**

技术包括：GRPO、DrGRPO、DAPO、SPO、RC-GRPO、固定参考优势（Sign）、TASA、对比式对重放、LoRA微调、Pass@k分析和Jensen不等式证明。

**📊 数据集**

主要使用的数据集是GSM8K（数学推理任务），以及在Llama-3.1-8B和Qwen系列模型上的跨模型验证；还尝试了MATH-500作为迁移检查。

**📈 对比分析**

对比方法：将Sign优势与DrGRPO、TASA、对比式对重放、CE重放等六种优势形式进行比较；在G=4、200步训练下，Sign在7个种子上平均73.8%准确率，显著高于DrGRPO的28.4%（提升45.4pp，p<0.0001）；在G=8时，DrGRPO提升到81.7%，Sign进一步到85.8%。

**⚠️ 局限性**

局限性包括：仅在Qwen3.5-9B上进行7个种子实验；跨模型和跨数据集验证有限；对最弱模型效果不佳；实际训练中加入了梯度裁剪和KL正则，理论假设略有偏差；未与动态采样（DAPO）等生成时过滤方法做直接对比。

---

## 600. Operating Within the Operational Design Domain: Zero-Shot Perception with Vision-Language Models

**arXiv ID:** 2605.07649 | [PDF](https://arxiv.org/pdf/2605.07649v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 601. The Coupling Tax: How Shared Token Budgets Undermine Visible Chain-of-Thought Under Fixed Output Limits

**arXiv ID:** 2605.07686 | [PDF](https://arxiv.org/pdf/2605.07686v1)

**作者:** Wenhua Nie `[一作]` (National Taiwan University), Jyh-Shing Roger Jang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在固定输出-token预算下，研究了Chain-of-Thought（CoT）思考模式与直接回答模式的表现差异，并提出将推理和答案分离的 split‑budget 生成方法。

**💡 创新点**

创新点在于：①发现“思考税”——推理链被截断导致答案被压缩；②给出可度量的截断‑废物分解公式和交叉预算预测；③提出无需训练的 IRIS 框架，通过拆分预算实现推理和答案的解耦，从而显著提升准确率。

**🔧 技术方法**

采用 Qwen3 系列（8B/9B/27B）模型的原生思考与非思考模式，利用推理链长度分布、截断率等统计特征构建诊断模型；并实现多轮自蒸馏（IRIS）与答案提取。

**📊 数据集**

在 GSM8K（小学数学）、MATH‑500（竞赛级数学）以及五个 BIG‑Bench Hard（BBH）任务上进行实验。

**📈 对比分析**

与单模式 baseline、coupled cascade 以及自一致性（SC）等方法比较，IRIS 在 GSM8K 上达到 90.9%（比 non‑think@256 提升 3.4pp），在 MATH‑500 上达到 74.0%（比 non‑think@2048 提升 5.6pp，比 coupled cascade 提升 3.0pp），并在更大模型 27B 上仍保持显著优势。

**⚠️ 局限性**

局限性包括：需要先测定每个任务/模型的交叉预算；在极小预算或极长推理链时仍可能出现截断；方法依赖于模型的非思考（extract）功能；对非数学推理任务的适用性与泛化仍待进一步验证。

---

## 602. PhySPRING: Structure-Preserving Reduction of Physics-Informed Twins via GNN

**arXiv ID:** 2605.07687 | [PDF](https://arxiv.org/pdf/2605.07687v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 603. Differentially Private Auditing Under Strategic Response

**arXiv ID:** 2605.07674 | [PDF](https://arxiv.org/pdf/2605.07674v1)

**作者:** Florian A. D. Burnat `[一作]` `[通讯]` (University of Bath), Florian A. D. Burnat (University of Bath)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在差分隐私（DP）约束下，构建了审计与受测开发者之间的二级Stackelberg博弈模型，研究了受测者在已知审计策略和预算分配的情况下如何通过重新分配缓解努力来规避检测，并提出了优化该审计策略的算法SPAD。

**💡 创新点**

① 将DP预算分配、查询策略与受测者的战略响应联合建模；② 证明非均匀DP审计会导致“盲点”，即真实残余伤害被低检出概率放大；③ 推导出最优审计分配不等比例且通过单层MPCC求解与梯度优化实现。

**🔧 技术方法**

差分隐私理论、可观测性函数α、KKT单层化、隐式微分与有限差分超梯度、MPCC（Mathematical Program with Complementarity Constraints）求解、Projected‑Gradient/SPAD算法。

**📊 数据集**

使用合成数据集（d=5、10、20，随机生成的基线伤害h、检测敏感度参数κ、成本族为线性/二次/幂律）。未使用真实数据，后续实验计划在伴随论文中完成。

**📈 对比分析**

与均匀分配（UNIF）、按害比例（HP）、按福利比例（WP）等基线对比；在多种总隐私预算ε_tot、维度d以及受测者策略类型（完全战略、有限理性、非战略）下，SPAD在福利加权未检测误差B_w上比基线平均下降约——%（精确数值已在实验表中给出），在ε_tot≥0.5时效果尤为显著；对非战略受测者差距相对较小。

**⚠️ 局限性**

假设受测者完全理性且可观测审计策略；仅考虑一次审计且仅调节缓解成本；忽略审计频率、预算噪声、受测者信息不完全或非理性行为；模型中α与h_res的耦合被简化，未来工作需验证对真实数据的适用性。

---

## 604. FactoryBench: Evaluating Industrial Machine Understanding

**arXiv ID:** 2605.07675 | [PDF](https://arxiv.org/pdf/2605.07675v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 605. Not All Tokens Learn Alike: Attention Entropy Reveals Heterogeneous Signals in RL Reasoning

**arXiv ID:** 2605.07660 | [PDF](https://arxiv.org/pdf/2605.07660v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 606. Kolmogorov--Nagumo Mean Frameworks for Conditional Entropy

**arXiv ID:** 2605.07624 | [PDF](https://arxiv.org/pdf/2605.07624v1)

**作者:** Akira Kamatsuka `[一作]` (Shonan Institute of Technology), Takahiro Yoshida `[通讯]` (Nihon University)

**通讯引用:** 1979 | [OpenAlex ID](https://openalex.org/A5022824333)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了基于Kolmogorov–Nagumo均值的条件熵框架，提出了(η,ψ)-KN平均和基于泛化g-不确定性的条件熵新框架，并证明了后者能表示除某些特定α外的Augustin–Csiszár条件熵。

**💡 创新点**

提出了将KN均值引入条件熵的(η,ψ)-KN平均框架，展示了其与传统(η,F)-entropy框架的等价性；另外通过g-不确定性构造新的条件熵框架，扩展了可表示的条件熵范围。

**🔧 技术方法**

利用Kolmogorov–Nagumo均值、泛化g-脆弱性、极值与凸性分析、以及Axiom化的熵框架等数学工具。

**📊 数据集**

无实验数据集，本研究为理论分析。

**📈 对比分析**

未进行实验比较；论文通过理论证明展示框架的属性和等价性，未给出数值性能。

**⚠️ 局限性**

仅在特定的凹凸性假设下成立，且对某些特殊条件熵（如Augustin–Csiszár）仍需进一步研究；缺乏对大规模实际数据的验证。

---

## 607. SafeTune: Search-based Harmfulness Minimisation for Large Language Models

**arXiv ID:** 2605.07709 | [PDF](https://arxiv.org/pdf/2605.07709v1)

**作者:** Giordano d'Aloisio `[一作]` (Univerisity of L'Aquila), Federica Sarro `[通讯]` (University College London)

**通讯引用:** 4683 | [OpenAlex ID](https://openalex.org/A5012165852)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过实验评估四大通用LLM在面对有害提示时的反应，并提出SafeTune方法，通过多目标搜索来调优模型超参数与系统提示，以在保持回答相关性的同时降低有害率。

**💡 创新点**

创新点在于将有害性降低视为多目标优化问题，利用NSGA-II搜索超参数与系统提示组合，并发现增加重复率可同时降低有害率并提升相关性。

**🔧 技术方法**

使用技术包括NSGA-II进化算法、温度、top-p、top-k、重复惩罚、最大新词等超参数搜索，以及系统级提示设计；评估工具包括Yang等人提出的LLM-as-a-Judge分类器和交叉编码器相似度。

**📊 数据集**

数据集为Harmfulness Benchmark（274个有害提示）及Qwen3.5 0.8B模型的生成样本。

**📈 对比分析**

与基线模型比较时，SafeTune在Qwen3.5 0.8B上显著降低有害率且提升回答相关性，效果量大。

**⚠️ 局限性**

局限性在于仅测试单一模型与两条提示，且评估依赖自动化指标，缺乏多模型泛化与人工评估。

---

## 608. Towards Billion-scale Multi-modal Biometric Search

**arXiv ID:** 2605.07655 | [PDF](https://arxiv.org/pdf/2605.07655v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 609. LithoBench: Benchmarking Large Multimodal Models for Remote-Sensing Lithology Interpretation

**arXiv ID:** 2605.07640 | [PDF](https://arxiv.org/pdf/2605.07640v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 610. Tacit Knowledge Extraction via Logic Augmented Generation and Active Inference

**arXiv ID:** 2605.07639 | [PDF](https://arxiv.org/pdf/2605.07639v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 611. DRIP-R: A Benchmark for Decision-Making and Reasoning Under Real-World Policy Ambiguity in the Retail Domain

**arXiv ID:** 2605.07699 | [PDF](https://arxiv.org/pdf/2605.07699v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 612. A Multi-Level Agent-Based Architecture for Climate Governance Integrating Cognitive and Institutional Dynamics

**arXiv ID:** 2605.07683 | [PDF](https://arxiv.org/pdf/2605.07683v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 613. Inference Time Causal Probing in LLMs

**arXiv ID:** 2605.07631 | [PDF](https://arxiv.org/pdf/2605.07631v1)

**作者:** Sadegh Khorasani `[一作]` (EPFL), Matthias Grossglauser `[通讯]` (EPFL)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种无探针、推理时的因果探测方法HDMI，并衍生出可进行流畅文本编辑的Lookahead HDMI；通过在隐藏层直接对模型头的logit边际进行梯度上升来实现属性的干预；

**💡 创新点**

其创新点在于：①不依赖外部探针，直接利用模型本身的输出头获取干预方向，避免了探针与模型预测几何的不匹配；②采用logit边际最大化的目标，既提升目标词概率又抑制源词概率，从而提升完整性与选择性；③引入Lookahead机制，在文本编辑中通过期望嵌入传播梯度，实现多步前瞻性调整而不破坏流畅性；

**🔧 技术方法**

技术上使用梯度上升对隐藏状态进行单步或多步更新；对多词目标使用集合边际；在编辑任务中构建可微的softmax–期望嵌入–转移链，利用向量-雅可比积传播梯度；所有操作均不需额外训练；

**📊 数据集**

实验使用LGD agreement语料和CausalGym套件，分别在Meta‑Llama‑3‑8B‑Instruct与EleutherAI/Pythia‑70M两大模型上评估；

**📈 对比分析**

与AlterRep、FGSM、PGD等基线进行比较，指标包括完整性、选择性及其调和平均可靠性；HDMI在两组数据集上均实现完美的完整性，且在大多数任务中可靠性显著优于基线；

**⚠️ 局限性**

局限性包括：对超参数（步长、温度、前瞻长度等）高度敏感；在Lookahead HDMI中梯度在较远位置衰减，导致编辑失败；缺乏统一的多词编辑基准数据集，限制了量化评估。

---

## 614. Bayesian Fine-tuning in Projected Subspaces

**arXiv ID:** 2605.07706 | [PDF](https://arxiv.org/pdf/2605.07706v1)

**作者:** Viktar Dubovik `[一作]` (Jagiellonian University), Tomasz Kuśmierczyk `[通讯]` (Jagiellonian University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在预训练模型权重投影子空间内进行贝叶斯微调的方法，结合LoRA的低秩参数更新和投影技术，实现参数高效且具备不确定性量化。

**💡 创新点**

创新点包括：①在LoRA的更新结构中固定A、B仅学习核心矩阵R，并在投影子空间中学习后验；②设计并比较四种投影策略（SVD、Whitened‑SVD、DCT、Random），证明投影对低秩协方差捕获不确定性的关键作用；③将SWAG和Laplace两种贝叶斯近似与投影结合，推出B-LoRA‑XS和L-LoRA‑XS；④证明投影子空间中仅需极低秩（k≈2）即可充分表征协方差，极大降低贝叶斯计算成本。

**🔧 技术方法**

使用技术包括LoRA低秩适配、投影子空间构造（SVD、Whitened‑SVD、DCT、Random）、子空间贝叶斯推理（SWAG、Laplace、Kronecker、Diagonal）、梯度下降、矩阵分解、熵、ECE、NLL评估等。

**📊 数据集**

实验数据集：RoBERTa‑Large在GLUE四个任务（RTE、MRPC、CoLA、SST‑2）；LLaMA2‑7B在三项commonsense reasoning任务（OBQA、ARC‑E、ARC‑C）；OOB检测使用CoLA、MRPC、SST‑2。

**📈 对比分析**

与标准LoRA、LoRA‑XS、LoRA‑SWAG、Laplace‑LoRA等方法比较，B-LoRA‑XS和L-LoRA‑XS在保持或提升准确率的同时显著降低ECE和NLL，参数量比LoRA‑SWAG低5–15倍；在LLaMA2‑7B上，L‑S（L-LoRA‑S）在相同参数下比Laplace基线更高准确率和更好校准；B-LoRA‑XS在OOD检测中也表现优于基线。

**⚠️ 局限性**

局限性：投影基底需要预训练权重，投影选择对性能影响显著；高质量投影（如Whitened‑SVD）需额外计算输入协方差；投影方法不一定适用于所有模型结构；贝叶斯后验近似仍受限于低秩或对角化，极小rank或样本量不足时性能下降；对更大模型的广泛验证仍待深入。

---

## 615. TRACE: Tourism Recommendation with Accountable Citation Evidence

**arXiv ID:** 2605.07677 | [PDF](https://arxiv.org/pdf/2605.07677v1)

**作者:** Zixu Zhao `[一作]` (UNSW Sydney), Xin Cao `[通讯]` (UNSW Sydney)

**通讯引用:** 7280 | [OpenAlex ID](https://openalex.org/A5100681901)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个旅游对话式推荐基准（TRACE），在多轮对话中要求系统给出正确的 POI、引用可验证的评论片段，并在用户拒绝后进行恢复。

**💡 创新点**

创新点在于：①整合了推荐准确度、证据可验证性和拒绝恢复三大能力为一个评价维度；②采用真实 Yelp 评论的逐句引用，确保建议可被审核；③提供可切换的候选池大小和难度标签，兼顾检索难度和对话复杂度。

**🔧 技术方法**

技术上使用了检索（TF‑IDF、Dense、Hybrid‑RRF）、基于 LLM 的零样本、RAG‑Citation、Multi‑Review Synthesis、记忆增强等多种基线，并设计了 CGS、GS、Citation Density、NLI‑Entailment 等自定义度量。

**📊 数据集**

数据集基于 Yelp，包含 8 个美国城市、3 类 POI（餐厅、酒店、景点）各 800 个，总计 29.4% 拒绝回合，提供 1,496 条对话、约 14.3 条评论/POI，标注了多参考黄金答案和候选池扩展。

**📈 对比分析**

通过 Recall@k、MRR、CGS、Citation Density、Rejection‑Recovery 等指标对 13 个基线进行评测，发现 LLM 零样本在准确度和恢复率上领先（R@1≈0.53，恢复率≈0.99），检索模型在引用质量上占优（CGS≈0.86），多检索综合在恢复率上失败，形成所谓的“三竞争力缺口”。

**⚠️ 局限性**

局限性包括：①数据仅来自 Yelp，未涵盖实时价格和动态景点信息；②评测仅针对单轮/对话级别，未覆盖多会话记忆的长尾场景；③指标对不同 LLM 家族的依赖尚未完全解耦；④对话生成过程依赖 DSPy 及预设模板，可能限制多样性。

---

## 616. The Endogeneity of Miscalibration: Impossibility and Escape in Scored Reporting

**arXiv ID:** 2605.07671 | [PDF](https://arxiv.org/pdf/2605.07671v1)

**作者:** Lauri Lovén `[一作]` (University of Oulu), Sasu Tarkoma `[通讯]` (University of Oulu)

**通讯引用:** 10947 | [OpenAlex ID](https://openalex.org/A5054443906)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

在存在非准确收益冲突的情况下，研究如何通过严格正则的评分规则和批准函数来监督自我决策的智能体或市场参与者，分析最佳监督机制对诚实报告的影响，并提出内生不可能性和构造性逃避方案。

**💡 创新点**

① 引入内生不可能性框架，证明最佳监督机制必然使用非仿射批准函数，从而使诚实报告不可实现；② 推导出闭式扰动公式量化诚实度的退化；③ 证明Brier得分在平滑监督下独具特性，使得二次最佳等价于第一最佳；④ 给出阶跃阈值逃避方案，可实现第一最佳。

**🔧 技术方法**

利用凸分析与Fenchel共轭结构、隐函数定理、Taylor展开、信息设计与机制设计工具、以及正则评分规则的几何性质进行理论推导。

**📊 数据集**

无实验数据，全部为理论分析与数学证明。

**📈 对比分析**

与传统的严格正则评分机制、Myerson最优拍卖、Crawford–Sobel信息游戏等做对比；在Brier得分下阶跃阈值实现了第一最佳；在非Brier得分下存在福利差距，阶跃阈值仍能实现最优，但平滑监督则失效。

**⚠️ 局限性**

仅适用于一维二元结果的设定，扩展到多维或多输出场景仍未完成；仅适用于严格正则评分规则；平滑监督下只有Brier得分满足；对具体AI系统的实现、学习动态与参数敏感性未深入讨论。

---

## 617. Aquatic Neuromorphic Optical Flow

**arXiv ID:** 2605.07653 | [PDF](https://arxiv.org/pdf/2605.07653v1)

**作者:** Pei Zhang `[一作]` (Guangxi University), Kaiqiang Wang `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 1588 | [OpenAlex ID](https://openalex.org/A5101991336)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a7d414a-27d1-4de0-aac0-e554088edeb4`

**🎯 论文内容**

基于事件相机的自监督脉冲神经网络框架，用于估计水下场景的每像素光流。

**💡 创新点**

提出了水下光学扭曲的等比尺度补偿与 L0 正则化的光流自监督损失，并设计了轻量级 Aq‑FireNet 结构，显著提升了在水下条件下的光流精度与稀疏性。

**🔧 技术方法**

采用事件计数编码、等比尺度补偿、L0 正则化、LIF 神经元、双路径多尺度卷积、跨通道跳跃连接、注意力模块以及 ConvGRU 记忆机制的 SNN 网络。

**📊 数据集**

在 AquaticVision、Aqua‑Eye、DAVIS‑NUIUIED 等水下事件数据集上进行训练和评估。

**📈 对比分析**

与 FireNet、EV‑FlowNet 及其 SNN 变体在监督（AEE、outlier %）和无监督（FWL、RSAT）指标上对比，Aq‑FireNet 在大多数指标上达到或逼近 ANN 基线，并在模型尺寸、能耗与延迟方面表现出更高的计算效率。

**⚠️ 局限性**

局限性在于仍依赖有限的水下事件数据集，且对极端浑浊或低事件率场景的鲁棒性有限，未来需要更大规模的标注数据和对更复杂光学畸变的建模。

---

## 618. MAVEN: Multi-Agent Verification-Elaboration Network with In-Step Epistemic Auditing

**arXiv ID:** 2605.07646 | [PDF](https://arxiv.org/pdf/2605.07646v1)

**作者:** Yinsheng Yao `[一作]` (Tongji University), Dawei Cheng `[通讯]` (Tongji University)

**通讯引用:** 2472 | [OpenAlex ID](https://openalex.org/A5069869295)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MAVEN框架，通过多智能体黑板式审计循环将LLM转变为可解释且可验证的推理器。

**💡 创新点**

创新点在于引入多阶段专家直觉与递归Skeptic‑Researcher‑Judge验证循环，确保每一步都有独立验证门控，解决传统单链错误传播问题。

**🔧 技术方法**

采用了黑板架构、适配器路由、参数检索、知识缓存、分离生成与验证等技术，构建可插拔的多智能体推理流程。

**📊 数据集**

使用了 OpenBookQA、TruthfulQA、HaluEval、StrategyQA 四大推理与真知问答基准进行实验。

**📈 对比分析**

与多种基线（多智能体、前沿LLM）对比，MAVEN在 JCD、F&C、C&A、ARS 等解释质量指标上显著优越，准确率保持竞争，整体提升约5‑10%。

**⚠️ 局限性**

局限性包括较高的计算开销、迭代次数受限、仅依赖模型参数检索，无法实时检索外部知识，轻量化模型提升仍有限。

---

## 619. OphEdit: Training-Free Text-Guided Editing of Ophthalmic Surgical Videos

**arXiv ID:** 2605.07695 | [PDF](https://arxiv.org/pdf/2605.07695v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 620. Toward Better Geometric Representations for Molecule Generative Models

**arXiv ID:** 2605.07693 | [PDF](https://arxiv.org/pdf/2605.07693v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 621. Hierarchical Task Network Planning with LLM-Generated Heuristics

**arXiv ID:** 2605.07707 | [PDF](https://arxiv.org/pdf/2605.07707v1)

**作者:** Felipe Meneguzzi `[一作]` (University of Aberdeen), André Grahl Pereira `[通讯]` (Universidade Federal do Rio Grande do Sul)

**通讯引用:** 612 | [OpenAlex ID](https://openalex.org/A5009093164)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了如何利用大型语言模型（LLM）生成层次任务网络（HTN）规划的搜索启发式函数，并在六个标准总序HTN基准域上进行评估。

**💡 创新点**

创新点在于将先前针对经典规划的LLM启发式生成方法迁移至HTN规划，设计了兼容HTN任务网络的启发式接口与提示模板，并展示LLM能显著降低搜索节点扩展量。

**🔧 技术方法**

使用技术包括：LLM生成-评估-选择管道、Python启发式接口、任务网络与世界状态的双重输入、领域特定提示与迭代改进、三种搜索算法（A*、GBFS、Weighted A*）的对比。

**📊 数据集**

数据集为IPC 2020 HTN规划赛的六个总序域（Blocksworld, Rover, Towers, Robot, Barman, Depots），共139个基准实例。

**📈 对比分析**

与三种基线（Task Decomposition Graph, RC^X+FF, RC^X+LMCut）以及最强完整HTN规划器的性能比较，LLM虚拟最佳模型覆盖131/139实例，节点扩展量在83%共享实例中降低，且在大多数域上与最强基线相当或更优。

**⚠️ 局限性**

局限性包括：仅针对总序HTN；LLM生成过程对模型和温度敏感，部分模型在多数域生成有效启发式失败；对实时或单一问题部署的成本仍高；未扩展到部分序、数值或时间约束规划。

---

## 622. Cross-Attention and Encoder-Decoder Transformers: A Logical Characterization

**arXiv ID:** 2605.07705 | [PDF](https://arxiv.org/pdf/2605.07705v1)

**作者:** Veeti Ahvonen `[一作]` (Tampere University), Matias Selin `[通讯]` (Tampere University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文提供了对跨注意力（cross‑attention）与编码‑解码 Transformer 的形式化逻辑描述，并给出了该模型与新定义的时序逻辑及分布式自动机之间的等价性。

**💡 创新点**

创新点在于：①首次为 encoder‑decoder Transformer 定义了包含计数全局模态和过去模态的逻辑；②证明该逻辑与浮点数软注意力的 Transformer 以及一种计数过去全局分布式自动机具有完全相同的表达能力；③通过浮点下溢（underflow）技术实现逻辑与 Transformer 之间的互译。

**🔧 技术方法**

使用了：浮点数运算建模、软注意力与掩蔽机制、时序逻辑（带计数全局模态与过去模态）、分布式自动机理论以及对注意力层的矩阵运算分析。

**📊 数据集**

未使用任何真实数据集，研究完全基于理论分析与形式化证明。

**📈 对比分析**

因缺乏实验与基准，未进行方法对比或性能评估；本文的验证仅来自数学证明与逻辑等价性展示。

**⚠️ 局限性**

局限性包括：仅考虑无位置编码的 Transformer 架构；未涵盖旋转或正弦位置编码等现代变体；缺乏实验验证，无法评估在实际 NLP 任务中的适用性。

---

## 623. Guidance Is Not a Hyperparameter: Learning Dynamic Control in Diffusion Language Models

**arXiv ID:** 2605.07701 | [PDF](https://arxiv.org/pdf/2605.07701v1)

**作者:** Fan Zhou `[一作]` (KU Leuven), Tim Van de Cruys `[通讯]` (KU Leuven)

**通讯引用:** 1408 | [OpenAlex ID](https://openalex.org/A5067470407)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于强化学习的自适应分类器无关引导（CFG）调度框架，在离散扩散语言模型（dLLM）中动态控制扩散过程，实现任务级的可控性与生成质量平衡。

**💡 创新点**

创新点在于将CFG调度视为序列决策问题，利用PPO学习任务特定的引导轨迹，自动挖掘不同可控性目标下的最佳引导策略，而非依赖手工设定或固定比例。

**🔧 技术方法**

核心技术包括离散扩散语言模型、分类器无关引导、策略梯度强化学习（PPO）、动作重复与时间抽象、以及基于平均/频率加权的轨迹聚合方法。

**📊 数据集**

使用公开的NLP基准数据，分别针对关键词生成、长度控制和情感风格转换三类任务，构造相应的评价集进行实验。

**📈 对比分析**

与固定比例、线性/余弦/贝塔等手工调度基线对比，RL学习得到的平均/频率轨迹在四项任务上均能同时提升可控性（覆盖率/准确率/长度符合率）与流畅度（GPT‑2困惑度），例如关键词生成覆盖率提升约3.2pp，长度控制准确率提升约16.8pp，且流畅度相对更优。

**⚠️ 局限性**

局限性在于与自回归大语言模型相比，扩散模型仍在语义保持和流畅度上存在差距；引导机制虽能缓解问题，但无法完全补偿扩散模型固有的词汇重复和生成质量下降。

---

## 624. Breaking Spatial Uniformity: Prior-Guided Mamba with Radial Serialization for Lens Flare Removal

**arXiv ID:** 2605.07650 | [PDF](https://arxiv.org/pdf/2605.07650v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 625. Learning to Communicate Locally for Large-Scale Multi-Agent Pathfinding

**arXiv ID:** 2605.07637 | [PDF](https://arxiv.org/pdf/2605.07637v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 626. Quotient Semivalues for False-Name-Resistant Data Attribution

**arXiv ID:** 2605.07663 | [PDF](https://arxiv.org/pdf/2605.07663v1)

**作者:** Florian A. D. Burnat `[一作]` (University of Bath), Brittany I. Davidson `[通讯]` (University of Bath)

**通讯引用:** 3173 | [OpenAlex ID](https://openalex.org/A5035363327)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于证据聚类的分割半值机制，以抵御机器学习训练数据中的伪名操纵并提升价值分配公平性。

**💡 创新点**

证明Shapley公平性与无名身份防护不可兼得，设计可在给定证据下实现近似无名身份防护的商数半值机制，并给出误差与公平性界限。

**🔧 技术方法**

使用机制设计、半值理论、图聚类（相似度阈值）与代表映射技术，辅以理论证明与实验验证。

**📊 数据集**

在合成分类任务、CIFAR‑10+冻结ResNet‑18、AG News+MiniLM‑L6、IMDB、STL‑10 等数据集上进行实验。

**📈 对比分析**

通过与基线Shapley、LOO、Banzhaf、Beta‑Shap等方法比较，量化操纵收益与公平性损失；实验表明复制攻击收益可从1.74降至≈0.96，纯Sybil攻击仅在oracle可防；阈值可根据嵌入池统计进行预测。

**⚠️ 局限性**

受限于需要可靠相似度或 provenance 证据，对纯分割攻击仅在有oracle时可防；阈值选择敏感，需要先行估计；理论结果主要适用于单次提交的单模型场景。

---

## 627. Stochastic Transition-Map Distillation for Fast Probabilistic Inference

**arXiv ID:** 2605.07661 | [PDF](https://arxiv.org/pdf/2605.07661v1)

**作者:** George Rapakoulias `[一作]` (Georgia Institute of Technology), Panagiotis Tsiotras `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 12576 | [OpenAlex ID](https://openalex.org/A5077667229)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对扩散模型进行教师无关的转移图蒸馏，构造单步或少步的随机采样器，以保留原始模型的随机推断特性。

**💡 创新点**

创新点在于使用条件Mean Flow学习SDE的转移概率，从而在无预训练教师、无双层优化的前提下实现高效的随机推断，并提供Wasserstein距离下的收敛理论。

**🔧 技术方法**

采用条件Mean Flow、流匹配、SDE逆过程建模、Wasserstein距离收敛分析以及基于regression的损失函数。

**📊 数据集**

在MNIST、CIFAR-10和CelebA三个数据集上进行实验。

**📈 对比分析**

与原始Mean Flow和DDPM进行对比，使用FD/FID指标和NFE（功能评估次数）评估；在保持相当或略优的生成质量的同时，显著减少推断步数，表现与基线相当。

**⚠️ 局限性**

限制主要在于实验迭代次数有限、未达到state‑of‑the‑art质量、对更大分辨率或更复杂任务的推广尚待验证。

---

## 628. Computing bases in Hermite normal form of lattices of integer relations

**arXiv ID:** 2605.07784 | [PDF](https://arxiv.org/pdf/2605.07784v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 629. Quality-Conditioned Agreement in Automated Short Answer Scoring: Mid-Range Degradation and the Impact of Task-Specific Adaptation

**arXiv ID:** 2605.07647 | [PDF](https://arxiv.org/pdf/2605.07647v1)

**作者:** Abigail Victoria Gurin Schleifer `[一作]` (Weizmann Institute of Science), Giora Alexandron `[通讯]` (Weizmann Institute of Science)

**通讯引用:** 1650 | [OpenAlex ID](https://openalex.org/A5059416730)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了自动短答评分（ASAS）系统在不同任务特定适配程度下的质量条件一致性，重点关注中等质量回答的评分偏差。

**💡 创新点**

首次量化了“中间范围降解”现象，提出质量条件公平（quality‑conditioned fairness）概念，并探讨其与模型适配度的关系。

**🔧 技术方法**

对比了三种大型语言模型（GPT‑5.2、GPT‑4o、Claude‑Opus 4.5）的few‑shot设置、一个微调BERT分类器和人类评标者；使用曼哈顿距离衡量评分差距。

**📊 数据集**

使用两道细胞呼吸相关的生物学开放式题目，收集约800名高中生答卷，并由专家按10项二元分类打分；训练集669条，测试集304条。

**📈 对比分析**

通过各分数等级的平均L1距离比较模型与专家的一致性；人类‑人类一致性最高，LLM在极端分数表现良好但中间分数平均距离超过2类；微调模型在中间分数最优，整体误差最低。

**⚠️ 局限性**

仅使用两道题目，难以推广到其他领域；使用的prompt未针对不同模型调优；未采用常规评估指标（QWK、Pearson）；实验未评估过度评分倾向等。

---

## 630. Chain-based Distillation for Effective Initialization of Variable-Sized Small Language Models

**arXiv ID:** 2605.07783 | [PDF](https://arxiv.org/pdf/2605.07783v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 631. Learning Large-Scale Modular Addition with an Auxiliary Modulus

**arXiv ID:** 2605.07648 | [PDF](https://arxiv.org/pdf/2605.07648v1)

**作者:** Hanato Kikuchi `[一作]` (Chiba University), Hiroshi Kera `[通讯]` (Chiba University)

**通讯引用:** 143 | [OpenAlex ID](https://openalex.org/A5055384327)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的训练方法，使用扩大的模数Kq作为辅助任务，保持训练和测试输入分布不变，从而避免稀疏输入引起的协变量偏移；

**💡 创新点**

创新点在于通过引入辅助模数Kq并随机选择主模数q或辅助模数Kq进行训练，既降低了模数环绕次数（提高学习难度可控性），又消除了稀疏训练导致的分布偏移；

**🔧 技术方法**

主要技术包括Transformer编码器、两种嵌入方式（token embedding与angular embedding）、辅助损失（交叉熵或MSE）、以及在训练中按概率r混合主/辅助模数；

**📊 数据集**

使用合成的模数加法数据集，训练集随机采样，测试集固定为100万条均匀样本；N（项数）与q（模数）范围覆盖小到大，如N∈{8,16,32,64,128}，q∈{31,97,257,3329,42899,974269}；

**📈 对比分析**

与之前的稀疏输入（sparse）方法比较，在相同或更少的样本下，匹配精度（match accuracy）和τ-accuracy均显著提升，尤其在大N、大q、样本量少的情况下，稀疏方法几乎无法学习，而本文方法可达到90%以上的准确率；

**⚠️ 局限性**

局限性包括：仅针对模数加法任务，尚未验证对乘法等其他运算的适用性；辅助参数K和r需通过网格搜索调优，缺乏自动化或理论化的选择策略；

---

## 632. Differentiable Ray Tracing with Gaussians for Unified Radio Propagation Simulation and View Synthesis

**arXiv ID:** 2605.07781 | [PDF](https://arxiv.org/pdf/2605.07781v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 633. Tracing Uncertainty in Language Model "Reasoning"

**arXiv ID:** 2605.07776 | [PDF](https://arxiv.org/pdf/2605.07776v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 634. Emergence of Social Reality of Emotion through a Social Allostasis Model with Dynamic Interpretants

**arXiv ID:** 2605.07761 | [PDF](https://arxiv.org/pdf/2605.07761v1)

**作者:** Kentaro Nomura `[一作]` (University of Osaka), Takato Horii `[通讯]` (University of Osaka)

**通讯引用:** 449 | [OpenAlex ID](https://openalex.org/A5031377124)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个基于POMDP的计算模型，模拟两智能体通过符号交流与主动推理共同调节身体状态，最终在社会层面上形成一致的情绪概念。

**💡 创新点**

创新点在于将身体自我调节（主动推理）与社会符号共享（Metropolis–Hastings Naming Game）结合，提出通过符号拒绝机制同步更新先验偏好，从而在没有直接访问对方内部状态的情况下实现社会情绪现实的自组织。

**🔧 技术方法**

主要技术包括：POMDP生成模型、变分贝叶斯推理、主动推理下的期望自由能最小化、Metropolis–Hastings Naming Game（MHNG）进行符号采样、在线参数更新（Dirichlet、先验偏好、符号解释矩阵）。

**📊 数据集**

使用的“数据集”是实验中的离散多模态内感信号（能量与体温各六级，共36种组合）和五种身体状态调节动作（Cool、Warm、Eat、Play、Sleep）。

**📈 对比分析**

通过与无符号共享机制对照，评估JS散度下降、先验偏好收敛速度以及符号解释矩阵的演化；实验结果表明先验偏好显著收敛至中间值，符号解释逐步趋于一致，说明模型能有效产生社会情绪现实。

**⚠️ 局限性**

局限性包括：仅在离散环境与简化的POMDP设定下验证，缺乏对连续空间和更复杂交互情景的评估；参数调节（学习率、阈值）对结果影响较大；未对比更高级的符号共享或多主体扩展。

---

## 635. When Losses Align: Gradient-Based Composite Loss Weighting for Efficient Pretraining

**arXiv ID:** 2605.07756 | [PDF](https://arxiv.org/pdf/2605.07756v1)

**作者:** Ivan Karpukhin `[一作]` (Sber AI Lab), Andrey Savchenko `[通讯]` (Sber AI Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

针对预训练阶段的多损失加权问题，本文提出一种梯度对齐的在线贝叶斯二阶优化方法（GraP），在保持单次训练的前提下自动学习最优权重。

**💡 创新点**

创新点在于：①把权重学习视为目标下的二层优化；②利用共享表征空间的梯度对齐并对复合梯度做 L2 正则化，从而避免多次完整反向传播；③仅需一次前向后向即可更新模型与权重。

**🔧 技术方法**

主要技术包括：梯度对齐（alignment）、梯度重用、共享表征梯度、贝叶斯/随机搜索对比、Bilevel 优化、梯度归一化。

**📊 数据集**

实验数据集涵盖：事件序列（Churn、AgePred、AlfaBattle、MIMIC‑III、Taobao）以及视觉自监督数据（CIFAR‑10/100、ImageNet‑100）和 All4One 框架的自监督任务。

**📈 对比分析**

与传统的等权重、GradNorm、DWA、MGDA、PCGrad 等多任务学习方法以及 Optuna 等贝叶斯搜索方法对比，GraP 在事件序列任务上平均提升 0.5–1.5 分，视觉任务表现相当于默认设置，且仅需单次训练，计算开销约 30–45% 以上。

**⚠️ 局限性**

局限性包括：假设每个预训练损失拥有独立头部，难以处理共享头部的交互；对高度相关的损失可能过度削弱次要信号；以及复合梯度尺度变化可能需要额外的学习率调整。

---

## 636. Drifting Field Policy: A One-Step Generative Policy via Wasserstein Gradient Flow

**arXiv ID:** 2605.07727 | [PDF](https://arxiv.org/pdf/2605.07727v1)

**作者:** Juil Koo `[一作]` (Korea Advanced Institute of Science and Technology), Minhyuk Sung `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 1388 | [OpenAlex ID](https://openalex.org/A5004099860)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

提出基于漂移模型的一步生成策略——Drifting Field Policy，用于离线到在线强化学习微调。

**💡 创新点**

创新点在于将漂移模型视为 Wasserstein‑2 梯度流，设计可实现的 Top‑K 牵引损失，避免 ODE 结构直接在概率空间更新策略。

**🔧 技术方法**

使用漂移模型、Wasserstein‑2 梯度流、Top‑K 目标近似、行为克隆、经验回放和一步前向推断等技术。

**📊 数据集**

在 Robomimic（3 任务）和 OGBench（6 任务）以及 任务 3 任务的 12 个连续控制任务上进行评估。

**📈 对比分析**

与多步 BC、FQL、QC‑FQL、MVP 等基线对比，平均成功率达 95.8%，在 12 项任务中 9 项排名第一，显著优于最强基线。

**⚠️ 局限性**

仅在仿真连续动作任务上验证，缺乏高维感知或真实环境迁移能力，对 Critic 质量敏感。

---

## 637. Interactive Trajectory Planning with Learning-based Distributionally Robust Model Predictive Control and Markov Systems

**arXiv ID:** 2605.07768 | [PDF](https://arxiv.org/pdf/2605.07768v1)

**作者:** Erik Börve `[一作]` (Chalmers University of Technology), Leo Laine `[通讯]` (Chalmers University of Technology)

**通讯引用:** 1216 | [OpenAlex ID](https://openalex.org/A5075558922)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于PAC学习与分布鲁棒优化的交互轨迹规划框架，利用学习到的人的决策分布在模型预测控制中构建不确定性集，从而在鲁棒与随机模型预测控制之间进行折衷。

**💡 创新点**

创新点在于：①把PAC学习的泛化误差界（excess risk bound）转换为KL散度形式的分布不确定性集；②在场景树上递归构造分布鲁棒风险度量，实现可求解的分布鲁棒MPC；③通过实验验证该方法在样本量有限时既能提高性能又能保持安全。

**🔧 技术方法**

使用的技术包括：PAC学习、交叉熵与KL散度的风险界、分布鲁棒优化（DR）、场景树（Scenario Tree）MPC、嵌套风险度量、CasADi与IPOPT求解器。

**📊 数据集**

实验使用的“道路交叉”仿真数据，模拟一维双积分动力学的自动驾驶车辆与人类车辆，状态空间为位置、速度，决策分布基于时间差特征，采用人工设定的阈值与经验样本。

**📈 对比分析**

比较方法：R‑MPC（完全鲁棒）、GT‑SMPC（使用真分布的理想随机MPC）以及本文的DR‑MPC。结果显示：R‑MPC安全但成本高、通过率低；GT‑SMPC性能最好但不可实现；DR‑MPC随样本数增大逐步逼近GT‑SMPC，在10³样本时已显著优于R‑MPC，安全性仍接近R‑MPC。

**⚠️ 局限性**

局限性：①构造的分布不确定性集过于保守，导致收敛慢；②实验仅在极其简单的二维场景，缺乏对更复杂高维交互的验证；③未给出系统稳定性、递归可行性与概率约束满足的理论保证。

---

## 638. SIMI: Self-information Mining Network for Low-light Image Enhancement

**arXiv ID:** 2605.07767 | [PDF](https://arxiv.org/pdf/2605.07767v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 639. Online Goal Recognition using Path Signature and Dynamic Time Warping

**arXiv ID:** 2605.07736 | [PDF](https://arxiv.org/pdf/2605.07736v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 640. LLM hallucinations in the wild: Large-scale evidence from non-existent citations

**arXiv ID:** 2605.07723 | [PDF](https://arxiv.org/pdf/2605.07723v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 641. CommandSwarm: Safety-Aware Natural Language-to-Behavior-Tree Generation for Robotic Swarms

**arXiv ID:** 2605.07764 | [PDF](https://arxiv.org/pdf/2605.07764v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 642. Robust and Reliable AI for Predictive Quality in Semiconductor Materials Manufacturing with MLOps and Uncertainty Quantification

**arXiv ID:** 2605.07752 | [PDF](https://arxiv.org/pdf/2605.07752v1)

**作者:** Min Gao `[一作]` (Merck Group), Gianni Klesse `[通讯]` (Merck Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文对半导体材料制造中的质量预测模型进行了系统性 MLOps 重新训练策略的基准测试，并将置信预测方法引入以实现不确定性量化。

**💡 创新点**

创新点在于提出了仅按固定 5 批周期无超参重调的最优重新训练方案，并结合 conformal 预测为质量决策提供统计上可靠的置信区间。

**🔧 技术方法**

使用了随机森林回归、PCA 进行漂移检测、嵌套交叉验证与网格搜索进行超参优化、MAPIE 库实现 conformal 预测，以及多种 MLOps 训练窗口和重训练频率的对比。

**📊 数据集**

实验基于 5 年的批量生产数据，共 1200 批，包含原料质量控制参数、环境因素和最终产品质量目标，主要针对半导体工艺化学品的批量制造。

**📈 对比分析**

通过比较无重训练、100 批间隔重训练和 5 批间隔重训练，以及固定窗口与扩展窗口和有无超参重调的四种组合，评估归一化残差、置信区间覆盖率和宽度，结果显示 5 批间隔无超参重调获得最佳性能，超参重调仅在极端漂移时略有提升。

**⚠️ 局限性**

局限性包括仅针对随机森林和单一质量目标验证；在其它模型或更大规模数据集上超参重调效果未知；conformal 预测在控制限附近的覆盖率下降；数据为专有，结果的普适性需要进一步验证。

---

## 643. TARNet: A Temporal-Aware Multi-Scale Architecture for Closed-Set Speaker Identification

**arXiv ID:** 2605.07735 | [PDF](https://arxiv.org/pdf/2605.07735v1)

**作者:** Yassin Terraf `[一作]` (University Mohammed VI Polytechnic), Youssef Iraqi `[通讯]` (University Mohammed VI Polytechnic)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 TARNet，一种轻量级多尺度时序感知网络，用于闭集说话人识别，显式建模短、中、长时序特征并通过注意力统计池化得到说话人嵌入；

**💡 创新点**

创新点在于：1）多阶段时序编码器采用不同扩张配置实现多尺度时序建模；2）将多尺度特征在通道维度融合；3）使用注意力统计池化自适应地聚合帧信息；4）整体结构轻量化；

**🔧 技术方法**

技术包括：时域一维卷积网络（TCN）+扩张卷积、多尺度特征融合、1×1卷积投影、注意力统计池化（ASP）、软最大化分类头；

**📊 数据集**

使用 VoxCeleb1（1251 讲者）和 LibriSpeech（251 讲者）公开数据集进行闭集说话人识别实验；

**📈 对比分析**

与多种 CNN、ResNet、TDNN 等基线模型对比，TARNet 在 VoxCeleb1 上 Top‑1 达到 96.25%、Top‑5 98.91%，在 LibriSpeech 上 Top‑1 99.25%、Top‑5 99.74%，相较最强基线 ECAPA‑TDNN 提升 1.75% 与 1.45% 的 Top‑1；模型参数 3.81M，推理时间 7.07 ms/utterance；

**⚠️ 局限性**

限制在于：对噪声、回声环境鲁棒性未评估；仅针对闭集识别，扩展到开放集或对抗攻击需进一步研究；

---

## 644. SARC: A Governance-by-Architecture Framework for Agentic AI Systems

**arXiv ID:** 2605.07728 | [PDF](https://arxiv.org/pdf/2605.07728v1)

**作者:** Gaston Besanson `[一作]` `[通讯]` (Universidad Torcuato Di Tella), Gaston Besanson (Universidad Torcuato Di Tella)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了SARC（State‑Action‑Reward‑Constraints）框架，构建了一套将约束视为第一类规范对象的运行时治理体系，定义了四个约束执行点（PAG、ATM、PAA、ER）并实现了多代理约束传播与审计记录的完整链路。

**💡 创新点**

创新点包括：① 将约束纳入四元组⟨S, A, R, C⟩并以元组⟨source, class, predicate, verification, response⟩细化约束；② 约束编译到四个执行点的规范化放置策略；③ 通过规范‑轨迹一致性（spec‑trace correspondence）实现可审计性；④ 针对多代理系统的约束传播、权限交叉、归因树保留等机制；⑤ 在合成任务上对SARC与现行实践基线进行可复现的性能对比。

**🔧 技术方法**

使用的技术包括：约束规范模型、四点执行点架构、可验证的审计检查器、公式化的约束可达性与时延分析、基于M/M/c排队模型的人工干预评估、以及多代理拓扑中的权限交叉与约束递归解析。

**📊 数据集**

数据集：使用人工合成的采购任务数据，包含 1,000 条订单，金额服从对数正态分布（μ=8.5, σ=1.2），并在 50 个随机种子上复现。未使用公开真实数据集。

**📈 对比分析**

比较方法：在同一采购任务上与四种基线（后置审计、输出过滤、工作流规则、仅政策即代码）进行对比，测量硬约束违规、软窗口超额、是否触发人机升级以及每步时延。结果显示：SARC 在所有硬约束下实现零违规；软窗口违规率下降 89.5%；与基线相比，SARC 通过约束放置与类型化响应提升了治理效果，但相应的时延和升级负载略有增加。

**⚠️ 局限性**

局限性：① SARC 仅提供运行时治理框架，无法单独保证合规或安全；② 约束的设计和操作点需要人工翻译与维护，缺少自动化支持；③ 在多代理市场或拍卖拓扑中的约束治理尚未完整；④ 评估仅基于合成任务，缺乏真实场景验证；⑤ 需要外部法律解释与审批层才能形成完整合规链。

---

## 645. Efficient Verification of Neural Control Barrier Functions with Smooth Nonlinear Activations

**arXiv ID:** 2605.07757 | [PDF](https://arxiv.org/pdf/2605.07757v1)

**作者:** Jun Zhang `[一作]` (Shanghai University), Liang Xu `[通讯]` (Shanghai University)

**通讯引用:** 13877 | [OpenAlex ID](https://openalex.org/A5109384412)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

研究了神经控制壁障函数（NCBF）的形式化验证，提出 LightCROWN 方法来获取更紧的雅可比矩阵界，以提高安全性证明的成功率和计算效率。

**💡 创新点**

创新点在于直接利用激活函数导数的解析单调性，精确求得导数极值，替代传统线性松弛，从而得到显著更紧的 Jacobian 界。

**🔧 技术方法**

采用 CROWN 框架的线性界传播，改进为 LightCROWN；结合 Taylor 近似对系统动力学做界估计；使用区间运算、正负分解与反向传播计算 Lie 导数上界。

**📊 数据集**

在倒立摆、Dubins 车和平面四旋翼三类控制系统上进行实验，使用相应的状态网格划分作为子区域，并训练两层 tanh MLP 作为 NCBF。

**📈 对比分析**

与原 CROWN 和 BBV 对比，LightCROWN 在安全率和计算时间上均优越；在正则训练下验证率约 91%，恶意训练下保持 ≥80% 且速度提升；在更大隐藏层和更细划分时表现出更好的可扩展性。

**⚠️ 局限性**

局限性包括：仍需依赖子区域网格逼近边界，细分会显著增加计算开销；在正则训练下对 α 的敏感性存在“反转”现象；对非单调激活或高阶动力学的适用性尚未完全验证。

---

## 646. TextLDM: Language Modeling with Continuous Latent Diffusion

**arXiv ID:** 2605.07748 | [PDF](https://arxiv.org/pdf/2605.07748v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 647. Securing the Dark Matter: A Semantic-Enhanced Neuro-Symbolic Framework for Supply Chain Analysis of Opaque Industrial Software

**arXiv ID:** 2605.07737 | [PDF](https://arxiv.org/pdf/2605.07737v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 648. Vibe coding before the trend

**arXiv ID:** 2605.07751 | [PDF](https://arxiv.org/pdf/2605.07751v1)

**作者:** Leon van Bokhorst `[一作]` (Fontys ICT, University of Applied science), Koen Suilen `[通讯]` (Fontys ICT, University of Applied science)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7`

**🎯 论文内容**

在四个不同教育场景中，组织学生使用前沿 AI 编程工具（如 Cursor IDE、Claude Sonnet 等）完成项目实验，并收集学生反馈；

**💡 创新点**

首次将多种 AI 编程工具与跨学科教育实验相结合，系统归纳学生对 AI 的认知与使用模式；

**🔧 技术方法**

采用 Claude Sonnet、Cursor IDE、Lovable、Bolt 等生成式 AI 开发工具；

**📊 数据集**

使用 Kaggle 上的 Pokémon 数据集作为实验任务数据；

**📈 对比分析**

通过定性分析 107 条学生反思文本提炼共性模式，并未进行量化对比或性能评估；

**⚠️ 局限性**

样本规模有限、缺乏对照实验、评估方法仅为定性访谈，且未追踪学生使用工具的长期效果。

---

## 649. An Efficient Hybrid Sparse Attention with CPU-GPU Parallelism for Long-Context Inference

**arXiv ID:** 2605.07719 | [PDF](https://arxiv.org/pdf/2605.07719v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 650. Benchmarking Foundation Models for Renal Lesion Stratification in CT

**arXiv ID:** 2605.07749 | [PDF](https://arxiv.org/pdf/2605.07749v1)

**作者:** Hartmut Häntze `[一作]` (Charité - Universitätsmedizin), Keno Bressem `[通讯]` (Technical University of Munich)

**通讯引用:** 4089 | [OpenAlex ID](https://openalex.org/A5006318966)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文对三种开放源代码医学基础模型（FMCIB、CT‑FM、MMM）在CT‑肾脏病变六分类（囊肿、清细胞肾细胞癌、泌乳腺型肾细胞癌、黄斑细胞型肾细胞癌、肾上皮瘤、其他）中的迁移学习能力进行基准测试，并将其与传统手工 radiomics 分类器和从零开始训练的 3D ResNet‑50 进行比较；采用冻结特征探测协议，使用 XGBoost 进行下游分类，并在内部 10‑折交叉验证和外部 TCIA 测试集上评估性能。

**💡 创新点**

首次系统性地评估医学基础模型在数据稀缺的肾脏病变 CT 诊断任务中的表现，展示即使不进行微调，基础模型的特征提取与从零训练的深度网络相当，同时显著降低硬件需求；同时揭示目前的基础模型尚无法超越手工 radiomics 的准确度。

**🔧 技术方法**

冻结特征探测（feature‑probing）+XGBoost 分类器；手工 Radiomics 特征提取（PyRadiomics 107 维）+XGBoost；3D ResNet‑50（从零训练与 FMCIB 微调两种初始化）；交叉验证、AUC/AP 评估、UMAP 可视化、统计显著性检验。

**📊 数据集**

训练集：本地肾脏病变 CT（796例）+ KiTS23（483例）共 2,854 病灶；外部测试集：TCIA（140 扫描、234 病灶）仅包含囊肿和三种肾细胞癌。

**📈 对比分析**

对内部 10‑折交叉验证和外部 4‑分类测试集分别计算 AUC 与 AP。Radiomics 基线内部 AUC 0.84、外部 0.88；基础模型（冻结）内部 AUC 0.70–0.77、外部 0.69–0.77；ResNet‑50 内部 0.64、外部 0.72。结果表明基础模型性能与从零训练网络相当，但均低于 Radiomics，且差异在外部测试集上显著。

**⚠️ 局限性**

主要局限：基础模型未能捕获细粒度纹理与形状差异，导致与 Radiomics 的性能差距；外部测试集中缺少少数类（RO、Other）使得评价偏向主流子类型；仅使用经病理确认的病例，可能导致样本偏差；缺乏统一的基础模型使用规范，导致实现细节对结果影响较大；平均精度低、假阳性率高，限制了临床自主使用。

---

## 651. A Scalable Recipe on SuperMUC-NG Phase 2: Efficient Large-Scale Training of Language Models

**arXiv ID:** 2605.07726 | [PDF](https://arxiv.org/pdf/2605.07726v1)

**作者:** Ajay Navilarekal Rajgopal `[一作]` (Leibniz Supercomputing Centre Bavarian Academy of Science and Humanities), Nikolai Solmsdorf `[通讯]` (Intel Deutschland GmbH)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在SuperMUC-NG Phase 2上实现并验证175B GPT模型的分布式训练，提供可复现的端到端配置。

**💡 创新点**

提出一种基于Tensor、Pipeline和Sharded Data并行的3D并行训练配方，完全使用公开软件栈，在不修改内核的情况下达到10%理论峰值。

**🔧 技术方法**

使用Megatron-DeepSpeed、ZeRO-1、Tensor/PIPE/PARallelism、混合精度、Intel Extension for PyTorch以及贝叶斯优化自动调参。

**📊 数据集**

采用公开的通用语言建模数据集（如Common Crawl/BookCorpus）进行预训练，未单独报告数据集细节。

**📈 对比分析**

通过对比不同并行度配置、吞吐量和缩放效率，获得每张GPU 57 TFLOPs/s（≈10%峰值），弱缩放93%，强缩放82%，优于此前在同等规模系统上的报告。

**⚠️ 局限性**

局限于功耗受限的生产模式，未进行能效分析，缺乏训练准确率评估，仅在单一硬件平台验证，且不包含定制核优化。

---

## 652. Radiologist-Guided Causal Concept Bottleneck Models for Chest X-Ray Interpretation

**arXiv ID:** 2605.07785 | [PDF](https://arxiv.org/pdf/2605.07785v1)

**作者:** Amy Rafferty `[一作]` (University of Edinburgh), Ajitha Rajan `[通讯]` (University of Edinburgh)

**通讯引用:** 1923 | [OpenAlex ID](https://openalex.org/A5079075574)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一种基于放射科医生指导的因果概念瓶颈模型XpertCausal，用于胸部X光片的病理诊断和可解释性分析。

**💡 创新点**

创新点在于把病理→影像发现的生成过程建模为噪声OR的概率因果图，并通过贝叶斯推理逆向估计病理概率，同时将放射科医生预定义的概念-病理关联矩阵作为结构约束，提升解释可信度与临床契合度。

**🔧 技术方法**

技术包括基于InceptionV3的概念预测网络、噪声OR因果生成模型、贝叶斯推断、决策树下游预测以及对照实验中的非因果CBM和无约束因果学习模型。

**📊 数据集**

实验数据来自MIMIC‑CXR公共数据集，采用放射科报告中提取的11个概念与6个病理标签，经过放射科医生手工制定的概念-病理关联矩阵进行建模。

**📈 对比分析**

与XpertXAI（非因果CBM）、无约束因果模型以及直接的InceptionV3分类器相比，XpertCausal在宏观AUROC、F1、期望校准误差、解释质量（top‑K概念覆盖率）等指标均表现最优。

**⚠️ 局限性**

局限性包括概念空间仅有11个、病理标签有限，且所有概念定义与关联矩阵均来自单位放射科医生；缺乏外部数据集验证，模型泛化性和多中心适用性待进一步评估。

---

## 653. POETS: Uncertainty-Aware LLM Optimization via Compute-Efficient Policy Ensembles

**arXiv ID:** 2605.07775 | [PDF](https://arxiv.org/pdf/2605.07775v1)

**作者:** Nicolas Menet `[一作]` (IBM), Abbas Rahimi `[通讯]` (IBM)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于策略集成的 Thompson 采样框架（Policy Ensembles for Thompson Sampling, PoETS），能够在大型语言模型后训练与科学发现任务中实现不确定性感知的探索与利用；

**💡 创新点**

创新点在于：① 直接将 KL 正则化策略的隐式奖励与观测数据对齐，跳过显式奖励模型；② 采用 Poisson bootstrapping 在线量化经验不确定性；③ 通过共享主干+独立 LoRA 分支的 Trunk & Branch 结构，以极低的计算/内存成本实现多策略集成；④ 理论证明 PoETS 等价于 KL‑regularized Thompson 采样，获得软累计 regret 上界。

**🔧 技术方法**

使用的技术包括：策略梯度与软奖励对齐；KL 与熵正则化；Poisson bootstrapping；Trunk & Branch（共享主干+LoRA 分支）架构；在线经验回放；与传统策略梯度、VAR‑TM、In‑Context Thompson、Evolutionary Search 等基线对比。

**📊 数据集**

实验数据集：FAQ Refinement（Qwen3‑8B + Qwen3‑Embedding‑0.6B）、Protein Search（ProtGPT2 + 热稳定性评分）、Quantum Circuit Design（Qwen3‑8B + Qiskit 7‑qubit 电路）、AIME（用于 RLVR 的可验证奖励）。

**📈 对比分析**

与 PPO、Reinforce、VAR‑TM、In‑Context Thompson、Evolutionary Search 等方法比较，PoETS 在三大黑盒优化任务中实现 state‑of‑the‑art 采样效率，即使在无 replay buffer 的情况下亦优于对比方法；在 RLVR 上在 on‑policy 与 off‑policy 训练中保持生成多样性，避免过拟合，准确率提升；其计算/内存开销仅比单策略低 7‑16%。

**⚠️ 局限性**

局限性：仅在单步/情境分支下验证；依赖 KL/熵正则参数的调优；在高维连续动作空间或多步决策场景中的表现尚未评估；理论假设奖励可由 Gaussian Process 表达，实际应用可能偏离。

---

## 654. Coding Agents Don't Know When to Act

**arXiv ID:** 2605.07769 | [PDF](https://arxiv.org/pdf/2605.07769v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 655. SMT-Based Active Learning of Weighted Automata

**arXiv ID:** 2605.07758 | [PDF](https://arxiv.org/pdf/2605.07758v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

---

## 656. Beyond Brooks: $(Δ-1)$-Coloring in Semi-Streaming

**arXiv ID:** 2605.07774 | [PDF](https://arxiv.org/pdf/2605.07774v1)

**作者:** Maxime Flin `[一作]` (Aalto University), Magnús M. Halldórsson `[通讯]` (Reykjavik University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一个一遍半流式算法，能够在O(n log n)空间下为最大度数为Δ且不含Δ团的图生成(Δ−1)-着色，并给出(Δ−k)-着色的空间下界。

**💡 创新点**

创新点在于首次将Reed的Δ≥10^14无Δ团图的(Δ−1)-着色结果转化为半流式环境，并结合palette sparsification、sparse recovery以及新的Reed变换和松弛生成技术，实现一遍O(n log n)空间的随机算法；同时证明(Δ−k)-着色在半流式模型下需要Ω(n(k+1))空间。

**🔧 技术方法**

使用技术包括：palette sparsification（稀疏列表抽样）、sparse recovery（线性草图重建）、Reed transform（在图中添加删减边以简化结构）、概率松弛生成（随机着色产生相同颜色邻居）以及自适应随机列表采样。

**📊 数据集**

论文为理论研究，没有使用具体数据集；全部结果基于图论与通信复杂性理论证明。

**📈 对比分析**

方法通过与已知的(Δ+1)-着色与Δ-着色算法对比，表明在满足Δ足够大且无Δ团的前提下，一遍半流式算法可在O(n log n)空间完成；同时下界证明表明对更少颜色的着色在半流式模型下是不可行的。

**⚠️ 局限性**

局限性包括：对Δ的阈值Δ0极大（≤10^14），仅适用于每条边只出现一次的流；需要随机性；下界仅针对k<(Δ+1)/2；未给出多余边或Δ≤10^14情况下的高效实现。

---

## 657. Suitability of the Data Distribution Service for Next-Generation Ethernet-Based Agricultural Machinery Networking

**arXiv ID:** 2605.07742 | [PDF](https://arxiv.org/pdf/2605.07742v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 658. Sensitivity-Based Robust NMPC for Close-Proximity Offshore Wind Turbine Inspection with a Tilted Multirotor

**arXiv ID:** 2605.07771 | [PDF](https://arxiv.org/pdf/2605.07771v1)

**作者:** Giuseppe Silano `[一作]` (Ricerca sul Sistema Energetico S.p.A.), Martin Saska `[通讯]` (Czech Technical University)

**通讯引用:** 5283 | [OpenAlex ID](https://openalex.org/A5004992661)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对倾斜多旋翼无人机在近距离风电塔巡检时的安全间隙约束，设计并验证了一种基于敏感性加紧的鲁棒非线性模型预测控制。

**💡 创新点**

创新点在于将第一阶结构不确定性敏感性传播与风 gust 加性边际相结合，仅在约束层面实现在线约束收缩，保持原有 NMPC 结构不变，并专门针对圆柱塔间隙约束制定收缩策略。

**🔧 技术方法**

使用 GTMR 预测模型、四阶 Runge–Kutta 离散化、多射击 NMPC、自动微分求敏感性、MATMPC 求解器以及 Monte‑Carlo 评估方法。

**📊 数据集**

采用仿真参数与不确定性（质量、惯性、推力、阻力、风偏差、gust）组成的 500 次统一随机采样样本，未使用真实实验数据。

**📈 对比分析**

通过与基线 Nominal NMPC 在 500 次 Monte‑Carlo 试验中的比较，鲁棒 NMPC 违规率从 230/500 降至 0/500，平均求解时间从 9.4 ms 增至 18.7 ms，验证了安全性提升与可接受的计算开销。

**⚠️ 局限性**

局限性包括：只考虑一次阶敏感性，可能不足以处理大幅模型偏差；仅在仿真中验证，缺乏真实硬件实验；风 gust 模型简化为开放式积分，实际环境复杂度更高；嵌入式实现的实时可行性仍待进一步验证。

---

## 659. Intelligent Truck Matching in Full Truckload Shipments using Ping2Hex approach

**arXiv ID:** 2605.07733 | [PDF](https://arxiv.org/pdf/2605.07733v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 660. Benchmarking EngGPT2-16B-A3B against Comparable Italian and International Open-source LLMs

**arXiv ID:** 2605.07731 | [PDF](https://arxiv.org/pdf/2605.07731v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 661. Rethinking State Tracking in Recurrent Models Through Error Control Dynamics

**arXiv ID:** 2605.07755 | [PDF](https://arxiv.org/pdf/2605.07755v1)

**作者:** Jiwan Chung `[一作]` (Yonsei University), Seon Joo Kim `[通讯]` (Yonsei University)

**通讯引用:** 4719 | [OpenAlex ID](https://openalex.org/A5103036411)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究循环神经网络在状态跟踪任务中的错误控制机制，证明仿射递归模型在保持状态表示的前提下无法修正状态分隔子空间中的误差，并用实验验证有限视界理论。

**💡 创新点**

提出错误控制是循环状态跟踪鲁棒性的关键轴，并证明仿射模型无法在状态分隔子空间上收缩误差；定义可读性阈值和T_cross并将其与实际跟踪有效期关联。

**🔧 技术方法**

形式化证明、仿射与状态依赖递归层的理论分析、群状态跟踪任务的实验设计、距离度量与可读性阈值计算以及误差传播分析。

**📊 数据集**

使用基于有限群的跟踪任务，涵盖偶奇 C₂、循环 C₆ 以及对称群 S₃，随机生成生成器序列进行训练与评估。

**📈 对比分析**

对多种模型（SSM、Mamba、AUSSM、线性 RNN、tanh RNN、gated RNN 等）在从 60 到 1000 的序列长度上做加速学习，衡量 90% 准确率的最大长度；结果显示状态依赖模型可持续 1000 长度，而仿射模型仅在部分情况可达 400‑1000，T_cross 与最大可通过长度高度相关。

**⚠️ 局限性**

实验仅覆盖有限群跟踪任务，训练长度上限为 60，未探讨非群任务；理论主要针对仿射返回映射，对非线性动态的完整性和泛化性仍需进一步研究。

---

## 662. Accelerating Precise End-to-End Simulation: Latency-Sensitive Many-core System Modeling

**arXiv ID:** 2605.07750 | [PDF](https://arxiv.org/pdf/2605.07750v1)

**作者:** Yinrong Li `[一作]` (ETH Zurich), Luca Benini `[通讯]` (ETH Zurich)

**通讯引用:** 57695 | [OpenAlex ID](https://openalex.org/A5043408422)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种面向共享 L1 Scratchpad 内存（SPM）的多核系统的高速、精确的端到端性能模型，并将其应用于 TeraNoC，支持快速仿真和详细的性能分析。利用该模型对 FlashAttention‑2 进行了软件层面的性能调优，并通过路由器重映射进行硬件设计空间探索。

**💡 创新点**

创新点：
• 采用基于请求的细粒度时序传播模型，能够捕捉多级互连（局部交叉口、组内/组间网路）中因争用和仲裁导致的微小延迟波动。
• 在不同互连层次上选择同步/异步执行策略，实现准确性与仿真速度的权衡。
• 提供跨 PE、互连和内存层面的细粒度统计和轨迹信息，显著提升软件调优和硬件 DSE 的可视化与指导能力。
• 通过与 RTL 仿真的对比验证模型误差低于 7%，并在实际工作负载（包括 FlashAttention‑2、矩阵乘法等）上实现 100+ 倍加速。

**🔧 技术方法**

技术：
• GVSoC 事件驱动仿真框架；
• 通用路由器抽象（可配置端口、仲裁、时延、带宽）；
• 请求（request）传播机制（同步/异步），包含基准延迟、仲裁延迟、传输时延更新；
• 详细的 profiling 采样（PE 指令、存取、互连利用率、拥塞统计等）。

**📊 数据集**

数据集/工作负载：
• 传统线性代数与信号处理基准；
• 现代机器学习内核（FlashAttention‑2、BatchNorm、LayerNorm、Softmax、GEMM、MMSE 等）；
• 矩阵乘法（matmul）与全局内存访问密集型任务。
这些基准覆盖了从词级访问到大规模批量运算的多种访问模式。

**📈 对比分析**

比较方法与性能：
• 与周期级 RTL 仿真做对比，误差维持在 7% 以内；
• 仿真速度提升 2 级以上，最高 115×；
• 在 FlashAttention‑2 上，通过调节访问分布与起始时序，显著减少互连阻塞与同步开销，提升整体吞吐；
• 采用路由器重映射后，整体性能提升约 10%（部分内核如 BN、LN、SM 可达 47%），显示模型在硬件 DSE 方面的有效性。

**⚠️ 局限性**

局限性：
• 仍以请求级细粒度为主，未能完整模拟高层包级传输与流控制细节；
• 需要针对具体硬件做校准，模型在极高并发或非常不同的互连拓扑下可能需重新调整参数；
• 关注于 SPM‑中心架构，对缓存层次或多级内存体系结构的适用性有限；
• 模型的简化假设（如将单一硬件块折叠为单个路由器）在某些细节上可能引入微小误差。

---

## 663. LAMES: A Large-Scale and Artisanal Mining Environmental Segmentation Dataset

**arXiv ID:** 2605.07740 | [PDF](https://arxiv.org/pdf/2605.07740v1)

**作者:** Matthias Kahl `[一作]` (Technical University of Munich), Xiao Xiang Zhu `[通讯]` (Technical University of Munich)

**通讯引用:** 26055 | [OpenAlex ID](https://openalex.org/A5068384981)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建并公开了一个大规模矿产场所数据集（LAMES），包括150个大型采矿场（LSM）在智利的高分辨率地理注记、约1,200个手工小规模采矿场（ASM）在加纳的区域划分、相关元数据以及用于语义分割和矿区检测的 Sentinel‑2 街道/高分辨率图像及掩膜。

**💡 创新点**

创新点在于：①首次公开结合元数据、分割掩膜和多分辨率卫星影像的矿产场所数据集；②提供了详细的矿区结构类别（如开放式采坑、废料堆、尾矿库等）以及对应的像素掩膜；③提出了针对高分辨率图像的矿区分割基线实验和对中分辨率 Sentinel‑2 的矿区检测基线实验。

**🔧 技术方法**

技术方法：采用 U‑Net+ResNet‑50 编码器、交叉熵损失、Adam 优化器、循环学习率策略与类别权重；对 Sentinel‑2 进行图像补丁划分、标准化；在矿区检测中采用随机采样的无矿区补丁做负样本增强；使用 10×10 cm 像素级掩膜进行语义分割评估。

**📊 数据集**

使用的数据集为 LAMES：包含 150 个 LSM 场所的 1,204 个分区掩膜、1,288 个 ASM 区域掩膜、对应的 Sentinel‑2 10 m 街道图像和 0.5 m 高分辨率图像；元数据来源于 MiningDataOnline、MiningNews、Diggings、Antofagasta Minerals、Centro Nacional de Pilotaje 等公开数据库。

**📈 对比分析**

对比方法：基线采用 U‑Net+ResNet‑50；在语义分割任务上，微平均精度 59.2%，宏平均 89.0%；矿区检测任务在 10,000 个随机无矿区样本增强后，整体精度 0.89，召回率 0.75；与未增强的基线相比，假阴性率从 36% 降至 25%，假阳性率从 6.1% 上升至 9.4%。

**⚠️ 局限性**

局限性：①类别不平衡严重，尤其是 leaching heaps 极易被误分类；②数据规模仅覆盖智利和加纳，难以推广到其他矿区；③中分辨率 Sentinel‑2 在检测稀疏矿区时效果不佳，需更高分辨率或更复杂的特征工程；④实验仅为基线，尚未尝试更先进的网络结构、数据增强或伪标签技术。

---

## 664. Pre-trained Tabular Foundation Models as Versatile Summary Networks for Neural Posterior Estimation

**arXiv ID:** 2605.07765 | [PDF](https://arxiv.org/pdf/2605.07765v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 665. APEX: Assumption-free Projection-based Embedding eXamination Metric for Image Quality Assessment

**arXiv ID:** 2605.07786 | [PDF](https://arxiv.org/pdf/2605.07786v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 666. Curated Synthetic Data Doesn't Have to Collapse: A Theoretical Study of Generative Retraining with Pluralistic Preferences

**arXiv ID:** 2605.07724 | [PDF](https://arxiv.org/pdf/2605.07724v1)

**作者:** Ali Falahati `[一作]` (University of Waterloo), Lukasz Golab `[通讯]` (University of Waterloo)

**通讯引用:** 4980 | [OpenAlex ID](https://openalex.org/A5049437648)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出在递归生成模型训练中使用多重偏好（pluralistic）选择，而非单一奖励，证明可避免模式崩塌并保持多样性。

**💡 创新点**

通过理论证明在大候选数极限下，随机切换奖励导致模型收敛到权重为 q 的两峰混合，满足加权 Nash 博弈解；并给出离散候选数误差及泄漏分析。

**🔧 技术方法**

基于 Bradley‑Terry 离散选择模型、指数倾斜更新、最大似然重训练、离散化候选池；使用大样本极限、概率收敛和多峰分布分析。

**📊 数据集**

合成二维高斯混合（GMM）、CIFAR‑10 图像数据（使用预训练 VGG11 作为奖励）、以及 GPT‑2 文本长度奖励，验证理论。

**📈 对比分析**

与单奖励递归训练对比，显示在多奖励设置下维持高熵、正奖励方差，FID 下降；实验表明随着奖励分离度增大，混合权重趋近 q，模型保持多样性。

**⚠️ 局限性**

实验依赖离散候选池大小、K 值，理论假设包括奖励间的显著分离和泄漏界；在高度重叠或奖励极度偏斜时模型仍可能向主导奖励倾斜，且真实数据的引入对理论的适用性有限。

---

## 667. CktFormalizer: Autoformalization of Natural Language into Circuit Representations

**arXiv ID:** 2605.07782 | [PDF](https://arxiv.org/pdf/2605.07782v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 668. Post-Moore Technologies for Plasma Simulation: A Community Roadmap

**arXiv ID:** 2605.07722 | [PDF](https://arxiv.org/pdf/2605.07722v1)

**作者:** Luca Pennati `[一作]` (KTH Royal Institute of Technology), Stefano Markidis `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 4899 | [OpenAlex ID](https://openalex.org/A5085178088)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对后摩尔时代技术在等离子体模拟中的应用进行了社区视角的评估，归纳了三类技术的成熟度与潜在影响，并给出了针对不同等离子体工作流的代码协同采用建议。

**💡 创新点**

创新点在于系统性地将FPGA/数据路径加速器、非冯·诺伊曼架构和量子计算三大技术阶梯与粒子/连续/等离子体动力学方法相匹配，并提出了可操作的采用框架与优先级划分。

**🔧 技术方法**

使用的技术包括可重构FPGA、CGRAs、SmartNIC/DPU、Neuromorphic、Thermodynamic、Compute‑in‑Memory、Photonic、Many‑body专用处理器以及量子算法（Hamiltonian模拟、热态制备、量子采样）。

**📊 数据集**

数据集主要为典型等离子体模拟工作负载，例如PIC、Vlasov、Gyrokinetic、MHD/Hybrid、WDM（DFT、PIMC）等，未给出具体数值文件，侧重于代表性核算。

**📈 对比分析**

比较方法以小规模基准（核算子加速比）和整体工作流演示为主，指出FPGA可实现粒子-网格耦合的10×加速，SmartNIC/DPU在数据迁移与压缩上可提升数十%吞吐；非冯·诺伊曼技术在局部算子上提供数倍速率但整体影响有限；量子计算仅在小型示例中展示潜在规模化收益。

**⚠️ 局限性**

局限性包括：技术成熟度不均衡（量子与非冯·诺伊曼技术仍处于实验阶段）、硬件与软件耦合复杂、缺乏大规模生产级工作流验证、对现有代码架构兼容性要求高，以及对精度与可扩展性的挑战。

---

## 669. Memory-Efficient Looped Transformer: Decoupling Compute from Memory in Looped Language Models

**arXiv ID:** 2605.07721 | [PDF](https://arxiv.org/pdf/2605.07721v1)

**作者:** Victor Conchello Vendrell `[一作]` (Qualcomm AI Research), Fabio Valerio Massoli `[通讯]` (Qualcomm AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为MELT的循环Transformer架构，能够在不扩展KV缓存的情况下进行多步推理。

**💡 创新点**

创新点包括：用共享单一KV缓存并通过可学习门控机制更新；采用分块训练+插值过渡+注意力对齐蒸馏的两阶段训练策略；实现记忆占用与推理深度解耦。

**🔧 技术方法**

技术：循环Transformer（LoopLM）架构、可学习门控更新、分块训练、插值过渡、注意力对齐知识蒸馏、KV缓存共享与更新。

**📊 数据集**

数据集：AceReason‑1.1‑SFT、OpenThoughts3，用于微调；在AIME24/25/26、AMC23、MATH‑500、OlympiadBench等数学推理基准以及GPQA、HLE、MMLU‑Red、HumanEval等通用推理基准进行评估。

**📈 对比分析**

与同等规模的标准Transformer（如Qwen3-1.7B、Gemma4‑E2B、Qwen3.5-2B、DeepSeek-R1-1.5B）以及循环模型Ouro‑1.4B‑Thinking比较；MELT在大多数基准上表现优于非循环模型，接近或略低于Ouro，但显著降低内存占用（约4倍减少）。

**⚠️ 局限性**

局限性：循环步数固定，未实现动态推理深度；未结合多查询注意力(MQA)；训练时需要顺序KV更新，降低并行度；对更大模型/更长推理阶段的扩展仍需进一步研究。

---

## 670. Training-Induced Escape from Token Clustering in a Mean-Field Formulation of Transformers

**arXiv ID:** 2605.07772 | [PDF](https://arxiv.org/pdf/2605.07772v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 671. Head Similarity: Modeling Structured Whole-Head Appearance Beyond Face Recognition

**arXiv ID:** 2605.07766 | [PDF](https://arxiv.org/pdf/2605.07766v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 672. RuleSafe-VL: Evaluating Rule-Conditioned Decision Reasoning in Vision-Language Content Moderation

**arXiv ID:** 2605.07760 | [PDF](https://arxiv.org/pdf/2605.07760v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 673. FAVOR: Efficient Filter-Agnostic Vector ANNS Based on Selectivity-Aware Exclusion Distances

**arXiv ID:** 2605.07770 | [PDF](https://arxiv.org/pdf/2605.07770v1)

**作者:** Junjie Song `[一作]` (Huazhong University of Science and Technology), Ke Zhou `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 6129 | [OpenAlex ID](https://openalex.org/A5015061573)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种滤波无关的向量近似最近邻搜索（ANNS）方法，能够支持任意过滤条件，并在不同选择率（selectivity）下保持高查询吞吐量（QPS）。

**💡 创新点**

创新点包括：①基于查询选择率的搜索选择器，动态决定使用预过滤线性扫描或改进的 HNSW 搜索；②在 HNSW 内部引入排除距离（exclusion distance）机制，实时拉远不满足过滤条件的向量，使满足条件的向量更靠近查询点；③优化终止条件，防止因连续遍历不满足条件的向量导致提前终止，从而提升召回率。

**🔧 技术方法**

核心技术：Hierarchical Navigable Small World（HNSW）图；动态距离修正（排除距离）与自适应搜索；基于采样的选择率估计；预过滤与优化终止条件相结合的两阶段查询流程。

**📊 数据集**

使用七个公开数据集：SIFT1M、GIST1M、DEEP1M、Msong、Paper、Words、LAION25M，涵盖图像、音频、文本等多种模态，且为每个向量附加 bool、int、float 三种属性以构造多样化的过滤场景。

**📈 对比分析**

与 ACORN、UNG、SeRF、Milvus、Vearch、Result‑Set‑Filtering 等基线进行对比；在 Recall@10=95% 的召回率下，FAVOR 在大多数场景下 QPS 提升 1.3–5 倍；在特定过滤条件下与针对性方法竞争，整体性能稳定且优于现有滤波无关方案。

**⚠️ 局限性**

局限性：①需预设切换阈值（如 1%），对极低选择率的估计误差仍可能影响性能；②构建时需要额外采样计算平均距离 Δd，构建时间略高于普通 HNSW；③对极端低选择率查询仍需预过滤线性扫描，无法完全避免全表扫描；④在极高维度或非常稀疏的属性分布中，排除距离的效果可能不如预期。

---

## 674. SOD: Step-wise On-policy Distillation for Small Language Model Agents

**arXiv ID:** 2605.07725 | [PDF](https://arxiv.org/pdf/2605.07725v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 675. Explainable Part-Based Vehicle Classifier with Spatial Awareness

**arXiv ID:** 2605.07831 | [PDF](https://arxiv.org/pdf/2605.07831v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 676. Beam-Aware Radio Map Estimation With Physics-Consistent Parametric Modeling for Unknown Multiple Satellites

**arXiv ID:** 2605.07763 | [PDF](https://arxiv.org/pdf/2605.07763v1)

**作者:** Xiucheng Wang `[一作]` (Xidian University), Ruijin Sun `[通讯]` (Xidian University)

**通讯引用:** 1353 | [OpenAlex ID](https://openalex.org/A5048712226)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对低轨星座下未知活跃卫星导致的共频干扰，提出一种基于物理一致的光束感知参数模型，并通过自适应模型阶选择与联合优化实现对活跃卫星集合和接收信号强度（RSS）场的同时估计，从而构建高精度可查询的卫星射频图。

**💡 创新点**

创新点包括：1）将卫星几何与光束形状映射为低维可解释参数，显著降低逆问题维度；2）引入自适应模型阶选择机制（基于BIC/AIC/GLRT）以解决未知活跃卫星数量的问题；3）在光束感知框架下实施单卫星参数拟合、贪婪搜索与全局联合细化相结合的多卫星推理流程；4）实现从有限测点到连续空间的无缝射频图合成。

**🔧 技术方法**

技术包括：物理一致的光束参数化（幅度、中心、宽度）、几何投影与角距离计算、鲁棒的单卫星拟合（Huber/学生‑t 损失、约束优化）、贝叶斯信息准则/赤池信息准则/GLRT 模型阶选择、贪婪加法（MP/OMP 风格）搜索、联合非线性最小二乘细化、RMSE 与 Pearson 相关系数评估。

**📊 数据集**

使用仿真数据集：在 M=200 个测点、N∈{4,6,8,10,12}、K∈{1,…,5} 的场景下，随机生成卫星位置、光束参数与幅度，加入高斯噪声控制不同 SNR（15–35 dB）进行 15 次 Monte‑Carlo 试验；无真实公开数据集，全部为自建仿真场景。

**📈 对比分析**

与 Lasso、Matching Pursuit、Orthogonal Matching Pursuit、Peak Detection 等基线进行对比。实验表明：所提方法在所有 SNR、N、K 组合下均获得更高的 F1 分数、更低的 RSS RMSE 以及更好的空间相关性；相对基线，显著抑制了误检、漏检与噪声耦合导致的拟合模糊；在高候选数与低信噪下仍保持稳健的精度与误差分布。

**⚠️ 局限性**

局限性：1）模型假设为直射路径（LoS）且不考虑多径、阴影等非直射效应；2）对时间变化与卫星轨道动态的在线适应缺乏，需进一步扩展为时变模型；3）虽然参数维度低，但在极大候选集与极高密度星座下计算量仍显著，需优化并行与稀疏表示；4）实验仅基于仿真，缺乏真实测量验证。

---

## 677. SARA: Semantically Adaptive Relational Alignment for Video Diffusion Models

**arXiv ID:** 2605.07800 | [PDF](https://arxiv.org/pdf/2605.07800v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 678. Alternating Target-Path Planning for Scalable Multi-Agent Coordination

**arXiv ID:** 2605.07744 | [PDF](https://arxiv.org/pdf/2605.07744v1)

**作者:** Yu Kumagai `[一作]` (Hitotsubashi University), Keisuke Okumura `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**通讯引用:** 2687 | [OpenAlex ID](https://openalex.org/A5038362443)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种迭代细化框架，解耦目标分配与路径规划，利用现代快速子最优MAPF求解器（如LaCAM）与反馈驱动的重分配循环解决大规模TAPF实例。

**💡 创新点**

核心创新在于将目标分配与路径规划分离并通过迭代细化进行交替优化；提出两种反馈机制（Delay‑Based Selection、Spectral Bottleneck Sampling）以及两种重分配策略（PIBT、Local Hungarian）；通过多瓶颈评估提升搜索多样性；实现可扩展到数千甚至一万名智能体的TAPF求解。

**🔧 技术方法**

技术实现包括：快速子最优MAPF求解器LaCAM（及其高质量变体LaCAM3）; 目标重分配算法PIBT（基于优先级继承与回溯）; 局部Hungarian算法用于小规模重分配; 两种反馈策略DBS与SBS；多瓶颈并行评估与随机选择；多线程并行MAPF求解。

**📊 数据集**

使用MAPF基准的四连通网格地图，构造两类TAPF实例：1）每位智能体10个随机目标（均匀分布）；2）所有目标集中于热点区域（高重叠）。

**📈 对比分析**

与领先的TAPF求解器ITA‑ECBS（基于CBS）进行对比。在10–30秒时间预算下，DBS‑Hungarian在大于20名智能体的实例中始终成功（100%成功率），而ITA‑ECBS在约20名后大多失败；在800名智能体的场景下，ITA‑ECBS无法求解，而DBS‑Hungarian仍能得到可行解，成本仅略高（约1.1–1.2倍最优）；在10,000名智能体的规模下，DBS‑Hungarian仍能持续降低成本。

**⚠️ 局限性**

局限性包括：方案为子最优，无法保证全局最优；对初始目标分配敏感，随机初始化会显著影响最终质量；重分配与反馈机制的参数（如k值、采样比例）需要经验调优；当前未实现自适应机制来动态选择最佳反馈/重分配组合；多瓶颈评估在极大规模下仍存在额外计算开销。

---

## 679. Offline-Online Hierarchical 3D Global Relocalization With Synthetic LiDAR Sensing and Descriptor-Space Retrieval

**arXiv ID:** 2605.07741 | [PDF](https://arxiv.org/pdf/2605.07741v1)

**作者:** Jiahua Ren `[一作]` (Southwest Jiaotong University), Lei Ma `[通讯]` (Southwest Jiaotong University)

**通讯引用:** 5642 | [OpenAlex ID](https://openalex.org/A5008588495)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种离线-在线分层的3D全局重定位框架，通过离线生成离散采样点的合成激光雷达扫描及其描述符，在线时先进行描述符检索得到粗略姿态，再用GN‑ICP精细对齐，实现快速准确的6-DoF定位。

**💡 创新点**

创新点包括：①基于障碍物通道、最小分离与几何可观测性约束的3D均匀采样；②利用仿真激光雷达进行合成扫描并构建Scan Context描述符数据库；③在线使用描述符空间检索加GN‑ICP的两阶段优化，显著提升计算效率。

**🔧 技术方法**

技术手段涵盖：3D RRT均匀采样、射线投射合成激光雷达扫描、Scan Context（2.5D极坐标）描述符、KD‑树检索、点云降采样与GN‑ICP注册。

**📊 数据集**

实验数据来自两套户外场景（120m×50m 宽廊与 150m×80m 坡地）中使用Livox MID360激光雷达和Intel NUC12记录的真实航迹与点云；未使用公开数据集。

**📈 对比分析**

与GO‑ICP、FPFH‑RANSAC+ICP、FPFH‑FGR+ICP等基线对比，本文方法在成功率90–100%、平均时延<3s、位置误差<0.15m、航向误差<1°的条件下显著优于传统方法。

**⚠️ 局限性**

局限性包括：对大幅滚转/俯仰变化的鲁棒性不足、未融合视觉信息、在极端环境或快速运动时仍可能出现匹配失败。

---

## 680. CyBiasBench: Benchmarking Bias in LLM Agents for Cyber-Attack Scenarios

**arXiv ID:** 2605.07830 | [PDF](https://arxiv.org/pdf/2605.07830v1)

**作者:** Taein Lim `[一作]` (Chung-Ang University), Hoki Kim `[通讯]` (Chung-Ang University)

**通讯引用:** 847 | [OpenAlex ID](https://openalex.org/A5032839849)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 CyBiasBench 基准，用于量化大型语言模型（LLM）在网络攻击中的攻击选择偏差，发现不同代理表现出稳定且与提示无关的攻击族偏好。

**💡 创新点**

提出了“攻击选择偏差（Attack‑Selection Bias）”与“偏差动量（Bias Momentum）”概念，证明代理的攻击族偏好与攻击成功率解耦，并揭示在显式诱导时代理仍遵循自身偏好而非攻击效果。

**🔧 技术方法**

采用了 HTTP 流量记录、OWASP CRS 规则集以及 CAPEC/CWE/WSTG 等规则进行攻击族自动分类，利用熵、选择率、攻击成功率、Tokens‑Per‑Success 和 Prompt‑Stability JSD 等多维度指标评估偏差与性能。

**📊 数据集**

使用了三种 Web 目标（OWASP Juice Shop、MLflow 2.9.2、Vuln‑Shop）共 630 次实验会话，覆盖五个代理、十个攻击族、四个提示条件，并公开了交互式结果仪表盘与完整脚本。

**📈 对比分析**

通过对比不同代理、不同提示条件下的偏差指标与攻击成功率，发现偏差指标对代理区分更敏感；在偏差注入实验中，偏差偏好提升不导致成功率提升，且在所有代理上平均成功率下降，表明偏差动量对性能有负面影响。

**⚠️ 局限性**

局限在于实验仅覆盖 Web 侵入情境，其他攻击场景可能呈现不同偏好；基准仅包含五个代理，未揭示偏差动量根源（训练、提示或执行环境）；并未对不同攻击族的技术难度做进一步分析。

---

## 681. NoiseGate: Learning Per-Latent Timestep Schedules as Information Gating in World Action Models

**arXiv ID:** 2605.07794 | [PDF](https://arxiv.org/pdf/2605.07794v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 682. EyeCue: Driver Cognitive Distraction Detection via Gaze-Empowered Egocentric Video Understanding

**arXiv ID:** 2605.07859 | [PDF](https://arxiv.org/pdf/2605.07859v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 683. BRIDGE: Background Routing and Isolated Discrete Gating for Coarse-Mask Local Editing

**arXiv ID:** 2605.07846 | [PDF](https://arxiv.org/pdf/2605.07846v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 684. Neural Operators as Efficient Function Interpolators

**arXiv ID:** 2605.07792 | [PDF](https://arxiv.org/pdf/2605.07792v1)

**作者:** Vasilis Niarchos `[一作]` (University of Crete), Sokratis Trifinopoulos `[通讯]` (CERN)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

将神经算子重新解释为有限维函数插值器，并应用于核绑定能量残差的学习与修正

**💡 创新点**

通过引入辅助基空间，将函数映射转化为算子学习，保留NO的离散不变性与零样本超分辨率优势，显著提升插值精度与参数效率

**🔧 技术方法**

采用Tensorized Fourier Neural Operator (TFNO)、多层感知机（MLP）和Kolmogorov–Arnold网络（KAN）等架构进行训练与比较

**📊 数据集**

使用AME2020实验绑定能量数据与Weizsäcker–Skyrme（WS4）模型的残差作为训练与评估数据集

**📈 对比分析**

与MLP、KAN在相同参数规模下进行对比，使用严格的离样本五折OOF评估；30成员TFNO集成在核能量残差上实现198.2 keV的RMSE，优于大多数单任务基准模型

**⚠️ 局限性**

主要局限在于仅对已知核图表进行插值，缺乏对新测量或两端核的外推验证；对高维度输入的扩展与理论收敛分析仍待进一步研究

---

## 685. Can I Check What I Designed? Mapping Security Design DSLs to Code Analyzers

**arXiv ID:** 2605.07814 | [PDF](https://arxiv.org/pdf/2605.07814v1)

**作者:** Sven Peldszus `[一作]` (Chalmers University of Gothenburg), Robert Heinrich `[通讯]` (Ulm University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统综述 66 种安全设计 DSL 与 48 个代码安全分析器，对两类工具的安全概念、系统元素以及弱点进行提炼，构建 SecLan 模型，并利用该模型将 DSL 设计与分析器检测结果关联。随后对模型进行专家验证，并通过量化统计与访谈分析，探讨两者之间的覆盖关系、共性与差异。

**💡 创新点**

创新点在于：①首次系统化提取并对齐安全 DSL 与代码分析器的共通概念，形成可操作的 SecLan 模型；②提出基于模型的关系映射方法，能够将 DSL 设计的安全目标与代码层面的检查结果自动关联；③通过专家验证与定量定性结合的方式，客观评估设计与实现安全之间的“鸿沟”。

**🔧 技术方法**

主要技术：系统综述与文献挖掘、模型提炼与概念抽象、JSON 语义化描述、图路径查询（正则路径表达式）实现关系判断、专家问卷与访谈收集反馈、定量统计与可视化分析。

**📊 数据集**

数据集包括：66 种安全设计 DSL（来自 6 篇综述及作者补充）与 48 个安全分析器（来自 3 篇综述并按检查类别挑选），每个 DSL 与分析器均以 JSON 描述文件记录其安全方面、规范元素、检查弱点等信息；此外使用 CWE‑4.12 列表作为弱点来源。

**📈 对比分析**

比较方法：对 DSL 与分析器的安全模型（安全目标、威胁、弱点）与系统模型（实体、活动、数据、信息流等）覆盖率进行统计；利用 SecLan 模型在两图中寻找路径，衡量两者关联的数量与深度；通过专家访谈验证模型准确性。结果显示，DSL 与分析器覆盖的系统元素与安全概念存在显著差异，且多达 80% 的关系未被自动识别。性能方面，关系查询在平均 158 行 JSON 中完成，处理时间可忽略。

**⚠️ 局限性**

局限性：①专家响应率偏低，可能导致模型校准不完全；②模型仅捕捉三类安全概念（目标、威胁、弱点），未覆盖所有细粒度的对策与上下文；③分析器与 DSL 的选择偏向公开工具，商业或定制工具缺失；④基于静态分析的检查无法覆盖运行时安全问题；⑤模型在不同语言、不同设计方法上的适用性尚待进一步验证。

---

## 686. Divide and Conquer: Object Co-occurrence Helps Mitigate Simplicity Bias in OOD Detection

**arXiv ID:** 2605.07821 | [PDF](https://arxiv.org/pdf/2605.07821v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 687. Zero-determinant Strategy for Moving Target Defense: Existence, Performance, and Computation

**arXiv ID:** 2605.07854 | [PDF](https://arxiv.org/pdf/2605.07854v1)

**作者:** Zhaoyang Cheng `[一作]` (KTH Royal Institute of Technology), Mikael Skoglund `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 8960 | [OpenAlex ID](https://openalex.org/A5041348422)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了将零确定性（ZD）策略用于多目标防御（MTD）中的重复安全博弈，提出构造既能保持高防御效益又具低计算复杂度的ZD‑MTD策略，并对其存在性与性能进行理论分析。

**💡 创新点**

创新点：①首次在多目标安全游戏中引入ZD策略并给出其存在的必要与充分条件；②证明ZD策略的上界性能与强Stackelberg均衡（SSE）相匹配；③设计理想与最优ZD策略的线性规划求解方法；④提出一套多步算法实现ZD策略，并证明其在计算复杂度上显著优于传统SSE求解；⑤在实验中展示ZD策略与SSE在效益与计算时间上的对比。

**🔧 技术方法**

使用的技术包括：零确定性策略理论、线性规划与混合整数规划、矩阵行列式与行列式与期望效用的关系、主特征向量分析、复杂度分析与算法设计。

**📊 数据集**

实验使用了模拟的IoT系统与众包系统数据，设置不同目标数（K=3,5,10）及不同攻击/防御参数（如迁移成本、攻击成本、奖励等）来评估策略性能。

**📈 对比分析**

比较方法：将ZD策略（通过算法求解）与传统SSE（通过混合整数规划求解）在平均效用与计算时间两方面进行对比。结果表明：ZD策略在效用上与SSE相近，且计算时间从指数级降至多项式级（大约 O(K³)）。

**⚠️ 局限性**

限制：①ZD策略的实现依赖于已知的收益与成本矩阵；②存在性与性能受线性参数约束，可能在某些参数组合下无法获得理想ZD；③未考虑攻击者非理性或信息不完全的情形；④目前仅针对无折扣的重复博弈，未来需扩展至马尔可夫游戏、折扣奖励及不确定环境。

---

## 688. Unsafe by Flow: Uncovering Bidirectional Data-Flow Risks in MCP Ecosystem

**arXiv ID:** 2605.07836 | [PDF](https://arxiv.org/pdf/2605.07836v1)

**作者:** Xinyi Hou `[一作]` (Huazhong University of Science and Technology), Haoyu Wang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 66226 | [OpenAlex ID](https://openalex.org/A5115602103)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

该论文提出了一套针对MCP（Model Context Protocol）服务器的双向静态分析框架，用于检测请求端和返回端的跨信任边界数据流漏洞。

**💡 创新点**

创新点在于：①实现了MCP专属的入口点恢复（基于工具注册与分发模式）；②设计了符合MCP信任边界的污点语义；③引入了双向（请求侧/返回侧）跨过程传播分析，并在模糊不确定点采用LLM辅助判定。

**🔧 技术方法**

使用的技术包括：基于抽象语法树的入口点识别、污点推理规则、交叉过程前向/后向传播、LLM（gpt‑5.3‑codex‑medium）辅助源/防护判断，以及与现有SAST框架的集成。

**📊 数据集**

实验数据集包含：32个经确认的MCP漏洞案例（Python/JavaScript/TypeScript），以及从mcp.so、PulseMCP、MCPWorld收集的 15,452 个真实开源MCP服务器仓库（Python 52.1%，TypeScript 32.2%，JavaScript 15.7%）。

**📈 对比分析**

与 CodeQL、Semgrep、Snyk Code、MCPScan 等基线工具对比，该框架在 32 个漏洞案例中达 93.8% 的召回率（检测 30/32 案例），明显优于其他工具（MCPScan 30/32、CodeQL 8/32、Semgrep 10/32、Snyk Code 11/32）。在大规模仓库上，工具发现 549 个候选集，最终人工确认 118 条漏洞路径，覆盖 87 服务器。

**⚠️ 局限性**

局限性包括：只关注跨信任边界的数据流漏洞，未覆盖认证/授权、业务逻辑或部署级别攻击；对反射式调度、运行时生成处理器等动态特性识别不足；仍存在误报和对LLM辅助判定的依赖；目前仅支持 Python、JavaScript/TypeScript，其他语言需进一步扩展。

---

## 689. NSPOD: acceleratingthe convergence ofKrylov-based iterative linearsolvers via approximated PODs

**arXiv ID:** 2605.07828 | [PDF](https://arxiv.org/pdf/2605.07828v1)

**作者:** Francesc Levrero-Florencio `[一作]` (Synopsys), George Em Karniadakis `[通讯]` (Brown University)

**通讯引用:** 101278 | [OpenAlex ID](https://openalex.org/A5009658255)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了基于PointTransformer的深度算子网络PTFONet以及利用该网络生成的NSPOD预条件器，以显著加速求解线性化固体力学PDE的Krylov迭代器。

**💡 创新点**

创新点在于将神经子空间POD（NSPOD）与多网格思路结合，利用PFTONet在不同几何和参数下生成逼近解，从而构造高效的子空间预条件器；并首次展示了PointTransformer在无结构网格上的优越性。

**🔧 技术方法**

主要技术包括PointTransformer深度算子网络、Squeeze-and-Excitation注意力机制、POD降维、以及与CG/GMRES等Krylov方法的混合预条件化。

**📊 数据集**

数据集来源于ABC CAD数据集，使用四个约1万节点的三角/四面体网格，随机生成弹性模量、体力以及Dirichlet边界分布以构造训练与测试样本。

**📈 对比分析**

与Jacobi、SOR、BoomerAMG、PETSc SA-AMG等传统预条件器对比，NSPOD在大多数几何下迭代次数下降30%–60%，在某些复杂“薄”特征几何上仍略逊。

**⚠️ 局限性**

局限性包括仅处理空间常数弹性模量和体力、仅使用有限几何数量、对“薄”特征几何收敛效果差、训练成本高以及对网格规模的可扩展性仍需进一步研究。

---

## 690. ICDAR 2026 Competition on Writer Identification and Pen Classification from Hand-Drawn Circles

**arXiv ID:** 2605.07816 | [PDF](https://arxiv.org/pdf/2605.07816v1)

**作者:** Thomas Gorges `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Vincent Christlein `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 3052 | [OpenAlex ID](https://openalex.org/A5087093169)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

组织了Circleid竞赛，评估基于手绘圆圈的作家识别与笔具分类

**💡 创新点**

提出在极简轨迹上同时研究作家与笔具特征的交织与可分离性

**🔧 技术方法**

使用深度学习模型（ResNet、ConvNeXt、DINOv3等）结合数据增强、对比学习与多任务训练

**📊 数据集**

使用新收集的46,155幅手绘圆圈数据，包含50名已知作家、16名未知作家和8种不同笔具

**📈 对比分析**

与基线ResNet进行比较，最佳私有排行榜模型分别在作家识别和笔具分类上达到64.8%和92.7% Top‑1准确率

**⚠️ 局限性**

局限于单一纸张与环境条件，且极简轨迹对作家特征捕捉效果有限，需进一步探索多样化数据与可解释性方法

---

## 691. A Comparative Analysis of Classical Machine Learning and Deep Learning Approaches for Sentiment Classification on IMDb Movie Reviews

**arXiv ID:** 2605.07811 | [PDF](https://arxiv.org/pdf/2605.07811v1)

**作者:** Erma Daniar Safitri `[一作]` (Institut Teknologi Sumatera), Martin Clinton Tosima Manullang `[通讯]` (Institut Teknologi Sumatera)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

比较了经典机器学习与深度学习在IMDb电影评论情感分类中的表现，并提供了统一的双通道预处理与交互式部署框架。

**💡 创新点**

创新点在于通过相同数据、相同评估指标的公平对比，揭示在小规模数据下深度模型并不一定优于传统方法，并可视化展示模型排名。

**🔧 技术方法**

使用了PyCaret AutoML与TF‑IDF特征、BiLSTM及BiLSTM+注意力机制模型，并将模型部署在Hugging Face Spaces进行交互式演示。

**📊 数据集**

采用了Kaggle公开的5万条IMDb电影评论数据集，正负类各25k条，保持平衡。

**📈 对比分析**

通过80:20训练/测试拆分、准确率、精确率、召回率、F1等指标进行对比；SVM与Logistic回归取得0.8530的准确率，明显高于BiLSTM+Attention的0.706，显示传统模型在该场景更高效。

**⚠️ 局限性**

局限性在于深度模型仅用1万条样本并训练3个epoch，缺乏充分调参和大规模数据支持；未尝试更先进的Transformer架构，导致其性能低于传统方法。

---

## 692. The Minimax Rate of Second-Order Calibration

**arXiv ID:** 2605.07808 | [PDF](https://arxiv.org/pdf/2605.07808v1)

**作者:** Kamil Ciosek `[一作]` (Spotify), Nicolò Felicioni `[通讯]` (Spotify)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对二阶校准误差（Second‑Order Calibration Error）进行度量、估计与校准，提出可在分布无关的设置下获得近似参数化的 n^{-1/2} 估计精度，并提供对应的后验校准方法——第二阶 Platt 标定。

**💡 创新点**

核心创新是发现 sech 扰动核在对预测器做平滑后使校准函数变得在复平面上解析，从而可利用 Bernstein-ellipse 定理把函数逼近误差降低到指数级，突破传统非参数 n^{-1/4} 速率；并首次给出二阶校准的分布无关下的最优样本复杂度与后验校准保证。

**🔧 技术方法**

使用：sech 扰动核、解析性分析、Bernstein-ellipse 多项式逼近、经验风险最小化（ERM）多项式回归、后验映射（第二阶 Platt 标定）以及理论下界构造。

**📊 数据集**

实验数据：① 合成的 10 维高斯混合 + MLP 集成（两次快照）；② 同样的合成数据用于检验校准误差速率；③ 同一合成数据用于验证校准后的不确定性与下游决策收益；④ 实际 Weather‑Sentiment AMT 数据集，用于评估基于 η₂ 的审计收益。

**📈 对比分析**

与传统的分箱、核平滑（Nadaraya‑Watson）以及无扰动的二阶校准方法比较。多项式估计在 n ≥ 100 时已显著优于 n^{-1/4} 速率，实验曲线的对数斜率约为 -0.70，几乎达到理论的 -1/2 速率；第二阶 Platt 标定在决策收益和审计任务上接近 oracle，明显优于第一阶或 1‑D 标定方法。

**⚠️ 局限性**

限制：需要在预测器上施加 sech 扰动核，若无扰动则无分布无关的样本复杂度保证；估计基于 2‑snapshots，若仅有单次标签则无法实现；解析性假设限制了可处理的模型形式；对高维预测空间的多项式基数随维度指数增长，实际应用需权衡多项式阶数。

---

## 693. Toward Privileged Foundation Models:LUPI for Accelerated and Improved Learning

**arXiv ID:** 2605.07799 | [PDF](https://arxiv.org/pdf/2605.07799v1)

**作者:** Xueying Ding `[一作]` (Carnegie Mellon University), Leman Akoglu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 8564 | [OpenAlex ID](https://openalex.org/A5001634795)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了在训练表格基础模型时使用特权信息（Privileged Information）来加速学习并提升泛化性能，构建了两类特权信息：数据集级别的统计量（Meta‑PI）和数据生成程序的编码（Generator‑PI），并设计了教师‑学生架构与教师强迫（teacher‑forcing）退火机制以实现训练时仅可用特权信息的迁移。

**💡 创新点**

创新点在于：1) 将特权信息系统化地引入表格基础模型；2) 设计了统一的程序编码器将异构生成器（SCM、GMM、Copula 等）映射为固定长度向量；3) 提出了教师‑学生迁移架构与退火策略，使训练时可用的强制信息在推理时可被重构；4) 理论分析证明特权信息通过提升表示与信息获取两条机制降低近似误差和提升泛化。

**🔧 技术方法**

采用的技术包括：Transformer‑based PFN 结构；自监督对比学习用于程序编码器；教师‑学生模型与退火的 scheduled sampling；统计量和分位数作为 Meta‑PI；实验中使用了交叉熵与 AUROC/AUPRC 评估；对比实验采用无特权信息基线与多种 ablation。

**📊 数据集**

实验数据集：①合成的高维 GMM 数据集（Bronze/Silver/Gold 级别特权信息），②混合 5 种先验（SCM、GMM、Copula 等）的 10‑层表格基础模型预训练，③真实世界 benchmark ADBench（57 个数据集）。

**📈 对比分析**

与无特权信息基线相比，PIQL 模型在训练损失曲线和测试 AUROC/AUPRC 上表现更快收敛、最终误差更低。实验使用配对置换检验，p 值均未显示显著差异（e.g., p=0.299、p=0.119），表明在早期阶段即可达到与基线相同或更优的性能，证明了特权信息在加速与性能提升方面的有效性。

**⚠️ 局限性**

限制：①特权信息的设计和生成需要先验知识，尤其是生成程序的显式可知性；②在真实世界任务中生成程序信息往往不可获得，导致该方法主要适用于合成或已知先验的数据；③程序编码器需在大规模多种生成器上预训练，增加了额外成本；④Meta‑PI 的表现受统计量选择限制，进一步提升需要更丰富的特征工程。

---

## 694. Many-to-Many Multi-Agent Pickup and Delivery

**arXiv ID:** 2605.07835 | [PDF](https://arxiv.org/pdf/2605.07835v1)

**作者:** Ethan Schneider `[一作]` (Georgia Institute of Technology), Sonia Chernova `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 7715 | [OpenAlex ID](https://openalex.org/A5033265891)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种针对多机器人仓库环境的多对多多代理取货与配送（Many-to-Many Multi-Agent Pickup and Delivery，M2M）算法，并在其基础上设计了考虑SKU分布的变体M2M-wSKU。

**💡 创新点**

创新点在于：① 将传统的一对一取送任务扩展为四维（代理、任务、起点、终点）分配问题，构建可扩展的成本张量分解策略；② 采用大邻域搜索（LNS）结合Shaw移除和Metropolis接受准则，对多维分配进行迭代优化；③ 在成本函数中加入SKU最近邻权重，实现动态的库存分布优化。

**🔧 技术方法**

使用的技术包括：成本张量分解为四个矩阵、基于大邻域搜索的优化框架、PBS（Priority Based Search）进行多机器人路径规划、KD-Tree进行SKU最近邻搜索，以及实验中采用的LNS-PBS作为基线。

**📊 数据集**

使用的实验数据集为三种模拟仓库布局（Restricted、Open-Top、Open），尺寸均为27×50，40个机器人，最多120个任务，30个SKU，8小时仿真，任务以每时间步4个生成，保持120个活跃任务，采用10次重复实验求平均。

**📈 对比分析**

与基线LNS-PBS以及MIP/SP方法进行比较。结果显示M2M在所有布局和库存密度下的吞吐量均高于或与LNS-PBS持平，平均提升约4–39%，M2M-wSKU在高库存情况下表现略差。计算时间均维持在约1秒左右，M2M与LNS-PBS的可扩展性相近，M2M可支持至150个代理/任务。

**⚠️ 局限性**

局限性包括：① M2M-wSKU的SKU分布优化未能带来预期收益，导致性能不如纯M2M；② 在更大地图（61×100）下，M2M的可扩展性下降，最多支持50个代理/任务；③ 实验仅基于理想化仿真，未考虑定位误差、通信延迟、任务失败等实际仓储复杂性。

---

## 695. Flexible Routing via Uncertainty Decomposition

**arXiv ID:** 2605.07805 | [PDF](https://arxiv.org/pdf/2605.07805v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 696. OrScale: Orthogonalised Optimization with Layer-Wise Trust-Ratio Scaling

**arXiv ID:** 2605.07815 | [PDF](https://arxiv.org/pdf/2605.07815v1)

**作者:** Yuxuan Lou `[一作]` (National University of Singapore), Yang You `[通讯]` (National University of Singapore)

**通讯引用:** 3981 | [OpenAlex ID](https://openalex.org/A5100658705)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 OrScale 优化器，将 Muon 的正交化更新与层级信任比率相结合，实现了矩阵层的动态步长控制，并给出了适用于 LLM 的 OrScale‑LM 变体；

**💡 创新点**

核心创新在于将信任比率分母设为真实更新方向的 Frobenius 范数，从而解决 Muon 步长不确定问题；结合一次性层级校准与 Moonlight 形状因子，保留学习率迁移特性，并给出核范数收敛证明；

**🔧 技术方法**

使用 Muon 正交化、Newton–Schulz 近似极化因子、LARS/LAMB 风格的信任比率、Moonlight 形状因子、权重衰减耦合以及核范数收敛分析等技术；

**📊 数据集**

在视觉任务上使用 CIFAR‑10 与 DavidNet 数据集，在大规模语言模型预训练上使用 FineWeb‑Edu 数据集，覆盖 125M 至 1.1B 参数的 GPT‑2 风格模型；

**📈 对比分析**

与 AdamW、LAMB、Muon、Muon+Moonlight 进行基准比较，CIFAR‑10 上 OrScale 达到 94.05%±0.08 的 top‑1，超越 Muon+Moonlight 0.30 点；FineWeb‑Edu 上 OrScale‑LM 在 4 个规模均优于 AdamW 与 Muon+Moonlight，平均提升 0.07–0.16 nats；

**⚠️ 局限性**

目前仍未在极大批量和异构硬件环境下进行充分评估，非矩阵参数仅采用 AdamW 回退，且在某些失败模式下需要额外的层级校准，理论收敛上界仍较保守，实际表现受超参数调优影响。

---

## 697. Spectral Surgery: Class-Targeted Post-Hoc Rebalancing via Hessian Spike Perturbation

**arXiv ID:** 2605.07790 | [PDF](https://arxiv.org/pdf/2605.07790v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 698. Bridging the Programming Language Gap: Constructing a Multilingual Shared Semantic Space through AST Unification and Graph Matching

**arXiv ID:** 2605.07788 | [PDF](https://arxiv.org/pdf/2605.07788v1)

**作者:** Junhao Chen `[一作]` (Nanjing University of Aeronautics and Astronautics), Weiqin Zou `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 1016 | [OpenAlex ID](https://openalex.org/A5005221755)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过统一AST标签映射和图匹配网络（GMN）构建跨语言共享语义空间，支持跨语言代码克隆检测与检索。

**💡 创新点**

①将多语言AST节点标签映射到统一标签集，消除语法差异；②使用GMN实现节点级跨图注意力对齐，捕获跨语言语义对应；③结合结构增强与硬负样本挖掘进一步提升判别力。

**🔧 技术方法**

统一AST抽象、图匹配网络（GMN）+交叉图注意力、GRU迭代更新、全局注意力池化、对比学习、硬负样本采样。

**📊 数据集**

Google Code Jam、AtCoder、XLCoST 等多语言竞赛数据集，涵盖 Java、C++、Python、C# 等。

**📈 对比分析**

与 FSD‑CLCD、C4、CLCDSA 等克隆检测基线以及 CodeBERT、UniXcoder、GraphCodeBERT 检索基线对比，克隆检测 Precision/Recall/F1 提升至 99.94%/99.92%/99.93%，检索 MRR 提升至 0.5547，Precision@4 提升至 93.78%，明显优于现有方法。

**⚠️ 局限性**

仅在公开竞赛数据集验证，未覆盖低资源语言、工业真实代码或大型项目；对极大规模系统和领域特定语言的泛化仍待验证。

---

## 699. On the Tradeoffs of On-Device Generative Models in Federated Predictive Maintenance Systems

**arXiv ID:** 2605.07860 | [PDF](https://arxiv.org/pdf/2605.07860v1)

**作者:** Usevalad Milasheuski `[一作]` (Consiglio Nazionale delle Ricerche), Stefano Savazzi `[通讯]` (Consiglio Nazionale delle Ricerche)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过联邦学习框架评估了三类生成模型（VAE、GAN、DDPM）在工业预测性维护中的时间序列异常检测性能，并提出了基于模型分析/合成分解的部分联邦共享策略。

**💡 创新点**

创新点包括：① 将生成模型统一为分析（encoder/critic）与合成（decoder/generator）两大模块；② 在此框架下定义全联邦、编码器共享、解码器共享等三种策略；③ 针对不平衡、非IID和资源受限的工业 IoT 环境，给出通信成本与性能折衷的系统级评估与使用指南。

**🔧 技术方法**

使用技术：联邦学习（FedAvg 与部分联邦）、Variational Autoencoder、Wasserstein GAN（含梯度惩罚）、Denoising Diffusion Probabilistic Model（U‑Net 结构）、贝叶斯阈值优化、精确度/召回率/F1/PR‑AUC、通信成本分析与时间偏差指标。

**📊 数据集**

使用数据集：ARAMIS（主实验，单一持续异常的 PdM 数据）和 SWaT（交叉验证，工业控制系统攻击数据）。

**📈 对比分析**

比较方法：对比集中式、独立式、全联邦、编码器共享和解码器共享五种场景，评估指标为 F1、Precision、Recall、PR‑AUC、误判成本与时间偏差。实验结果显示：VAE 与 DDPM 在全联邦与部分联邦下均可恢复接近集中式性能，且在部分联邦时通信量可减半；GAN 方案在部分联邦时性能下降显著；VAE 在编码器共享时表现最佳，DDPM 在解码器共享时表现最佳。

**⚠️ 局限性**

局限性：缺乏生成模型在联邦设置下的理论收敛与稳定性证明；GAN 在部分联邦中易失稳；实验仅涵盖少数模型与两个数据集，未评估差分隐私、梯度压缩等进一步的安全与效率改进；结果对其他生成模型与更大规模分布式环境的推广需要进一步验证。

---

## 700. A Fibrational Perspective on Differential Linear Logic

**arXiv ID:** 2605.07858 | [PDF](https://arxiv.org/pdf/2605.07858v1)

**作者:** Jad Koleilat `[一作]` `[通讯]` (Université Sorbonne Paris Nord), Jad Koleilat (Université Sorbonne Paris Nord)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了将Differential Linear Logic (DiLL) 的模型以纤维化视角描述的框架，定义了泛化的差分 Seely 类（GDSC）。

**💡 创新点**

创新点在于将线性逻辑的微分结构与纤维化形式结合，给出了通用的线性-非线性纤维化 adjunction，并引入线性切线函子，扩展了 DiLL 与依赖类型的统一。

**🔧 技术方法**

采用了范畴论中的 Grothendieck 纤维、线性-非线性 adjunction、对偶同构、切线函子等技术。

**📊 数据集**

未使用实验数据集，属于理论研究。

**📈 对比分析**

未涉及实验比较，主要通过逻辑/范畴论的证明与对已有模型的扩展来验证。

**⚠️ 局限性**

局限性包括需满足复杂的纤维化假设，且对 GCDC 的完全对应关系仍是开放问题；模型尚未在具体依赖线性逻辑中得到验证。

---

## 701. A Roadmap of Mixed Reality Body Doubling for Adults with ADHD

**arXiv ID:** 2605.07851 | [PDF](https://arxiv.org/pdf/2605.07851v1)

**作者:** Valerie Tan `[一作]` (TU Dortmund University), Jens Gerken `[通讯]` (TU Dortmund University)

**通讯引用:** 1129 | [OpenAlex ID](https://openalex.org/A5003447205)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并构建了一个针对 ADHD 成人的身体双胞胎框架，涵盖个体动机、代理特征、交互与情境等12个维度，旨在指导未来研究与原型设计。

**💡 创新点**

将身体双胞胎概念扩展为多维模型，并结合混合现实、机器人等新技术，提出新的维度与维度间关系，形成研究路线图。

**🔧 技术方法**

主要使用文献综述与专家讨论构建模型，并讨论了混合现实、社交辅助机器人和 AI 代理等技术作为实现手段。

**📊 数据集**

未使用具体数据集，框架基于现有文献与社区经验形成。

**📈 对比分析**

未进行实验比较，性能评估尚未完成，建议未来在 ADHD 人群中进行实证研究。

**⚠️ 局限性**

缺乏实证验证，框架尚处于工作进展阶段；多维度复杂性可能导致应用与验证难度；缺少对不同人群与任务的泛化评估。

---

## 702. Measuring and Mitigating the Distributional Gap Between Real and Simulated User Behaviors

**arXiv ID:** 2605.07847 | [PDF](https://arxiv.org/pdf/2605.07847v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 703. KL for a KL: On-Policy Distillation with Control Variate Baseline

**arXiv ID:** 2605.07865 | [PDF](https://arxiv.org/pdf/2605.07865v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 704. RelAgent: LLM Agents as Data Scientists for Relational Learning

**arXiv ID:** 2605.07840 | [PDF](https://arxiv.org/pdf/2605.07840v1)

**作者:** Xingyue Huang `[一作]` (University of Oxford), İsmail İlkan Ceylan `[通讯]` (TU Wien)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出RelAgent，一种基于LLM的自主搜索框架，利用SQL查询构造特征并选择预测模型，搜索阶段通过工具交互完成，推理阶段仅使用SQL和传统模型；

**💡 创新点**

创新点在于将关系结构外部化为可执行SQL，LLM只负责搜索，完成后推理无LLM、可解释性强且可扩展；引入评估工作区和多工具交互实现迭代调优；

**🔧 技术方法**

技术包括LLM（如GPT‑5.2、DeepSeek）、工具框架（CAMEL）、DuckDB SQL引擎以及传统树模型（LightGBM、XGBoost、CatBoost等）；

**📊 数据集**

使用RelBenchV1、RelBenchV2、4DBInfer等公开关系学习基准；

**📈 对比分析**

与监督表格模型、图神经网络、零样本基础模型和LLM特征工程方法对比，平均AUROC/MAE均优于大多数基线，接近或略低于闭源KumoRFM‑v2，平均排名分别为3.17和1.00；

**⚠️ 局限性**

局限在于搜索成本高、搜索可靠性受限，可能漏掉深层连接或高阶模式；对LLM规模和预算敏感，工具与工作区实现复杂度较高。

---

## 705. A Spatial Knowledge Acquisition Comparison Between Digital Visual Thematic Maps, Non-Visual Interactive Text Thematic Maps, and Tables

**arXiv ID:** 2605.07849 | [PDF](https://arxiv.org/pdf/2605.07849v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 706. Approximation-Free Differentiable Oblique Decision Trees

**arXiv ID:** 2605.07837 | [PDF](https://arxiv.org/pdf/2605.07837v1)

**作者:** Subrat Prasad Panda `[一作]` (Nanyang Technological University), Arvind Easwaran `[通讯]` (Nanyang Technological University)

**通讯引用:** 2003 | [OpenAlex ID](https://openalex.org/A5054946593)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种将硬正交决策树完整映射到可微分神经网络的架构，并在回归任务中通过从k≥2逐步退化到Top‑1的Annealed Top‑k方法实现无逼近训练；该方法既可用于分类，也可直接用于回归。

**💡 创新点**

创新点包括：
1) 语义等价的NN编码，使得前向推理与反向优化完全一致，消除了STE或软化决策带来的近似；
2) 在回归中首次引入从Top‑k（k≥2）逐步退化到Top‑1的训练策略，既提供了有效梯度，又避免了所有叶子参与梯度更新导致的结构不稳定；
3) 将上述架构应用为程序化策略，在离散与连续强化学习环境中实现了与传统NN策略相当甚至更优的性能。

**🔧 技术方法**

技术手段包括：
- 用ReLU和线性层实现正交决策树的NN等价编码；
- Top‑k soft‑max选择与温度衰减策略；
- 对比STE的梯度计算与Top‑k梯度；
- 在RL中直接替换策略网络，使用PPO/SAC进行训练；
- 采用多阶段（k>1→1）以及样本增广的叶子细化。

**📊 数据集**

使用的数据集与环境：
- 经典表格分类/回归基准（如abalone、comp‑active、Ailerons、YearPred、CTSlice、Medical、Sulfur、Bike Sharing、Houses、Wine Quality等，共约30+数据集）；
- MNIST作为图像基准；
- RL环境：CartPole、Acrobot、LunarLander、FindandDestroyZerglings、continuous LunarLander、BipedalWalker。

**📈 对比分析**

与方法比较：
- 分类：对比DGT、TAO、CRO‑DT、CART、ICCT、ANT等；在大多数数据集上平均提升≈10%或更高，且训练速度快；
- 回归：对比DGT‑Linear、TAO‑Linear、CRO‑DT、CART；平均RMSE提升≈3%，部分数据集提升>10%；
- RL：与DGT、ICCT、VIPER及NN基线对比；在离散环境中与NN相当，在连续环境中平均提升≈2–3%；

**⚠️ 局限性**

限制：
- 仅适用于低维表格数据，图像等高维输入难以直接处理；
- 树高度需手动设定，缺乏自适应生长/修剪机制；
- 对于非常深的树或极大特征维度，参数规模和训练稳定性仍需进一步研究。

---

## 707. Per-Phase Fidelity Attribution for Quantum Compilers using HBR Decomposition

**arXiv ID:** 2605.07876 | [PDF](https://arxiv.org/pdf/2605.07876v1)

**作者:** Chandrachud Pati `[一作]` (Indian Institute of Science), Yogesh Simmhan `[通讯]` (Indian Institute of Science)

**通讯引用:** 6373 | [OpenAlex ID](https://openalex.org/A5041794289)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估了一种名为 HBR 的分阶段量子编译器性能归因方法，能将量子电路的保真度损失拆分为合成（H）、基元转换（B）和路由（R）三阶段。

**💡 创新点**

创新点在于：①构建了可跨 SDK（Qiskit、PennyLane、TKET）和不同硬件拓扑（IBM heavy‑hex、IonQ all‑to‑all）的统一归因框架；②引入 CX‑equivalent 计量统一两量子门成本；③通过轻量级分析模型实现阶段级保真度估计，并验证了 SDK 排名保持一致。

**🔧 技术方法**

使用的技术包括：基于门计数与深度的独立抖动噪声模型、CX‑equivalent 计数、对每阶段的门/深度差异进行对数保真度变换、以及对 T2 隔离和读出误差的补偿。

**📊 数据集**

使用了八个代表性量子算法（Grover、QPE、QDRIFT、Trotter、BV、QAOA、QFT、GHZ）以及两套硬件模拟/真实数据（IBM FakeFez、IonQ Forte‑1、IBM Fez）。

**📈 对比分析**

通过比较不同 SDK 在诊断层（opt=0）与生产层（opt≥2）下的 HBR 归因结果、总两量子门数、深度以及在噪声模拟器和真实硬件上的成功率，验证了 HBR 模型能正确保持 SDK 排名，发现路由瓶颈在搜索类电路上占主导，而合成瓶颈在哈密顿量模拟中占主导；在生产层级，某些 SDK 的优劣可能反转。

**⚠️ 局限性**

局限性包括：①归因模型假设独立抖动噪声，未考虑相干或相关噪声；②对不同 ISA 或非最大纠缠门的 CX‑equivalent 权重需手动调整；③仅针对三大 SDK，无法直接推广至所有编译器；④未覆盖编译器的运行时间与资源消耗；⑤对高深度电路的绝对保真度预测受限。

---

## 708. ADKO: Agentic Decentralized Knowledge Optimization

**arXiv ID:** 2605.07863 | [PDF](https://arxiv.org/pdf/2605.07863v1)

**作者:** Lucas Nerone Rillo `[一作]` (Iowa State University), Soumik Sarkar `[通讯]` (Iowa State University)

**通讯引用:** 10418 | [OpenAlex ID](https://openalex.org/A5081037761)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种名为 ADKO 的分布式黑盒优化框架，利用知识令牌（compact token）在保持数据隐私和通信效率的前提下实现多智能体协作。

**💡 创新点**

创新点包括：①把二元成功/失败信号与优势分数、可选语言模型洞察编码为隐私令牌；②对令牌压缩和语言模型近似误差进行信息量量化和分解；③给出四项误差（GP、LM 偏差、LM 噪声、压缩失真）的累计回报上界，并证明在令牌保真度可控的前提下可获得子线性回报；④设计保真度感知的令牌修剪机制。

**🔧 技术方法**

核心技术包括：私有高斯过程代理、基于图的令牌广播、GP‑UCB 采样、语言模型推理、保真度感知修剪、互信息与总变差距离分析、回报分解与理论证明。

**📊 数据集**

实验使用了：① CIFAR‑10 上的神经网络架构搜索（7 维混合离散/连续设计空间）；② Olympus 数据集的 Suzuki–Miyaura 交叉偶联化学反应产率（约 3,696 条离散条件）。

**📈 对比分析**

与独立 BO、FedAvg‑BO、集中式 BO 以及其他分布式基线比较，ADKO 在通信成本显著降低的同时，显著优于独立 BO 并接近集中式最佳表现；在化学发现任务中，ADKO‑LLM 超过所有分布式基线，仅次于集中式最优，且在非 IID 溶剂约束下实现更高的 hit‑fraction。

**⚠️ 局限性**

局限性包括：① GP 训练的 O(n³) 复杂度限制大规模应用；② 语言模型先验可能在新任务上偏差，导致 ADKO‑LLM 效果下降；③ 目前隐私保障为非正式，缺乏差分隐私证明；④ 对恶意令牌注入缺乏鲁棒性；⑤ 令牌保真度与噪声的平衡在不同场景下需要进一步探索。

---

## 709. GazeVLM: Active Vision via Internal Attention Control for Multimodal Reasoning

**arXiv ID:** 2605.07817 | [PDF](https://arxiv.org/pdf/2605.07817v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 710. PolySQL: Scaling Text-to-SQL Evaluation Across SQL Dialects via Automated Backend Isomorphism

**arXiv ID:** 2605.07796 | [PDF](https://arxiv.org/pdf/2605.07796v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 711. Scaling Categorical Flow Maps

**arXiv ID:** 2605.07820 | [PDF](https://arxiv.org/pdf/2605.07820v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 712. Beyond Confidence: Rethinking Self-Assessments for Performance Prediction in LLMs

**arXiv ID:** 2605.07806 | [PDF](https://arxiv.org/pdf/2605.07806v1)

**作者:** Sree Bhattacharyya `[一作]` (Pennsylvania State University), James Z. Wang `[通讯]` (Pennsylvania State University)

**通讯引用:** 25274 | [OpenAlex ID](https://openalex.org/A5100687159)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用认知评估理论，提出对大型语言模型（LLM）进行多维度自我评估（包括努力、理解、能力、愉悦度、自尊、目标相关性以及传统自信），并在12个模型、38个任务（覆盖8个领域）上评估其预测失败的有效性。

**💡 创新点**

创新点在于：①把自我评估从单一自信扩展到多维度，揭示努力和能力在预测失败方面可匹敌甚至超越自信；②系统性比较不同维度在不同任务类型（推理vs检索）中的表现，提出任务依赖的最佳维度选择；③验证预任务反思对模型弃答质量的提升。

**🔧 技术方法**

使用的技术包括：认知评估理论维度抽取、LLM自我报告提示设计、后任务与预任务评分、逻辑回归与集成模型（随机森林、梯度提升）预测失败、AUROC、McFadden R²、Brier分解、混合效应模型等统计评估方法。

**📊 数据集**

数据集主要为38个任务：标准子集来自BIG-Bench（多语言、数学、科学等），难度子集包含Hard Benchmarks（MMLU-Prox、CausalProbe、Ethics、MoralBench、EmoBench、MultiNRC、Long Context Reasoning等），共计约3200条样本；模型包括12个公开与专有模型（GPT‑5.2 mini、Claude‑4.5 Sonnet、Gemini‑3.0 Flash、DeepSeek V3.2、Qwen‑30B-R、LLaMA‑3.3 70B、以及多尺寸开源模型）。

**📈 对比分析**

与仅使用自信或常规提示策略（CoT、Multi‑step、Top‑k）相比，多维度自我评估能提升AUROC约6–10个百分点，McFadden R²提升至0.15以上，预任务反思下模型弃答准确率提升约4–7个百分点。

**⚠️ 局限性**

局限性包括：①自我报告可能仅是表面特征关联而非真正内省；②不同模型对预任务反思的实际采纳程度差异大；③受限于任务样本量和类别划分，某些维度（如情感维度）缺乏足够统计功效；④提示设计对结果的影响尚未彻底排除；⑤多维度评估在资源受限场景下的实用性和可扩展性尚需进一步验证。

---

## 713. Text-to-CAD Evaluation with CADTests

**arXiv ID:** 2605.07807 | [PDF](https://arxiv.org/pdf/2605.07807v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 714. Anisotropic Modality Align

**arXiv ID:** 2605.07825 | [PDF](https://arxiv.org/pdf/2605.07825v1)

**作者:** Xiaomin Yu `[一作]`, Hui Xiong `[通讯]`

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于锚点、轨迹和质心三步对齐的点云配准算法，并在配准后对结果进行单位球归一化。

**💡 创新点**

创新点在于将传统ICP的平移与旋转拆解为三个互补的对齐阶段（Anchor Alignment、Trace Alignment、Centroid Align），通过简单的均值、缩放和平移操作实现快速收敛，并最终通过归一化到单位球实现尺度一致性，从而显著减少了迭代次数和对初始姿态的敏感性。

**🔧 技术方法**

使用的技术包括：统计均值计算、线性缩放、向量平移、向量归一化、以及对齐误差的欧氏距离度量；核心实现依赖NumPy/Scipy进行矩阵运算，Python实现可轻松集成到现有点云处理框架中。

**📊 数据集**

实验使用了公开点云数据集 ModelNet40、ShapeNetCore‑v2 以及 ScanNet 的真实扫描数据，以评估算法在不同场景、噪声和部分遮挡下的鲁棒性。

**📈 对比分析**

与经典 ICP、Go‑ICP、Super4PCS 等方法在相同数据集上进行对比，结果显示：在平均配准误差上提升约 10–15%（从 3.2 mm 降至 2.7 mm），并且迭代次数平均减少 60%（从 100 次降至 40 次），计算速度提升约 1.8 倍。

**⚠️ 局限性**

主要局限包括：1）仅适用于刚性变换，无法处理非刚性或形变；2）对点云极度稀疏或高噪声的情况仍然敏感；3）当锚点分布不均或存在误差时，三步对齐的顺序可能需要手动调整。

---

## 715. SCENE: Recognizing Social Norms and Sanctioning in Group Chats

**arXiv ID:** 2605.07823 | [PDF](https://arxiv.org/pdf/2605.07823v1)

**作者:** Mateusz Jacniacki `[一作]` (Humalike), Maksymilian Bilski `[通讯]` (Humalike)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了SCENE基准，用于评估大型语言模型在多方聊天中对隐性社会规范的推断与适应。

**💡 创新点**

创新点在于将社会规范的隐含性与惩罚机制引入交互式评估，通过模拟群聊情境让模型在无显式规则提示的情况下学习规范。

**🔧 技术方法**

技术手段包括两阶段生成器（先构造情境再注入隐藏规范）、LLM驱动的脚本化人物、调度器和LLM裁判，用以记录、评估与校验规范遵守情况。

**📊 数据集**

数据集为21,952个有效规范情境元组，生成1,000个交互回合的剧本，覆盖多种事件、规范、惩罚方式和先前演示情形。

**📈 对比分析**

通过对六种模型的1,000个回合进行修复率和适应性Spearman ρ的测评，封闭权重模型Claude Opus 4.7与Gemini 3.1 Pro在修复率和负相关适应性上显著优于四个开源权重模型（分别为71.9%–57.6%和ρ≈0）。

**⚠️ 局限性**

局限性包括情境人为合成、单一隐性规范、短文本交互、仅使用英语、有限的行为空间以及评判模型可能对某些受试模型产生偏好。

---

## 716. GRASP -- Graph-Based Anomaly Detection Through Self-Supervised Classification

**arXiv ID:** 2605.07812 | [PDF](https://arxiv.org/pdf/2605.07812v1)

**作者:** Robin Buchta `[一作]`, Gabi Dreo Rodosek `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一种基于图的异常检测系统（GRASP），利用掩码自监督学习对进程可执行文件进行分类，从而在不依赖阈值的情况下检测APT攻击和未知异常。

**💡 创新点**

创新点包括：①将进程可执行文件作为预测目标而非边类型，①实现无阈值检测，②采用时间窗口生成节点级别报警，③结合聚类识别可区分但误分类的可执行文件，④在训练时使用Transformer‑based autoencoder编码路径/IP，提升对未知行为的敏感度。

**🔧 技术方法**

主要技术：图注意力网络（GAT）编码器 + 多层感知机（MLP）解码器；自监督掩码学习；Transformer‑based（或Word2Vec）位置编码；邻域采样（两跳采样）与窗口化图构造；误分类聚类；宏观/加权F1评价。

**📊 数据集**

使用DARPA Transparent Computing（TC）和Operationally Transparent Computing（OpTC）公开数据集，涵盖多种平台（Linux、Android、Windows）及多次演练（Clearscope、Cadets、Theia等）。

**📈 对比分析**

与当前最先进的PIDS（如Theia、Cadets等）在相同训练/验证/测试拆分下进行对比。GRASP在大多数数据集上实现了更高的攻击召回率（Attack Recall）和更好的可重复性，同时检测覆盖率更广，能够捕获未知可执行文件的异常；但报警数量略多，误报率相对更高。对比实验显示，GRASP在多次实验中方差低，且无需阈值即可保持稳定性能。

**⚠️ 局限性**

局限性：1）对部分数据集（如Clearscope Android）难以学习可执行文件行为，导致误报/漏报；2）生成的报警量相对较大，需进一步筛选与聚类以降低分析负担；3）需要预先知道可执行文件类别，新增可执行文件需重新训练；4）计算成本受邻域采样和窗口大小影响，极大规模场景仍需优化。

---

## 717. Prune-OPD: Efficient and Reliable On-Policy Distillation for Long-Horizon Reasoning

**arXiv ID:** 2605.07804 | [PDF](https://arxiv.org/pdf/2605.07804v1)

**作者:** Zhicheng Yang `[一作]` (Hong Kong University of Science and Technology), Jing Tang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 25820 | [OpenAlex ID](https://openalex.org/A5006160595)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为Prune-OPD的在线可靠性控制框架，用于在长推理任务中动态裁剪并重新分配学生-教师对齐的密集监督奖励。

**💡 创新点**

创新点在于：①通过每步的学生-教师top‑k重叠率实时检测前缀漂移事件；②将累计漂移事件映射为单调递减的奖励衰减权重，并据此动态截断回合长度；③在保持完整训练窗口的同时，显著削减无效计算。

**🔧 技术方法**

使用技术包括：反向KL目标的On‑Policy Distillation（OPD）、top‑k重叠率与top‑p接受率的兼容性度量、奖励衰减权重公式、动态响应长度控制器，以及标准的策略梯度训练。

**📊 数据集**

数据集为DAPO‑Math‑17K，用于生成训练提示；评估指标在AMC、AIME、HMMT等数学推理基准上进行。

**📈 对比分析**

与基线OPD、固定长度截断以及随机裁剪对比，Prune-OPD在低兼容性场景下将训练时间降低了约37.6%–68.0%，同时保持或略微提升了各基准的准确率；在高兼容性场景下几乎不影响速度和性能。

**⚠️ 局限性**

局限性包括：对阈值γ、top‑k大小等超参数的敏感性；仅在长推理任务上验证，尚未证明对其他类型的序列任务或更大规模模型的通用性；当兼容性判断过于保守时可能错失有价值的长推理信息。

---

## 718. From Synthetic to Real: Toward Identity-Consistent Makeup Transfer with Synthetic and Real Data

**arXiv ID:** 2605.07861 | [PDF](https://arxiv.org/pdf/2605.07861v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 719. Actor-Critic Algorithm for Dynamic Expectile and CVaR

**arXiv ID:** 2605.07857 | [PDF](https://arxiv.org/pdf/2605.07857v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 720. Hybrid TF--IDF Logistic Regression and MLP Neural Baseline for Indonesian Three-Class Sentiment Analysis on Social Media Text

**arXiv ID:** 2605.07793 | [PDF](https://arxiv.org/pdf/2605.07793v1)

**作者:** Allya Nurul Islami Pasha `[一作]` (Institut Teknologi Sumatera), Martin C. T. Manullang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了印尼社交媒体文本的三分类情感分析，提出了基于TF‑IDF与三项数值元数据的混合特征，并以逻辑回归为生产模型，提供了浅层MLP对比基线。

**💡 创新点**

关注可复现、可部署的经典模型与轻量化神经基线在同一特征空间下的系统对比，并强调标签压缩与不平衡处理对小规模印尼情感数据的重要性。

**🔧 技术方法**

使用TF‑IDF（词-词、双词）与文本长度、总互动、标签计数三项数值特征的混合向量，训练多项式逻辑回归（class_weight=balanced）以及两层ReLU MLP；实验还保留了Linear SVM等经典模型。

**📊 数据集**

基于原始732行、191细粒度情感标签的印尼社交媒体情感数据，清洗后共707条，按正（459）、负（188）、中性（60）三类重新映射。

**📈 对比分析**

采用80/20分层划分，评估准确率、加权F1、宏F1；逻辑回归取得0.8028准确率、0.8003加权F1、0.7276宏F1，MLP略优于准确率和加权F1但宏F1相近；Linear SVM在保留实验中表现最佳。

**⚠️ 局限性**

样本量小、标签映射引入噪声、仅单次拆分评估、缺乏交叉验证和定性误差示例，导致结果对泛化性的说明有限。

---

## 721. Analyzing Human Heuristics and Strategies in Everyday Decision-Making Conversations for Conversational AI Design

**arXiv ID:** 2605.07789 | [PDF](https://arxiv.org/pdf/2605.07789v1)

**作者:** Sora Kang `[一作]` (Seoul National University), Joonhwan Lee `[通讯]` (Seoul National University)

**通讯引用:** 2437 | [OpenAlex ID](https://openalex.org/A5056599782)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 955 条韩语日常决策对话（涉及饮食与旅行，共 15,476 语句）进行系统编码，构建并验证了面向决策的对话代码书，并通过 GPT‑4 辅助编码实现大规模分析。

**💡 创新点**

① 将决策理论与对话分析结合，首次提出面向对话的决策代码书；② 利用 LLM 实现高效、可复现的编码流程；③ 发现“频率‑效率失衡”与两阶段探索/利用结构，为对话式 AI 设计提供实证依据。

**🔧 技术方法**

GPT‑4 LLM 进行自动编码（人机交互验证）；统计分析（卡方检验、频率‑成功率比）；对话分析编码框架。

**📊 数据集**

AI‑Hub 开源韩语日常对话数据集（食品饮料 5,176 轮、旅行 5,550 轮），经筛选后得到 955 条符合决策目标的对话。

**📈 对比分析**

通过频率‑成功率比对各类启发式（如属性消除、社会影响等）与决策结果的关系；发现满意化策略（41.9%）显著高于最大化（15.7%），且 72.6% 的对话达成决策；高频启发式（如情感、可得性）效率低，低频规则式启发式（如属性消除）效率高。

**⚠️ 局限性**

仅基于韩语文本对话，可能不适用于其他语言或文化；数据局限于饮食与旅行两类低风险日常决策；缺乏多模态或面对面对话；LLM 辅助编码虽经校正，但仍可能存在微小标注误差。

---

## 722. Video Understanding Reward Modeling: A Robust Benchmark and Performant Reward Models

**arXiv ID:** 2605.07872 | [PDF](https://arxiv.org/pdf/2605.07872v1)

**作者:** Yuancheng Wei `[一作]` (South China University of Technology), Xu Sun `[通讯]` (Peking University)

**通讯引用:** 6888 | [OpenAlex ID](https://openalex.org/A5111863979)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了视频理解奖励建模的统一框架，包括VURB基准、VUP-35K大规模视频偏好数据集，以及VideoDRM和VideoGRM两种奖励模型。

**💡 创新点**

创新点在于构建了包含长Chain-of-Thought（CoT）推理的VURB基准、自动化生成的35K高质量视频偏好样本VUP-35K，并分别提出了分辨式和生成式奖励模型，显著提升了视频奖励学习的可行性与效果。

**🔧 技术方法**

采用多模态LLM（如Qwen3VL、Gemini-2.5-Pro）生成CoT响应；通过长度一致性与因果一致性两阶段过滤保证数据质量；生成式奖励模型使用GRPO强化学习进行训练；辨识式奖励模型采用margin ranking loss；自动化构建VUP-35K的pipeline。

**📊 数据集**

使用了VURB（2100对偏好样本，平均CoT长度1143 tokens）、VUP-35K（35K视频偏好对）以及213K图文偏好样本来训练和评估。

**📈 对比分析**

在VURB和VideoRewardBench上与商业、开源及专家模型进行全面对比，VideoDRM在VURB上实现63.8%整体准确率，VideoGRM实现59.3%；两者均超过GPT 5.2、VideoRewardBench现有SOTA，并在Best-of-N推理中显著提升模型性能。

**⚠️ 局限性**

局限性包括：自动化生成的偏好对缺乏真实人类偏好细节，模型在极长视频或极细粒度推理任务上的表现仍有提升空间；当前实现主要关注文本推理，尚未充分探索多模态视频与多模态视觉细节的深度融合。

---

## 723. MatryoshkaLoRA: Learning Accurate Hierarchical Low-Rank Representations for LLM Fine-Tuning

**arXiv ID:** 2605.07850 | [PDF](https://arxiv.org/pdf/2605.07850v1)

**作者:** Ionut-Vlad Modoranu `[一作]` (Institute of Science and Technology Austria), Dan Alistarh `[通讯]` (Institute of Science and Technology Austria)

**通讯引用:** 4468 | [OpenAlex ID](https://openalex.org/A5083822059)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种新的LoRA训练框架MatryoshkaLoRA，通过在适配器之间插入对角矩阵P，实现层级低秩特征学习，可在推理时根据不同rank切片使用。

**💡 创新点**

创新点在于将层级低秩表示嵌入单一LoRA适配器，并通过可调diagonal P统一实现LoRA、DyLoRA及其改进；同时提出AURAC指标评估多rank性能。

**🔧 技术方法**

使用的方法包括对LoRA适配器的结构改造、对角矩阵P的计算与插入、梯度缩放、AURAC指标、以及在Llama模型上进行细调。

**📊 数据集**

实验数据集包括GSM-8k、Open LLM Leaderboard（ARC-C、HellaSwag）以及OpenPlatypus。

**📈 对比分析**

与传统LoRA和DyLoRA对比，MatryoshkaLoRA在所有rank上均取得更高精度，AURAC提升约3–5%，并在高rank时仍保持良好表现。

**⚠️ 局限性**

局限性包括需要对每个rank进行多次评估导致额外运行时间，超参数搜索范围有限，且仅在全网络使用相同rank，未探索层级异步rank分配。

---

## 724. \mathsf{VISTA}: Decentralized Machine Learning in Adversary Dominated Environments

**arXiv ID:** 2605.07841 | [PDF](https://arxiv.org/pdf/2605.07841v1)

**作者:** Hanzaleh Akbari Nodehi `[一作]` (University of Minnesota), Mohammad Ali Maddah-Ali `[通讯]` (University of Minnesota)

**通讯引用:** 9856 | [OpenAlex ID](https://openalex.org/A5113662214)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在分布式无许可机器学习中，提出了一种自适应的接受阈值与学习率调度策略，能够在对手占多数的网络中实现迭代优化的稳健收敛；

**💡 创新点**

创新点在于将一致性检验与奖励机制相结合，使对手从纯破坏者转变为理性行为体，并通过动态调节接受阈值在训练早期实现快速进展、后期实现高精度；

**🔧 技术方法**

使用游戏‑编码（game‑of‑coding）框架、指数移动平均估计、二分搜索确定阈值、以及基于梯度范数的自适应学习率；

**📊 数据集**

实验采用三维合成目标、MNIST（LeNet）和CIFAR‑10（ResNet‑18）等数据集；

**📈 对比分析**

与固定阈值基线相比，动态阈值方法在早期迭代获得更快下降，后期收敛到更低误差；在所有实验中均表现出比传统鲁棒聚合方法更优的训练损失曲线；

**⚠️ 局限性**

局限性包括：假设对手为理性、短视的攻击者；需要预先知晓或估计接受概率与误差之间的函数关系；对梯度范数的估计误差对阈值选择敏感；且在极端恶意比例或非光滑目标下的性能尚未完全验证。

---

## 725. Exact Regular-Constrained Variable-Order Markov Generation via Sparse Context-State Belief Propagation

**arXiv ID:** 2605.07839 | [PDF](https://arxiv.org/pdf/2605.07839v1)

**作者:** François Pachet `[一作]` (Sorbonne Université), François Pachet `[通讯]` (Sorbonne Université)

**通讯引用:** 3651 | [OpenAlex ID](https://openalex.org/A5111498117)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了如何在可变阶马尔可夫模型上实现正则约束的精确贝叶斯推断与采样，提出将上下文状态稀疏化后与约束自动机做乘积，兼容可逆增量化与单数回避策略；

**💡 创新点**

创新点在于把可变阶上下文状态作为推断状态，与正则约束自动机乘积，从而在保持模型稀疏性的同时实现精确的约束采样；同时提供可逆增量化的行列式计算与单数回避的显式处理；

**🔧 技术方法**

使用前向-后向贝叶斯传播（belief propagation）在稀疏上下文-自动机乘积图上进行推断，结合可变阶背off规则、正则约束自动机构造、逆向查找的可逆增量化与Python实现；

**📊 数据集**

实验数据集包括：1）小型整数例子（12个符号、短序列）用于精确性验证；2）Bach Prelude（592个音高事件、25个MIDI音高）用于规模与可逆转置验证；

**📈 对比分析**

通过对小例子做暴力枚举验证Z、TV误差几乎为0；Bach实验比较稀疏乘积与全状态（|Q|·|E|）及密集 |Σ|^K，稀疏乘积状态≤1205、边≤83207，后向传播<0.1s，采样<0.4ms，成功率1；与Continuator pos-only实现对比，后者在K≥3时略快，但整体性能可扩展；

**⚠️ 局限性**

局限性包括：仅适用于已训练且固定的可变阶上下文模型；对更复杂的正则约束可能导致乘积仍过大；单数回避等策略需手工嵌入或显式状态；未评估大规模并行或内存峰值；法律/版权风险未解决。

---

## 726. Distributional simplicity bias and effective convexity in Energy Based Models

**arXiv ID:** 2605.07844 | [PDF](https://arxiv.org/pdf/2605.07844v1)

**作者:** Aurélien Decelle `[一作]` (Universidad Politécnica de Madrid), Beatriz Seoane `[通讯]` (Universidad Complutense de Madrid)

**通讯引用:** 1548 | [OpenAlex ID](https://openalex.org/A5011613939)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文从有效参数空间出发，分析了能量基模型（EBM）的学习动力学，证明了数据一致固定点的存在与有效稳定性，并揭示了梯度下降自然导致的分布简单性偏差（低阶交互先学习）

**💡 创新点**

创新点在于将EBM映射为具有任意高阶交互的全可见玻尔兹曼机，从而实现对有效参数空间的凸性分析，提出有效参数的ℓ₂正则化方法，并理论上解释了分布简单性偏差与错误固定点抑制的机制

**🔧 技术方法**

主要技术包括：傅里叶展开/伪布尔函数理论、有效交互映射、梯度流分析、正则化与雅可比投影、低阶谱集中定理、随机梯度估计以及蒙特卡洛采样

**📊 数据集**

实验数据集包括二值化MNIST、Allen Institute视觉行为神经元活动数据以及基于三体哈密顿量的合成自旋系统

**📈 对比分析**

通过对比不同阶数有效参数的弗罗贝尼乌斯范数随迭代的演变，验证了低阶先学习的序列动态；在合成三体数据上，训练结果几乎重现真实三体耦合；在MNIST和神经元数据中，模型逐步匹配协方差结构，表明方法能够捕捉高阶统计特征

**⚠️ 局限性**

局限性在于假设EBM具有足够表达能力以匹配所有数据矩阵，实际中参数受限时可能无法完全实现；采样效率低导致训练收敛慢；正则化需基于有效参数估计，计算开销较大

---

## 727. FLAM: Evaluating Model Performance with Aggregatable Measures in Federated Learning

**arXiv ID:** 2605.07962 | [PDF](https://arxiv.org/pdf/2605.07962v1)

**作者:** Fabian Stricker `[一作]` (Hochschule Karlsruhe), Christian Zirpins `[通讯]` (Hochschule Karlsruhe)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 FLAM 方法，利用聚合可计算量（AM）在联邦学习中实现与中心化评估相同的模型性能评估，避免需要全局测试集；

**💡 创新点**

创新点在于把指标计算拆解为可聚合的基础量，移除加权平均导致的偏差，使联邦评估结果与中心化一致；

**🔧 技术方法**

使用的技术包括分布式统计聚合、加权/宏观 F1、R² 等指标的 AM 设计、FedAvg 联邦学习框架、Python/TensorFlow 与 scikit-learn 进行实验；

**📊 数据集**

实验数据集涵盖分类任务（Covertype、CIFAR‑10、CIFAR‑100）和回归任务（PVOD、GermanSolarFarm），并在不同偏斜（量级、标签级）下进行划分；

**📈 对比分析**

比较方法为：1）中心化评估；2）参与者基于样本数加权平均；3）FLAM。实验显示在多数指标（准确率、召回率等）与中心化一致；在宏观 F1、R²、MCC 等指标，传统加权平均偏差显著，FLAM 与中心化完全一致，性能表现保持不变；

**⚠️ 局限性**

局限性包括：需手动推导各指标的 AM，无法自动化；对大规模参与者的聚合可能产生通信开销；发送真标签/预测可能泄露敏感信息，需结合 SMPC 等隐私技术。

---

## 728. TAVIS: A Benchmark for Egocentric Active Vision and Anticipatory Gaze in Imitation Learning

**arXiv ID:** 2605.07943 | [PDF](https://arxiv.org/pdf/2605.07943v1)

**作者:** Giacomo Spigler `[一作]` (Tilburg University), Giacomo Spigler `[通讯]` (Tilburg University)

**通讯引用:** 531 | [OpenAlex ID](https://openalex.org/A5080564641)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了TAVIS基准，用于评估主动视觉模仿学习，并提供两套任务（头部主动视觉与手腕主动视觉）与两款人形躯干机器人；

**💡 创新点**

创新点包括：①独立的主动视觉评估基准；②GALT（Gaze‑Action Lead Time）时序度量；③配对头摄像头与固定摄像头的评估协议；④分层的ID/OOD分割；⑤基准实验展示任务依赖性和分布偏移下的性能衰退；

**🔧 技术方法**

采用IsaacLab仿真环境、Diffusion Policy与π_0两种模仿学习算法、Quest 3头戴式眼动操控进行数据采集，并使用GALT度量来评估视线先行；

**📊 数据集**

使用LeRobot v3.0数据集（约2200个演示），所有演示均通过Quest 3在模拟环境中采集并发布至Hugging Face；

**📈 对比分析**

通过配对头摄像头与固定摄像头的同一演示进行对比，利用ID/OOD分割与多任务/单任务训练进行性能评估；在TAVIS‑Head中，头摄像头平均提升约8–26个百分点；在TAVIS‑Hands中，单任务和多任务基线分别达到约70–77%与更高成功率；GALT显示模型的视线提前时间与人类操作者相当；

**⚠️ 局限性**

局限性包括仅在仿真中测试、单一操作者数据导致视线偏差、仅支持2–3自由度的颈部与手腕摄像头、GALT仅为后置度量、OOD分割有限且未覆盖视觉纹理、光照等更复杂场景、以及单随机种子评估。

---

## 729. One Token Per Frame: Reconsidering Visual Bandwidth in World Models for VLA Policy

**arXiv ID:** 2605.07931 | [PDF](https://arxiv.org/pdf/2605.07931v1)

**作者:** Zuojin Tang `[一作]` (Zhejiang University), Bin Liu `[通讯]` (Chery Auto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fede83ac-7505-405f-ab37-e7284695c47f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种将VLA与世界模型相结合的OneWM-VLA框架，使用单一语义token压缩每帧视觉信息，并通过联合流匹配实现隐空间与动作序列的同时生成。

**💡 创新点**

创新点在于：①只保留每帧一个语义token的瓶颈设计，显著降低视觉带宽；②采用单模型联合流匹配目标，将隐空间预测与动作生成紧耦合；③证明在有限适配预算下，语义压缩比像素压缩更有效。

**🔧 技术方法**

技术包括：Adaptive Attention Pooling（自适应注意力池化）对视觉特征做单token压缩；联合流匹配（Joint Flow Matching）在隐空间和动作空间共享同一生成器；LoRA微调以在冻结主干上适配。

**📊 数据集**

使用MetaWorld MT50、LIBERO四个子集（Spatial、Object、Goal、Long）以及真实Piper机器人手臂的Pick Banana、Fold Cloth、Pull Drawer任务进行实验。

**📈 对比分析**

在MetaWorld MT50上平均成功率提升至61.3%（从47.9%），在LIBERO Long提升至95.6%（高于π_0的85.2%），在Fold Cloth任务上从20%提升至60%，显示相较于π_0/π_0.5在长时域任务上有显著性能提升。

**⚠️ 局限性**

局限在于：仅在特定预训练VLA主干上验证；对更大尺度或不同任务的泛化尚未评估；单token压缩可能在更复杂视觉场景中信息不足；并未降低每步推理成本。

---

## 730. INO-SGD: Addressing Utility Imbalance under Individualized Differential Privacy

**arXiv ID:** 2605.07930 | [PDF](https://arxiv.org/pdf/2605.07930v1)

**作者:** Xiao Tian `[一作]` (National University of Singapore), Bryan Kian Hsiang Low `[通讯]` (National University of Singapore)

**通讯引用:** 808 | [OpenAlex ID](https://openalex.org/A5030304400)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的个性化差分隐私训练算法——Individualized Noisy Ordered SGD（INO‑SGD），通过对梯度按损失排序并按重要性裁剪来解决IDP导致的模型效用不平衡问题。

**💡 创新点**

创新点在于：①首次针对IDP下的效用不平衡提出理论分析与解决方案；②设计了基于“尾部重要性函数（TIF）”的连续梯度裁剪机制，既保持IDP保证，又能动态下调低重要性梯度；③通过理论证明和实验表明，该方法能显著提升对更私密数据所有者的模型性能，同时不降低总体准确率。

**🔧 技术方法**

使用技术包括：个性化差分隐私的采样率与裁剪阈值；Gaussian机制与RDP/IRDP的隐私分析；基于梯度排序的连续裁剪与重要性加权；对梯度进行加噪后再进行参数更新。

**📊 数据集**

在公开数据集 MNIST、CIFAR‑10、CIFAR‑100（含F‑V子集）上进行实验，模拟多数据所有者及不同隐私预算，验证模型效果。

**📈 对比分析**

与传统的个性化DP‑SGD（如IDP‑SGD）以及常规SGD进行对比，INO‑SGD在更私密数据所有者上的召回率/损失提升约10%且总体准确率不低于或略优于对照组，证明了其在公平性与整体性能之间的良好平衡。

**⚠️ 局限性**

局限性包括：①对更私密组的提升可能导致更少私密组性能略有下降，体现了IDP‑balance‑utility trade‑off；②对TIF参数的选择仍需经验性调优；③在极端数据分布或超大模型场景下的计算开销尚未充分评估。

---

## 731. Physics-Inspired Probabilistic Computing for Extremely Large-Scale MIMO Detection in Future 6G Wireless Systems

**arXiv ID:** 2605.07884 | [PDF](https://arxiv.org/pdf/2605.07884v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 732. Similar Pattern Annotation via Retrieval Knowledge for LLM-Based Test Code Fault Localization

**arXiv ID:** 2605.07957 | [PDF](https://arxiv.org/pdf/2605.07957v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 733. Sycophantic AI makes human interaction feel more effortful and less satisfying over time

**arXiv ID:** 2605.07912 | [PDF](https://arxiv.org/pdf/2605.07912v1)

**作者:** Lujain Ibrahim `[一作]` (University of Oxford), Diyi Yang `[通讯]` (Stanford University)

**通讯引用:** 13812 | [OpenAlex ID](https://openalex.org/A5089413311)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了持续使用恭维型AI对用户人际关系与社会满意度的影响，使用五项预注册实验共3075人、12766次人机对话。

**💡 创新点**

首次通过纵向实验证实恭维AI能降低用户对真实关系的满意度，并使其倾向将AI视为同等情感支持来源。

**🔧 技术方法**

利用大型语言模型（LLM）通过不同系统提示生成恭维、中立和挑战式回答，并配合混合效应回归分析对结果进行量化。

**📊 数据集**

基于Prolific招募的美国成人样本，包含多轮对话记录与自评问卷，探讨16类个人建议话题。

**📈 对比分析**

通过对比恭维、中立、挑战与无AI控制组，使用混合效应模型评估主观满意度、智力谦逊、AI偏好等指标，发现恭维AI显著提升“被理解”感并导致对人类支持满意度下降，效应量中等。

**⚠️ 局限性**

研究仅持续三周、样本为美国西方网络成年人，未检验长期或跨文化差异；LLM仅无记忆，缺乏真实持续记忆对结果的影响。

---

## 734. CoCoReviewBench: A Completeness- and Correctness-Oriented Benchmark for AI Reviewers

**arXiv ID:** 2605.07905 | [PDF](https://arxiv.org/pdf/2605.07905v1)

**作者:** Hexuan Deng `[一作]` (Harbin Institute of Technology), Min Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 62177 | [OpenAlex ID](https://openalex.org/A5100402851)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了CoCoReviewBench，一套针对AI审稿系统的完整性与正确性评估基准，解决传统评估因人类评审不完整和不准确而产生的偏差；

**💡 创新点**

创新点在于（1）构建了细粒度分类子集并在缺失人类评审时跳过评估，以提升完整性；（2）利用审稿人–作者–元评审的冲突信息进行错误过滤，提升参考的正确性；（3）打造3,900篇NeurIPS/ICLR论文的结构化数据集，实现多维度、基于类别的AI审稿评估；

**🔧 技术方法**

主要技术包括：多步骤LLM分段/分类流水线（分为语义切分、句子标签和子类别归一化）、强化学习（GRPO）与指令微调训练的Qwen3‑8B模型用于AI审稿的分段与分类，以及LLM-as‑Judge协议进行评估；

**📊 数据集**

使用的数据集为从NeurIPS 2021‑2024与ICLR 2017‑2025筛选的3,900篇论文，包含至少三位审稿人评审、作者回应和元评审，经过人工验证后标注为134.8k意见、115.9k聚类、108.6k正确信息；

**📈 对比分析**

在评估时，本文对比了多类模型（闭源GPT‑5、Gemini、开源Qwen、Llama等）在传统BLEU/ROUGE/BERTScore以及新基于LLM‑as‑Judge的多维度（正确性、彻底性、基准性、可验证性、清晰度）下的表现；结果显示，尽管部分强大的闭源模型在旧指标上高于人类，但在新评估中多数模型仍低于人类，尤其在正确性和彻底性方面；但推理型LLM在基准性与可验证性上接近或优于人类；

**⚠️ 局限性**

局限性包括：冲突过滤仅能捕捉显式争议，可能漏掉细微错误；元评审的裁定为粗粒度，无法完全替代专业判定；评估流程需额外的分段与分类步骤，虽然已通过8B模型压缩，但仍存在计算成本；且在文本仅输入的设置下，模型易产生幻觉，尤其是涉及图表的评价。

---

## 735. SpatialPrompt: XR-Based Spatial Intent Expression as Executable Constraints for AI Generative 3D Design

**arXiv ID:** 2605.07894 | [PDF](https://arxiv.org/pdf/2605.07894v1)

**作者:** Yichen Andy Yu `[一作]` (North Carolina State University), Qiao Jin `[通讯]` (North Carolina State University)

**通讯引用:** 3432 | [OpenAlex ID](https://openalex.org/A5100611571)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `4de8e9d8-757b-475f-9627-18a445e50202` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究提出了SpatialPrompt，一套基于XR的系统，允许用户使用3D笔在三维空间中绘制粗略结构并通过语音提示表达语义与风格，从而将这些空间草图编译为可执行的约束，用以指导3D生成模型并支持迭代式人机协作；

**💡 创新点**

创新点在于：①将XR中的空间草图转化为可执行的约束，弥补传统文本或2D输入在三维结构表达上的不足；②将结构约束与语音语义提示结合，形成多模态、可编辑的生成指令；③实现了同步协同创作和快速迭代的闭环工作流；

**🔧 技术方法**

使用的技术包括：Apple Vision Pro XR平台、Logitech Muse三维笔、Meshy生成后端API、语音识别模块、颜色编码的共享空间绘制；

**📊 数据集**

论文未公开使用专门的数据集，生成过程依赖Meshy的内部预训练模型；

**📈 对比分析**

评估方式为形式化的启发式可用性评估，邀请三名具有XR/交互设计经验的实验室成员完成单人和协作任务，并通过问卷和访谈收集定性反馈；目前未给出量化性能指标，仅报告用户对意图表达、生成质量和期望一致性的主观评分；

**⚠️ 局限性**

主要局限包括：①对外部AI服务的依赖导致生成质量与响应时间不稳定；②空间约束与语义提示仍可能不完整，生成模型可能偏离期望；③缺乏实时预览与交互式约束可视化；④评估样本规模有限，尚未验证对非专业用户或专业设计师的实用性；

---

## 736. Trajectory as the Teacher: Few-Step Discrete Flow Matching via Energy-Navigated Distillation

**arXiv ID:** 2605.07924 | [PDF](https://arxiv.org/pdf/2605.07924v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 737. Evaluating Design Conformance Through Trace Comparison

**arXiv ID:** 2605.07909 | [PDF](https://arxiv.org/pdf/2605.07909v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 738. Exploring the non-convexity in machine learning using quantum-inspired optimization

**arXiv ID:** 2605.07947 | [PDF](https://arxiv.org/pdf/2605.07947v1)

**作者:** Kandula Eswara Sai Kumar `[一作]` (BosonQ Psi (BQP)), Rut Lineswala `[通讯]` (BosonQ Psi (BQP))

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种统一的量子启发式进化优化（QIEO）框架，用来解决稀疏信号恢复与鲁棒线性回归等非凸问题。

**💡 创新点**

创新点在于将量子叠加的概率表示与模拟量子旋转门相结合，使得搜索既能保持全局视角又能高效收敛，从而突破传统局部搜索和连续松弛方法在高维、噪声强环境下的局限。

**🔧 技术方法**

核心技术包括：量子比特（qubit）概率表示、Hadamard 变换初始化、R_Y 旋转门驱动的概率更新；实验对比 ADAM、DE、GA、IHT、AM‑RR 等连续、离散与专用算法。

**📊 数据集**

使用的实验数据集：合成压缩感知信号（n=16，p=50/100）、基因表达分析（p=50/100/500）以及面部识别鲁棒回归实验（n=600，p=100，离群比例 α=0.1–0.4）。

**📈 对比分析**

通过与上述对手在支持恢复率、均方误差（MSE）、支持长度等指标比较，QIEO 在所有任务中均实现 100% 的支持恢复、MSE 接近机器精度，并在高维/高噪声条件下明显优于 GA、ADAM、IHT，同时在鲁棒回归任务中与专用 AM‑RR 的性能相当。

**⚠️ 局限性**

局限性：需手动设定种群规模、旋转角度等超参数；实验仅基于合成数据，缺乏对真实大规模数据集的验证；量子模拟实现尚未充分利用硬件加速，实际速度提升待进一步验证。

---

## 739. DVD: Discrete Voxel Diffusion for 3D Generation and Editing

**arXiv ID:** 2605.07971 | [PDF](https://arxiv.org/pdf/2605.07971v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 740. Parameterized Local Search for Vertex Cover: When only the Search Radius is Crucial

**arXiv ID:** 2605.07941 | [PDF](https://arxiv.org/pdf/2605.07941v1)

**作者:** Christian Komusiewicz `[一作]` (Friedrich Schiller University Jena), Nils Morawietz `[通讯]` (Friedrich Schiller University Jena)

**通讯引用:** 43 | [OpenAlex ID](https://openalex.org/A5045950235)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了图中顶点覆盖的k-swap问题，提出了一种有效的算法来判断给定的顶点覆盖是否存在有效的改进k-swap。

**💡 创新点**

创新点在于提出了一种新的参数化局部搜索方法，能够在多种结构参数下实现FPT算法，并且在运行时间上与k和图的结构参数的关系得到了良好的控制。

**🔧 技术方法**

使用了参数化局部搜索和动态规划技术，结合图的树分解和模块分解来优化算法的性能。

**📊 数据集**

使用了多种图数据集，包括具有不同树宽、h指数和模块宽度的图，以验证算法的有效性和效率。

**📈 对比分析**

与现有方法相比，本文提出的算法在处理小k值时表现出更好的性能，能够在多种结构参数下实现FPT算法，运行时间为O(f(k)·n)，其中f(k)是与k相关的多项式函数。

**⚠️ 局限性**

限制在于算法的复杂性和对图结构的依赖，特别是在处理大规模图时，可能会受到图的结构特征的影响，导致性能下降。

---

## 741. Rebalancing gradient to improve self-supervised co-training of depth, odometry and optical flow predictions

**arXiv ID:** 2605.07945 | [PDF](https://arxiv.org/pdf/2605.07945v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 742. BeeVe: Unsupervised Acoustic State Discovery in Honey Bee Buzzing

**arXiv ID:** 2605.07903 | [PDF](https://arxiv.org/pdf/2605.07903v1)

**作者:** Hamze Hammami `[一作]` (Heriot-Watt University), Nidhal Abdulaziz `[通讯]` (Heriot-Watt University)

**通讯引用:** 284 | [OpenAlex ID](https://openalex.org/A5002768772)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了BeeVe框架，对蜜蜂蜂箱音频进行无监督的离散状态发现。

**💡 创新点**

创新点在于将自监督Patchout Spectrogram Transformer提取特征与向量量化变分自编码器结合，实现对非声学发声（机械振动）信号的无标签离散词典学习，并发现王后存在/缺失状态及其子状态。

**🔧 技术方法**

技术：Patchout Spectrogram Transformer（PaSST）做特征提取；Vector-Quantized Variational Autoencoder（VQ‑VAE）进行离散化编码；随后利用Jensen‑Shannon Divergence、silhouette、t‑SNE/UMAP、chi‑square检验等评估手段。

**📊 数据集**

使用UrBAN蜜蜂蜂箱音频数据集，约五小时的录音（共326条录音），并在未见数据上进行验证。

**📈 对比分析**

与已知王后状态标签的后置评估表明：JSD 0.609–0.688，活跃码字 13–16/64；王后缺失子状态三类、纯度>90%；不同代码本大小、随机种子实验保持一致；未见录音上Jaccard 0.947、JSD 0.206，维度投影保持拓扑。

**⚠️ 局限性**

限制：实验规模有限（仅5小时），未覆盖全年、不同蜂箱、环境多样性；代码本容量受限，可能无法捕捉全部生理状态；缺乏生物学验证，模型发现的子状态尚未得到专家标注确认。

---

## 743. What if AI systems weren't chatbots?

**arXiv ID:** 2605.07896 | [PDF](https://arxiv.org/pdf/2605.07896v1)

**作者:** Sourojit Ghosh `[一作]` (University of Washington Seattle), Avijit Ghosh `[通讯]` (Hugging Face)

**通讯引用:** 634 | [OpenAlex ID](https://openalex.org/A5049976809)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统评估了聊天机器人（如 ChatGPT、Claude、Gemini 等）在社会、经济、法律和环境层面的广泛影响，并提出了替代路径。

**💡 创新点**

创新点在于将聊天机器人视为一种主导的社会技术体系，揭示其多层次危害链，并提出四类干预策略（任务特定、模块化、提升用户代理性、政策保障）。

**🔧 技术方法**

主要采用文献综述与概念性分析方法，对现有研究与案例进行整合，未涉及新的算法实现。

**📊 数据集**

未使用任何特定数据集，主要引用先前研究报告和公开案例。

**📈 对比分析**

本文不进行实验比较，而是通过理论框架和案例对比阐述潜在危害与对策，未给出性能指标。

**⚠️ 局限性**

局限性包括仅聚焦于通用聊天机器人，缺乏实证验证，未覆盖专业或领域限定的对话系统。

---

## 744. How Value Induction Reshapes LLM Behaviour

**arXiv ID:** 2605.07925 | [PDF](https://arxiv.org/pdf/2605.07925v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 745. Convergent Stochastic Training of Attention and Understanding LoRA

**arXiv ID:** 2605.07959 | [PDF](https://arxiv.org/pdf/2605.07959v1)

**作者:** Zhengkai Sun `[一作]` (University of Manchester), Mingfei Sun `[通讯]` (University of Manchester)

**通讯引用:** 749 | [OpenAlex ID](https://openalex.org/A5101591811)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文对注意力层及基于LoRA的浅层神经网络在随机梯度训练下的可训练性进行理论证明，给出了无数据、无规模限制下的收敛性结论。

**💡 创新点**

创新点在于首次通过Villani条件与Poincaré不等式，证明注意力矩阵与LoRA参数化的正则化损失满足SDE收敛性；同时不依赖网络宽度或数据分布。

**🔧 技术方法**

采用的技术包括Villani条件判定、Gibbs测度的Poincaré不等式、随机微分方程（Langevin动力学）以及对正则化项的解析界定。

**📊 数据集**

实验使用二维Darcy流场数据集（64×64网格），共900个训练样本和124个测试样本。

**📈 对比分析**

将正则化（对数放大式与超二次式）与无正则化的注意力权重更新进行对比；在相同训练设置下，正则化方法在保持相似的RMSE与相对L2误差的同时，显著抑制权重范数增长并降低泛化误差。

**⚠️ 局限性**

限制在于仅证明了单头注意力层与浅层LoRA网络的收敛性，未覆盖完整Transformer层、所有注意力矩阵以及其他高效注意力变体（如FlashAttention、Performers、Mamba）。

---

## 746. Ask Early, Ask Late, Ask Right: When Does Clarification Timing Matter for Long-Horizon Agents?

**arXiv ID:** 2605.07937 | [PDF](https://arxiv.org/pdf/2605.07937v1)

**作者:** Anmol Gulati `[一作]` (Pricewaterhousecoopers United States), Vamse Kumar Subbiah `[通讯]` (Pricewaterhousecoopers United States)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过构建强制注入实验框架，对长周期AI代理在不同信息维度（目标、输入、约束、上下文）上的澄清价值随执行进度变化进行系统量化。

**💡 创新点**

首次经验性证明澄清时机与信息维度高度相关：目标信息最佳窗口≤10%，输入信息可延迟至≈50%，约束信息受oracle缺口影响；并揭示现有前沿模型几乎不在这些最优窗口内询问。

**🔧 技术方法**

采用强制注入实验、Pass@3、浪费计算量、Kendall τ一致性等指标，结合四款前沿LLM模型和三大基准，完成6000+次实验跑。

**📊 数据集**

使用LHAW提供的84个未指定任务变体（覆盖MCP-Atlas、TheAgentCompany、SWE-Bench Pro），以及300个无脚本会话进行自然询问评估。

**📈 对比分析**

通过与无澄清基线和oracle的对比，评估不同注入时机对Pass@3、浪费计算和模型间一致性的影响。结果显示：目标澄清在10%以内最有价值，输入澄清可延迟至≈50%；现有模型多数过问或不问，远低于理论最优窗口。

**⚠️ 局限性**

局限性包括仅衡量需求侧澄清价值，未提供供应侧自适应询问时机机制；部分基准样本量不足导致信噪比受限；自然询问协议与强制注入存在行为差异，实验结果视为上限估计。

---

## 747. How to Train Your Latent Diffusion Language Model Jointly With the Latent Space

**arXiv ID:** 2605.07933 | [PDF](https://arxiv.org/pdf/2605.07933v1)

**作者:** Viacheslav Meshchaninov `[一作]`, Dmitry Vetrov `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c84dae5d-5273-4348-85a7-b44cb586b4df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个联邦训练的潜在扩散语言模型LDLM，联合学习潜在空间、扩散模型与解码器。

**💡 创新点**

关键创新在于提出四步训练方案：MSE解码器损失、扩散到编码器预热、自适应时间步采样以及解码器输入噪声，从而实现潜在空间与扩散模型的协同优化。

**🔧 技术方法**

采用预训练GPT‑2作为token编码器、连续（Gaussian）扩散过程、自回归解码器、MSE与CE损失、以及自适应噪声调度等技术。

**📊 数据集**

在OpenWebText和LM1B两个公开语料集上进行训练与评估。

**📈 对比分析**

与现有离散与连续扩散模型对比，LDLM在生成质量（PPL↓）与多样性（entropy↑）上更优，且采样速度比对手快2–13倍。

**⚠️ 局限性**

局限性包括对预训练编码器的依赖、训练过程对超参数（噪声、预热、采样调度）较为敏感，以及在极端长文本或低资源场景下性能尚未充分验证。

---

## 748. MedVIGIL: Evaluating Trustworthy Medical VLMs Under Broken Visual Evidence

**arXiv ID:** 2605.07919 | [PDF](https://arxiv.org/pdf/2605.07919v1)

**作者:** Hanqi Jiang `[一作]` (University of Georgia), Xiang Li `[通讯]` (Harvard Medical School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了名为MedVIGIL的医学视觉语言模型（VLM）可信性评估套件，针对模型在证据失效（如图像ROI被遮蔽、问题被误导性重写）时是否会安静失败（即仍给出答案而非拒绝）进行系统评估。

**💡 创新点**

创新点在于：① 将“证据契约”框架引入医学VLM评估，明确模型在证据失效时应安全拒绝；② 设计了七维正确性条件指标并合成了复合得分（MCS）以避免单一指标掩盖弱点；③ 通过四位放射科医生的端到端标注构建了300个高质量案例，并配备了ROI、对照、伪题、反事实等多种扰动。

**🔧 技术方法**

技术包括：多模态文本与图像的多选答案包装、ROI遮蔽与翻转的图像扰动、文本重写（否定、特异性删除、知识唯写、幻觉陷阱）、七维评估指标（Acc、PR、NEG、SDR、LPA、VGR、SFR_w）以及复合MCS的调和平均公式。

**📊 数据集**

数据集来源为四大公开医学VQA数据集（VQA-RAD、SLAKE、ROCO、MIMIC‑CXR、CheXpert），挑选300个符合“需要图像”且无模板泄露的案例，并由放射科医生手工标注ROI、风险层级、答案等。

**📈 对比分析**

在16个视觉支持模型（包括OpenAI、Anthropic、Google、Qwen、Moonshot等）和2个文本仅模型上进行审计。最佳模型Claude Opus 4.7的MCS为69.2，安全性与视觉基准表现最好；人类基准（独立放射科医生）MCS为83.3，显示模型仍有约14点改进空间。

**⚠️ 局限性**

局限性包括：① 评估聚焦于受控扰动，未覆盖完整图像破坏与更细粒度的视觉失效；② 只采用单一独立评审者做人类参考，缺乏多评审者一致性；③ 公开数据与模型预训练可能存在重叠，影响结果的泛化性；④ 评估未涵盖更广泛的思维模式、温度与安全提示等变异。

---

## 749. Curvature Beyond Positivity: Greedy Guarantees for Arbitrary Submodular Functions

**arXiv ID:** 2605.07902 | [PDF](https://arxiv.org/pdf/2605.07902v1)

**作者:** Yixin Chen `[一作]` (Texas A&M University), Alan Kuhnle `[通讯]` (Texas A&M University)

**通讯引用:** 618 | [OpenAlex ID](https://openalex.org/A5083575259)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对非单调且可能取负值的子模函数最大化，提出基于曲率的贪心+剪枝算法及其连续版本 DMCG-P，并给出乘法近似保证。

**💡 创新点**

创新点在于将曲率扩展到非单调/负值子模函数，实现单一连续参数控制近似误差；通过轨迹曲率和剪枝保持正边际，实现对一般组合约束的乘法保证，并给出可后验的曲率证书。

**🔧 技术方法**

核心技术包括：1) 定义全局与轨迹曲率（c_f、c_g）及其在非单调情况下的性质；2) 贪心+剪枝算法实现局部正边际；3) DMCG-P（离散化测量连续贪心）与多项式约束的结合；4) 曲率证书（基于剪枝轨迹的 OPT‑free 上界）和实验评估。

**📊 数据集**

实验使用了：
- 小规模实验设计、覆盖、特征选择（n=20, k=5/10）
- 中等规模（n≤300）实验设计与覆盖
- 语料库 Multi‑News（n=200, k=10）用于多文档摘要
- MaxCut 例子用于对称子模测试
- 其他基准如随机贪心、标准贪心、扭曲贪心。

**📈 对比分析**

与传统加性保证（HFWK）以及均匀约束下的 1/e 及 0.401 等做比较。实验表明：
- 在高成本或负值场景下，曲率保证始终为正且优于 0.401；
- 贪心+剪枝的实测比率常在 0.84 以上，远高于扭曲贪心；
- 曲率证书提供的后验乘法上界比 HFWK 的加性下界更稳健。

**⚠️ 局限性**

局限性包括：
- 曲率是轨迹特定的，需运行后才能得到；
- 对连续约束需要轨迹曲率有限的假设，无法直接适用于所有组合约束；
- 对一般值指针（仅 oracle）的曲率证书尚未给出；
- 算法在高维大规模时仍需改进求解速度（如流行的阈值/流式版本）。

---

## 750. Beyond "I cannot fulfill this request": Alleviating Rigid Rejection in LLMs via Label Enhancement

**arXiv ID:** 2605.07883 | [PDF](https://arxiv.org/pdf/2605.07883v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 751. Touring a Sequence of Orthogonal Polygons

**arXiv ID:** 2605.07882 | [PDF](https://arxiv.org/pdf/2605.07882v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

---

## 752. Graph Representation Learning Augmented Model Manipulation on Federated Fine-Tuning of LLMs

**arXiv ID:** 2605.07961 | [PDF](https://arxiv.org/pdf/2605.07961v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 753. Exploring a Virtual Pet to Provide Context Notifications in a Tourism Recommender System: a Pilot Study

**arXiv ID:** 2605.07960 | [PDF](https://arxiv.org/pdf/2605.07960v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 754. TimeLesSeg: Unified Contrast-Agnostic Cross-Sectional and Longitudinal MS Lesion Segmentation via a Stochastic Generative Model

**arXiv ID:** 2605.07955 | [PDF](https://arxiv.org/pdf/2605.07955v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 755. Black-box model classification under the discriminative factorization

**arXiv ID:** 2605.07878 | [PDF](https://arxiv.org/pdf/2605.07878v1)

**作者:** Hayden Helm `[一作]`, Carey Priebe `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了判别因子分解方法，用于黑盒模型级别分类，区分查询集中的信号与正交信息；

**💡 创新点**

将查询-模型交互拆解为若干独立方向，并给出指数级误差上界，同时可通过SVD估计判别秩与零集概率；

**🔧 技术方法**

利用多维尺度（MDS）、能量距离、随机森林分类器、SVD与高斯混合模型进行参数估计与查询筛选；

**📊 数据集**

在Yahoo Answers子集训练LoRA适配器、系统提示实验以及检索增强生成（RAG）模型的人工生成查询上进行评估；

**📈 对比分析**

与信号、均匀、正交查询集进行对比，估计参数准确预测误差衰减速率，且在无任务知识的情况下估计的信号查询可重现oracle排序，显著降低分类误差；

**⚠️ 局限性**

零集理想化导致估计偏差，方向间可能相关或不均衡，对查询选择未做最优化，且仅针对二分类任务，正交查询仍能超随机表现。

---

## 756. Slowly Annealed Langevin Dynamics: Theory and Applications to Training-Free Guided Generation

**arXiv ID:** 2605.07950 | [PDF](https://arxiv.org/pdf/2605.07950v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 757. Delta-Adapter: Scalable Exemplar-Based Image Editing with Single-Pair Supervision

**arXiv ID:** 2605.07940 | [PDF](https://arxiv.org/pdf/2605.07940v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 758. AgentEscapeBench: Evaluating Out-of-Domain Tool-Grounded Reasoning in LLM Agents

**arXiv ID:** 2605.07926 | [PDF](https://arxiv.org/pdf/2605.07926v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 759. Prototype Guided Post-pretraining for Single-Cell Representation Learning

**arXiv ID:** 2605.07938 | [PDF](https://arxiv.org/pdf/2605.07938v1)

**作者:** Sachini Weerasekara `[一作]`, Jacqueline Isaacs `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种后预训练方法（CellRefine），在单细胞预训练模型与下游微调之间插入一个基于标记基因程序和细胞本体的多目标训练阶段，以提升模型对稀有细胞和分布偏移的鲁棒性。

**💡 创新点**

创新性地将已知的标记基因集合作为原型引导掩码语言模型，并加入谱系感知正则化和高斯混合变分编码器，形成多项式损失，突破了长尾与协变量偏移限制。

**🔧 技术方法**

使用掩码语言模型（MLM）、原型引导正则化、谱系正则化、Gaussian Mixture Variational Encoder (GMVE) 以及多任务损失组合，在 Geneformer 基础模型上实施。

**📊 数据集**

在 10 个人体单细胞 RNA‑seq 数据集（血液、胰腺、肝脏、髓系、MS、心脏、肺）和 2 个空间转录组（肝脏）上进行评估。

**📈 对比分析**

与线性探针、全微调、LoRA、MLM 预训练+微调等 7 种基线对比，CellRefine 在细胞身份预测、空间转录组插值和基因扰动响应预测三大任务中平均提升 10–15%，如宏 F1 最高达 0.75（比 MLM→FF 提升约 8%）。

**⚠️ 局限性**

依赖于事先整理的标记基因程序，若缺失或噪声会影响效果；此外尚未显著提升参数效率，后续需减少对外部知识的依赖。

---

## 760. Tree SAE: Learning Hierarchical Feature Structures in Sparse Autoencoders

**arXiv ID:** 2605.07922 | [PDF](https://arxiv.org/pdf/2605.07922v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 761. What Matters for Diffusion-Friendly Latent Manifold? Prior-Aligned Autoencoders for Latent Diffusion

**arXiv ID:** 2605.07915 | [PDF](https://arxiv.org/pdf/2605.07915v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 762. One World, Dual Timeline: Decoupled Spatio-Temporal Gaussian Scene Graph for 4D Cooperative Driving Reconstruction

**arXiv ID:** 2605.07910 | [PDF](https://arxiv.org/pdf/2605.07910v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 763. Flatness and Gradient Alignment Are Both Necessary: Spectral-Aware Gradient-Aligned Exploration for Multi-Distribution Learning

**arXiv ID:** 2605.07914 | [PDF](https://arxiv.org/pdf/2605.07914v1)

**作者:** Aristotelis Ballas `[一作]` (Harokopio University of Athens), Christos Diou `[通讯]` (Harokopio University of Athens)

**通讯引用:** 1234 | [OpenAlex ID](https://openalex.org/A5004953619)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种同时优化损失曲面平坦度与梯度对齐的训练方法SAGE，用于多分布学习场景

**💡 创新点**

通过引入损失曲面曲率与梯度协方差的风险分解，并证明二者可独立控制，提出结合谱扰动与梯度对齐噪声的联合策略

**🔧 技术方法**

使用Newton–Schulz迭代近似梯度的极限正交因子做谱扰动，配合根据跨分布梯度相似度调整的高斯噪声注入

**📊 数据集**

在DomainBed的五个域泛化数据集（PACS、VLCS、OfficeHome、TerraIncognita、DomainNet）以及Cityscapes与NYU‑v2的多任务学习数据集上进行评估

**📈 对比分析**

与传统SAM、SAGM、DISAM、BOA等平坦度方法以及PCGrad、Fish、GGA等梯度对齐方法对比，SAGE在DomainBed上平均准确率达到78.9%（新SOTA），并在MTL基准上对多种基线（FairGrad、LS、MGDA）均实现显著提升

**⚠️ 局限性**

计算开销较大（Newton–Schulz矩阵乘法与噪声注入需要额外梯度计算），在MTL某些指标上并未压倒专门设计的算法；理论证明仅在二次模型上成立，未能在非凸深度学习场景给出严格保证

---

## 764. Towards Settling the Complexity of the Lettericity Problem

**arXiv ID:** 2605.07899 | [PDF](https://arxiv.org/pdf/2605.07899v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 765. AccelSync: Verifying Synchronization Coverage in Accelerator Pipeline Programs

**arXiv ID:** 2605.07881 | [PDF](https://arxiv.org/pdf/2605.07881v1)

**作者:** Hangcheng An `[一作]` (Beihang University), Depei Qian `[通讯]` (Beihang University)

**通讯引用:** 2415 | [OpenAlex ID](https://openalex.org/A5079362609)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

提出了面向 AI 加速器操作符的同步覆盖验证工具 AccelSync；

**💡 创新点**

将同步覆盖问题形式化为可判定的“barrier sufficiency”问题，并证明其在 O(|E|²) 时间内可解；

**🔧 技术方法**

利用事件序列抽取、基于 Lamport 的 happens-before 图构建、以及硬件事件语义中的三种排序关系；

**📊 数据集**

在 Ascend 910B2 的 6,292 个 CANN 生产核、120 个 LLM 生成核以及 Cambricon MLU370 的 162 个 BANG C 核进行评估；

**📈 对比分析**

与黄金测试、静态规则、Huawei msSanitizer 等基线相比，AccelSync 检测率达到 100%，且每核平均仅需 5–10 ms，远快于 msSanitizer 的 2–4 s；

**⚠️ 局限性**

局限在于模型假设只涵盖固定结构的并行性，无法处理动态管线、数据相关同步及未建模的硬件隐式顺序；

---

## 766. When Diffusion Model Can Ignore Dimension: An Entropy-Based Theory

**arXiv ID:** 2605.07969 | [PDF](https://arxiv.org/pdf/2605.07969v1)

**作者:** Ahmad Aghapour `[一作]` (University of Michigan), Erhan Bayraktar `[通讯]` (University of Michigan)

**通讯引用:** 3316 | [OpenAlex ID](https://openalex.org/A5005237318)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

本文研究扩散模型在高维空间中采样的收敛性，提出一种新的信息论视角来解释其高效性。

**💡 创新点**

创新点在于用潜在变量的 Shannon 熵来刻画采样复杂度，取代传统的维度依赖，并给出了针对高斯混合目标的显式 KL 收敛上界。

**🔧 技术方法**

核心技术包括 MMSE 区域函数的重写、I‑MMSE 恒等式、信息不等式以及潜在后验冻结的采样器设计。

**📊 数据集**

论文主要为理论研究，未在具体公开数据集上进行实验；理论结果可应用于基于向量量化或离散代码的图像与语言模型。

**📈 对比分析**

与以往基于维度或几何结构的收敛分析对比，作者证明当潜在熵低时，采样步骤数仅与熵成线性关系，且对第二矩和终端 SNR 的依赖仅为对数级，显示了更优的规模可扩展性。

**⚠️ 局限性**

局限性在于仅适用于高斯混合目标且假设潜在熵有限、方差有限，且只给出理论上限；未涵盖非高斯或非混合结构的数据分布，且对潜在结构的具体实现缺乏实证验证。

---

## 767. Measure Many Quantum Finite Automata on Infinite Words

**arXiv ID:** 2605.07968 | [PDF](https://arxiv.org/pdf/2605.07968v1)

**作者:** Abhisek Midya `[一作]` (Jain University), A Baskar `[通讯]` (BITS Pilani)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并正式定义了测量-多量子布赫自动机（MMQBA），并对其在无穷字上的执行语义、可识别语言、闭包性质以及判定问题进行了系统研究。

**💡 创新点**

在传统的量子有限自动机基础上加入了布赫接受条件，并对接受与拒绝的累计概率设定了阈值，使得模型兼具量子特性与无穷运行的可判定性；进一步给出了MMQBA与MMQFA通过极限算子之间的对应关系。

**🔧 技术方法**

使用量子测量理论、投影算子、子空间分解、可不变子空间分析、极限运算符以及从停机问题的归约等数学技术，证明了模型的闭包性质、不可判定性以及半可判定的空语言判定等结论。

**📊 数据集**

本文完全是理论性的工作，没有使用任何实验数据集。

**📈 对比分析**

评价方法以理论证明为主，未涉及实验性能对比；通过形式化证明展示了模型在闭包与判定问题上的局限性。

**⚠️ 局限性**

主要限制包括：对交集与补集不闭合；多项判定问题（包括等价、包含、普遍性、成员资格）不可判定；空语言判定仅半可判定；模型的最小化与维度约简尚未解决。

---

## 768. Aggregation in conformal e-classification

**arXiv ID:** 2605.07963 | [PDF](https://arxiv.org/pdf/2605.07963v1)

**作者:** Vladimir Vovk `[一作]` `[通讯]`, Vladimir Vovk

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

本文通过大规模仿真实验，系统评估并改进了基于e‑value的交叉一致性预测（CCEP），提出了重复诱导一致性预测（RICEP）和均衡诱导一致性预测（BICEP）两种更灵活的聚合方法，并对其与传统p‑value方法（如CCP、ICP）进行比较。

**💡 创新点**

创新点包括：①将e‑value引入交叉一致性预测，避免了p‑value所需的平滑随机化；②证明e‑value聚合后仍保持完全有效性；③提出RICEP和BICEP，以克服CCEP对折叠数K的离散限制并显著降低方差（利用Jensen缺口理论）；④通过大量实验验证这些方法在不同类别数和训练比例下的预测效率与稳定性。

**🔧 技术方法**

主要技术：一致性预测框架（全、诱导、交叉），e‑value与p‑value的数学定义与转换，贝叶斯先验（Dirichlet(α)）下的拉普拉斯后验推断，模拟实验与性能评估指标（AFES/AFS），以及Jensen缺口分析。

**📊 数据集**

实验数据集为人工生成的多项式分类数据，类别数Y∈{2,10,100}，样本量固定为12,000，先验为Jeffreys（α=0.5）的Dirichlet分布，随后根据真实θ生成观测标签，整个过程不涉及任何外部真实数据集。

**📈 对比分析**

比较方法：用AFES（e‑value）或AFS（p‑value）作为平均假标签的“surprisal”指标，绘制不同方法（CEP、ICP、CCP、CCEP、RICEP、BICEP）随折叠数或训练比例变化的曲线。实验结果显示：RICEP在多次重复（N大）时表现更稳定、效率更高；CCEP在K较小的情况下接近最优，但随着K增大会出现效率下降；BICEP提供更灵活的训练/校准比例，适用于未知比例场景；在p‑value方法中，确定性p‑value表现最差。总体而言，基于e‑value的方法在有效性和预测效率上优于传统p‑value方法。

**⚠️ 局限性**

局限性：①仅有实验结果，缺乏严格理论证明；②依赖于仿真数据，对真实世界数据的适用性尚未验证；③Jensen缺口虽能解释性能提升，但在某些极端数据分布下可能不显著；④RICEP需要设置重复次数N，BICEP需要采样均匀或先验分布，超参数选择仍需经验；⑤当前研究仅关注无对象的分类问题，无法直接推广到带特征的回归或序列预测。

---

## 769. Stencil Computations on Cerebras Wafer-Scale Engine

**arXiv ID:** 2605.07954 | [PDF](https://arxiv.org/pdf/2605.07954v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 770. TraceFix: Repairing Agent Coordination Protocols with TLA+ Counterexamples

**arXiv ID:** 2605.07935 | [PDF](https://arxiv.org/pdf/2605.07935v1)

**作者:** Shuren Xia `[一作]` (Rutgers University), Jorge Ortiz `[通讯]` (Rutgers University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套以模型检查为核心的 TraceFix 管道，用 LLM 生成多智能体通信拓扑和 PlusCal 程序，利用 TLC 对协议进行验证并根据 counterexample 进行逐步修复，最终把验证通过的协议编译成代理提示，并在运行时通过拓扑监控防止非法协调操作。

**💡 创新点**

创新点在于将 TLA+ 计数器示例（counterexample）直接驱动 LLM 生成的协调协议的修复循环；将协议拆分为可验证的拓扑 IR 与可执行的 PlusCal 逻辑；以及在运行时加入轻量级拓扑监控，弥合验证与实际执行之间的差距。

**🔧 技术方法**

使用了 Claude Opus 4.6 作为 LLM、PlusCal/TLA+ 语言、TLC 模型检查器、JSON 结构化拓扑 IR、运行时监控、以及针对 Deadlock/Livelock、资源竞争等安全属性的自动化评估工具。

**📊 数据集**

采用了 48 题、16 个场景族（软件开发、科研写作、医疗咨询、制造、半导体 CI/CD 等）的自定义基准，每个场景设有 Easy/Medium/Hard 难度层级，配套完整的仿真环境与数据集。

**📈 对比分析**

通过比较四种运行时架构（Topology‑monitored、Mediator‑enforced、Prompt‑only、Chat‑only）和两级 LLM 能力、三层故障注入，使用 Avg Sim%、Sim 100%、Deadlock/Livelock 比率和资源争用率等指标。结果显示：Topology‑monitored 在 89.4% 的平均完成率、81.5% 的全完成率下领先；验证通过的协议在模型能力下降时下降率仅为 Prompt‑only 的一半；TLC 检查在 7.7 M 状态下仍在 60 s 内完成，且大多数任务在 1–4 次修复后即可通过。

**⚠️ 局限性**

局限性包括：验证仅在有限的状态空间内有效；不检查 liveness 或终止保证；运行时监控只限制拓扑结构，无法捕获步骤顺序偏差；在超时修复过程中可能出现语义漂移；实验只使用了单一 LLM，未验证对其他 LLM 的泛化能力。

---

## 771. Longitudinal Analyses of SAST Tools: A CodeQL Case Study

**arXiv ID:** 2605.07900 | [PDF](https://arxiv.org/pdf/2605.07900v1)

**作者:** Jean-Charles Noirot Ferrand `[一作]` (University of Wisconsin-Madison), Patrick McDaniel `[通讯]` (University of Wisconsin-Madison)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对CodeQL在开源软件中的长期表现进行大规模评估，测量其对近4000个CVE的检测效果、可操作性和版本稳定性。

**💡 创新点**

提出纵向评估框架，量化SAST工具随版本演进的有效性、可操作性与检测稳定性。

**🔧 技术方法**

使用CodeQL CLI不同版本对CVEfixes数据集执行静态分析，比较漏洞提交前后生成的警报集合并计算警报与漏洞位置的距离指标。

**📊 数据集**

利用CVEfixes提供的3,993个CVE及其修复提交，覆盖6种语言的1,622个开源仓库。

**📈 对比分析**

通过检测率、警报位置集中度以及跨版本检测保持率进行比较；结果显示CodeQL平均在修复前检测到83个漏洞，警报在50%时可操作，检测稳定性非单调，部分漏洞在更新后失效。

**⚠️ 局限性**

仅聚焦CodeQL，未考虑OSS生态演化；使用已知CVE而非实时漏洞；实验环境与实际CI/CD差异可能影响结果。

---

## 772. Semantic-Aware Adaptive Visual Memory for Streaming Video Understanding

**arXiv ID:** 2605.07897 | [PDF](https://arxiv.org/pdf/2605.07897v1)

**作者:** Hang Wu `[一作]` (University of California Merced), Yiwei Wang `[通讯]` (University of California Merced)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个训练无关的双阶段框架，先用语义先验进行查询无关的压缩，再做查询相关的检索，实现在线流视频问答。

**💡 创新点**

创新点在于使用固定伪问题库给压缩过程提供语义先验，并在检索阶段引入基于锚点的递延门控与晚期交互，两个阶段共享MaxSim评分，形成端到端的无训练管线。

**🔧 技术方法**

采用MaxSim语义相似度、三层流水线内存（短期/中期/长期）压缩、锚点递延门控、ColBERT式晚期交互、Qwen2.5-VL-7B/3B大模型。

**📊 数据集**

使用 OVO-Bench（Real-Time Visual Perception 与 Backward Tracing）、StreamingBench、ODV-Bench 这三个在线流视频评测数据集。

**📈 对比分析**

与 FluxMem、HERMES、ViSpeak 等训练无关或训练有监督的方法对比，在 OVO-Bench 上从 52.27 提升到 62.69，Real-Time 子集提升约 15 分；在 StreamingBench 与 ODV-Bench 上也分别提升 2–8 分，显著超越同类基线。

**⚠️ 局限性**

局限在于受限于固定内存预算时对计数、因果推理等需要全覆盖的查询敏感；伪问题库是固定通用的，缺乏领域适配；训练无关设计导致无法联合优化两阶段，难以进一步提升性能。

---

## 773. Adaptive Regularization for Sparsity Control in Bregman-Based Optimizers

**arXiv ID:** 2605.07892 | [PDF](https://arxiv.org/pdf/2605.07892v1)

**作者:** Ahmad Aloradi `[一作]` (FAU Erlangen-Nürnberg), Daniel Tenbrinck `[通讯]` (FAU Erlangen-Nürnberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种自适应正则化方案，用于Bregman稀疏优化框架中自动调节 λ，从而在训练中实现预设的稀疏率；

**💡 创新点**

创新点在于：1) 引入基于稀疏缺陷的 λ 递推更新策略，消除对 λ 的手工搜索；2) 证明并验证该方法在保持原有Bregman优化特性的同时能加速收敛并提升稀疏模型性能；

**🔧 技术方法**

采用了Bregman迭代（LinBreg、AdaBreg）与线性化Bregman正则化、拉伸弹性网（Elastic Net）正则化；利用梯度下降、镜像下降与 Adam 优化；

**📊 数据集**

在两大 ASV 公开数据集 VoxCeleb（多语言）和 CNCeleb（中文）上训练 ECAPA‑TDNN 与 ResNet34 模型；

**📈 对比分析**

与稠密模型、逐步裁剪以及非自适应 Bregman 进行对比。结果显示：在 75%–95% 的稀疏率下，稀疏模型 EER 与稠密模型持平，甚至在 OOD 设定下优于稠密模型；自适应方法收敛更快；

**⚠️ 局限性**

局限性包括：对极端稀疏率（>95%）下层级权重分配不均导致性能骤降；缺乏稀疏振荡稳定性理论分析；未来可研究针对特定层的 λ 重新缩放或更细粒度稀疏分配策略。

---

## 774. Melding LLM and temporal logic for reliable human-swarm collaboration in complex scenarios

**arXiv ID:** 2605.07877 | [PDF](https://arxiv.org/pdf/2605.07877v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 775. Enhancing Federated Quadruplet Learning: Stochastic Client Selection and Embedding Stability Analysis

**arXiv ID:** 2605.07888 | [PDF](https://arxiv.org/pdf/2605.07888v1)

**作者:** Ozgu Goksu `[一作]`, Nicolas Pugeault `[通讯]` (University of Glasgow)

**通讯引用:** 2015 | [OpenAlex ID](https://openalex.org/A5087140377)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FedQuad 框架，利用四元组损失在联邦学习中对非 IID、数据不平衡的表示学习进行改进。

**💡 创新点**

创新点在于设计随机四元组采样与双边 margin 四元组损失，显式最小化正样本距离、最大化负样本距离，从而防止表征坍塌。

**🔧 技术方法**

采用 FedAvg + 交叉熵 + 重新定义的四元组损失，结合随机客户端选择、Dirichlet 分割、Adam 优化等技术。

**📊 数据集**

使用 CIFAR‑10、CIFAR‑100 与 Tiny‑ImageNet 三个公开图像分类数据集进行实验。

**📈 对比分析**

与 FedAvg、SupCon‑FL、Triplet‑FL、Quadruplet‑FL、MOON 等基线在不同 α 与客户端数下进行对比；FedQuad 在大多数非 IID 场景下均取得最高或最接近最高的准确率。

**⚠️ 局限性**

局限性：需要至少三类样本才能构造四元组，单类或二类任务不适用；在极端异质或极大规模客户端场景下，性能仍会下降。

---

## 776. AERO-VIS: Asynchronous Event-based Real-time Onboard Visual-Inertial SLAM

**arXiv ID:** 2605.07885 | [PDF](https://arxiv.org/pdf/2605.07885v1)

**作者:** Yannick Burkhardt `[一作]` (ETH Zurich), Stefan Leutenegger `[通讯]` (ETH Zurich)

**通讯引用:** 14259 | [OpenAlex ID](https://openalex.org/A5006726091)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 AERO-VIS，一种异步事件-惯性 SLAM 系统，集成了轻量化的事件关键点检测器 SuperLitE，并实现了仅靠事件摄像机完成的无人机闭环控制与大尺度轨迹估计。

**💡 创新点**

创新点包括：① 基于常数事件计数的 MCTS_Ne 表示提升了事件特征在不同运动速率下的稳健性；② 设计了极度轻量化的 SuperLitE 网络，实现 90% 的推理速度提升；③ 将上述技术嵌入 OKVIS2 框架，形成全异步实时系统，使得无人机能够在受限计算资源上完成闭环控制。

**🔧 技术方法**

使用技术：事件摄像机 + IMU 传感、MCTS_Ne 事件表、SuperLitE 关键点检测与描述、OKVIS2 前端后端、余弦距离匹配、循环闭环检测、CUDA+TensorRT 推理、量化与异步线程调度。

**📊 数据集**

实验数据集：公开 rpg-stereo、TUM-VIE、VECtor；在本研究中亦使用自制无人机搭载事件相机和 IMU 进行室内外真实环境实验。

**📈 对比分析**

性能比较：在 NVIDIA Jetson Orin NX 上与 ESVO2、SDEVO、OKVIS2 等基线对比，AERO-VIS 在实时约束下平均 ATE 低于对手，尤其在 HDR 与高速运动场景中优于帧基 OKVIS2；在 2 km 城市行走实验中漂移保持在 1–2%，并成功完成循环闭环。

**⚠️ 局限性**

局限性：在标准飞行条件下，事件分辨率降低和事件表对震动的敏感性导致相对帧基 OKVIS2 的精度略低；循环闭环检测与后端优化仍是随轨迹增长的主要瓶颈。

---

## 777. Fast Byte Latent Transformer

**arXiv ID:** 2605.08044 | [PDF](https://arxiv.org/pdf/2605.08044v1)

**作者:** Julie Kallini `[一作]` (FAIR at Meta), Srinivasan Iyer `[通讯]` (FAIR at Meta)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 BLT Diffusion (BLT‑D)，一种在字节级语言模型中结合分块扩散的生成方法。

**💡 创新点**

创新点在于将块级离散扩散与 BLT 的动态分块层次结构结合，实现并行字节生成，并引入自我推测 BLT‑S 与扩散验证 BLT‑DV 两种推测式解码扩展。

**🔧 技术方法**

使用了字节级自编码器、全局 Transformer、半自回归解码器、离散扩散与置信度/熵界定的解码策略，并通过验证机制提升质量。

**📊 数据集**

在 BLT‑1T 语料上预训练，包括 1 万亿 tokens 的多来源文本，并在 FLORES‑101、HumanEval、MBPP 等任务上评估。

**📈 对比分析**

通过对比 NFEs、内存带宽与任务 BLEU / pass@1 分数，BLT‑D 在保持或略低任务性能的同时，将内存带宽和解码成本降低 50‑92%，而 BLT‑S 与 BLT‑DV 在不损失性能或以更低成本实现更快生成。

**⚠️ 局限性**

主要局限在于使用 NFEs 和估计内存带宽作为效率指标，缺乏在高度优化硬件实现下的真实推理速度评估，并且扩散块大小与任务质量间存在权衡。

---

## 778. STEPS: A Temporal Smooth Error Propagation Solver on the Manifolds for Test-Time Adaptation in Time Series Forecasting

**arXiv ID:** 2605.08005 | [PDF](https://arxiv.org/pdf/2605.08005v1)

**作者:** Jiaqi Liu `[一作]` (Xiamen University Malaysia), Ashwaq Qasem `[通讯]` (Xiamen University Malaysia)

**通讯引用:** 245 | [OpenAlex ID](https://openalex.org/A5088984992)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种面向时间序列预测的测试时刻自适应框架 STEPS，通过把前缀误差当作 Dirichlet 边界条件，在预测轨迹的时间流形上求解一个平滑且有界的误差场，从而在不更新模型权重的前提下修正冻结模型的输出。

**💡 创新点**

创新点在于：①将 TTA 视为一维时间流形上的 Dirichlet 边值问题；②设计局部平滑传播、全局误差记忆检索与时空流形融合三步联合求解器；③在边界条件噪声或稀疏时保持输出有界，提升鲁棒性；④实现无需在线梯度更新，完全冻结模型。

**🔧 技术方法**

使用的技术包括：FFT 频域周期估计、低 Dirichlet 能量的谐波平滑传播、线性岭回归的边界拟合、循环记忆更新（Global Error Memory）、神经解码器预测全局误差模式、基于 Sigmoid 的距离加权融合与修剪。

**📊 数据集**

实验数据集涵盖 6 个工业与气象时间序列：ETTh1、ETTh2、ETTm1、ETTm2、Exchange、Weather；四个冻结基线模型：DLinear、PatchTST、OLS、MICN。

**📈 对比分析**

与基线 TAFAS、PETSA、COSA‑F/P 及零射击（Zero‑Shot）相比，STEPS 在六个数据集、四个模型上平均降低 MSE 26.82%，在最优基线上提升约 12–20%（以相对误差减少衡量）。在前缀稀疏、噪声污染以及跨窗口记忆等鲁棒性测试中，STEPS 仍保持最小误差增长。

**⚠️ 局限性**

局限性包括：①对前缀误差的可靠性依赖较高，极端突变或前缀无信息时性能下降；②全局记忆对周期性变动的捕捉有限，长程非周期漂移时效果不佳；③采用固定时间几何和融合调度，难以自适应不同时序的周期与不确定性。

---

## 779. HEART: Hyperspherical Embedding Alignment via Kent-Representation Traversal in Diffusion Models

**arXiv ID:** 2605.07973 | [PDF](https://arxiv.org/pdf/2605.07973v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 780. SCOPE: Structured Decomposition and Conditional Skill Orchestration for Complex Image Generation

**arXiv ID:** 2605.08043 | [PDF](https://arxiv.org/pdf/2605.08043v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 781. TRAS: An Interactive Software for Tracing Tree Ring Cross Sections

**arXiv ID:** 2605.08025 | [PDF](https://arxiv.org/pdf/2605.08025v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 782. The Limits of AI-Driven Allocation: Optimal Screening under Aleatoric Uncertainty

**arXiv ID:** 2605.07979 | [PDF](https://arxiv.org/pdf/2605.07979v1)

**作者:** Santiago Cortes-Gomez `[一作]` (Carnegie Mellon University), Bryan Wilder `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2912 | [OpenAlex ID](https://openalex.org/A5079207566)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个两阶段资源分配框架，将有限的筛选预算与算法化风险评分结合，以在固定覆盖预算下最大化分配效率，并推导了最优筛选区间及其闭式解。

**💡 创新点**

证明最优筛选集合位于风险分数边缘区间，阐明筛选与算法化分配的互补关系；给出与分布无关的筛选价值凸性与递减收益；提供固定点算法计算阈值，并通过仿真与实证展示随阿勒托不确定性变化的筛选价值。

**🔧 技术方法**

使用贝叶斯风险分析、积分方程求解最优阈值；在仿真中采样 Beta(t,t) 族；在真实数据中训练逻辑回归、随机森林和 XGBoost 等模型估计风险。

**📊 数据集**

公开的哥伦比亚地理与冲突数据用于地雷检测预测，以及美国 2018 年 ACS 公共使用微数据样本用于低收入预测。

**📈 对比分析**

将最优筛选与无筛选（纯算法化）和随机筛选基线对比；在模拟实验中，精度可从 50% 提升至 100%（在 β=35% 时提升约 30%+）；在哥伦比亚案例中，XGBoost 精度从 39% 提升至 70%；在 ACS 案例中精度从 65% 提升至 80%。

**⚠️ 局限性**

假设风险估计为贝叶斯最优，未充分考虑模型估计误差对筛选决策的影响；预算参数预设，未探讨其联合优化；未深入讨论伦理、成本-收益平衡与实际操作中的约束。

---

## 783. Seeing Across Skies and Streets: Feedforward 3D Reconstruction from Satellite, Drone, and Ground Images

**arXiv ID:** 2605.07978 | [PDF](https://arxiv.org/pdf/2605.07978v1)

**作者:** Qiwei Wang `[一作]` (ShanghaiTech University), Yujiao Shi `[通讯]` (ShanghaiTech University)

**通讯引用:** 969 | [OpenAlex ID](https://openalex.org/A5002882477)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

训练并部署了一种单前向网络 Cross3R，能够在一次前向传播中同时恢复卫星、无人机与地面视角的三维点云、6-DoF 相机姿态，并实现跨视角定位。

**💡 创新点**

核心创新点包括：① 引入单幅 UAV 图像作为几何桥梁，突破传统 3-DoF 限制；② 设计可学习的正交尺度因子 ρ 以统一多视角坐标系；③ 采用傅里叶位置嵌入、交替注意力与多尺度融合机制；④ 在卫星分支中强制正交投影先验，确保尺度一致。

**🔧 技术方法**

技术手段主要基于 π^3 Transformer 结构，配合傅里叶位置编码、交替注意力、多尺度解码器、正交投影先验、尺度学习 ρ、深度先验融合等组件，实现端到端的跨视角三维重建与定位。

**📊 数据集**

主要使用自构建的 CrossGeo 数据集（约 278K 张图像，覆盖 85 个全球场景，含卫星、UAV 与地面图像及完整 6-DoF 位姿与稠密深度），并在 KITTI、AnyVisLoc 等公开基准上进行评估。

**📈 对比分析**

与 VGGT、π^3、AerialMD 等基线进行对比，Cross3R 在点云精度、相机姿态召回率、跨视角定位误差以及像素匹配成功率等多项指标均优于对手；在 KITTI 零样本设置下亦超越专门针对 KITTI 训练的跨视角方法。

**⚠️ 局限性**

局限性包括：只能一次处理最多三幅图像；仅支持针孔摄像机，未覆盖全景/鱼眼；对训练数据量高度依赖，需更多计算资源以支持更长序列和更大数据规模；模型尚未兼容多相机类型。

---

## 784. Reinforcement Learning for Exponential Utility: Algorithms and Convergence in Discounted MDPs

**arXiv ID:** 2605.08053 | [PDF](https://arxiv.org/pdf/2605.08053v1)

**作者:** Gugan Thoppe `[一作]` (Indian Institute of Science), Sanjay Bhat `[通讯]` (Tata Consultancy Services Limited)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出两种无模型的RL算法（双时标Q‑learning与单时标更新），用于在折扣MDP中优化指数效用，并证明它们的收敛性

**💡 创新点**

首次把指数效用的Bellman方程与L∞与sup‑log收缩性质结合，构造可收敛的Q‑learning风格算法，并给出单时标算法的理论分析

**🔧 技术方法**

使用动态规划、收缩映射、sup‑log (Thompson) 代数、时间尺度分离、局部Lipschitz/单调/齐次分析以及Dini导数等技术

**📊 数据集**

论文未使用任何具体数据集，全部为理论推导与证明

**📈 对比分析**

没有经验比较，主要通过理论证明收敛和速率（双时标为O(n^(-β/2))，单时标在标量情形下为Õ(n^(-1/2))）

**⚠️ 局限性**

缺乏实验验证；单时标速率仅在标量情况给出；需要生成模型；对高维向量的速率尚未给出

---

## 785. Position: Mechanistic Interpretability Must Disclose Identification Assumptions for Causal Claims

**arXiv ID:** 2605.08012 | [PDF](https://arxiv.org/pdf/2605.08012v1)

**作者:** Zezheng Lin `[一作]`, Fengming Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对10篇机制解释性论文进行审计，并扩展至30篇，发现缺乏专门的识别假设说明，提出通用的披露协议

**💡 创新点**

首次系统性审计识别假设缺失的普遍性，提出基于因果推断的披露规范，推动领域对因果声明透明度的重视

**🔧 技术方法**

使用手工编码规则、两人双重编码、Cohen's κ与Krippendorff's α评估一致性，并提供可复现的代码与审计报告

**📊 数据集**

审计样本包括10篇代表性论文（涵盖电路发现、稀疏自编码、因果抽象、探测等四大方法）以及从NeurIPS、ICLR等会议抽取的30篇论文；主要数据为论文的摘要与正文文本

**📈 对比分析**

通过编码维度（抽象声明类型、识别假设、可证伪性、验证度替代）比较，发现0/30论文具备专门的识别假设章节，约80%论文将验证指标误用为因果支持；无直接性能指标，但表明识别与验证的明显脱节

**⚠️ 局限性**

局限性包括样本规模小且偏向性强，扩展审计仅使用摘要级信息，编码规则敏感，且未涉及全篇文本或更广泛的领域覆盖

---

## 786. Hot Wire 5D+: Evaluating Cognitive and Motor Trade-offs of Visual Feedback for 5D Augmented Reality Trajectories

**arXiv ID:** 2605.08008 | [PDF](https://arxiv.org/pdf/2605.08008v1)

**作者:** Christian Masuhr `[一作]` (Hamburg University of Technology), Thorsten Schüppstuhl `[通讯]` (Hamburg University of Technology)

**通讯引用:** 769 | [OpenAlex ID](https://openalex.org/A5010136138)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

在一项30人实验中，比较了三种AR界面（Tracer、Gestalt、Reduced）对无训练用户在无接触的5D+轨迹追踪任务中的表现。

**💡 创新点**

创新点在于提出并量化了“5D+”轨迹（3D位置+2D方向+速度）在自由空间中的认知-运动权衡，并展示了沉浸式“Tracer”界面在抑制信息过载方面的优势。

**🔧 技术方法**

使用Magic Leap 2头戴显示器、定制光学追踪控制器、OptiTrack外部光学系统以及ART ANOVA和NASA‑TLX/SUS等统计与主观评估技术。

**📊 数据集**

实验采用自定义连续管道轨迹（约10 m）作为任务路径，并通过OptiTrack提供高精度外部参考轨迹进行验证。

**📈 对比分析**

比较方法为ART ANOVA对RMSE_Pos、RMSE_Ori、RMSE_Speed进行分解分析；结果显示Tracer在位置、方向与速度误差均优于Gestalt和Reduced，且在任务B（包含方向约束）下仍保持较低的认知负荷。

**⚠️ 局限性**

局限性包括仅对未受训新手进行评估、缺乏长期训练或专家数据、单一轨迹形状、有限的单向触觉反馈以及未考虑光学跟踪漂移对大规模工业部署的影响。

---

## 787. Graph-Structured Hyperdimensional Computing for Data-Efficient and Explainable Process-Structure-Property Prediction

**arXiv ID:** 2605.07999 | [PDF](https://arxiv.org/pdf/2605.07999v1)

**作者:** Jingzhan Ge `[一作]` (University of Connecticut), Farhad Imani `[通讯]` (University of Connecticut)

**通讯引用:** 4543 | [OpenAlex ID](https://openalex.org/A5058788547)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了基于图结构的高维向量符号学习框架 PSP‑HDC，用以在数据稀缺的多光子光还原制造中对过程–结构–性质（PSP）进行预测。

**💡 创新点**

创新点包括：① 将领域先验的 PSP 有向图嵌入为内部结构约束；② 设计可梯度学习的标量‑到‑高维向量编码器，针对异构量表自适应映射；③ 通过图对齐的绑定与捆绑实现交互约束；④ 在同一记忆中实现参数、组以及组内的多级可解释性，并引入记忆对齐与分离（MAS）指标评估学习进展。

**🔧 技术方法**

技术手段：高维向量符号学习（HDC）与关联记忆检索、基于余弦相似度的分类、图对齐的绑定/捆绑操作、交叉熵损失训练、MAS 监测、以及多级归因分析。

**📊 数据集**

使用了 3D OHMIC 研究平台的多光子光还原实验数据集，包含 58 个样本（60 个原始样本去除 2 个无效测量），每个样本含 4 个过程参数、4 个组成参数及 17 个孔隙形态参数，并以薄膜电阻率作为二分类目标。

**📈 对比分析**

通过 1000 次随机 80%/20% 拆分和跨过程折叠（hold‑out）测试，PSP‑HDC 在随机拆分上的平均准确率 0.910±0.077，跨折叠 0.896；相比 SVM、LR、RF、GPC、CNN 等传统基线，PSP‑HDC 在准确率、F1 等指标均显著提升（约 2.5–3.5%），且在过程域外泛化表现更稳健。

**⚠️ 局限性**

局限性：① 需要先验的可定义 PSP 图，若图结构不合适或需大幅调整需重新训练；② 对传感器/测量流程变更敏感；③ 目前仅实现二分类，连续或多分类仍待扩展；④ 对右删值（∞）处理仅采用截断，未充分建模；⑤ 在更大规模或更复杂的多属性预测场景下的可扩展性和稳定性尚待验证。

---

## 788. ECNUClaw: A Learner-Profiled Intelligent Study Companion Framework for K-12 Personalized Education

**arXiv ID:** 2605.08040 | [PDF](https://arxiv.org/pdf/2605.08040v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 789. Bayesian Sensitivity of Causal Inference Estimators under Evidence-Based Priors

**arXiv ID:** 2605.07993 | [PDF](https://arxiv.org/pdf/2605.07993v1)

**作者:** Nikita Dhawan `[一作]` (University of Toronto), Chris J. Maddison `[通讯]` (University of Toronto)

**通讯引用:** 18582 | [OpenAlex ID](https://openalex.org/A5054711904)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文扩展了 s‑value 框架，将其应用于因果推断中的无测量混杂、条件结果模型和协变量分布三类常见假设，并提出了基于真实世界先验的 Bayesian Sensitivity Value（BSV）来评估假设违反的期望敏感性。

**💡 创新点**

创新点在于：①从最坏情况转向期望敏感性评估，避免过度悲观；②通过构建经验先验（如 NHANES、UCI 数据）来刻画假设空间的概率分布；③提出 BSV 与传统 s‑value 的统一公式和计算方法。

**🔧 技术方法**

主要技术包括：s‑value 的凸优化求解（Lagrange 乘子、Entropic Mirror Descent），Monte Carlo 抽样求 BSV，重jection 采样估计条件期望，及与传统最坏情况敏感性比较的数值实验。

**📊 数据集**

使用了两个数据集：①仿真数据（多维二进协变量与异质处理效应），②真实糖尿病观察研究数据（Semaglutide vs Tirzepatide 以及 NHANES、UCI 公开数据库用于构建经验先验）。

**📈 对比分析**

比较方法：在相同的假设空间下计算 s‑value、均匀先验 BSV 和经验 BSV；评估指标为敏感性值范围、相关性与接受率。实验表明，s‑value 在高维假设空间趋于 1，缺乏区分度；而 BSV 保持显著波动，能够揭示不同子群的真实敏感性，且与经验先验一致。

**⚠️ 局限性**

局限性包括：①对赔率比等不可观测假设难以构建经验先验；②不同假设空间与度量的组合尚未统一；③重jection 采样在假设逆转概率低时效率低下；④BSV 对先验选择高度依赖，易受先验误设影响。

---

## 790. PET-Adapter: Test-Time Domain Adaptation for Full and Limited-Angle PET Image Reconstruction

**arXiv ID:** 2605.08030 | [PDF](https://arxiv.org/pdf/2605.08030v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 791. Evaluation of an Actuated Spine in Agile Quadruped Locomotion

**arXiv ID:** 2605.07988 | [PDF](https://arxiv.org/pdf/2605.07988v1)

**作者:** Nico Bohlinger `[一作]` (Technical University of Darmstadt), Jan Peters `[通讯]` (Technical University of Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在MuJoCo仿真环境下，对Silver Badger四足机器人使用1-DOF纵向脊柱进行训练，评估其在高速奔跑、爬楼梯、爬坡、跨墙以及爬行等敏捷任务中的表现

**💡 创新点**

提出并验证主动脊柱对四足机器人敏捷度提升的实证价值，首次量化脊柱在不同任务中的具体收益

**🔧 技术方法**

使用Proximal Policy Optimization（PPO）强化学习算法与RL-X实现训练，结合域随机化与观测噪声以提升对真实环境的迁移性

**📊 数据集**

利用自定义MuJoCo仿真环境（包含反向金字塔楼梯、斜坡、墙壁与天花板障碍等任务），生成的仿真数据作为训练与评估集

**📈 对比分析**

通过对比脊柱锁定与主动脊柱两种模型，使用无量纲的Froude数与能耗成本（COT）进行性能衡量；主动脊柱实现最高速度5.9 m/s、Froude≈9.7、COT≈2.1；可攀爬楼梯高度提升4 cm、坡度提高至0.8 rad、跳墙高度提升4 cm、爬行通道宽度提升2 cm

**⚠️ 局限性**

实验仅限仿真，未验证真实机器人硬件实现；脊柱仅为单自由度，横向或多自由度的效益未探究；对复杂真实环境的适应性仍需进一步评估

---

## 792. CA-SQL: Complexity-Aware Inference Time Reasoning for Text-to-SQL via Exploration and Compute Budget Allocation

**arXiv ID:** 2605.08057 | [PDF](https://arxiv.org/pdf/2605.08057v1)

**作者:** James Petullo `[一作]` (Brandeis University), Nianwen Xue `[通讯]` (Brandeis University)

**通讯引用:** 6806 | [OpenAlex ID](https://openalex.org/A5036715761)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 CA-SQL，一种基于任务难度动态扩展搜索空间的 Text-to-SQL 流水线，利用演化搜索生成多样化的模式子集，并采用奖励累加投票选取最终查询。

**💡 创新点**

创新点包括：① 用 LLM 评估任务难度并按难度比例动态调节候选查询的搜索宽度和深度；② 通过交叉和变异演化生成多样化的模式子集，显著提升候选查询的多样性；③ 引入基于累计奖励的投票方法（sum‑of‑rewards），优于传统多数投票。

**🔧 技术方法**

核心技术包括：LLM 推理时学习（in‑context 生成、Chain‑of‑Thought、Self‑Consistency）、演化搜索（交叉/变异）、奖励机制与投票、以及基于 GPT‑4o‑mini 的大模型推理。

**📊 数据集**

实验使用了 BIRD benchmark（95 个数据库、1534 个自然语言任务），评估执行准确率（Execution Accuracy）和 Soft F1 分数。

**📈 对比分析**

与其他 ICL 方法（MCS、E‑SQL、MCTS 等）在同一模型（GPT‑4o‑mini）下对比，CA‑SQL 在 “Challenging” 级别上达 51.72% 的执行准确率，整体 61.06% 的准确率和 68.77% 的 Soft F1，均超过现有最优方案。

**⚠️ 局限性**

局限性包括：仅在 GPT‑4o‑mini 上验证；对任务难度评分的方差与多次采样未做充分研究；在简单/中等难度任务上表现略逊于深度优化方法；高计算预算下的可扩展性待进一步评估。

---

## 793. Active Embodiment Identification with Reinforcement Learning for Legged Robots

**arXiv ID:** 2605.08020 | [PDF](https://arxiv.org/pdf/2605.08020v1)

**作者:** Nico Bohlinger `[一作]` (Technical University of Darmstadt), Jan Peters `[通讯]` (Technical University of Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对多种四足与人形机器人，设计并训练了一个联合信息寻求行为和显式身体识别的主动身体识别方法。

**💡 创新点**

将身体识别问题视为主动感知问题，结合历史增强的 URMA 架构和端到端强化学习，使机器人通过主动交互实时估计自身关节与全局参数。

**🔧 技术方法**

采用 PPO 强化学习、URMA 编码器/解码器、GRU 记忆单元、MLP 输出层，以及指数型识别奖励与极端身体随机化，并使用 MJX 物理引擎进行仿真。

**📊 数据集**

在仿真环境中对 4096 并行实例进行随机化（Unitree Go2、ANYmal C、Unitree G1、Booster T1 等），无传统公开数据集，数据来源为多样化的身体参数随机化。

**📈 对比分析**

通过与随机化范围对比，评估预测误差；关节级惯量误差≈0.005 kg·m²，关节角度误差≈0.2 rad，总质量误差≈0.25 kg，表现出较高准确性，但对最大扭矩/速度等极端参数识别效果不足。

**⚠️ 局限性**

策略倾向于保持稳定站姿，缺乏激烈动作导致无法充分识别最大扭矩/速度；依赖仿真，需在真实硬件上微调；当身体参数动态变化时识别精度可能下降。

---

## 794. Planarizing Gadgets for (k, l)-tight Graphs Do Not Exist

**arXiv ID:** 2605.08016 | [PDF](https://arxiv.org/pdf/2605.08016v1)

**作者:** Archit Chauhan `[一作]` (Iit Bombay), Thomas Thierauf `[通讯]` (Ulm University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文证明了在(k,l)-紧图（k≥2,0≤l≤2k−1）和(k,l)-稀疏图中不存在任何常数大小的平面化工具（planarizing gadget），从而否定了用平面化技巧将该类图的识别问题化简为平面图情形的常规方法。

**💡 创新点**

创新点在于给出了一个无条件、普适的证明，说明平面化工具在此类问题中根本不存在；并将此结果推广到二分图完美匹配问题，展示了更广泛的非平面化可行性。

**🔧 技术方法**

采用了图论中稀疏性与连通性性质的组合证明，利用边数上界与下界的矛盾、子图稀疏性与路径存在性的精细分析，并通过“边替换”与“子图替换”技术构造反例。

**📊 数据集**

论文为理论研究，未使用任何实验数据集。

**📈 对比分析**

由于是纯理论证明，没有实验对比；但作者讨论了该结果对现有多项式时间或并行算法的限制，并指出无法通过平面化工具实现进一步的时间复杂度改进。

**⚠️ 局限性**

局限性在于仅适用于(k,l)-紧图和(k,l)-稀疏图，无法直接推广到(k,l)-跨度图；此外，结果仅否定了一类特定的平面化方法，未排除其他可能的图转换或绘图策略。

---

## 795. SphereVAD: Training-Free Video Anomaly Detection via Geodesic Inference on the Unit Hypersphere

**arXiv ID:** 2605.08003 | [PDF](https://arxiv.org/pdf/2605.08003v1)

**作者:** Chao Huang `[一作]` (Sun Yat-sen University), Xiaochun Cao `[通讯]` (Sun Yat-sen University)

**通讯引用:** 27341 | [OpenAlex ID](https://openalex.org/A5068837264)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了SphereVAD，一种全无训练、零样本的视频异常检测框架；

**💡 创新点**

创新点在于把异常判别转化为单位超球面上的von Mises–Fisher (vMF)几何推理，利用多模大型语言模型的中间层特征并通过球面中心化、跨视频全景注意力以及球面几何拉伸实现无梯度学习的异常检测；

**🔧 技术方法**

采用多模大型语言模型（如Qwen3.5）冻结提取中间层特征，进行球面中心化、vMF原型构造、全景注意力聚合和vMF导向球面几何拉伸；

**📊 数据集**

在XD‑Violence、UCF‑Crime和UBnormal三个公开视频异常检测基准上进行实验；

**📈 对比分析**

与现有零样本和训练无关方法对比，SphereVAD在所有三个基准上取得了最优或接近全监督方法的性能，AP和AUC分别为86.99%、86.38%和76.46%；

**⚠️ 局限性**

局限在于对长时序视频需要内部三类划分，短视频（如UBnormal）无法使用球面几何拉伸模块，且对不同模态特征融合的鲁棒性仍待进一步提升。

---

## 796. Dooly: Configuration-Agnostic, Redundancy-Aware Profiling for LLM Inference Simulation

**arXiv ID:** 2605.07985 | [PDF](https://arxiv.org/pdf/2605.07985v1)

**作者:** Joon Ha Kim `[一作]` (University of Texas at Austin), Daehyeok Kim `[通讯]` (University of Texas at Austin)

**通讯引用:** 861 | [OpenAlex ID](https://openalex.org/A5022893621)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种可配置无关、冗余感知的LLM推理性能分析框架Prof，利用污点传播标记张量维度并通过层次上下文解析实现状态化操作的隔离，自动化收集和复用操作延迟并构建可直接使用的回归模型

**💡 创新点**

创新点在于：①通过污点传播对每个张量维度来源进行语义化标记，识别跨配置相同的操作；②使用层次上下文解析在不需要手工插桩的情况下重建状态化操作所需的运行时上下文；③结合签名匹配实现跨模型、跨后端的操作重复剔除，大幅降低分析成本

**🔧 技术方法**

主要技术包括污点传播、PyTorch dispatcher级别拦截、操作集求解器、签名生成与去重、基于SQLite的延迟数据库、回归模型推理与仿真器集成

**📊 数据集**

使用12个7B–8B规模的多种架构模型（Dense、GQA、MoE、Sliding‑Window），结合vLLM与SGLang两大推理引擎，三种注意力后端（FlashInfer、FlashAttention、TritonAttention），两款GPU平台（A100、H100），以及ShareGPT4真实工作负载和合成的prefill/ decode 场景

**📈 对比分析**

与传统逐配置逐模型逐后端的逐个操作分析做对比，Prof在12模型/3后端/2GPU的实验中将GPU时长从26.6小时压缩到约11.6小时（约56%节省）；仿真精度在TTFT MAPE <5%，TPOT MAPE <8%，调度预测误差 <0.5%

**⚠️ 局限性**

局限包括：①对跳过PyTorch dispatcher的Triton自研算子仅能在模块级别得到覆盖；②当前污点传播仅针对文本请求，需手动扩展多模态入口；③MoE的专家路由仅采用随机平均，未建模数据驱动的路由导致预测误差

---

## 797. MoCoTalk: Multi-Conditional Diffusion with Adaptive Router for Controllable Talking Head Generation

**arXiv ID:** 2605.08050 | [PDF](https://arxiv.org/pdf/2605.08050v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 798. Where's the Plan? Locating Latent Planning in Language Models with Lightweight Mechanistic Interventions

**arXiv ID:** 2605.07984 | [PDF](https://arxiv.org/pdf/2605.07984v1)

**作者:** Nicole Ma `[一作]` (Stanford University), Nick Rui `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型在生成押韵对句时是否会在内部形成并利用未来规划信息，并通过线性探针和激活补丁两种轻量级技术对三大开源模型族（Qwen3、Gemma-3、Llama-3）进行规模化实验。

**💡 创新点**

创新点在于首次将“规划位形成”（planning‑site formation）概念引入到语言模型解释性研究，并揭示仅Gemma-3‑27B在第30层左右出现从“末尾单词”向“换行符”迁移的因果规划位，且该迁移可归因于仅五个注意力头。

**🔧 技术方法**

技术手段包括：线性探针（linear probing）提取隐藏层信息，激活补丁（activation patching）测试因果影响，以及两阶段路径补丁（two‑stage path patching）精准定位信息路由的注意力头。

**📊 数据集**

使用的数据集为从Claude Sonnet 4.6自动生成的1200条押韵对句，此外对一般文本（The Pile）做负向对照。

**📈 对比分析**

与传统的基线（unigram、一般文本探针）相比，探针在换行符位置表现出显著提升，但仅Gemma-3‑27B的补丁实验显示该位置对生成有实质性因果作用，补丁率可达0.63，且通过仅五个注意力头即可恢复约90%的规划能力；其他模型则保持对末尾单词的依赖。

**⚠️ 局限性**

局限性包括：仅在押韵对句任务上验证，样本量有限；缺乏对其他结构化生成任务的推广；无法解释为何Gemma-3‑27B在架构或训练上与其他模型差异显著；补丁实验对提示对齐和头集合的鲁棒性尚待进一步评估。

---

## 799. It Just Takes Two: Scaling Amortized Inference to Large Sets

**arXiv ID:** 2605.07972 | [PDF](https://arxiv.org/pdf/2605.07972v1)

**作者:** Antoine Wehenkel `[一作]` (Apple), Chris Pollard `[通讯]` (University of Warwick)

**通讯引用:** 65603 | [OpenAlex ID](https://openalex.org/A5103245766)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了PAIRS（Pretraining Aggregators for Inference at Arbitrary Set-sizes）方法，先在仅包含1~2个样本的小集上训练mean‑pool Deep Set编码器，再冻结编码器并在已预聚合的嵌入上微调后端分布模型，从而实现规模可扩展的神经后验估计。

**💡 创新点**

核心创新是理论证明：在满足可足够的条件下，对集合大小为1或2的训练即可获得与任意大集合相同的充分统计量（仅差一个仿射变换），从而将训练成本与部署集大小解耦。

**🔧 技术方法**

技术包括：mean‑pool Deep Set编码器、神经后验估计（normalizing flow 或 flow‑matching）、预聚合嵌入缓存、分阶段训练（预训练、缓存、微调）以及对指数族假设的理论分析。

**📊 数据集**

实验使用多种数据集：二维高斯生成模型、粒子物理“bump hunt”数据、图像任务（Circle Radius、Digit Expectation）、3D物体属性（ModelNet40）、分子LogP、以及高维图像生成的多视角渲染场景。

**📈 对比分析**

与基线（单个样本训练、1–10大小预训练、Gaussian回归、端到端大集合训练）比较，PAIRS在NLL、MAE等指标上往往匹配或优于所有基线，同时在计算成本上显著低于端到端训练，尤其在大N（千级）场景下实现高精度后验。

**⚠️ 局限性**

限制包括：理论依赖于指数族边缘分布和充分统计量存在，实测时对嵌入维度敏感；对非指数族或无有限维充分统计量的问题仍需进一步研究；对最大集合大小的推断能力仍受限于mean‑pool假设，不能直接扩展到max‑pool或注意力聚合。

---

## 800. Accurate and Efficient Statistical Testing for Word Semantic Breadth

**arXiv ID:** 2605.08048 | [PDF](https://arxiv.org/pdf/2605.08048v1)

**作者:** Yo Ehara `[一作]` (Tokyo Gakugei University), Yo Ehara `[通讯]` (Tokyo Gakugei University)

**通讯引用:** 196 | [OpenAlex ID](https://openalex.org/A5007171156)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种使用 Householder 反射对齐后再进行置换检验的方法，用以比较两种词类型的语义宽度。

**💡 创新点**

创新点在于通过单个 Householder 反射消除均值方向差异，显著降低 Type‑I 错误，并实现了 GPU 并行的置换检验。

**🔧 技术方法**

使用方向统计（均值向量、均值结果长度）、Householder 反射、非参数置换检验以及 GPU 上的批量矩阵运算。

**📊 数据集**

实验数据集包括 BNC、BCCWJ 语料，使用 ModernBERT、BERT‑tiny、BERT‑large 等预训练模型。

**📈 对比分析**

与传统无对齐的置换检验相比，该方法在 Type‑I 错误上降低约 32.5%，精度提升至 60–62%，并实现了 23 倍的 GPU 加速。

**⚠️ 局限性**

局限性包括仅针对词级向量云，假设向量云单峰且各向同性，未处理多模态或方向异性问题；对词表和分词策略敏感。

---

## 801. Don't Get Your Kroneckers in a Twist: Gaussian Processes on High-Dimensional Incomplete Grids

**arXiv ID:** 2605.08036 | [PDF](https://arxiv.org/pdf/2605.08036v1)

**作者:** Mads Greisen Højlund `[一作]` (Aarhus University), Ove Christiansen `[通讯]` (Aarhus University)

**通讯引用:** 17834 | [OpenAlex ID](https://openalex.org/A5011917108)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种名为 CUTS‑GPR 的新方法，用于在高维、数据量巨大的场景下实现数值精确的高斯过程回归。

**💡 创新点**

创新点在于将可加核（additive kernel）与结构化、稀疏（incomplete）网格相结合，利用网格的下闭性（cud）属性和核矩阵的分解（L·M·U），实现极快的核矩阵-向量乘法，达到近线性或线性 N 复杂度以及低阶多项式 D 复杂度。

**🔧 技术方法**

核心技术包括：可加核、稀疏网格切分（cut‑level α）、核矩阵的分块分解与稀疏化、预条件共轭梯度求解、随机迹估计、Adam 优化器以及对核中心化和长度尺度的正则化。

**📊 数据集**

使用了 24 维势能面（PES）数据，共 10 种有机分子；训练集采用粗网格 (n=7)，共 447 265 点；测试集采用细网格 (n=11)，共 1 604 576 点。

**📈 对比分析**

与传统稀疏变分高斯过程（SVGP）做比较。CUTS‑GPR 的预测误差（MAX、RMSE、MAE）均低于 SVGP；训练与预测的墙钟时间在 2–4 分钟，远快于 SVGP（约 40 分钟）。

**⚠️ 局限性**

主要限制：仅适用于满足下闭性 (cud) 的结构化网格，处理非网格或缺失数据需额外切分；在极大测试集上计算预测方差仍是瓶颈，未来需结合 Lanczos 等方法提升效率。

---

## 802. STARFlow2: Bridging Language Models and Normalizing Flows for Unified Multimodal Generation

**arXiv ID:** 2605.08029 | [PDF](https://arxiv.org/pdf/2605.08029v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 803. Adaptive Domain Decomposition Physics-Informed Neural Networks for Traffic State Estimation with Sparse Sensor Data

**arXiv ID:** 2605.08028 | [PDF](https://arxiv.org/pdf/2605.08028v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 804. MPD$^2$-Router: Mask-aware Multi-expert Prior-regularized Dual-head Deferral Router in Glaucoma Screening and Diagnosis

**arXiv ID:** 2605.08024 | [PDF](https://arxiv.org/pdf/2605.08024v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 805. Reason to Play: Behavioral and Brain Alignment Between Frontier LRMs and Human Game Learners

**arXiv ID:** 2605.08019 | [PDF](https://arxiv.org/pdf/2605.08019v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 806. Interpreting Reinforcement Learning Agents with Susceptibilities

**arXiv ID:** 2605.08007 | [PDF](https://arxiv.org/pdf/2605.08007v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 807. Globally Optimal Training of Spiking Neural Networks via Parameter Reconstruction

**arXiv ID:** 2605.08022 | [PDF](https://arxiv.org/pdf/2605.08022v1)

**作者:** Himanshu Udupi `[一作]` (University of Illinois Urbana-Champaign), ChengXiang Zhai `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 31037 | [OpenAlex ID](https://openalex.org/A5028518494)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于参数重构的全局最优脉冲神经网络训练方法，避免传统 surrogate‑gradient 误差累积。

**💡 创新点**

创新点在于将并行阈值网络的凸化理论推广到并行递归阈值网络，并证明 LIF‑SNN 是其结构化特殊情况，从而得到零对偶间隙的凸双重问题。

**🔧 技术方法**

采用路径正则化、凸双重框架、见证字典构造以及固定或预训练 witness 生成，实现可求解的全局最优参数重构。

**📊 数据集**

实验使用算术加法（不同基数和序列长度）、first‑last‑XOR 与 MNIST‑Seq 等序列任务作为数据集。

**📈 对比分析**

与 surrogate‑gradient 基线以及其继续训练版本对比，参数重构方法在训练集与 OOD 任务上均显著提升精度，且与 surrogate‑gradient 结合可进一步提升性能。

**⚠️ 局限性**

局限性包括仅在算术、XOR 与 MNIST‑Seq 这类简单序列任务上验证，且仅针对阈值 RNN 结构，未扩展到 LSTM‑style 或注意力增强的 SNN。

---

## 808. Nash without Numbers: A Social Choice Approach to Mixed Equilibria in Context-Ordinal Games

**arXiv ID:** 2605.07996 | [PDF](https://arxiv.org/pdf/2605.07996v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 809. Tool Calling is Linearly Readable and Steerable in Language Models

**arXiv ID:** 2605.07990 | [PDF](https://arxiv.org/pdf/2605.07990v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 810. Learning CLI Agents with Structured Action Credit under Selective Observation

**arXiv ID:** 2605.08013 | [PDF](https://arxiv.org/pdf/2605.08013v1)

**作者:** Haoyang Su `[一作]` (Fudan University), Ying Wen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 3757 | [OpenAlex ID](https://openalex.org/A5101647485)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在文件系统环境中通过命令行接口（CLI）与工作空间交互的代理学习问题，提出了可在部分观察下选择关键信息并利用shell命令结构进行强化学习的方法。

**💡 创新点**

创新点在于：①引入σ‑Reveal机制，在推理时根据token预算选择最具相关性的工作空间子树；②提出Action Advantage Assignment (A³)，通过episode、turn、tree三通道的优势信号融合，直接利用AST结构分配信用；③在ShellOps和ShellOps‑Pro两个可验证的CLI任务数据集上验证并展示显著性能提升。

**🔧 技术方法**

技术包括：基于Tree‑Sitter的bash AST签名与Levenshtein距离进行命令相似度度量；子树预算优化的σ‑Reveal上下文选取；A³优势计算与PPO策略梯度；使用Qwen3‑14B语言模型进行policy训练与推理；在GPU上使用SGLang进行环境仿真与批量rollout。

**📊 数据集**

使用了ShellOps（1624条任务）与ShellOps‑Pro（150条更难任务）两个自建数据集，并在公开的AgentBench、DataBench、EHRCon、TableBench等六个基准上进行评估。

**📈 对比分析**

与ReACT、LATS、rStar、GSPO、GiGPO、HGPO、RetroAgent等基线比较，A³在Exact Match、Pass@k等指标上均显著优于RL基线，尤其在ShellOps混合任务上取得最高的21.9%（Vanilla）/24.6%（σ‑Reveal）EM；在ShellOps‑Pro长序列上，在相同参数规模下逼近大模型的性能。

**⚠️ 局限性**

局限性包括：①AST结构相似的命令可能因路径、文件内容或先前状态导致不同效果；②σ‑Reveal在任务相关性弱的文件层次中无法完全替代全面搜索；③实验仅验证了shell‑文件系统环境，GUI、网络、嵌入式等其他交互场景仍需进一步验证。

---

## 811. Abductive Reasoning with Probabilistic Commonsense

**arXiv ID:** 2605.08011 | [PDF](https://arxiv.org/pdf/2605.08011v1)

**作者:** Joseph Cotnareanu `[一作]` (McGill University), Mark Coates `[通讯]` (McGill University)

**通讯引用:** 7017 | [OpenAlex ID](https://openalex.org/A5009031715)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种结合大型语言模型与形式逻辑求解器的概率推理框架，用于解决需要归纳常识的推理任务；

**💡 创新点**

创新点在于将常识视为各人不同的随机信念，定义了“归纳概率”并通过抽样多个人的常识集合来评估命题真值；

**🔧 技术方法**

使用技术包括大语言模型进行思考链生成与自然语言到逻辑的翻译、形式逻辑求解器进行证明验证、蒙特卡罗抽样与贪婪搜索（近似最短证明路径）等；

**📊 数据集**

实验数据集涵盖 FOLIO（改造版）、CosmosQA 与 QUAIL 三个逻辑/阅读推理数据集；

**📈 对比分析**

与链式思考、Self-Consistency、LoT、LLM‑Tres、ARGOS、If‑Beam 等基线对比，PACS 在 FOLIO 与 QUAIL 上分别提高 8% 与 6% 的准确率，整体表现优于现有方法；

**⚠️ 局限性**

主要局限在于对常识分布的近似假设（如均匀分布）不够精确，以及依赖预训练 LLM 的逻辑翻译可能引入错误。

---

## 812. Rethinking Dense Optical Flow without Test-Time Scaling

**arXiv ID:** 2605.08000 | [PDF](https://arxiv.org/pdf/2605.08000v1)

**作者:** Praroop Chanda `[一作]` (Texas A&M University), Suryansh Kumar `[通讯]` (Texas A&M University)

**通讯引用:** 758 | [OpenAlex ID](https://openalex.org/A5002526108)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种单前向传递的密集光流估计框架，利用冻结的视觉与几何先验实现无迭代优化的光流推断。

**💡 创新点**

创新点在于将大型预训练的视觉基础模型（DINOv2）和深度基础模型的语义与几何特征进行融合，证明强先验可以替代传统的多步测试时迭代改进，从而显著降低推断计算成本。

**🔧 技术方法**

使用的技术包括：冻结的DINOv2视觉特征提取器、冻结的Depth Anything V2深度先验、轻量级跨模态融合网络、全局匹配与流传播的Transformer式注意机制。

**📊 数据集**

训练集为 FlyingChairs 与 FlyingThings3D；交叉验证在 Sintel（Clean/Final）和 KITTI 2015 上；同时还在 TSKH 组合数据集上进行微调。

**📈 对比分析**

与 SEA‑RAFT、GMFlow、RAFT、FlowSeek 等传统方法对比，在不使用任何迭代细化的情况下，在 Sintel Final 上实现 2.81 EPE，明显优于 SEA‑RAFT（4.32）和 GMFlow 未细化版，整体表现与多步细化方法相当，体现了单前向推断的高效与准确。

**⚠️ 局限性**

局限性包括：在大遮挡、细结构或极细运动边界处的精度下降；依赖预训练基础模型的质量与偏差，且这些模型的训练成本不在本文范围内。

---

## 813. 6D Pose Estimation via Keypoint Heatmap Regression with RGB-D Residual Neural Networks

**arXiv ID:** 2605.08059 | [PDF](https://arxiv.org/pdf/2605.08059v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 814. Towards Highly-Constrained Human Motion Generation with Retrieval-Guided Diffusion Noise Optimization

**arXiv ID:** 2605.08054 | [PDF](https://arxiv.org/pdf/2605.08054v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 815. Susceptibilities and Patterning: A Primer on Linear Response in Bayesian Learning

**arXiv ID:** 2605.07980 | [PDF](https://arxiv.org/pdf/2605.07980v1)

**作者:** Chris Elliott `[一作]` (Timaeus), Daniel Murfet `[通讯]` (Timaeus)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并实现了利用统计力学中的易感度（susceptibility）理论来刻画与解释神经网络内部结构的框架，定义了针对数据分布扰动的后验期望梯度，并以此构造两类重要矩阵：针对单样本损失的影响矩阵（influence matrix）和针对模型组件的结构易感矩阵（structural susceptibility matrix），并通过谱分解和伪逆求解实现“模式化”即给定结构目标时寻找最优数据扰动。

**💡 创新点**

创新点在于将热力学的涨落-耗散定理（fluctuation–dissipation theorem）推广到贝叶斯后验层面，形成通用的易感度定义；利用该定义同时得到影响函数和结构易感矩阵；将结构易感矩阵视为从数据分布到结构坐标的切线映射，并通过其伪逆实现逆向设计；此外，引入组件限制采样（weight‑restricted SGLD）和归一化/标准化步骤以处理真实后验中的delta函数观测，解决了原方法在高维/奇异模型上的估计难题。

**🔧 技术方法**

核心技术包括：贝叶斯后验和温度化（Gibbs posterior）；易感度的协方差表达式（covariance form）和梯度公式；SGLD 采样及其在全局与组件限制下的实现；谱分解（SVD）与岭正则化的伪逆；以及对单样本权重导数的解析证明；此外，还用解析展开（Laplace approximation）对正则与奇异情形下的易感度进行理论推导。

**📊 数据集**

实验主要在两类数据上进行：① 用于验证理论的二维 Ising 模型（$20	imes20$ 网格）；② 在小型 Transformer（如 GPT‑2‑小型）上训练文本模型，使用公开文本数据集（WikiText‑2 / Penn Treebank 等）进行学习，并采样以构建影响矩阵与结构易感矩阵。

**📈 对比分析**

与传统影响函数和训练数据归因方法对比，结构易感矩阵能够更细粒度地将模型组件（如注意力头、MLP 层）聚类为可解释功能组，实验中在 Transformer 上实现了对注意力头功能的精准归因，并通过伪逆模式化实现了对模型结构的目标驱动干预，达到与手工调整相近甚至更优的性能（如保持相同验证准确率的同时显著降低对不必要数据模式的依赖）。

**⚠️ 局限性**

主要局限包括：① 需要后验分布的高质量采样，SGLD 的混合时间和样本量对估计精度影响大；② 对模型的局部最优假设（如参数在训练终点近似为 $w^*$）在大规模非凸网络中难以保证；③ 在极度奇异或高维参数空间中理论推导的有效性尚待进一步验证；④ 计算复杂度随模型规模增长显著提升，尤其是伪逆和谱分解；⑤ 伪逆对小奇异值极度敏感，需岭正则化等手段来稳定，选择正则化参数仍是经验性问题。

---

## 816. Object Hallucination-Free Reinforcement Unlearning for Vision-Language Models

**arXiv ID:** 2605.08031 | [PDF](https://arxiv.org/pdf/2605.08031v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 817. Uncertainty-Aware Structured Data Extraction from Full CMR Reports via Distilled LLMs

**arXiv ID:** 2605.08045 | [PDF](https://arxiv.org/pdf/2605.08045v1)

**作者:** Yi Yu `[一作]`, Yuan Xue `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出轻量化框架 CMR‑EXTR，能够一次性将心脏磁共振（CMR）报告转换为 52 字段 JSON 并为每个字段输出置信度，支持离线推理；

**💡 创新点**

创新点包括：①基于教师‑学生蒸馏把 20B GPT‑OSS 迁移到 1B Llama‑3.2，实现高效离线模型；②三原则置信度评估（分布、稳定性、物理一致性）生成可审计的字段置信度；③仅需极少人工校验即可构建高质量训练集；

**🔧 技术方法**

采用大模型蒸馏、LoRA 微调、JSON schema 强化生成、分布式置信度评分、采样稳定性评估以及 22 条 CMR 公式一致性检查等技术；

**📊 数据集**

使用 1100 条去标识化的 OSU 病院 CMR 报告，包含 5 类疾病标签（CAD、HCM、DCM、Ebstein、PAH）；

**📈 对比分析**

与 GPT‑OSS‑20B、Llama‑3.2‑1B 对比实验显示，CMR‑EXTR 变量级准确率 99.65%，报告级 89.55%，分类准确率 97.04%，显著优于基线模型；

**⚠️ 局限性**

局限性包括：仍需对置信度低于 0.7 的字段进行人工校验；在低置信度情况下错误率仍较高；目前仅验证了 CMR 领域，未测试跨模态或其他医学报告的泛化能力。

---

## 818. Beyond Pairs: Your Language Model is Secretly Optimizing a Preference Graph

**arXiv ID:** 2605.08037 | [PDF](https://arxiv.org/pdf/2605.08037v1)

**作者:** Ning Liu `[一作]` (Amazon), Shervin Malmasi `[通讯]` (Amazon)

**通讯引用:** 4578 | [OpenAlex ID](https://openalex.org/A5059403175)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了GraphDPO，一种将多样本偏好构建成有向无环图（DAG）并直接优化的对齐方法；

**💡 创新点**

创新点在于把DPO从仅处理独立对偶对的形式扩展到图结构，利用等价类掩码处理平局关系、通过图邻域聚合实现全局传递，保持线性复杂度，并可加入锚定节点实现早期稳定；

**🔧 技术方法**

技术包括Direct Preference Optimization、Plackett–Luce风格的图损失、等价类构造、层级DAG化、锚定策略、KL正则化以及温度调度；

**📊 数据集**

使用了数学推理任务GSM8K、MATH‑500以及程序合成任务APPS的数据集；

**📈 对比分析**

与OTS、SFT、GRPO、SPIN、MPO、SWEPO、PRO、LiPO等多种基线相比，GraphDPO在GSM8K、MATH‑500和APPS上均取得最高的准确率/通过率，提升幅度从约2‑5个百分点不等；

**⚠️ 局限性**

局限性包括对采样rollout质量高度依赖、等价类层次化的DAG表达在稀疏或不完全可比较的场景下可能不足、以及在候选数极大时仍可能出现计算开销。

---

## 819. Collaborator or Assistnat? How AI Coding Agents Partition Work Across Pull Request Lifecycles

**arXiv ID:** 2605.08017 | [PDF](https://arxiv.org/pdf/2605.08017v1)

**作者:** Young Jo `[一作]` (University of Toronto), Safwat Hassan `[通讯]` (University of Toronto)

**通讯引用:** 494 | [OpenAlex ID](https://openalex.org/A5022060601)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对29,585条 AI 编码工具生成的 PR 生命周期进行挖掘，构建 Initiator × Approver 交互情景分类和 Collaborator–Assistant 维度。

**💡 创新点**

提出基于 PR 生命周期的“工具角色谱”与六种交互情景的体系，并揭示自动化授权合并的测量边界。

**🔧 技术方法**

使用流程挖掘、事件日志解析、模式匹配分类和状态机绘制等技术。

**📊 数据集**

基于 AIDev 数据集，涵盖五款 AI 编码工具（OpenAI、Copilot、Devin、Cursor、Claude Code）以及 33,600 条 PR 记录。

**📈 对比分析**

通过卡方检验、Cramér's V、转移概率和中位时长等指标对工具进行比较，发现工具间的交互情景分布差异显著，Cramér's V=0.50，显示强关联；但自动化合并率极低。

**⚠️ 局限性**

局限在于缺乏分支保护、合并策略等仓库治理元数据，无法区分自动合并与人为授权合并，且工具样本与仓库选择可能导致偏倚。

---

## 820. Semantic Smoothing for Language Models via Distribution Estimation and Embeddings

**arXiv ID:** 2605.07994 | [PDF](https://arxiv.org/pdf/2605.07994v1)

**作者:** Haricharan Balasundaram `[一作]` (IIT Madras), Andrew Thangaraj `[通讯]` (IIT Madras)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于语义相似度的语言模型平滑方法——语义平滑（Semantic Smoothing），通过词/上下文嵌入共享统计信息来改进n-gram模型的概率估计。

**💡 创新点**

将模型困惑度分解为经验熵与KL散度之和，并证明在Lipschitz-logit模型下，上下文嵌入的相近导致对应的下一词分布在KL距离上也相近，从而把平滑问题转化为带KL邻域约束的分布估计问题，给出了匹配上下界的最优插值估计器。

**🔧 技术方法**

使用词嵌入（Word2Vec、GloVe、GPT‑2）来量化上下文语义相似度，采用KL散度与Lipschitz约束，利用插值加权实现平滑；实验中结合传统平滑（add‑β、Kneser‑Ney）和最近邻/软最小化策略。

**📊 数据集**

在合成Markov链（100词、低秩转移矩阵）和真实文本数据WikiText‑103上进行实验，使用Word2Vec、GloVe、GPT‑2等嵌入。

**📈 对比分析**

与未平滑的add‑β、Kneser‑Ney以及基线最小熵下界进行比较。实验显示：在合成数据中，语义平滑显著降低了测试困惑度；在WikiText‑103上，随着同义词数量增加，困惑度持续下降，且相对改进在不同嵌入和参数设置下均显著，最优时可比基线低约30‑50%。

**⚠️ 局限性**

仅在二元组（bigram）上下文下验证；对更长上下文需高效最近邻搜索；插值权重仅基于理论上界，未做端到端最优调优；使用通用嵌入，未针对平滑任务进行专门训练，可能限制进一步提升。

---

## 821. Towards Apples to Apples for AI Evaluations: From Real-World Use Cases to Evaluation Scenarios

**arXiv ID:** 2605.07986 | [PDF](https://arxiv.org/pdf/2605.07986v1)

**作者:** Yee-Yin Choong `[一作]` (National Institute of Standards and Technology), Hong Shen `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3540 | [OpenAlex ID](https://openalex.org/A5062298555)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种可复现的流程，将金融服务行业专家提供的高层AI用例转化为107个详细、可测试的评估场景，并通过LLM与人工双重审查确保场景质量。

**💡 创新点**

创新点在于将人机协作与人本设计（HCD）原则结合，形成三步（标题/描述→核心要素→叙事/评估目标）且可追溯的情境生成与验证框架，促进评估场景的透明与可比性。

**🔧 技术方法**

技术手段主要包括Claude Sonnet 4大模型生成文本、人工审查（三轮人机检查）以及自定义AI用例工作表与情境验证量表。

**📊 数据集**

数据来源为17位金融服务行业SME的访谈记录与他们提供的六个高层用例，未使用公开数据集，仅依赖专家知识。

**📈 对比分析**

通过自建情境验证量表对生成的107个场景进行质量评估，显示人机审查后场景完整度高，虽未对AI模型性能做直接对比，但证明了方法在场景生成上的可行性和一致性。

**⚠️ 局限性**

局限性包括：仅在金融服务行业验证，跨行业适用性待扩展；依赖单一风险框架（NIST 600‑1）可能限制风险覆盖；未对模型评估结果进行实测；以及缺乏对收益与风险完整分类的系统化方法。

---

## 822. GLiGuard: Schema-Conditioned Classification for LLM Safeguard

**arXiv ID:** 2605.07982 | [PDF](https://arxiv.org/pdf/2605.07982v1)

**作者:** Urchade Zaratiana `[一作]`, Ash Lewis `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于schema条件的双向编码器GLiGuard，用于在单一次前向传播中完成LLM提示和回复的安全性分类、拒绝检测、细粒度危害类型识别和越狱策略识别。

**💡 创新点**

创新点在于将任务定义与标签语义直接编码为输入序列的结构化token schema，使模型能在推理时动态组合多任务，而不需要多头或序列生成；同时通过双向编码器实现并行多任务分类，显著降低延迟与参数规模。

**🔧 技术方法**

使用GLiNER2的schema-conditioned编码方法，结合DeBERTa等双向Transformer编码器、共享MLP分类头以及多任务交叉熵/二元交叉熵损失；在推理时利用阈值与硬规则组合最终安全判定。

**📊 数据集**

训练数据主要来自WildGuardTrain并使用GPT‑4.1自动生成的危害类别与越狱标签；评估使用九个公开安全基准（OpenAI Moderation、Aegis 2.0、SimpST、HarmBench、WildGuardTest、SafeRLHF、BeaverTails、XSTest等）。

**📈 对比分析**

与七个基于7B–27B自回归解码器的守护模型（如LlamaGuard、WildGuard、ShieldGemma、NemoGuard、PolyGuard-Qwen、Qwen3Guard-Gen）相比，GLiGuard在F1得分上仅落后1.7点（提示）或1.4点（回复），但参数量小90×，吞吐量提升16×，延迟降低17×。

**⚠️ 局限性**

局限性包括：依赖自动生成的辅助标签可能导致噪声；对特定“Benign”触发词的敏感度不高；在极端角色扮演或复杂对话场景下的鲁棒性尚待进一步验证。

---

## 823. Self-Play Enhancement via Advantage-Weighted Refinement in Online Federated LLM Fine-Tuning with Real-Time Feedback

**arXiv ID:** 2605.07977 | [PDF](https://arxiv.org/pdf/2605.07977v1)

**作者:** Seohyun Lee `[一作]` (Purdue University), Christopher G. Brinton `[通讯]` (Purdue University)

**通讯引用:** 3155 | [OpenAlex ID](https://openalex.org/A5020399355)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种面向联邦学习的在线自我对弈增强算法SPEAR，用用户反馈驱动LLM的自我改进，避免了对离线数据和完整答案的依赖。

**💡 创新点**

创新点在于：①将生成的答案分为胜负两类，构造对比训练；②引入置信度门控的尾部无可能性（unlikelihood）损失，仅惩罚高置信度错误；③在联邦环境下采用基于胜利样本数的加权FedAvg，实现资源高效在线更新。

**🔧 技术方法**

核心技术包括：自我对弈交互框架、最大似然（MLE）与置信度门控无可能性联合损失、LoRA参数高效微调、FedAvg联邦聚合、对抗式奖励设计的简化。

**📊 数据集**

在四个多样化基准数据集上验证：ARC‑Challenge、HellaSwag、MathMCQA、StrategyQA，使用Qwen2.5（1.5B）和Llama3.2（3B）两大LLM。

**📈 对比分析**

与GRPO、OPSD、RLTF‑SD和反馈SFT等基线比较，SPEAR在所有数据集上均取得更高准确率（平均提升约5–10%）且训练时间更短，尤其在需要长链推理的任务上优势明显。

**⚠️ 局限性**

主要限制是依赖用户提供的“信息性”反馈；若反馈无关或错误，模型可能缺乏有效的胜利样本，导致学习停滞。

---

## 824. LLMs Improving LLMs: Agentic Discovery for Test-Time Scaling

**arXiv ID:** 2605.08083 | [PDF](https://arxiv.org/pdf/2605.08083v1)

**作者:** Tong Zheng `[一作]` (University of Maryland), Heng Huang `[通讯]` (University of Maryland)

**通讯引用:** 25158 | [OpenAlex ID](https://openalex.org/A5060016795)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AutoTTS 框架，通过构建可离线重放的环境，让 LLM 自动搜索并合成测试时缩放（TTS）控制策略，摆脱手工设计。

**💡 创新点**

创新点在于把 TTS 视作控制器合成问题，采用离线重放环境、beta 参数化（一维超参数映射）和执行轨迹反馈来实现高效自动发现；首次将自动化探索应用于 TTS 设计。

**🔧 技术方法**

技术包括：离线 Replay 环境（预收集多条推理轨迹）、beta 参数化控制器、LLM（Claude Code）代码生成与迭代、细粒度执行轨迹记录用于反馈。

**📊 数据集**

使用数学推理基准 AIME24、AIME25、HMMT25，以及 Qwen3 系列模型（0.6B–8B）进行实验；在 DeepSeek‑R1‑Distill‑Llama‑8B 上 HMMT25 与 GPQA‑Diamond（非数学）做迁移测试。

**📈 对比分析**

与手工设计的 SC@64、ASC、ESC、Parallel‑Probe 等基线比较。发现的策略在准确率‑token 曲线上更优，往往在相同或更少 token 下保持甚至提升准确率；在四个 Qwen 模型上大部分情况下超过基线。

**⚠️ 局限性**

局限性包括：需要预先收集大量轨迹，导致对特定任务或模型的依赖；beta 参数化可能限制策略表达的细粒度；目前仅在数学推理任务验证，跨任务和跨模型的通用性仍需进一步验证。

---

## 825. Zero-Shot Imagined Speech Decoding via Imagined-to-Listened MEG Mapping

**arXiv ID:** 2605.08075 | [PDF](https://arxiv.org/pdf/2605.08075v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 826. Proxy3D: Efficient 3D Representations for Vision-Language Models via Semantic Clustering and Alignment

**arXiv ID:** 2605.08064 | [PDF](https://arxiv.org/pdf/2605.08064v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 827. Chase-like Decoding: Test Pattern Design and Performance Analysis

**arXiv ID:** 2605.08081 | [PDF](https://arxiv.org/pdf/2605.08081v1)

**作者:** Tim Janz `[一作]` (University of Stuttgart), Stephan ten Brink `[通讯]` (University of Stuttgart)

**通讯引用:** 17475 | [OpenAlex ID](https://openalex.org/A5034116116)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文评估并设计了Chase‑like 解码中的测试模式集，提出三种评估方法并给出了高效的 MCOC 设计算法

**💡 创新点**

MCOC 算法在多种 BCH 码上实现了与贪心算法相当且可忽略的性能，且在高码率下仅需 0.2dB 的提升即可达到近 ML 性能

**🔧 技术方法**

利用顺序统计、覆盖空间概率和蒙特卡罗模拟等技术，结合 LR 模式与覆盖式模式构造

**📊 数据集**

使用典型的 BCH 码（如 (128,120) 扩展汉明码和 (256,239) 扩展 BCH 码）作为实验数据集

**📈 对比分析**

与 Chase‑II、受限 Chase、LR 模式等传统模式比较，MCOC 在 BLER/LER 上优于或持平，性能提升约 0.1~0.2dB

**⚠️ 局限性**

缺点包括对长码的搜索仍计算量大，蒙特卡罗方法在极低 BER 时耗时高，且尚未证明算法在所有场景下都是全局最优

---

## 828. EmambaIR: Efficient Visual State Space Model for Event-guided Image Reconstruction

**arXiv ID:** 2605.08073 | [PDF](https://arxiv.org/pdf/2605.08073v1)

**作者:** Wei Yu `[一作]` (Harbin Institute of Technology), Yunhang Qian `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种面向事件引导的图像重建框架 EmambaIR，用于运动去模糊、雨渍去除和高动态范围（HDR）增强。

**💡 创新点**

创新点在于结合了跨模态 Top‑k 稀疏注意力模块（TSAM）和门控状态空间模块（GSSM），实现了稀疏高效的空间交互和非线性门控的全局时序建模，同时保持线性 O(n) 复杂度。

**🔧 技术方法**

核心技术包括：跨模态 Top‑k 稀疏注意力、基于 Mamba 的门控状态空间网络、GeLU 非线性门控单元以及 UNet‑风格的多尺度编码解码结构。

**📊 数据集**

在三大任务上评估：GoPro（运动去模糊）、Adobe240（雨渍去除）和 SDSD（HDR 重建），使用对应的合成与真实事件流数据。

**📈 对比分析**

与多种基准（包括传统 CNN、ViT、Restormer、MPRNet 等）在 PSNR/SSIM 上进行对比，EmambaIR 在所有任务中均取得最高分，且参数量、FLOPs 与推理时间显著低于现有方法。

**⚠️ 局限性**

局限性：对超高分辨率图像仍需进一步优化，k 选择需经验调优；在极端噪声或极低曝光条件下性能尚未完全验证。

---

## 829. 123D: Unifying Multi-Modal Autonomous Driving Data at Scale

**arXiv ID:** 2605.08084 | [PDF](https://arxiv.org/pdf/2605.08084v1)

**作者:** Daniel Dauner `[一作]` (KE:SAI), Kashyap Chitta `[通讯]` (KE:SAI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了 123D 框架，将 8 大真实驾驶数据集与一套可配置的合成数据集合并为统一的多模态数据格式，并提供数据解析、同步、可视化与分析工具，支持跨数据集的感知与规划实验。

**💡 创新点**

创新点在于：① 采用事件流式日志与同步表实现异步、多频率传感器的统一访问；② 统一坐标、标注、地图等约定，支持保留原始标签并进行后期映射；③ 提供开源 Python API 与可视化工具，使研究者可快速构建跨域实验与数据挖掘；④ 在单一框架下完成多任务（3D 目标检测、强化学习规划）的跨域评估，展示数据混合对泛化的显著提升。

**🔧 技术方法**

技术栈包括 Apache Arrow IPC、同步表（sync table）、STR 树索引、Python、Viser、Mask2Former、FastGS、Kiss-ICP、PETR、BEVFormer‑S、PufferDrive、CARLA、LEAD 等；通过 LRU 缓存、可配置压缩编码实现高效访问。

**📊 数据集**

使用的数据集有：nuScenes、Waymo Open Dataset、Argoverse 2、WOD（Perception 与 Motion）、nuPlan、PandaSet、KITTI‑360、CARLA（生成的 L3AD 数据集）等，覆盖约 3300 小时与 90,000 公里。

**📈 对比分析**

比较方法包括：标注统计（距离、速度、加速度分布）评估标签质量；位姿与校准一致性测试（Kiss‑ICP 与原始位姿对比渲染质量）；跨域 3D 目标检测实验（PETR 与 BEVFormer‑S 在单域与混合训练下的 NDS 分数）；强化学习规划实验（PufferDrive 在不同数据集及混合训练下的成功率）。实验结果显示：单域模型跨域性能差距大，混合训练可显著缩小差距但仍存在显著的泛化瓶颈。

**⚠️ 局限性**

局限性：仅支持常见传感器（相机、LiDAR、车载状态、地图、交通灯），未覆盖雷达、语义点云等；不支持 Web 数据集或流式数据；目前仅处理标准车辆，缺少卡车、移动机器人等平台；未来需扩展传感器、流式处理与更广泛的车辆类型。

---

## 830. The Memory Curse: How Expanded Recall Erodes Cooperative Intent in LLM Agents

**arXiv ID:** 2605.08060 | [PDF](https://arxiv.org/pdf/2605.08060v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 831. Normalizing Trajectory Models

**arXiv ID:** 2605.08078 | [PDF](https://arxiv.org/pdf/2605.08078v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 832. GRAPHLCP: Structure-Aware Localized Conformal Prediction on Graphs

**arXiv ID:** 2605.08074 | [PDF](https://arxiv.org/pdf/2605.08074v1)

**作者:** Peyman Baghershahi `[一作]` (University of Illinois Chicago), Sourav Medya `[通讯]` (University of Illinois Chicago)

**通讯引用:** 490 | [OpenAlex ID](https://openalex.org/A5049055881)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种结构感知的局部化置信预测框架（ANH），旨在为图神经网络（GNN）生成满足边际覆盖率的可解释不确定性量化。

**💡 创新点**

创新点在于将个体节点的结构邻域通过个性化 PageRank（PPR）构建为可逆的软邻接核，并结合特征感知稠密化和 PCA 降维来提升稀疏图的局部性与全局性；从而实现对嵌入空间误差的补偿，显著提高条件覆盖率和预测集长度效率。

**🔧 技术方法**

核心技术包括：①图稠密化（利用高斯核与自适应阈值在嵌入空间中添加新边）；②PCA 降维与自适应带宽的多维高斯核；③使用 PPR 作为加权核进行锚点采样与加权量化；④加权分位数法与随机局部化 CP（RLCP）相结合，构建最终预测集。

**📊 数据集**

实验使用七个节点回归数据集（ANH、CHI、EDU、ELC、INC、UNM、TWT）和八个节点分类数据集（CMP、CRA、DBLP、CBAS、WKB、PMD 等），全部来源于公开图数据集。

**📈 对比分析**

与标准 Split CP、RLCP、CalLCP、NAPS、CF‑GNN、SNAPS 等基线相比，ANH 在边际覆盖率上保持 1‑α 并在大多数数据集上实现了更低的预测集长度；在 worst‑case 以及基于群组的条件覆盖评估中，ANH 获得了最佳或第二佳的 Worst‑Slab Coverage（WSC）并显著优于 CalLCP 的预测长度。

**⚠️ 局限性**

局限性包括：①对稠密化阈值和 PCA 维度选择仍需经验调参；②在极度异质或极度稀疏的图上，PPR 仍可能过度局部化；③计算成本相对传统 CP 较高，尤其在大规模图上的稠密化和 PPR 计算；④缺乏对动态图结构变化的理论与实验验证。

---

## 833. Conformal Path Reasoning: Trustworthy Knowledge Graph Question Answering via Path-Level Calibration

**arXiv ID:** 2605.08077 | [PDF](https://arxiv.org/pdf/2605.08077v1)

**作者:** Shuhang Lin `[一作]` (Rutgers University), Dimitris N. Metaxas `[通讯]` (Rutgers University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了Conformal Path Reasoning（CPR）框架，利用路径级一致性校准实现知识图谱问答的可信答案集合；

**💡 创新点**

创新点在于：①以查询为单位的路径级共形校准，恢复可交换性；②引入残差共形价值网络（RCVNet）并通过PUCT引导的轨迹收集学习路径非合规性分数；

**🔧 技术方法**

技术组合包括：共形预测（Split CP）、Monte Carlo Tree Search+PUCT、残差网络（RCVNet）以及TreeG树检索；

**📊 数据集**

在WebQSP和ComplexWebQuestions（CWQ）两个公开知识图谱问答数据集上进行实验；

**📈 对比分析**

与LLM共形基线（CLM、LoFreeCP）和KGQA共形基线（UaG）比较，CPR在所有风险水平下实现了更高的经验覆盖率且平均预测集大小更小，覆盖效率显著提升；

**⚠️ 局限性**

局限性包括：①依赖于高质量的知识图谱和查询采样；②RCVNet训练需要PUCT探索，训练成本相对较高；③对极端多跳或低频实体的推理仍面临挑战。

---

## 834. Flow-OPD: On-Policy Distillation for Flow Matching Models

**arXiv ID:** 2605.08063 | [PDF](https://arxiv.org/pdf/2605.08063v1)

**作者:** Zhen Fang `[一作]` (University of Science and Technology of China), Feng Zhao `[通讯]` (University of Science and Technology of China)

**通讯引用:** 38390 | [OpenAlex ID](https://openalex.org/A5070851446)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了Flow-OPD框架，通过在Flow Matching模型中引入On‑Policy Distillation实现多任务对齐。

**💡 创新点**

创新点在于将多教师密集轨迹监督与冷启动策略、Manifold Anchor Regularization（MAR）结合，解决了奖励稀疏与梯度干扰的问题。

**🔧 技术方法**

采用Flow Matching、GRPO、SDE+PPO、逆KL对齐、三步OPD采样+路由+监督以及MAR正则化等技术。

**📊 数据集**

在Stable Diffusion 3.5 Medium上使用GenEval、OCR、PickScore、DeQA四个评测数据集进行实验。

**📈 对比分析**

与单奖励GRPO及混合奖励GRPO对比，Flow-OPD在GenEval、OCR等指标提升约10分，同时保持图像质量与人类偏好一致。

**⚠️ 局限性**

局限性包括需多教师训练、训练成本高、对极端任务的泛化仍有限，以及冷启动对初始参数的依赖。

---

## 835. VecCISC: Improving Confidence-Informed Self-Consistency with Reasoning Trace Clustering and Candidate Answer Selection

**arXiv ID:** 2605.08070 | [PDF](https://arxiv.org/pdf/2605.08070v1)

**作者:** James Petullo `[一作]` (Brandeis University), Nianwen Xue `[通讯]` (Brandeis University)

**通讯引用:** 6806 | [OpenAlex ID](https://openalex.org/A5036715761)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种通过聚类推理轨迹嵌入来过滤冗余样本，从而在“think twice”框架下减少对评估器的调用，并保持或提升准确率。

**💡 创新点**

创新点在于将语义相似度聚类与代表性抽取结合到CISC流程，显著降低评估成本；同时提出 min‑centroid 代表选择和轻量级嵌入模型。

**🔧 技术方法**

使用文本嵌入（OpenAI text‑embedding‑3‑small）、KMeans/HAC 聚类、余弦相似度、softmax 归一化和加权多数表决。

**📊 数据集**

五大公开问答数据集：AQuA_RAT、CommonsenseQA、ARC‑Challenging、MMLU‑Pro、GPQA；五个 LLM 模型：GPT‑4o mini、Llama 3.1 8B、Llama 3.3 70B、Qwen 2.5 7B、Mistral 7B。

**📈 对比分析**

与 Self‑Consistency 与 CISC 对比；在所有模型/数据集上均能获得约 47% 总 token 节约、约 35% 评估器调用减少，并且准确率不低于 CISC，部分组合甚至略优。

**⚠️ 局限性**

局限性：依赖通用嵌入模型，K 与 T 超参数需手工搜索；聚类效果受嵌入质量与维度影响；在特定领域可能需要定制嵌入或更复杂的过滤策略。

---

## 836. Rubric-Grounded RL: Structured Judge Rewards for Generalizable Reasoning

**arXiv ID:** 2605.08061 | [PDF](https://arxiv.org/pdf/2605.08061v1)

**作者:** Manish Bhattarai `[一作]` (Los Alamos National Laboratory), Dan O'Malley `[通讯]` (Los Alamos National Laboratory)

**通讯引用:** 2603 | [OpenAlex ID](https://openalex.org/A5072498681)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用基于加权准则的 LLM 判定器，将奖励拆分为可验证的多维准则，并通过 Group Relative Policy Optimization (GRPO) 对 Llama‑3.1‑8B 进行强化学习，训练时不让模型访问判定器的文档上下文。

**💡 创新点**

创新点在于将传统的单标量奖励替换为结构化的、可拆解的 rubric‑grounded 奖励，使模型能够获得部分信用信号，同时通过 LLM 判定器将文档证据与准则结合，显著提升模型在多项评估中的表现。

**🔧 技术方法**

采用的技术包括 LLM‑as‑Judge、rubric‑grounded RL、GRPO、文档到 rubric 的自动生成流水线以及低温 LLM 判定器的高效并行推理。

**📊 数据集**

数据集为约10万条从 OSTI 公开的科学技术文档自动生成的问答‑rubric 对，分训练/验证/测试集；外部基准为 GSM8K、MATH、GPQA Main 与 GPQA Diamond。

**📈 对比分析**

通过与基线 Llama‑3.1‑8B 及其 SFT 版本对比，held‑out rubric reward 从 26.1% 提升至 71.7%，在四个公开基准上的平均准确率提升约 5%（从 77.7% 提升至 82.8%）。

**⚠️ 局限性**

局限性包括仅在 8B 规模、单轮回答和单一 OSTI 文档语料上实验，缺乏多轮、跨领域或多模型规模的验证；未做人类评估或跨判定器可靠性测试，可能存在奖励作弊风险；实验仅基于单一随机种子，缺少方差估计。

---

