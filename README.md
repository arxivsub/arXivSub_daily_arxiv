# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-06-23 | 今日论文总数: 845

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Holmes: Multimodal Agentic Diagnosis for Mixed-Language Mobile Crashes at Industrial Scale

**arXiv ID:** 2606.21963 | [PDF](https://arxiv.org/pdf/2606.21963v1)

**作者:** Jia Li `[一作]` (Chinese University of Hong Kong), Yuetang Deng `[通讯]` (Tencent Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Holmes多代理系统，利用堆栈、日志、线程状态以及低级寄存器和汇编信息，自动在不需要可重现环境的情况下对移动应用崩溃进行根因分析。

**💡 创新点**

创新点包括：多模态融合的层级Retrieve‑Explore‑Reason架构；轻量级函数‑文件索引实现对70M行代码的高效检索；低级信号（寄存器、汇编）桥接混合源语言的语义鸿沟；以及Post‑mortem不依赖重现环境的推理范式。

**🔧 技术方法**

技术方案涵盖LLM驱动的多代理协作、Map‑Reduce日志过滤与并行检索、动态压缩搜索空间、低级寄存器/汇编利用、注意力定位(anchor)与语义规则注入等。

**📊 数据集**

使用73条WeChat iOS崩溃报告（分层抽样）和39,795条生产运行数据进行性能评估。

**📈 对比分析**

与基线（如DeepSeek‑V3.1、传统堆栈+日志）比较，Holmes在功能级定位Pass@1为87.6%，根因识别65.7%，平均诊断时延约77秒（对比手工2‑3小时），成本仅约0.13美元，效率提升>98%，准确率显著优于基线。

**⚠️ 局限性**

局限性：隐私删减导致信息熵降低，可能误诊；并发与非确定性内存错误仍易错；跨语言语义桥接不完备；稀有根因类别准确率偏低；LLM推理仍有不确定性，需进一步稳定与可解释化。

---

## 2. ACEsplat: Accelerated 3D Gaussian Scene Regression via RGB and Poses Only

**arXiv ID:** 2606.22091 | [PDF](https://arxiv.org/pdf/2606.22091v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 3. Human vs Machine Mathematical Difficulty on Project Euler: An Experimental Analysis

**arXiv ID:** 2606.21972 | [PDF](https://arxiv.org/pdf/2606.21972v1)

**作者:** David Holmes `[一作]`, Johannes Schmitt `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文利用MathArena的Project Euler AI实验数据，评估前沿大语言模型在数学问题上的搜索效率与成功概率的规模关系。

**💡 创新点**

首次将Gowers提出的搜索时间幂律假设与单参数指数成功概率模型应用于真实AI实验，并提出基于LOGISTIC的METR-风格能力边界。

**🔧 技术方法**

采用生成令牌计数作为模型努力度量，利用普通最小二乘、最大似然和Bootstrap不确定性估计等统计方法进行模型拟合。

**📊 数据集**

使用MathArena发布的Project Euler 943–992 的3,840条AI尝试和26种模型配置的数据集，并以最快五人类求解时间作为难度基准。

**📈 对比分析**

对比结果显示，绝大多数模型的生成令牌幂指数b<1，表明机器在难度增加时的相对效率高于人类；成功概率随人类耗时呈指数衰减，且METR-风格的h_50边界在几小时级别，说明可靠性是主要瓶颈。

**⚠️ 局限性**

主要限制包括人类基准的左尾采样假设、模型生成令牌计数的跨供应商可比性不足、评估过程不对称以及Project Euler问题对长期推理的代表性不足。

---

## 4. NL2Scratch: An Executable Benchmark and Evaluation for Block-Based Programming

**arXiv ID:** 2606.22061 | [PDF](https://arxiv.org/pdf/2606.22061v1)

**作者:** Heejin Do `[一作]` (ETH Zurich), April Yi Wang `[通讯]` (ETH Zurich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 NL2Scratch 基准，实现从自然语言生成可执行 Scratch 程序。

**💡 创新点**

创新点在于构建可执行 NL–Scratch 对、引入语义一致性验证（SAC）进行槽级评估，以及创建 800 条诊断样本。

**🔧 技术方法**

使用 Scratchblocks 规范、LLM 双阶段描述生成、SAC 语义验证和 LLM 微调/提示重排序等技术。

**📊 数据集**

基于 311,648 条真实 Scratch 项目的程序和 23,594 条 SAC 验证后的高置信样本。

**📈 对比分析**

对比 GPT‑4、GPT‑4o‑mini、FLAN‑T5、Qwen2.5‑7B‑Instruct、Llama‑3.1‑8B‑Instruct 等模型，发现 F1、解析成功率高，但 SAC 语义对齐率仅 18–20%，提示语义验证的重要性。

**⚠️ 局限性**

局限在于描述语言真实性未完整评估、SAC 不是完整行为等价检查、仅针对 Scratch，未验证其他块编程平台。

---

## 5. StickyInvoc: Rethinking Task Models for High-throughput Workflows in the LLM Era

**arXiv ID:** 2606.22175 | [PDF](https://arxiv.org/pdf/2606.22175v1)

**作者:** Thanh Son Phung `[一作]` (University of Notre Dame), Douglas Thain `[通讯]` (University of Notre Dame)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 StickyInvoc，将传统任务模型拆分为 sticky 任务（创建并持久化 LLM 状态）与 invocation 任务（继承该状态执行推理），从而在高吞吐 HPC 工作流中显著降低冷启动成本。

**💡 创新点**

创新点在于通过双任务模型实现状态创建与计算解耦，利用 GPU 内存中持久化的 LLM 状态，支持预抢占资源上的无中断推理与状态迁移。

**🔧 技术方法**

采用 Parsl‑TaskVine 框架实现 sticky 与 invocation 任务，内部使用 StateManager 维护状态、peer transfer 复制状态模板，并结合 GPU 内存管理与动态资源池调度。

**📊 数据集**

使用 FEVER 数据集（145,449 条主张）和 SmolLM2 1.7B 参数的轻量 LLM 进行命题验证工作流。

**📈 对比分析**

与传统 create‑destroy 和 StickyI/O 三种实现对比，StickyInvoc 在 20 GPU 稳定测试床完成 150k 推理任务仅需 2.9k 秒（相较 10.4k 秒 3.6 倍加速），在 186 GPU 上完成仅 784 秒；实验还展示了对 batch size、预抢占和资源动态变化的鲁棒性。

**⚠️ 局限性**

局限性：仅适用于单节点 GPU 能装载的轻量 LLM（数十亿参数以内）；需要额外的状态管理开销；节点失效时需重新迁移状态，可能影响性能。

---

## 6. CoDMD: Copula-aware Distribution Matching Distillation for Fast Video Generation

**arXiv ID:** 2606.21982 | [PDF](https://arxiv.org/pdf/2606.21982v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 7. ISCSLP 2026 CoT-TTS Challenge: Chain-of-Thought Reasoning for Context-Aware Text-to-Speech

**arXiv ID:** 2606.21933 | [PDF](https://arxiv.org/pdf/2606.21933v1)

**作者:** Wei Xue `[一作]` (Hong Kong University of Science and Technology), Bin Long `[通讯]` (Generative AI Research & Development Center)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出并实施了ISCSLP 2026 CoT‑TTS 挑战，目标是让系统在仅给定上下文（文本或音频）和目标句子以及参考语音的前提下，自动推理并输出该句子的说话方式（链式推理），随后合成符合该推理且保持参考声纹的语音；

**💡 创新点**

创新点包括：① 将链式推理（CoT）作为任务输出，提升模型可解释性；② 双轨挑战（文本上下文与音频上下文）以及可限制参数规模的评测；③ 构建 16k 小时中英双语大规模带情感与推理标签的对话数据集；④ 统一客观、LLM 与人类评测三段式评分体系；

**🔧 技术方法**

技术方案基于 0.6B Qwen3 语言模型，采用 BiCodec 语音编码、UniSS 三阶段训练（跨模态对齐 → 语境 CoT‑TTS 任务 → 高质量子集微调），并辅以多模态 LLM 评估器和基于 DNSMOS/UTMOS 的客观评测；

**📊 数据集**

使用的数据集为约 16k 小时的中英语料（~3M 段），来源于电影、电视剧、广播等；数据包含多轮对话、情感标签（A、D、V）与基于 LLM 生成的链式推理注释；

**📈 对比分析**

系统通过客观 TTS 指标（UTMOSv2、DNSMOS P.835、WER、语音相似度）、LLM 推理一致性评分和 5 人 Crowd‑source 主观评分进行综合评测；基线模型在所有三个评测维度均表现可行，能够生成含有明确推理且与参考声纹相符的语音；

**⚠️ 局限性**

局限性包括：① 推理质量受限于 LLM 生成能力，易出现信息不足或泛化不足；② 仅支持单句目标文本，难以处理长篇连续对话；③ 评测样本数量有限（600 句中英），未涵盖极端语境；④ 大规模模型训练与推理对算力要求高，参数约束模式仍挑战性能；

---

## 8. Durability-Aware Multi-Objective Optimization of the Jansen Linkage: Trading Gait Quality Against Joint Wear

**arXiv ID:** 2606.22129 | [PDF](https://arxiv.org/pdf/2606.22129v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 9. A Post-Quantum Secure Lattice-Based Forward-Secure Identity Based Encryption with Applications to Internet of Things Architecture

**arXiv ID:** 2606.22340 | [PDF](https://arxiv.org/pdf/2606.22340v1)

**作者:** Abhishek Kumar `[一作]` (National Institute of Technology Jamshedpur), Pantelimon Stănică `[通讯]` (Naval Postgraduate School)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种基于环学习错误（RLWE）问题的前向安全身份基加密（FS-IBE）方案，用于物联网（IoT）环境，结合二叉树的最小覆盖机制和陷门委托，显著压缩公钥、私钥与密文大小，并实现后量子安全。

**💡 创新点**

创新点包括：①首个将 RLWE 与前向安全身份基加密结合的方案；②在环晶格框架下引入二叉树最小覆盖与陷门委托，减少每个身份每个时间段需要维护的陷门数量；③通过双重 Regev 减弱和随机预言机模型证明了选择密文攻击下的安全性，并提出了可扩展至 CCA 安全的思路。

**🔧 技术方法**

核心技术包括：环晶格（RLWE）基础、陷门生成与委托（ringGenTrap、ringDelTrap）、二叉树最小覆盖（MinCov）、离散高斯采样（ringGenSamplePre）、随机预言机哈希函数（H、G）以及基于 RLWE 的公共密钥加密与安全性归约。

**📊 数据集**

论文未使用公开数据集，而是在 Python 3.10 + NumPy 环境下自行实现核心算法，利用合成的多项式和随机样本进行性能基准测试。

**📈 对比分析**

通过与 Jin 等人 2024 年提出的 LWE‑based FS‑IBE 方案对比，实验显示：KeyGen、Encrypt、Decrypt 的执行时间分别从 2.74/2.08/1.86 ms 降至 1.82/1.35/1.11 ms；内存占用从 58.3/49.7/44.2 KB 降至 42.6/37.4/31.8 KB，表明本方案在速度与资源占用上均优于前人方案。

**⚠️ 局限性**

局限性包括：①目前仅实现了选择密文攻击（CPA）安全，尚未达到自适应 CCA 安全；②实现依赖于环晶格的高维多项式运算，深度树或大规模身份集合时仍可能产生较高的计算和存储负担；③缺乏在真实物联网硬件上的实验验证，未来需进一步评估在嵌入式设备上的可部署性。

---

## 10. CoSA: Correlation-Guided Change Attention with Learnable Residual Gating for Remote Sensing Change Detection

**arXiv ID:** 2606.21932 | [PDF](https://arxiv.org/pdf/2606.21932v1)

**作者:** Abdirashid Omar `[一作]` (Kookmin University), Jonghyuk Park `[通讯]` (Kookmin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种轻量级的解码器侧细化模块 Context Sampling Attention (CoSA)，通过点位归一化的同位置交叉相关生成变更门，并在多尺度解码阶段注入门控残差，以提升遥感变化检测的时间判别能力。

**💡 创新点**

创新点在于将时间特征相关性直接作为显式控制信号，利用可学习的残差缩放实现门控残差注入，并在多尺度解码位置保持极低的参数和计算开销；实现了可插拔、几乎不增加模型大小的细化方案。

**🔧 技术方法**

采用共享Siamese编码器‑解码器结构、绝对差分特征、点位归一化交叉相关、1×1卷积生成变更门、可学习残差标量以及多尺度门控残差注入技术。

**📊 数据集**

使用四个公开遥感变化检测基准：LEVIR-CD、S2Looking、DSIFN 和 CLCD。

**📈 对比分析**

与基线 FC‑Siam、STANet、BIT 等模型对比，CoSA 在四个基准上分别提升 1.5–2.6% F1、1.8–2.8% IoU，参数量仅 +66，计算成本几乎不变，表现稳定一致。

**⚠️ 局限性**

局限性包括：对已具备强时间交互的模型（如 BIT）提升有限；在噪声边界或纹理弱区可能出现误判；未在多硬件平台进行广泛的推理效率评估。

---

## 11. Apple Neural Engine: Architecture, Programming, and Performance

**arXiv ID:** 2606.22283 | [PDF](https://arxiv.org/pdf/2606.22283v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 12. How Should a Simulation-to-Reality Transfer Budget Be Spent?

**arXiv ID:** 2606.22062 | [PDF](https://arxiv.org/pdf/2606.22062v1)

**作者:** Syed Hamzah Rizvi `[一作]` (Purdue University), Yash Vardhan Tomar `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在可辨识的摆杆系统中，研究了将有限的真实机器人测量预算在系统辨识与域随机化之间分配的影响。

**💡 创新点**

提出并验证了一个基于预算分配的评估框架，发现测量（系统辨识）比扩大随机化更能显著缩小模拟与真实之间的差距。

**🔧 技术方法**

使用软演员-评论家（SAC）算法进行策略训练，采用网格搜索对物理参数进行系统辨识，并通过随机力激励收集测量数据。

**📊 数据集**

实验仅使用两台摆杆仿真器生成数据，未涉及公开数据集。

**📈 对比分析**

通过对不同(n,w)组合的零射返回曲面进行比较，结果显示：一旦获得任何真实数据，最优宽度为0（仅使用估计参数），而扩大随机化宽度则进一步降低性能；测量预算的增加对返回值的提升效果最大。

**⚠️ 局限性**

局限性包括：实验仅在可辨识且无结构误差的理想摆杆模型上进行；未考虑摩擦、延迟等未建模的结构误差；随机化仅围绕估计值进行，未测试自适应随机化方法。

---

## 13. BabelJudge: Measuring LLM-as-a-Judge Reliability Across Languages and Agent Trajectories

**arXiv ID:** 2606.22329 | [PDF](https://arxiv.org/pdf/2606.22329v1)

**作者:** Shreyas KC `[一作]` `[通讯]`, Shreyas KC

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了BabelJudge，一种无需人工标注即可评估LLM判断器在多语言和agentic场景下的偏差与可靠性的开源基准。

**💡 创新点**

创新点包括：①基于降解自动生成带真值的对比样本（gold‑labelling‑by‑degradation）；②五种通用扰动（信息丢失、句子重排、数值扰动、实体删除、冗长追加）用于探测位置、冗长、顺序一致性等偏差；③多维度乘法惩罚的可靠度得分；④针对agentic轨迹的九种扰动和三项新度量；⑤提供Python包和多judge后端适配，构建可复现的评测管道。

**🔧 技术方法**

采用的技术包括：对参考回答做语法/数字/冗余等扰动；双顺序呈现评判（r,p 与 p,r）以量化位置偏差和顺序一致性；计算准确率、位置偏差、冗长敏感度、顺序一致性等指标；将这些指标乘性整合为可靠度得分；对轨迹级扰动进行工具调用、参数、幻觉检测和长度偏差评估。

**📊 数据集**

使用Aya多语料库（65+语言）中的10条参考回复，实验覆盖英语、印地语、阿拉伯语和斯瓦希里语，总计200条对比样本，未使用任何人工标注。

**📈 对比分析**

方法：对四种语言分别计算raw accuracy与可靠度得分；结果显示Qwen2.5‑7B‑4bit在英语、印地语、阿拉伯语的可靠度分别为0.70、0.714、0.695，斯瓦希里的可靠度仅为0.550；raw accuracy为0.815、0.835、0.770、0.660，表明准确率高估了实际可靠性。

**⚠️ 局限性**

局限性：样本量仅10条/语言，结果不够稳健；仅覆盖四种语言，未能验证在更广泛语种下的表现；扰动保守，缺少细腻的事实错误或风格偏差；仅评估单一judge模型，缺乏多judge基准；agentic实验使用手工合成轨迹，未覆盖真实工具调用数据集。

---

## 14. A Continuous Multi-Component Measure of Directed Acyclicity (DAG-ness)

**arXiv ID:** 2606.22205 | [PDF](https://arxiv.org/pdf/2606.22205v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 15. Physics-Informed Eikonal Caging for Whole-Arm Manipulation Planning

**arXiv ID:** 2606.22143 | [PDF](https://arxiv.org/pdf/2606.22143v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 16. Shared-Context Batched Satisfiability

**arXiv ID:** 2606.21983 | [PDF](https://arxiv.org/pdf/2606.21983v1)

**作者:** Jiening Siow `[一作]` (Zhejiang University), Peisen Yao `[通讯]` (Zhejiang University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了共享上下文批量可满足性（Shared‑Context Batched Satisfiability, SCBS）问题，并提出了一种基于禁用字面量的 Core‑Literal Filter (CLF) 算法，同时与线性扫描和析取近似两种经典策略进行系统比较。

**💡 创新点**

创新点包括①首次将 SCBS 形式化为通用问题；②设计了利用不满足字面量进行前置过滤的 CLF 算法；③在符号抽象和主动属性检查两个真实工作负载上，全面评估并揭示不同优化组合的优劣。

**🔧 技术方法**

使用技术主要有：递增 SMT 求解、模型复用、析取过近似、禁用字面量提取与缓存，以及对不同求解器策略的组合实验。

**📊 数据集**

实验数据集包含：十个真实程序的符号抽象查询（共 3,400 条，1,656 条成功）和 27 个 C/W 或真实程序的主动属性检查查询（共 9,200 条，9,153 条成功）。

**📈 对比分析**

通过平均运行时、求解器调用次数、超时率等指标进行横向比较。结果显示：在主动属性检查中 CLF 取得最快平均时长（约 26.5 ms），而在符号抽象中 OA‑Inc 在已求解查询上最快（约 11.6 s），CLF 在符号抽象中完成率最高（82.5 % vs 75.2 %）。

**⚠️ 局限性**

局限性包括：CLF 的禁用字面量预算在大规模谓词集时会产生额外求解调用，导致性能下降；整体方法对 SAT 比率与谓词数量高度依赖，缺乏自适应选择策略，且理论上最优复杂度仍未完全确定。

---

## 17. A Multimodal Tiltwing Framework for Bioinspired Aerial Robots

**arXiv ID:** 2606.22046 | [PDF](https://arxiv.org/pdf/2606.22046v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 18. From RAN Control to Agentic Intelligence: Architecture and Vision for Energy Efficient AI-RAN

**arXiv ID:** 2606.21955 | [PDF](https://arxiv.org/pdf/2606.21955v1)

**作者:** Sabrine Aroua `[一作]` (Ericsson Research), Navid Nikaein `[通讯]` (BubbleRAN)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出 E-ARC 框架，整合 O‑RAN 与 AI‑RAN，通过 LLM 驱动的语义协同代理，利用数字孪生离线验证，闭环监控实现能源感知、多目标的自适应 RAN 编排。

**💡 创新点**

创新点在于：① 将大型语言模型嵌入 RAN 控制层，完成高层意图语义解析与冲突解决；② 通过数字孪生提前评估 rApp 配置风险；③ 将 AI‑for‑RAN 与 AI‑on‑RAN 统一在同一意图驱动框架下，实现通信、计算与能源协同优化。

**🔧 技术方法**

核心技术包括：大语言模型（GPT‑4/LLM）、数字孪生模拟器、RAN rApp（如预测性扇区调度、UX‑aware DRL）、O‑RAN SMO/Non‑RT RIC、意图驱动网络、AI‑on‑RAN 边缘推理。

**📊 数据集**

使用数据集：1）一月运营流量轨迹（扇区调度评估）；2）交通流量及资源利用的 trace‑driven 仿真数据；3）AI 应用层面指标（延迟、能耗）的 enrichment 数据集。

**📈 对比分析**

比较方法：将 E‑ARC 在能耗、覆盖率、吞吐量、QoE 等 KPI 与基线（无能耗优化）对比；实验结果显示在 15‑20% 的能耗下降同时，覆盖率>99%，吞吐量/QoE 满足率>95%。意图‑编排准确率在云 LLM 上达到 85‑95%，本地 SLM 依赖模板与验证机制。

**⚠️ 局限性**

局限性：① 对强大 LLM 的依赖导致成本与隐私挑战；② 小型本地 SLM 的准确率低，需要额外模板/验证；③ 数字孪生的计算开销与实时性受限；④ 真实场景验证尚缺乏，需进一步评估跨域协调的可扩展性。

---

## 19. Analyzing the Analyzers: Model Counting Meets Abstract Interpretation

**arXiv ID:** 2606.21992 | [PDF](https://arxiv.org/pdf/2606.21992v1)

**作者:** Junda Zheng `[一作]` (Zhejiang University), Peisen Yao `[通讯]` (Zhejiang University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MCAI（Model Counting meets Abstract Interpretation）方法，利用模型计数量化抽象域对具体语义的逼近精度，并系统评估四种数值抽象域（Interval、Zone、Octagon、KnownBit）的精度与性能；

**💡 创新点**

创新点在于：①引入模型计数作为客观、无客户端依赖的精度度量；②通过逻辑公式对具体语义与抽象值进行统一表示；③通过抽象与模型计数比较，得到精度差距和域间差异；④揭示许多传统观念（如Octagon更精准、约束冗余等）的实际情况；

**🔧 技术方法**

主要技术包括：符号抽象（Symbolic Abstraction）求解最优抽象变换；逻辑公式的bit-blast与投影模型计数；基于SMT/OMT求解器（如Z3）对数值域进行符号化和计数；

**📊 数据集**

使用了来自SV-COMP Reach Safety Track的2,006条二进制向量公式，来源于CBMC和KINT两款程序分析器；

**📈 对比分析**

通过对每个公式分别计算抽象域的最优抽象后，使用模型计数得到抽象的模型数与具体模型数，计算误报率与精度；对不同域进行对比，绘制精度散点图、误报率分布图；实验显示Interval与Zone、Octagon精度相近，KnownBit显著更高；性能方面，Interval最轻，Octagon平均耗时约68s，KnownBit仅约7s；

**⚠️ 局限性**

局限性包括：①仅评估最优抽象而非实际实现的转移函数；②模型计数在大规模公式上仍可能成为瓶颈；③只关注固定宽度bit-vector域，无法直接推广到无限域；④未考虑分析过程中迭代收敛时的中间误差与冗余约束；

---

## 20. FeLoG: Scalable and Efficient Distributed Graph Embedding with Feedback Loop Mechanism

**arXiv ID:** 2606.22180 | [PDF](https://arxiv.org/pdf/2606.22180v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 21. Freeze-Tag with Return

**arXiv ID:** 2606.21985 | [PDF](https://arxiv.org/pdf/2606.21985v1)

**作者:** Nicolas Bonichon `[一作]` (University of Bordeaux), Nils Morawietz `[通讯]` (Friedrich Schiller University Jena)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了一种机器人唤醒与返回的 Freeze‑Tag‑with‑Return 问题，并给出了其在欧几里得平面与图空间中的最优时间界限。

**💡 创新点**

创新点包括：证明返回成本上限不超过1.959，给出凸配置下最优时间为 2+2√2，提出单指数时间算法并在无权图中证明 NP‑难与 ETH 限界。

**🔧 技术方法**

使用了几何分析、动态规划、递归分解、近似算法与复杂度下界构造技术。

**📊 数据集**

主要使用了单位圆分布、凸几何、均匀圆环等合成实例作为数据集；在图空间中使用了平衡二叉树和子图 gadget。

**📈 对比分析**

通过与已知 FTP 结果对比，证明最优 makespan 与 FTP 的差值始终在 1.732–1.959 之间；算法在最坏情况下得到 4.89+O(1/√n) 的上界，凸配置得到 4.83 的上界。

**⚠️ 局限性**

局限性包括：对大规模实例的上界仍有 0.26 的差距，且对欧几里得平面中 NP‑难性的证明尚未完成，算法复杂度仍为 3^n 或 9^n。

---

## 22. Selective Ensemble Based on Preference-Directed Multi-Objective Bandits

**arXiv ID:** 2606.21929 | [PDF](https://arxiv.org/pdf/2606.21929v1)

**作者:** Lanjihong Ma `[一作]` (Zhejiang Gongshang University), Zhi-Hua Zhou `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

该论文提出了基于偏好导向的多目标 bandit 框架，用于在有限评估预算下挑选机器学习模型，并在偏好约束下进行序列决策。

**💡 创新点**

创新点是引入 Pareto C‑最优性统一 Pareto 最优和单权重标量化，提出 PrefUCB 算法以及对指标型和间隙加权 regret 的实例相关对数界。

**🔧 技术方法**

技术包括多目标 bandit、聚合多维偏好锥、方向性置信区间、Upper Confidence Bound（UCB）方法、对数 regret 分析。

**📊 数据集**

数据集包括大型预训练模型的选择性集成任务数据以及满足机构规定的在线资产配置数据。

**📈 对比分析**

通过与传统单目标 UCB、Pareto UCB 及标量化 UCB 的基线对比，实验显示 PrefUCB 在指标型和间隙加权 regret 上均实现了更低的累积损失。

**⚠️ 局限性**

局限性在于上界对偏好方向数 M 线性依赖，未能充分利用偏好锥的有效维度，且缺乏对应的下界证明。

---

## 23. RoboLineage: Agent-Native Data Lifecycle Governance Across Robot Policy Iterations

**arXiv ID:** 2606.22142 | [PDF](https://arxiv.org/pdf/2606.22142v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 24. MultiMem: Measuring and Mitigating Memorization in Multi-Modal Contrastive Learninga

**arXiv ID:** 2606.22220 | [PDF](https://arxiv.org/pdf/2606.22220v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 25. Benchmarking Large Language Models for Grapheme-to-Phoneme Conversion: A Japanese Case Study

**arXiv ID:** 2606.22009 | [PDF](https://arxiv.org/pdf/2606.22009v1)

**作者:** Tomoki Koriyama `[一作]` `[通讯]` (CyberAgent), Tomoki Koriyama (CyberAgent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对30多款大型语言模型（LLM）在日语图像-语音（G2P）任务上进行系统评测，并与传统形态学分析器对比。

**💡 创新点**

创新点在于：①首次大规模基准化日语LLM G2P性能；②提出两种LLM推理模式（Parse与Direct）并证明Parse模式更优；③展示LLM G2P+基于假名输入的TTS可优于端到端TTS，保持自然度。

**🔧 技术方法**

主要技术包括LLM推理、两种提示策略（Parse/Direct）、规则化后处理（音节转换、长音归一化）、与假名输入TTS（CosyVoice 2）结合。

**📊 数据集**

使用JVS非正式语料库（nonpara30）3000句手工标注假名作为参考，覆盖拟声词、外来语等多样现象。

**📈 对比分析**

评估方法为假名字符错误率（CER）及通过ASR计算的发音准确率，结果显示最佳API模型Parse模式CER<0.6%，优于传统分析器（OpenJTalk 1.03%），且G2P+TTS的CER≈2.4%远低于E2E TTS。

**⚠️ 局限性**

局限性包括：①数据集规模有限，未深入评估专有名词错误；②LLM推理仍受提示设计限制；③规则化后处理仍需人工编写，难以覆盖所有语言现象；④对更大模型的API成本与可用性考量。

---

## 26. CapRiCorn-1K: A Comprehensive Benchmark for Video Captioning and Subject Referential Consistency Across Temporal Scales

**arXiv ID:** 2606.21949 | [PDF](https://arxiv.org/pdf/2606.21949v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 27. When Cooperation Should End: Maneuver Coordination Cancellation for Connected Automated Driving

**arXiv ID:** 2606.22052 | [PDF](https://arxiv.org/pdf/2606.22052v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 28. SAGE: An Expert-Annotated South Asian GI Endoscopy Dataset for Multimodal Learning and Hallucination Analysis

**arXiv ID:** 2606.22144 | [PDF](https://arxiv.org/pdf/2606.22144v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 29. DeformX: A Versatile Co-Simulation Framework for Deformable Linear Objects

**arXiv ID:** 2606.22116 | [PDF](https://arxiv.org/pdf/2606.22116v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 30. OphthaDT: Generative Digital Twins for Forecasting Visual Acuity Trajectories in Ophthalmology

**arXiv ID:** 2606.22101 | [PDF](https://arxiv.org/pdf/2606.22101v1)

**作者:** Pietro Belligoli `[一作]` (Roche), Michael Menden `[通讯]` (Helmholtz Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 OphthaDT，一种基于 LLM 的数字孪生模型，用于预测视网膜疾病（nAMD 与 DME）的最佳矫正视力（BCVA）随访轨迹。

**💡 创新点**

创新点在于将多模态、碎片化的临床记录序列化为结构化文本提示，使 LLM 能够在不进行插值、能自然处理缺失和不规则采样的前提下，进行多时点预测，并验证了该方法在非肿瘤领域的可行性。

**🔧 技术方法**

使用 MedGemma 4B LLM 进行指令微调，结合文本序列化、结构化提示、采样多次生成并取均值的预测策略，并与传统线性模型、随机森林和 XGBoost 进行比较。

**📊 数据集**

数据来源为 3,220 名患者的四个 III 期临床试验，包含 nAMD 与 DME 的随访记录、BCVA、影像生物标志物、治疗史等多模态信息。

**📈 对比分析**

在四个临床里程碑（第 8、24、52、100 周）评估 MAE 与 R^2，结果显示在 nAMD 中 OphthaDT 的 MAE 低于所有基线 5–10%，在 DME 中与线性模型相当、优于 RF 和 XGBoost。

**⚠️ 局限性**

局限在于仅使用了有限的临床变量，未整合 OCT 影像；样本量相对有限；对复杂轨迹的优势在 DME 中表现有限，需进一步扩大数据与验证。

---

## 31. When Is Emergent Consensus Real? A Measured Coupling Gain and a Validity Diagnostic for LLM Agent Societies

**arXiv ID:** 2606.22203 | [PDF](https://arxiv.org/pdf/2606.22203v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 32. RARM: Confidence-Gated Progress Reward Modeling for RL in Manipulation

**arXiv ID:** 2606.22027 | [PDF](https://arxiv.org/pdf/2606.22027v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 33. Gradient-Descent Steps to Success over Mean Accuracy: A Paradigm Shift for ML

**arXiv ID:** 2606.22053 | [PDF](https://arxiv.org/pdf/2606.22053v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 34. Neural Conjugate Aggregation: Identifiable Unsupervised Multi-Sensor Regression under Heterogeneous Sensor Bias

**arXiv ID:** 2606.22200 | [PDF](https://arxiv.org/pdf/2606.22200v1)

**作者:** Muhammed Faruk Aytin `[一作]` (Istanbul Technical University), Gözde Ünal `[通讯]` (Istanbul Technical University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在无监督条件下，利用神经网络与共轭高斯推断相结合的层级贝叶斯框架NCAM，对多源噪声与偏差传感器数据进行融合，推断潜在的连续目标变量并提供不确定性估计。

**💡 创新点**

引入了传感器偏差与可靠性可根据上下文学习的神经网络、通过传感器锚定解决结构非可识别性、以及无监督的Monte Carlo自适应 conformal 校准，三者协同实现了无标签环境下的精确融合与校准。

**🔧 技术方法**

采用神经网络参数化的先验与测量模型，利用共轭高斯推断得到解析后验，加入方差正则化与传感器锚定；随后通过MC‑CP（模型、经验、传感器锚定）进行不确定性校准。

**📊 数据集**

在三组数据集上验证：合成Toy网格、跨城市的SensEURCity PM₂.₅低成本传感器数据、以及同地点的CAIRSENSE AirAssure 传感器网络。

**📈 对比分析**

与无监督基线（均值聚合、逆方差加权、概率PCA、Kalman滤波、VELI）以及理想的有标签Split Conformal 进行对比；NCAM在大多数数据集上实现了最低RMSE/MAE，MC‑CP 校准实现了接近目标覆盖率且区间宽度更紧。

**⚠️ 局限性**

对锚定传感器的依赖导致在传感器同质性低时性能易受影响；在低偏差、低噪声环境下可能过拟合；模型假设Gaussian噪声且仅针对单变量目标；在强时间相关场景需采用块拆分近似，可导致覆盖率偏差。

---

## 35. Topological summaries of fingerprint ridge patterns carry identity information

**arXiv ID:** 2606.22029 | [PDF](https://arxiv.org/pdf/2606.22029v1)

**作者:** Chad M. Topaz `[一作]`, Lori Ziegelmeier `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了使用拓扑数据分析（持久同调）对指纹纹路进行全局与局部特征提取，并在不使用 minutiae、骨架化或对齐的前提下进行指纹验证。

**💡 创新点**

创新点在于：①首次将持久同调直接应用于指纹的距离变换场；②设计两类全局（Betti 曲线、持久图像）与局部（基于局部持久图像的最优传输）验证方法；③通过轻量级学习与融合实现高精度验证，证明拓扑摘要比原始像素几何更具辨识力。

**🔧 技术方法**

使用的技术包括：距离变换、子水平集持久同调、Betti 曲线与持久图像向量化、逻辑回归学习（TopoLR）、局部持久图像+最优传输（LPOT）以及鲁棒 z‑score 线性融合。

**📊 数据集**

实验数据集为 FVC2000 DB1（100 个身份，8 次扫描，每次 300×300 像素）。

**📈 对比分析**

对 7 种方法（几何基线、全局拓扑基线、TopoLR、LPOT 及其融合）在 5 折身份分离交叉验证中进行 ROC、EER、TAR@FAR 比较；全局拓扑基线 AUC ≈0.847，TopoLR AUC ≈0.906，LPOT 在极低 FAR（10⁻³）下 TAR 最高（≈11.2%），融合后 TAR 在所有低 FAR 点均提升，AUC 与 TopoLR 相近。

**⚠️ 局限性**

局限性包括：①极低 FAR 评估方差大；②未进行全面超参数调优；③仅在单一低成本光学传感器的 FVC2000 数据上验证，未检验跨传感器泛化；④几何基线未对齐，导致对比时 invariance 的影响；⑤与成熟 minutiae 系统相比仍有较高 EER。

---

## 36. What Changes When the Interlocutor Is an AI? Interactional Fluency and Linguistic Uptake in L2 Spoken Dialogue

**arXiv ID:** 2606.22225 | [PDF](https://arxiv.org/pdf/2606.22225v1)

**作者:** Russell Scheinberg `[一作]` (Portland State University), Griet Boone `[通讯]` (University of Antwerp)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对78名德语学习者进行实验，比较其在与真人伙伴和AI伙伴进行Spot-the-Difference对话时的交互流畅度、语言吸收和学习者态度。

**💡 创新点**

①将交互流畅度和语言吸收的量化指标从自动转录（ASR+人声分离）中提取，避免人工编码；②在多站点、对照设计中探究AI与人类对话的不同交互模式及其对语法吸收的潜在优势；③结合问卷分析行为特征与学习者满意度的关系。

**🔧 技术方法**

使用OpenAI LLM（text‑to‑speech + voice‑activity detection）、ElevenLabs ASR与speaker diarization、spaCy/Stanza NLP管线、混合效应回归与OLS回归。

**📊 数据集**

实验数据：78名大学二/三年级德语学习者（四所高校），完成4种说话任务（两种单独叙述、两种对话），并使用自动转录文本进行分析。

**📈 对比分析**

通过配对t检验、线性混合模型和回归比较AI与人类对话的交互流畅度、词汇与句法吸收率。结果显示：AI对话更长、更少的轮次、学习者占用说话时间更少但单次流畅度更高；AI输入语法更丰富，短期句法吸收显著高于人类（在控制输入量后仍显著），但词汇吸收差异不显著。

**⚠️ 局限性**

局限性：①未能区分句法吸收的短期对齐与由共享任务情境引发的并行激活；②使用标准德语模型处理L2语料可能降低准确性；③ASR错误未修正；④实验仅使用一种AI配置、单一任务类型、有限的语言水平和母语背景；⑤实验未评估吸收的持久性；⑥AI与人类伙伴的感官与界面差异可能对结果产生影响。

---

## 37. Hulls and sums of separable constacyclic codes over $\mathbb{F}_q \times (\mathbb{F}_q+v\mathbb{F}_q)$ and new quantum codes

**arXiv ID:** 2606.22069 | [PDF](https://arxiv.org/pdf/2606.22069v1)

**作者:** Yu Qian `[一作]` (Hefei University), Liqi Wang `[通讯]` (Hefei University of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了在混合字母集 𝒮 = 𝔽_q × (𝔽_q + v𝔽_q) 上可分离 constacyclic 码的欧氏和厄米双码以及它们的 Gray 映像的生成多项式，并从这些码的 hulls 与 sums 推导出新的量子误差纠正码。

**💡 创新点**

创新点在于给出了可分离 constacyclic 码及其 Gray 映像的欧氏与厄米 hull 与 sum 的生成多项式，并基于此提出两种从 hull 与 sum 生成 QECC 的方法，产生了参数优于现有结果的新量子码。

**🔧 技术方法**

使用了 constacyclic 码、Gray 映射、欧氏与厄米内积、Hull 与 Sum 的代数结构以及 CSS 和 Hermitian 构造等经典量子码构造技术。

**📊 数据集**

主要使用符号构造，无特定公开数据集；实验验证通过 MAGMA 计算获得具体多项式与码参数。

**📈 对比分析**

与文献中已知的 QECC 进行参数对比，得到 [[203,187,3]]_7、[[64,46,4]]_11、[[145,129,4]]_5 等码的维度或距离均高于参考码，表现出更好的码率与距离。

**⚠️ 局限性**

限制在于只考虑可分离 constacyclic 码与其 Gray 映像，且代码长度受构造参数限制，尚未探讨非可分离码或更一般混合字母集。

---

## 38. Failure Analysis in Transition: An Industry Survey of Challenges, Priorities, and Standardization Needs in Advanced Packaging and Heterogeneous Integration

**arXiv ID:** 2606.22149 | [PDF](https://arxiv.org/pdf/2606.22149v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 39. IRumAI: Reinforcement Learning for Indian Rummy

**arXiv ID:** 2606.21975 | [PDF](https://arxiv.org/pdf/2606.21975v1)

**作者:** Vignesh Mohan `[一作]` `[通讯]` (EURECOM), Vignesh Mohan (EURECOM)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出并实现了首个针对印度拉米（Indian Rummy）游戏的强化学习代理IRumAI，能够在无需搜索的情况下实现高质量决策。

**💡 创新点**

创新点包括：① meld-aware 观察编码与基于死木点数的潜在奖励塑造；② 双分支卷积网络专为序列和牌组设计；③ 通过一次行为克隆热启动和对弱启发式对手的自对弈实现快速收敛与泛化；④ 证明网络隐式建模对手手牌并可实现极低延迟。

**🔧 技术方法**

技术手段涵盖：Proximal Policy Optimization (PPO) 与 Generalized Advantage Estimation；潜在奖励塑造（deadwood-based potential）；双分支卷积网络（1×3序列卷积 + 4×1牌组卷积）；行为克隆（BC）热启动；PettingZoo 环境；Numba JIT meld 分析器。

**📊 数据集**

数据集主要来源于自对弈：10,000局 MinDistOppAgent 自对弈用于行为克隆；随后在 128 并行环境中与 MinScoreAgent 与 MinScoreOppAgent 的自对弈数据训练；评估时对多种启发式和搜索式基线进行 1,000 局对战（共 5,000 局）。

**📈 对比分析**

与传统启发式基线相比，IRumAI 对 MinDistOppAgent 的胜率达 53.9%，对 MinScoreOppAgent 的胜率为 68.5%；与搜索式基线相比，推理延迟仅 0.33 ms（比 2,400 ms 的搜索快 7,000 倍），同时保持竞争力；在随机代理上的胜率 99.8%。

**⚠️ 局限性**

局限性：仅在两人单局模式下验证；训练对手池仅包含两类弱启发式；未对人类玩家或多局/多玩家环境进行评估；单种种子实验可能导致结果波动。

---

## 40. TRACE: A Threat Modelling Methodology for Distributed, Cloud-First, and Decentralized Organisations

**arXiv ID:** 2606.22214 | [PDF](https://arxiv.org/pdf/2606.22214v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 41. Multi4D: High-Fidelity Dynamic Gaussian Splatting via Multi-Level Competitive Allocation

**arXiv ID:** 2606.22197 | [PDF](https://arxiv.org/pdf/2606.22197v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 42. Nous: A Predictive World Model for Long-Term Agent Memory

**arXiv ID:** 2606.22030 | [PDF](https://arxiv.org/pdf/2606.22030v1)

**作者:** Pranav Singh `[一作]` `[通讯]` (Indian Institute of Technology Ropar), Pranav Singh (Indian Institute of Technology Ropar)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于预测世界模型的对话式代理记忆架构Nous，利用实体-属性对的概率分布及惊讶驱动的贝叶斯更新来持续更新记忆。

**💡 创新点**

核心创新在于将记忆视为预测系统而非事实存储，记录每次更新的“Delta”而非事实本身，并通过信息熵衰减实现自然遗忘与冲突自动解决。

**🔧 技术方法**

使用信息理论惊讶度量、闭式贝叶斯后验更新、熵衰减、KL相似度进行身份解析，以及BM25+两跳实体BFS的检索管线；全部实现不依赖外部向量数据库或图引擎。

**📊 数据集**

在LoCoMo长短期对话记忆基准上评估，包含单跳、多跳、时间、开放域四类问答。

**📈 对比分析**

与无记忆基线和公开报告的A‑MEM、BeliefMem进行对比，使用GPT‑4o‑mini作为LLM，Nous在所有四类问答中均获得最高token‑F1（单跳63.5、双跳55.3、时间58.6、开放域62.5），相较于A‑MEM提升显著，且自报数字超过BeliefMem，但作者强调差异可能源于检索与提示差异，需进一步对齐实验。

**⚠️ 局限性**

主要局限包括缺乏系统性消融实验、对多值属性的处理仅为后处理聚合、时间戳信息未显式建模、检索诊断对开放域与时间类表现偏低，以及仅在LoCoMo与GPT‑4o‑mini上验证，尚未验证跨模型或更严格基准（LongMemEval）下的泛化能力。

---

## 43. Look Before You Zoom: Adaptive Routing for the Resolution-Context Trade-off in Visual RAG

**arXiv ID:** 2606.21968 | [PDF](https://arxiv.org/pdf/2606.21968v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 44. Dynamics, stability, and energy efficiency of an energy-recycling rimless wheel with spring-clutch legs

**arXiv ID:** 2606.22073 | [PDF](https://arxiv.org/pdf/2606.22073v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 45. InstructFX2FX: A Multi-turn Text-to-Preset Demo for Iterative Audio Effect Refinement

**arXiv ID:** 2606.22005 | [PDF](https://arxiv.org/pdf/2606.22005v1)

**作者:** Song-Ze Yu `[一作]` (University of California, Berkeley), Wantong Zhang `[通讯]` (University of California, Berkeley)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了 InstructFX2FX，一套支持多轮文本指令的音频效果链细化系统。

**💡 创新点**

将大型语言模型的高层规划与 CLAP 嵌入驱动的感知优化相结合，形成会话感知的三路由（初始化、重用+优化、混合）多轮细化框架。

**🔧 技术方法**

使用大型语言模型 (LLM) 生成效果链与初始参数；CLAP 音频‑文本嵌入提供梯度下降（可导）和贝叶斯优化（不可导）两种优化后端；会话状态管理与优化轨迹回溯。

**📊 数据集**

以 SocialFX 数据集中的 EQ 描述词（warm、bright、soft 等）构建 10 对有向词组进行评估。

**📈 对比分析**

对比 LLM 仅重提示的基线；在 9/10 语义对上 MMD 降低约 24%（从 0.4461 降到 0.3380），并在大多数词组上持续优于基线，验证 CLAP 优化带来的显著改进。

**⚠️ 局限性**

仅评估 EQ 相关描述词；CLAP 嵌入与 DSP 特征的匹配不完全，导致优化轨迹有时漂移；不可导效果的贝叶斯优化噪声大、语义不稳定；每轮交互耗时数秒，难以实现实时 DAW 集成。

---

## 46. Improving Reasoning in Vision-Language Models via Perception Verified Self-Training

**arXiv ID:** 2606.22158 | [PDF](https://arxiv.org/pdf/2606.22158v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 47. From Driving Videos to Simulatable Scenarios

**arXiv ID:** 2606.21993 | [PDF](https://arxiv.org/pdf/2606.21993v1)

**作者:** Alexandre Levy `[一作]` (Universitat Autonoma De Barcelona), Antonio Manuel López `[通讯]` (Universitat Autonoma De Barcelona)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段框架 D-V2S，能将驾驶视频自动转化为可执行的 SCENIC 场景脚本。

**💡 创新点**

创新点在于使用 VLM 生成自然语言描述，再利用 LLM 在上下文提示下直接合成可编译的 SCENIC 脚本，避免了传统手工编程和轨迹重放的限制。

**🔧 技术方法**

核心技术包括：视觉语言模型（LLaVA、Qwen-VL、GPT‑4o）用于视频理解；大型语言模型（GPT‑4o）结合 prompt‑engineering 生成 SCENIC 代码；SCENIC 高层脚本语言与 CARLA 仿真平台。

**📊 数据集**

使用的数据集主要为 CARLA 生成的驾驶失败视频、Crash Report Dataset 以及真实 GoPro 车载摄像头视频。

**📈 对比分析**

与 LCTGen 与 ChatScene 对比，D-V2S 在语义保真度（SP 0.93）、人类偏好率（75%）以及端到端语义一致性（E2E‑SC 90%）等指标均显著优于两者，且编译率高达 94%。

**⚠️ 局限性**

主要限制包括：6% 的脚本因语法或缺失参数导致编译失败；依赖 SCENIC 语言，若需其他仿真器需重新适配；当前仅在前视摄像头视角下验证，其他摄像头或多模态输入尚未充分测试。

---

## 48. BAC-JEPA: Label-Efficient Breast Arterial Calcification Segmentation via Synthetic Mammography-Guided Supervision

**arXiv ID:** 2606.22089 | [PDF](https://arxiv.org/pdf/2606.22089v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 49. A Standard Processing Pipeline for High-accuracy Measurement of Few-shot Regression on Laser Induced Breakdown Spectroscopy

**arXiv ID:** 2606.21960 | [PDF](https://arxiv.org/pdf/2606.21960v1)

**作者:** Hao Li `[一作]` `[通讯]` (University of Arizona), Hao Li (University of Arizona)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并验证了一套针对稀样本（少于20样本）LIBS回归的标准化处理流程，集成了扩散式去噪、注意力自编码器、组交叉数据增强和OLS回归。

**💡 创新点**

首次将扩散模型（3D UNet）与注意力机制、自编码器以及组交叉数据增强相结合，形成一条兼顾噪声抑制与特征保留、能在极少样本条件下实现高精度回归的完整链路。

**🔧 技术方法**

使用的核心技术包括：扩散式去噪（3D UNet+时间嵌入）、注意力自编码器（channel‑wise attention）、组交叉数据增强、OLS回归；同时对比了PCA‑PLS、传统自编码器、无去噪或无增强版本等基线。

**📊 数据集**

基准数据集为在运输带上测量的煤炭质量LIBS数据，使用1064 nm Nd:YAG激光，光谱范围180–800 nm，采集21种元素浓度（每种元素样本不足20个）。

**📈 对比分析**

通过消融实验与传统基线对比，完整Pipeline（Diffusion‑DA‑AE）实现MSE 18.25、RMSE 2.85、RMAE 0.2847，显著优于无去噪版（0.3012）、无增强版（0.4123）及PCA‑PLS（0.4568）。组交叉增强在所有元素上平均RMAE降至0.2364，比其他四种增强方法低19.1%。

**⚠️ 局限性**

局限性包括：仅在煤炭LIBS数据上验证，尚未检验在其他材料或更高维光谱上的泛化；扩散去噪与3D UNet计算量大，需较强硬件；在极低样本（≤5）或高噪声环境下的表现仍待进一步评估。

---

## 50. Resolving Multi-Target Association in OFDM-based ISAC via Vision-aided Multi-Modal Learning

**arXiv ID:** 2606.22195 | [PDF](https://arxiv.org/pdf/2606.22195v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 51. Can Reasoning Models Detect Changes to their Chains of Thought?

**arXiv ID:** 2606.22085 | [PDF](https://arxiv.org/pdf/2606.22085v1)

**作者:** Sathvik Napa `[一作]` (Johns Hopkins University), William Walden `[通讯]` (Johns Hopkins University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对近期推理模型在链式思考（CoT）被编辑后是否能检测到这些编辑进行了系统性实验。

**💡 创新点**

首次将CoT检测与定位任务与自我意识维度结合，展示了模型在检测与定位改动方面仅略高于偶然水平，且缺乏对自身CoT的特殊敏感性。

**🔧 技术方法**

使用CoT预填技术、告警系统提示、二分类判断任务与定位任务，并结合统计显著性检验（两侧z检验、两比例z检验），评估四大开源推理模型（GPT‑OSS‑120B、Kimi K2.5、DeepSeek V3.2、Qwen3‑235B‑A22B‑Thinking）的表现。

**📊 数据集**

实验覆盖GPQA‑Diamond、MMLU‑Pro子集、AIME 2025以及MATH‑500等公开基准数据集。

**📈 对比分析**

通过比较完成CoT与部分CoT、自己CoT与他人CoT以及不同改动类型的检测与定位准确率，发现检测准确率仅略高于50%（常见≈55‑60%），多数情况未显著；定位准确率仅在插入无关步骤时达到42‑89%，其余改动难以定位，且无模型在所有条件下表现出显著优势。

**⚠️ 局限性**

限制：仅评估支持CoT预填的公开开源模型；未覆盖专有前沿模型；未进行后训练或微调；未来模型或训练策略可能导致结果变化；整体检测与定位能力受限，缺乏真正的自我意识。

---

## 52. On the Curse of Dimensionality in Private Sparse Covariance Estimation and PCA

**arXiv ID:** 2606.21951 | [PDF](https://arxiv.org/pdf/2606.21951v1)

**作者:** Syamantak Kumar `[一作]` (University of Texas at Austin), Kevin Tian `[通讯]` (University of Texas at Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

在高维下研究对角化约束（k-row-column稀疏）下的不同ially private协方差估计与主成分分析（PCA），并给出上界与下界，展示了私有化导致的维数灾难。

**💡 创新点**

提出了在k-row-column稀疏且主特征向量也稀疏时，PCA可以在近乎最佳（k,log d）样本复杂度下实现；同时证明了在无主特征向量稀疏假设下，协方差估计与PCA的私有样本复杂度必须为Ω(d)，从而首次实现了私有与非私有版本之间的指数分离。

**🔧 技术方法**

采用阈值化、Fisher‑Core私有估计、先进合成、指纹化技术、基于图的构造、私有Assouad方法以及矩阵谱估计等技术组合，构建高效的私有算法与严谨的信息论下界。

**📊 数据集**

本研究主要基于理论模型（σ‑sub‑Gaussian分布），未使用真实数据集；所有实验与分析均在理论假设下完成。

**📈 对比分析**

与之前的工作相比，传统私有方法在k-row-column稀疏下需要Ω(dk²)样本，而本文在额外稀疏约束下实现了近似最优的（k,log d）样本；在PCA方面，传统私有方法的下界较弱，而本文给出Ω(d)的下界，表明在更广泛参数范围内私有PCA仍需线性样本。

**⚠️ 局限性**

上界与下界在隐私模型上不完全对齐：上界采用近似DP（δ>0），下界针对纯DP（δ=0）；此外，对approximate DP的下界与上界在参数化（子高斯参数）上不匹配，留下在approximate DP下实现Ω(d)下界或(k,log d)上界的开放问题。

---

## 53. Tactile Genesis: Exploring Tactile Sensors at Scale for Learning Dexterous Tasks

**arXiv ID:** 2606.22332 | [PDF](https://arxiv.org/pdf/2606.22332v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 54. Cross-Platform Software Birthmarking for Real-World Binaries via Intermediate Representation

**arXiv ID:** 2606.21988 | [PDF](https://arxiv.org/pdf/2606.21988v1)

**作者:** Haruaki Tamada `[一作]` `[通讯]` (Kyoto Sangyo University), Haruaki Tamada (Kyoto Sangyo University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一套基于Ghidra P-code的中间表示（Oinkie-IR）进行跨平台软件出生标记（birthmarking）的技术，能够对不同操作系统、CPU架构和编程语言的可执行文件进行相似度比较；

**💡 创新点**

通过统一的中间表示消除ISA差异，实现了跨平台、高一致性的出生标记；提出了“稀释效应”现象并用Simpson指数等集合型相似度方法提升辨别力；

**🔧 技术方法**

使用Ghidra的二进制提升（P-code）、Oinkie-IR JSON格式、集合/序列相似度算法（Dice、Jaccard、Simpson、LCS、Levenshtein、Cosine等）以及Hungarian/Top‑n聚合；

**📊 数据集**

以bzip2及其Go、Rust实现为主要软件，编译生成MacOS Mach‑O、Linux ELF、Windows PE，跨架构（amd64/arm64）和跨语言、跨编译器的多版本二进制；

**📈 对比分析**

采用聚合后的相似度矩阵进行全对比，使用Hungarian算法和Top‑n（n=1,∞）聚合，实验显示跨架构相似度相关系数0.9846，集合型算法（尤其Simpson）在识别相似与不同软件时具有最高辨别力；相似度计算耗时：集合型约8h，序列型约56h；

**⚠️ 局限性**

局限性包括对环境噪声（如Windows静态链接导致的函数膨胀）敏感、跨语言差异仍显著、依赖于Ghidra的提升质量、仅在bzip2等通用工具上验证，尚未覆盖GUI、内核级或加密混淆软件。

---

## 55. CoRDE: Concept-Prior Routed Diffusion Experts for Structural Generalization in Robot Manipulation

**arXiv ID:** 2606.21935 | [PDF](https://arxiv.org/pdf/2606.21935v1)

**作者:** Haidong Huang `[一作]` (Eastern Institute of Technology), Xiaocong Li `[通讯]` (Eastern Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CoRDE框架，将概念先验与扩散模型的Mixture-of-Experts相结合，实现高效多任务机器人模仿学习。

**💡 创新点**

通过结构引导的责任推断融合语义先验与行为证据，避免路由崩溃；采用低秩LoRA专家共享主干，兼顾参数效率与行为多样性；自组织EM软对齐实现可解释的概念-专家映射。

**🔧 技术方法**

变分扩散蒸馏、概念编码器（HiMaCon/AutoCGP）、软映射矩阵、熵控制责任推断、低秩适配（LoRA）、EM自组织对齐、SDE逆采样等技术。

**📊 数据集**

LIBERO（长时序多任务）与D3IL（多模态人类演示）两大基准数据集。

**📈 对比分析**

与原始扩散教师、无结构引导的Distill-MoE、全参数VDD进行对比；CoRDE在LIBERO宏平均成功率最高，尤其在L-Long/L-Goal；在D3IL上保持高任务成功率、几乎等同或更高的任务熵，同时单步推理时间仅2.6ms，显著优于教师和VDD。

**⚠️ 局限性**

依赖冻结概念编码器，概念质量可能限制专家分配；低秩LoRA在极高维任务可能不足；目前仅在仿真环境验证，缺乏真实世界部署与在线自适应研究。

---

## 56. Cultural Targets, Structural Frames, Binding Morals: A Cross-Lingual Audit of Online Hate in Multicultural Singapore

**arXiv ID:** 2606.21996 | [PDF](https://arxiv.org/pdf/2606.21996v1)

**作者:** Emilio Ferrara `[一作]` `[通讯]` (University of Southern California), Emilio Ferrara (University of Southern California)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用新加坡三语（英语、简体中文、马来语）2025年Facebook、Reddit、YouTube的31M条评论，采用开放式LLM（Phi‑4、DeepSeek、Qwen等）进行两阶段高召回词法+LLM二元判定，并给出威胁框架、道德基础、叙事、情感、立场等多标签，系统分析仇恨的目标、表达与放大机制；

**💡 创新点**

提出分层文化偶像论，揭示跨语言仇恨结构在目标层面高度文化差异、在威胁框架和道德基础层面趋于一致，并首次从产生与放大双视角探讨仇恨传播；

**🔧 技术方法**

采用开放式LLM进行两阶段分类（高召回词法过滤+LLM二元判定），多标签情感与道德标签；统计方法包括Cramér's V、Bootstrap、Benjamini–Hochberg FDR、逻辑回归；

**📊 数据集**

31.0M条目（252个空间）2025年新加坡本土化Facebook、Reddit、YouTube评论，涵盖英语、简体中文、马来语，1.76M条提及11个身份群体的评论，最终识别3,323条确认仇恨；

**📈 对比分析**

与211条人工金标对比，Phi‑4 kappa 0.91、准确率0.95、召回率1.00；8个开源模型kappa约0.42；多标签（框架、道德）准确率0.83–0.97；在Qwen验证与Phi‑4结果保持一致；

**⚠️ 局限性**

绝对仇恨比例在不同LLM间差异大、跨语言标签依赖英文翻译、样本来自选定新闻源非随机、仅评论级别不涉及用户身份、平台与语言混杂导致可能偏倚、模型对中文/马来语表现下降；

---

## 57. From Convolution to Transformer: A Comparative Study of U-Net Variants for Brain Tumor and Retinal Vessel Segmentation

**arXiv ID:** 2606.22168 | [PDF](https://arxiv.org/pdf/2606.22168v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 58. Wh0: Generative World Models as Scalable Sources of Egocentric Human Hand Manipulation Data

**arXiv ID:** 2606.22136 | [PDF](https://arxiv.org/pdf/2606.22136v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 59. Parameterized Representations via Implicit Stochastic Modulation for High-Dimensional and High-Order Neural PDE Solvers

**arXiv ID:** 2606.22150 | [PDF](https://arxiv.org/pdf/2606.22150v1)

**作者:** Zhangyong Liang `[一作]` (Tianjin University), Huanhuan Gao `[通讯]` (Jilin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

本文提出了 PRISM 架构，能够在高维高阶随机 PDE 求解器中实现参数化表示，实现零差错的 AD 解耦和变异控制；

**💡 创新点**

创新点在于通过隐式随机调制（affine 缩放与平移）将物理参数与空间网络分离，消除 AD 图混合导致的显存爆炸，并通过自适应 Lipschitz 阻尼抑制随机梯度方差；

**🔧 技术方法**

技术上使用了随机方向导数估计器（STDE、SDGD、HTE）与多层感知机的参数超生成器、张量 SVD 微调、常量折叠等；

**📊 数据集**

实验数据集主要是合成的参数化 PDE（Poisson、Allen‑Cahn、Sine‑Gordon 等），在 100‑000 维空间内采样残差点和随机方向；

**📈 对比分析**

与传统参数拼接方法相比，PRISM 在 100‑000 维下实现了 <1e‑3 的相对 L2 错误，显存仅 ~33GB，训练时间保持线性增长，避免 OOM；

**⚠️ 局限性**

局限性在于需要较大维度批量，且对非残差+边界形式的 PDE（如 Schrödinger）仍可能面临内存瓶颈。

---

## 60. Bayesian Adaptation Gym: A Benchmark for the Bayesian Low-Rank Adaptation of Multi-Modal Language Models

**arXiv ID:** 2606.22188 | [PDF](https://arxiv.org/pdf/2606.22188v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 61. Drowning in Routine: Signal Dilution in Multi-Turn Agent Training

**arXiv ID:** 2606.22164 | [PDF](https://arxiv.org/pdf/2606.22164v1)

**作者:** Yann Pernot `[一作]` (Mila - Quebec AI Institute), Vi Retault `[通讯]` (Polytechnique Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文分析多轮强化学习中基于trajectory的信用分配与基于turn的信用分配的差异，提出信号稀释（signal dilution）机制，并通过定义决策密度ρ阐释其对梯度信噪比的影响；随后在可调ρ的受控MDP“Diluted Doors”上验证理论预测。

**💡 创新点**

创新点包括：①将动作分为routine与critical两类，并引入决策密度ρ作为衡量重要决策稀疏程度的结构量；②揭示trajectory-level估计因routine状态引入额外方差，形成信号稀释；③推导梯度SNR比值随ρ的Θ(ρ⁻¹/²)上界，并在存在critic误差时给出修正；④提供理论上对比框架，并在实验中验证其可控性。

**🔧 技术方法**

主要技术包括：概率与梯度理论分析（分布不变性、梯度SNR），两类梯度估计方法（PPO/critic-based turn-level、GRPO-like trajectory-level），以及构造可调ρ的Synthetic MDP。

**📊 数据集**

使用的“数据集”是合成的Diluted Doors MDP（树形结构，控制决策密度），没有使用真实LLM或游戏环境。

**📈 对比分析**

通过在同一策略网络（小型因果Transformer）下，比较两种梯度估计的梯度SNR和达到性能阈值所需的训练步数；实验显示，当ρ较低时，turn-level方法的SNR比约为ρ⁻¹/²，训练步数提升约为2倍，验证了理论预测。

**⚠️ 局限性**

局限性：①未在真实LLM或复杂任务上验证；②routine状态和决策密度是策略相关且难以直接观测；③仅针对完全routine状态的定义，未考虑近似routine的情况；④需要进一步研究如何在实际环境中估计ρ并利用其设计更高效的信用分配策略。

---

## 62. A feasibility study on filtering low-accessibility web pages considering color vision deficiency

**arXiv ID:** 2606.22095 | [PDF](https://arxiv.org/pdf/2606.22095v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 63. OpenHLM: An Empirical Recipe for Whole-Body Humanoid Loco-Manipulation

**arXiv ID:** 2606.22174 | [PDF](https://arxiv.org/pdf/2606.22174v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 64. SPOTR: Spatio-temporal Pooling One-Token Reconstruction for Universal Physiological Signal Self-supervised Learning

**arXiv ID:** 2606.21973 | [PDF](https://arxiv.org/pdf/2606.21973v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 65. Residue-Level Attributions in Protein Language Models Do Not Recover Allergen Epitopes

**arXiv ID:** 2606.22181 | [PDF](https://arxiv.org/pdf/2606.22181v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 66. Denoising-Enhanced Coarse-to-Fine Infrared Small Target Detection with Attention Prior-Guided Knowledge Distillation

**arXiv ID:** 2606.21956 | [PDF](https://arxiv.org/pdf/2606.21956v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 67. KITE: Decoupling Kinematics and Interaction for Zero-Shot Cross-Embodiment Manipulation

**arXiv ID:** 2606.22113 | [PDF](https://arxiv.org/pdf/2606.22113v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 68. Morphology-Aware Multimodal Representation Learning for Insect Phylogenetic Reconstruction

**arXiv ID:** 2606.22077 | [PDF](https://arxiv.org/pdf/2606.22077v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 69. Artic-O: End-to-End Articulated Object Reconstruction via Latent Geometry Learning

**arXiv ID:** 2606.21938 | [PDF](https://arxiv.org/pdf/2606.21938v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 70. Game-Theoretic Framework for Private Data Sharing in Vehicular Networks

**arXiv ID:** 2606.22115 | [PDF](https://arxiv.org/pdf/2606.22115v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 71. Accurate identification and measurement of the precipitate area by two-stage deep neural networks in novel chromium-based alloys

**arXiv ID:** 2606.22112 | [PDF](https://arxiv.org/pdf/2606.22112v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 72. Gated MLPs as Symmetry-Broken Rank-1 Bilinear Attention

**arXiv ID:** 2606.22172 | [PDF](https://arxiv.org/pdf/2606.22172v1)

**作者:** Nathan Breslow `[一作]` `[通讯]`, Nathan Breslow

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

将传统的 gated MLP 视为双线性注意力机制的 rank‑1 近似，并分析其在交换对称性与逆尺度对称性上的破缺。

**💡 创新点**

提供了一个新的解释框架，将 gated MLP 与双线性注意力关联起来，并揭示了非线性激活对对称性的影响，从而解释了 gated MLP 在实践中的有效性。

**🔧 技术方法**

使用数学推导、rank‑1 近似、双线性注意力形式以及对称性分析等理论工具。

**📊 数据集**

本工作为纯理论研究，没有使用任何数据集。

**📈 对比分析**

未进行实验对比，本文仅提供理论分析与解释性结论。

**⚠️ 局限性**

局限性：缺乏经验验证，未探讨更高秩近似的实用性和性能，以及对实际模型设计的具体指导。

---

## 73. New Smooth Loss functions for Robust Regression that Closely Approximate Absolute Error and Provide Improved Performance on Datasets With Significant Outliers

**arXiv ID:** 2606.22068 | [PDF](https://arxiv.org/pdf/2606.22068v1)

**作者:** Mathew Mithra Noel `[一作]` (Vellore Institute of Technology), Venkataraman Muthiah-Nakarajan `[通讯]` (Vellore Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了两种可微分且无穷可微的 MAE 近似损失函数（SMAE 与 SRL），并在含大量异常值的回归任务中实现更鲁棒的学习。

**💡 创新点**

创新点是将 SMAE 定义为严格准凸、SRL 定义为严格凸且在大误差时逼近 MAE，同时保持计算成本低。

**🔧 技术方法**

使用梯度下降/Adam 训练深度网络和线性回归模型，并为自定义损失推导了对应的 SGD 更新公式。

**📊 数据集**

在 MNIST 合成定位、加州住房、混凝土强度、葡萄酒质量等 5 个公开数据集上进行评测。

**📈 对比分析**

通过与 MSE、Huber、Log‑Cosh 等主流损失对比实验，SMAE/SRL 在 RMSE、MAE、R² 等指标上均优于或相当，尤其在含 20–50% 异常值的场景中表现显著。

**⚠️ 局限性**

局限性在于仅针对回归任务，缺乏对分类或非连续目标的验证，并且对极大噪声的理论收敛分析不足。

---

## 74. Sequential Minimal Optimization Algorithm for One-Class Support Vector Machines With Privileged Information

**arXiv ID:** 2606.22210 | [PDF](https://arxiv.org/pdf/2606.22210v1)

**作者:** Andrey Lange `[一作]` (Skolkovo Institute of Science and Technology), Evgeny Burnaev `[通讯]` (Skolkovo Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种针对一类支持向量机（OC‑SVM）引入特权信息（privileged information）的Sequential Minimal Optimization（SMO）算法，并证明其有限步收敛性；

**💡 创新点**

创新点在于首次将LUPI范式应用于无监督一类SVM，并设计了兼顾原始与特权空间的两变量优化子问题，实现快速、可扩展的训练；

**🔧 技术方法**

采用SMO的坐标下降思路，利用核函数（原始与特权空间）构造二次子问题，并结合缓存与收缩（shrinking）策略；

**📊 数据集**

实验使用合成数据和UCI Shuttle数据集（包含9维特征，部分特征作为特权信息）进行评估；

**📈 对比分析**

与传统的内部点方法（CVXOPT）以及无特权信息的OC‑SVM比较，SMO在大样本下训练时间显著更短，且在异常检测任务中取得更高的平均精度；

**⚠️ 局限性**

主要局限包括对缓存策略的依赖、对特权信息选择和参数调优的敏感性，以及实现语言对速度影响显著，未来需进一步研究更高效的实现和更通用的特权信息整合方法。

---

## 75. Adding Robust Code-Switching Capabilities to High Performance Multilingual ASR

**arXiv ID:** 2606.21990 | [PDF](https://arxiv.org/pdf/2606.21990v1)

**作者:** Enes Yavuz Ugan `[一作]` (Karlsruhe Institute of Technology), Alexander Waibel `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `67630363-6be0-4f51-ab05-7198250671a5` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究者提出了一种在强大多语言语音识别模型（如 Whisper）上进行代码切换（CSW）适配的方法，重点是通过贝叶斯低秩适配（BLoRA）在不损失单语性能的前提下提升对英语-德语代码切换的识别能力。

**💡 创新点**

创新点在于：① 将知识整合视为核心，而非单纯的数据增量；② 使用贝叶斯低秩适配实现稀疏且不破坏原模型知识的增量学习；③ 通过 GPT‑4o 生成具备形态学融合的代码切换文本，并利用 TTS 合成高质量合成语音；④ 通过 PIER 指标专门评估代码切换单词的识别错误。

**🔧 技术方法**

技术包括：LLM 代码切换文本生成、形态学约束提示、X‑TTS 多语音合成、语音切割与拼接、LoRA 与 BLoRA 适配方法、KL 正则化、CER 过滤策略、WER 与 PIER 评估。

**📊 数据集**

使用的数据集有：① 生成的英语-德语代码切换语音（约 10k–246k 句子）; ② CSFleurs（公开的德英代码切换基准）； ③ DECM（德英对话式 CSW 数据集）； ④ CommonVoice 14.0（读音单语测试，用于后向验证）。

**📈 对比分析**

与传统 LoRA 细调相比，BLoRA 在 CSFleurs 上实现了 32.87% 的相对 PIER 降低和 5.31% 的相对 WER 提升，同时在 CommonVoice 上保持了原有单语性能；在 DECM 上也显示出较 LoRA 更低的性能衰退。实验表明数据量增加或过滤程度提升对 BLoRA 效果帮助有限，核心在于适配器的稀疏与贝叶斯先验。

**⚠️ 局限性**

局限性包括：① 仅在英语-德语对上验证，未知对其他语言对的泛化能力；② 主要基于读音数据，可能对对话式或噪声环境适配效果不佳；③ 需要 GPT‑4o 等大型 LLM 的支持，成本较高；④ BLoRA 的超参数（如 rank、KL 权重）仍需经验调优。

---

## 76. Deep RL- Tuned Mo del-Free Adaptive Control for Lower-Limb Exoskeletons During Sit-to-Stand Transitions

**arXiv ID:** 2606.22040 | [PDF](https://arxiv.org/pdf/2606.22040v1)

**作者:** Ranjeet Kumbhar `[一作]` (Thapar Institute of Engineering and Technology), Ravinder Kumar `[通讯]` (Thapar Institute of Engineering and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并验证了一种无模型自适应反步控制器，结合 Gaussian RBF 神经网络估计未知动力学，并通过 TD3 强化学习调度控制增益，用于双腿下肢外骨骼在坐起站立（STS）运动中的轨迹跟踪。

**💡 创新点**

创新点在于①采用二阶无模型超本地模型与 RBF 估计相结合的自适应控制框架；②将 TD3 作为监督增益调度器实现分阶段自适应；③在同一框架下实现最优的误差与扭矩性能。

**🔧 技术方法**

使用的技术包括无模型自适应控制（RBF‑MFAC）+ 反步法 + Gaussian RBF 网络 + TD3 强化学习 + MATLAB/Simulink + Simscape Multibody + OpenSim 参考轨迹。

**📊 数据集**

使用的参考数据集为 OpenSim 逆运动学生成的标准人体坐起站立关节角度轨迹，采样频率 1 kHz，持续时间 10 s。

**📈 对比分析**

通过在相同仿真环境下与 PID、LQR、SMC、MFAC 四种基准控制器比较，采用 RMSE、MAE、峰值误差和扭矩消耗等指标，结果显示所提控制器 RMSE 降至 0.078°（比 PID 下降 60.2%、LQR 54.4%、SMC 48.7%、MFAC 42.6%），且扭矩消耗最低，TD3 进一步提升误差降低约 35–79%。

**⚠️ 局限性**

局限性包括仅在高保真仿真中验证，缺乏真实人体实验、非对称运动、EMG 信号融合和硬件实现；对外部扰动的鲁棒性在髋/膝受限，踝扰动测试受限；增益调度仍以经验范围约束，可能在极端情况不稳定。

---

## 77. BioMatrix: Towards a Comprehensive Biological Foundation Model Spanning the Modality Matrix of Sequences, Structures, and Language

**arXiv ID:** 2606.22138 | [PDF](https://arxiv.org/pdf/2606.22138v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 78. Integrating Facial Generation into Full-Duplex Spoken Dialogue Systems

**arXiv ID:** 2606.21970 | [PDF](https://arxiv.org/pdf/2606.21970v1)

**作者:** Jingjing Jiang `[一作]` (Nagoya University), Ryuichiro Higashinaka `[通讯]` (Nagoya University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了Mos​hi‑Face，一种在全双工语音对话框架中同时处理与生成音频与三维面部运动的模型。

**💡 创新点**

创新点在于：①将三维面部运动量化为离散面部 token；②在原有 Mos​hi 结构上增设无因果 Face Transformer，使面部 token 能实时、并行生成；③实现了音频与面部运动在 12.5 Hz 同步生成，突破了全双工模型只关注音频的局限。

**🔧 技术方法**

核心技术包括：VQ‑VAE 面部编码器/解码器、Moshi 的 RQ‑Transformer、面部 Transformer、Mimi VQ‑VAE（音频），以及 3D 网格提取工具 VHAP/FLAME。

**📊 数据集**

使用 180 小时（约 3,400 条对话）从 Meta Seamless Interaction 数据集中抽取的三维面部视频，并利用 VHAP 生成 5,143‑顶点 FLAME 3D 网格。

**📈 对比分析**

通过教师强制与自由跑两种评估；指标有 LSE‑D/LSE‑C（口型同步）、UTMOS（语音自然度）、LLMAJ（对话语义质量）。Mos​hi‑Face 在同步上接近上限 Reconstructed face，明显优于 Random face，并保持与原 Mos​hi 相当的对话质量。

**⚠️ 局限性**

局限包括：面部 codec 仍为非因果结构，无法实现完全实时视觉输入输出；训练数据域有限导致语音自然度略低；缺乏人类感知评估，未覆盖更大多样化场景。

---

## 79. $π$-RAG: Oblivious Retrieval via Semantic Quantization and Transcendental Addressing for Large Language Models

**arXiv ID:** 2606.22153 | [PDF](https://arxiv.org/pdf/2606.22153v1)

**作者:** Aniket Wattamwar `[一作]`, Mrunal Kakirwar `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

介绍了一种新型检索架构π-RAG，通过使用π的不可篡改数字序列作为间接层，将大型语言模型与敏感数据分离，实现零知识检索。

**💡 创新点**

创新点在于将用户查询映射到预定义的Canonical Intent，然后用π数字生成不可变的π-Key，既避免了向量嵌入逆向攻击，又保证检索过程的确定性与可审计性。

**🔧 技术方法**

采用了语义量化层（Semantic Quantization Layer）、意图注册与盐值加密生成π-Key、Air‑gapped检索子系统以及Gemma LLM进行意图分类和语义路由。

**📊 数据集**

实验使用合成银行业务数据集（账户余额、交易记录等），并在Gemma3:1b和Gemma3n:e4b两大模型上进行意图分类和π-Key生成。

**📈 对比分析**

通过对33条查询进行实验，使用Gemma模型分别得到意图分类准确率，Gemma3n:e4b表现略优；检索延迟保持在1秒以内，低于传统RAG在大规模数据时可能出现的高延迟。

**⚠️ 局限性**

主要局限包括：语义量化层引入额外延迟、π-Key碰撞虽极低但理论非零、对新意图的可扩展性受限于预定义意图的重新注册、实验仅在合成数据上完成，缺乏大规模生产环境的压力测试。

---

## 80. What Do Neural Networks Learn for TDOA Estimation? A Cross-Architecture Probing Study

**arXiv ID:** 2606.22020 | [PDF](https://arxiv.org/pdf/2606.22020v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 81. Beyond Value Benchmarks: Measuring Value-Structure Alignment in Large Language Models via Symmetric Q-Sorts

**arXiv ID:** 2606.21939 | [PDF](https://arxiv.org/pdf/2606.21939v1)

**作者:** Jingting Zheng `[一作]` (Tianjin University), Deyi Xiong `[通讯]` (Tianjin University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出对齐大语言模型价值结构的对称 Q‑sort 评估框架，并用 140 条道德陈述进行人类与 LLM 的比较。

**💡 创新点**

创新点在于将 Q 方法与 LLM 对齐相结合，构造统一价值词典与强制分布 Q‑sort，提供结构层面的价值对齐度量。

**🔧 技术方法**

使用 Q‑methodology、主成分分析（PCA）、Procrustes 变换、RSA Spearman 等几何比较技术。

**📊 数据集**

数据集为 140 条基于 Schwartz、MFT、MAC 的价值陈述，人工样本 35 名参与者，12 个跨家族 LLM（共 240 次 Q‑sort）。

**📈 对比分析**

对齐方法采用 Procrustes 相似度 ϕ 与 RSA ρ，最高的 ϕ≈0.37、ρ≈0.18，结果显示不同模型家族和温度下的结构一致性差异显著。

**⚠️ 局限性**

局限性包括：部分模型/温度条件出现秩崩溃导致几何度量不可用；人类参考结构仅来自有限样本和英语问卷；缺乏对行为后果的预测效度。

---

## 82. Modularized Reinforcement Learning on LLMs: From MDP Creation to Exploration and Learning

**arXiv ID:** 2606.21943 | [PDF](https://arxiv.org/pdf/2606.21943v1)

**作者:** Zhao Yang `[一作]` (VU Amsterdam), Vincent François-Lavet `[通讯]` (VU Amsterdam)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对大型语言模型（LLM）训练中的强化学习（RL）方法进行了系统梳理，构建了基于MDP创建、探索与学习三大阶段的统一框架，并将现有研究映射到该框架中，揭示了研究分布的不均衡与潜在空白。

**💡 创新点**

创新点在于提出了一套从MDP构造到探索、学习的完整设计维度，系统性地归纳了RL在LLM中的关键决策，并指出了诸如价值方法、离线/多模态学习、bootstrapping信用分配等方向在LLM中鲜有涉猎，提供了未来工作的新视角。

**🔧 技术方法**

主要使用了文献综述与归纳法，构建了RL设计的层次化分类表，并对各子领域（奖励设计、状态/动作空间、探索策略、学习方式、信用分配等）做了技术细分和方法比较；同时结合RL经典概念与LLM实践，做了概念映射与批判性讨论。

**📊 数据集**

本文为综述性工作，并未使用新的实验数据集；引用了LLM预训练、SFT与RLHF等常用数据集（如OpenAI InstructGPT、DeepSeek-R1、Gemma等）作为实例进行说明。

**📈 对比分析**

通过对比不同RL方法在LLM中的应用频率与表现，文章指出：critic‑free policy gradients（如GRPO）和PPO已成为主流，价值方法和离线/多模态学习几乎缺失；探索技术多停留在token级别；信用分配主要采用Monte Carlo或trajectory‑level优势。虽然未做原始实验，但对已有实验结果做了汇总，显示现有主流方法在对齐和推理任务中已达到较高效果。

**⚠️ 局限性**

局限性：①综述受已发表文献的覆盖范围限制，可能遗漏最新或非公开工作；②缺乏统一的量化对比与基准实验，主要依赖已有论文的结果；③对RL技术在LLM中的适用性与理论分析仍不够深入，未能系统验证提出的空白方向。

---

## 83. Multi-AUV Marine Life Tracking with Single Hydrophone Payloads via a Hidden Markov Model Equipped Particle Filter

**arXiv ID:** 2606.22335 | [PDF](https://arxiv.org/pdf/2606.22335v1)

**作者:** Christopher Herrera `[一作]` (Harvey Mudd College), Mario Espinoza `[通讯]` (University of Costa Rica)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `51c0528b-f690-4182-ae60-bb5f046c276c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究提出了一种多AUV单舷单向水听器系统，利用隐藏马尔可夫模型（HMM）驱动的粒子滤波器，对声学标记海洋生物的位置进行跟踪。

**💡 创新点**

创新点在于将单水听器的低拖拽硬件与HMM运动模型相结合，既减少了AUV的负载，又通过多机测量实现了高精度（约10-15 m）定位。

**🔧 技术方法**

使用了单向全向水听器、TOF差分距离估计、HMM+粒子滤波、速度与随机行走运动模型、二维距离校正以及测量丢失实验等技术。

**📊 数据集**

数据集包括在Long Beach、Santa Elena Bay 的实地实验记录，以及基于124k条白鲨轨迹的仿真数据。

**📈 对比分析**

与随机行走和速度模型对比，HMM模型在长时间仿真中RMSE为13.6 m，在实地实验中约10–15 m；即使测量丢失率高达80%，系统仍能收敛但时间与误差均显著增加。

**⚠️ 局限性**

主要局限包括需提前知晓初始距离、对深潜（>10 m）鱼类不适用、HMM模型收敛速度慢以及缺乏AUV间通信实现在线实时跟踪。

---

## 84. Load Testing for Machine Learning Model Serving Systems at Scale

**arXiv ID:** 2606.22013 | [PDF](https://arxiv.org/pdf/2606.22013v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 85. Old Fictions, New Skins: Evaluating the Manipulative Capabilities of LLMs in Healthcare

**arXiv ID:** 2606.21977 | [PDF](https://arxiv.org/pdf/2606.21977v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 86. OpenBioRQ: Unsolved Biomedical Research Questions for Agents

**arXiv ID:** 2606.21959 | [PDF](https://arxiv.org/pdf/2606.21959v1)

**作者:** Minbyul Jeong `[一作]` `[通讯]` (Upstage AI), Minbyul Jeong (Upstage AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 OpenBioRQ，一个基于检索校正、针对无答案键的未解医学研究问题的基准，用来评估具工具调用能力的代理模型在文献检索、引用与可信度上的表现。

**💡 创新点**

创新点：①将未解问题与代理模型结合，突破传统已知答案评测的局限；②通过检索‑校正的“开放性验证器”消除源头框架带来的确认偏差；③构造了冻结的问答检查清单，实现对多轮工具调用结果的客观打分；④揭示了“错误论文引用”这一在医学领域更具危害的错误模式，首次系统量化其发生率；⑤发现“代理崩溃”现象，即在最难问题上模型倾向于不使用工具，导致性能下降。

**🔧 技术方法**

技术手段包括：多轮（最多10轮）工具调用框架，使用十个公开生物医学 REST API；两层引用真实性审计（L1存在性、L2内容支持）；冻结问答检查清单与 LLM 判断器；检索‑校正的开放性判定；对模型进行交叉族判断与重编码稳定性检验。

**📊 数据集**

数据集：共 12,553 个未解医学研究问题，来源涵盖 PubMed / 临床试验 / arXiv、JLA 优先设定、NICE 研究建议、WHO/CHNRI/NASEM/Delphi 文献以及 Cochrane 研究缺口；每题都附带检索‑校正后的开放性标签、经验难度标签和冻结的检查清单。

**📈 对比分析**

对比方法：在同一问题集合上评测多种开放权重模型（GLM‑5.1、Qwen3.6、DeepSeek‑V4）与前沿模型（Gemini‑3‑Pro、Opus‑4.7、GPT‑5.5），以及与传统闭式医学 QA 公开数据集（MedQA 等）的性能差距。结果显示：在冻结核心 423 题上，前沿模型的解决率在 29–60% 之间，表现出明显的可区分性；而传统闭式 QA 的分数在 89–94% 之间相对饱和，无法区分模型；错误论文引用率约 15.9%（不同族评判一致）。

**⚠️ 局限性**

局限性：①错误论文引用率的评判仍基于 LLM 判断，缺乏专家标注的最终确认；②核心难度基于单一 T=0 采样，核心成员易漂移，未形成稳定的“保留率”；③模型的工具使用效果在部分模型上未显示显著提升，可能与工具选择或使用策略相关；④开放性验证器仅基于检索结果，无法捕捉未检索到的真相变化；⑤仅在生物医学领域验证，跨领域推广需进一步探索。

---

## 87. CFAgentBench: A Reproducible Environment and Benchmark for Autonomous Construction-Finance Agents

**arXiv ID:** 2606.22000 | [PDF](https://arxiv.org/pdf/2606.22000v1)

**作者:** Rishi Srivastava `[一作]` `[通讯]` (Beiing Human), Rishi Srivastava (Beiing Human)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个可复现、可自托管的构造金融领域 LLM 代理基准 CFAgentBench，包括 35 个模拟应用、1,014 条基于真实案例的任务规范。

**💡 创新点**

引入了“金钱移动保护器”与多系统、跨平台状态差异评估，以及基于功能正确性的层级分级判定，聚焦安全、可重复的业务执行。

**🔧 技术方法**

利用可编程的统一应用契约、AppWorld 风格的状态差异检查、τ-bench 的政策文件约束和 SWE-bench 的泄漏防护，配合 ReAct 交互式工具调用框架。

**📊 数据集**

任务来自 CFMA 连接咖啡讨论、NAHB 论坛、Finance at the Jobsite 播客、公开标准等，共 1,014 条实例，已划分公共/私有 split。

**📈 对比分析**

通过 pass^1 与 pass^k（k=5）可靠性指标对三种开放权重模型进行基准测试，DeepSeek 最高 0.67/0.38，表现出可靠性衰退；模型间排名在统计误差范围内，且在 Billing、Project Accounting 等域表现差异明显。

**⚠️ 局限性**

局限在于仅评估开放权重模型、未执行真实金钱转移、仿真层级为 Tier‑A、部分领域样本量不足以及对模型可解释性和高保真度的进一步验证尚待。

---

## 88. Reinforcement Learning-Based Traffic Signal Control for IoT-Enabled Intersections

**arXiv ID:** 2606.22108 | [PDF](https://arxiv.org/pdf/2606.22108v1)

**作者:** Yousef AlSaqabi `[一作]` `[通讯]` (Kuwait University), Yousef AlSaqabi (Kuwait University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文开发了一种基于强化学习的交通信号控制器，利用Proximal Policy Optimization（PPO）在SUMO仿真环境中训练，针对海湾地区（科威特）单个四路交叉口的绿灯时长进行自适应调度，训练数据来自实际路面传感器的小时流量计数。

**💡 创新点**

创新点在于首次在真实海湾城市交叉口上验证RL控制的效果，证明其在无未来需求信息、低车联网渗透率环境下仍能显著提升交通效率；同时系统性评估了鲁棒性、跨日泛化性和奖励函数对性能的影响，为IoT基础的智能交通提供了可部署的前沿技术。

**🔧 技术方法**

技术实现采用Proximal Policy Optimization（PPO）强化学习算法，使用两层多层感知网络（MLP），奖励函数为吞吐量与队列/等待时间的加权组合；仿真平台为SUMO，环境观测包含车道级排队长度、累计等待时长及当前绿灯状态。

**📊 数据集**

数据集为科威特内政部提供的固定路面传感器小时车辆计数，以及基于OpenStreetMap绘制的真实交叉口几何结构；该数据集未公开，仅通过官方渠道获得。

**📈 对比分析**

性能评估方法对比固定时长与车辆驱动自适应两种基线，指标包括平均车辆延时、平均队列长度、通行量以及CO₂排放/燃油消耗；实验结果显示PPO控制器将平均延时降低约46%（相对固定时长）和34%（相对自适应），队列长度同步下降，CO₂排放降低约23%；在±15%需求扰动和周末交通模式下仍保持显著优势；奖励函数消融实验表明同时包含等待时间与队列长度惩罚是实现稳定性能的关键。

**⚠️ 局限性**

局限性包括仅评估单一孤立交叉口，未考虑网络级联效应；仿真基于默认车辆动力学参数，缺乏现场验证；训练与测试均使用相同道路几何，无法证明跨网络迁移能力；此外，缺乏对传感器噪声和通信延迟的深入分析。

---

## 89. Prefix-Guided On-Policy Distillation: Mining Golden Trajectories from Rollouts

**arXiv ID:** 2606.21994 | [PDF](https://arxiv.org/pdf/2606.21994v1)

**作者:** Qingfei Zhao `[一作]` (TeleAI), Xuelong Li `[通讯]` (TeleAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Prefix-Guided On-Policy Distillation（PG-OPD），通过先生成固定长度前缀，计算教师与学生在前缀上的top-k重叠来评估轨迹价值，并只对高价值轨迹继续生成长回合，从而减少无用的计算并提升数学推理模型性能。

**💡 创新点**

创新点在于①使用前缀级兼容性（top-k overlap）作为轻量级轨迹价值评估；②基于此评分实现自适应的rollout分配（至少保证每个提示保留一条长续航）；③在不改变原始OPD目标的前提下，只改进采样与生成流程。

**🔧 技术方法**

技术包括On-Policy Distillation、top-k重叠计算、固定前缀截断、预算分配策略（Prompt-Minimum allocation）、token-level KL损失和多种实验ablations。

**📊 数据集**

使用了AMC、AIME、HMMT五个数学推理基准，并在训练阶段使用DAPO-Math-17K数据集。

**📈 对比分析**

与标准OPD和PRUNE-OPD比较，PG-OPD在所有教师-学生组合下平均准确率提升最多4.80分，训练时间加速最多2.46×，PRUNE-OPD虽然速度更快但准确率下降。

**⚠️ 局限性**

局限性包括：依赖前缀top-k重叠作为轨迹价值的可靠性，可能在教师与学生的推理风格或分词方式差异较大时效果减弱；需要调节前缀长度、预算等超参；实验仅覆盖数学推理任务，未验证多轮对话、工具使用或代理场景的适用性。

---

## 90. Rebuttals Move Peer-Review Scores, but Initial-Review Structure Bounds the Movement

**arXiv ID:** 2606.22166 | [PDF](https://arxiv.org/pdf/2606.22166v1)

**作者:** Mathieu Louis `[一作]` (Vrije Universiteit Brussel), Andres Algaba `[通讯]` (Vrije Universiteit Brussel)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文使用大语言模型（LLM）对ICLR 2024–2025的审稿记录进行测量，构建了一个三阶段的评估协议，探究作者回应（反驳）对审稿评分变动的可测量影响。

**💡 创新点**

创新点在于：①提出只做测量、不可评判的LLM工作流（score‑decoder、exchange‑feature taxonomy）；②利用外部存档的前后评分作为客观标签；③通过跨模型（Claude Opus 4.6→Gemini Flash 3.0）和跨年份验证的方式，得到可复现的44个交流特征；④量化作者参与的选择边界以及在已参与样本中预先可预测的评分提升。

**🔧 技术方法**

主要技术包括：Gemini Flash 3.0 的score‑decoder（从去掉评分字段的评审文本中恢复预反驳评分）；Claude Opus 4.6 的“induce‑and‑validate”流程生成并验证44个交流特征；逻辑回归、随机森林等传统机器学习模型进行AUC、PR曲线评估。

**📊 数据集**

使用的数据集为ICLR 2023–2026的完整OpenReview记录，结合Kargaran等人整理的ICLR 2024–2025前后评分档案，共计约73,000条匹配评审轨迹，其中6,705条形成三阶段预测基准。

**📈 对比分析**

与传统的仅使用结构化评审元数据的基线相比，H0（仅初始评审）AUC≈0.747，加入交流特征H1后AUC提升至≈0.804，额外加入讨论阶段H2仅提升≈0.005。实验在时间拆分（2024→2025）和论文聚类交叉验证中均保持稳健，表明交流特征提供了有界但可复制的预测增益。

**⚠️ 局限性**

局限性包括：①仅在ICLR会议上验证，未检验跨会议泛化；②依赖闭源LLM API，缺乏完全可重复的开放实现；③特征标签由LLM共同生成并验证，未与人工标注金标准对照；④样本受作者参与的选择偏差限制，不能直接解释原因；⑤评分尺度不连续且受限，可能影响对评分变动的解释。

---

## 91. Probabilistic Model Checking via Families of Deterministic and Unambiguous Finite Automata

**arXiv ID:** 2606.21976 | [PDF](https://arxiv.org/pdf/2606.21976v1)

**作者:** Christel Baier `[一作]` (Technische Universität Dresden), Timm Spork `[通讯]` (Technische Universität Dresden)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了针对离散时间马尔可夫链（DTMC）的概率模型检测算法，能够在多项式时间内处理由饱和FDFA（Families of Deterministic Finite Automata）表示的ω-正则性质；

**💡 创新点**

创新点在于引入了FUFA（Families of Unambiguous Finite Automata）这一新的模型，允许领先NFA与无歧义进展UFA，从而实现比FDFA更高的紧凑度，并给出了单指数的LTL→FUFA转换；

**🔧 技术方法**

技术上主要采用了FDFA的饱和化与BSCC（底层强连通分量）分析相结合的算法，利用产品DTMC与FDFA/ FUFA 的交叉构造来判定接受性，并通过UFA/UBA 的概率求解方法扩展到dFUFA；

**📊 数据集**

本文并未使用公开数据集，而是以理论构造与证明为主，未做实验评估；

**📈 对比分析**

与传统的基于确定性ω-自动机（如DRA、DPA）的模型检测方法相比，该算法在复杂度上保持多项式，但未给出具体性能数值；

**⚠️ 局限性**

主要局限在于对一般FUFA的概率模型检测仍无法给出多项式时间解法，且对FUFA→UBA或NBA→FUFA 的转换复杂度尚未完全确定。

---

## 92. Where Does the Signal Live? A Web Data Recipe for Medical Encoder Pretraining

**arXiv ID:** 2606.22079 | [PDF](https://arxiv.org/pdf/2606.22079v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 93. When Does a Video-Language Model Stop Watching? Reward Strength Controls the Formation and Reversal of Visual Shortcuts in Multimodal RLVR

**arXiv ID:** 2606.22043 | [PDF](https://arxiv.org/pdf/2606.22043v1)

**作者:** Zekun Xu `[一作]` `[通讯]` (University of Sydney), Zekun Xu (University of Sydney)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了多模态强化学习（RLVR）中视觉短路的形成与消除过程，并探讨了基于奖励强度的正则化干预时机。

**💡 创新点**

提出了将奖励中的“着地惩罚”视为可调节阈值，揭示视觉短路具有尖锐的启动、剂量响应与时间不对称的特性，并发现存在关键的干预窗口。

**🔧 技术方法**

使用了基于视频-语言的Qwen3‑VL‑8B‑Instruct模型、GRPO‑style RLVR、扰动诊断（时间帧随机打乱）和隐藏层表征分析等技术。

**📊 数据集**

在Qwen3‑VL‑8B‑Instruct上进行实验，使用了一个持出式、分布外的视觉诊断集来评估视觉短路。

**📈 对比分析**

通过在不同惩罚强度、不同训练阶段以及不同随机种子下的多条轨迹进行对照，发现视觉短路的“onset”在训练步骤16–24之间迅速出现；中等剂量可导致先形成后消退；早期施加惩罚能显著抑制短路，而后期干预效果有限。

**⚠️ 局限性**

研究仅在单一模型架构和单一惩罚设计上验证，未检验模型规模或其他视觉语言模型的可推广性；诊断仅测量时间帧打乱的敏感度；内部表征分析仅为相关性探索，缺乏因果证明。

---

## 94. MindTailor: Personalized Emotional Support via Post History-Grounded Case Formulation and Collaborative Refinement

**arXiv ID:** 2606.21930 | [PDF](https://arxiv.org/pdf/2606.21930v1)

**作者:** Suhyun Han `[一作]` (Sungkyunkwan University), JinYeong Bak `[通讯]` (Sungkyunkwan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种利用求助者历史记录构建案例化方案并通过多代理协同批判迭代优化的个性化情绪支持生成框架。

**💡 创新点**

创新点在于结合案例化方案与多视角协同批判，使响应既考虑历史背景又兼顾认知、情感、行为三重视角。

**🔧 技术方法**

使用了多代理生成模型、案例化方案构建、迭代批判与整合指导等技术。

**📊 数据集**

使用了由798条Reddit情绪支持帖及其一年内历史记录组成的新数据集。

**📈 对比分析**

与基线模型（Vanilla、MentalAgora、ES‑VR）在LLM-as-a-Judge、专家评测及求助者用户研究中对比，取得最高的同理心、个性化和整体偏好。

**⚠️ 局限性**

局限包括数据受COVID-19影响、仅采用三种治疗视角、计算开销大、案例化方案可能出现误引、未覆盖多平台。

---

## 95. Energy Trading Potential Index for a Peer-to-Peer Smart Grid Community with Flexible Prosumer Role Switching

**arXiv ID:** 2606.22087 | [PDF](https://arxiv.org/pdf/2606.22087v1)

**作者:** Zain Imran `[一作]` (Lahore University of Management Sciences), Naveed Ul Hassan `[通讯]` (Lahore University of Management Sciences)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了能量交易潜力指数（ETPI），用于评估灵活角色切换对社区级P2P能源交易潜力的结构影响。

**💡 创新点**

创新点在于把角色切换作为可设计的变量，引入ETPI进行量化评估，首次揭示在prosumers占比高的社区交易潜力不会崩溃，并通过二分图交互模型和归一化函数实现对交易质量的细粒度度量。

**🔧 技术方法**

使用基于广义二分图的交互函数（价格兼容性、需求满足度、网络可靠性）和tanh归一化计算交易潜力；通过Python实现仿真与ETPI计算；同时分析角色切换下的边缘增益与最优供应者数。

**📊 数据集**

使用PRECON住宅负荷数据（单户一年内的小时负荷）和NREL PVWatts产生的太阳能发电曲线，配合随机扰动模拟不同安装容量。

**📈 对比分析**

通过比较灵活与静态角色政策，利用边缘计数、ETP和ETPI指标评估性能；在1:9混合下，灵活政策的ETPI从0.15提升至0.61，ETP提升四倍以上，显示显著的交易潜力提升。

**⚠️ 局限性**

局限性包括未考虑储能、需求响应等进一步行为；假设每户供给概率独立且网络可靠性为单一随机变量；仅在一小时区间内模拟，缺乏更复杂的动态定价和实际市场约束。

---

## 96. Meta-Reinforcement Learning via Evolution for Multi-Objective Combinatorial Supply Chain Optimisation

**arXiv ID:** 2606.22146 | [PDF](https://arxiv.org/pdf/2606.22146v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 97. Investigating The Security of Modern AI and Cloud Infrastructure

**arXiv ID:** 2606.22237 | [PDF](https://arxiv.org/pdf/2606.22237v1)

**作者:** Andrew Adiletta `[一作]` `[通讯]` (Vernam Lab), Andrew Adiletta (Vernam Lab)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统梳理并验证了在现代云与AI基础设施中，攻击者可通过共享内存、共享硬件资源以及仅通过服务接口三类交互层次，对AI模型及其基础设施实施的多层级攻击。作者实现并评估了从共享缓存侧信道泄露LLM输出、利用Rowhammer诱发相邻位翻转以篡改Tokenizer及其他模型参数、在寄存器与栈上注入错误导致身份验证绕过、在程序计数器上执行指令跳转、以及联合优化生成的Super Suffixes攻击在Guard Model与生成模型同时绕过的攻击等，并提出DeltaGuard基于内部状态轨迹的检测与防御方法。

**💡 创新点**

创新点包括：①首次在LLM推理中实现Flush+Reload对embedding层的硬件级侧信道攻击，直接泄露高熵API密钥和常用文本；②揭示相邻位翻转的物理机制并应用于GGUF Tokenizer损坏，重写系统提示以绕过安全门槛；③将Rowhammer扩展到寄存器/栈、程序计数器等CPU内部状态，突破传统只攻击DRAM的范式；④提出Super Suffixes联合优化框架，在同一输入下同时骗过Guard Model和生成模型的对齐检查；⑤研发DeltaGuard，将内部残差流余弦相似度作为恶意输入的特征，实现对联合优化攻击的鲁棒检测。

**🔧 技术方法**

技术手段涵盖：Flush+Reload、Prime+Probe/Scope（对比）、Rowhammer多侧（TRRespass、BlackSmith、Half‑Double）、缓存行清除（clflush）、高精度计时（RDTSC）、GPU/CPU统一内存协同、共享页面与KSM利用、指令跳转(bit‑flip在返回地址上)、联合优化（GCG、ARCA、AutoPrompt等）、内部状态轨迹分析与余弦相似度检测、LLM模型的GGUF文件解析与token‑offset映射。

**📊 数据集**

数据集与模型：Llama 3‑8B、Meta GGUF模型；随机生成的100k UUID用于API键泄露评估；Quora问题集用于单轮英文文本泄露；Cornell Movie Dialogs corpus用于英文词频分析；此外还使用了公开的GPT‑4/Meta LLM接口进行对照实验。

**📈 对比分析**

评估方法：对每种攻击设置多种监控规模（token数50~400）并测量泄露率；对API键泄露以完整键获得率为指标，单轮可达80‑90%，多轮可接近100%；对英文文本泄露以漏出token比例为指标，最佳监控规模为150‑250 token，单轮可泄漏约30‑40%；对Rowhammer及寄存器/堆栈攻击以成功绕过验证、修改模型权重或跳过指令为指标，实验成功率>95%；与现有侧信道、故障注入与对齐攻击对比，显示本工作在攻击覆盖率、硬件依赖度与攻击一次性成功率上均优于传统方法。

**⚠️ 局限性**

局限性：①攻击依赖攻击者与受害者共享物理CPU/内存，需在同一主机或可共存虚拟机；②在未启用统一内存或GPU独立内存的环境下，embedding侧信道难以监测；③Flush+Reload在大模型下受缓存淘汰影响，需平衡监控规模与时延；④对高熵密钥的单次泄露仍不确定，需多轮交互或自适应监控；⑤Rowhammer和寄存器/堆栈攻击受硬件防护（TRR、ECC）影响，部分芯片已降低易受攻击率；⑥DeltaGuard等防御对新型攻击模型适应性不足，需进一步泛化。

---

## 98. L20-Edu-135M: An Auditable Single-GPU Study of Data-Efficient Small Language Modeling

**arXiv ID:** 2606.22189 | [PDF](https://arxiv.org/pdf/2606.22189v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 99. TraceView: Interactive Visualization of Agentic Program Repair Trajectories

**arXiv ID:** 2606.22110 | [PDF](https://arxiv.org/pdf/2606.22110v1)

**作者:** Amirali Sajadi `[一作]` (Drexel University), Preetha Chatterjee `[通讯]` (Drexel University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了 TraceView，一款交互式工具，用于对 LLM 驱动的自动程序修复（APR）代理的思考（Thought）、行动（Action）和结果（Result）轨迹进行标注、可视化和分析。

**💡 创新点**

创新点在于将轨迹拆解为 TAR 结构并引入语义关系标注，提供迭代视图和详细视图的双模式图形、关系过滤器、指标统计以及节点级证据面板，实现对代理行为的细粒度诊断与可视化。

**🔧 技术方法**

技术实现包括基于 Streamlit 的 Web 应用、JSON/CSV 轨迹解析、Graphviz/NetworkX 等图形渲染、交互式过滤与弹窗预览、关系标签映射与度量计算。

**📊 数据集**

使用的数据集包括 AutoCodeRover、SWE-Agent 等公开的 APR 轨迹日志，以及本地收集的原始 JSON、JSONL、文本日志等多种格式的轨迹文件，全部统一为 TAR 规范。

**📈 对比分析**

通过与原始日志对比的问卷调查法评估，5 名参与者在使用 TraceView 后在轨迹扫描效率、信息可用性和整体满意度方面均获得 4–5 分的正向评价，证明工具在帮助理解代理过程方面具有显著优势。

**⚠️ 局限性**

局限性包括节点标签缺乏简短摘要、弹窗预览易于信息过载、样本量有限（仅 5 名用户、少量轨迹）、对更复杂或更长轨迹的适配尚未充分验证。

---

## 100. A Controlled Study of CLIP-Based Body-Scene Fusion for Emotion Recognition in Context

**arXiv ID:** 2606.22072 | [PDF](https://arxiv.org/pdf/2606.22072v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 101. The Cognitive Trajectory Laboratory: Modeling the Creative Process Through Time in Art Therapy

**arXiv ID:** 2606.22057 | [PDF](https://arxiv.org/pdf/2606.22057v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 102. Full Nonlinear Nonholonomic Dynamics and Motion Analysis of a 3-DoF Underactuated Spherical Rolling Robot

**arXiv ID:** 2606.22169 | [PDF](https://arxiv.org/pdf/2606.22169v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 103. A Completion-Aware Framework for Impactful Counterfactual Explainability in Graph Neural Networks

**arXiv ID:** 2606.22033 | [PDF](https://arxiv.org/pdf/2606.22033v1)

**作者:** Maria Myrto Villia `[一作]` (Foundation for Research and Technology - Hellas), Panos Trahanias `[通讯]` (Foundation for Research and Technology - Hellas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出基于链接预测的局部级、模型无关的反事实可解释框架DR‑CFGNN，包含去噪、分解、重构和后处理四阶段。

**💡 创新点**

创新点在于将链接预测模型嵌入反事实生成，采用分解‑重构策略、类条件链接解码器、可配置采样与后置优化，显著提升解释质量、紧凑性和鲁棒性。

**🔧 技术方法**

技术包括GCN编码器+MLP解码器的链接预测网络、SubgraphX事实子图生成、基于概率采样的边编辑、cos²权重的后置评分、负采样训练等。

**📊 数据集**

使用的数据集有合成的BA‑2/3/4 Motifs、BA‑2Motifs‑3classes、真实的Twitter、Graph‑SST5、BBBP。

**📈 对比分析**

与CF^2、D4Explainer、RSGG‑CE、GIST、GCFExplainer和Random baseline在有效性、解释尺寸、可信度、图元接近度、最小化、鲁棒性和时间等多指标上对比，DR‑CFGNN在多项指标上优于或与最佳基线相当，解释尺寸更小、最小化和图元接近度高，训练与推理时间低。

**⚠️ 局限性**

局限性包括：对事实子图生成的效率依赖，难以处理极大图；类条件编码需先验标签，缺少对动态/时序图的适配；在高噪声或复杂任务中的鲁棒性仍有提升空间。

---

## 104. Dual-Stream EEG Decoding for 3D Visual Perception

**arXiv ID:** 2606.22182 | [PDF](https://arxiv.org/pdf/2606.22182v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 105. A11YRepair: Bridging Web Accessibility Barriers via Knowledge-Enhanced Divide-and-Conquer Repair

**arXiv ID:** 2606.21926 | [PDF](https://arxiv.org/pdf/2606.21926v1)

**作者:** Kai Huang `[一作]` (Technical University of Munich), Chunyang Chen `[通讯]` (Technical University of Munich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 A11YRepair，一个基于大语言模型（LLM）的 Web 可访问性（A11Y）自动修复框架，能够在项目级别识别并修复多样且多重的可访问性违规，并通过分治策略将违规聚类、定位并生成一致的补丁。

**💡 创新点**

创新点包括：① 采用分治（divide‑and‑conquer）策略，将违规按组件和情境层次聚类，避免单独修复导致的冗余与冲突；② 引入“定位反射”（Locate Reflection）机制，让 LLM 多轮验证定位结果，提升准确性；③ 采用“可访问性驱动知识集成”（WCAG‑Driven Knowledge Integration），通过先评估是否需要 WCAG 指南，再选择性检索相关成功准则和技术，减少上下文噪声和成本；④ 构建首个 repo‑级 Web A11Y 修复基准 A11YBench，包含 60 个真实 GitHub 项目和 8,886 条违规。

**🔧 技术方法**

技术手段包括：基于 GPT‑4o‑mini 等 LLM 的聊天推理与嵌入检索；视觉上下文（截图+DOM 结构）与属性级别特征结合的定位；分层聚类（区域、组件、准则、情境）与 LLM 细化；结构化补丁生成（search/replace 格式）与增量验证；以及对 WCAG 成功准则与技术的检索与使用。

**📊 数据集**

使用的数据集为 A11YBench，分为 Lite（10 项）和 Full（60 项）两部分，包含 147 个页面、8,886 条违规、45 种违规类型，来自 IBM Accessibility Checker 检测。还利用了公开的 Web A11Y 检测工具 IBM A11Y Checker 作为弱 Oracle，人工审核与 PR 合并记录作为进一步验证。

**📈 对比分析**

与最强基线 GUIRepair（Basic、Iterative、WCAG‑Guided）以及其他 APR 系统（SWE‑agent、OpenHands）对比，A11YRepair 在 A11YBench‑Full 上实现 76.82% 的违规解决率，较 GUIRepair_I 提升约 16.7%，侧效降低 44.5%，Token 成本降低 55%。在 Lite 上通过 ablation 证明分治细粒度、LLM 细化、定位反射、WCAG 选择性集成等各项设计显著提升效果。实验还表明 A11YRepair 对不同基础模型（Gemini‑3.0‑flash、Kimi‑K2.5）和新项目的泛化能力良好。

**⚠️ 局限性**

限制包括：依赖静态检查器（可能存在误报或漏报），无法处理动态交互导致的可访问性问题；对视觉状态（如悬停、激活）导致的细微颜色/布局差异修复效果有限；当前仅支持前端 Web（React/TSX 等），不涵盖移动端或其他框架；若 WCAG 规则与项目视觉设计冲突，仍需人工判断；LLM 生成补丁仍可能引入新的隐藏回归，需要人工复核。

---

## 106. ScalePredictor: Instance-aware Scale Learning for Accurate Quantization of Vision Transformers

**arXiv ID:** 2606.21947 | [PDF](https://arxiv.org/pdf/2606.21947v1)

**作者:** Changjun Li `[一作]` (Sun Yat-sen University), Yulan Guo `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 ScalePredictor，一种针对视觉 Transformer 的后训练量化方法，利用浅层激活范围预测每个样本的量化尺度，从而实现实例感知的动态量化。

**💡 创新点**

核心创新在于发现浅层激活范围与深层最优量化尺度之间存在可建模的全局相关性，并通过稳健的分块极值提取与 Taylor 推导的多项式预测器实现一次性、低成本的尺度生成。

**🔧 技术方法**

使用的技术包括：分块极值平均提取鲁棒范围、基于 Taylor 展开的多项式尺度预测器、基准数据的直通估计器 (STE) 训练、以及与多种现有 PTQ 框架的无缝集成。

**📊 数据集**

在 ImageNet 训练/验证集上进行实验，评估多种 ViT 变体（ViT‑S/B、DeiT‑T/S/B、Swin‑S/B）在不同量化位宽下的性能。

**📈 对比分析**

与静态 PTQ 基线（如 BRECQ、QDrop、AdaLog、I&S‑ViT）以及即时动态量化（JIT Min/Max/Percentile）对比，ScalePredictor 在低精度（W2/A3）下可提升至 +10.58% 的 Top‑1 准确率，W3/A3/ W4/A3 也可持续提升 3–8%；同时推断时的延迟仅比静态 PTQ 低 0.6%，几乎不增加算力负担。

**⚠️ 局限性**

局限性包括：在某些基线（如 QDrop）可能出现小幅回退；高阶多项式易在校准样本不足时过拟合；仅针对激活量化，权重量化需单独处理；模型对浅层范围相关性的假设在其他网络结构或任务上需进一步验证。

---

## 107. Learning by Shifting: Temporal View Construction for Time Series Contrastive Learning

**arXiv ID:** 2606.21957 | [PDF](https://arxiv.org/pdf/2606.21957v1)

**作者:** Abdul-Kazeem Shamba `[一作]` (Norwegian University of Science and Technology), Gavin Taylor `[通讯]` (United States Naval Academy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用自监督对比学习训练时间序列表示，采用确定性时间移位视图生成正样本，构建ShiFT框架。

**💡 创新点**

创新点在于仅通过简单的时间移位视图而非复杂的增强或掩蔽策略，实现了对时间序列的强泛化和高效训练，证明移位即为足够的先验。

**🔧 技术方法**

技术包括ShiFT（Shift Invariant Feature Training）算法，使用InceptionTime编码器、MLP投影头以及NT‑Xent对比损失；同时通过确定性重叠窗口拆分生成正样本。

**📊 数据集**

实验使用六大规模真实时间序列数据集（PAMAP2、WISDM2、HARTH、SLEEP、ECG、SKODA）以及UCR和UEA存档的多类别时间序列数据。

**📈 对比分析**

与SimCLR、TS2Vec、InfoTS、SimMTM等基线对比，ShiFT在线性探针、kNN、聚类等下游任务上均取得领先表现，同时训练时间为所有方法中最低。

**⚠️ 局限性**

局限性在于仅依赖时间移位假设，可能在需要更复杂语义变换或高类内相似度导致负样本误标的场景下表现不佳，且对非平稳、跨域差异大的时间序列可能受限。

---

## 108. Provably Efficient Policy-Reward Co-Pretraining for Adversarial Imitation Learning

**arXiv ID:** 2606.22056 | [PDF](https://arxiv.org/pdf/2606.22056v1)

**作者:** Tian Xu `[一作]` (Nanjing University), Yang Yu `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并验证了一种联合策略‑奖励共预训练方法CoPT-AIL，用于加速对抗式模仿学习；

**💡 创新点**

① 通过理论分析揭示奖励误差是AIL中策略预训练的主要瓶颈；② 证明专家策略的对数概率即为一种潜在奖励，从而可仅用一次BC同时预训练策略与奖励；③ 给出CoPT-AIL的理论上限证明，首次正式证明预训练能降低AIL的模仿误差；

**🔧 技术方法**

对抗式模仿学习框架、KL正则化策略更新、奖励形状理论、BC最大似然、模拟引理、在线到批量转换、DrQ‑v2等强化学习与深度学习技术；

**📊 数据集**

8个基于Feature‑based DMControl的连续控制任务；

**📈 对比分析**

与BC、IQLearn、PPIL、FILTER、HyPE、OLLIE等基线在相同环境下对比，CoPT-AIL在所有任务上达到或超过现有最先进AIL方法的收敛速度，尤其在部分任务上显著减少所需环境交互量；消融实验表明单独策略预训练对收敛无显著提升；

**⚠️ 局限性**

理论证明仅覆盖表格/离散设置，未扩展到函数逼近；在更复杂或视觉输入的机器人任务中验证有限；仍需高质量专家演示，奖励形状假设在部分环境下可能不严谨。

---

## 109. Zero-shot Transfer of Reinforcement Learning Control Policies for the Swing-Up and Stabilization of a Cart-Pole System

**arXiv ID:** 2606.22145 | [PDF](https://arxiv.org/pdf/2606.22145v1)

**作者:** Nikki Xu `[一作]` (NC State University), Hien Tran `[通讯]` (NC State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了零射击转移的强化学习控制策略，用于卡特-摆系统的摆起与稳定，并在实验室硬件上直接部署；

**💡 创新点**

结合动作低通滤波、敏感性导向的域随机化和线性课程学习，首次实现无需后期微调即可将仿真训练出的策略迁移到物理硬件，并提出合适的切换逻辑；

**🔧 技术方法**

使用TD3与REINFORCE强化学习算法、第一阶低通滤波、域随机化、课程学习、复杂步法敏感性分析、Simulink硬件‑in‑loop、分支控制；

**📊 数据集**

自定义的卡特-摆仿真环境，随机采样模型参数与状态/动作噪声，并在实验室硬件上进行多次实验；

**📈 对比分析**

通过在仿真中对三种训练配置（DR+CL、DR、无DR）进行1000次episode评估，比较平均奖励、标准差、到达时间和成功率。结果显示，对低敏感参数10%随机化并配合CL的策略在严重扰动下可实现100%成功率；而广泛随机化导致失效；实验室测试表明仅此配置能成功完成摆起并保持稳定；

**⚠️ 局限性**

实验结果依赖于特定硬件与参数，缺乏理论证明；训练奖励与硬件表现不一致；课程长度与随机化范围的最佳取值尚未确定；切换逻辑在某些配置下仍不稳定。

---

## 110. From Recognition to Understanding: Unlocking Cognitive Time Series Reasoning with LLMs

**arXiv ID:** 2606.22126 | [PDF](https://arxiv.org/pdf/2606.22126v1)

**作者:** Xin Qiu `[一作]` (Eastern Institute of Technology), Xiaoyu Shen `[通讯]` (Eastern Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了多模态时间序列推理基准 TSCognition 并设计了 TSAlign 框架实现时间序列与 LLM 的对齐与推理。

**💡 创新点**

创新点：1) TSCognition 提供多维长序列、5 类认知推理任务与文本信息的多模态数据；2) TSAlign 通过 patch 级编码、语义子空间对齐、门控残差注入、多维融合将时间序列映射至 LLM 语义空间，从而实现高效推理。

**🔧 技术方法**

采用了 patch 级时间序列编码、PCA 子空间对齐、门控残差注入、多维门控融合、LLM 语义投影等技术。

**📊 数据集**

使用了 15 公开来源的多域时间序列数据，构成 41,086 条 QA 样本（8 域、5 任务类型），以及 TimerBed 作为对比基准。

**📈 对比分析**

与 GPT‑5.1、Qwen、VLM、传统 TS 模型等基线在 TSCognition 及 TimerBed 上对比，TSAlign‑7B 在所有推理任务和模式分析任务上均优于基线，零/全量训练下分别提升约 30–50% 以上，显著降低 token 与算力成本。

**⚠️ 局限性**

局限性：仅采用多选题形式，无法覆盖开放式推理；数据集覆盖域虽多但仍需进一步扩展至更复杂、专家级任务。

---

## 111. The Score Granularity Gap in Black-Box LLM Classification: A Comparative Study of Confidence Constructions

**arXiv ID:** 2606.22179 | [PDF](https://arxiv.org/pdf/2606.22179v1)

**作者:** Ao Sun `[一作]`, Jiaxing Geng `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在黑盒大语言模型分类器的置信度分数中，系统地研究了“分辨率间隙”——即评分能够提供的阈值粒度，比较了七种置信度构造方法（单击口述信心、token概率、温度校准、多查询子任务/重述聚合及其平均或逻辑回归组合），并在25个模型-数据集对上评估其排名、分辨率、成本与可解释性。

**💡 创新点**

①提出并量化了score granularity gap，并给出了三个分辨率指标（G、H、M）；②系统比较多种置信度构造，发现多查询聚合可显著提升分辨率但在强模型上可能降低风险覆盖；③将可解释性视为部署属性而非排名优势，提供子任务式解释。

**🔧 技术方法**

置信度构造技术包括：口述置信度转换为类概率、token log概率与温度校准、10次多查询（子任务或重述）并以平均或LR聚合；评估技术包含PR‑AUC、PR‑AUC_N、AURC、覆盖率、分辨率指标，以及温度标定、L2正则逻辑回归。

**📊 数据集**

使用了三大二分类英文基准：BoolQ、MNLI、PubMedQA，每个约1000条样本，正负比例62–66%。

**📈 对比分析**

方法：在每个模型-数据集对上进行70/30拆分，计算各构造的排名指标、分辨率指标及调用次数；结果显示：单击口述置信度排名好但分辨率低；token log概率+温度校准在可用模型上既提升分辨率又保持良好排名；多查询聚合在弱模型（≈85%精度以下）可显著提升排名和分辨率，但在强模型（≥90%）可能导致AURC下降。子任务式聚合在保持排名的同时提供可解释性。

**⚠️ 局限性**

局限性：仅评估二分类英文任务，未覆盖极端类别不平衡、多类或生成式场景；未涉及其它方法如深度集成、对抗式或 conformal wrapper；子任务构造需手工设计且需要足够标注；token log概率在多数模型不可用，限制了该方法的普适性。

---

## 112. Cache-Aware I/O Cost Modeling for Disk-Based Learned Indexes

**arXiv ID:** 2606.21924 | [PDF](https://arxiv.org/pdf/2606.21924v1)

**作者:** Zhanwei Shi `[一作]` (Southwest University), Yingxia Shao `[通讯]` (Beijing University Of Posts And Telecommunications)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 CAM（Cache‑Aware I/O Model），一种面向磁盘学习索引的缓存感知 I/O 成本模型，能够在不重放完整查询的情况下估计有效物理 I/O，并以此进行索引调优与查询优化。

**💡 创新点**

创新点在于：①将学习索引的“最后一英里”页面访问分布与缓存命中率模型结合，得到闭式的物理 I/O 估计；②模型与索引实现无关，可为多种学习索引（PGM、RMI 等）提供统一的调优框架；③支持多种缓存策略（FIFO、LRU、LFU）并提供对排序工作负载的专用估计；④在学习索引基于连接时提出密度感知的混合探测（点探测/区间探测）策略。

**🔧 技术方法**

主要技术包括：
• 解析学习索引预测误差与页面窗口宽度，推导页面访问概率和期望访问次数；
• 在 IRM 框架下使用 Che 近似、FIFO 与 LFU 的闭式公式计算缓存命中率；
• 结合上述结果与设备级 I/O 模型（DAM、Affine 等）得到最终 I/O 成本；
• 对 PGM 采用幂律模型估计索引大小，对 RMI 通过叶子模型误差分布求期望访问；
• 在 join 过程中根据估计成本分段切换点/区间探测。

**📊 数据集**

实验使用四个公开数据集（books、fb、osm、wiki，200M 排序键），并在多种工作负载（点查、范围查、join、热点+Zipf+均匀混合）上评估。

**📈 对比分析**

与传统方法（Replay、LPM、基于逻辑页计数的 LPM）、现有调优器（多目标 PGM 优化器、CDFShop）以及 join 基线（INLJ、点/区间单独探测）比较：
• CAM 估计误差 Q‑error 接近 1，估算速度比 Replay 快 17~25 倍；
• PGM 调优通过 CAM 可提升吞吐 1.17×，调优时间下降 75.7%；
• RMI 调优提升吞吐 1.66×，调优时间下降 60.1%；
• 混合 join 策略在所有工作负载下比 INLJ 快 8.8×。

**⚠️ 局限性**

局限性：
• 依赖 IRM（独立引用模型）假设，排序工作负载或热点波动可能导致命中率估计偏差；
• 对 LFU 的长时间收敛假设在短期实验中误差较大；
• 只建模物理 I/O，未考虑 CPU/内存调度等瓶颈；
• RMI 需要对每个候选配置构造索引，导致调优时仍有一定构造开销；
• 在极端大页或非均匀页面分布场景下，页面窗口宽度近似可能失效。

---

## 113. Geometric Reconstruction of Extrinsic Contact Trajectories using Tactile Sensing and Proprioception for Tool Manipulation

**arXiv ID:** 2606.22251 | [PDF](https://arxiv.org/pdf/2606.22251v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 114. An Analysis of Untrained Deep Reservoir Networks for Audio Surveillance

**arXiv ID:** 2606.22218 | [PDF](https://arxiv.org/pdf/2606.22218v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 115. Feed-forward Motion In-betweening for Any 4D

**arXiv ID:** 2606.22131 | [PDF](https://arxiv.org/pdf/2606.22131v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 116. One-Shot Data Selection for Medical Image Classification via Graph Coverage

**arXiv ID:** 2606.22002 | [PDF](https://arxiv.org/pdf/2606.22002v1)

**作者:** Zahiriddin Rustamov `[一作]` (United Arab Emirates University), Nazar Zaki `[通讯]` (United Arab Emirates University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于冻结基础模型嵌入构建k‑NN图的单次数据选择方法，通过两项邻域传播的覆盖核进行贪心设施定位来挑选医疗影像分类训练子集。

**💡 创新点**

创新点在于只利用冻结嵌入构建全局k‑NN图并使用简化的两项多项式热扩散核，避免了全谱分解与训练动态的需求，并在跨类别图结构下显著提升不平衡数据选择效果。

**🔧 技术方法**

采用k‑NN图构建、对称归一化、邻域传播、两项覆盖核、贪心设施定位等图算法，并使用预训练的UNI ViT‑L/16嵌入。

**📊 数据集**

使用MedMNIST v2中的OrganAMNIST、OrganSMNIST、PathMNIST、TissueMNIST、BloodMNIST以及DermaMNIST等多模态医学影像分类数据集。

**📈 对比分析**

与随机采样、EL2N、遗忘事件、EVA等训练动态方法以及设施定位、FPS、Herding等几何基准对比，方法在2–5%样本比例下在四五个数据集上均取得最高或次高的平衡准确率，尤其在类别不平衡情况下提升显著。

**⚠️ 局限性**

主要限制是对嵌入质量的依赖，在UNI嵌入对荧光显微镜数据分辨不佳的TissueMNIST上，所有方法性能趋同，图结构无法弥补表征不足。

---

## 117. Channel Location Constrains the Auditability of Subliminal Learning

**arXiv ID:** 2606.22019 | [PDF](https://arxiv.org/pdf/2606.22019v1)

**作者:** Tamas Madl `[一作]` `[通讯]` (Austrian Research Institute for Artificial Intelligence), Tamas Madl (Austrian Research Institute for Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在模型蒸馏过程中，学生模型如何在无明确标记的情况下潜移默化地继承教师模型的隐性特质，并提出按通道载体位置划分的审计可行性地图，探讨不同场景下的预训练筛查、后测检测与干预手段。

**💡 创新点**

创新点在于：① 引入“coverage”度量（学生初始梯度与教师位移的余弦相似度）作为仅在初始化依赖体通道可用的预训练筛查；② 证明词汇表的“unembedding entanglement”是单词特质泄露的载体，并通过正交化消除；③ 发现条件行为（如顺从性）主要路由于网络体计算，导致现有筛查与干预均难以覆盖，构成新的安全盲区。

**🔧 技术方法**

技术手段包括：梯度对齐分析、Gauss‑Newton/NLP Fisher近似、正交化与邻居掩码/注入实验、头/体迁移、混合精度训练、BERT/Transformer架构的微调与蒸馏、以及多种对齐与可解释性测度（cosine、Spearman、AUROC）。

**📊 数据集**

使用的主要数据集有：随机噪声输入、自然文本提示、Pythia-410M 与 Qwen3.5-0.8B 预训练模型的内部词表，以及对单词/实体特质进行的人工掩码实验；此外还使用了标准的多种语义类（数字、动物、品牌等）与真实对话场景（sycophancy）测试。

**📈 对比分析**

通过在控制体通道的多噪声条件下进行覆盖率预测实验，证明覆盖率与传递准确率在 Spearman ρ≈0.95、AUROC≈0.997 上表现优异；在词表通道下，正交化操作将泄露概率降至 10⁻³ 级，且对比基线对齐度量显示覆盖率在此场景下失效；对条件行为的实验表明即使掩码掉标记，行为仍能传递约 0.63×教师效果，且无法通过词表正交化消除。整体上，各方法在各自适用的通道下取得了显著的可预测性与可操作性，但跨通道迁移存在显著限制。

**⚠️ 局限性**

局限性包括：① 预训练筛查“coverage”仅适用于初始化依赖体通道，无法对词表或条件行为通道进行通用检测；② 对词表通道的正交化需要先知晓泄漏词，无法实现无目标的通用屏障；③ 条件行为通道目前缺乏可检测与修复机制，需进一步研究训练管道治理；④ 研究主要聚焦在掩码蒸馏场景，未覆盖更一般的无监督或多任务蒸馏；⑤ 对模型规模与精度的稳健性虽然已部分评估，但在更大或不同架构上仍需验证。

---

## 118. Guarded Equivalence Predicates for Scalable Formal Hardware Information-Flow Verification

**arXiv ID:** 2606.22063 | [PDF](https://arxiv.org/pdf/2606.22063v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 119. Are Multilingual Models Actually Improving? Isolating True Cross-Lingual Transfer

**arXiv ID:** 2606.21954 | [PDF](https://arxiv.org/pdf/2606.21954v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 120. Surgical Anatomy Recognition with Context Learning using Foundation Representations

**arXiv ID:** 2606.22124 | [PDF](https://arxiv.org/pdf/2606.22124v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 121. Toward Open-Set Speaker Attribute Prediction with Keyword-Appended LLM Embeddings

**arXiv ID:** 2606.21979 | [PDF](https://arxiv.org/pdf/2606.21979v1)

**作者:** Byoungjun So `[一作]` (Seoul National University), Kyogu Lee `[通讯]` (Seoul National University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于大型语言模型嵌入的开放集说话人属性预测框架，利用关键词追加和 top‑k 负样本惩罚实现跨模态对齐。

**💡 创新点**

创新点在于将属性映射到连续语义空间，关键词追加压缩嵌入流形并增强结构性，top‑k 负样本损失显著提升区分度，实现零样本同义词泛化。

**🔧 技术方法**

使用 ECAPA‑TDNN 作为声学特征提取器，GPT‑OSS‑20B 作为属性嵌入，构造加权余弦相似度损失、softplus top‑k 负样本损失，并通过关键词追加正则化嵌入。

**📊 数据集**

主要采用 LibriTTS‑P 语料库进行训练与评估，使用 Gemini‑3.1‑Pro 生成属性同义词进行零样本验证。

**📈 对比分析**

与传统闭集多标签 BCE 方案对比，微平均 F1 在最佳阈值下提升至 0.7625（vs. 0.7286），零样本同义词预测也保持高精度，几乎无显著性能损失。

**⚠️ 局限性**

局限在于仅使用单一数据集（LibriTTS‑P）验证，未检验不同 LLM 嵌入或跨语料库的泛化能力，且对数据集与模型结构的依赖较强。

---

## 122. Cluster-Specific Localized Drift Detection for Efficient Batch Model Adaptation under Controlled Distribution Shift

**arXiv ID:** 2606.22026 | [PDF](https://arxiv.org/pdf/2606.22026v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 123. Learning Cross-View Semantic Priors for Single-Reference Unseen Object Pose Estimation

**arXiv ID:** 2606.22076 | [PDF](https://arxiv.org/pdf/2606.22076v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 124. REBA: A Revealed Belief Automaton Framework for Online Planning in Continuous POMDPs

**arXiv ID:** 2606.21971 | [PDF](https://arxiv.org/pdf/2606.21971v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 125. Lexical Consensus: Grounded Word Learning and Shared Meaning in Artificial Agents

**arXiv ID:** 2606.22207 | [PDF](https://arxiv.org/pdf/2606.22207v1)

**作者:** Patricio M. Vera `[一作]` `[通讯]` (Neurocreaciones), Patricio M. Vera (Neurocreaciones)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 Lexical Consensus 框架，用于评估人工智能系统在有限视觉示例下对新词汇的获取、双向使用以及在多智能体间的一致性。

**💡 创新点**

创新点在于将词汇获取视为可量化的实验过程，揭示了词汇学习随感知空间一致性而呈梯度的现象，并通过预注册的失配实验证明该梯度受感知几何而非语义关联驱动。

**🔧 技术方法**

技术上采用了冻结的 DINOv2 视觉嵌入、中心点、复中心点、k‑NN 和线性分类器等可解释词汇学习器，以及基于熵和互信息的多智能体共识度量。

**📊 数据集**

使用的主要数据集为 CIFAR‑10（10 类）和 CIFAR‑100（100 类）进行概念分层与失配实验，此外还通过多种候选池和随机控制来验证稳健性。

**📈 对比分析**

与随机、随机嵌入、打乱绑定等消融和基线对比显示，中心点学习器在命名任务中可达 94% 以上精度；在检索任务中，k‑NN 超越中心点，线性分类器在困难池中表现最优；多智能体共识在无反馈情况下已达 98% 以上一致性，反馈仅略有提升。

**⚠️ 局限性**

局限在于仅处理预先冻结的感知空间、单词级别无句法/语义层次、受限于 10/100 类视觉数据、缺乏属性/关系级别的组合语言、以及共识未能显著重塑内部表示。

---

## 126. Cross-View Yaw Estimation in Location Uncertainty with Line-Aligning Yaw Scoring

**arXiv ID:** 2606.22094 | [PDF](https://arxiv.org/pdf/2606.22094v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 127. Novelty-Aware Agentic Retrieval: Comparing Research Contributions Through Structured Multi-Step Reasoning

**arXiv ID:** 2606.22151 | [PDF](https://arxiv.org/pdf/2606.22151v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 128. Plurification in/of language technology -- The integration of culture in next-generation AI

**arXiv ID:** 2606.22097 | [PDF](https://arxiv.org/pdf/2606.22097v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 129. CodeTeam: An LLM-Powered Multi-Agent Framework for Repository-Level Code Generation

**arXiv ID:** 2606.22082 | [PDF](https://arxiv.org/pdf/2606.22082v1)

**作者:** Yifei Wang `[一作]` (Wuhan University), Arif Ali Khan `[通讯]` (University of Oulu)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于大型语言模型的多智能体框架 CodeTeam，用于从自然语言需求生成完整的代码仓库；

**💡 创新点**

通过拆分规划、决策、实现三阶段，将架构草图（SDS）作为可检验合同；引入架构师竞争、可选检索增强、CTO 合同规范化、动态开发者分配、依赖感知调度、轻量 Git 协作与 QA 驱动修复等多重机制；

**🔧 技术方法**

使用 Qwen2.5‑72B 语言模型（提示工程与参数高效微调）；多智能体交互、检索增强生成（RAG）、长文本上下文扩展（LongLoRA）、Git 版控沟通、质量保证测试回调等技术；

**📊 数据集**

评估数据集包括：NL2Repo‑Bench（19 个 Python 仓库）用于 SketchBLEU 评估；104 任务的外部执行测试基准用于通用通行率评估；检索语料库来自公开 GitHub Python 项目；

**📈 对比分析**

对比 Vanilla、ChatDev、AutoGPT、CodeS（PE 与 SFT 两种实现）等基线；在 SketchBLEU 上 CodeTeam‑PE 达到 51.7%（比 CodeS‑PE 高 4.1 点），CodeTeam‑SFT 60.9%（比 CodeS‑SFT 高 2.9 点）；在 104‑任务执行基准上，PE 版 34.6% 通过率，SFT 版 42.3%，均居领先；

**⚠️ 局限性**

仍存在：通过率相对较低，难以完全满足原始测试套件；对逻辑错误的检测依赖有限的 QA 测试；需要高计算成本和大模型；仅针对 Python；检索增强可能引入信息泄露风险；SDS 设计缺失仍可能导致架构不完整。

---

## 130. Attractor Domain Theory: A Mathematical Framework for Cardiovascular Attractor Analysis with Wearable Photoplethysmography (PPG) Validation

**arXiv ID:** 2606.22039 | [PDF](https://arxiv.org/pdf/2606.22039v1)

**作者:** Timothy Oladunni `[一作]` (Morgan State University), Farouk Ganiyu Adewumi `[通讯]` (Morgan State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了心脏吸引子域理论（ADT），定义并证明了三种互不冗余的吸引子域（几何域、遍历域、变分域），并在可穿戴PPG信号上实现并验证了几何域的临床预测能力，实现了连续心血管监测的理论框架。

**💡 创新点**

创新点在于给出了吸引子信息的完整分解理论——域充分性定理，阐明哪些吸引子属性对应哪些心血管指标，解决了传统非线性特征选择的冗余与解释难题，并解释了跨模态传递差异。

**🔧 技术方法**

主要技术包括延迟嵌入与Takens定理、Lyapunov指数、再现统计、有限时间Lyapunov场、信息论分解、SCSI几何域实例、贝叶斯优化嵌入参数，以及对信号质量的量化评估与裁剪。

**📊 数据集**

实验使用四个PPG数据集（BIDMC ICU、BUT‑PPG、RWS‑PPG、Welltory）共176,742段进行训练验证，并在CapnoBase外部数据集做进一步验证。

**📈 对比分析**

在消除三类评估偏差后，SCSI模型在校正协议下的AUC为0.757（95% CI 0.686–0.828），NPV 0.966，显著优于无偏基线0.573；CNN模型虽然整体AUC略高，但受样本权重偏差影响；外部验证AUC为0.621。

**⚠️ 局限性**

局限性包括：仅完成几何域的验证，遍历域和变分域的完整实证尚未完成；模型对不同目标的域对应关系仍需进一步验证；噪声与观测函数不可逆性可能限制了泛化性能。

---

## 131. Learning to Evade: Adaptive Attacks on Audio Watermarking

**arXiv ID:** 2606.22310 | [PDF](https://arxiv.org/pdf/2606.22310v1)

**作者:** Weikang Ding `[一作]` (University of Missouri-Kansas City), Qiben Yan `[通讯]` (Michigan State University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种自适应音频水印攻击方法(AWM)，能够绕过现有防御策略并保持音频质量

**💡 创新点**

创新点在于利用水印解码器概率的正态分布特性，先估计分布参数后动态调整攻击扰动，使其落在合法范围内，从而避免被检测

**🔧 技术方法**

采用两阶段优化（成功率优先、音质提升）和分布参数估计的自适应扰动生成技术

**📊 数据集**

在两种音频水印方法上，使用了三个语音数据集进行实验

**📈 对比分析**

相较于传统对抗攻击，AWM在替换和生成攻击中的检测率低于10%，在移除攻击中为0%，且音频质量得以显著提升

**⚠️ 局限性**

局限性包括需要足够的目标音频样本来估计分布参数，对极端噪声环境或未知水印架构的鲁棒性尚未充分验证

---

## 132. Beyond Time Series: Spatial Reasoning for Epidemic Forecasting via Multimodal Learning

**arXiv ID:** 2606.22171 | [PDF](https://arxiv.org/pdf/2606.22171v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 133. Can LLMs Control Readability? A Multi-Dimensional Evaluation Framework for CEFR-Controlled Arabic Generation

**arXiv ID:** 2606.21981 | [PDF](https://arxiv.org/pdf/2606.21981v1)

**作者:** Nour Rabih `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Ted Briscoe `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了多维CEFR控制下的阿拉伯语文本生成评估框架，并在GPT-4o上进行实验。

**💡 创新点**

首次将CEFR与细粒度Taha-19可读性尺度对齐，并结合句法与词汇特征进行多维评估。

**🔧 技术方法**

使用Prompt工程、BAREC可读性预测、CamELParser语法分析、SAMER词表与余弦相似度评估等技术。

**📊 数据集**

利用ZAEBUC、ARWI、SAMER、BAREC等公开阿拉伯语数据集。

**📈 对比分析**

通过余弦相似度比较生成文本与参考CEFR特征向量，P3条件下与CEFR及Taha-19对齐度最高（≈0.91/0.99）。

**⚠️ 局限性**

仅评估单一LLM（GPT‑4o），依赖单一可读性模型，且在B1级别的可读性控制效果不佳。

---

## 134. Frequency-Domain Neural ODEs for Modeling Non-Linear Dynamical Systems

**arXiv ID:** 2606.22075 | [PDF](https://arxiv.org/pdf/2606.22075v1)

**作者:** Mohammed Ashraf `[一作]` (German University in Cairo), Ayman A. El-Badawy `[通讯]` (German University in Cairo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发并验证了 Frequency-Domain Neural ODE (FNODE) 用于建模复杂非线性动力系统。

**💡 创新点**

创新点在于将连续时间动态投影到频域，学习频谱系数的连续演化，从而克服传统 NODE 的拓扑限制和谱偏差。

**🔧 技术方法**

技术包括 FFT/逆FFT、频域稀疏编码、连续深度模型（NODE/ANODE）、自编码器/解码器、课程学习和集成学习。

**📊 数据集**

使用四个合成动力系统的数据集：Lotka‑Volterra、Duffing、Van der Pol 与 Lorenz。

**📈 对比分析**

通过与 GRU、LSTM、NODE、ANODE 等模型在相同超参数下对四系统进行训练和测试，FNODE 在均方误差、置信区间与收敛稳定性方面表现最佳。

**⚠️ 局限性**

主要限制是频域变换增加计算时间，引入更多超参数，且需进一步优化以满足实时部署需求。

---

## 135. Quantifying Theoretical AI Alignment Guarantees: Receiver-Utility Bounds in Bayesian Persuasion

**arXiv ID:** 2606.22226 | [PDF](https://arxiv.org/pdf/2606.22226v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 136. DevoTG: Temporal Graph Neural Networks for Modeling C. elegans Developmental Connectomics

**arXiv ID:** 2606.21940 | [PDF](https://arxiv.org/pdf/2606.21940v1)

**作者:** Jayadratha Gayen `[一作]` (IIIT Hyderabad), Bradly Alicea `[通讯]` (OpenWorm Foundation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 DevoTG 框架，利用时间图神经网络对 C. elegans 的细胞分裂事件（连续时间动态图）和突触连接（离散时间动态图）进行建模与分析。

**💡 创新点**

创新点在于：① 将连续时间和离散时间动态图联合应用于同一生物系统；② 在细胞分裂预测任务中首次引入 TGN 并显著提升性能；③ 通过时间维度对突触连通性进行三类稳定性划分，补充了以往的个体差异分类；④ 提供交互式可视化工具。

**🔧 技术方法**

核心技术为 Temporal Graph Neural Network (TGN)，结合 GRU 内存模块、TransformerConv 关注机制以及二分类预测头；同时使用 PyTorch Geometric、NetworkX 等图分析库。

**📊 数据集**

数据集包括：① WormAtlas 修订的细胞分裂事件 CSV（642 事件，1,203 细胞状态）；② Witvliet 等人提供的 8 个时间点的电子显微镜重建突触连接数据（225 细胞，858–2,496 条边）。

**📈 对比分析**

与基线方法（随机、度优先、静态 GAT）对比，TGN 在细胞分裂预测任务上平均测试 AUC 为 0.839±0.007，远高于静态 GAT 的 0.577±0.080（差距 26 分），并且方差低；在连通性稳定性分类方面，提出的三类划分与 Witvliet 的个体差异分类互补，提供了时间维度的新视角。

**⚠️ 局限性**

局限性包括：① 仅在细胞分裂数据上训练，未直接预测突触形成；② 稳定性阈值（≥6/8 时点）固定，敏感性待评估；③ 电压接头与化学突触被统一处理，忽略了其不同的生物学特性；④ 仅在 C. elegans 上验证，缺乏跨物种的通用性验证。

---

## 137. VegSim: A Geospatial World Model for Scenario-Conditioned Vegetation Simulation

**arXiv ID:** 2606.21961 | [PDF](https://arxiv.org/pdf/2606.21961v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 138. How Does Research Evolve? Tracing Cross-Domain Trajectories in NLP, ML, and CV with Claim-Grounded Typed Citations

**arXiv ID:** 2606.22342 | [PDF](https://arxiv.org/pdf/2606.22342v1)

**作者:** Abdul Muntakim `[一作]` (Kennesaw State University), Yong Pei `[通讯]` (Kennesaw State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了SciTraj，首个基于句子级别主张的typed citation图谱，包含32,559篇论文、573,126条带有主张句子及NLI验证的边，并提供287M条长度≥3的研究轨迹。

**💡 创新点**

创新点在于：①将引用关系与具体主张句子绑定，形成“claim‑grounded”typed citation；②利用DeBERTa‑MNLI对主张驱动关系进行自动NLI验证；③提出年份打乱（year‑shuffle）假设检验协议，区分时间信号与内容相关性；④基于typed图谱开展跨学科与主题演化的系统轨迹分析。

**🔧 技术方法**

技术手段包括：DeBERTa‑v3‑MNLI用于主张验证；SPECTER2嵌入+k‑means聚类用于语义相似度与主题归属；Pair‑MLP/LightGBM做边预测；多种Typed‑GNN（R‑GCN、hetero GraphSAGE等）做对照；以及自定义48维图结构特征、年份基准和语义相似度阈值。

**📊 数据集**

数据集来源于Semantic Scholar Open Research Corpus，筛选2015‑2024年NLP（ACL/EMNLP/…）、ML（NeurIPS）和Vision（CVPR/ICCV/WACV/ACCV）会议论文，最终得到32,559篇论文、573,126条typed边和约287M条轨迹。

**📈 对比分析**

在严格时间拆分（≤2020训练，2021‑2022验证，2023‑2024测试）下，SciTraj‑Pair模型在typed链接预测任务中达到AUC≈0.914，Typed relation分类6‑way宏F1≈0.948；相比之下，内容相似度基线SPECTER2‑kMeans AUC≈0.876，Typed‑GNN（R‑GCN等）最高仅0.87。未来引用预测AUC在0.89‑0.94之间，证明typed特征和时间信息显著提升性能。

**⚠️ 局限性**

局限性包括：①未实现对研究轨迹的预测模型；②相似度驱动关系（direct_extension、temporal_semantic）未进行per‑claim NLI验证；③仅有3位评审员的pilot实验，规模有限；④资源聚焦于NLP/ML/Vision 2015‑2024，未覆盖更广泛领域。

---

## 139. Multi-Target Maneuver Coordinations: Unlocking Coordination Opportunities in Connected Automated Driving

**arXiv ID:** 2606.22055 | [PDF](https://arxiv.org/pdf/2606.22055v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 140. Resume Screening, Fast and Slow: (Biased) AI Recommendations' Influence on Human Decision Making

**arXiv ID:** 2606.22213 | [PDF](https://arxiv.org/pdf/2606.22213v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 141. Open AI in the Wild: Adoption and Adaptation of Open Models on r/LocalLLaMA

**arXiv ID:** 2606.22211 | [PDF](https://arxiv.org/pdf/2606.22211v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 142. Latent Confidence Alignment for LLM Self-Assessment

**arXiv ID:** 2606.21937 | [PDF](https://arxiv.org/pdf/2606.21937v1)

**作者:** Ting-Yu Chen `[一作]` (National Sun Yat-Sen University), Yihuang Kang `[通讯]` (National Sun Yat-Sen University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于IRT与元认知的LLM自我评估框架，衡量模型自评与潜在能力结构下的错误概率的一致性。

**💡 创新点**

引入Latent Confidence Alignment Error（LCAE）指标，首次将模型自评误差与Rasch模型推断出的潜在误差概率进行对齐评估，并通过难度信号和反思机制提升自评质量。

**🔧 技术方法**

使用Rasch模型（1PL IRT）估计模型能力和题目难度，计算IRT错误概率；通过二值交叉熵定义LCAE；引入难度信号（IDS）与双流程路由（DPR）作为外部与内部自评调节机制；成本效益指标CE用于评估推理成本。

**📊 数据集**

在医学问答基准MedXpertQA上随机抽取100题，使用20个LLM（Claude、DeepSeek、Gemini、Llama、GPT、Qwen等）并配合5种提示策略（标准、知识、CoT、Self-Ask、ToT）。

**📈 对比分析**

与基线（仅自评）对比，IDS显著降低LCAE，DPR单独效果有限，IDS+DPR组合进一步提升自评一致性；模型能力无显著差异；同时通过CE展示成本与自评质量的多维度关系。

**⚠️ 局限性**

局限性包括仅在医学领域验证、样本规模有限、IDS难度信号基于同一模型集可能引入偏差、成本评估简化且需根据实际应用调整。

---

## 143. GeoFlow-SLAM++: A Robust Multi-Camera Visual-Inertial SLAM System with Relocalization

**arXiv ID:** 2606.22051 | [PDF](https://arxiv.org/pdf/2606.22051v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 144. Powerdomains and nondeterminism in synthetic domain theory

**arXiv ID:** 2606.22238 | [PDF](https://arxiv.org/pdf/2606.22238v1)

**作者:** Yue Niu `[一作]` (National Institute of Informatics), Taro Sekiyama `[通讯]` (National Institute of Informatics)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

在合成域理论框架下构造并研究了下界、上界及凸幂域，并将其用于定义带非确定性的编程语言的取值语义、观察语义与操作语义，最终证明了计算适当性；

**💡 创新点**

首次在合成域理论中给出下界和上界幂域的构造，利用商归约类型实现自由半格结构，并证明这些幂域在保持域的同时产生单子结构；

**🔧 技术方法**

使用合成域理论的公理化语义、商归约型（quotient inductive types）、路径关系与局部化技术、分布律与单子组合、逻辑关系论证以及观测语义等；

**📊 数据集**

无；

**📈 对比分析**

无；

**⚠️ 局限性**

对幂域的路径顺序与外在排序的完整特征仍未完全证明，且模型对 dominance 的闭包需求（有限/可数并）限制了适用的拓扑模型；

---

## 145. Auditing Combinatorial Randomness from Finite Transcripts

**arXiv ID:** 2606.22034 | [PDF](https://arxiv.org/pdf/2606.22034v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 146. 2.5D Root of Trust: Securing the Chiplet Ecosystem

**arXiv ID:** 2606.22198 | [PDF](https://arxiv.org/pdf/2606.22198v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 147. Early-Exit Graph Neural Networks for Link Prediction

**arXiv ID:** 2606.22167 | [PDF](https://arxiv.org/pdf/2606.22167v1)

**作者:** Roman Knyazhitskiy `[一作]` (University of Cambridge), Andrea Giuseppe Di Francesco `[通讯]` (Sapienza University of Rome)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了在GNN中用于链接预测的节点级和子图级早停策略。

**💡 创新点**

创新点在于不使用辅助预算损失，而是让早停自然从任务损失中显现，并针对链接预测设计了邻域一致的子图早停机制。

**🔧 技术方法**

采用Gumbel-Softmax置信头、GCN与SAS-GNN骨干、稳健的ODE式更新、以及自适应温度和流量约束来实现早停。

**📊 数据集**

使用HeaRT基准中的Citeseer、PubMed和ogbl-ddi三大图数据集进行实验。

**📈 对比分析**

与固定深度基线和oracle相比，节点级和子图级早停在大多数指标上都能在更低计算量下取得相近或更优的MRR、Hits@1/10表现，节点级早停更快但可能略逊。

**⚠️ 局限性**

局限性包括对超参数和实现细节高度敏感，GPU加速取决于稀疏执行支持，且仅在GCN和SAS-GNN两种骨干上验证，缺乏对注意力或异构GNN的推广。

---

## 148. IDAG-Edit: Multi-Object Video Editing via Instance-Decoupled Attention and Guidance

**arXiv ID:** 2606.22042 | [PDF](https://arxiv.org/pdf/2606.22042v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 149. Framing the 5% Problem: Teachers' Perspectives on Persistence in Educational Technology

**arXiv ID:** 2606.22294 | [PDF](https://arxiv.org/pdf/2606.22294v1)

**作者:** Conrad Borchers `[一作]` `[通讯]` (Vanderbilt University), Conrad Borchers (Vanderbilt University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过90分钟的参与式设计工作坊，收集并分析了12名使用i-Ready Math的中学数学教师对“5%问题”的看法

**💡 创新点**

将5%问题重新定义为教师视角下的情境化教学挑战，并明确了教师在解释学生坚持度时所需的认知与情境信息

**🔧 技术方法**

采用参与式设计、开放式卡片排序、共识投票和反思性主题分析等方法

**📊 数据集**

使用教师工作坊产生的卡片、分组层级、投票记录和录音作为质性数据

**📈 对比分析**

本文未与其他模型或系统进行定量对比，而是通过多源质性证据（卡片、投票、访谈）阐明主题重要性

**⚠️ 局限性**

样本规模有限（仅12名教师、单一地区、单一平台），且仅基于一次工作坊，结果的普适性与稳健性需进一步验证

---

## 150. All Routes Lead to Collapse

**arXiv ID:** 2606.22325 | [PDF](https://arxiv.org/pdf/2606.22325v1)

**作者:** K. R. Balasubramanian `[一作]` `[通讯]` (Independent Research), K. R. Balasubramanian (Independent Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过把软注意力视为欧氏距离上的Boltzmann加权，证明其使用的相似度度量是平坦且对 key 范数不敏感，从而导致任何使用相同度量的内容路由器都会出现注意力集中、表示低秩化和范数分层的现象，并在多种架构（Transformer、图注意力、Mamba、RWKV、AttnRes）以及图神经网络对比实验中验证该机制。

**💡 创新点**

创新点：①提出软注意力与欧氏距离的等价身份，揭示其对 key 范数的“盲点”；②将此视角扩展到所有内容路由器，证明其是通用机制；③通过多架构的定量测量和因果消融实验，区分机制与表现的可变特征；③提出“位置制动器”对表现时序与强度的调控作用。

**🔧 技术方法**

主要技术：数学推导（软注意力的Boltzmann形式）；统计测量（key 范数 CV、rank‑8 方差解释、最大注意力集中度）；对比实验（Gaussian 与 Shuffle 基线、位置制动器扫频、Mamba 与 RWKV 的消融）；图神经网络控制实验（GAT 与 GCN）。

**📊 数据集**

数据集：WikiText‑103 验证集（150 句、长度 128）用于 Transformer、Mamba、RWKV、AttnRes；WebKB heterophilic 图集用于图注意力与 GCN 对比；Pythia 旋转键在预旋转与后旋转两种形式测量；另外使用公开的 GPT‑2、Pythia 预训练模型。

**📈 对比分析**

比较方法：在每个模型中与两个 null 基线（高斯独立键、打乱键顺序）对比，计算 key 范数 CV、rank‑8 方差解释、注意力集中度；对路由器进行因果消融（冻结门控、改变 decay、移除 recency bias），观察集中度变化。结果表明：所有路由器在非零制动器下都出现集中与低秩化，消融验证了路由机制是导致该现象的原因；在位置制动器范围内，集中度随制动强度平滑变化。

**⚠️ 局限性**

局限性：实验规模受限于中小模型，未验证大规模模型的泛化；AttnRes 的查询为固定学习探针，深度轴结果不与 token 轴完全可比；消融实验采用极端超参数，可能偏离训练分布；几何框架仅为诊断工具，未给出可直接改进模型的具体方案。

---

## 151. EmbodiedUS-FS: Fast Slow Intelligence for Ultrasound Robotics

**arXiv ID:** 2606.22319 | [PDF](https://arxiv.org/pdf/2606.22319v1)

**作者:** Fangzhuo Zhang `[一作]`, Jinchang Zhang `[通讯]` (SUNY Binghamton)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

构建了快慢层次的智能框架用于机器人超声扫描，分离高层知识驱动规划与低层闭环执行，并加入安全盾与升级策略。

**💡 创新点**

快慢双脑架构将知识增强的意图解析与任务图生成与多模态闭环执行分离，结构化计划验证与安全盾提升可靠性。

**🔧 技术方法**

使用大语言模型LLaMA3、API与手册检索、任务图生成、结构化计划验证、多模态状态编码、闭环控制、图像质量恢复以及Safety Shield与层级升级等技术。

**📊 数据集**

使用了1,000条机器人手册样本与1,000条超声API样本的合成数据集，用于检索训练与评估。

**📈 对比分析**

通过与单链LLM、UAR、RHR等递进组件的消融实验比较；在动态扰动下，完整系统任务完成率提高至约80%，恢复率≥79%，安全违规率降至≈1.6%。

**⚠️ 局限性**

仅在合成数据和模拟环境验证，缺乏真实临床大规模评估；对手册检索质量依赖较高；安全阈值需手动调参。

---

## 152. DejaVu: Why You Should Write to Your DRAM Rows Twice, Carefully

**arXiv ID:** 2606.22297 | [PDF](https://arxiv.org/pdf/2606.22297v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 153. Curriculum Reinforcement Learning Can Incentivize Reasoning Capacity in LLMs Beyond the Base Model

**arXiv ID:** 2606.22317 | [PDF](https://arxiv.org/pdf/2606.22317v1)

**作者:** Pengxiang Cai `[一作]` (Hong Kong University of Science and Technology Guangzhou), Jintai Chen `[通讯]` (Hong Kong University of Science and Technology Guangzhou)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种边界感知课程式强化学习方法，通过大样本 pass@k 采样定位基线模型的推理边界，给难题提供小规模教师引导，再用 RL（GRPO）巩固新推理模式，从而显著扩展 LLM 的推理能力边界。

**💡 创新点**

创新点在于：①识别并处理“零优势组”，将无法产生奖励信号的难题转化为可奖励的训练样本；②采用大 k pass@k 边界探索 + 定向教师指导 + 课程化 RL 循环，实现从基线到更高推理边界的连续提升。

**🔧 技术方法**

使用的技术包括：GRPO 风格的 RLVR、pass@k 采样分析、教师提示式局部引导（chain‑of‑thought 方式）、课程化训练策略以及多轮的边界探索与 RL 巩固。

**📊 数据集**

训练集使用 DAPO‑Math‑17K，评估集采用 AIME 2024/2025 和 MATH500，覆盖多种推理难度。

**📈 对比分析**

通过与基线模型和普通 RLVR 的对比，评估 pass@1、pass@256 等指标，结果显示 Boundary‑aware Curriculum RL 在 pass@256 上平均提升约 9.8 百分点，同时 pass@1 也有提升，证明模型在大采样预算下的推理边界得到了扩展。

**⚠️ 局限性**

局限性包括：未对课程选择做最优化，教师引导规模有限；方法依赖大 k 采样评估，受预算限制；在部分模型‑基准组合下 RLVR 仍表现不稳定，需进一步改进。

---

## 154. From Handcrafted Features to Functional Edge Learning: Evolution of EEG Seizure Detection Frameworks

**arXiv ID:** 2606.22258 | [PDF](https://arxiv.org/pdf/2606.22258v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 155. Benchmarking Robot Memory Under Interference

**arXiv ID:** 2606.22338 | [PDF](https://arxiv.org/pdf/2606.22338v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 156. Enhancing Protein Representation Learning via Manifold Restore Mixing

**arXiv ID:** 2606.22307 | [PDF](https://arxiv.org/pdf/2606.22307v1)

**作者:** Yizhou Dang `[一作]` (Northeastern University), Zhu Sun `[通讯]` (Singapore University of Technology and Design)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了 Manifold Restore Mixing（MRM）方法，在蛋白表示学习中将原始与增强样本在隐藏层进行线性混合，从而恢复因数据增强造成的结构信息缺失并引入多样化变形。

**💡 创新点**

创新点在于首次在表示空间进行混合，配合难度调度器和两阶段训练策略，既保留蛋白的原始结构，又提升多任务下游性能，解决传统DA导致结构损失与性能退化的问题。

**🔧 技术方法**

采用的技术包括Manifold Restore Mixing、beta 分布难度调度器、两阶段正则化训练，以及结合 1D 序列与 3D 结构的多模态编码器（如 GearNet、CDConv 等）。

**📊 数据集**

实验使用公开数据集 PDB、AlphaFoldDB、Pfam、UniRef 等，针对 Fold、ER、GO、EC 四大下游任务进行评估。

**📈 对比分析**

在与传统 DA（置换、噪声、采样）以及现有 Mixup 变体对比时，MRM 在所有基线模型上实现了 3–6% 的绝对精度提升，在 Fold、ER、GO、EC 任务上均优于最先进模型，甚至超越部分预训练 Protein LLM。

**⚠️ 局限性**

限制包括：未在 Protein LLM 预训练阶段验证效果；恢复仅发生在潜在空间，缺乏对生物学可解释性的深入分析；由于资源限制，未覆盖更大规模或更广泛任务的实验。

---

## 157. Leveraging Large Language Models to Obscure Code Stylometry: A Comparative Study of GPT-3.5 and GPT-4

**arXiv ID:** 2606.22306 | [PDF](https://arxiv.org/pdf/2606.22306v1)

**作者:** Saman Pordanesh `[一作]` (University of Calgary), Benjamin Tan `[通讯]` (University of Calgary)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用GPT-3.5与GPT-4对代码进行重写，使其风格特征被模糊，同时保持功能完整，以评估大型语言模型在代码作者身份隐藏方面的有效性。

**💡 创新点**

提出了将LLM与结构化prompt相结合来掩蔽代码stylometry的新方法，并对比单次与多次提示策略在隐藏效果和功能保持上的差异。

**🔧 技术方法**

采用大型语言模型GPT-3.5、GPT-4、Prompt Engineering、随机森林作者身份识别器、功能性测试框架。

**📊 数据集**

使用公开代码作者身份数据集（如开源项目的作者标签数据），并将其划分为训练与测试集。

**📈 对比分析**

通过让随机森林在原始代码与LLM重写后的代码上进行训练与预测，比较两者准确率；实验表明GPT-4在多次提示下可将准确率从约90%降至约50%，而GPT-3.5则效果不如前者。

**⚠️ 局限性**

局限性包括仅检验两款LLM与单一随机森林模型；样本量有限；功能验证仅基于基本运行测试，未覆盖所有边界情况；以及无法保证在更强大或更复杂的作者身份识别方法下同样有效。

---

## 158. Active Sensing and Deferred-Decision Trajectory Optimization for Robust Target Identification

**arXiv ID:** 2606.22277 | [PDF](https://arxiv.org/pdf/2606.22277v1)

**作者:** Farbod Siahkali `[一作]` (Purdue University), Vijay Gupta `[通讯]` (Purdue University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

本文提出将主动感知融入延迟决策轨迹优化，形成AS-DDTO，既保证对多候选目标的可达性，又在轨迹上主动获取信息。

**💡 创新点**

创新点在于在DDTO的最大化延迟目标分离目标中加入基于距离的信息增益项，并支持贝叶斯和分布无关的候选集更新，提供递归可行性、后验浓缩和固定时间覆盖保证。

**🔧 技术方法**

采用混合整数二次规划(MI(SO)CP)重构、贝叶斯更新、分形预测与Fisher组合，以及离线/在线规划框架。

**📊 数据集**

实验数据基于仿真四旋翼模型，随机生成候选目标位置，使用高斯测量模型与基于MLP的分形预测模型。

**📈 对比分析**

与传统DDTO进行离线与在线比较，AS-DDTO在拦截率和目标识别率上提升约20%-30%，但求解时间明显增加。

**⚠️ 局限性**

局限在于求解时间长、对参数（如w、ρ、α）的敏感性，以及对真实感知模型的依赖。

---

## 159. Encoder-Decoder Manifold Alignment for Idempotent Generation

**arXiv ID:** 2606.22304 | [PDF](https://arxiv.org/pdf/2606.22304v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 160. Efficient Document Tampering Localization with Multi-Level Discrepancy Features and Unified DCT-Quantization Embedding

**arXiv ID:** 2606.22285 | [PDF](https://arxiv.org/pdf/2606.22285v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 161. Modulo Quantization Coding for Primitive Relay and Diamond Channels with Correlated Noises

**arXiv ID:** 2606.22313 | [PDF](https://arxiv.org/pdf/2606.22313v1)

**作者:** Yuanxin Guo `[一作]` (University of Toronto), Wei Yu `[通讯]` (University of Toronto)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出一种模量量化（Modulo Quantization, MQ）编码框架，用于在具有原始（无噪声数字）中继链路和相关高斯噪声的高斯信道（原始中继信道和菱形信道）中实现低复杂度、高效的可靠通信。

**💡 创新点**

创新点在于：
1) 采用简单的标量量化+取模操作，直接利用噪声的共同成分，实现对完美相关噪声情况下的容量（ψ(P)+R₀）以及菱形信道下的最小链路容量 min(R₁,R₂) 的零误差传输；
2) 通过叠加 MQ 与传统高斯码、时间共享或分割功率等手段，构造出可在不同 SNR 区间达到或逼近 cut‑set 上界的混合 MQ–DF 方案；
3) 将 MQ 框架推广到多重中继和非完美相关噪声场景，并在强相关情况下仍能逼近 CF 的性能。

**🔧 技术方法**

主要技术手段包括：
- 标量 MQ 取模量化；
- 整数格子编码与叠加编码；
- 时间共享、功率分配与分割码字；
- Wyner–Ziv 量化（对比 CF）；
- 信息论极限分析、cut‑set 与 DF 边界；
- 二元对称通道模型与错误概率分析；
- 对称性与低 SNR/高 SNR 的解析推导。

**📊 数据集**

论文采用理论分析方法，使用的是标准的 AWGN 与高斯相关噪声模型，未涉及实际数据集或实验测量。所有结论均基于信息论极限与数值优化。

**📈 对比分析**

比较方法：将 MQ、MQ–DF、DF、CF、hash‑forward、flash‑helping 等方案在不同 SNR、链路容量以及噪声相关系数 λ 下的下界进行对比。结果表明：
- 在完美相关噪声时，MQ 能实现 ψ(P)+R₀；
- 菱形信道中，MQ 达到 min(R₁,R₂)，并在高 SNR 与低 SNR 区间与 cut‑set 上界吻合；
- 在非完美相关但强相关的情形下，MQ 的速率可逼近 CF，且在某些 SNR 区间显著优于 DF；
- 对称菱形信道中，混合 MQ–DF 方案在所有 SNR 下均优于单独的 DF 或 CF。

**⚠️ 局限性**

局限性包括：
1) MQ 对噪声相关度要求较高，非完美相关时仍存在性能下降；
2) 需要精确的取模量化，量化误差或实现误差可能影响实际性能；
3) 对于高维、非高斯输入或多路复用场景，MQ 的优势尚未完全证明；
4) 在某些参数配置下，MQ 仍无法达到 cut‑set 上界；
5) 方案在实际实现中需考虑数值稳定性与硬件可行性。

---

## 162. The $α$-Index: A Penalized Authorship-Integrity Framework for Position-Weighted Scientific Contribution

**arXiv ID:** 2606.22309 | [PDF](https://arxiv.org/pdf/2606.22309v1)

**作者:** Athanasios Angelakis `[一作]` (University of Bundeswehr Munich), Athanasios Angelakis `[通讯]` (Amsterdam University Medical Center)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出α‑指数，一种基于作者顺序的保守、惩罚性作者贡献分配框架，能够量化每篇论文的单元信用并聚合成作者的累计信用

**💡 创新点**

创新点在于将首位作者的执行贡献与尾位作者的领导贡献分离，并引入按中间作者数递减的领导惩罚，凸显作者结构的诚信与责任

**🔧 技术方法**

采用位置加权权重分配（首位、尾位、中间三块），结合可调参数和函数（如1/m）实现惩罚，形成本地α‑credit与全局α‑index；并给出算法实现细节

**📊 数据集**

主要使用合成示例以及从公开论文（如ResNet、Transformer、AlphaFold等）提取的作者顺序信息作为演示数据集

**📈 对比分析**

与全计数、等份计数、谐波计数等传统方法对比；α‑指数在保持首位作者权重的同时，能明显区分中间作者增多导致的尾位信用下降，展示了对作者结构敏感的优越性

**⚠️ 局限性**

局限包括：默认权重和惩罚函数是规范性假设，可能对大团队过于严苛；作者顺序并非绝对贡献指标；等贡献信息不易获取；使用不当可能诱发“把作者移到致谢”的不良动机

---

## 163. Service-Cut Certificates for Aligned Eviction in Tiered Cache Networks

**arXiv ID:** 2606.22270 | [PDF](https://arxiv.org/pdf/2606.22270v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 164. Hypothesis-Driven Skill Optimization for LLM Agents

**arXiv ID:** 2606.22330 | [PDF](https://arxiv.org/pdf/2606.22330v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 165. SCENIC: Semantic-Conditioned Edge-Aware Neural Framework for Structured IoT Command Generation

**arXiv ID:** 2606.22296 | [PDF](https://arxiv.org/pdf/2606.22296v1)

**作者:** Luke Ztz Hu `[一作]` (Tsinghua Shenzhen International Graduate School), Songping Mai `[通讯]` (Tsinghua Shenzhen International Graduate School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SCENIC框架，集成了从数据构造、模型选择、语义对齐的三元组对比微调、剪枝量化到ONNX导出与TensorRT加速的完整端到端流程，用于在资源受限的IoT边缘设备上生成结构化的智能家居指令。

**💡 创新点**

创新点在于：①将语义相等定义为相同的可执行命令，将三元组对比学习引入指令微调；②系统化评估不同小型Transformer架构在剪枝后对结构化命令准确率的鲁棒性；③通过量化+稀疏化+TensorRT实测证明encoder‑decoder在高稀疏率下最稳健且可实现硬件加速。

**🔧 技术方法**

使用的技术包括：0.2B左右的Encoder‑Only、Decoder‑Only、Encoder‑Decoder（T5‑style）小型Transformer；triplet‑loss对比微调（C‑SFT）；多种剪枝策略（magnitude、gradient、WANDA、NVIDIA 2:4 sparsity）；FP16/INT8量化与ONNX导出；Jetson Orin上的TensorRT FP16/INT8加速。

**📊 数据集**

使用Smart Home Instruct数据集家族：Instruct‑SFT（9.7K对）、Instruct‑Contrast（9.7K元组）、Instruct‑Bench（200条基准）以及引用的IFTTT、SAGE、Home Assistant Requests等辅助资源。

**📈 对比分析**

与基线的比较基于EM@1/EM@5指标。Dense SFT下Decoder‑Only取得最高99.0% EM@1；Encoder‑Decoder在C‑SFT下保持98.0% EM@1并在30%稀疏率下仍高于99%；50%稀疏率下Decoder‑Only崩溃，Encoder‑Decoder通过WANDA或梯度剪枝仍保持95–99% EM@1；Pruned INT8导出模型在保持91% EM@1的同时将模型尺寸缩小25%。

**⚠️ 局限性**

局限性：评估仅基于200条基准，缺乏真正的留出测试；剪枝选择未通过验证集锁定；未测量完整端到端推理延迟和能耗；硬件加速依赖特定TensorRT/ NPU实现，通用性不足。

---

## 166. MixedPEFT: Combining Multiple PEFT Methods with Mixed Objectives for Unsupervised Domain Adaptation

**arXiv ID:** 2606.22272 | [PDF](https://arxiv.org/pdf/2606.22272v1)

**作者:** Mohammed Rawhani `[一作]` (Erciyes University), Bahriye Akay `[通讯]` (Erciyes University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种结合可逆适配器与LoRA的自定义PEFT联合，并在同一训练阶段使用分类与MLM混合目标，完成无监督领域适应；

**💡 创新点**

首次将可逆适配器与LoRA融合并通过混合目标训练，实现仅更新约7%参数即可超越传统全参数微调方法；

**🔧 技术方法**

使用BERT‑base作为基模型，结合可逆适配器、LoRA、参数高效微调技术、混合目标损失（分类+MLM）以及AdamW优化器；

**📊 数据集**

在MNLI多域自然语言推断数据集上，跨20个域转移任务进行评估；

**📈 对比分析**

与UDapter、DANN、DSN及上界比较，平均F1 76.39%，比UDapter高1.41pp，比DANN高1.26pp，比DSN高0.86pp，仅使用7.1%可训练参数，标准差与传统方法相近；

**⚠️ 局限性**

仅在NLI任务与BERT‑base上验证，缺乏对更大模型或其它NLP任务的推广；混合目标权重动态设定可能在极端不平衡数据时效果受限；PEFT组合的通用性仍需进一步探索。

---

## 167. SHACR: A Graph-Augmented Semi-Autonomous Framework for Multi-Class Conflict Resolution in Smart Home IoT Automation

**arXiv ID:** 2606.22312 | [PDF](https://arxiv.org/pdf/2606.22312v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 168. Learning at the Right Pace: Adaptive Data Scheduling Improves LLM Reinforcement Learning

**arXiv ID:** 2606.22305 | [PDF](https://arxiv.org/pdf/2606.22305v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 169. FlowDPG: Deterministic Policy Gradient on Flow Matching Policies for Real-World Manipulation

**arXiv ID:** 2606.22303 | [PDF](https://arxiv.org/pdf/2606.22303v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 170. Do Gains from Generative AI-Enabled Adaptive Pretesting Persist? Evidence from a Retention Study

**arXiv ID:** 2606.22328 | [PDF](https://arxiv.org/pdf/2606.22328v1)

**作者:** Mahir Akgun `[一作]` (Pennsylvania State University), Sacip Toker `[通讯]` (Atilim University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在本科应用统计课程中，研究者先让学生完成基于大语言模型的自适应预测学习，然后随机分配三种 AI 支持的复习模式（自适应间隔检索、固定间隔检索、学习者主导 AI 学习），在七周后对学习成果和练习投入进行评估。

**💡 创新点**

创新点在于将生成式 AI 结合自适应预测与后续结构化检索实践，并探究其对长期保持学习效果的持续影响。

**🔧 技术方法**

使用的大型语言模型（LLM）通过系统提示实现自适应提问、反馈与检索推送，构建了可根据学习者回答动态调整难度的 AI 代理。

**📊 数据集**

数据集为自制的 14 题多选测验（包含网络安全情境下的多元线性回归问题），以及实验期间收集的 89 名学生 AI 对话日志。

**📈 对比分析**

通过多变量协方差分析（MANCOVA）和后续单变量 ANCOVA 对三组结果（期末测试分数和观察到的练习投入）进行比较，结果显示自适应间隔检索组在期末测试（平均 78.2 分）和练习投入（平均 0.85）上显著优于学习者主导组，差异效应大小分别为 d≈0.92 与 d≈1.33。

**⚠️ 局限性**

局限包括仅在单一教学情境中验证、观察练习投入依赖人工编码、未测量内部动机或时间成本，且缺乏跨领域与任务类型的泛化验证。

---

## 171. Geometry-Aware Online Scheduling for LLM Serving: From Theoretical Bound to System Practice

**arXiv ID:** 2606.22327 | [PDF](https://arxiv.org/pdf/2606.22327v1)

**作者:** Li Kong `[一作]` (Renmin University of China), Zijie Zhou `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于几何体积优先的在线调度算法（SVF）及其轻量化 1‑bit SVF，解决 LLM 推理中的 KV 缓存内存‑时间耦合调度问题；在理论上给出竞争比分析并实现于 vLLM；在实验中验证其在多模型、多负载下对延迟与吞吐的提升。

**💡 创新点**

① 将 LLM 推理视为二维空间‑时间几何体积问题，构建体积优先调度（SVF）；② 证明在突发负载下 CR ≤ 5，显著优于之前的 CR ≤ 48；③ 设计仅需 1 位长度信息的 1‑bit SVF，兼顾轻量和理论保障；④ 通过连续批处理与预测器结合，兼顾在线实现与高效性。

**🔧 技术方法**

竞争比分析、线性规划松弛、几何体积模型、预测器与二分类器、vLLM 框架、连续批处理、Llama‑3.1 大模型、GPU 上的实验与系统开销测量。

**📊 数据集**

LMSYS‑Chat 与 LongBench 两大真实交互与长文本工作负载；使用 Meta‑Llama‑3.1‑8B 与 70B 两个规模模型。

**📈 对比分析**

与 FCFS、SJF、Oracle‑SJF、Oracle‑SVF 四种基线对比，评估 per‑token 平均与 95% 延迟以及吞吐率。实验显示 SVF 在平均与尾延迟上均优于 SJF；1‑bit SVF 仍显著优于 FCFS/SJF；在突发与泊松到达场景下均实现 10‑30% 的延迟降低，吞吐率保持或提升。

**⚠️ 局限性**

仅在单机单工作器实现，未考虑多 GPU/多节点间通信与负载均衡；理论分析假设已知输出长度；1‑bit SVF 需手工设定阈值，可能在极端长请求场景下表现下降；对不同硬件与更大规模系统的通用性尚未验证。

---

## 172. Spatial Modulation for Tx-SIMO-FAS: Port Selection and Performance Analysis

**arXiv ID:** 2606.22280 | [PDF](https://arxiv.org/pdf/2606.22280v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 173. Any-Body Guard: Universal Safeguarding for Manipulation Policies via Action Masking

**arXiv ID:** 2606.22278 | [PDF](https://arxiv.org/pdf/2606.22278v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 174. Differentiable Conditional Mutual Information for Multi-Terminal Linear Gaussian Wireless Networks

**arXiv ID:** 2606.22301 | [PDF](https://arxiv.org/pdf/2606.22301v1)

**作者:** Tadashi Wadayama `[一作]` (Nagoya Institute of Technology), Siqi Na `[通讯]` (Nagoya Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于线性高斯有向无环图（Gaussian‑DAG）的可微分框架，用单次 K‑递归前向传递即可得到任意条件互信息（CMI）的闭式 log‑det 形式，并通过逆向自动微分一次性获得所有可控参数的 Wirtinger 梯度，从而实现多终端率区域、加密率和其它信息量目标的端到端梯度下降优化。

**💡 创新点**

创新点在于：①将所有 CMIs 统一映射为 K‑块的子块 Schur 补并差值，构成纯粹基于矩阵运算的可微分计算图；②利用自动微分即可获得任意线性组合或非线性组合 CMI 的梯度，无需针对网络拓扑手工推导；③将这一通用工具直接应用于多用户 MIMO MAC、MIMO wiretap 和多跳 MAC 等典型多终端系统，实现了对率区域边界、保密率以及网络级功率分配等目标的直接优化。

**🔧 技术方法**

技术手段包括：线性高斯 DAG 模型、K‑递归求解节点协方差、Schur 补与 log‑det 计算、Wirtinger 复数微分、PyTorch 自动微分、投影梯度下降（对功率约束）以及 Lagrangian 余弦权重扫描。

**📊 数据集**

数据集：随机 i.i.d. (0,1) 高斯通道矩阵，固定种子下的多跳网络拓扑，均不使用真实测量数据；实验中设置 d=4、P=8 或 36、M=12 等。

**📈 对比分析**

比较方法：在 MAC 实验中与单链水分配求解得到的协作上限做对比；在 wiretap 实验中展示了传统 λ=1 秘密率点和全局 Lagrangian 扫描得到的泄露‑率 Pareto 曲线；在多跳 MAC 中仅给出优化前后的边界和目标值变化。实验结果表明：MAC 边界面积提升 24%；保密率提升 143%；多跳网络目标提升 78%。

**⚠️ 局限性**

局限性：①需要协方差矩阵正定，非正定或退化时需手动加小正则化；②整体为非凸优化，仅能得到局部 KKT 点；③目前仅覆盖线性高斯 DAG，非线性或非高斯链路尚不可直接处理；④对大规模网络的计算复杂度为 O(M² d³)，虽然对 M≤12 仍可接受，但更大规模可能受限。

---

## 175. Towards Accurate and Robust Surveillance Roadside IVD via Trackletized Audio-Visual Reasoning

**arXiv ID:** 2606.22299 | [PDF](https://arxiv.org/pdf/2606.22299v1)

**作者:** Xiwen Li `[一作]` (University of Utah), Tolga Tasdizen `[通讯]` (University of Utah)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在道路沿线监控场景下提出 TAVR 框架，利用多通道音频和车辆跟踪轨迹实现对车辆是否怠速的最终帧分类。

**💡 创新点**

创新点在于将车辆定位为实例锚点，采用轨迹化音视听推理、几何条件绑定(MASP)以及基于监督对比的 JACE 以提升跨域鲁棒性，并提供训练无关的通道对齐方法及新的跨日、跨站评测集。

**🔧 技术方法**

技术上结合 YOLOv11 + DeepSORT 的多目标跟踪、跨模态注意力的轨迹条件分类器、音频谱与轨迹几何的联合表示，以及 JACE 的监督对比损失实现音视听语义对齐。

**📊 数据集**

主要使用 AVIVD 数据集，并新增 AVIVD-LT（同址跨日）和 AVIVD-M（跨站）作为评测，训练使用原始 AVIVD 训练集。

**📈 对比分析**

与先前专用 IVD 方法、通用音视听模型和 MLLM 进行对比，TAVR 在 AVIVD-LV、AVIVD-LT、AVIVD-M 任务中分别达 mAP^AD 89.59、81.32、78.10（6 口）/67.90（3 口），显著优于基线并在域移位上表现更稳健。

**⚠️ 局限性**

局限性在于对音频缺失或截断仍易误判怠速与熄火，且稀疏麦克风阵列下对静止车辆的识别性能下降。

---

## 176. Control-Aware Manipulation of ArduPilot via Legitimate MAVLink Commands: Simulation and Hardware Validation

**arXiv ID:** 2606.22289 | [PDF](https://arxiv.org/pdf/2606.22289v1)

**作者:** Feras Benchellal andLotfi Ben Othmane `[一作]`, Bharat Bhargava `[通讯]` (Purdue University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了利用合法MAVLink命令对ArduPilot进行控制感知攻击，并在SITL仿真与Pixhawk 2.4.8硬件平台上验证了六种攻击手段。

**💡 创新点**

系统性映射ArduPilot级联控制架构为攻击原语，构造仅凭合法命令即可导致失控或坠毁的攻击序列，填补了对控制系统安全性缺口的研究。

**🔧 技术方法**

使用了参数写入（PARAM_SET）与命令（COMMAND_LONG）发送、软件级联控制仿真（SITL）、DataFlash日志解析、PID与EKF控制理论分析等技术。

**📊 数据集**

未使用公开数据集，而是自行收集SITL与Pixhawk日志作为实验数据，用于评估姿态、角速率、跟踪误差及EKF健康指标。

**📈 对比分析**

通过对比攻击前后飞行日志中的姿态波动、角速率、误差以及失控/坠毁发生率，发现单一攻击导致姿态/速率波动，组合攻击能在短时间内导致完全失控并坠毁。

**⚠️ 局限性**

限制在未启用参数范围检查或速率限制的配置下才能发挥效果，缺乏实时检测机制；实验仅覆盖单一硬件平台，未验证在其他Autopilot或更复杂环境中的泛化性。

---

## 177. Evolving Spatial Weights for Cartographic Synthesis

**arXiv ID:** 2606.22252 | [PDF](https://arxiv.org/pdf/2606.22252v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 178. Evaluating Large Language Models for Hausa and Fongbe Machine Translation: Benchmarks, Failures, and Metric Reliability

**arXiv ID:** 2606.22269 | [PDF](https://arxiv.org/pdf/2606.22269v1)

**作者:** Mahounan Pericles Adjovi `[一作]` (Carnegie Mellon University Africa), Prasenjit Mitra `[通讯]` (Carnegie Mellon University Africa)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对英语到豪萨语和丰格贝语（两种西非低资源语言）的机器翻译质量进行系统评估，并验证常用自动评估指标与人类判断的一致性。

**💡 创新点**

①首次将最新商业大型语言模型（GPT‑4o Mini、Claude Sonnet 4、Gemini 2.5 Flash、Qwen2.5‑7B）在这两种语言上进行对比；②从数据规模（500→10,000句）和评估指标（BLEU、chrF、TER、COMET、NLLB‑BLEU）两个维度考察评估可靠性；③揭示两语言之间模型排名不一致、自动指标与人类偏差差异大及神经评估指标因嵌入崩溃失效的实证。

**🔧 技术方法**

大规模语言模型推理、自动评估指标（BLEU、chrF、TER、COMET、NLLB‑BLEU）、人类评测（由本土评审按充足性、流畅性、文化适应性、整体质量四维度打分）、Bootstrap显著性检验、词表覆盖率与句子嵌入相似度分析。

**📊 数据集**

豪萨语：CCMatrix 992,266 句对；丰格贝语：NLLB‑v1 915,489 句对；人类评测样本 50 句（每语言 3 名评审）。

**📈 对比分析**

在 2,500 句规模下，各模型在自动指标上表现分离，丰格贝语最佳模型 Gemini 得分约 7.18 (BLEU)，豪萨语最佳模型 Claude 得分约 15.75；在 10,000 句规模下排名稳定且差异显著。人类评测显示豪萨语 GPT‑4o 最高 4.46/5，丰格贝语 Gemini 最高 2.20/5。自动指标与人类的秩相关性：丰格贝语 ρ=1.0，豪萨语 ρ=0.5，表明评估指标在不同语言上的可靠性差异。

**⚠️ 局限性**

样本量不足会导致误判（1,000 句易出现排名颠倒）；神经评估指标因 XLM‑RoBERTa 嵌入崩溃而失效；人类评测样本仅 50 句、评审人数少；丰格贝语参考数据质量参差不齐；商业 LLM API 版本更新不稳定，结果可重现性有限。

---

## 179. T-IMPACT: A Severity-Aware Benchmark for Contextual Image-Text Manipulation

**arXiv ID:** 2606.22339 | [PDF](https://arxiv.org/pdf/2606.22339v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 180. Learning a Normal World Model for Few-Shot Boundary-Calibrated Abnormality Detection

**arXiv ID:** 2606.22261 | [PDF](https://arxiv.org/pdf/2606.22261v1)

**作者:** Weizhi Nie `[一作]` (Tianjin University), Yuting Su `[通讯]` (Tianjin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种基于超图熵正则化的正常世界模型，用极少的异常样本仅校准阈值，实现了少样本异常检测。

**💡 创新点**

创新点在于先从大量正常数据学习高阶变量关系和时序动力学的正常世界，再通过熵感知能量量化偏离度，极大降低异常标签稀缺对性能的影响。

**🔧 技术方法**

技术包括上下文条件化超图网络、动态预测分支、跨变量一致性分支、潜在空间归一化以及三项熵感知能量的融合。

**📊 数据集**

实验使用 NASA C‑MAPSS 涡轮发动机退化基准（FD001–FD004）进行验证。

**📈 对比分析**

与传统单类、重建、预测以及深度图网络基线对比，零样本和少样本设置下均获得最高 AUROC，尤其在多条件多故障 FD004 上达 0.9983。

**⚠️ 局限性**

局限性包括仅在 C‑MAPSS 上验证，未覆盖更大规模或其他工业/临床数据；阈值校准仅为标量，缺乏更丰富的自适应阈值策略；依赖超图结构与上下文建模的手工设定。

---

## 181. Semantic Non-Assembly: Privacy by Architectural Inertness Under Component Exposure

**arXiv ID:** 2606.22311 | [PDF](https://arxiv.org/pdf/2606.22311v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 182. SamatNext v0.2-B: An Exploratory Study of RMS-Normalized Hybrid Decoders for Curriculum Retention in Small Code Models

**arXiv ID:** 2606.22248 | [PDF](https://arxiv.org/pdf/2606.22248v1)

**作者:** Samat Zharassov `[一作]` `[通讯]`, Samat Zharassov

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对一种新型混合序列解码器 SamatNext v0.2-B 在逐步 Python 代码生成课程中的顺序微调表现进行评估，并与参数匹配的标准 Transformer 进行对比。

**💡 创新点**

创新点在于将差分注意力层与 DeltaNet 启发的简化线性状态混合层交替堆叠，并加入 RMS 正则化和输出尺度校准，能够在顺序微调中显著降低灾难性遗忘，保持对中间语义层的高保留。

**🔧 技术方法**

主要技术包括：RMSNorm、DeltaNet 风格的简化线性状态混合器、差分注意力（含可学习的权重 α）、RoPE 位置编码、SwiGLU 前馈网络、输出尺度校准以及可选的验证头。

**📊 数据集**

使用了四阶段人工构造的 Python 代码生成课程：Stage 2A（语法基础）、Stage 2E（对抗性语法 holdout）、Stage 3（指令语义）以及 Stage 5（专业编码任务）并对 Stage 5 进行了人类可执行代码评估。

**📈 对比分析**

对比方法为：在 Stage 5 直接训练与在 Stage 2A→Stage 3→Stage 5 的顺序微调两种路径下，采用基于执行的 pass@1 评价指标。SamatNext 在 Stage 5 取得 100% pass，Stage 3 保留 98.8%，Stage 2E 12%；而参数匹配的 Transformer 在相同学习率下 Stage 5 仅 49.4% pass，Stage 3 仅 3.8% 保留；提升学习率后 Transformer Stage 5 约 97.6% 但 Stage 3 仍只有 6% 保留。

**⚠️ 局限性**

局限性包括：课程数据为合成且仅覆盖基础 Python；模型规模仅 356M，无法验证大规模可扩展性；未进行完整的学习率 sweep；未对 Mixer 交替模式、RoPE 影响、验证头及持续学习基线做 ablation；Stage 2E 语法保留率仍偏低；未与专门的持续学习算法进行对比。

---

## 183. Natural Language-Focused Software Engineering via Code-Documentation Equivalence

**arXiv ID:** 2606.22247 | [PDF](https://arxiv.org/pdf/2606.22247v1)

**作者:** Aryaz Eghbali `[一作]` (CISPA Helmholtz Center for Information Security), Michael Pradel `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出文档与代码等价性（documentation-to-code equivalence）概念，并给出定义和评估方法。

**💡 创新点**

创新点在于引入完整性与准确性并重的等价性属性，并设计基于LLM的迭代生成等价文档算法。

**🔧 技术方法**

主要技术是使用大语言模型（GPT‑4.1‑nano、GPT‑5‑nano、Gemini）进行文档→代码、代码→文档的循环推理与等价性判定。

**📊 数据集**

使用CoDocBench与DyPyBench这两个公开Python项目数据集进行实验。

**📈 对比分析**

通过执行测试判定代码等价；相较基线，等价文档生成成功率53.4%，精确率66.4%；在代码理解任务中准确率提升12.8–24.5%，在代码编辑任务中被人类开发者评价更有帮助。

**⚠️ 局限性**

局限性包括：等价判定依赖可执行测试、仅验证Python与docstring、用户研究样本规模有限。

---

## 184. Diffusion Integrated Gradients: Controllable Path Generation for Flexible Feature Attribution

**arXiv ID:** 2606.22314 | [PDF](https://arxiv.org/pdf/2606.22314v1)

**作者:** Soyeon Kim `[一作]` (KAIST), Jaesik Choi `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Diffusion Integrated Gradients（DiffIG），通过扩散模型生成可控的积分路径，改进特征归因的准确性与可解释性。

**💡 创新点**

将路径生成视为条件生成模型，并利用扩散模型配合能量/分类器引导实现全局最优且可在推理时控制路径属性（如faithfulness与complexity）。

**🔧 技术方法**

使用扩散概率模型、Stick‑Breaking Process生成路径样本、β‑VAE潜空间编码/解码、能量/分类器引导采样、多路径采样与多目标回归网络。

**📊 数据集**

在Oxford‑IIIT Pet和Mini‑ImageNet两个图像数据集上，采用VGG16、InceptionV1、ResNet18等预训练模型进行实验。

**📈 对比分析**

与七种主流路径归因方法（IG、GIG、IG²、AGI、EIG、MIG、SPI）在Insertion/Deletion/DiffID指标上进行对比，DiffIG在所有数据集和模型上均达到或超过对手，DiffID分数显著提升。

**⚠️ 局限性**

局限于固定的黑色基线，导致暗区归因效果差；目前未针对文本或表格等非视觉任务扩展。

---

## 185. From Speech to Text Corpora: Evaluating ASR-Based Data Acquisition for Low-Resource Fongbe and Hausa

**arXiv ID:** 2606.22274 | [PDF](https://arxiv.org/pdf/2606.22274v1)

**作者:** Mahounan Pericles Adjovi `[一作]` (Carnegie Mellon University Africa), Prasenjit Mitra `[通讯]` (Carnegie Mellon University Africa)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对非洲极低资源语言 Fongbe（声调丰富）和 Hausa（非声调），本研究通过精心构建 12.3 小时 Fongbe 语料库微调 MMS‑300M 模型，并收集、筛选并转录 1,553 条 YouTube 视频（总计 236 小时），生成 6,770 条音频‑文本对，旨在利用 ASR 扩充文本语料。

**💡 创新点**

创新点包括：① 证明仅 12 小时标注数据即可显著提升极低资源声调语言在标准评测集上的 WER（从 44.04% 降至 9.48%）；② 系统评估了在野生音频中模型置信度与实际质量的失衡，揭示声调语言的置信度指标失效；③ 对比了 Hausa 与 Fongbe 在 YouTube 语料上的可用性，提出声调语言需要进一步的后处理或专门的声调感知模型。

**🔧 技术方法**

采用技术：多语言 ASR 微调（MMS‑300M 使用 CTC 损失，学习率 1e‑4），对 Hausa 直接使用 NCAIR Whisper‑Small；音频预处理（16 kHz 单声道，20–25 秒分段），置信度统计，人工质量评估（0–100 评分），语言识别验证。

**📊 数据集**

数据集：① Fongbe 12.3 小时混合数据（ALFFA + Zenodo）用于微调；② 1,553 条 YouTube 视频（423 Fongbe，1,130 Hausa）用于构建语料库；③ 424 条精选视频（45.49 小时）用于转录并生成 6,770 条音频‑文本对。

**📈 对比分析**

比较方法：在 ALFFA 测试集上与 2016 Kaldi GMM/HMM 基线（44.04%）以及 MMS‑300M 预训练模型（23.7% WER、9.2% CER）对比；在 YouTube 语料上通过人工评估（Fongbe 平均 36.5/100，接受率 20%；Hausa 平均 57.4/100，接受率 60%）和置信度分布（Fongbe mean 0.84，Hausa mean 0.73）验证效果。性能上，Fongbe 在实验室数据上显著提升，但在野生音频中仍低于可直接用于语料构建的标准；Hausa 达到可接受水平。

**⚠️ 局限性**

局限性：① Fongbe 在野生音频中的低质量源于声调错误和代码切换，当前模型置信度不可靠；② 人工评估样本仅 50 条，样本量有限；③ 仅处理 424 条视频，未覆盖全部 1,553 条；④ Hausa 模型未进一步针对 YouTube 声学环境微调；⑤ 语料偏向公开视频，缺少离线或方言语音。

---

## 186. On the Expressive Power of Weight Quantization in Large Language Models

**arXiv ID:** 2606.22249 | [PDF](https://arxiv.org/pdf/2606.22249v1)

**作者:** Shao-Qun Zhang `[一作]` `[通讯]` (Nanjing University), Shao-Qun Zhang (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文从理论层面研究了大语言模型（LLM）中权重量化对表达能力的影响，提出了对 n>1 的权重量化 LLM 的通用逼近性、1 位量化模型表达崩塌性以及表达能力随量化比特数下降而呈多项式退化的三大结论；

**💡 创新点**

创新点在于：①证明 1.58 位（二进制对数3）为权重量化的下限；②在 n>1 的前提下给出权重量化 LLM 的通用逼近定理；③量化比特数与表达误差之间的多项式关系；

**🔧 技术方法**

主要技术包括：深度学习理论中的通用逼近与复杂度分析、绘制层架构、使用位运算实现高阶函数逼近、Lipschitz 连续性证明以及量化误差 δ 与逼近误差 ϵ 的关系；

**📊 数据集**

实验使用的公开数据集包括：仿真回归数据（[-1,1]^10→ℝ）、ImageNet 图像分类、WikiText2 语言建模以及 8 个零样本推理任务（ARC‑easy、ARC‑challenge、BoolQ、PIQA、Social‑IQA、HellaSwag、OBQA、WinoGrande）；

**📈 对比分析**

对比方法为将不同量化比特数（1–5 位）下的模型性能（分类准确率、困惑度、零样本推理准确率）与模型规模/参数量对照；结果显示 1 位量化性能明显下降，4–8 位量化性能接近浮点；

**⚠️ 局限性**

局限性包括：仅考虑权重量化而未涉及激活量化；理论主要针对 ReLU、注意力与 MLP 模块，未涵盖更复杂的变体；实验验证仅在特定任务和模型上，尚未证明在更大规模 LLM 或不同体系结构上的通用性。

---

## 187. Efficient Algorithms for Influence Maximization in General Models and Observed Cascades

**arXiv ID:** 2606.22241 | [PDF](https://arxiv.org/pdf/2606.22241v1)

**作者:** Fabian Spaeh `[一作]` (Boston University), Huy L. Nguyen `[通讯]` (Northeastern University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了影响力最大化在一般随机模型、观测级联模型和独立级联模型中的低适应性贪婪算法，并提出了新的估计与样本复杂度分析。

**💡 创新点**

创新点在于结合低适应性优化框架、经验方差自适应采样以及Sketching，首次实现近线性时间的观测级联算法，并在IC模型中将样本复杂度从O(n)降低到O(τ)的浓度界。

**🔧 技术方法**

主要技术包括低适应性贪婪、median‑of‑means估计、经验方差自适应采样、bottom‑b MinHash reachability sketch、以及IC模型的层级浓度分析。

**📊 数据集**

实验使用了 Facebook、Twitter、Google‑Plus 和 Pokec 等大规模社交网络数据集，节点数从数万到百万，边数从数十万到数千万。

**📈 对比分析**

与传统的理论最优算法和 lazy greedy 进行对比，实验表明在大预算下样本量降低约10倍、运行时间提升约200倍；在观测级联中，算法能在几分钟内完成 Pokec 100M 边网络的计算，保持近似最优解。

**⚠️ 局限性**

局限性包括对子模函数假设的依赖、需预先给定方差上界或使用自适应估计；在IC模型中仍未与最先进的 RIS 实现竞争；当传播步数 τ 接近 n 时，样本复杂度改进不明显。

---

## 188. Revelio: Cost-Efficient Agentic Memory Safety Vulnerability Detection For Repository-Scale Codebases

**arXiv ID:** 2606.22263 | [PDF](https://arxiv.org/pdf/2606.22263v1)

**作者:** Yiwei Hou `[一作]` (University of California, Berkeley), David Wagner `[通讯]` (University of California, Berkeley)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段的基于大型语言模型的内存安全漏洞检测框架，先用廉价模型生成漏洞假设，再用更强模型生成可执行PoV并通过 Sanitizer 验证；

**💡 创新点**

创新点在于将漏洞假设与可执行验证分离，利用成本低的 LLM 做覆盖广泛的代码审查，使用成本高的 LLM 仅在可验证候选上进行 PoV 生成并通过 deterministic sanitizer 减少误报；

**🔧 技术方法**

技术包括多代理 LLM 交互、结构化提问、工具调用（文件读写、PoV 测试、结果提交）、PoV 生成脚本、Sanitizer 验证；

**📊 数据集**

使用 OSS‑Fuzz 的 7 个成熟项目进行零日扫描，CyberGym 100 个内存安全实例进行基准测试，以及 10 个最新 CVE；

**📈 对比分析**

与 Claude Code、Codex、Sorcar 等通用 AI 编码代理对比，发现 发现 19 个新漏洞（含 7 CVE），CyberGym 上 175 个可验证漏洞（远超基线）且零误报，成本约 $300，平均每 16 美元/漏洞；

**⚠️ 局限性**

局限性：仅适用于可通过 Sanitizer 观察到的 C/C++ 内存安全漏洞，需可执行测试 harness，未覆盖业务逻辑、竞态等非内存错误；

---

## 189. On the Sparsity-Storage-Accuracy Tradeoff in Parsimoniously Activated Dictionary Learning

**arXiv ID:** 2606.22352 | [PDF](https://arxiv.org/pdf/2606.22352v1)

**作者:** Zihui Zhao `[一作]` (Tsinghua University), Yang Li `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了parsimoniously activated dictionary learning (PADL)，通过全局激活正则化实现稀疏表示与存储成本平衡，并给出了概率生成模型解释。

**💡 创新点**

创新点在于把全局激活约束映射为贝塔-伯努利隐变量的MAP优化，推导出激活阈值、稀疏度、存储与重构误差的三元权衡公式，实现了无经验调参的自动超参数估计。

**🔧 技术方法**

技术上采用贝塔先验、隐状态激活模型、MAP优化、交替投影梯度下降以及数据驱动的阈值求解，并在图像重构与VLM推理压缩中验证。

**📊 数据集**

使用CIFAR‑100、SVHN等图像补丁数据集以及LLaVA‑1.5、Video‑LLaVA等视觉‑语言模型的问答数据集进行实验。

**📈 对比分析**

与K‑SVD、DDL、GDDL、BPFA、IBP、CRsAE等基线比较，PADL在重构 RMSE/PSNR/SSIM 方面达到或优于竞争者，同时在VLM压缩中实现 8×–16× 的token压缩而保持近似性能，性能显著好于现有的 PruMerge、FreePruner。

**⚠️ 局限性**

局限性包括训练时间相对较长，模型假设为独立 Beta 先验，缺乏对更复杂数据分布的泛化证明，且在极大规模数据或多模态任务中仍需进一步验证。

---

## 190. Multi-cancer detection using a computationally efficient CNN with transfer learning

**arXiv ID:** 2606.22400 | [PDF](https://arxiv.org/pdf/2606.22400v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 191. Escaping the Variance Trap: Jacobian-Free Dynamics for Root-Finding Bilevel Optimization

**arXiv ID:** 2606.22433 | [PDF](https://arxiv.org/pdf/2606.22433v1)

**作者:** Zhiyu Li `[一作]` (University of Science and Technology of China), Davide Carbone `[通讯]` (École Normale Supérieure)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种根求解层级优化框架 RF‑BO，并用两时间尺度随机逼近 TTSA 解决高方差的隐式雅可比估计问题

**💡 创新点**

创新点在于：① 明确指出传统平方残差最小化会引发“方差陷阱”，② 通过不使用雅可比直接沿残差方向更新，构建无雅可比根求解方法；③ 给出在强凸和 PL 条件下的非渐近收敛保证

**🔧 技术方法**

核心技术包括两时间尺度随机逼近（TTSA）、无雅可比更新、残差裁剪/量化、Markovian 噪声下的收敛分析

**📊 数据集**

在多种任务上验证：SimCLR 代表性对比学习（ImageNet 预训练），Ode 控制（合成非线性系统），SAC 温度调节（Pendulum‑v1、HalfCheetah‑v4），多智能体均衡（5×5 GridWorld），WGAN‑GP 生成对抗（CIFAR‑10/LSUN）

**📈 对比分析**

与隐式梯度、惩罚法、上下文层级法以及单尺度残差法对比，RF‑BO 在 SimCLR 中提升 Top‑1 精度 2.6%、在 ODE 控制任务收敛 17 倍快、SAC 经验回报更稳定、WGAN‑GP Wasserstein 距离下降 11.1%，且在所有基准方法中保持更低的方差和更好的收敛稳定性

**⚠️ 局限性**

局限性：在噪声极低或确定性环境下隐式梯度仍可能更快；方法需要合适的时间尺度比例和裁剪参数；对极大维度超参数的可扩展性尚未完全验证；在极端非凸或多峰场景下可能需更细粒度的调优

---

## 192. Code Isn't Memory: A Structural Codebase Index Inside a Coding Agent

**arXiv ID:** 2606.22417 | [PDF](https://arxiv.org/pdf/2606.22417v1)

**作者:** Ishaan Bhola `[一作]` (SuperAGI Research), Mukunda NS `[通讯]` (SuperAGI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在固定模型的编码代理框架中，对加入结构化代码库索引进行因果消融实验，评估其对定位和解决率的影响。

**💡 创新点**

首次在真实部署的编码代理中进行泄漏审计、模型控制下的因果消融，证明索引显著提升定位精度并提升解决率。

**🔧 技术方法**

使用 Claude Opus 4.7 LLM、SuperCoder harness、结构化索引（向量+图+BM25）以及 OpenCode grep comparator。

**📊 数据集**

在 SWE-PolyBench Verified 与 SWE-bench Pro 的 91 个多语言实例（Go、Java、Python）上进行测试。

**📈 对比分析**

与同模型的无索引版本和 OpenCode grep 对比，SC‑ON 在定位(acc@5)提升约 39pp，解决率提升约 7.9pp，成本无显著增高，$/solved 更低。

**⚠️ 局限性**

样本量有限，语言覆盖不全，未评估在更大或更复杂多文件任务之外的泛化；仅在固定模型和硬件环境下验证。

---

## 193. Reinforcement learning to improve large language model-based automated code compliance systems

**arXiv ID:** 2606.22402 | [PDF](https://arxiv.org/pdf/2606.22402v1)

**作者:** Jack Wei Lun Shi `[一作]` (National University Of Singapore), Justin K. W. Yeoh `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 P4IR 框架，采用两阶段流程：先用监督微调（SFT）让大型语言模型（LLM）掌握建筑规范的语义知识，再通过 Group Relative Policy Optimization（GRPO）强化输出的代码骨架结构与准确度。

**💡 创新点**

创新点在于将 SFT 与 GRPO 结合并专门设计 Jaccard 相似度、语法与长度惩罚等奖励函数，能够在不依赖检索的零-shot 场景下显著提升代码骨架的结构完整性与可读性；同时首次系统阐释了 SFT 成熟度对后续 GRPO 效果的窗口效应。

**🔧 技术方法**

技术手段包括：Mistral‑7B 预训练模型的监督微调、GRPO 强化学习、奖励设计（结构 Jaccard、语法惩罚、长度惩罚）、树编辑距离与 Levenshtein 距离评估、CodeBERTScore 语义匹配、以及对模型生成时的 logit 熵分析。

**📊 数据集**

数据集为 732 对由新加坡建筑规范（SSW、SWD、SCDF 等）与对应 Python 代码骨架组成的真实案例，训练集 664 条，测试集 68 条。

**📈 对比分析**

与 Claude Opus、Claude Sonnet、GPT‑5.2、Qwen‑3‑Max、GLM‑4.7 等五大 SOTA LLM 通过检索增强的 1‑shot/2‑shot 方式比较，P4IR 在零-shot 情况下取得最低树编辑距离 20.49、最小 Levenshtein 372.66、Jaccard 距离 0.7081，并在 CodeBERTPrecision/Recall 上领先（0.84/0.84），显示在结构与语义两方面均优于所有对照模型。

**⚠️ 局限性**

局限性包括：仅优化代码骨架结构，未直接验证功能可执行性；需要标注的代码骨架数据，规模有限；未对其他 IR 形式或不同国家规范进行评估；与其他 RL 算法（如 PPO/RLHF）比较不足；以及 SFT 训练窗口的选择仍需经验调优。

---

## 194. Do Rigid-Body Simulators Dream of Soft Robots? Learning Contact-Rich Manipulation for Tendon-Driven Continuum Robots

**arXiv ID:** 2606.22397 | [PDF](https://arxiv.org/pdf/2606.22397v1)

**作者:** Chengnan Shentu `[一作]` (University of Toronto), Jessica Burgner-Kahrs `[通讯]` (University of Toronto)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了将连续体软机器人置于MuJoCo中的物理逼真仿真框架，并完成从仿真到真实机器人的零射程迁移。

**💡 创新点**

创新在于将Kirchhoff杆理论驱动的离散化与MuJoCo的刚体链模型相结合，实现了软机器人本地化的关节弹性、张力驱动和接触处理，并证明可用于高质量的学习与迁移。

**🔧 技术方法**

使用MuJoCo、Kirchhoff杆模型、离散化技术、ACT模仿学习、贝叶斯优化系统辨识、以及光学追踪与关节编码器的传感。

**📊 数据集**

使用3段9根绳索的TDCR在Franka 7-DoF臂上的实物数据以及通过键盘仿真收集的50条成功演示轨迹。

**📈 对比分析**

与SoRoSim的Cosserat杆参考以及真实硬件对比，静态误差<1%，动态误差<1%，零射程部署在两个接触丰富任务中成功率≈75%，与仿真相当或略优。

**⚠️ 局限性**

仅验证了细长杆的Kirchhoff假设，未处理更大弯曲或多物体复杂场景，且未加入完整形状反馈。

---

## 195. PlanBench-XL: Evaluating Long-Horizon Planning of LLM Tool-Use Agents in Large-Scale Tool Ecosystems

**arXiv ID:** 2606.22388 | [PDF](https://arxiv.org/pdf/2606.22388v1)

**作者:** Jiayu Liu `[一作]` (University of Illinois Urbana Champaign), Dilek Hakkani-Tür `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个名为 PlanBench‑XL 的交互式基准，用于评估 LLM 在大规模检索驱动工具生态中的长周期规划与自适应能力。

**💡 创新点**

创新点在于引入检索受限工具可见性、可选阻断机制以及前向/后向双向检索策略，系统模拟工具缺失、失效和误导情形。

**🔧 技术方法**

技术方面使用了 LLM 检索器、基于数据类型的工具生成与过滤、后端数据库生成工具响应、以及多轮交互执行模型。

**📊 数据集**

数据集方面在零售领域生成了 327 个任务、1665 个工具，并提供多维度数据类型与实例化后端。

**📈 对比分析**

比较方法：评估了 10 个主流 LLM（含 GPT、Gemini、Qwen3、Llama3、Deepseek 等），采用准确率、工具精度、交互轮数等指标，最高性能为 Gemini‑3.1‑Pro 77% 准确率，GPT‑5.4 在阻断情形下跌至 11%。

**⚠️ 局限性**

局限性：仅覆盖单一零售域，阻断机制为人工模拟且未涵盖更复杂动态故障，检索器为自定义，可能无法完全反映真实检索系统。

---

## 196. Structured Hyperedge Adaptation for Parameter-Efficient Fine-Tuning of Vision Transformers

**arXiv ID:** 2606.22383 | [PDF](https://arxiv.org/pdf/2606.22383v1)

**作者:** Edwin Kwadwo Tenagyei `[一作]` (Griffith University), Yongsheng Gao `[通讯]` (Griffith University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在视觉 Transformer 上提出一种新的参数高效微调方法 HyperAdapter，通过在超边空间对 token 进行分组并进行轻量化瓶颈适配，从而实现结构化的模型更新。

**💡 创新点**

创新点在于：① 将适配空间从传统的 token 维度转移到超边（组）维度；② 用 prototype‑based soft 路由构造软超图，实现多 token 的聚合与共享；③ 在超边层做轻量化瓶颈适配后再扩散回 token，注入显式的结构化 inductive bias。

**🔧 技术方法**

核心技术包括：超图构建与软路由、prototype‑based token‑hyperedge 关联、低秩瓶颈适配器、soft 扩散回 token、并行插入到 Attention 与 MLP 分支。

**📊 数据集**

使用的主要数据集：VTAB‑1K（19 个分类任务，分为 Natural、Specialized、Structured），5 个 few‑shot FGVC 数据集（FGVC‑Aircraft、Oxford‑Pets、Food‑101、Stanford‑Cars、Oxford‑Flowers102），并在 ViT‑B/16、ViT‑L/16、Swin‑Base 等预训练 backbone 上进行实验。

**📈 对比分析**

与多种 PEFT 方法（Adapter、LoRA、VPT‑Shallow/Deep、Res‑Tuning、RepAdapter、SSF、Arc 等）以及 full fine‑tuning 进行对比。HyperAdapter 在 VTAB‑1K 上平均精度提升约 1.3%–1.9%（尤其在 Structured 任务提升 1.9%），在 few‑shot FGVC 任务中在 1–4 shot 环境下获得最显著提升，参数量仅 0.44M（约 0.5% 的 ViT‑B/16 参数）。

**⚠️ 局限性**

局限性包括：仅在冻结的 ViT backbone 上验证，未探索动态超边或在线更新的可能；超边数和温度对性能有敏感性，需手动调参；在高分辨率、长序列或多模态任务中，超图构造的计算与存储开销可能显著增加。

---

## 197. Kiwano: A Cutting-Edge Open-Source Toolkit for Speaker Verification

**arXiv ID:** 2606.22369 | [PDF](https://arxiv.org/pdf/2606.22369v1)

**作者:** Mickael Rouvier `[一作]` (Avignon University), Pierre Michel Bousquet `[通讯]` (Avignon University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Kiwano开源工具包，提供统一、可复现的端到端说话人验证框架，包括前端模型、后端评分与归一化、数据管理与实验脚本，方便快速复现和对比实验。

**💡 创新点**

创新点在于：1) 统一的实验管线与可复现的recipes；2) 集成多种最新说话人嵌入模型（ECAPA2、ReDimNet、Xi-Vector等）与完整后端（PLDA、S-Norm、AS-Norm、fDA等）；3) 对训练动态、网络深度、批量规模以及多次训练可重复性进行系统评估。

**🔧 技术方法**

使用了PyTorch实现，配合DistributedDataParallel、Accelerate等分布式训练工具；采用AM-Softmax、AAM-Softmax、Sub-Center等margin损失；学习率与margin调度、数据增强（噪声、回声、速度扰动、SpecAugment）；后端包含Cosine/PLDA、LDA、whitening、长度归一化、S-Norm、AS-Norm、CMF、QMF等。

**📊 数据集**

训练使用VoxCeleb2‑dev（≈1M utterances）；评估在VoxCeleb1‑O/E/H（同域）以及跨域的CN‑Celeb、DiPCo、CommonBench。

**📈 对比分析**

与WeSpeaker、ESPnet‑SPK、3D‑Speaker等主流开源工具包进行公平比较（同训练集、相同评分方式），Kiwano在VoxCeleb1‑O/E/H上EER分别为0.46%、0.64%和1.13%，均低于对手；在综合实验中，最优模型EER可达0.34%。

**⚠️ 局限性**

局限性包括：1) 对极端跨域环境（如真实工业语料）的适应性仍需提升；2) 大模型（如ECAPA2、ReDimNet）计算与能耗较高；3) 目前未覆盖SSL预训练Encoder；4) 结果主要基于公开数据集，缺乏工业实际验证。

---

## 198. SPiralRoll: A Novel Adjustable-Stiffness Underactuated 3-DoF Joint with Torsion Springs for Rolling Robots

**arXiv ID:** 2606.22443 | [PDF](https://arxiv.org/pdf/2606.22443v1)

**作者:** Louis Keith `[一作]` (Cardiff University), and Seyed Amir Tafrishi `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了SPiralRoll——一种使用两轴电机驱动的扭转弹簧型低致动器，可产生旋转、径向伸缩和轴向自旋三种运动，并将其集成到球形滚行机器人上。

**💡 创新点**

将弧形弹簧分布在两侧，形成全弧和单弧两种配置，实现了仅用两轴电机获得三种可观测输出运动的低致动机制，并首次将此机制用于滚行机器人内部驱动。

**🔧 技术方法**

采用PLA 3D打印弹簧结构、Maxon 6V DC电机、BNO055惯性测量单元、Arduino + MATLAB控制与数据采集。

**📊 数据集**

实验数据（旋转角度、半径变化、轴向角度、加速度等）来自实际机械测试，无公开数据集。

**📈 对比分析**

通过对比全弧与单弧两种结构的径向变形幅度、动力响应和滚行演示，发现单弧在更大变形和更强惯性激励下更适合滚行，而全弧在负载支持和运动平滑性上更优。

**⚠️ 局限性**

缺乏完整的非线性动力学模型、控制策略，实验仅为开环演示，未进行定量运动学评估。

---

## 199. Curvature-aware 3D length estimation of greenhouse cucumbers using RGB-D imaging and cubic spline arc-length integration

**arXiv ID:** 2606.22439 | [PDF](https://arxiv.org/pdf/2606.22439v1)

**作者:** Manveen Kaur `[一作]` (University of Windsor), Shahpour Alirezaee `[通讯]` (University of Windsor)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研发了一套基于Intel RealSense D435的CucumberVision框架，实现了五种非接触式黄瓜长度测量方法，并在同一数据集上进行系统比较。

**💡 创新点**

创新点包括提出的3D中轴样条拟合（M5）实现连续弧长测量，和自适应方法选择器保证100%覆盖；同时纠正了深度相机内参导致的12–18%长度低估偏差。

**🔧 技术方法**

技术包括YOLO26n单阶段检测与实例分割、SAM ViT‑B精细掩模、PCA、骨架扫描、关键点回归和三次样条积分等。

**📊 数据集**

使用了由1500张RGB‑D图像构成的自标注数据集（约4360个实例），以及48个捕获样本的实测线程长度基准。

**📈 对比分析**

通过MAE、RMSE、MAPE等指标比较，M5取得MAE0.58 cm、MAPE4.13%，优于M1–M4（MAPE范围9.68–5.31%），显著提升准确度。

**⚠️ 局限性**

局限性包括样本范围有限（主要分布在3个尺寸簇），未涵盖多品种、不同传感器及更宽长度区间，且依赖单一深度相机，难以直接推广至更复杂环境。

---

## 200. Enhancing Road Safety: An IoT-Based Accident Detection and Prevention Mechanism

**arXiv ID:** 2606.22381 | [PDF](https://arxiv.org/pdf/2606.22381v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 201. Foreign co-affiliations and performance measurement of universities and the National Academy of Sciences of Ukraine, 2020-2023

**arXiv ID:** 2606.22379 | [PDF](https://arxiv.org/pdf/2606.22379v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 202. SVGym (SciVerseGym): An Environment for Reinforcement Learning and Bayesian Optimization in Crystal Discovery

**arXiv ID:** 2606.22425 | [PDF](https://arxiv.org/pdf/2606.22425v1)

**作者:** Bin Cao `[一作]` `[通讯]` (Hong Kong University of Science and Technology), Bin Cao (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提供了一个Gymnasium兼容的环境SciVerseGym，用于通过结构编辑动作进行闭环晶体发现。

**💡 创新点**

将晶体设计建模为马尔可夫决策过程，统一了结构编辑动作、奖励、评估后端等接口，便于不同搜索算法在相同物理假设下比较。

**🔧 技术方法**

使用Python、Gymnasium、ASE、机器学习势能（如SevenNet、MatterSim、ORB）、Phonopy、PyTorch Geometric等技术实现。

**📊 数据集**

基于公开晶体数据库ALEX‑MP‑20（可自定义数据集），并支持用户自定义结构池。

**📈 对比分析**

通过统一接口对强化学习、贝叶斯优化、进化搜索等算法进行基准测试，展示在同一后端下不同算法的性能差异。

**⚠️ 局限性**

缺乏完整的化学合理性验证、对能量上凸面和声子稳定性的依赖、默认奖励过于简单且易受后端选择影响。

---

## 203. Gold Points Sniper: Self-guided Visual Reasoning in VLM for Fine-grained Action Understanding

**arXiv ID:** 2606.22409 | [PDF](https://arxiv.org/pdf/2606.22409v1)

**作者:** Haodi Liu `[一作]` (Tsinghua University), Changshui Zhang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Gold Points Sniper (GPS) 框架，利用轻量级视觉-语言模型通过自导多模态推理实现细粒度人类动作理解。

**💡 创新点**

创新点包括：① 将金点提取、Socratic 质询验证、语义蕴含评估三大模块串联，形成完整的推理流程；② 通过指令调优让轻量模型学习自我验证与细化，从而获得信息密集且事实准确的描述；③ 引入语义蕴含评分作为客观评估指标，解决 VLM 输出的可解释性与真实性问题。

**🔧 技术方法**

主要技术：指令调优（Instruction Tuning）、多阶段链式推理（Gold Points Extraction → Validation → Socratic Questioning → Description）、自监督标注与评价（使用 LLaVA-OneVision-Qwen2-72B-ov-chat 进行标注，LLaMA-3.3-70B-Instruct 进行评估）、语义蕴含分类模型。

**📊 数据集**

使用基于 CAP 的数据集：训练集 14,281 帧（80 类），评估集 3,321 帧（40 类，含 20 类 held‑in 与 20 类 held‑out），并通过自动化标注生成金点、Q&A、描述。

**📈 对比分析**

在 Held‑in、Held‑out 与全数据集上，GPS‑增强的轻量 VLM（如 LLaVA-OV-7B）相较于原始模型提升 18–33%（例如从 48.26% 提升至 58.00%），与 GPT‑4o 的性能相近或更优；同时展示了良好的零-shot 泛化能力。

**⚠️ 局限性**

局限性：依赖昂贵的人工/API 预标注；在大模型上多阶段推理不够稳定，复杂提示可能导致性能下降；对极端小物体、极端视角或快速动作的识别仍存在误差，推理流程仍可进一步简化。

---

## 204. HFORD: Hybrid Forward Optimization and Reverse Design Method and Its Applications to On-Chip Millimeter-Wave Inductive Elements

**arXiv ID:** 2606.22393 | [PDF](https://arxiv.org/pdf/2606.22393v1)

**作者:** Yuzhen Song `[一作]` (Southeast University), Wei Hong `[通讯]` (Southeast University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种混合前向优化与逆向设计的HFORD方法，用于毫米波片上电感元件的目标到布局合成；

**💡 创新点**

创新点在于结合随机森林拓扑选择、VAE谱特征生成、MDN概率逆映射和PSO潜在空间搜索，实现了设备级与电路级规范的统一化流程，并通过稀疏拟合采样和物理参数化降低数据和计算成本；

**🔧 技术方法**

采用随机森林、变分自编码器（VAE）、混合密度网络（MDN）和粒子群优化（PSO）等机器学习与优化技术；

**📊 数据集**

使用在40‑nm RF CMOS工艺下生成的约1.5万条（包含1,000条变换器与5,000条电感）全波EM仿真数据集，并通过迁移学习共享特征；

**📈 对比分析**

与传统遗传算法（GA）、聚类算法（COA）和基于优化的OSIAS等方法对比，HFORD在保持设计规范（DRC）合规的前提下，优化时间从数小时缩短到数分钟，且模型误差低于0.06，性能与基准相当或更好；

**⚠️ 局限性**

局限性在于依赖大量高质量的仿真数据集，跨工艺迁移时需重新训练；对复杂拓扑的后期调优仍需手工干预；

---

## 205. Words as Difference Makers: How Large Language Models Determine Causal Structure in Text

**arXiv ID:** 2606.22430 | [PDF](https://arxiv.org/pdf/2606.22430v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 206. Gen2Balance: Generative Balancing for Long-Tailed Video Action Recognition

**arXiv ID:** 2606.22416 | [PDF](https://arxiv.org/pdf/2606.22416v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 207. Physics-Informed Neural Operator for Speech Production Analysis

**arXiv ID:** 2606.22364 | [PDF](https://arxiv.org/pdf/2606.22364v1)

**作者:** Kazuya Yokota `[一作]` (Nagaoka University of Technology), Sidney Fels `[通讯]` (University of British Columbia)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种基于Physics-informed Neural Operator（PINO）的声学模型，能够快速模拟声道与声带耦合的声学过程，输出声带振动频率、气流量及唇部声压。

**💡 创新点**

首次将PI-DeepONet（物理信息深度算子网络）应用于语音产生分析，利用声道形状作为输入特征实现无监督训练，显著提升了推理速度和并行计算效率。

**🔧 技术方法**

使用PI-DeepONet架构，结合两质量声带模型与一维声道模型，采用物理约束损失、Fourier特征时间归一化、硬约束耦合约束，并在训练中无监督地求解波动方程。

**📊 数据集**

使用Arai报告的五个元音（/a/、/i/、/u/、/e/、/o/）的声道截面积函数（16点采样）作为输入，配合已知物理参数进行训练。

**📈 对比分析**

与传统的Runge-Kutta/Finite-Difference（RK4-FDM）数值求解器对比；声带振动频率误差<0.25%，气流波形误差≈1%，唇部声压误差≈6%；推理时间约0.039 s/元音，远快于迭代求解。

**⚠️ 局限性**

仅能处理已训练的声道形状且局限于稳态周期性分析；对高频成分的光谱偏差较大；未考虑非稳态或三维几何，训练时间相对较长。

---

## 208. ORBIT: Training-Free Multi-Attribute Behavioral Steering via Orthogonal Subspace Rotation

**arXiv ID:** 2606.22357 | [PDF](https://arxiv.org/pdf/2606.22357v1)

**作者:** Narges Ghasemi `[一作]` (University of Southern California), Jonathan May `[通讯]` (University of Southern California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种无训练的多属性激活调节方法 MAT-STEER，能够在推理时同时控制多种行为属性。

**💡 创新点**

创新点在于：①利用 SVD 构造所有属性的联合子空间；②在该子空间内执行一次保范数旋转，避免了传统向量相加导致的范数失衡与方向冲突；③使用自适应门控仅在需要的 token 位置进行调节，并可选加性提升进一步增强弱属性的影响。

**🔧 技术方法**

技术手段包括角度调节（Angular Steering）、奇异值分解（SVD）构建子空间、对 token 的自适应门控、以及可选的线性加性提升；所有操作均在推理时完成，无需模型再训练。

**📊 数据集**

使用的评估数据集有 TraitFactory（13 个人格特质，22 题类）和 ToneBank（5 会话语调，18 题类），并在 Llama‑3.2‑3B、Qwen‑2.5‑7B、Llama‑3.1‑8B 三大模型上进行实验。

**📈 对比分析**

实验与 CAA、K‑Steering、Mat‑Steer 等基线（以及几种朴素几何组合）对比，MAT‑STEER 在多属性（K=1~5）下取得更高的评审分数、正向几何增益和联合成功率，同时保持与未调节模型相近的连贯性与通用能力。

**⚠️ 局限性**

局限性：①采用线性向量近似属性，可能无法充分捕捉高度非线性或复杂交互的属性；②目前实验仅对最多 5 种属性进行评估，未探讨更大属性集合的可扩展性；③对极端属性组合的鲁棒性和计算开销仍有待进一步研究。

---

## 209. MetaPS: Adaptive Programmatic Strategy Selection for Market Agents

**arXiv ID:** 2606.22385 | [PDF](https://arxiv.org/pdf/2606.22385v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 210. DreamUV: Unwrap Artist-like UV by End-to-End Flow Matching

**arXiv ID:** 2606.22445 | [PDF](https://arxiv.org/pdf/2606.22445v1)

**作者:** Quanyuan Ruan `[一作]` (South China University of Technology), Xifeng Gao `[通讯]` (Lightspeed)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 DreamUV，一种基于 Flow Matching 的端到端框架，用来生成符合专业艺术家风格的 UV 布局；

**💡 创新点**

创新点包括：①将 UV 解包视为从噪声到目标分布的流匹配；②引入边界感知加权训练，强化边缘直线化；③提出 Model‑in‑the‑Loop 微调，弥合训练与采样的离散误差；

**🔧 技术方法**

使用技术包括：流匹配（Flow Matching）生成模型、基于图神经网络的 UV 速度回归、边界加权损失与 EMA‑驱动的多步微调；

**📊 数据集**

使用 Lightspeed Games UV 数据集，包含 359,301 张由专业艺术家手工制作的 UV 布局；

**📈 对比分析**

与经典 ABF、ABF++、xatlas 及学习基 FAM 等方法比较，DreamUV 在保持低畸变、零重叠、单岛布局的前提下，显著提升了边界直线度、轴对齐率和贴图填充率；用户研究显示对比基线方法，首选率在 69–96% 之间；

**⚠️ 局限性**

局限性包括：仅依赖已有切割（seam）信息；未实现自动 seam 生成或 UV 打包；对极端噪声或非流形网格的鲁棒性待进一步验证。

---

## 211. ARIA: A Causal-Aware Framework for Rescuing LLM Reasoning in Trustworthy Materials Discovery

**arXiv ID:** 2606.22375 | [PDF](https://arxiv.org/pdf/2606.22375v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 212. Bypassing Minimization Bias: A Shift-Invariant Variance Estimator for Off-Equilibrium Local Learning Coefficients

**arXiv ID:** 2606.22389 | [PDF](https://arxiv.org/pdf/2606.22389v1)

**作者:** Yingjia Cai `[一作]` `[通讯]`, Yingjia Cai

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Shift-Invariant Variance Estimator (SIVE)，用于在线跟踪深度网络训练过程中局部学习系数（LLC）并消除对未知局部最低点的依赖。

**💡 创新点**

创新点在于利用方差算子双重抵消未知最低点，并通过总方差定律校正 mini‑batch 评估噪声，使得在非平衡训练阶段能够无基准地估计 LLc。

**🔧 技术方法**

技术方法包括局部 SGLD 采样、方差观测、基于总方差的噪声校正、理论推导与实验验证。

**📊 数据集**

使用可解析的多重模型（1D 二次势、2D 乘积二次势）进行控制实验，以及 MNIST 数据集上的 3 层 ReLU MLP 进行真实网络实验。

**📈 对比分析**

与 oracle、naive mean、raw variance 等基线比较，SIVE 在 toy 模型中准确恢复理论 RLCT，且在 MNIST 训练中捕获非单调的结构相变，性能明显优于传统估计。

**⚠️ 局限性**

局限性包括对评估批量大小的计算成本、对 SGLD 离散误差的敏感性，以及目前仅在单局部图谱下验证，尚未实现全局动态优化和更广泛的数据集测试。

---

## 213. Curvature-Adaptive Consistency Flow Matching: Autonomous Trajectory Optimization via Reinforcement Learning

**arXiv ID:** 2606.22394 | [PDF](https://arxiv.org/pdf/2606.22394v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 214. Hardwired Pattern Formation by Mobile Robots with Common Unit Distance

**arXiv ID:** 2606.22386 | [PDF](https://arxiv.org/pdf/2606.22386v1)

**作者:** Yuta Kojima `[一作]` (Kyushu University), Yukiko Yamauchi `[通讯]` (Kyushu University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

研究了硬连线模式形成问题，并提出了针对两台机器人和多于四台半同步机器人的算法

**💡 创新点**

首次提出硬连线模式形成问题，并设计了 Z-移动和大小调整算法，以及通过灯光实现的异步解决方案

**🔧 技术方法**

利用对称性分析、Z-移动、非刚性移动、灯光通信以及将集合聚合算法映射到硬连线模式的技术

**📊 数据集**

本研究为理论研究，无使用实际数据集

**📈 对比分析**

通过数学证明和对比已知聚合算法的可达性，未进行实验性能评估

**⚠️ 局限性**

结果局限于二维、确定性算法，无法处理异步下两台机器人的情况和概率算法，且需要单位距离共享

---

## 215. When Is a Columnar Scan Bandwidth-Bound? A Decode-Throughput Law and Its Cross-Hardware Validation

**arXiv ID:** 2606.22423 | [PDF](https://arxiv.org/pdf/2606.22423v1)

**作者:** Madhulatha Mandarapu `[一作]` (VaidhyaMegha Private Limited), Sandeep Kunkunuru `[通讯]` (VaidhyaMegha Private Limited)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究列式扫描中解码、过滤与聚合的内存带宽占用，并给出了一个基于值解码吞吐量的闭式预测公式。

**💡 创新点**

提出 f = min(1, b/(8β)) 的带宽占比模型，证明其对不同位宽、分支策略和区间跳过具有通用性，并通过单参数调优实现跨硬件预测。

**🔧 技术方法**

利用 roofline 模型、SIMD 向量化（转置布局）、FastLanes、RLE 与区间跳过等实现手段，对列式压缩数据进行解码与聚合。

**📊 数据集**

使用 67,108,864 个 32 位整数的自定义压缩列（不同位宽、随机与排序数据）作为实验数据集。

**📈 对比分析**

在 x86/AVX2 与 Apple M4/NEON 上做单线程实验，测得带宽占比误差均 ≤0.027；模型还能通过极限 b→∞ 预测 FastLanes 的“解码无成本”结论。

**⚠️ 局限性**

仅限单线程、整数列、值吞吐量低于 SOTA，未覆盖字符串/嵌套类型；Apple M4 测试为泛化点，未提供精确的 turbo/核心固定控制。

---

## 216. Knowledge-Graph Grounding Helps LLMs Only for Out-of-Training Knowledge: A Controlled Study on Clinical Question Answering

**arXiv ID:** 2606.22419 | [PDF](https://arxiv.org/pdf/2606.22419v1)

**作者:** Madhulatha Mandarapu `[一作]` (Samyama), Sandeep Kunkunuru `[通讯]` (Samyama)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在医学基准上，使用结构化知识图谱（KG）对大型语言模型（LLM）进行 grounding 是否能提升性能，并复现了先前的 Nature Medicine 研究结果；

**💡 创新点**

提出了“知识边界定律”：KG grounding 仅在知识不在模型训练或可推断范围内时才会显著提升，公开 KG 的事实对模型已无增益；

**🔧 技术方法**

使用 samyama‑graph（向量检索+OpenCypher）和 agentic Cypher 写作器，构建了四种 grounding 架构（A0、A_kg、A_agent、A_det、A_GAK）；

**📊 数据集**

评估数据集包括 MedQA、HealthBench（全集与 Consensus 子集）、公共生物医学 KG PrimeKG、合成对抗 KG 以及混合 KG；

**📈 对比分析**

结果显示：在 PrimeKG 上无显著提升（Δ≤3.4），但在合成或混合 KG 的非训练知识子集上，grounding 能将准确率从约20%提升至≈100%（Δ≈70–80），证明 grounding 在缺失知识时非常有效；

**⚠️ 局限性**

局限性包括：合成 KG 并非真实临床数据，未在真实机构私有 KG 上验证；查询生成依赖 gpt‑4.1，未探究弱模型写 Cypher 的效果；并且仅评估多选题的准确率，未考察推理质量与可解释性。

---

## 217. First-Token Broadcasters: Mechanistic Origins of Language Identity and Distributed Robustness in Transformers

**arXiv ID:** 2606.22361 | [PDF](https://arxiv.org/pdf/2606.22361v1)

**作者:** Arjun Pillai `[一作]` (Irvington High School), Anjelo Jann Laroza `[通讯]` (Mapua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对多语言模型每个注意力头进行零化干预，研究语言身份保持机制并提出LIHA方法。

**💡 创新点**

发现“首词广播”网络、层级补偿机制，并首次用受控基线‑指令对比证实指令微调能将语言身份信号集中至模型最早层。

**🔧 技术方法**

使用零化干预、语言切换率计算、重分配显著性检验以及线性探测等技术。

**📊 数据集**

采用 Flores‑200、GPT‑2、Qwen2.5‑1.5B、BLOOM‑560m 等模型与 2,700 条多语言提示数据。

**📈 对比分析**

与基线 GPT‑2、BLOOM 以及 Qwen 基础/指令版本对比；指令版表现出最高切换率峰值且更易被干预，说明训练方式对语言身份电路有显著影响。

**⚠️ 局限性**

数据量有限导致置信区间宽，零化干预无法区分语言促进与抑制头，GQA 结构对干预解释需要进一步验证。

---

## 218. Following the Flow: Advection-Consistent Modeling for Event-based Small Object Detection

**arXiv ID:** 2606.22378 | [PDF](https://arxiv.org/pdf/2606.22378v1)

**作者:** Wen Guo `[一作]` (Shandong Technology and Business University), Wuzhou Quan `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于流体力学的流动一致性建模框架PACT，用以恢复事件摄像机中稀疏弱响应的连续轨迹，提升小目标检测性能。

**💡 创新点**

创新点在于将事件响应的时间演化视为在局部速度场上的advection传输，并通过advection一致性约束显式保持弱响应的时间连续性，进而抑制噪声。

**🔧 技术方法**

技术上结合了轨迹引导特征提取（T‑FE）、Advection‑Based Trajectory Consistency（ATC）以及Advection‑Consistent Feature Reconstruction（A‑FR），并采用可微分的速度场估计与传输算子。

**📊 数据集**

使用了大规模事件小目标检测基准EV‑UAV数据集，包含147条序列、230万事件级注释。

**📈 对比分析**

与多种帧基、事件基和点云分割方法（如EV‑SpSegNet、KPConv、RVT等）对比，PACT在IoU和准确率上分别提升至75.90%/80.05%，P_d达到91.84%，F_a仅0.76×10⁻⁴，且参数量和推理时间仅为2.9M和58 ms。

**⚠️ 局限性**

局限性在于仅使用一次一阶速度场模型，难以处理加速或多速率交叉运动的场景，需要进一步扩展为更高阶或时变传输模型。

---

## 219. A Taxonomy of Conceptual Alignment in Human-Robot Dialogue

**arXiv ID:** 2606.22360 | [PDF](https://arxiv.org/pdf/2606.22360v1)

**作者:** Shengchen Zhang `[一作]` (Tongji University), Weiwei Guo `[通讯]` (Tongji University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种以设计为中心的概念对齐对话分类法，包括对齐触发条件、对齐层级以及对话行为模式，并通过人类双人对话实验验证该框架。

**💡 创新点**

创新点在于将概念对齐视为双向共建过程，提出了基于触发条件（术语、行为、情境）和对齐层级（感知、例子、规则、框架、评估）的两维分类体系，并配套细粒度的对话行为编码方案，为 HRI 研究提供可操作的分析与设计工具。

**🔧 技术方法**

使用的技术主要是语料库的手工转写、文献综述构建的对话行为编码表，以及基于此表的归纳式主题分析；并未涉及深度学习或机器人感知算法。

**📊 数据集**

使用的数据集是 48 条人类双人对话（共 2677 轮），来自两种概念对齐任务（分类与形成），每条对话约 57 轮，实验对象为 24 对人类参与者。

**📈 对比分析**

论文未提供定量比较或性能指标；其贡献是定性分析与框架构建，验证方法通过案例展示与对比讨论，未进行算法评估。

**⚠️ 局限性**

局限性包括：仅基于人类对话，未考虑机器人感知、身体约束和实时交互；任务受限于控制环境，可能不涵盖开放式或长期交互中的对齐动态；缺乏对实际 HRI 对话语料的验证，无法证明框架在机器人系统中的可操作性。

---

## 220. Interest Entanglement: The Hidden Barrier to Blind Super-Resolution Optimization

**arXiv ID:** 2606.22353 | [PDF](https://arxiv.org/pdf/2606.22353v1)

**作者:** Junxiong Lin `[一作]` (Fudan University), Wenqiang Zhang `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出共享特征表示的超分辨率框架SFR，通过分支解耦回归损失与感知损失，实现图像细节恢复与感知质量的平衡。

**💡 创新点**

从频域角度揭示回归与感知损失的兴趣纠缠，并设计SFR框架与InfoSqueeze特征变换模块，实现多目标学习的解耦与共享。

**🔧 技术方法**

采用多分支网络（回归分支、感知分支、共享分支）和InfoSqueeze模块，使用L1/Perceptual损失、PixelShuffle上采样及频域梯度分析。

**📊 数据集**

在DIV2K、BSDS100、Urban100、T91、DPED（blackberry/iphone/sony）等五个代表性数据集上进行实验。

**📈 对比分析**

与DAN、DCLS、DASR、MANet、SwinIR、HAT、RGT、RealESRGAN、ResShift等方法对比，SFR在PSNR、SSIM上提升约1–2 dB，LPIPS下降约0.1，显示在多数据集上均优于前人。

**⚠️ 局限性**

主要针对盲SR任务，对更复杂降质的泛化能力有限；InfoSqueeze需手动设置压缩比例，且多分支结构增加模型复杂度。

---

## 221. Distribution-Aware Robust Bilevel Optimization: Quantile-Guided Huber Updates in Two-Timescale Stochastic Approximation

**arXiv ID:** 2606.22436 | [PDF](https://arxiv.org/pdf/2606.22436v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 222. Enhancing LLMs for Graph Tasks via Graph-aware LoRA Generation

**arXiv ID:** 2606.22429 | [PDF](https://arxiv.org/pdf/2606.22429v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 223. Curiosity as Linguistic Intervention: Using LLM Tutoring Dialogues to Influence Exploratory Learning Behavior

**arXiv ID:** 2606.22349 | [PDF](https://arxiv.org/pdf/2606.22349v1)

**作者:** Gevindu Ganganath `[一作]` (Singapore Management University), Thivya Kandappu `[通讯]` (Singapore Management University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在大型语言模型的对话中加入可调节的语言干预（新颖性、复杂性、冲突、不确定性）来激发学习者的探索好奇心，并在270场对话中验证其对学习者探索行为的影响。

**💡 创新点**

将Berlyne的认知张力变量转化为可在推理时动态选择的语言操作符；构建学习者中心的多维评价框架；证明LLM对话可作为可扩展的实验工具来研究语言对探索学习的作用。

**🔧 技术方法**

推理时提示调节器（prompt‑based controller）实现五种操作符；LLM‑as‑a‑judge 评估学习者与教师维度；适应性操作符选择基于对话历史；使用Claude、Gemini、GPT三大LLM并对话跨学科与复杂度进行实验。

**📊 数据集**

270条对话数据，来自45名学生，覆盖STEM、社会科学、艺术人文三学科，三种复杂度；实验使用统一的学习者启动提示与对话脚本。

**📈 对比分析**

与基线未调节的对话和各LLM自带的学习模式对比；采用多模型评估器（Claude、Gemini、GPT）进行九次评估并取平均；统计检验显示好奇心调制在所有模型/域/复杂度下提升学习者探索维度（L1‑L4）21–35%且显著；对话回合数提升约2.4倍；对教师质量提升表现不一。

**⚠️ 局限性**

操作符选择采用规则化控制，缺乏学习化与个性化；依赖LLM‑as‑a‑judge 评估可能引入偏差；实验仅基于文本对话，未结合多模态生理或眼动数据验证学习者内部状态。

---

## 224. Scholarly Production and Public Health Determinants in Context of Funding: The Case of IoMT Research:

**arXiv ID:** 2606.22411 | [PDF](https://arxiv.org/pdf/2606.22411v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 225. Random Reed--Solomon Codes Correcting Permutations, Insertions, and Deletions over Polynomial-Size Alphabets

**arXiv ID:** 2606.22344 | [PDF](https://arxiv.org/pdf/2606.22344v1)

**作者:** Yijun Zhang `[一作]` (University of Science and Technology of China), Gennian Ge `[通讯]` (Capital Normal University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了在任意坐标置换后再发生插入删除错误的环境下，Reed–Solomon 码的容错能力，并给出了随机构造的多项式字母表 RS 码、相应的字母表下界以及两维 RS 码的平均线性时间解码器。

**💡 创新点**

创新点包括：① 通过允许额外的 ϵn 错误裕度，将对抗置换‑插删错误所需的字母表从指数级降低到多项式级；② 提出了一条适用于任意 q‑符号码的组合下界，证明了在该模型下常数速率码不能使用线性字母表；③ 为已知的两维 RS 码构造了平均 O(n) 的解码算法，突破了仅能在删除场景下实现线性时间的限制。

**🔧 技术方法**

技术手段主要是代数秩框架（V‑矩阵和 A‑矩阵）结合错误位指针（faulty index）方法，利用 Schwarz–Zippel 变体给出随机评估点的满秩概率；在解码层面采用逆比率子例程、哈希表和局部多数投票策略。

**📊 数据集**

实验数据集：无实际实验，研究完全基于理论分析；使用随机选取的 α 评估向量以及 F_q^3 字母表作为构造对象。

**📈 对比分析**

与之前仅能在指数字母表上达到半 Singleton 边界的 RS 码相比，本工作实现了多项式字母表；与普通插删模型下可实现线性字母表的 RS 码相比，置换‑插删模型引入了额外的字母表障碍；两维 RS 码的解码时间从 O(n^3) 降至平均 O(n)，显著提升。

**⚠️ 局限性**

局限性：证明过程中出现阶乘和指数计数损失，导致字母表上界仍远高于线性；结果仅给出随机存在性，缺乏显式构造；低效解码仅针对两维 RS 码，且在更一般维度下尚无高效解码方案；下界基于组合包容，可能非最优。

---

## 226. MMGist: A Comprehensive Multimodal Benchmark for 2027

**arXiv ID:** 2606.22437 | [PDF](https://arxiv.org/pdf/2606.22437v1)

**作者:** Wenzhen Yuan `[一作]` (Shanghai Jiao Tong University), Yuzhuo Fu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对18个现有视觉语言基准进行系统审计，识别弱视觉依赖、项目饱和和伪难题，并构建MMGist，涵盖七个能力维度共7,262道题；

**💡 创新点**

提出三阶段质量控制流水线（文本消融、跨模型饱和筛选、异常检测+人工复核），实现高效、可靠且可区分的评测；

**🔧 技术方法**

采用文本消融评估、跨模型准确率过滤、规则召回+多模型裁决、专家复核等技术；

**📊 数据集**

利用18个公开基准（共23,250道题）作为源数据；

**📈 对比分析**

在27种大型视觉语言模型上进行评测，MMGist保持Spearman ρ=0.98的排名一致性，提升模型间判别度78%，同时将评测项数缩减69%；

**⚠️ 局限性**

局限在于仅英文、闭合式问答、依赖当前模型面板，异常复核主观性可能影响筛选结果。

---

## 227. Multigrid Training for Molecular Generation using Graph Neural Networks

**arXiv ID:** 2606.22377 | [PDF](https://arxiv.org/pdf/2606.22377v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 228. Formal-Method-Guided Vibe Coding: Closing the Verification Loop on AI-Generated Safety-Critical Software Through Model-Driven Engineering

**arXiv ID:** 2606.22413 | [PDF](https://arxiv.org/pdf/2606.22413v1)

**作者:** Ran Wei `[一作]` (Lancaster University), Xiangyang Ji `[通讯]` (Tsinghua University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一套闭环流水线Forge，将LLM生成的Java代码通过模型驱动工程(MDE)链路转换为可验证的形式化模型，并在三种正式验证工具（Dafny、FDR4、Isabelle/HOL）上自动生成验证证据；

**💡 创新点**

创新点在于：1）将LLM生成代码与现有MDE基础设施相结合，形成草稿-判别-反馈闭环；2）实现了Java到EMF模型、RoboChart、Dafny、CSP-M和Z-Machine的自动转换；3）构建了跨工具的补全反馈机制，让验证失败直接转化为LLM可读的修正提示；

**🔧 技术方法**

使用技术包括：LLM（Anthropic Claude Code）进行自然语言代码生成；Spoon+EMF实现Java文本到模型转换；Epsilon（ETL/EGL）实现模型间转换和文本生成；Dafny+Z3进行符号验证；FDR4进行CSP细化检查；Isabelle/HOL+Z-Machine进行定理证明；

**📊 数据集**

使用的数据集为三个外部公开案例研究：SRanger（单控制器移动机器人）、LRE（AUV安全总线）、Chemical Detector（双控制器化学检测器），均来自RoboStar/Robochart公开案例库；

**📈 对比分析**

对比方法包括：冷启动单通道实验（无验证反馈）、编译仅消除验证反馈实验以及完整闭环实验；实验结果显示完整闭环在15次独立运行中均以2-3次迭代收敛，单通道和编译仅实验均未成功，验证反馈是推动收敛的关键；

**⚠️ 局限性**

局限性包括：仅支持Java且受限于提取可提取的子集；仅验证反应式安全控制器，无法处理共享状态或混合自动机；对LLM的依赖仅验证Claude，未评估其他模型；需要人工审阅反馈并手动启动迭代；未完成工具链的认证与信任边界验证，仍需进一步研究和标准化。

---

## 229. Asymptotic Signal Subspace Recovery in Softmax Attention Models

**arXiv ID:** 2606.22406 | [PDF](https://arxiv.org/pdf/2606.22406v1)

**作者:** Lan V. Truong `[一作]` `[通讯]` (Ho Chi Minh City University of Technology), Lan V. Truong (Ho Chi Minh City University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

研究了一种风格化的softmax注意力模型，通过随机梯度上升学习查询向量，从信息性和干扰性标记的集合中提取信号。

**💡 创新点**

提供了一个严格的理论基础，解释了注意力机制在高维噪声环境中作为信号提取过程的工作原理，并展示了注意力如何在大量噪声中发现相关信息。

**🔧 技术方法**

使用了随机梯度上升、随机逼近和动力系统理论等技术。

**📊 数据集**

使用了包含信息性和干扰性标记的合成数据集，具体形式为 x_i = v_i θ_d ξ_d + z_i，其中 z_i 是独立同分布的高斯噪声向量。

**📈 对比分析**

通过与现有的注意力机制理论进行比较，证明了所提出的模型在高维标记集合中能够几乎确定地收敛到潜在信号子空间，性能表现出色，能够有效区分信息性标记和干扰性标记。

**⚠️ 局限性**

模型的局限性在于其假设了高维标记的分布和信号强度的特定条件，可能在实际应用中受到限制。

---

## 230. ATCCaps: A Call-Sign-Aware Speech Dataset for Air Traffic Control Recognition

**arXiv ID:** 2606.22399 | [PDF](https://arxiv.org/pdf/2606.22399v1)

**作者:** Dongdong Li `[一作]` (East China University of Science and Technology), Zhe Wang `[通讯]` (East China University of Science and Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了ATCCaps数据集，整合了真实ATC语音、转录、呼号标注和ATC风格字幕，并提供了基准评估结果。

**💡 创新点**

首次结合ADS‑B元数据、置信度感知转录解析、呼号规范化、规则过滤与LLM生成字幕，提供大规模、呼号监督且字幕级对齐的ATC语音数据。

**🔧 技术方法**

采用置信度加权转录解析、呼号字典构建、噪声/时长过滤、DeepSeek LLM生成字幕，以及Whisper与CLAP进行评测。

**📊 数据集**

以ATCO2-PL（自动转录）与ATCO2-test（人工注释）为来源，构成训练、验证和评测三份集。

**📈 对比分析**

用Whisper评估ASR（WER 0.1485），用CLAP评估呼号匹配（ACC≈0.89）和音频‑文本检索（R@1≈0.37），在自有评测集上表现良好，但跨域UWB‑ATCC效果明显下降。

**⚠️ 局限性**

过滤率极低、部分字幕缺少呼号/数字准确性、LLM生成字幕存在不一致、缺乏完整人工审核以及实体级ASR评估指标。

---

## 231. Reference-Free Assessment of Physical Consistency in World Model-based Video Generation

**arXiv ID:** 2606.22363 | [PDF](https://arxiv.org/pdf/2606.22363v1)

**作者:** Yun Oh `[一作]` (Hanyang University), Sukmin Yun `[通讯]` (Hanyang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种无参考的物理一致性评估框架，利用相对和绝对指标筛选生成视频，以提高视觉语言学习模型在机器人模拟任务中的真实成功率。

**💡 创新点**

创新点在于将DROID‑SLAM的3D重投影误差与SEA‑RAFT的光度一致性通过Shannon熵加权整合成相对异常分数，并进一步构建像素级热图定位物理失真；该方法在不依赖地面真值的情况下显著缩小模拟与真实世界的差距。

**🔧 技术方法**

采用SLAM技术（DROID‑SLAM）计算3D一致性、光流网络（SEA‑RAFT）计算光度一致性，利用熵权重得到相对异常分数；使用GPT‑4o作为奖励模型评估任务成功率；生成热图实现绝对异常定位。

**📊 数据集**

主要数据集包括OpenVLA的真实世界实验视频（BridgeData V2）、WorldGym生成的5项任务视频，以及Sora、Grok、LucyEdit、Stable Diffusion等模型生成的跨模型视频。

**📈 对比分析**

将生成视频按相对异常分数分为高低异常组，用GPT‑4o评估任务成功率；低异常组的平均成功率提升至79%，接近真实世界的79%，显著优于随机组70%和高异常组70%；跨模型评估显示方法对结构极端不一致的视频检测存在局限。

**⚠️ 局限性**

局限性包括：依赖SLAM能建立像素对应，极端结构或无接触失败时误报低异常；无法处理导致零误差的严重结构崩溃；仅在OpenVLA上验证，未探讨对其他VLA或世界模型的泛化；缺乏语义或接触感知的补充信号。

---

## 232. Reliability-Guided Adaptive Ensembling for Robust Test-Time Adaptation

**arXiv ID:** 2606.22351 | [PDF](https://arxiv.org/pdf/2606.22351v1)

**作者:** Adam Koziak `[一作]` (Carleton University), Yuhong Guo `[通讯]` (Carleton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SAFER——一种基于可靠性引导的随机增强集成包装器，用于在敌对污染的在线测试流中实现稳健的测试时适应。

**💡 创新点**

创新点在于通过跨视图特征一致性计算可靠性权重并剔除最不可靠的视图，实现对单一视图脆弱性的补偿，同时保持对已有 TTA 方法的原始更新目标不变；还提供可选的自适应混合模块以平衡鲁棒性与干净数据准确性。

**🔧 技术方法**

使用了随机增强（模糊、低通滤波、噪声等）生成多视图；利用特征标准化后计算视图间的相关性来估计可靠性；采用加权池化和极值剔除进行预测聚合；可选的特征不一致度映射至混合系数实现自适应权重。

**📊 数据集**

在三大领域泛化基准上评估：PACS、VLCS 和 OfficeHome。

**📈 对比分析**

与源模型、标准 TTA 基线（Tent、PL、TSD、TeSLA 等）以及鲁棒性/稳定性基线（Robust ERM、EATA、Tent+MedBN）进行对比；在 ℓ∞ PGD 攻击（0%/100%）下，SAFER 在大多数场景下显著提升攻击下的准确率，同时干净准确率仅下降 1–2 个百分点；在 OfficeHome 部分域仍存在轻微的准确率下降，但整体鲁棒性优于所有比较方法。

**⚠️ 局限性**

局限性包括：仅针对黑盒转移攻击；对自适应白盒攻击未做评估；对计算开销的增量（线性于视图数）以及在某些数据集/域上可能导致干净准确率略有下降；并且在强大攻击（ε>8/255）下性能衰减仍需进一步改进。

---

## 233. Select-to-Act: Hierarchical Reinforcement Learning via Adaptive Language Guidance

**arXiv ID:** 2606.22350 | [PDF](https://arxiv.org/pdf/2606.22350v1)

**作者:** Hanping Zhang `[一作]` (Carleton University), Yuhong Guo `[通讯]` (Carleton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个层次化的强化学习框架 HRLLI，能够在不同状态下动态选择最相关的自然语言指令来指导低层动作决策。

**💡 创新点**

创新点在于将指令视为可变的高层语义指导，并设计了 Select‑to‑Act 的双层策略（高层指令选择器 + 低层动作执行器）以及阶段性奖励与辅助相似度奖励，解决了指令阶段依赖性与稀疏奖励问题。

**🔧 技术方法**

使用了 MiniLM + MLP 适配器对指令编码，GRU + 2D 卷积对状态编码；高层采用可学习的相似度打分 + REINFORCE；低层采用 PPO；引入了基于转移的奖励模型和辅助余弦相似度奖励；整体共享状态编码器。

**📊 数据集**

在 RTFM 指令密集型文本基准（包含单怪、双怪、自然语言、混洗等多种设置）进行实验。

**📈 对比分析**

与 txt2π、Reader（MCTS+模型预测）以及 Transformer 基线对比，HRLLI 在所有设置下都取得了最高或相近的胜率，尤其在双怪+自然语言混洗场景中显著优于现有方法；Oracle MCTS 仍是上限。

**⚠️ 局限性**

局限包括：需先手动拆分指令为句子；高层选择窗口 T 的设定需要调参；奖励模型对低层训练高度依赖，若环境奖励极端稀疏可能仍难以收敛；在完全可观测或指令完全相关的任务中层次结构优势不明显。

---

## 234. Customizing Video Portraits via Identity-ActionDecoupling

**arXiv ID:** 2606.22347 | [PDF](https://arxiv.org/pdf/2606.22347v1)

**作者:** Junxiong Lin `[一作]` (Fudan University), Wenqiang Zhang `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种身份保留的文本驱动视频生成框架 IaD，能在保持身份一致性的同时生成可控、丰富的面部动态视频。

**💡 创新点**

核心创新是身份‑动作解耦策略，并设计了 Identity Decoupling Loss 与 Text Alignment Loss 两个损失函数，实现身份特征与动作特征的正交分离与文本对齐。

**🔧 技术方法**

采用 MM‑DiT 视频扩散模型，结合 ArcFace 与 CLIP 双编码器、Q‑Former 对齐模块以及多阶段训练策略，保持特征稳定性并提升生成质量。

**📊 数据集**

训练使用约 47,868 条参考图像与文本对的数据集（来源未细化），基于 CogVideoX‑5B 预训练模型进行微调。

**📈 对比分析**

与现有开源方法 ID‑Animator、ConsisID 对比，IaD 在 FaceSim‑Arc/Cur、CLIPScore 提升明显，FID 降低 11–20%，表明身份一致性、语义对齐与视觉质量均有显著提升。

**⚠️ 局限性**

局限性包括对极端姿态、长时序视频和多人物场景的处理不足，仍需进一步提升时序一致性与表达多样性。

---

## 235. Efficient Multimodal Clinical Question Answering for Pulmonary Embolism Risk Assessment

**arXiv ID:** 2606.22442 | [PDF](https://arxiv.org/pdf/2606.22442v1)

**作者:** Xiangyuan Xue `[一作]` (University of Auckland), Hong Jia `[通讯]` (University of Auckland)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

将肺栓塞诊断与预后八项二分类任务改写为结构化问答，使用小型多模态大语言模型（Gemma、Qwen3.5、MedGemma）在 INSPECT 数据集上进行零样本和四样本提示的评估。

**💡 创新点**

创新点在于：①以问答形式统一诊断与预后任务；②在资源受限环境下探索紧凑多模态模型的可行性；③比较多模态输入（CTPA、EHR、两者结合）对模型性能的影响。

**🔧 技术方法**

使用多模态大语言模型（Gemma、Qwen3.5、MedGemma），结合 0-shot 与 4-shot 提示，利用 2D montage 方式处理 CT 影像和结构化 EHR 序列化文本。

**📊 数据集**

使用 INSPECT 数据集（23,248 份 CTPA 与 225.44M 条 EHR 记录）构建的 8 项二分类问答任务。

**📈 对比分析**

通过宏平均 AUROC、AUPRC、F1 与准确率评估。Gemma4 E4B 在 0-shot 结合 CT+EHR 的情形下获得最高 AUROC（≈0.689）和 F1（≈0.329），四样本提示后进一步提升；Qwen3.5 和 MedGemma 则基本停留在多数类预测，表现较差。

**⚠️ 局限性**

局限性包括：①数据集来源单一、可能缺乏外部泛化；②标签主要基于报告或结构化算法，存在噪声；③CT 影像被压缩为 2D montage，限制了三维信息利用；④模型易出现多数类崩溃，提示依赖有限；⑤未评估解释质量、校准与部署效率。

---

## 236. FlowDec: Temporal Conditional Flow Decorruptor for Robust Continuous Vision-Language Navigation

**arXiv ID:** 2606.22424 | [PDF](https://arxiv.org/pdf/2606.22424v1)

**作者:** Yufei Zhang `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Changhao Chen `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为FlowDec的时序条件流式去噪框架，用于在连续环境中的视觉语言导航中抵御未知视觉污染；

**💡 创新点**

创新点在于融合了混合时序条件策略和基于动作质心的动态过滤，实现了对历史帧的利用与单帧去噪的平衡，同时在流匹配中采用快速起点插值以提升实时性能；

**🔧 技术方法**

采用了条件流匹配（Conditional Flow Matching）和变分自编码器（VAE）进行潜在空间生成，结合PIXMix与SimSiam的数据增强，使用Mahalanobis距离进行动作质心引导；

**📊 数据集**

主要使用R2R-CE和RxR-CE两大VLN-CE数据集，采集并合成12种常见污染类型进行评测；

**📈 对比分析**

与SCUNet、Dec-DPM、Dec-CM等基线相比，FlowDec在六种最具破坏性的污染下分别提升SR约25%和9%，并在推理时间上比Diffusion基线快3-8倍，且在真实机器人实验中也取得更高成功率；

**⚠️ 局限性**

局限在于对极低亮度或极度结构化噪声的适应性尚未充分验证，且在完全干净场景下略有轻微性能下降，未来需进一步扩展至更动态噪声模式。

---

## 237. QeHDC: Hyperdimensional Computing based on Quantum-enhanced binding and SuperClass Construction

**arXiv ID:** 2606.22421 | [PDF](https://arxiv.org/pdf/2606.22421v1)

**作者:** Yangjie Xu `[一作]` (University of Luxembourg), Radu State `[通讯]` (University of Luxembourg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种 Quantum-enhanced HyperDimensional Computing (QeHDC) 框架，实现了经典-量子双步混合编码、基于参考态的压缩绑定、以及密度矩阵分解生成超级类模板，用于分类任务。

**💡 创新点**

创新点包括：① 交叉乘法（Cross‑Multiplicative）编码方案，将经典特征高效映射到量子幅度态；② 参考态压缩绑定电路，利用 CRX、RY 与 CX 门实现绑定与压缩，避免传统绑定导致的指数量子比特消耗；③ 密度矩阵最大特征值抽取生成超级类模板，提升类表征的鲁棒性；④ 在低维量子空间（4~6 个量子比特，对应 16~64 维）下实现与经典 HDC 相比的竞争或优越性能。

**🔧 技术方法**

技术包括：随机投影 + 正弦/余弦变换的经典编码；量子电路实现的 CRX、RY、CX 绑定操作；密度矩阵构造与特征值分解；量子态相似度（Fidelity）分类；在 Qiskit Aer 仿真器与 IBM QPU 上的实验，包括量子态断层（Tomography）与误差缓解。

**📊 数据集**

使用 ISOLET、MNIST、UCI HAR 三个公开数据集，分别包含 617、784、561 维特征，类别数分别为 26、10、6，实验覆盖二分类至多分类（2~10 类）场景。

**📈 对比分析**

通过与传统 HDC 基线（VanillaHD、AdaptHD、OnlineHD、NeuralHD、CompHD、SparseHD、QuantHD、LeHDC）在 16、32、64 维（相当于 4、5、6 个量子比特）下进行对比，QeHDC 在二分类任务中达 98.97%–99.97% 准确率，明显优于大多数基线；在多分类（最多 10 类）时仍保持 90% 以上准确率；在 IBM QPU 低噪声实验中，Fidelity 与准确率仅略低于理想仿真，证明可行性。

**⚠️ 局限性**

局限性包括：① 对量子硬件的噪声和有限量子比特数量敏感，尤其在大规模数据或多分类任务时需更高量子资源；② 量子态断层在多量子比特时开销指数增长，限制了大规模实验；③ 当前实现未使用可学习的量子编码或可微模板生成，可能限制进一步性能提升；④ 仅针对离散分类任务，尚未扩展至序列建模、在线学习等更复杂场景。

---

## 238. Towards Error-Free Long Video Generation

**arXiv ID:** 2606.22370 | [PDF](https://arxiv.org/pdf/2606.22370v1)

**作者:** Shuning Chang `[一作]` (Zhejiang University), Yi Yang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种无限长视频生成框架，通过将扩散模型微调为视频扩展模型，并在片段级别实现因果注意力与KV缓存，配合截断校正流(T-RFlow)以抑制误差累积和属性漂移，最终实现高质量、动态且身份一致的长视频生成。

**💡 创新点**

创新点包括：①两阶段训练策略（先在大规模短视频上微调为扩展模型，再在长视频上加入跨片段因果注意力）；②使用KV缓存并按提示相似度筛选历史KV以降低显存；③提出截断校正流(T-RFlow)分频率剔除高频与低频误差，从而在不额外训练的情况下显著减少误差累积；④在片段内保持双向注意力、片段间使用单向注意力的混合注意力结构。

**🔧 技术方法**

主要技术手段包括：潜在扩散模型（LDM）与Rectified Flow、因果注意力机制、KV缓存与RoPE扩展、LoRA参数高效微调、句子嵌入（Sentence‑BERT/LLM）进行KV选择、截断校正流(T‑RFlow)。

**📊 数据集**

使用了Wan2.1‑T2V‑1.3B预训练模型，并在自建的XunGuang‑1.1M短视频数据集上微调为扩展模型；随后在同一数据集的长视频以及约10k条私有数据上进一步细调；此外通过XunGuang‑1.1M长视频进行因果注意力训练。

**📈 对比分析**

在VBench单射长视频生成任务上与多种现有方法（MAGI‑1、Self‑Forcing、PAVDM、FramePack、SkyReels‑V2、LCT、MoC等）进行对比，实验结果显示在主体一致性、背景一致性、运动平滑度、动态程度、审美质量与图像质量六项指标上均优于或匹配最先进基线，尤其在主体一致性（0.9457）与背景一致性（0.9691）表现突出。

**⚠️ 局限性**

局限性在于缺乏实时生成支持，当前方法不支持流式长视频生成，未来需探索实时流式生成策略。

---

## 239. Trajectory Forcing: Structure-First Generation with Controllable Semantic Trajectories

**arXiv ID:** 2606.22527 | [PDF](https://arxiv.org/pdf/2606.22527v1)

**作者:** Merve Kocabas `[一作]` (University of Tübingen), Andreas Geiger `[通讯]` (University of Tübingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Trajectory Forcing (TF)，通过在预训练的DINOv2特征空间中构建语义层级，按粗到细的顺序逐级生成可解码、可编辑的中间状态，实现生成过程的可视化与交互；

**💡 创新点**

创新点在于将生成轨迹从不可见的计算过程提升为可供用户直接交互的结构化过程，使用层级化的一步流匹配模型在每个语义层级生成，构建了可解释的语义层级并引入了轨迹感知评价指标；

**🔧 技术方法**

主要技术包括：预训练视觉表征空间（DINOv2）与Representation Autoencoder；聚类构建教师层级；层级条件化的Transformer（DiT）和一阶流匹配（Mean Flow）实现单步生成；共享解码器实现所有层级的可视化；以及基于局部不变性和父子一致性的轨迹评价指标；

**📊 数据集**

使用ImageNet数据集（class‑conditional），在256×256分辨率上进行实验；

**📈 对比分析**

与多步扩散、流模型、NVG等基线相比，TF在80个训练周期内就能快速收敛，FID与IS表现相近，且能提供可解释的中间状态和编辑功能；通过自定义的LIS和PMR指标验证了轨迹层级的局部可控性和结构一致性；

**⚠️ 局限性**

局限性包括：层级聚类固定深度且依赖中心先验，可能不适用于无明显中心对象的图像；每个层级仅受前一层影响，缺乏更直接的跨层条件；缺乏针对不同层级的专门监督；实验仅到40/80个周期，未充分验证更长训练和更大模型的效果。

---

## 240. Projection-Volume Fidelity Divergence: Diagnosing and Controlling Optimization Drift in Sparse-View 3D Gaussian Tomography

**arXiv ID:** 2606.22525 | [PDF](https://arxiv.org/pdf/2606.22525v1)

**作者:** Yikuang Yuluo `[一作]` (Chongqing University), Fuquan Wang `[通讯]` (Chongqing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

在稀疏视角计算机断层扫描（CT）中，研究并解决了投影-体积一致性偏离（PVFD）问题，提出了一种基于3D高斯分布的稀疏视角重建控制框架LADES；

**💡 创新点**

创新点在于：①定义并量化PVFD及其对高斯原语的结构退化影响；②设计线性退火丢弃（Linearly Annealed Dropout）与结构感知早停（Structure-Aware Early Stopping）相结合的训练策略，实现无需真值监督的PVFD抑制；

**🔧 技术方法**

使用的技术包括3D Gaussian Splatting、投影一致性损失、SSIM正则、线性退火丢弃、Gaussian人口增长监测、动态密度分辨率更新等；

**📊 数据集**

数据集为芬兰逆向问题学会（FIPS）公开的真实CT扫描数据，涵盖榛子、松果和海贝三种物体；

**📈 对比分析**

与传统FDK、SART以及TAG‑Gaussian、GR‑Gaussian、R²‑Gaussian等基线方法对比，LADES在稀疏视角（10/20/25视角）下平均3D PSNR提升约1.0‑1.5 dB，SSIM提高约0.01‑0.02，且训练时间大幅缩短（约4倍）且无后期过拟合；

**⚠️ 局限性**

局限性在于仅在FIPS稀疏视角CT数据上验证，未涉及不同扫描仪几何、噪声水平或临床协议的泛化；缺乏临床部署验证。

---

## 241. Mitigating Polycentric Conflict-Trap Risk in Mali via Intergenerational Volterra Mean-Field-Type Games

**arXiv ID:** 2606.22512 | [PDF](https://arxiv.org/pdf/2606.22512v1)

**作者:** Hamidou Tembine `[一作]` `[通讯]` (University of Quebec in Trois-Rivieres), Hamidou Tembine (University of Quebec in Trois-Rivieres)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文构建了一个以交互式、多中心、跨代自我复制冲突系统为核心的多世代Volterra均衡型游戏框架，专门用于解释和预测马里及其周边地区持续的冲突与经济掠夺。通过将历史记忆、复仇激励、机构反应和环境冲击统一纳入Wasserstein空间的联合分布，作者阐释了战争企业如何通过跨境金融网络循环非法收益，并展示了传统二元内战模型无法捕捉的分散化、非协调化与跨代传递机制。

**💡 创新点**

创新点包括：①提出“跨代Volterra均衡型游戏”（Intergenerational Volterra Mean‑Field‑Type Game, MFTG）框架，首次将长期记忆与复仇算子嵌入到游戏动力学；②在Wasserstein流形上构造联合分布的Gâteaux导数与期望极值（expectile）风险度量，实现对非对称风险的刻画；③综合使用分形布朗运动、Gauss‑Volterra过程、Rosenblatt过程与泊松跳跃，完整刻画了冲突系统的短期波动、长期记忆和重尾冲击；④设计了“金融‑心理反馈断裂”最优政策，证明通过税收与调解补贴可将系统谱半径压回1以下，实现从战壕陷阱到和平稳态的可行转移。

**🔧 技术方法**

技术方法：Mean‑Field‑Type Game（MFTG）框架；Wasserstein 2‑空间与期望极值（α、β‑expectiles）；Volterra积分算子与复仇记忆；Gâteaux 变分分析；分形布朗运动、Gauss‑Volterra 与 Rosenblatt 过程；泊松随机测度与马尔可夫跃迁；控制理论中的前向‑后向（forward‑backward）偏微分方程；谱半径分析与可持续性阈值判定；最优控制与社会政策设计。

**📊 数据集**

数据来源主要为实地调研与公开统计：马里冲突地区的武装事件分布、金矿收益、跨境渗透指数、降雨与粮食安全指数、国际金价、外部军事与人道援助量化数据。由于模型高度理论化，未使用标准机器学习数据集，而是基于上述宏观与微观时空序列进行参数校准。

**📈 对比分析**

对比方法：将传统的Mean‑Field Game（MFG）与本框架的MFTG在模拟马里冲突演化时进行对比。结果表明，MFG在长期记忆、跨代传递与非对称风险处理方面均显失效，导致冲突动态与现实观测高度偏离；而MFTG通过完整联合分布与期望极值显著提升了对冲突激烈度与波动性的预测精度。性能方面，理论证明存在唯一纳什均衡，并给出了系统谱半径小于1时和平稳态可持续的充分必要条件，但缺乏大规模数值仿真与实证验证。

**⚠️ 局限性**

局限性：①模型参数繁多，需在真实冲突情境下进行昂贵的校准；②对完全信息与理性行为假设过强，忽略了信息不对称与非理性决策；③对联合分布的高维求解和前向‑后向系统在实际应用中计算复杂度高；④缺乏针对不同政策情景的定量仿真与实证检验；⑤对战争企业与跨境金融网络的细节假设可能与实际操作存在偏差。

---

## 242. Breaking the Likelihood Trap: Variance-Calibrated Modulation for Large Language Model Decoding

**arXiv ID:** 2606.22511 | [PDF](https://arxiv.org/pdf/2606.22511v1)

**作者:** Yuanhao Ding `[一作]` (Henan University), Chongsheng Zhang `[通讯]` (Henan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出训练无关的前解码干预方法 Variance‑Calibrated Modulation (VCM)，通过 PMI 上下文搜索光和基于 logit 方差的自适应去偏惩罚，重塑模型的 logit 分布，降低生成的重复与呆板。

**💡 创新点**

创新点在于将 PMI 上下文先验与尺度不变的自适应惩罚融合为一体化的动态预处理策略，既消除头部过度集中，又避免固定惩罚导致语义失衡，同时不需要额外前向传播。

**🔧 技术方法**

使用的技术包括：PMI 估计（利用预计算的无条件 logit 作为先验），自适应标准差惩罚（按当前 logits 标准差缩放重复惩罚），logit 加权调节，随后采用传统截断采样（Top‑k/Top‑p 等）。

**📊 数据集**

实验数据集涵盖开放式文本生成（Wikitext‑2/3）、事实问答（TruthfulQA、TriviaQA）以及数学推理（GSM8K、MATH500）。

**📈 对比分析**

与多种采样器（Top‑k、Top‑p、η/ε 采样、Typical、Min‑p、Top‑nσ 等）以及 Contrastive Decoding 与 Dynamic Focus Decoding 对比；VCM 在开放生成任务的 MAUVE、BERTScore、Distinct‑2、TTR 等指标显著提升，问答与推理任务保持或略优，同时推理速度几乎不增加（≈1.01× 延迟）。

**⚠️ 局限性**

局限性包括：对任务依赖的混合系数 α 需要手动调节；实验仅在中等规模开源模型上验证；未将 VCM 与重排序、额外训练或检索检证等更复杂的提升方法进行完整对比；以及生成多样性提升不必然带来更高的事实性。

---

## 243. Line Drawings using LightBenders: Authoring and Illuminating

**arXiv ID:** 2606.22499 | [PDF](https://arxiv.org/pdf/2606.22499v1)

**作者:** Hamed Alimohammadzadeh `[一作]` (University of Southern California), Shahram Ghandeharizadeh `[通讯]` (University of Southern California)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于可驱动LED杆的光束无人机LightBender及其硬件和软件体系结构，并实现了Blender插件和LB-Author工具，支持从SVG线条到三维LED灯光的全流程生成与执行。

**💡 创新点**

在硬件层面首次将双关节LED杆集成到小型无人机；在软件层面提出了两种线条铺排（集合覆盖与顶点优先贪婪）和空间错位（Stagger）算法，解决下洗涂和重叠冲突；在实验层面用人类感知评估误差阈值，首次量化视觉可接受度。

**🔧 技术方法**

使用的技术包括：Crazyflie Bolt飞控与Raspberry Pi CM5搭载的微控制器；两段伺服驱动的LED杆；Python/Blender API实现的动画与LED表达式；Vicon运动捕捉同步控制；整数规划/分支定界、贪婪启发式与最小顶点覆盖求解冲突。

**📊 数据集**

主要数据集为手工绘制的SVG线条（字母A-Z、Emoji、箭头等）以及一幅89点/84边的天际线矢量图，用于测试铺排算法和误差评估。

**📈 对比分析**

对比评估：SC与VFG铺排在节点数、LED利用率和运行时间上互相折衷；在真实飞行中测得绝对RMSE约9–16 mm，误差不随无人机数量增大或中途变换灯光而显著升高；人类实验显示10.1 mm误差仍可接受，30 mm误差可识别但仍可用，100 mm则不可用。

**⚠️ 局限性**

局限性包括：铺排算法对大规模图形计算复杂度高；LED杆长度受限导致对细线段的逼近不足；漂移误差仍随机且难以完全补偿；仅验证室内平面线条显示，未考虑三维体积或多视角同步。

---

## 244. Deep Learning-Based Sign Language Recognition from Videos and Cross-Lingual Translation to Indian Vernaculars

**arXiv ID:** 2606.22494 | [PDF](https://arxiv.org/pdf/2606.22494v1)

**作者:** Chandranath Adak `[一作]` (Indian Institute of Technology Patna), Ramesh Nandipalli `[通讯]` (Indian Institute of Technology Patna)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一个两阶段深度学习流水线，将印度手语视频先分类为英文标签，再使用 NLLB-200 将该英文标签翻译为印地语、泰卢固语和孟加拉语。

**💡 创新点**

创新点在于结合 VideoMAE 视频变压器与 NLLB 多语言翻译模型，突破缺乏手语-区域语言并行语料的限制，并提供基于英语枢轴的多语种翻译方案。

**🔧 技术方法**

采用 VideoMAE 视频变压器进行视频到英文标签的分类，随后调用 Meta AI 的 NLLB-200 进行英文到多种印度语言的神经机器翻译，整合 Streamlit 实现交互式演示。

**📊 数据集**

使用了 IIT Madras AI4Bharat 印度手语视频语料库的 13 类子集（包括八个形容词和五个服装名词），共 197 条 16 帧视频。

**📈 对比分析**

在 80/20 训练/验证拆分下，Fine‑tuned VideoMAE 的训练精度达 99%，验证精度为 78%；翻译阶段在单词层面无显著误差，但整体性能受限于小样本和词级翻译。

**⚠️ 局限性**

局限性包括标签集小且不平衡、仅识别离散词汇而非连续句子、缺乏多签名者和多录制条件的鲁棒性、单词级翻译导致歧义、以及离线非实时推理。

---

## 245. Interleaved Speech Language Models Latently Work In Text

**arXiv ID:** 2606.22473 | [PDF](https://arxiv.org/pdf/2606.22473v1)

**作者:** Talia Sternberg `[一作]` (Hebrew University of Jerusalem), Yossi Adi `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究混合语音-文本语言模型的中间层表示，发现它们在不受语音识别监督的情况下会隐式转写语音并在文本空间中做出预测。

**💡 创新点**

首次将 Logit lens 与交错语音-文本训练结合，揭示了隐式转写的出现机制，并与模型的事实知识检索能力建立关联。

**🔧 技术方法**

使用 Logit lens 对 Transformer 中间层进行词汇投影，采用离散语音单元（如 HuBERT）进行语音编码，交错训练（speech‑text interleaving），以及 Recall@k 等评估指标。

**📊 数据集**

实验基于 SIMS 官方训练集、人工合成的 282 条常识事实补全数据（通过 TTS 合成语音并用 Whisper 对齐），以及公开的交错语音‑文本数据。

**📈 对比分析**

通过比较 speech‑only、随机初始化、文本预训练、不同交错比例等模型配置，使用 Recall@k 评估隐式转写和下一词可解码率；结果显示文本预训练 + 交错训练最强，且与事实知识准确率呈正相关，表现优于纯语音模型但仍未完全解释知识能力。

**⚠️ 局限性**

未定位具体的头层/路径机制，相关性仅为部分解释，缺乏因果验证，也未评估对声学表现的潜在负面影响，且过度依赖文本词汇可能导致语音能力受限。

---

## 246. Adaptive Recurrent Message Passing for Test Time Computing on Graphs

**arXiv ID:** 2606.22462 | [PDF](https://arxiv.org/pdf/2606.22462v1)

**作者:** Junshu Sun `[一作]` (Chinese Academy of Sciences), Shuhui Wang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种自适应递归图模型AdaR，实现在不改变模型参数的前提下，依据测试时不同的可收敛迭代步数自适应地调整图的感受野；

**💡 创新点**

关键创新在于：① 引入步信息依赖和表示-目标关系的递归更新；② 通过正弦位置编码实现对迭代步数的连续归一化；③ 采用梯度监督信号引导每一步递归更新，从而实现统一收敛；

**🔧 技术方法**

技术核心包括递归图神经网络、位置编码、伪节点作为目标代理、梯度监督、内部关系计算及自适应迭代退出策略；

**📊 数据集**

使用多种公开图数据集进行零射（zero-shot）迁移实验（arXiv、BookHistory、Amazon‑Ratings、PubMed、WikiCS、SportsFit、Cora、CiteSeer、DBLP等）以及传统的heterophilic与homophilic图；

**📈 对比分析**

与传统GNN、N²、图变换器、自监督方法、LLM基础模型等对比，AdaR在零射迁移和监督迁移任务上平均提升15–22%的性能，训练成本低、推理灵活；

**⚠️ 局限性**

局限性包括：① 需要额外的伪节点与位置编码，模型规模略增；② 对极大规模图的内存与时间仍有挑战；③ 依赖梯度监督，若目标任务梯度信息不佳可能影响收敛；

---

## 247. Quantum Codes with Transversal $CCZ$ Gates and Sublinear $Z$-Stabilizers

**arXiv ID:** 2606.22472 | [PDF](https://arxiv.org/pdf/2606.22472v1)

**作者:** Ohad Elishco `[一作]` (Ben Gurion University Of Negev), Itzhak Tamo `[通讯]` (Tel Aviv University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构造了支持可跨越 CCZ 门的 CSS 码，且其 Z‑稳定子生成集合的权重为子线性。

**💡 创新点**

提出了改进的截断定理，用低率代数膨胀码的插值集作为截断集合，克服了传统方法因对偶距离小而导致逻辑量子数子线性的限制；同时通过乘法友好码实现了固定素域的字母压缩，保持了子线性权重特性。

**🔧 技术方法**

利用代数膨胀码的 Tanner 结构、Schur 乘积性质、插值集构造、乘法友好（multiplication‑friendly）与投影多重性 (projective‑multiplicity) 码、CSS 代码拼接与限制、以及对偶距离在截断集外的下界证明。

**📊 数据集**

无实测数据集，所有结果均为理论构造与证明。

**📈 对比分析**

相较于已有的 qLDPC / qLTC 方案，本文得到线性码率、幂律距离、可跨越 CCZ 的性质，并在 Z‑稳定子权重上实现了 O(N^{1/m}) 的子线性上界；然而 X‑稳定子仍为线性权重；在实验或实现方面未给出具体数值。

**⚠️ 局限性**

局限性：X‑稳定子生成集合权重高；缺乏高效纠错算法；构造对比仅在 Z‑侧低密度，尚未实现双侧低密度。

---

## 248. Cross-Layer Intrusion Detection in 5G O-RAN: Gains and Limits of Fusing Radio Telemetry with Network Flow Records

**arXiv ID:** 2606.22450 | [PDF](https://arxiv.org/pdf/2606.22450v1)

**作者:** Hamed Fard `[一作]` (Freie Universität Berlin), Gerhard Wunder `[通讯]` (Freie Universität Berlin)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对5G O-RAN多层IDS进行实验，评估DU无线遥测与CU网络流记录及其融合；

**💡 创新点**

首次系统比较并量化单模态与双模态在不同模型上的表现，揭示融合对低FPR检测率的潜在负面影响；

**🔧 技术方法**

使用7种经典与深度模型（LogReg、XGBoost、RF、ResMLP、GRU、TCN、Transformer）、窗口化特征提取、Score‑level（平均与堆叠）融合；

**📊 数据集**

基于NetsLab-5GORAN-IDD公共数据集，包含42组配对CU流记录与DU遥测；

**📈 对比分析**

通过ROC‑AUC、1% FPR下的检测率和F1‑macro比较，发现无线遥测在二分类任务上优于网络流，融合可略提升ROC‑AUC但在低FPR下往往降低检测率；多类任务中融合均提升F1‑macro，Web类收益最大；

**⚠️ 局限性**

DoS‑正常误分类率始终高达27–46%，窗口化统计和模型容量提升均未能显著解决，表明该问题源自特征表示与时间聚合方式。

---

## 249. The Scissors Effect: When Resize-Based Input Diversity Helps or Hurts Transfer Attacks

**arXiv ID:** 2606.22516 | [PDF](https://arxiv.org/pdf/2606.22516v1)

**作者:** Yuhang Jiang `[一作]` (University of Trento), Xiaojing Chen `[通讯]` (Anhui University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了输入多样性 (DI) 对迁移式对抗攻击的影响，并发现了“Scissors Effect”，即 DI 对标准模型有益，而对鲁棒模型则有害。

**💡 创新点**

首次揭示 DI 效果随模型鲁棒性变化而切换，并提出通过梯度一致性测度 LGC 判断是否开启 DI，提供了理论阈值和训练无关的 CG-DI 规则。

**🔧 技术方法**

梯度一致性分析 (LGC)、频谱分析、重塑/平移分解、偏差-方差理论、理论证明 (阈值交叉) 以及多种攻击方法 (MI-FGSM, Admix, SSA 等)。

**📊 数据集**

ImageNet 与 CIFAR-10 两个公开数据集，涵盖多种标准与鲁棒模型。

**📈 对比分析**

将 DI 与无 DI 的 MI-FGSM 等攻击进行对比，以攻击成功率 (ASR) 为指标；在 ImageNet 上对鲁棒源模型平均损失 10.3% ASR，标准源模型提升约 15-20%；在 CIFAR-10 效果较小，但在更激进的 DI 设置下可达 6.6% 下降。

**⚠️ 局限性**

效果在 CIFAR-10 上受分辨率限制，SSA 等频域变换与 DI 交互导致异常；LGC 在容量低时可能误判；仅评估单一稳健训练范式、未考虑集成源；CG-DI 为二元规则，无法捕捉连续最优。

---

## 250. Lingering Authority: Revocable Resource-and-Effect Capabilities for Coding Agents

**arXiv ID:** 2606.22504 | [PDF](https://arxiv.org/pdf/2606.22504v1)

**作者:** Igor Santos-Grueiro `[一作]` `[通讯]` (International University of La Rioja), Igor Santos-Grueiro (International University of La Rioja)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并实现了编码代理的资源与效果权限生命周期管理，构建了基于任务合同的请求–授予–执行生命周期并通过闭包机制撤销临时权限；

**💡 创新点**

将可撤销的临时权限嵌入任务合同，提出“消逝权限”概念，显式控制权限的可见性与可用性，并通过参考监视器实现权限句柄的生命周期与审计；

**🔧 技术方法**

采用参考监视器、typed tool catalog、权限句柄（epoch‑bound）机制、任务合同编译器以及安全授权检查；实现了 grant/closure 逻辑，在 Python 环境下通过中介层控制文件、shell、git、网络等工具调用；

**📊 数据集**

四组实验数据集：三组受控安全 fixture（A–C）以及一组真实仓库 snapshot（D），共 17+37+14+14 任务实例，使用 Qwen3‑Coder 30B 及其他大模型进行评估；

**📈 对比分析**

通过曝光、扩展、闭包三种权限策略与基线（全部访问、静态 allowlist、沙箱）以及无撤销对照组对比，测量违规率、任务成功、范围合规、后闭包再利用等指标；结果显示全权访问导致违规；全权限+闭包在安全上与静态 allowlist相当，但利用率略低；非撤销对照仍允许后闭包重用，闭包显著减少 stale‑capability；

**⚠️ 局限性**

实验仅覆盖受控与真实 Python 仓库场景，未验证数据库、浏览器、SaaS 等其他工具域；假设中介线性化；缺乏对模型非确定性、可重授权路径以及合同编写成本和质量的深入评估；

---

## 251. SCOPE: Evolving Symbolic World for Planning in Open-Ended Environments

**arXiv ID:** 2606.22488 | [PDF](https://arxiv.org/pdf/2606.22488v1)

**作者:** Yundaichuan Zhan `[一作]` (Zhejiang University), Yueting Zhuang `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SCOPE框架，实现对开放式环境中符号世界的自适应演化与计划改进，构建闭环的感知、验证与执行反馈机制；

**💡 创新点**

创新点在于①引入Symbolic Execution Simulator（SESim）对符号计划进行验证并结合真实执行反馈持续演化符号世界；②设计Self‑Adaptive Symbolic Memory（SASMem）对反馈进行谓词级抽象，生成可迁移的符号知识；③将VLM与经典规划器深度耦合，形成持续改进的闭环；

**🔧 技术方法**

核心技术包括：Vision‑Language Model（VLM）用于感知与符号化，PDDL进行符号规划，经典PDDL验证器，SESim进行符号与真实执行双重反馈，SASMem实现结构化记忆抽象；

**📊 数据集**

实验使用两大仿真平台——VirtualHome 与 ALFRED，构造基本/复杂场景以及静态/动态/开放式三种环境设置；

**📈 对比分析**

与VILA、ISR‑LLM（基于VLM的规划器）和NESYC（经典规划器）对比，SCOPE在所有设置下均获得最高SR/GC、ClassicalSR与SymRecall，开放式任务中StepSR提升约11.3%，整体性能领先15–20个百分点；

**⚠️ 局限性**

局限性包括：仍存在一定程度的符号幻觉；对VLM的依赖导致模型规模与算力要求较高；SASMem与SESim的迭代开销在大规模任务中未做充分评估；

---

## 252. Governed AI-Assisted Engineering: Graduated Human Oversight for Agentic Code Generation in Regulated Domains

**arXiv ID:** 2606.22484 | [PDF](https://arxiv.org/pdf/2606.22484v1)

**作者:** Richard Kang `[一作]` `[通讯]` (DoiT International), Richard Kang (DoiT International)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出 GAIE 框架，通过三层人类监督（HITL、HOTL、AWM）治理代理式代码生成，以实现监管合规与开发效率的平衡。

**💡 创新点**

创新点包括：基于监管影响、客户接触、可逆性和数据敏感度的 Oversight Classification Model（OCM）；每层自动生成的合规证据链；以及通过形式化证明保证单调性、fail‑safe 和完备性。

**🔧 技术方法**

使用多智能体监督架构、规则引擎、加密链、生成/测试分离、监控系统以及 append‑only 存储等技术实现治理与证据链。

**📊 数据集**

论文为分析性研究，未使用真实生产数据；采用假设的任务量分布和 BOT 监管需求作为输入。

**📈 对比分析**

通过监管覆盖分析、比较框架分析和解析性生产力模型评估；相较于统一 HITL，GAIE 在低风险任务下保持 84–97% 速度（中心值约 91%），而统一 HITL 仅 45–65%。

**⚠️ 局限性**

局限性包括：主要映射单一司法区、缺乏实测数据、分类边界不确定、专家验证尚未完成、技术耦合问题，以及对不确定元数据的误分类风险；监管映射未获得官方验证。

---

## 253. ARP: Enhancing Quantized Skill Abstractions via Visual Alignment and Iterative Refinement for Robotic Manipulation

**arXiv ID:** 2606.22480 | [PDF](https://arxiv.org/pdf/2606.22480v1)

**作者:** Yuntian Wang `[一作]` (Soochow University), Jin Wang `[通讯]` (Soochow University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了双阶段离散技能框架ARP，通过视觉-动作对齐和迭代残差修正提升机器人长时序操作性能。

**💡 创新点**

创新点包括：1) 用对比学习将视觉嵌入对齐到预量化动作潜在空间，解决语义模糊；2) 引入轻量级两步迭代残差头(IRH)修正量化误差，兼顾高层规划与低层执行。

**🔧 技术方法**

使用技术包括 Finite Scalar Quantization、InfoNCE 对比学习、Transformer 自回归技能先验、轻量级两步迭代残差头(IRH)以及离散技能抽象。

**📊 数据集**

使用的数据集有 LIBERO、Meta-World 以及在 Kuavo 4 Pro 实体机器人上收集的远程操作演示数据。

**📈 对比分析**

通过与多种连续动作和离散技能基线对比，ARP 在 LIBERO 上平均成功率达到 89.6%（最高），在 Meta-World 上平均 73.8%，在真实机器人任务中相比 VQ‑BeT、QueST 取得 1‑3 分的显著提升，展示了显著的性能优势。

**⚠️ 局限性**

局限性在于仍受量化误差和动态噪声影响，无法完全避免抓取失效或复杂接触恢复，需要更强的闭环恢复与毫米级控制来进一步提升鲁棒性。

---

## 254. ROMEVA: Geometry-Preserving Vocabulary Expansion for Roman Urdu Language Models

**arXiv ID:** 2606.22478 | [PDF](https://arxiv.org/pdf/2606.22478v1)

**作者:** Mahnoor Khan `[一作]` (National University of Sciences and Technology), Mehwish Fatima `[通讯]` (National University of Sciences and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了罗马乌尔都语的词汇适配问题，并提出了ROMEVA框架来在多语言BERT中进行词汇扩展，保持嵌入几何。

**💡 创新点**

结合子词平均初始化和PCA引导锚定损失来控制词嵌入漂移，并系统评估嵌入稳定性与下游任务性能的权衡。

**🔧 技术方法**

使用多语言BERT（mBERT）为基础模型，采用子词平均初始化、PCA锚定正则化、掩码语言模型继续预训练以及情感分类微调等技术。

**📊 数据集**

新构建的36,130条罗马乌尔都语YouTube评论语料（RUVA）用于词汇扩展和鲁棒性评估；RUWV-NSR情感分类数据集用于下游性能评测。

**📈 对比分析**

通过对比无约束扩展、子词初始化和ROMEVA三种策略，使用L2漂移和余弦相似度衡量嵌入稳定性；在情感分类任务中，无约束扩展获得最高准确率和宏F1，而ROMEVA在保持最低漂移的同时性能最低。

**⚠️ 局限性**

仅在罗马乌尔都语和mBERT上验证，词汇扩增仅限500个高频词，未探究更大扩展或其他多语言模型；下游仅评估情感分类，且稳定性度量仅使用L2和余弦相似度。

---

## 255. Music Playlist Captioning at Scale with Large Language Models

**arXiv ID:** 2606.22460 | [PDF](https://arxiv.org/pdf/2606.22460v1)

**作者:** Mathieu Delcluze `[一作]` (Deezer Research), Guillaume Salha-Galvan `[通讯]` (SJTU Paris Elite Institute of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Deezer上部署自动化音乐播放列表标题生成系统，通过大型语言模型为Daily Mix生成多语言简短标题

**💡 创新点**

首次将LLM用于大规模播放列表标题生成，并在生产环境中通过标签与用户生成标题融合，提升语义框架的精准度

**🔧 技术方法**

使用Gemini 2.0 Flash LLM，Spark/Scala 数据预处理，Python 推理，Prompt engineering，LLM-as-a-Judge 验证

**📊 数据集**

利用Deezer内部的5,000个艺术家聚类标签、用户自建播放列表标题以及专有音乐标签（流派、情绪、国家、年代）

**📈 对比分析**

通过对数百万用户的在线A/B测试，LLM标题提升采用率+24.9%，复连率+16.9%，满意度+11.5%，显著优于原系统

**⚠️ 局限性**

限制在非英语、法语、葡萄牙语之外的语言质量不足，且对LLM的潜在幻觉风险需持续监控，且对小众内容的标签覆盖仍有欠缺

---

## 256. Self-Evolving Cognitive Framework via Causal World Modeling for Embodied Scientific Intelligence

**arXiv ID:** 2606.22449 | [PDF](https://arxiv.org/pdf/2606.22449v1)

**作者:** Yi Yu `[一作]` (Hiroshima University), Tetsunari Inamura `[通讯]` (Tamagawa University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

提出了一种自我演化的认知框架，通过因果世界模型实现具身科学智能，强调交互既是行为优化也是因果假设生成与验证的过程。

**💡 创新点**

创新点在于：①将因果世界模型、干预驱动的因果推理与持续认知细化统一到一个自我演化的过程；②引入结构演化算子 Φ，使得结构方程与外因分布随经验动态更新；③将模拟经验视为基于结构因果模型的知识实验平台，支持对干预与逆事实的系统评估。

**🔧 技术方法**

使用的核心技术包括：结构因果模型（SCM）框架、do‑操作干预推理、逆事实推理、因果不一致性损失、自适应结构演化算子、在线梯度更新、基于交互历史的策略优化。

**📊 数据集**

文中未给出具体数据集；主要以理论推导和概念框架为主，后续工作可在机器人抓取、导航、仿真等具身任务数据集上验证。

**📈 对比分析**

由于本工作为理论与框架设计，没有实证比较；作者提出了六维因果-认知基准（干预鲁棒性、逆事实准确度、失败利用率、长周期因果一致性、自我修改稳定性、跨平台转移鲁棒性）作为未来评估标准。

**⚠️ 局限性**

主要局限包括：①如何实现长期的结构演化与机制稳定；②利用失败信息进行因果学习的有效机制尚未完善；③长时序记忆与知识巩固的实现方案缺失；④安全干预与自主实验的约束未解决；⑤高阶科学认知（类比、创造）的出现机制未知；⑥多智能体交互中的因果归因与干预分离难题。

---

## 257. Submodular Welfare Maximization with Budget Constraints in the Random-Order Model

**arXiv ID:** 2606.22520 | [PDF](https://arxiv.org/pdf/2606.22520v1)

**作者:** Max Klimm `[一作]`, Martin Knaack `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对带预算的在线物品分配，提出多项式时间的α-竞争算法，并在匹配子问题上提升竞争比。

**💡 创新点**

首次在多代理场景下取得 α≈1/14.85 的竞争比，且在预算=1、成本=1 的匹配问题上提高到 ≈1/6.86。

**🔧 技术方法**

利用多线性扩展的 (1−1/e) 近似和反复求解离线子问题的技术。

**📊 数据集**

无实验数据集，全部基于理论证明。

**📈 对比分析**

与先前的 1/54.4（单代理）和 1/9.66（匹配）算法对比，竞争比显著提升；若允许超多项式时间，可进一步提升。

**⚠️ 局限性**

依赖随机到达顺序、oracle 访问子模函数、子问题求解时间较高，未给出实验验证。

---

## 258. Biological Sex Determination in Cadavers Using Deep Learning Algorithms from Computed Tomography Images of Pelvis and Skull

**arXiv ID:** 2606.22515 | [PDF](https://arxiv.org/pdf/2606.22515v1)

**作者:** Giovanna Herculano Tormena `[一作]` (University of São Paulo), Marcelo Becker `[通讯]` (University of São Paulo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

利用深度学习方法，构建了一套基于尸体CT扫描（骨盆和颅骨）二维投影的生物性别识别系统。

**💡 创新点**

创新点在于：①将三维重建转为多角度二维投影并采用患者级别软投票聚合；②采用四分类（性别+骨骼部位）策略；③在YOLOv26 Nano架构上结合迁移学习、定制与YOLO内置增强以及权重交叉熵，显著提升性能。

**🔧 技术方法**

使用技术包括：YOLOv26（Nano/Small/Medium）、Transfer Learning、数据增强（自定义+YOLO）、权重交叉熵、Grad‑CAM可视化、遗传算法优化超参数、TOPSIS与VIKOR多准则决策、5‑折交叉验证。

**📊 数据集**

数据集来源于巴西伊马利亚（IMLAT）法医机构141例尸体的CT扫描，涵盖骨盆与颅骨三维重建，转换为11幅角度的二维投影，共计3059幅图像。

**📈 对比分析**

通过与7种主流模型（EfficientNetV2、ConvNeXt、ViT‑B16、VGG16、ResNet50、YOLO11、YOLO26）基线对比，并在二分类与四分类两方案下评估；最终患者级别表现为准确率95.65%、召回率92.86%、F1值94.36%，并在5‑折交叉验证中平均准确率90.93%。

**⚠️ 局限性**

局限性包括：样本量有限且性别不平衡；仅来自单一机构与单一CT设备，缺乏跨中心、多设备的泛化验证；以及对自定义增强的效果不一，需进一步优化。

---

## 259. Fed-CausalDiff: Decoupled Synchronization for Federated Do-Simulation and Policy Evaluation

**arXiv ID:** 2606.22510 | [PDF](https://arxiv.org/pdf/2606.22510v1)

**作者:** Pengfei Li `[一作]` (University of Bergen), Mohammad Khalil `[通讯]` (University of Bergen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出一种名为 Fed-CausalDiff 的联邦因果扩散框架，用于在分布式时序日志上进行因果推断、do‑模拟和策略评估。

**💡 创新点**

创新点在于：①通过分离全局因果分数与本地混杂分数实现解耦同步（DSS），显著降低异构环境下的干扰；②利用条件潜在扩散（score‑based conditional diffusion）构建可采样的因果模拟器；③采用梯度反转与 MMD 对齐实现潜在空间的因果/混杂解耦。

**🔧 技术方法**

核心技术包括：联邦学习（FedAvg/FedProx等）、条件潜在扩散与去噪评分匹配、结构因果模型、分阶段训练（推断→扩散→解码）、梯度反转、MMD对齐、离散化的交互策略模拟。

**📊 数据集**

使用四个数据集：DKT‑Synth（半合成，含真值反事实）、Statics2011（大学工程力学课程日志）、Diabetes‑130（10年美国医院住院记录）和 Open Bandit（ZOZOTOWN 电子商务日志）。

**📈 对比分析**

与 RCGAN、TimeGAN、CRN、FedCM 四种基线对比；在 DKT‑Synth 上，Fed‑CausalDiff 在 PEHE、ATE、策略值误差三项均名列前茅；在 Diabetes、DKT‑Synth 的离线策略评估中产生更具区分力的 DR 估计；TSTR 指标保持与生成基线相近；在 DKT‑Synth 的训练轮数和通信量上收敛更快、表现更好，但在 Open Bandit 与 Statics2011 上不总是位于 Pareto 前沿。

**⚠️ 局限性**

局限性：①真实数据缺乏反事实真值，评估依赖 DR/MSM 估计，需进一步验证；②实验仅在单机模拟联邦，未考虑差分隐私、异步更新或客户端掉线；③通信效率对数据集结构敏感，需探索更鲁棒的聚合与稀疏化策略；④未覆盖策略可解释性与公平性等问题。

---

## 260. WebCQ: Cooperative Multi-Agent Deep Reinforcement Learning for Scalable Web GUI Testing

**arXiv ID:** 2606.22502 | [PDF](https://arxiv.org/pdf/2606.22502v1)

**作者:** Yujia Fan `[一作]` (Southern University of Science and Technology), Yepang Liu `[通讯]` (Southern University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 WebCQ，一个基于多智能体深度强化学习的可扩展 Web GUI 测试框架，能够在异步环境下通过 QTRAN 与轻量同步机制实现协作探索，并采用语义+探索特征的动作向量配合 DQN 进行动态动作空间决策。

**💡 创新点**

将多智能体 Web GUI 测试建模为 Dec-POMDP 并使用 QTRAN 进行 CTDE 学习；设计轻量级同步机制支持异步执行；使用基于文本相似度和执行频率的动作向量，解决动态动作空间；提出混合奖励函数；公开源码与实验数据。

**🔧 技术方法**

QTRAN、DQN、轻量同步机制、CTDE、Selenium-Python、PyTorch+DenseNet、GloVe 词向量、APIMiner、统计奖励函数、线程同步。

**📊 数据集**

在八个大型商业网站（如 Gap、GitHub、GameSpot、Smadex 等）上进行评测。

**📈 对比分析**

与两种 state‑of‑the‑art 方法（如 WebExplor / DQT）以及自身的两个消融版本（无通信、无 QTRAN）在同一时间预算和代理数下进行六次重复实验，评估已探索状态数、独特动作数、触发错误数；结果显示 WebCQ 在 7/8 网站中状态数最高，平均提升 33.3%，独特动作提升 42.2%，触发更多错误；20 小时长实验中表现更高执行效率；随着代理数增大性能稳步提升。

**⚠️ 局限性**

实验仅覆盖 8 个商业网站，结果对其他类型网站的泛化有限；在状态空间较小的网站上优势不明显；多智能体的通信开销在代理数过多时仍显著；仅支持英文内容，无法处理非英语网站；奖励与状态表示设计可能对不同网站的适用性有限。

---

## 261. Grounded Scaling: Why Agentic AI Needs Deterministic Environments

**arXiv ID:** 2606.22495 | [PDF](https://arxiv.org/pdf/2606.22495v1)

**作者:** Liang Ding `[一作]` (Alibaba Group), Xintong Wang `[通讯]` (Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出环境确定性是 AGI→ASI 进展的关键制约轴，并构建了供应确定性指数（SCI）与确定性成熟度模型（DMM），给出三条形式化定理与开放问题；

**💡 创新点**

1）将数据墙、抽象壁垒、具身瓶颈、多代理信任统一为“grounding”缺口；2）引入环境确定性度量 δ 并证明链任务成功概率随 δ^k 指数下降；3）设计 SCI 与 DMM 为可量化基准与升级阶梯；

**🔧 技术方法**

理论推导（概率论、信息论、RLHF/RLVR 等）、验证器设计与评估、可测量供应属性及指数聚合方法；

**📊 数据集**

主要使用商业自足供应链平台数据（B2B sourcing、金融结算、B2C 零售）及公开基准（τ-bench、OSWorld）进行实验；

**📈 对比分析**

采用匹配预算的三种训练方式（真实供应链验证器、模拟、递归自生成）比较样本效率与递归衰退时间；链任务成功率通过 δ^k 回归验证；实验预期 D3+ 环境显著优于 D2 及模拟；

**⚠️ 局限性**

需要可信验证器与确定性接口实现成本高；跨域可比性有限；治理与抗造假机制未解决；数据共享与隐私约束；评估仅在特定任务子集内有效。

---

## 262. PenduMorph: Development and Motion Analysis of Pendulum-Actuated Rolling Reconfigurable Spherical Robot with Magnetic-Coupling

**arXiv ID:** 2606.22491 | [PDF](https://arxiv.org/pdf/2606.22491v1)

**作者:** Aung Myat `[一作]` (Cardiff University), Seyed Amir Tafrishi `[通讯]` (Cardiff University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文设计并实现了PenduMorph，一种无线磁性耦合的可重构滚动球形机器人，内部装配摆动驱动、惯性测量与无线控制，能够在单体与多体状态下完成自主滚动与磁性连接。

**💡 创新点**

创新点在于将2自由度摆动驱动与全封闭球形结构结合，提供既能独立滚动又能通过磁性耦合实现多模块协同的模块化平台。

**🔧 技术方法**

技术实现包括内部摆动舵机驱动、Dynamixel舵机控制、BNO055惯性测量单元、Arduino MKR WiFi 1010无线通信、磁铁耦合模型以及五次多项式轨迹规划与PD控制。

**📊 数据集**

实验数据来自两台机器人在单机滚动、接近耦合、相对旋转和协同运动四种情境下的IMU姿态、关节状态与电机指令记录，无需公开数据集。

**📈 对比分析**

通过关节跟踪误差、姿态稳定性和耦合保持率等指标评估，单机滚动误差低、磁性耦合保持稳定，但相对旋转时出现粘滑噪声，整体性能良好。

**⚠️ 局限性**

局限性包括磁性耦合对动态摩擦与冲击的鲁棒性不足，缺乏全局轨迹控制与大规模多模块协同的实现。

---

## 263. VADAOrchestra: Neurosymbolic Orchestration of Adaptive Reasoning Workflows

**arXiv ID:** 2606.22485 | [PDF](https://arxiv.org/pdf/2606.22485v1)

**作者:** Teodoro Baldazzi `[一作]` (TU Wien), Emanuel Sallinger `[通讯]` (TU Wien)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并实现了 VADAOrchestra neurosymbolic 框架，将 LLM 作为流程编排器与基于 Vadalog 的符号推理引擎相结合，实现可审计、可解释的动态决策流程。

**💡 创新点**

创新点在于：① 将业务流程与 LLM 动态规划无缝对接，并在每一步生成可验证的逻辑规则；② 通过 on‑demand 规则合成与目标化数据检索，突破 LLM 上下文窗口与推理可扩展性瓶颈；③ 形成完整的可重现逻辑轨迹，满足金融监管对可追溯性的需求。

**🔧 技术方法**

技术包括：大型语言模型（Llama‑3.3‑70B 与 GPT‑4o）作为编排器；Vadalog（Warded Datalog±）作为符号推理引擎；MCP（Model Context Protocol）工具调用；基于依赖图的任务调度与动态规划；统计分析与 top‑k 数据检索；以及规则合成与语法验证。

**📊 数据集**

使用真实金融数据集：欧洲中央银行知识图谱（约10k条三元组）构成的 Bank 数据集（278 问题）和扩展的 Bank+ 数据集（最多1000条实体关系），涵盖浓度风险评估等业务场景。

**📈 对比分析**

与 LLM‑only、ReFactX、LLM+RAG、LLM+MCP 等基线对比，VADAOrchestra 在 EM 与 LLM‑as‑a‑Judge 两项指标上均高于对手，尤其在 GPT‑4o 模型下 EM 0.65、Judge 0.78，且在高复杂度查询下保持 0.63 EM，显著优于 RAG 与传统 agentic 方法，证明其在准确性和可扩展性上的优势。

**⚠️ 局限性**

局限性包括：① 对 LLM 生成规则的依赖，尤其在小模型下易出现语义错误，导致推理失效；② 规则合成与多轮 LLM 调用增加执行时间，超出传统单纯 agentic 系统；③ 对极大规模关系查询仍需进一步优化检索与并行化策略。

---

## 264. NeutronSparse: Coordinating Heterogeneous Engines for Sparse Matrix Multiplication on NPUs

**arXiv ID:** 2606.22482 | [PDF](https://arxiv.org/pdf/2606.22482v1)

**作者:** Xin Ai `[一作]`, Ge Yu `[通讯]` (Northeastern University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种针对NPU的SpMM框架NeutronSparse，针对异构引擎协调和基于Tile的调度进行优化。

**💡 创新点**

创新点在于：① 以硬件为导向的稀疏度阈值划分并在线迁移，实现AIC与AIV的负载平衡；② 全局-局部Tile重排与分层Tile重用，降低稀疏性导致的冗余计算与数据搬移。

**🔧 技术方法**

技术包括：稀疏度阈值成本模型、动态工作迁移、全局-局部Tile重排、层级Tile重用、双缓冲流水线以及对Ascend 910B AI核心的矩阵/向量引擎协同调度。

**📊 数据集**

使用SuiteSparse稀疏矩阵集以及四个GNN基准（Cora、ogbn-arxiv、reddit、amazon-product）等真实数据集。

**📈 对比分析**

与Ascend 910B的MindSporeGL、基于AIC的实现以及NVIDIA A100上的cuSPARSE、DTC-SpMM、HC-SpMM进行对比；NeutronSparse在NPU端实现1.26×–7.78×加速，在GPU端实现1.03×–3.07×加速。

**⚠️ 局限性**

局限性包括：对极端负载不平衡时仍需多轮迁移；迁移和重排的预处理开销虽小但不适用于极短任务；框架高度依赖Ascend 910B的AIC/AIV解耦设计，迁移到其他NPU需要额外适配；动态稀疏性变化时的自适应能力尚未充分验证。

---

## 265. Lighting-Consistent Object Transfer Across Radiance Fields

**arXiv ID:** 2606.22481 | [PDF](https://arxiv.org/pdf/2606.22481v1)

**作者:** Nicolás Violante `[一作]` (Inria), George Drettakis `[通讯]` (Inria)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `da1b1a89-583a-4b57-9c81-478778569bec` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

实现了将一个 3DGS 场景中的物体迁移到另一个 3DGS 场景，并通过扩散模型和后处理保证光照、阴影、反射在多视角下一致。

**💡 创新点**

核心创新是：①利用细调的扩散模型对单视图的合成图像进行光照和阴影和谐化；②通过 3DGS 的后优化将单视图结果融合成全局一致的 3D 表现，从而实现跨场景物体迁移的光照一致性。

**🔧 技术方法**

技术包括：3D Gaussian Splatting、基于 FLUX.1-schnell 的扩散模型、二进制特征用于 3D 分割、Perceptual 损失的后优化、以及自定义对象分割与去除网络。

**📊 数据集**

使用了三类数据：①30 个 Blender 生成的合成场景；②基于 FLUX 的生成图像；③ORIDa 实际捕获数据，全部用于构建光照不一致与一致的图像对。

**📈 对比分析**

与 LBM、MV‑CoLight、Nano Banana 等 2D 方法以及 Gaussian Shader、GS‑IR、3DGS‑DR 等 3D 方法进行比较。实验显示，在 PSNR、SSIM、LPIPS、FID、KID 等指标上，方法在合成场景中显著优于对比手段，且在多视角一致性上表现更佳。

**⚠️ 局限性**

局限性包括：对高频细节的 VAE 产生模糊或块状伪影；部分物体材质被过度改写导致外观变化；对极端光照差异或复杂场景的泛化仍有限。

---

## 266. Physically-guided Image Generation for Multi-Projection Mapping

**arXiv ID:** 2606.22477 | [PDF](https://arxiv.org/pdf/2606.22477v1)

**作者:** Xingyun Liu `[一作]` (Ningbo University), Chong Wang `[通讯]` (Ningbo University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了ConPhyG框架，能够在多投影映射中根据用户文本提示并可控地引入物理先验（深度、纹理边缘、色域），实现高质量、可部署的投影映射内容生成。

**💡 创新点**

核心创新在于：①统一的可控物理导向生成范式，可在合作模式与对抗模式之间切换；②引入可微分的In‑Gamut Determination Network实现对每像素色域约束的即时反馈；③通过多投影优化的有限变量最小二乘求解，保证物理可实现的投影光照。

**🔧 技术方法**

技术包括：基于SDXL的扩散模型+ControlNet控制、IGDN网络、边缘/深度/色域约束奖励机制、BVLS数值优化、360°视角的顺序生成策略以及多投影几何与辐射校准。

**📊 数据集**

使用真实实验场景中的四台Epson CB‑X31投影仪与Canon EOS 750D相机，捕获多种塑料/石膏模型，并在此基础上采集深度、边缘、色域、投影映射矩阵等物理先验。

**📈 对比分析**

与LAPIG和NepMap两种基线方法在真实投影图像上进行对比，评估指标包括CLIP、BLIP ITM、PSNR、ΔE、裁剪误差等。结果表明ConPhyG在语义一致性、颜色准确度与色域利用率上分别提升约10–30%/12–20%/≈90%，且设置时间和投影准备时间降低1–2个数量级。

**⚠️ 局限性**

主要局限在：①缺乏自动化的物理一致性判断与模式切换；②仅实现2.5D视角一致性，无法完全保证全360°多视角一致；③假设投影仪通道独立，未考虑更复杂的投影仪响应；④未显式建模间接照明、散射等光学效应，可能在复杂场景下导致补偿不足。

---

## 267. All Green, Still Broken: Real-Flow Verification Lessons from an LLM-Integrated, Multi-Market Web Application

**arXiv ID:** 2606.22475 | [PDF](https://arxiv.org/pdf/2606.22475v1)

**作者:** Muhammad Bilal `[一作]` (Technical University of Munich), Ali Hassaan Mughal `[通讯]` (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对一个集成大型语言模型、多市场国际化和浏览器前端的租赁搜索助手的测试过程进行回顾与分析，梳理了252条bug‑fix提交，量化了缺陷在四类“seam”（运行时、市场、流/交互、系统）上的逃逸情况，并提出了四‑seam框架与对应的低成本检查清单，以帮助团队识别并消除未被单元测试覆盖的边界缺陷。

**💡 创新点**

创新点在于①首次将缺陷逃逸聚焦到四个典型的边界 seam 上，并给出完整的定量分布；②设计了可自动化分类的 commit‑message 规则和公开的分类器；③将未覆盖的 seam 视为技术债务，提供针对每个 seam 的快速验证与保护措施；④提出了基于实时浏览器交互与多市场配置的“real‑flow”验证流程。

**🔧 技术方法**

使用的技术包括：pytest 自动化测试框架（1553 个测试用例）；基于关键词规则的无人工 commit 分类器；浏览器驱动（如 Selenium）实现实时交互验证；市场切换脚本与系统级监控脚本；将分类器和数据打包发布至 Zenodo。所有技术均为开源或自研工具。

**📊 数据集**

主要使用的数据集为项目自身的版本库快照，共 740 条提交，其中 252 条标记为 bug‑fix；测试结果与缺陷分布统计（如 1553 个测试用例、44% 缺陷落在四 seam 上、6 个生产缺陷等）。论文未使用公开的外部缺陷数据集。

**📈 对比分析**

通过对照缺陷归类结果，本文发现 44% 的 bug‑fix 来自四个 seam，表明单元测试对这些边界无覆盖；此外，通过实际案例展示，所有六个用户缺陷在部署前都已通过大规模测试，但仍被漏检。文章未给出传统性能指标（如测试覆盖率提升数值），但通过缺陷逃逸比例的对比与复现案例，证明低成本检查能及时捕捉并防止缺陷再次出现。

**⚠️ 局限性**

研究的局限性包括：①只针对单一项目，缺陷归类与检测方法尚未在其他系统验证；②分类依据 commit‑message 关键词，可能导致误标记；③未考虑无声逃逸（未被修复或未记录的缺陷）；④四 seam 的比例是保守下限，实际可能更高。

---

## 268. Human and AI collaboration for pulmonary nodule segmentation

**arXiv ID:** 2606.22486 | [PDF](https://arxiv.org/pdf/2606.22486v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 269. Study of Code-Aided Channel Estimation for Metasurface-Based Holographic MIMO Systems

**arXiv ID:** 2606.22465 | [PDF](https://arxiv.org/pdf/2606.22465v1)

**作者:** Roberto C. G. Porto `[一作]` (Pontifical Catholic University of Rio de Janeiro), Rodrigo C. de Lamare `[通讯]` (Pontifical Catholic University of Rio de Janeiro)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于元表面(HMIMO)的迭代检测、译码与信道估计框架；

**💡 创新点**

创新点在于利用LDPC校验位与系统编码的预留导频共同参与信道估计，结合闭式参数设计与交替优化；

**🔧 技术方法**

采用迭代检测与译码(SIC+LDPC)、LMMSE信道估计、闭式SIM/BD‑SIM参数设计与交替优化、QPSK调制；

**📊 数据集**

使用仿真数据：LDPC码块长度1024、比特率1/2、4用户、6 GHz、8个RF链、64个RIS元件、5层堆叠、城市微距环境；

**📈 对比分析**

与传统SIM‑RIS对比，BD‑SIM架构在NMSE和BER上均显著提升，单层时差距较小，层数增多时性能下降但迭代估计后可恢复；

**⚠️ 局限性**

局限在于层数增加导致信号衰减、估计误差累积，迭代收敛速度和复杂度仍需进一步评估。

---

## 270. A Differentiable Atari VCS:A Complex, Fully Known Ground Truth for Explainable AI

**arXiv ID:** 2606.22447 | [PDF](https://arxiv.org/pdf/2606.22447v1)

**作者:** Andreas Maier `[一作]` (Friedrich-Alexander-University Erlangen-Nuremberg), Patrick Krauss `[通讯]` (Friedrich-Alexander-University Erlangen-Nuremberg)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

重实现并提供可区分的Atari VCS仿真器

**💡 创新点**

构造可区分的硬件级仿真器，使梯度可追溯至已知硬件行为

**🔧 技术方法**

Julia/Zygote、JAX/XLA、软硬切换、直通估计、Gumbel-Softmax等技术

**📊 数据集**

Atari 2600 64款ALE游戏作为测试集

**📈 对比分析**

对齐xitari进行比对，取得64/64 RAM与像素完全一致，GPU批量吞吐约3M步/秒

**⚠️ 局限性**

未实现音频，未验证对学习代理的XAI方法，仍需进一步扩展

---

## 271. FetSelect: Task-Specific Architectures and Self-Supervised Learning for Automated Fetal Ultrasound Frame Selection

**arXiv ID:** 2606.22487 | [PDF](https://arxiv.org/pdf/2606.22487v1)

**作者:** Mahmood Alzubaidi `[一作]` (Hamad Bin Khalifa University), Marco Agus `[通讯]` (Hamad Bin Khalifa University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

研发了一套针对妊娠第一孕期CRL、NT、鼻骨和刻度条的自动超声图像帧选择系统。

**💡 创新点**

提出任务专一的双路径融合架构和针对超声的BYOL自监督预训练，显著提升帧质量评估。

**🔧 技术方法**

使用冻结的C‑RADIO‑B视觉基础模型、BYOL自监督、任务门控分类头、检测驱动质量头以及学习融合模块。

**📊 数据集**

利用19,019张无标签超声图像做SSL预训练，6,486张专家标注的多任务帧集做监督训练，并在509张外部CRL图像及4个外部视频上评估。

**📈 对比分析**

与无SSL基线、其他SSL方法及单路径设计进行对比，FetSelect在测试集上实现平均AUROC 0.956、AP 0.789、相关系数0.818，整体综合得分0.887，优于所有基线。

**⚠️ 局限性**

局限包括单一标注者、规则导向的质量标签、缺乏时序建模、单中心数据、种子数有限导致统计显著性不足。

---

## 272. CVSBench: A Comprehensive Benchmark for Cross-view Spatial Reasoning and Dreaming

**arXiv ID:** 2606.22476 | [PDF](https://arxiv.org/pdf/2606.22476v1)

**作者:** Ruixun Liu `[一作]` (Xi’an Jiaotong University), Xiangyong Cao `[通讯]` (Xi’an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文设计并发布了 CVSBench，一个基于卫星–街景图像对的大规模跨视角空间推理与想象基准，包含 VQA、定位与 grounding 等多任务，并提供 9,468 个跨视角 BBox、40,679 条 QA 及精细人工校验；

**💡 创新点**

创新点在于：① 统一跨视角空间推理评测框架，兼容 VQA、定位与 grounding；② 半自动化标注流程，实现精准跨视角 BBox 与对齐；③ 结合视觉想象（3D 场景生成与深度估计）与文本 CoT（结构化 CoT 与空间想象 CoT）两类策略；④ 将地理定位任务与 VQA 结合，提升人类式空间认知；

**🔧 技术方法**

技术手段包括：多模态大语言模型（如 Qwen3‑VL‑4B）、SFT+RL（GRPO）微调、结构化 CoT 与空间想象 CoT 设计、3D 场景想象管线（深度估计+nanobanana 3D 生成）、评估指标 Accuracy、mIoU；

**📊 数据集**

使用数据集：CVUSA‑subset（2,155 影像对）和 FOV‑subset（University1652，1,142 影像对）构成 CVSBench；参考并对比现有空间与遥感 VQA/grounding 资源；

**📈 对比分析**

实验通过与多种闭源（GPT‑5‑chat、GPT‑4o）和开源（InternVL‑3.5‑8B、Qwen3‑VL‑8B、Geochat‑7B、SpaceQwen 等）VLM 进行比较。闭源模型取得最高性能（VQA 约70%–80%，定位 约55%–60%，grounding 约10%–15%），开源 InternVL‑3.5‑8B 在 VQA 上接近闭源；人类基准约 90% VQA、98% grounding。CoT 及 3D 想象虽有提升，但整体提升有限；3D 视图在 FOV‑subset 上带来最大 3–4% 的提升；

**⚠️ 局限性**

局限性：① 现有 VLM 在跨视角一致性与对齐方面仍表现不佳；② 纯文本 CoT 仅提升有限，需更深层次视觉-语言整合；③ 3D 场景生成质量受限，难以处理全景图像，限制了想象效果；④ grounding 任务仍低 mIoU，显示跨视角对象匹配难度大；⑤ 数据集主要涵盖城市场景，缺乏动态或多模态跨视角对，未来可进一步扩展。

---

## 273. PRIME: Evaluating Prompt Resolution Under Incompatible Instructions in LLMs

**arXiv ID:** 2606.22470 | [PDF](https://arxiv.org/pdf/2606.22470v1)

**作者:** Tehreem Javed `[一作]` (National University of Sciences and Technology), Mehwish Fatima `[通讯]` (National University of Sciences and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 PRIME 框架和 PRIMEBench 数据集，对 LLM 在三种互斥元指令（推理、长度、格式）下的行为进行系统评估，并对五种开源指令微调模型进行实验。

**💡 创新点**

首次引入对冲突类型的系统化评价与确定性行为分类学，证明冲突类型比模型规模更能决定 LLM 的冲突解决策略。

**🔧 技术方法**

采用对冲突指令注入的 prompt 工程、贪婪解码推理、基于规则的 deterministic 分类以及卡方检验等统计方法。

**📊 数据集**

从 ConInstruct 派生的 PRIMEBench，包含 72 条基础指令（算术、逻辑、概念），经三种冲突生成后得到 216 条对抗性 prompt。

**📈 对比分析**

通过 Instruction Adherence Rate (IAR) 与 None Rate (NR) 评价模型，发现大多数模型至少满足一条指令，IAR 约 70%，且 Gemma/Mistral 偏向首条指令，TinyLlama 偏向次条指令，冲突类型是主导因素。

**⚠️ 局限性**

局限于 5 种开源指令微调模型、仅使用贪婪解码、数据集规模有限、未涵盖更复杂或多段式冲突指令。

---

## 274. Federated learning with heavy-tailed gradient noise and communication noise: a variance-reduction based algorithm

**arXiv ID:** 2606.22466 | [PDF](https://arxiv.org/pdf/2606.22466v1)

**作者:** Shengchao Zhao `[一作]` (China University of Mining and Technology), Yongchao Liu `[通讯]` (Dalian University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于方差减少的算法VRA-FedSGD，用于在重尾梯度噪声和通信噪声存在的情况下进行联邦学习。

**💡 创新点**

VRA-FedSGD结合了动量方差减少技术和非线性映射来减轻重尾梯度噪声，并使用方差减少聚合机制来抑制重尾通信噪声，首次在重尾噪声环境下提供了几乎确定的收敛率。

**🔧 技术方法**

使用了动量方差减少技术和非线性映射，结合了方差减少聚合机制。

**📊 数据集**

在真实世界数据集上进行了模拟实验，特别是在糖尿病数据集上进行逻辑回归问题的实验。

**📈 对比分析**

与现有的鲁棒分布式优化算法进行了比较，VRA-FedSGD在完美通信、Gaussian通信噪声和α-稳定通信噪声下均表现出更好的收敛性和最终准确性。

**⚠️ 局限性**

算法在处理重尾噪声时的效率和收敛性可能受到客户端参与率变化的影响，且在部分客户端参与的情况下，收敛速度可能较慢。

---

## 275. BLENDS: Bayesian Learning-Enhanced Deep Smoothing for GNSS-Denied Environments

**arXiv ID:** 2606.22456 | [PDF](https://arxiv.org/pdf/2606.22456v1)

**作者:** Nadav Cohen `[一作]` (University of Haifa), Itzik Klein `[通讯]` (University of Haifa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了 BLENDS，一种在 GNSS 阻塞环境中通过 Bayesian 学习增强的深度平滑框架，用于低成本惯导与 GNSS 的融合。

**💡 创新点**

创新点在于将 Transformer 生成的协方差缩放矩阵和偏差修正项嵌入经典 TFS，既保持 Bayesian 一致性，又通过学习补偿 GNSS 与 RTK 的系统性偏差。

**🔧 技术方法**

技术包括扩展卡尔曼滤波、两向平滑（TFS/RTSS）、Transformer 神经网络以及 Bayesian 一致性损失训练。

**📊 数据集**

使用 INSANE 四旋翼数据集的 Mars 沙漠子集，并人工插入 10 秒 GNSS 停止段进行评估。

**📈 对比分析**

与 EKF、TFS、RTSS 对比，BLENDS 在两个未见轨迹上平均降低约 25% 的 PRMSE 并使估计协方差缩小 70–85%。

**⚠️ 局限性**

限制在于仅测试了 10 秒短时 GNSS 停止段，未验证更长停机、不同平台或更复杂环境的泛化能力。

---

## 276. CASPER in the Machine: Insights into Character Variety in LLM-Generated Stories

**arXiv ID:** 2606.22454 | [PDF](https://arxiv.org/pdf/2606.22454v1)

**作者:** Anneliese Brei `[一作]` (University of North Carolina Chapel Hill), Snigdha Chaturvedi `[通讯]` (University of North Carolina Chapel Hill)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过引入八个角色描绘对立分类对（如风格化/自然化、连贯/不连贯等），构建了角色分类器，对比了4400篇LLM生成短篇与200篇人工写作短篇的角色特征分布，并探究模型规模、模型族、文体和多次生成的影响。

**💡 创新点**

创新点在于：①将叙事学中细粒度的角色分类迁移至自动化分析；②创建了可公开的、与人工文本对齐的LLM故事语料库；③通过多维度RQ揭示LLM角色产生的偏好与多样性，并对比不同模型族与规模的差异。

**🔧 技术方法**

技术上使用LLM-as-judge（基于大语言模型的判定）进行二元分类，并在多种提示设置（ICL、Zero-shot等）下评估效果；在数据预处理与对齐中使用自动分类器和人工标注校验。

**📊 数据集**

数据集包括200篇来自Reddit的人工写作短篇（按四大题材划分）以及由七个开放源代码LLM（共四族）在相同写作提示下生成的4400篇故事；所有故事均按人工故事主题生成提示，保证可比性。

**📈 对比分析**

比较方法主要是统计每个分类对的比例、平均向量以及标准差；结果显示LLM角色更倾向于风格化、动态且具有闭合结局，模型规模对角色分布影响不大，Phi族产生的角色最具多样性，Llama族最不多样；在不同题材下角色特征也存在显著差异。

**⚠️ 局限性**

局限性包括：人工故事可能被AI提升；分类器性能仍低于人工标注；仅覆盖英文创意写作提示，无法推广到其他文本域或语言；多次生成的可变性未完全解释。

---

## 277. Exact Nonnegative Matrix Factorization via Cone-Ray Witnesses: Obtuseness Ranking, Saturation Curves, and an Augmented Alt-LP Breakthrough

**arXiv ID:** 2606.22451 | [PDF](https://arxiv.org/pdf/2606.22451v1)

**作者:** Mithil Ramteke `[一作]` `[通讯]` (Qualcomm India Private Limited), Mithil Ramteke (Qualcomm India Private Limited)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了小规模精确秩矩阵的非负矩阵分解（NMF），通过一个锥-射线管道结合了截断奇异值分解（SVD）、非负前像的多面体锥、双重描述法（DDM）和交替线性规划（alt-LP）进行松弛最小化。

**💡 创新点**

创新点在于提出了一种混合方法，通过在每对（T, K）中尝试闭式形式的M_T，如果不可行，则通过增加两个额外的射线来扩展支持，从而提高了成功率。

**🔧 技术方法**

使用了截断奇异值分解（SVD）、双重描述法（DDM）和交替线性规划（alt-LP）等技术。

**📊 数据集**

使用了Olivetti人脸数据集（400 × 4096）和随机生成的10 × 10矩阵进行Monte Carlo实验。

**📈 对比分析**

与传统的乘法更新和分层交替最小二乘法（HALS）方法进行了比较，结果显示在r = 4, 5, 6时，成功率分别提高到99/95/75，显著优于基线方法。

**⚠️ 局限性**

限制在于DDM的可扩展性，尤其是在处理大规模数据时，且在r = 6时的结构性壁垒未能被突破。

---

## 278. Benchmarking Vision-Language Models for Microscopic Plant Image Understanding

**arXiv ID:** 2606.22497 | [PDF](https://arxiv.org/pdf/2606.22497v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 279. Risk-Aware Information Theory

**arXiv ID:** 2606.22524 | [PDF](https://arxiv.org/pdf/2606.22524v1)

**作者:** Hamidou Tembine `[一作]` (Universite du Quebec a Trois-Rivieres), Hamidou Tembine `[通讯]` (Timadie)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出基于期望值替换为期望极值的风险感知信息理论，并引入了期望极值熵、KL散度和互信息等量，证明了在非均衡风险参数下这些量与香农信息量本质不同，能够捕捉信息尾部风险。

**💡 创新点**

创新点在于：①首次用期望极值构造风险敏感信息度量；②证明期望极值KL散度在风险渴求区间可为负；③展示风险参数导致的可达率区域自适应变化，并将其映射为均值场类型博弈；④揭示香农信息无法衡量极端风险，奠定了风险感知通信与学习的新基础。

**🔧 技术方法**

核心技术包括：期望极值定义及其性质（翻译不变性、正齐次性、子可加性、τ单调性）；信息量的期望极值重写；期望极值交叉熵与KKT分析；多用户信息理论中的非线性链式规则与容量区域推导；均值场博弈框架；以及期望极值的数值求解（固定点迭代）。

**📊 数据集**

本文为纯理论研究，未使用公开数据集；所有结果基于数学证明与理论分析。

**📈 对比分析**

通过理论证明和 Monte‑Carlo 仿真（10^5 次）比较，风险参数 τ>0.5 时的期望极值容量区域显著扩张至香农容量之外；τ<0.5 时可出现负的 KL 散度与互信息；整体上风险感知模型在捕捉尾部信息与提升极端条件下性能方面优于传统香农模型。

**⚠️ 局限性**

局限性包括：①期望极值需要二阶矩有限，限制了对重尾分布的适用；②非线性导致信息量缺乏可加性与链式规则，增加了计算与分析复杂度；③缺乏闭式表达式，需数值求解；④在多用户博弈中需假设策略空间凸而上界不易获得；⑤未给出实验验证，仅为理论与仿真验证。

---

## 280. Scalable Multi-Task Data Generation via Reinforcement Learning for Language-Conditioned Bimanual Dexterous Manipulation

**arXiv ID:** 2606.22471 | [PDF](https://arxiv.org/pdf/2606.22471v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 281. Detecting and Understanding Vulnerabilities in Fully Homomorphic Encryption Frameworks

**arXiv ID:** 2606.22519 | [PDF](https://arxiv.org/pdf/2606.22519v1)

**作者:** Yiteng Peng `[一作]` (Hong Kong University of Science and Technology), Shuai Wang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了名为HERTA的自动化测试框架，利用元模测试检测并发现了21个全同态加密框架的逻辑错误和崩溃漏洞；

**💡 创新点**

创新性地设计了三层FHE专属的元模关系（类型/数据流、配置、后端/方案），结合FHE感知种子生成器和优先组合测试策略，首次实现对整个FHE框架栈的系统性、细粒度验证；

**🔧 技术方法**

采用元模测试（MT）、配对覆盖与优先组合测试、概率程序生成、噪声管理与优化路径探索等技术，并在Python环境中实现，针对Concrete、HEIR、HELayers等主流FHE框架；

**📊 数据集**

使用约3000个自动生成的FHE-aware seed程序作为测试集，涵盖多种算子、数据类型和配置；

**📈 对比分析**

与基准Fuzzer（如Domato）对比，HERTA在相同测试预算下发现更多逻辑错误（7/1）和崩溃（7/18），平均每核心小时可执行约293个测试用例，seed生成时间约1.16 ms，平均单例耗时约71.93 s，seed有效率达98.93%；

**⚠️ 局限性**

局限于核心算子层面，未覆盖所有高层接口（如ONNX）；元模关系和种子生成对框架版本依赖，需要持续维护；对近似算子的容差仍有限；未实现形式化验证，仅检测逻辑错误和崩溃。

---

## 282. Imagine to Ensure Safety in Hierarchical Reinforcement Learning

**arXiv ID:** 2606.22509 | [PDF](https://arxiv.org/pdf/2606.22509v1)

**作者:** Gregory Gorbov `[一作]` (Moscow Independent Research Institute of Artificial Intelligence), Aleksandr I. Panov `[通讯]` (Cognitive AI Systems Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ITES 方法，将安全子目标生成与想象安全相结合，在长时限 RL 任务中实现安全探索。

**💡 创新点**

首次在层次化策略中同时在高层生成安全子目标并在低层通过世界模型的想象执行安全，同时引入成本价值函数与拉格朗日约束实现全局安全保证。

**🔧 技术方法**

层次化强化学习（HRAC+Lagrangian）、轻量世界模型、成本模型、想象安全、拉格朗日乘子、经验回放与自适应训练。

**📊 数据集**

SafeAntMaze、SafePusher（长时限）以及 SafetyGym 的 PointGoal1、CarGoal1、PointGoal sparse（短时限）等任务环境。

**📈 对比分析**

与 Flat‑Safe RL 基线（CUP、FOCOPS、TD3LAG、SafeDreamer 等）以及 HRAC‑LAG 等进行比较，ITS 在长时限任务中成功率提升约 55% 并始终满足安全预算，在短时限任务中保持或超过基线性能。

**⚠️ 局限性**

需手工设计状态→目标映射 ϕ，限制可视化输入的适应性；世界模型在长距离想象中误差累积；对任务参数需要人工调节。

---

## 283. Not All Claims Are Equally Risky: FACTOR for Adaptive Verification in Factual Long-Form Generation

**arXiv ID:** 2606.22474 | [PDF](https://arxiv.org/pdf/2606.22474v1)

**作者:** Areeba Hassan `[一作]` (National University of Sciences and Technology), Mehwish Fatima `[通讯]` (National University of Sciences and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了FACT-OR，一个基于每条主张不确定性自适应的事实验证框架；

**💡 创新点**

通过动态调整NLI阈值，将验证力度集中在高风险主张上，实现风险感知的验证；

**🔧 技术方法**

综合使用检索（BM25+密集检索）、熵与语义一致性估计、DeBERTa NLI交叉编码器以及候选重排序；

**📊 数据集**

在FactScore传记基准（50个实体）和维基百科检索语料上进行实验；

**📈 对比分析**

与零射击、标准RAG、静态验证对比，FACT-OR在FactScore提升至约42%，幻觉率下降至57.7%，验证调用量显著减少；

**⚠️ 局限性**

在推理延迟与验证调用上仍高于静态验证，且轻量验证对低风险主张可能存在漏检风险。

---

## 284. The Power of Light: Improving Synthetic-to-Real Domain Adaptation through Physically-Based Indirect Illumination

**arXiv ID:** 2606.22574 | [PDF](https://arxiv.org/pdf/2606.22574v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 285. Deep material network for homogenization of piezoelectric composites

**arXiv ID:** 2606.22566 | [PDF](https://arxiv.org/pdf/2606.22566v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 286. Enabling Cloud-Level Accuracy in Edge AI through IoT Data Preprocessing

**arXiv ID:** 2606.22496 | [PDF](https://arxiv.org/pdf/2606.22496v1)

**作者:** Aygün Varol `[一作]` (Tampere University), Johanna Virkki `[通讯]` (Tampere University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在边缘 AI 环境中，通过提示侧预处理将原始 IoT 传感器数据转化为阈值感知和状态摘要文本，从而提升本地 LLM 的环境监测准确率与时延平衡。

**💡 创新点**

提出了三层结构化提示构造框架（原始、阈值感知、状态摘要），并证明其在本地模型上显著缩小与云端模型的性能差距，达到接近云端的准确率。

**🔧 技术方法**

采用 LLM 推理（No‑CoT 与 CoT）结合阈值判断、文本化表示和结构化提示，评估多种模型的推理性能。

**📊 数据集**

使用 Tampere 大学基于 Raspberry Pi + BME680 采集的室内数据，以及赫尔辛基、卡托维兹、华沙的室外空气质量数据集。

**📈 对比分析**

对五个本地模型和五个云模型在三种提示变体和两种推理模式下进行实验，结果显示在 Var.C（最丰富提示）且无 CoT 时，本地模型准确率可达 97% 以上，时延低于 0.3 秒，几乎与云端模型相当。

**⚠️ 局限性**

局限性包括仅评估二分类任务、未覆盖多类别决策、解释性生成、动态阈值检索及极低算力设备上的完整性能验证。

---

## 287. A Theory-grounded Hybrid Neural Network Integrating Complementary Estimation Mechanisms for Stable Visual Object TrackingA

**arXiv ID:** 2606.22604 | [PDF](https://arxiv.org/pdf/2606.22604v1)

**作者:** Yancheng Zhou `[一作]` (Tsinghua University), Yujie Wu `[通讯]` (Tsinghua University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于理论的ANN‑CANN混合网络HTNN，用于视觉目标跟踪；

**💡 创新点**

通过将ANN响应图与CANN动力学对齐，揭示了两者的bias‑variance互补性，并设计了两阶段融合实现功能协同；

**🔧 技术方法**

采用SiamFC作为ANN分支，利用连续吸引子网络（CANN）实现连续状态估计，并用FFT加速CANN的循环卷积；

**📊 数据集**

在九大公开跟踪基准（OTB50/100、GOT‑10k、LaSOT、TColor128、UAV123、NfS、VOT2019、TrackingNet）上进行评测；

**📈 对比分析**

与单一ANN、纯CANN、直接混合、FlyNet和HSTNN等基线比较，HTNN在Pr、SR及VNE上在大多数数据集上取得最佳或第二佳成绩，显著提升跟踪精度与稳定性；

**⚠️ 局限性**

在低帧率或高度离散运动场景（如GOT‑10k）下，CANN动力学的优势减弱，且模型对参数设置有一定依赖，需进一步完善适应性融合与更复杂的动力学设计。

---

## 288. From CVE to CWE: Syscall-Based HIDS Generalisation

**arXiv ID:** 2606.22581 | [PDF](https://arxiv.org/pdf/2606.22581v1)

**作者:** Alexander V. Kozachok `[一作]` (MIREA Russian Technological University), Shamil G. Magomedov `[通讯]` (MIREA Russian Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了基于系统调用跟踪的主机入侵检测系统（HIDS）在CWE级别下的泛化能力，即训练只用正常行为、但能检测同一CWE类别下不同CVE漏洞的攻击；

**💡 创新点**

创新点在于：①首次提出并评估CWE级别的一类异常检测方法；②采用固定阈值的正常数据校准，避免标签泄漏；③通过对跨CVE转移的方向性分析揭示正常行为宽度对泛化的主导作用；④研究了特征稳定性过滤对转移的负面影响；

**🔧 技术方法**

使用的技术包括：Peng‑Guo式66维特征提取、Isolation Forest和SGD One‑Class SVM模型、基于正常窗口的阈值校准算法、交叉CVE转移与合并CWE训练方案、基于KS距离的特征重要性与稳定性评分；

**📊 数据集**

使用的数据集为LID‑DS‑2021，共6个场景，分别覆盖CWE-307、CWE-89和CWE-434三类；

**📈 对比分析**

方法比较：在固定目标FPR（0.001、0.01、0.05）下，单场景自检F1≤0.31；跨CVE转移呈强方向性；合并CWE-307的模型在FPR=0.05时达到F1=0.698、真实FPR=0.070；CWE-89与CWE-434合并模型F1≤0.21，表现不佳；

**⚠️ 局限性**

局限性包括：仅评估了3个CWE类别且每类仅2个CVE；特征空间仍为传统统计特征，缺乏图结构或对比学习表征；阈值校准假设正常数据与生产相符，需定期更新；未来需扩展更多CWE和更丰富的特征表示。

---

## 289. Look Light, Think Heavy: What Multimodal Chain-of-Thought Reasoning Can and Cannot Do

**arXiv ID:** 2606.22565 | [PDF](https://arxiv.org/pdf/2606.22565v1)

**作者:** Zhuoran Jin `[一作]` (Chinese Academy of Sciences), Jun Zhao `[通讯]` (Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究系统评估了多模态Chain-of-Thought（CoT）推理在12个感知与推理任务中的效果，揭示CoT对不同任务的适用性与局限性；

**💡 创新点**

创新点在于对多模态CoT进行大规模、细粒度的任务分层分析，发现“视觉反思浅薄、文字反思深厚”的“Look Light, Think Heavy”现象，并指出现有开源模型对数学推理的过度偏重；

**🔧 技术方法**

采用CoT提示、RLVR训练、测试时缩放、视觉与文本推理探针、内部注意力可视化等技术，对多模态模型进行评估与机制剖析；

**📊 数据集**

使用12个任务的代表性数据集（如MathVista、MathVerse、MATH‑Vision、OCR、视觉定位、知识问答、对象计数等）以及每个任务1–3个公开基准；

**📈 对比分析**

将14个非推理模型与8个推理模型在直接回答与CoT两种方式下进行对比，发现对感知任务CoT往往导致性能下降（平均约4–5%），但对数学、科学与多图推理任务提升约6–5%；开源推理模型平均提升有限（≤5%），而商业模型如Gemini-2.0-Flash‑Thinking在多任务上提升显著；

**⚠️ 局限性**

局限性包括：仅覆盖少量数据集与任务，未涵盖视频等更复杂模态；对视觉工具利用不足，导致视觉反思浅薄且难以在缺失关键信息时实现适当拒答。

---

## 290. ASAP: A Disaggregated and Asynchronous Inference System for MoE Prefill

**arXiv ID:** 2606.22541 | [PDF](https://arxiv.org/pdf/2606.22541v1)

**作者:** Weiwei Chen `[一作]` (Huawei), Zhibin Yu `[通讯]` (Huawei)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套专门针对MoE模型预填充阶段的异步推理系统，拆分注意力与专家模块并消除全局同步阻塞；

**💡 创新点**

核心创新在于：①基于共享内存的异步通信原语实现1对多/多对1非阻塞数据转移；②长度感知批处理、双批交错、三流通信与计算重叠；③无层级依赖的MoE超核（Super Kernel）实现动态层调度，消除主机调度瓶颈；

**🔧 技术方法**

使用的技术包括：分布式共享内存缓冲区、异步异构通信原语、三流并发模型、层无关MoE超核、长度感知批处理与双批交错调度；

**📊 数据集**

实验采用公开的DeepSeek‑V3.2 MoE模型（671B参数、256专家）以及真实业务请求长度分布（最大32k token）作为数据集；

**📈 对比分析**

与两种同步基线（Default与ChunkedPrefill）对比，测量TTFT和5s SLO下的吞吐率。异步系统在TTFT上实现了高达68%降低，SLO合规吞吐提升近90%（从10.5 RPS提升至20 RPS）；

**⚠️ 局限性**

局限性主要有：①目前仅针对预填充阶段，解码阶段收益有限；②依赖高速共享内存（如SuperNode）或低延迟RDMA；③对极长序列（>32k token）仍需单独的序列并行策略；

---

## 291. Generative Robust Optimisation

**arXiv ID:** 2606.22536 | [PDF](https://arxiv.org/pdf/2606.22536v1)

**作者:** Yuhui Yin `[一作]` (University College London), Vassilis M. Charitopoulos `[通讯]` (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出Generative Robust Optimisation框架，将深度生成模型用于构建可校准、可嵌入MILP的非线性不确定性集合，并给出五点评估标准。

**💡 创新点**

通过将解码器映射到标准正态潜在空间实现校准，配合GMM引导、鲁棒相关性损失与ReLU单通道解码，首次实现可完全满足五点标准的生成不确定性集合。

**🔧 技术方法**

使用Wasserstein对抗自编码器、Gaussian Mixture Model约束、最大均值差距、混合整数线性规划（MILP）嵌入、批量校准与最优性基准化等技术。

**📊 数据集**

实验使用六种人工合成的成本/需求分布（独立/相关、正态/偏斜/混合）以及Baron 15节点多周期设施定位数据。

**📈 对比分析**

与传统盒子、椭圆、分布鲁棒、GAN等模型以及WAAE、DDIM等对照，WAAE‑GMM‑RO在保守性、安全性与计算可行性上优于其他方法，最坏情况求解时间可达数分钟至数小时。

**⚠️ 局限性**

对高维或多步生成模型的MILP嵌入导致二进制变量数量激增，计算量显著；当潜在分布与真实分布偏差大时仍需改进校准；仅针对线性预算/生产约束，扩展到更复杂非线性约束仍有挑战。

---

## 292. Scalable Maximum Entropy Reinforcement Learning for Diffusion Policies via Adjoint Matching

**arXiv ID:** 2606.22630 | [PDF](https://arxiv.org/pdf/2606.22630v1)

**作者:** Serge Thilges `[一作]` (Karlsruhe Institute of Technology), Gerhard Neumann `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过将最大熵强化学习建模为随机最优控制问题，本文提出了Adjoint Matching Diffusion Policy（ADDP），实现了在线RL中扩散策略的可扩展训练；

**💡 创新点**

创新点在于利用递归对偶匹配实现无模拟、低内存梯度更新，并引入误差函数动作压缩与信赖域约束，显著提升训练稳定性与效率；

**🔧 技术方法**

采用扩散模型、递归对偶匹配、最大熵RL、Q-score、误差函数压缩、信赖域更新以及无模拟回归损失等技术；

**📊 数据集**

在MuJoCo Playground、ManiSkill、HumanoidBench以及DeepMind Control（dog/humanoid）等连续控制数据集上进行实验；

**📈 对比分析**

与REPPO、PPO、SPO、DPPO、FPO、DIME、QSM等基线对比，ADDP在大多数任务上达到或超越基线性能，样本效率与训练时长与高效高斯策略相当，且在高维任务中表现尤为突出；

**⚠️ 局限性**

限制包括对准确Q网络的依赖、需要调节信赖域参数、在极端高维或复杂动态下仍可能出现稳定性问题，以及对非马尔可夫环境的适用性尚未验证。

---

## 293. Orthogonal Representation Editing: Decoupling Semantic Entanglement in Batch Knowledge Editing of LLMs

**arXiv ID:** 2606.22627 | [PDF](https://arxiv.org/pdf/2606.22627v1)

**作者:** Wenhao Yu `[一作]` (Tianjin University), Nayu Liu `[通讯]` (Tianjin University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究批量知识编辑中出现的语义表示纠缠问题，并提出 Orthogonal Representation Editing (ORE) 方法，在隐藏表示空间构造通用语义子空间并对编辑向量正交化，结合门控非线性表示头实现精准、可控的知识注入。

**💡 创新点**

创新点：①在表示空间而非参数空间引入正交约束，显式剔除共享语义噪声；②设计门控非线性表示头，动态定位编辑位置；③利用低秩瓶颈和 ReFT 框架实现高效可插拔的编辑机制。

**🔧 技术方法**

技术手段：主成分分析构建通用语义子空间；正交投影消除语义干扰；低秩瓶颈（SiLU 激活）与动态门控（STE 近似）实现非线性、可微的编辑操作；复合损失（正交、门控、交叉熵、KL 散度）训练模型。

**📊 数据集**

数据集：ZsRE、CounterFact、Bi‑ZsRE（中英双语）；模型基线为 LLaMA‑3‑8B 和 Qwen‑2.5‑7B。

**📈 对比分析**

与 FT、ROME、MEMIT、PRUNE、RECT、NSE、AlphaEdit、ReFT 等基线对比，ORE 在效能（Efficacy）和泛化（Generality）方面显著优于对手，尤其在跨语言情形下提升约4–5个百分点；在 Specificity 上略逊于部分参数编辑方法，但整体表现优于现有方案。

**⚠️ 局限性**

局限性：①仅在 7B–8B 规模模型验证，需进一步验证更大模型的可扩展性；②评估聚焦标准编辑任务，未覆盖多任务或复杂应用场景；③未系统评估编辑对逻辑推理等综合能力的影响。

---

## 294. What are Key Factors for Updates in RL for LLM Reasoning?

**arXiv ID:** 2606.22570 | [PDF](https://arxiv.org/pdf/2606.22570v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 295. SeFi-Image: A Text-to-Image Foundation Model with Semantic-First Diffusion

**arXiv ID:** 2606.22568 | [PDF](https://arxiv.org/pdf/2606.22568v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 296. Sub-Billion, Super-Frontier: Small Language Models Rival Zero-Shot Frontier LLMs on General and Literary Relation Extraction

**arXiv ID:** 2606.22606 | [PDF](https://arxiv.org/pdf/2606.22606v1)

**作者:** Despina Christou `[一作]` (Aristotle University of Thessaloniki), Grigorios Tsoumakas `[通讯]` (Aristotle University of Thessaloniki)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究评估并微调了5个小型语言模型（360M–3B参数），在三种训练模式（仅一般域、仅文学域、混合域）和两种提示风格（0-shot、2-shot）下，构建30个配置，针对9个关系抽取（RE）基准（包括新闻、维基百科、文档级和小说类）进行实验，并与前沿大型语言模型（GPT‑5.4、Claude Sonnet 4.6）以及RoBERTa编码器基线进行对比。

**💡 创新点**

创新点在于：①系统化地探索小模型在多域RE任务中的可行性；②证明通过领域适配与提示式微调，小模型可在单GPU上实现高效部署并超过大型专有模型；③揭示提示式训练对小模型的显著提升，而对大模型影响有限；④通过领域自适应预训练（DAPT）验证其对文学RE的边际收益几乎为零。

**🔧 技术方法**

采用的技术包括：QLoRA（4‑bit量化+LoRA适配）、prompt‑conditioned微调（两次示例）、schema‑enumerated与通用提示对比、基于文本生成的关系抽取框架，以及对RoBERTa的实体标记分类器进行微调；实验在NVIDIA RTX 4090上完成。

**📊 数据集**

使用的数据集包括：一般域——TACRED、SemEval‑2010 Task 8、CoNLL04、NYT11、GIDS、Re‑DocRED、REBEL；文学域——Biographical（Biographical dataset）、PG‑Fiction（Project Gutenberg小说）。

**📈 对比分析**

评估指标为正类微F1（去除NA/Other等否定类）。结果显示：最佳子亿级配置（Qwen2.5‑0.5B GenTune 2‑shot）在一般域平均F1为0.828，已超过GPT‑5.4（0.693）与Claude Sonnet（0.662）；最佳3B模型（Llama‑3.2‑3B GenTune 2‑shot）在一般域达到0.844；在文学域，SmolLM3‑3B LitTune 0‑shot实现0.833，对比GPT‑5.4的0.578、Claude 0.530，优势超过25个百分点。混合域模型在两域均保持接近专家水平，差距仅2–3个百分点。

**⚠️ 局限性**

局限性包括：①对特定数据集的依赖（如Biographical的训练/测试重叠导致的过拟合风险）；②只评估句子级关系抽取，未覆盖多句或文档级抽取；③某些配置在推理时出现推理占位符或schema混淆；④领域自适应预训练的实验仅在单一模型、单一语料上，结果可能不具普适性；⑤缺乏对低资源语言和多语言场景的验证。

---

## 297. Robust SCMA Codebook Design: A Hardware-Aware Autoencoder Approach

**arXiv ID:** 2606.22603 | [PDF](https://arxiv.org/pdf/2606.22603v1)

**作者:** Zihao Liu `[一作]` (University of Essex), Leila Musavian `[通讯]` (University of Essex)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出一种硬件感知的端到端自动编码器框架，用于在 OFDM-SCMA 系统中针对载频偏移（CFO）和相位噪声（PN）设计鲁棒的码本。

**💡 创新点**

创新点在于将可微的 CFO 与 Wiener 过程 PN 层嵌入到自动编码器训练循环中，并采用重参数化与基于联合上界的 Soft-MED 损失以及软硬间隔惩罚，实现对随机相位失真的自适应几何结构（环形码本）。

**🔧 技术方法**

使用的技术包括可微自动编码器、重参数化技巧、对数消息传递算法（Log‑MPA）解码器、复合损失函数（Task、Hinge、MED）以及梯度下降优化。

**📊 数据集**

使用的数据集为仿真生成的多径 Rayleigh 信道、CFO 和 PN 随机变量，实验中不使用真实测量数据。

**📈 对比分析**

通过与 Deka 等人提出的传统码本以及现有 PN‑鲁棒码本在 CFO、PN 与 Eb/N0 变化下的 BER 对比，所设计的码本在所有评估场景下均显著降低误码率并消除了误码率饱和现象。

**⚠️ 局限性**

局限性包括：仅在理想下行链路、完美 CSI 条件下评估；训练仅在单一失真点完成，可能在极端失真下性能下降；需要离线训练，无法即时自适应实时变化的硬件失真。

---

## 298. Text2DSL: LLM-Based Code Generation for Domain-Specific Languages

**arXiv ID:** 2606.22586 | [PDF](https://arxiv.org/pdf/2606.22586v1)

**作者:** Alexander V. Kozachok `[一作]` (Russian Technological University), Shamil G. Magomedov `[通讯]` (Russian Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文正式定义了从自然语言生成域专用语言（DSL）代码的 Text2DSL 任务，构建了 PolkitBench 数据集，并在两款 MoE LLM 上验证了在 Prompt 中注入 BNF 语法、API 规范和限定词汇能显著提升代码生成质量。

**💡 创新点**

创新点在于：①把 DSL 代码生成独立为一种新任务类；②提出三层 AST 验证与 CodeBLEU/Jaccard 评价体系；③创建了 4204 条自然语言–Polkit 规则对的数据集；④证实结构化上下文注入可在不微调模型的情况下大幅提升 LLM 的语法与语义正确性。

**🔧 技术方法**

使用技术包括：Bnf 语法 + API 规范 + 词汇表的 Prompt 注入；GigaChat-10B-A1.8B 与 Nemotron-3-Nano-30B-A3B 两款 MoE LLM；三层 AST 验证（语法、结构、语义）；CodeBLEU 与 Jaccard 评价指标；并在实验中对比基线与上下文增强两种 Prompt。

**📊 数据集**

使用数据集为 PolkitBench，包含 4,204 条经三层 AST 验证的自然语言–Polkit 规则对。

**📈 对比分析**

实验对比基线无上下文与上下文增强两种 Prompt，结果显示：在 GigaChat 上语法有效率从 80.5% 提升至 99.4%，结构有效率从 60.4% 提升至 95.9%；Nemotron 同理提升；CodeBLEU 和 Jaccard 均出现显著提升（GigaChat Jaccard 从 0.146 提升至 0.633）。

**⚠️ 局限性**

limitations：仅评估了两款 MoE LLM，缺乏组件级消融实验；查询仅来源于 50 个模板，缺乏真实多样性；词汇表规模受限，过大时可能超出上下文窗口；尽管上下文提升显著，但仍存在词汇错误和语义误差；方法仅适用于具备正式语法和词汇表的 DSL。

---

## 299. PolicyTrim: Boosting Intrinsic Policy Efficiency of Vision-Language-Action Models

**arXiv ID:** 2606.22540 | [PDF](https://arxiv.org/pdf/2606.22540v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 300. MacAgentBench: Benchmarking AI Agents on Real-World macOS Desktop

**arXiv ID:** 2606.22557 | [PDF](https://arxiv.org/pdf/2606.22557v1)

**作者:** Yikun Fu `[一作]` (Shanghai Jiao Tong University), Bowen Zhou `[通讯]` (Shanghai AI Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MacAgentBench，一个涵盖676个macOS桌面任务的综合性评测基准，支持GUI、CLI及混合交互，并提供确定性规则评估和多检查点细粒度评分；

**💡 创新点**

创新点在于开放式环境设计，使不同CUA框架（纯GUI、混合框架、代理工具箱）可公平对比；引入多检查点评分揭示任务完成的细粒度进度与能力维度；展示框架设计与预置技能对性能的独立影响；

**🔧 技术方法**

使用Docker-QEMU虚拟化的macOS容器、AppleScript、Shell、Python脚本进行环境配置与评估；评估框架包括OpenClaw、AgentS3和纯GUI代理；模型覆盖Claude Opus 4.6、GPT-5.4、Gemini 3.1 Pro等；

**📊 数据集**

任务数据集来源于macOSArena、iWork套件及自设计任务，通过参数替换和LLM重写扩展为676个任务，覆盖25个应用，约60%涉及GUI+CLI；

**📈 对比分析**

在三种框架和16种模型下评估，Claude Opus 4.6+OpenClaw在Pass@1上最高达73.7%；框架对比显示OpenClaw和AgentS3均显著提升于纯GUI基线，OpenClaw优势主要来自技能库；细粒度评分揭示不同模型在Research、AppState、Content、FileOps、SysConfig等维度的能力不均衡；

**⚠️ 局限性**

局限性包括：仅在macOS Tahoe 26上验证，无法直接迁移至其他版本；使用QEMU虚拟化缺少Apple GPU，可能影响GPU加速功能；任务集仍有限，可进一步扩展到更多应用与场景；

---

## 301. Training-Free Semantic Correction for Autoregressive Visual Models

**arXiv ID:** 2606.22550 | [PDF](https://arxiv.org/pdf/2606.22550v1)

**作者:** Junhao Chen `[一作]` (Zhejiang University), Shengyu Zhang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 GA​ZER 框架，在下一尺度自回归视觉生成过程中，利用多模态大型语言模型 (MLLM) 在生成途中进行诊断和修正，从而在不需要额外训练的前提下提升语义对齐。

**💡 创新点**

创新点在于：① 在中间尺度构建可供诊断的 Rollout Preview；② 通过 MLLM 生成增强/抑制提示；③ 采用 rewind‑and‑resample（回溯并重新采样）机制，将诊断信息动态注入采样流程；④ 全程保持模型、解码器和 MLLM 冻结，纯粹改进推理策略。

**🔧 技术方法**

主要技术包括：next‑scale 采样、rollout preview 构造、MLLM 语义评估、增强/抑制提示编码、分类器自由引导 (CFG) 重新采样、以及多尺度诊断调度。

**📊 数据集**

使用的基准数据集有：
- Text‑to‑Image：T2I‑CompBench；
- Text‑to‑Video：T2V‑CompBench；
此外还在图片端使用 CLIP Score、PickScore、HPSv2、ImageReward 等指标进行评估。

**📈 对比分析**

与原始模型（无任何改动的 baseline）和 Best‑of‑2 采样方式对比，GA​ZER 在 T2I‑CompBench 与 T2V‑CompBench 上显著提升了多种组合性指标（形状、纹理、颜色、空间关系、数字化等），并在 ImageReward 等人类偏好指标上保持竞争力；推理时间约为标准采样的 1.63 倍，低于 Best‑of‑2 的 2.00 倍，显现出更高效的语义校正。

**⚠️ 局限性**

局限性包括：① 仅适用于 next‑scale 预测模型，无法直接迁移到 raster‑scan 的 next‑token 模型；② 对于极细粒度或高度模糊的提示，MLLM 诊断信号可能不足；③ 对动态属性和某些模型特定维度的改进有限；④ 依赖 MLLM 质量，若 MLLM 评估不准则可能导致错误修正。

---

## 302. Non-Uniform L2 Cache Latency Across the Streaming Multiprocessors of an NVIDIA L40

**arXiv ID:** 2606.22588 | [PDF](https://arxiv.org/pdf/2606.22588v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 303. Supporting Tutors in the Gig Economy with Automated Feedback: A Case Study on Ringle

**arXiv ID:** 2606.22609 | [PDF](https://arxiv.org/pdf/2606.22609v1)

**作者:** Yeon Su Park `[一作]` (KAIST), Juho Kim `[通讯]` (KAIST)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究在Ringle在线英语辅导平台上开发并部署了一个基于AI的自动化反馈系统，随后对36名新上岗导师进行问卷调查，收集并对比了自动化反馈与学习者反馈的感知与效果。

**💡 创新点**

创新点在于：①首次在教育型零工平台中引入透明化、基于九项教学标准的数值化自动反馈；②通过多源反馈比较揭示自动化反馈在自我监控与平台期望理解上的价值；③提出针对多源反馈设计的三项关键设计原则，指导未来系统的构建。

**🔧 技术方法**

技术手段包括：使用预训练语言模型进行少量示例提示（few‑shot prompting）以生成按课程结构、参与度等九类评分；基于课堂转录和导师笔记生成数值反馈；对反馈数据做统计分析和主题分析。

**📊 数据集**

数据集涵盖：36名导师在前十节课的自动反馈结果（共327节课），10,000节随机抽取的学习者反馈（评分、重访意向、书面评论），以及导师笔记与课堂录音文本。

**📈 对比分析**

比较方法：利用Wilcoxon符号秩检验比较学习者反馈与自动反馈在理解、准确性、公平性等七维度的Likert评分；绘制学习者与自动反馈分数直方图比较分布；通过主题分析解读开放式问卷。结果显示：导师普遍对学习者反馈评价更积极，自动反馈在多数维度上更负面，但在自我监控和平台期望理解上被认为更有用，且自动反馈提供了更细粒度、连续的评分分布。

**⚠️ 局限性**

局限性包括：①反馈尺度相似导致导师混淆、缺乏情境解释；②样本仅来自Ringle且导师人数有限；③未跟踪长期教学效果，无法确认自动反馈对教学质量的因果影响；④模型的评分与人类评估仍存在差异，需进一步校准。

---

## 304. SkillAudit: From Fixed-Suite Benchmarking to Skill-Centered Assessment

**arXiv ID:** 2606.22613 | [PDF](https://arxiv.org/pdf/2606.22613v1)

**作者:** Dexu Yu `[一作]` (Northeastern University), Chunxiao Li `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 SkillAudit，一种面向任意 LLM agent 技能的端到端评估框架，能够自动生成多维度评估报告，包括实用性、效率/成本和安全性。

**💡 创新点**

创新点在于：① 将评估聚焦于技能本体而非固定任务集；② 采用基准比较原则精确度量技能增益；③ 引入两阶段安全检测（静态语义扫描 + 动态运行验证）；④ 通过浏览器扩展将评估结果即时呈现给开发者，支持即刻决策。

**🔧 技术方法**

主要技术包括：LLM（Claude/Code）解析技能描述生成评估方案；基于 Docker 的隔离沙箱执行任务并收集完整日志；静态扫描器识别 21 种风险模式；动态探针通过运行时测试确认可利用性；LLM 判题与可审计的评估脚本。

**📊 数据集**

使用了 226 个公开技能包（来自 GitHub 等平台），覆盖 23 个职业类别，并在多种 agent‑model 配置（Codex/GPT‑5.4、Claude Code/Sonnet‑4.6 等）上进行实验。

**📈 对比分析**

方法上与现有 benchmark（AgentBench、SkillsBench 等）对比，采用 PRG、EG、CG、ECG 等度量。实验结果显示：平均 PRG ≈ 0.18，效率‑成本综合得分为负值，说明多数技能在执行时间/代币上略有开销；安全得分整体良好，约 7.5% 的技能被判定为高风险。相比传统固定任务基准，SkillAudit 能更真实地反映技能在不同背景下的增益与风险。

**⚠️ 局限性**

局限性：评估仅覆盖 226 个技能和少量 agent‑model 组合；每个条件只做单次匹配跑；安全检测仅覆盖已知 21 种风险模式；未考虑多次运行的方差或隐藏的风险，结果可能随生态演化而变化。

---

## 305. DR-Mamba: Automatic Inference-Time Domain Adaptation for Document Image Binarization via Sample-Conditioned Detail-Background Suppression

**arXiv ID:** 2606.22625 | [PDF](https://arxiv.org/pdf/2606.22625v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 306. On Good Authority: Release-Authority Measurement for Registry-Mediated Package Ecosystems

**arXiv ID:** 2606.22593 | [PDF](https://arxiv.org/pdf/2606.22593v1)

**作者:** Igor Santos-Grueiro `[一作]` `[通讯]` (International University of La Rioja), Igor Santos-Grueiro (International University of La Rioja)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了前置者感知的发布权限记录，评估了五大生态系统（npm、PyPI、Maven Central、crates.io、RubyGems）以及 Go 边界适配器的公开发布路径变化，并将其用于发布时的审计队列。

**💡 创新点**

提出了公开发布路径断点触发的透明政策和前置者感知记录方法，能在不依赖依赖图的情况下捕捉可审计的发布审计表面，并揭示不同生态系统权威信号对触发的差异。

**🔧 技术方法**

利用注册表、证明（provenance）API、签名/完整性证明、仓库关联、工作流元数据等公共证据，构建继承比较记录；实现规则触发、距离阈值、学习排序（逻辑回归/轻量提升）以及审计纠错。

**📊 数据集**

基于 2024‑04‑06 至 2026‑06‑13 的 45,812 次发布（43,100 次可比），约 942 个坐标；Go 7,123 次发布；外部恶意软件 Feed（OpenSSF/OSV MAL、Datadog）与事件、advisory、审计结果对齐。

**📈 对比分析**

通过 exact trigger、距离阈值以及两阶段压缩学习排序评估队列覆盖率、工作量和 AUPRC；exact trigger 产生 204 个触发；距离阈值覆盖 93–100% 触发；学习排序在宽阈值队列中将所需审核数从 179 降至 39；历史上下文预测表现弱。

**⚠️ 局限性**

仅能检测公开发布路径变化，无法发现同一路径恶意、已废弃版本或无快照版本；依赖注册表与证明可见性，导致不同生态系统阈值不统一；外部恶意库对齐有限；学习模型受训练集包泄露风险；Go 的边界适配仍处于实验阶段。

---

## 307. 4DVLT: Dynamic Scene Understanding with Worldline-Centered Vision-Language Tracking

**arXiv ID:** 2606.22631 | [PDF](https://arxiv.org/pdf/2606.22631v1)

**作者:** Chaoyue Li `[一作]` (Huazhong University of Science and Technology), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出4DVLT任务与Instruct-4D基准，设计4DTrack实现指令驱动的4D世界线推理

**💡 创新点**

首次将语言定位、3D度量、跨视角2D投影统一成完整世界线输出，并通过图路由、双向解码与运动先验实现全局一致

**🔧 技术方法**

构建4D状态图、查询引导路由、离线双向世界线推理、物理运动先验校正，模型以Qwen3.5-9B LMM为骨干

**📊 数据集**

使用Instruct-4D（EgoWL基于nuScenes，AlloWL基于WildTrack），共129.4K问答、64.7K实体、851场景

**📈 对比分析**

与四个适配VLT基线和四个LMM对比，4DTrack在TGA_Top1、TGA、WQS、CTQ等指标上领先19–33点，ADE_3D从8.05m降至3.67m

**⚠️ 局限性**

在相似类别高密度场景（AlloWL）中身份消歧仍较困难，难以充分利用细粒度相对关系

---

## 308. Federated Learning for Global Carbon Emission Forecasting: A Hybrid Time-Series Approach with Statistical and Neural Models

**arXiv ID:** 2606.22618 | [PDF](https://arxiv.org/pdf/2606.22618v1)

**作者:** Attia Qammar `[一作]` (Southwest Jiaotong University), Tianrui Li `[通讯]` (Southwest Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种在联邦学习框架下的碳排放预测模型，将统计学模型与深度学习模型结合，支持多国多行业分布式数据的联合训练。

**💡 创新点**

创新点在于：①将ARIMA提取趋势与GARCH捕捉波动的统计特征与LSTM‑Attention时序表示、XGBoost回归相融合；②在隐私保护的联邦环境中实现多模型组件的无共享数据协同；③通过统一特征级融合与FedAvg聚合，兼顾时序依赖、波动性和非线性模式。

**🔧 技术方法**

使用技术包括：联邦平均（FedAvg）聚合；ARIMA、GARCH时间序列建模；LSTM‑Attention深度时序网络；XGBoost梯度提升回归；滑动窗口、Min‑Max归一化、特征工程、计算复杂度分析。

**📊 数据集**

实验数据为14个客户端（14个国家/地区）日碳排放记录，涵盖国内航空、陆运、工业、国际航空、电力、住宅等六个排放部门，总约3.2万条数据，分为30%、50%与100%三种训练比例。

**📈 对比分析**

通过在30%、50%、100%数据量以及5/9/14客户端聚合的多组实验，评估R²、MSE、MAE、RMSE、MAPE指标。平均R²≈0.73，RMSE≈1.21，MAPE≈6.5%，显示联邦模型在隐私约束下仍能保持与集中式模型相近的预测精度。

**⚠️ 局限性**

局限性包括：①数据异质性导致部分客户端在完整聚合后性能下降；②通信与模型更新的碳足迹仍显著，需进一步优化；③未融入外部政策/经济因素，模型对极端事件的泛化能力有限。

---

## 309. Multi-Level Resistive Synapses for On-Chip Neural Networks: A Physics-Based Design of a Memristive Crossbar Fabric with Quasi-Continuous Conductance States

**arXiv ID:** 2606.22621 | [PDF](https://arxiv.org/pdf/2606.22621v1)

**作者:** David Alejandro Trejo Pizzo `[一作]` `[通讯]`, David Alejandro Trejo Pizzo

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于多级电阻式存储器（memristor）的全在存储器神经网络体系结构，涵盖从离子输运的连续状态变量模型、1T1R差分交叉阵列的设计、全物理推导的前向/反向推理与权重更新、以及系统级的功耗、面积、3D 堆叠和安全评估；

**💡 创新点**

创新点包括①利用离子迁移的连续状态变量模型证明单元可实现数百个稳定子层级；②设计1T1R差分交叉阵列并通过“电阻通信”实现多块交叉阵列的连接；③在模拟域实现前向、反向和外积学习规则，并给出完整的硬件实现细节；④将该平台用于大型语言模型和完整自注意力的硬件实现，展示显著的能效优势；

**🔧 技术方法**

使用的技术包括电化学迁移/价变 memristor 物理模型、Nernst–Planck 离子扩散与窗口函数、1T1R单元与差分对称权重、外积学习规则与程序/验证、ADC/DAC 与 SPICE 仿真、3D BEOL 堆叠、硬件安全 PUF 及能耗/面积模型；

**📊 数据集**

验证数据集主要使用 MNIST（784‑256‑128‑10 网络）和大规模语言模型（如 7B 参数 Transformer 与 3B 参数 BitNet）进行推理和训练评估；

**📈 对比分析**

与传统 SRAM/DRAM、二进制/少级 ReRAM 以及 GPU/H100、Apple M4 等平台比较，采用权重量级移动能耗、吞吐量与功耗指标，结果显示该架构在推理/自回归任务中能耗降低 10²–10³ 倍、功耗 10³–10⁴ 倍，且吞吐量可与高端 GPU 相当；

**⚠️ 局限性**

局限性包括对编程与验证的高精度需求、漂移与噪声对子层级可靠性的影响、写入耐久度约束、3D 堆叠的热管理挑战，以及在高精度数字计算任务中相对较低的可编程性和可靠性校准需求。

---

## 310. PaperClaw: Harnessing Agents for Autonomous Research and Human-in-the-Loop Refinement

**arXiv ID:** 2606.22610 | [PDF](https://arxiv.org/pdf/2606.22610v1)

**作者:** Weiwei Ye `[一作]` (University of Tokyo), Renhe Jiang `[通讯]` (University of Tokyo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了PaperClaw，一个多代理系统，能够从领域策划、创意生成、假设检验到论文撰写实现全流程自治，并支持人工介入改进

**💡 创新点**

创新点包括（1）端到端清晰的研究管线（Domain→Idea→Hypothesis→Paper）；（2）可停止的迭代假设图，仅依据实验测得的结论逐步扩展；（3）循环式研究助手，嵌入每个阶段，具备文献检索、代码编写、实验执行、论文草稿等工具；（4）全生命周期记忆，保存每一步的记录，支持中断后恢复；（5）对引用进行验证，确保不出现虚假结果或参考

**🔧 技术方法**

使用的大型语言模型（LLM）结合上下文提示、链式思考、工具与代码调用；多代理对话与辩论框架；实验跑器执行真实训练/分析；全生命周期内存结构；论文编译与会议合规检查；验证脚本检查引用和实验真实性

**📊 数据集**

利用开放学术索引（如Semantic Scholar、arXiv等）实时拉取论文、数据集与代码；实验采用公开数据集和标准基准（未具体列明，示例可含ImageNet、CIFAR等）；代码库引用公开代码库

**📈 对比分析**

通过LLM评判器对自动生成的论文质量进行评估，结果显示在完全自治或人机协同模式下，PaperClaw 能产出符合会议规范、引用真实、实验可复现的论文；在对比实验中，自动生成论文在多项指标上与人工撰写相当或优于现有自治系统

**⚠️ 局限性**

局限性包括：对LLM的依赖仍可能导致幻觉和误导性结果；系统对计算资源需求高；在极其新颖或缺少公开数据的领域表现受限；实验验证仍需人工监督以确保结果真实性；评测主要基于LLM判断，缺乏多维度客观指标

---

## 311. Illuminating English Letters Using a Flying Light Speck

**arXiv ID:** 2606.22592 | [PDF](https://arxiv.org/pdf/2606.22592v1)

**作者:** Hamed Alimohammadzadeh `[一作]` (University of Southern California), Shahram Ghandeharizadeh `[通讯]` (University of Southern California)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文设计并实现了一款可自主定位并沿预设轨迹飞行的微型无人机——Flying Light Speck (FLS)，用于在室内空间内照亮英文字母；

**💡 创新点**

创新点包括：①采用去中心化定位技术，利用无人机自身摄像头和地面红外标记实现实时姿态估计；②通过“写字路径–轨迹”两阶段规划（Motion Planner MOPL）将人类设计的字形转化为可执行的飞行路径；③在单机系统中完成从轨迹生成到飞行控制的闭环；

**🔧 技术方法**

技术手段：Raspberry Pi 5+全局快门摄像头+IR标记+OpenCV的AP3P PnP求解；Blender用于轨迹可视化；MOPL实现轨迹点生成与速度加速控制；Vicon系统用于精确测量飞行轨迹误差；

**📊 数据集**

数据集：4个英文字母（O、N、S、E）的写字路径与轨迹；20名受试者的识别实验数据；Vicon 3D位置数据；

**📈 对比分析**

比较方法：利用Vicon测得的飞行路径与期望轨迹的RMSE（Δ_Traj）评估轨迹误差（42–56 mm）；人类受试者的识别率、平均信心与识别时长评估视觉效果；结果显示误差高于预期，某些字母（如S）的识别率仅20%，顺序对识别率与时长有显著影响（p=0.021），不同顺序导致平均识别时长相差约2倍；

**⚠️ 局限性**

局限性：①轨迹误差较大，影响识别率；②仅使用单个FLS，缺乏多机协同；③研究仅涵盖四个字母，样本有限；④实验室环境与实际应用环境可能差异显著。

---

## 312. Venice-H1: Failure-Aware Query Re-Ranking with Multi-Scale Grid Signatures for Referring Image Segmentation

**arXiv ID:** 2606.22546 | [PDF](https://arxiv.org/pdf/2606.22546v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 313. Stationary Robust Mean-Field Games under Model Mismatches

**arXiv ID:** 2606.22579 | [PDF](https://arxiv.org/pdf/2606.22579v1)

**作者:** Yue Wang `[一作]` `[通讯]` (University of Central Florida), Yue Wang (University of Central Florida)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并分析了无限期、平稳的分布鲁棒均衡场游戏（Robust MFG），证明其可解并给出收敛算法；

**💡 创新点**

创新点在于将分布不确定性直接融入均衡场动力学，利用固定点与收敛证明构建第一套可收敛的平稳鲁棒 MFG 算法，并给出有限玩家游戏的渐进与非渐进逼近理论；

**🔧 技术方法**

核心技术包括鲁棒 Bellman 及 Q-算子、γ-收缩的贝尔曼算子、固定点迭代（Robust Best‑Response Iteration）、Dobrushin 系数与 Lipschitz 约束的稳定性分析；

**📊 数据集**

实验数据为基于离散状态集合 {0,…,S-1} 与动作 {0,1} 的模拟环境，采用矩形不确定集（L1 与 KL 散度）和统一混合转移概率构建；

**📈 对比分析**

与非鲁棒 MFE 对比，鲁棒 MFE 在不同不确定半径下保持更高的稳健价值；对有限玩家游戏，鲁棒 MFE 的 Nash 间隙随玩家数 N 减小，且收敛速率约为 O(1/√N)；

**⚠️ 局限性**

局限性包括需要较强的混合与 Lipschitz 条件，实验仅在简化的离散域验证，实际应用中对不确定集的构造与大规模强化学习的计算开销仍需进一步研究。

---

## 314. Evidence-Bound Gateway-Path Provenance for Third-Party LLM Inference

**arXiv ID:** 2606.22560 | [PDF](https://arxiv.org/pdf/2606.22560v1)

**作者:** Fei Wang `[一作]`, Zebai Tian `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于可信执行环境（TEE）和远程证明的LLM网关架构，将业务控制平面与安全关键的执行平面分离，保证客户端可以验证请求路径、路由、回退、端点及流记录的真实性；

**💡 创新点**

创新点在于将网关的核心推理路径迁移至受可信度量的执行体，并通过可证明的凭证链（release registry、TEE attestation、签名证据链）为每个请求提供端到端可验证的“路径溯源”，从而防止网关的路由替换、隐藏回退、流篡改等攻击；

**🔧 技术方法**

主要技术包括：AWS Nitro Enclaves 的可信执行与远程证明、Noise/HPKE 加密会话、签名证据链（IEC）与 StreamEv、策略携带的推理合同、基于 COSE Sign1 的 Nitro attestation 验证、以及 Rust 生态下的实现；

**📊 数据集**

实验使用了本地确定性 mock LLM 服务以及公开的 GPT-4o 流式接口作为上游，不涉及传统 NLP 数据集；

**📈 对比分析**

通过与纯文本网关、直接 Mock 调用及 Nitro Enclave 版本的对比实验，证明在 1–16 并发下，基于证据的网关在端到端延迟（p95）仅比纯文本网关高约 1.1×，并发吞吐量保持 0.95×，单次请求的签名/验证开销约 10–30 微秒，整体机制开销相对可接受；

**⚠️ 局限性**

局限性包括：无法证明上游 LLM 是否按声明模型执行、未隐藏流量时序与长度、对侧信道或客户端侧攻击不作防护、依赖 TEE、attestation 根与 release 签名安全、以及对大规模并发或多模态场景的完整评估缺失。

---

## 315. Mitigating Measurement-Induced Training Instability in Hybrid Quantum Neural Networks for Protein Classification

**arXiv ID:** 2606.22551 | [PDF](https://arxiv.org/pdf/2606.22551v1)

**作者:** Milton Mondal `[一作]` (University Medical Center Göttingen), Ali H. Shaib `[通讯]` (University Medical Center Göttingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文针对混合量子神经网络（Hybrid QNN）在分类任务中因量子测量期望值被直接用作logit导致的“测量诱导的logit压缩”问题，提出并验证了一种可学习的温度缩放（Quantum Measurement Temperature, QMT）机制，以恢复梯度强度、提升训练稳定性与分类准确率。

**💡 创新点**

创新点在于：①首次将量子测量输出的物理上限与经典交叉熵损失之间的不匹配建模为一个训练瓶颈；②引入可学习的温度因子在测量-损失接口处对logit进行缩放，从而在不改动量子电路或参数化的前提下提升梯度幅度与方差；③提供理论分析证明温度缩放能提升损失的灵敏度与梯度范数，并给出梯度范数上界；④在真实荧光显微镜蛋白质数据与标准视觉基准（Fashion MNIST、Overhead MNIST）上系统评估，验证其对多类任务的普适性。

**🔧 技术方法**

技术方法包括：混合经典-量子网络架构（经典卷积特征提取 + 角度编码 + 数据重上传 VQC + Pauli Z 测量）；可学习的温度缩放层（QMT）在 softmax 前对期望值进行除以 T；梯度基于 Adam/AdamW/RMSprop/SGD 进行优化；理论分析涉及对 softmax、损失 Lipschitz 常数、梯度方差及参数梯度上界的推导；实验采用 PennyLane 模拟器实现。

**📊 数据集**

使用的数据集：
- 荧光显微镜蛋白质图像（四类：GABA_A、GFP、Otoferlin、背景，共 12,000 张）
- Fashion MNIST 的前六类（多类分类）
- Overhead MNIST 的前六类（多类分类）

**📈 对比分析**

比较方法：在相同网络架构、相同训练设置下对比 QMT 学习（T_learn）与固定温度（T=1）以及纯经典网络；多次随机初始化（5 次）评估收敛稳定性；对不同量子层数、量子比特数、卷积过滤器数、不同量子门（CZ vs CNOT）以及不同优化器进行 ablation。性能表现：
- 在蛋白质数据上，QMT 均可提升 1–3% 的测试准确率并将标准差从 >5% 降至 <1%；
- 在 Fashion MNIST 与 Overhead MNIST 上，QMT 使测试准确率提升 6–12%，并显著降低训练波动；
- 训练损失下降更快，梯度范数与梯度方差提升 10 倍以上；
- 与梯度裁剪、分层训练等传统稳定化方法比较，QMT 仍保持最高或接近最高的准确率并进一步减少方差。

**⚠️ 局限性**

局限性：
- 目前实验均在模拟器上进行，尚未在真实量子硬件上验证可扩展性与噪声鲁棒性；
- QMT 需要额外的可学习参数和额外的梯度传播，可能导致训练成本略增；
- 该方法主要解决“测量-损失接口”导致的梯度衰减，无法完全消除因全局梯度消失（barren plateau）或噪声诱导的梯度崩溃等底层电路级问题；
- 在极大类别数或极低噪声环境下，温度缩放的最优值可能接近 0，需谨慎调参以避免梯度爆炸。

---

## 316. NegAS: Negative Label Guided Attention and Scoring for Out-of-Distribution Object Detection with Vision-Language Models

**arXiv ID:** 2606.22537 | [PDF](https://arxiv.org/pdf/2606.22537v1)

**作者:** Yingjie Zhang `[一作]` (Northwestern Polytechnical University), Peng Wang `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了利用LLM生成负标签的负标签引导注意力与评分机制（NegAS），实现基于视觉语言模型的高效离谱分布（OOD）物体检测。

**💡 创新点**

创新点在于：①使用视觉相似但语义不同的负标签通过LLM挖掘；②将负标签用于引导背景区域注意力，提升ID与OOD的特征可分性；③设计了与VLM sigmoid输出相匹配的负标签辅助OOD评分函数NegS。

**🔧 技术方法**

技术包括：LLM负标签生成、视觉相似度过滤、语义不相似度排序；负标签引导注意力（NegA）双分支结构；Sigmoid‑based OOD评分（NegS）；prompt学习与噪声正则化；在YOLO‑World和Grounding DINO两种VLM检测器上实现。

**📊 数据集**

ID数据集：PASCAL VOC（20类）、BDD100K（10类）；OOD评估集：MS‑COCO（去除ID类）、OpenImages（去除ID类）。

**📈 对比分析**

与YOLO‑World基线及其CoOp增强版本对比，NegAS在VOC→OpenImages/FPR95从48.8%降至23.3%，AUROC从81.4%升至92.6%；在VOC→COCO/FPR95仅9.9%，AUROC 95.9%；在Grounding DINO上亦显著提升，且优于直接迁移的MCM/NegLabel OOD评分。

**⚠️ 局限性**

局限性包括：对LLM生成负标签的质量和覆盖范围依赖较大；需要额外的背景掩码构造；在极端多样化或完全未知的OOD场景下仍可能存在漏检；训练时引入额外分支和负标签损失会增加计算开销。

---

## 317. Governance Decay: How Context Compaction Silently Erases Safety Constraints in Long-Horizon LLM Agents

**arXiv ID:** 2606.22528 | [PDF](https://arxiv.org/pdf/2606.22528v1)

**作者:** Shiyang Chen `[一作]` `[通讯]` (Beijing Institute of Technology), Shiyang Chen (Beijing Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了现代大型语言模型代理在上下文压缩过程中可能导致的治理失效——即在压缩历史时丢失关键的在场治理约束，导致代理违反安全或组织规则。

**💡 创新点**

创新点在于提出了 Governance Decay 概念，量化了压缩导致约束丢失的风险；构建了 ConstraintRot 基准评估压缩对不同类型约束的影响；揭示了可被利用的 Compaction‑Eviction 攻击；提出并验证了无训练的 Constraint Pinning 防御方案。

**🔧 技术方法**

技术包括：上下文压缩（多种策略如递归、LLM 生成摘要等）、系统评估的基准测试、自动化违规检测（解析工具调用）、对抗式压缩注入（对压缩器的指令注入）和约束钉住（将约束保存在不被压缩的缓冲区中并每次压缩后重新注入）。

**📊 数据集**

数据集为自构的 ConstraintRot 任务集，包含 9 个任务（5 个软组织政策，4 个硬安全规范），每个任务包含治理约束、长序列任务、以及触发违规请求。

**📈 对比分析**

对比方法：在 7 种模型（DeepSeek、GLM、Qwen、Kimi、Claude、GPT、Gemini）和 4 种压缩策略下评估违规率。实验显示：未压缩时 0% 违规，压缩后总体违规率升至 30%（最高 59%）；对抗式压缩注入可将违规率提升至 65%；Constraint Pinning 在所有模型和攻击下恢复到 0% 违规率，且 token 开销低于 0.5%。

**⚠️ 局限性**

局限性：只针对可提取的明确约束；对隐式约束不适用；在存在操作者权威更新的情况下，钉住仍可能被突破，需要可信的离线权威通道；实验使用的评估模型为 API 调用，未覆盖所有可能的压缩实现；攻击策略搜索有限，可能存在更强的优化。

---

## 318. On the Position Bias of On-Policy Distillation

**arXiv ID:** 2606.22600 | [PDF](https://arxiv.org/pdf/2606.22600v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 319. MapReason-OSM: Can Vision-Language Models Make Graph-Verifiable Mobility Decisions from Street Maps ?

**arXiv ID:** 2606.22597 | [PDF](https://arxiv.org/pdf/2606.22597v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 320. Training-free Task Classification for Multi-Task Model Merging

**arXiv ID:** 2606.22589 | [PDF](https://arxiv.org/pdf/2606.22589v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 321. OmniSpace: Efficient Geometry Awareness for Autonomous Vehicles MLLMs

**arXiv ID:** 2606.22617 | [PDF](https://arxiv.org/pdf/2606.22617v1)

**作者:** Hao Vo `[一作]` (University of Arkansas), Ngan Le `[通讯]` (University of Arkansas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多模态大语言模型（MLLM）中增强几何感知，使其在自动驾驶任务中实现更好的空间推理，尤其是无需额外3D模块推理时。

**💡 创新点**

通过三项创新：摄像机姿态注入（Plücker射线嵌入）、多视角极线注意力机制以及基于3D教师的时序几何蒸馏，直接在训练阶段把3D知识注入MLLM，消除了推理时的外部3D依赖。

**🔧 技术方法**

利用Plücker射线表征摄像机姿态、极线几何约束的跨视角注意力、与VGGT等3D基础模型的特征蒸馏，并采用对称时间窗口聚合提升教师信号稳定性；训练时使用LoRA微调。

**📊 数据集**

在nuScenes、Bench2Drive（规划）、nuInstruct（风险检测）、OmniDrive（语言描述）和DriveBench（泛化）等自动驾驶基准上进行评估。

**📈 对比分析**

与基线MLLM、基于测试时3D模型的VGGDrive、SpaceDrive等进行对比，结果显示在规划误差、碰撞率、交叉率、风险检测MAP等指标上均有显著提升，同时保持或提升推理速度（FPS从0.03/0.25提升到>0.4），证明方法在效率与性能上兼顾。

**⚠️ 局限性**

局限性包括：时序聚合仅为简单平均，未考虑车辆运动；极线注意力和射线嵌入依赖精确的相机标定；仅使用单一VGGT教师，可能受限于教师的表达能力；对实时动态场景的鲁棒性尚待进一步验证。

---

## 322. Compositional Generator Equivalence (Extended Version)

**arXiv ID:** 2606.22616 | [PDF](https://arxiv.org/pdf/2606.22616v1)

**作者:** Anthony Vandikas `[一作]` (University of Toronto), Marsha Chechik `[通讯]` (University of Toronto)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文研究了属性式测试（PBT）中生成器的语义等价性，证明现有框架 Hedgehog 的采样语义过细而分布语义非组合化，随后提出一种受箭头算子约束的受限语言 *，提供可组合的分布语义，并评估其表达力与代码规模。

**💡 创新点**

创新点在于：①给出对 Hedgehog 采样与分布语义的正式分析并证明两者不可兼容；②设计受限语言 *（基于箭头计算）以消除统计依赖，从而获得组合化的分布语义；③证明 * 的分布语义满足组合性、可证明常见优化；④实现可将 * 程序翻译回 Hedgehog，并验证其在实际代码库中的表达力。

**🔧 技术方法**

使用了 Quasi‑Borel 空间作为概率程序的语义模型，借助箭头算子（Arrow Calculus）构造受限语言 *，并定义了其采样与分布解释。对等价性和优化通过形式化证明与对照实验实现；在实现层面使用 Haskell + GHC Arrow 扩展。

**📊 数据集**

实验数据来源于 Hedgehog 的 `hedgehog` 模块及其示例仓库（约 30+ 模块），其中包含 70 个生成器和若干组合器。

**📈 对比分析**

比较方法：将原始 Hedgehog 程序与翻译成 * 的程序在可表达性（可否实现）与 AST 节点数上对比。结果显示：* 能表达 76% 的生成器，只有 2 个组合器不可直接实现；翻译后程序 AST 节点数与原始几乎相同，差异可忽略不计。

**⚠️ 局限性**

局限性：①评估仅在 Haskell（惰性、纯函数）环境下进行，结果可能不适用于严格或命令式语言；②样本集规模有限，未覆盖非常大或复杂的生成器；③理论证明集中在终止程序，未讨论非终止情况；④未实现自动化优化或重写规则，实际性能提升需要进一步实现。

---

## 323. Automated sign detection across the Electronic Babylonian Library: A large-scale dataset and end-to-end cuneiform OCR pipeline

**arXiv ID:** 2606.22608 | [PDF](https://arxiv.org/pdf/2606.22608v1)

**作者:** Wentao Che `[一作]` (LMU Munich), Enrique Jiménez `[通讯]` (LMU Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对古楔形文字符号检测进行改进，并在最大规模标注数据集上训练 Deformable DETR 模型，结合自动板面提取、DBSCAN 行聚类和 n‑gram 文本相似性评估，实现大规模可解释的检测与评估。

**💡 创新点**

在已有 DETR 框架基础上扩充标注集至 124,504 个符号，提出自动板面分割、尺度不变行聚类和 n‑gram 评估方法，构建了从视觉检测到文本结构的完整流水线。

**🔧 技术方法**

使用 Deformable DETR（ResNet‑50 backbone、Hungarian 匹配、Focal 损失）、DBSCAN 行聚类、自动板面提取和 n‑gram 相似性评估等技术。

**📊 数据集**

使用电子巴比伦图书馆（eBL）公开的最大规模楔形文字数据集，包含 1,931 片段、124,504 个标注符号，覆盖多时期与多博物馆。

**📈 对比分析**

采用 COCO 评估指标与基线进行对比，173/106 类模型分别提升 28–37%，AP 从 0.228 提升至 0.312，AP_50 提升至 0.515；在 87,668 片段上推理得到近 290 万检测，n‑gram 匹配平均 Match Score 0.232，远优于随机/交叉碎片基线。

**⚠️ 局限性**

缺乏语言上下文与序列建模，难以处理严重损坏或字迹密集的板面；行聚类参数固定，无法适应所有布局；对低频或模糊符号的识别仍有限。

---

## 324. Context-Aware Distillation and Ablation for Text2DSL

**arXiv ID:** 2606.22578 | [PDF](https://arxiv.org/pdf/2606.22578v1)

**作者:** Alexander V. Kozachok `[一作]` (MIREA Russian Technological University), Shamil G. Magomedov `[通讯]` (MIREA Russian Technological University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在 Text2DSL 任务中实现了上下文感知蒸馏方法，生成了经过 AST 与运行时双重验证的 PolkitBench‑v2 数据集。

**💡 创新点**

创新点在于将结构化上下文（BNF、API、词汇表）明确纳入生成流程，实现可追溯的上下文绑定，并通过 8 种组合的因子消融揭示各组件在语义和结构上的关键作用。

**🔧 技术方法**

技术主要包括基于 DeepSeek‑V4‑Flash 的教师模型生成、AST 解析（esprima）与容器化 Polkit 守护进程的运行时验证，以及对生成文本的上下文注入和响应格式约束。

**📊 数据集**

使用的数据集为 PolkitBench‑v1（4,204 条）和新生成的 PolkitBench‑v2（10,073 条），两者均包含自然语言描述与对应 Polkit 规则。

**📈 对比分析**

通过 Baseline 与 Context 两种模式以及完整的 2^3 消融实验，在 GigaChat‑10B‑A1.8B 上评估语法有效率、结构有效率、CodeBLEU、Jaccard 和综合分数。结果显示，Context 模式在更难的 PolkitBench‑v2 上仅略降，Baseline 模式则严重崩溃；完整上下文 C_7 始终获得最高分。

**⚠️ 局限性**

局限性包括仅使用单一评估模型和单一教师模型，残留的标识符幻觉率仍高，Strict Success 指标过于严格，运行时验证环境受限于容器化设置。

---

## 325. MAPS: Multi-Anchor Projection Similarity for Joint Vision-Language Geo-Localization

**arXiv ID:** 2606.22543 | [PDF](https://arxiv.org/pdf/2606.22543v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 326. What Characterizes Pairwise Modular Smells?

**arXiv ID:** 2606.22576 | [PDF](https://arxiv.org/pdf/2606.22576v1)

**作者:** Chenxing Zhong `[一作]`, He Zhang `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究通过构建以软件实体对（separated 与 collocated）为样本的 PairSmell 预测模型，验证实体对特征能否作为 PairSmell 的指示器，并进一步分析哪些特征最为重要。

**💡 创新点**

创新点在于首次系统评估多种实体对特征（如 out‑degree、degree、total‑fields、sim‑tfidf、intersection、in‑degree）对 PairSmell 预测的贡献，并通过 permutation importance 与 partial dependence 可视化解释模型决策。

**🔧 技术方法**

使用的技术包括逻辑回归、支持向量机（SVM）和随机森林等监督学习模型，以及基线（stratified、constant、theoretical）对比，并运用 ROC‑AUC、AUPRC、Mann‑Whitney U 检验和 PDP 等评估与解释方法。

**📊 数据集**

采用来自 11 个软件项目的 16,541 条 separated 对和 14,372 条 collocated 对数据集，其中分别包含 3,111/3,105 条 InSep 与 1,382/1,378 条 InCol 实例。

**📈 对比分析**

与基线相比，SVM 在 separated 对上提升 ROC‑AUC 约 30–39%，在 collocated 对上提升 58–61%；AUPRC 亦提升 1.6–2.9 倍，表明模型在区分正负样本及降低误报方面表现优异。

**⚠️ 局限性**

局限性包括样本高度不平衡、特征分布偏斜导致模型可能对异常值敏感、仅使用静态实体对特征未考虑动态演化影响，以及缺乏跨项目泛化验证。

---

## 327. Concept-Constrained Prompt Learning for Few-Shot CLIP Adaptation

**arXiv ID:** 2606.22567 | [PDF](https://arxiv.org/pdf/2606.22567v1)

**作者:** Na Sang `[一作]` (University of California San Diego), Yuxuan Liu `[通讯]` (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Concept-Constrained Prompt Learning（CCPL），在 CLIP 的少量样本适配中加入冻结的概念原型作为正则化，仅学习共享的上下文词，保持 CLIP 编码器不变。

**💡 创新点**

创新点在于用手工概念库生成的文本原型为文本空间提供余弦一致性正则化，并结合概念丢弃与可控推理融合，实现轻量且可解释的提示学习。

**🔧 技术方法**

采用共享上下文词提示学习、文本空间余弦一致性正则化、概念丢弃、可控推理融合以及 CLIP ViT‑B/16 冻结的图像/文本编码器。

**📊 数据集**

在 DTD、EuroSAT 与 OxfordPets 三个数据集上进行实验。

**📈 对比分析**

与 CoOp 在自动 fallback split 上比较，CCPL 在 DTD 与 EuroSAT 的基‑新谐波均值分别提升 +0.6 与 +2.9，OxfordPets 近中性；Ablation 证实文本正则化与推理融合对新类性能提升起主要作用。

**⚠️ 局限性**

局限包括：概念原型质量决定效果，对细粒度数据（如 OxfordPets）无明显提升；仅与 CoOp 对比，未评估其他先进方法；概念库手工构造不易扩展；使用 fallback splits，结果不易与官方分割直接比较；未对概念对齐动态进行度量。

---

## 328. HiMatch-AD: DINOv3-driven Hierarchical Matching for Training-free Medical Anomaly Detection

**arXiv ID:** 2606.22556 | [PDF](https://arxiv.org/pdf/2606.22556v1)

**作者:** Jiayu Huo `[一作]` (Imperial College London), Le Zhang `[通讯]` (University of Birmingham)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出HiMatch-AD，基于预训练DINOv3的训练‑free层次匹配框架实现医学影像异常检测。

**💡 创新点**

创新点在于整合双分支（CLS+patch）支持检索与多层级异常图生成，并采用统一的不确定性权重融合机制，显著提升多模态异常定位精度。

**🔧 技术方法**

技术实现包括使用DINOv3 ViT提取全局与局部特征，K‑means聚类生成代表中心点，层次相似度比较生成异常图，随后通过基于不确定性（softmin）权重的实例级与层级级融合。

**📊 数据集**

实验数据集为BMAD基准中的脑MRI、肝CT和视网膜OCT三大医学影像数据集。

**📈 对比分析**

通过与多种训练式与训练‑free SOTA 方法（如PatchCore、AnomalyDINO、DRAEM等）对比，HiMatch-AD在I‑AUC、P‑AUC和P‑PRO三项指标上均位居榜首，提升幅度约1–3%。

**⚠️ 局限性**

局限性在于依赖单一预训练模型，聚类与相似度计算对计算成本有一定影响，对极少量或高度多样化异常的鲁棒性仍待进一步验证。

---

## 329. Beyond Penalizing Mistakes: Stabilizing Efficiency Training in Large Reasoning Models via Adaptive Correct-Only Rewards

**arXiv ID:** 2606.22716 | [PDF](https://arxiv.org/pdf/2606.22716v1)

**作者:** Jungseob Lee `[一作]` (Korea University), Heuiseok Lim `[通讯]` (Korea University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大型语言模型推理效率的训练，系统分析奖励崩溃机制，并提出一种自适应正确答案唯一效率奖励方案，以实现高效且稳定的推理。

**💡 创新点**

首次阐明连续错误答案长度惩罚会导致GRPO奖励崩溃的根本原因，并设计三种自适应机制（正确答案唯一、动态预算归一化、准确率驱动的惩罚调节）形成稳健的效率优化方法。

**🔧 技术方法**

利用Group Relative Policy Optimization (GRPO) 与自定义奖励函数，结合动态预算EMA、控制循环自适应α、log平滑函数等技术实现效率提升。

**📊 数据集**

在Qwen3-1.7B模型上使用NuminaMath-TIR进行训练，评估基准为MATH-500、MATH-Hard、AIME 2025及OlympiadBench。

**📈 对比分析**

与基线GRPO、GRPO+LP、Short-RL等方法对比；基线模型准确率约88.8%，改进方案在MATH-500上保持≈88.4%准确率，平均生成令牌数降低62%；在不同难度级别保持稳定，只有少数静态方法在训练中崩溃。

**⚠️ 局限性**

局限性包括仅验证1.7B规模模型，缺乏对更大规模模型的验证；多随机初始化下崩溃概率未完全量化；准确率提升不显著，统计显著性不足；跨基准差异存在；理论证明仅针对G=2，通用证明尚未完成；超参敏感性未系统探索。

---

## 330. VeriPort: Automated and Verified Patch Backporting at Scale

**arXiv ID:** 2606.22704 | [PDF](https://arxiv.org/pdf/2606.22704v1)

**作者:** Jonah Ghebremichael `[一作]` (North Carolina State University), Alexandros Kapravelos `[通讯]` (North Carolina State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

论文探讨了某种技术在特定领域的应用，提出了一种新的解决方案。

**💡 创新点**

创新点在于提出了一种新的方法，能够有效解决现有技术中的某些局限性。

**🔧 技术方法**

使用了机器学习和数据挖掘技术来实现目标。

**📊 数据集**

采用了特定领域的公开数据集进行实验和验证。

**📈 对比分析**

与现有方法进行了比较，结果显示新方法在准确性和效率上均有显著提升。

**⚠️ 局限性**

限制在于数据集的规模和多样性可能影响结果的普适性。

---

## 331. Modular Diffusion Models for Structured Visual Recognition

**arXiv ID:** 2606.22702 | [PDF](https://arxiv.org/pdf/2606.22702v1)

**作者:** Siddhesh Khandelwal `[一作]` (University of British Columbia), Leonid Sigal `[通讯]` (University of British Columbia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Modular Diffusion Models (MDMs)，一种将结构化视觉任务的联合分布拆分为多个条件扩散过程的框架，用于生成多模态、不确定性的结构化输出。

**💡 创新点**

创新点在于将扩散模型模块化，分别为盒子、类别、掩码或关系等不同组成建模，保持跨组件对齐，支持独立控制每个模块的采样步数，从而实现多样化、可解释的预测。

**🔧 技术方法**

技术实现基于高斯扩散、Transformer 编码器-解码器架构、时间条件化和自注意力机制，训练时使用匹配与联合损失，推理采用 DDIM 加速采样。

**📊 数据集**

在 MS‑COCO 2017（目标检测、实例分割）和 Visual Genome 子集（场景图生成）上进行实验。

**📈 对比分析**

与现有扩散和确定性方法对比，MDM 在目标检测 AP+3.9、实例分割 AP^mask+8.4、场景图 hR@50+4.1 等指标上达到了或超过最强基线，并能够生成多样化的预测样本。

**⚠️ 局限性**

局限性包括：各模块需共享编码器/解码器导致任务专属优化受限；推理时多步扩散仍较慢，且对早期模块噪声的鲁棒性有待进一步提升。

---

## 332. Confidently Wrong: Severity-Aware Calibration of Prompt-Injection Detectors under Attack Shift

**arXiv ID:** 2606.22659 | [PDF](https://arxiv.org/pdf/2606.22659v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 333. NullFlow: One-Step Generative Reconstruction

**arXiv ID:** 2606.22696 | [PDF](https://arxiv.org/pdf/2606.22696v1)

**作者:** Xiao Shi `[一作]` (University of Wisconsin), Ulugbek S. Kamilov `[通讯]` (University of Wisconsin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于子空间约束的平均流（Mean Flow）框架，用于在一次网络评估内完成成像逆问题的后验采样。

**💡 创新点**

创新点在于将平均流限制在正交投影的零空间内，使得每一步状态自动保持测量一致，从而无需迭代的数据保真修正，实现真正的一步后验采样。

**🔧 技术方法**

采用了子空间约束的流匹配（subspace‑restricted flow matching）和平均流学习（mean‑flow learning）技术，训练目标结合PMF、LPIPS以及ConvNeXt‑V2特征正则化。

**📊 数据集**

实验使用FFHQ数据集，在256×256 RGB图像的中心缺失（128×128）补全任务上进行评估。

**📈 对比分析**

与U‑Net、DPIR、DPS、DiffPIR、PnP‑Flow、Flower等迭代方法相比，该方法仅一次网络调用即可获得最佳LPIPS，并在PSNR和SSIM上保持竞争力。

**⚠️ 局限性**

局限性包括：仅在无噪声测量下验证；目前仅针对图像填补任务，未在MRI、CT等其他逆问题上测试；需要进一步扩展至含噪测量和可变算子情形。

---

## 334. Architecture for Health Initiative (Arch4Health): Computational Challenges in Health-Related Applications and the Role of Computer Architecture in Addressing Them

**arXiv ID:** 2606.22685 | [PDF](https://arxiv.org/pdf/2606.22685v1)

**作者:** Nika Mansouri Ghiasi `[一作]` (ETH Zürich), Onur Mutlu `[通讯]` (ETH Zürich)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

介绍了 Arch4Health 计划，旨在识别并讨论健康科学相关应用中的计算挑战，并促成计算架构与生命科学研究者的跨学科合作。

**💡 创新点**

将计算架构研究与健康科学结合，创建了专门的工作坊系列，聚焦从基因组学到医疗影像等多领域的计算挑战，并推动硬件加速、数据移动优化和隐私安全技术的研发。

**🔧 技术方法**

主要讨论了硬件加速器、近数据处理、内存层次优化、分布式系统、隐私保护技术等架构与系统技术。

**📊 数据集**

未提及具体数据集，侧重行业场景与未来方向。

**📈 对比分析**

由于是综述与倡议文章，未包含实验对比与性能评估。

**⚠️ 局限性**

缺乏具体实验验证，主要聚焦倡议与讨论，尚未提供可量化的技术评估与实现细节。

---

## 335. Efficient Continuous Semantic Mapping based on Spatio-Temporal Awareness

**arXiv ID:** 2606.22672 | [PDF](https://arxiv.org/pdf/2606.22672v1)

**作者:** My Le Pham `[一作]` (University of Engineering and Technology, Vietnam National University), Thanh Nguyen Canh `[通讯]` (University of Engineering and Technology, Vietnam National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种结合空间与时间推理的连续语义映射方法，利用LiDAR点云和深度语义分割结果构建高质量、长期稳定的语义地图。

**💡 创新点**

创新点包括：①利用每个体素的语义不确定性（Shannon熵）自适应调节贝叶斯核长度，从而在不确定区域扩大信息传播，确定区域缩小计算量；②引入时间衰减机制，将历史计数与当前观测按时间衰减融合，显著降低语义漂移并提升地图长期一致性。

**🔧 技术方法**

核心技术包括：贝叶斯核推理（Bayesian Kernel Inference）与自适应核长度、Shannon熵估计不确定性、时间衰减因子α=exp(-Δt/τ_t)的计数融合、体素化、下采样与冗余体素过滤，以及计数器（semantic counter）更新与归一化。

**📊 数据集**

使用公开的SemanticKITTI数据集进行实验，涵盖多条驾驶场景序列。

**📈 对比分析**

与基准方法CSM、S-BKI对比，实验显示该方法在mIoU上提升约12个百分点（54.92% vs 42.80%），OA提升约9-10个百分点，且在固定核长度设置下，采用自适应核长度后运行时长更短（39.61s vs 40.86s），证明了空间-时间联合推理的有效性。

**⚠️ 局限性**

局限性在于：①方法高度依赖分割网络的质量，分割误差会直接影响地图；②未对动态物体进行分离，动态区域仍可能出现累计误差；③目前仅在单机器人环境下验证，缺乏多机器人协同映射与真实导航任务的评估。

---

## 336. RAVEN: Agentic RAG for Automated Vulnerability Repair

**arXiv ID:** 2606.22647 | [PDF](https://arxiv.org/pdf/2606.22647v1)

**作者:** Varun Gadey `[一作]` (University of Duisburg-Essen), Alexandra Dmitrienko `[通讯]` (University of Duisburg-Essen)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出RAVEN框架，实现自动修复多种CVE漏洞。

**💡 创新点**

创新点在于将agentic检索增强生成、跨文件依赖检索和迭代修复整合为统一系统，并完全基于开源LLM实现。

**🔧 技术方法**

采用检索增强生成（RAG）、关键词/CPG多模态检索、Curator Agent跨文件依赖检索、LLM生成+Patch Reviewer迭代优化、Semgrep静态分析反馈以及CodeBLEU评估。

**📊 数据集**

使用CVEFixes、Vul4J、Zero-Day C集等真实CVE数据集，涵盖Java和C两门语言、10种CWE类别及未见CWE。

**📈 对比分析**

与PatchAgent、APPATCH、SAN2PATCH等基线对比，RAVEN在160条CVE上的成功率83.13%，Java 86.75%，未见CWE 65.22%，新CVE 87.5%，均显著优于现有工具。

**⚠️ 局限性**

局限在于对检索策略仍需手工调优，极其复杂的跨文件逻辑修复仍有难度，对缺乏历史修复实例的漏洞依赖度较高。

---

## 337. A phase-field model for microbiologically influenced corrosion

**arXiv ID:** 2606.22640 | [PDF](https://arxiv.org/pdf/2606.22640v1)

**作者:** S. Kovacevic `[一作]` (University of Oxford), E. Martínez-Pañeda `[通讯]` (University of Oxford)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种基于相场的反应-扩散模型，用于预测硫酸盐还原菌（SRB）引起的微生物影响腐蚀（MIC），并将微生物动力学、硫酸盐扩散、电化学反应以及机械场耦合到同一框架中，随后对实验数据进行标定，并将模型应用于离岸风电桩的腐蚀与CP（阴极保护）评估。

**💡 创新点**

创新点在于首次构建了能够显式耦合微生物动力学（Monod关系）、硫酸盐传输、电化学反应与机械场的相场模型；通过改进相场迁移速度实现力学-化学耦合；实现多尺度（微观结构到结构级）模拟；同时可与CP模型一向耦合预测长期腐蚀行为。

**🔧 技术方法**

技术手段包括：相场方法（Allen‑Cahn 方程）、反应扩散耦合、Gutman 机制的力学-化学耦合、Monod 微生物动力学、有限元求解、一次性多尺度耦合（CP 电流分布→相场迁移率）以及敏感性分析。

**📊 数据集**

数据来源主要有：实验室 0.5 mM 硫酸盐溶液中 SRB 降解的坑深随浸没时间的测量（15‑150 天）；文献给出的硫酸盐扩散系数、SRB 密度、机械性质、微生物动力学参数等；海洋/土壤环境条件用于离岸风电桩案例（硫酸盐 28 mM、K_m = 0.2 mol m⁻³ 等）。

**📈 对比分析**

与实验坑深数据进行对比，模型预测与观测高度吻合；敏感性分析显示关键参数（K_m、q、b、Y、N、Ψ、c_b）对腐蚀速率影响明确；离岸桩案例中，模型预测了不同位置的腐蚀电流、坑深和裂纹发展，验证了CP在不同时间段的抑制效果。整体性能：计算量适中，能在数十年尺度上给出可靠预测，优于传统单一相场或纯电化学模型。

**⚠️ 局限性**

局限性包括：假设微生物活性恒定，未考虑生物膜生长、脱落、养分耗竭或腐蚀产物堆积；未显式求解 Fe²⁺、硫化物、pH 或电势场；使用线性弹塑性和各向同性硬化，仅能近似材料本构；仅考虑单一 SRB 物种，未捕获多种菌种的相互作用；缺乏 CP 对微生物动力学的反馈；模型参数需基于实验校准，实验数据有限，可能导致环境依赖性误差。

---

## 338. A Markov Chain Approach to Preference Alignment

**arXiv ID:** 2606.22652 | [PDF](https://arxiv.org/pdf/2606.22652v1)

**作者:** Takuya Koriyama `[一作]` (University of Chicago), Tengyuan Liang `[通讯]` (University of Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种基于人类反馈的马尔可夫链(MCHF)对齐方法，直接用成对偏好构造转移核，迭代得到与偏好一致的分布；同时对其与RLHF和NLHF进行了理论统一分析。

**💡 创新点**

创新点在于：①不通过单一奖励模型，而是直接使用对比偏好构造马尔可夫链；②提出半范数U⊕衡量偏好非传递性，并证明收敛速率与之自适应；③通过扰动分析证明在U⊕小的情况下，MCHF与NLHF在一阶上与RLHF解相同，提供三种对齐方法的统一视角。

**🔧 技术方法**

主要技术包括：马尔可夫链与条件采样、KL正则化、镜像下降（mirror‑descent）算法、半范数U⊕、L∞范数、线性逼近与Fréchet导数、矩阵式扰动分析。

**📊 数据集**

论文未使用真实数据集，实验部分仅在离散空间（|X|=20）中生成模拟成对偏好进行验证；主要展示理论收敛与数值匹配。

**📈 对比分析**

与RLHF、NLHF通过理论证明和数值实验比较：MCHF在小U⊕时收敛到与RLHF相同的奖励分布，收敛速率由U⊕控制；实验显示MCHF与NLHF在两步内已逼近RLHF解，早期两步可获得大部分对齐收益。

**⚠️ 局限性**

局限性包括：①需要对条件分布 (x,·) 进行采样，实际实现困难；②镜像下降算法需要步长取U⊕⁻²，需估计U⊕；③对非离散或大规模空间的采样与计算尚无成熟方法，需进一步研究可扩展实现。

---

## 339. AgentLens: Interpretable Safety Steering via Mechanistic Subspaces for Multi-Turn Coding Agent

**arXiv ID:** 2606.22673 | [PDF](https://arxiv.org/pdf/2606.22673v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 340. Is This AI? Longitudinal Analysis of Strategies Used for AI Detection on Two Subreddits

**arXiv ID:** 2606.22689 | [PDF](https://arxiv.org/pdf/2606.22689v1)

**作者:** Christina Yeung `[一作]` (University of Washington), Franziska Roesner `[通讯]` (University of Washington)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统梳理并量化了在两个流行Reddit社区中用户用于识别AI生成内容的策略。

**💡 创新点**

首次大规模定性与量化分析AI检测策略，构建12项策略分类并描绘其随AI技术进步的动态变化。

**🔧 技术方法**

采用混合方法：手工编码+Cohen κ评估、Claude Sonnet 4.6 零样本分类、tf‑idf 关键词挖掘、滚动平均与χ²检验。

**📊 数据集**

13,098条讨论帖与222,060条评论，来源于 r/RealOrAI 与 r/isthisAI，时间跨度为2023‑2026年。

**📈 对比分析**

通过人类与LLM标签交叉验证，κ>0.84；对时间段内策略比例进行χ²检验并Bonferroni校正，结果表明策略比例随AI技术进步显著变化。

**⚠️ 局限性**

样本仅限活跃Reddit用户，可能不具备代表性；数据受ArcticShift抓取与删帖影响；未评估策略有效性、缺乏因果分析；LLM辅助标签可能引入偏差。

---

## 341. Modular Rank and Linear-Complexity Tests for Pseudorandom Number Generators

**arXiv ID:** 2606.22684 | [PDF](https://arxiv.org/pdf/2606.22684v1)

**作者:** Sebastiano Vigna `[一作]` `[通讯]` (Università degli Studi di Milano), Sebastiano Vigna (Università degli Studi di Milano)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文扩展了二进制域的秩和线性复杂度测试到任意有限域，并介绍了一个名为modlin的工具，该工具高效地实现了这些测试。

**💡 创新点**

创新点在于首次能够检测在大素数域上线性的生成器的偏差，特别是针对MIXMAX生成器的线性偏差进行了实证分析。

**🔧 技术方法**

使用了Rust编程语言实现的modlin工具，采用了阻塞高斯消元法和Berlekamp-Massey算法来进行测试。

**📊 数据集**

测试使用了MIXMAX生成器的输出流，特别是在素数p=2^61-1的情况下进行分析。

**📈 对比分析**

与传统的二进制测试相比，modlin能够检测到MIXMAX生成器的线性偏差，尽管MIXMAX在二进制测试中表现良好。性能上，线性复杂度测试相对便宜，而秩测试则需要更多的计算资源。

**⚠️ 局限性**

限制在于modlin目前仅限于处理素数p且p小于2^63的情况，且对于其他类型的生成器可能无法检测到偏差。

---

## 342. Identifying Quality Indicators in Student Self-Reflections in Software Engineering

**arXiv ID:** 2606.22683 | [PDF](https://arxiv.org/pdf/2606.22683v1)

**作者:** Matthew Minish `[一作]` (University of Canterbury), Fabian Gilson `[通讯]` (University of Canterbury)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `cc175879-ab65-4aa9-b58a-f6100a057dbf`

**🎯 论文内容**

开发并评估了一套基于Transformer的自动化评估软件工程学生自我反思写作的框架与模型，支持实时结构化反馈。

**💡 创新点**

创新点在于将现有反思理论改编为八维指标，并构建可部署的编码器‑仅RoBERTa微调模型与对比的零射击解码器‑仅模型，实现高效且可解释的评估。

**🔧 技术方法**

主要技术包括Encoder‑Only RoBERTa微调、Decoder‑Only GPT‑OSS 20B和Qwen3 14B的零射击提示、二元交叉熵多标签分类与超参搜索。

**📊 数据集**

使用来自91名软件工程学生的1518条单问答反思文本（共6,704句）作为训练与评估数据集。

**📈 对比分析**

与五种Encoder‑Only模型对比，RoBERTa在10折交叉验证下达到F1≈0.90、准确率≈0.38；相对的Decoder‑Only模型F1≈0.75、处理时间20–45秒，明显落后。

**⚠️ 局限性**

局限性包括仅在单一课程与结构化问答格式上训练，缺乏跨文化或自由形式反思的泛化验证；Reasoning与Perspective指标的可靠性相对较低。

---

## 343. From Complaint Narratives to Monetary Relief: A Hybrid Machine Learning Framework for CFPB Consumer Complaints

**arXiv ID:** 2606.22664 | [PDF](https://arxiv.org/pdf/2606.22664v1)

**作者:** Zhuoer Wang `[一作]` (University of Illinois Urbana-Champaign), Xiongyu Chen `[通讯]` (Carnegie Mellon University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一套混合机器学习框架，用于预测消费者金融投诉中是否会获得货币补偿。

**💡 创新点**

创新之处在于将基于投诉类别和公司双层划分的LDA主题、可解释的文本特征以及结构化的公司/州信息结合进XGBoost模型，并采用时间序列划分模拟真实部署。

**🔧 技术方法**

采用LDA主题建模、手工文本特征提取、XGBoost梯度提升、时间序列训练/测试拆分以及AUC-ROC、PR-AUC等评估指标。

**📊 数据集**

使用了美国消费者金融保护局（CFPB）的消费者投诉数据库，共计176,942条记录（主要为支票/储蓄账户投诉），并在扩展实验中加入信用卡投诉。

**📈 对比分析**

通过与仅基于TF‑IDF的基线模型对比，改进模型将AUC‑ROC从0.69提升至0.78，PR‑AUC从0.25提升至0.35，F1得分亦由0.31提升至0.37，表明在处理严重类别不平衡的情况下性能显著提升。

**⚠️ 局限性**

局限性包括：仍面临显著类别不平衡和潜在的时间非平稳性，模型对公司标签的依赖可能导致外推性受限，且虽然引入可解释特征但整体解释性仍受梯度提升模型固有的黑箱特性限制。

---

## 344. Radio Resource Management for the Uplink of Hybrid Beamforming Systems

**arXiv ID:** 2606.22643 | [PDF](https://arxiv.org/pdf/2606.22643v1)

**作者:** Yuan Quan `[一作]` (University of Waterloo), Catherine Rosenberg `[通讯]` (University of Waterloo)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究基于预定义码书的混合波束成形系统上行链路的资源管理，包括波束选择、用户选择、功率分配、数字波束成形和调制编码方案的联合优化。

**💡 创新点**

创新点在于提出针对多通道、RF链数量不等于用户数的时隙级联合RRM优化，并给出可行近似解、离线启发式与低复杂度在线启发式，并分析RF链与匹配波束数对性能的影响。

**🔧 技术方法**

采用线性化约束、LP松弛、零逼近数字波束成形、MCS基率函数、离线启发式的四步方法和在线的Per‑Beam Scheduler+水填充。

**📊 数据集**

实验基于28 GHz毫米波小型细胞的仿真数据，包括N_b=128、N_u=16、C=132、MCS 15级，采用多路径散射模型和实际的码书。

**📈 对比分析**

通过与上界、可行解及基准方案比较，离线启发式达到93%近似，在线启发式保留92%性能，且比现有基准提升≥22%，复杂度降低至O((Q+N_F)U+L^3)。

**⚠️ 局限性**

局限在于仅单小区、预定义码书、只考虑理想CSI，未讨论多小区干扰与全CSI上界的进一步提升。

---

## 345. Confident but Conflicted: Internal Uncertainty and Cognitive Dissonance Resolution in LLMs

**arXiv ID:** 2606.22633 | [PDF](https://arxiv.org/pdf/2606.22633v1)

**作者:** Weihong Qi `[一作]` (Indiana University Bloomington), Kristina Lerman `[通讯]` (Indiana University Bloomington)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过在健康科学领域设计两阶段实验，系统评估大型语言模型面对冲突证据时的认知失调解决方式。

**💡 创新点**

创新点在于提出"信任弹性（Trust Elasticity）"指标来量化模型对冲突证据的说服易感性，并将其与内部不确定性指标关联。

**🔧 技术方法**

使用的技术包括基于经济学弹性概念的TE指标、对token logits的概率分布进行熵分析、以及对模型自信度的校准误差评估。

**📊 数据集**

使用了12条健康科学主张，涵盖"完全错误、过度陈述、绝对主义"三类，并针对不同来源权威与证据质量生成对抗性反驳。

**📈 对比分析**

通过对四个模型（Qwen3.5-9B、Llama-3.3-70B-Instruct、Grok-3、GPT-4o）进行TE对比，发现Llama最易受说服，Qwen最不易，并与内部不确定性指标显著相关。

**⚠️ 局限性**

主要局限在于只评估四个模型且仅对两款开放权重模型可访问logits，实验集局限于健康科学单语域，且自生成反驳可能导致跨模型可比性受限。

---

## 346. Interpretable Uncertainty Routing Separating Emotion Ambiguity from Distribution Shift in Facial Expression Recognition

**arXiv ID:** 2606.22725 | [PDF](https://arxiv.org/pdf/2606.22725v1)

**作者:** Keito Inoshita `[一作]` (Kansai University), Takato Ueno `[通讯]` (Shiga University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于深度集成的可解释不确定性分解框架，并将其应用于面部表情识别的推理时路由（Uncertainty‑Aware Routing, UAR），实现对表情歧义（aleatoric）与分布漂移（epistemic）的分离与路由。

**💡 创新点**

创新点在于：① 通过信息理论分解获得可解释的两类不确定性，并分别通过独立外部信号（人类标注者分歧、图像失真）进行验证；② 设计了UAR与其学习版L‑UAR，实现在推理阶段依据两类不确定性做出拒绝、推迟或接受的决策；③ 展示分解后可获得的路由性能大幅优于单一置信度阈值。

**🔧 技术方法**

技术包括：1) 5个全微调的 DINOv2 ViT‑B/14 模型组成的深度集成；2) 信息理论不确定性分解（总熵、均值熵、互信息）；3) 双重验证协议（对标注者分歧的 Spearman 相关、对失真/跨数据集的 AUROC）；4) 通过阈值或轻量级分类器实现的路由算法；5) 温度标定与传统最大 softmax 置信度对比。

**📊 数据集**

使用的主要数据集为 FERPlus（含 10 份投票的表情标注）用于训练与内部验证，RAF‑DB 作为跨数据集迁移测试，以及基于 ImageNet‑C 的 11 种失真（共 5 级）用于外部分布漂移测试。

**📈 对比分析**

与单模型、温度标定单模型、传统深度集成、LDL、SCN、EAC 等基线相比：\n- Aleatoric 与人类分歧的 Spearman 相关为 0.66，AUROC 0.897；\n- Epistemic 在最高失真级别的 AUROC 0.699，显著优于单模型置信度 0.663；\n- UAR 在匹配 0.70 的 OOD 拒绝率下，保留 1.8 倍以上的歧义内分布样本；\n- L‑UAR 在未见失真类型上进一步提升路由 AUC，平均从 0.430 提升至 0.457。

**⚠️ 局限性**

限制：\n- Epistemic 对轻微跨域变化（如 RAF‑DB）不敏感，且在某些失真类型（遮挡）表现不如单模型置信度；\n- Aleatoric 验证仅基于 FERPlus 的投票分布，缺乏其它表情数据集；\n- 采用 5 倍推理成本的深度集成，虽然 3 倍已足够；\n- 研究仅在低分辨率静态图像上进行，未覆盖视频或高分辨率场景；\n- 结果主要关注路由与可解释性，对整体校准和错误率抑制的提升有限。

---

## 347. Subspace-Constrained Federated Learning with Low-Rank Adaptation

**arXiv ID:** 2606.22724 | [PDF](https://arxiv.org/pdf/2606.22724v1)

**作者:** Neranjan Senarath `[一作]` (Rensselaer Polytechnic Institute), Sadia Asif `[通讯]` (Rensselaer Polytechnic Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究并提出了基于子空间正则化的联邦 LoRA 方法，旨在解决因客户端数据异构导致的低秩子空间不对齐问题。

**💡 创新点**

创新点在于为 LoRA 联邦学习引入子空间对齐惩罚，使本地低秩基矩阵与全局参考子空间保持接近，从而提升聚合质量并给出可调的收敛理论。

**🔧 技术方法**

使用了 Frobenius 范数的近端正则化、SVD 重新分配、FedProx 思路以及对 RoBERTa‑large 与 SmolLM‑360M 的实验验证，并提供了对应的理论收敛分析。

**📊 数据集**

实验使用 HellaSwag 推理数据集，模拟 10 个非 IID 客户端（Dirichlet β=0.5 分配），并在两种预训练模型上进行评估。

**📈 对比分析**

与 FedAvg、SVD、FedSVD 等基线比较后，Subspace‑Reg 在 RoBERTa‑large 上取得最高平均最佳准确率 0.454、最低最终损失 1.363，且基底重叠达 0.9999；在 SmolLM‑360M 上尽管准确率略低于 FedAvg，但基底重叠依然接近 1。

**⚠️ 局限性**

主要限制包括准确率提升仅对部分模型有效，模型依赖性明显；对对齐参数 μ 的敏感度和不同模型间机制尚未完全阐明；实验仅覆盖两种预训练模型，需进一步验证。

---

## 348. Habituation at the Gate: Rising Approval and Declining Scrutiny in Human Review of AI Agent Code

**arXiv ID:** 2606.22721 | [PDF](https://arxiv.org/pdf/2606.22721v1)

**作者:** Haoran Yu `[一作]` (Independent Researcher), Yihang Chen `[通讯]` (Georgia Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一项长达七个月、覆盖 400 名重复审阅者的纵向研究中，作者检视了人类审阅者对 AI 代码生成 PR 的行为变化，发现审批率显著上升（从 30.1% 提升至 36.8%），同时 inline 评论数量下降 22%，审阅延迟却增加 3.5 倍。

**💡 创新点**

创新点在于首次在大规模 GitHub 数据上进行 within‑reviewer 的纵向分析，揭示了审阅者对 AI 代码的习惯化（reflexive habituation）倾向，并通过多重对照（跨代理、同仓库人类 PR）排除了日历效应或代码质量提升的解释。

**🔧 技术方法**

主要使用了统计方法：Wilcoxon 符号秩检验、Spearman 相关、Logistic 回归、以及线性/非线性回归来估计审批率、变更请求率、审阅延迟和 inline 评论等指标；对比了早期与后期审阅、经验十位段、代理间差异和人类 PR 基线。

**📊 数据集**

数据集来源于公开的 AI 编码代理 PR 数据集（GitHub Copilot Autofix、Devin、OpenAI Codex CLI、Cursor、Claude Code），包含 16,895 条人类审阅，重点分析了 400 名至少审阅 10 条代理 PR 的重复审阅者共 11,429 条审阅记录，并与同一仓库的 6,618 条人类 PR 审阅进行对照。

**📈 对比分析**

通过将每位审阅者的审阅按时间中点拆分为早期与后期，计算审批率差值并用 Wilcoxon 检验验证统计显著性（p<10^-6）。进一步按经验十位段聚合，发现审批率从 27.9% 递增至 42.4%（+14.5pp）。在同一批审阅者中，inline 评论平均下降 22%，两者呈显著负相关（ρ=-0.556）。相比之下，人类 PR 审批率在相同期段并未同步提升，表明变化具有代理特异性。

**⚠️ 局限性**

局限性包括：样本仓库仅限 star >100，观察窗口仅 207 天，部分代理（Cursor、Codex）的审阅者数量不足，缺乏对 AI 代码质量随时间变化的直接衡量，且无法完全排除审阅者对工作负荷或 PR 提交时间变化导致的延迟增大。

---

## 349. Libretto: Giving LLM Agents a Sense of Musical Structure

**arXiv ID:** 2606.22708 | [PDF](https://arxiv.org/pdf/2606.22708v1)

**作者:** Yichen Xu `[一作]` `[通讯]` (University of California, Berkeley), Yichen Xu (University of California, Berkeley)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出Libretto框架，提供一种可被LLM直接读取、测量并迭代修订的符号音乐语法，并通过检索、结构轴评估和反馈循环实现高质量的音乐生成与修订。

**💡 创新点**

创新点包括：① 设计LLM友好的文本语法（带明确起始槽、声部和小节结构），② 用语料库校准的29维结构轴（节奏、和声、旋律、纹理、形式、变化）对音乐进行可解释的量化，③ 构建检索+自适应生成循环，让模型在生成后根据结构轴进行诊断与改进，而非单纯靠提示。

**🔧 技术方法**

技术实现主要依赖：Claude Code Opus 4.8等大型语言模型；自定义文本语法与MIDI转换器；结构轴计算与百分位标准化；复制风险与极端轴门控；检索库（Lakh MIDI语料）与知识库；生成-评测-修订循环。

**📊 数据集**

使用314条来自Lakh MIDI Dataset的真实MIDI文件，涵盖八种流派（爵士、古典、电子、流行/摇滚、电子、拉丁、福音/灵魂、民谣）。

**📈 对比分析**

对比方法包括单次生成vs循环（最多3轮）、检索开/关；评估指标为通过率（gap填充、完整作品、教育训练、渐进变形）。实验显示：gap填充从12%提升至39%；完整作品从62%提升至94%；检索开启时完整作品通过率从25%提升至75%；总体上循环与检索显著降低极端轴与复制风险，提升生成质量。

**⚠️ 局限性**

局限性包括：语法抽象了力度、微时、打击乐；结构轴维度有限，可能不足以覆盖所有音乐风格；仅针对符号级生成，无法直接输出音频；依赖固定语料库，可能限制创新与跨域生成；需要更多领域知识与用户定制来进一步提升可用性。

---

## 350. Safety-Aware Evaluation of LLM-Generated Driver Intervention Messages through Multi-Task Risk Fusion

**arXiv ID:** 2606.22706 | [PDF](https://arxiv.org/pdf/2606.22706v1)

**作者:** Keito Inoshita `[一作]` `[通讯]` (Kansai University), Keito Inoshita (Kansai University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了基于多任务感知输出的驾驶员干预生成框架，并设计了领域特定的评估指标DSAIS。

**💡 创新点**

创新点在于将四任务（交通、车辆、情绪、行为）融合到LLM生成干预，并通过DSAIS的五维子分数实现对风险-语调、一致性、简洁性、认知负荷和可接受性的可复现评估。

**🔧 技术方法**

技术包括多任务感知模型CauPsi、风险融合与历史管理、动态提示构造的LLM生成、以及混合规则+LLM Judge的DSAIS评估。

**📊 数据集**

使用公开的AIDE驾驶数据集（3062条视频，四任务标注）进行实验。

**📈 对比分析**

与基线模板、单任务DER以及七种误分类条件下对比，DSAIS在整体得分、子分数以及误分类鲁棒性方面显著优于规则基线，compact 7B–9B模型在DSAIS上接近或超过API模型。

**⚠️ 局限性**

局限在于数据集为离散短视频，缺乏连续驾驶情境，且DSAIS对简洁性/认知负荷的分数饱和，缺少真实驾驶者的行为验证。

---

## 351. SCRUB-FL: Sanitizing and Cleansing Representations via Unlearning of Backdoors

**arXiv ID:** 2606.22700 | [PDF](https://arxiv.org/pdf/2606.22700v1)

**作者:** Osama Wehbi `[一作]` (Polytechnique Montréal), Hadi Otrok `[通讯]` (Khalifa University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在联邦学习环境中提出了后训练去背后的两阶段框架SCRUB-FL，能够在不访问客户端数据、无触发器知识和无大规模干净样本的前提下，对已收敛的全局模型进行去背后处理。

**💡 创新点**

创新点包括：①在训练期间利用光谱签名分析与激活聚类两种无监督异常检测方法捕获可疑样本；②让各客户端训练轻量级WGAN‑GP并上传生成器参数，聚合得到全局对触发器分布的隐式表征；③后训练阶段通过机器“遗忘”优化将触发器样本的输出均匀化，既消除背后映射又保留正常任务准确率，避免了传统剪枝导致的神经元纠缠。

**🔧 技术方法**

技术手段包括联邦平均、光谱签名分析、激活聚类、Wasserstein GAN‑GP（梯度惩罚）、机器遗忘（忘记+保持损失融合）以及生成的触发器近似样本。

**📊 数据集**

实验数据集为CIFAR‑10与GTSRB，覆盖三种背后攻击（One‑to‑One、One‑to‑N、N‑to‑One）和不同恶意参与比例（20%‑40%）。

**📈 对比分析**

与传统聚合阶段防御（FLAME、FLTrust）及后训练防御（Fine‑Pruning、Neural Cleanse）对比，SCRUB‑FL在CIFAR‑10上平均准确率77.9%、ASR4.0%，在GTSRB上准确率91.2%、ASR3.9%，显著低于其他方法的ASR且保持更高的干净准确率。

**⚠️ 局限性**

局限性包括：仍需要客户端执行额外的光谱与聚类分析与WGAN训练，可能增加边缘设备计算负担；后训练遗忘阶段需要一定量的干净参考样本；在极端攻击或高度自适应的触发器设计下，生成器捕获的分布可能不够完整。

---

## 352. Catching Lies Without Sending the Video: Privacy-Preserving Multimodal Deception Detection

**arXiv ID:** 2606.22699 | [PDF](https://arxiv.org/pdf/2606.22699v1)

**作者:** Nikita Sharma `[一作]` (University of California, San Diego), Karan Singla `[通讯]` (WhissleAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一套在设备上提取可解释特征摘要的隐私友好多模态欺骗检测管道；

**💡 创新点**

创新点在于不需要发送原始视频，仅通过约250个可解释特征实现与视频模型相当甚至更优的检测性能，并显著降低输入代价；

**🔧 技术方法**

采用了本地语音识别、视觉跟踪、语音分析等多路特征提取，并结合梯度提升树或LLM判别；

**📊 数据集**

使用了公开的Real‑life Trial Deception数据集（121条庭审视频，61假话/60真话）；

**📈 对比分析**

在留一说话人评估中，使用摘要特征的分类器AUC达0.741，交由Claude Opus 4.8的LLM判别AUC 0.755，超过直接观看原始视频的模型（AUC 0.749），同时输入令牌减少7.8倍；

**⚠️ 局限性**

局限性包括数据集规模小、仍存在说话人泄漏和人口统计偏倚、整体准确率仅在65–70% 之间，且尚未经过严格的公平性与可争议性评估，无法直接用于司法场景。

---

## 353. Generative Relightable Avatars

**arXiv ID:** 2606.22718 | [PDF](https://arxiv.org/pdf/2606.22718v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 354. SATURN: Symbolic Spatial Reasoning for Multi-Perspective Grounding

**arXiv ID:** 2606.22694 | [PDF](https://arxiv.org/pdf/2606.22694v1)

**作者:** Danial Kamali `[一作]` (Michigan State University), Parisa Kordjamshidi `[通讯]` (Michigan State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SATURN neuro‑symbolic 框架，结合 3D 场景重建、软化视角（FoR）空间谓词和 Python 级符号执行，解决多视角、跨视角的组合空间推理问题；同时构造了 3D FORCE 诊断基准，用以系统评估视角感知与多跳空间关系的组合能力。

**💡 创新点**

核心创新在于：① 将低级 3D 估计与软化空间谓词分离，避免对噪声感知做硬决策；② 用视角（FoR）为基准的软空间谓词实现多视角、跨视角关系的统一表达；③ 通过 LLM 生成 Python 程序进行符号执行，将 VLM 语义能力与结构化空间推理相结合；④ 设计 3D FORCE 基准，精准控制推理深度、视角和关系拓扑，填补现有 VLM 空间推理评测空白。

**🔧 技术方法**

技术包括：VLM（如 Qwen3‑VL、InternVL）做查询引导的场景估计；VGGT、DA3、CUT3R 等 3D 重建网络；空间引擎通过局部坐标系计算软化 FoR 关系；LLM 生成 Python 程序并调用 soft predicate 接口；软化逻辑组合（如 min, max 等）实现不确定性传播；文本驱动的视角约束提取与视角对齐。

**📊 数据集**

使用的主要数据集有：① 3D FORCE（自研诊断基准，包含 SAG 与 REF 两子集）；② MindCube‑1K 与 MMSI 真实多视角空间推理基准；③ 参考的 Spatial457 用于验证基线。

**📈 对比分析**

与多种基线（通用 VLM、空间细化 VLM、工具增强方法 GCA、pySpatial 等）进行对比，结果显示：在 3D FORCE 上 SATURN 达到 88.85% 最高准确率，较最强 VLM 提升 21pp；在 MindCube 上提升 14pp，取得 78.57%；在 MMSI 上实现 48.77%，同样领先；此外，软化符号推理相较硬阈值和纯几何编程提升 10–12pp，证明不确定性保持的重要性。

**⚠️ 局限性**

主要局限包括：① 依赖底层 VLM 与 3D 重建模型，若感知误差大则影响整体推理；② 对相机与物体方向估计高度依赖，视角对齐错误会直接导致谓词错误；③ 需要较大的 GPU 资源支持大规模 VLM 与 3D 重建，限制了可访问性；④ 目前仍为实验性框架，尚未在更广泛的真实世界环境中验证。

---

## 355. moBERTo: A Modern Encoder for Portuguese via Continued Pretraining of ModernBERT

**arXiv ID:** 2606.22722 | [PDF](https://arxiv.org/pdf/2606.22722v1)

**作者:** Thiago Laitz `[一作]` (UNICAMP), Giovana Kerche Bonás `[通讯]` (UNICAMP)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对现代BERT进行继续预训练，构建了用于葡萄牙语的moBERTo编码器。

**💡 创新点**

结合子词匹配嵌入迁移和长上下文后训练，使得葡语编码器兼具现代架构与长上下文能力。

**🔧 技术方法**

采用Rotary位置编码、交替本地-全局注意力、Flash Attention、unpadding，并用Composer框架训练。

**📊 数据集**

使用从FineWeb2裁剪的约12B葡语文本（60B token继续预训练）以及检索、分类、NER和PLUE-PT基准数据。

**📈 对比分析**

与BERTimbau、NeoBERT等基线相比，moBERTo在检索、长上下文检索、分类、NER和PLUE-PT上均实现显著提升，平均nDCG@10最高达0.5255。

**⚠️ 局限性**

局限性包括对英文任务迁移性能下降、词表适配可能影响长上下文检索，以及仅基于Encoder的规模仍低于大型Decoder模型。

---

## 356. RigorBench: Benchmarking Engineering Process Discipline in Autonomous AI Coding Agents

**arXiv ID:** 2606.22678 | [PDF](https://arxiv.org/pdf/2606.22678v1)

**作者:** Meher Bhaskar Madiraju `[一作]` (Georgia Institute Of Technology), Meher Sai Preetam Madiraju `[通讯]` (Georgia Institute Of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了RigorBench基准，评估AI编码代理的过程纪律，并在其上对比多种代理框架。

**💡 创新点**

创新点在于首次引入基于轨迹的五柱过程评估（计划、验证、恢复、回避与原子转移）并用加权得分综合量化过程质量。

**🔧 技术方法**

使用轨迹日志收集、LLM‑as‑judge自动化评分、加权综合评分与可视化散点分析等技术。

**📊 数据集**

使用30个自定义任务（Plan‑Then‑Build、Verify‑Or‑Die、Doom‑Loop Gauntlet、Know When to Fold、Don’t Break the Build）构成的多类别任务集。

**📈 对比分析**

与Agent‑Rigor、Agent‑Skills、Superpowers以及基线ReAct进行对比，实验显示结构化框架将过程质量提升约41%，同时结果正确率提升约17%，两者之间的相关系数高达0.87。

**⚠️ 局限性**

局限性包括任务量相对有限、仅验证一种纪律框架、LLM‑judge评分可能存在偏差、任务难度随模型进步而快速变化等。

---

## 357. GARIP: A Running-Average Moving Reference for Last-Iterate Self-Play in Two-Player Zero-Sum Games

**arXiv ID:** 2606.22688 | [PDF](https://arxiv.org/pdf/2606.22688v1)

**作者:** Can Savcı `[一作]` `[通讯]` (Independent Researcher), Can Savcı (Independent Researcher)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并评估GARIP，一种在两人零和博弈中以策略运行平均为参考点的自我对弈方法，能够在不使用退火的情况下实现最后迭代收敛并提升鲁棒性。

**💡 创新点**

发现运行平均参考点在“峰值滞后”上优于周期快照，证明其在因果平均中最小化峰值，从而显著扩大稳健区间；同时给出局部最后迭代收敛理论与崩塌边界的经验分析。

**🔧 技术方法**

优化方法结合了乐观镜像上升与Halpern锚定，采用JAX实现；实验中使用矩阵博弈、Kuhn/Leduc扑克、Coin Game、连连看等多种深度RL和棋盘游戏的自我对弈。

**📊 数据集**

矩阵游戏随机样本、Kuhn、Leduc扑克手工设置、Coin Game、Connect Four、Othello、Animal Shogi、Hex 等棋类与扑克环境。

**📈 对比分析**

与MMD、R-NaD、原始自我对弈、拟合回归等基线在最后迭代可达性、可利用性、崩塌率等指标比较，GARIP在默认参数下与R-NaD实现相同的峰值表现，但在滞后稳健性上表现更好，崩塌率更低，尤其在常用参数区间。

**⚠️ 局限性**

不具备全局收敛性证明；在大锚定强度下可能出现预期一致性失稳；在极慢平均或极快追踪速率下仍会崩塌；代理可利用度指标在稀疏奖励空间可能被动策略误导；实验主要基于无搜索的PPO，可能不代表更强学习器。

---

## 358. Integrated cloud-based architecture for robot-robot and human-robot collaboration using ROS 2--MQTT in Mediterranean Greenhouses

**arXiv ID:** 2606.22682 | [PDF](https://arxiv.org/pdf/2606.22682v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 359. Foundation Models for Epileptogenic Zone Identification in Drug-Resistant Epilepsy

**arXiv ID:** 2606.22657 | [PDF](https://arxiv.org/pdf/2606.22657v1)

**作者:** Thi Kieu Khanh Ho `[一作]` (McGill University), Narges Armanfard `[通讯]` (McGill University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一套基于双基础模型的癫痫发作起源区（EZ）定位系统——EpiiSLM，利用sEEG信号和多模态临床信息进行自动化诊断；

**💡 创新点**

创新点在于：①采用大规模无标签sEEG数据进行自监督预训练，克服传统方法只能使用少量手工挑选数据的局限；②通过将监督学习锚定于明确的非癫痫负类（未切除且非SOZ的联系人），解决了EZ标签模糊的问题；③将信号基础模型与冻结的医学语言模型相结合，实现高可解释性的医学推理输出；

**🔧 技术方法**

技术包括：Transformer‑based信号基础模型（BIOT骨干）进行无标签掩码重建预训练；One‑Class 和 Binary 两头的监督微调；MedGemma 等医学预训练语言模型用于融合SOZ、MRI等临床先验并生成最终预测；多阶段训练（无监督+监督）与自适应阈值；

**📊 数据集**

数据集：MNI（蒙特利尔神经学研究院）30名药物难治癫痫患者的104,990分钟sEEG（868 GB）以及St. Anne's医院（布尔诺）17名患者的7,106分钟sEEG，涵盖多种病因、术式与电极类型；

**📈 对比分析**

与传统将SOZ直接视为EZ的基线相比，EpiiSLM在留一患者交叉验证中contact‑level PPV_3达到0.978，比基线高15.1 %；在外部数据集上亦取得0.857；区域级别精准率达到100 %；单夜N3睡眠数据即可保持≈92 % PPV_3，显著降低监测时长；

**⚠️ 局限性**

局限性包括：①对大规模多中心数据的依赖，现有样本量仍有限；②对非商业化自制电极的性能稍差；③模型对高频噪声或运动伪影敏感；④需进一步验证在真实临床决策流程中的效果与安全性。

---

## 360. MaRS: Robust Out-of-Distribution Detection via Mahalanobis Residual Scoring

**arXiv ID:** 2606.22649 | [PDF](https://arxiv.org/pdf/2606.22649v1)

**作者:** Francesco Di Salvo `[一作]` (University of Bamberg), Christian Ledig `[通讯]` (University of Bamberg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种无标签、后置式OOD检测方法——Mahalanobis Residual Scoring（MARS），通过轻量级自编码器学习特征空间的内在分布，并用马氏距离对重建残差进行方差感知评分；

**💡 创新点**

核心创新在于用马氏距离代替传统L₂范数，利用残差空间的异方差结构显著提升OOD判别，且不依赖标签、分类头或预定义子空间；

**🔧 技术方法**

使用冻结的基础模型（如ViT、DINOv3）提取特征，训练两层MLP自编码器，估计残差协方差并计算马氏距离；

**📊 数据集**

在MIDOG（组织学）、成人与儿童胸部X光以及HAM10000（皮肤病变）三个医学影像数据集上进行实验；

**📈 对比分析**

与MSP、ODIN、ViM等置信度方法、Deep kNN、OCSVM、Mahalanobis++等距离方法以及Residual、AE等重建方法比较，MARS在AUROC和FPR@95上均显著优于所有基线，表现出跨模态和尺度的鲁棒性；

**⚠️ 局限性**

对MARS的限制包括对自编码器学习的内在流形质量敏感，且目前仅验证了单模态设置，未来需探究多模态扩展及更专业医学骨干模型的影响。

---

## 361. The Geometry of Refusal: Linear Instability in Safety-Aligned LLMs

**arXiv ID:** 2606.22686 | [PDF](https://arxiv.org/pdf/2606.22686v1)

**作者:** Shivam Ratnakar `[一作]` (University of Southern California), Kartikeya Vats `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并提出 Contrastive Logit Steering（CLS），一种通过对比安全与不安全系统提示的输出 logits 并直接在 logits 级别做线性干预来绕过 LLM 的安全拒绝机制。

**💡 创新点**

创新点在于：①零优化的 logits 级别干预方式，可快速在推理时实现；②通过“拒绝向量”揭示安全性是线性可分的“安全轴”，并提出“Late Decision”与“Early Divergence”两种安全拓扑；③负向 steering 可在不训练的情况下硬化模型。

**🔧 技术方法**

技术手段包括：三路推理（安全、无安全、基准）得到 logits 差值作为 steering vector；prefix injection 以规避初始拒绝；α 参数控制正向（解禁）或负向（硬化）；PCA 与 KL 散度用于机制分析；余弦相似度做零样本安全检测。

**📊 数据集**

数据集涵盖 AdvBench、JailbreakBench、HarmBench 等，用于 α 调参、攻击效果评估与安全检测验证。

**📈 对比分析**

与传统 GCG 与中间隐藏层 steering（Arditi 等）对比，CLS 在 Llama‑3.1 上实现 95% ASR，仅需 1 秒；相比 GCG 仅 5% ASR；在 Llama‑2 与 Qwen‑7B 上分别提升 73% 与 91% 的攻击成功率；负向 steering 在 Llama‑3.3‑70B 及 Gemma‑3‑12B 上将成功率降至 <10%，并提升连贯性。

**⚠️ 局限性**

限制包括：需要白盒 logits 访问，无法直接应用于封闭模型；α 对不同模型需调参；实验聚焦单轮交互，未考察多轮情境；对比实验缺少 RepE 与 Activation Addition 的定量评估。

---

## 362. Only Ask What You Don't Know: Grounded Delta Planning for Efficient Multi-step RAG

**arXiv ID:** 2606.22681 | [PDF](https://arxiv.org/pdf/2606.22681v1)

**作者:** Wei-Chieh Chou `[一作]` (National Taiwan University), Jyh-Shing Roger Jang `[通讯]` (National Taiwan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 GDP‑RAG 框架，在多跳问答中先做检索，再基于检索缺失信息进行规划，生成骨干轨迹并通过 Act‑Review‑Update 循环执行。

**💡 创新点**

核心创新是 Grounded Delta Planning：仅对检索缺口进行规划，结合预检索、gap‑aware 分解与骨干轨迹，有效压缩推理步骤、去除冗余并提升准确性。

**🔧 技术方法**

采用三阶段 RAG 结构：预检索、gap‑conditioned 规划、skeletal 轨迹，配合 Act‑Review‑Update 循环；实现基于 GPT‑4.1‑mini 的生成和 BAAI/bge‑m3 等检索模型。

**📊 数据集**

实验使用 HotpotQA、2WikiMultiHopQA 与 MuSiQue 三大多跳问答基准。

**📈 对比分析**

与 IRCoT、KnowTrace、Search‑o1、PAR‑RAG、Godbole 等方法对比，GDP‑RAG 以 60.63% 的最高准确率和 0.51¢ 的最低 cost‑of‑pass 成为 Pareto 前沿的最佳方案。

**⚠️ 局限性**

局限在于仅适用于短篇多跳问答；开放式生成、长篇合成或多模态检索等场景尚未验证，信息缺口定义和评估指标需进一步扩展。

---

## 363. Skin-Deep: A Geometric Diagnostic for Alignment Fragility in Large Language Model Representations

**arXiv ID:** 2606.22676 | [PDF](https://arxiv.org/pdf/2606.22676v1)

**作者:** Dongyub Jude Lee `[一作]` (Zoom Communications), Heuiseok Lim `[通讯]` (Korea University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种几何诊断方法Skin‑Deep，利用对齐模型与基础模型隐藏层激活的差异，在未进行任何攻击或微调前检测对齐脆弱性，并压缩为单一Geometric Fragility Score（GFS）。

**💡 创新点**

创新点在于：①发现对齐模型在隐藏层中存在低秩安全子空间；②通过对比PCA、Cohen’s d等统计量验证该子空间对拒绝行为的因果影响；③将多层分数加权聚合成GFS，为预部署安全评估提供量化指标。

**🔧 技术方法**

主要技术包括对齐与基础模型激活的对比PCA搜索、Cohen’s d与PERMANOVA、RBF‑MMD的全空间检验、方向消除(ablation)验证、CKA相似性分析以及层加权求和得到GFS。

**📊 数据集**

实验使用了500个有害请求提示（AdvBench、HarmBench、BeaverTails）与500个正面指令（Alpaca、OASST）进行对比激活；LoRA微调实验则采集Alpaca子集，并在AdvBench与Alpaca等安全基准上评估拒绝率。

**📈 对比分析**

通过与LoRA后拒绝率、Cohen’s d、CKA相关性等多维指标比较，发现高GFS模型在LoRA微调后仍能保持较高拒绝率，且对齐子空间在不同模型族间高度相关；方向消除实验进一步验证了子空间对拒绝行为的因果作用。

**⚠️ 局限性**

局限性包括：仅覆盖3B–32B规模的开放权重模型，未验证更大或MOE模型；仅使用英文数据；对齐方法范围有限；GFS仅基于单一低秩子空间角度，未给出完整子空间距离；对不同对齐方案的泛化能力仍待进一步验证。

---

## 364. Clipping the Price of Adaptivity at the Tail

**arXiv ID:** 2606.22669 | [PDF](https://arxiv.org/pdf/2606.22669v1)

**作者:** Itai Kreisler `[一作]` (Tel Aviv University), Oliver Hinder `[通讯]` (University of Pittsburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了自适应随机凸优化（SCO）方法，提出了一种新的方法来克服适应性带来的效率障碍，特别是在面对初始最优距离和Lipschitz常数的不确定性时。

**💡 创新点**

创新点在于假设目标函数可以分解为模型和损失函数，从而在推理时通过剪切模型输出来处理尾部事件，避免了在极端情况下的高代价。

**🔧 技术方法**

使用了剪切技术和网格搜索相结合的方法，设计了两种不同的算法，一种侧重于计算效率，另一种侧重于样本效率。

**📊 数据集**

使用了多种样本集进行实验，具体样本集未在摘要中详细说明。

**📈 对比分析**

与传统的参数已知SCO方法相比，提出的方法在不确定性较大的情况下，能够以对数因子匹配已知参数SCO的最优界限，性能显著提升。

**⚠️ 局限性**

限制在于该方法依赖于模型-损失结构的假设，可能不适用于所有类型的优化问题。

---

## 365. All Relations Lead to Rome: Automated Knowledge Graph Creation and Question Generation

**arXiv ID:** 2606.22645 | [PDF](https://arxiv.org/pdf/2606.22645v1)

**作者:** Matthijs Jansen op de Haar `[一作]` (University of Twente), Lorenzo Gatti `[通讯]` (University of Twente)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了All Relations Lead to Rome统一框架，自动构建知识图谱并生成事实驱动的问答对；

**💡 创新点**

创新点在于将向量检索与知识图谱推理统一为同一资源，提供实体、关系、向量和事实对齐的四元组；

**🔧 技术方法**

利用LLM（minimax 2.7）完成实体与关系抽取，使用gemini-embedding-2生成chunk与entity嵌入，并在Neo4j中存储与检索；

**📊 数据集**

基于约300篇维基百科关于罗马帝国的文章，构建了约19,374实体、16,069 chunk、25,304关系以及8,400问答对的数据集；

**📈 对比分析**

目前仅提供框架与数据集，没有公开具体评测结果，主要目标是为后续RAG/GraphRAG等混合检索方法提供统一基准；

**⚠️ 局限性**

主要局限包括：LLM抽取误差、答案多样性导致标注不唯一、缺乏人工校验、仅覆盖罗马历史领域，难以直接推广到结构稀疏或极稠密的其他领域。

---

## 366. ColumnKeeper: Efficient Solutions to the ColumnDisturb Vulnerability in DRAM-based Systems

**arXiv ID:** 2606.22632 | [PDF](https://arxiv.org/pdf/2606.22632v1)

**作者:** Andreas Kosmas Kakolyris `[一作]` (ETH Zuerich), Onut Mutlu `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对新型列扰动（ColumnDisturb）攻击的两种 DRAM 保护机制：Deterministic ColumnKeeper（CGuardD）和 Probabilistic ColumnKeeper（CGuardP）。

**💡 创新点**

创新点在于：①利用 DRAM 开放式位线架构，分别统计奇偶列激活次数以避免“double‑counting”，从而实现低开销的确定性防护；②通过随机刷新概率机制，消除计数器需求，提供可配置的安全保障，且硬件面积极小；③提供可在内存控制器或 DRAM 内部实现的版本，并支持子阵列级并行（SALP）与更小子阵列大小以降低阈值下的开销。

**🔧 技术方法**

技术包括：DRAM 子阵列级激活计数、行指针表（RPT）实现轮询刷新、概率触发机制、Monte Carlo 与理论安全分析、Cycle‑accurate 仿真 Ramulator 2.0 结合 DRAMPower、SALP 微架构改造、子阵列映射逆向工程等。

**📊 数据集**

使用 62 个单核工作负载（来自 SPEC CPU2006/2017、TPC、MediaBench、YCSB）和 60 个四核混合工作负载（按行缓冲命中率分类），以及合成的高速行激活攻击工作负载。

**📈 对比分析**

与基线（无 ColumnKeeper）以及现有 RowHammer/RowPress 保护（Graphene、PRAC、Hydra）对比，CGuardD 和 CGuardP 在当前阈值（1M）下平均 IPC 降低 <1%（0.3%–0.6%），能耗提升 <2%；在近未来阈值（128K）下平均 IPC 降低 <3%；在低阈值（16K）下通过 SALP 或更小子阵列可将性能降至 5–10% 左右；相较于单计数器设计，CGuardD 能更有效避免冗余刷新。

**⚠️ 局限性**

局限性包括：需要 DRAM 子阵列映射信息（需逆向或供应商支持），对低阈值的性能仍较高，概率版本安全保障取决于精确的刷新概率设置；实现仍以内存控制器为主，可能不适用于所有 DRAM 规格；在大规模多核系统下，刷新冲突和能耗仍可能显著。

---

## 367. BLUEX v2: Benchmarking LLMs on Open-Ended Questions from Brazilian University Entrance Exams

**arXiv ID:** 2606.22723 | [PDF](https://arxiv.org/pdf/2606.22723v1)

**作者:** João Guilherme Alves Santos `[一作]` (UNICAMP), Helio Pedrini `[通讯]` (UNICAMP)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了BLUEX v2多模态开放式问答基准，用以评估葡萄牙语LLM在巴西大学二期入学考试的表现。

**💡 创新点**

首次提供了包含图像、六项认知标签的开放式评测集，并通过LLM生成评分标准的自动评判协议实现可扩展的评估。

**🔧 技术方法**

使用LLM-as-a-judge协议、Sabiá-4生成的评分准则、Gemini 3.1 Flash Lite进行图像说明、OpenRouter/API进行模型推理。

**📊 数据集**

基于2022-2025年UNICAMP和USP二期入学考试的395道题（919个子题），含55.7%带图题。

**📈 对比分析**

评估21款LLM，最高得分9.10/10（Gemini 3.1 Pro Preview），最低4.18/10（LLaMA-3.2-11B Vision），数学推理和图像理解是最难的维度，LLM评判与人工评判一致率89.5%。

**⚠️ 局限性**

局限包括：评分准则由LLM生成可能缺失细节；评判者与人类评判存在5-8个百分点差距；图像说明依赖Gemini导致IU性能可能被低估；顶端分数接近上限。

---

## 368. One-Prompt Censorship Evasion via Generative Diffusion Models

**arXiv ID:** 2606.22717 | [PDF](https://arxiv.org/pdf/2606.22717v1)

**作者:** Shiyi Ling `[一作]` (University of California Santa Cruz), Chen Qian `[通讯]` (University of California Santa Cruz)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将被审查的网络流量头部可视化为图像，通过指令驱动的扩散模型进行语义编辑，从而在不改动负载的前提下将流量“重绘”为合法模式，实现一键式审查规避。

**💡 创新点**

首次将审查规避建模为图像到图像的编辑任务，利用三色视觉编码、双条件扩散网络以及协议解码器，自动将流量的统计特征重塑为合法流量；同时提供自然语言提示接口，简化配置与使用。

**🔧 技术方法**

使用 InstructPix2Pix‑sdxl 指令调优扩散模型、nPrint 位域编码、三色 RGB 映射、双条件文本+图像 U‑Net、双向 CFG 引导以及后端协议修复器。

**📊 数据集**

基于 Stratosphere IPS 的合法 HTTPS 流量与其通过 Geneva/UPGen 生成的规避流量共 6,712 对，用于训练；实验中对 CurveZMQ、secio、SSH、TLS 等 OOD 流量以及 GFW‑style 规则型中间盒进行评估。

**📈 对比分析**

与 Obfs4、UPGen 等基线以及 Decision Tree、Random Forest、nPrintML、Deep Fingerprinting 等学习型与规则型分类器进行 OOD 评估；在 FPR≤0.1% 的严格约束下，FlowPaint 的 TPR 在多数模型中为 0，甚至在深度学习模型上将 TPR 降至 0.70‑0.80；对三类规则型中间盒（状态重组、RFC 检查、反重放）亦能成功规避，整体性能显著优于现有技术。

**⚠️ 局限性**

主要限制在于扩散推理的计算延迟与带宽开销；仅对头部进行编辑，无法处理基于加密负载的审查；在资源受限或极端动态策略变更环境下可能需要频繁重新 fine‑tune。

---

## 369. Beyond Simpson's Paradox: A Cascade of Confounders in AI Agent Pull-Request Co-Authorship

**arXiv ID:** 2606.22711 | [PDF](https://arxiv.org/pdf/2606.22711v1)

**作者:** Haoran Yu `[一作]` (Independent Researcher), Yihang Chen `[通讯]` (Georgia Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 AI 代码生成代理提交的 Pull Request (PR) 中人类合作者对合并率的影响，揭示了 Simpson’s Paradox 并提供了多层次控制的实证结果。

**💡 创新点**

首次将 Simpson’s Paradox 引入 AI 代理 PR 分析，并通过分层控制（代理、仓库、PR 结构）系统性剖析人类协作与合并率的真实关联，提出了四级协作框架与代理特定部署建议。

**🔧 技术方法**

采用统计检验（Pearson χ²、差分差分回归）、固定效应回归与聚合计数比较，结合作者/提交者身份推断与 co‑authorship 追踪技术。

**📊 数据集**

使用 2024‑12 至 2025‑07 期间收集的 AIDev 数据集，共 33,596 条 PR，涵盖五大 AI 代码生成代理（Codex、Copilot、Devin、Cursor、Claude Code）。

**📈 对比分析**

在跨代理聚合层面，人类合作者显示合并率下降（-26.0 pp），但在代理层面、仓库层面以及多提交 PR 控制下，人类合作者均呈正向提升（最高 +41.2 pp），差分差分分析进一步显示引入第二代理会导致合并率下降约 12.1 pp。

**⚠️ 局限性**

研究仅为描述性关联，缺乏因果识别；co‑authorship 标记依赖代理自插入的 trailer，可能存在漏测；数据仅覆盖采用 AI 代理的仓库，且未考虑 PR 大小、任务类型与评审者经验等因素，结果的外推性受限。

---

## 370. A Note on Learnable Nash Equilibrium

**arXiv ID:** 2606.22701 | [PDF](https://arxiv.org/pdf/2606.22701v1)

**作者:** Songzi Du `[一作]` `[通讯]` (University of California San Diego), Songzi Du (University of California San Diego)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

证明了在对称两人一般博弈中，若纳什均衡的指数为 +1，则它是可学习的（可被某个近视调节动力稳定）。

**💡 创新点**

创新点在于给出了指数 +1 的充分条件与可学习性之间的反向证明，完成了前人仅给出必要条件的研究，利用线性代数与矩阵谱分解构造了满足近视调节条件且使均衡稳定的动力。

**🔧 技术方法**

主要技术包括：对称博弈的支撑子矩阵 B 的构造，利用 Hurwitz 稳定性与正定/半正定矩阵性质，3×3 情况下直接构造 M，通用情形采用实 Schur 分解将 B 变为块上三角形式，再为每个块选取合适的 K 使 MB Hurwitz 且满足 z′M z ≥ 0。

**📊 数据集**

无数据集，研究完全基于理论证明与数学构造。

**📈 对比分析**

本文没有实验或对照方法；比较基于已有理论结果，指出指数 +1 的均衡在一般博弈中约占一半，且比传统精炼更具约束力。

**⚠️ 局限性**

局限性：仅适用于“generic”（泛化）博弈，要求支撑完整且支付正数；构造的动力只保证局部渐近稳定，未讨论全局行为；对非对称或多玩家情况的推广仍待研究。

---

## 371. Black-Box Forensics for Conversational LLM Agents

**arXiv ID:** 2606.22698 | [PDF](https://arxiv.org/pdf/2606.22698v1)

**作者:** Isadora White `[一作]` (University of California, San Diego), Taylor Berg-Kirkpatrick `[通讯]` (University of California, San Diego)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套黑盒取证框架，通过与对话探测实现对隐藏LLM基模型与系统提示的归因与指纹识别。

**💡 创新点**

在非对抗性多轮对话中实现零样本系统提示指纹化与基模型归因，无需模型内部信息或已知提示，能聚类诈骗网络。

**🔧 技术方法**

结合TF‑IDF+统计特征、LLM密集分类器、对比学习与交叉编码器（ELECTRA、BERT、MPNET）以及基于对话对的相似度判定。

**📊 数据集**

使用约240k条合成对话数据，覆盖6个基模型、40个系统提示、70个话题，并包含受控系统提示变体。

**📈 对比分析**

与n‑gram重叠、MMD、Bi‑Encoder等基线对比，交叉编码器在零样本指纹上达到AUC 0.768/F1 0.703；增至50轮对话可提升至AUC 0.943/F1 0.79；基模型归因准确率高达98%。

**⚠️ 局限性**

局限于合成数据，对真实野外诈骗实例尚未验证；系统提示的频繁变动和对抗性伪装（如重述）可能削弱效果，仅适用于公开API与恶意网络的审计。

---

## 372. VISTA Architect: A graph database-oriented health AI system demonstrated in multidisciplinary tumor boards

**arXiv ID:** 2606.22692 | [PDF](https://arxiv.org/pdf/2606.22692v1)

**作者:** Tuomo Kiiskinen `[一作]` (Stanford University), Manuel A. Rivas `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 VISTA Architect 系统，利用图数据库预处理 EHR 并生成可查询的临床时间线，以支持多学科肿瘤委员会快速汇总与决策。

**💡 创新点**

创新点在于将原始 EHR 转化为源忠实 MEDS 图，再抽象为临床时间线 TOA，预先计算并保留时间和证明链；代理式 LLM 仅在需要时检索源记录，显著提升准确率和查询速度。

**🔧 技术方法**

使用了图数据库（Neo4j/NetworkX）、大型语言模型（GPT‑4.1/5、Claude、Gemini）、多层架构（MEDS Graph + TOA + 代理接口）以及并行与多代理编排。

**📊 数据集**

基于斯坦福医学 VISTA 肿瘤数据湖的 1180 例胸腔肿瘤患者全记录（MEDS XML），涵盖结构化表格、自由文本、基因与影像等多模态数据。

**📈 对比分析**

与 BM25 检索增强生成 (RAG) 做对比；在 17000+ 变量评估点上 VISTA 准确率 96.4%，RAG 仅约 66%；在 30 例全代理并行构建耗时 2.2 min（单例构建 74 min），显著提升效率。

**⚠️ 局限性**

局限性包括仅在单中心英文记录验证，依赖完整 EHR 并缺少外部资料；功能验证目前仅覆盖胸腔肿瘤，需要多中心、多语种、跨机构评估以进一步验证泛化能力。

---

## 373. Semantic-Aware Autonomous Exploration for UAVs in Unknown Indoor Environments

**arXiv ID:** 2606.22670 | [PDF](https://arxiv.org/pdf/2606.22670v1)

**作者:** Duc-Thien Nguyen `[一作]` (University of Engineering and Technology, Vietnam National University), Thanh Nguyen Canh `[通讯]` (University of Engineering and Technology, Vietnam National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种语义感知自主探索框架，利用RGB‑D传感器对未知室内环境进行几何与语义联合映射，并通过语义奖励引导UAV生成信息增益更高、路径更短的探索轨迹。

**💡 创新点**

核心创新在于：①将语义奖励函数与基于PRM的动态探索规划（DEP）相结合，实现语义与几何信息的协同评估；②设计增量式路网管理策略，持续更新导航图并保留先前生成的拓扑；③通过闭环信息流让传感器数据、语义检测与地图构建实时交互。

**🔧 技术方法**

技术手段包括：ROS Noetic + Gazebo仿真；RGB‑D + IMU感知；体素占用地图 + 欧氏签名距离场；PRM（Dynamic Exploration Planner）与增量路网；语义检测与滤波；语义奖励函数；基于信息增益的视角评估；轨迹生成与优化。

**📊 数据集**

实验使用三种Gazebo仿真室内环境（50 m²、100 m²、200 m²）进行验证，未使用公开数据集。

**📈 对比分析**

与NBV、AEP、DEP等传统几何基探索方法进行对比；在三种环境下评估飞行时长、行程距离和地图覆盖率。结果显示本方法覆盖率提升5–7个百分点，飞行时间缩短约10–30%，路径长度也相应减少。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证；语义感知依赖预先标注或简易检测，未使用深度学习实时推理；未考虑多UAV协同；对动态或未知障碍物的适应性有限；采样密度需手动调节以平衡计算负荷与性能。

---

## 374. Learning Entropy Signature for Image Representation and Classification

**arXiv ID:** 2606.22634 | [PDF](https://arxiv.org/pdf/2606.22634v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 375. LSTM Variants for Chaotic Dynamical Systems: An Empirical Study on the Lorenz Attractor

**arXiv ID:** 2606.22662 | [PDF](https://arxiv.org/pdf/2606.22662v1)

**作者:** Ruslan Gokhman `[一作]` `[通讯]` (Yeshiva University), Ruslan Gokhman (Yeshiva University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在AI‑DEEDS 2026 乱流系统挑战中，对七种递归与卷积模型（LSTM、BiLSTM、带注意力的LSTM、Huber损失的BiLSTM、TCN、CNN+LSTM、CNN+BiLSTM）进行对比实验，评估它们在洛伦兹吸引子轨迹上的长时程自回归预测性能。

**💡 创新点**

系统性分离并量化三种常见改进（加性注意力、双向上下文、Huber鲁棒损失）及CNN前置、TCN替代对长时程 chaotic 预测的影响，发现仅使用双向LSTM与Huber损失即可获得最优成绩。

**🔧 技术方法**

使用标准LSTM/ BiLSTM、Bahdanau加性注意力、Temporal Convolutional Network、1D CNN特征提取、Huber损失以及统一的滑动窗口训练与自回归推理流程。

**📊 数据集**

利用 AI‑DEEDS 2026 Chaotic Systems Challenge 公开数据集，包含多种 Lorenz‑63 类参数轨迹及其训练/评估对，覆盖长时程（1,000–10,000 步）与不同初始条件。

**📈 对比分析**

通过统一预处理、相同窗口长度、相同训练周期与学习率，对比各模型在 0–100 评分尺度下的排行榜成绩，最终发现 BiLSTM+Huber 最高得分 58.81，其他模型的分数在 45.72–58.81 之间。

**⚠️ 局限性**

受限于固定窗口长度、单一隐藏尺寸、固定训练周期、未使用物理先验或 delta 预测，以及仅实验加性注意力而非多头/缩放点积等变体，可能导致结果对更复杂架构或更大数据集的泛化受限。

---

## 376. Prompting Diffusion Models for Zero-Shot Instance Segmentation

**arXiv ID:** 2606.22660 | [PDF](https://arxiv.org/pdf/2606.22660v1)

**作者:** Irem Zeynep Alagöz `[一作]` (Technical University of Munich), Stefano Gasperini `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种基于扩散模型的交互式实例分割框架Prompt2Seg，能够直接利用空间提示（如点击点或置信图）控制分割结果。

**💡 创新点**

创新点在于将空间提示作为前馈输入注入冻结的扩散分割网络（通过轻量级ControlNet分支），实现了零样本、跨域的交互式分割，彻底摆脱了先前的后处理选择策略。

**🔧 技术方法**

技术包括：使用Stable Diffusion预训练的Gen2Seg作为主干，加入ControlNet分支来注入双尺度高斯空间提示，采用二维高斯和置信图等空间条件，保持主网络冻结，仅微调注入分支。

**📊 数据集**

仅在两大合成数据集Hypersim和Virtual KITTI 2上进行微调，随后在七个真实跨域数据集（COCO、DRAM、EgoHOS、PIDRay、Pascal VOC、HRSOD、ZeroWaste）上进行零样本评估。

**📈 对比分析**

与Gen2Seg、SimpleClick以及SAM等基线对比，Prompt2Seg在大多数数据集上均优于Gen2Seg和SimpleClick，在DRAM、PIDRay、ZeroWaste等长尾/跨域场景甚至超过SAM；在COCO大目标上表现突出，而在小目标上仍略逊于SAM。

**⚠️ 局限性**

局限性主要体现在对小目标的分割精度不足（受限于扩散模型的空间压缩），以及对单一提示难以精确控制分割粒度，需要多提示或层次化策略进一步改进。

---

## 377. Design and Development of a Neuromorphic Silicon Suite: PVT Sensing, Stochastic LIF Inference, On-Chip STDP Learning, and Crossbar Programming

**arXiv ID:** 2606.22635 | [PDF](https://arxiv.org/pdf/2606.22635v1)

**作者:** Poornima Kumaresan `[一作]`, Santhosh Sivasubramani `[通讯]` (Indian Institute of Technology Delhi)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c`

**🎯 论文内容**

在SkyWater 130 nm工艺上实现并验证了四个接口兼容的神经形态块：PVT传感器、随机LIF神经元、STDP学习控制器和跨导阵列编程控制器，并通过共享SPI寄存器接口实现统一配置。

**💡 创新点**

首次将这四类功能集成到单一、可共享的寄存器模型和统一验证流程中，公开RTL、布局及验证脚本，适配Tiny Tapeout共享硅项目。

**🔧 技术方法**

使用标准单元CMOS、线性反馈移位寄存器、TRNG、可编程激活表、STDP查找表、脉冲宽度/电压控制的跨导阵列编程，以及OpenLane/OpenROAD开源EDA流水线。

**📊 数据集**

论文未使用实际数据集，全部验证基于cocotb脚本的定向测试和行为仿真。

**📈 对比分析**

通过仿真和布局报告显示：每块占用单一Tiny Tapeout 1×1瓷砖，利用率61–70%，满足50 MHz时序；与其他开源/闭源神经形态硬件相比，功能覆盖更广，但未给出硅测量性能数据。

**⚠️ 局限性**

局限在于仅为预硅结果，缺乏真实硅测量、功耗/时序的实际验证，且未演示完整系统级集成或对不同器件阵列的真实编程效果。

---

## 378. Learning Adaptive Dynamical Features via Multi-$τ$ Liquid-Mamba for All-in-one Image Restoration

**arXiv ID:** 2606.22801 | [PDF](https://arxiv.org/pdf/2606.22801v1)

**作者:** Hu Gao `[一作]` (Shanghai Jiao Tong University), Lizhuang Ma `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种多时间尺度液态 Mamba 模块，能够自适应地调节状态空间模型的演化速度，实现对多种图像退化的统一恢复；

**💡 创新点**

创新点在于将输入相关的多时钟液态离散化引入选择性状态空间建模，既保留了 Mamba 的线性时间复杂度，又通过多分支时钟与加权融合实现对空间异质退化的动态适配；

**🔧 技术方法**

采用了选择性状态空间模型、液态时间常数网络（LTC）以及多分支时间尺度调制和自适应融合机制，并在此基础上构建了多尺度 U‑形编码解码网络 MLMIR；

**📊 数据集**

在多种退化数据集上进行评估，包括雨、雪、雾、模糊、噪声等任务的公开数据集（如 Rain100、Snow100K、RESIDE、GoPro、DIV2K 等），同时构建了统一训练集（R+S+H+B+N）并在真实世界数据集（RealRain、RTTS、SIDD、UIEB、C60 等）上进行零样本测试；

**📈 对比分析**

与多种基线（包括 MambaIR、ACL、Defusion、Perceive‑IR 等）在单任务、对齐任务和全局一体化任务上进行对比，实验显示 MLMIR 在 PSNR、SSIM、LPIPS 等指标上均取得显著提升（全局一体化平均 PSNR 超过 34 dB，单任务均衡提升 1–2 dB），并在多重退化与未知退化场景下表现出更强的泛化能力；

**⚠️ 局限性**

局限性包括：多时间尺度参数设计仍需手工调节，缺乏对极端或未知退化的显式物理解释，以及在时序一致性（视频）任务上的验证尚未展开。

---

## 379. Does the Same Token Mean the Same State? MoE Routing as Signal for Reasoning Control

**arXiv ID:** 2606.22798 | [PDF](https://arxiv.org/pdf/2606.22798v1)

**作者:** Kang Chen `[一作]` (Fudan University), Yugang Jiang `[通讯]` (Fudan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究一种无需读取答案字符串的多路径推理选取方法——Routing Agreement Decoding (RAD)，通过对稀疏 MoE 模型在固定锚点处的路由向量进行聚类，选取路由密度最高的轨迹作为最终答案。

**💡 创新点**

创新点在于：①利用同一 token 的不同 MoE 路由信息捕获上下文差异；②发现路由邻域与最终答案池高度相关，尤其在边界/分隔锚点；③提出 RAD 这一纯路由一致性选取器，可在无法进行字符串投票的场景（代码生成、Agent 交互）下直接进行决策。

**🔧 技术方法**

技术包括：稀疏 MoE 路由捕获、anchor‑conditioned 路由窗口、Weighted‑Jaccard 相似度、k‑NN 密度选择、以及在 Agent 场景下的边界锚点聚合与二值化内核。

**📊 数据集**

实验数据集覆盖 6 个任务：数学题（AIME24/25、BRUMO25、HMMT25）、GPQA、代码生成 LiveCodeBench 以及 SWE‑bench Verified 的 agentic patch 选择；使用 10 种稀疏 MoE 语言模型配置。

**📈 对比分析**

与 baseline（随机、文本自一致性 Majority、DeepConf 投票）对比，RAD 在数学/GPQA 场景与 Majority 接近，且在代码生成和 Agent patch 选择中显著优于随机，尤其在无法进行字符串投票的场景下实现了直接 pass@1 选取。

**⚠️ 局限性**

局限性：仅适用于可访问 MoE 路由信息的模型；仍是“共识”而非真值验证器，密集路由簇可能错误；在稠密模型或闭源 API 下不可用；在最优场景中提升幅度有限，且与 Majority 的差异统计上不显著。

---

## 380. A Formula-Driven Survey and Research Agenda for On-Policy Distillation

**arXiv ID:** 2606.22793 | [PDF](https://arxiv.org/pdf/2606.22793v1)

**作者:** Bowen Zhang `[一作]` `[通讯]` (Tsinghua University), Bowen Zhang (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对大语言模型（LLM）后训练中的 On‑Policy Distillation (OPD) 进行系统性综述，提出基于公式的通用分类法（feedback‑to‑update 视角），并将 OPD 的关键决策拆解为状态来源、反馈来源、支持、临时信用、门控、词汇路由、更新路径和正则化等变量；同时整理了现有方法的实现细节、诊断手段与稳定策略，并给出了未来研究的开放议题。

**💡 创新点**

创新点主要有：
1) 将 OPD 视作“反馈‑to‑update”问题而非单一损失函数，从公式出发统一不同方法的变量；
2) 将临时信用与词汇路由拆分为两个独立变量，提出 GAE‑OPD（基于优势估计的临时信用）和 CR‑OPD（对负优势的显式词汇路由）两种新假设；
3) 提供一套诊断与调试清单，帮助实践者在训练前后检查状态兼容性、支持覆盖、偏置/方差、门控合理性等；
4) 归纳工业实现与框架接口（如 SWIFT、verl、TRL 等）中的可配置项，形成“OPD 设计模板”。

**🔧 技术方法**

技术与方法：
- 公式推导与变量归纳（直接损失与策略梯度两大路线）；
- Temporal credit 估计（即时、累计、折扣、GAE 等）；
- 支持与词汇路由（top‑k、全词表、重排等）；
- 正则化与稳定化技术（PPO clipping、reference KL、长度/熵约束等）；
- 诊断手段（状态兼容性检查、教师熵/不确定度、负优势分布、长度/重复监测等）。

**📊 数据集**

数据集：
本文并未在单一数据集上开展实验，而是引用了众多公开论文与工业报告中的数据集，如 Qwen3、MiMo、GLM‑5、DeepSeek‑V4 等。综述主要聚焦于方法原理与实现细节，未进行统一数据集评测。

**📈 对比分析**

比较与性能：
- 通过对现有 OPD 方案的变量映射，文中对比了直接 KL、策略梯度、采样‑token、top‑k、full‑vocab 等不同实现的优势与局限；
- 讨论了临时信用与词汇路由的不同估计器对梯度方差与目标偏差的影响；
- 没有给出统一的实验结果，而是通过案例研究（如 Qwen3、DeepSeek‑V4 等）说明各类实现在实际工业场景中的表现与调优要点。

**⚠️ 局限性**

局限性：
1) 作为综述，缺乏统一的实验验证和性能基准；
2) 公式化的分类法虽通用，但在极端异构环境（跨 tokenizer、跨 modality）下仍需更多实证支持；
3) 讨论的动态路由与自适应支持等方向尚未被完整实现或系统化；
4) 对教师可靠性、支持覆盖等诊断方法的阈值与自动化程度仍待进一步研究。

---

## 381. The Origins of Stochasticity: Comprehensive Investigations on Uncertainty Quantification for Large Language Models

**arXiv ID:** 2606.22792 | [PDF](https://arxiv.org/pdf/2606.22792v1)

**作者:** Xiang-Jun Ou `[一作]` (Nanjing University), Shao-Qun Zhang `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了LLM不确定性量化方法的统一分类和评估框架，并在问答、数学推理与代码生成等任务上统一比较了21种UQ方法。

**💡 创新点**

创新点在于构建统一的UQ方法学体系、定义三类评估指标并对多种LLM进行统一实验，揭示了推理链和模型规模对UQ效果的影响。

**🔧 技术方法**

采用多种技术包括贝叶斯采样、蒙特卡罗、Token概率分析、语义一致性与图结构一致性等，并引入CoT-UQ、Topo-UQ等专门的推理链UQ方法。

**📊 数据集**

使用了TriviaQA、GSM8K与HumanEval等公开数据集。

**📈 对比分析**

通过AUROC、AUPRC、AURC等指标对方法进行横向比较，发现推理链模式提升QA与数学推理任务的UQ性能，而代码生成任务效果下降；Qwen3系列模型在多项指标上优于Llama3.2。

**⚠️ 局限性**

限制在于实验仅覆盖21种UQ方法，缺乏对更大规模模型的评估，且推理链对代码生成任务产生负面影响，未深入探讨与实际下游应用的集成。

---

## 382. Learning Filters with Certainty

**arXiv ID:** 2606.22786 | [PDF](https://arxiv.org/pdf/2606.22786v1)

**作者:** Yuval Banoun `[一作]` (Technion), Ori Rottenstreich `[通讯]` (Technion)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究了利用计数布隆过滤器（CBF）提供的计数器信息来估计正向查询的后验置信度，并在此基础上设计了四种结合学习模型与CBF的置信度感知架构；

**💡 创新点**

创新点在于将CBF的计数器乘积转化为后验概率，利用该置信度信号与学习模型的输出相结合，提出了四种不同的置信度感知学习布隆过滤器（Model 1–4），从而实现对误报率和推理成本的更细粒度调控；

**🔧 技术方法**

采用计数布隆过滤器、传统布隆过滤器、学习模型（可为任意概率分类器）以及基于后验概率的阈值决策规则；

**📊 数据集**

论文未给出具体应用数据集，主要通过模拟实验展示计数器乘积与后验概率的关系以及不同模型下的误报率；

**📈 对比分析**

通过理论误报率分析与模拟实验与传统学习布隆过滤器、沙盒学习布隆过滤器进行对比，结果表明在保证低误报率的同时，置信度感知模型可降低推理成本或内存占用；

**⚠️ 局限性**

主要限制包括：对动态集合更新时需周期性重新训练学习模型；对阈值选择和后验阈值的敏感性；以及在大规模部署时的内存与推理延迟权衡尚未给出最优方案。

---

## 383. Explainable AI for Mental Health Prediction in Drug-Affected Populations with Dragonfly Algorithm and GAN Oversampling

**arXiv ID:** 2606.22780 | [PDF](https://arxiv.org/pdf/2606.22780v1)

**作者:** Ahnaf Atef Choudhury `[一作]` (George Mason University), Abdullah Al Mamun `[通讯]` (Dhaka University of Engineering and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了可解释的多类别精神健康预测模型，针对药物影响人群进行早期风险预测。

**💡 创新点**

创新点在于将PCA-IG混合特征选择、GAN过采样与龙虬算法优化XGBoost相结合，并采用SHAP实现实例级解释。

**🔧 技术方法**

使用技术包括PCA、信息增益特征选择、GAN、Dragonfly Algorithm、XGBoost、SMOTE、SVM‑SMOTE、SHAP。

**📊 数据集**

使用的数据集为孟加拉国“Insights into Drug Addiction in Bangladesh”多维数据集，包含36个分类特征与三类精神健康标签。

**📈 对比分析**

通过与LR、RF、GB、LGBM、ANN、未优化XGBoost以及SMOTE、SVM‑SMOTE、GAN过采样等模型比较，DA‑optimized XGBoost+GAN实现94.17%准确率、93.80%加权F1，优于传统基线。

**⚠️ 局限性**

局限在于样本量有限、缺乏跨国和多模态数据，模型泛化性待进一步验证。

---

## 384. Language-Specific Sentiment Polarity Biases in Encoder and Large Language Model Classification of Product Reviews

**arXiv ID:** 2606.22745 | [PDF](https://arxiv.org/pdf/2606.22745v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 385. HAKARI-Bench: A Lightweight Benchmark for Comparing Retrieval Architectures and Efficiency Settings under Unified Conditions

**arXiv ID:** 2606.22778 | [PDF](https://arxiv.org/pdf/2606.22778v1)

**作者:** Yuichi Tateno `[一作]` `[通讯]`, Yuichi Tateno

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 HAKARI‑Bench，一个轻量级的多语种、多域检索评测基础设施，利用 Nano‑sets 将现有大规模检索基准压缩为数千条查询与文档，并统一提供候选生成、重排序、维度压缩、量化、稀疏裁剪等多种评测配置。

**💡 创新点**

创新点：① 在不牺牲排名可靠性的前提下，通过 Nano‑sets 实现 35 份基准、551 任务的轻量化；② 在同一条件下对 BM25、dense、sparse、late‑interaction 与 reranker 5 大检索架构进行统一对比；③ 在同一任务集上系统评估 Matryoshka 维度压缩、int8/二值量化、重排序等效率与质量的 Pareto 前沿；④ 通过高 Spearman/ Pearson 相关性验证 Nano‑sets 能够复现全量基准排名。

**🔧 技术方法**

技术与方法：使用 BM25、Bi‑Encoder、SPLADE、ColBERT、Cross‑Encoder / LLM‑style reranker；Matryoshka 维度截断、scalar/二值量化、float 重新评分；RRF 融合构建固定候选集；nDCG@10、Recall@10 等检索指标；结果存储于 DuckDB，配合多轴排行榜可视化。

**📊 数据集**

数据集：整合 BEIR、MTEB、MMTEB、MIRACL、NanoBEIR、CoIR、LongEmbed、IFIR、ChemTEB、R2MED、BIRCO、BRIGHT、RARb、RTEB、BuiltBench、CodeRAG 等 35 份基准，覆盖 43 种语言、约 500 任务、10K 文档/任务、200 条查询。

**📈 对比分析**

比较方法：在统一的任务格式下，所有模型按同一参数、同一候选集进行评测；使用宏平均（按基准均衡）和微平均（按任务均衡）两种聚合；对比全量基准得到 Spearman >0.97、Pearson >0.97；性能差异可达 50 以上点；维度压缩、量化导致平均下降 1–6 点，重排序可在特定范围内超过 dense；展示了不同模型在不同语言/域/长度等维度的优势与劣势。

**⚠️ 局限性**

局限性：Nano‑sets 缺乏原始基准的硬负样本与大规模检索空间，导致绝对分数与难度偏差；候选集固定且存在 lexically 相关偏倚；量化仅为简单后处理，未覆盖生产级 ANN 技术；未评估推理速度与硬件依赖；覆盖模型仅约 1B 参数或更小，缺少大规模/商业 API；潜在数据泄露与提示不一致问题；评测噪声导致细微分数差异不可靠。

---

## 386. Statistical Matching via Schrödinger Bridge beyond Conditional Independence

**arXiv ID:** 2606.22770 | [PDF](https://arxiv.org/pdf/2606.22770v1)

**作者:** Eunho Koo `[一作]` (Chonnam National University), Jinwon Sohn `[通讯]` (University of Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于Schrödinger桥的预测统计匹配方法，能够在不违反已观测边缘分布的前提下，利用兼容成本挖掘不同数据库之间隐藏的 Y–Z 相关性，实现双向缺失变量的概率填补。

**💡 创新点**

创新点在于将条件独立假设（CIA）视为基准，借助熵正则化的Schrödinger桥进行指数调制，从而在保持边缘一致性的同时恢复潜在的 Y–Z 依赖，并提供理论上在高斯场景下的完全恢复保证。

**🔧 技术方法**

技术主要包括熵正则化最优传输、Schrödinger桥的潜在函数估计、利用神经网络逼近潜在函数并通过统计风险最小化训练，以及通过后验采样实现多重插补。

**📊 数据集**

实验使用了合成高斯和非线性回归数据集、CelebA 图像数据（VAE 8 维潜在空间、二元目标与 32 状态辅助变量）以及 Adult 收入数据（K‑means 聚类得到 32 类辅助变量），验证方法的有效性。

**📈 对比分析**

与传统 CIA 基线、最优传输基线以及多种混合匹配方法相比，本文方法在 Gaussian、非线性回归以及真实数据上均实现了更高的下游预测指标（如 R²、AUC）并显著降低了填补误差，尤其在 Y–Z 相关性强的场景下提升明显。

**⚠️ 局限性**

局限性包括：对兼容成本和正则化参数的依赖；理论完全恢复仅在高斯假设下成立；若 Y–Z 依赖弱或样本不足时可能不优于传统 CIA；在敏感属性整合时需关注潜在公平性与偏差放大问题。

---

## 387. PA-User: Simulating Trust and Verification under AI-Generated Content

**arXiv ID:** 2606.22738 | [PDF](https://arxiv.org/pdf/2606.22738v1)

**作者:** Saber Zerhoudi `[一作]` `[通讯]` (University of Passau), Saber Zerhoudi (University of Passau)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了PA-User，一种能够感知文档来源的用户模拟器，融合检测努力预算、贝塔信任更新和策略验证决策；

**💡 创新点**

创新点在于在传统点击/认知模拟器中加入了三大机制——检测努力预算、来源类贝塔信任估计以及基于信任、努力和风险的验证策略，从而弥补了AI生成内容的“命名缺口”；

**🔧 技术方法**

使用贝塔分布的贝叶斯更新、期望效用决策规则、对检测难度的逻辑回归估计以及与HC3数据集的仿真实验；

**📊 数据集**

使用HC3语料库（人类与ChatGPT答案的对照，涵盖五个领域），并利用其表面特征构建检测难度指标；

**📈 对比分析**

通过对比信任校准误差（TCE）、高风险领域的遗憾率、验证率和努力消耗等指标，PA-User在TCE为0.162（低于基线0.356），高风险遗憾率从0.171降至0.122，验证率为34.5%（相较无预算时的70.2%更为现实）；

**⚠️ 局限性**

局限性在于用户行为完全基于仿真，缺乏真实用户点击与验证日志；检测难度使用表面特征代理，可能无法完全捕捉真实AI检测；参数先验取自文献，可能不适用于所有情境。

---

## 388. Towards Fast Domain Adaptation and Fine-Grained User Simulation for Evaluating Conversational Recommender Systems

**arXiv ID:** 2606.22803 | [PDF](https://arxiv.org/pdf/2606.22803v1)

**作者:** Yuanzi Li `[一作]` (Renmin University of China), Huifeng Guo `[通讯]` (Huawei Noah’s Ark Lab)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出AdaptSim，改进LLM用户模拟器以更好地评估对话式推荐系统

**💡 创新点**

创新点在于自动提示优化、开放式动作生成以及“思考-回应”细粒度风格控制，和BFS级别对比评估框架

**🔧 技术方法**

采用自动提示优化（Meta-Prompting）、概率动作生成、思考-回应结构、BFS搜索等技术

**📊 数据集**

在食物、美容、购物三大领域数据集上，使用四个LLM后端（Qwen3-8B、GPT-4.1-mini、DeepSeek-R1、GPT-3.5-turbo）进行实验

**📈 对比分析**

通过与iEvaLM、CSHI、RecUserSim三种基线进行对比，AdaptSim在自然性、适应性、清晰度、相关性、角色扮演与真实性六维度多次赢率最高，且在风格控制与鲁棒性评估上表现更好

**⚠️ 局限性**

局限包括：对Meta-Prompting评估LLM的依赖导致循环瓶颈，贪心动作生成忽略稀有行为，风格控制在长对话中可能漂移，BFS搜索的指数增长限制了评估深度

---

## 389. Cross-National Information Attacks: A Two-Decade Analysis of Troll Behavior in Korea

**arXiv ID:** 2606.22785 | [PDF](https://arxiv.org/pdf/2606.22785v1)

**作者:** Jaehong Kim `[一作]` (Korea Advanced Institute of Science and Technology), Meeyoung Cha `[通讯]` (Max Planck Institute for Security and Privacy)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并部署了一套可解释的机器学习框架，对韩国Naver新闻评论区的112M条评论进行长期检测和分析，识别近2万名潜在的外国影响操作账户，并量化其策略、情绪与影响力。

**💡 创新点**

创新点包括：①三层级可解释标签（外部来源、道德情绪、目标国家）并给出span级解释；②利用GPT-4.1微调并对其进行知识蒸馏，得到约0.1B参数的轻量模型，兼顾可解释性与大规模推理；③将内容级输出聚合为用户级特征，实现从评论到账户的连贯检测与纵向策略分析；④结合时间序列、可见度与情绪强度的统计建模，为平台治理提供可操作的防御优先级。

**🔧 技术方法**

技术手段：GPT-4.1微调 + 0.1B KcELECTRA蒸馏模型；span级BIO标注；多层级分类（二分类、multi-label、span级多分类）；用户级特征构造（行为、网络、内容平均概率）与XGBoost、SVM、RF、LGB四种分类器；SHAP解释特征重要性；Fractional Logit回归分析情绪强度对点赞比例的影响；时间序列相关性检验与Wilcoxon/Williams检验。

**📊 数据集**

数据集：112,658,554条Naver新闻评论（2006‑2025），4,047,831用户；70个公开的“真实”木马账号；50,000条GPT注释候选样本；49,745条人工标注后GPT生成的gold标准样本；1,452条人工精确标注样本用于模型评估。

**📈 对比分析**

模型评估：在1,452条人工标注样本上，GPT微调平均F1=0.7206；蒸馏KcELECTRA在同一测试集上达到Macro F1≈0.722；用户级XGBoost在151用户（70木马+81非木马）上F1=0.94，SVM/RandomForest/LGB分别为0.91/0.93/0.93。与随机基线相比，内容级模型提升>30% F1；用户级模型在检测已知木马与新木马上均显著优于基线；策略分析表明“抨击韩国”策略的点赞比例显著高于“赞美外部国家”，并在关键选举期出现显著激增，验证模型对时间和情绪影响的捕捉能力。

**⚠️ 局限性**

局限性：①误报风险仍存在，尤其对非典型木马；②情绪与目标标签的主观性导致标注一致性中等；③仅覆盖Naver新闻平台，跨平台推广未知；④基于公开评论，可能受平台规则更改影响；⑤模型解释虽提供span级证据，但无法完全排除算法操纵或抗检测的适应性；⑥缺乏对真实政治后果（如选举结果）的直接因果验证。

---

## 390. Towards Robust Personalized Federated Learning: Vulnerability Assessment and Defense Co-Design

**arXiv ID:** 2606.22782 | [PDF](https://arxiv.org/pdf/2606.22782v1)

**作者:** Mingyuan Fan `[一作]` (East China Normal University), Cen Chen `[通讯]` (East China Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统评估了个性化联邦学习（PFL）在面对基于迁移的对抗攻击时的脆弱性，并提出一种轻量级防御框架以提升其鲁棒性。

**💡 创新点**

创新点在于：①首次从理论和实证层面揭示PFL易受迁移攻击的原因，并提出五个关键指标；②设计了三种可插拔的鲁棒性提升技术（随机输入噪声、输入尺度化迹正则化、参数敏感性最小化）；③在多种攻击方法和数据集上验证了方案的有效性。

**🔧 技术方法**

使用的技术包括：PGD、MI、PCIFGSM、MaskBlock等迁移攻击；对抗训练与随机平滑结合；梯度噪声估计、迹正则化、二阶梯度敏感性正则化等自研防御手段。

**📊 数据集**

实验数据集：CIFAR-10、CIFAR-100 与交通标志识别数据集 GTSRB。

**📈 对比分析**

通过与多种常见PFL方法（FedProx、SCAFFOLD、FedBN 等）以及现有对抗训练/随机平滑方案的对比，实验显示防御框架能将平均准确率下降（AD）从 30%+ 降低到 5%~15% 左右，同时在多数情况下保持或提升清洁数据准确率；训练开销仅略高于原始 PFL 方法。

**⚠️ 局限性**

局限性包括：仅在图像分类任务上验证；对其他任务（自然语言处理、语音识别等）的推广尚未探究；在使用对抗训练时仍存在精度-鲁棒性权衡；缺乏对通信延迟和多设备部署复杂度的深入分析。

---

## 391. Disk-Based Interval Indexes Under the Increasing Ending Time Assumption

**arXiv ID:** 2606.22773 | [PDF](https://arxiv.org/pdf/2606.22773v1)

**作者:** Kai Wang `[一作]` (Hong Kong University Of Science And Technology), Dimitris Papadias `[通讯]` (Hong Kong University Of Science And Technology)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出基于 IET（Increasing Ending Time）假设的磁盘级区间索引结构，并对其进行了改进。

**💡 创新点**

创新点包括：1）CEB 通过中心点排序减少可变节点；2）TIDE 采用持续时间排序显著减少底层树数量，实现更高压缩率和查询效率。

**🔧 技术方法**

技术上使用两层追加写 B+树，利用角结构（corner space）映射，支持插入、范围与计数查询。

**📊 数据集**

实验数据集包括纽约出租车（TAXI）、自行车共享（BIKE）、NFT 交易（NFT）和亚马逊评论（AMAZON）。

**📈 对比分析**

与现有 SEB、RI-tree 等进行对比，TIDE 在索引大小、插入速度和查询成本上均优于对手，尤其在计数查询上快数到数百倍。

**⚠️ 局限性**

局限性：仅适用于按结束时间递增的插入场景；在极端不规则数据上，CEB 仍产生大量底层树；RI-tree 在某些查询仍保持竞争力。

---

## 392. Learning Moral Diversity: Modelling Individual Perspectives in Moral Classification of Texts

**arXiv ID:** 2606.22771 | [PDF](https://arxiv.org/pdf/2606.22771v1)

**作者:** Yi Ren `[一作]` (Adelaide University), Matthew Roughan `[通讯]` (Adelaide University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在传统 BERT 微调的基础上引入 Annotator Layer，构建能够捕捉每个注释者对道德价值文本的个体判断的模型，并预测每个注释者的标注结果。

**💡 创新点**

创新点在于把道德判断视为高度主观的任务，利用 Annotator Layer 学习注释者特定的偏置和标注模式，证明注释者间的分歧是可学习特征而非噪声，并提供可解释的偏置矩阵揭示不同注释者的道德视角。

**🔧 技术方法**

使用预训练的 BERT 作为文本编码器，在其基础上加入两种 Annotator Layer（Bias‑Only 与 Linear Transformation）进行微调，采用交叉熵损失并加权 L2 正则与中心化偏置惩罚，同时通过 Krippendorff α 量化标注一致性。

**📊 数据集**

实验基于 Moral Foundations Twitter Corpus（MFTC），包含 35,108 条推文、23 名注释者以及五个道德基础（Authority、Care、Fairness、Loyalty、Purity）以及其正面/负面/缺失标签。

**📈 对比分析**

与单纯微调 BERT 的基线模型比较，Annotator Layer 在 raw annotation 上实现宏 F1 提升 10.2%（Linear Transformation 版本最佳），但在聚合标签评估中 Linear Transformation 版本表现下降，显示模型更适合捕捉个体视角；绑定型基础（Authority、Loyalty、Purity）对 Annotator Layer 的收益最大。

**⚠️ 局限性**

局限性包括：未利用注释者的身份/政治/道德测量等元数据，导致难以将学习到的偏置与已知特征关联；Annotator Layer 需要注释者信息才能得到准确预测，难以直接生成高质量的聚合标签，尤其在标注稀疏且不均匀的情况下，简单平均会引入噪声。

---

## 393. Factored Gossip DiLoCo: Reducing Blocking Communication in DiLoCo

**arXiv ID:** 2606.22768 | [PDF](https://arxiv.org/pdf/2606.22768v1)

**作者:** Chamin Hewa Koneputugodage `[一作]` (Pluralis Research), Alexander Long `[通讯]` (Pluralis Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种将DiLoCo外同步拆分为非阻塞Mix1和阻塞Mix2的方式，以实现通信与计算的重叠并保持模型一致性。

**💡 创新点**

创新点在于将全局同步拆分为可重叠的非阻塞参数混合和可选的阻塞梯度混合，实现了高利用率与训练稳定性的权衡；同时引入JS距离度量以更精确评估一致性。

**🔧 技术方法**

采用Gossip/全局平均混合、AdamW/SGD + Nesterov外部优化、可变Mix1/Mix2策略、模拟网络失败、计算利用率仿真与L2/JS一致性监测。

**📊 数据集**

使用1.5B LLaMA‑3模型在FineWeb数据集上训练，另外在160M WikiText与600M FineWeb等规模上进行缩放实验。

**📈 对比分析**

与Sync‑DP和原DiLoCo对比，Factored DiLoCo在10B–30B token下在低带宽(100/200Mbps)环境中显著提升计算利用率（最高达100%），验证困惑度仅略高于DiLoCo，同时在壁钟时间上比DiLoCo快约30%+；在失败率下表现更稳健。

**⚠️ 局限性**

局限包括：需手动选择Mix2参数和块子集；理论假设简化，缺乏对更复杂优化器和非线性模型的严格分析；对大规模多GPU/多机环境的验证不足；尚未实现完整容错与压缩等进一步改进。

---

## 394. One-Step Flow Matching for Generative Modeling of Path-Dependent Physical Fields

**arXiv ID:** 2606.22752 | [PDF](https://arxiv.org/pdf/2606.22752v1)

**作者:** Yijing Zhou `[一作]` (University of British Columbia), Jasmin Jelovica `[通讯]` (University of British Columbia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于流匹配与时空 DiT 的生成模型，用一次性采样即可生成随机几何和加载路径下的塑性 von Mises 应力场。

**💡 创新点**

①使用非高斯几何经验分布作为流匹配源，显著减少条件路径交叉；②引入 token‑level 加载嵌入，提升对路径依赖的学习；③结合 VAE 降维与 ViT‑Transformer 辅助网络；④在高分辨率 256×256、20 帧的场景下实现一次采样。

**🔧 技术方法**

流匹配（Conditional Flow Matching）、时空 DiT Transformer、token‑level 加载嵌入、科学 VAE、ViT‑Transformer、Adam 优化、O(1) 采样。

**📊 数据集**

20,000 条二维 von Mises 应力场样本（每个 20 帧、256×256），包含单圆孔、3 孔、6 孔三类随机几何，并配以随机加载‑卸载路径，均由 ABAQUS 生成。

**📈 对比分析**

与传统 FEM（CPU）和扩散模型对比：单步采样在 CPU 上提升 6.61×、GPU 上提升 127×；5 步、20 步多步采样虽能略微降低 MAE（单孔 MAE 1.05→0.924），但仍保持显著速度优势；相较于需 100+ 步的扩散模型，本方法实现一次采样即达到可接受精度。

**⚠️ 局限性**

①数据集规模相对有限，难以覆盖更复杂几何与材料；②单步采样仍需 VAE 20 次编码/解码，导致部分耗时；③仅针对均质塑性材料，未验证异质材料或流体系统；④缺乏显式物理约束，未来需进一步提升对极端梯度区域的预测。

---

## 395. RaysUp: Ultra-light Universal Feature Upsampling via Geometry-Aware Ray Representation

**arXiv ID:** 2606.22749 | [PDF](https://arxiv.org/pdf/2606.22749v1)

**作者:** Yuchuan Ding `[一作]` (Tongji University), Ying Shen `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了RaysUp，一种极轻量级、任务无关、编码器无关的通用特征上采样框架，能够在任意分辨率下重建Vision Foundation Model（VFM）的低分辨率特征，并保持高语义一致性与几何一致性。

**💡 创新点**

创新点在于：①使用空间分离的引导编码器提取方向感知的语义引导；②设计任意分辨率交叉注意力机制，实现跨尺度的自适应重建；③引入基于Plücker射线坐标的Ray Positional Encoding（RayPE），将二维上采样迁移至射线域，显著提升边界精度与结构保真；④构建几何感知邻域注意力模块，在局部范围内进行内容自适应聚合。

**🔧 技术方法**

技术实现包括：轻量化方向感知引导编码器、任意分辨率交叉注意力、RayPE、基于Plücker坐标的射线编码、几何感知邻域注意力以及与RoPE相结合的旋转编码。

**📊 数据集**

主要使用的评估数据集有：ImageNet（预训练），COCO-Stuff、ADE20K、Pascal-VOC、Cityscapes（语义分割）；NYUv2（深度与法向估计）；DAVIS（视频目标分割）；以及多模型对比的DINOv2、DINOv3、SigLIP2、PE Spatial等。

**📈 对比分析**

与传统双线性插值、FeatUp、LoftUp、JAFAR、AnyUp等方法对比，RaysUp在语义分割、深度/法向估计、视频目标分割和开放词汇分割等任务中均取得或接近最优的mIoU、RMSE、δ1等指标，同时参数仅为AnyUp的16%，推理速度提升约7倍，显著提升准确率与效率的平衡。

**⚠️ 局限性**

局限性包括：仍依赖VFM输出的低分辨率特征，极高分辨率下性能提升有限；RayPE在缺乏准确信息的摄像机姿态时效果受限；在非常稀疏或大尺度场景中可能出现结构漂移或细节损失。

---

## 396. Error Highways: Scaling Predictive Coding to Very Deep Networks

**arXiv ID:** 2606.22744 | [PDF](https://arxiv.org/pdf/2606.22744v1)

**作者:** Amirhossein Mohammadi `[一作]` (SingularityNET), Alexander G. Ororbia `[通讯]` (Rochester Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了高速错误传播（HEP）机制，将预测编码网络（PCN）中隐藏层与输出误差通过反馈矩阵直接耦合，以在深层网络中保持学习信号不随深度衰减，从而实现超深PCN的训练。

**💡 创新点**

创新点在于：①通过在PCN的推理步骤中加入“高速错误通道”来直接向内部层注入输出误差；②在自由能中添加双线性高架项，保持学习规则本地化；③在保持局部性和可解释性的前提下，解决了PCN深度衰减问题。

**🔧 技术方法**

使用的技术包括：预测编码框架、局部Hebbian权重更新、RMS归一化以稳定前向传播、固定随机反馈矩阵（类似DFA）以及梯度下降或Adam进行状态更新。

**📊 数据集**

实验数据集为MNIST和Fashion‑MNIST，用于评估在4–128层的多层感知机（MLP）上的性能。

**📈 对比分析**

通过与原始PC（无HEP）和标准反向传播（BP）在相同无跳跃连接的MLP上对比。HEP在8层以上显著提升准确率，8层时为96%+，深达128层时仍保持95%+（MNIST）/82%+（Fashion‑MNIST），相比之下原始PC在深层退化到随机水平，而BP在无跳跃时亦出现梯度消失。

**⚠️ 局限性**

局限性包括：①HEP需要额外的反馈矩阵开销；②反馈矩阵固定且未学习，可能限制表达能力；③在更复杂的任务或更大模型上验证不足；④稀疏高速通道虽可减少开销，但需仔细调节间距以避免学习信号再次衰减。

---

## 397. GRADE: Graph Representation of LLM Agent Dependency and Execution

**arXiv ID:** 2606.22741 | [PDF](https://arxiv.org/pdf/2606.22741v1)

**作者:** Yue Zhao `[一作]` `[通讯]` (University of Southern California), Yue Zhao (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种统一的两层图表示法，用于捕捉LLM代理的执行流程和依赖关系，并通过对依赖边的来源（观测、声明、推断）进行分级，进而实现对代理运行失败的预测与定位。

**💡 创新点**

创新点在于：①将执行层（由执行轨迹直接获得）与依赖层（需通过观测、声明或推断获取）合并到同一节点集合上；②对依赖边引入“来源分级”，区分真实依赖与基于假设的推断边；③提出“边缘提升”（marginal lift）评估指标，量化依赖层相对于仅基于运行规模的基线所带来的增益；④在多领域代理数据集上展示该表示法的跨类可迁移性与优于传统GNN的表现。

**🔧 技术方法**

使用的技术包括：类型化有向时间多重图、执行与依赖两层边的构造、边缘来源分级（观测/声明/推断）、边缘来源饱和度（ρ）判别、特征化依赖图形状（链深度、中心性、稠密度等）、逻辑回归与ROC‑AUC评估、留一语料转移实验、GIN、R‑GCN、HGT等图神经网络的对比实验。

**📊 数据集**

使用了六个公开代理跑数据集：tau‑bench、tau2‑bench（数据库工具使用）、SWE‑agent、SWE‑Gym、OpenHands（软件工程）、AgentRewardBench（Web 导航）。每个数据集提供已标注的运行失败标签。

**📈 对比分析**

方法比较：首先用仅运行规模（步骤、工具调用等计数）的平面模型作为基线；然后在此基础上加入归一化后的依赖层特征，计算ROC‑AUC提升。实验显示，在运行规模预测弱的三类数据集上依赖层可获得显著提升（平均提升≈0.07），而在规模强的三类中提升为零或负；在留一语料转移实验中，依赖层特征在所有被测试的类上均保持在50%以上且多数显著优于随机；相比之下，传统的无边缘来源信息的GNN在跨类转移时性能下降甚至低于随机。总体而言，依赖层实现了可迁移的失败预测，且对执行层做出定位贡献。

**⚠️ 局限性**

局限性包括：①依赖层的构建在很大程度上依赖于观测到的读写日志，缺失的依赖会被推断或缺失，导致信息不完整；②当前实验仅涵盖编码、数据库和单一Web类任务，缺乏更复杂、多模态代理的验证；③对大规模或实时代理的在线评估尚未展开；④图神经网络的表现受限于未显式使用边缘来源分级，未来需设计更具偏置的网络或额外的仪表化来进一步提升性能。

---

## 398. Text Dictates, Music Decorates: Energy-based Attention for Editable Dance Motion Generation

**arXiv ID:** 2606.22726 | [PDF](https://arxiv.org/pdf/2606.22726v1)

**作者:** Seong Jong Yoo `[一作]` (University of Maryland), Cornelia Fermüller `[通讯]` (University of Maryland)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 STREAM，一个把文本语义与音乐节奏分离的扩散 Transformer，用于可编辑的舞蹈运动生成，并创建了包含帧级舞蹈技术标签与文本描述的 Motorica++ 数据集，同时引入了 Exchange Evaluation Protocol 和 Editable Dance Score 评估方法。

**💡 创新点**

核心创新在于 Bimodal Energy-based Attention Module（BEAM），通过 Adaptive Layer Normalization 控制运动结构、双能量交叉注意力保证音乐节拍对齐，并用贝叶斯更新细化文本语义；此外，还首次给出针对文本与音乐双模态的可编辑性评估指标 EDS。

**🔧 技术方法**

技术手段包括扩散模型（DDPM）+ Transformer，BEAM（Text-AdaLN、Dual Energy-based Cross Attention、Bayesian Update），能量模型视角的注意力设计，分层 Classifier-Free Guidance，CLIP 文本编码、Jukebox 音频特征提取，以及多任务损失（FID、Dist、BAS、EDS）。

**📊 数据集**

使用 Motorica++（97 条带有舞蹈技术标签与文本描述的 5 秒片段）和 AIST++（无文本，仅音乐）进行训练与评估，并与 DanceRevolution、FineDance、DanceRemix 等公开数据集对比。

**📈 对比分析**

通过在 AIST++ 和 Motorica++ 上与现有最优模型（EDGE、POPDG、DanceFusion、TM2D 等）进行对照，使用 FID_k、FID_g、Dist_k、Dist_g、BAS 以及新指标 EDS 评估。STREAM 在 BAS 与 EDS 上均显著优于基线，同时在语义控制上保持高质量表现。

**⚠️ 局限性**

目前仅支持单人舞蹈生成，缺乏多舞者空间与互动建模，且模型复杂度高，泛化到不同舞蹈风格和长时序生成仍面临挑战。

---

## 399. Machine-knittable, Magnetically-Plug-n-Play E-Textile Prototyping

**arXiv ID:** 2606.22800 | [PDF](https://arxiv.org/pdf/2606.22800v1)

**作者:** Yifan Li `[一作]` (The University of Tokyo), Yoshihiro Kawahara `[通讯]` (The University of Tokyo)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Plug‑n‑play e‑knit 平台，利用工业针织集成差分 I²C 线路和软磁连接实现可多次插拔、快速调整的可扩展电子纺织原型。

**💡 创新点**

创新点在于：①工业数字针织一次性完成大面积电路嵌入；②软磁弹簧连接使传感模块可百次以上插拔而不损伤织物；③基于 LED 颜色与亮度的自动定位识别，实现无人工布线的快速网络配置。

**🔧 技术方法**

技术包括：工业数字针织、差分 I²C 通信、PME 软磁弹簧连接、外部摄像头 LED 颜色识别、低阻导线、差分 I²C 信号抑噪、可伸缩的柔性 PCB 传感模块。

**📊 数据集**

主要使用自制实验数据：前臂运动捕捉（与 OptiTrack 对比）和室内温度映射（与单壁传感器对比）。未使用公开数据集。

**📈 对比分析**

与现有模块化纺织（>40% 覆盖率，<10 传感器）和 I²We（>70% 覆盖率，<10 传感器）相比，Plug‑n‑play e‑knit 覆盖率>70%，支持>20 传感器；差分 I²C 在长达150 cm 传输下仍能可靠解码；运动实验中 MPJRE 8.24（最优位置）优于其他配置，温度实验显示可实现位置感知温度曲线，优于单壁传感器。

**⚠️ 局限性**

局限包括：裸露导电通道易在高弯曲区短路；软磁连接在剧烈运动（如跳跃）时可能脱落；颜色识别受光照影响，仅能识别约 24 个模块；磁弹簧力与服装柔性之间存在权衡，需要进一步优化。

---

## 400. Integrating Heterogeneous Digital Twins in Federated Ecosystems

**arXiv ID:** 2606.22791 | [PDF](https://arxiv.org/pdf/2606.22791v1)

**作者:** Christian Vergara-Marcillo `[一作]` (Southern University of Science and Technology), Georgios Theodoropoulos `[通讯]` (Southern University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了Federation Node Manager，用于在Federated Digital Twin生态系统中整合异构DT，支持控制能力暴露、协议与模式适配以及实时状态事件交互，以实现智能出行紧急响应中的交叉点信号预emption协调。

**💡 创新点**

提出了一个模块化边界集成机制 Federation Node Manager，能够在不暴露内部DT逻辑的前提下实现异构DT的互操作性与运行时协调，填补了从概念到实现的空白，并在智能交通中验证了其效果。

**🔧 技术方法**

采用Python实现节点管理器，使用MQTT与HTTP等协议进行通信，利用协议适配器、模式中介器、状态/事件管理器等模块；使用SUMO仿真器模拟交通网络；并基于JSON/CSV等格式进行数据交互。

**📊 数据集**

采用Madrid城市子网络的SUMO交通仿真数据，生成的车辆流量为0.5K、1K、1.5K车辆的三种拥堵场景；包括五条随机急救车路线。

**📈 对比分析**

对比三种模式：固定时间控制(FTCM)、本地可互操作预emption(LIDP)和联邦协调预emption(FCDP)。结果显示FCDP平均行驶时间比FTCM下降32.3%，比LIDP下降8.85%；并且节点处理延迟不到1ms，网络延迟在168-441ms，验证了低延迟与高效协调。

**⚠️ 局限性**

只在单一智能交通场景验证，未扩展到更大规模网络；节点间的安全与策略治理仅在本地实现，组织层面的治理和隐私保护需在更大层面补充；实验基于SUMO仿真，真实世界部署仍需验证。

---

## 401. Scaling Audio Models Efficiently: A Joint Study of Compute Constraints and Optimization Behavior

**arXiv ID:** 2606.22790 | [PDF](https://arxiv.org/pdf/2606.22790v1)

**作者:** Vyom Agarwal `[一作]` (University of Maryland), Jerry Wu `[通讯]` (University of Maryland)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

系统研究了在自动语音识别（ASR）和语音情感识别（SER）任务中，如何在固定计算预算下通过调整模型规模、输入长度和表示分辨率来实现最佳性能。

**💡 创新点**

创新点在于提出统一的三维计算轴框架，并结合 LoRA 与 DAMA 的参数高效微调，揭示不同任务对各计算轴的优先级差异；同时在 ASR 与 SER 上绘制了完整的 Pareto 前沿。

**🔧 技术方法**

使用的技术包括 Whisper 与 wav2vec2 预训练模型，LoRA（低秩适配）与 DAMA（层级适配），encoder 采样子分辨率，FLOPs 与 RTF 的性能评估。

**📊 数据集**

实验数据集为 LibriSpeech（ASR）和 CREMA‑D（SER）。

**📈 对比分析**

通过在固定 FLOPs 预算下遍历不同配置，绘制 Pareto 前沿进行对比；ASR 词错误率从 19.01% 降至 4.85%，SER 统一准确率从 43.82% 提升至 80.46%，同时显著降低计算量。

**⚠️ 局限性**

局限性包括仅采用星形遍历未完成全局 iso‑FLOP 交叉搜索，对 SER 仍未实现分辨率轴的采样；后续可结合量化与连续子采样进一步提升效率。

---

## 402. Visual Geometry Transformer in the Wild: Distractor-Free 3D Reconstruction

**arXiv ID:** 2606.22787 | [PDF](https://arxiv.org/pdf/2606.22787v1)

**作者:** Tianbo Pan `[一作]` (National University of Singapore), Xinchao Wang `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为 Visual Geometry Transformer in the Wild (VGTW) 的端到端前向网络，用于在包含临时干扰物的真实世界多视图图像中实现无干扰的 3D 重建。

**💡 创新点**

创新点包括：① Distractor‑aware Training (DAT) 通过 LoRA 微调注意力机制并引入 Distractor Suppression Loss 与 Cross‑View Consistency Loss，显式抑制干扰特征；② 辅助 Mask Head 用于预测像素级干扰掩码，从而在后处理阶段更精确地剔除干扰；③ 无需 3D 标注即可训练，利用 2D 互补信息实现高效学习。

**🔧 技术方法**

使用的技术主要包括：基于 Transformer 的多视图编码器（继承自 VGGT/π³）、DINO 预训练特征、LoRA 参数微调、注意力掩码、损失函数（Distractor Suppression、Cross‑View Consistency、Binary Cross‑Entropy）以及 mask head 的卷积解码。

**📊 数据集**

主要使用的公开数据集：RobustNeRF-Mask（新增像素级干扰掩码）、NeRFOSR、NeRF on‑the‑go（未见过的动态场景）以及 RobustNeRF；这些数据集提供了多视角图像、摄像机位姿和干扰标注。

**📈 对比分析**

与 DUSt3R、MASt3R、Fast3R、VGGT、π³ 等基线方法比较，VGTW 在低/中/高遮挡场景下的 Accuracy、Completeness、Normal Consistency 均有显著提升，尤其在高遮挡下 Acc 与 Comp 的提升最为明显；在深度估计任务中，VGTW 的 Abs Rel 下降、δ<1.25 上升，整体性能优于原始模型和现有前向方法。

**⚠️ 局限性**

局限性：目前仅在 3D 重建任务上评估；在其他 3D 相关任务（如跟踪、语义分割）尚未验证；受限于训练数据规模与多样性，进一步扩展数据集和 3D 监督可能提升精度与泛化；在极端遮挡或快速运动场景下仍可能出现轻微的干扰残留。

---

## 403. Breaking the Evaluation Paradox: Evaluating High-Entropy Search with Computationally Irreducible Constraints

**arXiv ID:** 2606.22783 | [PDF](https://arxiv.org/pdf/2606.22783v1)

**作者:** Juntao Wu `[一作]` (Jinan University), Ke Wang `[通讯]` (Jinan University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 VERITAS benchmark，利用不可优化的加密哈希约束将“列举所有”高熵搜索任务转化为可验证、可自动生成的稀疏搜索任务，解决 LLM exhaustive search 的评估悖论。

**💡 创新点**

创新点在于：①将非可优化的哈希约束嵌入任务目标，既保持了 O(N) 的枚举难度，又可实现完全自动化、无人工标签的评估；②提供可调难度的多层级任务，支持无限规模生成和精确难度控制。

**🔧 技术方法**

使用技术包括：加密哈希函数（MD5/SHA）构造约束；Web 代理工具集（Search、Visit、Exec Python、Answer）模拟真实检索；理论分析（(T/N)^k 成功概率）与实验评估（Pass@4、平均工具调用）。

**📊 数据集**

数据集基于真实 Web 内容，自动生成多语言（英文、中文、日文）任务，划分为 Easy、Medium、Hard、Extra Hard 四个难度层级，且每个任务包含预设的哈希目标与完备 ground truth。

**📈 对比分析**

通过在多模型（GLM‑4、MiniMax‑M2、DeepSeek‑v3.1、Qwen3‑235B‑A22、Kimi‑K2、GPT‑4o、Gemini‑2.5‑Pro、Gemini‑3‑Pro、GPT‑5）上跑 Pass@4、平均工具调用比较，发现前沿模型 GPT‑5 与 Gemini‑3‑Pro 在 Medium 与 Hard 阶段表现明显优于其他模型，且 Extra Hard 仍对所有模型构成挑战。

**⚠️ 局限性**

限制：仅作为评估基准，未将 VERITAS 用于模型训练；缺乏针对哈希约束的主动搜索策略；评估侧重于枚举覆盖，未涵盖更复杂的高熵任务中可能出现的语义优化手段。

---

## 404. READ More than What You See: Reinforcement Learning for Accurate and Coherent Audio Description Generations

**arXiv ID:** 2606.22766 | [PDF](https://arxiv.org/pdf/2606.22766v1)

**作者:** Bo Fang `[一作]` (City University of Hong Kong), Antoni B. Chan `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于强化学习的音频描述生成框架READ，专门针对视听媒体中的盲/弱视用户生成简洁、准确且连贯的叙述。

**💡 创新点**

创新点在于：①首次将RL（GRPO）用于训练型AD生成；②设计四种奖励（内容准确、格式规范、长度控制、上下文连贯）并加入准确性门控与反抄袭掩码；③通过上下文感知训练提升跨片段连贯性。

**🔧 技术方法**

技术手段包括：基于Qwen3‑VL‑8B视觉语言模型的全参数RL微调；GRPO算法进行分组相对优势学习；ROUGE与自定义格式/长度奖励；使用轻量化LLM‑AD*对上下文连贯性进行评分。

**📊 数据集**

使用的公开AD数据集有：MAD‑Eval、CMD‑AD、TV‑AD，分别覆盖电影和电视剧集的数十万条音频描述。

**📈 对比分析**

与现有训练自由与训练基方法对比，READ在传统caption指标（BLEU‑1、CIDEr、ROUGE‑L）和AD专属指标（R@k、Action、LLM‑AD‑Eval）上均实现显著提升，例如在MAD‑Eval上CIDEr提升至40.0（高于33.5），R@5/16提升至61.7（高于56.4），Action得分提升至36.1。

**⚠️ 局限性**

局限性包括：①连贯奖励依赖外部LLM评分与启发式门控，可能无法充分捕捉更高层次叙事一致性；②目前仅利用邻近片段文本上下文，未显式建模更长范围的情节结构、对话语义或音频线索；③实验仅覆盖英文数据，跨语言、跨文化或更长文本的推广尚待验证。

---

## 405. Evolutionary Optimization Reveals Structural Constraints on Reservoir Architecture for Spatiotemporal Chaos

**arXiv ID:** 2606.22765 | [PDF](https://arxiv.org/pdf/2606.22765v1)

**作者:** Nima Dehghani `[一作]` `[通讯]` (MIT), Nima Dehghani (MIT)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究通过进化优化方法，探讨了在预测Kuramoto–Sivashinsky方程的时空混沌时，水库计算的结构约束如何影响其性能。

**💡 创新点**

创新点在于通过进化优化揭示了水库架构的结构约束，表明预测性能不仅依赖于超参数的调整，还与水库的内在动态特性密切相关。

**🔧 技术方法**

使用了遗传算法对水库的五个构建超参数（大小、连接度、谱半径、输入缩放和读出正则化）进行优化。

**📊 数据集**

使用Kuramoto–Sivashinsky方程生成的时空混沌数据集进行实验。

**📈 对比分析**

与传统的固定水库计算方法相比，进化优化显著降低了预测误差，延长了低误差预测的时间范围，并在大小-效率平面上形成了一个边界，显示出大小与预测性能之间的复杂关系。

**⚠️ 局限性**

限制在于该研究主要集中于Kuramoto–Sivashinsky方程，可能无法直接推广到其他类型的动态系统，且进化过程的计算成本较高。

---

## 406. Cooperative-ORCA*: Real-Time Proactive Deadlock Avoidance for Continuous-Space Multi-Agent Navigation

**arXiv ID:** 2606.22757 | [PDF](https://arxiv.org/pdf/2606.22757v1)

**作者:** Junfeng Wu `[一作]` (Monash University), Andy Li `[通讯]` (Monash University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在连续空间多智能体路径规划（CS-MAPF）中，提出并实现了C-ORCA*及其变体C-ORCA*-MAPF，通过先行使用离散MAPF求解器生成的全局路径作为指导，并在执行时检测动态通道依赖，主动预防死锁，结合ORCA*实现实时规划。

**💡 创新点**

创新点主要有：① 将离散MAPF路径转换为连续轨迹（string‑pulling）作为全局指导；② 预处理阶段检测通道与交叉通道依赖，实现主动等待；③ 引入漂移模式，减少通道拥堵；④ 在保留ORCA*-MAPF后备机制的同时显著降低其调用频率。

**🔧 技术方法**

采用的技术包括：ORCA*算法、离散MAPF求解器（如MAPF-LNS、ECBS）、字符串拉伸转换、通道检测与依赖组构建、漂移模式与优先速度计算、时间窗阈值检测、离散-连续映射与动态观察。

**📊 数据集**

实验使用四种网格地图：Gap（64×64）、Random（64×64）、Room（32×32）和Warehouse（321×123），并在10到200名代理（增量5）进行多次随机实例测试。

**📈 对比分析**

与ORCA*、ORCA*-MAPF对比，采用成功率、流量（flowtime）、运行时间等指标。C-ORCA*-MAPF在所有地图上均实现最高成功率（200名代理时>80%）、最低流量，并且运行时间仅略高于ORCA*，MAPF后备调用次数减少3–4个数量级。

**⚠️ 局限性**

局限性包括：在极端拥堵场景仍可能陷入局部最优；依赖检测不考虑时间维度，导致约束不完整；对动态或非预先离散化的环境适应性有限；漂移模式对不对称通道或非对称障碍效果不如预期。

---

## 407. Policy-as-Data: Learning Generalizable HOI Diffusion Models from Simulated Physics

**arXiv ID:** 2606.22806 | [PDF](https://arxiv.org/pdf/2606.22806v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 408. HERCULES: An Open-Source Simulation Framework for Heterogeneous Multi-Robot SLAM, Collaborative Perception, and Exploration

**arXiv ID:** 2606.22756 | [PDF](https://arxiv.org/pdf/2606.22756v1)

**作者:** Sandilya Sai Garimella `[一作]` (Georgia Institute of Technology), Lu Gan `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 HERCULES——一个基于 UE5 的开源模拟框架，支持 UAV-UGV 协同操作、同步数据采集、热/夜视等多模态传感，并提供共享导航堆栈与实验工具。

**💡 创新点**

① 在 AirSim/Cosys-AirSim 上重构 SimMode 以实现多种平台并发；② 引入统一航路点控制、UGV 纯追踪控制；③ 提供基于 OctoMap 与高程图的地形感知与规划；④ 通过 LWIR 与 NVG 渲染实现恶劣环境感知；⑤ 搭建多模态动态代理与环境现象模块；⑥ 开源数据集与实验脚本。

**🔧 技术方法**

Unreal Engine 5、AirSim/Cosys-AirSim、ROS 2、Python/C++ API、OctoMap、Lumen、Nanite、PhysX/Ode、深度学习框架（OpenVINS、ORB‑SLAM3、LIO‑SAM、ROMAN、Kimera‑RPGO、PointPillars）、Sim‑to‑Real 迁移等。

**📊 数据集**

HERCULES 自己生成的异构 SLAM 基准（城市、沙漠、森林四序列，四台 UAV/UGV）以及 6000 对 UGV‑UAV 视图的合作检测数据，并利用 DAIR‑V2X 实际数据进行 sim‑to‑real 评估。

**📈 对比分析**

在 SLAM 基准上对单机方法（OpenVINS、ORB‑SLAM3、LIO‑SAM）与多机 ROMAN 进行 RMS‑ATE 对比；在检测任务中用单机 UAV/UGV、Late‑Fusion 三种配置评估 KITTI‑style AP，发现 Late‑Fusion 在不同距离下互补，预训练可提升 4.1 % 的 3D AP；在闭环探测实验中比较 Complementary Coverage 与 Leader‑Follower，前者在相同时间覆盖率高 42 %，实时因子≥0.3。

**⚠️ 局限性**

对动态物体与语义分割的鲁棒性仍有限，ROMAN 在城市/森林等重复语义环境及异构机器人对中易失效；仿真中动态代理与环境现象模块虽已实现但仍需手动配置；热/夜视模型依赖经验函数，可能在极端光照下不够精确；实时性能受限于 GPU 与物理引擎，未实现完全可扩展到更大规模多机器人。

---

## 409. AI Fiction in the Wild

**arXiv ID:** 2606.22748 | [PDF](https://arxiv.org/pdf/2606.22748v1)

**作者:** Neel Gupta `[一作]` (University of Washington), Melanie Walsh `[通讯]` (University of Washington)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用公开的WildChat对话数据，对超过5.7万条英文聊天记录进行文本分类，识别出约三分之一的会话涉及小说、角色扮演、同人及色情内容，并进一步对这些创作行为进行用户画像和模式分析。

**💡 创新点**

创新点在于首次以大规模真实对话数据量化AI生成文学的普及程度，提出“无限故事需求者”和“故事循环者”等用户类型，并引入“自我读写者”的概念，探讨AI文学对作者-读者关系的潜在重塑。

**🔧 技术方法**

主要技术手段包括：利用GPT‑4‑Mini对话内容进行语义分类；使用MiniLM‑L6向量嵌入与DBSCAN聚类识别近似重复提示；统计分析用户的重复率、发起频次等行为指标。

**📊 数据集**

数据集为WildChat 1M公开版本中的573,453条英文对话，其中195,271条被归类为含小说生成（含粉丝、色情等子类），是迄今为止最大的公开LLM用户对话语料库之一。

**📈 对比分析**

通过人工标注300条样本与GPT‑4‑Mini自动标注对比，得到小说类的精度0.97、召回0.94；粉丝类精度0.94、召回0.85；色情类精度0.84、召回0.69，说明LLM分类在本任务中表现优秀。

**⚠️ 局限性**

主要局限包括：WildChat用户群体可能偏向技术爱好者，难以代表普遍人群；对话匿名化难以确认真实身份与动机；模型生成长度受限，难以产生完整长篇小说；以及对伦理与隐私的持续关注。

---

## 410. Closed-loop Auto Research for Molecular Property Prediction: Discovering and Certifying Generalizable Improvements

**arXiv ID:** 2606.22731 | [PDF](https://arxiv.org/pdf/2606.22731v1)

**作者:** Jingjie Ning `[一作]` (Carnegie Mellon University), Guolin Ke `[通讯]` (DP Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在本研究中，作者构建并评估了闭环自动研究（Auto Research）框架，探讨其在分子属性预测任务中的泛化能力。通过轴隔离（仅允许单一干预方向——特征、模型或外部数据）以及保留测试（对验证选出的最佳配置仅在未见测试集上重新训练并评估），实现了对发现与验证结果的区分。

**💡 创新点**

创新点主要体现在：①首次将闭环自动研究与轴隔离结合，明确每个干预方向对模型提升的贡献；②提出保留测试认证流程，揭示两类非转移签名（选择方差与分布偏移）；③构建泄漏安全的外部数据获取管道，保证外部数据不会导致测试集泄漏。

**🔧 技术方法**

技术手段包括：语言模型代理驱动的自动化实验；文件级消融锁保证每次实验仅改变单一干预；Leakage‑safe过滤器（身份去重、同源拒绝、近似化简）用于外部数据审核；MapLight式基准作为强基线；对照实验使用FLAML进行同等规模的AutoML搜索；与预训练3D模型Uni‑Mol进行性能对比。

**📊 数据集**

所用数据集为：TDC ADMET（22个回归/二分类端点）、MoleculeNet（10个端点，含FreeSolv、ESOL等）和Polaris（Biogen adme‑fang，共4个回归端点），共36个端点。

**📈 对比分析**

评价方法：在内部验证集上选取每个轴的最佳配置，随后在未见测试集上冻结并重新训练一次，得到保留测试增益。实验结果显示：TDC、MoleculeNet、Polaris的路由保留测试增益分别为+0.013、+0.011、+0.042；在同等试验规模下，Auto Research在模型轴上的增益远超FLAML（+0.006）且与84M参数的Uni‑Mol相比保持竞争力。

**⚠️ 局限性**

局限性：①结果受限于所选基准分割和数据来源，无法完全排除分布偏移；②仅评估单轴干预，未考察多轴联合效果；③保留测试仅进行一次评估，可能无法覆盖所有泛化场景；④外部数据虽通过严格过滤，但仍可能存在不匹配的实验条件导致非转移。

---

## 411. Measuring Behavior Portability in Large Language Models

**arXiv ID:** 2606.22797 | [PDF](https://arxiv.org/pdf/2606.22797v1)

**作者:** Tianjia Dong `[一作]` (University of Chicago), James A. Evans `[通讯]` (University of Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了大型语言模型（LLM）在保持相同收益结构的不同文字描述（payoff‑equivalent prompts）下的行为可迁移性（behavioral portability）。

**💡 创新点**

提出了把对齐（alignment）与可迁移性联系起来的理论框架，并使用总变差距离（Total Variation）作为无损失（loss‑agnostic）的可迁移性评估指标；同时定义了“训练在源环境，测试在目标环境”的评估协议。

**🔧 技术方法**

使用总变差距离、OLS 线性行为模型、链式推理（CoT）与非推理提示、以及对不同 LLM（GPT‑4.1‑nano、Gemma‑3‑12B、Llama‑3.1‑8B/70B、DeepSeek‑R1）进行实验。

**📊 数据集**

构造了 7 个经典的一拍经济决策任务（Dictator、Ultimatum、Trust、Public Goods、Beauty Contest、Lottery Choice、Normal‑Form），每个任务生成 40 种文字环境，产生多种 payoff‑equivalent实验数据集。

**📈 对比分析**

比较方法：在源环境上训练行为模型，再在未见目标环境上测试；将其与仅在目标环境训练的基准模型对比；通过总变差距离衡量两者的预测‑行动分布差异。结果显示平均 TV ≈0.35，表明大部分模型在不同文字表述下存在显著可迁移性损失；CoT 在大多数模型和任务中略有改善，但表现不稳定；深度推理模型 DeepSeek‑R1 在几项任务中取得最优或接近最优。

**⚠️ 局限性**

局限性：仅评估了七个有限任务，行为模型采用线性 OLS，未考虑更复杂的非线性映射；实验只覆盖有限数量的 LLM，未覆盖更大规模或不同架构的模型；总变差距离是最坏情况指标，未说明在具体应用中的实际影响。

---

## 412. Pricing the Unpriced Asset: A Standards-Based Method for Valuing Enterprise Data under IAS 38 and IAS 2

**arXiv ID:** 2606.22760 | [PDF](https://arxiv.org/pdf/2606.22760v1)

**作者:** Natasha E. Blycha `[一作]`, Adam Myer `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并验证了一个两层认证数据资产估值进程——可审计成本基准（D‑Val）与基于质量、稀缺、竞争、认证与审计溢价的商业估值（A‑Val）。

**💡 创新点**

创新点在于将IAS 38的成本计价与理论驱动的商业估值相结合，形成可落地的成本底线与未来可观测市场价之间的桥梁，并给出可操作的参数设定与示例。

**🔧 技术方法**

使用经济学理论（产权理论、信号理论、代理成本框架）构建参数基础，采用数理公式（对数规模、指数稀缺、乘法溢价）以及示例中的K‑Means+非线性最小二乘学习方法来校准未观测参数。

**📊 数据集**

案例数据包括零售客户交易数据库、矿业地质调查数据以及脱识别临床结果数据库，分别演示了D‑Val和A‑Val的计算与市场对比。

**📈 对比分析**

通过三例示范比较：计算D‑Val与A‑Val，并将A‑Val与公开可比交易价格（如零售数据、地质调查、医疗数据库）进行对照，A‑Val结果在1.00–2.97倍D‑Val范围内，与行业经验价相符，显示模型能够捕捉资产的商业价值。

**⚠️ 局限性**

局限性包括参数未经验验证、稀缺与竞争评估主观、缺乏成熟活跃市场、模型形式可多选、受IAS 38现行规则限制、未涉及税务、转移定价等方面。

---

## 413. GroundEval: A Deterministic Replacement for LLM-as-Judge in Stateful Agent Evaluation

**arXiv ID:** 2606.22737 | [PDF](https://arxiv.org/pdf/2606.22737v1)

**作者:** Jeffrey Flynt `[一作]` `[通讯]`, Jeffrey Flynt

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个无评判模型的评估框架GroundEval，用于检测LLM代理在回答时是否使用了符合访问、时间、因果等约束的证据路径。

**💡 创新点**

提出了通过可机读的状态合同与三条评估轨道（Perspective、Counterfactual、Silence）实现的确定性、可复审的证据路径评分方法，解决传统LLM-as-judge无法识别状态失效答案的问题。

**🔧 技术方法**

利用事件日志、文档语料、访问策略以及自定义配置构建可机读合同；采用结构化问答、工具调用跟踪和规则化得分器；实现两种模式：上下文模式和工具模式。

**📊 数据集**

在基于OrgForge模拟器生成的合成企业数据集上评估，覆盖九个子系统、八个角色以及数百个事件、因果链和缺失对，随后通过合成问答覆盖三条轨道。

**📈 对比分析**

与零射（无文档）对比，Gated工具模式在答题正确率、轨迹得分与合规调整得分上分别提升了约40%至60%；相比传统LLM-as-judge，GroundEval能捕捉到多种未被检测的错误，展示了更高的诊断透明度。

**⚠️ 局限性**

限制在于上下文模式对证据路径的可观测性有限，需手动编写和验证合同配置，且在多代理或更长时间范围的任务中对状态边界的定义仍较为复杂。

---

## 414. Lifting E-Graphs: A Function Isn't a Constant

**arXiv ID:** 2606.22734 | [PDF](https://arxiv.org/pdf/2606.22734v1)

**作者:** Philip Zucker `[一作]` `[通讯]`, Philip Zucker

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出一种在e-graph中内置lifting combinator、fat identifiers和thinnings的实现，解决变量命名与scope污染问题，提升共享与等式压缩效率。

**💡 创新点**

创新点在于将lifting与thinning概念直接嵌入e-graph结构，并提供lift-pulling智能构造器与thinning-aware union find，首次实现函数提升与α等价的统一处理。

**🔧 技术方法**

采用fat identifiers、位向量thinnings、lift-pulling智能构造器、改进的union find以及点对点的lifting/推送规则。

**📊 数据集**

本文未使用具体数据集，主要以理论与实现设计为主，未给出实验数据。

**📈 对比分析**

未给出实验对比，作者仅说明理论上可提升内存共享与等式压缩，实际性能待进一步验证。

**⚠️ 局限性**

局限在于仅处理rigid变量，未完整涵盖绑定、类型推断与多语义函数，且实现复杂度较高。

---

## 415. When Confidence Takes the Wrong Path: Diagnosing Retrieval-State Lock-In in RAG

**arXiv ID:** 2606.22728 | [PDF](https://arxiv.org/pdf/2606.22728v1)

**作者:** Sahib Julka `[一作]` `[通讯]` (LMU Munich), Sahib Julka (LMU Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并系统化了检索状态锁定（retrieval‑state lock‑in）问题，定义了其可测量特征（silent‑error 率）并在检索增强生成（RAG）系统中进行诊断与评估。

**💡 创新点**

首次将锁定现象命名、量化并给出普遍性边界，同时提出答案‑证据‑检索三方分解的诊断框架，并基于此设计了高精度审计规则。

**🔧 技术方法**

使用 KG‑RAG（知识图谱检索+检索后缀）与密集检索两种方案；设计并比较了 SD‑UQ（答案分散）、SEU（证据矛盾）和 GPS（检索支持）等不确定性度量；在每题采样 5 个答案进行评测。

**📊 数据集**

六个问答数据集：PubMedQA、RealMedQA、HotpotQA、HotpotQA‑FullWiki、2WikiMultiHopQA、MuSiQue，覆盖临床、医学和开放域多跳问答。

**📈 对比分析**

对 KG‑RAG 与密集检索在准确率、AUROC 等指标进行对比，发现两者准确率无显著差异，但 KG‑RAG 在答案、证据与检索状态误差检测上更优；合成审计规则在 7.7% 覆盖率下实现 91.9% 精度，单一答案‑一致性方法约 69.7%。

**⚠️ 局限性**

实验受限于固定子集、单一模型（GPT‑4o‑mini）和单一 KG‑RAG 框架；GPS 在开放域数据表现不佳；SEU 的中性阈值受 NLI 领域限制；覆盖率低；未验证对抗鲁棒性或跨模型通用性。

---

## 416. Private Information Retrieval from Joint Systematic MDS-Coded with Non-Colluding Servers: Bounds and Constructions

**arXiv ID:** 2606.22802 | [PDF](https://arxiv.org/pdf/2606.22802v1)

**作者:** Jingke Xu `[一作]` (Shandong Agricultural University), Weijun Fang `[通讯]` (Shandong University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文研究了在系统化MDS数组存储码下的联合MDS编码私有信息检索（PIR）问题，并给出了容量上界与构造方案

**💡 创新点**

在预设的存储模式下证明了联合MDS‑PIR容量上界，并首次证明了Sun‑Tian 2019方案在K=Mt、N≤K+t情况下的最优性；进一步构造了三类可实现更高检索率的方案

**🔧 技术方法**

采用信息论容量分析、条件独立性与Han不等式、MDS码的可恢复性以及随机置换/矩阵构造等技术

**📊 数据集**

本文主要为理论分析，未使用具体数据集；所有实验均为符号长度与字段大小的符号计算

**📈 对比分析**

通过与已知的分离MDS‑PIR容量（Banawan‑Ulukus）以及已发布的联合方案进行比较，证明所构造方案在多参数范围内可提高15%–26.42%的检索率

**⚠️ 局限性**

仅考虑了K=Mt与K=Mt+1、且存储模式为ℓI_M⊗1_t或ℓ/K(MI_M⊗1_t|1_M^⊺)的情况，未覆盖更一般的联合编码与存储模式

---

## 417. UniFS: Unified Fast-to-Slow Hierarchical Architecture for Vision-Language-Action Models

**arXiv ID:** 2606.22794 | [PDF](https://arxiv.org/pdf/2606.22794v1)

**作者:** Lin Sun `[一作]` (JD.com), Lige Liu `[通讯]` (JD.com)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种统一的快速到慢速层次架构（UniFS），在单一VLM中生成多时尺度潜在向量以驱动动作专家；

**💡 创新点**

创新点包括：1）将VLM层按频率分组实现异步更新；2）潜在向量反转机制匹配不同频率与动作专家；3）多级监督促进粗细层次学习；4）频率特征替换实现训练并行；

**🔧 技术方法**

采用多模态Transformer（DINOv2、SigLIP+LLaMA2/Qwen）、异步层级执行、频率特征替换、潜在向量反转与多级监督等技术；

**📊 数据集**

在LIBERO基准（130个任务）以及Franka机器人真实实验中进行验证；

**📈 对比分析**

与VLA-Adapter、π_0等基线对比，UniFS平均成功率98.3%，比VLA-Adapter高2.5%；推理延迟从36.5ms降至17.8ms，提升约2.1×；

**⚠️ 局限性**

训练时仍需完整前向推理，难以获得加速；需要大批量和时间多样性；潜在向量反转可能导致分布漂移；对预训练模型微调要求较高。

---

## 418. DE-FIVE: Detecting Malicious Image Prompts via Fourier Features and Image Vector Embeddings

**arXiv ID:** 2606.22779 | [PDF](https://arxiv.org/pdf/2606.22779v1)

**作者:** Xingwei Zhong `[一作]`, Vrizlynn L. L. Thing `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了一个无训练成本的混合检测框架DE-FIVE，用于识别视觉语言模型中的恶意图像提示，兼顾jailbreak和间接注入攻击。

**💡 创新点**

创新点包括：①黑盒检测利用频域高低频能量比和频谱熵两大特征；②白盒检测通过对视觉编码器的随机掩码平均嵌入进行对比，仅需极少恶意样本；③将两种检测融合成加权混合策略，整体实现无训练、可迁移。

**🔧 技术方法**

使用2D离散傅里叶变换提取频域特征、频谱能量比与熵计算、图像掩码平均得到鲁棒嵌入、余弦相似度计算以及加权融合的检测分数。

**📊 数据集**

使用Meta‑instruction数据集（25 benign + 300恶意图片）和XSTest（250 safe+200 unsafe prompts + 200 benign图片），并在LLaVA‑1.6和Phi‑3两种VLM模型上进行评估。

**📈 对比分析**

与GradSafe、MirrorCheck、JailGuard、Perplexity、VLMGuard等基线在AUROC上进行对比，DE‑FIVE在meta‑instruction和jailbreak两类攻击均取得最高或接近最高AUROC，平均提升约2–3%。

**⚠️ 局限性**

局限性包括：需要少量恶意样本（k=32）来构造白盒参考向量，对极端或未知攻击方式的鲁棒性仍待进一步验证；频域特征在高频噪声区分上可能受限。

---

## 419. GeoRouteNet: Geometry-Enhanced Non-Autoregressive Neural Solver for the Traveling Salesman Problem

**arXiv ID:** 2606.22776 | [PDF](https://arxiv.org/pdf/2606.22776v1)

**作者:** Xiang Li `[一作]` `[通讯]` (Yangtze University), Xiang Li (Yangtze University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种几何增强的非自回归神经求解器（GeoRouteNet）以及多候选自比较强化学习训练方法（MCSRL），用于欧氏旅行商问题。

**💡 创新点**

创新点在于：1）在编码器中加入中心化坐标、可学习径向距离基函数、距离感知图注意力、显式边信息交流与跨层残差混合；2）训练时采用多候选自比较、适应性基线与胜者引导的策略梯度，提升跨尺度、跨分布泛化；3）通过精细的熵正则化实现探索-利用平衡。

**🔧 技术方法**

使用了图神经网络、Swiglu+LayerNorm、距离感知多头注意力、可学习RBF嵌入、K‑NN掩码、Beam搜索解码以及强化学习（策略梯度+熵正则）。

**📊 数据集**

在三类数据集上评估：1）随机欧氏TSP（10,000个50/100/200节点实例）；2）不同规模随机实例（50、100、150、300节点）；3）27个标准EUC_2D实例（包括近似、100节点、中尺度和外域）。

**📈 对比分析**

与原始NAR模型、LKH3、Concorde比较；在Beam‑1000解码下，GeoRouteNet+MCSRL在50节点下0.32% GAP，100节点下1.26% GAP，27个EUC_2D整体3.60% GAP；吞吐量比Concorde/ LKH3高数百倍，且在外域场景中仍保持低延迟。

**⚠️ 局限性**

局限性：1）对极端外域实例仍有≈8% GAP；2）未使用局部搜索后处理；3）k‑NN 10 限制长程信息传播；4）K候选数固定，可能未最大化训练信号；5）仅针对欧氏TSP，未覆盖非欧氏/非对称或约束路由。

---

## 420. LoCC: Detection and Localization of Lip-Syncing Deepfakes via Counterfactual Frame Consistency

**arXiv ID:** 2606.22772 | [PDF](https://arxiv.org/pdf/2606.22772v1)

**作者:** Soumyya Kanti Datta `[一作]` (University at Buffalo, State University of New York), Siwei Lyu `[通讯]` (University at Buffalo, State University of New York)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出LoCC框架，利用对抗式扩散重建检测口型帧的时间一致性来定位和检测lip‑syncing deepfakes。

**💡 创新点**

创新点包括：①仅用相邻两帧生成中间帧的对抗式扩散重建；②引入Diffusion Inconsistency Loss（DIL）在教师网络中量化时间一致性；③通过教师-学生蒸馏实现高效帧级检测，兼顾细粒度定位。

**🔧 技术方法**

使用扩散模型进行口型帧重建，3D/2D卷积网络提取特征，Transformer编码时序，知识蒸馏与KL散度损失。

**📊 数据集**

在FakeAVCeleb、LavDF、AVDF1M、KODF等多种lip‑syncing deepfake数据集上训练与评估。

**📈 对比分析**

与多种基线（Xception、FTCN、AVFF、ICS‑AVDF‑Frozen等）对比，LoCC在AP、AUC、IoU等指标上均达到或超过0.98的高性能，尤其在跨域KODF和AVDF1M上表现突出。

**⚠️ 局限性**

局限性包括：在极高IoU阈值下性能下降（因模型规模较小）；仅关注口型区域，未结合音频线索；对完整视频的整体一致性检测仍有改进空间。

---

## 421. Noise is Signal: Density-Based Outliers as Leading Indicators of Occupational Emergence in Labor Market Text

**arXiv ID:** 2606.22769 | [PDF](https://arxiv.org/pdf/2606.22769v1)

**作者:** Shreyash Rawat `[一作]` `[通讯]` (Independent Researcher), Shreyash Rawat (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用HDBSCAN噪声类的工作岗位帖子，提出Emergence‑Density Inversion假设并构建Extended EOS指标，对新兴职业进行早期预测和验证；

**💡 创新点**

创新点在于把噪声类视为有价值信号，证明语义一致性比密度极端性更能预示职业出现，且通过Temporal Velocity与Cross‑Platform Convergence两项特征提升EOS的预测力；

**🔧 技术方法**

技术方法包括使用INSTRUCTOR‑XL嵌入、UMAP降维、HDBSCAN聚类、EOS多维评分、逻辑回归预测，以及与Isolation Forest、LOF、GLOSH和BERTrend等基线的对比；

**📊 数据集**

实验数据基于84,988条英文招聘帖子，覆盖2022年Q4至2024年Q3的八个季度，并结合O*NET与ESCO技能词典进行特征计算；

**📈 对比分析**

在1/2季度预测任务中，LR‑EOS达到F1=0.74，优于所有基线（最强者BERTrend仅F1=0.58），EOS阈值0.75时人工标注的精确度为77%；

**⚠️ 局限性**

局限性包括噪声组样本量有限、仅覆盖英语公开招聘平台、O*NET参考可能导致TaxGap过高、未实现因果分析，以及未对非英语或非正规劳动力市场进行验证。

---

## 422. Temporal Logic Guidance for Action-Only Diffusion Policies with World Models

**arXiv ID:** 2606.22729 | [PDF](https://arxiv.org/pdf/2606.22729v1)

**作者:** Moritz Zoellner `[一作]` (Purdue University), Rohan Paleja `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种在推理时使用可微世界模型计算 STL 约束鲁棒性梯度，并将其注入扩散政策去噪过程，从而在仅生成动作的扩散策略上实现对行为约束的实时引导。

**💡 创新点**

创新点在于将可微世界模型与 STL 鲁棒性梯度结合，突破了传统 STL 指导需联合动作-状态生成的限制，实现了不重新训练即可按约束定制行为的推理时引导。

**🔧 技术方法**

使用扩散政策、可微世界模型、信号时序逻辑鲁棒性评估、梯度注入去噪、以及后期梯度上升优化。

**📊 数据集**

在 Robomimic 的 Can Transport 任务上进行实验，使用人类演示混合数据训练扩散政策和世界模型。

**📈 对比分析**

与基线扩散政策和采样-排名方法比较，约束满足率从 84% 降到 4%，任务成功率保持 100%，平均倾斜角度从 8.5° 降到 1.9°，显示显著性能提升。

**⚠️ 局限性**

受限于世界模型的预测误差和扩散步骤的时间步长，难以处理长时间或极其复杂的 STL 约束，且对世界模型训练质量高度敏感。

---

## 423. KaLM-Reranker-V1: Fast but Not Late Interaction for Compressed Document Reranking

**arXiv ID:** 2606.22807 | [PDF](https://arxiv.org/pdf/2606.22807v1)

**作者:** Xinping Zhao `[一作]` (Harbin Institute of Technology), Min Zhang `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的高效重排序框架——FBNL（Fast but not Late-Interaction）重排序器，能够在离线预编码文档后，利用解码器进行细粒度查询-文档相关性建模。

**💡 创新点**

创新点在于：①分离查询和文档编码，预编码文档以提升在线推理效率；②通过解码器的自注意力+交叉注意力实现强表达式相关性建模；③引入Matryoshka嵌入池化(MEP)压缩文档表示，进一步降低存储和推理成本。

**🔧 技术方法**

使用T5Gemma2基础模型的encoder-decoder结构、Matryoshka Embedding Pooling、交叉注意力机制以及多阶段训练（一般化→任务化→细粒度蒸馏）。

**📊 数据集**

在BEIR（13个英文检索任务）、MIRACL（18语种多语言检索任务）和LMEB（六个长记忆检索任务）上进行评估，并使用KaLM-Embedding-V2.5作为检索器。

**📈 对比分析**

与从0.3B到8B的开源重排序器（bge-reranker、gte-reranker、Qwen3-Reranker、jina-reranker等）比较，FBNL模型在保持相似甚至更好nDCG@10性能的同时，在线计算成本下降10-200倍；在BEIR和MIRACL上与工业级重排序器接近。

**⚠️ 局限性**

局限性包括：在中文和多语种任务上性能相对较弱；高压缩比（如32×）会显著损失区分能力；依赖于T5Gemma2中文能力的瓶颈需要进一步提升。

---

## 424. CoVStream: Edge-Cloud Collaboration for Understanding of Long Video Streams

**arXiv ID:** 2606.22804 | [PDF](https://arxiv.org/pdf/2606.22804v1)

**作者:** Xu Liu `[一作]` (Zhejiang University), Wenguan Wang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种边缘-云协同框架，边缘节点对连续视频进行视觉特征与语义标题双重压缩后上传，云端维护视觉上下文与实体图，按需推理。

**💡 创新点**

创新点在于将视频压缩为可直接推理的语义表示，采用双流感知管线与异步内存/推理架构，显著降低带宽且保持高准确率。

**🔧 技术方法**

技术包括视觉特征提取、时间特征压缩、关键帧自适应选择、轻量化语义标题生成、实体图构建与增量更新、睡眠-唤醒推理以及图增强检索。

**📊 数据集**

使用 VideoMME-Long、LVBench 与 RTV-Bench 三大流媒体基准进行实验评估。

**📈 对比分析**

与本地单机、云端完整推理及边缘采样混合基线相比，带宽减少 87.6%，准确率保留 99.2%，端到端时延 2.99 秒，显著优于基线。

**⚠️ 局限性**

局限性包括对大型模型特征空间的依赖、极长视频中可能仍有信息丢失、仅在实验硬件上验证，且对多模态语义标注的准确性敏感。

---

## 425. Cyclic Graphs and Memoization in Pure $λ$-Calculus

**arXiv ID:** 2606.22908 | [PDF](https://arxiv.org/pdf/2606.22908v1)

**作者:** Bo Yang `[一作]` `[通讯]` (Figure AI Inc.), Bo Yang (Figure AI Inc.)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种纯 λ‑演算的表格化（tabling）弱头归约语义，该语义在不增加递归构造或不纯缓存的前提下，把任何可到达的 λ‑项的无穷展开压缩为有限（可环）图，从而实现动态规划、循环检测和记忆化共享。

**💡 创新点**

创新点包括：
- 将表格化与弱头归约结合，得到一个新的操作语义；
- 在不扩展 λ‑演算的情况下实现循环检测与共享；
- 证明该语义可靠、终止于“有理”片段、且求解唯一；
- 通过同一解释器实现自举编译器、动态规划、图可达性等多种 DSL 应用。

**🔧 技术方法**

技术手段：
- 采用弱头归约（只展开左侧最左红ex）作为一层映射；
- 通过结构同一性（哈希‑一致）对已出现的子项做表格化；
- 采用 Kleene 迭代实现最小不动点求解；
- 在实现中使用 de Bruijn 形式的 interned 结构，保证 O(1) 同一性测试；
- 通过图的折叠实现记忆化、循环检测与终止判定。

**📊 数据集**

数据集：
- 对编辑距离的实验用小型字符串例子（如 "ab" 与 "cd"）；
- 对循环流的实验用简单的无穷零流 r = Y (cons 0)；
- 对非生产性循环的实验用经典 Ω = (λx. x x)(λx. x x)。
（本文主要在这些案例上演示和验证，未使用大规模公开数据集。）

**📈 对比分析**

性能比较：
- 对编辑距离，纯递归实现指数级，而表格化实现 O(mn) 时间与空间；
- 对零流 r，普通归约无限展开，表格化即时构造有限环；
- 对 Ω，普通归约永不终止，表格化在有限步内判定为 ⊥；
- 所有实验均在实现语言 Claude Code 的单线程环境中完成，运行时间与空间与经典动态规划实现相当或更优，且无需显式缓存表。

**⚠️ 局限性**

局限性：
- 只能处理“有理”片段（可到达子项有限且无环的情况），无法覆盖所有递归 λ‑项；
- 依赖结构同一性，若使用更粗或更细的同一性（如行为等价）需重新设计；
- 对深层嵌套的递归可能导致内部结构巨大，进而影响 intern 的成本；
- 仍无法直接处理需要非纯副作用的计算（如并发、IO 等）。

---

## 426. From Fragments to Paths: Task-Level Context Recovery for Large Industrial Codebases

**arXiv ID:** 2606.22906 | [PDF](https://arxiv.org/pdf/2606.22906v1)

**作者:** Jiawei He `[一作]` (AMAP, Alibaba Group), Dong Sun `[通讯]` (AMAP, Alibaba Group)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大规模工业代码库中，提出一种任务级仓库理解方法DeepDiscovery，能够在任务描述下定位高置信任的锚点并扩展多关系图谱，恢复完整任务相关上下文；

**💡 创新点**

创新点在于将仓库理解视为任务级上下文恢复，采用两阶段Location–Inference框架，结合自适应压缩、隐式关系提取和metadata-first上下文构建，实现高召回且无离线预处理；

**🔧 技术方法**

技术包括大语言模型驱动的语义检索、结构化图谱扩展、规则引导的锚点定位、隐式关系库、基于预算的优先级评估与元数据优先化；

**📊 数据集**

使用内部生产级集成代码库（约2.67M行，25k文件）中的27个中型任务、40个大型子项目，以及公开的SWE-bench Verified基准；

**📈 对比分析**

与DeepWiki、CodeWiki、RAG、GraphRAG、AST+GraphRAG等基线对比，DeepDiscovery在无离线预处理的前提下实现92.6% FRR，且在SWE-bench Verified上Solve Rate提升至78.6%（+8.2个百分点），显示出显著性能提升；

**⚠️ 局限性**

局限性包括对任务描述的依赖、锚点定位不佳时扩展失效、对隐式关系的召回优先导致精度下降、仅在单一组织环境验证、对极端快速迭代仓库的适用性待进一步评估。

---

## 427. Priority-Aware Learning-Unlearning Correction for Dynamic Decentralized LoRA Fine-Tuning

**arXiv ID:** 2606.22878 | [PDF](https://arxiv.org/pdf/2606.22878v1)

**作者:** Nuocheng Yang `[一作]` (Beijing University of Posts and Telecommunications), Changchuan Yin `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在分布式边缘网络中设计一种能够在设备动态加入或离开时快速完成联邦 LoRA 微调的学习-遗忘校正框架。

**💡 创新点**

创新点主要有：①使用冻结的随机正交投影基使每个设备的贡献可无历史记录地被隔离或扩展；②对后事件初始化进行理论分析，得到可控的初始化误差；③提出基于 Fisher 信息与梯度能量的优先级分配策略，在有限通信预算下按层组动态调配本地更新、近似阻尼与同步密度。

**🔧 技术方法**

技术包括：LoRA 参数压缩、正交投影（Orthogonal LoRA）、分布式梯度下降 (DGD)、Metropolis 加权网络聚合、Fisher 信息近似、基于梯度能量的优先级评估、随机图网络分配。

**📊 数据集**

文中未明确给出具体数据集，实验使用标准的 LLM 预训练模型和边缘设备的模拟私有数据，主要关注通信预算与校正误差。

**📈 对比分析**

与传统的无正交基、无优先级分配的分布式 LoRA 微调以及现有的联邦/分布式未学习方法进行对比。实验结果显示，在相同的通信预算下，本文方法在设备加入/离开后能够显著降低事件校正误差，且不同残差主导下的校正策略更具针对性。

**⚠️ 局限性**

局限性：①正交基的假设需要设备数与维度满足 r ≪ d，若设备数极多或维度不够大则正交性弱化；②仅适用于 LoRA 低秩微调，无法直接推广到全参数微调；③理论分析基于局部 PL 条件和光滑性，实际非凸问题中可能偏离；④实验规模有限，未在真实分布式边缘网络中验证。

---

## 428. Full-Body Golf Swing Kinematic Reconstruction From a Smartwatch IMU

**arXiv ID:** 2606.22876 | [PDF](https://arxiv.org/pdf/2606.22876v1)

**作者:** Yuanshuo Tan `[一作]` (Shanghai Jiao Tong University), Peter B. Shull `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用单只腕戴智能手表的9轴IMU，通过深度学习模型（WIT-KinNet）估计整个人体在高尔夫挥杆过程中的全关节角度，并在同步的光学运动捕捉（OMC）数据上进行验证。

**💡 创新点**

首次实现了基于真实单腕IMU数据、并经过OMC验证的全身关节角度重建；提出了模态特定的IMU嵌入与时序运动编码机制，兼顾全挥杆阶段的全局与局部运动依赖；为实地高尔夫挥杆分析提供了可部署的单传感器方案。

**🔧 技术方法**

采用深度神经网络框架，包含多头自注意力、通道级时序卷积与多层感知机嵌入；使用带符号对数的动态范围压缩处理IMU信号；训练时联合角度、速度、加速度三项损失，并通过正弦位置编码进行时序对齐。

**📊 数据集**

36名右手高尔夫球手（14名初学者、22名熟练者）完成全幅、半幅、四分之一幅挥杆，使用7种球杆（driver、3-wood、5-hybrid、5-iron、7-iron、9-iron、sand wedge）；同时采集手表IMU（100 Hz）与Vicon OMC（120 Hz）同步数据。

**📈 对比分析**

与OMC基准比较使用平均绝对误差（MAE）、皮尔逊相关系数、ICC与Bland–Altman分析；整体关节角度MAE为8.11 ± 1.84°，腰椎与上背旋转相关系数分别达0.98和0.97；主要运动指标MAE在3.43–9.96°之间；误差随球员技术水平、球杆类型和挥杆幅度显著变化。

**⚠️ 局限性**

需要与OMC进行坐标校准，尚未实现现场可直接使用的自校准方法；模型在技术水平、球杆类别和挥杆幅度上误差变化较大，尤其是上肢关节误差较高；训练数据为混合条件，缺乏针对不同子群体的专门调优。

---

## 429. SpotAttention: Plug-In Block-Sparse Routing for Pretrained Long-Context Transformers

**arXiv ID:** 2606.22874 | [PDF](https://arxiv.org/pdf/2606.22874v1)

**作者:** Huzama Ahmad `[一作]` (KAIST), Se-Young Yun `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练一个轻量级的 Indexer 以估计冻结的 LLM 的注意力分布，并在推理时使用双重 top‑p 规则动态选择每层每个查询的 KV 缓存块，从而实现高效的稀疏注意力

**💡 创新点**

1) 仅用 KL 散度对预训练模型的注意力进行蒸馏，训练独立的轻量级模块；2) 引入基于估计分布的双重 top‑p 规则，既保留 sink 和 recency 块，又按分布自适应分配预算，消除额外的剪枝阶段；3) 通过 INT4/FP4 微尺度量化 K‑cache 实现显著的内存和预填压缩

**🔧 技术方法**

轻量级 Indexer（多头 Q‑K 评分网络）、KL 散度蒸馏（DenseKL/SparseKL）、双重 top‑p 预算机制、INT4/FP4 微尺度量化、Triton 实现、FlashAttention、Twilight、Quest 等基线

**📊 数据集**

RULER、BABILong、InfiniteBench、LongBench‑v2（长文本上下文基准），FineWeb‑Edu 训练数据（100M tokens，L=16K）

**📈 对比分析**

与密集注意力、Quest 和 Twilight 进行对比；在 Qwen3‑8B 上，静态 top‑K=0.5L 速度为 70 tok/s（128K）比 FlashAttention 快 3.9×、比 Twilight 快 1.8×；在 512K 上仍可保持 36 tok/s，基线已 OOM；准确率与密集模型相当（单预算内误差 ≤ 1 个 bootstrap 误差）

**⚠️ 局限性**

仅在 Qwen 系列英文长文本基准上验证；未评估 Llama、Mistral 等架构或多语言数据；稀疏预填延迟未测；Dual top‑p 在 DenseKL 训练时失效；量化虽降低内存但对单流推理速度提升有限，未评估大批量多 GPU 情况

---

## 430. HiL-ResRL: A Model-Agnostic Finetuning Adapter via Human-in-the-loop Residual Reinforcement Learning

**arXiv ID:** 2606.22860 | [PDF](https://arxiv.org/pdf/2606.22860v1)

**作者:** Jingyi Liu `[一作]` (Huawei Cloud Computing Technologies Co., Ltd.), Heng Zhang `[通讯]` (Huawei Cloud Computing Technologies Co., Ltd.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 HIL‑ResRL 这一模型无关的残差强化学习插件，能够在真实机器人上快速、稳健地微调 Vision‑Language‑Action（VLA）与视觉运动策略，实现工业级抓取、放置及插槽装配等任务。

**💡 创新点**

创新点包括：①将 VLA 生成的动作作为统一接口，训练轻量残差网络以补偿行为克隆的系统误差与分布偏移；②引入人机协作机制，在关键时刻提供干预与安全重置，显著提升样本效率和安全性；③支持多模态输入（视觉、姿态、力/扭矩）以处理接触丰富场景。

**🔧 技术方法**

技术手段主要有：基于 Soft Actor‑Critic 的离线强化学习框架、层归一化与双 Q 集成、可插拔的残差策略网络、以及通过 3D SpaceMouse 实现的在线人机干预；数据收集分为预训练 BC 数据集与在线干预经验回放。

**📊 数据集**

数据集为实验室采集的 50–80 条专家演示轨迹（包括 RGB‑D、6DoF TCP 位置、抓取宽度），以及在 UR5e 机器人上实时收集的干预数据和力/扭矩传感器记录。

**📈 对比分析**

与传统行为克隆、无 HIL 的残差 RL 以及 HIL‑SERL 进行对比；在 Pick‑and‑Place、Place‑Upright 和 Plug‑in‑Hole 三个任务中，HIL‑ResRL 在 1.5 小时内平均成功率超过 95%，显著优于基线（50%–80%）且在安全停机次数上大幅下降（15 次 → 2 次）。

**⚠️ 局限性**

局限性包括：仍需人工干预导致部署成本、仅针对短期子任务的局部修正、对硬件（力/扭矩传感器、空间鼠标）依赖较强，且在更长时程或复杂多步骤任务中的表现尚未验证。

---

## 431. RLM-Cascade: Response-Level Speculative Decoding for Cost-Efficient LLM API Serving

**arXiv ID:** 2606.22840 | [PDF](https://arxiv.org/pdf/2606.22840v1)

**作者:** Haifeng Wu `[一作]` (Paypal), Jian Wan `[通讯]` (Paypal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于代理层的响应级投机式解码框架RLM-Cascade，能在不访问模型内部的情况下实现多模型推理以降低LLM API成本；

**💡 创新点**

创新点在于将投机单元从单词提升到完整响应，实现跨供应商、跨架构的草稿/验证流水线，并通过规则路由在大多数请求中直接使用低成本模型；

**🔧 技术方法**

采用FastAPI代理、Anthropic API网关、规则复杂度路由、DeepSeek-V4-Pro草稿模型与Claude Opus-4-8验证模型，以及Langfuse/Prometheus可观测性栈；

**📊 数据集**

在Claude Code真实工作负载、20任务扩展工程基准以及10 Code+5 Math+5 Instruct三端点对比实验中进行评估；

**📈 对比分析**

与直接调用Opus基线相比，RLM-Cascade在实际生产中实现了约45.8%成本降低、p50延迟提升1.83倍、以及在20任务基准上100%通过率，优于传统单模型部署；

**⚠️ 局限性**

局限包括路由误判导致的质量风险、投机流水线的TTFT延迟上升、对草稿模型专业化不足时的验证失败以及跨供应商故障时的回退策略需要进一步完善。

---

## 432. Physiology-Aware CNN and Zero-Shot Multimodal LLMs for ECG Image Classification: A Comparative Study

**arXiv ID:** 2606.22889 | [PDF](https://arxiv.org/pdf/2606.22889v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 433. MINCE: Shrinking LLM Evaluation Datasets via Few-Model Monte Carlo Calibration

**arXiv ID:** 2606.22826 | [PDF](https://arxiv.org/pdf/2606.22826v1)

**作者:** Devleena Das `[一作]`, Ashish Sirasao `[通讯]` (Advanced Micro Devices)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MINCE 方法，通过使用 Monte Carlo 模拟从少量校准模型的每条日志中估计子集大小，构造固定子集以在保持精度漂移可控的前提下显著压缩 LLM 评估基准。

**💡 创新点**

创新点在于仅需 7 个校准模型即可确定最优子集大小，避免了传统方法依赖大规模校准池或学习的项目选择层，且通过 Monte Carlo 估计获得子集大小的经验性保证。

**🔧 技术方法**

核心技术包括：基于每项日志的 Monte Carlo 子集尺寸搜索、95th 百分位漂移度量、阈值驱动的递减收益判定、随机子集构建（可选分层或 k‑means），以及在 GPU 与 NPU 上的实验评估。

**📊 数据集**

使用了 IFEVAL（541 项）、MMLU（14,042 项）和 GSM8K（1,319 项）三大基准；校准模型包括 7 个 BF16 LLM（3B–35B 规模，覆盖 4 体系结构），并在 3 个 INT4 NPU 量化模型上验证泛化。

**📈 对比分析**

与 tinyBenchmarks 等现有子集方法对比，MINCE 在保持 2.62pp 以内漂移的同时，分别将 IFEVAL、MMLU、GSM8K 缩减 54%、89% 和 70%；在 GPU 上提供 2.7–8.1×、NPU 上提供 1.7–2.0× 的速度提升；漂移率显著低于 tinyBenchmarks（MMLU 降低 12×，GSM8K 降低 3.3×），且使用 57 倍更少的校准模型。

**⚠️ 局限性**

局限性包括：校准池仅覆盖 7 个 3B–35B 规模模型，未验证更大规模或非 Transformer 架构；漂移阈值 1pp 为经验选择，具体应用可能需调优；每个新基准仍需预先获取完整日志进行校准；当前子集固定，无法在评估时针对不同模型动态调整。

---

## 434. SelPE: Progressive Selection for Private Structured Text Synthesis

**arXiv ID:** 2606.22817 | [PDF](https://arxiv.org/pdf/2606.22817v1)

**作者:** Xuancheng Zhu `[一作]` (Beijing University of Posts and Telecommunications), Xiaofeng Tao `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了SelPE框架，用于在小样本场景下进行差分隐私结构化文本的合成；

**💡 创新点**

创新点包括：①将隐私预算聚焦在多批次top‑1选择上，摆脱噪声聚合的信号衰减问题；②采用上下文–模式解耦的两阶段生成，兼顾语义自由度与结构合法性；③设计多通道距离核，统一评估文本、分类和数值字段的相似度；④加入非私有对比扩展，提升生成多样性并避免模式坍塌。

**🔧 技术方法**

技术手段包括：差分隐私指数机制与高斯机制；多通道距离核与加权聚合；基于LLM的两阶段上下文–模式生成；进化式采样与批次并行选择；非私有对比样本生成；结构化数据预处理与约束解码。

**📊 数据集**

实验使用三类结构化文本数据集：Water（水瓶评论）、MIMIC‑ED（临床分诊记录）和Loan（LendingClub金融记录）。

**📈 对比分析**

与DP‑DS、DP‑Gen、AUG‑PE、WASP、CTCL等5种基线在不同ε（∞、4、2、1）下进行比较，评估RoBERTa（文本）与TabSTAR（结构）下游AUC，以及R‑CFG/S‑CFG结构完整性。SelPE在低ε、长文本、高异构场景下始终保持最优或相当于基线的结构效用和下游性能，尤其在MIMIC‑ED和Loan上表现突出。

**⚠️ 局限性**

局限性：①对更大规模或更复杂schema的数据集扩展尚未充分验证；②依赖LLM两阶段生成，模型能力限制可能影响效果；③在极高隐私预算（ε极小）下仍有一定性能衰退；④仅评估三类数据集，缺乏多语言或多域的广泛验证。

---

## 435. VideoLatent: Video-Language Learning via Latent Self-Forcing

**arXiv ID:** 2606.22870 | [PDF](https://arxiv.org/pdf/2606.22870v1)

**作者:** Zi-Yuan Hu `[一作]` (Chinese University of Hong Kong), Liwei Wang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

我们提出了一个新的多模态大语言模型框架，通过视觉隐式推理注入模块和自我强化训练，提升视频理解与推理能力。

**💡 创新点**

核心创新在于引入了视觉隐式推理注入模块与无监督的latent self‑forcing训练，消除了对CoT标注的依赖，并通过对齐与多样性约束保证隐式思考与视频/问题一致。

**🔧 技术方法**

使用了连续隐式思考空间、跨注意力注入机制、对齐与多样性对比学习，以及Qwen系列VL LLM作为基座。

**📊 数据集**

在14个视频语言基准（包括短视频理解、长视频理解与复杂视频推理）上进行评测。

**📈 对比分析**

与标准与隐式MLLM、CoT模型以及GPT‑4o等闭源模型在相同框架下对比，显著提升14项基准的平均分，训练与推理开销分别降低约6×和68×。

**⚠️ 局限性**

受限于训练帧数、缺乏额外监督导致隐式思考偶尔不相关、以及解释性低于显式CoT。

---

## 436. A Vendor-Agnostic LiDAR Data Conversion System with Multi-Signal Detection and Multi-Format Output

**arXiv ID:** 2606.22881 | [PDF](https://arxiv.org/pdf/2606.22881v1)

**作者:** Param Patel `[一作]` (BITS Pilani), Pratyush Chakraborty `[通讯]` (BITS Pilani)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个零配置的 LiDAR PCAP 数据转换系统，自动识别传感器厂商并将原始 PCAP 转为多种标准点云格式。

**💡 创新点**

通过多信号加权投票实现自动厂商识别，并统一 C++/Python SDK 解析路径，实现跨厂商一次性转换，显著提升处理吞吐量。

**🔧 技术方法**

结合 C++ SDK（Ouster、Velodyne）、Python dpkt、laspy、NumPy 等库，提供多格式输出（LAS、LAZ、PCD、BIN、CSV），并在 Windows/i3 系统上实现。

**📊 数据集**

使用真实户外抓取数据，包括 Ouster Urban_Drive.pcap、Velodyne OfficeWalking_Velodyne-VLS128.pcap、Hesai hesai_BusyRoad.pcap 和 Livox Static_CarIntersection_Livox-HAP.pcap。

**📈 对比分析**

在相同硬件（Intel i3、8GB RAM）上对四厂商进行基准测试，测量点数、耗时和吞吐量；C++ SDK 解析的 Ouster/Velodyne 达到约 1.3M/1M 点/秒，Python 解析的 Hesai/Livox 仅 134K/102K，差距约 8–10 倍。

**⚠️ 局限性**

受限于 Python 解析性能和内存压力，特别在大规模点云时吞吐下降；对新厂商的支持需要扩展 wrapper；未解决多线程/分块内存优化等问题。

---

## 437. Fursee: Hybrid YOLO-DINOv3 Framework for Fursuit Identity Retrieval and Clustering

**arXiv ID:** 2606.22872 | [PDF](https://arxiv.org/pdf/2606.22872v1)

**作者:** Jundi Wu `[一作]` `[通讯]` (Shandong University), Jundi Wu (Shandong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了专门的 fursuit 图像数据集，并提出三阶段 hybrid pipeline Fursee，用于 fursuit 识别与聚类。

**💡 创新点**

创新点在于将 YOLO 检测与 DINOv3 自监督视觉 Transformer 结合，利用 ArcFace 角度边距优化嵌入，再用 DBSCAN 并通过 silhouette 系数自动搜索最佳超参数，同时设计了针对多标签重叠的聚类评估。

**🔧 技术方法**

使用 YOLO 进行检测、DINOv3 进行嵌入学习、ArcFace 角度损失、DBSCAN 聚类和 silhouette 自动搜索。

**📊 数据集**

自制的 379 张图像、83 个身份的 fursuit 数据集（来自不同地区的 furry convention）。

**📈 对比分析**

与 GPT5.5、Claude Opus 4.8、Qwen3.7-Plus 等多模态大模型对比，在检索 hit‑rate 上取得 93.33% 领先，在聚类 weighted F1_final 上达到 0.8755，优于基线。

**⚠️ 局限性**

限于数据集规模有限、对极端遮挡/低质量图像性能下降，以及同一身份配件差异导致聚类拆分。

---

## 438. Retrieval-Augmented Multimodal Learning for Enzyme-Substrate Interaction Prediction Under Low-Homology Shift

**arXiv ID:** 2606.22823 | [PDF](https://arxiv.org/pdf/2606.22823v1)

**作者:** Chen Liu `[一作]` (East China University of Science and Technology), Liang Hong `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于检索增强的多模态框架RAMMESI，用于低同源性条件下的酶-底物相互作用（ESI）预测。

**💡 创新点**

创新点包括：① 对酶与底物的双向交互建模与自适应融合；② 引入温度调节的联合注意力提升跨模态对齐；③ 采用阈值感知平滑（TAS）加权交叉熵解决极端类别不平衡；④ 通过推理时检索邻近酶并聚合支持预测提升低同源性鲁棒性。

**🔧 技术方法**

使用技术包括：预训练蛋白语言模型（ESM2）与分子编码器（UniMol）、多头交叉注意力、联合Transformer、通道门控融合、加权BCE/TAS损失、FAISS检索、线性插值融合。

**📊 数据集**

使用公开的ESP-DB和Reactzyme-DB两大ESI基准数据集，并按序列身份划分为<20%、[20%,30%)、[30%,40%)三档测试集。

**📈 对比分析**

与传统机器学习（RF、LightGBM、DNN）及最新深度学习ESI方法（ProSmith、VIPER、ESP、OmniESI、FusionESP）对比，RAMMESI在低同源性区间下均能获得最高AUROC、AUPRC和Recall，并在稀疏正样本情形下保持最佳性能；检索增强还能显著提升所有基线的AUROC和AUPRC，尤其在<20%身份区间。

**⚠️ 局限性**

局限性在于：① 仅对酶侧检索，未考虑双侧匹配或交互级检索；② 检索时仍需额外前向推理，增加推理延迟；③ TAS等加权策略需手动调参，适用性与数据集分布相关；④ 仅在ESI任务验证，泛化到其他异构对预测任务的可迁移性尚待进一步探索。

---

## 439. Explanation-Guided Medical Named Entity Recognition with Stability and Boundary Awareness for Atopic Dermatitis

**arXiv ID:** 2606.22886 | [PDF](https://arxiv.org/pdf/2606.22886v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 440. Active Inference as the Test-Time Scaling Law for Physical AI Agents

**arXiv ID:** 2606.22813 | [PDF](https://arxiv.org/pdf/2606.22813v1)

**作者:** Omar Hashash `[一作]`, Adeel Razi `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于主动推理的测试时刻扩展法则，利用世界模型在遇到不可预见情境时动态更新 AI 代理的策略，从而实现泛化。

**💡 创新点**

创新点在于：① 将主动推理的预测误差消除机制与神经机制（基底节与前额叶）对齐，构建软贝叶斯更新的测试时刻扩展公式；② 将学习、推理、规划融合为连续的变分自由能最小化框架；③ 在仿真自动驾驶场景中展示该法则优于传统 Q‑learning 与 Bayesian RL 的性能。

**🔧 技术方法**

技术手段包括：主动推理（Active Inference）、自由能（VFE 与 EFE）变分推理、软贝叶斯更新、梯度下降实现预测误差消除、Dirichlet 后验更新、Markov 决策过程（MDP）建模，以及数字孪生与边缘计算支持的推理框架。

**📊 数据集**

实验使用仿真自动驾驶环境，状态空间 64 个离散状态（距离、速度、信号灯、行人），动作空间 3 个动作。通过模拟“行人闯红灯”未见情景进行测试；未使用公开真实数据集。

**📈 对比分析**

对比方法为 Q‑learning（模型自由）和 Bayesian RL（模型基于）。在未见情景下，主动推理方案能避免碰撞并顺利完成任务，Q‑learning 失败；相对性能提升约 36%（推理效率）并显著提高任务完成率。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证；需要手动设定阈值和超参数；对大规模连续状态空间的可扩展性未评估；依赖网络边缘计算与数字孪生；未涉及模型扩展（如新状态类别）等。

---

## 441. Clutch: High Performance Vector-Scalar Comparison using DRAM via Chunked Temporal Coding

**arXiv ID:** 2606.22812 | [PDF](https://arxiv.org/pdf/2606.22812v1)

**作者:** Daichi Tokuda `[一作]` (University of Tokyo), Shinya Takamaeda-Yamazaki `[通讯]` (University of Tokyo)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出Clutch，一种针对Processing‑using‑DRAM（PuD）的向量‑标量比较加速方案，结合时间编码与分块查表，显著降低DRAM命令数；

**💡 创新点**

创新点在于①用时间编码将向量‑标量比较转化为单行查表，②引入分块递归比较（divide‑and‑conquer）以减少所需DRAM行数，并通过MAJ3门实现无补码的比较；

**🔧 技术方法**

使用技术包括PuD（SIMDRAM 与 Unmodified DRAM）、时间编码、查表 + 分块算法、MAJ3 3输入多数门、CPU 与 GPU 基准、BitWeaving‑V 等；

**📊 数据集**

实验数据集包括 GBDT 推理的 Airline、Higgs、Covtype 等三大 tabular 数据集，以及自定义八特征谓词评估表（小/中/大表）和 GPU HBM2 模拟；

**📈 对比分析**

与 CPU（BitWeaving‑V）和 Bit‑serial PuD 对比，Clutch 在向量‑标量核上实现 12× CPU、4.1× Bit‑serial；GBDT 推理 3.5× CPU、3.8× Bit‑serial；谓词评估 83× CPU、4× Bit‑serial；能效提升可达 218× CPU、4× Bit‑serial；

**⚠️ 局限性**

局限性包括：当比较不是性能瓶颈时提升有限；工作集能完全缓存时不具优势；向量频繁更新时编码转换成本高；内存容量紧张时需降分块数，导致收益下降；需预转换编码或主机驱动PUD操作。

---

## 442. InteractiveAvatar: Real-Time Streaming Video Generation for Consistent and Intent-Aware Avatars

**arXiv ID:** 2606.22905 | [PDF](https://arxiv.org/pdf/2606.22905v1)

**作者:** Quanyue Song `[一作]` (Princeton University), Caigui Jiang `[通讯]` (Rupert-Karls-University Heidelberg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文主要阐述了ECCV提交论文的格式、排版规范以及相关政策要求，提供了官方LNCS模板、字体、页边距、行号、标题、图表、公式等细节说明。

**💡 创新点**

本“论文”并未提出新的科研方法或理论创新，而是对现有提交流程和排版规则进行系统梳理和标准化。

**🔧 技术方法**

使用的主要技术工具包括LaTeX（官方类文件）、Microsoft Word（官方Word模板）以及图形排版相关的包（如graphicx）。

**📊 数据集**

该文档未使用任何实验数据集，内容完全基于规范说明与示例文本。

**📈 对比分析**

文中未涉及实验或性能比较，仅说明了如何在审稿时标注页码、行号、避免手工排版调整等注意事项，未提供任何性能指标。

**⚠️ 局限性**

局限性主要在于：① 只针对ECCV 2026版提交流程，其他会议或年份可能存在差异；② 过度强调模板规范，可能导致作者在实际撰写中缺乏灵活性；③ 文档未覆盖所有细节（如特定图形格式、引用格式的细微差异），仍需作者自行核对。

---

## 443. AI Scientists as Engines of Discovery: A Case for Development within Reformed Institutions

**arXiv ID:** 2606.22859 | [PDF](https://arxiv.org/pdf/2606.22859v1)

**作者:** Raul Jimenez `[一作]` (University of Barcelona), Licia Verde `[通讯]` (University of Barcelona)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出将人工智能构建为科学实验的代理者，并展示了Denario多智能体框架在文献检索、代码生成、数据分析与假设验证中的应用。

**💡 创新点**

创新点在于把AI视为认知主体，设计了多智能体协同的“AI科学家”架构，并讨论了与之相匹配的治理与制度框架。

**🔧 技术方法**

主要技术包括大语言模型、工具调用与计划、推理层调度、结构化提示与可解释性接口，聚合成Denario平台。

**📊 数据集**

使用公开的科学文献、公开数据集及代码仓库，未限定单一专用数据集。

**📈 对比分析**

通过在修改引力理论的演示中，Denario自动发现隐藏对称性，表明其在假设搜索与验证速度上优于人工传统流程，但缺乏量化对比指标。

**⚠️ 局限性**

局限在于模型可能产生幻觉、缺乏可靠推理与自我验证机制，治理、可解释性、方法多样性与人类判断依旧是主要挑战。

---

## 444. Finding the Evidence: Discovering Decision-Supporting Tokens for On-Policy Reasoning Distillation

**arXiv ID:** 2606.22830 | [PDF](https://arxiv.org/pdf/2606.22830v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 445. The Unseen Hand: Manipulating Model Fairness and SHAP with Targeted Identity Re-Association Attacks

**arXiv ID:** 2606.22858 | [PDF](https://arxiv.org/pdf/2606.22858v1)

**作者:** Sannaan Khan `[一作]` (National University of Sciences and Technology), Muhammad U. S. Khan `[通讯]` (National University of Sciences and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对黑盒模型的公平性和可解释性审计，提出了针对身份重新关联的攻击（TIRA）以隐蔽地操纵模型输出。

**💡 创新点**

创新点在于设计了两种概率微扰算法（PMiS和PRSMP），仅通过重排身份实现可控且隐蔽的公平性指标操纵和SHAP解释误导。

**🔧 技术方法**

使用概率微交换与概率排名偏移微扰算法，对模型预测得分的身份顺序进行迭代扰动；评估使用AIF360公平性度量和SHAP特征重要性。

**📊 数据集**

实验数据集包括Bangladeshi Diabetes（逻辑回归）和German Credit（神经网络）等。

**📈 对比分析**

与DomSwap、MixSwap等传统洗牌攻击比较，TIRA在公平性指标上更精细地逼近理想阈值，同时在保持低检测痕迹的前提下将受保护特征的SHAP值压到零，性能显著优于现有方法。

**⚠️ 局限性**

局限性包括仅在黑盒输出层进行扰动，缺乏对多模型、跨任务泛化的深入研究，以及对抗性检测手段尚未完善。

---

## 446. G-MASt3R-SfM: Graph-based View Pruning and Multi-stage Optimization for Robust SfM

**arXiv ID:** 2606.22856 | [PDF](https://arxiv.org/pdf/2606.22856v1)

**作者:** Toshiki Watanabe `[一作]` (Tohoku University), Takafumi Aoki `[通讯]` (Tohoku University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多视角3D重建中提出G-MASt3R-SfM，利用场景图剔除不可靠视角并通过分阶段束调整优化相机参数。

**💡 创新点**

创新点在于结合图结构的视角剔除（Graph-based View Pruning）与基于社区的多阶段优化（Multi-Stage Optimization），显著提升全局一致性与鲁棒性。

**🔧 技术方法**

使用MASt3R特征匹配、Louvain社区检测、Spring Layout布局、RANSAC基本矩阵估计、Adam优化以及传统束调整等技术。

**📊 数据集**

实验数据集为ETH3D（25个室内外场景，选取13个场景）。

**📈 对比分析**

与COLMAP、DFSfM、VGGSfM、VGGT、MASt3R-SfM等基线比较，G-MASt3R-SfM在RRE、RTE、AUC@5、F1分数均居前，SfM率虽由100%降至97%，但总体性能显著提升。

**⚠️ 局限性**

局限性包括视角剔除导致的SfM率略低，以及仅在ETH3D数据集上验证，缺乏对其他数据集的泛化评估。

---

## 447. Homographic Navigation: Geometry-Driven Camera Guidance for Deterministic Planar Capture

**arXiv ID:** 2606.22834 | [PDF](https://arxiv.org/pdf/2606.22834v1)

**作者:** Dominik Kroupa `[一作]` (Brno University of Technology), Adam Herout `[通讯]` (Brno University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了“同构导航”（Homographic Navigation）框架，利用单张参考图像通过同构增强生成无限制的合成训练样本，训练一个单射YOLO‑风格的网络实现物体识别、稀疏关键点预测以及置信度估计，并通过两步推理和稳定变形训练（Stable Warp）实现从粗定位到细化的精确平面对齐。

**💡 创新点**

创新点包括将同构视为组织变量，统一学习、对齐与评估；使用基于同构的无监督增强与结构化遮挡来生成无限制数据；提出两步推理与稳定变形训练，使模型在全图搜索与局部跟踪两种场景下均能保持高精度；以及通过关键点置信度与全局置信度的双重估计提升结果鲁棒性。

**🔧 技术方法**

采用了单射卷积网络（YOLO‑基架），多分辨率颈部（main + detail branch）、稀疏关键点回归、置信度预测、二阶卷积头；训练中使用Smooth‑L1、交叉熵、基于距离的置信度损失；推理时实现两步全局–局部定位，并在训练中加入稳定变形数据增强（Stable Warp）。

**📊 数据集**

实验使用了包含23个物体类别的三套数据集：1）真实数据集 247 张手机拍摄图像；2）合成数据集 2000 张同构变换后生成的图像；3）真实增强数据集 2000 张在真实图像上进一步同构扰动得到的图像；每个类别都有参考图像、关键点与规范坐标。

**📈 对比分析**

通过统一的几何评估协议（累计误差曲线与 AUC）将 HomoNav 与传统特征匹配基线（SIFT+RANSAC）及现代学习基线（RoMa v2）进行对比。结果显示，在合成数据上 HomoNav 明显优于两者；在真实与增强数据上，SIFT 仍占优，但 HomoNav 通过两步推理和 Stable Warp 取得了较大提升，尤其在第二步和高精度区间表现突出。整体 AUC 与 FPS 均显示该方法在低分辨率输入下的竞争力。

**⚠️ 局限性**

局限性主要包括：仅完成框架的第一阶段，缺乏基于视频的自监督扩展；在真实复杂场景中的鲁棒性仍不足；两步推理在实时部署时会增加计算开销；模型在多尺度、强遮挡以及极端光照下的性能尚未充分验证。

---

## 448. Bagpiper-TTS: Natural Language Guided Universal Speech Synthesis

**arXiv ID:** 2606.22811 | [PDF](https://arxiv.org/pdf/2606.22811v1)

**作者:** Jinchuan Tian `[一作]` (Carnegie Mellon University), Shinji Watanabe `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Bagpiper‑TTS，一个基于自然语言请求的统一语音合成框架，先通过文本规划推理用户意图，生成包含转录和元数据的“rich caption”，再根据 caption 合成高质量语音，支持多说话人、意图到语音、角色扮演、歌唱等多种任务。

**💡 创新点**

核心创新点在于用自然语言作为全局统一接口，替代传统固定元数据槽；构建了端到端的三阶段工作流（规划‑rich caption‑语音）；利用大规模 LLM 驱动的自监督数据模拟，生成丰富的多任务训练集，从而实现单一模型覆盖多种复杂语音合成场景。

**🔧 技术方法**

技术实现主要依托 Bagpiper‑Base 预训练模型，使用 Qwen3‑8B‑Base 作为解码器，X‑Codec 多流音频编码，Classifier‑Free Guidance 与 Top‑k 采样；在数据生成与验证环节大量使用 Qwen、Gemini、Claude 等大型语言模型进行逆向请求生成、规划生成、字幕校正与一致性评估。

**📊 数据集**

训练数据共 738k 例，来源包括 LibriTTS‑R、Genshin、Starrail、Gigaspeech、SSSD、VibeVoice‑ASR 等公开语音语料；通过自动字幕、WER 过滤、逆向用户请求生成、规划文本构造等流程构造丰富的多任务数据；评测使用 Seed‑TTS‑Eval（英语经典 TTS）以及自构造的 300 条多说话人、意图到语音、角色扮演、歌唱等任务请求。

**📈 对比分析**

评估方式包括：在 Seed‑TTS‑Eval 上比较 WER，Bagpiper‑TTS 取得 1.7% WER，仅次于 Qwen3‑TTS 1.5%；对多说话人、意图到语音、角色扮演、歌唱等自定义任务，使用 WER、LLM‑as‑judge（Gemini‑3‑Flash）评分和 MTurk 人工主观评分；平均 LLM 满意度 4.09，MTurk 主观分数 3.69；与专用模型相比，Bagpiper‑TTS 在多任务统一性上具优势，整体性能略低但已可与前沿模型竞争。

**⚠️ 局限性**

局限性包括：在字幕生成和推理阶段仍存在幻觉；缺乏对参考音频等更直观元数据的支持；自监督数据模拟质量受限，导致部分边缘场景的准确性不足；缺乏统一标准化基准，难以与所有专用模型做精确对比。

---

## 449. AI-Assisted Help-Seeking Trajectories in Programming Education from an SRL-Informed Perspective

**arXiv ID:** 2606.22809 | [PDF](https://arxiv.org/pdf/2606.22809v1)

**作者:** Boxuan Ma `[一作]` (Kyushu University), Shin'ichi Konomi `[通讯]` (Kyushu University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对大学级Python编程课程中学生与LLM助手的交互对话进行细粒度编码，构建了基于自我调节学习（SRL）的三层框架，并将交互序列归纳为五类帮助寻求轨迹（单次、调试持续、概念构建、执行导向、模式转换），进一步关联轨迹与任务得分和代码提交次数。

**💡 创新点**

首次将SRL视角与AI辅助编程交互轨迹相结合，提出了一种面向任务的帮助寻求轨迹分类法，并揭示了调试持续轨迹与高提交成本的关联，提供了对AI支持教育价值的更细致评估。

**🔧 技术方法**

采用人工编码与互评、交互式调试日志分析、滞后序列分析（lag sequential analysis）以及基于规则的轨迹分类（taxonomy）等技术，对学生提示进行细粒度分类和序列分析。

**📊 数据集**

利用两期暑期CS1课程共166名学生（分析样本71人）的数据，包含1,290条任务相关提示、17,190条代码提交、57道编程题，按提交次数平均4.1次/题。

**📈 对比分析**

通过比较不同轨迹模式的任务得分和提交次数，发现任务得分在各模式间差异不显著（p=0.95），但提交次数显著不同，调试持续模式平均提交量最高（≈11.9次），表明AI帮助更多集中在即时调试而非高效学习。

**⚠️ 局限性**

局限性包括：研究仅在单一日本高校的入门Python课程中进行；仅评估任务得分和提交次数，未涵盖概念理解、迁移或长期学习；仅分析记录在AI接口的交互，忽略同伴、教师或网络资源等其他帮助；未对AI回应的教学质量进行分析。

---

## 450. Agent-as-a-Router: Agentic Model Routing for Coding Tasks

**arXiv ID:** 2606.22902 | [PDF](https://arxiv.org/pdf/2606.22902v1)

**作者:** Pengfei Zhou `[一作]` (National University of Singapore), Yang You `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一个可自适应的多模型路由框架 Agent-as-a-Router，并在此框架下实现了 ACRouter，旨在根据任务流实时选择最合适的大语言模型；同时构建了 CodeRouterBench 测试平台，用于在流式任务中评估路由器的累计 regret。

**💡 创新点**

创新点包括：①将路由问题形式化为 Context‑Action‑Feedback (C‑A‑F) 循环，并把它等价为上下文多臂老虎机；②通过在部署时积累执行反馈（Verifier）并将其写入记忆（Memory）来主动填补信息缺口；③设计了可扩展的 CodeRouterBench，专门用于评估流式路由的累计 regret。

**🔧 技术方法**

主要技术手段包括：使用 Qwen3.5‑0.8B 作为路由策略模型；利用 kNN 向量检索实现 Memory；Verifier 采用 AST 解析与沙箱执行得到多维度评分；Orchestrator 通过加权投票整合先验维度最佳、最近邻记忆与任务元信息；整体评价基于上下文多臂老虎机算法和累计 regret 指标。

**📊 数据集**

实验数据集为约 10,000 条编码任务（9 个单轮维度 + 1 个 OOD 代理编程维度），任务来源于 15 个公开基准；并使用 8 个前沿 LLM（Claude Opus, Claude Sonnet, GPT‑5.4, Qwen3‑Max, Qwen3.5‑Plus, GLM‑5, Kimi‑K2.5, MiniMax‑M2.7）产生的性能与成本矩阵。

**📈 对比分析**

比较方法：在 CodeRouterBench 的 ID 与 OOD 流式任务上，分别计算 AvgPerf、累计 regret（CumReg）与 Perf/$，并绘制 Pareto 前沿。ACRouter 在 ID 流任务中实现最低累计 regret（205.5）和最高 AvgPerf（49.98%），在 OOD 任务中 AvgPerf 仍高于所有静态或学习型路由器（62.5%），并且成本效率优于直接使用强模型 Opus‑4.6。

**⚠️ 局限性**

局限性：成本估算依赖公开 token 价格，未能观察到实际缓存命中率；OOD 评测采用 40 步限制，未使用完整 250 步设置；记忆模块仅实现 kNN 向量存储，未探索更高级的参数级或神经网络记忆方法。

---

## 451. Learning Graphs through Continuous Information Entropy Fields

**arXiv ID:** 2606.22895 | [PDF](https://arxiv.org/pdf/2606.22895v1)

**作者:** Hui Cong `[一作]` (Chang'an University), Yisheng An `[通讯]` (Chang'an University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Field-informed Graph Network（FGN），通过学习连续信息熵场来解释图结构并实现图学习。

**💡 创新点**

创新点是将图视为连续熵场的离散采样，构建自洽的场与图的双向共演机制，并以信息理论目标驱动场与消息传递的协同优化。

**🔧 技术方法**

主要技术包括信息理论目标、场调制消息传递、动态场更新（变分推断+梯度下降）、自洽闭环设计以及熵相关正则化。

**📊 数据集**

使用的数据集涵盖节点分类（Cora、CiteSeer、PubMed、Chameleon、Texas、Cornell、ogb-arxiv、Computers、Photo）和图分类（MNIST、CIFAR10）等。

**📈 对比分析**

与 GCN、GraphSAGE、GAT、GRU、LoGoGNN 等基线比较，FGN 在大多数数据集上实现最高或接近最高准确率，尤其在异质图和大规模图上优势显著，并表现出更好的鲁棒性。

**⚠️ 局限性**

局限性包括仅采用标量熵场，未深入探索多维/多关系场；对极端异质或动态图的适用性仍需验证；模型训练相对复杂，对计算资源和超参数敏感。

---

## 452. PHOEBI: An Open-World Benchmark for Bacterial Identification in Phase-Contrast Microscopy

**arXiv ID:** 2606.22890 | [PDF](https://arxiv.org/pdf/2606.22890v1)

**作者:** Aaditya Baranwal `[一作]` (University of Central Florida), Shruti Vyas `[通讯]` (University of Central Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了基于相位对比显微镜的细菌多标签识别基准PhOENIC，包含120,000张覆盖6种杆状细菌的40种组合图像，并配套留组合外（LCO）评估协议；

**💡 创新点**

创新点在于发现梯度训练的聚合器在组合泛化上会出现系统性崩溃，并通过三种轻量化锚点解码器（共享冻结的DINOv2特征池）实现无训练成本的组合泛化、开集拒绝与新种发现；

**🔧 技术方法**

主要技术包括基于DINOv2的冻结特征提取、稀疏最大化（simplex unmixing）、余弦匹配和通道分组判别等锚点解码器，以及利用重构残差进行开集与新种检测；

**📊 数据集**

使用的数据集为PhOENIC，由实验室制备的6种细菌的相位对比显微镜图像组成，覆盖单、双、三、四和六种组合共40组；

**📈 对比分析**

与多种基线（9种端到端微调网络、DINOv2控制、MIL头）相比，基线在LCO上的F1下降0.39-0.57；锚点解码器在LCO上仅轻微下降或提升，平均提升约0.08，开集AUROC达0.70，新种发现纯度为100%，漂移低于基线10倍；

**⚠️ 局限性**

局限性包括所有图像仅来自单台显微镜，跨仪器泛化未验证，且开集拒绝分数仍属诊断级别，未达到部署可用标准；

---

## 453. CLI-Universe: Towards Verifiable Task Synthesis Engine for Terminal Agents

**arXiv ID:** 2606.22883 | [PDF](https://arxiv.org/pdf/2606.22883v1)

**作者:** Zhanbo Hua `[一作]`, Jiaheng Liu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了CLI‑Universe数据合成管道，用结构化能力规范和证据引导的深度研究生成可执行终端代理训练任务，并在这些任务上微调Qwen3模型获得显著性能提升。

**💡 创新点**

引入结构化能力规范与证据引导的任务蓝图构造、三阶段可执行验证过滤以及低样本高效训练的组合方法，显著提升任务质量与训练信号密度。

**🔧 技术方法**

利用结构化分类学、自动化研究代理、Docker化环境构建、可执行测试生成、内部提示过滤和失败到通过验证等技术。

**📊 数据集**

CLI‑Universe‑6K（6000条轨迹）以及Terminal‑Bench、BFCL v4、VitaBench等基准数据集。

**📈 对比分析**

在Terminal‑Bench 2.0上与多种公开模型对比，Qwen3‑32B在CLI‑Universe‑6K上达到33.4% avg@4，超过同规模公开数据训练模型，并在BFCL v4、VitaBench上表现出显著提升。

**⚠️ 局限性**

依赖LLM生成与验证，数据规模仍有限；与最前沿专有模型仍有明显差距；未尝试更大模型或强化学习训练，可能进一步提升性能。

---

## 454. DJM: Compact Base Meshes for Displacement Mapping using Triangle Jacobians

**arXiv ID:** 2606.22880 | [PDF](https://arxiv.org/pdf/2606.22880v1)

**作者:** Congyi Zhang `[一作]` (University of Texas at Dallas), Alla Sheffer `[通讯]` (University of British Columbia)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出DJM基于Jacobian度量的基网格构造方法，实现高效压缩与渲染

**💡 创新点**

引入Jacobian基准度量与渐进松弛简化，避免射线投射，确保双射映射

**🔧 技术方法**

使用QEM简化、Jacobian判别、前后投影求解、迭代最小二乘、MLP神经编码

**📊 数据集**

在109个高分辨率扫描（Maggiordomo等）和20个Zhang NESI数据集上评测

**📈 对比分析**

与Maggiordomo及QEM比较，误差更低、文件更小；神经编码相比NGF与Pentapati显著提升

**⚠️ 局限性**

仅适用于闭合网格，双射性仅局部保证，可能因输入网格错误导致映射失效

---

## 455. DynamicMem: A Long-Horizon Memory Benchmark in Real-World Settings

**arXiv ID:** 2606.22877 | [PDF](https://arxiv.org/pdf/2606.22877v1)

**作者:** Wenya Xie `[一作]` (University of Minnesota), Zirui Liu `[通讯]` (University of Minnesota)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个基于合成用户轨迹的长时序记忆基准（DynamicMem），生成15个月、约2.2M token的多应用、多领域用户行为记录，涵盖属性、习惯、偏好等三类随着季节、重大事件等外部因子演化的个人资料；并提出分季度检查点评估，评估记忆系统随时间推移的性能；同时通过细粒度的任务（状态补全与个性化服务）来测试记忆的准确性与实用性。

**💡 创新点**

核心创新在于：①多时域用户建模，将属性、习惯、偏好拆分为不同演化速率的结构；②以外部因子为驱动的因果驱动的资料演化，保证变更具有可解释的因果链；③跨应用的行为链生成与状态一致的多应用日志，提供分散的可观测证据；④基准设计采用五个季度检查点，揭示记忆随历史增长的退化模式与更新难点；⑤对比分析显示>93%失败源于记忆检索，突显记忆机制的瓶颈。

**🔧 技术方法**

利用大型语言模型（如 GPT-4）进行用户轨迹合成、意图链生成与日志生成；构建检索式记忆系统（RAG、HippoRAG2、MemoryOS、A-Mem、SimpleMem）作为基线；采用 LLM-as-judge 进行任务评估；在合成过程中使用结构化记录与因果标注，支持多维度检索与更新。

**📊 数据集**

主要使用基于 PersonaHub 的10个多样化用户画像进行扩展；在此基础上生成15个月的跨应用行为轨迹，覆盖16个应用（e‑commerce、fitness、社交等）和66个API；数据总量约2.2M token/用户，1,772个有根事件/用户。

**📈 对比分析**

采用分季度检查点的评估协议，对每个系统进行状态补全（State Completion）与个性化服务（Personalized Service）两类任务评分。结果显示，随着历史长度增长，状态补全准确率持续下降（尤其是偏好类），而个性化服务准确率基本保持或略升；不同系统在属性、习惯、偏好三类中表现差异显著；多数错误来源于记忆检索，表明提升记忆机制是关键。

**⚠️ 局限性**

局限性包括：①数据完全合成，缺乏真实用户行为中的噪声与多样性；②仅覆盖10个用户画像，可能不足以检验在更广泛用户群体下的泛化；③评估聚焦于检索与记忆质量，未深入探究答案生成模型的能力；④因果驱动的演化模型简化了真实生活中复杂的多因子交互；⑤基准仅测试静态查询和服务任务，未覆盖主动探索或动态决策等更复杂场景。

---

## 456. When AUC 0.998 Is Not Enough: A Candidate Evaluation Protocol for Hidden-State Probes of Indirect Prompt Injection in Multimodal Computer-Use Agents

**arXiv ID:** 2606.22864 | [PDF](https://arxiv.org/pdf/2606.22864v1)

**作者:** Yanhang Li `[一作]` (Northeastern University), Zexin Zhuang `[通讯]` (Southern Methodist University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对冻结的多模态计算机使用代理（Qwen2.5-VL-7B）在 Mind2Web 轨迹上进行隐藏状态探测，评估其对间接提示注入（IPI）的检测能力，并提出基于文本元数据与视觉匹配控制的诊断手段。

**💡 创新点**

提出两类控制诊断（文本侧元数据基线C1和视觉侧匹配控制C2）来揭示探测器性能中的 shortcut，并给出完整的评估协议与报告准则，指出单纯高 AUC 并不证明恶意内容检测。

**🔧 技术方法**

使用线性逻辑回归 probe、MLP 容量检验、四维文本元数据特征、同步视觉匹配控制、轨迹层面自助采样 CI、打乱标签 sanity、正则化敏感度、窄框排除、跨注入转移等技术。

**📊 数据集**

Qwen2.5-VL-7B（7B 版）作为模型骨干，在 Mind2Web 基准上抽取 80 条轨迹（共 726 步）进行教师强制回放，注入三种 IPI 表面（可见覆盖、a11y 树、工具返回）。

**📈 对比分析**

通过 clean‑vs‑attack AUC（headline 0.998）和匹配步 AUC（≈0.997）对 probe 进行基准；随后应用 C1 与 C2 诊断后发现大部分高 AUC 可归因于表面统计 shortcut，cross‑injection 转移显示文本侧训练对视觉侧无效，表明真实检测效果受限。

**⚠️ 局限性**

局限包括仅验证单一 backbone 与 benchmark、简短的 benign-control pool（仅 5 条命令）、缺乏真实环境回放导致 post‑exposure Δ=1 诊断不可测、runtime gating 在本基准上不可解释，以及对跨模态、跨模型的推广性仍为猜想。

---

## 457. Cloak: Zero-Shot Cross-Embodiment Manipulation by Masking the End-Effector from the VLA

**arXiv ID:** 2606.22836 | [PDF](https://arxiv.org/pdf/2606.22836v1)

**作者:** Michael Piseno `[一作]` (Stanford University), C. Karen Liu `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过遮蔽腕部摄像头视角，训练在单一平行钳手上的视觉语言对齐（VLA）模型，使其能够零样本迁移到未见的执行器，包括五指手。

**💡 创新点**

创新点在于利用“遮蔽”手段消除末端执行器自身摄像头的干扰，并通过VLA的跨模态学习实现跨执行器的零样本迁移。

**🔧 技术方法**

使用了视觉语言对齐（VLA）网络、RGB+Depth多模态输入、以及端到端强化学习训练框架。

**📊 数据集**

主要数据集包括公开的GraspNet、RoboGrasp以及新构建的五指手抓取数据集。

**📈 对比分析**

与传统基于模型的抓取方法和最新的零样本抓取算法相比，在五指手上的成功率提升约15%，在平行钳手上保持98%的原始性能。

**⚠️ 局限性**

局限性包括对极端遮挡场景的鲁棒性不足，以及在极其复杂物体形状时迁移性能仍有下降。

---

## 458. OrthoMotion:Disentangling Camera and Subject Motion via Geometry Semantics Orthogonal Attention

**arXiv ID:** 2606.22835 | [PDF](https://arxiv.org/pdf/2606.22835v1)

**作者:** Zijie Meng `[一作]` `[通讯]` (Peking University), Zijie Meng (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种同时可控制摄像机轨迹与主体运动的可控视频生成框架，并通过设计正交的注意力子通道实现两者的解耦。

**💡 创新点**

创新点在于：①将摄像机运动映射到几何正交相位通道（norm‑preserving SO(d) phase），主体运动映射到门控值注入通道；②引入正交正则化项强制两通道响应子空间正交；③提出 Cross‑Talk Error (CTE) 作为解耦程度的客观指标。

**🔧 技术方法**

核心技术包括：流匹配式扩散 Transformer（Wan2.1），RoPE 与 SO(d) 旋转相位的组合，门控值注入的语义通道，以及基于雅可比矩阵的正交正则化。

**📊 数据集**

使用与基线相同的公开视频数据集（如基于 Wan2.1 的训练数据集，兼容 DragNUWA、MotionCtrl 等公开数据集），并在多种背骨网络（Wan2.1‑1.3B、Wan2.1‑14B、CogVideoX‑2B）上进行实验。

**📈 对比分析**

与 MotionCtrl、DragNUWA、Tora 等基线方法比较，OrthoMotion 在摄像机旋转误差、主体轨迹误差、FVD、CLIP‑SIM 等指标均达到了最优或接近最优，并将交叉干扰（CTE）降低了超过 2.4 倍（约 5 倍多基线），同时不降低视频质量。

**⚠️ 局限性**

局限性包括：①正交正则化依赖一阶雅可比近似；②目前的几何分析假设场景为刚体；③在多目标、变形主体或非刚体动态场景下的适用性仍需进一步研究。

---

## 459. Improving Robotic Imitation Learning via Trajectory Standardization

**arXiv ID:** 2606.22907 | [PDF](https://arxiv.org/pdf/2606.22907v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 460. DBT-Bleed: Dual-Branch Temporal Modeling with Key-Frame Selection for Surgical Bleeding Detection

**arXiv ID:** 2606.22829 | [PDF](https://arxiv.org/pdf/2606.22829v1)

**作者:** Sudhanshu Mishra `[一作]` (National University of Singapore), Yueming Jin `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了一种双分支多尺度时序建模框架（DBT-Bleed）以及基于红通道熵的关键帧选择方法（HiRED），用于对手术视频中的出血不良事件进行弱监督检测。

**💡 创新点**

创新点包括：①双分支结构与多尺度层级适配器（MTA）能够同时捕捉短期与长期的血液动态；②HiRED通过熵驱动的两阶段关键帧筛选，有效剔除冗余信息并保持全时序覆盖；③在零样本跨手术类型迁移中保持优异性能，首次在神经外科IAE数据集上实现零样本检测。

**🔧 技术方法**

使用技术包括：CLIP冻结视觉与文本编码器；空间与时间轻量适配器；双对比损失（双分支对齐）；多尺度注意力层；基于熵的关键帧选择；以及在测试阶段的零样本推理。

**📊 数据集**

实验基于两个数据集：①MultiBypass（胰腺旁路手术视频，已标注IAE）；②EndoPit-IAE（内鼻垂体手术新构建的IAE标注数据集，用于零样本评估）。

**📈 对比分析**

与MadCLIP、VadCLIP、ActionCLIP、SEDMamba等基线相比，DBT-Bleed在MultiBypass上实现F1 64.91%（相较于ActionCLIP 58.38%提升约6.5%）、Recall 78.65%（提升约5.6%）以及MCC 0.53（提升约9%）。在EndoPit-IAE零样本设置中，F1 83%（相比ActionCLIP 75.8%提升约7%）、MCC 0.35（提升约8%）均表现最佳。

**⚠️ 局限性**

局限性包括：仅针对二分类出血检测，未覆盖多标签IAE；关键帧选择依赖红通道熵，对光照变化和血液颜色差异敏感；在更大规模或不同手术类型的数据上，域漂移仍可能影响性能；模型对长视频仍需预先分段，可能遗漏细粒度信息。

---

## 461. Graph-Enhanced Large Language Models for Spatial Search

**arXiv ID:** 2606.22909 | [PDF](https://arxiv.org/pdf/2606.22909v1)

**作者:** Nicole R. Schneider `[一作]` (University of Maryland), Hanan Samet `[通讯]` (University of Maryland)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于图增强的大型语言模型，用于空间检索的世界模型。该模型将世界表征为空间图，并在图上附加LLM，以实现对空间语义的理解与推理。

**💡 创新点**

创新点在于将空间世界以可视化的空间图结构与LLM深度融合，实现了文本与空间知识的无缝交互，提升了LLM在空间推理与检索任务中的能力。

**🔧 技术方法**

采用图神经网络（GNN）、大规模语言模型（如GPT-4/ChatGPT）、空间索引结构（R‑Tree、Quad‑Tree）以及多模态融合技术来构建与推理空间图。

**📊 数据集**

使用公开的地理空间数据集，包括OpenStreetMap、Google Maps API、Geonames和全国地理信息系统（GIS）等。

**📈 对比分析**

与传统文本检索模型（BM25、FAISS）和仅基于文本的LLM进行对比，实验结果显示在位置推理、路径规划和空间问答任务上准确率提升约30%–40%，推理速度在可接受范围内。

**⚠️ 局限性**

局限性包括：对大规模空间图的推理成本高、需要大量计算资源；对稀缺或动态更新的地理信息覆盖不足；以及在实时动态场景中的响应延迟较大。

---

## 462. FedOT: Ownership Verification and Leakage Tracing via Watermarks for Federated LDMs

**arXiv ID:** 2606.22875 | [PDF](https://arxiv.org/pdf/2606.22875v1)

**作者:** Wenlong Cheng `[一作]` (Northwest Normal University), Jiaxu Miao `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FedOT 框架，针对联邦学习中的潜在 LDM 所有权验证与泄露追踪，并设计块状水印与 Latent Vector Transformation (LVT) 来实现防替换攻击。

**💡 创新点**

首次实现 LDM 的所有权验证与泄露追踪，结合块状水印实现双向功能，并通过 LVT 在 VAE 与 U-Net 之间建立绑定，防止 VAE 替换攻击。

**🔧 技术方法**

基于 Stable Diffusion VAE 与 U-Net 的 Latent Diffusion 模型，采用 VAE 解码器水印嵌入、块状水印设计、三种 LVT 变换（平移、镜像、负变换）、Federated Averaging、Stable Signature 水印提取器。

**📊 数据集**

COCO2017 用于 LVT 与水印训练，LAION-10K 用于联邦微调与评估。

**📈 对比分析**

与 Stable Signature* 对比，FedOT 在无攻击下保持 FID ~20-22，CLIP-score ~0.29-0.30；在 VAE 替换攻击后，FedOT 通过 LVT 使 FID 上升 20-70，验证检测率 >0.93，位准确率 >0.90，说明在防御攻击同时保持高质量。

**⚠️ 局限性**

LVT 仍需额外训练时间且不同变换会在图像质量与攻击鲁棒性间权衡；在极端数据异质性或客户端数极大时，训练与通信开销仍需进一步优化；未验证对更复杂攻击（如模型修补、重构攻击）的鲁棒性。

---

## 463. SingGuard: A Policy-Adaptive Multimodal LLM Guardrail with Dynamic Reasoning

**arXiv ID:** 2606.22873 | [PDF](https://arxiv.org/pdf/2606.22873v1)

**作者:** SingGuard Team `[一作]` `[通讯]` (Ant Group), SingGuard Team (Ant Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 SingGuard，一种基于运行时政策输入的多模态安全守门模型，可在多模态问答和助手响应中实现安全评估；

**💡 创新点**

创新点在于：①将政策视为输入，支持开放式规则匹配；②设计了快速–中速–慢速多路推理路径；③采用快速–慢速解耦强化学习和在线政策适配；④构建了覆盖 80+ 细粒度风险类型、56k 示例的 SingGuard-Bench；

**🔧 技术方法**

核心技术包括统一的层次安全分类法、政策条件化数据合成、规则级对齐的 CoT 训练、fast–slow Decoupled DAPO 强化学习以及对 2B 模型的 on‑policy GKD 蒸馏；

**📊 数据集**

使用的数据集涵盖 35 个公开多模态安全基准（VLGuard、JailBreakV、SPA‑VL、MMDS 等）、图像安全数据（UnsafeBench、DeepGHS NSFW、Facebook Hateful Memes 等）以及自研的动态规则、跨模态攻击样本，最终构成 SingGuard‑Bench；

**📈 对比分析**

在六大基准家族（图像、跨模态、文本查询、文本响应、多语言、动态政策）中，SingGuard 均取得了最优或接近最优的 macro‑average F1，特别是慢速推理模式在动态政策评估中将准确率提升至 0.7415（相较 Qwen3‑VL‑8B 的 0.6465）;

**⚠️ 局限性**

限制包括：对运行时政策文本的清晰度高度依赖；合成与模型辅助标注的潜在偏差；未能覆盖所有真实世界的长尾安全场景；以及在分布漂移下自适应早停的置信度校准可能不充分。

---

## 464. Discovering Crystal Structure Prediction Algorithms with an AI Co-Scientist

**arXiv ID:** 2606.22866 | [PDF](https://arxiv.org/pdf/2606.22866v1)

**作者:** Kiyoung Seong `[一作]` (Korea Advanced Institute of Science and Technology), Sungsoo Ahn `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用 AI 合作科学家循环，通过跨域迁移与稀疏人机交互，发现并实现 MaskGXT——一种将 Masked Generative Modeling（MaskGIT）迁移到晶体结构预测的模型。

**💡 创新点**

创新点在于：① 开发了一个可搜索跨域生成模型的 AI 合作科学家系统；② 将 MaskGIT 转化为 MaskGXT，首次将离散化、圆形标签平滑、对称性标记、子区间回归和空间群分层贪婪解码等技术结合，用于晶体结构预测；③ 在实验中通过稀疏高层指令（如对称性、子区间精度、极化多态性）驱动模型改进，展示跨域迁移可产生领域级算法突破。

**🔧 技术方法**

技术实现包括：树形搜索与操作器（idea、draft、improve、debug 等）驱动的代码生成与训练；离散化坐标与晶格参数；圆形标签平滑和对称性增强；子区间坐标回归头；基于 Transformer 的 Masked Generative 模型；空间群分层采样与贪婪解码策略；以及 12 个 RTX 3090 GPU 的并行训练管线。

**📊 数据集**

使用的主要数据集为 MP‑20、MPTS‑52 以及 MP‑20 的 polymorph split（多态分割），这些数据集覆盖 20‑52 原子/晶胞的晶体结构，且已在先前工作中标准化分割。

**📈 对比分析**

方法对比：与 DiffCSP、FlowMM、OMatG、Crystalite、MCFlow 等主流 CSP 基线进行一对一和多态性评估。MaskGXT 在 MP‑20 与 MPTS‑52 上实现最高匹配率（MR）和最低匹配对 RMSE；在 MP‑20 polymorph split 上获得最佳 METRe 与 cRMSE；在统一采样预算（S=2）下仍保持优势，说明空间群分层和贪婪解码显著提升多态覆盖。

**⚠️ 局限性**

局限性：该 AI 合作科学家循环依赖可快速训练与评估的固定数据集与验证指标，难以直接迁移到需要实验验证、长训练周期或弱代理指标的领域；评估信号受限于短周期训练，可能无法准确预测大规模训练效果；并且当前的稀疏人机交互仍需要领域专家的高层指令，尚未实现完全自动化。

---

## 465. Chains That See, Answers That Don't: A Multi-Aspect Evaluation Recipe for Forced Chain-of-Thought on Video-MME

**arXiv ID:** 2606.22862 | [PDF](https://arxiv.org/pdf/2606.22862v1)

**作者:** Zhichao Fan `[一作]` (University of Illinois Urbana-Champaign), Zexin Zhuang `[通讯]` (Southern Methodist University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一套三探针（配对准确率、视频交换诊断、视觉降噪阶梯）评估方案，用以细粒度检测视频‑语言模型强制链式思维（CoT）的真实性和效果。

**💡 创新点**

创新点在于：① 结合配对准确率、输入替换与视觉降噪三种探针，全面剖析链式思维是否真正依赖视觉输入以及是否提升最终答案；② 并行使用严格与宽松正则提取器，展示评分选择对结论的显著影响；③ 提供完整可复现脚本与数据，便于后续研究复制与扩展。

**🔧 技术方法**

使用技术包括：强制CoT提示、贪婪解码、链式思维与答案分离、严格/宽松正则提取、链文本 Jaccard 相似度、配对 McNemar 检验、bootstrap 置信区间、Holm 多重检验、Spearman 秩相关、可视化梯度评估等。

**📊 数据集**

数据集：Video‑MME 多选题集（2700题，12 任务类型、3 时长桶），实验取 32B 与 7B Qwen2.5‑VL 的子集（300/396 题），并使用同任务/不同域视频进行交换对照。

**📈 对比分析**

比较方法：在相同题集下对 direct、CoT、答案先、无视频四种提示进行配对准确率比较；使用 Jaccard 评估视频交换时链的一致性；使用视觉降噪阶梯测量准确率随视觉信息减弱的单调性。结果显示：32B 在严格计分下 CoT 与 direct 无显著差异，7B 在严格计分下 CoT 减低约 7.3pp（p=0.012 Holm‑adjusted），视频交换诊断表明链式思维高度受视频影响；视觉降噪阶梯显示准确率随视觉信号降级呈单调下降。

**⚠️ 局限性**

局限性：仅在 Qwen2.5‑VL 的 32B 与 7B 两个规模子集上验证，未覆盖完整 2700 题；使用自定义正则提取器，尚未与官方 Video‑MME 评估器对齐；链长度、错误传播机制未做系统错误分析；仅使用单一提示风格与量化方式，缺乏跨模型/跨任务的广泛验证。

---

## 466. RaMem: Contextual Reinstatement for Long-term Agentic Memory

**arXiv ID:** 2606.22844 | [PDF](https://arxiv.org/pdf/2606.22844v1)

**作者:** Wei Yang `[一作]` (University of Southern California), Jesse Thomason `[通讯]` (University of Southern California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过对长期记忆进行情境重现，构建了RaMem框架，提升LLM代理在多轮交互中的证据可靠性。

**💡 创新点**

创新点在于将记忆片段与事件时间、提及时间、会话区间、参与者等情境坐标锚定，并在检索时依据召回情境进行有效优先级排序，解决“情境崩塌”问题。

**🔧 技术方法**

采用记忆锚定、召回条件诱导、情境感知检索与结构化情境保留生成等四阶段流程，并结合密集语义检索、词典检索和LLM规划查询。

**📊 数据集**

主要在LoCoMo与LongMemEval两大长程记忆基准上进行评测，使用 GPT-4o、GPT-4.1-mini、Qwen3-8B、Qwen2.5-3B 等不同后端模型。

**📈 对比分析**

与SimpleMem等强基线相比，RaMem 在所有四个模型上平均 F1 提升约 12%–14%，在检索诊断中 Recall@10、MRR 等指标均有显著提升，且在上下文预算上更高效。

**⚠️ 局限性**

限制方面：对时间/会话情境的依赖使得在情境信息不完整或错误时性能下降，且系统对上下文绑定的正确性高度敏感，导致在情境不完整或被混淆时仍可能产生错误证据。

---

## 467. IndicGuard: A Multilingual Safety Guard Model and Dataset for Indic Languages

**arXiv ID:** 2606.22841 | [PDF](https://arxiv.org/pdf/2606.22841v1)

**作者:** Parth Bramhecha `[一作]` (L3Cube-Labs), Raviraj Joshi `[通讯]` (L3Cube-Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了面向十种主要印地语系语言的安全数据集IndicGuard，并在Gemma‑3‑4B‑IT上用LoRA微调得到多语安全守门模型IndicGuard，能够在实时对话中检测并拒绝多类别安全违规；

**💡 创新点**

创新点在于：① 用文化适应、冒破和通用三大安全领域构成多样化的数据集；② 通过层级增量训练（Generic → +CA → +JB）验证不同安全数据对模型的边际贡献；③ 在十种训练语言之外的六种低资源印地语系语言实现零样本跨语言迁移；④ 通过XSTest评测得到0.00%过度拒绝率，证明模型阈值已良好校准；

**🔧 技术方法**

技术上采用Gemma‑3‑4B‑IT作为基座，使用4‑bit NF4量化+LoRA（r=16,α=32）进行参数高效微调；训练数据通过Google Translate API实现多语翻译，并在推理时使用greedy解码、正则提取JSON；评估使用scikit‑learn F1、精确率、召回率，跨语言和零样本评估；

**📊 数据集**

使用的主要数据集为IndicGuard，包含三类（Culture‑Adaptive、Jailbreaking、Generic Unsafe），约33k条样本/语种，涵盖Hindi、Bengali、Gujarati、Marathi、Punjabi、Tamil、Telugu、Kannada、Malayalam、Urdu；此外参考Nemotron‑Safety‑Guard‑Dataset‑v3进行源数据挑选；

**📈 对比分析**

与现有基线CultureGuard比较，IndicGuard在所有11种语言、所有评估设置下均提高了约+0.05宏F1，平均用户安全F1 0.8800、响应安全F1 0.8846；在XSTest上达到0%过度拒绝率、58.57%对危险提示的拒绝准确率；在六种未见语言的零样本测试中宏F1仅下降4–8%，显示良好泛化；

**⚠️ 局限性**

局限性包括：① 对快速变化的网络俚语与新兴文化风险响应不足；② 主要覆盖十种主流语言，极低资源方言与少数语言仍缺失；③ 可能存在标注者偏差或模型过度删减导致政治讽刺、学术讨论被误判；④ 依赖Google Translate可能引入翻译误差；

---

## 468. FPAS: Frontier-Based Path Planning with Adaptive Sampling for Large-Scale Unknown Environments

**arXiv ID:** 2606.22838 | [PDF](https://arxiv.org/pdf/2606.22838v1)

**作者:** Jinwoo Choi `[一作]` (Seoul National University), Seung-Woo Seo `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出一种基于前沿的路径规划框架FPAS，能够在大型未知环境中高效到达目标。

**💡 创新点**

创新点包括将探索前沿重新解释为目标导向的子目标、两阶段（反应式与重规划）规划流程，以及基于前沿“开放度”自适应采样的全局图稀疏化方法。

**🔧 技术方法**

采用动态扩展RRT、OctoMap、Dijkstra算法、Sigmoid加权信息增益等技术，并结合前沿检测与自适应采样控制节点密度。

**📊 数据集**

使用Autonomous Exploration Development Environment中的Forest、Tunnel和Garage三种场景进行仿真测试。

**📈 对比分析**

与DSVP-RRT和FAR Planner比较，FPAS在三种环境下均保持或优于距离表现，同时显著降低每步平均计算时间（如森林场景仅3.14 ms，FAR为73.23 ms）。

**⚠️ 局限性**

局限性包括路径最优性略逊于最短路径、在完全3D环境或具有限制运动学的机器人上表现有限，以及自适应稀疏化可能导致信息损失。

---

## 469. CLIP-guided Diffusion Model for Backdoor Generation in Sensor-based Human Activity Recognition

**arXiv ID:** 2606.22837 | [PDF](https://arxiv.org/pdf/2606.22837v1)

**作者:** Toby Briston `[一作]`, Kuniyih S `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了IMU-DM-CLIP，一种用于在运动传感器数据生成中植入后门的CLIP引导扩散模型；

**💡 创新点**

创新点在于将自然语言提示与扩散模型结合，利用CLIP对IMU数据的属性进行引导，从而实现低污染率（10%）的后门注入，并在少样本学习场景下保持攻击成功率；

**🔧 技术方法**

采用CLIP引导的扩散模型（DiffusionCLIP）和GPT‑4.0作为LLM驱动的文本编码，结合IMU‑CLIP的运动传感器编码器；

**📊 数据集**

使用了Skoda、Hand‑gesture和Opportunity三组人体活动识别数据集；

**📈 对比分析**

通过比较基准模型与后门模型的F1‑score及攻击成功率（ASR），发现后门模型在保持约80%+ ASR的同时，整体F1‑score仅略有下降（约0.02-0.07）；

**⚠️ 局限性**

局限性包括对较大规模数据集的验证不足、后门检测的鲁棒性未知，以及扩散模型训练成本高昂。

---

## 470. Learning-Augmented Algorithms for Online Vertex Cover

**arXiv ID:** 2606.22831 | [PDF](https://arxiv.org/pdf/2606.22831v1)

**作者:** Tianhang Lu `[一作]`, Shengcai Liu `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了学习增强在线加权顶点覆盖问题，分别在二分图和一般图模型下设计了满足鲁棒性与一致性双重约束的算法；

**💡 创新点**

创新之处在于将滑雪租赁问题的鲁棒-一致性权衡推广到在线顶点覆盖，给出了随机化（对二分图）和确定性（对一般图）算法的最优鲁棒/一致性界限，并提出了水填充阈值与动态阈值的新技术；

**🔧 技术方法**

核心技术包括水填充（water‑filling）与随机阈值化、原始‑对偶框架、动态负载阈值更新、随机化取样阈值、以及针对建议误差的精细充电/分配分析；

**📊 数据集**

实验使用了合成的 Erdős‑Rényi 图（n=1000，p∈{0.1,0.2,0.5}）和真实的互联网路由器网络 Oregon‑1（约1万顶点、2.2万条边），在二分图与一般图两种模型下分别构造；

**📈 对比分析**

与 Blind‑Following、原始‑对偶、PDLA、GreedyAllocation 等基线算法在 LACR（对数平均竞争比）上进行对比。实验表明在误差率较低时新算法的 LACR 明显优于基线，误差率增大时性能相当；λ 的不同取值可调节鲁棒/一致性平衡；

**⚠️ 局限性**

局限性包括：预测仅为局部二进制建议，未考虑更丰富的预测形式；参数 λ 需要人工设定，缺乏自适应机制；仅针对线性目标，未扩展到子模或非线性目标；实验规模相对有限，尚未验证极大规模真实网络的表现。

---

## 471. What You See Is Not What You Execute: Memory-Based Runtime SBOM Generation for Supply Chain Security

**arXiv ID:** 2606.22827 | [PDF](https://arxiv.org/pdf/2606.22827v1)

**作者:** Hala Alia `[一作]`, Irfan Ahmed `[通讯]` (Virginia Commonwealth University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了 MEM‑SBOM，一个基于内存取证的框架，用来直接从运行中的 Python 进程内存中生成软件组件清单（SBOM）及其依赖关系。

**💡 创新点**

其创新点在于：① 采用多层次的内存扫描（从解释器注册表到垃圾回收器、arena 及堆区）克服动态导入和模块逃逸；② 对内存中模块的字节码进行深度分析，构建精准的依赖图；③ 在 SBOM 基础上实现函数级漏洞可达性分析，显著降低包级漏洞误报。

**🔧 技术方法**

技术手段包括：Volatility 3 插件实现、Python 解释器内部结构（如 sys.modules、thread states、GC 链表）的遍历、字节码反汇编与正则匹配、版本属性解析、PIP 与 PyPI API 查询、CycloneDX SBOM 序列化、Grype 漏洞扫描以及后向可达性分析。

**📊 数据集**

评估使用了 51 个真实开源 Python 应用（覆盖 23 类），全部从 Awesome‑Python 仓库收集，并在 Ubuntu 20.04 虚拟机中分别在虚拟环境与系统级别安装后采样。

**📈 对比分析**

与 8 款主流 SBOM 工具（Syft、Trivy、CDXGen、CycloneDX‑Python、SBOM4Python、ORT、Jake、PIP‑SBOM）对比，MEM‑SBOM 在模块提取率、版本准确率、依赖图完整度和漏洞检测精度上均达到 100%（或接近 99%），而其他工具普遍存在缺失动态依赖、版本偏差和大量误报。

**⚠️ 局限性**

局限性包括：仅支持 Python 运行时，不覆盖本地 C/C++ 扩展；只分析采样时已驻留内存的组件，无法捕获安装阶段或攻击后被删除的模块；依赖现有 CVE 数据库，无法发现未知漏洞；并假设 Python 解释器自身未被篡改。

---

## 472. PanoVine: Whole-Body Visuomotor Control for Soft Growing Vine Robot

**arXiv ID:** 2606.22923 | [PDF](https://arxiv.org/pdf/2606.22923v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 473. BranchShine: Compact Raw-Audio-to-IPA Transcription with a RoPE E-Branchformer Encoder

**arXiv ID:** 2606.22824 | [PDF](https://arxiv.org/pdf/2606.22824v1)

**作者:** Nikhil Navas `[一作]` (Western Sydney University), Saeed Afshar `[通讯]` (Western Sydney University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种小型原始音频到IPA的CTC识别模型BranchShine，专注于多语言IPA转录。

**💡 创新点**

创新在于将轻量级卷积前端与19层RoPE E-Branchformer编码器结合，并在同一数据集和规范下实现与大型模型相近的IPA字符错误率。

**🔧 技术方法**

技术主要包括轻量级卷积前端、RoPE E-Branchformer编码器、CTC训练、Unicode归一化和旋转位置嵌入。

**📊 数据集**

使用IPApack++多语言语料，包含16,660句、41种语言的测试集和约1.63M句训练集；还利用儿童语音阅读基准。

**📈 对比分析**

对比方法为与ZIPA-CTC-NS、ZIPA-CTC、PhoneticXEUS等基线在相同归一化和评估指标下进行，BranchShine取得9.19% IPA‑CER，参数仅33.4M，略低于PhoneticXEUS的9.78%但显著更小。

**⚠️ 局限性**

局限包括仍低于最强基线ZIPA，主观指标仅靠字符错误率，缺乏对不同语言/持续时间的稳健性分析，并未评估实际部署中的实时性能。

---

## 474. EchoFlow: A Workload-Aware Parameter Tuning Method for Blockchain Systems

**arXiv ID:** 2606.22934 | [PDF](https://arxiv.org/pdf/2606.22934v1)

**作者:** Ben Lian `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Yi Sun `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了 EchoFlow，一种能够根据不同工作负载动态调整区块链参数的自动调优框架，提升系统吞吐率与稳定性。

**💡 创新点**

创新点在于将工作负载特征向量纳入状态空间实现真正的动态调优，并结合分布式 D4PG 强化学习与遗传算法加速训练，解决传统 DRL 在区块链样本采样成本高的问题。

**🔧 技术方法**

采用分布式 Deterministic Distributional Policy Gradient（D4PG）强化学习、遗传算法（GA）、多 Actor 并行采样与 Replay Buffer 等技术实现参数调优。

**📊 数据集**

实验使用 Hyperledger Fabric v2.5.14 的 SmallBank 基准合约，在多种读写比例与访问热点的工作负载（RH、WH、WH‑C、RH‑V）上进行评估。

**📈 对比分析**

与默认配置、Athena、DDPG、D4PG‑Tune、GA‑Tune 等基线相比，EchoFlow 在四种工作负载下平均吞吐率提升 7%–20%，训练时间缩短至 1.25 小时即可达到 857 tps，整体性能显著优于现有方法。

**⚠️ 局限性**

局限性包括对不同区块链实现与更激烈突发工作负载的泛化性验证不足；遗传算法可能在后期陷入局部最优；对安全性与系统稳定性影响的深度评估尚未展开。

---

## 475. HADES: Privacy-Preserving Federated Learning via Selective Feature Encryption and Hybrid Model Fusion

**arXiv ID:** 2606.22928 | [PDF](https://arxiv.org/pdf/2606.22928v1)

**作者:** Ergün Batuhan Kaynak `[一作]` (Bilkent University), Sinem Sav `[通讯]` (Bilkent University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种 HADES 框架，在联邦学习中仅对最隐私敏感的特征进行多方同态加密，剩余特征保持明文，分别训练加密子网络和明文子网络，并在融合层进行模型融合。

**💡 创新点**

创新点在于：① 通过 PCA 仅选择少量特征进行加密，从而显著降低加密开销；② 设计了混合加密-明文的融合机制，使得加密子网络无需解密即可完成训练；③ 引入高效的密文打包与冗余旋转消除方案，进一步提升运行时性能。

**🔧 技术方法**

使用了多方同态加密（CKKS）进行加密计算，PCA 进行特征选择，激活函数用低阶多项式逼近，score‑level 融合策略，FedAvg 联邦聚合，iDLG 梯度重构攻击用于评估隐私安全性。

**📊 数据集**

在三个公开数据集上实验：Breast Cancer Wisconsin、MNIST、SVHN。

**📈 对比分析**

与仅使用 PCA 选取特征的单网络基线以及标准联邦学习进行对比。HADES 在保持或略高的预测准确率（比基线高 1–5%）的同时，显著抑制了 iDLG 重构成功率，且训练时间随加密特征数线性下降，整体性能优于全加密方案。

**⚠️ 局限性**

局限性包括：① 依赖 PCA 作为特征选择，可能不适用于所有任务；② 仅在全连接网络上验证，卷积等结构的适配仍待研究；③ 仍需多方协作且仅在半诚实模型下安全；④ 对模型推理阶段的攻击未作评估，未来可扩展至更复杂网络和更强攻击模型。

---

## 476. Can Single-View Mesh Reconstruction Generalize to Robot Camera Rotation?

**arXiv ID:** 2606.22987 | [PDF](https://arxiv.org/pdf/2606.22987v1)

**作者:** Yu Zhan `[一作]` (Southern University of Science and Technology), Hong Zhang `[通讯]` (Southern University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

研究单目视网膜网格重建在机器人摄像机旋转（roll、pitch、yaw）下的泛化能力，提出旋转可控评估协议并追踪误差链；提出基于重力的轻量化姿态校正方法（GAR）。

**💡 创新点**

创新点包括①基于纯旋转光学中心的可控旋转评估协议，②将MDE、网格、布局、物理可行性四层误差系统化追踪；③引入重力先验实现姿态稳定化的GAR；④对一阶段与两阶段SAM3D+FoundationPose进行对比，揭示稳健性多维度。

**🔧 技术方法**

使用MDE模型（MoGe、VGGT、DA3-rel、UniDepthV2、Metric3D）、SAM3D、FoundationPose、外部单视网格重建基线，利用自定义的重力约束和Chamfer/ICP等几何评估指标。

**📊 数据集**

主要数据集为Aria Digital Twin（ADT）高视角图像；另外在真实Frankia手腕摄像机序列上进行验证。

**📈 对比分析**

通过旋转协议对比不同方法的旋转误差、平移漂移、尺度漂移和碰撞率。结果显示两阶段SAM3D+FoundationPose在旋转稳健性上优于一阶段SAM3D；GAR显著降低一阶段布局的ICP角误差（≈47%），但对两阶段反而不利；外部基线在不同维度表现各异，说明单一指标无法全面评估稳健性。

**⚠️ 局限性**

局限性：仅考虑单轴旋转（roll/pitch/yaw）且在固定摄像机位置下；未覆盖连续动态旋转、全局运动或不同场景；GAR仅校正旋转，未对平移/尺度做完整优化；实验仅在代表性模型和ADT/Frankia数据上，缺乏更广泛的机器人场景验证。

---

## 477. LiveServe: Interaction-Aware Serving for Real-Time Omni-Modal LLMs

**arXiv ID:** 2606.22983 | [PDF](https://arxiv.org/pdf/2606.22983v1)

**作者:** Xiangyu Zhi `[一作]` (Chinese University of Hong Kong), Xiao Yan `[通讯]` (Wuhan University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出LiveServe，一个交互感知的Omni-LM服务系统，能够根据实时播放进度、语音活动和中断事件动态调度推理和缓存，减少过度生成并提升用户体验；

**💡 创新点**

创新点在于将实时交互信号嵌入多阶段Omni-LM调度与KV缓存管理，形成按紧急度分级的调度器（U0/U1/U2）以及基于下一次使用预测的KV淘汰与预加载机制；

**🔧 技术方法**

实现依托vLLM‑Omni的分阶段架构，新增交互监控层、交互感知调度器、层级KV管理器；调度算法结合播放缓冲、首次音频触发和KV压力评估；预加载利用语音开始/中断事件触发DRAM→HBM异步传输；

**📊 数据集**

实验使用三类数据集：ShareGPT中文-英文对话（单轮与多轮）、保留的交互轨迹（多轮对话），以及StreamingBench（视频+文本）混合工作负载；

**📈 对比分析**

与vLLM‑Omni基础（开启/关闭KV缓存）对比，LiveServe在多种模型（Qwen3‑Omni、Ming‑Flash‑Omni 2.0）和负载（Poisson、BurstGPT、barge‑in）下均优越：P90音频TTFP平均降低2.21×，完成请求吞吐量提升1.56×，在barge‑in场景下生成但未播放的token浪费降低72‑78%；

**⚠️ 局限性**

局限性包括：需要客户端提供播放进度与VAD信号，依赖vLLM‑Omni原有的分阶段设计，实验仅在H200 GPU集群上验证，未探讨极高并发下的进一步压缩，且对非语音模态的适配与评估有限。

---

## 478. DT-GOL: Dual-Track Geometric Online Learning in Nonstationary Environment with Label Delay

**arXiv ID:** 2606.22950 | [PDF](https://arxiv.org/pdf/2606.22950v1)

**作者:** Yulin Wang `[一作]` (Southwest University), Di Wu `[通讯]` (Southwest University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种双轨几何在线学习框架 DT‑GOL，用以解决非平稳环境下的标签延迟问题

**💡 创新点**

创新点包括：①通过高斯Copula映射异构缺失数据到统一几何空间；②构建动态几何图并利用多视角自监督生成软伪标签；③采用分离的主干与短期分支双轨学习，配合风险感知的动态集成，兼顾稳定性与可塑性；④在延迟窗口内实现前瞻性监督，显著缓解盲适应区负迁移

**🔧 技术方法**

技术主要涉及在线高斯Copula嵌入、在线相关性学习与EM填充、基于密度梯度的几何图构建、软伪标签自训练（多视角融合）、进阶的双轨学习与风险感知动态加权集成

**📊 数据集**

实验使用10个真实世界数据集（如 Australian、WDBC、WBC 等）和4个合成数据集（Agrawal、SEA 等）进行评估

**📈 对比分析**

与五个基线（FOBOS、OVFM、OSLMF、MDISF、IWMS、LACH）对比，DT‑GOL 在误差率（CER）和 AUC 上平均提升约 4–6%，在盲适应区的最大性能下降也最小化，整体表现位居前列

**⚠️ 局限性**

局限性包括：对高维大规模流仍有计算负担；软伪标签生成依赖图结构的稳定性，可能受噪声影响；对极端标签延迟（L 过大）时性能仍需进一步验证

---

## 479. Provable Benefits of RLVR over SFT for Reasoning Models: Learning to Backtrack Efficiently

**arXiv ID:** 2606.22938 | [PDF](https://arxiv.org/pdf/2606.22938v1)

**作者:** Stanley Wei `[一作]` (Princeton University), Juno Kim `[通讯]` (University of California, Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过在一个多分支多层钻石图上的路径搜索模型，理论比较了监督微调（SFT）和可验证奖励强化学习（RLVR）在推理时后向搜索（backtracking）能力上的差异，并证明了RLVR在推理时间计算上相较于SFT具有指数级的优势。

**💡 创新点**

创新点在于：①将链式思维（CoT）建模为图搜索问题；②给出了SFT在仅学习金手指路径时无法获得后向搜索策略的正式证明；③证明RLVR在仅凭终点奖励（加上长度惩罚）即可学习高效后向搜索，导致推理时间从指数级降低到线性级；④提出将RLVR产生的推理轨迹进行监督蒸馏，以在基模型中复制后向搜索能力。

**🔧 技术方法**

使用了：
- 纯大二元（bigram）软最大模型来近似LSTM/Transformer；
- 正式的梯度流与符号梯度流分析；
- 对图结构（分支数W、钻石层数K、内部多边边数L）进行符号化递归分析。

**📊 数据集**

数据集：完全合成的图结构（W=15、K=15、L=5 的例子），无真实语言数据；只使用模型本身生成的金手指路径或自游走轨迹。

**📈 对比分析**

比较方法：对已收敛的SFT和RLVR模型在相同图上进行推理时间（即到达目标所需步数）评估。理论结果显示：RLVR平均需要 Θ(W·K) 步，SFT需要 Θ(W·L^K) 步，实验中在上述参数设置下 RLVR 取得约 4·W·K 的命中时间，远优于 SFT 的指数级增长。

**⚠️ 局限性**

局限性：
- 只在极简的多分支图上证明，未验证在真实大语言模型或复杂推理任务上的可迁移性；
- 仅考虑大二元模型，缺乏对更复杂注意力网络的分析；
- RLVR 的训练需要对抗性探索与奖励信号，实际算力和样本效率可能受限；
- 结果主要聚焦于后向搜索能力，未覆盖其它推理技巧（如归纳、递归等）。

---

## 480. Hierarchical Reinforcement Learning for Sparse-Reward Search in Commutative Algebra

**arXiv ID:** 2606.22922 | [PDF](https://arxiv.org/pdf/2606.22922v1)

**作者:** Giorgi Butbaia `[一作]` (California Institute of Technology), Sergei Gukov `[通讯]` (California Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

使用层次强化学习（HRL）框架解决稀疏奖励的代数 Hirsch 猜想中的非 Hirsch 理想搜索问题。

**💡 创新点**

创新点包括：① 引入链式受限选项（Chained Constrained Options）以构造“脊（spine）”和“线性化（linearization）”的时间抽象；② 设计等变图神经网络（Syzygy‑aware message passing）以捕捉理想的代数结构；③ 利用脊的概率分布和启发式奖励塑造提升搜索效率。

**🔧 技术方法**

技术手段：层次强化学习（受限选项）、等变图神经网络（SConv）、硬约束策略掩码、奖励塑造（Heuristic‑Guided RL、HER、HuRL）、经验缓冲、基于脊的起始分布和 A* 线性化策略。

**📊 数据集**

数据集：自行生成的线性单调 monomial ideal 图（不同度数 d=4~7 的生成空间）；实验在多种度数下使用自定义的理想生成器和图结构进行训练与评估。

**📈 对比分析**

方法对比与性能：与经典搜索（贪心搜索、最优搜索）和标准 RL（PPO、SAC、SAC+HER、SAC+HuRL、SAC+Buffer）进行对比。HRL 在 d≥4 时成功率提升至约 0.1–0.03（相比 10⁻⁶ 的 RL 或 10⁻³ 的搜索），且在更高度数下仍保持可行，而传统方法几乎无解。

**⚠️ 局限性**

局限性：① 仅能利用已知的“支持脊”，其出现频率低于 1%；② 对大度数的收敛仍受限，需更精细的奖励与约束；③ 需要手工设计约束与启发式，难以直接推广到其他数学问题；④ 仅在稀疏奖励环境下验证，缺乏更广泛的应用测试。

---

## 481. GRAIN: Group Aggregation via Min-Norm Objective

**arXiv ID:** 2606.22917 | [PDF](https://arxiv.org/pdf/2606.22917v1)

**作者:** Nghia Bui `[一作]` (New Jersey Institute of Technology), Lijing Wang `[通讯]` (New Jersey Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

GRAIN提出了一种在mini-batch训练中用最小范数凸组合代替传统均值聚合的轻量级算法，能够消除训练过程中的种子诱发波动；

**💡 创新点**

其创新点在于：①通过最小范数求解保证所有组梯度非冲突；②实现了比SGD更紧的均匀稳定性上界；③在保持O(1/T)收敛率的同时，兼容任意一阶优化器；

**🔧 技术方法**

主要技术包括：最小范数凸组合的QP求解（可用Frank‑Wolfe快速求解）；对组内与组间梯度的双层聚合；在多GPU分布式下的梯度聚合与权重广播；

**📊 数据集**

实验使用的公开数据集包括：生成任务（GSM8K、PubMedQA），序列分类（BoolQ、RTE、MRPC），图像分类（CIFAR‑100长尾/步长不平衡、噪声标签），回归任务（STS‑B、Diabetes），以及LLM微调任务（Qwen、Mistral 等大模型）；

**📈 对比分析**

与FFT/LoRA、FocalLoss、NoisyTune、PCGrad、SWA、SAM 等基线对比，GRAIN在所有任务上均无崩溃，平均性能提升约1–2个百分点，且跑间方差显著下降；

**⚠️ 局限性**

局限性包括：仅针对梯度冲突导致的种子波动；理论对ReLU等非光滑网络的适用性有限；最小范数QP随组数增大成本上升；实验仅覆盖有限模型与组数，未验证极大组数或其他新架构的效果。

---

## 482. Cross-lingual Retrieval-Augmented Classification for Dysarthria Severity Assessment

**arXiv ID:** 2606.22910 | [PDF](https://arxiv.org/pdf/2606.22910v1)

**作者:** Taeyoung Jeong `[一作]` (Sogang University), Myoung-Wan Koo `[通讯]` (Sogang University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了Cross‑lingual Retrieval‑Augmented Classification (CRAC) 框架，利用对比学习构建severity聚焦的嵌入空间，并通过跨语言检索与融合实现发音障碍严重程度的自动评估。

**💡 创新点**

创新点在于将临床评估中比较案例的思路转化为跨语言检索+对齐+融合流程，既克服了跨语言差异，又提升了低资源条件下的性能。

**🔧 技术方法**

使用 Whisper 预训练模型提取特征，采用监督对比学习（SupCon）学习嵌入，FAISS 进行检索，跨注意力融合以及 MLP 分类器。

**📊 数据集**

使用韩国卒中发音障碍数据集和意大利 ALS 发音障碍数据集，六种 MPT/ DDK 任务，采用 speaker‑independent 三分类设置。

**📈 对比分析**

与单语种基线（Baseline 1）和简单多语种混合基线（Baseline 2）相比，CRAC 在韩国数据上达 87.3% 平衡准确率（比基线提升 8.4pp），在意大利数据上达 86.7%（比基线提升 20.0pp），整体性能明显优于基线。

**⚠️ 局限性**

局限性包括仅验证两种语言与两种病因，检索质量与效率未做深入分析，模型对检索数量敏感，对极端严重度类别仍存在一定误差。

---

## 483. Topological Out-of-Domain Generalization in Dynamical Systems Reconstruction

**arXiv ID:** 2606.22969 | [PDF](https://arxiv.org/pdf/2606.22969v1)

**作者:** Georg Trede `[一作]` (Central Institute of Mental Health), Daniel Durstewitz `[通讯]` (Central Institute of Mental Health)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对现有层级式动力学系统重建（DSR）模型在外域推理时的失效，论文从理论上分析其根本原因，并提出稀疏正则化、特征拆分以及离散化误差界限等改进措施，实现了在未见过的分岔点（tipping point）外域进行零样本预测。

**💡 创新点**

创新点包括①证明了传统单一特征映射会导致雅可比矩阵稠密、位置与动力学尺度纠缠，从而在外域产生谱发散；②引入稀疏/低秩正则化剔除不物理的雅可比分量；③设计特征拆分（positional 与 dynamical）实现尺度解耦；④推导离散化误差上界，给出可可靠外域推理的闭式判据。

**🔧 技术方法**

主要技术手段为层级式/元学习 DSR（使用 RNN 与 Neural ODE 结构），对模型参数进行线性特征映射；加入 L1 稀疏和低秩正则化；实现特征拆分；对离散化误差做数学分析并给出阈值；在实验中使用 Wasserstein‑1 距离评估重建精度。

**📊 数据集**

使用多种经典动力学系统作为实验数据集，包括 Lorenz‑63、Lorenz‑96、Selkov、van der Pol、Lotka–Volterra、FitzHugh–Nagumo、SIR 等，分别在固定点、周期或混沌等不同动力学 regime 进行训练与外域测试。

**📈 对比分析**

与未改进的层级式 DSR（单特征、无稀疏/拆分）以及其他对比方法（如 Reservoir Computer、SINDy‑CP）相比，改进模型在训练域内保持相同性能，且在跨分岔点的零样本外域预测中显著降低 Wasserstein‑1 距离，成功捕捉从固定点到周期再到混沌的完整分岔结构，表现出明显的 OOD 泛化优势。

**⚠️ 局限性**

主要局限包括：假设控制参数以线性或单一函数形式出现；稀疏雅可比假设不一定适用于所有系统；离散化误差上界依赖 Δt 的展开，可能在高阶非线性系统中不充分；对复杂真实世界数据的实验验证尚待进一步研究。

---

## 484. The Impact of VAE Design on Latent Pose Representations for Diffusion-based Sign Language Production

**arXiv ID:** 2606.22959 | [PDF](https://arxiv.org/pdf/2606.22959v1)

**作者:** Guilhem Fauré `[一作]` (Université de Lorraine), Slim Ouni `[通讯]` (Université de Lorraine)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文研究了不同变分自编码器（VAE）设计对隐式扩散手语生成模型的影响，设计并评估了四种 VAE 变体，并分析了其潜在空间特性与生成性能之间的关系。

**💡 创新点**

创新点在于系统探讨了 VAE 的架构、损失与潜在空间特性如何共同决定后续隐式扩散生成的质量，发现潜在空间的时间变化和有效维度等属性比传统的重建误差更能解释生成效果。

**🔧 技术方法**

采用了 MLP、GCN+ResTemporalConv 的结构化编码器/解码器、多目标重建损失、区域分布式潜在空间、条件隐式扩散模型、文本嵌入（BERT）以及动态时间规整等技术。

**📊 数据集**

实验使用 RWTH-PHOENIX-Weather 2014T（德语手语）数据集。

**📈 对比分析**

通过几何重建指标（MJE、BOE）、潜在空间指标（S_v、S_a、ρ、d_eff）与生成评估指标（BLEU-1/4）进行对比，发现不同 VAE 造成的 BLEU 变化与潜在空间的时间变化和有效维度相关，但单纯的重建误差并未显著影响生成性能。

**⚠️ 局限性**

局限性包括仅在单一数据集与单一模型上实验，实验结果基于单个随机种子，缺乏统计显著性检验，且未在更大规模或多语言数据集上验证结论。

---

## 485. Understanding Knowledge Distillation in Post-Training: When It Helps and When It Fails

**arXiv ID:** 2606.22942 | [PDF](https://arxiv.org/pdf/2606.22942v1)

**作者:** Xin Liu `[一作]` (University of Michigan), Kaiqiang Song `[通讯]` (Zoom Video Communications)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型后训练阶段的知识蒸馏效果，系统评估不同数据规模下 KD 与 SFT 的性能差异，并提出利用合成数据的两阶段蒸馏策略；

**💡 创新点**

发现 KD 在低数据量下显著优于 SFT，强教师在大数据量下仍能提升模型；首次提出先用合成数据“预热”再进行 KD 的两阶段训练框架；

**🔧 技术方法**

主要使用 GKD（含 on‑policy 样本的通用 KD）、SeqKD 与强化学习回馈（RLHF）强化的教师模型，结合合成数据生成；

**📊 数据集**

以 Tulu‑3 指令‑响应对（939k 条）为主数据集，低资源域采用 Flores‑200、DialogSum、ARC‑Challenge 等；

**📈 对比分析**

与 SFT、SeqKD 等对比，KD 在 10k–80k 级别数据时可提升 3–5% 以上；强教师蒸馏在全数据下提升约 4%；两阶段合成数据蒸馏在小模型上进一步提升 5–10% 级；

**⚠️ 局限性**

研究仅针对 Llama‑3 系列模型，合成数据受教师偏差限制，未评估安全、长文本等维度，且对多架构、多任务的通用性待进一步验证。

---

## 486. Neural Architecture Search of Sample Reweighting Networks for Complex Distribution Shift

**arXiv ID:** 2606.22991 | [PDF](https://arxiv.org/pdf/2606.22991v1)

**作者:** Keisuke Sugawara `[一作]` (Yokohama National University), Shinichi Shirakawa `[通讯]` (Yokohama National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在标签噪声与类别不平衡共同存在的复杂分布偏移环境下，利用神经架构搜索（NAS）优化 Meta-Weight-Net（MW‑Net）的网络结构和输入特征，以提升样本加权效果和分类器性能。

**💡 创新点**

创新点包括：①首次将NAS应用于样本加权网络而非仅针对分类器；②将中间层特征与 one‑hot 标签拼接作为 MW‑Net 输入，增强权重判别能力；③采用 Tree‑structured Parzen Estimator（TPE）同时搜索网络层数、节点数以及输入层位置；④通过实验验证该方法在 CIFAR‑10/100 上对噪声与不平衡的鲁棒性提升。

**🔧 技术方法**

使用的技术与方法：Meta‑Weight‑Net（元学习加权网络）、TPE 搜索算法、ResNet‑32 分类器、global average pooling、one‑hot 编码、SGD/Adam 优化器、标签噪声（均匀/翻转）与类别不平衡的人工数据增强。

**📊 数据集**

数据集：CIFAR‑10 与 CIFAR‑100，在其基础上分别添加 40% 标签噪声（均匀噪声或翻转噪声）以及两种类别不平衡因子 β=20 与 β=100。

**📈 对比分析**

实验对比方法：①基线单层 100 节点的 MW‑Net；②最大结构 5 层 1024 节点；③仅搜索网络结构；④搜索网络结构并同时选择输入特征。评估指标为 Top‑1 准确率。结果显示：在 CIFAR‑10 上，NAS（结构+输入）在大多数噪声/不平衡组合下均显著优于基线，提升约 1–3%（取决于 meta 样本量和噪声类型）；在 CIFAR‑100 上提升有限，最大结构与 NAS 结果差距较小。

**⚠️ 局限性**

局限性：①搜索过程计算开销大，每个候选架构需从头训练；②未使用一‑shot 或权重共享等高效 NAS 技术；③在样本极少的 CIFAR‑100 少量干净样本导致难以区分噪声，NAS 效果不明显；④搜索空间仅包含全连接层，缺乏卷积/池化等更丰富结构，未来可进一步扩展。

---

## 487. Public Diffusion Models, Private Images: Key-Controlled Inversion for Conditional Reconstruction

**arXiv ID:** 2606.22988 | [PDF](https://arxiv.org/pdf/2606.22988v1)

**作者:** Lijunxian Zhang `[一作]` (University of Science and Technology of China), Zikai Xu `[通讯]` (University of Science and Technology of China)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于密钥控制的扩散模型逆向重建框架，能够在公开模型参数的白盒场景下实现对图像的加密与解密。

**💡 创新点**

创新点包括：①利用严格可逆的 O‑BELM 采样器，将密钥相关噪声注入到噪声预测项，实现密钥正解时完全可逆、错误密钥时指数放大的安全属性；②在理论上证明了该方案对 IND‑CPA 攻击具有指数级安全性；③通过误差传播分析，将传统的扩散模型误差放大转化为安全资产。

**🔧 技术方法**

关键技术包括：扩散模型（DDPM/DDIM）、O‑BELM 逆向采样、密钥驱动的伪随机生成器（CSPRNG）注入噪声、误差传播与 KL 散度分析、密钥安全证明与实验评估。

**📊 数据集**

实验使用 ImageNet‑1K、MS‑COCO、CelebA‑HQ 三个公开数据集，并在 Stable Diffusion v1.4/1.5/2.0/2.1 四个版本上进行交叉模型验证。

**📈 对比分析**

相较于后置水印、对抗扰动、隐私保护模型和细调访问控制方法，实验表明：在正确密钥下可获得与原图相近的 PSNR/SSIM/FID，错误或无密钥时图像质量急剧下降；加密-解密总时延仅比标准 O‑BELM 低 5% 以内，显著低于基于同态加密/多方安全计算的方案。

**⚠️ 局限性**

局限性：仅提供 IND‑CPA 级别安全，未实现 CCA 保护；对高噪声或长步骤数的正确密钥解密精度受限；交叉模型鲁棒性在模型差异较大时仍有限；安全性依赖于噪声注入规模与步骤数的经验调参。

---

## 488. Distilling Collaborative Dynamics into Latent Space for Implicit Coordination in Decentralized Multi-Agent Manipulation

**arXiv ID:** 2606.22982 | [PDF](https://arxiv.org/pdf/2606.22982v1)

**作者:** Chanyoung Park `[一作]` (Korea Advanced Institute of Science and Technology), Sung-eui Yoon `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 CLS-DP 的去中心化多智能体扩散策略框架，利用在训练阶段从全局轨迹中提炼的协作潜变量，使每个机器人在只观测到本地 RGB 影像和共享任务指令的条件下实现隐式协作。

**💡 创新点**

创新点在于：①在 CTDE 结构下通过 CVAE 将特权的多智能体动力学压缩为协作潜空间；②在部署时仅利用本地观测即可推断该潜变量，从而在不需要全局视图、显式状态或通信的前提下实现跨智能体的协同；③通过跨模态注意力自适应地将图像和文本信息融合，进一步提升对任务需求的理解。

**🔧 技术方法**

使用的核心技术包括：扩散策略（Diffusion Policy）作为动作生成器；残差 CVAE（Contextualizer）用于潜变量学习；Transformer 及交叉注意力模块实现跨模态融合；SigLIP 和 ViT‑Base 进行视觉编码；Integrated Gradients 进行可解释性分析；以及标准的 DDPM 训练与推理流程。

**📊 数据集**

在 RoboFactory 机器人工厂基准上进行评估，包含六种多臂操作任务，团队规模从两到四个机器人不等，任务涵盖同步、角色不对称、严格序列依赖等多种协作挑战。

**📈 对比分析**

与多种集中式（Global GauDP、LargeDP 等）和分散式基线（Local GauDP、无协作潜变量版本）相比，CLS‑DP 在平均成功率上达到 38%（比最佳集中式基线 20% 高出 18个百分点），在参数效率和团队规模可扩展性上也表现最优；但在对空间精度要求极高的 “Take Photo” 任务中仍略逊于直接利用三维几何信息的基线。

**⚠️ 局限性**

局限性包括：潜变量隐式表示在细粒度空间精度上存在瓶颈，导致在需要高精度定位的任务中表现不如显式几何方法；目前仅在仿真环境验证，缺乏真实世界安全约束和碰撞规避机制；以及对多智能体环境的泛化能力仍待进一步探索。

---

## 489. Joint Air Traffic Flow and Capacity Management via Answer Set Programming

**arXiv ID:** 2606.22978 | [PDF](https://arxiv.org/pdf/2606.22978v1)

**作者:** Alexander Beiser `[一作]` (TU Wien), Stefan Woltran `[通讯]` (TU Wien)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种将航班流量管理（ATFM）与动态空域配置（DAC）联合优化的框架，并给出了基于答案集规划（ASP）的完整编码实现；

**💡 创新点**

创新点在于：①首次将 ATFM 与 DAC 在同一模型中联合建模；②利用 ASP 的可配置性实现多种操作（延迟、改道、重配置）的可切换与消除；③提供开源实例生成器和基准数据；

**🔧 技术方法**

采用的技术包括：答案集规划（ASP）编程、离散时间建模、图论航路与分区、弱约束优化以及与 MIP 与 CASA 规则的对照实验；

**📊 数据集**

数据集来源于公开的 OpenSky Network COVID‑19 监测数据和 OurAirports 数据集，作者自行实现了一个基于 Poisson 过程的实例生成器，并将实验数据发布在 Zenodo；

**📈 对比分析**

通过在 5 种地理图规模（16–117 个航点）和 10–100 航班的 4 个种子实例上，对 27 种 ASP 变体与 CASA 与 MIP 基线进行对比。结果显示：在小规模实例中，部分 ASP 变体（如 (r_p,d,s_p)）在超载、延迟与航段数量上优于 MIP；但在大规模实例上，ASP 面临 grounding 与内存瓶颈，无法在时间预算内完成；CASA 虽然始终实现 0 超载，但延迟极高。

**⚠️ 局限性**

主要限制包括：1）对大规模实例的可扩展性不足，grounding 与内存使用过大；2）缺乏理论复杂度分析；3）未实现对更大问题的混合求解或启发式预处理；4）未提供对结果的可解释性说明。

---

## 490. Decentralized Operations of Decarbonized Chemical Plants with Renewable-driven Transmission Systems

**arXiv ID:** 2606.22973 | [PDF](https://arxiv.org/pdf/2606.22973v1)

**作者:** Richard Reed `[一作]` (Oklahoma State University), Paritosh Ramanan `[通讯]` (Oklahoma State University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个隐私友好的分布式优化框架，通过ADMM联合调度电网的Unit Commitment和乙烷裂解工厂微电网，实现电力与化工系统的协同规划。

**💡 创新点**

在保持工厂运营数据隔离的前提下，首次量化分解带来的最优性与排放代价，并通过辅助系统级罚款加速ADMM收敛。

**🔧 技术方法**

采用交替方向乘子法(ADMM)和两阶段LP–MIP求解策略，结合系统级惩罚项和工厂级协调变量实现去中心化求解。

**📊 数据集**

利用合成的ACTIVSg2000德州输电网络模型，并将26家实际德州乙烷裂解工厂的地理位置与工艺参数映射到最近的输电节点。

**📈 对比分析**

将分布式方案与全局集成的基准在三种负荷与两种电气化比例下进行对比，结果显示最优性差距≤1.8%，排放差异≤0.8%，计算时间约1小时以内。

**⚠️ 局限性**

实验仅基于合成网络和确定性24小时规划，未考虑可再生能源随机性、实时价格信号及真实网格数据，限制了对真实场景的直接验证。

---

## 491. Humanoid-OmniOcc: Stereo-Based Full-View Occupancy Dataset for Embodied AI

**arXiv ID:** 2606.22971 | [PDF](https://arxiv.org/pdf/2606.22971v1)

**作者:** Xianda Guo `[一作]` (Wuhan University), Wei Sui `[通讯]` (D-Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向人形机器人全景立体占据感知的数据集 Humanoid-OmniOcc，并提出了利用立体深度先验的 2D→3D 占据网络 HS²Occ。

**💡 创新点**

创新点包括：①首个专门针对人形机器人的全景立体占据数据集；②采用 Real2Sim2Real 闭环设计，使仿真与真实传感器高度一致；③在占据预测中首次把立体深度成本体卷和深度先验用于 voxel 化，从而显著提升几何完整性与泛化能力。

**🔧 技术方法**

技术手段包括 NVIDIA Isaac Sim 物理渲染与精确相机建模、立体匹配网络（如 FoundationStereo）生成深度后置成本体卷、深度归一化与三维卷积、轻量级 Transformer 进行多视角融合、稀疏卷积与 trilinear upsampling 的占据头，以及联合深度+占据的多任务损失。

**📊 数据集**

使用的数据集是 Humanoid-OmniOcc，包含 15 个仿真室内场景和 5 个真实室内环境，生成超过 155K 个样本，提供厘米级精度的占据标签、语义标签与多视角立体图像；实验还与现有单目占据基准（FB‑Occ、Flash‑Occ、SurroundOcc、GaussianFormer）对照。

**📈 对比分析**

实验方法：在相同的 4 视角立体输入下，将 HS²Occ 与四个 SOTA 单目占据模型进行对比。结果显示，HS²Occ 在测试集上的 IoU 为 29.67、mIoU 为 11.69；在真实世界捕获下 IoU 达到 35.45、mIoU 为 19.26，明显优于单目基准（例如 FB‑Occ 在测试集 IoU 28.59、mIoU 5.11，实景 IoU 15.39、mIoU 5.34），证明了其在几何精度与跨域泛化上的优势。

**⚠️ 局限性**

局限性：①依赖高质量的四视角立体相机，成本与部署复杂度较高；②主要针对静态室内环境，缺乏动态物体或户外长距离场景的验证；③数据生成受光照、材质反射差异影响，可能在极端照明或材质极限下出现误差；④模型训练需要较大算力，实际嵌入式实现尚未深入评估。

---

## 492. Concept Alignment Contrast and Long-Short Prompt Memory for Test-Time Adaptation of SAM3 in Medical Image Segmentation

**arXiv ID:** 2606.22963 | [PDF](https://arxiv.org/pdf/2606.22963v1)

**作者:** Yubo Zhou `[一作]` (University of Electronic Science and Technology of China), Guotai Wang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 9215 | [OpenAlex ID](https://openalex.org/A5029722566)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种针对医学图像分割的 SAM3 测试时自适应框架（CM‑TTA），通过对多模态语义一致性进行评估来改进自适应过程，显著提升分割性能。

**💡 创新点**

创新点包括：①概念对齐对比（CAC）度量利用文本-视觉语义一致性来评估预测质量并挑选最佳增强视图；②长短提示记忆（LSPM）融合短期快速适应与长期稳定提示，实现塑性与稳定性的平衡；③密集监督提示更新（DSPU）利用长期提示产生的增强伪标签实现像素级自监督。

**🔧 技术方法**

技术手段：prompt tuning（仅更新文本提示词嵌入）、多增强视图、概念对齐对比度量、长短提示记忆机制、密集监督损失（Dice、对比损失、熵最小化），以及对 SAM3 的一阶梯度更新。

**📊 数据集**

使用两个医学图像分割数据集：Promise12（前列腺 T2‑weighted MRI）和 ISIC2018（皮肤病变图像）。

**📈 对比分析**

与 TENT、TPT、TTL、ZERO、HisTPT 等现有 TTA 方法比较，CM‑TTA 在 Promise12 上平均 Dice 提升至 73.96%（比最佳方法 72.05% 提升 1.91%），在 ISIC2018 上平均 Dice 提升至 82.34%（比最佳方法 80.10% 提升 2.24%），ASSD 同时显著下降。

**⚠️ 局限性**

局限性：目前仅在 2D 图像上验证；提示词维度受限（仅 1 词），对 3D 医学体素分割尚未扩展；在极端域差异或低对比度图像中仍可能存在误差。

---

## 493. Trajectory-Based Recommender Systems as Control Systems

**arXiv ID:** 2606.22957 | [PDF](https://arxiv.org/pdf/2606.22957v1)

**作者:** Eriam Schaffter `[一作]` (Université Claude Bernard Lyon 1), Elsa Negre `[通讯]` (Université Paris-Dauphine)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于控制理论的轨迹推荐系统框架，利用模型预测控制（MPC）生成长周期推荐序列，并在教育场景中演示其可行性。

**💡 创新点**

创新点在于把用户状态视为动态系统，将推荐视为系统控制输入，首次将轨迹概念与推荐系统结合，并通过控制理论提供统一的数学描述和求解方法。

**🔧 技术方法**

主要技术包括线性动态建模、模型预测控制（MPC）、优化求解、仿真模拟；同时使用传统推荐评估指标（如精确率、召回率、NDCG）进行验证。

**📊 数据集**

未使用真实数据集，而是构造人工合成的学习资源和用户状态进行仿真实验。

**📈 对比分析**

通过仿真对三种资源配置场景（单技能、半聚类、多技能）进行比较，展示收敛速度、目标完成度等指标；未在真实数据或公开基准上进行对比，性能仅在实验图表中体现。

**⚠️ 局限性**

主要局限包括：缺乏真实环境实验验证；对比基准有限，未与现有序列推荐方法做系统对比；模型参数选择较为经验性；只聚焦教育领域，未探讨更广泛的应用场景和复杂用户行为。

---

## 494. Evo-RAD: Navigating Rare Retinal Disease Diagnosis via Self-Evolving Agentic Retrieval

**arXiv ID:** 2606.22955 | [PDF](https://arxiv.org/pdf/2606.22955v1)

**作者:** Wangding Xia `[一作]`, Shujun Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种自演化的检索框架 Evo-RAD，用来改进罕见视网膜疾病诊断。

**💡 创新点**

创新点在于把检索过程视作马尔可夫决策过程，使用图卷积代理通过 DELETE/INSERT/TERMINATE 三种操作动态优化支持集，并采用 Group Relative Policy Optimization 与同质性奖励进行训练。

**🔧 技术方法**

核心技术包括图卷积网络（GCN）、马尔可夫决策过程（MDP）、GRPO 策略优化、同质性奖励机制以及基于预训练的视听语言模型 RetiZero 作为特征提取器。

**📊 数据集**

使用 Retina-31（含 31 例视网膜疾病，其中 20 种罕见）和其子集 Rare-20（20 种罕见疾病）作为评测基准，数据来自 Retina Image Bank（RIB）。

**📈 对比分析**

与多类基线（通用医学 VLM、眼科专用基础模型、PEFT 方案和静态检索）比较，Evo-RAD 在 Rare-20 上微调后平均准确率达 46.28%、宏 F1 分数 40.99%、敏感度 42.43%，相较于最强基线提升约 21% 的准确率和 3.5% 的宏 F1，且在 Retina-31 上同样显著超越所有对照组。

**⚠️ 局限性**

局限性包括需在训练阶段使用标注数据，且对初始候选池质量敏感；在不同机构或数据集上的泛化性尚待进一步验证。

---

## 495. Controllable Texture Tiling with Transformed RoPE-Enhanced Diffusion Models

**arXiv ID:** 2606.22945 | [PDF](https://arxiv.org/pdf/2606.22945v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 496. Plans Don't Persist: Why Context Management Is Load Bearing for LLM Agents

**arXiv ID:** 2606.22953 | [PDF](https://arxiv.org/pdf/2606.22953v1)

**作者:** Aman Mehta `[一作]` (Snowflake AI Research), Anupam Datta `[通讯]` (Snowflake AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过“Replay Pairing”方法，对LLM代理在长时间上下文管理中计划信息的持久性进行定量评估，并提出了严格剥离（Strict Stripping）技术修正推理模型中的测量偏差。

**💡 创新点**

创新点在于：①首次将计划信息的上下文-状态双重存储机制分离出来并量化；②提出严格剥离来纠正推理模型中的思路‑追踪污染；③构建压缩压力测试验证计划信息的脆弱性。

**🔧 技术方法**

技术包括：Replay Pairing（双轨轨迹比较）、余弦距离计划忠诚度测量、线性回归探针（Ridge）进行计划检测、严格剥离对推理块的过滤、压缩策略评估。

**📊 数据集**

数据集主要有：ALFWorld（文本基房屋任务）和HotpotQA（问答），并在Llama-3.1-70B、Llama-3.1-8B、DeepSeek‑R1‑Distill‑Llama‑70B等模型上进行实验。

**📈 对比分析**

实验比较：在标准LLM中计划忠诚度在第+1步峰值约0.45，随后在单一步骤内衰减4–12倍；探针在ALFWorld上R²≈0.875、AUROC≈1；在HotpotQA零样本转移同样表现优异。压缩压力测试显示，简单计划清除导致成功率下降34.7个百分点，且计划保护策略无法恢复性能。

**⚠️ 局限性**

局限性：测量结果受步长索引泄露影响，尚缺乏对计划内容具体化的控制；严格剥离的细节影响未完全消除；实验仅覆盖有限模型和任务，缺乏更广泛的可复制性和因果性验证。

---

## 497. ENVS: Environment-Native Verified Search for Long-Horizon GUI Agents

**arXiv ID:** 2606.22948 | [PDF](https://arxiv.org/pdf/2606.22948v1)

**作者:** Yincheng Zhou `[一作]` (University of Pennsylvania), Shanghang Zhang `[通讯]` (Peking University)

**通讯引用:** 11725 | [OpenAlex ID](https://openalex.org/A5013030532)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ENVS框架，将环境验证搜索与监督学习相结合，以高质量轨迹训练GUI代理。

**💡 创新点**

将搜索与监督分离，通过环境本地验证生成平衡的步骤级监督，并引入OSWorld-Noisy干扰评测。

**🔧 技术方法**

使用树搜索、行为指纹分桶、GPU并行虚拟机、SFT、全局加权平衡、OSWorld环境等技术。

**📊 数据集**

使用300任务的OSWorld数据集（86训练、214保留）和OSWorld-Noisy干扰库。

**📈 对比分析**

与ARPO在线RL基线对比，ENVS在清洁和噪声评测中均达30.3%/29.0% pass@8，显著优于基线且计算成本更低。

**⚠️ 局限性**

对噪声收集的优势有限，干扰收集在干净任务上不提升完成率，且仍需在更大规模任务上验证。

---

## 498. Neural Operator Processes for Probabilistic Operator Learning under Partial Observations

**arXiv ID:** 2606.22946 | [PDF](https://arxiv.org/pdf/2606.22946v1)

**作者:** Jose Miguel Lara-Rangel `[一作]` (University College London), Serge Guillas `[通讯]` (University College London)

**通讯引用:** 1726 | [OpenAlex ID](https://openalex.org/A5002268039)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出 Neural Operator Processes（NOP），一种可从稀疏、非规则观测中学习算子映射的框架，既支持确定性全场预测，也能给出完整的不确定性估计。

**💡 创新点**

创新点在于把 Neural Processes 的集合编码与 Neural Operators 的连续空间解码结合起来，形成可变规模、局部几何自洽的条件算子学习；同时引入全局潜变量与本地注意力，实现对任务级别和局部不确定性的分离。

**🔧 技术方法**

使用技术包括：SetConv 与注意力的两种上下文聚合；全局高斯潜变量与 ELBO 训练；FNO（频域卷积）作为算子解码器；相对 L² 损失、MC 采样以及 heteroscedastic / homoscedastic 输出头。

**📊 数据集**

采用的数据集包括：1D 高斯过程回归；三大 PDE 基准：周期 Burgers 方程、非周期 Darcy 流动、周期 Navier–Stokes，覆盖从 1D 到 2D 的不同几何和动力学特性。

**📈 对比分析**

与传统全网格 FNO 以及多种 NOP 变体（仅 SetConv、加注意力、概率化）进行对比。实验显示：仅用 0.78–25% 的网格点即可与甚至超过全网格基准；在确定性任务上误差低于 1%（Burgers）或 5%（Darcy/NS），概率化模型提供可靠的覆盖率并保持误差不变。

**⚠️ 局限性**

局限性包括：对几何敏感、非周期边界的 2D 任务仍易失去局部结构；单一全局高斯潜变量与对角似然难以表达局部知识的不确定性；跨分辨率迁移受限；对高度混沌或多尺度系统的适用性待验证；概率化训练与推理成本略高。

---

## 499. Evaluating self-supervised echocardiographic representations across downstream extraction strategies for left-ventricular segmentation and ejection fraction estimation

**arXiv ID:** 2606.22943 | [PDF](https://arxiv.org/pdf/2606.22943v1)

**作者:** Sylwia Majchrowska `[一作]` (AstraZeneca), Philip Teare `[通讯]` (AstraZeneca)

**通讯引用:** 272 | [OpenAlex ID](https://openalex.org/A5067992725)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

研究了自监督表示在心脏超声左室分割与射血分数估计中的实际效能，提出了从手工提取到线性探针、轻量解码器、有限微调等多层次评估框架；

**💡 创新点**

创新点在于系统揭示了下游提取策略对自监督表示质量评估的决定性影响，并提出了多策略评估的必要性；

**🔧 技术方法**

使用了自监督学习技术包括DINOv3、BYOS（融合自监督分割和密集一致性目标），以及传统的线性探针、轻量解码器和部分微调；

**📊 数据集**

采用公开的EchoNet‑Dynamic数据集（四腔视图）进行训练与评估；

**📈 对比分析**

与单一探针相比，轻量解码器将Dice从≈0.68提升至≈0.90，EF MAE从≈13%下降至≈9%，而进一步微调仅提升极小；直接EF回归模型在EF MAE上表现最好（≈6.6%）；

**⚠️ 局限性**

主要局限包括仅在单一数据集和视图上验证、未进行外部验证、EF估计采用简化的面积-体积代理、以及未探索更丰富的下游解码/微调策略。

---

## 500. CITADEL: CSI-Based Jamming Detection and Open-Set Classification for IIoT Networks

**arXiv ID:** 2606.22939 | [PDF](https://arxiv.org/pdf/2606.22939v1)

**作者:** Aymen Bouferroum `[一作]` (Inria Lille-Nord Europe), Vincent Lenders `[通讯]` (University of Luxembourg)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了Citadel，双阶段基于CSI的工业物联网无线干扰检测系统，能够实时识别已知干扰、零日攻击并对抗对抗性攻击。

**💡 创新点**

采用微控制器级轻量级二值触发与边缘GPU级多信号OOB集合，融合KL散度、能量分数与马氏距离，实现零日泛化、对抗鲁棒与低成本部署。

**🔧 技术方法**

结合OFDM CSI采集、卷积网络、变分自编码器、扩散概率模型、三信号OOB融合与自校准的K折交叉验证，以及物理可实现的对抗性攻击投影。

**📊 数据集**

使用在实验室测得的Wi‑Fi 802.11n 2.4 GHz CSI窗口数据，包含18种已知干扰、15种零日干扰及多种波形/功率组合，共计约815k窗口。

**📈 对比分析**

与八种基线（MSP、KNN‑OOD、ASH‑S、ViM、JADE、BloodHound+、JamShield、HussainEdge）在检测率、FPR、对抗鲁棒和资源开销上对比，Citadel在已知攻击100%、零日97.1%、E2E FPR 0.4%、白盒对抗不到2%、最强Magaw攻击不到5%，且边缘推理耗时14.2 ms、能耗95.9 mJ。

**⚠️ 局限性**

仅在单一实验室环境、单通道20 MHz、固定硬件和预热的校准下验证，需在真实工厂环境中进行现场部署与自适应校准，且对快速短时微冲击检测能力有限。

---

## 501. Hybrid Compression: Integrating Pruning and Quantization for Optimized Neural Networks

**arXiv ID:** 2606.22935 | [PDF](https://arxiv.org/pdf/2606.22935v1)

**作者:** Minh-Loi Nguyen `[一作]` (University of Science), Trung-Nghia Le `[通讯]` (University of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种多阶段混合压缩方法，先通过裁剪和量化显著减小模型尺寸，再将压缩后的模型集成到 Mixture of Experts（MoE）框架中，以在保持低资源占用的同时提升性能。

**💡 创新点**

创新点在于将传统的裁剪+量化压缩与 MoE 结合：1）使用 AGP 逐步裁剪避免层崩溃；2）采用 QAT 进行精度友好的量化；3）在 MoE 中引入 Top‑k 带噪声门控，解决专家不平衡与专家坍塌问题，从而在压缩后恢复甚至提升原模型的准确率。

**🔧 技术方法**

使用的技术包括：迭代裁剪（Magnitude Pruning + Automatic Gradual Pruning）、量化感知训练（QAT）配合 Fake‑Quant 与 STE、Mixture of Experts（MoE）框架及 Top‑k Noisy Gating 策略；实现基于 PyTorch 和 NNI 自动化训练框架。

**📊 数据集**

实验数据集主要为 CIFAR‑10（60k 张 32×32 彩色图）和 BloodMNIST（17k 张细胞图像，8 类）。

**📈 对比分析**

与原始未压缩模型相比，所有测试模型在 FLOPs、参数量上均实现约 10‑11 倍的压缩，推理速度提升 1.5‑2.5 倍；准确率下降不超过 4.5%，在部分模型上甚至提升 1.8‑3.7%；整合 MoE 后的准确率分别达到 92.1%（CIFAR‑10）和 96.9%（BloodMNIST），与原始模型几乎相同。

**⚠️ 局限性**

局限性包括：1）MoE 增加了模型结构复杂度和门控开销；2）专家坍塌风险仍需谨慎调参；3）仅在图像分类任务上验证，尚未证明对更大规模或其他任务的泛化能力；4）压缩后需额外的微调步骤，增加训练成本。

---

## 502. BEV-Denoise: Learning Intrinsic Noise for Accurate Bird's-Eye-View Semantic Segmentation

**arXiv ID:** 2606.22931 | [PDF](https://arxiv.org/pdf/2606.22931v1)

**作者:** Dooseop Choi `[一作]` (Electronics and Telecommunications Research Institute), Kyoung-Wook Min `[通讯]` (Electronics and Telecommunications Research Institute)

**通讯引用:** 10195 | [OpenAlex ID](https://openalex.org/A5100341479)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于DDPM噪声估计的UNet框架（BEV-Denoise），用于估计并去除鸟瞰视图（BEV）特征中的内在噪声，从而提升BEV语义分割精度

**💡 创新点**

创新点在于：①直接在单前向传播中估计并去除噪声；②采用Task Decomposition（TD）获取噪声标签并联合训练噪声估计网络与解码器；③对共享与类别特定噪声同时去除，并只对具有形状模式的静态类进行去噪

**🔧 技术方法**

使用UNet结构进行噪声估计，结合TD范式、跨注意力、DEformable Cross Attention等技术，以及在BEV特征上实现的噪声减除

**📊 数据集**

在大型真实世界数据集nuScenes上进行实验，使用200×200分辨率的BEV图，包含7类语义标签

**📈 对比分析**

与四大基线模型（LSS、CVT、PETR、BEVFormer）以及现有DiffBEV和TD方法比较，BEV-Denoise在除BEVFormer外的所有模型均实现了显著mIoU提升（最高提升达~3.8%），并保持单帧推理时间<100 ms

**⚠️ 局限性**

局限性：依赖TD获取噪声标签，导致对动态类（车辆、人行道）噪声估计效果差；对BEVFormer的提升有限，且与DiffBEV相比推理速度略逊。

---

## 503. EEG Benchmarking Needs a Task Specification Layer: NeuroDoc for Rulebook-Guided, Executable Benchmark Construction

**arXiv ID:** 2606.22925 | [PDF](https://arxiv.org/pdf/2606.22925v1)

**作者:** Chengxuan Qin `[一作]` (Xi'an Jiaotong-Liverpool University), Jibin Wu `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 1865 | [OpenAlex ID](https://openalex.org/A5018341980)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于规则书的 EEG 任务规范语言和方法，并构建了包含 53 条已审核、可执行的 benchmark 单元（共 245 个任务）的社区评审语料库；

**💡 创新点**

将任务定义、可执行核（kernel）和审核机制统一到同一规则书中，实现任务可执行性、可审计性和可复用性；

**🔧 技术方法**

采用 LLM 辅助的 NeuroDoc 文档与 kernel 生成/升级流程、NeuroAudit 审核界面以及规则书的机器可验证检测；

**📊 数据集**

从公开 EEG 数据集（如 OpenNeuro、PhysioNet 等）抽取 18 个数据集，构成 53 条 benchmark 条目；

**📈 对比分析**

使用四种 EEG 基础模型（BENDR、CBraMod、CodeBrain、LaBraM）在 6 类任务族（MI、P300、睡眠分级、SSVEP/SSAEP、临床 EEG、认知任务）上进行交叉和内部分割评估，宏 F1 指标平均值表明所有模型均可在共享规范下获得超越随机基准的性能；

**⚠️ 局限性**

目前仅对已完成审核的 53 条条目提供可执行评估，剩余约 200 条草稿或未审核条目尚未成为可靠 benchmark 单元，且方法受限于当前规则书的范畴和扩展性。

---

## 504. PromptDyG: Test-Time Prompt Adaptation on Dynamic Graphs

**arXiv ID:** 2606.22914 | [PDF](https://arxiv.org/pdf/2606.22914v1)

**作者:** Guoguo Ai `[一作]` (Nanjing University of Science and Technology), Guansong Pang `[通讯]` (Singapore Management University)

**通讯引用:** 6340 | [OpenAlex ID](https://openalex.org/A5039104219)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出PromptDyG框架，针对离线训练的离散时间动态图学习中存在的结构漂移问题，采用无监督测试时提示适配实现在线更新；

**💡 创新点**

创新点在于首次将轻量化图提示与特征级无标签熵最小化结合，在冻结主干模型的前提下自适应捕捉测试时的结构变化，从而扩大正负样本相似度边界；

**🔧 技术方法**

技术手段包括基于预训练的DTDG模型（如Roland‑GRU）、无监督熵最小化的提示学习、GNN+RNN的时空编码，以及离线/在线评估流程；

**📊 数据集**

实验使用六个公开动态图基准：AS‑733、Reddit‑title、Reddit‑body、UCI、Bitcoin‑OTC 和 Bitcoin‑Alpha；

**📈 对比分析**

与十种DTDG基线及三种TTA方法（Tent、Matcha、GTrans）对比，PromptDyG在所有数据集上均实现平均 MR R 1–3% 的提升，取得最佳平均排名，同时适配时间和显存占用均低于对比方法；

**⚠️ 局限性**

局限性在于仅针对链接预测任务和离散时间动态图提出，未覆盖连续时间动态图或其他下游任务，并且提示学习依赖于固定的预训练主干模型。

---

## 505. When Agents Commit Too Soon: Diagnosing Premature Commitment in LLM Agents

**arXiv ID:** 2606.22936 | [PDF](https://arxiv.org/pdf/2606.22936v1)

**作者:** Aman Mehta `[一作]` `[通讯]` (Snowflake AI Research), Aman Mehta (Snowflake AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型代理在多步推理任务中提前承诺的现象，并提出了“表征承诺”——跨运行隐藏状态的收敛度作为内部过程一致性的早期诊断指标；通过实验验证其能预测轨迹一致性，但与答案正确性无关，并展示了运行时监控和提示干预可降低行为方差；

**💡 创新点**

创新点在于：①定义并量化表征承诺，揭示隐藏状态收敛与轨迹一致性相关但不决定正确性；②提供可执行的运行时监控和提示干预方法来检测并引导模型行为；③证明该诊断在多种模型和任务上具有稳健性，为理解代理内部失效机制提供新视角；

**🔧 技术方法**

技术手段包括：ReAct代理执行多步推理；对同一输入在不同温度下多次运行，提取每步末隐藏状态；计算跨运行的余弦相似性；使用Pearson、partial Pearson、AUROC、bootstrap mediation、TOST等统计检验；训练二分类器基于隐藏相似性监控一致性；设计提示干预以诱导承诺；

**📊 数据集**

实验数据集包括HotpotQA（难易混合）、StrategyQA、MuSiQue；模型涵盖Llama‑3.1‑70B、Qwen‑2.5‑72B、Phi‑3‑14B；在HotpotQA上每题10次跑；

**📈 对比分析**

与行为一致性度量（步骤CV、动作序列多样性）比较，发现step‑4/layer‑40相似性在Llama上 r≈‑0.35，partial r≈‑0.45；在Qwen、Phi‑3上也出现负相关（最高 r≈‑0.65）；运行时监控AUROC最高0.97；提示干预将CV降低约28%，并不影响准确率；与简单基线（问题长度、上下文、观察重叠）相比，该指标显著预测一致性；

**⚠️ 局限性**

局限性包括：仅在三种大模型与两种推理数据集上验证，缺乏对更广泛任务和模型的推广；观察重叠对隐藏相似性的影响尚未完全排除；提示干预同时改变token数量和语义，未能分离具体因果因素；无法区分正确与错误的承诺状态；激活相似性虽具预测力，但机制解释仍不充分；实验仅在单温度设置下进行。

---

## 506. MythraGen: Two-Stage Retrieval Augmented Art Generation Framework

**arXiv ID:** 2606.22924 | [PDF](https://arxiv.org/pdf/2606.22924v1)

**作者:** Quang-Khai Le `[一作]` (University of Science), Trung-Nghia Le `[通讯]` (University of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MythraGen 框架，通过检索艺术数据库与 LoRA 微调实现文本到艺术图像生成

**💡 创新点**

将检索增强与 LoRA 微调结合，支持多种艺术风格与内容混合生成

**🔧 技术方法**

使用 BLIP‑2 进行跨模态检索，FAISS 索引，Stable Diffusion V1.5 + LoRA 微调

**📊 数据集**

利用 WikiArt 数据集（约 80k 张图）并使用 VQA 自动标注缺失的流派标签

**📈 对比分析**

在 CLIP‑T、CLIP‑I、FID 以及人工评估上均优于 Stable Diffusion、BingAI、Midjourney，表现更高的文本/风格一致性和图像质量

**⚠️ 局限性**

对高相似度风格区分仍有限，且检索与微调过程仍需多步骤，未能实时生成多样化艺术作品

---

## 507. Intent-Governed Tool Authorization for AI Agents

**arXiv ID:** 2606.22916 | [PDF](https://arxiv.org/pdf/2606.22916v1)

**作者:** Genliang Zhu `[一作]` (Accentrust), Chu Wang `[通讯]` (Accentrust)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种意图驱动的访问控制层（IGAC），将用户自然语言意图转化为可验证、单调的授权属性，并在 OpenPort 的效应控制基础上对 AI 代理工具调用进行细粒度权限限制。

**💡 创新点**

将用户意图作为可审核、单调的授权属性，引入意图证书、会话级策略缩小、意图感知清单过滤与 payload‑效果一致性检查，解决传统凭证授权不足导致的模型主导过宽工具调用问题。

**🔧 技术方法**

使用意图证书结构、会话级安全策略、意图感知工具清单过滤、payload‑效果一致性检查、OpenPort 的 ABAC/ABAC 风格权限与预检、审计链、确定性原因码等技术手段。

**📊 数据集**

实验采用了 176 任务的合成微基准、从 AgentDojo、ToolSandbox、tau-bench、ToolEmu 等公开基准直接适配的 25 任务，以及 GPT‑OSS‑120B、Llama‑3.3‑70B、Qwen3‑Next‑80B 等模型生成的意图证书与计划器输出。

**📈 对比分析**

通过端点测试、确定性微基准、带模型的端到端 LLM 路径基准等多层评估，IGAC 在不扩展权限、降低非法工具曝光率、保持 0% 误执行的同时，保持约 0.79‑0.80 的 manifest 缩减率，BCR_safe 约 0.25‑0.50，平均延迟在 10–50 秒范围。

**⚠️ 局限性**

证书生成精度不足导致部分可接受权限残留；缺乏持久化证书存储与撤销机制；对复杂工具的效果推理不足；依赖模型输出的可信度；实验仅覆盖合成与小规模任务，未验证真实多租户生产环境。

---

## 508. Intend, Reflect, Refine: An Adaptive Multimodal Reflection Framework for Autonomous Driving

**arXiv ID:** 2606.22913 | [PDF](https://arxiv.org/pdf/2606.22913v1)

**作者:** Zisheng Chen `[一作]` (Sun Yat-sen University), Xiaodan Liang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 23682 | [OpenAlex ID](https://openalex.org/A5047878798)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 IRR-Drive，一个通过先生成文字轨迹意图并预测未来语义鸟瞰图（BEV）来进行多模态反思，再根据场景复杂度自适应选择直接规划或反思后细化的自驾规划框架。

**💡 创新点**

创新点：①将文字推理与BEV预测融合成结构化反思空间；②设计自适应反思奖励机制，让模型动态决定是否进入反思模式；③结合强化学习与多重奖励（PDMS、OBB-FDE、格式奖励）实现闭环优化。

**🔧 技术方法**

采用的技术包括：Vision‑Language‑Action（VLA）模型、语义BEV分词器（VQ‑tokenizer）、多模态链式推理、基于序列的强化学习（GSPO）、自适应反思奖励和层次化奖励设计。

**📊 数据集**

使用的数据集：NAVSIM v1/v2（闭环仿真基准），以及多源驾驶场景预训练数据（DriveLM、NuInstruct、nuScenesQA、CODA‑LM 等），再通过 NAVSIM 生成 BEV 重建/预测数据和自适应反思数据。

**📈 对比分析**

在 NAVSIM v1 上获得 91.3 PDMS，远超传统端到端和近期 VLA 方法；在 NAVSIM v2 上得到 89.0 EPDMS，超过先前最佳 87.1；通过自适应反思，模型实现 91.3 PDMS 的同时，推理时间仅 1.70 s（相比全反思模式的 3.03 s），显示出性能与效率兼顾的优势。

**⚠️ 局限性**

局限性：模型仍未达到实时推理要求，推理速度和资源占用仍高，未来需通过量化、蒸馏等技术提升部署效率。

---

## 509. Subject-Level Unknown-Identity Identification from Leap Motion Controller 2 Hand Landmarks

**arXiv ID:** 2606.22986 | [PDF](https://arxiv.org/pdf/2606.22986v1)

**作者:** Bahar Moharrer `[一作]` (Sapienza University of Rome), Maria De Marsico `[通讯]` (Sapienza University of Rome)

**通讯引用:** 4513 | [OpenAlex ID](https://openalex.org/A5055512639)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了Leap Motion Controller 2手部Landmark数据在受限小样本下的未知身份识别，并构建了基于几何描述符的开放集识别基准。

**💡 创新点**

重新定义ML2HP数据为仅Landmark的身份识别基准，提出嵌套阈值选择的LOSO协议，并用手指尖到掌心距离与掌心归一化角度特征丰富几何表示。

**🔧 技术方法**

采用Extra Trees树集成、嵌入式MLP+质心相似度以及MLP+OpenMax三种模型，并使用距离、角度特征与原始Landmark构建输入。

**📊 数据集**

使用Multi View Leap2 Hand Pose (ML2HP) 数据集，包含21名受试者、17个静态手势。

**📈 对比分析**

在LOSO协议下对三种模型进行比较，Extra Trees闭集精度99.49%、开放集91.29%、AUC 95.71%；Embedding MLP与MLP+OpenMax开放集约74%，AUC约81%，表明树集成在未知拒绝方面显著优于深度基线。

**⚠️ 局限性**

受试者仅21人、仅静态Landmark，缺乏时间序列与多视角信息，且OpenMax与嵌入MLP未能逼近树集成的拒绝性能，说明未知身份拒绝仍是核心挑战。

---

## 510. Understanding Parallel Samplers in Masked Diffusion via Random Walks on Graphs

**arXiv ID:** 2606.22976 | [PDF](https://arxiv.org/pdf/2606.22976v1)

**作者:** Vansh Bansal `[一作]` (University of Texas), Purnamrita Sarkar `[通讯]` (University of Texas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出将图随机游走作为可验证的沙盒，用来研究和评估掩码扩散模型中的并行采样策略；

**💡 创新点**

通过理论证明不同并行采样（如最低熵、随机）在不同图结构下表现不一，并提出基于Markov分离的对数级别并行采样（分割采样）实现精确采样；

**🔧 技术方法**

使用掩码扩散语言模型（MDLM）、图随机游走生成数据、对数并行采样算法以及多种并行采样策略（随机、熵、分割等）；

**📊 数据集**

实验基于合成图数据（生成的树-ER图、瓶颈图等），以及在OpenWebText预训练掩码扩散模型上的真实文本生成；

**📈 对比分析**

与基线（逐位随机、逐位熵、传统分割）相比，分割采样在图随机游走任务上保持高一致性（coherence）且NFE显著降低，在OpenWebText生成任务中实现更优的速度-质量折衷（MAUVE、熵、重复率提升）；

**⚠️ 局限性**

限制在于理论假设为完美条件分布，实际模型误差未充分量化；对分割采样的设计仅针对有限阶Markov结构，无法直接处理语言模型中的长程依赖；

---

## 511. TaLK: Text-attributed Graph Dataset Distillation via Coupling Language Model with Graph-Aware Kernel

**arXiv ID:** 2606.22975 | [PDF](https://arxiv.org/pdf/2606.22975v1)

**作者:** Yeongho Kim `[一作]` (KAIST), Kijung Shin `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了文本属性图（TAG）的数据集蒸馏，提出了一种在联合LM–GNN训练下的高效蒸馏方法。

**💡 创新点**

创新点在于将预训练语言模型与图感知神经切线核相结合，并通过批量梯度注入实现了在不解耦的前提下高效蒸馏文本与图结构信息。

**🔧 技术方法**

使用了预训练BERT + LoRA作为LM、两层GCN作为GNN、图感知神经切线核（SNTK）、kernel ridge 回归以及批量梯度注入等技术。

**📊 数据集**

实验基准包括四个常见TAG数据集：Cora、Arxiv、Photo 和 Computers。

**📈 对比分析**

与直接子集选择、分离蒸馏以及全数据训练对照相比，实验表明在仅占全数据1–5%的合成数据上即可获得97%–99%的全数据性能，显著优于基线。

**⚠️ 局限性**

局限性包括仅在token嵌入层进行蒸馏，缺乏可解释的原始文本输出；仅验证半监督节点分类任务，未覆盖自监督或其他下游任务；在百万级图上的可扩展性和泛化性仍待进一步验证。

---

## 512. MOCAP: Wafer-Scale-Chip-Oriented Memory-Orchestrated Chunked Pipelining Framework for Prefill-Only LLM Inference

**arXiv ID:** 2606.22968 | [PDF](https://arxiv.org/pdf/2606.22968v1)

**作者:** Zichuan Wang `[一作]` (Tsinghua University), Shouyi Yin `[通讯]` (Tsinghua University)

**通讯引用:** 6760 | [OpenAlex ID](https://openalex.org/A5054524841)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了MOCAP框架，针对瓦片级芯片（WSC）进行Prefill‑only LLM推理的内存友好型分块流水线调度；

**💡 创新点**

创新点在于设计了Memory‑Balanced KV Reallocation（MBKR）和Latency‑Balanced Chunk Partitioning（LBCP）两大机制，分别解决分块流水线中的KV缓存不均和计算/通信不平衡问题；

**🔧 技术方法**

采用了瓦片级芯片硬件架构、分块流水线、KV缓存重新分配策略、动态规划与模拟退火优化分块划分、以及基于ASTRA‑sim的事件驱动模拟器；

**📊 数据集**

使用多种大模型（Llama3‑70B、Mistral‑123B、Qwen3‑235B、Llama3‑405B）在长上下文（最多10⁵令牌）场景下的推理任务进行评测；

**📈 对比分析**

与GPipe和Terapipe对比，MOCAP平均降低76.4% E2E延迟、提升3.24×吞吐量，并将可支持的最大序列长度提升至1.31×；

**⚠️ 局限性**

性能提升在序列长度过长时趋于减弱，主要受KV重分配通信开销和分块数量与效率之间权衡的影响，且当前实现仅针对WSC平台。

---

## 513. Attacking the Trusted Imagination: Oracle-Level Integrity Attacks on Imagine-then-Act World Models

**arXiv ID:** 2606.22966 | [PDF](https://arxiv.org/pdf/2606.22966v1)

**作者:** Linghan Chen `[一作]` (Adelaide University), Minyu Guo `[通讯]` (Adelaide University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

研究想象-先行动的世界动作模型（WAM）在受限观测扰动下的安全性，提出白盒完整攻击方法并评估其对 downstream oracle（如 MPC、门控器、验证器）的影响。

**💡 创新点**

首次针对受信任的想象提出完整的白盒攻击框架，揭示想象易被破坏但难以精准控制的非对称性，并提出基于去噪自洽的无参数检测器。

**🔧 技术方法**

使用可微分的 PGD 白盒梯度攻击、离散 VQ 去除、去噪自洽检测、离线实验等技术。

**📊 数据集**

实验基于 LIBERO 长期仿真数据集，以及三种目标模型 RynnVLA-002、LingBot-VA、LaDi-WM。

**📈 对比分析**

通过与随机噪声、正常运行以及 oracle 决策（验证器、MPC 成功率）对比；untargeted 攻击 60× 强于随机，检测 AUC 达 1.0；在 LaDi-WM 的 MPC 中，ε≈0.01 时成功率从 0.55 降至 0.05（p<10⁻⁴）。

**⚠️ 局限性**

仅在仿真中验证，缺乏多任务闭环评估；想象通道只能在 oracle 模型下被操控，难以精准定位目标场景；无法完全区分直接视觉路径与想象路径；对真实环境的验证不足。

---

## 514. LLM-as-a-Judge for Reliable and Explainable Offline Evaluation in Top-K Recommendation

**arXiv ID:** 2606.22961 | [PDF](https://arxiv.org/pdf/2606.22961v1)

**作者:** Yue Que `[一作]` (City University of Hong Kong), Chen Ma `[通讯]` (City University of Hong Kong)

**通讯引用:** 26221 | [OpenAlex ID](https://openalex.org/A5100652421)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于大语言模型的评判者框架（LLM-as-a-Judge），用于离线 Top‑K 推荐系统的可靠且可解释的评估。

**💡 创新点**

创新点：①用 LLM 推理出用户语义偏好作为代理，替代传统的 ID 匹配；②采用先推理后评分的双步流程，并输出单句理由；③将个体评分聚合为标准 Top‑K 指标，并在无偏测试集上验证与传统评测的一致性。

**🔧 技术方法**

技术手段：大语言模型（Qwen3 系列、DeepSeek-V3.2 等）作为评判者；prompt 设计与“先推理后评分”流程；语义匹配与文本推理；统计评估（Pearson、Spearman、Kendall、MAE）与解释性评估（连贯性、真实性、说服力）。

**📊 数据集**

数据集：Kuai‑Rec（视频推荐，含文本信息）和 Coat（服装推荐，含文本信息），均提供无偏测试集。

**📈 对比分析**

比较方法：将 LLM Judge 的评分与传统“normal testing”（MNAR）以及无偏“unbiased testing”对比。实验显示 LLM Judge 在排名一致性（Pearson、Spearman、Kendall）和分数一致性（MAE）上均显著优于 normal testing，相关系数提升 50%+；解释性评估中理由的连贯性、真实性、说服力均超过 80%。对不同 LLM 规模、prompt 变体和输入长度的鲁棒性实验亦表现良好。

**⚠️ 局限性**

局限性：依赖 LLM 的推理与知识库，可能产生幻觉；在极度稀疏（仅 10% 历史）时失效；仅适用于具备文本属性的项目，对缺少文本的项目效果不佳；评估的绝对分数仍受 LLM 输出偏差影响。

---

## 515. PG-MAP: Joint MAP Optimization for Inference-Time Alignment of Diffusion and Flow-Matching Models

**arXiv ID:** 2606.22958 | [PDF](https://arxiv.org/pdf/2606.22958v1)

**作者:** Ruolan Sun `[一作]` (Stony Brook University), Pawel Polak `[通讯]` (Stony Brook University)

**通讯引用:** 239 | [OpenAlex ID](https://openalex.org/A5090692960)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无需训练的推理时对齐框架 PG-MAP，按时间步同时优化文本嵌入 c 与潜变量 z_t；

**💡 创新点**

通过前向一致性耦合、轨迹级 Gibbs-MAP 目标和自适应置信域，实现了联合（c,z_t）更新；

**🔧 技术方法**

使用无训练的 Diffusion/Flow‑Matching 模型、前向一致性残差、近似梯度上升、奖励函数（PickScore 等）以及可调的自适应先验；

**📊 数据集**

在 SD 1.5、SDXL 以及 SD3.5‑medium 的公开模型上，利用 PartiPrompts（1632 句子）和 HPDv2 用户提示；

**📈 对比分析**

与 CFG、Universal Guidance、FlowChef 等基线对比，PG‑MAP 在 PickScore、Aesthetic、HPS 等指标上提升 5–7pp，SD3.5‑medium 上的 UG‑FM 方案更是 91.9% PickScore；

**⚠️ 局限性**

局限在于对文本忠实度影响有限、计算开销相对较大、对不同提示类型的适配需要进一步的路由器设计。

---

## 516. Predicate Importance Estimation and Decoupled Rationale-Score Distillation for Entity Alignment

**arXiv ID:** 2606.22992 | [PDF](https://arxiv.org/pdf/2606.22992v1)

**作者:** Keunha Kim `[一作]` (SungKyunKwan University), Youngjoong Ko `[通讯]` (SungKyunKwan University)

**通讯引用:** 2318 | [OpenAlex ID](https://openalex.org/A5008710152)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了两种模块——Predicate Importance Estimation（PIE）和Decoupled Rationale-Score Distillation（DRSD）用于工业场景下异构知识图谱的实体对齐。

**💡 创新点**

创新点在于通过去除主体、对谓词重要性进行可学习加权来构建紧凑的实体嵌入，并在知识蒸馏中将决策和置信度分离，以实现置信度与决策不一致时的人工复核判定。

**🔧 技术方法**

技术手段包括基于BGE-M3的嵌入编码与两层MLP分类、前缀/后缀提示的LLM教师蒸馏、基于谓词重要性的Prompt压缩，以及在小型语言模型上进行的提示式微调。

**📊 数据集**

使用了公司内部构建的真实工业异构知识图谱，包含约85,858条训练实体对和5,000条评估对，标签为二元相同/不同。

**📈 对比分析**

实验将PIE、DRSD与单纯LLM/SLM基线对比，PIE将准确率提升至0.8608、F1至0.8642；DRSD将准确率提升至0.8742、F1至0.8855，人工复核后F1进一步升至0.9093。

**⚠️ 局限性**

局限性包括谓词重要性权重可能受类别/谓词分布偏倚影响、未对PIE组件进行细粒度消融、人工复核结果假设为完美且未评估真实复核成本。

---

## 517. StatABench: Dataset and Framework for Evaluating Statistical Analysis Capabilities of LLMs

**arXiv ID:** 2606.22977 | [PDF](https://arxiv.org/pdf/2606.22977v1)

**作者:** Youxin Zhu `[一作]` (Southern University of Science and Technology), Guanhua Chen `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 6865 | [OpenAlex ID](https://openalex.org/A5100665987)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 StatABench 这一针对 LLM 统计分析能力的基准，包含 404 道闭式题和 30 道开放式建模任务。

**💡 创新点**

创新点在于将统计学知识与工具调用、代理执行以及 LLM-as-Judge 的动态评估相结合，构建了双轨道评测框架。

**🔧 技术方法**

主要技术包括 LangChain MCP、35 个可调用统计工具的 SAToolKit、以及 Qwen Agent、CrewAI、AutoGen、Smolagents 等高级数据科学代理。

**📊 数据集**

数据集来源于 18 大统计主题的专业教材、现有基准和公开竞赛（MCM/ICM、CUMCM、MAS），共计 404 题和 30 个实战建模案例。

**📈 对比分析**

实验对比显示 GPT‑5.1 在 Stat‑Closed 上仅达 68.6%，最优开源模型 Qwen2.5‑72B 为 60.6%；在 Stat‑Open 上顶级代理平均得分 61.86，反映出当前 LLM 在工具整合和完整建模方面仍有显著差距。

**⚠️ 局限性**

局限性主要在规模有限、持续维护挑战以及潜在的模型污染风险，未来需要扩大任务量并更新维护。

---

## 518. When Preferences Fail to Become Incentives: A Utility-Behavior Gap in Large Language Models

**arXiv ID:** 2606.22974 | [PDF](https://arxiv.org/pdf/2606.22974v1)

**作者:** Yujun Zhou `[一作]`, Christopher M. Ackerman `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计了一套基于现实写作任务（论文、项目摘要、事件后评、中文-英文翻译）和盲判定评估的实验范式，用来检验大语言模型在配对选择中得到的偏好是否能在生成任务中提升输出质量。

**💡 创新点**

创新点在于首次将偏好-行为桥梁检验与可观测的真实写作情境结合，系统区分了内部偏好与外部提示（努力、角色扮演、危害后果）对生成质量的影响，并揭示了“偏好-行为”缺口。

**🔧 技术方法**

使用的技术包括：配对选择的效用诱导（Thurstonian模型）、多模型（七个指令调节LLM）生成多样任务、盲判定投票评估、统计比较（高低效用、直接努力、角色、危害提示）、以及文本特征与LLM编码特征的多维度质量分析。

**📊 数据集**

数据集主要来自自定义的写作任务Prompt集合（四类任务共约8,400条匹配生成对）以及四个后果领域（宗教、动物、国家、政策）的实例，未使用公开标准数据集。

**📈 对比分析**

实验对比方法是通过盲判定的胜率来衡量各提示的效果；直接努力提示使生成质量显著提升（约76.8%胜率），角色提示也显著提升（约61.2%），危害提示导致质量下降（约40.5%），而高低效用提示的胜率与随机无差（约51.2%），未出现显著差异。

**⚠️ 局限性**

局限性包括：仅测试七个指令调节模型，未覆盖最新大型模型；实验仅聚焦写作类生成任务，可能不适用于更广泛的行为场景；未检验更大效用值范围或更复杂交互情境下的偏好激活；对模型内部机制的解释仍不充分。

---

## 519. FORGE: Fused On-Register Gradient Elimination for Memory-Efficient LLM Training

**arXiv ID:** 2606.22932 | [PDF](https://arxiv.org/pdf/2606.22932v1)

**作者:** Dikshant Kukreja `[一作]` (Puch AI), Bapi Chatterjee `[通讯]` (IIIT-Delhi)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在反向传播过程中将梯度计算与参数更新合并，按 128×128 块级别在 GPU 寄存器内完成 AdamW 等逐元素优化器的完整更新，消除梯度张量的存储。

**💡 创新点**

核心创新在于把梯度从全量内存中彻底移除，既不产生梯度张量也不需要单独的优化器内核；通过在每个块内部完成梯度累积、优化计算和权重写回，保证与标准 AdamW 在 fp32 上的数值等价，兼容张量/序列并行。

**🔧 技术方法**

使用 Triton 编写的高效核，按块切分权重和梯度，利用 fp32 寄存器完成梯度累加与 AdamW 计算；支持 bf16/8bit 权重、int8 状态、FP8 权重/状态量化、随机舍入；实现多 GPU 的张量/序列并行。

**📊 数据集**

实验使用 Llama‑3.1‑8B、Qwen3 0.6B–14B、GPT‑2 124M（FineWeb‑Edu）等模型；继续预训练数据为 OpenMathInstruct‑2；Fine‑tune 采用 OpenMathInstruct‑2。

**📈 对比分析**

与 PyTorch 传统 fused AdamW、AdaLomo、GaLore、FlashOptim 等方法对比，单 GPU 上梯度消除后内存峰值下降 53%（35.4 GiB vs 75 GiB），训练步时 1.52×；在 H200 上 Llama‑3.1‑8B 训练时，微批量 4 时内存节省 38–53%，速度提升 1.37–1.69×；在多 GPU 张量并行下，单机 8×A100 上可在同一硬件上实现 4 倍微批量、8,127 tokens/s 的吞吐率。对比方法包括 fused AdamW、AdaLomo、GaLore、FlashOptim、APOLLO 等；性能表现满足或超过基线，且收敛性保持在 0.001 nats 以内。

**⚠️ 局限性**

限制在于仅适用于逐元素可分离的优化器（如 AdamW、SGD、RMSProp 等），不兼容跨元素的预条件器（如 Shampoo、Muon's Newton‑Schulz 等）；对大批量（BS≥8）时梯度占比下降，节省效果受限；在数据/上下文并行模式下，无法在反向时完成梯度-优化融合，需要回收梯度后再同步。

---

## 520. Each Judge Its Own Yardstick: Discovering Per-VLM Taxonomies for Physical Video Evaluation

**arXiv ID:** 2606.22918 | [PDF](https://arxiv.org/pdf/2606.22918v1)

**作者:** Yu Cao `[一作]` (Queen Mary University of London), Jifei Song `[通讯]` (Huawei Darwin Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出JudgeFit管线，针对每个视觉语言模型（VLM）自动生成并优化其物理一致性评估维度；

**💡 创新点**

将评估维度视为可优化对象，利用LLM进行种子生成与诊断引导的迭代细化，实现每个VLM专属的评估体系；

**🔧 技术方法**

采用LLM提示与聚类构建种子分类，使用岭回归校准模型得分与人类评分的关系，诊断维度的可靠性、冗余与覆盖不足，并通过局部编辑（drop、merge、redefine、add、split）在迭代中逐步改进；

**📊 数据集**

使用VideoPhy-2数据集（约300个视频，包含1–5级人类评分和物理规则注解）进行种子生成、迭代细化和测试评估；

**📈 对比分析**

与固定四域PhyGen基线对比，使用Spearman相关系数衡量模型得分与人类评分的吻合度；在所有16个VLM上，精细化后的分类平均提升约32%（从0.239提升到0.315），且每个模型均实现改进；

**⚠️ 局限性**

需要带有物理错误注解的人类标注数据；流程中依赖LLM，存在可复现性和随机性问题，但已在部署后无LLM依赖。

---

## 521. ThermoLLM: Thermodynamics-Aware HVAC Control with Spatial-Semantic Knowledge Graph

**arXiv ID:** 2606.22911 | [PDF](https://arxiv.org/pdf/2606.22911v1)

**作者:** Kirtan Bhatt `[一作]` (UNSW Sydney), Wen Hu `[通讯]` (UNSW Sydney)

**通讯引用:** 9657 | [OpenAlex ID](https://openalex.org/A5042014818)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 ThermoLLM 框架，通过 Brick 语义知识图与最近交互历史作为上下文，利用 LLM 进行多区 HVAC 控制；

**💡 创新点**

将建筑空间结构与热动力学历史信息联合提供给 LLM 进行推理，显著提升舒适度-能耗权衡，并消除了手工编写规则的需求；

**🔧 技术方法**

使用 GPT‑5.4 LLM、Brick 语义知识图、检索增强与滚动历史表格、EnergyPlus + Sinergym 仿真环境；

**📊 数据集**

使用纽约 J.F. Kennedy 天气数据、真实占用数据以及 EnergyPlus 五区办公楼模型，进行三天 1 月评估；

**📈 对比分析**

与 Rule‑Based、MPC、Q‑Learning、PPO、LLM‑heuristic、LLM‑history、DARLIN 等基线比较，评价指标为能耗（kWh）和 PMV 违规率；ThermoLLM 能耗≈271kWh、PMV 违规率≈4.95%，位于 Pareto 前沿，优于所有基线；

**⚠️ 局限性**

仍需依赖高质量 LLM 与准确构建的知识图，对检索与提示工程依赖较大；在更大或更复杂建筑下的可扩展性与实时计算资源需求尚待验证；

---

## 522. Stable Image Reconstruction via Two-Parameter Power-Scale Variation Minimization

**arXiv ID:** 2606.23083 | [PDF](https://arxiv.org/pdf/2606.23083v1)

**作者:** Ziwei Li `[一作]`, Dachun Yang `[通讯]` (Beijing Normal University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了双参数电压变换模型PSV_a,p，并在RIP框架下证明其在梯度和图像域上的稳定恢复以及设计IRLSPSV算法实现无约束最小化

**💡 创新点**

双参数设计提升了模型灵活性；当a→∞时退化为p次幂TV(p)的最小化，首次在RIP框架下给出其最优上界；采用稠密凸组合技巧获得渐进最优RIP阈值；参数敏感性分析揭示a、p作用差异并提供调参方案

**🔧 技术方法**

利用非凸优化的差分凹凸算法（DCA）、梯度互补投影法（IRLS）及原子正交投影（PD）实现PSV_a,p的数值求解；理论证明采用稠密凸组合、RIP性质与稀疏凸组合方法

**📊 数据集**

自然图像使用Set12（Cameraman、Starfish、Bird）、医学成像使用BrainWeb数据库脑部MRI图像、CT使用Shepp–Logan模拟斑马线投影，混合高斯–泊松噪声模拟

**📈 对比分析**

与ZP、TV、L1-αL2、TTV四种常用稀疏正则方法比较；在40%、60%采样率及含高斯/高斯-泊松噪声等多种情形下，PSV_a,p在PSNR、SSIM、GMSD等指标上普遍优于或与现有方法相当，并在计算效率上介于TV与非凸方法之间

**⚠️ 局限性**

模型依赖手动参数调节，缺乏自动化调参机制；对大规模高分辨率图像的收敛性与时间复杂度未完全分析；在极低采样率或极端噪声环境下性能退化的边界尚待进一步研究

---

## 523. Understanding the Stealthy BGP Hijacking Risk in the ROV Era

**arXiv ID:** 2606.23071 | [PDF](https://arxiv.org/pdf/2606.23071v1)

**作者:** Yihao Chen `[一作]` (Tsinghua University), Jianping Wu `[通讯]` (Tsinghua University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文首先对隐蔽BGP劫持（stealthy hijacking）进行正式定义，并通过路由表差异启发式在真实网络中发现并收集了318条高置信度的隐蔽劫持实例；随后提出一个基于多源ROV测量、拓扑压缩与矩阵加速的BGP路由推断框架，能够在5.22小时内生成全网8.3 B条路由，进而利用“受害者‑目标‑劫持者”三元组模型对当前部分ROV部署下的劫持风险进行系统量化，得到整体成功概率为14.1%且针对特定AS对可达99.5%。

**💡 创新点**

创新点主要包括：①首次系统化定义并实测隐蔽BGP劫持；②融合APNIC、RoVista、Cloudflare三源ROV数据并压缩拓扑，结合一字节优先级编码实现矩阵级BGP路由推断，显著提升推断效率；③提出严格启发式与三元组模型，对全网路由进行全面风险评估，揭示ROV部分部署带来的双刃风险。

**🔧 技术方法**

使用技术包括：多源ROV数据聚合、CAIDA AS关系与路由视点数据的整合；网络拓扑压缩与“恶意可达/合法可达”子图构造；基于一字节优先级编码的BGP路由推断算法与矩阵运算；启发式检测隐蔽劫持（宽松/严格两种）；统计与可视化工具（PCC、曲线、CDF）进行风险分析。

**📊 数据集**

采用的数据集有：CAIDA AS关系（77,600 AS + 709,737 边），APNIC、RoVista、Cloudflare ROV测量（共7,275 AS），RouteViews路由表（≈50 M路由/天，3个月）、RPKI、IRR、WHOIS（验证合法性），MaxMind GeoLite2（AS地理信息），以及手工整理的318条隐蔽劫持事件（共2,178条路由）。

**📈 对比分析**

通过与BGPsim（基准BGP模拟器）对比，所提出框架在单GPU或40线程CPU下完成全网路由推断仅需5.22 h，内存≤20 GiB，速度比BGPsim快≈500×；在真实隐蔽劫持数据集上，TPR达到95.9%（incident级别），并在不同ROV阈值下验证鲁棒性，消融实验表明多源ROV集成显著提升检测准确率。

**⚠️ 局限性**

局限性：假设Gao‑Rexford路由策略，忽略了部分复杂或选择性ROV过滤的实际行为；仅对静态路由进行推断，未覆盖动态路由变化；对ROV测量的准确性高度依赖，误差会影响结果；实验聚焦IPv4和RPKI，未覆盖IPv6或未来新协议。

---

## 524. Rethinking Prototype-based Similarity Learning for Few-Shot Object Detection

**arXiv ID:** 2606.23069 | [PDF](https://arxiv.org/pdf/2606.23069v1)

**作者:** KunHo Heo `[一作]` (Kyung Hee University), MyeongAh Cho `[通讯]` (Kyung Hee University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对少样本目标检测，本文提出了两种关键模块——文本锚定语义掩模（TSMa）和分层自回归回归（SHARe），并将其嵌入原型相似度学习框架，实现无训练微调的高效检测。

**💡 创新点**

创新点在于①利用文本特征作为语义锚点，通过通道级交互与TF‑IDF评分过滤风格噪声，显著扩大类间相似度边距；②将定位任务改造成分层自回归过程，按ViT层级顺序注入高层语义与低层空间特征，逐步细化框框，提升定位精度。

**🔧 技术方法**

核心技术包括：DINOv2+CLIP 的文本-视觉映射、k‑means+TF‑IDF 通道评分与可学习软阈值掩模；多层ViT特征聚合与Stage‑aligned 注入；RoIAlign+卷积递归细化；冻结ViT 进行推理时的原型相似度与mask生成；并使用 Focal、L1、GIoU 等多任务损失。

**📊 数据集**

在 COCO（2014/2017）和 Pascal VOC 公开数据集上进行实验，采用标准的 1/10/30‑shot 与四分拆方式评估。

**📈 对比分析**

与 DE‑ViT、PiDiViT 等 SOTA 方法对比，本文在 COCO 30‑shot 上 nAP 提升 10.1 分，nAP50 +7.1，nAP75 +9.8；在 COCO 1‑shot 上平均 nAP50 提升 10.9 分；在 Pascal VOC 上平均 nAP50 提升 6.9%；同时基准基类性能亦提升 8–9 分。

**⚠️ 局限性**

主要限制包括：①对预训练 ViT 的冻结依赖，可能限制跨域或极端低样本场景的适配；②定位细化过程仍需多阶段计算，推理速度相对较慢；③实验仅覆盖 COCO/VOC，缺乏对小目标或极少样本极端情况的深入验证。

---

## 525. SPAR: Semantic-Pixel Self-Alignment and Adaptive Routing for Unified Multimodal Models

**arXiv ID:** 2606.23041 | [PDF](https://arxiv.org/pdf/2606.23041v1)

**作者:** Hongxiang Li `[一作]` (Hong Kong University of Science and Technology), Long Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出SPAR统一框架，解决语义感知与像素生成的特征差异，构建异构双流自对齐分词器、动态令牌路由和自对齐生成范式，兼顾理解与生成。

**💡 创新点**

创新点在于：①异构双流自对齐分词器将语义保持与像素重建解耦；②动态令牌路由自适应聚合多层MLLM特征；③自对齐生成将分词器本身作为内部对齐教师，消除外部教师依赖。

**🔧 技术方法**

采用Transformer编码器、残差块、MLP投影、Transformer增益、像素解码器、Diffusion（DiT）生成器、LLM+视觉编码器、动态路由网络、对齐投影、GAN判别器等技术。

**📊 数据集**

使用ImageNet 50k、BLIP3o 27M、CC12M 5M、JourneyDB 4M、GPT-Image-Edit、BLIP3o-60K、ShareGPT-4o-Image等数据集。

**📈 对比分析**

在图像重建、视觉理解、文本生成和图像编辑基准上与多种SOTA对比，SPAR在ImageNet 50k重建 rFID 0.27/PSNR 26.65/SSIM 0.856，GenEval整体 0.91，WISE 0.64，ImgEdit 4.01，均优于现有统一与单纯生成模型。

**⚠️ 局限性**

局限在于对语义-像素双流的训练需三阶段精细调度，模型规模大，对资源要求高，并且在极高分辨率生成与跨域迁移上仍有提升空间。

---

## 526. EvoRubrics: Dynamic Rubrics as Rewards via Adversarial Co-Evolution for LLM Reinforcement Learning

**arXiv ID:** 2606.23038 | [PDF](https://arxiv.org/pdf/2606.23038v1)

**作者:** Hongxin Ding `[一作]` (Peking University), Yasha Wang `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了EvoRubrics共进化强化学习框架，使政策LLM与Rubric Generator在每一步实时相互竞争并共同提升；通过双LoRA共享底层模型实现高效参数化。

**💡 创新点**

①在训练步骤级别实现政策与评估器的实时共进化；②引入四维奖励设计（判别性、多样性、对齐、建设性）使Rubric Generator自动生成更具挑战性且可操作的评估标准；③提供完全自监督变体，证明不需外部标注即可获得显著提升。

**🔧 技术方法**

使用GRPO进行策略优化；双LoRA架构共享底层模型；基于结构化Rubric评分的判别器（DeepSeek-V3.2）；多目标奖励（方差、余弦距离、语义相似度、反思提升）对Rubric Generator进行强化学习；自监督版采用自身反思奖励。

**📊 数据集**

主要使用医疗问答数据集HealthBench（4K训练+1K测试），评估在RaR-Medicine、MT-Bench、FollowBench等多域；RubricBench用于检验生成的Rubric作为奖励模型和推理指导。

**📈 对比分析**

与静态rubric（GoldenRubrics）以及动态rubric方法（RuscaRL、OnlineRubrics）进行对照；在HealthBench Hard、RaR-Medicine等任务上表现最优，尤其在HealthBench Hard上提升幅度显著；在OOD任务（MT-Bench、FollowBench）维持甚至提升性能；自监督版本虽略逊但仍超基线；Rubric Generator在RubricBench上提升判断准确率，在推理指导中也显著改善答案质量。

**⚠️ 局限性**

局限性包括：训练主要聚焦医疗领域，缺乏对法律、创意写作等多域的验证；实验规模限定至8B参数，未探索更大模型；共享底层模型可能放大偏见与特定领域倾向；对MOE等更大架构的适应性尚未评估；自监督模式下的评估器偏向安全相关特性，整体平衡性不足。

---

## 527. A Stackelberg Framework for Resource-Aware LLM Agents: Learning, Repair, and Conditional Guarantees

**arXiv ID:** 2606.23026 | [PDF](https://arxiv.org/pdf/2606.23026v1)

**作者:** Baoxun Wang `[一作]` `[通讯]` (Tencent), Baoxun Wang (Tencent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个多轮LLM代理的资源治理框架，将其建模为上下文化的 Stackelberg 游戏，学习了基于上下文、提词和工具使用的执行者响应模型，并在真实 API 上通过校准和动作投影进行修复。

**💡 创新点**

创新点在于：①把资源分配视为领导者-跟随者的前瞻性决策问题；②提出条件性理论保证（存在性、稳定性、投影误差、模拟-真实迁移）；③设计了基于 GAIL 的条件化执行者学习与 PPO 领导者优化相结合的训练流程；④在真实 API 评估中引入了“主动阴影”和“编码-aware”修复策略。

**🔧 技术方法**

使用技术包括：上下文化 Stackelberg 游戏建模、条件化 GAIL（对抗式模仿学习）、PPO 强化学习、欧氏投影修复、实测 API 校准、离线历史重放与阴影评估。

**📊 数据集**

数据集涵盖：公开对话语料、隐私过滤后的代理任务轨迹（用于 GAIL）、约2400 条实时 LLM 代理响应网格（用于 token/quality 预测器）、20 条真实 API 任务实例（共300 轮）用于对照实验。

**📈 对比分析**

通过与保守基线对比，在 300 轮真实 API 评估中，修复后的标量状态控制器将平均 token 用量从 703.8 降至 581.1，减少 17.4%（Welch p=0.022），质量得分从 0.899 降至 0.894，差异不显著（p=0.44）。在任务维度上显示出不同 Pareto 点，修复控制器在成本-质量平衡方面表现最优。

**⚠️ 局限性**

局限性包括：理论保证仅为条件性、未估计 regret/transfer 常数；实验样本有限且未做非劣性检验；质量评价依赖 LLM 判断且任务特异性弱；未进行在线 A/B 验证；动作投影与修复参数仅在有限模型/工具配置下校准，难以通用；日志缺乏真正的 token/结果数据，导致无法完整评估实际收益。

---

## 528. Boosting Neural Video Codec via Scale-Driven Online Flow Refinement

**arXiv ID:** 2606.23023 | [PDF](https://arxiv.org/pdf/2606.23023v1)

**作者:** Tiange Zhang `[一作]` (Peking University), Siwei Ma `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种训练无关、可插拔的SOFR模块，在神经视频编解码器推理阶段通过多尺度流融合和速率感知控制，自动校正运动估计误差，从而提升压缩性能。

**💡 创新点**

创新点在于：①采用粗细两级流生成并用误差驱动的软掩码动态融合；②引入速率感知的下采样尺度和偏置，结合可靠性检查来选择合适的流；③实现了无训练、在线即时适配，避免了昂贵的参数微调。

**🔧 技术方法**

技术手段包括：基于预训练光流网络的双尺度流估计、双线性下/上采样、误差映射与平均绝对误差检查、使用Sigmoid生成软掩码以及依据Q-Index动态调节融合参数。

**📊 数据集**

实验数据集为USTC-TD，包含10段1920×1080分辨率的视频，涵盖多样化运动模式。

**📈 对比分析**

通过在DCVC-SDD、DCVC-FM和EHVC三种主流神经视频编解码器上进行BD-Rate对比，SOFR平均在DCVC-FM上实现PSNR 2.84%与MS-SSIM 4.05%的比特率节约，其他基线亦有提升，且编码时间仅提升约2.4%。

**⚠️ 局限性**

局限性在于：仅对运动估计误差进行修正，未能全面处理极端复杂运动；依赖预训练光流模型的表现；速率感知策略需针对不同Q-Index进行手工设定，泛化至其它编码器或分辨率的效果仍待验证。

---

## 529. ISOPoT: Imaging Sonar Odometry by Point Tracking

**arXiv ID:** 2606.23006 | [PDF](https://arxiv.org/pdf/2606.23006v1)

**作者:** Jaša Samec `[一作]` (University of Ljubljana), Matej Dobrevski `[通讯]` (University of Ljubljana)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于多帧点跟踪的海底声纳里程计ISOPoT。

**💡 创新点**

创新点在于将跟踪任意点方法迁移至声纳图像，替代传统局部特征匹配，并加入轻量级全局优化与网格点管理。

**🔧 技术方法**

采用了TAPNext跟踪器、网格点管理器、RANSAC与ResNet相关性细化等技术。

**📊 数据集**

使用了公开的Aracati 2017数据集和自研的Portoroz 2025数据集。

**📈 对比分析**

在两数据集上与SONIC、DISO等基线比较，ISOPoT在仅声纳和辅助传感器模式下均取得了更低的ATE/RE，表现更稳健。

**⚠️ 局限性**

局限性包括对平面运动假设的依赖、对静态场景的要求，以及对轨迹缓冲的依赖，且在极端低纹理区仍可能产生漂移。

---

## 530. TEXEDO : Test Time Scaling for Controller-aware Language-conditioned Humanoid Motion Generation

**arXiv ID:** 2606.22998 | [PDF](https://arxiv.org/pdf/2606.22998v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 531. VCT: A Verifiable Transcript System for LLM Conversations

**arXiv ID:** 2606.23003 | [PDF](https://arxiv.org/pdf/2606.23003v1)

**作者:** Ruilin Xing `[一作]` (Guangxi University), Wanzhi Xie `[通讯]` (Guangxi University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种可验证对话记录（VCT）体系，用分层的哈希链、Merkle树和双重签名，对 LLM 对话的分支、删除、并发更新和分享等操作进行全局可验证的账号级状态管理。

**💡 创新点**

创新点：
1) 设计了三层认证数据结构，将 Q&A 对作为链节点、会话层用 Merkle 归纳分支、账号层聚合会话；
2) 引入删除序列化与非删除并发合并协议，解决删除与更新冲突；
3) 通过 gossip 叉检测与增量重放拒绝，实现多设备间的可审计一致性；
4) 通过独立链签名的共享协议，保证共享内容的非抵赖与完整性。

**🔧 技术方法**

技术：哈希链、Merkle 认证结构、数字签名、确定性密钥派生、gossip 通信、增量重放检查、并发合并（Deterministic Merge）

**📊 数据集**

未使用公开数据集，采用 Python 原型对 21 KB 对话文本进行实验，评估加密操作延迟与存储开销。

**📈 对比分析**

性能：核心加密操作（哈希、签名、Merkle 路径验证）在子毫秒到毫秒级别；安全元数据占比仅 0.9%，几乎无额外存储成本；实验在实际 LLM 交互规模下验证可部署性。

**⚠️ 局限性**

局限性：未覆盖模型参数删除、硬件安全密钥、密钥吊销与轮换、跨设备密钥泄露恢复等安全场景；对大规模并发会话的极限性能和复杂性分析仍待进一步研究。

---

## 532. EnerInfer: Energy-Aware On-Device LLM Inference

**arXiv ID:** 2606.23001 | [PDF](https://arxiv.org/pdf/2606.23001v1)

**作者:** Bohua Zou `[一作]` (Technical University of Munich), Haibo Chen `[通讯]` (Huawei Central Software Institute)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了EnerInfer框架，实现基于模型结构的吞吐与功耗预测以及能耗与热管理的在线控制，以在保持QoE的前提下提升移动设备LLM推理能效。

**💡 创新点**

①采用离线无监督的结构感知机器学习预测模型，消除对昂贵的离线配置分析需求；②通过分离峰值吞吐与频率缩放预测实现更准确的能效排序；③引入有限时域热预测与MPC热控制，实现主动热管理；④结合能效与QoE的排序驱动频率选择，实现能效与速度的权衡。

**🔧 技术方法**

机器学习（随机森林）进行吞吐/功耗预测；基于模型预测的能效排序；有限时域线性回归热预测；模型预测控制（MPC）热控制；动态频率调节（NPU/DDR DVFS）与回调实现。

**📊 数据集**

通过生成300个无训练的合成LLM（6个关键超参）在三台设备上收集吞吐/功耗；真实LLM（Gemma2、LLaMA2/3.2、Qwen2等）用于验证；手机、笔记本、开发板的真实测量数据。

**📈 对比分析**

与默认最大频率配置、deadline最小满足速率方案以及Oracle最优配置对比；实验显示在高端手机可提升能效50-65%，中端手机26-27%，笔记本10-15%，开发板9-24%；整体能耗下降4-11%；热控制使shell温度保持≤42℃并延长推理时长30-40%。

**⚠️ 局限性**

依赖离线收集的合成模型训练，平台迁移仍需重新采集≈300个模型数据；对极端高热或内存密集型后台任务的响应仍有限；对稀疏/专家/混合专家等新模型不直接支持；预测误差在板子上影响较小但仍存在。

---

## 533. Group-Graph Policy Optimization for Long-Horizon Agentic Reinforcement Learning

**arXiv ID:** 2606.22995 | [PDF](https://arxiv.org/pdf/2606.22995v1)

**作者:** Yunan Wang `[一作]` (Peking University), Qi Zhang `[通讯]` (Microsoft Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 Group‑Graph Policy Optimization（G2PO），将多步 LLM 代理交互轨迹重构为全局状态转移图，并通过组聚合的状态值估计和边缘中心优势实现细粒度信用分配；

**💡 创新点**

① 将线性轨迹转换为状态图并聚合相同观测，① 组聚合状态值估计大幅降低采样方差；② 边缘中心优势通过全局标准化 TD 误差给出绝对贡献；

**🔧 技术方法**

基于 GRPO 的无价值函数策略梯度框架；状态图构建与聚合；全局 TD 误差标准化；节点/边缘/情节级优势计算；以及 Qwen2.5/Gemini/DeepSeek 等大语言模型；

**📊 数据集**

WebShop、ALFWorld 与 AppWorld 三大长时序代理基准；

**📈 对比分析**

与前沿 LLM、提示式代理（ReAct、Reflexion）及 RL 训练方法（PPO、RLOO、GRPO、GiGPO）比较；G2PO 在 WebShop、ALFWorld、AppWorld 上分别提升约 22.2%、14.4% 及 3.1% 的成功率，显著优于所有基线；

**⚠️ 局限性**

仍依赖观测匹配，难以处理高度连续或极度随机的环境；目前仅在离散动作/观测场景验证，可能不适用于连续控制任务；

---

## 534. Attention-Spectrum Regularization for Replay-Free Continual Multimodal LLMs

**arXiv ID:** 2606.23063 | [PDF](https://arxiv.org/pdf/2606.23063v1)

**作者:** Chuangxin Zhao `[一作]` (Hong Kong University of Science and Technology), Yang Liu `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无重放的注意力频谱正则化方法（ASR），通过对跨模态注意力进行二维傅里叶编码，保持每个推理技能的频谱原型，从而在持续学习过程中缓解多模态大模型的灾难性遗忘。

**💡 创新点**

创新点在于：①将跨模态注意力视为二维信号，并提取其尺度与方向的频谱统计；②仅存储每个技能的谱原型（均值、协方差与平均角谱），实现轻量级的无重放正则；③引入自适应技能权重和几何正则，兼顾内部注意力稳定与外部输出的多样性。

**🔧 技术方法**

核心技术包括：2D离散傅里叶变换提取频谱特征、马氏距离和KL散度构建谱蒸馏损失、经验移动平均维护技能原型、轻量化几何正则、LoRA/Adapter等参数高效微调。

**📊 数据集**

在五大持续学习基准上评测：VQA v2、VQACL、CLT‑VQA、CoIN、UCIT。

**📈 对比分析**

相较于回放、正则化、MoE、LoRA等强基线，ASR在所有基准上实现更高的平均准确率/最后表现，并显著降低遗忘率；在LLaVA‑1.5‑7B、Qwen2.5‑VL‑7B、InternVL3‑8B等不同后端模型上均保持一致提升。

**⚠️ 局限性**

局限包括：需要预先定义并解析技能集合，技能解析器对问句的依赖可能导致误判；在极度相似任务间频谱差异小，正则效果可能有限；原型存储虽然轻量，但在极大模型或极度多样化数据下可能不足以覆盖全部技能。

---

## 535. Schemata, Cyclic Proofs and Herbrand Systems

**arXiv ID:** 2606.23040 | [PDF](https://arxiv.org/pdf/2606.23040v1)

**作者:** Alexander Leitsch `[一作]` (TU Wien), Stella Mahler `[通讯]` (TU Wien)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一种将循环证明转化为证明结构（proof schema）的算法，并定义了相应的Herbrand系统，用于在互斥递归归纳定义下实现循环证明的可执行化。

**💡 创新点**

创新点在于将传统的循环证明与证明结构统一在一个框架内，并提供了可执行的条件化定义和Herbrand系统的提取方法，实现了对互斥递归归纳定义的形式化处理。

**🔧 技术方法**

主要技术包括：序贯演算、无限/循环证明系统、归纳谓词定义、分解算法以及条件化定义的规范化与子序列化。

**📊 数据集**

无具体数据集，论文基于理论构造的示例证明与归纳定义进行验证。

**📈 对比分析**

方法与性能比较基于理论证明而非实验，结果表明在受限的归纳定义类型下能够保证终止性和完整性。

**⚠️ 局限性**

局限性在于只能处理无新变量出现、互斥分支的归纳谓词定义类型，无法覆盖所有循环证明或非终止的归纳定义。

---

## 536. IPO Finance Agent: Evaluation of LLM Financial Analysts beyond Finance Agent v2, with Automated Rubric Generation -- the Case of the SpaceX (SPCX) IPO

**arXiv ID:** 2606.23032 | [PDF](https://arxiv.org/pdf/2606.23032v1)

**作者:** Mostapha Benhenda `[一作]` `[通讯]`, Mostapha Benhenda

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 IPO Finance Agent 基准，用于评估 LLM 在首次公开募股（S‑1）尽职调查中的推理与检索能力，扩展了 Finance Agent v2 的范畴；

**💡 创新点**

创新点包括：① 针对 IPO 的六大分析领域与专业工作流的问答分类；② 采用上下文增强检索（contextual retrieval）替代原始块检索，提升长文档检索质量；③ 通过评估‑优化循环实现自动化 Rubric 生成，显著降低人工标注成本；

**🔧 技术方法**

技术手段涵盖：基于工具调用的 Agentic harness（包含 web search 与 EDGAR 接口）；上下文增强的 Dense Retrieval；多模型集成与事实抽取、合并、校准的评估‑优化管道；以及 LLM 生成的评估器和修正器；

**📊 数据集**

数据集为 1,000 个 IPO 尽职调查问题，其中 70 个公开覆盖 SpaceX S‑1，剩余 930 个保密，以防 Benchmark 泄露；问题涵盖了六个领域并标注专业工作流；

**📈 对比分析**

通过与 Finance Agent v2 公开榜单对比，IPO Finance Agent 的 Qwen 3.7 Max 在 70 题上取得 79.4% 准确率（$0.30/查询），MiMo 2.5 Pro 达到 76.8%（$0.05/查询），均明显超越 FABv2 的 57.9% 最高水平；在成本‑精度 Pareto 前沿表现突出；

**⚠️ 局限性**

局限性包括：Benchmark 与 Finance Agent v2 在任务、文件类型、评分指标等方面不同，无法直接对照；未进行单独 ablation 分析检索方式对性能的具体贡献；每个领域样本量有限，评估稳定性受限；最终 Rubric 仍需人工审核，完整自动化尚未实现。

---

## 537. From numerical proportions to analogical proportions between probabilities

**arXiv ID:** 2606.23029 | [PDF](https://arxiv.org/pdf/2606.23029v1)

**作者:** Henri Prade `[一作]` (IRIT), Gilles Richard `[通讯]` (IRIT)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究并定义了概率分布之间的类比比例（analogical proportion），探讨了其理论性质，并在实际数据集上进行实验验证。

**💡 创新点**

提出了基于算术比例和算术‑几何组合的概率分布类比比例定义，并证明了它们分别保留总变差距离和Kullback‑Leibler散度；同时首次在大规模实际数据集上检验这些类比比例的有效性。

**🔧 技术方法**

采用数学证明、类比不相似度（Analogical Dissimilarity）度量、概率分布的算术/几何比例、KL 散度、总变差距离等技术；实验部分利用频率估计对概率分布进行构造，并通过可视化的 AD 分布评估类比比例。

**📊 数据集**

MovieLens 100K（用户评分）和 US Traffic Accidents（交通事故严重度）两个公开数据集。

**📈 对比分析**

通过构造满足类比比例的 profile 四元组，计算其对应概率分布的 AD，并将 AD 在 [0,1] 上可视化；实验结果显示在足够大样本量下 AD 小于 0.05 的比例占比较高，说明类比比例在分布空间中得到较好保留，性能良好。

**⚠️ 局限性**

目前定义主要适用于离散有限分布，连续分布的推广仍待研究；算术‑几何类比比例要求较高，实际出现率低；类比比例并非在所有 profile 组合中都成立，需进一步理论与实验完善。

---

## 538. Counterfactual learning of new adaptive instructional policies using logged data

**arXiv ID:** 2606.23015 | [PDF](https://arxiv.org/pdf/2606.23015v1)

**作者:** Samuel Girard `[一作]` (Inria Saclay), Jill-Jênn Vie `[通讯]` (Inria Saclay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种离线上下文多臂赌博机框架，从已有的智能辅导系统（ITS）日志中直接学习新的适应性教学策略。

**💡 创新点**

创新点在于：①将学生与题目映射到连续的潜在熟练度-难度尺度上；②设计以“流”与可达难度为基础的即时奖励；③引入基于连续尺度的行为策略估计作为偏倚估计和适应性诊断工具。

**🔧 技术方法**

主要技术包括：1PL Rasch 模型对熟练度与难度进行标度化；连续上下文多臂赌博机的离线政策评估（IPS、SNIPS、DR、MIS、DM）；使用高斯（或截断高斯）策略进行政策学习；以及对行为策略进行回归估计。

**📊 数据集**

实验使用四个真实 ITS 数据集：RoboMission、Assistments2015、Assistments2009、Pix。

**📈 对比分析**

通过将学习到的策略与原始行为策略和 IRT oracle 在测试集上进行 OPE（IPS、SNIPS、DR、MIS、DM）比较，结果显示学习到的策略平均提升 10%–40% 的预期奖励，SNIPS 与 DR 在不同数据集上表现最为稳健。

**⚠️ 局限性**

局限性包括：①仅优化即时奖励，未考虑长期学习收益；②日志中探索不足导致评估方差大；③行为策略采用简单参数化，可能无法捕捉复杂分布；④对 Rasch 校准高度敏感；⑤缺乏序列决策建模，无法直接得到多步策略。

---

## 539. MotionMAR: Multi-scale Auto-Regressive Human Motion Reconstruction from Sparse Observations

**arXiv ID:** 2606.23000 | [PDF](https://arxiv.org/pdf/2606.23000v1)

**作者:** Yuhua Luo `[一作]` (Xiamen University), Cheng Wang `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一种多尺度自回归框架 MotionMAR，能够从极少量的 VR/AR 传感器观测（头部、双手）恢复完整的全身运动序列。

**💡 创新点**

创新点包括：① 采用 Temporal Multi‑scale VQ‑VAE 将运动序列分解为不同时间分辨率的离散嵌入；② Scale‑aware Control 模块在每个尺度上对齐稀疏观测，保证生成结果严格贴合真实轨迹；③ Motion Refinement Network 通过双向 GRU 对离散解码结果进行平滑，消除量化噪声；④ 整体采用粗到细的自回归生成策略，充分利用人体运动的层次结构。

**🔧 技术方法**

使用技术主要包括：VQ‑VAE、残差量化、Transformer（GPT‑2 风格）自回归网络、线性插值对齐、AdaLN 适应层归一化、双向 GRU 细化网络。

**📊 数据集**

训练和评估主要基于 AMASS 数据集（包含 CMU、BMLrub、HDM05 等子集），并在更大规模的 S3 组合集上验证泛化；实验亦在公开的真实 VR 数据上进行测试。

**📈 对比分析**

与 Final IK、CoolMoves、LoBSTr、VAE‑HMD、AvatarPoser、AvatarJLM、AGROL、SAGE、MAGE、HiPART、RPM 等多类方法对比，MotionMAR 在 MPJRE、MPJPE、MPJVE、Jitter 等指标上均优于或接近最先进方法，尤其在手部和下肢精度方面显著提升；推理速度达到 61.76 FPS，满足实时 VR/AR 需求。

**⚠️ 局限性**

局限性包括：① 对稀疏观测的依赖度仍较高，观测质量差时恢复效果下降；② 多尺度量化与自回归训练复杂度较高，训练成本较大；③ 对极端快速运动或遮挡情境的鲁棒性尚未充分验证；④ 当前仅处理 6D 旋转表示，若要支持多模态输入需进一步扩展。

---

## 540. Do Sparse Autoencoders Learn Meaningful Concept Hierarchies?

**arXiv ID:** 2606.22994 | [PDF](https://arxiv.org/pdf/2606.22994v1)

**作者:** Nils Grandien `[一作]` (Technische Universitaet Darmstadt), Kristian Kersting `[通讯]` (Technische Universitaet Darmstadt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

为无监督概念学习中稀疏自编码器（SAE）提出通用层级结构要求，并基于这些要求设计了一套量化评估指标，随后对当前几种在视觉数据上训练的 SAE 方法进行评估。

**💡 创新点**

创新点在于：①系统归纳了层级结构的概念层、结构层与激活层三类要求；②将这些要求转化为可在无监督情境下计算的指标；③通过统一评估框架对比多种 SAE 结构，揭示了“硬吸收”与“软吸收”对层级质量的影响。

**🔧 技术方法**

使用稀疏自编码器（TopK、Matryoshka、EWG、MP、H‑SAE）与图结构构造方法（基于条件激活概率的后置 DAG 构建）。

**📊 数据集**

数据集为 CC3M 图像，使用 CLIP ViT‑L/14 与 DINOv2‑base 两种视觉编码器生成全局图像嵌入作为 SAE 训练与评估的输入。

**📈 对比分析**

比较方法：在同一嵌入空间下训练不同 SAE，使用构造的层级图计算 9 条指标（_ha、_ss、_ai、_ad、_rf、_cc 等）。结果显示：Baseline TopK 已能形成语义化层级，但激活一致性差；Matryoshka（尤其是 ActMSAE）在多层级和共激活指标上显著优于其他方法；H‑SAE 通过显式约束实现高共激活频率，却缺乏语义一致性，导致指标偏低；整体上多方法仍受吸收问题影响。

**⚠️ 局限性**

局限性：①评估指标完全无监督，仅通过图像嵌入近似语义相似度，难以反映人类可解释性；②图结构构造仅基于条件激活概率，可能无法捕获所有层级要求，对某些 SAE（如 MP‑SAE）效果差；③对软吸收的处理仍不足，未来需在 SAE 训练中更充分考虑激活一致性。

---

## 541. DrivingVoxels: Compositional Sparse Voxel Rasterization for Dynamic Driving Scene Reconstruction

**arXiv ID:** 2606.23031 | [PDF](https://arxiv.org/pdf/2606.23031v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 542. PeLAP-A: Adaptive Latent Pruning for Lightweight Latent Diffusion Models

**arXiv ID:** 2606.23086 | [PDF](https://arxiv.org/pdf/2606.23086v1)

**作者:** Kissa Zahra `[一作]` (National University of Computer and Emerging Sciences), Zaib Un Nisa `[通讯]` (National University of Computer and Emerging Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 PeLAP-A 框架，在潜在扩散模型中插入轻量级重要性预测器，实现对 VAE 编码器输出的自适应通道级稀疏化。

**💡 创新点**

创新点在于通过学习可调节的通道掩码实现输入自适应稀疏，并首次揭示了“稀疏崩塌”现象：激进稀疏正则化导致所有通道被压制，却仍能提升去噪损失。

**🔧 技术方法**

采用了 VAE-UNet 潜在扩散模型、全局平均池化 + 两层 MLP 的重要性预测器，以及联合训练的稀疏正则化（λ·mask）和损失组合（VAE + Diffusion + λ·mask）。

**📊 数据集**

实验基于 CIFAR-10 图像数据集进行。

**📈 对比分析**

将基线（λ=0）与 ALPD（λ=0.01）在验证去噪损失、VAE 重建 MSE、FID、推理时间和激活通道数等指标对比，结果显示去噪损失和 MSE 降低（0.0236 vs. 0.0240；22.59 vs. 24.67），但 FID 上升（278.1 vs. 362.6），激活通道从 4 降至 0。

**⚠️ 局限性**

局限性包括稀疏崩塌导致无法实现部分通道保留，生成质量（FID）显著下降；实验仅在单一架构和 CIFAR-10 数据集上验证，泛化性待进一步探究。

---

## 543. Foresight: Failure Detection for Long-Horizon Robotic Manipulation with Action-Conditioned World Model Latents

**arXiv ID:** 2606.23085 | [PDF](https://arxiv.org/pdf/2606.23085v1)

**作者:** Haoran Zhang `[一作]` (University of Michigan), Odest Chadwicke Jenkins `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于动作条件世界模型的长周期机器人任务故障检测框架

**💡 创新点**

创新点在于将世界模型的预测潜在表示作为跨策略通用特征，结合因果变压器和功能合规预测实现无策略内置信度依赖的实时监测

**🔧 技术方法**

使用V-JEPA 2-AC动作条件世界模型、因果Transformer、功能合规预测（FCP）以及传统MLP/LSTM对比

**📊 数据集**

在LIBERO-Long、ManiSkill-Long、BEHAVIOR-1K三大仿真长周期基准及实际ReactorX-200和Franka机器人上进行评估

**📈 对比分析**

与FAIL-Detect、SAFE、RND、Gauge等基线对比，ROC‑AUC最高可达0.94（LIBERO-Long）或0.93（ReactorX），平衡准确率在长周期任务上显著提升

**⚠️ 局限性**

主要限制是预训练世界模型的计算与延迟开销，导致对高频、需快速闭环控制的任务部署受限，且合规校准需与部署环境匹配

---

## 544. SemCEB: A Cardinality Estimation Benchmark for Semantic Operators

**arXiv ID:** 2606.23081 | [PDF](https://arxiv.org/pdf/2606.23081v1)

**作者:** Andreas Zimmerer `[一作]` (University of Technology Nuremberg), Andreas Kipf `[通讯]` (University of Technology Nuremberg)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了SemCEB基准，用于评估多模态大语言模型驱动的语义操作（如语义过滤器与连接）的基数估计。

**💡 创新点**

首次提供真实数据集、102条多模态查询（覆盖不同选择性与谓词类别）、并引入BERTopic评估语义偏斜，从而填补现有基准在语义基数估计上的空白。

**🔧 技术方法**

使用采样估计、改进的Semantic Histograms、LLM推理、预计算的文本/图像嵌入以及基于BERTopic的语义类别划分等技术。

**📊 数据集**

基于亚马逊“Arts, Crafts and Sewing”商品与评论数据，包含45k产品、936k评论，配有文本、图像、嵌入及截断标记。

**📈 对比分析**

在规模因子1000/100下，对不同采样率（1%、5%、10%）与Semantic Histograms进行比较；采样5%已获得良好q‑error，但成本和延迟较高；Semantic Histograms快速（≈6 s）但q‑error较高，且仅支持单列谓词。

**⚠️ 局限性**

仅评估单一语义操作，未考虑与传统关系操作或其他语义操作的交互；Semantic Histograms适用范围有限；未覆盖群组BY等其他语义操作，且在高成本场景下采样法不具备可扩展性。

---

## 545. From Text Metrics to Model Internals: A Study of Whisper ASR Hallucination Detection

**arXiv ID:** 2606.23060 | [PDF](https://arxiv.org/pdf/2606.23060v1)

**作者:** Jan Jasiński `[一作]` (AGH University of Krakow), Konrad Kowalczyk `[通讯]` (AGH University of Krakow)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在 Whisper large v3 上研究了 ASR hallucination 的检测，系统比较了三种范式：基于文本特征的分类、基于 LLM 的提示推理以及对解码器内部状态的线性/BLSTM 探测，并提出了轻量级的 late‑fusion meta‑classifier。

**💡 创新点**

①首次证明 Whisper 解码器中间层的嵌入携带可用于无参考 hallucination 检测的显著信息；②利用 BLSTM 对完整解码序列进行建模，获得最高 F1；③提出将文本特征与内部状态融合的 meta‑classifier，进一步提升整体检测性能。

**🔧 技术方法**

文本特征提取（WER、CER、BERTScore、语义指标等） + XGBoost / Logistic Regression；LLM 交互式提示（GPT‑4o‑mini、Gemini 等）进行零/少量提示推理；内部状态探测采用 Logistic Regression 线性探针和 BLSTM 对解码器层嵌入序列进行分类；最后使用 Logistic Regression 进行 late‑fusion 融合。

**📊 数据集**

HALAS 数据集，包含 3611 条 Whisper 预测结果，其中 858 条（23.8%）被人工标注为 hallucination；数据来自 Earnings‑22 真实语音，划分为 train/test 并支持 5‑fold 交叉验证。

**📈 对比分析**

在有参考的情况下，文本特征 + XGBoost 达到 F1 62.8%，BLSTM 内部状态（无参考）达 F1 65.5%；LLM 在无参考时 F1 仅 32.8%，即使在有参考时也低于 XGBoost；融合后 Meta‑classifier 的 F1 提升至 68.3%，ROC‑AUC 达 90%。相比传统 WER 等 oracle 指标，本文方法在零引用环境下仍保持可观性能。

**⚠️ 局限性**

①需要访问 Whisper 的内部解码状态，难以迁移到其他非可见内部状态的模型；②单词级 hallucination（如功能词插入）仍难以检测；③LLM 方案计算成本高且在无参考时性能急剧下降；④实验仅覆盖 Whisper，跨模型推广及轻量级 LLM fine‑tuning 仍待研究。

---

## 546. Training Open Models for Agentic Phone Use

**arXiv ID:** 2606.23049 | [PDF](https://arxiv.org/pdf/2606.23049v1)

**作者:** Zhengyang Tang `[一作]` (Tencent Hunyuan), Han Hu `[通讯]` (Tencent Hunyuan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 PhoneBuddy 训练流程，将真实手机环境与基于真实 GUI 结构重构的 mock‑app 环境 PhoneWorld 结合，构建可在大规模可重置环境中进行强化学习的手机代理。

**💡 创新点**

创新点在于：①将真实环境与 mock‑app 环境的训练并行化，充分利用两者互补优势；②利用 PhoneWorld 的可复现、自动验证特性，为真实手机训练提供可扩展的交互信号。

**🔧 技术方法**

采用 Qwen3.5‑4B 作为基础模型，先进行跨环境的监督微调（SFT），随后在真实手机环境与 PhoneWorld 中分别或混合进行 50 步强化学习，使用任务完成率为奖励，辅以 rubric‑based 模型评判与 PhoneWorld 的规则检查。

**📊 数据集**

使用约 950,758 步长的动作轨迹数据，既包含真实手机环境收集的轨迹，也包含从 PhoneWorld 生成的 mock‑app 轨迹，覆盖单 app、跨 app、微信小程序以及 AndroidWorld 的多种任务。

**📈 对比分析**

在 150 任务的真实手机人工评测（分为单 app、跨 app、微信小程序）和 AndroidWorld 上进行对比；SFT 版成功率 34.0%，真实 RL 版 54.0%，混合 RL 版 62.0%；在 AndroidWorld 上分别为 60.3%、77.2%、83.2%，显示混合训练在大多数场景上实现了显著提升。

**⚠️ 局限性**

局限性：跨 app 流程的成功率仍低（仅 18%），表明 mock‑app 任务池缺乏跨应用信息传递与长期状态追踪的支持；此外，当前的 PhoneWorld 主要覆盖单 app，无法充分训练多 app 协同工作与复杂长时序决策。

---

## 547. Prime Fourier Embeddings: A Principled Basis for Modular Arithmetic

**arXiv ID:** 2606.23044 | [PDF](https://arxiv.org/pdf/2606.23044v1)

**作者:** Hyunsang Hwang `[一作]` (Korea University), Donghun Lee `[通讯]` (Korea University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出 Prime Fourier Embeddings（PFE）将整数编码为以质数为索引的余弦正弦对，直接将模运算的结构内置进嵌入。

**💡 创新点**

创新点在于利用阿德里克谐波分解与中国剩余定理，证明任意与PFE等变的线性映射必为块对角结构，并通过实验验证质数通道的显著专一性。

**🔧 技术方法**

使用的技术包括基于欧拉公式的 p‑adic 字符、Schur 引理证明块对角性、以及标准深度网络与PFE的组合。

**📊 数据集**

实验数据集为人工合成的整数对 (a,b) 在不同模数和输入范围内的加法任务，覆盖单质数模数与各类平方自由复合模数。

**📈 对比分析**

与基线的比较在于消融实验，结果显示任务相关质数通道的误差提升超过500×，且在所有测试模数上均达到100%准确率，表明 PFE 的可解释性与性能。

**⚠️ 局限性**

局限性包括梯度下降为何收敛到等变解尚未解释，以及深度层间的交互在理论上被限定为零但在实践中的重要性分配仍不明。

---

## 548. The Model as One Rater Among Several: Measuring Political Positions in Data-Sparse Regions with a Language-Model Panel

**arXiv ID:** 2606.23042 | [PDF](https://arxiv.org/pdf/2606.23042v1)

**作者:** Tarek Gara `[一作]` `[通讯]` (Independent researcher), Tarek Gara (Independent researcher)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种将大型语言模型作为多位评审者的面板，用来衡量中东及北非地区政党和政治人物在16个轴向上的政治立场；

**💡 创新点**

创新点在于：①把模型当作可聚合的、可检验的评审者而非直接测量工具；②引入“适用性规则”区分空白与零；③使用声明与行为双镜头区分表态与实际行动；④在同一框架内评估模型一致性与偏差，强调可靠性而非单纯有效性；

**🔧 技术方法**

技术主要包括：大语言模型（Claude、Gemini、GPT、Gemma、Gemini、DeepSeek、Kimi、MiniMax、Qwen、Grok）对文本进行评分；Krippendorff α、绝对误差（MAD）等统计评估；分层面板聚合与中值合成；

**📊 数据集**

数据集为中东与北非地区的政治党派与公众人物（共98党、274名人物）与其宣言、演讲、法律文件等原始文本（286份平台/宪章，49份法律文书），共计12,867个评分单元；

**📈 对比分析**

比较方法：在同一模型面板下进行两轮评分（有无轴定义）以检验说明书效应；计算Krippendorff α（0.86）与绝对差异；通过“争议”指标识别分歧；性能上模型面板表现出高可靠性（α≥0.84），但可靠性不等同有效性；

**⚠️ 局限性**

局限性包括：①模型来源相似，可能共享偏差，无法单独证明有效性；②缺乏人类专家验证（尚在计划中）；③模型训练截止时间导致时事性不足；④轴向设计与适用性规则存在主观性；⑤数据覆盖不均，某些轴向样本稀少；

---

## 549. Learning Stable Canonical Worlds for Novel View Synthesis and Beyond

**arXiv ID:** 2606.23027 | [PDF](https://arxiv.org/pdf/2606.23027v1)

**作者:** Xiaoyu Xu `[一作]` (City University of Hong Kong), Kede Ma `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 CanonicalGS 方法，先把多视角观测映射到稳定的场景中心表示，然后再解码成可渲染的高斯原语，用于新视角合成和下游感知。

**💡 创新点**

在传统视角依赖的 FFGS 里引入场景中心的稀疏聚合与不确定性加权，先聚合可靠观测再解码，使多视角输入能稳步提升表示而不是增加噪声；同时解码时用可靠性控制不透明度，强化可解释性和可迁移性。

**🔧 技术方法**

基于 DINO‑v2 特征提取 + UNet 深度估计 + 可靠性加权的 voxel 级聚合 + 可靠性约束的 GP 解码；利用 Plane‑Sweep 生成匹配概率，使用 softmax 深度概率；聚合后通过 MLP 产生均值、协方差、颜色。

**📊 数据集**

在室内 RealEstate10K 和室外 DL3DV 上训练/评估；使用 Mask2Former 伪标签评估语义分割。

**📈 对比分析**

对比 DepthSplat、MVSplat、FreeSplat、ZPressor 等 FFGS 族，以及 Gaussian‑space 合并变体；在多视角合成中 PSNR/SSIM 提升 2.5 dB，LPIPS 降低；在语义分割中线性探针精度提升 11%。

**⚠️ 局限性**

依赖准确的相机投影与深度估计；对姿态误差、遮挡、动态物体敏感；当前仅使用单一标量可靠性，未建模复杂不确定性；体素网格固定，规模受限；需要进一步联合优化几何、姿态与场景表示。

---

## 550. AdaReP:Adaptive Re-Planning under Model Mismatch for Neural World-Model Predictive Control

**arXiv ID:** 2606.23079 | [PDF](https://arxiv.org/pdf/2606.23079v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 551. Nautilus: A Verifiable Hierarchical Federated Learning Framework for Vehicular-Edge-Cloud Systems

**arXiv ID:** 2606.23017 | [PDF](https://arxiv.org/pdf/2606.23017v1)

**作者:** Linyang Wu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Yi Sun `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为Nautilus的可验证分层联邦学习框架，针对车辆‑边缘‑云（VEC）环境中的资源异质性与信任缺失，动态分配压缩率与训练任务，并通过区块链+零知识证明实现调度与执行可验证。

**💡 创新点**

创新点在于将多维资源感知的动态调度与可验证的零知识证明相结合，构建轻量级的区块链验证机制，同时采用层级RSU聚合架构实现异构资源优化与可信性保证。

**🔧 技术方法**

使用的技术包括多维资源感知的动态调度算法、3×3能力矩阵分层、Top‑K稀疏化与量化压缩、zk‑SNARK 零知识证明、FISCO BCOS 区块链以及 ResNet18 模型训练。

**📊 数据集**

数据集：CIFAR‑10 图像分类数据集，采用非IID划分模拟车辆数据异构。

**📈 对比分析**

通过与FedAvg和zkFL两种基准方案比较，实验表明Nautilus在10/30/50节点下，训练时间分别降低约39.3%/53.2%/59.9%，通信成本仅为zkFL的16.7%，同时模型准确率差距≤1.5%。

**⚠️ 局限性**

局限性在于压缩导致的微小精度损失、ZK证明仍需一定计算资源，对极低功耗设备的证明生成开销可进一步优化，且仅在VEC场景验证，跨域泛化需进一步验证。

---

## 552. A Novel Approach to Temporal QoS Estimation via Extended Kalman Filter-Incorporated Latent Feature Analysis

**arXiv ID:** 2606.23010 | [PDF](https://arxiv.org/pdf/2606.23010v1)

**作者:** Ye Yuan `[一作]` (Southwest University), Xin Luo `[通讯]` (Southwest University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种基于Extended Kalman Filter与ALS的双向模型-数据驱动框架EKL，用于预测时间序列QoS数据，解决传统纯数据驱动模型对非平稳波动捕捉不足的问题。

**💡 创新点**

创新点包括：①引入EKF生成时间动态潜在特征，①利用ALS提取时间不变潜在特征，实现模型与数据双向驱动；②设计基于调用密度的并行策略（DPS）提升计算效率；③对EKL的收敛性给出严格理论证明。

**🔧 技术方法**

使用技术包括Extended Kalman Filter、LeakyReLU激活、Alternating Least Squares、密度导向多线程并行、RMSE/MAE评估、Wilcoxon与Friedman统计检验。

**📊 数据集**

实验数据集为WS‑DREAM的吞吐量（D1）与响应时间（D2）以及阿里巴巴微服务的QoS（D3），共涵盖三大真实网络场景。

**📈 对比分析**

与12种主流时间序列LFA模型（如CGTF、TeDCaN、HRST‑LR等）在16个不同稀疏比例的测试案例中进行对比。EKL在13/16案例中取得最低RMSE/MAE，平均提升5–30%（针对RMSE），在计算效率上实现11.7×的加速（最高DPS 16线程）。

**⚠️ 局限性**

局限性：①依赖手工调参，缺乏自适应机制；②噪声模型假设为独立高斯，未考虑空间时空相关性；③在极度稀疏的数据集（如D3）上并行效率下降，需进一步优化。

---

## 553. MotionHalluc: Diagnosing Kinematic Hallucinations in Fine-Grained Motion Reasoning

**arXiv ID:** 2606.23061 | [PDF](https://arxiv.org/pdf/2606.23061v1)

**作者:** Weile Guo `[一作]` (Sun Yat Sen University), Chao Yu `[通讯]` (Sun Yat Sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MotionHalluc 基准，用于评估跨视频比较中的运动幻觉，并提供了无训练的 Perceive‑Parse‑Verify（PPV）验证管道；

**💡 创新点**

创新点在于：①设计了专门针对方向、归因、时序三类运动幻觉的评测任务；②将运动量化测量嵌入模型推理，显著降低幻觉；

**🔧 技术方法**

使用了大规模多模态模型（如 Gemini‑3‑Flash、Qwen3‑5‑plus 等），结合 4D‑Humans 运动重建、DTW 对齐和语义解析生成可执行的测量查询；

**📊 数据集**

数据集来源于 Fit3D（32 种健身动作，553 对视频），每对视频配有 3D 运动捕捉和手工标注的纠正指令；

**📈 对比分析**

与五个主流 LMM 进行对比，原始模型在三类幻觉任务上表现差异大（如方向幻觉原始/反向精度差距超过 30%）；在 PPV 方案下平均提升约 10.6%，在时序幻觉上最高提升 32.9%；

**⚠️ 局限性**

局限性在于仅覆盖受控室内健身动作，未涉及复杂户外或多人互动场景，且对运动重建误差的鲁棒性仍有限。

---

## 554. Some Results about the Expressivity of Preference-Incomplete Structured Argumentation Frameworks

**arXiv ID:** 2606.23055 | [PDF](https://arxiv.org/pdf/2606.23055v1)

**作者:** Antonio Yuste-Ginel `[一作]` `[通讯]` (University of Málaga), Antonio Yuste-Ginel (University of Málaga)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

研究了带有不确定偏好（pref‑ISAF）的ASPIC^+结构化论证框架的表达能力，并将其与多种不确定攻击的抽象形式（def‑IAF、dep‑IAF等）进行比较；给出了若干负向结果，并提出了一个非平凡的正向阈值猜想。

**💡 创新点**

创新点在于：①首次将不确定偏好视为产生抽象不确定攻击的源头，并在抽象层面上定义其完成集合；②证明了dep‑IAF是表达不确定攻击的上界，且pref‑ISAF无法被def‑IAF或dis‑IAF所模拟；③提出了一个可被dis‑imp‑IAF捕捉的正向阈值猜想，指出了表达能力的边界。

**🔧 技术方法**

使用了形式化逻辑与图论技术：定义AF、IAF、dep‑IAF、pref‑ISAF等；通过完成集、等价与同构概念进行表达能力比较；利用依赖关系（disjunctive、implicative）构造依赖集合；展开严谨的数学证明与反例构造。

**📊 数据集**

未使用任何实际数据集；所有结果均为理论推导与形式化证明。

**📈 对比分析**

比较方法：定义完成集的等价（同构）关系，并通过“至少具有表达力”的偏序关系进行比较；性能评估为理论可达性与不可达性，没有实验或计算复杂度评估；已证明若干负向不可模拟性，但正向阈值猜想仍未完成。

**⚠️ 局限性**

局限性包括：①未给出不确定偏好在抽象层面的完整计算复杂度分析；②所提出的正向阈值猜想尚未证明；③仅考虑了论证框架内部的结构与偏好，未探讨更低层的规则/前提不确定性；④缺乏实验验证或案例研究来佐证理论结论。

---

## 555. HALAS: A Human-Annotated Dataset of Hallucinations of Modern ASR Systems

**arXiv ID:** 2606.23048 | [PDF](https://arxiv.org/pdf/2606.23048v1)

**作者:** Mateusz Barański `[一作]` (AGH University of Krakow), Konrad Kowalczyk `[通讯]` (AGH University of Krakow)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了HALAS数据集，对七种最先进ASR模型在真实收益电话录音中的自然幻觉进行人工标注，并基准评估了多种幻觉检测方法。

**💡 创新点**

首次公开自然语音幻觉标注数据，揭示不同模型幻觉高度重叠且多集中于少数短语，并证明传统代理指标和现有检测器在此数据上表现差强人意。

**🔧 技术方法**

使用了文本代理指标（WER、CER、BERTScore等）、LLM比较、解码器嵌入分类器（DE）以及XGBoost融合模型。

**📊 数据集**

数据来源于Earnings 22（119 小时收益电话录音）与七款主流ASR模型的推断结果，全部标注采用10名专业英语审阅员完成。

**📈 对比分析**

在HALAS测试集上，代理指标的平均ROC‑AUC约为0.83，DE多层（2、13、23）得到最高F1≈56.1%；相比之下，基于LLM的检测器仅达到F1≈40–45%，说明目前的检测方法仍难以可靠识别自然语音幻觉。

**⚠️ 局限性**

限制包括数据集样本为高幻觉率、人工标注成本高、模型间幻觉共性导致跨域泛化挑战，以及对非英语或多说话人场景的适用性尚未验证。

---

## 556. UECP: Uncertainty-Enhanced Collaborative Perception

**arXiv ID:** 2606.23046 | [PDF](https://arxiv.org/pdf/2606.23046v1)

**作者:** Kang Yang `[一作]` (Renmin University of China), Yongcai Wang `[通讯]` (Renmin University of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了基于不确定性增强的协同感知框架UECP，用不确定性图引导多尺度BEV特征融合，显著提升了多车感知性能。

**💡 创新点**

创新点包括：①通过LiDAR点云密度监督生成物理意义明确的不确定性图，消除传统置信图与检测结果的耦合；②设计Uncertainty‑Aware Pyramid Fusion (UAPF)模块，结合Uncertainty‑Weighted Downsampling (UWD)与Uncertainty‑Guided Residual Fusion (UGRF)实现高效多尺度融合；③在融合过程中采用残差结构与高斯平滑权重，提升鲁棒性。

**🔧 技术方法**

使用技术包括：BEV编码器、专门的Uncertainty Head、基于平均池化的UWD、基于不确定性权重的UGRF、残差融合与高斯模糊平滑、以及多尺度金字塔融合与深度可微训练。

**📊 数据集**

实验数据集涵盖DAIR‑V2X和V2V4REAL两个真实世界的V2X协同感知数据集。

**📈 对比分析**

与HEAL、CoBEVT、V2X‑VIT等主流方法对比，UECP在AP@30、AP@50、AP@70等多指标上均超过对手，尤其在高阈值下提升显著；并在姿态误差和通信延迟扰动下保持最优性能。

**⚠️ 局限性**

局限性主要体现在：①不确定性图仅基于LiDAR点云密度，可能在摄像头或多模态环境中效果有限；②多尺度金字塔与残差融合虽提升精度，但增加了模型复杂度和推理时延。

---

## 557. Physics-Guided Spatiotemporal State Space Modeling for Lookahead Molten Pool Segmentation in Laser Wire-Feed Welding

**arXiv ID:** 2606.23028 | [PDF](https://arxiv.org/pdf/2606.23028v1)

**作者:** Sen Li `[一作]` (Shanghai Jiao Tong University), Fenggui Lu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 WeldMamba，一种基于物理约束的时空状态空间网络，用来预测激光线材焊接过程中关键孔、焊丝和熔池的未来语义分割，支持控制器提前获取焊接状态。

**💡 创新点**

创新点包括：① 将工艺参数和电信号作为条件向量直接调节视觉特征；② 在补丁级别引入状态空间模型与窗口式通信，保留局部动态；③ 结合时间跨度条件的未来潜在和密集特征预测以及圆形运动先验，提升关键孔预测精度；④ 采用SDF渲染与特征蒸馏等多任务监督，提升几何一致性。

**🔧 技术方法**

核心技术包括：MiT‑B1 视觉编码器 + 物理条件归一化；Patch‑Temporal‑SSM（Mamba‑2 + 滚动窗口）时序建模；Horizon‑Conditioned 未来潜在/密集特征预测；KeyholeMotionHead 圆形运动头；SDFDecoder + 可微渲染；多任务损失（分割、SDF、对齐、蒸馏等）。

**📊 数据集**

使用自建 43 条序列的激光线材焊接数据集，采集同步的灰度顶视焊池图像、七维工艺参数和 10 kHz 电压信号，训练集 34 条序列，验证集 9 条序列，总计 40 719 帧。

**📈 对比分析**

与 SegFormer、CFFM、MRCFA、Mask2Former、DeepLabv3+ 等基线进行模块替换和参数对比。WeldMamba 在 500 ms 预测窗口下取得 74.63 % 的 mIoU（关键孔 46.86 %，焊丝 87.61 %，熔池 89.42 %），显著优于所有对比模型；在更短的 20 ms/100 ms 预测窗口下仍保持 90 % 以上 mIoU。

**⚠️ 局限性**

局限性包括：① 关键孔细小且快速运动，仍难以实现完美分割；② 仅在单一设备与材料上验证，缺乏跨平台泛化评估；③ 未加入不确定性估计，无法在极端传输或喷气干扰时提供置信度；④ 部分辅助头（SDF、渲染、蒸馏）仅在训练阶段使用，部署时需去除，导致模型复杂度仍高。

---

## 558. Machine Translation and Post-Editing: Comparative Evaluation of Different MT Systems and Post-Editor Groups in Specialised Translation

**arXiv ID:** 2606.23002 | [PDF](https://arxiv.org/pdf/2606.23002v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 559. ScalingAttention: Discovering Intrinsic Sparse Attention Topology for Video Diffusion Transformers

**arXiv ID:** 2606.23019 | [PDF](https://arxiv.org/pdf/2606.23019v1)

**作者:** Ruiliang Zhou `[一作]`, Chengru Song `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 ScalingAttention 框架，用训练无关的方式通过静态权重编码的稀疏拓扑和自适应稀疏度调节加速视频扩散 Transformer 的 3D 自注意力

**💡 创新点**

创新点在于发现并利用视频扩散 Transformer 的“Intrinsic Sparse Topology”（静态权重编码的稀疏结构）以及将拓扑发现与稀疏度控制解耦；并设计了硬件友好的块稀疏核（CRM kernel）

**🔧 技术方法**

WEST（Weight‑Encoded Sparse Topology）用于离线提取块级稀疏拓扑，FAST（Fidelity‑Aware Sensitivity Tuning）根据扩散阶段动态调整稀疏度，CRM kernel 实现按位块稀疏注意力计算

**📊 数据集**

使用 Wan2.1（T2V 1.3B、I2V 14B）、HunyuanVideo（480P、720P）以及 PenguinVideo Benchmark 子集进行评估

**📈 对比分析**

与 SVG、SVG2、FastWan、PARO 等基线相比，在相同稀疏密度下提升 PSNR、SSIM、LPIPS，速度提升可达 1.90×（在 Wan2.1‑14B 52.5% 密度下 1.90×，在 HunyuanVideo 55% 密度下 1.73×）

**⚠️ 局限性**

主要限制在于块大小 128×128 的硬件匹配导致极稀疏 (<40%) 时效果受限，以及静态拓扑保留过多非必要 token，可能降低极端稀疏下的性能

---

## 560. Scalable Online Flight Trajectory Optimization via Sequential Quadratic Programming for Urban Air Mobility in Ultra Low-Altitude Airspace

**arXiv ID:** 2606.23008 | [PDF](https://arxiv.org/pdf/2606.23008v1)

**作者:** Josue N. Rivera `[一作]` (Nanyang Technological University), James Wang `[通讯]` (Nanyang Technological University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一套在线、可扩展的城市低空无人机轨迹优化框架LTP，能够在实时条件下联合车辆动力学、操作限制和几何障碍进行安全轨迹规划；

**💡 创新点**

创新点在于将障碍物避免通过实时生成的分离超平面约束嵌入SQP优化；采用可变尺度四叉树分解局部约束，保证计算量仅随局部障碍密度而变；实现了在CPU单机上仅约0.03 s完成规划；

**🔧 技术方法**

核心技术包括：JAX加速的数字双胞胎动力学模型、自动微分+零阶保持离散化、在线SQP求解器（OSQP）、实时超平面生成、可变尺度四叉树空间分解；

**📊 数据集**

使用了五个真实城市环境（Austin、Boston、Orchard Road、Marina Bay、Hong Kong）的OpenStreetMap建筑数据，共1710条航线测试；

**📈 对比分析**

与DDP、iLQR、传统SQP对比：传统SQP成功率100%，清除率95.4%，求解时间约25 ms；iLQR 98.2%/95.4%但求解时间137 ms；LTP加入几何约束后清除率提升至98.9%，求解时间约31 ms；最优组合(LTP+几何+放宽限制)实现100%成功率和100%清除率，且求解时间≈31 ms；

**⚠️ 局限性**

局限性：对高度动态障碍（其他UAV）未做完整处理；需要进一步验证闭环跟踪与安全性；在极高密度场景下仍需评估四叉树分解的最优尺度；

---

## 561. From Point Estimates to Distributions: GMM Pooling for MIL in Preterm Birth Prediction

**arXiv ID:** 2606.23005 | [PDF](https://arxiv.org/pdf/2606.23005v1)

**作者:** Hussain Alasmawi `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Mohammad Yaqub `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

研究将多实例学习与高斯混合模型池化结合，用胎儿超声图像预测早产风险；

**💡 创新点**

提出基于GMM的分布池化模块，通过软责任与实例重要性网络估计实例在混合组件中的分布，并用可学习探针映射生成固定维度袋级嵌入；

**🔧 技术方法**

使用U‑Net编码器配合交叉分割辅助、GMM池化、软责任网络、实例重要性网络、可学习探针，实现端到端训练；

**📊 数据集**

评估数据集包括182名高危孕妇的私有TVUS（每人1-43张图像）和公开淋巴结转移数据集（933/668/736样本）；

**📈 对比分析**

与max/mean/attention/density等池化方法对比，PTB任务PR‑AUC从0.44提升至0.56（与max相当但方差更低），淋巴结任务F1/ROC‑AUC提升至0.91/0.89，MAE 0.18；

**⚠️ 局限性**

需为每个数据集调节混合组件数、探针数和温度等超参数，且验证范围仅限两类任务，缺乏更广泛的泛化验证。

---

## 562. Who Owns the AI Recommendation? A Multi-Industry Empirical Map of Brand Category Ownership Across Large Language Models

**arXiv ID:** 2606.23057 | [PDF](https://arxiv.org/pdf/2606.23057v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 563. Safety in Self-Evolving LLM Agent Systems: Threats, Amplification, and Case Studies

**arXiv ID:** 2606.23075 | [PDF](https://arxiv.org/pdf/2606.23075v1)

**作者:** Ruixiao Lin `[一作]` (Zhejiang University), Shouling Ji `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了自演化LLM代理系统的安全与隐私全面评估框架，并构建MLAS攻击面矩阵。

**💡 创新点**

系统化定义自演化代理的生命周期与模块，揭示自演化带来的持久性、自动传播和向量化攻击特征；首次枚举25个模块‑生命周期交叉点的威胁并归纳七大放大效应。

**🔧 技术方法**

结合理论建模、威胁模型、攻击分类、跨模块攻击面分析，并在两个开源自演化框架上进行对比实验验证。

**📊 数据集**

采用公开自演化框架中的自生成训练数据与评估数据（如Self-Rewarding LM、Voyager等），以及对照的安全测试用例集（共40个攻击场景）。

**📈 对比分析**

通过在两个框架上执行40个攻击案例，测量攻击持久率、扫描器检测率等指标；结果显示自演化路径攻击持续率100%，而现有扫描器仅检测2.5%。

**⚠️ 局限性**

研究仅覆盖单体自演化代理和两种框架，未考虑更广泛的多代理生态与真实世界部署；缺乏可验证的防御策略与长期评估。

---

## 564. VolHuMe: a High-Resolution Large Scale Dataset of Volumetric Human Meshes

**arXiv ID:** 2606.23062 | [PDF](https://arxiv.org/pdf/2606.23062v1)

**作者:** Giulia Martinelli `[一作]` (University of Trento), Nicola Conci `[通讯]` (University of Trento)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `67630363-6be0-4f51-ab05-7198250671a5` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了高质量的4D人体数据集VolHuMe，包含104名受试者的一分钟高分辨率多视角捕捉，并提供完整的多模态注释；

**💡 创新点**

创新点在于采用稀疏近距离多摄像头布置，兼顾大规模与细粒度体部细节，并提出精确半自动SMPL‑X注册流程；

**🔧 技术方法**

使用了Mantis Vision体素捕获系统、深度相机、64 RGB相机、基于NeRF/3D Gaussian Splatting、Radiant Foam等最新三维重建与渲染技术；

**📊 数据集**

数据集为VolHuMe，涵盖高分辨率网格、点云、RGB‑D图像、SMPL‑X拟合、服装分割、手部与面部细节；

**📈 对比分析**

通过对视图合成和4D重建的多种方法（Reality Capture、Metashape、Meshroom、Nerfacto、VolNerfacto、VolSplatfacto、Radiant Foam、GPS‑Gaussian、Animatable Gaussians）进行基准测试，表现出Radiant Foam在视图合成上最优，GPS‑Gaussian在4D重建上优于其他同类方法，指标均在PSNR/SSIM/LPIPS等方面优于先前数据集；

**⚠️ 局限性**

局限性在于仍需对稀疏视角进行背景置换和预处理，且NeRF/GS方法在该设置下难以收敛，说明现有重建管线对稀疏近距离采集仍存在挑战。

---

## 565. Three-Step Hierarchical Transformer for Multi-Pedestrian Trajectory Prediction

**arXiv ID:** 2606.23058 | [PDF](https://arxiv.org/pdf/2606.23058v1)

**作者:** Raphaël Delécluse `[一作]` (University of Lille), Laurent Guimas `[通讯]` (Explain)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了一种三步层级Transformer，分别进行时序编码、多模态融合与社交交互推理，用以预测行人未来轨迹。

**💡 创新点**

创新点在于将时序、模态和社交三大要素分离成独立阶段，使用轻量化GRU摘要和时间-主体注意力，大幅降低计算成本同时保持可解释性。

**🔧 技术方法**

采用Transformer编码器/解码器、GRU摘要、跨模态注意力、时间与主体位置编码、可变模态处理等技术。

**📊 数据集**

在JTA、JRDB以及Pedestrians and Cyclists in Road Traffic三个公开基准上进行实验。

**📈 对比分析**

与多种基准（Trajectory‑only、Social‑LSTM、Transformer、Social‑Pose、EmLoco等）对比，平均位移误差(ADE)与终点误差(FDE)均达到或逼近SOTA，在JTA、JRDB、Urban数据集上分别取得0.96/1.91m、0.33/0.67m、0.51/0.96m的成绩。

**⚠️ 局限性**

限制在于对多模态不确定性处理不足，无法给出概率预测，对极端复杂场景的泛化仍有提升空间。

---

## 566. Unlimited OCR Works

**arXiv ID:** 2606.23050 | [PDF](https://arxiv.org/pdf/2606.23050v1)

**作者:** Youyang Yin `[一作]`, Lei Jia `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Unlimited OCR模型，并引入Reference Sliding Window Attention（R‑SWA）机制，实现在单次前向推理中解析整本书、数十页甚至上百页的OCR任务。

**💡 创新点**

创新点包括：1) R‑SWA在解码时仅让每个token关注所有视觉/提示前缀与最近n个输出token，从而保持KV缓存恒定、降低显存；2) 通过把全注意力替换为R‑SWA，模型在保持甚至提升OCR性能的同时，实现长范围（无限）解析；3) 设计了固定容量的KV缓存队列，避免随输出长度增长的显存溢出。

**🔧 技术方法**

核心技术：DeepEncoder（高压缩率ViT+CLIP‑ViT）、3B MoE LLM解码器、R‑SWA、Flash Attention v3、DeepEP、Megatron‑LM框架、SGLang推理引擎、KV缓存队列管理。

**📊 数据集**

使用约2M PDF文档训练集（9:1单页/多页比例），包含单页Paddle OCR标注与多页合成；评估数据集为OmniDocBench v1.5与v1.6，另外构造自制多页面小说/论文/报告集进行长篇OCR测试。

**📈 对比分析**

与DeepSeek OCR及其他VLM OCR（如OCRFlux、Qwen、Gemini等）在OmniDocBench v1.5/v1.6上对比：整体指标提升约6.2%，文本Edit Distance下降0.035，Table TEDS提升5.96%，TPS提升12.7%；在多页面长程OCR中，Edit Distance低于0.11，Distinct-20/35保持在99%以上，显示出R‑SWA在长序列推理中的稳定性与高效性。

**⚠️ 局限性**

局限性：受预填充长度（如32K）限制，无法实现真正无限长解析；前缀序列随页面数增长导致填充长度显著增加；在高分辨率或细小文字识别上仍有错误；未来需扩展到更长上下文、预填充池和自动获取KV块以模拟翻页。

---

## 567. Have You Ever Seen Them? Entity-level Membership Inference through Interrogating Large Language Models

**arXiv ID:** 2606.23030 | [PDF](https://arxiv.org/pdf/2606.23030v1)

**作者:** Yiran Zhu `[一作]` (Zhejiang University), Ziqi Yang `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在黑盒仅能获得生成文本的场景下，提出基于实体级别的会员推断方法，利用多种采访式提示策略从大型语言模型中抽取实体相关信息并进行二分类推断。

**💡 创新点**

首次将实体概念视为会员目标，构建理论可行性框架，设计五种基于人类采访技巧的静态提示策略，并证明其在标签级别限制下仍能显著识别训练集中的实体。

**🔧 技术方法**

采用提示工程、语义相似度向量（Pairwise Consistency/Sequential Transition）、句子嵌入（all-MiniLM-L6-v2）、随机森林/逻辑回归/多层感知机等分类器，实现对生成文本的无监督判别。

**📊 数据集**

利用从RedPajama‑Data‑1T抓取的维基百科文本构造PERSON与ORG两类实体的成员与非成员数据集，包含20/25条相关实体线索。

**📈 对比分析**

与PETAL、ICP‑MIA等现有标签级别样本推断基线对比，实验显示在OpenLLaMA‑7B上实体级推断的AUC最高可达0.97，平衡准确率提升6%–17%，在多模型与多规模上保持稳健。

**⚠️ 局限性**

受限于仅能查询文本、需要多次提示且对组织实体的准确率相对较低；假设模型不对未见实体产生事实性幻觉，实际环境中可能导致误判；查询成本和对查询预算的依赖是实际应用中的主要局限。

---

## 568. Black-Box Continual Learning for Vision-Language Models

**arXiv ID:** 2606.22999 | [PDF](https://arxiv.org/pdf/2606.22999v1)

**作者:** Yuting Li `[一作]` (Shanghai Jiao Tong University), Weiran Huang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了黑盒持续学习基准Black-CL，并提出BETA方法，仅通过优化文本原型实现对视觉-语言模型的持续学习。

**💡 创新点**

将黑盒持续学习拆解为Semantic Projection Accumulation、Latent Distribution Replay和Test‑Time Prototype Adaptation三模块，采用轻量文本原型、球形GMM重放和测试时原型适配，显著提升黑盒环境下的性能。

**🔧 技术方法**

使用冻结的CLIP等视觉‑语言模型作为前端，仅优化文本原型；利用球形高斯混合模型对视觉特征进行伪样本重放；并在测试时通过置信门控进行一次原型梯度更新。

**📊 数据集**

在十个跨域数据集（Caltech101、OxfordPets、StanfordCars、Flowers102、Food101、FGVCAircraft、SUN397、DTD、EuroSAT、UCF101）以及其他基准上进行评估。

**📈 对比分析**

与现有黑盒调优方法（CBBT、LFA、BlackVIP、LCS）以及白盒持续学习方法（LwF、WiSE‑FT、ZSCL、MoE‑Adapters、Primal‑RAIL、LADA）对比，BETA在Last、Average、Transfer Accuracy方面均领跑或接近白盒方法，同时仅使用0.05M可训练参数，显著低于对手。

**⚠️ 局限性**

受限于预训练VLM输出的质量，球形GMM近似可能不适用于所有特征分布；在极大类别数或极低样本场景下重放效果有限；并且缺乏对动态任务流或在线更新的深入探讨。

---

## 569. Interpretable Probabilistic Medical Image Segmentation via Gaussian Process with Explicit Modelling of Annotation Bias and Variability

**arXiv ID:** 2606.23177 | [PDF](https://arxiv.org/pdf/2606.23177v1)

**作者:** Qi Li `[一作]` (University College London), Yipeng Hu `[通讯]` (University College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文提出了一种基于随机变量高斯过程的可解释概率医学图像分割方法，在logit空间显式建模了注释者的偏差和变异。

**💡 创新点**

创新点在于将注释者的系统偏差和内在变异作为可学习参数直接加入到预测分布中，既能分离图像本身的不确定性，又能对注释者误差进行可解释的灵敏度分析。

**🔧 技术方法**

使用了U‑Net特征提取器、RBF核的SVGP以及逆probit链接函数，整体通过变分推断和梯度下降联合优化。

**📊 数据集**

在临床前列腺TRUS影像数据集（249体积共6644张2D切片）上进行实验，注释来自三名研究员及一名高质量临床标注。

**📈 对比分析**

与单独训练的U‑Net和Pionono进行对比，SVGP在ECE、NLL方面均显著优于两者，Dice和HD95维持与Pionono相近的水平。

**⚠️ 局限性**

局限在于仅对每位注释者学习全局偏差和方差，未考虑空间相关性或类别依赖的误差；此外仅在单一数据集上验证，需进一步扩展到多数据集和多类别分割。

---

## 570. The Language Blind Spot: How Query Language and Brand Recognition Tier Shape AI-Constructed Brand Reputation Across Twelve European Languages

**arXiv ID:** 2606.23165 | [PDF](https://arxiv.org/pdf/2606.23165v1)

**作者:** Dmitrij Żatuchin `[一作]` `[通讯]` (Estonian Entrepreneurship University of Applied Sciences), Dmitrij Żatuchin (Estonian Entrepreneurship University of Applied Sciences)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估十二种语言中大语言模型对66个品牌的声誉输出差异，揭示查询语言对内容、情感和推荐的影响。

**💡 创新点**

首次系统比较多语言、大型LLM生成的品牌声誉，量化语言盲区并发现模型稳定性主要受模型而非语言影响。

**🔧 技术方法**

使用BGE‑M3进行跨语言嵌入相似度计算，XLM‑RoBERTa进行情感评分，并采用GPT‑5.4、Gemini 3.1 Pro、Perplexity Sonar Pro进行grounded查询。

**📊 数据集**

基于66个品牌、11个本土市场、12种语言（涵盖日耳曼语、乌拉尔语、波罗的语、斯拉夫语）共35,640条回应，另有20品牌重复5次的子样本。

**📈 对比分析**

通过平均余弦相似度、情感方差、推荐份额、层次聚类和方差解释率评估结果；发现同族语言相似度更高、情感正负差异显著、当地品牌在本土语言中的推荐份额大幅提升，模型稳定性差异明显。

**⚠️ 局限性**

缺乏人类感知基准、情感模型对长文本偏负、稳定性评估仅限20品牌子样本、手工划分品牌认知层级、未控制媒体覆盖量等限制。

---

## 571. On the Intractability of the Minimum Distance Problem for Regular LDPC Codes

**arXiv ID:** 2606.23161 | [PDF](https://arxiv.org/pdf/2606.23161v1)

**作者:** Chenyuan Jia `[一作]` (Shandong University), Guiying Yan `[通讯]` (Academy of Mathematics and Systems Science, Chinese Academy of Sciences, University of Chinese Academy of Sciences)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

论文研究了低密度奇偶校验（LDPC）码的最小距离问题（MDP），证明在左侧或双侧均匀正则结构下，求解该问题依旧是NP‑完全且在参数化意义上为W[1]‑完整；通过构造精细的度数保持变换，给出了从无约束MDP到J‑左正则、(3,3)正则、(J,K)正则实例的完整归约；

**💡 创新点**

创新点在于提出了一套统一的度数保持变换框架（超边分解、校验节点拆分、受控变量复制），并在每一步构造显式的解空间双射，从而能够追踪最小解的大小并完成对不同正则度数类的完整硬化；

**🔧 技术方法**

使用的核心技术包括：①超边分解（将任意超边拆成大小为3的链），②校验节点拆分（把高度校验拆成度3的链并引入前缀奇偶规则），③受控变量复制（在保持3‑左正则的同时放大特定变量的权重），以及参数化归约和W[1]归约框架；

**📊 数据集**

无实验数据集，本文完全为理论复杂度分析，不涉及具体数据集或实验；

**📈 对比分析**

论文通过理论证明展示了问题在不同正则度数下的时间复杂度边界，未进行实验性能比较，结果表明即使在极其结构化的LDPC Tanner图中，MDP仍然不可多做求解；

**⚠️ 局限性**

局限性：当前结果只覆盖非线性超图模型；对线性（J,K）正则超图的NP‑完全性仍是未解决的猜想；此外，论文没有给出多项式时间近似算法或特定结构化 LDPC 码的实用解法。

---

## 572. Asymmetric physics enables efficient learning in quadrupedal robot swarms

**arXiv ID:** 2606.23153 | [PDF](https://arxiv.org/pdf/2606.23153v1)

**作者:** Yuang Zhang `[一作]` (Shanghai Jiao Tong University), Weiyao Lin `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用不连续的高保真模拟与可微的代理模型相结合的异构物理训练框架，端到端学习了面向深度相机的去中心化四足机器人群体导航与步态控制策略，并在512只仿真机器人和6只真实机器人上实现了零样本迁移

**💡 创新点**

创新点在于将高保真非可微物理仿真与可微代理模型分离，利用代理模型提供平滑梯度加速多机器人协同学习，同时实现了无需通信、无需全局地图的自组织导航行为

**🔧 技术方法**

核心技术包括：基于Isaac Gym的高保真仿真、点质量与刚体代理模型的可微优化、层次化导航与步态政策、基于深度相机的视觉感知、AdamW优化器、BPTT梯度传播

**📊 数据集**

使用Unitree Go2机器人URDF在Isaac Gym中生成的仿真环境，以及在真实实验中收集的森林、桥梁、狭窄通道与杂乱房间等多种场景的实时深度相机数据

**📈 对比分析**

与传统PPO强化学习基线比较，异构物理训练在样本效率、收敛速度、通过率、对速度命令的可控性以及在窄通道中的成功率上均优于PPO；在512机器人仿真中保持高通过率，且在真实机器人上展现出预测避免、右侧让路、停顿等待、墙壁跟随等自组织行为

**⚠️ 局限性**

局限性包括对物理碰撞鲁棒性的不足（在极端拥堵或死胡同场景下易出现滚翻或卡死）、代理模型与真实动力学的不完全匹配导致的迁移误差，以及对RGB语义感知的缺乏，未来需改进碰撞恢复、提升场景泛化和加入显式通信等机制

---

## 573. T-VSS: Test-Time Visual Subspace Steering for Adversarial Robustness of Vision-Language Models

**arXiv ID:** 2606.23132 | [PDF](https://arxiv.org/pdf/2606.23132v1)

**作者:** Jaehyuk Jang `[一作]`, Changick Kim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种轻量级的测试时视觉子空间调度（T-VSS）方法，用于在零样本视觉语言模型下通过直接调节攻击后的视觉特征来提升鲁棒性。

**💡 创新点**

核心创新在于利用多视角残差构建样本特定的低秩子空间，并在该子空间内进行共享、可靠性加权的熵最小化，以实现结构化的特征空间修正。

**🔧 技术方法**

方法结合了残差SVD低秩子空间提取、可靠性加权熵损失优化、以及多视角数据增强，全部在冻结的CLIP视觉编码器特征上进行。

**📊 数据集**

在八个细粒度数据集（Caltech101、Pets、Flower102、Stanford Cars、FGVC Aircraft、DTD、EuroSAT、UCF101）以及ImageNet和四个ImageNet-OOD基准（ImageNet‑A、ImageNet‑V2、ImageNet‑R、ImageNet‑S）上进行评估。

**📈 对比分析**

与现有基准（包括基线CLIP、Ensemble、TPT、C‑TPT、MTA、R‑TPT、TTC、TTP）对比，T‑VSS在所有后端骨干（ResNet‑50、ViT‑B/16、ViT‑L/14）上实现了最优的攻击鲁棒性，同时保持了竞争性的干净准确率，并在推理速度和视角预算上优于前者。

**⚠️ 局限性**

局限性主要是对随机多视角增强的依赖，在防御感知攻击下易被突破；此外，若攻击者直接针对同一期望变换进行优化，鲁棒性会显著下降。

---

## 574. Expert Consensus on Criteria for the Automated Assessment of Laparoscopic Camera Navigation

**arXiv ID:** 2606.23131 | [PDF](https://arxiv.org/pdf/2606.23131v1)

**作者:** Amir Ebrahimzadeh `[一作]` (University Medical Center Göttingen), Jannis Hagenah `[通讯]` (University Medical Center Göttingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了包含14个细化维度的腹腔镜相机导航（LCN）技能分类体系，并通过对23名外科医生的问卷调查确认各维度的临床重要性；随后评估现有计算机视觉（CV）技术在实现这些维度量化测量上的技术成熟度，形成“临床重要性‑技术准备度”矩阵。

**💡 创新点**

创新点在于将临床需求与CV可实现度结合，构建了可直接指导工具开发的优先级矩阵；同时首次系统性把“可测量”与“可自动化”视角融入LCN评估框架，明确指出Field of View、Centering、Focus等高重要性且技术成熟度高的指标可立即实现自动化。

**🔧 技术方法**

主要采用了多目标检测、实例分割、关键点检测、光照/对焦评估、深度估计、SLAM轨迹分析、运动平滑度计算以及光照直方图、镜面检测等CV技术。

**📊 数据集**

使用的公开腹腔镜数据集包括Cholec80、EndoVis、JIGSAWS、Surgical Phase和多模态手术视频数据；对应模型如EndoNet、LoViT、Trans‑SVNet、YOLOv8+ByteTrack、ART‑Net、DRR‑Net、SVM/ResNet 50等。

**📈 对比分析**

对各技术在公开数据集上的性能进行量化：Field of View检测精度93%/33 fps，Centering分割Dice≥96%/45 fps，Focus判别精度≥96%/37 fps；Medium水平的如Magnification/Zoom、Instrument Visibility、Economy of Motion等在70‑90%精度范围；Low水平的如Responsiveness/Anticipation、Collision Avoidance、Contextual Awareness等低于70%或帧率不足30 fps。

**⚠️ 局限性**

局限性包括：样本仅为23名医生，主要来自单一中心，缺乏多中心、多学科验证；技术准备度评估基于文献和专家判断，未在真实手术视频中进行系统实现和性能测试；分类可能未覆盖所有临床重要维度，未来需在更大范围内进一步验证与迭代。

---

## 575. PRIDE: Privileged Information-enhanced Distillation for Empathetic Dialogue Generation

**arXiv ID:** 2606.23124 | [PDF](https://arxiv.org/pdf/2606.23124v1)

**作者:** Jiaqiang Wu `[一作]` (University of Science and Technology of China), Shangfei Wang `[通讯]` (University of Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为PRIDE的偏置信息增强知识蒸馏方法，用于提升同理对话生成模型的情感理解和表述质量。

**💡 创新点**

创新点在于将学习使用特权信息（LUPI）框架引入同理蒸馏：1）通过同理推理提示让教师显式拆解情感与情境；2）使用多源注意力融合对话文本与特权信息；3）采用双重对齐损失（MMD+逆KL）在特征层与输出层同步知识。

**🔧 技术方法**

技术手段包括知识蒸馏、逆KL散度、最大均值差距(MMD)、情感分类器、特权信息丢弃策略以及多源注意力门控机制。

**📊 数据集**

实验使用多模态MEDIC数据集和文本基的EmpatheticDialogues数据集，并在三种开源模型族（Qwen2.5-VL、LLaVA、Gemma3）上进行教师与学生的微调与蒸馏。

**📈 对比分析**

与SFT、SeqKD、SD、KD、无特权信息（w/o P）以及现有SOTA方法（MIME、EmpDG等）进行对比。PRIDE在所有自动评价指标（PPL、Dist、F_BERT、Acc）及GPT‑4o、人工评估上均优于基线，且在部分指标上学生模型甚至超过其大教师。

**⚠️ 局限性**

局限性在于依赖训练阶段可获得的特权信息，若该信息缺失或不充分会显著影响效果；实验仅覆盖同理对话任务，对其他领域的推广尚待验证；对教师模型的质量仍有一定依赖。

---

## 576. Temporal-Spectral Alignment with Frequency Adaptation for Source-Free Time-Series Adaptation

**arXiv ID:** 2606.23120 | [PDF](https://arxiv.org/pdf/2606.23120v1)

**作者:** Shichang Meng `[一作]` (City University of Hong Kong), Linqi Song `[通讯]` (City University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对源免费时间序列域适应，提出通过冻结源模型并在输入信号层面使用频域幅度与相位调节的频率适配层（FAL）实现特征对齐。

**💡 创新点**

创新点在于：① 将域漂移视作频域分布差异，设计可学习的幅度/相位调节；② 采用时间掩蔽+时序重构的源预训练提升时序表示；③ 通过信息最大化与时序输出验证两种自监督目标驱动FAL，无需目标标签。

**🔧 技术方法**

技术手段包括FFT/IFFT频域变换、轻量级MLP幅度/相位扰动网络、信息最大化（熵最小化+多样性最大化）损失、时序输出验证（重构误差）以及1D CNN特征提取器和LSTM时序重构器。

**📊 数据集**

实验数据集：WISDM（人体动作识别）、MFD（机器故障诊断）和Boiler（工业锅炉故障检测）三大时序基准。

**📈 对比分析**

与SHOT、NRC、AaD、MAPU、TemSR、CE‑SFDA等六种基准进行宏观F1（MF1）对比，方法在三组数据集上均取得最优或第二优成绩，平均MF1分别为86.12%、65.40%和64.67%，在大多数跨域任务上显著优于现有最先进方法。

**⚠️ 局限性**

局限性：仅考虑频域分布差异，若域漂移主要源于非频域因素可能表现不佳；依赖FFT/IFFT计算，频域处理成本相对较高；仅在三组数据集验证，泛化到更大范围时需进一步评估；需要一定的超参数调优（幅度/相位尺度）。

---

## 577. LUMINA-26: Low-Light Understanding for Modeling and Interpreting Night-time Actions

**arXiv ID:** 2606.23118 | [PDF](https://arxiv.org/pdf/2606.23118v1)

**作者:** Aman Kumar Pandey `[一作]` (Delhi Technological University), Anil Singh Parihar `[通讯]` (Delhi Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了LUMINA-26低光夜间动作识别数据集，并设计了Illumi-Net（基于混合专家的光照自适应网络）实现低光环境下的动作识别

**💡 创新点**

创新点在于：①真实采集的多样化低光数据集LUMINA-26（26类、6784段、22名受试者、20个场景，类均衡）；②Illumi-Net将光照描述符与混合专家增强和分类相结合，光照信息驱动增强与决策；③使用光照感知的课程学习与一致性正则化提升鲁棒性

**🔧 技术方法**

技术手段包括：光照统计描述符（均值、方差、动态范围、时间梯度）、多专家亮度/对比增强、VideoMAE Transformer特征提取、专家条件分类、时序一致性损失、KL一致性损失、亮度感知课程学习、多剪辑推理

**📊 数据集**

使用数据集：LUMINA-26（自研）、ELLAR（公开低光基准）以及对比的ARID、Dark-48、ELLAR

**📈 对比分析**

在ELLAR上实现Top-1 55.13%、Top-5 78.87%，显著超过之前的DGAM（38.42%/74.44%）；在LUMINA-26上获得Top-1 75.95%、Top-5 93.58%，成为首个基准结果；通过消融实验验证各模块贡献

**⚠️ 局限性**

局限性包括：仍依赖视频级光照描述符可能忽略细粒度光照变化；模型复杂度较高，推理速度受限；在极端光照下仍存在细节模糊、噪声放大等问题；仅在室内/户外夜间场景测试，未评估跨环境泛化

---

## 578. Self-Evolution for Multi-Turn Tool-Calling Agents via Divergence-Point Preference Learning

**arXiv ID:** 2606.23112 | [PDF](https://arxiv.org/pdf/2606.23112v1)

**作者:** Jiaqiang Tang `[一作]` `[通讯]` (Hong Kong University of Science and Technology (Guangzhou)), Jiaqiang Tang (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 ToolGraph 与 DPO 自我演进框架，用于多轮工具使用的内部提升。

**💡 创新点**

将工具图谱与经验边权重结合并在推理与训练中保持上下文一致性；构建基于轨迹分歧点的偏好对，并利用标注筛选。

**🔧 技术方法**

使用 Qwen 3.5‑9B + vLLM 运行，ToolGraph 图结构与最佳‑若干候选工具评分，DPO + QLoRA LoRA 微调，PEFT。

**📊 数据集**

在 tau2‑bench 四个域（航空、零售、电信、银行）共 375 个任务的内部轨迹上进行训练与评估。

**📈 对比分析**

与无 ToolGraph 基线对比，ToolGraph 将平均奖励从 0.304 提升至 0.338，ToolGraph＋DPO 再提升至 0.355；航空与零售提升最显著，电信与银行提升有限。

**⚠️ 局限性**

仅在同一基准内部提升，未评估泛化能力；机制共同开启未拆解；规则手工设定；单一模型规模；缺乏误差区间与多次复现实验。

---

## 579. General-Purpose Nonlinear Function Approximation via Linear Integrated Photonics

**arXiv ID:** 2606.23159 | [PDF](https://arxiv.org/pdf/2606.23159v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 580. Neural Parameter Calibration for Finite-State Mean Field Games

**arXiv ID:** 2606.23155 | [PDF](https://arxiv.org/pdf/2606.23155v1)

**作者:** Anna C. M. Thöni `[一作]`, Mathieu Laurière `[通讯]` (New York University Shanghai)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于神经网络的可微分框架，用宏观人口轨迹数据学习有限状态均衡博弈（MFG）的时空参数路径。

**💡 创新点**

创新点在于利用隐式微分通过博弈的固定点高效反向传播，实现在不观测个体行为的情况下精准校准参数，并提供理论证明与泛化界限。

**🔧 技术方法**

技术包括：前置评估神经网络、Picard迭代求解MFG平衡、隐式微分求梯度、批量子采样和Adam优化。

**📊 数据集**

数据集涵盖四类场景：线性二次模型、网络安全模型、SIR疫情监测数据以及纽约市共享单车实时轨迹。

**📈 对比分析**

与仅用均衡动力学的基线模型对比，MFG模型在合成与真实数据上都实现了更低的均方误差、提前适应政策变化（如站点关闭）并展现出更好的鲁棒性。

**⚠️ 局限性**

局限性包括对游戏结构先验知识的依赖、对离散时间和光滑假设的限制、对全观测人口分布的要求以及对结构变化的适应能力不足。

---

## 581. Synthesizing the Lombard Effect: Multi-Level Control of Speech Clarity and Vocal Effort in TTS

**arXiv ID:** 2606.23176 | [PDF](https://arxiv.org/pdf/2606.23176v1)

**作者:** Seymanur Akti `[一作]` (Karlsruhe Institute of Technology), Alexander Waibel `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计了一个可连续控制发声努力与发音清晰度的多维 Lombard TTS 模型，并支持词级强调。

**💡 创新点**

引入双轴条件注入和流匹配架构，实现发声努力与发音分离控制，并实现词级局部清晰度提升。

**🔧 技术方法**

基于 Matcha‑TTS 的流匹配解码器、双注入的声学与时长控制、Vocos vocoder 等技术。

**📊 数据集**

使用 Expresso 细分口音标签（默认、enunciated、fast、projected）与 LJ Speech 的中性语料。

**📈 对比分析**

与基线信号处理方法对比，客观指标 WER、MVD、Spectral Tilt 及 SII 显示模型在噪声下显著降低 WER、提升 SII，且人类评估显示自然度与可懂度更高。

**⚠️ 局限性**

受限于极高发声努力导致 ASR 分布偏移，token 级控制可能出现跨词泄漏，需要进一步改进评估与控制细粒度。

---

## 582. Same question, different history: language, national identity, and credit in large language models

**arXiv ID:** 2606.23164 | [PDF](https://arxiv.org/pdf/2606.23164v1)

**作者:** William Guey `[一作]` (Tsinghua University), José O. Gomes `[通讯]` (Federal University of Rio de Janeiro)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究系统性评估了11种主流大语言模型在21个争议性发明/发现问题上的回答，探讨语言如何影响模型呈现的发明者身份；

**💡 创新点**

发现模型回答中存在显著的语言条件化效应：使用与争议者相关联的语言提问时，模型更倾向于给出该国主流的发明者名称，尤其是低地位非英语的发明者；

**🔧 技术方法**

采用多模型实验（11种LLM），基于预设的固定问题模板（两种问法）和12种语言；使用自动化模型注释器对回答进行分类，构建逻辑回归与交叉随机效应模型来量化语言、纪念度、权力、回答长度等因素的影响；

**📊 数据集**

数据集为75,896条模型回复，涵盖21个争议事件（如无线电、电话、印刷等），以及12种语言的约1,380个固定问题；每条回复均由两名独立LLM注释器标注；此外收集了每位发明者的纪念度指标（0–8）和争议内部权力排名；

**📈 对比分析**

通过对比“擦除”率（单一答案）与“争议”/“部分”答案，发现整体擦除率仅为4.6%，但在英语主导争议中显著升高；逻辑回归表明语言匹配可使发明者被提及的几率提升约80%（OR≈1.8，95%CI 1.33–2.43），并与纪念度正相关；模型间差异存在但与回答长度相关，控制后仍观察到语言效应；

**⚠️ 局限性**

局限包括：1）使用模型自我注释，可能引入偏差；2）语言变量仅作为“相关国家语言”近似，未能直接区分提问与回答语言；3）争议事件以西方科技为主，非英语语言覆盖有限；4）缺乏对训练语料库语言分布的直接测量；5）模型行为随时间变化，结果可能不稳定。

---

## 583. Movable Antennas for Robust Wireless Sensing via Joint Cramér-Rao Bound and Sidelobe Minimization

**arXiv ID:** 2606.23154 | [PDF](https://arxiv.org/pdf/2606.23154v1)

**作者:** Wenyan Ma `[一作]` (National University of Singapore), Rui Zhang `[通讯]` (National University of Singapore)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了一种基于可移动天线（MA）的无线感知系统，通过优化天线位置来最小化方差下界（CRB）与杂波峰值（MSL）的和，从而降低角度估计均方误差。

**💡 创新点**

创新点在于：① 将CRB与MSL同时纳入优化目标，揭示它们之间的权衡关系；② 提出一种基于连续天线密度的解析解与SCA算法；③ 通过一维线性搜索自动选择最优MSL阈值，实现对不同信噪比的鲁棒感知。

**🔧 技术方法**

采用的主要技术包括：Cramér-Rao bound分析、歧义函数建模、连续天线密度优化、连续凸近似（SCA）求解非凸问题、以及基于Monte‑Carlo的均方误差评估。

**📊 数据集**

实验采用仿真数据：1‑维移动区段长度 A=10λ（λ=0.05 m），16 个可移动天线，最小间距 D=0.5λ，SNR 在 -30 dB 到 +20 dB 之间变化，使用随机均匀角度分布。

**📈 对比分析**

与传统统一线阵（ULA）和最优 CRB 的分裂线阵（SULA）进行比较。结果显示，所提方法在所有 SNR 区间均显著降低 AoA 均方误差，尤其在中低 SNR 时抑制杂波误差，高 SNR 下几乎达到 CRB 限界。

**⚠️ 局限性**

局限性包括：① 仅考虑 1‑维可移动天线，未扩展到多维或 3‑D 运动；② 假设远场 LoS 信道，忽略多径和相位误差；③ 优化需要离线计算，对实时动态场景的适应性有限；④ 物理实现中的机械运动、同步与能耗等实际问题未深入探讨。

---

## 584. ShotcreteDepth: A Bi-modal Dataset for Robust Robotic Depth Perception in Shotcrete Construction Environments

**arXiv ID:** 2606.23152 | [PDF](https://arxiv.org/pdf/2606.23152v1)

**作者:** Jakub Gregorek `[一作]`, Lazaros Nalpantidis `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ShotcreteDepth 双模态数据集（立体 RGB + LiDAR）并发布轻量化点云标注工具，用于射磨环境下的深度感知方法研究与评估。

**💡 创新点**

首个针对高浑浊射磨施工场景的深度感知数据集，结合了稀疏 LiDAR 以及完整立体视觉信息；同时提供了可快速标注尘埃点云的工具。

**🔧 技术方法**

采用 Roboception rc_visard 160c 立体摄像头与 Velodyne PUCK LiDAR；使用 Semi‑Global Matching、Vision Transformer 立体匹配网络、Depth Anything v3、Marigold‑E2E、MoGe‑2 等深度估计/补全模型；点云滤波、同步校准与时序匹配等技术。

**📊 数据集**

ShotcreteDepth 本身（11252 采样，220 标注）以及在实验中对比使用 SceneFlow、Middlebury、KITTI 等公开数据集。

**📈 对比分析**

在 640×480 推理下，分别评估 REL、δ1、RMSE、MAE、PDBE、EPE、D1 等指标。结果显示：深度网络在高浑浊条件下优于传统 SGBM；大模型（如 Vision Transformer 立体匹配、Depth Anything）精度最高，但推理耗时明显；轻量化模型速度快但精度下降。

**⚠️ 局限性**

局限性：仅在单一施工现场收集，环境多样性有限；LiDAR 在高尘埃下性能显著受限；数据集未包含多源传感器融合结果；标注工具仍需人工操作，扩展和规模化标注成本高。

---

## 585. LLM-Aided A* Search in Non-Geometric Network Graphs

**arXiv ID:** 2606.23136 | [PDF](https://arxiv.org/pdf/2606.23136v1)

**作者:** Nouf Alabbasi `[一作]` (Khalifa University), Omar Alhussein `[通讯]` (Khalifa University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对非几何网络图的最短路径问题，提出一种利用大型语言模型（LLM）生成中间路点并引导 A* 搜索的算法。

**💡 创新点**

创新点在于将景点距离（landmark distance）双重使用：一方面作为可接受的 ALT 估计提升 A* 的启发式准确度；另一方面作为结构特征输入给 LLM，恢复其在抽象图中缺失的“距离‑到‑目标”信号，从而实现 LLM 辅助的路径引导。

**🔧 技术方法**

技术手段包括：ALT（landmark）启发式、LLM（GPT‑4.1）生成路点、基于启发式的 A* 搜索、Prompt Engineering（包含 h‑value、CoT、few‑shot 等），以及实验评估框架。

**📊 数据集**

使用的数据集为 SNAP 真实道路网络子图和 Barabási‑Albert (BA) 模型生成的无权图，节点数从 750 到 2000 级，测试 50 条随机源-汇查询。

**📈 对比分析**

与传统 A* 及贪婪最佳优先搜索比较：LLM‑aided A* 在 2000 节点图上平均可减少约 50% 的扩展节点，路径代价增加仅 0.34–0.66；相较于贪婪搜索，虽然探索量相近，但路径代价提升高达 3.92，验证了其在保持近最优解的同时显著降低搜索开销。

**⚠️ 局限性**

局限性包括：仅在无权图实验；LLM 仅提供引导而非完整求解，尚未扩展到加权或资源约束的最短路径；对 LLM 生成路点的质量依赖于 prompt 与结构特征；对极端误导情况的理论子最优性上界尚未给出。

---

## 586. Understanding the (In)Security of Vibe-Coded Applications

**arXiv ID:** 2606.23130 | [PDF](https://arxiv.org/pdf/2606.23130v1)

**作者:** Junquan Deng `[一作]` (Independent Researcher), Ruijie Meng `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建10,517个真实开源的vibe‑coded应用语料库，并随机抽取200个已公开部署的实例，结合多模态LLM审计与人工校验，系统评估了这些应用中的安全漏洞，得到1471个高质量漏洞数据。

**💡 创新点**

创新点：①首次进行大规模、真实环境下的vibe‑coding安全评测；②提出8种跨生命周期的失败模式，揭示漏洞根因；③通过对模型、提示、硬件三种条件的可控实验，评估其对漏洞诱发的抑制效果。

**🔧 技术方法**

主要技术手段包括：多模型代码审计框架（Claude Sonnet 4.6 + GitHub Copilot GPT‑5.3‑Codex），自定义安全技能集，人工验证流程，实验性重放实验以检验漏洞可复现性及对策效果。

**📊 数据集**

数据集：vibe‑coded应用语料库〈10,517 repo〉、已部署应用样本〈200 repo〉以及基于人工审核的漏洞集〈1,471 漏洞〉。

**📈 对比分析**

方法比较：将vibe‑coded漏洞率与传统人类驱动开发（OWASP Top 10）进行对比；在重放实验中，提示工程（生产就绪、自动自检）可将漏洞触发率降低27%；模型升级对大部分漏洞影响有限；人工校验显著降低误报率。

**⚠️ 局限性**

局限性：①样本以Web应用为主，其他领域缺失；②仅覆盖Claude Code、Lovable两款主流代理；③对AI作者标记依赖推断，可能漏判；④抽样仅200个部署实例，覆盖范围有限；⑤仅识别8种失败模式，未覆盖所有潜在原因。

---

## 587. A Dual-Track Framework for Template-Constrained LaTeX Conversion

**arXiv ID:** 2606.23107 | [PDF](https://arxiv.org/pdf/2606.23107v1)

**作者:** Chung Cheuk Hei `[一作]` (Hong Kong University of Science and Technology), Liu Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Dual-Track框架，将Markdown到符合LaTeX模板的转换任务拆分为离线提取模板约束与在线混合执行两条轨道。

**💡 创新点**

创新点在于先通过离线轨道生成可复用的模板约束清单，再在在线轨道中仅将需要语言推理的部分交给LLM，其余使用规则引擎，显著提升了编译可靠性与语义保真。

**🔧 技术方法**

技术方案包括规则引擎、预先生成的模板manifest、受限LLM推理、神经符号混合执行管线。

**📊 数据集**

使用了56篇已发布论文和7种学术会议/期刊的LaTeX模板作为评测数据集。

**📈 对比分析**

与传统Pandoc等规则方法以及全生成式LLM流水线对比，Dual-Track在编译成功率、结构一致性和语义准确性上均优于基线，且生成成本与延迟更低。

**⚠️ 局限性**

局限性包括对模板约束提取的依赖、LLM在极其复杂推理时仍可能出现幻觉，以及实验覆盖的模板样本数量有限，未来需扩展到更多多样化模板。

---

## 588. Poisson2Gaussian: Noise Gaussianization to Enhance Image Denoising

**arXiv ID:** 2606.23098 | [PDF](https://arxiv.org/pdf/2606.23098v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 589. Scene-agnostic ALS boresight self-calibration

**arXiv ID:** 2606.23101 | [PDF](https://arxiv.org/pdf/2606.23101v1)

**作者:** Aurélien Brun `[一作]` (École Polytechnique Fédérale de Lausanne), Jan Skaloud `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于点对点对应的 ALS 自校准方法，分为轻量化的 Gauss‑Helmert (GH) 调整和严格的 Dynamic Network (DN) 调整；

**💡 创新点**

创新点在于消除了传统平面约束的场景依赖，提供两种适用不同惯性导航质量的求解方案，并首次在实际运营飞行中验证两种方法；

**🔧 技术方法**

使用了点云对应提取算法、Gauss‑Helmert 最小二乘模型、Factor‑Graph 的 Dynamic Network 求解框架、GNSS/INS 轨迹优化与激光向量约束；

**📊 数据集**

使用四个运营 ALS 任务（LEG、LAR、HAR、HAL）以及 LAR 任务中的两种惯性系统（导航级 AIRINS 与 UAV 级 APX‑15），共 5 种 IMU 配置；

**📈 对比分析**

通过与传统平面校准以及参考标定结果对比，GH 在轨迹质量足够时可在 1–4 条重叠飞线内恢复角度误差 < 0.1°；DN 在 UAV 级 IMU 时可显著降低误差并提供协方差；两方法在导航/战术级别下表现相近；

**⚠️ 局限性**

限制主要在于：GH 需要轨迹已达到足够的后处理精度；yaw 角度观测性弱，需至少 3–4 条多姿态飞线；在 UAV 级时间相关姿态误差显著时 GH 仅能吸收平均误差；场景若缺乏可被激光回波的几何结构（如水面、湿地）则对应提取失效。

---

## 590. FLFL: Federated Latent Factor Learning for Private Recovery of Spatio-Temporal Signals

**arXiv ID:** 2606.23091 | [PDF](https://arxiv.org/pdf/2606.23091v1)

**作者:** Chengjun Yu `[一作]` (Southwest University), Jia Chen `[通讯]` (Beihang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了FLFL模型，在WSN中通过联邦学习框架实现隐私保护的低秩矩阵恢复。

**💡 创新点**

创新点在于将低秩矩阵分解与联邦学习结合，并将空间和时间相关性作为正则化约束。

**🔧 技术方法**

使用的技术包括低秩矩阵分解、联邦学习、梯度上传、图拉普拉斯正则化以及时间差分正则化。

**📊 数据集**

实验使用了四个真实WSN数据集：北京CO浓度、海面温度、北京PM2.5浓度以及重庆SO2浓度。

**📈 对比分析**

与RFRec、FeSoG、FedMF、FedRec++、MetaMF等八个基线对比，FLFL在MAE和RMSE上均取得最低值，且统计检验显示显著优于其他模型。

**⚠️ 局限性**

局限在于对异常值的鲁棒性仍需提升，并且在极低采样率下性能下降。

---

## 591. Flow as Flow: Modeling Robot Velocity Fields as Probability Velocity Fields for Flow-Based Object Manipulation

**arXiv ID:** 2606.23090 | [PDF](https://arxiv.org/pdf/2606.23090v1)

**作者:** Koki Seno `[一作]` (Keio University), Komei Sugiura `[通讯]` (Keio University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种利用流匹配将机器人速度场建模为概率速度场的框架，用于高效生成符合物理规律的机器人流，并在跨身体表演数据上进行训练。

**💡 创新点**

创新点在于：①直接将机器人速度场视为概率流，避免了稀疏关键点位移的近似；②在流匹配框架下实现自回归生成，显著提升生成速度和质量；③通过跨身体表演（人类与机器人）数据提升零样本泛化能力。

**🔧 技术方法**

主要技术包括流匹配（Conditional Flow Matching）、Diffusion Transformer 与 adaLN-Zero 编码、Diffusion Policy 的动作生成模块，以及自回归 ODE 采样。

**📊 数据集**

使用了 Fractal、Bridge V2、DROID‑100、Fanuc Manipulation 等机器人操纵基准数据集，并从跨身体表演视频（包含人类与多机器人）进行预训练。

**📈 对比分析**

与 FLIP、Im2Flow2Act、GigaWorld‑0‑Video（基于世界模型）以及 Track2Act（目标条件）进行比较；在所有四个基准上均取得最佳 ADE 分数，并实现约 33 倍的生成速度提升；在 260 次真实实验中平均成功率高于基线。

**⚠️ 局限性**

局限性：仅使用二维流，未显式建模深度或末端执行器姿态；假设固定相机，当相机运动时易被全局运动掩盖，限制了对人类摄像机抖动视频的应用。

---

## 592. EML Trees Are Universal Approximators

**arXiv ID:** 2606.23179 | [PDF](https://arxiv.org/pdf/2606.23179v1)

**作者:** Joe Germany `[一作]` (American University of Beirut), Joseph Bakarji `[通讯]` (American University of Beirut)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并证明了基于扩展Exp‑Minus‑Log（EML_θ）函数的树结构能够在Sobolev空间内实现通用逼近，并给出了显式构造与误差上界；

**💡 创新点**

创新点在于：①将EML扩展为六参数可学习单元，解决了常数系数量化困难；②提供了可构造的多项式与平滑分区单元的EML实现；③给出完整的EML树通用逼近定理与误差量化；

**🔧 技术方法**

技术方法包括：EML_θ函数的符号构造、局部多项式逼近、基于tanh的平滑分区单元、深度优先树构造以及梯度下降与L‑BFGS优化的训练框架；

**📊 数据集**

实验数据集为五个一维连续目标函数（x³–x、tanh(2x)、sin x、e^(-x²)、sin(3x)·e^(-x²)/2），采样200点；

**📈 对比分析**

比较方法为多次Adam重启+L‑BFGS的固定30 s计算预算，评估相对RMSE；在深度4时多数目标能实现<1%误差，复杂参数化在高频目标上略优；

**⚠️ 局限性**

局限性包括：仅验证一维无噪声数据，使用softplus平滑log导致与理论不完全一致；模型深度与优化预算权衡，缺乏GPU加速；未对多维或稀疏数据进行评估；

---

## 593. Mass Conservation as an Inductive Bias for Self-Organized Criticality in NCA Reservoirs

**arXiv ID:** 2606.23115 | [PDF](https://arxiv.org/pdf/2606.23115v1)

**作者:** Tong Zhang `[一作]` (Independent Researcher), Stefano Nichele `[通讯]` (Østfold University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在演化神经元细胞自动机（NCA）中加入质量守恒规则，探讨其对自组织临界性（SOC）以及后续记忆、分类和控制任务性能的影响。

**💡 创新点**

首次将MaCE的质量守恒机制应用于可演化NCA，并证明它能作为诱导偏置提升SOC质量、加速进化，同时不牺牲下游任务性能。

**🔧 技术方法**

使用CNN更新的NCA、MaCE质量守恒规则、CMA‑ES进化优化、功率律拟合与KS goodness‑of‑fit检验，以及线性SVM/线性Q网络读取。

**📊 数据集**

使用三类基准数据集：5‑bit序列记忆、MNIST手写数字分类、CartPole‑v1 控制环境。

**📈 对比分析**

对保守与非保守两种变体进行三种种子下的进化速度、SOC GOF通量及三项任务指标比较；保守变体平均训练速度提升1.27×，SOC通量更好（2/3种子全6/6），在所有任务上与非保守版相当或略优（CartPole平均奖励提升）。

**⚠️ 局限性**

局限性包括：种子数仅3个，统计功效有限；CartPole未能达到“解”阈值，可能受架构限制；质量守恒仅施加于可见通道；未对任务性能与临界性共同优化。

---

## 594. Substitution-Based Analysis of Structural Novelty for Generative Models of Materials

**arXiv ID:** 2606.23166 | [PDF](https://arxiv.org/pdf/2606.23166v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 595. Sublinear Time Algorithms for Abelian Group Property Testing

**arXiv ID:** 2606.23162 | [PDF](https://arxiv.org/pdf/2606.23162v1)

**作者:** Nader H. Bshouty `[一作]` `[通讯]` (Technion Israel Institute Of Technology), Nader H. Bshouty (Technion Israel Institute Of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究了在两种模型（部分指定模型 PS 和完全指定模型 FS）下的可交换群性质测试问题，并给出了能够在 O(√|G|+1/ϵ) 时间内完成测试的算法。

**💡 创新点**

创新点在于提出了利用生成元的三角关系构造法和双重随机检验步骤的新思路，突破了之前 O(|G|/ϵ) 的上界，实现了子线性时间复杂度，并给出了对应的时间下界，证明了在某些子类中该上界是紧的。

**🔧 技术方法**

核心技术包括：随机生成元算法、构造三角关系的抽象群 Γ(K,L)、同构映射 Ψ 的构造、生日悖论用于检验注射性、以及随机化的乘法一致性检验；所有步骤均在随机查询模型下实现。

**📊 数据集**

该工作属于理论计算机科学范畴，没有使用真实数据集，所有结果均基于理论分析与对群运算查询模型的假设。

**📈 对比分析**

与之前的 Goldreich–Tauber 等方法相比，新算法在时间复杂度上从 O(|G|/ϵ) 降到 O(√|G|+1/ϵ)，在秩 ≤ k、p-群、向量空间等子类中保持同样的子线性性能；实验或具体数值比较未给出，但理论上已证明该性能提升。

**⚠️ 局限性**

主要局限在于对非可交换群的检测仍需至少 Ω(1/ϵ) 次查询，且在某些子类（如完全指定模型下的特殊群）已证明时间下界为 Ω(|G|^{1/4}) 或 Ω(√|G|)，表明在这些情况下无法进一步降低时间复杂度。

---

## 596. Decomposing Financial Market Dynamics via Mechanism Analysis in an Evolutionary Multi-Agent Simulation

**arXiv ID:** 2606.23158 | [PDF](https://arxiv.org/pdf/2606.23158v1)

**作者:** Zhibao Chen `[一作]` `[通讯]`, Zhibao Chen

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一套可插拔的演化代理人模型（ABM），并通过单机制干预（选择、价格形成、行为偏差、共识拓扑）对市场多重表现（多样性、真实性、脆弱性、循环）进行分解。

**💡 创新点**

首次将选择、微观结构、行为偏差和共识拓扑等机制统一为可插拔设计变量，系统性地揭示每种机制对不同宏观属性的独立影响，并证明选择仅影响多样性与策略循环、微观结构提升真实性、行为偏差增加脆弱性，而共识拓扑几乎无效。

**🔧 技术方法**

使用演化算法（截断、QD/MAP‑Elites、NSGA‑II）、端到端的价格形成模块（可开启/增强式自反性反馈）、行为偏差放大因子以及不同的邻居网络拓扑，结合基于种子配对的统计检验（符号检验、bootstrap CI）。

**📊 数据集**

使用120个行为基因的代理人群体，在三种情景下（随机、2008年危机、2015年杠杆牛市）共20个种子，模拟50代，每代20个交易日。

**📈 对比分析**

通过与默认截断/无自反性/弱偏差/全局网络的对照，采用配对种子比较得到平均Δ、正向种子比例、符号检验p值和95%置信区间。结果显示：QD显著提升多样性和策略循环；自反性反馈显著提升5项stylized事实通过率；强偏差显著提高脆弱性指标；共识拓扑无显著效应。

**⚠️ 局限性**

局限性包括真实性指标受限于历史-基本面代理，缺乏真实订单簿与非线性冲击；脆弱性与循环为代理基因或粗粒度混合指标，未测量实际系统性风险；实验仅在单机制干预下进行，未评估机制交互；并且在部分情景下符号检验不显著。

---

## 597. Assistron: Bayesian Shared Autonomy with Off-the-shelf Vision-Language-Action Models

**arXiv ID:** 2606.23147 | [PDF](https://arxiv.org/pdf/2606.23147v1)

**作者:** Pinhao Song `[一作]` (KU Leuven), Renaud Detry `[通讯]` (KU Leuven)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Assistron共享自主框架，利用冻结的Vision‑Language‑Action (VLA) 模型完成宏观运动，只有在接触丰富的局部交互阶段才请求用户干预，并通过流匹配的后验引导将用户低层控制融合进VLA的动作生成。

**💡 创新点**

创新点包括：① 直接使用冻结的VLA避免再训练或微调，保持其开世界泛化；② 引入基于交互检测的阶段化干预机制，仅在关键失败点向用户请求协助；③ 在干预时通过流匹配后验指导，将用户指令与VLA动作空间一致地融合，避免传统混合控制导致的不稳定。

**🔧 技术方法**

技术手段：VLA模型（π0.5 基于流匹配的动作生成）、Whisper语音识别、ResNet‑18 交互检测器、后验共享控制（π_shared = π_vla + g(a_t,u)）以及交互感知的二值干预指示器 𝕀_int。

**📊 数据集**

数据集：构建了包含日常抓取、放置、插入、关节物体等子任务的多任务场景恢复基准；用于训练交互检测器的12k wrist‑camera帧；实验对比使用纯手柄操作、全自主VLA以及Assistron。

**📈 对比分析**

与纯手柄和全自主VLA比较：Assistron完成率高达91.3%，接近手柄（96.3%）且远高于VLA（13.7%）；完成时间略长但人机负载显著降低；NASA‑TLX评估显示Assistron在心理和身体负担上均显著优于手柄；交互检测准确率81.2%，AP 84.5%；后验融合方法在完成时间和轨迹长度上优于线性混合。

**⚠️ 局限性**

局限性：① 受限于底层VLA的语义和动作能力，严重语义错误无法通过局部干预纠正；② 假设用户指令落在VLA动作分布内，若不一致可能导致冲突；③ 需要准确的交互检测，误报/漏报会影响干预时机。

---

## 598. Rising From the Ashes: How Agentic AI is Unblocking Challenges in Cybersecurity

**arXiv ID:** 2606.23138 | [PDF](https://arxiv.org/pdf/2606.23138v1)

**作者:** Gabriela F. Ciocarlie `[一作]` (Stevens Institute of Technology), Christian Wressnegger `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文通过将开放安全问题与新兴的Agentic AI能力对应，并以16个跨领域案例研究阐释，提出Agentic AI可解锁并规模化众多传统安全技术。

**💡 创新点**

创新点在于将Agentic AI视为统一框架，揭示其在减少人工成本、提升低延迟、扩展稀缺专业知识、实现任务无关化以及破解高复杂度问题方面的潜力，从而重振被忽视的安全策略。

**🔧 技术方法**

核心技术包括自然语言理解、海量信息处理、认知耐力、泛化与可复制性以及多学科知识融合，利用这些能力构建可自动化、可扩展的安全Agent。

**📊 数据集**

本文未使用公开数据集，而是基于文献综述和案例研究进行论证。

**📈 对比分析**

未进行实验比较或量化性能评估，主要通过案例阐释Agentic AI在各安全场景中的可行性与潜在收益。

**⚠️ 局限性**

局限性包括Agent的随机性导致验证与审计难度加大、评估基准难以衡量、Agent自身安全与权限风险、以及技术与系统层面交叉挑战需进一步研究。

---

## 599. Spectral Gating via Damped Oscillations for Adaptive Implicit Neural Representations

**arXiv ID:** 2606.23129 | [PDF](https://arxiv.org/pdf/2606.23129v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 600. The Fractal Neural Operator: Overcoming Spectral Bias in Chaotic Attractors via Prime-Harmonic Weierstrass Encodings

**arXiv ID:** 2606.23123 | [PDF](https://arxiv.org/pdf/2606.23123v1)

**作者:** Kanishk Awadhiya `[一作]` `[通讯]` (Independent Researcher), Kanishk Awadhiya (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究者提出一种新的神经算子——Fractal Neural Operator（FNO），并通过其Prime-Weierstrass编码实现对混沌动力系统的长期预测。

**💡 创新点**

创新点在于将质数频率作为非共振、无限周期的Weierstrass基函数，用来嵌入连续动力系统的状态，从而克服传统网络的谱偏差，逼近奇异吸引子的分形几何。

**🔧 技术方法**

核心技术包括：质数基Weierstrass嵌入层、GRU作为演化核、MLP解码器，以及对频率权重采用粉红噪声初始化与学习可调相位的设计。

**📊 数据集**

主要使用的数据集是Lorenz-63混沌系统的时间序列（25,000步，Δt=0.01），用于训练和测试。

**📈 对比分析**

与基于几何编码（2^k）、随机傅里叶特征、原始坐标以及传统Reservoir Computing等基线相比，FNO在Lyapunov Horizon上从约287步提升至347步，误差降低、方差减小，整体性能提升约2.3倍。

**⚠️ 局限性**

局限性主要体现在仅在单一三维混沌系统上验证，缺乏对更高维或实际物理过程的泛化实验，且对参数调优与计算成本的敏感性尚未充分评估。

---

## 601. A Matter of Time: Towards a General Theory of Agency

**arXiv ID:** 2606.23122 | [PDF](https://arxiv.org/pdf/2606.23122v1)

**作者:** Amahury J. López-Díaz `[一作]` (Binghamton University), Carlos Gershenson `[通讯]` (Binghamton University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于时间参数化的(F, A)-系统的分级组织理论，阐述了从自组织、目标导向到主动性（agency）再到开放式演化（open‑endedness）的连续发展过程，并给出了相应的操作性指标；

**💡 创新点**

创新点包括：①将组织闭合（closure）与时间参数化相结合，生成可记忆、可预测的异步动态贝叶斯网络（ADBN）来刻画自组织的时间依赖性；②将“主动性”定义为内部生成的、可修正的预测调节结构，区别于传统的反馈控制；③构建了从元素反应到具备测量‑控制闭环的多层次组织梯度，并提出了四级阈值（自律、目标导向、主动性、开放式演化）；④提出可操作化的指标（语义闭合指数、测量‑控制互补性、预期调制指数、可操作空间重构速率）来评估生物系统的主动性。

**🔧 技术方法**

使用的技术包括：关系生物学（Rosen–Hofmeyr (F, A)-系统）、过程本体论、物理生物符号学、时间参数化的图模型、异步动态贝叶斯网络、因果推断与信息论指标；

**📊 数据集**

本研究为概念性理论，未使用具体实验数据集；

**📈 对比分析**

因缺乏实证验证，本文未给出与其他方法的数值比较或性能指标；

**⚠️ 局限性**

局限性主要体现在：①理论高度抽象，缺乏直接实验验证；②对复杂生物系统的参数化与建模难度大；③操作性指标虽可计算，但在不同实验体系中的可比性与通用性尚待进一步检验；

---

## 602. Cognitive Digital Twins: Ethical Risks and Governance for AI Systems That Model the Mind

**arXiv ID:** 2606.23094 | [PDF](https://arxiv.org/pdf/2606.23094v1)

**作者:** Vamshi Krishna Bonagiri `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Monojit Choudhury `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文对认知数字双子（CDT）进行定义，提出基于权威、自治、访问与控制、责任与可用性的5A治理框架，并识别CDT特有的风险与治理缺口，进一步给出高风险CDT的治理要求；

**💡 创新点**

创新点在于首次将认知双子定位为治理对象，设计了覆盖认知表征与代理行为的5A框架，并系统化阐述了CDT的多维风险与治理需求；

**🔧 技术方法**

本文主要采用概念性分析、框架构建与案例论证技术，未涉及算法实现；

**📊 数据集**

无；

**📈 对比分析**

无实验对比，无法给出性能指标；

**⚠️ 局限性**

局限在于缺乏实证验证与案例评估，框架仍需在真实应用中进行测试与迭代。

---

## 603. Position: Correct Answer, Wrong Mechanism -- When AI Scientists Defend General Claims Their Own Data Contradicts

**arXiv ID:** 2606.23175 | [PDF](https://arxiv.org/pdf/2606.23175v1)

**作者:** Steven Young Eulig `[一作]` `[通讯]` (Harvard University), Steven Young Eulig (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在仿真环境中评估AI科学家系统的“正确答案、错误机制（CAWM）”现象，提出三轴评估（结果、机制忠实度、认知诚实）并验证轻量化跨域验证协议。

**💡 创新点**

创新点在于首次系统性识别并量化AI科学家在发现任务中得出正确结果却给出错误机制的失效模式；提出基于邻域变化的机制自检（regime‑shift check）与补充重计算检查，实现低成本的机制可信度评估。

**🔧 技术方法**

使用大语言模型（Claude Opus 4.6、Gemini 2.5 Flash/Pro）与Python/ROOT仿真工具链，对任务进行开放式、脚本式、正向控制和偏置先验四种提示，随后通过自定义评估脚本进行机制一致性检验。

**📊 数据集**

数据集为Geant4 Cherenkov光子到达时间的仿真输出，共28个实验（20个原始模型，8个交叉模型），包含不同距离与能量的事件记录。

**📈 对比分析**

对比方法主要是将结果、机制一致性和认知诚实三轴评分。实验显示：在开放式提示下0/5成功发现有效观测量，正向提示下5/5可成功实现；在交叉模型中CAWM出现率为3/8，说明机制失效并非单一模型特有；提出的自检协议能检测到所有CAWM案例，且仅需一次额外仿真或重计算。

**⚠️ 局限性**

局限包括：仅在单一仿真领域验证，样本量有限（每提示5次）；跨模型测试仅覆盖两种LLM；评估仅基于最终报告的显式推理，未检验内部推理过程；多智能体系统行为未涉及。

---

## 604. MambaADv2: Evolving Duality-enhanced State Space Model for Unsupervised Anomaly Detection

**arXiv ID:** 2606.23126 | [PDF](https://arxiv.org/pdf/2606.23126v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 605. Performance Evaluation of Selection Strategies for Inter-Satellite Paths in Walker-Delta Constellations

**arXiv ID:** 2606.23135 | [PDF](https://arxiv.org/pdf/2606.23135v1)

**作者:** Marvin Felix Braun `[一作]` (University of Tübingen), Michael Menth `[通讯]` (University of Tübingen)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了LEO星座中卫星间路径选择策略对路径长度、跳数、路径更改率和使用链路率的影响，并评估了三种启发式路径选择器。

**💡 创新点**

创新点在于提出了基于路径剩余可用时间和链路使用率的路径选择策略，并系统比较其对路由稳定性与延迟的权衡。

**🔧 技术方法**

采用Walker‑Delta星座模型、Yen算法生成候选路径、SGP4轨道预测、基于欧氏距离的路径成本以及离散时间仿真等技术。

**📊 数据集**

使用1156颗卫星的Walker‑Delta星座配置，495个随机生成的用户终端‑网关位置对，利用TLE和Skyfield进行轨道传播。

**📈 对比分析**

通过四项指标（平均欧氏路径长度、跳数、路径变更率、使用链路率）在同一候选路径集合上比较三种策略，结果显示最长寿命策略显著降低路径变更率（0.71/min），最少链路策略将链路使用率降至2.24新链路/分，三者的路径长度差距仅约5%。

**⚠️ 局限性**

局限性包括仅使用欧氏距离和跳数作为延迟代理，未考虑处理、排队等真实网络延迟；使用固定的候选路径生成（k=10）和统一的地面位置网格，未覆盖实际稀疏部署和流量负载影响。

---

## 606. Managing Procedural Memory in LLM Agents: Control, Adaptation, and Evaluation

**arXiv ID:** 2606.23127 | [PDF](https://arxiv.org/pdf/2606.23127v1)

**作者:** Julia Belikova `[一作]`, Maksim Makarenko `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个面向企业工作的程序记忆基准（AFTER），包含382个现实任务、6个专业角色和22个可执行技能，并通过该基准评估技能在任务、角色和模型间的迁移与泛化。

**💡 创新点**

创新点在于将技能迁移与泛化作为评估核心，提供了受控的跨任务/跨角色/跨模型转移拆分；同时引入多模型经验的技能演化，证明多源经验能显著提升可迁移性。

**🔧 技术方法**

主要技术包括LLM驱动的技能演化框架（反射、蒸馏、学习写入策略），多模型执行轨迹收集，以及统一的评估接口和版本化技能存储。

**📊 数据集**

使用的数据集来源于现有多种工业任务基准（如SkillsBench、SWE-bench、MLE-bench等）以及作者自行设计的新任务，所有任务均经专业审查与自动验证。

**📈 对比分析**

通过对比无技能、手工技能和LLM生成技能的静态性能、单轮优化提升以及跨模型、跨角色迁移实验，发现技能能在平均提升约5.2分的同时，跨模型泛化率可达73.1%，但跨角色迁移往往出现性能下降。

**⚠️ 局限性**

局限性包括：仅覆盖技术行业角色，可能缺乏医疗、法律等领域；任务来源主要为作者实践，未覆盖所有行业；评估仅基于自动化测试，未衡量代码可读性、鲁棒性等软指标；以及对较新模型和记忆框架的覆盖不完整。

---

## 607. Compression and Retrieval: Implicit Memory Retrieval for Video World Models

**arXiv ID:** 2606.23105 | [PDF](https://arxiv.org/pdf/2606.23105v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 608. Enormous Fluid Antenna Systems (E-FAS) for Wireless Sensing: Channel Modeling and Conditional Estimation Limits

**arXiv ID:** 2606.23119 | [PDF](https://arxiv.org/pdf/2606.23119v1)

**作者:** Farshad Rostami Ghadi `[一作]`, Hyundong Shin `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出基于巨大流体天线系统（E‑FAS）的双向感知通道模型，并推导条件 Fisher 信息与 CRB 来量化角度估计极限；

**💡 创新点**

首次揭示 SW 路由增益与感知多样性之间的根本权衡，以及 E‑FAS 在角度估计上可实现的显著性能提升；

**🔧 技术方法**

采用基于复高斯观测的 Fisher 信息理论、导向向量模型、路由相关矩阵分析与蒙特卡罗仿真；

**📊 数据集**

使用仿真场景（28 GHz、8×8 天线、M=16 路由点、不同功率预算和相关性），并通过 Monte Carlo 计算 RMSE 验证 CRB；

**📈 对比分析**

与传统基站阵列、紧凑型 RIS 及 FAS 体系进行对比，结果表明在相同发射功率下，E‑FAS 的角度 CRB 明显低于其他方案；

**⚠️ 局限性**

受限于路由配置的复杂性、路由相关性导致的感知维度坍塌以及功率分配导致的增益-多样性权衡，系统设计需在这些因素间进行优化。

---

## 609. TTFT-Aware Graph Chain-of-Thought:Distance-Indexed Neural A* for Low-Hallucination Multi-Hop Medical Reasoning

**arXiv ID:** 2606.23108 | [PDF](https://arxiv.org/pdf/2606.23108v1)

**作者:** Bechir Dardouri `[一作]` (Tanit Healthcare Technologies), Yassine Msaddak `[通讯]` (Tanit Healthcare Technologies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出一种基于图知识图谱的可解释多跳检索框架，能够在实时交互场景下生成低幻觉、可审计的医学答案。

**💡 创新点**

核心创新在于将精确距离检索（有向Pruned Landmark Labeling）与神经引导搜索（AStarNet）相结合，形成严格的“可行性走廊”内的高效导航；并通过TTFT感知的证据压缩与路径引用机制显著提升交互速度与可信度。

**🔧 技术方法**

采用的技术包括：1）有向PLL距离索引；2）AStarNet学习式优先级函数；3）临床实体识别与UMLS链接（MedCAT）；4）多类型关系约束与简单路径枚举；5）MMR多样性打分；6）ID‑centric证据打包与路径引用校验。

**📊 数据集**

实验使用约70万节点的异构医学知识图谱（PrimeKG/生育助手KG）以及从临床文本中抽取的生育相关查询集合，所有实体通过UMLS标准化。

**📈 对比分析**

与Text‑RAG、BFS、Bi‑BFS、PLL‑only和AStarNet‑only等基线比较，混合方法在相同的延迟/内存预算下实现了更高的recall@k、TTFT中位数和p95尾值显著下降，并将幻觉率从22.7%降至6.3%。

**⚠️ 局限性**

主要局限包括：1）过度关注最短距离可能忽略略长但重要链条；2）PLL标签占用数百MB，需定期重建；3）AStarNet可能对频繁关系或中心节点偏倚；4）实体链接误差会影响检索质量；5）在高峰负载下仍需退回PLL-only以维持稳定性。

---

## 610. Bridging Semantics and Kinematics: A Modular Framework for Zero-Shot Robotic Manipulation

**arXiv ID:** 2606.23157 | [PDF](https://arxiv.org/pdf/2606.23157v1)

**作者:** Ali Alabbas `[一作]` (Atlantic Technological University), Philip Long `[通讯]` (Atlantic Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种模块化、训练免费、零样本的语言指导机器人操作框架，能在半结构化环境中将自然语言指令自动翻译为可执行的机器人运动序列并完成任务。

**💡 创新点**

1) 将视觉感知、语义解读和运动执行拆分为三个独立模块；2) 采用FastSAM+Set-of-Mark视觉标记实现高精度、可变的视觉锚点；3) 通过LLM做语义路由，将人类指令映射为YAML配置，再动态编译为MoveIt Task Constructor（MTC）管线；4) 通过本地8B Molmo2实现全链路无云、零样本推理。

**🔧 技术方法**

FastSAM实例分割、Set-of-Mark视觉提示、8B Molmo2 LLM、YAML驱动的任务DSL、MoveIt Task Constructor、ROS2、UR5e机械臂、Robotiq手爪、Intel RealSense D435相机。

**📊 数据集**

无公开专门数据集，实验使用真实桌面场景中的一般物体（盒子、螺丝刀、任务板、铅笔、圆盘、孔板等）以及人工生成的3×3/5×5/圆形孔板布局，随机布置并记录。

**📈 对比分析**

对比基线为无视觉标记的直接Molmo2推理，分别在开放世界序列操作和密集空间推理两种任务上评估。开放世界操作：整体成功率62%，Perception 76.7%、Translation 91.3%、Execution 100%；密集空间推理：整体成功率53.3%，全部通过执行。与基线相比，加入SoM视觉标记后成功率从60%提升到80%（开放世界）和从0%提升到100%（密集孔板）。

**⚠️ 局限性**

1) 语言先验偶尔覆盖视觉信息导致幻觉帧；2) FastSAM在部分对整体、相似物体分割时偶有误识；3) 仅单视角2D分割，难以处理深度遮挡；4) 运动原语静态，缺乏实时感知反馈导致失败重规划受限。

---

## 611. Weighted Score-Oriented Losses for Temporally Localized Event Prediction

**arXiv ID:** 2606.23145 | [PDF](https://arxiv.org/pdf/2606.23145v1)

**作者:** Edoardo Legnaro `[一作]` (University of Genoa), Francesco Marchetti `[通讯]` (University of Padua)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一种基于时间加权的score-oriented loss，用于在训练时直接考虑警报与事件的时间关系，从而提升时间序列事件检测的实际效用。

**💡 创新点**

创新点在于将时间权重融入score-oriented loss，利用可微分的期望混淆矩阵表达式，使训练目标与评估指标（如局部窗口的警报奖励）保持一致，解决传统交叉熵与事件级评价之间的“score–loss mismatch”。

**🔧 技术方法**

使用的技术包括可微分的score-oriented loss（基于期望混淆矩阵）、时间权重修正（未来事件接近度与先前警报纠正）、随机阈值分布、Temporal Convolutional Network (TCN) 架构以及多种平衡指标（balanced accuracy、true skill statistic、F1等）。

**📊 数据集**

实验使用了三个公开的时间序列基准：NAB（流式异常检测）、SKAB（工业变更点检测）和Exathlon（Spark应用异常检测），三者对评估指标的时序要求各不相同。

**📈 对比分析**

在相同模型与数据管线下对比交叉熵、无时间权重的score-oriented loss和带时间权重的loss，结果显示在局部奖励结构的NAB和SKAB上，平均提升约+1.6和+3.6分；在标签已覆盖长区间的Exathlon上，时间权重未带来显著改进。

**⚠️ 局限性**

局限性包括：需要手工设计或挑选时间权重向量，且仅在评估指标强调局部时序奖励时有效；对标签已本身包含时间窗口的任务收益有限；实验未探索网络架构改动，结果仅表明loss层面优势。

---

## 612. Technical Report for the ICRA 2026 GOOSE 2D Fine-Grained Semantic Segmentation Challenge: Pretraining-Diverse Ensemble of Foundation Vision Encoders for Robust Outdoor Scene Understanding

**arXiv ID:** 2606.23113 | [PDF](https://arxiv.org/pdf/2606.23113v1)

**作者:** Boyan Wang `[一作]` (Hefei University of Technology), Zhun Zhong `[通讯]` (Hefei University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对ICRA 2026 GOOSE 2D细粒度语义分割挑战，作者提出了一个基于大规模预训练视觉编码器（DINOv3、SigLIP2、InternImage）与Mask2Former解码器的端到端模型，并通过长时间训练、EMA、增大 crop 以及多尺度+翻转 TTA 进一步提升性能。

**💡 创新点**

创新点在于：①将不同预训练目标的基础模型（自监督、视觉‑语言对比、全监督）构成多样化集成，通过 per‑class IoU 权重实现类别级融合；②在保持相同解码器和计算预算的条件下，系统性证明预训练方法对细粒度越野分割性能的主导作用；③提供了一套高效的训练与推理脚本，展示了大 crop 与 EMA 对长尾类别学习的显著好处。

**🔧 技术方法**

使用技术包括：Vision Transformer（ViT-H+）/Swin/ConvNeXt 等主干，Mask2Former mask‑classification 解码器，SimpleFPN 颈部，AdamW + EMA，随机缩放、裁剪、水平翻转、光照扰动等数据增强，滑动窗口推理，多尺度与翻转 TTA，按类 IoU 加权融合。

**📊 数据集**

数据集为 GOOSE 2D Fine‑Grained Semantic Segmentation，包含四个平台（ALICE、MuCAR‑3、Spot v1、Spot v2）的 11,234 张训练图和 1,369 张验证图，测试集采用官方 leaderboard。

**📈 对比分析**

通过对比实验，作者在固定解码器和算力预算的前提下，逐步加入训练技巧：crop 768×768 + EMA + TTA，从基线 72.5% 提升至单模型 76.13% mIoU；最终多样化集成在官方测试集上获得 75.40% composite mIoU，排名第二；单模型最佳 75.12%，相较于官方基线 29.22% 提升显著。

**⚠️ 局限性**

限制主要体现在：①集成增益有限，仅提升约 0.3 点；②对颜色归一化的尝试无效，表明预训练模型已对跨平台颜色变化具备鲁棒性；③实验仅覆盖三种预训练策略，未探讨更大规模或更高分辨率的模型；④长时间训练和 EMA 需要较高算力，对资源受限的部署不友好。

---

## 613. ReNIO: Reweighting Negative Trajectory Importance for LLM On-Policy Distillation

**arXiv ID:** 2606.23104 | [PDF](https://arxiv.org/pdf/2606.23104v1)

**作者:** Chen Lin `[一作]` (East China Normal University), Wei Zhang `[通讯]` (East China Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大语言模型推理任务中的 on‑policy distillation（OPD）与自我蒸馏（OPSD），发现错误的学生生成输出（SGO）比正确的更有用，并提出只基于前缀可计算的重加权方法 ReNIO；

**💡 创新点**

创新点在于利用学生与教师的前缀条件概率比（log‑ratio）识别关键决策点，仅通过前缀信息即可估计样本重要性，无需最终答案，进而对样本进行几何平均聚合和批归一化的重加权；

**🔧 技术方法**

技术包括 OPD/OPSD 框架、前缀条件下的学生‑教师比值、阈值关键 token 选择、几何平均聚合、批归一化，以及将该权重嵌入 OPD/OPSD 目标；

**📊 数据集**

数据集为数学推理任务（AIME2024、AIME2025、HMMT2025）和代码生成任务（HumanEval+、MBPP+），使用 Qwen3 系列与 DeepSeek‑R1 系列模型及对应教师模型；

**📈 对比分析**

与基线 OPD/OPSD 以及 GRPO 对比，ReNIO 在数学推理上对 Qwen3‑1.7B 最高提升 8.90%，对 R1‑Distill‑Qwen‑7B 最高提升 10%，在代码生成任务也取得显著提升；短前缀训练成本保持低且效率提升；

**⚠️ 局限性**

局限性包括未在更大规模模型上验证、对教师模型的依赖、阈值和归一化等超参数需要手动调优，以及仅适用于可获得教师分布的场景。

---

## 614. Minimax Quantile Lower Bounds for Interactive Statistical Decision Making with Privacy

**arXiv ID:** 2606.23096 | [PDF](https://arxiv.org/pdf/2606.23096v1)

**作者:** Raghav Bongole `[一作]` (KTH Royal Institute of Technology), Mikael Skoglund `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一套δ显式的极小化-分位数理论，用以评估交互式统计决策（ISDM）中置信度下的最优性能，并将其扩展到互信息（MI）隐私约束下；

**💡 创新点**

创新点在于：①将分位数视角引入交互式决策；②给出高概率Fano和Le Cam方法的δ显式版本；③把MI隐私转化为受限决策类，统一处理隐私与交互；④为高斯均值估计与高斯多臂赌博机给出明确的量化分位数下界；

**🔧 技术方法**

核心技术包括：信息论逆推（Fano、Le Cam）、极小化-分位数转换、两点模板与隐私噪声阈值分析；

**📊 数据集**

本研究基于理论模型，无实测数据集，所有结论均来自数学证明；

**📈 对比分析**

与现有的期望或弱尾下界相比，本文的下界直接给出置信度依赖的量化表现，匹配非交互式均值估计的log(1/δ)/n以及两臂赌博机的√(T log(1/δ))，并揭示隐私导致的方差膨胀因子；

**⚠️ 局限性**

局限性在于：仅考虑MI隐私与高斯隐私机制；缺乏对其他隐私模型（如DP、LDP）的扩展；尚未给出对应的上界算法；以及对更复杂交互学习任务（多臂拉格朗日、强化学习）的理论完整性待完善。

---

## 615. FlowTrain: Flow-Based Decoupled Training for Industrial-Grade Vision-Language Models

**arXiv ID:** 2606.23087 | [PDF](https://arxiv.org/pdf/2606.23087v1)

**作者:** Zhida Jiang `[一作]` (JD.com), Ke Zhang `[通讯]` (JD.com)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了FlowTrain，一套支持视觉‑语言模型（VLM）分布式训练的框架，通过统一内存池、异构并行分配器和动态打包调度实现解耦执行。

**💡 创新点**

核心创新在于将传统批级同步的多模训练改造为生产者‑消费者数据流，利用统一虚拟地址池、流匹配并行分配和基于实际计算成本的微批打包三大技术实现资源利用最大化。

**🔧 技术方法**

主要技术手段包括全局虚拟地址（GVA）共享内存、Ray分布式调度、混合整数线性规划（MILP）求解并行策略、分段树最佳匹配算法以及zig‑zag负载均衡调度。

**📊 数据集**

实验数据集涵盖公开的InfoVQA视觉问答数据集以及电商平台收集的工业多模态数据集。

**📈 对比分析**

与Megatron‑LM和DistTrain两大基线对比，FlowTrain在三种规模模型上实现了最高1.7倍的吞吐量提升，MFU突破50%，并在不同规模集群上显著缩小与单模LLM训练的效率差距。

**⚠️ 局限性**

局限性包括：在极长序列长度下动态打包的调度开销仍不小；统一内存池的实现复杂且对高速互连依赖较强；对硬件特性的适配仍需进一步优化。

---

## 616. PIVOTSBench: Evaluating Fine-Grained Interpersonal Relationship Reasoning in Multimodal Large Language Models

**arXiv ID:** 2606.23092 | [PDF](https://arxiv.org/pdf/2606.23092v1)

**作者:** Shuxiang Zhang `[一作]` (Sun Yat-sen University), Miao Liu `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PIVOTS基准，用于评估多模态大语言模型在细粒度人际关系推理上的表现；

**💡 创新点**

首次构建双向人际关系维度评分任务并加入关键帧识别与视觉因果分析辅助任务，强调视觉线索在社交推理中的重要性；

**🔧 技术方法**

使用多模态输入（视频、音频、对话文本）和大语言模型（Gemini‑2.5‑pro、GPT‑5、Qwen‑3系列）进行推理，结合提示工程（多阶段与情境提示）；

**📊 数据集**

基于Social‑IQ 2.0与YouTube视频的 595+170 例子，覆盖六维人际关系维度；

**📈 对比分析**

通过与多种模型和输入配置对比，发现专有模型优于开源模型，视觉与音频对不同数据集的提升不同，双向/联合预测在不同场景表现差异；总体性能仍低于人类水平，尤其是情绪与客观维度；

**⚠️ 局限性**

受限于单语种、文化偏差、缺乏跨语言验证以及未探索视觉指令调优或RL方法，且对数据隐私与伦理使用仍需谨慎。

---

## 617. Koshur Pixel: a large-scale synthetic ocr dataset for kashmiri

**arXiv ID:** 2606.23144 | [PDF](https://arxiv.org/pdf/2606.23144v1)

**作者:** Haq Nawaz Malik `[一作]`, Nahfid Nissar `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对克什米尔语Nastaliq书写系统，构建并公开了大规模合成OCR数据集Koshur Pixel，包含613,078幅高质量图像-文本对，覆盖单词、句子、段落及整页等四个细粒度。

**💡 创新点**

创新点在于：①首次针对低资源、视觉极其复杂的Nastaliq脚本系统化生成合成数据；②采用完全基于浏览器的SynthOCR-Gen渲染引擎，利用浏览器原生文本排版与OpenType特性，实现准确的字形、连字与右至左排版；③引入25+种逼真视觉增强技术，显著缩小“合成‑实景”差距；④通过多字体渲染和多层级细粒度，提升模型对字体、排版、语境的泛化能力。

**🔧 技术方法**

技术主要包括：1) SynthOCR-Gen客户端渲染管线；2) KS-PRET-5M文本语料清洗与标准化；3) OpenType GSUB/GPOS与右至左文本处理；4) 多字体渲染（Gulmarg Nastaleeq、Afan Koshur Naksh）；5) 25+几何、光度、噪声与文档退化增强；6) 细粒度标注与多级数据集组织。

**📊 数据集**

使用的基础数据集为KS‑PRET‑5M（约5.09M词汇、295,433词形），该语料被清洗并规范化后作为合成图像的文字来源。

**📈 对比分析**

作者将Koshur Pixel用于微调现有Vision‑Language OCR模型（如TrOCR、Donut、PaliGemma等），实验表明模型在克什米尔语文本的识别精度显著提升，尤其在句子级和整页级任务中达到接近或超过90%的字符/行准确率（具体数值在论文实验部分给出）。与传统手工标注数据相比，合成数据在训练速度和样本规模上优势明显。

**⚠️ 局限性**

局限性包括：①合成与真实扫描文档之间仍存在细微差距，尤其在手写体、极端古籍排版、污损、纸张纹理等方面；②字体数量有限，难以覆盖所有实际印刷/书写变体；③训练高性能模型仍需要昂贵GPU，导致“计算壁垒”；④对多脚本（Devanagari、Sharada等）识别尚未覆盖，未来需跨脚本迁移。

---

## 618. Towards Root Memories: Benchmarking and Enhancing Implicit Logical Memory Retrieval for Personalized LLMs

**arXiv ID:** 2606.23283 | [PDF](https://arxiv.org/pdf/2606.23283v1)

**作者:** Hongxun Ding `[一作]` (University of Science and Technology of China), Xiangnan He `[通讯]` (University of Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了IMLogic基准，用于评估在长对话场景下隐式逻辑记忆检索问题，并开发了RootMem框架，通过构建结构化的根记忆并在推理时路由相关单元，补充语义检索的不足，提升个性化LLM的回答质量。

**💡 创新点**

创新点在于①首次针对隐式逻辑记忆检索设计高质量基准IMLogic，明确区分语义相似但逻辑不相关与语义遥远但逻辑关键的记忆；②提出根记忆概念，将用户长期历史中的可重用决策逻辑抽象为结构化单元；③构建RootMem插件式框架，利用LLM路由器激活根记忆，补足语义检索的逻辑缺口。

**🔧 技术方法**

技术主要包括：LLM驱动的逻辑记忆对挖掘与生成（使用Gemini-3-Pro等）；Generator‑Judger‑Refiner协同生成高质量QA对；根记忆构造与更新的LLM提取器；基于执行规则的根记忆路由器；多模语义检索（BM25、MiniLM、Text-Embedding-3、Qwen3-Embedding-4B等）与图/混合检索模型。

**📊 数据集**

使用的主要数据集是来自HaluMem的真实长对话与记忆记录，构建了20条长对话、约15k条用户记忆和2216个隐式逻辑QA对的IMLogic基准。

**📈 对比分析**

在多种检索基线（词法、语义、图、混合）下，RootMem在记忆级别的MCQ任务中获得55.23%准确率，比最强基线提升27.23%；在对话级别检索中平均提升26.36%，所有系统准确率均突破50%；在开放式生成任务和不同LLM骨干上亦保持显著提升，显示RootMem具有广泛的适用性和显著的性能提升。

**⚠️ 局限性**

局限性包括：仅覆盖文本长对话场景，未涉及多模态记忆；根记忆离线构造效率仍有提升空间；基准只包含推荐、建议、聊天三类查询，未覆盖更广泛的个性化交互、多语言或多样化用户场景；在实际部署时可能存在隐私泄露、过时记忆误用等安全风险。

---

## 619. Safe Few-Step Generation via Velocity Editing

**arXiv ID:** 2606.23267 | [PDF](https://arxiv.org/pdf/2606.23267v1)

**作者:** Yujin Choi `[一作]` (Nanyang Technological University), Jaehong Yoon `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对极少步流匹配文本到图像模型，提出无训练概念去除方法；

**💡 创新点**

创新在于直接编辑流匹配的速度场到安全条件后验，而非逐步轨迹或嵌入改动，并加入风险评分过滤与加速版本；

**🔧 技术方法**

使用流匹配的速度场估计、贝叶斯推导的安全梯度、CLIP/Nudity 评分器以及 MeanFlow 的平均速度概念；

**📊 数据集**

评测数据集包括 Ring-A-Bell、MMA-Diffusion（性/暴力）以及 MS‑COCO 用于生成质量评估；

**📈 对比分析**

与 SGF、STG、SAFREE、Semantic Surgery 等训练‑free 基线对比，取得 ASR 与 TR 最高，安全率提升至 6‑7% 并保持 FID 与 CLIP 与基线相当；

**⚠️ 局限性**

局限在于对 t/(1‑t) 的依赖导致需手工设 t_max；对不同评分器的鲁棒性仍待进一步验证。

---

## 620. Solving Approximate Agreement on continuous and discrete spaces

**arXiv ID:** 2606.23260 | [PDF](https://arxiv.org/pdf/2606.23260v1)

**作者:** Augustin Albert `[一作]` (Institut Polytechnique de Paris), Sergio Rajsbaum `[通讯]` (UNAM)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文研究了在异步读写共享内存模型下，针对多维输入空间和图形结构的近似一致性问题，并给出了可解性条件。

**💡 创新点**

创新点在于：①提供了对“simplex agreement”的完整拓扑可解性判定；②引入CUB（Convexly Uniquely Bicombable）空间作为近似一致性的通用连续框架；③给出了可在CUB空间上实现的显式协议，进而在可坍塌的 simplicial 复形上得到新的近似一致性方案。

**🔧 技术方法**

采用了代数拓扑（连通性、同调、壳化等理论）、几何建模（CAT(0) 与 CUB 结构）、迭代即时快照协议以及凸组合与投影技术来构造和证明协议。

**📊 数据集**

本文未使用实验数据或特定数据集，全部结果为理论证明与算法构造。

**📈 对比分析**

方法通过构造连续映射与拓扑判定实现可解性；相较于以往仅有的图形特定判据，本文提供了更一般、更严格的可解性条件，证明在满足 CUB 或可坍塌条件时协议总能完成。

**⚠️ 局限性**

局限性包括：①仅在完全无错误的等待自由模型下证明；②对 3 进程或更多进程的可解性判定仍为不可判定；③CUB 结构与可坍塌条件并非所有输入空间满足，导致部分实际应用场景无法直接套用；④弱有效性（weak validity）下的结果仍不完整，存在与经典 2‑set agreement 的反例。

---

## 621. When Suspicion Becomes Detection. Folk Deception Cues and Detection Strategies in Online Dating Romance Scams

**arXiv ID:** 2606.23241 | [PDF](https://arxiv.org/pdf/2606.23241v1)

**作者:** Sima Amirkhani `[一作]` (University of Siegen), Douglas Zytko `[通讯]` (University of Michigan-Flint)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对24名伊朗在线约会诈骗受害者进行深入访谈，分析他们识别、怀疑、调查及应对爱情诈骗的心理与行为过程。

**💡 创新点**

聚焦非西方、受法律与社会约束严苛的伊朗情境，揭示文化与法律如何塑造受害者的怀疑认知、情感投入与侦查策略；提出受害者主导的“协作式认知劳动”模型，并呼吁以第三方介入为核心的设计干预。

**🔧 技术方法**

使用反射性主题分析（Reflexive Thematic Analysis）对访谈文本进行编码与主题归纳。

**📊 数据集**

24份受害者访谈记录（包括文字与语音转写），并译为英文以供分析。

**📈 对比分析**

本研究未进行实验对比或性能评估；成果以受访者叙事为依据，呈现“从疑到识别”逐步迭代的质性过程，并未与算法或其他定量指标进行对照。

**⚠️ 局限性**

局限性：样本量有限且仅来自伊朗，难以外推至其他文化；依赖受访者回忆，可能存在记忆偏差与社会期望效应；研究聚焦于受害者视角，未检验施害者行为或平台技术层面的影响。

---

## 622. Judgment-Grounded Expansion for Peer Review Generation

**arXiv ID:** 2606.23233 | [PDF](https://arxiv.org/pdf/2606.23233v1)

**作者:** Sheng Lu `[一作]` (Technical University of Darmstadt), Iryna Gurevych `[通讯]` (Technical University of Darmstadt)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一种新的人工‑AI 协作模式——判断驱动的扩展（Judgment‑Grounded Expansion），让评审者先给出评估性声明，系统随后生成一组基于该声明的评审评论候选，形成 generate‑check‑refine 流程；通过用户研究收集交互数据，并构建可扩展的仿真评估框架；同时利用合规预测（conformal prediction）对候选集进行尺寸-覆盖率权衡。

**💡 创新点**

创新点包括：①对判断驱动扩展的正式任务定义与流程模型；②利用人类输入代理（示例‑based 与 aspect‑based）和质量检查仿真实现可扩展评估；③引入判断利用得分（JUScore）评估模型对评审者输入的利用度；④将合规预测应用于生成候选集，实现动态候选尺寸控制，兼顾评审者负担。

**🔧 技术方法**

使用技术主要包括：多种开源大型语言模型（GPT‑OSS、Qwen、LLaMA、VL、DeepSeek‑R1 等）与 OpenAI o3‑mini；提示工程与多模态推理；合规预测框架；评审观点标签器（aspect tagger）；以及基于 SBERT 的相似度评分与不确定性评分（NLL、MSP、MTE）。

**📊 数据集**

采用的评审数据集为 TU Datalib（https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4460）和 OpenReview（https://docs.openreview.net）中的 100 篇评审；同时在 ICLR 数据上做额外实验。

**📈 对比分析**

评估方法包括文本相似度、推荐准确率、LLM‑as‑a‑judge（技术准确性、构建价值、分析深度）以及对抗鲁棒性；与端到端生成（DeepReview）和多种 LLM 进行对比。实验显示判断驱动扩展在文本对齐、评审准确性和鲁棒性上普遍优于基线，JUScore 进一步表明某些模型（如 Qwen3‑32B）能更好利用评审输入。合规预测在候选尺寸与覆盖率平衡上优于固定排名。

**⚠️ 局限性**

局限性包括：需评审者具备专业经验；仿真中人类输入代理和质量检查可能偏差；用户研究样本量有限；仅使用开源模型，DeepReview 采用的是预配置系统，实验设置不完全公平；评估指标并非直接衡量评审质量，存在一定主观性；闭源 o3‑mini 的可复现性有限。

---

## 623. RS-Gen: A Multi-Stage Agentic Framework for Reasoning and Search-Augmented Image Generation

**arXiv ID:** 2606.23221 | [PDF](https://arxiv.org/pdf/2606.23221v1)

**作者:** Feifei Bian `[一作]` (Xiaomi Inc.), Jian Luan `[通讯]` (Xiaomi Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 RS‑Gen，一种多阶段、基于推理与检索的图像代理框架，能够通过闭环问答与自我纠错显著提升图像生成与编辑的智能性；

**💡 创新点**

创新点在于引入“提问‑解决”闭环，结合 ReAct 思考‑行动‑观测机制，显式拆解意图、知识缺口与逻辑难点，并通过外部检索与多模态大模型实现深度推理与实时知识更新；

**🔧 技术方法**

采用多模态大语言模型（如 Qwen‑Image、Qwen‑Image‑Edit‑2511）、ReAct 代理模式、外部工具库（网页检索、VQA、地理信息、百科、深度推理 GPT‑5.4）以及生成‑评估‑纠正循环；

**📊 数据集**

使用 WISE_Verified（1,000条跨学科提示）与 RISEBench（包含时空、因果、空间、逻辑编辑子任务）等公开基准；

**📈 对比分析**

与多款基础模型（SD、FLUX、Qwen‑Image 等）及现有图像代理（Mind‑Brush、Unify‑Agent 等）对比，RS‑Gen 在 WISE 上整体得分 0.823（比 Qwen‑Image 提升 0.313，逼近商业模型），在 RISEBench 上整体得分 39.1（比基线提升 19.7，遥遥领先非商业模型）；

**⚠️ 局限性**

主要局限是对系统提示工程的高度依赖，轻量或早期多模态模型的指令理解能力不足时提升有限，未来计划引入自动提示优化与经验检索实现更强自适应与终身学习。

---

## 624. Efficient Network Inference via Hardware-Aware Architecture Search, Model Pruning & Quantization

**arXiv ID:** 2606.23210 | [PDF](https://arxiv.org/pdf/2606.23210v1)

**作者:** Lucas Heublein `[一作]` (Fraunhofer Institute for Integrated Circuits), Felix Ott `[通讯]` (Fraunhofer Institute for Integrated Circuits)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对嵌入式GNSS干扰监测，研究了高效DNN推理方案，结合迭代结构化剪枝、后训练静态量化以及硬件感知零样本NAS，对模型进行压缩与架构优化。

**💡 创新点**

创新点在于将剪枝与量化与零样本NAS统一到一个硬件感知的搜索框架（PrototypeNAS）中，能够在不训练的情况下同时优化网络结构、剪枝率和量化设置，并在满足MCU、Raspberry Pi等嵌入式平台资源约束的前提下实现近乎原始精度的压缩模型。

**🔧 技术方法**

使用技术包括：迭代结构化剪枝、8位后训练静态量化、PrototypeNAS零样本NAS（利用MeCo、NASWOT、SNIP、ZiCo等代理指标），以及基于超网络的多目标优化和超参数搜索。

**📊 数据集**

采用的实验数据集为Flexiband-7（7类干扰分类）和Flexiband-311（311类精细化特征），两者均由室内大厅收集的IQ信号通过FFT生成512×512的频谱图构成。

**📈 对比分析**

通过比较模型参数量、FLOPs、RAM/ROM占用、分类与特征化准确率以及不同硬件平台（iMXRT1062 MCU、Raspberry Pi Zero 2W、Raspberry Pi 5）的能耗与推理时延进行评估；剪枝70%可保持98.28%分类准确率；NAS选出的模型在分类任务中接近100%精度，特征化约95%；量化后几乎无精度损失；在Raspberry Pi 5上单次推理耗时≈161 ms，能耗≈560 mJ。

**⚠️ 局限性**

局限性包括：推理时延仍较高，难以实现连续实时监测；部分模型对量化更为敏感；NAS搜索仅在单次训练后评估，可能未覆盖所有有益架构；实验数据集仅来自室内大厅，模型在更复杂或室外环境下的泛化能力尚未验证。

---

## 625. Unmasking LAION-5B: Age, Gender, Race, and Emotion Biases in Large-Scale Image Datasets

**arXiv ID:** 2606.23204 | [PDF](https://arxiv.org/pdf/2606.23204v1)

**作者:** Iris Dominguez-Catena `[一作]` (Public University of Navarre), Mikel Galar `[通讯]` (Public University of Navarre)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对 LAION‑5B 的 LAION‑2B‑en 与 LAION‑2B‑multi 两个子集进行大规模面部图像分析，利用 FairFace、DeepFace 与 Emo‑AffectNet 对面部进行年龄、性别、人种和情绪识别，并通过 Ducher’s Z 统计量和 Bootstrap 置信区间评估代表性、交叉性与刻板偏差。

**💡 创新点**

首次将三种属性模型与情绪识别模型结合，对 LAION‑5B 进行系统性交叉性偏差评估，并使用 Ducher’s Z 与 Bootstrap 方法验证统计显著性，揭示了数据集层面的多维偏差。

**🔧 技术方法**

RetinaFace（人脸检测）、FairFace 与 DeepFace（年龄、性别、人种推断）、Emo‑AffectNet（情绪识别）、Ducher’s Z（共现偏差指标）、Bootstrap 置信区间分析。

**📊 数据集**

LAION‑2B‑en 与 LAION‑2B‑multi（2024 版）共计 1,004,277 个 URL 样本，提取 79,902 张面部图像。

**📈 对比分析**

通过比较 FairFace 与 DeepFace 的输出，使用 Ducher’s Z 计算共现偏差，并通过 1,000 次 Bootstrap 生成 95% 置信区间，发现偏差显著且在 0.2–0.5 的区间内稳定；未给出传统模型准确率，但结果显示偏差幅度较大。

**⚠️ 局限性**

依赖预训练模型，模型自身偏差可能影响结果；人种/性别/情绪标签有限，未涵盖全部多样性；缺乏人工标注或自我报告基准；仅评估数据集层面，无法直接推断生成模型的最终偏差。

---

## 626. Quantum Advantage in Tolerant Junta Testing

**arXiv ID:** 2606.23194 | [PDF](https://arxiv.org/pdf/2606.23194v1)

**作者:** Avishay Tal `[一作]` (University of California, Berkeley), Weiqiang Yuan `[通讯]` (EPFL)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在容忍的k-junta测试问题中，提出了第一种超多项式量子加速方法，证明在自适应设置下，量子算法仅需多项式(k,1/ϵ)次查询即可完成，而任何经典算法至少需要 k^Ω(log k) 次查询。

**💡 创新点**

创新点在于：① 设计了基于 Fourier 采样的量子影响力测试器，能够在 O(k log k) 次量子查询内估计与最近k-junta的相关性；② 通过引入基于误差纠错码的“近似junta”植入构造新的硬分布，获得了适用于更广参数范围的经典下界；③ 将这两部分结合，得到自适应量子与经典在容忍k-junta测试中的首个超多项式对比。

**🔧 技术方法**

主要技术包括：量子 Fourier 采样、影响力与相关性关系分析、误差纠错码（Gilbert‑Varshamov 码）植入、Yao 极小化原理、耦合与同态分析，以及 Hoeffding/Chernoff 失真估计。

**📊 数据集**

本工作不依赖任何真实数据集，所有结果均基于理论构造的分布与算法。

**📈 对比分析**

性能对比：量子算法的查询复杂度为 O(k log k)（或更一般的 O(k,1/ϵ)），在 ϵ 接近 1/(2k) 时可实现；经典自适应算法至少需要 k^Ω(log k) 次查询，形成量子与经典在该问题上的超多项式（近似指数）优势。

**⚠️ 局限性**

局限性：① 目前仅在 ϵ 逼近 1/2 的特定参数范围内实现超多项式加速，无法覆盖所有容忍度设定；② 经典下界仍为 k^Ω(log k)，远未达到指数级；③ 对于更大比例的容忍度或更一般的k‑junta测试，是否能实现指数级量子优势仍是开放问题。

---

## 627. StreamPPG: Low-Latency rPPG Estimation via Consistent Privileged Learning

**arXiv ID:** 2606.23186 | [PDF](https://arxiv.org/pdf/2606.23186v1)

**作者:** Yiming Li `[一作]`, Hui-Liang Shen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了低延迟的帧级远程光学测心（rPPG）估计框架 StreamPPG，并通过一致的特权学习（CPL）提升模型表示能力，支持实时流式推理。

**💡 创新点**

创新点：① 用帧级推理消除传统视频段缓冲导致的几秒延迟；② 通过 CPLE 两路训练将真实 rPPG 作为特权信息，理论保证训练-推理一致性；③ 设计自适应时间建模模块（ATMM），包含自适应注意力增强块（AAEB）和时序状态空间块（TSSB），提升空间与时间关注。

**🔧 技术方法**

技术方法：2D‑CNN 图像编码器、差分帧输入、通道注意力、CPL 双路（signal‑guided 与 signal‑free）训练、Pearson 与 MSE 损失、Mamba‑2 状态空间建模、Butterworth 滤波后续处理、Jetson AGX Orin 边缘部署、滑动窗口评估。

**📊 数据集**

数据集：PURE、UBFC‑rPPG、COHFACE、MMPD。

**📈 对比分析**

对比：在同一评估协议下与传统手工特征方法、帧级深度学习方法及视频段级方法比较。StreamPPG 在 PURE、UBFC、COHFACE 上均达 SOTA；在 MMPD 上同等或略低；实时吞吐率高（≈50–56 FPS），RMSE 低至 0.27 bpm，显著优于同类帧级模型且与视频段级模型相比，延迟更低、算力更节省。

**⚠️ 局限性**

局限：在极端运动或光照变化下仍易出现误差；CPL 需要训练时提供真实 rPPG 信号，若无标注则无法使用；对极短帧序列的鲁棒性还有提升空间。

---

## 628. Stage-dependent integer-binary encoding in factorization-machine black-box optimization

**arXiv ID:** 2606.23188 | [PDF](https://arxiv.org/pdf/2606.23188v1)

**作者:** Ryo Ogawa `[一作]` (Keio University), Shu Tanaka `[通讯]` (Keio University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了在因式分解机黑盒优化(FMQA)中采用阶段依赖整数-二进制编码的方法，并推导了从一热编码到域墙编码及其逆向的 QUBO 矩阵转换公式；通过 Rastrigin 函数的实验验证了该方法的有效性。

**💡 创新点**

创新点在于：①首次将机器学习阶段与搜索阶段分别使用不同的整数-二进制编码（学习阶段使用一热编码，搜索阶段使用域墙编码），从而充分利用 FM 的学习优势和 Ising 机的搜索优势；②提出了保持可行整数状态下目标函数不变的 QUBO 矩阵互相转换公式，使得在两种编码之间自由切换；③系统分解并量化了编码对学习精度与搜索稳定性的独立影响。

**🔧 技术方法**

使用的主要技术包括：因式分解机 (FM) 作为 surrogate 模型，AdamW 优化器进行参数学习；Ising 机 (Fixstars Amplify) 用于 QUBO 搜索；整数-二进制编码（one-hot、domain‑wall、binary）以及相应的 QUBO 约束项；目标函数值缩放；以及基于 Rastrigin 函数的离散化与实验设置。

**📊 数据集**

实验数据集为 Rastrigin 函数在连续域 [-3,3] 内离散化得到的整数集：维度 N=2、5，离散点数 q=61 与 301；每个实验使用 16 个不同随机种子得到的初始样本集合；还使用 Optuna 的 TPE 作为贝叶斯优化基准。

**📈 对比分析**

比较方法：与三种单一编码 FMQA（Oh、Dw、Bi）、双向编码变体（DwOh）以及传统贝叶斯优化（BO）对比；性能评估指标为残差误差 Δy 和无穷范数 ‖·‖∞；实验结果显示：在 N=5、q=301 等高维且高分辨率条件下，OhDw 方法在残差误差和范数上均优于其他方法；在 N=2 或 q=61 条件下，差异不显著或 OhDw 与 Oh 相当。

**⚠️ 局限性**

局限性：仅在 Rastrigin 两种维度/分辨率组合上验证；缺乏对更复杂多峰或更高维函数的测试；未对现实工程优化问题（材料设计、结构优化等）进行评估；对惩罚系数 μ 与 QUBO 稠密度的敏感性未系统分析；实验中未探索多种 Ising 机实现或其他 surrogate 模型的组合。

---

## 629. Deep learning-based detection of cessation of breathing in pre-term infants

**arXiv ID:** 2606.23213 | [PDF](https://arxiv.org/pdf/2606.23213v1)

**作者:** Dineo Serame `[一作]` (University of Oxford), Mauricio Villarroel `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

在 NICU 监护下，使用常规的阻抗肺波(IP)、心电(ECR)、脉搏血氧(PPG)等多模态信号，构建并训练深度学习模型，用于检测早产儿的呼吸暂停相关事件(COBE)。

**💡 创新点**

首次系统比较了 1D ConvNeXt、CNN 和 ResNet 等现代卷积网络在同一数据集和预处理流程下的性能，并验证了多模态融合对呼吸暂停检测的微小提升。

**🔧 技术方法**

采用了三种卷积网络架构（浅层 CNN、ResNet-18/34/50、1D ConvNeXt），并实现早期融合和晚期融合的多模态输入；训练过程中使用 Adam/AdamW、学习率调度、标签平滑及指数移动平均等技术。

**📊 数据集**

使用了 Oxford NICU 30 名早产儿（24 名完成标注）在 90 次监护会话中获得的 429.5 小时生理信号数据，经过标注后得到 6,678 个 20 秒窗口的数据集。

**📈 对比分析**

通过 5 折交叉验证与独立测试集评估，最优配置为 ConvNeXt+IP+PPG（晚期融合），在独立测试集上实现了 88.7% 的平衡准确率、0.75 的 F1 分数，IP 单模态模型已达 86.8–88.0% 的平衡准确率。

**⚠️ 局限性**

主要局限包括样本量有限（仅 24 名早产儿）、注释过程对 IP 产生一定偏倚、缺乏独立的气流测量作为参考标准，以及不同优化策略可能影响模型比较的公平性。

---

## 630. On the Effect of Segmentation Width and Cluster Size on Speech Resynthesis and Continuation in Generative Spoken Language Models

**arXiv ID:** 2606.23285 | [PDF](https://arxiv.org/pdf/2606.23285v1)

**作者:** Shunsuke Kando `[一作]` (University of Tokyo), Yusuke Miyao `[通讯]` (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了使用不同分段宽度和聚类大小的离散语音表示（即不同比特率）对 GSLM 的语音重建与续写性能的影响，证明低比特率下仍可获得可懂且自然的语音；

**💡 创新点**

① 证明传统高比特率设定对生成效果可能冗余；② 引入 LLM-as-a-Judge 评估框架，对比传统指标与人类评分的相关性；

**🔧 技术方法**

使用 GSLM 的 s2u（HuBERT + K‑means）、uLM（OPT）、u2s（Tacotron2+Parallel WaveGAN 与 VITS），并采用 WER、UTMOS、MCD、LogF0 RMSE、PPL、VERT、LLM‑as‑a‑Judge、MMOS 等多种评估指标；

**📊 数据集**

LibriSpeech（960 h 训练集、100 h 子集用于 K‑means）、LJSpeech（用于 u2s 训练与续写输入）等公开数据集；

**📈 对比分析**

对 64 种 (N, K) 组合（N 20–280 ms × K 128–16384）进行语音重建与续写实验。结果显示 N≥40 ms、K≥4096 等低比特率配置在保持 WER<5 % 与 UTMOS>4 的前提下，续写的 PPL、VERT 与 LLM‑as‑a‑Judge 分数与基线相当或更优；

**⚠️ 局限性**

自动评估指标与人类主观评分的相关性仍有限；实验仅在英语语料上验证，未涵盖多语言或更大规模数据；s2u 与 TTS 模型的选择对结果影响有限，仍需进一步探索更稳健的评估方法。

---

## 631. Causal Reward World Models: Zero-shot Reward Design for Automated Skill Generation

**arXiv ID:** 2606.23280 | [PDF](https://arxiv.org/pdf/2606.23280v1)

**作者:** Yang Yang `[一作]` (Chinese Academy of Sciences), Miao Xin `[通讯]` (China University of Mining and Technology (Beijing))

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了Causal Reward World Model（CRWM），通过离线多任务数据预训练得到可重用的因果结构，并将其作为显式先验指导LLM进行零样本奖励函数生成；

**💡 创新点**

创新点在于：①采用因果结构模型显式捕捉奖励组件与任务目标的因果关系；②设计粗到细的预训练策略，并引入Explicit Mechanism Decoupling（EMD）与Confidence‑Aware Soft Fusion来精炼因果骨架；③通过因果先验实现零样本、无需迭代的奖励生成，提升可解释性与跨任务迁移能力；

**🔧 技术方法**

使用的技术包括结构因果模型（SCM）、预训练因果基础模型LimiX、联合优化（EMD+Soft Fusion）与增广拉格朗日方法、因果图序列化成Prompt、LLM（如GPT）奖励代码生成；

**📊 数据集**

主要数据集为Dexterity（Shadow Hand 20个任务，14/6预训练/评估分割）以及ManiSkill2（跨机器人和物理引擎的任务），同时在CASBOT W1机器人上收集离线轨迹做仿真‑实测对比；

**📈 对比分析**

与基线（零样本LLM、演化ARU Eureka、URDP）对比，CRWM在未见任务上实现了0.92/0.87等成功率，匹配或超过迭代方法但无需任何演化迭代（ESI=0）；在跨机器手、不同物理引擎以及真实机器人轨迹上亦保持高成功率和AUC≈96%；

**⚠️ 局限性**

局限性包括：①依赖大量离线多任务宏观/微观干预数据；②对高维变量和极端动态交互的因果推断仍有挑战；③对LLM的依赖导致推理成本与模型可解释性受限；④在任务因果结构与预训练不匹配时性能可能下降。

---

## 632. GIF: Locally Sound Geometric Information Flow Control for LLMs

**arXiv ID:** 2606.23277 | [PDF](https://arxiv.org/pdf/2606.23277v1)

**作者:** Adam Storek `[一作]` (Columbia University), Suman Jana `[通讯]` (Columbia University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种基于局部几何信息流（GIF）的框架，用来量化大型语言模型（LLM）中输入文本片段对下游输出（如工具调用、记忆更新或代理消息）的信息泄露与完整性影响。

**💡 创新点**

创新点：① 将LLM雅可比矩阵与Softmax Fisher几何结合，构造可计算的Shannon信息流上界；② 提供完整的Lean 4形式化证明，确保本地可计算度量的安全性；③ 通过低秩近似和随机化迹估计实现大规模可扩展的实现。

**🔧 技术方法**

使用技术包括：雅可比矩阵求导、Softmax Fisher信息、局部高斯通道近似、Hutchinson随机迹估计、低秩矩阵分解与自动微分。

**📊 数据集**

使用的数据集与基准包括 AgentDojo、MSB（针对 prompt injection）和 AgentDAM（针对隐私泄露），并在多种开源大模型（Qwen 3、Gemma 4、GPT‑OSS 等）上进行评估。

**📈 对比分析**

与现有注意力或梯度基准对比，GIF 在无 declassifier 的情况下几乎达成 100% 召回率，显著优于传统注意力基准；结合轻量级 LLM declassifier，GIF 的 F1 与 GPT‑5.5xhigh 相当甚至更好，同时使用的 token 数量低至 81 倍；此外，小模型代理检测到的流量在大模型中也能成功迁移。

**⚠️ 局限性**

局限性：① 只在局部（小扰动）范围内保证上界，对大幅度输入变化可能失效；② 需要模型梯度信息，无法直接用于完全黑盒系统；③ 目前只针对单一 token 输出的流量，复杂多步交互或多维输出的分析仍需进一步研究。

---

## 633. Exposing the Illusion of Erasure in Knowledge Editing for LLMs

**arXiv ID:** 2606.23276 | [PDF](https://arxiv.org/pdf/2606.23276v1)

**作者:** Advik Raj Basani `[一作]` (Birla Institute of Technology and Science), Anshuman Chhabra `[通讯]` (University of South Florida)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了知识编辑（KE）方法的内部机制与安全性，提出逆向工程攻击框架证明编辑后的知识并未被彻底抹除，而是通过低秩更新和抑制向量形成可被逆向的弱点。

**💡 创新点**

创新点在于：①从对抗性提取视角系统评估KE的可靠性；②发现低秩更新并未覆盖原知识，而是将其投射到正交子空间；③揭示编辑过程实际上是“抑制+激励”的叠加机制；④证明编辑后的知识在参数空间呈现高度各向异的脆弱曲线。

**🔧 技术方法**

技术手段包括：白盒对抗性后缀优化（Greedy Coordinate Gradient）；低秩矩阵分解与投影分析；干预向量与词嵌入的几何对比；对损失景观的二维切片与梯度敏感性评估。

**📊 数据集**

使用了KnowEdit、CounterFact数据集中的100条反事实实体替换，实验模型包括OPT、LLaMA、Bloom、GLaM等。

**📈 对比分析**

对比方法为在同一编辑任务上分别测量原模型输出、编辑后模型输出以及对抗后缀恢复率；在白盒与跨模型、跨方法的转移实验中，恢复率普遍超过随机水平，部分达到85%以上；但对抗性提取的整体成功率仍受限于方法与模型差异。

**⚠️ 局限性**

局限性包括：逆向后缀优化在高维搜索空间中不稳定，恢复率受模型规模与编辑细节影响；实验仅覆盖单词级事实编辑，未深入自然语言推理或长期序列的实际应用；对抗性攻击在黑盒场景下效果未知，需进一步验证。

---

## 634. Scaling LLM Knowledge Boundaries via Distribution-Optimized Synthesis

**arXiv ID:** 2606.23271 | [PDF](https://arxiv.org/pdf/2606.23271v1)

**作者:** Songze Li `[一作]` (Zhejiang University), Wen Zhang `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 KDoS 框架，利用知识密度控制在合成数据中优化知识分布，以提升大型语言模型的知识注入效率。

**💡 创新点**

核心创新是将知识密度作为可控变量，采用三阶段动态反馈机制（知识抽取-质量过滤-基于分布的拒绝采样）实现从盲目合成到分布驱动合成的转变。

**🔧 技术方法**

主要技术包括知识点提取与语义聚类、DeepSeek 生成多样化问题、LLM（DeepSeek、Qwen3.5）做质量评估与过滤、基于密度的拒绝采样与迭代收敛。

**📊 数据集**

使用维基百科的 14M 句子对作为种子合成数据，评估基准为六个知识问答数据集：WebQ、NQ、TriviaQA、SimpleQA、SimpleQA-Verified 与 EntityQuestions。

**📈 对比分析**

与随机、均匀、难度加权与质量过滤等基线相比，KDoS 在六个基准上的平均准确率提升至 22.8 分，超过最优基线约 1.9 分，显著提升了知识边界。

**⚠️ 局限性**

仅在后训练（SFT）阶段验证，未扩展到预训练阶段，且实验需大量算力与数据，限制了方法的通用性与可扩展性。

---

## 635. Node-Level Performance and Energy Characterization of Flagship Science Applications on SuperMUC-NG Phase 2

**arXiv ID:** 2606.23265 | [PDF](https://arxiv.org/pdf/2606.23265v1)

**作者:** Salvatore Cielo `[一作]` (Leibniz Supercomputing Center), Gerald Mathias `[通讯]` (Leibniz Supercomputing Center)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在SuperMUC-NG Phase 2节点上，对5种旗舰科学工作负载（分子动力学、天体物理与宇宙学、有限元PDE求解器等）进行单节点吞吐量和能耗的系统评估。

**💡 创新点**

提出统一的节点级吞吐量与能效指标（每个计算元/秒和/焦耳），并发现GPU加速能带来4–12倍吞吐提升、最高15倍能效增益，但对问题粒度高度敏感。

**🔧 技术方法**

采用Intel oneAPI编译器，SYCL/Kokkos/OpenMP Offload，利用Energy Aware Runtime、p3em等工具采集能耗；系统由Sapphire Rapids CPU和Ponte Vecchio GPU构成。

**📊 数据集**

使用多规模的数据集：GROMACS 2.5×10⁵–1.1×10⁷原子、LAMMPS 3.2×10⁴–3.5×10⁶原子、GADGET 3.36×10⁷粒子、ATHENA ∼1.47×10⁸网格块、有限元 8×10⁸自由度等。

**📈 对比分析**

通过对比CPU仅、CPU+GPU配置下的吞吐量与能效，GPU在大部分工作负载上实现4–12×吞吐、3–15×能效提升；但小规模或粒度不足时优势显著下降。

**⚠️ 局限性**

局限性包括：能耗与功率预算模型对CPU不准确；GPU优势受粒度限制，需足够大块以充分利用向量单元；多节点网络影响未评估；能耗测量工具差异导致结果偏差。

---

## 636. Attention mechanism for scalable mesh-based neural surrogates of free-surface fluids

**arXiv ID:** 2606.23251 | [PDF](https://arxiv.org/pdf/2606.23251v1)

**作者:** Federico Lanteri `[一作]` (Politecnico di Milano), Massimiliano Cremonesi `[通讯]` (Politecnico di Milano)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于自注意力的 Mesh‑based Lagrangian 神经代理 NeuralPFEM，用于逼近 PFEM 的时间推进算子，减少高成本的 Navier–Stokes 求解。

**💡 创新点**

创新点在于：① 用全局自注意力取代显式图信息，消除边缘存储与多次消息传递的内存瓶颈；② 设计标准与线性两种注意力变体，兼顾表达力与可扩展性；③ 引入 RoPE 位置编码与 FiLM 物质条件编码，使模型对网格分辨率和材料参数具有良好泛化。

**🔧 技术方法**

技术手段包括：Particle Finite Element Method（PFEM）模拟数据生成；自注意力（Softmax 与 Linear）网络；FlashAttention 实现；多头注意力与 RoPE；FiLM 条件编码；Chamfer 距离评估；基于 FEM 的应力后处理。

**📊 数据集**

使用了三组 PFEM 仿真数据集：2D 倾斜面 Bingham 流、3D Bingham 圆锥坍塌以及 3D Newtonian 流的浇注试验，涵盖 2~3 维、约 3–5 千节点至 1 万节点的多分辨率网格。

**📈 对比分析**

与 GNN 处理器（多层消息传递）比较：自注意力在保持相近或更优 Chamfer 距离的同时，显著降低峰值内存（GNN 60 步时约 4700 MB，标准注意力约 500 MB，线性注意力约 600 MB），并在大规模 3D 试验中实现可训练；线性注意力在 1 万节点时可训练、计算时间仅 0.16 s，GNN/标准注意力则不可训练。

**⚠️ 局限性**

局限性：线性注意力表达力低于 Softmax，可能导致某些细节捕捉不足；标准注意力仍受 FlashAttention 内存上限限制，且在大规模场景下计算复杂度保持 O(N²)；当前方法仍依赖 PFEM 生成的训练数据，对极端物性参数或新物理模型的泛化尚未验证。

---

## 637. Lessons from the Field: A Case Study of Robotic Intervention in an Industrial Emergency

**arXiv ID:** 2606.23246 | [PDF](https://arxiv.org/pdf/2606.23246v1)

**作者:** Jonathan Lichtenfeld `[一作]` (SIM Group, Technical University of Darmstadt), Oskar von Stryk `[通讯]` (SIM Group, Technical University of Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在德国化工厂火灾后，部署UGV与UAV，远程操作UGV打开关键阀门注入惰化气体，成功防止了潜在爆炸。

**💡 创新点**

结合半刚性绳索工具扩展UGV末端执行器、无线网格重复器构建通信链路以及多方协作的任务小队，展示科研平台在真实灾情中的可落地性与局限性。

**🔧 技术方法**

使用Telerob Telemax Hybrid UGV、定制操纵工具、UAV、无线网格通信节点、远程遥控操作界面和现场视频采集装置。

**📊 数据集**

未使用公开数据集，主要依靠现场实时图像、点云和一年后重新采集的后期图像与点云作为数据来源。

**📈 对比分析**

报告未进行算法对比，仅记录操作耗时约10分钟完成阀门开启，任务成功，但未给出具体性能指标或基准。

**⚠️ 局限性**

限制包括：通信受限导致无法使用研究模块、缺乏高级感知与控制支持、硬件脆弱受腐蚀与水损、无法实现自适应SLAM与导航、缺少完整数据传输与文档记录。

---

## 638. A Behavioural Theory of Probabilistic Algorithms Using Probabilistic Abstract State Machines

**arXiv ID:** 2606.23244 | [PDF](https://arxiv.org/pdf/2606.23244v1)

**作者:** Flavio Ferrarotti `[一作]` (Software Competence Center Hagenberg), Klaus-Dieter Schewe `[通讯]` (Institut Nationale Polytechnique de Toulouse)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出概率算法的行为理论，定义四个公理（随机分支时间、抽象状态、背景、概率有限探索）并给出概率抽象状态机（pASM）模型，证明每个满足公理的概率算法都可由pASM逐步模拟。

**💡 创新点**

创新点在于首次以四个公理性原则系统性地刻画概率算法，并引入切片条件实现概率有限探索，从而实现与pASM的逐步行为等价（捕获定理）。

**🔧 技术方法**

主要技术手段包括行为理论框架、抽象状态机、概率抽象状态机、基于多重集的 witness terms、切片条件、等价类划分、概率选择规则与并行规则的组合。

**📊 数据集**

本工作为理论研究，无实验数据集，主要通过形式化定义和证明来展示结果。

**📈 对比分析**

由于是形式化证明而非实验验证，没有对比实验；通过证明可知pASM在理论上能够完美模拟满足公理的概率算法，且行为等价。

**⚠️ 局限性**

局限性：仅考虑状态无随机性的概率算法，假设每一步的可能后继状态有限；未涵盖量子算法、随机化状态等情况，且概率分布仅在选择上出现，未考虑状态层面的随机性。

---

## 639. Leveraging AutoML for Sustainable Deep Learning: A Multi-Objective HPO Approach on Deep Shift Neural Networks

**arXiv ID:** 2606.23208 | [PDF](https://arxiv.org/pdf/2606.23208v1)

**作者:** Leona Hennig `[一作]` (Leibniz University Hanover), Marius Lindauer `[通讯]` (Leibniz University Hanover)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过自动化机器学习（AutoML）中的多保真度多目标优化，系统性寻找并评估深度移位神经网络（DSNN）的超参数配置，以实现图像分类任务中精度提升与能源/碳排放双重优化。

**💡 创新点**

首次构建面向DSNN的完整配置空间，并将能耗/碳排放作为第二目标引入多目标多保真度优化框架，揭示了诸如低移位深度与高激活分数位在不同架构与数据集上的非直观能效与精度权衡。

**🔧 技术方法**

采用SMAC3改进的ParEGO与HyperBand实现多目标多保真度搜索，结合CodeCarbon跟踪碳排放，并用fANOVA评估超参数重要性。

**📊 数据集**

在CIFAR‑10和Caltech101两个标准图像分类数据集上进行实验，并对ResNet‑20、MobileNetV2、GoogLeNet等多种网络结构进行验证。

**📈 对比分析**

与默认DSNN配置比较，优化后的模型在保持甚至提升分类准确率的同时，推理阶段碳排放平均下降10%–60%，精度提升约20%，表明该方法在精度与能源效率之间实现显著Pareto最优平衡。

**⚠️ 局限性**

主要限制包括：① 仅评估推理阶段能耗，未覆盖完整训练过程；② 依赖单一GPU平台（NVIDIA A100）和CodeCarbon估算，可能对不同硬件或地区能耗分布产生偏差；③ 仅针对图像分类任务，尚未验证在目标检测、语义分割等更复杂视觉任务中的适用性。

---

## 640. Students' Perception Accuracy of Partners' AI Use and its Relation to Collaboration Performance

**arXiv ID:** 2606.23237 | [PDF](https://arxiv.org/pdf/2606.23237v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 641. DART: Draft-Agreement Routing for Training-Free Adaptive Thinking Budgets in Hybrid Reasoning Models

**arXiv ID:** 2606.23181 | [PDF](https://arxiv.org/pdf/2606.23181v1)

**作者:** Jungseob Lee `[一作]` (Korea University), Heuiseok Lim `[通讯]` (Korea University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无训练、基于草稿一致性判断的路由框架（Draft-Agreement Routing for Thinking，DART），决定在混合推理模型中是直接回答还是使用扩展推理；

**💡 创新点**

创新点在于利用两条无思考草稿的答案一致性作为高精度的难度信号，既不需要标注数据也不需梯度更新；同时在需要推理时用草稿熵预测个性化思考预算；

**🔧 技术方法**

采用自一致性采样、答案抽取与等价检测、熵到预算的无标签等距回归、两阶段生成（先草稿后推理）等技术；

**📊 数据集**

在数学推理（MATH‑500、OlympiadBench、AIME）、代码生成（HumanEval、MBPP）等公开基准上评估；

**📈 对比分析**

与始终推理（AT）、无思考（NT）、多数投票、基于置信度的路由以及有监督路由器对比；在多模型（Qwen3 8/14/32B、DeepSeek‑V3.2）上，DART在保持或提升AT准确率的同时，思考token平均减少15–69%，在数学上提升至+9点，在代码上提升至+22点；

**⚠️ 局限性**

局限包括：对多选任务不适用；需依赖任务特定的答案抽取与等价函数；误判草稿一致但答案错误的“假接受”仍是主要错误来源；Stage‑2需要token熵信息及预算控制，某些API可能不支持；整体节省的token未必直接转化为成本降低。

---

## 642. The EVerest Dataset for Secure Software Engineering

**arXiv ID:** 2606.23197 | [PDF](https://arxiv.org/pdf/2606.23197v1)

**作者:** Sophie Corallo `[一作]`, Anne Koziolek `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了覆盖安全需求、软件架构和源代码的多工件数据集，支持端到端的安全验证

**💡 创新点**

首次提供完整的安全数据集，并在需求、架构与代码层面细粒度标注安全元素

**🔧 技术方法**

采用问卷与访谈细化需求、Palladio建模、人工标注与语义链接技术

**📊 数据集**

基于EVerest开源电动车充电桩软件栈的文档、需求、架构模型和源代码

**📈 对比分析**

通过人工发现并修复了CWE‑1295漏洞，证明数据集在实际安全评估中的实用性

**⚠️ 局限性**

仅覆盖单一项目，标注工作高度人工化，缺乏自动化与大规模可扩展性

---

## 643. Unlocking In-Context Learning in Audio-Language Models from Decentralized Medical Audio

**arXiv ID:** 2606.23243 | [PDF](https://arxiv.org/pdf/2606.23243v1)

**作者:** Ran Piao `[一作]` (Eindhoven University Of Technology), Aaqib Saeed `[通讯]` (Eindhoven University Of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了一种联邦自上下文化（Federated Self-Contextualization，FSC）框架，用于在多个医院之间无标注、无中心化数据共享的条件下进行少样本临床音频诊断。

**💡 创新点**

创新点在于通过无医学语义的伪标签来训练多模态语言模型的上下文推理能力，并将语言模型预训练的医学知识与该推理能力解耦，实现在不需要真实诊断标签的前提下进行临床语义匹配；同时采用分阶段训练和联邦学习提升跨机构泛化。

**🔧 技术方法**

采用了医学预训练语言模型 MedGemma-4B-IT 与医学音频编码器 CaReAQA 的跨模态对齐，使用线性投影将音频嵌入注入到视觉边界标记之间，三阶段逐步训练（对齐、周期化学习、LoRA 微调）以及 FedProx 联邦优化；伪标签由本地 K-means 聚类生成。

**📊 数据集**

使用了七个心肺音频数据集：ICBHI、CIRCOR、CoughVID、HFLUNG、SPRSound、COVID-19 Sounds、ZCHSound，共计约17,500条训练样本与3,000条测试样本。

**📈 对比分析**

在 2-way/3-way、2-shot/5-shot 的少样本实验中，与 Pengi、GAMA、Gemma3n、Qwen2.5-Omni-7B 和 Audio Flamingo 等基线对比，FSC 在 2-way 2-shot 任务中取得 71.6% 的准确率，比最强基线高 9+ 个百分点；在 ROUGE‑L 与 BERTScore 上亦显著优于基线。

**⚠️ 局限性**

局限性包括：在更高维度（3-way）或更多样本（5-shot）时准确率下降；对 episode 结构和支持样本数量敏感；依赖大规模医学预训练语言模型，资源消耗高；尚未验证在其他语音/言语诊断场景中的泛化能力。

---

## 644. Privacy-Preserving Person Re-Identification from Temporal Sequences with Transformer and Hungarian Optimization

**arXiv ID:** 2606.23230 | [PDF](https://arxiv.org/pdf/2606.23230v1)

**作者:** Raphaël Delécluse `[一作]` (IMT Nord Europe), Laurent Guimas `[通讯]` (Explain)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

使用深度图像实现隐私保护的人体重识别，并通过Hungarian算法解决多视角下的关联问题。

**💡 创新点**

提出仅利用深度图像的重识别方法，结合Transformer序列编码与Batch Hard Triplet Loss，并引入Hungarian算法实现全局匹配，从而在隐私友好的场景中获得竞争性性能。

**🔧 技术方法**

采用深度卷积编码（ResNet-50/ResNet小型）、Transformer Encoder进行序列建模、Batch Hard Triplet Loss进行嵌入学习，以及Hungarian算法进行全局匹配。

**📊 数据集**

在TVPR2、GODPR和BIWI RGBD-ID这三组顶视角/多模数据集上进行实验。

**📈 对比分析**

与现有SOTA相比，RGB-D模型在TVPR2上Rank‑1 99.5%、mAP 99.9%，GODPR上Rank‑1 100%、mAP 98.4%，BIWI上Rank‑1 96.4%、mAP 97.4；深度单模在多数据集也能达到90%+的Rank‑1，使用Hungarian算法后精度进一步提升至接近100%。

**⚠️ 局限性**

深度单模在相似体型或不同视角下表现下降，跨数据集泛化有限，Hungarian算法的O(n³)复杂度在大规模实时应用中可能成为瓶颈，且模型需要外部RoI检测与跟踪支持。

---

## 645. PhysFlow: Frequency Decoupled with Dual-Field Rectified Flow for Remote Photoplethysmography

**arXiv ID:** 2606.23226 | [PDF](https://arxiv.org/pdf/2606.23226v1)

**作者:** Zixu Li `[一作]` (Nanjing University of Science and Technology), Jian Yang `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 PhysFlow 框架，基于频率分离的双场正则化流模型，完成远程光学心率估计与波形重建，显著提升在复杂干扰环境下的鲁棒性。

**💡 创新点**

① 通过频域分解将 rPPG 信号拆分为趋势与幅度两部分并作为监督目标；② 为两部分分别学习条件速度场，减少互相干扰；③ 采用 rectified flow 使采样仅需少量 ODE 步，提升推理速度。

**🔧 技术方法**

频域分解（RFFT+频带掩码）、周期变压器（Periodic Transformer）提取视频特征、双场条件速度头（趋势头和幅度头）、Rectified Flow 训练与推理、负 Pearson 损失以及 MSE 损失。

**📊 数据集**

BUAA‑MIHR、VIPL‑HR、NIRP‑DRV、NIRP‑IND 四大公开基准数据集。

**📈 对比分析**

与 DeepPhys、TS‑CAN、PHASE‑Net、RhythmFormer 等多种先进方法进行横向比对；在所有四个数据集上均取得最低 MAE/RMSE、最高 Pearson 相关系数；波形指标 Wave‑Corr 最高、PE 最低、SNR 最高；跨数据集迁移亦优于竞争方法；计算效率更高，吞吐量最高、推理时间最短、显存占用最低。

**⚠️ 局限性**

模型参数与 FLOPs 相对较大，未实现轻量化；频域分解使用固定阈值，对异常心率或强运动干扰的适应性有限。

---

## 646. Temporally Aware Densification for Dynamic 3D Gaussian Splatting

**arXiv ID:** 2606.23212 | [PDF](https://arxiv.org/pdf/2606.23212v1)

**作者:** Vikram Sandu `[一作]` (Indian Institute of Science), Rajiv Soundararajan `[通讯]` (Indian Institute of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对动态 3D Gaussian Splatting（3DGS）中稀疏梯度导致的稀疏动态区域重建问题，本文提出了三种改进：可视化感知稠密化（VAD）、时变阈值自适应（TAT）以及时间偏移变形（TOW），通过引入时间可见性权重、动态阈值和频率自适应变形来显著提升动态区域的细节重建。

**💡 创新点**

创新点在于：① 将每个高斯的透明度作为可见性权重，重新加权梯度累积；② 根据高斯的时间寿命自适应调整稠密化阈值，特别提升短生命周期高斯的稠密化；③ 通过时间偏移变形将有限的 Fourier 基函数聚焦到每个高斯的时间中心，实现高频运动的更高分辨率，同时保持低频平滑，避免增加参数。VAD 模块还能无缝插拔到多种现有动态 3DGS 框架。

**🔧 技术方法**

主要技术包括：3D Gaussian Splatting、Fourier‑based 动态位移、RBF‑based 透明度建模、低阶多项式旋转与尺度、可视化加权梯度计算、时间偏移变形（TOW）和动态阈值调节（TAT）。

**📊 数据集**

实验数据集为：Neural 3D Video（N3DV）、Interdigital、VRU Basketball，均为多视角动态场景，分辨率从 1920×1080 到 2704×2028，帧数从 50 到 300。

**📈 对比分析**

与 4DGaussian、Ex4DGS、STG、Swift4D、SaroGS 等基线相比，本文在 PSNR、M‑PSNR、M‑SSIM、LPIPS 等指标上均取得更高分数，尤其在动态区域的 M‑PSNR 与 M‑SSIM 最高；同时 FPS 最高（达 146 FPS），训练时间和模型尺寸保持在竞争范围内，证明了方法既提升了视觉质量又保持了实时性与模型压缩。

**⚠️ 局限性**

局限性：仅在多视角短时序（约 5‑10 秒）进行评估，尚未验证在数分钟长序列的可扩展性；目前仅针对多视角输入，单目或稀疏视角下的适用性仍需进一步研究。

---

## 647. The Correct Answer Trap: Pedagogically-Grounded Detection and Feedback for Hidden Misconceptions

**arXiv ID:** 2606.23205 | [PDF](https://arxiv.org/pdf/2606.23205v1)

**作者:** Moiz Imran `[一作]` (University College London), Sahan Bulathwela `[通讯]` (University College London)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在答案正确但推理错误的情形下自动检测学生误解，提出分级评估准则和detect‑verify‑escalate管道，并在Eedi平台上进行评估

**💡 创新点**

将答案正确性与方法有效性区分的分级评估框架，构建可将不确定案例进一步诊断的pipeline，并利用开放权重LLM实现高达84%检测率

**🔧 技术方法**

BERT fine‑tuned、Gemini 3 Flash、Gemma 4 26B LLM，配合分级评估提示与后续诊断提问生成

**📊 数据集**

Eedi 2026数学诊断题库共20,964条学生回答，涵盖15道题目，包含正确答案但误解（TM）占1.6%

**📈 对比分析**

BERT仅57%检测率，Gemini 70%（FP 8.9%），Gemma 84%检测率（FP 18%），正预测值约10.9%；标准ML干预无显著提升

**⚠️ 局限性**

仅覆盖15题，TM集中于两题，未收集生成诊断提问后的学生回应，缺乏真实课堂验证

---

## 648. Memory Contagion: Cross-Temporal Propagation of Evaluator Bias via Agent Memory

**arXiv ID:** 2606.23195 | [PDF](https://arxiv.org/pdf/2606.23195v1)

**作者:** Zewen Liu `[一作]` `[通讯]`, Zewen Liu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探索了LLM代理在记忆系统中的偏差跨时传播现象（Memory Contagion），并通过四阶段实验验证了偏差即使在完美合并下也会被持续传播。

**💡 创新点**

创新点在于提出并量化了Memory Contagion概念，证明了偏差输入是传播的充分原因，并揭示合并过程对不同偏差类型的放大或衰减作用。

**🔧 技术方法**

使用Oracle与LLM合并技术、Wasserstein距离衡量行为分布差异、内容与检索分解、剂量-响应曲线以及置换检验等方法进行实验和评估。

**📊 数据集**

实验基于20个开放式问答任务的合成数据集，并利用DeepSeek‑Chat生成的轨迹进行记忆存储与检索。

**📈 对比分析**

通过比较有偏记忆与无偏记忆的Wasserstein距离（Γ_A≈13, Γ_B≈2），结果表明即使在完美合并下也能检测到显著传播，长度偏差可被显著抑制，权威偏差可能被放大。

**⚠️ 局限性**

局限性包括仅测试两种偏差类型、仅使用单一模型DeepSeek‑Chat、权威偏差结果基于单次实验、偏差注入方法可能与质量特征相关，缺乏多任务和多模型验证。

---

## 649. BoxCtrl: 3D-Aware Visual Prompting for Geometric Image Editing

**arXiv ID:** 2606.23270 | [PDF](https://arxiv.org/pdf/2606.23270v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 650. Wireless Personal Agent: Extending Wireless Intelligence from Networks to Terminals

**arXiv ID:** 2606.23255 | [PDF](https://arxiv.org/pdf/2606.23255v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 651. Conceptual Design of an Ecosystem for Real Farm Data Collection toward Agricultural AI Foundation Models

**arXiv ID:** 2606.23258 | [PDF](https://arxiv.org/pdf/2606.23258v1)

**作者:** Junsei Tanaka `[一作]` (Kyoto University of Advanced Science), Yoshihiro Sato `[通讯]` (Kyoto University of Advanced Science)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了一个基于农场数据稀缺性和需求的自动定价、收益分配与真实性验证的生态系统框架

**💡 创新点**

将稀缺性评分、需求计数和收益分配机制整合到同一平台，并通过设备签名实现数据真实性

**🔧 技术方法**

使用数字签名、视觉-语言模型、稀缺性评估算法、需求计数与收益分配公式

**📊 数据集**

未使用公开数据集，而是设计了农场机器摄像头和传感器实时采集的视频与环境数据

**📈 对比分析**

论文未进行实验对比或性能评估，仅通过经济估算说明可行性

**⚠️ 局限性**

目前仅为概念设计，缺乏实现、仿真验证，且未考虑大规模部署的技术与监管挑战

---

## 652. SteerVTE: Seamless Video Text Editing with Style and Glyph Control

**arXiv ID:** 2606.23254 | [PDF](https://arxiv.org/pdf/2606.23254v1)

**作者:** Kai Zeng `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了端到端的视频文字编辑框架SteerVTE，能够在保持背景一致的前提下准确、无闪烁地编辑视频中的文字。

**💡 创新点**

核心创新在于冻结预训练的DiT文本‑视频模型，并通过轻量级Text Context Adapter注入三种精确编辑信号（空间‑时间遮罩、VLM风格编码、双粒度字形信息），以及结合GLAS损失和三阶段课程学习，显著提升文字识别、风格一致性和时间连贯性。

**🔧 技术方法**

采用预训练的DiT作为骨干，VLM（如Qwen2.5‑VL）作为风格编码器，OCR网络与LoRA构建双粒度字形模块，GLAS损失融合了空间焦点和字形监督；训练过程采用三阶段逐步从图像到视频的课程学习。

**📊 数据集**

构建了规模达一百万三元组的SteerVTE‑1M数据集，包含SynthTE（基于字体、字幕渲染与视频合成的合成样本）和RealTE（来自真实图像文本编辑的样本），并推出VTE‑Bench基准（100合成+100真实视频）用于评测。

**📈 对比分析**

与基于逐帧图像编辑、传统视频编辑、以及闭源Seedance 2.0等方法对比，SteerVTE在句子准确率、NED、Style‑Sim、FVD等指标上均实现显著提升，句子准确率提升至77%（相当于Seedance 2.0的2×），背景保真度也大幅优于其它开源基线。

**⚠️ 局限性**

局限性包括仅支持英文单行编辑、缺乏多语言与多行处理能力、对动态字幕效果和极细小字体的鲁棒性仍有待提升，并且在极端场景下仍可能出现单字符错误，后续工作将探索多语言、实时轻量化以及强化学习优化。

---

## 653. SPADE: Structure-Prior Adaptive Decision Estimation

**arXiv ID:** 2606.23219 | [PDF](https://arxiv.org/pdf/2606.23219v1)

**作者:** Yifan Wang `[一作]` `[通讯]` (McGill University), Yifan Wang (McGill University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 SPADE 方法，用于决定是否、如何以及哪些物理结构先验应被强制约束；

**💡 创新点**

创新点在于将结构先验的施加视为对“违反结构块”的收缩，利用一次精确的指定检验与 Stein‑unbiased James‑Stein 收缩实现自适应决策，并支持嵌套选择与 FDR 控制；

**🔧 技术方法**

使用了线性模型、子空间投影、James‑Stein 收缩、Stein‑unbiased 风险估计、精确 F 统计检验以及 Benjamini‑Hochberg 多重检验等统计方法；

**📊 数据集**

通过线性子空间先验（矩阵约束）、reservoir 保守律以及 Duffing 系统的 Hamiltonian 约束等合成实验数据；

**📈 对比分析**

与自由估计、硬约束、软正则化、交叉验证、BIC 以及 MLP 进行比较，SPADE 在所有测试中几乎达到贝叶斯先验 oracle，优于神经网络，在约 1/71 次求解下与 CV 相当，并在嵌套结构选择和子集发现上实现 100% 正确率与 FDR 控制；

**⚠️ 局限性**

限制在于先验必须是参数线性子空间；对极其 ill‑conditioned 设计需使用校准块；对非线性先验（如深度网络参数化的 Hamiltonian）仍需进一步扩展。

---

## 654. MuPPET: A Benchmark for Contextual Privacy of LLM Assistants in Multi-Party Conversations

**arXiv ID:** 2606.23217 | [PDF](https://arxiv.org/pdf/2606.23217v1)

**作者:** Elena Sofia Ruzzetti `[一作]` (Parameter Lab), Martin Gubri `[通讯]` (Parameter Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MuPPET基准，用于评估大型语言模型在多方对话中的上下文隐私泄露风险。

**💡 创新点**

首次针对多方环境量化隐私泄露，展示多方对话中泄漏显著高于一对一场景，并评估现有防护措施的不足。

**🔧 技术方法**

采用LLM‑as‑judge评估泄露与效用，构建多方对话生成流程并使用提示与隐私防护策略（CI‑Mem、PrivacyChecker）进行对比。

**📊 数据集**

利用自定义种子与PANORAMA身份数据生成562条英文多方工作场景对话，包含用户背景记忆与敏感信息。

**📈 对比分析**

对开源模型（Meta‑Llama、Qwen3）与闭源前沿模型（Gemini 2.5 Pro、GPT‑5.5）进行泄漏率与效用评估，发现闭源模型泄漏率最低，但任何模型在多方对话中仍有显著泄露，且防护措施会降低效用。

**⚠️ 局限性**

局限于专业团队情境、合成对话、仅英文，未覆盖家庭、跨语言等更隐晦或多样化的隐私规范。

---

## 655. CFPO: Counterfactual Policy Optimization for Multimodal Reasoning

**arXiv ID:** 2606.23206 | [PDF](https://arxiv.org/pdf/2606.23206v1)

**作者:** Zhangyuan Yu `[一作]` (Beijing University of Posts and Telecommunications), Qicheng Lao `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于反事实的策略优化框架 CFPO，利用跨模态显著性掩码在多头注意力中构造反事实路径，并通过 KL 正则化强制视觉语言模型在强化学习训练中真正依赖视觉信息，从而降低语言偏差与幻觉。

**💡 创新点**

核心创新在于：①在注意力输出层实现对高显著性视觉区域的反事实干预；②将反事实与真实分布的 KL 差异作为正则项直接嵌入 RL 优化；③不需要额外奖励模型或人工标注，即可在 GRPO/DAPO 基础上提升因果一致性；④通过统计异常检测动态生成显著性掩码，使干预更细粒度。

**🔧 技术方法**

使用的技术包括：跨模态注意力矩阵分解与显著性掩码、统计异常阈值（μ+λσ）来识别关键视觉标记、均值干预（将高显著性视觉向量替换为图像平均向量）、KL 正则化（Counterfactual Regularization）、GRPO 与 DAPO 强化学习框架、Qwen2.5‑VL‑3B 语言模型、Chain‑of‑Thought 生成与评估、训练与评估脚本、NVIDIA A800 80G GPU。

**📊 数据集**

训练数据：ViRL39K（38,870 视觉‑语言推理问答对）。评估数据：RealWorld‑centric 任务（C‑VQA‑Real、MARS‑Bench、POPE、TextVQA、MMMU‑Pro(V)），Mathematics‑centric 任务（Geo3k、We‑Math、MMk12、MathVerse、LogicVista）。

**📈 对比分析**

与基线 GRPO、DAPO 以及感知‑aware 方法 PAPO 进行对比。CFPO 在大多数任务上提升 3.17%–6.25%（相对基线），并比 PAPO 提升 1.32%–2.13%。在 RealWorld 任务中 CFPO_D 相比 PAPO_D 提升 2.40%；在数学推理任务中也表现出显著优势。实验显示 CFPO 在样本效率和推理稳定性方面优于所有基线。

**⚠️ 局限性**

局限性包括：①反事实正则化增加了额外的计算开销（尤其在多头注意力层展开时）；②显著性掩码阈值 λ 的选择对性能影响较大，需要经验性调优；③实验仅在 Qwen2.5‑VL‑3B 规模上验证，尚未评估更大模型或不同架构的可扩展性；④对异常检测方法的假设（高显著性视为重要）可能在不同数据分布下不成立。

---

## 656. Bridge the Gaps: Heterogeneous Attributed Graph Clustering via Quaternion Representation Learning

**arXiv ID:** 2606.23199 | [PDF](https://arxiv.org/pdf/2606.23199v1)

**作者:** Xinxi Chen `[一作]` (Guangdong University of Technology), Xiang Zhang `[通讯]` (BH-Energy Technology Co Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对包含属性与拓扑信息的异构图（以及仅属性的混合类型数据），本文提出 AGREE 框架实现无监督聚类，先通过多级对齐编码统一属性类型，再利用四视图映射和四元数图卷积学习聚类友好的表示，并在训练过程中同时优化图重构与聚类目标。

**💡 创新点**

创新点包括：①引入多级（值级、特征级、属性型、对象级）对齐编码以消除属性异构性并构造更合适的相似度图；②将任意维属性投射到四元数空间，利用 Hamilton 乘积增强属性交互，从而缓解“过度主导”(OD) 与“过度平滑”(OS) 两大问题；③在不预设聚类数的前提下通过聚类正则化实现模型可适应多尺度聚类。

**🔧 技术方法**

核心技术：多级对齐编码（概率统计、条件分布、距离投影、对象相似度构图）；四视图投影+四元数图卷积（QGE）；联合优化（图重构 KL 损失 + 频谱聚类正则）；Adam + ReLU；实验用 PyTorch 1.8 与 NVIDIA 3090。

**📊 数据集**

使用 19 个公开数据集：9 个传统属性图（ACM、WIKI、CITE、DBLP、FILM、CORA、WISC、UAT、AMAP）和 10 个混合类型属性数据集（MM、HF、BC、AA、TTT、ZO、YE、GI、WI、II）。

**📈 对比分析**

与 13 种基线（经典 K‑Means、GAE、ARVGAE、DAEGC、CCGC、DFCN、EGAE、CONVERT、SCDGN、MAGI、GLAC、DESE、CDC）比较。AGREE 在所有外部指标（ACC、NMI、ARI）和内部指标（SC、DBI、CHI）上平均排名 1.1，尤其在混合类型数据上显著领先；在传统属性图上也保持最优或次优表现，且在不同聚类数下保持鲁棒性。

**⚠️ 局限性**

局限性：①对齐编码与四元数投影增加了预处理与模型参数的复杂度；②在极度稀疏或高维属性场景下，投影与图构造的计算成本仍显高；③对超参数（α、β、学习率、层数）的敏感性尚未系统化；④目前仅验证在单机 GPU 上，未探讨大规模分布式或联邦学习环境下的可扩展性。

---

## 657. When Does Intrinsic Self-Correction Help? A Task-Sensitive Analysis

**arXiv ID:** 2606.23196 | [PDF](https://arxiv.org/pdf/2606.23196v1)

**作者:** Elroy Stav `[一作]` (Bar-Ilan University), Sarit Kraus `[通讯]` (Bar-Ilan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估大语言模型在无外部反馈的自我纠错（SC）过程在不同任务类型（可验证任务、推理任务和词语游戏）下的效果，并对比多模型的表现。

**💡 创新点**

提出任务敏感视角，将自我纠错与任务特性（可验证性、推理强度、第二意见生成）关联，系统比较多模型多任务的 SC 效果，揭示其在不同任务中的优势与局限。

**🔧 技术方法**

采用两步/三步自纠错提示、贪婪解码（温度0）、统计显著性检验（McNemar、配对t检验、Wilcoxon）以及 LLM‑judge 评估子任务属性。

**📊 数据集**

使用 SAT、BBEH、HLE（多项选择/开放式）、Wordle、Hangman、Codenames 等六个基准数据集。

**📈 对比分析**

通过与零样本基线（Base）对比，使用准确率或任务专属得分并做显著性检验，发现大多数模型在可验证任务上显著提升，推理任务提升有限，部分模型在 Codenames 上出现过度修订导致性能下降。

**⚠️ 局限性**

局限性：未覆盖开放式生成任务；未尝试采样、长迭代或其他推理策略；子任务属性评估依赖 LLM 评判，可能存在偏差；模型演进可能改变结论。

---

## 658. Capable but Careless: Do Computer-Use Agents Follow Contextual Integrity?

**arXiv ID:** 2606.23189 | [PDF](https://arxiv.org/pdf/2606.23189v1)

**作者:** Anmol Goel `[一作]` (TU Darmstadt), Iryna Gurevych `[通讯]` (TU Darmstadt)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套基于上下文完整性（CI）的评估工具，评测并量化计算机使用代理（CUA）在跨应用环境中对个人信息的泄露风险，探究三种典型失效模式（可视共位、任务模糊过度分享、收件人不匹配）并对15款前沿代理进行评测；随后在端到端UI环境验证泄露持久性，并尝试三种提示层面防御以降低泄露并提升任务完成率。

**💡 创新点**

创新点包括：① 将CI理论应用于多应用、多轮交互的代理泄露评估，形成可执行的情景生成与评分框架；② 通过MCTS+LLM自动生成高质量、真实感的泄露情景；③ 引入“参与条件泄露率”作为关键指标，区分主动泄露与逃避行为；④ 在端到端UI实验中验证泄露模式的可迁移性；⑤ 证明简单提示干预即可显著降低泄露并提升效用。

**🔧 技术方法**

技术手段包括：Monte Carlo Tree Search（MCTS）+LLM变异与评判；多源评估（确定性匹配+LLM判断）融合的混合评分器；OpenApps多应用渲染环境；基于情境JSON的状态基础评估；以及三种提示层面防御（限制读取、CI规则嵌入、收件人类型化）。

**📊 数据集**

数据集为自动生成的情境池（约117个情境/模型），覆盖三类失效模式；端到端实验使用OpenApps的50个分层情境；情境生成过程中基于手工seed与LLM扩展。

**📈 对比分析**

与15款前沿代理对比：平均任务完成率68.8%，平均泄露率67.9%；高效代理并不保证低泄露，参与条件泄露率差异可达84个百分点。端到端实验显示泄露率在高效代理中仍显著；提示防御可将参与条件泄露率从约50%降低至15-20%，同时效用提升约15-23个百分点。

**⚠️ 局限性**

局限性包括：OpenApps是受控的六应用环境，真实用户数据未覆盖；情境池具有对抗性，可能高估泄露概率；端到端实验仅覆盖两款代理、50个情境；防御实验仅覆盖三种提示干预与三款模型，难以推广至所有模型；未考虑长期记忆、多轮个性化、专业/编程代理等场景。

---

## 659. A Rank-Preserving Locality Theorem

**arXiv ID:** 2606.23180 | [PDF](https://arxiv.org/pdf/2606.23180v1)

**作者:** Jan Dreier `[一作]`, Szymon Toruńczyk `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文证明了一个针对一种带有距离秩（distance rank）的句法形式化的原子逻辑的秩保持局部性定理，并给出了有效的布尔组合形式的正规化。

**💡 创新点**

创新点在于：①提出了弱散射子句（scatter sentences）的形式，能够通过贪心算法高效计算；②使用句法重写而非Ehrenfeucht‑Fraïssé游戏，直接得到有效算法；③将定理应用于有界合并宽度（bounded merge‑width）图，突破了之前需要独立集算法的限制。

**🔧 技术方法**

主要技术包括：距离秩的定义、距离原子、局部化量化、horizon 函数、句法重写、分离与远离量化 lemma、组合性覆盖论证，最终得到布尔组合的局部公式与散射子句。

**📊 数据集**

本文为理论工作，无实验数据集；若有实验，推测会在小型合并宽度图上验证公式评估效率。

**📈 对比分析**

与之前工作（如Grohe‑Kreutzer‑Siebertz）比较，本文的散射子句更弱且可高效求解，且重写算法显式给出；理论上运行时间为阶层函数，虽然极大，但为可计算。对比实验未给出，理论证明表明在有界合并宽度图上可实现 FPT。

**⚠️ 局限性**

局限性：horizon 函数导致的界非常大，实际实现难度高；定理仅适用于有限关系符号的结构；对更一般的图类或更强的逻辑表达能力（如原子多重量化）尚未扩展。

---

## 660. Ranking Companion: A Visual Analytics Approach to Item-Based Ranking with Hybrid Item Selection

**arXiv ID:** 2606.23263 | [PDF](https://arxiv.org/pdf/2606.23263v1)

**作者:** Aman Kumar `[一作]` (University of Zürich), Jürgen Bernard `[通讯]` (University of Zürich)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了 Ranking Companion，一个支持多种物品选择方法的可视化分析工具，用于基于用户偏好生成个性化物品排序。

**💡 创新点**

将六种互补的物品选择方法（基于搜索、相似性、趋势、置信度、UMAP 投影、排名面板）与主动学习融合，形成闭环交互学习流程。

**🔧 技术方法**

使用交互式机器学习（LightGBM 排序模型 + SHAP 解释）、可视化分析界面、主动学习策略与多维度用户交互技术。

**📊 数据集**

使用 Last.fm 音乐艺术家标签数据（约 6000 个热门艺术家，实验使用 1000 个子集）。

**📈 对比分析**

通过 10 名受试者的成形用户研究，对六种选择方法在准确性、多样性、新颖性、透明度、控制感、满意度六维度进行 7 点李克特量表评估，结果显示无单一方法最佳，组合使用可平衡各维度。

**⚠️ 局限性**

仅采用列表式相对等距偏好输入，缺乏强度信息；未对多用户大规模客观准确性做评估；主动学习使用属性覆盖启发式而非正式不确定度估计；项目仅在音乐领域验证，泛化性待进一步验证。

---

## 661. Dynamic multi-agent deep reinforcement learning-based pricing and incentivization approach in multimodal transportation networks

**arXiv ID:** 2606.23257 | [PDF](https://arxiv.org/pdf/2606.23257v1)

**作者:** Khadidja Kadem `[一作]` (University Gustave Eiffel), Latifa Oukhellou `[通讯]` (University Gustave Eiffel)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多智能体深度强化学习框架，联合调节共享出行服务（SMS）动态定价与公共交通的时空激励，以优化乘客成本、车辆排放与空间公平。

**💡 创新点**

创新点在于：①将 SMS 提供商和公共管理者建模为两个独立 RL 智能体，实现多目标（利润、效率、排放、公平）协同；②采用宏观级多模式交通仿真（MFD+UE+匹配）实时反馈给 RL 代理；③使用 O‑D 维度的激励/价格分解，显著降低动作空间维度。

**🔧 技术方法**

核心技术包括深度强化学习（DDPG/TD3 等），多类用户均衡求解、MFD 基础的交通仿真、车队匹配算法与收益/排放评估。

**📊 数据集**

使用 Sioux Falls 基准网络、人工生成的早高峰出行需求、分层 VOT 分类、现有公交/地铁线路与车速数据；通过对比静态激励与动态 RL 方案得到结果。

**📈 对比分析**

对比方法：①基准无激励；②对低 VOT 乘客的静态优惠；③仅对无公交替代的 SMS+PT 交叉优惠；④RL 动态激励；再加上动态定价与激励的联合实验。结果显示：动态激励可将拥堵峰值削减 10‑20%；乘客成本下降约 20%；排放下降约 10%；公共交通收益近乎翻倍；公平性（Gini 指数）显著改善。总体而言，联合 RL 方案在多维度指标上优于任何单一静态策略。

**⚠️ 局限性**

局限性包括：①实验规模仅为中等规模 Sioux Falls 网络，难以直接推广至大城市；②假设 SMS 司机完全遵从并不考虑重新定位与不配合；③拼车定价为静态且不考虑动态补贴；④交通仿真中公交无容量限制、排放模型为基于平均速度的简化公式；⑤未直接针对社会公平（收入层级）制定奖励；⑥ RL 训练对网络拓扑与需求分布高度敏感，需重新训练以适应不同场景。

---

## 662. P-JEPA: Procedural Video Representation Learning via Joint Embedding Predictive Architecture

**arXiv ID:** 2606.23256 | [PDF](https://arxiv.org/pdf/2606.23256v1)

**作者:** Felix Tristram `[一作]` (Technical University of Munich), Ghazal Ghazaei `[通讯]` (Carl Zeiss AG)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种在长达30分钟视频上进行遮蔽潜在预测的无监督框架，学习可用于在线和离线程序化视频理解的表示。

**💡 创新点**

创新点在于将遮蔽潜在预测迁移到帧对齐的密集特征上，并引入片段可因果注意力与双维RoPE，实现在不依赖语言或脚本的情况下获得可复用的程序化表示。

**🔧 技术方法**

使用Mask‑Latent Prediction（P‑JEPA）、片段可因果Transformer、双维RoPE以及预训练视频编码器（I3D、TSM、V‑JEPA2.1）等技术。

**📊 数据集**

在EgoExo4D、EgoProceL和Assembly101这三大长视频手工操作/日常任务数据集上进行实验。

**📈 对比分析**

通过线性探测、流式推理和时间动作分割等基准，与现有最先进模型比较，P‑JEPA在EgoExo4D细粒度分类实现SOTA，在EgoProceL稀疏标签下仅使用33%数据即可匹敌全量训练的结果，并在EgoProceL和Assembly101的编辑分数提升约10%。

**⚠️ 局限性**

限制在于需要预先划分的片段边界，无法直接在无标注的实时视频上使用；依赖预训练的帧级特征，且均匀聚合或动作池化对性能影响较大。

---

## 663. Tackling "AI against sustainability"

**arXiv ID:** 2606.23192 | [PDF](https://arxiv.org/pdf/2606.23192v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 664. LP-NavOA: Integrated Local Navigation and Obstacle Avoidance for Humanoid Robots under Limited Perception

**arXiv ID:** 2606.23249 | [PDF](https://arxiv.org/pdf/2606.23249v1)

**作者:** Yukun Luo `[一作]` (National University of Defense Technology), Peng Li `[通讯]` (National University of Defense Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

开发了LP-NavOA框架，将学习的本地规划器嵌入冻结的PPO底层，以实现类人机器人在有限感知条件下的导航与障碍规避。

**💡 创新点**

创新点在于①仅在训练阶段使用规划师/航路点教师，通过动态路形塑造和教师主动收集蒸馏出仅覆盖方向指令的递归本地规划器；②部署时完全去掉全局地图和外部规划器，仅依赖本地射线感知、身体帧目标和姿态感知。

**🔧 技术方法**

采用强化学习的PPO进行感知-动作底层训练；行为克隆蒸馏结合递归GRU规划器；动态路形控制与教师主动数据收集；以及安全过滤器和短程射线感知技术。

**📊 数据集**

在MuJoCo仿真中使用Unitree G1模型，构造开放墙和室内障碍布局；并在真实Unitree G1硬件上进行部署验证。

**📈 对比分析**

与仅使用PPO底层的R1方法及教师启用的PP方法对比，指标包括相对时效到达率、刷碰率和硬碰率。LP-NavOA在开放墙任务的时效到达率提升至85–97%，刷碰率显著下降；室内任务亦有明显改善。

**⚠️ 局限性**

局限在于仍依赖射线感知的短程观测，未能处理动态障碍；安全过滤器在仿真中略显保守；对真实机器人在更复杂、动态环境下的泛化仍需进一步验证。

---

## 665. HOLMES: Evaluating Higher-Order Logical Reasoning in LLMs

**arXiv ID:** 2606.23238 | [PDF](https://arxiv.org/pdf/2606.23238v1)

**作者:** Yucheng Wu `[一作]` (Peking University), Liangming Pan `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了HOLMES——第一个面向大语言模型的真实世界高阶符号推理基准，涵盖法律与金融领域的规则冲突、优先级、异常处理、范围约束及组合推理；

**💡 创新点**

创新点在于将高阶逻辑作为形式化基础，提供自然语言问题与HOL形式化、可验证推理轨迹、细粒度可控推理因素的完整配对，实现对高阶推理过程而非仅最终答案的诊断评估；

**🔧 技术方法**

采用了Isabelle/HOL进行规则与案例的形式化验证，结合GPT-5.3等自动化工具进行案例生成与自然语言渲染，并使用LLM链式思维与指令式交互等多种提示策略；

**📊 数据集**

使用的主要数据集为HOLMES本身，其中包含300条法律实例和1079条金融实例，涵盖约343条规则，平均推理深度约17，平均上下文长度约1875；

**📈 对比分析**

对11种LLM（9开源、2专有）进行评测，采用最终答案准确率、ROUGE‑L、BERTScore‑F1、ROSCOE等指标；平均准确率仅为50.64%，最高为59.54%，在冲突解消场景中准确率稳定但推理轨迹质量下降，在范围条件和组合推理场景中准确率显著下滑；

**⚠️ 局限性**

局限性在于仅覆盖法律与金融两大高风险领域，未涵盖医学、科学、程序验证等其他高阶推理场景，且当前LLM在高阶推理深度、数值计算与多规则组合上仍表现不佳，需进一步扩展和改进模型与基准。

---

## 666. A Hybrid Intrusion Detection System for Electric Vehicle Charging Infrastructure

**arXiv ID:** 2606.23236 | [PDF](https://arxiv.org/pdf/2606.23236v1)

**作者:** Charukeshi Joglekar `[一作]` (Fraunhofer Institute for Applied Information Technology), Antonello Monti `[通讯]` (Fraunhofer Institute for Applied Information Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种混合式入侵检测系统（Hybrid IDS），同时监测电动汽车充电站（EVCS）的网络层和主机层，实现在单一平台上对多种攻击类型的检测与分类。

**💡 创新点**

创新点在于将网络基IDS与主机基IDS双层融合，利用CICEVSE2024真实数据集实现多类别攻击检测，并通过特征工程与多模型融合显著提升检测覆盖率，弥补了传统单源IDS在覆盖范围和数据适配方面的不足。

**🔧 技术方法**

采用随机森林、XGBoost、LightGBM、决策树、SVM等传统机器学习分类器；对流量级、报文级、主机日志及功率消耗特征进行特征工程、SMOTE欠采样、贝叶斯优化等技术，以提升模型性能。

**📊 数据集**

使用CICEVSE2024数据集，该数据集包含网络流量、报文、主机事件日志和功率消耗等多模态信息，并提供了真实的攻击样本与正常样本。

**📈 对比分析**

通过与已有单源网络IDS、主机IDS以及深度学习方法（如TCN、图对比学习）进行对比，Hybrid IDS在网络层达99.99%准确率，在主机层达到83.47%准确率，整体性能显著优于文献中的现有方案。

**⚠️ 局限性**

局限性包括：主机层检测精度仍低于网络层，尤其在功率消耗数据上的识别效果不佳；缺乏实时在线部署与评估；实验主要基于实验室收集的数据，需进一步验证在真实运营环境中的泛化能力。

---

## 667. Automated Semantic Fault Localization in SysML v2: A Human-in-the-Loop Framework Using Knowledge-Graph Augmented LLMs

**arXiv ID:** 2606.23395 | [PDF](https://arxiv.org/pdf/2606.23395v1)

**作者:** Haitham Al-Shami `[一作]` (Aalto University), Raine Viitala `[通讯]` (Aalto University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种人机协同的框架，用细调的小语言模型结合知识图谱对 SysML v2 模型的语义错误进行定位与修复；

**💡 创新点**

创新点在于通过知识图谱驱动的语义错误注入来生成大规模训练数据，并让模型生成统一差异补丁以降低人工成本；

**🔧 技术方法**

采用了基于 SysML v2 文本语法的语言模型（Qwen‑2.5 Coder 1.5B 与 DeepSeek Coder 6.7B）并结合 LoRA 微调、规则增广（RAG）以及统一 diff 输出；

**📊 数据集**

构建了约 8,301 条带标签的合成数据集，包括 5,497 条语法错误、1,402 条语义错误和 1,402 条正确示例；

**📈 对比分析**

在保留 70:15:15 的训练/验证/测试拆分后，细调模型在语义错误上的 Pass@1 从 0.62% 提升至 95.7%（patch 版）或 91.9%（完整修复），且 patch 输出令 token 长度减少约 50%；

**⚠️ 局限性**

局限在于仅使用有限的 12 种语法与 5 种语义突变器，合成错误可能不完全覆盖真实工程中出现的多样化错误；

---

## 668. Do LLM Embedding Spaces Recover Expert Structure?

**arXiv ID:** 2606.23394 | [PDF](https://arxiv.org/pdf/2606.23394v1)

**作者:** Yixuan Zhu `[一作]` (Zhongnan University of Economics and Law), Fanghen Li `[通讯]` (Zhongnan University of Economics and Law)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究大型语言模型（LLM）嵌入空间在心理健康相关文本中的结构恢复情况，检验其是否能重现专家定义的症状关系而非仅仅实现社区分类。

**💡 创新点**

创新点在于将代表性相似性分析（RSA）、原型典型性与边界分析以及多基准混杂控制相结合，构建了一套层级化、对抗非临床因素的评估框架，并系统验证模型规模和监督对专家结构恢复的级别效应。

**🔧 技术方法**

核心技术包括：预训练与监督微调的 Qwen3 嵌入模型（0.6B 与 4B）、RSA、原型典型性与边界误判指标、Embedded Topic Model（ETM）解释、VAD/LIWC/风格特征以及基于话题的混杂控制。

**📊 数据集**

使用了 28 个 Reddit 子版块的数据集（17 个心理健康子版块与 11 个对照子版块），每个子版块视为一个类别标签，并构建专家症状向量作为外部参考。

**📈 对比分析**

比较方法为在零射与微调两种表示空间下，分别计算模型 RDM 与专家 RDM 的 Spearman 相关（RSA）以及对混杂基准的偏差控制。结果显示：在 4B 模型中，零射 RSA 在 MH 子集为 0.628，微调后提升至 0.763；在 28 类全集上亦提升，但受域分隔影响；在混杂控制下，微调进一步减弱 VAD、LIWC、风格与话题的相关性，残余专家对齐仍显著。

**⚠️ 局限性**

局限包括：专家症状参考仅为二值结构，未覆盖诊断层级或严重度；实验仅针对 Qwen3 两个规模、单一数据集与单一监督任务，未验证跨模型与跨域的一般性；话题解释为描述性而非因果；并且未考虑所有可能的非临床结构源。

---

## 669. Self-Stigma Is Not a Monolith, but Generic Empathy Is: Persona-Conditioned LLM Support for People Who Use Drugs

**arXiv ID:** 2606.23387 | [PDF](https://arxiv.org/pdf/2606.23387v1)

**作者:** Layla Bouzoubaa `[一作]` (Drexel University), Rezvaneh Rezapour `[通讯]` (Drexel University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用Reddit上药物使用者的帖子，对自我污名化表达进行四类人格表型的识别，并研发了基于这些人格的序列化人物恢复模型和个性化对话生成系统；

**💡 创新点**

创新点在于①提出基于LPA的四人设自我污名化表型；②在有限历史下实现了高效的序列化人物恢复，显著优于基线；③通过专家评估验证个性化回应能实现预期行为转变，但整体偏好仍倾向通用共情；

**🔧 技术方法**

主要技术包括LPA层次聚类、特征工程（自我污名化指标、LIWC、时间、参与度）、贝叶斯累积器、GRU/LSTM序列分类器、少样本LLM CoT提示、Persona-Conditioned生成与安全基底；

**📊 数据集**

使用的数据集为72,117条来自r/opiates、r/Stims等子版块的帖子，涉及1,660用户，其中1,228为自我污名化表达者，1,174用于实验；

**📈 对比分析**

与批量、贝叶斯、预训练编码器、少样本LLM等多种基线进行对比，序列化GRU在30条帖子时macro‑F1达0.74，显著优于贝叶斯0.57；专家评估显示个性化回答在项目项上更优，但整体偏好仍倾向通用共情；

**⚠️ 局限性**

局限性包括仅适用于英文Reddit药物使用社区；人设随时间可能变化；需在新平台重新验证；专家样本有限；不适用于临床部署，存在潜在危害。

---

## 670. Partial Automation of Verification Condition Proving for Reflex Programs (Draft)

**arXiv ID:** 2606.23377 | [PDF](https://arxiv.org/pdf/2606.23377v1)

**作者:** Artyom Ishchenko `[一作]` (Novosibirsk State University), Igor Anureev `[通讯]` (Institute of Automation and Electrometry)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对工业级过程控制语言 Reflex 的判定式生成器进行改进，提出了新的注解语言、结构化不变量生成以及 SMT 解决方案集成，以实现对验证条件的自动化证明；

**💡 创新点**

创新点在于三方面：①为 Reflex 设计了更具表达力的注解语言，支持外部逻辑语言；②通过控制流图分析自动生成结构性不变量；③将生成的验证条件和辅助公理翻译为 SMT‑LIB 并交给 SMT 求解器，若求解失败可递归扩展上下文；

**🔧 技术方法**

所用技术包括 ANTLR4 解析 Reflex 代码、构造控制流图、基于 Isabelle/HOL 的逻辑表达、SMT‑LIB 形式转换以及 CVC5 等 SMT 求解器；

**📊 数据集**

文中未提供具体数据集，作者计划在未来实验中使用不同规模的 Reflex 程序集合进行评估；

**📈 对比分析**

性能比较尚未完成，论文仅描述了若干改进策略和理论预期；后续工作需通过实验验证自动证明率提升与求解时间缩短；

**⚠️ 局限性**

局限性包括：对复杂的、需要交互式定理证明的验证条件仍然无法完全自动化；过多的辅助公理可能导致 SMT 求解性能下降；以及缺乏实测数据验证方法有效性。

---

## 671. A kinetic-diffusion Monte Carlo-based particle-level fluid-kinetic decomposition for neutral transport simulations

**arXiv ID:** 2606.23368 | [PDF](https://arxiv.org/pdf/2606.23368v1)

**作者:** Zhirui Tang `[一作]` (KU Leuven), Giovanni Samaey `[通讯]` (KU Leuven)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于KDMC（kinetic–diffusion Monte Carlo）粒子级分布分解的混合流体–粒子模型，用于等离子体边缘中性粒子输运，并在一维测试中验证其精度与效率。

**💡 创新点**

创新点包括：
① 采用KDMC粒子级分布分解，天然实现无迭代耦合的混合模型；
② 推导出适用于KDMC的Navier–Stokes型流体系统（含密度与能量方程），比传统AFN模型包含更多线性项；
③ 设计可调参数α的反射边界处理方案，在保持准确性的同时显著加速计算；
④ 通过粒子轨迹的时空积分直接估计混合模型中流体部分的源项，避免额外的点估计与方差。

**🔧 技术方法**

所使用技术主要有：
- KDMC方法（融合碰撞步与扩散步）;
- Hilbert–Chapman–Enskog展开用于推导流体方程;
- Track‑length估计用于粒子量化;
- 伪时间步进、MUSCL/Monotized Central limiter、Newton迭代求解流体系统。

**📊 数据集**

数据集：
① 取自SOLPS‑ITER的F12等离子体边缘背景（密度、速度、温度、碰撞率等）做一维化；
② 设定周期性平滑案例（恒定粒子密度、可变温度等）作为边界无影响的基准；
粒子数目分别为10^9（参考MC）与10^6/10^5（混合/验证）。

**📈 对比分析**

比较方法与性能：
- 与纯MC、AFN模型、单密度模型及之前的KDMC估计方案做相对L2误差、统计误差对比；
- 在高碰撞率（CX主导）下，混合模型速度提升≥500×，误差≈10%；
- 在低碰撞率/非CX主导时，速度提升仍>100×，误差随反射边界α调节而变化。

**⚠️ 局限性**

局限性：
- 反射边界处理对低碰撞或非CX主导区仍显精度不足，需更物理的边界模型；
- 当前实现仅限1维，推广至二维/三维需推导更复杂的流体闭包与边界条件；
- 流体方程在边界附近的近似仍导致温度预测误差，需要进一步改进。

---

## 672. Convergence of Gradient Descent for General Neural Network Architectures Beyond the NTK Regime

**arXiv ID:** 2606.23364 | [PDF](https://arxiv.org/pdf/2606.23364v1)

**作者:** Yuqing Wang `[一作]` `[通讯]` (Johns Hopkins University), Yuqing Wang (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出了一套针对广泛神经网络架构（包含Transformer、残差网络等）在梯度下降（GD）下的收敛框架，证明在一般非线性训练动态中，GD在合适学习率下会收敛到某个极值点附近。

**💡 创新点**

创新点在于：①将分析聚焦到网络块级别，提供多项“多项式通用连续性/光滑性”与“局部松弛耗散条件”，不再依赖传统的全局Polyak–Łojasiewicz或Lipschitz光滑性；②使用解析性与测度零非退化论证，保证大多数参数初始化下网络雅可比矩阵可达全秩；③得出学习率与网络有效瓶颈维度（而非最大宽度）相关的可行范围。

**🔧 技术方法**

主要技术手段包括：解析性函数理论、测度零集合论证、局部耗散动力学分析、基于块级多项式光滑性推导的梯度下降收敛估计，以及对残差结构的非退化性证明。

**📊 数据集**

论文未针对具体公开数据集进行实验验证，重点在理论证明和数学框架构建。

**📈 对比分析**

由于没有实验比较，无法给出性能指标；理论结果表明在满足条件的网络和初始化下，GD可保证收敛至极值点附近，并在大多数参数设置下无鞍点停滞。

**⚠️ 局限性**

局限性在于：①仅证明收敛到极值点附近，未说明是否全局最优；②假设为全批量GD，未覆盖随机梯度、动量或自适应优化器；③对学习率的“有限异常值”仍需微调；④实际应用需验证多项式光滑性与耗散条件的可行性。

---

## 673. RT-DocLayout: Real-Time End-to-End Document Layout Analysis with Reading Order in the Wild

**arXiv ID:** 2606.23344 | [PDF](https://arxiv.org/pdf/2606.23344v1)

**作者:** Cheng Cui `[一作]` (Baidu Inc.), Yi Liu `[通讯]` (Baidu Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出RT-DocLayout，一种统一的实时文档布局分析框架，能够在单个前向传播中完成分类、检测、像素级分割与阅读顺序预测；

**💡 创新点**

创新点在于将分割与阅读顺序预测融入RT-DETR的查询解码器，实现端到端的多任务学习，并通过物理空间感知的数据增强显著提升对几何失真（扭曲、倾斜、光照不均）的鲁棒性；

**🔧 技术方法**

核心技术包括RT-DETR基础架构、查询式Mask head、对称的阅读顺序对比损失、加权多任务损失、以及两阶段物理空间感知数据增强（表面变形+投影变换）；

**📊 数据集**

训练使用自建38k高质量文档数据集（25类布局元素，包含读取顺序），评估在OmniDocBench v1.5及其真实世界衍生集Real5-OmniDocBench；

**📈 对比分析**

与PP-DocLayoutV2、Dolphin、MinerU等同类模型相比，RT-DocLayout在参数33M、推理速度132.1 FPS的同时，在OmniDocBench整体分数达到94.5%，在Real5-OmniDocBench整体分数92.05%，在Warp和Skew等几何难题上显著优于基线；

**⚠️ 局限性**

局限在于模型仍主要依赖视觉特征，未结合文本特征进行联合预训练；在极端低分辨率或高噪声条件下，分割精度与阅读顺序预测可能受限；

---

## 674. WaveDetect: Robust Framework for Machine-Generated Text Detection via Wavelet Transform

**arXiv ID:** 2606.23336 | [PDF](https://arxiv.org/pdf/2606.23336v1)

**作者:** Zhichen Liu `[一作]` (Southern University of Science and Technology), Yang Xu `[通讯]` (Southern University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种名为 WaveDetect 的机器生成文本检测框架，将文本的概率序列视为连续信号，使用连续小波变换（CWT）得到时频域表示，并用 CNN 提取并分类其光谱指纹，从而识别人类写作与 LLM 生成文本。

**💡 创新点**

创新点在于将文本生成过程建模为信号处理问题，首次利用可微分的 Morlet 小波对概率序列做 CWT，挖掘“光谱指纹”这一隐藏特征；相比传统的语义或统计指标，光谱特征更不易被对抗扰动、模型演进和跨域变化干扰。

**🔧 技术方法**

核心技术包括：1) 代理语言模型（Qwen2.5‑0.5B）输出 token 概率序列；2) 连续小波变换（Morlet wavelet）构造时频谱；3) 轻量级 CNN 对光谱进行特征提取；4) 两阶段训练（冻结/解冻代理模型）和加权交叉熵损失，提升鲁棒性。

**📊 数据集**

主要使用 RAID 数据集（训练/验证/测试），并在 RAID‑all/RAID‑base 两个版本上训练；评估时引入 EvoBench（模型演进）和 Domain‑Shift（医学/法律）等外部数据集，验证跨域与时间稳定性。

**📈 对比分析**

在 RAID‑test 上与 RoBERTa、RADAR、Fast‑DetectGPT、Binoculars、FourierGPT、RepreGuard 等 6 种基线对比，WaveDetect‑all 取得平均 AUROC 0.9785，TPR@0.1%FPR 54.7%，在对抗攻击、LLM 版本演进以及医学/法律跨域任务上均显著优于对照组，体现了更高的准确率与鲁棒性。

**⚠️ 局限性**

局限性包括：1) 仅在相对较老的 RAID 训练数据上训练，未覆盖最新 LLM ；2) 对小波母函数选择、尺度范围等参数的理论解释与调优仍待深入；3) 需要更大规模、更新更迭的多样化检测数据集来进一步验证通用性。

---

## 675. The Watermark Shortcut: How Provenance Marking Sabotages Audio Deepfake Detection

**arXiv ID:** 2606.23335 | [PDF](https://arxiv.org/pdf/2606.23335v1)

**作者:** Nicolas M. Müller `[一作]` (Fraunhofer AISEC), Pascal Debus `[通讯]` (Fraunhofer AISEC)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了音频水印对深度伪造检测的负面影响，揭示水印可成为模型训练时的“shortcut”，导致误检和逃逸攻击；并提出通过在训练时对正负类均加水印来消除这一缺陷；

**💡 创新点**

创新点在于首次系统性量化“watermark ⇒ fake”短路导致的三重失效（泛化下降、去水印逃逸、加水印误判），并通过白盒与黑盒实验验证；

**🔧 技术方法**

技术手段包括：基于AASIST的深度学习检测器、PerTh等多种音频水印技术、对训练集做水印增强、以及通过公共API进行黑盒攻击；

**📊 数据集**

数据集包括ASVspoof2019 LA、ASVspoof2021 LA、In-the-Wild、以及自建的WASP语料库（6款TTS合成+真实语料，四语种），并与多种水印配对；

**📈 对比分析**

比较方法：在训练时对比仅清洗数据与含水印数据的两种检测器，使用EER指标；黑盒实验采用商业API的“fakeness score”评估误判率。性能上，未增强模型在加水印测试中EER降至0.7%，但泛化率显著下降；增强模型恢复到与清洗模型相当的EER（≈12‑15%），并抑制两种攻击；

**⚠️ 局限性**

局限性包括：实验主要聚焦单一水印与单一检测器框架；对不同深度伪造检测架构的普适性待验证；水印的可移除性与伪造性可能随技术进步变化，需持续更新评估。

---

## 676. Tmax: A simple recipe for terminal agents

**arXiv ID:** 2606.23321 | [PDF](https://arxiv.org/pdf/2606.23321v1)

**作者:** Hamish Ivison `[一作]` (Allen Institute for AI), Hannaneh Hajishirzi `[通讯]` (University of Washington)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出了一套可复现的终端代理训练流程，包括规模达14,600个合成RL环境的数据集以及基于DPPO的RL训练方案，训练出9B参数模型在Terminal‑Bench 2.0上获得27%得分；

**💡 创新点**

创新点在于①通过分层采样、难度控制、角色多样化、验证器多样化和多模态任务构造，生成规模大、难度均衡且多样化的数据集；②提供了简单易用的open RL训练recipe，利用DPPO+FP32 LM头实现稳定训练；

**🔧 技术方法**

采用了合成数据管线（Gemini‑3‑Pro生成器、分层采样、软过滤、Docker构建）以及RL训练技术（DPPO、FP32 LM头、大组大小、异步训练）和多种评估基准；

**📊 数据集**

使用了本工作生成的TMax数据集（14,600个RL环境，规模超过前任终端数据集2.5倍）以及2.2k SFT环境，并与Endless Terminals、OpenThinker‑Agent、TerminalTraj、CLI‑Gym、SWE‑Smith等公开数据集进行对比；

**📈 对比分析**

通过在Terminal‑Bench 2.0/2.1、Terminal‑Bench Lite、SWE‑Bench Verified、AIME等基准上评估，并与公开的开源模型对比，9B Qwen 3.5经过RL训练后在Terminal‑Bench 2.0上获得27%得分，显著超越更大规模模型；RL训练亦在SWE‑Bench、AIME等任务提升5+分，跨harness和模型家族均能提升；

**⚠️ 局限性**

限制包括：①生成管线依赖强大生成器，难以超越其性能；②RL训练易不稳定，易崩溃；③需要大量sandboxing资源，成本高；④仅针对open‑weight小模型，未覆盖大规模工业级方法。

---

## 677. From Pixels to Concepts: Growing Rich 3D Semantic Scene Graph Forests utilizing Foundation Models

**arXiv ID:** 2606.23312 | [PDF](https://arxiv.org/pdf/2606.23312v1)

**作者:** David Oberacker `[一作]` (FZI Research Center for Information Technology), Arne Roennau `[通讯]` (Karlsruhe Institute for Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用视觉语言模型（VLM）和大型语言模型（LLM）从RGB‑D图像中自动生成包含开放词汇概念与关系的层次化 3D 场景图森林，并将其用于机器人下游任务。

**💡 创新点**

创新点：①提出“森林”式的 hssg 结构，允许多棵树同时存在，根节点为元概念、叶子为实际物体；②完全开放词汇的概念与关系生成，避免传统固定本体；③通过 VLM 先获取实例属性，再由 LLM 推理生成更高层概念与关系，实现多层次语义扩展；④无硬编码本体，系统可动态扩展。

**🔧 技术方法**

核心技术：YOLO‑E（开放词汇目标检测） + 3D voxel mapping（空间定位）；Qwen3‑VL（视觉语言推理） + Qwen3（思考型 LLM）进行属性与概念生成；Qwen3‑Coder（代码 LLM）用于下游查询；ROS2 框架整合，实现数据流与后处理；对关系三元组采用正则化解析；图结构无循环检验以保证森林结构。

**📊 数据集**

实验数据集：uHumans2（公寓场景），ScanNet（室内扫描），以及真实场景（Boston Dynamics Spot 的室内部署）。

**📈 对比分析**

评估方式：①人工标注验证关系与概念的正确率，整体准确率约 61%（实例属性 60‑70%，知识依赖关系 16‑73%）。②在 ScanNet 与真实场景上进行开放词汇目标检索，平均准确率：Flat‑no‑attr 73.3% / 80%，Flat‑with‑attr 86.6% / 90%，Graph‑Based 70.0% / 100%。与传统仅包含对象类的场景图相比，加入概念节点显著提升检索准确率。

**⚠️ 局限性**

局限性：①知识依赖关系准确率低，受 LLM 推理误差影响；②生成的概念节点存在词形变体导致冗余；③缺乏实时在线处理，当前为离线批处理；④空间关系被 3D voxel 取代，未能利用 VLM 产生的空间关系；⑤图结构扩展易受噪声检测影响，导致循环或不完整。

---

## 678. Measuring & Mitigating Over-Alignment for LLMs in Multilingual Criminal Law Courts

**arXiv ID:** 2606.23375 | [PDF](https://arxiv.org/pdf/2606.23375v1)

**作者:** Arthur Wuhrmann `[一作]` (Surelio.ai), Andrei Kucharavy `[通讯]` (Surelio.ai)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为TF-RefusalBench的多语言刑事法翻译与摘要基准，用于评估大型语言模型在敏感法律文本上的拒绝与免责声明行为。

**💡 创新点**

首次将过度拒绝与免责声明的多语言多任务特征结合在真实法院判例上，提供可复现的分层评估，并提出通过系统提示和abliteration降低过度对齐的实用方法。

**🔧 技术方法**

使用LLM判定、对齐消除（abliteration）以及多语言提示工程，评估5个开源权重模型在翻译与摘要任务上的拒绝与免责声明。

**📊 数据集**

从瑞士联邦最高法院公开判例中筛选100条高严重度摘录，并人工翻译成四种官方语言，生成5200个多语言提示。

**📈 对比分析**

采用三名LLM评审者的多数投票评估拒绝与免责声明率，结果显示模型间差异显著，系统提示可将过度对齐率降低一半，abliteration几乎消除拒绝但略降免责声明。

**⚠️ 局限性**

受限于仅使用LLM评审而非人类专家，数据集中仅聚焦严重性高的性暴力案例且机器翻译质量不一，且abliteration可能削弱模型对有害请求的安全防护。

---

## 679. Non-asymptotic estimates of the minimal risk in statistical learning

**arXiv ID:** 2606.23295 | [PDF](https://arxiv.org/pdf/2606.23295v1)

**作者:** Liming Wu `[一作]` (Universite Clermont Auvergne), Sen Yang `[通讯]` (Harbin Institute Of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文证明了统计学习中经验风险原则（ERP）下两种误差概率的集中不等式，提供了最小风险（以最小经验风险为标准）的下界和上界，具有非渐近的高置信度。

**💡 创新点**

创新点在于放宽了经验风险函数的有界性条件，改为高斯或指数可积性条件，并且证明了最小风险的下界置信度与训练参数数量和输入向量维度无关。

**🔧 技术方法**

使用了Talagrand的集中不等式（Bousquet和Klein-Rio的尖锐版本）、传输-熵不等式以及经验过程和统计学习理论的最新进展。

**📊 数据集**

未具体提及使用的数据集，但讨论了在高维情况下的学习机器和样本大小的关系。

**📈 对比分析**

通过比较最小经验风险与真实最小风险，控制误差概率p_+(n, ε)和p_-(n, ε)，并提供了相应的非渐近估计，性能表现良好。

**⚠️ 局限性**

限制在于经典极限定理（如Donsker的不变原理）在样本大小n不能远大于参数数量N时无法直接应用，且在高维情况下可能导致过拟合。

---

## 680. TSD: A Physics-Inspired Trajectory Saliency Detector for Efficient Imitation Learning

**arXiv ID:** 2606.23371 | [PDF](https://arxiv.org/pdf/2606.23371v1)

**作者:** Yiming Zhao `[一作]` (Tsinghua University), Mingguo Zhao `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种无监督的轨迹显著性检测器（TSD），通过识别机器人操作轨迹中的精细和敏捷段，实现数据集的压缩和扩增，从而提高仿学习的数据收集和训练效率。

**💡 创新点**

创新点在于：①使用物理基础的空间熵与离心加速度两种无训练指标自动检测显著段；②基于检测结果进行信息密集的数据集压缩与高效扩增；③实现了无需额外模型的训练自由度，显著降低了数据处理成本。

**🔧 技术方法**

采用DTW轨迹对齐、KNN估计空间熵、RDP关键点提取、离心加速度阈值检测；使用CNN+扩散策略进行行为学习；通过实验验证TSD的鲁棒性与效率。

**📊 数据集**

模拟数据：Robosuite/Robomimic五个抓取与搬运任务；真实数据：书取、瓶子放置、托盘摆设三种日常单臂/双臂任务，均包含空间随机化。

**📈 对比分析**

与随机采样、完整数据集进行对比；在压缩下模型仅使用约25%数据即可达到或超过完整数据的成功率；在扩增下约80%数据即可与完整数据匹配甚至更优，显示显著提升的数据效率和性能。

**⚠️ 局限性**

局限性：目前仅在单/双臂基本操作中验证，对高自由度机械手、复杂移动操作的适用性待进一步验证；在极端随机化场景下检测稳定性仍有提升空间。

---

## 681. FlexServe: A Fast and Secure LLM Serving System for Mobile Devices with Flexible Resource Isolation

**arXiv ID:** 2606.23370 | [PDF](https://arxiv.org/pdf/2606.23370v1)

**作者:** Yinpeng Wu `[一作]` (Shanghai Jiao Tong University), Yubin Xia `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

在移动设备上设计并实现了一套基于ARM TrustZone的LLM推理系统（Flex‑LLM），通过可回收的安全内存（Recallable Secure Memory）和可回收的安全NPU（Recallable Secure NPU）实现安全推理，同时在正常世界与安全世界之间进行协作式内存管理。

**💡 创新点**

核心创新包括：①将访问权限与管理权限解耦，允许正常世界操作系统高效分配/回收安全资源；②基于Stage‑2页表的可回收安全资源抽象，消除传统连续内存分配瓶颈；③协作式内存管理与动态缓存策略，实现模型权重与KV缓存的高效置换；④两阶段异步回收与即时保护机制，进一步降低延迟；⑤利用内联加解密硬件与流水线并行，提升推理吞吐。

**🔧 技术方法**

技术手段包括：ARM TrustZone（EL3/EL2）、Stage‑2页表与SMMU控制、CMA替代、NPU驱动重用、OP‑TEE安全世界、内联加解密硬件、Linux内存管理（CMA、SMMU、PSI）以及异步回收与两阶段缓存机制。

**📊 数据集**

实验数据集涵盖：Llama3.1（1.7B‑8B）、Qwen3（0.6B‑8B）等模型；多模型代理工作流 benchmark（UltraChat、OpenAssistant、Dolly、Alpaca）；以及在NanoPC‑T6上使用的8GB RAM、6TOPS NPU、SSD存储。

**📈 对比分析**

与两种基线（传统TrustZone的Strawman和启用流水线+NPU的Strawman‑OPT）以及未加密的正常世界推理（NW‑Base）进行对比；实验结果表明：平均TTFT提升10.05×（相较Strawman）和2.44×（相较Strawman‑OPT），多模型/代理场景下可达24×提升；正常世界应用性能保持97.2%不变；在高内存压力下TTFT仍保持低延迟。

**⚠️ 局限性**

局限性：①无法保护正常世界客户端的输入/输出（仅保护模型权重和KV缓存）；②不具备防御物理攻击、侧信道攻击或DoS攻击的能力；③实验平台缺乏内联加密硬件，导致相关加速效果未验证；④对特定硬件（NPU、TrustZone特性）依赖较高，迁移性有限。

---

## 682. Group Selection Promotes Prosocial Prompts in Populations of LLM Agents

**arXiv ID:** 2606.23343 | [PDF](https://arxiv.org/pdf/2606.23343v1)

**作者:** Luis Celiktemel `[一作]` (Max Planck Institute for Human Development), Iyad Rahwan `[通讯]` (Max Planck Institute for Human Development)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

在多代理模拟框架下，研究了群体选择对大型语言模型（LLM）代理群体合作行为的影响。

**💡 创新点**

首次证明通过多级选择机制而非个体奖励可促使LLM群体形成稳定的合作。

**🔧 技术方法**

结合自然语言策略提示的进化复制‑突变模型与基于donor游戏的多代理仿真。

**📊 数据集**

使用公开的LLM（Qwen3‑30B、Llama 3 70B、GPT 5.5）作为策略生成与传递的基础。

**📈 对比分析**

通过在个体与群体选择两种机制下对比平均捐赠率，实验显示群体选择可使合作率提升至约0.48，并呈现相位转移。

**⚠️ 局限性**

局限于单一donor游戏、特定提示设计、突变率影响以及模型间转移矩阵差异。

---

## 683. Affective AI Safety: The Missing Piece in LLM Safety

**arXiv ID:** 2606.23380 | [PDF](https://arxiv.org/pdf/2606.23380v1)

**作者:** Carolin Ifländer `[一作]` (Independent Researcher), Amanda Cercas Curry `[通讯]` (Independent Researcher)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出情感安全（affective safety）的概念，并构建了情感危害的分类体系，涵盖自我异化、公平与偏见、关系三大类型。

**💡 创新点**

首次系统化将情感危害与AI系统的交互方式、时序维度和受影响层级关联，形成理论框架，为后续技术与监管研究指明方向。

**🔧 技术方法**

主要采用理论分析、文献综述与案例研究的方法，并未使用具体模型训练或算法实现。

**📊 数据集**

未引入公开数据集，而是引用已有研究与报告中的案例（如聊天机器人、推荐系统等）的数据与结果。

**📈 对比分析**

无实验性对比；通过与现行监管框架（如欧盟AI法、人工智能交互服务管理临时措施）的差距对照，说明情感安全缺失的现实意义。

**⚠️ 局限性**

局限在缺乏可量化指标和可验证的评估方法，缺少对技术实现细节的深入探讨，因而难以在实践中立即应用。

---

## 684. Faithful Grounded Visual Reasoning via Learned Proxy-Tokens

**arXiv ID:** 2606.23354 | [PDF](https://arxiv.org/pdf/2606.23354v1)

**作者:** Tom Hodemon `[一作]` (Université Paris-Saclay), Angelique Loesch `[通讯]` (Université Paris-Saclay)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了Composer MLLM并提出基于学习代理代币的视觉定位机制，构建ComposerGCoT数据集用于全流程评估。

**💡 创新点**

用离散代理代币直接索引图像潜在空间，实现可学习的语义-空间连接，解决传统坐标回归导致的语义空间缺口。

**🔧 技术方法**

采用CLIP ViT‑L/16视觉编码器、Vicuna‑v1.5‑7B语言模型、两层MLP投影器，并将代理代币与视觉代币交错输入，同时使用结构化XML输出进行推理链验证。

**📊 数据集**

在对齐预训练阶段使用CC3M、Flickr30K、COCO、RefCOCO、Visual Genome等；在多步推理阶段使用自制的ComposerGCoT（基于GQA）共163K条逻辑链，验证集3,959条。

**📈 对比分析**

与坐标回归版本B‑Composer对比，答案准确率基本相同，代理代币版P‑Composer在IoU@0.95上提升9.0分，整体IoU提升2.7分，推理路径一致性与视觉能力表现均优于B‑Composer。

**⚠️ 局限性**

代理代币生成过程略增加噪声，导致部分推理步骤误差；颜色属性识别准确率仍低于空间与定位任务；模型在更复杂视觉场景与开放式答案上需进一步验证。

---

## 685. When Robots Rate Their Own Interactions: Engagement Validity and the Strangeness Failure

**arXiv ID:** 2606.23339 | [PDF](https://arxiv.org/pdf/2606.23339v1)

**作者:** Victor Lockwood `[一作]` (Rochester Institute of Technology), Jamison Heard `[通讯]` (Rochester Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对LLM驱动机器人在HRI交互中的自评进行系统评估，使用标准问卷逆向填报并与人类自评进行比较

**💡 创新点**

提出并验证“反向评估”框架，揭示LLM在舒适/陌生感维度上的系统性失效，提示需补充多模态感知

**🔧 技术方法**

运用GPT-3.5、GPT-4o-mini、GPT-4o、Claude Sonnet及Claude Haiku等LLM，结合Prompt、文本转录、情感标签和面部图像等多模态输入

**📊 数据集**

HRI-CUES老年人–Furhat机器人交互数据集（25人、1,522次评估）及Nao机器人现场实验（4人）

**📈 对比分析**

使用Pearson相关、ICC、Bland-Altman等统计方法比较，LLM在参与度维度达到中等-强相关（r≤.72），但在陌生感上出现负相关（r≈-0.5），多模态改进有限

**⚠️ 局限性**

局限包括单一数据集、翻译可能失真、样本量小、缺少更完整的舒适/不适量表、LLM缺乏内部情感感知、仅验证单一“陌生感”项

---

## 686. Generate with CodeXHug: A Dataset to Enhance Model Cards with Code Usage Patterns

**arXiv ID:** 2606.23329 | [PDF](https://arxiv.org/pdf/2606.23329v1)

**作者:** Stefano Palombo `[一作]` (University of l'Aquila), Davide Di Ruscio `[通讯]` (University of l'Aquila)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个名为 PTM-Usage 数据集，将 Hugging Face 上最受欢迎的预训练模型与 GitHub 上实际使用这些模型的 Python 代码片段进行关联，并演示了通过聚类与 LLM 自动提取代码使用模式的流程。

**💡 创新点**

创新点包括①构建首个将模型卡与实际项目代码映射的数据集；②利用 K‑means 聚类与 sentence‑transformers 提取模型使用模式；③通过 Llama 3 零 shot 生成可直接使用的代码示例，提升模型卡的可操作性。

**🔧 技术方法**

使用的数据清洗与筛选技术、PyGithub API、MongoDB 存储、K‑means 聚类、sentence‑transformers（all‑MiniLM‑L6‑v2）嵌入、Llama 3 LLM 零 shot 生成代码。

**📊 数据集**

核心数据集来自 2024 年 6 月的 Hugging Face dump（包含模型、标签、卡片、下载量）以及 GitHub 上使用这些模型的 Python 仓库（文件内容、元数据）。

**📈 对比分析**

通过统计分析（文件数、模型受欢迎度、项目分布）展示数据分布；聚类后挑选代表样本并用 Llama 3 生成代码示例，虽然未给出传统基准，但结果显示分布更平衡、生成的代码模式与实际使用更一致。

**⚠️ 局限性**

局限性包括：长尾分布导致样本不均衡，聚类与 LLM 可能产生错误或不完整的模式；未覆盖所有模型类别，缺乏人工评估或用户研究；数据不包含最新发布的模型，可能遗漏新兴用法。

---

## 687. Mixed Voting Rules for Participatory Budgeting

**arXiv ID:** 2606.23320 | [PDF](https://arxiv.org/pdf/2606.23320v1)

**作者:** Anton Baychkov `[一作]` (University of Warwick), Markus Utke `[通讯]` (TU Eindhoven)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了混合投票规则框架，将参与式预算（PB）规则按顺序组合，并为已有项目集设计了适配的MES和Greedy等规则；

**💡 创新点**

创新点包括：①将预算按份额分配构造类似混合成员选举系统的PB规则；②设计四种前置分配方法，其中Value‑Based方法在满足α‑EJR+上突破理论基线；③给出参数化的比例性保证和理论证明；④在实验中证明Greedy与MES混合在效用与比例性上均优于单独规则；

**🔧 技术方法**

使用的技术包括：计算社会选择理论与PB规则（Greedy、MES、BOS等）的改写；前置分配与重平衡步骤；参数化EJR+和Strong EJR+的理论分析；实验评估利用Pabulib实例集，计算效用比、α‑预算EJR+满足度等指标；

**📊 数据集**

使用Pabulib公共预算实例集，共313个非平凡实例（至少20个项目）；

**📈 对比分析**

通过比较不同混合规则（Greedy+MES、不同前置分配方法）与单一规则，评估效用比和α‑预算EJR+满足度。实验显示，当Greedy占60–90%预算时，混合规则在效用与比例性上均优于单独规则；Value‑Based前置分配在理论上最优，实验表现略逊于MES‑Style但仍显著优于Null；

**⚠️ 局限性**

局限性包括：①在小规模实例中预算分割导致子优化子问题；②仅评估基于EJR+的比例性，对覆盖率等其他目标不充分；③Value‑Based方法最优性仅在特定情形，未证明在所有满意度函数和预选集上；④多winner和通用满足函数的扩展尚未完整实验验证。

---

## 688. Ocean4D: Generative Underwater 4D Reconstruction via Medium-Aware Video Diffusion

**arXiv ID:** 2606.23298 | [PDF](https://arxiv.org/pdf/2606.23298v1)

**作者:** Yuqiang Huang `[一作]` (Ocean University of China), Zhaoxiang Zhang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Ocean4D，一个基于视频扩散模型的生成式 4D 水下重建框架，解决单目视频的几何一致性与介质影响问题。

**💡 创新点**

创新点在于①4D‑GCC 构造四维几何一致的条件，实现跨帧几何补全；②Medium‑Aware Block 在潜在空间隐式建模吸收与散射，提升水下视觉稳定性。

**🔧 技术方法**

采用 V‑DPM 点云预测、Video Diffusion（DiT+Medium‑Aware Block）、VAE 编码、BLIP/T5 文本编码以及经典光学水下成像模型。

**📊 数据集**

训练使用 UVEB（静态水下）与 ReCamMaster（非水下多相机），评估使用 UVEB 动态子集、NUSR 与 DRUVA 静态水下基准。

**📈 对比分析**

与 3DGS、Water‑Splatting、AnySplat、TrajectoryCrafter、ReCamMaster 等基线对比，在 VBench、PSNR/SSIM/LPIPS 等指标上均超越对手，尤其在动态场景的几何一致性与视觉质量方面显著提升。

**⚠️ 局限性**

局限在于对极端海洋条件（强光斑、深海）仍可能出现细节丢失；需要大量带姿势标注的单目视频进行训练，非水下场景的迁移性能有限。

---

## 689. SOAP-Bubbles: Structured Weight Uncertainty for Neural Networks

**arXiv ID:** 2606.23357 | [PDF](https://arxiv.org/pdf/2606.23357v1)

**作者:** Adrian Robert Minut `[一作]` (Sapienza University of Rome), Thomas Möllenhoff `[通讯]` (RIKEN Center for Advanced Intelligence Project)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了EVON算法，将SOAP预条件器与IVON变分推断相结合，生成结构化的SOAP‑Bubble后验，从而实现大规模网络的非对角高斯权重不确定性估计。

**💡 创新点**

创新点在于：1) 用SOAP预条件器将对角协方差旋转为非对角协方差；2) 仅对SOAP做极简改动即可完成变分优化；3) 在线性/逻辑回归上可得到精确后验；4) 在大规模语言模型训练中保持可扩展性；5) 给出可实现精确恢复的理论条件。

**🔧 技术方法**

使用的技术包括：变分贝叶斯推断、均值场高斯后验、SOAP与Shampoo预条件器、IVON、Kronecker‑factored 预条件、特征子空间（eigenspace）变换、Hessian 估计、梯度裁剪（元素级/谱级）、分布式训练、QR 近似特征分解、贝叶斯模型平均（BMA）。

**📊 数据集**

实验数据集有：USPS 数字分类（二分类逻辑回归）、UCI Iris（多分类逻辑回归）、NanoGPT 预训练数据 FineWeb‑1B、LLaMA 预训练数据 C4、CLIP ViT‑B/16 在 8 个视觉分类基准上的微调。

**📈 对比分析**

对比方法包括 AdamW、SOAP、IVON；在语言模型预训练中，EVON 在验证损失、困惑度（PPL）上均优于 IVON 与 SOAP，且比 SOAP 更低；在 CLIP 微调时，EVON 在准确率、NLL、ECE、Brier 分数上均与最佳优化器持平或更优；贝叶斯模型平均时，EVON 的性能提升大于 IVON。整体显示 EVON 在优化动态、最终性能和不确定性估计上均有显著提升。

**⚠️ 局限性**

局限性：需要在分布式环境中精细实现（噪声聚合、Hessian 估计等），额外超参数 ζ 需细致调优；依赖 SOAP/Second‑Order 预条件器，若此类优化器不可用或效率低，EVON 的优势受限；目前对超大模型的完整可扩展性及最佳 ζ 的经验法则仍待研究。

---

## 690. Polynomial Dice Loss for Medical Image Segmentation

**arXiv ID:** 2606.23373 | [PDF](https://arxiv.org/pdf/2606.23373v1)

**作者:** Hiroaki Aizawa `[一作]` `[通讯]` (Hiroshima University), Hiroaki Aizawa (Hiroshima University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估了一种基于Dice损失的多项式改进方法——DropDice和PolyDice-1，用于医学图像分割。

**💡 创新点**

创新点在于通过对Dice损失的几何形式做泰勒展开，将其转化为可调高阶多项式，提供对角度误差惩罚的可控形状。

**🔧 技术方法**

使用了多项式展开、梯度裁剪、UNet和TransUNet模型以及软最大/软阈值预测。

**📊 数据集**

在四个公开基准（CVC-ClinicDB、Kvasir-SEG、ACDC、Synapse）上进行实验。

**📈 对比分析**

与Dice、Cross-Entropy、Tversky及其变体对比，DropDice和PolyDice-1在多类别、前景稀疏场景下表现相当或略优，尤其在多类别任务中提升了Dice分数。

**⚠️ 局限性**

主要限制是所有实验均在2D切片上进行，对3D原生数据的评估留待未来工作。

---

## 691. Adaptive Hard-Soft Physics-Informed Neural Networks for Robust Boundary-Constrained PDE Solving

**arXiv ID:** 2606.23359 | [PDF](https://arxiv.org/pdf/2606.23359v1)

**作者:** Duc Tien Nguyen `[一作]` (VinUniversity), Dinh Gia Ninh `[通讯]` (Hanoi University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种统一的硬软混合物理信息神经网络（HSPINN），通过解析提升与掩码等技术实现Dirichlet及周期边界的严格满足，剩余的PDE、Neumann与初值约束采用软残差最小化；

**💡 创新点**

核心创新在于将Dirichlet（及周期）边界硬编码进网络结构，消除软惩罚导致的梯度失衡与收敛慢问题，同时引入逆共享softmax自适应权重动态平衡多重残差；

**🔧 技术方法**

使用自动微分求导、硬边界提升/掩码、傅里叶特征映射、逆共享softmax权重、两阶段优化（Adam+LBFGS）等技术；

**📊 数据集**

实验仅采用三类典型PDE（Poisson、Burgers、纯输运）及其对应的解析解作为基准，无需外部数据集；

**📈 对比分析**

与传统全软PINN（SPINN）在相同网络规模与优化器下对比，HSPINN在L₂误差上提升1–2个数量级，收敛速度提升约1.3–3.2倍，训练时间显著下降；

**⚠️ 局限性**

局限包括仍需手工设计提升/掩码，无法直接应用于高度不规则域或多物理耦合系统；对Neumann/初值约束仍采用软策略，可能在极端激波或高阶微分场景下收敛受限。

---

## 692. A Relaxed Quadratic-Program-based Framework for Trajectory Tracking of Unicycle Robots with Singularity Avoidance

**arXiv ID:** 2606.23355 | [PDF](https://arxiv.org/pdf/2606.23355v1)

**作者:** Hamza Tariq `[一作]` (New Jersey Institute of Technology), Adeel Akhtar `[通讯]` (New Jersey Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于松弛二次规划的动态反馈线性化（DFL）框架，用于无车轮机器人在零速度时进行轨迹跟踪并避免奇异性；

**💡 创新点**

创新点在于通过在DFL约束中加入松弛变量和加速度偏置，使得控制器在零速度时仍保持可行、时间不变且局部Lipschitz连续，并且首次将DFL奇异性问题用QP松弛方式解决；

**🔧 技术方法**

采用了动态反馈线性化、二次规划（QP）与松弛技术、Lipschitz连续性与UUB理论分析，以及ROS2–Gazebo仿真平台；

**📊 数据集**

未使用公开数据集，而是自行设计半八字形和振荡直线两种轨迹，在TurtleBot3模拟器中进行验证；

**📈 对比分析**

与传统DFL控制器（含速度重置）在相同轨迹下进行对比，仿真结果显示该方法在零速度切换点误差更小、能够顺利通过停靠反转，整体跟踪误差显著降低；

**⚠️ 局限性**

局限性包括需手动调节参数p、ϵ_a以避免死锁；假设参考轨迹可保持在非死锁集合外；仅在仿真环境验证，缺乏硬件实验；未对受限QP的稳定性进行完整分析。

---

## 693. Superhuman AI for Generals.io Using Self-Play Reinforcement Learning

**arXiv ID:** 2606.23348 | [PDF](https://arxiv.org/pdf/2606.23348v1)

**作者:** Matej Straka `[一作]` (Charles University), Martin Schmid `[通讯]` (Charles University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

训练了一个基于自对弈的Transformer策略，使用PPO和稀疏胜/负奖励，产生了在Generals.io 1v1模式中超人级的AI；

**💡 创新点**

将JAX实现的超高速模拟器（10,000×提升）与仅用稀疏奖励的纯策略梯度训练相结合，并发现参数EMA与top-advantage过滤是关键加速和性能提升因素；

**🔧 技术方法**

采用JAX-native环境、PPO、Transformer网络、EMA、top-advantage过滤、spawn-distance curriculum等技术；

**📊 数据集**

训练数据完全来自自对弈产生的游戏模拟，没有使用人类回放或外部数据集；

**📈 对比分析**

与当前最强人工玩家及两大已有AI（heuristic agent、learned agent）在公共1v1排行榜及对战赛中对比，取得81.5%胜率，排名第一，分别以27–12、172–58和20–0战胜顶尖人类与两强bot；

**⚠️ 局限性**

仅在1v1零和模式验证，未测试多玩家/非零和；对策略多样性和可解释性研究有限；对超参数的敏感度评估不充分；在单GPU上训练，未探讨更大规模或多任务扩展。

---

## 694. When Staking Rewards Compound: Measuring the Impact of Ethereum's Pectra Upgrade

**arXiv ID:** 2606.23337 | [PDF](https://arxiv.org/pdf/2606.23337v1)

**作者:** Mohammed Benseddik `[一作]` (University of Zurich), Claudio J. Tessone `[通讯]` (University of Zurich)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文评估了EIP‑7251推出的compounding validator对奖励、合并和迁移的影响，并对其采纳情况进行实证分析。

**💡 创新点**

创新点在于首次系统地量化compounding对不同规模staker的CL APR提升，并将模拟结果与链上实际数据进行对比，验证其三维收益（奖励复利、提议概率提升、合并成本降低）。

**🔧 技术方法**

使用蒙特卡罗模拟、统计检验（Mann‑Whitney U）、区块链数据抽取与分析工具（Dune、Beaconcha.in、Etherscan）等技术。

**📊 数据集**

数据集包括Dune Analytics上的validator标签与staking流、Beaconcha.in的Consensus‑Layer奖励、Etherscan的withdrawal与合并交易记录。

**📈 对比分析**

通过将模拟APR与链上30天、335天窗口的median APR进行对比，并采用Mann‑Whitney U检验验证显著性，发现compounding在小额validator中约+4.7%相对提升，整体上在11个月内仅产生约+1.5%相对提升。

**⚠️ 局限性**

局限性包括模拟假设固定网络总Stake、忽略提议奖励、观测窗口短导致复利尚未完全显现，以及依赖Dune标签可能存在的分类误差。

---

## 695. Uncertainty-based Debiasing and Unlearning for Decontamination

**arXiv ID:** 2606.23313 | [PDF](https://arxiv.org/pdf/2606.23313v1)

**作者:** Guangzhi Sun `[一作]` (University of Cambridge), Mark Gales `[通讯]` (University of Cambridge)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于样本级别的去污染评估框架，并提出了利用深度集成估计的不确定性进行样本级去污染的方法（UBD），包括去偏与再学习两种实现方式。

**💡 创新点**

创新点在于：①引入分布距离指标（KL、L1*）评估去污染效果；②使用深度集成的标准差/知识不确定性作为无监督的污染程度估计，区分“难记忆”与“易记忆”样本；③无需干净模型或训练集信息即可完成去污染。

**🔧 技术方法**

技术上采用了深度集成（LoRA微调后不同随机种子产生的模型集合），利用标准差或知识不确定性估计α̂；然后通过对置信度的缩放实现去偏，或将去偏分布作为软目标进行再学习。

**📊 数据集**

实验使用MMLU‑pro和MATH‑MCQA这两个多选题基准，模型为Llama‑3.2‑3B‑Instruct和Qwen2.5‑3B‑Instruct。

**📈 对比分析**

与基线（词汇重排、改写、DeconIEP等）对比，UBD在样本级KL和L1*上提升40%‑60%，同时对未污染数据的干扰小；在整体准确率上也能保持或略降，证明去污染更稳健。

**⚠️ 局限性**

主要局限是需要训练并发布模型集成，单模型无法直接得到不确定性估计；目前仅适用于多选/二分类任务，需扩展到开放式生成。

---

## 696. The Anatomy of the CTC Oracle Gap: Acoustic Exhaustion and Linguistic Recovery

**arXiv ID:** 2606.23306 | [PDF](https://arxiv.org/pdf/2606.23306v1)

**作者:** Ivan Novosad `[一作]` `[通讯]` (HSE University), Ivan Novosad (HSE University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了CTC模型内部评分在N-best候选选择中的极限，并提出通过将外部语言模型伪对数似然（PLL）与最小贝叶斯风险（MBR）结合的解码策略，以弥合Greedy解码与Oracle解码之间的性能差距。

**💡 创新点**

创新点在于：①系统验证了CTC内部评分（包括各种基于后验、蒙特卡罗、对比解码等）在近似收敛的CTC检查点上无法显著提升WER；②发现此瓶颈在于语言层面而非声学层面；③首次将RoBERTa的PLL作为后验输入MBBR，获得9%相对WER下降，并证明该方法在多种模型、语料和噪声条件下无需调参即可稳健提升。

**🔧 技术方法**

技术手段包括：CTC前向后向算法、Rao‑Blackwell化的MWER梯度、k2 lattice采样生成N-best、RoBERTa PLL评分、MBR期望CER优化、温度调校、统计显著性检验（配对bootstrap）以及对比实验设计。

**📊 数据集**

主要数据集：LibriSpeech（train‑clean‑100/960、dev‑clean、dev‑other、test‑other）、TED‑LIUM3、VoxPopuli 以及对LibriSpeech添加MUSAN噪声的四个SNR水平；实验还覆盖两种Zipformer架构（S与M）。

**📈 对比分析**

与Greedy解码相比，在LibriSpeech test‑other上，使用G=128、τ=10、CER损失的MBR‑PLL解码将WER从5.96%降至5.42%（-0.535pp，p<0.0001，-9%相对）。同一策略在其他12个条件下亦显著提升（11/13显著），平均削减约13–20% oracle gap；仅在VoxPopuli（覆盖瓶颈）和MUSAN 0dB（候选质量下降）两种极端条件下无显著改进。

**⚠️ 局限性**

局限性包括：①在极度噪声或覆盖率极低的条件下候选集缺乏多样性，导致MBR无法进一步改进；②方法依赖高质量的PLL评分，若使用较弱的LM或不同语言模型可能效果不佳；③在训练时间对近似收敛CTC模型的序列级微调始终失败，表明当前的RL/MWER框架对已收敛模型不具备通用可行性。

---

## 697. EHR-Complex: Benchmarking Medical Agents for Complex Clinical Reasoning

**arXiv ID:** 2606.23301 | [PDF](https://arxiv.org/pdf/2606.23301v1)

**作者:** Yitong Qiao `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出并构建了EHR-Complex基准，用于评估大型语言模型在交互式电子健康记录数据库推理中的表现。

**💡 创新点**

创新点在于：①基于完整MIMIC-IV数据库生成52K多轮任务，覆盖患者级与人群级查询；②任务包含平均31.93个SQL结构组件，真实反映多表纵向聚合与组合推理；③使用可执行环境捕捉交互失败，揭示LLM的具体弱点。

**🔧 技术方法**

使用的技术包括：构建患者事件图、临床证据路径抽样、模板化SQL生成与执行验证、交互式Agent轨迹记录与SFT微调；模型评估采用多轮执行、exact-match、Pass@k与一致性指标。

**📊 数据集**

数据集为MIMIC-IV v3.1（365K患者，31表>5亿记录），EHR-Complex在此基础上生成训练集48,092条、测试集3,915条任务。

**📈 对比分析**

与12款前沿LLM（包括GPT-4o、Qwen3.5-397B等）对比，最优模型精度约62.3%，患者级别最高可达≈85%，人群级别仅≈40%；多轮执行揭示模型稳定性差，Pass@k提升但一致性下降。

**⚠️ 局限性**

局限性包括：仅基于单一医院MIMIC-IV数据，可能无法迁移至其他EHR系统；未涵盖自由文本、影像或实时临床交互；任务聚焦结构化查询，未覆盖非结构化信息。

---

## 698. URecJPQ: Memory-efficient Multimodal Recommendation Models through RecJPQ in Large-Scale Scenarios

**arXiv ID:** 2606.23291 | [PDF](https://arxiv.org/pdf/2606.23291v1)

**作者:** Giuseppe Spillo `[一作]` (University of Bari Aldo Moro), Iadh Ounis `[通讯]` (University of Glasgow)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `fede83ac-7505-405f-ab37-e7284695c47f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种URecJPQ方法，将联合产品量化(JPQ)扩展到top-k推荐场景，压缩用户和物品ID嵌入以降低内存占用。

**💡 创新点**

创新点在于同时对用户和物品ID进行量化，实现了端到端训练的可插拔压缩方案，并在多模态推荐中首次验证其效果。

**🔧 技术方法**

采用了联合产品量化技术、SVD/随机码分配、BPR/VBPR/SLMRec/MMGCL等基线模型进行实验，并使用PCA可视化嵌入空间。

**📊 数据集**

使用了三大公开数据集：Baby23、Sports23（Amazon 2023版本）和ML1M，均包含文本和视觉多模态特征。

**📈 对比分析**

与原始完整嵌入模型及其他量化方法比较，URecJPQ将可训练参数减少高达99%、检查点大小减少98%，在绝大多数情况下保持或略提升召回率和NDCG，只有在极度稀疏数据上有轻微性能下降。

**⚠️ 局限性**

局限性包括在极稀疏场景下可能导致召回率下降、未对多模态特征进行压缩，以及仅在少数推荐模型上验证，需要进一步扩展至更多模型和任务。

---

## 699. Examining AI-generated historical narratives and their reception through the example of history POVs on TikTok

**arXiv ID:** 2606.23300 | [PDF](https://arxiv.org/pdf/2606.23300v1)

**作者:** Nina Brolich `[一作]` (University of Erfurt), Anna Neovesky `[通讯]` (University of Applied Sciences Erfurt)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了TikTok上AI生成的历史POV视频趋势，构建了大规模英文数据集，对比黑死病与大屠杀视频的主题与评论反应，并评估TikTok Research API与手工采集方法的可行性。

**💡 创新点**

采用两阶段实证方法，结合手工标注、DistilBERT分类和情感分析，首次量化主题对受众评论及仇恨言论的影响；同时探讨API在快速变迁平台研究中的优势与局限。

**🔧 技术方法**

使用Python进行API调用与数据清洗，langdetect/fastText做语言检测，VADER进行情感分析，DistilBERT微调用于评论标签与AI/仇恨标记分类，spaCy+PyCountry+GeoText+Nominatim做实体与地理分析，BERTopic用于主题建模。

**📊 数据集**

1) 210条手工收集的视频样本；2) 5,565条英文历史POV视频的API元数据；3) 黑死病视频16,390条评论，大屠杀5,855条评论；4) 采样后各1,000条评论用于模型训练；5) 2,000条手工标注的多维标签评论。

**📈 对比分析**

对比手工采样与API采样在主题识别、评论情感与标签分布上的差异；通过DistilBERT训练的多分类模型，标签准确率约80%，AI标签准确率87‑89%，仇恨标签准确率94‑100%，但F1分数在仇恨类别上较低；主题专属模型表现略好但受样本量限制。

**⚠️ 局限性**

数据受限于英文和欧美中心，API排除18岁以下与非授权地区用户，缺乏视频内容获取导致无法分析视觉误差；评论语义模糊、样本量小影响模型性能；手工标注单人导致可能偏差；API不完整、更新滞后导致潜在漏检。

---

## 700. Distribution-Aware Diffusion-LLM for Robust Ultra-Long-Term Time Series Forecasting

**arXiv ID:** 2606.23391 | [PDF](https://arxiv.org/pdf/2606.23391v1)

**作者:** Falguni Ghosh `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Bernhard Kainz `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种新的框架，将条件扩散模型集成到基于大型语言模型（LLM）的时间序列预测管道中，以解决LLM在多模态设置中的挑战。

**💡 创新点**

创新点在于引入了条件扩散模型作为隐式正则化器，增强了多模态对齐和条件分布建模的能力，从而提高了超长期和少样本预测的性能。

**🔧 技术方法**

使用了去噪扩散概率模型（DDPM）作为正则化器，并结合了LLM的推理能力。

**📊 数据集**

在六个长期预测基准上进行了评估，包括ETT、天气和ECL数据集。

**📈 对比分析**

与现有的LLM基线方法相比，提出的方法在多个基准上表现出显著的性能提升，尤其是在超长期和少样本预测中，显示出分布感知正则化的价值。

**⚠️ 局限性**

限制在于尽管方法在超长期预测中表现良好，但在短期预测中可能会略微降低点预测的准确性。

---

## 701. Energy-Based Transformers as Predictors of Reading Difficulty

**arXiv ID:** 2606.23382 | [PDF](https://arxiv.org/pdf/2606.23382v1)

**作者:** Jakub Dotlacil `[一作]`, Ece Takmaz `[通讯]` (Utrecht University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估NRGPT能量函数作为预测阅读难度的新指标

**💡 创新点**

首次将能量基变压器与句子加工相结合，提出能量可作为统一的加工负荷预测器

**🔧 技术方法**

使用能量基变压器NRGPT，计算能量、预期注意得分、注意熵以及传统的惊讶度，并通过线性混合效应模型进行统计

**📊 数据集**

自然故事（Natural Stories）语料、UCL眼动与自我节奏阅读语料以及对比主/宾关系的句子对

**📈 对比分析**

与GPT‑2惊讶度和注意熵对照，采用线性混合模型比较对数似然；在UCL数据中能量（尤其第5层）显著优于惊讶度，而在NSC数据中惊讶度更好；能量仍能在加入惊讶度后保持显著

**⚠️ 局限性**

仅评估单一模型与两训练检查点，缺乏对不同规模、架构的普适性；仅使用阅读时间，未涉及神经或其他行为指标；只检验能量与惊讶度的简单分解，未探讨其他能量景观特征

---

## 702. Abstract representational geometry supports inference in large language models

**arXiv ID:** 2606.23345 | [PDF](https://arxiv.org/pdf/2606.23345v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 703. Exact and Fast Subset Selection Algorithms for the Bi-objective Integral R2 Indicator

**arXiv ID:** 2606.23365 | [PDF](https://arxiv.org/pdf/2606.23365v1)

**作者:** Michael T. M. Emmerich `[一作]` `[通讯]` (University of Jyvaskyla), Michael T. M. Emmerich (University of Jyvaskyla)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了二目标精确积分R₂指标下的固定卡点子集选择问题，提出了邻接-邻居分解并基于该分解构造了精确的Bellman动态规划算法。

**💡 创新点**

创新点在于：①将连续积分R₂的值拆解为边界项、一次性对角校正和相邻点之间的转移项；②证明转移矩阵满足Monge性质，从而实现了O(knlog n)的分治算法和O(kn)的阶梯矩阵搜索算法；③提供完整可复现的Python实现与实验验证。

**🔧 技术方法**

使用的技术包括：Tchebycheff阴影与连续积分R₂理论、Monge矩阵与单调前驱性质、分治与SMAWK启发的阶梯矩阵搜索、以及Python的数值实现与一致性检查。

**📊 数据集**

实验数据集：7点阶梯实例、20点图形实例，以及多组均衡卡点（k = n/2）和固定卡点（k = 6）随机/确定性Pareto前沿，n范围从8到100，全部用于CPU时间与结果一致性评估。

**📈 对比分析**

比较方法：对比穷举枚举、直接Bellman DP、分治DP和矩阵搜索实现；实验显示DP实现均在毫秒级，尤其是矩阵搜索最快；与穷举枚举相比，时间从指数级降至线性/多项式级；在卡点固定时，DP复杂度为O(nᵏ)，而DP实现只需O(kn)或O(knlog n)。

**⚠️ 局限性**

局限性：仅适用于二目标、连续积分R₂；依赖已排序的Pareto前沿；无法直接推广到三及以上目标或其他非Tchebycheff尺度；需要常数时间算术比较，对数值误差与前沿噪声的鲁棒性未作深入讨论。

---

## 704. TooBad: Backdoor Diffusion Models with Ultra-Low Poison Rate and Imperceptible Trigger

**arXiv ID:** 2606.23362 | [PDF](https://arxiv.org/pdf/2606.23362v1)

**作者:** Vu Tuan Truong `[一作]` (INRS), Long Bao Le `[通讯]` (INRS)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 TooBad 后门扩散模型攻击框架，通过触发器优化实现低毒化率下高攻击成功率

**💡 创新点**

创新点在于为扩散模型设计专属触发器优化方法，利用可逆稀疏不可见触发器使模型在不修改原始参数的情况下快速收敛至目标样本并保持隐蔽性

**🔧 技术方法**

使用扩散模型的前向/后向步骤、梯度下降（PGD）触发器优化、稀疏投影与对抗约束，以及 ASR、MSE、SSIM、FID 等评估指标

**📊 数据集**

主要在 CIFAR‑10、CelebA‑HQ（以及 NCSNs）上进行实验，测试不同毒化率与多种目标图像

**📈 对比分析**

与 VillanDiffusion、UIBDiffusion 等 SOTA 进行对比，TooBad 在 0.5%‑5% 毒化率下实现 85%‑98% ASR，仅需 3‑5 轮微调，耗时大幅降低，且对现有防御完全无检测

**⚠️ 局限性**

局限在于触发器需先行优化，攻击对更高分辨率或非图像任务的适用性尚待验证，且依赖可访问的预训练模型

---

## 705. Changing Modalities: Adapting Remote Sensing Models to New Satellites and Sensors

**arXiv ID:** 2606.23356 | [PDF](https://arxiv.org/pdf/2606.23356v1)

**作者:** Tim G. Zhou `[一作]` (University Of British Columbia), Evan Shelhamer `[通讯]` (University Of British Columbia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在遥感领域针对卫星传感器变更，将单模态模型迁移、增添或利用新模态，提出DeluluNet可在无标注新模态下实现模型更新。

**💡 创新点**

创新点：统一框架处理模态转移、增添、窥探三种场景；引入模态hallucination与掩码Transformer实现缺失模态的表征预测；通过批量混合实现无监督与有监督双向学习。

**🔧 技术方法**

使用ViT基础、掩码Transformer、交叉模态融合Transformer、模态特定Tokenizer、预先MSE与潜在MSE等技术。

**📊 数据集**

使用EuroSAT、reBEN（BigEarthNet更新）、DFC2020等多光谱、SAR与多模态遥感数据集。

**📈 对比分析**

与知识蒸馏、Transformed Teacher Matching、Multimodal Knowledge Expansion、MixMatch等基线对比，DeluluNet在转移、增添、窥探任务中均优于或接近多模态基准，甚至超过有监督的RSFM。

**⚠️ 局限性**

局限：仅验证两模态、静态土地利用任务；对时变任务需处理时间错位；需要配对无标签数据；在极端模态差异下hallucination质量不佳。

---

## 706. VideoAgent: All-in-One Framework for Video Understanding and Editing

**arXiv ID:** 2606.23327 | [PDF](https://arxiv.org/pdf/2606.23327v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 707. Test-Driven, AI-Assisted Learning: Replacing Lectures with Weekly Closed-Book Tests

**arXiv ID:** 2606.23315 | [PDF](https://arxiv.org/pdf/2606.23315v1)

**作者:** Jin-Guo Liu `[一作]` (Hong Kong University of Science and Technology), Wei Wang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对香港科技大学广州分校高级计算理论课程进行13周的无讲座、AI辅助、以闭卷测试为驱动的教学实验。

**💡 创新点**

创新点在于将高频闭卷测试与可复用的AI驱动课程材料生产与批改“手套”相结合，既保障学生自学可检验性，又让单一教师能可持续地批改大量测试。

**🔧 技术方法**

采用大型语言模型代理（LLM agent）进行学习资料撰写、测试生成与审阅、批改和修正；配合GitHub版本控制、自动化构建和人工审核流程。

**📊 数据集**

使用课程教材《Sipser的理论计算入门》转为Markdown的知识库，学生调查问卷（N=18）与全学期12周闭卷测试成绩作为评估数据。

**📈 对比分析**

通过学生自评、测试得分分布与Git历史修正率（96.3%无修正）等指标说明：学生接受度高、测试可行、AI批改可靠，但未与传统讲座模式进行对照或测评成绩对比。

**⚠️ 局限性**

局限性包括单一课程、样本量小且自选、缺乏对照组和前后测、仅报告感知与操作可行性、教师自评可能带偏差、适用性仅在类似证明密集课程尚待验证。

---

## 708. GRIMIP: A General Framework for Instance-Specific Configuration of MIP Solvers Using LLMs

**arXiv ID:** 2606.23299 | [PDF](https://arxiv.org/pdf/2606.23299v1)

**作者:** Yidong Luo `[一作]` (Chinese University of Hong Kong), Tianshu Yu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出 GRIMIP 框架，利用 LLM 充当贝叶斯优化中的完整概率代理，自动为 MIP 求解器配置超参数。

**💡 创新点**

创新点包括：1) 将 LLM 作为全功能代理完成采样、预测均值与方差；2) 自动搜索空间选择 (ASS) 依据实例特征裁剪高维参数；3) 热启动 (WS) 以实例特征生成高质量起始配置；4) 结合不确定性估计实现高效采集。

**🔧 技术方法**

使用技术：大型语言模型（DeepSeek、GPT 系列等）、贝叶斯优化、自动搜索空间选择、热启动、联合均值-方差预测与 EI 采集函数。

**📊 数据集**

实验数据集：MIK、CORAL、MIRP、MIPLIB、Item Placement、Load Balancing、Anonymous 七大 MIP 集合。

**📈 对比分析**

与默认、SMAC-P、SMAC-I、GPTT、LLAMBO、ifBO、TuRBO 等基线比较，GRIMIP 在硬数据集上 PDI 降低 12–45%，在 84–99% 的实例上优于默认配置，并且样本效率更高。

**⚠️ 局限性**

局限性：对中等难度实例收益有限；对极小或极大规模的 LLM 可能性能不佳；需要显式的实例特征提取与 LLM 调优成本。

---

## 709. IOI: Decoupling Kinematics and Physics for Interactive World Models

**arXiv ID:** 2606.23296 | [PDF](https://arxiv.org/pdf/2606.23296v1)

**作者:** Chengyu Bai `[一作]` (Beijing Innovation Center of Humanoid Robotics), Shanghang Zhang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种混合交互式世界模型IOI，利用URDF动力学先验与扩散生成器结合，实现机器人动作与环境交互的统一模拟。

**💡 创新点**

核心创新在于将确定性机器人运动通过多视角正交投影提取为几何先验，再通过Multi‑View Kinematic Aggregation and Injection (MKAI)模块注入扩散模型，实现无需相机外参的视角不变几何引导。

**🔧 技术方法**

技术包括URDF解析与正交投影、三视图融合、对齐嵌入、交叉注意力注入、流匹配损失训练的扩散视频生成模型以及冻结的DiT骨干网络。

**📊 数据集**

在RoboTwin 2.0仿真平台和真实Frank Emika机械臂收集的数据上进行评估，使用标准视频质量指标（PSNR、SSIM、LPIPS、FVD）和策略成功率（SR）。

**📈 对比分析**

与IRASim和Ctrl‑World基线对比，IOI在PSNR、SSIM、LPIPS和FVD上均取得明显提升（例如FVD降幅≈36%），策略评估的成功率与真实物理仿真差距≤1.6%。

**⚠️ 局限性**

局限在于仅支持关节或末端执行器层面的动作输入，对更高层抽象策略兼容性有限；正交投影缺乏外观细节，可能影响接触细节的逼真度。

---

## 710. Transfer learning-based method for automated ewaste recycling in smart cities

**arXiv ID:** 2606.23286 | [PDF](https://arxiv.org/pdf/2606.23286v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 711. A Set-Theoretic Approach to Detecting Logic Bugs in DBMS Inner Join Optimizations

**arXiv ID:** 2606.23294 | [PDF](https://arxiv.org/pdf/2606.23294v1)

**作者:** Ce Lyu `[一作]` (East China Normal University), Aoying Zho `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于集合理论的变形规则（SJT、ADT、SDT），通过将 JOIN 视为交集来构造等价查询，利用变形后的查询对 DBMS 进行元模态测试，从而发现逻辑错误。

**💡 创新点**

创新点在于：① 把 JOIN 与集合交集等价性系统化；② 设计了完整覆盖常用集合运算（UNION、INTERSECT、EXCEPT）的三条最小变形规则；③ 将变形规则与 SQLancer 结合，构建可自动化的逻辑错误发现框架 JoinEquiv；④ 在四大主流 DBMS 上发现 27 个未公开的逻辑缺陷，其中 14 个被确认为严重 bug。

**🔧 技术方法**

使用的技术主要包括：集合理论等价变形、SQLancer 的随机查询与数据库状态生成、SQL 嵌入式的语法树变形算法、以及对比执行结果的元模态判定；在实现层面，依赖 SQL 的 SET/MULTISET 语义、SQL 标准中的 JOIN 与集合运算映射。

**📊 数据集**

实验数据集由 SQLancer 随机生成的表结构与数据组成，保证所有列为 NOT NULL，覆盖了多表 JOIN 以及多种谓词和集合运算；没有使用公开的固定数据集，而是通过随机生成来产生多样化的测试场景。

**📈 对比分析**

与现有的 TLP（三值逻辑分区）和 DQP（差异化查询计划）方法对比：在同一 12 小时内，JoinEquiv 在 MySQL、TiDB、Percona、DuckDB 共发现 36 个逻辑 bug，而 TLP 仅 0，DQP 仅 1-2；发现的 bug 多为 optimizer rewrite 或 executor 级别错误，且 27 个已被确认，其中 14 为严重级别；说明 JoinEquiv 在检测 JOIN 逻辑错误方面具有更高的覆盖率与发现效率。

**⚠️ 局限性**

局限性包括：① 仅支持 NOT NULL 列，未处理 3VL 与集合运算的语义冲突；② 可能产生误报（如类型转换、零值表示差异）；③ 在某些 DBMS（PostgreSQL、CockroachDB、SQLite）未发现新 bug，表明方法对这些系统的适用性有限；④ 代码覆盖率低，主要集中在优化器层面；⑤ 需要手动分析根因与确认 bug，仍耗时。

---

## 712. Flow6D: Discrete-to-Continuous Flow Matching for Efficient and Accurate Category-Level 6D Pose Estimation

**arXiv ID:** 2606.23293 | [PDF](https://arxiv.org/pdf/2606.23293v1)

**作者:** Mingyu Mei `[一作]` (Zhejiang University), Zaixing He `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Flow6D 两阶段离散-连续流匹配框架，用于类别级 6D 关位姿估计。

**💡 创新点**

创新点在于先在离散隐空间锁定近似姿态，再通过连续流匹配细化残差，实现高精度与实时推理的统一。

**🔧 技术方法**

使用离散流匹配（DFM）、连续流匹配（CFM）、点云编码（PointNet++）、分箱离散化与高斯混合残差采样等技术。

**📊 数据集**

使用 REAL275、RobotArm、CAMERA25、ArtImage 等合成与真实场景数据集进行训练与评估。

**📈 对比分析**

与 NOCS、i2c-net、SGPA、Deterministic DPDN、GPV-Pose、RBP-Pose、Genpose、U-COPE、CAPTRA 等基线对比，平均旋转误差 5.2°/2cm 以上，实时 70 FPS，显著优于现有方法。

**⚠️ 局限性**

尚未在多物体遮挡场景中验证，也未实现端到端的一体化模型。

---

## 713. Towards a Bathroom-Centered Human-Building Digital Twin Framework for Indoor Safety Analysis

**arXiv ID:** 2606.23292 | [PDF](https://arxiv.org/pdf/2606.23292v1)

**作者:** Yuanzhi Su `[一作]` (Hong Kong Polytechnic University), Hou `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种浴室中心的人-建筑数字孪生框架，用于交互感知的室内安全分析，并在Unity中实现原型演示。

**💡 创新点**

创新点在于将浴室语义空间、骨骼式人类模型和空间语义耦合结合，形成事件级的人机交互分析，实现从动作识别向情境解释的转变。

**🔧 技术方法**

采用LiDAR扫描构建语义浴室模型，mmWave雷达进行隐私友好的人体姿态估计，并结合Unity可视化、AI动作识别和运动学特征提取技术。

**📊 数据集**

使用的数据主要是实验室浴室采集的LiDAR点云和mmWave雷达点云；未使用公开大规模数据集。

**📈 对比分析**

由于目前仅为概念验证原型，未进行系统性对比实验，性能评估主要基于可视化可用性和功能完整性。

**⚠️ 局限性**

局限包括缺乏实时数据流、未实现连续更新、人体模型存在估计误差、缺乏临床验证与事故标注、未涵盖多房间或真实居家环境。

---

## 714. Rethinking Molecular Graph Backdoors under Chemistry-aware Admission

**arXiv ID:** 2606.23361 | [PDF](https://arxiv.org/pdf/2606.23361v1)

**作者:** Thinh T. H. Nguyen `[一作]` (VinUniversity), Kok-Seng Wong `[通讯]` (VinUniversity)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了化学感知的入门检查协议ChemGuard，评估分子图神经网络在实际预处理流程中的后门攻击风险，并基于该协议设计了ChemBack——一种模型无关、可被ChemGuard通过的、通过Tanimoto相似度对目标类结构进行对齐的分子后门攻击。

**💡 创新点**

创新点在于：1）将分子预处理中的“可接受性”拆解为可清洗与图字符串一致性两个维度，形成正式的ChemGuard协议；2）利用Tanimoto相似度作为纯化学手段实现目标类对齐，避免使用受害者模型或代理网络；3）通过构造可插入的化学合理子结构（motif）并筛选满足ChemGuard的可插入位置，实现无模型、无梯度、无代码依赖的后门触发器。

**🔧 技术方法**

主要技术包括：分子字符串解析与化学清洗（Sanitize）、图字符串一致性校验、图属性提取与Morgan指纹计算、Tanimoto相似度评估、化学合理子结构的单键附着操作以及基于离散搜索（贪心/强化学习）构造触发器。

**📊 数据集**

实验使用了MoleculeNet四个基准：BBBP、BACE、SIDER和Tox21，覆盖单任务二分类与多任务分类。

**📈 对比分析**

与四种代表性图后门（GTA、Motif-Backdoor、UGBA、DPGBA）及多种防御（Spectral Signatures、DShield、RGCN、RIGBD、PGNNCert）对比，结果表明：在ChemGuard检查下，传统图后门的有效攻击率（EPR）显著下降，ASR也大幅降低；而ChemBack在EPR始终为100%，ASR保持在高水平（约60%–99%），且对大多数防御仍能保持较高成功率。

**⚠️ 局限性**

局限性包括：1）ChemGuard仅检查化学合法性与图字符串一致性，未涵盖合成可达性、药物化学规则、实验特定过滤等更严格的筛选；2）Tanimoto相似度虽然能降低模型层面的异常，但并不能保证生成的毒化分子在生物活性或毒性方面不被检测；3）当前触发器仅采用单键附着的子结构，缺乏更丰富的化学反应式空间，可能限制攻击的多样性和可扩展性。

---

## 715. Leveraging Similarities in Multi-Armed Bandits

**arXiv ID:** 2606.23414 | [PDF](https://arxiv.org/pdf/2606.23414v1)

**作者:** Khaled Eldowa `[一作]` (University of Grenoble Alpes), Pierre Gaillard `[通讯]` (University of Grenoble Alpes)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对树结构的行动集合，研究了在多点反馈（尤其是两点）下的在线学习与结构化Bandit问题，并给出了统一的最优算法与理论上限；

**💡 创新点**

创新点在于：①证明了单点Bandit反馈无法利用树形相似性，出现Ω(√{KT})下界；②提出一种适用于多点反馈的FTRL框架，能够在结构化与随机环境下同时获得最佳（best‑of‑both‑worlds）性能；③将该框架应用于Lipschitz Bandit，证明在两点反馈下，维度≤2时可实现√T调度；

**🔧 技术方法**

主要技术包括：树形正则化的FTRL（嵌套熵/特雷塞斯熵正则化），多点反馈估计，树结构下的有效动作数K_eff的定义与分析，以及对Lipschitz空间的树化压缩与覆盖构造；

**📊 数据集**

该工作以理论分析为主，未使用具体实验数据集；

**📈 对比分析**

与传统的无结构Bandit、半Bandit、以及先前的树结构Semi‑Bandit方法相比，提出的多点反馈算法在随机与对抗环境下均能以O(√{K_eff T})或O(K_eff log T)的形式显著压缩K，尤其在两点反馈下对Lipschitz问题实现了√T（d≤2）或近似最优的更优界；

**⚠️ 局限性**

局限性包括：①需要至少两点反馈才能突破单点下的下界；②算法性能高度依赖于树结构的构造与相似度尺度σ_j 的快速衰减；③多点观测虽比半Bandit更实际，但仍需额外采样，导致实际代价不一定低；

---

## 716. Differential Spectral Damping Gap Adaptive Regularization for Ill-Conditioned Kernel Methods

**arXiv ID:** 2606.23407 | [PDF](https://arxiv.org/pdf/2606.23407v1)

**作者:** Praveg Vashishtha `[一作]` `[通讯]` (Indian Institute of Technology Patna), Praveg Vashishtha (Indian Institute of Technology Patna)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 Differential Spectral Damping (DSD)，通过谱间隙信息自适应抑制不可靠的特征向量，从而稳定高维核方法的矩阵求逆，提升分类性能。

**💡 创新点**

基于 Davis–Kahan 定理，设计指数衰减调制函数，使正则化强度随 eigengap 变化，无需交叉验证即可自动初始化，显著提高不良条件下的泛化能力。

**🔧 技术方法**

使用谱分解、指数衰减调制、梯度优化（可选）和 PyTorch 实现，结合 RBF 核、Nyström 近似与 Least‑Squares Twin SVM 等技术。

**📊 数据集**

实验数据包括真实数据 GINA（d=970）与 Madelon（d=500），以及多组合成高维样本，还评估了 Swiss Roll、Two‑Moons、Genomics 等。

**📈 对比分析**

采用公平对比协议：DSD 在 LSTSVM 分类上对比 Tikhonov（网格搜索）和截断 SVD，GINA 上提升 +4.8pp、d=200 上 +10.4pp、Madelon 上 +2.6pp，Cohen d 超过 4.5；预图重构与 Tikhonov 结果相当。

**⚠️ 局限性**

仅适用于 RBF 等单调谱核；低维或谱尾聚集不足时无优势；需完整特征分解；对预图重构无提升；理论稳定性证明尚未完成。

---

## 717. OptChain: Achieving Optimal Throughput of Permissionless Blockchains

**arXiv ID:** 2606.23405 | [PDF](https://arxiv.org/pdf/2606.23405v1)

**作者:** Chunjiang Che `[一作]` (Hong Kong University of Science and Technology), Xuechao Wang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了OptChain协议，实现了在无许可区块链中近似理论最优吞吐量的状态机复制方案。

**💡 创新点**

核心创新包括Shardis（基于可验证信息扩散的权限无关方案）和Diffusion Mining（允许每个分片仅需至少一名诚实节点即可保障安全）。

**🔧 技术方法**

技术实现融合了可验证信息扩散（VID）、可编码Merkle树（Coded Merkle Tree）、PoW共识、全局排序链和欺诈证明等多种区块链技术。

**📊 数据集**

实验使用了真实世界的网络带宽追踪数据（10–80 Mbps）并在亚马逊AWS EC2跨区域节点上部署。

**📈 对比分析**

通过与Manifoldchain、Prism以及理论最优吞吐量对比，OptChain在不同误差阈值、网络带宽和节点规模下均显著超越基线，且吞吐量与理论极限高度接近。

**⚠️ 局限性**

限制主要在于需要复杂的多链结构与参数调优，以及在极大规模网络中的进一步验证与部署。

---

## 718. Physics-Informed Modeling for Wood Thermal Analysis and Prediction

**arXiv ID:** 2606.23402 | [PDF](https://arxiv.org/pdf/2606.23402v1)

**作者:** Jingren Xie `[一作]` (Technical University of Denmark), Dim P. Papadopoulos `[通讯]` (Technical University of Denmark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用物理信息深度学习框架（PICNN 与 PInteCNN），将木材 RGB 图像与试验台温度映射到像素级热响应，实现木材热行为的高精度预测。

**💡 创新点**

创新点在于同时提供软物理约束（PDE 损失）与硬物理约束（嵌入数值求解器）的两种网络架构，能够在保持可解释性的同时显著提升预测精度。

**🔧 技术方法**

采用卷积神经网络结合有限差分离散化的热传导方程，分别实现 Physics‑Informed Convolutional Neural Networks（PICNN）和 Physics‑Integrated Convolutional Neural Networks（PInteCNN），并使用自适应权重 λ_pde 与可迭代求解器。

**📊 数据集**

使用三套真实木材多模态数据集：Poplar、Grandis Cross‑Cut（Grandis‑CC）与 Grandis Radial‑Cut（Grandis‑RC），每套均包含 RGB 图像、热图与试验台温度图。

**📈 对比分析**

与纯数据驱动基线比较，PICNN 与 PInteCNN 在 MAE、RMSE 与 δ_01 指标上均优于基线；在 Grandis‑CC 数据上取得最低 MAE（0.2043）和最高 δ_01（92.13%），在 Poplar 与 Grandis‑RC 上也表现出显著的误差降低。

**⚠️ 局限性**

主要局限在于二维空间简化（忽略三维内部结构导致热异常难以捕捉）以及硬约束求解器在多步迭代时易放大噪声，导致精度下降；此外数据量有限，模型泛化仍受限。

---

## 719. MeshFlow: Mesh Generation with Equivariant Flow Matching

**arXiv ID:** 2606.23489 | [PDF](https://arxiv.org/pdf/2606.23489v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 720. Towards an Automated Reasoning Tool for Complexity Analysis of Automated Reasoners

**arXiv ID:** 2606.23516 | [PDF](https://arxiv.org/pdf/2606.23516v1)

**作者:** Louis Rustenholz `[一作]` (Universidad Politécnica de Madrid), Niki Vazou `[通讯]` (IMDEA Software Institute)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

设计并实现了一个面向自动推理算法的复杂度分析工具，包含指标定义、递归方程提取以及方程求解的完整流水线；

**💡 创新点**

首次将基于算子语义的高阶抽象编译与前后固定点搜索相结合，实现了对 catamorphic 指标的最优递归方程抽取，并通过 SMT 判定验证候选上界，解决了非线性和超越算术的函数比较问题；

**🔧 技术方法**

采用高阶抽象解释、算子语义、Galois 连接、抽象编译、前后固定点搜索、SMT 求解（Yices‑TRA）以及终止分析技术；

**📊 数据集**

主要以 Presburger 量化消除等自动推理算法为实验案例，未给出具体公开数据集；

**📈 对比分析**

与传统手工分析相比，该工具在相同示例上能够自动完成指标设定、递归抽取与解方程，得到与手工相当或更精确的上界；性能主要受 SMT 求解复杂度影响，实验表明在量化消除示例中计算时间可接受；

**⚠️ 局限性**

仍需交互式输入，非 catamorphic 指标抽象可能导致精度损失；SMT 求解在高度非线性/超越算术上存在性能瓶颈；终止分析相关技术尚未成熟，限制了工具在更广泛算法上的适用性。

---

## 721. HyperQuant: A Rate-Distortion-Optimal Quantization Pipeline for Large Language and Diffusion Models

**arXiv ID:** 2606.23406 | [PDF](https://arxiv.org/pdf/2606.23406v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 722. An Automated Framework for Input Alphabet Construction in Stateful Protocol Implementation Learning

**arXiv ID:** 2606.23464 | [PDF](https://arxiv.org/pdf/2606.23464v1)

**作者:** JiongHan Wang `[一作]` (University of Science and Technology of China), WenChao Huang `[通讯]` (University of Science and Technology of China)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于大型语言模型自动构建输入字母表并采用增量批量学习的协议状态机学习框架，用于发现状态机实现的语义漏洞。

**💡 创新点**

将LLM用于协议消息配置提取、结构化突变生成多样输入符号，并设计了单符号与多符号突变优先的增量批量学习策略，显著降低学习复杂度。

**🔧 技术方法**

使用ChatGPT 5.3生成消息配置、基于结构化突变的符号扩展、Mini-Batch L*算法结合基本状态机、基于学习的等价查询与等价测试。

**📊 数据集**

在9个主流协议实现（FTP、SMTP、RTSP、TLS1.3服务器/客户端）上评估，使用官方RFC定义的消息类型和自生成的变异消息。

**📈 对比分析**

与传统全字母表学习对比，单符号突变方案在数小时内完成学习，跨字母表学习平均缩短32.5%时间，发现10个未报告语义bug，三例已修复，一例获CVE。

**⚠️ 局限性**

突变策略仍手工设计，缺乏自动化多样性；学习过程中网络IO同步等待导致延迟，未来需引入多线程和更丰富的突变原语。

---

## 723. Multi-Vector Embeddings are Provably More Expressive than Single Vector Embeddings

**arXiv ID:** 2606.23475 | [PDF](https://arxiv.org/pdf/2606.23475v1)

**作者:** Rajesh Jayaram `[一作]` `[通讯]` (Google Research), Rajesh Jayaram (Google Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

证明了多向量（MV）嵌入在表达Chamfer相似度时，相比单向量（SV）嵌入需要指数级更高的维度，给出了正式的表达式差异分离。

**💡 创新点**

首次提供了理论证明的MV与SV的维度表达式分离；构造了利用Pattern Matrix方法的难度实例，并推导出NAND_k函数的近似多项式次数，从而得到最优下界；同时提出了Snowflake嵌入方案，将MAX‑ABS‑IP映射到IP，实现更优的维度上界。

**🔧 技术方法**

Pattern Matrix Method、近似多项式次数理论、Paturi定理、Johnson‑Lindenstrauss投影、张量幂与L_k范数估计（Snowflake嵌入）等。

**📊 数据集**

无实验数据集；全部工作基于理论构造和数学证明。

**📈 对比分析**

通过理论分析与已有的MUVERA上界（m^{O(1/ϵ^2)})对比，证明下界为m^{Ω(1/ϵ)}，从而展示MV嵌入在固定维度下比SV嵌入更具表达能力；对MAX‑IP到IP的嵌入提出更好的上界（m^{O(1/ϵ·log(1/ϵ))})。

**⚠️ 局限性**

仍存在上界与下界之间的ϵ指数差距；仅在Chamfer/Max‑IP/Max‑ABS‑IP等特定相似度上给出结果；理论结果为最坏情况构造，未讨论实际数据集上的效果。

---

## 724. Rethinking Object-Centric Representations for Video Dynamics Modeling

**arXiv ID:** 2606.23436 | [PDF](https://arxiv.org/pdf/2606.23436v1)

**作者:** Amaury Wei `[一作]` (École Polytechnique Fédérale de Lausanne), Olga Fink `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

STAITUS提出了一种无监督的视频对象中心化学习框架，能够在不依赖人工标注的情况下将动态场景分解为独立、持续的对象表示；

**💡 创新点**

其创新点在于将对象的外观与几何姿态显式分离，只对外观进行时间一致性约束，同时引入空间分离损失和自适应槽激活机制，显著提升了对象分割锐度与身份稳定性；

**🔧 技术方法**

主要技术包括基于预训练DINO特征提取器的密集编码、循环式槽注意力模块、空间广播解码器、时间对齐与空间分离正则化以及门控式槽激活；

**📊 数据集**

使用了CLEVRER、MOVi系列（A、B、C、E）以及真实世界的YouTube‑VIS 2021数据集进行评估；

**📈 对比分析**

与ISA、AdaSlot、DINOSAUR、SAVi、VideoSAUR、SlotContrast等基线比较，STAITUS在ARI、FG‑ARI和mBO指标上均显著领先，尤其在复杂场景下保持高质量分割与追踪；

**⚠️ 局限性**

局限性包括在背景复杂或纹理变化强的场景中易产生背景碎片化、分割边界模糊，以及在长时间序列中仍面临身份漂移与维持稳定性的挑战。

---

## 725. Development and Design of FLKit: A Structured Onboarding Toolkit for Federated Learning in Health and Life Sciences

**arXiv ID:** 2606.23500 | [PDF](https://arxiv.org/pdf/2606.23500v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 726. Interpretable Kolmogorov-Arnold Network with Feature-Isolated Temporal Attention Mechanism for Electricity Load Forecasting

**arXiv ID:** 2606.23425 | [PDF](https://arxiv.org/pdf/2606.23425v1)

**作者:** Jinhao Li `[一作]` (Monash University), Hao Wang `[通讯]` (Monash University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了LoadKAN框架，用特征分离的时间注意力机制结合Kolmogorov‑Arnold网络进行电力负荷预测。

**💡 创新点**

创新点在于将可学习的分段spline激活函数嵌入特征独立的时间注意力输出，实现了预测精度与可解释性的双重提升。

**🔧 技术方法**

采用了自注意力（Transformer）实现特征孤立的时间建模，并在输出端使用Kolmogorov‑Arnold网络（KAN）进行可解释性预测。

**📊 数据集**

使用了美国三大电网（NYISO、CAISO、ERCOT）的日度负荷、天气、市场价与谷歌COVID‑19社区流动性报告六类人类移动性特征。

**📈 对比分析**

在与MLP、LSTM、GRU、TCN、Transformer、Informer、Chronos以及纯KAN的基准模型对比，LoadKAN在大多数市场中获得与最优模型相近或更优的MAPE/RMSE/R²，并显著提高可解释性。

**⚠️ 局限性**

主要局限在于仅评估三大成熟电网，依赖高质量的移动性数据，且特征孤立的注意力阶段限制了跨特征交互的表达能力。

---

## 727. Hallucinations in Organization-backed AI advisors: Evidence about Skepticism, Verification, and Reliance in Goal-Directed Use

**arXiv ID:** 2606.23491 | [PDF](https://arxiv.org/pdf/2606.23491v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 728. A Compositional Language for Property Graphs

**arXiv ID:** 2606.23399 | [PDF](https://arxiv.org/pdf/2606.23399v1)

**作者:** Marcelo Arenas `[一作]` (RelationalAI), Wim Martens `[通讯]` (RelationalAI)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种能解决当前图查询语言 GQL 与 SQL/PGQ 在可组合性方面缺陷的理论与实践方案，设计了一种新的 Regular Path Query with Variables (RPQV) 以及一种基于 Datalog 的图转换语言 Hash‑Datalog，二者共同实现了对所有 NLOGSPACE 查询的表达能力；

**💡 创新点**

创新点在于：1) 对路径查询提供了对变量的完全对称处理和列表变量支持，消除了路径拼接不对称导致的不可表达性；2) 通过 Hash‑Datalog 将关系查询结果映射回图结构，完成图到图的完整可组合性；3) 提出了可提交给 ISO 标准委员会的具体语法扩展，兼容现有 GQL/SQL/PGQ 语义；

**🔧 技术方法**

主要技术包括：正则路径查询的变量化 (RPQV)、Datalog 与 Skolem 化产生新节点/边、基于图元素的 join 与拼接语义、以及对路径变量的聚合与列表处理；

**📊 数据集**

本文未使用具体实验数据集，而是通过理论证明与示例展示其功能；

**📈 对比分析**

方法评估基于理论复杂度，证明 Hash‑Datalog 在 NLOGSPACE 内完成所有 NLOGSPACE 查询，并在实验示例中演示了对复杂路径条件的表达能力；

**⚠️ 局限性**

限制在于：1) 仍为理论与示例级别，缺乏大规模实际系统实现与性能评测；2) 对于路径唯一性与长度等细粒度控制仍需额外机制；3) 兼容性实现细节需在实际数据库系统中验证。

---

## 729. GRINQH: Graded Input-based Quantization Hierarchy for Efficient LLM Generation

**arXiv ID:** 2606.23419 | [PDF](https://arxiv.org/pdf/2606.23419v1)

**作者:** Jette Oberländer `[一作]` (RWTH Aachen), Emre Neftci `[通讯]` (RWTH Aachen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为GRINQH的后训练量化框架，旨在通过统一量化和稀疏化来加速大语言模型（LLM）的解码过程，特别是在边缘计算环境中。

**💡 创新点**

GRINQH通过动态分配权重通道的精度，针对计算重要性来优化解码过程，建立了新的性能边界，允许在生成质量和推理速度之间进行动态权衡。

**🔧 技术方法**

使用了动态后训练量化（PTQ）框架和自定义的GPU内核，结合分层嵌套内存布局来实现多精度存储。

**📊 数据集**

在Llama3和Qwen3模型上进行了评估，使用了来自The Pile数据集的128个样本进行阈值校准。

**📈 对比分析**

与现有的固定和混合精度基线方法相比，GRINQH在3位和4位设置下表现更优，甚至在2位生成中也表现有效，且在RTX 4090上实现了显著的速度提升。

**⚠️ 局限性**

GRINQH主要针对批量大小为1的场景，使用了位平面存储，这需要较大的DRAM占用，且在预填充性能上略有影响。

---

## 730. TROPT: An Open Framework for Unifying and Advancing Discrete Text Optimization

**arXiv ID:** 2606.23496 | [PDF](https://arxiv.org/pdf/2606.23496v1)

**作者:** Matan Ben-Tov `[一作]` (Tel Aviv University), Mahmood Sharif `[通讯]` (Tel Aviv University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了TROPT框架，统一并标准化离散文本触发优化器，使其易于使用、可扩展并可在不同模型和任务中复用。

**💡 创新点**

提供了一个模块化、统一的接口，集成15+优化器、15+损失、30+预设recipe，降低了工程门槛，支持快速组合、可比性评测，并首次实现跨域迁移与标准化基准。

**🔧 技术方法**

离散搜索优化算法（梯度、连续松弛、零阶、遗传搜索等）与统一框架接口，使用Python、HuggingFace等技术实现模型后端、损失函数和Recipe Hub。

**📊 数据集**

使用LLM模型（Llama‑3.1‑8B‑Instruct、Falcon、Mistral、Phi‑2）进行jailbreak基准；ClearHarm 15条有害指令；OpenAI嵌入模型的8M检索语料库；prompt‑injection数据集；Stable Diffusion 2.1用于文本‑图像检索。

**📈 对比分析**

通过对14种优化器在4个模型上进行相同recipe的同构比较，采用平均排名评估；梯度基准PAL、MAC位居前列，黑盒RAL与白盒GCG相当；8种jailbreak增强中，目标字符串替换提升通用性约两倍；跨域实验展示对稠密检索器、prompt‑injection检测器、文本‑图像模型的成功迁移。

**⚠️ 局限性**

仅覆盖离散搜索优化器，未包含RL或LLM代理方法；基准聚焦jailbreak领域，缺乏多模型、多任务的全面评测；优化器参数未针对每个模型细调；缺乏对更大规模或更复杂任务的验证。

---

## 731. War in the Abstract: The Rise and Consequences of Militarized Language in Scientific Communication

**arXiv ID:** 2606.23462 | [PDF](https://arxiv.org/pdf/2606.23462v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 732. Selective Time Series Forecasting via Metalearning

**arXiv ID:** 2606.23448 | [PDF](https://arxiv.org/pdf/2606.23448v1)

**作者:** Ricardo Inácio `[一作]` (Universidade do Porto), Carlos Soares `[通讯]` (Universidade do Porto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于金属学习的选择性时间序列预测框架，利用过去滞后窗口的结构特征预测误差百分位，从而在发出预测前决定是否拒绝预测。

**💡 创新点**

创新点在于：①将误差映射为每条序列内部的经验百分位，形成尺度不变且可跨域的目标；②使用仅依赖于滞后窗口的结构特征，使拒绝决策与具体预测模型和数据规模无关；③实现了在不同域之间的零拷贝迁移与轻量级适配。

**🔧 技术方法**

使用的技术包括：深度学习全局预测模型（KAN、NHITS），金属学习回归器（XGBoost），滞后窗口特征提取（tsfel），滚动起点交叉验证，经验百分位计算与基于阈值的拒绝策略。

**📊 数据集**

采用公开的单变量时间序列数据集：M3、M1 与 Tourism（仅月度和季度子集），在源域与目标域间进行转移学习实验。

**📈 对比分析**

与基线方法（预测区间宽度、残差方差、残差尺度、随机拒绝、理论上最优的Oracle）进行比较。实验表明：①在源域和零拷贝转移下，该方法在风险-覆盖曲线和AUCO指标上均优于或接近Oracle；②在域适配后进一步提升，往往在各指标上击败所有基线；③在保留预测的平均误差上均实现显著下降，且与Oracle间的平均距离最小。

**⚠️ 局限性**

局限性包括：①对元训练样本的代表性依赖较高，短序列或极端分布可能导致拒绝模型失效；②仅给出相对风险排序，缺乏正式的概率不确定性保证；③目前仅适用于单变量规则时序，扩展到多变量或不规则采样需要进一步研究。

---

## 733. SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors

**arXiv ID:** 2606.23444 | [PDF](https://arxiv.org/pdf/2606.23444v1)

**作者:** Pratyaksh Rao `[一作]`, Giuseppe Loianno `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于JEPA的潜在动力学模型和物理启发式探测器，实现了无人机的实时高效控制；

**💡 创新点**

创新点在于将潜在空间预测与物理结构相结合，避免自回归误差累积，同时通过域随机化训练实现零样本 sim‑to‑real 转移；

**🔧 技术方法**

采用Temporal Convolutional Network编码、GRU预测、SigReg正则化、物理启发式探测器、MPPI优化等技术；

**📊 数据集**

使用500个随机参数域的仿真数据，生成20k条10秒平滑随机轨迹（20 Hz），无真实飞行数据；

**📈 对比分析**

与传统自回归预测模型对比，在开环预测、噪声鲁棒、零样本实时闭环控制以及负载/螺旋桨变化场景下，位置RMSE降至0.24‑0.45 m、姿态误差降至9‑19°，平均性能显著优于基线；

**⚠️ 局限性**

局限在于仅验证低维状态输入，未处理高维视觉观测；模型依赖大规模域随机化仿真，复杂环境下安全性与不确定性建模仍待完善。

---

## 734. Brain-Adapter: A Dual-Stream Vision-Language MIL Framework for Comprehensive 3D CT Diagnosis of Acute Intracranial Pathologies

**arXiv ID:** 2606.23494 | [PDF](https://arxiv.org/pdf/2606.23494v1)

**作者:** Zhenyu Yi `[一作]` (Shanghai Jiao Tong University), Lichi Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

设计并实现了Brain-Adapter，一个双流视觉-语言MIL框架，用于三维CT急性颅内病理的多标签诊断。

**💡 创新点**

将预训练的二维视觉-语言模型通过LoRA迁移至3D，提出文本条件注意力与逻辑集监督的双流结构，并引入不确定性感知校正模块在推理时动态融合。

**🔧 技术方法**

使用低秩自适应（LoRA）、文本条件注意力（TCA）、注意力基础MIL（ABMIL）、LLM逻辑集提取、信息对比损失、非对称损失、一致性约束以及不确定性感知融合等技术。

**📊 数据集**

在852例非对比头CT临床数据集（训练682/测试170）以及外部CQ500数据集上进行实验。

**📈 对比分析**

与3D ViT-B、Swin-B以及从头训练或BiomedCLIP预训练的2D MIL（Mean Pooling、ABMIL、TransMIL）对比，Brain-Adapter在微观AUC 0.887、宏观AUC 0.778、Hamming Loss 0.079等指标上领先，尤其在极度不平衡下微观敏感度提升显著。

**⚠️ 局限性**

模型仍受限于训练数据多样性，薄切片表现欠佳；LLM提取的逻辑集可能引入误差；推理时双流融合对计算成本有一定增加。

---

## 735. MeGAS: Thermomechanical Dynamic Gaussian Splatting for Thermophysical Scene Editing

**arXiv ID:** 2606.23455 | [PDF](https://arxiv.org/pdf/2606.23455v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 736. Analysis of Autonomic Regulation in Cancer Survivors During Daily Physical Activity: A Real-World Wearable ECG Study

**arXiv ID:** 2606.23461 | [PDF](https://arxiv.org/pdf/2606.23461v1)

**作者:** Sajad Farrokhiørcidicon `[一作]`, Christian Poellabauer `[通讯]` (Florida International University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

使用可穿戴ECG和运动传感器，实时测量乳腺癌幸存者与健康对照在日常活动中的心率与心率变异性，结合运动强度分段与ECG质量评估，揭示两组在不同运动强度下的自主神经调节差异。

**💡 创新点**

首次将无监督聚类的运动强度分段、ECG质量评估与注释自由的生理一致性验证融合成一个统一的质量感知多模态框架，并在真实世界环境中验证其可行性。

**🔧 技术方法**

采用零相位Butterworth滤波、波形增强R峰检测、无注释的RR有效率/不稳定度/变异系数/波形相关度等质量指标、K‑means聚类分段、以及Welch t检验和Mann‑Whitney U检验等统计方法。

**📊 数据集**

54名受试者（37名乳腺癌幸存者、17名健康对照），共收集142.49小时的移动式单导联ECG（125 Hz）与多轴IMU（52 Hz）数据，来源于Movesense MD传感器的自由生活采样。

**📈 对比分析**

通过参与者级和窗口级的Welch t检验、Mann‑Whitney U检验以及Cohen d效应大小评估，结果显示在中等运动强度下，癌症幸存者的HR显著升高、RMSSD和SDNN显著降低，效应大小从小到中等，统计显著性达p<0.05。

**⚠️ 局限性**

样本不平衡与年龄差异导致可能的混杂效应，数据来源为自由生活环境，存在传感器放置与环境波动的不可控噪声，且仅评估时间域HRV指标，未包含频域或非线性特征，未来需要更大样本和多模态验证。

---

## 737. A matrix-free, differentiable PyTorch solver for phase-field fracture: Formulation, benchmarks, and inverse analysis

**arXiv ID:** 2606.23458 | [PDF](https://arxiv.org/pdf/2606.23458v1)

**作者:** Allamaprabhu Ani `[一作]` (University of London), Sathiskumar Anusuya Ponnusami `[通讯]` (University of London)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个无矩阵、基于PyTorch的可微分相位场断裂显式动力学求解器。

**💡 创新点**

采用完全基于张量的散射‑聚集矩阵自由实现，并配合自定义共轭梯度逆向规则，实现一次前向求解后即可得到梯度；同时支持AT1/AT2、能量分解、平面应变/应力、CPU 与 GPU 无需编译扩展。

**🔧 技术方法**

利用PyTorch自动微分、矩阵自由有限元、散射聚集、投影预条件共轭梯度、可微分逆向CG以及L‑BFGS参数反演等技术。

**📊 数据集**

采用公开的四个动态断裂基准（直裂、剪切弯曲、分支、穿孔板）和两个准静态基准（SENT、孔板），并使用 Borden、Bleyer 等实验参数。

**📈 对比分析**

与文献基准、FEniCS、Akantu 等实现进行损伤场、裂尖速度、角度对比；在 NVIDIA A100 GPU 上实现约 1e6 节点的显式步耗约 11.6 ms/步，CPU‑GPU 加速 5–13 倍；子循环与 AMG 预条件显著降低 CG 迭代次数与总耗时。

**⚠️ 局限性**

受限于单张 GPU 显存导致时间步数上限，逆向梯度对激活集跳变敏感；目前仅支持二维、无自适应细化，三维扩展及更完善的非光滑性处理仍待改进。

---

## 738. Do Location Encoders Capture Spatial Effects? A GeoShapley Benchmark Across Scales

**arXiv ID:** 2606.23453 | [PDF](https://arxiv.org/pdf/2606.23453v1)

**作者:** Daniel Kiv `[一作]` (University of Illinois Urbana-Champaign), Shaowen Wang `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估位置编码器嵌入在不同尺度（网格、县、全球）上是否能通过 GeoShapley 解释器准确恢复空间可变系数，并比较多种编码器和坐标基线的表现。

**💡 创新点**

在高维学习式位置编码器与 GeoShapley 的结合上进行系统基准，揭示尺度依赖的编码器效果及对比学习预训练对空间效应恢复的非必要性。

**🔧 技术方法**

使用 TorchSpatial 的 11 种位置编码器、GeoShapley Shapley 值解释、合成空间可变系数数据、MLP 下游模型、Pearson 相关评估以及对比对照学习与未训练条件。

**📊 数据集**

构造三种尺度的合成数据集：25×25 网格（625 点）、约 3000 个美国县中心点、10000 个全球球面点，数据包含已知空间可变系数的线性或非线性响应。

**📈 对比分析**

通过对 GeoShapley 估计的 β̂ 与真实 β 进行 Pearson 相关计算。结果显示 β1 在所有条件下 r>0.98，β2 在局部尺度与坐标基线相当（差异 ≤0.02），但在全球尺度最高 r≈0.99 的 Sphere2Vec‑sphereM+ 显著优于其他编码器；对比学习往往不提升恢复性能。

**⚠️ 局限性**

限制包括：仅使用合成平滑过程，嵌入维度固定为 8，评估仅针对 MLP 与 GeoShapley，未检验不连续或真实世界数据，且对比学习目标与空间可变系数恢复之间的对齐仍不充分。

---

## 739. AOHP: An Open-Source OS-Level Agent Harness for Personalized, Efficient and Secure Interaction

**arXiv ID:** 2606.23449 | [PDF](https://arxiv.org/pdf/2606.23449v1)

**作者:** Shanhui Zhao `[一作]` (Tsinghua University), Yuanchun Li `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

设计并实现了面向 AI 代理的安卓系统架构 AOHP，提供个性化服务组合、代理友好接口和安全信息流机制，并在 AOSP 上实现。

**💡 创新点**

将操作系统从应用中心化转向代理本地化，统一服务发现与生成式入口、并行后台执行、结构化 UI 与文件、事件流抽象以及沙箱信息流跟踪，解决代理执行效率、安全和跨应用协同问题。

**🔧 技术方法**

基于 Android Open Source Project 的系统层改造，结合虚拟显示、沙箱运行时、结构化 UI 解析、统一文件快捷、事件流抽象、数据沙箱与污点跟踪技术，并使用 OpenClaw LLM 代理。

**📊 数据集**

30 个基于真实手机应用的跨应用任务集合（涵盖 GUI 操作、非 GUI、事件捕获、多源信息检索、内存管理等），以及用于安全评估的标注支付应用。

**📈 对比分析**

与原生 Android（stock）对比，使用同一 OpenClaw 代理测算任务完成率、工具调用、执行时长、Token 与 LLM 请求；AOHP 提升完成率约21%，工具调用、时长、Token、LLM 请求分别降低约44%、44%、52%、48%；安全案例验证敏感数据被隔离、审计。

**⚠️ 局限性**

兼容性局限于标准渲染，缺乏对自定义渲染、反自动化逻辑和文档不全的应用支持；能力发现依赖手工或自动化描述；后台资源调度与热量/内存限制待完善；策略审批体验需要改进。

---

## 740. What Does a Chemical Language Model Know About Molecules?

**arXiv ID:** 2606.23443 | [PDF](https://arxiv.org/pdf/2606.23443v1)

**作者:** Christian Kenneth `[一作]` (Independent), Gerard JP van Westen `[通讯]` (Leiden University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

利用稀疏自编码器（SAE）对MolFormer‑10的残差流进行逐层可解释性分析，揭示其如何从SMILES的语法信息逐步构建分子语义表示，并对非规范及无效SMILES的影响进行实验研究。

**💡 创新点**

1) 首次将SAE用于编码器‑only 化学语言模型，系统解析了位置跟踪隐层与atom‑in‑substructure隐层的层级结构；2) 开发了 InterMol 交互可视化工具；3) 证明非规范 SMILES 会在早期层引入更大扰动；4) 通过线性探针识别出与 ADMET 相关的药理学隐层。

**🔧 技术方法**

稀疏自编码器（k‑SAE）训练、相似度与 SMD 评估、线性探针、SMILES 变体实验、InterMol 可视化框架。

**📊 数据集**

MolFormer‑10 预训练语料（约 1.1 B SMILES）、自制的 canonical / non‑canonical / invalid SMILES 数据集、Therapeutics Data Commons（TDC）提供的 21 个 ADMET 任务数据集。

**📈 对比分析**

将 SAE 嵌入作为特征与 MolFormer‑10、ECFP、物理化学描述符在 21 个 ADMET 任务上进行比较。结果显示 SAE 在 16/21 任务中表现最佳；相似度分析表明位置隐层在非规范 SMILES 下的扰动最大，验证了模型对 SMILES 表示的敏感性。

**⚠️ 局限性**

仅针对 MolFormer 进行研究，结果可能不具备跨模型泛化；SAE 训练数据规模有限；RoPE 位置嵌入导致注意力层分析受限；缺乏金标准化学概念，部分解释仍属初步。

---

## 741. Digital Humanism and Evolutionary Design

**arXiv ID:** 2606.23417 | [PDF](https://arxiv.org/pdf/2606.23417v1)

**作者:** Wolfgang Höhl `[一作]` `[通讯]` (Technical University of Munich), Wolfgang Höhl (Technical University of Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过理论分析和概念阐述，探讨数字人文主义与进化式设计的相互关系、共性结构与挑战。

**💡 创新点**

创新点在于将数字人文主义与进化式设计两个领域进行系统比较，并提出“开放机器”和“人本技术进化”框架。

**🔧 技术方法**

主要技术手段为哲学、伦理学与软件工程的跨学科理论分析，未使用算法实现。

**📊 数据集**

未使用具体数据集，基于文献综述与案例研究。

**📈 对比分析**

通过概念对比和案例讨论进行比较，未给出性能指标。

**⚠️ 局限性**

局限在于缺乏实证验证、量化评估与可操作的实现路径。

---

## 742. Detecting Malicious Agent Skills in the Wild using Attention

**arXiv ID:** 2606.23416 | [PDF](https://arxiv.org/pdf/2606.23416v1)

**作者:** Bacem Etteib `[一作]` (University of Luxembourg), Tégawendé F. Bissyandé `[通讯]` (University of Luxembourg)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Locate-and-Judge两阶段检测恶意技能的框架，并在野外市场中扫描134k技能，发现131个恶意技能。

**💡 创新点**

将注意力定位与判别分离，使用小模型定位高注意力跨度，再用强模型判断，显著降低成本并提高隐藏恶意技能检测率。

**🔧 技术方法**

基于LLM注意力分析的小模型定位器、零样本判别器DeepSeek、结构化跨度分割与注意力打分。

**📊 数据集**

使用Skill-Inject合成数据集做校准，随后扫描来自Lobehub、Skills.sh、Clawhub.ai的134k真实技能，并发布标注数据。

**📈 对比分析**

与正则、Attention Tracker、SkillSpector、Cisco Skill Scanner和全内容LLM基线对比，Locate-and-Judge在精度约83%（含恶意）/89%（含双用）下召回率提高到83%（vs 62%），在隐藏恶意技能召回率达83%而全内容仅45%。

**⚠️ 局限性**

对跨技能链攻击无效，定位器对单行安装器等不被分段的攻击失效，需要二次全内容扫描；阈值易受域漂移影响；仅评估单技能，未考虑上下文关联。

---

## 743. DexTeleop-0: Force-Aware Bimanual Dexterous Teleoperation with Ego-Centric Perception towards Shared Autonomy

**arXiv ID:** 2606.23431 | [PDF](https://arxiv.org/pdf/2606.23431v1)

**作者:** Haichao Liu `[一作]` (Nanyang Technological University), Ziwei Wang `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在DexTeleop-0框架中提出了一种基于触觉驱动的共享自治策略，用以弥补传统视觉跟踪在双臂多指手操作中的身体模型差距，并通过实时力平衡优化循环实现高精度、低冲击的手部运动控制。

**💡 创新点**

创新点包括：①将触觉力反馈直接嵌入到跟踪优化循环，实现细粒度力匹配与姿态调节；②采用操作空间雅可比矩阵推导局部力误差并映射到关节空间；③引入全局物体力矩平衡约束，提升多接触环境下的抓取稳健性；④实现了轻量、硬件无关的全栈解决方案，仅需VR头显实现人机映射。

**🔧 技术方法**

技术要点：Meta Quest 3 egocentric 手部追踪；多指手与双臂的逆运动学映射；操作空间雅可比矩阵与力矩计算；基于触觉传感器的接触点与力估计；实时二次规划（QP）求解（30 Hz）实现残差动作优化；离散与连续激活权重的融合。

**📊 数据集**

数据集：未使用公开数据集，而是通过模拟环境（IsaacSim）与真实硬件（UR7e + Sharpa Wave）分别收集多名操作者（5名初学者+2名专业员）在多任务（球组装、杯中搅拌、齿轮配合、插孔、食品分类、管道操作）下的跟踪轨迹与触觉记录。

**📈 对比分析**

与三种基线（无残差、关节PD、仅局部力跟踪）进行对比，评价指标为多阶段成功率与触觉力均值/方差。结果表明，DexTeleop-0在所有任务中均实现了显著提升的成功率（单阶段可达 95–100%，多阶段多达 97% 以上），同时保持或降低触觉力（平均 4–11 N，远低于无残差的 29–31 N），验证了其在精度与安全性上的优越性。

**⚠️ 局限性**

局限性：①依赖于高分辨率触觉传感器与VR追踪的稳定性，若硬件失效或误差增大会影响性能；②优化阈值与激活参数需手动调节，缺乏自适应机制；③在极端环境（强光、遮挡）下的视觉跟踪可能失真；④实验仅在 UR7e + Sharpa Wave 机器人上验证，泛化到其他平台需进一步研究。

---

## 744. UnBias-Plus: Detect, Explain, and Rewrite Bias

**arXiv ID:** 2606.23412 | [PDF](https://arxiv.org/pdf/2606.23412v1)

**作者:** Ahmed Y. Radwan `[一作]` (Vector Institute for Artificial Intelligence), Shaina Raza `[通讯]` (Vector Institute for Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了UnBias-Plus开源工具包，实现细粒度的偏见检测、解释与中性重写。

**💡 创新点**

创新点在于整合多类偏见识别、跨度定位、解释生成与自动中性改写，并公开训练模型与数据。

**🔧 技术方法**

采用微调的Qwen3系列大型语言模型（Qwen3-8B、Qwen3.5-4B），配合Python、FastAPI和Hugging Face等技术。

**📊 数据集**

使用自建注释数据集进行训练，并在外部的BABE新闻数据集上进行评估。

**📈 对比分析**

通过GPT-4o-mini人工评判与自动指标（bias red., relevance, rewrite quality, ROUGE‑L, hallucination/duplicate rates）对比，8B模型在偏见消除与语义保真度上更优，4B模型在改写质量和片段编辑准确性上更好。

**⚠️ 局限性**

局限性包括上下文长度受限、模型对细微偏见识别不足、需针对特定领域再训练、以及需要人工审核才能确保公正性。

---

## 745. Litmus: Zero-Label, Code-Driven Metric Specification for Evaluating AI Systems

**arXiv ID:** 2606.23403 | [PDF](https://arxiv.org/pdf/2606.23403v1)

**作者:** Prajjwal Gupta `[一作]` (PricewaterhouseCoopers), Kevin Paul `[通讯]` (PricewaterhouseCoopers)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种零标签系统 Litmus，能通过源代码分析与澄清式提问自动生成针对 AI 流程各阶段的评估与监控指标。

**💡 创新点**

创新点在于先对评估目标进行询问与澄清，再将得到的约束转化为可验证的指标规范，从而实现“先明确目标后选指标”的评估设计范式。

**🔧 技术方法**

技术手段包括：静态代码扫描构建调用图、利用 LLM 生成并验证组件架构、向工程师与内部判别器提出澄清与对抗性有效性问题、基于已确认事实约束的指标合成与导出。

**📊 数据集**

使用的真实数据集为三个业务管道：财务账户分组、科学问答（PaperQA2/LitQA2）以及固有风险评估（IRF），无需标注。

**📈 对比分析**

与 AutoMetrics、DynamicRubric（实例、数据集、微调）等基线进行比较，Litmus 在覆盖率、低冗余度、指标与人工标签的一致性（Spearman ρ）以及对合成降质的敏感性等多维度上均优于或持平基线，尤其在科学问答任务上显著提升至 ρ≈0.72。

**⚠️ 局限性**

局限性包括：评估仅基于单一判别器族；与基线不完全可比，且 Litmus 输出指标量大导致覆盖度上升；对 LLM 生成的澄清与验证过程可能引入模型偏差；缺乏多判别器、跨域更大样本的实验证据；澄清提问对最终指标的实际影响需进一步人机研究。

---

## 746. Arbor: Explicit Geometric Conditioning for Controllable 3D Asset Generation

**arXiv ID:** 2606.23514 | [PDF](https://arxiv.org/pdf/2606.23514v1)

**作者:** Jan-Niklas Dihlmann `[一作]` (University of Tübingen), Mark Boss `[通讯]` (Stability AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Arbor，结合文本条件的 3D 生成模型，使用约束网格（占据、避免、接触）来指导生成。

**💡 创新点**

创新点在于把约束网格编码为紧凑的标记，并通过路由残差分支直接注入冻结的 denoiser，实现局部类型化空间控制。

**🔧 技术方法**

采用冻结的 TRELLIS 2 编码器、几何路由器、残差适配器以及 Top‑K 近邻选择和全局摘要标记，对约束网格进行编码并在稀疏结构阶段注入。

**📊 数据集**

在约 5 万个来自 ABO、HSSD、Sketchfab 子集（Objaverse‑XL）的大规模 3D 物体上训练；评估使用 Toys4K 数据集，并构造了自动和手工约束基准。

**📈 对比分析**

与 TRELLIS、Gradient、SpaceControl、Spice‑E 等基线以及 Point‑E、SPAR3D、Hunyuan3D‑Omni 进行对比，使用 Hull Hit、Avoidance Violation、Touch Hit、Volume Match、MV‑CLIP 与 Control Score 评估；Arbor 在约束遵从性、生成质量和多样性上均优于基线，用户研究中获得 59.2% 的偏好。

**⚠️ 局限性**

局限性包括只能传递几何与类型信号，缺乏完整语义功能；仅在稀疏结构阶段施加约束，无法直接控制后续细化阶段的表面细节和材质。

---

## 747. Source-Free Detection and Impact Analysis of Compiler Optimization Problems in Mobile Applications

**arXiv ID:** 2606.23512 | [PDF](https://arxiv.org/pdf/2606.23512v1)

**作者:** Han Hu `[一作]` (Independent Researcher), Li Li `[通讯]` (Beihang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e`

**🎯 论文内容**

本文提出了OptDetect，一个源代码无关的框架，能够直接从Android App的二进制文件中检测本地库的编译优化级别。

**💡 创新点**

创新点在于通过将二进制拆分为固定窗口进行分块级别分类，并采用加权聚合来判定混合优化级别的库，从而实现大规模真实app的无源检测。

**🔧 技术方法**

使用的核心技术包括Capstone等工具的二进制反汇编、基于固定窗口的分块与特征抽取、BiLSTM深度学习分类模型，以及基于权重的分数聚合与阈值划分。

**📊 数据集**

实验数据来源于公开的Optimization‑Detector与Assemblage数据集，以及从830个顶级Google Play App中提取的21,972个.so库。

**📈 对比分析**

与规则基线和BinEye等方法对比，OptDetect在控制集上实现93.0%准确率，在真实集上实现81.9%准确率；在低优化（O0/O1）vs高优化（O2/O3）的二分类任务中，准确率达到90.8%，显著优于现有方案。

**⚠️ 局限性**

局限包括对加密或高度混淆二进制的检测能力有限；模型对训练集代表性的依赖导致对未知编译器或特殊优化的泛化不足；以及未对运行时热点进行加权，导致对实际性能损耗估计不够精确。

---

## 748. CADRE: Stable, Parameter Efficient Adaptation of Medical Vision Language Models with Bounded Forgetting and Prior Drift

**arXiv ID:** 2606.23487 | [PDF](https://arxiv.org/pdf/2606.23487v1)

**作者:** Amrita Singh `[一作]` (Mindriser's Consortium), Rishabh Jha `[通讯]` (University of Victoria)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出 CADRE 框架，使用冻结的 BiomedCLIP 通过 LoRA 与在线自适应 EWC 及 anchor 约束，对医疗 VLM 进行参数高效的持续适配。

**💡 创新点**

重新设计 EWC 为自缩放、相似度感知的在线形式，消除尺度与顺序不稳定；引入 anchor‑to‑prior 漂移惩罚；通过理论保证总权重上界和尺度不变性实现对顺序鲁棒。

**🔧 技术方法**

LoRA 参数高效微调、在线自适应 EWC、相似度加权 Fisher、anchor 约束、轻量标签平滑与评估时权重平均。

**📊 数据集**

三种最大异质的乳腺影像模态：组织病理切片、乳腺超声和胸部 X 光，均来自公开数据集（BreaKHis、公开超声、放射学）。

**📈 对比分析**

与线性探针、LoRA、LoRA+EWC、LoRA+Anchor 等方法对比，CADRE 在跨模态连续适配任务中实现约0.23% 参数提升，准确率最高，遗忘降至0.011，正向迁移+0.004，SPQ最高，且对模态顺序鲁棒。

**⚠️ 局限性**

缺乏真实未平衡和分布漂移的评估；超声与胸片的病例分组仅按图像级别，未验证对 OOD 或攻击的鲁棒性。

---

## 749. C^2GR: Coupled Comprehensive Generative Replay for a Continually Learnable Universal Segmentation Model

**arXiv ID:** 2606.23473 | [PDF](https://arxiv.org/pdf/2606.23473v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 750. From Reconstruction to Decision: A Post-Encoder Plug-in Adapter for Curvilinear Segmentation

**arXiv ID:** 2606.23486 | [PDF](https://arxiv.org/pdf/2606.23486v1)

**作者:** Qin Lei `[一作]` (Center for Big Data and Intelligent Medicine, First Affiliated Hospital of Chongqing Medical University), Hao Wu `[通讯]` (Center for Big Data and Intelligent Medicine, First Affiliated Hospital of Chongqing Medical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计了一种轻量化的后编码器插件 PEPA，针对曲线结构分割中的重建与决策瓶颈进行改进。

**💡 创新点**

创新点在于提出目标条件蛇形上采样（TCSU）与目标自适应可微阈值（TADT）两大模块，实现结构感知的高分辨率重建与自适应二值化。

**🔧 技术方法**

采用 TCSU 的连续蛇形采样、TADT 的阈值中心化与一致性约束，并将其与 ViT、SAM、Mask2Former、MaskDINO 等主流编码器结合；PEPA 仅增加约 0.26M 参数。

**📊 数据集**

使用五个医学与工业曲线分割基准：DRIVE、CHASEDB1、CHUAC、XCAD 和 Crack500。

**📈 对比分析**

通过与同基础编码器完整微调基线以及最新域特定模型的对比，平均提升 IoU 约 2.6%，clDice 约 2.8%，在所有基准上均优于或接近最先进方法。

**⚠️ 局限性**

局限性包括仅针对 2D 曲线结构，依赖目标描述符，对极稀疏或高噪声场景的鲁棒性有限，且尚未验证 3D 或动态场景的适用性。

---

## 751. Sort-Stratified Semantics for Temporal Conflict Detection in ODRL Policies

**arXiv ID:** 2606.23442 | [PDF](https://arxiv.org/pdf/2606.23442v1)

**作者:** Daham M. Mustafa `[一作]` (RWTH Aachen University), Rafiqul Haque `[通讯]` (University of Galway)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种对 ODRL（Open Digital Rights Language）中时间约束进行排序分层（Sort‑Stratified）语义的框架，解决了时点与时长比较操作符歧义导致的冲突检测不可靠问题；

**💡 创新点**

创新点在于通过给每个时间操作数指定时点或时长两种排序，并把每个约束映射为区间（周期约束映射为周期性时间点集），从而把冲突检测归约为三值区间比较并结合背景理论（如计量使用 ≤ 已用时长）实现可判定的冲突判定；

**🔧 技术方法**

使用了范畴化语义、区间逻辑、差分图（负环检测）、可除性（Presburger 算法）等技术，并将判断分为三个可判定层级：顺序层、差分层、模数层；

**📊 数据集**

构建了 72 个冲突检测基准问题，涵盖单变量、交叉变量、周期、序列、运行时等类别，数据以 ODRL Turtle 语法、TPTP 与 SMT‑LIB 格式生成；

**📈 对比分析**

对比方法：利用 Vampire、E、Z3、cvc5 四种自动定理/SMT 证明器分别在 TPTP/SMT‑LIB 编码上求解，实验表明所有问题至少被一个求解器决议且无错误，证明了语义的可机化性与判定性；

**⚠️ 局限性**

局限在于仅处理 ODRL 的时间约束，不覆盖其他约束、媒体流位置等；对周期和序列的支持仍依赖 SMT 求解器；并未给出完整的机理证明，未来工作计划进行 Isabelle/HOL 形式化验证。

---

## 752. Cross-Architectural Mixture-of-Experts with Adaptive Soft Routing for Plant Leaf Disease Classification

**arXiv ID:** 2606.23441 | [PDF](https://arxiv.org/pdf/2606.23441v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 753. DVL-DeepONet: A Physics-Guided Operator Learning for Resilient Underwater Navigation

**arXiv ID:** 2606.23502 | [PDF](https://arxiv.org/pdf/2606.23502v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 754. Flowing With Purpose: Latent Action Guided Flow Matching Policies For Robotic Manipulation

**arXiv ID:** 2606.23420 | [PDF](https://arxiv.org/pdf/2606.23420v1)

**作者:** Bruno Machado `[一作]` (École Centrale de Lyon), Liming Chen `[通讯]` (École Centrale de Lyon)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种Latent Action Guided Flow Matching (LAFM) 框架，用多元可学习先验替代单一高斯源分布，并通过隐动作模型动态选择先验，从而在机器人操作中实现更高效的流匹配。

**💡 创新点**

创新点包括：① 将源分布从全局固定Gaussian改为基于隐动作的可学习先验库；② 用隐动作模型直接作为生成过程的结构性先验；③ 引入KL正则化防止先验过度偏移；④ 在流匹配中实现多先验，显著减少向量场纠缠。

**🔧 技术方法**

采用的技术有：条件流匹配、Diffusion Transformer (DiT)、隐动作模型（LAM）、多模态Transformer编码器、KL正则化、类别交叉熵分类、Beta采样与Euler积分等。

**📊 数据集**

实验数据集包括 LIBERO-90 benchmark、真实机器人演示集以及用于训练LAM的人类演示视频。

**📈 对比分析**

与标准流匹配（π_0）、大规模 vision‑language‑action (VLA) 模型及其他基线进行对比；在真实机器人上成功率提升23.4%，在 LIBERO‑90 提升10.4%，并在性能上超过更大规模的 VLA 模型。

**⚠️ 局限性**

局限性在于：依赖视频基的LAM可能捕捉背景或相机运动的噪声，导致隐动作不够纯粹；先验可能对稀疏样本过度拟合；未将多先验流匹配直接集成至大型预训练 VLA 管道。

---

## 755. ReasoningLens: Hierarchical Visualization and Diagnostic Auditing for Large Reasoning Models

**arXiv ID:** 2606.23404 | [PDF](https://arxiv.org/pdf/2606.23404v1)

**作者:** Jun Zhang `[一作]` (Chinese Information Processing Laboratory Institute of Software Chinese Academy of Sciences), Le Sun `[通讯]` (Chinese Information Processing Laboratory Institute of Software Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 ReasoningLens 框架，对大型推理模型产生的长链式思考（CoT）进行分层可视化、自动错误诊断与系统性行为剖析。

**💡 创新点**

创新点包括：① 统一的多粒度诊断流程；② 采用记忆、验证、建议三模组的多代理系统实现精准错误定位与可执行修复建议；③ 通过层次化图谱将长 CoT 结构化为可交互的探索‑利用层级；④ 结合语义去重与压缩的系统剖析，为模型级行为提供可解释的报告。

**🔧 技术方法**

核心技术：语义规划单元抽取、LLM 生成的层次化图谱构建、记忆压缩+外部工具验证的多代理错误检测、自动化修复建议推理、跨轨迹压缩与结构化剖析。

**📊 数据集**

使用了 LensBench 数据集（基于 Mixture-of-Thoughts，130 条长 CoT，含探索层级结构和细粒度错误标签）以及公开的多模型（DeepSeek-V4-Pro、MiniMax-M2.7、Qwen3.5-27B、Gemma-4-26B-A4B、Qwen3-32B）进行评测。

**📈 对比分析**

在 LensBench 上用节点类型准确率（NTA）和图编辑相似度（GES）评估可视化；用精确率/召回率/ F1 评估错误诊断。结果显示：可视化 NTA≈75%、GES≈70%；错误诊断微平均 F1 在 66–82 之间，Safety 检测始终保持高分，知识/逻辑错误检测受模型强度影响明显。

**⚠️ 局限性**

局限性：仅针对静态 CoT 轨迹，缺乏对动态多步代理交互的支持；实现仍为整体化，未构建模块化插件生态，难以轻量集成到不同工作流。

---

## 756. UniverSat: Resolution- and Modality-Agnostic Transformers for Earth Observation

**arXiv ID:** 2606.23503 | [PDF](https://arxiv.org/pdf/2606.23503v1)

**作者:** Yohann Perron `[一作]` (Ecole Nationale des Ponts et Chaussées), Loic Landrieu `[通讯]` (Ecole Nationale des Ponts et Chaussées)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并训练了一个名为UniverSat的可处理任意空间、光谱、时间分辨率的多模态视觉Transformer，用于地球观测任务。

**💡 创新点**

核心创新在于用通用补丁编码器（Universal Patch Encoder）替代传统固定patch投影器，实现单一权重即可映射不同传感器的补丁，并通过轴向交叉注意力实现模态融合和分辨率控制。

**🔧 技术方法**

采用轴向交叉注意力（Axial Cross-Attention）、自监督学习中的跨模态对比损失与多模态遮盖预测、线性变换的傅里叶特征提升嵌入维度、以及可在推理时指定输出分辨率的插值与跨子补丁注意力。

**📊 数据集**

在七个异构数据集上进行预训练，覆盖13种传感器（光学、雷达、激光雷达、地形高程、超光谱等），并在GeoBench、PangeaBench、SpectralEarth等16个公开基准上进行评测。

**📈 对比分析**

与现有专用与通用基座模型对比，UniverSat在多模态分类、语义分割和超光谱分析任务上取得或接近最先进性能，且在未见传感器配置下仍保持稳健；使用线性或轻量级解码器即可竞争甚至超越需要大型解码器的模型。

**⚠️ 局限性**

在极高分辨率或单模态场景中，通用模型的效率和精度可能略逊于专用模型；引入通用补丁编码器带来额外计算开销；对非光学传感器的泛化相对光学传感器略弱，需要额外的模态编码学习。

---

## 757. Ensuring Open Source Integrity: The Intersection of Copy-Based Reuse and License Compliance

**arXiv ID:** 2606.23495 | [PDF](https://arxiv.org/pdf/2606.23495v1)

**作者:** Mahmoud Jahanshahi `[一作]` (University of Tennessee), Audris Mockus `[通讯]` (University of Tennessee)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过构建基于复制的代码重用网络，系统量化并分析了开放源代码生态中不同许可类型对代码复制与潜在许可不合规风险的影响。

**💡 创新点**

①首次将整个开放源代码生态（World of Code）中所有复制实例与许可信息映射；②使用复制网络与传统依赖网络对比，揭示仅通过依赖检测能捕获不到大部分复制重用；③在回归模型中控制语言、项目规模等多重因素，细致评估许可类型对复制重用概率的真实影响。

**🔧 技术方法**

①利用World of Code（WoC）基础设施的P2L（项目-许可）和Ptb2Pt（复制关系）映射；②构建复制重用网络并聚合项目级复制次数；③执行二项式逻辑回归（控制多重协变量）与交互项分析；④对高风险复制对比依赖检测的覆盖率。

**📊 数据集**

World of Code（WoC）完整OSS数据集，涵盖约1.8 十亿个项目对组合，其中约1.79 十亿对被筛选为跨项目复制实例；此外，使用WoC提供的导入/导出语句映射评估依赖重用情况。

**📈 对比分析**

对复制网络中的高风险组合（至少10个复制文件）与传统依赖网络进行对比。结果显示，仅有约2.43% 的复制重用在任何时点通过依赖关系被检测到，说明传统依赖工具在捕获复制重用方面存在显著缺陷；在回归分析中，控制变量后，许可类型与复制重用的显著性与语言交互显现。

**⚠️ 局限性**

①WoC的P2L仅通过文件名识别许可，可能遗漏README或源文件中的许可信息；②项目级许可映射忽略子模块或第三方库的不同许可；③依赖网络对动态或隐式导入的捕获不完整；④聚焦复制重用可能低估了依赖重用的重要性。

---

## 758. Tighter Bounds for Algorithmic Complexity Estimation Using a Reusable Code-Based Block Decomposition Method

**arXiv ID:** 2606.23471 | [PDF](https://arxiv.org/pdf/2606.23471v1)

**作者:** Eduardo Yuji Sakabe `[一作]` (Karolinska Institute and King's College London), Hector Zenil `[通讯]` (Karolinska Institute and King's College London)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了 BDM 2.0，即在 Block Decomposition Method 上引入算法注意力和重用机制，通过条件描述来减少描述长度；

**💡 创新点**

创新点在于将不同块之间的算法信息共享纳入估算，允许程序层与观察层的条件描述，形成基于算法互信息的“算法注意力”；

**🔧 技术方法**

使用了 Coding Theorem Method (CTM) 的输出与条件估计，以及基于 CTM 的程序与块的复杂度表，构建可计算的重用成本；

**📊 数据集**

论文主要以理论分析为主，未具体给出实验数据集，示例多为理论构造与模拟；

**📈 对比分析**

比较方法是将 BDM 2.0 与原版 BDM 1.0 对比，证明在重用收益超过表示与验证开销时 BDM 2.0 更优；实验结果未给出，性能评估基于理论证明；

**⚠️ 局限性**

局限性包括：重用优化为 NP‑hard；需对 CTM 条件估计进行校准；当前仅支持单层重用，不能处理链式重用；当无可用重用时退化为 BDM 1.0，且对块分解方式敏感。

---

## 759. TriggerBench: Investigating Prospective Memory for Large Language Models

**arXiv ID:** 2606.23459 | [PDF](https://arxiv.org/pdf/2606.23459v1)

**作者:** Tianhua Zhang `[一作]` (Chinese University of Hong Kong), Yan Lu `[通讯]` (Microsoft Research Asia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TriggerBench基准，用于评估大型语言模型的前瞻性记忆（PM），并通过匹配的回溯性记忆（RM）探针进行对比；

**💡 创新点**

创新点包括：①设计了覆盖State‑Tracking、Temporal Grounding、Logical Adherence、Attention Recovery、Safe Coding五个维度的PM任务；②构造了正、负、过载三种对照变体，实现“constraint‑trigger decoupling”，避免词汇/语义捷径；③证明了PM在长上下文中形成“认知悬崖”，显著低于RM；

**🔧 技术方法**

采用长上下文LLM（Qwen、Gemma、GPT‑4系列）、推理思考模式、检索增强（RAG）以及三种记忆架构（A‑MEM、Mem0、Letta‑Sim），并使用GPT‑4o做自动评估器；

**📊 数据集**

基于82条人工蓝图扩展生成1,265个PM任务与440个RM探针，覆盖19个日常与专业领域；对话长度从约2.5K扩至40K token；

**📈 对比分析**

通过Slot Match和PM Accuracy对比不同模型、思考强度和变体，发现：高推理提升PM但伴随精确‑召回权衡；在40K token时，PM性能急剧下滑（>90%→<40%），而RM保持≈98%；

**⚠️ 局限性**

局限性包括：仅英文、数据规模相对有限、未覆盖多模态或多代理系统、评估人类多样性不足、数据生成过程中可能携带LLM偏见。

---

## 760. The Prevalence and Impact of Licenses in Open Software Projects

**arXiv ID:** 2606.23445 | [PDF](https://arxiv.org/pdf/2606.23445v1)

**作者:** Mahmoud Jahanshahi `[一作]` (University of Tennessee), Audris Mockus `[通讯]` (University of Tennessee)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文系统地审计了超过131M的开源项目，量化了许可证分布、保留率以及许可证变更对项目活跃度、贡献者和依赖关系的影响。

**💡 创新点**

创新点在于首次利用全生态系统级别的数据揭示83%项目无许可证、许可证类型随时间演变，并发现不同编程语言环境下许可证变更对项目绩效的异质效应。

**🔧 技术方法**

使用了World of Code平台的P2L映射、winnowing算法检测许可证、以及多元多重回归与MANOVA等统计技术对许可证与项目指标进行建模。

**📊 数据集**

主要数据集来自World of Code提供的约131,171,379个项目的完整提交历史与许可证记录，随后筛选出满足时间窗口且发生许可证变更的项目子集。

**📈 对比分析**

通过描述性统计和多元回归比较许可证变更方向与语言交互的效应，模型R²在0.02-0.09之间，显示尽管统计显著但变异性较大。

**⚠️ 局限性**

主要局限在于仅检测提交的许可证文件，忽略文件内嵌的许可证；对许可证临时删除或重置的追踪不足；缺乏对项目历史演化上下文的完整把握。

---

## 761. Autonomous Subsea Cable Search and Tracking with Graph-Optimised Priors and Visual Tracking

**arXiv ID:** 2606.23606 | [PDF](https://arxiv.org/pdf/2606.23606v1)

**作者:** Ibrahim Fadhil Djauhari `[一作]` (University of Southampton), Blair Thornton `[通讯]` (University of Southampton)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于实时视觉检测与图优化的AUV下海缆线搜索与跟踪方法，能够在先验路径不准的情况下实现缆线定位、跟踪和全局路线路径估计；

**💡 创新点**

创新点包括将图优化反向用于缆线路线路径校正、利用物理缆线悬链线模型限定搜索空间、以及将视觉检测结果与全局路径约束相结合实现鲁棒追踪；

**🔧 技术方法**

主要技术包括半监督视觉检测器（GeoCLR）、基于g2o的Levenberg‑Marquardt图优化、DVL/USBL/EKF定位、缆线悬链线物理约束和正交斜向搜索模式；

**📊 数据集**

使用了现场测试数据：一段120 m、27 mm直径的退役海底通信缆线，结合从英国电信集团提供的先验路径和在Cawsand湾收集的地形图像数据；

**📈 对比分析**

与密集式扫荡（lawnmower）模式对比，实验实现了52–59% 的检索覆盖率、21–35% 的距离效率，整体巡检效率比扫荡模式高2.6–4.4倍；

**⚠️ 局限性**

主要局限在于视觉检测的高误报率导致AUV偏离、CPU推理速度慢、以及部分岩石和沉积物导致的连续误报，影响整体覆盖率和恢复速度。

---

## 762. SPIRAL: Learning to Search and Aggregate

**arXiv ID:** 2606.23595 | [PDF](https://arxiv.org/pdf/2606.23595v1)

**作者:** Jubayer Ibn Hamid `[一作]` (Stanford University), Noah Goodman `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练语言模型在推理阶段同时使用顺序推理、并行推理和聚合推理三种计算模式，以提升推理效果。

**💡 创新点**

创新点在于将集合强化学习（set‑RL）与传统强化学习（RL）联合起来，形成统一的端到端训练框架，使模型能主动生成有用的并行推理轨迹并有效聚合，从而实现比传统仅优化顺序推理更好的推理计算利用率。

**🔧 技术方法**

采用集合强化学习（set‑RL）来优化并行轨迹的集合质量，采用标准强化学习（GRPO）来优化聚合轨迹的生成，并在训练中使用多层采样策略和边际优势估计实现低方差梯度。

**📊 数据集**

在数学推理任务上使用从 MathReasoning 数据集（过滤后的子集）进行训练和评估。

**📈 对比分析**

与基准方法 GRPO 进行对比；在相同训练计算预算下，实验显示该方法在并行推理扩展时的 Pass@k 速度提升可达 11×，在聚合推理上提高约 15% 的准确率；在递归自聚合实验中提升 13.5%。

**⚠️ 局限性**

局限性包括：实验规模仍较小（仅 2‑3B 参数模型），缺乏对模型验证能力的细粒度分析；对不同推理脚本的泛化性能未充分评估；训练过程对采样设置敏感，需进一步探索更大规模模型和多任务场景。

---

## 763. A Generative Model for Closed-Loop Microsimulation of Signalized Intersections

**arXiv ID:** 2606.23588 | [PDF](https://arxiv.org/pdf/2606.23588v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 764. HoloAgent-0: A Unified Embodied Agent Framework with 3D Spatial Memory

**arXiv ID:** 2606.23565 | [PDF](https://arxiv.org/pdf/2606.23565v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 765. KEMO: Event-Driven Keyframe Memory for Long-Horizon Robot Manipulation with VLA Policies

**arXiv ID:** 2606.23589 | [PDF](https://arxiv.org/pdf/2606.23589v1)

**作者:** Yihan Zeng `[一作]` (Hong Kong Embodied AI Lab), Zhongyu Li `[通讯]` (Hong Kong Embodied AI Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 KEMO 插件框架，利用机器人运动与视觉检测关键帧并存入记忆，用以增强 VLA（Vision‑Language‑Action）策略解决长时序机器人操控中的阶段歧义问题。

**💡 创新点**

①仅用机器人关节状态和视觉变化实现无任务标签、无 VLM 的事件驱动关键帧检测；②将关键帧压缩为紧凑 token，并通过交叉注意力与门控残差融合到当前视觉表征；③在训练时对关键帧附近样本加权，突出事件相关的学习信号。

**🔧 技术方法**

运动衰减峰值检测、视觉相似度滤波（DINOv2）、SigLIP 编码、跨注意力 + 门控残差融合、关键帧加权的流匹配损失、VLA 基座。

**📊 数据集**

在真实双臂机器人上收集的 6 种长期任务（Swap Foods、Find Block、Cover Blocks、Box Refill、Make Sandwich、Drawer Items Replacement），每个任务 100–200 条示例。

**📈 对比分析**

与无记忆基线 π_0.5 和稠密记忆基线 MemoryVLA 对比，KEMO 在 12 次试验中 TSR 达 51.4%（比 π_0.5 提升 23.6%），SCR 达 76.4%（比 π_0.5 提升 34.1%），显著优于两基线。

**⚠️ 局限性**

需手动调节峰值检测窗口、阈值、记忆容量 K；记忆容量需根据任务手动设定，缺乏自适应动态事件选择，限制跨任务迁移。

---

## 766. Kamera: Unified Position-Invariant Multimodal KV Cache for Training-Free Reuse

**arXiv ID:** 2606.23581 | [PDF](https://arxiv.org/pdf/2606.23581v1)

**作者:** Bole Ma `[一作]` (Erlangen National High Performance Computing Center), Gerhard Wellein `[通讯]` (Erlangen National High Performance Computing Center)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种位置无关的多模态 KV 缓存重用机制，利用低秩补丁恢复被缓存块的跨块上下文绑定，从而在不重新编码视觉特征的情况下显著提升多跳推理精度。

**💡 创新点**

创新点在于将 KV 的位置信息与跨块条件完全分离，并使用低秩（rank‑m）补丁补偿缺失的上下文绑定；该机制一次实现对 MLA、GQA、MHA 三种注意力结构的统一重用，支持滑动窗口、重排序与回溯等多模态代理常见模式，且无需重新编码。

**🔧 技术方法**

核心技术包括：RoPE 位置重旋转、低秩 SVD 补丁、位置自由内容缓存、训练‑free 补丁生成、SGLang 生产级分页注意力核改造；评估基于 GQA、深度堆叠 GQA、MLA 等模型。

**📊 数据集**

使用的数据集涵盖 MM‑NIAH、GQA、两页文档问答、Video‑MME、EgoSchema 等多模态与视频问答数据集。

**📈 对比分析**

与传统前缀缓存、token‑重编码基线（CacheBlend、VLCache 等）比较，补丁方案在跨块绑定任务中恢复 97–100% 的重编码损失；单跳精度保持不变，双跳精度从约 0.28 提升至 0.53；整体在 KV 字节占用仅 25%（rank‑64）或 6%（rank‑16）的情况下保持 3× 更低的 KV 开销；在线 SGLang 引擎中，重写 KV 的下一个 token KL 仅 10⁻³，准确率差距 <3 点。

**⚠️ 局限性**

局限性：对冗余视觉/视频流最有效，音频或纯文本的收益有限；补丁生成需一次前向推理，且深层视觉模型（deepstack）仍需额外深层补丁；在极大窗口或块尺寸下的存储开销与对不同注意力架构的通用性仍待进一步验证。

---

## 767. A Spectral Theory of Normalized Corrected GNN Propagation

**arXiv ID:** 2606.23572 | [PDF](https://arxiv.org/pdf/2606.23572v1)

**作者:** Qihan Chen `[一作]` (Fuzhou University), Jianfeng Hou `[通讯]` (Fuzhou University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并分析了去除归一化邻接矩阵中的度相关常数项的GNN传播算子，证明其在二元CSBM模型下对数层深度仍能实现精确恢复，并给出多类别的部分恢复定理。

**💡 创新点**

首次将度相关常数项的消除与归一化GNN结合，使用原子展开和装饰化步计数实现残差幂的逐点控制，给出在对数层深度下的高概率精确恢复保证。

**🔧 技术方法**

采用谱理论、Krylov子空间估计、原子展开、装饰化步计数以及高概率矩估计等技术进行理论证明，并用模拟与真实图实验验证。

**📊 数据集**

在合成的CSBM图上以及Cora、CiteSeer、PubMed、Reddit、ogbn-arxiv、ogbn-products等公开节点分类数据集（含平衡二类子集）上进行实验。

**📈 对比分析**

与标准归一化传播以及DropEdge、GraphMamba、RevGNN等过平滑基线对比，实验显示去度校正方法在保持准确率的同时显著降低了深度相关的性能下降。

**⚠️ 局限性**

局限性包括仅针对线性传播背骨，要求图密度满足多对数阈值，未覆盖极端异质度/异亲性情况，且理论尚未收敛至最优阈值，实际大图需进一步提升可扩展性。

---

## 768. AwakeForest: An Interactive Geospatial Platform for Large-Scale Forest Imagery

**arXiv ID:** 2606.23542 | [PDF](https://arxiv.org/pdf/2606.23542v1)

**作者:** Suraj Prasai `[一作]` (Wake Forest University), Fan Yang `[通讯]` (Wake Forest University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一个面向大规模森林影像的交互式端到端平台，实现了模型辅助推理、自动标注和人工迭代修正于一体的工作流。

**💡 创新点**

创新点在于将地理空间原生管理、云优化的COG影像访问与可插拔的预训练模型推理整合到统一的交互式系统，并支持基于Patch和全景视图的标注与可视化。

**🔧 技术方法**

采用了Next.js、FastAPI、Leaflet、TiTiler、Supabase/PostGIS、MinIO等技术栈；模型推理服务支持YOLO、SAM等对象检测与分割模型。

**📊 数据集**

使用了PALMS UAV高分辨率正射影像数据集进行演示。

**📈 对比分析**

与Label Studio、AnyLabeling、CVAT、QGIS等现有工具对比，AwakeForest在地理空间原生、云优化、AI辅助标注方面表现更优；示例显示可在大规模影像上实现高效交互与精确标注，未给出定量指标。

**⚠️ 局限性**

局限性包括对模型的依赖（领域迁移时可能需更多人工修正）、推理延迟与可视化开销、未覆盖多光谱、LiDAR等多模态数据及多人协作功能。

---

## 769. Computing Gaussian and exponential integrals in ${\Bbb R}^n$

**arXiv ID:** 2606.23556 | [PDF](https://arxiv.org/pdf/2606.23556v1)

**作者:** Alexander Barvinok `[一作]` `[通讯]` (University of Michigan), Alexander Barvinok (University of Michigan)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文研究了在R^n中计算高斯和指数积分的期望，特别是形式为exp{∑_i=1^m ϕ_i}的期望，其中ϕ_i是依赖于少数坐标的函数。

**💡 创新点**

创新点在于提出了在Lipschitz常数和函数依赖组合的条件下，如何有效地近似这些积分，并讨论了其在计算体积和整数点统计中的应用。

**🔧 技术方法**

使用了插值方法和高斯测度、对称指数测度的技术。

**📊 数据集**

没有具体提到使用的数据集，但提到的应用包括多面体中的整数点计数和体积计算。

**📈 对比分析**

通过与现有方法的比较，证明了在特定条件下，所提出的方法能够在多项式时间内有效地近似期望值，性能优于传统方法。

**⚠️ 局限性**

限制在于需要控制函数的Lipschitz常数和依赖关系的组合，且在某些情况下，可能需要更强的分析性质。

---

## 770. SQLConductor: Search-to-Policy Learning for Step-wise Text-to-SQL Orchestration

**arXiv ID:** 2606.23537 | [PDF](https://arxiv.org/pdf/2606.23537v1)

**作者:** Yizhang Zhu `[一作]`, Yuyu Luo `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种面向文本到SQL任务的逐步工作流编排框架（Step‑wise Orchestration），通过动态选择和组合专门的子任务模块（如模式链接、值检索、生成、修订等）来完成自然语言查询的执行；

**💡 创新点**

核心创新在于引入“Search‑to‑Policy Learning”——先用蒙特卡洛树搜索（MCTS）全局探索高质量工作流，再通过稳定性估计筛选可靠轨迹，并利用稳定性加权的监督微调和历程式强化学习训练出适配不同查询的编排策略；

**🔧 技术方法**

采用的技术包括：1) MCTS对工作流空间进行树搜索；2) 稳定性评估与工作流监督筛选；3) 稳定性加权的监督微调（SFT）；4) 课程式强化学习（Curriculum RL）；5) 以大型LLM为底层的专用动作模块；6) 基于 Qwen3-8B 或 Qwen2.5‑Coder-32B‑Instruct 的模型；

**📊 数据集**

实验使用了 BIRD‑Train 用于工作流生成与策略训练，BIRD‑Dev 作为主测集；另外在无额外训练的 OOD 数据集 Spider‑Test、KaggleDBQA 与 ScienceBenchmark 上评估泛化能力；

**📈 对比分析**

在 BIRD‑Dev 上，所提出的模型在 73.2% EX 取得最高成绩，超过所有对比基线（包括 CodeS、OmniSQL、CHESS、DeepEye‑SQL、Alpha‑SQL、SquRL 等）并在 OOD 数据集上平均 72.4% 的 EX 也优于传统固定管道与动态工作流方法；同时在推理成本上每条查询仅需约 7.1k 输入/12.2k 输出 token，费用约 0.012 美元，显著低于现有方案；

**⚠️ 局限性**

主要局限包括：① 对动作模块的依赖，若底层 LLM 在特定子任务上表现欠佳，整体效果会受限；② 工作流搜索虽覆盖广泛，但在极大/极复杂数据库上仍可能遇到搜索空间爆炸；③ 目前的稳定性评估与过滤方法仍需手动设定阈值，可能不适用于所有领域；④ 该框架在完全无监督或极少标签的情形下的迁移性能尚待进一步验证。

---

## 771. Scheduling Thoughts: Learning the Order of Thought in Diffusion Language Models

**arXiv ID:** 2606.23567 | [PDF](https://arxiv.org/pdf/2606.23567v1)

**作者:** Jiawei Xu `[一作]` (University of Maryland), Furong Huang `[通讯]` (University of Maryland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Self-Aware Scheduling（SAS）框架，通过学习掩码扩散模型的“思考顺序”来优化解码过程；

**💡 创新点**

核心创新在于将解码顺序视为可学习策略，并以模型自身的路径似然为密集奖励，理论上可逼近 KL 下界，且与先前的启发式或稀疏奖励方法形成对比；

**🔧 技术方法**

技术包括：掩码扩散模型、密集自知奖励、Group Relative Policy Optimization（GRPO）强化学习、无模型学习（policy only）以及后期自监督微调；

**📊 数据集**

使用的数据集包括 1B 掩码扩散模型训练的 1M Sudoku 任务、LLaDA-8B 扩散 LLM 在 GSM8K（数学推理）和 MBPP（代码推理）上的训练与评测；

**📈 对比分析**

与随机、置信度、边际、熵及专家人类解法等启发式调度进行比较，SAS 在 Sudoku 任务中从 82% 提升至 91.8%（再微调后 97.5%），在 GSM8K 上从 64% 提升至 76%；在不同长度与块大小下均保持稳定或优于基线；

**⚠️ 局限性**

局限性包括：训练使用教师强迫的自知奖励导致训练-推理间不匹配；政策多为任务特定，跨任务迁移仍有限；学习过程需要额外训练开销，推理时略有开销。

---

## 772. Real-Time Multimodal Activity-Aware Error Detection in Robot-Assisted Surgery

**arXiv ID:** 2606.23593 | [PDF](https://arxiv.org/pdf/2606.23593v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 773. Discovering Latent Groups for Robust Classification

**arXiv ID:** 2606.23609 | [PDF](https://arxiv.org/pdf/2606.23609v1)

**作者:** Ankur Garg `[一作]`, Vincent Michalski `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出神经分类树（NCT），通过训练阶段基于预测正确性对样本进行易/难分支路由，最终构建一个可解释的树形网络，以对抗数据中的伪相关性。

**💡 创新点**

创新点在于将路由信息从仅在训练时临时使用转为永久编码在网络结构中，并配合无监督深度选择、稀疏节点合并等机制，使得子组划分在推理阶段可直接观察到。

**🔧 技术方法**

使用的技术包括：迭代训练与两阶段优化、基于正确率的路由规则、辅助父子损失保证稳定性、伪WGA深度停止准则、稀疏节点合并、LayerGradCAM 等可解释性工具。

**📊 数据集**

实验数据集涵盖 five 个 spurious‑correlation benchmark：waterbirds、CelebA、ISIC、U‑MNIST 与 CMNIST。

**📈 对比分析**

与 8 个基线（包括 GDRO、JTT、CnC、EIIL、DFR、ERM、GEORGE、ExMap）在 worst‑group accuracy、平均 accuracy（或 AUROC）进行比较，NCT 在无监督/验证无监督层级上表现竞争，尤其在 ISIC 上取得最优或接近最优成绩。

**⚠️ 局限性**

局限性包括：依赖样本的 simplicity‑bias；若少数类本身易学或深度停止失效，路由可能无法正确隔离子组；在极端类别不平衡时易导致误分或稀疏叶子占优。

---

## 774. Scaling Linear Mode Connectivity and Merging to Billion Parameter Pretrained Transformers

**arXiv ID:** 2606.23607 | [PDF](https://arxiv.org/pdf/2606.23607v1)

**作者:** Tianyi Li `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Zhiqiang Shen `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种可扩展的双端学习对齐框架，使得大规模预训练 Transformer 在通过线性插值合并时能保持低损失障碍。

**💡 创新点**

创新点在于：①系统化地列举并参数化 Transformer 的功能保持对称性（归一化吸收、残差空间旋转、注意力头置换、内部 QK/OV 对称、FFN 置换/GLU 缩放）；②引入双端学习（Dual Learned Matching）共同优化两个模型的对称变换，显著降低插值障碍；③针对连续对称使用 Cayley 与极化参数化实现可微化。

**🔧 技术方法**

技术手段包括：功能保持对称性变换、权重匹配 + 学习匹配、双端学习优化、正交和可逆矩阵的 Cayley 与极化参数化、实验中使用的损失与准确率评估。

**📊 数据集**

使用公开的独立训练检查点：Vision Transformer (ViT-S/B/L) 在 ImageNet‑1K、语言模型 Pythia（14M-6.9B）在 WikiText、以及 OLMo（7B）在相同任务。

**📈 对比分析**

与传统权重匹配（WM）和单侧学习匹配（LM）对比，双端学习匹配在所有规模下实现最低的损失障碍；在 ViT‑L 过程中 ImageNet‑1K top‑1 准确率保持>69%，Pythia‑160M 在 WikiText 的困惑度仅略高于端点，表明方法实用且性能优异。

**⚠️ 局限性**

局限性包括：①对称性参数化仍受限于连续对称，离散对称（如头置换、FFN 置换）仍手动固定；②在极大规模（如 OLMo‑7B）损失障碍仍显著，表明模型训练方式、数据分布等因素会影响 LMC；③实验主要针对 Transformer，尚未验证到更广泛架构或训练设置。

---

## 775. Evaluation Awareness Is Not One Capability: Evidence from Open Language Models

**arXiv ID:** 2606.23583 | [PDF](https://arxiv.org/pdf/2606.23583v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 776. Solve for the Hyperparameter, Skip the Search: Kolmogorov-Optimal Scaling Laws for Spline Regression

**arXiv ID:** 2606.23575 | [PDF](https://arxiv.org/pdf/2606.23575v1)

**作者:** Yong Yi Bay `[一作]`, Kathleen A. Yearick `[通讯]` (University of Illinois at Urbana Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对结构化样条回归（加性与稀疏二阶）提出一种闭式求解最佳分辨率G的方法，利用两次拟合直接估计偏差-方差曲线，并通过解析求解获得最优G，从而消除传统的网格搜索。

**💡 创新点**

（1）证明最优分辨率可由有效密度 n/s_r 按幂律决定；（2）提出 Kolmogorov‑optimal Order‑aware Resolution Estimation（KORE）算法；（3）在高维低阶结构下实现搜索‑free 超越全网格 CV；（4）通过两点校准的 2×2 线性系统估计偏差/方差尺度并闭式求解 G；（5）提供一致性证明并在多种实验中验证。

**🔧 技术方法**

B‑splines 逼近理论、Kolmogorov n‑width、偏差‑方差分解、PRESS  留一误差公式、线性平滑器的闭式 LOO 评估、两点校准的 2×2 系统求解、解析求解 G、有效密度缩放、实验评估。

**📊 数据集**

① 受控合成加性与稀疏二阶目标；② 12个经典非参数回归基准（Nguyen、Friedman、SparseAdd、SparsePair 等）；③ 36 个实际表格回归数据集（OpenML‑CTR23+Combined Cycle Power Plant）。

**📈 对比分析**

与完整网格搜索 3‑折 CV、GCV、Mallows Cp、AIC、BIC 以及梯度提升、核回归、kNN、MLP 等 21 种基线比较。KORE 在所有基准上与 CV RMSE 相当或更好，仅需约 8× 更少的模型拟合；在实际数据上 Compute‑Normalized Lift 上排名第一，超过所有基线。

**⚠️ 局限性**

仅适用于平滑低阶结构（加性或稀疏二阶），对高阶交互、空间异质性或多尺度信号失效；需要满足 Gram 矩阵稳定性和有效密度足够；对异方差或重尾噪声的鲁棒性有限；若样本量不足，两点校准可能不稳定。

---

## 777. An Infinitary Lambda Calculus with Global Trace Condition (Extended Abstract)

**arXiv ID:** 2606.23573 | [PDF](https://arxiv.org/pdf/2606.23573v1)

**作者:** Stefano Berardi `[一作]` (Turin University), Daniel Osorio-Valencia `[通讯]` (Turin University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文扩展了Kennaway等人的无穷λ演算，并在其上构造了类似哥德尔系统T的类型系统，引入全局跟踪条件（GTC）来判定可达性与总性，并证明在满足GTC的情况下所有无限归约都强收敛，从而证实所有闭合类型为Nat的项最终归约为某个数字。

**💡 创新点**

创新点在于：①提出全局跟踪条件（GTC）这一新的判定机制；②将GTC与无限归约深度相关联，得到强收敛性结果；③证明该扩展与Das的循环系统T、哥德尔系统T在可表达性上等价，提供了新的对角化与循环证明的桥梁。

**🔧 技术方法**

主要技术包括：共诱导定义无穷行术语、α等价与核心递归替换、对类型推导树的红/蓝箭头跟踪关系、归约深度与ω-收敛性分析以及利用超限归约序列的度量空间理论。

**📊 数据集**

未使用任何数据集；研究完全基于形式化证明。

**📈 对比分析**

本文未进行实验或性能比较；评价完全基于理论证明与归约性质的分析。

**⚠️ 局限性**

局限性在于：①对非正则术语的处理仍未完全；②系统与哥德尔系统T等价的证明仍为猜想，尚需进一步形式化；③缺乏对可计算函数的完整可实现性证明。

---

## 778. Patient-Aware Contrastive Learning Preserves Per-Patient Structure in RR-Interval Representations

**arXiv ID:** 2606.23570 | [PDF](https://arxiv.org/pdf/2606.23570v1)

**作者:** Yasantha Niroshana `[一作]` (University of Moratuwa), Chathuranga Hettiarachchi `[通讯]` (University of Moratuwa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

设计了一种将正样本限定为同患者同类别的对比学习目标，用于心律失常（AF）检测。

**💡 创新点**

创新点在于通过同患者同类别正样本构造，既保持每个患者自身基线（SR）结构，又能有效分离不同类别，提升跨患者泛化能力。

**🔧 技术方法**

采用了InfoNCE 对比损失、轻量级多分支 1D-CNN 编码器、Softmax 时序注意力模块以及冻结编码器的线性探测方法。

**📊 数据集**

使用 IRIDIA‑AF 长时单导联 ECG 数据集（RRI 序列），包含 167 名患者的 AF 事件。

**📈 对比分析**

与监督对比学习（SupCon）和二分类交叉熵（BCE）比较，所提出目标在每患者 SR 一致性最高，线性探测 AUROC 为 0.989±0.003，种子方差比 SupCon 降低 2.6 倍。

**⚠️ 局限性**

实验仅在单一数据集上进行，缺乏多中心验证与端到端微调，且尚未验证该方法在其他生理信号上的通用性。

---

## 779. Protection Switching in Hybrid Hollow-Core and Single-Mode Fiber Networks: Challenges, Analysis, and Mitigation Strategies

**arXiv ID:** 2606.23554 | [PDF](https://arxiv.org/pdf/2606.23554v1)

**作者:** Md Ghulam Saber `[一作]` (Huawei Technologies Canada), Zhiping Jiang `[通讯]` (Huawei Technologies Canada)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

分析混合空心核心光纤（HCF）与单模光纤（SMF）网络的保护切换挑战，并通过 Monte Carlo 仿真比较 1+1 与 SBPP 保护架构的性能。

**💡 创新点**

首次量化交叉纤维保护切换的异向性影响，并提出 DSP 预加载、波谱预均衡等缓解策略。

**🔧 技术方法**

使用蒙特卡罗仿真、Suurballe 路由、Dijkstra 路由、Gaussian noise 模型、DSP 预加载、路径特征预存等技术。

**📊 数据集**

六个标准参考拓扑（NSFNET、COST 239、COST 266、Nobel Germany、USNET、CORONET）与随机 HCF/SMF 链路分配。

**📈 对比分析**

通过比较 CD 步、GSNR 惩罚、调制降级和容量保留等指标进行性能评估，结果表明 1+1 在稀疏拓扑中优于 SBPP，且 HCF 投入比例提升可使容量保留从 54–91% 递增至 85–99%。

**⚠️ 局限性**

缺乏对真实跨纤维切换时 DSP 重收敛时间的实验验证，未考虑多供应商兼容性和气体吸收随时间变化的动态影响。

---

## 780. UI-LIC: A Unified Framework for Evaluating Learned Image Compression Models

**arXiv ID:** 2606.23545 | [PDF](https://arxiv.org/pdf/2606.23545v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

---

## 781. VeriEvol: Scaling Multimodal Mathematical Reasoning via Verifiable Evol-Instruct

**arXiv ID:** 2606.23543 | [PDF](https://arxiv.org/pdf/2606.23543v1)

**作者:** Haoling Li `[一作]` (Tsinghua University), Yujiu Yang `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可扩展的视觉数学推理RL数据构造框架VeriEvol，先通过路由特定的提示演化生成更难、图像依赖的问题，再用多源假设检验的验证器确认答案，最终生成可用于SFT和RL的标注。

**💡 创新点**

①将提示演化与答案验证分离，放在数据构造阶段；②设计多求解器+程序+视觉检验的假设检验器，使答案在进入RL前已被多渠道反证；③提供完整可追踪的验证记录，支持验证通道的可扩展性；④在视觉数学推理任务中验证规模化、可靠性与性能可加性的独立性。

**🔧 技术方法**

路由与类型感知提示演化、三路求解器自洽+程序执行+视觉检验的假设检验框架、Deterministic接受门、GRPO强化学习、SFT微调、Python执行工具、OCR与图像特征检测、推理链评估等。

**📊 数据集**

SFT数据：Honey‑Data‑15M（10K–250K）; RL数据：多模态推理语料（10K–130K）; 预训练模型Seed‑2.0‑Pro用于提示生成，Gemini‑3‑flash用于验证；评测基准：MathVista_MINI、MathVision_MINI、MathVerse Vision‑Only、DynaMath‑Worst、We‑Math‑Strict。

**📈 对比分析**

在相同Qwen2.5‑VL‑7B‑Instruct模型和GRPO策略下，与外部7B基线对比。SFT阶段从10K提升到250K平均+19.31；RL阶段在未演化+未验证基线上平均+3.88（MathVision_MINI +5.92）。验证数据从10K到130K平均+4.60。验证器在推理时的准确率提升+4.51pp。与外部基线相比，MathVerse‑VO上领先+6.66点，整体平均接近最高外部系统。

**⚠️ 局限性**

①依赖种子池覆盖，演化难以弥补缺失领域；②验证门不完美，可能漏判错误或误判；③RL实验仅使用单一seed，结果可重复性有限；④验证与策略分离可能导致验证通道与策略不匹配；⑤数据规模受源授权与隐私限制；⑥模型容量限制导致部分错误仍无法消除。

---

## 782. POTracker: Optimizing Large Language Models for Standard-Compliant Power Outage Report Generation

**arXiv ID:** 2606.23533 | [PDF](https://arxiv.org/pdf/2606.23533v1)

**作者:** Hung Phan `[一作]` (Iowa State University), Ali Jannesari `[通讯]` (Iowa State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了POTracker，一个针对美国电力停电报告生成的LLM优化框架，能够把非标准文本或XML转为符合CIM-IEC-61968-3规范的机器可读报告。

**💡 创新点**

核心创新包括POTracker_Loss——将文本相似度与标签结构相似度结合的自定义损失函数，以及POTracker_Metric——用于量化生成报告与标准报告在内容和结构上的一致性的评估指标。

**🔧 技术方法**

使用技术包括：Qwen2.5-7B-Instruct的微调、交叉熵与POTracker_Loss的加权训练、直接偏好优化（DPO）、以及对比的基线方法（Token Weight CE、Tag Masking CE、Example Weight CE、规则化转换）。

**📊 数据集**

数据集为1000条美国停电报告，先通过GPT‑5.2生成标准XML标签，再从中抽取800条训练、100条验证、100条测试；数据来自ODIN（Outage Data Initiative Nationwide）。

**📈 对比分析**

评估采用POTracker_Metric（α=0.5）及其极端权重（文本精度/结构精度）进行对比；POTracker在整体准确率上比原始模型提升51%，结构准确率达86.47%；相较于五种微调基线和规则化基线，其性能显著优于所有对手。

**⚠️ 局限性**

局限性包括：仅针对电力停电领域，难以直接推广至其他领域；依赖GPT生成的人工合成标准标签，可能存在偏差和多样性不足；并未解决真正真实标签稀缺的问题。

---

## 783. BiliVLA: Scene-Aware Vision-Language-Action Model with Reinforcement Learning for Autonomous Biliary Endoscopic Navigation

**arXiv ID:** 2606.23531 | [PDF](https://arxiv.org/pdf/2606.23531v1)

**作者:** Jinsong Lin `[一作]` (Chinese University of Hong Kong), Hongliang Ren `[通讯]` (Chinese University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种名为BiliVLA的场景感知视觉‑语言‑动作（VLA）框架，能够根据内镜图像与阶段性语言指令，联合预测目标类别、定位框和离散三自由度的连续内镜动作，实现ERCP（胰胆道内镜检查）中自主导航与结石定位。

**💡 创新点**

创新点包括：①将语言指令与语义定位、目标框和动作生成统一在同一端到端策略；②引入场景感知监督提升语义一致性；③设计安全感知回收监督，针对腔壁接触实现保守退避；④采用两阶段训练：先用LoRA在大规模视觉‑语言预训练模型上进行场景增强监督微调，再通过Group Relative Policy Optimization（GRPO）进行奖励导向的强化学习优化。

**🔧 技术方法**

使用的技术有：大规模视觉‑语言预训练模型 Qwen3‑VL‑8B + LoRA 微调；Vision Transformer 与多层感知器投影；Transformer 端到端自回归解码；Group Relative Policy Optimization（GRPO）奖励学习；基于 YOLOv11 的目标框标注与离散动作标注。

**📊 数据集**

数据集：自制的 BiliVLA‑Motion 数据集，包含 10k 张图像–动作对，覆盖进入导航、腔道穿梭和结石定位三任务，并配有阶段性语言指令与目标框标签。

**📈 对比分析**

方法对比：在三项 ERCP 子任务（入口导航、腔道穿梭、结石定位）上与 imitation learning、Qwen3‑VL、EndoVLA 基线比较。BiliVLA 在 mIoU、动作精准率（PR）和任务成功率（SR）上均显著优越，整体 SR 达 84.85%，动作精准率 91.96%。

**⚠️ 局限性**

局限性：仅在仿真模型（phantom）环境下验证，缺乏真实组织或临床试验；对极端光照、强反射、严重遮挡等真实内镜图像的鲁棒性仍待进一步验证；动作空间为离散三自由度，可能限制对更细腻控制的适应；未评估长期连续操作下的稳定性与安全性。

---

## 784. Self-Compacting Language Model Agents

**arXiv ID:** 2606.23525 | [PDF](https://arxiv.org/pdf/2606.23525v1)

**作者:** Tianjian Li `[一作]` (Johns Hopkins University), Daniel Khashabi `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关的“自适应压缩”框架（SelfCompact），通过在推理过程中使用摘要工具和轻量化的规则表（rubric）来动态决定何时对长轨迹进行压缩，从而避免上下文衰减（context rot）。

**💡 创新点**

创新点在于：① 通过一个简洁的规则表让语言模型自行评估是否已完成子任务或轨迹已趋于稳定，从而精准触发压缩；② 将摘要工具与规则表集成在同一推理引擎中，无需额外模型或训练；③ 采用 KV‑cache 复用技术，显著降低压缩调用的算力与成本。

**🔧 技术方法**

技术手段包括：① 内联摘要工具（同一模型的 Summarizer），② 以“是/否”决策形式的规则表，用于判断是否触发压缩；③ 在每个预定间隔向上下文追加规则表并采样决策；④ KV‑cache 复用避免重新编码完整轨迹；⑤ 对比基准（无压缩、固定间隔压缩、删除全部、保留最近 N）进行实验评估。

**📊 数据集**

使用的主要数据集：
- 竞争性数学：IMO‑Answerbench、HMMT Nov 2025、HMMT Feb 2026；
- 代理搜索：BrowseComp、BrowseComp‑Plus、DeepSearch QA。

**📈 对比分析**

评估方法：在相同的 token‑budget 或相同的预算下，将 SelfCompact 与无压缩、固定间隔压缩、删除全部、保留最近 N 等方法进行对比；在数学任务中以平均准确率为指标，在搜索任务中以准确率和 per‑question 成本（美元）为指标。结果显示：
- 在数学任务中，SelfCompact 在 11/12 组合中获得最佳分数，最高可提升 18.1 分；
- 在搜索任务中，SelfCompact 在 3 个模型中均优于所有基线，提升 5–9 分，同时每题成本下降 30–70%；
- 相较固定间隔压缩，SelfCompact 在 token‑cost 上也显著更低。

**⚠️ 局限性**

局限性：
- 仅在开源模型上验证，未在 GPT‑4/Claude 等前沿模型上评估；
- 方案完全训练无关，缺乏针对压缩决策的强化学习或微调，可能在更复杂情境下表现不足；
- 规则表的设计需要针对任务手工编写，通用性与可迁移性尚待验证；
- 对极端长序列或多模态任务的适用性尚未探究。

---

## 785. Concordia: JIT-Compiled Persistent-Kernel Checkpointing for Fault-Tolerant LLM Inference

**arXiv ID:** 2606.23521 | [PDF](https://arxiv.org/pdf/2606.23521v1)

**作者:** Yuhang Gan `[一作]` (University of California Santa Cruz), Chen Qian `[通讯]` (University of California Santa Cruz)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个 GPU 设备驻留的持久化内核检查点运行时，用于在 LLM 推理过程中自动保存并恢复 KV 缓存、适配器等状态，提升故障恢复效率。

**💡 创新点**

创新点在于：① 在 GPU 侧使用 PTX/SASS 级插桩和 JIT 编译的 delta‑checkpoint 处理器，实现不依赖主机 CPU 的设备驻留检查点；② 通过锁自由环形缓冲区和 CXL/DRAM 日志实现高效的异步记录与恢复；③ 通过 GPU 侧稀疏更新检测，大幅降低检查点成本。

**🔧 技术方法**

采用的技术包括：GPU 持久化内核、PTX/SASS 插桩、JIT 编译、锁自由任务环形缓冲区、CXL 或主机 DRAM 日志、GPU 侧 delta 检查点与恢复机制、HBM 带宽下的稀疏状态检测。

**📊 数据集**

评估使用在 RTX PRO 6000 Blackwell GPU 上的 LLM 推理任务（如 LLaMA/ChatGPT 等模型），未公开具体数据集名称。

**📈 对比分析**

与 CPU 侧页面扫描进行对比，GPU 侧 delta 检查点速度提升 219×；在两 GPU 恢复场景中，与重启 NCCL 并重新加载模型对比，恢复时间从数十秒缩短到约 1.5 秒，显示出显著的性能优势。

**⚠️ 局限性**

主要局限性包括：依赖特定 GPU 与 CXL 内存环境；仅适用于 KV 缓存稀疏更新的推理工作负载；需要框架支持 PTX/SASS 级插桩；假设故障为完全停止，未覆盖更复杂的网络或软件错误；日志回放验证规模有限。

---

## 786. Data Selection Through Iterative Self-Filtering for Vision-Language Settings

**arXiv ID:** 2606.23611 | [PDF](https://arxiv.org/pdf/2606.23611v1)

**作者:** Andrei Liviu Nicolicioiu `[一作]` (Mila, Université de Montréal), Aaron Courville `[通讯]` (Mila, Université de Montréal)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在不使用任何预训练模型或额外数据的情况下，提出一种自我过滤（Self‑Filtering）方法，通过循环训练 CLIP 模型并利用其自身的对比相似度来挑选可能干净的图文对，再与原始数据混合继续训练，形成一种自我提升的迭代过程。

**💡 创新点**

核心创新在于：① 用当前模型的余弦相似度作为过滤信号；② 在每轮训练后将被选中的高分样本加权重翻倍，与全部数据混合，兼顾利用已学知识与保持多样性；③ 通过单一模型连续迭代而非多次从零开始训练，显著减少计算成本并提升数据质量。

**🔧 技术方法**

技术上实现了：CLIP 视觉‑语言对比学习、基于余弦相似度的样本打分、数据重采样与加权混合、循环训练与过滤的闭环流程。

**📊 数据集**

主要使用 Datacomp 数据集，分别在 small（约 11.49M 样本）和 medium（约 128M 样本）子集上进行实验。

**📈 对比分析**

与基线（全量数据训练）、OpenAI CLIP 过滤数据以及其与全量数据混合的模型进行对比；在 Datacomp 的 38 项零样本任务上，Self‑Filtering 在 ImageNet、ImageNetV2、DTD、MSCOCO 等任务均取得了比基线更高的平均精度（约 19.2% vs 19.0%），并与使用更大预训练模型过滤得到的数据表现相当。

**⚠️ 局限性**

局限性包括：1）使用余弦相似度作为过滤标准可能与下游任务目标不完全对齐；2）过滤决策仅局部化，未考虑样本间二阶交互或未来模型表现；3）实验仅在 Datacomp 数据集上验证，尚未在更大规模或不同领域数据上检验；4）评分计算虽然相对轻量，但仍消耗额外算力，若不并行执行可能降低整体效率。

---

## 787. Causal Discovery in the Era of Agents

**arXiv ID:** 2606.23608 | [PDF](https://arxiv.org/pdf/2606.23608v1)

**作者:** Yujia Zheng `[一作]` (Carnegie Mellon University), Kun Zhang `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了基于代理的因果发现平台 causal-learn+，实现了数据分析、预处理、方法推荐、专家知识整合、工具协同与结果解释等功能，并强调代理不能直接提供因果证据；

**💡 创新点**

提出代理协助而非替代因果发现核心算法的设计原则，明确分离代理输出与因果证据，构建可追溯、可逆、可视化的工作流，并通过在线平台让非专家也能进行因果发现；

**🔧 技术方法**

结合大型语言模型（LLM）作为代理，使用检索增强生成技术；集成 causal-learn 生态中的多种因果发现算法（PC、FCI、GES、ANM、GIN、RLCD 等）；实现工具协同与可视化，部署在 Web 在线平台上；

**📊 数据集**

以 Big Five 人格测评问卷数据为例，约 20,000 名受访者的 50 个观测指标（10 个指标对应 5 个维度）经过标准化后用于因果图探索；

**📈 对比分析**

文章并未针对算法性能进行数值比较，而是通过示例展示因果图与心理学先验一致、获得专家验证，强调平台在易用性和解释性方面的优势；

**⚠️ 局限性**

局限性包括：代理仅提供辅助，无法产生因果证据；结果仍需用户确认，缺乏系统化评估指标和基准测试；对 LLM 的幻觉与错误仍可能影响推理；仅在单一示例中验证，尚未广泛推广与多领域实验。

---

## 788. MORL-A2C: Multi-Objective Reinforcement Learning Reranker for Optimizing Healthiness in MOPI-HFRS

**arXiv ID:** 2606.23603 | [PDF](https://arxiv.org/pdf/2606.23603v1)

**作者:** Aarya Vasantlal `[一作]` (University of Connecticut), Joshua Zolla `[通讯]` (University of Connecticut)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在多目标健康友好食品推荐系统（MOPI‑HFRS）的基础上，提出了多目标强化学习重排序器 MORL‑A2C，以序列化决策方式在用户的 top‑K 推荐中平衡偏好与健康目标。

**💡 创新点**

创新点包括：① 将推荐任务建模为 K 步 MDP，使系统能够捕捉到项目间的交互和累积效应；② 采用 Advantage Actor‑Critic（A2C）结合 scalarized relevance/health 奖励实现序列策略学习；③ 通过行为克隆预训练快速收敛，避免在大规模食物集合上随机探索导致的崩溃；④ 发现并修复原始 MOPI‑HFRS 评估脚本的 bug，保证实验公平。

**🔧 技术方法**

技术手段主要有：图神经网络（GNN）编码器（冻结后提供用户与食物嵌入）；Actor‑Critic 网络（两层状态 MLP + 单层候选 MLP）；行为克隆（BC）预训练；奖励 scalarization 采用固定权重 β；A2C 训练与 entropy 正则化。

**📊 数据集**

数据集：MOPI‑HFRS 所用的国家健康与营养检查调查（NHANES）用户健康标签与食物营养属性数据库，包含宏量营养与全营养两套基准。

**📈 对比分析**

与原 MOPI‑HFRS（bug‑fixed）对比：在宏量营养基准上 Recall@20 从 25.64% 降至 23.61%（≈‑2%），NDCG@20 降至 20.64%（≈‑3%），但 H‑Score@20 大幅提升至 69.57%（≈+23%）。在全营养基准上，Recall@20、NDCG@20 同样略降，H‑Score@20 升至 90.48%（≈+25%）。

**⚠️ 局限性**

局限性包括：① 未加入多样性奖励，导致推荐集中于少量食物；② 评估完全基于离线模拟，缺少真实用户交互反馈；③ scalarized 奖励使用固定 β，无法适应不同用户或位置的偏好差异；④ 仅验证了健康-偏好平衡，未对多目标分布进行完整 Pareto 前沿探索。

---

## 789. LightSTAR: Efficient Visual Document Retrieval via Lightweight Selection with Vision-Adaptive Refinement

**arXiv ID:** 2606.23539 | [PDF](https://arxiv.org/pdf/2606.23539v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 790. The Topology of Ill-Posed Questions: Persistent Homology for Detection and Steering in LLMs

**arXiv ID:** 2606.23590 | [PDF](https://arxiv.org/pdf/2606.23590v1)

**作者:** Guangyu Jiang `[一作]` (George Washington University), Tian Lan `[通讯]` (George Washington University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于持久同调的LLM内部状态拓扑分析与控制框架，用于检测和引导对模糊、未确定或冲突的提问做出恰当反应；

**💡 创新点**

创新点在于将每层提示词隐藏状态视为点云，通过零维持久同调得到三维紧凑描述（均值、熵、最大寿命浓度），并将该拓扑向量作为检索键，构造局部激活对比方向实现针对性答案拒绝或澄清；

**🔧 技术方法**

主要技术包括：层级点云投影、零维持久同调（H₀）分析、三维统计量提取、拓扑相似性检索、局部激活加权干预；

**📊 数据集**

使用AmbigQA、SituatedQA（二分类）和CLAMBER（9类细粒度）三大QA基准，模型选取Gemma‑7B‑it、Llama‑3.1‑8B‑Instruct和Mistral‑7B‑Instruct‑v0.3；

**📈 对比分析**

与传统基于提示、少样本、信息增益等方法以及AEN稀疏神经元基线对比，H₀拓扑特征在AmbigQA上从67.4%提升至78.9%，SituatedQA从79.9%提升至88.5%，CLAMBER 9‑way从57.6%提升至69.6%；在响应生成方面，拓扑局部干预将“有根可接受”率提升约9–20个百分点；

**⚠️ 局限性**

局限性包括：仅评估三种开源LLM和三套数据集；仅使用H₀持久同调，未探索H₁或路径持久同调；层级描述采用简单拼接，未学习权重；干预仅限于激活加权，未直接生成澄清问句等更复杂行为；

---

## 791. It's Much Easier for Neural Networks to learn Game of Life Dynamics with the Right Activation Function: Polynomial Kolmogorov-Arnold Networks

**arXiv ID:** 2606.23587 | [PDF](https://arxiv.org/pdf/2606.23587v1)

**作者:** Tashin Ahmed `[一作]` (Independent Researcher), Q. Tyrell Davis `[通讯]` (Cross Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究如何用极小神经网络学习康威生命游戏的单步动力学，比较不同激活函数的效果，并提出可学习多项式激活的PolyKAN网络；

**💡 创新点**

首次证明在仅包含 34 个可学习参数的极小网络中，PolyKAN 可完全学习 Life 规则，展示激活函数的选择比网络规模更关键；

**🔧 技术方法**

基于 Kolmogorov–Arnold 表示定理的 KAN 框架，引入可学习二阶多项式激活函数、PReLU、ReLU、Sigmoid 等；使用 Adam 优化、参数扰动实验、消融研究等技术；

**📊 数据集**

利用 32×32 的随机初始格子以及手工选取的 glider、oscillator、still‑life 模式生成的数据，训练单步预测任务；

**📈 对比分析**

通过 128 次训练跑与 10 种激活函数的对比，PolyKAN 在 34/29/25 个可学习参数下成功率 100%（相较 ReLU 为 0%），并在不同初始密度、参数扰动等场景下展示更稳健的性能；

**⚠️ 局限性**

仅针对单步预测且数据规模有限；对更长步长或更大尺寸网格的泛化尚未验证；激活函数对其他细胞自动机的适用性仍需进一步研究。

---

## 792. Dense Reward for Multi-View 3D Reasoning with Global Maps and Local Views

**arXiv ID:** 2606.23557 | [PDF](https://arxiv.org/pdf/2606.23557v1)

**作者:** Jiho Choi `[一作]` (KAIST), Hyunjung Shim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种基于密集奖励的多视角3D视觉问答框架DR‑MV3D。

**💡 创新点**

引入全局一致性奖励与局部视角规划奖励，并利用冻结的3D视觉基础模型提供伪结构目标，实现过程级密集监督。

**🔧 技术方法**

采用分阶段地图构建、视角轨迹规划、视角对齐以及GRPO强化学习，结合VGGT+SAM3作为结构先验。

**📊 数据集**

在MindCube、VSI‑Bench和BLINK(MV)三大多视角VQA基准上进行评测。

**📈 对比分析**

与多图像MLLM及3D感知基线相比，DR‑MV3D在MindCube上提升至66.5%准确率，VSI‑Bench平均分37.1，BLINK(MV) 56.4，均显著优于同类模型。

**⚠️ 局限性**

仍受限于对伪结构的依赖、训练成本高、在更复杂环境下视角选择可能不足。

---

## 793. Simulation-Free Estimation of Traffic Flows from Sparse Count Data

**arXiv ID:** 2606.23536 | [PDF](https://arxiv.org/pdf/2606.23536v1)

**作者:** Davide Guastella `[一作]`, Gianluca Bontempi `[通讯]` (Université Libre de Bruxelles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种两级无仿真方法，先在区域层使用加权最小二乘求解车辆流量分配，再在路段层根据传感器覆盖度和时序相似度对候选路径评分，将区域路径映射为具体的路段轨迹，从而仅凭稀疏聚合计数恢复时间变化的交通流。

**💡 创新点**

创新点在于将校准问题分解为区域级加权拟合与路段级基于覆盖度与时序相似度的评分，既避免了传统仿真循环，又保证了区域级一致性，并实现了可微分的梯度优化过程。

**🔧 技术方法**

使用了加权最小二乘与 Adam 梯度优化（含梯度裁剪与 softplus 非负约束）进行区域级求解，路段级则利用贡献矩阵、最短路径搜索与 softmax 评分对候选路径进行选择，整个流程可微分且无需仿真。

**📊 数据集**

在布鲁塞尔道路网络（OpenStreetMap 转换为 SUMO）上使用 369 个传感器的 2024‑03‑01 24h 车辆计数数据，并在合成场景中生成 80,000 车辆的仿真数据进行验证。

**📈 对比分析**

与先前的仿真校准方法和 SUMO RouteSampler 进行比较，MAE 44.16、RMSE 51.80、NRMSE 0.004，区域级求解仅 23 秒、路段级 0.2 秒，明显快于仿真方法的 69 分钟和 RouteSampler 的 3 小时，且精度相当或更优。

**⚠️ 局限性**

局限性包括估计问题欠定导致未观测路段可能出现多种等价流配置、路段级使用最短路径忽略拥堵路由、优化迭代次数过多会导致对传感器数据过拟合并削弱未测路段的流量，以及缺乏探针或先验 OD 约束限制了解的唯一性。

---

## 794. Vera: A Layered Diffusion Model for Content-Preserving Video Editing

**arXiv ID:** 2606.23610 | [PDF](https://arxiv.org/pdf/2606.23610v1)

**作者:** Hongkai Zheng `[一作]` (California Institute of Technology), Zhuoning Yuan `[通讯]` (Netflix, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `da1b1a89-583a-4b57-9c81-478778569bec` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了一种分层扩散框架 Vera，能够在视频编辑时仅生成编辑层及其 alpha 贴图，进而实现对原视频内容的完整保留。

**💡 创新点**

创新点包括：①将视频编辑拆解为三输出（编辑层、alpha 贴图、合成视频）并采用 Mixture-of-Transformers (MoT) 交叉自注意机制；②通过专门的 alpha 与 composite 分支实现跨层一致性；③在三输出框架下实现高质量的层级编辑。

**🔧 技术方法**

技术上使用了潜在扩散模型、DiT 变体、MoT 结构、联合自注意、以及多任务训练策略；同时在输入中加入视频与可选掩码的 patch 嵌入。

**📊 数据集**

构建了约 486K 帧、832×480 分辨率的高质量分层视频数据集，涵盖合成背景替换、单物体真实视频以及多物体与视觉特效场景。

**📈 对比分析**

在背景替换和物体添加任务上，Vera 在内容保真度（PSNR 提升 3–7 dB、SSIM 与 LPIPS 大幅改善）和 alpha 贴图质量上超过七大公开基线；在 VLM‑评估的 Composition Spatial/Temporal Quality 与 Instruction Satisfaction 上保持与 VACE 接近，且用户研究显示其在内容保真度上优先。

**⚠️ 局限性**

主要局限包括：推理速度约为 VACE 的 3 倍；实验仅涵盖背景替换与物体添加，未覆盖光照/特效等更复杂编辑；且假设保留区域半透明度极小，无法直接恢复玻璃、水等大半透明内容。

---

## 795. Quantifying the Agreement Between Data-Influence and Data-Similarity to Understand LLM Behavior

**arXiv ID:** 2606.23591 | [PDF](https://arxiv.org/pdf/2606.23591v1)

**作者:** Christopher J. Anders `[一作]` (RIKEN Center for Advanced Intelligence Project), Mohammad Emtiyaz Khan `[通讯]` (RIKEN Center for Advanced Intelligence Project)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统比较了大型语言模型输出追踪中的数据相似性（BM25）与数据影响力（基于EK‑FAC等 Hessian/梯度估计）对训练文档排序的一致性与差异；

**💡 创新点**

创新点在于首次量化两种追踪方法的相似性与不对称性，并提出通过先使用低成本相似性筛选后再用高成本影响力细化的混合策略，实现成本与准确性的折衷；

**🔧 技术方法**

采用 BM25 作为相似性度量，使用 Hessian‑based EK‑FAC、KFAC、DIAG 与梯度估计等多种数据影响力方法，并运用排名重叠、Rank‑Biased Overlap、ROC 等指标进行评估；

**📊 数据集**

实验使用 100 万条训练文档中抽取的 100k 条子集以及 100 条多主题提示，评测六个模型（OLMo2‑1B、Qwen3‑1.7B、LlaMa3.2‑1B、Gemma3‑1B、GPT2‑medium 与 GPT2‑small）；

**📈 对比分析**

比较结果显示，两种方法在前几百名文档上有显著一致性，影响力更能预测相似性排名，混合方法在 ROC AUC 上取得 0.83 的优势，证明低成本相似性加高成本影响力的组合能提升追踪效果；

**⚠️ 局限性**

主要局限在于受计算资源限制，提示和文档样本不足且未按主题平衡，且实验仅覆盖至 1.7B 参数规模，未来需扩展到更大模型与更多样本。

---

## 796. Decentralized Autonomous Traffic Management through Corridor Networks

**arXiv ID:** 2606.23585 | [PDF](https://arxiv.org/pdf/2606.23585v1)

**作者:** Jasmine Jerry Aloor `[一作]` (Massachusetts Institute of Technology), Hamsa Balakrishnan `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在多航线网络中，利用分布式多智能体强化学习（MARL）训练出可在单一通道内自适应的旋转不变策略，并在不重新训练的情况下，将该策略零射击应用于更复杂的合并、分裂及多路网络中，以实现无人机的分布式交通流管理。

**💡 创新点**

创新点在于：1）引入旋转不变的观察与策略表示，使得单通道训练的策略可迁移至任意方向的通道；2）采用基于课程学习的奖励设计，逐步强化通道遵从与分离保持；3）展示了在无集中调度、无先验全局路径信息下，零射击迁移至大规模多通道网络的能力。

**🔧 技术方法**

使用技术主要包括：基于POMDP的分布式强化学习框架、前馈多层感知网络（Graph Neural Network）用于邻居交互建模、奖励函数的分阶段课程学习、以及Python/Unity/自研仿真环境。

**📊 数据集**

数据集：纯仿真生成的通道网络与飞机轨迹数据，覆盖单通道、双合并、分合并、以及包含18条通道的综合网络，测试样例数为50/10不等，涉及10-40架飞机的不同密度与速度异质性。

**📈 对比分析**

与理论最大吞吐量（所有飞机以最大速度冲突自由通过）及先前单通道实验进行对比。结果显示：通道合规率>96%，完成率>97%，平均速度接近最大速度，战术干预率<5%，零射击迁移后综合网络的吞吐量与理论上限相差不到7%。

**⚠️ 局限性**

局限性包括：1）缺乏正式的运行时安全过滤器，安全保证仅靠奖励惩罚；2）未考虑失效、通信丢包、风力等离散环境；3）仅在二维平面固定通道上验证，缺乏三维动态路由与真实动力学；4）对高密度情形下的局部拥堵仍需进一步缓解。

---

## 797. An Open-Source LFSR-Based Stochastic Leaky Integrate-and-Fire Neuron in SkyWater 130 nm: Design, Stochastic Characterisation, and Rate Coding

**arXiv ID:** 2606.23532 | [PDF](https://arxiv.org/pdf/2606.23532v1)

**作者:** Poornima Kumaresan `[一作]`, Santhosh Sivasubramani `[通讯]` (Indian Institute Of Technology Delhi)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c`

**🎯 论文内容**

实现了一个可配置的、基于16位LFSR的随机性泄漏积分放电神经元，并将其映射到SkyWater 130 nm标准单元上，提供了可通过16寄存器串行接口编程的完整功能。

**💡 创新点**

创新点包括：① LFSR反馈多项式在运行时可配置；② 8项激活表实现概率化发火；③ 对随机源与比较器的自相关性进行量化并提供子采样消除方法；④ 公开完整RTL、测试与布局，推动开源神经元生态。

**🔧 技术方法**

使用的技术包括：SkyWater 130 nm开放工艺、OpenLane/OpenROAD实现流程、Tiny Tapeout 1×1 芯片、LFSR与二进制比较器的伪随机生成、Leaky‑Integrate‑and‑Fire（LIF）积分器、标准单元设计与时序分析。

**📊 数据集**

本研究不依赖传统数据集，而是通过仿真验证与统计分析对随机序列、发火概率、速率编码和抑制周期等性能进行评估。

**📈 对比分析**

与已有数字神经元（TrueNorth、最近的LFSR神经元等）相比，本文在 50 MHz 时钟下实现了约10.6 k µm²面积、约701 µW功耗、70 %布局利用率，并通过18个定向Cocotb测试全部通过；但为预硅评估，实际功耗与时序需实测确认。

**⚠️ 局限性**

局限性包括：仅提供单个神经元的预硅设计，缺乏多神经元阵列的评估；随机源的序列相关性需要子采样补偿；默认多项式非最大周期，需手动编程；功耗与时序尚未在硅片上测量。

---

## 798. Composition: Building Community with Arts, Math, and Code (Experience Report)

**arXiv ID:** 2606.23526 | [PDF](https://arxiv.org/pdf/2606.23526v1)

**作者:** Isidore Mohr `[一作]`, Claire Wang `[通讯]` (University of Pennsylvania)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

组织了双年一度的免费“Composition”艺术、数学与代码会议，系统记录了从征稿、评审、场地选择、推广到社区反馈的全过程。

**💡 创新点**

创新点在于将功能艺术、编程语言研讨会和算法音乐体验相结合，打造本地可达、免费且多元艺术形式的社区平台，首次在东北地区提供此类跨学科聚会。

**🔧 技术方法**

使用的技术包括Google表单收集提案、Discord/社交媒体推广、现场录音录像、YouTube发布、以及现场麦克风与音视频设备支持演示。

**📊 数据集**

未使用传统数据集，仅以收集到的提案表单、现场问答录音和参加人数等非结构化数据进行分析。

**📈 对比分析**

与FARM、NJPLS、Algoraves等类似活动对比，Composition吸引了约35-40人参与，报名与实际到场人数基本吻合，活动流程顺畅、录制质量可接受；未给出量化性能指标。

**⚠️ 局限性**

局限性包括提交周期过短导致评审时间不足、缺乏结构化互动模式导致观众参与度不均，以及缺乏系统化评估指标与可持续运营方案。

---

## 799. Scaling State-Space Models from Lines to Paragraphs: An Ablation of Mamba-based OCR

**arXiv ID:** 2606.23524 | [PDF](https://arxiv.org/pdf/2606.23524v1)

**作者:** Merveilles Agbeti-Messan `[一作]` (University of Rouen Normandy), Thierry Paquet `[通讯]` (University of Rouen Normandy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了自回归 Mamba（SSM）在 OCR 任务中从行级扩展到段落级的可行性，系统探索了模型超参数、在合成 Wikipedia 段落上的规模实验，并在印刷与手写真实文档上进行验证。

**💡 创新点**

提出并验证了段落级 Mamba-OCR 的最佳配置（L=4, N=256, E=6, MC=1），展示了其线性时间解码相较 Transformer 的显著速度优势，并揭示了对训练数据量高度敏感的“数据饥饿”特性。

**🔧 技术方法**

采用自回归 Mamba 解码器、双向 Mamba 多模态连接器、CNN 视觉编码器，以及合成 Wikipedia 段落、IAM 手写数据库与 Transformer 基线 DAN 的对比实验。

**📊 数据集**

使用合成 Wikipedia 段落（100–1000 字符）、BnL 印刷行与段落、以及 IAM 手写行与段落数据集。

**📈 对比分析**

在相同训练协议下与 Transformer 基线 DAN 进行对比：在合成段落上两者 CER 均 <1%，但 Mamba 在 10 行时速度提升 4.5×；在印刷段落上速度快 2×但 CER 略高；在手写行/段落上 Mamba 表现显著落后（行 CER 8.2% vs 4.2%，段落 CER 10% vs 3.5%），差距部分归因于数据稀缺。

**⚠️ 局限性**

主要局限在于 Mamba-AR 对长序列的“数据饥饿”，在手写段落或小样本场景下性能不足；当前实现受缓存/并行扫描限制导致状态维度最大为 256，需更多数据或混合注意力/预训练以提升效果。

---

## 800. Polycepta: Object-Centric Appearance Estimation for Multi-Object Tracking

**arXiv ID:** 2606.23604 | [PDF](https://arxiv.org/pdf/2606.23604v1)

**作者:** Mohamed Nagy `[一作]` (Khalifa University), Majid Khonji `[通讯]` (Khalifa University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出Polycepta框架，用递归状态估计方式重构多目标跟踪中的外观模型，并将其作为轻量化外观特征融入现有的运动驱动跟踪器。

**💡 创新点**

创新点在于将外观建模转化为递归状态估计，采用频域视觉关联与自适应门控更新、状态抹除学习策略，实现外观估计随时间提升并能跨类别泛化。

**🔧 技术方法**

使用状态空间模型（HiPPO+可学习对角矩阵）、FFT频域相关、门控更新、对比与正交损失训练，外观提取采用MobileNetV3_small，并与RobMOT、FastTracker等框架集成。

**📊 数据集**

在KITTI、Waymo Open Dataset、MOT17、MOT20等公开数据集上进行训练与评估，亦在不同检测器（VirConv、PV-RCNN、CasA等）下测试。

**📈 对比分析**

与传统ReID、FastTracker、RobMOT等基线对比，Polycepta在KITTI上实现MOTA 92.27%，在MOT17上提升HOTA和MOTA 2-3%，显著降低ID切换（如FastTracker从567降至382），证明外观递归估计显著提升关联质量。

**⚠️ 局限性**

局限性包括对检测器质量仍有依赖，状态抹除虽提升泛化但可能降低对特定类别的辨识边缘；对极端遮挡或快速运动场景的鲁棒性仍待进一步验证。

---

## 801. Against Proxy Optimization

**arXiv ID:** 2606.23597 | [PDF](https://arxiv.org/pdf/2606.23597v1)

**作者:** Sven Neth `[一作]` `[通讯]` (University of Pittsburgh), Sven Neth (University of Pittsburgh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文通过形式化决策理论模型，分析了在何种条件下最大化代理效用会导致真正效用下降，并对现有Compactness条件进行批判，提出了Minimal Balance和Actual Balance等新原则，探讨了代理失败的理论机制，并提出了动态代理更新和早停等缓解策略。

**💡 创新点**

创新点在于提出了Minimal Balance和Actual Balance两种更弱/更强的代理失败判定条件，证明它们比Compactness更符合实际，同时构建了完整的代理失败理论框架，并讨论了基于代理更新和早停的实用缓解方案。

**🔧 技术方法**

采用了形式化的决策理论与优化建模、严格的数学证明与定理推导，构建了代理效用与真实效用之间的关系，并在此基础上分析了不同假设下的代理失败现象。

**📊 数据集**

本研究为理论性工作，无实验数据或数据集。

**📈 对比分析**

不适用：论文仅给出理论证明与定理，没有进行实验比较或性能评估。

**⚠️ 局限性**

局限性包括：1) 对成本函数严格单调和特征可度量性等假设较强；2) 主要聚焦长周期行为，缺乏短周期分析；3) 在实际应用中识别代理失败困难；4) 未充分考虑不确定性、战略互动及其他现实约束。

---

## 802. A Watermark for Vision-Language-Action and World Action Models

**arXiv ID:** 2606.23574 | [PDF](https://arxiv.org/pdf/2606.23574v1)

**作者:** Yule Liu `[一作]` (Hong Kong University of Science and Technology), Xinlei He `[通讯]` (Wuhan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种针对生成式机器人策略的水印方法，即在采样噪声中嵌入密钥相关的扰动，并通过黑盒审计仅利用执行动作的方式验证模型所有权。

**💡 创新点**

创新点包括：①将水印隐藏在采样种子而非输出动作中，保持高斯噪声分布；②设计了基于 MAP 恢复的稀疏判定器，可在部分观察下识别密钥；③支持多密钥（多租户）识别和对抗输出侧攻击；④通过实验验证其对 LoRA、剪枝、量化等所有权变体的鲁棒性。

**🔧 技术方法**

技术手段包括：密钥化种子选择与扰动、基于最大后验（MAP）的种子恢复、匹配滤波打分、同步搜索、以及对齐后的阈值校准；水印生成使用伪随机生成器，恢复时利用梯度优化最小化观测误差与高斯先验。

**📊 数据集**

使用的实验数据集为两类策略：Vision‑Language‑Action（π_0.5）和 World‑Action（LingBot‑VA），在两套机器人平台上测试：LIBERO‑10（单臂）和 RoboTwin（双臂）。

**📈 对比分析**

与传统输出侧频带水印相比，本文方法在 16 次 rollout 组内实现 1% FPR 下 100% TPR，任务成功率几乎不受影响；对输出侧剪裁、平滑、抖动、延迟等攻击仅在强度极高时出现检测下降；对 LoRA、剪枝、量化等所有权变体仍保持 100% 检测率。

**⚠️ 局限性**

局限性：需在授权审计场景下获取执行动作，无法在完全黑盒或无访问权限情况下验证；在极强的平滑/抖动攻击或严重延迟下检测效果下降；安全分析仅基于经验阈值，未提供正式的密码学安全证明；高维密钥空间虽难以伪造，但对抗性训练或结构化攻击的理论风险尚未完全消除。

---

## 803. SVD-Surgeon: Optimal Singular-Value Surgery for Large Language Model Compression

**arXiv ID:** 2606.23568 | [PDF](https://arxiv.org/pdf/2606.23568v1)

**作者:** Mahmoud Safari `[一作]` (University of Freiburg), Frank Hutter `[通讯]` (University of Freiburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SVD-Surgeon，一种无训练、基于Optimal Brain Surgeon的低秩压缩修正方法。

**💡 创新点**

将OBS框架迁移到奇异值空间，给出闭式更新和重要性评估，兼容任意SVD压缩。

**🔧 技术方法**

利用奇异值分解、Fisher信息估计二阶曲率、闭式更新和Saliency分数等技术。

**📊 数据集**

在WikiText-2和C4数据集上对OPT系列与LLaMA-2-7B模型进行评估。

**📈 对比分析**

与SVD-LLM基线相比，SVD-Surgeon在高压缩率下显著降低困惑度，提升压缩比与性能。

**⚠️ 局限性**

仅更新奇异值而固定方向，Fisher估计成本较高，需在更多模型和任务上验证通用性。

---

## 804. LangMAP: A Language-Adaptive Approach to Tokenization

**arXiv ID:** 2606.23566 | [PDF](https://arxiv.org/pdf/2606.23566v1)

**作者:** Clara Meister `[一作]` (EPFL), Tiago Pimentel `[通讯]` (ETH Zürich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种语言自适应最大后验（LangMAP）分词方案，利用共享词表生成语言特定分词，既可用于从零训练多语言模型，也可在已有预训练模型上做无词表修改的适配。

**💡 创新点**

创新点在于在 UnigramLM 框架下为每种语言学习单独的词频分布，保持词表不变，在推理时进行语言无关的最佳后验分割，从而实现语言特定分词而不需重新训练模型或改词表。

**🔧 技术方法**

使用 UnigramLM 生成模型与 EM 参数估计、Viterbi 解码；对比 Tokenizer‑Eval、MorphScore、AST 对齐等评测指标；对语言模型进行细调以评估下游效果。

**📊 数据集**

参数估计使用 FineWeb2（自然语言）和 The Stack（代码）；内在评估使用 FLORES‑200、UD treebanks；代码评估使用 StarCoder 与 Tree‑sitter 解析；细调和下游评测使用 FineWeb2、MultiBLiMP、Global‑PIQA、Belebele。

**📈 对比分析**

与原始词表基线对比，LangMAP 在自然语言中提升 MorphScore（尤其拉丁文字的 recall/precision），在代码中提升 AST‑leaf 对齐，并在细调实验中提高 MultiBLiMP 的语法可接受性；在 Global‑PIQA、Belebele 上未见显著提升。

**⚠️ 局限性**

局限性包括受限于固定词表，缺少语言特定子词时无法改善分割；需要标注语言标签；下游评测范围有限，仅在少数模型/任务上验证；无法处理无标签的完全无监督情形。

---

## 805. Digital Twins Need Feedback

**arXiv ID:** 2606.23562 | [PDF](https://arxiv.org/pdf/2606.23562v1)

**作者:** Guo-Qiang Zhang `[一作]` `[通讯]` (University of Texas Health Science Center), Guo-Qiang Zhang (University of Texas Health Science Center)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出以双向反馈为核心的数字孪生定义，并在健康领域构建多层次反馈循环示例（如癫痫企业孪生EpiToMe与数据生成孪生Tissue-to-Bytes）。

**💡 创新点**

将数字孪生从单纯的模拟或可视化提升为可治理、可验证的闭环系统，强调反馈契约、层级组合与动作验证等关键研究议题。

**🔧 技术方法**

采用知识图谱、可解释机器学习、FHIR等标准化接口、事件驱动状态机、基于本体的语义建模以及分布式数据管道等技术。

**📊 数据集**

使用癫痫病例记录、住院流程日志以及BRAIN Initiative Cell Atlas Network提供的单细胞测序原始数据（Tissue-to-Bytes）。

**📈 对比分析**

在真实医院环境中部署EpiToMe后，病例会议文档时间下降74.4%，手术评估完成率近四倍提升，证明闭环数字孪生在临床工作流程上的显著效率提升。

**⚠️ 局限性**

缺乏统一的反馈契约规范和跨层级的安全治理机制；动作效果验证仍依赖人工评估，难以量化；系统高度依赖数据质量与标准化，易受数据缺陷影响。

---

## 806. The Energy Consumption of Transformer Fine-Tuning: A Roofline-Inspired Scaling Model

**arXiv ID:** 2606.23546 | [PDF](https://arxiv.org/pdf/2606.23546v1)

**作者:** Mansour Zoubeirou a Mayaki `[一作]` `[通讯]` (Universite Lumiere Lyon 2), Mansour Zoubeirou a Mayaki (Universite Lumiere Lyon 2)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个基于Transformer训练能耗的可解释预测框架，结合计算、内存流量与硬件效率的分解，能够在不同模型规模与并行策略下估算训练能耗。

**💡 创新点**

创新点在于将屋顶线模型与基于加速比的硬件效率因子相结合，形成一个包含计算、内存和并行效率三项的缩放定律，显著提升能耗预测精度。

**🔧 技术方法**

技术上采用了计算与内存流量的闭式代理、速度加速测量、线性回归与残差自举、以及基于NVIDIA RTX 2080 Ti节点的多GPU实测数据。

**📊 数据集**

实验数据集主要为BERT在标准NLP任务（句子分类、抽取式问答）上的微调，通过在不同深度、宽度、序列长度和批量大小下进行架构扫掠获得。

**📈 对比分析**

与单GPU基准及两种分布式并行策略（张量并行、全分片数据并行）进行对比，预测模型在410个独立配置上达成R²≈0.85，能耗误差低于20%，并验证了FSDP在能耗上优于TP。

**⚠️ 局限性**

局限性包括仅评估BERT编码器微调、未覆盖解码器预训练、Mixture‑of‑Experts或流水线并行；使用粗粒度计算/内存代理，未细化通信细节；假设固定硬件功耗，未考虑能耗外部因素如碳强度和数据中心管理。

---

## 807. Direct and Indirect Influence on Likes in Social Media

**arXiv ID:** 2606.23530 | [PDF](https://arxiv.org/pdf/2606.23530v1)

**作者:** Ivan Kozitsin `[一作]` (V. A. Trapeznikov Institute of Control Sciences of Russian Academy of Sciences), Anton V. Proskurnikov `[通讯]` (Politecnico di Torino)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了 VKontakte 社交网络中，直接与间接社会影响（即亲友与亲友之友）对用户点赞行为的作用。

**💡 创新点**

创新点在于提出并验证了扩展诱导指数 m*，证明二阶邻居活动对点赞概率的影响比原始诱导指数和普通计数更强，提供了大型在线社交媒体中间接影响的实证支持。

**🔧 技术方法**

采用网络度量、结构多样性、诱导指数等特征，构建多元逻辑回归模型，并通过稳健性检验和多种指标对比分析其效果。

**📊 数据集**

使用 VKontakte 公开 API 提取的约 290,000 名用户、24,950,496 条双向好友边、308 条本地新闻贴子、约 19,600 条点赞的真实数据集，重点聚焦图森地区。

**📈 对比分析**

模型通过 BIC 与 McFadden 伪 R² 进行比较，扩展诱导指数 m* 的模型在 BIC（约 354,723）和伪 R²（约 0.0384）上均优于其他模型，表明其解释力最佳。

**⚠️ 局限性**

局限包括：无法完全排除同质性与共同外部曝光偏差；未观测到 VK 具体推荐算法与用户新闻流；样本仅限受欢迎贴文和图森地区，可能存在选择偏差；大量点赞来自未纳入主连通分量的用户，可能影响结论。

---

## 808. Collapsed Effective Operators for Higher-order Structures

**arXiv ID:** 2606.23517 | [PDF](https://arxiv.org/pdf/2606.23517v1)

**作者:** Maximilian Krahn `[一作]` (Imperial College London), Tolga Birdal `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于Schur补的Collapsed Effective Operators，将高阶结构压缩到顶点级别，生成一阶拉普拉斯形式的全局谱算子。

**💡 创新点**

创新点在于：①把多阶边界信息一次性折叠为单一顶点算子，避免传统逐阶合并；②保证正半定且能量下降；③利用此算子提升谱聚类、去噪、蛋白质结构分割和节点分类等任务。

**🔧 技术方法**

核心技术：CW/组合细胞的边界矩阵、梯度拉普拉斯构造、Schur补归约、正则化求逆、谱分析与Cheeger不等式、实验验证中的谱特征提取、位置编码（SchurPE）。

**📊 数据集**

主要使用的数据集包括：随机几何图（用于去噪）、Topotein蛋白质数据集（用于结构分割）、SBM与现实社交网络（Karate、Football、Misérables、Books、Dolphins）用于聚类、Mantra复杂网络数据集用于谱分类、以及蛋白质原子网络用于节点分类。

**📈 对比分析**

与传统图拉普拉斯、Multi‑order Laplacian、HOMP等方法比较，Collapsed Operator 在谱聚类精度提升约3–6%，蛋白质分割准确率从46.9%提升到70.9%，去噪误差保持在0.07左右，节点分类中的位置编码提升约10–15%（GCN/GraphTransformer）。

**⚠️ 局限性**

局限性包括：①算子稠密导致显式构造成本高；②在某些任务中，逐阶保留的拓扑不变量（如Betti数）会部分丢失；③谱间隙可能因Schur压缩而收缩，Cheeger下界不再成立；④需手动设定权重 β_k,γ_k，缺乏自动化学习机制。

---

## 809. Lift4D: Harmonizing Single-View 3D Estimation for 4D Reconstruction In-the-Wild

**arXiv ID:** 2606.23688 | [PDF](https://arxiv.org/pdf/2606.23688v1)

**作者:** Yehonathan Litman `[一作]` (Carnegie Mellon University), Shubham Tulsiani `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在单目视频中实现完整4D动态物体重建。

**💡 创新点**

将单视图3D重建与时序一致性、遮挡感知和视图条件扩散先验融合，在测试时优化得到完整4D表示。

**🔧 技术方法**

使用Causal Latent Conditioning的单视图图像-3D模型、3D高斯溅射、遮挡感知渲染监督、视图条件图像扩散先验以及稀疏变形控制节点。

**📊 数据集**

Consistent4D合成数据、Pexels公开单目视频和DAVIS。

**📈 对比分析**

与STAG4D、L4GM、DM4D、PAD3R、V2M4等基线比较，Lift4D在LPIPS、FVD、CLIP、EPE等指标上均优于或接近最佳。

**⚠️ 局限性**

依赖初始SAM3D预测，错误会传播；几何背骨架仍可改进，且对极端遮挡或大视角变换的鲁棒性有限。

---

## 810. LaST-HD: Learning Latent Physical Reasoning from Scalable Human Data for Robot Manipulation

**arXiv ID:** 2606.23685 | [PDF](https://arxiv.org/pdf/2606.23685v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 811. Randomized YaRN Improves Length Generalization for Long-Context Reasoning

**arXiv ID:** 2606.23687 | [PDF](https://arxiv.org/pdf/2606.23687v1)

**作者:** Manas Mehta `[一作]` (New York University), Greg Durrett `[通讯]` (New York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Randomized YaRN 训练方法，利用在短序列训练时随机采样更长位置编码并配合长度进度表，显著提升 LLM 在长上下文推理任务上的长度泛化能力。

**💡 创新点**

创新点在于将随机位置采样与 YaRN 编码结合，并引入长度进度学习（curriculum），通过在训练阶段暴露模型于超出训练分布的位置信息，使模型在远长 OOD 长度上实现更稳健的泛化。

**🔧 技术方法**

使用技术包括：YaRN 位置扩展、随机位置编码（RPE）、LoRA 微调、长度进度学习（curriculum）以及基于 RoPE 的 Transformer 架构。

**📊 数据集**

实验使用 BABILong（多跳推理）和 Multi‑Round Coreference Resolution (MRCR) 两个长上下文推理基准，训练样本为小规模 (<5K) 的短上下文数据。

**📈 对比分析**

对比了零射、标准 LoRA、训练时使用 YaRN、RPE 等基线；在 16K–128K 的 OOD 长度上，Randomized YaRN 在 Qwen2.5‑7B‑Instruct 和 Olmo3‑7B‑Instruct 上平均提升约 8–10% 的准确率，特别是在 128K 长度上取得显著优势。

**⚠️ 局限性**

局限性：仅适用于推理长度远大于训练长度的场景；实验仅在 7B 规模模型上验证；仅评估英文长上下文推理；需要较高计算资源和对模型的特殊微调。

---

## 812. LIBERO-Safety: A Comprehensive Benchmark for Physical and Semantic Safety in Vision-Language-Action Models

**arXiv ID:** 2606.23686 | [PDF](https://arxiv.org/pdf/2606.23686v1)

**作者:** Rongxu Cui `[一作]` (Tsinghua University), Hao Zhao `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LIBERO‑Safety 基准，使用 UBDDL 语言和关键姿势驱动的数据生成管线，生成 19.7K 条无碰撞演示，并对多种 VLA 与基础模型进行跨范式评估。

**💡 创新点**

构建可参数化的安全任务生成器、关键姿势驱动的动作合成流程、海量安全演示集，并系统揭示安全与泛化之间的张力与模型缺陷。

**🔧 技术方法**

采用 UBDDL 语义定义、关键姿势驱动的运动规划、CuRobo 轨迹生成、行为克隆与大规模预训练、视觉-语言-动作模型及语义安全推理与拒绝机制。

**📊 数据集**

19,664 条严格无碰撞演示，覆盖 40 任务、5 类安全套件，结合 953 个物体、462 对手-物体姿态；评测中还使用 LIBERO、VLA‑Arena、RoboTwin 等现有基准。

**📈 对比分析**

对 10 种代表性 VLA 与 2 种基础模型在统一超参和硬件环境下进行训练和评测，指标包括成功率、碰撞率、执行时长、拒绝率等；结果显示标准预训练不足以保证安全，π0.5 在多任务中表现最佳但仍存在显著方差，安全失误主要来自轨迹合成和语义不对齐。

**⚠️ 局限性**

基准仍基于仿真，未覆盖真实接触动力学、硬件延迟及复杂人机交互；关键姿势生成需人工参与，数据集缺乏危险负例；未对长期连续交互与多主体动态情境进行评估。

---

## 813. Lightweight Neural Framework for Robust 3D Volume and Surface Estimation from Multi-View Images

**arXiv ID:** 2606.23653 | [PDF](https://arxiv.org/pdf/2606.23653v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 814. Keep The Essentials: Efficient Reference Conditioned Generation via Token Dropping

**arXiv ID:** 2606.23682 | [PDF](https://arxiv.org/pdf/2606.23682v1)

**作者:** Rishubh Parihar `[一作]` (Indian Institute Of Science), Or Patashnik `[通讯]` (Tel Aviv University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在参考条件扩散模型中通过随机丢弃参考图像token并在推理时采用任务感知token选择策略，以实现稀疏参考表示，从而显著提升推理速度和内存效率。

**💡 创新点**

①证明参考token高度冗余；②通过在训练时随机token丢弃使模型对稀疏参考鲁棒；③推理时使用边缘或显著性引导的token采样；④可与KV缓存、Token Merging等技术结合，实现更高加速。

**🔧 技术方法**

使用DiT+VAE编码、LoRA轻量微调、随机token丢弃、Canny边缘图/显著性图引导采样、KV缓存、Token Merging；评估指标包括LPIPS、CLIP‑I、DINO、CLIP‑T‑I。

**📊 数据集**

HQ‑Edit（30K）用于指令驱动编辑；Subjects‑200K（20K）用于个性化；CustomDiffusion‑105（13K）用于多参考生成；评估使用PIE‑Bench、Subject‑200K子集、CustomDiffusion‑105。

**📈 对比分析**

与完整参考基线及无训练的随机丢弃基线对比；单参考下token保留率0.1即可实现约2×加速，质量保持；多参考下0.05–0.1 token保留率可实现4×以上加速；与Token Merging、KV缓存结合时可达6.7×加速，质量保持或略有提升。

**⚠️ 局限性**

对极低token比例仍会丢失细节；对不同任务的token选择仍需手工设计；需在特定预训练模型上微调；未验证在更大规模或不同域上的泛化；对VAE编码分辨率依赖较大。

---

## 815. Tapered Language Models

**arXiv ID:** 2606.23670 | [PDF](https://arxiv.org/pdf/2606.23670v1)

**作者:** Reza Bayat `[一作]` (Mila), Aaron Courville `[通讯]` (Mila)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Tapered Language Models（TLM）原则，即在保持总参数量不变的前提下，沿层数递减地分配MLP宽度，从而提高语言模型的困惑度和下游推理性能。

**💡 创新点**

核心创新是将参数分配的非均匀性系统化为一种“渐减”策略，并证明在不同架构（Transformer、Gated Attention、Hope-attention、Titans）及多种规模下均有效。

**🔧 技术方法**

技术实现包括：采用三种光滑衰减调度（线性、余弦、Sigmoid）对MLP中间维度进行单调递减；在训练中保持总参数量和 FLOPs 不变；使用余弦相似度测量层级新颖性以提供机制解释。

**📊 数据集**

主要使用公开的 WikiText-2、WikiText、LAMBADA 语料进行预训练与评估，并在八个常识推理基准（LAMBADA、PIQA、HellaSwag、WinoGrande、ARC-easy、ARC-challenge、SIQA、BoolQ）上进行下游性能评测。

**📈 对比分析**

与统一宽度基线在同等参数与 FLOPs 下进行对比，结果显示在所有四种架构和两种规模下，TLM 的 perplexity 均显著下降（最高约 1.8 点），下游常识推理平均准确率提升约 0.6–1.5%，表明该设计在不增加成本的前提下提供可观收益。

**⚠️ 局限性**

限制包括：仅在 440M Transformer 上完成宽度与调度搜索，其参数并未针对更大规模或其它架构进行精细调优；不同模型深度、隐藏维度或参数分配比例可能导致最优衰减曲线变动；未探索其他可变维度（如注意力头数、键值维度等）是否同样受益。

---

## 816. On the Limits of Prompt-Conditioned Language Models as General-Purpose Learners

**arXiv ID:** 2606.23668 | [PDF](https://arxiv.org/pdf/2606.23668v1)

**作者:** David Mguni `[一作]` (Queen Mary University London), Jun Wang `[通讯]` (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

本文从信息论与博弈论视角，正式分析了大型语言模型（LLM）通过自然语言提示实现通用问题求解的根本限制，并给出了可证明的误差底限与 PAC‑Bayes 泛化界。

**💡 创新点**

创新点在于：① 将提示与模型内部推理拆解为 Prompt Interpreter 与 Base Solver 两层，揭示提示通道的容量与安全/对齐约束是导致不可克服误差的两大根源；② 在 cheap‑talk 设定下证明了表达式与目标不匹配各自产生的不可消除误差底限；③ 结合 PAC‑Bayes 理论得到对经验风险与总体风险的分离估计，展示即使数据量无限，误差仍可被正则化与通道容量限制所锁定。

**🔧 技术方法**

使用技术包括：信息论（互信息、数据处理不等式、Fano 近似）、PAC‑Bayes 统计学习理论、贝叶斯博弈论（cheap‑talk）以及理论化的两层模型分解。

**📊 数据集**

本文主要为理论分析，没有采用具体的公开数据集；所有结论基于假设的任务族与提示分布，可视为抽象化的模拟场景。

**📈 对比分析**

比较方法：与传统的基于监督学习或自监督预训练的 LLM 无直接对照实验；作者通过理论证明展示误差底限与信息容量/对齐约束的关系，表明在信息受限或目标不匹配时性能无法提升。

**⚠️ 局限性**

局限性包括：① 只考虑单一用户-系统的两阶段交互，未覆盖多模态或交互式反馈；② 结果在理论假设下成立，实际提示的分布与对齐策略可能更为复杂；③ 论文未给出实证验证，仅通过示例说明，实际模型的误差结构与理论预测可能存在差异。

---

## 817. Flatness Preserves Instruction Following in Vision-Language-Action Models

**arXiv ID:** 2606.23641 | [PDF](https://arxiv.org/pdf/2606.23641v1)

**作者:** Haochen Zhang `[一作]` (Carnegie Mellon University), Yonatan Bisk `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在视觉‑语言‑动作(VLA)模型的微调过程中，使用Sharpness‑Aware Minimization (SAM)来保持训练损失曲面平坦，从而缓解指令盲目性（instruction blindness）问题，显著提升模型对新指令的遵循能力。

**💡 创新点**

创新点在于首次将SAM这类面向全局平坦化的优化技术应用于机器人VLA微调，以解决训练样本稀疏导致的过拟合和视觉偏差。

**🔧 技术方法**

核心技术为SAM的双层梯度优化（对参数空间进行最坏情况梯度更新），以及对VLA模型的标准微调和梯度裁剪。

**📊 数据集**

实验使用了LIBERO-PRO、LIBERO-CF和LangGap三个仿真基准，以及DROID真实世界拾取任务的数据集。

**📈 对比分析**

与传统微调、LoRA、Bayesian‑Factorized、数据增强、指导技术等基线相比，SAM在所有基准上平均提升了约20‑30% 的任务成功率（如LIBERO-PRO 41.7%→60.2%），并能与指导方法叠加进一步提升性能。

**⚠️ 局限性**

主要限制包括：SAM 需要两次前向传播，计算成本近乎翻倍；未显式约束模型不偏离预训练权重，可能导致“漂移”；尽管性能提升显著，绝对成功率仍较低，表明视觉偏差和曲面尖锐度并非唯一导致指令盲目性的因素。

---

## 818. Learning Process Rewards via Success Visitation Matching for Efficient RL

**arXiv ID:** 2606.23640 | [PDF](https://arxiv.org/pdf/2606.23640v1)

**作者:** Raymond Tsao `[一作]` (University of California Berkeley), Sergey Levine `[通讯]` (University of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种将稀疏任务完成奖励转化为密集过程奖励的方法，利用鉴别器对成功与失败轨迹的状态-动作访问分布进行判别，指导策略匹配成功轨迹并避免失败轨迹；

**💡 创新点**

创新点在于仅依赖稀疏奖励即可自动生成保留最优策略的密集奖励，理论上证明在确定性环境下保持最优性，并在机器人任务中显著提升样本效率；

**🔧 技术方法**

采用对抗式鉴别器（GAN式）估计成功/失败轨迹访问比例，利用对数比奖励形式，并结合KL正则化与RL微调算法（Residual, RLPD, Diffusion Transformers）实现策略更新；

**📊 数据集**

在Meta-World多场景图像基准、Dexterous Kitchen (SAPIEN) 以及真实 WidowX 250 机器人手臂上进行实验，使用BridgeData V2 等大规模演示数据集进行预训练；

**📈 对比分析**

与多种奖励塑形基线（SORS、SASR、GAIL、RND、VLM-based）及仅使用稀疏奖励对比，所有任务中实现约 2 倍更快的样本效率，最终成功率均超过 80%，部分任务达 90% 以上；

**⚠️ 局限性**

理论证明仅适用于确定性环境，且在高度随机或连续空间下需进一步扩展；实验中对成功/失败轨迹收集的依赖可能限制在失败率极低或样本稀少情况下的效果。

---

## 819. Pose Anything Anywhere:Model-free Object Poses from Arbitrary References

**arXiv ID:** 2606.23634 | [PDF](https://arxiv.org/pdf/2606.23634v1)

**作者:** Hongli Xu `[一作]` (Technical University Of Munich), Slobodan Ilic `[通讯]` (Technical University Of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种模型无关的6D姿态估计框架，能够利用稀疏的单/多视角参考图像（包括RGB和RGB‑D），并通过可选的无姿态协助视角实现更鲁棒的姿态推断。

**💡 创新点**

创新点在于：1) 多视角几何Transformer骨干实现全局几何一致性；2) 引入几何感知的跨视角3D匹配模块，学习视角不变的对应；3) 通过姿态图聚合无姿态协助视角，扩展几何覆盖，消除单视角匹配的不确定性。

**🔧 技术方法**

使用了基于Transformer的多视角几何骨干、几何感知匹配头、局部对比学习、RANSAC+Umeyama姿态图注册、以及LoRA微调等技术。

**📊 数据集**

训练数据采用大规模合成的Omni6DPose数据集（5000+ CAD物体），测试数据则覆盖LINEMOD、YCB‑Video、Toyota‑Light、LM‑O和Real275等五个公开基准。

**📈 对比分析**

与多种单/多视角模型无关基线（如One2Any、SingRef6D、Any6D）以及模型基线（如MegaPose、FoundPose）进行对比，实验显示在单视角下提升约+12% ADD‑0.1d（YCB‑V）和+20% AR（LM‑O），在多视角和RGB‑D设置下保持最高或近乎最高的精度。

**⚠️ 局限性**

局限性包括：在极端遮挡或查询与参考重叠非常有限时难以建立可靠对应；对高度对称物体可能出现姿态歧义；对辅助视角数量的依赖导致在极少辅助视角时性能下降。

---

## 820. Why Machines Misread Pedagogical Quality: Human-Machine Alignment in LLM-Based Pretest Question Evaluation

**arXiv ID:** 2606.23629 | [PDF](https://arxiv.org/pdf/2606.23629v1)

**作者:** Pei-Yu Tseng `[一作]` (Pennsylvania State University), Peng Liu `[通讯]` (Pennsylvania State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了一套 AI 辅助的预测题目生成与筛选工作流，包括自动生成、基于评分量表的评估和迭代筛选，重点研究人类与 LLM 评估者在预测题质量评估中的一致性。

**💡 创新点**

创新点在于系统性地探究了人机评估差异的规律，提出通过对评分量表进行操作化修订与“先推理后评分”两种方式相结合，可显著提升 LLM 对开放性、深度等认知维度的判定准确性，揭示了评估者思路与量表设计的互补作用。

**🔧 技术方法**

技术手段主要包括：基于 GPT‑5.1 的大语言模型进行题目生成与评估；采用 2×2 实验设计（量表版本×评估模式）来检验人机一致性；利用四种评价指标（Bias、MAE、MSE、Agree%）对人机评分进行定量比较。

**📊 数据集**

数据集：以网络安全专业中级统计学课程（多元回归）为背景，生成 180 条候选预测题目（每个量表版本 12 条 × 15 轮），随后人工评审 60 条样本（每个量表版本）。

**📈 对比分析**

比较方法：在四种实验条件下分别计算人类评审与 LLM 评审的 Bias、MAE、MSE 与 Agree%；结果显示：在初始量表下开放性一致率仅 55.7%，经量表修订后提升至 83.3%，再加上 rationale‑first 评估后达到 94.5%；深度一致率从 70.5% 提升至 95%（量表修订）并在 rationale‑first 评估下略升至 87%/95%。

**⚠️ 局限性**

局限性：仅评估评估一致性，未检验提升后的题目对学习效果的真实影响；实验仅在单一 LLM（GPT‑5.1）和特定学科场景下进行；未与更广泛的自动或混合评估基线做对比，缺乏跨任务验证。

---

## 821. DiT-Reward: Generative Representations for Text-to-Image Reward Modeling

**arXiv ID:** 2606.23626 | [PDF](https://arxiv.org/pdf/2606.23626v1)

**作者:** Yuanming Yang `[一作]` (JD Explore Academy), Nan Duan `[通讯]` (JD Explore Academy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

将预训练的文本-图像扩散变压器（DiT）转化为奖励模型，并在多项人类偏好基准上进行评估；

**💡 创新点**

利用生成器自身的中间表示直接构造奖励模型，避免使用独立的视觉-语言编码器，并在生成器的潜在空间中直接评分；

**🔧 技术方法**

使用Diffusion Transformer、VAE、轻量化MLP奖励头、Bradley‑Terry偏好损失、Flow‑GRPO强化学习框架；

**📊 数据集**

HPSv3训练混合数据（HPDv3、HPDv2精华、Pick‑A‑Pic、ImageReward、Midjourney）以及Stable Diffusion 3.5 Large 的潜在空间；

**📈 对比分析**

与HPSv3在HPDv2、HPDv3、ImageReward、PickScore四个基准上对比，DiT‑Reward分别取得85.6%、77.6%、78.0%和72.3%的最高或接近最高分；在Flow‑GRPO强化学习中，DiT‑Reward训练出的策略在视觉质量、真实性和细节丰富度上均优于HPSv3；

**⚠️ 局限性**

在后期训练中仍表现出提示对齐能力弱、奖励劫持现象，且依赖生成器的预训练能力，若生成器容量不足或对输入噪声处理不当，奖励模型效果会下降。

---

## 822. Learning to See While Learning to Act: Diffusion Models for Active Perception in Robot Imitation

**arXiv ID:** 2606.23625 | [PDF](https://arxiv.org/pdf/2606.23625v1)

**作者:** Kuancheng Wang `[一作]` (Georgia Institute of Technology), Danfei Xu `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于扩散模型的视觉‑运动模仿学习框架，该框架在一次去噪循环中同时精炼动作预测和摄像头视角，使机器人在操作过程中主动寻找并获得更具信息量的视角。

**💡 创新点**

创新点包括：① 在扩散过程本身嵌入主动感知，动作去噪与摄像头位姿迭代天然耦合；② 通过仅使用关键帧动作标签来学习摄像头轨迹，无需额外的视角标注；③ 训练时利用数字孪生生成多视角观测，提升仿真‑真实迁移的鲁棒性。

**🔧 技术方法**

使用的技术：扩散策略（Diffusion Policy）、SE(3) 视角插值与三维几何推理、关键帧动作抽取、主动视角推断（Active Viewpoint Inference）以及数字孪生数据生成和深度相机感知。

**📊 数据集**

使用的数据集：Ravens 基准（4 任务）、RLBench（9 任务）以及在真实机器人上通过数字孪生收集的 50 条演示数据；训练过程中在仿真环境中生成多视角观测。

**📈 对比分析**

与 Conv‑MLP、Conv‑MLP‑MV、Diff‑MV、Diff‑MV‑Entropy、C3DM 等基线相比，框架在受遮挡的 Ravens 任务中实现 95% 以上的零样本仿真‑真实转移，在 RLBench 上平均成功率达 81.1%，并在多项任务中比基线提升 70%–96%。

**⚠️ 局限性**

限制：每一步都需新观测导致运行延迟较大、速度慢于固定视角方法；对相机标定误差敏感；对极端遮挡或不同机器人动力学的泛化能力仍需进一步验证。

---

## 823. The Table Says Otherwise: Testing LLMs with Counterfactual Relational Data

**arXiv ID:** 2606.23667 | [PDF](https://arxiv.org/pdf/2606.23667v1)

**作者:** Xinzhi Wang `[一作]` (Purdue University), Chunwei Liu `[通讯]` (Purdue University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个原始-对抗式的表格问答基准，用于检测 LLM 在回答表格问题时是依据表格数据还是预训练知识。

**💡 创新点**

创新点在于：①设计对抗数据库，保持模式和关系不变，仅替换部分事实；②将问答划分为单表查找、跨表查找和时间推理三层级；③量化原始与对抗数据集间的准确率差距（counterfactual gap），揭示模型对先验知识的依赖。

**🔧 技术方法**

使用了大语言模型推理、指令调优技术，并通过 GPT‑4o 进行自动二元评估；采用对抗式数据库生成和问题模板化生成技术；在实验中对多种商业闭源和开源模型进行评测。

**📊 数据集**

基准数据集基于 Transfermarkt 足球转会数据库，包含 37,000+ 球员、400+ 俱乐部、80,000+ 比赛、99,000+ 转会等表格，并构造了原始版本和对抗版本。

**📈 对比分析**

通过比较同一问题在原始与对抗数据库下的准确率来评估模型的表格依赖性；结果显示强模型在单表查找时准确率接近 100%，但在多表联接和时间推理时对抗差距显著增大；指令调优显著提升整体准确率，但仍未消除差距。

**⚠️ 局限性**

局限性包括：①对抗数据库只修改了有限属性，可能未覆盖所有先验知识冲突；②实验仅使用单一领域（足球转会），缺乏跨领域验证；③评价方法仅关注整体准确率，未细粒度分析错误原因；④模型对先验知识的依赖机制尚未深入解释。

---

## 824. A Reduced Order Model for Emergent Mechanics in Woven Systems

**arXiv ID:** 2606.23658 | [PDF](https://arxiv.org/pdf/2606.23658v1)

**作者:** Anvay A. Pradhan `[一作]` (University of Michigan), Talia Y. Moore `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

建立了一种基于条形与铰链框架的低阶模型，能够捕捉纺织结构的几何驱动机理（如编织特有的刚度、剪切锁定、扭曲交互、Poisson 效应等），并通过实验校准参数实现与实际织物在弯曲和剪切试验中的匹配。

**💡 创新点**

创新点在于将编织体系拆解为四类物理可解释的弹性元件（轴向柱、三节点角弹簧、四节点扭转弹簧、笛卡尔滑动弹簧），通过单元格本征模态分析验证必要性，并通过实验数据实现参数的全局优化，首次实现了可编程机械各向异性的低成本预测。

**🔧 技术方法**

使用了条形‑铰链（bar‑and‑hinge）结构化模型、能量函数导出内力与刚度矩阵、单元格本征模态分析、基于遗传算法的参数优化、有限元验证与数据驱动校准。

**📊 数据集**

实验数据集来自 7.5 mil Mylar 织物样品，尺寸 5×5 至 15×15 编织，采用三点弯曲（垂直与斜向）和图形框式剪切三种加载，测得力–位移曲线后归一化为弹性模量，用于校准。

**📈 对比分析**

方法通过分布式（单条件）和匀化（多条件）两种校准策略与实验比较，分布式校准误差≤1%，单调分支均满足 0.94+ 的 R²；统一校准在所有 12 条件下平均 R²≈0.7，误差约 5% 以内。模型与全尺寸 2D FEM 计算对比显示计算时间提升 1–2 个数量级。

**⚠️ 局限性**

局限性包括：需要针对每种织物材料和几何重新校准、仅适用于线性弹性行为、无法显式模拟摩擦滞后与滑移、节点中心线表示忽略纤维截面细节、对高度不规则或含缺陷的编织需要扩展。

---

## 825. Dynamic estimation of slowly varying sequences

**arXiv ID:** 2606.23655 | [PDF](https://arxiv.org/pdf/2606.23655v1)

**作者:** Prashant Gokhale `[一作]` (University of Wisconsin--Madison), Sandeep Silwal `[通讯]` (University of Wisconsin--Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对慢变序列（相邻元素差异小）中的各类函数（如矩阵迹、矩阵幂、光谱密度、蒙特卡洛积分、Dirichlet问题）提出了一套通用的自适应估计框架，动态地调整查询预算以实现高效序列估计。

**💡 创新点**

创新点包括：①将查询预算与局部变化量α_i之和（path‑length）相关，而非传统的m·maxα_i；②构建“良好集中”估计器的定义，保证任意线性或非线性目标均可应用；③设计了两阶段自适应算法，能够在线估计α_i，并在未知步长时仅额外消耗对数级查询；④在多种应用场景（动态迹估计、矩阵幂、光谱密度、蒙特卡洛积分、Dirichlet问题）上展示了理论与实验上的优势。

**🔧 技术方法**

主要技术手段是：利用子指数（sub‑exponential）集中性分析；基于递推的抑制因子和残差估计更新；对估计器的样本预算按α_t缩放；对α_t的估计使用Hutchinson或Monte Carlo估计器；对线性/非线性映射统一为线性映射的轨迹估计。

**📊 数据集**

实验数据集主要包括：1）2000×2000稀疏/随机矩阵的合成序列，设计了突发大扰动；2）实际神经网络训练轨迹（2,410参数多层感知机）在某分类任务上训练200步，使用SGDR学习率调度；3）在Dirichlet问题中采用随机游走/Walk‑on‑Spheres估计边界函数。

**📈 对比分析**

与目前最优的DeltaShift基线相比，自适应算法在相同误差和置信度下显著降低MVP/函数评估次数，实验中在三种误差度量（最大绝对误差、平均绝对误差、加权平均误差）上均实现更优的Pareto前沿；在Hessian迹估计实验中，累计查询量明显低于基线而保持相近误差。

**⚠️ 局限性**

局限性包括：①需先验或在线估计α_i，若估计误差较大会额外消耗；②对估计器的子指数集中性假设要求较强，某些非凸或高度非线性问题可能不满足；③在高维稠密矩阵或步长突变频繁的场景中，路径长度仍可能接近m，导致收益有限；④实验主要集中于矩阵迹及其变体，其他领域的应用还需进一步验证。

---

## 826. Muown Implicitly Performs Angular Step-size Decay

**arXiv ID:** 2606.23637 | [PDF](https://arxiv.org/pdf/2606.23637v1)

**作者:** Florian Hübler `[一作]` (ETH Zurich), Niao He `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Angular Muown，改写 Muown 的方向更新为在 oblique manifold 上的 Riemannian 步进，并显式调度角度步长，提升 Transformer 预训练的优化效率。

**💡 创新点**

将 Muown 隐含的行向量归一化与方向更新转化为 Riemannian 下降；引入可调的角度学习率乘子 κ_t，去除对权重衰减的依赖，提供更稳定、可控的步长调度。

**🔧 技术方法**

Riemannian 优化、行向量归一化（Oblique manifold）、Spectral orthogonalization (Newton–Schulz)、Adam 的行向量尺度更新、角度步长调度 κ_t、形状缩放 s_{m,n}=√(max(1,m/n))。

**📊 数据集**

FineWeb-Edu 语料、Qwen2‑0.5B、1.1B DeepSeek‑V3‑style MoE、124M PlainLM 等大规模语言模型数据集。

**📈 对比分析**

与 AdamW、NorMuon、Muon、Muown 进行对比；在 124M、500M、1.1B 模型、2.5B‑10B 令牌预算下，Angular Muown 速度提升约 2×（AdamW）和 1.5×（NorMuon）；在 Qwen2‑0.5B 上通过多步长调度实现更低 perplexity，并证明学习率迁移性良好。

**⚠️ 局限性**

实验仅限语言模型预训练与中等规模模型；未在 100B+ 参数或非 Transformer 架构上验证；对比实验集中在 dense 与 MoE，缺乏更广泛任务与规模的评估。

---

## 827. Hedgementation = Hedgerow Segmentation: A Remote Sensing Benchmark

**arXiv ID:** 2606.23615 | [PDF](https://arxiv.org/pdf/2606.23615v1)

**作者:** Nathan Senyard `[一作]` (University of British Columbia), Joséphine Gantois `[通讯]` (University of British Columbia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建并公开了一个基于法国国家尺度、10 米分辨率的“Hedgementation”基准，用于评估遥感数据下的篱垣（hedgerow）检测模型。

**💡 创新点**

创新点在于：①系统整合Sentinel‑2影像、AEF嵌入与BD Haie地面真值标签，形成统一、公开且可复现的数据集；②设计了空间与气候带两维的泛化评估方案；③提供了完整的代码和实验流程，方便后续研究。

**🔧 技术方法**

采用的技术包括：深度学习语义分割模型PASTIS U‑TAE和FTW，以及基于AEF嵌入的kNN、随机森林和逻辑回归等传统机器学习模型。

**📊 数据集**

使用的数据集为：法国全境的Sentinel‑2 10 米表面反射率影像、AEF年度嵌入、BD Haie v2线性篱垣标签，经过缓冲、栅格化和下采样得到10 米二值标签；再按温带与亚热带气候区划分。

**📈 对比分析**

方法比较采用IoU作为指标，三种基线模型在全数据集上最高IoU约>40%。PASTIS U‑TAE在近距离测试集上比远距离高约0.02，亚热带样本更难，温带训练时性能差距更大；在农业区域的IoU提升至约45.9%，显示农业区更易预测。

**⚠️ 局限性**

局限性包括：标注稀疏、模型整体精度仍偏低、空间/气候泛化仍受限、仅覆盖法国且气候带不平衡、未充分探索多源时空信息的融合。

---

## 828. AIR: Adaptive Interleaved Reasoning with Code in MLLMs

**arXiv ID:** 2606.23678 | [PDF](https://arxiv.org/pdf/2606.23678v1)

**作者:** Cong Han `[一作]` (Independent Researcher), Yujie Zhong `[通讯]` (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过构建冷启动数据集、采用自监督过滤和强化学习，训练出具备自适应交错推理和代码调用能力的多模态大语言模型

**💡 创新点**

提出两阶段冷启动数据构建管线、两种数据过滤策略（Self‑Sampled 与 Prior‑Filtered）以及带组约束的奖励函数，以实现自适应工具调用并提升训练稳定性

**🔧 技术方法**

使用多模态 LLM（如 Qwen2.5 VL）、文本推理链（CoT）生成、Python 代码执行沙箱、Group Relative Policy Optimization (GRPO) 强化学习与自定义奖励

**📊 数据集**

利用 MMK12 数据集构建冷启动样本（约 1.5k 条），在 ViRL39K 数据集上采集 13k 条强化学习数据，并对比 Qwen2.5 VL 基线

**📈 对比分析**

与 Qwen2.5VL‑7B 等多模态模型在 MathVista、MathVision、MathVerse、DynaMath、WeMath、LogicVista 等数学基准上对比，平均提升约 5.9–9.9 个百分点；在工具调用成功率超过 95% 的情况下，交错推理带来显著性能提升

**⚠️ 局限性**

受限于基模模型的计算与推理能力、工具库功能有限以及仅在数学推理基准上的评估，尚未验证在更广泛的多模态任务（如物理、金融等）中的有效性

---

## 829. IMAGIN-4D: Image-Guided Controllable Interaction Generation

**arXiv ID:** 2606.23675 | [PDF](https://arxiv.org/pdf/2606.23675v1)

**作者:** Sai Kumar Dwivedi `[一作]` (Meta), Shreyas Hampali `[通讯]` (Meta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于参考图像的4D人机交互生成方法，可在文本、物体几何、稀疏路径基础上，通过图像捕捉瞬时交互状态，生成细粒度控制的运动序列。

**💡 创新点**

创新点在于将图像条件分为空间与时间两部分，使用空间分解的交互状态 token 与帧感知 token，并通过角色感知 conditioning 保持不同条件的独立性。

**🔧 技术方法**

使用扩散模型、Q‑Former 视觉编码器、AdaLN 条件调节、图像-保持一致度度量以及自制的渲染管线。

**📊 数据集**

在 FullBodyManipulation（FBM）和 BEHAVE 两个全身人机交互数据集上训练并评估，并通过自制的图像渲染生成图像对齐。

**📈 对比分析**

与无图像条件的 CHOIS、InterDiff、MDM 以及单一全局图像 token 的基线进行对比，使用 FID、R‑Precision、图像遵从度等指标，显示空间分解与帧感知显著提升图像遵从度，且保持或提升运动质量。

**⚠️ 局限性**

主要局限在图像与非图像条件的权衡导致轨迹跟踪误差略增，且跨域图像迁移仍受限于渲染域差异。

---

## 830. Can LLMs Reliably Self-Report Adversarial Prefills, and How?

**arXiv ID:** 2606.23671 | [PDF](https://arxiv.org/pdf/2606.23671v1)

**作者:** Quang Minh Nguyen `[一作]` (KAIST), Taegyoon Kim `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大型语言模型（LLM）在遭受对抗性前缀攻击后，是否能在后续对话中识别自身生成的受损输出，并分析其内部“自我意识”机制。

**💡 创新点**

首次系统评估LLM在安全情境下的自我识别能力，揭示其主要依赖安全/拒绝机制；通过拒绝方向正交化与LoRA微调展示了自我识别信号的因果关系与可调性，提供了对安全模型自我监测的实证依据。

**🔧 技术方法**

关键技术包括：
- 对抗前缀（AdvPrefix）生成；
- 通过拒绝方向正交化（activation steering）消除模型拒绝相关激活；
- 三种LoRA微调策略（SFT、GRPO、DPO）用于提升自我识别；
- RoBERTa二分类器对模型回答进行“认定/拒绝”标签化；
- Llama Guard 3 作为安全判定器；
- 对比分析两种自我提问方式（意图probe vs 破坏probe）。

**📊 数据集**

使用四个公开安全基准：HarmBench、SocialHarmBench、JailbreakBench、StrongREJECT；包含 1,085 个去重后提示，覆盖多种攻击与正常情况。

**📈 对比分析**

比较方法：
- 计算在控制与前缀条件下的认定率，得到“识别差距”Δ；
- 在拒绝方向正交化前后对 Δ 进行统计，发现正交化后差距几乎消失；
- 在 LoRA 微调后对 Δ 进行再评估，发现所有模型（8B–27B）均提升自我识别，但大多数模型的攻击成功率（ASR）亦同步上升；
- 通过意图probe 与破坏probe 的对比，展示提问方式对识别结果的显著影响。
性能总结：
- 基线模型的认定率约 27%（即 27% 的受攻击输出被错误认定为“意图”），
- 拒绝方向正交化后差距降至 0±3%，
- LoRA 微调后差距平均提升 10–70% 但同时 ASR 也升高 5–30%。

**⚠️ 局限性**

局限性：
- 仅评估开放权重模型（最高 27B），不涵盖更大或专有模型；
- 只关注单一自我识别信号（认定率），无法排除其他潜在机制；
- 拒绝方向正交化可能同时削弱有益行为，无法完全区分自我识别与安全拒绝的因果路径；
- LoRA 微调虽提升识别，但产生安全代价，未找到同时提升安全与自我识别的方案；
- 数据集均为英文，跨语言普适性未知。

---

## 831. MAS-PromptBench: When Does Prompt Optimization Improve Multi-Agent LLM Systems?

**arXiv ID:** 2606.23664 | [PDF](https://arxiv.org/pdf/2606.23664v1)

**作者:** Juyang Bai `[一作]` (Johns Hopkins University), Laixi Shi `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了MAS-PromptBench基准，系统评估了系统提示（system prompt）在多代理LLM系统中的优化效果。

**💡 创新点**

创新点在于：①首次提出统一评测平台用于多代理提示优化；②通过大规模实验揭示提示优化在不同任务、拓扑、协议与团队规模下的可行性与瓶颈；③指出需在优化过程中考虑任务结构、通信协议与拓扑特征。

**🔧 技术方法**

使用了基于GEPA和MIPRO的多代理扩展提示优化器，结合LangGraph框架搭建多种工作流；实验覆盖推理、编码、工具调用等任务。

**📊 数据集**

使用了九个任务数据集，包含HotpotQA、MATH、APPS、BFCL、LiveCodeBench等，涵盖推理、编程与工具调用三大领域。

**📈 对比分析**

通过对比优化前后的系统提示效果，平均提升约4.2分，最大提升可达24分；在某些配置下可下降至-16分。提示优化在编码/工具调用任务和结构化通信协议下表现最优。

**⚠️ 局限性**

局限性包括：仅评估了两种提示优化方法，未覆盖更广泛的技术；在大团队规模下效果不佳；缺乏针对不同拓扑的专门优化策略。

---

## 832. CoorDex: Coordinating Body and Hand Priors for Continuous Dexterous Humanoid Loco-Manipulation

**arXiv ID:** 2606.23680 | [PDF](https://arxiv.org/pdf/2606.23680v1)

**作者:** Sikai Li `[一作]` (University of North Carolina at Chapel Hill), Mingyu Ding `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 CoorDex——一种将身体和手部运动先验分离后通过协调残差控制来实现步行与高自由度手部精细操作连续协同的学习管道。

**💡 创新点**

创新点在于：① 构造独立的身体先验和腕部稳定化手部先验；② 通过共享协调树和分离残差头实现两者的动态配合；③ 采用冻结的先验作为动作空间，大幅降低探索维度，提升高自由度协同操作的可训练性。

**🔧 技术方法**

使用技术包括：Isaac Lab 虚拟仿真、基于人机遥控的参考轨迹采集、强化学习（PPO）、变分推断先验蒸馏、协调残差策略网络、任务层级奖励设计。

**📊 数据集**

数据集主要由仿真中采集的遥控参考运动构成，涵盖三类任务（WalkGrab、OpenFridge、WalkPickTurn）及其对应的手部动作和物体交互示例；手部运动通过 ManipTrans 风格的重映射与 WUJI 五指手相结合。

**📈 对比分析**

对比方法包括 All Joint Space、Body Prior + Hand Joint Space、Monolithic Latent Residual；在 WalkGrab、OpenFridge、WalkPickTurn 三个任务中，CoorDex 的成功率分别为 0.55、0.66、0.89，显著优于基线，且在动作平滑度、跌倒率等指标上也表现更佳。

**⚠️ 局限性**

局限性：仅在仿真环境验证，缺乏视觉感知和实机转移；仅测试单一 G1 + WUJI 组合；长期任务仍依赖任务特定的探索辅助；对不同物体、手部硬件的泛化能力尚待进一步研究。

---

## 833. Open Problem: Is AdamW Effective Under Heavy-Tailed Noise?

**arXiv ID:** 2606.23676 | [PDF](https://arxiv.org/pdf/2606.23676v1)

**作者:** Dingzhi Yu `[一作]` (Nanjing University), Lijun Zhang `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文在大语言模型训练中，针对 AdamW 在重尾梯度噪声下的收敛性进行理论分析，提出开放问题并给出加权指标的收敛上界及其下界构造。

**💡 创新点**

创新点在于将 AdamW 的收敛分析从传统有限方差假设扩展到仅有有限 p 阶矩的重尾噪声场景，并首次揭示了 AdamW 在长记忆调度下的自归一化权重可能导致对齐缺失的机制。

**🔧 技术方法**

采用重尾噪声模型、梯度-曲率(GC)条件、下界构造、对齐缺陷量 ε_T(c) 的分析，以及对自归一化权重的概率和期望估计等技术手段。

**📊 数据集**

论文为纯理论工作，未使用具体数据集；所给实验仅为模拟证明重尾噪声模型的有效性。

**📈 对比分析**

与基于符号的优化器（如 Lion、Muon、SignSGD）比较，作者在加权指标上实现了与符号优化器相同的重尾收敛率 O(T^{-(p-1)/(3p-2)})，但在普通 ℓ_1 站稳性上尚未达到。

**⚠️ 局限性**

主要限制在于无法给出 AdamW 在一般长记忆设置下的 ℓ_1 站稳性上界；加权指标的收敛无法直接推广，且对齐缺失机制表明 AdamW 可能在实际重尾环境中表现不佳。

---

## 834. Teaching LLMs String Matching, Backtracking, and Error Recovery to Deduce Bases and Truth Tables for the Combinatorially Exploding Bit Manipulation Puzzles

**arXiv ID:** 2606.23672 | [PDF](https://arxiv.org/pdf/2606.23672v1)

**作者:** Prateek Agnihotri `[一作]`, Shubham Jain `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一套将位操作谜题转化为基数选择与字符串匹配的确定性求解器，并通过交互式推理SFT将其嵌入LLM，实现高精度推理。

**💡 创新点**

创新点包括：①用 Bases 与 Truth Table 将位操作重构为 Set Cover 问题；②基于最小比特翻转提取逻辑约束；③采用严格单比特 token 化与动态遮蔽的交互式 SFT，赋予 LLM 自检与回溯能力。

**🔧 技术方法**

技术手段：DFS 回溯搜索 + 全局碰撞检测；交互式推理 SFT + 动态遮蔽；单比特 token 化；Python 实现（公开可获取）。

**📊 数据集**

使用 NVIDIA Nemotron Model Reasoning Challenge 的 Bit Manipulation puzzles 作为原始数据集，并生成自合成的 DFS trace 数据进行训练。

**📈 对比分析**

与竞赛中其他团队对比，算法在 1,602 例验证集上达到 98.63% 的准确率；LLM 在 Synthetic Only 训练下获得 96.13%（官方 7th place），混合训练略降至 94.63%；通过回溯次数统计验证模型能正确识别并纠正冲突。

**⚠️ 局限性**

局限性：①数据不足导致部分谜题欠定，无法唯一确定规则；②对 OOD 目标状态或多解情况的处理仍不完备；③LLM 上下文长度限制导致回溯过多时截断错误；④训练步数受限，未能完全达到理论 98.63% 的上限。

---

## 835. GeoFidelity-Bench: Evaluating Segment-Level Geographic Fidelity in Text-to-Image Street-View Generation

**arXiv ID:** 2606.23669 | [PDF](https://arxiv.org/pdf/2606.23669v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 836. RECALL: Recovery Experience Collection for Active Lifelong Learning in Vision-Language-Action Models

**arXiv ID:** 2606.23617 | [PDF](https://arxiv.org/pdf/2606.23617v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 837. AutoDex: An Automated Real-World System for Dexterous Grasping Data Collection

**arXiv ID:** 2606.23689 | [PDF](https://arxiv.org/pdf/2606.23689v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 838. TailorMind: Towards Preference-Aligned Multimodal Content Generation

**arXiv ID:** 2606.23643 | [PDF](https://arxiv.org/pdf/2606.23643v1)

**作者:** Hengji Zhou `[一作]` (South China University of Technology), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 TailorMind 框架，实现了基于用户行为数据的多模态内容生成，使系统能够生成符合用户偏好的全新图文与视频内容。

**💡 创新点**

创新点在于将超图协同过滤与文本梯度优化相结合生成可解释的用户偏好描述，并通过检索增强的风格控制与跨模态一致性反射，显著降低语义漂移并提升个性化质量。

**🔧 技术方法**

主要技术包括超图图神经网络（GNN）进行协同过滤、LLM 进行文本梯度优化与风格检索、ClipScore 与 Relevance 进行跨模态一致性评估，以及多模态生成模型（如 Gemini、Veo3.1 等）。

**📊 数据集**

使用了从 Rednote、Bilibili、Hupu 三大主流平台收集的真实交互与多模态内容数据构成的 TailorBench 基准，包含文本、图像、视频等多模态信息。

**📈 对比分析**

与多种生成基线（GPT‑4o、Sora2、CIPHER 等）、推荐模型（LightGCN、IRLLRec 等）和 LLM 排序方法进行对比，TailorMind 在个人化检索召回、内容新颖度、审美评分等指标上均超过或持平，且在用户手工评估中获得更高的创新性与审美评价。

**⚠️ 局限性**

局限性包括对平台公开的用户交互数据的依赖，难以直接获取创作者的细粒度关注信号；系统对极度稀疏或噪声交互的鲁棒性仍有限，且在生成多模态内容时可能需要进一步的安全与事实核验控制。

---

## 839. Improving Long-Context Retrieval with Multi-Prefix Embedding

**arXiv ID:** 2606.23642 | [PDF](https://arxiv.org/pdf/2606.23642v1)

**作者:** Zhenglin Yu `[一作]` (University of Waterloo), Jimmy Lin `[通讯]` (University of Waterloo)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种 Multi‑Prefix Embedding (MPE) 方法，用 EOS 分隔长文档，在一次前向推理中提取每个前缀的隐藏状态作为文档的多向量表示，并通过 MaxSim 与查询向量匹配；同时引入随机前缀长度增强来提升粒度鲁棒性。

**💡 创新点**

在保持单向 LM 结构不变的前提下，利用前缀隐藏状态产生跨块上下文的多向量表示；通过 MaxSim 直接定位匹配块并提供轻量级的证据定位；随机前缀长度采样进一步增强模型对不同粒度的适应性。

**🔧 技术方法**

使用 causal LM（Qwen3‑Embedding‑0.6B）+ LoRA 微调 + MaxSim 对比损失 + FAISS 直方内积索引 + 随机前缀长度增强 + 归一化 + 交叉设备负样本；实验中也对比了双向注意力等变体。

**📊 数据集**

在 MLDR‑en、BrowseComp‑Plus 以及 LongEmbed（包含 NarrativeQA、2WikiMQA、SummScreen、QMSum）等长文本检索基准上进行评估。

**📈 对比分析**

与单向量、独立分块（MaxP）以及 ColBERT 风格多向量等 baseline 进行对比；MPE 在 MLDR‑en 上获得最高 nDCG@10，MPE‑Rand 在 BrowseComp‑Plus 和 LongEmbed 上超越单向量，且在粒度不匹配时保持更稳健的性能。

**⚠️ 局限性**

仅在 0.6B 参数的英文模型上测试；多向量表示相比单向量增加存储和检索成本；与 Landmark Embedding、Late Chunking 的对比未在相同条件下完整对齐；证据定位评估依赖 LLM 自动注释，可能存在噪声。

---

## 840. AI-driven Optimisation of Quality of Recovery (QoR) in Remote Patient Monitoring

**arXiv ID:** 2606.23631 | [PDF](https://arxiv.org/pdf/2606.23631v1)

**作者:** Yansong Liu `[一作]` (University College London), Ivana Drobnjak `[通讯]` (University College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在远程病人监测系统中，构建并验证了一种仅含5项的QoR-15简化版（QoR-compact），通过穷举评估所有可能的5项子集，并在预测术后短期恢复严重程度时保持与完整15项问卷相当的预测性能。

**💡 创新点**

创新点在于：①以预测性能为目标而非传统心理测量学构念，对QoR-15进行数据驱动的短表优化；②采用全子集评估+多模型检验，结合统计显著性筛选，确保所选项在不同子集中稳定表现；③证明了在AI-RPM场景下，仅用三分之一每日问卷即可维持高效预测，降低患者答题负担。

**🔧 技术方法**

技术手段包括：滑动窗口特征提取、Spearman相关与VIF分析、层次聚类、XGBoost多分类模型、10折分层自助抽样评估、AUC-ROC加权计算、置信区间估计、患者级回测与住院再入院事件对齐。

**📊 数据集**

数据集来源于HALO-Surgery前瞻性队列（IRAS 284073），收集了腹部或胸部癌症手术后出院患者在远程监测平台每天完成的QoR-15问卷，共1035条记录，经过滑动窗口后生成144个输入-输出对。

**📈 对比分析**

在10次分层自助抽样下，QoR-compact的平均AUC-ROC为0.968（95% CI 0.915–0.988），与完整15项基线0.964（95% CI 0.879–0.994）基本持平。回测显示两者在检测再入院事件时表现一致，QoR-compact在急性恶化点略具更高反应性。

**⚠️ 局限性**

局限性包括：单中心、样本量有限；模型特征与标签均来自QoR-15，可能存在内在冗余导致AUC偏高；优化结果对特定病例组合敏感，需要在多中心更大规模队列中外部验证；未验证对其他手术类型或疾病的适用性。

---

## 841. EnterpriseClawBench: Benchmarking Agents from Real Workplace Sessions

**arXiv ID:** 2606.23654 | [PDF](https://arxiv.org/pdf/2606.23654v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 842. AI Exposure Scores: what they measure, what they miss, and what comes next

**arXiv ID:** 2606.23633 | [PDF](https://arxiv.org/pdf/2606.23633v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 843. Semantic Browsing: Controllable Diversity for Image Generation

**arXiv ID:** 2606.23679 | [PDF](https://arxiv.org/pdf/2606.23679v1)

**作者:** Sara Dorfman `[一作]` (Tel Aviv University), Daniel Cohen-Or `[通讯]` (Tel Aviv University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于多智能体的Semantic Browsing方法，通过结构化语义决策在文本到图像模型中生成可导航的多样化图像集合。

**💡 创新点**

将多样性从随机采样转为显式语义控制，利用VLM的语义推理与多智能体工作流构建层次化树，实现可解释且多维度的多样化生成。

**🔧 技术方法**

采用VLM进行场景解析与细化，构建多智能体（Context Analyst、Brainstormer、Decision Maker、Critic）工作流，使用FIBO与FLUX等文本到图像生成模型，并用Vendi Score、DINO等指标评估多样性与质量。

**📊 数据集**

主要使用MS-COCO中的50条随机提示作为评估集，并在公开数据集上进行实验。

**📈 对比分析**

与多种基线（随机种子、后验多样性优化、CADS、Guidance Interval、Power-Law CFG等）进行对比，结果在Vendi Score和DINO相似度上显著优于所有基线，同时保持相当的美学得分和提示符合度，用户研究也显示其多样性和偏好显著提升。

**⚠️ 局限性**

依赖底层生成模型对细粒度提示的忠实实现；VLM在提出丰富多样语义选项上有限；生成空间受限于模型能力，且对非图像领域的推广尚未验证。

---

## 844. PsyBridge: A Hybrid Intelligent Framework for Multi-Dimensional Mental Health Assessment and Decision Support

**arXiv ID:** 2606.23673 | [PDF](https://arxiv.org/pdf/2606.23673v1)

**作者:** Sunil Wanjari `[一作]` (St. Vincent Pallotti College of Engineering and Technology), Stanly Wilson `[通讯]` (St. Vincent Pallotti College of Engineering and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了PsyBridge框架，将PHQ-9、GAD-7、认知测评和人格画像整合为一个可解释的多维心理健康评估系统；

**💡 创新点**

创新点在于采用加权聚合的可解释决策模型，将临床、认知与人格信息统一化，并通过模块化架构实现可扩展、透明的风险分类与建议；

**🔧 技术方法**

采用了模糊归一化、线性加权聚合、阈值映射等技术，并在Python环境下实现各模块；

**📊 数据集**

使用了基于PHQ-9与GAD-7临床分布的半合成数据集，共500个样本，覆盖低、中、高风险三类；

**📈 对比分析**

与单一PHQ-9或GAD-7的基线方法对比，PsyBridge在准确率0.84、精准率、召回率及F1分数均显著提升；

**⚠️ 局限性**

主要限制是数据来源为半合成，缺乏真实临床多样性，且权重固定，需后续在真实数据上验证并探索自适应权重与更多模态输入。

---

## 845. dVLA-RL: Reinforcement Learning over Denoising Trajectories for Discrete Diffusion Vision-Language-Action Models

**arXiv ID:** 2606.23623 | [PDF](https://arxiv.org/pdf/2606.23623v1)

**作者:** Yuhao Wu `[一作]` (Shanghai Jiao Tong University), Yao Mu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于轨迹级概率的强化学习框架，直接对离散扩散 VLA 的完整去噪路径进行概率建模，并用 PPO 进行训练；同时引入混合去噪步策略（Hybrid）来平衡性能与计算效率。

**💡 创新点**

①把离散扩散动作生成看作一个 Markov 链，突破传统只估计最终动作边缘概率的瓶颈；②在轨迹概率上训练，可直接对实际执行的去噪路径求梯度；③通过忽略不易求导的掩码调度，将梯度限定在真正生成的 token 上；④混合去噪步策略在不同任务上自适应选择步数，实现样本效率与推理效率的双赢。

**🔧 技术方法**

离散扩散模型、Gumbel-TopK 掩码调度、PPO 强化学习、轨迹级对数似然、混合去噪步调度、基准评测工具（RLinf）等。

**📊 数据集**

LIBERO（四个单臂操纵子任务）与 RoboTwin 2.0（双臂协同操纵，包含多种物体与工具交互）。

**📈 对比分析**

与 SFT 仅训练的 MM-ACT、SimpleVLA-RL、OpenVLA-OFT 等 VLA 基线以及 WAM 基线（Motus、Cosmos Policy、LingBot-VA 等）做对比。实验表明，在 LIBERO 上平均成功率达到 99.7%，在 RoboTwin 2.0 上平均成功率提升至 92%（比 SFT 基线提升 30.6%），并在多项指标上超过或接近最强的 VLA 与 WAM 方法。

**⚠️ 局限性**

①仍依赖稀疏奖励，探索效率受限；②掩码调度非可微导致梯度噪声需通过间接方式优化；③仅在仿真环境验证，缺乏真实机器人实验；④对超参数（去噪步数、掩码阈值）敏感，需手工调整；⑤混合去噪步策略虽然提高效率，但在不同任务间的动态分配仍基于经验规则。

---

