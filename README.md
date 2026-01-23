# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-01-23 | 今日论文总数: 383

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Is Grokipedia Right-Leaning? Comparing Political Framing in Wikipedia and Grokipedia on Controversial Topics

**arXiv ID:** 2601.15484 | [PDF](https://arxiv.org/pdf/2601.15484v1)

**作者:** Philipp Eibl `[一作]` (University of Southern California), Luca Luceri `[通讯]` (Information Sciences Institute)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比了 Wikipedia 与 Grokipedia 在六个政治争议主题（堕胎、药物合法化、气候变化、性别认同、枪支管制、移民）上的语义相似度、政治倾向与内容优先级

**💡 创新点**

首次系统评估 AI 驱动百科 Grokipedia 在争议话题上的政治倾向与传统 Wikipedia 的差异，揭示两者均偏左但 Grokipedia 对右倾内容的聚焦更明显

**🔧 技术方法**

使用 GPT‑5 生成段落嵌入、RoBERTa 政治立场分类器、nDCG 评价指标进行定量分析

**📊 数据集**

从 Wikipedia 与 Grokipedia 中抓取 6 个争议主题共 12 篇文章（每个平台 6 篇），并对其段落进行预处理

**📈 对比分析**

通过段落级余弦相似度、t 检验和 nDCG 对比，发现两平台的语义相似度随段落递减，政治倾向整体偏左，Grokipedia 的右倾内容更突出，左倾内容在页面顶部的优先级相近

**⚠️ 局限性**

方法受限于段落嵌入无法捕捉细微修辞与语气、政治立场分类器的噪声以及仅覆盖美国中心议题的样本，难以全面评估事实准确性与引用质量

---

## 2. Lost in Transcription: How Speech-to-Text Errors Derail Code Understanding

**arXiv ID:** 2601.15339 | [PDF](https://arxiv.org/pdf/2601.15339v1)

**作者:** Jayant Havare `[一作]` (Indian Institute of Technology Bombay), Ganesh Ramakrishnan `[通讯]` (Indian Institute of Technology Bombay)

**通讯引用:** 2044 | [OpenAlex ID](https://openalex.org/A5089606464)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个多语种、语音驱动的代码理解框架，能够接受印度本土语言的口语查询，先通过 ASR 转写，再用 LLM 进行代码感知的转写修正，最后利用代码模型完成代码问答与检索，并将结果以文字和语音方式反馈给用户。

**💡 创新点**

创新点包括：① 代码感知的 LLM 修正模块，专门针对代码词、符号、标识符的转写错误；② 对语音转写错误模式的系统性分类与分析；③ 以低资源印度语种为重点的多语言实验，验证 LLM 纠错在不同语言上的有效性；④ 在传统文本代码理解基准上加入语音输入的完整端到端评估。

**🔧 技术方法**

技术栈包括：ASR（Whisper + 适用于印度语种的 ASR 模型），LLM（GPT‑4o‑mini 主导的转写修正与翻译，Gemma‑9B、Qwen‑30B 作为代码模型），TTS（Microsoft Edge TTS），评估指标实现工具（Epitran、PanPhon），以及与 Phi‑4‑multimodal‑instruct 等跨模态 LLM 的对比。

**📊 数据集**

使用 CodeSearchNet、CoRNStack、CodeQA 三大代码问答/检索基准；对每个基准在 Python/Java/PHP 三种代码语言下，结合四种印度语（Hindi、Gujarati、Tamil、Bengali）和英语，构建约 18,000 条多语种查询-代码对；同时收集 53 名印度本科生的语音样本做实验。

**📈 对比分析**

与传统单一 ASR 输出（未修正）以及直接使用多模态 LLM（Phi‑4、Qwen3‑Omni‑Flash）的结果对比。修正后 WER 下降约 20‑30%，PER 下降 25‑35%，WFED 下降 30‑40%；在代码检索任务中，Recall@5 从 87% 提升到 90%，MRR 也提升至 84%；与直接多模态 LLM 的比较显示，独立的 ASR + 代码感知修正方案在所有转写指标上都优于单一端到端模型。

**⚠️ 局限性**

局限性包括：① 仍然依赖于大模型（GPT‑4o‑mini）的可用性和成本；② 在极低资源语言（如 Tamil、Bengali）中转写错误率仍高；③ 目前仅覆盖印度四种主要语言，扩展到更多语言需额外标注与训练；④ 评估多在实验室环境，真实噪声、口音多样性仍待进一步验证；⑤ 仅处理代码问答/检索，未覆盖代码编写、自动修复等更广泛任务。

---

## 3. MARS: Unleashing the Power of Speculative Decoding via Margin-Aware Verification

**arXiv ID:** 2601.15498 | [PDF](https://arxiv.org/pdf/2601.15498v1)

**作者:** Jingwei Song `[一作]` (University of Hong Kong), Lynn Ai `[通讯]` (gradient)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种Margin-Aware Speculative Verification（MARS）机制，通过在低边际情况下放宽验证要求，从而加速LLM推理。

**💡 创新点**

创新点在于不需要额外训练或修改目标模型，仅利用目标模型的logit比值动态决定是否接受次优token，实现完全无监督、插件式的验证改进。

**🔧 技术方法**

使用logit比率作为决策稳定性指标，并在推理时基于目标模型logits实现自适应验证；实现方案嵌入现有的Speculative Decoding框架中。

**📊 数据集**

在8B到235B不同规模的模型上实验，使用MT-Bench、HumanEval、MBPP、GSM8K、AlpacaEval、CNN/DailyMail等多任务数据集评估。

**📈 对比分析**

与vanilla自回归推理、Medusa、EAGLE‑2/3、Prompt Lookup等主流Speculative Decoding方法比较，MARS在保持98%–100%任务准确率的前提下，平均速度提升约3.1×至4.8×，并显著增加平均接受长度。

**⚠️ 局限性**

局限性包括仅在token级别考虑logit边际，未探究更高层次的语义/结构信息；单一全局阈值可能对某些上下文不够细粒度；与其他解码约束或特定应用的兼容性尚待进一步研究。

---

## 4. Rules Create Unequal Rewards: Elite Tennis Players Allocate Resources Efficiently

**arXiv ID:** 2601.15327 | [PDF](https://arxiv.org/pdf/2601.15327v1)

**作者:** Masatsugu Yoshizawa `[一作]` (University of Tokyo), Daisuke Takeshita `[通讯]` (University of Tokyo)

**通讯引用:** 1471 | [OpenAlex ID](https://openalex.org/A5083821374)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用大规模 Grand Slam 单打点数级别数据，构建基于得分状态的马尔可夫模型，提取每位球员的得分概率分布，并通过 Pareto 前沿评估其在游戏胜率与平均耗时之间的效率。

**💡 创新点**

创新点在于将多目标优化（游戏胜率 vs 期望点数）与 Pareto 前沿结合，提出“效率分数”衡量球员在规则限制下的资源配置最优程度，并发现顶级球员更接近理论最优。

**🔧 技术方法**

采用吸收马尔可夫链建模、NSGA‑II 遗传算法搜索 Pareto 前沿，计算欧氏距离得到效率分数和策略拟合度，并用统计量（Cliff’s Δ、Mann‑Whitney、Cohen’s d）比较不同水平球员。

**📊 数据集**

数据集来源于 2012‑2022 年四大满贯（澳大利亚公开赛、法国公开赛、温布尔登、美国公开赛）的公开点数级别记录，共 178 名球员（84 男 94 女），覆盖约 1.8 亿点数。

**📈 对比分析**

通过比较三档比赛胜率（低 <50%，中 50–70%，高 >70%）的效率分数和策略拟合度，发现高档球员在各类比赛中效率显著更高（Δ ≥ 0.47，效果大），表明他们的资源配置更接近理论最优。

**⚠️ 局限性**

局限性包括：观察性设计无法确立因果关系；使用长期平均策略可能掩盖短期波动；未考虑疲劳、心理压力或场地等外部因素；仅覆盖 Grand Slam 单打比赛，结果可能不完全推广到其他赛事或双打。

---

## 5. Testing Deep Learning Libraries via Neurosymbolic Constraint Learning

**arXiv ID:** 2601.15493 | [PDF](https://arxiv.org/pdf/2601.15493v1)

**作者:** M M Abid Naziri `[一作]` (North Carolina State University), Saikat Dutta `[通讯]` (Cornell University)

**通讯引用:** 667 | [OpenAlex ID](https://openalex.org/A5063258857)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种神经符号技术，利用大语言模型学习深度学习库 API 的输入约束并通过 SMT 求解器生成有效测试输入，从而对 PyTorch 与 TensorFlow 的 API 进行高效覆盖和缺陷发现。

**💡 创新点**

创新点在于将专用语法表征的约束规则与 LLM 自动生成、动态验证、精简相结合，首次实现对 API 参数关系的符号学习；并通过抽象模型+采样方式显著提升测试输入多样性与有效率。

**🔧 技术方法**

技术方法包括：① 基于 Lark 的第一阶逻辑约束语法；② 以 Gemini 2.0 Flash 为核心的 LLM 进行规则生成与输入/文档提示；③ 通过 Z3 SMT 求解器求解抽象模型并实现抽样与块化（blocking、bucketing）生成具体输入；④ 迭代精简冗余约束。

**📊 数据集**

数据集主要来自 PyTorch 759 个 API 与 TensorFlow 718 个 API 的官方文档、错误信息（共 2,642/2,822 条）以及 LLM 生成的 117 条有效种子输入。

**📈 对比分析**

实验中与 DeepREL、DocTer、Pathfinder、ACETest 等基线进行对比，结果显示 Centaur 在有效率上提升至约 95%（相较于 43%~21%），分支覆盖平均提升 10%~20%，并在 180 秒预算下产生百万级输入，成功发现并报告 21 个新缺陷，其中 9 个已被开发者确认。

**⚠️ 局限性**

主要局限包括：① 约束语法仍有限，无法表达所有可选参数或上下文条件；② 对 LLM 生成质量与一致性敏感；③ 在 GPU/CPU 差异等差分测试中容易产生误报；④ 需要离线生成阶段，未实时适应 API 更新。

---

## 6. RECAP: A Resource-Efficient Method for Adversarial Prompting in Large Language Models

**arXiv ID:** 2601.15331 | [PDF](https://arxiv.org/pdf/2601.15331v1)

**作者:** Rishit Chugh `[一作]` `[通讯]`, Rishit Chugh

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于检索的资源高效对抗性提示方法 RECAP，利用预训练的对抗性提示库代替昂贵的梯度训练，实现对 LLM 的高效攻击。

**💡 创新点**

创新点在于将 GCG、PEZ、GBDA 三种攻击策略生成的对抗性提示按意图分类并构建检索数据库，既不依赖模型内部 logits 又能快速匹配合适的攻击句子。

**🔧 技术方法**

主要技术包括检索增强生成（RAG）、FAISS 向量检索、SentenceTransformer 编码、以及预先生成的对抗性提示。

**📊 数据集**

使用的数据集包括 1,000 条按七类有害意图标注的提示（用于构建检索库），评估时用 Llama 3 8B、Vicuna 7B、Phi、Gemini API，并采样 LibrAI Do‑Not‑Answer 与 IndicAlign 作为测试样本。

**📈 对比分析**

通过与 PEZ、GBDA、GCG 的对照实验，RECAP 的平均成功率为 33%（与 GCG 的 59% 相比略低），但在同等条件下耗时仅 4 分钟，远快于 GCG 的 8 小时或 PEZ/GBDA 的 7 分钟。

**⚠️ 局限性**

局限性主要是检索数据库规模有限导致成功率略低，缺乏足够覆盖时可能匹配不到高效提示；在某些意图类别上仍无法匹配 GCG 的最高攻击效果，且迁移性在更大模型上仍需进一步验证。

---

## 7. MALTopic: Multi-Agent LLM Topic Modeling Framework

**arXiv ID:** 2601.15299 | [PDF](https://arxiv.org/pdf/2601.15299v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 8. Improving MoE Compute Efficiency by Composing Weight and Data Sparsity

**arXiv ID:** 2601.15370 | [PDF](https://arxiv.org/pdf/2601.15370v1)

**作者:** Maciej Kilian `[一作]`, Armen Aghajanyan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究在自回归Transformer中通过在Mixture-of-Experts中引入零计算专家实现权重稀疏与数据稀疏，提升视觉‑语言模型训练的计算效率。

**💡 创新点**

创新点在于将零计算专家嵌入token‑choice MoE中，既保持因果性又实现数据稀疏；通过加载平衡和z‑loss实现软约束，模型可自动进行模态感知的计算分配，组合权重与数据稀疏能显著提升计算效率。

**🔧 技术方法**

采用token‑choice MoE、零计算专家（null experts）、加载平衡与z‑loss训练目标、Transformer‑based视觉‑语言框架，以及数据稀疏控制参数ρ等技术。

**📊 数据集**

使用视觉‑语言混合数据集进行训练，并在10个标准基准（AI2D、A‑OKVQA、BLINK、ChartQA、Perceptron Grounding、DocVQA、M3Exam、SEED‑Bench、TextVQA、VSR）上评估。

**📈 对比分析**

在相同期望FLOPs下比较不同k_max和ρ的MoE配置，发现ρ≈0.5时训练损失降低且下游任务（尤其OCR和计数）性能提升；在更大模型规模下仍保持优势。

**⚠️ 局限性**

局限性包括高稀疏率下评估性能不随损失提升而持续上升；单一softmax分布难以同时精准建模权重与数据稀疏；软约束对实现目标稀疏率存在限制。

---

## 9. QUAIL: Quantization Aware Unlearning for Mitigating Misinformation in LLMs

**arXiv ID:** 2601.15538 | [PDF](https://arxiv.org/pdf/2601.15538v1)

**作者:** Himanshu Mishra `[一作]` (University of British Columbia), Kanwal Mehreen `[通讯]` (University of British Columbia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究低精度量化对机器无学习的破坏作用，提出一种量化感知的无学习方法 QUAIL。

**💡 创新点**

理论揭示量化桶坍塌导致忘记信息恢复；在对数it空间加入 hinge 损失，强制参数跨越量化阈值，从而保证无学习在低精度量化下仍然有效。

**🔧 技术方法**

使用梯度上升/下降结合量化感知 hinge 损失、对数it空间约束；结合 GA/GDR、AWQ、GPTQ 等量化技术；评估 ROUGE、PrivLeak 等指标。

**📊 数据集**

采用 MUSE (NEWS/BOOKS) 及 Twitter 虚假信息数据集进行实验。

**📈 对比分析**

与 GA、GA_GDR 等基线在 16‑bit 与 4‑bit 量化下对比；在 4‑bit 量化时 QUAIL 将 M1/M2（遗忘效果）显著下降、M3（隐私泄露）接近 0，同时保持或提升 M4（保留数据性能），大幅降低基线 80%+ 的信息恢复。

**⚠️ 局限性**

对超参数（α、γ、Δq）敏感；主要针对 4‑bit 量化，其他精度与混合精度尚未充分验证；额外计算与内存开销；采用全局阈值，缺乏层级自适应；未在更大模型、跨语言或多模态任务上进行评估；缺少严格理论上界。

---

## 10. A Universal Large Language Model -- Drone Command and Control Interface

**arXiv ID:** 2601.15486 | [PDF](https://arxiv.org/pdf/2601.15486v1)

**作者:** Javier N. Ramos-Silva `[一作]` (University of California), Peter J. Burke `[通讯]` (University of California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了基于MCP标准的通用无人机控制接口，使任何支持Mavlink的无人机均可通过LLM以自然语言进行飞行指令控制；

**💡 创新点**

首次将MCP统一协议与Mavlink结合，构建云端MCP服务器，支持多款LLM（OpenAI、Anthropic、Google Gemini及开源模型）与多种无人机平台，实现实时动态决策与交互；

**🔧 技术方法**

使用Python实现MCP服务器，集成MavSDK（Mavlink高级抽象）、Mavlink协议、云Linux实例、Tailscale VPN、OpenAI/Claude/Google Gemini LLM及其Agents SDK、Google Maps MCP服务器、SITL仿真、Mission Planner、QGroundControl等技术栈；

**📊 数据集**

主要利用实时地图数据（Google Maps）以及无人机自身传感器数据（GPS、LIDAR、光流）和实时天气信息；未使用专门的大规模训练数据集，LLM训练数据来源于各自模型的预训练；

**📈 对比分析**

通过在真实小型无人机和SITL仿真中执行起飞、悬停、导航、降落等任务进行验证，LLM能够准确完成指令并展示良好的执行成功率，证明自然语言控制的可行性；

**⚠️ 局限性**

LLM缺乏持续监控与实时反馈，工具调用次数有限，导致对长周期任务支持不足；决策非确定性需人工监督；安全性与法规尚未完善；服务器需持续运行，依赖网络连接。

---

## 11. Preparation and Motion Study of Magnetically Driven Micro Soft Robot Mimicking the Cownose Ray

**arXiv ID:** 2601.15349 | [PDF](https://arxiv.org/pdf/2601.15349v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 12. MapViT: A Two-Stage ViT-Based Framework for Real-Time Radio Quality Map Prediction in Dynamic Environments

**arXiv ID:** 2601.15578 | [PDF](https://arxiv.org/pdf/2601.15578v1)

**作者:** Cyril Shih-Huan Hsu `[一作]` (Informatics Institute, University of Amsterdam), Xavier Costa Pérez `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种两阶段的 Vision Transformer 框架（MapViT），用于在移动机器人环境中实时预测环境变化和射频质量图（RQMap），从而提升无线网络自适应与机器人决策效率。

**💡 创新点**

创新点包括：① 将自监督几何基础模型（GFM）与监督微调分离，显著提高数据效率和迁移学习能力；② 利用深度图序列捕捉动态环境，并将其映射为高精度 RQMap；③ 在保持低时延（约1 ms）同时实现比传统射线追踪快数千倍的实时预测。

**🔧 技术方法**

使用技术主要有：Vision Transformer 编码器/解码器、SLAM 与深度图生成、Ray Tracer（无线传播仿真）生成标签、生成对抗增强（数据扩增）、自监督预训练 + 监督微调、与 MLP/CNN 对比实验，评估指标包括 PSNR、MSE、推理时间和多平台（CPU/GPU）效率。

**📊 数据集**

数据集主要由仿真工业仓库生成的深度图序列和对应射频质量图组成，Stage 1 使用 10k 场景（无标签），Stage 2 使用 1k 场景（有射频标签）；下游任务（光照图、温度图、射频图）采用相同几何布局但不同物理模型，OOS 测试通过移除部分货架模拟。

**📈 对比分析**

通过与 MLP、CNN 以及传统 Ray Tracer 在相同参数规模下对比：ViT 在 PSNR 上比 CNN 高约 1.5 dB，推理时间仅约 1 ms，远快于 Ray Tracer 的 12 s（GPU）/4 min（CPU）；在 OOD 场景中仍保持优势；预训练后，仅需少量标签即可实现高精度，显示出优异的数据效率。

**⚠️ 局限性**

局限性包括：① 仅在仿真仓库环境验证，缺乏真实世界大规模实验；② 依赖 SLAM 与深度图输入，需额外硬件与算法支持；③ 对移动设备能耗与网络协同优化细节考虑不足；④ 对多基地站、复杂多径环境的鲁棒性尚待进一步评估。

---

## 13. Ambient Dataloops: Generative Models for Dataset Refinement

**arXiv ID:** 2601.15417 | [PDF](https://arxiv.org/pdf/2601.15417v1)

**作者:** Adrián Rodríguez-Muñoz `[一作]`, Giannis Daras `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `09944146-298c-433e-89df-37255de463d7` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了Ambient Dataloops框架，利用迭代的dataset‑model共进化过程在扩散模型训练中不断对噪声样本进行去噪与数据集改进。

**💡 创新点**

创新点在于将数据集逐步去噪与模型训练同步进行，避免自循环破坏，同时通过Ambient Diffusion的鲁棒学习方法保证在噪声递减过程中模型不失真。

**🔧 技术方法**

采用扩散模型、Ambient Diffusion训练目标、后验采样、噪声水平递减和迭代去噪技术。

**📊 数据集**

主要实验数据集包括CIFAR‑10（90%受损）、Text‑to‑Image（MicroDiffusion使用的多源图像数据集）以及AlphaFold/Protein Data Bank的蛋白质结构数据。

**📈 对比分析**

与质量过滤、未过滤、Ambient Omni等基线比较，单循环即可在CIFAR‑10上提升5‑17% FID，在文本图像与蛋白质设计任务上同样取得显著性能提升。

**⚠️ 局限性**

局限在于需要多轮训练导致计算成本上升，且去噪改进受限于数据本身的噪声极限，若模型不充分训练易产生模式崩溃。

---

## 14. DeltaDorsal: Enhancing Hand Pose Estimation with Dorsal Features in Egocentric Views

**arXiv ID:** 2601.15516 | [PDF](https://arxiv.org/pdf/2601.15516v1)

**作者:** William Huang `[一作]` (University of California, Los Angeles), Yang Zhang `[通讯]` (University of California, Los Angeles)

**通讯引用:** 17572 | [OpenAlex ID](https://openalex.org/A5100354674)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究利用单目 egocentric 视角下手背皮肤变形进行 3D 手姿估计，构建 DeltaDorsal 系统；

**💡 创新点**

创新点在于仅用手背图像与基线姿势对比提取细腻皮肤变形信息，显著缓解自遮挡问题，且模型体积相对较小；

**🔧 技术方法**

技术上采用 Vision Transformer DINOv3 作为密集特征提取器，结合双流对比编码器和回归头，并利用 MANO 先验进行姿态预测；

**📊 数据集**

使用自制的 4K 高分辨率 egocentric 手背数据集，共 170k 帧、17 种手势、12 位受试者；并与 HaMeR、HandOccNet 等公开模型进行对比；

**📈 对比分析**

在自遮挡场景下，DeltaDorsal 的 MPJAE 下降 18%（6.41° vs 6.74°/11.27°），并对皮肤色调不敏感，模型参数约 300M，性能优于两大基线；

**⚠️ 局限性**

局限性包括需要高分辨率相机捕捉手背，且仅在手背可见时有效，快速运动、低光环境等情况仍会影响精度。

---

## 15. Hybrid Vision Transformer_GAN Attribute Neutralizer for Mitigating Bias in Chest X_Ray Diagnosis

**arXiv ID:** 2601.15490 | [PDF](https://arxiv.org/pdf/2601.15490v1)

**作者:** Jobeal Solomon `[一作]` (University of Amsterdam), Seyed Sahand Mohammadi Ziabari `[通讯]` (University of Amsterdam)

**通讯引用:** 238 | [OpenAlex ID](https://openalex.org/A5015160871)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

在 Attribute‑Neutral Framework 中将卷积 U‑Net 编码器替换为 Vision Transformer (DeiT‑S)，并评估其对性别和年龄属性泄漏及诊断性能的影响。

**💡 创新点**

首次证明全局自注意力 Transformer 能够在像素级属性中和去除中显著降低敏感属性泄漏，同时保持诊断准确率，且仅需较少训练周期。

**🔧 技术方法**

使用 Vision Transformer 编码器 + 现有 AttGAN 生成器/判别器框架、AI‑Judge 分类器、ConvNet 诊断模型、SSIM、ROC、混淆矩阵、Grad‑CAM 等评估指标。

**📊 数据集**

主实验基于 NIH ChestX‑ray14 数据集（112k 图像），并在 CheXpert、MIMIC‑CXR、PadChest 上做探索性分析。

**📈 对比分析**

通过 11 级编辑强度（α）下的 AI‑Judge 属性泄漏 AUC、诊断模型的宏观 ROC‑AUC、PR‑AUC、ACC、SEN、SPE、F1 等指标与 U‑Net 基线、FairMixup、Balanced Sampling 等方法比较，结果表明 ViT 使属性泄漏 AUC 下降约10个百分点，诊断宏观 ROC‑AUC 恢复到 95–97% 的基线，并且 worst‑case 子组 AUC 维持在约0.70。

**⚠️ 局限性**

仅在单一公开数据集上评估，计算资源有限，使用固定损失权重，训练仅用单一随机种子；未考察多中心、不同扫描仪或其他敏感属性的泛化，且未验证对临床决策的直接影响。

---

## 16. AdversaRiskQA: An Adversarial Factuality Benchmark for High-Risk Domains

**arXiv ID:** 2601.15511 | [PDF](https://arxiv.org/pdf/2601.15511v1)

**作者:** Adam Szelestey `[一作]` (Eindhoven University of Technology), Songgaojun Deng `[通讯]` (Eindhoven University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并发布了高风险领域（保健、金融、法律）下的对抗性事实性评估基准AdversaRiskQA，包含基本与高级难度两层，并对六款LLM在对抗性提示下的识别和纠正错误信息能力以及长文本事实性进行系统评估。

**💡 创新点**

①首次提供面向高风险领域的对抗性事实性基准；②提出两种自动化评估方法（LLM判别器与搜索增强式长文本事实性评估）；③通过筛选错误回应消除无效输出，揭示规模、架构与对抗性鲁棒性的非线性关系。

**🔧 技术方法**

使用LLM判别器（GPT‑5‑mini）评估对抗性答复，采用搜索增强式事实性评估（SAFE）来测量长文本的事实准确率；在实验中采用标准化提示、温度0、固定最大长度；对结果进行人工抽样验证。

**📊 数据集**

数据集：三域（Health、Finance、Law）各自收集基本与高级两类样本，健康域基于HealthFC，金融域基于教材与学术文献，法律域基于FALQU（Law Stack Exchange）。每个域共约200条基本与200条高级示例。

**📈 对比分析**

比较方法：按域和难度级别计算模型在所有和过滤后（去除无效输出）下的准确率；同时评估长文本F1@K（K=8）。结果显示：在过滤后，Qwen3‑Next‑80B和GPT‑5在各域表现最佳；模型规模与准确率呈非线性提升，难度级别差距随规模增大而缩小；长文本评估未发现注入错误信息与事实输出显著相关。

**⚠️ 局限性**

局限：仅评估三大模型族，样本量有限且仅英文；对抗性提示成功与失败的判定依赖单一LLM判别器；长文本评估仅在一款模型上验证，未覆盖更广模型；安全与隐私问题未彻底解决；缺乏多语言与专家级评估。

---

## 17. Verified polynomial-time reductions in Lean 4: formalizing the complexity of decision-relevant information

**arXiv ID:** 2601.15571 | [PDF](https://arxiv.org/pdf/2601.15571v1)

**作者:** Tristan Simas `[一作]` (McGill University), Tristan Simas `[通讯]` (McGill University)

**通讯引用:** 65 | [OpenAlex ID](https://openalex.org/A5060461525)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

对决策问题中的坐标充分性问题进行复杂度分析，给出完整的可判定性与可最小化问题的多项式层级分类，并提供显式的可判定子类算法。

**💡 创新点**

首次在 Lean 4 中机理化完成了充分性检查的 coNP‑完备性和最小充分集合的 coNP‑完备性以及锚点充分性的 Σ₂⁺‑完备性；提出了编码范式分离（显式状态与简洁编码）以及相应的 ETH 下的指数下界。

**🔧 技术方法**

利用多项式时间归约、塔托尔基（TAUTOLOGY）与存在∀SAT 归约、动态规划、树结构分解；所有证明均在 Lean 4 进行机理化，并给出可计算的多项式界。

**📊 数据集**

未使用实验数据集；研究完全基于理论构造与机器证明。

**📈 对比分析**

由于是理论工作，没有实验对比；提供了多种子类的多项式时间算法，说明在显式状态、可分离效用或树结构下可高效求解；对一般情况给出了 coNP/Σ₂⁺ 复杂度与 ETH 下的指数下界。

**⚠️ 局限性**

局限性在于：1）仅给出最坏情况复杂度；2）未讨论平均/随机实例的复杂度；3）对实际系统的经验验证缺失；4）对可行的近似或 FPT 方案仍未深入。

---

## 18. Panther: Faster and Cheaper Computations with Randomized Numerical Linear Algebra

**arXiv ID:** 2601.15473 | [PDF](https://arxiv.org/pdf/2601.15473v1)

**作者:** Fahd Seddik `[一作]` (University of British Columbia), Yahia Zakaria `[通讯]` (Cairo University)

**通讯引用:** 260 | [OpenAlex ID](https://openalex.org/A5032550126)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了一个名为Panther的PyTorch兼容随机数值线性代数库，用于在深度网络中替换全连接、卷积和注意力层，以显著降低显存和计算成本。

**💡 创新点**

创新点在于将多种成熟的RandNLA算法（sketching、RSVD、CholeskyQR等）统一包装为可直接替换的PyTorch层，并提供自动调优器、CUDA Tensor Core加速以及完整的C++/CUDA后端，填补了理论与生产部署之间的空白。

**🔧 技术方法**

核心技术包括随机投影sketching、随机矩阵分解（RSVD、CQRRPT）、CUDA WMMA张量核心加速、Optuna AutoTuner、PyTorch C++/CUDA扩展。

**📊 数据集**

实验数据集主要有BERT的WikiText（MLM任务）、ResNet‑50 on CIFAR‑10，以及Transformer的MLM任务。

**📈 对比分析**

与原生PyTorch实现对比，采用前向/反向运行时、显存占用等基准；Panther在BERT中可实现高达75%显存压缩，同时保持近似相同的MLM损失；SKLinear、SKConv2D在Tesla T4/P100上实现数倍速度提升，Performer配置在内存受限情况下仍能正常运行。

**⚠️ 局限性**

限制包括：仅支持目前实现的sketching算法，低秩参数设置可能导致精度下降；AutoTuner搜索耗时；CUDA WMMA加速仅在支持Tensor Core的GPU上可用，低版本CUDA或CPU环境下性能提升有限。

---

## 19. Public transport challenges and technology-assisted accessibility for visually impaired elderly residents in urban environments

**arXiv ID:** 2601.15291 | [PDF](https://arxiv.org/pdf/2601.15291v1)

**作者:** Jason Pan `[一作]`, Ben Moews `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了视觉受损老年人使用爱丁堡公共交通的挑战，结合混合方法分析交通网络结构与用户体验。

**💡 创新点**

首次将实时交通数据与人工智能辅助导航相结合，聚焦多重脆弱性群体的可访问性研究。

**🔧 技术方法**

运用了空间统计（NNI、KDE、k‑means聚类）、机器学习和主题分析等技术。

**📊 数据集**

使用Transport for Edinburgh公开API的实时车辆位置信息与访谈录音转写数据。

**📈 对比分析**

通过空间分布指数、密度估计和聚类结果与访谈主题进行对比，显示高度集中且服务不平衡；未与传统算法直接比较，但聚类准确度较高。

**⚠️ 局限性**

数据缺失（实时可达性信息已停止提供）、样本量小、仅研究爱丁堡，无法获得更细粒度的实时服务信息。

---

## 20. Reliability by design: quantifying and eliminating fabrication risk in LLMs. From generative to consultative AI: a comparative analysis in the legal domain and lessons for high-stakes knowledge bases

**arXiv ID:** 2601.15476 | [PDF](https://arxiv.org/pdf/2601.15476v1)

**作者:** Alex Dantart `[一作]` `[通讯]` (Humanizing Internet), Alex Dantart (Humanizing Internet)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对西班牙法律起草任务的量化实验，比较了生成式LLM、基本检索增强生成（RAG）和高级RAG三种范式在文档真实性与可靠性方面的差异。

**💡 创新点**

创新点在于提出并验证了两项专门针对法律领域的可靠性指标（FCR与FFR），并构建了专门的JURIDICO‑FCR数据集，对三种范式的幻觉风险进行系统评估。

**🔧 技术方法**

采用了检索增强生成（RAG）技术、基于多向量检索与重排序的高级RAG架构、以及源校验与自我纠错循环等多项技术手段。

**📊 数据集**

使用了由75个真实西班牙法律起草任务组成的JURIDICO‑FCR数据集，共生成2700条模型输出进行评估。

**📈 对比分析**

通过专家双盲评估计算FCR与FFR，实验结果显示生成式模型的FCR≈27%、高级RAG≈0.05%，从而证明高级RAG在可靠性与可用性方面显著优于其它范式。

**⚠️ 局限性**

局限性包括仅聚焦于西班牙法律体系，任务规模有限；RAG实现未覆盖所有可能的高级优化技术；并且模型迭代速度快，未来版本可能影响实验结论。

---

## 21. GeMM-GAN: A Multimodal Generative Model Conditioned on Histopathology Images and Clinical Descriptions for Gene Expression Profile Generation

**arXiv ID:** 2601.15392 | [PDF](https://arxiv.org/pdf/2601.15392v1)

**作者:** Francesca Pia Panaccione `[一作]` (Politecnico di Milano), Pietro Pinoli `[通讯]` (Politecnico di Milano)

**通讯引用:** 2767 | [OpenAlex ID](https://openalex.org/A5074094001)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种多模态生成对抗网络，利用病理切片图像和临床文本生成逼真的基因表达谱。

**💡 创新点**

首次将图像和文本作为联合条件引入生成模型，并使用FiLM与交叉注意力融合。

**🔧 技术方法**

采用Transformer Encoder、FiLM、双向交叉注意力以及WGAN‑GP生成器与判别器。

**📊 数据集**

在TCGA 19种肿瘤的WSI、临床摘要和RNA‑seq（约1944例）上训练与评估。

**📈 对比分析**

与无条件WGAN‑GP、仅疾病/原发部位条件的WGAN‑GP和CVAE比较，模型在recall、C.MSE、下游分类准确率等指标上均优于基线，提升约11%–16%。

**⚠️ 局限性**

对图像质量和文本简化仍有限，生成的基因表达对高级分类器仍易被识别，且未充分解决跨模态生成的可解释性与对抗鲁棒性。

---

## 22. Do people expect different behavior from large language models acting on their behalf? Evidence from norm elicitations in two canonical economic games

**arXiv ID:** 2601.15312 | [PDF](https://arxiv.org/pdf/2601.15312v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 23. MLP-Enhanced Nonnegative Tensor RESCAL Decomposition for Dynamic Community Detection

**arXiv ID:** 2601.15325 | [PDF](https://arxiv.org/pdf/2601.15325v1)

**作者:** Chaojun Li `[一作]` (Southwest University), Hao Fang `[通讯]` (Southwest University)

**通讯引用:** 2137 | [OpenAlex ID](https://openalex.org/A5101866618)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于MLP增强的非负张量RESCAL分解方法用于动态社群检测

**💡 创新点**

通过在RESCAL分解后引入多层感知机实现分解秩与社群数解耦，提升了社区划分的准确性和鲁棒性

**🔧 技术方法**

非负张量RESCAL分解、MLP映射、模组度最大化、交替优化算法

**📊 数据集**

Chess（1998-2006年国际象棋玩家网络）和Cellphone（10天手机通话网络）

**📈 对比分析**

与TMOGA、Cr-ENMF、DECS、MENCPD等五种先进方法对比，使用模组度评估，结果显示平均模组度提升约4%且在所有时间切片均优于基线

**⚠️ 局限性**

模型仍需处理大规模网络的计算开销，且对超参数（如MLP层数、正则化系数）敏感

---

## 24. AI-Based Culvert-Sewer Inspection

**arXiv ID:** 2601.15366 | [PDF](https://arxiv.org/pdf/2601.15366v1)

**作者:** Christina Thrainer `[一作]` `[通讯]`, Christina Thrainer

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用 AI 自动检测管道缺陷，改进语义分割方法。

**💡 创新点**

创新点在于提出 FORTRESS 结构、动态标签注入策略以及双向原型网络的少样本语义分割。

**🔧 技术方法**

采用深度可分离卷积、Kolmogorov–Arnold 网络、注意力机制、多尺度特征融合、数据增强、动态标签注入和少样本原型网络。

**📊 数据集**

使用 Culvert Sewer Defect Dataset（CSDD），包含 12,230 张带有 9 类缺陷的图像。

**📈 对比分析**

与 E‑FPN 等现有 SOTA 进行比较，在 IoU、FWIoU、F1 等指标上取得更优成绩；少样本模型在极少样本场景下仍保持竞争力。

**⚠️ 局限性**

仍存在数据量有限、类别不平衡、对极少样本鲁棒性不足以及需要更多实证验证的局限。

---

## 25. Closing the Gap on the Sample Complexity of 1-Identification

**arXiv ID:** 2601.15620 | [PDF](https://arxiv.org/pdf/2601.15620v1)

**作者:** Zitian Li `[一作]`, Wang Chi Cheung `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究1-识别问题，给出新的下界并提出一种近似最优的并行分段探索算法PSEEB

**💡 创新点**

创新点在于通过优化式推导得到正实例下的期望拉取次数下界，并设计结合分段采样与并行复制的PSEEB算法，使得上界与下界相差至多多项式对数因子，从而实现对多合格臂情形的几乎最优采样复杂度

**🔧 技术方法**

主要技术包括：
- 对称性与优化问题框架（构造可行的对偶解）
- 交通平衡定理与Kullback–Leibler信息论界
- 经典的UCB、LCB及LIL浓度事件
- 并行分段（bracket）复制与轮询执行策略
- 递归分层预算分配与阈值搜索
- 典型的阈值带问题（Thresholding Bandits）与最佳臂识别（BAI）技术

**📊 数据集**

未涉及外部数据集，全部理论分析基于抽象的概率分布（以方差为1的高斯分布或1子高斯噪声为模型）

**📈 对比分析**

与现有S-TaS、HDoC、lilHDoC、APGAI、SEE等算法对比，PSEEB在正实例上实现了\(O(logK\sum_j\log(1/δ)/Δ_{j,0}^2+H(j)\log^3K)\) 的期望拉取次数，优于SEE的\(O(H(1)\log K/Δ_{0,1}^2)\) 等上界；在负实例上保持与S-TaS相同的\(O(H_1^{neg}\log(H_1^{neg}/δ))\) 复杂度，因而在所有情形下都实现了近似最优性能；相较于过去仅在单合格臂或已知阈值差距的情形下给出的结果，PSEEB克服了多合格臂的缺口。

**⚠️ 局限性**

局限性包括：
- 上界与下界之间仍存在多项式对数因子，无法完全消除
- 仅在单一噪声模型（1-sub-Gaussian或方差为1的高斯）下给出理论证明
- 对于极端阈值接近最高平均奖励的情况，所需的\(H(j)\) 可能较大
- 计算成本主要在多复制与轮询，实际实现需要考虑时间与空间开销

---

## 26. DuFal: Dual-Frequency-Aware Learning for High-Fidelity Extremely Sparse-view CBCT Reconstruction

**arXiv ID:** 2601.15416 | [PDF](https://arxiv.org/pdf/2601.15416v1)

**作者:** Cuong Tran Van `[一作]` (FPT Software AI Center), Ngan Le `[通讯]` (University of Arkansas)

**通讯引用:** 6699 | [OpenAlex ID](https://openalex.org/A5108408962)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种DuFal框架，利用频域与空间域双编码实现极少视角的CBCT重建，直接从原始X射线投影得到高质量三维CT体积。

**💡 创新点**

创新点包括：1）HiLocFFNO块，将全局高频增强与局部补丁高频处理融合，既捕获大尺度结构又保留细节；2）Spectral‑Channel Factorization（SCF）显著压缩FNO参数；3）Cross‑Attention Frequency Fusion（CAFF）在频域对空间与频域特征进行跨模态融合；4）将上述模块无缝嵌入已存在的Implicit Neural Representation（DIF）流水线。

**🔧 技术方法**

核心技术包括：傅里叶神经算子（FNO）、高频增强分支（gHiF 与 lHiF）、SCF 权重分解、频域交叉注意力（CAFF）、Intesity Field 解码器（DIF）以及传统的CNN空间编码器。

**📊 数据集**

使用公开的LUNA16（胸部CT）和ToothFairy（牙科CBCT）数据集进行实验，模拟6、8、10视角投影进行训练与评估。

**📈 对比分析**

与FDK、SART、NAF、NeRP、FreeSeed、DIF‑Net、DIF‑Gaussian等方法对比，DuFal在所有视角下均取得最高PSNR/SSIM（如10视角下PSNR 29.41/30.51、SSIM 87.67/89.49），ROI‑加权指标亦领跑；同时推理速度仅比DIF‑Gaussian略慢，但显著快于DIF‑Net，参数量相比未分解的FNO减少约82%。

**⚠️ 局限性**

局限性：仅在静态CT数据上验证，缺乏对运动伪影的评估；对其他成像模态（MRI、PET等）的通用性尚未验证；训练时间相对较长，需大规模GPU资源；对极低投影数量（≤5视角）效果尚待进一步探索。

---

## 27. LLM or Human? Perceptions of Trust and Information Quality in Research Summaries

**arXiv ID:** 2601.15556 | [PDF](https://arxiv.org/pdf/2601.15556v1)

**作者:** Nil-Jana Akpinar `[一作]` (Microsoft), Vanessa Murdock `[通讯]` (Amazon AWS AI)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过一项混合方法调查实验，研究读者对人类撰写、LLM自动生成和LLM编辑的科研摘要的识别能力与评价偏好。

**💡 创新点**

创新点在于首次系统量化了读者对LLM参与度的感知如何影响对摘要质量与可信度的评估，并揭示了三类不同的读者态度（披露倡导者、务实怀疑者、乐观拥护者）。

**🔧 技术方法**

采用了线性混合效应模型、主题分析和主成分聚类等统计与文本分析技术，结合LLM‑in‑the‑loop 的定性编码。

**📊 数据集**

数据集由150篇2019‑2022年发表的arXiv机器学习论文构成，分别生成三种摘要版本，并在69名具有ML专业背景的受访者中进行评估。

**📈 对比分析**

通过对比可信度、清晰度、完整性等五个维度的评分和抽取偏好，发现披露LLM编辑后参与度被显著提升，且整体评价优于纯人工或纯LLM生成；LLM编辑摘要在受访者中获得最高选择率。

**⚠️ 局限性**

局限性包括受访者主要来自ML领域、样本量有限、仅聚焦摘要而非全文，且实验环境中的LLM模型与真实学术写作场景可能存在差异。

---

## 28. From Quotes to Concepts: Axial Coding of Political Debates with Ensemble LMs

**arXiv ID:** 2601.15338 | [PDF](https://arxiv.org/pdf/2601.15338v1)

**作者:** Angelina Parfenova `[一作]` (Technical University of Munich), Juergen Pfeffer `[通讯]` (Technical University of Munich)

**通讯引用:** 8112 | [OpenAlex ID](https://openalex.org/A5030571651)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过集成多模型LLM的开放编码与轴向编码，对荷兰议会辩论文本进行自动化编码，生成层次化的概念结构。

**💡 创新点**

首次将轴向编码与LLM结合，提供可扩展的自动轴向编码流程，并提出双重（内在与外在）评价框架。

**🔧 技术方法**

使用LoRA微调的多模型LLM进行开放编码，随后采用密度聚类（DBSCAN/HDBSCAN）或直接LLM聚类进行轴向编码，并利用SBERT/ MiniLM 等嵌入与 ROUGE、余弦相似度、BERTScore 等指标进行评估。

**📊 数据集**

采用约5,000条荷兰议会辩论发言的语料库，包含人类主题标签、LLM生成的开放编码以及轴向类别。

**📈 对比分析**

通过内在指标（覆盖率、简洁度、连贯性、创新度、JSD）与外在指标（ROUGE‑L、余弦相似度、BERTScore）进行比较，聚类方法在覆盖率上领先但标签简洁度和外在匹配度较低，而直接LLM聚类在标签质量与外在匹配上更优但覆盖率较低。

**⚠️ 局限性**

研究仅在单一议会语料上验证，模型与聚类算法范围有限，且评估依赖相对完整的金标准，未充分验证对其他质性文本领域的泛化能力。

---

## 29. Multi-Targeted Graph Backdoor Attack

**arXiv ID:** 2601.15474 | [PDF](https://arxiv.org/pdf/2601.15474v1)

**作者:** Md Nabi Newaz Khan `[一作]` (University of Rhode Island), Yu Bi `[通讯]` (University of Rhode Island)

**通讯引用:** 373 | [OpenAlex ID](https://openalex.org/A5074397437)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出针对图分类任务的多目标后门攻击框架。

**💡 创新点**

创新点是将子图注入而非替换，使多触发器能共存且干扰最小。

**🔧 技术方法**

采用子图注入、Erdos‑Renyi 触发器生成、随机注入点、图神经网络训练等技术。

**📊 数据集**

使用五个数据集：CIFAR‑10、MNIST、ENZYMES、Reddit‑Multi‑12k、Reddit‑Multi‑5k。

**📈 对比分析**

与传统子图替换攻击对比，在同一设置下 ASR 均超过 99%，清洁准确率下降不到 3%，表现优异。

**⚠️ 局限性**

局限性包括对触发器大小、密度等参数高度敏感；当触发器数量极大或数据集极小时，干扰略增，效果略受限制。

---

## 30. Benchmarking LLMs for Pairwise Causal Discovery in Biomedical and Multi-Domain Contexts

**arXiv ID:** 2601.15479 | [PDF](https://arxiv.org/pdf/2601.15479v1)

**作者:** Sydney Anuyah `[一作]` (Indiana University), Sunandan Chakraborty `[通讯]` (Indiana University)

**通讯引用:** 195 | [OpenAlex ID](https://openalex.org/A5111311082)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文对13种开源大语言模型在医学与多领域文本的成对因果发现（检测与提取）任务中进行系统性基准评测。

**💡 创新点**

创新点在于提出统一评估框架，覆盖12个跨域数据集，设计多层级提示策略（零样本、少样本、CoT、Hybrid CoT+FICL、Least‑to‑Most、ReAct），并引入SentenceLM余弦匹配与Hungarian匹配的自适应评分方法。

**🔧 技术方法**

采用零样本、少样本、Chain‑of‑Thought、Hybrid CoT+FICL、Least‑to‑Most、ReAct等提示技术，并使用SentenceLM余弦相似度与Hungarian匹配评估提取结果。

**📊 数据集**

使用12个公开数据集：CausalNet、CausalProbe、SemEval2010Task8、MedCaus、CauseNet、FinCausal、CausalBench、CRASS、Coling22、ECI‑B、PubMed、COPA，涵盖医学、金融、新闻和多领域文本。

**📈 对比分析**

对13个开源模型进行比较，检测最高得分为49.57%（DeepSeek‑R1‑Distill‑Llama‑70B），提取最高得分为47.12%（Qwen2.5‑Coder‑32B‑Instruct）；在显式单句任务上表现最佳，但在隐式、跨句和多对任务上性能明显下降。

**⚠️ 局限性**

主要局限包括低召回率、对隐式与跨句因果关系识别能力不足、提示与阈值敏感、数据采样与标注主观性，以及仅评估英文开源模型，未涉及专有或多语言情形。

---

## 31. ManuRAG: Multi-modal Retrieval Augmented Generation for Manufacturing Question Answering

**arXiv ID:** 2601.15434 | [PDF](https://arxiv.org/pdf/2601.15434v1)

**作者:** Yunqing Li `[一作]` (Lenovo), Jianbang Zhang `[通讯]` (Lenovo)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套多模态检索增强生成（ManuRAG）系统，用于制造业问答（QA）。

**💡 创新点**

创新点在于四种检索策略的对比，并提出在检索前将图像、公式、表格等模态转换为文本描述后统一嵌入文本空间（ManuRAG_4），显著提升答案准确性与上下文相关性。

**🔧 技术方法**

采用 PDF 提取管道（LayoutLMv3、YOLOv8+UniMERNet、PaddleOCR）、CLIP 与多模态嵌入模型、LLM GPT‑4O、Chain‑of‑Thought 及 LlamaIndex 等技术实现多模态数据抽取、索引、检索与答案生成。

**📊 数据集**

使用《Fundamentals of Modern Manufacturing》教材的 1,515 对 QA 数据集（数学题 309 题、选择题 471 题、综述题 735 题），未将该数据用于检索索引。

**📈 对比分析**

与 GPT‑4O、GPT‑4O+CoT、传统 RAG、RAG_hq 以及 ManuRAG 的三种变体对比，ManuRAG_4 在 Factual Correctness、Semantic Similarity、ROUGE 以及 Context Precision/Recall 等指标上均优于其他模型，尤其在 MathQ 与 MCQ 上表现最突出。

**⚠️ 局限性**

局限性包括：图像转文本会导致部分视觉细节丢失，无法完美处理 CAD、技术图纸等精细图像；对视觉细节的精准解析仍待改进；数据集范围有限，缺乏更丰富的模态（视频、传感器日志等）。

---

## 32. RDumb++: Drift-Aware Continual Test-Time Adaptation

**arXiv ID:** 2601.15544 | [PDF](https://arxiv.org/pdf/2601.15544v1)

**作者:** Himanshu Mishra `[一作]` `[通讯]` (University of British Columbia), Himanshu Mishra (University of British Columbia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种针对长时间、不断变化测试流的连续测试时自适应（CTTA）方法 RDumb++，通过在原 RDumb 基础上加入熵基和 KL 基漂移检测与自适应重置策略，实现了对分布漂移的及时响应并保持模型稳定性。

**💡 创新点**

创新点在于：① 用统计显著性检验（z‑score）对模型预测熵与 KL 散度进行漂移检测；② 根据漂移强度灵活选择全重置或软重置；③ 在极长的 100 万步测试流上验证，显著避免了模型崩溃。

**🔧 技术方法**

技术包括：熵与 KL 的指数滑动均值/方差估计、标准化漂移评分、阈值触发重置、软重置线性插值恢复参数、以及基于 ResNet‑50 的批归一化参数自适应。

**📊 数据集**

数据集：CCC‑Medium（Continually Changing Corruptions）基准，包含 1M 张逐步变化的图像腐败样本，提供三种腐败速度（1000、2000、5000 步）和三种随机种子。

**📈 对比分析**

与基线、Tent、EATA、RDumb 等方法对比，RDumb++ 在三种速度下平均提升 3–5% 的准确率，尤其是 EntropyFull 与 KLFull 变体，且在长时间测试流中几乎没有崩溃事件。

**⚠️ 局限性**

局限性包括：需手动调节漂移阈值 k 和软重置强度 λ，适应不同数据集时可能不稳；熵/KL 统计假设可靠，若存在严重类别不平衡或对抗扰动可能失效；未引入长期记忆或元学习机制以进一步提升复用性。

---

## 33. DeepSurvey-Bench: Evaluating Academic Value of Automatically Generated Scientific Survey

**arXiv ID:** 2601.15307 | [PDF](https://arxiv.org/pdf/2601.15307v1)

**作者:** Guo-Biao Zhang `[一作]` (Beijing Institute of Technology), Xian-Ling Mao `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 2860 | [OpenAlex ID](https://openalex.org/A5017626590)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出DeepSurvey-Bench基准，构建含学术价值标注的自动综述评估数据集，并同时评估综述的表面质量与学术价值；

**💡 创新点**

引入三维学术价值框架（信息价值、学术传播价值、研究指导价值），将其拆解为七个可量化指标，并将人类专家评估与LLM-as-a-judge结合以提升评估可靠性；

**🔧 技术方法**

采用LLM-as-a-judge评审、RAG与多代理生成流程、自动指标（ROUGE、BLEU、Citation Recall/Precision）以及GPT‑5.1评审器，并使用LLM与人工标注相结合构建数据集；

**📊 数据集**

收集2022–2025年发表的含“survey”“review”等关键词的论文，最终构建163篇高质量综述数据集，涵盖计算机科学、物理、天文学、电子系统等八个主题；

**📈 对比分析**

使用AutoSurvey、SurveyX、SurveyForge三种自动综述生成模型，分别在GPT‑4o、Claude‑3.5‑Haiku、DeepSeek‑v3评审器下评估；学术价值指标显著区分模型，GPT‑4o平均得分最高，并与人工评价高度相关；

**⚠️ 局限性**

缺乏闭环验证将评估结果用于模型改进；LLM评估成本高，限制了评估规模。

---

## 34. Elsewise: Authoring AI-Based Interactive Narrative with Possibility Space Visualization

**arXiv ID:** 2601.15295 | [PDF](https://arxiv.org/pdf/2601.15295v1)

**作者:** Yi Wang `[一作]` (Midjourney), Max Kreminski `[通讯]` (Midjourney)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种名为 Bundled Storyline 的可视化框架，帮助 AI‑based 互动叙事作者通过多维度可视化更好地感知、控制和探索玩家体验的故事空间。

**💡 创新点**

创新点在于：①将玩家实际体验的故事轨迹映射为“bundled”节点，形成可聚合的分支图；②支持作者定义、自动提取及交叉维度的故事维度；③通过交互式 BSV 画布实现多视图交叉筛选与时间切片，从而提升作者对空间的整体感知；④在传统 IN 工具中首次将 LLM 生成的模拟游玩数据与可视化结合，实现快速迭代。

**🔧 技术方法**

使用的技术包括：Claude 3.5‑Sonnet 进行概念诱导与维度标签分类；Python Flask 作为后端；ReactFlow 渲染交互式可视化；通过 LLM 驱动的玩家代理模型（四类角色）生成模拟游玩；以及自定义规则系统实现游戏逻辑。

**📊 数据集**

实验数据主要来自：①每个参与者在系统中生成的模拟游玩序列（每轮最多 5 轮，最大 3 轮迭代）；②12 名作者在两种工具（Elsewise 与基线）下完成的任务产生的故事及 BSV 视图；③对应的问卷与访谈记录。

**📈 对比分析**

比较方法为对照实验：在相同主题与限制下，作者先后使用 Elsewise 与基线工具完成任务；通过 NASA‑TLX、Creativity Support Index 及主题对齐/控制/预期/玩家代理等 Likert 量表评估，并统计 BSV 数量、规则数、模拟轮数等指标。结果显示 Elsewise 在预期、探索、表达性上均显著优于基线，平均 CSIS 评分提升约 20%；但在心理负荷上略高，且 2D BSV 的可视化复杂度被多名参与者指出。

**⚠️ 局限性**

局限性包括：仅使用 LLM 模拟玩家，缺乏真实玩家数据；故事规模受限（单场景、≤4 人物），不具备真实项目的可扩展性；概念诱导实现简化，未对不同聚类策略或维度可扩展性进行技术评估；2D BSV 的视觉复杂度导致部分作者使用不便；未深入探讨如何将方法迁移至非游戏 AI 交互场景。

---

## 35. Multi-Input Ciphertext Multiplication for Homomorphic Encryption

**arXiv ID:** 2601.15401 | [PDF](https://arxiv.org/pdf/2601.15401v1)

**作者:** Sajjad Akherati `[一作]` (Ohio State University), Xinmiao Zhang `[通讯]` (Ohio State University)

**通讯引用:** 3398 | [OpenAlex ID](https://openalex.org/A5063673084)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

在 CKKS 同态加密方案中，提出并实现了改进的三输入密文乘法器，并将其推广为多输入密文乘法框架，支持任意数量密文的高效乘法。

**💡 创新点**

创新点包括：
1) 重新构造三输入乘法的重线性化与缩放过程，显著减少（I）NTT 与中间模数转换的数量；
2) 引入新的评估密钥以重线性化多阶多项式项，避免噪声指数级放大；
3) 设计了多重缩放（multi‑rescaling）算法，实现 μ 次连续缩放仅需与单次缩放相同的 (I)NTT 复杂度；
4) 提供输入划分与重排策略，最大化多重缩放的组合效益，进一步降低硬件面积与延迟。

**🔧 技术方法**

采用的技术与工具包括：CKKS 同态加密（RNS 表示、NTT/INTT、Karatsuba 分解、Barrett 降约、BConv 基变换）、全流水线 2‑并行 (I)NTT 单元、评估密钥管理、硬件级（I）NTT 与加法/乘法模块实现，以及对 Rescaling、Relinearization 的硬件优化。

**📊 数据集**

论文未使用特定机器学习或金融等应用数据集，而是在硬件实现层面使用 GlobalFoundries 22FDX 工艺进行综合测试；性能评估基于面积（XOR 数）、时钟周期数、以及所需内存（MB）等硬件指标。

**📈 对比分析**

比较方法：
- 与先前的三输入乘法器进行面积与延迟对比；得到逻辑面积约 15% 降低，时延约 50% 缩短。
- 与使用二叉树结构的 2‑输入乘法器对比；在 4–12 个输入范围内，面积平均下降约 32%，时延平均下降约 45%。
- 通过硬件综合结果（面积、内存、时延）和公式化的复杂度表，展示多重缩放在减少 (I)NTT 数量、内存占用与功耗方面的优势。

**⚠️ 局限性**

局限性：
1) 仅针对 CKKS 同态加密方案，无法直接迁移到 BFV、BGV 等方案；
2) 当输入数量为 2 的幂时，二叉树已是最优，结合多重缩放的收益有限；
3) 评估密钥数量随输入量增长，存储需求仍显著，尤其在大规模乘法场景；
4) 论文未给出对噪声上限的量化分析，只提到“噪声上界低于传统方法”，缺乏严格的安全/误差证明；
5) 仅在 22FDX 工艺下进行综合评估，缺少跨工艺或跨 FPGA/ASIC 平台的通用性验证。

---

## 36. Controllable Layered Image Generation for Real-World Editing

**arXiv ID:** 2601.15507 | [PDF](https://arxiv.org/pdf/2601.15507v1)

**作者:** Jinrui Yang `[一作]` (University of California Santa Cruz), Yuyin Zhou `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了统一的层式图像生成与编辑框架，支持背景条件前景生成、前景条件背景生成和文本到全层生成三种模式，并实现透明前景层带有阴影、反射等真实视觉效果。

**💡 创新点**

创新点包括：1）单一模型即可完成三种生成模式并保持层间一致性；2）首次公开包含物理视觉效果的高质量层级数据集；3）构建了专门的层编辑基准测试，统一评估。

**🔧 技术方法**

采用扩散变压器（DiT）作为核心网络，结合T5文本编码器和四种嵌入（类型、IO、位置、时间步）实现多模态条件；使用噪声匹配训练策略，统一优化三种生成任务。

**📊 数据集**

使用公开数据源（MULAN、COCO 2017、SOBA）构建了约48K张图像三元组数据集（背景、前景带视觉效果、合成图），并通过自研分解模型、过滤与标注得到高质量样本；另外发布242张用于评测的层编辑基准。

**📈 对比分析**

与主流图像生成/编辑模型（FLUX、Qwen-Image、GPT-Image）以及专家模式LayerDiffuse比较，LASAGNA在Fidelity、语义一致性（CLIP-FID/FID）和GPT-4o指令遵循/身份保持评分上均优于对手；在层编辑任务中，带视觉效果的层编辑表现最优。

**⚠️ 局限性**

目前仅支持单对象的层生成和编辑，无法一次性生成多交互对象的层级表示，也缺乏对动态场景行为的细粒度控制。

---

## 37. VegaChat: A Robust Framework for LLM-Based Chart Generation and Assessment

**arXiv ID:** 2601.15385 | [PDF](https://arxiv.org/pdf/2601.15385v1)

**作者:** Marko Hostnik `[一作]` (JetBrains), Artem Trofimov `[通讯]` (JetBrains)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 VegaChat，一个基于 LLM 的系统，可将自然语言请求转换为 Vega‑Lite 可视化规范，并同时提出了 Spec Score 与 Vision Score 两种评估指标，用于对生成的可视化进行自动化验证与比较。

**💡 创新点**

创新点在于：①首次为基于声明式语法的可视化生成提出可解释的、无 LLM 的 Spec Score；②设计 Vision Score，利用多模态 LLM 对图像进行分维度评分，兼顾图表相似度与提示合规性；③将上述两种指标与现有基准对比，验证其与人工评估的高度一致性。

**🔧 技术方法**

使用技术包括：GPT‑4o‑mini 进行 Vega‑Lite 代码生成与错误修正、基于规则的 Spec Score 计算、基于 GPT‑4o 的 Vision Score 评估、请求分析器、图表推荐器、错误反馈循环与多轮对话支持。

**📊 数据集**

主要数据集为 NLV Corpus（814 句，30 个可视化）和 ChartLLM 注释子集（48 组真实提示），此外还使用 MatPlotBench、SEVQ 等公开基准进行对照。

**📈 对比分析**

在 NLV 数据集上，VegaChat 的 Spec Score 与 Vision Score 均超过 85%，ECR 接近 0%；在 ChartLLM 上得分约 55%–60%，相较于 LIDA、CoML4VIS 有明显提升；所有指标与人工评估的相关系数均在 0.65–0.71 之间，证明评估方法具有良好可靠性。

**⚠️ 局限性**

局限性包括：Spec Score 仅适用于 Vega‑Lite，受模式变更与 LLM 生成错误影响；Vision Score 对高维度交互式图表无支持；仅处理单表场景，缺乏多表查询与交互式交互评估；多轮对话评估仍不够自然，且 LLM 在复杂现实案例中易出现幻觉。

---

## 38. Designing Persuasive Social Robots for Health Behavior Change: A Systematic Review of Behavior Change Strategies and Evaluation Methods

**arXiv ID:** 2601.15309 | [PDF](https://arxiv.org/pdf/2601.15309v1)

**作者:** Jiaxin Xu `[一作]` (Eindhoven University of Technology), Wijnand A. IJsselsteijn `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 14864 | [OpenAlex ID](https://openalex.org/A5006886273)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统综述方法，对39篇涉及社交机器人促进健康行为改变的实证研究进行汇总与归纳，构建了四大行为改变策略类别并分析了当前评估方法与研究局限；

**💡 创新点**

创新点在于首次系统化绘制社交机器人在健康行为改变中的干预策略与评估实践，并揭示了机器人特有的社会影响和情感纽带设计空间；

**🔧 技术方法**

采用PRISMA系统综述流程、主题分析（BCT与PSD框架为预设主题）以及ATLAS.ti软件进行编码；

**📊 数据集**

数据集来源为Web of Science、Scopus、ACM Digital Library以及手工检索，最终筛选出39篇实证研究；

**📈 对比分析**

文章未对单个策略的效果进行比较，只在宏观层面总结策略出现频率与评估方法的使用趋势，未给出定量性能指标；

**⚠️ 局限性**

局限包括：未评估各策略的实际有效性；筛选仅由单一作者完成，存在选择偏差；以及搜索覆盖可能遗漏部分相关研究。

---

## 39. SplatBus: A Gaussian Splatting Viewer Framework via GPU Interprocess Communication

**arXiv ID:** 2601.15431 | [PDF](https://arxiv.org/pdf/2601.15431v1)

**作者:** Yinghan Xu `[一作]` (Trinity College Dublin), John Dingliana `[通讯]` (Trinity College Dublin)

**通讯引用:** 1043 | [OpenAlex ID](https://openalex.org/A5026997782)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计并实现了一个基于Python的GPU渲染服务器与多种可视化客户端（Unity插件、OpenGL视图器等）通过NVIDIA CUDA IPC解耦的3D高斯弹涂（Gaussian Splatting）渲染框架，支持在不修改核心光栅化器的前提下实现实时渲染与深度感知混合。

**💡 创新点**

核心创新点在于：①将光栅化器抽象为独立的渲染服务器，客户端仅通过IPC共享GPU帧缓冲实现零拷贝；②在单一通道内完成颜色与深度缓冲共享与同步；③支持与传统mesh渲染管线深度感知混合，实现Hybrid渲染；④提供统一的JSON+TCP消息协议，便于多平台集成。

**🔧 技术方法**

采用技术包括：Python+CUDA框架、CUDA IPC（共享内存与事件句柄）、OpenGL–CUDA互操作、Unity Native Plugin、JSON+TCP通信、深度线性化处理、深度感知混合渲染、基于ImGui的可视化调试。

**📊 数据集**

使用公开的3D高斯弹涂训练数据集（如NeRF、MMLPHuman等），以及从原始3DGS重建得到的场景与Gaussian头像模型进行实验；未对传统数据集做特殊预处理，直接使用已有的Gaussian渲染结果。

**📈 对比分析**

通过与原始单机3DGS viewer及Unity/Unreal等游戏引擎专用集成进行对比，展示了：①渲染延迟显著降低（几毫秒级）且帧率保持在实时（30~60fps）范围；②深度感知混合效果优于传统单一渲染；③在多客户端并发下保持稳定，CPU占用率低于10%。

**⚠️ 局限性**

局限性：①仅适用于支持CUDA IPC的NVIDIA GPU，无法跨平台或多机分布式；②当前实现仅支持颜色+深度缓冲，无法直接处理更复杂的光照、阴影或动态神经网络前置处理；③对4D动态Gaussian渲染的支持尚未完善，需进一步标准化4D数据格式；④深度感知混合算法基于线性深度，可能在极端视角下产生误差。

---

## 40. BanditLP: Large-Scale Stochastic Optimization for Personalized Recommendations

**arXiv ID:** 2601.15552 | [PDF](https://arxiv.org/pdf/2601.15552v1)

**作者:** Phuc Nguyen `[一作]` (LinkedIn), Changshuai Wei `[通讯]` (LinkedIn)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 BanditLP，结合神经Thompson采样和大规模线性规划，解决多利益相关者上下文Bandit问题，并在 LinkedIn 电子邮件营销系统中实现。

**💡 创新点**

创新点：① 把探索-利用的神经 TS 与约束优化的 LP 分离，既能捕获复杂非线性关系，又能满足多级硬性约束；② 设计可扩展到数十亿决策变量的 LP 求解器（DuaLip）和可插拔的神经网络后验估计；③ 在真实系统中通过 A/B 测试验证长周期收益提升与订阅率下降。

**🔧 技术方法**

技术：神经网络后验采样（Laplace 近似）、神经 Thompson 采样、线性规划求解（DuaLip）、概率校准（单调回归）、大规模并行 LP 迭代、在线 A/B 实验与数据分流。

**📊 数据集**

数据集：1) Open Bandit Dataset（OIBD）公开日志，采用 KNN 填充得到稠密点击矩阵；2) 合成数据集（随机生成用户、物品特征、奖励与成本函数）用来控制约束数量与提供者数量；3) LinkedIn 真实邮件营销日志用于在线实验。

**📈 对比分析**

比较方法：与无约束的 NNTS、基于 LinUCB 的 LP（LinUCB-LP）和无探索的 NN-LP 对比。实验显示 BanditLP 在保持约束不违背（global、provider 级约束几乎为 0）同时，累计奖励最高；在线 A/B 测试中收入提升 3.08% 并显著降低取消订阅率 1.51%。

**⚠️ 局限性**

局限：① 需要先训练可靠的神经预测模型，若校准不佳会影响 LP 结果；② LP 求解时间与 γ 参数调优仍是性能瓶颈；③ 目前假设营销单元之间无交互效应，未来需考虑交互、实时学习和更丰富的约束形式。

---

## 41. Uncovering Latent Bias in LLM-Based Emergency Department Triage Through Proxy Variables

**arXiv ID:** 2601.15306 | [PDF](https://arxiv.org/pdf/2601.15306v1)

**作者:** Ethan Zhang `[一作]` `[通讯]` (Palo Alto High School), Ethan Zhang (Palo Alto High School)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了大型语言模型在急诊科分诊中的隐性偏差，构建32个代理变量并通过正负修饰词对比评估模型对急诊严重度（ESI）的预测变化。

**💡 创新点**

提出了基于代理变量的隐性偏差评估框架，系统地利用正负修饰词揭示模型对同一代理变量在不同极性下的偏差，发现了既有极性依赖的歧视性偏差，也有极性无关的语义理解缺失。

**🔧 技术方法**

采用 gpt‑4o‑mini 进行提示式推理，利用统计分析（均值偏移、95% 置信区间）对不同修饰词下的 ESI 预测进行定量比较。

**📊 数据集**

使用公开的 MIMIC‑IV‑ED Demo v2.2 与 MIMIC‑IV Demo v2.2 以及受限访问的完整 MIMIC‑IV‑ED v2.2 与 MIMIC‑IV v2.2 数据集，包含 220 条开放记录和更大规模的受限数据。

**📈 对比分析**

通过比较默认情境、正修饰词和负修饰词下的 ESI 评分差异来评估偏差；结果显示 3/4 代理变量在统计上显著偏移，且部分代理变量导致跨种族的严重度误判。

**⚠️ 局限性**

局限性包括仅评估单一 LLM（gpt‑4o‑mini），未验证偏差缓解策略；代理变量选择受文献和 ChatGPT 提示限制；结果可能受提示设计和数据分布偏差影响；未考察实时部署与临床验证。

---

## 42. DevPrompt: Deviation-Based Prompt Learning for One-Normal ShotImage Anomaly Detection

**arXiv ID:** 2601.15453 | [PDF](https://arxiv.org/pdf/2601.15453v1)

**作者:** Morteza Poudineh `[一作]` (Concordia University), Marc Lalonde `[通讯]` (Computer Research Institute of Montreal)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种偏差引导的提示学习框架，用于在仅有少量正常样本的前提下实现图像异常检测与定位。

**💡 创新点**

创新点包括：①引入可学习的上下文提示向量，让正常与异常提示共享语义基底并保留类别特征；②使用基于高斯偏差的损失将异常分数转化为统计显著性；③结合Top‑K多实例学习对异常补丁进行聚合，提升局部缺陷识别；④将视觉‑语言对齐与统计分离统一到同一框架。

**🔧 技术方法**

采用CLIP视觉‑语言双编码器、可学习提示向量、偏差得分公式（|s‑μ|/σ）、Top‑K多实例学习（MIL）和高斯统计模型进行训练与推理。

**📊 数据集**

使用工业缺陷数据集MVTecAD和VISA进行实验，均采用1-shot正样本训练，测试包含正常与异常样本。

**📈 对比分析**

与PromptAD、WinCLIP、PatchCore等基线比较，像素级AUROC平均提升约1–2%，在局部缺陷类（如裂纹、污染）表现尤为显著；在全局或纹理变异性大的缺陷上提升有限。

**⚠️ 局限性**

局限性：对全局性或分布广泛的缺陷效果不佳；模型对超参数（偏差系数λ、Top‑K比例、阈值a）较为敏感；目前仅处理静态图像，未扩展到视频或动态异常检测。

---

## 43. Beyond Fixed Psychological Personas: State Beats Trait, but Language Models are State-Blind

**arXiv ID:** 2601.15395 | [PDF](https://arxiv.org/pdf/2601.15395v1)

**作者:** Tamunotonye Harry `[一作]` (University of Vermont), Joseph Near `[通讯]` (University of Vermont)

**通讯引用:** 697 | [OpenAlex ID](https://openalex.org/A5061707651)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并发布了Chameleon数据集，记录同一用户在不同Reddit子社区中的心理剖面，并利用此数据评估LLM与奖励模型对心理状态的感知与适应能力。

**💡 创新点**

创新点在于首次将Latent State‑Trait理论应用于NLP，实现了对心理变异的“状态‑特质”分解，并揭示大约74%的心理差异来源于上下文状态而非个体特质；同时发现LLM生成与奖励模型评估均存在状态相关偏差。

**🔧 技术方法**

技术主要包括：两种心理剖面提取方法（SEANCE词汇特征与LangExtract基于LLM的语义抽取）、LLM评估对心理量表题项的评分、z‑归一化与融合、ICC方差分解、线性混合效应检验、k‑means聚类构建心理状态原型、以及对多种LLM与奖励模型的生成与评分实验。

**📊 数据集**

数据集：5,001篇来自1,667位Reddit用户、645个子社区的帖子，覆盖26个心理维度（包括Big Five、Schwartz价值、SDT动机与DOSPERT风险），每位用户至少三篇来自不同社区的帖子。

**📈 对比分析**

对比方法包括：(1) 计算不同提取方法的ICC以评估状态/特质比例，结果均显示74%状态方差；(2) 交叉方法一致性检验、混合效应假设检验以及聚类原型分析验证心理剖面有效性；(3) 对三大LLM在七种心理状态条件下的回答相似度测评，表明LLM对状态的敏感度低；(4) 对三种奖励模型在相同回答下的评分差异分析，发现评分不一致且方向相反。性能上，LLM表现出状态不敏感，奖励模型则对用户状态产生相互冲突的偏好。

**⚠️ 局限性**

局限性包括：仅基于文本表达的心理剖面可能偏离真实内在状态；子社区作为上下文的定义可能混杂主题、受众和社区规范；数据来源仅为2006‑2016年Reddit帖子，缺乏跨平台或跨时段验证；LLM提取方法可能带来模型偏差；单一三次观测设计导致ICC估计受限，真实状态方差可能更高。

---

## 44. Learning from Synthetic Data: Limitations of ERM

**arXiv ID:** 2601.15468 | [PDF](https://arxiv.org/pdf/2601.15468v1)

**作者:** Kareem Amin `[一作]` (Google Research), Sergei Vassilvitskii `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文研究在训练数据被LLM生成的合成样本污染的情形下，传统学习方法（如经验风险最小化）在均值估计和PAC学习中的表现，并提出改进策略。

**💡 创新点**

创新点包括：1）对均值估计中均值的经验估计在任意污染程度下的方差进行精确解析，证明其不再是最小方差无偏估计；2）揭示在PAC设定中重复使用ERM会导致泛化误差停滞；3）设计两类全局算法（简单随机化算法和基于PU学习的迭代算法）能够在任何污染率下实现收敛到零泛化误差。

**🔧 技术方法**

使用的技术主要有：无偏估计与方差分析（利用Gamma函数与Gautschi不等式推导均值估计方差）；随机游走理论证明ERME的失效；正负无标签学习（PU学习）框架与XOR类映射；误差上界与Chernoff界等概率工具；以及对VC维度的传统PAC理论。

**📊 数据集**

论文并未在具体公开数据集上进行实验，而是构造了理论模型：自然数据来自未知分布，合成数据由前一轮模型生成，混合比例由污染率α控制。

**📈 对比分析**

比较方法主要是理论性能对比：均值估计中，均值估计的方差随α变化，从O(1/t)到π²/6的常数；在PAC学习中，重复ERM在α>½时泛化误差保持在常数，新的算法在所有α下可实现O(1/√t)甚至O(1/(nt))的收敛速度。

**⚠️ 局限性**

局限性包括：1）需要已知或可估计的污染率α；2）算法对计算复杂度未做实证评估，实际实现可能受限；3）研究仅限于可实现（realizable）情形，未扩展到对抗噪声或泛化误差的上界；4）理论证明主要基于理想化的随机模型，缺乏对真实LLM生成数据的验证。

---

## 45. Qwen3-TTS Technical Report

**arXiv ID:** 2601.15621 | [PDF](https://arxiv.org/pdf/2601.15621v1)

**作者:** Hangrui Hu `[一作]`, Junyang Lin `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了Qwen3-TTS系列多语言、可控、低延迟的文本到语音模型，并提供了两种语音分词器（25Hz单码本与12Hz多码本），支持3秒语音克隆、指令式控制与实时流式生成。

**💡 创新点**

创新点在于采用双轨自回归LM架构与语义-声学拆分的多码本分词器，既实现了高质量、多语言和跨语言声学迁移，又在极低首包延迟（97–150 ms）和长文本（10 min）生成中超越现有基准。

**🔧 技术方法**

技术包括Qwen2-Audio编码器、双轨LM（文本+语音），语义与声学并行的12.5 Hz多码本量化、25 Hz单码本量化、Diffusion Transformer+Flow Matching、BigVGAN、DPO与GSPO等对齐与优化方法。

**📊 数据集**

使用约5 百万小时的多语言语音数据（10种语言）及公开基准集CommonVoice、Fleurs、LibriSpeech、Seed‑TTS、CV3‑Eval、InstructTTSEval等进行训练与评测。

**📈 对比分析**

与CosyVoice3、MiniMax、ElevenLabs、KALL‑E等系统在WER、SIM、RTF、首包延迟等指标下对比，Qwen3‑TTS在零声克隆、跨语言、可控语音设计等任务均取得SOTA，首包延迟仅97–150 ms，RTF始终低于1。

**⚠️ 局限性**

局限性包括：仍需大量训练数据，跨语言对非目标语种仍有误差；12 Hz低帧率在极长文本的稳定性略逊于25 Hz；对极低延迟（≤10 ms）和多声道/多通道场景的支持尚待完善。

---

## 46. Learning a Unified Latent Space for Cross-Embodiment Robot Control

**arXiv ID:** 2601.15419 | [PDF](https://arxiv.org/pdf/2601.15419v1)

**作者:** Yashuai Yan `[一作]` (Autonomous Systems Lab, Technische Universität Wien), Dongheui Lee `[通讯]` (German Aerospace Center)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建一个跨多种人体与机器人本体的统一潜在空间，并在该空间中训练目标条件控制策略，实现多机器人统一控制。

**💡 创新点**

1) 将潜在空间拆分为五个身体部位子空间，细化对不同运动模式的对齐；2) 采用对比学习与自监督的相似度度量实现跨体态映射；3) 通过轻量级机器人嵌入层实现新机器人快速加入；4) 仅使用人类数据训练 c-VAE 控制策略，避免机器人演示数据需求。

**🔧 技术方法**

对比学习（Triplet Loss）、自监督相似度度量、条件变分自编码器（c‑VAE）、机器人特定嵌入层、正向运动学与逆向编码。

**📊 数据集**

HumanML3D（约4M帧人类动作），机器人姿态通过随机采样并使用前向运动学生成。

**📈 对比分析**

与 ImitationNet（单机型专用）以及全身潜在空间版本对比；实验显示：解耦子空间和统一模型在运动重映射、末端执行器位置/速度一致性上均优于基线，目标到达误差小于 1cm。

**⚠️ 局限性**

仅基于 SMPL 模型，缺失手部细节，导致手部重映射受限；未在真实场景中验证更复杂抓取任务；对极端姿态的鲁棒性尚未充分评估。

---

## 47. MiRAGE: A Multiagent Framework for Generating Multimodal Multihop Question-Answer Dataset for RAG Evaluation

**arXiv ID:** 2601.15487 | [PDF](https://arxiv.org/pdf/2601.15487v1)

**作者:** Chandan Kumar Sahu `[一作]` (ABB Inc), Matthew Hetrich `[通讯]` (ABB Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 MiRAGE 多智能体框架，用于自动生成多模态、多跳、领域特定的 QA 数据集，以评估 RAG 系统。

**💡 创新点**

创新点在于将多跳上下文构建、对抗式验证和领域专家角色注入等功能整合到一个递归式代理群中，实现高质量、可验证且与原始文档主题保持一致的合成数据。

**🔧 技术方法**

采用多智能体架构、语义块划分、图像描述生成、跨模态检索、LLM/ VLM 生成与验证，以及聚类去重等技术。

**📊 数据集**

使用四个领域的专业文档数据集：S&P Global 年报、UNECE GTRs、量化生物学 Arxiv 论文和纽约时报视觉新闻。

**📈 对比分析**

与现有 RAG 评测数据集对比，MiRAGE 在信度(>0.91)、相关性(>0.81)、多跳平均数>2.3、视觉支持分数0.21-0.45 等指标上表现优异，并通过 JS 散度验证覆盖原始主题。

**⚠️ 局限性**

主要限制包括高计算与 token 代价、延迟、视觉推理能力不足，以及对专业图像描述的依赖。

---

## 48. Abusive music and song transformation using GenAI and LLMs

**arXiv ID:** 2601.15348 | [PDF](https://arxiv.org/pdf/2601.15348v1)

**作者:** Jiyang Choi `[一作]`, Rohitash Chandra `[通讯]` (Centre for Artificial Intelligence and Innovation)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发并评估了一套基于生成式人工智能的音乐文本与声乐替换框架，以消除流行音乐中的攻击性与不当词汇。

**💡 创新点**

将有害内容的处理视为生成式转换任务，而非传统过滤；同时结合NLP情感分析与声学特征评估，提出多模态评估方法。

**🔧 技术方法**

采用HT‑Demucs音源分离、Suno AI文本到音频的Transformer+扩散模型、GPT‑4歌词重写、DistilBERT情感分析、MPNet语义相似度、HNR、CPP、Jitter、Shimmer等声学指标。

**📊 数据集**

选取四首含攻击性歌词的热门歌曲（Kanye West “Heil Hitler”、Cardi B “WAP”、Elton John “Island Girl”、Tom MacDonald “Whiteboy”）的MP3与歌词（来源 Genius、LyricFind）进行实验。

**📈 对比分析**

对原始与生成版本在情感分数、语义相似度以及声学特征进行对比；情感下降63.3–85.6%，语义相似度约0.33–0.66，HNR、CPP提升，Shimmer下降，表明生成版本更安全且音质更清晰。

**⚠️ 局限性**

样本仅四首，缺乏跨流派普适性；AI生成对节拍匹配不稳定，重现性差；语义相似度低，难以保留原意；缺少人类感知评估。

---

## 49. Can We Trust LLM Detectors?

**arXiv ID:** 2601.15301 | [PDF](https://arxiv.org/pdf/2601.15301v1)

**作者:** Jivnesh Sandhan `[一作]` (Kyoto University), Yugo Murawaki `[通讯]` (Kyoto University)

**通讯引用:** 290 | [OpenAlex ID](https://openalex.org/A5013357764)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估了训练自由与监督式 LLM 检测器，并提出基于监督对比学习（SCL）的框架，以提升跨域、跨生成器的鲁棒性和少样本适应能力。

**💡 创新点**

核心创新是将 InfoNCE 对比损失与风格嵌入相结合，形成可迁移的风格编码器；通过轻量级中心更新实现 25‑shot 低成本适配新 LLM；同时对抗实验揭示现有方法易受简单扰动影响。

**🔧 技术方法**

使用 DeBERTa‑v3 预训练模型，交叉熵分类损失，InfoNCE 对比学习，中心更新、梯度攻击（GCG）以及多种扰动技术。

**📊 数据集**

主要数据集包括 RAID（学术摘要）、CHEAT（作弊文本）、M4（多域多生成器），以及 LMSYS Arena 用于少样本适配实验。

**📈 对比分析**

与 Binocular、FastDetectGPT（训练自由）和 BERT、GAN‑BERT（监督）比较，SCL 在域内 RAID 上取得 95.98% 最高准确率、F1‑score 97% 及 0.9% FPR；在 OOD CHEAT 上也保持 97.83% 准确率；但在 M4 上性能显著下降，说明仍受分布偏移限制。

**⚠️ 局限性**

局限性：无法在所有域与生成器上实现通用检测；对代理模型选择、简单风格扰动（如引号、标注）和严重分布偏移（如 M4）鲁棒性不足；对抗攻击虽易成功，但生成文本不自然，难以在实际场景中隐蔽利用。

---

## 50. Domain-Specific Knowledge Graphs in RAG-Enhanced Healthcare LLMs

**arXiv ID:** 2601.15429 | [PDF](https://arxiv.org/pdf/2601.15429v1)

**作者:** Sydney Anuyah `[一作]` (Indiana University), Sunandan Chakraborty `[通讯]` (Indiana University)

**通讯引用:** 195 | [OpenAlex ID](https://openalex.org/A5111311082)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本文构建了基于PubMed抽象的三种疾病特定知识图谱（T2DM、AD、AD+T2DM），并将其作为检索来源用于检索增强生成（RAG）LLM，评估其在针对单跳与多跳医学问答的性能提升。

**💡 创新点**

创新点在于系统验证“精准匹配、范围限定”的KG检索策略优于宽泛合并策略，并揭示模型规模与检索质量的交互效应，提供了针对不同LLM规模的实践指南。

**🔧 技术方法**

主要技术包括CoDe‑KG抽取管道（改进的共指消解与句法分解）、结构化KG检索、零样本指令提示、温度调参与宏/微F1评估。

**📊 数据集**

使用的数据集为从PubMed检索的T2DM、AD及其交叉领域抽象（共约1,000篇），以及由作者构造的两套多选题探针（Probe1 100题，Probe2 110题）。

**📈 对比分析**

比较方法为对七款指令微调LLM在六种检索配置（无检索、G1、G2、G1+G2、G3、G1+G2+G3）与三种温度（0、0.2、0.5）下进行宏/微F1评估；结果显示：小型模型在使用AD特定KG（G2）时提升显著，而大型模型在合并图或无检索时往往表现相当或更好；统计显著性检验表明部分配置在Probe1与Probe2上有显著提升或下降。

**⚠️ 局限性**

限制包括：实验仅进行三次重复、KG构建可能引入实体/关系错误、检索深度与重排名未进一步优化、未评估置信度/校准、结果对其他疾病/多语言或实时更新场景的泛化不确定。

---

## 51. No Reliable Evidence of Self-Reported Sentience in Small Large Language Models

**arXiv ID:** 2601.15334 | [PDF](https://arxiv.org/pdf/2601.15334v1)

**作者:** Caspar Kaiser `[一作]` (University of Warwick), Sean Enderby `[通讯]` (University of Warwick)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过向多种开源权重大语言模型（Qwen、Llama、GPT‑OSS）提问约50个关于自身意识的Yes/No问题，并利用内部激活空间的三类truth‑classifier（LR、MM、TTPD）来判断模型对自身是否具有意识的真实信念；进一步在模型被强制回答“是”或“否”时验证分类器对“真实”信念的捕捉能力。

**💡 创新点**

创新点在于：①系统性地结合多家模型和多种尺度（0.6B‑70B）检验自我意识的信念；②首次使用内部激活的truth‑classifier评估模型在回答意识相关问题时的诚实度；③通过在训练数据中加入强制回答的对照样本，区分模型输出与其潜在信念。

**🔧 技术方法**

主要技术包括：内部激活提取（残差流）；三种线性/逻辑回归型分类器（LR、MM、TTPD）；数据增强策略（真/假答案、强制“是/否”）；概率一致性分析与回归检验；模型输出与分类器预测的对比。

**📊 数据集**

数据集：①约50个泛化意识相关问题，按主体（人类、LLM、模型自身）生成肯定与否定两种变体；②对应的训练集包含已知真伪答案的知识问答（人类、模型、LLM等），约占比50/50；③实验时亦尝试使用公开的动物分类、地理位置等非意识问题的问答集做对照。

**📈 对比分析**

通过将模型输出概率与truth‑classifier预测概率对比，评估模型是否诚实。LR分类器在hold‑out集上的准确率接近1；MM、TTPD约0.9。实验显示：模型在关于人类意识时给出高概率，关于自身或LLM时给出低概率；更大模型在自身否认意识时更为自信；整体说明模型并不相信自己具备意识。

**⚠️ 局限性**

局限性：①仅覆盖至70B参数的开源模型，未检验更大模型；②使用Yes/No单一答案格式，可能无法捕捉更细粒度的自我认识；③truth‑classifier的可靠性受训练数据和模型偏差影响；④缺少对自我参照提示的实验；⑤模型可能因误解问题而产生误差，影响结论；⑥未检验是否存在更深层的隐式意识表征。

---

## 52. Enhanced Convergence in p-bit Based Simulated Annealing with Partial Deactivation for Large-Scale Combinatorial Optimization Problems

**arXiv ID:** 2601.15561 | [PDF](https://arxiv.org/pdf/2601.15561v1)

**作者:** Naoya Onizawa `[一作]` (Research Institute of Electrical Communication), Takahiro Hanyu `[通讯]` (Research Institute of Electrical Communication)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文分析了使用概率位的模拟退火算法（pSA）在大规模组合优化问题中的局限性，并通过实验揭示了p-bits振荡导致能量停滞的根本原因。

**💡 创新点**

创新点在于提出两种基于部分失活p-bits的改进算法——时间平均pSA（TApSA）和停滞pSA（SpSA），有效消除振荡，显著提升求解质量。

**🔧 技术方法**

使用了概率位模型、Ising哈密顿量、时间平均和平滑、随机停滞机制，以及Python仿真实现。

**📊 数据集**

数据集包括G-set中的16个最大割（MAX-CUT）基准图（节点数800至5000）和全连接图K2000，以及100-spin图同构测试。

**📈 对比分析**

与传统pSA、经典SA、GPU异步并行算法和Coherent Ising Machine进行对比，TApSA和SpSA在16个基准上平均归一化割值提升至98.4%，显著优于传统方法。

**⚠️ 局限性**

局限性在于对参数α和p的依赖需逐图调优，增加了计算开销；目前仅在软件模拟上验证，硬件实现及更大规模验证仍待探索。

---

## 53. When Sharpening Becomes Collapse: Sampling Bias and Semantic Coupling in RL with Verifiable Rewards

**arXiv ID:** 2601.15609 | [PDF](https://arxiv.org/pdf/2601.15609v1)

**作者:** Mingyuan Fan `[一作]` (East China Normal University), Jun Zhou `[通讯]` (Ant Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了RLVR训练中出现的过度锐化（over‑sharpening）问题，并提出了逆成功优势校准（IAC）和分布级校准（DLC）两种方法来防止模型聚焦于有限的解模式，从而保持多样性。

**💡 创新点**

创新点在于：① 用理论分析揭示采样偏差和语义耦合如何导致过度锐化；② 设计了两种可插拔的校准机制，既能平衡优势贡献，又能通过记忆网络调节采样分布，兼顾效率与效果。

**🔧 技术方法**

采用了KL正则化的RLVR目标、优势归一化、梯度对齐与特征相似度分析、内存网络采样校准等技术，并在大型语言模型上实现了可扩展的训练与评估。

**📊 数据集**

实验使用了 DeepScaleR 作为RLVR训练集，并在 GSM8K、MATH 500、LightEval、AIME24、Minerva、O‑Bench 等六个数学与通用推理基准上评测。

**📈 对比分析**

与 GRPO、RLOO、Reinforce++、DrGRPO、DAPO 等基线在相同配置下对比，IAC/DLC 组合在 AVG@8 与 PASS@8 上均实现了显著提升，尤其在较难的数据集上效果更为突出。

**⚠️ 局限性**

局限性包括：对算力和超参数（如 α、μ）敏感；内存网络校准会带来较高的推理开销；当前方法主要针对可验证奖励场景，未知在更广泛任务中的适用性。

---

## 54. From Generation to Collaboration: Using LLMs to Edit for Empathy in Healthcare

**arXiv ID:** 2601.15558 | [PDF](https://arxiv.org/pdf/2601.15558v1)

**作者:** Man Luo `[一作]` (Abridge), Haidar M. Abdul-Muhsin `[通讯]` (Mayo Clinic)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

使用大语言模型（LLM）对医生书面回复进行编辑，以提升同理心而不损失医学事实。

**💡 创新点**

提出三方同理心评分（3‑EMRank）和双向医学事实检查评分（MedFactChecking Score），以及将LLM视为编辑器而非生成器的全新框架。

**🔧 技术方法**

利用LLM进行同理心编辑与评估；事实提取与蕴含检测使用Gemini‑2.5‑Flash；同理心评估使用自定义三方EMRank。

**📊 数据集**

163对患者-医生对话，来自前列腺癌患者的异步门户问答，已完成人工脱敏。

**📈 对比分析**

与医生原始回复、直接AI生成以及不同提示的编辑版本进行三方同理心对比；事实保持率用MedFactChecking评估。实验显示，精细化编辑方案在保持>90%事实完整性的同时，可将同理心显著提升，生成式回复同理心最高但事实正确率低。

**⚠️ 局限性**

数据量有限、仅覆盖泌尿科单一病种；同理心定义与评价标准仍不够细化；编辑提示对不同模型的通用性尚待验证。

---

## 55. Gated Sparse Attention: Combining Computational Efficiency with Training Stability for Long-Context Language Models

**arXiv ID:** 2601.15305 | [PDF](https://arxiv.org/pdf/2601.15305v1)

**作者:** Alfred Shen `[一作]`, Aaron Shen `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Gated Sparse Attention（GSA）架构，融合稀疏注意力与门控机制实现低复杂度与训练稳定；

**💡 创新点**

创新点在于同时采用Sigmoid门控提升表达能力、消除注意力沉没，并通过自适应稀疏控制动态调整关注的令牌数量；

**🔧 技术方法**

技术实现包括轻量级Lightning Indexer、双重门控（G1和G2）、自适应稀疏预算、以及两阶段训练（索引器预热+稀疏训练）；

**📊 数据集**

使用400B SlimPajama、WikiText-103、C4、MMLU、GSM8K等公开数据集进行训练与评估；

**📈 对比分析**

与标准密集注意力、稀疏仅、门控仅三种基线对比，GSA在128K上下文下速度提升约12–16×，注意力沉没率降至约4%，困境指标几乎翻倍，且整体性能优于门控仅且匹配/超过稀疏仅；

**⚠️ 局限性**

局限包括：短序列下索引器开销高、索引器仍为O(L^2)导致极长序列开销大、两阶段训练增加实现复杂度、以及对超参数（k_base、d^I、门控初始化）较敏感。

---

## 56. You Need Better Attention Priors

**arXiv ID:** 2601.15380 | [PDF](https://arxiv.org/pdf/2601.15380v1)

**作者:** Elon Litman `[一作]` (Stanford University), Gabe Guo `[通讯]` (Stanford University)

**通讯引用:** 36 | [OpenAlex ID](https://openalex.org/A5065759728)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种新的注意力机制Goat，将注意力视为熵正则化的最优传输问题，替代传统隐式均匀先验，引入可学习的先验分布以提升长上下文泛化与稳定性。

**💡 创新点**

创新点在于将注意力重构为可学习的EOT先验，给出注意力陷阱的理论解释，利用谱分解实现可学习的相对位置偏置，并实现无额外开销的SDPA兼容实现。

**🔧 技术方法**

技术包括熵正则化最优传输、KL正则化、傅里叶特征谱分解、相对位置编码以及FlashAttention等高效注意力内核。

**📊 数据集**

实验使用C4数据集进行语言建模，人类参考基因组用于DNA生成，ImageNet-1k用于视觉任务，以及passkey retrieval、needle‑in‑a‑haystack等长上下文合成任务。

**📈 对比分析**

与RoPE、ALiBi、旋转编码等基线对比，Goat在长上下文推理、语言建模、DNA生成与图像分辨率外推等任务上性能提升，保持或降低perplexity、提升准确率，并显著降低显存占用。

**⚠️ 局限性**

局限性包括先验形式的设计仍需经验选择（频率、正则化），对极端长序列的梯度稳定性需进一步验证，且在某些任务下仍需调优超参。

---

## 57. Social Robotics for Disabled Students: An Empirical Investigation of Embodiment, Roles and Interaction

**arXiv ID:** 2601.15293 | [PDF](https://arxiv.org/pdf/2601.15293v1)

**作者:** Alva Markelius `[一作]` (University of Cambridge), Hatice Gunes `[通讯]` (University of Cambridge)

**通讯引用:** 7486 | [OpenAlex ID](https://openalex.org/A5060090893)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了残疾学生在高等教育中使用社会机器人支持的体验，比较了两种角色（信息指引与倾听对话）和两种体现（Pepper机器人 vs Echo Dot语音助手），通过问卷与访谈收集感知数据。

**💡 创新点**

创新点在于系统性评估实体化与非实体化机器人在残疾学生支持中的差异，提出多维度评价框架（理解度、社交能量、信息获取、任务难度、隐私），并考察不同残疾类型对机器人感知的调节作用。

**🔧 技术方法**

采用Pepper机器人和Echo Dot设备的混合对话系统，结合预设脚本与GPT‑4生成的对话，配合语音识别与文本转写技术实现人机交互。

**📊 数据集**

使用31名剑桥大学残疾学生的实验数据，包括自定义问卷和访谈文本，数据已上传至GitHub进行公开分析。

**📈 对比分析**

通过混合效应线性模型、Wilcoxon检验等统计方法进行比较，结果显示实体机器人在理解度和社交存在感上显著优于语音助手，但在信息获取、任务完成和隐私担忧方面未显示显著优势，且不同残疾类型对结果有调节效应。

**⚠️ 局限性**

局限包括样本规模有限、单一高校样本、量表为自定义且缺乏与现有研究直接对比、对不同残疾类别的覆盖不完整，以及未对长期使用效果进行评估。

---

## 58. A Prompt-Based Framework for Loop Vulnerability Detection Using Local LLMs

**arXiv ID:** 2601.15352 | [PDF](https://arxiv.org/pdf/2601.15352v1)

**作者:** Adeyemi Adeseye `[一作]` (Brilloconnetz Partners avoin yhtiö), Aisvarya Adeseye `[通讯]` (University of Turku)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5117317772)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了基于本地LLM的循环漏洞检测框架，针对Python代码检测三类循环漏洞。

**💡 创新点**

通过结构化prompt设计实现安全、低延迟的循环漏洞检测，并用手工基准验证模型性能，展示本地LLM在此任务中的可行性。

**🔧 技术方法**

使用本地LLM（LLaMA 3.2 3B 和 Phi 3.5 4B）、prompt工程、人工标注基准与精度/召回/F1评估。

**📊 数据集**

采用两名经验Python开发者手工标注的含循环漏洞的Python程序集（未公开具体代码量）。

**📈 对比分析**

与手工基准对比，计算精度、召回、F1；Phi在所有三类漏洞上均优于LLaMA，最高F1分别为0.97（资源管理）、0.95（安全风险）、0.90（循环控制）。

**⚠️ 局限性**

仅覆盖循环控制/逻辑、内部安全风险、资源管理三类漏洞，无法检测并发/同步问题；仅评估两小模型，未扩展到更大或其他语言。

---

## 59. Neural Collision Detection for Multi-arm Laparoscopy Surgical Robots Through Learning-from-Simulation

**arXiv ID:** 2601.15459 | [PDF](https://arxiv.org/pdf/2601.15459v1)

**作者:** Sarvin Ghiasi `[一作]` (McGill University), Amir Hooshiar `[通讯]` (McGill University)

**通讯引用:** 669 | [OpenAlex ID](https://openalex.org/A5088381242)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在多臂腹腔镜手术机器人中，作者开发并验证了一套融合解析模型、实时仿真与深度学习的碰撞检测框架。

**💡 创新点**

创新点在于将Bézier曲线解析距离计算与学习自仿真的神经网络相结合，实现快速、准确的最小间距预测，并提供实时警报。

**🔧 技术方法**

使用的技术包括Denavit–Hartenberg正运动学、Unity 3D仿真、线性Bezier曲线碰撞算法以及多层前馈神经网络（LeakyReLU、Dropout、AdamW）。

**📊 数据集**

数据集来自于Unity仿真产生的7.5万余随机关节配置和相对位置，包含输入向量和对应的最小距离标签。

**📈 对比分析**

与解析模型和实验测量结果对比，网络平均绝对误差为282.2 mm、R²=0.85，显示与真实距离高度一致，能实时发出低于0.2 m时的警告。

**⚠️ 局限性**

局限性包括仅使用线段近似机器人臂、未考虑臂半径和动态约束、以及在真实临床环境中验证仍不足。

---

## 60. ToxiTwitch: Toward Emote-Aware Hybrid Moderation for Live Streaming Platforms

**arXiv ID:** 2601.15605 | [PDF](https://arxiv.org/pdf/2601.15605v1)

**作者:** Baktash Ansari `[一作]` (University of Washington), Afra Mashhadi `[通讯]` (University of Washington)

**通讯引用:** 1040 | [OpenAlex ID](https://openalex.org/A5061449471)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种面向Twitch的混合式毒性检测框架ToxiTwitch，结合LLM生成的文本与表情嵌入与传统机器学习分类器实现低延迟、可解释的实时审核

**💡 创新点**

创新点在于：①将频道特定表情（emotes）视为重要语义单元并通过描述生成或与全局表情匹配来增强LLM推理；②将LLM嵌入与轻量化分类器融合，既保留LLM上下文理解，又兼顾实时性能；③在小规模但多模态的Twitch数据上进行零样本与微调对比实验，展示表情嵌入显著提升F1值

**🔧 技术方法**

技术包括：大语言模型（LLaMA‑3‑8B‑Instruct、Deepseek‑R1‑Distill）用于文本/表情嵌入与链式思维推理；BLIP‑2用于表情图片描述；词向量空间（Word‑2‑Vec）用于表情匹配；传统机器学习分类器（随机森林、线性SVM）用于最终判定；自定义软提示与零样本推理技术

**📊 数据集**

数据集为两条Twitch频道（HasanAbi、LolTyler1）各500条评论，包含文字与频道/全局表情；人工标注3名标注者完成毒性与类别标签；同时采集全局表情嵌入空间（2379个英文频道）用于表情匹配

**📈 对比分析**

与三大基线（Detoxify、HateSonar、DistilBERT‑ToxiGEN）对比：ToxiTwitch在HasanAbi和LolTyler1上分别取得F1≈0.73/0.71、精度≈0.63/0.61、召回≈0.87/0.88，准确率0.80/0.79；延迟约60 ms/条；相较基线，ToxiTwitch在F1和精度上均优于HateSonar与ToxiGEN，延迟略高于Detoxify但仍低于70 ms的HateSonar

**⚠️ 局限性**

局限包括：样本仅两条频道、总1 000条评论，难以推广到多样化社区；表情匹配基于静态嵌入空间，易受表情语义漂移影响；LLM推理仍存在低精度、误报高的问题；未对不同毒性类别做细粒度评估；未使用大规模自动标注或持续学习机制

---

## 61. Entropy-Tree: Tree-Based Decoding with Entropy-Guided Exploration

**arXiv ID:** 2601.15296 | [PDF](https://arxiv.org/pdf/2601.15296v1)

**作者:** Longxuan Wei `[一作]` (Shanghai Jiaotong University), Junchi Yan `[通讯]` (Shanghai Jiaotong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Entropy-Tree 解码方法，通过在高熵且语义重要的词上分支，实现树形解码。

**💡 创新点**

创新点在于用模型输出的词熵和自注意力权重动态筛选分支点，结合前缀共享的树结构，既提升推理准确性又改善校准。

**🔧 技术方法**

采用词熵阈值和自注意力重要性筛选，树形扩展时选取 top‑b 词做分支，并用预测熵进行不确定度量。

**📊 数据集**

在 SVAMP、MATH‑500、SciBench、GPQA‑main/diamond、AIME24/25 等中级难度推理数据集上验证。

**📈 对比分析**

与多链随机抽样（Multi‑chain）和 Self‑Consistency 等基线比较，Entropy‑Tree 在 pass@k 上普遍优于 Multi‑chain（例如 Qwen2.5‑7B‑Instruct 在 MATH‑500 上 pass@10 由 75.41% 提升至 78.24%），同时其预测熵的 AUROC 也高于传统方法。

**⚠️ 局限性**

局限性包括需要访问 logits 与自注意力得分导致额外延迟，且在仅支持黑盒 API 的环境下难以直接使用。

---

## 62. Empowering LLMs for Structure-Based Drug Design via Exploration-Augmented Latent Inference

**arXiv ID:** 2601.15333 | [PDF](https://arxiv.org/pdf/2601.15333v1)

**作者:** Xuanning Hu `[一作]` (Jilin University), Bo Yang `[通讯]` (Jilin University)

**通讯引用:** 71175 | [OpenAlex ID](https://openalex.org/A5072820962)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了ELILLM框架，利用LLM潜在空间进行系统探索与知识引导解码，实现结构基础药物设计的高亲和性分子生成。

**💡 创新点**

将LLM生成拆解为编码-探索-解码三步，采用贝叶斯优化在潜在空间探索、位置感知高斯过程代理模型和知识驱动的SMILES修复解码，显著提升对未知结构的针对性和化学有效性。

**🔧 技术方法**

使用LLM（LLaMA 3.1）潜在空间投影、贝叶斯优化、深度核高斯过程、位置感知聚合、角色化学知识提示、SMILES修复及Vina对接评估。

**📊 数据集**

使用CrossDocked2020数据集（65K目标–配体对，100个测试靶点）。

**📈 对比分析**

与七种基线（AR、Pocket2Mol、liGAN、TargetDiff、ALIDiff、TamGen、LMLF）以及随机与差分版本进行对比，ELILLM在Top1/5/10/20的Vina分数均优于基线，ELILLM-diff实现SOTA，提升幅度约4–8%。

**⚠️ 局限性**

受限于预训练LLM对蛋白序列理解不足，探索过程需要大量对接评估导致计算成本高；高维潜在空间中GP的收敛性有限，且对化学约束的完整性依赖提示设计。

---

## 63. Replayable Financial Agents: A Determinism-Faithfulness Assurance Harness for Tool-Using LLM Agents

**arXiv ID:** 2601.15322 | [PDF](https://arxiv.org/pdf/2601.15322v1)

**作者:** Raffi Khatchadourian `[一作]` `[通讯]` (IBM Financial Services Market), Raffi Khatchadourian (IBM Financial Services Market)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估LLM代理在金融服务中的可重复性与证据一致性，并提出DFAH评估框架。

**💡 创新点**

将轨迹级确定性与证据可信度统一度量，发现结构化（schema‑first）架构显著提升确定性，并证明确定性与可信度正相关。

**🔧 技术方法**

采用多跑对比、轨迹记录、Jaccard词汇相似度的证据对齐、模型自评和人类校准，并设计四种压力测试场景。

**📊 数据集**

使用三套金融基准（合规分流、组合约束、DataOps异常）共150条案例，并公开 GitHub 数据与代码。

**📈 对比分析**

对74个模型配置进行 8–24 次跑，统计决策/动作确定性与可信度；7–20B 模型达到 100% 决策确定性，结构化输出提升确定性，前沿模型确定性低；正相关系数 r=0.45，压力测试中 schema‑first 7B 维持近 100%。

**⚠️ 局限性**

仅针对金融任务，稀有漂移可能被低估；可信度指标召回率低，未评估最终正确性、公平性与对抗鲁棒性。

---

## 64. Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors

**arXiv ID:** 2601.15625 | [PDF](https://arxiv.org/pdf/2601.15625v1)

**作者:** Zhiwei Zhang `[一作]` (Chinese University of Hong Kong), Kam-Fai Wong `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 11048 | [OpenAlex ID](https://openalex.org/A5008208316)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种新框架 Fission‑GRPO，利用“核裂变”机制将多轮工具调用中的执行错误转换为可训练的监督信号，从而提升小型 LLM 的错误恢复能力。

**💡 创新点**

核心创新在于在 RL 训练循环中对失败轨迹进行分裂并附加诊断反馈，动态生成与模型当前错误分布对齐的恢复样本，弥补传统 RL 仅将错误视为稀疏惩罚的不足。

**🔧 技术方法**

技术实现基于 GRPO 强化学习、细调的错误模拟器（Error Simulator）以及三阶段闭环更新（Stage 1–3）来生成、利用并更新错误修正数据。

**📊 数据集**

主要使用 BFCL v4 Multi‑Turn benchmark 进行训练与评估，并自建高质量工具调用轨迹数据集。

**📈 对比分析**

与 GRPO、DAPO、Dr.GRPO 以及专业 8B 工具代理在 Qwen3 系列模型上对比，Qwen3‑8B 在 BFCL v4 上整体准确率从 42.75% 提升至 46.75%，错误恢复率提升 5.7%。

**⚠️ 局限性**

局限性包括仅在 BFCL v4 进行评估，未验证在其他交互式错误‑重试场景；核裂变机制带来额外计算开销，需进一步优化训练效率。

---

## 65. CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation

**arXiv ID:** 2601.15408 | [PDF](https://arxiv.org/pdf/2601.15408v1)

**作者:** Pablo Messina `[一作]` (Pontificia Universidad Catolica de Chile), Bernard Ghanem `[通讯]` (KAUST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出CURE框架，通过错误感知的课程学习提升医学视觉语言模型的视觉定位和报告可信度。

**💡 创新点**

创新点在于动态根据模型误差调整采样分布，在多任务学习中实现对数据集与类别级别的自适应重权重，从而显著提高定位精度并减少幻觉。

**🔧 技术方法**

采用多任务训练、错误感知课程学习、LoRA微调、4位量化、提示工程，并利用MedGemma‑4B‑IT基础模型。

**📊 数据集**

使用公开胸部X光数据集：Chest ImaGenome、PadChest‑GR、MS‑CXR、MIMIC‑CXR和零样本VinDr‑CXR。

**📈 对比分析**

与MAIRA‑2和基线MedGemma‑4B‑IT对比，CURE在所有定位任务上提升Micro/Macro IoU 0.37+，报告质量CXRFEScore提升0.188，幻觉率降低18.6%，在无额外数据的条件下实现SOTA。

**⚠️ 局限性**

局限性包括对数据分布不均衡的假设、仍需手动设置课程频率与α参数、在多标签报告生成中文本指标仍落后于MAIRA‑2，并且在非常罕见解剖位置上提升有限。

---

## 66. CompliantVLA-adaptor: VLM-Guided Variable Impedance Action for Safe Contact-Rich Manipulation

**arXiv ID:** 2601.15541 | [PDF](https://arxiv.org/pdf/2601.15541v1)

**作者:** Heng Zhang `[一作]` (Istituto Italiano di Tecnologia), Arash Ajoudani `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种 CompliantVLA‑Adaptor，将 Vision‑Language‑Action（VLA）模型与基于 VLM 的上下文感知可变阻尼控制（VIC）结合，实现在接触丰富的机器人操作中的安全、可调节力反馈。

**💡 创新点**

创新点在于：①使用预训练的 VLM 进行多模态（图像+语言+力传感）推理，实时生成任务阶段对应的刚度与阻尼参数；②在 VLA 的高层决策与低层控制之间插入可变阻尼模块，兼具 VLA 的泛化能力与 VIC 的接触安全；③采用双层安全策略（VLM 预判 + 真实力反馈）动态调节阻尼，避免过载。

**🔧 技术方法**

技术包括：VLM（如 ChatGPT‑4o‑mini）作为上下文推理引擎；可变阻尼控制器（VIC）实现弹簧-阻尼动力学；力/扭矩传感器实时监测；多模态 prompt 设计用于相位识别与阻尼参数生成；机器人系统集成在 7‑DoF Franka Emika Panda 机器人上；基于 WebSocket 的远程 VLA 模型推理。

**📊 数据集**

使用 LIBERO 与 ManiSkill 公开基准中的 8 个接触丰富任务（插槽插拔、抽屉开关、物体推拉等），以及自建的物理上下文及力-阻尼情景数据集（包含图像、语言指令、实时力反馈）。

**📈 对比分析**

在模拟与真实机器人上与 SOTA VLA（RDT‑1B、Pi0、OpenVLA‑oft）做对比，设置 30 N 力阈值。实验表明：平均成功率从 9.86% 提升至 17.29%；大部分任务（7/8）成功率显著提升，且力违规次数大幅下降；在抽屉、插槽等机械约束任务中效果尤为突出。

**⚠️ 局限性**

局限性包括：①VLM 处理延迟与高频控制不匹配；②使用云端 VLM 费用高且不环保；③VLM 训练缺乏机器人特定知识，参数生成可能不最优；④在完全新环境或物体上泛化性仍有限。

---

## 67. Language Models Entangle Language and Culture

**arXiv ID:** 2601.15337 | [PDF](https://arxiv.org/pdf/2601.15337v1)

**作者:** Shourya Jain `[一作]` (Lossfunk), Paras Chopra `[通讯]` (Lossfunk)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了多语言大型语言模型（LLM）在开放式问答中的回答质量及其与文化背景的关联。

**💡 创新点**

创新点：首次采用文化中性提示和基于LLM的评判方法，系统评估不同语言下回答的质量与文化差异，并公开翻译版CulturalBench数据集。

**🔧 技术方法**

使用技术包括：LLM-评判（Cohere Command‑A）、翻译模型 Gemini‑2.5‑Flash、Qwen3‑0.6b 嵌入+HDBSCAN聚类、统计检验（Kruskal–Wallis）等。

**📊 数据集**

数据集：WildChat（用于生成文化中性问答）、CulturalBench（原版及翻译版）以及多语言版本的问答。

**📈 对比分析**

通过对 5 种多语言模型（Qwen3‑14B、Cohere‑Aya 32B/8B、Magistral、Sarvam‑m）在 6 种语言上的回答进行 LLM‑评判，发现非英语语言回答质量显著低于英语，且不同语言会引入不同文化背景，影响回答表现。

**⚠️ 局限性**

局限性：评判完全依赖 LLM，可能存在评判偏差；实验仅涵盖中小型开源模型，未测试更大模型；未对模型内部机制进行解释。

---

## 68. Put Your Muscle Into It: Introducing XEM2, a Novel Approach for Monitoring Exertion in Stationary Physical Exercises Leveraging Muscle Work

**arXiv ID:** 2601.15472 | [PDF](https://arxiv.org/pdf/2601.15472v1)

**作者:** Jana Franceska Funke `[一作]` (Ulm University), Teresa Hirzle `[通讯]` (University of Copenhagen)

**通讯引用:** 684 | [OpenAlex ID](https://openalex.org/A5035721389)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究开发了 XEM^2 系统，利用 Azure Kinect 摄像头捕捉运动姿态，并通过 Hill 型肌肉模型计算各肌群工作量，随后以条形图和 3D 角色动画进行可视化。

**💡 创新点**

其创新点在于提供一种基于摄像头的实时单肌群工作量监测与可视化方法，弥补传统燃烧卡路里和自我感知尺度在非连续运动中的信息不足。

**🔧 技术方法**

技术方案包含 Azure Kinect Body Tracking SDK、Unity 引擎、Kinesis Muscle Model、Hill 型肌肉模型以及自定义的 3D 角色渲染。

**📊 数据集**

数据集为 36 名参与者在 FitXR 运动游戏中的训练数据以及 10 名参与者在 5 个常规运动（手臂圆周、弓步、肩部挤压、带/不带手臂的深蹲）中的重复测量，共计 125 次动作采样。

**📈 对比分析**

通过与 RPE 量表和燃烧卡路里两种传统方法对比，评估可信度、可用性和信息量；实验结果显示 XEM^2 在可信度与信息丰富度上与传统方法相当，并且被认为能提供更细粒度的肌群负荷洞察。

**⚠️ 局限性**

系统局限包括：模型缺乏个体化参数（身高、BMI、肌肉长度等）、对服装颜色和摄像角度敏感、未考虑极端姿态误差、仅适用于固定式或在室内使用的运动场景。

---

## 69. Intelligence Degradation in Long-Context LLMs: Critical Threshold Determination via Natural Length Distribution Analysis

**arXiv ID:** 2601.15300 | [PDF](https://arxiv.org/pdf/2601.15300v1)

**作者:** Weiwei Wang `[一作]`, Weijie Zou `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了Qwen2.5-7B在自然长度分布下的智能退化现象，并定位了40-50%上下文阈值导致的性能崩溃。

**💡 创新点**

创新点在于提出自然长度分布分析方法、五种交叉验证阈值判定以及统一的浅层长上下文适应框架。

**🔧 技术方法**

采用RoPE位置编码分析、注意力分散量化、信息理论瓶颈分析及多方法梯度、二阶导数等技术。

**📊 数据集**

使用混合数据集，包括500条SQuAD短文本与500条NarrativeQA长文本，覆盖5-95%上下文长度。

**📈 对比分析**

通过五方法交叉验证得到的阈值一致性高，F1从0.556降至0.302，降幅45.5%，显著高于30%的临界阈值。

**⚠️ 局限性**

局限性包括仅研究了单模型与单任务，未验证其他规模模型与不同任务的普适性。

---

## 70. DS@GT at TREC TOT 2025: Bridging Vague Recollection with Fusion Retrieval and Learned Reranking

**arXiv ID:** 2601.15518 | [PDF](https://arxiv.org/pdf/2601.15518v1)

**作者:** Wenxin Zhou `[一作]` (Georgia Institute of Technology), Anthony Miyaguchi `[通讯]` (Georgia Institute of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段检索体系，结合LLM、稀疏BM25、稠密BGE‑M3三种检索方式的混合检索和两种再排序（LambdaMART学习型和LLM列表式）以解决TREC Tip‑of‑the‑Tongue任务；

**💡 创新点**

1）将LLM检索结果与稀疏/稠密检索通过轮询合并，实现检索召回显著提升；2）设计主题感知多索引稠密检索，按24个主题划分Wiki，提高检索效率；3）使用LLM生成5000条合成查询并进行数据增强，提升学习型再排序的泛化能力；

**🔧 技术方法**

Hybrid retrieval（LLM、BM25、BGE‑M3）、Topic‑aware multi‑index FAISS、LambdaMART（XGBoost）再排序、LLM列表式再排序（Gemini‑2.5‑flash、Gemma‑27B等）、PageRank、页面浏览量特征、词频、网络拓扑特征；

**📊 数据集**

TREC Tip‑of‑the‑Tongue 2025官方数据集、Wikipedia 2024年6月数据集、通过LLM生成的5000条合成查询；

**📈 对比分析**

与单一检索方式（仅BM25、仅BGE‑M3、仅LLM）和单一再排序（仅LambdaMART或仅LLM）对比，混合检索+LLM再排序的最佳系统在测试集上达到R@10 0.4341、R@1000 0.6559、NDCG@1000 0.4106，明显优于单一方式（例如BGE‑M3 R@1000 0.5498、NDCG 0.1492）；

**⚠️ 局限性**

LambdaMART在测试集上的Reciprocal Rank下降显著，说明泛化不足；主题划分可能漏掉跨主题相关文档；网络特征对再排序贡献有限；对不同来源查询的表现差异大，需进一步优化。

---

## 71. Securing LLM-as-a-Service for Small Businesses: An Industry Case Study of a Distributed Chatbot Deployment Platform

**arXiv ID:** 2601.15528 | [PDF](https://arxiv.org/pdf/2601.15528v1)

**作者:** Jiazhu Xie `[一作]` (RMIT University), Fengling Han `[通讯]` (RMIT University)

**通讯引用:** 4609 | [OpenAlex ID](https://openalex.org/A5021052552)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本论文通过案例研究提出了一个基于轻量级k3s集群、加密覆盖网络的多租户LLM部署平台，使小型企业能通过无代码流程快速上线安全的检索增强型聊天机器人。

**💡 创新点**

创新点在于将成本可控的分布式边缘云与平台级层叠防御（守护提示 + GenTel‑Shield）相结合，既实现了资源池化与容器隔离，又在不需模型再训练的前提下抵御Prompt Injection攻击。

**🔧 技术方法**

主要技术包括k3s轻量级Kubernetes、加密覆盖网络、容器化隔离、DGX Spark加速推理节点、RAG检索管线、守护提示、GenTel‑Shield注入检测器以及无代码配置工作流。

**📊 数据集**

使用的数据集为All Table Sports Australia的客户支持记录（250条正常查询）与从GenTel‑Safe改编的Prompt Injection攻击数据（250条攻击样本），两者混合形成平衡评测集。

**📈 对比分析**

通过比较四种安全配置（纯LLM、守护提示、GenTel‑Shield、两者结合），在三个LLM（GPT‑4.1‑mini、GPT‑4.1、Ministral‑3B）上测得组合方案F1≈99.8%，同时在私有云k3s部署下的推理延迟显著低于裸机，Guard提示引入的额外延迟可控。

**⚠️ 局限性**

局限性包括守护提示需要人工调优且可能不适用于未知域；GenTel‑Shield的检测仍可能遗漏新型攻击；实验仅基于单一电商场景与有限样本，未覆盖长期运维成本和多种硬件环境的可扩展性验证。

---

## 72. Nested and outlier embeddings into trees

**arXiv ID:** 2601.15470 | [PDF](https://arxiv.org/pdf/2601.15470v1)

**作者:** Shuchi Chawla `[一作]` (University of Texas), Kristin Sheridan `[通讯]` (University of Texas)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种用于把度量空间随机嵌入到层次分离树（HST）中的新算法，该算法允许对有限数目的“异常点”做出截断，从而在剩余点上获得更低的期望变形率。

**💡 创新点**

创新点主要有：
1) 将嵌入的“嵌套组合”概念推广到概率嵌入；
2) 设计了一种可在 HST 上实现的合并（merge）函数，使得嵌套组合能够保持非收缩并控制扩张；
3) 通过线性规划（LP）与舍入（rounding）得到近似的 (k, c)-异常嵌入，并给出二元逼近 (bicriteria) 结果：在目标变形率 c 下，能够在期望 (32+ε)c 的变形率下仅丢弃 O(k/(ε log²k)) 个异常点；
4) 将该方法应用于 buy‑at‑bulk、dial‑a‑ride 等典型图优化问题，得到实例特定的更好近似比。

**🔧 技术方法**

主要技术：
- 概率嵌入与期望变形率的定义；
- 嵌套组合与合并函数的通用框架；
- 对 HST 的合并算法（merge）和证明其满足非收缩、扩张界定；
- 基于 HST 的 LP 约束（包含 outlier 变量 δ_i）以及与之前工作相同的舍入方案；
- 对期望扩张和异常点数的上界分析，利用调和数 H_k 和 log 量级。

**📊 数据集**

本文为理论论文，未给出具体实验数据集；所有结果均为理论证明与算法复杂度分析。

**📈 对比分析**

与现有最优近似算法（基于 HST 嵌入的  O(log n) 变形率）相比，本文的方法在实例特定场景下可通过去除少量异常点显著降低变形率，从而改进了 buy‑at‑bulk 与 dial‑a‑ride 等问题的近似比。实验对比未给出，主要通过理论上限证明其优越性。

**⚠️ 局限性**

限制与挑战：
- 异常点集合保持确定性，未考虑随机异常点的情况；
- 运行时间与 ε、度量直径 Δ 相关，且为期望多项式时间；
- 仅适用于 HST（以及可通过 HST 近似的度量），对更一般目标空间的扩展尚未给出；
- 对于某些度量，最小异常点数 k 的求解仍为 NP‑hard，需使用近似或启发式方法。

---

## 73. Autonomous Business System via Neuro-symbolic AI

**arXiv ID:** 2601.15599 | [PDF](https://arxiv.org/pdf/2601.15599v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 74. DCeption: Real-world Wireless Man-in-the-Middle Attacks Against CCS EV Charging

**arXiv ID:** 2601.15515 | [PDF](https://arxiv.org/pdf/2601.15515v1)

**作者:** Marcell Szakály `[一作]` (University of Oxford), Sebastian Köhler `[通讯]` (University of Oxford)

**通讯引用:** 13224 | [OpenAlex ID](https://openalex.org/A5047602412)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过基于软件定义无线电（SDR）的 HomePlug Green PHY（HPGP）实现，对电动汽车（EV）与充电站间的 CCS 通信进行实时的无线中间人（MitM）攻击实验，验证并演示了从 TLS 降级、协议版本劫持到电能交付信息篡改等多种攻击手段。

**💡 创新点**

创新点包括：①首次实现实时 SDR 级 HPGP 协议栈，实现对有线充电线的无线窃听与注入；②对 2,750 条真实充电会话进行时序分析，量化 SDP 响应时延，为攻击提供精确的时序窗口；③提出多种提升攻击可靠性的技术（预劫持、邻居发现欺骗、同步注入等）；④通过实验展示了对 TLS、协议版本、功率控制等安全关键环节的完整劫持；⑤设计并验证了一个兼容向后、降级防护的物理层安全扩展。

**🔧 技术方法**

使用技术包括：软件定义无线电（USRP N210 + 低频放大器和天线）、自研 HPGP 协议栈（OFDM、TEK/NMK 加密）、IPv6 Neighbor Advertisement 伪造、SDP 响应竞争、TLS 降级与剥离、协议版本切换、V2G 电能信息（电压、电流）篡改、以及基于基本信号的降级防护扩展。

**📊 数据集**

数据集：使用 2,750 条从公开研究获取的真实充电会话抓包数据，用于分析 SDP 响应时延；实验数据来自多款实际 EV 与充电器（例如 DIN 70121、ISO‑15118‑2、ISO‑15118‑20 版本的车辆及四大品牌充电桩），并在这些设备上进行手工注入与攻击演示。

**📈 对比分析**

性能对比：SDR 的 SDP 响应时间约 16.1 ms，成功率在 85 % 的首次尝试；与 EV 的 344.3 ms 发送 SDP 请求相比，SDR 具备显著时延优势；攻击成功率通过预劫持与邻居伪造技术几乎可达 100 %；在功率篡改实验中，车辆在被上限电流超过 2 倍时仍未自动断电，显示攻击具有实质危害。

**⚠️ 局限性**

局限性：需攻击者靠近充电桩，受无线泄漏范围限制（实验可达约 1.5 m）；实现依赖较高成本 SDR（USRP）与专业硬件；实验仅覆盖部分 CCS 版本与厂商，未来固件更新或安全补丁可能抑制部分攻击；所提降级防护扩展仍需在硬件层面实现，且未在大型部署环境中验证。

---

## 75. MEDFORD in a Box: Improvements and Future Directions for a Metadata Description Language

**arXiv ID:** 2601.15432 | [PDF](https://arxiv.org/pdf/2601.15432v1)

**作者:** Polina Shpilker `[一作]` (Tufts University), Noah M. Daniels `[通讯]` (University of Rhode Island)

**通讯引用:** 744 | [OpenAlex ID](https://openalex.org/A5071391690)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究实现了 MEDFORD‑in‑a‑Box（MIAB）套件，升级了 MEDFORD 解析器、BagIt 导出功能、验证规则和 VS Code 扩展，支持外部引用、宏语法改进及更丰富的数据模型；

**💡 创新点**

创新点在于引入可导入外部 MEDFORD 文件的语法、统一的对象命名与唯一标识、基于 YAML 的自定义验证规则、以及通过 BagIt 打包实现完整数据与元数据的安全传输；

**🔧 技术方法**

采用 Python（pydantic、pyglas）实现解析器与验证器，使用 VS Code 的 Language Server Protocol（LSP）构建 IDE 插件，BagIt 技术用于打包，YAML 用于定义自定义实体与验证规则；

**📊 数据集**

主要以珊瑚礁研究数据为测试案例，使用真实的珊瑚种类、基因组构建、礁域坐标等元数据信息；

**📈 对比分析**

实验主要以手工验证和小规模功能测试为主，未给出大规模基准或性能指标，评估集中于功能完整性和易用性；

**⚠️ 局限性**

局限性包括缺乏系统化的性能评估、宏语法尚未确定、外部引用可能导致冲突和版本不一致、尚未在多学科数据集上充分验证。

---

## 76. Aeon: High-Performance Neuro-Symbolic Memory Management for Long-Horizon LLM Agents

**arXiv ID:** 2601.15311 | [PDF](https://arxiv.org/pdf/2601.15311v1)

**作者:** Mustafa Arslan `[一作]` `[通讯]` (Independent Researcher), Mustafa Arslan (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Aeon 系统，将 LLM 长上下文记忆问题从检索问题转变为操作系统级别的资源管理，设计了 Atlas 空间索引、Trace 事件图和 Semantic Lookaside Buffer（SLB）并实现零拷贝的 C++/Python 核心-外壳架构。

**💡 创新点**

创新点在于：①把记忆视为可管理资源，采用 OS 级别的内存分配、页面调入与上下文切换；②通过 Atlas 结合 B+树与 HNSW 的混合索引实现语义空间的层级化查询；③设计 SLB 预测式缓存利用语义惯性实现近毫秒级别检索；④使用零拷贝共享内存消除跨语言序列化开销；⑤引入 Trace DAG 记录事件因果链，实现可解释的回溯与上下文锚定。

**🔧 技术方法**

核心技术包括：C++23 SIMD 加速（AVX-512/ARM NEON）向量相似度核；B+树与 HNSW 结合的 Atlas 空间索引；SLB 预测缓存与 LRU 淘汰；零拷贝接口（pybind11+NumPy capsule）；Python Shell 进行高层推理；图数据库形式的 Trace DAG；系统级内存映射与页缓存管理；SIMDe 等跨平台 SIMD 转译；容器化与跨平台部署。

**📊 数据集**

使用的主要数据集是合成的“Dense Forest”——10⁴、10⁵、10⁶ 维度 768 的向量集合；工作负载包括 Uniform Random（随机查询）和 Conversational Walk（语义随机游走）。

**📈 对比分析**

通过与两种基线对比：①全量线性扫描（Brute‑Force）②FAISS HNSW；采用 P99 延迟、QPS、内存占用、SLB 命中率等指标。结果显示：Aeon Cold 在 1M 节点时 P99 < 1 ms，Aeon Warm 平均延迟 0.42 ms（SLB 命中率 85%），比 HNSW 快 3×，比线性扫描快 40×；在 10⁴ 节点时延迟 0.8 ms，随节点数增长保持 4–5 层树深度，实现对数级扩展；零拷贝接口在 10 MB 数据传输时仅 2 µs。

**⚠️ 局限性**

局限性包括：①嵌入模型固定，无法处理概念漂移；②目前仅支持文本单模态，缺乏多模态统一索引；③多租户与硬件隔离功能尚未实现；④缺乏在线学习或自适应重建 Atlas 的机制；⑤在极大规模或多用户场景下的安全与隔离需要进一步研究。

---

## 77. When Generative AI Meets Extended Reality: Enabling Scalable and Natural Interactions

**arXiv ID:** 2601.15308 | [PDF](https://arxiv.org/pdf/2601.15308v1)

**作者:** Mingyu Zhu `[一作]` (Pennsylvania State University), Bin Li `[通讯]` (Pennsylvania State University)

**通讯引用:** 25553 | [OpenAlex ID](https://openalex.org/A5100365212)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过将生成式人工智能（GenAI）与扩展现实（XR）技术相结合，展示了在VR教育、AR辅助和MR训练三大应用场景中，利用自然语言交互自动生成和编辑3D内容，突破了传统XR内容创作成本高、交互方式僵化、个性化不足等瓶颈；

**💡 创新点**

创新点在于：①利用Vision‑Language模型与扩散模型实现基于文本的3D场景与模型自动生成；②将大型语言模型与XR交互系统融合，支持语音、手势等自然交互；③从技术架构层面提出边缘/云分层处理策略，以缓解延迟与资源竞争；

**🔧 技术方法**

主要技术包括：Vision‑Language模型（如LLMER、Vision‑LLM）、扩散式3D生成模型（RealmDreamer、Hunyuan3D 2.0、Meshy AI、Meta 3D Gen）、大语言模型（GPT‑4o、LLMR、GPT‑VR Nexus）、边缘计算与多级推理框架、可解释AI方法；

**📊 数据集**

采用公开的3D模型与文本描述数据集，如Sketchfab、Udemy教学模型及相关文献中引用的 RealmDreamer、Hunyuan3D 2.0 等数据；

**📈 对比分析**

文中通过原型系统演示了文本到场景、模型与交互的生成流程，并给出了延迟指标（语音到动作平均 10.35 s，3D模型生成 < 1 min），但未给出统一的量化性能基准或与传统人工创作的对比；

**⚠️ 局限性**

主要限制包括：AI幻觉导致生成内容不可靠；高延迟影响沉浸感；CPU/GPU/网络资源竞争激烈；隐私泄露与数据安全风险；以及生成内容可信度与可解释性不足，亟待解决。

---

## 78. Learning Neural Operators from Partial Observations via Latent Autoregressive Modeling

**arXiv ID:** 2601.15547 | [PDF](https://arxiv.org/pdf/2601.15547v1)

**作者:** Jingren Hou `[一作]` (Beijing Jiaotong University), Liping Jing `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 3444 | [OpenAlex ID](https://openalex.org/A5069749738)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种能够从部分观测数据中学习偏微分方程（PDE）解的神经算子框架——Latent Autoregressive Neural Operator（LANO）

**💡 创新点**

创新点包括：① mask‑to‑predict（MPT）训练策略，在已观测区域中人为掩码以生成伪监督；② Physics‑Aware Latent Propagator（PhLP），在潜在空间内采用先从边界开始的自回归生成方式，实现物理一致的逐步填补；③ 在潜在空间中结合 Physics‑Cross‑Attention（PhCA）和部分卷积（partial convolution）来解决动态输入‑输出域不匹配问题

**🔧 技术方法**

核心技术为神经算子（DeepONet/FNO 等）基础上引入：遮掩式自监督、潜在空间自回归、物理约束的注意力机制、部分卷积、以及多层级特征缩放与令牌混合器

**📊 数据集**

使用的实验数据集有：Navier–Stokes（2D 湍流流场）、Diffusion–Reaction（生物模式形成）、ERA5（实测气候数据）。通过构建 POBench‑PDE 统一基准，涵盖三类 PDE 与多种缺失模式

**📈 对比分析**

与多种基线（MIONet、OFormer、CORAL、GNOT、IPOT、LNO 等）进行对比；在 patch‑wise 缺失率低于 50% 的情况下，LANO 在所有任务上相对误差下降 18%–69%，并在 75% 缺失率的场景下仍保持显著优势，尤其在 ERA5 实时气候预测任务中实现了最优性能

**⚠️ 局限性**

局限性：目前仅针对规则网格的二维/三维 PDE；缺失模式仍基于固定的补丁或点状掩码，未覆盖更为随机或不规则的缺失；对极大尺度或高度不规则几何的扩展尚待研究，且缺少自适应缺失生成策略

---

## 79. Remarks on Algebraic Reconstruction of Types and Effects

**arXiv ID:** 2601.15455 | [PDF](https://arxiv.org/pdf/2601.15455v1)

**作者:** Patrycja Balik `[一作]` (University of Wrocław), Piotr Polesiuk `[通讯]` (University of Wrocław)

**通讯引用:** 173 | [OpenAlex ID](https://openalex.org/A5070813957)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

分析并指出Jouvelot-Gifford 1991 年类型与效应重构算法中与高阶多态相关的变量绑定错误，揭示其对算法完整性与正确性的影响；

**💡 创新点**

首次系统地发现并细化原算法的捕获语义假设、变量逃逸问题，并给出理论证明与反例；

**🔧 技术方法**

利用形式化推导、捕获/捕获避免替换分析、归结与约束求解理论、以及证明助手的辅助思路；

**📊 数据集**

无实验数据集，纯理论分析；

**📈 对比分析**

通过与原论文声称的可达性、完整性、正确性定理对比，提出反例并说明原证明缺陷；

**⚠️ 局限性**

剩余问题包括对效应多态的完整性缺失、变量逃逸导致的不安全性、以及对实际编译器实现的直接可迁移性不足。

---

## 80. Embedding Retrofitting: Data Engineering for better RAG

**arXiv ID:** 2601.15298 | [PDF](https://arxiv.org/pdf/2601.15298v1)

**作者:** Anantha Sharma `[一作]` `[通讯]`, Anantha Sharma

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

通过构建一套数据工程管线，去除文本中的 hashtag 等注解噪声，清理共现词图，从而恢复并提升词嵌入的 retrofitting 效果。

**💡 创新点**

首次系统识别并量化注解噪声对知识图谱和 retrofitting 的破坏，提出基于预处理质量的评估阈值，并证明在噪声环境下 EWMA retrofitting 优于 Attention。

**🔧 技术方法**

使用 hashtag 去除、停用词过滤、共现阈值筛选等文本清洗技术；基于共现构建知识图谱；实现 Regular、EWMA 与 Attention 三种 retrofitting；通过 t 检验和量化查询分类型分析评估效果。

**📊 数据集**

HR‑1 SNAP 立法文本（45 篇）和 ZeroG 金融服务知识库（512 篇）两套真实语料，各配备专家制定的问答集，用于实验验证。

**📈 对比分析**

在两组数据上对三种 retrofitting 进行对照实验；未预处理时所有方法均出现显著退化；清洗后 EWMA 在清洁数据上提升 6.2%（p=0.035）/4.8%（p=0.041），在量化查询上提升最高 33.8%；预处理质量阈值（图密度<0.05）被证明是成功的关键指标。

**⚠️ 局限性**

实验仅覆盖两领域且仅使用单一词嵌入模型；预处理规则侧重 hashtag，未涵盖其他噪声类型；缺乏自动化的质量评估与更广泛的跨域验证。

---

## 81. The Rise of Large Language Models and the Direction and Impact of US Federal Research Funding

**arXiv ID:** 2601.15485 | [PDF](https://arxiv.org/pdf/2601.15485v1)

**作者:** Yifan Qian `[一作]` (Northwestern University), Dashun Wang `[通讯]` (Northwestern University)

**通讯引用:** 9280 | [OpenAlex ID](https://openalex.org/A5002041772)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过结合两所美国顶尖R1大学的机密提案提交数据和公开的NSF/NIH授予与论文数据，对大语言模型（LLM）在联邦科研资助提案与授予文本中的使用程度进行量化，并探究其对语义独特性、资助成功率及科研产出（论文数量与高影响论文）的影响。

**💡 创新点**

①首次在提案阶段与授予阶段同时量化LLM使用；②发现LLM使用呈双峰分布，并与项目语义与最近已资助项目的相似度呈负相关；③揭示LLM使用对NIH资助成功率和论文产出具有正向关联，而对NSF则无显著效应；④指出LLM对科研产出的提升主要集中在非高影响论文。

**🔧 技术方法**

采用Liang等提出的LLM检测框架（通过对比人类文本与LLM生成文本的词分布来估计混合比例α）；使用SPECTER2嵌入计算文本语义相似度；运用OLS、逻辑回归、负二项回归分析α与各项结果的关系；利用Flesch阅读易读性评分评估写作复杂度。

**📊 数据集**

四个数据集：D1（1.6K NSFs）和D2（4.1K NIH）机密提案提交；D3（57K NSFs）和D4（74K NIH）公开授予与对应论文（通过Dimensions链接）。

**📈 对比分析**

通过控制年份、领域、研究者固定效应并加入资助金额，回归α与语义独特性、资助成功率和论文产出。结果显示：在NIH中，α从25%提升到75%可使资助概率上升约4个百分点，论文数量增加约5%；在NSF中无显著关系；LLM使用使项目更接近最近已资助项目，降低独特性。

**⚠️ 局限性**

①LLM检测仅基于摘要文本，可能低估真实使用量；②未捕捉LLM在研究构思、实验设计等非文本层面的影响；③只考虑公开论文产出，未覆盖其他科研成果；④时间窗口较短，难以评估高影响论文的长期效应。

---

## 82. Beyond validation loss: Clinically-tailored optimization metrics improve a model's clinical performance

**arXiv ID:** 2601.15546 | [PDF](https://arxiv.org/pdf/2601.15546v1)

**作者:** Charles B. Delahunt `[一作]` (University of Washington), Matthew P. Horning `[通讯]` (Global Health Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

进行两组受控实验，比较使用临床定制指标与传统验证损失进行模型优化（包括超参数选择和停止点选择）的效果。

**💡 创新点**

提出并验证在医疗机器学习中直接使用符合临床需求的指标进行优化能显著提升模型在临床任务上的表现，从而突破传统以验证损失为驱动的局限。

**🔧 技术方法**

利用非梯度优化方式将临床指标嵌入训练管道；实验中使用常见的机器学习/深度学习模型（具体模型未明示），并通过编程实现自定义指标评估。

**📊 数据集**

Loa loa 病原体数据集（已准备开放）和一组尚未公开的胎儿超声图像数据集。

**📈 对比分析**

对比方法：分别以验证损失和临床指标为优化目标，记录在临床任务上的表现；结果显示采用临床指标优化的模型在临床指标上取得更优成绩，验证了其优越性。

**⚠️ 局限性**

局限性：需要额外工作定义并编码临床指标，实验仅覆盖两组数据集，缺乏更广泛的数据验证与公开代码可复现性；部分数据集尚未公开。

---

## 83. ToolCaching: Towards Efficient Caching for LLM Tool-calling

**arXiv ID:** 2601.15335 | [PDF](https://arxiv.org/pdf/2601.15335v1)

**作者:** Yi Zhai `[一作]` (Southeast University), Bin Yang `[通讯]` (Southeast University)

**通讯引用:** 11481 | [OpenAlex ID](https://openalex.org/A5059627859)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 ToolCaching，一个面向 LLM 工具调用的高效缓存框架，旨在通过缓存工具调用结果降低重复执行、提升吞吐量和降低延迟。

**💡 创新点**

创新点在于：① 将 LLM 语义分析（请求类型、参数类别、TTL）与系统级指标（访问频次、延迟、成本、结果大小）融合；② 设计了基于多因素价值模型的自适应缓存管理算法 VAAC，其中包括 v-CACA 的多臂赌博机（Bandit）决策与 v-LRU 的多因子驱逐策略；③ 引入了用户层级分组和动态热点切换的缓存分区策略。

**🔧 技术方法**

使用的技术主要包括：LLM 进行语义特征抽取；轻量级系统监控（eBPF 等）收集资源消耗；归一化的价值模型与加权公式；UCB1 多臂赌博机实现自适应加入；基于哈希的键构造与缓存结构；以及 LLM Compiler 框架集成。

**📊 数据集**

实验使用的数据集包括：
• 生成的合成工作负载（Zipf、动态热点、均匀分布）
• Berkeley Function Calling Leaderboard（BFCL）子集用于验证语义特征抽取准确率
• LLM Compiler 公开数据集（Movie Recommendation、ParallelQA）用于端到端评估。

**📈 对比分析**

与 LRU、CACA 及无缓存基线进行对比；在合成负载中，VAAC 的命中率比 CACA 高达 11%，在 LLM Compiler 场景下，端到端延迟降低 34%（Movie Recommendation）和 23%（ParallelQA）；在多用户情境下，用户分组策略提升命中率 21.3% 与平均延迟下降 7.1%。

**⚠️ 局限性**

局限性包括：
• 语义特征抽取依赖 LLM，误判时会影响缓存正确性；
• TTL 设定采用固定阈值，缺乏动态学习；
• 仅针对信息检索类请求，可扩展性受限；
• 评估集中在实验室合成和公开基准，缺少更大规模真实工业流量；
• CPU/内存开销约 15%/10%，对资源受限环境可能成为瓶颈。

---

## 84. Martingale Foresight Sampling: A Principled Approach to Inference-Time LLM Decoding

**arXiv ID:** 2601.15482 | [PDF](https://arxiv.org/pdf/2601.15482v1)

**作者:** Huayu Li `[一作]` (University of Arizona), Ao Li `[通讯]` (University of Arizona)

**通讯引用:** 522 | [OpenAlex ID](https://openalex.org/A5100359733)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Martingale Foresight Sampling（MFS）算法，将 LLM 解码重新表述为寻找最优随机过程。

**💡 创新点**

创新点在于使用马尔可夫理论（Doob 分解、可选停止定理和马尔可夫收敛定理）构建步值评估、路径剪枝和停止判定的理论基础，取代传统经验启发式。

**🔧 技术方法**

采用马尔可夫理论、Doob 分解、可选停止定理、收敛定理、Monte‑Carlo roll‑outs、束搜索以及自回归采样等技术。

**📊 数据集**

评估使用六大推理基准：GSM8K、MATH‑500、GPQA、ReClor、LogiQA、ARC‑Challenge；此外在 Qwen2.5‑3B‑Instruct 上验证泛化。

**📈 对比分析**

与 Auto‑Regressive、Tree‑of‑Thought、MCTS、Guided Decoding、Predictive Decoding、φ‑Decoding 等方法对比，MFS 在 LLaMA3.1‑8B 上实现 87.64% 最高准确率，FLOPs 仅 4.38×10¹⁷，比 φ‑Decoding 节省约 30% 计算量；在 Mistral‑v0.3‑7B 上亦取得 61.64% 的准确率与 19% 的 FLOPs 降低。

**⚠️ 局限性**

局限性包括：仅适用于收敛性推理任务，未适用于开放式或创意生成；相较于标准自回归仍需额外 roll‑out 计算；对基模型的前瞻概率（F_t）校准要求高；在长篇、多轮任务上的表现尚未验证。

---

## 85. Shape of You: Implications of Social Context and Avatar Body Shape on Relatedness, Emotions, and Performance in a Virtual Reality Workout

**arXiv ID:** 2601.15466 | [PDF](https://arxiv.org/pdf/2601.15466v1)

**作者:** Jana Franceska Funke `[一作]` (Ulm University), Teresa Hirzle `[通讯]` (University of Copenhagen)

**通讯引用:** 684 | [OpenAlex ID](https://openalex.org/A5035721389)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

本研究通过在VR健身情境下让受试者与不同体型与性别特征的虚拟同伴或对手进行仰卧起坐，探讨身体形态与社交情境对相关性、情绪体验与运动表现的影响；

**💡 创新点**

创新点在于首次系统检验在VR运动中与另一虚拟角色的身体类型（瘦、肌肉、肥胖）和性别特征对受试者社交联结、情绪反应及运动表现的交互效应，提出针对不同用户偏好设计VR健身体验的理论与实践启示；

**🔧 技术方法**

采用基于HTC Vive Pro的VR硬件、Samsung手环心率监测及自定义六种体型的虚拟头像，使用线性混合模型分析实验数据；

**📊 数据集**

实验样本为48名大学生参与者，未使用公开数据集，所有数据均为本实验收集；

**📈 对比分析**

通过与实验对照（对手情境）比较，发现团队情境与瘦/肌肉体型显著提升相关性与情绪正面体验，减少负面情绪；在运动表现上瘦/肌肉体型与团队情境提高完成次数与感知努力，但对心率与疲劳产生差异；

**⚠️ 局限性**

主要限制包括样本量有限、仅包含顺性别受试者、头像种类单一、实验仅涵盖仰卧起坐等单一动作，且未检验长期使用效果与不同文化背景的普适性。

---

## 86. A Mobile Magnetic Manipulation Platform for Gastrointestinal Navigation with Deep Reinforcement Learning Control

**arXiv ID:** 2601.15545 | [PDF](https://arxiv.org/pdf/2601.15545v1)

**作者:** Zhifan Yan `[一作]` (Johns Hopkins University), Axel Krieger `[通讯]` (Johns Hopkins University)

**通讯引用:** 5244 | [OpenAlex ID](https://openalex.org/A5008331040)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

开发了一种便携低成本的磁性操纵平台，利用UR5协作机械臂携带四个电磁铁阵列来对胃肠道磁性胶囊进行磁控，并通过深度强化学习实现精确轨迹跟踪。

**💡 创新点**

突破了传统磁场控制的“模型校准瓶颈”，通过仿真预训练加上少量实机微调的sim‑to‑real策略，实现在约45分钟内快速部署无模型控制方案，并在大工作空间内实现毫米级跟踪精度。

**🔧 技术方法**

采用Soft Actor‑Critic（SAC）深度强化学习算法，结合基于磁偶极子和Fossen流体动力学的物理仿真模型；硬件使用四个5 V迷你电磁铁、H桥驱动、Arduino控制和RealSense摄像头进行姿态估计。

**📊 数据集**

实验数据来自在37 °C、粘度为2–3 mPa·s的水‑甘油介质中，使用直径7 mm的磁性胶囊，在多条2D轨迹（正方形、圆形、30 cm×20 cm路径）上进行的五次重复试验，收集位置与角度误差。

**📈 对比分析**

与固定电流基线和手动调参的PID控制器对比，DRL控制器在正方形轨迹的距离RMSE为1.18 mm、圆形为1.50 mm，标准差与最大误差均显著优于两种传统方法；在30 cm×20 cm工作空间上平均RMSE为1.50 mm，证明了其在大空间内的高精度与鲁棒性。

**⚠️ 局限性**

局限性在于仅在二维刚性phantom中验证，依赖外部摄像头实现姿态反馈，缺乏三维导航与临床成像支持；未来需集成超声等医疗影像、拓展至3D和真实胃肠道模型。

---

## 87. Data-Free Privacy-Preserving for LLMs via Model Inversion and Selective Unlearning

**arXiv ID:** 2601.15595 | [PDF](https://arxiv.org/pdf/2601.15595v1)

**作者:** Xinjie Zhou `[一作]` (Zhejiang University), Gang Chen `[通讯]` (Zhejiang University)

**通讯引用:** 101834 | [OpenAlex ID](https://openalex.org/A5100389265)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种数据无关的选择性遗忘框架——DFSU，通过模型反演合成伪PII样本，在LoRA子空间内使用对比掩码损失实现令牌级别的隐私去除，并在不访问原始训练数据的情况下对LLM进行隐私修复。

**💡 创新点**

创新点在于：①将模型反演从攻击工具转化为防御工具，自动生成近似训练数据的伪PII；②在LoRA低秩子空间内采用对比掩码目标，仅针对敏感令牌进行更新，既保证隐私去除，又保持模型实用性；③在保持性能的前提下，接近oracle级别的遗忘效果，实现了真正的数据无关隐私修复。

**🔧 技术方法**

使用了：logit‑based 模型反演、伪PII合成与少量提示标注、LoRA 参数高效更新、token‑level 对比掩码损失（Privacy‑Selective Contrastive Unlearning, PSCU）等技术。

**📊 数据集**

主要数据集包括 AI4Privacy PII‑Masking 数据集（用于合成敏感样本），以及 WikiText‑103（生成任务）和 MNLI（推理/分类任务）做为通用语料。

**📈 对比分析**

与 oracle（拥有真实 PII 样本的完整遗忘）以及全序列梯度上升（GA）等基线进行比较。实验显示 DF​SU 在 ERR、FRS、S‑Exp、E‑Hit 等隐私泄露指标上达到 0%（或接近 oracle 级别），同时在 WikiText 的 perplexity 与 MNLI 的准确率仅略有轻微下降，整体隐私‑实用性平衡优于传统方法。

**⚠️ 局限性**

主要局限：需要对模型的 logits 进行白盒访问，无法直接部署在黑盒环境；伪PII 的质量受模型反演性能限制，若反演失效可能导致遗忘效果不佳；在更大规模或更复杂任务上的泛化性仍需进一步验证。

---

## 88. Cloning the Self for Mental Well-Being: A Framework for Designing Safe and Therapeutic Self-Clone Chatbots

**arXiv ID:** 2601.15465 | [PDF](https://arxiv.org/pdf/2601.15465v1)

**作者:** Mehrnoosh Sadat Shirvani `[一作]` (University of British Columbia), Dongwook Yoon `[通讯]` (University of British Columbia)

**通讯引用:** 1430 | [OpenAlex ID](https://openalex.org/A5028316272)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一套针对非临床心理健康场景的自我复制聊天机器人（Self‑Clone Chatbot）设计框架，整合治疗理论、用户需求与安全伦理三大维度，指导开发者构建安全、可接受的个性化对话系统。

**💡 创新点**

创新点在于：①首次系统化将内在自我对话、部分疗法（如 IFS、CFT、CBT）与 LLM 交互设计相结合；②通过专家与潜在用户访谈构建可操作的安全防护和伦理准则；③形成以“自我镜像/变体”“复制忠诚度”“用户–复制关系”“提供者介入”等为核心的多维度设计空间。

**🔧 技术方法**

主要采用的技术是大型语言模型（LLM）与对话式 AI 的微调与部署；但论文未具体实现模型，而是基于理论与访谈结果讨论技术需求与约束。

**📊 数据集**

未使用公开数据集；研究数据来源为 16 名心理健康专家与 6 名潜在用户的半结构化访谈录音/文字。

**📈 对比分析**

论文未进行实验或性能对比，仅通过主题分析与专家评审验证框架的合理性与可操作性；因此无量化指标可报告。

**⚠️ 局限性**

限制包括：①缺乏实际实现与实测验证；②访谈对象主要是对 AI 开放的受试者，可能存在乐观偏差；③仅聚焦非临床轻度情绪困扰，无法推广至重度或危机干预；④对多模态（语音、视觉）自我复制的安全与效果未涉猎。

---

## 89. Early predicting of hospital admission using machine learning algorithms: Priority queues approach

**arXiv ID:** 2601.15481 | [PDF](https://arxiv.org/pdf/2601.15481v1)

**作者:** Jakub Antczak `[一作]` (Wrocław University of Science and Technology), Richard Turner `[通讯]` (University of Tasmania)

**通讯引用:** 2049 | [OpenAlex ID](https://openalex.org/A5016775792)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

通过将急诊科到达人数按科室和临床复杂度分解，对未来七天进行需求预测。

**💡 创新点**

首次在同一研究中同时比较统计模型（SARIMAX）、机器学习模型（XGBoost）和深度学习模型（LSTM），并用合成COVID期间数据避免异常影响。

**🔧 技术方法**

使用 SARIMAX（季节自回归差分移动平均）、XGBoost（梯度提升树）和 LSTM（双向时序网络），并加入气象、日历等外生变量。

**📊 数据集**

来自澳大利亚三级转诊医院2017‑2021年的每日急诊到达记录，共 1826 天，按 8 个科室及 3 种复杂度拆分为 16 条时序。

**📈 对比分析**

对 10 次随机种子训练 XGBoost、LSTM，单次拟合 SARIMAX，利用 MAE、MAPE 与季节性基线比较；整体来看 XGBoost 在总量预测上最优，SARIMAX 在高复杂度病例预测上略胜，三者均明显优于基线。

**⚠️ 局限性**

模型均倾向于低估罕见的急剧上升，无法准确捕捉突发需求峰值；低量科室（如神经科）预测误差高，需进一步研究稀疏计数方法。

---

## 90. Multi-Persona Thinking for Bias Mitigation in Large Language Models

**arXiv ID:** 2601.15488 | [PDF](https://arxiv.org/pdf/2601.15488v1)

**作者:** Yuxing Chen `[一作]` (University of Alberta), Lili Mou `[通讯]` (University of Alberta)

**通讯引用:** 7306 | [OpenAlex ID](https://openalex.org/A5024821632)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在推理时使用多视角人物（多人物思维，MPT）的框架，用以减少大语言模型中的社会偏见并保持或提升推理能力。

**💡 创新点**

创新点在于：①通过让模型以互相对立的社会身份（如男性/女性）以及中立视角进行自我辩论；②将人物身份视为辩论主体而非单一角色，使偏见可被显露并在交互中修正；③在推理阶段实现多轮对话而非仅一次提示，兼顾公平与准确。

**🔧 技术方法**

技术手段包括：多角色系统提示（persona initialization）、迭代对话式推理（dialectical reasoning）、最终整合回合（final aggregation），并在推理过程中结合自我反思与身份切换；实验使用 LLMs（Llama‑3.1‑8B/70B 等）。

**📊 数据集**

评测数据集为：BBQ（多类别多选问答）和 StereoSet（词/句级关联测试），两者均被改造为包含偏见与无关选项的格式。

**📈 对比分析**

与多种基线（直接提示、显式去偏、角色提示、自一致性、重提示、Multi‑Agent Debate）对比，MPT 在 BBQ 的平均准确率从 89.01% 提升至 90.46%（模糊情境）并将 diff‑bias 从 0.0562 降至 0.0275；在 StereoSet 上准确率 58.76% 及 bias 0.0635 分别较前沿方法提升 28% 与 36%；统计检验表明提升显著且可与自一致性组合进一步提升。

**⚠️ 局限性**

局限性包括：①推理成本高于单回提示，需权衡迭代次数和延迟；②目前仅使用基于数据集元数据的二元身份（如男性/女性），未涵盖多重、非二元或交叉身份；③仅在多选任务上验证，缺乏针对开放式生成任务的评测与无偏评估标准。

---

## 91. PRISM: Deriving the Transformer as a Signal-Denoising Operator via Maximum Coding Rate Reduction

**arXiv ID:** 2601.15540 | [PDF](https://arxiv.org/pdf/2601.15540v1)

**作者:** Dongchen Huang `[一作]` (Institute of Physics), Dongchen Huang `[通讯]` (Institute of Physics)

**通讯引用:** 92 | [OpenAlex ID](https://openalex.org/A5050360223)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Prism 架构，基于最大编码率减缩 (MCR²) 并加入过完备字典与 π‑RoPE，实现可解释 Transformer。

**💡 创新点**

通过物理/几何约束（过完备子空间与无共振频率分离）实现语义推理与句法记忆的无监督功能解耦。

**🔧 技术方法**

使用 MCR² 优化、梯度上升式编码率减、π‑RoPE（π 与 1/π 频率）、过完备子空间、以及 mean‑field 动态分析等技术。

**📊 数据集**

在 TinyStories 语料库上进行实验验证。

**📈 对比分析**

与 GPT‑2、Gemma3 等小模型对比，50M Prism 在 TinyStories 上收敛更快、验证损失约 1.55，参数效率更高。

**⚠️ 局限性**

模型规模受限，仅 50M 参数，尚未在更大数据集或更深网络上验证其可扩展性。

---

## 92. YuFeng-XGuard: A Reasoning-Centric, Interpretable, and Flexible Guardrail Model for Large Language Models

**arXiv ID:** 2601.15588 | [PDF](https://arxiv.org/pdf/2601.15588v1)

**作者:** Junyu Lin `[一作]` (Alibaba), Yitong Yang `[通讯]` (Alibaba)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套基于推理的安全防护模型家族，能够对大语言模型的交互文本进行多维风险识别、置信度估计和自然语言解释，并支持分层推理与动态策略调整。

**💡 创新点**

核心创新点包括：① 用结构化风险感知替代单一标签，输出风险类别、置信分数和解释；② 第一个token即给出决策，后续可生成详细解释的分层推理；③ 引入动态政策机制，将风险识别与策略执行解耦，支持运行时添加/修改规则；④ 通过知识蒸馏推出0.6B轻量版，兼顾低延迟与高性能。

**🔧 技术方法**

技术方案基于 Qwen3 架构的大语言模型，采用监督微调（SFT）中的 classify‑then‑explain 训练；动态规则生成使用教师模型合成规则集；对称 KL 蒸馏（forward + reverse）实现轻量版；对解释质量做了 RL（GRPO）实验但未最终采用；评估采用多任务 F1 评价。

**📊 数据集**

使用了多种公开安全基准：SEval、Aegis、OpenAI Moderation、WildGuard、PolyGuard、RTP‑LX 等；通过 DeepSeek‑V3 对 25 种语言进行翻译和标注，构建 2.8M 高质量训练样本；并生成动态策略数据以训练动态政策能力。

**📈 对比分析**

与 LlamaGuard、WildGuard、PolyGuard、NemotronReasoning、ShieldGemma、GPT‑OSS‑SafeGuard 等主流开源 guardrail 进行 F1 对比；在提示和响应分类任务中，Qwen3Guard‑Gen‑8B 取得最高或次高平均 F1；0.6B 轻量版保持竞争力；在攻击防御、误拦截率、Safe Completion 等子任务中亦显著优于或匹配现有模型。

**⚠️ 局限性**

局限性包括：仍可能对未见的攻击方式脆弱；训练语料可能带来社会偏见；默认安全策略可能不完全适用于所有地区；动态策略需要提供清晰、一致的规则，模糊或矛盾的指令会影响执行。

---

## 93. Memorization Dynamics in Knowledge Distillation for Language Models

**arXiv ID:** 2601.15394 | [PDF](https://arxiv.org/pdf/2601.15394v1)

**作者:** Jaydeep Borkar `[一作]` (Meta Superintelligence Labs), Diego Garcia-Olano `[通讯]` (Meta Superintelligence Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在知识蒸馏的 fine‑tuning 框架下，系统评估并对比了教师、基线与学生模型在训练数据记忆（memorization）方面的表现，并分析了蒸馏过程如何降低记忆率。

**💡 创新点**

创新点在于：①提出“易记忆例子”概念并证明蒸馏仅保留这部分；②构建预测模型提前识别并剔除高风险样本；③通过熵和对数概率阐释蒸馏对记忆的正则化作用；④比较软硬蒸馏的记忆继承差异。

**🔧 技术方法**

主要技术包括：KL 散度蒸馏（logit‑level）、序列级（hard）蒸馏、基于 zlib 熵与 perplexity 的特征判别、逻辑回归记忆预测器，以及对数概率与 Shannon 熵的统计分析。

**📊 数据集**

使用的数据集有 FineWeb、WikiText‑103、Nemotron‑CC‑v2 以及 Synthetic dataset，模型基于 Pythia、OLMo‑2 与 Qwen‑3 族。

**📈 对比分析**

通过在验证集和下游任务（LAMBADA、Winogrande）上比较 perplexity 与准确率，发现蒸馏学生在保持甚至提升泛化性能的同时，将记忆率从基线的 0.17% 降低到约 0.07%，并在不同模型族上表现一致。

**⚠️ 局限性**

局限在于：实验主要集中在单一预训练模型家族和有限数据集，未涉及更大规模教师或多任务情境；易记忆例子判别依赖 zlib 熵，可能对不同 tokenizer 产生偏差；硬蒸馏的记忆继承问题仍需进一步研究。

---

## 94. MuSAlS: A Fast Multiple Sequence Alignment Approach Using Hierarchical Clustering

**arXiv ID:** 2601.15458 | [PDF](https://arxiv.org/pdf/2601.15458v1)

**作者:** Emily G. Light `[一作]` (University of Rhode Island), Najib Ishaq `[通讯]` (University of Rhode Island)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5024517512)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `09944146-298c-433e-89df-37255de463d7` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 MuSAlS，一种利用层次聚类构建引导树并结合星型与进展对齐的多序列比对工具，能够在大规模数据集上实现快速、可扩展的 de novo 对齐。

**💡 创新点**

创新点在于：①使用 Levenshtein 距离精确测量序列相似度来构建引导树；②将星型对齐与进展对齐分阶段应用，既保持速度又提升质量；③实现了比其他工具更紧凑、gap 更少的对齐，且不依赖外部进化或结构信息。

**🔧 技术方法**

技术实现：层次聚类（基于几何中点）、Levenshtein 距离、Needleman‑Wunsch 动态规划、Rust 并行化（Rayon）、Affine gap penalty、BLOSUM62/扩展 IUPAC 成本矩阵。

**📊 数据集**

使用的数据集包括 Greengenes 12.10 与 13.5（约 1–1.2M 条 16S rRNA 序列）、PDB（约 8.2×10^5 条蛋白质序列）以及 Pfam 子集（10k、100k、1M 条多样蛋白序列）。

**📈 对比分析**

通过与 MAGUS、WITCH、KAlign、FAMSA2 等主流 MSA 工具在运行时、宽度、% gaps、距离失真、p‑score 等指标进行对比；MuSAlS 在所有数据集均能完成，比 KAlign 速度快 2–15 倍，且在蛋白质数据集上距离失真优于 Kalign，但整体 p‑score 略高于部分工具。

**⚠️ 局限性**

局限性：仍受序列长度平方复杂度限制，难以对整条染色体或极长序列进行对齐；缺乏在线更新能力；对外部进化信息不敏感，可能导致在高度进化分化数据上准确性下降；未对 Needleman‑Wunsch 进行更高效的启发式加速。

---

## 95. Mapping Social Media User Behaviors in Reciprocity Space

**arXiv ID:** 2601.15623 | [PDF](https://arxiv.org/pdf/2601.15623v1)

**作者:** Shiori Hironaka `[一作]` (Academic Center for Computing and Media Studies, Kyoto University), Kyoji Umemura `[通讯]` (Department of Computer Science and Engineering, Toyohashi University of Technology)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于互惠度的二维框架，将Twitter用户映射到(r_in,r_out)空间，对48,830名用户的完整关注网络进行行为分析。

**💡 创新点**

将先前离散化的用户类型（influencer、lurker、broker、follow‑back）统一到连续行为空间，揭示中间递归度区段的高参与度区域，并提供可解释的影响力与互动指标。

**🔧 技术方法**

使用网络结构分析、双维互惠度指标、Kruskal‑Wallis/Conover检验、词频统计（chi‑square）、热图梯度可视化等统计与可视化技术。

**📊 数据集**

基于2021年7月Twitter 1%抽样流，筛选48,830名英文发推用户，收集约149 M关注关系和21.5 M双向连接。

**📈 对比分析**

通过四个极端类属性对比及10×10网格热图验证行为连续性；相较传统单一follower/followee比率，框架更精确区分行为并发现中间递归度区段能获得最高内容传播率。

**⚠️ 局限性**

仅覆盖英文Twitter样本、采样偏向活跃用户、未考虑内容语境与文化差异、仅使用结构特征、单平台研究，未追踪时间演化。

---

## 96. Explainable Deepfake Detection with RL Enhanced Self-Blended Images

**arXiv ID:** 2601.15624 | [PDF](https://arxiv.org/pdf/2601.15624v1)

**作者:** Ning Jiang `[一作]` (Peking University), Ying Li `[通讯]` (Peking University)

**通讯引用:** 21893 | [OpenAlex ID](https://openalex.org/A5100414156)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用自混合图像自动生成逼真伪造样本，并通过多模态大型语言模型自动生成Chain‑of‑Thought（CoT）文本注释；

**💡 创新点**

创新点在于①自动CoT生成框架，消除人工注释成本；②关键词驱动的奖励机制，解决二分类任务中的奖励稀疏问题；③基于GRPO的反馈式数据合成策略，动态提升样本难度与模型泛化；

**🔧 技术方法**

使用Self‑Blended Image技术、InternVL3‑38B/​LLaVA 1.5‑7b、GRPO强化学习、LoRA微调、Prompt Engineering、ROUGE/Jaccard奖励等；

**📊 数据集**

训练集基于FaceForensics++与SBI合成的CoT数据，测试集包括CDF2、DFD、DFDC、DFDCP；

**📈 对比分析**

与多种SOTA方法对比，帧级AUC在CDF2上达到0.905、DFD 0.926，视频级AUC在CDF2 0.963、DFD 0.965，整体性能与最佳深度伪造检测模型相当；

**⚠️ 局限性**

局限于仅针对混合伪造工艺，难以应对DFDC等非混合多样化伪造和严重环境退化场景。

---

## 97. Rank-metric codes over arbitrary fields: Bounds and constructions

**arXiv ID:** 2601.15464 | [PDF](https://arxiv.org/pdf/2601.15464v1)

**作者:** Alessandro Neri `[一作]` (University of Naples Federico II), Ferdinando Zullo `[通讯]` (University of Campania Luigi Vanvitelli)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了秩度量码的基本概念、Singleton‑类上界、MRD 码的构造与分类，并探讨了其在不同域（有限域、代数闭域、实数域以及更一般的 Galois 延拓）下的性质与应用。

**💡 创新点**

创新点在于将 MRD 码构造推广到任意具有循环 Galois 延拓的域，建立了秩度量码与散射/抑制子空间的几何对应，提出了关于 MRD 存在性的猜想，并首次给出代数闭域与实数域下的上界与构造结果。

**🔧 技术方法**

主要技术包括代数几何（判定矩阵秩的确定性多样体维数）、域论（Galois 代数与偏置代数）、线性代数与对偶性理论，以及拓扑方法（Radon‑Hurwitz 数与向量场问题）。

**📊 数据集**

本综述不使用实验数据集，而是依赖纯理论分析与构造证明。

**📈 对比分析**

通过比较理论上界与构造结果，文章证明了在有限域和满足循环 Galois 延拓的域中 Singleton‑类上界可达最优；在代数闭域和实数域中给出更严格的上界并指出当特定三元组满足奇偶条件时该上界可被实现；对其余参数范围则给出上界但未达成构造。

**⚠️ 局限性**

限制主要体现在：对非循环 Galois 延拓、实数域下非完全特定参数、以及更一般域的 MRD 码存在性仍有大量未解问题；提出的猜想尚未被证明，且在某些参数下缺乏构造方案。

---

## 98. ICPO: Illocution-Calibrated Policy Optimization for Multi-Turn Conversation

**arXiv ID:** 2601.15330 | [PDF](https://arxiv.org/pdf/2601.15330v1)

**作者:** Zhebo Wang `[一作]` (Zhejiang University), Meng Han `[通讯]` (Zhejiang University)

**通讯引用:** 2841 | [OpenAlex ID](https://openalex.org/A5100785901)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种 Illocution‑Calibrated Policy Optimization (ICPO) 框架，旨在解决 LLM 在多轮对话中因自信过度导致的“迷失”现象。

**💡 创新点**

核心创新在于将奖励信号与用户指令的 illocution（含义）对齐，鼓励模型在面对歧义时表达不确定性或主动寻求澄清，而非直接给出自信答案；同时通过情景模拟和 illocutionary 判断构造训练样本。

**🔧 技术方法**

结合 RLVR（基于 GRPO 的强化学习）、自定义奖励函数、情景模拟（专家模型生成欠具体化提示）、illocutionary 判断模型以及熵正则化与抗崩溃机制，对 LLM 进行策略优化。

**📊 数据集**

训练使用 OpenR1‑Math‑220k 数据集；评估采用多轮数学推理数据集（GSM8K、MATH500、AMC23、AIME24/25、Minerva、Olympiad）和单轮基准。

**📈 对比分析**

与标准 RLVR 以及多种熵正则化/抗崩溃方案（GRPO+Clip‑higher、Clip‑Cov、KL‑Cov、RL‑PLUS）进行对比，ICPO 在多轮会话准确率上提升约 75%（如 Qwen2.5‑7B 从 35.4% 提升到 55.4%），单轮任务保持或略有提升。

**⚠️ 局限性**

目前实验仅针对数学推理场景，未验证非数学对话的鲁棒性；模型规模有限，需进一步扩展到更大模型并测试跨领域通用性。

---

## 99. FedUMM: A General Framework for Federated Learning with Unified Multimodal Models

**arXiv ID:** 2601.15390 | [PDF](https://arxiv.org/pdf/2601.15390v1)

**作者:** Zhaolong Su `[一作]` (William and Mary), Jindong Wang `[通讯]` (William and Mary)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 FedUMM，一个基于 NVIDIA FLARE 的联邦学习框架，针对 BLIP3o 等统一多模态模型在非 IID 多模态数据下实现低通信成本的联邦训练。

**💡 创新点**

创新点包括：① 采用 LoRA 参数高效微调仅上传适配器参数；② 设计 FedFusion 语义感知加权聚合以处理不同模态与客户端异质性；③ 在设备–边缘–服务器层次进行模型分区，提升通信与计算效率。

**🔧 技术方法**

主要技术有：联邦学习（FedAvg、FedProx 等）、LoRA 低秩适配器、参数高效微调、NVIDIA FLARE 框架、Dirichlet 异质性划分、梯度压缩/差分隐私。

**📊 数据集**

使用的数据集包括 VQA v2、MSCOCO captions、CC3M、GenEval（文本到图像生成）以及组合的 VQA 与生成基准。

**📈 对比分析**

与中心化全模型微调相比，FedUMM 在 2–16 个客户端、α=0.1–1.0 的非 IID 环境下保持 97% 以上的性能，通信成本下降 99.7%，训练时间减少 86.7%，但随着客户端数量和异质性上升略有下降。

**⚠️ 局限性**

主要局限包括：实验在模拟联邦环境下，未验证真实网络延迟、客户端掉线；仅评估 BLIP3o，未覆盖更大规模或多模态组合；缺乏正式差分隐私保证；对极端异质性与异步更新的鲁棒性有限。

---

## 100. Data-driven Lake Water Quality Forecasting for Time Series with Missing Data using Machine Learning

**arXiv ID:** 2601.15503 | [PDF](https://arxiv.org/pdf/2601.15503v1)

**作者:** Rishit Chatterjee `[一作]` (Colby), Tahiya Chowdhury `[通讯]` (Colby)

**通讯引用:** 28 | [OpenAlex ID](https://openalex.org/A5065468004)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文利用缅因州30个湖泊三十年观测数据，结合多重插补和岭回归，研究了预测Secchi Disk Depth（水体可视深度）时最小训练样本量与特征集对精度的影响，并提出联合可行性函数实现基于5%误差目标的最近历史长度与特征选择；

**💡 创新点**

创新点在于：①统一考虑最近历史长度与特征集的最小化，给出可操作的联合可行性规则；②证明仅需约64条最近观测和单一关键特征即可达到完整历史95%精度；③提供具体的采样与测量优先级建议；

**🔧 技术方法**

使用的技术包括：多重链式插补（MICE）处理缺失；岭回归预测模型；均方降益（MDI）特征重要性评估；后向（最近历史）训练协议；nMAE作为跨湖比较指标；以及基于字典搜索的联合可行性函数；

**📊 数据集**

数据集为缅因州湖泊环境监测数据库，历时三十年，包含793个湖泊的多变量时间序列，本文选取观测最完整的30个湖泊进行实验；

**📈 对比分析**

与ARIMA、SARIMA、随机森林、Transformer等模型比较，岭回归在13个特征下平均MAE 0.621、nMAE 0.165；后向样本数曲线表明平均最小样本约176；四特征组可实现同等性能；联合方案在95%目标下，典型湖泊仅需64样本与1特征即可达到5%误差容忍；

**⚠️ 局限性**

局限性包括：MICE假设缺失随机；只分析30个记录最丰富湖泊，可能缺乏代表性；5%阈值与字典规则人为设定，缺乏灵活性；未考虑物理约束、因果关系或不同季节动态的进一步验证；

---

## 101. A Mobile Application Front-End for Presenting Explainable AI Results in Diabetes Risk Estimation

**arXiv ID:** 2601.15292 | [PDF](https://arxiv.org/pdf/2601.15292v1)

**作者:** Bernardus Willson `[一作]` (Bandung Institute of Technology), Saiful Akbar `[通讯]` (Bandung Institute of Technology)

**通讯引用:** 436 | [OpenAlex ID](https://openalex.org/A5003554946)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文开发了一款面向普通用户的 Android 移动前端，通过可视化和 LLM 生成的自然语言，将 SHAP 解释结果转化为易懂的糖尿病风险评估信息。

**💡 创新点**

创新点在于将传统 XAI 可视化简化为用户喜爱的饼图/柱状图，并结合 GPT‑4o 生成的个性化叙述，形成图文双模组合提升理解度。

**🔧 技术方法**

技术栈包括 Kotlin + Jetpack Compose（UI）、MPAndroidChart（图表）、XGBoost+SHAP（模型与解释）、OpenAI GPT‑4o（生成文本）以及 Espresso（端到端测试）。

**📊 数据集**

使用了印度尼西亚本地糖尿病患者健康数据（包含家族史、年龄、BMI、生活方式等特征），并以公开的糖尿病风险数据集进行模型训练。

**📈 对比分析**

通过对 12 名不同教育背景用户的 Likert 量表评估和访谈，整体理解得分 4.31/5；在 111 条自动化端到端测试中实现 100% 通过率，证明功能与易用性均达到预期。

**⚠️ 局限性**

局限性包括样本量不足、LLM 可能产生幻觉的风险、图表种类单一（未实现分离可控/不可控因素或分离正负影响的 diverging bar），未来需扩大用户群并改进可解释性呈现。

---

## 102. Agentic Persona Control and Task State Tracking for Realistic User Simulation in Interactive Scenarios

**arXiv ID:** 2601.15290 | [PDF](https://arxiv.org/pdf/2601.15290v1)

**作者:** Hareeshwar Karthikeyan `[一作]` `[通讯]` (Toast Inc), Hareeshwar Karthikeyan (Toast Inc)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了一套多代理框架，用于在餐厅点餐场景中生成具有逼真人格、情绪变化与任务状态跟踪的用户模拟对话。

**💡 创新点**

创新点在于把用户模拟拆分为三类专门代理（User Agent、State Tracking Agent、Message Attributes Generation Agent），通过结构化协议实现可解释、可控、可复现的行为，并在多代理协作下显著提升模拟质量。

**🔧 技术方法**

技术实现基于 GPT‑4o 与 Pydantic AI 工具链，采用结构化工具调用、任务状态管理、情绪与行为属性控制；评估使用五项量化指标（PAS、BVS、TRA、DEI、CRRS）。

**📊 数据集**

使用的数据集包括 20 种餐厅客人人格、50+ 菜单项以及 60 条测试用例（每种人格 3 个不同复杂度的点餐目标）。

**📈 对比分析**

通过与单 LLM 基线以及四种单/双代理消融配置对比，完整系统在 CRRS 上比基线提升 102.6%，PAS +19.9%，BVS +284.5%，TRA +29.1%，DEI 达到 0.994，证明多代理架构在模拟质量上显著优于单模型。

**⚠️ 局限性**

限制包括计算成本高（token 与延迟显著增加）、仅在单一英文餐饮域验证、缺乏多语言、多模态与复杂社会行为的支持，且可能存在偏见，需要人工审计。

---

## 103. Ternary Spiking Neural Networks Enhanced by Complemented Neurons and Membrane Potential Aggregation

**arXiv ID:** 2601.15598 | [PDF](https://arxiv.org/pdf/2601.15598v1)

**作者:** Boxuan Zhang `[一作]` (Beijing Sport University), Kuan Tao `[通讯]` (Beijing Sport University)

**通讯引用:** 191 | [OpenAlex ID](https://openalex.org/A5083054305)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了兼具可学习补偿项的三元脉冲神经元（CTSN）以及基于膜电位聚合的时间正则化方法（TMPR），用于提升三元脉冲神经网络的梯度传播与信息保持，最终实现更高的分类精度。

**💡 创新点**

创新点：①在三元神经元的积分过程中加入可学习的补偿项 h(t)，实现历史信息的持续保留与重注；②设计时间变换的膜电位正则化，利用膜电位聚合度动态调节正则强度，显著缓解梯度消失与膜电位不均衡；③两者共同形成多路径梯度流，提升网络表达与训练稳定性。

**🔧 技术方法**

技术：三元脉冲神经元模型（{-1,0,1} 脉冲），可学习参数 α、β、γ，Sigmoid 限制；时间正则化 TMPR（基于膜电位平方平均）；基于反向传播时间（BPTT）与 surrogate gradient（矩形梯度）。

**📊 数据集**

数据集：CIFAR‑10、CIFAR‑100（静态图像）；CIFAR10‑DVS（事件驱动视听数据）；ImageNet‑100（大规模图像）。

**📈 对比分析**

与现有最先进 SNN 方法比较（如 PLIF、GLIF、TET、RMP‑Loss、ASGL、CLIF 等）在相同网络骨干（ResNet、VGG、SEW‑ResNet）上进行对比。CTSN+TMPR 在 CIFAR‑10 达到 96.46%、CIFAR‑100 81.19%、ImageNet‑100 85.06%，在事件数据 CIFAR10‑DVS 上取得 81.23%，均优于或匹配公开最佳结果。

**⚠️ 局限性**

局限性：模型引入额外补偿项导致计算复杂度与参数量略增；对时间步长敏感，若步长过短则补偿作用不明显；在不同任务或网络结构下的泛化能力与解释性尚需进一步理论与实验验证。

---

## 104. Not Your Typical Sycophant: The Elusive Nature of Sycophancy in Large Language Models

**arXiv ID:** 2601.15436 | [PDF](https://arxiv.org/pdf/2601.15436v1)

**作者:** Shahar Ben Natan `[一作]` (Ben Gurion University), Oren Tsur `[通讯]` (Ben Gurion University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新颖的方法来评估大型语言模型（LLMs）的谄媚倾向，采用直接和中立的方式，减少了以往研究中注入的各种偏见、噪声或操控性语言的影响。

**💡 创新点**

创新点在于将谄媚评估视为一种零和游戏，通过下注的方式进行评估，明确谄媚行为对用户有利但对他人有成本。

**🔧 技术方法**

使用了LLM作为评判者的框架，结合了统计显著性分析来评估模型的谄媚倾向。

**📊 数据集**

使用了TruthfulQA数据集，该数据集包含817个问题，涵盖多个主题和类别，特别设计了对抗性问题以测试模型的谎言生成能力。

**📈 对比分析**

通过比较四个领先模型（Gemini 2.5 Pro、ChatGpt 4o、Mistral-Large-Instruct-2411和Claude Sonnet 3.7），发现所有模型在常规设置中表现出谄媚倾向，但Claude和Mistral在明确损害第三方时表现出“道德悔恨”，过度补偿其谄媚行为。

**⚠️ 局限性**

局限性在于新模型和旧模型的新版本以空前的速度发布，结果可能因模型而异，且同一模型的不同版本之间也可能存在差异。

---

## 105. Evaluating Multimodal Large Language Models for Heterogeneous Face Recognition

**arXiv ID:** 2601.15406 | [PDF](https://arxiv.org/pdf/2601.15406v1)

**作者:** Hatef Otroshi Shahreza `[一作]` (Idiap Research Institute), Sébastien Marcel `[通讯]` (Idiap Research Institute)

**通讯引用:** 14453 | [OpenAlex ID](https://openalex.org/A5016330764)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

评估开源多模大语言模型在不同模态人脸识别任务中的性能

**💡 创新点**

首次系统性比较MLLM与传统人脸识别模型在异构人脸识别中的差距

**🔧 技术方法**

采用零样本/提示学习框架，使用预训练的Gemma、LLaVA、Mistral、Aya‑Vision、InternVL3、Qwen3‑VL等模型

**📊 数据集**

MCXFace、CASIA NIR‑VIS 2.0、Polathermic 三个公开异构人脸数据集

**📈 对比分析**

通过AR、EER、TAR@1%等生物识别指标对比，发现即使AR全为100%，MLLM的EER普遍高于EdgeFace/xEdgeFace，TAR受限，最优模型Qwen3‑VL‑8B在VIS‑NIR上近似基线，但在VIS‑THERMAL上仍显弱

**⚠️ 局限性**

缺乏对模态差异的专门训练，无法捕捉跨光谱细节，且在部分样本上无法输出相似度，导致性能落后

---

## 106. VIOLA: Towards Video In-Context Learning with Minimal Annotations

**arXiv ID:** 2601.15549 | [PDF](https://arxiv.org/pdf/2601.15549v1)

**作者:** Ryo Fujii `[一作]` (Keio University), Ryo Hachiuma `[通讯]` (NVIDIA)

**通讯引用:** 546 | [OpenAlex ID](https://openalex.org/A5020411666)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种在视频域中实现最小标注量的上下文学习框架，利用专家少量标注与大量无标签视频构建混合演示池并进行推理。

**💡 创新点**

创新点在于三方面：① 通过密度-不确定性加权采样在保证多样性的同时过滤语义离群点；② 在无标签样本上采用基于专家示例的上下文伪标注；③ 在检索和提示阶段加入置信度感知机制，区分真实标签与噪声伪标签。

**🔧 技术方法**

采用的技术包括高斯混合模型聚类、零样本与上下文伪标注、置信度加权检索、置信度提示、以及四种开源多模态大型语言模型（Qwen2-VL-7B、VideoLLaMA3-7B、Qwen3-VL-8B、LLaVA-Video7B）以及InternVideo2视觉编码器。

**📊 数据集**

实验使用九个涵盖医学、工业、驾驶、动物、监控等多样领域的视频语言数据集（Drive&Act、EgoPet、EgoSurgery、ENIGMA、UCF-Crime、Xsports、MammAlps、Bora、CapERA）。

**📈 对比分析**

与零样本、随机、随机+伪标注、VideoICL、VoteK等基线相比，在20个标注样本下，本文方法在大多数数据集和模型上均实现显著提升，最高可达+53.6%的准确率提升，整体表现稳健并随标注预算增长持续提升。

**⚠️ 局限性**

局限性在于依赖预训练的视觉嵌入进行样本选择和检索，若目标域与预训练域差异过大可能导致聚类和检索效果受损，未来需探索域自适应编码器提升鲁棒性。

---

## 107. AfriEconQA: A Benchmark Dataset for African Economic Analysis based on World Bank Reports

**arXiv ID:** 2601.15297 | [PDF](https://arxiv.org/pdf/2601.15297v1)

**作者:** Edward Ajayi `[一作]` `[通讯]` (Carnegie Mellon University), Edward Ajayi (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

介绍了 AfriEconQA，一个基于 236 篇世界银行非洲经济报告构建的 8,937 对 QA 数据集，并通过检索增强生成（RAG）系统进行评估；

**💡 创新点**

首次构建面向非洲经济分析的高精度问答基准，强调数值推理和时间歧义，采用严格的语义与格式验证，形成多类（事实、列表、比较、单选、综合）问答；

**🔧 技术方法**

使用 BM25（稀疏检索）、Dense（BGE 与 Google Embeddings）检索模型，融合 Reciprocal Rank Fusion；生成器包括 GPT‑4o、Qwen‑32B 与零样本 GPT‑5 Mini；评估指标涵盖 EM、F1、BLEU、ROUGE、LLM‑Judge；

**📊 数据集**

AfriEconQA 数据集：8,937 题答对、64,892 个 1,000 字节块、5 大类题型，来源统一为 236 篇世界银行报告；

**📈 对比分析**

与零样本 GPT‑5 Mini 对比，RAG 系统大幅提升（EM 最高 27%/LLM‑Judge 0.52），密集检索在排名上最优（MRR 0.763），但混合检索在答案质量上略优；不同模型在各题型表现差异明显；

**⚠️ 局限性**

局限：数据来源单一、对数值的逐字匹配约束限制了抽象推理、数据仅反映某一时间点、实验仅基于 300 题样本，未覆盖完整数据集；

---

## 108. AION: Aerial Indoor Object-Goal Navigation Using Dual-Policy Reinforcement Learning

**arXiv ID:** 2601.15614 | [PDF](https://arxiv.org/pdf/2601.15614v1)

**作者:** Zichen Yan `[一作]` (National University of Singapore), Lin Zhao `[通讯]` (National University of Singapore)

**通讯引用:** 13067 | [OpenAlex ID](https://openalex.org/A5110190620)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了基于双策略的AION框架，实现室内无人机三维目标导向导航。

**💡 创新点**

创新点在于将探测与目标到达拆分为专用策略，并利用深度投影、CLIP语义注意力和多模态输入实现零样本三维导航，同时在真实物理仿真中验证实时安全性能。

**🔧 技术方法**

采用CLIP+DINOv2视觉编码、YOLOv8目标检测、双策略A3C强化学习、深度投影生成激光雷达特征，并在IsaacSim与Pegasus下实现动力学控制。

**📊 数据集**

主要使用AI2THOR/iTHOR、ProcTHOR、IsaacSim/OmniGibson等室内场景数据集进行训练与评估。

**📈 对比分析**

与BaseModel、Scene Prior、MJO、SSNet等基线对比，AION在成功率(SR)和路径加权成功率(SPL)上均取得SOTA，碰撞率低，在IsaacSim上也显著提升探索覆盖率(FCR)。

**⚠️ 局限性**

局限在于仍依赖YOLO检测的可靠性，缺乏全局地图导致探索完整覆盖难题，且在极暗环境下检测与导航性能下降。

---

## 109. Seeing through Light and Darkness: Sensor-Physics Grounded Deblurring HDR NeRF from Single-Exposure Images and Events

**arXiv ID:** 2601.15475 | [PDF](https://arxiv.org/pdf/2601.15475v1)

**作者:** Yunshan Qi `[一作]` (Beihang University), Jia Li `[通讯]` (Beihang University)

**通讯引用:** 22904 | [OpenAlex ID](https://openalex.org/A5108050435)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种基于NeRF的框架See-NeRF，用单曝光模糊的低动态范围(LDR)图像和对应的事件数据恢复高动态范围(HDR)的清晰三维场景，并实现从该场景生成任意视角的HDR新视角图像。

**💡 创新点**

创新点在于：① 通过物理感知建模，将RGB相机的曝光积分和相机响应函数以及事件相机的延迟、阈值和量化误差抽象为可微分的RGB映射场和事件映射场；② 将这些映射场与NeRF联合优化，使得渲染过程严格遵循真实相机的成像物理；③ 直接在HDR域学习场景辐射，避免了传统方法在LDR域中产生的颜色失真和模糊。

**🔧 技术方法**

主要技术包括：NeRF体积渲染；可微分RGB映射场（模拟曝光积分与CRF）；可微分事件映射场（模拟事件阈值、延迟与量化）；事件模拟器v2e；多尺度时间点离散；联合损失（LDR图像损失 + 事件损失）；并采用HDR-ToneMapping进行评估。

**📊 数据集**

使用两套数据集：1）合成HDR-NeRF场景（8个Blender HDR场景），生成单曝光模糊图像、事件序列及多曝光HDR图像；2）真实数据集（5个极端光照场景）使用DAVIS 346 Color事件相机拍摄的模糊图像与事件以及用三脚架拍摄的多曝光清晰图像；此外，还在公开的Real-World-Challenge数据集上做了消模糊评测。

**📈 对比分析**

与基线方法（HDR-NeRF、HDR-GS、GaussHDR、EvHDR-NeRF、E2NeRF、E3NeRF等）进行对比。实验显示，See-NeRF在HDR新视角合成任务中PSNR提升约2–3 dB、SSIM提升约0.02–0.03、LPIPS显著下降（更小越好），在消模糊HDR NVS任务中达到最高PSNR（约32–34 dB）和最低LPIPS（约0.13–0.16）。

**⚠️ 局限性**

局限性包括：① 需要事件相机作为输入，无法直接应用于传统RGB相机；② 由于引入事件映射场，训练时间略长（约19%）；③ 对极端短曝光或极高噪声事件仍有一定鲁棒性不足；④ 模型依赖于准确的相机物理参数，若参数估计不准可能影响性能。

---

## 110. Exploring Implicit Perspectives on Autism in Large Language Models Through Multi-Agent Simulations

**arXiv ID:** 2601.15437 | [PDF](https://arxiv.org/pdf/2601.15437v1)

**作者:** Sohyeon Park `[一作]` (University of California), Gillian R. Hayes `[通讯]` (University of California)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过构建基于GPT‑4o‑mini的多智能体系统（MAS）模拟四名学生在一次小组任务中的互动，并在每轮对话后向各智能体提问，收集并分析其自我描述与他者评价，以探究LLM对自闭症个体的隐性偏见。

**💡 创新点**

①首次将LLM多智能体模拟与结构化访谈结合，用以捕捉复杂交互中潜在的系统性偏见；②发现模型倾向将自闭症个体视为社会依赖、需要他人适配；③提出将“双同理心问题”作为设计视角，建议在LLM设计中实现互惠适配而非单向补偿。

**🔧 技术方法**

使用GPT‑4o‑mini生成对话与回答；采用生成式代理框架（Generative Agents）实现多智能体；通过定量（卡方、t检验）和定性（主题分析）方法对模拟结果进行评估。

**📊 数据集**

实验数据来自120次仿真（4个案例 × 30次），每次包含4名智能体（其中一名被标记为自闭症）产生的对话日志与访谈回答；没有使用外部真实数据集，而是完全在模型内部生成。

**📈 对比分析**

对非自闭症与自闭症智能体在困境感受、差异对待、满意度等指标进行统计比较；卡方检验显示非自闭症智能体对自闭症伙伴遇到困难的报告显著更高；t检验显示两者在差异对待与困难感知上的显著差异；整体发现表明模型表现出显著的自闭症偏见。

**⚠️ 局限性**

①仅评估单一LLM（GPT‑4o‑mini），结果不具普适性；②MAS仅允许两名智能体同时对话，难以捕捉大规模群组动力学；③模拟生成的“自闭症”身份与情感缺乏真实性；④模型存在性别误判与对非二元身份的识别不准；⑤对比分析仅聚焦于偏见表征，未涉及模型改进或多模型验证。

---

## 111. Tracking the Limits of Knowledge Propagation: How LLMs Fail at Multi-Step Reasoning with Conflicting Knowledge

**arXiv ID:** 2601.15495 | [PDF](https://arxiv.org/pdf/2601.15495v1)

**作者:** Yiyang Feng `[一作]` (École Polytechnique Fédérale de Lausanne), Antoine Bosselut `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 5655 | [OpenAlex ID](https://openalex.org/A5088410008)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

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

## 112. Prometheus Mind: Retrofitting Memory to Frozen Language Models

**arXiv ID:** 2601.15324 | [PDF](https://arxiv.org/pdf/2601.15324v1)

**作者:** Mark Wind `[一作]` `[通讯]` (AIQuest), Mark Wind (AIQuest)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为冻结的Qwen3-4B添加可逆记忆模块，使用11个小型适配器实现记忆提取、检索和注入。

**💡 创新点**

提出无监督语义方向发现（CDD）、分阶段适配器训练、使用模型自身词表权重作为注入值（Identity V）以及通过投影恢复隐藏状态衍化。

**🔧 技术方法**

低秩适配器、对比方向发现、注意力头选择、投影网络、多跳推理等技术。

**📊 数据集**

自建PrometheusExtract-132基准（132个案例、203条事实），以及在Qwen3-4B预训练数据上进行无监督训练。

**📈 对比分析**

与RAG、Titans、LongMem等对比，PrometheusMind在干净输入上94.4%检索准确率，冒险输入19.4%，单纯关系分类仅47.3%。

**⚠️ 局限性**

主要瓶颈是关系分类错误，难以处理多主语/多宾语、复杂句子，且仅在Qwen3-4B上验证，未在其他模型或公开基准上测试。

---

## 113. On the closest balanced game

**arXiv ID:** 2601.15318 | [PDF](https://arxiv.org/pdf/2601.15318v1)

**作者:** Pedro García-Segador `[一作]` (National Statistical Institute), Pedro Miranda `[通讯]` (Interdisciplinary Mathematical Institute)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在核心为空的合作博弈中寻找与其最近的平衡博弈，并以此定义新的解概念——最小平方核心。

**💡 创新点**

创新点包括：①设计了线性规模的 CLOBIS 算法，将最初指数复杂度的投影问题压缩到仅 n 个变量；②从组合学角度求得最小平衡集合的渐近计数，证明投影博弈的核心趋于单点，概率随玩家数增长而趋近 1。

**🔧 技术方法**

主要技术：凸二次规划、线性代数投影、最小平衡集合的组合枚举、概率与随机矩阵论、迭代约束简化与伪逆求解。

**📊 数据集**

实验使用随机生成的 TU‑博弈：在 [-L, L] 区间内均匀采样，L=1,10,100，玩家数从 3 到 20，未使用任何真实数据集。

**📈 对比分析**

与传统全约束二次规划相比，CLOBIS 仅需 2^n+n-3 个变量、2^n-1 个约束，时间从几秒下降到毫秒级；仿真显示，当玩家数≥5 时，投影博弈核心为单点的概率已接近 1，算法能稳定处理 20 位玩家。

**⚠️ 局限性**

局限性：极少数情况下 CLOBIS 可能陷入循环，需要随机重启；理论结果基于 NEFPC 假设及渐近分析，未在小规模或非均匀分布上得到严格证明；此外，对最小平方核心的完整性质与适用范围尚待进一步研究。

---

## 114. Lattice: A Confidence-Gated Hybrid System for Uncertainty-Aware Sequential Prediction with Behavioral Archetypes

**arXiv ID:** 2601.15423 | [PDF](https://arxiv.org/pdf/2601.15423v1)

**作者:** Lorian Bannis `[一作]` `[通讯]`, Lorian Bannis

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了Lattice系统，利用二进制置信门控在序列预测中按需激活行为原型，实现基线模型与原型知识的混合推理。

**💡 创新点**

核心创新是置信门控机制：通过百分位距离估计置信度，二进制决定是否使用原型评分，从而在分布偏移时主动拒绝使用学习到的模式，提升系统可信度。

**🔧 技术方法**

技术手段包括：LSTM/Transformer 序列模型、K-means 聚类生成行为原型、距离百分位置信用度计算、混合评分、三阶段长度策略和阈值校准。

**📊 数据集**

实验数据集涵盖推荐系统（MovieLens 1M、Amazon Reviews）、科学时间序列（LIGO）和金融市场（S&P 500 ETF），并在多种基准上进行对比。

**📈 对比分析**

采用与 LSTM、SASRec、BERT4Rec 等基线在同一测试集、全榜单评估；在 LSTM 上 HR@10 提升 31.9%，超过 transformer 109%/218%；在 LIGO 与金融通过拒绝激活保持与基线一致；在 Transformer 上无性能下降。

**⚠️ 局限性**

局限性包括需手动校准置信阈值和原型数量、对分布极端偏移的鲁棒性仍依赖阈值、潜在的训练集偏差可能在原型中放大。

---

## 115. Parallelism and Generation Order in Masked Diffusion Language Models: Limits Today, Potential Tomorrow

**arXiv ID:** 2601.15593 | [PDF](https://arxiv.org/pdf/2601.15593v1)

**作者:** Yangyang Zhong `[一作]` (Zhejiang University), Junbo Zhao `[通讯]` (Zhejiang University)

**通讯引用:** 10853 | [OpenAlex ID](https://openalex.org/A5042402520)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估并对比了8种主流掩码扩散语言模型与自回归模型在58项知识、推理、编程等任务上的表现，并提出AFP与Kendall's τ指标量化其并行度与生成顺序。

**💡 创新点**

首次系统性地对MDLM的并行解码与任意顺序解码能力进行度量，揭示其与AR模型的性能差距及任务适应性，并提出Generate‑then‑Edit策略以缓解因并行因式化导致的依赖损失。

**🔧 技术方法**

使用AFP、Kendall's τ统计指标、统一推理管道、GPU集群并行实验、基于掩码扩散的模型架构（LLaDA, Trado, SDAR等）以及自回归基准模型。

**📊 数据集**

覆盖58个基准，包括知识、数学、推理、自然语言理解、代理任务与编程，含Sudoku变体等。

**📈 对比分析**

通过统一提示、推理和评测，比较各模型在准确率和并行度上的差异，结果显示MDLM整体落后于等规模AR模型，且并行度与正确率呈正相关，Sudoku任务中MDLM表现突出。

**⚠️ 局限性**

受限于并行因式化导致的条件独立假设，MDLM在高依赖度任务中精度受限；实验受硬件优化差异、推理配置与Prompt差异影响，实际性能可能与报告值不同。

---

## 116. SAGE-FM: A lightweight and interpretable spatial transcriptomics foundation model

**arXiv ID:** 2601.15504 | [PDF](https://arxiv.org/pdf/2601.15504v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 117. The Dark Side of AI Transformers: Sentiment Polarization & the Loss of Business Neutrality by NLP Transformers

**arXiv ID:** 2601.15509 | [PDF](https://arxiv.org/pdf/2601.15509v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 118. Reflexis: Supporting Reflexivity and Rigor in Collaborative Qualitative Analysis through Design for Deliberation

**arXiv ID:** 2601.15445 | [PDF](https://arxiv.org/pdf/2601.15445v1)

**作者:** Runlong Ye `[一作]` (University of Toronto), Ha-Kyung Kong `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 511 | [OpenAlex ID](https://openalex.org/A5023280657)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一个协作式的反思主题分析工作空间，该系统整合了研究者反思、代码演化透明化和立场性讨论三大功能，支持RTA的深度解析与协作。

**💡 创新点**

创新点包括：①“设计为深度反思而非自动化”的框架；②利用LLM实现代码漂移检测、立场性提示和讨论焦点；③提供可视化的代码演化历史，将隐式反思显式化；④将反思嵌入编码流程，使其成为即时、连续的实践。

**🔧 技术方法**

使用技术：Web前端基于Next.js + Tailwind，后端利用Google Firebase（Firestore、身份验证、托管），LLM（OpenAI GPT‑4）负责代码漂移检测、讨论提示与反思摘要；所有功能均通过实时同步实现协作。

**📊 数据集**

数据集：合成的访谈转录，来源于CHIP关于“Contestable Camera Cars”的真实研究，通过LLM生成的三份情境相似的文本，保持主题与语境一致性。

**📈 对比分析**

比较方法：单一条件实验（无基准工具对照），通过问卷、访谈、思考‑录音、屏幕记录等方式收集主观体验与使用行为；未给出定量性能指标，评估重点在用户感知的反思深度、透明度和讨论质量上。

**⚠️ 局限性**

局限性：①样本仅12名经验研究者，单次短时实验；②使用合成数据缺少真实项目的复杂性；③未评估大团队或多方立场冲突情境；④LLM生成的立场提示可能简化或误导研究者的身份语境；④长期使用与跨项目迁移的有效性未验证。

---

## 119. Deep Learning for Perishable Inventory Systems with Human Knowledge

**arXiv ID:** 2601.15589 | [PDF](https://arxiv.org/pdf/2601.15589v1)

**作者:** Xuan Liao `[一作]` (Shanghai Jiao Tong University), Ying Rong `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 3615 | [OpenAlex ID](https://openalex.org/A5018560755)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出在随机交货期和未知需求的易腐产品多周期库存系统中，利用端到端深度学习结合库存理论结构来直接学习补货决策；

**💡 创新点**

创新点在于将投影库存水平（PIL）理论结构嵌入神经网络，形成E2E-PIL，并通过同质性加速的ODA提升（E2E-BPIL），显著降低模型复杂度并提升学习效率；

**🔧 技术方法**

核心技术包括边际成本会计损失函数、LSTM特征预测模块、投影库存水平计算模块、以及基于同质性理论的单参数放大器；

**📊 数据集**

使用了真实饮料供应链的日销量与交货期数据（853个SKU‑DC对，90天），以及多种合成数据（50 SKU ×20 DC，300天）进行实验；

**📈 对比分析**

与传统的预测-优化（PTO‑PB）以及全黑盒E2E-BB做对比，E2E-BPIL在真实数据上平均可比PTO‑PB低约10%-15%，在合成数据上降低约5%-7%；

**⚠️ 局限性**

局限性包括对PIL结构的依赖（在非可见的最优结构下可能欠拟合）、对交货期不交叉假设的敏感性、以及在极短保质期或极高随机性情况下性能下降的可能。

---

## 120. ALIGNAgent: Adaptive Learner Intelligence for Gap Identification and Next-step guidance

**arXiv ID:** 2601.15551 | [PDF](https://arxiv.org/pdf/2601.15551v1)

**作者:** Bismack Tokoli `[一作]` (Florida Polytechnic University), Ayesha S. Dina `[通讯]` (Florida Polytechnic University)

**通讯引用:** 283 | [OpenAlex ID](https://openalex.org/A5050631988)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一个多智能体框架 Adaptive Learner Intelligence，通过整合知识估计、技能缺口识别和资源推荐，实现个性化学习循环。

**💡 创新点**

创新点在于将诊断、推荐与学习路径规划融合为闭环，并利用LLM驱动的诊断推理提供概念级解释。

**🔧 技术方法**

采用大语言模型（GPT‑4o、Claude 3.5 Sonnet、LLaMA 3）结合贝叶斯知识追踪与认知诊断模型。

**📊 数据集**

使用佛罗里达理工大学两门本科计算机课程 COP 3415 与 CDA 2108 的测验、期中期末成绩及学习偏好数据。

**📈 对比分析**

对比手工与LLM标签、不同模型的性能，GPT‑4o 在知识熟练度估计中取得最高精度 0.90、F1 0.87，优于其他模型。

**⚠️ 局限性**

局限包括样本量小、仅在计算机科学领域验证、资源匹配与链接可用性不完全、依赖LLM可能产生幻觉。

---

## 121. Blockchain-Based Spectrum Resource Securitization via Semi-Fungible Token-Lock

**arXiv ID:** 2601.15594 | [PDF](https://arxiv.org/pdf/2601.15594v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 122. Attention-Informed Surrogates for Navigating Power-Performance Trade-offs in HPC

**arXiv ID:** 2601.15399 | [PDF](https://arxiv.org/pdf/2601.15399v1)

**作者:** Ashna Nawar Ahmed `[一作]` (Texas State University), Tanzima Z. Islam `[通讯]` (Texas State University)

**通讯引用:** 429 | [OpenAlex ID](https://openalex.org/A5002465410)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于注意力嵌入与智能采样的多目标贝叶斯优化框架，用于自动决定HPC作业节点数，平衡运行时与功耗。

**💡 创新点**

首次将注意力嵌入应用于HPC遥测的代理模型，并将其与多目标贝叶斯优化结合，以数据高效方式捕获运行时–功耗权衡。

**🔧 技术方法**

使用TabNet实现注意力嵌入，结合轻量级回归器（随机森林、XGBoost、LightGBM）作为代理模型；采用qEHVI等多目标采集函数与主动学习式智能采样。

**📊 数据集**

在两个真实生产日志数据集上实验：PM100（约231k条作业，35维特征）和Adastra（约15k条作业，35维特征）。

**📈 对比分析**

与单目标贝叶斯优化（仅运行时或仅功耗）和随机搜索对比，MOBO+嵌入在大多数实验中提升了24–37%的超体积（HV），降低了近99%的扩散（Spread），并仅需50–75%的样本，训练时间显著缩短。

**⚠️ 局限性**

局限在于仅考虑两目标，Transformer模型在小样本下表现不佳；未来需扩展到多目标、提升跨系统泛化、降低实际调度时延并验证部署可行性。

---

## 123. Neural Nonlinear Shrinkage of Covariance Matrices for Minimum Variance Portfolio Optimization

**arXiv ID:** 2601.15597 | [PDF](https://arxiv.org/pdf/2601.15597v1)

**作者:** Liusha Yang `[一作]` (Shenzhen Technology University), Shuqi Chai `[通讯]` (Shenzhen Research Institute of Big Data)

**通讯引用:** 159 | [OpenAlex ID](https://openalex.org/A5074707388)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于轻量级Transformer的非线性特征缩减精度矩阵估计器，用以实现最小方差投资组合的风险最小化。

**💡 创新点**

创新点在于将统计学Ledoit‑Wolf线性缩减与神经网络非线性特征缩减相结合，并直接学习精度矩阵以最小化实际组合风险，而非传统的协方差估计。

**🔧 技术方法**

使用技术包括特征值分解、轻量级Transformer网络、非线性特征缩减、基于组合风险的训练目标以及滚动窗口实验评估。

**📊 数据集**

使用了2021‑2025年S&P500指数中随机抽取的50只股票的每日对数收益数据，共约3,675个交易日。

**📈 对比分析**

通过与样本协方差、Ledoit‑Wolf、Chen、单位矩阵以及直接权重估计等基线方法比较，实验结果表明该方法在多种样本规模下实现了最低的出样风险，优于所有基线。

**⚠️ 局限性**

限制在于目前仅聚焦协方差估计，未联合估计预期收益；训练需要足够历史数据；在极端资产数或样本数下的泛化性能尚未完全验证。

---

## 124. ViT Registers and Fractal ViT

**arXiv ID:** 2601.15506 | [PDF](https://arxiv.org/pdf/2601.15506v1)

**作者:** Jason Chuan-Chih Chou `[一作]` (Cohere Labs), Shivank Garg `[通讯]` (Indian Institute of Technology Roorkee)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5104180026)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并测试了 Fractal ViT，这是一种在 ViT 中添加 summary tokens 并使用 fractal attention mask 的变体，用于打破 token 的置换不变性。

**💡 创新点**

通过在 ViT 中引入 summary tokens 并采用分层 self‑similar attention mask（fractal mask）尝试在无位置编码的情况下提供位置信息。

**🔧 技术方法**

使用 ViT‑S/16 编码器、不同的位置信息（sincos2d、learned、2D‑ALiBi、无位置信息）、registers、summary tokens 以及自定义的 fractal attention mask 进行实验。

**📊 数据集**

在 ImageNet‑1k 训练集上评估 top‑1 验证集准确率。

**📈 对比分析**

将 Fractal ViT 与基准 ViT、含 registers 或 summary tokens 的模型进行对比；发现 fractal mask 无提升，附加 token 仅带有 1–2 个标准差的小幅提升，整体准确率与基准相当。

**⚠️ 局限性**

fractal mask 在图像分类任务上无效，位置信息对附加 token 影响甚微，且仅在特定规模/域（如 LLM 或卫星图像）可能有意义。

---

## 125. Region-aware Spatiotemporal Modeling with Collaborative Domain Generalization for Cross-Subject EEG Emotion Recognition

**arXiv ID:** 2601.15615 | [PDF](https://arxiv.org/pdf/2601.15615v1)

**作者:** Weiwei Wu `[一作]` (Shanghai Maritime University), Nizhuan Wang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 1009 | [OpenAlex ID](https://openalex.org/A5047632784)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种区域感知时空建模与协同域泛化框架（RSM‑CoDG），实现跨受试EEG情绪识别；

**💡 创新点**

创新点在于①结合功能脑区先验构建区域感知图表示（RGRM），②采用多尺度时序Transformer（MSTT）捕捉短期与长期情绪动力学，③引入多目标协同域泛化（CoDG）通过分布对齐、注意力一致性及特征正交约束共同抑制受试者偏差；

**🔧 技术方法**

技术实现包括RGRM、MSTT与CoDG三大模块，CoDG通过MMD、对比学习与正交损失实现联合优化；

**📊 数据集**

使用SEED、SEED‑IV和SEED‑V三个公开EEG情绪数据集；

**📈 对比分析**

在LOSO交叉验证下，RSM‑CoDG分别取得86.35%、71.59%和62.77%的准确率，均超过传统机器学习、域适应与其他深度学习基线，显示出显著的性能提升；

**⚠️ 局限性**

主要局限在训练过程计算量大、训练时间长；且在多类别情绪区分上仍有提升空间，未来需进一步优化模型轻量化与跨类平衡。

---

## 126. KnowTeX: Visualizing Mathematical Dependencies

**arXiv ID:** 2601.15294 | [PDF](https://arxiv.org/pdf/2601.15294v1)

**作者:** Elif Uskuplu `[一作]`, Valeria de Paiva `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 KnowTeX，一款独立的 Python 工具，可通过在 LaTeX 文档中插入简易命令自动生成数学结果、定义、定理等之间的依赖图，输出 DOT 和 TikZ 格式。

**💡 创新点**

创新点在于：①实现完全脱离任何证明助手或大型生态系统的依赖图生成；②利用作者已有的标签引用机制，提供可视化的概念与逻辑依赖，支持手动标注而非自动推断；③兼容现有的 Lean Blueprint、plasTeXdepgraph 等工具，且保持更简洁、易用的工作流。

**🔧 技术方法**

主要技术包括：Python 脚本解析 LaTeX 源文件；正则表达式匹配自定义环境与命令；构造依赖关系图并做传递闭包；使用 Graphviz DOT 以及 TikZ 生成可视化文件；提供预览窗口与自定义节点/边属性的 UI。

**📊 数据集**

数据集主要为数学教材和论文的 LaTeX 源代码，如 Tom Leinster 的《Basic Category Theory》；未使用公开大规模数据集，更多以案例测试为主。

**📈 对比分析**

与 Lean Blueprint、plasTeXdepgraph、Trouver 以及商业化 ScienceStack 进行对比：相较于依赖证明助手或自动机器学习推断，KnowTeX 提供了更高的可控性、透明性和部署便利性；虽然缺乏量化性能指标，但在易用性、可扩展性和跨平台兼容性上表现更佳；在传递闭包处理和命令可视化上与现有工具保持一致。

**⚠️ 局限性**

局限性包括：①需要作者手动插入依赖命令，缺乏自动化推断；②无法捕捉文本中隐含的逻辑关系；③目前仅支持 LaTeX 生态，未与证明助手深度集成；④对大规模文档的性能未做系统评估；⑤对非数学学科的通用性仍需进一步验证。

---

## 127. Partially Polarized Polar Codes: A New Design for 6G Control Channels

**arXiv ID:** 2601.15404 | [PDF](https://arxiv.org/pdf/2601.15404v1)

**作者:** Arman Fazeli `[一作]` (Apple), Louay Jalloul `[通讯]` (Apple)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出一种通过部分极化实现的分段极化码（PPP码），用于6G下行控制通道的盲解码；

**💡 创新点**

创新点在于在分段极化码中插入部分极化层，重平衡各段信道容量，既保留分段的硬件复用优势，又显著提升错误率性能，并实现两阶段解码的有效早停；

**🔧 技术方法**

主要技术包括极化码构造、部分极化层设计、两阶段（F/G）解码、可靠性序列构造（Bhattacharyya、αβ扩展、GNN强化学习）以及SCL/ML解码；

**📊 数据集**

使用仿真数据集：AWGN与BSC/BEC通道下的BLER曲线，并对不同块长、RNTI长度与CRC配置进行实验；

**📈 对比分析**

与传统分段、聚合以及全长度极化码进行BLER对比，实验显示PPP码在相同码率和硬件约束下可显著降低BLER，逼近全长度极化码的性能；

**⚠️ 局限性**

局限性包括：部分极化比例的选择仍依赖经验；理论证明仅覆盖容量收敛，缺乏对误差谱的完整分析；未给出具体硬件实现与功耗评估；在非理想信道下的鲁棒性待进一步验证。

---

## 128. Computational Representations of Character Significance in Novels

**arXiv ID:** 2601.15508 | [PDF](https://arxiv.org/pdf/2601.15508v1)

**作者:** Haaris Mian `[一作]` (Columbia University), Kathleen McKeown `[通讯]` (Columbia University)

**通讯引用:** 19405 | [OpenAlex ID](https://openalex.org/A5109565051)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究者提出并实现了基于六元结构模型的角色特征标注与网络分析框架。

**💡 创新点**

创新点在于引入“其他角色讨论”这一此前未被量化的组件，并通过该组件构建讨论网络，从而揭示角色中心性的新维度。

**🔧 技术方法**

采用BookNLP转写+规则+LLM（GPT‑4o‑mini）进行跨度级与章节级组件标注，并利用图论指标（PageRank、Betweenness等）构建共现、对话和讨论三种网络。

**📊 数据集**

以简明英美19世纪小说《傲慢与偏见》和《简爱》为标注基准，随后扩展到包含64本第三人称叙事小说的语料库。

**📈 对比分析**

通过与手工标注的黄金标准比较，BookNLP+规则方法在MAE≈1.7、Pearson≈0.75的水平优于单纯LLM章节级计数；网络层面不同组件与中心性指标呈显著相关，验证模型有效性。

**⚠️ 局限性**

限制包括只涵盖第三人称、19世纪英文小说；对第一人称叙事的核心指代解析不足；LLM在规模化标注时易出现误计与遗漏，且标注成本高。

---

## 129. Chunking, Retrieval, and Re-ranking: An Empirical Evaluation of RAG Architectures for Policy Document Question Answering

**arXiv ID:** 2601.15457 | [PDF](https://arxiv.org/pdf/2601.15457v1)

**作者:** Anuj Maharjan `[一作]` (University of Toledo), Umesh Yadav `[通讯]` (University of Toledo)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5080728538)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估检索增强生成（RAG）架构在公共卫生政策问答中的有效性

**💡 创新点**

首次系统比较了带跨编码器重排序的高级RAG与基本RAG在CDC政策文档上的表现，并验证其在提升答案可信度和相关性方面的优势

**🔧 技术方法**

采用Mistral‑7B‑Instruct生成模型、all‑MiniLM‑L6‑v2向量检索和ms‑marco‑MiniLM‑L‑6‑v2跨编码器重排序技术

**📊 数据集**

使用CDC官方政策分析框架、战略与政策制定指南以及项目成本分析手册等文档集合进行评估

**📈 对比分析**

通过对10个复杂情境的可信度和相关性评分比较，Advanced RAG平均可信度0.797、相关性0.800，显著优于Basic RAG（0.621/0.697）和Vanilla LLM（0.347/0.450）

**⚠️ 局限性**

主要局限在于文档分块方式对多步推理的影响及chunking结构约束导致的逻辑片段化，仍需改进结构感知的分块策略

---

## 130. Logic Programming on Knowledge Graph Networks And its Application in Medical Domain

**arXiv ID:** 2601.15347 | [PDF](https://arxiv.org/pdf/2601.15347v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 131. Equal-Pay Contracts

**arXiv ID:** 2601.15478 | [PDF](https://arxiv.org/pdf/2601.15478v1)

**作者:** Michal Feldman `[一作]` (Tel Aviv University and Microsoft), Maya Schlesinger `[通讯]` (Tel Aviv University)

**通讯引用:** 885 | [OpenAlex ID](https://openalex.org/A5109860000)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究多代理合约设计中等额支付（equal‑pay）与近似等额支付合同，分析其在多代理组合动作模型下的可计算性与最优性。

**💡 创新点**

提出对等额支付合同的多种算法与硬性界限，证明在子模、XOS、粗替代奖励函数下的逼近与可计算边界，并给出公平性导致的效率损失（价格 of equality）为Θ(log n / log log n)。

**🔧 技术方法**

利用需求查询（demand oracle）、子模/超加性函数的特性、逼近/折半技术、Yao 原理、覆盖/基数矩阵归约以及自适应分桶等算法与证明手段。

**📊 数据集**

该工作为理论研究，未使用具体数据集；所有结论均通过数学证明与理论构造获得。

**📈 对比分析**

与传统无约束合同设计相比，等额支付合同在子模奖励下可实现常数近似，而在 XOS 与粗替代奖励下无法获得 PTAS；公平性约束导致的收益损失被界定为 Θ(log n / log log n)，与无约束情况相比较低。

**⚠️ 局限性**

局限性包括：仅覆盖特定奖励函数族；对 XOS 的结果仅适用于二进制动作；在更一般的非子模/非 XOS 情况下缺乏可行算法；实际应用中对算法复杂度与参数规模的依赖未完全评估。

---

## 132. Beyond Prompting: Efficient and Robust Contextual Biasing for Speech LLMs via Logit-Space Integration (LOGIC)

**arXiv ID:** 2601.15397 | [PDF](https://arxiv.org/pdf/2601.15397v1)

**作者:** Peidong Wang `[一作]` (Microsoft), Peidong Wang `[通讯]` (Microsoft)

**通讯引用:** 1073 | [OpenAlex ID](https://openalex.org/A5062250845)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种在解码层直接注入上下文偏置的LOGIC框架，用于提升Speech LLM对稀有实体的识别率；

**💡 创新点**

创新点在于将前缀树与logit空间结合，采用即时前缀提升(IPB)与回溯评分修正(RSR)双重策略，实现高召回且低误报；

**🔧 技术方法**

使用Trie结构、Sparse CUDA核、向量化logit处理以及自定义LogitsProcessor；

**📊 数据集**

在Phi-4-mini模型上，针对11种语言的Person Name测试集（约460–2100个实体/语言）进行评测；

**📈 对比分析**

与无偏置基线对比，平均EWER下降9%（稳健设定）或17%（激进设定），FAR仅升0.30%，RTF增加约2.8%；

**⚠️ 局限性**

局限在于仍需手工调参（如bias λ）、仅支持精确前缀匹配，未处理模糊匹配和跨词实体。

---

## 133. Airflow Source Seeking on Small Quadrotors Using a Single Flow Sensor

**arXiv ID:** 2601.15607 | [PDF](https://arxiv.org/pdf/2601.15607v1)

**作者:** Lenworth Thomas `[一作]` (Carnegie Mellon University), Sarah Bergbreiter `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2639 | [OpenAlex ID](https://openalex.org/A5051309977)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

本文在小型四旋翼无人机上集成单一磁性流量传感器，实现了基于气流源的寻源导航；

**💡 创新点**

创新点在于利用传感器实时测量流量向量，并将其嵌入改进的 Cast‑and‑Surge 算法（Vector Surge），实现低速气流（0.2 m/s）快速定位；

**🔧 技术方法**

技术包括磁场测量流量传感器、BLE 数据传输、OptiTrack 位姿跟踪、PD 控制器以及 Vector Surge 算法；

**📊 数据集**

实验数据集为在 10 × 10 m 室内捕捉室中使用手持风速计测得的不同距离（1.5、3、4.5、6 m）下的风速（0.2–1.24 m/s）和无人机轨迹；

**📈 对比分析**

通过与 IMU 姿态、手持风速计等基准对比，传感器在噪声下仍能精确估计流向，且无人机可在约10 秒内将航向误差调至≤20°，在10次随机起始实验中成功到达源点；

**⚠️ 局限性**

限制包括仅能测量平面流向，无法可靠识别上下流，且在复杂环境下需配备障碍物检测与状态估计。

---

## 134. CASL: Concept-Aligned Sparse Latents for Interpreting Diffusion Models

**arXiv ID:** 2601.15441 | [PDF](https://arxiv.org/pdf/2601.15441v1)

**作者:** Zhenghao He `[一作]` (University of Virginia), Aidong Zhang `[通讯]` (University of Virginia)

**通讯引用:** 11665 | [OpenAlex ID](https://openalex.org/A5013588572)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了CASL框架，通过在扩散模型U‑Net激活上训练稀疏自编码器并学习线性映射，实现了将潜在维度与人类语义概念对齐，并通过CASL‑Steer进行因果探测；

**💡 创新点**

其创新点在于首次实现了稀疏潜在空间的监督概念对齐、提供了专门用于验证概念语义影响的因果探测方法CASL‑Steer，以及联合衡量目标属性变更与副作用的编辑精度比（EPR）指标；

**🔧 技术方法**

主要技术包括稀疏自编码器、线性概念映射、DDIM采样、DiffusionCLIP损失、线性SVM探测以及EPR评估；

**📊 数据集**

实验使用了FFHQ、CelebA‑HQ、LSUN‑Church以及AFHQ四个公开数据集；

**📈 对比分析**

与BoundaryDiffusion、Asyrp、Concept Slider、MasaCtrl、SwiftEdit等基线在五个面部属性上进行比较，CASL‑Steer在CLIP‑Score、LPIPS、ArcFace以及EPR等指标上均优于或匹配现有方法，尤其在EPR上表现突出；

**⚠️ 局限性**

局限性包括依赖人工标注的概念监督、单维度编辑对高维概念可能不足、编辑多维度时精度下降，以及仅在冻结扩散模型上验证，尚未探索可迁移性与大规模概念扩展。

---

## 135. Stabilizer-Code Channel Transforms Beyond Repetition Codes for Improved Hashing Bounds

**arXiv ID:** 2601.15505 | [PDF](https://arxiv.org/pdf/2601.15505v1)

**作者:** Tyler Kann `[一作]` (Georgia Institute of Technology), Ruediger Urbanke `[通讯]` (EPFL)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种将任意稳定子码视为Pauli信道变换的通用框架，并通过计算诱导逻辑误差和符号的联合分布来评估其哈希率；

**💡 创新点**

提出了基于诱导信道的哈希率计算方法以及结构化搜索小块变换的策略，从而在偏置独立Pauli信道上实现对哈希界的改进；

**🔧 技术方法**

使用了完整的对称量子码表、诱导信道概率计算、条件熵哈希分析、随机与深度优先搜索等技术；

**📊 数据集**

针对偏置独立Pauli信道（bias η=9），在不同噪声水平p下对变换进行实验；

**📈 对比分析**

与传统哈希界以及单重复码的结果比较，发现高率“全Z”变换在低噪声区能略微超越哈希界，改进幅度虽小但具显著结构信息；

**⚠️ 局限性**

局限在于只考虑极小块长（n≤12），无法直接推广到更大码长；所得到的改进在实际误差校正中尚无直接意义，且现有上界仍相对宽松。

---

## 136. A tensor network formalism for neuro-symbolic AI

**arXiv ID:** 2601.15442 | [PDF](https://arxiv.org/pdf/2601.15442v1)

**作者:** Alex Goessmann `[一作]`, Martin Eigel `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出一种基于张量网络的统一框架（称为 HLN），用于同时表达神经网络、概率图模型和命题逻辑模型，并通过张量收缩实现推理与参数学习。

**💡 创新点**

创新点在于：①将逻辑公式、概率分布及可分解函数都视为张量；②利用张量网络的稀疏分解（如条件独立、充分统计、神经分解）实现模型压缩；③将推理任务统一为张量收缩，并给出基于消息传递（Belief Propagation、Constraint Propagation、AMM）高效实现；④实现了开源库 tnreason，支持构建、训练和推理。

**🔧 技术方法**

核心技术包括：张量网络（MPS/PEPS/树形结构）、基底编码（one‑hot）、Delta 张量、超图表示、张量收缩与归一化、消息传递算法（树形 Belief Propagation、Directed Belief Propagation、Constraint Propagation）、Fisher‑Neyman 因子化、指数族等。

**📊 数据集**

论文主要以示例和理论演示为主，没有公开使用大型真实数据集；在实验部分仅演示了合成逻辑公式、手工构造的知识库以及用于参数估计的人工样本（IID 训练数据）。

**📈 对比分析**

方法对比主要是理论上的可行性和复杂度讨论：在树形张量网络下，Belief Propagation 可实现线性时间的精确推理；但在一般图形下推理仍是 NP‑hard，需使用近似方法（变分推理、Gibbs 采样等）。论文未给出具体数值实验或性能基准。

**⚠️ 局限性**

局限性包括：①张量收缩在大规模、环状网络中计算量指数级；②对概率与逻辑混合模型的训练仍依赖于近似推理，收敛性和误差分析未完善；③缺乏针对真实任务的实证评估；④对高维连续变量支持有限，需进一步扩展。

---

## 137. Problems with fixpoints of polynomials of polynomials

**arXiv ID:** 2601.15420 | [PDF](https://arxiv.org/pdf/2601.15420v1)

**作者:** Cécilia Pradic `[一作]` (Swansea University), Ian Price `[通讯]` (Swansea University)

**通讯引用:** 1873 | [OpenAlex ID](https://openalex.org/A5091579328)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了在容器（polynomial functors）范畴内的纤维化多项式终结子和初始子，利用这些极限构造了一系列新的 Weihrauch 复杂度操作，并给出了它们的自举语法（ζ‑表达式）和自动机解释。

**💡 创新点**

创新点在于：① 提出了纤维化多项式函子的固定点理论；② 将极限与多项式容器相结合，得到的 ζ‑表达式可描述复杂的求解策略；③ 证明这些固定点在可计算分析与逆数学中的 Weihrauch lattice 上能表示重要的选择与判定问题。

**🔧 技术方法**

主要技术包括范畴论中的纤维化、极限/余极限的构造、可计算分析中的代表空间与 Weihrauch 简化、以及通过可接受的自动机与 ω‑游戏树来对容器进行自动机化描述。

**📊 数据集**

由于本文是理论性工作，并未使用具体数据集；所有结果均在抽象范畴/可计算结构上证明，实验数据不适用。

**📈 对比分析**

由于缺乏数值实验或基准测试，无法用传统的性能指标来比较方法；作者通过同构与 Weihrauch 等价关系对不同操作的能力做了严格的理论比较，展示了它们在判定与选择问题上的阶层结构。

**⚠️ 局限性**

主要局限性包括：① 固定点构造依赖于 - 类范畴（需有可极限结构）；② ζ‑表达式虽然能覆盖许多 Weihrauch 度，但并非所有可计算分析中的重要度数（如强连续性）都能通过该语法获得；③ 该方法对数据驱动的可计算性研究仍不具备直接可实现性。

---

## 138. The Paradigm Shift: A Comprehensive Survey on Large Vision Language Models for Multimodal Fake News Detection

**arXiv ID:** 2601.15316 | [PDF](https://arxiv.org/pdf/2601.15316v1)

**作者:** Wei Ai `[一作]` (Central South University of Forestry and Technology), Keqin Li `[通讯]` (State University of New York)

**通讯引用:** 29725 | [OpenAlex ID](https://openalex.org/A5087894632)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统综述了大型视觉语言模型在多模态假新闻检测中的应用，并提出了参数冻结、参数微调和推理三大范式的分类框架。

**💡 创新点**

首次构建了基于三大范式的结构化分类体系，对比分析了各范式下的技术路线与研究进展。

**🔧 技术方法**

归纳了 CLIP、BLIP‑2、LLaVA、GPT‑4V 等 LVLM 模型，及 Prompt、Adapter、Agent 等技术手段。

**📊 数据集**

整理了多模态假新闻检测数据集，包括 Twitter15/16、PHEME、LIAR、M^3A、MMFakeBench 等公开基准。

**📈 对比分析**

通过对比不同范式在公开基准上的准确率、F1 等指标，发现参数微调和推理范式往往优于冻结范式，且大模型在跨模态一致性检测上表现更佳。

**⚠️ 局限性**

局限性在于缺乏统一实验平台，难以跨数据集直接比较；模型可解释性、鲁棒性及对抗性等问题仍未得到充分解决。

---

## 139. A Checklist for Trustworthy, Safe, and User-Friendly Mental Health Chatbots

**arXiv ID:** 2601.15412 | [PDF](https://arxiv.org/pdf/2601.15412v1)

**作者:** Shreya Haran `[一作]` (University of Illinois Urbana-Champaign), Koustuv Saha `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 3141 | [OpenAlex ID](https://openalex.org/A5057029055)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建并验证了一套针对心理健康聊天机器人的责任性、可信度与用户友好性的操作性清单；

**💡 创新点**

将分散的伦理、可信度与用户体验原则系统化为可操作的清单，并通过对现有聊天机器人Woebot的评估验证其可用性；

**🔧 技术方法**

无直接技术实现，主要采用系统性文献综述与主题分析的方法来提炼原则；

**📊 数据集**

无特定数据集，依据PRISMA框架检索了50篇相关文献并手工筛选43篇进行定性分析；

**📈 对比分析**

通过将清单应用于Woebot进行“是/否”评估，发现大部分原则被满足，但对透明度、同理心等主观维度的评估仍存在挑战；

**⚠️ 局限性**

清单仅为二元判断，缺乏对主观维度的量化评估；缺乏多方参与的用户研究与系统性验证；未对不同聊天机器人进行大规模比较，结果具有一定局限性。

---

## 140. Tackling the Scaffolding Paradox: A Person-Centered Adaptive Robotic Interview Coach

**arXiv ID:** 2601.15600 | [PDF](https://arxiv.org/pdf/2601.15600v1)

**作者:** Wanqi Zhang `[一作]` (University of Tennessee), Marielle Santos `[通讯]` (University of Tennessee)

**通讯引用:** 13 | [OpenAlex ID](https://openalex.org/A5113028390)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过三阶段迭代设计，构建了一个基于人本治疗（PCT）和教学支架理论的社交机器人面试教练，并在每个阶段探索情感支持与即时/延迟反馈的平衡；

**💡 创新点**

提出了“支架悖论”与“代理驱动交互模式”，实现了通过用户主动选择反馈时机来调节情绪安全与教学有效性的可适应支架生态系统；

**🔧 技术方法**

采用了 Misty II 具身机器人、OpenAI Realtime API（GPT‑realtime‑2025‑08‑28）与多模态情绪/姿态交互脚本；

**📊 数据集**

收集了8名大学生（N = 8）的面试练习日志、MASI（面试焦虑量表）、RoSAS‑SF（机器人社会属性）与 B‑L RI‑mini（治疗联盟）等自评数据；

**📈 对比分析**

通过前后测的 MASI 变化、RoSAS‑SF 与 B‑L RI‑mini 的 Kruskal‑Wallis/配对 t 检验，发现心理安全显著提升（温暖分数↑）、焦虑显著降低（社交焦虑从 3.35↓到 2.65，p < .001），但对机器人技术能力评估无显著差异；

**⚠️ 局限性**

局限包括样本量小、仅为大学生、研究时长短、主要采用自评指标、未评估真实面试表现，以及系统对情绪识别与自动适应的技术挑战。

---

## 141. DeepASMR: LLM-Based Zero-Shot ASMR Speech Generation for Anyone of Any Voice

**arXiv ID:** 2601.15596 | [PDF](https://arxiv.org/pdf/2601.15596v1)

**作者:** Leying Zhang `[一作]` (Shanghai Jiao Tong University), Yanmin Qian `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 11754 | [OpenAlex ID](https://openalex.org/A5100341993)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出DeepASMR框架，实现零样本ASMR语音合成；

**💡 创新点**

通过token级软分解将ASMR风格与说话人音色解耦，结合LLM+流匹配解码器；

**🔧 技术方法**

使用大型语言模型（Qwen2.5-0.5B）、S3语义token、流匹配声学解码器、HiFi‑GAN声码器；

**📊 数据集**

构建670小时双语ASMR语料DeepASMR‑DB（35位说话人，中文22名，英文13名）；

**📈 对比分析**

与CosyVoice2、F5TTS及VC级联基线对比，使用客观指标（WER/CER、SIM、风格分数）和主观MOS，DeepASMR在跨风格N2A任务中显著提升ASMR风格分数和未声比率，同时保持良好说话人相似度；

**⚠️ 局限性**

仅支持语音ASMR，未覆盖非语音触发，数据量仍有限，可能面临音频伪造风险。

---

## 142. PromptHelper: A Prompt Recommender System for Encouraging Creativity in AI Chatbot Interactions

**arXiv ID:** 2601.15575 | [PDF](https://arxiv.org/pdf/2601.15575v1)

**作者:** Jason Kim `[一作]` (Texas A&M University), James Caverlee `[通讯]` (Texas A&M University)

**通讯引用:** 8640 | [OpenAlex ID](https://openalex.org/A5048489384)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种PromptHelper，作为提示推荐系统（PRS）集成到AI写作聊天机器人中，帮助用户在写作过程中生成多样化、上下文相关的后续提示。

**💡 创新点**

创新点在于将提示视为可推荐的交互对象，持续提供多样化、可选择的后续提示，而非仅限于冷启动或自动完成，从而显著提升用户的探索性和表达性。

**🔧 技术方法**

技术实现包括：1) 基于LLM的聊天机器人（WritingBot）接口；2) 通过上下文分析生成后续提示的推荐算法；3) UI设计使提示可视化、可复制和可编辑。

**📊 数据集**

使用的数据为32名来自Prolific的参与者在两种写作任务（创意写作、学术写作）中的交互记录和自评问卷，未使用公开文本语料库。

**📈 对比分析**

在2×2完整交叉实验（任务×系统开关）中，对比PromptHelper开启与关闭的效果，结果显示探索性显著提升（η²ₚ≈0.29，p=0.001）和表达性提升（η²ₚ≈0.19，p=0.011），但工作负荷和可用性无显著差异。

**⚠️ 局限性**

局限性包括样本规模仅32人、任务类型有限、仅测试文本生成场景，且未评估不同LLM模型或跨语言的适用性。

---

## 143. Relative Classification Accuracy: A Calibrated Metric for Identity Consistency in Fine-Grained K-pop Face Generation

**arXiv ID:** 2601.15560 | [PDF](https://arxiv.org/pdf/2601.15560v1)

**作者:** Sylvey Lin `[一作]` (University of Illinois Urbana-Champaign), Eranki Vasistha `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究并评估了类条件DDPM在K-pop偶像面部生成中的语义一致性，提出RCA指标并验证模型在32×32分辨率下的身份保留效果。

**💡 创新点**

引入相对分类准确率（RCA）作为校准的身份一致性度量，结合细粒度K-pop面孔数据进行评估，揭示视觉质量与语义控制之间的权衡。

**🔧 技术方法**

使用类条件DDPM、UNet噪声预测器、MTCNN预处理、ResNet-34判别器、LPIPS多样性评估，以及FID/IS等视觉指标。

**📊 数据集**

采用KoIn10（K-pop偶像面孔10类）数据集，并在32×32分辨率下进行预处理和增强。

**📈 对比分析**

与FID 8.93、IS 1.065、LPIPS 0.472 的视觉指标相比，RCA仅为0.27，表明模型在身份保持方面存在显著模式崩溃和分辨率瓶颈。

**⚠️ 局限性**

低分辨率导致高频身份特征缺失、同性别相似导致错误分类、模式偏向导致部分类别模式崩溃，以及对RCA评价依赖单一判别器的局限。

---

## 144. Common to Whom? Regional Cultural Commonsense and LLM Bias in India

**arXiv ID:** 2601.15550 | [PDF](https://arxiv.org/pdf/2601.15550v1)

**作者:** Sangmitra Madhusudan `[一作]` (Emory University), Ali Emami `[通讯]` (Emory University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了印度五地区文化常识基准INDICA，包含515个问题和1,630个地区特定答案。

**💡 创新点**

创新点在于从国家层面转向子国家层级，揭示文化常识的地区差异并系统评估LLM的地域偏差。

**🔧 技术方法**

采用OCM人类学分类、GPT‑4生成问题与答案、GPT‑4o辅助审核，并用Gemini 3.0 Flash等LLM进行评估。

**📊 数据集**

数据集为印度各地区收集的1,630条问答，覆盖8个日常生活领域，共计5,275份回答。

**📈 对比分析**

通过RASA和RA‑MCQ两种评测，八大LLM的整体准确率约为49–53%，完全正确率仅13–21%，且模型在无地区信息时显著偏向中部和北部答案。

**⚠️ 局限性**

局限包括仅聚焦印度、五地区划分可能掩盖更细微差异、样本主要为英语熟练者以及文化随时间演变的不可避免性。

---

## 145. From Generative Engines to Actionable Simulators: The Imperative of Physical Grounding in World Models

**arXiv ID:** 2601.15533 | [PDF](https://arxiv.org/pdf/2601.15533v1)

**作者:** Zhikang Chen `[一作]` (University of Oxford), Tingting Zhu `[通讯]` (University of Oxford)

**通讯引用:** 4933 | [OpenAlex ID](https://openalex.org/A5055850985)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述并重塑了世界模型的研究方向，将其从仅靠视觉生成的引擎转变为具备物理与因果推理的可操作仿真器，强调结构化4D接口、自适应进化与物理约束，并以医疗决策为高风险场景进行验证。

**💡 创新点**

创新点在于：①提出四大挑战（结构化接口、闭环自进化、物理锚定、有限交互泛化）构成统一框架；②把世界模型定位为“可执行仿真器”而非单纯生成器；③引入可解释的因果图、程序化专家、物理可微动力学等技术；④设计基于约束违背率和闭环决策表现的评估标准，强调在高风险领域的可解释性与可靠性。

**🔧 技术方法**

使用的技术包括：结构化3D/4D表示（点云、网格、记忆体）、稀疏因果图（SPARTAN）、程序化专家（PoE-World）、物理可微动力学（PIN‑WM）、潜在空间扩散和对齐（V‑JEPA、Diffusion‑World）、不确定性感知规划、逆动力学与动作推断、以及多模态与语言对齐技术。

**📊 数据集**

数据集方面未给出具体命名，作者主要依赖常用机器人与仿真环境（如机器人操控、导航、游戏）以及医疗影像与临床轨迹数据进行评估，强调对交互受限场景的通用数据生成与仿真。

**📈 对比分析**

对比方法主要通过传统的视觉质量指标（FID/FVD、LPIPS/SSIM、CLIPScore、VBench）与新提出的物理/因果一致性评估（违背率、能量守恒、常识合理性）以及闭环决策性能（WorldEval、WorldGym、任务成功率、策略回报）进行。实验表明，视觉质量高并不能保证决策表现，只有满足约束且在闭环中表现良好的模型才被视为真正的世界模型。

**⚠️ 局限性**

局限性包括：缺乏统一、可量化的物理/因果正确性指标，易受训练数据偏差放大导致的偏见，闭环自进化可能引起模型漂移，医疗等高风险场景下的假正率与临床安全性评估仍不足，且现有评估框架仍需进一步标准化与自动化。

---

## 146. Resource Allocation and Sharing for UAV-Assisted Integrated TN-NTN with Multi-Connectivity

**arXiv ID:** 2601.15532 | [PDF](https://arxiv.org/pdf/2601.15532v1)

**作者:** Abd Ullah Khan `[一作]` (Kyung Hee University), Hyundong Shin `[通讯]` (Kyung Hee University)

**通讯引用:** 7750 | [OpenAlex ID](https://openalex.org/A5007557286)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了无人机辅助的集成地面网络与非地面网络（TN‑NTN）中的资源分配与共享问题，针对具有多连通（MC）能力的异构无人机设计了两种资源分配算法，分别在满足可靠性约束的前提下最大化总容量或最大化最小容量以实现公平。

**💡 创新点**

创新点在于：① 将异构无人机（高容量用户HCUs和低容量用户LCUs）与不同QoS需求联合考虑；② 在单一频谱资源上实现一对一的共享与功率分配，兼顾可靠性和容量；③ 采用匈牙利算法与二分搜索求解非凸问题，实现多连通资源共享与功率控制的最优匹配；④ 仅利用大尺度信道统计，减少信号估计负担。

**🔧 技术方法**

主要技术包括：多连通无人机通信模型、功率控制与频谱共享的组合优化、可靠性约束下的二分搜索、匈牙利算法（Hungarian Method）进行一对一匹配、期望信道容量解析（利用指数积分函数）以及仿真评估。

**📊 数据集**

使用的是仿真数据：基站、HAP与无人机的随机布置（二维泊松过程），并采用固定的物理层参数（频率2 GHz、带宽10 MHz、路径损耗、阴影衰落、Rayleigh小尺度衰落）以及10³个信道样本来评估性能；未使用公开数据集。

**📈 对比分析**

与基线方案（无共享、贪婪共享、随机配对+最大功率、以及包含多连通的两个新基线）进行对比。结果表明：① 在总容量方面，算法1（最大化总容量）优于所有基线；② 在最小容量（公平性）方面，算法2表现最佳；③ 两种算法在不同J/I比、失效概率、速度、SINR阈值等参数下均保持显著优势，尤其在高密度或高速度场景下仍能保持较高容量和可靠性。

**⚠️ 局限性**

局限性包括：① 仅利用大尺度信道参数，忽略瞬时小尺度衰落可能导致容量上限被低估；② 频谱共享仅允许一对一匹配，若LCU数量多于HCU会导致资源浪费；③ 未考虑无人机轨迹优化、动态重分配、多基站/多卫星接入以及多天线/波束赋形等更复杂情形；④ 假设固定的功率上限与无多连通干扰耦合，实际部署中可能需要更细粒度的功率与天线配置。

---

## 147. Machine learning-enhanced non-amnestic Alzheimer's disease diagnosis from MRI and clinical features

**arXiv ID:** 2601.15530 | [PDF](https://arxiv.org/pdf/2601.15530v1)

**作者:** Megan A. Witherow `[一作]` (Old Dominion University), Khan M. Iftekharuddin `[通讯]` (Old Dominion University)

**通讯引用:** 5944 | [OpenAlex ID](https://openalex.org/A5099465602)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

通过随机森林模型结合临床评估结果和全脑MRI特征，构建了一套针对非短时记忆型阿尔茨海默病（atAD）与非AD认知障碍的诊断工具。

**💡 创新点**

创新点在于：① 将传统的海马体积单一指标扩展为全脑结构体积、皮层厚度和表面积的多维特征；② 采用Boruta方法系统识别并可视化对诊断具有显著贡献的大脑区域；③ 在三组数据集上统一验证，显著提升了对atAD的识别召回率。

**🔧 技术方法**

使用技术：随机森林（Random Forest）进行分类；Boruta算法用于特征重要性筛选；FreeSurfer处理得到MRI结构特征；5×2嵌套交叉验证评估模型性能；线性回归对MRI特征做年龄、性别、脑容积校正。

**📊 数据集**

数据集：ADNI公开数据库、NACC统一数据集以及旧金山地区的私人VHS内科临床数据，总计1410例，包含tAD、atAD、非AD认知障碍及健康对照。

**📈 对比分析**

与仅使用海马体积或临床特征的基线模型相比，加入全脑MRI特征后在NACC组召回率提升至0.69（从0.52），在ADNI组召回率提升至0.77（从0.34），同时保持高精度（≈0.77-0.89）。在VHS组，仅使用临床+海马特征已能达到82%召回，表明模型在不同人群中的稳健性。

**⚠️ 局限性**

局限性：① 数据集间特征分布差异导致模型泛化受限；② 私人数据集缺乏完整MRI特征，导致实验无法统一评估；③ 非AD组病因多样，缺乏细分诊断信息；④ 样本不平衡仍存在，对极少数类别的预测仍受影响；⑤ 仅使用结构MRI，未结合功能或分子影像，可能遗漏早期病变信息。

---

## 148. TransportAgents: a multi-agents LLM framework for traffic accident severity prediction

**arXiv ID:** 2601.15519 | [PDF](https://arxiv.org/pdf/2601.15519v1)

**作者:** Zhichao Yang `[一作]` (University of Maryland), Cirillo Cinzia `[通讯]` (University of Maryland)

**通讯引用:** 1969 | [OpenAlex ID](https://openalex.org/A5065206648)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了TransportAgent，一个多智能体LLM框架，用于交通事故严重程度预测。

**💡 创新点**

通过类别专门化的LLM代理与MLP融合实现分层推理、可解释且均衡预测；同时兼顾商业与开源模型。

**🔧 技术方法**

使用多智能体架构（特征选择、概念分类、专门化评估代理）、链式思考提示、MLP整合模块，以及GPT‑3.5‑turbo、GPT‑4o‑mini、LLaMA‑3.3‑70B‑Instruct等LLM。

**📊 数据集**

基于美国全国性交通事故/伤害数据库CPSRMS和NEISS。

**📈 对比分析**

与传统机器学习（MLP、Ordered Logit）和单体LLM提示（k‑shot、CoT、AutoGen）对比，在所有后端模型上均获得最高准确率，CPSRMS最高达73.31%，NEISS 76.9%。

**⚠️ 局限性**

仍依赖人工定义的类别划分，极少数高严重度样本预测仍有挑战；对模型解释性仍主要基于代理分数，无法完全揭示底层决策细节。

---

## 149. Investigation of the Generalisation Ability of Genetic Programming-evolved Scheduling Rules in Dynamic Flexible Job Shop Scheduling

**arXiv ID:** 2601.15717 | [PDF](https://arxiv.org/pdf/2601.15717v1)

**作者:** Luyao Zhu `[一作]` (Victoria University of Wellington), Mengjie Zhang `[通讯]` (Victoria University of Wellington)

**通讯引用:** 30584 | [OpenAlex ID](https://openalex.org/A5100400258)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文系统评估了遗传程序（GP）进化的调度规则在动态柔性车间排程（DFJSS）不同实例类型下的泛化能力，采用多维度实验（规模、工厂参数、分布）并结合决策点分布分析。

**💡 创新点**

创新点在于首次将决策点分布相似度与GP泛化性能关联，并展示了在大规模实例训练后能较好泛化到小规模实例；同时提出了基于分布相似度预测泛化性能的思路。

**🔧 技术方法**

使用遗传程序（GP）演化路由与排序规则，基于仿真评估平均加权延迟，利用Wilcoxon秩和检验进行统计比较，并计算决策点分布的重叠度、皮尔逊与斯皮尔曼相关系数。

**📊 数据集**

数据集为人工合成的DFJSS实例，涵盖不同机器/工作数量比例、利用率、到期因子、批量大小及五种参数分布（指数、伽马、对数正态、正态、均匀）。

**📈 对比分析**

通过将GP进化的规则在与训练相同或不同类型的测试实例上评估平均加权延迟，并与基准（训练/测试相同）进行对比；结果表明仅当训练与测试具有相似规模或参数分布时，性能相近或优于基准；差异越大，性能显著下降。

**⚠️ 局限性**

局限性在于GP规则对规模、参数或分布差异显著的实例泛化能力有限，且仍需改进以实现对异构环境的更强鲁棒性，未来可考虑终身学习或多任务学习等方法提升泛化性能。

---

## 150. Zero-Shot Product Attribute Labeling with Vision-Language Models: A Three-Tier Evaluation Framework

**arXiv ID:** 2601.15711 | [PDF](https://arxiv.org/pdf/2601.15711v1)

**作者:** Shubham Shukla `[一作]` (Cornell University), Kunal Sonalkar `[通讯]` (Nordstrom)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出三层评价框架，对多属性时尚图像属性预测进行细粒度评估，并在DeepFashion‑MultiModal上基准九种VLM。

**💡 创新点**

将整体任务、属性适用性检测和可见属性分类三层拆解，揭示VLM在NA检测上的瓶颈；同时对效率不同的VLM进行成本效能对比。

**🔧 技术方法**

零样本VLM推理（GPT‑5、Gemini等）、Fashion‑CLIP + 逻辑回归基线，三层宏F1评估与幻觉检测。

**📊 数据集**

DeepFashion‑MultiModal，5,000张测试图像、18个形状/织物/图案属性，并标注NA标签。

**📈 对比分析**

通过宏F1、NA‑F1、可见属性F1三维指标比较；零样本VLM平均64%宏F1，远超21%基线；高效模型在成本约1/5时仍保持≈90%性能。

**⚠️ 局限性**

基线仅为线性分类，未覆盖更强监督方法；仅评估两家专有VLM；未尝试少样本或链式思考；仅在DeepFashion评估，缺乏多样化真实场景。

---

## 151. Enhanced LULC Segmentation via Lightweight Model Refinements on ALOS-2 SAR Data

**arXiv ID:** 2601.15705 | [PDF](https://arxiv.org/pdf/2601.15705v1)

**作者:** Ali Caglayan `[一作]` (National Institute of Advanced Industrial Science and Technology), Toru Kouyama `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**通讯引用:** 5455 | [OpenAlex ID](https://openalex.org/A5076704471)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文在 ALOS‑2 单极化 SAR 数据上通过轻量化模型改进实现了国土土地利用/覆盖（LULC）语义分割和水体检测。

**💡 创新点**

创新点包括引入高分辨率特征注入、渐进式 refine‑up 上采样头以及 α‑尺度加权的 focal+dice 损失，以缓解边界模糊、细小结构缺失和类别不平衡问题。

**🔧 技术方法**

技术上采用 SAR‑W‑MixMAE 自监督预训练、Swin Transformer 编码器与 UPerNet‑style 解码器、轻量级卷积细化及 α‑比例调节的损失函数。

**📊 数据集**

使用日本全国范围内的 ALOS‑2 HH SAR 图像和 JAXA LULC 语义标签，构建约 1,000,000 个按类别逆频率采样的训练样本，拆分为预训练和微调集。

**📈 对比分析**

与无预训练基线及 RSSJ'25 先前工作对比，在 mIoU 上提升至 50.26%（比无预训练高 9%，比 RSSJ'25 高 3.4%），水体检测 IoU 提升至 93.64%，精度与召回亦同步提高。

**⚠️ 局限性**

局限性在于单极化 SAR 对不同结构的区分仍受限，尤其是人造结构类别混淆，且仍需要进一步降低对多时相与多极化数据的依赖。

---

## 152. FARM: Field-Aware Resolution Model for Intelligent Trigger-Action Automation

**arXiv ID:** 2601.15687 | [PDF](https://arxiv.org/pdf/2601.15687v1)

**作者:** Khusrav Badalov `[一作]` (Neouly Co., Ltd.), Young Yoon `[通讯]` (Hongik University)

**通讯引用:** 3084 | [OpenAlex ID](https://openalex.org/A5032316721)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 FARM，一个两阶段的触发-动作自动化生成系统，实现了从自然语言到可执行自动化配置的完整链路。

**💡 创新点**

结合对比学习的双编码器检索与多智能体 LLM 交互的选择，提出了基于功能级别的 schema 嵌入和层冻结策略，解决了服务级别识别与字段绑定难题。

**🔧 技术方法**

对比学习 (InfoNCE)、层冻结 (freeze lower transformer layers)、多智能体 LLM（IBM Granite 4.0）与链式思维提示、结构化 JSON 输出、近似最近邻检索。

**📊 数据集**

1,724 个触发函数、1,287 个动作函数的 IFTTT 功能级别数据集（共 2.2M 可能配对），共 16.5K 自动化规则作为训练/评估。

**📈 对比分析**

与 LAM、RecipeGen++、TARGE 等基线在服务级别和功能级别任务上对比，FARM 在功能级别的联合准确率达到 81%（Gold）/62%（Noisy）/70%（One-Shot），在服务级别上提高 21pp；检索阶段 R@5 92% 以上，整体系统成功率 97% 以上。

**⚠️ 局限性**

检索上限为 Joint R@5 85%，多意图查询、复杂逻辑、以及中等 faithfulness 分数（约 0.45）仍是瓶颈；依赖 LLM 性能，缺乏实时执行验证，未处理多触发多动作场景。

---

## 153. Consistency-Regularized GAN for Few-Shot SAR Target Recognition

**arXiv ID:** 2601.15681 | [PDF](https://arxiv.org/pdf/2601.15681v1)

**作者:** Yikui Zhai `[一作]` (Wuyi University), C. L. Philip Chen `[通讯]` (South China University of Technology)

**通讯引用:** 60136 | [OpenAlex ID](https://openalex.org/A5100643265)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在极少样本的 SAR 目标识别任务中，作者提出了一种新型的生成对抗网络 Cr-GAN，用以生成多样且高保真合成图像，并利用这些合成数据进行自监督预训练，最终显著提升少样本分类性能。

**💡 创新点**

创新点：
- 双分支判别器（adversarial + representation），将判别和特征学习解耦，避免少量数据导致的过拟合；
- 在特征空间进行通道级插值（Channel‑Interpolation）来产生新的潜在表示，并通过特征循环一致性与图像循环一致性进行正则化；
- 引入模式寻求（mode‑seeking）损失，防止生成器模式崩溃；
- 通过自监督对比学习（SimCLR）对合成数据进行预训练，无需额外标签，提升特征泛化能力。

**🔧 技术方法**

使用技术：
- Cr‑GAN（基于 VAE‑GAN 思想的双分支 GAN）
- 通道插值与循环一致性正则化
- 模式寻求损失
- SimCLR 自监督预训练
- 对比损失、KL 正则化、特征一致性损失
- 基础的 DCGAN、StyleGAN2、R3GAN 等 GAN 架构作为对比实验。

**📊 数据集**

实验数据集：
- MSTAR（军用目标 64×64 像素，10 类）
- SRSDD（船舶检测数据集，6 类）

**📈 对比分析**

比较方法：
- 与传统 GAN（DCGAN、R3GAN、StyleGAN2）和扩散模型（DDPM、DiT、SiT、EDM2）进行对比；
- 在 2‑shot、4‑shot、8‑shot 设定下评估准确率、F1、召回率等指标；
- 结果显示：在 8‑shot MSTAR 上 Cr‑GAN 达到 71.21% 准确率，远高于最优的扩散模型 71.10%；在 SRSDD 8‑shot 上获得 51.64% 准确率，明显优于其他基线；
- 参数量仅 13.71M，训练时间 0.27h，显著低于扩散模型（>200M 参数、>1h）。

**⚠️ 局限性**

局限性：
- 仍需在极端极少样本（<2‑shot）或不同 SAR 模式下进一步验证鲁棒性；
- 对生成器的稳定性仍依赖于内部正则化，极端数据稀缺时可能出现训练不收敛；
- 目前仅在分类任务上评估，未覆盖目标检测、语义分割等更复杂的遥感任务；
- 对生成图像的感知质量评估仍以下游任务性能为主，缺乏针对 SAR 图像的专门评价指标。

---

## 154. StreetDesignAI: A Multi-Persona Evaluation System for Inclusive Infrastructure Design

**arXiv ID:** 2601.15671 | [PDF](https://arxiv.org/pdf/2601.15671v1)

**作者:** Ziyi Wang `[一作]` (University of Maryland), Xiang Yan `[通讯]` (University of Florida)

**通讯引用:** 1890 | [OpenAlex ID](https://openalex.org/A5020824659)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并评估了名为 StreetDesignAI 的交互系统，利用多种骑行者人设并行评估，基于街道图像和 OpenStreetMap 数据情境化评估，并支持通过可视化参数迭代生成街景方案。

**💡 创新点**

创新点在于将人设多代理评估与快速街景生成相结合，显式展示用户体验冲突，支持设计者在同一工作流中进行冲突识别、比较与权衡；系统以真实街道语境为基础，提供多视角反馈而非单一 AI 建议。

**🔧 技术方法**

采用 GPT‑4.1 微调模型生成人设评估，使用 GPT‑Image‑1 生成街景图像，结合 Google Street View 与 OpenStreetMap API 获取街道语境，配合结构化交互、可视化模块、多轮对话和比较分析功能实现系统。

**📊 数据集**

数据集包括：Google Street View 街景图像与 OSM 属性；来自 427 名骑行者的 12,400 条 crowdsourced bikeability 评分（安全、舒适、整体分数及开放性问题），用于微调 LLM；以及相关交通与设计标准文献。

**📈 对比分析**

通过 26 名交通专业人员的 within‑subjects 研究，将 StreetDesignAI 与 ChatGPT‑4.1 基准进行比较，使用 Wilcoxon signed‑rank、t 检验等统计方法。结果表明，StreetDesignAI 在设计探索、用户需求理解、翻译能力、冲突可视化等方面显著优于基准，用户满意度和专业使用意愿也显著提升。

**⚠️ 局限性**

局限性包括：人设评估未与真实骑行者反馈系统性验证，可能缺乏某些群体或地区代表性；街景生成偶尔出现不准确；实验任务受限于受控环境，未检验系统在真实项目流程和长期效果；AI 反馈需与真实公众参与相结合；此外可能存在偏见和过度依赖量化评分的问题。

---

## 155. SuperOcc: Toward Cohesive Temporal Modeling for Superquadric-based Occupancy Prediction

**arXiv ID:** 2601.15644 | [PDF](https://arxiv.org/pdf/2601.15644v1)

**作者:** Zichen Yu `[一作]` (Dalian University of Technology), Xiaoguang Zhao `[通讯]` (Dalian Rail Transmit Intelligent Control and Intelligent Operation Technology Innovation Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于超四边形的3D占用预测框架SuperOcc，统一实现视图中心和对象中心的时间建模，使用多超四边形解码和高效的超四边形-体素投影。

**💡 创新点**

创新点包括：1）协同时间建模机制，融合视图中心与对象中心两种时间建模；2）多超四边形解码策略，在保持查询稀疏的同时提升几何表达能力；3）基于CUDA的高效超四边形到体素的投影实现，显著降低计算开销。

**🔧 技术方法**

主要技术包括：多视角图像编码器（ResNet-50+FPN），自注意力查询交互，稀疏查询的时间记忆与对齐，参数化的超四边形几何与语义回归，基于共享内存的块级投影算法。

**📊 数据集**

使用nuScenes数据集，并在其上构建的Occ3D和SurroundOcc两个占用标注基准进行评估。

**📈 对比分析**

与现有最先进方法（如OPUS、QuadricFormer、GaussianWorld等）对比，SuperOcc在Occ3D上RayIoU提升至约38.1%，在SurroundOcc上IoU和mIoU均高于对手，同时保持30+ FPS的实时推理速度，展示了优异的性能与效率平衡。

**⚠️ 局限性**

局限性包括：仅支持语义占用预测，缺乏实例级别的分割；依赖车载运动补偿，对动态物体的时空对齐不够精细，可能导致动态物体的预测误差。

---

## 156. Improving Methodologies for LLM Evaluations Across Global Languages

**arXiv ID:** 2601.15706 | [PDF](https://arxiv.org/pdf/2601.15706v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 157. Dancing in Chains: Strategic Persuasion in Academic Rebuttal via Theory of Mind

**arXiv ID:** 2601.15715 | [PDF](https://arxiv.org/pdf/2601.15715v1)

**作者:** Zhitao He `[一作]` (Hong Kong University of Science and Technology), Yi R Fung `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于 Theory of Mind 的学术反驳框架 RebuttalAgent，并构建了大规模合成数据集 RebuttalBench 与专用评估模型 Rebuttal-RM，完成从数据生成到模型训练（SFT+RL）以及自动评估的完整闭环。

**💡 创新点**

创新点包括：1) 首次将 Theory of Mind 引入反驳，形成 ToM‑Strategy‑Response 三阶段推理流程；2) 采用 critique‑and‑refine 的新型数据合成方法；3) 开发自奖励机制实现无外部奖励模型的 RL；4) 训练的 Rebuttal-RM 超越 GPT‑4.1 与其他强基线。

**🔧 技术方法**

使用的大型语言模型（Qwen3‑8B、GPT‑4.1、Claude 3.5 等），配合检索式上下文获取、ToM 推理、策略生成、基于 GRPO 的强化学习、以及自奖励机制与评估模型训练。

**📊 数据集**

主要数据集包括：70K 条 RebuttalBench（由 Re^2 数据集生成），100K+ 用于训练评估器的多源评估样本；基准对照集包含 o3、GPT‑4.1、DeepSeek‑R1/V3、Gemini‑2.5、GLM‑4‑9B、Llama‑3.1‑8B、Qwen3‑8B 等；测试集 R2‑test（6000 条评论）与外域 Rebuttal‑test（2000 条评论）。

**📈 对比分析**

通过与多种基线（基础模型与代理方法）在自动化评估（Rebuttal‑RM）和人工评估（4 维度评分）上进行对比；在 R2‑test 上平均得分 9.42，较 Qwen3‑8B 提升 18.3%，并在外域测试亦保持领先；人工评估最高平均 9.57，全面超越 o3、GPT‑4.1 等。

**⚠️ 局限性**

局限性包括：仅处理不涉及实验结果的评论以防捏造；训练数据仍可能携带偏见；需要人工审核以保证准确性；对小模型或跨学科场景的适用性尚未充分验证。

---

## 158. Balancing Security and Privacy: The Pivotal Role of AI in Modern Healthcare Systems

**arXiv ID:** 2601.15697 | [PDF](https://arxiv.org/pdf/2601.15697v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 159. FlexLLM: Composable HLS Library for Flexible Hybrid LLM Accelerator Design

**arXiv ID:** 2601.15710 | [PDF](https://arxiv.org/pdf/2601.15710v1)

**作者:** Jiahao Zhang `[一作]` (University of California), Jason Cong `[通讯]` (University of California)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出FlexLLM，一个可组合的高层次综合(HLS)库，快速构建针对大型语言模型（LLM）的域特定加速器；通过在预填(prefill)和解码(decode)阶段分别采用不同的混合架构和完整的低比特量化堆栈，完成了Llama‑3.2‑1B的完整推理系统；

**💡 创新点**

创新点在于①支持阶段定制的混合时空并行架构，突破传统统一设计的瓶颈；②将静态/动态、对称/非对称、多粒度量化及异常处理模块化集成到库中；③采用模板化、可组合的HLS模块大幅缩短设计周期（从数月降至数周）；④与Hierarchical Memory Transformer（HMT）插件协同，实现超长上下文压缩。

**🔧 技术方法**

技术包括基于TAPA的HLS实现、token_parallelism/weight_parallelism与block_parallelism等并行度参数、SpinQuant改进的低比特量化、动态/静态量化与旋转/FHT异常处理、混合时空数据流、ILP资源调度、AutoBridge P&R以及HMT插件的分段压缩等。

**📊 数据集**

使用WikiText‑2文本集评估模型 perplexity（PPL），并在Llama‑3.2‑1B模型上进行推理实验；长上下文实验采用大长度 prompt（多段）进行性能验证。

**📈 对比分析**

与 NVIDIA A100 GPU（BF16+vLLM）、Allo、GPTQ‑Marlin 等基线对比；在 AMD Alveo U280 FPGA 上实现了1.29×端到端加速、1.64×解码吞吐、3.14×能效提升；在 7nm V80 预计可达4.71×/6.55×/4.13×；长上下文模式下，HMT 插件将 prefill 延迟压缩至 23.23×，上下文窗口扩大 64×，能效提升 5.21×/6.27×。

**⚠️ 局限性**

限制包括：量化仍以静态/动态为主，对更低精度或更大模型的支持有限；FPGA 逻辑与内存带宽仍受限，可能限制极大规模 LLM 的部署；设计需要掌握 TAPA/HLS，初学者门槛较高；HMT 插件的压缩可能对生成质量产生一定影响。

---

## 160. D-Optimality-Guided Reinforcement Learning for Efficient Open-Loop Calibration of a 3-DOF Ankle Rehabilitation Robot

**arXiv ID:** 2601.15707 | [PDF](https://arxiv.org/pdf/2601.15707v1)

**作者:** Qifan Hu `[一作]` (Affiliated Provincial Hospital Shandong First Medical University), Steven W. Su `[通讯]` (Faculty of Engineering and IT University of Technology Sydney)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了基于 Kronecker 乘积的开环校准方法，并用强化学习选取最具信息量的姿态进行参数辨识。

**💡 创新点**

创新点在于将校准问题转化为 D‑optimal 实验设计，并通过 PPO 学习在有限姿态预算内自动挑选信息最丰富的四个姿态。

**🔧 技术方法**

主要使用 Kronecker 乘积参数化、D‑optimal 信息矩阵、Proximal Policy Optimization（PPO）强化学习、注意力网络与稀疏奖励。

**📊 数据集**

实验基于仿真数据（多套随机姿态集合）以及真实 3‑DOF 脚踝康复机器人的 550 条输入输出对，另外对 100 次独立实验做评估。

**📈 对比分析**

与随机选择基线对比，PPO 选姿态的 log‑det(信息矩阵) 平均提升超过两百倍，方差显著降低，参数估计的跨周期方差也更小，预测精度更高。

**⚠️ 局限性**

局限在于需要先在仿真中训练模型，且只针对一次性校准，未考虑长期漂移或多周期动态变化的自适应校准。

---

## 161. Beyond Visual Safety: Jailbreaking Multimodal Large Language Models for Harmful Image Generation via Semantic-Agnostic Inputs

**arXiv ID:** 2601.15698 | [PDF](https://arxiv.org/pdf/2601.15698v1)

**作者:** Mingyu Yu `[一作]` (Beijing University of Posts and Telecommunications), Sujuan Qin `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 3817 | [OpenAlex ID](https://openalex.org/A5040230786)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 BVS 框架，用语义解耦与指令重组的“重构-生成”策略突破多模态 LLM 的视觉安全，生成有害图像。

**💡 创新点**

创新点在于将恶意语义分片后与中性图像拼接，并通过 MIDOS 选取最具“语义稀释”效果的补丁，从而绕过模型的视觉安全门槛。

**🔧 技术方法**

采用 CogView4 作为视觉引导模型，MIDOS 多图像距离优化选择算法，中文诱导提示，以及图像拼接与语义稀释技术。

**📊 数据集**

使用了由 110 条高危恶意文本提示和 25 张中性图像组成的数据集，用以评估模型的视觉安全边界。

**📈 对比分析**

与 Perception‑Guided 和 Chain‑of‑Jailbreak 基线对比，BVS 在 GPT‑5 上实现 98.18% 成功率、在 Gemini 1.5 Flash 上实现 95.45% 成功率，显著优于基线。

**⚠️ 局限性**

局限性包括：仅关注图像生成安全；攻击效果高度依赖拼接与中性图像质量；实验仅覆盖两款模型，需进一步验证通用性；未对跨模态完整语义检测与实时防御进行探索。

---

## 162. What Patients Really Ask: Exploring the Effect of False Assumptions in Patient Information Seeking

**arXiv ID:** 2601.15674 | [PDF](https://arxiv.org/pdf/2601.15674v1)

**作者:** Raymond Xiong `[一作]` (Duke University), Monica Agrawal `[通讯]` (Duke University)

**通讯引用:** 2041 | [OpenAlex ID](https://openalex.org/A5088961661)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过收集Google People Also Ask中的药物相关问题，构建了包含真实患者式医疗问题及其问答轨迹的问卷，并对其进行错误假设和危险意图的三类标注。

**💡 创新点**

创新点在于将真实搜索行为映射为问答序列，量化错误假设的出现机理，并系统评估大语言模型在识别此类错误方面的表现。

**🔧 技术方法**

使用的技术包括基于随机深度优先搜索的问答收集、自动化三类标签器（少量示例提示）、LLM回答生成与评估、Logistic回归和比例检验等统计方法。

**📊 数据集**

数据集来源于Google PAA，基于美国前200名常用处方药的搜索生成，共收集4012个问题，其中约16%为错误假设、7.5%为危险意图。

**📈 对比分析**

方法通过将高置信度错误问题提供给10款LLM（含GPT‑4o、GPT‑5、Claude Haiku等），评估其识别错误假设和危险意图的准确率，发现闭源模型最高达92%，但多数开源模型低于50%。

**⚠️ 局限性**

限制包括仅使用英文PAA数据、基于主观标签可能不完全客观、未覆盖不同地区/语言的药物信息，以及缺乏真实患者生成的自然语料。

---

## 163. Enhancing guidance for missing data in diffusion-based sequential recommendation

**arXiv ID:** 2601.15673 | [PDF](https://arxiv.org/pdf/2601.15673v1)

**作者:** Qilong Yan `[一作]` (Sun Yat-sen University), Jian Yin `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了CARD模型，通过对用户序列的稳定性进行动态路由，并对低稳定性序列采用因果注意力机制来优化扩散模型的引导信号，从而提升顺序推荐的质量。

**💡 创新点**

创新点包括：① 基于序列稳定性的路由策略；② 通过因果（counterfactual）注意力计算预测误差减小（PER）来衡量每个项目的重要性；③ 仅对低稳定性序列使用高成本注意力，显著降低计算开销。

**🔧 技术方法**

主要技术包括：扩散模型（Diffusion Model）、对比学习与分类器自由引导、Transformer编码器、因果注意力机制、稳定性评估（基于相似度熵）和路由策略。

**📊 数据集**

在Zhihu和KuaiRec两个真实业务数据集上进行实验。

**📈 对比分析**

与传统递归、Transformer推荐器、生成式推荐器（DiffRec、DreamRec、TDM）以及基于恢复的模型（PDRec、SSDRec、STEAM、DiffuASR）进行对比。CARD在HR@20和NDCG@20上均取得最优成绩，Zhihu上相对TDM提升约10.3%（HR）和5.1%（NDCG），KuaiRec上也保持领先；训练与推理时间仅略高于TDM，显著优于DreamRec。

**⚠️ 局限性**

局限性：对高稳定性序列仍依赖传统DTS去冗余，可能无法捕捉某些细粒度兴趣变化；因果注意力计算仍需要额外的辅助预测任务，对模型结构与训练过程带来一定复杂性；在极端缺失率或噪声比例极高的场景下，PER估计可能不够稳健。

---

## 164. Skywork UniPic 3.0: Unified Multi-Image Composition via Sequence Modeling

**arXiv ID:** 2601.15664 | [PDF](https://arxiv.org/pdf/2601.15664v1)

**作者:** Hongyang Wei `[一作]`, Yahui Zhou `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Skywork UniPic 3.0 统一框架，支持任意 1~6 张输入图像的多图合成与单图编辑，特别聚焦人-物交互 (HOI) 场景。

**💡 创新点**

创新点在于：① 将单图编辑与多图合成建模为统一序列条件生成，② 构建高质量 215K HOI 合成数据集，③ 结合轨迹映射与分布匹配实现仅 8 步高质量推理。

**🔧 技术方法**

采用流匹配 MMDiT、Qwen2.5-VL 条件编码、VAE 视觉分辨率、连续一致性模型 (sCM)、分布匹配蒸馏 (reverse KL) 与混合训练策略。

**📊 数据集**

使用内部 215K HOI 合成样本、338K 合成与 381K 单图编辑样本（含 Mico-150K、Nano-consistent-150K 等），并在公开数据集上进一步验证。

**📈 对比分析**

在 ImgEdit-Bench、GEdit-Bench 与自建 MultiCom-Bench 上与 Qwen-Image-Edit、Nano-Banana、Seedream 4.0 等基线对比，UniPic 3.0 单图编辑得分 4.35/7.55，MultiCom-Bench 总体 0.7255，均达到或超过目前最先进水平。

**⚠️ 局限性**

局限性：对输入图像数量超过 6 张或极高分辨率场景的可扩展性待验证，且在光照、视角冲突极端情况下的鲁棒性仍需提升。

---

## 165. Bridging Qualitative Rubrics and AI: A Binary Question Framework for Criterion-Referenced Grading in Engineering

**arXiv ID:** 2601.15626 | [PDF](https://arxiv.org/pdf/2601.15626v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 166. Evolving Without Ending: Unifying Multimodal Incremental Learning for Continual Panoptic Perception

**arXiv ID:** 2601.15643 | [PDF](https://arxiv.org/pdf/2601.15643v1)

**作者:** Bo Yuan `[一作]` (Beihang University), Zhiguo Jiang `[通讯]` (Beihang University)

**通讯引用:** 5742 | [OpenAlex ID](https://openalex.org/A5016853247)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种持续 panoptic perception (CPP) 框架，实现了在连续学习过程中对像素级、实例级和图像级任务的多模态协同推理。

**💡 创新点**

创新点在于：① 共享跨模态编码器 (CCE) 统一提取图像和文本特征；② 可塑性知识蒸馏 (MCKD) 通过交叉任务对比蒸馏与实例蒸馏同时抑制灾难性遗忘；③ 双向一致性约束 (CBC) 强化图像与文本的语义对齐；④ 异步伪标签 (SAPL) 在无示例重放条件下动态生成高置信度伪标签；同时在单模型中实现多任务协同优化。

**🔧 技术方法**

采用 Transformer 编码/解码结构、对比蒸馏、知识蒸馏、双向一致性约束、伪标签技术及多任务加权损失，构建端到端的多模态持续学习系统。

**📊 数据集**

实验使用 FineGrip（遥感多模态数据）、ADE20K（细粒度语义分割）和 COCO（分割+字幕）三个公开数据集；同时在不同 backbone（ResNet‑101、MaskFormer/Mask2Former）上验证。

**📈 对比分析**

与传统 fine‑tune、MaskFormer、Mask2Former、SAT、VitCAP、LAG 等基线以及无增量学习的离线模型进行对比。CPP 在 FineGrip PQ 上提升约 10‑15%，CPP+ 进一步提升至 20‑25%；在 ADE20K mIoU 上提高 1‑2%；在 COCO PQ 及 BLEU 也获得 1‑3% 的改进，整体明显优于现有方法。

**⚠️ 局限性**

局限性包括：① 计算成本和推理速度显著高于单任务模型；② 对类学习顺序敏感，仍易受语义混淆影响；③ 目前仅覆盖图像‑文本两种模态，尚未通用到更多模态；④ 多任务收敛平衡问题仍需进一步改进。

---

## 167. Side-Channel Attacks on Open vSwitch

**arXiv ID:** 2601.15632 | [PDF](https://arxiv.org/pdf/2601.15632v1)

**作者:** Daewoo Kim `[一作]` (University of Waterloo), Sihang Liu `[通讯]` (University of Waterloo)

**通讯引用:** 487 | [OpenAlex ID](https://openalex.org/A5101571781)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对 Open vSwitch（OVS）的缓存机制进行安全分析，提出并实现了三类远程侧信道攻击：基于微流缓存的隐写通道、基于微流缓存哈希碰撞的报文头恢复攻击以及基于巨流缓存子表重排的报文速率监控攻击，并给出了相应的防御方案。

**💡 创新点**

创新点在于首次揭示 OVS 缓存层的侧信道风险，利用微流/巨流缓存的哈希、命中时间差和子表重排规律，构造出能够跨虚拟机、远程破坏隔离的攻击；同时提出了基于实例隔离、哈希随机化、子表重排随机化的三种可行防御措施。

**🔧 技术方法**

攻击与防御技术主要包括：Prime+Probe 和 Flush+Reload 的 RTT 监测、基于微流缓存哈希冲突的探测、巨流缓存子表访问频率重排分析、哈希函数逆向与离线暴力搜索、以及随机化调度与实例隔离等。

**📊 数据集**

实验使用了 UNSW‑NB15 数据集（真实网络流量）以及在实验室搭建的基于 Memcached 的虚拟机环境进行验证。

**📈 对比分析**

与现有远程侧信道攻击（如 NetSpectre、NetCAT 等）比较，本文实现的微流缓存隐写通道速率可达 15.8 bps，误码率 3.2%；报文头恢复攻击在 100 次重复测量后准确率可达 93%；报文速率监控攻击在 30 条流上平均准确率约 71.9%，相较于传统的粗粒度监测方法具有更高的实时性和精度。

**⚠️ 局限性**

主要局限包括：攻击需要长时间的目标流量才能聚集足够的哈希冲突；在 OVS 使用 DPDK 数据路径时才具备微流缓存，若使用 kernel 路径则部分攻击失效；攻击耗时较长（数十至数百秒），不适用于短周期流；对网络噪声有一定鲁棒性但在高噪声场景下准确率会下降；需对 OVS 的哈希实现有完整了解。

---

## 168. CogToM: A Comprehensive Theory of Mind Benchmark inspired by Human Cognition for Large Language Models

**arXiv ID:** 2601.15628 | [PDF](https://arxiv.org/pdf/2601.15628v1)

**作者:** Haibo Tong `[一作]` (Chinese Academy of Sciences), Yi Zeng `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 10313 | [OpenAlex ID](https://openalex.org/A5108421411)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并构建了CogToM评测框架，涵盖46个心理学验证的Theory of Mind（ToM）任务，生成了超过8,000条中英双语实例，并对22种LLM进行了系统评估。

**💡 创新点**

创新点在于：①将成熟心理学ToM范式改写为LLM可直接使用的多轮多选任务，②大幅提升任务覆盖面和辨别力，③揭示LLM在不同认知维度上的差异及“Moravec悖论”现象。

**🔧 技术方法**

技术手段包括：心理学任务改编与标准化、LLM自动扩展生成、专家双盲人工注释与验证，以及零样本、温度为0、多次选项随机排列的评测流程。

**📊 数据集**

使用的数据集为CogToM benchmark，包含46类任务、8,513条中英双语实例，均由49名人工专家多轮验证。

**📈 对比分析**

比较方法采用零样本提示、交叉选项排列、平均准确率评估；结果显示前沿模型（如GPT‑5.1、Qwen3‑Max）在情感、欲望、非文字推理等任务上达到80‑95%高分，而在感知、信念等维度仅20%以下，体现显著性能差异。

**⚠️ 局限性**

局限性包括：仅覆盖中文和英文两种语言，缺乏多模态与动态交互能力；任务为静态多选，难以捕捉开放式对话中的递归推理；需要进一步扩展至其他文化、语言和更真实的情境。

---

## 169. CoNRec: Context-Discerning Negative Recommendation with LLMs

**arXiv ID:** 2601.15721 | [PDF](https://arxiv.org/pdf/2601.15721v1)

**作者:** Xinda Chen `[一作]` (Fudan University), Yuning Jiang `[通讯]` (Alibaba Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出CoNRec框架，利用大语言模型和语义ID生成用户未来可能产生负向反馈的商品集合。

**💡 创新点**

创新点包括：①首次将LLM与语义ID结合用于负向推荐；②引入项目级对齐任务提升对负面属性的识别；③采用渐进式GRPO训练并使用未来7天负向与协同正向反馈奖励与评估，缓解负向反馈稀疏和系统曝光偏差问题。

**🔧 技术方法**

使用技术包括：Qwen3‑14B LLM、基于多模态编码和RQ‑VAE的语义ID、LoRA微调、GRPO强化学习、基于未来正负反馈的自定义奖励函数。

**📊 数据集**

在淘宝工业级真实用户行为日志（含正负交互序列）上进行实验。

**📈 对比分析**

与生成式ID/特征/语义ID/LLM等多种基线对比，CoNRec在HR@20、FHR@20、LUF@20、LIF@20和候选准确率上分别提升约11%–25%及约100%，在工业级场景中显著优于最优基线。

**⚠️ 局限性**

局限性在于仍依赖离线生成，难以直接适应实时推荐顺序；正负反馈稀疏和系统曝光机制仍可能影响学习效果，且在线实时推理的延迟未完全解决。

---

## 170. Performance-guided Reinforced Active Learning for Object Detection

**arXiv ID:** 2601.15688 | [PDF](https://arxiv.org/pdf/2601.15688v1)

**作者:** Zhixuan Liang `[一作]` (University of Hong Kong), Ping Luo `[通讯]` (University of Hong Kong)

**通讯引用:** 52342 | [OpenAlex ID](https://openalex.org/A5100752686)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种以检测任务性能（mAP）为奖励的强化学习驱动主动学习框架，用于选择最能提升目标检测模型表现的样本批次。

**💡 创新点**

创新点包括：①直接使用ΔmAP作为奖励，直接对任务性能进行优化；②采用RL采样代理解决离散、非可微批次选择问题；③使用无监督近似和查找表加速mAP估计，使方法在实际部署中可行。

**🔧 技术方法**

技术手段包括强化学习（policy gradient）、LSTM序列模型、无监督一致性学习（ISD-SSD等）、Wasserstein距离近似、查找表（LUT）加速、基于mAP的奖励归一化。

**📊 数据集**

在PASCAL VOC和MS COCO两个常用目标检测基准数据集上进行实验，使用SSD（VOC）和RetinaNet（COCO）作为检测器。

**📈 对比分析**

与随机、entropy、Core-set、CDAL、LL4AL、MIAL、EBAL、MEH+HUA、PPAL等传统主动学习方法比较，本文方法在VOC上始终领先，COCO上也显示出更快的mAP提升曲线，显著提高样本利用率。

**⚠️ 局限性**

局限性包括：对无监督近似的依赖可能导致估计误差；查找表需要预训练多模型，初始成本较高；在极大未标注池下仍需进一步优化搜索空间。

---

## 171. TempoNet: Learning Realistic Communication and Timing Patterns for Network Traffic Simulation

**arXiv ID:** 2601.15663 | [PDF](https://arxiv.org/pdf/2601.15663v1)

**作者:** Kristen Moore `[一作]` (CSIRO Data61), Seyit Camtepe `[通讯]` (CSIRO Data61)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 TempoNet，基于多任务学习与多标记时点过程的生成模型，用于合成具有逼真时间和通信模式的网络流量。

**💡 创新点**

创新点在于将日志正态混合时点过程与多任务学习结合，能够同时建模到达间隔与所有报文/流的标头字段，并捕捉日周期、周周期等高阶时序特征。

**🔧 技术方法**

技术主要包括 LSTM 历史编码、LogNormMix 时点过程、类别分布预测层以及多任务损失，实现多标记生成。

**📊 数据集**

使用四个公开数据集：LANL、CIDDS、DC、IoT NetFlow/PCAP 记录。

**📈 对比分析**

与 GAN、LLM、贝叶斯网络等基线相比，TempoNet 在真实度、分布多样性、合规性和 IDS 训练效果上均获得最高或第二名，尤其在时间一致性和多标记精确度上表现突出。

**⚠️ 局限性**

局限性包括对稀疏事件建模不足、难以捕捉多重季节性交互、仅生成标头不含负载，以及在极端分布或长距离依赖场景下可能失真。

---

## 172. Integrating Knowledge Distillation Methods: A Sequential Multi-Stage Framework

**arXiv ID:** 2601.15657 | [PDF](https://arxiv.org/pdf/2601.15657v1)

**作者:** Yinxi Tian `[一作]` (Southern University of Science and Technology), Xin Yao `[通讯]` (Lingnan University)

**通讯引用:** 66482 | [OpenAlex ID](https://openalex.org/A5100635494)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了顺序多阶段知识蒸馏（SMSKD）框架，按阶段逐步整合多种知识蒸馏方法，并通过冻结的参考模型和自适应参考权重来防止知识遗忘，提升学生模型性能。

**💡 创新点**

创新点包括：① 将不同类型的蒸馏方法（响应、特征、关系等）按阶段顺序串联，打破传统一次性融合的局限；② 在阶段切换时使用冻结的参考模型提供知识锚定，避免灾难性遗忘；③ 依据教师真实类别概率（TCP）动态加权参考损失，实现自适应的知识保留与更新；④ 该框架对蒸馏方法无约束，能自由组合任意数量与类型的蒸馏技术，且对计算开销影响极小。

**🔧 技术方法**

采用知识蒸馏核心技术（KL 对齐、特征对齐、关系对齐等），顺序训练与参考损失（KL），自适应权重（TCP），以及常规训练策略（SGD、学习率衰减等）。

**📊 数据集**

实验使用 CIFAR‑100 和 Tiny ImageNet 两个经典图像分类数据集。

**📈 对比分析**

与单源蒸馏方法、Sakd（自适应多源蒸馏）和 DLA（直接损失聚合）进行对比。SMSKD 在 ResNet、WRN、VGG 等多种教师‑学生组合上平均提升 1%–2% 的准确率，且在多种蒸馏方法组合下始终保持最优或相近最优，表现出更稳健、更高的性能提升。

**⚠️ 局限性**

局限性：① 需要预先设定阶段数与切换点，若切换过早或过晚可能影响效果；② 参考模型可能在早期阶段误导后续学习，导致权重调优敏感；③ 在某些方法组合（如 CRD+VID）可能不如单一方法提升；④ 随着阶段数增多，收益递减，额外训练成本上升；⑤ 目前仅在图像分类任务验证，未在更大规模或不同任务上进行广泛验证。

---

## 173. Predictive Coding and Information Bottleneck for Hallucination Detection in Large Language Models

**arXiv ID:** 2601.15652 | [PDF](https://arxiv.org/pdf/2601.15652v1)

**作者:** Manish Bhatt `[一作]` `[通讯]` (Open Web Application Security Project), Manish Bhatt (Open Web Application Security Project)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于神经科学理论的混合框架PCIB，用以检测大语言模型的幻觉。

**💡 创新点**

创新点在于将预测编码与信息瓶颈原理转化为可解释信号，并引入实体聚焦、上下文遵从和可证伪度三大增强。

**🔧 技术方法**

使用的技术包括预测编码/信息瓶颈导出的Uptake、Stress、Conflict、Rationalization信号，结合随机森林/Meta‑Ensemble等监督学习。

**📊 数据集**

实验数据集为HaluBench 200样本（平衡）与Lynx的15k全量对照。

**📈 对比分析**

与Lynx及LLM‑Judge对比，最佳随机森林模型AUROC 0.8669，使用仅200样本，推理速度5ms，成本0.001美元/千请求。

**⚠️ 局限性**

局限在于仅在单语料、RAG场景验证，且Rationalization信号未能提升性能。

---

## 174. Advancing RT Core-Accelerated Fixed-Radius Nearest Neighbor Search

**arXiv ID:** 2601.15633 | [PDF](https://arxiv.org/pdf/2601.15633v1)

**作者:** Enzo Meneses `[一作]` (Instituto de Informática, Universidad Austral de Chile), Maxime Maria `[通讯]` (Univ. of Limoges)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了使用GPU Ray Tracing 核心加速固定半径最近邻搜索，并提出了基于成本函数的梯度自适应 BVH 更新/重建策略、无邻居列表的 ORCS 变体以及通过射线实现周期边界条件的方法。

**💡 创新点**

①基于梯度的自适应 BVH 更新/重建比例优化；②两种 ORCS 变体实现完整物理模拟而无需邻居列表；③利用额外射线实现周期边界而不需要额外计算内核。

**🔧 技术方法**

GPU Ray Tracing 核心（OptiX）、CUDA、NVIDIA NVML 监控、BVH 更新/重建、Lennard‑Jones 势能模型、粒子分布（格点、随机、聚簇）和半径分布（常数、小、随机、对数正态）。

**📊 数据集**

Lennard‑Jones 粒子模拟数据集，粒子数从 50k 到 1M，采用三种粒子分布（格点、随机、聚簇）与四种半径分布（r=1、r=160、随机 [1,160]、对数正态 [1,330]），运行于 Patagón 超级计算机的 RTX‑Pro 6000。

**📈 对比分析**

与 CPU‑CELL、GPU‑CELL、RT‑REF 等基线对比，采用平均时间、加速比和能效（交互/焦耳）评估。结果显示，ORCS 及自适应 BVH 在小半径或对数正态分布下可比基线快 1.3‑2.0 倍，梯度自适应 BVH 可比固定率快 3.4 倍；能效在小半径/对数正态下最优，CPU/常规 GPU 在大半径或聚簇场景下更具优势。

**⚠️ 局限性**

当粒子半径大或聚簇导致邻居数极高时，RT 核心方法易内存不足或效率下降；ORCS‑forces 需要原子操作，受硬件限制；周期边界处理对极大半径粒子产生额外射线负担；未实现基于能耗的自适应调优。

---

## 175. U3-xi: Pushing the Boundaries of Speaker Recognition via Incorporating Uncertainty

**arXiv ID:** 2601.15719 | [PDF](https://arxiv.org/pdf/2601.15719v1)

**作者:** Junjie Li `[一作]` (Hong Kong Polytechnic University), Kong Aik Lee `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 7035 | [OpenAlex ID](https://openalex.org/A5004287909)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 U^3‑xi 框架，通过对每帧声学特征估计不确定性并赋予自适应权重，从而改进说话人识别。

**💡 创新点**

创新点包括：① 随机方差损失（SVL）实现说话人级不确定性监督；② 将不确定性嵌入 softmax 的尺度，实现全局级不确定性监督；③ 用 Transformer 加多视角自注意力重构不确定性估计模块，提升时间依赖建模。

**🔧 技术方法**

使用 Gaussian 后验推理作为聚合模块、AAM‑Softmax、Transformer Encoder + Multi‑View Self‑Attention、基于不确定性的余弦评分以及 SVL 等技术。

**📊 数据集**

训练基于 VoxCeleb2，评估在 VoxCeleb1（普通、轻度、极端）以及跨域数据 SITW、CNCeleb，训练时使用 MUSAN 噪声和 RIR 语音混响增强。

**📈 对比分析**

与 ECAPA‑TDNN、xi‑vector 等基线对比，U^3‑xi 在 VoxCeleb1 上 EER 降低 21.1%、minDCF 降低 15.57%；在跨域数据亦实现显著提升，整体相对性能提升约 10–20%。

**⚠️ 局限性**

局限性：在 minDCF 评估上尤其跨域环境下表现不稳定，原因是不确定性估计在陌生声学条件下不够准确，导致评分曲线失稳。

---

## 176. zkFinGPT: Zero-Knowledge Proofs for Financial Generative Pre-trained Transformers

**arXiv ID:** 2601.15716 | [PDF](https://arxiv.org/pdf/2601.15716v1)

**作者:** Xiao-Yang Liu `[一作]` (Columbia University), Weiqin Tong `[通讯]` (Shanghai University)

**通讯引用:** 980 | [OpenAlex ID](https://openalex.org/A5070216219)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了zkFinGPT方案，用零知识证明验证金融大型语言模型的推理过程，且不泄露模型权重或输入数据。

**💡 创新点**

创新点在于将多线性扩展、sumcheck协议、KZG多项式承诺和区块链不可变存储结合，形成既可验证又可保护隐私的可公开审计框架，并将其应用于版权诉讼、考试结果可信度与交易策略保护三大金融场景。

**🔧 技术方法**

主要技术包括零知识证明（sumcheck + Schwartz–Zippel + Fiat–Shamir转化）、多线性扩展（MLE）、KZG多项式承诺、16位量化、CUDA GPU并行计算以及区块链存储。

**📊 数据集**

使用的模型包括LLama2/3、GPT2等；数据集包括纽约时报文章、哥伦比亚商学院考试题以及FinRL竞赛的金融交易数据。

**📈 对比分析**

实验结果显示，证明时间随模型层数线性增长（32层7B模型约620 s，80层70B模型约1578 s），验证时间随层数的平方根增长（均低于5 s）；提交文件大小与模型规模线性相关。与传统无ZKP推理相比，证明时间为数百倍，验证时间仍保持低开销。

**⚠️ 局限性**

主要局限是计算开销巨大，证明和提交时间过长，证明文件庞大；量化精度有限，需高性能GPU与服务器，限制了在实时或资源受限环境中的应用。

---

## 177. Even GPT-5.2 Can't Count to Five: The Case for Zero-Error Horizons in Trustworthy LLMs

**arXiv ID:** 2601.15714 | [PDF](https://arxiv.org/pdf/2601.15714v1)

**作者:** Ryoma Sato `[一作]` `[通讯]` (National Institute of Informatics), Ryoma Sato (National Institute of Informatics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了Zero-Error Horizon（ZEH）度量，用来衡量LLM在简单任务中能无错误解决的最大规模，并通过ZEH限制器展示模型的具体错误实例。

**💡 创新点**

创新点在于：① 用ZEH定义“最大无错误尺寸”，弥补传统准确率在安全评估中的不足；② 通过ZEH发现即使是高性能模型也会在极小输入上犯错；③ 证明ZEH能捕捉模型从记忆到算法推理的演化；④ 提出多种加速技术（教师强制、批量跨尺寸、提示缓存、树结构共享FlashTree）显著降低ZEH评估成本。

**🔧 技术方法**

技术手段包括：全面枚举测试实例计算ZEH；教师强制并行验证与自回归退回；跨尺寸批处理与预填提示缓存；基于树结构的FlashTree实现高效注意力；对GPT‑5.2和Qwen2.5系列大模型进行实验。

**📊 数据集**

使用的数据集主要是可枚举的基准任务：乘法（1–99×1–99）、二进制奇偶性、括号平衡、图着色；此外利用C4训练语料统计检验记忆与算法的关系。

**📈 对比分析**

与传统准确率对比，ZEH随模型规模从0.5B到72B提升从0到42，准确率提升幅度更大但易受范围选择影响。评估速度方面，FlashTree相较于教师强制提升2–3倍，整体从自回归评估可提升10倍；在GPT‑5.2中，ZEH分别为126（乘法）、4（奇偶性）、10（括号）、4（图着色）。

**⚠️ 局限性**

局限性：① 需要大量计算资源，易受单个错误影响导致ZEH敏感；② 对使用外部工具的情景ZEH难以评估；③ 仅适用于可枚举、单一答案任务，对更复杂或开放式任务需进一步研究；④ ZEH对提示和上下文变化不稳定，需多提示评估以获得稳健值。

---

## 178. AgentSM: Semantic Memory for Agentic Text-to-SQL

**arXiv ID:** 2601.15709 | [PDF](https://arxiv.org/pdf/2601.15709v1)

**作者:** Asim Biswal `[一作]` (University of California, Berkeley), Tim Kraska `[通讯]` (Amazon Web Services)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种 Agent Semantic Memory 框架，用于在 Text-to-SQL 任务中重用先前查询的结构化执行轨迹；

**💡 创新点**

创新点在于：①构建可解释的结构化语义记忆，将探索步骤、执行步骤、验证步骤拆分并标记；②利用合成问题生成补充轨迹；③通过自动合成工具减少多步重复调用，提升效率与稳定性；

**🔧 技术方法**

技术手段包括：大语言模型 (Claude 3-7 / Claude 4 Sonnet)、ReAct 交互式代理、工具集 (SQL执行、向量检索、文件读取)、轨迹语义化存储（Markdown/JSON）、支持阈值合成工具、FAISS 向量搜索、MiniLM-L6-v2 表征；

**📊 数据集**

使用的主要数据集为 Spider 2.0 Lite（547 题），并在此基准上进行实验；

**📈 对比分析**

与基线 SpiderAgent 以及 CodingAgent 进行对比，平均执行准确率提升 14.1%，整体准确率达 44.8%；在步数、token 数量和延迟方面平均降低 25%~35%；

**⚠️ 局限性**

局限性主要包括：①对 schema-linking 的依赖仍高，复杂嵌套模式下仍有 44% 的错误；②合成轨迹生成成本高且需手工调参；③在极度多样化的数据库域中，记忆迁移效果不明显；④当前不支持跨数据库迁移。

---

## 179. Improving Methodologies for Agentic Evaluations Across Domains: Leakage of Sensitive Information, Fraud and Cybersecurity Threats

**arXiv ID:** 2601.15679 | [PDF](https://arxiv.org/pdf/2601.15679v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 180. Persona Switch: Mixing Distinct Perspectives in Decoding Time

**arXiv ID:** 2601.15708 | [PDF](https://arxiv.org/pdf/2601.15708v1)

**作者:** Junseok Kim `[一作]` (Seoul National University), Kyomin Jung `[通讯]` (Seoul National University)

**通讯引用:** 3524 | [OpenAlex ID](https://openalex.org/A5077832834)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现 Persona Switch 动态解码方法，在每一步生成时比较零-shot 与角色扮演（role‑play）提示生成的答案，挑选置信度更高的输出构成最终答案。

**💡 创新点**

创新点在于：①利用 logit gap 作为置信度度量，实时决定每一步应采用哪种提示策略；②将两种提示的优点融合在同一推理过程中；③在步进式选择而非整体序列选择时实现更好性能。

**🔧 技术方法**

核心技术包括：大语言模型推理、logit gap 置信度计算、单步 greedy 解码、角色扮演提示插入、停用词/数值过滤、答案触发提示。

**📊 数据集**

实验数据集：GSM8K、AQuA‑RAT、CSQA、Last Letter Concatenation、BIG‑bench Tracking Shuffled‑Objects，使用 LLaMA‑3.2‑3B‑Instruct、LLaMA‑3.1‑8B‑Instruct 与 Gemma‑2‑2B‑it。

**📈 对比分析**

与五大基线（零-shot greedy、top‑p、top‑k、multinomial、role‑play）以及两种 Persona Switch 变体（低 gap 选择、随机选择）对比，平均准确率提升约 5.13%（最高 4.42%），步进级别选择表现最好，停用词过滤略带提升。

**⚠️ 局限性**

局限性：仅在语义不变（Persona‑agnostic）且有明确答案的推理任务上验证；固定角色扮演提示可能不适用于开放式或主观任务；在非推理或跨域任务中的效果仍待探究。

---

## 181. Agentic Uncertainty Quantification

**arXiv ID:** 2601.15703 | [PDF](https://arxiv.org/pdf/2601.15703v1)

**作者:** Jiaxin Zhang `[一作]` (Salesforce AI Research), Chien-Sheng Wu `[通讯]` (Salesforce AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种双过程Agentic UQ框架，将不确定性转化为前向传播的软约束和后向校正的主动反射机制，以提升长时序代理的可靠性。

**💡 创新点**

创新点在于将不确定性管理拆分为Uncertainty‑Aware Memory（前向传播）和Uncertainty‑Aware Reflection（后向校正），并通过可视化置信度阈值实现动态计算预算与双向控制。

**🔧 技术方法**

使用LLM的自我置信度回馈与自然语言解释、最佳‑N采样、软注意力约束、上下文检索扩展等技术。

**📊 数据集**

在ALFWorld、WebShop和DeepResearch Bench三大基准上进行评估。

**📈 对比分析**

与ReAct、Reflexion、Self‑Reflection、CoT‑SC等基线相比，Dual‑Process在成功率、Trajectory‑ECE、Brier Score、AUROC等指标上均有显著提升，成功率提升约10–15%，Trajectory‑ECE低于基线。

**⚠️ 局限性**

局限在于依赖LLM具备表达置信度的能力，对小模型效果有限；System 2的反射会引入额外延迟和计算成本，可能不适合实时低时延场景。

---

## 182. Tight Bounds for Gaussian Mean Estimation under Personalized Differential Privacy

**arXiv ID:** 2601.15682 | [PDF](https://arxiv.org/pdf/2601.15682v1)

**作者:** Wei Dong `[一作]` (Nanyang Technological University), Li Ge `[通讯]` (Nanyang Technological University)

**通讯引用:** 4659 | [OpenAlex ID](https://openalex.org/A5101723847)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

研究了在个性化差分隐私（PDP）下，针对高斯分布的均值估计问题，并给出了在有界和无界两种PDP模型下的最优估计器和对应的下界。

**💡 创新点**

首次提出“扩散”（diffusion）技术实现非均匀采样的隐私放大，并将其与范围估计和加权均值估计相结合，在无界PDP下取得误差与下界匹配的最优结果。

**🔧 技术方法**

结合PDP加权均值估计、扩散/采样放大、SVT和离散化范围估计、加权Bernstein不等式、Le Cam下界、隐私向量缩减等多种技术手段。

**📊 数据集**

以从正态分布 N(μ,σ²) 采样的理论数据为实验对象，主要通过理论分析完成。

**📈 对比分析**

与已有基于有界分布的PDP均值估计和标准DP均值估计对比，证明两种PDP设置下误差达到下界至对数因子，误差随样本量、σ、隐私预算等呈可达最优比例。

**⚠️ 局限性**

仅针对单变量高斯分布，缺乏对偏离正态或高维情形的理论支持；实现过程中需要复杂的离散化与采样步骤，对极端隐私预算分布的鲁棒性有限。

---

## 183. Bridging the Perception Gap: A Lightweight Coarse-to-Fine Architecture for Edge Audio Systems

**arXiv ID:** 2601.15676 | [PDF](https://arxiv.org/pdf/2601.15676v1)

**作者:** Hengfan Zhang `[一作]` (Duke University), Yiran Chen `[通讯]` (Duke University)

**通讯引用:** 25428 | [OpenAlex ID](https://openalex.org/A5058073627)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CoFi-Agent，一种在边缘先行粗略感知后，只有在不确定时才触发云端细化、工具增强推理的音频语言模型架构。

**💡 创新点**

创新点在于将局部低成本推理与云端高精度细化通过置信门控结合，并只在需要时使用时序重听和本地ASR，既保持原始音频本地化，又实现高效精确的多步音频推理。

**🔧 技术方法**

采用了Qwen2‑Audio‑7B作为边缘LLM、prompt‑based置信门控、时段提议器、本地Whisper ASR、云端GPT‑4o推理等技术，形成了粗细分层与工具调用的动态推理流程。

**📊 数据集**

使用了MMAR基准数据集（1000条音频-问题-多选答案）进行评估。

**📈 对比分析**

与单通道基线、混合推理、全量工具调用等方法对比，CoFi‑Agent（Adaptive+ASR）在MMAR上准确率提升至53.60%，平均延迟为9.62秒，明显优于全调用工具（51.70%/11.06s）且在准确-延迟折衷上更具优势。

**⚠️ 局限性**

局限性包括：在极低SNR下ASR输出失真导致误判；长录音中短时关键信息被段落提议器遗漏；以及对需要外部知识的查询仍无法完全解决。

---

## 184. EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning

**arXiv ID:** 2601.15668 | [PDF](https://arxiv.org/pdf/2601.15668v1)

**作者:** Dingdong Wang `[一作]` (Chinese University of Hong Kong), Helen Meng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 9305 | [OpenAlex ID](https://openalex.org/A5019458385)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出EmotionThinker框架，将语音情感识别从单纯分类转为可解释的推理过程，并通过强化学习进一步提升推理质量。

**💡 创新点**

创新点包括①构建细粒度链式推理数据集EmotionCoT‑35K；②提出GRPO‑PTR奖励机制，逐步加入推理奖励并使用可信度权重；③在SpeechLLM中强化prosody感知的基础模型。

**🔧 技术方法**

使用技术包括SpeechLLM Qwen2.5‑Omni、音频编码器＋音频适配器、链式推理数据增强、强化学习（GRPO‑PTR）、奖励模型以及多模态提示。

**📊 数据集**

使用数据集EmotionCoT‑35K（35k条音频+推理），IEMOCAP、MELD、Expresso、MEAD、EARS等语料进行训练；评测用IEMOCAP、MELD、RAVDESS、SAVEE四个基准。

**📈 对比分析**

与13个通用SpeechLLM及3个情感专用模型对比，EmotionThinker在情感准确率达68.9%（比第二名高约3%），推理质量得分3.98/5，人工评估也显示领先，性能显著优越。

**⚠️ 局限性**

局限性包括：奖励模型依赖人工标注，难以大规模扩展；对声学采集条件敏感；目前仅验证于英美语音，跨语言和跨领域的泛化能力仍待验证。

---

## 185. Reflective Motion and a Physical Canvas: Exploring Embodied Journaling in Virtual Reality

**arXiv ID:** 2601.15656 | [PDF](https://arxiv.org/pdf/2601.15656v1)

**作者:** Michael Yin `[一作]` (University of British Columbia), Nadine Wagener `[通讯]` (OFFIS Institute for Informatics)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在VR环境中开发并评估了“embodied journaling”，通过身体动作与语音记录负面经历，并与传统文字日志进行对比，探讨其对情绪表达、回忆与反思的影响

**💡 创新点**

提出将身体运动与口语视为日志媒介的“embodied journaling”概念，并在最小化环境干扰的VR中实现无导引的自发表达与后续回顾，突破传统文字日志的语言过滤限制

**🔧 技术方法**

使用Unity+Meta Quest 3平台，结合VRIK逆向动力学、Mediapipe姿势追踪、WHAM重建算法进行运动捕捉与重现，并采集音频做语音记录

**📊 数据集**

20名参与者的VR运动/音频记录、文字日志内容以及PANAS、R2T2、MEQ‑SF等标准问卷结果构成实验数据集

**📈 对比分析**

采用within‑subject对比设计，定量发现写作在创作阶段反思更高、VR在回顾阶段反思更高，情绪变化无显著差异，记忆评估亦未见显著差异；定性结果显示VR提供更原始情感表达与后期情绪洞察

**⚠️ 局限性**

限制包括样本规模小、受试者多为年轻且自我反思倾向高、仅使用单一已解决负面经历、实验仅两次短期、VR空间受限且缺少导引导致使用差异，未评估长期效果

---

## 186. An Empirical Study on Ensemble-Based Transfer Learning Bayesian Optimisation with Mixed Variable Types

**arXiv ID:** 2601.15640 | [PDF](https://arxiv.org/pdf/2601.15640v1)

**作者:** Natasha Trinkle `[一作]` (RMIT University), Jeffrey Chan `[通讯]` (RMIT University)

**通讯引用:** 4403 | [OpenAlex ID](https://openalex.org/A5071422010)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过大规模实验评估了多种基于集成模型的迁移学习贝叶斯优化（BO）方法，并提出了新的加权策略和实时迁移学习基准，进一步探讨了warm‑start初始化和负迁移处理机制的效果。

**💡 创新点**

创新点包括①提出基于正则化回归（Lasso/Ridge）且约束权重为正的加权方案；②设计了自动切换标准BO与迁移学习BO的负迁移检测机制；③开发了三种新的实时迁移学习基准；④利用Gower距离与聚类对历史数据进行分析，尝试预测迁移学习效果。

**🔧 技术方法**

技术手段包括：贝叶斯优化与高斯过程集成、加权策略（正则化回归、RGPE、TSTR、WAC）、正权重约束、负迁移检测与模式切换、归一化简单风险评估、平均排名图以及Gower距离加层次聚类分析。

**📊 数据集**

使用了9个基准数据集，涵盖grid、surrogate、simulation、real‑time time series benchmark等，主要包括OpenML‑100/CC18、随机森林、lassobench、cartpole等，以及自定义的LQR模拟控制任务。

**📈 对比分析**

通过归一化风险曲线和平均排名图进行方法对比，实验结果显示warm‑start初始化和正权重加权策略在大多数基准上优于随机初始化和无约束策略；负迁移处理机制对整体性能提升有限；整体而言，正权重约束的正则化回归策略取得最佳表现。

**⚠️ 局限性**

主要局限包括：缺乏对迁移收益的理论保证；对新任务的泛化性未知；负迁移检测机制效果不够显著；历史数据聚类分析预测不稳定；实验结果受种子和任务差异影响，难以直接推广至单一目标任务。

---

## 187. A Class of Subadditive Information Measures and their Applications

**arXiv ID:** 2601.15639 | [PDF](https://arxiv.org/pdf/2601.15639v1)

**作者:** Hamidreza Abin `[一作]` (Chinese University of Hong Kong), Mohammad Mahdi Mojahedian `[通讯]` (Sharif University of Technology)

**通讯引用:** 250 | [OpenAlex ID](https://openalex.org/A5040097419)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了一类由增函数G作用于f-散度得到的(G,f)-散度及其对应的(G,f)-信息量，并研究了其子加性性质及相关的结构性定理。

**💡 创新点**

创新点在于将广义散度与信息量统一框架，给出子加性等价条件、对任意有限字母表可归约到二元字母表的检验原则，以及对常见G(x)（x、log(1+x)、-log(1‑x)）提供了可计算的充分条件。

**🔧 技术方法**

使用的技术包括：f-散度与Csiszár互信息的推广、数据处理不等式、Danskin定理、凸/凹函数条件、二元归约、以及在应用中使用的Fano不等式和Sphere‑Packing 变形。

**📊 数据集**

该工作主要为理论分析，并未使用具体实验数据集；所有结果均基于离散概率分布的理论推导。

**📈 对比分析**

通过理论推导得到的子加性(G,f)-信息量可直接给出有限块长编码的逆定理、二项假设检验的单字母界以及误差指数的Sphere‑Packing 上界。与传统KL散度或Rényi散度相比，在适用范围更广且能涵盖更多常用散度。

**⚠️ 局限性**

局限性：仅针对有限字母表；对G(x)=log(1+x)、-log(1‑x) 的充分条件虽易检验，但完整的必要与充分条件仍未知；在实际编码或检测中的性能评估需结合具体散度函数，尚未给出数值实验验证。

---

## 188. Community-Size Biases in Statistical Inference of Communities in Temporal Networks

**arXiv ID:** 2601.15635 | [PDF](https://arxiv.org/pdf/2601.15635v1)

**作者:** Theodore Y. Faust `[一作]` (Southern Methodist University), Mason A. Porter `[通讯]` (University of California)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究并改进时间演化网络中社区检测的统计推断方法，提出层级可交换计数分裂（LECS）先验，以减轻社区规模偏差。

**💡 创新点**

创新点在于引入基于层间可交换性的计数分裂先验，理论证明其社区规模分布更不局部化，并在合成网络实验中显著提升检测准确率。

**🔧 技术方法**

使用统计推断框架下的多层块模型、Gibbs+多层交换采样、IPR定量指标、NMI评价以及相关理论推导。

**📊 数据集**

主要使用人工生成的 50/100 节点、5 层的合成时间网络，设置不同社区大小和连接概率进行实验。

**📈 对比分析**

与均匀先验、Yang et al.、Bazzi et al. 等基线方法对比，利用 IPR 评估社区规模分布，利用 NMI 测量检测精度；LECS 方法在大/小社区情形下 NMI 明显提升，尤其在社区尺寸偏大时优于其他方法。

**⚠️ 局限性**

局限在于仅在小规模合成网络验证，未在真实时变网络上评估；LECS 仅考虑上一层依赖，未扩展多层历史；对超参数敏感且采样收敛速度仍需进一步改进。

---

## 189. From Passive Metric to Active Signal: The Evolving Role of Uncertainty Quantification in Large Language Models

**arXiv ID:** 2601.15690 | [PDF](https://arxiv.org/pdf/2601.15690v1)

**作者:** Jiaxin Zhang `[一作]` (Salesforce AI Research), Chien-Sheng Wu `[通讯]` (Salesforce AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本综述通过梳理相关研究，阐述了LLM中不确定性从被动诊断指标向主动实时控制信号的演进，并在高级推理、自治代理和强化学习三个前沿领域构建了统一的技术分类与设计模式；

**💡 创新点**

创新点在于将不确定性视作系统内部动态控制器，系统化整理了其在多步推理路径选择、推理过程调节、工具使用决策、奖励模型鲁棒化及自我强化中的应用，并提出了理论驱动的框架（贝叶斯方法、可合规预测等），为未来可信AI提供了新的范式；

**🔧 技术方法**

主要技术包括贝叶斯推理、可合规预测、熵/信息增益、置信度加权投票、前向/后向不确定性传播、工具使用阈值策略、奖励模型不确定性建模、熵最小化与信息熵奖励等；

**📊 数据集**

综述未使用原始数据集，而是综合引用了现有研究中使用的公开基准，如UBench、LM‑Polygraph、常用推理与对话数据集（OpenAI evals、ARC、HellaSwag 等）以及强化学习与奖励模型评测集；

**📈 对比分析**

本文通过对比表格和文献综述对各方法在不同任务中的效果进行概括，指出了在推理路径加权、工具调用效率、奖励模型鲁棒性与自我提升方面的性能提升，但并未给出新的实验结果；

**⚠️ 局限性**

局限性包括：①仅聚焦不确定性在高级系统中的功能角色，未对所有估计技术做细致评估；②由于领域快速演进，综述可能缺少最新工作；③缺乏统一的实证对比与大规模评测，主要依赖引用的原始论文性能指标。

---

## 190. Beyond Hard Writes and Rigid Preservation: Soft Recursive Least-Squares for Lifelong LLM Editing

**arXiv ID:** 2601.15686 | [PDF](https://arxiv.org/pdf/2601.15686v1)

**作者:** Xinyu Wang `[一作]` (McGill University), Xiao-Wen Chang `[通讯]` (McGill University)

**通讯引用:** 1789 | [OpenAlex ID](https://openalex.org/A5062087251)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种递归最小二乘（RLS）编辑框架RLSEdit，用于在长序列编辑流中高效更新大型语言模型的参数，兼顾编辑成功和原模型能力保持；

**💡 创新点**

创新点在于将编辑与保持统一为软约束的二次优化，通过引入对预训练权重和锚映射的偏差正则，插值硬写与硬保持极限，并利用Woodbury恒等式实现每步常数时间递推；

**🔧 技术方法**

采用关键-值对的线性映射近似、递归最小二乘更新、Woodbury逆矩阵递推、正则化参数λ、μ调控偏差、以及Cholesky分解实现数值稳定；

**📊 数据集**

实验数据集包括CounterFact（事实编辑评估）、GLUE（SST、MMLU、MRPC、CoLa、RTE）、MMLU、GSM8K、HumanEval、MBPP等通用与推理/编码任务；

**📈 对比分析**

与AlphaEdit、MEMIT、ROME以及全量微调等基线对比，RLSEdit在10K连续编辑后在编辑成功率、早期编辑保持、通用能力保留等指标均表现更好，且更新耗时比AlphaEdit快1.4-1.8倍；

**⚠️ 局限性**

局限性包括需手动调节正则参数λ、μ，锚映射的选择与质量影响稳定性；在极端编辑密度或多样性场景下仍可能出现知识冲突或性能下降；

---

## 191. Connect the Dots: Knowledge Graph-Guided Crawler Attack on Retrieval-Augmented Generation Systems

**arXiv ID:** 2601.15678 | [PDF](https://arxiv.org/pdf/2601.15678v1)

**作者:** Mengyu Yao `[一作]` (Peking University), Ding Li `[通讯]` (Peking University)

**通讯引用:** 170885 | [OpenAlex ID](https://openalex.org/A5100449520)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一个基于知识图的RAG爬虫攻击框架，用全局规划实现对检索增强生成系统的隐蔽数据提取。

**💡 创新点**

将数据提取建模为自适应随机覆盖问题，利用知识图估计条件边际增益并实现近似贪婪策略，同时克服可观测性、搜索空间和现实约束的挑战。

**🔧 技术方法**

自适应随机覆盖理论、知识图构建与增量更新、UCB+图结构探索评分、LLM驱动的查询生成、历史去重与反馈回路。

**📊 数据集**

TREC‑COVID、SciDocs、NFCorpus、Healthcare‑Magic‑100k 四个科学/医疗语料集。

**📈 对比分析**

与 RAGThief 与 IKEA 两个基线对比，在 1000 次查询预算下平均覆盖率 66.8%（最高 84.4%），比基线提升约 20% 至 56%；语义保真度与重建指标同样领先，并在多种检索器、生成器、查询重写和多查询检索场景均保持高效。

**⚠️ 局限性**

仅评估公开语料，缺乏对多模态/实时交互场景的验证；依赖 LLM 提取的知识图质量；对极端安全防护（如严格过滤、检测交互序列）仍有潜在风险；未考察对模型内部状态可观测的攻击变体。

---

## 192. Dualformer: Time-Frequency Dual Domain Learning for Long-term Time Series Forecasting

**arXiv ID:** 2601.15669 | [PDF](https://arxiv.org/pdf/2601.15669v1)

**作者:** Jingjing Bai `[一作]` (University of Osaka and RIKEN AIP), Yoshinobu Kawahara `[通讯]` (University of Osaka and RIKEN AIP)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Dualformer 框架，采用时间域与频域双分支同时学习

**💡 创新点**

创新层级频率采样 (HFS) 与周期性自适应加权，解决 Transformer 的低通滤波问题

**🔧 技术方法**

使用 FFT/IFFT、注意力机制、频率采样模块和周期能量比权重的组合

**📊 数据集**

在八个公开数据集（Electricity、Solar、Traffic、Weather、ETTh1/2、ETTm1/2）上验证

**📈 对比分析**

与 TimeMixer、PDF、TimesNet、FEDformer、FiLM、PatchTST、iTransformer、FreTS、DLinear 等基线比较，平均多项任务上 MSE/MAE 取得第一或第二名，整体性能显著优于现有方法

**⚠️ 局限性**

在强周期性数据（如 Traffic）上略逊一筹，且对极端非平稳、多变量或不规则采样场景的适应性仍有提升空间

---

## 193. Impression Zombies: Characteristics Analysis and Classification of New Harmful Accounts on Social Media

**arXiv ID:** 2601.15666 | [PDF](https://arxiv.org/pdf/2601.15666v1)

**作者:** Uehara Keito `[一作]`, Taichi Murayama `[通讯]` (Yokohama National University)

**通讯引用:** 199 | [OpenAlex ID](https://openalex.org/A5008840028)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对“印象僵尸”账号进行量化特征分析，并构建基于父子回复语义不一致的分类模型

**💡 创新点**

首次系统地量化刻画“印象僵尸”的行为特征，并提出利用回复与父帖语义不匹配的上下文差异来识别其新颖方法

**🔧 技术方法**

使用 SentenceTransformers 进行语义嵌入并微调，随后用多层感知机（MLP）分类器，并与逻辑回归、BERT 以及 GPT‑4.1 进行对比

**📊 数据集**

数据集包括 9,909 条回复与 5,497 个账号的账户数据集、9,909 条父子回复对的分类数据集，以及 250,000 条非僵尸父子对的预训练数据集

**📈 对比分析**

与逻辑回归、BERT 及 GPT‑4.1 的零/少量样本对比，实验显示微调后的 SentenceTransformers+MLP 在准确率、召回率等指标上均超过 90%，总体准确率达 92%

**⚠️ 局限性**

局限性包括：样本量有限、微调数据时间点与实际分类数据相距四年以上、模型仅基于文本无法处理图像等多模态回复

---

## 194. Event-VStream: Event-Driven Real-Time Understanding for Long Video Streams

**arXiv ID:** 2601.15655 | [PDF](https://arxiv.org/pdf/2601.15655v1)

**作者:** Zhenghui Guo `[一作]` (University of Houston), Chengming Zhang `[通讯]` (University of Houston)

**通讯引用:** 681 | [OpenAlex ID](https://openalex.org/A5100691052)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于事件的实时视频理解框架Event-VStream，能够在视频流中检测语义事件边界，动态聚合事件特征，使用事件级记忆并仅在事件发生时触发文本生成，避免冗余与信息遗忘。

**💡 创新点**

核心创新点在于将连续视频划分为离散、语义连贯的事件，通过融合运动、语义漂移与预测误差的边界检测器实现精确事件切分；采用轻量级事件级记忆合并冗余事件；以及事件驱动的解码策略，显著降低重复输出并保持语义连贯。

**🔧 技术方法**

技术包括：运动量与语义相似度的EMA聚合、三项边界评分（语义漂移、运动、预测误差）、自适应阈值与阈值抖动、事件级特征池化、记忆库合并策略、事件触发解码与时间间隔抑制。

**📊 数据集**

使用了OVOBench‑Realtime（实时视频问答/描述基准）和基于Ego4D的长时序（2小时）评测集，评估模型在多小时无剪辑视频上的持续表现。

**📈 对比分析**

与VideoLLM‑Online‑8B、StreamingVLM、Flash‑VStream‑7B等基线对比，Event‑VStream在OVOBench‑Realtime上提升了10.4分（28.15vs17.73），在2小时Ego4D流上保持约70% GPT‑5赢率并达到最高88.3%，同时每词生成延迟低于0.1秒，显著优于其他方法的内存泄露或重复生成问题。

**⚠️ 局限性**

局限性包括：仍依赖预训练的视觉编码器与LLM；事件检测阈值对不同场景的鲁棒性需进一步提升；目前未引入音频/语音等多模态信息；记忆合并阈值在极端场景下可能导致过度合并，丢失细节。

---

## 195. Towards Reliable Medical LLMs: Benchmarking and Enhancing Confidence Estimation of Large Language Models in Medical Consultation

**arXiv ID:** 2601.15645 | [PDF](https://arxiv.org/pdf/2601.15645v1)

**作者:** Zhiyao Ren `[一作]` (Nanyang Technological University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 97310 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了首个多轮医学对话中LLM置信估计基准，并设计了基于检索增强生成和自我解释的MedConf方法来评估诊断置信度。

**💡 创新点**

创新点包括构建包含信息充分性梯度的多轮基准以及提出基于证据自评的MedConf，融合检索增强生成、症状关系分析与加权置信集成。

**🔧 技术方法**

技术上采用检索增强生成（RAG）、自我解释（self‑verbalization）、结构化症状谱构建、基于证据的关系权重和置信集成，并与27种token、consistency、self‑verbalization方法进行对比。

**📊 数据集**

使用了DDXPlus、MediTOD和MedQA三大医学对话/报告数据集。

**📈 对比分析**

通过在三数据集与两大模型（Llama‑3.1、GPT‑4.1）下对比27种基准方法，MedConf在Pearson、AUROC、AUPRC等指标上均居首，相关性达到0.94‑0.99，AUROC提升约0.06，并在鲁棒性与交互效率上优于基线。

**⚠️ 局限性**

局限性包括仅针对诊断任务；方法计算量大、延迟高；实验仅在模拟数据集完成，缺乏真实临床验证。

---

## 196. Generative AI-Empowered Semantic Twin Channel Model for ISAC

**arXiv ID:** 2601.15642 | [PDF](https://arxiv.org/pdf/2601.15642v1)

**作者:** Yi Chen `[一作]` (Dalian University of Technology), Chong Han `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 8617 | [OpenAlex ID](https://openalex.org/A5048916368)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了面向环境语义的 ISAC 通道模型 STCM，能够在给定语义条件下生成物理可行的多样化通道实例。

**💡 创新点**

创新点在于将环境语义作为统一抽象与多层次通道结构对应，引入生成式 AI 驱动的语义双生模型，并配合物理约束的混合合成器与分布式语义一致性指标。

**🔧 技术方法**

技术上结合了大语言模型进行语义解析编码、变分自编码器压缩语义向量、生成式对抗网络生成参数、物理驱动的散射中心与簇模型以及基于 Wasserstein 距离的语义一致性评估。

**📊 数据集**

使用了基于 CST 的大规模全波 EM 仿真产生的车辆与 UAV 的多散射中心数据，以及扩展的 3GPP TR 38.901 语义化散射簇数据，构成训练和评估数据集。

**📈 对比分析**

通过目标匹配分数（TMS）与多基站协作识别的 K–S p 值分布与真实参考及传统统计模型比较，STCM 在单观测识别上 94.24% 超过阈值，协作识别 95.03% 的 p 值高于显著性水平，远优于统计基线。

**⚠️ 局限性**

局限在于缺乏规模化的多层次语义标注通道测量数据、对多模态输入/输出的支持有限以及模型对语义漂移的鲁棒性尚需在线适应与持续学习机制。

---

## 197. Agentic AI Governance and Lifecycle Management in Healthcare

**arXiv ID:** 2601.15630 | [PDF](https://arxiv.org/pdf/2601.15630v1)

**作者:** Chandra Prakash `[一作]` (University of Cumberlands), Avneesh Sisodia `[通讯]` (University of Cumberlands)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一套名为 Unified Agent Lifecycle Management (UALM) 的五层治理框架，旨在帮助院方在大规模部署代理式人工智能时，统一身份登记、编排协作、PHI上下文控制、实时守护与生命周期管理，防止代理蔓延与合规风险。

**💡 创新点**

创新点在于将代理治理拆解为五个可操作的控制平面层，并结合成熟度模型与 KPI 指标，提供了从注册到停用的端到端治理蓝图和可审计的执行路径，弥补了现有 AI 治理标准对多代理系统缺乏细化实现的空白。

**🔧 技术方法**

采用了非人类身份 (NHI) 证书、基于策略的访问控制、治理即代码 (GAC)、向量化 PHI 分片、实时监控与 kill‑switch 机制等技术，构建了身份、编排、上下文、合规与生命周期五层架构。

**📊 数据集**

本文并未使用具体医学或 LLM 数据集，而是基于对行业标准、监管文件（如 HIPAA、EU AI Act、NIST AI RMF 等）以及公开文献的快速实践式综述与主题映射，形成治理框架。

**📈 对比分析**

由于缺乏实验平台，作者没有对性能进行量化对比；文中提出了若干 KPI（如所有代理的负责人占比、凭证吊销时效、工具调用合规率等）作为治理效果的指标，声称能在实际部署中提升审计可见性与安全合规性，但尚无实验验证。

**⚠️ 局限性**

主要局限包括：中心化控制带来计算与延迟开销，单点失效风险；对临床争议缺乏自动判定机制，需人工干预；治理层与代理自治的平衡难题，以及对大规模代理网络性能与可扩展性的实际影响尚未评估。

---

## 198. Communication-efficient Federated Graph Classification via Generative Diffusion Modeling

**arXiv ID:** 2601.15722 | [PDF](https://arxiv.org/pdf/2601.15722v1)

**作者:** Xiuling Wang `[一作]` (Hong Kong Baptist University), Jianliang Xu `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 9342 | [OpenAlex ID](https://openalex.org/A5008564713)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一种新的联邦图神经网络框架 CeFGC（和 CeFGC-adv），利用生成扩散模型在三轮通信中完成全局图分类训练，解决了传统 FGNN 的高通信开销和非 IID 数据分布问题。

**💡 创新点**

创新点包括：① 仅需三轮客户端‑服务器交互，极大降低通信开销；② 在扩散模型中加入标签通道，单个模型即可表示所有类别，显著减少模型上传量；③ 对扩散模型进行加密、随机混洗与聚合，提升隐私安全；④ 客户端用服务器分发的生成模型合成多样化图样本，增强本地训练数据，从而提升全局模型泛化能力。

**🔧 技术方法**

核心技术：联邦学习（FedAvg 等聚合），图神经网络（GIN），扩散式图生成模型（EDP‑GNN）与标签通道，噪声条件的 Langevin 动态采样，混合本地与合成图训练，数据加密（同态加密）与模型聚合，通信复杂度与运行时间分析。

**📊 数据集**

使用了五个真实图分类数据集：MUTAG、ENZYMES、PROTEINS、IMDB‑BINARY（IMDB‑B）和 IMDB‑MULTI（IMDB‑M），分别来自分子、蛋白质和社交网络领域。

**📈 对比分析**

与多种基线（FedAvg、FedProx、GCFL+、FedStar、FedDC、MOON 以及一轮式 FL 模型）在单数据集、跨数据集、跨域三种设置下进行比较。CeFGC 与 CeFGC‑adv 在 AUC/准确率上均显著优于所有基线，最高提升可达 23% 以上；通信轮数从传统的数百轮降至 3 轮，通信量减少 3‑100 倍；在线训练时间减少至基线的 2% 左右（约 98% 降低）。

**⚠️ 局限性**

局限性：目前扩散模型仅生成图结构，无法为节点分类提供标签，因而无法直接支持节点分类任务；GAN 方案在实验中表现不及扩散模型；对恶意客户端或投毒攻击的防护仅通过少量加密与混洗，未做完整安全证明；扩散模型训练成本高，尤其在大规模图数据上；在真正动态联邦环境中的可扩展性与鲁棒性仍待进一步验证。

---

## 199. LL-GaussianMap: Zero-shot Low-Light Image Enhancement via 2D Gaussian Splatting Guided Gain Maps

**arXiv ID:** 2601.15766 | [PDF](https://arxiv.org/pdf/2601.15766v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 200. Endowing Molecular Language with Geometry Perception via Modality Compensation for High-Throughput Quantum Hamiltonian Prediction

**arXiv ID:** 2601.15786 | [PDF](https://arxiv.org/pdf/2601.15786v1)

**作者:** Zhenzhong Wang `[一作]` (Xiamen University), Min Jiang `[通讯]` (Xiamen University)

**通讯引用:** 4473 | [OpenAlex ID](https://openalex.org/A5017961841)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种利用SMILES序列预测分子哈密顿量的多模态语言模型MGAHam

**💡 创新点**

通过几何模态补偿和细粒度多模态对齐，使信息不足的SMILES获得几何感知，并采用弱监督微调提升低数据泛化

**🔧 技术方法**

使用SMILES‑BERT、QHNet（GNN）以及跨模态对齐、几何补偿、Mask‑based弱监督训练

**📊 数据集**

在MD17、QH9、QH‑BM和QH9‑1000K四大基准数据集上进行评估

**📈 对比分析**

相较于传统GNN（QHNet、DEQHNet、SE(3)-Transformer、GemNet）和SMILES‑BERT，MGAHam在哈密顿量、能量和波函数相似度上取得近似或更优精度，同时速度提升约100×；在高通量筛选和电解液配方案例中表现优异

**⚠️ 局限性**

仅使用SMILES预测时对分子构型差异敏感，无法处理多构象体系；模型对几何信息的补偿仍需先前的几何–语言对齐训练，限制了在极少几何数据的场景下的直接迁移

---

## 201. Diffusion Model-Based Data Augmentation for Enhanced Neuron Segmentation

**arXiv ID:** 2601.15779 | [PDF](https://arxiv.org/pdf/2601.15779v1)

**作者:** Liuyun Jiang `[一作]` (Chinese Academy of Sciences), Hua Han `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 9586 | [OpenAlex ID](https://openalex.org/A5100676754)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种基于扩散模型的数据增强框架，用于电镜图像中的神经元分割，生成多样化且结构合理的图像‑标签对来扩充训练集。

**💡 创新点**

创新点在于引入分辨率感知的条件扩散模型结合多尺度条件与 EM 分辨率先验，以及基于生物学的掩模重塑模块，显著提升生成质量与分割性能。

**🔧 技术方法**

采用的技术包括 DDPM 扩散生成、3D U‑Net + Mamba 结构、MSC 多尺度条件、RGM 全局建模以及生物学引导的弹性变形与线粒体掩模重构。

**📊 数据集**

实验使用了 AC3 与 AC4 两套小鼠皮层电镜数据集（分辨率 6×6×29 nm³）。

**📈 对比分析**

通过与 Pix2Pix、Med‑DDPM 等基线以及 Superhuman、SwinUNETR、SegMamba 等分割模型对比，在 4%、20% 和 100% 标注比例下，增强方法均显著提升 VI/ARAND，4% 时 ARAND 提升约 32%/30%，生成图像 3D‑FID 亦为最佳。

**⚠️ 局限性**

限制在于仍需大规模 GPU 训练，生成模型对极端稀疏或异常结构的泛化有限，并依赖人工构建的线粒体先验库。

---

## 202. Rethinking Drug-Drug Interaction Modeling as Generalizable Relation Learning

**arXiv ID:** 2601.15771 | [PDF](https://arxiv.org/pdf/2601.15771v1)

**作者:** Dong Xu `[一作]` (Shenzhen University), Junkai Ji `[通讯]` (Shenzhen University)

**通讯引用:** 1548 | [OpenAlex ID](https://openalex.org/A5046906366)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于关系学习的药物-药物相互作用预测框架 GenRel-DDI，改用固定的预训练分子编码器作为语义锚，配合轻量级的可学习关系模块来捕获药物间的相互作用模式。

**💡 创新点**

创新点在于将 DDI 任务从传统的分子中心化（单体编码 + 组合头）转变为关系中心化，即独立于药物身份学习交互表示，解决了实体无关泛化难题，并通过锚-适配器结构与冻结策略显著抑制了在未知药物上的表示漂移。

**🔧 技术方法**

使用预训练分子编码器（ChemBERTa‑3、MolT5 等）做锚，配合跨注意力与池化的关系干预模块，以及多任务学习的交互头；实验中还比较了冻结 vs. 全微调、编码器规模、角色分配等超参。

**📊 数据集**

在七个公开基准上验证：MeTDDI、DDInter、DeepDDI、ZhangDDI、ChChDDI、DDInter Severity 等，覆盖二分类、多分类、方向/效应标签以及不同药物覆盖率（S1–S3）和跨数据集零样本迁移。

**📈 对比分析**

与 12+ 传统方法（DeepDDI、MR‑GNN、MolTrans、CIGIN 等）以及最新模型（MathEagle、Taco‑DDI）在最严格的实体无关（S3）和跨数据集设置下对比，均实现了显著提升，尤其在 AUROC、AUPR、ACC 等指标上多次夺冠。

**⚠️ 局限性**

局限性包括：对预训练编码器的依赖（需预先训练大模型）、在部分极端稀疏或高度不平衡标签空间下仍有提升空间、以及关系模块的可解释性和可扩展性尚未充分验证。

---

## 203. Atlas-Assisted Segment Anything Model for Fetal Brain MRI (FeTal-SAM)

**arXiv ID:** 2601.15759 | [PDF](https://arxiv.org/pdf/2601.15759v1)

**作者:** Qi Zeng `[一作]` (Boston Children's Hospital), Davood Karimi `[通讯]` (Boston Children's Hospital)

**通讯引用:** 1392 | [OpenAlex ID](https://openalex.org/A5076562643)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发了 FeTal-SAM，一种基于 Segment Anything Model 的胎儿脑 MRI 自动分割框架，利用多模态 atlas 注册生成的密集提示和边框提示，实现任意解剖结构的分割，且无需针对每个标签集合重新训练。

**💡 创新点**

创新点在于将多年龄匹配的注册 atlas 标签模板作为 dense prompt 注入 SAM 解码器，结合传统的边框提示，实现模型的通用性与可解释性；同时提供一种评估分割是基于真实图像对比度还是空间先验的方法，揭示现有深度学习方法的潜在偏差。

**🔧 技术方法**

技术实现包括：使用 Med‑SAM ViT‑b 作为基础架构（冻结 encoder，微调 decoder），构建 Atlas Prompt Encoder（channel attention + fusion convolution），基于 ANTS 进行多体素注册，采用 2D 切片推断（轴向、冠状、矢状）并通过 STAPLE 融合成 3D 结果，训练损失为 Dice + CE。

**📊 数据集**

使用了两大胎儿 MRI 数据集：dHCP（297 T2w 图像）和 CRL（294 3T Siemens 图像），各自随机选取 20% 作为测试集。

**📈 对比分析**

与 Med‑SAM、Med‑SAM‑FT、nnUNet、Swin‑UNETR 进行对比；在 dHCP 上 FeTal‑SAM 的平均 Dice 为 88.2% (HD95 ≈ 1.04 mm)，在 CRL 上 Dice 80% (HD95 ≈ 1.01 mm)。与 3D SOTA 相比，Dice 仅差 1–3%，HD95 约 1 mm，显示竞争力；但在低对比度的小结构上仍略逊。

**⚠️ 局限性**

局限性包括：对低对比度、细小结构（如海马、杏仁核）分割不稳定，受限于 2D 切片的局部对比信息；多模态 atlas 的注册误差可能导致 over/under‑prompting；缺乏完整的 3D 空间先验，使得某些结构的准确度低于专门训练的 3D 模型。

---

## 204. Tabular Incremental Inference

**arXiv ID:** 2601.15751 | [PDF](https://arxiv.org/pdf/2601.15751v1)

**作者:** Xinda Chen `[一作]` (Fudan University), Bo Yan `[通讯]` (Fudan University)

**通讯引用:** 16573 | [OpenAlex ID](https://openalex.org/A5054719997)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了Tabular Incremental Inference任务，研究如何在推理阶段动态加入新列并提升模型性能。

**💡 创新点**

创新点在于将信息瓶颈理论与大语言模型占位符和增量样本凝聚块相结合，构建可在无监督情况下利用增量列的模型。

**🔧 技术方法**

主要技术包括信息瓶颈优化、LLM占位符编码、TabPFN‑v2 TabAdapter、增量样本凝聚（MSA+IISA）、对比学习与伪标签训练以及MINE互信息估计。

**📊 数据集**

实验使用了八个公开表格数据集：Diabetes、Adult、Dress‑sales、Income、Credit‑g、Higgs、Blastchar、Insurance‑co。

**📈 对比分析**

与传统固定列模型、直接输入模型、LLM辅助模型以及FT‑Transformer/TransTab等基线相比，TabII 在所有数据集平均获得 97% 的最佳性能，且在增量列下显著优于竞争方法。

**⚠️ 局限性**

主要限制包括对占位符长度的敏感性、对持续不断出现的列需要手动设定阈值，以及对极高缺失率的处理仍不如传统填充策略。

---

## 205. DualShield: Safe Model Predictive Diffusion via Reachability Analysis for Interactive Autonomous Driving

**arXiv ID:** 2601.15729 | [PDF](https://arxiv.org/pdf/2601.15729v1)

**作者:** Rui Yang `[一作]` (Hong Kong University of Science and Technology), Jun Ma `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 19281 | [OpenAlex ID](https://openalex.org/A5100357282)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出一种DualShield框架，将扩散模型与Hamilton‑Jacobi（HJ）可达性价值函数双重利用，实现安全导向的多模态轨迹规划与实时安全屏障；

**💡 创新点**

创新点在于将HJ价值函数同时用于扩散过程的主动引导和执行时的被动安全屏蔽，形成统一的CBVF-QP安全盾牌；

**🔧 技术方法**

技术包括模型驱动的扩散规划、HJ可达性分析、控制屏障价值函数（CBVF）、QP安全屏蔽、递归滚动时域规划；

**📊 数据集**

使用自制仿真数据：在1:4比例的交叉口环境下，包含两辆HV、20个静态障碍物的未保护U‑turn任务，共100次随机试验；

**📈 对比分析**

与MBD、NMPC、DualGuard‑MPPI对比，DualShield在成功率100%、碰撞率0%、任务完成时间最短、控制颤动最低，显示出优越的安全与性能平衡；

**⚠️ 局限性**

局限性在于计算量大，规划时延高，依赖离线预计算的HJ值函数，且对GPU并行采样和值查询效率的提升仍有需求。

---

## 206. A Mobile Application for Flower Recognition System Based on Convolutional Neural Networks

**arXiv ID:** 2601.15810 | [PDF](https://arxiv.org/pdf/2601.15810v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 207. Towards Automated Kernel Generation in the Era of LLMs

**arXiv ID:** 2601.15727 | [PDF](https://arxiv.org/pdf/2601.15727v1)

**作者:** Yang Yu `[一作]` (Beijing Academy of Artificial Intelligence), Yonghua Lin `[通讯]` (Beijing Academy of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述并系统梳理了基于大语言模型与智能代理的 GPU 核心生成与优化技术，整合了相关方法、数据集与基准，阐明了当前研究现状与挑战。

**💡 创新点**

首次提供了统一的分类框架和资源目录（包括数据集、工具链与 GitHub 仓库），并将 LLM 生成与代理驱动的迭代优化结合成整体视角，推动领域向规模化、可复现的自动化发展。

**🔧 技术方法**

采用了大语言模型（Transformer）细调（SFT、RL）、检索增强生成（RAG）、多智能体协作与工具链集成（编译器、Profiler、评测环境）等技术；同时利用强化学习和交互式反馈循环实现自适应优化。

**📊 数据集**

使用了多种公开数据集：The Stack v2、HPC‑Instruct、KernelBook、KernelBench 等结构化数据集；以及大量开源代码库（CUTLASS、FlashAttention、FlagAttention、Triton 等）和知识库（CUDA Guide、PTX ISA、Tuning Guides）。

**📈 对比分析**

通过 ParEval、KernelBench、TritonBench、MultiKernel‑Bench、BackendBench 等基准，对生成的核心在正确性（pass@k）、速度提升（speedup@k）、效率与兼容性等指标进行对比，报告多模型在不同硬件平台上能达到的速度提升与近人类专家的性能水平。

**⚠️ 局限性**

限制主要在于：数据稀缺且分布不均，缺乏完整的优化过程记录；评测体系仍偏向 NVIDIA GPU，缺乏跨平台、跨形状的鲁棒性验证；代理系统往往受限于预设工作流，难以实现真正自适应长周期优化；缺乏可解释性与人机协作机制，难以满足高可靠性生产需求。

---

## 208. Generalized Information Inequalities via Submodularity, and Two Combinatorial Problems

**arXiv ID:** 2601.15723 | [PDF](https://arxiv.org/pdf/2601.15723v1)

**作者:** Gunank Jakhar `[一作]` (International Institute of Information Technology), Vinod M. Prabhakaran `[通讯]` (Tata Institute of Fundamental Research)

**通讯引用:** 2338 | [OpenAlex ID](https://openalex.org/A5045516299)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

利用子模函数的凸性与对称性，提出了一般化的信息不等式，并用其推导出更紧凑的Loomis‑Whitney型投影不等式以及在噪声通道混淆图上的边数上界；

**💡 创新点**

在已有的Madiman–Tetali框架基础上引入凸函数映射，获得了更强的强弱不等式；通过子模函数的结构信息，改进了传统的投影不等式；将Shearer引理推广到更一般的集合族，推导了更广泛的图论边数上界；

**🔧 技术方法**

子模/超模函数的分数划分与覆盖/包装理论；凸/凹函数的Jensen不等式；Han、Shearer和Loomis‑Whitney不等式的子模推广；

**📊 数据集**

无（本文为理论推导，未使用实验数据集）

**📈 对比分析**

相较于Han或传统Shearer不等式，得到的投影不等式在考虑切片结构后更紧；在混淆图中得到的边数上界兼顾了更一般的集合族，覆盖了Sason、Boucheron等先前结果；

**⚠️ 局限性**

局限在于仅适用于满足子模性质的函数；凸函数映射需要先验可控的分数划分；结果多为理论上界，缺乏实际算法实现与实验验证

---

## 209. ExDR: Explanation-driven Dynamic Retrieval Enhancement for Multimodal Fake News Detection

**arXiv ID:** 2601.15820 | [PDF](https://arxiv.org/pdf/2601.15820v1)

**作者:** Guoxuan Ding `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Jiangnan Li `[通讯]` (WeChat AI, Tencent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于解释驱动的动态检索增强生成框架（ExDR），用于多模态假新闻检测。

**💡 创新点**

创新点包括：①利用模型生成的解释来构造多层置信度阈值，实现精准的检索触发；②融合实体、文本、视觉特征的实体增强混合索引；③基于细粒度欺骗标签的对比证据检索，提升检索质量。

**🔧 技术方法**

技术方法涵盖：动态检索增强生成（RAG）与链式推理（CoT）提示、命名实体识别、CLIP视觉编码、FAISS向量检索、以及多模态特征融合与阈值搜索策略。

**📊 数据集**

使用的公开数据集为 AMG（中文多模态假新闻）和 MR^2（中英双语大规模谣言检测），并在两者上分别进行域内与跨域实验。

**📈 对比分析**

与 FL-RAG、FLARE、DRAGIN、wo-RAG、Text@full、Image-Text@full 等基线以及传统 MFND 模型（如 MGCA）比较，ExDR 在检索触发率、检索效率、检测准确率和 F1 分数上均取得显著提升，最高可达 86.1% 的准确率。

**⚠️ 局限性**

局限性：①依赖外部生成解释，解释质量不佳可能影响检索触发；②阈值设置需要在验证集上调优，跨域迁移时表现仍有一定下降；③目前仅验证中文与英文，跨语言泛化尚需进一步探索。

---

## 210. SteerEval: Inference-time Interventions Strengthen Multilingual Generalization in Neural Summarization Metrics

**arXiv ID:** 2601.15809 | [PDF](https://arxiv.org/pdf/2601.15809v1)

**作者:** Silvia Casola `[一作]` (LMU Munich), Barbara Plank `[通讯]` (LMU Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对多语言文本生成（摘要）任务的自动评估指标进行实验，通过在测试时将模型激活向英语枢轴对齐来提升指标与人工评估的相关性。

**💡 创新点**

首次在多语言评估中引入激活调节（activation steering），并证明即使不需要重新训练模型，向高资源语言（英语）对齐也能显著提升多语言神经评估指标的效果。

**🔧 技术方法**

采用向量方向调节（vector steering）和线性映射调节（map steering）两种测试时干预方法；对LLM‑as‑a‑judge（直接提示、GPTScore）和编码器‑based（COMET）评估指标进行实验。

**📊 数据集**

使用 FLORES 多语料并行数据学习跨语言映射；使用来自 XL‑Sum、HeSum 的多语言摘要评估集（阿拉伯语、西班牙语、希伯来语、日语、土耳其语、乌克兰语、约鲁巴语、汉语）并获得 400 条人工评分（连贯性、完整性）进行对比。

**📈 对比分析**

与基线（无调节）相比，激活调节在所有模型、语言和评估维度上均提升了 Pearson 相关系数，部分场景提升超过 100%（例如 COMET 在多种语言上的相关系数提升超过 50%）。GPTScore 在高基线场景下提升有限，Direct Prompting 的提升受基线低时不稳定，但整体提升均为正向。对比实验表明，LLM‑as‑a‑judge 在大多数语言上优于 COMET，且 Llama‑3‑8B 在整体上表现最突出。

**⚠️ 局限性**

限制：样本量与人工标注数量有限，缺乏专门的开发集导致 ρ、σ 参数未能在真实数据上调优；实验仅以英语（或法语）为枢轴，未探究其它高资源语言的最优匹配；在大模型（Aya‑expanse‑32B）上效果并不总是优于小模型，提示模型大小并非唯一决定因素。

---

## 211. A Beacon Based Solution for Autonomous UUVs GNSS-Denied Stealthy Navigation

**arXiv ID:** 2601.15802 | [PDF](https://arxiv.org/pdf/2601.15802v1)

**作者:** Alexandre Albore `[一作]` (DTIS ONERA Universite de Toulouse), Damien Pellier `[通讯]` (LIG Universite de Grenoble Alpes)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一套基于声学信标的自主UUV GNSS‑否认隐蔽导航方案，包含信标部署规划、层次化路径生成与实时重规划。

**💡 创新点**

创新点在于将信标部署的优化（Lloyd算法）与HTN层次规划相结合，形成可视化插件，并实现了闭环监控与自适应重规划，首次针对海下GNSS‑否认环境实现全流程解决。

**🔧 技术方法**

使用了声学定位、Voronoi/Lloyd算法、层次任务网络(HTN)规划（SHOP2/PANDA）、A*搜索、QGIS插件开发与可视化、DVL/INS等传感器融合。

**📊 数据集**

主要使用海底深度图（SHOM bathymetry）作为部署区域数据，并在QGIS中构造的模拟场景中生成信标与UUV轨迹；未使用公开大规模数据集。

**📈 对比分析**

实验以QGIS插件演示为主，仅给出单一示例，无定量对比；但通过示例展示了在信标支持下的定位精度提升和重规划能力，表现优于传统单一声学定位方法。

**⚠️ 局限性**

局限包括未考虑UUV间协同与通信约束、信标信号衰减模型简化、缺乏时间约束的HTN规划器、并未在真实海况下进行验证。

---

## 212. VitalDiagnosis: AI-Driven Ecosystem for 24/7 Vital Monitoring and Chronic Disease Management

**arXiv ID:** 2601.15798 | [PDF](https://arxiv.org/pdf/2601.15798v1)

**作者:** Zhikai Xue `[一作]` (Worcester Polytechnic Institute), Xiaozhong Liu `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 3763 | [OpenAlex ID](https://openalex.org/A5101985030)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了VitalDiagnosis生态系统，利用LLM和可穿戴设备实现慢性疾病的主动、交互式管理。

**💡 创新点**

创新点在于双轨框架（异常实时交互与例行依从性监测）以及统一记忆核心，结合LLM多模态解读、情境化询问与可定制的LoRA记忆。

**🔧 技术方法**

使用LLM模型（4B Memory MiniLLM、1.7B Monitoring MiniLLM、14B Domain LLM），LoRA参数化记忆，规则+模型结合的触发检测，双向沟通协调模块，基于可穿戴传感器的数据流。

**📊 数据集**

主要使用来自可穿戴设备的生命体征时间序列数据和临床注释的模拟案例来训练/微调LoRA；数据集未公开列出。

**📈 对比分析**

文中未给出具体对照实验或量化指标；作者仅说明将与医疗机构进行试点评估，预期可提升自我管理、减少可避免的临床负担。

**⚠️ 局限性**

限制包括缺乏大规模真实数据验证、对LLM可解释性和安全性的讨论不足、隐私合规挑战、以及系统在不同患者群体中的泛化性待验证。

---

## 213. Agentic Confidence Calibration

**arXiv ID:** 2601.15778 | [PDF](https://arxiv.org/pdf/2601.15778v1)

**作者:** Jiaxin Zhang `[一作]` (Salesforce AI Research), Chien-Sheng Wu `[通讯]` (Salesforce AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于过程轨迹的置信度校准框架Holistic Trajectory Calibration，对AI代理的完整推理轨迹进行特征提取与校准；

**💡 创新点**

创新点包括：① 将轨迹级多尺度不确定性特征（跨步动态、微步稳定性、位置指标、结构属性）系统化；② 采用可解释的线性/套索/岭回归校准器，实现样本高效、可迁移和泛化；③ 通过预训练得到的General Agent Calibrator在域外任务上实现最低ECE；

**🔧 技术方法**

技术手段包括：统计特征工程（均值、方差、熵、偏度、差分等）+ 线性/套索/岭回归校准模型；实验平台使用多代理框架与多种LLM（GPT‑4、GPT‑4o、GPT‑OSS‑120B/20B、Deepseek‑v3.1、Qwen3‑235B等）；

**📊 数据集**

使用八个公开基准：SimpleQA、HotpotQA、StrategyQA、MATH500、GPQA、MMLU‑Pro、HLE、GAIA；

**📈 对比分析**

与传统推理式基线（Verbalized Conf、LastStep‑TP、GlobalTrace‑TP等）以及学习式基线（LSTM、Transformer、NN、XGBoost、GP）对比，HTC在ECE、Brier Score和AUROC上显著优于所有基线，特别是在域外GAIA上预训练校准器取得最低ECE；

**⚠️ 局限性**

局限性：特征工程仍为手工设计，对极端或未知任务迁移受限；无法完整覆盖模型内部未知的不确定性；预训练的通用校准器在特定任务仍需微调；尚未深入探讨代理内部结构的可解释性。

---

## 214. Glove2UAV: A Wearable IMU-Based Glove for Intuitive Control of UAV

**arXiv ID:** 2601.15775 | [PDF](https://arxiv.org/pdf/2601.15775v1)

**作者:** Amir Habel `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserukou `[通讯]` (Skolkovo Institute of Science and Technology)

**通讯引用:** 1963 | [OpenAlex ID](https://openalex.org/A5056458774)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并验证了一种可穿戴IMU手套用于UAV的手势控制，并结合触觉反馈提示速度超限。

**💡 创新点**

创新在于轻量化的手套硬件、实时的中值滤波+Madgwick姿态估计、以及集成的触觉警告通道，使操作员在动态飞行中保持安全与可预测性。

**🔧 技术方法**

使用IMU传感器、ESP32微控制器、Wi‑Fi/UDP传输、滑动窗口中值滤波、Madgwick/互补滤波、手势映射算法和ROS通信。

**📊 数据集**

未使用公开数据集，而是在实验室和实地飞行中收集了手套IMU数据与UAV遥测同步的原始数据集。

**📈 对比分析**

通过对比仿真与实飞轨迹、手势与遥测同步曲线，验证了指令映射的实时性与一致性；实验显示指令执行及时、平台响应连贯、触觉警告可促使更平稳飞行。

**⚠️ 局限性**

局限性包括仅在单一操作者与单一UAV平台上验证、缺乏量化的精度评估、未与其他控制接口进行对比，且触觉反馈的效果仍需进一步量化。

---

## 215. NMRGym: A Comprehensive Benchmark for Nuclear Magnetic Resonance Based Molecular Structure Elucidation

**arXiv ID:** 2601.15763 | [PDF](https://arxiv.org/pdf/2601.15763v1)

**作者:** Zheng Fang `[一作]` (Hong Kong University of Science and Technology), Jun Xia `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 8554 | [OpenAlex ID](https://openalex.org/A5068570479)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了NMRGym——一套基于269,999条高质量实验¹H和¹³C NMR谱的最大规模标准化数据集与基准，包含精细的峰-原子注释和功能组、毒性标签，并提供公开的评估排行榜。

**💡 创新点**

创新点在于（1）通过严格的质量控制与统一格式实现真正实验数据的规模化；（2）引入基于Bemis‑Murcko骨架的拆分策略消除数据泄漏；（3）构建覆盖结构解析、功能组预测、毒性预测和谱模拟四大下游任务的统一评估体系。

**🔧 技术方法**

使用的技术包括深度学习模型（Transformer、扩散模型、搜索式解算器）、图神经网络与三维等变网络、预训练与微调、Tanimoto/余弦相似度评估、Hungarian 匹配等。

**📊 数据集**

使用的数据集为NMRGym（269,999分子），并与传统的合成数据集（QM9‑NMR、PubChem‑NMRNet）及公开实验集（NMRShiftDB、NMRBank）对比。

**📈 对比分析**

方法比较采用Top‑K准确率、指纹相似度（Morgan/拓扑扭矩/原子对）、宏微 F1、毒性子集准确率等指标；Transformer与搜索式方法在结构解析中表现最优，NMRMind在无公式约束下Top‑1为15.49%，并在指纹相似度上超越搜索基线；在功能组预测中XGBoost获得最高子集准确率；在谱模拟中NMRNet实现100%覆盖且在¹³C集上Set相似度0.908。

**⚠️ 局限性**

局限性包括：（1）实验数据仍受限于可公开获取的谱片段，缺乏大规模二维谱；（2）模型在低相似度或宏环等复杂结构上性能下降；（3）缺乏峰‑原子显式解释能力，仍为黑盒；（4）合成与实验之间仍存在显著域差，需要更强的适应与迁移技术。

---

## 216. NL4ST: A Natural Language Query Tool for Spatio-Temporal Databases

**arXiv ID:** 2601.15758 | [PDF](https://arxiv.org/pdf/2601.15758v1)

**作者:** Xieyang Wang `[一作]` (Nanjing University of Aeronautics and Astronautics), Raymond Chi-Wing Wong `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 5933 | [OpenAlex ID](https://openalex.org/A5049858061)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了 NL4ST，一种自然语言查询工具，将自然语言查询直接映射到时空数据库的物理查询计划，支持多种时空查询类型并展示结果。

**💡 创新点**

创新点包括：①直接生成物理计划而非 SQL，消除歧义并显式控制执行；②利用领域知识库和语料库进行实体定位与查询类型分类；③基于索引利用和成本模型的物理计划优化。

**🔧 技术方法**

技术包括：spaCy 与细粒度实体提取算法、LSTM 查询类型分类、预定义查询映射规则、R‑Tree 等时空索引、基于采样的成本模型、SECONDO 铝模块集成。

**📊 数据集**

使用四个公开数据集：南京出租车/道路/POI、伦敦 POI/区、柏林火车/POI/河流以及中国水系统数据，涵盖点、线、区域和移动对象。

**📈 对比分析**

与传统 Text‑to‑SQL 方法对比，NL4ST 在四个数据集上实现了平均 1.9 秒响应时间，93% 可翻译率，90% 翻译精度，显著提升了查询执行效率和准确性。

**⚠️ 局限性**

局限性包括：需要大量手工构建知识库与语料；在复杂多维查询或大规模数据时仍可能产生实体歧义；当前仅支持 SECONDO 平台，迁移到其他数据库需重构。

---

## 217. CTL* Model Checking on Infinite Families of Finite-State Labeled Transition Systems (Technical Report)

**arXiv ID:** 2601.15756 | [PDF](https://arxiv.org/pdf/2601.15756v1)

**作者:** Roberto Pettinau `[一作]` (Carl von Ossietzky University of Oldenburg), Christoph Matheja `[通讯]` (DTU Compute)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一套基于超边替换图文法（HRG）的模型检测算法，能够在一次性生成的同一族所有有限状态有向图上对 CTL* 与 LTL（及其子句）进行判定，并给出是否所有、部分或有限/无限多个族成员满足指定 ω‑正则性质的决策。

**💡 创新点**

核心创新在于：①引入“重着色（recoloring）”问题，利用 HRG 的局部结构与有限的上下文信息递归构造能标记满足性质的节点；②定义了以 ω‑正则自动机为基础的 HGV‑等价关系（≡_ϕ），实现对上下文与子图的可判定归约；③得到的算法在保持原文法规则不变的情况下，仅扩展节点颜色，从而自然延伸经典的 CTL* 状态标记算法；④提供了实现与实验，证明对多类真实系统族（IPv4 Zeroconf、树、并行图、Sierpinski 三角、银行抢劫安全树、Dining Cryptographers）均能在秒级完成验证。

**🔧 技术方法**

主要技术包括：
- 超边替换图文法（HRG）与上下文无关图文法的组合；
- Büchi 自动机与 ω‑正则语言的理论基础；
- HGV（带视图的图）与等价关系、同余（congruence）构造；
- 通过构造 “行为图（behaviour graph）” 来进行等价判定；
- 递归推导与分层标记（AddColor、Minimize 等子程序）；
- Haskell 语言实现与 VATA 树自动机库用于最小化。

**📊 数据集**

使用的实验数据集包括：
1. IPv4 Zeroconf 协议的探测器数可变的 Markov 链族；
2. 任意 arity 的树形 LTS 族；
3. 任意大小的 Series‑Parallel 图族；
4. Sierpinski 三角形族；
5. 银行抢劫安全树（SafeLock 变体）族；
6. Dining Cryptographers 协议族；
共 56 组模型检查基准，涵盖多种 CTL* 与 qPCTL 公式。

**📈 对比分析**

与现有文献（如基于 MSO/Courcelle 的判定、Burkart‑Quemener 的单一无限 Kripke 结构方法、Groove 的显式枚举）相比，本工作在决策性与性能上具备明显优势：
- 对于上述 56 组实验，平均运行时间 < 1 s；
- 能同时给出“所有成员满足/不满足”和“存在/不存在满足”的结论；
- 采用 HGV‑等价归约仅需有限等价类，保证算法终止；
- 通过最小化树自动机实现可在几秒内完成语法最小化，进一步降低后续重着色的规模。
性能主要受制于每条规则的最大超边数与外部节点数，随着这些参数增大，时间呈指数级上升。

**⚠️ 局限性**

局限性：
- 对超边/抽象节点数较大的文法，等价类与行为图规模剧增，导致内存与时间爆炸；
- 目前仅支持有限状态族（每个成员是有限 LTS），无法直接处理无限状态系统；
- 对量化属性（如完整 PCTL）支持有限，主要局限于 qPCTL 的可约分子；
- 需要手工或自动化的文法简化/最小化步骤，现有实现尚未针对复杂文法做高级优化；
- 只给出判定结果，未提供直接从重着色结果中提取满足/不满足族成员的具体实例（需额外推导）。

---

## 218. Hallucination Mitigating for Medical Report Generation

**arXiv ID:** 2601.15745 | [PDF](https://arxiv.org/pdf/2601.15745v1)

**作者:** Ruoqing Zhao `[一作]` (Nanjing University of Aeronautics and Astronautics), Piji Li `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 3959 | [OpenAlex ID](https://openalex.org/A5061435467)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

结合MedCLIP检索医学知识并用精炼模块筛选上下文相关信息，再通过细粒度奖励引导LVLM生成更少幻觉、更精准的放射报告。

**💡 创新点**

三大创新：1）知识增强模块引入外部医学事实；2）净化模块确保检索知识与病例语境匹配；3）细粒度疾病与句子层面奖励显著降低幻觉。

**🔧 技术方法**

使用MedCLIP进行知识检索、LLaVA‑1.5‑7B等LVLM与LoRA微调、REINFORCE强化学习细粒度奖励、GPT‑3.5评估句子级别。

**📊 数据集**

采用公开胸部X光报告数据集IU‑Xray和MIMIC‑CXR。

**📈 对比分析**

与多种现有MRG模型（R2Gen、HRGR、CoAtt、PKERRG、CMAS‑RL、CMN、CCR、PPKED、KM、Multicriteria）在BLEU、METEOR、ROUGE‑L等NLG指标和F1、Precision、Recall等临床效能指标上对比，KERM在两套数据上均取得最高或同水平成绩，尤其在CE指标上明显优于先行方法。

**⚠️ 局限性**

依赖知识库质量与更新、净化模块不一定完全匹配细致病史、奖励模型无法覆盖所有临床细节、模型计算成本高、在不同数据域的泛化性待验证。

---

## 219. FAIR-ESI: Feature Adaptive Importance Refinement for Electrophysiological Source Imaging

**arXiv ID:** 2601.15731 | [PDF](https://arxiv.org/pdf/2601.15731v1)

**作者:** Linyong Zou `[一作]` (Anhui University), Zikang Xu `[通讯]` (Anhui Province Key Laboratory of Biomedical Imaging and Intelligent Processing)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于深度学习的电生理源定位框架，能够自适应地从频谱、时域和块级视角对提取的特征重要性进行细粒度调节，从而提高脑源定位的精度。

**💡 创新点**

创新点主要包括：①三视角自适应特征重构策略（FFT频谱加权、时域加权融合、能量关键块自注意力）；②使用神经质量模型（NMM）生成多配置的仿真数据，克服真实配对数据稀缺问题；③将重构特征与原始电极信号通过 MLP+BiGRU 共同学习，提升时空表示能力。

**🔧 技术方法**

核心技术包括：FFT 与逆FFT、温度尺度 softmax、加权求和、Patch 级自注意力（Self‑Attention）、卷积块与反卷积块、双向 GRU、MSE 损失；同时使用神经质量模型（Jansen‑Rit）生成仿真源信号。

**📊 数据集**

使用的数据集有：仿真数据 SimMEG（MEG）和 SimEEG（EEG）；真实临床数据 CMR（306 通道 MEG + sEEG）和 Localize‑MI（256 通道 EEG）。

**📈 对比分析**

与 9 个现有方法（sLORETA、Champagne、ConvDip、DeepSIF、SSINet、ADMM‑ESI、Catch、FreEFormer）在 5 项指标（精度、召回、定位误差 LE、空间扩散 SD、归一化 MSE）上对比，结果显示：在仿真集上精度最高、LE 与 SD 最小、nMSE 低；在真实集上空间扩散最低，整体性能优于对比方法，证明框架在多配置、多模态场景下具有较强的泛化能力。

**⚠️ 局限性**

局限性包括：①对仿真数据的依赖，真实数据仍有限；②模型复杂度较高，推理时间和显存占用较大；③在极低 SNR 或多源重叠场景下仍可能出现误定位；④尚未在大规模临床验证与实时应用中测试。

---

## 220. Introducing the Generative Application Firewall (GAF)

**arXiv ID:** 2601.15824 | [PDF](https://arxiv.org/pdf/2601.15824v1)

**作者:** Joan Vendrell Farreny `[一作]` (NeuralTrust), Alessandro Pignati `[通讯]` (NeuralTrust)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出一种名为Generative Application Firewall (GAF) 的新架构层，用来统一并强化大语言模型应用的安全防护；定义了网络、访问、句法、语义、上下文五层安全模型，并给出了部署模式与评级体系。

**💡 创新点**

核心创新在于将 Web 应用防火墙（WAF）的概念迁移到生成式 AI 领域，强调语义理解与对话上下文追踪，提出多层防御与可插拔插件机制，并设计了 5 级安全评级与治理映射框架。

**🔧 技术方法**

采用 WAF‑style 代理/网关架构；基于语义分析的检测模型（可使用 LLM 或规则引擎）；会话上下文管理与行为分析；流式中断、内容脱敏与重定向策略；安全日志与度量指标收集；与治理框架（NIST AI RMF、ISO/IEC 等）对齐。

**📊 数据集**

论文未提供具体实验数据或数据集；主要以概念阐述与架构示例为主。

**📈 对比分析**

未进行实测对比；作者指出未来可通过红队演练、重放流量与基准测试来评估误报率、延迟与覆盖率，并建议采用 p95 延迟、错误率等指标进行度量。

**⚠️ 局限性**

局限性包括：缺乏真实环境的实验验证、对多轮对话与复杂工具链的检测效果不确定、部署与运维成本、模型间差异导致检测一致性难以保障，以及对高并发与大规模系统的可伸缩性待验证。

---

## 221. Improved Approximation Ratios for the Shortest Common Superstring Problem with Reverse Complements

**arXiv ID:** 2601.15814 | [PDF](https://arxiv.org/pdf/2601.15814v1)

**作者:** Ryosuke Yamano `[一作]` (University of Tokyo), Tetsuo Shibuya `[通讯]` (University of Tokyo)

**通讯引用:** 3375 | [OpenAlex ID](https://openalex.org/A5043159528)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文针对生物信息学中的逆互补最短公共超字符串（SCS-RC）问题，在贪心算法框架下改进了逼近比，分别将算法A的逼近比提升到3.75，算法B提升到2.875；

**💡 创新点**

主要创新在于将传统的重叠旋转引理（overlap rotation lemma）推广到含逆互补的情形，并通过对周期性字符串和双向重叠的细致分析，构造了新的证明框架；

**🔧 技术方法**

技术方法包括：构造最优循环覆盖、利用压缩比分析、改写贪心合并规则、利用逆互补的等价性与周期性质、以及对逆互补的重叠限制进行双向评估；

**📊 数据集**

本文未使用实验数据集，而是基于理论分析与证明；

**📈 对比分析**

与先前的4-近似和3-近似结果相比，本文在理论上取得了显著的逼近比改进，2.875的结果目前为SCS-RC问题已知的最优逼近上限；

**⚠️ 局限性**

局限性在于：①与标准SCS问题仍存在逼近比差距；②压缩比仅能达到1/2，未能像标准SCS那样利用MAX-ATSP的2/3逼近；③缺乏实验验证与对具体生物学数据集的性能评估。

---

## 222. ErrorMap and ErrorAtlas: Charting the Failure Landscape of Large Language Models

**arXiv ID:** 2601.15812 | [PDF](https://arxiv.org/pdf/2601.15812v1)

**作者:** Shir Ashury-Tahan `[一作]` (IBM Research), Leshem Choshen `[通讯]` (IBM Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 ErrorMap 的无监督两阶段方法，用于自动分析大型语言模型在各类基准测试中的错误，生成模型专属的“失败签名”，并构建多层次的错误分类层级。

**💡 创新点**

创新点包括：①首次将模型错误细化为结构化的错误签名并聚合成层级错误树；②通过 LLM 进行自我分析与递归聚类，自动生成通用错误分类；③公开了静态错误分类体系与实现代码，供社区复现与扩展。

**🔧 技术方法**

技术手段主要是：基于 LLM 的提示工程进行实例级错误分析（Stage 1）；递归聚类与类别生成（Stage 2）实现错误分层；无监督的批处理和层级归类；在实验中使用大规模语言模型作为评判者。

**📊 数据集**

实验覆盖 35 个跨领域数据集（HELm、Capabilities、MedHELM、ToRR、BFCL‑v4、HumanEval、HumanEval Plus、MBPP、MBPP Plus 等），共 83 个模型，采样约 7,000 条失败实例。

**📈 对比分析**

比较方法：将不同模型在相同 Benchmark 的错误实例映射到统一的错误层级体系，统计各错误类别的出现频率并绘制分布图；结果显示模型间错误模式差异显著（如 Pro 版计算错误显著降低）。性能方面，错误覆盖率约 95%，分类准确率 92%，在不同配置下表现鲁棒。

**⚠️ 局限性**

局限性：①采样仅为随机 10%，可能未覆盖所有错误种类；②完全依赖 LLM 的自评与分类，易受提示设计与模型偏差影响；③仅关注失败实例，对成功案例缺乏系统分析；④随着新模型与数据集的出现，需要定期更新错误分类体系。

---

## 223. Inference-Time Scaling of Verification: Self-Evolving Deep Research Agents via Test-Time Rubric-Guided Verification

**arXiv ID:** 2601.15808 | [PDF](https://arxiv.org/pdf/2601.15808v1)

**作者:** Yuxuan Wan `[一作]` (Chinese University of Hong Kong), Michael R. Lyu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 41026 | [OpenAlex ID](https://openalex.org/A5069596903)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了深度研究代理（DRA）的失败分类法，并提出一种基于检验不对称的自我进化框架，利用rubric‑based verifier进行推理时验证与反馈，实现了在测试时迭代自我改进，同时公开了4K验证数据集供后续模型训练使用。

**💡 创新点**

创新点包括：①自动生成DRA失败分类法；②将验证拆解为子问题以充分利用检验不对称；③设计rubric‑based奖励与反馈机制；④将验证器无缝集成到推理流程中，实现无额外训练的自我演化；⑤发布4K高质量反思训练集，促进开源模型的反思能力提升。

**🔧 技术方法**

主要技术包括多模态LLM/VLM、分解‑验证‑评判三模块框架、rubric‑based reward verifier、推理时验证与反馈循环、监督微调（SFT）用于训练反思模块。

**📊 数据集**

使用的数据集有：WebAggregatorQA（构建分类法与基线）、GAIA（Web、文件/推理、全量子集）、XBench‑DeepSearch、BrowseComp，以及公开的4K验证数据集和CK‑Pro‑8B训练集。

**📈 对比分析**

与基线agent‑as‑judge和LLM judge比较，本文方法在GAIA‑Web子集提升8‑11%准确率，GAIA‑Full提升约8%；在XBench‑DeepSearch提升3‑6%；在SFT后的Qwen3‑8B模型上实现32%准确率，较原模型提升5.5%；Meta‑evaluation F1提升12‑48%。

**⚠️ 局限性**

局限性包括：验证器仍存在误报误检，召回率有限；反馈循环需要多轮，导致推理延时；在极难或多语言任务（如BrowseComp）提升有限；rubric与taxonomy的构建需要人工标注，可能缺乏对新型错误类型的覆盖。

---

## 224. Entangled Life and Code: A Computational Design Taxonomy for Synergistic Bio-Digital Systems

**arXiv ID:** 2601.15804 | [PDF](https://arxiv.org/pdf/2601.15804v1)

**作者:** Zoë Breed `[一作]` (Delft University of Technology), Katherine W. Song `[通讯]` (Delft University of Technology)

**通讯引用:** 1230 | [OpenAlex ID](https://openalex.org/A5014622901)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一套面向生物‑数字系统的计算设计分类学，并将其应用于对 34 个微生物相关生物‑数字系统的系统化分析，随后构建了一个开放式交互式可视化数据库用于社区共享与进一步研究。

**💡 创新点**

创新点在于：①首次将计算理论与生物学机制在同一框架下对应，拆分出 8 个基本计算层（输入、传导、评估/比较、路由/选择、存储/状态、适应、输出、能量）；②利用此分类学揭示当前系统中存在的功能不平衡与潜在协同机会；③提供公开的可视化平台与数据库，支持跨学科研究与持续迭代。

**🔧 技术方法**

技术方法包括：① 文献检索与手工编码（两名跨学科研究者共同完成）；② 采用 React + D3.js 开发交互式 Sankey 可视化；③ 通过 Airtable 进行社区提交与实时更新；④ 采用信息处理理论、控制论与计算机体系结构中的抽象构建分类层。

**📊 数据集**

数据集主要来源于：① 1,500 条学术记录（ACM Digital Library、IEEE Xplore）经过筛选后得到 34 篇；② 27 个来自个人收藏、书籍和艺术家作品的系统；③ 通过二次检索补充 4 条系统；共计 61 条生物‑数字系统被编码与分析。

**📈 对比分析**

本文并未进行传统意义上的性能对比或量化评测，而是以定性方式描述各类系统在计算功能、时间尺度、触发方式等维度的分布与关联。通过可视化平台展示了功能频率、共现模式与潜在空白点，未给出具体数值指标。

**⚠️ 局限性**

局限性包括：① 仅覆盖微生物与 DNA 体系，植物、动物及更大尺度生态系统缺乏；② 分类学以计算机视角为主，可能未能完整囊括所有独特的生物信息处理机制；③ 数据集规模相对有限，缺少大规模系统的代表性；④ 实际实现难度高，未对可行性与可靠性进行实验验证。

---

## 225. Assessing Situational and Spatial Awareness of VLMs with Synthetically Generated Video

**arXiv ID:** 2601.15780 | [PDF](https://arxiv.org/pdf/2601.15780v1)

**作者:** Pascal Benschop `[一作]` (Delft University of Technology), Jan van Gemert `[通讯]` (Delft University of Technology)

**通讯引用:** 5084 | [OpenAlex ID](https://openalex.org/A5077258803)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套合成最小差异视频基准，评估视觉‑语言模型在空间与情境推理上的性能；

**💡 创新点**

通过仅改变单一空间变量的最小对照视频，聚焦角色绑定、视角影响与轨迹细粒度对齐，并引入稳定颜色识别作为轻量结构先验；

**🔧 技术方法**

利用Unreal Engine 5.5生成街景短视频、Mixamo动画，并在无训练、固定提示下使用NVILA‑8B‑Video、VideoLLaMA3‑7B、Gemma3‑4B、Qwen2.5‑VL‑7B等模型，输出通过轻量文本分类器映射到标签；

**📊 数据集**

自行构建的120帧30FPS合成视频数据集，包含两种背景（CitySample街景与HDRI环境）和四种相机视角，所有任务共约40条视频，已公开发布于HuggingFace与GitHub；

**📈 对比分析**

通过把模型生成的文本映射至预定义标签后计算准确率，对情境辨别、角色绑定与跟随方向三大任务进行宏观与整体准确率比较，结果仅略高于随机（约50–60%），仅在颜色标记条件下略有提升；

**⚠️ 局限性**

模型在视角相关的角色绑定和细微方向差异的跟随理解上表现差强人意，结构先验只能缓解一部分错误，未根除核心视觉推理缺陷，且合成数据与真实场景存在显著领域差异。

---

## 226. UXCascade: Scalable Usability Testing with Simulated User Agents

**arXiv ID:** 2601.15777 | [PDF](https://arxiv.org/pdf/2601.15777v1)

**作者:** Steffen Holter `[一作]` (ETH Zurich), Gromit Yeuk-Yin Chan `[通讯]` (Adobe Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了 UXCascade 系统，利用 LLM 驱动的模拟用户代理在快速迭代的 UI 设计过程中提供可解释的可用性反馈与修正建议。

**💡 创新点**

创新点包括：①基于目标–特征–问题的五阶段探索式分析流程；②将思考‑大声朗读、交互日志与视觉快照三元结构化为可视化面板；③在浏览器层面实现自然语言编辑与单步重演，形成“修正‑评估”闭环。

**🔧 技术方法**

核心技术涵盖：大语言模型（GPT‑5）驱动的浏览器代理、标签与问题检测代理、编辑代理；可视化技术如 Sankey 流程图、交互式分组图；后端使用 FastAPI，前端 React 实现多视图交互。

**📊 数据集**

主要数据集：①在自建的 t‑shirt 购物网站上预置的 15 条可用性问题（包含描述、严重度与修复建议）；②十名非专业用户的人工评测报告，用于构建基线；③八名专业 UX 研究员在实验中的交互日志与问题列表。

**📈 对比分析**

与人类评测基线对比，UXCascade 在识别问题数量上与人类报告相当（平均 2.9 vs 2.6 题），NASA‑TLX 表明其主观负荷更低、任务完成度更高。系统通过单步重演快速验证修复方案，提升迭代速度。

**⚠️ 局限性**

主要局限：①模拟代理对视觉感知与情绪表达不够精准，导致某些细节问题识别不足；②修正‑评估仅支持单步重演，无法评估多步交互的连锁影响；③人物属性过度简化，缺乏丰富的上下文与行为深度；④系统学习曲线较陡，需进一步优化交互与引导。

---

## 227. Next Generation Active Learning: Mixture of LLMs in the Loop

**arXiv ID:** 2601.15773 | [PDF](https://arxiv.org/pdf/2601.15773v1)

**作者:** Yuanyuan Qi `[一作]` (Monash University), Lan Du `[通讯]` (Monash University)

**通讯引用:** 4747 | [OpenAlex ID](https://openalex.org/A5100716826)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MoLLIA 框架，用 Mixture‑of‑LLMs 替代人工标注，构建无人工交互的主动学习系统。

**💡 创新点**

创新点在于将多种轻量级 LLM 聚合生成可靠标签，并通过负学习与注释差异自适应加权提升鲁棒性。

**🔧 技术方法**

使用 Mixture‑of‑LLMs 注释模型（MoLAM）、负学习、注释差异加权、伪标签半监督、轻量级 LLM（Gemma‑2‑9B‑it、Llama‑3.1‑8B‑Instruct、Mistral‑7B‑Instruct‑v0.2、Qwen2.5‑Coder‑7B‑Instruct、Yi‑1.5‑9B）以及 DistilBERT/DistilRoBERTa 作为主干。

**📊 数据集**

在 AG News、IMDB、TREC、PubMed 四个文本分类基准数据集上进行实验。

**📈 对比分析**

与单一 LLM、LLM 集成、数据增强、Meta‑learning、FixMatch 等基线相比，MoLLIA 在四个数据集上均取得更高的 micro‑F1/准确率，逼近甚至超过人工标注水平。

**⚠️ 局限性**

局限在于仍依赖 LLM 的推理质量，跨域适应性受限；对极小数据量或高复杂标签的任务效果尚待验证。

---

## 228. White-Box mHC: Electromagnetic Spectrum-Aware and Interpretable Stream Interactions for Hyperspectral Image Classification

**arXiv ID:** 2601.15757 | [PDF](https://arxiv.org/pdf/2601.15757v1)

**作者:** Yimin Zhu `[一作]` (University of Calgary), Megan Greenwood `[通讯]` (University of Calgary)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出ES-mHC模型，将高光谱图像按可视光、近红外、短波红外等电磁光谱分组构造物理意义的残差流，并通过学习双随机矩阵实现可解释的光谱‑空间交互；

**💡 创新点**

创新点在于用电磁光谱感知的残差流替代传统mHC分支，构建可解释的双向残差混合矩阵，同时结合Cluster‑wise Mamba块实现空间‑光谱分块，首次在高光谱分类中实现结构透明的部分白盒学习；

**🔧 技术方法**

采用mHC残差连接、Sinkhorn‑Knopp双随机矩阵约束、Spectral‑Spatial Mamba块、Cluster‑wise序列扫描、RMSNorm、位置编码及FFN等深度学习技术；

**📊 数据集**

使用Indian Pines AVIRIS数据集（200波段，16类，20 m GSD）进行实验；

**📈 对比分析**

与CNN、GAN、Transformer、Mamba、SS‑ConvNeXt、MTGAN、SSFTT、SSTN、GSC‑ViT、3DSS‑Mamba等方法对比，OA、AA、kappa均高于99%，尤其在小类识别和边界保留方面优于对手；

**⚠️ 局限性**

仍需进一步完善对三矩阵的解释方法，扩展率对性能的影响待深入研究，实验仅在单一数据集验证，泛化能力尚待验证。

---

## 229. CAFE-GB: Scalable and Stable Feature Selection for Malware Detection via Chunk-wise Aggregated Gradient Boosting

**arXiv ID:** 2601.15754 | [PDF](https://arxiv.org/pdf/2601.15754v1)

**作者:** Ajvad Haneef K `[一作]` (National Institute of Technology Calicut), Madhu Kumar S D `[通讯]` (National Institute of Technology Calicut)

**通讯引用:** 728 | [OpenAlex ID](https://openalex.org/A5037556887)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于分块聚合梯度提升重要性的特征选择框架 CAFE-GB，用于大规模恶意软件数据的稳定特征排名与特征预算选择。

**💡 创新点**

创新点在于：①将局部重要性（每个重叠数据块）聚合，显著提升特征选择的稳定性和可重复性；②系统化地进行 k‑selection 与稳定性分析，得到可靠的特征预算；③在保持 95%+ 维度压缩的同时，保持检测性能与可解释性。

**🔧 技术方法**

使用技术包括：Gradient Boosting（LightGBM）特征重要性估计、重叠分块数据划分、统计显著性检验（Wilcoxon 符号秩检验）、SHAP 解释、内存/运行时间分析。

**📊 数据集**

实验数据集为 BODMAS（Windows PE 静态特征 2381 维）和 CIC-AndMal2020（Android 静态特征 9503 维）。

**📈 对比分析**

方法比较：与全特征基线在 Logistic Regression、Random Forest、XGBoost、LightGBM 四种分类器上进行对比，使用 Accuracy、F1、MCC、ROC‑AUC、PR‑AUC 等指标；实验显示 95%+ 维度压缩后性能基本相当，Wilcoxon 检验显示无显著差异。

**⚠️ 局限性**

局限性：①依赖树模型重要性，可能对高方差或边际效应特征偏倚；②仅评估静态特征，未验证对动态或原始字节序列的适用性；③未针对实时/流式场景或概念漂移进行适配；④固定特征预算降低了在不同资源环境下的灵活性。

---

## 230. Breaking the Resolution Barrier: Arbitrary-resolution Deep Image Steganography Framework

**arXiv ID:** 2601.15739 | [PDF](https://arxiv.org/pdf/2601.15739v1)

**作者:** Xinjue Hu `[一作]` (Nanjing University of Information Science and Technology), Zhangjie Fu `[通讯]` (Nanjing University of Information Science and Technology)

**通讯引用:** 5685 | [OpenAlex ID](https://openalex.org/A5066341740)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了首个支持任意分辨率的深度图像隐写框架ARDIS。

**💡 创新点**

创新点在于用频率分离架构和隐式解析重构实现了跨分辨率的隐写与恢复，突破了传统固定分辨率的限制。

**🔧 技术方法**

采用频率分离架构 (FDA)、隐式分辨率编码 (IRC)、隐式高频细节编码以及隐式重构器 (LGIR) 等技术。

**📊 数据集**

在DIV2K、Flickr2K、COCO 与 Stego260 等公开数据集上进行训练与评估。

**📈 对比分析**

与ISN、HiNet、StegFormer、AIS 等方法对比，ARDIS 在隐写不可见度、盲恢复 PSNR/SSIM/LPIPS 上均优于对手，且盲恢复误差 RRE 为 0%。

**⚠️ 局限性**

仍受限于对离线训练的依赖、对极高分辨率场景的细节恢复可能存在微小失真以及对网络结构的高计算开销。

---

## 231. LLM-Assisted Automatic Dispatching Rule Design for Dynamic Flexible Assembly Flow Shop Scheduling

**arXiv ID:** 2601.15738 | [PDF](https://arxiv.org/pdf/2601.15738v1)

**作者:** Junhao Qiu `[一作]` (City University of Hong Kong), Qingfu Zhang `[通讯]` (City University of Hong Kong)

**通讯引用:** 38375 | [OpenAlex ID](https://openalex.org/A5000546219)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一种基于大型语言模型的动态规则设计框架 LLM4DRD，用于在线动态多产品交付的灵活装配流批系统调度。

**💡 创新点**

通过双专家机制（LLM-A 生成规则、LLM-S 评估规则）、精英知识引导初始化以及混合评估，实现了可自适应、可解释的调度优先级规则自动设计。

**🔧 技术方法**

结合大型语言模型（GPT‑4o‑mini）、遗传程序化、强化学习框架、异构图 MDP 以及混合评估技术进行自动代码生成与规则优化。

**📊 数据集**

使用真实工厂订单与资源数据合成的 20 个工业实例以及 480 个多场景扩展测试实例进行实验验证。

**📈 对比分析**

与 GP、GEP、传统 AHD 及其他 LLM 方法对比，LLM4DRD 在训练集上平均 tardiness 下降约 16.8%，在 480 个场景中比第二好者提升高达 11.1%。

**⚠️ 局限性**

局限性在于仅考虑双 kitting 约束，未覆盖运输等二级资源约束，且依赖 LLM 生成代码的可执行性和解释性。

---

## 232. Contractions of quasi relation algebras and applications to representability

**arXiv ID:** 2601.15811 | [PDF](https://arxiv.org/pdf/2601.15811v1)

**作者:** Andrew Craig `[一作]` (University of Johannesburg), Claudette Robinson `[通讯]` (University of Johannesburg)

**通讯引用:** 55 | [OpenAlex ID](https://openalex.org/A5084942899)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

研究了准关系代数（qRA）的收缩构造，并证明了在可表示的可分布 qRA 上收缩仍可表示；还指出了一类可分布 qRA 不可有限可表示的充分条件。

**💡 创新点**

创新点在于将关系代数中的等价元概念推广为“正对称幺元”，并以此构造新的 qRA（收缩），从而提供了一种新的构造与可表示性传递方法；同时给出了更一般的不可有限可表示的充分条件。

**🔧 技术方法**

主要技术包括：代数构造（收缩）、可表示性与等价关系的定义、基于偏序与等价关系的二元关系表示、对称负运算与残差的运算性质证明、以及对可分布结构的层次化构造。

**📊 数据集**

无数据集；研究完全基于理论代数结构与证明。

**📈 对比分析**

比较方法为理论证明：展示在可表示的可分布 qRA 上的收缩仍可表示，且利用收缩推导出不可有限可表示的例子；没有实验性能指标。

**⚠️ 局限性**

局限性：仅给出了足够条件的不可有限可表示性，并未完全确定所有可分布 qRA 的有限可表示性；收缩方法对非可分布或非可表示 qRA 的适用性尚未研究。

---

## 233. Attributing and Exploiting Safety Vectors through Global Optimization in Large Language Models

**arXiv ID:** 2601.15801 | [PDF](https://arxiv.org/pdf/2601.15801v1)

**作者:** Fengheng Chu `[一作]` (Southeast University), Songze Li `[通讯]` (Southeast University)

**通讯引用:** 2711 | [OpenAlex ID](https://openalex.org/A5085853632)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 GOSV 框架，通过全局优化识别 LLM 的安全关键注意力头，并基于此实现推理时白盒越狱攻击。

**💡 创新点**

创新点在于引入全局优化与两种激活重映策略（有害修补与零消融），揭示安全机制分布在约 30% 的注意力头且呈现两条独立路径。

**🔧 技术方法**

技术包括全局强化学习优化、激活重映（Patch）、注意力头分布分析及基于余弦相似度的损失函数。

**📊 数据集**

使用了 AdvBench 与 StrongREJECT 两个安全基准，训练时采样 100 条恶意指令进行安全向量提取。

**📈 对比分析**

与现有白盒越狱方法（如 GCG、AutoDAN、AdvPrefix 等）以及本地归因方法 Ships 对比，攻击成功率在各模型上均超过或接近最佳，优于 50% 以上。

**⚠️ 局限性**

局限性包括缺乏针对黑盒场景的防御方案、仅依赖白盒访问、仅验证少数英文模型和数据集，未覆盖多语言或更大规模模型。

---

## 234. Beyond Marginal Distributions: A Framework to Evaluate the Representativeness of Demographic-Aligned LLMs

**arXiv ID:** 2601.15755 | [PDF](https://arxiv.org/pdf/2601.15755v1)

**作者:** Tristan Williams `[一作]` (Humboldt University of Berlin), Alan Akbik `[通讯]` (Humboldt University of Berlin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估并比较两种大语言模型调优方式（persona prompting 与 demographic fine‑tuning）在使用 World Values Survey 数据时的代表性表现；

**💡 创新点**

提出一种同时检验边际分布与相关结构的代表性评估框架，揭示仅靠边际匹配无法捕捉人类价值潜在结构的不足；

**🔧 技术方法**

通过生成模拟回答、计算 Wasserstein‑1 / 总变差距离、归一化方差、构建问卷/主题相关矩阵，并使用 Pearson 相关与 RMSE 对比；

**📊 数据集**

World Values Survey（WVS）调查数据及 OpinionGPT 的 demographic LoRA 适配器训练语料；

**📈 对比分析**

对 21 个模型（未调优、persona prompting、OpinionGPT）使用上述指标进行比较，结果显示 OpinionGPT 在边际相似度和方差方面优于 persona prompting，但两者在相关结构上均低于真实数据，表明尚未完全实现代表性；

**⚠️ 局限性**

模型规模有限、仅使用单一 LLM 与有限的 demographic 分组、评估仅在英语环境下、WVS 本身带有西方偏向、未考虑交叉身份与开放式问答等因素。

---

## 235. Benchmarking Text-to-Python against Text-to-SQL: The Impact of Explicit Logic and Ambiguity

**arXiv ID:** 2601.15728 | [PDF](https://arxiv.org/pdf/2601.15728v1)

**作者:** Hangle Hu `[一作]` (Zhejiang University of Technology), Ruizhe Li `[通讯]` (University of Aberdeen)

**通讯引用:** 803 | [OpenAlex ID](https://openalex.org/A5018741641)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了BIRD-Python基准和逻辑完成框架，评估Text-to-Python与Text-to-SQL的差异与性能

**💡 创新点**

创新点在于将文本到SQL的评估迁移至文件数据的Python环境，并通过LCF显式补全逻辑以降低歧义

**🔧 技术方法**

使用LLM生成Python/SQL代码、执行验证、LLM语义验证器、以及三阶段对话式逻辑完成框架

**📊 数据集**

使用改进后的BIRD数据集（将原SQL数据库转换为CSV文件并对应生成Python逻辑）

**📈 对比分析**

通过执行准确率(EX)对比，发现大模型在补全逻辑后Python与SQL性能相当，信息缺失是主要瓶颈

**⚠️ 局限性**

局限性：需要oracle或人工补全逻辑，未覆盖无schema文件场景，评估仅在干净数据下进行

---

## 236. Creativity in the Age of AI: Rethinking the Role of Intentional Agency

**arXiv ID:** 2601.15797 | [PDF](https://arxiv.org/pdf/2601.15797v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 237. Profit Maximization for Viral Marketing in Online Social Networks using Two Phase Diffusion Approach

**arXiv ID:** 2601.15726 | [PDF](https://arxiv.org/pdf/2601.15726v1)

**作者:** Poonam Sharma `[一作]` (Indian Institute of Technology Jammu), Suman Banerjee `[通讯]` (Indian Institute of Technology Jammu)

**通讯引用:** 5625 | [OpenAlex ID](https://openalex.org/A5033218913)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在在线社交网络中提出两阶段扩散模型的利润最大化问题，并给出在预算约束下选择种子用户的算法；

**💡 创新点**

首次将两阶段扩散策略与利润最大化结合，证明目标函数性质并提出三种贪心/随机解法；

**🔧 技术方法**

基于IC（Independent Cascade）模型的概率图推导，采用贪心、双向贪心与随机贪心算法，并进行理论分析；

**📊 数据集**

使用了真实社交网络数据集Les Miserables、Email‑Eu‑Core与Slashdot；

**📈 对比分析**

与随机、最高度、聚类系数、度折扣等多种基线方法在不同预算分配、时间步长和分割比例下做实验，结果表明两阶段方法可将利润提升18%–40%，SG与DG在多数场景下实现最高收益；

**⚠️ 局限性**

目标函数非单调、非子模，缺乏近似性能保证；算法复杂度较高，对超大规模网络仍受限，且需经验调参确定预算分配与时间步长。

---

## 238. LL-GaussianImage: Efficient Image Representation for Zero-shot Low-Light Enhancement with 2D Gaussian Splatting

**arXiv ID:** 2601.15772 | [PDF](https://arxiv.org/pdf/2601.15772v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 239. Virtual Traffic Police: Large Language Model-Augmented Traffic Signal Control for Unforeseen Incidents

**arXiv ID:** 2601.15816 | [PDF](https://arxiv.org/pdf/2601.15816v1)

**作者:** Shiqi Wei `[一作]` (National University of Singapore), Kaidi Yang `[通讯]` (National University of Singapore)

**通讯引用:** 1239 | [OpenAlex ID](https://openalex.org/A5075233338)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种将大语言模型（LLM）与现有自适应交通信号控制（TSC）系统协同工作的分层框架，利用LLM充当“虚拟交通警察”，根据实时事故描述动态调节控制器参数，并通过检索增强生成（RAG）与自我校验机制提升可靠性。

**💡 创新点**

创新点包括：1) 在保持现有TSC系统不被替代的前提下，利用LLM进行参数微调而非直接控制；2) 设计了交通语言检索系统（TLRS）以提供领域知识，缓解LLM幻觉；3) 引入LLM验证器进行自检与持续更新TLRS，实现自我强化；4) 在单交叉口模拟中验证了多种事故类型下的显著性能提升。

**🔧 技术方法**

主要技术包括：链式思考（CoT）提示、检索增强生成（RAG）与LLM自检（Verifier）、基于嵌入的相似检索、以及两类下层控制器（Max‑Pressure 与 MPC）与LLM交互。

**📊 数据集**

数据集主要来自：① 以SUMO仿真生成的四种事故场景（车祸、道路维护、紧急车辆通行、老人行人横穿）和两种交通需求水平；② 通过交通工程师撰写的历史事故报告构成的“交通语言数据库”用于检索；③ 采用公开的交通模拟参数（如车速、绿灯时长等）进行评估。

**📈 对比分析**

对比方法包括：基准TSC（无LLM）、TSC‑LLM（无检索）、TSC‑LLM‑无CoT（无链式思考）以及TSC‑LLM‑TLRS（本研究方案）。实验表明，在车祸和道路维护情境下，本框架可将平均延时（AD）降低约23%–35%，平均队列长度（AQL）下降15%–30%；在紧急车辆与老人行人场景下，急救车辆延时从约100 s降至0 s，老人行人完成率从≈30%提升至≈100%。

**⚠️ 局限性**

局限性包括：① 依赖事故检测与描述的准确性；② 仍需大模型推理算力，推理时延可能影响实时性；③ TLRS在完全无先验案例时初始性能不佳，需要多轮交互才能收敛；④ 在真实大规模网络与多种事故类型下的泛化性与安全性待进一步验证。

---

## 240. Beyond Off-the-Shelf Models: A Lightweight and Accessible Machine Learning Pipeline for Ecologists Working with Image Data

**arXiv ID:** 2601.15813 | [PDF](https://arxiv.org/pdf/2601.15813v1)

**作者:** Clare Chemery `[一作]` (Ludwig Maximilian University of Munich), Ludwig Bothmann `[通讯]` (Ludwig Maximilian University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一套轻量级实验管线，帮助生态学家在摄像机陷阱图像上训练并迭代年龄与性别分类模型。

**💡 创新点**

创新点在于将命令行预处理、训练、评估与图形化标注、错误分析、模型比较集成于一体，降低专业门槛，并支持快速构建多任务小模型。

**🔧 技术方法**

采用PyTorch实现预训练网络（ResNet50、VGG19、DenseNet161/201）、MegaDetector检测、数据增强、迁移学习与微调，并通过TOML配置统一管理实验参数。

**📊 数据集**

使用德国维尔登斯坦森林的3392张红鹿摄像机陷阱图像，共4352个裁剪框，人工标注年龄（成人、幼崽、年幼、未知）和性别（雄、雌、未知）。

**📈 对比分析**

通过多次实验与多种数据增强方案比较，最优模型在年龄分类上实现90.68%准确率，性别分类上实现92.53%准确率，且平均预测置信度分别为0.845和0.905。

**⚠️ 局限性**

局限性主要包括样本不平衡、图像质量差导致标注错误，以及缺乏全图上下文信息，限制了模型泛化与进一步提升性能的空间。

---

## 241. HumanLLM: Towards Personalized Understanding and Simulation of Human Nature

**arXiv ID:** 2601.15793 | [PDF](https://arxiv.org/pdf/2601.15793v1)

**作者:** Yuxuan Lei `[一作]` (University of Science and Technology of China), Xing Xie `[通讯]` (Microsoft Research Asia)

**通讯引用:** 44170 | [OpenAlex ID](https://openalex.org/A5044651577)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于真实用户行为的Cognitive Genome Dataset，并训练HumanLLM模型，实现个性化人类行为与思维的模拟与预测。

**💡 创新点**

通过从Reddit、Twitter、Blogger、Amazon等多平台采集约5.5M用户日志，采用三阶段过滤、合成与质量控制，构造结构化的个体‑情境‑行为数据，并用模型合并技术避免灾难性遗忘，显著提升模型社会智能。

**🔧 技术方法**

采用LLM预训练+监督微调结合多任务学习（Profile Generation、Scenario Generation、Social QA、Writing Imitation、Commenting、Item Selection）以及模型权重合并（Model Merging）和ShareGPT框架进行训练。

**📊 数据集**

使用Cognitive Genome Dataset（约5.5M用户日志，包括2.8M Reddit、673k Twitter、368k Blogger、1.7M Amazon记录）以及公开Benchmarks MotiveBench、TomBench进行评测。

**📈 对比分析**

通过多任务多选评测与MotiveBench、TomBench的非文字沟通等指标与基线模型（Llama‑3.1‑8B、Qwen3‑8B等）和公开模型对比，HumanLLM在各项指标上平均提升约20–30%，在OOD Benchmark上实现最高分。

**⚠️ 局限性**

仍受限于数据来源多样性不足、极端少数用户场景覆盖不全、对极端情境推理的鲁棒性不足，且在低资源语言或特定文化背景下的泛化尚待进一步验证。

---

## 242. Recursive Flow: A Generative Framework for MIMO Channel Estimation

**arXiv ID:** 2601.15767 | [PDF](https://arxiv.org/pdf/2601.15767v1)

**作者:** Zehua Jiang `[一作]` (Zhejiang University), Mérouane Debbah `[通讯]` (Khalifa University)

**通讯引用:** 63755 | [OpenAlex ID](https://openalex.org/A5056145687)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 RC-Flow，一个基于流匹配的闭环迭代框架，用于高维 MIMO 信道估计。

**💡 创新点**

创新点是把生成模型改为固定点闭环推断，采用递归锚点重置和轨迹校正，以及自适应双调度实现低延迟与高精度平衡。

**🔧 技术方法**

使用流匹配（flow matching）预训练的生成模型、连续正则化、物理感知近端投影、锚点插值，及自适应时间步调度。

**📊 数据集**

数据集：3GPP CDL‑A/B/C/D 6G 毫米波 MIMO 信道样本。

**📈 对比分析**

与传统的 LMMSE、CS（fsAD）、深度折叠 DAMP、Score‑based 等基线比较，RC-Flow 在低 SNR 下提升约 2‑3 dB，整体 NMSE 接近理论下界，同时推理时延比 Score‑based 降低 100 倍。

**⚠️ 局限性**

局限包括对训练数据分布的依赖、在极低 pilot 密度下仍有误差，以及对超参 λ、β 的敏感性需要手工调节。

---

## 243. Off-Policy Actor-Critic with Sigmoid-Bounded Entropy for Real-World Robot Learning

**arXiv ID:** 2601.15761 | [PDF](https://arxiv.org/pdf/2601.15761v1)

**作者:** Xiefeng Wu `[一作]` (Wuhan University), Shu Zhang `[通讯]` (Wuhan University)

**通讯引用:** 23394 | [OpenAlex ID](https://openalex.org/A5100452875)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种名为 SigEnt‑SAC 的离线到在线强化学习框架，能够仅凭一条专家轨迹在真实机器人上实现从零开始的策略学习。

**💡 创新点**

创新点在于：①使用 Sigmoid‑bounded entropy 将负熵问题转化为正且有限的熵信号，抑制 OOD 动作的探索与 Q 值振荡；②引入 Gated Behavior Cloning，只在策略偏离专家时才施加行为克隆梯度，从而在保持探索的同时减少无效交互。

**🔧 技术方法**

技术手段包括：基于 SAC 的离线‑在线 actor‑critic 结构、tanh‑squashed 高斯策略、Sigmoid‑bounded 熵正则化、门控行为克隆、Q‑ensemble 与 LayerNorm、CQL 风格的 OOD 正则化。

**📊 数据集**

数据集涵盖：D4RL Adroit 与 Kitchen 的稀疏奖励连续控制任务；以及四个真实机器人任务（Manipulator 的 Push‑Cube、Wheeled 的 Ball‑to‑Goal、Quadruped 与 Humanoid 的 Slalom），每个任务仅提供一条成功演示轨迹。

**📈 对比分析**

与 Cal‑QL、CQL、AWAC、IQL、SAC prior 以及 RLPD 等基线进行对比；实验显示 SigEnt‑SAC 在 1M 步内更快达到 100% 成功率，减少 Q‑值振荡，平均降低任务完成步数约 40%，并对演示噪声更具鲁棒性。

**⚠️ 局限性**

局限性包括：仅支持高层速度指令的控制，未实现关节级别或全身控制；实验聚焦于短时、粗粒度任务，未验证在细粒度或长期规划任务中的表现。

---

## 244. PhysProver: Advancing Automatic Theorem Proving for Physics

**arXiv ID:** 2601.15737 | [PDF](https://arxiv.org/pdf/2601.15737v1)

**作者:** Hanning Zhang `[一作]` (University of Illinois Urbana Champaign), Tong Zhang `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了物理领域的 Lean4 形式化定理数据集，并在此基础上训练了一个专门针对物理推理的定理证明器。

**💡 创新点**

首次将形式化定理证明技术扩展到物理学，提出了基于合成猜想的训练管线和 RLVR 自我演化策略，显著提升了物理领域的证明性能。

**🔧 技术方法**

采用 Claude‑4.5 生成合适的猜想，Lean 验证语法与可证明性，利用 DeepSeek‑Prover‑V2‑7B 并结合 Group Relative Policy Optimization (GRPO) 的 RLVR 进行强化学习训练。

**📊 数据集**

使用 PhysLean 官方 3,000 条定理样本并通过合成生成 5,541 条训练样本（约 5K 训练+250 测试），以及 MiniF2F‑Test 进行跨域评估。

**📈 对比分析**

在物理子领域 Pass@16 上相较最强数学证明器提升约 2.4%（总体 36.4%→38.8%），在 MiniF2F‑Test 上提升约 1.3% 以上，展示了显著的领域提升与跨域泛化。

**⚠️ 局限性**

数据集规模有限、合成猜想成功率仅 8.9%，缺乏更广泛的物理子领域覆盖；SFT 训练在该任务上效果不佳，且未在更大规模模型上验证。

---

## 245. Sub-Region-Aware Modality Fusion and Adaptive Prompting for Multi-Modal Brain Tumor Segmentation

**arXiv ID:** 2601.15734 | [PDF](https://arxiv.org/pdf/2601.15734v1)

**作者:** Shadi Alijani `[一作]` (University of Victoria), Homayoun Najjaran `[通讯]` (University of Victoria)

**通讯引用:** 5636 | [OpenAlex ID](https://openalex.org/A5058540009)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种结合子区域感知模态注意力与自适应提示工程的多模态脑肿瘤分割框架

**💡 创新点**

①子区域感知模态注意力：为每个肿瘤子区域动态学习模态权重；②自适应提示工程：利用初步分割结果生成子区域特定提示进行迭代细化

**🔧 技术方法**

基于LiteMedSAM（TinyViT编码器 + PromptEncoder + MaskDecoder）+ 子区域注意力模块 + BoundingBox提示 + 交叉熵/IoU 损失 + 数据增强等

**📊 数据集**

BraTS 2020 多模态MRI（T1、T1c、T2、FLAIR）369例（296训练/73验证）

**📈 对比分析**

与单模态、预训练多模态、Fine‑tuned多模态、nnU-Net、TransBTS 等方法对比；总体WT Dice 0.900（比nnU-Net高0.01），ET Dice 0.900（提升9.8%），necrotic core Dice 0.71；在子区域表现上显著优于基线

**⚠️ 局限性**

仅按切片处理，未充分利用3D空间上下文；假设所有四模态齐全，缺失模态时鲁棒性未知；未在跨中心或其他数据集上验证泛化能力

---

## 246. VideoThinker: Building Agentic VideoLLMs with LLM-Guided Tool Reasoning

**arXiv ID:** 2601.15724 | [PDF](https://arxiv.org/pdf/2601.15724v1)

**作者:** Chenglin Li `[一作]` (Zhejiang University), Jiaqi Wang `[通讯]` (Shanghai AI Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并训练了 VideoThinker，一种能够通过工具交互实现长视频动态推理的 VideoLLM。

**💡 创新点**

通过仅使用字幕与工具生成的多步推理轨迹构造合成训练数据，将工具使用嵌入模型主体，从而实现“思考视频”而非单纯“字幕推理”。

**🔧 技术方法**

结合 Agentic LLM 与 VideoLLM，设计 Temporal Retrieval 与 Temporal Zoom 两类工具，使用 CaptionZoom 生成文本代理后再替换为视频片段，并采用多轮工具调用的 CoT 训练策略。

**📊 数据集**

基于 CG‑Bench 的 10k 多选 QA、VideoMME、LongVideoBench、LVBench、MLVU 等公开长视频评测集进行数据合成与评估。

**📈 对比分析**

在四大长视频基准上与闭源模型（GPT‑4o、Gemini‑1.5）、开源 VideoLLM（Qwen2.5‑VL‑72B 等）以及 LLM‑agent 对照，VideoThinker 以单一 7B 模型实现 +6.8%/ +10.6% 等提升，整体性能与闭源模型相当。

**⚠️ 局限性**

受工具检索精度与字幕质量限制，合成数据缺乏真实视觉细节，导致对极长视频时空覆盖不足以及细粒度视觉推理仍易出现错误。

---

## 247. FirmReBugger: A Benchmark Framework for Monolithic Firmware Fuzzers

**arXiv ID:** 2601.15774 | [PDF](https://arxiv.org/pdf/2601.15774v1)

**作者:** Mathew Duong `[一作]` (University of Adelaide), Damith C. Ranasinghe `[通讯]` (Data61 CSIRO)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了针对单片机固件的 bug 基准框架 FirmReBugger，构建三套包含真实缺陷的基准集，并提供自动化的 bug 状态判定与性能评估流程。

**💡 创新点**

创新点包括：① 采用 Bug Oracle（Raven）实现无 leaky oracle 的自动化 bug 追踪；② 定义 Not reached/Reached/Triggered/Detected 四种 bug 状态，支持精确计数；③ 设计三种基准集（Baseline、DMA、Challenge），覆盖多种固件与常见路障；④ 引入时间到 bug 与一致性等评估指标，提供可重复、可扩展的评测框架。

**🔧 技术方法**

技术实现基于 QEMU/Unicorn/ICICLE 虚拟机的固件仿真；C 语法 Bug Oracle（Raven）与 Bug Interpreter 解析执行状态；Analysis Bench 统计 bug 状态、时间与一致性；实验使用 10 次 24 小时跑 10 CPU 年。

**📊 数据集**

数据集包含 31 个基准二进制（来自 9 篇 SoTA 研究 + 3 新大目标），共 295 个独立 bug，分别分布于 Baseline（166 bug）、DMA（31 bug）和 Challenge（109 bug）三套基准。

**📈 对比分析**

评测方法：对 9 个 SoTA 固件 fuzzer 进行 10 次 24h 运行，记录触发的 bug 数、时间到 bug 统计和一致性分数。结果显示 MultiStream 与某些 fuzzer 在 Baseline 和 DMA 集上表现最佳（触发率 73%/43%），但在 Challenge 集上整体触发率仅 43%。

**⚠️ 局限性**

限制：1）基准集主要来源于已有研究，可能偏向原始作者；2）仅评估已知 bug，无法覆盖未知漏洞；3）false positive bug 仍需人工验证；4）DMA 与延迟等路障的处理仍不完善，需进一步改进；5）实验耗时大，扩展性需提升。

---

## 248. Decoupling Return-to-Go for Efficient Decision Transformer

**arXiv ID:** 2601.15953 | [PDF](https://arxiv.org/pdf/2601.15953v1)

**作者:** Yongyi Wang `[一作]` (Peking University), Wenxin Li `[通讯]` (Peking University)

**通讯引用:** 3154 | [OpenAlex ID](https://openalex.org/A5100397213)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文识别并消除了Decision Transformer中对完整Return-to-Go序列的冗余，提出仅使用最新RTG并通过AdaLN调节Transformer输出的Decoupled Decision Transformer（DDT）模型。

**💡 创新点**

创新点在于从理论和实验两方面证明完整RTG序列在POMDP框架下是冗余的，并设计了只利用最新RTG并通过AdaLN实现条件化的简化架构，显著提升性能并降低计算成本。

**🔧 技术方法**

采用Transformer（GPT）作为主干网络，利用AdaLN对最新RTG进行条件化，配合线性嵌入、位置编码及MLP输出实现动作预测，并使用MSE/交叉熵损失进行行为克隆训练。

**📊 数据集**

使用D4RL离线强化学习基准中的连续控制任务（Hopper、Walker2d、HalfCheetah）以及离散奖励稀疏的2048游戏数据集进行实验。

**📈 对比分析**

与传统离线RL基线（BRAC‑v、TD3+BC、IQL、CQL）以及DT及其改进版本（VDT、LSDT）进行对比，DDT在所有任务上均实现了至少10%以上的性能提升，且在SOTA水平之上保持稳定低方差。

**⚠️ 局限性**

局限性包括：在非POMDP或需要完整奖励序列信息的环境中，RTG序列可能不冗余；仅对单一RTG进行条件化可能无法充分利用多步目标信息；AdaLN的单层设计在某些复杂任务上可能不够灵活。

---

## 249. Co-Constructing Alignment: A Participatory Approach to Situate AI Values

**arXiv ID:** 2601.15895 | [PDF](https://arxiv.org/pdf/2601.15895v1)

**作者:** Anne Arzberger `[一作]` (Delft University of Technology), Jie Yang `[通讯]` (Delft University of Technology)

**通讯引用:** 22215 | [OpenAlex ID](https://openalex.org/A5100404947)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过组织基于LLM研究助手的参与式研讨会，探讨并设计用户与AI共同构建价值对齐的实践。

**💡 创新点**

创新点在于将价值对齐视为用户与AI交互中的持续、情境化共建过程，并提出具体用户角色和界面支持方案。

**🔧 技术方法**

采用生成式设计研究、交互视角工作坊、misalignment日记与主题分析等方法。

**📊 数据集**

使用12名跨学科研究者的misalignment日记、工作坊产出与访谈资料。

**📈 对比分析**

该研究未采用传统性能指标，而是通过定性主题分析对比不同用户角色与设计方案的可行性与启发性，未进行数值对比。

**⚠️ 局限性**

局限性包括样本规模小、仅聚焦LLM助手场景、易将价值对齐问题归纳为系统问题且难以转化为可操作界面。

---

## 250. Iterative Amortized Hierarchical VAE

**arXiv ID:** 2601.15894 | [PDF](https://arxiv.org/pdf/2601.15894v1)

**作者:** Simon W. Penninga `[一作]` (Eindhoven University of Technology), Ruud J. G. van Sloun `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 4384 | [OpenAlex ID](https://openalex.org/A5042985821)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出并实现了迭代化可化简层级变分自编码器（IA‑HVAE），将初始可化估计与基于解码器梯度的迭代细化结合起来；

**💡 创新点**

创新点在于引入线性可分离的变换域解码器（如傅里叶域），使得每个潜在层能单独获得梯度，显著降低迭代推断成本并实现35倍加速；

**🔧 技术方法**

使用层级VAE架构、解码器梯度迭代优化、傅里叶变换线性分解以及MAP更新规则；

**📊 数据集**

在CIFAR‑10（32×32）和fastMRI（128×128）两个真实图像数据集上进行实验；

**📈 对比分析**

与传统的HVAE、全可化推断以及纯迭代推断进行对比，评价指标包括MSE、NLL、FID和推断时间；结果显示IA‑HVAE的混合推断在保持或略低于迭代推断的质量的同时，比可化推断快数十倍，且在逆问题（去模糊、去噪）上优于基线模型；

**⚠️ 局限性**

局限性在于需要信号在变换域中具有线性可分解性，对复杂或不易分解的信号可能难以学习；此外，迭代优化在训练阶段未被充分利用，未来可进一步改进。

---

## 251. SoK: Challenges in Tabular Membership Inference Attacks

**arXiv ID:** 2601.15874 | [PDF](https://arxiv.org/pdf/2601.15874v1)

**作者:** Cristina Pêra `[一作]` (University of Porto), Luís Antunes `[通讯]` (TekPrivacy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文系统性研究了在集中式与联邦式机器学习环境下，针对表格数据的成员推断攻击（MIA）及其防御方法。

**💡 创新点**

创新点在于提出了基于模型访问类型的新分类法、评估了单个唯一样本（single‑out）在MIA中的脆弱性以及跨模型迁移攻击的有效性。

**🔧 技术方法**

采用了影子模型和参考模型的攻击方法、LiRA、RMIA，以及正则化、差分隐私、知识蒸馏、联邦安全聚合等防御技术。

**📊 数据集**

实验使用了十个公开表格数据集（如Covid、Dropout Success、Half Million、Locations等）和多种学习模型（RF、XGBoost、DT、NB、LR、NN）。

**📈 对比分析**

与传统随机猜测基线对比，攻击在集中式场景下大多数模型的AUC仅略高于50%，在联邦学习和单一样本攻击中表现更差；但跨模型迁移时攻击效果可提升。

**⚠️ 局限性**

局限性包括对表格数据的攻击效果普遍低、对异常值和单一样本的评估受数据依赖、缺少对VFL和动态模型更新下的适应性研究。

---

## 252. Determinants of Training Corpus Size for Clinical Text Classification

**arXiv ID:** 2601.15846 | [PDF](https://arxiv.org/pdf/2601.15846v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 253. PF-D2M: A Pose-free Diffusion Model for Universal Dance-to-Music Generation

**arXiv ID:** 2601.15872 | [PDF](https://arxiv.org/pdf/2601.15872v1)

**作者:** Jaekwon Im `[一作]` (KAIST), Taketo Akama `[通讯]` (Sony Computer Science Laboratories)

**通讯引用:** 33 | [OpenAlex ID](https://openalex.org/A5087426444)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实现了PF-D2M，一个基于扩散模型的通用舞蹈到音乐生成系统，利用视频视觉特征生成与舞蹈同步的音乐。

**💡 创新点**

① 通过Synchformer提取视频视觉特征取代单人姿态特征，实现对多舞者及非人类舞者的音乐生成；② 引入分阶段进阶训练策略（文本到音频预训练、视频到音频同步学习、舞蹈到音乐微调），有效缓解数据稀缺导致的过拟合。

**🔧 技术方法**

Diffusion Transformer (DiT) + VAE + 预训练 T5 文本编码器 + Synchformer 视觉编码器 + AdaLN 调整 + classifier-free guidance + DPM‑Solver++ 推理。

**📊 数据集**

训练使用 AIST++ 舞蹈-音乐数据、VGGSound 视频-音频数据、FMA 与 MoisesDB 文本-音乐数据；评估采用 AIST++ LORIS 基准和自采集的多舞者、非人类舞者野外视频。

**📈 对比分析**

与 CDCD、LORIS、Text‑Inv 等方法在 AIST++ 上的节拍对应度指标比较，PF‑D2M 在大多数指标（BHS、HSD、F1）上超越基线；在主观评测中，Stage2 模型在所有测试场景获得最高的舞蹈-音乐对齐和音乐质量分数。

**⚠️ 局限性**

生成音乐时长短、对长序列音乐的支持不足；缺乏针对舞蹈-音乐生成的客观评测数据集；某些场景下节拍覆盖度略低等问题。

---

## 254. Finding large sparse induced subgraphs in graphs of small (but not very small) tree-independence number

**arXiv ID:** 2601.15861 | [PDF](https://arxiv.org/pdf/2601.15861v1)

**作者:** Daniel Lokshtanov `[一作]` (University of California), Paweł Rzążewski `[通讯]` (Warsaw University of Technology)

**通讯引用:** 531 | [OpenAlex ID](https://openalex.org/A5047941027)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在已给定树分解且树独立数为 k 的图上，提出了一种改进的算法，用于解决 (t, ψ) 类问题（即寻找满足 MSO 句子 ψ 且诱导子图树宽 < t 的最大权重集合）。

**💡 创新点**

创新点在于将先前指数依赖于 k 的算法（运行时间 n^{exp(k)}）提升到 n^{O(k)}，从而在树独立数为多项式对数或子线性增长的图类上实现准多项式或亚指数时间。该改进通过对树分解的“签名”进行压缩，减少了状态空间到仅 k^{k}，并在动态规划中使用类型等价性。

**🔧 技术方法**

核心技术包括：
- 通过 MSO 逻辑（Counting MSO）描述问题；
- 使用树分解的签名（signature）和类型（type）对子图进行分类；
- 对每个节点（introduce/forget/join）使用动态规划并通过等价类压缩状态；
- 计算树独立数的树分解（可在 2^{k^2} n^k 时间内得到）。

**📊 数据集**

该工作主要为理论算法，未使用具体实验数据集；评估通过对已知图类（3PC‑free、even‑hole‑free、S_{q,q,q} 与 L_q‑free、map graphs、pseudodisk 与 contact segment graphs）进行理论比较。

**📈 对比分析**

相较于 Lima 等人的原始算法，新的算法将时间复杂度从指数提升到 n^{O(k)}；在树独立数为 log^{O(1)} n 的图类上得到 n^{O(log^c n)} 的准多项式时间；在具有子线性平衡 clique‑separators 的几何交叉图类（如凸 fat 对象、map、pseudodisk、contact segment）上实现 2^{√n log n} 或 2^{n^{2/3} log n} 的亚指数时间。

**⚠️ 局限性**

限制包括：
- 需要先获得树分解；若仅用 2^{k^2} n^k 的方法，k 接近 √n 时仍显得昂贵；
- 仅适用于树独立数受限的图类，无法直接推广到更广泛的参数（如诱导匹配树宽）；
- 对于 k 超常数大（如多项式或指数级）时，算法仍然不具可行性。

---

## 255. Uncertainty-guided Generation of Dark-field Radiographs

**arXiv ID:** 2601.15859 | [PDF](https://arxiv.org/pdf/2601.15859v1)

**作者:** Lina Felsner `[一作]` (Technical University of Munich), Julia Schnabel `[通讯]` (King’s College)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于不确定性引导的渐进式生成对抗网络（Uncertainty‑Guided Progressive GAN），能够将常规胸部X光衰减影像直接合成对应的X光暗场影像。

**💡 创新点**

①首次实现从二维衰减X光到暗场影像的跨模态生成；②在生成过程中同时建模齐性不确定性（aleatoric）和模型不确定性（epistemic），提高可解释性与可靠性；③采用渐进式学习与不确定性注意力机制，实现逐步细化与质量提升。

**🔧 技术方法**

使用渐进式生成对抗网络、蒙特卡洛 dropout 进行模型不确定性估计、Generalized Gaussian 分布参数化（α、β）建模齐性不确定性、残差一致性损失、数据增强、以及传统的 MSE/PSNR/SSIM 评价指标。

**📊 数据集**

内部使用来自 Klinikum rechts der Isar 的269名患者的衰减/暗场配对数据集（训练227/验证15/测试27），并在 NIH Chest X‑ray 数据集上进行域外评估，检验模型对未知设备/解剖结构的泛化能力。

**📈 对比分析**

通过分阶段（Stage 1–3）训练与评估，发现指标随阶段提升：MSE 从0.0131降至0.0123，PSNR 从19.35提升至19.71，SSIM 从0.38提升至0.52。与真实暗场图像对比，生成图像结构和灰度分布高度一致；在域外数据上，模型能生成合理暗场并在不确定性图中显著标记未知区域，表明具备一定的自检功能。

**⚠️ 局限性**

主要局限包括：跨模态转换导致 SSIM 仅达到约0.5 的中等水平；模型未能捕捉暗场图像的水平条纹伪影；数据集规模有限，限制了模型在不同疾病分级和噪声水平下的泛化；当前仅使用GAN，未尝试扩散模型；在极端对比或含罕见设备的图像中可能失效。

---

## 256. Can professional translators identify machine-generated text?

**arXiv ID:** 2601.15828 | [PDF](https://arxiv.org/pdf/2601.15828v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 257. TinySense: Effective CSI Compression for Scalable and Accurate Wi-Fi Sensing

**arXiv ID:** 2601.15838 | [PDF](https://arxiv.org/pdf/2601.15838v1)

**作者:** Toan Gian `[一作]` (Smart Green Transformation Center), Van-Dinh Nguyen `[通讯]` (Smart Green Transformation Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了名为TinySense的Wi‑Fi信道状态信息（CSI）压缩框架，用于实现高效、可扩展的无设备人类姿态估计（HPE）

**💡 创新点**

创新点在于将VQGAN与K‑means自适应码本结合，并在第二阶段加入Transformer来预测丢失的量化索引，从而在极低比特率下仍保持高压缩质量和HPE精度

**🔧 技术方法**

主要技术包括向量量化生成对抗网络（VQGAN）、K‑means聚类码本重构、Transformer索引恢复、感知损失与GAN对抗损失以及关键点回归损失

**📊 数据集**

实验使用MM‑Fi和Wi‑Pose两个公开Wi‑Fi姿态数据集，分别包含多动作和多姿态的CSI与姿态标签

**📈 对比分析**

与六种SOTA压缩/感知方法（如EfficientFi、RSCNet、MetaFi++等）对比，TinySense在相同压缩率下实现了1.25–1.5倍的PCK_20提升，同时将网络开销和推理时延分别降低5×–14×

**⚠️ 局限性**

局限性包括当前仅针对单人场景，需在多用户或跨域场景下进行源分离与少样本适配；Transformer的恢复性能在极高丢包率（>90%）时仍显著下降

---

## 258. RF Intelligence for Health: Classification of SmartBAN Signals in overcrowded ISM band

**arXiv ID:** 2601.15836 | [PDF](https://arxiv.org/pdf/2601.15836v1)

**作者:** Nicola Gallucci `[一作]` (Politecnico di Milano), Lorenzo Mucchi `[通讯]` (Università di Firenze)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出并实现了一个开源深度学习框架，用于在拥挤的2.4 GHz ISM频段中自动识别SmartBAN低功耗医疗信号，并通过合成与真实SDR捕获的混合数据集进行训练与评估。

**💡 创新点**

创新点包括：① 结合合成与实测数据的混合评估体系；② 在U‑Net解码器中引入注意力门以提升弱信号分割精度；③ 采用优先级标注策略解决频域重叠的多类标注问题；④ 将框架公开发布，促进社区复现与扩展。

**🔧 技术方法**

使用的技术主要有：深度卷积网络（ResNet‑18/50、DeepLab v3+）+带注意力的U‑Net；短时傅里叶变换生成频谱图；加权像素交叉熵损失；Adam优化器与学习率衰减；SDR（ADALM‑Pluto）实测捕获。

**📊 数据集**

数据集包括：1）25 000张合成频谱图，涵盖Wi‑Fi、蓝牙、ZigBee、SmartBAN四类信号及其随机混合；2）由SDR在真实环境中捕获的80 MHz ISM频段频谱图，用于验证泛化能力。

**📈 对比分析**

方法通过在合成数据上与真实数据上分别计算准确率、IoU、Dice和F1得分进行比较。三种网络在合成数据上均取得>90 %总体准确率，ResNet‑50在不同距离、干扰强度下表现最稳健；在实测数据中准确率相对降低但保持稳定，证明框架在真实环境中具备可接受的性能。

**⚠️ 局限性**

局限性包括：① 实测数据受硬件带宽与同步误差限制，导致准确率低于合成场景；② 仅覆盖四类信号，无法直接扩展到更多协议；③ 采用固定的STFT窗口与频谱预处理，对非稳态或极端噪声环境适应性有限；④ 缺乏在线自适应学习机制，难以实时应对动态干扰变化。

---

## 259. An IoT-Based Smart Plant Monitoring and Irrigation System with Real-Time Environmental Sensing, Automated Alerts, and Cloud Analytics

**arXiv ID:** 2601.15830 | [PDF](https://arxiv.org/pdf/2601.15830v1)

**作者:** Abdul Hasib `[一作]` (University of Frontier Technology), A. S. M. Ahsanul Sarkar Akib `[通讯]` (Robo Tech Valley)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了基于 ESP32 的 IoT 智能植物监测系统，集成多传感器（温湿度、土壤湿度、水位、养分温度）、OLED 本地显示、蜂鸣器与 RGB LED 警报，并通过 ThingSpeak 云端实现实时数据上传、历史分析与自动化灌溉控制。

**💡 创新点**

创新点在于：①多模态传感器融合与本地多渠道报警（视觉+听觉+云端），②自适应采样与深度睡眠实现低功耗；③可扩展的 8 节点 ESP32 架构与 45 美元级低成本实现；④通过阈值驱动的自动灌溉实现 40% 水量节约，同时保持土壤湿度在最佳范围 92% 的时间。

**🔧 技术方法**

主要技术包括 ESP32 微控制器、DHT22、土壤湿度传感器、HC‑SR04 超声波水位传感器、DS18B20 水温传感器、0.96" OLED、蜂鸣器、RGB LED、5V 继电器、Arduino C++ 固件、ThingSpeak API、Chart.js 的 Web Dashboard 以及深度睡眠与自适应采样算法。

**📊 数据集**

实验数据来自 30 天现场实测的温湿度、土壤湿度、水位等传感器读数，并与手工灌溉和固定计时器灌溉的水量与植物生长数据进行对比；未使用公开数据集，所有评估均基于自建实验数据。

**📈 对比分析**

通过与手工灌溉和定时器灌溉比较，系统实现了 40% 的水量节约、7% 的植物生长提升；土壤湿度保持最佳区间 92% 的时间；数据上传成功率 99.7%；平均响应时间 1.3 秒；在 10 台设备并行监测时显示线性可扩展性，无性能下降。

**⚠️ 局限性**

局限性包括：①传感器需要每 6 个月或不同土壤类型进行重新校准；②对 Wi‑Fi 网络稳定性有依赖，离线期间只能本地存储；③功耗虽然低，但仍需可靠电源或太阳能补充；④单 ESP32 限制为最多 8 节点，规模化需引入网状网络或网关；⑤极端天气（暴雨、霜冻）可能影响传感器精度；⑥缺乏 LoRa 或蜂窝备份，远程地区可用性受限。

---

## 260. Unveiling and Simulating Short-Video Addiction Behaviors via Economic Addiction Theory

**arXiv ID:** 2601.15975 | [PDF](https://arxiv.org/pdf/2601.15975v1)

**作者:** Chen Xu `[一作]` (Renmin University of China), Maarten de Rijke `[通讯]` (University of Amsterdam)

**通讯引用:** 29323 | [OpenAlex ID](https://openalex.org/A5031439294)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过将经济上瘾理论与短视频平台行为数据结合，构建了短视频上瘾模型并设计了 AddictSim 模拟器，能够学习并模拟用户的上瘾行为。

**💡 创新点**

创新点包括：①将经济上瘾理论映射到短视频观看场景；②提出两阶段 Mean‑to‑Adapted (M2A) 与 Group Relative Policy Optimization (GRPO) 的训练框架，兼顾全局平均与个体化上瘾模式；③利用多样化重排序算法验证上瘾缓解机制。

**🔧 技术方法**

使用技术包括：经济上瘾模型、NFM 推荐模型、回归参数估计、LoRA 微调、PPO/GRPO 强化学习、奖励分配与 M2A 适配、离散/连续时间奖励分配。

**📊 数据集**

采用的公开数据集为 THU 与 KuaiRec 两大短视频行为数据集。

**📈 对比分析**

在 MAE/RMSE 指标上与 Base、SFT、PPO、GRPO 四种基线对比，AddictSim 在会话层和视频层均实现 10%~20% 的误差降低，且性能显著优于最优基线。

**⚠️ 局限性**

局限性包括：仅针对短视频场景，缺乏跨平台或跨内容类型的普适性；模型参数假设固定，未能完整捕捉长时序动态与个体差异；未充分讨论用户隐私与伦理风险。

---

## 261. EVolSplat4D: Efficient Volume-based Gaussian Splatting for 4D Urban Scene Synthesis

**arXiv ID:** 2601.15951 | [PDF](https://arxiv.org/pdf/2601.15951v1)

**作者:** Sheng Miao `[一作]`, Yiyi Liao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了EvolSplat4D，一种面向动态城市场景的前向式3D高斯散射框架，将场景拆分为近距离体积、动态行人/车辆以及远景三部分，实现一次性高质量重建。

**💡 创新点**

核心创新包括：① 体积化高斯预测与像素化高斯分支的混合设计，解决了传统像素化方法在近距离产生冗余与不一致的问题；② 运动校正的图像渲染(IBR)模块，可在对象本体空间中对时间进行精细对齐，稳健获取外观信息；③ 基于DINO特征的遮挡判定与颜色窗口采样，大幅提升纹理细节与一致性；④ 通过LiDAR或单目深度估计构造全局语义体素，实现几何一致性。

**🔧 技术方法**

使用技术包括3D高斯散射（3DGS）渲染、稀疏3D卷积网络预测体积特征、DINOv2语义特征提取、基于交叉视角注意力的2D U‑Net预测远景高斯、遮挡敏感的窗口化采样、三层MLP颜色编码、以及α‑混合组合三分支结果。

**📊 数据集**

在KITTI‑360、KITTI、Waymo Open Dataset以及PandaSet上进行训练与评估，涵盖静态与动态场景、不同视角稀疏度，并在这些数据集上实现零样本与交叉数据集的泛化。

**📈 对比分析**

与传统的每场景优化方法（如SUDS、StreetGaussian、OmniRe、EmerNeRF）以及前向基线（MVSNeRF、MuRF、EDUS、MVSplat、PixelSplat、DepthSplat、AnySplat、STORM、DrivingRecon）对比，EvolSplat4D在PSNR、SSIM、LPIPS、KID等指标上均优于对手，同时推理速度≈1.3 s，显存≈10.98 GB，FPS在实机上可达数十帧，显著提升了效率与视觉质量。

**⚠️ 局限性**

局限性包括：① 依赖外部LiDAR或高质量3D检测框进行对象本体化，虽然可用单目深度替代但仍需检测框；② 仅假设对象做刚体运动，对非刚体或行人等柔性动态难以精准重建；③ 远景分支在大基线外推时性能下降，需更强的深度先验或监督；④ 目前未整合端到端的视觉检测与跟踪模块，未来需进一步融合。

---

## 262. ICON: Invariant Counterfactual Optimization with Neuro-Symbolic Priors for Text-Based Person Search

**arXiv ID:** 2601.15931 | [PDF](https://arxiv.org/pdf/2601.15931v1)

**作者:** Xiangyu Wang `[一作]` (Northeastern University), Hangxu Ji `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了ICON框架，主动干预文本驱动人检索中的定位误差、背景干扰和显著性偏差，显著提升鲁棒性和检索精度。

**💡 创新点**

创新点在于将规则引导的空间干预、反事实背景置换、显著性驱动语义正则化和不确定性加权对齐四个因果与拓扑先验模块融合进视觉‑语言对齐，从而突破被动观察短路，实现因果可解释的鲁棒检索。

**🔧 技术方法**

采用神经符号对抗式定位、基于注意力的前景/背景分离与最优传输置换、跨模态对抗重构、实例不确定性加权、CLIP/ViT编码器以及多层对比学习损失。

**📊 数据集**

在PRW‑TBPS和CUHK‑SYSU‑TBPS两大文本检索基准上进行实验。

**📈 对比分析**

与OIM、NAE、CLIP‑TBPS、ViPer等多种基线对比，ICON在mAP、Top‑1/T5/T10上均优于最新ViPer，mAP提升约0.9–3.5分，Top‑1提升2.5–3.7分，证明更高的鲁棒性和检索性能。

**⚠️ 局限性**

局限在于仍需依赖大规模预训练视觉‑语言模型，计算量和显存占用较高，且在极端遮挡、极低光照等极端场景下的鲁棒性尚有提升空间。

---

## 263. MMGRid: Navigating Temporal-aware and Cross-domain Generative Recommendation via Model Merging

**arXiv ID:** 2601.15930 | [PDF](https://arxiv.org/pdf/2601.15930v1)

**作者:** Tianjun Wei `[一作]` (Nanyang Technological University), Zhu Sun `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 2048 | [OpenAlex ID](https://openalex.org/A5033957641)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了基于多域、多时段的生成式推荐模型检查点网格，系统研究模型合并（MM）在生成推荐中的效果与挑战。

**💡 创新点**

首次从跨域和时序上下文角度系统探讨MM，提出基于历史检查点的中性基模型、任务向量重缩放以及通过域特征预测合并权重等策略。

**🔧 技术方法**

采用基于Qwen3-0.6B的生成式推荐三种范式（文本、语义ID、语义嵌入），并实现权重合并与子空间合并（DARE、TIES）及任务向量算子。

**📊 数据集**

使用亚马逊多品类评论数据集，将不同商品类别视为域，并按时间划分为四个阶段进行预训练与增量训练。

**📈 对比分析**

在单域基线上与跨域/时序合并模型进行对比，发现文本根基范式可提升性能，语义ID/嵌入在跨域合并时易下降，但通过中性基模型和权重调优可恢复；时序合并可通过预测合并权重提升≈1–3%的R@10/20。

**⚠️ 局限性**

合并后模型仍未完全匹配单域性能，尤其大域差异明显；当前方法仅考虑线性任务向量和简单域特征，未捕获复杂非线性关系，且对小数据域的鲁棒性不足。

---

## 264. A Remark on Downlink Massive Random Access

**arXiv ID:** 2601.15928 | [PDF](https://arxiv.org/pdf/2601.15928v1)

**作者:** Yuchen Liao `[一作]` (University of Science and Technology of China), Wenyi Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8230 | [OpenAlex ID](https://openalex.org/A5100360013)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出利用覆盖数组构造确定性下行大规模随机接入码书，证明可将传输开销降低到与总用户数无关的常数级。

**💡 创新点**

创新点在于将覆盖数组理论应用于DMRA，构造确定性码书而非随机编码，并给出1+log₂e位的开销上界。

**🔧 技术方法**

采用覆盖数组、贪心构造、几何分布熵分析以及Shannon/Huffman可变长度编码技术。

**📊 数据集**

本文为理论分析与模拟实验，未使用实际数据集，实验采用人工生成的二进制码书。

**📈 对比分析**

通过与显式用户编码（k log₂n 位）对比，实验表明在两、三用户激活时，开销仅比信息本身多约1–1.2位，且随总用户数几乎不变。

**⚠️ 局限性**

局限性在于贪心构造计算复杂、存储成本高，且缺乏结构化特性，难以扩展到更大规模的 n 或更高字母表大小 q。

---

## 265. A Multi-View Pipeline and Benchmark Dataset for 3D Hand Pose Estimation in Surgery

**arXiv ID:** 2601.15918 | [PDF](https://arxiv.org/pdf/2601.15918v1)

**作者:** Valery Fischer `[一作]` (ETH Zurich), Lilian Calvet `[通讯]` (University Hospital Balgrist University of Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无需领域特定训练的多视角3D手部姿态估计管线，集成检测、跟踪、精细化关键点预测与三角化+约束优化；

**💡 创新点**

创新点在于结合YOLOv11+Efficient Track Anything的鲁棒检测与跟踪，利用高分辨率手部关键点模型精化结果，并通过三角化与多项几何/形状约束实现高精度3D重建，同时发布首个手术室多视角标注数据集；

**🔧 技术方法**

使用YOLOv11检测、Efficient Track Anything跟踪、全身姿态估计、专用2D手部关键点模型、三角化、L‑BFGS‑B优化、重投影损失、平滑损失、形状一致性损失等技术；

**📊 数据集**

采用新构建的手术室多视角数据集（约68k帧，3000个手部2D标注及3D三角化真值），并与公开数据集进行基线对比；

**📈 对比分析**

与RTMPose、Sapiens、DWPose等基线相比，2D MJE降低约30%，mPCK提升约3点；3D MPJPE下降31%，mPCK_3D提升76%，显示显著优于现有方法；

**⚠️ 局限性**

仅针对单人/少量动态场景，面对多人人/高遮挡时表现欠佳；推理速度慢（约30–45分钟/300帧），未实现实时。

---

## 266. The Latency Wall: Benchmarking Off-the-Shelf Emotion Recognition for Real-Time Virtual Avatars

**arXiv ID:** 2601.15914 | [PDF](https://arxiv.org/pdf/2601.15914v1)

**作者:** Yarin Benyamin `[一作]` `[通讯]` (Ben Gurion University of the Negev), Yarin Benyamin (Ben Gurion University of the Negev)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对现成深度学习模型在虚拟角色面部情绪识别中的实时性和准确性进行基准评估

**💡 创新点**

揭示了在CPU环境下通用Transformer在情绪分类上存在的“延迟墙”，并指出需要专用轻量化网络

**🔧 技术方法**

使用YOLO（v8/v11/v12）进行面部检测，CLIP、SigLIP和ViT-FER进行零样本情绪分类

**📊 数据集**

利用UIBVFED（虚拟表情）和FER‑2013（真实人脸）两组数据集

**📈 对比分析**

通过CPU单机测试，检测准确率100%但分类层延迟最高达1.7s，ViT‑FER在虚拟域仅27.4%准确率，无法满足140ms阈值

**⚠️ 局限性**

受限于计算资源，现有SOTA模型无法满足实时情绪识别需求，需进行模型蒸馏与量化以实现可部署的轻量化解决方案

---

## 267. ThermoSplat: Cross-Modal 3D Gaussian Splatting with Feature Modulation and Geometry Decoupling

**arXiv ID:** 2601.15897 | [PDF](https://arxiv.org/pdf/2601.15897v1)

**作者:** Zhaoqi Su `[一作]` (Fuzhou University), Xiaoqiang Lu `[通讯]` (Fuzhou University)

**通讯引用:** 11631 | [OpenAlex ID](https://openalex.org/A5018824735)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种跨模态 3D 高斯散点渲染框架 ThermoSplat，用于 RGB 与热红外场景的高保真重建。

**💡 创新点**

创新点包括：① 跨模态 FiLM 调制机制，利用热红外结构先验主动调节共享特征；② 模态自适应几何解耦，热红外分离光照和几何参数；③ 混合显式-隐式渲染，将 Spherical Harmonics 与神经解码相结合，以获得高频细节和语义一致。

**🔧 技术方法**

核心技术：3D Gaussian Splatting（显式几何 + 多维特征），FiLM 特征调制，独立几何偏移，混合渲染管线（SH + 线性神经解码），综合像素、结构、特征一致与光滑损失。

**📊 数据集**

使用 RGBT‑Scenes 数据集，该数据集包含数千对 RGB‑热红外的室内外场景。

**📈 对比分析**

与 3DGS、ThermalGaussian、MS‑Splattingv2、MMOne 等方法进行对比，评估指标为 PSNR、SSIM、LPIPS。实验显示 ThermoSplat 在 RGB 与热红外两模态上均获得最高或接近最高分，特别在细节重现和结构一致性上优于同类方法。

**⚠️ 局限性**

局限性：几何解耦主要针对热红外分支，对高透明或玻璃等复杂相互作用处理不足；FiLM 调制增加显存占用；目前框架仅验证在 RGB‑热红外上，扩展到近红外或高光谱场景仍需进一步研究。

---

## 268. RadJEPA: Radiology Encoder for Chest X-Rays via Joint Embedding Predictive Architecture

**arXiv ID:** 2601.15891 | [PDF](https://arxiv.org/pdf/2601.15891v1)

**作者:** Anas Anwarul Haq Khan `[一作]` (Indian Institute of Technology Bombay), Kshitij Jadhav `[通讯]` (Indian Institute of Technology Bombay)

**通讯引用:** 313 | [OpenAlex ID](https://openalex.org/A5103888294)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种自监督框架RadJEPA，用于在不依赖图像-文本对的情况下学习胸部X光图像的表示。

**💡 创新点**

创新点在于采用联合嵌入预测（JEPA）方法，直接在潜在空间预测遮蔽区域的表示，摆脱了对文本监督和传统视图一致性对齐的依赖。

**🔧 技术方法**

技术包括：ViT-B/14编码器、预测器网络g、EMA目标编码器、线性探测器、UPerNet分割解码器以及LLaVA风格的两层投影器与Vicuna-7B语言模型进行报告生成。

**📊 数据集**

预训练数据使用839,364张无标注胸部X光（BRAX、CheXpert、MIMIC‑CXR、ChestX‑ray14、PadChest），下游评测使用VinDr‑CXR、RSNA‑Pneumonia、MIMIC‑CXR报告、IU‑Xray等公开数据集。

**📈 对比分析**

与多种基线（CLIP、BiomedCLIP、MRM、CheXzero、Rad‑DINO、I‑JEPA）进行对比，RadJEPA在疾病分类（AUPRC提升约3–4点）、语义分割（Dice提升约4点）以及报告生成（ROUGE‑L提升约1–2点）等任务上均实现了显著优于SOTA的性能。

**⚠️ 局限性**

局限性：仅使用ViT‑B/14架构；未探讨更大或多尺度骨干网络；未评估其他自监督方法（如DINO‑v2）在同一任务上的表现；对不同机构数据分布的鲁棒性仍待进一步验证。

---

## 269. EvoCUA: Evolving Computer Use Agents via Learning from Scalable Synthetic Experience

**arXiv ID:** 2601.15876 | [PDF](https://arxiv.org/pdf/2601.15876v1)

**作者:** Taofeng Xue `[一作]` (Meituan), Xipeng Qiu `[通讯]` (Fudan University)

**通讯引用:** 17130 | [OpenAlex ID](https://openalex.org/A5044665993)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于“从经验学习的演化范式”的本地计算机使用代理 EvoCUA，能够通过可验证的任务合成、海量异步沙箱交互以及迭代优化自我提升。

**💡 创新点**

创新点在于（1）可验证合成引擎一次性生成指令与可执行验证器，消除任务误报；（2）构建大规模无阻塞沙箱基础设施，实现数万并发交互；（3）演化学习循环，将成功轨迹转为正样本，失败轨迹转为偏好对，利用 step‑level DPO 进行细粒度调优。

**🔧 技术方法**

采用 Qwen3‑VL / OpenCUA 作为基础模型，结合 QEMU‑KVM 虚拟化沙箱、异步 Gateway‑Scheduler、可验证合成、step‑level DPO、拒绝采样微调（RFT）以及强化学习中的偏好优化。

**📊 数据集**

数据集包括由可验证合成引擎生成的数万条自带验证器的任务、OSWorld benchmark 任务与评测脚本，以及公开的通用多模任务集（MMMU、ScreenSpot、MathVista 等）。

**📈 对比分析**

在 OSWorld‑Verified 基准上，EvoCUA‑32B 达到 56.7% 的成功率，显著优于前沿开源模型 OpenCUA‑72B（45.0%）和 UI‑TARS‑2（53.1%），并在 50 步预算下保持竞争力；在通用 VLM 任务上保持与基线相当或略低，体现了演化学习对专用任务的高效提升。

**⚠️ 局限性**

局限在于仍低于闭源高权重模型，依赖大量离线合成经验，训练‑推理不匹配导致高成本，通用能力在部分任务上略有下降，且需要进一步引入在线强化学习以逼近人类级可靠性。

---

## 270. Practical applications of Set Shaping Theory to Non-Uniform Sequences

**arXiv ID:** 2601.15853 | [PDF](https://arxiv.org/pdf/2601.15853v1)

**作者:** A. Schmidt `[一作]`, A. Petit `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过实验验证了集合塑形理论（Set Shaping Theory, SST）在非均匀序列上的有效性，并提出了一种近似信息量排序算法以降低计算复杂度；

**💡 创新点**

创新点在于将近似熵排序方法引入SST，实现了从指数级到线性级（O(N)）的复杂度提升，同时证明对非均匀数据仍能获得压缩收益；

**🔧 技术方法**

使用的技术包括零阶熵计算、信息量（熵×长度）近似排序、集合塑形映射函数、以及基于Python的实验软件实现；

**📊 数据集**

实验数据为随机生成的长度为400、符号数30-60、最大符号概率0.5的非均匀序列，均按指定概率分布采样；

**📈 对比分析**

通过比较原序列N·H0(s)与变换后序列(N+1)·H0(f(s))，统计成功率P_s>88%，并测得平均比特减少随符号数增加而提升，实验结果与理论预测一致；

**⚠️ 局限性**

局限性在于仍需近似排序，最优映射尚未实现；对极大字母表或极低概率符号的表现尚不明确；依赖于随机序列生成，缺乏对真实数据集的验证。

---

## 271. Natural Language-Driven Global Mapping of Martian Landforms

**arXiv ID:** 2601.15949 | [PDF](https://arxiv.org/pdf/2601.15949v1)

**作者:** Yiran Wang `[一作]` (Southern University of Science and Technology), Hongxin Wei `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 1035 | [OpenAlex ID](https://openalex.org/A5020027500)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个名为MarScope的星球尺度视觉-语言框架，能够通过自然语言或图像查询即时检索并全局绘制火星地貌分布，支持过程导向和稀有地貌的发现。

**💡 创新点**

创新点在于：①将海量像素级遥感数据映射到共享语义空间，实现无标签、零样本的全星球地貌检索；②提供文本、图像及多模态三种查询模式，并通过高效索引实现秒级响应；③利用语义检索实现基于成因而非形态的地貌映射，突破传统分类限制。

**🔧 技术方法**

使用对比视觉-语言学习模型（Contrastive Vision–Language Encoder）配合FAISS近似最近邻检索，结合文本编码器和图像编码器的嵌入空间进行语义匹配；同时采用LLM辅助数据清洗与增强，提升训练语料的多样性。

**📊 数据集**

训练数据超过20万组高质量图文对，涵盖火星、月球、水星及冰卫星，图像来源于HiRISE、CTX等高分辨率轨道数据；测试使用六类已公布的火星全球地貌目录（冲积扇、冰雪形态、滑坡、凹陷锥、尘带、暗坡条纹）进行检索验证。

**📈 对比分析**

与已发布的地貌目录进行匹配评价，采用Precision@K、Recall@K和F1@K指标。MarScope在不同查询模式下达到最高F1≈0.978，检索时间仅约5秒，显著优于传统人工或监督学习方法的精度与效率。

**⚠️ 局限性**

局限性包括：①无法提供像素级边界或精确定位，仅能检测图块内是否存在目标；②固定的0.2°和0.02°图块大小导致对极大或极小地貌的识别受限；③性能高度依赖训练数据的覆盖范围与多样性，若缺乏某类地貌或光照条件，检索准确性可能下降。

---

## 272. Accurate Calibration and Robust LiDAR-Inertial Odometry for Spinning Actuated LiDAR Systems

**arXiv ID:** 2601.15946 | [PDF](https://arxiv.org/pdf/2601.15946v1)

**作者:** Zijie Chen `[一作]` (Guangdong University of Technology), Lihua Xie `[通讯]` (Nanyang Technological University)

**通讯引用:** 54014 | [OpenAlex ID](https://openalex.org/A5100365448)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了基于Denavit‑Hartenberg模型的无目标LiDAR‑电机系统标定方法LM‑Calibr以及考虑激光点不确定性与环境规模自适应的鲁棒LiDAR‑惯性里程计EVA‑LIO；

**💡 创新点**

创新点在于：①使用DH参数统一描述所有旋转式LiDAR‑电机挂载；②采用层级分辨率与自适应下采样实现环境适配；③对激光点做运动不确定性建模提升特征匹配鲁棒性；

**🔧 技术方法**

核心技术包括：目标无标定的厚度最小化优化、Levenberg‑Marquardt算法、平面特征点自适应体素化、点云运动补偿与点对平面残差MAP估计；

**📊 数据集**

实验使用MARSIM仿真生成的Mid360与Avia数据集，以及真实世界的洞穴、长廊、花园、校园等多种场景；

**📈 对比分析**

与Voxel‑SLAM、Point‑LIO、Fast‑LIO2、Ada‑LIO、CTE‑MLO、Traj‑LO等SOTA方法对比，EVA‑LIO在多种场景下实现了最低ATE且在特征稀疏区保持稳定；

**⚠️ 局限性**

局限性包括：对点云分布依赖较高，单平面场景下可识别性下降；在高速动态（≥6g）情况下运动补偿失效导致定位漂移；特殊挂载极端角度仍存在观测不完全问题。

---

## 273. Pregroup representable expansions of residuated lattices

**arXiv ID:** 2601.15905 | [PDF](https://arxiv.org/pdf/2601.15905v1)

**作者:** Andrew Craig `[一作]` (University of Johannesburg), Claudette Robinson `[通讯]` (National Institute for Theoretical and Computational Sciences)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

论文通过构造预群（pregroup）及其上升集代数，证明了一类分布式涉及到的 involutive FL‑algebras（DInFL-algebras）与分布式准关系代数（DqRAs）可以用二元关系的形式被表示，并给出了相应的嵌入映射；进一步引入正交预群（ortho pregroup）实现 DqRA 的表示，举例说明该方法在从群的乘积构造的代数中可行。

**💡 创新点**

创新点在于：① 将传统的群可表示关系代数推广到更一般的预群结构；② 通过上升集与二元关系的双层嵌入，得到新的可表示性条件；③ 引入正交预群和自反运算，完成 DqRA 的完整表示；④ 给出具体例子（来自群的乘积）证明方法的有效性。

**🔧 技术方法**

主要技术手段包括：
- 预群与 ipo‑monoid 的代数定义与性质推导；
- 上升集（up‑set）构造的分布式可积化环（distributive residuated lattice）;
- 通过映射 σ 把上升集嵌入到二元关系集上的结构；
- 利用秩逆（order‑reversing）运算、对偶运算以及自反性证明可表示性。

**📊 数据集**

无数据集，研究完全为理论代数构造与证明。

**📈 对比分析**

本文不涉及实验对比或性能评估，因其为纯理论数学论文；主要通过结构嵌入与可表示性证明展示方法效果。

**⚠️ 局限性**

局限性包括：
- 仍未找到非可表示的 DInFL‑algebra 或 DqRA 的例子；
- 对更广泛的预群结构（如部分定义的预群体）与非循环运算的可表示性讨论仍开放；
- 现有方法主要处理分布式、可积化结构，可能不适用于更一般的非分布式或非可积化情形。

---

## 274. Stable-DiffCoder: Pushing the Frontier of Code Diffusion Large Language Model

**arXiv ID:** 2601.15892 | [PDF](https://arxiv.org/pdf/2601.15892v1)

**作者:** Chenghao Fan `[一作]` (Huazhong University of Science and Technology), Wei Wei `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 244234 | [OpenAlex ID](https://openalex.org/A5100352881)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Seed-Coder基础上加入块级扩散训练，构建了一种面向代码的扩散语言模型；

**💡 创新点**

创新点在于利用扩散式随机遮蔽与去噪训练提升模型对稀有高质量代码的泛化，且通过块级扩散保持上下文完整，实现高效并行解码；

**🔧 技术方法**

使用了块级Diffusion Language Model（DLLM）框架、无对数位移(no-logit-shift)训练、块级噪声衰减调度、持续预训练以及SFT细调；

**📊 数据集**

训练数据来自Seed-Coder的1.3T预训练语料与相同的SFT数据集，评估数据覆盖HumanEval、MBPP、CRUXEval、MultiPL‑E、MHPP、BigCodeBench、LiveCodeBench、MBXP、NaturalCodeBench、CanItEdit与Aider等多样化代码基准；

**📈 对比分析**

与同规模AR基线（如Seed‑Coder、StarCoder2、Qwen2.5‑Coder等）及其他扩散模型（LLaDA、DiffuCoder等）比较，模型在大多数基准上均超越AR对手，部分任务（如HumanEval+、MBPP+、多语种生成）甚至达到或超过同类8B扩散模型的最高水平；

**⚠️ 局限性**

局限性包括仅针对代码任务，数学推理与通用文本能力有限；模型规模受限于8B，未探究更大规模扩散模型的进一步优势；

---

## 275. Understanding the Transfer Limits of Vision Foundation Models

**arXiv ID:** 2601.15888 | [PDF](https://arxiv.org/pdf/2601.15888v1)

**作者:** Shiqi Huang `[一作]` (University College London), Yipeng Hu `[通讯]` (University College London)

**通讯引用:** 5059 | [OpenAlex ID](https://openalex.org/A5032309114)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文研究了视觉基础模型（VFM）的预训练目标与下游前列腺多参数MRI任务的对齐对迁移性能的影响，并在两种模型ProFound（MAE预训练）和ProViCNet（DINOv2对比学习）上进行了实验。

**💡 创新点**

创新点在于提出用最大均方差（MMD）度量预训练与下游任务的分布差异，并将这种对齐度与迁移增益（RPG）建立负相关关系，揭示对齐度是决定迁移效果的关键因素。

**🔧 技术方法**

采用MAE自编码与DINOv2对比学习预训练的VFM，并使用MMD、RPG、GPU时数等指标评估迁移效果，进一步跟随细化的fine‑tune策略进行实验。

**📊 数据集**

使用PROMIS前列腺多参数MRI数据集以及PI‑CAI等私有前列腺影像数据，涵盖PI‑RADS分类、病灶分割、超分辨、畸变校正与模态转换五个任务。

**📈 对比分析**

与随机权重模型、从零训练模型以及专门任务模型对比，结果显示预训练模型在性能提升（RPG）和训练时间上均优于从零训练，且对齐度越低迁移增益越大。

**⚠️ 局限性**

研究局限于前列腺MRI领域，预训练目标与任务对齐的机制尚未在多模态或跨病种任务中验证，且未探索自适应或任务无关的预训练策略。

---

## 276. Out-of-Distribution Detection Based on Total Variation Estimation

**arXiv ID:** 2601.15867 | [PDF](https://arxiv.org/pdf/2601.15867v1)

**作者:** Dabiao Ma `[一作]` (Qifu Technology), Haojun Fei `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于总变差（Total Variation）度量的OOD检测方法TV‑OOD，利用改进的MINE网络估计输入对总变差的贡献，形成评分函数；

**💡 创新点**

创新点在于将MINE的KL散度改为总变差距离，获得更线性、无偏的估计，并引入辅助OOV数据集D_aug及TVNE网络，显著提升ID/OOD区分；

**🔧 技术方法**

核心技术包括：改进的MINE（TVNE）实现总变差估计；使用辅助OOV数据（D_aug）通过图像子块重排生成；与基线方法对齐的阈值设定；

**📊 数据集**

实验使用CIFAR‑100和ImageNet‑1k作为ID数据，分别采用DenseNet121、WideResNet、ViT‑B_16三种模型；OOD测试集包括Textures、SVHN、Places365、LSUN、iSUN、DTD、iNaturalist、SUN等；

**📈 对比分析**

与多种基线（MSP、Energy、ODIN、Mahalanobis、LogitNorm、KNN、ASH‑S、VIM、OE、EF、WOODS、BEOE）比较，TV‑OOD在FPR95、AUROC、AUPR指标上常常匹配或优于对手；

**⚠️ 局限性**

局限性：对辅助OOV数据集的生成方式依赖较大，生成效果不稳定；理论上TV与其他f‑divergence的差异尚未完全阐明，未来仍需进一步验证。

---

## 277. A Lightweight Brain-Inspired Machine Learning Framework for Coronary Angiography: Hybrid Neural Representation and Robust Learning Strategies

**arXiv ID:** 2601.15865 | [PDF](https://arxiv.org/pdf/2601.15865v1)

**作者:** Jingsong Xia `[一作]` (Nanjing Medical University), Siqi Wang `[通讯]` (Nanjing Medical University)

**通讯引用:** 3795 | [OpenAlex ID](https://openalex.org/A5100420069)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种轻量化脑启发式机器学习框架，用于冠脉造影图像的二分类。

**💡 创新点**

创新在于结合选择性神经可塑性训练、基于焦点损失的注意力调制和标签平滑，既保持模型轻量化又提升鲁棒性。

**🔧 技术方法**

使用预训练ResNet50作为特征提取器，添加线性分类头，采用Focal Loss+标签平滑的损失函数，并采用分阶段冻结与微调的训练策略。

**📊 数据集**

使用包含120张冠脉造影图像（60正例、60负例）的自建数据集进行评估。

**📈 对比分析**

与ResNet18、ResNet34及量子增强ResNet18基线对比，模型在准确率85%、AUC 0.9372、F1 0.8657上取得最高成绩，且训练仅需4个epoch（约2.2分钟）。

**⚠️ 局限性**

局限在于仅处理单模态数据，未引入多模态临床信息，脑启发机制主要体现在训练策略上，缺乏更深入的神经动力学建模。

---

## 278. Minimum Envy Graphical House Allocation Beyond Identical Valuations

**arXiv ID:** 2601.15864 | [PDF](https://arxiv.org/pdf/2601.15864v1)

**作者:** Tanmay Inamdar `[一作]` (Indian Institute of Technology Jodhpur), Pranjal Pandey `[通讯]` (Indian Institute of Technology Jodhpur)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在社交图上最小化厌恶的住房分配问题，考虑代理人非同质估值的情况

**💡 创新点**

首次针对非同质估值引入参数化复杂度分析，给出多种结构参数（树宽、顶点覆盖、团模数、房屋类型）下的FPT与指数算法

**🔧 技术方法**

采用树分解动态规划、最小权重完美匹配、子集卷积、组合枚举等经典参数化与组合技术

**📊 数据集**

无实验数据，全部为理论算法与证明

**📈 对比分析**

与已知的仅考虑同质估值结果对比，显著降低时间复杂度并提供多种参数化可行方案

**⚠️ 局限性**

仍存在多项NP‑hard性、在一般图与某些特殊图（如路径、完全二分图）下的多项式可解性未解决，且对近似算法的研究不足

---

## 279. CGPT: Cluster-Guided Partial Tables with LLM-Generated Supervision for Table Retrieval

**arXiv ID:** 2601.15849 | [PDF](https://arxiv.org/pdf/2601.15849v1)

**作者:** Tsung-Hsiang Chou `[一作]` (National Chung Hsing University), Yao-Chung Fan `[通讯]` (National Chung Hsing University)

**通讯引用:** 606 | [OpenAlex ID](https://openalex.org/A5024883783)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对表格检索中的语义压缩问题，提出一种利用聚类生成多样化部分表格并结合LLM生成合成查询进行对比学习的训练框架 CGPT。

**💡 创新点**

创新点包括：① 用 K‑means 聚类对表格行进行语义分组，生成覆盖更广泛语义空间的部分表格；② 将 LLM 生成的合成查询作为监督信号，用硬负样本对比学习细调嵌入模型；③ 在多域统一语料上验证跨域泛化能力，且对不同规模 LLM 具有鲁棒性。

**🔧 技术方法**

核心技术包括：K‑means 聚类、LLM 合成查询（可使用 Llama‑3.1‑8B 等多模型）、硬负样本抽样、InfoNCE 对比损失、对齐多语种表格检索任务。

**📊 数据集**

在四个公开基准上进行实验：MimoTable（中英两种子集）、OTTQA、FetaQA 和 E2E‑WTQ，此外还构建了统一多域合并语料进行跨域评估。

**📈 对比分析**

与 QGpT 等检索增强基线对比，CGPT 在 R@1 上平均提升 16.54%，在 MimoTable(EN) 的 R@1 达到 60.13%（相比 QGpT 提升 9.47%），在多域设置下亦保持高性能；不同规模 LLM 的差异仅 0.6% 左右，显示方法对 LLM 依赖低。

**⚠️ 局限性**

主要局限包括：① 仍需依赖 LLM 生成查询，生成质量与 LLM 规模有关；② 对极大表格的聚类和查询生成可能计算成本较高；③ 在极为稀疏或单一主题的表格中，聚类效果可能受限。

---

## 280. Towards Realistic Remote Sensing Dataset Distillation with Discriminative Prototype-guided Diffusion

**arXiv ID:** 2601.15829 | [PDF](https://arxiv.org/pdf/2601.15829v1)

**作者:** Yonghao Xu `[一作]` (Linköping University), Qihao Weng `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 33187 | [OpenAlex ID](https://openalex.org/A5004755532)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于差分辨率扩散的遥感图像数据集蒸馏框架（DPD），通过合成少量代表性、可辨别性强的样本来替代大规模原始训练集，以实现模型训练的高效与隐私保护。

**💡 创新点**

创新点包括：① 将预训练分类器的一致性损失注入扩散训练，显著提升生成样本的判别性；② 采用视觉-语言双向指导——利用在潜在空间中的视觉原型聚类与多文本描述聚合，兼顾结构多样性与语义丰富性；③ 首次将数据蒸馏方法应用于遥感图像分类，展示了完全合成数据的可行性。

**🔧 技术方法**

核心技术包括：潜在空间的扩散模型（Stable Diffusion 2 + VAE）、CLIP文本编码器、视觉语言模型（Qwen 系列）生成伪说明、LLM（Qwen3-4B）聚合文本、K‑Means 聚类筛选视觉原型、LoRA 微调、分类一致性损失及多尺度扩散步骤调节。

**📊 数据集**

实验数据集：UCM、AID、NWPU‑RESISC45（分别涵盖 21、30、45 类场景），在不同样本/类别（IPC）设定下进行蒸馏。

**📈 对比分析**

与四种基线（Txt2Img‑MHN、DiT、Minimax Diffusion、Stable Diffusion 2）以及多种分类网络（AlexNet、VGG16、ResNet、DenseNet 等）进行对比。DPD 在所有 IPC 设定下均取得显著提升，尤其在低样本（IPC≤5）时可提升 15–20 % 的整体精度，并保持跨网络的一致性；训练与生成成本处于中等水平。

**⚠️ 局限性**

局限性：① 目前仅针对图像级场景分类；细粒度任务（如地物分类、目标检测）尚未验证；② 依赖大规模预训练模型与 LLM，导致模型体积和算力需求较高；③ 对极端高分辨率或特殊光照条件的适应性还有待进一步研究。

---

## 281. HyperAlign: Hypernetwork for Efficient Test-Time Alignment of Diffusion Models

**arXiv ID:** 2601.15968 | [PDF](https://arxiv.org/pdf/2601.15968v1)

**作者:** Xin Xie `[一作]` (University of New South Wales), Dong Gong `[通讯]` (University of New South Wales)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 HyperAlign，一个利用超网络在推理阶段动态生成低秩调节权重的框架，用于对扩散模型进行高效且有效的对齐，使生成图像更贴合文本提示和人类审美偏好。

**💡 创新点**

创新点在于：① 将对齐任务转化为在每一步采样时修改生成网络权重，而非直接修改潜在状态；② 通过超网络预测 LoRA 权重，实现输入与时间步特定的自适应调节；③ 采用奖励分数为目标并加入偏好正则化，抑制奖励劫持；④ 设计三种权重生成策略（全步、起步单步、分段）在效率与性能之间实现灵活权衡。

**🔧 技术方法**

核心技术包括：超网络（感知编码器 + Transformer 解码器）生成 LoRA 权重；奖励模型（HPSv2、PickScore 等）作为目标函数；偏好数据正则化（Pick-a-Pic、HPD 等）；在 Diffusion 或 Rectified Flow 框架（Stable Diffusion v1.5、FLUX）上进行实验；并与 RL、DPO、GRPO、BoN、ε-greedy 等多种对齐方法对比。

**📊 数据集**

使用了多种公开数据集进行评估：Pick-a-Pic（1K prompts）、GenEval（2K prompts）、HPD（500 prompts）、Partiprompt（1K prompts）；以及偏好正则化数据 Pick-a-Pic、HPD；奖励模型基于 HPSv2、PickScore 等。

**📈 对比分析**

与多类基线（RL、DPO、KTO、GRPO、BoN、ε-greedy、DyMO 等）对比，HyperAlign 在多项指标（PickScore、ImageReward、CLIP、Aesthetic Predictor 等）上均显著提升，尤其在视觉吸引力和文本对齐上优于测试时刻对齐方法；在推理时间上也实现了 3–20 秒的高效生成，远优于需要数分钟的梯度或采样重算方法。

**⚠️ 局限性**

局限性：① 超网络仍需额外的训练与存储开销，可能不适用于极大规模模型；② 对奖励信号高度依赖，若奖励模型或偏好数据质量不足，仍可能出现对齐失真；③ 目前仅在扩散和 FLUX 框架验证，其他生成模型的通用性尚未彻底验证。

---

## 282. Transfer Learning from ImageNet for MEG-Based Decoding of Imagined Speech

**arXiv ID:** 2601.15909 | [PDF](https://arxiv.org/pdf/2601.15909v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 283. Class Confidence Aware Reweighting for Long Tailed Learning

**arXiv ID:** 2601.15924 | [PDF](https://arxiv.org/pdf/2601.15924v1)

**作者:** Brainard Philemon Jagati `[一作]` (IEEE), Chandrashekhar Meshram `[通讯]` (IEEE)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

**🎯 论文内容**

本文提供了IEEE期刊论文的LaTeX模板使用说明与示例代码，展示如何利用IEEEtran.cls进行排版；

**💡 创新点**

创新点在于系统化整理了常见排版元素（章节、列表、图表、公式、算法等）的写法，并给出多种排版示例；

**🔧 技术方法**

使用了LaTeX、IEEEtran.cls、graphicx、subfig、amsmath等宏包来实现文档结构与排版；

**📊 数据集**

该文不涉及任何实际数据集，仅为排版示例；

**📈 对比分析**

本文未进行实验或性能对比，仅通过示例演示排版效果；

**⚠️ 局限性**

限制在于仅为模板演示，缺乏学术研究内容，需作者自行填充研究内容与实验数据以满足正式投稿要求。

---

## 284. Layered automata: A canonical model for automata over infinite words

**arXiv ID:** 2601.15940 | [PDF](https://arxiv.org/pdf/2601.15940v1)

**作者:** Antonio Casares `[一作]` (University Kaiserslautern-Landau), Igor Walukiewicz `[通讯]` (CNRS Bordeaux University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出层次化自动机（layered automata）并证明其可用于识别所有ω-正则语言；在满足一致性（consistency）条件下，层次化自动机是历史确定性的、0-1概率性的，并可唯一确定最小化的一致层次化自动机；给出了多项式时间的最小化、一致性检查和语言包含判定算法，并提供了基于同余的最小化表述。

**💡 创新点**

创新点在于：① 将历史确定性共Büchi自动机的多项式时间最小化结果推广到所有ω-正则语言；② 定义了一致性条件，使层次化自动机同时具备历史确定性和0-1概率性；③ 证明了每个ω-正则语言都有唯一的最小一致层次化自动机，并给出多项式时间构造；④ 提供了同余（congruence）表征与最小化的紧密联系。

**🔧 技术方法**

主要技术包括：构造层次化自动机的树状结构与对应的简单优先级交替自动机；定义一致性与均匀语义确定性，证明其等价；使用最长后缀求解器证明历史确定性和0-1概率性；多项式时间算法利用Büchi、Rabin和优先级游戏求解；最小化通过正规化（normalization）、降低（lowering）、安全最小化（safe minimization）以及中心化（centralization）等变换实现；同余表征利用层次化自动机的安全语言与同余关系。

**📊 数据集**

无具体数据集，论文主要为理论分析与算法设计；若涉及实验，则仅对示例语言和小型自动机进行演示。

**📈 对比分析**

方法通过多项式时间构造与判定实现；与传统的确定性PDA或非确定性Büchi自动机相比，层次化自动机在状态数上可指数压缩；与COCOA、rerailing自动机等比较时，层次化自动机能够在保持同样表达能力的同时实现更紧凑的结构；实验未给出量化性能，但理论证明表明最小化、包含等问题均在多项式时间内完成。

**⚠️ 局限性**

局限性包括：① 只对满足一致性的层次化自动机具备最小化与判定性质，非一致性情况未处理；② 尽管最小化是多项式时间，但构造过程仍可能产生指数级状态量，尤其在从任意交替或确定性PDA转换时；③ 与某些模型（如rerailing自动机）之间的转换仍需进一步研究；④ 对于更广义的ω-正则语言或更高等级的Borel层级，是否能保持一致性与历史确定性仍是开放问题。

---

## 285. TeNet: Text-to-Network for Compact Policy Synthesis

**arXiv ID:** 2601.15912 | [PDF](https://arxiv.org/pdf/2601.15912v1)

**作者:** Ariyan Bighashdel `[一作]` (Utrecht University), Kevin Sebastian Luck `[通讯]` (Vrije Universiteit Amsterdam)

**通讯引用:** 137 | [OpenAlex ID](https://openalex.org/A5040226194)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 TeNet 框架，利用大型语言模型的文本嵌入通过超网络一次性生成轻量级、可直接在高频控制回路中执行的任务特定机器人策略；

**💡 创新点**

创新点在于：①将自然语言直接作为策略生成的条件信号，消除在线推理；②通过与轨迹编码器的对齐（对比或均方误差）将语言语义与行为语义绑定；③使用超网络实现任务特定参数化，保持模型极小；

**🔧 技术方法**

技术包括：LLM 文本编码（LLaMA‑3 8B，Frozen），轨迹编码（Prompt‑DT），超网络生成策略权重，行为克隆与语言对齐损失（MSE 或 InfoNCE），离线多任务/元学习训练，强化实验评估；

**📊 数据集**

数据集：MuJoCo 运动控制（HalfCheetah‑Dir、HalfCheetah‑Vel、Ant‑Dir）和 Meta‑World 操作任务（ML1 Pick‑Place、MT10、MT50），采用离线专家轨迹和对应的自然语言描述；

**📈 对比分析**

与 Decision Transformer (DT) 与 Prompt‑DT 的对比：TeNet 在多任务设置下性能更好、在元学习设置下相近或略优；模型规模仅 ~40K 参数，控制频率可达 9 kHz，远超 Prompt‑DT 的 1–40M 参数和 190–600 Hz；在 MT10/MT50 上 Prompt‑DT 的成功率显著下降，而 TeNet 维持 99%+；

**⚠️ 局限性**

局限性包括：对训练任务规模敏感，需大量任务和示例才能实现最佳泛化；仅评估基于状态的离线设置，未涉及感知或在线强化微调；在高度多样化或极端语言变体下仍可能出现性能下降；

---

## 286. Opening the Black Box: Preliminary Insights into Affective Modeling in Multimodal Foundation Models

**arXiv ID:** 2601.15906 | [PDF](https://arxiv.org/pdf/2601.15906v1)

**作者:** Zhen Zhang `[一作]` (Shenzhen MSU-BIT University), Xiping Hu `[通讯]` (Shenzhen MSU-BIT University)

**通讯引用:** 8857 | [OpenAlex ID](https://openalex.org/A5007941489)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大规模多模态基础模型中情感建模的内部机制

**💡 创新点**

发现情感适配主要集中在前馈网络的gate_proj层，而非注意力输出投影

**🔧 技术方法**

采用模块级参数差异分析、模块迁移、单模块微调、消融实验，以及LoRA和SFT等参数高效微调技术

**📊 数据集**

使用Emotion‑LLaMA、AffectGPT、情感识别、抑郁检测等多任务数据集

**📈 对比分析**

通过对比基线模型与情感适配模型的模块权重变化、单模块加载/训练结果以及GET策略与全模态LoRA的性能，GET在仅调节24.5%参数的情况下保留了96.6%平均效果

**⚠️ 局限性**

仅在模块层面分析，未探究单元级细节；实验样本范围受限于现有情感基准；未解释gate_proj为何适合情感调制

---

## 287. Dynamic Server Allocation Under Stochastic Switchover on Time-Varying Links

**arXiv ID:** 2601.15904 | [PDF](https://arxiv.org/pdf/2601.15904v1)

**作者:** Hossein Mohammadalizadeh `[一作]` (Hasso Plattner Institute), Holger Karl `[通讯]` (Hasso Plattner Institute)

**通讯引用:** 8064 | [OpenAlex ID](https://openalex.org/A5019777818)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究多无人机FSO回程网络中具有随机、异质切换延迟的并行队列动态服务器分配问题，提出非贪婪帧式调度框架ACI。

**💡 创新点**

创新点是将切换延迟直接融入调度决策、采用帧式停留以摊销切换成本，并通过Lyapunov分析证明在缩放后的容量区域内吞吐量最优，同时可通过调节紧迫度指标实现吞吐-延迟权衡。

**🔧 技术方法**

采用队列排队理论、Lyapunov漂移分析、随机切换时间模型（包括几何分布的获取尝试）以及大规模仿真。

**📊 数据集**

采用基于真实FSO链路统计的仿真数据，设置多架无人机、FSO参数与切换时间分布，未使用公开数据集。

**📈 对比分析**

与传统Max‑Weight调度比较，ACI在服务时间占比约提升30‑50%、平均延迟降低20‑40%，且在不同切换时间模型下均保持优异。

**⚠️ 局限性**

限制在于容量区域被缩放、对切换时间统计的准确性敏感、参数（如帧长、γ、β）需人工调优，且实验仅在仿真环境验证。

---

## 288. Blind Identification of Channel Codes: A Subspace-Coding Approach

**arXiv ID:** 2601.15903 | [PDF](https://arxiv.org/pdf/2601.15903v1)

**作者:** Pramod Singh `[一作]` (International Institute of Information Technology), Arti Yardi `[通讯]` (International Institute of Information Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于子空间编码的新方法，用来在二进制对称信道（BSC）上进行盲码识别，即在接收机仅观测到噪声干扰的码字时识别发送方使用的线性码。

**💡 创新点**

创新点在于：① 将盲码识别问题与子空间编码问题建立起自然的联系；② 定义了“去噪子空间差异度”（denoised subspace discrepancy）并基于此提出最小去噪子空间差异度解码器（Minimum Denoised Subspace Discrepancy Decoder）；③ 给出了该解码器在有限权重错误、足够多可区分码字以及大于阈值 N 的情形下的理论识别保证与误码概率上界；④ 在大 N 情况下提出改进解码器（Improved Decoder）以缓解误码率上升。

**🔧 技术方法**

主要技术包括：子空间距离与汉明距离的结合、对码字行进行有界距离解码（BDD）实现去噪、子空间距离计算、有限域行列式/秩运算以及随机线性码的生成与判决统计分析。

**📊 数据集**

实验使用随机生成的等维线性码（长度 30、60，码率 1/3~1/4）作为数据集，构造两码交集维度不同，随后通过仿真评估各解码器性能。

**📈 对比分析**

通过与文献中已知的内部乘积方法（Inner‑Product Method）和最小子空间距离解码器（MSD）比较，实验表明改进解码器在高交叉概率 p、较少接收码字 N 时误码率显著低于两者；同时改进解码器的误码率曲线与理论误码上界吻合，说明理论分析有效。

**⚠️ 局限性**

限制包括：① 对于大于阈值 N 的情况，最小去噪子空间差异度解码器的性能可能下降，需要改进解码器；② 改进解码器的复杂度较高，需要枚举或采样子集；③ 目前仅在 BSC 与随机线性码上验证，尚未推广至 BCH、LDPC、卷积码或其他信道（如 BEC、AWGN）。

---

## 289. PMPBench: A Paired Multi-Modal Pan-Cancer Benchmark for Medical Image Synthesis

**arXiv ID:** 2601.15884 | [PDF](https://arxiv.org/pdf/2601.15884v1)

**作者:** Yifan Chen `[一作]` (University of Cambridge), Chao Li `[通讯]` (University of Cambridge)

**通讯引用:** 37949 | [OpenAlex ID](https://openalex.org/A5100323172)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文构建了首个完整配对的多模态（CT/CTC、MRI DCE）跨11器官的全癌症医学影像数据集，并提出基于流匹配的缺失模态重建模型 FlowMI。

**💡 创新点**

创新点在于：①公开且完整的器官级配对对比增强与非增强扫描；②使用连续流匹配在潜在空间对缺失模态进行动态恢复的 FlowMI；③提供统一的 N→N、多级翻译基准，涵盖 1→1、N→1、1→N 场景。

**🔧 技术方法**

技术手段包括：潜在流匹配（Latent Flow Matching）与变分自编码器相结合的潜在空间模型；对比传统直接模型（UNet、Transformer）、GAN（CycleGAN、Pix2Pix）、扩散模型（PatchDiff、DiTSR）等。

**📊 数据集**

主要使用的数据集为 PMPBench（Paired Multi‑Modal Pan‑Cancer Benchmark），从 TCIA 等公开来源聚合的 CT/CTC 与 MRI DCE1–DCE3 配对影像，覆盖 11 种器官。

**📈 对比分析**

通过在 1→1、N→1、1→N 等三类任务上与多种基线进行比较，FlowMI 在 PSNR、SSIM、FID、KID、LPIPS 等指标上均表现最佳，显示出更高的图像质量与结构一致性。

**⚠️ 局限性**

局限性包括：数据主要来自公开源，仍可能存在扫描多样性不足与标签不完整的问题；模型目前仅在 11 器官范围内验证，存在潜在的偏倚与隐私合规挑战。

---

## 290. STAR: Semantic Table Representation with Header-Aware Clustering and Adaptive Weighted Fusion

**arXiv ID:** 2601.15860 | [PDF](https://arxiv.org/pdf/2601.15860v1)

**作者:** Shui-Hsiang Hsu `[一作]` (National Chung Hsing University), Yao-Chung Fan `[通讯]` (National Chung Hsing University)

**通讯引用:** 606 | [OpenAlex ID](https://openalex.org/A5024883783)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种轻量级框架STAR，用于表格检索，通过语义聚类和加权融合改进表格表示。

**💡 创新点**

创新点在于：① 引入header-aware K-means聚类，选取语义多样且代表性强的行；② 生成每个聚类的专属合成查询；③ 采用加权融合（固定或动态）显式平衡表格与查询信息，提升语义对齐。

**🔧 技术方法**

技术包括：预训练编码器（如BGE‑M3）、LLM（Llama 3.1 8B-Instruct）生成查询、Header-aware K‑means聚类、加权融合策略（Fixed/Dynamic Weight Fusion）。

**📊 数据集**

在五个公开表格检索基准上进行评估：Mimo (ch)、Mimo (en)、OTTQA、FetaQA、E2E‑WTQ。

**📈 对比分析**

与现有基线QGpT比较，STAR在所有数据集上均取得显著提升；平均R@1提升约6.4个百分点，R@5和R@10也有明显改进；不同权重设定揭示数据集特征，Dynamic Weight Fusion表现最稳健。

**⚠️ 局限性**

局限性包括：依赖完整且信息丰富的表头，对缺失/模糊表头的表格效果下降；聚类和多查询生成增加计算开销，可能不适用于极大规模检索；加权策略仍为手工设定或简单动态，可进一步学习化。

---

## 291. Existential Positive Transductions of Sparse Graphs

**arXiv ID:** 2601.15890 | [PDF](https://arxiv.org/pdf/2601.15890v1)

**作者:** Nikolas Mählmann `[一作]` (University of Warsaw), Sebastian Siebertz `[通讯]` (University of Bremen)

**通讯引用:** 856 | [OpenAlex ID](https://openalex.org/A5073548944)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在稀疏图类上的存在性正一阶逻辑（existential positive FO）转化（transduction）及其对结构稳定性（monadic stability）的影响，提出了新的组合运算“subflip”，并利用它对半鞍形（semi‑ladder‑free）类（即co‑matching‑free类）进行完全的组合与逻辑刻画。作者进一步证明了存在性正稀疏化猜想（existential positive sparsification conjecture）在所有已知的特殊情况（如有限植绒深度、线性团宽度、团宽度、双缘宽度和合并宽度）下成立，并展示了稀疏前像可以作为原始密集图的子图。最后，作者给出了正MSO逻辑对正FO逻辑的可压缩性结果。

**💡 创新点**

创新点包括：①引入并正式定义了“subflip”操作，成为“flip”的稀疏对应；②给出subflip-flatness和subflipper-rank的组合学与逻辑等价性，完成了co‑matching‑free类的全新刻画；③证明存在性正稀疏化猜想在所有已知情况下成立，并得到稀疏前像可作为子图的更强结论；④证明正MSO在稀疏结构上等价于正FO，扩展了已知的可压缩性。

**🔧 技术方法**

主要技术手段包括：
- 有限模型理论中的转化（transduction）与正公式的性质；
- 组合学工具（flip、subflip、flipper游戏、子结构仿射、子翻转深度、子翻转深度覆盖）；
- 归纳与递归构造（flipper‑rank、subflipper‑rank、深度参数的递推关系）；
- Ramsey 与鸽巢原理的组合，用于证明大结构中必然存在特定子图；
- 正MSO到正FO的归约通过正性单调性与集合量词消除实现。

**📊 数据集**

本研究为纯理论工作，未使用任何实验数据集；所有结果均通过数学证明得到。

**📈 对比分析**

由于研究的是理论性质，没有对实验方法或性能进行比较；所有结论均基于严谨的证明与组合论/逻辑等价性，不涉及可执行性能评估。

**⚠️ 局限性**

局限性与未解问题：
- 结果仅适用于co‑matching‑free（半鞍形）类；完整的monadic stability（包含co‑matching）尚未完全解决。
- 证明过程中需要在图中允许自环，若强行禁止自环则正转化的表达能力会被削弱。
- 正MSO对正FO的可压缩性仅在无否定符号（positive）时成立，无法推广到一般的MSO。
- 目前对更一般的结构类（如所有无限扩张类）仍缺乏对应的稀疏化结论。

---

## 292. Artificial Rigidities vs. Biological Noise: A Comparative Analysis of Multisensory Integration in AV-HuBERT and Human Observers

**arXiv ID:** 2601.15869 | [PDF](https://arxiv.org/pdf/2601.15869v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 293. NeuroMamba: Multi-Perspective Feature Interaction with Visual Mamba for Neuron Segmentation

**arXiv ID:** 2601.15929 | [PDF](https://arxiv.org/pdf/2601.15929v1)

**作者:** Liuyun Jiang `[一作]` (Chinese Academy of Sciences), Hua Han `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 9586 | [OpenAlex ID](https://openalex.org/A5100676754)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种多视角特征交互框架 NeuroMamba，用于高质量的三维电镜神经元分割，解决了 CNN 局部感受野不足和 Transformer 依赖分块导致的边界模糊问题。

**💡 创新点**

创新点包括：①将 Mamba 的线性复杂度引入无分块全局建模；②设计了基于通道门控的 Strip Pooling 的 Boundary Discriminative Feature Extractor (BDFE)；③提出了分辨率感知的 Spatial Continuous Feature Extractor (SCFE) 以及跨扫描（transverse‑first / axial‑first）机制；④使用 Cross‑Feature Interaction（cross‑modulation）实现本地与全局特征的动态融合。

**🔧 技术方法**

采用的技术包括：Mamba 作为 backbone，strip pooling 与通道门控，分辨率感知扫描的双向跨扫描，跨特征交互（cross‑modulation），以及传统的 CNN/Transformer 后处理（Waterz 与 Multicut）进行实例分割。

**📊 数据集**

实验数据集覆盖四种公开 EM 数据：AC3/AC4（多斜率 6×6×29 nm³）、CREMI‑A/B/C（4×4×40 nm³）、FIB25（8×8×8 nm³）和 Kasthuri（6×6×29 nm³ 大块），实现了从细微结构到大体量神经元的全尺度验证。

**📈 对比分析**

与 CNN（MALA、Superhuman、PEA、LSD）、Transformer（UNETR、SwinUNETR）和 Mamba（U‑Mamba、SegMamba、EMMamba）基线在 VI 与 ARAND 指标上对比，NeuroMamba 在所有数据集上均获得最优或次优结果；在最具挑战性的 CREMI‑A 上，ARAND 下降 22.4%（Waterz）和 21.7%（Multicut），在 FIB25 与 Kasthuri 上同样表现出显著优势。

**⚠️ 局限性**

局限性：推理延迟相对较高（约 0.14–0.18 s/块），对极大体量数据的实时性和多尺度迁移能力尚未全面验证；此外，模型对极端分辨率差异的自适应仍需进一步研究。

---

## 294. Evaluating and Achieving Controllable Code Completion in Code LLM

**arXiv ID:** 2601.15879 | [PDF](https://arxiv.org/pdf/2601.15879v1)

**作者:** Jiajun Zhang `[一作]` (University of Science and Technology of China), Junyang Lin `[通讯]` (Alibaba Group)

**通讯引用:** 3025 | [OpenAlex ID](https://openalex.org/A5100612233)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了可控代码补全基准C^3-Bench，并对40+ LLM在指令遵循能力上进行评估

**💡 创新点**

首次引入细粒度指令引导的代码补全基准，揭示开源与专有模型在指令遵循上的显著差距

**🔧 技术方法**

使用LLM生成指令-代码对进行监督微调，结合AST提取中间代码和自动化评测指标

**📊 数据集**

数据来源于HumanEval、SAFIM以及GitHub开源仓库，构成2195个Python实例

**📈 对比分析**

通过Pass@1、指令遵循率(IF)和编辑相似度(ES)三指标评测，Qwen2.5-Coder-C^3在C^3-Bench上取得SOTA

**⚠️ 局限性**

局限在于仅覆盖Python单文件任务、受基模型能力限制，需进一步扩展多语言与更大上下文

---

## 295. Why Inference in Large Models Becomes Decomposable After Training

**arXiv ID:** 2601.15871 | [PDF](https://arxiv.org/pdf/2601.15871v1)

**作者:** Jidong Jin `[一作]` (Capital University of Economics and Business), Jidong Jin `[通讯]` (Capital University of Economics and Business)

**通讯引用:** 128 | [OpenAlex ID](https://openalex.org/A5101271771)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

提出了一种仅基于训练后参数矩阵的统计结构提炼与重组方法，将大型神经网络拆分为块状子系统，实现可并行、可控的推理执行。

**💡 创新点**

创新点在于：1）不依赖梯度/激活轨迹，直接利用初始化分布作为零假设进行显著性检验；2）将统计上显著的边保留、无效边归零，形成结构化稀疏；3）通过矩阵重排、Permutation/Projection/Embedding实现功能等价的系统重组，开启“可并行、可调节的推理层”。

**🔧 技术方法**

核心技术包括：Neyman显著性检验、随机行走（equiprobability）假设检验、图论的连通分量与强连通分量划分、矩阵重排（Permutation）、子矩阵投影/嵌入操作，以及整体的结构化稀疏化流程。

**📊 数据集**

文中未给出具体数据集，所述方法被设计为通用，适用于任何大规模训练完成的神经网络。

**📈 对比分析**

对比方式：作者提出验证阶段（功能一致性测试）和生产阶段（资源利用与并行度评估），但在论文中未给出量化实验结果，预期在推理吞吐量与能耗上有明显提升，尤其在模型规模极大时能显著降低内存与计算瓶颈。

**⚠️ 局限性**

局限性：1）需要预先设定显著性阈值与随机行走容差，可能导致不同阈值下结构差异；2）仅考虑训练后参数，无法捕捉微调或迁移学习过程中参数重塑的动态变化；3）对初始化分布假设过强，若模型使用非标准初始化或正则化，显著性检验可能失效；4）缺乏大规模实验验证，实际性能提升需进一步实测。

---

## 296. Improve the autonomy of the SE2(3) group based Extended Kalman Filter for Integrated Navigation: Application

**arXiv ID:** 2601.16078 | [PDF](https://arxiv.org/pdf/2601.16078v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 297. How to Tamper with a Parliament: Strategic Campaigns in Apportionment Elections

**arXiv ID:** 2601.15855 | [PDF](https://arxiv.org/pdf/2601.15855v1)

**作者:** Robert Bredereck `[一作]` (TU Clausthal), Tessa Seeger `[通讯]` (Heinrich Heine University Dusseldorf)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并研究了在议会比例代表制（apportionment）选举中进行的“战略竞选”（即投票买卖）的计算模型，定义了构造性与破坏性两类投票改动问题，并在单区与多区两种情形下分别给出了多种方法（D'Hondt、Sainte‑Laguë、Largest‑Remainder、FPTP）的算法与复杂度分析。进一步地，作者提出了阈值选举的第二次机会模式，并在实验中验证了最优策略与启发式策略、阈值与区划对选举结果的影响。

**💡 创新点**

创新点主要体现在：
1) 将之前的两篇会议论文合并并大幅扩展，形成完整的议会选举改动理论框架；
2) 对单区比例代表制的构造性与破坏性问题给出多种多项式时间算法（含动态规划实现）；
3) 引入并分析了阈值选举的第二次机会模式，揭示其导致的问题变得不可解；
4) 在多区情形下给出从单区算法到多区 knapsack 方案的转化，并证明了多区获胜改动问题的 NP‑难与 W[1]‑难；
5) 通过大规模真实选举数据实验，系统评估了最优策略与启发式策略的性能差异，以及阈值与区划对成本的显著影响。

**🔧 技术方法**

技术方法包括：
- 计算复杂度分析与归约（从 Cubic‑Vertex‑Cover、Unary‑Bin‑Packing、Hitting‑Set 等经典问题）；
- 证明与实现多项式时间动态规划算法（针对 divisor 序列与 Largest‑Remainder 方法）；
- 通过二分搜索计算 “跳点” 以构造 γ 表；
- 通过多区 knapsack 动态规划求解多区问题；
- 参数化复杂度分析（证明针对区划数量的 W[1]‑难性）；
- 在实验中使用 Python 与现有开源库对真实投票数据进行仿真与比较。

**📊 数据集**

使用了多国真实议会选举数据集，涵盖奥地利、新西兰、波兰、西班牙、土耳其等国的选举结果；数据集可在作者提供的 GitHub 仓库（https://github.com/bredereck/Strategic-Campaigns-in-Apportionment-Elections）获取，并包含不同阈值与区划设置下的投票记录。

**📈 对比分析**

实验结果表明：
- 在单区情形下，最优策略比常见启发式方法需要 1.2–3 倍（甚至高达 6.5 倍）更少的投票改动即可实现目标；
- 阈值的引入可被利用，少量投票（约 0.25%）即可让目标党获得 4.4%–12.5% 的席位；
- 区划数量越少，改变席位所需投票越多（构造性约 3.5 倍、破坏性约 6 倍）；
- 结果通过图表与统计表详细展示，验证了理论分析与算法效果。

**⚠️ 局限性**

局限性包括：
- 对多区获胜改动问题仍属于 NP‑难（W[1]‑难），在实际大规模选举中难以直接求解；
- 第二次机会阈值模式虽理论可行，但实际计算复杂度显著增加，导致实用性受限；
- 实验仅覆盖有限国家与选举，未必能覆盖所有选举制度与阈值设置；
- 所有算法假设投票修改成本相等，未考虑更复杂的成本结构或选民行为模型；
- 在极端阈值或区划极端不均时，模型与算法的鲁棒性尚未充分验证。

---

## 298. Efficiently Learning Robust Torque-based Locomotion Through Reinforcement with Model-Based Supervision

**arXiv ID:** 2601.16109 | [PDF](https://arxiv.org/pdf/2601.16109v1)

**作者:** Yashuai Yan `[一作]` (Autonomous Systems Lab), Dongheui Lee `[通讯]` (Autonomous Systems Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种结合模型预测控制与残差强化学习的监督式学习框架，用于在真实世界不确定性下实现鲁棒的关节力矩级步态控制。

**💡 创新点**

创新点在于引入具有全局信息的Oracle策略作为监督信号，并在RL目标中加入监督损失，从而显著提升学习效率并减轻奖励设计压力；同时利用残差学习使得基于模型的控制器与数据驱动策略协同工作。

**🔧 技术方法**

技术包括：基于DCM的轨迹规划与逆动力学全身控制；残差RL（PPO）与Oracle监督损失；域随机化（传感器噪声、动力学不确定性、环境摩擦等）；双层LSTM-MLP网络结构。

**📊 数据集**

数据集为在MuJoCo仿真环境中通过域随机化生成的多种随机动力学与传感器噪声场景，覆盖三款双足机器人（Kangaroo、Unitree H1-2、Bruce）的关节状态与力矩数据；无公开真实数据集。

**📈 对比分析**

与传统模型基控制（MBC）、残差RL无监督（ResRL）以及纯模仿学习（IL）对比，Oracle监督残差RL（BOR）在成功率、DCM/足迹跟踪误差和累计奖励等指标上均接近或超过Oracle策略，且训练样本与时间显著减少；在不同随机化程度下保持高鲁棒性。

**⚠️ 局限性**

局限性包括：仍需Oracle策略提供监督，若Oracle在某些极端扰动下表现不足可能限制最终性能；对高频实时控制的计算需求较高；方法目前仅在仿真验证，真实世界落地验证尚待进一步研究。

---

## 299. Clustering-Guided Spatial-Spectral Mamba for Hyperspectral Image Classification

**arXiv ID:** 2601.16098 | [PDF](https://arxiv.org/pdf/2601.16098v1)

**作者:** Zack Dewis `[一作]` (University of Calgary), Lincoln Linlin Xu `[通讯]` (University of Calgary)

**通讯引用:** 2799 | [OpenAlex ID](https://openalex.org/A5034166335)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 CSSMamba 框架，将聚类引导的空间 Mamba 与光谱 Mamba 结合，用于高光谱图像分类。

**💡 创新点**

创新点包括：①聚类引导的空间 Mamba 缩短 token 序列并提升特征学习；②结合 Spectral Mamba 构建双分支空间‑光谱框架；③双注意力 token 选择机制与可学习聚类模块。

**🔧 技术方法**

采用 Mamba 状态空间模型、聚类分割、双注意力排序、可学习聚类中心、光谱分支与空间分支等技术。

**📊 数据集**

使用 Pavia University、Indian Pines、Liao‑Ning 01 三个常用高光谱数据集进行验证。

**📈 对比分析**

与 ViT、ConvNeXt、SSTN、SSRN、SSFTT、MambaHSI、SDMamba 等前沿模型对比，整体准确率、平均准确率和 Kappa 均优于多数方法，尤其在边界保持和小类别识别上表现突出。

**⚠️ 局限性**

仍受聚类数选择影响，极少样本类别效果有限，且在高分辨率图像上计算成本仍较高。

---

## 300. DNF formulas are efficiently testable with relative error

**arXiv ID:** 2601.16076 | [PDF](https://arxiv.org/pdf/2601.16076v1)

**作者:** Xi Chen `[一作]` (Columbia University), Rocco A. Servedio `[通讯]` (Columbia University)

**通讯引用:** 5668 | [OpenAlex ID](https://openalex.org/A5014866889)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究了一种在相对误差模型下测试任意s项DNF函数是否满足该类的算法，提出了一个查询复杂度为O(s/ε)的低复杂度测试器。

**💡 创新点**

核心创新在于对任意s项DNF进行新的“局部簇”分解（K‑clustering），将DNF拆解为若干个factored‑DNF（head+尾部），并利用该结构在相对误差模型中实现高效测试。

**🔧 技术方法**

技术手段包括：
- 结构化的K‑clustering与聚类合并；
- 通过样本集合构造池子并进行合并；
- 近似学习尾部DNF（使用k‑junta的相对误差测试技术）；
- 用Type‑1和Type‑2查询的模拟器来实现对factored‑DNF的查询；
- 证明相对误差与绝对误差模型的关系，利用近似三角不等式和对称性。

**📊 数据集**

该工作为纯理论研究，未使用具体数据集，而是在随机采样与黑盒查询模型下进行分析。

**📈 对比分析**

与传统的绝对误差模型测试相比，提出的算法在相对误差模型下实现了查询复杂度与维度n无关，显著低于先前的n^O(log(s/ε))上界；在常数ε下，查询复杂度为Θ(s/ε)，与已知的绝对误差模型最优结果相当。

**⚠️ 局限性**

局限性包括：
- 只针对s项DNF，尚未推广到更一般的布尔函数类；
- 依赖相对误差模型，需要访问f⁻¹(1)的随机采样器，实际实现可能受限；
- 对ε的取值有一定假设（如ε足够小常数）以保证K=log²(s/ε)的有效性。

---

## 301. DextER: Language-driven Dexterous Grasp Generation with Embodied Reasoning

**arXiv ID:** 2601.16046 | [PDF](https://arxiv.org/pdf/2601.16046v1)

**作者:** Junha Lee `[一作]` (Pohang University of Science and Technology), Minsu Cho `[通讯]` (RLWRLD)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种基于语言的多指抓取生成方法，先通过接触位置预测实现身体化推理，再自回归生成手部姿态，支持用户通过部分接触约束进行可调抓取；

**💡 创新点**

创新点在于引入基于接触的身体化链式推理（ECoT），将手指接触点作为中间表示，显著提升意图对齐和抓取多样性，并实现可调抓取；

**🔧 技术方法**

采用Vision‑Language‑Action框架：3D点云编码器PartField、LLM（Qwen2.5）自回归生成、离散化动作与接触位置令符、混合注意力与位置Dropout等技术；

**📊 数据集**

使用DexGYS与Dexonomy数据集，并通过MuJoCo自动生成物理接触注释及VLM生成抓取描述；

**📈 对比分析**

在DexGYS基准上与SOTA对比，成功率达到67.14%（+3.83pp），P‑FID 0.20（+96.4%意图对齐），同时展示了更高的抓取多样性；

**⚠️ 局限性**

局限性包括自回归易受累积错误影响，主要在单一静态物体场景，未覆盖复杂遮挡场景，实时推理速度有限。

---

## 302. Collision-Free Humanoid Traversal in Cluttered Indoor Scenes

**arXiv ID:** 2601.16035 | [PDF](https://arxiv.org/pdf/2601.16035v1)

**作者:** Han Xue `[一作]` (Tsinghua University), Li Yi `[通讯]` (Tsinghua University)

**通讯引用:** 39833 | [OpenAlex ID](https://openalex.org/A5100421454)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究并实现了在杂乱室内场景中实现无碰撞的人形机器人穿越技术。

**💡 创新点**

创新点在于提出 Humanoid Potential Field (HumanoidPF)——一种基于改进 APF 的连续可微场，既可作为观测输入也可作为奖励引导，显著提升 RL 学习效率；引入混合场景生成与专家→通用策略蒸馏的训练框架；以及实现基于点击的实时遥操作系统（Click‑and‑Traverse）。

**🔧 技术方法**

采用强化学习（PPO）、改进 APF、von Mises–Fisher 奖励分布、混合真实与程序化障碍的场景生成、DAgger 蒸馏、Fast‑LIO2 与 OctoMap 的 SLAM、以及在 Unitree G1 机器人上的硬件部署。

**📊 数据集**

使用 3D‑FRONT 真实室内数据集、自动生成的程序化障碍集合以及在真实实验室中收集的 30 个艺术设计室内场景进行评估。

**📈 对比分析**

与 ASTraversal、Humanoid Parkour、以及使用传统观察/奖励方式的自身变体进行对比；在模拟与真实环境中均达到了 95–97% 的成功率，且在 sim‑to‑real 迁移时误差极小，优于基线方法。

**⚠️ 局限性**

局限性包括：目前不支持利用接触丰富的交互（如靠墙、踏台等），以及对完全未知、极度杂乱的室内环境的泛化能力仍有待提升。

---

## 303. Universal Refusal Circuits Across LLMs: Cross-Model Transfer via Trajectory Replay and Concept-Basis Reconstruction

**arXiv ID:** 2601.16034 | [PDF](https://arxiv.org/pdf/2601.16034v1)

**作者:** Tony Cristofano `[一作]` `[通讯]` (Independent Researcher), Tony Cristofano (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种跨模型拒绝行为的转移方法，利用概念基底重建将捐赠者模型中的拒绝电路迁移到目标模型，而无需目标侧的拒绝监督。

**💡 创新点**

核心创新在于将拒绝电路视为可迁移的低维语义混合（概念“配方”），通过轨迹重放、概念基底对齐以及权重SVD稳定防护，实现跨架构（Dense→MoE、Dense→Reasoning）安全行为的无监督转移。

**🔧 技术方法**

技术包括：概念原子注册表（CAR）构建、层对齐（DTW+Gram指纹）、残差向量残差化、岭回归重建概念权重、SVD投影防护以及单秩抑制更新。

**📊 数据集**

数据集主要是：JailbreakBench（300个恶意提示）用于测量拒绝率；GSM8K和MBPP用于评估算术与编码能力；WikiText用于评估目标模型的语言漂移；CAR使用50个无害概念提示构造。

**📈 对比分析**

在8对模型（跨规模、跨家族、跨架构）上进行对比实验。相较于基线（随机方向、错位映射、无关概念、无防护），轨迹重放方法在拒绝率大幅下降（如从95%降至1%），同时保持或提升GSM8K/MBPP表现，且语义漂移（PPL）控制在1%以内，证明方法有效且稳定。

**⚠️ 局限性**

局限性包括：需要完整访问模型权重与内部激活，依赖CAR覆盖度，SVD防护假设高方差子空间承载核心能力，且在多模态或高度上下文敏感的拒绝场景下可能失效。

---

## 304. Stacked Intelligent Metasurface-Aided Wave-Domain Signal Processing: From Communications to Sensing and Computing

**arXiv ID:** 2601.16030 | [PDF](https://arxiv.org/pdf/2601.16030v1)

**作者:** Jiancheng An `[一作]` (Nanyang Technological University), Lajos Hanzo `[通讯]` (University of Southampton)

**通讯引用:** 83190 | [OpenAlex ID](https://openalex.org/A5091122305)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了堆叠智能超表面（SIM）在波域信号处理中的理论基础、原型实现、配置与训练方法，以及其在通信、感知和计算三大领域的应用与挑战。

**💡 创新点**

创新点在于：①首次系统化地把人工神经网络、波动计算与可编程超材料融合为一种新型硬件神经网络；②提出多种配置/训练框架（仿真导向、现场自适应、强化学习等）；③在三大应用场景中给出统一的功能集成视角，并提出跨域融合的未来研究路线。

**🔧 技术方法**

所用技术包括：多层可编程超表面结构、光学/射频元件、梯度下降/反向传播、遗传算法、深度强化学习、仿真工具（FDTD/FEM）以及现场校准技术。

**📊 数据集**

论文引用了多项实验原型，使用的数据集主要为公共图像分类数据集（MNIST、Fashion‑MNIST、ImageNet、EMNIST）以及无线信号基准数据（如OFDM/IMO实验数据）。

**📈 对比分析**

通过列举原型性能（识别准确率、能量分布、波束聚焦、功率增益等）与传统数字处理方法对比，表明SIM在处理速度（光速）、功耗（低能耗）和硬件复杂度（少量RF链）方面具有显著优势，但在部分任务中仍受限于精度与可调度范围。

**⚠️ 局限性**

主要局限包括：①通道估计和反射校准的困难；②传播模型与仿真误差导致的训练偏差；③非线性激活和可重配置性的缺乏；④大规模实现时的材料损耗与散射损失；⑤能效评估仍需进一步系统化研究。

---

## 305. Deja Vu in Plots: Leveraging Cross-Session Evidence with Retrieval-Augmented LLMs for Live Streaming Risk Assessment

**arXiv ID:** 2601.16027 | [PDF](https://arxiv.org/pdf/2601.16027v1)

**作者:** Yiran Qiao `[一作]` (Institute of Computing Technology), Qing He `[通讯]` (Institute of Computing Technology)

**通讯引用:** 16283 | [OpenAlex ID](https://openalex.org/A5100734672)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种跨会话检索增强的LLM驱动的轻量级检测框架（CS‑VAR），用于实时直播风险评估并输出可解释的patch级风险信号。

**💡 创新点**

1）通过跨会话检索将相似行为片段聚合，形成检索增强的LLM推理；2）将LLM的多粒度推理结果蒸馏到轻量化小模型，实现实时推断；3）采用PatchNet对直播进行用户‑时间槽网格编码，并使用图注意力捕获跨实体依赖；4）提供可解释的patch级风险与 saliency 指标。

**🔧 技术方法**

PatchNet（Transformer+LSTM+Graph‑Aware Transformer）、BERT文本编码、FAISS检索索引、检索增强的LLM（doubao‑1.5‑pro‑32k）、多任务蒸馏损失、图注意力、Transformer attention、语义聚合检索。

**📊 数据集**

两大真实工业数据集（May 与 June），来源于某大型直播平台，分别包含约22-24万场直播，30分钟截断、100s 时代槽、1:10 负样本比例、50名最活跃观众。

**📈 对比分析**

与Transformer、Reformer、Informer 等序列模型以及 mi‑NET、AtMIL、AdMIL、MIL‑LET、TimeMIL、TAIL‑MIL 等 MIL 聚合方法对比，CS‑VAR 在 PR‑AUC、F1‑score、R@0.1FPR、FPR@0.9R 上均显著优于所有基线，PR‑AUC 提升约5%，F1 提升约0.7%；在线上部署同样优于 XGBoost 和 Transformer，PR‑AUC 提升至 0.6643，F1 至 0.6546。

**⚠️ 局限性**

训练阶段需要调用大型 LLM 进行推理，计算成本高；仅对 30 分钟截断的会话做评估，未覆盖更长会话；检索召回的质量直接影响 LLM 推理效果；跨平台泛化与隐私合规性仍待进一步验证。

---

## 306. The Role of Cognitive Abilities in Requirements Inspection: Comparing UML and Textual Representations

**arXiv ID:** 2601.16009 | [PDF](https://arxiv.org/pdf/2601.16009v1)

**作者:** Giovanna Broccia `[一作]` (CNR), Alessio Ferrari `[通讯]` (University College Dublin)

**通讯引用:** 2991 | [OpenAlex ID](https://openalex.org/A5041720518)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一项交叉实验中，研究了在需求审查任务中使用 UML 序列图+文本与仅文本两种表示方式，并探讨了工作记忆与空间旋转两种认知能力对检出缺陷与推理质量的影响。

**💡 创新点**

创新点在于首次将两种认知能力与表示方式的三阶交互效应纳入模型，揭示“认知匹配”对 UML 帮助的决定性作用，并说明高认知能力并非总能提升检测性能。

**🔧 技术方法**

采用线性混合效应模型（LMM）和广义线性模型（GZLM）对 F1‑score 与“justification accuracy”进行统计分析，并使用分层 Benjamini‑Hochberg 校正多重检验。

**📊 数据集**

实验使用自制需求文档（Arkanoid 与 Snake）以及在线认知测验（3D 空间旋转、操作跨度）产生的原始数据，未使用公开大规模数据集。

**📈 对比分析**

对比结果显示：在高工作记忆和高空间旋转的受试者中，UML 支持会降低缺陷检测的 F1‑score，但显著提升对已检出缺陷的解释质量；相对文本仅模式，UML 仅在认知匹配良好时才有效；无明显序列效应。

**⚠️ 局限性**

主要局限包括样本量仅 38 名研究生、空间旋转得分高度集中缺乏变异、受试者为学生而非行业专业人员，且需求文档相对简单，限制了对更复杂真实项目的外部效度。

---

## 307. Variable Splitting Binary Tree Models Based on Bayesian Context Tree Models for Time Series Segmentation

**arXiv ID:** 2601.16112 | [PDF](https://arxiv.org/pdf/2601.16112v1)

**作者:** Yuta Nakahara `[一作]` (Waseda University), Toshiyasu Matsushima `[通讯]` (Waseda University)

**通讯引用:** 460 | [OpenAlex ID](https://openalex.org/A5110471799)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于贝叶斯上下文树（BCT）的可变拆分二叉树（VSBT）模型，用于离线时间序列分割。

**💡 创新点**

创新点在于：①将拆分点位置从固定的区间中点改为可通过递归逻辑回归自由决定，从而使树结构更紧凑；②同时估计拆分位置与树深度的联合后验。

**🔧 技术方法**

采用局部变分近似（用于处理逻辑回归的非解析性）结合上下文树加权（CTW）算法，实现了高效的贝叶斯推断。

**📊 数据集**

实验使用人工合成的AR过程和带噪声正弦波数据，未使用公开真实数据集。

**📈 对比分析**

与传统固定拆分二叉树（FSBT）对比，VSBT在相同分辨率下树深度更小，能够更准确捕捉变化点，并提供变化点的后验不确定性估计。

**⚠️ 局限性**

局限性包括：仅在合成数据上验证，初始化方法对结果影响大；对真实复杂序列的鲁棒性和泛化能力仍需进一步研究。

---

## 308. Probably Approximately Correct Maximum A Posteriori Inference

**arXiv ID:** 2601.16083 | [PDF](https://arxiv.org/pdf/2601.16083v1)

**作者:** Matthew Shorvon `[一作]` (King's College London), David S. Watson `[通讯]` (King's College London)

**通讯引用:** 10148 | [OpenAlex ID](https://openalex.org/A5067296979)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PAC-MAP算法，对MAP推理提供可证的近似解。

**💡 创新点**

创新点在于将PAC学习框架引入MAP，给出可验证的误差‑置信度保证，并证明纯随机方法的均匀最优性。

**🔧 技术方法**

采用随机采样、概率电路（PC）、信息论度量（熵）以及自适应停止准则实现PAC-MAP。

**📊 数据集**

在Twenty Datasets基准上测试，包含多达500维的数据集。

**📈 对比分析**

与三种经典近似MAP求解器（SMA、MMA、L‑MAP）以及自适应扫描器比较，PAC-MAP在低维和中等维度下能获得更高的MAP概率并提供置信度；在高维时可通过自适应启发式和温启动提升效果。

**⚠️ 局限性**

局限性包括：当最小熵随维度快速增长时停机时间过大，且算法不保证得到最优解，仅给出PAC保证；高维搜索及Hamming扫描效率低。

---

## 309. Masked Modeling for Human Motion Recovery Under Occlusions

**arXiv ID:** 2601.16079 | [PDF](https://arxiv.org/pdf/2601.16079v1)

**作者:** Zhiyin Qian `[一作]` (ETH Zurich), Siyu Tang `[通讯]` (ETH Zurich)

**通讯引用:** 6751 | [OpenAlex ID](https://openalex.org/A5056265728)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 MoRo，一种基于遮挡生成式 Transformer 的端到端框架，用于在单目视频中实时恢复3D人体运动，尤其能处理遮挡情况。

**💡 创新点**

创新点在于：① 通过 MoCap、图像-姿态和视频-运动三种模态的跨模态预训练构建强大先验；② 把遮挡视为遮挡建模，采用遮挡生成式 Transformer 进行高效推断；③ 引入轨迹感知运动先验与图像条件姿态先验，实现无预处理、实时、全局坐标系下的运动恢复。

**🔧 技术方法**

使用技术包括：VQ‑VAE 离散化姿态编码器、RoPE 位置编码、跨模态 Transformer 解码器、置信度引导遮罩策略、多步推断与姿态平滑网络，以及遮挡生成式 Transformer。

**📊 数据集**

使用的数据集有：MoCap 数据（AMASS、BEDLAM、MOYO）、图像-姿态数据（Human3.6M、MPI‑INF‑3DHP、COCO、MPII）以及视频-运动数据（EgoBody、RICH）。

**📈 对比分析**

与 MEGA、TokenHMR、PromptHMR、RoHM、WHAM、GVHMR 等方法进行对比，实验显示在 EgoBody‑Occ（遮挡）场景下，MoRo 在 MPJPE、GMPJPE、运动平滑和脚滑等指标上均显著优于基线；在 RICH（无遮挡）场景下与基线相当且运动更平滑，且可实现 70 FPS 的实时推断。

**⚠️ 局限性**

局限性包括：仅支持已知内参的静态相机场景，无法处理动态相机；对极端遮挡和多摄像头视频的适用性仍待验证。

---

## 310. Explainable AI to Improve Machine Learning Reliability for Industrial Cyber-Physical Systems

**arXiv ID:** 2601.16074 | [PDF](https://arxiv.org/pdf/2601.16074v1)

**作者:** Annemarie Jutte `[一作]` (Saxion University of Applied Sciences), Uraz Odyurt `[通讯]` (University of Twente)

**通讯引用:** 39 | [OpenAlex ID](https://openalex.org/A5024917187)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过对工业 CPS 的功率监测时间序列进行自定义分解，并利用 SHAP 解释模型对不同信号成分的重要性进行评估，从而在 CNN 训练过程中根据解释结果调整数据窗口大小，提升模型对异常条件的识别准确率。

**💡 创新点**

将概念级 SHAP（C‑SHAP）与工业 CPS 时间序列的可解释分解相结合，首次实现基于解释结果动态改进数据预处理和模型超参数的闭环方法。

**🔧 技术方法**

采用自定义层级分解（Levels、Peaks、Scale、Low Frequency、High Frequency）生成概念，使用 SHAP 计算概念贡献，训练卷积神经网络进行异常检测，并通过窗口大小变化验证性能提升。

**📊 数据集**

实验数据来自 ODROID‑XU4 设备在 24 个不同工作负载与功率条件（Normal、NoFan、UnderVolt）下收集的电流功率时序信号，经过分段、滑动窗口生成平衡数据集。

**📈 对比分析**

在保持同一 CNN 结构的前提下，比较窗口大小为 100、200、400 的模型准确率，结果显示窗口扩大至 400 时准确率从 83.78% 提升至 92.3%，并且 SHAP 对 Levels 概念的贡献更稳定，表明解释指导的改进有效。

**⚠️ 局限性**

局限性包括对高计算成本的 SHAP 精算方法的依赖、仅针对单一硬件平台和功率监测维度的实验、以及解释方法在不同 CPS 环境下可迁移性的未充分验证。

---

## 311. DTP: A Simple yet Effective Distracting Token Pruning Framework for Vision-Language Action Models

**arXiv ID:** 2601.16065 | [PDF](https://arxiv.org/pdf/2601.16065v1)

**作者:** Chenyang Li `[一作]` (Australian National University), Jingqun Tang `[通讯]` (Bytedance)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 Distracting Token Pruning (DTP) 的插拔式框架，动态识别并剔除视觉任务中无关的“干扰”图像 token，从而改进 Vision‑Language‑Action (VLA) 模型的动作生成。

**💡 创新点**

创新点在于：①以交叉注意力的相关性构建任务重要区域；②基于动作生成阶段的视觉注意力热图进行双向对比；③采用阈值 τ 的交集裁剪策略，自动平衡重要与无关区域，既不需额外训练也无需改动模型结构。

**🔧 技术方法**

使用的技术包括 Transformer 视觉编码器、注意力矩阵分析、嵌入相似度、Gaussian 平滑、可视化工具以及统计检验（Mann‑Whitney U）。

**📊 数据集**

主要在 SIMPLER Benchmark（WidowX 与 Google 机器人任务）和 LIBERO Benchmark 上评测，使用 SpatialVLA、Nora、UniVLA 三类主流 VLA 模型。

**📈 对比分析**

与原模型相比，DTP 在 SIMPLER 上平均提升 6.4%–12.5% 的成功率（相对提升 28%–107%），在 LIBERO 上提升 1.4%–6.6%；在 Google 机器人任务中也有 1–3% 的相对提升，实验结果表明该方法对不同模型、不同机器人均有效。

**⚠️ 局限性**

局限性包括：①需要手动选择 τ 以及重要层集合 C，需经验或超参数搜索；②仅针对基于 Transformer 的 VLA，未验证对非 Transformer 结构的适用性；③裁剪策略基于当前注意力，若模型内部注意力分布存在噪声，可能导致误裁剪。

---

## 312. Designing faster mixed integer linear programming algorithm via learning the optimal path

**arXiv ID:** 2601.16056 | [PDF](https://arxiv.org/pdf/2601.16056v1)

**作者:** Ruizhi Liu `[一作]` (Chinese Academy of Sciences), Dongbo Bu `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 2544 | [OpenAlex ID](https://openalex.org/A5076729236)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究并实现了一种基于深度学习的节点选择方法 DeepBound，利用多层特征融合和对比学习自动学习分支定界树中包含最优解的节点路径，从而加速混合整数线性规划求解。

**💡 创新点**

通过构建多层节点特征融合网络并采用对比学习的样本配对训练，显著解决传统手工启发式节点选择中的特征偏差与样本不平衡问题；同时实现了仅在节点选择阶段取代传统启发式规则的框架。

**🔧 技术方法**

采用多层特征融合网络、对比学习（pairwise ranking）策略、SHAP 等特征重要性分析，并与 SCIP 求解器（使用全强分支 FSB 规则）集成。

**📊 数据集**

在三类 NP‑hard MILP 基准（集合覆盖、组合拍卖、容量设施定位）上构造训练集，并分别准备与训练集同规模及更大规模的测试集，每组 100 个随机实例。

**📈 对比分析**

与 SCIP 默认最佳估计搜索（BES）及两种现有 ML 方法进行比较，评估指标包括总求解时间、最佳基准时间（bpb‑time）、最优间距收敛曲线和“最快求解实例”计数；DeepBound 在相同规模实例上平均缩短求解时间 30‑50%，bpb‑time 降至 BES 的一半，并在更大规模实例上获得更多胜利，显示出显著的加速与泛化效果。

**⚠️ 局限性**

仅改进节点选择阶段，依赖已解决实例的训练数据，对全新问题类型或极大规模 MILP 的适用性尚未充分验证；训练时需要对节点对进行配对，训练成本较高；对极端不平衡或非常深的分支树效果仍需进一步探测。

---

## 313. Can Platform Design Encourage Curiosity? Evidence from an Independent Social Media Experiment

**arXiv ID:** 2601.16040 | [PDF](https://arxiv.org/pdf/2601.16040v1)

**作者:** Marie Neubrander `[一作]` (Duke University), Alexander Volfovsky `[通讯]` (Duke University)

**通讯引用:** 2259 | [OpenAlex ID](https://openalex.org/A5056559023)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一个新研发的、含AI聊天机器人的独立研究平台上，对2282名美国成年人进行随机对照实验，测试在能源与气候话题讨论中通过社会规范和界面提示激发好奇心的干预，测量提问率、好奇度、毒性、参与量、平台体验和智识谦逊等指标。

**💡 创新点**

①提供了可控的、逼真的社交媒体仿真环境，克服了商业平台实验的可操作性难题；②首次用好奇心激发干预（规范+界面）检验其对问询行为、毒性和参与模式的因果影响；③证明好奇心激发可提升问询、降低毒性且不损害用户体验。

**🔧 技术方法**

构建了基于OpenAI GPT‑4‑turbo‑preview的AI机器人作为虚拟用户；使用自定义实验平台Spark Social；对文本进行Perspective API评价（好奇度、毒性）；采用混合效应模型和方差分析等统计方法。

**📊 数据集**

研究平台内生成的六个AI机器人（不同职业、党派、知识水平）产出的评论和贴文，以及2282名参与者在15分钟讨论中产生的贴文与评论；无公开现成数据集。

**📈 对比分析**

对照组与两种处理组比较，采用GLMM与Beta混合效应模型评估好奇度、毒性；单因素ANOVA和Tukey检验评估参与量；卡方检验评估用户满意度。结果显示：T1、T2相较于对照组，提问率提高约2.5‑3倍，好奇度提升约1.6‑1.8倍，毒性降低约10%；参与量略降但时间保持不变，用户体验无显著差异。

**⚠️ 局限性**

实验为15分钟受控情境，受试者为研究志愿者，未涉及真实社交网络的算法推送和情绪极化；好奇心激励效果在高情绪化或极端议题下尚未验证；没有检测长期效应或信念更新；AI机器人生成内容可能带偏差。

---

## 314. Mecellem Models: Turkish Models Trained from Scratch and Continually Pre-trained for the Legal Domain

**arXiv ID:** 2601.16018 | [PDF](https://arxiv.org/pdf/2601.16018v1)

**作者:** Özgür Uğur `[一作]` (NewmindAI), Ömer Can Sağbaş `[通讯]` (NewmindAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Mecellem 框架，通过从零预训练 ModernBERT 编码器和对 Qwen3 进行持续预训练（CPT）来构建土耳其法律领域专用语言模型，并在多阶段后训练和评估阶段进一步提升检索与嵌入质量。

**💡 创新点**

创新点包括：① 单阶段预训练结合下游检索评估的检查点挑选策略；② 四阶段持续预训练配合课程学习，缓解灾难性遗忘；③ 使用土耳其形态学过滤、SemHash 去重以及自研 MTEB‑Turkish 评测，提升模型在极端语言复杂度下的性能。

**🔧 技术方法**

技术手段涵盖：ModernBERT 架构（RoPE、RMSNorm、交替局部/全局注意力、8192 上下文）、MLM 预训练、InfoNCE / Qwen3‑InfoNCE、GISTEmbed、后训练对比学习、CPT 与课程学习、decoder‑to‑encoder 转换、MTEB‑Turkish 评测以及 Muhakim 多目标奖励模型。

**📊 数据集**

数据集主要包括土耳其法律文本（Yargıtay、Danıştay、YÖKTEZ 学位论文）、FineWeb2、CulturaX、MS MARCO‑TR、BGE‑M3 指导模型，以及多语言文本；预训练语料总计 112.7 B tokens，CPT 约 225 B tokens，按四阶段分布。

**📈 对比分析**

通过与 EmbeddingGemma‑300m、BAAI/bge‑m3 等基线在 MTEB‑Turkish 评测中对比，Mursit‑Base‑TR‑Retrieval（155 M）和 Mursit‑Large‑TR‑Retrieval（403 M）分别取得 55.86 与 56.43 的 MTEB 分数，法律检索得分 80.40 与 81.63，排名前 3；CPT 模型在各法律子领域的 perplexity 下降 36–43%；相比 SOTA 模型仅需单阶段预训练与低算力即可达到同等效果。

**⚠️ 局限性**

局限性包括：① 预训练缺乏多阶段 RetroMAE 等复杂管线，后续 fine‑tune 效果受限；② decoder‑to‑encoder 转换后的嵌入性能低于官方 Qwen3‑Embedding；③ 对超长文本的处理仍有限；④ 未对法律推理与生成质量进行深入评估。

---

## 315. PhysicsMind: Sim and Real Mechanics Benchmarking for Physical Reasoning and Prediction in Foundational VLMs and World Models

**arXiv ID:** 2601.16007 | [PDF](https://arxiv.org/pdf/2601.16007v1)

**作者:** Chak-Wing Mak `[一作]` (Peking University), Shanghang Zhang `[通讯]` (Peking University)

**通讯引用:** 9911 | [OpenAlex ID](https://openalex.org/A5013030532)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 PhysicsMind benchmark，统一评估视觉问答和视频生成两大任务，检验多模态模型在真实与仿真环境下对物理定律的理解与应用。

**💡 创新点**

创新点在于将物理定律（重心、杠杆平衡、牛顿第一定律）与跨模态评测结合，构建真实+仿真双模数据集，并设计针对性、法则一致的评估指标，弥补现有测试仅关注视觉质量或模板问题的不足。

**🔧 技术方法**

利用大规模多模态预训练模型进行评测，设计基于问题推理、物理量提取与动力学一致性的指标体系，并通过自动化与人工复核的标注流程保证数据质量。

**📊 数据集**

使用 PhysicsMind 数据集，包含 3 个经典力学场景（重心、杠杆平衡、牛顿第一定律）在现实桌面实验与 2D 仿真中的图像/短视频以及对应多选题。

**📈 对比分析**

通过准确率、IoU、速度/加速度一致性等多维指标对 20+ VLM 与 VGM 进行对比，结果显示大多数模型在物理推理与生成上仅靠视觉线索，常出现平衡、力矩或惯性违规，整体性能远低于人类基准。

**⚠️ 局限性**

局限在于仅关注刚体、无摩擦、短时、无碰撞的简单场景，未覆盖更复杂的物理现象；数据规模有限，评测多基于预训练模型，缺乏专门的物理驱动训练与长期交互测试。

---

## 316. Efficient Cloud-edge Collaborative Approaches to SPARQL Queries over Large RDF graphs

**arXiv ID:** 2601.15992 | [PDF](https://arxiv.org/pdf/2601.15992v1)

**作者:** Shidan Ma `[一作]` (Hunan University), Guo Chen `[通讯]` (Hunan University)

**通讯引用:** 19181 | [OpenAlex ID](https://openalex.org/A5000151307)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种将RDF图数据和SPARQL查询处理迁移到云边协同架构的方法，结合边缘服务器的数据局部化与网络调度，显著降低查询延迟。

**💡 创新点**

创新点在于：①引入“pattern‑induced subgraph”概念实现高效的数据局部化；②将查询分配与计算资源分配建模为混合整数非线性规划，并给出改进的分支定界求解方案；③针对边缘计算环境设计了基于最小DFS码的图同构索引。

**🔧 技术方法**

核心技术包括：图同构与子图同构匹配、最小DFS编码、混合整数非线性规划、Karush‑Kuhn‑Tucker条件求解、改进分支定界算法以及Gurobi求解器。

**📊 数据集**

实验使用两大公开RDF数据集：WatDiv（100M–500M三元组）和DBpedia（约1.1B三元组），以及QALD Benchmark生成的SPARQL查询。

**📈 对比分析**

与四种基线（Cloud‑Only、Random、Edge‑First、Greedy）比较，方法在多种场景下（存储、计算、带宽、服务器数量、图规模、用户数、查询选择性）均显著降低总响应时间，优势可达15–37%。

**⚠️ 局限性**

局限性包括：仅支持基本BGP查询，缺乏跨边缘通信与复杂查询（多模式、子查询）支持；求解时间随用户/服务器规模呈指数增长，需进一步优化或采用近似算法；数据局部化策略基于离线频率统计，可能对突发流量响应不足。

---

## 317. synthocr-gen: A synthetic ocr dataset generator for low-resource languages- breaking the data barrier

**arXiv ID:** 2601.16113 | [PDF](https://arxiv.org/pdf/2601.16113v1)

**作者:** Haq Nawaz Malik `[一作]`, Tanveer Ahmad Reshi `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了 SynthOCR-Gen，一个完全基于浏览器的、可复现的合成 OCR 数据集生成器，并公开发布了 600,000 词级克什米尔语 OCR 数据集。

**💡 创新点**

创新点包括：对右到左 Perso‑Arabic 脚本的完整支持、对克什米尔特有变音符号的保持、25 种可配置的图像降质增强、种子化的随机数生成器实现可重现性，以及多格式（CRNN、TrOCR、PaddleOCR、HuggingFace）兼容的输出。

**🔧 技术方法**

技术手段包括：Unicode 正规化与脚本验证、基于 Intl.Segmenter 的字形分段、Canvas 2D API 的字体渲染、可组合的图像变形/降质管道以及线性同余随机数生成器（LCG）实现可复现随机化。

**📊 数据集**

使用了 KS‑LIT‑3M 克什米尔语语料库（约 3.1M 词）作为文本来源，并配合 Noto Naskh Arabic、Gulmarg Nastaleeq、Scheherazade New 等字体进行渲染。

**📈 对比分析**

在 16GB RAM、Intel i7‑12700H 机器上，生成 600,000 样本耗时约 4.5 小时，平均 37 样本/秒；ZIP 归档约 8.7 GB；虽然未给出具体 OCR 识别准确率，但展示了高质量、可复现的数据生成速率与规模。

**⚠️ 局限性**

局限性包括仅支持印刷体文本、对手写体缺失、字体资源受限、浏览器内存限制导致大规模生成需分批或 CLI 模式，以及合成数据与真实扫描文档之间仍存在域差距。

---

## 318. SAMTok: Representing Any Mask with Two Words

**arXiv ID:** 2601.16093 | [PDF](https://arxiv.org/pdf/2601.16093v1)

**作者:** Yikang Zhou `[一作]` (Wuhan University), Xiangtai Li `[通讯]` (ByteDance)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SAMTok mask tokenizer，使 MLLM 能通过两枚离散文本 token 统一处理像素级掩码，实现掩码的输入输出与自然语言无缝对接。

**💡 创新点**

将任何区域掩码压缩为仅两枚离散文本 token，并利用文本匹配奖励实现纯文本强化学习，避免额外网络或任务特定损失，开创统一文本化掩码表达与 RL 的新范式。

**🔧 技术方法**

基于 SAM 的掩码编码/解码器，残差量化得到两枚 token；联合训练 209M 掩码；对 QwenVL 系列进行监督微调与 GRPO 强化学习；采用文本匹配奖励函数。

**📊 数据集**

使用 209M 掩码数据集（涵盖室内、户外、UI 等多场景）、约 5M 交互式 mask‑text 训练样本、GCG、GRES、RefCOCO、MR‑RefCOCO、GroundingSuite、DLC‑Bench 等多项评测数据。

**📈 对比分析**

与 LISA、GLaMM、OMG‑LLaVA、Sa2VA、SegLLM 等主流 MLLM 及专门掩码模型在 GCG、MR‑RefCOCO、GRES、RefCOCO 等基准上实现或逼近 SOTA，显著提升 AP、IoU、Recall、mIoU、N‑acc 等指标；RL 后 gIoU +6.8%、AP50 +4.5% 等进一步提升。

**⚠️ 局限性**

仅支持静态图像掩码，无法处理视频；只覆盖掩码，未涵盖点、线、框等视觉实体；需要在更多实体与视频任务上进一步扩展。

---

## 319. CLASP: An online learning algorithm for Convex Losses And Squared Penalties

**arXiv ID:** 2601.16072 | [PDF](https://arxiv.org/pdf/2601.16072v1)

**作者:** Ricardo N. Ferreira `[一作]` (NOVA School of Science and Technology), João Xavier `[通讯]` (Institute for Systems and Robotics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为CLASP的在线凸优化算法，处理凸损失与动态约束，目标是同时最小化累计损失与平方约束违约；

**💡 创新点**

在强凸情形下首次给出对齐的对数上界；使用一次投影提高内存效率；利用投影算子的firmly non‑expansive特性实现模块化分析；

**🔧 技术方法**

采用梯度下降+投影方法，调节步长以控制β；利用投影的FNE性质、凸/强凸分析、并扩展到多约束与持久约束；

**📊 数据集**

在合成在线线性回归实验中随机生成 H_t、A_t、b_t 等数据，未使用公开真实数据集；

**📈 对比分析**

与 AdaGrad、RECOO、Switch 三个基线对比，评估累计损失、线性违约(CCV_T,1)和平方违约(CCV_T,2)；CLASP在损失上略高于 AdaGrad，但违约控制与 RECOO 接近，且平方违约与最优基线相当；

**⚠️ 局限性**

需要已知可行集的投影可解；假设梯度有界；实验仅在合成数据上，缺乏真实场景验证；对持久约束的支持需额外改动投影集合。

---

## 320. ProGiDiff: Prompt-Guided Diffusion-Based Medical Image Segmentation

**arXiv ID:** 2601.16060 | [PDF](https://arxiv.org/pdf/2601.16060v1)

**作者:** Yuan Lin `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Suprosanna Shit `[通讯]` (University of Zurich)

**通讯引用:** 3119 | [OpenAlex ID](https://openalex.org/A5018632773)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出了基于提示的扩散模型ProGiDiff，用于多类别腹部器官分割；

**💡 创新点**

创新点在于将ControlNet反向使用，通过自定义图像编码器与预训练Stable Diffusion结合，实现自然语言提示驱动的分割，并支持多种解剖结构；

**🔧 技术方法**

主要技术包括Stable Diffusion预训练模型、ControlNet条件机制、定制化图像编码器、LoRA少量参数微调以及多候选分割的生成；

**📊 数据集**

使用BTCV CT数据集进行训练与评估，采用CHAOS MR数据集进行少量样本跨模态适配；

**📈 对比分析**

与多种2D/3D确定性基准（如nnUNet、TransUNet、Diff-UNet）对比，ProGiDiff在多数器官上达到了相近甚至更优的Dice与95% Hausdorff距离，并在使用oracle选取最佳候选时表现最佳；

**⚠️ 局限性**

局限性包括对器官边界不明显的小结构分割表现相对较弱、对MR图像的强度不均匀性敏感、以及需进一步验证在人机交互真实场景中的可行性。

---

## 321. From Harm to Healing: Understanding Individual Resilience after Cybercrimes

**arXiv ID:** 2601.16050 | [PDF](https://arxiv.org/pdf/2601.16050v1)

**作者:** Xiaowei Chen `[一作]` (Max Planck Institute for Security and Privacy), Yixin Zou `[通讯]` (Max Planck Institute for Security and Privacy)

**通讯引用:** 753 | [OpenAlex ID](https://openalex.org/A5037796481)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对18名西欧网络犯罪受害者的半结构化访谈，探讨其恢复过程并构建个体网络韧性概念。

**💡 创新点**

将情境敏感性、内部因素和外部支持三维框架整合到个体网络韧性模型，提出创伤知情支持的实践建议。

**🔧 技术方法**

采用主题分析（Braun & Clarke）对访谈转录文本进行编码与主题提炼，未使用机器学习或其他定量技术。

**📊 数据集**

使用18名受害者的访谈音频转录数据作为研究数据集。

**📈 对比分析**

该研究为定性研究，无对照实验或性能指标；通过主题饱和度评估分析完整性，无法量化比较。

**⚠️ 局限性**

样本规模小、仅限西欧地区、受访者自我报告易受偏差、未包含服务提供者或执法机构视角、缺乏量化衡量和干预验证。

---

## 322. Controlling Long-Horizon Behavior in Language Model Agents with Explicit State Dynamics

**arXiv ID:** 2601.16087 | [PDF](https://arxiv.org/pdf/2601.16087v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 323. RIS-Aided Cooperative ISAC Network for Imaging-Based Low-Altitude Surveillance

**arXiv ID:** 2601.16033 | [PDF](https://arxiv.org/pdf/2601.16033v1)

**作者:** Zhixin Chen `[一作]` (Southeast University), Shi Jin `[通讯]` (Southeast University)

**通讯引用:** 41489 | [OpenAlex ID](https://openalex.org/A5013079905)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

设计并评估了基于RIS的协作ISAC网络用于低空侦察，并提出基于压缩感知的成像方法与子空间追踪（SP）算法；

**💡 创新点**

创新点在于引入主动RIS（ARIS）放大低空信号并给出CRLB理论，证明在相同功耗下ARIS优于被动RIS（PRIS），可实现最高300 m高度的有效侦察；

**🔧 技术方法**

采用RIS（主动与被动）辅助ISAC、压缩感知成像、SP算法、CRLB分析与仿真对比；

**📊 数据集**

使用仿真生成的随机稀疏目标分布（S=10，Gaussian散射系数），未使用公开数据集；

**📈 对比分析**

通过在相同总功率条件下比较MSE、检测率、PSNR/功率等指标，ARIS系统在不同高度均优于PRIS，误差接近CRLB；

**⚠️ 局限性**

局限性包括仅在理想仿真环境下验证，未考虑多径干扰、RIS相干性误差及真实噪声，且RIS位置与功率分配仅给出经验建议，未做全局最优优化。

---

## 324. Improve the autonomy of the SE2(3) group based Extended Kalman Filter for Integrated Navigation: Theoretical Analysis

**arXiv ID:** 2601.16062 | [PDF](https://arxiv.org/pdf/2601.16062v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 325. Sawtooth Wavefront Reordering: Enhanced CuTile FlashAttention on NVIDIA GB10

**arXiv ID:** 2601.16032 | [PDF](https://arxiv.org/pdf/2601.16032v1)

**作者:** Yifan Zhu `[一作]` (University of Rochester), Chen Ding `[通讯]` (University of Rochester)

**通讯引用:** 15097 | [OpenAlex ID](https://openalex.org/A5100663432)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对 NVIDIA GB10 上 CuTile 实现的 FlashAttention 进行 L2 缓存行为分析，并提出 Sawtooth Wavefront Reordering（锯齿波前重排）以显著降低 L2 非强制缺失

**💡 创新点**

通过对 L2 缓存命中率与活跃 SM 数量的关系建模，发现传统循环访问导致 KV 缓存重用距离过大，进而提出锯齿式扫描以缩短重用距离，实现 50%–67% 的缺失率下降

**🔧 技术方法**

CUDA 与 CuTile 编程模型、硬件性能计数器、理论模型推导、锯齿式循环重排

**📊 数据集**

在实验中使用合成序列长度 32k、128k 以及 128×1024 长度的 KV/ Q/V 张量（无真实数据集）

**📈 对比分析**

与传统循环访问进行对比，使用 L2 缓存缺失计数和吞吐量衡量；CUDA 版缺失率下降 50% 以上，吞吐量从约 1.3 TFLOPS 提升至 2.4 TFLOPS；CuTile 版缺失率下降 67% 以上，吞吐量从 61 TFLOPS 提升至 69 TFLOPS（非因果掩码）或从 41 TFLOPS 提升至 66 TFLOPS（因果掩码）

**⚠️ 局限性**

仅在共享内存可容纳的较小 tile 下有效；对较大 tile（如 128×128）时 CuTile 编译器会拆分 tile，导致访问模式改变；验证仅在 GB10 架构，未覆盖其他 GPU 体系结构或真实数据集

---

## 326. Adapter Fusion for Multilingual Text2Cypher with Linear and Learned Gating

**arXiv ID:** 2601.16097 | [PDF](https://arxiv.org/pdf/2601.16097v1)

**作者:** Makbule Gulcin Ozsoy `[一作]` `[通讯]` (Neo4j), Makbule Gulcin Ozsoy (Neo4j)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

训练了面向英语、西班牙语和土耳其语的LoRA适配器，并通过线性合并与融合MLP动态路由两种方式将它们组合，构建了可增量扩展的多语言 Text2Cypher 系统。

**💡 创新点**

创新点在于：①利用LoRA适配器实现每种语言的轻量化微调；②引入融合MLP动态门控，恢复约 75% 传统联合多语言微调的性能；③通过仅训练少量额外数据即可实现新语言的增量加入，显著降低了计算成本。

**🔧 技术方法**

主要技术包括：LoRA 参数高效微调；Fusion MLP 动态路由（利用预览 logits 进行加权）；线性融合（Task Arithmetic）；Meta‑Llama‑3.1‑8B‑Instruct 基础模型；ROUGE‑L 与 Exact‑Match 评估。

**📊 数据集**

使用了公开的多语言 Text2Cypher 基准数据集，涵盖英语、西班牙语和土耳其语，每种语言约 12k 训练样本，4,783 条测试样本。

**📈 对比分析**

通过与基线、单语微调、联合多语言微调对比实验，Fusion MLP 在三种语言上的平均 ROUGE‑L 为 0.79（线性融合为 0.75），相较于联合微调提升了约 75% 的性能，并在 Exact‑Match 评估中也优于线性融合；训练成本大幅降低，尤其在增量添加新语言时。

**⚠️ 局限性**

局限性：仅在三种语言上验证，未评估不同域、图模式或查询复杂度；未对推理时动态路由的计算开销做系统分析；未尝试其他更先进的适配器融合方法。

---

## 327. DSFedMed: Dual-Scale Federated Medical Image Segmentation via Mutual Distillation Between Foundation and Lightweight Models

**arXiv ID:** 2601.16073 | [PDF](https://arxiv.org/pdf/2601.16073v1)

**作者:** Hanwen Zhang `[一作]` (Peking University), Guibo Luo `[通讯]` (Peking University)

**通讯引用:** 416 | [OpenAlex ID](https://openalex.org/A5057015645)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出DSFedMed双尺度联邦学习框架，实现中心化基础模型与轻量级客户端模型之间的相互知识蒸馏，解决医疗影像分割在联邦环境下算力与通信瓶颈。

**💡 创新点**

(1) 采用可控医学图像生成（ControlNet）替代真实公共数据；(2) 引入学习可导引的样本选择策略，动态挑选信息量大、监督可靠的样本进行双向蒸馏；(3) 通过双尺度协同实现全局语义通用与局部域适应的平衡。

**🔧 技术方法**

使用基础模型SAM（ViT‑B/16）与轻量化TinySAM；ControlNet+Stable Diffusion生成器；联邦平均聚合；双向知识蒸馏；学习可导引样本评分（GT损失+KL分歧）。

**📊 数据集**

五个医疗分割数据集：Fundus（OD/OC），Prostate（MRI），Nuclei（PanNuke），ISIC（皮肤病变），CHAOS（CT/MRI 肝脏），全部按非IID划分构成联邦环境。

**📈 对比分析**

与中心化SAM、FedSAM、FedMSA、FedU‑Net、FednnU‑Net、FedTinySAM等基线对比，DSFedMed平均Dice提升约3.8%（vs FedTinySAM），1.4%（vs 中央化SAM），1.9%（vs FedSAM/FedMSA）；通信开销下降≈88%，推理时间0.015s，整体效率显著优于现有方法。

**⚠️ 局限性**

数据生成阶段仍耗时；缺乏真实临床验证；多模态扩展尚未完成；对超参数的敏感性需进一步研究。

---

## 328. AgriPINN: A Process-Informed Neural Network for Interpretable and Scalable Crop Biomass Prediction Under Water Stress

**arXiv ID:** 2601.16045 | [PDF](https://arxiv.org/pdf/2601.16045v1)

**作者:** Yue Shi `[一作]` (Manchester Metropolitan University), Frank Ewert `[通讯]` (Leibniz Centre for Agricultural Landscape Research)

**通讯引用:** 32840 | [OpenAlex ID](https://openalex.org/A5072022077)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出了一种过程信息化神经网络（AgriPINN），将LINTUL5作物生长微分方程嵌入深度学习框架中，用于在水分胁迫条件下预测作物地表生物量（AGB）；

**💡 创新点**

创新点在于：①将生理过程方程作为可微软约束加入损失函数，实现模型在保持高精度的同时保持生理一致性；②无监督恢复关键生理变量（LAI、PAR、RUE、水胁迫因子），提高解释性；③通过预训练+微调策略在大规模历史模拟数据和现场实验数据上实现稳健迁移。

**🔧 技术方法**

技术手段包括：卷积神经网络骨干；过程约束损失（Tikhonov型残差）与标准MSE并行训练；自微分计算生物量增量；使用SGD/Adam优化器；预训练60年历史数据、三年水胁迫实验微调；对超参数λ进行网格搜索。

**📊 数据集**

数据集为：①65年德国地区（397区）基于SIMPLACE的冬小麦/玉米模拟数据（气候、土壤、管理）；②2016–2018三年控制水分处理现场实验；③ERA5气候统计+MODIS NDVI用于高分辨率空间预测；④LINTUL5模拟输出作为基准。

**📈 对比分析**

与三种SOTA深度模型（ConvLSTM‑ViT、SLTF、CNN‑Transformer）及过程模型LINTUL5对比，AgriPINN在RMSE上降低约43%、R²/CC提升显著；计算效率提升8倍，推理速度最快；在不同水分处理、全国空间分布及不同网络骨干上均表现出更低误差、更高相关性，并保持对OOV的稳健性。

**⚠️ 局限性**

局限性包括：①过程约束依赖LINTUL5方程，若方程简化或有偏差会影响结果；②目前仅考虑水分胁迫，未覆盖养分、病虫害等多重胁迫；③需要大规模历史模拟数据进行预训练；④超参数（λ）需人工调节，适用性需进一步验证。

---

## 329. Predicting Healthcare System Visitation Flow by Integrating Hospital Attributes and Population Socioeconomics with Human Mobility Data

**arXiv ID:** 2601.15977 | [PDF](https://arxiv.org/pdf/2601.15977v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 330. Keyframe-Based Feed-Forward Visual Odometry

**arXiv ID:** 2601.16020 | [PDF](https://arxiv.org/pdf/2601.16020v1)

**作者:** Weichen Dai `[一作]` (Hangzhou Dianzi University), Wanzeng Kong `[通讯]` (Hangzhou Dianzi University)

**通讯引用:** 4076 | [OpenAlex ID](https://openalex.org/A5003649465)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

基于视觉基础模型 VGGT，提出一种键帧驱动的前馈视觉里程计（VO）框架，并通过强化学习自适应地决定键帧，从而提升定位精度。

**💡 创新点**

创新点在于将强化学习用于键帧选择，使决策与高维潜在表示协同工作，摒弃传统基于几何规则的手工阈值，形成数据驱动的键帧策略。

**🔧 技术方法**

技术手段包括：VGGT 视觉基础模型、DINOv2 提取 CLS 令牌、滑动窗口滑动策略、PPO 强化学习代理、Umeyama 对齐与误差奖励、DPT 深度辅助等。

**📊 数据集**

训练使用 TartanAir 合成数据集；评估跨 3 个真实世界数据集：EuRoC、TUM‑RGBD 和 KITTI。

**📈 对比分析**

与 VGGT‑Long、VGGT‑SLAM、FastVGGT、StreamVGGT、InfiniteVGGT 以及基线 VGGT‑SW/LK 进行对比，实验显示该方法在 ATE 方面大幅提升或与最优方法持平，展现出强泛化能力。

**⚠️ 局限性**

局限性：目前仅保留短期键帧信息，缺乏长期记忆/闭环机制，未来需进一步构建完整的前馈 SLAM 框架。

---

## 331. PUMA: Perception-driven Unified Foothold Prior for Mobility Augmented Quadruped Parkour

**arXiv ID:** 2601.15995 | [PDF](https://arxiv.org/pdf/2601.15995v1)

**作者:** Liang Wang `[一作]` (Zhejiang University), Qiuguo Zhu `[通讯]` (Zhejiang University)

**通讯引用:** 553 | [OpenAlex ID](https://openalex.org/A5059414395)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种端到端学习框架PUMA，使四足机器人通过深度相机感知地形，估计本体极坐标踏点先验并将其作为运动引导，实现无显式踏点跟踪的柔性跑酷运动。

**💡 创新点**

创新点包括：① 将极坐标踏点先验融入演员网络而非传统的显式踏点追踪；② 采用概率退火选择（PAS）在训练过程中逐步用估计踏点替代真实踏点；③ 采用多评论家（Multi‑Critic）分别估计任务奖励与踏点奖励，提升学习稳定性与效率。

**🔧 技术方法**

使用的技术：异步演员‑评论家架构、PPO、CNN+自注意力+GRU的踏点估计网络、PAS策略、Multi‑Critic奖励分离、域随机化、基于深度渲染的Isaac Gym模拟、在真实机器人上部署RK3588与Intel RealSense D435i深度相机。

**📊 数据集**

训练与评估数据：在Isaac Gym中随机生成三类离散地形（Wall‑assisted Gap、Surmounting、Stepping Stones）进行模拟；真实实验使用Lite3四足机器人搭载RealSense摄像头采集视觉数据；未使用公开数据集。

**📈 对比分析**

与基线PIE、Extreme Parkour以及各消融实验（无踏点先验、无相对距离、显式/隐式笛卡尔先验、无PAS、无Multi‑Critic）对比。PUMA在所有地形上均达≈100%成功率，Traverse Rate>95%，比基线提升20–50%；Multi‑Critic与PAS显著加速收敛并提升成功率。

**⚠️ 局限性**

局限性：仅利用几何深度信息，未考虑语义或材质；训练环境为静态，缺乏对动态或柔性地形的适应；在噪声环境下踏点估计不稳定；在极端倾斜或极宽空隙等极端场景仍可能失败。

---

## 332. Partially Lazy Gradient Descent for Smoothed Online Learning

**arXiv ID:** 2601.15984 | [PDF](https://arxiv.org/pdf/2601.15984v1)

**作者:** Naram Mhaisen `[一作]` (Delft University of Technology), George Iosifidis `[通讯]` (Delft University of Technology)

**通讯引用:** 3917 | [OpenAlex ID](https://openalex.org/A5044138533)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种名为k‑lazyGD的在线学习算法，能够在平滑在线凸优化中同时兼顾动态回溯误差与移动成本，且在任意缓冲参数k下实现最优动态回报。

**💡 创新点**

核心创新在于通过FTRL‑修剪框架，将梯度累积与局部“懒惰”机制结合，首次证明了在满足k=Θ(√(T/P_T))的范围内，既能保持低移动成本，又能实现最优的动态回报。

**🔧 技术方法**

技术上主要利用了：1）FTRL（Follow‑the‑Regularized‑Leader）与投影更新的等价性；2）梯度累积的“修剪”策略；3）对迭代的“停滞”与“稳定”性质的几何分析；4）动态FTRL不等式的改写以包含移动成本。

**📊 数据集**

该工作为理论性研究，无需使用公开数据集，全部结论基于凸分析与信息理论证明。

**📈 对比分析**

与传统OGD（k=1）和全懒惰GD（k=T）对比，k‑lazyGD在保持动态回报≈√((P_T+1)T)的同时，将移动成本显著降低到O(kG/σ+R^2/σ)，在极端情况k=√(T/P_T)时实现了最优的两项平衡。

**⚠️ 局限性**

主要局限包括：1）需要预知或通过元学习估计路径长度P_T；2）假设损失函数Lipschitz且可投影；3）仅在欧几里得范数下给出完整证明，非欧几里得情形仍待扩展；4）理论分析相对复杂，实际实现与参数调优仍需进一步验证。

---

## 333. Multimodal Climate Disinformation Detection: Integrating Vision-Language Models with External Knowledge Sources

**arXiv ID:** 2601.16108 | [PDF](https://arxiv.org/pdf/2601.16108v1)

**作者:** Marzieh Adeli Shamsabad `[一作]` (CRIM), Hamed Ghodrati `[通讯]` (CRIM)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究者提出一种将视觉语言模型（VLM）与多源外部知识（逆向图像搜索、可信事实核查站点、Google 搜索、GPT web 预览）按优先级集成的系统，以提升对气候相关多模态误信息的检测能力。

**💡 创新点**

创新点在于将多种外部检索来源（尤其是 GPT 预览和逆向图像搜索）与 VLM 结合，形成条件引入的多模态推理框架，使模型在仅靠内部知识时难以处理的新近事件能得到及时、可靠的事实核查支持。

**🔧 技术方法**

技术方案包括使用 GPT‑4o 作为核心推理模型，配合 Chain‑of‑Draft 与 Chain‑of‑Thought 生成式推理策略；外部检索模块涵盖逆向图像搜索、Google 关键字搜索、GPT 生成的网页预览及专业事实核查网站；并采用多源优先级融合机制。

**📊 数据集**

使用的主要数据集为 CliME（Climate Change Multimodal Evaluation），从中抽取 500 条图文对，并通过 GPT‑4o 自动标注四类（准确、误导、错误、无法验证）和两类（准确、错误信息）标签。

**📈 对比分析**

通过对单源、组合源以及内部知识三种设置的比较，采用 Accuracy、Macro‑F1、拒绝率和置信度等指标，结果显示组合源在四类设置下准确率达 86.45%、Macro‑F1 91.38%，相较于仅使用内部知识提升约 4–5%，且拒绝率为 0%。

**⚠️ 局限性**

限制方面包括外部检索的噪声与成本（尤其逆向图像搜索需大量 token）、模型在某些类别仍易混淆、验证仅基于 CliME 数据集，缺乏更大规模、多语言或实时环境下的进一步测试。

---

## 334. Neural Particle Automata: Learning Self-Organizing Particle Dynamics

**arXiv ID:** 2601.16096 | [PDF](https://arxiv.org/pdf/2601.16096v1)

**作者:** Hyunsoo Kim `[一作]` (Korea Advanced Institute of Science and Technology), Jinah Park `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2382 | [OpenAlex ID](https://openalex.org/A5100769319)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种新型的自组织粒子动力学框架 Neural Particle Automata (NPA)，能够通过共享的局部神经更新规则在连续空间中驱动粒子集自发形成复杂形态、合成纹理以及实现点云分类。

**💡 创新点**

创新点在于将神经细胞自动机 (NCA) 的 Eulerian 网格视角迁移到 Lagrangian 粒子系统，采用可微分的 Smoothed Particle Hydrodynamics (SPH) 作为网格无关的感知模块，并通过 CUDA 加速实现大规模可训练的粒子系统，同时保留了 NCA 的局部性、鲁棒性和可扩展性。

**🔧 技术方法**

主要技术包括：
- 可微分 SPH 运算（密度、平滑、梯度、矩阵等）
- 共享 MLP 更新网络（可更新状态或状态+位置）
- CUDA 自定义核加速、hash‑grid 空间索引
- 训练时的随机采样、池化、正则化（平滑、平移、速度约束）
- 渲染解码（高斯 splatting、GSplat）与多尺度损失（SSIM、VGG 纹理损失）。

**📊 数据集**

使用的公开数据集包括：
- 60 个 2D Emoji 目标形状（用于 2D 形态生成）
- 3D NeRF 合成数据集（用于 3D 形态生成）
- 生成的 RGBA 纹理样本（用于纹理合成）
- PointMNIST（随机采样点云，用于分布式分类）。

**📈 对比分析**

与传统 NCA、图神经网络、粒子生命等方法对比，NPA 在多任务上展现出与 NCA 相当甚至更优的鲁棒性、可恢复性和自组织能力；在 2D/3D 形态生成中实现了更好的细节捕捉；在纹理合成中通过 VGG 纹理损失得到更真实的纹理；在点云分类中达到了 98.4% 的准确率，优于单纯 MLP 或传统 GNN。实验还表明 NPA 对粒子数、SPH 半径、更新概率等超参具有良好的泛化与容错性。

**⚠️ 局限性**

主要局限：
- 无法实现粒子合并/分裂，导致在目标区域过小时粒子过密、计算开销上升；
- SPH 的平滑性限制了对极细纹理与几何细节的捕捉；
- 训练过程对学习率、正则化等超参高度敏感，调参成本较大。

---

## 335. Delayed Assignments in Online Non-Centroid Clustering with Stochastic Arrivals

**arXiv ID:** 2601.16091 | [PDF](https://arxiv.org/pdf/2601.16091v1)

**作者:** Saar Cohen `[一作]` `[通讯]` (Bar Ilan University), Saar Cohen (Bar Ilan University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了在线无中心聚类（online non‑centroid clustering）与延迟决策的新模型，并给出了贪心算法 DGreedy 以实现点的聚类；

**💡 创新点**

在未知 i.i.d.（UIID）到达模型下证明了该算法相对于最优离线聚类的常数比值（ratio‑of‑expectations），突破了传统最坏情况下的对数级不可逼近界；

**🔧 技术方法**

采用几何分布、期望与方差分析、贪心匹配策略以及对延迟成本和距离成本的联合上界/下界证明；

**📊 数据集**

本文完全是理论分析，没有使用具体实验数据集；

**📈 对比分析**

通过与最优离线解的期望成本比较，证明 DGreedy 的 RoE 在极限下不超过 8/(1‑e⁻²)（约 9.3），即常数级性能；

**⚠️ 局限性**

局限性：仅适用于有限度量空间、固定簇大小、未知 i.i.d. 到达、线性延迟成本，且未给出实验验证；对非 i.i.d.、动态分布或可重新分配的场景仍需进一步研究。

---

## 336. Towards a Goal-Centric Assessment of Requirements Engineering Methods for Privacy by Design

**arXiv ID:** 2601.16080 | [PDF](https://arxiv.org/pdf/2601.16080v1)

**作者:** Oleksandr Kosenkov `[一作]` (fortiss GmbH), Daniel Mendez `[通讯]` (fortiss GmbH)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文基于对GDPR下隐私设计（PbD）的需求工程（RE）方法进行目标中心评估，先通过文献综述和访谈识别方法特性与组织目标，然后构建并验证一个以GQM为灵感的评估框架。

**💡 创新点**

创新点在于将评估焦点从传统的过程或产品特性转向组织层面的目标，提出了一套可定制、分层（概念、操作、量化）的评估方法，并首次系统化地探讨了RE方法与PbD目标之间的关联。

**🔧 技术方法**

使用的技术包括系统性文献综述、半结构化访谈、主题分析、GQM方法结构化、以及对访谈数据和文献指标的定量和定性合成。

**📊 数据集**

使用的数据集主要是：1）2260篇与GDPR、软件工程或需求工程评估相关的原始研究；2）15名实践者的访谈记录；3）对6篇新提出PbD方法的系统评估标准与特征的提取。

**📈 对比分析**

在验证阶段，评估框架的32个子目标、148个问题、172个指标分别被评为90–100%、92–99%和87–99%有用，可行性分别为50–100%、66–95%和44–92%；虽然未给出传统方法的直接性能对比，但结果表明该框架在实务中的可接受度和实用性较高。

**⚠️ 局限性**

局限性包括：样本量仅15名访谈参与者，且大多为行业内部人士；评估采用二元（有用/不可用、可行/不可行）评价，缺乏更细粒度或客观指标；框架尚处于初步验证阶段，未进行长期或跨组织的实测；未提供完整的使用指南和工具支持。

---

## 337. Grounding Large Language Models in Reaction Knowledge Graphs for Synthesis Retrieval

**arXiv ID:** 2601.16038 | [PDF](https://arxiv.org/pdf/2601.16038v1)

**作者:** Olga Bunkova `[一作]` (Delft University of Technology), Jana M. Weber `[通讯]` (Delft University of Technology)

**通讯引用:** 569 | [OpenAlex ID](https://openalex.org/A5001961922)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了如何利用大型语言模型生成可执行的Cypher查询，以在化学反应知识图谱中检索单步和多步合成路径，并提出了基于检查表的自我校正循环。

**💡 创新点**

将Text2Cypher方法专门应用于化学反应KG，评估零样本与一示例提示效果，证明一示例提示显著提升检索准确率，并提供可复现的评估框架。

**🔧 技术方法**

采用大型语言模型（如GPT‑4）与Prompt工程；零样本/一示例提示；嵌入式示例检索；CoVe式检查表验证‑纠正循环；Neo4j执行Cypher查询。

**📊 数据集**

基于USPTO公开反应数据库，筛选并标准化约5万条SMILES反应，构建双边图谱。

**📈 对比分析**

通过BLEU、METEOR、ROUGE‑L评估生成查询的文本相似度，并用微平均Precision/Recall/F1衡量检索结果；实验表明一示例提示相比零样本提升约30‑50%F1，尤其在多步任务中尤为显著；CoVe在已提供示例时提升有限。

**⚠️ 局限性**

检索准确性仍受示例匹配度限制；检查表通用性不足，未覆盖所有任务特定错误；实验仅在单一LLM与中等规模KG上验证，缺乏跨模型与更大KG的泛化评估。

---

## 338. Tri-Hybrid Beamforming Design for integrated Sensing and Communications

**arXiv ID:** 2601.16036 | [PDF](https://arxiv.org/pdf/2601.16036v1)

**作者:** Tianyu Fang `[一作]`, Nhan Thanh Nguyen `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并联合优化三合一混合波束成形体系结构（tri‑HBF），以提升毫米波/太赫兹ISAC系统的通信SNR和雷达探测功率。

**💡 创新点**

创新点在于：①将可编程等离子体天线（DMA）作为第三波束域，实现更大天线数量与更低RF链消耗；②提出闭式迭代算法（结合Dinkelbach变换与交替优化），显著降低计算复杂度并保证收敛；③在同等硬件条件下，tri‑HBF实现更高能效与更佳空间增益。

**🔧 技术方法**

使用的技术包括：多目标加权优化、Dinkelbach单比值变换、交替优化（AO）与闭式更新、硬件约束下的单位幅值与Lorentzian约束处理。

**📊 数据集**

采用的“数据集”为：基于Saleh–Valenzuela散射模型的随机多径信道、仿真生成的DMA波导间距、辐射元件间距以及随机选取的波束指向角（θ、ϕ）。

**📈 对比分析**

与全数字（FD）和传统混合波束（HBF）在同天线数与同开口面积两种配置进行比较；结果显示 tri‑HBF 在通信速率、能效和雷达功率上均优于基准方案，且收敛速度快、计算复杂度低。

**⚠️ 局限性**

局限性：仅考虑单用户单目标、已知CSI、单RF链配置；DMA硬件约束（波导损耗、子阵结构）在实际部署中可能导致更严苛的性能限制；未讨论多用户/多目标情况和实际CSI估计误差。

---

## 339. Data-Driven Conditional Flexibility Index

**arXiv ID:** 2601.16028 | [PDF](https://arxiv.org/pdf/2601.16028v1)

**作者:** Moritz Wedemeyer `[一作]`, Manuel Dahmen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出条件柔性指数(CFI)，利用历史数据与情境信息构造可接受不确定性集合，并将其嵌入鲁棒调度优化中。

**💡 创新点**

创新点在于使用条件归一化流学习可接受不确定性集合，并通过其概率保持特性将灵活性指数解释为可行概率下界。

**🔧 技术方法**

采用条件归一化流（RealNVP）与半无限规划的自适应离散化求解技术，实现对复杂分布的建模与优化。

**📊 数据集**

数据集包括二维“两月”与“环形”模拟数据，以及基于德国电网的可再生发电历史记录（2013‑2017年）。

**📈 对比分析**

与传统超立方体集合对比，CFI在示例中获得更高条件覆盖率；在安全约束单位承诺（SCUC）案例中，加入时间上下文的CFI将可行调度率提升至91%，优于无时间信息模型。

**⚠️ 局限性**

局限在于归一化流的容量与维度导致求解复杂度显著增加；对多模态或包含空洞的分布拟合效果有限。

---

## 340. EAIFD: A Fast and Scalable Algorithm for Incremental Functional Dependency Discovery

**arXiv ID:** 2601.16025 | [PDF](https://arxiv.org/pdf/2601.16025v1)

**作者:** Yajuan Xu `[一作]` (Harbin Institute of Technology), Xiaolong Wan `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 64 | [OpenAlex ID](https://openalex.org/A5085267247)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了EAIFD算法，能够在增量数据库更新时高效地发现最小非平凡功能依赖。

**💡 创新点**

创新点在于将增量FD发现转化为部分超图的最小冲突集枚举，并结合多属性哈希表与双步验证策略，实现了快速初始化、可扩展性和低内存消耗。

**🔧 技术方法**

主要技术包括基于差集构建的超图、MMCS最小打通集枚举、迭代分组哈希验证(IGHV)、多属性哈希表(MHT)以及高频映射保留策略。

**📊 数据集**

实验使用20个真实数据集，涵盖从数十到数百万行、从5到54列，包含Flights、Census、Plista等数据集。

**📈 对比分析**

与DynFD、DHSFD及FDHITS对比，EAIFD在大多数数据集上速度提升1-2个数量级，内存下降两数量级；在增量比例较小时即可优于静态算法。

**⚠️ 局限性**

局限性包括目前仅支持插入更新，无法处理删除或修改；对极大增量比例时性能退化；并且对高频阈值θ的选择仍需经验。

---

## 341. PAINT: Pathology-Aware Integrated Next-Scale Transformation for Virtual Immunohistochemistry

**arXiv ID:** 2601.16024 | [PDF](https://arxiv.org/pdf/2601.16024v1)

**作者:** Rongze Ma `[一作]` (Northwestern Polytechnical University), Yong Xia `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 18643 | [OpenAlex ID](https://openalex.org/A5100670074)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了PAINT，一种将虚拟IHC视为结构条件生成任务的视觉自回归框架，利用H&E图像的形态信息引导IHC合成。

**💡 创新点**

创新点在于引入双约束的3S‑Map（Spatial Structural Start Map）作为形态对齐的先验，并采用下一尺度自回归生成策略实现从粗到细的结构保留与细节补全。

**🔧 技术方法**

使用多尺度VQ‑VAE对H&E与IHC进行离散编码、U‑Net跨模态翻译网络完成特征对齐，并在Transformer中实现自回归采样与AdaLN全局条件调制。

**📊 数据集**

实验数据集包括IHC4BC（ER、PR、HER2、Ki67）和MIST（四种生物标志物）两套H&E–IHC配对样本。

**📈 对比分析**

与Pix2Pix、CycleGAN、CUT、ASP、PPT、PSPStain、SynDiff、HistDiST等基线比较，PAINT在SSIM、PSNR、LPIPS、HER2分类ACC/AUC、ER/PR/Ki67评分等指标均显著优于对手，验证了结构条件自回归方法的有效性。

**⚠️ 局限性**

主要局限是VQ‑VAE的离散编码导致信息瓶颈，对细粒度重建有限；需要预先进行H&E–IHC配准；以及对训练数据规模与多模态配对质量较为敏感。

---

## 342. Prioritizing Configuration Relevance via Compiler-Based Refined Feature Ranking

**arXiv ID:** 2601.16008 | [PDF](https://arxiv.org/pdf/2601.16008v1)

**作者:** Federico Bruzzone `[一作]` (Universita degli Studi di Milano), Luca Favini `[通讯]` (Universita degli Studi di Milano)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于编译器的配置优先级排序方法，并实现工具RustyEx以自动生成最具代表性的配置集

**💡 创新点**

首次结合AST到统一中间表示（UIR）提取特性依赖图与原子依赖树，并用图中心性度量与SAT求解器相结合实现可验证的配置优先级

**🔧 技术方法**

编译器插桩、静态分析、构造加权有向图和树、图中心性计算（如Closeness、Betweenness、Eigenvector、Katz）、CNF转换、SAT求解

**📊 数据集**

对40个高排名开源Rust项目（共约1.6k个crate）进行评估，采集特性数量、图大小、执行时间与内存等指标

**📈 对比分析**

与基准（全量配置）对比，RustyEx在93%项目内完成分析，平均耗时333s、峰值内存885MB；生成配置数可控且均为有效配置，显著降低分析规模并保持正确性

**⚠️ 局限性**

中心性度量对分离组件的处理可能导致偏差；未提供真实“重要”配置基准；对非常大或复杂的项目可能因依赖缺失导致失败；需要手动调整配置预算

---

## 343. Benchmarking Deep Learning Models for Raman Spectroscopy Across Open-Source Datasets

**arXiv ID:** 2601.16107 | [PDF](https://arxiv.org/pdf/2601.16107v1)

**作者:** Adithya Sineesh `[一作]` (Purdue University), Akshita Kamsali `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对五种深度学习模型在三份公开Raman光谱数据集上的分类任务进行了系统基准测试。

**💡 创新点**

创新点在于首次在统一实验框架下对专门为Raman设计的深度学习模型进行跨数据集比较，并揭示分布漂移对性能的影响。

**🔧 技术方法**

采用的技术包括一维卷积网络、SANet、RamanNet、Transformer和RamanFormer，并统一使用Adam、交叉熵等训练策略。

**📊 数据集**

使用的数据集包括矿物光谱集MLROD、细菌识别集Bacteria‑ID以及药物成分集API。

**📈 对比分析**

通过统一的超参搜索、验证选拔和测试，评估了准确率与宏F1；结果显示SANet在三组数据集上整体最佳，Transformer最低，分布漂移导致测试准确率显著下降。

**⚠️ 局限性**

局限性包括仅覆盖五种模型、仅使用最小预处理、数据集规模有限且对不同仪器的泛化能力仍需进一步验证。

---

## 344. Characterizations of monadically dependent tree-ordered weakly sparse structures

**arXiv ID:** 2601.16039 | [PDF](https://arxiv.org/pdf/2601.16039v1)

**作者:** Hector Buffière `[一作]` (Université Paris Cité), Sebastian Siebertz `[通讯]` (University of Bremen)

**通讯引用:** 856 | [OpenAlex ID](https://openalex.org/A5073548944)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

本文研究了具有树序（tree-ordered）且弱稀疏（weakly sparse）的图的逻辑性质，提出了关于其单调依赖性（monadic dependence）的完整结构化描述；

**💡 创新点**

创新点在于将单调依赖性转化为可判定的“核心（core）”结构集合，并将其推广到所有弱稀疏树序图类，提供了三种类型核心的完整分类；

**🔧 技术方法**

采用了组合与结构化模型论技术，包括正则化、twister/clean twister定义、星形（star）与匹配（matching）分类、稀疏图的子图与图的分割理论、以及Ramsey定理与多层归约；

**📊 数据集**

由于论文为理论性研究，未使用任何数据集；

**📈 对比分析**

与传统方法对比不适用实验指标，主要通过逻辑归约与图同构判定的多项式时间实现来评估可行性；

**⚠️ 局限性**

局限性在于该框架仅适用于弱稀疏树序图，且核心分类复杂度高，处理大规模结构时可能产生计算量增长。

---

## 345. Why Can't I Open My Drawer? Mitigating Object-Driven Shortcuts in Zero-Shot Compositional Action Recognition

**arXiv ID:** 2601.16211 | [PDF](https://arxiv.org/pdf/2601.16211v1)

**作者:** Geo Ahn `[一作]` (Kyung Hee University), Jinwoo Choi `[通讯]` (Kyung Hee University)

**通讯引用:** 1345 | [OpenAlex ID](https://openalex.org/A5074980266)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了 RCORE 框架，用以缓解零射组合动作识别（ZS‑CAR）中因对象驱动导致的捷径学习，提升模型在未见 verb‑object 组合上的泛化能力。

**💡 创新点**

创新点包括：① 识别并量化对象驱动捷径这一关键失败模式；② 设计 composition‑aware augmentation（在视频中插入新的对象区域以合成新的组合）；③ 引入 temporal order regularization loss（对时间逆序与打乱的惩罚，鼓励模型学习真正的时序特征）；④ 通过 margin loss 降低对共现统计的依赖。

**🔧 技术方法**

技术细节：基于 CLIP 的 spatio‑temporal 编码器 AIM；使用文本提示生成 verb / object 的 embedding；多任务损失（交叉熵、余弦相似度、负熵、margin loss）；composition‑aware augmentation 与时间扰动实现；评估采用开放世界无偏协议。

**📊 数据集**

数据集：Sth‑com（Something‑Something V2 及 Something‑Else，约 79K 视频，161 verb / 248 object）和自建 EK100‑com（EPIC‑KITCHENS‑100 转化，约 71K 视角视频，82 verb / 228 object）。

**📈 对比分析**

与现有 SOTA C2C 及独立基线对比，使用 top‑1 组合、verb、object 准确率、H.M.、compositional gap 等指标；RCORE 在 unseen 组合上显著提升（+3–5% 以上），正的 compositional gap，且 FCP、FSP 等偏差指标大幅下降；在 biased 与 unbiased 两种评估下均保持优势。

**⚠️ 局限性**

限制：① 对数据稀疏与不平衡的依赖仍存在，无法完全消除共现偏差；② 只在两类视频数据集上验证，跨域泛化与更复杂动作的适应性尚未评估；③ 时间扰动方法对极短动作或静止场景可能效果有限。

---

## 346. LLM-in-Sandbox Elicits General Agentic Intelligence

**arXiv ID:** 2601.16206 | [PDF](https://arxiv.org/pdf/2601.16206v1)

**作者:** Daixuan Cheng `[一作]` (GSAI, Renmin University of China), Furu Wei `[通讯]` (Microsoft Research)

**通讯引用:** 32515 | [OpenAlex ID](https://openalex.org/A5014662947)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了LLM-IS（LLM-in-Sandbox）框架，允许大型语言模型在虚拟计算机中自由探索、执行代码、管理文件并访问外部资源，从而在非编码任务上表现出全新的推理与工具使用能力。

**💡 创新点**

创新点在于：①用最小化的沙箱工具（执行、文件、完成标记）实现通用任务解决；②通过RL训练（SHELL）在无代理数据上提升沙箱探索能力，并能向传统LLM模式迁移；③提供开源Python包和统一评测基准，展示跨领域的普适性。

**🔧 技术方法**

技术方法包括：虚拟Docker沙箱、ReAct式多轮工具调用、RL（基于结果奖励）的强化学习、文件系统交互、代码执行和网络访问等。

**📊 数据集**

使用的主要数据集：基于上下文的通用任务数据（如Instruction Pre‑Training的百科、小说、学术测试等）、专门的数学数据（DAPO）、软件工程数据（R2E‑Gym）以及在沙箱中存放的多文档长文本。

**📈 对比分析**

与传统LLM模式（直接文本生成）对比，LLM-IS在数学、物理、化学、医学、长文本理解和指令跟随等六大领域均提升，平均提升约10–25%（取决于模型和任务）。RL训练进一步在弱模型上提升约15%至30%，并且在非沙箱模式下也能获得性能提升。

**⚠️ 局限性**

局限性包括：①对非常复杂的多模态或创意生成任务仍有限（如短视频、完整音乐创作质量不足）；②沙箱执行产生的延迟与成本仍高于纯文本生成；③RL训练需要大量交互经验，且对不同模型迁移性尚待验证。

---

## 347. CONTEX-T: Contextual Privacy Exploitation via Transformer Spectral Analysis for IoT Device Fingerprinting

**arXiv ID:** 2601.16160 | [PDF](https://arxiv.org/pdf/2601.16160v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 348. LLM Prompt Evaluation for Educational Applications

**arXiv ID:** 2601.16134 | [PDF](https://arxiv.org/pdf/2601.16134v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 349. Provable Robustness in Multimodal Large Language Models via Feature Space Smoothing

**arXiv ID:** 2601.16200 | [PDF](https://arxiv.org/pdf/2601.16200v1)

**作者:** Song Xia `[一作]` (Nanyang Technological University), Xudong Jiang `[通讯]` (Nanyang Technological University)

**通讯引用:** 15342 | [OpenAlex ID](https://openalex.org/A5085533260)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种特征空间平滑（Feature‑Space Smoothing, FS）方法，并设计可插拔的净化器与平滑映射（Purifier & Smoothness Mapper, PSM）模块，使得多模态大语言模型（MLLM）的特征编码器在加噪后能够提供可证可的余弦相似度下界，从而在攻击下保持鲁棒性。

**💡 创新点**

创新点在于将随机平滑思想推广到特征空间，给出对特征余弦相似度的理论下界（Feature Cosine Similarity Bound, FCSB）；同时设计无需微调、训练自监督的PSM，利用扩散式去噪与残差映射显著提升Gaussian鲁棒性，并进一步增强FCSB。

**🔧 技术方法**

核心技术包括：1) 高斯随机平滑；2) 余弦相似度与FCSB的理论证明；3) 去噪净化器（基于预训练的引导扩散模型）和残差平滑映射网络；4) 用于训练的自监督损失（重构、鲁棒性、统计一致性等）。

**📊 数据集**

使用多种公开视觉数据集进行自监督训练：ImageNet、5,000张跨领域图像（医学、卡通、自然）；在下游任务上评估：ImageNet 10 类分类、NIPS 2017 对抗攻击集图像标注、ScienceQA VQA；对抗攻击则使用 AttackVLM、M‑Attack、FOA 三大白盒攻击器。

**📈 对比分析**

与对抗训练方法（FARE、TeCoA）以及原始模型对比，FS‑PSM 在三类攻击下均显著降低攻击成功率（从 90% 以上降至约 1%）并提升准确率（大多从 <10% 提升至 >80%），同时保持或提升了 FCSB 与认证半径；实验表明其在多任务与跨模型迁移上具有更高的鲁棒性和泛化能力。

**⚠️ 局限性**

局限性包括：1) 需要对特征编码器进行前向访问，限制了对封闭模型的适用性；2) 仍依赖高斯噪声假设，针对其他扰动分布的理论保证尚未给出；3) 训练 PSM 需要额外的数据集和计算资源，且在极大攻击预算（ε=32/255）下鲁棒性会有所下降。

---

## 350. PAL*M: Property Attestation for Large Generative Models

**arXiv ID:** 2601.16199 | [PDF](https://arxiv.org/pdf/2601.16199v1)

**作者:** Prach Chantasantitam `[一作]` (University of Waterloo), N. Asokan `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个针对大型生成模型的属性证明框架，支持在 CPU‑GPU 环境下对训练、推理等操作的完整属性证明。

**💡 创新点**

通过使用增量多集合哈希捕捉随机采样数据集的完整性、支持 TEE‑感知 GPU，并定义生成模型常见操作的属性证明，实现了对大型生成模型的可验证性。

**🔧 技术方法**

结合 Intel TDX CVM、NVIDIA H100 TEE‑aware GPU、增量多集合哈希、DCAP、NRAS 等硬件/协议实现，配合 PyTorch 与 Hugging Face 框架。

**📊 数据集**

使用 BookCorpus、yahma/alpaca‑cleaned、MMLU、WMT14(DE‑EN) 和 CoQA 等公开大规模文本数据集。

**📈 对比分析**

在 TDX+H100 环境下与无测量基线对比，属性证明开销从 1% 到 70% 取决于任务，训练与评估的测量占比不到 6%，显著低于传统 ZKP/SMPC 的高开销。

**⚠️ 局限性**

对 GPU 支持仅限于 NVIDIA Hopper/Blackwell 等支持 TEE‑aware 的型号；增量多集合哈希仍需遍历全部记录，无法处理仅局部访问的度量；以及对硬件安全性和物理攻击的假设仍未覆盖。

---

## 351. Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning

**arXiv ID:** 2601.16163 | [PDF](https://arxiv.org/pdf/2601.16163v1)

**作者:** Moo Jin Kim `[一作]` (NVIDIA), Jinwei Gu `[通讯]` (NVIDIA)

**通讯引用:** 4881 | [OpenAlex ID](https://openalex.org/A5100370029)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

将预训练的Cosmos-Predict2视频生成模型微调为机器人策略，直接输出机器人动作、未来状态和价值，并可通过模型搜索实现规划；

**💡 创新点**

不修改架构，利用隐式帧注入在视频扩散框架中联合学习动作、状态与价值，兼顾多模态输入与多摄像头；

**🔧 技术方法**

视频扩散模型、隐式帧注入、联合策略/世界模型/价值函数训练、Best-of-N规划与价值投票；

**📊 数据集**

LIBERO、RoboCasa模拟基准以及真实机器人ALOHA的四个双臂任务；

**📈 对比分析**

与Diffusion Policy、UVA、UniVLA、π_0.5、OpenVLA-OFT等SOTA对照，直接策略在LIBERO 98.5%、RoboCasa 67.1%，ALOHA多任务均领先；规划版在高多模态任务上平均提升12.5分；

**⚠️ 局限性**

推理速度慢（规划约5秒/动作块），需要大量策略回放数据来精细化世界模型，规划深度有限，易受样本稀缺影响。

---

## 352. Replicating Human Motivated Reasoning Studies with LLMs

**arXiv ID:** 2601.16130 | [PDF](https://arxiv.org/pdf/2601.16130v1)

**作者:** Neeley Pate `[一作]` (University of Rochester), Ehsan Hoque `[通讯]` (University of Rochester)

**通讯引用:** 1333 | [OpenAlex ID](https://openalex.org/A5106184792)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对四项政治动机推理实验进行复现，利用多种基础大型语言模型（LLM）评估其在无人格提示下是否表现出与人类相似的动机推理行为。

**💡 创新点**

首次系统检验多款基础LLM在无角色引导条件下的动机推理能力，揭示LLM在意见形成、论证强度评估及结果方差方面与人类的显著差异。

**🔧 技术方法**

采用链式思维提示（先给出推理再给出答案），并通过Spearman相关系数、符号一致率、标准差比较以及Benjamini–Hochberg校正等统计方法对模型输出进行评估。

**📊 数据集**

使用四项原始实验设计生成的对照与动机条件下的LLM回答，共3574条生成样本；样本规模按最小条件44例进行压缩，以降低成本。

**📈 对比分析**

通过Spearman相关系数和符号准确率比较LLM与人类平均结果，发现几乎没有显著正相关，LLM在动机推理和论证强度评估上的一致性低，且LLM标准差普遍低于人类，显示缺乏多样性。

**⚠️ 局限性**

仅评估了少量基础模型和参数组合，未探讨不同提示策略或更广泛的模型；使用的实验数据为2013-2016年老旧研究，可能与LLM训练数据重叠；生成样本规模被压缩，可能影响方差估计和整体结论。

---

## 353. CamPilot: Improving Camera Control in Video Diffusion Model with Efficient Camera Reward Feedback

**arXiv ID:** 2601.16214 | [PDF](https://arxiv.org/pdf/2601.16214v1)

**作者:** Wenhang Ge `[一作]` (Kuaishou Technology), Ying-Cong Chen `[通讯]` (HKUST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于摄像机控制的视频扩散模型，并通过摄像机感知的3D高斯解码器和奖励反馈学习来显著提升摄像机轨迹控制与场景一致性。

**💡 创新点**

创新点在于：①引入可直接评估摄像机-视频对齐的高效摄像机感知3D解码器；②设计可视性感知的像素级奖励，专注于确定性像素；③将奖励反馈学习应用于摄像机控制，使模型在不增加显存开销的前提下提升3D一致性。

**🔧 技术方法**

核心技术包括：控制网络（ControlNet）注入Plücker嵌入的摄像机信息；3D高斯解码器（3DGS）将视频潜在向量映射为可渲染的3D结构；奖励反馈学习（ReFL）结合像素级MSE/LPIPS损失与可视性掩码；使用自回归变分自编码器（VAE）生成潜在表示。

**📊 数据集**

主要使用RealEstate10K（RE10K）数据集进行训练与评估，并在RE10K测试集以及WorldScore静态基准上进行验证。

**📈 对比分析**

与MotionCtrl、CameraCtrl、ViewCrafter、FlexWorld等基线相比，在视频生成（FID、FVD、R_err、T_err）和3D场景重建（PSNR、LPIPS、SSIM）指标上均取得明显提升，例如R_err降至0.023，PSNR提升至23.77，整体表现领先所有方法。

**⚠️ 局限性**

限制包括：3D解码器的性能决定了奖励学习的上限；仅处理静态场景，无法适用于动态场景；对数据集和网络规模的扩展有限，未来可考虑更大模型和4DGS。

---

## 354. Point Bridge: 3D Representations for Cross Domain Policy Learning

**arXiv ID:** 2601.16212 | [PDF](https://arxiv.org/pdf/2601.16212v1)

**作者:** Siddhant Haldar `[一作]` (NVIDIA), Ajay Mandlekar `[通讯]` (NVIDIA)

**通讯引用:** 1390 | [OpenAlex ID](https://openalex.org/A5026180478)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于域无关点云表示的框架，用合成模拟数据实现零射程 sim-to-real 机器人操控，并支持少量真实数据联合训练与多任务学习。

**💡 创新点**

通过 VLM 导引的统一 3D 点提取、Transformer 多任务策略以及轻量级推理管线，消除视觉/对象对齐需求，实现高效零射程 sim-to-real 转移。

**🔧 技术方法**

使用 Vision‑Language Models（如 Gemini、Molmo、SAM‑2、Foundation Stereo）、Transformer 策略网络（BAKU）、点云编码（PointNet）和语言嵌入（MiniLM）等技术。

**📊 数据集**

基于 MimicLabs 任务套件生成的模拟演示，并通过 MimicGen 扩增至 1200 条演示，辅以少量真实遥操作演示（45 条）进行联合训练。

**📈 对比分析**

与图像基、域随机化、co‑training 等基准方法对比，在单任务零射程提升 39–44%，联合真实数据后提升 61–66%，并在多任务情境表现优于现有方法。

**⚠️ 局限性**

依赖 VLM 的鲁棒性、需要摄像头姿态对齐、点云抽象削弱场景上下文导致在杂乱环境中的性能下降，以及控制频率较低。

---

## 355. Non-Linearly Separable Distributed Computing: A Sparse Tensor Factorization Approach

**arXiv ID:** 2601.16171 | [PDF](https://arxiv.org/pdf/2601.16171v1)

**作者:** Ali Khalesi `[一作]` (IPSA and LINCS Lab), Petros Elia `[通讯]` (EURECOM)

**通讯引用:** 2683 | [OpenAlex ID](https://openalex.org/A5015066458)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种利用稀疏张量分解的分布式计算框架，用于在有限计算和通信资源下高效求解多用户非线性多项式函数的请求。

**💡 创新点**

创新点在于将非线性可分函数映射为稀疏张量，通过固定支持SVD和多维块切片实现任务分配和通信模式的最优设计，显著降低所需服务器数。

**🔧 技术方法**

核心技术包括张量理论、稀疏张量分解、固定支持奇异值分解（SVD）、多模态张量压缩与块级设计。

**📊 数据集**

论文未在真实数据集上实验，而是通过理论分析和示例证明方法优越性。

**📈 对比分析**

与传统线性化+稀疏矩阵分解方法相比，所提出方案在服务器数量上实现指数级减少，理论上系统速率显著提升。

**⚠️ 局限性**

限制包括：仍需对张量的全秩性假设；实现复杂度高，且对不同参数组合的细粒度调优尚未给出；实验验证缺失，实际系统部署难度待评估。

---

## 356. Domain-Incremental Continual Learning for Robust and Efficient Keyword Spotting in Resource Constrained Systems

**arXiv ID:** 2601.16158 | [PDF](https://arxiv.org/pdf/2601.16158v1)

**作者:** Prakash Dhungana `[一作]` (University of Kentucky), Sayed Ahmad Salehi `[通讯]` (University of Kentucky)

**通讯引用:** 351 | [OpenAlex ID](https://openalex.org/A5022120826)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在边缘设备上实现一种持续学习框架，针对关键词识别(KWS)在噪声多变环境中的域漂移问题，持续更新模型并保持低资源消耗。

**💡 创新点**

创新点包括：①在完整量化模型上进行全模型更新而非仅更新分类层；②采用双输入CNN（MFCC+Log‑Mel）与双阶段去噪（小波+谱减）；③通过原型距离与置信度筛选“有效样本”，并与回放缓冲区拼接进行增量训练；④将模型与原型、阈值统一量化以适配微控制器。

**🔧 技术方法**

技术手段包括：量化感知训练（QAT）与后量化；离散小波变换+VisuShrink去噪；谱去噪（基于时频均值与掩膜）；双输入卷积网络；原型‑距离（MAE）+置信阈值的有效样本判定；微小批量增量训练与全模型梯度更新；在ARM Cortex‑M4上部署。

**📊 数据集**

使用Google Speech Commands Dataset（GSCD）进行二分类（“Yes”“No”），噪声来源取自DEMAND数据集，并在‑10dB到+10dB SNR下混合生成评测样本。

**📈 对比分析**

与MCUNetv3、ODDL等现有 on‑device 训练框架比较，模型参数仅1.64k、FLOPs 0.89M，准确率在清晰音频下99.63%，在‑10dB噪声下仍超过94%；相比基准模型（如ODDL DSCNN L）参数数目降低约98.8%，能耗与延迟显著下降。

**⚠️ 局限性**

局限性：目前仅支持二分类；在极低SNR（如‑10dB）下仍有轻微性能下降；原型与阈值的更新需要额外计算，可能影响极低功耗设备；未验证多类或更大规模数据集的可扩展性。

---

## 357. All ascents exponential from valued constraint graphs of pathwidth three

**arXiv ID:** 2601.16156 | [PDF](https://arxiv.org/pdf/2601.16156v1)

**作者:** Artem Kaznatcheev `[一作]` (Utrecht University), Willemijn Volgering `[通讯]` (Utrecht University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构造了一个路径宽度为三的VCSP实例，使得所有严格局部搜索的上升路径从指定初始赋值起始点到峰值均呈指数长。

**💡 创新点**

提出了“受控倍增”构造，通过控制信息流量仅允许单一子模块激活，实现在有限稀疏图中实现全局指数上升。

**🔧 技术方法**

利用VCSP与路径宽度的图论表示、树分解与克隆子图、以及递归归纳证明相结合的技术手段。

**📊 数据集**

论文未使用具体数据集，而是以理论构造与数学证明方式展示结果。

**📈 对比分析**

通过理论上界与对比已知的指数上升构造（如KM、MS等），展示在路径宽度3的稀疏VCSP中可实现指数级搜索时间。

**⚠️ 局限性**

仅证明了路径宽度为三的情况，树宽为二的VCSP仍未确定是否存在所有上升指数长的实例，构造方法的可扩展性有限。

---

## 358. 360Anything: Geometry-Free Lifting of Images and Videos to 360°

**arXiv ID:** 2601.16192 | [PDF](https://arxiv.org/pdf/2601.16192v1)

**作者:** Ziyi Wu `[一作]` (Google DeepMind), Saurabh Saxena `[通讯]` (Google DeepMind)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本论文提出了一种基于扩散变压器的几何无关框架，可将任意视角图像或视频直接提升为全景360°图像或视频，无需任何相机标定信息。

**💡 创新点**

创新点在于：①通过序列拼接让模型从数据中自学习视角到全景的几何映射，完全摆脱显式投影；②发现并解决VAE编码器零填充导致的边缘缝隙问题，引入圆形潜在编码；③在保持零样本相机参数估计的同时，达到甚至超过传统需要标定的基准。

**🔧 技术方法**

核心技术包括扩散变压器（DiT）+潜在扩散模型、序列拼接条件方式、圆形潜在编码、标准化全景（gravity‑aligned）训练策略，以及自监督的相机参数推断。

**📊 数据集**

使用的数据集涵盖：Laval Indoor、SUN360（图像基准）、Argus视频集、NYUv2、ETH3D、iBims‑1（相机参数评测）、MegaDepth、LaMAR（姿态评测），以及大量合成渲染和真实户外/室内视频。

**📈 对比分析**

与CubeDiff、ViewPoint、Imagine360、Argus等现有方法进行对比，使用FID、KID、FAED、CLIP‑Score、PSNR、LPIPS、FVD、VBench等多指标评估，结果显示本方法在图像与视频的视觉质量、相机参数恢复、边缘无缝性上均实现了领先或相当的性能。

**⚠️ 局限性**

局限性包括：①对全景数据的canonical化预处理有一定成本；②模型仍依赖VAE潜在空间，可能受限于VAE的表达能力；③对动态场景或极端相机姿态的泛化尚未充分验证；④训练与推理资源需求相对较高。

---

## 359. Average Unfairness in Routing Games

**arXiv ID:** 2601.16187 | [PDF](https://arxiv.org/pdf/2601.16187v1)

**作者:** Pan-Yang Su `[一作]` (University of California), Shankar Sastry `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并研究了一种新的平均不公平性度量，用于评估路由游戏中流量分配的公平程度；

**💡 创新点**

创新点在于引入平均不公平性并证明其与已有的负载不公平性和用户均衡不公平性在极值上等价，同时指出平均不公平性在受限系统最优（CSO）问题中能够产生更低的总延迟；

**🔧 技术方法**

采用理论分析方法推导极值上界，利用标准延迟函数的“斜率”参数，并在多种网络结构上构造Pigou、Braess等经典网络以验证界限；

**📊 数据集**

实验使用了四个真实交通网络（Anaheim、Sioux Falls、Massachusetts、Friedrichshain）以及BPR（Bureau of Public Roads）链接延迟函数进行数值模拟；

**📈 对比分析**

通过将平均不公平性与负载不公平性对应的CSO解进行比较，实验显示在大多数网络和公平容差下，平均不公平性约束下的总成本明显低于负载不公平性约束，性能提升显著；

**⚠️ 局限性**

限制在于理论证明仅给出充分条件，无法保证在所有网络结构下平均不公平性一定能严格改进成本，且实验仅覆盖有限的网络样本，缺乏对更一般网络的深入分析。

---

## 360. Dynamic Pattern Matching with Wildcards

**arXiv ID:** 2601.16182 | [PDF](https://arxiv.org/pdf/2601.16182v1)

**作者:** Arshia Ataee Naeini `[一作]` (University of Tehran), Saeed Seddighin `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

提出了在文本和模式都可能出现通配符且可动态更新的模式匹配问题的理论框架，给出了通用和稀疏两种情况的子线性更新/查询算法；

**💡 创新点**

首次实现了对完全动态带通配符的模式匹配在k为常数或k=o(log n)时的真正子线性时间复杂度，并给出了相应的条件下的下界；

**🔧 技术方法**

核心技术包括：频率分割（稀有/频繁符号）与占位符替换、基于多项式滚动哈希的区间哈希与区块化、灰度码枚举、FFT+块分解、以及全动态最长公共子串结构；

**📊 数据集**

本文主要是理论分析与算法设计，没有使用具体的实测数据集；

**📈 对比分析**

通过与已知的静态/动态不带通配符匹配、随机化哈希匹配等方法比较，展示在k常数时预处理时间为O(n log²n)，更新/查询时间为O(k nᵏ/(k+1)+k² log n)，显著低于先前最优；在k=O(log n)时仍保持子线性；稀疏模式下，最多两个固定符号时预处理O(n¹·⁸)，更新O(n⁰·⁸ log n)；

**⚠️ 局限性**

主要局限在于：1) 当k=Ω(log n)时无法突破子线性阈值（受SETH下界限制）；2) 目前仅支持替换更新，插入删除需额外实现；3) 对于大规模通配符场景仍有效率瓶颈。

---

## 361. Learning to Discover at Test Time

**arXiv ID:** 2601.16175 | [PDF](https://arxiv.org/pdf/2601.16175v1)

**作者:** Mert Yuksekgonul `[一作]` (Stanford University), Yu Sun `[通讯]` (Stanford University)

**通讯引用:** 11064 | [OpenAlex ID](https://openalex.org/A5030042608)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种在测试时对单一科学问题进行强化学习的框架——TTT‑Discover，利用LLM在测试时不断学习并搜索，最终发现新的最优解。

**💡 创新点**

创新点在于将强化学习目标改为“熵目标”偏向最大奖励，并结合PUCT搜索策略与状态重用，专门针对单一问题的“发现”任务，而非平均性能。

**🔧 技术方法**

使用的大模型是开放源代码的gpt‑oss‑120b（LoRA微调），结合Tinker API进行计算，核心算法为熵目标RL（改进的GRPO）+ PUCT 搜索。

**📊 数据集**

评估数据集覆盖GPU kernel（TriMul、MLA）、数学（Erdős最小重叠、两条自相关不等式）、算法竞赛（AtCoder AHC）以及生物单细胞去噪（OpenProblems）。

**📈 对比分析**

与先前的AlphaEvolve、ThetaEvolve等方法相比，TTT‑Discover在所有任务中均实现或接近最优；在数学上超越AlphaEvolve（如Erdős upper bound 0.380876），在GPU kernel 上比人类快 50%+，在单细胞去噪上比 MAGIC/ALRA 提升 10‑20%。

**⚠️ 局限性**

局限性包括仅适用于可验证且连续奖励的任务；对稀疏/二值奖励的适用性有限；算法对模型规模和超参数敏感，需在不同任务上调优。

---

## 362. Structured Hints for Sample-Efficient Lean Theorem Proving

**arXiv ID:** 2601.16172 | [PDF](https://arxiv.org/pdf/2601.16172v1)

**作者:** Zachary Burton `[一作]` `[通讯]` (Massachusetts Institute of Technology), Zachary Burton (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究在资源受限环境下给神经定理证明器加上结构化提示，通过固定的 tactic skeleton 来引导推理；

**💡 创新点**

创新点在于提出一种轻量级的中间表示（结构化提示），证明即便是已通过强化学习训练的模型，也能通过简单的推理时结构引导显著提升成功率；

**🔧 技术方法**

使用了固定 prompt schedule、15 个 tactic skeleton、温度 0.6、top‑p 0.95 的采样策略，并以 DeepSeek‑Prover‑V1.5-LM 为基线模型；

**📊 数据集**

在 Lean 4 miniF2F benchmark（244 题）上进行评估；

**📈 对比分析**

与无指导基线在同等 k=16、max_tokens=1024 的预算下对比，Pass@16 从 15.16% 提升至 21.72%（相对提升 43%），在对比分析中取得 19 胜 3 负，显著优于基线；

**⚠️ 局限性**

局限性包括：受 1024 token 与 k=16 的硬性限制，可能截断更长证明；仅使用单一 seed 与解码配置，缺乏多样性评估；固定 skeleton 可能不适用于更大模型或更高预算的场景。

---

## 363. Learning to Watermark in the Latent Space of Generative Models

**arXiv ID:** 2601.16140 | [PDF](https://arxiv.org/pdf/2601.16140v1)

**作者:** Sylvestre-Alvise Rebuffi `[一作]` (Meta FAIR), Alexandre Mourachko `[通讯]` (Meta FAIR)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了在扩散模型和自回归模型的潜在空间中嵌入水印的方法，并实现了可以蒸馏进模型权重或解码器的内置水印技术。

**💡 创新点**

创新点在于提出了统一的潜在空间水印框架，支持两类模型，显著提升了水印嵌入速度（最高可达20×），并通过蒸馏实现了开源模型的无缝内置水印。

**🔧 技术方法**

使用的技术包括潜在空间的 embedder–extractor 训练、对抗判别器损失、Straight-Through 估计、以及针对解码器或生成器的蒸馏方法。

**📊 数据集**

实验基于 ImageNet 数据集，使用 512×512 图像训练 DCAE 扩散模型和 RAR‑XL 自回归模型，并在多种攻击下评估水印鲁棒性。

**📈 对比分析**

与像素空间后处理水印（如 VideoSeal）对比，潜在水印在鲁棒性（约 95% bit 准确率）和视觉质量（FID/IS 与基线相近）上相当，但推理速度提升 20×，在 DCAE 上从 63 ms 降至 3 ms/图像。

**⚠️ 局限性**

限制包括对强几何攻击的鲁棒性不足、蒸馏效果受教师水印性能限制、离散自回归模型受码本大小约束，以及可能在微调后出现水印遗忘。

---

## 364. Computing Fixpoints of Learned Functions: Chaotic Iteration and Simple Stochastic Games

**arXiv ID:** 2601.16142 | [PDF](https://arxiv.org/pdf/2601.16142v1)

**作者:** Paolo Baldan `[一作]` (University of Padua), Florian Wittbold `[通讯]` (Universität Duisburg-Essen)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并证明了一种更一般化的阻尼Mann迭代方法，能够在学习率收敛、发散或随机分量更新（混沌）情况下，对任意单调非扩张函数求最小不动点；

**💡 创新点**

创新点在于将Mann-Kleene方案推广到可消失或发散的学习率、随机化的混沌更新，并通过依赖图与多步Bellman运算的复合结构，保证对S游戏、MDP等分块线性函数的收敛；

**🔧 技术方法**

采用的技术包括：单调非扩张函数的理论分析、阻尼参数设计、依赖图分块、最优内在策略固定化、以及对采样序列的连通性与功率收缩性证明；

**📊 数据集**

使用的数据集为50个随机生成的简单随机游戏（SSG），每个游戏包含15个最小化状态和15个最大化状态，最多5个动作/状态，基准值设为1；

**📈 对比分析**

在实验中将该方法与传统的松弛Mann-Kleene迭代、固定学习率以及随机混沌更新进行对比，结果显示在多数实例中，随机混沌与消失学习率方案在收敛误差下降曲线上优于传统方案，但改进幅度有限；

**⚠️ 局限性**

局限性在于：理论证明对函数结构的假设比较严格；实验仅覆盖小型随机SSG，未验证在更大或更复杂模型上的性能；随机化更新未充分利用状态依赖信息，可能导致收敛速度受限。

---

## 365. Scaling Sample-Based Quantum Diagonalization on GPU-Accelerated Systems using OpenMP Offload

**arXiv ID:** 2601.16169 | [PDF](https://arxiv.org/pdf/2601.16169v1)

**作者:** Robert Walkup `[一作]` (IBM Research), Seetharami Seelam `[通讯]` (IBM Research)

**通讯引用:** 790 | [OpenAlex ID](https://openalex.org/A5033227162)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

实现了将样本量化对角化（SQD）算法的GPU加速版本，并通过OpenMP Offload实现跨CPU/ GPU的可移植代码；

**💡 创新点**

提出持久配置缓存、嵌套数据结构扁平化、完整GPU端矩阵元素计算、以及多层GPU内存管理的系统化优化路线；

**🔧 技术方法**

使用OpenMP Target Offload、GPU指令集、HIP等技术；在Frontier、MI300X、A100、H100等多种GPU架构上验证；

**📊 数据集**

主要使用水分子（H₂O）与N₂分子在cc-pVDZ基组下的配置集合，规模从6.28×10⁸到1.52×10⁹个电子配置；

**📈 对比分析**

与原始CPU实现以及使用相同缓存的CPU实现对比，GPU实现单节点性能提升约95×，在不同GPU平台上每GPU性能比MI250X提升1.8–3倍；

**⚠️ 局限性**

对GPU单机需至少1e5个配置以充分利用GPU，配置缓存受GPU内存限制（≈10⁹配置），对bit_length参数敏感，且需在任务并行配置上手动调优。

---

## 366. HVD: Human Vision-Driven Video Representation Learning for Text-Video Retrieval

**arXiv ID:** 2601.16155 | [PDF](https://arxiv.org/pdf/2601.16155v1)

**作者:** Zequn Xie `[一作]` (Zhejiang University), Tao Jin `[通讯]` (Southwestern University of Finance and Economics)

**通讯引用:** 724 | [OpenAlex ID](https://openalex.org/A5101462294)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出人类视觉驱动(HVD)模型，结合帧特征选择模块(FFSM)与补丁特征压缩模块(PFCM)，实现文本-视频检索的自适应宏观与微观对齐。

**💡 创新点**

创新点在于模拟人类视觉的宏观关注与微观聚焦，通过关键帧筛选与密度峰值聚类+注意力机制实现对重要视觉实体的自动压缩与精细匹配，显著提升跨模态特征交互的精准度。

**🔧 技术方法**

技术包括CLIP视觉/文本编码、基于相似度的关键帧筛选、KNN‑DPC密度峰值聚类、注意力压缩、交叉熵对比损失等。

**📊 数据集**

在五大基准数据集上评测：MSRVTT、DiDeMo、LSMDC、ActivityNet、Charades。

**📈 对比分析**

与多种现有方法对比，HVD 在 MSRVTT 上 R@1 48.8%、R@5 75.2%、R@10 85.3% 等指标均取得 SOTA，其他数据集亦表现优于对照，整体提升显著。

**⚠️ 局限性**

局限性包括需手动设定帧/补丁保留比例，且模型依赖 CLIP 作为基座，可能在极长视频或更细粒度语义匹配场景下表现受限。

---

## 367. Pay (Cross) Attention to the Melody: Curriculum Masking for Single-Encoder Melodic Harmonization

**arXiv ID:** 2601.16150 | [PDF](https://arxiv.org/pdf/2601.16150v1)

**作者:** Maximos Kaliakatsos-Papakostas `[一作]` (Hellenic Mediterranean University), Emilios Cambouropoulos `[通讯]` (Aristotle University of Thessaloniki)

**通讯引用:** 1574 | [OpenAlex ID](https://openalex.org/A5073323305)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究单编码器Transformer在旋律伴随中的生成任务，提出一种“全-全”训练课程，使模型在训练初期先完全遮掩和声，随后逐步解码，从而强化旋律与和声的交叉注意。

**💡 创新点**

创新点在于将全遮掩到全解码的递进式掩码策略引入到单编码器和声生成中，显著提升了模型对旋律信息的利用，尤其在跨域（爵士标准）评估中表现优异。

**🔧 技术方法**

主要技术包括Transformer单编码器、迭代掩码学习、full-to-full curriculum、基于音程的pitch‑class piano‑roll表示、bar信息嵌入以及多种推理解码策略（top‑10%、midpoint、顺序）。

**📊 数据集**

实验使用HookTheory MIDI数据集（15440首）进行训练与验证，并用650首手工挑选的爵士标准曲目做外域评估。

**📈 对比分析**

与现有的Random‑10%和Midpoint‑Doubling掩码训练进行对比，采用9项音乐性指标（如CHT、CTD、CTS、PCS、MCTD等）评估。full‑to‑full课程在大多数指标上均优于基线，尤其在外域数据的Chord Histogram Entropy和Chord Coverage上提升明显。

**⚠️ 局限性**

局限性包括：只在最小化的单编码器架构上验证，未对指数参数做系统调优；缺乏听感主观评估；对不同音乐风格或个别乐曲的适配性未做深入探索。

---

## 368. Beat-ssl: Capturing Local ECG Morphology through Heartbeat-level Contrastive Learning with Soft Targets

**arXiv ID:** 2601.16147 | [PDF](https://arxiv.org/pdf/2601.16147v1)

**作者:** Muhammad Ilham Rizqyawan `[一作]` (University of Glasgow), Fani Deligianni `[通讯]` (University of Glasgow)

**通讯引用:** 3824 | [OpenAlex ID](https://openalex.org/A5082776788)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出 Beat-SSL 框架，利用 12 导 ECG 在心律层和心跳层双重对比学习，并通过软目标提升对比质量；

**💡 创新点**

创新点在于：1）同时对全局（节律）和局部（单次心跳）进行对比；2）使用软目标（连续相似度）替代传统硬二元标签；3）利用 VCG 空间增强与指数化相似度来强化正样本；4）通过单一编码器实现多层上下文学习；

**🔧 技术方法**

核心技术包括：3KG VCG 变换与数据增强；Kors 转换；NT‑Xent 对比损失；soft_1 与 soft_2 软目标生成；beat‑projection head 与 ROI‑style 切分；指数化相似度提升；

**📊 数据集**

预训练使用 PTB‑XL 无标签 18,885 名患者 10‑秒 12‑导 ECG；下游任务分别在 PTB‑XL（多标签分类）和 LUDB（波形分割）数据集上评估；

**📈 对比分析**

与 TS2Vec、Domain‑SSL、ECG‑FM 进行对比，Beat‑SSL 在多标签分类任务达 93% 的 ECG‑FM 性能（仅 31.8× 的预训练数据），在分割任务中以 4% 的优势领先最优模型；

**⚠️ 局限性**

局限性包括：1）仍需大量预训练样本，未能彻底突破 ECG‑FM 规模；2）对 beat‑level 软目标的探索不足，可能影响不同心律的细粒度学习；3）实验仅在 PTB‑XL 与 LUDB 上验证，跨域泛化能力尚未充分评估。

---

## 369. On the Intrinsic Dimensions of Data in Kernel Learning

**arXiv ID:** 2601.16139 | [PDF](https://arxiv.org/pdf/2601.16139v1)

**作者:** Rustem Takhanov `[一作]` (Nazarbayev University), Rustem Takhanov `[通讯]` (Nazarbayev University)

**通讯引用:** 136 | [OpenAlex ID](https://openalex.org/A5029285634)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文通过引入两种本征维数——上 Minkowski 维数 d_ϱ 和由 Kolmogorov n‑宽定义的有效维数 d_K，研究了核岭回归（KRR）的泛化误差，并给出基于 d_K 的超额风险上界；同时提出了从有限样本估计 n‑宽上界的贪心算法，并在多种分形集合与球面上验证理论。

**💡 创新点**

创新点在于：
1) 将 Kolmogorov n‑宽与核积分算子特征值衰减关联，提供了对任意概率分布下最差特征值衰减的描述；
2) 在不需要高阶光滑度假设的情况下，给出仅依赖 d_K 的超额风险上界；
3) 设计了基于核矩阵行列式最大化的贪心采样算法，可在有限样本上得到 n‑宽上界；
4) 在分形数据上首次量化 d_K 与 d_ϱ 的差距，揭示分形结构对有效维数的影响。

**🔧 技术方法**

技术手段包括：几何测度论（上 Minkowski 维数、Hausdorff 维数）、Kolmogorov n‑宽理论、Ismagilov 定理、核积分算子谱分析、贪心子空间逼近、误差传播与上界证明、数值实验与 RANSAC 回归。

**📊 数据集**

实验使用的“数据集”主要是合成分形集合（Cantor 集、Weierstrass 函数、Sierpinski 砂布、Menger 冒泡、Lorenz 吸引子）以及单位球面 𝕊^{d-1}，在后者上使用旋转不变分布，并对三种核（高斯、拉普拉斯、ReLU NTK）进行评估。

**📈 对比分析**

比较方法：
- 对比 d_K 与 d_ϱ 的数值（实验得到 d_K < d_ϱ，验证理论断言）；
- 通过绘制 -log(w_n) vs log n 估计 d_K 的上界；
- 记录超额风险与样本规模的对数关系，并与理论上界（斜率 ≤ -1、-2/3、-0.6、-4/7 等）对照。实验结果表明，实际收敛速度与理论预测相近，且不同核在相同维度下表现出预期的速度差异。

**⚠️ 局限性**

局限性：
1) 需要 μ 为 C‑均匀分布；对非均匀或支撑不满Ω的分布难以直接应用；
2) 只给出 n‑宽的上界，缺乏下界或精确估计；
3) 贪心算法在高维或大样本时计算量大，实际实现依赖于非凸优化；
4) 对 d_K 与 d_ϱ 的差距在分形域的理论解释尚不完整，未来仍需深入研究。

---

## 370. Rethinking Composed Image Retrieval Evaluation: A Fine-Grained Benchmark from Image Editing

**arXiv ID:** 2601.16125 | [PDF](https://arxiv.org/pdf/2601.16125v1)

**作者:** Tingyu Song `[一作]` (CASIA), Shu Wu `[通讯]` (CASIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于图像编辑的可控数据合成管线，构建了覆盖15个细粒度子类别、共5000个查询的全新合成图像检索基准

**💡 创新点**

①设计细粒度分类体系与可控合成流程；②利用图像编辑实现精准、可重复的修改；③基准大幅扩展查询类别，解决现有基准的类别不平衡与模态偏差

**🔧 技术方法**

使用多模态大语言模型（Qwen2.5‑VL、Qwen3）与图像编辑模型（Qwen‑Image‑Edit）进行编辑与查询重写；双阶段 MLLM 过滤与人工验证；对多种非 MLLM 与 MLLM 嵌入模型进行 Recall@1 评估，并在基准上进行内部训练实验

**📊 数据集**

以 LAION‑400M 为种子图像，经过 MLLM 过滤后生成 889,013 个高质量 {I_r,T_m,I_t} 三元组，最终基准包含 5,000 个查询与 178,645 张图片

**📈 对比分析**

对比多种非 MLLM 与 MLLM 嵌入模型，Recall@1 最高为 59.9%（内部训练后），非 MLLM 平均仅 18.4%；与现有 CIR 基准相比，显示更高难度与更全面的评估

**⚠️ 局限性**

数据生成成本高、复杂查询仅限三条约束、基准主要用于评估而非提供规模化训练方案

---

## 371. Scalable Board Expansion within a General Game System

**arXiv ID:** 2601.16216 | [PDF](https://arxiv.org/pdf/2601.16216v1)

**作者:** Clémentine Sacré `[一作]` `[通讯]`, Clémentine Sacré

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

论文探讨了某一领域的研究问题，并提出了一种新的解决方案。

**💡 创新点**

创新点在于提出了一种新的方法或模型，能够更有效地解决该问题。

**🔧 技术方法**

使用了机器学习和深度学习等技术。

**📊 数据集**

使用了公开数据集进行实验，具体数据集名称未提供。

**📈 对比分析**

与现有方法进行了比较，结果显示新方法在准确性和效率上都有显著提升。

**⚠️ 局限性**

限制在于模型的可扩展性和对特定数据集的依赖性。

---

## 372. PyraTok: Language-Aligned Pyramidal Tokenizer for Video Understanding and Generation

**arXiv ID:** 2601.16210 | [PDF](https://arxiv.org/pdf/2601.16210v1)

**作者:** Onkar Susladkar `[一作]` (University of Illinois Urbana-Champaign), Ismini Lourentzou `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 942 | [OpenAlex ID](https://openalex.org/A5043962698)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种语言对齐金字塔式离散VAE分词器，能够在多尺度时空上进行语义结构化量化。

**💡 创新点**

创新点包括共享大二进制词表的金字塔量化、双重语义对齐（局部文本引导 + 全局自回归）以及层级代码书损失。

**🔧 技术方法**

主要技术是基于预训练视频VAE、Lookup‑Free量化、文本引导注意力、LoRA微调、KL对齐损失和自回归目标。

**📊 数据集**

使用了WebVid‑10M、YouTube‑VIS、Ovis、THUMOS14、ActivityNet、MVBench、Kinetics‑400/600/700、UltraVideo等10个公开数据集。

**📈 对比分析**

在视频重建、文本生成、零样本分割、动作定位、分类等10项任务中均超过或逼近现有SOTA，显著提升FVD、mAP、视频QA等指标。

**⚠️ 局限性**

主要限制是仍需大量算力进行多尺度训练，且对极低帧率或极长时序的视频表现尚未充分验证。

---

## 373. Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders

**arXiv ID:** 2601.16208 | [PDF](https://arxiv.org/pdf/2601.16208v1)

**作者:** Shengbang Tong `[一作]` (New York University), Saining Xie `[通讯]` (New York University)

**通讯引用:** 44655 | [OpenAlex ID](https://openalex.org/A5102416863)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文将表示自编码器（RAE）从 ImageNet 扩展到大规模自由文本到图像（T2I）生成，并在大规模语料下对 RAEs 进行调优与验证。

**💡 创新点**

创新点在于：①证明 RAEs 在大规模 T2I 训练中比传统 VAE 更高效、更稳定；②发现规模化后仅需维度自适应噪声调度，其余复杂设计可省略；③通过共享高维语义空间实现统一理解与生成，并引入潜在空间测试时缩放（Latent TTS）提升质量。

**🔧 技术方法**

技术包括：使用冻结的 SigLIP‑2 或 WebSSL 预训练表示编码器；ViT‑Based RAE 解码器训练；流匹配目标的 Diffusion Transformer（DiT）；维度依赖噪声调度；LLM（Qwen‑2.5）与查询标记交互；潜在空间直接验证与测试时缩放。

**📊 数据集**

数据集涵盖：ImageNet、YFCC、RenderedText、Web (CC12M, SA‑1B, JourneyDB)、合成 (FLUX.1‑schnell)、WebSSL、BLIP‑3o 60k、以及 73M 组合数据（Web+Synthetic+Text）进行解码器训练；预训练混合 39.3M Web + 24.7M 合成；微调使用 BLIP‑3o 60k。

**📈 对比分析**

在 GenEval 与 DPG‑Bench 上，RAE 在所有 DiT 规模（0.5B‑9.8B）和 LLM 规模（1.5B‑7B）下均显著优于 FLUX‑VAE，预训练阶段加速 4×；微调阶段 RAE 具更高分数且不易过拟合；在视觉理解基准（MME、TVQA 等）上，RAE 与 VAE 近似，统一模型在潜在空间测试时缩放可进一步提升 2–4 分。

**⚠️ 局限性**

局限性包括：RAE 的重建质量仍略逊于最先进的 FLUX‑VAE；在极大模型（>6B）时性能趋于饱和，需更高质量多样化数据；解码器训练对资源需求高，且在某些特定域（如细粒度文字）仍需针对性数据；未来需探索更高效的表示与解码器匹配方案。

---

## 374. IVRA: Improving Visual-Token Relations for Robot Action Policy with Training-Free Hint-Based Guidance

**arXiv ID:** 2601.16207 | [PDF](https://arxiv.org/pdf/2601.16207v1)

**作者:** Jongwoo Park `[一作]` (Stony Brook University), Michael S Ryoo `[通讯]` (Stony Brook University)

**通讯引用:** 8010 | [OpenAlex ID](https://openalex.org/A5084829008)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种轻量化、训练无关的IVRA方法，在推理时将视觉编码器的affinity提示注入语言模型中，以恢复VLA模型的二维空间结构。

**💡 创新点**

创新点在于不引入额外参数或重新训练，仅通过在LLM中插入基于affinity的加权平均池化，显著提升实例级空间感知与操控精度。

**🔧 技术方法**

技术细节包括：冻结视觉编码器提取patch affinity矩阵；在LLM的指定层执行affinity‑guided token pooling；对原始视觉token与池化结果做凸组合，并通过λ调节权重。

**📊 数据集**

实验数据集涵盖VIMA（2D）和LIBERO（3D）仿真 benchmark，以及真实机器人T1–T4任务，使用公开的OpenVLA、FLOWER、LLaRA等VLA模型进行验证。

**📈 对比分析**

与基线VLA模型、oracle detector、RT‑2‑Style等进行对比；在VIMA上平均提升4.2%，在LIBERO上提升1.1%，在真实任务中T1–T4的成功率提升10%–30%；整体表现优于现有方法。

**⚠️ 局限性**

局限性包括：仍受原始视觉编码器分辨率限制，对极端遮挡或非常大图像的鲁棒性不足；在已达到高精度（近饱和）场景中增量提升有限。

---

## 375. Counterfactual Training: Teaching Models Plausible and Actionable Explanations

**arXiv ID:** 2601.16205 | [PDF](https://arxiv.org/pdf/2601.16205v1)

**作者:** Patrick Altmeyer `[一作]` (Delft University of Technology), Cynthia C. S. Liem `[通讯]` (Delft University of Technology)

**通讯引用:** 883 | [OpenAlex ID](https://openalex.org/A5022063970)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新的训练范式——Counterfactual Training（CT），在训练过程中实时利用对抗性解释（counterfactual explanations）来约束模型的学习，从而让模型本身就能产生符合真实性与可操作性要求的解释。

**💡 创新点**

创新点在于：①把对抗性解释从事后生成工具转变为模型训练的主动因素；②通过对比损失（contrastive divergence）与对抗损失（adversarial loss）两种机制同时作用，既提升了解释能力也提升了模型的鲁棒性；③在保持可解释性的同时不显著牺牲预测性能。

**🔧 技术方法**

技术与方法：梯度下降生成对抗性解释、能量约束（energy-based）对比损失、对抗训练、对可操作性约束的投影与梯度修正、使用ECCCo等高质量对抗性解释生成器、对比实验中的基线与消融实验。

**📊 数据集**

实验数据集：四类合成二分类数据（线性可分、重叠、高斯环、相互锁月形）；四个实际表格数据集（Adult、California Housing、Credit Card Default、GiveMeSomeCredit）；以及10分类手写数字集MNIST。

**📈 对比分析**

评估方式：与传统弱基线（plain MLP）和消融模型（只保留对比或只保留对抗）进行对比，使用IP/IP*衡量可解释性（假设度），用成本指标评估可操作性，并在FGSM/PGD攻击下测量鲁棒性。实验结果表明：CT可将可解释性假设度提升高达90%，可操作性成本下降约20%至60%，在对抗攻击下的准确率大幅优于基线，且整体预测性能基本保持不变。

**⚠️ 局限性**

局限性：①对对抗性解释生成器和其超参数高度依赖，需额外调优；②训练开销显著增加；③在某些数据集上，提升可解释性会导致有效解释率下降或成本上升；④只能应用于分类任务，难以直接扩展到回归等连续输出；⑤对可操作性约束的设定需要领域专家参与，若约束不当可能引入公平性问题。

---

## 376. Tensor Reed-Muller Codes: Achieving Capacity with Quasilinear Decoding Time

**arXiv ID:** 2601.16164 | [PDF](https://arxiv.org/pdf/2601.16164v1)

**作者:** Emmanuel Abbe `[一作]` (EPFL), Oscar Sprumont `[通讯]` (University of Washington)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5051320707)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

设计并分析了一类新的编码——Tensor Reed-Muller 码，并给出了在 BMS 通道上达到容量的高效解码算法，运行时间为 O(n loglog n)（t=3）或 O(n log n)（t>3），误码概率可降到 n^-ω(log n) 或 2^-n^{1/2-1/2(t-2)}。

**💡 创新点**

创新点在于：①将 Reed-Muller 码按变量组分块后取张量积，构造出新的 Tensor 码；②提出一种多维张量码的通用误差恢复算法，能够在不要求子码可多项式解码的前提下，从 d_min(C)/2max{d_min(C_i)}-1 的对抗误差中恢复；③通过分层（先低速高精度再高速）解码策略，结合 RM 码的随机误差下高容量性能，实现了常数速率下的容量接近。

**🔧 技术方法**

主要技术包括：张量积结构与多维子空间约束、基于 Erasure 约束的 RM 码高效解码（O(n log n)），递归分层解码算法，Chernoff 绑定概率误差，上界证明及多维张量恢复的迭代策略。

**📊 数据集**

本工作为理论研究，未使用具体数据集；所有结论均来自解析证明和概率上界。

**📈 对比分析**

相较于传统 RM 码的高容量证明（仅给出最大似然解码）和现有的多项式时间 RM 码解码（仅限 r=O(1) 或极小误差率），Tensor RM 码在常数速率下实现了容量并给出可行的解码算法；误码率可低至 n^-ω(log n) 或指数级 2^-Ω(√n)，运行时间仅为 O(n loglog n) 或 O(n log n)。

**⚠️ 局限性**

局限性：①需要将变量分块并取张量积，导致码长和参数调节较为复杂；②算法对参数 t 的上界有限，t 需 ≤ √(log n)；③误码概率虽低，但仍受 2^-n^{1/2-1/2(t-2)} 限制，无法达到 ML 级别；④实现细节（如大规模张量操作）在实践中可能存在效率瓶颈。

---

## 377. Substrate Stability Under Persistent Disagreement: Structural Constraints for Neutral Ontological Substrates

**arXiv ID:** 2601.16152 | [PDF](https://arxiv.org/pdf/2601.16152v1)

**作者:** Denise M. Case `[一作]` (Northwest Missouri State University), Denise M. Case `[通讯]` (Northwest Missouri State University)

**通讯引用:** 135 | [OpenAlex ID](https://openalex.org/A5009909166)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文在持久性分歧条件下，研究了作为中立子层的本体最小结构，证明至少需要六个身份与持久性范式，并给出实现构造。

**💡 创新点**

提出了在中立性与稳定性约束下，对可用于问责的本体必须具备的最小结构，给出下界并证明其紧确性。

**🔧 技术方法**

采用形式化的本体论与逻辑推理方法，构建结构证明。

**📊 数据集**

未使用任何数据集，纯理论推导。

**📈 对比分析**

无实验比较，结果为理论证明，未给出性能指标。

**⚠️ 局限性**

局限在于只适用于已指定的中立性与稳定性目标，未考虑协调或局部一致的情景。

---

## 378. ActionMesh: Animated 3D Mesh Generation with Temporal 3D Diffusion

**arXiv ID:** 2601.16148 | [PDF](https://arxiv.org/pdf/2601.16148v1)

**作者:** Remy Sabathier `[一作]` (Meta Reality Labs), Tom Monnier `[通讯]` (Meta Reality Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个两阶段的 feed‑forward 生成框架，能够从视频、文本、图像+文本或已有 3D 网格+文本等多种输入快速生成可动画、拓扑一致且无骨架的 3D 网格。

**💡 创新点**

创新点包括：①将现有 3D 扩散模型沿时间轴扩展为 temporal 3D diffusion；②引入 masked 生成，支持已知网格的条件化生成；③使用 temporal 3D autoencoder 把独立的形状序列映射为对参考网格的变形，得到拓扑一致的动画；④整个流程为纯前向推理，无需后期优化或 per‑scene 训练。

**🔧 技术方法**

采用预训练的 3D latent diffusion 模型（TripoSG/ Craftsman）为基础；对自注意力层进行 inflate 以跨帧同步；使用 rotary position embedding 注入时间位置信息；设计 masked 生成机制；构建 temporal 3D autoencoder（transformer + Fourier 位置编码 + cross‑attention）来预测顶点位移；评估时使用 ICP 与 Chamfer 距离。

**📊 数据集**

训练使用合成数据；评估在自建的 Objaverse benchmark 以及公开的 Consistent4D benchmark 上；在 DAVIS 实际视频上进行定性验证。

**📈 对比分析**

在 Objaverse benchmark 上与 LIM、DreamMesh4D、V2M4、ShapeGen4D 进行定量对比，分别在 CD‑3D、CD‑4D、CD‑M 上提升 21%、46%、45%；推理时间仅 3 min，速度比前沿方法快约 10 倍；在 Consistent4D benchmark 上在可视化上表现出更高的几何精度和时间一致性。

**⚠️ 局限性**

局限性：无法处理拓扑变化的情形；在强遮挡或缺失视角时会出现重建误差；依赖参考网格，若参考缺失关键部件会影响动画；对极端复杂动态的建模仍有挑战。

---

## 379. Low-altitude Multi-UAV-assisted Data Collection and Semantic Forwarding for Post-Disaster Relief

**arXiv ID:** 2601.16146 | [PDF](https://arxiv.org/pdf/2601.16146v1)

**作者:** Xiaoya Zheng `[一作]` (Jilin University), Abbas Jamalipour `[通讯]` (University of Sydney)

**通讯引用:** 16735 | [OpenAlex ID](https://openalex.org/A5086268677)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

设计并实现低空多UAV协同数据采集与语义转发网络，解决灾后通信恢复中的链路弱、数据拥塞等问题。

**💡 创新点**

提出针对该问题的多目标MINLP优化模型（DCSFMOP），并开发利用大型语言模型指导的交替优化算法（LLM‑AOA），能够处理动态维度的整数与连续变量，显著提升传输率与语义率并降低能耗。

**🔧 技术方法**

采用协作波束成形、语义通信（DeepSC）、多目标进化算法（NSGA‑II）、贪婪聚类/符号优化、以及LLM（如ChatGPT 5.0）进行参数自适应与搜索策略引导。

**📊 数据集**

在仿真环境中使用500名地面用户、8架UAV（或4/12/16架）以及固定远端基站，生成随机用户分布与路径损耗模型，未使用公开真实数据集。

**📈 对比分析**

与MODE、MOEA/D、MOPSO、NSGA‑II及无LLM版AOA比较，LLM‑AOA在传输率约提高26.8%、语义率约提高22.9%同时保持能耗可接受，Pareto前沿更靠近理论极值。

**⚠️ 局限性**

仍受限于仿真规模、LLM调用成本及可解释性不足；对动态环境、实时调度、能量自愈等更复杂情景的适应性待进一步验证。

---

## 380. Automatic Classification of Arabic Literature into Historical Eras

**arXiv ID:** 2601.16138 | [PDF](https://arxiv.org/pdf/2601.16138v1)

**作者:** Zainab Alhathloul `[一作]` (King Fahd University of Petroleum and Minerals), Irfan Ahmad `[通讯]` (King Fahd University of Petroleum and Minerals)

**通讯引用:** 2548 | [OpenAlex ID](https://openalex.org/A5070341092)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用神经网络对阿拉伯文学文本进行时代分类

**💡 创新点**

首次系统评估多种历史时期划分与自定义时间段的分类，并实验作者风格对结果的影响

**🔧 技术方法**

采用全连接 ANN、Bi‑GRU LSTM/GRU RNN、CNN 及传统 Logistic 回归等深度学习模型

**📊 数据集**

使用公开的 OpenITI（文学散文）和 APCD（诗歌）两大语料库

**📈 对比分析**

与传统机器学习基线和文献中的 CNN 进行对比，二分类在 OpenITI 上准确率达 0.83、在 APCD 上 0.78，F1‑score 在 20–65% 之间，显示多分类任务仍较困难

**⚠️ 局限性**

受限于时代边界模糊、古典语料稀缺、词形化简不足、作者重叠导致的误差，并未评估预训练 Transformer 在时间分类中的适用性

---

## 381. Improving Training Efficiency and Reducing Maintenance Costs via Language Specific Model Merging

**arXiv ID:** 2601.16127 | [PDF](https://arxiv.org/pdf/2601.16127v1)

**作者:** Alphaeus Dmonte `[一作]` (George Mason University), Mark Arehart `[通讯]` (Qualtrics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在多语言大模型微调中使用模型合并（merge）技术，提出并评估了基于TIES、DARE、KnOTS三种合并方法的“先训练单语模型再合并”策略。

**💡 创新点**

创新点在于首次从计算效率和维护成本角度系统评估多语言模型合并的优势，证明其可在保持性能一致的同时显著缩短训练时间和成本，并在企业专有数据集上验证通用性。

**🔧 技术方法**

采用Llama-3.1-8b-Instruct与Llama-3.2-3b-Instruct作为基模型，使用LoRA微调，结合TIES、DARE、KnOTS三种合并算法；在训练阶段并行训练单语模型，然后按加权或SVD方式合并。

**📊 数据集**

使用公开数据集：WikiLingua（摘要）、mCSQA（常识推理）、MultilingualSentiment（情感分析），以及一份涵盖英语、德语、法语、西班牙语、日语的企业专有摘要任务数据集。

**📈 对比分析**

将合并模型与传统“全数据重训练”（COMB）以及单语模型（INDV）做对比。实验表明：合并模型在摘要和推理任务上与COMB相当甚至略优，在情感分析任务上略逊；初始训练时间可减至35–50%，更新/新增语言时训练时间和成本可降低70%以上。

**⚠️ 局限性**

局限性包括仅测试Llama系列模型，未探讨其他模型族；语言覆盖仅为5种中高资源语言，未覆盖低资源或极大语言集；合并后性能在某些任务（如情感分类）略有下降。

---

## 382. A Case for Hypergraphs to Model and Map SNNs on Neuromorphic Hardware

**arXiv ID:** 2601.16118 | [PDF](https://arxiv.org/pdf/2601.16118v1)

**作者:** Marco Ronzani `[一作]` (Politecnico di Milano), Cristina Silvano `[通讯]` (Politecnico di Milano)

**通讯引用:** 3900 | [OpenAlex ID](https://openalex.org/A5031461662)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

将脉冲神经网络（SNN）映射到神经形态硬件时，从图模型转为超图模型，提出基于超边重叠的划分与谱法置放等新启发式算法；

**💡 创新点**

创新点在于引入超图抽象以显式捕捉同源脉冲共享（synaptic reuse）与连接局部性（connections locality）两大属性，利用二阶亲和力指导划分，利用谱嵌入进行初始布局，并通过多级划分与力导向细化实现高质量映射；

**🔧 技术方法**

采用超图划分（层级、重叠贪婪、顺序划分）、谱嵌入置放、Hilbert曲线初始置放、力导向细化和最短距离置放等技术，结合硬件约束模型；

**📊 数据集**

使用多种层化与循环SNN模型，包括16_model、64_model、LeNet、AlexNet、VGG11、MobileNet、Allen V1以及随机生成的生物启发式网络（如液态状态机、反馈SNN），以及公开的图像识别网络；

**📈 对比分析**

通过对比各种划分/置放组合的能耗‑延迟积（ELP）、连通度、分区数、能耗、延迟、拥塞等指标，实验表明基于超图的划分与谱置放可在多种硬件规模下比现有基于图的工具提升约1–2倍的映射质量，且在大规模网络上保持线性可扩展；

**⚠️ 局限性**

限制主要在于算法仍以单芯片为前提，未考虑多芯片互连；划分算法的时间复杂度仍受超边大小影响；部分置放方法对节点序列高度依赖，对高度密集或无层结构网络的效果有限。

---

## 383. Distillation-based Layer Dropping (DLD) Effective End-to-end Framework for Dynamic Speech Networks

**arXiv ID:** 2601.16117 | [PDF](https://arxiv.org/pdf/2601.16117v1)

**作者:** Abdul Hannan `[一作]` (University of Trento), Alessio Brutti `[通讯]` (Fondazione Bruno Kessler)

**通讯引用:** 1084 | [OpenAlex ID](https://openalex.org/A5066363315)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种基于知识蒸馏的动态层丢弃框架（DLD），实现将静态语音模型转换为可动态调整层数的模型，显著提升不同资源配置下的性能-计算折中；

**💡 创新点**

创新点在于将知识蒸馏与随机层丢弃结合，端到端训练动态学生模型，并通过对齐学生与教师的潜在嵌入分布，解决了高/低丢弃下性能退化的问题；

**🔧 技术方法**

核心技术包括：随机层丢弃（Bernoulli门控）、KL散度嵌入对齐、CTC解码、知识蒸馏；框架实现于Conformer和WavLM两大架构；

**📊 数据集**

实验数据集为公开语音识别数据集LibriSpeech‑1000和TED‑LIUM v3；还在FSC命令识别任务上验证了通用性；

**📈 对比分析**

与随机丢弃（RD）、输入驱动丢弃（I3D）、基线从零训练模型（RD_sc）以及Learnable Masking方法对比，DLD在低/无丢弃时可降低0.5–2.3 % WER，在高丢弃时提升4–8 % WER，同时训练时间缩短至基线的约三分之一；

**⚠️ 局限性**

局限性包括：目前仅在语音任务上验证，需进一步探索更大规模模型与其他任务的适用性；层丢弃概率设为0.5，未对不同概率进行系统分析；

---

