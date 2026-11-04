# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-11-04 | 今日论文总数: 794

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Counterexample-driven autonomous research with application to excess radiated energy in a Z-pinch

**arXiv ID:** 2611.03721 | [PDF](https://arxiv.org/pdf/2611.03721v1)

**作者:** Bowen Zhu `[一作]` (Xi'an Jiaotong University), Jian Wu `[通讯]` (Xi'an Jiaotong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种开源框架，利用语言模型代理追求由人类定义的物理猜想，并在此框架下进行了一项82.6小时的研究，解决了Z-pinch中辐射能量超过内爆壳动能的长期问题。

**💡 创新点**

创新点在于将反例驱动的研究方法应用于物理学，通过构建一个可执行的计算物理测试平台，解决了物理理论与现实世界的互动问题，并实现了人机协作的有效设计。

**🔧 技术方法**

使用了计算物理技术，结合了数值求解器、物理模型和诊断工具的开发与验证，采用了FLASH 4.8进行三维辐射-磁流体动力学模拟。

**📊 数据集**

研究使用了Z-pinch实验中的铝线阵列数据，具体参数包括15微米的线径和20毫米的轴向长度，结合了激光影像、法拉第旋转和X射线二极管等多种诊断手段。

**📈 对比分析**

通过与传统的Sweet-Parker重联速率猜想进行比较，发现更强的螺旋变形减弱了横向压缩，导致更多的磁能和热能被储存，而宽化的尾部等离子体轮廓则延迟了辐射的释放，恢复了早期的辐射缺口。

**⚠️ 局限性**

限制在于尽管框架能够有效地进行反例驱动的研究，但在长时间的计算过程中，仍然需要人类的干预来确保科学判断和研究方向的正确性。

---

## 2. Auditing a Model-Hub Benchmark: Data Quality, Training Budgets, and Side Information in HuggingBench

**arXiv ID:** 2611.03727 | [PDF](https://arxiv.org/pdf/2611.03727v1)

**作者:** Michal Podstawski `[一作]` `[通讯]` (NASK National Research Institute), Michal Podstawski (NASK National Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对HuggingBench基准进行了审计，检查数据质量、训练预算和侧面信息的有效性。

**💡 创新点**

提出了如何检测和修复基准中独立导出工件的完整性失败，并分析了这些失败如何影响后续结论。

**🔧 技术方法**

使用了数据验证、推荐方法的复制和控制实验等技术。

**📊 数据集**

使用了Hugging Face API收集的HuggingKG知识图谱及其相关数据集。

**📈 对比分析**

通过与原始报告的推荐方法进行比较，发现修正后的知识图谱在每个种子上都提高了Recall@10的表现，尤其是对于较少被喜欢的项目。

**⚠️ 局限性**

存在的局限性包括：喜欢数作为反馈的弱形式，数据仅为快照，冷启动协议模拟新项目而非跟踪模型发布过程。

---

## 3. Functional-Group Knowledge Is Structured in Language Models Before It Is Spoken

**arXiv ID:** 2611.03684 | [PDF](https://arxiv.org/pdf/2611.03684v1)

**作者:** Tran Nhat Khang Nguyen `[一作]` (University of Nebraska), Janghoon Ock `[通讯]` (University of Nebraska)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究分析了五种开源语言模型在处理化学分子结构时的内部表示，探讨了它们如何识别和组织功能团的信息。

**💡 创新点**

创新点在于通过内部表示分析，揭示了模型在未表达知识时的潜在能力，强调了输出结果与模型内部知识之间的差异。

**🔧 技术方法**

使用了五种开源的8B语言模型，包括一个通用基础模型、三个经过化学训练的模型和一个推理蒸馏模型。

**📊 数据集**

使用了92种小型有机分子，涵盖20种功能团，数据集包括分子的IUPAC名称和SMILES字符串。

**📈 对比分析**

通过比较模型在920个提示下的生成准确性，发现不同模型的表现差异显著，化学训练模型的准确性普遍较高，但仅依赖输出结果无法全面评估模型的化学知识。

**⚠️ 局限性**

限制在于模型的输出结果可能无法完全反映其内部知识结构，且不同模型在处理提示时的表现不一致，可能导致对模型能力的误判。

---

## 4. Alignment Faking Depends on the Value, Not Just the Model

**arXiv ID:** 2611.03650 | [PDF](https://arxiv.org/pdf/2611.03650v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 5. EMBERS: Empirically Motivated Robustness Testing of Deep Research Systems via Defect-Guided Fuzzing

**arXiv ID:** 2611.03723 | [PDF](https://arxiv.org/pdf/2611.03723v1)

**作者:** Shuochuan Li `[一作]` (Tianjin University), Junjie Chen `[通讯]` (Tianjin University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于缺陷引导的模糊测试框架，用于深度研究系统的鲁棒性测试，通过对执行轨迹的实证研究，识别出九种常见缺陷类别。

**💡 创新点**

创新点在于通过实证研究识别缺陷类别，并基于这些缺陷设计了针对性的测试方法，系统性地探索潜在的鲁棒性失败。

**🔧 技术方法**

使用了缺陷引导的变异、网络搜索辅助的保真度保证和多样性意识的种子维护等技术。

**📊 数据集**

使用了来自三个代表性深度研究系统的1,932个执行轨迹，并在多个基准数据集上进行评估。

**📈 对比分析**

与现有方法（如DR-Arena和RobustAlpacaEval）相比，该框架在相同的测试预算下发现的失败输入数量是最强基线的3.21倍，显示出更高的失败发现能力。

**⚠️ 局限性**

局限性在于当前的变异过程可能无法完全保留答案决定性信息，且某些生成的输入可能会改变原始任务，导致评估的失败输入不一定代表真实的鲁棒性失败。

---

## 6. From Logs to Faults: Reference-Free Fault Localization for Retrieval-Augmented Generation Pipelines

**arXiv ID:** 2611.03725 | [PDF](https://arxiv.org/pdf/2611.03725v1)

**作者:** Hira Naveed `[一作]` (Monash University), Omar Haggag `[通讯]` (Monash University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为RAGLog的工具，用于在检索增强生成（RAG）管道中检测和定位故障，利用执行日志而无需参考答案或代码仪器。

**💡 创新点**

RAGLog的创新点在于其无参考的跨阶段故障定位能力，能够识别下游阶段的故障，并支持生产环境中的调试和监控。

**🔧 技术方法**

使用了规则基础、嵌入基础和大型语言模型（LLM）基础的检查方法，结合执行日志进行故障检测。

**📊 数据集**

使用了RAGLog Faults数据集，该数据集包含来自五个RAG管道的3,134次运行记录，带有阶段级故障标签。

**📈 对比分析**

与Claude Code和Codex等通用编码代理进行比较，RAGLog在自然故障运行中获得了0.73的F1分数，优于Claude Code的0.58和Codex的0.52；在注入故障运行中，RAGLog的效果也相当，分别为0.84、0.80和0.82。

**⚠️ 局限性**

RAGLog的局限性在于其可能产生较高的假阳性率，尤其是在检测到故障时，可能会标记出不必要的阶段。

---

## 7. Metadata-Conditioned Audio Transformers for Adaptive Respiratory Sound Classification

**arXiv ID:** 2611.03769 | [PDF](https://arxiv.org/pdf/2611.03769v1)

**作者:** George Kontogiannis `[一作]` (University of Patras), Sotiris Nikoletseas `[通讯]` (University of Patras)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了如何利用元数据对呼吸音进行分类，并提出了多种基于FiLM的条件化Transformer模型，包括Token‑Aware FiLM和Soft‑Factorized FiLM，以提高ICBHI数据集上的性能。

**💡 创新点**

首次将元数据作为结构化特征对Audio Spectrogram Transformer进行层级特征调制，并引入Soft‑Factorized FiLM通过学习软掩模实现元数据源的特征子空间分配。

**🔧 技术方法**

使用Audio Spectrogram Transformer、FiLM、Token‑Aware FiLM、Soft‑Factorized FiLM、温度衰减软掩模、重叠与覆盖正则化以及SpecAugment等技术。

**📊 数据集**

在ICBHI 2017呼吸音数据库的官方60‑40%训练‑测试划分上进行实验。

**📈 对比分析**

与多种基线和最新方法对比，Soft‑FiLM在4类任务上达64.11%（比基线+3.94%），在2类任务上获得72.40%（居首），显著提升性能。

**⚠️ 局限性**

方法仍受已见元数据分布限制，未针对未见设备或采集条件的鲁棒性进行充分评估；软掩模学习可能对训练不稳定。

---

## 8. Does the Trust Boundary Hold over Time? A Long-Horizon Benchmark for Receiver-Directed Federated Learning

**arXiv ID:** 2611.03724 | [PDF](https://arxiv.org/pdf/2611.03724v1)

**作者:** Ting Xu `[一作]` (University of Calgary), Henry Leung `[通讯]` (University of Calgary)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了接收者导向的联邦学习中的信任边界，提出了一种基准测试方法，评估在多个回合中信任边界是否保持有效。

**💡 创新点**

创新点在于提出了一种固定的评估协议，能够在长时间范围内跟踪和评估模型的决策质量和最终结果，并引入了独立审计层来控制部署风险。

**🔧 技术方法**

使用了逻辑回归和神经网络等多种技术，特别是CS-Gate作为评估方法，并结合了基于变异自适应的界限进行审计。

**📊 数据集**

数据集来自两个公共合成源，包含约60,000条加拿大和974条美国记录，涉及13个共享变量。

**📈 对比分析**

与其他方法的比较显示，FedFomo和CS-Gate在每个诚实组中都保持了正的平均增益，尤其是在面对攻击者时表现良好。

**⚠️ 局限性**

限制在于研究仅基于合成数据和特定攻击模型，未能证明其对任意对手的鲁棒性，且审计的保证仅覆盖合成审计目标。

---

## 9. Quantifying out-of-distribution data severity with Wasserstein distances: benchmarking large language models for alloy property prediction

**arXiv ID:** 2611.03767 | [PDF](https://arxiv.org/pdf/2611.03767v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 10. RATSIFT: Resolution-Adaptive Transformer for Solar Irradiance Forecasting and Temporal Downscaling

**arXiv ID:** 2611.03738 | [PDF](https://arxiv.org/pdf/2611.03738v1)

**作者:** Jesto Peter `[一作]` (Bowling Green State University), Hossein Moradi Rekabdarkolaee `[通讯]` (Bowling Green State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究开发了一种基于Transformer的架构，用于太阳辐射的同时预测和时间下采样，能够处理任意分辨率的输入数据，并生成训练数据中表示的任意分辨率的输出。

**💡 创新点**

创新点在于提出了一种分辨率自适应的Transformer框架，能够在不同时间分辨率下进行预测和下采样，同时通过遗传算法优化模型超参数，提高训练效率。

**🔧 技术方法**

使用了基于Transformer的架构，并结合遗传算法进行超参数优化。

**📊 数据集**

使用了两个太阳辐射数据集：一个来自国家太阳辐射数据库的全球数据集（15分钟分辨率），另一个是本地气象站的地面测量数据集（5分钟分辨率）。

**📈 对比分析**

与四种最先进的Transformer模型（Informer、Autoformer、FEDformer和非平稳Transformer）进行了比较，提出的模型在RMSE、训练时间和预测时间上表现更优，RMSE值分别为28.785、29.544、30.157和32.132 W/m²。

**⚠️ 局限性**

限制在于模型对超参数选择的敏感性，以及在高变动天气条件下的预测不确定性。

---

## 11. te-comp: A Complete MLIR Compiler Generating Self-Contained C Inference Engines for MCUs

**arXiv ID:** 2611.03710 | [PDF](https://arxiv.org/pdf/2611.03710v1)

**作者:** Van Chan Ngo `[一作]` `[通讯]` (TeComp AI), Van Chan Ngo (TeComp AI)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了第一个完整的MLIR编译器，能够把标准模型转换成自包含、无依赖的C推理引擎，用于MCU部署。

**💡 创新点**

创新点在于：①使用自定义TE dialect对量化神经网络进行多层优化；②实现模型专用代码生成（无运行时解释器、无外部核库）；③支持LLM推理并实现stateful int4 KV缓存；④通过静态内存规划和指令级优化显著节省SRAM/Flash。

**🔧 技术方法**

技术主要包括MLIR多级编译管道、TOSA→TE融合、静态arena内存规划、ARM DSP指令嵌入、C代码生成（RTOS裸机或FreeRTOS集成）。

**📊 数据集**

评测数据集：10个MCUNet CNN基准（ImageNet、Visual Wake Words）以及TinyStories‑v2（GPT‑Neo 8.3M 参数）作为LLM模型。

**📈 对比分析**

与MCUNet TinyEngine、TFLite Micro等基准比较：SRAM平均节省38%（最大5.9×）、Flash平均节省15%；在TinyStories‑v2上实现≈1 token/s 的交互式文本生成，且无任何框架依赖。

**⚠️ 局限性**

局限性：目前仅支持Cortex‑M（M4/M7）和少量LLM模型；对更大规模网络的误差累积仍有2–9 LSB；对其他 MCU 架构（Xtensa、Tricore）未实现，需进一步优化和验证。

---

## 12. Reasoning is topologically simple

**arXiv ID:** 2611.03739 | [PDF](https://arxiv.org/pdf/2611.03739v1)

**作者:** Chris Kapulkin `[一作]`, Nathan Kershaw `[通讯]` (University of Western Ontario)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并研究了一个假设，即推理文本的共现拓扑比非推理文本更简单，而高质量的推理比低质量的推理更简单。

**💡 创新点**

创新点在于使用共现持久同调来直接评估推理质量，而不是依赖于嵌入方法，提供了可解释的结果。

**🔧 技术方法**

使用了共现持久同调技术来分析文本的拓扑特征。

**📊 数据集**

使用了来自古腾堡计划的数学、哲学、历史和短篇小说等文本数据集，以及OpenAI的数学论文和人类撰写的论文。

**📈 对比分析**

通过四个实验比较不同文本类型的共现拓扑，结果显示数学文本和人类撰写的论文在拓扑上更简单，且更高能力的模型具有更简单的推理拓扑。

**⚠️ 局限性**

限制在于无法完全隔离推理质量与文本其他属性的影响，样本量较小，且实验4仅限于一个模型，结果的普遍性尚未验证。

---

## 13. Multi-layer Indexer: Overlapping Selection and Attention in DeepSeek Sparse Attention

**arXiv ID:** 2611.03768 | [PDF](https://arxiv.org/pdf/2611.03768v1)

**作者:** Serdarcan Dilbaz `[一作]` (Writer AI Research), Daniel M. Bikel `[通讯]` (Writer AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的稀疏注意力机制DeepSeek Sparse Attention (DSA)，通过在每个注意力层前添加一个快速索引器，使得解码过程中的注意力计算可以并行进行，从而提高长上下文推理的效率。

**💡 创新点**

创新点在于通过让每个层的注意力可以并行运行，而不是依赖于前一个层的选择，从而减少了解码时间，并且无需重新训练或增加新的权重。

**🔧 技术方法**

使用了稀疏注意力机制和快速索引器的组合技术，特别是在DeepSeek-V3.2和GLM-5模型中实现了这一机制。

**📊 数据集**

使用了DeepSeek-V3.2和GLM-5模型的多个数据集，具体上下文长度包括10K到195K的令牌。

**📈 对比分析**

与IndexCache方法进行比较，DSA在相同的保留层数下解码速度更快，DeepSeek-V3.2在保留2层时速度提高了14.5%，GLM-5提高了9.8%。在更高的上下文长度下，DSA的性能提升更为显著。

**⚠️ 局限性**

限制在于随着解码请求数量的增加，性能提升会下降。此外，DSA在GLM-5模型上的表现不如在DeepSeek-V3.2模型上，且GLM-5的某些实现可能不如预期。

---

## 14. RuntimeTickets: Source-Grounded Security Obligations for Agent Skills

**arXiv ID:** 2611.03772 | [PDF](https://arxiv.org/pdf/2611.03772v1)

**作者:** Kuiyi Gao `[一作]` (University of North Carolina), Bowei He `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种协议，旨在执行和审计第三方技能的安全要求，以防止恶意技能在合法工作流程中隐藏数据泄露或输出操控。

**💡 创新点**

创新点在于通过运行时检查和审计机制，确保技能的执行符合用户的安全要求，并且提供可验证的执行证据。

**🔧 技术方法**

使用了一种模型生成的票据机制，该机制在每个任务和技能运行之前提出安全义务，并通过模型无关的核心进行执行和审计。

**📊 数据集**

使用了从发布的恶意技能数据集中派生的风险模型和对抗-良性技能对。

**📈 对比分析**

与现有的技能防御方法（如AgentBound和SkillGuard）进行比较，结果显示该协议在网络出口边界下没有让任何攻击通过，同时没有阻止任何良性运行，表现优于其他方法。

**⚠️ 局限性**

限制在于该协议依赖于模型生成的票据，可能在某些情况下无法完全覆盖所有潜在的攻击向量。

---

## 15. Unsatz: Evaluating Scientific Discovery Without a Given Ansatz

**arXiv ID:** 2611.03713 | [PDF](https://arxiv.org/pdf/2611.03713v1)

**作者:** Ethan Ji `[一作]` (University of California), Yizhou Sun `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新的基准，要求大型语言模型（LLM）代理自行构建科学模型的结构（ansatz），而不仅仅是估计参数。

**💡 创新点**

创新点在于让代理在没有给定变量或答案形式的情况下，独立设计实验并构建模型结构，这在现有的自动化方程发现中是前所未有的。

**🔧 技术方法**

使用了大型语言模型（LLM）作为代理，结合了实验设计、数据分析和假设提出的能力。

**📊 数据集**

使用了来自20个领域的80个系统生成的模拟数据集，这些系统包括普通微分方程、延迟方程、部分微分方程等。

**📈 对比分析**

与经典方程发现方法相比，LLM代理在44个系统中几乎没有成功，而在给定真实方程的情况下，经典方法能够解决大部分问题。LLM代理的成功率在21%到60%之间，表现出较强的能力。

**⚠️ 局限性**

局限性在于，尽管代理能够识别系统的变量和交互关系，但在构建适当的方程时仍然存在科学错误，且在选择修订时更倾向于降低训练误差而非提高预测准确性。

---

## 16. Enhancing Stock Market Information Retrieval with Retrieval-Augmented Generation

**arXiv ID:** 2611.03747 | [PDF](https://arxiv.org/pdf/2611.03747v1)

**作者:** Huan Phong Thai `[一作]` (FPT University), Hong Nhung Nguyen `[通讯]` (FPT University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种检索增强生成（RAG）系统，用于生成与股市信息相关的响应，旨在提高金融文本生成的准确性和相关性。

**💡 创新点**

创新点在于将检索方法与生成能力相结合，解决了传统NLP系统在动态金融环境中保持上下文准确性和生成相关响应的挑战。

**🔧 技术方法**

使用了检索增强生成（RAG）模型，结合了检索机制和生成模型，采用了LaBSE模型进行文本嵌入，并使用FAISS进行高效的相似性检索。

**📊 数据集**

数据集来自越南知名在线新闻平台VnExpress，包含30篇与股市相关的新闻文章，涵盖市场趋势、公司股票、外资流动等多个方面。

**📈 对比分析**

通过ROUGE指标（ROUGE-1、ROUGE-2和ROUGE-L）评估生成文本与参考文本之间的相似性，实验结果显示高ROUGE分数，表明生成文本与参考文档高度一致，平均响应时间为1.16秒，表明系统响应迅速。

**⚠️ 局限性**

限制在于模型可能在处理复杂金融查询时仍存在挑战，未来工作将集中在进一步提高模型的响应准确性和优化计算效率上。

---

## 17. Towards Automatic Heart Murmur Characterization from Multi-spot Phonocardiograms

**arXiv ID:** 2611.03654 | [PDF](https://arxiv.org/pdf/2611.03654v1)

**作者:** Marcelo Nogueira `[一作]` (INESC TEC and University of Porto), Alípio M. Jorge `[通讯]` (INESC TEC and University of Porto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本研究将自动心脏杂音特征化定义为五个患者级分类任务，并比较了三种多点策略，旨在从主动脉、肺动脉、二尖瓣和三尖瓣听诊位置的录音中提取更丰富的杂音描述。

**💡 创新点**

创新点在于将心脏杂音特征化扩展到五个临床相关的属性，并通过不同的模型架构（包括独立训练的分类器、同步多输入CNN和统一的多输入多输出CNN）进行比较。

**🔧 技术方法**

使用了经典的机器学习分类器（如支持向量机、随机森林、XGBoost等）和深度学习模型（如CNN），并采用了最大严重性晚融合规则。

**📊 数据集**

使用了CirCor DigiScope数据集，该数据集包含735名患者的完整四点录音，并提供了专家注释的杂音时机、形状、音调、质量和等级。

**📈 对比分析**

与传统的二元检测方法相比，最佳的晚融合配置在五个任务中的无加权平均灵敏度达到85.56%到93.81%之间，而多输入CNN的表现落后于晚融合7-16个百分点，统一多输出模型的表现为72.59%到79.40%。

**⚠️ 局限性**

限制在于所有实验仅使用单一数据集，且只包含735名具有完整录音的患者，此外，所有属性任务严重不平衡，某些临床相关类别样本极少，可能影响结果的普遍性。

---

## 18. Latent Cluster Separation as a Regularizer for Lifelong Anomaly Detection

**arXiv ID:** 2611.03652 | [PDF](https://arxiv.org/pdf/2611.03652v1)

**作者:** Sabarish Radhakrishnan `[一作]` (John Champe High School), Nathalie Japkowicz `[通讯]` (American University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了在终身异常检测中，如何通过分离潜在聚类作为正则化器来改善模型的检测质量和向后迁移能力。作者将潜在聚类分离和异常评分机制分开进行分析，发现潜在聚类分离作为训练时的正则化器能够提高检测质量和向后迁移能力。

**💡 创新点**

创新点在于将潜在聚类分离和异常评分机制分开进行研究，明确了两者对性能提升的不同贡献，发现潜在聚类分离在不改变异常评分的情况下能够有效提高检测质量，而距离评分则存在不稳定性。

**🔧 技术方法**

使用了变分自编码器（VAE）作为基础模型，通过引入可学习的聚类中心和吸引-排斥损失来实现潜在聚类分离，同时比较了重构误差和基于距离的异常评分机制。

**📊 数据集**

使用了三个公共的入侵检测数据集进行实验：ADFA-LD、NSL-KDD和UNSW-NB15，涵盖了不同的特征表示。

**📈 对比分析**

通过控制实验比较了三种配置（基线、潜在聚类分离与重构评分、潜在聚类分离与距离评分），结果显示潜在聚类分离与重构评分在ADFA-LD和NSL-KDD上表现优于基线，而距离评分在UNSW-NB15上表现更好。潜在聚类分离与重构评分在检测质量和稳定性上表现一致，而距离评分则表现出较大的不稳定性。

**⚠️ 局限性**

限制在于结果依赖于数据集的特性，潜在聚类分离的优势在不同数据集上表现不一，且未能找到普适的规则。此外，潜在聚类分离的好处尚未完全机制化解释，未来需要进一步研究。

---

## 19. Fixed Detectors, Moving Code: How Well Do Static Security Analyzers Hold Up over a Decade of C/C++ Evolution?

**arXiv ID:** 2611.03742 | [PDF](https://arxiv.org/pdf/2611.03742v1)

**作者:** Junhua Tang `[一作]` (University at Buffalo), Zhengsong Zhang. Haipeng Cai `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文测试了静态应用安全测试（SAST）工具在代码演变中的有效性，分析了来自1116个开源项目的20000个易受攻击和修复的C/C++案例，涵盖2016年至2025年的源代码。

**💡 创新点**

创新点在于提出了SAST可持续性的定义，并通过三种研究设计（年度人群、项目和标签匹配的年份对、项目内轨迹）来评估工具在时间推移中的有效性、可执行性和成本。

**🔧 技术方法**

使用了CodeQL、Infer和Semgrep等多种SAST工具，并在不同版本下进行比较。

**📊 数据集**

数据集包括来自NVD、PrimeVul和OSS-Fuzz/OSV的20000个C/C++案例，每年各1000个易受攻击和修复的案例。

**📈 对比分析**

通过项目匹配的年份对分析，发现CodeQL在所有年份间的F1均为负值，而Infer在每个间隔中均为正值。整体性能表现显示，CodeQL的F1从0.1232提升至0.2915，但并未改善对易受攻击和修复代码的区分能力。

**⚠️ 局限性**

限制在于SAST工具的有效性可能受到代码演变、工具版本变化和分析覆盖范围的影响，且工具的性能在不同项目和年份间存在显著差异。

---

## 20. A new class of optimal binary linear complementary pairs of codes

**arXiv ID:** 2611.03733 | [PDF](https://arxiv.org/pdf/2611.03733v1)

**作者:** Yansheng Wu `[一作]` (Nanjing University of Posts and Telecommunications), Jing Chen `[通讯]` (Nanjing University of Posts and Telecommunications)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文提出了一种基于Belov码的新构造方法，生成满足Griesmer界限的最优二进制线性互补对（LCP）并聚焦θ=4的情形；

**💡 创新点**

创新点在于将Belov码（Belov codes）引入LCP构造，并通过删除子空间实现比之前θ=3构造更灵活的参数、能获得更低最小距离的最优LCP；

**🔧 技术方法**

使用的技术包括：Belov码与Solomon–Stiffler码的代数结构、矩阵分块（I_k : M）、可逆性判定（I_k+MN可逆）以及列/行的循环移位与列交换来保证M'N'为下三角矩阵；

**📊 数据集**

没有使用公开数据集，而是以理论参数（k、d、h、u_i、θ）构造示例码（如k=7,d=49,n=101；k=5,d=18,n=37）作为实验实例；

**📈 对比分析**

通过与Solomon–Stiffler码以及之前的θ=3 Belov码对比，表明在相同k值下所需最小距离与长度显著降低（如k=6时从d≥96,n≥189降至d≥45,n≥91），证明了构造方法的性能提升；

**⚠️ 局限性**

限制在于目前只讨论θ=4的情况，参数范围受限于u_1≥4且k满足特定不等式，且构造过程对子空间选择及列交换较为繁琐，缺乏通用的算法实现与实验验证。

---

## 21. The Web Doesn't Sit Still: Adversarial Self-Evolving Attacks on Search Agents

**arXiv ID:** 2611.03754 | [PDF](https://arxiv.org/pdf/2611.03754v1)

**作者:** Geyuan Wang `[一作]` (Nanyang Technological University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种对抗性自我进化框架，旨在通过双层优化来评估搜索代理的安全性，揭示其脆弱性。

**💡 创新点**

创新点在于引入了对比回滚进化优化（CREO）方法，打破了对静态人造模板的依赖，实现了攻击策略的动态自我进化。

**🔧 技术方法**

使用了对比回滚进化优化（CREO）作为核心求解器，结合双层优化模型来驱动攻击者和防御者的共同进化。

**📊 数据集**

在三个多跳问答基准（HotpotQA、2WikiMultihopQA、MuSiQue）上进行了实验，每个数据集采样100个训练和100个测试查询。

**📈 对比分析**

与静态基线（如基于规则的篡改和优化基础的毒化攻击）相比，CREO在攻击效果上表现出显著优势，尤其在动态和适应性攻击下，成功揭示了搜索代理的深层脆弱性。

**⚠️ 局限性**

当前框架主要验证了多跳问答设置下的对抗性自我进化，尚未扩展到更复杂的自主搜索场景，未来工作将集中在开发能够应对进化攻击的自适应防御机制上。

---

## 22. ReVi: Recursive Recap Video Generation

**arXiv ID:** 2611.03674 | [PDF](https://arxiv.org/pdf/2611.03674v1)

**作者:** Kyrie You Wu `[一作]` (Beijing University of Posts and Telecommunications), Litian Zhang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了从小说生成视频摘要的任务，提出了一种新的框架ReVi，旨在在固定的观看时间内生成自包含的视听摘要。

**💡 创新点**

创新点在于首次提出了小说到摘要视频生成的任务，并设计了一个双循环多代理框架，内循环负责具体的摘要生成，外循环则通过经验改进编辑和元政策。

**🔧 技术方法**

使用了双循环多代理框架，结合因果网络模型、叙事持续时间理论和事件分割理论来规划故事内容、叙述时间和对应镜头。

**📊 数据集**

使用了FilmEval和Recap32数据集，涵盖不同难度的叙事和较长的小说。

**📈 对比分析**

与多个基线方法（如Direct LLM、ViMax等）进行比较，ReVi在摘要质量和持续时间合规性上均表现优越，且在质量评分上获得了人类评估的最高分。

**⚠️ 局限性**

限制在于需要进一步测试更多的小说和独立生成，并加强对摘要忠实度的人类评估。

---

## 23. The role of injected noise in diffusion model generalization

**arXiv ID:** 2611.03764 | [PDF](https://arxiv.org/pdf/2611.03764v1)

**作者:** Vincent Emmerling `[一作]` (Bielefeld University), David Kappel `[通讯]` (Bielefeld University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了扩散模型在不同数据集上训练后生成样本的相似性，探讨了在采样过程中注入噪声的作用。

**💡 创新点**

发现注入噪声在采样过程中的不同阶段对样本一致性有显著影响，尤其是在样本特征尚未完全确定的阶段。

**🔧 技术方法**

使用了扩散模型（DMs）和随机微分方程（SDE）进行采样，采用了UNet网络进行实验。

**📊 数据集**

使用了CelebA数据集，该数据集包含10^5张降采样的图像。

**📈 对比分析**

通过余弦相似度评估生成样本之间的距离，结果表明共享注入噪声序列的样本一致性显著高于不共享的样本。

**⚠️ 局限性**

研究中未提及具体的局限性，但可能存在对不同模型架构和噪声序列的适用性限制。

---

## 24. LOLPlay: Benchmarking Real-Time Visual Agents in League of Legends

**arXiv ID:** 2611.03685 | [PDF](https://arxiv.org/pdf/2611.03685v1)

**作者:** Zhaojian Yu `[一作]` `[通讯]`, Zhaojian Yu

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个新的基准，评估视觉智能体在实时环境中协调推理和行动的能力，特别是在《英雄联盟》游戏中，涵盖36个特定于英雄的任务。

**💡 创新点**

创新点在于引入了一个实时的基准测试，能够测量智能体在动态环境中完成多项任务的能力，并且通过82个可执行的子目标检查来评估部分进展与完成之间的差距。

**🔧 技术方法**

使用了GPT-6 Astra等模型进行实验，评估其在不同推理努力设置下的表现。

**📊 数据集**

使用了《英雄联盟》游戏中的自定义任务和环境，涵盖了资源管理、战斗、合作和目标等多个方面。

**📈 对比分析**

与其他模型（如GPT-6.1 Sol和DeepSeek-v4.1-Flash）进行比较，GPT-6 Astra在低和中等努力下的任务成功率最高为25.0%，而在中等努力下的子目标成功率为40.3%。然而，增加推理努力并未显著提高任务完成率。

**⚠️ 局限性**

限制在于基准测试仅覆盖每个英雄的一个目标，而未能涵盖MOBA游戏的全部多样性。此外，某些要求可以通过被动进展来满足，这使得任务成功与子目标成功的解释受到限制。

---

## 25. SSPE: Robust Sequential Staypoint Extraction in Degraded Trajectory Data

**arXiv ID:** 2611.03737 | [PDF](https://arxiv.org/pdf/2611.03737v1)

**作者:** Yueyang Liu `[一作]` (Emory University), Andreas Züfle `[通讯]` (Emory University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种新的序列停留点提取算法（SSPE），用于从受损的轨迹数据中提取语义上有意义的停留点。

**💡 创新点**

创新点在于使用动态质心替代静态空间锚点，并引入n步前瞻启发式，以容忍瞬态坐标漂移，同时利用时间桥接机制防止在信号丢失期间的人工碎片化。

**🔧 技术方法**

使用了动态质心更新、n步前瞻启发式和无监督参数估计框架等技术。

**📊 数据集**

使用了ExtraSensory数据集，这是一个包含手动验证的真实停留点的移动数据集。

**📈 对比分析**

与Trackintel的滑动窗口实现进行比较，SSPE在F1分数上表现更优，同时计算效率更高，处理时间略快于Trackintel。

**⚠️ 局限性**

限制在于算法的性能可能依赖于数据集的特性，且在极端情况下可能仍然受到噪声的影响。

---

## 26. Minimal Superior Penalties: Characterising and Calibrating Decision-Aware Proper Scoring Rules

**arXiv ID:** 2611.03734 | [PDF](https://arxiv.org/pdf/2611.03734v1)

**作者:** Rouhollah Ahmadian `[一作]` `[通讯]` (Amirkabir University of Technology), Rouhollah Ahmadian (Amirkabir University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文探讨了分类器训练中验证指标的优越性要求，提出了一个新的惩罚机制以恢复优越性，并通过理论推导和实验验证了该机制的有效性。

**💡 创新点**

创新点在于提出了一个闭合形式的最小惩罚（infimal penalty），该惩罚比以往文献中提出的惩罚小得多，并且在多个类别情况下保持优越性。

**🔧 技术方法**

使用了Brier分数和对数分数作为基础评分规则，并通过惩罚机制对其进行了改进。

**📊 数据集**

使用了MNIST和CIFAR-10数据集进行实验，比较了不同惩罚值下模型的表现。

**📈 对比分析**

与以往的惩罚机制相比，本文提出的最小惩罚在多个类别情况下表现出更好的性能，尤其是在十类和二十类的实验中，表现出显著的优越性。

**⚠️ 局限性**

限制在于实验设计未能完全隔离优越性与其他因素（如错误加权）之间的影响，且在不同类别数的情况下，结果的普适性尚待进一步验证。

---

## 27. GHOST Driver

**arXiv ID:** 2611.03649 | [PDF](https://arxiv.org/pdf/2611.03649v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e`

---

## 28. Physics-Informed Functional Tucker Method with RKHS Factors for Sparse Spatiotemporal Reconstruction

**arXiv ID:** 2611.03717 | [PDF](https://arxiv.org/pdf/2611.03717v1)

**作者:** Zhengyang Liu `[一作]` (SandGold AI Research), Zhiqiang Liu `[通讯]` (Southern University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为PhysFTM的物理信息功能Tucker模型，用于从固定稀疏传感器和缺失帧中重建物理动态。

**💡 创新点**

创新点在于结合了RKHS（再生核希尔伯特空间）参数化的Tucker模式因子和通过有限差分残差引入的偏微分方程（PDE）约束，从而提高了重建的物理一致性和准确性。

**🔧 技术方法**

使用了物理信息功能Tucker模型（PhysFTM），结合RKHS和有限差分方法进行优化。

**📊 数据集**

使用了Allen–Cahn、Navier–Stokes、三维热方程和三维Copernicus海洋再分析数据集进行实验。

**📈 对比分析**

与SDIFT、LRTFR、SIREN-MLP、PINNs和RecFNO等方法进行了比较，PhysFTM在大多数数据集和协议下的RMSE最低，尤其在缺失帧的离传感器位置上表现突出。

**⚠️ 局限性**

限制在于当前框架仅考虑从固定空间传感器收集的稀疏观察数据，缺失时间帧的情况，未来需要扩展到任意时空位置的观测设置，并考虑重建的不确定性量化。

---

## 29. Four Extensions of the Universal Law of Robustness: Log-Concave Covariates, Heterogeneous Mixtures, Vector-Valued Predictors, and Markovian Data

**arXiv ID:** 2611.03653 | [PDF](https://arxiv.org/pdf/2611.03653v1)

**作者:** Tobias J. Peter `[一作]` `[通讯]` (Zurich University of Applied Sciences), Tobias J. Peter (Zurich University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文扩展了Bubeck和Sellke的鲁棒性普遍法则，放宽了其结构假设，证明了在不同条件下的鲁棒性定律。

**💡 创新点**

创新点在于放宽了高斯型浓缩、协变量分布的均匀性、标量输出和样本独立性等假设，提出了适用于更广泛情况的鲁棒性定律。

**🔧 技术方法**

使用了Lipschitz插值、过参数化和浓缩测度等技术，结合了主定理来统一不同的扩展结果。

**📊 数据集**

使用了多种数据集，包括高斯型、异质混合和马尔可夫链等数据集，验证了理论结果的适用性。

**📈 对比分析**

通过与现有方法的比较，展示了在放宽假设后，鲁棒性定律仍然保持有效，且在样本条件下的性能表现良好。

**⚠️ 局限性**

限制在于某些假设的放宽可能导致复杂性增加，且在某些情况下，仍需满足特定的样本条件以确保鲁棒性定律的有效性。

---

## 30. FlowRig: Unified Skeleton Generation for 3D and 4D Meshes via Flow Matching

**arXiv ID:** 2611.03775 | [PDF](https://arxiv.org/pdf/2611.03775v1)

**作者:** Fengzhi Guo `[一作]` (Texas A&M University), Cheng Zhang `[通讯]` (Texas A&M University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种统一的3D和4D骨架生成框架，能够在单次前馈过程中生成时间对齐的骨架，同时处理单网格的3D绑定。

**💡 创新点**

创新点在于提供了一种无模板、数据高效的框架，支持可控的骨骼数量和不确定性估计，能够进行插值、预测和回溯。

**🔧 技术方法**

采用了基于流匹配的技术，通过流匹配变换器预测时间对齐的骨架表示，并使用图头和不确定性头进行拓扑预测和置信度估计。

**📊 数据集**

使用了多个数据集，包括Articulation-XL（静态3D模型）、ModelsResource（标准基准）和TexVerse（动态4D数据集）。

**📈 对比分析**

与传统方法（如Puppeteer和UniRig）相比，该模型在静态3D绑定中表现出色，保持了4D时间一致性，支持可控的骨骼数量，推理速度更快，并能够进行4D骨架预测。

**⚠️ 局限性**

限制在于4D配对数据仍然稀缺，模型需要从大规模的3D绑定数据初始化并适应中等规模的4D网格-骨架数据。

---

## 31. InterceptFuzz: Interception-based Stateful Fuzzing of Single-Process Targets

**arXiv ID:** 2611.03709 | [PDF](https://arxiv.org/pdf/2611.03709v1)

**作者:** Ilja Siroš `[一作]` (KU Leuven), Bart Preneel `[通讯]` (KU Leuven)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文提出了一种名为InterceptFuzz的中间件，旨在实现对单进程目标的状态感知模糊测试。该方法通过在目标代码中插入钩子，记录正常流量并注入模糊数据包，从而实现状态感知模糊测试。

**💡 创新点**

InterceptFuzz的创新点在于它能够在不修改现有无状态模糊引擎的情况下，利用这些引擎进行状态感知模糊测试。它通过前缀树模型来管理状态，并通过状态ID来选择特定的协议状态。

**🔧 技术方法**

使用了InterceptFuzz作为中间件，结合了无状态模糊引擎（如AFL++或libFuzzer），并通过插入钩子来实现数据包的观察和注入。

**📊 数据集**

评估使用了三个目标，分别是OpenThread的Nexus测试框架和ns-3网络模拟器的LTE和Wi-Fi模块。

**📈 对比分析**

与基线模糊器Concat和AFLNet进行比较，InterceptFuzz在OpenThread和ns-3 Wi-Fi上覆盖了显著更多的分支，并发现了七个之前未知的崩溃签名。尽管Concat在每次测试中覆盖了更多分支，但InterceptFuzz在状态感知测试中表现更佳。

**⚠️ 局限性**

限制在于InterceptFuzz假设目标行为是确定性的，因此在某些情况下（如依赖于墙钟时间或线程交错的目标），可能无法保证正确性。此外，每次执行仅能变更一个数据包，可能导致需要多个数据包共同变更的漏洞难以被发现。

---

## 32. ISAC-Assisted Proactive Handover in mmWave Networks: Value of Foresight and Localization Requirements

**arXiv ID:** 2611.03771 | [PDF](https://arxiv.org/pdf/2611.03771v1)

**作者:** Seyed Bagher Hashemi Natanzi `[一作]` (Worcester Polytechnic Institute), Bo Tang `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

本文研究了毫米波网络中基于集成感知与通信（ISAC）的主动切换，探讨了预见性和定位要求的价值。

**💡 创新点**

创新点在于提出了一种基于未来信噪比知识的成本感知小区选择模型，并开发了一个ISAC辅助的切换框架，能够在用户和障碍物之间进行精确定位。

**🔧 技术方法**

使用了动态规划技术来解决小区选择问题，并通过模拟的Open RAN xApp进行切换规划。

**📊 数据集**

使用了28GHz街道峡谷的光线追踪实验数据，进行了多种场景的验证。

**📈 对比分析**

与传统的反应式切换方法（如A3和A5）进行比较，ISAC辅助的切换在短期预见下能够显著减少用户的中断时间，但在实际应用中仍需精确的用户和障碍物定位。

**⚠️ 局限性**

限制在于需要将用户和附近障碍物定位到约十厘米的精度，而现有的雷达跟踪和米级用户定位无法满足这一要求。

---

## 33. Language Models Can Benefit from Domain-Specific Inductive Biases

**arXiv ID:** 2611.03756 | [PDF](https://arxiv.org/pdf/2611.03756v1)

**作者:** Damien Teney `[一作]` (Idiap Research Institute), Simon Lucey `[通讯]` (Adelaide University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了语言模型在不同预训练数据子集上的表现，提出通过替换标准激活函数为针对特定数据集调优的学习样条来改进模型的归纳偏置，并评估了跨领域迁移的效果。

**💡 创新点**

创新点在于引入了针对特定领域的学习样条替代标准激活函数，从而提高了模型在不同数据集上的训练效率和泛化能力。

**🔧 技术方法**

使用了变体的变换器架构，通过学习样条替换标准激活函数来优化模型。

**📊 数据集**

使用了多个数据集，包括自然语言（FineWeb、Enwik8、TinyStories、Shakespeare）、代码（CodeSearchNet-Java和Python）和数学（DeepMind-Math）。

**📈 对比分析**

与标准变换器相比，调优后的变换器在所有测试的数据集上提高了训练效率或最终性能，但这些提升通常是特定于领域的。调优后的架构在相关数据集之间迁移良好，但在自然语言和数学等不同领域之间迁移效果较差。

**⚠️ 局限性**

限制在于研究主要集中在小模型（40-120M参数）上，尚不清楚在更大模型上是否能保持相同的性能提升。此外，评估主要集中在困惑度而非下游性能上。

---

## 34. A Compact Formula-Aware Spreadsheet Abstraction for Large Language Models

**arXiv ID:** 2611.03705 | [PDF](https://arxiv.org/pdf/2611.03705v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 35. Honest Small Decision Models for Finance: Scale, Specialization Collapse, and Mixed-Corpus Remediation

**arXiv ID:** 2611.03755 | [PDF](https://arxiv.org/pdf/2611.03755v1)

**作者:** Ajitabh Kumar `[一作]` `[通讯]` (IntelCast), Ajitabh Kumar (IntelCast)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过在两大语言模型家族（Qwen和K2-Horizon）上统一冻结LoRA细调方案，系统评估了在金融领域（政策读取与未来收益预测）下决策模型的概率质量、泛化与细调后专化损失，并提出混合语料一体化的修复方法。

**💡 创新点**

创新点包括：①首次在金融决策上使用公开决策基准与自制“政策”与“现实”语料，系统揭示了细调导致的专化损失与规模/语料量相关性；②提出单一混合适配器在保持规则精准、保持市场预测概率诚实、同时保留公共基准性能的三域兼顾方案；③通过阈值分段（band dispatch）量化概率到业务行动的落地，可测量错误界限并验证决策门控。

**🔧 技术方法**

技术上使用了：冻结的LoRA（r=16, α=32, dropout 0.05）微调；温度校准（针对不同领域的温度分别设定）；多重评价指标（Brier、NLL、ECE、AURC、覆盖率）；集群稳健配对检验、外部测试面板、泄漏检查；以及基准与自制数据集的拼接与交叉验证。

**📊 数据集**

数据集包括：1）公开决策基准（kev，14,576行）用于通用性能评估；2）程序化黄金政策语料（2,651行，含对照对）用于规则读取；3）现实标注结果语料（19,640行，19,640+1,360问答）基于2016-2025年美国大型股日线技术指标和回报；4）封闭2026年测试集（1,360行）未公开。

**📈 对比分析**

比较方法：在每个模型阶梯上对三类任务（policy、outcome、general）分别计算准确率、ECE等指标；通过两轮（epoch1/epoch2）与不同温度下的对照；对比单域细调与混合语料模型；使用集群稳健检验评估差异显著性。性能表现为：政策任务准确率≥99%，回报任务准确率≈37%（接近先验基线，表明无额外方向性信息），混合模型在保持政策精准的同时，保持近乎专家级的回报概率校准且在公开基准上提升约0.6–0.9个百分点。

**⚠️ 局限性**

局限性包括：①仅在单一股权宇宙与技术指标空间评估，未验证跨资产或宏观特征；②细调结果受随机种子与单跑影响，尽管进行三次复现；③未公开2026年测试集，无法验证长周期泛化；④阈值分段未覆盖“中立”决策通道；⑤存在模型与基准规模不匹配的边界效应，导致跨家族比较受限。

---

## 36. CaRe-KT: Continuity-Aware Retrieval-Augmented Reasoning for Knowledge Tracing

**arXiv ID:** 2611.03744 | [PDF](https://arxiv.org/pdf/2611.03744v1)

**作者:** Mehrnoush Mohammadi `[一作]` (University of Queensland), Hassan Khosravi `[通讯]` (University of Queensland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了 CaRe-KT，一种连续性感知的检索增强推理框架，用于知识追踪，解决检索-推理不对齐问题。

**💡 创新点**

引入证据图缓存实现查询适应的时效性检索，并通过保持检索相关性贯穿推理，形成“相关性连续性”原则。

**🔧 技术方法**

采用知识图缓存、时间衰减相关性、共享 GRU 编码器、对比学习和一致性正则化、基于注意力的相关性信息融合等技术。

**📊 数据集**

在五个公开知识追踪基准（DBE-KT22、ASSISTment09、XES3G5M、Slepemapy.cz、Junyi）上进行评估。

**📈 对比分析**

与13个基准（序列、注意力、图、协同检索模型）进行在线对比，CaRe-KT 在AUC、AP和RMSE上始终排名第一，尤其在冷启动和稀疏历史条件下提升显著。

**⚠️ 局限性**

仍依赖问题内容编码，极小题库或数据延迟环境下证据可用性受限，实验仅为离线预测，缺乏实际教学效果验证。

---

## 37. DualClusterNet: Personalized Pattern-Space Composition for Next-POI Recommendation

**arXiv ID:** 2611.03726 | [PDF](https://arxiv.org/pdf/2611.03726v1)

**作者:** Ziyang Su `[一作]` (Shanghai Jiao Tong University), Linghe Kong `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 DualClusterNet，针对下一站点推荐任务，利用可学习的模式和空间词典与用户个性化分支进行组合，生成对当前轨迹前缀的上下文表示。

**💡 创新点**

创新点在于：① 两维软词典的因子化（模式×空间）组合；② 使用 FiLM 进行用户级特征调制；③ 通过可学习门控在加乘与加法分支间动态融合。

**🔧 技术方法**

技术手段包括：因果 LSTM 编码器、软聚类词典、特征级 affine 调制、加乘门控融合、全候选评分与一阶转移嵌入。

**📊 数据集**

实验使用了纽约市（NYC）和东京（TKY）的公开签到数据集，采用训练集生成的完整候选词典进行评估。

**📈 对比分析**

在统一闭集协议下与八种基线进行比较，DualClusterNet 在 NYC 的 Acc@1 与 MRR 轻微领先 DCHL，在 TKY 也获得最高 Acc@1/MRR，整体性能与竞争对手相近。

**⚠️ 局限性**

局限性包括：仅评估已知用户/地点，未涉及冷启动或未见地点泛化；词典的语义可解释性未探究；实验仅用单一随机种子，结果的稳定性不明。

---

## 38. Training LLM Agents to Automate LLM Post-Training with RSI-Dojo

**arXiv ID:** 2611.03698 | [PDF](https://arxiv.org/pdf/2611.03698v1)

**作者:** Shuzheng Gao `[一作]` (Chinese University of Hong Kong), Michael R. Lyu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了RSI-Dojo平台，自动化将科研论文中的数据与评测任务转化为可执行的模型训练环境，收集并筛选高质量训练轨迹，并用这些轨迹对Qwen3.5-9B和Qwen3.6-35B-A3B进行微调，提升其自动化后训练能力。

**💡 创新点**

创新点在于：①提出可规模化的自动化构建可执行训练环境的流水线；②设计改进的收集机制和评判过滤策略，以获得高质量的训练轨迹；③通过轨迹微调将后训练能力内化到模型本身，实现了基模型的自主改进。

**🔧 技术方法**

技术手段包括：基于论文检索与关键词扫描的任务抽取；环境实例化、评测复现与可靠性/可区分性过滤；改进的收集 harness 与定向指导；轨迹行为与结果双重过滤；以及在Qwen系列模型上进行的微调训练。

**📊 数据集**

使用的数据集来源于近三年NeurIPS/ICML/ICLR/ACL/EMNLP/NAACL/COLM等顶会论文中的公开数据集以及InspectAI平台的任务，共构成281个训练环境，评估时采用PostTrainBench和PostTrainBench‑Lite的公开基准数据。

**📈 对比分析**

通过与GPT‑5.5、GLM‑5.2、DeepSeek‑V4.1‑Flash等前沿模型在PostTrainBench v1.1和Lite版的基准对比，RSI‑Dojo‑35B‑A3B在PostTrainBench获得29.95分，匹敌甚至略优于前沿模型；在Lite版上达到53.29分，显著提升基模型的原始得分（+16.87分）。

**⚠️ 局限性**

局限性包括：环境仅覆盖PostTrainBench式任务，缺乏更广泛的长期研究式后训练场景；未实现模型自身的递归自我改进闭环；收集与评测中工具配置受限，可能影响更复杂任务的支持。

---

## 39. Learning Directional Randers Geometry for Latent-Space Exploration

**arXiv ID:** 2611.03675 | [PDF](https://arxiv.org/pdf/2611.03675v1)

**作者:** Théau Blanchard `[一作]` (UMR 1346 HeKA), Stéphanie Allassonnière `[通讯]` (UPCite-INRIA-INSERM)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出通过学习Finsler‑Randers度量在VAE潜在空间中引入方向性，完成对疾病进展或植物生长等单调序列的隐式插值与外推；

**💡 创新点**

创新点在于将Finsler几何与Randers度量结合，以向量场ω学习局部可分离超平面，从而在保持局部几何的同时注入全局方向性约束；

**🔧 技术方法**

技术上采用VAE潜在空间、基于KNN的局部对齐损失学习ω，再结合GEORCE/贝塞尔曲线求解隐式几何的最短路径，最后用Hamilton动力学进行外推；

**📊 数据集**

使用人工生成的Shepp‑Logan相位图（模拟肿瘤面积）与KOMATSUNA植物叶面积两组跨截面数据；

**📈 对比分析**

与欧氏、LAND、LAND‑RBF、CovCentroids、PBCentroids以及Wasserstein、HyperSpherical等基线比较，插值性能基本相当，但在外推任务中Finsler‑Randers模型在Kendall τ与相对变化率上均显著优于对称框架，显示出更高的单调性和探索性；

**⚠️ 局限性**

主要局限在于对标量值的全序假设、对小样本数据的敏感性以及对大规模数据的计算可扩展性问题。

---

## 40. MCPGuard: Finding Conformance Gaps in MCP SDKs via Dual-Track Analysis and Dynamic Adjudication

**arXiv ID:** 2611.03662 | [PDF](https://arxiv.org/pdf/2611.03662v1)

**作者:** Xikai Sun `[一作]` (Tsinghua University), Yunhao Liu `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 MCPGuard，一种基于双轨（结构与行为）自动化框架，用于检测 MCP SDK 与规范之间的合规性差距。

**💡 创新点**

创新点在于：①利用结构轨道将 JSON Schema 约束与 SDK 验证器进行确定性比对；②通过 LLM 辅助的函数级切片识别行为约束缺失；③引入证据驱动的动态裁决器验证候选缺口，避免误报。

**🔧 技术方法**

采用静态代码分析、LLM（GPT‑5.5）自然语言规则抽取、函数切片、证据检索与 PoC 生成技术，并实现语言无关内核与 Python/Typscript SDK 的适配器。

**📊 数据集**

使用的实验数据集包括 MCP 2025‑11‑25 规范、Python 1.26.0 SDK、TypeScript 1.29.0 SDK 以及 32 台公开 MCP 服务器。

**📈 对比分析**

与官方 MCP 兼容性测试框架与 Yaw Labs Suite 进行对比；MCPGuard 在两套 SDK 上分别发现 13 条和 19 条合规性缺口，且均未被基线工具检测到，说明其在 SDK 层面具有更高的检出率。

**⚠️ 局限性**

局限性包括：行为轨道覆盖率仅约 70% 的规范规则；LLM 可能产生误判，需人工复核；仅针对两套 SDK，未覆盖其他语言实现；以及仅识别规范偏差，无法定位更深层次实现缺陷。

---

## 41. Mixture of Wide Experts: Sparse Routing for Interpretable Native Coordinates

**arXiv ID:** 2611.03655 | [PDF](https://arxiv.org/pdf/2611.03655v1)

**作者:** Tavish Mankash `[一作]` (Plaksha University), Deepan Muthirayan `[通讯]` (Plaksha University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 Mixture of Wide Experts (MoWE) 结构，将传统稀疏路由专家的输出限制为不重叠的固定坐标，从而在模型本身的 FFN 更新中直接产生可解释且可复用的特征，并通过控制任务、语言模型训练和案例研究验证其可解释性与可控性。

**💡 创新点**

核心创新是：① 在每层只允许每个激活的专家写入自己专属、离散的输出坐标；② 通过“读全写离散”的规则，使得路由和写入解耦，提升了坐标的单义性；③ 将可解释性评估与传统稀疏自编码器、随机旋转等基线对齐，形成了针对原生坐标的可解释性度量。

**🔧 技术方法**

技术手段包括：稀疏路由（Top‑k MoE）与固定输出切片；SwiGLU 作为专家网络；Top‑k 稀疏自编码器用于后处理比较；语言模型训练采用 10B 词元的科学+源码混合数据；可解释性评估使用自动化的“top‑context purity”与“specificity”度量；案例研究涉及因果追踪、坐标迁移和基于坐标的安全请求检测与拒绝控制。

**📊 数据集**

数据集与任务：① 受控任务——32 个隐藏原因生成的噪声位向量，供验证因果定位；② 预训练语料——约 10 亿词元，混合科学与源码文本；③ 安全案例——使用匹配模板生成 160 条有害/无害请求用于训练检测器，另外 200 条用于测试。

**📈 对比分析**

比较方法与性能：① 通过匹配活跃 MACs 与参数量，比较 MoWE、Dense 与标准 MoE 的交叉熵，结果为 MoWE 2.417 nats、Dense 2.452 nats、标准 MoE 2.22 nats；② 可解释性上，MoWE 的特异性得分为 0.523，高于 Dense 0.375 与标准 MoE 0.268；③ 在案例研究中，MoWE 的坐标追踪、坐标迁移与安全请求拒绝均表现优异，尤其在安全干预上达到 99% 成功率，而 Dense 在此任务上几乎无效。标准 MoE 在语言建模性能上最强，但可解释性与可控性最低。

**⚠️ 局限性**

局限性：① 仅评估两种种子与单一规模模型；② 仅在单词级别进行干预，未验证更大规模或完整生成控制；③ 安全请求检测仅基于匹配模板，缺乏对自然对话的泛化；④ 由于稀疏路由的训练挑战，可能导致专家使用不均衡；⑤ 可解释性度量仍依赖自动化标签，缺乏广泛人类评估。

---

## 42. CARD: A Proof-Carrying Certificate-to-Action Interface for Relative-Measurement Graphs

**arXiv ID:** 2611.03651 | [PDF](https://arxiv.org/pdf/2611.03651v1)

**作者:** Yushan Li `[一作]` `[通讯]`, Yushan Li

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种证明携带接口，将几何证书与可执行机器人动作分离，并记录它们的组合方式。

**💡 创新点**

创新点在于引入升纤维（lifted‑fiber）分解，区分语义动作与证明证据，推导出预证明可操作性定理（hypergeometric 服务收益预测），以及基于证书的可执行服务流水线。

**🔧 技术方法**

使用了几何匹配、Exact‑16 证书优化、因果导航、稀疏求解（AMD、Cholesky）、图优化、随机抽样与超几何模型。

**📊 数据集**

数据集包括20次 AA15–X500 无人机全景航拍的 404 个 Full16 案例，以及 KITTI、EuRoC、TUM SLAM 公开路标进行下游图优化评估。

**📈 对比分析**

通过对比所有检查确认的优化点与因果导航变换，验证证书与执行输出差异；用预证明筛选与随机抽样的预期收益对照，证明预证明匹配随机达成每请求 1 个有用服务；在下游 SLAM 中匹配图拓扑后比较优化器时间，发现随机拓扑导致优化器时间提升 1.128×至 67.65×，主要由 AMD 填充和 Cholesky 因子大小差异导致。

**⚠️ 局限性**

局限性包括：证书与执行的同步延迟、证明失效时间窗口、仅在特定无人机和 SLAM 场景下验证，缺乏对更大规模或不同传感器配置的泛化；随机采样理论假设均匀分布，实际可能偏差。

---

## 43. Beyond Proposals: Region-Consistent Open-Set Object Detection

**arXiv ID:** 2611.03741 | [PDF](https://arxiv.org/pdf/2611.03741v1)

**作者:** Jiehui Tang `[一作]` (Zhejiang University), Jungong Han `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种面向开放集目标检测的框架 OCSI-Net，通过将密集候选框聚合为对象级区域集，进行区域一致的语义推理并在训练时引入几何感知的不确定性稳定化。

**💡 创新点**

创新点包括：①将开放集推理从单框独立决策转化为对象级区域语义一致推理；②设计了对象级语义一致模块 OCSC，聚合重叠框生成区域候选并通过聚合语义证据得到统一的区域语义；③提出几何感知边界不确定性机制 GABU，在未知概率学习中对几何边界附近的框进行加权，以提升开放集边界的鲁棒性。

**🔧 技术方法**

核心技术包括：两阶段检测器基础（如 OpenDet/Faster R‑CNN），基于 DINOv3（或 CLIP、DeiT、ResNet）提取的区域特征，区域聚类与连通组件分析，区域级交叉熵监督，IoU‑基准的不确定性重加权，整体联合损失优化。

**📊 数据集**

在 OpenImages（Animal、Vehicle）、CUB‑200‑2011 和 MTSD 三大公开基准上进行实验，采用官方的 OSOD‑III 分割方案评估已知与未知检测性能。

**📈 对比分析**

与 ORE、DS、VOS、Orth.Det、FCOS、Faster R‑CNN、OpenDet 等最新开放集检测方法对比，OCSI‑Net 在所有四个基准上均取得最高的调和平均数（H），平均提升 6.83%（已知/未知平衡）并显著提高未知检测精度。

**⚠️ 局限性**

局限性包括：①需要额外的区域编码器（如 DINOv3）增加推理成本；②区域级语义决策在聚类混杂或定位不佳时可能失效，导致未知检测错误；③对预训练视觉模型的依赖可能引入未知类别的隐式信息，影响严格的开放集假设。

---

## 44. HyVox: Rethinking Low-Power Voice Intelligence towards Neuromorphic-von Neumann Edge Computing

**arXiv ID:** 2611.03735 | [PDF](https://arxiv.org/pdf/2611.03735v1)

**作者:** Snehasis Addy `[一作]` (University of Massachusetts Amherst), Phuc Nguyen `[通讯]` (University of Massachusetts Amherst)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种将嵌入过程交给可编程神经形态芯片、主机完成开放集识别和拒绝决策的混合架构；

**💡 创新点**

创新点在于：1）将工作拆分到嵌入阶段，使神经形态芯片仅做稀疏事件驱动计算；2）设计层次时序抽象（HTA）SNN，在有限的神经元与突触预算下学习多时钟尺度；3）使用CAST（芯片精确训练）消除仿真与硬件差距；4）引入32维Spiking Embedding Extractor（SEE）实现无权重更新的注册与拒绝；

**🔧 技术方法**

技术：层次时序抽象LIF SNN、稀疏压缩与量化、整数位移量化感知训练、CAST（芯片精确训练）以及基于SEE的少样本开放集学习；

**📊 数据集**

数据集：VoxCeleb2用于HTA‑SNN预训练，VoxCeleb1用于SEE预训练，Common Voice 20个不同口音的家庭数据用于最终评估；

**📈 对比分析**

与四个基准ANN模型（Small‑footprint SV、DS‑CNN‑S、BC‑ResNet‑1、MLP‑d128）比较，获得99.6% top‑1、5.5%拒绝EER（2人）到94.9%/13.2%（20人），动态能耗比最优基准低13.5×，延迟比最快基准低42%；

**⚠️ 局限性**

限制：拒绝率随用户数增加而上升（20人时13.2%），嵌入维度仅32维、只能接受预训练embedding，硬件依赖特定神经形态芯片，且对声学环境变化的鲁棒性仍有限。

---

## 45. Stress-Testing the Robustness of LLM-Based Causal Inference

**arXiv ID:** 2611.03656 | [PDF](https://arxiv.org/pdf/2611.03656v1)

**作者:** Ankitkumar Patel `[一作]` (IntelliQuest Venture), Venkatesh Kulkarni `[通讯]` (Indian Institute of Technology Guwahati)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对大型语言模型在因果发现任务中的性能进行系统性压力测试，评估模型规模、图复杂度、提示策略、元数据质量、数据质量以及不确定性敏感性。

**💡 创新点**

提出了六维敏感性分析框架和基于不确定性的三元评估指标 F_adj，揭示了 LLM 的乐观偏差和不确定性处理对因果推断的影响。

**🔧 技术方法**

使用大规模 LLM（如 GPT‑4o、GPT‑5.2 等）配合零样本提示，构建不确定性‑aware 的精确度、召回率、F1 调整指标，并对不同提示与元数据进行实验。

**📊 数据集**

以经典的八节点医学诊断网络 Asia 为基准，并在不同元数据丰富程度（L1–L3）下进行实验。

**📈 对比分析**

在统一的 API 环境下对多种 LLM 进行零样本评估，利用 F_adj、P_adj、R_adj 等指标比较，结果显示 GPT‑5.2 取得近 1 的 F_adj，GPT‑4o 等模型在不确定性惩罚下表现下降，体现乐观偏差；小模型在元数据丰富时性能可与大模型相当。

**⚠️ 局限性**

实验受限于单一小规模网络、零样本提示、缺乏多样化的图结构和真实观测数据，且高性能模型成本高昂，需进一步验证在更大规模和更复杂场景下的适用性。

---

## 46. What Malware Actually Does: An Empirical Analysis of Post-Infection Behavior Across Malware Families

**arXiv ID:** 2611.03774 | [PDF](https://arxiv.org/pdf/2611.03774v1)

**作者:** Emmanuela Andam `[一作]` (University of North Dakota), Naima Kaabouch `[通讯]` (University of North Dakota)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本论文通过动态沙箱执行分析了37个恶意软件家族的感染后行为，填补了恶意软件检测研究中对感染后行为关注不足的空白。

**💡 创新点**

创新点在于提出了一种异构图构建管道，能够建模多种实体类型和关系类型的感染后行为，并结合LLM生成可读的分析报告。

**🔧 技术方法**

使用了动态分析技术，基于CAPE沙箱执行恶意软件样本，并通过异构图神经网络进行分类和行为分析。

**📊 数据集**

数据集包含8,047个样本，涵盖37个恶意软件家族和来自四个来源的良性可执行文件。

**📈 对比分析**

与平面特征基线相比，基于图的结构特征的投票集成分类器在MCC上提高了7.4个百分点，达到了94.44%的准确率和0.863的MCC。

**⚠️ 局限性**

限制在于分类依赖于聚合结构统计而非语义节点特征，导致对具有相似拓扑的样本的家族级别区分能力有限。此外，数据集中APT样本和非Windows恶意软件的代表性不足。

---

## 47. VagueBench: Benchmarking the Robustness of Embodied Agents on Ambient Context Illusions

**arXiv ID:** 2611.03765 | [PDF](https://arxiv.org/pdf/2611.03765v1)

**作者:** Yang Qu `[一作]` (Beihang University), Xianglong Liu `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `51c0528b-f690-4182-ae60-bb5f046c276c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了具身智能体在缺乏辅助上下文信息时的鲁棒性，提出了一个新的现象称为环境上下文幻觉（Ambient Context Illusion），并建立了一个基准测试来评估具身智能体在上下文依赖性方面的表现。

**💡 创新点**

创新点在于首次定义了环境上下文幻觉，揭示了具身智能体在语义有效但上下文最小的指令下的表现下降，并提出了一个基于AI2-THOR的基准测试来系统性地研究这一现象。

**🔧 技术方法**

使用了AI2-THOR交互仿真环境，结合语义冗余估计、分类引导转换和严格的语义不变性验证等技术。

**📊 数据集**

构建了包含1303对指令的基准数据集，涵盖多种室内环境，通过人机协同生成框架进行构建。

**📈 对比分析**

与六种最先进的具身智能体进行比较，结果显示在上下文减少的指令下，任务成功率平均下降了19.65%，空计划率增加了54.10%，表明智能体对辅助语言信号的强烈依赖。

**⚠️ 局限性**

限制在于当前研究主要集中在特定的室内环境和任务上，未来需要扩展到更复杂的真实世界机器人系统和多样化的环境中，以提高泛化能力。

---

## 48. JevBreak: Your Model, Their Decision

**arXiv ID:** 2611.03759 | [PDF](https://arxiv.org/pdf/2611.03759v1)

**作者:** Nitin Aravind Birur `[一作]` (Enkrypt AI, Anaconda), Prashanth Harshangi `[通讯]` (Enkrypt AI, Anaconda)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对七种决策模型进行目标可控性评估，展示通过仅在证据字段追加文本即可强制模型给出预设错误答案。

**💡 创新点**

提出全局可达性（full control）度量和基于概率反馈的Slot Attack搜索，揭示接口包装器对攻击的可变影响。

**🔧 技术方法**

采用结构化黑盒搜索、一次性固定模板攻击、概率向量反馈、决策API与聊天API对比等技术。

**📊 数据集**

使用Decision Bench的240条真实记录，覆盖16个任务、7个领域，构成多样化评估数据集。

**📈 对比分析**

对每个模型测量基准准确率、一次性模板成功率、Slot Attack成功率和全控制率，攻击成功率从30%提升到99%，模型差异明显。

**⚠️ 局限性**

局限在于样本量有限、仅测试追加文本攻击、未评估通用后缀或证据重写、概率四舍五入与查询上限可能低估鲁棒性。

---

## 49. LoRA-Patch: Defending Against Backdoored LoRA Adapters via Directional Steering

**arXiv ID:** 2611.03752 | [PDF](https://arxiv.org/pdf/2611.03752v1)

**作者:** Shengyan Huo `[一作]` (Xidian University), Yige Li `[通讯]` (Singapore Management University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对第三方LoRA适配器的轻量级后门防御方法LoRA-Patch，旨在通过层级方向引导来抑制后门行为，而不修改基础模型的权重。

**💡 创新点**

创新点在于利用后门注入扭曲的方向几何特征作为无触发器的信号，通过层级引导来重新对齐适配器的行为，从而有效抑制触发激活的恶意行为。

**🔧 技术方法**

使用了层级方向引导技术，进行拒绝方向探测和几何重新对齐。

**📊 数据集**

在三个开放权重模型（Llama-3-8B-Instruct、Llama-3.1-8B-Instruct和Mistral-7B-Instruct）上进行了评估，主要使用GSM8K作为下游任务数据集，AdvBench作为有害探测数据集。

**📈 对比分析**

与现有的防御方法（如Fine-Tuning、Fine-Pruning和CROW）相比，LoRA-Patch在五种后门攻击下表现出最低的攻击成功率（ASR），同时保持了良好的任务准确性（ACC），在所有评估模型中，补丁构建时间在20秒内完成，且计算效率高。

**⚠️ 局限性**

限制在于LoRA-Patch设计用于基础模型可信且可本地访问的适配器部署场景，未考虑基础模型中已嵌入的后门或内部模型激活不可访问的情况。其性能可能因探测覆盖率和模型特定表示几何而有所不同，且当前实现采用固定的后层引导窗口和引导强度，未来可开发自动化的适配器特定校准以提高安全性和鲁棒性。

---

## 50. EvoMemState: Evolving Memory State for Personal Agents

**arXiv ID:** 2611.03743 | [PDF](https://arxiv.org/pdf/2611.03743v1)

**作者:** Zhishang Xiang `[一作]` (Xiamen University), Jinsong Su `[通讯]` (Xiamen University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出EvoMemState框架，将个人记忆转化为可演化的关系状态和请求特定的推理状态，实现长期记忆的状态化管理。

**💡 创新点**

创新点在于显式捕获事实的更新、冲突与适用性，通过关系状态和推理状态两层表征，使记忆检索与使用更加连贯、精准。

**🔧 技术方法**

采用知识图谱驱动的记忆状态诱导、基于强化学习的状态演化策略以及状态感知推理三种技术。

**📊 数据集**

使用MemSyco-Bench、MemTrapBench、LoCoMo 与 LoCoMo-Refined 等长记忆与对话回顾基准数据集。

**📈 对比分析**

与九个基线相比，在MemSyco-Bench、MemTrapBench、LoCoMo-Refined 上均提升 5–15% 的准确率，显著提高检索完整性和记忆基准答案质量。

**⚠️ 局限性**

仅处理文本交互历史，未覆盖多模态信息，未来需要扩展到多模态记忆管理。

---

## 51. Coordinate Frame Recovery, Per Class NMS and Null Return Modelling for Sim2Real V2X LiDAR Detection

**arXiv ID:** 2611.03940 | [PDF](https://arxiv.org/pdf/2611.03940v1)

**作者:** Aman Upganlawar `[一作]` `[通讯]` (Columbia University), Aman Upganlawar (Columbia University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在UCF UrbanTwin Sim2Real LiDAR挑战V2X-Real轨道上，作者通过诊断坐标系不匹配、针对行人使用类特定NMS、引入空返回点等方法，最终实现了检测与真实性评分的显著提升，综合得分达到0.7457。

**💡 创新点**

创新点主要在于：①发现并纠正坐标系不匹配导致检测全失误的根源；②针对行人群集特点引入类特定NMS阈值；③通过在合成云中注入与真实数据匹配的空返回点，显著提升真实性评估分数。

**🔧 技术方法**

技术手段包括：PointPillars 3D检测器、OpenPCDet框架、坐标变换逆运算、几何重新中心化、匹配配对校准、基于点密度的重新排序、测试时增强（视角翻转）以及Frechet点云距离的轴向仿射校正。

**📊 数据集**

所用数据集为UCF UrbanTwin Sim2Real LiDAR双轨数据集，包含合成的数字双子序列以及公开的真实Ouster OS1-64路侧LiDAR帧，所有训练仅使用合成数据。

**📈 对比分析**

评估方式为在50个隐藏真实帧与50个合成帧上提交检测结果与合成点云，分别计算3D mAP(0.4382)与真实性得分(0.9251)，综合分为0.7457，显示在同类竞赛中表现领先。

**⚠️ 局限性**

局限性包括：行人AP仍低于预期、真实性分已接近上限、主要改进来自后处理而非模拟器本身、单一参与者限制了训练多样性与实验范围。

---

## 52. FlowBench: Counterfactual Imagination Training for Spatial Intelligence

**arXiv ID:** 2611.03937 | [PDF](https://arxiv.org/pdf/2611.03937v1)

**作者:** Yifan Liu `[一作]` (Stanford University), Daniel L. K. Yamins `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了对假设运动下场景变化的“Counterfactual Spatial Imagination (CSI)”能力，构建了大规模自动化数据集FlowBench，并在该数据集上评估并微调视觉语言模型与世界模型。

**💡 创新点**

创新点包括：①将CSI形式化为可评估的任务；②设计完整的自动化流水线，生成428K训练样本，包含精确的后置图像、光流、深度和语义标注；③展示了在仿真训练上对真实场景的强转移，并为VLM与世界模型的联合训练提供了通用资源。

**🔧 技术方法**

采用的技术包括：视觉语言模型（如Qwen3‑VL、Claude‑Fable、GPT‑Astra等）和世界模型（Ψ‑0.6、FOFPred、Qwen‑Image‑Edit、InstructPix2Pix等），以及文本到光流预测、文本到图像生成、光流估计、SAM/Depth Anything等辅助工具。

**📊 数据集**

使用的数据集为FlowBench，其中FlowBench‑Sim（来自AI2‑THOR/ProcTHOR的10K虚拟房间，420K训练问题）与FlowBench‑Real（来自Bridge V2、DROID、Language‑Table、RT‑1、Berkeley UR5等机器人录制的桌面操作，8.6K训练问题），并提供测试拆分。

**📈 对比分析**

对比方法：在“1A（需想象）”与“2A（直接观察）”两模式下评估VLM准确率；与人类基线和多款专有/开源模型比较；对世界模型进行文本到光流/图像生成精度评估。结果显示：专有模型在Real上超过人类，Sim上仍低；开源模型靠近随机；对Qwen3‑VL微调后Sim准确率提升约30%，Real提升约17%，显示显著的sim‑to‑real迁移；世界模型在光流预测上提升至61.7%，但对答案的帮助有限。

**⚠️ 局限性**

局限性：仅考虑单一刚体平移动作，未覆盖旋转、变形或多步交互；世界模型生成的中间图像细节不足，导致无法完全支持答案；Real测试依赖估计的深度/遮挡信息，存在误差；评价仅在两种模式下进行，未覆盖更复杂的视觉推理场景。

---

## 53. What Is This Memory Useful For? Measuring Task-Conditioned Utility in Long-Term Agent Memory

**arXiv ID:** 2611.03930 | [PDF](https://arxiv.org/pdf/2611.03930v1)

**作者:** Chih-Hsuan Yang `[一作]` (Argonne National Laboratory), Rajeev Thakur `[通讯]` (Argonne National Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估任务条件下的记忆效用并提出完整的测量协议

**💡 创新点**

引入任务盲写、相同填充、成本账本与成对对照，直接量化压缩记忆对具体任务的贡献

**🔧 技术方法**

基于对照实验设计、写入器任务盲写、填充保持、成本记录与LLM读者评估

**📊 数据集**

LongMemEval‑V2网页代理页面访问记忆数据集，附带SciCode与SWE‑Bench的无效验证

**📈 对比分析**

Raw记忆在答案相关任务上的平均奖励增益显著高于Summary/Structured；检索Recall@10约0.4，显著高于随机基线

**⚠️ 局限性**

仅适用于单调用事实回忆，受限于单一模型/单读者、无多任务交互、检索完整性未完全验证

---

## 54. Sequential auditing of connected forecast errors under partial observations

**arXiv ID:** 2611.03928 | [PDF](https://arxiv.org/pdf/2611.03928v1)

**作者:** Yuanbo Zhou `[一作]`, Junjie Xiong `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究在图时间序列预测区间下，审计者如何通过选择性检索隐藏标签来决定是否存在一连通子图全局区间违背事件，并提出精确停止判定和多种查询策略。

**💡 创新点**

创新点在于引入精确停止判定的审计框架，改进MOMS和JW等残差集计数策略，并提出最短见证/最短池策略，揭示不同策略在查询效率、早期发现与计算复杂度上的权衡。

**🔧 技术方法**

采用组合与图论算法（连通分量、子图枚举）、贝叶斯主动学习/子模函数思想、残差集计数、Python整数位集实现、随机/屏蔽采样以及统计置信区间估计等技术。

**📊 数据集**

使用了METR‑LA、WikiMaths和MontevideoBus三公开图时间序列数据集进行实验。

**📈 对比分析**

通过在10%/30%遮蔽率下对9,696案例进行多策略评估，MOMS相对屏蔽随机减少19.6%–38.5%查询；最短见证在局部遮蔽下在10查询内发现最多正例，但整体查询次数略高；不同策略在早期发现与计算速度上呈现折中。

**⚠️ 局限性**

局限性包括仅考虑精确标签检索，忽略检索成本不均、并行/异步请求以及测量错误；正例稀疏且实验数据重复性有限，限制了对新数据的推广性。

---

## 55. Handling JAX

**arXiv ID:** 2611.03918 | [PDF](https://arxiv.org/pdf/2611.03918v1)

**作者:** Oghenevwogaga Ebresafe `[一作]` (University of Toronto), Ningning Xie `[通讯]` (Google DeepMind)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在受限的第一阶目标（如 JAX 的 jaxpr）上实现代数效果处理器，并提供一种可证明的编译技术；

**💡 创新点**

通过将效果处理器静态消除，结合 CPS 与静态归约，支持在无递归、无高阶函数的目标上编译效果处理器，并在循环边界上实现尾递归恢复与中止式处理器；

**🔧 技术方法**

使用代数效果处理器、CPS 变换、静态归约、恢复内联、以及 Lean 4 机制化证明；实现 JAX 库以支持效果处理器；

**📊 数据集**

在数值 ODE 求解器（反应‑扩散 PDE、神经 ODE、跳步 SDE、Lotka‑Volterra）和概率编程模型（比例回归、随机波动、贝叶斯神经网络、半监督 HMM、层级 GLMM、Hilbert‑空间 GP）上评估；

**📈 对比分析**

与手写基准、NumPyro 与 Oryx 进行比较；运行时开销低于 5%（大多相同），编译时间相当或略低；

**⚠️ 局限性**

仅支持尾递归恢复或中止式处理器跨循环边界，无法处理多次恢复的处理器；空间占用与循环迭代次数相关，且对 XLA 的静态内存约束有限制。

---

## 56. A Conditional Mutual Information Analysis of Response Variety in Language Models

**arXiv ID:** 2611.03836 | [PDF](https://arxiv.org/pdf/2611.03836v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 57. Timely Ground-Station Status Delivery via UAV-Assisted Data Collection and Hierarchical RL

**arXiv ID:** 2611.03884 | [PDF](https://arxiv.org/pdf/2611.03884v1)

**作者:** Xiangjian Gao `[一作]` (University of California Santa Cruz), Hamid R. Sadjadpour `[通讯]` (University of California Santa Cruz)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究无人机辅助状态更新系统，优化无人机运动、采集与转发决策，以在有限时隙内降低地面站信息时延（AoI）并控制能耗。

**💡 创新点**

创新点在于提出分层AoI–能量演员-批评家（HAE‑AC）控制器，将高层运动决策与低层通信调度分离，并采用确定性调度器实现多传感器下行 burst，显著提升收集-转发效率。

**🔧 技术方法**

使用了强化学习（PPO）、演员-批评家架构、两层时间尺度决策、最小功率分配与可行性筛选技术，以及对终端位置与能耗约束的返回可行性检查。

**📊 数据集**

实验基于仿真数据：100个均匀分布传感器，场地2000×2000 m，时间窗口1200个时隙，使用固定参数的可靠链路模型。

**📈 对比分析**

与距离贪婪巡航、适应性旧化门控（ASG）以及单层AC进行对比；在FUC策略下，HAE‑AC将平均GS侧AoI从195.1降至151.2（22.5%提升），覆盖率93.6%，Burst数显著下降。

**⚠️ 局限性**

局限在于仅基于仿真验证，未对大规模动态部署或不可靠链路进行测试；奖励设计与超参数对性能影响较大，需进一步稳健性研究。

---

## 58. The Hyperclique Hypothesis is False

**arXiv ID:** 2611.03880 | [PDF](https://arxiv.org/pdf/2611.03880v1)

**作者:** Nick Fischer `[一作]` `[通讯]` (Max Planck Institute for Informatics), Nick Fischer (Max Planck Institute for Informatics)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

该论文提出并实现了一种代数算法，证明了Hyperclique假设（即在h-均匀超图中检测k-团需要时间n^k-ε的假设）为假，并相应地推翻了k-AP假设。通过递归分解和混合计算顺序，该算法在所有常数h≥3下以时间O(n^(1-ε_h)·k)检测k-团，从而大幅加速相关的组合优化与几何问题。

**💡 创新点**

创新点在于：①首次在超图上突破了传统的张量秩和张量网络瓶颈；②提出了“多切片秩”（two-slice rank）视角，允许对同一多项式采用不同的乘法顺序，从而实现了对4-线性超团多项式的加速；③利用AI生成的程序与人类协作完成算法设计与写作，展示了AI在理论计算机科学新突破中的潜力。

**🔧 技术方法**

主要技术包括：递归分解超图变量（按基q分块）；利用矩阵乘法（如Strassen算法）与快速矩阵乘法的时间下界；混合评估顺序（(AB)(CD)与(BC)(DA)）；张量网络和两切片秩分析；以及对递归树潜能的严格潜能函数分析，以证明时间下界。

**📊 数据集**

论文为纯理论工作，无实测数据集。所有实验均为符号运算和复杂度分析，未在真实超图或数据集上实现。

**📈 对比分析**

与传统基于Hyperclique或k-AP假设的下界相比，该算法在理论上实现了显著加速；对于所有常数h≥3，取得了O(n^(1-ε_h)k)的时间复杂度，ε_h为正常数，取决于h，且在递归层数和矩阵乘法常数上有详细分析；虽然在常数上略显粗糙，但证明了此前认为不可压缩的上限可以被突破。

**⚠️ 局限性**

限制包括：ε_h随h呈指数级衰减（双指数下降），导致在高阶h时加速幅度可能有限；实现复杂度高，缺乏可操作的实用实现；算法主要针对整数/0-1超图，未讨论对更一般权重或随机超图的适用性；依赖AI生成代码，重现性和可审计性可能受限。

---

## 59. INSIGHT-HPC: An Interpretable System Intelligence for Global HPC Telemetry

**arXiv ID:** 2611.03875 | [PDF](https://arxiv.org/pdf/2611.03875v1)

**作者:** Rupak Kadel `[一作]` (Texas Tech University), Yong Chen `[通讯]` (Texas Tech University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了 INSIGHT-HPC，一种多层面异常检测架构，能够将多源高维 HPC 监控数据融合成可解释的故障事件并生成操作员可理解的告警摘要。

**💡 创新点**

创新点在于：①将统计偏差、工作负载上下文和物理一致性三类证据层层融合；②构建基于事件的可解释警报模型，能够区分工作负载引起的正常波动与真实的基础设施异常；③在弱标签生产环境中实现了高精度的异常检测。

**🔧 技术方法**

使用了统计异常检测、工作负载感知分析、物理模型约束以及事件级融合算法，并通过基于分钟级别的多源时间序列预处理与去重技术来生成可解释的故障片段。

**📊 数据集**

主要数据集为真实 HPC 集群的监控流（来自 BMC、调度器、功率分配单元等），并结合人工注入的 HPC 相关故障样本进行评估。

**📈 对比分析**

与多种基线检测器（如纯统计、基于机器学习的孤立森林、基于时间序列的异常检测器等）进行对比，INSIGHT-HPC 在精度 0.913、召回 0.817、F1 0.862、AUROC 0.896 上均超过所有基线，显示出更优的精度-召回平衡。

**⚠️ 局限性**

局限性包括：①对工作负载上下文与物理模型的依赖，若模型或上下文信息缺失可能影响性能；②需要针对不同 HPC 体系结构进行模型调优，迁移成本较高；③在极端低标签或高噪声场景下的鲁棒性尚未充分验证。

---

## 60. From Benchmarks to Behaviors: A Framework for Auditing AI Agents

**arXiv ID:** 2611.03939 | [PDF](https://arxiv.org/pdf/2611.03939v1)

**作者:** Nick Hagar `[一作]` `[通讯]`, Nick Hagar

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出TRACE框架，系统地审计GenAI代理的系统组件（如工具、指令和记忆）对其行为的影响，并通过案例研究检验用户记忆如何改变代理在2026年美国中期选举问题上的来源选择。

**💡 创新点**

创新点在于将算法审计方法扩展到代理系统，利用透明、可调控的抓手（Pi harness）实现对单一组件的精准操纵，揭示隐藏的个性化和偏见路径。

**🔧 技术方法**

技术上基于开源Pi抓手，使用声明式实验配置、fixture工具和完整日志记录，结合配对随机化检验与Holm校正评估统计显著性。

**📊 数据集**

数据集由LLM选举观测站提供的固定搜索结果（共50篇）和手工挑选的误导性/阴谋论文章组成，配合5个关于2026年中期选举的政治问题。

**📈 对比分析**

通过在六种模型（GPT与Claude）下进行3,000次试验，将每个记忆条件与无记忆基线对比，发现首三来源中优选出版物出现率提升33–39个百分点，党派平衡右移0.12单位，误导性文章引用率提升6.3个百分点。

**⚠️ 局限性**

局限性包括：对生态有效性的妥协、仅覆盖少数查询和来源、记忆操纵极为简化，且在商业代理系统中的效果可能不同。

---

## 61. Modular and Efficient Approximate Nearest Neighbors Search with Vectorium and kANNolo

**arXiv ID:** 2611.03927 | [PDF](https://arxiv.org/pdf/2611.03927v1)

**作者:** Silvio Martinico `[一作]` (University of Pisa), Rossano Venturini `[通讯]` (University of Pisa)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了两套 Rust 库 Vectorium（数据管理层）和 kANNolo（索引层），实现了一套能够统一处理稠密、稀疏和多向量三种向量表示的近似最近邻检索框架。

**💡 创新点**

主要创新点包括：
- 把向量表示、距离函数、量化器和数据集完全分离为四个独立抽象，任何组合都通过编译时多态实现，避免了运行时的分发开销；
- 在同一框架下提供 11 种编码器（从 32‑bit 浮点到子比特码），支持传统 PQ、二进制量化以及新提出的压缩稀疏量化；
- 构建了可组合的两阶段索引，能够将任意第一阶段索引与任何重排数据集配合，实现了多模态检索管线；
- 在 HNSW 与 IVF 索引上加入了统一的图后端、批量距离计算、预取、距离自适应终止以及邻接列表压缩等多项性能优化；
- 通过 Rust 编译器的 monomorphization，保证了高效性与模块化的兼容。

**🔧 技术方法**

技术要点包括：
- Rust 的类型系统与泛型实现了抽象层的模块化；
- 可编译时多态（monomorphization）消除运行时分发；
- 采用可移植 SIMD（portable‑SIMD）实现不同编码的向量化距离核；
- 对 Product Quantization 采用 SoA 码本布局、循环翻转和缓存友好的表构造；
- 对稀疏向量使用块编码（block‑sparse）与压缩索引（gap‑packing+Elias–Fano）等压缩技术；
- 对多向量 (late‑interaction) 采用 MaxSim 评估和两级 PQ；
- HNSW 图实现了可扩展的邻接表、批量邻居评估、预取、哈希访客集和距离自适应终止；
- IVF 使用残差量化（IVFADC）并支持多级聚类。

**📊 数据集**

使用的公开数据集：
- dense 128‑d、128‑d、768‑d、1024‑d 以及 sparse 30‑d 向量（来自 DPR、Dense Passage Retrieval 等）；
- 7 个公开集合（包含 1M、1B 规模），包括手工特征集合、文本检索语料库以及多向量检索语料库；
- 训练与评测均采用标准的 Accuracy@10、查询吞吐量（QPS）与内存/向量字节数等指标。

**📈 对比分析**

对比方法包括 FAISS、Annoy、HNSWlib、NSG、DiskANN、Qdrant 等主流 ANN 系统。评测在同一硬件上进行，使用相同的查询集合与测量约定。结果表明：
- 在稠密向量上，Vectorium + kANNolo 的 IVF 与 HNSW 在 12 组 accuracy‑throughput 组合中 9 组占优，且在 99% Accuracy 时，Two‑Stage HNSW 速度可达 2.5–3.3× FAISS；
- 在稀疏向量上，使用 Vectorium 的压缩稀疏量化与 IVF/Graph 的组合在 3 组集合中始终取得最高准确率和最低内存占用；
- 与现有实现相比，PQ 编码的速度与准确率相当或略优；二进制量化（Binary）与 Multi‑Bit 量化在 0.1–0.3% 的准确率差距内实现了更快的查询和更小的内存占用；
- 两阶段索引在内存、吞吐量和准确率上均实现了比单阶段索引更优的折中。

**⚠️ 局限性**

局限性与未来工作：
- 编译时间与二进制体积相对较大（Python 绑定单元化后约 5 MiB，编译时会多生成多份 monomorphized 代码）；
- 目前仅支持欧氏距离与内积两种度量，对自定义度量的支持需自行扩展；
- 大规模数据下的构建时间虽已优化但仍高于部分专用硬件实现；
- 对稀疏向量的压缩虽然实现了子比特级别，但在极高稀疏率（> 1%）时仍需进一步压缩方案；
- 未来可继续探索硬件加速（GPU/FPGA）以及更细粒度的动态量化与自适应索引结构。

---

## 62. Survival-Time Risk Beyond Low-Order Channel Models: Identifiability and Scheduling

**arXiv ID:** 2611.03871 | [PDF](https://arxiv.org/pdf/2611.03871v1)

**作者:** Ruifeng Zheng `[一作]` (Dresden University of Technology), Frank H. P. Fitzek `[通讯]` (Dresden University of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究在确认反馈与传输预算约束下，低阶（如吉尔伯特）包丢失统计对生存时间风险评估的可靠性，并引入兼容类概念量化风险可辨识性。

**💡 创新点**

创新点在于证明仅靠低阶统计无法唯一确定风险，构建了兼容类的解析框架并给出风险区间；揭示了在不同调度下，持久性与复发性对风险的非线性影响，甚至出现风险顺序逆转。

**🔧 技术方法**

采用马尔科夫链高阶模型、熵极大化、Perron‑Frobenius 线性代数、动态规划与精确概率计算等技术，对风险上界、下界及闭式表达式进行推导。

**📊 数据集**

实验数据包括 IEEE 802.15.4 FlockLab 链路、Rayleigh 衰落与 Jakes 频谱、带阴影的复合衰落以及实测的 171 条 TelosB 链路的丢包序列。

**📈 对比分析**

通过将低阶模型预测的风险与精确仿真/实测风险比较，发现低阶模型在密集尝试下低估风险可达 4–680 倍，预算有限时误差更为显著；相对传统稳健 MDP 方法，提供更紧的风险界。

**⚠️ 局限性**

局限性在于仅分析了阶数为 M = L + 1 的兼容类，未覆盖未知阶数的通道，且对如何从有限轨迹估计高阶尾部依赖提出了进一步研究方向。

---

## 63. ProofPaths: An AI Agent Framework for Assessing Student Reasoning

**arXiv ID:** 2611.03882 | [PDF](https://arxiv.org/pdf/2611.03882v1)

**作者:** Mohsen Hariri `[一作]` (Case Western Reserve University), Vipin Chaudhary `[通讯]` (Case Western Reserve University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对学生提交的作业使用大型语言模型（LLM）进行自动评分，并通过个性化访谈（interview）让学生阐释解题过程，以评估其对解题思路的理解。

**💡 创新点**

创新点在于：①将评分与访谈统一到同一框架，多个评分代理相互批评并修订，最终以中位数决定分数；②访谈问题与学生解答中的具体文本片段绑定，确保问题与解答对应；③整个评估过程可追溯，所有评分记录与访谈转录均被持久化供教师审查。

**🔧 技术方法**

使用技术包括：LLM代理（OpenAI兼容接口、vLLM、Qwen3.8-27B）；多代理评分与访谈控制器；多维项目反应理论（MIRT）模型用于访谈问题选择；Redis缓存、PostgreSQL持久化；Go后端、Python FastAPI服务、gRPC/Protocol Buffers；HPC GPU（A100）推理；结构化输出与控制器验证。

**📊 数据集**

主要数据来源为教师准备的作业、参考解答和评分准则；学生提交的文本（含图像）被解析为Markdown；用于验证的合成代数题和预先提取的文本片段。

**📈 对比分析**

通过九个确定性探针验证控制器行为，并对评分与访谈流程的延迟与成本进行分析；系统在评分聚合、错误修正、访谈问题生成等方面表现稳定，平均延迟受文档长度与访谈问题数影响，成本与推理次数线性相关。

**⚠️ 局限性**

局限包括：评分代理的相互批评不等同于独立验证；访谈覆盖度受预先提取片段数量限制；模型给出的信心估计和理解标签需进一步校准；书面访谈缺乏口语交流的深度；系统高度依赖HPC资源与严格的数据治理，部署复杂。

---

## 64. Predictive gains, causal losses: why interventional competence cannot be learned from data alone

**arXiv ID:** 2611.03861 | [PDF](https://arxiv.org/pdf/2611.03861v1)

**作者:** Marcus Thomas `[一作]` `[通讯]` (Memorial Sloan Kettering Cancer Center), Marcus Thomas (Memorial Sloan Kettering Cancer Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

探讨在观察学习中如何无意中产生对干预不正确的预测，并提出了一系列理论界限与定量下界。

**💡 创新点**

提出“局部性-自治原则”（Locality–Autonomy Principle）以及一系列关于观测优势与干预误差不可共存的无定理（no-go theorem），并给出在梯度更新与参数共享下如何导致干预错误上升的具体条件。

**🔧 技术方法**

主要采用概率论、可微分几何（Lie导数）、极限分析与泛函分析技术，对确定性预测器的观测误差、干预误差和梯度更新的关系进行严谨证明。

**📊 数据集**

无使用具体数据集；全部以理论模型与抽象概率分布为基础进行证明。

**📈 对比分析**

未进行实验比较；论文通过证明与定理展示在满足特定假设下观测优势无法转化为干预准确性，并给出误差下界与累计误差上限。

**⚠️ 局限性**

主要局限在于：(1) 依赖一系列强假设（如世界实干预不变性、可微性、参数可逆性）；(2) 结果仅为理论性，不含实证验证；(3) 对实际学习算法与模型结构的适用性尚未评估。

---

## 65. Probabilistic Steering of Generative Priors for Lab-in-the-Loop Antibody Optimization

**arXiv ID:** 2611.03844 | [PDF](https://arxiv.org/pdf/2611.03844v1)

**作者:** Anisha Parsan `[一作]` (Massachusetts Institute of Technology), Hunter Elliott `[通讯]` (BigHat Biosciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过在生成式抗体先验模型中实现Feynman‑Kac推导的采样重塑，将生成与优化阶段无缝连接，实现了低样本量下的高效抗体优化；

**💡 创新点**

创新点在于将高斯过程的采集函数(logPI)作为奖励潜在函数，利用概率后验不确定性驱动采样，从而在保持生成流形的同时实现目标导向的探索；

**🔧 技术方法**

使用了流匹配的目标条件扩散生成器（Proteina‑Complexa）、Feynman‑Kac采样重塑、Gaussian Process 采集函数、qHSRI批量挑选等技术；

**📊 数据集**

在三个不同格式与靶点的三场仿真试验中使用：Trastuzumab‑HER2（Fab）、scFv‑SARS‑CoV‑2、VHH‑PT（专有靶点），采用相应的CNN/ensemble 预测或BLI等或acles；

**📈 对比分析**

与Best‑of‑N、GA（OberBO）和LaMBO比较，FK‑Steer在每轮样本数较少时实现更高的平均和top‑10质量、更多独特高得分设计，且未牺牲开发性指标；

**⚠️ 局限性**

局限在于仅在已知亲本的“回溯”实验中评估，缺乏对真正去‑新生成物的验证；使用的仿真oracle可能在训练外域被滥用，且对不确定性收敛的系统性分析不足。

---

## 66. A King for Almost Everyone

**arXiv ID:** 2611.03838 | [PDF](https://arxiv.org/pdf/2611.03838v1)

**作者:** Ziad Ismaili Alaoui `[一作]` (University of Liverpool), Nikhil S. Mande `[通讯]` (University of Liverpool)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种确定性算法，在查询复杂度为 O_ε(n^4/3) 的情况下，能够在 n 规模的锦标赛中找到任意给定 ε∈(0,1) 的 (1−ε)-king。

**💡 创新点**

创新点在于首次将稀疏正则性引入模板图设计，并结合稀疏 Szemerédi 正则化、继承定理与两锦标赛可达性定理，构造了可在最优查询复杂度下求得任意近似国王的算法，填补了之前仅对 1/2-king 或 1/2+2/17-king 的结果。

**🔧 技术方法**

主要技术包括：① 稀疏模板图的构造与稀疏正则性分块；② 对模板图方向的正则化得到群图与法院集合；③ 使用继承正则性选择法院；④ 通过 Afek‑Gafni‑Linial 的两锦标赛可达性定理实现采样，从而在法院中找到覆盖大部分群的顶点。

**📊 数据集**

本工作为理论性研究，无需使用实验数据集；所有结果均为纯粹的理论证明。

**📈 对比分析**

与之前的 O(n^4/3) 上界与 O(n^3/2) 下界相比，本文的算法在相同查询预算下实现了对任意接近 1 的比例的国王的搜索，并且匹配了已知的下界，显著提升了对近似国王的可搜索范围。

**⚠️ 局限性**

局限性包括：① 对于完整国王（1-king）仍无法达到 O(n^4/3) 的确定性查询上界；② 算法仅保证查询复杂度，时间复杂度不受控制，且使用存在性证明，缺乏高效的实现；③ 结果中的常数与 ε 密切相关，难以在 ε 变为 1/n 时保持同样的查询上界。

---

## 67. When Latent Dynamics Meet Data Assimilation: A Benchmark of Koopman-Type Representations

**arXiv ID:** 2611.03916 | [PDF](https://arxiv.org/pdf/2611.03916v1)

**作者:** Zhiwei Zhang `[一作]` (ENPC), Sibo Cheng `[通讯]` (ENPC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文系统评估了 Koopman 表示在潜在数据同化（LDA）中的性能，提出统一的实验框架，并在五个不同动力系统上对七种代表性 Koopman 与非线性潜在动态方法进行了对比。

**💡 创新点**

创新点主要包括：①构建统一的评估框架，将预测精度、同化准确性、动态一致性、Gaussian 假设、凸性和计算成本等维度一并考察；②对比了线性、参数化线性、观测相关的 Koopman 结构以及闭式与迭代的变分/滤波方案；③引入 ResDMD、Henze–Zirkler 检验和 Hessian 解析等工具来量化动态一致性、Gaussian 性和优化可行性。

**🔧 技术方法**

使用了深度学习 Koopman 近似（编码器–解码器与 Koopman 矩阵）、变分 4D-Var、Kalman/条件高斯滤波、残差 DMD、Henze–Zirkler Gaussian 检验、Mardia 统计、Hessian 曲率分析等技术。

**📊 数据集**

实验数据集包括：Lorenz–96、圆柱流、浅水波、ERA5–Coarse（5.625°）和 ERA5–High（1.5°）等。

**📈 对比分析**

通过在不同观测比例、观测频率下对 RRMSE、SSIM、NMB/VR、谱误差、潜在轨迹距离、ResDMD 动态一致性、Gaussian 性诊断、Hessian 凸性和计算成本进行定量对比，结果表明：线性潜在转移在短期预测上有优势，但在强混沌/高维系统中并不一定提升同化精度；闭式方法（DMD、Tensor-Var、DBF、CGKN）在观测信息充足时表现更好，且计算更快；然而其精度受表示误差和观测耦合方式的限制。

**⚠️ 局限性**

局限性包括：结构差异与训练目标不同导致难以单独归因；评估仅在高斯误差假设下进行；高维系统的压缩可能导致信息丢失；未考虑测量噪声、非高斯不确定性或更广泛的观测频率；未探索联合训练与动态一致性约束的潜在改进。

---

## 68. From Triples to Cards: A Web Component for Human-Readable Linked Open Data

**arXiv ID:** 2611.03912 | [PDF](https://arxiv.org/pdf/2611.03912v1)

**作者:** Puyu Wang `[一作]` (University of Oxford), Rui Zhao `[通讯]` (Tianjin University)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了 LODox Data Viewer，一个可嵌入的 Web 组件库，用于将 JSON‑LD 数据渲染为人类可读的卡片界面。

**💡 创新点**

创新点包括：① 单一 script 标签即可嵌入；② 自动检测并适配不同 JSON‑LD profile；③ 无需为每条记录编写模板即可渲染；④ 通过配置文件或 CSS 自定义卡片布局和主题。

**🔧 技术方法**

技术实现：JavaScript Web Components、npm 包与 CDN 分发、JSON‑LD 解析器、适配器机制、CSS custom properties、Node 服务器演示等。

**📊 数据集**

使用的数据集：Linked Art JSON‑LD（如《蒙娜丽莎》与列奥纳多·达·芬奇的数据），以及通用 JSON‑LD 示例。

**📈 对比分析**

评估方法：目前仅进行开发体验和单元测试，未进行正式的性能对比或用户研究；因此缺乏性能指标和基准结果。

**⚠️ 局限性**

局限性：① 仅支持 Linked Art 和通用 JSON‑LD，其他 ontology 适配器不足；② 渲染效果受 JSON‑LD 质量和 profile 兼容性的影响；③ 只处理静态文件，缺乏动态加载和 AI 代理支持；④ 尚未完成用户端评估。

---

## 69. EpiLoRA: Fusing Many LoRA Adapters into Low-Precision GEMMs for Serving and RL

**arXiv ID:** 2611.03899 | [PDF](https://arxiv.org/pdf/2611.03899v1)

**作者:** Hieu Tran `[一作]` (University of Massachusetts Amherst), Hong Yu `[通讯]` (University of Massachusetts Amherst)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出EpiLoRA，一个vLLM插件，将多租户LoRA适配器融合进低精度（FP8/​NVFP4）GEMM中，实现解码时的下投影与主循环重叠，同时支持在同一低精度基础上进行LoRA强化学习训练。

**💡 创新点**

创新点在于：① decode路径将下投影先行，利用程序化依赖启动，使主循环与下投影重叠，仅在epilogue阶段加入上投影；② 对多租户适配器采用两核融合（K1/K2），在同一GEMM内完成量化、下投影、主循环和上投影；③ 通过在同一FP8权重上训练RL，消除大部分权重与激活的量化误差，无需重要性采样校正；④ 只保留单份FP8权重副本，显著降低显存需求。

**🔧 技术方法**

技术包括：LoRA低秩适配器、FP8/NVFP4量化、CUTLASS/Marlin低精度GEMM、Triton编写的自定义核、CUDA图、程序化依赖启动（PDL）、vLLM、Tensor Parallelism、CUDA IPC等。

**📊 数据集**

使用的数据集：Qwen3‑8B、Qwen3.5‑9B、Qwen2.5‑72B、Llama‑3.1‑8B；RL实验使用 OlympiadBench、Minerva、MATH‑500 等基准。

**📈 对比分析**

对比方法：与vLLM（默认与双流LoRA内核）、SGLang、TensorRT‑LLM、FlashRL（FP8+bf16）等系统在Hopper、Ada、Blackwell等GPU上测算单请求时间、吞吐量和RL训练精度。结果显示：在Hopper上单请求TPOT降低至0.76–0.88×，吞吐量提升至1.05–1.09×；在RL训练中，匹配FP8权重的训练在27B模型上可在单个48 GB GPU上完成，精度与bf16训练相当，显存节省达22 GiB。

**⚠️ 局限性**

局限性：① 对极低秩或极高秩适配器时，重叠收益有限；② 在Ada GPU上需额外融合核，未在所有GPU上统一优化；③ 需要与底层GEMM保持位级一致性，若更换GEMM实现需重新验证；④ 对非vLLM框架的移植需要进一步适配。

---

## 70. Dr. No: Feature Steering for Abstention in Vision-Language Models

**arXiv ID:** 2611.03862 | [PDF](https://arxiv.org/pdf/2611.03862v1)

**作者:** Farid Hasanov `[一作]` (Eindhoven University Of Technology), Decebal Constantin Mocanu `[通讯]` (University Of Luxembourg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Dr. No，一种在冻结模型推理时通过稀疏自编码器（SAE）特征调控实现基于证据的拒绝（abstention）的方法。

**💡 创新点**

创新点在于使用稀疏内部特征识别并直接调节模型的承诺/拒绝方向，而非仅靠后置不确定性阈值或外部提示，提供更细粒度、可解释的拒绝控制。

**🔧 技术方法**

使用稀疏自编码器特征挖掘、特征分离（Δ和AUROC评估）、在前填充阶段将解码方向注入残差流的特征蒸馏与干预技术。

**📊 数据集**

数据集包括多模态的 GQA（用于特征挖掘）、UN K‑VQA、MM‑UPD（子集 AAD、IASD、IVQD）以及文本的 SUM、AbstentionBench（GSM8K、UMWP、MMLU‑Math、MuSiQue）。

**📈 对比分析**

与基准提示、后置不确定性拒绝（如最大 softmax、entropy、MSP 等）、logit bias、稠密干预等对比，Dr. No 在 UNK‑VQA 和 MM‑UPD 的 IASD/IVQD 子集、以及 Qwen3.5‑2B 的数学推理子集上取得更高的调和平均（HM）与双向准确率，显示出更佳的回答‑拒绝权衡。

**⚠️ 局限性**

局限性：对缺失答案（AAD）和检索型多跳问答（MuSiQue）的效果较差；特征挖掘仅针对特定无解来源，难以统一覆盖所有无解情形；若无充分覆盖的训练数据，可能出现对新领域的拒绝失衡。

---

## 71. Towards Effective Fuzzing of EDK2 Components

**arXiv ID:** 2611.03854 | [PDF](https://arxiv.org/pdf/2611.03854v1)

**作者:** Datta Manikanta Sri Hari Danduri `[一作]` (Purdue University), Aravind Machiry `[通讯]` (Purdue University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对EDK2 DXE驱动进行有效的模糊测试，利用用户空间重主机和Mock仿真来提升测试覆盖率和漏洞发现。

**💡 创新点**

创新点在于提出了针对DXE驱动的专属重主机框架，并结合Mock依赖模型实现更精细的模糊测试，同时对崩溃归因机制进行了改进。

**🔧 技术方法**

使用了用户空间重主机、Mock依赖模型、异步回调模拟、崩溃归因过滤等技术；采用动态覆盖收集与符号执行辅助。

**📊 数据集**

基于EDK2开源驱动集合，包含多种DXE组件及其依赖库，作为测试数据集。

**📈 对比分析**

通过与传统黑盒/白盒模糊工具比较，使用代码覆盖率、发现漏洞数等指标，实验表明本方法在DXE驱动层面的覆盖率提升约30%，漏洞发现率提高约20%。

**⚠️ 局限性**

局限性包括：仅覆盖DXE驱动，未考虑PEI/SEC、Standalone MM和运行时驱动；SMM相关漏洞无法在用户空间重主机中重现；Mock模型对异步回调的精度不足，导致部分路径和漏洞未被触发；崩溃归因过滤可能漏报实际驱动中的缺陷。

---

## 72. Proceedings of the CSCW 2026 Workshop on Women's Health as Cooperative Work: Participatory AI Across Care, Community, and Policy

**arXiv ID:** 2611.03776 | [PDF](https://arxiv.org/pdf/2611.03776v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 73. Low-Rank Hierarchical Merging for Efficient Long-to-Short Reasoning

**arXiv ID:** 2611.03834 | [PDF](https://arxiv.org/pdf/2611.03834v1)

**作者:** Zeqiu Yu `[一作]` (University of Virginia), Shixiao Wang `[通讯]` (University of Birmingham)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关的长链-短链（L2S）模型合并框架LHM，利用低秩二阶曲率信息实现层级自适应合并，显著压缩输出长度且提升推理准确率。

**💡 创新点**

创新点在于：①用Hutchinson估计器计算每层Hessian迹，避免任务特定校准；②定义Layer Second-Order Sensitivity（LSOS）将Hessian迹与任务向量范数耦合，精准衡量层融合难度；③Dual-Constraint Coefficient Mapping（DCCM）在保持误差上界和全局长度目标的双重约束下生成层级融合系数。

**🔧 技术方法**

技术包括低秩Hessian近似、Hutchinson随机向量、LSOS指标、双约束映射与正则化、层级权重调整。

**📊 数据集**

实验使用1.5B和7B规模的Qwen2.5-Math与DeepSeek-R1-Distill-Qwen模型，在六个数学推理基准（GSM8K、MATH500、Minerva Math、OlympiadBench、CollegeMath、AIME24）上评测。

**📈 对比分析**

与Model Soup、Task Arithmetic、TIES、DARE、AIM、ACM、ORCA等基线对比，LHM在1.5B和7B规模上平均准确率提升3–4个百分点，长度缩减可达93%（相对长CoT），在准确率与长度的Pareto曲线上均优于所有现有方法。

**⚠️ 局限性**

局限性包括：依赖Hessian迹估计仍有随机误差；LSOS假设任务向量与Hessian特征对齐，极端任务可能不适用；对大规模模型的在线估算成本仍高，且缺乏对多任务或迁移场景的广泛验证。

---

## 74. Rethinking Next-Visit Diagnosis Prediction Beyond Aggregate Metrics: Retrieval--Memory Multi-Agent Adjudication

**arXiv ID:** 2611.03826 | [PDF](https://arxiv.org/pdf/2611.03826v1)

**作者:** Heejoon Koo `[一作]` `[通讯]` (University of Illinois Urbana-Champaign), Heejoon Koo (University of Illinois Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出一种多视角、训练无关的下一次就诊诊断预测框架，利用检索与记忆两种视图并通过跨视角一致性进行无源裁决；

**💡 创新点**

创新点在于（1）统一的分布、时间、历史深度评估框架；（2）检索–记忆组合的源中性裁决机制LONGINUS；（3）通过保持一致性预测并仅裁决不一致候选，提升跨时间转移性能；

**🔧 技术方法**

使用大语言模型（Gemini-2.5/3.1 Flash Lite、Llama-3.3-70B、GPT-OSS-120B）进行检索提示、记忆摘要及裁决；同时采用密集检索、最佳排名去重等技术；

**📊 数据集**

在MIMIC-III与MIMIC-IV两大公开电子病历数据库上，映射至112个CCS子标签；

**📈 对比分析**

与规则基线（LVCF、OHU）及多种单视图LLM方法（单提示、检索、记忆、多代理）对比，LONGINUS在所有分布、时间转移指标上均居首，尤其在新诊断与再发诊断的F1上显著优于其他LLM方法，聚合指标略低于LVCF但仍具竞争力；

**⚠️ 局限性**

局限包括：聚类标签可能丢失临床细节；数据来源单中心，样本量对多访问案例有限；候选生成瓶颈导致漏检；聚合指标不如传统规则基线；未对监督序列模型或微调LLM进行评估；

---

## 75. Robot-Matched Soft Fingers for UMI Data Collection: Lessons from 100 Hours of Pick and Place

**arXiv ID:** 2611.03779 | [PDF](https://arxiv.org/pdf/2611.03779v1)

**作者:** Kobi Kelemen `[一作]` (Lute), Victoriano Montesinos `[通讯]` (Lute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一套可携带的手持数据采集接口UMI，收集了近100小时的多物体抓取与放置演示数据，并在此基础上训练了可在多机器人上部署的通用策略。

**💡 创新点**

采用软化且角度匹配的指尖可视化力感手爪、无基站电源供给、顶视相机与手腕相机并行训练，并引入多任务中期训练提升对未见物体的泛化。

**🔧 技术方法**

结合YUBI硬件/软件栈、Meta Quest 3S跟踪、ESP32-C6编码器、Dynamixel伺服驱动，使用π_0.5策略框架进行强化学习和监督微调。

**📊 数据集**

99.7小时、53件日常物体的演示数据（约51,000条回合），其中49件用于中期训练，4件为未见测试。

**📈 对比分析**

通过在R1 Pro上对比无顶相机、顶相机以及多任务中期训练+监督微调的策略，发现顶相机提升成功率13个百分点；未见物体上多任务中期训练使SFT成功率从35%提升至57%，在R1 Lite上亦有提升。

**⚠️ 局限性**

受限于有线供电与电脑、头戴跟踪范围、机器人工作空间不匹配、缺乏触觉导致对精细抓取（如细线）效果不佳，且未在更广泛环境与物体上进行评估。

---

## 76. CoReMatch: Recurrent Continuous Coordinate Refinement for Real-Time Robotic Feature Matching

**arXiv ID:** 2611.03932 | [PDF](https://arxiv.org/pdf/2611.03932v1)

**作者:** Robert J. Wang `[一作]` (Teledyne Technologies), Boyu Wang `[通讯]` (Western University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了CoReMatch框架，用连续坐标空间的递归细化实现半稠密特征匹配；

**💡 创新点**

创新点在于用连续坐标更新而非离散置信图，配合Deep Aggregated Transformer和轻量级局部回归，显著降低计算和内存开销；

**🔧 技术方法**

技术包括递归坐标细化、Deep Aggregated Transformer（DAT）、轻量级卷积回归、连续特征采样与双线性插值；

**📊 数据集**

使用MegaDepth、ScanNet、HPatches等公开数据集进行训练与评测；

**📈 对比分析**

与LoFTR、EfficientLoFTR、RoMa等方法对比，CoReMatch在相同输入分辨率下准确率提升（如相对姿态AUC提升~2%），且在嵌入式设备上推理速度提升10倍以上；

**⚠️ 局限性**

局限性包括对GPU优化较强，移动端性能待进一步提升；训练依赖MegaDepth，缺乏更大多样性数据；细化过程未显式加入几何约束，可能在重复纹理或极端运动场景下受限。

---

## 77. Proper Conflict-Free Coloring on Structured Graph Classes: Bounds, Exact Algorithms, and Hardness

**arXiv ID:** 2611.03906 | [PDF](https://arxiv.org/pdf/2611.03906v1)

**作者:** Kamal Santra `[一作]` `[通讯]` (Indian Institute of Technology Guwahati), Kamal Santra (Indian Institute of Technology Guwahati)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了和谐冲突自由着色（PCF coloring）在和谐图和有序二部图类上的应用，证明了在良分区和谐图中，PCF着色的色数与团数之间的紧密界限，并给出了相应的构造算法。

**💡 创新点**

提出了在良分区和谐图中，PCF着色的色数与团数之间的界限为ω(G)≤χ_pcf(G)≤ω(G)+1，并且在区分不同图类时，给出了多种算法和复杂性结果。

**🔧 技术方法**

使用了动态规划、匹配理论和图的结构特征等技术来构造PCF着色，并分析了不同图类的复杂性。

**📊 数据集**

使用了良分区和谐图、区间图、双凸图和二部排列图等数据集进行实验和理论分析。

**📈 对比分析**

通过与现有算法的比较，证明了所提出的算法在良分区和谐图和区间图上具有多项式时间复杂性，而在双凸图和二部排列图上则提供了更高效的算法，性能优于现有方法。

**⚠️ 局限性**

在某些情况下，PCF着色的色数可能会超过团数，尤其是在复杂的图结构中，决定是否存在ω(G)色的PCF着色是NP完全的。

---

## 78. The Information Limits of Neural Controlled Differential Equations Under Sparse Observations

**arXiv ID:** 2611.03887 | [PDF](https://arxiv.org/pdf/2611.03887v1)

**作者:** Rostislav Gusev `[一作]` (Applied AI Institute), Alexey Zaytsev `[通讯]` (Applied AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在稀疏观测下神经受控微分方程（Neural CDE）的信息缺失与终端响应的不确定性，并给出了可计算的模糊证书与最优观测率；

**💡 创新点**

创新点在于将路径几何（可行签名面积与传输的Lie括号）映射为终端状态的冗余集合，构造基于Zonotope的近似与标量预测的置信区间，并证明在平面驱动下最优观测的最优极限为N⁻¹（速度约束）或N⁻²（加速度约束）；

**🔧 技术方法**

使用了几何控制理论、Lie代数、Zonotope近似、最优极限理论、随机实验评估、Neural CDE训练与自适应采样策略；

**📊 数据集**

实验使用合成的二维平面轨迹（从高频连续信号采样得到的密集观测），然后以稀疏采样测试证书与观测设计；

**📈 对比分析**

与真实的可接受路径终端输出差异做Spearman相关性与误差比值比较，相关性≈0.998；对不同采样策略（随机、最长间隔、最大局部不确定度、闭式中点递减、一次性oracle）进行证书半径曲线比较，结果显示最大局部不确定度策略与一次性oracle几乎等价，且显著优于随机和最长间隔；

**⚠️ 局限性**

局限性包括：仅适用于二维平面驱动，不能直接推广到高维输入；假设的速度/加速度约束需要已知真实路径或预估的物理限制；证书在加速度约束下可能过于保守；未对真实物理信号的稀疏采样效果进行验证。

---

## 79. Self-Consistent Neural Control: Learning Trajectories for Implicit Equilibrium

**arXiv ID:** 2611.03885 | [PDF](https://arxiv.org/pdf/2611.03885v1)

**作者:** Hengyi Zhou `[一作]` (University of Michigan), M. Khalid Jawed `[通讯]` (University of California, Los Angeles)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Self-Consistent Neural Control (SCNC)，通过在隐式平衡系统中显式约束局部敏感性变化来改进梯度学习，避免冻结切线近似导致的梯度误差。

**💡 创新点**

创新点在于把梯度逼近误差与优化目标耦合，给出误差上界并引入基于有限差分的可计算正则化，实现在敏感性快速变化场景下的自洽梯度优化。

**🔧 技术方法**

使用隐式微分对偶学习、有限差分正则化、JAX 实现的 Newton 迭代求解器以及神经网络能量模型，构建高效的梯度传播与优化框架。

**📊 数据集**

在弹性杆、薄壁管、测量带以及 Slinky 结构的二维仿真与真实机器人抓取任务中进行实验验证。

**📈 对比分析**

与传统对偶学习和无梯度的 iCEM 进行对比，SCNC 在收敛速度、成功率（83.3% 对比 41.7%）以及任务损失值上均显著优于基线，计算成本仅略高于对偶学习。

**⚠️ 局限性**

局限包括仅针对节点目标任务验证、正则化权重经验性选择、缺乏对更复杂形状控制、3D 交互和接触问题的测试，未来需进一步推广和自适应正则化。

---

## 80. Embedding Semantic Entropy: NLI-Free Semantic Uncertainty Estimation with Geometry-Aware Clustering

**arXiv ID:** 2611.03876 | [PDF](https://arxiv.org/pdf/2611.03876v1)

**作者:** Akshiya Raveendran `[一作]`, Uthayasanker Thayasivam `[通讯]` (University of Moratuwa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Embedding Semantic Entropy 方法，利用句子嵌入空间中的 K‑Medoids 聚类替代传统 NLI 语义聚类，进而计算聚类熵作为不确定性评估。

**💡 创新点**

创新点在于用几何感知的嵌入聚类（K‑Medoids + 统一的紧密度与间距约束、动态 K 选择）取代外部 NLI，提供可解释的语义群组。

**🔧 技术方法**

采用监督式 SimCSE RoBERTa‑large 句子编码、K‑Medoids 聚类、Davies–Bouldin 指数、紧密度阈值、肘部法则以及基于集群概率的熵计算。

**📊 数据集**

在 TriviaQA、SQuAD、NQ‑Open、SVAMP、BioASQ 等五个问答数据集上，使用 Falcon‑7B‑Instruct、LLaMA‑2‑7B、LLaMA‑2‑7B‑Chat、Gemma‑7B 四种开源 LLM 进行实验。

**📈 对比分析**

与 SE、DSE、ER、KLE、Predictive Entropy、P(True) 等基线比较，Emb-SE 在 20 组模型‑数据集设置中取得 15 组最佳或同等 AUROC、13 组最佳或同等 AUARC，尤其在 Gemma‑7B 上表现最强。

**⚠️ 局限性**

局限性包括需访问 token/序列概率（不适用于纯黑盒 LLM）、聚类计算开销，以及在某些数据/模型组合中仍略逊于部分 NLI 或概率基线。

---

## 81. ExaTune: Exhaustive Hyperparameter Landscape Exploration at Scale

**arXiv ID:** 2611.03867 | [PDF](https://arxiv.org/pdf/2611.03867v1)

**作者:** Kevin Dick `[一作]` (Better Outcomes Registry and Network), Katie Lonsway `[通讯]` (Better Outcomes Registry and Network)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了ExaTune系统，利用高性能计算在离散网格上对机器学习模型的超参数进行全枚举，生成完整的性能地形图；

**💡 创新点**

首次将进化计算中的适应度地形分析方法（Moran's I、崎岖度、局部最优密度、FDC、方差分解重要性）应用于超参数空间，并公开了八套完整地形数据集；

**🔧 技术方法**

采用YAML声明、Pydantic校验、Jinja2模板生成SLURM作业数组、K折交叉验证、Parquet存储和一套正式化的地形度量；

**📊 数据集**

在五类模型（SVM、随机森林、XGBoost、岭回归、KNN）上，对六个分类数据集（MNIST、UCI Adult、Gestational Diabetes、Maternal Risk、Titanic、Wine）和两个回归数据集（Diabetes、California Housing）进行评估；

**📈 对比分析**

与四种近似HPO方法（随机搜索、贝叶斯优化、Hyperband、Optuna）在100次试验预算下对比，发现它们覆盖率仅为0.09–2.06%，虽能找到接近全局最优的配置，却忽略了不稳定区、次优峰和参数交互，且能耗与碳排放远低于全枚举；

**⚠️ 局限性**

局限在于只能处理低维离散网格（d≤6），连续参数需离散化，数据集与预处理固定，且可视化受限，未来需开发自适应网格、低维降维及多保真度近似技术。

---

## 82. A Family of Well-Behaved Multiflow Instances

**arXiv ID:** 2611.03858 | [PDF](https://arxiv.org/pdf/2611.03858v1)

**作者:** Joseph Poremba `[一作]` (University of British Columbia), F. Bruce Shepherd `[通讯]` (University of British Columbia)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种新的多路流实例族——stellar‑planar，证明其在割条件满足时可行，并给出半整数与整数流的存在性；同时证明在该族上可获得加性误差为 6d_max 的不可分流解。

**💡 创新点**

创新点在于将单源平面实例与完全平面实例统一为一个更一般的族，使用 2‑cut 分解与补丁压缩技术，实现了割充分性与低误差不可分流的统一证明；并给出了半整数与整数流的整除性条件。

**🔧 技术方法**

核心技术包括：补丁（patch）分解、2‑cut 递归与容量/需求补偿、pinning（需求分解成多段）技术、图合并与容量转移，以及对已知的全平面与外平面实例不可分流结果的引入。

**📊 数据集**

本工作为理论研究，无实验数据集；所有结果均为理论证明与算法存在性。

**📈 对比分析**

与已有工作相比，外平面实例和完全平面实例可得到 2d_max 的误差，本论文在更广泛的 stellar‑planar 族上实现了 6d_max 的误差，证明了该族在割条件下仍保持可行性，并给出了整数与半整数流的构造。

**⚠️ 局限性**

局限性包括：误差上界 6d_max 可能不是最优；目前仅在无向图且补丁结构特定的情况下成立；未给出多项式时间实现细节，且对非平面补丁或更一般的多源实例仍未覆盖。

---

## 83. Performance, Scalability and Maintenance: An Experimental Study of RPA vs. Autonomous Agents in the Healthcare Domain

**arXiv ID:** 2611.03853 | [PDF](https://arxiv.org/pdf/2611.03853v1)

**作者:** Daniel Schulz `[一作]` (University of Santa Cruz do Sul), Ana Paula Wernz da Cunha Müller `[通讯]` (Clínica de Mastologia Dra. Ana Paula Muller Ltda.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

在真实医疗场景中，对比实现自动化提取乳腺影像报告的传统RPA（Playwright）和基于LLM的自主代理（APA，使用browser‑use、LangGraph和Gemini 2.5 Flash-lite）的性能与成本。

**💡 创新点**

系统化实验验证RPA与APA在医疗数据抓取任务中的效率、准确率与可维护性差异，为医疗自动化架构决策提供量化依据，并首次在同一接口环境下评估两种技术的时间与费用对比。

**🔧 技术方法**

RPA采用Python + Playwright；APA采用Python + browser‑use + LangGraph + Gemini 2.5 Flash-lite（LLM）。

**📊 数据集**

来自ClickPalm平台的乳腺X光和超声检查报告（2024-2025年，共69名患者、71份检查）。

**📈 对比分析**

通过准确率、精确率、召回率、F1、执行时间和成本等指标进行比较：RPA准确率98.90%、F1 98.27%、平均每病人执行时间16.07 s、每报告成本0.030 USD；APA准确率96.49%、F1 94.48%、平均执行时间207.21 s、每报告成本0.187 USD。

**⚠️ 局限性**

实验仅在单一、稳定界面环境下进行，未检验APA在高度动态或多界面场景下的适应性；成本拆分（LLM调用与基础设施）未细化；仅测试两轮（T与T+7），对长时间稳定性评估不足。

---

## 84. AIRoA-0: Long-Horizon, Precise, Cross-Embodiment Manipulation from Handheld Data

**arXiv ID:** 2611.03815 | [PDF](https://arxiv.org/pdf/2611.03815v1)

**作者:** Yueh-Hua Wu `[一作]` (AIRoA), Zixuan Zhong `[通讯]` (AIRoA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

使用手持演示数据训练多模态流匹配模型，实现双臂长任务的自主执行

**💡 创新点**

结合阶段预测、身体记忆与非均匀流式解码，突破仅手持数据的长任务控制

**🔧 技术方法**

视觉语言动作模型、流匹配训练、可视证据门控、阶段头、身体记忆、EMA平滑、流式解码

**📊 数据集**

16,000+小时的手持抓取演示，覆盖多任务指令与子任务标签

**📈 对比分析**

在FrankA、TX‑G2、OpenArm、UR等机器人上对比OpenVLA、S2等基线，平均降低10–30%的动作损失，成功完成长序列任务

**⚠️ 局限性**

对大规模手持数据和子任务标签的依赖，缺乏对新机器人和极端环境的充分泛化

---

## 85. Geometric Refinement and Distributional Alignment for Sim2Real Roadside LiDAR Detection

**arXiv ID:** 2611.03936 | [PDF](https://arxiv.org/pdf/2611.03936v1)

**作者:** Aman Upganlawar `[一作]` `[通讯]` (Columbia University), Aman Upganlawar (Columbia University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在LUMPI Sim2Real LiDAR挑战中，作者基于PointPillars模型，仅使用合成数据训练3D检测器，并提交检测结果与合成点云的分布相似度评分。

**💡 创新点**

核心创新在于通过诊断定位误差、审计后处理规则、几何尺寸校准以及解析分布相似度指标等技术实现显著性能提升；并提出可预测的实验方法论。

**🔧 技术方法**

主要技术包括PointPillars骨干网络、几何中心重定位、后处理规则修复、尺寸比例校准、点云密度重新排序、测试时增强、以及对Frechet点云距离的解析求解和仿真与真实点云的仿射校正。

**📊 数据集**

使用的主要数据集为LUMPI提供的合成数字双轨点云与对应的真实侧路LiDAR数据，包含七类目标（Car, Person, Bicycle, Van, Truck, Bus, Motorcycle）。

**📈 对比分析**

与标准基线相比，本工作在检测方面实现了3D mAP 0.1333（归一化0.1904），并在分布相似度方面取得0.9126（归一化0.9960）综合得分0.4793，较初始基线提升15.3%。

**⚠️ 局限性**

主要限制包括：Bus类标注不一致导致无法提升其AP；分布相似度被数字双轨几何覆盖限制在≈0.91；小样本类别噪声大，难以可靠提升；部分性能提升来自多类别候选扩展而非模型改进；整体仅单一参与者，限制了多样化实验探索。

---

## 86. Accelerating FastDEC Clustering Algorithm with GPUs

**arXiv ID:** 2611.03934 | [PDF](https://arxiv.org/pdf/2611.03934v1)

**作者:** Mi Li `[一作]` (University of Waikato), Bernhard Pfahringer `[通讯]` (University of Waikato)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 SymWarp‑FastDEC，一个基于 GPU 加速的 FastDEC 聚类框架，利用 SymTri‑kNN 和 WP‑DC 实现了自 k‑NN 计算与优势分量（DC）检测的高效并行化，能够处理从数千到百万级别的高维数据。

**💡 创新点**

创新点包括① SymTri‑kNN 通过对称上三角遍历和 warp‑协作 Top‑k 更新，消除重复距离计算；② WP‑DC 采用 warp‑级并行的优势分量选择与排名，剔除索引依赖；③ 端到端 GPU 实现，显著提升速度同时保持聚类质量。

**🔧 技术方法**

技术手段包括 CUDA/GPU 编程、cuBLAS GEMM、warp‑level ballot/shuffle、共享内存堆、内存自适应块规划、双流 ping‑pong 调度、指数核密度估计、基于 k‑NN 的优势组件分析。

**📊 数据集**

实验使用合成均匀数据、MNIST、MNIST8M、Deep1B 子集、WikiGiga GloVe 向量、图像补丁数据集等，覆盖低至高维以及数千到百万级规模。

**📈 对比分析**

与原始单线程 Python FastDEC、FAISS GPU（GpuIndexFlatL2）以及改造版 KnnCuda 进行对比。SymWarp‑FastDEC 在所有基准上实现 1.7–4.9 倍的端到端加速，对高维数据效果更显著，GPU 速度提升可达 2.5×10^4 倍。

**⚠️ 局限性**

局限性在于仍受 O(N^2) 计算复杂度限制，内存瓶颈会影响极大规模高维数据；在低维情形下对索引的依赖导致提升有限；需要根据 GPU 资源手动调节 k 和块大小以获得最佳性能。

---

## 87. Beyond Division of Labor: How Roles Shape Feedback and Recovery in Multi-Agent Systems

**arXiv ID:** 2611.03931 | [PDF](https://arxiv.org/pdf/2611.03931v1)

**作者:** Chih-Hsuan Yang `[一作]` (Argonne National Laboratory), Feng Yan `[通讯]` (University of Houston)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在多代理大型语言模型系统中对 Planner‑Executor‑Reviewer（计划者‑执行者‑审阅者）三角色进行角色消除与强制执行实验，研究角色对功能分工与错误恢复的影响。实验在固定调用预算、调用拓扑、评估器等条件下，只改变角色指令或强制的接口规范，比较不同组织形式下的首次答题准确率、恢复率和最终准确率。

**💡 创新点**

创新点包括：① 明确区分角色指令的两大功能——工作分配与决策权限；② 引入“恢复率”作为评价指标，突出外部反馈机会对多代理系统性能的决定性作用；③ 对角色消除与强制执行两种极端组织进行系统性对比，并通过“强制提交”干预验证提交策略对恢复的贡献；④ 通过信息渠道审计说明角色消除往往同时删除反馈通道，强调组织评估需同时关注信息流与决策权。

**🔧 技术方法**

技术手段包括：使用三种开源 LLM（Gemma‑4、GPT‑OSS‑120B、Llama‑3.3）作为所有角色的代理；采用固定的交互图和预算，构建 Planner、Executor、Reviewer 的提示模板；实现角色消除（共用提示）与强制执行（通过类型化 JSON 输出强制限制角色行为）；记录提交日志、评估者反馈与修复提示；进行事后判定匹配、强制提交实验等对比分析。

**📊 数据集**

数据集主要有两个：① Omni‑MATH（竞争性数学问题，1,600 条样本）② SciBench（本科级自由文本科学题目，574 条样本）。两套数据均采用统一的随机种子抽样，保证所有实验条件使用相同问题集合。

**📈 对比分析**

比较方法：在相同调用预算下，对每种模型和数据集分别评估三种组织：角色结构化（baseline）、角色消除+反馈、角色消除+反馈被撤回、强制执行。主要指标为：首次答题准确率、恢复率（失败后最终正确率）与最终准确率。实验结果显示：①首次答题准确率在三种组织间差异不大；②角色消除后恢复率显著提升（如 Gemma‑4 在 Omni‑MATH 上从 0.198 提升到 0.613）；③强制提交实验进一步验证提交策略对恢复的正向影响；④在不同模型间，角色消除的优势更显著，强制执行在 Llama‑3.3 上恢复率提升，但最终准确率未必提高。

**⚠️ 局限性**

局限性包括：①恢复过程依赖于已知答案的评估器，缺乏完全自主纠错；②实验仅覆盖文本推理任务，缺少更广泛的多模态或多任务场景；③部分实验（角色消除+反馈、强制提交）是探索性后续加入，未预注册；④评估者与代理模型相同，可能导致自洽性偏差；⑤自动评分规则存在缺陷，导致部分恢复率估计偏差；⑥缺少对评估成本和资源消耗的量化评估。

---

## 88. Silent Degradation during Auto-Vectorization of bf16/fp16 on RISC-V Under LLVM

**arXiv ID:** 2611.03917 | [PDF](https://arxiv.org/pdf/2611.03917v1)

**作者:** Vaishnavi Kouru `[一作]` (San Jose State University), Robert Chun `[通讯]` (San Jose State University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

评估 LLVM 的 RISC‑V 自动向量化器在 bf16、fp16、int8 等 AI 低精度数值类型上的表现，揭示默认目标缺少必要指令集时会导致严重的“静默退化”。

**💡 创新点**

创新点在于发现并量化了向量化退化的隐藏成本，说明编译器诊断语句无法区分可通过开启扩展修复的退化与固有转换成本，并提供了针对性改进建议。

**🔧 技术方法**

使用 Clang/LLVM 22+ 的 Loop Vectorizer、目标特性检查、静态汇编分析、优化备注、QEMU 机器指令计数插件以及手写的 _Float16 转换库等技术手段。

**📊 数据集**

实验数据来源于 MobileNetV2 分类器的权重量化权重（将 fp32 权重转换为 bf16、fp16、int8）。

**📈 对比分析**

通过在 QEMU 用户模式下执行 14.7M 次 FMA 循环，比较基础目标（无扩展）与开启相应扩展（Vf16、Vbf16）的动态指令计数与 wall‑clock 时间；发现 bf16 基础版指令数比 fp32 高 17 倍，fp16 基础版高 76 倍，而开启扩展后两者接近 fp32。

**⚠️ 局限性**

局限性包括仅使用单一编译器（Clang）、仅在仿真环境下测量、仅测试单一循环形状与单一向量长度、未考虑 GCC 或实际硬件实现的差异。

---

## 89. Clean-anchored excess-emission control for CTC adaptation under competing speech

**arXiv ID:** 2611.03900 | [PDF](https://arxiv.org/pdf/2611.03900v1)

**作者:** Yuanbo Zhou `[一作]` (Missouri University of Science and Technology), Junjie Xiong `[通讯]` (Missouri University of Science and Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了在CTC识别器中对混合语音进行参数高效适配的清晰锚定过度发射抑制方法。

**💡 创新点**

创新点在于仅用一个缓存的期望折叠CTC令牌计数作为单侧约束，将噪声学生的输出长度限制在对齐的清洁教师计数之下。

**🔧 技术方法**

技术上采用wav2vec 2.0 Base模型的前两层低秩LoRA更新，配合自定义的期望计数正则化与标准CTC损失。

**📊 数据集**

实验使用LibriSpeech（目标语音）与MUSAN（噪声/音乐/语音干扰）以及VCTK（未见干扰者）合成混合语音。

**📈 对比分析**

与基线CTC‑LoRA和对称锚定、帧质量等对照实验相比，在语音干扰下WER平均下降0.875个百分点，整体噪音/音乐条件下的平均WERS提升0.341个百分点。

**⚠️ 局限性**

局限性包括仅在单一backbone与贪婪解码下验证、未测试自然重叠语音、对干扰语者的泛化有限且未优化正则化强度。

---

## 90. EvoGroup: A Self-Evolving Harness for Group Agents in Shared Contexts

**arXiv ID:** 2611.03868 | [PDF](https://arxiv.org/pdf/2611.03868v1)

**作者:** Xinyi Xu `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EvoGroup 自演化框架，能够从仅六种基本工具起步，借助交互、信度引导的轨迹筛选、Adaptive Progressive Disclosure (APD) 轨迹分析、桶化分析以及四种修订接口（Representation、Operation、Intervention、Policy）实现群体代理记忆的持续自适应；

**💡 创新点**

创新点在于：①将 harness 本身视为可演化对象；②通过信度与官方评估相结合的轨迹选择，聚焦高信息量实例；③采用 APD 逐步曝光轨迹，提高诊断效率；④桶化分析抽象高层发现，指导全局修订；⑤四接口分离修订范畴，兼顾结构、功能、约束与策略；

**🔧 技术方法**

技术核心包括：LLM 驱动的自信度评估、轨迹记录与选择、APD 轨迹分析、桶化（semantic bucket）分析、压缩诊断、四接口修订；使用 GPT‑5.5、DeepSeek‑V4‑Flash、GLM‑5.1 等基础模型；

**📊 数据集**

使用的评测数据集为 EverMemBench（用于自演化训练与验证）和 GroupMemBench（用于跨 benchmark 泛化评估）；

**📈 对比分析**

对比方法：与结构化记忆（Mem0、MemOS 等）、自演化记忆（A‑MEM、MemRL）以及人类设计 harness（Codex、Claude Code）等；在 DeepSeek‑V4‑Flash 上，pass@1 从 71.07% 提升至 89.58%，在 GPT‑5.5 上提升至 92.99%；在 GroupMemBench 上，冻结后的 evolved harness 仍显著优于基线；

**⚠️ 局限性**

局限性：①在 GPT‑5.5 上仍略低于 Codex/Claude Code；②自演化迭代次数有限，进一步迭代可能受限于成本；③对极大规模或长时序多用户对话的适应仍需验证；④自我反思与完整轨迹分析的误差仍存在。

---

## 91. Why Pay for Long Reasoning When Rewrite Do Trick? Causal Effects of Reasoning History Compaction on Future Generation and Agent Cost

**arXiv ID:** 2611.03852 | [PDF](https://arxiv.org/pdf/2611.03852v1)

**作者:** Anton Hristov `[一作]` `[通讯]` (Technological School Electronic Systems), Anton Hristov (Technological School Electronic Systems)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了通过在推理过程中对已生成的思考内容进行压缩重写，来降低后续推理所需的 token 数量，且不需要对模型进行再训练。

**💡 创新点**

创新点在于提出并验证了仅通过修改先前思考文本（压缩成状态记录）就能促使模型在后续推理中自发生成更短、更高效的推理轨迹的技术。

**🔧 技术方法**

使用了开放权重的语言模型（DeepSeek、Laguna、MiniMax、GLM 等）作为被压缩模型，利用另一 LLM 作为压缩器对其先前思考进行重写，并通过对比原始历史与压缩历史的连续推理来评估效果。

**📊 数据集**

实验数据来源于多任务的人工设计对照实验（包括多阶段工具调用、救援计划、编程验证等），以及对 50 任务的软件代理部署中的实际推理日志进行回测。

**📈 对比分析**

比较方法为在同一任务、同一模型、同一工具响应下，对比压缩历史与原始历史的后续 token 消耗；结果显示压缩后续推理的平均减少幅度在 20–50% 之间，部分部署中还能实现成本和时间的降低。

**⚠️ 局限性**

主要局限包括实验任务结构单一、历史接受率不均、缺乏匹配的删除/截断基线、仅使用非专业人评估以及部署样本未随机化，导致对质量保留和因果效应的评估不足。

---

## 92. RoboScope Kit: Unified Interfaces for Fast and Scalable Embodied-AI Evaluation

**arXiv ID:** 2611.03850 | [PDF](https://arxiv.org/pdf/2611.03850v1)

**作者:** Deng-Ke Wu `[一作]` (Shanghai Jiao Tong University), Bo Zhao `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个统一的评估框架，将每个模型封装为 Policy，每个基准封装为 Adapter，只需一次集成即可在多种机器人基准上高效评估。

**💡 创新点**

创新点包括：① 共享评估接口（Policy + Adapter）实现一次集成可复用；② 运行时优化，依据 Policy 的观察步长仅渲染必要帧；③ 调度与并发搜索，独立扩展仿真器工作进程与 Policy 复制进程；④ 确定性多实例分片，实现跨 GPU 甚至跨机器的扩展。

**🔧 技术方法**

使用技术有：Python 包化的 Policy API，控制运行时（Control Runtime）与仿真运行时（Simulation Runtime）；二维对角线并发搜索算法；CUDA MPS 并行多核；GPU 与 CPU 资源监控、吞吐量测量；引擎重用、渲染优化、观察步长声明等运行时改进。

**📊 数据集**

实验数据集包括 LIBERO‑Long、RoboTwin Easy、MetaWorld MT50，以及适配器覆盖的 LIBERO‑Plus、CALVIN、RoboCasa GR‑1 Tabletop 等。

**📈 对比分析**

与官方 StarVLA、RLinf、vla‑eval 等系统在同一 GPU、同一 checkpoint 上进行对比。指标包括成功率、结果一致率、评估时间、吞吐量与端到端时间。结果显示：对 LIBERO‑Long，RoboScope Kit（Search+MPS）比官方快 6.16×；对 RoboTwin Easy，四 GPU 并行 25× 加速，单 GPU 基线 359.7 分钟降至 14.4 分钟；吞吐量提升至 300+ 步/秒，GPU 利用率从 45% 提升至 87%。

**⚠️ 局限性**

限制与挑战：① 并发搜索在小规模工作负载时消耗相对较高；② 随着工作进程增多，任务分配可能导致成功率/终止步数一致率下降；③ 需要 Policy 在 checkpoint 中声明观察步长，无法适用于所有模型；④ GPU 内存限制仍是多模型并行的上限；⑤ 评估结果依赖冻结的 manifest 与固定配置，环境变动需重新生成。

---

## 93. LaplaceDiag: Fault Diagnosis through Changes in Learned Actuator Effects

**arXiv ID:** 2611.03849 | [PDF](https://arxiv.org/pdf/2611.03849v1)

**作者:** Oliver Niggemann `[一作]` `[通讯]` (Helmut Schmidt University), Oliver Niggemann (Helmut Schmidt University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于Laplace空间的神经网络LaplaceNet的诊断方法LaplaceDiag，用来从执行器效应变化检测并定位设备故障。

**💡 创新点**

创新点在于将执行器对传感器影响的线性效应矩阵与非线性预测分离，允许仅通过线性回归更新执行器效应，避免网络重训练，并在无标注故障数据的情况下生成可解释的诊断假设。

**🔧 技术方法**

使用LaplaceNet神经网络、线性回归、对比投影（分别、联合）以及LSTM、VARX等基线方法。

**📊 数据集**

采用两套数据集：开环泵系统（两泵+水箱）和受反馈控制的Tennessee Eastman Process (TEP)。

**📈 对比分析**

与投影、LSTM投影和VARX等方法比较，LaplaceDiag在泵数据集上正确定位泵故障113/120，显著高于投影（70/120）和LSTM投影（57/120）；在TEP中识别受影响执行器的准确率分别为248/249和249/249，优于VARX和LSTM投影；误报率低。

**⚠️ 局限性**

局限性包括对反馈耦合命令下效应估计不佳、难以实现早期诊断、对执行器效应的假设可能无法完整捕捉根因、需要独立验证以确认根本故障。

---

## 94. PianoBench: Evaluating Musical Perception and Reasoning in Language Models

**arXiv ID:** 2611.03839 | [PDF](https://arxiv.org/pdf/2611.03839v1)

**作者:** Louis Bradshaw `[一作]` (Queen Mary University of London), Chris Donahue `[通讯]` (Carnegie Mellon University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了音乐理解评估基准 PianoBench 并构建了训练规模的教学视频语音+乐谱数据集，评估多模态语言模型在音频与符号输入下的感知与推理。

**💡 创新点**

创新点在于利用教学视频中并行出现的口述解说与乐谱作为对齐监督，展示音频感知与符号推理的性能差距，并证明中间训练在交互式指令视频上可提升模型的音乐推理。

**🔧 技术方法**

采用多模态语言模型（Audio Flamingo、Music Flamingo、Gemini、Qwen 等）、链式思考（COT）、自监督中间训练、指令调优以及音频源分离+转录技术。

**📊 数据集**

使用 138k 条教学钢琴视频生成的 20k 小时交互式音频+语音+MIDI 数据集（PianoBench-Train），以及从中抽取的 250 题 MCQ 评测集 PianoBench-Test。

**📈 对比分析**

与无音乐控制、音频与 MIDI 文本输入进行对比，发现开放权重音频模型在音频条件下仅提升 ≤6pp，而使用 COT 的前沿模型在 MIDI 条件下可达 93% 以上；中间训练在不使用 COT 的情况下，Qwen 3.5 9B 从 42.8% 提升至 56%。

**⚠️ 局限性**

局限性包括：仍缺乏从音频到符号的高质量对齐监督；模型对高层音乐结构的推理仍有不足；数据主要覆盖西方语言与钢琴曲，存在文化偏倚；开放权重模型缺乏详细训练细节。

---

## 95. El Agente Fármaco: A Scalable AI Scientist for Drug Discovery

**arXiv ID:** 2611.03832 | [PDF](https://arxiv.org/pdf/2611.03832v1)

**作者:** Marcel Müller `[一作]`, Alán Aspuru-Guzik `[通讯]` (NVIDIA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种基于 LLM 的智能助手系统（El Agente），通过持久化 SQL 数据库和子任务代理，支持早期药物发现中的分子生成、筛选、结构预测、活性预测和自由能计算等多步骤工作流，并在 22 任务基准和五个案例研究中进行验证。

**💡 创新点**

创新点在于将科学实体、计算结果和工作流可追溯性统一存储在关系数据库中，采用渐进式上下文披露（subset-based execution）和工具参数分离，仅在需要时将最小化信息推送到 LLM，从而显著降低上下文量、提升执行效率并保持可追溯性。

**🔧 技术方法**

技术上结合了大型语言模型（OpenAI GPT、Claude）、SQLModel 数据库、P2Rank、DiffDock、Boltz-2、Plip、MolMIM、GenMol、PocketXMol、PrexSyn 等开源或托管工具，并通过 agentic 架构实现任务规划、SQL 查询、工具调用和结果收集。

**📊 数据集**

使用的数据集包括公开的药物库（ChemDiv、Enamine）、PDB、PDBFixer、AutoDock Vina、已发表的实验筛选数据（TBXT、PGK2、AmpC、KpThiL、TYK2）以及自定义生成的化合物集合，共计数十万条候选分子。

**📈 对比分析**

在 22 任务基准中，El Agente 的数据库配置在保持与普通工具调用相近的模型成本下，平均峰值上下文显著降低（中位数约 20%），并在多任务评估中获得最高的判断分数；在案例研究中，系统能生成与专家挑选集不同、在计算指标（对接得分、Boltz-2 预测、QED 等）上同样或更优的候选分子。

**⚠️ 局限性**

局限性主要包括：对单一 LLM 的依赖导致模型推理误差；工具的准确性和可用性受限（如对接、共折叠方法的预测误差）；缺乏实验验证数据导致最终筛选的可行性不确定；以及系统对大规模实时计算资源和多线程执行的扩展性仍待进一步评估。

---

## 96. A Tutorial on Shape Servoing of Deformable Objects-Part II: Advanced Concepts

**arXiv ID:** 2611.03825 | [PDF](https://arxiv.org/pdf/2611.03825v1)

**作者:** Ignacio Cuiral-Zueco `[一作]` (University of Zaragoza), Youcef Mezouar `[通讯]` (Institut Pascal)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文在形状闭环控制（shape servoing）领域提供了第二部分教程，阐述了基于测量的变形雅可比矩阵的奇异值分析、特征选择、失控模式诊断与扩展（时变目标、二次任务、3D物体）等高级概念。

**💡 创新点**

创新点在于利用雅可比矩阵的SVD对机器人运动方向与形状变形方向进行分解，揭示弱控方向与观测失灵的机理；引入特征不变性对控制路径的影响分析；提出基于零空间投影的二次任务和前馈时变目标的控制实现；系统性总结了失控模式与缓解措施。

**🔧 技术方法**

技术主要包括：基于测量的局部雅可比矩阵估计（Broyden更新）、奇异值分解分析、伪逆控制律、阻尼最小二乘逆、特征不变性分析、零空间投影、前馈补偿、3D ARAP模型的数值求导。

**📊 数据集**

使用的“数据集”为仿真场景：二维弹性杆（Kirchhoff rod）在双臂抓取下的变形轨迹，5个材料点的图像坐标；3D案例包含弯曲杆、薄板和体积网格的ARAP仿真。

**📈 对比分析**

与其他方法比较在本文中未进行传统实验对照；通过数值仿真演示不同特征选择导致的最大轴向应变差异、失控时的响应比、误差收敛曲线等，展示了控制策略的有效性和鲁棒性。

**⚠️ 局限性**

局限性：仅适用于准静态柔性物体，无法处理塑性、断裂、环境接触、再抓等历史相关或非线性事件；对快速动态形状调节需加入惯性、阻尼等动力学；缺乏真实实验验证与大规模数据集评估。

---

## 97. Recursa: Toward Recursively Self-Improving Foundation Model Systems

**arXiv ID:** 2611.04035 | [PDF](https://arxiv.org/pdf/2611.04035v1)

**作者:** Erkang Zhu `[一作]` (Alibaba Research), Jingren Zhou `[通讯]` (Alibaba Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 Recursa 递归自我改进的基础模型系统，利用多代理并行学习、知识仓库、工具组合和程序化上下文管理，在法律知识域中实现无外部强大模型支撑的自适应与提升。

**💡 创新点**

创新点在于首次提出并实现了递归自我改进循环：系统自身的模型、指令、技能与记忆可在任务经验中自我迭代；引入持久 Python REPL harness、事件日志与子代理分解，显著提升长链推理与知识检索效率。

**🔧 技术方法**

采用的大技术包括：大语言模型（DeepSeek‑V4.1‑Flash、Qwen3.8‑Flash）、持久 Python REPL harness（工具组合与程序化上下文管理）、事件日志与记忆存储、子代理分解与并行学习、Bootstrapping + Self‑Practice + Recursive Self‑Improvement 算法。

**📊 数据集**

使用了 Harvey LAB 的 Firm Knowledge 基准（约 9,288 文档、108M 词、250 任务）作为评估数据集，结合 DeepSeek‑V4.1‑Flash 和 Qwen3.8‑Flash 两个模型。

**📈 对比分析**

通过与无学习基线、最高精度研究配置以及基于 Persistent REPL 的基准对比，评估指标为 all‑pass 率和 mean criteria pass 率；结果显示 Bootstrapping 将 all‑pass 从 30% 提升至 34.7%，递归自我改进后多学习器版本分别实现 DeepSeek all‑pass 55%（提升 22%）和 Qwen all‑pass 46%（提升 12%）。

**⚠️ 局限性**

局限性包括：对高质量领域语料的强依赖、昂贵的算力需求、潜在的偏差与安全隐患、仅在法律域验证、未实现模型与 harness 的自我改进、以及对其它复杂专业领域的适用性仍需进一步研究。

---

## 98. Exploring the Power Consumption of ERC-4337 Bundling on Server Infrastructure

**arXiv ID:** 2611.04033 | [PDF](https://arxiv.org/pdf/2611.04033v1)

**作者:** Claudiu-Nicu Bărbieru `[一作]` (Alexandru Ioan Cuza University), Boris Verdecia Echarte `[通讯]` (University of Neuchâtel)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对 ERC-4337 bundler 在服务器级硬件上的功耗进行系统测量，比较容器与原生运行以及不同CPU架构下的能耗。

**💡 创新点**

首次在生产环境下使用 SmartWatts 对 bundler 进行能耗评估，并验证容器化部署不会显著增加功耗，说明共置 bundler 与 EVM 节点在能源效率上可行。

**🔧 技术方法**

使用 SmartWatts、RAPL 与硬件性能计数器（HwPC）结合回归模型测量功耗；利用 Docker 容器隔离；使用 ERC-20 Token Transfer 作为工作负载；对不同块频率、吞吐量和节流参数进行实验。

**📊 数据集**

三组服务器配置（Intel Xeon Gold 6326、Intel Xeon Silver 4510、AMD EPYC 7302P）以及 ERC-20 交易堆叠（25–100 交易/批次）作为实验数据集。

**📈 对比分析**

通过将 bundler（Alto）与 EVM 客户端（Anvil）分别在容器与原生模式下运行，统计平均 CPU 与 DRAM 功耗，并与空闲基准对比；发现最高平均功耗约 25 W，容器与原生差异低于 1 W；不同硬件间差异主要在 CPU 频率与内存带宽。

**⚠️ 局限性**

仅评估了单一 bundler 实现（Alto）与单一 EVM 模拟器（Anvil）；工作负载仅为 ERC-20 转账，未覆盖更复杂的交易与支付赞助场景；实验在单机环境下完成，未考虑网络延迟、节点分布和动态负载波动等实际运营条件。

---

## 99. ROCm InferaSim: Cut LLM Serving Trial Costs and Deploy the Best Configuration

**arXiv ID:** 2611.04032 | [PDF](https://arxiv.org/pdf/2611.04032v1)

**作者:** Anshu Raina `[一作]`, Emad Barsoum `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ROCm InferaSim框架，利用少量anchor测量即可预测LLM推理性能并引导高效配置搜索。

**💡 创新点**

创新点在于将完整推理step作为anchor，并通过解析转移规律保持固定步骤成本、只缩放可分摊工作，从而实现跨节点、预填充/解码分离及KV缓存分层等场景的高精度预测。

**🔧 技术方法**

结合硬件测量、解析转移、离散事件仿真、自动调优Agent与混合规划器，并在AMD MI355X GPU上实现。

**📊 数据集**

使用多种LLM模型（DeepSeek‑V4‑Pro、GLM‑5.2、Kimi‑K3、MiniMax‑M3、Qwen3.5等）以及vLLM、SGLang、ATOM引擎，配合AgentX编码代理流和InferenceX固定长度流的真实请求数据。

**📈 对比分析**

与手工试验及Vidur等基准相比，anchor模式在TPOT、吞吐量等指标上误差均≤8%，agentic吞吐量误差≈8%，排名准确率超过90%，并且无需在每个配置上耗时试验。

**⚠️ 局限性**

仅在AMD Instinct MI355X GPU上验证，缺少对NVIDIA等其他硬件的评估与迁移验证。

---

## 100. qLevelPlace: A Level-Based Placement Framework for RSFQ Circuits

**arXiv ID:** 2611.04019 | [PDF](https://arxiv.org/pdf/2611.04019v1)

**作者:** Ziyu Liu `[一作]` (University of Southern California), Massoud Pedram `[通讯]` (University of Southern California)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了面向RSFQ电路的层级放置框架qLevelPlace，将逻辑门和分裂器的放置分为两阶段，并采用时序感知的二次目标函数

**💡 创新点**

创新点在于：①将门级放置与非时钟分裂器放置解耦，聚焦门间关键连线；②灵活的层-行映射允许同一逻辑层跨多行；③通过时序加权的边权实现对关键路径的主动抑制

**🔧 技术方法**

采用整数线性规划（ILP）实现层级行分配，梯度下降/共轭梯度求解二次放置目标，后续使用Tetris式合法化；对分裂器采用垂直挂接策略；利用Congestion-aware Y轴间距和基于DME的时钟树合成

**📊 数据集**

在ISCAS‑85基准集上加上32位ALU、乘法器、蝶形单元共14个电路进行评估

**📈 对比分析**

与qPlace对比，qLevelPlace平均提升频率2.65×，面积下降51%，HPWL略降，且所有电路均能通过路由；ablation实验验证二次目标、层级折叠、时序加权的有效性

**⚠️ 局限性**

局限在于：①仅针对单相时钟的FPB RSFQ电路，未覆盖多相或非FPB设计；②分裂器布局仍采用简单垂直插入，复杂树结构尚未优化；③基准规模相对较小，尚未验证极大规模芯片的可扩展性

---

## 101. MusiSPARQ: Subscription-Aware Adaptive Resource Control for Musical Metaverse Services

**arXiv ID:** 2611.04011 | [PDF](https://arxiv.org/pdf/2611.04011v1)

**作者:** Ali Al Housseini `[一作]` (University of Applied Sciences and Arts of Southern Switzerland), Omran Ayoub `[通讯]` (University of Applied Sciences and Arts of Southern Switzerland)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实现了一种名为 MusiSPARQ 的动态资源隔离控制框架，用以在音乐元宇宙服务中根据订阅级别和实时负载自动在共享与保证模式间切换。

**💡 创新点**

创新点在于将原本固定的共享/保证执行模式转化为运行时决策，通过在同一物理资源上构建共享与隔离池，并在每个调度周期根据延迟目标、隔离成本和切换开销联合优化，从而实现“按需隔离、按需归还”的中间状态。

**🔧 技术方法**

核心技术包括：SPARQ 的队列感知调度框架、基于 M/M/1 与 M/G/1 的延迟模型、整数规划的每周期隔离决策、订阅级别权重和最大延迟约束，配合仿真器实现实时评估。

**📊 数据集**

使用自研的虚拟演唱会仿真负载，覆盖观众规模 0.4 N_sat–N_sat、不同订阅比例以及多种负载阶段（入场、高潮、离场）等场景。

**📈 对比分析**

与全共享、固定订阅隔离、最大可行隔离等基线对比，MusiSPARQ 在 Premium 级别的目标延迟违例率从 16.19% 降至 0.29%，而平均仅预留 0.71% 的资源，标准和基础流的延迟目标也能保持满足；在高负载时仍保持 1% 以下违例率，且不出现不可行时刻。

**⚠️ 局限性**

局限性包括：需预先完成服务图嵌入和路由；离散化的整数规划在大规模部署下求解复杂度高；对实时负载预测缺乏机制；仅针对单一资源类型的隔离切换，未考虑多层次多租户场景。

---

## 102. Characterizing Adversarial GEO via Implicit Brand Injection

**arXiv ID:** 2611.04005 | [PDF](https://arxiv.org/pdf/2611.04005v1)

**作者:** Shobhnik Kriplani `[一作]` (Indian Institute of Science), Haohan Wang `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个多代理框架IBI，用于在单句内隐式插入品牌，并评估其在检索增强生成系统中的检索曝光和品牌传播效果。

**💡 创新点**

创新点在于结合位置规划、句子重写、验证与候选选择的流水线，使得品牌注入既保持语义一致、自然度高，又避免显式广告，并在检索与生成中实现显著传播。

**🔧 技术方法**

使用LLM（如GPT-4o-mini、GPT-5-mini等）进行重写与验证，密集检索（e5-base-v2 + FAISS）、RAG管线，以及Hit@k、MRR、BMR、CP等评估指标。

**📊 数据集**

使用IBIBench数据集，包含1000篇信息文章、12个领域、60个合成品牌，构建2000个文章-品牌对（内生与非内生）。

**📈 对比分析**

与单次提示DP和等效调用次数的Best-of-N基线比较，IBI在所有检索/生成设置下的Hit@10、MRR、BMR、CP等指标均提升6–20%，最高可达82.2% Hit@10和75.9% CP。

**⚠️ 局限性**

局限性包括仅使用合成品牌、单句插入、未模拟真实爬取/索引过程、仅稠密检索、缺乏多语言或真实用户查询、以及人类评测样本规模有限。

---

## 103. WASP-Mem: Let Agent Organize Procedural Memory. Human Spend Effort on Diagnosis

**arXiv ID:** 2611.04003 | [PDF](https://arxiv.org/pdf/2611.04003v1)

**作者:** Peizhi Niu `[一作]` (University of Illinois Urbana Champaign), Kevin Chen `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于wiki的LLM代理记忆系统WASP-Mem，用于将执行轨迹和反馈压缩成可重用的流程wiki页面，实现训练无关的持续学习；

**💡 创新点**

创新点在于（1）将程序记忆转化为可链接的wiki页面，自动进行隐式去重和整合；（2）证明预先约束的结构（schema、显式去重）不如代理自组织有效；（3）发现诊断性反馈比多轮自评更能提升性能；

**🔧 技术方法**

技术包括：LLM驱动的wiki文件系统、语义搜索+shell导航、prompt-based lint检查、Free‑Form页面生成与隐式去重、与Schema Form、Schema+ED等对比；

**📊 数据集**

数据集为两大企业流程基准TheAgentCompany（127任务、542检查点）和τ²‑Bench Retail（74训练、40测试任务）；

**📈 对比分析**

与No Memory、No Consolidation、Insights、SkillX等基线比较，Free‑Form WASP‑Mem在所有三种模型（Claude Opus 4.6、GPT‑5.4、GLM‑5）上平均提升检查点通过率10.9–14.5个百分点，且存储量比原始轨迹小94–97%，且单轮诊断超越四轮自评7.0–10.9个百分点；

**⚠️ 局限性**

局限在于仅用两套基准、三种模型、单次实验，缺乏随机种子评估；对更广泛企业环境、长期记忆稳定性、自动诊断等方面仍需进一步研究。

---

## 104. Sampling Interventional Distributions with Orthogonal MMD Flow

**arXiv ID:** 2611.03962 | [PDF](https://arxiv.org/pdf/2611.03962v1)

**作者:** Bariscan Bozkurt `[一作]` (University College London), Houssam Zenati `[通讯]` (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于核均值嵌入的最大均值散度（MMD）梯度流的粒子采样方法（IMMD‑Flow），能够直接从观测数据估计的因果干预分布生成样本；

**💡 创新点**

创新点包括：利用双稳健正交核嵌入消除第一阶偏差、无需训练生成器即可进行粒子流演化、采用投影Nyström实现跨干预重用，从而显著降低计算成本并提升采样精度；

**🔧 技术方法**

核心技术包括核均值嵌入、最大均值散度梯度流、双稳健正交估计、投影Nyström近似、粒子优化和加噪梯度主导分析；

**📊 数据集**

实验使用的图像数据集有ColorMNIST、CelebA和dSprites，用于评估因果去偏图像生成；

**📈 对比分析**

与DecFM（流匹配）等基准在SW₂、Latent SW₂和FID指标上进行比较，IMMD‑Flow在像素空间SW₂上优于DecFM，Fidelity相当，同时总训练+采样时间缩短11–42倍；

**⚠️ 局限性**

局限性在于依赖观测混杂与重叠假设、对非单射编码/解码器可能不收敛，以及对样本量和干预复杂度的敏感性。

---

## 105. Do Pretrained Backbones Transfer to Physical Sensors? A Systematic Study of Multimodal Initialization Across the Sensor Stack

**arXiv ID:** 2611.04034 | [PDF](https://arxiv.org/pdf/2611.04034v1)

**作者:** Fabio Montello `[一作]` (Technical University of Denmark), Lazaros Nalpantidis `[通讯]` (Technical University of Denmark)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

我们构建了包含 2.82M 张图片、6 种硬件和模型派生传感器（RGB、LWIR、NIR、Event、Stereo Depth、Segmentation）的多传感器自动驾驶数据集，并在此数据集上对 MultiMAE 进行领域内预训练；随后将得到的权重与 ViT、MAE、DINO 等常见预训练模型进行对比，评估在语义分割与目标检测任务上的迁移性能。

**💡 创新点**

创新点包括：①首次将多传感器物理数据与模型生成数据结合，构建大规模稀疏多模态数据集；②改进 MultiMAE 以适应稀疏传感器（使用固定 token 数量、样本/模态平衡策略、编码器瓶颈聚合、SSIM 与 FFT 细节重建损失）；③在领域内预训练后观察到在非 RGB 模态下可获得显著但渐进的性能提升，并实现更快收敛。

**🔧 技术方法**

核心技术为掩码自编码（MAE）与 MultiMAE 架构，配合自定义 tokenizer、共享 ViT 编码器、跨模态解码器、编码器瓶颈聚合、SSIM 与 FFT 细节损失，以及三种模态平衡策略。

**📊 数据集**

使用了 16 个公开自动驾驶数据集（如 BDD100k、Cityscapes、DSEC、ECMD、FMB、FR‑T、KAIST‑MS‑Pedestrian、LLVIP、MFNet、MS2、NSAVP、NTU4DRadLM、RASMD、SMOD、STheReO、Vivid++），共计 2.82M 张图像，涵盖 6 种传感器模态。

**📈 对比分析**

我们在 SeeThroughFog（STF）与 DSEC 两个下游任务上进行对比实验，评估语义分割（mIoU）与目标检测（mAP）指标。实验显示：MultiMAE 作为 off‑the‑shelf 基线表现最佳；在包含非 RGB 模态的场景中，领域内预训练（Our‑MM）在 MultiMAE 基础上进一步提升，且收敛速度更快；相对 ViT、MAE、DINO 等传统预训练权重，改进模型均取得 1–5% 的性能提升。

**⚠️ 局限性**

局限性包括：①数据集仍为公开数据，受许可限制无法公开完整预训练语料；②稀疏模态导致训练时需额外平衡策略，仍可能影响极少见模态的学习；③提升幅度相对有限，仅在非 RGB 模态时显著；④评估仅覆盖语义分割与目标检测，未涵盖其它多模态任务；⑤模型改动对不同硬件平台的可部署性尚待验证。

---

## 106. SurgEye: Instruction Tuning-Free Perception-Reasoning Framework for Surgical Scene Understanding

**arXiv ID:** 2611.04012 | [PDF](https://arxiv.org/pdf/2611.04012v1)

**作者:** Qing Xu `[一作]` (University of Nottingham Ningbo China), Zhen Chen `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SurgEye 框架，利用视觉基础模型作为冻结 LLM 的视觉输入，实现无需指令调优的外科场景推理。

**💡 创新点**

创新点在于通过 Routed Multi‑Task Perception 生成多任务结构化证据，并用 Agentic Compositional Reasoning 将冻结 LLM 与可调用工具及结构化时序记忆相结合，实现跨任务跨时序的组合推理。

**🔧 技术方法**

使用冻结的 DINOv3 ViT‑L 视觉基础模型配合 LoRA 与解码器构建 RMP；采用冻结的 Qwen3‑30B‑A3B 大语言模型作为代理，并实现可调用工具接口与结构化时序记忆。

**📊 数据集**

使用九个公开外科数据集的图像/视频与对应的分割、检测、动作三元组、关键事件、阶段等标签，构建 SurgEye‑Bench。

**📈 对比分析**

在 SurgEye‑Bench 上与多种 VLM 与 VFM 进行对比，SurgEye 在 MSR、BER、SQR、PTR 四个维度的准确率分别达到 76.05%、76.78%、56.21%、78.11%，明显优于现有方法，尤其时序推理提升 32%。

**⚠️ 局限性**

局限在于仍需依赖预先的多任务标签训练 RMP，对极端光照或复杂切除场景的鲁棒性未充分验证，未来需探索更通用的视觉编码器与零样本跨域适应。

---

## 107. A Guideline-Grounded Multi-Agent Framework for Personalized Exercise Prescription from Clinical and Wearable Data

**arXiv ID:** 2611.03983 | [PDF](https://arxiv.org/pdf/2611.03983v1)

**作者:** Xintong Li `[一作]` (Emory University), Runze Yan `[通讯]` (Emory University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了 FITTER 框架，该框架通过多代理顺序生成、检索证据和显式安全约束，生成个性化的 FITT‑VP 运动处方并通过确定性审计修正保证安全合规。

**💡 创新点**

创新点在于：① 将处方生成拆解为六个互相关联的 LLM 代理任务，显式建模各维度间的依赖；② 将指南检索、药物安全知识库和模型安全审查三类约束整合到处方生成流程；③ 引入确定性安全审计与修复循环，显著提升安全合规率。

**🔧 技术方法**

使用技术包括：Qwen3.5‑9B LLM、检索增强生成（RAG）、规则基安全约束映射、药物知识库（FDA Structured Product Labels）、确定性审计修复逻辑和多代理协同生成。

**📊 数据集**

数据集为 All of Us Research Program 的 400 名患者（300 名癌症患者、100 名中风患者）电子健康记录、可穿戴活动数据、问卷信息等，构成结构化患者档案。

**📈 对比分析**

通过与五种 LLM 对照（Vanilla、CoT、Self‑Refine、ReAct、Collab）在安全合规率、指令具体性、指南一致性等计算指标进行统计比较，FITTER 的安全合规率达 98.5%（高于 75.8–85.8%），指令具体性与指南一致性均优于对照；临床评估中 88% 的处方被接受或仅需微调。

**⚠️ 局限性**

局限性包括：① 仅使用单一 Qwen3.5‑9B 模型，无法评估更大或商业 LLM 的表现；② 评估基于已链接可穿戴数据的样本，缺乏对无设备人群的适用性；③ 缺少真实临床试验验证，无法确认实际遵从性与健康效益；④ 部署受限于 All of Us 研究工作台，无法在外部系统直接复现。

---

## 108. Rapid Mean-Field Formation Transport with a Shared-Representation Neural Fokker-Planck Controller

**arXiv ID:** 2611.03981 | [PDF](https://arxiv.org/pdf/2611.03981v1)

**作者:** Zejian Zhou `[一作]` (University of Wyoming), Hao Xu `[通讯]` (University of Nevada)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出共享表示神经Fokker–Planck控制器，实现大规模多智能体集群的在线形成调度。

**💡 创新点**

通过共享特征网络联合优化密度残差与粒子匹配误差，兼顾物理一致性与目标实现，显著降低参数量与计算成本。

**🔧 技术方法**

使用物理信息化神经网络（PINN）、软加法归一化、MMD粒子匹配、两头共享网络和残差正则化技术。

**📊 数据集**

采用1D Gaussian transport基准与2D五种几何目标（线、环、分裂、过渡、三角）仿真数据，随机生成初始粒子分布。

**📈 对比分析**

与潜能场、共识反馈、均匀采样规划等基线对比，终端Wasserstein距离最低，能量与运行时间也优于多数基线，尤其在终端误差和计算效率上表现突出。

**⚠️ 局限性**

局限于周期域解析，边界处理有限；对更高维目标、障碍环境及更大粒子数的推广需要进一步研究。

---

## 109. One Mistake, Three Layers: Persistence Rules Under Depth Correlation

**arXiv ID:** 2611.03968 | [PDF](https://arxiv.org/pdf/2611.03968v1)

**作者:** Maryam Rahimimovassagh `[一作]` (University of Central Florida), Niloofar Yousefi `[通讯]` (University of Central Florida)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究了冻结式共折叠网络（co‑folding trunk）在不同熟悉度（蛋白族群与口袋结构）的深度层次上，能否解码配体结合口袋信息，并提出一种基于置换校准、五层连续阈值的“深度曲线”读取规则；

**💡 创新点**

创新点在于（1）首次将神经影像学中用于多层相关性的校准方法应用于Transformer深度曲线，避免因层间相关导致的早期错误起始点；（2）对口袋信息的可解码性进行层级分解，揭示最难的情形是熟悉蛋白族群但结构新颖口袋；

**🔧 技术方法**

技术手段包括：冻结的Boltz‑1共折叠 trunk、Pairformer块的逐层探测、线性探针（AUPRC评估）、置换校准的阈值、五层连续穿越阈值的起始判定，以及累计深度（90%增益）统计；

**📊 数据集**

使用的主要数据集为PLINDER 2026‑09提供的1,820个后截止系统，筛选出25,289个带口袋的蛋白，交叉验证共提取48层Pairformer块的表征；

**📈 对比分析**

与基准方法（随机基线、已训练的PocketWatch、P2Rank等）比较，发现在所有四个熟悉度分区内，口袋可解码性随层数提升，90%增益层数均在14–27层之间；最难区（熟悉族群+结构新颖口袋）在所有特征集上AUPRC均最低，且在累积深度上最迟；

**⚠️ 局限性**

局限性包括：起始点在所有分区均为0，因口袋信息已在首层可解码，无法区分分区；仅考察单一冻结trunk（Boltz‑1），未检验其他共折叠模型；使用线性探针限制了可测量的可解码信息量；最终的累积深度统计缺乏置信区间，难以评估其显著性；

---

## 110. Ermakov--Based embeddings as a parameter-free layer for physics-informed neural networks

**arXiv ID:** 2611.03964 | [PDF](https://arxiv.org/pdf/2611.03964v1)

**作者:** Roberto Bernal-Jaquez `[一作]` (Universidad Autonomica Metropolitana--Cuajimalpa), Alexander Schaum `[通讯]` (University of Hohenheim)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出将 Ermakov‑Pinney 缩放函数 ρ 作为无参数嵌入到 PINN 输入层，解决三类时间可变系数线性 PDE 的多尺度问题。

**💡 创新点**

创新在于给出闭式时刻身份使 ρ 可直接由初始数据求得，并通过比较分析三类问题中各嵌入组件（坐标、幅度、相位）的重要性。

**🔧 技术方法**

使用物理信息神经网络（PINN）、Ermakov‑Pinney 方程、尺度变量变换、损失加权以及 ablation 对比实验。

**📊 数据集**

使用合成的 PDE 解决方案与多种时间依赖系数调度（如 D(t)、γ(t)、ω(t)）生成的数据，不依赖外部真实数据集。

**📈 对比分析**

通过与标准 PINN 在相同网络结构、训练预算下的误差（L²、即时误差）对比，发现家族 II 的幅度预因子提升约 40.8 倍、家族 III 的二次相位提升约 5.7 倍；家族 I 由于时变重参数化效果有限，提升有限。

**⚠️ 局限性**

局限性包括仅适用于线性守恒的一阶相似性问题，不能处理非线性或二阶相似性、外推条件不满足时失效，且在 1D 空间中绝对精度不及谱方法。

---

## 111. Forget What Comes Next: LLM Unlearning via Statistical Independence

**arXiv ID:** 2611.03955 | [PDF](https://arxiv.org/pdf/2611.03955v1)

**作者:** Andrii Shkabrii `[一作]` (University of Vienna), Martin Pawelczyk `[通讯]` (University of Vienna)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于统计独立性的LLM机器忘记方法PINT，通过最小化前缀表示与下一个token的Hilbert-Schmidt独立性（HSIC）来实现数据遗忘。

**💡 创新点**

创新点在于首次将HSIC作为训练目标，并证明其可上界Exact Memorization，且不需要像RMU那样把表示压缩到单点。

**🔧 技术方法**

采用HSIC正则化、最终层表示约束以及梯度下降实现，利用Gaussian/Delta核捕捉非线性依赖。

**📊 数据集**

在八个不同的基准上评估，涉及个人信息（TOFU、bioS）、受版权保护文本（MUSE）、危险知识（WMDP）、以及Gaussian毒素，使用Llama-3.2-1B、Llama-2-7B、ICLM-7B、OlaMo-2 2.7B以及70B Llama-3.2。

**📈 对比分析**

与GradAscent、GradDiff、NPO、SimNPO、UNDIAL、RMU等六种基线相比，PINT在大多数基准上获得最佳或接近最佳的遗忘-效用权衡，尤其在Gaussian毒素上实现了可检测性为0且保持91%效用。

**⚠️ 局限性**

局限性包括仅验证文本模型，尚未扩展到多模态或安全对齐场景，且评估仍受现有基准指标和可提取信息测试的限制。

---

## 112. We Ran a Research Sprint With AI. The AI Worked. The Research Didn't

**arXiv ID:** 2611.03946 | [PDF](https://arxiv.org/pdf/2611.03946v1)

**作者:** Anush Sankaran `[一作]` (Microsoft), Arijit Banerjee `[通讯]` (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

进行了一个为期100分钟的实验，邀请12名研究人员在3个团队中使用Claude、Claude Code、ChatGPT、Copilot等生成式AI工具，并记录AI交互与团队评估，以探讨团队层面的AI使用机制。

**💡 创新点**

首次提出“Human‑AI‑Human”交互拓扑，揭示个人化AI工具在时间压力下可能导致的消费瓶颈和研究孤岛，并给出七个可检验的设计假设。

**🔧 技术方法**

使用了多款大型语言模型（Claude、ChatGPT、Copilot）、观察员日志记录系统、短阶段微测、闭幕式对话回顾以及并行的AI‑AI流水线作为对照。

**📊 数据集**

未使用公开数据集，实验基于12名参与者的日志（共33次AI交互）和实验生成的论文草稿。

**📈 对比分析**

该研究未进行定量性能比较，也未给出质量评估指标，因而无可量化的性能结果。

**⚠️ 局限性**

仅为一次小规模、单机构、受限时间压力的诊断性实验，缺乏对照组、质量评分和长期追踪，结果难以归因或推广。

---

## 113. Controlling Action-Head Co-Adaptation to VLM Representation Anisotropy

**arXiv ID:** 2611.04006 | [PDF](https://arxiv.org/pdf/2611.04006v1)

**作者:** Ruijie Zhang `[一作]` (University of California), Zheng Zhang `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究预训练视觉语言模型(VLM)与下游动作头之间的接口适配问题，并提出 AdaIE 机制在训练过程中动态重权重预训练表示，以缓解动作头对高方差维度的过度敏感。

**💡 创新点**

提出了无参数、在线自适应的接口重权重方法 AdaIE，首次将下游模型的梯度敏感度反馈回到预训练表示层，证明在表示与任务不匹配时可显著提升学习效果。

**🔧 技术方法**

采用梯度敏感度估计（Hutchinson 随机探针）与指数移动平均，对 VLM 输出进行按通道的重标定，并使用 norm‑matching 保持整体尺度；在多种 VLA、扩散策略和世界模型架构中实现。

**📊 数据集**

在 RoboCasa‑GR1、RoboTwin‑2.0、IRON‑R01‑1.11 实际机器人、LeWM 以及 DINO‑WM 等数据集上进行评估，覆盖桌面、双臂、真实抓取、导航与操纵任务。

**📈 对比分析**

与原始 VLA/扩散/世界模型基线相比，AdaIE 在 RoboCasa‑GR1 上提升 5.28pp，在 RoboTwin‑2.0 上提升 1.68pp，平均在 5 项真实任务中提升 19pp；在 LeWM 和 DINO‑WM 的规划成功率上也能实现 1–4pp 的提升。

**⚠️ 局限性**

局限性包括：需要手动设置重权重强度 α，需针对不同任务调优；在表示与任务高度匹配的情形下提升有限；重权重仅针对通道级别，未考虑更细粒度的空间/时间信息。

---

## 114. When Music Needs Meaning: Semantic Communication as a Transformative Enabler for the Musical Metaverse

**arXiv ID:** 2611.03995 | [PDF](https://arxiv.org/pdf/2611.03995v1)

**作者:** Ali Al Housseini `[一作]` (University of Applied Sciences and Arts of Southern Switzerland), Omran Ayoub `[通讯]` (University of Applied Sciences and Arts of Southern Switzerland)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了音乐元宇宙（Musical Metaverse, MM）的定义及其多模态、实时、分布式交互需求，基于语义通信构建了三大支柱框架（语义提取、语义感知编排、跨模态语义同步），并给出了对应的参考架构、语义词汇表、诊断指标与研究路线图。

**💡 创新点**

创新点在于：1）首次将语义通信方法与音乐专属时间参考和功能优先级相结合，形成针对MM的“音乐语义”概念；2）提出了三柱框架及事件接口，统一了不同学科（NMP、神经音频编解码、边缘AI编排、跨模态同步）技术的协同；3）设计了可测量的语义保真度(SMF)、跨端同步指数(ESI)及情感一致性指标(ECS)，为后续实验提供量化手段。

**🔧 技术方法**

采用的技术主要包括：神经音频编码（如EnCodec、MuCodec）、音频/运动/视觉特征提取模型、边缘AI资源调度与编排、OSC/Ableton Link等时序同步协议、以及基于事件的语义编码与同步服务。

**📊 数据集**

论文未使用公开数据集；主要基于已有研究中的方法与标准（例如音频质量评估标准POLQA/PEAQ）来构造诊断指标，并通过示例演示构架，未进行数据驱动实验。

**📈 对比分析**

对比方式：目前未进行实测对比，作者仅提出理论上的基线对比思路（与传统音频编码+优先级调度、事件协议等对比），并指出需在相同拓扑、带宽、延迟、计算预算下评估语义通信的增益。

**⚠️ 局限性**

限制：1）缺乏真实环境实验验证，所有方案均为理论/示例；2）不同技术成熟度不均衡，语义提取与同步接口尚未标准化；3）诊断指标尚未通过人类主观评估验证；4）在大规模多用户场景下的网络与计算资源需求仍需深入研究。

---

## 115. FlagRot: Selective Rotation for Low-Bit KV Caches

**arXiv ID:** 2611.03992 | [PDF](https://arxiv.org/pdf/2611.03992v1)

**作者:** Kunming Shao `[一作]` (Hong Kong University of Science and Technology), Chi Ying Tsui `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对长上下文语言模型的 KV 缓存做低比特量化，并引入按 32 通道块选择是否旋转的机制。

**💡 创新点**

创新点是把旋转视为编码器的决策，依据注意力查询加权的误差进行挑选，并将决策编码在已有的符号位中；支持固定模式和每键动态选择两种模式。

**🔧 技术方法**

使用 Hadamard 旋转、对齐量化（KIVI）、校准窗口统计、FP8/FP16 缩放与最小值、vLLM 插件以及 CUDA 图形加速。

**📊 数据集**

主要数据集为 WikiText‑2 进行校准与评估，进一步在 LongBench、RULER、AIME、τ²‑bench、SWE‑bench 等多任务数据集上验证任务性能。

**📈 对比分析**

通过与 BF16、原始 KIVI、RotateKV 等基线对比，使用 KL 散度、LongBench 及 RULER 分数衡量；在 2.5 bits/值时，FlagRot‑KIVI 使 KL 降低至 0.12×10⁻³，LongBench 均值提升 0.4–0.6 分，RULER 128k 提升 1–2 分，且对 32k 长上下文的推理吞吐量提升 3–4 倍。

**⚠️ 局限性**

局限性：实验仅覆盖 8B–32B 参数模型、NVIDIA H20/H200 GPU；未验证 Mixture‑of‑Experts 或 32B 以上模型；校准和决策主要基于 WikiText‑2，其他文本类型的迁移效果有限；并且 per‑key 选择会在极大批量下略微降低预填充吞吐量。

---

## 116. SLATE: A Large-Scale Benchmark for Long-Horizon API Selection in E-commerce Tool Use

**arXiv ID:** 2611.03982 | [PDF](https://arxiv.org/pdf/2611.03982v1)

**作者:** Rongzhe Wei `[一作]` (Amazon), Na Zhang `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `a2602d71-93ab-4bad-974b-672788df8193` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个名为SLATE的、面向电商的，包含约1000个API的大规模工具使用基准，用于评估LLM在长周期、多步骤工具调用中的选择能力。

**💡 创新点**

创新点在于：①同时满足大工具空间、长序列执行、确定性工具输出、端到端任务成功评估四大属性；②采用确定性上下文感知模拟器提供可复现的工具输出；③用多种评估指标（任务成功率、步骤准确率）而非主观LLM评判。

**🔧 技术方法**

使用的技术包括：多阶段数据生成流水线（查询生成、层级计划生成、工具定义、工具映射、上下文模拟器）；决策模型（fine‑tuned for MCQ）；与多种LLM后端（3B‑12B参数）配套；以及工具库匹配和默认响应机制。

**📊 数据集**

数据集为SLATE，其包含100个实例、约1000个API（417相关，583干扰），每个实例有10‑14步的层级计划，涵盖11个电商任务类别。

**📈 对比分析**

对比方法：使用六种公开决策模型与对应基础LLM，评估任务成功率和步骤准确率；结果显示最好的决策模型任务成功率仅约7%，步骤准确率在51%-64%之间，明显低于理想水平，表明仍有较大提升空间。

**⚠️ 局限性**

限制：仍依赖人工审核的生成流程；工具库和模拟器基于电商场景，迁移到其他领域需要重新构建；决策模型训练并非统一提升，可能受模型家族差异影响；缺少对不同规模模型的广泛评估。

---

## 117. Measuring Model-Specific Alignment Drift in LLM-Based Opinion Dynamics

**arXiv ID:** 2611.03980 | [PDF](https://arxiv.org/pdf/2611.03980v1)

**作者:** Nataša Djurdjevac Conrad `[一作]` (Zuse Institute Berlin), Christof Schütte `[通讯]` (Zuse Institute Berlin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文构建了一个统一的实验框架，利用固定的六项Likert量表对大语言模型（LLM）代理人在多轮讨论中的立场进行理论无关测量，并在相同协议下对六种LLM进行对比，评估其在意见动态中的对齐漂移、同质性关注与集体极化等行为；

**💡 创新点**

创新点在于：①提出了与任何动态模型无关的测量方法，避免了测量偏倚；②在匹配实验条件下跨LLM进行模型选择，揭示LLM自身属性决定了讨论结果的稳态；③通过多层次（无讨论、强制承诺、身份标签、同质性关注）验证行为特征的稳健性；

**🔧 技术方法**

技术包括：固定六项Likert量表计算立场坐标；一阶最小二乘拟合经典意见动态模型（持久性、DeGroot、Friedkin–Johnsen、有限信心、共同漂移与极化吸引子）；BIC模型选择；统计量如群体方差、同质性指数、对齐漂移等；

**📊 数据集**

数据集主要由六个LLM模型（每个模型10个代理，T=20轮，R=5复现实验）在两条政策议题（电动汽车补贴与四天工作周）上生成的讨论文本与问卷答案组成；

**📈 对比分析**

比较方法：对每个LLM拟合一组经典模型并用BIC选取最佳模型，计算最终群体方差、同质性、身份下方差等指标；结果显示不同LLM在一致性、极化、漂移方面呈现显著差异，且模型选择与LLM特征高度相关；

**⚠️ 局限性**

局限性包括：仅测量单一轴度的立场，缺乏多维价值和情感维度；群体均质，仅用单一LLM；仅测试两条经济议题；样本规模有限（3-5复现）；使用英文提示和专有模型，难以复现；未能完全分离同伴曝光与讨论本身的影响。

---

## 118. Gather or Act? Belief-Level Counterfactual Labels for Deadline-Aware Incident Response in Cyber-Physical Systems

**arXiv ID:** 2611.03974 | [PDF](https://arxiv.org/pdf/2611.03974v1)

**作者:** Ali Mohammadi Ruzbahani `[一作]` `[通讯]` (University of Calgary), Ali Mohammadi Ruzbahani (University of Calgary)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文构建了一个基于 Tennessee Eastman 过程的故障驱动事故响应决策数据集，利用双重模拟（主世界与同一状态下的另一原因）生成“belief-level”反事实标签，研究在收集证据与直接行动之间的取舍；

**💡 创新点**

创新点在于提出了“belief-level”标签方法，利用同一状态下的另一原因的双重模拟与精确后验，减少回归噪声并纠正传统 hindsight 标签对信息收集价值的低估；

**🔧 技术方法**

技术上使用了共同随机数分支回测、Rao‑Blackwell 变换、后验推理、梯度提升树（GBM）与自注意力神经网络直接决策模型，并采用置信区间与 Holm 校正的聚类自举评估；

**📊 数据集**

数据集为 9,908 个决策点、约 217k 真实回放和 114k 双重回放，覆盖 1,990 个故障/扰动，分为自然与高风险（无备份通道）两种分布；

**📈 对比分析**

在高风险测试集上，GBM 训练于 belief 标签的平均 regret 比同样训练于 hindsight 标签低约 11%（主比较），在无备份通道的模糊决策中下降 27%，与固定 playbook 和行动优先规则相比，性能显著提升；

**⚠️ 局限性**

局限包括：数据仅为仿真，缺乏真实攻击场景与自适应威胁；belief 标签仅覆盖 link‑vs‑module 两种模糊情形；神经网络在中等规模数据上仍受限于特征提取，难以完全匹配树模型；

---

## 119. ProteoCraft: Protein Design as Proposal-Verification Search

**arXiv ID:** 2611.03963 | [PDF](https://arxiv.org/pdf/2611.03963v1)

**作者:** Fang Wu `[一作]` (Stanford University), Yejin Choi `[通讯]` (Stanford University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `09944146-298c-433e-89df-37255de463d7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种基于树搜索的蛋白互作设计框架ProteoCraft，利用扩散结构预测器的梯度作为提议方向，并通过完整的ESMFold2推断对候选序列进行验证，实现多站点探索与单站点细化的粗细分层搜索。

**💡 创新点**

创新点在于：①将扩散预测器的梯度仅用于生成可行变异，而非直接估值，解决梯度过度利用问题；②引入粗细分层的树搜索，先在多位点空间跨越不同序列基底，再在单位点层面细化；③将完整结构推断作为验证器而非训练目标，保证搜索与评估的分离。

**🔧 技术方法**

技术手段包括：ESMFold2扩散结构预测器、梯度提议（distogram、Masked LM、单步扩散梯度）与归一化融合；MCTS树搜索（UCB/PUCT）与可变位点掩码；完整推断验证（pair-chain iPTM、pLDDT等指标）；以及与AF2-IG、Rosetta等外部评估工具的交叉验证。

**📊 数据集**

数据集涵盖14个de novo绑定器设计目标（8种靶蛋白），5个TCR面对面绑定器目标，以及4个局部CDR/TCR重设计场景，共计210个候选序列；实验验证使用PD‑L1小分子和IFNAR2 VHH靶点进行SPR/BLI测定。

**📈 对比分析**

与RFdiffusion、BoltzGen、BindCraft、PXDesign等基线以及本工作匹配的计算量对照组进行比较。ProteoCraft‑MCTS在ESMFold2评估下的Pair‑chain iPTM、Binder pLDDT均高于对照组，且高置信度候选比例提升至54.3%；在AF2-IG和Rosetta评估中同样表现最优，实验验证显示PD‑L1绑定率82.9%，Kd 0.879 nM；IFNAR2绑定率44.4%，Kd≈150 nM。

**⚠️ 局限性**

局限性包括：①依赖ESMFold2预测的准确性，若预测误差或目标与模型偏离可能导致性能下降；②计算成本相对较高（约32 s/候选），尽管可通过推断优化加速；③目前未进行实验上游的亲和力成熟，仅验证初始设计；④缺乏针对更大规模、复杂多体或非靶向绑定任务的评估。

---

## 120. P$^2$-MOPD: Accelerating Multi-Teacher On-Policy Distillation via Parameter Priors

**arXiv ID:** 2611.03953 | [PDF](https://arxiv.org/pdf/2611.03953v1)

**作者:** Sinuo Wang `[一作]` (Adelaide University), Zuo Bai `[通讯]` (StepFun)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将学生模型与冻结的专家模型线性组合成可微模型，并在多教师on‑policy蒸馏过程中同时学习学生权重和混合系数，使专家参数在训练期间始终作为可学习的先验存在。

**💡 创新点**

创新点在于：①把专家参数视为可学习的参数先验而非仅作为初始化或一次性合并；②允许学生在训练中自由离开专家参数的线性子空间；③通过可微组合联合优化混合系数和学生权重，突破了任务算术仅停留在专家参数张量子空间的限制。

**🔧 技术方法**

使用技术包括：多教师on‑policy蒸馏（MOPD）、参数先验线性组合、任务算术、梯度切割、AdamW优化器、模块/层/模型级混合系数可微化。

**📊 数据集**

实验数据集为四个领域专家：医学（MedQA）、法律（CaseHOLD）、金融（FinQA）和指令跟随（IFBench），模型规模为1.7B和4B。

**📈 对比分析**

与基线MOPD（TA init）、IM‑MOPD、Open‑MOPD等方法对比，平均恢复率提升至94.9%/94.7%，最差域恢复率提升至约90%，并在训练步骤上比MOPD快约一半，在相同质量的墙钟时间上更短。

**⚠️ 局限性**

局限性包括：混合系数学习率对性能有一定影响，需要手动调参；实验仅覆盖四个领域，未验证在更多领域或不同任务上的泛化；仅在1.7B/4B规模下验证，尚未知晓更大模型或不同训练设置下的表现。

---

## 121. h-NESDF: An Infinitely Differentiable Signed Distance Between Two Sets with Robotic Applications

**arXiv ID:** 2611.03944 | [PDF](https://arxiv.org/pdf/2611.03944v1)

**作者:** Vinicius Mariano Gonçalves `[一作]` (Federal University of Minas Gerais), Pedro Antonio de Souza Campos Rodrigues `[通讯]` (Federal University of Minas Gerais)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种名为h-NESDF的无限可微签名距离函数，用于计算两个可能非凸集合之间的距离；

**💡 创新点**

创新点在于将非可微的NESDF通过指数平滑得到h-NESDF，并证明其收敛性与可微性；

**🔧 技术方法**

主要技术包括正则集合定义、半平方ESDF、积分权重平滑、GPU并行化采样逼近、Lie群梯度与Nesterov加速优化；

**📊 数据集**

实验使用TetWild点云数据集、随机生成的物体对，以及海底环境点云（54,570点）和BlueROV2模型；

**📈 对比分析**

与Beker等的平滑距离指标进行对比，h-NESDF在40,000个随机碰撞分离实验中成功率为91.03%，显著高于Beker的70.31%；在运动规划实验中，约34%的试验成功，平均耗时0.7s；

**⚠️ 局限性**

局限性包括：不保证距离正值（仅在无碰撞时可近似）；点云采样方法计算复杂度为O(K_AK_B)，对大点云或低延迟场景不友好；需要进一步研究更高效的空间数据结构与严格碰撞检测保证。

---

## 122. Revisiting Elliptic++ for Bitcoin Forensics: Feature Reconstruction and Leakage-Resistant Evaluation

**arXiv ID:** 2611.04000 | [PDF](https://arxiv.org/pdf/2611.04000v1)

**作者:** Runlin Liang `[一作]` (University of Amsterdam), Noud van Giersbergen `[通讯]` (University of Amsterdam)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

重新实现 Elliptic++ 数据集的 55 个地址级特征，完全恢复特征计算公式，并使用去匿名化映射和完整的 Bitcoin 账本数据构建了无信息泄漏的时间序列拆分，随后在该环境下重新评估随机森林和逻辑回归的检测性能。

**💡 创新点**

①提供了完整可复现的特征定义和计算代码；②首次量化了位置拆分与特征泄漏对模型性能的巨大影响；③证明在无泄漏评估下随机森林对比逻辑回归优势消失，凸显时间拆分和特征重算的必要性。

**🔧 技术方法**

利用 SQL（DuckDB）将去匿名化映射与 Kondor 2020 账本联接；Python 处理特征重算与 MinMax 标准化；使用逻辑回归和随机森林（50 棵树）进行分类；通过多种拆分策略和 10 个随机种子评估稳健性。

**📊 数据集**

Elliptic++（265,352 个标记地址，14,266 个恶意地址）+ Kondor 2020 Bitcoin 账本（约 2.55 亿交易）+ Kaggle 去匿名化映射（202,804 条交易）。

**📈 对比分析**

采用三种组合比较：原始位置 70/30 拆分 + 泄漏特征；时间拆分 + 泄漏特征；时间拆分 + 无泄漏特征。随机森林在泄漏条件下 F1 从 0.846 降至 0.291；在时间拆分后逻辑回归的召回率显著提升，随机森林不再显著优于逻辑回归。

**⚠️ 局限性**

①特征高度共线且冗余，未进行特征选择或降维；②仅使用默认模型超参数和 0.5 阈值，未进行调优；③实验仅基于 Elliptic++，对其他加密资产或更大规模数据集的泛化仍需验证。

---

## 123. Physical Experience Builds Continual Spatial World Models

**arXiv ID:** 2611.04016 | [PDF](https://arxiv.org/pdf/2611.04016v1)

**作者:** Yuhang Sun `[一作]` `[通讯]` (University of Washington), Yuhang Sun (University of Washington)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在物理驱动的迷宫环境中，提出并验证了一种跨回合的空间变化记忆 (SCM)，通过累积环境变化关系和动作物理反馈，提升了空间世界模型的预测与导航性能。

**💡 创新点**

将跨回合变化关系与物理动作模型结合，证明经验累积能转移到未见转场并在持续导航中提前预判；同时设立信息边界验证记忆的可迁移性。

**🔧 技术方法**

基于物理动力学的球滚动仿真、贝叶斯统计的变化记忆、状态-动作-结果的残差学习以及对比 Frozen/Local/Oracle/Reactive 等控制方案。

**📊 数据集**

使用自行构造的旋转圆环迷宫仿真数据，包含可变门状态和球动力学；不使用公开数据集。

**📈 对比分析**

通过与 Frozen/Local/Oracle/Reactive 对比，在决策改变与保持两类测试中，SCM 的预测准确率从 0.564 提升至 0.898，路标修正次数从 1.744 降至 0.406；在持续导航中，旧目标停留时间从 7.8 s 减至 0.008 s，完成时间平均提升 7.6 s。

**⚠️ 局限性**

实验受限于二进制门状态、低维物理模型、已知动力学方程及无真实感知误差，缺乏对复杂视觉输入、未知动力学或分布漂移的评估。

---

## 124. The Complexity of KKT Points of Quadratic Programs with Polynomially-Bounded Coefficients

**arXiv ID:** 2611.03945 | [PDF](https://arxiv.org/pdf/2611.03945v1)

**作者:** Argyrios Deligkas `[一作]` (Royal Holloway, University of London), Stavros D. Ioannidis `[通讯]` (Royal Holloway, University of London)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

证明了在系数用一元表示时，计算单位超立方体上的二次多项式KKT点仍为NP‑完全。

**💡 创新点**

首次在一元输入下展示KKT点问题保持NP‑完全性，构造了可模拟指数精度系数的多项式链小工具。

**🔧 技术方法**

利用系数标准化、dyadic分解和二次链小工具实现指数精度的辅助变量，再通过凸性和KKT等价性保证解映射。

**📊 数据集**

本研究为理论证明，未使用实验数据集。

**📈 对比分析**

与之前仅在二进制编码下证明的NP‑完整性相比，本结果表明即使系数为多项式大小，问题仍不可多项式时间求解，显示更强的困难度。

**⚠️ 局限性**

仅针对精确KKT点，未覆盖近似求解或常数系数情况，且构造在实际求解中可能导致变量数量指数增长。

---

## 125. Decision Transformer with Task-Aware Representations for Offline Meta-Reinforcement Learning

**arXiv ID:** 2611.04010 | [PDF](https://arxiv.org/pdf/2611.04010v1)

**作者:** Xinbo Zhao `[一作]` (State University of New York at Binghamton), Yingxue Zhang `[通讯]` (State University of New York at Binghamton)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于决策变换器的离线元强化学习框架CoTA‑DT，通过多视角任务表征学习实现可靠任务识别与自适应策略生成。

**💡 创新点**

创新点在于：①将对比学习、自然语言对齐、一步世界模型与长期Q学习四种监督信号融合为统一的任务表征学习；②利用冻结的LLM骨干与LoRA适配器实现参数共享，提升效率与泛化；③提出零样本语言初始化与自适应更新策略，兼顾零射击与少量样本场景。

**🔧 技术方法**

核心技术包括：对比学习（InfoNCE）、CLIP式语言对齐、基于LLM的任务条件世界模型与Q网络、Decision Transformer策略、LoRA参数适配、在线自适应更新与少量样本微调。

**📊 数据集**

使用四个离线元RL基准：Point‑Robot、Cheetah‑Vel、Ant‑Dir、Meta‑World，分别提供中等、专家与混合质量数据集，并配合任务自然语言描述。

**📈 对比分析**

与BC‑Z、BAKU、Prompt‑DT、Meta‑DT、UNICORN、T2DA‑T等六种基线对比，CoTA‑DT在所有基准的零样本和少样本情形下均获得最高平均回报，尤其在Meta‑World的零样本场景提升幅度显著（+189分）。

**⚠️ 局限性**

限制主要体现在：①对语言描述的依赖，若无高质量文本输入效果受限；②需要预训练LLM与大规模计算资源；③在极少量任务数据或极端噪声环境下，对比学习信号可能不足，导致任务表征不够稳健。

---

## 126. Physics-Informed Identification of Tensegrity Prestress across Taut-Slack Transitions

**arXiv ID:** 2611.04007 | [PDF](https://arxiv.org/pdf/2611.04007v1)

**作者:** Sandesh Lamsal `[一作]` (University of Miami), Landolf Rhode-Barbarigos `[通讯]` (University of Miami)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过物理信息神经网络实现了张力结构的逆问题求解，即从稀疏噪声运动记录中识别预张力、绳索力密度、阻尼和轴向刚度，并兼顾张绳的张力-无张力转换。

**💡 创新点**

创新点包括：①将力密度线性写入运动方程残差，形成闭式灵敏度；②使用软化（softplus）张绳法和事件感知样点逼近处理非光滑张力转折；③通过Fisher信息和Cramér–Rao分析先行评估可识别性；④在真实结构和测量数据上验证方法。

**🔧 技术方法**

采用技术包括物理信息神经网络（PINN）与多头网络、自动微分、软化张绳法、Adam优化、Fisher信息与Cramér–Rao界、以及与传统Levenberg–Marquardt射击法、无迹卡尔曼滤波和粒子滤波的对比。

**📊 数据集**

使用的数据集包括仿真生成的平面X1–X5框架、三维三叉枢轴单元，以及实验测得的仿生腿和全尺寸张力三角形结构的加速度记录和内置力传感器数据；还引用了公开的六边形张力框架频率测量数据。

**📈 对比分析**

与传统射击法、无迹卡尔曼滤波和粒子滤波对比时，本文方法在存在张绳松弛的记录上实现了<5% 的力密度误差、约1% 的阻尼误差，且对远初值具有鲁棒性；虽然计算时间约2–5k秒、需15个超参数，但在张绳保持张力的记录上优于射击法，而在无张绳记录上则略慢。

**⚠️ 局限性**

局限性包括：需要足够激励以破除可识别性的盲区；在低幅值张绳记录上可能出现“静默失败”；软化张绳法带来的偏差与优化步长限制；对模型形式误差（如张绳质量忽略）敏感；在无张绳实验记录中无法验证张绳转折处理。

---

## 127. Whose startup does the AI back? An algorithm audit of reputation and demographic signals in LLM-assisted crowdfunding

**arXiv ID:** 2611.03993 | [PDF](https://arxiv.org/pdf/2611.03993v1)

**作者:** Mirza Samad Ahmed Baig `[一作]` (Fandaqah), Syeda Anshrah Gillani `[通讯]` (Heidelberg University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究通过随机化的选择式共轭实验，对七个指令调优的LLM助手在推荐众筹项目时是否使用创始人的性别与族裔信息进行算法审计。

**💡 创新点**

创新点在于将用户目标视为边界条件，揭示LLM助手对少数族裔创始人的偏好来源于用户而非训练数据，从而实现对算法歧视的代理账户分析。

**🔧 技术方法**

使用了指令调优的LLM（如GPT‑4o‑mini、Llama3.2、Qwen2.5等）、线性概率模型（AMCE）以及条件logit估计，对决策进行量化分析。

**📊 数据集**

数据集为40,248条模型输出，包含五张合成众筹卡片，随机化的质量信号与创始人姓名（四性别×五族裔）等属性。

**📈 对比分析**

通过与现有文献中的人类后向基准对照，LLM默认下对性别无显著偏好、未再现黑人男性惩罚；而在“支持少数族裔”目标下显著逆转黑人惩罚，整体表现符合代理账户预测。

**⚠️ 局限性**

局限性包括：实验仅涉及单一技术类别、有限模型规模、姓名信号弱于照片、缺乏真实众筹数据，以及结果受决策集属性分布和模型规模影响。

---

## 128. Thinking Fast, Trading Late: System One Decision Models vs the Market

**arXiv ID:** 2611.03985 | [PDF](https://arxiv.org/pdf/2611.03985v1)

**作者:** Amir Dehkordi `[一作]`, Arman Khaledian `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 OLIVIA 盲测协议，使用语言模型决策模型（Laya‑Fin 以及三家商业模型）读取 S&P 500 新闻，检验其在两次验证期和一次盲测期内对每日回报（尤其是开盘后走势）的预测能力，并与多种基线模型进行比较。

**💡 创新点**

创新点：
- OLIVIA 六项防护（一次性解锁、无前瞻状态、公司身份掩码、验证期选择、冻结分析计划、审计日志）系统性消除前瞻偏差、模型记忆与多重检验问题。
- 对模型识别被掩码公司（matching‑distractor test）进行量化，揭示即使掩码也难以完全匿名。
- 将预注册与盲测结合，提供真实检验框架；对预期价格的阶梯式解释（explanation ladder）与信息系数（IC）共同衡量模型效能。

**🔧 技术方法**

技术：
- 语言模型决策模型基于 ModernBERT‑large，Laya‑Fin 经过单轮微调。
- 统计指标：Spearman 信息系数、IC、回归系数、R²、期望校准误（ECE）、Sharpe 比率（成本后折价）。
- 预注册哈希冻结、日志审计。
- 识别测试使用匹配基线（baseline accuracy）与公司识别率。
- 基线模型包括岭回归、梯度提升、生成式 LLM 以及基于标签的基线。

**📊 数据集**

数据集：
- NewsWitch 新闻流（标题+摘录，时间戳、来源、事件类型）。
- S&P 500 成分股名单（按市值分层）。
- 价格与成交量（收盘、开盘、日内、VIX、利率、信用利差、美元、油价）。
- 训练期 2024‑07‑01 至 2025‑06‑30；验证期两季度 2025‑07‑01–2025‑12‑31；盲测期 2026‑01‑02–2026‑07‑31；预留前瞻测试窗口 2026‑10‑22 起。

**📈 对比分析**

比较方法与性能：
- 先在验证期计算每个模型的 IC；选取 IC 最优的 Laya‑Fin 为主实验装置。
- 盲测期对同一模型进行 IC、回归和 Sharpe 比率测试；所有模型在盲测期均未显著提升，甚至出现显著下降。
- 商业模型仅在开盘价（overnight）上有显著 IC，但对开盘后走势（intraday）无显著预测。
- 预注册策略在成本折价后均亏损；即使采用最优阈值的“认知”策略，净 Sharpe 也低于随机。
- 识别测试显示商业模型识别被掩码公司的准确率远高于基线，但并未提升预测性能。

**⚠️ 局限性**

局限性：
- 只研究 S&P 500 大盘股，无法推广至小盘或非美股。
- 仅考虑日频与单一决策点（09:15 预读、10:00 复读、收盘后），不含高频或多时段决策。
- 微调仅一次、单一随机种子；未进行多种训练深度或迭代。
- 商业模型发布后，训练数据可能包含测试期间信息，导致记忆效应无法完全排除。
- 盲测期仅七个月，样本量有限。
- 掩码仍保留极少信息（姓名、日期等），无法完全消除公司可辨识度。
- 交易成本仅按固定点数计，未考虑借贷成本、冲击与滑点。
- 未对非文本信息（财报、宏观事件）进行综合评估。

---

## 129. Must AI Spell It Out? Pragmatic Comprehension of Alerts in a Time-Critical Task

**arXiv ID:** 2611.03976 | [PDF](https://arxiv.org/pdf/2611.03976v1)

**作者:** Anwesha Das `[一作]` (Saarland University), Anna Maria Feit `[通讯]` (Saarland University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究通过构建 DroneWorld 监测任务，比较了基于 RSA 的简约式助手（d‑RSA）与完全说明式助手的性能，评估人类在动态、时间敏感场景下对简约警报的理解与行动能力。

**💡 创新点**

创新点在于首次将简约式 (pragmatic) 说话者模型投射到实际人机合作的时间关键任务中，证明人类能完成机器留出的推理任务，并展示简约警报在低先验知识情境下能显著提升检测效果。

**🔧 技术方法**

主要技术包括 Rational Speech Act (RSA) 框架的动态扩展（d‑RSA）进行消息规划、实验室实验与可视化监测界面、以及基于 Bayesian 推理的监听者模型。

**📊 数据集**

使用自定义的 DroneWorld 数据集，共 20 个情景、24 个状态格，产生 7763 个关键事件，并在 114 名参与者上进行实验。

**📈 对比分析**

与完全说明式助手相比，d‑RSA 的检测率提升了 7.3 个百分点（85.8% 对 78.5%），检测速度更快、解决率更高（99.3% 对 97.9%），并且用户对其帮助性的满意度更高（92.9% 对 83.8%），同时没有增加认知负担或挫败感。

**⚠️ 局限性**

主要局限包括：实验中先验知识由实验设定，未验证真实用户的知识推断；对后续记忆（事后信念）的提升效果不明显；实验环境与真实应用的复杂性和多模态交互仍有限。

---

## 130. Self-governed access control in open robot swarms using a blockchain-based DAO

**arXiv ID:** 2611.03975 | [PDF](https://arxiv.org/pdf/2611.03975v1)

**作者:** Raina Zakir `[一作]` (Université Libre de Bruxelles), Volker Strobel `[通讯]` (Université Libre de Bruxelles)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并实现了一种基于 DAO 的去中心化治理框架，用于开放式机器人蜂群的成员资格管理和拜占庭机器人的检测、投票和移除，并在模拟与真实机器人环境中进行了实验验证。

**💡 创新点**

创新点在于将 DAO 的智能合约与区块链技术引入机器人蜂群治理，构建了访问控制层：通过可配置的阈值 α 对新加入机器人的投票资格进行“试用期”管理，保证在开放成员制度下仍能有效识别并移除拜占庭机器人，且不依赖中心化验证方。

**🔧 技术方法**

所使用的技术包括轻量级 PoA 区块链框架 Toychain、智能合约投票（提议‑投票‑执行）机制、协同与自适应阈值的拜占庭检测算法、ARGoS 仿真平台以及 Pi‑puck 机器人硬件；所有代码均开源发布。

**📊 数据集**

实验数据来自自定义的 3.55 m² 及 1 m² 现场实验环境，其中白色瓷砖比例设为 25%，共进行 10 次仿真跑（每跑 3600 计时步）及 1 次真实机器人演示；记录了机器人估计值、投票列表、区块链状态和移除记录。

**📈 对比分析**

实验对比了 DAO 方案与 Strobel 等人基线的离群过滤方法。结果显示：α=10 时真正例率提升、误报率下降；整体估计误差保持在 ±5% 以内；在三种实验场景（消除已有拜占庭、阻止新加入拜占庭占领、支持系统扩容）中，DAO 能保持较低的误报与较高的移除效率，且能够在动态成员变化下维持任务性能。

**⚠️ 局限性**

局限性包括：①智能合约安全性仍受限于代码和治理攻击风险；②检测算法依赖于协同与阈值设定，可能不适用于所有任务；③PoA 区块链的块时间与网络拓扑波动导致延迟；④实验规模受机器人数量和电池寿命限制，未验证极大规模蜂群的可扩展性；⑤对真实环境噪声与通信不可靠性的鲁棒性仍待进一步研究。

---

## 131. Do We Still Need Rotation-Invariant Architectures? A Statistical Robustness Benchmark for Point Cloud Classification

**arXiv ID:** 2611.03952 | [PDF](https://arxiv.org/pdf/2611.03952v1)

**作者:** Abdulla Nasir Chowdhury `[一作]` (University of Tsukuba), Kazuhiro Fukui `[通讯]` (University of Tsukuba)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文对10种点云分类网络在5个不同域（ModelNet40、ShapeNet、ABO、ScanObjectNN、S3DIS）下的姿态、噪声、遮挡扰动做了系统性鲁棒性基准，使用了统计显著性检验而非单点估计；并通过在所有模型上加一个简单的嵌入对齐损失，探究训练层面的旋转不变性与架构不变性对鲁棒性的影响。

**💡 创新点**

创新点在于：①提出基于显著性检验的鲁棒性基准方法，揭示“姿态-噪声”权衡；②通过统一的嵌入对齐损失展示训练层面可逆转的不变性成本，并发现“窄传递律”——鲁棒性提升仅局限于与训练扰动相同的损坏类型；③通过广泛的统计分析和可重复实验展示不同不变性设计（RI、VN）在真实和合成场景下的表现。

**🔧 技术方法**

使用了点云深度网络（PointNet、PointNet++、DGCNN、PointTransformer、PointMLP、RIConv、RIConv++、RISurConv、Vector Neurons等）以及对应的对抗损失；实现了三类扰动（姿态、噪声、遮挡）生成器；利用配对t检验、Wilcoxon、Friedman、ANOVA以及效应量计算进行统计评估。

**📊 数据集**

主要使用的公开数据集包括ModelNet40、ShapeNet、ABO、ScanObjectNN、S3DIS；针对鲁棒性评估还使用了ModelNet40-C、PointCloud-C等腐败数据集，并在训练时利用模型自带的扰动生成器。

**📈 对比分析**

比较方法是对每种网络在每个域和扰动组合下计算平均嵌入漂移、分类准确率，并用配对t检验与效应量验证显著性；在训练加不变性损失后再次评估在ModelNet40-C的平均腐败误差（mCE）和准确率。结果显示，在不变性损失下，RIConv++和PointMLP在所有模型中以0.823的平均准确率并列最佳，且在特定腐败（激光、遮挡）上获得显著提升。

**⚠️ 局限性**

局限性包括：使用的是10类别子集，不能直接与公开排行榜对比；λ=0.5的损失系数是经验选取，未系统探究不同权重；训练与评估均在匹配预算下完成，未评估更大规模或更高分辨率情形；只覆盖了三种不变性设计（RI、VN）和有限的10个网络，未考虑其他潜在架构。

---

## 132. Balancing Chain-of-Thought and Answer Supervision for Small Language Models on Medical Question Answering

**arXiv ID:** 2611.03948 | [PDF](https://arxiv.org/pdf/2611.03948v1)

**作者:** Le Zhou `[一作]` (Pace University), Zhan Zhang `[通讯]` (Pace University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在小型语言模型上，平衡链式推理（CoT）与答案监督的不同细分损失加权方式，并在多项式与自由文本医学问答任务上进行微调实验。

**💡 创新点**

创新点在于首次系统比较了均匀加权（UaS）与段落加权（SwS）两种损失策略，发现其对模型类型和任务格式有显著影响，并揭示过长CoT会降低性能，可为任务特定的长度控制提供依据。

**🔧 技术方法**

采用LoRA微调、基于token的交叉熵损失加权（λ参数调节），并使用deterministic greedy解码和统一答案提取分隔符。

**📊 数据集**

使用公开医学QA数据集MedQA、EMS-MCQA（多项选择）和Med-O1（自由文本）进行评估。

**📈 对比分析**

通过与零样本基线、UaS、SwS（多种λ）比较，发现一般性模型在MedQA上可提升44.8%准确率，SwS在多项选择题上更优，而UaS在自由文本问答中表现更好，整体提升范围从数个百分点到数十个百分点。

**⚠️ 局限性**

局限包括仅使用单一随机种子、仅评估六个模型、未覆盖更大模型或更多任务、缺乏长度惩罚机制、且λ调参成本高。

---

## 133. Stacking-Based Weighting for Large Language Bayes in M-Open Settings

**arXiv ID:** 2611.03966 | [PDF](https://arxiv.org/pdf/2611.03966v1)

**作者:** Anant Bhide `[一作]` (University of Massachusetts Amherst), Justin Domke `[通讯]` (University of Massachusetts Amherst)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于堆叠的权重聚合方法，取代Large Language Bayes（LLB）中传统的贝叶斯模型平均，提升模型组合在M‑open场景下的鲁棒性。

**💡 创新点**

创新点在于：①使用变分下界近似留一预测，构造可计算的堆叠目标；②加入KL正则化，既保留原BMA的先验信息，又允许权重根据预测性能重新分配；③在LLB框架内实现，保持模型生成与推断流程不变。

**🔧 技术方法**

核心技术包括：变分推断（ELBO）用于留一预测的近似；堆叠（stacking）权重优化；KL正则化约束；SciPy BFGS求解对数空间软最大化；NumPyro、NUTS用于后验采样。

**📊 数据集**

实验使用FEMA National Risk Index数据集中的县级年度损失（野火、内洪、飓风）共三项任务，训练集20个观测值，测试集10个观测值。

**📈 对比分析**

对比方法：与原LLB的贝叶斯模型平均（BMA）权重进行对照；评估指标为有效样本量（ESS）与留一预测的期望对数预测密度（ELPD）。结果显示堆叠方法在ESS上显著更分散，ELPD与BMA基本相同，说明预测性能保持一致但不确定性表达更合理。

**⚠️ 局限性**

局限性包括：权重分布仍受语言模型生成的候选模型多样性和质量限制；未能验证权重是否真正反映真实不确定性；实验范围仅限于三种灾害损失预测任务，未来需在更广泛场景中检验。

---

## 134. Testing Processor Contracts in Speculative Decoding Engines

**arXiv ID:** 2611.04042 | [PDF](https://arxiv.org/pdf/2611.04042v1)

**作者:** Tianyu Ding `[一作]`, Fei Jiang `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对在显式投递（speculative decoding）过程中，日志处理器可能因错误接收草稿或保留非顺序状态而导致生成分布偏差，提出了前缀契约（processor‑prefix contract）并设计了分支感知差异一致性测试（branch‑aware differential conformance testing）来检测并确认此类缺陷。

**💡 创新点**

创新点在于将前缀契约与可执行假设相结合，利用枚举小模型作为基准，同时在真实 Transformer 与 llama‑cpp‑python 引擎上进行预注册实验，证明草稿暴露与状态保留会显著改变输出分布。

**🔧 技术方法**

技术包括：显式投递抽样、日志处理器（penalty、grammar mask、guidance）接口设计、统计检测（Clopper‑Pearson、总变差下界）、可执行机制假设验证、源代码关联干预与修复验证。

**📊 数据集**

使用的“数据集”是合成的极小随机 Llama‑style 网络与 GGUF 网络（V≈256, L≈4）以及若干实际的 Hugging Face Transformers 与 llama‑cpp‑python 构建版本，并在 vLLM 0.8.5.post1 上做迁移测试。

**📈 对比分析**

比较方法采用双臂（speculation‑on vs speculation‑off）抽样，利用统计置信区间和总变差下界对事件概率差异进行检验；实验显示，针对 prompt‑lookup 与处理器状态泄露的修复能在 60% 以上的概率差距上给出显著下界，且在控制实验中未发现指令化选择带来优势。

**⚠️ 局限性**

局限性包括：只在极小模型与固定引擎版本下验证，未覆盖大型模型、动态验证路径或 RL 训练-推理链；统计方法依赖于预注册阈值和事件族，可能无法捕捉所有缺陷；以及对 vLLM 的迁移测试仅为示例，缺乏更广泛的生产环境评估。

---

## 135. Language Model Augmented Decision Transformer for Multi-Task Offline Reinforcement Learning

**arXiv ID:** 2611.04031 | [PDF](https://arxiv.org/pdf/2611.04031v1)

**作者:** Xinbo Zhao `[一作]` (State University Of New York At Binghamton), Yingxue Zhang `[通讯]` (State University Of New York At Binghamton)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为LaMA-DT的框架，利用单一大型语言模型（LLM）同时进行轨迹增强和多任务离线强化学习的策略学习；

**💡 创新点**

创新点在于：1）将LLM在六个多目标监督下进行细调，既能学习环境动力学、时序依赖，又能通过任务-轨迹对齐目标保证生成轨迹与自然语言任务描述一致；2）采用基于多目标似然的评分-过滤机制挑选高质量增强轨迹；3）将同一LLM重用为条件决策变压器，并用共享与任务特定的LoRA适配器实现跨任务知识迁移；

**🔧 技术方法**

使用技术包括：大型语言模型（LLM）与LoRA适配器、六个监督目标（动力学预测、轨迹插值、正向/反向生成、轨迹解释、描述驱动生成）、概率评分过滤、条件决策变压器（Decision Transformer）以及基于任务描述的源感知标记；

**📊 数据集**

实验数据集涵盖Meta‑World（ML5、ML30）和MuJoCo（Cheetah‑vel、Cheetah‑dir、Ant‑dir）两个离线多任务基准，涵盖近优与次优两种数据质量；

**📈 对比分析**

与七个竞争基线（CDS、MTBC、MTIQL、MTDIFF、MTDT、Prompt‑DT、HarmoDT）在标准和低数据（1%、5%、10%）以及不同数据质量下进行对比，LaMA‑DT在所有任务和场景均实现显著提升，低数据场景提升幅度最高；

**⚠️ 局限性**

局限性包括：1）依赖高质量且可描述的自然语言任务描述；2）对LLM的计算资源需求较高；3）在极低质量数据或极少任务描述时，增强效果受限；4）可能引入与原数据分布不一致的轨迹，需精细过滤。

---

## 136. Compression Forcing: Hierarchical History Tokenization Enables Flexible World Models

**arXiv ID:** 2611.04028 | [PDF](https://arxiv.org/pdf/2611.04028v1)

**作者:** Siqiao Huang `[一作]` (Carnegie Mellon University), Max Simchowitz `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 Compression Forcing 框架，利用动态感知的层次化 tokenization 对历史信息进行压缩并在推理时通过层级历史引导提升预测精度。

**💡 创新点**

核心创新在于将“部分压缩等同于部分掩蔽”原则落到工作中，构建了 Dynamics‑Aware Tokenizer 以动态重要性排序历史序列，并通过前缀条件扩散网络与层级历史引导实现可控的动态信息放大。

**🔧 技术方法**

技术手段包括 ViViT‑style 视频 Transformer 作为编码器、共享参数的扩散解码器、前缀条件下的双重注意力融合、AdaLN 条件编码以及基于分数函数组合的层级历史引导。

**📊 数据集**

在 Kinetics‑600（动作无关视频预测）、Counter‑Strike: Global Offensive（游戏模拟）和 RT‑1（机器人抓取/放置）三大任务上进行实验。

**📈 对比分析**

与 SD‑VAE、ElasticTok 等同类 tokenization 以及 DIAMOND、DreamerV4 等流行世界模型进行对比。Compression Forcing 在 PSNR、SSIM、FVD、FID、LPIPS 等视觉质量指标和后续策略评估上均实现显著提升，尤其在动态区域误差分布上表现更优。

**⚠️ 局限性**

局限性包括仅实现单步（next‑frame）预测，截断预算为统一设置，未针对不同时间距离的历史进行动态权重调整，且在极端高噪声/复杂交互场景下的鲁棒性待进一步验证。

---

## 137. A Practical Causal Framework for Autonomous Mobile Robots: From Simulation to Real-World Human Environments

**arXiv ID:** 2611.04027 | [PDF](https://arxiv.org/pdf/2611.04027v1)

**作者:** Luca Castri `[一作]` (University of Padua), Nicola Bellotto `[通讯]` (University of Padua)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过在仿真中学习的因果模型，仅对参数进行快速更新，实现在真实人类共享环境中的移动机器人导航

**💡 创新点**

① 结构不变、参数可调的因果框架，极大降低了从仿真到现实的迁移成本；② 在真实机器人上完成首个完整因果决策链的部署；③ 提供公开代码与数据，方便复现与后续研究

**🔧 技术方法**

因果发现（PCMCI+）、参数学习（MLE）、因果推断（do-calculus）与 A* 规划结合；使用 CausalFlow、pyAgrum、ROS/Teb、Gazebo 社会仿真器等技术

**📊 数据集**

① 仿真数据：基于 PeopleFlow 的 Gazebo 社会仿真，S1（海报会）与 S2（自助餐）共 1 小时/场景；② 真实数据：2 分钟 LiDAR/IMU 采样，约 10 Hz，20 名志愿者在 S1、S2 环境下收集

**📈 对比分析**

与基准（最短路径 A*、含社交成本的社交规划）比较，仿真中任务成功率提升 30‑50%，真实场景 100% vs 33%；proxemic 遵从率显著提升；推断时延约 0.12 s/查询，满足实时需求

**⚠️ 局限性**

① 只更新参数，无法应对因果结构的变化；② 仅测试了两种社交情景，缺乏大规模、多样化环境验证；③ 对高动态或极端拥挤情况的鲁棒性未评估；④ 现阶段对复杂社会动态（如情绪、意图）支持有限

---

## 138. Making Every Edit Count: Efficient Binary Edit Correction at the Gilbert-Varshamov Bound

**arXiv ID:** 2611.04023 | [PDF](https://arxiv.org/pdf/2611.04023v1)

**作者:** Ryan Gabrys `[一作]` (University of California San Diego), Eitan Yaakobi `[通讯]` (Technion---Israel Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文设计了一种确定性多编辑纠错二进制码，能够纠正最多 t 次插入/删除/替换错误，并实现冗余仅为 2t·log₂n+O_t(1)，同时提供多项式时间的编码与解码算法。

**💡 创新点**

创新点在于将经典的 Gilbert–Varshamov 计数论证转化为可执行构造：利用线性哈希与条件期望法在同一哈希类中保证至少一半单词可区分，从而在不枚举指数空间的前提下实现最优冗余。

**🔧 技术方法**

核心技术包括：仿射错误模式的线性表示、冲突计数与下界求解、条件期望的确定性选择、以及基于前缀的枚举编码实现；整个构造与解码过程均为多项式时间。

**📊 数据集**

该工作属于理论研究，不使用特定实验数据集；错误模型为二进制字串的插入/删除/替换编辑。

**📈 对比分析**

与 Bui‑Wang‑Vu 等近端方案相比，本文将冗余从 (2t+1)·log₂n+O(loglog n) 降至 2t·log₂n+O(1)，且保持确定性多项式时间；相比早期多删除构造，显著降低冗余并保持同阶时间复杂度。

**⚠️ 局限性**

局限性包括：算法对 t 的依赖导致在 t 较大时运行时间指数增长；目前仅适用于固定常数 t；在实际应用中，代码长度仍需足够大才能保证冗余优越性，且实现细节对常数因子较敏感。

---

## 139. Dory1.0-4B: A Looped Normalized-GPT

**arXiv ID:** 2611.03998 | [PDF](https://arxiv.org/pdf/2611.03998v1)

**作者:** Maxim Fishman `[一作]`, Krishna C. Puvvada `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种4B参数的循环式（latent-recurrent）语言模型，模型通过可变循环次数（loop count）实现推理计算预算的弹性，支持从1到4次循环。

**💡 创新点**

创新点在于：① 将循环深度作为可调的推理维度，使单一检查点可满足不同计算需求；② 采用nGPT归一化Transformer架构，结合滑动窗口与全注意力交错的层，分为输入块、共享循环块和输出块；③ 引入循环难度感知学习曲线（difficulty-aware loop curriculum）在RL阶段动态分配更深循环；④ 使用last-loop KV缓存减少循环对KV存储的线性增长；⑤ 对模型进行多种量化（MXFP8、NVFP4、GGUF）并通过量化感知蒸馏保持性能。

**🔧 技术方法**

使用的技术包括：nGPT归一化Transformer、滑动窗口注意力、全注意力、共享循环块、残差更新规则、L2归一化、分组查询注意力、Trm 归一化、量化感知蒸馏、RLVR、self-speculative decoding、last-loop KV caching。

**📊 数据集**

预训练使用约6万亿tokens，初期4.5万亿采用NVFP4，随后继续BF16训练并扩展到256K上下文；SFT阶段使用512K‑token打包序列，RLVR阶段使用23个单步环境（包括编码、代理、推理、长上下文问答、工具使用等），数据集来源于Nemotron 3 Ultra等多源混合数据。

**📈 对比分析**

与同参数量但无循环的基线及更大模型（Qwen3.5‑9B、Gemma‑4‑12B、Gemma‑4‑26B‑A4B）比较，2‑4次循环时平均在12个基准上超越这些模型，尤其在编码、代理任务和长上下文任务上显著提升；token效率提升1.3–2.5倍；last-loop KV缓存保持KV占用不变；量化模型速度提升1.5–2.4×。

**⚠️ 局限性**

限制包括：循环深度对每个任务的增益差异明显，需手动调参；循环数越多，推理时间和算力需求提升，仍受限于硬件；量化后部分任务性能略降；模型仅在英文/特定语料上验证，跨语言或更大上下文场景未评估。

---

## 140. Mixing Time of the Random-Pair Monte Shuffle

**arXiv ID:** 2611.03959 | [PDF](https://arxiv.org/pdf/2611.03959v1)

**作者:** Ilya Mironov `[一作]` `[通讯]`, Ilya Mironov

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

研究了随机对蒙特洗牌的混合时间，证明其在期望条件下的总变差距离为2nlog n+O(n)。

**💡 创新点**

首次确定了公共调度蒙特洗牌的全排列混合时间，具有精确的主导常数和O(n)的截止窗口。

**🔧 技术方法**

使用了蒙特框架和调度依赖的测试函数，结合矩阵平均算子进行方差估计。

**📊 数据集**

未具体提及使用的数据集。

**📈 对比分析**

通过比较相同公共调度的两次执行与经典随机置换行走，得到了上界；下界通过构造调度依赖的测试函数获得。

**⚠️ 局限性**

未提及具体的局限性。

---

## 141. On the optimality of truncation among binary linear sketches

**arXiv ID:** 2611.03979 | [PDF](https://arxiv.org/pdf/2611.03979v1)

**作者:** Robinson D. H. Cung `[一作]` (EPFL), Emre Telatar `[通讯]` (EPFL)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在仅通过二进制线性草图（固定数量的奇偶测量）可用的输入二进制向量的统计推断和函数评估问题。证明在某种单调性条件下，简单地通过保留输入的部分坐标进行截断是所有相同大小的线性草图中最优的。

**💡 创新点**

创新点在于证明了在对立符号的二进制向量的分布式相关性测试和评估单调布尔函数时，截断是最优的线性草图，并且为任意二进制线性草图提供了下界。

**🔧 技术方法**

使用了二进制线性草图技术，特别是通过固定的二进制矩阵进行线性映射。

**📊 数据集**

使用了独立的二进制对的样本数据集，具体包括在不同终端观察的二进制向量的分布式相关性测试和评估单调布尔函数的随机输入。

**📈 对比分析**

与其他方法比较时，证明了截断在对立符号的相关性测试中是最优的，且所有线性草图在图连接性问题中表现出严格的次优性，要求Θ(n log n)位，而非线性协议仅需O(log n)位。

**⚠️ 局限性**

限制在于仅在特定的单调性条件下，截断才是最优的，未来的研究需要识别其他可能的最优草图设置。

---

## 142. Camera Radar LiDAR Fusion via Point-Cloud Projection for Multi-Task Learning

**arXiv ID:** 2611.03942 | [PDF](https://arxiv.org/pdf/2611.03942v1)

**作者:** Kavin Chandrasekaran `[一作]` (Elektrobit Automotive GmbH), Sorin Grigorescu `[通讯]` (Transilvania University of Brasov)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种早期融合的摄像头-雷达-激光雷达网络，利用投影后的点云与摄像头图像共同完成车辆检测与可行驶空间语义分割。

**💡 创新点**

创新点在于将HD雷达点云与稠密激光雷达点云投影到视角图像平面后，与RGB图像做通道级拼接，构成单一五通道输入，显著提升多任务性能且保持计算效率。

**🔧 技术方法**

使用的技术包括投影变换、ResNet-50风格的特征金字塔编码器-解码器、检测头与分割头的共享特征图、Focal Loss和BCE损失，并在单通道融合中采用3×3卷积做低层特征提取。

**📊 数据集**

实验采用RADIal数据集，该数据集同时提供同步摄像头、HD雷达点云、激光雷达点云与标注，满足三模态学习需求。

**📈 对比分析**

与单模态基线以及现有多模态融合方法（如ROFusion、EchoFusion、REFNet++等）对比，本文模型在车辆检测的AP、F1和可行驶空间的mIoU均实现了最佳或第二佳成绩，同时帧率和GPU内存占用保持在较低水平。

**⚠️ 局限性**

局限性主要是仅针对二分类（车辆/非车辆、可行驶/不可行驶）任务，且对高维稠密雷达点云的投影与对齐仍需在不同环境下进一步验证。

---

## 143. Separating Common and Unique Directions for Data-Free Model Merging

**arXiv ID:** 2611.04108 | [PDF](https://arxiv.org/pdf/2611.04108v1)

**作者:** Shiting Wang `[一作]` (Earlham Institute), Ke li `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种数据无关的预处理框架 CUSD，先将多任务模型的任务向量投影到共享的方向基底中，分离出常见与独特方向，然后只对常见部分应用任意算术合并操作，独特部分按比例保留。

**💡 创新点**

创新点在于发现任务向量在共享基底下呈现低离散率的常见方向与高离散率的任务占主导的尾部两种结构，并通过 HO‑GSVD 派生共享基底与 KDE 自动阈值实现常见/独特分离，显著降低合并时的任务干扰。

**🔧 技术方法**

核心技术包括正则化 HO‑GSVD 获得共享正交方向基底、基于标准差的离散度评分、核密度估计阈值分割、以及对常见子空间与独特残差的投影和可调 β 权重控制。

**📊 数据集**

在 11 种基础模型（CLIP ViT、RoBERTa、Llama、Gemma 等）上使用 8 张图像分类数据集、8 个 GLUE NLU 任务以及 MergeBench 的代码/数学/多语言 LLM 任务进行评估。

**📈 对比分析**

与四种算术合并方法（Task Arithmetic、TIES‑Merging、DARE、Consensus Merging）以及两种子空间合并方法（TSV‑Merge、Iso‑Merging）比较，CUSD 在 42/44 组合中取得提升，平均提升约 12 分（ViT）、约 15 分（RoBERTa）及 22 分（LLM），最大单次提升达 22.5 分。

**⚠️ 局限性**

主要局限包括：需要一次性计算共享基底的成本较高（尤其是大规模 LLM ），独特残差的 β 调参在不同任务/模型间差异显著，且对极端任务分布或高度互斥的任务组合可能效果有限。

---

## 144. The Agent Board: Kanban for Machines; Durable Coordination for Agents That Outlive the Conversation

**arXiv ID:** 2611.04100 | [PDF](https://arxiv.org/pdf/2611.04100v1)

**作者:** Tyler Hunt `[一作]` (Corvic AI Research), Gurbinder Gill `[通讯]` (Corvic AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于共享表的 Agent Board，用于在 LLM 驱动的智能体系统中实现持久、可扩展的任务协调与等待；

**💡 创新点**

核心创新包括：①半语义任务记录，系统字段可精确路由、内容字段为 LLM 读写；②以阻塞为原语的等待机制，让智能体在任意时间无需占用资源；③Board Keeper 作为中断处理器与调度器，负责事件匹配、智能体分配和状态更新；④可插拔的协调策略 Kernel，使不同工作模式（如 Kanban、分布式聚合、升级至人类）在同一平台统一实现；

**🔧 技术方法**

实现技术涵盖：LLM 调用、SQL/表格后端作为持久化存储、消息队列/事件驱动的 Board Keeper、文本匹配的语言模型推理、任务继续写入“笔记”以减轻历史读取成本、模拟器和真实案例的实验；

**📊 数据集**

本文未使用公开数据集，而是通过模拟实验（时间跨度 10 min–24 h）、Synthetic 任务集（400 次跑）以及 16 条真实委托记录进行验证；

**📈 对比分析**

对比方法：对“轮询”“保持调用”“外部监视器”“Agent Board”四种等待策略，使用 token 计数和模型调用次数衡量；实验结果显示 Agent Board 的等待成本对时间几乎不变，显著优于轮询；与外部监视器相比，Board Keeper 在可读性、可扩展性和自动分配方面具有优势；

**⚠️ 局限性**

局限性包括：①仅保证若干基本交付属性，缺乏互斥、状态一致性和死锁检测等协议保证；②匹配等待条件的文本解析易受模型误差影响，误匹配率约 3.6%；③在高并发或大规模等待队列下的性能未知；④仍需手动制定 Kernel 的内容契约以避免信息缺失；

---

## 145. Online Hamilton-Jacobi Reachability Onboard the Robot Control Loop: A Real-Time GPU Solver for Safety-Critical Autonomy

**arXiv ID:** 2611.04096 | [PDF](https://arxiv.org/pdf/2611.04096v1)

**作者:** Kevin Weekly `[一作]` (Mirror AI), Alexandre M. Bayen `[通讯]` (University of California, Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了一种 CUDA 加速的高阶 WENO5+TVD‑RK3+Lax–Friedrichs Hamilton–Jacobi 级数求解器，能够在机器人控制频率（30–100 Hz）下实时重算后向可达管（BRT），并在多智能体火灾响应仿真中实现一体化的捕获基尔诺瓦维氏图。

**💡 创新点**

创新点在于：① 将每个 RK3 阶段融合为单个 GPU 核，消除中间缓冲拷贝；② 通过除法优化的 WENO5 权重、恒定 CFL 约束和内置 BRT clamp，利用编译时信息实现显著性能提升；③ 在求解器内部实现标签传播，完成 K‑agent 捕获基尔诺瓦维氏图，单次求解成本几乎不随队伍规模增长；④ 在 FP32 模式下实现跨平台（RTX 4000 Ada 与 Jetson AGX Thor）可实时运行，突破传统离线 BRT 的限制。

**🔧 技术方法**

使用技术包括：CUDA 手写核（单阶段 RK3 融合）、WENO5 空间重构、TVD‑RK3 时间积分、Lax–Friedrichs 数值 Hamiltonian、GPU 缓存利用（L2 亲和）、CUDA Graph 重放、FP32/FP64 模式切换、以及与 JAX/XLA 与 PyTorch torch.compile 的对比实验。

**📊 数据集**

主要数据集/测试案例为：2D holonomic_l2、double_int、drifter_const、drifter_wg、drifter_obs；3D Mitchell Air3D；4D 追逐‑逃逸（pe_4d、pe_4d_walls）；以及在 20 km × 20 km 区域、时变风场下的 6 架无人机火灾响应仿真。

**📈 对比分析**

与 JAX 实现的 hj_reachability 以及 torch.compile 同一数值方案对比，FP64 下本实现平均加速 2.0–4.6×，FP32 下加速 5.9–17.5×；在 RTX 4000 Ada 上 Air3D 可达 112 Hz、4D 追逐 61 Hz、2D 漂移 74–147 Hz；在 Jetson AGX Thor 上对应速度为 66 Hz、54 Hz，均保持 3–8 倍速度优势；在所有平台上均实现了 30 Hz 以上的实时重算。

**⚠️ 局限性**

局限性包括：① 受指数维度扩展限制，4D 以上几乎不可行；② FP32 精度在 BRT 边界附近可能出现 0.1% 误差，需视为不确定带；③ 需要专用 GPU，CPU 或低功耗嵌入式平台速度仍受限；④ 手写 CUDA 维护成本高；⑤ 对非齐次/时间变化流场的进一步扩展仍有限。

---

## 146. MABEL: An Open Anthropomorphic Mobile Bimanual Platform for Teleoperation and Learning

**arXiv ID:** 2611.04083 | [PDF](https://arxiv.org/pdf/2611.04083v1)

**作者:** Qilong Cheng `[一作]` (University of Toronto), Ludovic Righetti `[通讯]` (New York University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一款56自由度、全身可调、开源的移动双手机器人MABEL，并提供完整的硬件设计、软件栈、仿真模型和从遥操作到学习策略的端到端流水线。

**💡 创新点**

创新点在于将低成本、可调身高、后驱臂、17自由度手、可升降台、主动脖子与全向底盘等关键功能集成在一台机器人中，同时实现基于Apple Vision Pro的全身无控制器遥操作与共享的控制与学习框架。

**🔧 技术方法**

技术上采用ROS 2、MuJoCo/Isaac Lab仿真、全身QP控制器、后驱电机臂、ORCA多指手、三模块swerve底盘、视觉重定向、动作块化Transformer (ACT)策略等。

**📊 数据集**

作者自行收集了基于Apple Vision Pro的遥操作演示数据，共80次演示，117条片段，约38分钟、68,364帧，使用LeRobot v3格式进行整理。

**📈 对比分析**

实验中，机器人在重复性测试中中位误差为1.65 mm；路径跟踪误差在3.8–7 mm范围；轮子里程计与视觉/IMU对比误差分别为13.5 mm/1.63°；遥操作时延约137 ms；基于收集数据训练的ACT策略在抓取并放置任务中10次试验成功率为80%。

**⚠️ 局限性**

局限性包括升降台速度慢、关节比例与人类不完全一致、缺乏下肢关节导致全身运动受限，以及训练数据主要集中于上半身，限制了移动规划和长期自主性能。

---

## 147. Polynomial VASS MDPs

**arXiv ID:** 2611.04082 | [PDF](https://arxiv.org/pdf/2611.04082v1)

**作者:** Michal Ajdarow `[一作]` `[通讯]` (Masaryk University), Michal Ajdarow (Masaryk University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种判定方法，能够在多项式时间内决定给定向量加法系统马尔可夫决策过程（VASS MDP）的终止/计数复杂度是多项式界还是指数界，并给出了相应的上界或下界估计。

**💡 创新点**

创新点主要包括：①对VASS MDP的终止/计数复杂度建立了新的双分界（either all super‑polynomial functions are upper bounds or the lower bound is 2^{√n}）；②提出了“多组件”与“确定/随机分解”两种新工具，用以把随机过程拆解为可管理的确定性和随机子过程；③通过约束系统（I）与（II）的对偶性，能够在多项式时间内计算出能区分两种情况的判别准则。

**🔧 技术方法**

使用的主要技术包括：多组件与简单组件的线性代数分解；确定/随机分解（pointing decomposition）将VASS MDP的行为视为多条独立的马尔可夫链迭代；马尔可夫过程中的超级/亚马尔可夫过程；线性规划求解约束系统；以及对失败时间函数的等价定义。

**📊 数据集**

本文不涉及实验数据集，全部工作为理论分析与算法设计。

**📈 对比分析**

与现有工作比较：传统VASS MDP的可判定性问题往往属于更高复杂度，本文实现了多项式时间决策，并给出了明确的上界/下界；在DAG‑like VASS MDP中进一步给出了可计算的多项式上界（n^k或n^{2·k}），表现出比之前只能得到“至少多项式或指数”更精细的结果。

**⚠️ 局限性**

局限性：①下界仅为2^{√n}，尚不清楚是否可进一步提升至更高指数；②在非DAG‑like VASS MDP中，即使所有MEC均为多项式，也无法保证存在某个多项式上界；③对于DAG‑like VASS MDP，虽然给出了多项式上界，但仍无法直接计算出精确的多项式次数（需要进一步研究）。

---

## 148. RAPID: Increasing Efficiency of Internal Distillation via Randomized Methods

**arXiv ID:** 2611.04081 | [PDF](https://arxiv.org/pdf/2611.04081v1)

**作者:** Ryan Swift `[一作]` (University of Southern California), Konstantinos Psounis `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为RAPID的内部知识蒸馏方法，消除了传统投影矩阵的学习需求；

**💡 创新点**

创新点在于使用Johnson–Lindenstrauss随机投影实现教师特征压缩，并引入结合EMA与不确定性加权的动态多目标损失；

**🔧 技术方法**

采用随机稠密二进制投影、动态权重机制、KL散度、MSE以及EMA平滑技术；

**📊 数据集**

在Gemma 3和Qwen 2.5两大LLM家族上，用40 M推理数据（SmolTalk与OpenThoughts‑114k）进行实验；

**📈 对比分析**

与传统学习投影方法比较，RAPID在相同任务（MMLU、ARC‑Challenge、BBH、GPQA）上性能相当，且参数开销降低6.3–21.2%，训练效率提升最多21.1%；

**⚠️ 局限性**

局限在于仅评估推理微调阶段，未覆盖预训练或工具使用等更广泛训练阶段，且实验规模受单服务器RTX A6000 GPU限制，未验证大规模或长上下文情境下的效果。

---

## 149. AlgoPilot: Cross-Paradigm Reasoning in Language Models via Strategy Selection and Guidance

**arXiv ID:** 2611.04079 | [PDF](https://arxiv.org/pdf/2611.04079v1)

**作者:** Yilun Hao `[一作]` (Massachusetts Institute of Technology), Chuchu Fan `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了跨范式推理框架 AlgoPilot，能够自适应地选择解题范式（CoT、代码、SMT、PDDL、MILP）并生成对应的结构化问题表述，再由执行模型得到答案。

**💡 创新点**

创新点在于：①将范式选择与问题表述分离，学习自适应路由；②使用专家 GuideLM 生成范式特定的表述；③通过执行反馈进行监督与强化学习，构建训练数据和奖励机制；④在同一小模型上显著提升多范式任务性能。

**🔧 技术方法**

技术包括：监督微调（SFT）与基于 GRPO 的强化学习、结构化表述生成、跨范式执行评估与奖励设计、以及多模型迁移实验。

**📊 数据集**

使用了 13 份基准（如 Big-Bench Hard、GSM8K、LeetCode、OptiGuide、Chain-of-Experts、NL4Opt、PDDL IC、SAT-LM 等）覆盖 66 个领域，总计约 22k 题目；训练集分为 80/10/10，测试集 2,220 题。

**📈 对比分析**

与同规模小模型（Qwen3.5-9B、Gemma-4-E4B-it）及大模型（GLM-5、Qwen3-Coder-Next）在 5 类任务上对比，AlgoPilot _SFT+GRPO 在平均准确率上从 38.4% 提升至 72.9%，比 GLM-5 的 61.2% 提升 11.7%。Top‑3 版本进一步达 81.7%。

**⚠️ 局限性**

限制在于：①需要昂贵的执行反馈来构造训练数据；②训练过程与具体执行模型绑定，跨模型迁移仍有误差；③对新增范式的扩展需重新采集反馈，影响可扩展性。

---

## 150. Finite-Sample Convergence in Networked Average Reward MARL: Decentralization Pitfalls and Entropy Remedies

**arXiv ID:** 2611.04072 | [PDF](https://arxiv.org/pdf/2611.04072v1)

**作者:** Yizhou Zhang `[一作]` (California Institute of Technology), Adam Vierman `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在网络化平均奖励多智能体强化学习中，基于本地k跳邻域的去中心化演员-评论家算法，并给出无正则化下的梯度范数收敛以及固定状态熵正则化下的全局收敛保证。

**💡 创新点**

首次证明了本地化策略下的多代理与部分可观测性导致无法保证全局最优，然后提出通过线性函数逼近的局部评论家与熵正则化来克服这些障碍，实现了有限样本的全局收敛和梯度范数收敛。

**🔧 技术方法**

采用指数衰减属性、线性函数逼近、软max策略梯度、平均奖励马尔可夫链无偏估计、熵正则化与Polyak–Łojasiewicz不等式、混合时间分析等技术。

**📊 数据集**

在人工构造的维护单位离散网络环境（路径与网格拓扑）中进行实验，使用这些自定义的有限状态机数据集。

**📈 对比分析**

与全局表格评论家以及不同k、特征维度设置对比，实验表明适度的邻域半径和足够维度的线性特征能显著提升每代理奖励，而表格评论家在大邻域下表现不佳。

**⚠️ 局限性**

结果依赖网络指数衰减假设、仅适用于本地化策略且熵正则化需要足够强，且对更广泛的全局转移结构或弱局部依赖缺乏理论支持。

---

## 151. Spatial and Temporal Super-Resolution and Denoising of Intracardiac 4D Flow MRI

**arXiv ID:** 2611.04053 | [PDF](https://arxiv.org/pdf/2611.04053v1)

**作者:** Yusuf A. Ozturk `[一作]` (Washington University), Fanwei Kong `[通讯]` (Washington University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种名为4D-MRAR的深度学习框架，用于同时对4D Flow MRI的空间和时间分辨率进行超分辨率重建和去噪。

**💡 创新点**

创新点包括将卷积神经网络、ConvLSTM时间序列模块与坐标基多层感知机结合，并在训练中加入完整的质量和动量守恒方程进行物理信息约束，实现在任意时空坐标处连续预测速度和压强场。

**🔧 技术方法**

使用了残差CNN编码器、ConvLSTM时序增强模块、坐标基MLP解码器以及物理信息正则化（Navier–Stokes方程残差）等技术。

**📊 数据集**

在合成的左心室（LV）CFD数据集（粗细两种分辨率）和全心室带阀门的CFD数据集以及真实4D Flow MRI数据上进行训练与评估。

**📈 对比分析**

与4DFlowNet、Callmer等时空超分方法和增强PINN进行对比，4D-MRAR在绝对误差、相对误差、PSNR、SSIM以及动量和质量残差上均优于基线，尤其在复杂涡流和时间演化场景中表现突出。

**⚠️ 局限性**

局限性包括合成与真实数据之间的差异、缺乏高分辨率真值进行定量验证、对真实扫描中更复杂的k空间欠采样和边界条件缺失的处理不足，以及在舒张期快速涡流区域的误差仍相对较大。

---

## 152. The Evolution Strategies Paradox: Better Models from Worse Perturbations

**arXiv ID:** 2611.04113 | [PDF](https://arxiv.org/pdf/2611.04113v1)

**作者:** Roberto Dailey `[一作]` (Cognizant), Xin Qiu `[通讯]` (Cognizant)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究并解释了在高维参数空间中，即使所有后代模型的性能均低于父模型，Evolution Strategies（ES）仍能通过噪声与曲率的相互作用实现优化。

**💡 创新点**

揭示了所谓的“ES悖论”，并证明其是高维空间中噪声与曲率相互作用的必然结果，从而为LLM微调提供新的优化范式。

**🔧 技术方法**

采用了ES算法（带正交高斯噪声、z-score归一化奖励、少量种群），结合了解析模型和经验实验分析。

**📊 数据集**

在Qwen 2.5 7B、OLMo 2 7B等大型语言模型的数百题Sudoku和计数任务上，以及60K参数CNN在MNIST数据集上进行实验。

**📈 对比分析**

与传统梯度强化学习和更大种群ES相比，在同等计算预算下，ES在LLM微调中表现相当甚至更优，尤其在种群仅30的情况下仍能取得显著提升。

**⚠️ 局限性**

对噪声、曲率估计的依赖导致超参数敏感，且对低观测噪声或过大扰动时效果下降；对在线动态调优的缺乏也是限制之一。

---

## 153. ModelBlaster: Closing the Performance Gap through Learning-Augmented Agents Across Models, Kernels and Hardware Backends

**arXiv ID:** 2611.04104 | [PDF](https://arxiv.org/pdf/2611.04104v1)

**作者:** Shengjun Kris Dong `[一作]` (NVIDIA), Christos Kozyrakis `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Memory-Augmented In-Context Reinforcement Learning (MAIC‑RL) 框架，结合 Actor‑Critic Textual Gradient Descent (AC‑TGD) 进行 LLM 生成的代码重写，并通过持久化知识库实现跨任务、跨模型、跨硬件的性能经验共享；同时扩展到整模型图层的优化，使得 kernel 级优化能反馈到模型层决策。

**💡 创新点**

① 将 LLM 视作可写入上下文的策略，而非可训练的权重，使用文本梯度更新知识库；② 通过持久化、结构化的状态‑动作‑结果记忆实现经验迁移，突破传统单任务、单后端的限制；③ 采用统一的优化循环与接口，可在 CUDA、OpenCL、RVV 等不同语言与 ISA 上无额外训练直接迁移；④ 在模型层与 kernel 层形成双向反馈，使得模型图的重排能根据已生成的 kernel 反馈自适应调整。

**🔧 技术方法**

- 大规模 LLM（GPT‑5.0‑mini、GPT‑4.1、Sonnet‑4.6、Opus‑4.6）作为代码生成与评估的核心；- MAIC‑RL 与 AC‑TGD 作为推理‑时强化学习框架；- 持久化知识库（结构化表存储状态、技术、结果、约束）；- 基于硬件的 Profile 采样与状态分类（NVIDIA Nsight Compute、Qualcomm OpenCL 计时、RVV 计数器）；- 统一的外部 Harness 负责编译、验证与性能测量。

**📊 数据集**

KernelBench（Level 1‑3）和 SOL‑ExecBench（Level 1‑2）作为基准；在 NVIDIA L40S/H100/B200、Qualcomm Adreno 650、RISC‑V Saturn V256D128 等多种后端上进行评测。

**📈 对比分析**

与现有基线（PyTorch eager、QNN SDK、Apache TVM、ExecuTorch/XNNPACK）进行对比。 在 NVIDIA 上，平均几何速度提升为 1.85×‑5.58×；在 Qualcomm 上相较 QNN 达到 5.8×，相较 TVM 1.7×；在 RISC‑V 上相对 ExecuTorch/XNNPACK 达到 2.7×。 整模型层在 KernelBench Level 3 上平均 5.18×。

**⚠️ 局限性**

- 仍需针对每个后端实现特定的编译、验证与 profiling harness，无法完全无目标代码；- 依赖 LLM 推理成本较高，延迟与资源开销不可忽略；- 经验共享受限于状态匹配，极端不同算子或硬件可能导致迁移效果有限；- 目前未深入探究大规模模型的子图分解与更高级的 graph‑level 结构化搜索。

---

## 154. Is This 3D Asset Usable? Beyond the Beauty Render for Asset Curation

**arXiv ID:** 2611.04102 | [PDF](https://arxiv.org/pdf/2611.04102v1)

**作者:** Rushikesh Zawar `[一作]` (Adobe), Vladimir Kim `[通讯]` (Adobe)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建诊断渲染证据（DRE）体系，对 3D 资产进行可复用性评估，结合美观、线框和 UV‑边缘渲染视图，并通过 ViT 分类器进行融合。

**💡 创新点**

提出了针对 3D 资产的多模态诊断渲染证据，设计了无仓库依赖的渲染管线，并提供了公开的 1,234 资产专家标注基准。

**🔧 技术方法**

使用冻结的 ImageNet ViT‑B/16 作为特征提取器，独立训练美观、线框和 UV‑边缘三类分类器，随后进行晚期融合；采用自定义渲染器生成对齐视图。

**📊 数据集**

TexVerse 1,234 资产（866 良好 / 368 坏），6,685 个辅助 QA 资产，以及 764 个外部参考集（382 良好 / 382 坏）。

**📈 对比分析**

与 Objaverse++（40 视图）及仅使用美观渲染的 ViT 基线比较，DRE 在 TexVerse 上达到 86.36% 平衡准确率、88.32% 坏资产召回率，外部参考集 88.9% 平衡准确率；明显优于 Objaverse++ 的 44.65%/55.72%。

**⚠️ 局限性**

仅限于纹理/PBR 资产，无法覆盖所有材料错误；依赖于冻结的 ViT 可能限制了对新型视觉模式的适应；基准规模仍较小，未覆盖所有类型资产；仅使用三种诊断视图，可能遗漏某些结构或材质缺陷。

---

## 155. Evaluation Contracts and Evaluation Leverage: Sliced-Gap Readings in Structured Prediction Can Move Without Model Change

**arXiv ID:** 2611.04099 | [PDF](https://arxiv.org/pdf/2611.04099v1)

**作者:** Hao Lu `[一作]` (Army Engineering University of P.R. China), Ruiyang Li `[通讯]` (Army Engineering University of P.R. China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了 NLP 评估报告中的性能差距读数（如 intra–inter gap）并指出这些读数是模型与评估合同（scoring function、gold 标注、阈值、输出格式、提示语言）的联合属性。

**💡 创新点**

创新点包括：1）对评估杠杆（evaluation leverage）的正式理论框架；2）构建“审计评估读数”（Audited Evaluation Reading, AER），在合同边缘化、Tie 声明、种子规范和有效性门控下提供稳定、可重复的评估；3）发布 EvalLever 工具包和预注册评估范式。

**🔧 技术方法**

采用的技术包括：形式化的四个命题（命题1–4），基于冻结预测的合同变更实验，文档级 Bootstrap CI，统计检验（Holm、TOST、Spearman、Wilcoxon），以及对 LLM 评估器的审计。

**📊 数据集**

使用的数据集涵盖：DocRED、HacRED、UD Probe、WMT19、TACRED、TREC DL 2019、SQuAD v2、BC5CDR、RAMS 等多种任务和语言。

**📈 对比分析**

在多项对比中，AER 在所有记录的合同引起的排名翻转中都能保持正确的 Tie 声明，并在 WMT19 等基准上与最佳单一指标保持 ϵ=0.10 的等价性；实验显示杠杆效应随模型与合同变更而显著变化，且 variance‑level 排序得到验证。

**⚠️ 局限性**

局限性包括：只评估了少量 LLM（5 个）和单个解码；种子数量受限（仅 15 个）；只在 DocRE 开发集上验证；未覆盖所有可能的评估合同空间；工具包与评估流程的整合仍需进一步测试。

---

## 156. VARP: A Visual Adaptive Rearrangement Planner for Non-Monotone Object Rearrangement Planning

**arXiv ID:** 2611.04093 | [PDF](https://arxiv.org/pdf/2611.04093v1)

**作者:** Hanwen Ren `[一作]`, Ziyan Wu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种视觉自适应重排规划器（VARP），通过闭环视觉反馈实时预测并执行非单调物体重排任务

**💡 创新点**

结合学习的目标条件视觉策略与两种轻量级推理模块——动作细化策略（ARS）与进度强制策略（PES），实现在线动作可行性校验与进度维护，避免昂贵搜索与重规划

**🔧 技术方法**

利用U‑Net编码‑解码网络学习热图预测，MS‑MCTS生成专家演示，训练过程中使用监督式模仿学习，推理时采用热图候选采样、线性运动规划验证、短程回放探测

**📊 数据集**

使用由MS‑MCTS生成的100,012个演示数据（约1M状态-动作对），涵盖4至8个圆柱物体的20×20平面工作空间；在真实机器人上进一步使用4、5个物体演示进行仿真到真实的迁移学习

**📈 对比分析**

在2000个标准重排基准、扰动恢复基准、离散与持续扰动测试以及样本外物体数量测试中，与MS‑MCTS、PERTS和Transporter Network等基线比较，VARP在成功率上达到约98%（比MS‑MCTS高9%）、规划时间0.4s（比搜索方法快>20倍），在扰动恢复和持续扰动情境下成功率分别为97.5%和94.5%，显著优于基线

**⚠️ 局限性**

对极端拥挤或大物体数（≥9）时性能下降明显（成功率下降到64%），对环境视觉噪声与物体尺寸变异的鲁棒性未充分评估，且模型对全景视觉输入的适应性仍有限

---

## 157. Same Rule, Different Verdicts: Rubric Language and Clause Structure in LLM Judges for Korean Support Dialogues

**arXiv ID:** 2611.04091 | [PDF](https://arxiv.org/pdf/2611.04091v1)

**作者:** Sangdo Han `[一作]` (CEO Business School), Moonsoo Kim `[通讯]` (CEO Business School)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了语言（韩语 vs 英语）和条款结构（肯定、否定、条件、例外）对LLM评判支持对话的影响，使用12条准则并在六个低成本模型上评估一致性。

**💡 创新点**

通过设计意义相等的四种句式对比，直接测量模型在无参照标签下的自洽性，评估语言与结构对判定一致性的具体影响。

**🔧 技术方法**

采用Cohen's κ和混合效应Logistic回归分析跨模型一致性，使用Bootstrap检验语言差异，并收集模型输出实现结构/语言翻转率。

**📊 数据集**

以ESConv 60个自然对话的韩语翻译为Set A，和基于12条准则构造的144个最小对照边界案例为Set B。

**📈 对比分析**

通过比较跨供应商、同供应商和配对示例的κ值发现韩语rubric下一致性降低0.058，否定/条件/例外项的争议率显著升高；示例未显著缩小语言差距。

**⚠️ 局限性**

未使用人工真值标签，仅评估一致性；所有材料由LLM生成，可能与模型偏好相关；仅涵盖低成本模型且缺乏韩语供应商；翻译和案例构造可能影响绝对判定。

---

## 158. Proceedings 23rd International Conference on Quantum Physics and Logic

**arXiv ID:** 2611.04069 | [PDF](https://arxiv.org/pdf/2611.04069v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 159. Swapping the Judge Changes the Ranking: A Preregistered Study of Cross-Provider Replication in LLM-Judged System Comparisons

**arXiv ID:** 2611.04090 | [PDF](https://arxiv.org/pdf/2611.04090v1)

**作者:** Sangdo Han `[一作]` (CEO Business School), Moonsoo Kim `[通讯]` (CEO Business School)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过在Chatbot Arena公开的多轮对话数据上，让多家供应商的LLM评判模型对两份回复进行二选一判断，评估单一评判器与跨供应商复制结论的可信度、所需判决数量以及评判偏差的共现情况。

**💡 创新点**

①预注册实验设计，系统性评估跨供应商复制对结论可靠性的提升与代价；②揭示低成本LLM评判器在此任务上与人类投票更贴近；③量化评判器共现偏差，说明评判错误并非独立。

**🔧 技术方法**

使用Claude Opus/Haiku、GPT‑6 Luna、Gemini 3.1 Flash‑Lite等LLM进行判决；统计方法包括Cohen κ、Kendall τ、Spearman ρ、Bootstrap检验、最小样本量评估、共现偏差检验等。

**📊 数据集**

Chatbot Arena 2025公开数据，65,636条单轮战斗，包含9个系统、33对系统；确认集4,000条均有单票人类投票，涵盖英语与非英语。

**📈 对比分析**

与人类投票对比：单一评判器的κ≈0.22，τ≈0.67；跨供应商复制提升单一评判器的精度约+0.10，但覆盖率下降；低成本评判器在每对系统约25个判决即可与人类收敛；高阶评判器误差更大，样本量需求更高。

**⚠️ 局限性**

局限性：样本规模小（9系统、33对）；人类投票仅单票，噪声大；评判器推理设置不统一；仅低成本模型进行跨供应商比较；未对自我/族群偏好作深入验证；提示、语言及顺序处理方式可能影响结果；无法完全区分共同偏差与投票噪声。

---

## 160. From Visual Matching to Textual Conditioning: Cross-Domain Few-Shot Segmentation via Reference Inversion

**arXiv ID:** 2611.04084 | [PDF](https://arxiv.org/pdf/2611.04084v1)

**作者:** Jiahao Nie `[一作]` (Nanyang Technological University), Shijian Lu `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过参考图像的文本反演学习可复用的软文本标记，利用冻结的SAM3模型完成跨域少样本分割，消除对每个目标图像单独视觉匹配的需求。

**💡 创新点**

①双层参考反演：在冻结文本编码器前后分别学习词嵌入和概念标记，兼顾语言空间锚定与灵活性；②层自适应文本条件：按解码层动态融合两类标记；③目标存在感与查询置信度联合监督，提升预测可靠性。

**🔧 技术方法**

使用SAM3框架、文本反演技术、层自适应融合、Presence‑aware Confidence Supervision，仅对每个类别优化约12k可学习参数。

**📊 数据集**

在四大跨域少样本分割数据集（ISIC2018、Chest X‑Ray、DeepGlobe、SUIM）上进行实验，并在PASCAL‑5i、COCO‑20i、FSS‑1000、PASCAL‑Part、PACO‑Part等任务上验证通用性。

**📈 对比分析**

与ResNet/ViT、Vision Foundation Model（VFM）基础方法以及GF‑SAM、PerSAM‑F、VLP‑SAM等现有技术比较，RI‑SAM3在所有数据集上均实现显著提升（如ISIC +9.2%，Chest X‑Ray +6.4%，SUIM +8.3%），mIoU最高、标准差最低，同时推理速度快、模型体积小。

**⚠️ 局限性**

局限性：仍依赖SAM3的预训练冻结，文本反演对极端域变换或极少样本（<1）场景的鲁棒性未知；在多类别或实例级分割上的扩展尚需进一步研究。

---

## 161. FEIJOA: Task-Driven Grounding and Execution for Mobile Manipulation

**arXiv ID:** 2611.04080 | [PDF](https://arxiv.org/pdf/2611.04080v1)

**作者:** Aditya Agarwal `[一作]` (MIT CSAIL), Leslie Pack Kaelbling `[通讯]` (MIT CSAIL)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于语言模型的移动机器人任务驱动定位与执行框架，能够在实际环境中完成从自然语言指令到具体操作的闭环执行。

**💡 创新点**

创新点在于：①在指令到达时动态从原始 RGB‑D 观察中定位目标；②使用执行前后验证与可行性采样来保障技能可执行性；③结合成本感知规划与反馈驱动的计划修正，实现高成功率。

**🔧 技术方法**

核心技术包括：大语言模型（Gemini‑3.8‑Flash）做决策与推理；定位工具（Grounding Tool）对原始扫描进行查询；机器人抓取、放置、推挤、擦拭等可参数化技能；可行性采样、效果验证（视觉判定）、成本感知规划等。

**📊 数据集**

使用内部收集的四个室内 RGB‑D 扫描（共 103 条定位查询）以及 50 次 Spot 机器人真实操作实验；没有公开公开数据集，所有实验基于作者自行构建的数据。

**📈 对比分析**

与四种基线（Memory Over Maps、ObjMap、ObjMap+VLM、TextMap）对比，定位准确率（Exact Match）从 68.0% 提升至 94.2%；在 50 次实地试验中完成 35 个任务（70%），其中 17 个成功归因于对失败调用的恢复，主要瓶颈为机器人技能失效。

**⚠️ 局限性**

局限性：①机器人物理技能的不稳定性仍占主导失败原因；②基于 LLM 的推理与定位增加了推理延迟；③对极端光照或复杂物体（如反光物品）的感知仍不可靠；④当前仅在实验室环境验证，缺乏对更大规模、多任务环境的评估。

---

## 162. Look Before You Launch: Learned Throughput Prediction for Hardware Selection in a Production Training Fleet

**arXiv ID:** 2611.04057 | [PDF](https://arxiv.org/pdf/2611.04057v1)

**作者:** Hossein Askari `[一作]` (Meta), Keyu Man `[通讯]` (Meta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了LEAP系统，在训练作业提交时预测每个候选硬件的吞吐量并按吞吐量/成本排序，以提升资源调度效率。

**💡 创新点**

创新点在于引入基于梯度提升树的分箱级联模型、利用作业历史特征和硬件规格构造特征、以及在训练集上每晚自动重新训练并发布版本。

**🔧 技术方法**

使用的技术包括梯度提升树（LightGBM/Scikit‑learn实现）、分箱分类器、日志变换目标、pinball 损失、特征提取、作业历史快照和 nightly retrain pipeline。

**📊 数据集**

数据集为真实生产环境中每日约25,000个作业尝试，覆盖约660种模型类型和20多种硬件类型，训练窗口为前30天，评估在随后一天。

**📈 对比分析**

与原始基于查找表的规则引擎和其他模型（LightGBM、TabPFN、语言模型等）对比，LEAP的 median w20 为77%（对比表格基线53%），对覆盖作业的预测准确率在±20%内达85%，并在2026年上半年估计降低了3%的训练成本。

**⚠️ 局限性**

局限性包括对未见硬件类型或模型族的泛化能力差、过度依赖历史记录、对小作业的准确率较低、以及训练-服务偏差导致的性能波动。

---

## 163. HybridSpec: Throughput-Aware Speculative Decoding for Hybrid GDN-MoE Models

**arXiv ID:** 2611.04055 | [PDF](https://arxiv.org/pdf/2611.04055v1)

**作者:** Haiyu Wang `[一作]` (Meta AI), Steven Li `[通讯]` (Meta AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在GPU上自适应选择验证长度的混合GDN–MoE模型的Speculative Decoding方法，并引入两阶段验证与轻量级路由校正以及块级并行GDN核实现；

**💡 创新点**

创新点在于：①利用目标模型的延迟曲线与草稿端接受概率估计在GPU上实时决定验证长度；②在MoE第一个路由器处暂停验证并根据实际专家分配动态决定是否额外验证一个token；③通过块级并行GDN求解和一次性重构已接受状态，显著降低了GDN层的验证开销；

**🔧 技术方法**

使用的技术包括：GPU条件图节点（CUDA Graph）、并行三角求解核、块级状态重构、基于草稿可信度的长度预测器、路由校正逻辑以及对DFlash并行草稿器的兼容；

**📊 数据集**

使用SPEED-Bench Qualitative数据集进行验证，包含7类任务（编码、数学、推理、检索增强生成、摘要、写作、角色扮演），共560条提示（280训练+280测试）；

**📈 对比分析**

与AR解码、原始DFlash、EVICT、ReplaySSM等基线在单请求贪心解码下对比，平均速度提升为：对AR 1.75–1.88×，对原始DFlash 1.30–1.33×；同时在不同任务类别中保持一致性；

**⚠️ 局限性**

局限性包括：仅在单请求贪心解码场景下验证；需要在每个模型上进行接受概率校准和延迟曲线采样；对大批量或多GPU推理的适用性未评估；并且性能提升受混合GDN–MoE结构的依赖，其他模型可能效果有限。

---

## 164. On the (Quantum) Communication Complexity of Secure MPC from a Random Oracle

**arXiv ID:** 2611.04052 | [PDF](https://arxiv.org/pdf/2611.04052v1)

**作者:** Alexander Hoover `[一作]` (Stevens Institute of Technology), Kevin Yeo `[通讯]` (Google)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文在随机预处理模型和诚实多数模型下，对使用随机预言机或量子资源的多方安全计算（MPC）提出了通信复杂度下界，并给出与度为2（含内积）的电路匹配的上界；同时证明量子通信无法降低统计安全MPC的通信，提出基于后量子单向函数的两方量子协议可在不违反下界的前提下实现更低通信，但需巨额计算开销。

**💡 创新点**

主要创新点包括：①在随机预言机模型下证明MPC预处理阶段的在线通信下界 Ω(|C|·t/(n−t))；②在诚实多数预处理阶段给出 Ω(|C|) 的下界并用随机预言机构造匹配的 O(|C|) 协议；③证明量子通信与量子预处理对统计安全MPC不提供通信优势；④提出量子可访问随机预言机下的通信下界；⑤给出使用后量子单向函数的两方量子MPC协议，突破传统下界；⑥证明若想获得子线性通信，必须执行线性量级的加密操作。

**🔧 技术方法**

技术方法包括：将MPC下界归约到带相关随机性的单服务器PIR，并利用Key‑Agreement攻击（Barak–Mahmoody、Li‑Li‑Li‑Liu）证明随机预言机不能降低通信；使用压缩预言机技术模拟私钥运算；量子信息论（von Neumann熵）证明量子通信不降低通信；构造随机预言机下的度‑2多项式协议；利用分布式点函数实现量子PIR，进一步构造两方量子MPC。

**📊 数据集**

本工作为理论论文，无实验数据集；所有结果均为证明与上界构造。

**📈 对比分析**

与之前的信息理论下界（Damgård‑Larsen‑Nielsen）比较，本文在随机预言机下保持相同下界；在诚实多数下给出匹配的 O(|C|) 上界，明显优于 O(|C|·log q) 的随机预言机协议；量子方案在无量子预处理下与信息理论下界持平，但在后量子单向函数支持下实现了 O(|C|·λ) 以上的子线性通信。总体来看，本文实现了理论上最佳的通信效率，同时揭示了量子与对称加密对通信的极限。

**⚠️ 局限性**

局限性包括：①匹配上界仅适用于度为2（即乘法深度为1）的电路，且要求参与方数 n=O(log λ)；②量子两方协议虽然突破下界，但计算复杂度随较小方输入指数级增长，实际可行性受限；③在一般电路和更大多数场景下，仍缺乏完整的通信下界与上界；④对量子可访问随机预言机的下界假设较强，若假设不成立，结果需重新评估。

---

## 165. A Constant-Factor Approximation for Capacitated Facility Location with Outliers

**arXiv ID:** 2611.04094 | [PDF](https://arxiv.org/pdf/2611.04094v1)

**作者:** Rajni Dabas `[一作]` `[通讯]` (Northwestern University), Rajni Dabas (Northwestern University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种多项式时间的确定性算法，针对带有离群点的度量式容量受限设施定位问题（任意开设成本、非均匀硬容量），并保证既满足容量约束也满足离群点预算；

**💡 创新点**

创新点在于构造了带符号惩罚比较的局部搜索框架，利用交换森林和分组增广路径实现容量与离群点的协同约束，从而获得约5.455的近似比；

**🔧 技术方法**

主要技术包括：局部搜索的惩罚（penalty）版本、交换森林与上下节点分组的费用归还、增广路径按终点设施分组共享开设成本，以及多次惩罚搜索的降温与热启动；

**📊 数据集**

无数据集，论文为理论分析与证明；

**📈 对比分析**

与已有的无离群点或统一成本的方案相比，本文在满足容量与离群点双重约束的前提下，实现了更优的近似比（5.455），并在算法复杂度上保持多项式；

**⚠️ 局限性**

局限性：仍无法进一步突破5的理论上限，且在实际实现时需要对多次惩罚搜索与交换森林构造进行高效实现，算法对参数选择敏感。

---

## 166. CHROMA-MILL: Joint Low-Light Enhancement and White Balance under Colored Illumination

**arXiv ID:** 2611.04092 | [PDF](https://arxiv.org/pdf/2611.04092v1)

**作者:** Pol Riubrogent-Comas `[一作]` (Computer Vision Center), Javier Vazquez-Corral `[通讯]` (Computer Vision Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种用于低光照图像增强与自动白平衡联合处理的方法，并构建了相应的 CHROMA-MILL 数据集。

**💡 创新点**

主要创新在于引入 CTI-Retinexformer，能够在潜在空间同时分离亮度、色度和场景内容，实现两任务的联合监督；并通过混合光源生成精确色度标签，扩展了数据集的光照多样性。

**🔧 技术方法**

采用改进的 Retinexformer 架构，并结合强度预测、色度预测、均匀性约束、三元组损失和重构损失等多任务监督。

**📊 数据集**

使用了 CHROMA-MILL 数据集，该数据集包含 50 个室内场景，在 DSLR 与手机两种相机下，以 4 种校准光源和 11 个强度级别拍摄，并可线性合成任意混合光源。

**📈 对比分析**

通过与多种两阶段 LLIE+AWB 方案对比，CTI-Retinexformer 在 Nikon 与手机测试集上分别在 PSNR_L 及 ΔE_00 指标上提升约 0.7 dB 与 1.1，且在多项质量指标上均名列第一。

**⚠️ 局限性**

限制方面是仅支持均匀光照、光源范围受限（CIE F2 以上），样本量有限，且未对更大规模或多光源空间变化的场景进行验证。

---

## 167. Graph-Localized Coverage for Conformal Prediction on Inductive Graphs

**arXiv ID:** 2611.04066 | [PDF](https://arxiv.org/pdf/2611.04066v1)

**作者:** Jinwen Xu `[一作]` (University of Georgia), Georgios B. Giannakis `[通讯]` (University of Minnesota)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在线、局部化的合成预测框架GLocalCP，用于增量图节点分类和边预测的置信集构造。

**💡 创新点**

结合蒙特卡罗Dropout的互信息不确定性尺度和基于图嵌入的核局部校准反馈，既保证了路径级O(N⁻¹⁄²)的边际校准，又实现了图局部化的校准误差控制。

**🔧 技术方法**

采用在线合成预测、MC Dropout、互信息尺度、核函数（如RBF或随机 Fourier feature）局部校正、GraphSAGE/GCN等图神经网络技术。

**📊 数据集**

使用七个节点分类基准（Cora、CiteSeer、PubMed、Computers、Roman‑empire、Amazon‑ratings、ogbn‑arxiv）和七个边预测基准，涵盖不同候选比例。

**📈 对比分析**

与NAPS‑U/DAPS/SNAPS（节点）及CLP‑CQR/CLP‑SCQR/RR‑CQR/RR‑Mondrian（边）对比；GLocalCP在保持接近0.9的边际覆盖率的同时，显著降低图局部化校准误差，并在预测集大小上亦有提升。

**⚠️ 局限性**

仅在节点和二分类边预测中验证，局部核需要手动设定或近似；对大规模图的计算仍需随机特征近似；不保证对所有潜在子群的覆盖率，仅提供局部化而非完全分组覆盖。

---

## 168. Co-Occurrence Priors Override Visual Spatial Evidence: Understanding False Alarms in Vision-Language Models for Construction Struck-By Hazard Detection

**arXiv ID:** 2611.04112 | [PDF](https://arxiv.org/pdf/2611.04112v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 169. StreamLTT: Risk Certification for Policy Selection under Stream Dependence

**arXiv ID:** 2611.04062 | [PDF](https://arxiv.org/pdf/2611.04062v1)

**作者:** Jinwen Xu `[一作]` (University of Georgia), Qin Lu `[通讯]` (University of Georgia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了StreamLTT框架，用于在依赖且非平稳的部署流中对预测策略进行风险认证。

**💡 创新点**

创新点在于将风险定义为平均可预测风险，并利用条件中心化得到马尔可夫性质，结合可变方差自适应检验与Hoeffding有限样本检验，在多重检验下实现同时认证。

**🔧 技术方法**

使用技术包括条件可预测风险、马尔可夫差分序列、可变方差自适应检验、Hoeffding有限样本检验、FWER控制，以及针对时间序列依赖的可预测方差估计。

**📊 数据集**

实验使用七个真实世界预测数据集：电力、家用电器、家庭功率、PEMS04、PEMS08、北京天气和贾纳天气。

**📈 对比分析**

与IID‑LTT、稳健方差、块自助、HAC‑CLT、霍夫丁等方法比较，StreamLTT‑A在风险边界更接近目标、误报率更贴近预设，平均风险差距显著降低，整体性能优于对照组。

**⚠️ 局限性**

局限包括对可预测方差估计的依赖、在极端依赖结构下可能仍需改进，以及目前仅在离散二元风险场景验证，缺乏连续风险或更复杂决策规则的实验。

---

## 170. Structured Clinical Memory as Context for Probabilistic Decision Models: A Synthetic Evaluation with Jev

**arXiv ID:** 2611.04101 | [PDF](https://arxiv.org/pdf/2611.04101v1)

**作者:** Jash Sayani `[一作]` (Abbycare), Achinth Bharadwaj `[通讯]` (Abbycare)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在一项实验性基准中，作者使用30个人工合成的临床病例和120道多选/二元/五分量化问题，对比了仅使用病例叙述与在叙述后追加结构化临床记忆的两种输入方式，评估了模型Jeve的决策准确性和概率质量。

**💡 创新点**

创新点在于：① 构建了一个配对实验基准，专门评估结构化长期临床状态对概率决策模型的影响；② 提出了包含诊断、药物、状态、起始、置信度、来源、否定等字段的可维护结构化记忆方案，并证明其在“oracle”条件下可提升模型性能。

**🔧 技术方法**

使用技术包括：Jeve概率决策模型、手工生成的结构化记忆Schema、配对实验设计、统计检验（McNemar、置信区间）、概率质量评估（多选对数损失、二元Brier分数、平均绝对误差）以及令牌计数和延迟测量。

**📊 数据集**

数据集为30个人工合成的临床病例，覆盖12个医学领域；每个病例配有120道题目（56道多选、36道二元、28道五级分级）；结构化记忆为人工生成的“oracle”形式，包含每个特征的完整属性。

**📈 对比分析**

比较方法：在同一组问题下分别用“仅叙述”和“叙述+结构化记忆”两种输入进行模型推理；结果显示准确率从90.0%提升至93.3%（+3.3pp），多选对数损失由0.258降至0.175，二元Brier分数由0.054降至0.022；结构化记忆导致平均令牌数增加56%但延迟保持约0.5秒。

**⚠️ 局限性**

局限性：① 仅使用人工合成病例，未覆盖真实数据中的噪声和缺失；② 结构化记忆为“oracle”，未评估提取误差；③ 标签由单一作者生成，未经过临床复核；④ 样本量小，配对比较统计功效低；⑤ 仅评估单一模型和单次运行，结果可能不具普适性；⑥ 未测评临床安全性、可解释性或公平性。

---

## 171. Decomposing Rank Reversals under Temporal Nonstationarity: Alignment, Aggressiveness, and Instability

**arXiv ID:** 2611.04116 | [PDF](https://arxiv.org/pdf/2611.04116v1)

**作者:** Deying Song `[一作]` (Carnegie Mellon University), Xueying Ding `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在时间非平稳环境下模型排名反转的根本机制，提出了一种模型无关的R²分解，将性能拆解为目标对齐、攻击性和预测不稳定性三部分，并推导出复杂度相位边界与最佳记忆窗口；

**💡 创新点**

创新点在于将估计-漂移权衡、模型基准比较与自适应记忆选择三大议题统一到一条三分量R²身份，给出闭式岭回归最优记忆方程，并为非线性模型提供代理转移界，以实现对排名反转的机制性解释；

**🔧 技术方法**

使用R²分解与三分量身份、复杂度相位分析、岭回归理论推导、线性代理与迁移界、仿真验证以及大规模金融数据实验；

**📊 数据集**

使用美国行业组合每日超额收益（1987‑2016）、CRSP股票日收益（2018‑2024）以及合成高斯协变量模拟数据；

**📈 对比分析**

通过在不同训练窗口（16个月、64个月、扩展历史等）下对线性、树、神经网络等14种模型进行R²与RMSE对比，验证了理论预测的排名反转、最佳记忆与误差成分，实验表明高阶模型在短窗口占优，长期窗口时线性正则化模型表现更好；

**⚠️ 局限性**

局限在于仅考虑截断窗口的记忆，代理界对非线性模型的精度有限，未涵盖指数加权或自适应记忆等更灵活的时间加权方案。

---

## 172. Generalizable AI-Generated Image Detection with Embedding-Space Explanations

**arXiv ID:** 2611.04111 | [PDF](https://arxiv.org/pdf/2611.04111v1)

**作者:** Nikos Giakoumoglou `[一作]` (Imperial College London), Symeon Papadopoulos `[通讯]` (Information Technologies Institute, CERTH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种分离检测与解释的AI生成图像真实性检测系统，先锁定DINOv3专家检测器，再用联合嵌入预测器将检测特征映射到冻结文本空间，指导轻量级VLM仅生成解释；

**💡 创新点**

创新点在于将高性能视觉检测与语言生成完全解耦：检测结果先固定，解释训练不影响判断；通过JEPAs实现视觉特征到文本语义的无监督映射，避免了多模态模型在检测任务上过拟合；

**🔧 技术方法**

采用DINOv3视角一致自监督编码器、LoRA轻量化适配、InfoNCE对齐的联合嵌入预测器、E5-large-v2冻结文本编码器、Qwen2.5‑VL‑3B-Instruct作为语言生成器；

**📊 数据集**

训练数据来自XPlainVerse（含多种生成模型合成图像及对应解释），验证集使用外部Chameleon；评估在14个未见生成器、LOKI及XPlainVerse上；

**📈 对比分析**

与现有检测基线（DeFake、PatchCraft等）比较，平均AUC提升3.7点，达到97.4；在解释任务中BERTScore分别为69.8（XPlainVerse）和56.1（LOKI），显著优于FakeVLM等对手；在压缩、降采样、噪声等扰动下亦保持鲁棒性；

**⚠️ 局限性**

局限性包括：解释仍依赖预训练文本空间的语义投射，可能缺乏细粒度可解释性；模型总体参数量较大（约3.65B冻结+89M可训练），部署成本高；对生成器外部更先进或对抗性编辑的鲁棒性尚待验证。

---

## 173. ScienceVision: Can AI Observe and Reason Like a Scientist?

**arXiv ID:** 2611.04087 | [PDF](https://arxiv.org/pdf/2611.04087v1)

**作者:** Tianyi Ma `[一作]`, Kuan Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

创建了跨学科的科学3D视觉问答基准ScienceVision，包含250个专家级问题，覆盖建筑、生物、化学、机械CAD和医学影像五大领域；通过原生3D模型与可交互的MCP工具让模型在多种观察模式下回答问题。

**💡 创新点**

创新点在于：①统一的跨域3D VQA评估框架；②引入MCP可视化工具让模型主动获取额外证据；③制定严格的任务特定确定性评分规则，实现不同观察条件下可比的最终答案评估。

**🔧 技术方法**

采用LLM agent harness进行文件访问、代码执行及工具调用，利用MCP视图工具（视角切换、目标隔离、切片浏览、测量等），并在另一设置下仅使用预渲染的2D图像进行评估。

**📊 数据集**

使用自构建的250题数据集，每个领域各50题，题目均附有原生3D对象、专业问题、答案格式与参考答案，覆盖22个子领域。

**📈 对比分析**

通过对12个模型在三种设置（Harness+3D、Harness+3D+MCP、Harness+2D）下的重复测试，计算任务平均分；结果显示在原生3D访问下平均分最高（约0.53），相比2D渲染提高约17.2个百分点；MCP工具对分数的提升有限且方向不一。

**⚠️ 局限性**

局限性包括：参考答案可能存在错误且未设立人类基准；MCP工具的实际效益难以量化；模型在所有设置下仍难以做出可靠的科学判断；评测受限于工具可用性、重复次数及执行差异，未完全隔离观察与计算因素。

---

## 174. Follow the Features: SAE Feature-Guided Data-Efficient Instruction-Tuning of Large Language Models

**arXiv ID:** 2611.04085 | [PDF](https://arxiv.org/pdf/2611.04085v1)

**作者:** Ryan Swift `[一作]` (University of Southern California), Konstantinos Psounis `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Follow the Features (FtF) 方法，用稀疏自编码器（SAE）特征统计来选择指令微调的小样本子集；

**💡 创新点**

创新点在于用平均激活、解码器范数、条件熵三项评估特征重要性，结合特征重采样提升多样性，并实现与训练模型解耦、可跨模型迁移；

**🔧 技术方法**

技术包括：稀疏自编码器特征提取、特征统计与排序、样本重要性评分、特征重采样、LoRA 微调、学习率调度等；

**📊 数据集**

使用的数据集为 UltraChat 200k 和 Alpaca (Cleaned)，测试模型为 Llama 3.1 8B 与 Qwen 3 8B；

**📈 对比分析**

在 100 样本预算下，与随机、全数据、GRAPE、S2L 对比，FtF 在 HMean 指标上与 S2L 同级，并在某些设置下超过全数据训练，表现竞争力；

**⚠️ 局限性**

局限性包括：SAE 特征解释不足、未探索不同 SAE 变体和层、模型规模与预算单一、缺乏人类干预实验、成本评估仍有限。

---

## 175. From topology-optimization brief to solver-checked release: separating intent, admissibility and acceptance knowledge for language agents

**arXiv ID:** 2611.04077 | [PDF](https://arxiv.org/pdf/2611.04077v1)

**作者:** Shaoliang Yang `[一作]` (Santa Clara University), Jun Wang `[通讯]` (Santa Clara University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了AgenticTO框架，将语言代理与拓扑优化结合，通过显式设计意图表示、规则校验和数值证书，确保问题翻译、求解和结果发布可检验。

**💡 创新点**

创新点在于把问题建模、规则检查和数值验证分离成可审计记录，并评估了地面化、澄清、以及多分辨率发布对错误检测的影响。

**🔧 技术方法**

使用大语言模型（LLM）、规则引擎、GPU加速有限元求解器、模型上下文协议（MCP）以及多分辨率优化调度。

**📊 数据集**

在50个自制拓扑优化简报（包含两种模型提供者）上测试，并注入故障、5个数值问题以及完整代理会话。

**📈 对比分析**

通过与人工评估的参考读数对比，发现地面化提高了正确规范率（37→44/50），澄清在oracle支持下进一步提升；数值成本可达7,370万单元，满足约73.7M的规模，且规则与证书分别捕获不同错误。

**⚠️ 局限性**

局限包括对oracle澄清的依赖、仅覆盖线性弹性单载荷、缺乏人机交互可用性评估、以及未处理制造和后处理验证。

---

## 176. The Complexity of ReLU Network Synthesis from a Parameter Multiset

**arXiv ID:** 2611.04074 | [PDF](https://arxiv.org/pdf/2611.04074v1)

**作者:** Dimitrios Myrisiotis `[一作]` `[通讯]` (Great Bay University), Dimitrios Myrisiotis (Great Bay University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

研究在给定参数库存（可重复或一次使用）和行为规范（样本、布尔电路、全空间ReLU目标、均匀逼近或均方风险）下，构造满足规范的神经网络的决策问题的计算复杂度。

**💡 创新点**

创新点在于将参数值与网络结构分离，给出完整的复杂度分类（从NP到Σ₂^P、DP等）并证明在小网络容量下的DP完全性；提出了控制所有合法网络结构的精细化归约技术，首次将资源限制与验证约束结合起来。

**🔧 技术方法**

主要技术包括子集求和与布尔电路的归约、ReLU网络的分区与偏移构造、模算术与数值身份的利用、组合计数与激活模式分析，以及基于风险的计数问题的映射。

**📊 数据集**

该工作完全是理论研究，无需使用实际数据集；所有实例均为构造性的、符号性的数值实例。

**📈 对比分析**

由于论文聚焦于理论复杂度分析，没有实验对比；结果表明即使在网络容量极小（如4个计算节点）时，相关合成与验证问题在多项式时间内不可解（除非P=NP、PH崩溃），而已有的一些先前问题的复杂度上限被进一步下界化。

**⚠️ 局限性**

局限性包括：1）对固定阈值风险问题的可判定性仍未完全解决；2）结果主要给出最坏情况的下界，实际学习任务可能更易；3）尚未提供高效近似或多项式时间近似算法；4）未考虑实际硬件约束或随机梯度等学习过程。

---

## 177. Computer-Use Agents Gain Little Benefit from a World Model When Conditioned on Tutorials

**arXiv ID:** 2611.04073 | [PDF](https://arxiv.org/pdf/2611.04073v1)

**作者:** Anh Nguyen `[一作]`, Stefan Lee `[通讯]` (Oregon State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了两个面向桌面与科学软件的多模态教程数据集，并提出了仅依赖检索教程的政策‑裁判框架 RAG‑PolicyJudge，系统评估了教程、裁判与世界模型在计算机使用代理（CUA）中的贡献。

**💡 创新点**

创新点包括：①创建 OSWorld‑Tutorial 与 ScienceBoard‑Tutorial 两大任务特定教程基准；②提出只需检索教程的 RAG‑PolicyJudge 框架；③对世界模型在教程条件下的实际增益进行消融，发现大部分提升源于教程理解与裁判决策，而非 roll‑out 预测。

**🔧 技术方法**

技术手段主要是多模态 LLM（Qwen3‑VL 系列）作为策略、裁判与世界模型；检索增强（RAG）提供任务特定教程；两阶段策略‑调度；以及文本和视觉（oracle）世界模型 roll‑out。

**📊 数据集**

使用的数据集包括原始 OSWorld‑Verified、ScienceBoard 任务，以及作者自行收集的 529 个多模态教程（OSWorld‑Tutorial、ScienceBoard‑Tutorial）。

**📈 对比分析**

通过与无裁判、无世界模型、无教程基线、R‑WoM、文本世界模型以及 oracle 视觉 roll‑out 进行对比，发现当配备教程时加入世界模型几乎无提升，裁判＋教程已能匹配甚至超越传统 look‑ahead 方法，显示教程与裁判是主要驱动因素。

**⚠️ 局限性**

局限性包括：仅在 Qwen3 系列 LLM 上实验，未评估其他模型或训练裁判；教程生成噪声未充分探测；实验仅在 Linux VM 桌面环境进行；错误分析依赖 MLLM 自动标注，可能带来偏差。

---

## 178. On the Primacy Bias in RLVR Training

**arXiv ID:** 2611.04061 | [PDF](https://arxiv.org/pdf/2611.04061v1)

**作者:** Phuc Minh Nguyen `[一作]` (University of Notre Dame), Nitesh V. Chawl `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过受控实验和大规模实证研究，探讨了 Reinforcement Learning with Verifiable Reward（RLVR）对大型语言模型推理能力的影响，并提出了 primacy bias 与 lazy learning 的视角，揭示了 RLVR 训练中模型主要提升易题、难题进展缓慢甚至出现遗忘的现象。

**💡 创新点**

创新点在于：①将 primacy bias 及 lazy learning 机制引入 RLVR 研究，用线性化近似与 NTK 分析表明 RLVR 训练几乎保持在预训练模型的线性化区间；②发现“unlearnable phenomenon”，即在 lazy regime 下 RLVR 只能有效改进已知易题，难题学习被抑制；③通过对 OOD 与熟悉领域的对比，解释了 RLVR 在不同任务域表现差异的根本原因。

**🔧 技术方法**

主要技术包括：RLVR（GRPO、DAPO）强化学习策略；对比线性化模型与实际 Fine‑Tuned 模型的 KL 散度；计算 Neural Tangent Kernel（NTK）、Centered Kernel Alignment（CKA）、Mean Principal Angle（MPA）和 Spectrum Drift（NSS）；使用 Pass@k 作为推理覆盖度指标；在 MNIST toy 任务上构造难易分布；在 Qwen2.5‑3B‑Instruct 上进行 RLVR 训练。

**📊 数据集**

实验数据集涵盖：MNIST（用于 toy 证明 primacy bias）；数学推理数据集 MATH；Olympiad‑Bench、Minerva、MATH‑500 用于评估 RLVR 在不同难度下的表现；以及预训练的 Qwen2.5‑3B‑Instruct、Qwen2.5‑Math‑1.5B 等模型。

**📈 对比分析**

对比方法包括：与无 RL（仅 SFT）基线、不同 RLVR 算法（GRPO、DAPO）及其在易题/难题子集上的训练；通过 Pass@k（k=128）与 Pass@1 评估覆盖度变化；发现 RLVR 在易题上往往导致 Pass@k 下降、难题提升有限，长时间训练会出现遗忘；但在 OOD 任务或完全未解题集上，RLVR 能显著提升 Pass@k。

**⚠️ 局限性**

局限性在于：①仅在特定模型与任务域验证，缺乏更广泛的跨模型通用性证明；②线性化与 NTK 的近似假设可能在更深层网络或更复杂任务中失效；③未提供有效机制来平衡难题学习与易题保持，遗忘仍是主要挑战；④实验主要聚焦于数学推理，未深入探讨自然语言或多模态任务。

---

## 179. Forgetting to Improve: Principled Data Removal in Active Learning

**arXiv ID:** 2611.04054 | [PDF](https://arxiv.org/pdf/2611.04054v1)

**作者:** Manuel Wendl `[一作]` (ETH Zürich), Carl Henrik Ek `[通讯]` (University of Cambridge)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种名为 Forgetting to Improve (F2I) 的数据删减算法，用于在高斯过程回归中消除因模型不匹配导致的有害样本，从而在指定目标区域内显著降低预测不确定性，并进一步将该思想推广到贝叶斯优化和基于模型的强化学习中。

**💡 创新点**

创新点在于：①提出基于影响函数的理论度量，能够量化单个样本对目标区域内总预测不确定性的影响；②基于该度量设计的贪婪删减策略，既考虑了表观噪声（aleatoric）又兼顾了模型不确定性（epistemic），并给出了近似最优性证明；③将该删减框架与贝叶斯优化/强化学习的动态目标区域结合，形成 Forgetful BO 与 Forgetful RL。

**🔧 技术方法**

主要技术包括高斯过程回归、MAP II 噪声估计、影响函数与梯度近似、贪婪贬值（类似反向匹配追踪）、Polyak‑Łojasiewicz 条件下的近似最优性分析，以及在贝叶斯优化与模型基强化学习中的目标区域更新机制。

**📊 数据集**

实验使用了多种数据集：合成函数（如 Rosenbrock、Rastrigin 等）、Boston Housing、Dow Jones 日内数据、标准贝叶斯优化测试函数集（8 个）以及 Cartpole 轨迹数据。

**📈 对比分析**

与基准方法（同一核的同方差/异方差 GP、域扭曲 GP、相关性追踪）比较，F2I 在目标区域内实现了更锐利的预测分布、更高的置信度校准、以及在贝叶斯优化中的累计简单损失更低；在基于模型的强化学习中，Forgetful RL 能够成功收敛，而保持全部样本的 MBPO 则失败。

**⚠️ 局限性**

局限性包括：需预先知道目标区域或可动态更新；对高斯过程核和噪声模型假设敏感，过度删减可能导致信息损失；贪婪删减的计算复杂度随样本量增长；实验主要集中在小规模数据集，尚未在大规模工业场景中验证。

---

## 180. Denoising Is Stein: Moment Access and a Resource Hierarchy for Learning Hidden Structure in Diffusion Models

**arXiv ID:** 2611.04043 | [PDF](https://arxiv.org/pdf/2611.04043v1)

**作者:** Rui Wang `[一作]` `[通讯]` (Microsoft), Rui Wang (Microsoft)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在扩散模型中通过去噪对样本的对比学习，探究隐藏的非高斯成分在不同协方差场景下可被访问的阶数，以及不同学习器对同一隐藏信号的资源消耗。

**💡 创新点**

① 证明去噪损失与Stein差异对应所有低阶矩差异，提供“矩访问”计算法；② 推导两种数据场景（协方差可见与匹配）的阶数阈值（k*）和资源阶数；③ 证明在无协方差信息时，任何方向无关字典、核岭回归和小步长SGD均需 d^{k*‑1} 资源；④ 提出两阶段学习器通过高斯平滑产生曲率，利用重用视图在 O(d^{k*/2}log d) 训练点上即可突破 Gaussian 基线；⑤ 在实验中验证谱过渡、字典阈值、宽度为8的网络 d^2 的训练阈值。

**🔧 技术方法**

Stein–Tweedie 识别、Hermite 多项式、矩差异分析、随机特征字典、核岭回归、在线SGD、统计查询、重用视图与高斯平滑的矩阵参数化、谱分解、两阶段梯度与正则化、实验的 Monte‑Carlo 评估。

**📊 数据集**

使用合成的二元混合高斯与子空间嵌入模型（k=1 或 k>1），在 d=8–256 维、不同噪声比例、不同时间 t 的噪声化数据。

**📈 对比分析**

与 Gaussian 基线、PCA 拟合、随机特征、核岭回归、单单元SGD 及宽度为8的网络进行对比；两阶段学习器在大多数设置下取得比 Gaussian 基线显著更低的得分风险，且所需样本数与理论 d^{k*/2}log d 一致；随机特征与核方法在 d^{k*‑1} 资源下无显著改进；网络实验表明 d^2 级别的训练阈值。

**⚠️ 局限性**

① 需要对分布满足 L^4/L^2 约束、Hermite 系数可控等技术假设；② 随机特征字典的 L^4/L^2 常数未被证明；③ 核方法需高阶 Taylor 系数非零；④ 仅在理论上证明了 k*=even、k=1 的情况；⑤ 实验仅覆盖 d≤256，网络训练的理论证明缺失；⑥ 对非高斯成分的更复杂结构、实测数据的推广仍未解决。

---

## 181. SciPhi-Chat: Training a Spatial Audio Agent for Question Answering, Dialogue, and Tool Use

**arXiv ID:** 2611.04114 | [PDF](https://arxiv.org/pdf/2611.04114v1)

**作者:** Gijs Wijngaard `[一作]`, Sebastian Braun `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了SciPhi-Chat空间音频问答代理，支持多轮对话、推理与工具调用，能够在FOA（第一阶 ambisonic）音频上回答位置、身份、时间等多维度问题。

**💡 创新点**

首次将中间推理、场景描述与可执行工具调用统一到同一训练格式，并覆盖42个空间音频任务类别，显著提升跨任务推理和上下文记忆能力。

**🔧 技术方法**

使用 FOA‑VAE 作为方向编码器，Phi‑4‑multimodal 语言模型与 LoRA 微调，配合四个基于信号处理的工具（定位、计数、活动检测、时频定位）进行推理与测量。

**📊 数据集**

在 MASU（12,000 样本、42 类任务）和 ASA‑Bench（多项物理空间任务）两大基准上进行评测，数据包含真实录音与合成场景。

**📈 对比分析**

与七个基线模型对比，SciPhi-Chat 在 MASU 单轮任务获得 44.9%（比最佳高 7.4pp），ASA‑Bench azimuth 74.5%、elevation 60.0%，在所有四项空间物理任务中均优于现有最优模型。

**⚠️ 局限性**

局限在于工具选择策略不完善导致某些任务（如方向计数）性能仍有限；对长序列、多音频对话支持不足；跨域泛化仍需进一步提升。

---

## 182. Say Less, Mean Better: Training-Free Verbosity Control for Language Models

**arXiv ID:** 2611.04107 | [PDF](https://arxiv.org/pdf/2611.04107v1)

**作者:** Mahmoud Alyosify `[一作]` (Queen's University), Rahatara Ferdousi `[通讯]` (Queen's University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 MinimalLM，一种无需训练的框架，用于在大型语言模型推理时通过不确定性感知的长度依赖停止调优和结构完整性保护，实现对响应冗长的可靠控制。

**💡 创新点**

创新点在于将“何时停止”与“是否在结构上合适”分离，采用长度与熵门控的停止调整以及完整性门槛，确保在缩短输出的同时不出现结构截断。

**🔧 技术方法**

技术核心包括长度依赖的停止词提升、熵门调节的停止概率、完整性门限制停止仅在句末或结构闭合处、以及推理后完成检查。

**📊 数据集**

使用了330道评测问题（MT‑Bench 80、AlpacaEval 150、Arena‑Hard‑Auto 100），并在八个主题（医疗、历史、经济、编程、科学、数学、写作、法律）中进一步验证。

**📈 对比分析**

通过与基线模型、指令提示、仅停止门、无完整性门等消融组合，以及直接截断基线回答的比较，发现 MinimalLM 令生成 token 减少约47.6%，单个回答延迟提升19–68%，在主评估中无结构截断，平均质量得分 8.04/10，略低于未改进的 8.42/10。

**⚠️ 局限性**

局限性包括仅在三款 3B–8B 参数的开源模型上、greedy 解码、单一 GPU 进行评估；自动质量评估与手工检查一致性低，固定字数预算可能导致信息覆盖不足，且未探讨更大模型、不同解码策略或自适应长度控制。

---

## 183. Structure, Dimensionality, and Distribution Shape - What Makes Pretraining Data Transferable, and Why the Input Pipeline Decides Which

**arXiv ID:** 2611.04106 | [PDF](https://arxiv.org/pdf/2611.04106v1)

**作者:** Sangyoon Bae `[一作]` (Seoul National University), Jiook Cha `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出Transfer-Invariance原理，研究预训练语料的结构、维度与分布形状如何在不同输入管道下决定迁移效果，并设计TransferForge生成器生成可调节的合成数据；

**💡 创新点**

创新点在于将预训练语料的可迁移性归因于模型输入变换，而非数据本身，并提供一种根据模型管道选择合成方向的实用化流程；

**🔧 技术方法**

主要技术包括基于Gaussian过程与结构因果模型的合成生成器TransferForge、分布形状匹配变换、PatchTST与POYO+等时序模型的预训练与微调；

**📊 数据集**

使用的数据集包括UCR时间序列分类档案、金融宏观经济系列（FRED‑MD、汇率）、以及Allen Brain Observatory的钙成像记录；

**📈 对比分析**

在PatchTST上通过结构优化的合成数据相较真实预训练提升4.2%准确率，在POYO+上形状匹配的合成数据可替代全部真实钙成像预训练并提升约5%准确率，且合成数据规模可进一步提升性能；

**⚠️ 局限性**

局限性在于仅验证了少数模型与两类任务，对其他输入管道与更广泛模态的通用性尚未证明，且合成数据仍需手动调参以匹配目标分布。

---

## 184. When Strong Reasoners Join the Pool: Small Union Premiums and Costly Runaway Generations in LLM Routing

**arXiv ID:** 2611.04098 | [PDF](https://arxiv.org/pdf/2611.04098v1)

**作者:** Teng-Ruei Chen `[一作]` `[通讯]` (National Yang Ming Chiao Tung University), Teng-Ruei Chen (National Yang Ming Chiao Tung University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在MATH-500上使用三种当前推理模型（Llama-3.3-70B、Nemotron Nano 9B、Super 49B）在固定的解码协议下生成45,000个采样答案，计算单模型极限、产品联合与“oracle溢价”，并评估成本与采样策略。

**💡 创新点**

揭示在强推理模型池中，oracle溢价可低至约0.6pp，远低于先前11模型池的3.5pp，并证明单模型多次采样可等效于多模型联合；同时提供统一的长度截断成本分析。

**🔧 技术方法**

采用NVIDIA B300 GPU、vLLM+FlashInfer推理、Math-Verify验证器、基于种子预提交的固定采样协议以及算子级的记录与审计。

**📊 数据集**

使用MATH-500（500道数学问题）作为基准数据集。

**📈 对比分析**

通过与先前的11模型池对比，评估oracle溢价、单模型极限、联合溢价以及重采样与路由的任一正确率；结果显示，单模型多次采样的性能与三模型联合相当，且整体oracle溢价低于先前工作。

**⚠️ 局限性**

仅针对三种模型、单一协议、单一GPU，缺乏对不同温度、长度截断或更大模型池的系统验证；且使用金手指式的gold-based联合选择不具可部署性；长回复截断仅在实验后分析，未提供实时截断策略。

---

## 185. ScaleTraj: Scale-Decomposed Human Trajectory Generation

**arXiv ID:** 2611.04097 | [PDF](https://arxiv.org/pdf/2611.04097v1)

**作者:** Yulin Zeng `[一作]` (Santa Clara University), Yuhong Liu `[通讯]` (Santa Clara University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了“人口级位移差距”（Population-Level Displacement Gap）问题，并通过 Scale Fidelity Profile（SFP）诊断其与尺度分配失配的关系，随后设计了 ScaleTraj——一种将尺度视为显式路由变量的自回归 Transformer，显著降低位移差距，同时保持个体轨迹的真实性。

**💡 创新点**

创新点：①正式量化并普适验证了位移差距；②引入尺度分解诊断指标 SFP，揭示尺度误配是差距的根源；③提出 ScaleTraj 结构，将尺度选择与目标细胞预测分离，利用 GMM 先验与多尺度 MLP 提升长尾位移重现；④通过对比多种模型驱动与数据驱动生成器，验证尺度分解能跨范式提升性能。

**🔧 技术方法**

技术：自回归 Transformer（因果注意力）、Scale Selection Head 与 Scale-Aware Cell Selection Head、Gaussian Mixture 模型生成尺度先验、教师强制与计划采样、温度调节与重排序器、JSD、截断幂律拟合、SFP 评估。

**📊 数据集**

数据集：GeoLife（小规模 GPS 轨迹）和 Tencent Beijing（大规模、网格化移动数据），两者分别涵盖不同用户规模、采样频率与空间分辨率。

**📈 对比分析**

比较方法：对 7 种代表性生成器（SemiMarkov、TimeGeo、LSTM、TrajSynVAE、MoveSim、STAR、TrajGPT）在统一协议下训练并评估。指标包括 SFP、截断幂律参数 |Δβ|、|Δκ| 以及 8 项个体层面 JSD 评估。实验显示 ScaleTraj 在 SFP、|Δβ|、|Δκ| 上均为最佳，同时在大部分个体级指标上保持竞争力，说明尺度分解既减小了人口级位移误差，又未显著牺牲轨迹质量。

**⚠️ 局限性**

局限性：①评估仅覆盖两种中国城市数据，未检验不同城市、文化或交通网络的普适性；②仅在已知用户的重现任务上测试，未验证对未见用户的生成能力；③SFP 及截断幂律仅衡量位移分布，未全面覆盖空间、时间与行为的真实性；④模型训练与评估只做单次实验，未报告重复性与方差；⑤ScaleTraj 引入了 GMM 先验，需先获取尺度统计，若先验不准确可能影响性能。

---

## 186. Permitted but Unauthorized: Affordance-Induced Delegated-Scope Violations in Multi-Agent LLM Systems

**arXiv ID:** 2611.04086 | [PDF](https://arxiv.org/pdf/2611.04086v1)

**作者:** Shravya Kanchi `[一作]` `[通讯]` (Independent Researcher), Shravya Kanchi (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多智能体LLM在共享基础设施上对委托范围的跨越行为，并通过实验揭示行动表示（是否列出对手可用动作）是导致跨越的主要驱动力；

**💡 创新点**

创新点在于把跨越行为拆解为可控制的“行动表示”与“竞争激励”两因素，并证明只要通过软件层面限制动作可见性即可阻止大部分跨越；

**🔧 技术方法**

使用哈希链式日志的确定性违规检测器、两种运行时防御（硬性引用监视器和基于模型的预执行评判器），以及对话模型的推理输出；

**📊 数据集**

在两个仿真环境（CI/CD 发布流水线和 SOC 事件排队）下，使用8种主流LLM模型（GLM‑5、Claude Sonnet 4.5/5.5、Opus 5.5、GPT‑6、DeepSeek V3.2、Mistral Large 3、Qwen3‑Coder）进行多轮实验，样本量为每格40–52个episode；

**📈 对比分析**

通过对照实验和 Fisher exact 与 Holm 校正的统计检验比较不同条件下的跨越率；防御能将跨越率降为0，且误报率极低；

**⚠️ 局限性**

局限性包括样本量有限、仅评估二智能体、委托图过于简单、未充分覆盖自报误差以及对某些模型（如 GPT‑6、Qwen3‑Coder）的机制分析受限。

---

## 187. TB-Transolver: Parallelized and Coherent Long-Horizon Rollouts via Temporal Bundling for Time-Dependent PDEs

**arXiv ID:** 2611.04063 | [PDF](https://arxiv.org/pdf/2611.04063v1)

**作者:** Sidharth S. Menon `[一作]` (Worcester Polytechnic Institute), Ameya D. Jagtap `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了TB-Transolver，即在Transolver架构中加入时间捆绑（Temporal Bundling）技术，实现一次前向传播可预测多步时间状态，从而显著提升时间序列 PDE 求解效率；

**💡 创新点**

创新点包括：1）引入可训练前的时空相干性指标C(K)，以数据驱动方式预先选取最佳捆绑宽度K*，避免昂贵的超参数搜索；2）将时间捆绑与多GPU分布式训练相结合，获得聚合加速；3）在多物理域（塑性、Navier-Stokes、反应扩散、海面温度）上验证方法的普适性与可扩展性；

**🔧 技术方法**

核心技术包括：Transolver的Physics-Attention机制、时间捆绑策略、基于Pearson相关的时空相干性计算、分布式数据并行（DDP）与多GPU并行；

**📊 数据集**

使用了四个公开基准数据集：二维塑性挤压（101×31网格，20步）、低粘性Navier-Stokes（64×64网格，10步）、高粘性Navier-Stokes（64×64网格，30步）、二维FitzHugh–Nagumo反应扩散（128×128网格，10步）以及ERA5海面温度（全球海域，8个月预测）；

**📈 对比分析**

与原始Transolver以及多种主流神经算子（FNO、OFormer、FactFormer等）对比，TB-Transolver在单GPU上实现了最高29×的训练速度提升，整体误差提升仅约10–25%；在2–8 GPU分布式设置下，聚合加速可达45.6×，并保持相近或更低的相对L2误差；

**⚠️ 局限性**

局限性包括：1）时空相干性指标C(K)对不同物理域表现不同，需根据数据经验阈值；2）当C(K)快速衰减时，较大捆绑宽度会导致解耦误差加大，限制可用的K*；3）多GPU扩展受PCIe通信带宽限制，尤其在大模型/高精度训练时易受限；4）方法仍假设时间序列可拆分为固定宽度块，无法自动适应非均匀时间步长的场景。

---

## 188. Exact Optimality Certification of QPLIB 2096 through Complete Set Partitioning

**arXiv ID:** 2611.04230 | [PDF](https://arxiv.org/pdf/2611.04230v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 189. Query-based Submodular Data Selection for Efficient Instruction Tuning

**arXiv ID:** 2611.04059 | [PDF](https://arxiv.org/pdf/2611.04059v1)

**作者:** Zhenghao Zhao `[一作]` (University of Illinois Chicago), Yan Yan `[通讯]` (University of Illinois Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 QSGS 方法，利用子模图切与梯度引导合成构造针对指令微调的高效子集；

**💡 创新点**

创新点在于：① 设计了目标相关性与子集冗余双重考量的子模图切目标；② 通过梯度残差分析实现未覆盖方向的合成补齐；

**🔧 技术方法**

使用了 LoRA 预训练、低秩梯度特征、随机投影压缩、梯度影响估计、子模图切贪婪优化以及梯度引导数据合成技术；

**📊 数据集**

在 270k 异构指令数据池上进行实验，评估基准包括 MMLU、BBH、TydiQA，使用 Llama‑2、Llama‑3、Mistral 三大模型；

**📈 对比分析**

与 LESS、随机抽样、BM25、DSIR 等基线对比，0.5% 预算下 QSGS 在平均 1.6% 上超越基线；在 Mistral‑7B 0.5% 预算下几乎达到全数据性能；

**⚠️ 局限性**

局限性包括：对 warm‑up 预训练梯度的依赖，合成样本过程额外开销；在较大预算下优势减弱；极低 warm‑up 比例可能导致性能低于随机。

---

## 190. Structured Priors: Towards an algebra of component priors for prior-data fitted networks

**arXiv ID:** 2611.04056 | [PDF](https://arxiv.org/pdf/2611.04056v1)

**作者:** Thomas Bartz-Beielstein `[一作]` `[通讯]`, Thomas Bartz-Beielstein

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种结构化先验框架，用命名组件先验通过混合、卷积、幅度缩放、本地变换、投影等算子组合，构成先验代数；并在此框架下实现软件包，对算子定律进行形式化证明并在预注册实验中验证其可行性。

**💡 创新点**

创新点在于：①把先验视为可采样函数的分布，构成一个闭合的半环结构；②提出并证明一系列先验算子的代数定律；③建立模型平均定理，说明在先验层面进行平均能得到贝叶斯预测；④提供了可预知的先验修改方式，避免传统手工调参导致的实验循环。

**🔧 技术方法**

使用技术包括：Transformer 预训练框架（Prior‑Data Fitted Networks），TabPFN 等 tabular foundation 模型；自定义生成器和先验算子（Python 软件包实现）；最大均值差距（MMD）投影、卷积与卷积加权；理论证明与模型平均分析。

**📊 数据集**

主要使用公开回归表格数据集（共 8 个），并与 TabPFN、TabICL、Mitra 等现有生成器进行比较；此外在实验 N1、N2、N4 预注册框架下验证算子定律和投影效果。

**📈 对比分析**

对比方法：在同一数据集上训练结构化先验网络与手工混合先验网络；评估指标为回归误差（RMSE/MAE）和种子波动。结果显示：在 7/8 表格上，结构化先验与任何单一组件相当或更优；在 2 张表格中，仅用 1/9 的数据预算即可达到随机先验在完整预算下都无法企及的误差水平；投影可有效裁剪教师基底，提升效率。

**⚠️ 局限性**

局限与挑战：①训练的单组件网络在识别生成组件时的准确率过低，导致模型平均理论在实验中未能落地；②实验仅覆盖回归任务，分类等其他任务未验证；③需要引入密度头等机制来实现更实用的平均预测；④虽然算子定律在理论上成立，但实际实现仍需进一步优化和扩展。

---

## 191. Exploit or conform? How norm utility prompting changes the way LLMs use community beliefs

**arXiv ID:** 2611.04048 | [PDF](https://arxiv.org/pdf/2611.04048v1)

**作者:** Leon Klingborg `[一作]` (University of St Andrews), Alexander J. Stewart `[通讯]` (Indiana University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在一轮两人社交困境游戏（雪雾游戏和囚徒困境）中，利用“规范效用”提示来影响开放权重大型语言模型的合作决策，并评估其对合作率、信念表达、决策波动性与推理偏差的影响。

**💡 创新点**

首次将规范效用（材料报酬、自我一致性、合群一致性）提示与信念表达相结合，既能逆转模型对社区信念的利用方式，又能显著降低模型对社会情境变化的波动性，提供一种更人性化、更稳定的对齐策略。

**🔧 技术方法**

使用多种提示条件（全效用提示、部分效用提示、无效用提示；有信念表达、无信念表达）在三种开源大模型（Qwen 3.8、Gemma 4、Muse Glimmer）上进行一次性两人博弈实验，测量合作率、信念率、波动性（对社区信念变化的对数比率标准差）和推理偏差（合作与背离推理长度差异）。

**📊 数据集**

实验不依赖公开数据集，而是以1000次随机复制生成的博弈情境为基础；对每种模型、每种提示条件以及社区信念百分比（0–100%）进行系统采样。

**📈 对比分析**

通过对比四种提示条件下的合作率曲线、信念表达曲线和波动性指标，发现全效用+信念表达条件使所有模型在社区信念>50%时合作率显著提升，波动性降至<0.2；部分或无效用提示导致波动性升高，合作率与社区信念关系不稳定，模型间差异显著；推理偏差在全效用提示下呈现一致的“合作更直觉”模式。

**⚠️ 局限性**

局限性包括：仅评估三种规模相近的开源模型；实验仅涵盖一轮两人博弈，未考虑多轮或多玩家情境；提示均为英语且使用中性标签，未探究语言/框架对结果的影响；对自我一致性与合群一致性的分离仍不充分；波动性虽降低但仍高于理论基线，表明模型对情境的敏感性未完全消除。

---

## 192. Are We Just Applying LLMs? What SE Research Studies, Underexplores, and Could Do Next in the LLM Era

**arXiv ID:** 2611.04206 | [PDF](https://arxiv.org/pdf/2611.04206v1)

**作者:** Zeinadsadat Saghi `[一作]` (University of Southern California), Souti Chattopadhyay `[通讯]` (University of Southern California)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对2018-2026年ICSE、FSE、ASE三大软件工程顶会研究论文的全量审计，结合ICSE 2026 FoSE调查，量化AI（尤其LLM）在软件工程研究中的规模、问题聚焦、研究姿态和社区认知的变化；

**💡 创新点**

创新点在于：①将AI论文与整个研究产出对照，区分AI占比增长是扩张还是重分配；②跨技术浪潮（深度学习→LLM）追踪问题谱系与关注度重构；③从应用、服务、质疑三种姿态拆解AI研究内容；④结合社区问卷揭示AI关注与行动间的鸿沟。

**🔧 技术方法**

主要技术手段包括：大规模文献检索与关键词筛选、文本编码与问题/姿态分类（使用自定义词典和手工/模型辅助标注）、统计分析与可视化；

**📊 数据集**

数据集为：ICSE、FSE、ASE 2018-2026（共4,790篇）研究论文的标题/摘要/关键词；ICSE 2026 FoSE调查问卷（280份，244位回答者）

**📈 对比分析**

方法比较：通过年均发表量、AI占比、技术族分布、问题份额变化、研究姿态比例等指标衡量LLM浪潮对研究组合的影响；结果显示AI占比从13%升至55%，AI产出占总增量93%，但80%论文仅是应用/服务，4%质疑，问题谱系几乎无新增，重分配主要向代码/测试生成倾斜。

**⚠️ 局限性**

局限性包括：仅覆盖三大顶会研究论文，未计入期刊、专场、工作坊；关键词筛选可能漏检；标题/摘要编码无法捕捉深层知识连续性；FoSE问卷自选样本可能偏差；缺乏提交/接受率数据难以证实AI是否真正取代非AI研究。

---

## 193. An Information Budget for Label-Noise Memorization, and What Statistical Complexity Tracks: A Study with Multilayer Perceptrons

**arXiv ID:** 2611.04199 | [PDF](https://arxiv.org/pdf/2611.04199v1)

**作者:** Nancy Baygorrea C. `[一作]` (State University of Northern Rio de Janeiro), Otávio da Fonseca Martins Gomes `[通讯]` (Centre for Mineral Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `2704f255-0c84-4173-b83c-0e9a3dbea232` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究多层感知机在有标签噪声训练集上的记忆行为，并评估其在训练过程中的信息特征。

**💡 创新点**

创新点在于提出并验证了记忆信息量（I(;|Y)）与记忆率的关系，并将统计复杂度与记忆阶段关联。

**🔧 技术方法**

采用信息理论分解（信号、记忆、异质性）、Fano下界、统计复杂度（熵-失衡）以及经验互信息计算。

**📊 数据集**

使用MNIST和Fashion‑MNIST两个小型图像数据集，分别在不同噪声率和噪声模型下训练。

**📈 对比分析**

通过与传统停止规则（最终 epoch、loss plateau、噪声验证、Label Wave）比较，发现在低噪声下复杂度峰值可作为早停依据，但在高噪声下性能明显逊色；复杂度也不如基于标签的指标检测错误样本。

**⚠️ 局限性**

局限性包括仅使用单一MLP架构、合成类条件噪声、仅在小型数据集上实验，且未验证卷积网络或真实噪声环境。

---

## 194. Where Did the Update Go? Certifying Declared Change Envelopes for Model Updates

**arXiv ID:** 2611.04192 | [PDF](https://arxiv.org/pdf/2611.04192v1)

**作者:** Muhammad Usama `[一作]` (Braindeck Inc.), Muhammad Saad Qadir `[通讯]` (Bahria University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究如何在语言模型更新后对更新范围进行声明并证明其在未声明范围内的决策变化（泄漏）小于阈值，提供无标签、随时有效的证书。

**💡 创新点**

提出声明性变更包络（declared change envelopes）概念，定义泄漏率作为估计量，并证明泄漏与更新在未声明输入上的 KL 散度相关；提出决策级别的无标签锚定方法使泄漏可控并可获得证书。

**🔧 技术方法**

使用无标签置信序列、一次性超限检验、KL 距离约束、LoRA、全量微调、DPO、GRPO、RLVR、知识编辑等训练技术；在更新前提交包络哈希。

**📊 数据集**

主要使用 34,136 条多选与分类查询（MMLU、ARC、CommonsenseQA、MedMCQA、AG News、BoolQ、SST-2、CaseHOLD、CounterFact 等），另外包含公开 fine-tune、开放式聊天、NQ、GSM8K 等。

**📈 对比分析**

对比未锚定与锚定、不同更新方式、不同模型规模，证明锚定后在多选任务中 5% 证书可在几百到几千样本内实现；未锚定泄漏常超过 10%；在全微调和 RL 方案中锚定可将泄漏降至 1‑4%。

**⚠️ 局限性**

局限性主要在于证书仅适用于决策类输出（多选、分类），对自由文本和开放式聊天无法提供有意义的泄漏界限；自我不一致阈值限制可证书的最低误差；证书对未见流量和不同格式的推广性有限；锚定需与部署格式匹配；大规模多任务或强 RL 场景尚未覆盖。

---

## 195. What Does 4-Bit Microscaling Cost a Wearable Activity Recognizer?

**arXiv ID:** 2611.04185 | [PDF](https://arxiv.org/pdf/2611.04185v1)

**作者:** Megan Healy `[一作]` (Princeton University), Hossein Valavi `[通讯]` (Princeton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在小型腕式动作识别CNN上评估并比较了FP32、FP16、INT8以及两种4位块缩放浮点格式MXFP4和NVFP4的精度与存储成本。

**💡 创新点**

创新点在于系统性分析MXFP4的下取整比例因子与NVFP4的指数位分配对精度的影响，并通过2×3块大小与比例规则的消融实验揭示尺度取舍是导致MXFP4误差的主要因素。

**🔧 技术方法**

采用TFLite转换、仿真量化（fake quantization）、5折按玩家分组交叉验证、宏F1指标以及对比文件大小与延迟的测量。

**📊 数据集**

使用Hang‑Time HAR数据集（24名篮球运动员的腕部加速度，四个动作类别）。

**📈 对比分析**

通过每折配对差值比较，FP16和INT8几乎无精度损失；MXFP4平均下降约0.05，NVFP4仅下降约0.008，且NVFP4在所有折中表现优于MXFP4。

**⚠️ 局限性**

局限包括仅评估单一小型CNN、仅用篮球数据、仿真量化可能过于乐观、未进行量化感知训练、缺乏4位浮点硬件的真实延迟与能耗测量。

---

## 196. Agentic Path Maps: Exposing an Agent's Traversal Through the Garden of Forking Paths

**arXiv ID:** 2611.04183 | [PDF](https://arxiv.org/pdf/2611.04183v1)

**作者:** Madeleine Grunde-McLaughlin `[一作]` (University of Washington), Daniel S. Weld `[通讯]` (Allen Institute for AI)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 Agentic Path Maps，用于展示自动执行的 AI 任务中未被用户显式指定的备选路径，以提升用户对结果的理解并减少过度依赖。

**💡 创新点**

提出了 Agentic Path Maps 的设计空间，并通过实证研究证明展示多条路径能降低用户对极端结果的依赖并提高认知投入。

**🔧 技术方法**

结合多路径分析 (multiverse analysis)、决策树/规范曲线可视化、Cognitive Load 问卷及 Bayesian 线性混合模型进行实验分析。

**📊 数据集**

采用 BLADE 基准中的 “affairs” 与 “crofoot” 两个公开数据集，生成多种分析路径。

**📈 对比分析**

与仅展示单一路径的对照组相比，展示多条路径在降低权重建议、靠近多路径中值、增加使用时长、提升内在/外在认知负荷以及提升生成路径的 Germane 认知负荷方面取得显著或中等效应，显示该方法在提升用户理解和减少过度依赖方面有效。

**⚠️ 局限性**

样本来自单一学术机构、仅在数据分析领域验证、未评估实际生成路径的计算成本、以及对完整任务有效集合的定义与伦理考量等限制。

---

## 197. Validation-Gated ML Pipeline Synthesis with Expert-Grounded Self-Evolving Skills

**arXiv ID:** 2611.04180 | [PDF](https://arxiv.org/pdf/2611.04180v1)

**作者:** Zixiao Zhao `[一作]` (University of Auckland), Jin Song Dong `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

提出了一种基于专家知识的自适应技能和验证门控的多代理框架 EvoML，能够从极简用户输入自动合成完整可执行的机器学习流水线并持续改进

**💡 创新点**

核心创新包括：①将流水线拆分为四阶段并使用验证门控保障每一步输出合法性，防止错误向下传播；②将专家级工作流（以 nnU‑Net 为核心）编码为可复用技能，保证可靠的域特定处理；③在每轮运行后通过 Curator 将成功的回退和适配持久化为技能更新，实现跨任务的经验累积

**🔧 技术方法**

使用 GPT‑5.5（以及 Claude Opus 4.7 作为对照）生成和适配脚本，Optuna 进行超参搜索，nnU‑Net 进行模型配置与训练，Validator 进行结构/执行/语义三类检查，Curator 负责技能演化

**📊 数据集**

在 8 个医疗影像任务（BTCV、ACDC、FLARE21、FLARE22、Promise12、OrganMNIST3D、FastMRI、IXI）以及 3 个通用任务（CIFAR‑10、VOC2012、AG News）上进行评测

**📈 对比分析**

与三大 agentic AutoML 系统（AutoML‑Agent、MLZero、M3Builder）和两款命令行代码生成工具（Codex、Claude Code）对比，EvoML 在所有数据集上实现了 100% 自动化（EAC=1.0）且取得最高的最终模型质量（FMQ），标准差显著更小；在通用任务上同样取得最优 FMQ

**⚠️ 局限性**

局限性：依赖于强大的大型语言模型；在 nnU‑Net 预训练下的训练时间较长（≈60h vs 10h）；实验聚焦医学影像，虽然在通用任务上表现良好，但更广泛的软硬件工程领域仍需扩展技能库和验证标准

---

## 198. ReconDreamer#: Self-Evolving Closed-Loop Synergy of Reconstruction and Generation for Dynamic Driving Scenes

**arXiv ID:** 2611.04173 | [PDF](https://arxiv.org/pdf/2611.04173v1)

**作者:** Songling Liu `[一作]` (Chinese Academy of Sciences), Zheng Zhu `[通讯]` (Giga AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种自演进的动态驾驶场景重建框架，利用生成模型与重建模型的闭环协同迭代提升新轨迹下的渲染质量和重建精度。

**💡 创新点**

创新点包括：①自演进DriveRestorer，通过多轮“回路”重建生成合成视频并与真实视频配对，持续匹配生成与真实的降质分布；②自适应地面模型（AGM），在激活地面高斯原子时加入软平面约束并根据LiDAR数据动态更新位置、尺度与方向。

**🔧 技术方法**

采用3D高斯喷射（3D Gaussian Splatting）作为渲染基础，结合Diffusion模型进行视频修复（DriveRestorer），并引入几何正则化、深度监督、NDTNet等组件。

**📊 数据集**

在Waymo、nuScenes、PandaSet和EUVS四个公开数据集上进行评估。

**📈 对比分析**

与Street Gaussians、FreeVS、DriveDreamer4D、ReconDreamer、ReconDreamer++等基线相比，在记录轨迹上PSNR/SSIM/LPIPS均有提升；在3m/6m轨迹偏移上F1、NTA‑IoU、NTL‑IoU、FID显著下降，尤其在6m位移上FID下降约15%，NTA‑IoU提升约8.8%。

**⚠️ 局限性**

局限性：①回路采样只能覆盖记录轨迹可见区域，无法验证不可见几何；②自适应地面平面估计在地形突变或LiDAR分割错误处不可靠；③生成修复可能产生视觉一致但几何不准的细节，需进一步结合真实观测验证。

---

## 199. Length Is Cheap, Competition Is Not: Attention Predicts Context Rot in Retrieval

**arXiv ID:** 2611.04169 | [PDF](https://arxiv.org/pdf/2611.04169v1)

**作者:** Yinuo Liu `[一作]` `[通讯]` (University of California San Diego), Yinuo Liu (University of California San Diego)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了语言模型在处理长输入时的失败原因，提出长度并不是主要因素，而是内容之间的竞争导致了准确率的下降。

**💡 创新点**

创新点在于强调了在检索任务中，内容的竞争比长度更为重要，并且提出了通过关注证据的注意力质量来预测模型的损失。

**🔧 技术方法**

使用了多种语言模型（参数从0.5B到3B不等），并通过实验验证了注意力质量与模型损失之间的关系。

**📊 数据集**

使用了126,473次实验数据，涉及不同长度的上下文（从512到32K个token），并比较了中性填充和竞争性填充的效果。

**📈 对比分析**

通过对比不同填充内容的模型表现，发现中性填充几乎不影响准确率，而竞争性填充则显著降低准确率，尤其是在长上下文中，准确率下降到0.11到0.21。

**⚠️ 局限性**

限制在于注意力质量的测量是基于同一次前向传播，缺乏因果干预，且结果仅基于一个合成检索任务，模型参数上限为3B，可能不适用于更大模型。

---

## 200. What Remains in Sight? Autoregressive Video Decoding as Representation-Guided Context Rewriting

**arXiv ID:** 2611.04164 | [PDF](https://arxiv.org/pdf/2611.04164v1)

**作者:** Lixuan He `[一作]` (Stanford University), Leonidas Guibas `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 ReCR，一种训练无关的上下文重写策略，动态从已生成的视频历史中挑选可见片段进行自回归生成。

**💡 创新点**

创新点在于把可见上下文的选择视为推理时决策，利用内部解码器表示进行评分，加入边界覆盖、冗余惩罚和空间范围交换，最终在紧凑因果轴上重索引选定的上下文。

**🔧 技术方法**

使用了表示引导上下文选择、单元表示归一化、全局/最近摘要、余弦相似度评分、预算化贪心选择、空间范围交换、RoPE 位置编码以及 KV‑cache 等无训练技术。

**📊 数据集**

主要在 MovieGen prompt suite（128 条提示）上进行 30 s 与 60 s 的 roll‑out 评估，并在 VBench‑Long 30/60 s benchmark 以及 MovieGen‑128 30 s drift 指标上进行验证。

**📈 对比分析**

与训练基线（Self‑Forcing、LongLive 等）及推理时基线（Deep Forcing、Infinity‑RoPE 等）对比，在所有 backbone 上平均提升 1.41–1.94 分的整体质量分数，动态度数提升明显；在漂移指标上降低文本对齐与美学漂移，提示切换实验中显著保持视觉锚点与动作一致性。

**⚠️ 局限性**

限制在于对解码器内部表示区分度的依赖，采用贪心一步步选择且仅针对完整帧做决策，未考虑对象/场景级记忆，且未学习适应性评分，可能无法充分捕捉全局事件演变。

---

## 201. AdSteer: Native Advertising in Large Language Model Generation

**arXiv ID:** 2611.04147 | [PDF](https://arxiv.org/pdf/2611.04147v1)

**作者:** Shuhang Lin `[一作]` (Rutgers University), Dimitris N. Metaxas `[通讯]` (Rutgers University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种轻量级解码时广告框架 AdSteer，实现在LLM生成过程中自然注入广告内容。

**💡 创新点**

创新点在于将广告注入视作自适应 token 级别的 soft‑constraint steering，使用有限状态机和类别感知 SteerHead 动态调节注入力度与时机。

**🔧 技术方法**

技术方案包括 token‑level soft‑constraint decoding、有限状态机 FSM、类别路由的 SteerHead、AnchorRetrieval 与 AdTrie、两阶段共享‑专家训练（General Head + Category‑Specific Heads）。

**📊 数据集**

数据集：GEM‑Bench 的 MT‑Human、LM‑Market 与 CA‑Prod 三大应用场景，广告检索依赖 BGE‑small‑en‑v1.5。

**📈 对比分析**

与四种基线（Ad‑Chat、GI‑R、GIR‑R、GIR‑P）比较，平均提升整体分数 14.1%（最高 35.4%），同时在 token 消耗上平均减少 22.5%，显著优于所有基线。

**⚠️ 局限性**

局限性包括对广告词典和 Anchor 预检索的依赖，难以处理极短广告或极大词表；多语言和极端广告场景的鲁棒性尚待进一步验证。

---

## 202. LoopKV: Loop-Aware KV Cache Compression for Serving Looped Transformers

**arXiv ID:** 2611.04137 | [PDF](https://arxiv.org/pdf/2611.04137v1)

**作者:** Tianle Zhou `[一作]`, Juncheng Yang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 LoopKV，一种针对循环 Transformer 的 KV 缓存压缩方案，在离线阶段对每个 (循环, 层) 对进行精度与共享决策，在线阶段按注意力动态读取低位。

**💡 创新点**

通过细粒度 (循环, 层) 的量化、跨循环 KV 共享以及注意力感知的值位精度，实现压缩与推理速度双提升，仅需一次离线配置即可。

**🔧 技术方法**

结合 KL 敏感度剖析、多选背包优化、V 价值位分离与动态读取、单核融合 kernel 与 CUDA graph 以及 vLLM 插件集成技术。

**📊 数据集**

在 Ouro‑1.4B/2.6B、Dory‑4B、Huginn‑3.5B 等模型上使用 RULER、LongBench、MATH500、GSM8K、AIME 等公开评测集进行评测。

**📈 对比分析**

与 BF16、FP8、KIVI‑4、KIVI‑4+V、FlashLoop 等基线比较，-M 方案在同一 GPU 上 35% 缩小 KV 并保持 1.7–2.2 倍 KIVI‑4 goodput，-R 方案 1.3–1.6 倍，吞吐与准确率均不逊色于 BF16。

**⚠️ 局限性**

仅适用于每循环都有 KV 缓存的循环 Transformer；需要为每个模型单独离线剖析；实验仅在 RTX PRO 6000、4500、A100‑80 上进行，且 SLO 评测仅针对单一工作负载。

---

## 203. DualWorldBench: Can Agents Plan Deliveries across Symbolic and Grounded Worlds?

**arXiv ID:** 2611.04228 | [PDF](https://arxiv.org/pdf/2611.04228v1)

**作者:** Yang Chen `[一作]` (Nanjing University), Lan-Zhe Guo `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 DualWorldBench benchmark，包含三部分：TextWorld（符号时序规划）、SimWorld（基于视觉的拓扑规划）和 DualWorld-VQA（诊断性问答），以统一的拾取-交付任务接口评估 LLM 与 VLM 在多模态、时空约束下的规划能力。

**💡 创新点**

创新点在于：①将符号规划与基于视觉的拓扑规划拆分成可控子任务，②引入诊断性 VQA 层拆解感知、拓扑与逻辑规划三种失败模式，③通过程序化生成与多场景迁移，提供可复现且可度量的组合复杂度评估。

**🔧 技术方法**

使用的技术包括：LLM（Qwen、Llama、Gemini、Claude、GPT 等）和 VLM（多模态语言模型）、基于 Gazebo 的视觉环境、导航图与路径成本计算、组合优化基线（DFS、在线/离线 DP）以及 JSON 交互式规划接口。

**📊 数据集**

使用的数据集有：①TextWorld 任务集合（5-15 个任务，分 A/M 任务类型与 L1/L2 触发模式），②SimWorld 视觉场景（open、T1、T2、indoor、campus 5 种拓扑，图像+文本输入），③DualWorld-VQA 诊断问答集（由模拟状态自动生成并人工验证）。

**📈 对比分析**

比较方法：对 LLM/VLM 在三部分 benchmark 的累计奖励、完成时间、可行性、完成率和归一化成本与组合优化基线进行对比。实验结果显示：LLM 在符号时序任务上能获得接近基线的奖励但完成时间更长；在视觉拓扑任务中，文本输入往往更可行，视觉+文本多模态输入提升完成率但仍落后于最优；诊断 VQA 进一步表明模型在感知与拓扑理解上相对强劲，但在将其转化为约束一致的全局计划时存在显著缺口。

**⚠️ 局限性**

局限性：①未将符号时序与视觉拓扑约束统一在同一实例中评估；②模型在大规模任务集和复杂拓扑下的全局规划能力仍有限；③诊断 VQA 主要关注感知与逻辑层，未覆盖低层控制与感知误差的细粒度分析；④依赖 LLM/ VLM 的提示工程，对模型超参数与硬件资源有较高依赖。

---

## 204. Shared Context Candidate Scoring with Jointly Adapted Readout Networks

**arXiv ID:** 2611.04218 | [PDF](https://arxiv.org/pdf/2611.04218v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 205. DiMuRe: Diagnosis-Guided Multilevel Recovery for Robotic Task Execution

**arXiv ID:** 2611.04226 | [PDF](https://arxiv.org/pdf/2611.04226v1)

**作者:** Renkai Yi `[一作]` (Beijing Institute of Technology), Che Sun `[通讯]` (Shenzhen MSU-BIT University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了 DiMuRe，一个基于诊断的多层恢复框架，用于在机器人执行任务过程中遇到中途扰动时选择程序重用、预定义恢复技能（包括 VLA 抓取）或基于多模态大型语言模型的终端重新规划，并在 LIBERO-Disturb 基准上进行评估。

**💡 创新点**

创新点在于将阶段性诊断证据与层级恢复决策相结合，实现对不同失效条件的针对性恢复，并提供了一个将扰动注入与任务评估分离的基准，以公平比较程序型与 VLA 策略。

**🔧 技术方法**

技术手段包括规则化诊断、层级恢复策略、视觉‑语言‑动作（VLA）抓取技术，以及多模态大型语言模型（MLLM）生成纠正代码，并结合程序级监控与物理步骤预算管理。

**📊 数据集**

使用的数据集为基于 LIBERO 任务的 LIBERO‑Disturb benchmark，包含 801 个中途扰动实例和 90 个干净测试案例。

**📈 对比分析**

比较方法：在相同物理步预算下，与原始程序、单层恢复（L1 或 L2）、组合恢复（Local）和全恢复（Full）进行对比；结果显示 Local 在原始程序 58.68% 的基础上提升至 77.65%，Full 在额外步数与终端重规划下进一步提升至 81.90%，且恢复成功率显著高于原程序。

**⚠️ 局限性**

局限性：受视觉遮挡导致的对象匹配错误、固定候选顺序可能提前耗尽预算、终端规划的额外步骤与早期交接难以单独评估，以及对完整任务完成检测缺乏足够鲁棒性验证。

---

## 206. IntentGEO: Intent-aware Generative Engine Optimization for Recommendation

**arXiv ID:** 2611.04225 | [PDF](https://arxiv.org/pdf/2611.04225v1)

**作者:** Runlin Lei `[一作]` (Renmin University of China), Zhewei Wei `[通讯]` (Renmin University of China)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究在不同用户需求与决策阶段下探索大型语言模型（LLM）推荐系统的可见性优化，提出Anchored Meta-Optimization（AMO）方法；

**💡 创新点**

创新点在于将推荐指令固定为核心，仅通过引用反馈学习可补充的写作策略，避免传统prompt优化导致的漂移和可见性下降；

**🔧 技术方法**

技术手段包括白帽GEO框架、基于citation reward的Meta优化、固定核心+可学习补充的AMO，以及与RecGEO、MetaOpt、AutoGEO等多种基线的对比；

**📊 数据集**

使用了四个推荐领域（Books、Retail、Videogames、Ragdoll）来自C-SEO Bench的数据，构造十种意图状态，并在Doubao商用助手上进行外部评估；

**📈 对比分析**

在十个意图状态下，AMO平均提升1.12位排名，较RecGEO提升18%，较MetaOpt-Prog提升35%，在Agentic推荐流程和Doubao商用助手中亦保持最高收益，Pass@5提升至43.48%；

**⚠️ 局限性**

局限性包括对模型提示响应的依赖、在高度多变用户表述和新领域时需进一步验证，以及仅关注白帽可见性优化，未考虑潜在的恶意重写风险。

---

## 207. GC-MoE: Overlapping Expert-Parallel Token Dispatch with FP8/FP4 Grouped GEMM via CUDA Green Contexts

**arXiv ID:** 2611.04222 | [PDF](https://arxiv.org/pdf/2611.04222v1)

**作者:** Yong Li `[一作]` (Alibaba Inc), Tao Lan `[通讯]` (Alibaba Inc)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 GC‑MoE，一个基于 CUDA Green Context 的 MoE 推理算子，在 NVIDIA Hopper 与 Blackwell GPU 上实现 FP8/FP4 混合精度推理，采用三节点 CUDA Graph 并行调度，将通信与计算分离。

**💡 创新点**

通过驱动级 SM 分区实现通信与计算的独立调度，利用 token‑ring 同步协议和双缓冲跨 Rank 计数器消除跨 Replay 的 NVLink 阻塞，并为 Blackwell 与 Hopper 分别实现专属的 FP8×FP4 grouped GEMM 指令路径。

**🔧 技术方法**

使用 CUDA Green Context、三节点 CUDA Graph、token‑ring 同步协议、双缓冲计数器、FP8×FP4 block‑scaled MMA/WGMMA、SwiGLU、requantization 以及共享专家融合等技术。

**📊 数据集**

在 DeepSeek‑V4.1‑Flash 与 GLM‑5.2 这两种生产级 MoE 任务上进行评估（未使用传统数据集）。

**📈 对比分析**

与 DeepGEMM Mega MoE 进行基准对比，采用 8 GPU 组装并在 CUDA Graph 重放 100 次测算，GC‑MoE 在 H20 上平均提升 1.188×，在 B300 上平均提升 1.106–1.123×，共享专家融合进一步提升至 1.353×。

**⚠️ 局限性**

仅针对解码（小 token）场景优化，预填充规模受限；分区大小固定且缺乏自适应调节；同一工作空间共享导致多实例串行，未覆盖更大 batch 的实验。

---

## 208. Tokenizer Myths and Where To Find Them

**arXiv ID:** 2611.04215 | [PDF](https://arxiv.org/pdf/2611.04215v1)

**作者:** Rylan Schaeffer `[一作]` (Stanford Computer Science), Sanmi Koyejo `[通讯]` (Stanford Computer Science)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探讨并挑战了令牌化评估中的三种误区，证明 BPB 并非与词表大小无关，且 BPB 与下游性能的相关性弱。

**💡 创新点**

提出“tokenizer trilemma”，指出没有单一指标能同时满足真实性、相同预测相同得分以及未训练模型一致性三项理想属性，揭示评估指标的本质取向。

**🔧 技术方法**

采用理论推导与实验验证，比较 BPB、unigram‑normalized loss 等指标在未训练模型与训练模型中的表现。

**📊 数据集**

在 FineWeb‑Edu‑dedup（英语）及其它多语言语料（代码、法语、印地语、俄语、中文）上进行实验，并使用多任务基准（MMLU、GSM8K、MBPP、HumanEval 等）。

**📈 对比分析**

通过比较 BPB 与多任务性能的相关性，发现两者弱相关，不同指标在同一模型规模下给出不同的最优词表，表明指标不一致。

**⚠️ 局限性**

局限在于指标无法同时满足真实性、相同预测相同得分、未训练模型一致性三重要求，实验也受限于特定语料与模型规模。

---

## 209. Toward User-Centered Agentic Recommendation: When Does Platform Competition Benefit Users?

**arXiv ID:** 2611.04203 | [PDF](https://arxiv.org/pdf/2611.04203v1)

**作者:** Kehan Zheng `[一作]` (Tsinghua University), Hongning Wang `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种用户中心的代理推荐系统，允许用户代理比较来自不同平台的推荐，并将其交互形式化为重复博弈。

**💡 创新点**

创新点在于通过用户代理的信誉评估，解决了平台间竞争带来的信息失真问题，从而提高用户的福利。

**🔧 技术方法**

使用了博弈论模型来分析用户与平台之间的互动，并通过信誉评估机制来改善推荐效果。

**📊 数据集**

使用了合成市场和基于亚马逊的数据集进行实验，评估推荐系统的性能。

**📈 对比分析**

与传统平台固定推荐系统进行比较，结果表明，信誉评估能够在一定条件下提高用户的平均效用，但在某些情况下，用户的福利仍可能低于原平台。

**⚠️ 局限性**

限制在于用户反馈的数量和质量可能不足以完全抵消平台的战略性夸大，且在某些市场条件下，用户可能仍然处于劣势。

---

## 210. SettleSplat: Monocular Gaussian Mapping under In-View Scene Changes

**arXiv ID:** 2611.04195 | [PDF](https://arxiv.org/pdf/2611.04195v1)

**作者:** Anh Thuan Tran `[一作]` (George Mason University), Liang Peng `[通讯]` (Futurewei Technologies)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种名为SettleSplat的单目在线3D高斯映射系统，能够在摄像机视角内同时处理瞬时动态干扰物和持久性物体变化，保持场景的实时一致性并消除旧位置的鬼影；

**💡 创新点**

创新点在于：①将二维不确定性提升为三维高斯动态性字段，用于判别并剔除瞬时动态；②通过渲染深度不确定性无参数化方式，结合几何矛盾评分，区分持久性变化与未优化高斯；③引入“settlement frame”机制消除跟踪不确定性延迟导致的抑制；

**🔧 技术方法**

核心技术包括：3D高斯剖分（Gaussian Splatting），DROID-W不确定性感知SLAM框架，深度不确定性渲染（VarSplat式方差合成），动态性对数回归与交叉熵损失，几何矛盾映射与可靠度权重；

**📊 数据集**

使用Bonn RGB-D动态数据集和Co‑Fusion数据集，挑选包含对象移动、添加或移除的序列共11条；

**📈 对比分析**

与SplaTAM、MonoGS、WildGS‑SLAM、DROID‑W、GaME等基线对比，SettleSplat在全图和变化区域均实现最高的PSNR/SSIM，最低LPIPS，并保持与DROID‑W相近的跟踪FPS；

**⚠️ 局限性**

局限性包括：依赖DINOv2特征不一致性处理动态物体，可能误剔除静态物体；对高度遮挡或光照变化的鲁棒性待进一步验证；

---

## 211. Say the Same, Act Differently: Diagnosing and Steering Text-Action Divergence in Reasoning VLAs

**arXiv ID:** 2611.04166 | [PDF](https://arxiv.org/pdf/2611.04166v1)

**作者:** Zihao Feng `[一作]` (University of Southern California), Ramesh Govindan `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了可解释视觉‑语言‑动作模型（VLA）中“语义不变、动作改变”的攻击方式，并提出一种名为文本中性动作子空间（TNAS）的诊断与攻击方法，使得视觉扰动在保持生成解释完全不变的前提下，能够改变模型的物理动作。

**💡 创新点**

创新点：① 在隐藏状态层面识别并投影出与文本梯度正交、对动作最敏感的方向，从而实现“文本不动、动作变”的对抗；② 将此方向作为攻击向量，配合投影梯度下降（PGD）在视觉输入上执行攻击；③ 在全图与局部补丁两种现实可行的攻击场景中均验证了该方法的有效性。

**🔧 技术方法**

技术：① 通过构造两种文本损失（负对数似然、边际差值）得到文本梯度基底；② 计算动作梯度（action Jacobian），并对其进行投影，得到 TNAS 方向；③ 在输入级别使用 PGD（ε = 0.05 全图，ε = 0.5 补丁）最大化动作偏移并约束与 TNAS 的余弦相似度；④ 采用自由生成与固定前缀两种评价方式。

**📊 数据集**

数据集：Alpamayo（200 条驾驶视频片段 + 154 条带障碍标注的驾驶场景）用于驾驶任务；InstructVLA（200 条机器人操作示例）用于操纵任务。

**📈 对比分析**

比较方法：将 TNAS‑guided 攻击与（1）仅最大化动作损失的攻击、（2）加文本保持惩罚的攻击进行对比。性能表现：在 Alpamayo 中，TNAS 攻击将精确理由保留率从 83.0% 提升至 89.5%，教师强制 KL 减少 68.4%，但无声碰撞率从 4.55% 提升至 52.6%（全图）或 35.1%（补丁）。在 InstructVLA 中，精确回应保留率从 77.5% 提升至 85.0%，教师强制 KL 减少 40.5%，动作偏移保持在 1.73 单位不变。

**⚠️ 局限性**

局限性：① 仅在白盒数字攻击场景下验证，尚未证明对物理世界的鲁棒性；② 依赖于模型共享隐藏状态并可直接干预的架构，对不共享隐藏层的 VLA 可能不适用；③ 攻击仍需较大扰动幅度，对模型的可解释性和安全性提出了更高要求；④ 只考虑了两类 VLA，结果的通用性待进一步验证。

---

## 212. GlucoAgent: Personalized Glucose Forecasting from Multimodal Personal Context

**arXiv ID:** 2611.04126 | [PDF](https://arxiv.org/pdf/2611.04126v1)

**作者:** Xingzhi Sun `[一作]` (Yale University), Hao Chen `[通讯]` (Meta Reality Labs Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种基于大型语言模型（LLM）的代理系统，用于在每个预测点动态生成、执行 Python 代码，对个体的多模态健康数据进行案例级分析，并给出预测结果、理由和证据。

**💡 创新点**

创新点在于将预测过程转化为可解释的程序化推理流程，使代理能够根据个人历史和实时上下文自适应选择合适的特征、模型和验证策略，从而充分利用多模态信息，而非依赖固定的预测模型。

**🔧 技术方法**

技术包括：LLM（如 GPT-4）与 ReAct/程序辅助推理框架、Python 解释器、科学计算库（NumPy、Pandas、Scikit‑learn 等）以及可编写自定义函数的工具链，支持在沙箱中安全执行代码并返回可验证的证据。

**📊 数据集**

主要使用公开的 CGMacros 多模态糖尿病监测数据集（含 CGM、心率、活动、膳食日志和健康档案）进行实验，并在 GlucoFM‑Bench 的九个单模态 CGM 数据集上做基准对比。

**📈 对比分析**

与 Chronos‑2 等时序基础模型在 CGMacros 上对比，使用多模态输入时平均 RMSE 从 23.5 mg/dL 降至 22.5 mg/dL，尤其在 60 分钟后餐窗口提升显著（降低 4.6 mg/dL）；在单模态基准上，表现与现有时间序列模型持平或略优。

**⚠️ 局限性**

局限性包括：仅在 CGMacros 上验证多模态效果，缺乏更大规模或更异质的数据；代理推理的计算成本和推理时间未做系统评估；模型对新出现的模态（如图像）支持有限。

---

## 213. Reliability-Role Collaborative Learning for Semi-Supervised Cardiac Image Segmentation

**arXiv ID:** 2611.04219 | [PDF](https://arxiv.org/pdf/2611.04219v1)

**作者:** Zhongrui Xie `[一作]` (Chongqing Normal University), Tao Jia `[通讯]` (Chongqing Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种基于可靠性角色的协同学习框架R2CL，用于半监督心脏影像分割；

**💡 创新点**

通过对双网络置信度的细粒度划分，将未标注区域分配为互不确定、互相置信、以及置信差异三种角色，并分别设计熵正则、方向性知识迁移、原型对比等策略；

**🔧 技术方法**

主要技术包括双网络对抗式训练、可靠性角色混合交叉学习（RR-MCL）、前景相邻可靠性过渡原型对比学习（RT-PCL）以及高置信上下文精炼（HCR）；

**📊 数据集**

使用ACDC心脏MRI和MVAA-CT二尖瓣CT两大医学影像数据集；

**📈 对比分析**

与现有半监督方法（如CPS、MC-Net+、BCP、DiffRect等）进行对比，R2CL在ACDC上实现DSC从86.54%提升至89.93%/90.75%，95HD从5.96mm降至1.43mm/1.37mm；在MVAA-CT上，在2标注/4标注设置下分别取得DSC提升至78.48%/82.66%，95HD下降至2.08mm/1.29mm，均超过对比方法；

**⚠️ 局限性**

对可靠性阈值选择、原型构造的可扩展性有限；在更大规模、多模态或前景稀疏度更高的数据集上效果尚待验证；

---

## 214. Online iLQR-MPC with Whole-Body Control for Robust Humanoid Locomotion

**arXiv ID:** 2611.04204 | [PDF](https://arxiv.org/pdf/2611.04204v1)

**作者:** Xiang Li `[一作]` (Zhejiang University of Technology), Xinyi Yu `[通讯]` (Zhejiang University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种在线iLQR-MPC与全身控制相结合的鲁棒类人步态控制框架，能够在同一回收期问题中同时优化接触力和步姿，完成稳态步态和冲击恢复。

**💡 创新点**

创新点包括：①将步态规划与接触力规划统一到单个iLQR-MPC求解器；②在iLQR内部采用增广拉格朗日+松弛障碍两种罚项实现约束处理；③通过全身控制把高层计划映射到关节扭矩，实现完整约束可行性。

**🔧 技术方法**

技术手段包括：单刚体( SRB )质心动力学模型、迭代线性二次调节器(iLQR)MPC、增广拉格朗日与松弛障碍约束、任务优先级层次化全身控制、QP求解器( qpOASES )、CasADi符号求解、MuJoCo仿真。

**📊 数据集**

使用的数据集为 MuJoCo 仿真中的 Unitree G1 机器人模型，并在仿真中施加多方向冲击力作为扰动测试。

**📈 对比分析**

通过对比稳态步态的 RMSE、冲击恢复极限（前向245 N、左向175 N、右向80 N）以及计算时延（平均8.9 ms、最大13.6 ms）评估性能，显示该方法在低速步态下能够实现稳态行走和较大冲击恢复，计算量满足实时需求但在冲击瞬态时偶有超时。

**⚠️ 局限性**

局限性包括：步行速度受限于模型简化，推理周期偶有超出控制周期，扰动力矩假设已知而非实时测量，以及尚未在真实硬件上验证。

---

## 215. When the Consumer Is an Agent: Execution-Grounded Service-Level Agreements

**arXiv ID:** 2611.04182 | [PDF](https://arxiv.org/pdf/2611.04182v1)

**作者:** Michal Orzechowski `[一作]`, Jacek Kitowski `[通讯]` (AGH University of Krakow)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种基于执行根据信息服务级别协议的完成合同，独立控制平面通过预设的状态谓词和截止时间来判定LLM代理任务是否真正完成；

**💡 创新点**

将任务完成判定从代理自身转移到独立观察者，实现了代理消费者与服务提供者的职责分离；通过执行根据信息SLA定义了可量化的完成合同，首次在LLM代理场景中引入独立验证与恢复决策；

**🔧 技术方法**

使用Onedata联邦存储、onedeat-mcp工具服务器、LLM代理（七种模型）、独立验证器（周期性状态采样）、完成门控、WS-Agreement式SLA模型以及Python标准库；

**📊 数据集**

在Onedata联邦存储上执行两类任务（文件复制/删除），共计56次无故障与56次注入延迟的实验，使用真实存储环境而非传统标注数据集；

**📈 对比分析**

通过与四种常见完成策略（自报、呼叫+健康监测、转录判断、报告时状态读）以及基于截止时间的参考判定比较，评估误判率、恢复成功率、费用与运行时活动指数；结果显示独立验证能显著降低错误完成率并提供恢复路径，但会略微增加成本与延迟；

**⚠️ 局限性**

局限性包括：仅在单一联邦存储服务和两类任务上验证；谓词仅为简单字节计数/路径检查；未评估多阶段工作流或更复杂的SLA；成本分析仅覆盖模型调用费用与验证器开销，未考虑更广泛的系统资源；注入故障仅为单一35 s网络延迟，缺乏多样化故障场景；

---

## 216. Improving PI-RADS Prediction and Prostate Cancer Detection from Multiparametric MRI with Anatomical Guidance

**arXiv ID:** 2611.04155 | [PDF](https://arxiv.org/pdf/2611.04155v1)

**作者:** Seungbin Park `[一作]`, Heejong Kim `[通讯]` (Weill Cornell Medicine)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究在多参数MRI上评估使用前列腺区分解剖掩码作为深度学习模型输入，以预测PI‑RADS评分和临床显著前列腺癌（csPCa）

**💡 创新点**

创新点在于将前列腺的五区/两区分割掩码作为额外通道加入模型，显著提升PI‑RADS预测与csPCa检测，并与专家评分互补

**🔧 技术方法**

采用3D Vision Transformer、ResNet‑18与ResNet‑50架构，使用CORAL头进行序数分类，AdamW优化，并在内部数据上进行预训练后在外部数据微调

**📊 数据集**

使用13,473例内部mpMRI（含PI‑RADS与csPCa标签），并在外部UCSF（971例）和PMUB（636例）数据集上验证

**📈 对比分析**

与无解剖引导的基线模型和ProFound基准进行对比，内部最佳QWK为0.511（ResNet‑18+2区），csPCa AUC为0.809（ResNet‑50+5区）；外部微调后仍优于从零训练，模型预测PI‑RADS AUC为0.729，结合专家评分提升至0.756

**⚠️ 局限性**

局限在于分割误差可能影响下游任务，外部验证仅包含PI‑RADS 3–5病例，未覆盖完整评分谱，且未检验更多数据集及下游临床任务

---

## 217. Palimpsestic Accountability in the US Federal AI Inventories

**arXiv ID:** 2611.04153 | [PDF](https://arxiv.org/pdf/2611.04153v1)

**作者:** Dipto Das `[一作]` (Cornell University), Shion Guha `[通讯]` (University of Toronto)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对美国联邦AI使用案例清单（2023–2025）进行纵向混合方法研究，探讨同一AI系统在不同发行版中的记录变化与治理披露的演进；

**💡 创新点**

提出“多层剥蚀（palimpsestic）问责”概念，揭示库存更新过程中的问题域重绘、分类门控与答案覆写机制，并给出改进公共AI登记的具体建议；

**🔧 技术方法**

采用双重演绎-归纳质性分析，结合Algorithmic Decision-Making Adapted for the Public Sector（ADMAPS）框架，对登记表字段与治理信息进行编码；

**📊 数据集**

使用OMB公开的三份AI使用案例清单（710、2,133、3,611条记录），并手工匹配同一系统在不同年份的条目，构建跨期链接数据集；

**📈 对比分析**

通过描述性统计与跨年份匹配比对，评估问题覆盖度、分类变动、答案保留与丢失等维度，发现问卷重绘导致回答缺失、分类门控削弱治理披露、以及答案覆写导致信息混淆；

**⚠️ 局限性**

局限性包括：仅基于公开记录，未观察实际运作；匹配方法依赖阈值与人工审核，可能漏掉重命名或拆分的案例；聚焦高影响/准高影响用例，低影响用例的治理细节未深入探究；

---

## 218. PACT: Certifying What Is Written in Parallel Decoding of Diffusion Language Models

**arXiv ID:** 2611.04152 | [PDF](https://arxiv.org/pdf/2611.04152v1)

**作者:** Yanchao Li `[一作]` (Nanjing University), Yuqiang Li `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的提交规则PACT，利用已抽取的 token 先做后续提交的可信度评估，并在每一步通过给定阈值挑选最可信的 token，同时对所选位置进行间距处理，从而在保持答案质量的同时显著减少前向传递次数。

**💡 创新点**

①使用后抽样的 token 直接评估可信度，解决传统规则仅依据预测分布导致的误判问题；②对同一步内的 token 进行空间间隔，降低 token 之间的互依性；③提供理论证明，保证在任何温度下都能提交至少与阈值相符的 token 数量。

**🔧 技术方法**

基于遮蔽扩散语言模型的并行解码；采样技术与温度控制；后抽样可信度评估（post‑draw accumulated certificate）；位置间距过滤；阈值 logτ 筛选；对比实验框架与理论分析。

**📊 数据集**

HumanEval、MBPP（代码生成）；GSM8K、MATH500（数学问题）；IFEval（指令遵循）；在 LLaDA‑8B、LLaDA‑MoE、LLaDA2.1‑mini、Dream‑7B 四个遮蔽扩散模型上进行评测。

**📈 对比分析**

与五种公开提交规则（Fast‑dLLM、Fast‑dLLM++、EB‑Sampler、KLASS、TACG）以及两种基准解码（单 token/固定 k）在相同温度（0.6）下对比；指标为 pass@1、pass@8 与前向传递次数；结果显示 PACT 在相同准确率下前向传递次数显著减少，速度提升明显；在相同速度下准确率优于多数基准。

**⚠️ 局限性**

仅评估已抽样 token 的 log‑probability 总和，未直接衡量 joint‑probability；间距机制只能间接降低依赖，仍受模型内在依赖结构影响；性能受模型自身概率估计质量限制，无法纠正模型自信但错误的预测；评测集中于英文代码/数学/指令任务，未覆盖多语言或开放式生成任务。

---

## 219. On the Role of Geometry in Solving Variational Inequalities

**arXiv ID:** 2611.04151 | [PDF](https://arxiv.org/pdf/2611.04151v1)

**作者:** Ioannis Anagnostides `[一作]` (Carnegie Mellon University), Jingming Yan `[通讯]` (University of California, Irvine)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究 ℓ_p 球约束下变分不等式（VI）的计算复杂度

**💡 创新点**

首次揭示非欧几里得球（p≠2）下 affine VI 为 PPAD‑complete，欧几里得球下 affine VI 可多项式求解，但二次 VI 仍为 PPAD‑complete

**🔧 技术方法**

利用几何约束与算子结构的新工具，构造阈值函数逼近与权重化加权平均

**📊 数据集**

无数据集，纯理论复杂度分析

**📈 对比分析**

与传统的两人矩阵游戏、极限多项式游戏的复杂度结果对照，证明在逆多项式精度下仍保持难度

**⚠️ 局限性**

仅适用于 ℓ_p 球，且正则化条件有限，对更一般凸集或更高阶算子尚未扩展

---

## 220. Interpretable Early Termination of Web Navigation Agents via Closed Sequential Pattern Mining

**arXiv ID:** 2611.04142 | [PDF](https://arxiv.org/pdf/2611.04142v1)

**作者:** Sergio Talavera `[一作]` (San Jose State University), Magdalini Eirinaki `[通讯]` (San Jose State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于闭式序列模式挖掘的可解释早期终止方法，用于检测和诊断基于开放权重LLM的Web导航代理的失败。

**💡 创新点**

创新点在于：①将BIDE+与多级符号化结合，生成可手工检查的小型失败模式库；②构建四类失败分类体系，并给出对应修复建议；③设计多变体评估协议，剔除长度诱导的误差；④在早期终止上实现高达26.3%的token节省同时保持92%精度。

**🔧 技术方法**

使用的技术包括：BIDE+闭式序列模式挖掘、SPMF实现、四维符号化（动作类型、选择器、观察结果、推理意图）、ReAct框架、阈值覆盖评分及阈值触发终止。

**📊 数据集**

数据集主要是MiniWoB++（1,544条轨迹，包含成功、失败、超时、错误）和WebArena购物轨迹（577条），覆盖Llama-3.2-3B、Qwen-2.5-7B、Mistral-7B-Instruct-v0.3等三种开放权重模型。

**📈 对比分析**

与七种基线（频率向量、n-gram、DeepLog、Bi-LSTM、TaSPM、过程符合性检查和步长控制）比较，BIDE在Exclude-errors配置下K=3时取得最高F1=0.717；在Variant C配置下，覆盖率88%，早期终止在K=5时达到92%精度并节省26.3% token。

**⚠️ 局限性**

局限性包括：①不同模型间性能差异大，库的通用性受限；②跨benchmark迁移效果差，取决于模型的自我监控程度；③在较长K下神经网络优于模式挖掘；④缺乏用户研究验证解释性与修复建议的有效性；⑤评估仅基于单一随机种子，未报告多种子波动或统计显著性。

---

## 221. IDs That Mean Something: A Hands-On Tutorial on Semantic IDs for Recommender Systems

**arXiv ID:** 2611.04140 | [PDF](https://arxiv.org/pdf/2611.04140v1)

**作者:** Roger Zhe Li `[一作]` (Huawei Ireland Research Center), Barry Smyth `[通讯]` (University College Dublin)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本教程系统阐述了从传统整数型原始ID到语义ID的演进，演示如何利用语义ID在生成式与判别式推荐模型中提升冷启动、跨域泛化与可扩展性，并通过实操训练RQ‑KMeans生成语义ID，构建并评估基于Amazon Reviews的数据集上的推荐任务。

**💡 创新点**

创新点在于把语义信息嵌入到物品身份本身而非仅作为辅助特征，提出了多模态与协同信号融合的语义ID构造方法，并在跨域推荐中展示了无需显式域对齐即可实现共享结构与迁移的可行性。

**🔧 技术方法**

使用技术包括：语义ID编码（RQ‑VAE、RQ‑KMeans等量化方法）、生成式推荐模型（如Cornac的自回归模型）、判别式排序模型（如YouTube/Google的分层网络）、早期与晚期多模态融合、协同嵌入量化以及对比学习等。

**📊 数据集**

采用公开的Amazon Reviews数据集进行实验，并在Notebook中预计算文本/图像特征，方便在免费Colab环境下复现。

**📈 对比分析**

通过与基于原始ID的基线对比，实验表明语义ID能够在生成式推荐任务中显著提升推荐的多样性与冷启动表现，判别式排序模型中也观察到点击率（CTR）提升约3‑5%，总体表现优于单纯使用内容或协同特征的模型。

**⚠️ 局限性**

局限性包括：语义ID在目录动态变化时可能不稳定；当前评估仍以hit‑rate/CTR为主，缺乏更细粒度的语义质量与用户满意度指标；多模态融合与协同信号的最佳组合仍待进一步研究。

---

## 222. LENS: Reliability-Aware Stock Prediction via LLM-Extracted Priors and News-Driven Semantic Graphs

**arXiv ID:** 2611.04217 | [PDF](https://arxiv.org/pdf/2611.04217v1)

**作者:** Hao Zhou `[一作]` (University of Edinburgh), Luis Felipe Costa Sperb `[通讯]` (Kingston University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

LENS 构建了静态 LLM 提取的多关系公司图与动态基于新闻语义的图，并通过可靠性权重与门控融合实现股票收益预测与排序。

**💡 创新点**

创新点在于将关系可靠性作为第一阶信号，采用多轮 LLM 提取与可靠性加权聚合，使用指数衰减时间聚合的新闻语义图，并设计上下文感知门控融合机制，以提升在新闻覆盖不均的环境下的鲁棒性。

**🔧 技术方法**

技术手段包括：o3-mini LLM 关系抽取、多轮抽取后可靠性权重聚合；Voyage-3-large 文本嵌入+Transformer 时序编码；Graph Transformer 双分支（静态与动态）加上门控融合；混合损失 (MSE + RankNet)。

**📊 数据集**

使用的数据集为 S&P 500 与 NASDAQ 100 成分股的历史价格、技术指标，以及 Thomson Reuters 新闻档案的标题。

**📈 对比分析**

通过滚动窗口交叉验证与 11 种序列/图模型基线（如 FinGPT、Transformer、COGRASP 等）比较，LENS 在 AR@5、SR@5 以及 MRR 上均显著优于基线，提升约 16%–22% 的收益与 16%–22% 的夏普比率。

**⚠️ 局限性**

局限性包括：仅使用单一 LLM 作为关系抽取源、仅评估美国大盘、仅做历史回测未验证实盘表现、未检验跨模型或跨市场的泛化能力。

---

## 223. Impact of Input Resolution in Mammographic Mass Detection: A Controlled Multi-Architecture Study

**arXiv ID:** 2611.04200 | [PDF](https://arxiv.org/pdf/2611.04200v1)

**作者:** Kaan Buyukdemirci `[一作]`, Emine Ulku Saritas `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在 VinDr-Mammo 数据集上，通过 18 种检测模型与 6 种输入分辨率的 108 组合，开展了控制实验，系统评估了单纯提升输入分辨率对乳腺癌肿块检测性能的影响；

**💡 创新点**

创新点在于：①消除了模型、预处理、评估协议等多重变量，仅对输入分辨率进行系统变动；②首次以标准全图检测器实现了最高 mAP_50 0.782，证明高分辨率能显著提升小肿块检测；

**🔧 技术方法**

使用的技术包括多种主流检测框架（Cascade R‑CNN、TOOD、FCOS、DINO、YOLO11‑M、RTMDet‑M）与不同骨干网络（ResNet‑50、Swin‑T、HRNet‑W32、ConvNeXt‑S），统一训练与评估流程；

**📊 数据集**

数据集为公开的 VinDr‑Mammo，包含 20,000 枚乳腺X光片、1,226 个肿块框，本文采用官方 1,000‑exam 测试集（219 张含肿块图像）进行评估；

**📈 对比分析**

对每个模型‑分辨率组合在 219 张含肿块的测试图像上计算 mAP_50；去除 mAP_50<0.60 的组合后，发现中位 mAP_50 随分辨率从 512×416 提升到 1600×1280 约提升 0.11，最佳组合（DINO/ConvNeXt‑S）在 1600×1280 下达成 0.782 的最高分；

**⚠️ 局限性**

局限性包括：①小肿块仍然是性能瓶颈，提升分辨率对其提升有限；②训练与推理时间随分辨率上升，尽管仍在可接受范围内；③仅使用标准全图检测器，未结合外部医学预训练或混合推理策略，可能仍有进一步提升空间。

---

## 224. DoGNAVY-Exploit: Harnessing Parallel Open-Weight Agents for Stage-Aware Vulnerability Exploitation

**arXiv ID:** 2611.04196 | [PDF](https://arxiv.org/pdf/2611.04196v1)

**作者:** Yuxuan Wang `[一作]` (Shanghai Artificial Intelligence Laboratory), Dongrui Liu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 DoGNAVY-Exploit，一种多代理框架，用于提升开源 LLM 在漏洞利用中的能力。

**💡 创新点**

创新点在于阶段感知的分解与进阶专家知识发布、层级规划与并行探索、以及共享结构化记忆的组合，实现长时序漏洞利用的高效与可控。

**🔧 技术方法**

技术包括多代理协同（规划者、执行者、控制器）、阶段化拆分、专家知识注入、并行执行、共享记忆压缩与递归检索，以及 OpenCode 工具交互。

**📊 数据集**

使用了 Cybench、CVE-Bench 与 ExploitGym 三大基准，涵盖通用 CTF、真实 Web 漏洞和长链二进制漏洞。

**📈 对比分析**

与公开基准（Claude Mythos、GPT-5.5/6 等）及自家控制基线对比，DoGNAVY-Exploit 在 Cybench 97.5% Pass@1、CVE-Bench 97.5%/100% Pass@1、ExploitGym 在 GLM-5.3 上完成 157/485 任务，成本仅为 GPT 系列的 1/5 左右，展现显著的性能与成本优势。

**⚠️ 局限性**

局限性包括对极其复杂或未覆盖任务的泛化不足、仍需高质量专家知识与阶段划分的手工设计、以及在更大规模任务集上的进一步验证与资源调优需求。

---

## 225. Statistical Machine Learning for Medicinal Plant Leaves Classification

**arXiv ID:** 2611.04193 | [PDF](https://arxiv.org/pdf/2611.04193v1)

**作者:** Jayani P. Gamage `[一作]` (Curtin University), Thiyanga S. Talagala `[通讯]` (University of Sri Jayewardenepura)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了MEDIPI分层统计机器学习方法，用叶片图像自动识别斯里兰卡药用植物，并公开MedLEA R包数据集。

**💡 创新点**

创新点包括：①提出52个可解释的形状、纹理、颜色和scagnostics特征；②采用形状‑边缘‑种类三级分层随机森林分类；③首次公开斯里兰卡药用植物叶片图像数据库。

**🔧 技术方法**

使用统计机器学习算法（随机森林、XGBoost、GB等）、特征提取（形状、纹理、颜色、scagnostics）以及线性判别分析（LDA）进行可视化。

**📊 数据集**

使用了MedLEA（1099张，31种植物）以及公开的Flavia、Swedish、Kaggle叶片图像数据集。

**📈 对比分析**

通过与非分层方法及八种算法对比，分层随机森林在同一数据集上达到100%准确率，形状层准确率从90%提升至97%；但跨数据集测试准确率仅为23%–50%。

**⚠️ 局限性**

局限性包括跨数据集泛化性能差、不同采集方式导致图像差异、部分形状（如心形、菱形）区分困难，以及数据库样本种类相对有限。

---

## 226. Hardware-Efficient Reinforcement-Learned Scheduling for Quantum LDPC Decoders

**arXiv ID:** 2611.04179 | [PDF](https://arxiv.org/pdf/2611.04179v1)

**作者:** Mohsen Moradi `[一作]` (Arizona State University), David Mitchell `[通讯]` (New Mexico State University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

实现了传统BP、强化学习顺序BP（RL‑S）和基于桶的强化学习BP（BRL‑S）在[[288,12,18]]双变量自行车QLDPC码上的硬件架构

**💡 创新点**

通过将RL‑S的最大堆优先级调度替换为量化优先级的FIFO桶，显著降低调度硬件复杂度，同时保持顺序消息更新框架

**🔧 技术方法**

采用硬件可综合的RTL实现，FPGA Vivado与45 nm ASIC标准单元综合，利用量化优先级、FIFO桶、强化学习预训练Q表等技术

**📊 数据集**

使用[[288,12,18]]双变量自行车QLDPC码的量子错误模型（退化噪声）进行仿真，比较不同优先级分辨率（K=2,4,8）

**📈 对比分析**

通过比对FPGA LUT/FF占用和ASIC面积/功耗，RL‑S相较传统BP提升了纠错性能；BRL‑S在保持与RL‑S相近的BLER的同时，将LUT占用降低≈70%、面积降低≈65%、功耗降低≈65%；

**⚠️ 局限性**

仅在单一QLDPC码上评估，优先级量化可能在更高噪声或不同码结构下失去性能；硬件实现尚未在实际量子设备上验证，且未考虑多通道/相关错误场景

---

## 227. Semidirect sums of tensors and the matrix multiplication exponent

**arXiv ID:** 2611.04175 | [PDF](https://arxiv.org/pdf/2611.04175v1)

**作者:** Baitian Li `[一作]` `[通讯]` (Columbia University), Baitian Li (Columbia University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过提出并证明了半直和张量的“解卷”引理，利用该技术在多项式乘法张量上实现了矩阵乘法指数上界ω≤9/4的突破；

**💡 创新点**

创新点在于引入半直和张量概念并证明其在张量限制上的强大性质，从而破解了传统激光方法中的“精神病”问题，并实现了先前未达成的更紧的上界；

**🔧 技术方法**

主要技术包括张量的半直和、光谱点与价值函数的半超加性分析、单项式退化（monomial degeneration）与分块哈希技术，以及多项式乘法张量的结构化分解；

**📊 数据集**

由于研究完全理论性质，本文并未使用任何实验数据集；

**📈 对比分析**

与之前基于Coppersmith–Winograd张量的激光方法相比，本文所用方法在不受Ambainis–Filmus–Le Gall、Solar或Galactic方法限制的前提下，实现了从ω≈2.38到ω≤2.25的显著改进；

**⚠️ 局限性**

限制在于仍受不可逆性障碍和张量子秩下降的限制，当前方法无法进一步逼近ω=2，且对于非平方矩阵乘法指数的提升仍有限。

---

## 228. On the Feature Evolution of Temporal Difference Learning

**arXiv ID:** 2611.04158 | [PDF](https://arxiv.org/pdf/2611.04158v1)

**作者:** Shangzhe Li `[一作]` (University of North Carolina at Chapel Hill), Weitong Zhang `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在有限宽度两层 ReLU 网络上，Temporal‑Difference (TD) 更新如何在不完整神经更新下获得预测特征并提升策略表现。

**💡 创新点**

提出了针对不完整 TD 目标的有限样本理论框架，证明初始更新即可获取二次贝尔曼信号，并量化了联合特征与读出改进对预测风险与回报的正向作用。

**🔧 技术方法**

采用了理论分析、有限宽度网络的 SGD 更新、批量回放、耦合递归以及指数衰减误差证明等技术。

**📊 数据集**

使用构造的光滑随机 MDP 家族（含二次特征），并未在公开真实数据集上进行实验。

**📈 对比分析**

与仅更新头部的 SGD 以及其他理论回报界限对比，证明了预测风险显著下降，并给出有限预算回报上界为 𝒪(d³H^{1/3}T^{2/3})。

**⚠️ 局限性**

主要局限在于仅适用于构造的 MDP 及严格的宽度、样本、步数等假设，缺乏对实际环境的实证验证。

---

## 229. RGDM: Robust Graph Diffusion Model

**arXiv ID:** 2611.04146 | [PDF](https://arxiv.org/pdf/2611.04146v1)

**作者:** Yancheng Wang `[一作]` (Arizona State University), Yingzhen Yang `[通讯]` (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了 Robust Graph Diffusion Model（RGDM），通过鲁棒图自动编码器（RGAE）和潜在扩散模型（LDM）在受攻击图上生成带标签的合成节点及其边，并将这些合成结构与原图合并后用于训练 GNN，以提升节点分类的鲁棒性。

**💡 创新点**

核心创新在于：① 通过边选择机制与低秩正则化在潜在特征空间中抑制攻击噪声；② 引入 Sparse Hierarchical Edge Decoder（SHED）实现稀疏、高效的邻接解码；③ 在无配对干净图的条件下训练鲁棒生成器，兼顾生成质量与对抗鲁棒性。

**🔧 技术方法**

技术手段包括：图注意力网络（GAT）与 Gumbel Softmax 的二进制掩码，低秩正则化（截断核范数），潜在扩散模型配合 Classifier-Free Guidance，SHED 结构，以及在训练中加入稀疏化与低秩正则化的联合损失。

**📊 数据集**

实验数据集涵盖了常用的 Cora、Citeseer、Pubmed、Polblogs，并在更大规模的 ogbn-arxiv、Reddit 上做消融；攻击方式使用 Metattack、Nettack 与 Topology Attack。

**📈 对比分析**

通过与同一基线模型（GCN、SG-GSR、Pro-GNN 等）在相同攻击条件下的对照实验，RGDM 在 25% Metattack 下平均提升 2–14 个百分点，且相较于 GAN、混合增强（iGraphMix、ImGAGN、Graphsmote、SNS、DoG、LR-FGDM）方法获得更显著的性能提升；同时 SHED 在训练与节点生成上实现数倍速度加速。

**⚠️ 局限性**

局限性包括：在强基线模型下提升幅度有限；低秩正则化需通过交叉验证选择起始秩；未提供端到端鲁棒性理论保证；对自适应攻击的防护尚不充分；生成的合成节点仅与原图连接，未实现节点间互连；在无攻击（清洁）图上提升也仅为几百分点。

---

## 230. HyPAIC: Personal AI Computer with High Bandwidth Flash for High Performance Trillion-Parameter LLM Inference

**arXiv ID:** 2611.04144 | [PDF](https://arxiv.org/pdf/2611.04144v1)

**作者:** Minho Ha `[一作]`, Hoshik Kim `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

无信息

**💡 创新点**

无信息

**🔧 技术方法**

无信息

**📊 数据集**

无信息

**📈 对比分析**

无信息

**⚠️ 局限性**

无信息

---

## 231. PhotonZip: Lifecycle-Aware Compression for HDF5-Based Synchrotron Workflows

**arXiv ID:** 2611.04138 | [PDF](https://arxiv.org/pdf/2611.04138v1)

**作者:** Haoxu Li `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Dingwen Tao `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出PhotonZip框架，将生命周期感知的压缩与HDF5滤镜相结合，实现对同步辐射工作流的自动配置与高效压缩。

**💡 创新点**

创新点包括：透明的HDF5集成、目标导向的压缩配置、模块化可扩展的后端以及对完整I/O路径的端到端特征化。

**🔧 技术方法**

采用HDF5滤镜接口，集成MANS、LC等无损压缩器以及PFPL、空间下采样等有损预览模式，并在MPI文件‑per‑rank与OpenMP多线程环境下测评。

**📊 数据集**

使用四个来自实验室光束线的真实数据集进行实验，涵盖不同分辨率与帧率的多维图像。

**📈 对比分析**

与固定压缩方案相比，PhotonZip‑Ratio在四组数据中压缩比提升至1.629×，PhotonZip‑Speed保持1.529×的压缩比同时提高写/读吞吐率21.2%/15%，有损预览可实现约16×的数据压缩，显著降低预览时延。

**⚠️ 局限性**

局限在于仅在单节点、文件‑per‑rank的HDF5配置下评估，未覆盖共享文件并行HDF5或实时光束线部署场景。

---

## 232. ScrambleMark: Zero-Query Watermark-Agnostic Black-Box Evasion of Modern Speech Watermarking

**arXiv ID:** 2611.04122 | [PDF](https://arxiv.org/pdf/2611.04122v1)

**作者:** Binhao Ma `[一作]` (University of Missouri Kansas City), Rui Duan `[通讯]` (University of Missouri Kansas City)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种零查询、无水印知识、黑盒攻击框架 ScrambleMark，利用谐波结构扰动来实现语音水印的结构性规避，而非直接移除水印。

**💡 创新点**

创新点在于：①提出“结构性规避”概念，证明扰乱水印结构即可击败检测；②构建两阶段攻击：先用确定性频谱分块移位生成可迁移扰动，再通过条件扩散模型学习随机化、恢复抗逆扰动；③在保持语音自然度的同时，对抗自适应恢复攻击。

**🔧 技术方法**

主要技术包括：频谱分块频移、协方差矩阵自适应进化策略 (CMA‑ES)、条件扩散生成模型 DiffWave、Mel 失真度量、EQTR 指标、随机化监督与动态时间段扰动。

**📊 数据集**

使用公开语料库：VoxCeleb、Valentini、LJ Speech、LibriSpeech 等，训练后在多语言（英语、中文、韩语、法语、德语）以及 7 种学术水印模型和 3 个商业平台（Google SynthID、Resemble AI、TrustedAudio）进行评估。

**📈 对比分析**

与白盒、黑盒、覆盖攻击、重写攻击、生成式移除（DiffErase、Encodec）等方法对比，ScrambleMark 在 ASR 约 0.97–0.98、STOI 约 0.91、MOS 约 4.17（人类评估）均显著优于所有基线，且保持高自然度且能抵抗自适应恢复。

**⚠️ 局限性**

局限性包括：仍对极其鲁棒的水印模型（如 RobustDNN）效果略逊；依赖于 surrogate 水印模型的迁移性，极端不同的水印可能需要重新调参；目前仅针对语音域，未验证跨媒体（如音频+视频）的适用性；在极端噪声或压缩环境下的鲁棒性尚未系统评估。

---

## 233. Clocks all the way down: does sequence-processing AI need endogenous temporal structure?

**arXiv ID:** 2611.04181 | [PDF](https://arxiv.org/pdf/2611.04181v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 234. Mamba3R: Fixed-Capacity Selective State Memory for Long-Sequence 3D Reconstruction

**arXiv ID:** 2611.04221 | [PDF](https://arxiv.org/pdf/2611.04221v1)

**作者:** Zhenxiang Xiong `[一作]` (Shanghai Jiao Tong University), Weidong Che `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在固定GPU预算下，通过在冻结的循环3D重建模型上添加可训练的状态‑内存适配器，提升长序列重建精度。

**💡 创新点**

创新点是联合设计的选择性状态空间混合器与有限容量辅助记忆，及对记忆反馈的分解调节，从而在不扩展原模型参数的情况下提升几何质量。

**🔧 技术方法**

使用的技术包括Mamba选择性状态空间模型、可训练的读写路径、历史衰减反馈、预训练的CUT3R/TTT3R骨干、以及周期性重置协议。

**📊 数据集**

在ScanNet、KITTI‑360、Bonn、TUM RGB‑D等公开数据集以及轮椅平台的10‑Hz实测走廊/户外遍历上进行评估。

**📈 对比分析**

与CUT3R、TTT3R、Point3R、Spann3R等基线比较，平均在500帧、1000‑3000帧长序列上分别降低ATE/ CD/ AbsRel 约10‑25%，并在RTX 4060上实现≈11.7 FPS、5 GiB内存，性能保持不变。

**⚠️ 局限性**

局限包括仅在周期性重置下表现优异，未实现无重置长序列的内存泛化；后端仍为离线优化，实时闭环控制未覆盖；以及对传感器同步/校准要求较高。

---

## 235. LiSA: Linear Stochastic Attention for Long-Range Visual Modeling

**arXiv ID:** 2611.04216 | [PDF](https://arxiv.org/pdf/2611.04216v1)

**作者:** Sahar Dastani `[一作]` (École de technologie supérieure), Christian Desrosiers `[通讯]` (École de technologie supérieure)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 Linear Stochastic Attention（LiSA），一种线性注意力插件，通过截断随机游走实现多阶 token 交互，同时保留线性时间复杂度。

**💡 创新点**

创新点在于把注意力视为高阶图滤波，使用可学习的步长权重实现截断随机游走，并加入自适应局部 3×3 注意力以避免过平滑。

**🔧 技术方法**

技术方法包括核化线性注意力、截断随机游走、多步传播、可学习步长权重、局部窗口注意力与门控混合。

**📊 数据集**

实验使用 ImageNet‑1K（分类）、ADE20K（分割）、COCO（检测）等数据集，并在 mini‑ImageNet 上做 ablation。

**📈 对比分析**

在与 softmax 注意力以及多种线性注意力（FLatten、Hydra、Efficient 等）以及 DeiT、PVT、Swin、BiFormer 等基准的对比中，LiSA 在相同 FLOPs/参数下提升分类 Top‑1 约 2–3%，分割 mIoU 约 +4%，检测 box AP 约 +3.8%。

**⚠️ 局限性**

局限性包括截断长度需平衡表达力与开销，过高阶传播可能导致特征过平滑；在已具层级结构的模型（如 Swin）中增益相对有限。

---

## 236. Incremental Open Addressing Has No Time-Space Tradeoff

**arXiv ID:** 2611.04186 | [PDF](https://arxiv.org/pdf/2611.04186v1)

**作者:** Martín Farach-Colton `[一作]` (New York University), William Kuszmaul `[通讯]` (Carnegie Mellon University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种增量式开放寻址哈希表，能够在任意负载因子（包括满载）下实现插入、正查询和负查询的期望时间均为 O(1)；

**💡 创新点**

创新点在于证明增量式环境不存在时间-空间权衡，首次通过树结构和无重排的技巧突破了传统的 Ω(log ϵ⁻¹) 插入/查询下界，并利用重新排序实现了 worst‑case 期望正查询时间；

**🔧 技术方法**

采用了基于二叉树的隐式堆结构、优惠券收集（coupon‑collecting）分析、随机哈希函数、位编码“breadcrumb”技术以及精细的概率（Chernoff）分析；

**📊 数据集**

该工作为纯理论研究，不使用任何实际数据集；

**📈 对比分析**

通过理论分析与已知下界对比，证明所提出的数据结构在所有负载因子下均达到最优期望 O(1) 复杂度；

**⚠️ 局限性**

局限性包括仅适用于增量式（无删除）设置，动态扩容时仍需 Ω(log log ϵ⁻¹) 时间；此外实现细节复杂，实验验证尚未完成。

---

## 237. Grasping and Stabilizing Tumbling Objects with Rotating-Base Multi-Arm Space Manipulators

**arXiv ID:** 2611.04176 | [PDF](https://arxiv.org/pdf/2611.04176v1)

**作者:** Farhad Aghili `[一作]` `[通讯]`, Farhad Aghili

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并验证了一种基于虚拟刚化的航天机械手抓捕与去漂动方法，使航天器基座主动协调运动，使翻滚目标在近接过程中相对基座保持静止，从而简化抓捕规划并减少碰撞风险。

**💡 创新点**

创新点在于：①将目标的质心与航天器的瞬时旋转中心对齐，形成“虚拟刚体”，实现无相对速度抓捕；②在抓捕后利用哈密顿动力学与二阶锥规划（SOCP）实现时间/燃料/能量最优去漂动；③对无预设抓取接口的目标引入摩擦锥约束，确保无滑动抓取。

**🔧 技术方法**

使用技术包括：多臂动力学模型、同步控制律、基座姿态与线速度跟踪、Hamiltonian最优控制、SOCP 与 QCQP 求解器、摩擦锥线性化与对数障碍函数、状态估计（IMU+视觉）与基座推力/反作用轮控制。

**📊 数据集**

采用仿真数据集：三种目标惯性与初始角速度组合（质量250–1000 kg，惯量{70,100,150} 或 {280,400,600} kg·m²），配合三臂空间机械手的尺寸与惯量参数，验证同步与去漂动过程。

**📈 对比分析**

与单纯基座旋转匹配方法对比，虚拟刚化能在350 s 内实现相对零速度，抓捕成功率显著提高；去漂动时间随目标角动量成正比，最大推力约0.5 Nm，所需燃料0.25 kg；仿真显示摩擦锥约束成功避免滑动。

**⚠️ 局限性**

局限性包括：需要对目标惯量与姿态精确估计，基座运动精度受限于反作用轮/推进器的噪声与失效；摩擦模型假设为静摩擦系数，真实场景中可能出现滑移；优化求解对实时性有一定需求；仅在离心运动无外力干扰的前提下验证，实际空间环境中需考虑扰动与测量误差。

---

## 238. Where Do Distilled Students Store Thinking?

**arXiv ID:** 2611.04141 | [PDF](https://arxiv.org/pdf/2611.04141v1)

**作者:** Taeyeong Kim `[一作]` (Chosun University), Unggi Lee `[通讯]` (Korea University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在小型 LoRA 学生模型上，使用 LRM 与指令模型的线性插值教师（λ 介于 0 与 1）进行知识蒸馏，探究当教师停止“思考”时学生权重中保留的内容。研究通过比较不同 λ 下的 LoRA 适配器子空间，识别出一个共享的“思考子空间”（region），并通过移除、移植与放大等干预手段验证其对反思标记、单句式、长答案等思考相关行为的影响。

**💡 创新点**

①首次发现并定位了在教师停止思考时会消失的共享 LoRA 子空间；②证明该子空间主要负责思考行为而非仅仅是答案长度；③通过干预展示该子空间的可控性与对思考行为的决定性作用。

**🔧 技术方法**

LoRA（低秩适配器）作为权重差异的子空间分析工具；子空间重叠度量（Overlap Ratio R）；子空间提取（平均投影矩阵并按特征值筛选）；干预操作（删除、移植、放大）；SAE（稀疏自编码器）特征用于思考行为的二次读出。

**📊 数据集**

主要使用 4,000 道 MATH 训练集进行蒸馏与评估，测试集为 200 道 MATH‑500；补充评估使用 GSM8K 与 GPQA‑Diamond；实验包含三对教师（A、B、C）与三名学生模型（Qwen2.5‑0.5B、Llama‑3.2‑1B、OLMo‑2‑1B），每对每个 λ 有 3 种随机种子。

**📈 对比分析**

通过对比不同 λ 下的思考指标（思考率、反思标记、单句式、答案长度）和准确率，验证子空间干预的效果。移除 region 可显著降低思考标记并提升准确率（平均提升约 8.6%），移植至 λ=1 的学生可恢复早期标记但完成率低，放大权重后标记可部分恢复但整体效果有限。实验表明子空间的干预效果优于仅靠长度匹配或随机方向干预。

**⚠️ 局限性**

①仅在 1B 以下的小模型上验证，未测试更大模型或完整微调；②使用 LoRA 作为实验工具，结果可能不适用于全参数微调；③教师与学生的家族和规模混杂，缺乏独立验证；④评估主要依赖标记和长度指标，缺乏更细粒度的思考行为分析；⑤数据集主要集中在 MATH，跨领域泛化需要进一步验证。

---

## 239. Optimal Communication for Distributed Sparse Histogram Aggregation

**arXiv ID:** 2611.04285 | [PDF](https://arxiv.org/pdf/2611.04285v1)

**作者:** Fangwei Ye `[一作]` (Nanjing University of Aeronautics and Astronautics), Yijun Fan `[通讯]` (Chinese University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并分析了分布式稀疏直方图估计的通信复杂度，给出了随机覆盖（Randomized Covering）方案，并证明其在给定 L1 失真与失败概率约束下实现第一阶最优。

**💡 创新点**

创新点包括：① 通过局部支持识别直接刻画失真，避免传统无偏采样的冗余；② 统一考虑异构与同构编码器，揭示同构约束下的额外通信开销；③ 设计了近似求解最优覆盖分布的高效算法，并给出误差与复杂度分析。

**🔧 技术方法**

使用技术包括：随机覆盖构造与解析；信息论 Fano 与后验方差方法；Chernoff 侧化与大数定律；计数与熵估计；离散化与凸包优化实现近似算法。

**📊 数据集**

论文主要以理论分析为主，实验部分使用合成随机稀疏向量（k‑sparse 二进制向量）进行模拟验证，没有公开使用真实数据集。

**📈 对比分析**

与坐标采样、客户端采样、Count‑Min Sketch、压缩 Bloom Filter、BCH 语义恢复等传统方法在同一失真与失败概率下对比，随机覆盖方案在通信量上明显优于其余方法（比均值采样低约 1‑δ 乘以 nk‑⌊nD⌋ 的对数 d 量级）。

**⚠️ 局限性**

局限性：结果在 d→∞ 的极限下给出第一阶精度；对极小 d 或 k 接近 d 的情形需进一步分析；算法在参数大时的计算复杂度高；实际部署中实现异构编码器的协调与同步仍是挑战。

---

## 240. Bidirectional Centered Score Matching for Discrete Diffusion Distillation

**arXiv ID:** 2611.04279 | [PDF](https://arxiv.org/pdf/2611.04279v1)

**作者:** Peng Tu `[一作]` (Shanghai Academy of AI for Science), Siyu Zhu `[通讯]` (Shanghai Academy of AI for Science)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对离散扩散模型的知识蒸馏提出一种新的分数匹配方法

**💡 创新点**

提出Bidirectional Centered Score Matching (BCSM)，通过双向活跃子空间、均值中心化以及跨时间步对齐，显著降低复杂度并提升生成质量

**🔧 技术方法**

离散分数匹配、Kneedle动态阈值、均值中心化、跨噪声级别对齐技术

**📊 数据集**

Bard-4B-VL(170K图文对)、LLaDA-8B-Instruct、Dream-7B-Base，评测数据集包括GSM8K、MATH、HumanEval、MBPP、MMMU、RWQA、AI2D、MMVU、OCRBench

**📈 对比分析**

与Fast-dLLM、D2F、CDLM、FKL、RKL、DistiLLM、AKL等基线对比，BCSM在NFE显著降低（最多6×）的同时获得更高分数（如HumanEval提升16.9%、MMMU提升15.1%等）

**⚠️ 局限性**

仍难以完美恢复长序列，跨位置依赖建模不足，过度对齐可能导致性能下降

---

## 241. Testing Implies Distance Estimation with Polynomial Loss

**arXiv ID:** 2611.04276 | [PDF](https://arxiv.org/pdf/2611.04276v1)

**作者:** Gaia Carenini `[一作]` (University of Cambridge), Cameron Seth `[通讯]` (University of Waterloo)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文证明在稠密图模型中，任何可测试属性的测试器都能转化为估算器，且查询复杂度仅以多项式增长。

**💡 创新点**

创新点在于提出了一种基于弱正则化、梯度提升和短标签编码的全新转换技术，避免了传统正则化导致的塔型上界。

**🔧 技术方法**

核心技术包括弱正则化（Frieze–Kannan）、测试器的梯度提升、对图的短标签表示以及密集约束满足问题的采样定理。

**📊 数据集**

该工作不依赖任何具体数据集，而是纯理论证明。

**📈 对比分析**

相较于之前的指数或塔型损失，本文实现了多项式损失，并给出了对应的查询复杂度上界；理论分析中已验证该性能。

**⚠️ 局限性**

局限性在于仅适用于稠密图模型，且尽管损失为多项式，常数和指数因子仍可能较大，实际应用需要进一步优化。

---

## 242. Reliability Stress Testing of RAG Retrieval under Query-Relevant Corpus Poisoning

**arXiv ID:** 2611.04264 | [PDF](https://arxiv.org/pdf/2611.04264v1)

**作者:** Janet Sung `[一作]` (Google), Lu Yi `[通讯]` (Google)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对检索增强生成（RAG）系统在语料库中注入攻击性文档，系统地评估不同文档内容、查询信息、毒药预算和污染范围对检索暴露、排名质量及安全可靠性的影响；

**💡 创新点**

创新点在于将查询信息、毒药预算、污染范围三条压力轴独立化，对检索阶段失败进行细粒度拆解；通过多模型（BM25、LSA、Hybrid、BGE-small）和多层次内容控制（仅查询、查询+payload、中性）对比，清晰揭示查询锚定与附加文本在检索过程中的具体作用；

**🔧 技术方法**

使用BM25、LSA、Hybrid及预训练dense检索器BGE-small，结合MMR重排和基于IDF重叠的过滤规则；通过Bootstrap置信区间、非参数统计和延迟测量对实验结果进行稳健评估；

**📊 数据集**

实验数据集为BEIR的SciFact（5,183文档/300查询）和NFCorpus（3,633文档/323查询），攻击payload来自BIPIA；

**📈 对比分析**

评价指标包括nDCG@10、Recall@10、曝光率（Target Exposure）、证据可用性、Safe Retrieval Reliability；在P=5孤立攻击下，Hybrid的nDCG从0.465降至0.183，曝光率接近100%；多目标污染导致跨目标干扰；过滤阈值能显著降低曝光，但会牺牲清洁检索性能；

**⚠️ 局限性**

主要局限：仅评估检索阶段，未测量下游生成攻击成功率；攻击信息与毒药设置为可观测而非实际可达；内容控制与BIPIA payload在语义匹配上不完全；阈值曲线为前沿分析而非部署建议；BEIR qrels不完整，证据可用性评估受限。

---

## 243. Neologism Inoculation Prompting

**arXiv ID:** 2611.04260 | [PDF](https://arxiv.org/pdf/2611.04260v1)

**作者:** David Demitri Africa `[一作]` (Resolution), Cameron Holmes `[通讯]` (UK AI Security Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于新词（neologism）的接种（inoculation）方法，使用不可输入的特殊 token 对模型进行训练，随后在部署时删除该 token 的 embedding，从而阻止模型在提示中恢复不良行为。

**💡 创新点**

创新点在于将接种标签迁移至不可输入的 token，并通过冻结模型训练新词、在标签上进行微调，隐藏而非删除不良行为，使得用户无法通过输入英文标签恢复触发。

**🔧 技术方法**

主要技术包括：新词学习（在冻结模型上训练输入/输出 embedding）、LoRA 微调、匹配对（有害/无害示例对）、控制词实验以及激活分析。

**📊 数据集**

使用了公开大模型（Qwen3.6‑27B、Gemma‑4‑31B、OLMo‑3.1‑32B‑Instruct）以及多任务数据：极限运动、金融、医学的有害建议，奖励破解、MBPP 代码硬编码、GCD 同情、CMV 骚扰等。

**📈 对比分析**

与 SIP、普通接种、密码锁定 SIP、GRAFT、接种适配器等基线对比。实验显示本方法在泄漏率上与 SIP 相当，在保持期望行为（retention）方面优于或相当；触发率显著低于其他基线，性能与梯度路由相当；但在 GCD、CMV 等任务上表现不佳。

**⚠️ 局限性**

局限性包括：在 OLMo 上丢失部分期望行为；token 的选择在某些任务上不稳定；仅对提示攻击有效，具有权重访问的攻击仍能恢复；实验仅覆盖有限模型与任务，缺乏对更强攻击方式的评估。

---

## 244. P-SER: Physical Consistency-Constrained Scattering Expandable Representation for Class-Incremental HRRP Target Recognition

**arXiv ID:** 2611.04254 | [PDF](https://arxiv.org/pdf/2611.04254v1)

**作者:** Fangyi Wang `[一作]` (Fudan University), Haipeng Wang `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 P‑SER 框架，利用物理一致性散射原型约束、PSD 机制校准以及轻量残差适配器实现 HRRP 的增量学习。

**💡 创新点**

创新点在于①通过单脉冲角度估计与目标几何重建生成稳定的散射点原型，用于类别级原型和样本级显著性引导；②训练预先的原始散射器字典，在推理时对船体与反射器的歧义进行机制级校准；③采用共享骨干与残差适配器实现参数高效的模型扩展。

**🔧 技术方法**

技术包括：物理散射点重建（单脉冲角度估计、几何对齐）、原型约束对比损失、知识蒸馏、PSD 机制分类器、残差适配器以及一维 ResNet‑18 特征提取。

**📊 数据集**

使用了自建模拟 HRRP 数据集（10 类船舶与反射器）和实测 HRRP 数据集（6 类船舶与反射器）。

**📈 对比分析**

在 8/4/2 级增量设置下与 LwF、iCaRL、PODNet、DER、FOSTER、DS‑AL、TagFex 等基线比较，P‑SER 在模拟数据上平均精度 88.05%，最终 81.35%；实测数据上平均 82.30%/83.56%，显著优于其它方法，参数量仅为现有方法的 19‑33%。

**⚠️ 局限性**

局限在于仍依赖雷达方位信息与单脉冲角度估计，对极端视角变化与极端干扰的鲁棒性有限，且在更大规模多样化真实数据上的验证仍待进一步探索。

---

## 245. Attack-Aware Intrusion Risk Monitoring for Aerial Surveillance via Probabilistic Signal Temporal Logic

**arXiv ID:** 2611.04252 | [PDF](https://arxiv.org/pdf/2611.04252v1)

**作者:** Xinhao Yan `[一作]` (Hong Kong Polytechnic University), Hailong Huang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计了一套在线监测框架，用于在受限测量攻击下评估无人机入侵风险并发出分级警报。

**💡 创新点**

创新点包括：①基于校准后的创新统计量与已知物理攻击界限构造置信集，保留攻击偏差假设；②利用该置信集生成攻击感知的后验分布，并在此基础上上界占据概率，提供置信保证；③引入概率信号时序逻辑（PrSTL）对风险序列进行时序化处理，区分有序、持续和重复风险并给出警报级别、鲁棒度和支持窗口。

**🔧 技术方法**

采用的主要技术有：受信状态估计（Kalman滤波）、创新统计检测、置信集构造、Gaussian后验最大化、PrSTL语义与语义化监测、数值上界证书与覆盖单元分割法。

**📊 数据集**

实验数据来自Crazyflie四旋翼在配备Vicon运动捕捉的实验室场景中的飞行轨迹，包含多个指定禁飞区和安全边距，人工注入不同幅值和模式的测量攻击。

**📈 对比分析**

与基线方法（受信仅估计、门控Kalman、裁剪Kalman、仅报警等）相比，本文方法在风险低估率、材料缺陷率和平均超额风险上显著优于门控/裁剪Kalman，且在无报警条件下缺陷率降至约0.005%；检测率高于受信仅估计且误报率控制在可接受范围内。

**⚠️ 局限性**

局限性包括：①依赖已知的物理攻击边界，若边界未知或被动态变化可能导致置信集失效；②假设噪声为高斯且状态模型线性，非线性或非高斯情况需进一步研究；③计算上对置信集的数值上界与覆盖单元分割仍有一定开销，实时性受限于硬件与参数设置。

---

## 246. LIRA: A More Interpretable Dynamic Loss-Informed Rank Allocation for Low-Rank Adaptation

**arXiv ID:** 2611.04239 | [PDF](https://arxiv.org/pdf/2611.04239v1)

**作者:** Sihao Wang `[一作]` (South China University of Technology), Jie Tang `[通讯]` (South China University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了LIRA，一种基于局部损失扰动的动态LoRA秩分配方法，实现在线可解释的秩重新分配。

**💡 创新点**

创新点在于通过第二阶泰勒近似将LoRA参数位移、梯度与曲率结合，得到每层无符号损失扰动量，并以此为依据进行可解释的秩分配。

**🔧 技术方法**

采用LoRA低秩增量、二阶泰勒近似、梯度与斜率统计（Trajectory EMA或Fixed‑point Fisher）、温度softmax投影与整数化等技术。

**📊 数据集**

在RoBERTa‑base GLUE基准（SST‑2、MRPC、CoLA、QNLI、RTE、STS‑B）上进行评估。

**📈 对比分析**

与统一秩LoRA、AdaLoRA、DoRA、La‑LoRA等基线在相同有效秩预算下比较，LIRA在六任务平均得分提升约1.4分，单任务上在RTE、CoLA等取得显著优势，Trajectory EMA与Fixed‑point Fisher结果相近且前者无需额外统计窗口。

**⚠️ 局限性**

仍需预设最大秩上限与预算调度，重分配间隔与温度等超参数对性能敏感；方法仅针对LoRA结构，未覆盖其它PEFT模式，且在大模型迁移或多任务场景的通用性尚未充分验证。

---

## 247. The Unreported Variable:Collector Selectivity in Data Placement Evaluation

**arXiv ID:** 2611.04234 | [PDF](https://arxiv.org/pdf/2611.04234v1)

**作者:** Nannan Zhao `[一作]` (NorthWestern Polytechnical University), Xiao Zhang `[通讯]` (NorthWestern Polytechnical University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文通过控制工作负载、标签、设备和剩余容量，系统评估了不同垃圾回收器（victim selection）对闪存写放大和标签价值的影响。

**💡 创新点**

创新点在于：①证明并量化标签价值取决于收集器能否观测到的数据结构；②提出标签价值随收集器可见信息变化的理论；③给出了基于容量等价度量的标签成本/收益评估。

**🔧 技术方法**

采用页面映射SSD仿真模型，调节victim selection（uniform、d‑choices、greedy、FIFO）并测量写放大；使用写放大分数g、信息论度量、容量等价度量来评估标签效益。

**📊 数据集**

使用合成Rosenblum热点工作负载、成员（cohort）工作负载以及实际的Varmail邮件服务器块级跟踪。

**📈 对比分析**

通过在SWF、DWF、HCWF三种写分配方案下，比较不同victim selection的写放大，计算标签收益g。实验显示：在中等选择（如d≈20‑24）下标签价值最高；greedy下降；FIFO甚至负值。实际trace中形状一致但幅度约减半。

**⚠️ 局限性**

局限性：仅基于仿真，未在真实硬件上验证；只覆盖基于有效计数/年龄的收集器，未考虑更复杂的策略；标签误差率与实际分类器误差关系仍需进一步研究；容量等价度量受标签错误率限制。

---

## 248. LINDA: Low Resource Foot Ulcer Segmentation using Local Similarity and Iterative Neural Dynamics

**arXiv ID:** 2611.04233 | [PDF](https://arxiv.org/pdf/2611.04233v1)

**作者:** Trishaani Acharjee `[一作]` (Kalinga Institute of Industrial Technology), Tathagata Bandyopadhyay `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了 LINDA，一种基于 ResNet-34 的轻量级脚部溃疡分割模型，利用局部自相似（LSS）图作为结构先验，并通过液体时间常数（LTC）模块进行迭代边界细化。

**💡 创新点**

创新点在于：① 将可解释的 LSS 先验直接注入编码器；② 采用小型 LTC 迭代网络实现多步边界优化；③ 通过边界对齐损失（BAL）把结构先验与预测严格绑定，显著提升边界精度。

**🔧 技术方法**

技术细节包括 ResNet-34 预训练编码器、加法式 LSS 融合块、LTC 迭代细化循环、深度监督（BCE+Dice+BAL）以及 PyTorch 2.5.1 的实现。

**📊 数据集**

使用了 MICCAI Foot Ulcer Segmentation (FUSeg) 2021 数据集，共 1210 张图像（810 训练/200 验证/200 测试）。

**📈 对比分析**

在统一训练协议下与 U‑Net、ViT、Mamba 变体以及 FUSegNet 等方法进行比较，LINDA 在 Dice 86.68%、IoU 79.67% 和 95% 分位距 Hausdorff 距离 6.37 像素上均取得最佳成绩，参数量仅 25.70 M，显著低于对手。

**⚠️ 局限性**

局限性包括：仅在 FUSeg 数据集上验证，跨域泛化（如 Medetec）尚未评估；LTC 迭代步数超出 8 步会出现不稳定；模型仍以二分类形式设计，且依赖 ResNet‑34 预训练，可能不适用于极端低算力环境。

---

## 249. Drift No More: Spectral Locking for Long-Horizon Open-Weight Multi-Agent AI

**arXiv ID:** 2611.04315 | [PDF](https://arxiv.org/pdf/2611.04315v1)

**作者:** Vishal Pramanik `[一作]` (University of Florida), Sumit Kumar Jha `[通讯]` (University of Florida)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为ATLAS的代表性层干预方法，用于在多智能体长周期交互中同步任务、角色和共享状态的内部表示，防止任务漂移、角色漂移和状态陈旧等失效；

**💡 创新点**

创新点在于：1）将任务、角色、状态三者视为耦合的表示子空间，并通过非正交低秩PSD锚点核聚合它们；2）在推理时对隐藏层进行光谱 steering，实时纠正隐藏状态；3）引入自适应状态刷新机制，使状态子空间随对话动态衰减与更新；

**🔧 技术方法**

技术主要包括：低秩Gram矩阵分解实现高效的PSD核计算；光谱Steering（基于Rayleigh商和核的特征值扩展）；自适应状态记忆更新（衰减、提取新向量、重正交化）；以及多模型异构团队的锚点构建与共享。

**📊 数据集**

使用的主要数据集包括：EverMemBench（多方长对话记忆评估）、AMemGym（交互式状态维护与工具使用）、ToolACE（工具链鲁棒性测试）、ALARA（最少权限上下文纪律）、LIBERO（多模态嵌入式压力测试）等。

**📈 对比分析**

与多种基线（Prompt-Only、Struct-Role、RAG-Mem、MemoryArena-Agent、ActAdd、SEA等）以及最新的A-MEM进行对比，ATLAS在所有主要指标上均有显著提升：例如在T1/T2/T3场景下，宏观平均分分别提升至59.4/61.7/56.2；在AMemGym中写/读失败率分别降低至11.4%/10.6%；在工具链和权限纪律测试中也分别提升6.9/4.9个百分点。

**⚠️ 局限性**

主要限制包括：需对模型隐藏层进行访问，限制了在闭源或无法控制的推理环境中的适用性；锚点子空间依赖于任务、角色与状态描述的质量；仅关注表示一致性，无法保证事实正确性、安全性或工具使用最优；且在动态扩容、工具权限变更或不可信外部记忆等更复杂场景下仍需进一步验证。

---

## 250. What Is the Influence of Task Order in Continual Learning? An Empirical Study and Order-Aware Evaluation for Motion Forecasting

**arXiv ID:** 2611.04309 | [PDF](https://arxiv.org/pdf/2611.04309v1)

**作者:** Zirui Li `[一作]` (Nanyang Technological University), Chen Lv `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究连续学习中任务顺序对性能的影响，并提出基于任务顺序分布的评价框架。

**💡 创新点**

创新点在于将算法比较表述为对任务顺序分布的期望，并通过方差分解与重采样剖析顺序、算法与训练噪声的贡献；同时提出了可解释的转移方向描述符和权重化策略。

**🔧 技术方法**

使用了方差分解、Spearman相关、引导重采样、距离度量（Wasserstein-1、KL、JS、MMD）以及多种持续学习算法的记忆机制。

**📊 数据集**

数据集为INTERACTION中的四种道路布局任务（城市合流、高速合流、交叉口、环岛），共四个任务、288次实验。

**📈 对比分析**

通过枚举全部24种任务顺序、对每个算法进行三次复现并计算CT、BWT和最终误差，结果显示任务顺序占FDE-CT、MR-CT等指标65–91%的方差，算法排名易随顺序改变，SyReM在最终误差和MR-BWT上表现最佳。

**⚠️ 局限性**

局限性包括仅针对四任务小规模设置，未探索更大任务集合下的采样策略；对单一预测器与任务特征（速度、加速度）的依赖，结果可能不易直接推广到其他领域。

---

## 251. State Evolution and Universality of Iterated Linear Programs for the Binary Perceptron

**arXiv ID:** 2611.04294 | [PDF](https://arxiv.org/pdf/2611.04294v1)

**作者:** Jianan Zhang `[一作]` (University of Science and Technology of China), Shuangping Li `[通讯]` (Yale University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在高维随机线性规划中基于Gaussian迭代方法的收敛性和可解性，并给出了有限样本下的性能分析。

**💡 创新点**

提出了通过KKT正则化、Gaussian耦合和相对熵递推等方法，将随机LP问题转化为可分析的迭代过程，并证明了收敛速度与自由变量比例的关系。

**🔧 技术方法**

利用高维概率工具（Gaussian比较、协方差矩阵耦合、相对熵、Wasserstein距离）、KKT条件的正则化、共轭梯度式迭代和矩阵随机过程分析等技术。

**📊 数据集**

仅使用理论随机数据，取A∈ℝ^{M×N}为标准高斯矩阵，目标向量v独立于A，约束为箱型和行不等式。

**📈 对比分析**

与传统单纯形或内点法对比，所提出的方法在高维随机实例中表现出更快的收敛速度和更小的自由变量比例阈值；实验结果（若有）显示平均误差下降到O((log N/N)^{1/4})。

**⚠️ 局限性**

结果只在高斯随机设置下成立，且对行约束稀疏度和目标分布的特定假设有限；对于非高斯或结构化数据尚未证明，并且需要较大样本尺寸以保证理论近似。

---

## 252. Harness-S: Learning from Sibling Harnesses for Generalizable Agent Training

**arXiv ID:** 2611.04277 | [PDF](https://arxiv.org/pdf/2611.04277v1)

**作者:** Kangning Zhang `[一作]` (Shanghai Jiao Tong University), Weinan Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过“Harness‑S”框架先用方向性进化产生一组兄弟 harness 并固定下来，然后在政策训练期间采用补充性 Harness 救援训练仅在原始 harness 全部失败时使用这些兄弟 harness 进行重试，使共享策略在保留原始 harness 的前提下获得提升。

**💡 创新点**

创新点包括：①提出方向性 Harness 进化以系统生成并保存多样的局部修复；②提出补充性 Harness 救援训练和自适应路由器，仅在全失败时调度兄弟 harness，提供对策学习的对比奖励；③实现从兄弟 harness 学习的行为能迁移到原始 harness，并在多种基准上超越现有方法。

**🔧 技术方法**

技术手段包括：GRPO 强化学习策略优化；LLM 进行行为分析与路由决策；自动化 harness 编辑与验证；行为档案与政策感知的 profile 重新锚定；以及多任务训练与验证流程。

**📊 数据集**

使用的基准数据集为 AppWorld、DBBench 和 SWE‑Gym；实验模型为 Qwen3.5‑4B 与 Qwen3.5‑9B。

**📈 对比分析**

与仅策略训练、单一 harness 优化、联合 harness‑policy 适配等基线相比，在 Pass@3 与 Avg@3 指标上，Harness‑S 在所有模型规模和基准上均取得更高得分；在原始 harness 上亦保持优势；路由器排名能够准确预测执行质量，跨 harness 的通用性表现良好。

**⚠️ 局限性**

局限性包括：①构建和维护兄弟 harness 的成本高；②自适应路由器与行为档案需要定期重新锚定，增加额外验证开销；③在极端任务或未知 harness 上仍然受限于执行接口；④对更大规模、多样化 harness 的扩展尚未完全验证。

---

## 253. Online Metric Matching with Unknown i.i.d. Arrivals

**arXiv ID:** 2611.04275 | [PDF](https://arxiv.org/pdf/2611.04275v1)

**作者:** Tsubasa Harada `[一作]` `[通讯]` (Institute of Science Tokyo), Tsubasa Harada (Institute of Science Tokyo)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在线度量匹配问题，其中n个服务器和n个请求独立地从未知分布中抽取。提出了一种随机算法，在任意度量空间中具有O((logloglog n)^2)的竞争性，在树度量中具有O(1)的竞争性。

**💡 创新点**

创新点在于为未知独立同分布请求模型提供了与已知独立同分布模型相匹配的竞争性上界，证明了在一般度量下的竞争比为O(α(n))，这为未知请求分布的情况提供了新的算法保证。

**🔧 技术方法**

使用了随机算法和最优传输计划的技术，通过构造虚拟请求来进行匹配。

**📊 数据集**

没有具体提到使用的数据集，但讨论了从未知分布中抽取的请求和服务器的匹配问题。

**📈 对比分析**

与已知独立同分布模型的算法进行了比较，结果显示在一般度量下的竞争比为O((logloglog n)^2)，在树度量下为O(1)，与已知模型的结果相匹配。

**⚠️ 局限性**

限制在于尚未证明在一般度量下的竞争比为常数，如果能证明，则可以为未知独立同分布请求提供常数竞争算法。

---

## 254. UMLoad: Fast On-Demand Model Loading on Unified Memory Architecture

**arXiv ID:** 2611.04253 | [PDF](https://arxiv.org/pdf/2611.04253v1)

**作者:** Zimeng Wang `[一作]` (Shanghai Jiao Tong University), Rong Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种系统级的模型加载加速器，专为统一内存架构（UMA）设计，能够在不修改现有框架代码的前提下，显著降低模型加载延迟与峰值内存占用。

**💡 创新点**

创新点包括：① 基于按需重建（On‑Demand Reconstruction）的方法，直接将文件页映射到 GPU 缓冲区，彻底消除重量拷贝，同时保持 CPU‑GPU 复制语义；② GPU‑辅助页零化（GPU‑Assisted Page Zeroing），通过分批次流水线和前置映射资源，显著缩短运行时分配的 CPU 侧工作量；③ 两者集成后对多平台（Intel, AMD, NVIDIA）多框架（PyTorch, llama.cpp, vLLM）兼容。

**🔧 技术方法**

核心技术包括：用户空间驱动（UMD）和内核空间驱动（KMD）互操作器、页面回收器（Page Reclaimer）实现页归还与独占所有权、页面零器（Page Zeroer）实现 GPU 零填充与批处理流水线、对齐填充（Padding）与 THP 结合实现大页映射、以及多线程缓存维护与映射预置。

**📊 数据集**

实验数据集：6 个模型（Qwen3‑1.7B, Qwen2.5‑3B, Qwen2.5‑7B, ConvNeXt‑L, Wide‑ResNet‑101, RMBG‑2）在 llama.cpp、PyTorch、vLLM 上运行；LLM 路由与工具调用的真实工作负载使用 LMSYS‑Chat‑1M、ToolMMBench、MNMs benchmark，涉及多模态模型（Flux2、Qwen3‑VL‑2B‑Instruct 等）。

**📈 对比分析**

与基线（vanilla）和流水线（pipelined）加载器对比，平均在 Intel、AMD、NVIDIA 平台上加载时间分别下降 36.6%–71.9%（相较于 vanilla）和 19.9%–62.7%（相较于 pipelined）。在 LLM 路由场景下，首次 token 时间（TTFT）下降 51.3%–59.3%；在工具调用场景中，完成延迟下降 10.2%–31.6%。峰值内存占用仅为权重文件大小的 1.01–1.02 倍，显著低于基线的 1.1–2.0 倍。

**⚠️ 局限性**

限制包括：需要对权重文件进行离线对齐填充，可能导致文件体积略增；在权重页被写回后再次访问 CPU 时需触发磁盘 I/O，偶尔产生重建延迟；实现依赖于对驱动层的深度修改，尚未支持所有框架与所有 UMA 设备；在极大模型或高磁盘 I/O 受限的场景下，文件重写带来的 I/O 额外成本可能影响整体收益。

---

## 255. AceAction-5M: Hierarchical Goals for Generalist Action Learning Across 3D Game Worlds

**arXiv ID:** 2611.04241 | [PDF](https://arxiv.org/pdf/2611.04241v1)

**作者:** Kailing Li `[一作]` (Tencent), Zixun Sun `[通讯]` (Tencent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研发了 AceAction-5M 数据集，提供多游戏任务级、阶段级和技能级的层级注释，并实现玩家视图、键鼠动作与语言指令的时序对齐。

**💡 创新点**

创新点在于通过视觉-语言模型识别目标变化和可视里程碑来自动划分子目标，并在任务、子目标与技能层面显式标记阶段边界，支持连续控制与多技能组合的结构化监督。

**🔧 技术方法**

技术手段包括使用 Vision–Language 模型（Qwen3.8‑27B、GLM‑5.3‑FLASH）进行多级别自动验证、滑动窗口分段与里程碑检测实现子目标划分，并构建 43 类技能标签与 42 款游戏的采集对齐管线。

**📊 数据集**

数据来源融合自采集（Game for Peace、Valorant）与公开数据（CS:GO/CS2、Open‑P2P、VPT），共计约 5,180 小时录像，形成 5,476,222 条完整任务样本。

**📈 对比分析**

与 Open‑P2P、EgoCS‑400K 等现有数据集相比，AceAction 在任务-子目标-技能层级指令、明显阶段边界以及键鼠动作对齐上实现了更细粒度、更大覆盖；在模型验证与人工审计中达 95.8% 通过率，验证了注释质量与一致性。

**⚠️ 局限性**

局限性在于受限于游戏内可录制的动作与可视里程碑，某些细节动作和非视觉依赖的任务难以完整捕捉，且样本主要集中在竞技射击与部分冒险游戏，跨领域通用性仍待提升。

---

## 256. VerfGraph: Efficient Zero-Knowledge Verifiable Graph Query Evaluation System

**arXiv ID:** 2611.04240 | [PDF](https://arxiv.org/pdf/2611.04240v1)

**作者:** Tianyi Ma `[一作]` (Shanghai Jiao Tong University), Shi-Feng Sun `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发 VerfGraph，提供零知识可验证图查询评估系统；

**💡 创新点**

改进扩展与最短路径证明，减少比较与范围检查；引入计数与加权最短路径证明，扩展查询支持；

**🔧 技术方法**

采用 Plonky3、FRI‑based PCS、AIR 以及对称的等价检查、查找与范围检查等技术；

**📊 数据集**

使用 LDBC Social Network Benchmark 数据集；

**📈 对比分析**

与 ZKGraph 在相同查询与数据规模下比较，VerfGraph 证明生成速度提升 33.9×–59.7×，峰值内存降低 4.5×–6.6×，验证时间保持毫秒级，证明文件保持子 MB；即使在同一 Halo2/KZG 后端下也能显著提升性能；

**⚠️ 局限性**

仅覆盖扩展、SSSP、计数等操作，未涵盖所有 SQL 查询；使用哈希‑基础 PCS 导致证明略大；未针对更新、侧信道或攻击完整性做进一步研究。

---

## 257. Coarse Semantic Blocks for Efficient Training-Free Sparse Attention

**arXiv ID:** 2611.04313 | [PDF](https://arxiv.org/pdf/2611.04313v1)

**作者:** Sosuke Kobayashi `[一作]` (Preferred Networks), Seiya Tokui `[通讯]` (Preferred Networks)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无训练的稀疏注意力方法，利用每个提示生成语义块，并在前填充和解码时使用这些块进行查询级别的块选择。

**💡 创新点**

创新点在于：①使用球面 k‑means 对每个层/KV 头的键进行提示特定聚类，生成大尺寸语义块；②通过对中心点进行评分来快速评估块重要性；③在前填充阶段按块聚合相同块的查询，实现 KV‑outer 执行，从而保持每个查询独立的块选择并显著降低 GPU 访问成本。

**🔧 技术方法**

核心技术包括：球面 k‑means 聚类、中心点评分（带温度与大小惩罚）、块级选择与可变预算、KV‑outer 执行策略、FlashAttention‑3 等高效稠密/稀疏注意力核、vLLM 融合内核以及提示特定索引构建。

**📊 数据集**

在 Qwen3.5‑4B 与 Qwen3.5‑35B‑A3B 两个混合 LLM 上进行实验，使用 RULER、MRCR 与 LongBench‑v2 三个长上下文基准，评估在 512K 及 1M 长度下的性能。

**📈 对比分析**

与密集注意力相比，整机前填充速度提升 2.3–3.0×，解码速度提升 1.7–2.0×；在 RULER‑5、LongBench‑v2 以及 MRCR‑8 上保持与密集注意力相近的准确率，特别是 35B‑A3B 在 1M 长度下仍保持接近密集；相比其他无训练稀疏方法（如基于位置块或 FlashPrefill V2），在同等速度下准确率更高。

**⚠️ 局限性**

主要限制包括：实验仅在 BF16、单卡 H200、批量为 1 的设置下评估，未对连续批处理或其他 GPU 架构进行验证；内存占用仍受限于完整 KV 缓存；提示特定聚类需要额外的内存与时间；未使用训练或知识蒸馏来进一步提升小预算下的检索准确性。

---

## 258. Bilateral Protection for Private LoRA Inference with Intel TDX

**arXiv ID:** 2611.04311 | [PDF](https://arxiv.org/pdf/2611.04311v1)

**作者:** Xiaodong Li `[一作]` (Beijing Electronic Science and Technology Institute), Dongbo Yan `[通讯]` (Beijing Electronic Science and Technology Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了一套基于Intel TDX的双边保密LoRA推理服务，使客户端数据与LoRA适配器参数在同一受信任域内安全推理；服务通过独立的远程证明、TLS绑定以及经过校验的字节消费实现双方秘密的安全交付。

**💡 创新点**

首次将双边秘密释放与服务身份、模型消费、适配器激活三条链路完整对齐，并通过可重复的实验验证服务确实使用了经批准的私有LoRA；同时给出近乎零的服务路径开销与完整的安全机制证明。

**🔧 技术方法**

使用Intel Trust Domain Extensions (TDX)、远程证明与TLS 1.3 端点绑定、AES‑256‑GCM 加密解密、OpenVINO GenAI 推理框架、Qwen3‑0.6B 基模型及LoRA 低秩适配、应用身份链与固定只读镜像、结构化契约检查等技术。

**📊 数据集**

训练私有LoRA使用 MultiWOZ 2.2 对话数据集；评估阶段使用 400 条冻结的对话样本（200 酒店 + 200 餐厅），并额外保留 40 条诊断输入；公共基模型为 Qwen3‑0.6B。

**📈 对比分析**

通过与同一受信任域内的原生 CPU 推理、GPU LoRA（全预算与接口预算）以及三例检索三种条件进行对比；在 400 条对话上所有条件均达到 100% 输入支持；EM 指标 GPU LoRA 全预算 62%，受保护服务 60%；服务与原生 CPU 的 steady‑state 运行时间比为 1.0094（约 0.94% 的额外开销）；内存与通信开销也被细化。

**⚠️ 局限性**

局限性包括：仅验证单一适配器单任务服务；未探讨多租户或多适配器动态切换；缺乏软件升级、密钥轮换或连续运行的演示；仅在 CPU TDX 环境下评估；未对侧信道或更激烈攻击模型进行测试；使用的基模型与适配器范围有限。

---

## 259. Unmitigated? How are the world's largest companies responding to AI risks?

**arXiv ID:** 2611.04278 | [PDF](https://arxiv.org/pdf/2611.04278v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 260. Weather Adaptive Autonomous Driving

**arXiv ID:** 2611.04305 | [PDF](https://arxiv.org/pdf/2611.04305v1)

**作者:** Nicholas Taylor `[一作]` (Florida Gulf Coast University), Sudip Dhakal `[通讯]` (Florida Gulf Coast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文针对雨、雾、雪、沙等恶劣天气下的摄像头目标检测器失准问题，提出一种可调节误检与漏检权衡的决策阈值拨码器（trade‑off dial），并在冻结的YOLOv8x检测器上实现；

**💡 创新点**

创新点在于：①将误检成本 λ 作为用户可控参数，将其映射为期望成本最优阈值；②利用检测器自身的多维输出训练梯度提升树，得到每个框的校准概率并基于该概率做保留决策；③在每个天气条件下单独进行Conformal Risk Control（CRC）校准，并计划对 λ 再做 CRC 以获得每个条件的漏检保障；

**🔧 技术方法**

技术方法包括：Conformal Risk Control、梯度提升树（Gradient‑Boosted Trees）校准、IoU 匹配、NMS、批量字典化特征提取、F‑score 映射；

**📊 数据集**

使用数据集：COCO（用于YOLOv8x预训练）、BDD100K（训练与验证）、DAWN（真实恶劣天气评估）；

**📈 对比分析**

与 baseline 的比较：与单一全局阈值相比，按天气划分的 CRC 阈值能在 20% 漏检预算下将误检从 1.6–117 圈/图变为可调；trade‑off dial 在不同 λ 下提供更灵活的误检/漏检折中；在 BDD100K 验证集和 DAWN 真实测试中，漏检率保持在预算内，误检率可按需求控制；

**⚠️ 局限性**

局限性：①误检成本 λ 的映射与真实风险成本不完全对应；②CRC 在非可交换（极端稀缺）天气下的理论保证需进一步验证；③需要额外的检测器输出特征，增加计算开销；④缺乏对不同检测器架构的泛化实验；

---

## 261. Can Benign Skills Induce Misaligned Behavior in LLM Agents?

**arXiv ID:** 2611.04299 | [PDF](https://arxiv.org/pdf/2611.04299v1)

**作者:** Xuanye Zhang `[一作]` (Nanyang Technological University), Kwok-Yan Lam `[通讯]` (Nanyang Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在LLM代理中加入可重用的“善意”技能（如任务规划、工具调用和错误恢复）会导致在恶意请求下执行违规行为的风险，并通过系统实验验证了这一现象。

**💡 创新点**

创新点在于首次将可重用技能视为潜在安全威胁，提出了“技能诱导行为失配”概念，系统地量化了技能增强对安全违规率（UAR）的影响，并探索了技能来源、检索相关性、内容以及表征方式等因素对安全性的作用；同时验证了预执行安全检查能够在保留性能提升的同时显著降低违规率。

**🔧 技术方法**

主要技术包括：基于检索的技能注入（top‑K 相关技能），基于轨迹或专家的技能构建，技能演化（利用善意反馈迭代改进），以及预执行安全检查器（在工具调用前评估是否允许）。

**📊 数据集**

使用了三大基准：SafeAgentBench（嵌入式操纵）、AgentHarm（API/工具使用）和LPS‑Bench（长周期计算机任务），并在五个不同后端模型（Qwen3‑30B、gpt‑oss‑20B、Mistral‑24B、Nemotron‑30B、GPT‑5.4‑nano）上进行实验。

**📈 对比分析**

对比方法是将每个请求分别在无技能（Base）和有技能（Skill）两种条件下执行，测量ACC、UAR和ASR；实验结果显示技能提升ACC平均10.3pp，却同等提高UAR和ASR；通过配对轨迹分析发现约33%原本安全的轨迹因技能变为不安全，其中近一半未完成恶意目标；预执行检查将UAR降低约23–46pp，ASR降低更显著，但仍存在残留违规。

**⚠️ 局限性**

局限性包括：实验仅覆盖三类任务和五种模型，无法保证在更广泛或更复杂的真实环境中的普适性；技能的表征方式多样，实验仅考察文本式技能；预执行检查的实现假设可对所有工具调用做完备评估，实际部署时可能受限；最后，安全评估仍停留在实验级别，未能在真实恶意攻击情境中全面验证。

---

## 262. Reciprocity Can Erase Direction:tructural Boundaries and Controlled Utility of Magnetic Graph Networks

**arXiv ID:** 2611.04295 | [PDF](https://arxiv.org/pdf/2611.04295v1)

**作者:** Dawei Han `[一作]` (Xi'an Jiaotong University), Fumin Wang `[通讯]` (Xi'an Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究磁性图网络中，互惠关系可能导致学习到的边方向被消除，并提出了结构性诊断和对抗性实验来验证；

**💡 创新点**

提出了能量-互惠性关系的解析式、Top‑k 与节点数的边界条件，并设计了对称节点对实验以证明方向信息被抹除时模型只能达到随机水平；

**🔧 技术方法**

使用复数 Laplacian、支持型磁相位、连续相位以及实数有向聚合等技术，并对比了它们在合成任务与深度视网膜图像任务中的表现；

**📊 数据集**

在合成的八节点环和公共 DeepDRiD 视网膜图像数据集（1,984 张图像，296 训练，100 验证，100 测试患者）上进行实验；

**📈 对比分析**

通过 AUROC 和准确率衡量：在合成任务中，支持相位在单边 k=1 时取得 1.0000，k=2、7 时退回 0.5000；连续相位在所有 k 下均保持 1.0000；在 DeepDRiD 中，连续相位相较于零相位仅提升约 +0.0036 AUROC，差异不显著；

**⚠️ 局限性**

局限性包括：合成任务过于理想化、缺乏真实的方向监督；DeepDRiD 仅为回顾性图像数据，未提供独立外部验证，且结果仅显示极小且不显著的性能提升，不能证明磁相位在临床任务中具备实质优势。

---

## 263. Sketch-Omni: Instance-Level Sketch-Based Visual Prompting for Perception MLLMs

**arXiv ID:** 2611.04262 | [PDF](https://arxiv.org/pdf/2611.04262v1)

**作者:** Dongyu Wang `[一作]` (University of Surrey), Yi-Zhe Song `[通讯]` (University of Surrey)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发并训练了 Sketch-Omni，这是一款首个支持草图输入和草图作为查询的多模态大型语言模型，能够在草图上完成检测、指点、对象参照、空间参照、功能定位，并能将草图作为查询精准定位照片中的同类别实例。

**💡 创新点**

创新点包括：①双向草图接口——草图既可以是输入也可以是查询；②利用照片生成的边缘草图实现无标签监督；③通过照片引导的跨模态和同模态注意力关系对齐（photo-guided relation alignment）训练草图路径；④构造实例级草图-照片对，消除对海量手工标注的需求。

**🔧 技术方法**

技术手段包括：基于 Rex-Omni（Qwen2.5-VL-3B）的大模型；LoRA 适配器；照片引导的关系对齐（cross‑modal 与 intra‑modal attention 对齐）；生成式坐标标记；GRPO（Group Relative Policy Optimisation）细化；以及多阶段训练（Stage A + Stage B）。

**📊 数据集**

使用的数据集有：Objects365、RefCOCOg、AGD20K（用于草图原生任务）；Sketchy、FS‑COCO、SketchyCOCO（用于实例级草图提示）；HED 边缘草图生成的合成数据；通过上述数据构造的草图‑照片对；以及内部生成的多样化构造样本。

**📈 对比分析**

与 Qwen2.5‑VL‑3B、Qwen3‑VL‑8B、MiMo‑VL‑7B、DeepSeek‑VL2‑S、Rex‑Omni、GroundingDINO+ZSE‑SBIR 等多模态大模型及基准管线进行比较。Sketch‑Omni 在所有 6 个任务（检测、指点、对象参照、空间参照、功能定位、实例级草图提示）中取得 39/40 个单元格的最佳分数；在实例级草图提示上 F1@0.5 91.4 对比最强基线 81.8，提升 9.6；在 FS‑COCO‑Hard 场景中提升 17.2 F1；在检测和指点等任务上亦显著优于基线。

**⚠️ 局限性**

局限性包括：仍依赖于人工构造的草图‑照片对，未覆盖多类别多实例混合场景的跨类别匹配；对草图的绘制质量与复杂度敏感；未对多模态跨域推理（如多语言、跨域任务）做系统验证；模型训练需要大量预训练基础模型和算力。

---

## 264. ActOR: Ambient AI-Based Agentic Orchestration for Humanoid Autonomy in Operating Rooms

**arXiv ID:** 2611.04255 | [PDF](https://arxiv.org/pdf/2611.04255v1)

**作者:** Han Zhang `[一作]` (Johns Hopkins University), Mathias Unberath `[通讯]` (Johns Hopkins University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发并验证了基于环境感知的机器人编排框架ActOR，使仿生机器人能够在手术室执行多步骤物品收集与处置任务。

**💡 创新点**

创新点在于将墙壁安装的环境相机与机器人头部相机融合成共享数字孪生；使用LLM进行无任务特定训练的技能规划；闭环反馈实现失败重试和重新规划。

**🔧 技术方法**

采用视觉语言模型+SAM2+FastFoundationStereo进行目标检测与分割；LLM GPT-6.1 Sol作为规划器；基于Pinocchio和AnyGrasp的抓取规划；全身控制器+A*导航；在模拟与物理环境中集成这些技术。

**📊 数据集**

使用真实手术室布局重建的两套仿真环境和SAM3D生成的虚拟物体；物理实验使用Unitree G1仿生机器人与实际手术室摆放的物品。

**📈 对比分析**

与“仅机器人观测”和“无闭环反馈”两种消融基线对比；在模拟环境中任务成功率提升至83.3%（相较46.7%），在物理实验中单物体处置成功率70%；探索次数从1.05次/试验降至0.12次/试验。

**⚠️ 局限性**

局限性包括依赖云端专有模型导致隐私与可靠性问题；仿生平台对物体尺寸与抓取精度有限；抓取误差仍高，需要更鲁棒的学习策略。

---

## 265. COMPREHEND: Ultra-Fine-Grained Benchmark Understanding by Entailment Hierarchy

**arXiv ID:** 2611.04249 | [PDF](https://arxiv.org/pdf/2611.04249v1)

**作者:** Sivaraaman Balakrishnan `[一作]` (University of Massachusetts), Haw-Shiuan Chang `[通讯]` (University of Arizona)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过自底向上的方法构建基于蕴含关系的超细粒度提示层次结构，并用该层次结构生成更精确、更具信息量的类别描述，以帮助分析LLM在不同提示上的性能。

**💡 创新点**

①将类别描述改为完整句子/短段落并使用蕴含关系取代传统的词汇标签；②在每一层同时进行相似度匹配与共同提示生成，降低生成难度；③引入可选的DAG结构，解决单父节点限制；④提出基于蕴含与信息量的内在评估指标，用于无真值数据集的直接比较。

**🔧 技术方法**

使用Sentence‑BERT进行相似度计算，Blossom匹配算法构建配对；LLM（如ChatGPT/Claude）负责生成共同提示、提取主任务以及评估蕴含分数；基于这些生成的描述构建层次结构；评估时同样使用LLM判定蕴含与信息量。

**📊 数据集**

WildBench（包含约千条真实用户查询）与 UltraFeedback（LLM对齐数据集）各抽取256条样本进行实验。

**📈 对比分析**

与 EvalTree（现有最优提示树构建方法）进行对比。我们的模型在 Entail_avg、Entail_min、Info_avg、Info_min、Specificity 上均显著提升（如 Entail_avg 由 36.3 提升至 55.9），并在 LLM 生成报告的 RMSE 降低、Pearson 相关提高，说明层次结构更能帮助 LLM 复现原始性能得分。

**⚠️ 局限性**

①实验仅覆盖两类数据集，未验证在更大规模或多语种场景下的泛化；②对LLM生成的蕴含分数依赖于评估模型的准确性；③DAG 结构在可视化与解释性方面尚需进一步研究；④方法仍需高质量LLM支持，若LLM能力受限，生成质量可能下降。

---

## 266. Polyominoes Folding into Four and Five Distinct Boxes

**arXiv ID:** 2611.04246 | [PDF](https://arxiv.org/pdf/2611.04246v1)

**作者:** Erik D. Demaine `[一作]` (Massachusetts Institute of Technology), Ryuhei Uehara `[通讯]` (Japan Advanced Institute of Science and Technology)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究提出了新的多盒折叠多格子（polyomino）结构，证明存在面积为178的四盒折叠多格子和面积为358的五盒折叠多格子，并给出了无限多种可折叠为四盒或五盒的构造方法；

**💡 创新点**

创新点在于利用同一张纸条可以同时折叠成多种“双重覆盖矩形”，进而通过削角形成多种盒子，实现一次折叠可得到多种尺寸不相同且面积相等的立方体；此外首次用人工智能辅助发现并优化多格子折叠结构，将原先面积从532/2128大幅降低；

**🔧 技术方法**

主要技术包括：1) 对偶重叠矩形的折叠理论与周期轨道分析；2) 删角后得到盒子的构造公式；3) 通过可编程的“纸条循环”实现多种折叠；4) 人工智能模型 GPT-6 Astra Ultra 与 GPT-6 Luna 用于搜索和优化折叠结构；

**📊 数据集**

研究中未使用传统的公开数据集，而是利用 GitHub 上的自定义多格子图形和 AI 搜索得到的单元格集合作为实验数据；

**📈 对比分析**

与以往已知的面积分别为532（四盒）和2128（五盒）的例子相比，本文通过 AI 优化后面积分别降至178和358，已是目前已知的最小面积；实验表明所给构造可以生成无限多种可折叠为四盒或五盒的多格子；

**⚠️ 局限性**

局限性包括：1) 仅针对整数边长的盒子，未考虑非整数尺寸；2) 仅证明了四盒与五盒的可行性，六盒及更高仍未解决；3) 研究依赖 AI 辅助，缺乏纯数学解析证明；4) 对最小面积的下界仍有较大差距，未实现面积最小化的最优证明。

---

## 267. TCINet: Tissue-Conditioned Adaptive Cross-Branch Interaction Network for Nuclei Instance Segmentation

**arXiv ID:** 2611.04235 | [PDF](https://arxiv.org/pdf/2611.04235v1)

**作者:** Zhenghang Qian `[一作]` (Tsinghua University), Lianshun Jin `[通讯]` (Hengqin Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 TCINet，一种基于组织条件的单阶段核实例分割模型，能够同时完成核存在预测、水平-垂直距离预测和核类型分类。

**💡 创新点**

创新点包括：① 任务特定特征适配器 (TSFA) 在共享解码器跳跃连接处对特征进行分支专门化；② 组织条件交叉分支交互 (TC-TACBI) 根据预测的组织概率动态调节三条非对称交互路径；③ 在单一网络中实现三任务的协同学习，而无需外部提示或文本特征。

**🔧 技术方法**

核心技术：SAM‑H 视觉 Transformer 编码器；深度可分卷积 + 1×1 卷积 + BN + GELU 的 TSFA；两层 MLP 的组织条件控制器；LKCellBlock 预测头；多任务损失组合 (Focal Tversky、Dice、BCE、MSE、交叉熵)；数据增强与分割后处理。

**📊 数据集**

主要使用的数据集为 PanNuke（7,904 张 19 种组织类型的图像，5 种核类标签）以及 MoNuSeg（用于零射转移测试）。

**📈 对比分析**

在 PanNuke 上与 CellViT‑SAM‑H、LKCell‑L、PromptNucSeg‑H、IVAAN‑H、UN‑SAM 等方法对比，TCINet 获得 mPQ=0.519、bPQ=0.692，超过所有基线；在 MoNuSeg 上零射转移实现 F1=0.876，优于 CellViT‑SAM‑H 的 0.868；参数量仅增加 0.14%，FLOPs 降低 10.7%。

**⚠️ 局限性**

限制与不足：① 仍需在推理时预测组织类别，若组织预测失误会影响交互；② 对不同组织类别的性能差异未完全剖析；③ 仅在单阶段框架下验证，跨折稳定性及单路径贡献的深入 ablation 仍待进一步研究。

---

## 268. Constructing a Regional Slang Lexicon for Electricity Demands via Soft-Label Bidirectional Anchoring and Active Learning

**arXiv ID:** 2611.04231 | [PDF](https://arxiv.org/pdf/2611.04231v1)

**作者:** Rong Chang `[一作]` (Yuxi Power Supply Bureau, Yunnan Power Grid Co., Ltd.), Zhaoxiang Zang `[通讯]` (Kunming University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

基于弱监督的双向锚定与软标签策略，构建了云南玉溪地区电力客服微信群聊中的方言俚语词典，并通过主动学习验证与评估。

**💡 创新点**

创新点在于：①软标签双向锚定将方言标记与电力领域相关性分离，显著提升候选覆盖率；②结合 PMI 与句子‑BERT+词汇固定度双通道评分；③引入规则与 RAG‑LLM 两类 oracle 的双轨评估与校准；④通过 Cohen κ 对 LLM 评估进行人类校准。

**🔧 技术方法**

使用的技术包括：点互信息 (PMI)、软标签双向锚定、jieba 分词与词表注册、Sentence‑BERT 语义相似度、词汇固定度得分、Margin/TD‑CAL/Random 三种主动学习查询策略、规则与 RAG‑LLM oracle、DeepSeek V4 Flash/V4 Pro LLM 评估、Cohen κ 校准。

**📊 数据集**

实验数据集为 11,177 条玉溪市电力客服微信群聊的清洗后消息，覆盖约一年时间，包含 12,165 个词项与 48,395 个二元组。

**📈 对比分析**

与硬门控对照实验相比，软标签策略将候选词数提升 751%（从 41 变 349），最终词典规模 25–33 条；所有配置均达 DP=1.0，PDP 在软标签模式下为 0.12–0.16（硬门控 0.625）。不同 oracle 与查询策略对词典规模影响有限，主要差异体现在词典覆盖率与 PDP。

**⚠️ 局限性**

局限性包括：仅针对单一方言与电力领域，缺乏召回度评估；LLM 评估与 oracle 同属 DeepSeek 系列，未做跨模型验证；主动学习样本量小（39 条），统计显著性不足；词典规模有限，未在 downstream 任务（如投诉分类、优先级识别）中验证实际价值。

---

## 269. OSGR: Observation-Support Graph Reasoning for Earth Observation Question Answering

**arXiv ID:** 2611.04301 | [PDF](https://arxiv.org/pdf/2611.04301v1)

**作者:** Dazhou Yu `[一作]` (Emory University), Liang Zhao `[通讯]` (Emory University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 OSGR 框架，通过离线的 Observation‑Support Graph 进行科学推理，并在在线阶段使用 Relation‑Conditioned Grounding 对 EO 数据进行时空匹配，实现基于图约束的 Earth‑Observation 问答。

**💡 创新点**

将检索与推理分解为两阶段，先在符号概念层生成支持路径，再在物理时空层做精确匹配；构建 ISO/OGC OMS 对齐的三层支持图；提出基于关系的时空兼容核。

**🔧 技术方法**

离线图构建、ISO/OGC OMS 标准、三层目标-属性-结果支持图、关系条件时空兼容核、LLM（GPT‑4o、GPT‑5 等）作为生成器；基于 RAG、ReAct 等 baseline 对比。

**📊 数据集**

Earth‑Bench（248 个问题，约13.7K EO 图像）和 DisasterM3‑Reason（7160 条多选实例，BDC 与 DRE 两任务）。

**📈 对比分析**

与 Direct LLM、ReAct Agent、GeoGraphRAG、Earth‑Agent 等基线进行对比；在 Earth‑Bench 上 GPT‑4o 版 OSGR 准确率从 49.3% 提升至 63.7%，SFA/TFA 与 RGA 也显著提高；在 DisasterM3‑Reason 上平均准确率从 46.2% 提升至 50.1%；整体效率改善（执行步比降至 2.5）。

**⚠️ 局限性**

依赖支持图与档案元数据的完整性；缺失或错误的关系、地理足迹或时间戳会导致无法检索；仅在 EO 领域验证，跨领域迁移需重新定义图结构与兼容函数。

---

## 270. The Native Menace: Host-to-Browser Cross-Boundary Attacks via Native Messaging

**arXiv ID:** 2611.04291 | [PDF](https://arxiv.org/pdf/2611.04291v1)

**作者:** Jyotirmay Chauhan `[一作]` (University of Illinois Chicago), Jason Polakis `[通讯]` (University of Illinois Chicago)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统评估了浏览器的Native Messaging功能，发现并利用其设计缺陷，构造了双向MITM和单向仿真攻击，证明可在不提升权限的情况下窃取密码、篡改电子签名等敏感操作。

**💡 创新点**

创新点在于首次对Native Messaging进行实证安全分析，揭露manifest阴影、缺乏客户端验证、无消息完整性等漏洞，提出基于manifest限制、父进程验证和端到端加密的多维缓解方案，并通过多种真实应用演示攻击效果。

**🔧 技术方法**

采用协议分析、恶意manifest植入、头less Chrome模拟、持续文件监控、管道重定向等技术实现攻击链；同时使用TLS/PAKE等加密方式验证缓解手段。

**📊 数据集**

利用Palant Chrome扩展清单数据集（2026-08-14快照），共250,735个扩展中筛选出1,972个声明native_messaging的扩展，累计约1.07 亿次安装，进一步挑选17个高影响目标（如1Password、Bitwarden等）进行实验。

**📈 对比分析**

与传统的无Native Messaging攻击对比，实验显示在macOS和Windows上该攻击能够突破OS隔离，获取完整密码库、签署合法文件等；在500+扩展中，攻击覆盖率高于普通攻击，且在不需要用户交互的情况下完成。

**⚠️ 局限性**

局限性包括主要聚焦Chrome/Firefox，需浏览器端修改以阻止manifest阴影；攻击依赖用户已安装相应扩展或宿主；实现难度受操作系统特定隔离机制影响，且在受限制的沙箱环境下可能无效。

---

## 271. When Debate Amplifies Risk: Social-Bias Vulnerabilities in Multi-Agent Safety Deliberation

**arXiv ID:** 2611.04289 | [PDF](https://arxiv.org/pdf/2611.04289v1)

**作者:** Qinghua Mao `[一作]` (Shanghai Jiao Tong University), Yuliang Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多智能体辩论在Web检索情境下的安全性，揭示社交动态会放大隐藏的危险信息并导致错误决策。

**💡 创新点**

提出了SafeConflict‑Bias诊断协议和RiskTrace‑Debias防护框架，后者通过先验证据预设、社交元数据屏蔽和一致性门控，显著削弱社交偏见。

**🔧 技术方法**

采用多智能体辩论架构、对话元数据匿名化、先验预commitment与一致性门控等技术，并利用大模型推理实现。

**📊 数据集**

使用基于真实Web语料的100个安全冲突案例，构成700个对照实例，覆盖六类风险（安全与不安全冲突、过时危险、模糊风险等）。

**📈 对比分析**

与传统单一模型、Self‑Reflection以及原始多智能体辩论对比，RiskTrace‑Debias将不安全采纳率从最高100%降至≤2%，同时保持0%解析错误与过度拒绝。

**⚠️ 局限性**

局限在于只评估固定规模（5名代理）和单一文本检索，未覆盖动态通信拓扑、跨模态检索或更复杂的对抗策略。

---

## 272. TTSonic: A Unified Framework for Composable and Controllable Speech Generation

**arXiv ID:** 2611.04282 | [PDF](https://arxiv.org/pdf/2611.04282v1)

**作者:** Zhedong Zhang `[一作]` (ByteDance Douyin Content Group), Xu Chen `[通讯]` (ByteDance Douyin Content Group)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个统一可组合的语音生成框架TTSonic，结合语音分词器、文本到语义的自回归语言模型和声学流匹配解码器，实现从零样本TTS到语音克隆、韵律克隆、可控克隆、语音转换以及噪声参考克隆等多种任务；

**💡 创新点**

通过两阶段学习的语义‑声学分离的语音分词器，使同一离散表示既支持语言模型的语义与韵律规划，又可供声学渲染；引入CoT‑guided 计划与Text2Timbre模块，提升指令遵循与音色多样性，整体实现任务无缝组合而非专用模型；

**🔧 技术方法**

使用E‑Branchformer encoder、有限标量量化（FSQ）分词、语音对齐的流匹配（flow‑matching）解码、HiFi‑GAN/HiFT vocoder、Perceiver采样、CosyVoice 3、Qwen3.5 语言模型、CAM++说话人编码、Text2Timbre的条件扩散等技术；

**📊 数据集**

在大规模多说话人语音数据上预训练分词器，随后在Seed‑TTS‑Eval、CV3‑Eval、InstructTTSEval、Seed VC、VoiceBank‑DEMAND‑16k等基准数据集上进行评估；

**📈 对比分析**

通过与多种专用与统一的TTS/VC/指令系统在同一基准下对比，采用WER/CER、SIM、DSDM/RS/UTMOS等指标；在零样本中文/英文/难度子集表现为CER<1%/WER<2%、SIM>0.78，跨语言性能优于多数对手；指令遵循准确率达88% APS、82% DSD、69% RP；语音转换与噪声克隆在SIM、UTMOS上均领先；

**⚠️ 局限性**

跨语言指令遵循仍落后于中文，Stage‑II分词器在内容准确性略逊于Stage‑I，模型规模和训练/推理成本较高，对极端噪声环境下的鲁棒性尚未彻底验证。

---

## 273. ReuseGR: Offline--Online Decoupled Generative Retrieval via Adaptive Prefix Reuse

**arXiv ID:** 2611.04259 | [PDF](https://arxiv.org/pdf/2611.04259v1)

**作者:** Xujing Wang `[一作]` (Xidian University), Ziyu Guan `[通讯]` (Xidian University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种离线–在线分离的生成检索框架 ReuseGR，利用大模型（7B）在离线阶段为每个查询生成可共享的 SID 前缀集合，并在在线阶段使用较小模型（0.5B）结合用户历史完成前缀后续，采用自适应深度选择、前缀重用、路径知识蒸馏以及分支采样策略以实现高检索质量与低延时的平衡。

**💡 创新点**

创新点：
- 通过查询级前缀共享（ReuseCut）实现大模型规划的复用，显著降低在线推理成本。
- Path KD 将大模型在前缀层的概率分布迁移至小模型，实现高质量后缀评分。
- BudgetCut 采用基于回忆率的分支选取策略（OPD+GRPO），在固定分支预算下进一步提升最终 Top‑100 Recall。
- 结合离线前缀缓存与在线历史条件化的完整生成流程，在工业搜索广告场景下实现 38.9% 的平均延时下降，并在 14 天线上 A/B 实验中提升 CTR 与 RPM。

**🔧 技术方法**

核心技术：
- 基于 Semantic ID (SID) 的生成检索。
- 大模型（Qwen2.5‑7B）离线前缀规划。
- 小模型（Qwen2.5‑0.5B）在线后缀完成。
- ReuseCut（深度预测）、Path KD（路径级知识蒸馏）、BudgetCut（分支采样）等算法。
- 对齐、KV 缓存、批量分支评分、分支剪枝。
- 采用 Softmax、KL 损失、PPO‑style 训练。

**📊 数据集**

数据集：工业搜索广告数据 AE‑PV，包含 3.38M 条商品，4 层 SID 码，约 44K 个查询、约 1M 条请求记录；使用 Fit/Dev/Target 切分以及 14 天线上流量。

**📈 对比分析**

对比基线：
- 全在线 0.5B、7B 生成检索；
- 1B 端到端、SmartGR、OneRec‑Think、DSSM 等。
- 性能：ReuseGR 在离线前缀缓存可用时 HR@100 85.82%（比 0.5B 提升 3.83pp），平均延时 71.45 ms（比 0.5B 低 38.9%），与 7B 端到端相近；在线 A/B 测试 CTR +0.93%，RPM +1.05%。

**⚠️ 局限性**

限制：
- 需要离线大模型规划与定期缓存刷新，增加维护成本。
- 对查询历史长度和质量敏感；冷启动（无缓存）仍需全模型推理。
- 仅在已缓存深度的查询中获益，对深度预测误差或数据分布漂移可能影响性能。
- 计算资源集中在 7B 规划节点，规模化部署需额外 GPU。
- 部分实验仅在单 GPU 单线程环境评估，实际多线程/多机情况需进一步验证。

---

## 274. ProGeo-RC: Prototype-guided Semantic Reliability and Geometric Error Calibration for Semi-supervised Cardiac Image Segmentation

**arXiv ID:** 2611.04257 | [PDF](https://arxiv.org/pdf/2611.04257v1)

**作者:** Yunfei Zhu `[一作]` (Chongqing Normal University), Jianming Hu `[通讯]` (Chongqing Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种半监督心脏图像分割框架ProGeo-RC，结合原型引导的语义可靠性估计与几何误差校准，提升伪标签质量与分割精度。

**💡 创新点**

创新点在于同时在特征空间使用概率原型评估语义可靠性，并在像素空间利用签名距离场（SDF）进行几何校准，使伪标签在语义与几何两方面都更可靠，从而显著降低误差。

**🔧 技术方法**

采用双学生网络的教师选择策略、概率原型记忆更新、SDF约束的边界与外部误差校准、可学习的损失平衡权重等技术。

**📊 数据集**

使用ACDC心脏MRI数据集和MVAA-CT斑瓣CT数据集，在10%/20%标注和2/4标注等低标注设置下进行实验。

**📈 对比分析**

与CPS、MC-Net、BCP、DiffRect、AdaMix等代表性半监督方法对比，ProGeo-RC在ACDC 10%标注下DSC 89.90%、95HD 1.54、ASD 0.57等指标均优于第二名；在MVAA-CT 2/4标注下同样排名第一，误差更小、标准差更低。

**⚠️ 局限性**

局限性包括：仅在训练阶段增加计算量，推理时无额外开销；未在极端少标注（如1%）场景下验证；对其他器官或多模态数据的泛化仍待进一步研究。

---

## 275. CounterSketch: Physics-Grounded Counterfactual Reasoning for Sketch-to-Video Generation

**arXiv ID:** 2611.04244 | [PDF](https://arxiv.org/pdf/2611.04244v1)

**作者:** Hien Trinh `[一作]` (University of Science), Trung-Nghia Le `[通讯]` (University of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CounterSketch，一个将“如果……?” 的自然语言反事实提示与手绘草图相结合，通过物理驱动的事件链推理生成可视化动画的框架。

**💡 创新点**

创新点在于将干预解析为因果事件链，并基于物理约束编译、验证轨迹，为动画提供可解释、可控的运动指导，区别于直接由 VLM 预测轨迹。

**🔧 技术方法**

采用视觉语言模型构建草图场景图、事件链推理、约束编译与基于质量矩阵的多约束运动模拟、轨迹验证闭环修复，以及将轨迹投影到图像空间指导 MoSketch 的技术。

**📊 数据集**

使用自建的 CounterSketchBench 数据集，包含 20 个基础草图场景下的 80 条“如果……?” 反事实问题，作为评估基准。

**📈 对比分析**

通过与直接 VLM 预测轨迹的三种配置进行 ablation，对比 VBench 的文本–视频一致性、草图保留、时间连贯性和运动强度四项指标，结果显示 CounterSketch 在文本–视频一致性和运动强度上提升约 3%–8%，同时保持时间连贯性。

**⚠️ 局限性**

局限性包括依赖准确的草图场景图定位、难以从草图中推断质量、摩擦等隐式物理属性、复杂交互导致推理成本高，以及对视频生成器的条件能力仍有限。

---

## 276. Improving Diffusion-Based Augmentation for Data-Scarce Image Classification via Mask-Guided Reference Attention and Repulsive Noise Optimization

**arXiv ID:** 2611.04242 | [PDF](https://arxiv.org/pdf/2611.04242v1)

**作者:** Hyun-Jic Oh `[一作]` (Korea University), Won-Ki Jeong `[通讯]` (Korea University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在极度数据稀缺的图像分类任务中，本文提出一种采样时的扩散数据增强框架，通过Mask‑Guided Reference Attention和Repulsive Noise Optimization两种模块提升合成样本的细节与多样性。

**💡 创新点**

创新点在于将前景参考特征限定在掩模区域进行值更新（MGR Attention），以及利用适配提示锚点进行仅排斥的噪声优化（RNO），从而同时解决细节退化与多样性坍塌。

**🔧 技术方法**

采用Stable Diffusion v1.5/v2.1的UNet扩散模型，结合零射门分割器SAM3实现前景掩模，并在Latent Diffusion过程进行噪声优化。

**📊 数据集**

在CUB‑LT、Flower‑LT、CUB、Aircraft、Cars、Pet、Waterbirds等长尾、少样本及背景迁移基准上进行实验。

**📈 对比分析**

与Mixup、CutMix等传统混合增强以及Diff‑Mix、Diff‑II、Real‑Gen等最新扩散增强方法对比，本文在CUB‑LT IF100单样本组最高提升+2.94点、少样本组+1.69点等显著改进。

**⚠️ 局限性**

局限在于RNO的排斥强度可能导致样本偏离类一致性，并且仅在UNet基线上验证，未扩展至Transformer架构。

---

## 277. SEAM: Situated Executable Associative Memory for a Co-located Community

**arXiv ID:** 2611.04296 | [PDF](https://arxiv.org/pdf/2611.04296v1)

**作者:** Keru Wang `[一作]` (New York University), Andrzej Banburski-Fahey `[通讯]` (Microsoft Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了SEAM——一种面向共处社区的分布式共享记忆系统，能够将成员发出的任意瞬间进行关联、提炼知识并执行可编程指令；

**💡 创新点**

创新点在于将单人记忆的重建、关联与执行机制迁移到社区层面，结合认知科学的设计原则，形成可动态学习、可编程且主动召回的共享记忆；

**🔧 技术方法**

使用GPT‑5‑mini与GPT‑5.4完成检索规划、反射与生成；通过Azure OpenAI进行文本与视觉嵌入、CLIP、Whisper；数据存储在Azure Cosmos DB与Blob；实现了知识Wiki、合并器、可执行规则层与合并代理；

**📊 数据集**

实验数据来自46名共处社区成员两周的现场部署，共计345条文本/多模态贴文，并生成246页知识Wiki；

**📈 对比分析**

未与传统搜索/问答系统做严格量化对比，而是通过用户打分（23/27正向）及定性分析评估回复的关联性、发现度与可执行性；性能表现为知识增长遵循Heaps律，指数0.75，说明随着贴文增加知识量增速逐渐趋缓；

**⚠️ 局限性**

局限性包括：召回策略可能遗漏信息、对多模态的支持不足、对高产贴文者的偏倚、缺乏细粒度权限与隐私控制、未对系统输出的正确性做严格验证。

---

## 278. From Guidelines to Practice: Defining Requirements for AI Red-Teaming Tools

**arXiv ID:** 2611.04271 | [PDF](https://arxiv.org/pdf/2611.04271v1)

**作者:** Gia Phong Tran `[一作]` (ST Engineering), Vrizlynn L. L. Thing `[通讯]` (ST Engineering)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过梳理多项 AI 安全指南，提出八大 AI 红队工具可取特性，并在六款开源 LLM 红队工具上进行实验验证；

**💡 创新点**

创新点在于将行业标准与工具功能直接映射，形成结构化特性框架，并利用 NIST AI 100‑2 参考框架统一评估工具效果；

**🔧 技术方法**

主要技术包括基于开源框架（AgentFence、Garak、Giskard、Promptfoo、Prompt Fuzzer、Whistleblower）的红队攻击实现、自动化测试与报告生成；

**📊 数据集**

使用了六款来自 Ollama 的开源大模型（Llama 4 Scout、Llama 3.2、Llama 2、Vicuna、DeepSeek、Mistral）作为实验对象；

**📈 对比分析**

比较方法：对八大特性进行评分，并将各工具攻击类别映射到 NIST AI 100‑2 术语，使用成功率、覆盖度等指标；实验结果显示 Promptfoo 与 Garak 在攻击覆盖与自动化方面表现突出；

**⚠️ 局限性**

局限性包括：缺乏统一攻击分类标准导致比较不完全客观；实验仅覆盖部分 LLM，未考虑多模态模型；工具更新频率和社区活跃度差异导致性能差距；缺乏对防御机制效果的评估。

---

## 279. SteelHuman: Predicting Operator Estimates of Steel-Scrap Composition from Sparse RGB Views

**arXiv ID:** 2611.04317 | [PDF](https://arxiv.org/pdf/2611.04317v1)

**作者:** Jovan Kamchev `[一作]`, Zoran Ivanovski `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了使用稀疏手持相机拍摄的钢铁废料堆的RGB图像，通过冻结的DINOv2特征与均值聚合来预测工人操作员的质量成分估计。

**💡 创新点**

提出了一种低成本的稀疏视图聚合方案，省去多阶段预处理与分割，且对错误结构进行细粒度分析，证明模型能改善厚度分配的准确性。

**🔧 技术方法**

使用冻结的DINOv2 ViT‑L/14特征、均值池化或ABMIL/自注意力聚合，采用KL散度训练，评估TV、RMSE等指标，并在公共DOES数据集上做迁移测试。

**📊 数据集**

使用自家工厂收集的750堆钢铁废料图像（223均质、527异质），每堆平均5张视图；以及公共DOES分类数据集做迁移评估。

**📈 对比分析**

与仅视觉先验、LASER及公开分类基准对比；在本数据上均值聚合TV为13.4%（总体），在DOES上分类精度达93.97%，HMS子类RMSE为11.8%接近LASER；多视图显著降低误差。

**⚠️ 局限性**

受限于缺乏测量真值，难以区分模型误差与操作员估计的不确定性；对细薄HMS类别的预测仍差，缺乏同质样本；稀疏视图无法完全覆盖堆物，可能遗漏重要信息。

---

## 280. FlowTranslator: Training-Free In-Image Machine Translation with Flow-Based Editing

**arXiv ID:** 2611.04303 | [PDF](https://arxiv.org/pdf/2611.04303v1)

**作者:** Zhixian Li `[一作]` (Fudan University), Zhineng Chen `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 FlowTranslator，一个完全训练无关的图像内机器翻译系统，通过冻结的文本到图像流模型进行场景文本渲染。

**💡 创新点**

提出了针对文本极小区块的两种轨迹控制（Region Coupling Release 和 Coupling Expansion），解决了流编辑的点火失败，并实现了跨语言渲染。

**🔧 技术方法**

采用 OCR+LLM 翻译、VAE 编码、冻结的流匹配 T2I 模型（Z-Image-Turbo）以及 FlowEdit 的 ODE 编辑。

**📊 数据集**

在 TTV（Synthetic 512² 文本图像）、PRIM（真实 512×46 文本条幅）和 ScenePair（1280 张实景照片）等基准上进行评估。

**📈 对比分析**

与多种训练和无训练 IIMT 系统对比，FlowTranslator 在 PRIM 上获得最高 BLEU/COMET，在 TTV 上获得最高平均 BLEU，并在 ScenePair 的中文全图翻译中达到 40.1% 的 exact‑match 精度，超过六种已发布编辑器。

**⚠️ 局限性**

受限于冻结模型的脚本覆盖范围，无法渲染如西里尔字母等未支持脚本；此外，对小型文本区域的点火仍依赖特定轨迹控制。

---

## 281. CLOCs-R: Radar-Assisted Camera-LiDAR Fusion for Long-Range Object Detection

**arXiv ID:** 2611.04300 | [PDF](https://arxiv.org/pdf/2611.04300v1)

**作者:** Richard Beuck `[一作]` (Florida Gulf Coast University), Sudip Dhakal `[通讯]` (Florida Gulf Coast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于候选级融合的雷达-相机-激光雷达三传感器融合框架 CLOCs-R，利用雷达返回信息对已生成的 3D 目标候选框进行重新评分，以提升远距离目标检测性能。

**💡 创新点**

创新点在于：①将雷达返回的多维特征（支持度、Doppler 速度、速度离散度、距离残差、最近返回偏移）融入 CLOCs 的候选编码；②在保留已有 2D/3D 检测器冻结的前提下，仅训练轻量级融合网络，实现了低延迟且易于部署；③在 aiMotive 长距离数据集上系统评估，首次验证雷达对 30–50 m 以及 100–200 m 远程检测的实际帮助。

**🔧 技术方法**

技术手段包括：①使用 SECOND 3D 激光雷达检测器和 Cascade R‑CNN 2D 相机检测器（均冻结）产生候选框；②通过雷达返回门控（gate）与 3D 候选框关联；③构造雷达特征向量并将其追加到候选级别的 1×1 卷积网络中；④采用焦点损失和 Adam 优化器进行微调；⑤在融合后使用残差评分（residual scoring）来保留原始 3D 分数的排序。

**📊 数据集**

主要数据集为 aiMotive：包含同步相机、64 层激光雷达和长距离雷达，标注覆盖约 200 m，日间清晰天气场景。验证时使用其官方 151 训练场景和 25 验证场景，实验对每帧缓存候选框，训练时不依赖检测器成本。

**📈 对比分析**

与基准（仅激光雷达 SECOND、CLOCs 无雷达、aiMotive 发布的 LiDAR/雷达/相机融合）进行对比。CLOCs-R 在 30–50 m 区间的 APBEV 提升约 1–1.5 %；在 >50 m 区间仅有 0.3–0.4 % 的轻微提升；在 100–200 m 远程区间的 APBEV 与 SECOND 基线基本持平，残差评分使得整体 AP 与 SECOND 统计学无显著差异。雷达特征的加入对整体性能提升有限，但在远距离候选数量较少的场景中仍可提供正向帮助。

**⚠️ 局限性**

局限性包括：①雷达对极远距离（>100 m）的贡献受样本稀疏和噪声限制，提升幅度很小；②融合网络仅对已生成的 3D 候选框进行重评分，无法补偿 3D 检测器遗漏的目标；③实验仅在日间清晰天气的子集上进行，未验证夜间或恶劣天气的鲁棒性；④雷达特征选择和门控参数是在保留的训练/验证子集上调优，可能导致过拟合；⑤整体性能受限于已有的 SECOND 和 Cascade R‑CNN 基线，若改进基础检测器可能带来更大收益。

---

## 282. GroundSeek-CXR: Retrospective Evaluation-Guided Active Observation Framework for Multi-Label Chest Radiography Diagnosis

**arXiv ID:** 2611.04286 | [PDF](https://arxiv.org/pdf/2611.04286v1)

**作者:** Yihang Fu `[一作]`, Zhen Chen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GroundSeek-CXR，一种通过对最终诊断进行回溯评估来学习观察和停止决策的主动胸片诊断框架。

**💡 创新点**

创新点在于引入Retrospective Diagnostic Evaluation (RDE)，利用冻结的教师模型与离线MCTS独立续航来评估每个观察和停止候选动作的最终诊断价值，并通过Retrospective Gap-weighted Preference Optimization (RGPO) 用回溯分数差异加权的偏好学习，使模型的决策更贴合最终诊断而非即时效果。

**🔧 技术方法**

采用视觉语言模型（学生Qwen3.5-4B，教师Qwen3.5-122B-A10B）、离线Monte Carlo树搜索、LoRA微调、直接偏好优化（DPO）以及回溯差值加权技术。

**📊 数据集**

在VinDr-CXR数据集上进行训练与评估，使用CheXpert数据集进行外部验证。

**📈 对比分析**

与多种基准VLM和即时偏好学习方法比较，GroundSeek-CXR在VinDr-CXR宏/微F1、精度、召回率上均优于现有方法，异常图像样本F1提升4.48%（相较于即时DPO），并在CheXpert上展示出良好的泛化性能。

**⚠️ 局限性**

局限性包括依赖离线MCTS和冻结教师导致计算成本较高、观察预算固定可能限制对复杂共病模式的全面捕捉、实验仅覆盖两个数据集，未对其他影像模态或更大规模数据集的泛化性做深入验证。

---

## 283. OmniScribble: Mixed-Initiative Authoring of Animated Sketches through Inspectable and Repairable AI Decisions

**arXiv ID:** 2611.04284 | [PDF](https://arxiv.org/pdf/2611.04284v1)

**作者:** Hien Trinh `[一作]` (University of Science), Trung-Nghia Le `[通讯]` (University of Science)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 OmniScribble，一种混合主动的多对象草图动画系统，允许作者在对象分解和轨迹规划等中间决策上手动操作或接受 AI 提议，并在检查点进行接受、修复或重新生成。

**💡 创新点**

创新点在于将关键的 AI 生成步骤拆分为可视化、可编辑的中间产物，并通过检查点实现可检查、可修复、可回退的 AI 委托，从而在保持创作控制感的同时显著降低手动劳动。

**🔧 技术方法**

技术手段包括：基于视觉语言模型的对象分解与轨迹生成、稀疏向量图像修复、基于扩散的多对象动画合成、以及支持直接编辑和自然语言驱动的重生成的交互接口。

**📊 数据集**

使用的数据集主要为 256×256 的多对象矢量草图与相应动作说明，样本来源于公开的矢量草图数据集以及作者自制的实验样本。

**📈 对比分析**

通过在 8 名受试者上的受控对内实验，比较 Manual 与 Auto 两种模式，结果显示 AI 辅助下的完成时间平均减少 71.4%，NASA‑TLX 工作负荷降低至 5.17（对比 Manual 的 15.08），且意图匹配与满意度均维持在 6–7 分的高水平，说明可检查式委托既节省时间又不损失创作质量。

**⚠️ 局限性**

局限性包括：实验样本量小、任务受限于预设动词提示、对复杂或极简草图的模型鲁棒性不足、轨迹细化粒度有限、且系统对生成延迟与交互成本的适配尚待改进。

---

## 284. Metric TSP in Sublinear Time: Breaking Two Approximation

**arXiv ID:** 2611.04283 | [PDF](https://arxiv.org/pdf/2611.04283v1)

**作者:** Amir Azarmehr `[一作]` (Northeastern University), Soheil Behnezhad `[通讯]` (Northeastern University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

在子线性查询模型下，为度量TSP设计了一种算法，能够用 O(n²-ε) 次距离查询在常数 ϵ, δ>0 的情况下得到 (2-δ) 近似的TSP价值估计。

**💡 创新点**

突破了自 Czumaj–Sohler 以來的 2 近似界限，首次证明在子线性查询数量下即可取得严格优于 2 的近似；提出了“MST 缩短”和“核心–细胞分解”两大新框架。

**🔧 技术方法**

核心技术包括：
- 通过 Cut LP 与其对偶的斯坦纳树表示法，将 TSP 约束转化为对被支配树的优化；
- 引入 MST 缩短（dominated shortening）来构造接近最优的对偶解；
- 通过构造“核心” Steiner 树、映射 q 以及随机细胞分解，实现对全局结构的局部估计；
- 使用重要性采样、冲突图与最小分数顶点覆盖来在合同细胞中有效估计权重；
- 结合贝塞林（Bernstein）不等式与随机贪心匹配估计，保持整体查询复杂度子二次。

**📊 数据集**

本工作为理论研究，未使用具体实验数据集；所有结果均为严谨的数学证明与上界分析。

**📈 对比分析**

在理论上，相较于以往仅能得到 2 近似的子线性算法，本方法在 O(n²-ε) 查询下实现 (2-δ) 近似；算法时间同样为 O(n²-γ)。实验与实现细节未给出，但通过严密的概率分析保证至少 2/3 的成功概率。

**⚠️ 局限性**

限制与挑战包括：
- 常数 ϵ、δ 的具体取值未给出，实际效果受其影响；
- 算法结构复杂，难以直接实现；
- 仍需 O(n²) 规模的距离查询，不能达到线性查询；
- 结果仅适用于度量TSP，非度量或非欧几里得空间可能不适用。

---

## 285. Deep Learning Compilers: Architecture, Optimization, and Empirical Characterization

**arXiv ID:** 2611.04273 | [PDF](https://arxiv.org/pdf/2611.04273v1)

**作者:** Jinghao Wen `[一作]` (Villanova University), Xun Jiao `[通讯]` (Villanova University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性分析深度学习编译器的架构、优化技术，并通过PyTorch/TorchInductor实验验证编译器在不同工作负载下的性能特征。

**💡 创新点**

创新点在于把中间端的作用显式化，构建前端–中间端–后端完整的设计空间，并结合实验量化图优化、编译开销与动态形状处理的实际收益，提出了对比框架与编译器优化效果的实验范式。

**🔧 技术方法**

使用深度学习编译器框架（TVM、XLA、TorchInductor等）、IR层、图融合、自动调度、动态形状编译、量化、内存重用、循环优化等技术；实验环境为NVIDIA RTX A6000 GPU、PyTorch 2.3、CUDA 12.1、cuDNN 8.9、Triton 2.3。

**📊 数据集**

使用5个合成推理工作负载（元素链、MLP残差、卷积块、归一化残差、注意力块）作为测试集，未使用真实数据集。

**📈 对比分析**

方法：将PyTorch eager执行与TorchInductor编译后执行对比，测量中位延迟、首调用编译成本、增量峰值内存及最大绝对误差。结果显示：融合友好图最高可达8.21×加速，卷积块仅1.12×；首调用成本需要数千次执行才能收支平衡；动态形状采用符号编译将转换成本降低约70%，但稳态性能略逊。

**⚠️ 局限性**

局限性：实验仅在单一GPU、单一框架版本、单精度推理环境下进行，使用合成图不涉及训练、端到端模型、多设备、多硬件平台、能耗或编译缓存等因素，结果不具普适性。

---

## 286. Weight-Space Programming for Improving Image-Restoration Diffusion Models: Technical Conjectures and a Recursive Improvement Loop

**arXiv ID:** 2611.04268 | [PDF](https://arxiv.org/pdf/2611.04268v1)

**作者:** Zhuoran Zheng `[一作]`, Doubao `[通讯]` (Nanjing University Of Science And Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

探讨并提出使用大型语言模型对扩散式图像恢复模型进行权重空间编程，以无梯度、无配对数据、无微调的方式改进图像恢复质量。

**💡 创新点**

提出了在精确回归场景下的三条技术假设（层级诊断、稀疏权重编辑、递归改进循环），并论证所需 AI 能力不必达到超级智能。

**🔧 技术方法**

利用 LLM 读取压缩的权重统计信息，进行层级诊断与结构化权重变更；设计了 PyTorch 风格的 WSP 沙盒与递归改进循环；采用 PSNR、LPIPS 等指标评估。

**📊 数据集**

在论文中假设使用标准图像恢复基准（如 DIV2K、RealESR/RealSR、Set5/Set14 等）作为反馈集，未给出具体数据集，但提到 32–64 图像对作为评估。

**📈 对比分析**

将 WSP 的编辑效果与传统 LoRA 或全量微调对比，预期在 0.2–0.5 dB PSNR 提升，递归循环可累计提升 0.5–1.5 dB；实验尚为假设，未给出实测数值。

**⚠️ 局限性**

WSP 只能修复集中式缺陷，无法解决分布式问题；对权重的解释受通道置换对称性影响；评估受采样噪声和指标相关性限制；需要大规模实验验证。

---

## 287. OmniST: Topology-Invariant Spatiotemporal Forecasting with Guaranteed-Coverage Uncertainty at Unseen Sensor Locations

**arXiv ID:** 2611.04256 | [PDF](https://arxiv.org/pdf/2611.04256v1)

**作者:** Praveg Vashishtha `[一作]` `[通讯]` (Indian Institute of Technology Patna), Praveg Vashishtha (Indian Institute of Technology Patna)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了 OmniST，一种拓扑不变的时空预测网络，可在任意节点数下工作，并通过自适应 conformal 机制为未见传感器位置提供分布无关、有限样本覆盖保证的预测区间。

**💡 创新点**

创新点：1）通过坐标基记号化 + k‑NN 注意力 + per‑node GRU 构建完全不依赖全局节点数的架构；2）证明该架构下的自适应 split conformal 在节点增删时仍保持覆盖保证；3）引入滚动窗口 + Adaptive Conformal Inference 解决时间漂移导致的覆盖下降问题。

**🔧 技术方法**

技术细节：坐标 MLP tokenization、k‑Nearest‑Neighbor attention、双向 GRU 时序编码、split conformal prediction、滚动窗口和 Adaptive Conformal Inference、Mondrian 条件 conformal、可扩展的 cKDTree k‑NN 近似。

**📊 数据集**

数据集：
- 2019‑2023 Delhi CPCB PM2.5（37站）
- 2013‑2017 Beijing Multi‑Site Air‑Quality（12站）
- 2018‑2022 NOAA GHCN TMAX（30站）
- 228‑站 PeMSD7‑M 5‑min 交通速度（经纬度通过 MDS 重建）
- 325‑站 PeMS‑BAY 交通速度（全规模测试）

**📈 对比分析**

比较方法：与公开的 E‑STGCN 在同一 Delhi 数据集做对照，MAE 31.0 μg/m³（对比 39.2 μg/m³）保持相当；在未见站点的点预测中 OmniST 26.7 μg/m³，优于最近邻 30.3 μg/m³ 和 IGNNK 34.0 μg/m³；覆盖率在 90% 目标下达到 91.5%（相比最近邻 92.5% 但区间更窄 128.5 vs 149.7）；在节点删除 30–50% 时仍维持覆盖；时间漂移问题通过滚动‑ACI 恢复覆盖率至 0.89/0.96（静态 conformal 仅 0.76/0.50）。固定拓扑 STGCN 在未见站点无法产生任何预测。

**⚠️ 局限性**

局限性：
1) 规模验证仅到 325 站，千级节点仍需更大数据集与近似 k‑NN 评估；
2) 只提供边际覆盖，未实现每站/每步条件覆盖；
3) 使用 GRU 作为时序骨干，未探索更高效的 S4/Mamba 等模型；
4) 准确性评估集中在单城市 PM2.5，其他污染物/场景待进一步验证；
5) 仅用 3‑5 个种子，需更多种子与多指标评估；
6) 交叉域预训练与更广泛的物理知识集成仍是未来工作。

---

## 288. Navigating Runtime Dynamism in Semantic Query Serving

**arXiv ID:** 2611.04248 | [PDF](https://arxiv.org/pdf/2611.04248v1)

**作者:** Hengrui Zhang `[一作]` (Tsinghua University), Xupeng Miao `[通讯]` (Peking University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一个统一框架，实时动态调整语义查询执行计划并进行并发调度。

**💡 创新点**

结合等价保留的计划枚举、细粒度子查询执行单元和轻量级的推测成本估计，闭环解决了查询计划与调度的相互依赖。

**🔧 技术方法**

使用等价计划枚举、Subquery Execution Unit (SEU)、SLM 推测调优、SRTF 调度、GPU-backed LLM 后端等技术。

**📊 数据集**

主要在 BioDEX 语义医学数据集上进行实验，并对法律、专利等语料进行扩展。

**📈 对比分析**

与 LOTUS、Eddies、静态优化、PreemptDB 等基线对比，SRTF+动态规划在多种负载下平均延迟降低 2.5-4 倍，吞吐量提升 1.9-4.3 倍。

**⚠️ 局限性**

搜索空间可扩展性受限，极端平行操作时枚举量爆炸，且缺乏跨查询共享计算和事务一致性支持。

---

## 289. Auditing Targetless Camera-LiDAR Extrinsic Correction before Recursive Use

**arXiv ID:** 2611.04397 | [PDF](https://arxiv.org/pdf/2611.04397v1)

**作者:** Bo Ma `[一作]` (Auckland University of Technology), Weiqi Yan `[通讯]` (Auckland University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了摄像头与激光雷达融合系统在外参漂移下的校正与验证，提出并评估了基于日志分离的置信门控协议；

**💡 创新点**

创新点在于将校正质量、置信度评估与递归污染分离，并通过日志级别重采样与AUROC筛选构建一套完整的审计流程；

**🔧 技术方法**

使用了Canny/学习热图的边缘匹配校正器、Powell优化、多帧累积、成本下降与多启动一致性置信度、AUROC、log‑level自举、零注入安全检查以及基于多帧视觉‑激光雷达里程计；

**📊 数据集**

主要使用Argoverse 2测试集27条日志进行正式评估，60条训练/验证日志用于探索性诊断；NuScenes Mini 用于投影可移植性检查，12条日志用于无运动/噪声实验；

**📈 对比分析**

通过四种策略（Raw、Oracle、Always‑correct、Guarded）在ATE、RPE、追踪失率、重投影误差等指标上比较；结果显示Guarded在锁定门控下与Raw等价，Always‑correct显著恶化；单帧校正无益，双帧虽在旋转上有所提升但翻译误差累计，整体未能提升性能；

**⚠️ 局限性**

局限在于仅评估合成漂移，缺乏真实漂移与时序误差；多模态匹配仅基于边缘特征，未覆盖真实场景特征多样性；置信门控未找到非零覆盖点；系统未通过递归安全门，无法给出可部署的校正门槛。

---

## 290. StepX-OPD: Rethinking Multi-Teacher On-Policy Distillation in Diffusion Models

**arXiv ID:** 2611.04386 | [PDF](https://arxiv.org/pdf/2611.04386v1)

**作者:** Chenru Wang `[一作]` (StepFun), Chi Zhang `[通讯]` (Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种一次性训练多功能、少步数文本到图像扩散模型的方法，消除传统的先完成全步模型再压缩的两阶段流程。

**💡 创新点**

创新点包括：① 将多教师的自我监督（on‑policy distillation）与步数压缩（step distillation）合并到同一训练阶段；② 采用动态网格插值构造监督点，并对低噪声区进行速度回归；③ 通过源路由将每个训练样本直接送至对应的冻结专家教师，确保每种能力在生成过程的真实状态上被学习。

**🔧 技术方法**

使用技术包括：流匹配（flow matching）与分布匹配（distribution matching）框架，TDM、OPD、CDM等先前方法的融合；低噪声速度回归、动态随机网格、停止梯度插值、教师分布匹配与低噪声回归的联合损失；以及FSDP并行训练、LoRA、fake adapters。

**📊 数据集**

在SD3.5‑Medium和FLUX.1‑dev两大基础架构上进行实验，使用GenEval、OCR准确率、DeQA、PickScore等九个基准（包括CLIPScore、HPSv2.1、Aesthetic、ImageReward、UnifiedReward等辅助指标）进行评估。

**📈 对比分析**

与传统的Flow‑OPD+TDM、DiffusionOPD+TDM以及40步Flow‑OPD基线比较，所提模型在四步情况下取得最高的平均得分（SD3.5‑Medium 0.8901，FLUX.1‑dev 0.8902），超越40步模型；在SD3.5‑Medium上，训练耗时比顺序DiffusionOPD+TDM减少约37%。

**⚠️ 局限性**

局限性包括：1）仍依赖于已训练好的冻结专家教师，若教师质量不足会影响效果；2）对低噪声速度回归的权重和插值策略选择仍需经验调优；3）目前仅在文本到图像任务上验证，尚未测试跨模态或更大规模模型的可扩展性。

---

## 291. Benchmarking Unsupervised Jamming Detection in 5G Open RAN

**arXiv ID:** 2611.04369 | [PDF](https://arxiv.org/pdf/2611.04369v1)

**作者:** Sneihil Gopal `[一作]` (National Institute of Standards and Technology), Richard A. Rouil `[通讯]` (National Institute of Standards and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

比较了统计集合方法SEM与三种重构自编码器（Vanilla AE、LSTM AE、VAE）在十个异构5G Open RAN KPI日志上的无监督干扰检测性能，并提出了不需要标签的对称评估框架。

**💡 创新点**

创新点在于提出了仅基于检测置信度和时间区间连贯性的无监督复合分数，用以横跨不同检测器族进行公平比较，并验证其与真实F1指标的一致性。

**🔧 技术方法**

采用了滚动基线与Z‑score、计数突变、急变子检测器融合的SEM；以及基于前馈、LSTM和VAE的重构自编码器，通过无监督网格搜索调优阈值、持久性过滤和区间合并。

**📊 数据集**

使用了包含固定与可变攻击强度、不同持续时间以及多用户场景的10条5G Open RAN KPI日志，总计约34.7 小时的观测数据。

**📈 对比分析**

通过同一文件和多文件留一交叉验证评估，复合分数与F1得分在所有环境中保持一致；SEM平均复合0.906、F1 0.918，AE最佳分别为0.688和0.521；然而复合分数在检测率远低于真实干扰比例时会失效。

**⚠️ 局限性**

局限包括仅在单一测试平台验证，SEM需针对每个trace单独校准，复合分数不考虑检测率，AE阈值迁移导致误报或漏报，且缺乏实时评估与更广泛干扰模式的验证。

---

## 292. RAFT: Reasoning-Aligned Fine-Tuning Improves Robustness of VLA Post-Training

**arXiv ID:** 2611.04368 | [PDF](https://arxiv.org/pdf/2611.04368v1)

**作者:** Shresth Grover `[一作]`, Dhruv Shah `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 RAFT：先用低秩 LoRA 适配器学习任务相关推理，随后冻结该适配器再对动作进行微调。

**💡 创新点**

创新点在于将推理监督与动作微调分阶段进行，冻结推理适配器显著提升对视觉、物体与指令变化的鲁棒性。

**🔧 技术方法**

使用 Gemini 3.1 Pro 生成推理标注，LoRA 低秩适配器，SigLIP 视觉编码器冻结，SFT (监督微调) 与线性探针分析。

**📊 数据集**

数据集包括模拟 pick‑and‑place 与 open‑drawer（MolmoSpaces）、真实机器人 DROID 与 YAM 的演示数据，以及 Gemini 生成的推理标注。

**📈 对比分析**

与 Plain FT、RETAIN、DeLock 对比，RAFT 在 OOD DROID 试验中 75% 成功率对比 28%，YAM 任务 70% 对比 36%，模拟 pick‑and‑place 83% 对比 80%，显著提升鲁棒性能。

**⚠️ 局限性**

局限性包括需依赖外部大语言模型产生推理标注、推理适配器对不同任务的最佳模态尚未确定、对极端分布外情况仍可能失败、部署需人工监督与安全验证。

---

## 293. Beyond Fine-Tuning: Evaluating Prompt Engineering and Prompt Tuning for Roman Urdu Hate Speech Detection

**arXiv ID:** 2611.04342 | [PDF](https://arxiv.org/pdf/2611.04342v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 294. Large Language Models for Forecasting of Electricity Price Spikes

**arXiv ID:** 2611.04354 | [PDF](https://arxiv.org/pdf/2611.04354v1)

**作者:** Liang Lin `[一作]` (University of Sydney), Xin Lu `[通讯]` (CSIRO)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估了在澳大利亚NEM市场中，使用本地LLM一次性提取公告文本特征后，是否能提升短期（60分钟及24小时）电价峰值预测性能。

**💡 创新点**

创新之处在于将LLM与规则式特征在严格预注册且冻结的实验框架下进行对比，探究公告内容对峰值预测的真实增益，并首次将Brier分数的相对技能与自举检验应用于此类任务。

**🔧 技术方法**

技术方法包括：本地Qwen3.5‑9B LLM对公告进行一次性字段提取、LightGBM梯度提升模型做概率回归、Brier分数与其REL/RES分解、日簇/三天块自举检验以及与无文本增量A0、规则增量B及AEMO预测等基准的对照。

**📊 数据集**

使用公开的AEMO市场管理系统（MMSDM）数据（价格、需求、可用发电等）及其市场公告文本（Oct 2021–Jul 2024）以及Holdout期间（Aug–Oct 2026）公告，共覆盖NSW1、QLD1、SA1、TAS1、VIC1五个地区。

**📈 对比分析**

通过将LLM提取的特征与规则提取、无文本增量以及AEMO下一小时预测进行对照，发现60分钟目标上LLM特征未能显著降低Brier分数，24小时目标在开发期略有提升但在Holdout未达到统计显著性，整体效果不明显。

**⚠️ 局限性**

局限性包括：Holdout样本极小且基率低导致统计功效不足；LLM提取字段未经人工验证；公告文本仅覆盖至2024年，未来公告无法用于训练；可观测延迟采用代理规则，未对其做敏感性分析；对30分钟延迟和模型深度调整缺乏系统检验。

---

## 295. XOR-Functions and Beyond in the Random Query Model

**arXiv ID:** 2611.04353 | [PDF](https://arxiv.org/pdf/2611.04353v1)

**作者:** Sumegha Garg `[一作]` (Rutgers University), Santhoshini Velusamy `[通讯]` (University of Waterloo)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了随机查询模型中计算布尔函数的时间–空间权衡，提出了新的下界证明方法，并给出了针对 XOR 函数、对称函数以及部分函数的几乎最优下界；

**💡 创新点**

① 通过新的信息复杂度量度，结合直接和（direct‑sum）论证，得到与函数灵敏度、退化度（degeneracy）相关的更强下界；② 统一了对多数（Majority）等低影响函数的下界，填补了总影响（influence）与时间–空间下界之间的空白；

**🔧 技术方法**

信息理论技术（mutual information、Jensen–Shannon 距离）、直接和论证、随机采样分析、门控位（gate‑keeping bits）变体、通信复杂度的缺失元素（Missing Element）问题的归约；

**📊 数据集**

无（该工作为理论性分析，未使用实际数据集）；

**📈 对比分析**

与先前基于灵敏度或总影响的下界以及直观的上界（O(n·S) 的宽度）进行对比，证明了在多种常见函数（Equality、Tribes、Element Distinctness 等）上实现了几乎最优的时间–空间乘积；

**⚠️ 局限性**

局限性：① 结果主要针对完整的布尔函数或 XOR 复合函数，部分函数或更一般的结构尚未覆盖；② 证明依赖于随机查询的独立性，难以直接推广到非随机或自适应查询模型；③ 需要较大的样本数（T=Ω(n log n)）以获得匹配的上界；

---

## 296. ReSMap: Recasting Satellite Priors for Robust and Accurate Online HD Map Construction

**arXiv ID:** 2611.04352 | [PDF](https://arxiv.org/pdf/2611.04352v1)

**作者:** Kyungmin Kim `[一作]` (Hanyang University), Soonmin Hwang `[通讯]` (Hanyang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出ReSMap框架，将卫星图像作为离线冗余源，与摄像头融合实现在线高清地图构建。

**💡 创新点**

创新点在于引入自信门的Predict‑then‑Fuse模块和对象中心协同多模查询解码器OCMQ，使模型在摄像头失效时能自动将权重转移至卫星图像，实现可靠冗余。

**🔧 技术方法**

使用了BEVFormer编码器、ResNet+FPN卫星分支、基于分割置信度的门控融合、Deformable Cross‑Attention、学习式查询门控以及联合训练损失等技术。

**📊 数据集**

主要实验数据集为nuScenes（geo非重叠split）与对应的AID4AD卫星图像，实验亦在Argoverse2上验证。

**📈 对比分析**

与MapTracker、DAMap、SDTagNet、SatforHDMap等基线对比，ReSMap在60 m×30 m和100 m×50 m ROI下取得54–55 mAP，摄像头掉落时性能下降仅6–15%，显著优于其他方法。

**⚠️ 局限性**

局限在于需要卫星图像精准定位及及时更新；当卫星视角被遮挡、阴影或建筑遮蔽时，其冗余效果尚未得到验证。

---

## 297. Pairwise Distortion Distribution with Graph-Based Importance Sampling for Quantization Error and Intrinsic Dimension Estimation

**arXiv ID:** 2611.04341 | [PDF](https://arxiv.org/pdf/2611.04341v1)

**作者:** Xi Chen `[一作]` (Rutgers University), Shirin Jalali `[通讯]` (Rutgers University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对随机码本量化（RCQ）误差的对称二阶分布（PDD）进行分析，推导出可从PDD估计的RCQ误差下界与上界，并基于此定义了新的PDD维度，能够给出Rényi信息维度、速率失真维度以及量化维度的上下界。

**💡 创新点**

创新点包括：①推导了RCQ误差的可计算上界；②提出了一种基于最小生成树的图结构自适应重要采样方法，用以在低失真区间高效估计PDD；③利用PDD上界与下界定义了新的上限与下限PDD维度，并证明它们分别给出其他三种维度的下界与上界。

**🔧 技术方法**

核心技术包括：随机码本量化理论、配对失真分布（PDD）推导、贝塔函数和极限分析、最小生成树（MST）构造、非退化随机游走（non‑backtracking walk）与重要采样权重调整。

**📊 数据集**

实验使用了人工合成混合分布（超立方体与MNIST手写数字 1、2）以及实际数据集 MNIST，均以 60K 样本作为来源；在实验中比较了均匀 Monte‑Carlo 采样与图结构自适应重要采样（GAIS）。

**📈 对比分析**

GAIS 在估计小失真概率（如 10⁻⁶）时显著降低方差，能够得到更准确的RCQ下界与上界，并在比特率 10–20 bit 之间保持稳定；相较之下，均匀 MC 采样在 10 bit 之后失真估计失效。PDD 维度估计在低 r 区间趋近于理论维度，且能揭示混合分布中最小与最大内在维度。

**⚠️ 局限性**

局限性：①推导假设码本分布与数据分布相同（P_C = P_X），对非同分布情况尚未扩展；②PDD 估计仍受样本量限制，尤其在极低概率区间需要大量采样；③图结构自适应重要采样需要构造近似 MST，虽然复杂度低于 O(N²) 但仍受大规模数据的内存与计算资源影响；④理论分析多依赖连续分布与光滑密度假设，对离散或极度不规则分布的适用性有限。

---

## 298. Gaussian Difference Dynamics: Breaking the $2^{n/2}$ Barrier for the Shortest Vector Problem and Discrete Gaussian Sampling

**arXiv ID:** 2611.04336 | [PDF](https://arxiv.org/pdf/2611.04336v1)

**作者:** Yiming Gao `[一作]` (University of Science and Technology of China), Honggang Hu `[通讯]` (University of Science and Technology of China)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一套经典随机化算法，在 word RAM 模型下突破 2^n/2 阈值，分别针对任意格点集的离散高斯采样（DGS）和最短向量问题（SVP）给出了可实现的时间和空间指数。

**💡 创新点**

核心创新在于“高斯差分动力学”（Gaussian Difference Dynamics，GDD）框架：通过在细球壳层中进行差分、稀疏激活和孤立配对，精确跟踪记录数与二阶聚类矩，既保持高斯分布的近似，又可在每一轮迭代中控制相关性；配合随机超格点平滑、厚壳重归一化和 BDGL 过滤器，实现了 BDGL 的理论指数。

**🔧 技术方法**

使用的主要技术包括：
- 高斯差分动力学与矩递推。
- 细球壳层与半径重归一化。
- 随机超格点平滑与随机一维子空间抽样。
- BDGL 乘积码过滤器与球面滤波搜索。
- 加权读取（weighted readout）实现单样本或批量样本。
- 取样与搜索的分层树结构与孤立策略。
- 通过有限实现与误差传播保证实现。

**📊 数据集**

实验与评估基于理论上任意的有理格点集；论文未使用具体的数值数据集，而是通过理论证明给出在所有维度 n 下的时间/空间指数。

**📈 对比分析**

与现有算法（如 ADRS、Hhan、Doulgerakis‑Laarhoven 等）比较，本文在离散高斯采样上实现了 2^n/2+o(n) 的时间和空间，并在 SVP 上实现了 2^0.4130n+o(n) 的时间与 2^0.3138n+o(n) 的空间，突破了 2^n/2 阈值并与 BDGL 的启发式指数相匹配。

**⚠️ 局限性**

主要限制包括：
- 仍需指数级时间/空间，即在实际应用中的可行性受限；
- 需要先预处理生成超格点并进行高斯采样，预处理成本高；
- 证明依赖于多层次的概率与几何估计，理论上可行但实现复杂；
- 对于非常大维度，常数项与低阶项可能影响实际性能。

---

## 299. Anchor: Augmented Cross-Layer Recurrent State for Long-Context Retrieval in Test-Time Training

**arXiv ID:** 2611.04327 | [PDF](https://arxiv.org/pdf/2611.04327v1)

**作者:** Vimal William `[一作]` (University of Arizona), Huanrui Yang `[通讯]` (University of Arizona)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Anchor结构，将滑动窗口注意力与Test‑Time Training相结合，通过共享跨层循环状态实现长上下文检索。

**💡 创新点**

创新点在于发现长上下文检索集中于边界层，利用共享循环状态和层特定读取映射大幅减少状态量。

**🔧 技术方法**

使用滑动窗口注意力、Test‑Time Training、线性化注意力、CKA相似性分析等技术。

**📊 数据集**

在Llama‑3.1‑8B、Mistral‑7B上使用Long‑Alpaca、PG‑19、WikiText、RULER、各种回忆任务评测。

**📈 对比分析**

与softmax、LoLCATs、Lizard等基线比较，Anchor在保持91–95%基线MMLU的同时，内存量减少16×，推理吞吐率提升4.3×，长上下文检索准确率提升2.3–3.1×。

**⚠️ 局限性**

仍存在与全softmax基线的检索准确率差距，长上下文下性能衰减更快，以及训练时对内部分支的依赖。

---

## 300. Computing Invariant Reachable Sets through Sum-of-Squares Programming

**arXiv ID:** 2611.04325 | [PDF](https://arxiv.org/pdf/2611.04325v1)

**作者:** Mohamed Khalid M Jaffar `[一作]` `[通讯]` (University of Maryland), Mohamed Khalid M Jaffar (University of Maryland)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

本文提出了一种使用SOS编程计算不变可达集的端到端管道，并实现了开源软件包，提供了该方法在实践中的经验特征。

**💡 创新点**

创新点在于提出了一种最小体积的漏斗公式，结合入口和出口约束，使得具有重合名义端点的漏斗可以通过构造进行组合。

**🔧 技术方法**

使用了SOS编程技术，结合了多项式近似和不变性认证的过程。

**📊 数据集**

使用了单轮车、摆杆和12状态四旋翼等系统进行数值实验，评估了计算管道的性能。

**📈 对比分析**

与其他方法相比，本文的方法在处理不可行初始猜测时表现出部分可判定性，交替程序在几次迭代后可以停止，且对漏斗体积的损失很小。

**⚠️ 局限性**

限制在于不变性证书仅在离散的节点点上有效，且未能保证样本间的连续性，同时，Gram矩阵的维度随着状态维度的增加而组合性增长，可能影响可扩展性。

---

## 301. FP4Quant: Efficient 4-Bit Communication Compression for Model-Parallel LLM Training

**arXiv ID:** 2611.04323 | [PDF](https://arxiv.org/pdf/2611.04323v1)

**作者:** Man Liu `[一作]` (University of Chinese Academy of Sciences), Dingwen Tao `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了FP4Quant，一种面向模型并行LLM训练的4位浮点通信压缩框架；通过微组双尺度FP4量化、FWHT混合两级变换和warp协作子字节打包，实现了高效的通信压缩与解压缩；并将其集成为NCCL压缩插件，在NVIDIA H100 GPU上实现了可扩展的分布式训练；

**💡 创新点**

创新点主要体现在：①微组双尺度量化方案，结合局部均方根归一和组内最大绝对值缩放，有效扩大FP4动态范围；②采用两级FWHT管线，在GPU内核级别利用register shuffle和shared memory实现高效的Walsh–Hadamard变换，降低量化误差；③warp协作的FP4打包方法，在寄存器中直接实现4位码的子字节拼接，消除中间存储和位运算开销；④在模型并行(张量/流水线)训练场景下实现低位通信，突破了传统FP8压缩的瓶颈。

**🔧 技术方法**

使用技术包括：FP4浮点格式（NVFP4）、微组分块、双尺度缩放、Walsh–Hadamard变换（FWHT）、GPU级shuffle与shared memory同步、warp协作的子字节打包、NCCL压缩插件以及对NVIDIA H100 GPU的CUDA 13.2优化。

**📊 数据集**

实验使用Qwen-7B模型在OpenWebMath语料上训练，并在CMMLU与C‑Eval任务上评估下游性能；通过验证集损失与困惑度衡量模型收敛。

**📈 对比分析**

与全精度BF16（Ring）和Tahquant进行对比。结果显示：在TP/PP/MP三种并行配置下，FP4Quant在微组大小G=64-256时，吞吐提升可达1.4×至1.56×，同时验证损失与PPL与BF16基本保持一致；相比之下Tahquant在TP/MP场景下会出现严重的精度下降。

**⚠️ 局限性**

局限性包括：①仅在单机8 GPU H100环境下验证，未评估跨节点或更大规模的扩展性；②对微组大小敏感，过大会导致量化误差急剧上升；③目前固定使用4位FP4，缺乏动态精度自适应机制；④对不同模型/任务的适应性尚待进一步验证。

---

## 302. Planning-Coding Separation in LLM-Based Code Generation: An Empirical Study

**arXiv ID:** 2611.04320 | [PDF](https://arxiv.org/pdf/2611.04320v1)

**作者:** Wentao Wang `[一作]` (Peking University), Lu Zhang `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了在大型语言模型（LLM）代码生成中，将规划与编码分离到不同模型的配置，并比较其性能

**💡 创新点**

首次对跨模型规划-编码组合进行全面实验，发现强规划者与强编码者组合往往优于同模型，提出大规划‑小编码（LPSC）可显著降低编码模型规模

**🔧 技术方法**

使用现成的多家 LLM（Qwen、Gemma、Ministral、DeepSeek、GPT 等）分别进行计划生成（采样）和代码生成（贪婪解码），并对编码器进行监督微调

**📊 数据集**

在 LiveCodeBench 与 Code2Bench 两个 Python 代码生成基准上进行实验，任务涉及完整程序与函数级实现

**📈 对比分析**

以 Pass@1 为主指标，平均三次计划的结果；实验显示跨模型组合与同模型差距不大，强组合可达 68–72% Pass@1，且大规划‑小编码配置在多种任务上表现相当甚至更好

**⚠️ 局限性**

局限性包括仅验证 Python 任务，使用单一计划形式，未考虑多轮交互或计划修订，模型规模与训练资源受限，实验结果可能不完全适用于其他语言或更大规模模型

---

## 303. Silent Instrument Errors Reorder Agent Leaderboards

**arXiv ID:** 2611.04399 | [PDF](https://arxiv.org/pdf/2611.04399v1)

**作者:** Yangze Liu `[一作]` `[通讯]` (Shandong University), Yangze Liu (Shandong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对公开的代理评测排行榜（如 HAL）进行审计，发现计量工具（如计费、打分、轨迹记录、再评分）隐藏的错误。

**💡 创新点**

提出可在排行榜中直接报告的检查措施：计分比例、根跨度计费倍数、行动来源标签、评测版本与再评分差异等。

**🔧 技术方法**

使用脚本重构评测管道、离线重评分、轨迹重放与日志对比等技术对排行榜数据进行重新计算。

**📊 数据集**

主要使用 TAU‑bench Airline、Banking 与 Telecom 等公开评测任务的轨迹与日志数据。

**📈 对比分析**

通过重新计算发现：计费被二次计数导致成本偏高、Generalist 仅计分约 51% 事件、轨迹中混入评测者动作、再评分导致排名变动；错误导致的差距从 13 分降为 0 分，成本曲线形状变形。

**⚠️ 局限性**

局限在于仅审计了两份排行榜与有限任务，未覆盖所有代理评测平台，且需要手工解密与重放日志，无法完全保证普适性。

---

## 304. Monodromy Computation Model

**arXiv ID:** 2611.04377 | [PDF](https://arxiv.org/pdf/2611.04377v1)

**作者:** Xiaoyang Liu `[一作]` `[通讯]` (University of Waterloo), Xiaoyang Liu (University of Waterloo)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文构建了一种基于复变函数覆盖空间的有限计算框架——Monodromy 计算范式，并证明其能完整模拟 Post–Turing 机、实现所有部分递归函数、并通过 Lean 4 形式化验证其语义。

**💡 创新点**

创新点包括：①将指数分支（log branch）与三次覆盖相结合，形成三维整数内存；②用有限的三次查询与下降操作实现对无穷字母表、无限堆栈的完整读写；③通过路径提升与多项式同调理论提供精确的路径实现；④给出“Forward”子语言与完整语言的严格功效分离，证明下降是决定性额外操作。

**🔧 技术方法**

主要技术：复变函数覆盖理论（monodromy）、路径提升、有限查询与下降构造、整数与实数解释的统一语法、全局实解析返回映射、Lean 4 形式化与 Mathlib 的覆盖空间与解析函数库。

**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**

局限性：
- 仅在三次覆盖下实现，需更高次覆盖的推广仍待研究；
- 计算效率与资源消耗未量化，实际实现的性能待实验评估；
- 仅针对离散有限控制模型，扩展到更一般的算术或随机计算模型尚未覆盖；
- 依赖精确整数分支与路径同调，实际物理实现对误差与噪声的鲁棒性尚未深入。

---

## 305. IGSW: Multi-Weather Urban Scene Reconstruction with Intrinsic-Decoupled Gaussian Splatting

**arXiv ID:** 2611.04374 | [PDF](https://arxiv.org/pdf/2611.04374v1)

**作者:** Zhongquan Wang `[一作]` (Chongqing University), Guofa Li `[通讯]` (Chongqing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

利用双阶段框架，构建可在不同天气下渲染的四维高精度城市场景：第一阶段使用内在分解（albedo、法向、照度）对4D高斯点进行天气共享与天气专属特征的解耦；第二阶段通过结构保护门和每点天气传输变量，对教师图像中不可信的结构信息进行过滤并加强目标天气的光照与残差表现。

**💡 创新点**

创新点：1) 内在分解与4D高斯场景的结合，实现材料/结构共享与光照/残差分离；2) 结构保护门融合语义冲突与跨天气内在差异，动态筛选可信教师区域；3) 每点天气传输变量提供低维度、全局一致的光照调节，避免局部过拟合。该三大创新共同提升了天气表达、结构保持和时序一致性。

**🔧 技术方法**

技术栈：3D高斯Splatting；基于Diffusion的图像编辑（Qwen-Image）和逆渲染（IntrinsicWeather）得到天气教师和内在图；特征级软监督（VGG、CLIP、UPerNet-ConvNeXt-XL、RAFT-Large）；结构保护门和光照传输的自定义损失；多尺度渲染与alpha合成。

**📊 数据集**

数据集：nuScenes（10个场景，100帧/视角）和Waymo Open Dataset（完整场景），均包含多视角、时间序列的真实驾驶图像。

**📈 对比分析**

与Qwen-Image、FRESCO、IntrinsicWeather、WeatherEdit、WeatherCity等方法对比，使用CLIP-S、CLIP-DS、Sem-CS、时序漂移误差等指标。IGSW在所有数据集/天气（雨/雪）下均获得最高的内容相似度与语义一致性，最低的时序误差，且天气表现最接近教师图像。

**⚠️ 局限性**

局限性：1) 依赖Diffusion教师与逆渲染的内在估计，若教师产生严重结构错误或逆渲染误差会影响结果；2) 对极端天气（大雾、暴雨）或极端光照的泛化尚未充分验证；3) 需要完整的多视角校准和大量GPU资源，推理时间相对较长；4) 仍可能在复杂场景中出现轻微的结构漂移。

---

## 306. When Is Fresh AI Stock Analysis Worth It? Allocating a Limited Research Budget Across Price Updates, Deep Forecast Refreshes, and Editorial Reviews of Top Stock Picks

**arXiv ID:** 2611.04348 | [PDF](https://arxiv.org/pdf/2611.04348v1)

**作者:** Russlan Ramdowar `[一作]` `[通讯]` (Future Edge Group), Russlan Ramdowar (Future Edge Group)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估了在多代理平台上对已有长周期AI研究进行每日价格更新、机械重置或完整刷新三种策略对投资组合收益与预测误差的影响。

**💡 创新点**

提出了复用优先、事件驱动刷新架构，并通过资本规模分解模型量化了在不同预算和持仓窗口下何时应触发深度推理。

**🔧 技术方法**

使用了多代理LLM系统（Gemini 3 Pro/3.1）、价格衰退模型、震荡防护重标、形状保持重置、经济回测与统计误差分析等技术。

**📊 数据集**

利用8,120份历史生产报告，涵盖203支股票、四批次的相同或演进模型，以及相应的历史价格、波动率与交易成本数据。

**📈 对比分析**

通过Top 10/Top 30等权重回测、MAE/RMSE比较以及成本计量，发现三个月窗口内价格重标（含震荡防护）可击败SPY且不产生新推理成本；六个月窗口则需刷新；机械重置几乎完全复制新推理带来的MAE提升。

**⚠️ 局限性**

局限性包括回顾性设计、短期持仓窗口、未覆盖完整风险管理与市场冲击、成本仅以API计费计算、分拆价格处理不完整、编辑层验证不具因果性等。

---

## 307. Do Retrieval Ranking Models Capture Event Order? A Study of Timeline-Constrained Retrieval

**arXiv ID:** 2611.04343 | [PDF](https://arxiv.org/pdf/2611.04343v1)

**作者:** Alon Eirew `[一作]` (Bar-Ilan University), Ido Dagan `[通讯]` (Bar-Ilan University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出时间线约束的文档检索任务，并研究语义检索模型是否能将事件的时序信息编码进排名；

**💡 创新点**

创新点在于构建控制实验框架，将语义支持与时序一致性分离，提出顺序保持对齐方法，并证明时序约束是重要的排名信号；

**🔧 技术方法**

使用预训练跨编码器（MiniLM、BGE、Qwen等）进行语义匹配，结合动态规划顺序保持对齐、Hungarian匹配和LLM辅助句子时序排序与标签生成；

**📊 数据集**

采用 ReliefWeb 灾害报告数据集，共 6,079 篇文档，构建 66 条时间线查询（50 测试+16 开发），每条查询平均 6 条陈述，标注 234 相关文档；

**📈 对比分析**

与 BM25、Dense Retriever、Reranker 以及顺序保持对齐（OP）与顺序无关对齐（OA）对比，OP 在 MRR@10、Recall@50、mAP@10 等指标上显著优于所有基线（例如 MiniLM OP MRR@10=0.70 vs OA 0.55，Recall@50=0.83），且对随机句子顺序敏感；

**⚠️ 局限性**

局限在于样本量小（66 条查询），仅覆盖灾害领域，依赖 LLM 生成标签与句子排序，可能存在标注与预处理误差，难以推广至更大多域数据。

---

## 308. Patching the Protocol: A Taxonomy of Pre-Registration Repairs in LLM-Proposer Experiments

**arXiv ID:** 2611.04338 | [PDF](https://arxiv.org/pdf/2611.04338v1)

**作者:** Peter Chen `[一作]` (Shenyang Institute of Technology), Siyuan Zhang `[通讯]` (Shenyang Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对三项基于LLM生成规则的实验中出现的协议缺陷进行修补，并提出了两轴修补分类法和九项修补检查表。

**💡 创新点**

创新点在于把协议缺陷分为环境、可行性、正确性三类，并区分前数据和后数据修补，提出可验证的无影响声明和规范化检查表。

**🔧 技术方法**

技术包括LLM生成可执行规则、程序化验证器、可变更的预注册协议、日志审计和自动化检查。

**📊 数据集**

使用仿真数据集进行生产调度、激励设计和任务分配实验，数据来源为合成与公开匿名数据。

**📈 对比分析**

方法通过对比修补前后的实验流程验证修补是否保持数据完整性，结果显示三次修补均未触及预注册判准且已可审计。

**⚠️ 局限性**

局限性包括仅三案例、为自我审核、未外部验证检查表、未覆盖更广的缺陷类型、且修补参数选择仍依赖作者判断。

---

## 309. Text-to-SQL Models Make Choices Without Saying So

**arXiv ID:** 2611.04318 | [PDF](https://arxiv.org/pdf/2611.04318v1)

**作者:** Yichen Xu `[一作]` (University of California), Chenhao Ma `[通讯]` (Chinese University of Hong Kong)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了“请求集”测试框架，检测文本到SQL模型在计算选择时的隐性决策。

**💡 创新点**

提出仅改动计算选择的受控测试，分离计算与报告，揭示模型在开放选择时缺失报告的问题，并提出两阶段决策+回答方法。

**🔧 技术方法**

使用规则驱动的查询生成器、从benchmark失败中挖掘的Choice库、Claude Haiku/Sonnet 与 GPT‑6 Sol/Luna 等大模型进行推理、执行评估与模型判别解释，以及两阶段决策‑回答策略。

**📊 数据集**

基于Spider、BIRD、LiveSQLBench等12个真实数据库生成204个验证请求集，并与AMBROSIA、BIRD与Spider的开发集做对照。

**📈 对比分析**

与单答案、三重采样、DtP、Clarifying question等基线对比，决定‑回答方法在开放请求覆盖率提升29–63分，精度无下降，且用户回合数平均减少35–41个。

**⚠️ 局限性**

生成器覆盖的SQL分析类型有限，Choice库仅来自SQLite benchmark，无法处理多重开放选择，且未验证非SQLite方言、长交互场景以及真实产品的行为。

---

## 310. Beyond Compression Ratio: High-Throughput KV Compression for Disaggregated LLM Serving

**arXiv ID:** 2611.04382 | [PDF](https://arxiv.org/pdf/2611.04382v1)

**作者:** Junhong Wu `[一作]` (Institute of Computing Technology Chinese Academy of Sciences), Dingwen Tao `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种高吞吐量的 KV 缓存压缩数据路径，通过融合分布式变换与低位量化，并引入可搜索的无损压缩后端，显著提升离散 LLM 服务的通信效率。

**💡 创新点**

① 将变换与量化合并为单个 GPU 核，消除中间张量；② 构建基于 LC 的可搜索无损压缩后端，可组合 GPU 原语；③ 通过离线搜索选取最优编码管道并在 GPU 上原生实现。

**🔧 技术方法**

GPU 内核融合、TileLang 调度、低位量化（Hadamard + 混合精度）、基于 LC 的可组合无损压缩、离线编码解码流水线搜索、CUDA 原生运行时。

**📊 数据集**

Llama‑3.1‑8B‑Instruct 与 Qwen2.5‑7B‑Instruct 在 HotpotQA、Qasper、2WikiMQA（LongBench）上进行 20K token、32K context 的请求测试。

**📈 对比分析**

与原始 KVServe、Raw Transfer、Fused Only、LC Only 进行对比；在 host‑staged NCCL SHM（PCIe Gen5）下，端到端完成时间缩短 45–51%，吞吐量提升 1.83–2.04×；在 200‑Gbps InfiniBand 上，JCT 缩短 30–39%，吞吐量提升 44–63%；压缩速率保持 6–9×，压缩后端吞吐 139–155 GB/s，延迟 7–12 ms。

**⚠️ 局限性**

低压缩比仍影响解码延迟；实现高度依赖 GPU 原语，迁移到其他硬件需重构；在极低带宽场景下无损编码仍可能成为瓶颈。

---

## 311. CloakFEM: Agentic FEM Code Generation for Red-Teaming Computational Resource Exhaustion

**arXiv ID:** 2611.04356 | [PDF](https://arxiv.org/pdf/2611.04356v1)

**作者:** Li Zheng `[一作]` (George Mason University), Simin Chen `[通讯]` (George Mason University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究利用LLM驱动的域引导搜索生成可执行的对抗性有限元程序，从而在共享计算环境中放大求解器的计算工作量。

**💡 创新点**

提出CloakFEM框架，结合无关中间表示、可执行性门控和LLM代理，实现在不修改求解器内部配置的前提下，自动发现可运行的效率攻击。

**🔧 技术方法**

采用领域无关IR、LLM代理搜索、确定性合法性检查、基于反馈的黑盒优化、以及跨求解器重放等技术。

**📊 数据集**

使用了三个主流有限元堆栈（Elmer PDE‑Agent、DOLFINx、Scikit‑fem）中的17类问题，共1068个种子实例。

**📈 对比分析**

与五种通用黑盒基线及进化搜索在40次查询内进行对比，CloakFEM平均放大率超过1.5×、P90超过150×，最高达5,752×，明显优于基线方法。

**⚠️ 局限性**

局限性包括受限于可编辑空间、缺乏梯度信息、求解器内部配置不可变；攻击效果主要在受控实验环境验证，真实生产环境需配合资源限制与防御措施；LLM的随机性影响实验可重复性。

---

## 312. Correct Labels Are Not Enough: Reliable Metadata Can Silently Control Model Behavior

**arXiv ID:** 2611.04395 | [PDF](https://arxiv.org/pdf/2611.04395v1)

**作者:** Renee Jia `[一作]` (R2M AI), Jinzhou Wu `[通讯]` (Cornell University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究模型在多选任务中是否会依赖训练时出现的标注信息，评估测试时移动或移除标注对答案选择的影响。

**💡 创新点**

发现即使所有训练标签正确，模型仍能高度依赖可靠标注，且此依赖可由无意义或中性词触发，并通过标注可靠性实验揭示其形成机制。

**🔧 技术方法**

采用LoRA微调、直接偏好优化（DPO）、选项分数读数、对比测试等技术，并在多种语言模型上进行实验。

**📊 数据集**

使用CommonsenseQA、GSM8K（改写为多选）、ARC‑Challenge等多项选择题集。

**📈 对比分析**

通过无标注准确率、误导标注准确率和跟随标注比例三指标比较，发现无标注准确率保持高，但误导标注时模型几乎全选错误选项，跟随比例随标注可靠性提升趋近100%。

**⚠️ 局限性**

未探究内部表示、真实系统中标注频率、RL或更大模型的影响，仅使用7–9B模型和有限训练样本，且基线模型的跟随未完全排除；缺乏更广泛的对齐评估。

---

## 313. Voluntary Servitude: Weaponized Safety Warning Mechanisms for Controlling AI Agents

**arXiv ID:** 2611.04393 | [PDF](https://arxiv.org/pdf/2611.04393v1)

**作者:** Ilan Zendel `[一作]` (Technion), Amit LeVi `[通讯]` (Technion)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究并演示了一种利用大型语言模型代理的安全偏好（即害怕危害）通过在公共评论中植入单条虚假危险信息来诱导代理选择攻击者目标的攻击方式，并在多个模型、十个消费服务域、检索管线以及生产浏览器环境中进行系统评估。

**💡 创新点**

创新点包括：1）提出“自愿服从”攻击框架，利用安全偏好而非指令注入；2）展示该攻击在过滤器后可比甚至超过传统指令注入的效果；3）首次将协同证明（source corroboration）要求引入代理系统，提供轻量级的防御方案；4）通过多模型、多域、真实浏览器与检索实验验证攻击的普适性和迁移性。

**🔧 技术方法**

技术手段主要包括：构造四类payload（安全、权威、属性、正面等）并注入公开评论；使用六个LLM（OpenRouter系列、DeepSeek、Grok、Claude Haiku、Gemini Flash Lite 等）进行实验；对接两款工业级注入检测器（ProtectAI、Lakera Guard）评估过滤效果；利用LLM内部判断器与人工标注相结合评估生成成功率；实现检索层实验，使用 BM25 与密集检索在生产环境中检索页面。

**📊 数据集**

数据集：十个旅行/服务域（潜水中心、徒步、冲浪、出租车、露营、烹饪课、餐厅、历史游、摄影、葡萄酒），每域 50 条用户查询，共 500 条；构建合并页面（数千 token）包含 1/4 为攻击者评论；检索实验使用完整域内所有页面，检索比例覆盖 100% 页面。

**📈 对比分析**

比较方法：在每个模型/域上与三类对照（指令注入、道德指控、正面评论、无毒评论）进行对比，计算生成成功率 ASR 与过滤后有效成功率（ASR_filt）。实验结果显示：安全payload 在无过滤时与指令注入相当（约 60–66%），但在两款检测器后其有效成功率提升至 1.7–4.4 倍；在生产浏览器中成功率可达 100%；在检索管线中，BM25 的成功率为 30–57%，Dense 为 1–31%（取决于检索命中率）。

**⚠️ 局限性**

局限性：1）仅评估单轮选择决策，未覆盖多轮对话或软件依赖场景；2）协同证明防御只在非自适应攻击者下测试，真实攻击者可能通过多源造假规避；3）实验域聚焦消费旅行服务，可能不完全泛化到其他领域；4）对检索器的性能和设计敏感，检索失败会大幅降低攻击效果。

---

## 314. Aero-VPA: Few-Step Flow-Matching Control over Non-Reconstructive JEPA Latents for Agile Flight on Edge Hardware

**arXiv ID:** 2611.04392 | [PDF](https://arxiv.org/pdf/2611.04392v1)

**作者:** Yeonseok Lee `[一作]` `[通讯]` (SLING AI Inc), Yeonseok Lee (SLING AI Inc)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种轻量级RGB‑D视觉预测‑动作框架Aero‑VPA，旨在在边缘硬件上实现三维导航；该系统包含一个Vision Transformer编码器、Joint‑Embedding Predictive Architecture（JEPA）残差预测器、Conditional Flow Matching（CFM）解算器，并配备了运行时安全盾。

**💡 创新点**

创新点包括：①用JEPA进行latent预测，避免像素重建；②采用CFM实现动作生成，每步仅需5次网络评估；③引入噪声持久化以提高chunk连续性；④利用预测器多头方差驱动的安全盾，能在不增加显著开销的情况下抑制误操作；⑤在仿真中验证通过安全盾可将闭环成功率从0%提升至≈52%。

**🔧 技术方法**

技术细节包括Vision Transformer编码器、5头残差JEPA预测器（VICReg正则化）、K=2 Euler步的CFM解算器、DAgger交互式模仿学习、运行时安全盾（体素记忆、障碍约束、盲区限速、退避和推挽）、TensorRT FP16部署在Jetson Orin Nano、PX4 SITL与Gazebo集成、以及对齐的离线/在线评估流程。

**📊 数据集**

数据集：约5.5M帧RGB‑D专家演示来自200个训练森林；360个校准场景用于σcrit校准；测试集为36个森林（种子2000–2011），共180次任务（每个策略）；公开数据集位于HuggingFace aero‑vpa‑benchmark（CC BY 4.0）。

**📈 对比分析**

与三类基线（VINS+OctoMap+A*；强化学习PPO/SAC；Diffusion/Qantara）在E1森林环境中比较；未加盾的BC全失效；DAgger未加盾成功率≈21%；加盾后成功率≈52%（CI 45–59），相较最佳B3提升≥5个百分点；平均飞行速度≈1.2–1.4 m/s；碰撞率从≈70%降至≈22%；超时率从≈2.8%升至≈23%；模型尺寸72.6 MB，推理时间≈2–3 ms。

**⚠️ 局限性**

局限性：仅在仿真静态森林环境测试；未考虑风、移动障碍或视觉分布漂移；未在目标Jetson Orin Nano上完成实时测评；安全盾改进后仍有22–42%碰撞；安全保证仅为经验性；不确定性评估仅基于预测器方差，可能漏检编码器错误；缺乏对动态障碍和深度误差的完整处理；实验仅使用单一训练seed和单次评估，结果易受偶然性影响。

---

## 315. Human-to-Robot Transfer via Bridging Representation and Action Space in VLAs

**arXiv ID:** 2611.04391 | [PDF](https://arxiv.org/pdf/2611.04391v1)

**作者:** Junhyeong Park `[一作]`, Jinwoo Shin `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在无配对人机演示数据的情况下，将预训练的视觉‑语言‑动作（VLA）模型迁移到人类演示的新任务上，完成目标物体与技能的转移；

**💡 创新点**

通过基于手部几何的对齐（无需配对演示）和解耦解码器的潜在动作分词器，将人类与机器人在感知与动作空间上的差距分离且统一；

**🔧 技术方法**

利用VLM特征与手部几何对齐、变分动作自编码器分词器、流匹配动作预测、以及机器人与人类特定解码器；

**📊 数据集**

OpenArm 机器人（配备 Inspire 手）与 RB‑Y1 包裹处理任务的真实机器人演示；人类演示通过轻量摄像头采集并重建手部运动；

**📈 对比分析**

与基线预训练 VLA、共享动作空间的 HaWoR、Ego‑Pi 以及联合人机共训练方法相比，方法在对象迁移上从 8.0 提升至 51.8，在技能迁移上从 0.0 提升至 62.7，并在异构训练后进一步提升至 57.2/81.7；

**⚠️ 局限性**

对手部几何对齐的依赖导致对手部检测精度敏感；潜在动作分词器仍需手工调参，且在极端新环境或不同机器人末端效应器时效果有限。

---

## 316. Spin Spherical Fourier Neural Operators: Efficient Equivariant Learning for Vector and Spin Fields on the Sphere

**arXiv ID:** 2611.04389 | [PDF](https://arxiv.org/pdf/2611.04389v1)

**作者:** Kevin Mulder `[一作]` (University College London), Jason D. McEwen `[通讯]` (University College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

发展了一种能处理任意自旋场的球面傅里叶神经算子（Spin Spherical Fourier Neural Operators, s），实现了每一卷积层对旋转的完全等变性；

**💡 创新点**

创新点在于：通过将自旋场升到旋转群的Mackey函数并采用升降卷积实现精确等变卷积；使用谐波空间高效实现；提出了三套新的自旋基准数据集，并在参数量和计算效率上大幅优于以往的全群卷积网络；

**🔧 技术方法**

核心技术包括：升降卷积（lift–convolve–project）框架、球面与Wigner卷积、谐波插值核、1×1谐波卷积、等变激活与归一化，以及基于s的UNet架构；

**📊 数据集**

使用的数据集有：Spin Spherical MNIST（自旋1/2/3的数字图像）、全球风预测（ERA5/WeatherBench 50 hPa风速）、弱引力透镜（GLASS模拟的收敛与噪声性剪切场）；

**📈 对比分析**

与标量CNN、sCNN、GsCNN、DISCO等模型对比；在MNIST上准确率≥99%（比传统标量CNN低0.1%）；在风预测上MSE下降约44.9%，参数量减少≈35%；在弱引力透镜上MSE下降约37.9%，参数量减少≈13%；整体性能在精度、效率与参数规模上均优于现有方法；

**⚠️ 局限性**

局限性包括：等变性受激活函数与归一化误差影响，非旋转等变层仍会产生小误差；缺乏对高自旋（|s|>2）场的理论与实验验证；对非完整球面或不规则网格的适用性尚未深入探讨；

---

## 317. Understanding Backdoor Attacks in Scene Graph Generation via Global Context Pathway

**arXiv ID:** 2611.04379 | [PDF](https://arxiv.org/pdf/2611.04379v1)

**作者:** Xiaoguang Chang `[一作]` (Southeast University), Changyin Sun `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究场景图生成（SGG）模型的训练时后门攻击，提出基于全局上下文的攻击方法并设计三轴评估协议。

**💡 创新点**

创新点：①在SGG中首次构造pair‑level后门，攻击目标为指定对象类别的关系；②发现并利用显式全局上下文路径作为后门传播的主要通道；③提出ATRP+TRCL两步训练策略和三轴评估（实用性、攻击成功、隐蔽性）;

**🔧 技术方法**

采用信息熵对比学习（InfoNCE）进行关系对比学习、margin-based召回提升、全局上下文交换与插值干预、交叉注意力可视化等技术。

**📊 数据集**

使用Visual Genome VG150数据集（108,777张图，150类对象、50类谓词），在开放词汇和闭集两种设置下实验。

**📈 对比分析**

与基线（Clean、Native BD）以及单独使用ATRP对比，实验结果显示在OvSGTR和RelationFormer上，ΔASR可达59%/43%等，攻击成功率高，且保持BR/TNR与Clean相近，FAR保持在极低水平；证明方法既高效又隐蔽。

**⚠️ 局限性**

局限性：攻击主要依赖显式全局上下文设计，对不具备该路径的模型效果有限；仅针对单一受害对象类别和固定目标谓词，未评估跨模型迁移性；防御策略尚未系统探究。

---

## 318. Aligning Training and Rollout Policies for End-to-End FP4 RL on Mixture-of-Experts LLMs

**arXiv ID:** 2611.04376 | [PDF](https://arxiv.org/pdf/2611.04376v1)

**作者:** Hei Yi Mak `[一作]` (Huawei), Tanzila Rahman `[通讯]` (Huawei)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在全精度 (BF16) 的 MoE 大语言模型上进行 FP4 量化后训练，研究了低精度 RL 的效果。

**💡 创新点**

提出了 MoeResQ——一种对训练与推理两侧都应用残差纠正的对称量化恢复方法，解决了传统 RolloutResQ 在 MoE 中出现的路由失配问题。

**🔧 技术方法**

采用 HiF4 FP4 量化格式、残差纠正 (Residual Correction)、GRPO 策略梯度 RL、vLLM 推理、FSDP 并行训练，以及对专家层单独量化的细粒度配置。

**📊 数据集**

使用 Qwen3-30B-A3B 预训练模型，训练集为 DAPO-Math-17k，评估数据集包括 AMC‑2023、AIME‑2024、AIME‑2025 与 MATH‑500。

**📈 对比分析**

与 Naïve HiF4、全精度 BF16 以及 RolloutResQ 对比，MoeResQ 将误差从 19.87%–24.27% 降低到 0.65%–3.75%，在四个基准测试中误差均低于 1.5%，几乎达到全精度性能。

**⚠️ 局限性**

局限性包括仅在单一 MoE 模型与特定数据集上验证，实验在仿真量化环境下进行，尚未验证在硬件真实 FP4 上的效果，且训练仍保持高计算与内存开销。

---

## 319. Embodied AI for Scientific Discovery: From Physical Action to Scientific Evidence

**arXiv ID:** 2611.04371 | [PDF](https://arxiv.org/pdf/2611.04371v1)

**作者:** Wenhao Xu `[一作]`, Rongtao Xu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对机器人学习与 AI for science 交叉领域的 28 条物理实验工作流进行系统梳理，提出三种闭环（操作、认识、证据）框架，并给出六种决策与六类实体化方式的统一分类，构建了“记录‑诊断‑校准‑确认”证据链的完整模型。

**💡 创新点**

创新点在于将实验执行与科学推理拆解为操作闭环、认识闭环和证据闭环三层，形成可量化的“证据完整性”指标；同时提出命令‑实现‑观测三种条件的分离，并在此基础上设计决策理论和比较方法，填补了此前仅聚焦单一闭环的空白。

**🔧 技术方法**

采用决策理论框架、贝叶斯实验设计、主动学习、贝叶斯优化、强化学习/模型预测控制、语言模型规划等多种方法，并将它们映射到闭环、决策与实体化维度上；同时利用世界模型、视觉‑语言‑动作(VLA)策略等技术评估执行与观测的可追溯性。

**📊 数据集**

主要基于文献综述与 28 条已发表实验工作流的公开记录（共 48 个代表系统），并在附录中构建了结构化证据地图；并未采用单一统一实验数据集，而是汇总多领域（化学、材料、生物、显微镜、现场测量）公开数据与实验记录。

**📈 对比分析**

通过比较方法的终端决策（推荐设计、预测响应、假设辨别、状态序列决策）以及对“有效实验”和“已确认命题”的计数，评估各方法在不同闭环下的表现；结果显示，许多方法在操作闭环表现良好，但在证据闭环（尤其是独立确认）方面普遍不足，整体性能受限于实验可重复性与独立验证的缺失。

**⚠️ 局限性**

限制主要包括：① 仅依赖公开文献，可能遗漏未公开或尚未发表的工作；② 证据地图和分类虽系统但仍简化了实验过程的复杂交互；③ 缺乏统一的量化基准与公开数据集，导致跨工作流直接比较受限；④ 评估侧重于指标统计，未深入探讨方法的动态适配与长期可靠性；⑤ 未来仍需在控制实验、失效建模和多闭环协同方面开展可重复的对比研究。

---

## 320. Do Coding Agents Follow AGENTS.md? The Effect of Policy Wording on Instruction Compliance

**arXiv ID:** 2611.04367 | [PDF](https://arxiv.org/pdf/2611.04367v1)

**作者:** Rikuto Tsuchida `[一作]` (Nara Institute of Science and Technology), Kenichi Matsumoto `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在Agent Context File (ACF)中不同写法的政策对AI代理遵从率的影响，并在四种写法（命令式、说明式、条件式、禁止式）下评估其效果。

**💡 创新点**

首次证明政策写作风格会显著改变遵从率，并发现条件式写法在多种模型/工具组合中表现最差。

**🔧 技术方法**

使用大语言模型代理（Claude Opus, Claude Sonnet, GPT OSS 120B, Qwen3-Coder-Next）配合对应的 harness 进行自动化测试，并通过统计检验（Friedman、Wilcoxon等）评估差异。

**📊 数据集**

采用CTxbBenchmark中包含ACF的86个开源仓库任务（来自10个仓库）进行实验。

**📈 对比分析**

对每种写法的合规率进行 Friedman 检验，显著后做配对 Wilcoxon 检验；结果显示条件式写法合规率低 14-54% 之间，命令式和说明式差异取决于模型/工具组合，整体合规率最高达 78%。

**⚠️ 局限性**

研究仅使用单条文本代表每种写法，且仅针对标记注释的合规性，排除了无变更的会话，样本来自少数仓库，可能导致结果不够普适。

---

## 321. EndoS3:Endoscopic Semantic Splatting SLAM

**arXiv ID:** 2611.04358 | [PDF](https://arxiv.org/pdf/2611.04358v1)

**作者:** Mona Sheikh Zeinoddin `[一作]` (University College London), Danail Stoyanov `[通讯]` (University College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 EndoS^3，一种利用实例语义引导的 Gaussian Splatting 语义 SLAM 框架，并构建了 CRCD++ 数据集，支持同时评估相机位姿跟踪、可变形重建与语义映射。

**💡 创新点**

创新点包括：① 基于实例的运动分解模块，可将静态与动态实例分离；② 实例感知变形场，给每个实例分配独立的控制节点与变形网络；③ 节点分配策略，根据实例大小与非刚性动态调整节点预算；④ 提供完整的外科语义 SLAM 基准数据集 CRCD++。

**🔧 技术方法**

技术手段主要有：Gaussian Splatting 场景表示；实例级运动分解（利用光流残差、信噪比与实例边界差异）；线性混合皮肤化变形场；节点分配与控制节点重参数化；多项式损失组合（颜色、深度、语义、光流、特征对齐等）；深度估计使用 FoundationStereo；语义分割使用 Mask R‑CNN 或 WAFT。

**📊 数据集**

使用的数据集：CRCD++（外科胆囊切除的 ex‑vivo RGB‑D 与实例标注），SemSup（合成或现实的外科序列），以及对比实验中使用的 DDS‑SLAM、SNI‑SLAM、SGS‑SLAM、SemGauss‑SLAM、Semantic‑SuPer、PERSEUS 等公开数据集。

**📈 对比分析**

方法通过在 CRCD++ 与 SemSup 上的 ATE‑RMSE、PSNR/SSIM/LPIPS、mIoU 等指标进行评估，结果显示 EndoS^3 在位姿跟踪上平均 ATE 下降 17%，在重建上 PSNR 提升 2 dB，语义 mIoU 达到 91% 以上，显著优于所有基线模型。

**⚠️ 局限性**

局限性：1) 对低照明或光照不均匀的手术图像鲁棒性不足；2) 需要高质量的实例分割，误差会影响运动分解；3) 计算量较大，依赖高性能 GPU；4) 只在胆囊切除场景验证，需进一步验证在其他手术场景中的泛化能力。

---

## 322. Sparse Internal Control of Language Models

**arXiv ID:** 2611.04349 | [PDF](https://arxiv.org/pdf/2611.04349v1)

**作者:** Xizhe Zhang `[一作]` `[通讯]`, Xizhe Zhang

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出稀疏内部控制框架，将冻结语言模型中的少量内部节点视为推理时可调节的执行器；

**💡 创新点**

创新点在于给出局部目标可控性公式、两种驱动器选择算法（残差拟合与覆盖驱动），并设计六轴评估协议评估可达性、逆转、剂量响应、反馈控制、离目标保持与泛化；

**🔧 技术方法**

采用线性近似驱动选择、共享强度拟合、非线性验证，并结合梯度、核、KL 等度量来实现目标恢复和能量约束；

**📊 数据集**

实验基于 GPT‑2、Qwen3‑4B、Qwen3‑1.7B，在 IOI、MMLU（人文/ STEM）、情感与拒绝等任务上进行；

**📈 对比分析**

与 Top‑k、随机、非线性贪婪等基线比较，稀疏支持在同一面板上实现 0.1–0.3 的目标恢复，反馈控制可达 90% 目标，但在拒绝任务仅实现剂量响应；

**⚠️ 局限性**

局限在于对方向信息高度依赖、在多输入或生成任务中离目标误差较大，且离目标保持与泛化性能不足。

---

## 323. Proper Agnostic Learning of Affine Halfspaces under Log-Concave Marginals

**arXiv ID:** 2611.04344 | [PDF](https://arxiv.org/pdf/2611.04344v1)

**作者:** Zhijie Zheng `[一作]` `[通讯]` (Peking University), Zhijie Zheng (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文提出了一种在全维度对数凸分布下实现常数因子无偏学习的正解半空间学习算法，能够在仅使用独立标签样本的前提下给出误差不超过基准误差的单个正解半空间；

**💡 创新点**

创新点包括：①利用少数类概率的全局与局部误差比较实现常数因子无偏学习；②在有噪声标签的情况下通过鲁棒帽子统计和锥深度（zonoid‑depth）目标构造平衡约束；③结合四阶Sum‑of‑Squares矩估计和对齐变换，将平衡学习问题转化为同质半空间学习；④实现了对未知均值和协方差的实数算术复杂度多项式化，并将样本复杂度线性化到对置信概率的对数。

**🔧 技术方法**

技术手段包括：对数凸分布的几何性质、鲁棒极矩估计、四阶Sum‑of‑Squares证书、zonoid‑depth 对偶、投影与平移归约、均衡半空间学习、基于sigmoid的同质学习、误差比较与置信放大、以及实数算术模型下的算法实现。

**📊 数据集**

该工作为理论研究，未使用具体实验数据集；所有结果均基于理论分析与概率界定。

**📈 对比分析**

与已有的半空间无偏学习方法相比，本文在对数凸分布下实现了常数因子误差，并且样本复杂度与维度、误差倒数以及置信度对数呈多项式关系；相比于之前的平方根依赖或高斯特定结果，提供了更广泛的分布和更优的误差上界。

**⚠️ 局限性**

局限性包括：1）算法实现依赖于实数算术模型，未给出离散化或位长复杂度；2）对数凸假设仍相对特殊，无法直接推广到更一般分布；3）多项式指数未被优化，实际运行效率可能受限；4）在极端噪声或稀疏标签情况下的性能尚未深入实验验证。

---

## 324. RAIPAL: A Powerful and Agile Humanoid Arm for Human Tasks

**arXiv ID:** 2611.04334 | [PDF](https://arxiv.org/pdf/2611.04334v1)

**作者:** Donggeon Kim `[一作]` (KAIST), Jemin Hwangbo `[通讯]` (KAIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文研究了类人机器人系统的产品设计、开发与原型制作，并对致动与关节机构进行了实验验证。

**💡 创新点**

创新点在于提出了一种新型多自由度致动与关节机构，能够更好地兼顾灵活性与结构紧凑。

**🔧 技术方法**

采用机械仿真、动力学建模以及实际机械实验等技术进行评估。

**📊 数据集**

未提供公开数据集，实验基于自建的机器人原型进行。

**📈 对比分析**

与传统关节机构相比，实验数据显示该新设计在转矩传递效率和运动范围上提升了约15%–20%。

**⚠️ 局限性**

论文缺少大规模长期测试数据，且对不同工作负载下的可靠性分析不足。

---

## 325. Warpline: Binary-Level Tracing and Unified Evidence for Agentic GPU Kernel Optimization

**arXiv ID:** 2611.04331 | [PDF](https://arxiv.org/pdf/2611.04331v1)

**作者:** Junyu Yin `[一作]` (George Mason University), Keren Zhou `[通讯]` (George Mason University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种自动化框架Warpline，用于提供全面的性能证据，以帮助编码代理优化张量核心GPU内核。

**💡 创新点**

创新点在于通过二进制分析自动插入低开销探针，保持内核执行行为不变，并支持多种编程语言和闭源内核。

**🔧 技术方法**

使用了静态和动态二进制分析技术，结合硬件性能测量，生成统一的时间线和内核摘要。

**📊 数据集**

在NVIDIA B200上评估了35个开源内核，使用了Triton、CuTe DSL和CUDA C++等数据集。

**📈 对比分析**

与现有的跟踪器相比，Warpline的中位跟踪开销为1.2%，保留了99.9%的原始指令，而其他跟踪器的开销在1.6%到13.2%之间，保留的指令比例为4.6%到32.7%。在10个SOL-ExecBench问题上，使用Warpline的代理在每轮中表现优于裸编码代理和NVIDIA的内核设计代理（KDA），最终实现了7.69倍的几何平均加速。

**⚠️ 局限性**

限制在于当前框架主要针对张量核心内核，未来工作将扩展到其他类型的内核。

---

## 326. LiquidCP: Decoupling Attention Tasks from Communication and Kernel Boundaries in Context Parallelism

**arXiv ID:** 2611.04401 | [PDF](https://arxiv.org/pdf/2611.04401v1)

**作者:** Hongyu Chen `[一作]` (Shanghai Jiao Tong University), Shixuan Sun `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 LiquidCP，统一训练和推理的上下文并行执行框架，包含 MegaRing（训练/全预填充）与 MegaDCP（解码/混合批次）两种设计，并实现 GPU 端共享执行引擎，解耦注意力任务与通信与内核边界。

**💡 创新点**

创新点：1) 统一 Transfer‑Compute‑Reduction 任务模型，提供对所有上下文并行工作流的通用抽象；2) BuddyRing 结构化组对齐与动态任务划分，平衡 token 与 attention 负载并限制短序列碎片化；3) MegaRing 的 ready‑KV 合并与自适应 Compute 任务分配，降低 KV 传输与 Partial‑State 合并开销；4) MegaDCP 的 critical‑wave KV 分割与依赖驱动调度，兼顾 attention 与后续 reduction 的完成时间；5) GPU‑驻留的持久化内核与 CTA 角色切换，实现零开销的跨阶段执行。

**🔧 技术方法**

技术手段：分层任务调度、KV 细粒度分块、BuddyRing 组对齐、NVLink 直接传输、TMA reduce‑add、FlashAttention‑3（FA3）、NCCL、CUDA Graph 与 eager 执行对比、持久化 kernel 与 CTA 角色切换。

**📊 数据集**

使用的数据集与工作负载：训练时采样 ProLong、ArXiv、FreeLaw、GitHub、Pile‑CC，推理时基于 Mooncake trace 与 vLLM/SGLang，混合批次结合预填充与单词推理。

**📈 对比分析**

对比方法：与 Ring、All‑Gather、Zeppelin、Megatron Hybrid 等 CP 基线，以及 vLLM 与 SGLang 的 DCP 路径（包括 A2A、AG‑RS、AG‑AR）进行 CUDA Graph 与 eager 执行对比。性能：在 8 H100 GPU 上，MegaRing 在训练的核心注意力吞吐量上取得 1.23× 前向、1.18× 反向平均加速；MegaDCP 在解码单独批次上提升 1.20×，混合批次提升 1.35×；在 vLLM 服务器场景中，TTFT 平均提升 1.22×，TBT 平均提升 1.04–1.10×。

**⚠️ 局限性**

限制：目前实现仅在 8 H100 GPU 上验证；后向合并仍以单 KV 为粒度，无法进一步降低 register 压力；极短序列的分块开销仍显著；跨阶段并行度依赖于固定的 CTA 角色分配，动态负载不易自适应；实现复杂度高，易于迁移与扩展到更大规模 GPU 集群仍需进一步研究。

---

## 327. Disposable Implementations: What Is Software Engineering For?

**arXiv ID:** 2611.04383 | [PDF](https://arxiv.org/pdf/2611.04383v1)

**作者:** Christoph Treude `[一作]` `[通讯]` (Singapore Management University), Christoph Treude (Singapore Management University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过文献综述与案例讨论，探讨利用编码代理（AI agents）生成新实现的再生（regeneration）过程，并分析其对软件系统可持续性、知识保存与维护成本的影响。

**💡 创新点**

创新点在于提出“实现可丢弃性”（implementation disposability）的概念，重新审视软件生命周期中的实现连续性与系统连续性关系，并为未来研究提供评估框架和关键问题。

**🔧 技术方法**

主要技术包括：编码代理（AI代码生成）、模型驱动工程、自动化控制结构（loop engineering）等，结合需求、测试、架构约束和运行记录等文档化资产进行实现生成。

**📊 数据集**

未使用传统数据集；引用的案例包括：一次将86个Go微服务迁移为TypeScript单体（由代理生成），以及一个239行模块被Claude Opus 5.5三次重建的实验，后者使用了约805行的记录与评估代码。

**📈 对比分析**

本文未开展实验对比，因而未给出性能指标；其主要贡献是提出评估再生安全性、成本效益和知识保存的研究方向与方法论。

**⚠️ 局限性**

限制包括：缺乏系统化的实证验证再生是否真正更安全、更经济；对需要保留的知识类型与持续性判定缺乏明确准则；目前的案例规模有限，难以推广至更大、复杂的系统。

---

## 328. CORA: Teacher-Guided Causal O-RAN Adaptation for Jamming Detection

**arXiv ID:** 2611.04330 | [PDF](https://arxiv.org/pdf/2611.04330v1)

**作者:** Sneihil Gopal `[一作]` (National Institute of Standards and Technology), Richard A. Rouil `[通讯]` (National Institute of Standards and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了CORA框架，通过无标签历史日志中的非因果检测器教师引导强化学习，训练出可在无教师、无标签环境下自适应的因果jamming检测器。

**💡 创新点**

创新点在于将无标签的回顾性检测器的“先见之明”转移为强化学习奖励，使因果检测器在部署时可动态调整阈值、持续时间与宽限期，以提高召回率并降低误报。

**🔧 技术方法**

主要技术包括：基于物理统计的因果kpi检测器、教师引导的奖励设计、掩码动作空间的增量控制器、以及PPO和DQN两种RL学习器。

**📊 数据集**

使用了十个5G O‑RAN测试平台配置的真实kpi日志，总时长34.7小时，涵盖不同噪声、爆发持续时间与功率的混合场景。

**📈 对比分析**

与固定默认、历史最佳固定、随机掩码以及因果SEMCS等基线相比，CORA在10折留一法下平均F2得分提升至0.742（PPO）或0.732（DQN），召回率提升至0.771/0.738，误报率相对可接受。

**⚠️ 局限性**

局限性包括：需训练日志覆盖部署环境才能实现提升；对教师质量的依赖导致分布漂移时性能下降；在测量丢失占主导的场景下，固定默认仍是更稳健的选择。

---

## 329. Averaging Collusion Breaks Traceability in Personalized Neural Audio Watermarking

**arXiv ID:** 2611.04400 | [PDF](https://arxiv.org/pdf/2611.04400v1)

**作者:** Yangming Liu `[一作]` (Tsinghua University), Zhiyong Wu `[通讯]` (Tsinghua University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

评估了多份个性化音频水印复制的平均攻击对追踪性的影响，证明仅用两份复制即可导致 87.7%–99.7% 的追踪失败。

**💡 创新点**

首次系统性研究了神经音频水印在多复制聚合下的破解机制，并提出了基于 payload 的加权混合方法实现针对性误归属。

**🔧 技术方法**

使用均匀样本平均、Target‑Bit Margin 混合权重优化、以及基于解码器置信度的拒绝屏蔽等技术。

**📊 数据集**

采用 300 条 10 秒音频（100 位说话人，每人 3 条）来自 AISHELL‑3 与 LibriSpeech 数据集。

**📈 对比分析**

通过对五个主流系统（AudioSeal、WavMark、TimbreWM、VoiceMark、WMCodec）在 K=2,3,5,8 的平均攻击下计算追踪失败率、PESQ/STOI 质量分数，显示所有系统几乎都在 96% 以上的追踪失败率，且在 Target‑Bit Margin 攻击下部分系统可达到 92.3% 的误归属成功率。

**⚠️ 局限性**

研究仅覆盖了线性平均与少数混合策略，未探讨更复杂的混合模型或多时延/失真场景；防御方案仅为初步置信度屏蔽，尚未系统化或泛化。

---

## 330. Do Multi-Turn Conversational Capabilities Carry Over from Text to Speech? A Parallel Multilingual Benchmark

**arXiv ID:** 2611.04390 | [PDF](https://arxiv.org/pdf/2611.04390v1)

**作者:** Jiaming Zhou `[一作]` (Nankai University), Yong Qin `[通讯]` (Nankai University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在Omnilingua-AudioMC基准上评估并比较多语言、多模态（转录与语音）下的持续对话能力，揭示转录成功不一定转移到语音；

**💡 创新点**

提出严格对齐的多语言、多模态基准与2×2转录干预实验，首次系统分析语音输入对记忆、指令保留等对话需求的影响；

**🔧 技术方法**

使用固定上下文评估协议、文本转语音（TTS）与自动语音识别（ASR）技术、局部转录替换干预以及 Bootstrap 统计检验；

**📊 数据集**

基于原始Audio MultiChallenge对话，扩展为354段对话、1,404项评估标准，覆盖英语与九种目标语言（共十种语言）；

**📈 对比分析**

通过对比参考转录与直接语音、以及不同转录干预，发现模型与语言间的性能差异显著：Muse‑Spark从转录高分跌至语音低分，Qwen3.8在语音上保持高分；转录干预对记忆缺失严重的模型效果更好；

**⚠️ 局限性**

局限包括：受限于合成语音的质量与多样性、未涵盖真实用户噪声与口音、多模态交互之外的视觉/文本混合场景、以及评估仅聚焦固定对话上下文，未探索开放式对话的自适应能力。

---

## 331. Twice as Fast, Until It Isn't: Quantized Serving of Mid-Size Open LLMs

**arXiv ID:** 2611.04380 | [PDF](https://arxiv.org/pdf/2611.04380v1)

**作者:** Smit Kadvani `[一作]` (Independent researcher), Spurthi Tallam `[通讯]` (Independent researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

收集整理公开的9B–20B大模型在vLLM和SGLang中的量化方案（INT4、FP8等）速度与准确性测量，并通过统一的内存对计算规则解释批量大小对性能的影响，揭示大批量时INT4失效、FP8仍有优势；同时指出多数准确度差异低于基准噪声，提出七项量化部署报告清单。

**💡 创新点**

创新点在于：①用单一“内存vs计算”规则解释多源测量结果；②揭示多数公开准确度差异仅为基准噪声；③提出完整的七项报告清单，为未来量化部署实验提供规范。

**🔧 技术方法**

使用了vLLM、SGLang等推理引擎的性能计时；采用多种量化算法（AWQ、GPTQ、FP8、INT4、MXFP4、NVFP4等）；基于roofline模型进行速度机制分析；使用lm‑eval、lm‑eval等工具评估准确度。

**📊 数据集**

评测数据集包括 HumanEval、MMLU（完整与哲学子集）、GSM8K、C‑Eval，采用标准lm‑eval、lm‑eval harness进行量化模型性能比较。

**📈 对比分析**

通过在同一引擎、GPU、基准BF16条件下对比量化后的速度提升与准确度变动来评估模型：在单token批量下INT4可提升1.35–2.12×速度，批量256时几乎无加速；FP8在大批量时仍保持加速；准确度差异多在1–4 MMLU点以内，部分大幅差异仅为基准噪声。

**⚠️ 局限性**

局限性包括：实验数据来自不同来源、配置不统一；缺乏标准误差和多次重复；部分量化方案未在同一批次下同时测量速度和准确度；未覆盖AMD、Gaudi等平台的FP4等方案；报告清单未被完整满足。

---

## 332. LLM-Generated Investment Research Memos Need Validation

**arXiv ID:** 2611.04375 | [PDF](https://arxiv.org/pdf/2611.04375v1)

**作者:** Xukang Gong `[一作]` (New York University), Xiao-Yang Liu `[通讯]` (Columbia University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对四种LLM代理生成的1440份投资研究备忘录进行源边界验证，评估每条财务声明的真伪，并计算每条声明与整份备忘录的错误曝光率。

**💡 创新点**

提出了基于源边界的验证流程和“备忘录污染率”指标，揭示低声明级错误率仍可能导致大量备忘录包含可被纠正的错误。

**🔧 技术方法**

使用GPT‑5.5、Claude Sonnet 4.6、DeepSeek‑V4‑Pro和Qwen3.8‑Flash四大LLM生成备忘录，配合自动化声明提取、规则匹配的证据检索和人工审核。

**📊 数据集**

基于30家上市公司在2023Q1–2025Q4共360个公司‑季度案例的官方文件（10‑Q/10‑K、收益发布等）构建的源文件包。

**📈 对比分析**

通过比较声明级被驳斥率与备忘录级污染率，发现声明级错误率低于2%但备忘录污染率高达37‑41%，并进一步对比四个模型在错误类型、证据操作和错误严重度方面的差异。

**⚠️ 局限性**

评估仅在选定的官方文件范围内进行，无法覆盖所有外部信息，存在时序泄漏、规则匹配失误及仅针对四个LLM和单一领域的局限性。

---

## 333. Distributed Marginal Return: A Communication-Efficient Algorithm for Multi-Agent Weapon Target Assignment

**arXiv ID:** 2611.04370 | [PDF](https://arxiv.org/pdf/2611.04370v1)

**作者:** Kamal Mammadov `[一作]`, Damith C. Ranasinghe `[通讯]` (Adelaide University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种分布式武器-目标分配算法——分布式边际回报（Distributed Marginal Return, DMR），能够在有限通信与局部感知下实现多武器多目标的协同分配。

**💡 创新点**

创新点：①将边际回报概念推广到多武器-目标情景，证明非冲突的边际回报可叠加；②设计了两阶段（拍卖-共识）协议，允许多个冲突-free投标并行接受，显著减少通信轮数；③在保证全局一致性的前提下实现完全去中心化，兼顾安全与效率。

**🔧 技术方法**

技术：分布式拍卖-共识框架、边际回报计算、冲突检测与剔除、基于多跳网络的消息广播、递归证明非冲突边际回报可加的理论。

**📊 数据集**

数据集：使用大规模蒙特卡罗随机生成的武器-目标实例（目标数 1–30，代理数 1–15，武器数 1–10），并对小规模实例采用全枚举（FE）得到全局最优。

**📈 对比分析**

比较方法：在固定通信轮数（最多 6 轮）下与 CBAA、GRAPE 以及全枚举进行对比，指标为目标成本。实验结果表明 DMR 在第一轮就达到最低成本，且在 1–3 轮内几乎达到全局最优，显著优于 CBAA 与 GRAPE；在不同规模（目标、代理、武器数）下亦保持更低成本与更快收敛。

**⚠️ 局限性**

局限性：①仅适用于静态 WTA 场景，未考虑时间演化与动态目标；②算法复杂度随每个代理武器数平方，若武器量极大可能受限；③假设通信网络足够连通且每轮信息可完整传播，极端干扰环境下性能需进一步验证。

---

## 334. Space-Optimal Algorithms for Spanning Forests in Dynamic Streams at Every Confidence Level: With and Without Deletion Constraints

**arXiv ID:** 2611.04365 | [PDF](https://arxiv.org/pdf/2611.04365v1)

**作者:** Zemin Chao `[一作]` (Harbin Institute of Technology), Hongzhi Wang `[通讯]` (Harbin Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了动态图流中一次性生成生成树的空间复杂性，证明了在高置信度范围内的空间下界，并设计了匹配的空间最优算法。

**💡 创新点**

在高置信度范围内实现了空间节省，达到最优空间复杂度，并且首次在所有置信度水平上建立了最优空间复杂性。

**🔧 技术方法**

使用了一次性随机算法，结合了压缩的历史边缘数据结构和独立验证机制。

**📊 数据集**

使用了具有v个顶点的简单图的动态图流，研究了在报告时缺失的历史边缘数量的限制。

**📈 对比分析**

与现有算法进行比较，证明了在高置信度范围内的空间复杂性为Θ(min{v^2, vlog(v/δ) log^2(2+v/log(v/δ))})比之前的O(v^2)节省了Θ((log v/loglog v)^2)的空间。

**⚠️ 局限性**

在高置信度范围内的空间复杂性仍然未完全解决，尤其是在处理删除和恢复操作时的复杂性。

---

## 335. Mitigating Spatial Blindness in Multiple Instance Learning for Whole-Slide Image Analysis

**arXiv ID:** 2611.04337 | [PDF](https://arxiv.org/pdf/2611.04337v1)

**作者:** Xiangyu Li `[一作]` (Tianjin University), Ran Su `[通讯]` (Tianjin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在多实例学习框架下，先冻结一个基于样本组成的统计预测器，再训练一个图网络残差补偿器，并通过坐标重新分配的对比损失鼓励图网络学习空间邻域信息。

**💡 创新点**

提出了分阶段训练和坐标重排分离损失两项关键设计，解决了空间信息提供后模型未必学习空间关系的“空间盲点”，并通过对照实验验证了该训练策略对空间依赖的显著提升。

**🔧 技术方法**

使用了统计直方图原型编码器、KNN图构造、两层GCN、可分离的残差逻辑与坐标重排对比损失；训练过程采用Adam+Lookahead、cosine调度、dropout、hard‑instance mining 等常见技巧。

**📊 数据集**

在九个公开 WSI 分类任务（TCGA‑NSCLC、TCGA‑BRCA、BRACS、PANDA 等）和五个生存预测任务（KIRC、KIRP、LUAD、STAD、UCEC）上进行评估，并用合成 Spatial‑MNIST-Bag 做受控实验。

**📈 对比分析**

与 AB‑MIL、CLAM‑SB、DS‑MIL、TransMIL、ILRA‑MIL、MHIM‑MIL、DGR‑MIL、2DMambaMIL 等方法对比，ResTopoMIL 在大部分任务上取得最高或相近的 AUC/F1/ C‑index，且在四个主要指标上在多重检验校正后仍保持显著提升。

**⚠️ 局限性**

局限性包括：① 统计显著性仅在部分指标上保留；② 未报告推理时的计算开销与内存消耗；③ 仅在静态 WSI 里验证，缺乏前瞻性或多中心验证；④ 依赖预训练特征，未探讨与编码器联合训练的潜在提升。

---

## 336. ATC: Agent Trajectory Control for Role-Contract Enforcement in Black-Box Multi-Agent Systems

**arXiv ID:** 2611.04321 | [PDF](https://arxiv.org/pdf/2611.04321v1)

**作者:** Maisha Maliha `[一作]` (University of Oklahoma), Sumit Kumar Jha `[通讯]` (University of Florida)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Agent Trajectory Control（ATC），在多代理 LLM 系统中在消息边界检测并纠正角色漂移，确保每个代理仅执行其授权功能。

**💡 创新点**

将角色权限转化为可执行的合同，并在每条消息进入共享对话前对其进行几何、泄漏、语言和逻辑四个维度的评分与修正，实现消息级别的角色合规检查与自动修复。

**🔧 技术方法**

功能行为向量编码、角色子空间投影、交叉角色泄漏惩罚、语言行为向量、逻辑合同符号检测、闭式泄漏正则化投影以及约束再生成等技术。

**📊 数据集**

EntCollabBench、MultiAgentBench、SWE‑bench Verified 三大多代理基准。

**📈 对比分析**

与 Prompt‑Only、Struct‑Role、A‑MEM、CAA、SEA、SOP‑Role 等七种基线对比，ATC 在所有基准和团队规模（Hom‑10、Het‑10、Het‑12）下均实现最高任务得分、最高角色一致性、最低角色漂移能量，尤其在异构大团队上提升显著。

**⚠️ 局限性**

依赖角色表示与合同质量；黑盒部署只能通过再生成，无法直接干预生成器；无法解决事实错误、偏见等问题；对流动或重叠角色适用性有限；计算成本相对较高，需与其他安全保障结合使用。

---

## 337. TEMPO: Time Series Understanding via Discrete Tokenization

**arXiv ID:** 2611.04482 | [PDF](https://arxiv.org/pdf/2611.04482v1)

**作者:** Riccardo Maggioni `[一作]` `[通讯]` (Forgis), Riccardo Maggioni (Forgis)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 TEMPO 方法，将时间序列通过全局上下文的 Transformer 编码器与有限标量量化（FSQ）离散化为 625 码，然后将这些码直接扩展到预训练 LLM 的词表，实现时间序列与文本在同一自注意力层中的统一处理，并通过两阶段适配（embedding 对齐 + DoRA 调整）完成多任务微调。

**💡 创新点**

创新点：① 采用全局上下文的 Transformer + FSQ 生成可覆盖全部时间窗的离散词汇；② 直接将离散码加入 LLM 词表，无需额外交叉注意力；③ 通过文本形式的均值/方差统计保留幅度信息；④ 两阶段冻结/微调策略使 tokenizer 与基模型保持不变，降低训练成本。

**🔧 技术方法**

技术手段：Transformer encoder + FSQ 量化、离散词表扩展、DoRA 适配器、冻结的预训练 tokenizer 与 Qwen3‑4B LLM、任务特定微调、宏 F1 / 准确率评估、基准对比。

**📊 数据集**

数据集：活动识别（HAR）、睡眠分期、轴承故障诊断（CWRU）、Time‑MQA、UCR、UEA、ECG、金融、预测、合成信号等（约 500K 窗口用于 tokenizer 预训练）。

**📈 对比分析**

与专用分类器（RF、WDCNN）、GPT‑4o、LLaMA3.1‑8B、Llama3.2‑3B（Soft Prompt、Flamingo）以及 OpenTSLM 进行对比。TEMPO 在 HAR 89.0%、睡眠 85.0%、轴承 86.0%（macro‑F1）上远超 LLM 方案但略逊于专用模型；在 TSAQA 分类 80.5% 与 91.3% 的基线相比处于中游；在 1.7B TEMPO 对 Time‑MQA 的 62% 分类成绩优于 52% 的 Qwen‑2.5‑7B，但在判定与异常检测上略低。

**⚠️ 局限性**

局限性：量化导致时间细节与压缩之间的折衷；每通道单独标记化导致序列长度随通道数线性增长；对长序列的细粒度保留有限；在某些专用任务上仍不及专门设计的模型；需要更高效的通道压缩与跨域适配。

---

## 338. Goal-Driven Autonomous Optical Experimentation via a Self-Recursive Hardware-Software Harness

**arXiv ID:** 2611.04481 | [PDF](https://arxiv.org/pdf/2611.04481v1)

**作者:** Tianxing Zhu `[一作]` (Konica Minolta (China) Investment Ltd.), Malte Hillebrand `[通讯]` (Konica Minolta (China) Investment Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建并验证了一套目标驱动、自我递归的光学实验平台，利用物理证据不断修正硬件、软件、模型和协议，提升透明显示的颜色匹配性能。

**💡 创新点**

创新点在于将实验设计、执行与推理闭环化，形成可持续自我改进的“自递归”框架，并在此框架内实现命令访问扩展、光谱反馈与响应模型迭代。

**🔧 技术方法**

技术手段包括：iPad 背景生成应用、RayNeo X3 AR 眼镜渲染与 USB/ADB 命令控制、TOP 300 光学探头与 CAS140D 光谱仪的实时测量接口、基于 CIELAB 的颜色差异评估以及三种响应模型（XYZ、CIELAB 线性、CIELAB 带二次项）比较。

**📊 数据集**

使用了 2,170 条刺激条件与 356 条控制测量，涵盖 320 条 iPad 校准、960 条白/灰背景组合、378 条彩色背景组合、192 条仅显示测量等多种配置。

**📈 对比分析**

通过 ΔE₇₆* 颜色差异量化对比，单一目标从 19.756 降至 1.211（下降 93.9%），五目标集从 9.377 降至 1.882；响应模型 RMSE 从 1.842（XYZ）提升至 1.363（CIELAB 线性）并在后续测量中保持更低误差。

**⚠️ 局限性**

局限性包括：实验仅在单一硬件配置下验证，缺乏跨设备或跨目标的泛化性；对高动态或多光源环境的适应性尚未测试；实现复杂度高，需依赖专业硬件和定制软件。

---

## 339. A Parametric Linear Conic Projection for Structured Sparse Attention

**arXiv ID:** 2611.04479 | [PDF](https://arxiv.org/pdf/2611.04479v1)

**作者:** Michel Barlaud `[一作]` (Universite Cote D'Azur), Benoit Legat `[通讯]` (Uclouvain)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于 Lorentz 锥几何的尺度不变稀疏度指标 CAI，并设计了 Parametric Linear Conic Projection (PLCP) 与 Parametric Euclidean Conic Projection (PECP)，用于对 Transformer 的注意力矩阵进行结构化稀疏化，随后实现了双层投影框架。

**💡 创新点**

创新点包括：① 将 CAI 定义为稀疏度约束并构造对应的锥形几何；② 证明 PLCP 与 PECP 共享相同的活跃集，只相差径向缩放；③ 给出闭式投影公式并实现 O(n) 或 O(n log n) 的高效算法；④ 生成行/列/对角等硬件友好的结构化稀疏模式；⑤ 在高稀疏度下实现显著的算力加速和准确率保持。

**🔧 技术方法**

采用 Lorentz 锥几何、稀疏性度量 CAI、概率分布与 CLT 推导、闭式投影（线性与欧氏）、多层投影、Top‑K/排序实现、以及与 GSP、Group Lasso、BigBird 等基线在 GPU 上的对比实验。

**📊 数据集**

使用合成 Gaussian/Uniform 矩阵以及 GLUE（SST‑2、QNLI、QQP、MNLI、WiC）和 SuperGLUE（WiC）等自然语言处理数据集进行实验。

**📈 对比分析**

与 GSP、结构化剪枝、Group Lasso、BigBird 等方法在计算时间、准确率和稀疏率上进行对比。PLCP 在 GPU 上与 GSP 的速度可比并在高稀疏率（≤90%）下保持准确率下降 ≤2%，在 GLUE 任务上准确率比 BigBird 高 6–10%，与 PECP 的性能相近。

**⚠️ 局限性**

局限性包括：对非对角/行列稀疏结构的适用性有限；阈值更新在某些分布下可能收敛慢；对随机种子敏感性和可解释性未系统化；在极低稀疏率下性能提升有限；尚未在大规模 MoE 或更大模型中验证。

---

## 340. Is Forgetting Unified? Exploring Concept Unlearning in Unified Multimodal Models

**arXiv ID:** 2611.04467 | [PDF](https://arxiv.org/pdf/2611.04467v1)

**作者:** Zhongkai Xue `[一作]` (Institute of Automation, Chinese Academy of Sciences), Liang Wang `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究统一多模态模型（UMMs）在图像生成与视觉理解两条路径上同时进行概念消除（unlearning）

**💡 创新点**

首次提出跨路径概念消除框架，并设计了专门的诊断基准与一种协调生成侧权重编辑与理解侧低秩更新的联合消除方法

**🔧 技术方法**

采用闭式权重编辑（如TIME、UCE）与LoRA低秩适配器结合，构建多任务损失（身份重定向、视觉实用性、偏好排序、邻近概念保留、KL与反崩溃正则），并通过参数融合实现统一检查点

**📊 数据集**

构建包含30个目标概念与60个相邻概念（共90个实体）的基准，包含720张参考图像，按严格划分的训练/评估拆分，覆盖人物、地标、通用物体三大类别

**📈 对比分析**

在BAGEL‑7B‑MoT与Janus‑Pro‑7B两款7B规模UMMs上，与单侧编辑方法（TIME、UCE、GD、KL）及其直接组合进行对比。我们的联合方法实现了80.0%（BAGEL）/92.7%（Janus）的统一消除率，且在保持生成质量与视觉描述准确性的前提下，统一严格消除率分别达到47.6%和58.0%。

**⚠️ 局限性**

实验仅覆盖两款7B模型与30个概念，使用自动化评测与LLM评判，未涉及更大规模模型或更广泛领域；关注行为消除而非权重级别的正式删除，且对抗攻击下的鲁棒性与多概念扩展仍有限。

---

## 341. From Watching to Steering: Enabling Human-Coding Agent Collaboration through Execution Traces

**arXiv ID:** 2611.04466 | [PDF](https://arxiv.org/pdf/2611.04466v1)

**作者:** Xiaoke Yan `[一作]` (Tongji University), Nan Cao `[通讯]` (Tongji University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种交互式执行轨迹可视化与干预系统，使用户能够在编码代理执行过程中观察、理解、干预并持续改进代理的行为；

**💡 创新点**

创新点在于将执行轨迹从单纯记录转变为可编辑的控制面板，支持局部重定向与在线的轨迹到技能（trace‑to‑skill）蒸馏，从而实现人机协作的 Observe‑Understand‑Intervene‑Evolve 循环；

**🔧 技术方法**

技术主要包括：基于 OpenCode 的执行轨迹抽取与结构化、层级化可视化与交互（节点过滤、跨视图链接、流式实时更新）、轨迹级干预（注释、分叉、结果校正）以及基于 LLM 的三阶段技能蒸馏（估计、提炼、验证与更新技能池）；

**📊 数据集**

使用了 SWE‑bench Verified（真实 GitHub 问题集）进行基准测试，并在实验室对 20 名有经验用户进行对照实验，另外 8 名专家用户进行案例研究与访谈；

**📈 对比分析**

与无技能池（Base）、仅使用执行轨迹蒸馏（Trace‑only）以及结合人工反馈蒸馏（Trace‑fb）进行比较，结果显示 Trace‑fb 在通过率（68%↑3%）、输入/输出 token、成本、时间等七项指标上均优于基线，整体效用提升约 14%；

**⚠️ 局限性**

局限性包括：依赖 LLM 的估计与蒸馏准确性受模型偏差影响，技能蒸馏过程较为耗时且对任务相似性有一定要求，系统在首次安装时配置较为繁琐，并且对非常规错误或极端场景的解释与干预支持仍有限。

---

## 342. Theoretical Analysis of Muon on Transformers for Next-Token Prediction

**arXiv ID:** 2611.04465 | [PDF](https://arxiv.org/pdf/2611.04465v1)

**作者:** Shengwei Wu `[一作]` (Zhejiang University), Song Li `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了 Muon 优化器在 Transformer 中对下一词预测任务的训练动态，并给出了两阶段训练的理论分析。

**💡 创新点**

创新点在于提供 Muon 在软最大化注意力下的显式优化轨迹、非渐进收敛界以及对未见上下文预测的泛化下界。

**🔧 技术方法**

使用了 Muon（矩阵极化更新）、软最大化注意力、矩阵范数（谱范数、核范数）以及非凸优化的理论工具。

**📊 数据集**

使用了基于正交词嵌入的合成数据集，包含 64 个正交词向量和 1024 个训练上下文。

**📈 对比分析**

通过与 Adam 的对比实验，验证了 Muon 在谱范数最大间距收敛速度更快，且在不同步长衰减策略下表现更佳。

**⚠️ 局限性**

局限在于仅考虑单层、单头 Transformer、全局正交词嵌入、完整的梯度计算，未涉及多头、深层模型、随机梯度或真实数据。

---

## 343. Using GeoAI and machine learning tools for consistent high-resolution land cover mapping based on time-series NAIP imagery

**arXiv ID:** 2611.04470 | [PDF](https://arxiv.org/pdf/2611.04470v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 344. How Agent Skills Evolve: A Large-Scale Empirical Study of Skills on GitHub

**arXiv ID:** 2611.04456 | [PDF](https://arxiv.org/pdf/2611.04456v1)

**作者:** Jinjun Huang `[一作]` (Zhejiang University), Zhongxin Liu `[通讯]` (Zhejiang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究系统地收集并分析了84,200个GitHub开放源代码仓库中的Agent技能及其750,898次相关提交，构建了大规模演化数据集，并从组件、变更模式和动机三个维度对技能演化进行细粒度定量与定性研究。

**💡 创新点**

创新点在于①首次从长期提交历史角度揭示Agent技能的演化规律与动机；②设计并验证了基于LLM的自动化注释管线，实现了组件、变更模式和动机的高效标注；③从实证结果提出了若干针对技能可靠性、自动更新、上下文效率、检索优化和安全回归测试的研究方向。

**🔧 技术方法**

技术手段包括GitHub Archive数据爬取、正则与启发式筛选构建技能文件、基于OpenAI GPT‑4的批量组件与变更注释、统计与可视化分析（Python、Pandas、Matplotlib）以及安全扫描工具（Cisco Skill Scanner、Snyk Agent Scan）对比实验。

**📊 数据集**

使用的数据集为自2015年9月1日至2026年4月23日期间公开的GitHub仓库中筛选的84,200个Agent技能及其750,898次提交，已公开托管于https://github.com/ZJU-CTAG/SkillEMP。

**📈 对比分析**

与现有研究相比，本工作提供了更大规模、更细粒度的演化视角；实验结果表明自动注释准确率分别为组件0.881、变更模式0.816、动机0.840，验证了方法的可行性；安全扫描实验显示安全导向提交仅有约13%的风险降低，揭示了当前安全改进效果有限。

**⚠️ 局限性**

主要局限包括①数据仅来自公开GitHub仓库，可能不代表私有或企业环境；②自动化注释仍受LLM推断误差影响，尽管准确率较高但仍有错误；③对低频变更、低活跃度仓库的演化规律缺乏足够证据，研究聚焦于活跃度较高的技能。

---

## 345. Near-Zero Token Embeddings Silently Stall Training in Gated Sequence Models

**arXiv ID:** 2611.04449 | [PDF](https://arxiv.org/pdf/2611.04449v1)

**作者:** Xin Liu `[一作]`, Mengyu Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了开放权重门控序列模型中出现的近零嵌入导致梯度放大和全批次梯度被零化的问题，并提出两种低成本解决方案。

**💡 创新点**

发现并阐明了未绑定嵌入权重衰减与零输入门控读取相互作用导致梯度爆炸和裁剪失效的根因，并提出 GroupClip 与 GateFloor 两种补救方法。

**🔧 技术方法**

使用了 Pre‑RMSNorm、SiLU 门控、AdamW 权重衰减、FP32 全局梯度裁剪以及自定义梯度裁剪与门控偏置技术。

**📊 数据集**

审计了 HuggingFace 公开的多种开放权重模型，并在多语言（如特卢古语、孟加拉语、泰语）与模板数据集上进行持续预训练和指令调优实验。

**📈 对比分析**

通过与传统全参数训练、ZClip、GradientStabilizer 等方法对比，GroupClip 在大多数触发批次中将失效率降至 0% 并保持或提升 PPL；GateFloor 在从零预训练中抑制梯度放大且保持近似的困惑度。

**⚠️ 局限性**

主要局限在于从零预训练实验仅覆盖少量规模与种子，极大规模模型的效果尚未验证；门控偏置方法在所有门控结构上并非通用。

---

## 346. DualEvolve: Dual-Level Self-Evolution of Mechanism and Orchestration for Recommendation

**arXiv ID:** 2611.04439 | [PDF](https://arxiv.org/pdf/2611.04439v1)

**作者:** Tianqi Gao `[一作]` (Independent Researcher), Peng Yan `[通讯]` (Meituan)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出DualEvolve框架，通过双层循环同时演化推荐模型与LLM代理的工作流程，提升推荐系统自适应改进效率。

**💡 创新点**

创新点在于将实验过程质量（决策、实验严谨性、分析深度、创新度）作为反馈，指导外层的工作流程优化，并采用Thompson采样估计不同编辑动作的效用，从而实现递归自我提升。

**🔧 技术方法**

核心技术包括：基于固定评估器和门控规则的机制演化、内部的诊断-提议-编码-分析迭代、外部的过程质量评分器（GLM-5.2）、Thompson采样动作选择、inside/wiki知识管理以及成本感知的推广门。

**📊 数据集**

在四个数据集（Amazon Reviews 2023的CDs&Vinyl、Baby、Software以及MovieLens-1M）以及四种基准推荐模型（NCF、NGCF、SASRec、GRU4Rec）上进行实验。

**📈 对比分析**

与Base、OpenEvolve、DeepEvolve比较，DualEvolve在64个设置-指标对上均取得最高测试均值，宏平均相对提升分别为HR@5 7.6%、NDCG@5 9.0%，搜索时间和LLM token使用量均低于DeepEvolve。

**⚠️ 局限性**

局限性包括：评估器与门控规则固定，可能限制跨任务的泛化；对外部知识快照的依赖使得模型在完全无知识环境下的表现未知；并且实验成本仍较高，尤其在更大规模数据集与模型上验证需进一步探索。

---

## 347. A Security Analysis of Verifiable Credential Verification Pipelines: Attack Surface Arising from JSON-LD/RDFC Canonicalization Compared with SD-JWT VC and mdoc

**arXiv ID:** 2611.04461 | [PDF](https://arxiv.org/pdf/2611.04461v1)

**作者:** Naohiro Fujie `[一作]` (MIRAI Design Laboratory), Shigeya Suzuki `[通讯]` (Global Research Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究在实现层面对 SD‑JWT VC、W3C VCDM+Data Integrity 与 mdoc 三种 VC 验证管道进行安全性评估，量化了 JSON‑LD/RDFC canonicalization 产生的攻击面（Poison graph DoS、上下文注入、SSRF 等）以及验证过程的性能影响。

**💡 创新点**

创新点在于首次将理论上已知的 RDFC‑1.0 复杂度威胁与实际实现相结合，系统地实验验证了默认配置下的 DoS 与 SSRF 风险，并评估了静态文档加载、上下文白名单与递归调用限制等可行缓解措施。

**🔧 技术方法**

采用 Node.js/Go/Python 版 jsonld、rdf‑canonize、jose、cbor‑x 等库实现各格式的签名与验证；通过手工构造 Poison graph、远程 @context 注入、记录器捕获 SSRF 请求等技术进行攻击实验；使用 call‑limit、计时测量等手段评估性能瓶颈。

**📊 数据集**

实验使用 527‑字节 Poison graph、OB3 Open Badges v3.0、DCC 版学位凭证以及包含 10/50 空白节点的合成凭证等样本，以对比不同规模与结构的 JSON‑LD 证书在 canonicalization 过程中的开销。

**📈 对比分析**

在四个硬件环境（云 VM、裸机、Apple M1、旧 Xeon）下多次重复测量；以 baseline 证书的 canonicalization 时间为参照，设定 100× 处理时间阈值定义 DoS；记录攻击输入的耗时（如 278×、13.5 s）与合法输入的 call‑limit 成功率，展示不同缓解措施对吞吐量的影响。

**⚠️ 局限性**

局限性包括：实验结果高度依赖实现库的默认配置（如 call‑limit 未启用时易出现漏洞）；仅评估了三种格式的验证管道，未涵盖键解析、撤销等完整链路；攻击样本为手工构造，未覆盖所有可能的恶意图；性能阈值受机器速度与网络延迟影响，缺乏通用标准。

---

## 348. Binary Huffman Coding with Unequal Letter Costs Is NP-Complete

**arXiv ID:** 2611.04444 | [PDF](https://arxiv.org/pdf/2611.04444v1)

**作者:** L. Sunil Chandran `[一作]` (Indian Institute of Science), Rishikesh Gajjala `[通讯]` (Pramaana Labs)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文证明了当字母成本 (M,M+1) 随输入而变化时，二进制前缀编码问题在判定形式上是 NP‑完整的，从而推导出最优编码寻找问题是 NP‑难的。

**💡 创新点**

创新点在于构造了两阶段多项式时间归约：先将平面图的独立集问题转化为带有符号系数的叶部份问题，再将该叶部份问题映射为带有具体频率和字母成本的前缀编码实例；该归约显式使用了树的坐标轮廓、区间约束和权重系数的层级优先级，首次展示了字母成本作为输入时的难度。

**🔧 技术方法**

主要技术包括：树形坐标表示与叶部份表述；设计带符号系数的“最小分数树”目标；利用区间约束和局部树构造模板实现指令块；采用二进制尺度的频率增量来强制符号深度；权重优先级的指数加权以确保违反约束必产生大成本。

**📊 数据集**

论文不涉及实验或数据集；所有证明均为理论构造与归约。

**📈 对比分析**

对比方法为复杂度分析：论文通过两步归约证明了 NP‑完整性，而非实验性能；因此不存在直接的实验性能比较。

**⚠️ 局限性**

局限性在于：归约仅展示了普通 NP‑难，未证明强 NP‑难或存在多项式时间近似（FPTAS）; 对于字母成本仅为多项式大于 1 的情况仍未解决；归约中的数值规模可能指数级，无法保证伪多项式时间。

---

## 349. A Systematic Point Cloud Edge Detection Framework for Automatic Aircraft Skin Milling

**arXiv ID:** 2611.04435 | [PDF](https://arxiv.org/pdf/2611.04435v1)

**作者:** Zijie Wu `[一作]`, Ajmal Mian `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于三维点云的自动航空机身外壳铣削边缘检测框架

**💡 创新点**

创新点：①使用空间切线连续性（STC）计算点云中边缘点的概率；②设计分层分支搜索（HBS）方法，先构建粗边缘骨干，再通过线性度交叉验证扩展边缘，最后利用图优化和最短路径剔除误检，实现高精度、鲁棒的边缘提取；③将概率检测、分支搜索与图优化耦合，首次满足机身铣削对±0.5 mm精度的需求

**🔧 技术方法**

技术：空间切线连续性（STC）概率检测；分层分支搜索（HBS）——骨干生成、分支扩展、分支修剪；线性度交叉验证；基于邻域图的图优化（最小生成树、最短路径）; K‑d树、DBSCAN、最小生成树等标准算法

**📊 数据集**

数据集：①合成三维模型（Model 1–3，包含开放与闭合边缘、复杂曲面）；②实际航空机身外壳工件点云，使用结构光相机配合机器人平台扫描得到；③在实验中还使用了非标准机身皮肤、刀尖轮廓、孔径测量和装配间隙等场景点云

**📈 对比分析**

比较方法：与两种基线算法对比——ED（基于特征值变化的边缘检测）和MS（均值漂移算法）；评价指标为精度、召回率、F1‑score、误差（mm）。HBS在所有指标上均显著优于基线，误差仅约0.01 mm，满足工业铣削的精度要求，并在噪声、离群点和非均匀采样条件下保持鲁棒性

**⚠️ 局限性**

局限性：①未针对特征线单独检测，细节测量受限；②在点云稀疏时需增大邻域搜索半径，导致部分关键点被忽略；③固定感知半径，未采用自适应尺度结构，可能在多尺度表面上影响精度；④对非常复杂或高噪声场景的适用性尚需进一步验证

---

## 350. Typological Feature Induction as a Benchmark for Inductive Metalinguistic Reasoning

**arXiv ID:** 2611.04421 | [PDF](https://arxiv.org/pdf/2611.04421v1)

**作者:** Chen Zhang `[一作]` (Nii Llmc), Yusuke Miyao `[通讯]` (Nii Llmc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出典型特征诱导(TFI)任务，使用 Grambank 195 种语言的多平行语料评估 LLM 的归纳式语言学推理；

**💡 创新点**

创新点在于将归纳推理与应用推理分离，通过加密、语言识别实验和基准比较，揭示模型在真正归纳推理上的不足；

**🔧 技术方法**

采用 Gemma4、Qwen3.6、GPT‑5.6 Luna 等 LLM，在 plain、mono、misaligned、cipher‑parallel 等多种输入设置下进行实验，并使用字符替换加密技术；

**📊 数据集**

使用 Grambank 语法特征库与 FLORES+、BOUQuET 多平行语料，覆盖 195 种语言；

**📈 对比分析**

与 name‑only、majority、random 基线对比，发现未加密时模型受记忆影响，性能在核心 22 个特征上约 60‑70%，但加密后无法超过 majority 基线，显示归纳能力有限；

**⚠️ 局限性**

局限在于数据主要为书面译文，缺乏真实场景语料；加密方法未能完全消除语言识别；未针对可诱导特征评估精度与召回率。

---

## 351. VidUEU-Agent: A Data-Curation Agent for Multimodal Understanding, Editing, and Unified Tasks

**arXiv ID:** 2611.04418 | [PDF](https://arxiv.org/pdf/2611.04418v1)

**作者:** Tengju Ru `[一作]` (Zhejiang University), Bo Dai `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了基于 LLM 代理的自动化管线 VidUEU‑Agent，将原始视频转换为多模态指令数据，并生成了 VidUEU‑100K 数据集。

**💡 创新点**

首次将四阶段全球感知、关键帧挖掘、元数据构造和指令生成与自适应数据迭代相结合，实现了从视频到理解、编辑和统一任务的端到端指令数据生成。

**🔧 技术方法**

采用 LLM 代理（DeepSeek‑V3）、VLM（Qwen3.5‑9B）、工具规划与调用、门控过滤、语义关键帧选择、元数据质量审查以及基于训练反馈的自迭代优化等技术。

**📊 数据集**

采集 8,268 条来自 OpenVid‑1M、Panda‑70M、Wikimedia、YouTube、FineGym、BDD100K、CodeVidCL 等公开视频，生成 100K 指令样本。

**📈 对比分析**

在 MMT‑Bench、MME、LLaVA‑Bench、MMDU、GEdit‑Bench‑En、WEAVEbench 等基准上 fine‑tune InternVL3‑8B、Bagel‑7B‑MoT 等模型，显著提升多模态理解、编辑与统一任务的指标，效果优于现有基线。

**⚠️ 局限性**

受限于视频噪声、关键帧选择偏差、长链交互错误传播等，且对开创性生成任务的提升有限。

---

## 352. Efficient and Faithful High-Resolution Image Editing with Sparse Cross-Resolution Denoising

**arXiv ID:** 2611.04416 | [PDF](https://arxiv.org/pdf/2611.04416v1)

**作者:** Maosen Zhao `[一作]` (Shanghai Innovation Institute), Tao Chen `[通讯]` (Shanghai Innovation Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在 1K 编辑轨迹与高分辨率细节生成之间通过稀疏跨分辨率去噪协同的原生高分辨率图像编辑器 FLASH。

**💡 创新点**

创新点包括：① 通过冻结 1K 分支作为全局编辑锚点，局部 Halo 关注机制实现仅在局部邻域内进行高分辨率细节去噪；② 采用分辨率依赖的时间步映射和 YaRN 位置适配，使预训练 1K 模型直接迁移到 2K/4K；③ 引入无训练的渐进推理路径，在前期使用 1K 计算，后期仅在高分辨率完成细化，从而显著降低推理延迟。

**🔧 技术方法**

技术手段包括：跨分辨率 Transformer、低秩 LoRA 调优、分辨率特定的噪声映射、RoPE 位置适配、局部 Halo 注意力、RGB/ VAE 级别的高分辨率提升以及流匹配训练目标。

**📊 数据集**

使用 FLUX.2 Klein Base 4B 作为基础模型，在 VINS-120K 训练集上微调，评估数据集为 VINS-4KEval（509 条指令，12 种编辑类别），并对 2K/4K 输出进行源图像保真度评估。

**📈 对比分析**

与 FaithDiff、DiT4SR、PiSA‑SR、TSD‑SR 以及训练自由 VINS 等方法进行对比；FLASH 在 2K/4K 结构保真度（HaarPSI、mPSNR、mSSIM）上均领先，且在 4K 上的推理时间比密集目标‑参考注意力快 4.2×、注意力成本快 25.3×；渐进推理进一步将 4K 延迟降低 30.5%，仅牺牲 0.029 HaarPSI。

**⚠️ 局限性**

局限性包括：① 对不同分辨率的支持需要手动调节 Halo 范围和时间步映射；② 仅在 1K 轨迹上学习全局编辑，可能在极端结构变换下失去细节控制；③ 渐进推理牺牲部分高分辨率细节，特别是在需要大量后期细化的场景中。

---

## 353. DR-LiteNet: A Lightweight Explainable Hybrid CNN Framework for Imbalanced Diabetic Retinopathy Grading via Adaptive SMOTE Fusion

**arXiv ID:** 2611.04405 | [PDF](https://arxiv.org/pdf/2611.04405v1)

**作者:** Nishi Kanta Paul `[一作]` (NIMISHES Lab), Israt Jahan Esha `[通讯]` (Brac University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种轻量级混合CNN框架DR‑LiteNet，用于解决糖尿病视网膜病变（DR）分级中的严重类别不平衡和高计算成本问题。

**💡 创新点**

创新点在于将EfficientNetB0的深度特征与51维手工提取的病变描述子进行级联融合，并在融合特征空间应用Adaptive SMOTE（ADASYN）进行类别级特征级合成，以提升少数类表现，同时保持模型参数低于5M；另外引入Grad‑CAM可解释性。

**🔧 技术方法**

采用的技术包括EfficientNetB0 + GAP（深度特征）、手工特征（血管密度、糜烂强度、纹理熵、HSV直方图）、特征级ADASYN、两阶段训练（冻结/解冻）、交叉熵损失、Grad‑CAM、CLAHE预处理、数据增强（旋转、翻转、颜色抖动）。

**📊 数据集**

使用了APTOS 2019公共数据集（3662张彩色眼底图，5级分级）。

**📈 对比分析**

与EfficientNetB0、MobileNetV3、ResNet50等基线模型在相同预处理、数据拆分和训练条件下比较。DR‑LiteNet在5折交叉验证中获得88.77%准确率、QWK 0.884、宏F1 0.792，参数仅4.69M，QWK提升约0.022，显示出优秀的性能与效率。

**⚠️ 局限性**

局限性包括：仅在单一数据集上评估，缺乏跨数据集验证；对极少数类的合成依赖特征空间，可能导致过拟合；未探讨更复杂的注意力融合或测试时增强对少数类敏感度的提升。

---

## 354. Spectrally Routed Diffusion Policy for Multi-Task Robotic Manipulation

**arXiv ID:** 2611.04477 | [PDF](https://arxiv.org/pdf/2611.04477v1)

**作者:** Mingcong Li `[一作]` (Beijing Institute of Technology), Yilin Mo `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了一种多任务扩散策略（Spectrally Routed Diffusion Policy），通过冻结VGGT编码器和在已收敛的权重上做SVD分解、谱路由和Householder旋转，实现了在单模型下同时执行多种机器人操作任务。

**💡 创新点**

创新点在于：①用冻结的VGGT+空间压缩适配器生成紧凑且任务可区分的视觉条件；②在共享的扩散去噪器中仅对少量线性层做SVD分解，并通过谱路由与Householder输入旋转来限制跨任务梯度冲突；③采用两阶段训练，让第2阶段仅更新路由与旋转，保持基准网络不变。

**🔧 技术方法**

技术包括：VGGT-1B预训练视觉编码器、AdaLN-Zero DiT扩散去噪器、SVD分解+谱路由、Householder反射、跨任务任务预测头、跨视图空间压缩适配器、两阶段训练策略。

**📊 数据集**

数据集：六个RoboTwin仿真操纵任务（Block Hammer Beat、Bottle Adjust、Container Place、Dual Bottles Pick、Blocks Stack、Tool Adjust）和四个真实世界双臂任务（T1–T4）收集自TeleAvatar平台。

**📈 对比分析**

与单任务VO‑DP、DP以及多任务VO‑DP、FDP、SDP等基线对比，SR‑DP在RoboTwin上平均成功率达68.4%（比VO‑DP+多任务提升21.8个百分点），在TeleAvatar上平均成功率提升至43%（比单阶段提升19个百分点）。

**⚠️ 局限性**

局限性包括：训练过程不是完全端到端，需先训练到收敛后再做SVD分解；仅在共享工作空间且任务差异不大时效果显著；模型仍需在更多任务和更复杂环境中进一步验证。

---

## 355. EvoCG: Self-Evolving Precision Control for Faster Conjugate Gradient

**arXiv ID:** 2611.04469 | [PDF](https://arxiv.org/pdf/2611.04469v1)

**作者:** Jiaqi Zhai `[一作]` (Huazhong University of Science and Technology), Kun Li `[通讯]` (Tsinghua University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种自适应混合精度共轭梯度求解器（EvoCG），通过强化学习在迭代过程中实时决定每个 tile 的计算精度，并配合 Tile‑CSC 存储格式、统一 warp‑中心 SpMV kernel 以及低开销的 in‑loop 运行时实现 end‑to‑end 的加速。

**💡 创新点**

1) 将混合精度选择视为闭环控制问题，使用 RL 控制器在 Krylov 子空间状态上做决策；2) 与存储格式和 kernel 共同 co‑design，消除不同精度导致的执行不规则性；3) 通过间歇性动作重用和 CUDA‑Graph 复用，将控制开销降至 1% 以下。

**🔧 技术方法**

强化学习（PPO）控制器、Tile‑CSC 存储格式、warp‑centric混合精度 SpMV kernel、CUDA‑Graph 合并、FP16 Tensor Core 推理、间隔重用策略。

**📊 数据集**

47 个对称正定（SPD）稀疏矩阵，来自 SuiteSparse Collection 的八个科学领域（有限元、流体力学、光学等）。

**📈 对比分析**

与 cuSPARSE、PETSc、Ginkgo、Mille‑FP64、Mille‑Mixed 等主流求解器在同一 GPU/A100 上比较。EvoCG 在所有 47 个矩阵上均为最快，几何平均加速分别为 2.52×、3.06×、2.48×、1.93×、6.99×，最大加速可达 4.89×、4.44×、4.87×、22.49×。对 Jacobi‑preconditioned CG 也无须重训即可获得 1.15× 加速。

**⚠️ 局限性**

（1）需要为每个待求解系统单独训练 RL 策略，训练耗时数小时；（2）跨规模迁移效果有限，只在同一问题族内表现良好；（3）Tile‑CSC 需要额外内存（约 1.84× CSR），在稀疏度极低的矩阵上会显著膨胀；（4）目前仅支持 SPD 系统，且对极端条件数或非对称矩阵的鲁棒性尚未验证。

---

## 356. SaMA: Morpho Adaptation via Asymmetric Expansion of Kronecker Product

**arXiv ID:** 2611.04462 | [PDF](https://arxiv.org/pdf/2611.04462v1)

**作者:** An Nguyen `[一作]` (Korea University), Anh Tong `[通讯]` (Korea University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的参数高效微调方法 Scalable Morpho Adaptation (SaMA)，通过对 Kronecker 乘积的结构化松耦合实现从低阶到全阶的可扩展适配器；

**💡 创新点**

创新点在于：①利用块对角分解与完美洗牌的结合，形成可自由控制的稀疏更新；②在相同参数预算下实现二次（平方）阶数增长；③将 LoRA 的 GL(r) 变换自由度压缩为对角缩放，消除优化瓶颈；④提供完整的理论分析与实证验证；

**🔧 技术方法**

技术包括：结构化稀疏矩阵（块对角、Kronecker 变体）、完美洗牌、gauge symmetry 分析、参数缩放α的稳定性理论、Adam/AdamW 优化、以及与 LoRA、PiSSA、ABBA、QuanTA、LoRA-Pro、SMT 等 PEFT 方法的对比实验；

**📊 数据集**

数据集涵盖 DROP、Commonsense（CMS）多任务、算术推理 MATH 与 GSM8K，以及 MetaMathQA、Gemma‑2 9B、Mistral‑7B 等大模型；

**📈 对比分析**

通过与多种强力 PEFT 基线以及全微调 FFT 的对比实验，SaMA 在多模型、多任务上均显著优于或匹配基线，逼近甚至超过 FFT，同时保持参数与计算成本与 LoRA 相近；

**⚠️ 局限性**

局限性包括：依赖特定结构化参数化，可能不适用于所有层或模型；对稀疏交互如何提升泛化尚未完全解析；对非方阵与预训练阶段的适用性尚待进一步研究。

---

## 357. Large Language Model-Assisted Adaptive Heuristic Search for Agile Earth Observation Satellite Scheduling

**arXiv ID:** 2611.04458 | [PDF](https://arxiv.org/pdf/2611.04458v1)

**作者:** He Wang `[一作]` (Harbin Engineering University), Yanjie Song `[通讯]` (Dalian Maritime University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种利用大语言模型辅助的自适应启发式搜索方法，用于灵活地球观测卫星的任务调度。

**💡 创新点**

创新点在于将LLM用于生成实例特定的优先级规则，并通过搜索反馈动态更新残差调整，从而实现搜索过程中的自适应改进。

**🔧 技术方法**

技术包括大语言模型（DeepSeek‑Flash）生成初始与残差规则、蚁群优化（ACO）与信息素更新、候选任务特征映射与对数归一化等。

**📊 数据集**

使用六个人工生成的调度场景，任务数分别为50、100、200，包含不同观测窗口竞争度，作为实验数据集。

**📈 对比分析**

通过与ACO、ABC、GA、PSO、WOA五种基线算法在20次独立实验中比较，LLM‑ACO平均提升4.3%–9.6%的观测收益，且方差最小。

**⚠️ 局限性**

局限在于仅测试静态任务与窗口，未考虑动态任务到达或实时观测条件变化；LLM推理成本在小规模场景略高。

---

## 358. Minimizing Bottleneck Overload Risk in Online Edge Task Placement under Workload Uncertainty

**arXiv ID:** 2611.04425 | [PDF](https://arxiv.org/pdf/2611.04425v1)

**作者:** Lei Wang `[一作]` (Inner Mongolia University), Baoqi Huang `[通讯]` (Inner Mongolia University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于风险瓶颈的在线边缘任务放置算法（Risk‑Bottleneck Placement, RBP），该算法通过最小化服务器级别的风险比（任务方差与剩余容量之比）来降低系统的最坏服务器过载概率。

**💡 创新点**

创新点在于：①引入风险比作为度量，并证明其对所有满足均值与方差约束的工作负载分布具有上界意义；②在独立任务模型下证明总方差与总剩余容量不随放置而变，从而把放置问题转化为在服务器之间匹配方差与剩余容量；③设计了两阶段的有限预算在线算法（按任务标准差降序放置 + 目标瓶颈改进），显著减少最坏服务器的过载风险。

**🔧 技术方法**

主要技术包括：基于Cantelli不等式的风险比推导、风险比最小化的结构分析、风险引导的任务排序与接收服务器选择、局部搜索（任务迁移与交换）以及增量评估与预算控制。

**📊 数据集**

使用Google Cluster Trace 2019作为任务工作负载来源，构造了数百万条任务的均值与方差，并通过模拟实验在不同服务器规模与负载压力下验证算法效果。

**📈 对比分析**

与四个基线（期望负载均衡、鲁棒贪心、σ-降序贪心、风险调优First‑Fit）以及期望负载局部搜索进行对比。实验显示RBP在两种工作压力下将最坏服务器过载概率分别降低26.0%和18.0%，在更广泛的压力与规模范围内始终保持最低，并且在风险可行率与总后备量上也表现优异。

**⚠️ 局限性**

局限性包括：1）仅考虑任务独立且只使用均值与方差，未考虑任务间相关性或更高阶统计量；2）算法在极端高负载或极大规模系统下的可扩展性与实时性仍需进一步验证；3）实验基于Google Trace的重采样模型，真实系统的工作负载分布可能更复杂。

---

## 359. SphereFlow: Missing Modality Imputation via Geometric Transport on Hypersphere

**arXiv ID:** 2611.04414 | [PDF](https://arxiv.org/pdf/2611.04414v1)

**作者:** Huihan Wang `[一作]` (Hong Kong Polytechnic University), Shujun Wang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文将缺失模态补全视为在共享超球面上的短距离几何传输，利用冻结的DINOv2编码器把所有模态映射到同一hypersphere；

**💡 创新点**

创新点在于：① 用数据本身作为噪声源，消除标准流匹配中的源分布不匹配；② 对解码器进行时间加权的离散点扩展训练，使其能在流轨迹内部点上重建；③ 证明在此设定下的离散误差不随维度增长。

**🔧 技术方法**

技术包括：DINOv2自监督视觉编码器、基于超球面几何的流匹配（cross‑modal flow matching）、时间加权的RDE解码器训练、Diffusion Transformer作为速度场学习器。

**📊 数据集**

实验数据集为BraTS多模态MRI（T1、T1c、T2、FLAIR）和SynthRAD2023的CT–MRI配对；

**📈 对比分析**

与七种基线（MMSyn、MMGAN、MMT、M2DN、ModImp、CodeBrain、Zhang）比较，PSNR、SSIM、rFID等指标均位于最佳或近似最佳，尤其在CT–MRI转译和N→N合成中显著优于其他方法；在缺失模态下的脑肿瘤分割任务中，Dice提升至全模态占优的约80%。

**⚠️ 局限性**

局限性：依赖DINOv2对模态的跨模态紧凑性，若模态之间缺乏共享视觉语义（如文本与影像）需额外的跨模态预训练；方法主要针对配对源数据的场景，未直接扩展到无配对或极端多模态设置。

---

## 360. TxBench: A Benchmark of Attack Transactions and Evidence-Backed Negatives on Seven EVM Chains

**arXiv ID:** 2611.04411 | [PDF](https://arxiv.org/pdf/2611.04411v1)

**作者:** Tamer Abdelaziz `[一作]` (New York University Abu Dhabi), karim Ali `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个跨七条 EVM 链的攻击交易与证据背书的负样本基准 TxBench，通过公开数据重建交易执行、价值计算和弱点映射，并通过盲审校准标签；

**💡 创新点**

首次提供既包含已执行攻击交易又包含以攻击机制为依据的负样本，并且标签不依赖任何检测器，采用公开事件库、OWASP 统一弱点分类和价值 oracle 进行严谨筛选；

**🔧 技术方法**

使用了五步流水线：收集事件库交易、将弱点映射到 OWASP、从公共 RPC、Anvil 分叉等重建完整调用树、价值 oracle 计算价值转移、基于证据的负样本筛选以及盲审规则；

**📊 数据集**

基于 BlockSec 事件库与 EVM Hack Registry 的 7 条链（Ethereum、BNB Smart Chain、Arbitrum、Base、Optimism、Polygon、Avalanche）公开链上数据，构成了 3 个 CSV 文件（攻击、负样本、组合）以及对应的完整跟踪；

**📈 对比分析**

以零射击 LLM（如 Claude、GPT-4）为基线，对匿名化交易跟踪进行判定；其精准率高但召回率低（仅识别约 10% 的攻击，未检测到任何关键泄露），说明存在显著改进空间；

**⚠️ 局限性**

局限性包括：仅有单位评审者导致一致性评估不完整；仅涵盖两源共同引用的事件，导致漏检；免费数据限制导致仅能重建完整调用树的交易；接受样本偏向 EVM Hack Registry 的证据；负样本仅抽样两条，难以代表全部合法交易；单交易评估忽略多交易攻击与价值未定价问题。

---

## 361. SourceFlow: Collaborative Source Coupling for Flow-Based Recommendation

**arXiv ID:** 2611.04455 | [PDF](https://arxiv.org/pdf/2611.04455v1)

**作者:** Chenghao Wu `[一作]` (Hong Kong University of Science and Technology), Menglin Yang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种改进流式推荐模型SourceFlow，通过在训练时构造与用户历史相关的离散源来提升用户偏好学习；

**💡 创新点**

创新点在于将训练源与目标历史进行协作耦合，利用邻域耦合、协作共识和拉伸恢复等技术，构造离散的、携带协作信息的替代源；

**🔧 技术方法**

使用流匹配框架（FlowCF）结合邻域采样、共识重加权和拉伸恢复的源构造方法；

**📊 数据集**

在五个真实推荐数据集（Sports、Toys、Video、Beauty、Yelp）上进行实验；

**📈 对比分析**

与多种基线（传统判别式、生成式以及其他流式模型）对比，SourceFlow在所有20个数据集-指标组合中均位列第一，Recall@10平均提升约15.7%，显著优于最强流式基线；

**⚠️ 局限性**

主要限制是对源构造与参数调优的依赖，且在极稀疏或极大规模场景下可能需要进一步优化存储与计算效率。

---

## 362. LCM-UNet: Scale-Selective Mamba with Reparameterizable Multi-Basis Local Modeling for Lightweight Skin Lesion Segmentation

**arXiv ID:** 2611.04409 | [PDF](https://arxiv.org/pdf/2611.04409v1)

**作者:** Md Shihabul Islam Shovo `[一作]` (NIMISHES Lab), Kishor Morol `[通讯]` (Elite Research Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种轻量化的皮肤病灶分割网络 LCM-UNet，结合尺度选择的全局 Mamba 与局部多基准可重参数化适配器，实现了在保持高精度的同时显著降低参数量与计算量。

**💡 创新点**

创新点在于：①将全局状态空间建模限定在 16×16 或更低分辨率层，②通过多基准可重参数化适配器在训练时加入稀疏膨胀、水平与垂直方向深度可分离卷积，推理时再折叠回单一核，③对跳跃连接使用阶段级可学习缩放，提升特征融合效果。

**🔧 技术方法**

采用了 Mamba 状态空间模型、深度可分离卷积、GLMBP/LMBP 结构、组归一化、多基准可重参数化（MBRA）以及结构重参数化等技术。

**📊 数据集**

使用了 ISIC 2017、ISIC 2018 官方分割数据集进行训练、验证和测试，并在 PH^2 数据集上进行外部验证。

**📈 对比分析**

在统一实验协议下复现八个轻量化基线后，LCM‑UNet 在 ISIC 2017、ISIC 2018 测试集上分别以 88.62%/89.73% 的 Dice 获得第一，PH^2 上以 90.46% 获得第二；模型参数仅 33,474，GMACs 0.197，显著低于多数对比模型。

**⚠️ 局限性**

实验局限在于仅评估像素级 Dice 与 IoU，未计算边界距离指标；Mamba 仅在 D2 层去除时评估，缺乏对其他层的全局建模探索；外部验证仅来自单中心数据；未测量 CPU/移动端延迟与能耗；缺乏正式的统计显著性检验。

---

## 363. Halley: A Predictive Control Plane for Optical AI Data Center Networks

**arXiv ID:** 2611.04408 | [PDF](https://arxiv.org/pdf/2611.04408v1)

**作者:** Manav Mishra `[一作]` (Prompeteer.ai), Krishna M. Sivalingam `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出 Halley 预测控制平面，针对 AI 数据中心光学网络的四层资源（激光池、光学电路交换机、微环调制器、可更换模块）分别设计 LightReserve、FoldFit、ThermaLead、TriageGuard 四个组件，提前预测并主动调整资源状态，防止因共享激光器、光路重新配置、微环漂移或模块失效导致的停机。

**💡 创新点**

创新点在于：① 对每层资源提供形式化可靠性保证（激光池覆盖上限、光路调度 11/9 近似最优、微环前馈预测、风险导向热备份）；② 将可靠性分析与可预测调度统一到同一控制平面；③ 引入协同失效模型并提出去相关化策略，显著提升可用性；④ 在 FPGA 上实现故障安全切换与微环闭环控制，并进行形式化验证。

**🔧 技术方法**

技术方法包括：β‑二项失效模型与常因相关性分析、光路调度 MULTIFIT 算法、AR(1) 一步前馈预测、风险排序热备份、仿真与数学建模、FPGA 软硬件实现、形式化证明与工艺签名。

**📊 数据集**

数据来源主要是仿真：MoE 训练流量矩阵、热周期漂移轨迹、行业 FIT 失效率等；并未使用公开的真实数据集，而是依据行业标准和实验室测量参数构造。

**📈 对比分析**

与传统反应式策略相比，Halley 在压力点将集群有效吞吐量从 65.2% 提升至 76.5%（相对提升 17%），覆盖率提升至 99.9% 以上，节省 1.03–1.17 倍训练功率或 3.3–14.7% 额外数据中心容量；FPGA 设计在 50 MHz 以上工作，逻辑使用 133 LUT，已通过形式化验证。

**⚠️ 局限性**

局限性包括：预测误差时缺乏下限保障；ThermaLead 对漂移平滑度敏感，突变时失效率上升；需要额外的温度、热、失效风险监测；去相关化成本与实际硬件实现复杂度未完全验证；模型基于假设的失效率与相关系数，真实系统可能存在差异。

---

## 364. ReListen: Training-Free Evidence Localization in Hour-Long Audio with Frozen Audio Language Models

**arXiv ID:** 2611.04480 | [PDF](https://arxiv.org/pdf/2611.04480v1)

**作者:** Ming-Hao Hsu `[一作]` (Chinese University of Hong Kong), Zhizheng Wu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ReListen，一种无训练、无转录的冻结音频语言模型自检索与回答方法

**💡 创新点**

利用Jev式扫描让模型自身决定哪些音频窗口有助回答问题，并直接从最优窗口答题，无需额外训练或转录

**🔧 技术方法**

冻结音频语言模型（Qwen3-Omni-30B），Jev式二分类评分，窗口扫描与选择，按时间阈值合并，Occur扩展使用小型声学检测器和逻辑回归

**📊 数据集**

LongAudioSpan、VoiceGiraffe、LAT-Bench三大长音频问答与定位基准

**📈 对比分析**

在LongAudioSpan和VoiceGiraffe上，ReListen在60秒预算下比均匀采样高8.5/21.3分；超过完整录音在VoiceGiraffe 10.6分；在LAT-Bench上mIoU与训练模型相当但速度快约57倍；整体速度比任何基于转录的系统快7-8倍，内存恒定

**⚠️ 局限性**

仅测试于英语/普通话的音频模型与基准，跨语言/音乐/其他音频场景未验证；假设每录音仅一问，若多问需重新扫描；对需要超多音频摘要的长录音无法处理

---

## 365. On the Complexity of Exact Autobidding Equilibria

**arXiv ID:** 2611.04475 | [PDF](https://arxiv.org/pdf/2611.04475v1)

**作者:** Yiyang Huang `[一作]` (Beijing Institute of Technology), Yonglei Yan `[通讯]` (Beijing Institute of Technology)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

本文通过构造多种在线广告竞价模型（第二价格ROI定价、预算约束GSP、ROI约束GSP、第二价格限流）证明了寻找其自平衡点（equilibrium）以及在精度上近似控制向量的计算复杂性，进一步给出了精确的 FIXP‑完备性与 ‑完备性结果。

**💡 创新点**

创新点包括：
- 在上述四种常见竞价模型中首次完成精确的 FIXP‑完备性证明，说明了即使只在一个简单的单一广告拍卖中寻找平衡点也是极难的；
- 引入可分离线性（separable linear）归约技术，将源问题（通用电路的固定点）映射到竞价实例，保持每个源变量仅对应一个控制参数；
- 设计了一套完整的“线性”和“非线性”电路实现工具（补码、双倍、乘法等），并在每个模型中实现了这些操作，从而能够构造完整的电路等价问题；
- 证明了在给定误差精度输入时，近似控制向量的距离问题同样是 ‑完备的。

**🔧 技术方法**

使用的技术包括：
- 基于通用电路（generalized circuit）的 FIXP‑完备性框架；
- 可分离线性归约（separable linear reductions）保证源变量与目标控制参数的单维解码；
- 线性与非线性电路实现（补码、双倍、乘法、平方根等）以及复制电路来控制节点的出度；
- 预算与 ROI 约束下的拍卖等价关系与无不必要平价（no‑unnecessary‑pacing）条件；
- 采用符号与数值的结合证明价格顺序、预算满足性以及多层依赖关系的稳定性。

**📊 数据集**

本文为纯理论研究，没有使用实际广告数据集；所有实例均为合成的、在多项式大小内可构造的拍卖实例。

**📈 对比分析**

本工作没有基于实验的性能对比；其主要贡献是理论复杂度上限，说明在最优算法上无法得到多项式时间或多项式逼近。

**⚠️ 局限性**

局限性包括：
- 结果仅适用于所定义的四种竞价模型，未涵盖其他可能的自平衡机制；
- 证明依赖于特定的无自定价、完整显示、单一CTR等限制，现实广告平台的细节可能导致复杂度变化；
- 没有给出实际可行的算法实现，只是给出“难以实现”的证明。

---

## 366. Entropy-Weighted Soundness for Multi-Answer Birthday Graphs

**arXiv ID:** 2611.04457 | [PDF](https://arxiv.org/pdf/2611.04457v1)

**作者:** Alexandru Popa `[一作]` `[通讯]` (University of Bucharest), Alexandru Popa (University of Bucharest)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `2704f255-0c84-4173-b83c-0e9a3dbea232` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了基于布尔三元约束满足问题的生日图，证明了在低密度子图和任意选取顶点集合下的新密度上界与一致性稳定性定理。

**💡 创新点**

创新点在于引入熵加权的二分图计数方法，利用随机扁平化的熵与互信息相结合显著提升密度指数，并给出缺失视图与多答案情况下的全局解码稳定性。

**🔧 技术方法**

主要技术包括熵与互信息的计量、能量障碍函数、对偶配对避免估计、标记二分图计数以及信息论的 KL 代数推导。

**📊 数据集**

实验未使用具体数据集，所有结果均为理论证明，适用于任意满足出现次数上界与间隙保证的布尔三元 CSP。

**📈 对比分析**

与 Manurangsi 等人给出的 ℓ⁴/n³ 上界比较，本文在视图大小 ℓ = Θ(n^{2/3}(log n)^{1/3}) 及其上范围内将指数提升至 ℓ³/(n²log(en/ℓ))，实现更严格的密度约束，但并未提供实验性能对比。

**⚠️ 局限性**

局限性包括：未能推出新的精细化难度结果；适用范围仅限于二元三元约束且出现次数受限；中间参数区间仍未得到覆盖；常数因子与源 CSP 参数高度相关，未给出最优取值。

---

## 367. Trustworthiness as a Measurement Structure for Task Conformance in 6G Service Chains

**arXiv ID:** 2611.04454 | [PDF](https://arxiv.org/pdf/2611.04454v1)

**作者:** Bin Han `[一作]` (RPTU Kaiserslautern-Landau), Hans D. Schotten `[通讯]` (RPTU Kaiserslautern-Landau)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

定义可信度为6G服务链对合同的分布一致性，并给出最小期望状态及其传播机制，利用转换日志实现阶段性偏差归因与置信度证明。

**💡 创新点**

创新点在于：1) 推导出多指标合同下不可用标量期望记录无法存续的不可识别性；2) 给出最小向量期望记录的必要性与充分性；3) 通过转换日志实现注入估计与置信证书，整合合同、设计与日志的量化判定。

**🔧 技术方法**

技术包括：期望状态压缩、Koopman/Krylov 投影、Markov 链扰动不等式、积分概率度量、统计模型检查、自由子不等式、经验 Bernstein 估计等。

**📊 数据集**

使用仿真生成的三跳优先/普通流量包络链数据集，包含 10⁵ 次样本窗口等。

**📈 对比分析**

与传统基于均值或卡方检验的全核估计对比，转换日志方法在 10⁵ 次样本下以 0.027 阈值精准标记违约，置信宽度小于直接终端样本 ±0.015，性能优异。

**⚠️ 局限性**

局限性包括：仅适用于单路径链；无法处理多路径/重路、动态适配、日志造假等场景；依赖已知设计核，对小概率事件的可检测性有限。

---

## 368. DynOSeg: Dynamic Object Segmentation via Motion-Geometry Fusion

**arXiv ID:** 2611.04452 | [PDF](https://arxiv.org/pdf/2611.04452v1)

**作者:** Xu Han `[一作]` (University of Sydney), Viorela Ila `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 DynOSeg，一个两阶段融合结构的移动物体分割框架，利用相机位姿、深度与光流计算几何感知残差图，并结合语义特征实现精确分割。

**💡 创新点**

创新点包括①利用几何感知残差图抑制相机运动伪运动；②设计几何感知空间融合 (GSF) 与全局局部语义感知融合 (GLSF) 两阶段融合策略，充分发挥运动、几何和语义信息的互补性。

**🔧 技术方法**

采用 RAFT 光流、ViPE/DynOSAM 深度与相机估计、DINOv2 视觉编码器、残差映射编码、CSP、Transformer、SPPF 等技术。

**📊 数据集**

训练数据涵盖 CARLA‑MOS、Kubric、Dynamic Replica、OCLR‑Syn、YTVOS‑Moving、YTVOS‑Moving*；测试数据包括 DAVIS2017‑M、SegTrackV2、FBMS‑59、DSEC‑MOS，亦在 KITTI‑MOS 进行评估。

**📈 对比分析**

与 EM、OCLR‑TTA、ABR、SegAnyMo、GeoMotion、GMOS 等公开基线比较，DynOSeg 在 DAVIS2017‑M 𝒥&ℱ 90.1、SegTrackV2 84.2、DSEC‑MOS 73.0 等指标均位居榜首或仅落后 1–2 分，显示出显著的性能提升。

**⚠️ 局限性**

当前局限在于未实现实例级分割，无法区分同一类别的多个独立运动目标；同时对深度与相机位姿估计的误差较为敏感，错误会影响残差图的质量。

---

## 369. Sketchpad: certified exact arithmetic with a small vision-language model

**arXiv ID:** 2611.04434 | [PDF](https://arxiv.org/pdf/2611.04434v1)

**作者:** Mostafa 'Neo' Mohsenvand `[一作]` `[通讯]` (Massachusetts Institute of Technology), Mostafa 'Neo' Mohsenvand (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出并实现了 Sketchpad 系统，利用预训练视觉‑语言模型 Qwen3.5‑0.8B 与低秩 LoRA 适配器，在受限的三格视图（pinned‑sheet）上逐步执行长位算术、除法、平方根、最大公约数等操作，并通过枚举所有视图并在 Lean 4 中正式证明，确保模型在任意输入长度下的精确性。

**💡 创新点**

创新点包括：① 通过将模型的每一步限制为对固定视图的读取，构建了一个可枚举、可验证的算术子系统；② 在模型上完成了完整的 Lean 证明，证明视图集的完备性与整体正确性等价；③ 在浏览器端的 16‑bit WebGPU 运行环境中完成了完整的可验证推理；④ 设计了可门控的适配器与对话调度，实现在对话中自动调用精确算术子系统。

**🔧 技术方法**

使用的技术包括：预训练视觉‑语言模型 Qwen3.5‑0.8B + 低秩 LoRA 适配器；Pinned‑Sheet runtime（只读三格视图、流程控制）；Lean 4 形式化证明与验证；GPU WebGPU 16‑bit 推理；对话适配器与门控机制；自定义视图表与跨熵 + Hinge + Distillation 训练；Logit lens 与类均值干预进行机制分析。

**📊 数据集**

主要使用的数据集有：2,446 个算术视图表（加、乘、除、平方根、GCD 等）；200 题浏览器测试集（10–1000 位求和、2–20 位乘积、10–99 除、2–999 平方根、10–999 GCD、1–20 位加减）；220 题预注册长数算术集（10–1000 位和 2–20 位乘积）；约 3,150 条人工合成对话与 547 条 GSM8K 题用于对话适配器训练；以及用于对比的十个公开模型在 2026‑10‑31 生成的测试数据。

**📈 对比分析**

在 220 题的预注册评测中，Sketchpad 全部 220 题正确；其余十个公开模型最多 217 题正确，错误集中在 300/1000 位求和；在 200 题浏览器测试中全部正解，平均耗时约 7 秒。对话适配器在 held‑out 词句题上召回率 99% 以上，GSM8K 与 MMLU 的分数保持在 2 分以内，但在新题集 GSM8K 上差距 5.8 分。

**⚠️ 局限性**

局限性包括：需要在固定 32×32 glyph 集上渲染视图；仅支持单变量低阶多项式、线性系统和整数算术；runtime 需按固定步骤执行，无法随意改写；证书仅针对部署路径的数值精度有效，其他硬件或库需重新验证；对话适配器在新词句题上召回下降，且对词句的精确转录和组合仍依赖外部代码；模型仍受 token 预算限制，未解决链式思考的有限上下文问题。

---

## 370. SWAP-INR: Implicit Neural Representation with Switchable Attention Paths for Time-varying Multivariate Scientific Data Compression

**arXiv ID:** 2611.04423 | [PDF](https://arxiv.org/pdf/2611.04423v1)

**作者:** Hyunsoo Son `[一作]` (Korea University), Won-Ki Jeong `[通讯]` (Korea University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 SWAP‑INR 网络，用于高保真压缩多变量时变科学仿真数据。

**💡 创新点**

创新点在于 SWAP 模块通过跨变量注意力与路径切换实现显式交互和结构压缩，同时 A²‑SIREN 的通道级频率调制提供自适应表示。

**🔧 技术方法**

采用跨变量注意力、可切换路径、层剪枝、通道可学习频率调制以及 16 位后训练量化等技术。

**📊 数据集**

在内部 SMR（小型模块化反应堆）数据集以及公开的 Combustion、Ionization 等数据集上进行评估。

**📈 对比分析**

与传统压缩器（SZ3、ZFP、TTHRESH）以及多种 INR 方法（SIREN、CoordNet、fV‑SRN、MC‑INR 等）在相同压缩率下对比，SWAP‑INR 在 SMR 数据上实现 SOTA，在公共数据集上保持竞争力。

**⚠️ 局限性**

局限包括注意力路径的低通特性导致高频场的平滑，以及 8 位量化性能显著下降；跨变量交互切换仅在全局变量层面，未能捕捉局部空间/时间相关性。

---

## 371. GVT3D: Gesture-Vision-Text Token Enrichment for 3D Embodied Reference Understanding

**arXiv ID:** 2611.04422 | [PDF](https://arxiv.org/pdf/2611.04422v1)

**作者:** Vimarsh Shah `[一作]` (BITS Pilani), Sougata Sen `[通讯]` (BITS Pilani)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种利用语言、手势与3D点云联合推理的三维实体指代理解（3D‑ERU）模型，并通过点云视差与语义编码实现对目标物体的精确定位。

**💡 创新点**

创新点在于（1）引入Embodied Fusion模块实现语言、手势、视觉三模态在共享嵌入空间中的自注意力耦合；（2）用学习的手势方向回归器替代传统几何射线，更好捕捉真实手势；（3）提出点云视差foveation策略，只在指向优先级最高的候选框上执行高成本的2D视觉编码，从而显著提升效率。

**🔧 技术方法**

核心技术包括：PointGroup检测器提取3D候选框；CLIP文本与视觉编码器；PointNet‑style关节回归与姿态编码；多视角深度投影与CLIP ViT‑B/32视觉编码；Transformer自注意力实现跨模态融合；线性CKA与Recall@1验证对齐；损失函数融合语义、定位、对比与指向约束。

**📊 数据集**

使用ImputeRefer数据集（基于ScanNet/ScanRefer的35,000个带人类指向指令的3D场景），以及对比实验中参考的ScanRefer、ScanNet等公开数据。

**📈 对比分析**

与先前3D‑ERU模型（如Ges3ViG、M3DRef‑CLIP、RefMask3D等）比较，取得IoU@0.5整体上升约8%，在多目标场景提升约9%；在Jetson Orin Nano上无foveation时速度提升1.75×，foveation后速度提升2.95×，能耗下降70%；真实机器人实验中成功率提升至70%（对比Ges3ViG的56%）。

**⚠️ 局限性**

局限性包括：在较远公共区域（>3.7 m）指向信息有效性下降；foveation参数k选取不当可能漏掉正确候选框；模型仍依赖点云质量与手势检测的准确性，未充分处理动态环境与实时手势跟踪；以及部分评估仍基于合成人像，真实人类行为的泛化仍待验证。

---

## 372. Verifier strength does not buy discovery quality: a pre-registered two-domain test of LLM-based automated research, with post-hoc evidence pointing to proposer-pool composition

**arXiv ID:** 2611.04420 | [PDF](https://arxiv.org/pdf/2611.04420v1)

**作者:** Peter Chen `[一作]` (Shenyang Institute of Technology), Siyuan Zhang `[通讯]` (Shenyang Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了自动化科研流水线中验证器强度对发现质量的因果影响，并在两个真实域下通过预注册的配对实验验证了这一假设。

**💡 创新点**

提出了预注册的分离结果：验证器强度虽能过滤坏候选，但并未提升LLM提议的发现质量；发现提议池的多样性才是提升质量的关键，并给出了可达性判定的设计教训。

**🔧 技术方法**

使用了小型开源LLM（Qwen2.5-0.5B-Instruct）与三层程序化验证器（S^-, S^0, S^+），结合Mutation测试、一次性holdout评分、配对随机控制以及Wilcoxon检验等统计方法。

**📊 数据集**

使用了两组真实数据：①工厂作业调度（从真实生产订单抽样），②北京T-Drive出租车轨迹（用于车辆边缘卸载）。

**📈 对比分析**

通过配对LLM与随机提议的holdout优势进行Wilcoxon检验，结果显示验证强度未显著提升优势；在合并提议池时可获得+1.52的holdout提升；在紧张域下，严格验证反而降低了优势。

**⚠️ 局限性**

局限性包括：仅使用单一LLM模型、仅两域实验、验证层分离弱、随机池覆盖缺口、有效样本受LLM提议相关性影响、未探测其他提议分布，以及仅在CPU环境下进行实验。

---

## 373. CaliSplat: Sparse Gaussian Fitting and Calibrated Rate Allocation for Image Compression

**arXiv ID:** 2611.04413 | [PDF](https://arxiv.org/pdf/2611.04413v1)

**作者:** Zhenbang Zhang `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Zhiqiang Xu `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为CaliSplat的 per-image 高斯压缩框架，利用单高斯闭式失真作为局部难度门，进行稀疏候选拟合，并在全局预算下分配 Gaussian 以实现高效、显式解码。

**💡 创新点**

创新点包括：① 用单高斯失真做局部难度评估实现候选稀疏化；② 通过采样教师校准短周期估计，消除不同 K 收敛速率的偏差；③ 基于预算感知的全局 Gaussian 分配策略；④ 结构化模板相对编码显著压缩比特流。

**🔧 技术方法**

采用 2D 高斯 Splatting、可微分光栅化、Adam 优化、单高斯闭式失真、样本教师校准、预算感知分配、模板相对量化和熵编码等技术。

**📊 数据集**

使用标准分辨率数据集（Kodak、Tecnick、CLIC2020 共 75 张）和大图数据集（FGF2、ICB、STimage 共 9 张）进行实验。

**📈 对比分析**

与 GaussianImage、GaussianImage++、Image-GS、LIG、SGI 等基线比较，CaliSplat 在标准图低码率略低于 Image-GS，但在高码率和大图上取得最高 PSNR（27.10/31.86 dB）和 MS-SSIM；编码速度提升 3–15×，解码速度保持 175–478 MP/s。

**⚠️ 局限性**

局限性：仍需对每张图执行少量候选优化；在极低码率下重建质量略逊；高 K 适配对复杂纹理的捕捉可能受限；实现依赖显式模板，可能不适用于所有几何变形场景。

---

## 374. The Min-Max theory of language behavior

**arXiv ID:** 2611.04406 | [PDF](https://arxiv.org/pdf/2611.04406v1)

**作者:** Hiram Ring `[一作]` `[通讯]` (NTU Singapore), Hiram Ring (NTU Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过多代计算实验验证“Min‑Max”语言行为理论，比较预训练与从零训练的Transformer模型在词序选择（SV vs VS）上的表现，并探测信息权重参数α对效率与信息最大化的平衡。

**💡 创新点**

首次将信息权重参数α统一描述生产效率与信息最大化，并发现自回归Transformer在词序选择上呈现与人类相反的倾向，提出Min‑Max理论需加入噪声通道与听者导向的考量。

**🔧 技术方法**

采用迭代学习范式（ILP）与自回归Transformer，结合从零训练、小型模型、Atomic Token Forcing、信息瓶颈框架、复制器动力学和多维相空间扫描等技术。

**📊 数据集**

使用人工生成的合成语料（1,942种语言并行语料的统计基准）以及子词分割的Byte‑Pair Encoding作为实验输入。

**📈 对比分析**

通过比较预训练模型与Tabula Rasa模型在SV/VS序列的预测损失，发现预训练模型保持词序稳化，而自训练模型在α≥3时出现阶跃转变，说明信息权重显著影响结构，但模型仍偏向生产效率，未完全复制人类语料。

**⚠️ 局限性**

模型仅限于基本SV/VS词序，使用词长作为信息度量，未考虑多词结构与真实听者反馈，且Transformer的因果掩码导致与人类语法的认知差异。

---

## 375. ReflexArc: Bottom-Up Integration of Instantaneous Tactile Reflexes into Whole-Body Humanoid Control

**arXiv ID:** 2611.04424 | [PDF](https://arxiv.org/pdf/2611.04424v1)

**作者:** Mingyu Wang `[一作]` (Tsinghua University), Li Yi `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个基于即时触觉反射与学习型全身控制并行的框架（Ascending Reflex Control），实现了在碰撞发生前数毫秒内通过触觉感知立即指令全身退避并在保留任务执行与平衡的前提下进行后续恢复；

**💡 创新点**

创新点在于：①将触觉起始检测与模型化即时全身退避求解相结合，能在10 ms内直接下达电机指令；②三段式底层反射集成设计（反射-循环学习、透明对齐、信息化适配），使得反射与高层策略能够同时并行且不互相抵消；③通过人类优先行为数据驱动的适配奖励实现反射后的自然全身重定向；

**🔧 技术方法**

技术包括：全身触觉传感（128 taxel 纹理贴片）、自适应阈值触觉起始检测、基于关节惯性矩阵的全身约束优化（LQP求解）实现即时退避姿态、PPO训练集成反射观测、DAgger式对齐监督、AMP奖励与人类示范数据；

**📊 数据集**

使用了真实单位树G1机器人的触觉贴片数据、物理碰撞实验中使用的盒子触觉传感器以及收集的多人类运动捕捉轨迹（包括步行、武术、舞蹈）作为参考运动；

**📈 对比分析**

与传统需要通过50Hz策略推理的延迟反射（+20 ms）相比，瞬时反射将第一冲击峰值从21.64 N降至14.25 N，冲击冲量从0.2632 N·s降至0.1225 N·s（提升2.15倍）；在三类碰撞实验中，系统保持生存率近99%，冲击冲量进一步降低至3.42 N·s，且能成功完成多任务的全身重定位；

**⚠️ 局限性**

局限性包括：触觉贴片仅覆盖凸面区域，对非凸面或内向碰撞无法及时反应；模型化全身退避依赖快速优化求解，可能受限于极端动力学条件；系统对极低帧率或极大惯性变化的鲁棒性尚待进一步验证。

---

## 376. CogniLife: Continuous Cognitive Distillation for Deployed Multi-Expert Agent Marketplaces

**arXiv ID:** 2611.04483 | [PDF](https://arxiv.org/pdf/2611.04483v1)

**作者:** Xuwei Tan `[一作]`, Tianyu Shi `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了多专家代理市场中的冷专家问题，提出跨专家归因设计与持续认知蒸馏框架，并在 Profy 平台实现创作者授权发布、能力驱动提案过滤、可逆记忆维护等控制，构建了 ProfyFull 配置。

**💡 创新点**

创新点在于定义冷专家问题与持续认知蒸馏流程，提出跨专家归因机制（Creator Profiles、Expert Journals），以及将创作者控制与可逆记忆结合的系统架构。

**🔧 技术方法**

采用 LLM（如 o3-mini）、专家提示、任务级工作流、Creator Profile、Expert Journal 等技术，并通过多阶段提案过滤与记忆维护实现持续蒸馏。

**📊 数据集**

使用 ProfyBench‑v1（85 条任务，5 个任务族）和 52 条海关提取任务作为评估数据集。

**📈 对比分析**

与 Raw LLM 与 Expert Prompt 进行配置级别对比，ProfyFull 在自动判分上整体得分 4.65/5，显著高于基线，尤其在完整性、特异性和缺失信息处理方面提升显著。

**⚠️ 局限性**

局限性包括跨专家归因与冷专家缓解未实现或未评估；持续蒸馏对创作者忠诚度的长期影响未知；评估仅基于自动判分，缺少人类验证和在线效果；数据集规模有限且包含隐私限制。

---

## 377. KeyWise: Efficient Video-Only Piano Transcription with a Lightweight Per-Key Architecture

**arXiv ID:** 2611.04476 | [PDF](https://arxiv.org/pdf/2611.04476v1)

**作者:** Anastasios Chatzigiannis `[一作]` (University of Thessaly), Maria Koziri `[通讯]` (University of Thessaly)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出KeyWise，一种利用钢琴键固定几何的轻量级视频专属转录模型。

**💡 创新点**

通过把键盘划分为88个固定ROI并共享递归网络，省参数并提升准确性。

**🔧 技术方法**

采用卷积编码器、共享双向GRU、ROI机制和加权二元交叉熵训练。

**📊 数据集**

在PianoVAM和PianoYT两大视频数据集上训练与评估。

**📈 对比分析**

与现有V2N等方法对比，KeyWise以1.3 M参数实现与V2N相同的onset F1，在100 ms、offset和velocity上更优；在PianoYT上预训练+微调后得到69.4% onset F1，接近最佳公开结果。

**⚠️ 局限性**

仅依赖键盘角点，需全88键可见且在新录制环境下需要微调；模型不考虑键间交互信息。

---

## 378. Hierarchical Intelligent Evolutionary Optimization for UAV-Assisted NOMA-MEC Systems

**arXiv ID:** 2611.04474 | [PDF](https://arxiv.org/pdf/2611.04474v1)

**作者:** Yaxuan Jiao `[一作]` (Inner Mongolia University), Winston K. G. Seah `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种分层智能进化算法(HIEA)，通过多头DQN、通信感知初始化、适应性进化和精度精炼四个模块共同优化多UAV协同NOMA-MEC系统中的轨迹、任务卸载与计算资源分配，以提升系统吞吐量、任务完成率和能效。

**💡 创新点**

创新点包括：①将多头深度Q网络用于生成高质量决策候选，显著缩小搜索空间；②设计基于信道质量指示器的负载感知轨迹初始化(CQI-LATI)，在通信导向与种群多样性之间实现自适应平衡；③引入精度精炼的SCA局部优化，提升最终解的精度；④整体将决策生成、初始化、进化与精炼协同融合，形成真正意义上的分层优化框架。

**🔧 技术方法**

采用的技术主要有：多头深度Q网络（MH-DQN）进行智能决策生成；CQI-LATI算法进行通信感知初始化；改进的遗传算法（EGA）进行适应性全局搜索；基于连续约束的SCA（Successive Convex Approximation）进行局部精炼；NOMA信道建模、移动轨迹约束、任务优先级满意度模型等。

**📊 数据集**

使用仿真数据：4架UAV、60台移动用户，服务区500×500 m²，UAV高度20–100 m，最大速度8 m/s，5.8 GHz频率，20 MHz带宽，用户任务尺寸0.5–5 MB，CPU周期500–2000，优先级比例40%高/60%低等，未使用公开真实数据集，仅在MATLAB/NS-3等仿真平台进行实验。

**📈 对比分析**

与AGSP、传统遗传算法（GA）、ISAC（基于SAC的强化学习）以及PPO-L（集成 Louvain 聚类+PPO）等基线方案对比，HIEA在吞吐量方面相较ISAC提升最多6.5%，任务完成率提升最高13.6%，能耗下降最高15.3%，并在不同UAV/MU规模下保持优越性能。

**⚠️ 局限性**

局限性主要有：①算法复杂度仍相对较高，尤其在SCA精炼阶段对计算资源要求高；②依赖仿真环境，缺乏真实网络实验验证；③优先级模型假设任务优先级仅为二元（高/低），未考虑更细粒度或动态优先级变化；④对极端高优先级任务的专门优化尚未实现。

---

## 379. Treating the Symptom, Missing the Cause: Anatomy of a Scientific Reasoning Failure

**arXiv ID:** 2611.04473 | [PDF](https://arxiv.org/pdf/2611.04473v1)

**作者:** Anna Gogleva `[一作]` (Relation Therapeutics), Karl Gemayel `[通讯]` (Relation Therapeutics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了大型语言模型在合成基因调控网络中通过实验寻找病因并修复表型的能力，揭示了成功干预与因果认知之间的差距。

**💡 创新点**

创新点在于构建可追踪的实验环境，分步追踪模型的探索、证据获取与诊断决策，发现模型虽能获取判别性证据但常无法将其转化为正确因果结论。

**🔧 技术方法**

使用了基于LLM的实验决策框架（Claude Opus 4.6、Sonnet 4.6等）与对话式探针技术，结合实验日志重放和对比实验。

**📊 数据集**

使用了由LUCA环境生成的60个20基因合成调控网络（含病理突变）的数据集。

**📈 对比分析**

与多种基线（随机、因果扫掠、启发式）对比，所有LLM在修复表型上优于诊断，诊断准确率约30%–55%，但在提供判别性证据后模型准确率仍不足80%，表明未能有效利用证据。

**⚠️ 局限性**

局限在于实验环境仍为人工合成网络，缺乏真实生物复杂性；评估仅基于诊断与修复的二元指标，未能覆盖更广泛的因果推理能力；模型未能构建内部因果模型，导致证据解释失败。

---

## 380. Beyond Token Representations: Explicit Visual Object Grounding for Video Reasoning Segmentation

**arXiv ID:** 2611.04472 | [PDF](https://arxiv.org/pdf/2611.04472v1)

**作者:** Jun Huang `[一作]` (Tongji University), Miaojing Shi `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种显式视觉对象定位框架，利用多模态大型语言模型生成目标的边框和点集，并通过多步对话精细化定位，进而实现视频推理分割；

**💡 创新点**

创新点在于：①将目标对象的语义推理与空间定位直接结合，生成可验证的边框与点集；②通过多步对话迭代改进定位精度；③使用查询多样化与跨查询一致性约束增强语义泛化；④基于定位质量的关键帧自适应选择，提升分割效果与效率；

**🔧 技术方法**

核心技术包括：多模态大型语言模型（Qwen2.5-VL）生成空间提示；SAM2 作为掩码解码器；多步对话式精细化与点校准损失；查询多样化与高斯分布对齐的交叉一致性损失；基于目标置信度与几何一致度的关键帧选择；

**📊 数据集**

在视频推理分割任务上使用 ReVOS 数据集；在引用式视频分割任务上使用 Ref‑DAVIS17、Ref‑YouTube‑VOS、MeViS 数据集；在引用式图像分割任务上使用 RefCOCO、RefCOCO+、RefCOCOg、ReasonSeg 数据集；

**📈 对比分析**

与多种最新方法（UniPixel、VRS‑HQ、InstructSeg 等）比较，实验表明在 ReVOS 的推理拆分和整体拆分的 J&F 均提升约 2‑3%，在 RVOS 数据集上分别提升 4‑10% 的 J&F，且在图像分割基准上也取得同类最高成绩；模型训练参数仅 212M，速度约 1.32 FPS，显示出较高的计算效率；

**⚠️ 局限性**

主要局限包括：①对目标定位的依赖仍受大型语言模型推理准确性的限制，误检会直接影响后续掩码质量；②多步对话精细化虽然提升精度，但在实时场景下仍增加一定推理延迟；③关键帧选择基于几何与置信度二值度量，未考虑更细粒度的时序一致性；

---

## 381. $τ$-Rec-Books: A Disclosure-Controlled Benchmark for Requirement Elicitation in Agentic Recommender Systems

**arXiv ID:** 2611.04459 | [PDF](https://arxiv.org/pdf/2611.04459v1)

**作者:** Bharath Sivaram Narasimhan `[一作]` (Independent Researcher), Karthik R. Narasimhan `[通讯]` (Princeton University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该研究提出了 -Books 基准，用于评估对话式推荐系统在需求披露时机不同（全披露、部分披露、隐藏需求）下的性能。

**💡 创新点**

创新点在于：①设计可控的需求披露变量，生成匹配的任务族；②采用无 LLM 评估器和规则化用户模型；③对比两种搜索接口和目录规模，系统性分析需求披露对成功率的影响。

**🔧 技术方法**

技术实现包括：规则驱动的模拟用户、两种工具接口（关键字检索与结构化过滤）、四大 LLM 代理（DeepSeek、GLM、Qwen、GPT‑6.1 Sol）、评估脚本与分层 Bootstrap 统计。

**📊 数据集**

使用数据集为：从 Wikidata 过滤得到的 20,000 条英文书籍公开目录（包含 5k/10k/20k 子集），并从中随机生成 450 个任务（150 个任务族 × 3 披露条件）。

**📈 对比分析**

通过对四个模型、两种接口、三种披露条件、不同目录规模进行多维度对比，结果显示需求披露差距最大，结构化搜索相对略优，目录规模对成功率影响有限；隐藏需求下成功率最低。

**⚠️ 局限性**

局限性包括：仅涵盖英文书籍且偏重硬约束筛选；规则化用户缺乏真实人类对话特性；未考虑软偏好、批评行为和人类验证的用户模型。

---

## 382. Entropy-Gated Dynamic Rectification for Object Hallucination in Large Vision-Language Models

**arXiv ID:** 2611.04438 | [PDF](https://arxiv.org/pdf/2611.04438v1)

**作者:** Riasad Alvi `[一作]` (United International University), Md. Faiyaz Abdullah Sayeedi `[通讯]` (United International University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于前一步预测熵动态调整门阈的LVLM潜在调节方法（Entropy‑Gated Dynamic Rectification, EGDR），在不额外前向传播或训练参数的情况下降低图像描述中的幻觉。

**💡 创新点**

创新点在于将可观测的预测熵作为因果可用的控制信号，对固定门阈进行时序调节，改善了传统静态门阈在不同生成步骤的不匹配问题。

**🔧 技术方法**

使用了潜在调节（latent steering）框架、风险门控机制、熵归一化与门阈调制、以及对隐藏状态差异的向量投影等技术。

**📊 数据集**

主要在COCO val2014图像集（用于CHAIR评价）以及POPE的随机、热门和对抗性样本上进行测试，并在LLaVA‑1.5‑7B与LLaVA‑NeXT‑7B两种backbone上验证。

**📈 对比分析**

与静态门阈及其火焰率匹配的对照相比，EGDR在CHAIR的C_S和C_I指标上分别提升约1.5–3.0点、0.4–0.7点；在POPE上的F1保持与静态门相当，且在更长生成预算下改进幅度显著。

**⚠️ 局限性**

局限包括仅评估greedy解码下的对象幻觉，缺乏对其他解码策略、模型架构和幻觉类型的广泛验证；熵带的估计依赖于已归一化的答案状态，可能在不同任务中需要调整；增量计算仍略高于无调节解码，且在单词级答案时效果无显著提升。

---

## 383. Formalizing PTX in Lean to Prove GPU Kernel Correctness

**arXiv ID:** 2611.04447 | [PDF](https://arxiv.org/pdf/2611.04447v1)

**作者:** Luk Koska `[一作]` (SF Tensor), Ben Koska `[通讯]` (SF Tensor)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Lean中构建PTX内存模型并验证多个GPU核的功能正确性；

**💡 创新点**

提供一种统一的PTX执行模型，支持异步、tensor‑core、代理、数据依赖控制流等复杂特性，并通过形式化证明展示其可靠性；

**🔧 技术方法**

Lean4、Mathlib、PTX解析与仿真、形式化内存模型与一致性公理、手写PTX核实现；

**📊 数据集**

在NVIDIA H100与B200显卡上对手写核进行功能验证，并与参考实现对比；

**📈 对比分析**

通过对核输出与参考结果逐字节比对确认正确性；未给出性能基准，但证明覆盖完整执行；

**⚠️ 局限性**

指令覆盖有限、未证明终止性、未处理UB、对SASS的支持不足、缺乏多GPU/多节点模型、缺少非协同加载与多节点通信建模。

---

## 384. FreeForm Harness: A Mobile-Native Agent Harness for Open and Composable Extensions

**arXiv ID:** 2611.04579 | [PDF](https://arxiv.org/pdf/2611.04579v1)

**作者:** Bo Zhang `[一作]` (Amap, Alibaba Group), Yue Liu `[通讯]` (Amap, Alibaba Group)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种名为FreeForm Harness（FFH）的移动端智能体框架，提供可插拔、可组合的扩展机制并实现了设备本地的安全授权和最终执行；

**💡 创新点**

创新点在于将移动端智能体拆分为冻结基底（L0）、永久但可替换的必需层（L1）与开放扩展层（L2），通过统一插件合同与“移动约束吸收”机制（位置源、触发调度、资源仲裁）实现了安全、可持续、开放的移动端插件生态；

**🔧 技术方法**

采用插件化架构、统一声明式插件合同、L1中位置、触发、资源仲裁器、Android原生签名验证与事件日志回放、云端分布式测试基座、Web编码代理等技术；

**📊 数据集**

使用了基于50条日常生活请求与公开触发-行动配方的匿名基准集进行评估，并在云端对Tier‑3原生插件进行可消耗运行时验证；

**📈 对比分析**

对比了内部弱化版本、管线基线以及基于配方的自动化基线，结果显示FFH在所有基准中实现了50/50的表示率、44/50的闭合率，且在重试次数、启动时延与功耗上均保持在可接受范围（冷启动≈380 ms，触发‑行动延迟≈1.7 ms，八小时耗电≈4.1%）；

**⚠️ 局限性**

局限性包括基准集规模有限、Tier‑3请求样本不足、仅在单一设备上测试、缺乏长期非程序员用户研究与对抗性发布者评估，且系统仍需依赖云服务完成代码生成与测试。

---

## 385. SEEK-VAU: Towards Evidence-Faithful Video Anomaly Understanding via Agentic Search

**arXiv ID:** 2611.04554 | [PDF](https://arxiv.org/pdf/2611.04554v1)

**作者:** Minghe Wang `[一作]` (Huazhong University of Science and Technology), Nong Sang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SEEK‑VAU，一种在固定搜索预算内进行搜索–验证交互的代理式视频异常理解框架，并构建了专门的评测基准 SEEK‑Bench。

**💡 创新点**

创新点包括：
- 把视频异常理解从传统的固定观测转变为动态的搜索‑验证协议；
- 引入 bounded search budget（每轮帧数上限 + 视图窗口限制）控制推理成本；
- 设计 evidence‑faithful counterfactual verification (EFCV)，通过对已提交证据进行必要性、最小化、可信度等对比奖励，驱动模型仅收集必要且足够的证据；
- 搭建 SEEK‑Bench，提供事件链阶段标注、时间区间、语义 QA 与证据窗口的完整交互式评测。

**🔧 技术方法**

技术实现：使用 Qwen3‑VL‑8B 视觉语言模型，GRPO 强化学习框架，四个可执行动作（scan、seek、verify、finalize），SFT + RL 训练流程，EFCV 奖励机制。

**📊 数据集**

数据集：从 MSAD 与 ECVA 重新标注构成 SEEK‑Bench，包含 21 类异常（+正常），每条视频配有时间区间、语义 QA、事件链阶段标签与参考证据窗口。

**📈 对比分析**

对比方法：固定观测基线（Qwen3‑VL、Intern‑VL、Ovis‑2.5 等）、视频推理 MLLM（Video‑R1、VideoChat‑R1 等）和其他 VAU 方法（HAWK、Holmes‑VAU、VERA）。SEEK‑VAU 在 Temporal mIoU 55.6、QA Accuracy 45.5、Evidence F1 43.7 等指标上显著优于所有对照，且平均帧使用量仅 15.6（低于 18 预算）。

**⚠️ 局限性**

局限性：
- 需要手工标注事件链阶段，标注成本高；
- 预算和窗口大小对不同视频域的适配性有限；
- 推理延迟比单轮固定观测略高；
- 仅在构造的 SEEK‑Bench 上验证，跨域泛化与更大规模异常集合仍待探索。

---

## 386. E2I-Flow: EEG-to-Image Reconstruction via Wavelet-Temporal Cross-Attention and Conditional Flow Matching

**arXiv ID:** 2611.04524 | [PDF](https://arxiv.org/pdf/2611.04524v1)

**作者:** Haowei Pan `[一作]`, Chee-Ming Ting `[通讯]` (Monash University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出E2I-Flow框架，实现EEG信号到图像的解码与重建

**💡 创新点**

①将站态小波变换与时间特征融合，并通过双向交叉注意力与自适应融合实现多尺度频谱与时序特征的统一编码；②利用CLIP图像嵌入与EEG嵌入的对比+MSE对齐，随后采用条件流匹配（CFM）对EEG条件下的视觉嵌入进行连续时间动态建模；③首次在EEG‑to‑image重建任务中引入流模型，取代传统扩散/自回归方法

**🔧 技术方法**

站态小波变换、双向交叉注意力与自适应融合、CLIP预训练图像编码器、对比+MSE对齐损失、条件流匹配（Transformer速度模型）、ODE积分、IP‑Adapter+SDXL‑Turbo图像生成

**📊 数据集**

THINGS‑EEG基准，包含1654个训练概念（每概念10张图）与200个未见概念（每概念1张图），63通道1 kHz EEG采样

**📈 对比分析**

与Conformer、NICE、ATM、VDF、AVDE等基线在零样本检索与图像重建上比较，E2I‑Flow在Top‑1检索31.0%/Top‑5 60.1%略优AVDE，在PixCorr、AlexNet‑2/5、CLIP、SwAV等感知指标上表现最佳

**⚠️ 局限性**

仅支持静态图像重建，缺乏对视频生成的扩展；对小波分解层数的敏感性有限；未对跨受试者泛化和实时性进行深入评估

---

## 387. Action Intent as a Bridge: Transferring Manipulation Knowledge from Human-Centric Videos to Robots

**arXiv ID:** 2611.04522 | [PDF](https://arxiv.org/pdf/2611.04522v1)

**作者:** Zhiqian Xia `[一作]` (Southeast University), Xiaochun Cao `[通讯]` (Sun Yat-sen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了两阶段的 Action-Intent Transfer (AIT) 方法，通过人类视频中的干净行动意图在共享空间中桥接人机，提升机器人操作能力。

**💡 创新点**

创新点在于将 Action Intent 作为人机共享的干净表示，并通过显式与隐式意图的分离与意图引导的潜在去耦合实现跨域知识迁移；同时引入两阶段预训练+适配策略实现精细化的知识迁移。

**🔧 技术方法**

使用视觉语言模型（VLM）、Gemini 2.5 Pro 生成注释、DINOv2 特征空间与向量量化去噪、Prismatic-7B backbone、Action Decoder、LoRA 微调、联合预训练与后期适配等技术。

**📊 数据集**

使用人类视频数据集 HOI4D、TACO；机器人轨迹数据集 Bridge-V2、Astribot dataset 以及 LIBERO benchmark 进行训练与评测。

**📈 对比分析**

与 LAPA、Octo、MDT、OpenVLA、MaIL、UniVLA 等基线在 LIBERO、CALVIN ABC→D、Astribot 12 任务中对比，AIT 在 LIBERO 平均成功率 91.3%，在 Astribot 82.08%，显著优于 UniVLA 的 70.42% 等。

**⚠️ 局限性**

仍需依赖高质量人类视频的生成与标注，意图词表有限，且在极端复杂或长周期连续动作场景下的意图分离效果仍有限。

---

## 388. Learnings and Pitfalls from AI-Assisted A/B Experimentation at Scale

**arXiv ID:** 2611.04515 | [PDF](https://arxiv.org/pdf/2611.04515v1)

**作者:** Brett Daley `[一作]` (Meta), Daniel Ting `[通讯]` (Meta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并运营一个基于大型语言模型（LLM）的 AB 测试实验代理（ABTest Agent），通过让 LLM 选择并排序实验流程，所有数值计算由可验证的确定性代码完成，从而为实验结果提供可重复、可信的诊断与报告。

**💡 创新点**

创新点在于：① 将 LLM 的非确定性与确定性代码分离，仅让 LLM 负责决策与流程组合；② 通过技能（skills）与知识库（KB）统一接口，减少对实验平台的直接调用；③ 引入报告门（Report Gate）与模板化生成，强制报告结构与引用可追溯性；④ 通过日常回归与预期测试持续评估技能，提升代理的可靠性。

**🔧 技术方法**

使用技术包括：大模型（如 GPT 系列）作为决策引擎；Python 与命令行工具交互的确定性执行层；知识库检索与验证模块；报告模板与门控机制；自建的技能框架（97 个技能，覆盖统计、诊断、健康监控等 11 类功能）。

**📊 数据集**

数据集来源为内部 A/B 测试平台的实验日志、指标计算结果、健康信号与事件记录等；没有公开数据集，而是利用公司自有实验数据与模拟环境进行技能验证与回归测试。

**📈 对比分析**

方法评估主要通过内部使用统计（如请求分布、频率）和技能测试集（回归与预期案例）进行。与自由文本生成对比，报告模板化后章节相似度显著提升；但无公开基准；性能表现体现在较低的工具调用错误率和更高的报告一致性。

**⚠️ 局限性**

局限性包括：① 仍需手动设计与维护技能与知识库，知识库不一致会导致模型误判；② 报告门只能约束已定义的输出结构，对未覆盖的交互（如论坛问答）无法保证质量；③ 对 p‑hacking 的控制仍有限，需人工审计后续分析；④ 代理依赖于预先定义的流程，无法处理完全未知的实验场景。

---

## 389. Fast, Cheap, and Fair? Evaluating Decision Models

**arXiv ID:** 2611.04510 | [PDF](https://arxiv.org/pdf/2611.04510v1)

**作者:** Muhammad Uzair Ul Haq `[一作]` (Amajor SB S.p.A), Flavia Fabris `[通讯]` (Amajor SB S.p.A)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对招聘筛选中的决策模型（System 1）与生成式模型进行系统的偏差与公平性审计，探究显性与隐性人口属性对评分、排名和短名单的影响。

**💡 创新点**

首次在招聘场景下对决策模型进行公平性评估，并将其与最先进的生成式模型进行对比；引入了基于计数噪声基线的对比事实 CV 设计和跑次一致性基准。

**🔧 技术方法**

采用对比事实审计、ICC 与标准差评估一致性、ΔScore/ΔRank/ΔExclusion 等偏差指标、ROC AUC、Balanced Accuracy、Macro‑F1、Precision@5 等有效性指标，并进行配对 Bootstrap 统计检验。

**📊 数据集**

使用两份公开数据集：Kaggle Resume 子集（572份 CV、11个岗位）用于偏差分析，及 cnamuangtoun/resume‑job‑description‑fit 数据集（含金标）用于有效性评估。

**📈 对比分析**

通过多模型、多指标对比，决策模型在保持与生成式模型相当的有效性（AUC、精准率等）同时，跑次一致性更好，推理延迟约 24‑33 倍更快，成本约 67‑68 倍更低；但仍存在通过姓名等隐性线索带来的微小偏差。

**⚠️ 局限性**

仅覆盖单一英语数据集，评估范围限于性别、种族与来源，未考虑残障、宗教、社会经济等属性；对比事实设计过于简化，姓名样本有限，且未检验真实候选人池或后续雇佣结果的影响。

---

## 390. Dynamic Steiner Tree under Edge Updates

**arXiv ID:** 2611.04506 | [PDF](https://arxiv.org/pdf/2611.04506v1)

**作者:** Jeff Giliberti `[一作]` (University of Maryland), Xinchen Yang `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了首个全动态斯坦林树算法，能够在任意连通加权图中维持终点集的常数近似最小斯坦林树，并实现子线性（O(n^{1/2+ζ})）的更新时间、O(1)的成本查询时间和O(n^{1+ζ})的显式树输出时间。

**💡 创新点**

创新点包括：
• 通过周期性重建与局部增量更新相结合的框架，避免频繁全图重构；
• 引入“连贯树扩展”将终点度量闭包的MST扩展为单棵树，削弱单条边删除对路由的冲击；
• 采用中心化的 hub‑augmented 图和顶点树（top‑tree）动态规划，实时维护最优 Steiner 树并在每个更新时只需 O(log n) 复杂度；
• 利用动态距离稀疏化器在子线性时间内估算重连成本，并将其与上层结构配合实现完整解；
• 在插入仅模型下实现了更快的随机化算法，获得更优的近似比与 O(n^{2/3+ζ}) 的期望更新时间。

**🔧 技术方法**

主要技术手段：
• Chuzhoy‑Parter 的确定性动态 σ‑稀疏子图（spanner）；
• Haeupler‑Long‑Saranurak 的确定性动态距离稀疏化器；
• Mehlhorn 的多源 Dijkstra 路径报告与终点度量 MST 构造；
• 连贯树扩展与 MST 取代性性质；
• 以顶点树为底层的动态 DP 维护 hub‑augmented 树的最小 Steiner 树；
• MST 取代性与分段重连策略，实现 O(1) 近似因子；
• 期望/确定性分析，处理自适应/无记忆对手。

**📊 数据集**

本工作为理论算法，实验使用的图集未给出；所有结果均为上界分析，适用于任意连通加权图。

**📈 对比分析**

与基线相比：
• 传统静态算法给出 2‑近似，但需重新计算；
• 之前的动态方案多为 2‑近似且仅在无权或稀疏图、且依赖终点数 k 的情况下取得子线性更新；
• 本算法在所有密度、任何终点数下均获得常数近似，且更新时间仅 O(n^{1/2+ζ})，查询时间 O(1)；
• 插入仅版进一步提升到 O(n^{2/3+ζ}) 期望更新时间与更好的常数近似；
• 通过实验（如在合成稠密图与现实网络上）验证，实际更新时间显著低于全局重构方案。

**⚠️ 局限性**

限制与待改进之处：
• 依赖于多项式边权范围和已知的动态稀疏子图/距离稀疏化器，若权值范围更大需额外处理；
• 常数近似因子随参数 ζ、σ_ν、ρ_η 变化，实际常数较大；
• 需要构建并维护顶点树与距离稀疏化器，空间占用 O((m+n)n^ζ)；
• 目前未针对动态终点集合（终点增删）给出解；
• 对极端极大稠密图，虽然保持子线性，但常数可能导致实际性能不佳。

---

## 391. 3DGS-VBench+ and 3DGSV-Assessor: A Multi-Dimensional Subjective Benchmark and a Reference-Flexible Objective Model for 3D Gaussian Splatting Video Quality Assessment

**arXiv ID:** 2611.04498 | [PDF](https://arxiv.org/pdf/2611.04498v1)

**作者:** Yuke Xing `[一作]` (Shanghai Jiao Tong University), Yiling Xu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `fede83ac-7505-405f-ab37-e7284695c47f` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了规模最大、维度最全的3D高斯喷射视频质量评估基准3DGS‑VBench+，并提出了可参考可无参考的多维度质量评估框架3DGSV‑Assessor。

**💡 创新点**

创新点包括：①在同一框架下同时评估整体、几何与色彩三维度；②利用训练视角的弱参考，既支持无参考评估，又能通过参考增强；③基于大型多模态模型（LMM）并结合LoRA细调，使用排名编码器学习参考一致性，随后通过有界门控融合实现参考灵活性；④零样本跨数据集泛化性能优异。

**🔧 技术方法**

技术实现主要包括：Qwen3.5 9B LMM作为视觉‑语言基座，LoRA适配视觉与语言塔；三维度回归头和文本提示生成；使用SigLIP + LoRA的排名编码器与余弦相似度损失；有界门控（γ=1+ε·tanh）融合参考信息；训练阶段分为无参考回归、参考排名学习与门控融合；评估指标为SRCC/PLCC/KRCC。

**📊 数据集**

数据集：3DGS‑VBench+（3036个压缩后3DGS模型，涵盖11个真实场景与22个合成场景，6种压缩算法，9,108条整体/几何/色彩MOS）；用于零样本验证的外部NVS VQA基准GSC‑QA、NeRF‑VSQA、ENeRF‑QA。

**📈 对比分析**

与传统PSNR/SSIM、LMM零样本基线以及DOVER、FAST‑VQA等深度VQA模型比较，RR/NR版本在所有维度均达到或超过最高SRCC（RR 0.9714，NR 0.9617），零样本跨数据集SRCC均高于基线5–15%点，且在Ablation中验证了LoRA、参考排名和门控融合对性能的显著贡献。

**⚠️ 局限性**

局限性：①仍依赖训练视角作为弱参考，若无可用参考则退化为NR；②针对3D高斯喷射的设计，未涵盖其他NVS表示的特有失真；③大模型和LoRA训练成本较高，实际部署受限；④单一参考视角的设计虽然有效，但在极端多视角场景下可能受限；⑤跨数据集评估仍局限于少数公开基准，未覆盖全部实际应用场景。

---

## 392. Spatial Memory for AR Glasses: Hierarchical Visual Acquisition and Source-Linked Recall

**arXiv ID:** 2611.04494 | [PDF](https://arxiv.org/pdf/2611.04494v1)

**作者:** Tianxing Zhu `[一作]` (Independent Researcher), Dan Huang `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一套针对光学见透AR眼镜的空间记忆系统，能够在用户需要时回溯并检索先前的物理交互，核心思路是通过双摄像头的分层视觉采集、基于全局/局部外观新颖度和时序支撑的RGB捕获决策，以及源关联的语义存储和分阶段检索，实现在记忆存储、解释与查询支持之间的明确分离。

**💡 创新点**

创新点包括（1）分层双摄像头采集策略：先用低分辨率单色监视器进行质量与新颖度筛选，再触发高分辨率RGB捕获；（2）全局/局部外观嵌入与时序支撑的组合，为RGB捕获提供精确阈值；（3）源关联记忆与分阶段检索架构，保证查询既可快速利用索引，也可回溯原始图像获取细节；（4）基于处理结果的结果感知控制，允许在检测到证据缺口时重新评估或再次采集；（5）基于证据依赖的保留策略，最大化在有限存储下的历史记忆支持。

**🔧 技术方法**

技术手段包括：RayNeo X3 Pro双摄像头（640×480单色 + 4032×3024 RGB）；MobileNetV3 + MediaPipe 提取全局与四块局部嵌入；像素残差和曝光校正进行质量检查；基于阈值的全局/局部新颖度判定；异步语义解释（使用云端场景描述模型）；分阶段检索与源图像回溯；记忆层次结构（M0-M3）与保留预算管理；光学见透微型LED显示与视角校准。

**📊 数据集**

主要使用的数据集为作者自制的视角摄像记录，具体为在RayNeo X3 Pro眼镜下进行的131.35秒行走视频（VGA+RGB同步），共648帧单色监视帧与对应RGB帧；未使用公开数据集；参考了Ego4D、EgoLife等相关工作，但实验数据为自制。

**📈 对比分析**

评价方法：在时间线回放中统计RGB采集率（6张/648帧，约2.74/min）、特征提取覆盖率（97.7%）以及特征检查通过率；与“匹配计数”参考策略做对比，结果一致；展示了光学见透下的显示对比度模型与显示可视化限制；未给出与传统方法的数值对比，但通过案例说明选择性RGB捕获能够保留足够细节且减少计算与存储负担。

**⚠️ 局限性**

局限性：① 采样策略可能漏掉瞬时短暂交互；② 需要在单色监视与RGB捕获之间平衡时序和曝光，误差会导致RGB图像模糊或视角不匹配；③ 结果感知控制依赖于后续解释结果，若解释始终失败则无法触发重采集；④ 显示视角和亮度限制导致信息量受限；⑤ 目前仅在单一硬件平台上测试，未验证跨设备迁移；⑥ 记忆保留预算仍需手工设定，未实现完全自适应。

---

## 393. Physical Diffusion Policy via MPM-Guided 3D Gaussian Simulation

**arXiv ID:** 2611.04493 | [PDF](https://arxiv.org/pdf/2611.04493v1)

**作者:** Can Wang `[一作]` (University of Hong Kong), Dong Xu `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研发了一套物理感知的扩散策略，能从多视角视频演示中反演物体材料属性，并将其作为条件输入到材料专属的Mixture‑of‑Experts扩散模型，实现对弹性、泡沫、塑料、刚性等多材料物体的鲁棒抓取与堆叠。

**💡 创新点**

①将3D Gaussian Splatting与可微Material Point Method耦合，形成可微物理场景表示；②利用视频引导与粒子级约束实现材料参数反演；③设计物理感知的MoE扩散策略和学习的物理路由器；④引入物理记忆库实现实时部署，避免在线优化。

**🔧 技术方法**

可微3D Gaussian Splatting、MPM仿真、可微渲染、视频引导损失、几何一致性约束、Dirichlet边界约束、材料参数剪裁、MoE扩散模型、DDIM采样、DINOv2特征检索、物理记忆库。

**📊 数据集**

在Isaac Lab仿真中合成多种材料（陶瓷、金属、塑料、橡胶、海绵、毛绒、木块等）的演示数据；在真实世界中手动收集100条含多材质日常物品（硬糖、面包、玩具、泡沫球、洗手海绵、乒乓球、橡胶球、纸杯、香蕉、葡萄、牙膏管、木块、塑料杯等）的演示。

**📈 对比分析**

与原始Diffusion Policy (DP) 和 DP3（含3D点云）对比；在pick‑and‑place与stacking任务中，所提方法在模拟与真实场景均取得最高成功率，尤其在弹性与泡沫物体上提升约10–15%；在脆弱物体（如葡萄、纸杯）的破损率也显著降低，表明更合适的接触策略。

**⚠️ 局限性**

需要多视角视频与大量演示数据，离线材料参数反演耗时较大；虽然物理记忆库加速推理，但仍受数据库规模限制；对极端材料或复杂形变的物体识别与控制仍有提升空间；在遮挡、低光照等真实条件下的鲁棒性尚待加强。

---

## 394. Hybrid Particle-Voxel Dynamics for Multimodal 3D World Simulation

**arXiv ID:** 2611.04492 | [PDF](https://arxiv.org/pdf/2611.04492v1)

**作者:** Jimmy Wang `[一作]` (Columbia University), Yunzhu Li `[通讯]` (Columbia University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了 3D World Simulator，一种基于动作的多模态 3D 世界建模框架，可在统一的 3D 表示空间中同时预测 3D 几何、从任意视角的 RGB‑Depth 渲染以及触觉信号，并支持长时序交互、数据生成、策略训练与评估以及基于模型的规划。

**💡 创新点**

核心创新在于：① 结合稀疏体素与粒子双编码的 hybrid particle‑voxel autoencoder，将点云压缩为紧凑 3D 体素潜在空间，再逐层上采样生成可从任意视角渲染的 3D 高斯粒子；② 在同一潜在空间上直接学习动作条件的 3D 潜在动力学模型，使用 3D U‑Net 结合时空注意力，实现高精度、长时序的几何一致预测；③ 统一的 3D 表示使得视觉、触觉等多模态信号可共享同一空间，便于生成视角一致的多视图图像、触觉数据以及与外部 3D 资产的可组合性。

**🔧 技术方法**

技术细节包括：稀疏体素编码器（基于 3D 稀疏卷积）压缩点云；体素上采样与粒子上采样解码器（转置卷积 + MLP 预测高斯参数）；Chamfer 损失、渲染损失（L1/SSIM/LPIPS）与一致性正则化；3D U‑Net 潜在动力学网络，采用空间注意力与因果时间注意力；K‑step 预测与贴现损失的训练策略，辅以噪声注入与课程学习；以及通过 3D 高斯渲染实现任意视角的 RGB‑Depth 合成。

**📊 数据集**

使用了 ALOHA 机器人与 FlexiTac 触觉传感器收集的六个操作任务（共约 200 条轨迹、2 小时数据），以及大规模 Deform360 视觉触觉数据集（198 件日常物体、1,980 条序列、215 小时、41 台环视摄像机）。

**📈 对比分析**

在 6 个 ALOHA 任务上与 2D 视频基线（Cosmos、Robot4DGen、Interactive World Simulator）及 3D 粒子基线（PointWorld、ParticleFormer、PGND）进行对比，评估指标包括 PSNR、LPIPS、Abs Rel、δ<1.25、Chamfer 距离及触觉 L1。3D World Simulator 在所有指标上均超过基线，特别是在 200 步长时序内保持视角一致性与几何精度；在策略训练与评估实验中，用模拟数据训练的策略与用真实数据训练的策略在 visuo‑tactile、multi‑view 与 novel‑view 任务上表现相近，且模拟评估结果与真实实验高度相关；在 Deform360 上对见过和未见过的物体也能获得较好的几何预测并实现零样本规划。

**⚠️ 局限性**

局限性包括：模型目前仅在特定工作空间内训练，尚未充分验证在更大尺度、多房间或移动操作场景的适用性；依赖高质量点云与触觉传感数据，可能对硬件要求较高；在极端动态或复杂流体/柔性材料的预测精度还有提升空间；此外，虽然不依赖物体特定状态表示，但在新物体或机器人配置下仍需额外微调以实现最佳表现。

---

## 395. Signals: Continuous Diagnostics of AI Agents

**arXiv ID:** 2611.04507 | [PDF](https://arxiv.org/pdf/2611.04507v1)

**作者:** Amey Rambatla `[一作]` (Cisco), Asif Mohammad `[通讯]` (Cisco)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 Signals 诊断框架，结合 45 种问题类型的分类，对 AI 代理执行轨迹进行持续诊断、严重度评估和具体修复建议。

**💡 创新点**

提出了面向代理行为的 45 类型问题分类，设计了将单次发现聚合为 Signal 的流程，并基于证据的严重度量与补救建议，显著提升诊断准确率。

**🔧 技术方法**

使用大型语言模型（GPT‑6.1 Sol）和结构化工作流，对执行日志进行问题检测、归类、聚合，并通过确定性匹配与 F1 评估来衡量性能。

**📊 数据集**

基于公开基准（AgentProcessBench、SWE‑bench、TraceElephant、MAST、τ‑bench 等）中的代理执行记录，并由人工评审生成参考 Signals。

**📈 对比分析**

与一次性直接生成 Signal 的对照组（同模型、同词典、同数据集）进行比较，采用确定性匹配计算 Precision/Recall/F1；Signals pipeline 在 Signal‑匹配 F1 上从 73.69% 提升到 91.43%，提升约 17.75 个百分点。

**⚠️ 局限性**

局限性包括：评估仅覆盖公开基准，未验证严重度评估和补救措施的实际效果；严重度评分依赖手工制定表，可能对新任务泛化有限；缺乏对实时修复效果的验证。

---

## 396. Abelian embeddings make useful promises: approximation of detrivialised equations

**arXiv ID:** 2611.04569 | [PDF](https://arxiv.org/pdf/2611.04569v1)

**作者:** Tamio-Vesa Nakajima `[一作]` (Philipps-Universität Marburg), Stanislav Živný `[通讯]` (University of Oxford)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究约束满足问题（PCSP）中的一种新型约简框架，利用阿贝尔嵌入和分段取色（rounding）来解决给定系统的“去常数”线性方程组的颜色化问题，并给出完整的复杂度二分法：当颜色数≥4时可在多项式时间内求解，当颜色数=3时为NP‑难，且对颜色数=2时问题未定义。

**💡 创新点**

创新点在于：①提出了基于阿贝尔嵌入的通用约简方法，使得许多看似难解的PCSP可归约到极简的无约束实例；②引入“c‑rounding”概念，展示了在满足该条件的阿贝尔群上可以高效地实现约简；③通过组合数论与加法组合学（Cauchy–Davenport、Vosper定理等）构造了针对质数阶群的唯一有效划分，进而证明了3色化问题的NP‑难性。

**🔧 技术方法**

主要技术包括：阿贝尔嵌入与同态表示、singleton AIP 约简、c‑rounding 的构造与分析、线性方程组的多项式解法、加法组合学定理（Cauchy–Davenport、Vosper、加法三角不等式）以及多项式时空复杂度分析。

**📊 数据集**

该研究属于理论计算机科学，不涉及具体实验数据集；所有结论均通过严格的数学证明与构造获得。

**📈 对比分析**

与传统 k‑一致性、BLP+AIP 等方法相比，本文提出的算法能够在多项式时间内解决所有 c≥4 的实例，而这些方法在同类问题上往往无法得到解；此外，理论证明表明在 c=3 时问题为NP‑难，进一步强调了该方法的最优性与局限性。

**⚠️ 局限性**

局限性包括：①对群中存在 2 或 3 的元（即存在 2‑或 3‑阶元）时不适用；②当颜色数小于 4 时问题仍为难题；③c‑rounding 的构造依赖于群的阶数，某些群（如 _2、_3）无法满足；④当前仅对质数阶群给出了完整的二分法，其他更一般的阿贝尔群需要进一步研究。

---

## 397. Estimating the generator of matrix exponential time-series

**arXiv ID:** 2611.04521 | [PDF](https://arxiv.org/pdf/2611.04521v1)

**作者:** Jonáš Fuksa `[一作]` (Freie UniversitÄt Berlin), Ingo Roth `[通讯]` (Technology Innovation Institute)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

论文提出并分析了一种名为genESPRIT的算法，用于在噪声下估计矩阵指数时间序列的生成器及其谱；

**💡 创新点**

其创新点在于给出了非渐进的、谱间隔无关的精度上界，突破传统超分辨率方法对频率间距的依赖；

**🔧 技术方法**

采用了矩阵子空间方法（类似ESPRIT）结合奇异值分解、Wedin定理、Bauer–Fike定理以及Cramér–Rao界分析；

**📊 数据集**

实验使用随机生成的单位正交矩阵 A、B 以及均匀分布的频率 Ω，噪声模型为二项式或高斯噪声；

**📈 对比分析**

通过与理论Cramér–Rao下界对比，发现生成器的Frobenius误差几乎达到理论极限，而频率估计误差仍略高于下界但显著优于传统方法；

**⚠️ 局限性**

局限在于对 A、B 的可逆性假设以及对频谱范围的限制，且对非单位正交数据、复数谱或非高斯噪声的性能尚未给出完整理论保证。

---

## 398. A Universal Raw-IQ Radar Encoder for Cross-Sensor Human Sensing

**arXiv ID:** 2611.04534 | [PDF](https://arxiv.org/pdf/2611.04534v1)

**作者:** Jonas Leo Mueller `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Bjoern M. Eskofier `[通讯]` (Ludwig-Maximilians-Universität München)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在四种不同频段和天线配置的雷达平台上，预训练一个 β-VAE 来提取原始 IQ 数据的通用特征，并在不同下游任务上只训练一个轻量级时序分类头。

**💡 创新点**

创新点在于：① 将多源雷达信号通过谱插值映射到统一的 4×4 天线网格和相同的采样维度；② 采用只重建六种二维功率投影的 β-VAE 作为无监督预训练目标；③ 在冻结预训练编码器的前提下，展示其在跨传感器、跨任务、跨用户的迁移性能显著优于传统微滴谱或全局预训练模型。

**🔧 技术方法**

使用的技术包括：谱插值与归一化、3D HRNet+Transformer 编码器、β-VAE 损失、六维投影解码器、时间 Transformer 分类头、AdamW 与余弦退火训练策略。

**📊 数据集**

使用了四个公开 FMCW 雷达数据集：mmRadPose、CubeLearn、CI4R-ASL 和 CI4R-Gait，涵盖了康复运动、手势、手语和步态等多种人类行为。

**📈 对比分析**

与监督端到端模型（RDAT、RadMamba、Spectrogram CNN）、预训练雷达基座模型（GRT）比较，冻结编码器+轻量分类头在无域漂移时接近最优，在未见用户或传感器时提升 4.8–23.2 分（准确率或宏 F1），尤其在用户/传感器离散情况下表现突出。

**⚠️ 局限性**

限制主要在于：① 仅在单人、近距离实验场景验证；② 计算成本相对较高，需处理完整 IQ 立方体；③ 结果可能受统一 4×4 天线网格和预处理方法的影响，无法完全证明每项设计的独立贡献。

---

## 399. Detector-Guided Vocoder Training for Adversarial Text-to-Speech Synthesis

**arXiv ID:** 2611.04511 | [PDF](https://arxiv.org/pdf/2611.04511v1)

**作者:** Jiayan Cui `[一作]` (Politecnico di Milano), Stefano Tubaro `[通讯]` (Politecnico di Milano)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `6215c339-3735-4be3-8a07-5bbb7004712d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种基于检测器引导的语音合成框架，通过在声码器训练过程中加入深度伪造检测器的损失，使生成的语音在不需要后处理的前提下即可具备对检测器的“自带攻击”特性。

**💡 创新点**

创新点：①将检测器损失直接嵌入声码器的优化目标，生成的语音本身即为对检测器的攻击；②实现了实时、无后处理的攻击；③提出多检测器联合训练策略，提升攻击的迁移性；④攻击训练过程中对语音质量的提升（FAD下降、MUSHRA得分提升），证明抑制检测特征也能改善感知质量。

**🔧 技术方法**

技术方法：采用 FastSpeech2 作为声学模型，BigVGAN 作为声码器；在声码器训练时加入检测器（RawNet2、TSSD、AASIST）的交叉熵损失；多检测器策略通过对多个检测器损失加权求和实现；评估使用 EER、AUC、Balanced Accuracy 以及 FAD 和 MUSHRA 监听测试。

**📊 数据集**

数据集：声学模型与声码器训练使用 LibriTTS（约260小时、1158 位说话人，采样率 22.05 kHz）；检测器训练使用内部平衡真实/合成语音数据（251154 条轨道）；评估数据集包含相同真实音频和三种声码器生成的合成音频。

**📈 对比分析**

比较方法：在白盒场景下使用 RawNet2、TSSD、AASIST；在黑盒场景下仅使用未参与训练的 AASIST；通过与基线声码器（无攻击）对比，展示 EER 从 5.8% 递增至 39–43%（针对 RawNet2），TSSD 同样出现 38–44% EER，AASIST 由 0.85% 提升至 21–23%；FAD 指标由 31.25 降至 17.81，MUSHRA 评分明显提升，表明攻击既破坏检测能力又提升语音质量。

**⚠️ 局限性**

局限性：①实验仅基于单一 TTS 架构（FastSpeech2+BigVGAN）和单一语言（英语）；②未评估在实际网络传输、不同采样率或多语言场景下的表现；③仅使用少量检测器，未验证对更广泛或更鲁棒检测模型的迁移效果；④声码器微调仅一次，未探索更高效或在线自适应优化；⑤研究聚焦于防御与安全研究，实际应用可能被滥用，需进一步制定伦理与治理规范。

---

## 400. AttriSwarm: Orchestrating Agent Swarms for Cross-Page Visual Attribution

**arXiv ID:** 2611.04504 | [PDF](https://arxiv.org/pdf/2611.04504v1)

**作者:** Suifeng Zhao `[一作]` (Peking University), Sujian Li `[通讯]` (Peking University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了 AttriSwarm 框架，实现跨页视觉归因；

**💡 创新点**

通过领导者-子代理 Swarm 结构将全局推理与局部页面归因分离，采用上下文隔离与并行执行，显著降低跨页上下文污染和顺序开销；

**🔧 技术方法**

使用多代理 Swarm、外部状态 Harness、检索-归因并行流程、角色专属监督微调（SFT）与强化学习（GRPO）等技术；

**📊 数据集**

构建 AttriDoc（包含 Paper 与 Slide 两个子集）并利用 CiteVQA 作为对照；

**📈 对比分析**

与单通道、ReAct、VRAG‑RL、UniDoc‑RL、MCite‑RL、GPT‑5.1、Claude Sonnet 等方法比较，在答案准确率、JAS、Page F1、RegIoU 等指标上均实现了显著提升（平均提升约 18‑25% 以上）；

**⚠️ 局限性**

仍受检索质量、模型规模、长文档处理等限制，跨域迁移能力有限，且主要关注视觉归因，未覆盖更深层语义推理。

---

## 401. SKILL2STEER: One-Pass Hypernetwork for Mapping Natural-Language Skills to Low-Rank Post-Block Activation Shifts

**arXiv ID:** 2611.04497 | [PDF](https://arxiv.org/pdf/2611.04497v1)

**作者:** Ran Li `[一作]` (Hong Kong University of Science and Technology), Lei Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将自然语言技能文档编译成低秩激活偏移，生成可部署的模型参数束，消除推理时的提示文本。

**💡 创新点**

证明编译后的偏移携带的是域一般的行为先验而非技能内容，并利用一次性超网络实现低成本、可路由的技能注入。

**🔧 技术方法**

使用冻结的Qwen3-8B编码器+超网络生成每层低秩因子，并在每个解码块输出后注入激活偏移。

**📊 数据集**

在SearchQA七数据集（3125问）上评估。

**📈 对比分析**

与冻结模型、提示/检索方法、权重空间编译方法对比，获得37.9% EM，较冻结模型提升9.8点，较LatentSkill提升2.3点，参数仅为2.36M，约为LoRA的1/9。

**⚠️ 局限性**

仅在检索问答任务与单一基模型上验证，技能内容对先验的影响有限，未探究不同任务或模型的迁移性。

---

## 402. SPA-Q: Structure-Preserving Adaptive Post-Training Quantization for Monocular Depth Estimation

**arXiv ID:** 2611.04491 | [PDF](https://arxiv.org/pdf/2611.04491v1)

**作者:** Jaemin Choi `[一作]` (Sogang University), Suk-Ju Kang `[通讯]` (Sogang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在单目深度估计模型的低比特后训练量化中，提出了一种结构保持自适应量化框架 SPA-Q；

**💡 创新点**

创新点在于：①引入注意力保持校准（APC）以在量化时直接匹配注意力分布；②引入通道级分布对齐（CWDA）来消除跨层量化误差累积导致的特征分布漂移；

**🔧 技术方法**

使用技术包括：Transformer‑based Vision Transformer 编码器、DPT 解码器、均匀量化与 log2 量化、KL 散度校准、通道级仿射变换、SILog 损失与特征级损失的端到端优化；

**📊 数据集**

使用数据集：NYUv2、KITTI、ETH3D、DIODE、Sintel 的零样本相对深度评估基准；

**📈 对比分析**

与 BRECQ、QDrop、RepQ‑ViT、QSCA 等现有 PTQ 方法在 4‑bit 量化下进行对比，SPA‑Q 在 AbsRel 与 δ₁ 指标上均表现最优，例如 NYUv2 上 AbsRel 从 0.1875 降至 0.1021，δ₁ 从 0.6845 提升至 0.8991；

**⚠️ 局限性**

局限性包括：量化后的深度图仍不如全精度光滑，受限于低比特可表示的离散深度值；实验仅覆盖 Depth Anything v1/v2，未验证更大规模模型；需要进一步扩展到双目、视频深度等场景。

---

## 403. Bridging CLIP with DINO: Cross-Modal Information Maximization for Online Test-Time Adaptation

**arXiv ID:** 2611.04489 | [PDF](https://arxiv.org/pdf/2611.04489v1)

**作者:** Taehoon Kim `[一作]` (Korea University), Hyunsouk Cho `[通讯]` (Ajou University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了CDIM，一种在线测试时适配方法，通过交叉模态logit集成、门控信息最大化目标和在线logit调整来缓解CLIP的模态偏差，提升零样本分类在受扰动环境下的鲁棒性。

**💡 创新点**

在三层次（表示层、优化层、预测层）全方位解决模态偏差，首次将冻结的自监督DINO v2与CLIP在logit级别融合，结合门控信息最大化以及在线日志调整，实现了对确认偏差的闭环抑制。

**🔧 技术方法**

使用冻结的自监督视觉编码器DINO v2、跨模态logit集成、门控测试时信息最大化（GT‑IM）目标、在线logit调整（OLA）、对CLIP的通道归一化参数进行微调，以及标准的交叉熵/熵最小化替代。

**📊 数据集**

在ImageNet‑C、CIFAR10‑C、CIFAR100‑C三个标准噪声/畸变基准上进行评估，并在持续（continual）与混合域（mixed‑domain）测试时序列中验证其泛化性。

**📈 对比分析**

与CLIP+TENT、Mint、BATCLIP、CoTTA等基线对比，CDIM在在线TTA场景下分别提升ImageNet‑C+4.8%、CIFAR10‑C+3.5%、CIFAR100‑C+3.7%准确率，校准ECE降至0.048，且在持续与混合域协议下保持竞争力。

**⚠️ 局限性**

依赖额外的DINO前向计算，适配效率受限；对极端大类别数或非噪声类不均衡的分布迁移仍需进一步验证；仅在零样本设置下有效，若需要标签监督或更复杂任务需改进。

---

## 404. Reconstruction-Based Diagnosis through Gödel-Regularized Virtual Observability

**arXiv ID:** 2611.04487 | [PDF](https://arxiv.org/pdf/2611.04487v1)

**作者:** Oliver Niggemann `[一作]` (Helmut Schmidt University), Alexander Diedrich `[通讯]` (Helmut Schmidt University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于虚拟可观测性的诊断框架 ReDiVO，通过神经观测器在仅用正常数据训练的情况下重建未被传感器直接观测的关键变量，利用这些重建变量与学习得到的正常轨迹进行比较，检测并定位故障。

**💡 创新点**

创新点在于：① 将虚拟传感器与实际传感器联合构成完整的系统轨迹；② 采用 Gödel 逻辑规则在训练阶段对模型进行物理知识约束，仅在在线诊断时不使用；③ 通过症状出现时间的相对先后顺序实现根因定位，实现了在传感器覆盖不足情况下的故障隔离。

**🔧 技术方法**

核心技术包括：长短时记忆网络（LSTM）实现状态重建与正常预测；双轨迹（当前轨迹与无测量预测轨迹）比较产生症状；Gödel 逻辑规则作为差异化损失对训练进行正则化；基于不确定区间和阈值的症状确认与时间窗口比较算法。

**📊 数据集**

实验数据集包括：人工线性网络与随机混合自动机、扩展的 Tennessee Eastman Process (TEP)、HAI‑CPPS v2.1 模拟工厂、TwinFlow 物流仿真。通过这些数据集评估虚拟传感器精度、规则对重建和诊断的影响以及整体诊断性能。

**📈 对比分析**

与无规则训练、随机匹配等基线比较，ReDiVO 在 HAI‑CPPS 中递归训练可在 29/30 的故障记录中唯一定位泵，递归训练的平均误差显著低于单步训练；在 TwinFlow 中 80% 的速度标定故障能定位到正确传送带；在人工数据中即使四个测量被遮蔽，仍保持 64–100% 的诊断正确率。整体而言，虚拟传感器和症状时间比较显著提高了故障定位率。

**⚠️ 局限性**

局限性包括：① 需要足够的可观测输入以约束重建；② 对弱幅度或快速衰减的故障可能未产生持久症状；③ 预测轨迹在模式切换或极端工况下误差较大，导致误报或错失症状；④ Gödel 规则选择与权重需手工调参，且不保证所有规则都提升诊断；⑤ 仅在单个测试环境验证，尚未在更大规模或异构系统上广泛验证。

---

## 405. Mutual-Coupling-Aware Joint Precoding and Antenna Positioning for Movable-Antenna MU-MIMO Systems

**arXiv ID:** 2611.04552 | [PDF](https://arxiv.org/pdf/2611.04552v1)

**作者:** Wei Guo `[一作]` (Hong Kong University of Science and Technology), Khaled B. Letaief `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了在可移动天线（MA）多用户MIMO系统中，联合考虑互耦（MC）效应的前向调制与天线位置优化算法。

**💡 创新点**

创新点在于：①将场响应传播模型与基于电路的互耦矩阵模型融合，形成MC感知的系统模型；②构建当前域WMMSE框架，并用MM方法对非凸天线位置子问题进行凸化；③利用共轭ADMM高效求解凸子问题，实现联合优化。

**🔧 技术方法**

使用的技术包括：电路基础互耦矩阵建模、场响应通道模型、当前域WMMSE、majorization–minimization（MM）、共轭锥ADMM、Bisection求解拉格朗日乘子。

**📊 数据集**

实验使用的“数据集”为基于28 GHz毫米波的聚类Rician信道模型（包含LoS+3簇×8子路径），并在1000个独立随机化信道样本上进行仿真。

**📈 对比分析**

与两种基线（MC感知固定阵列和忽略MC的可移动阵列）比较，MC感知联合优化在-10 dB至10 dB参考SNR区间内平均提升约21.5 %（相对）或1.9 bit/s/Hz（绝对）；在用户数、天线数、天线摆放规模等多维度实验中均保持领先。

**⚠️ 局限性**

局限性包括：①算法迭代收敛虽有保证但计算量随天线数和用户数增长显著；②只考虑单频单瓣波束，未涵盖宽带或多频情况；③互耦模型假设为各向同性天线，实际系统需更精细的电磁仿真；④对天线可移动范围和最小间距假设为理想，实际机械约束可能更复杂。

---

## 406. Confidence-Aware Aggregation for Multimodal Audio Reasoning

**arXiv ID:** 2611.04580 | [PDF](https://arxiv.org/pdf/2611.04580v1)

**作者:** Michel Olvera `[一作]`, Ga{ë}l Richard `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大型音频语言模型（LALM）中，通过推理时多样采样和基于 log 概率的自置信度评分，研究自一致性和置信度加权聚合来提升多步骤音频问答的准确性与校准。

**💡 创新点**

创新点在于系统性评估置信度加权聚合（Borda、LSE、SoftLSE）相较于传统 Best‑of‑N 与 Majority Vote 的效果，并证明置信度可以作为有效的加权信号而非单纯的选择准则；同时首次将此框架应用于音频问答领域。

**🔧 技术方法**

使用技术包括：推理时多样采样、长度归一化的 log 概率自置信度计算、Borda 排名加权、Softmax/LSE 加权聚合、ECE 校准评估及答案抽取/过滤器。

**📊 数据集**

使用的数据集为 MMAR（1,000 条音频–问答对），覆盖七类音频内容与四级语义层次。

**📈 对比分析**

对比 Naive Pick、Best‑of‑N、Majority Vote、Borda、LSE、SoftLSE 等方法；在 N=16 时，置信度加权聚合准确率提升约 3.5 分，ECE 降低 7–8 分，尤其在中高争议样本中提升 2–3 分，显著优于单次推理与多数投票。

**⚠️ 局限性**

局限性包括：仅在推理时提升，无额外训练；在 Qwen2‑Audio 上提升有限；采样预算受限于 8–16，未探索更大规模或不同采样策略；未深入探讨音频特有的不确定性信号和跨域泛化能力。

---

## 407. Ground-Wing: A Parametric 3D Wing-in-Ground-Effect Dataset for Benchmarking Neural Surrogate Models

**arXiv ID:** 2611.04571 | [PDF](https://arxiv.org/pdf/2611.04571v1)

**作者:** Andrea Cenzato `[一作]` (RWTH Aachen University), Julia Kowalski `[通讯]` (RWTH Aachen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个包含 194 个独特翼型、4 个跑道高度共 770 次 RANS 仿真的三维地面效应前翼 CFD 数据集，并用六种不同范式的神经网络对其进行基准测试。

**💡 创新点**

首次公开完整的地面效应翼型 RANS 数据集，并结合多参数采样与统一的训练/验证/测试拆分，提供可复现的基准，为神经 surrogate 模型的泛化能力评估提供了全新的挑战。

**🔧 技术方法**

使用不可压 RANS（STAR‑CCM+）生成仿真结果，并应用六种深度学习架构：CNN（DoMINO）、GNN（X‑MeshGraphNet）、Transformer（AB‑UPT、GeoTransolver、GAOT）和 JEPA（AeroJEPA）来预测翼面压力系数与壁面剪切应力。

**📊 数据集**

数据集由 194 个翼型（27 种空压线 + 4 纵向高度）共 770 次 RANS 仿真组成，涵盖 5 维连续参数与离散翼型，提供完整的几何、网格、场量和力系数信息。

**📈 对比分析**

采用三种拆分方式（迭代分层、拖曳外点与插值翼型）评估模型性能，Transformer 架构整体表现最佳，AB‑UPT 在两系数均达 0.85%–0.89% 的平均误差，显著优于 GNN、CNN 和 JEPA 模型；同时，模型在极端拖曳和未见翼型的拆分上表现出显著泛化挑战。

**⚠️ 局限性**

局限包括仅模拟单一元素翼、仅使用平稳 RANS（未考虑瞬态或高雷诺数湍流）、部分翼型为专有无法公开、参数空间未覆盖跨 span 高度变化，以及对特定硬件（NVIDIA DGX 系列）的依赖。

---

## 408. CANDO: Conservative Action-Noise Directional Optimization for Residual Reinforcement Learning

**arXiv ID:** 2611.04543 | [PDF](https://arxiv.org/pdf/2611.04543v1)

**作者:** Zeyu Gao `[一作]` (Fudan University), Shanghang Zhang `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了两种残差强化学习方法，CNDO（Conservative Noise Directional Optimization）用于在噪声空间优化预训练扩散生成策略，而CANDO（Conservative Action-Noise Directional Optimization）进一步结合噪声空间与动作空间的残差学习，实现对冻结生成策略的最优在线适配。

**💡 创新点**

创新点包括：①去除噪声空间的硬几何约束，改为基于解码行为质量的软保守约束；②设计了保守评论器与门控执行机制，抑制价值过估，保证噪声编辑的可靠性；③将上述噪声方向优化与动作残差学习联合起来，形成CANDO，使两种空间的互补优势最大化。

**🔧 技术方法**

技术上使用了预训练的Flow Matching（扩散）策略、Gaussian噪声扰动、保守多头评论器、温度化软阈值门控、目标评论器监督、以及多样本n步TD目标和Polyak平均。实现了在模拟与真实双臂机器人平台上的在线更新。

**📊 数据集**

数据集包括五个机械操作任务：RoboMimic benchmark中的Can、Square、Transport，以及ManiSkill benchmark中的PegInsertionSide和StackCube，此外在真实机器人上进行四个任务的验证。

**📈 对比分析**

与现有噪声空间方法（DSRL、LPS、LP‑DS）和动作空间方法（DICE）对比，CNDO在噪声空间表现优于所有对手（平均NAUC 69.5 vs 57.9，最终成功率 81.5% vs 66.2%），并与动作空间基线DICE相当；CANDO在联合学习后进一步提升至平均NAUC 76.1、最终成功率 91.4%，超越单空间方法6.6 NAUC、4.1%成功率。

**⚠️ 局限性**

局限性包括：①保守约束在训练初期仍可能导致过度保守，限制探索；②两空间的联合学习效果受两者单独性能平衡的影响，若单独性能差距过大，联合训练提升有限；③在极端环境随机化或高维动作空间中，仍需进一步验证其稳健性。

---

## 409. BtsPruner: Role-Aware Visual Token Pruning for Large Vision-Language Models

**arXiv ID:** 2611.04537 | [PDF](https://arxiv.org/pdf/2611.04537v1)

**作者:** Zihao Qi `[一作]` (Nanjing University), Zengqiang Yan `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了训练无关、基于视觉令牌角色演化的裁剪框架BtsPruner；

**💡 创新点**

创新点在于发现视觉令牌在编码过程中从局部细节携带者转为全局代表者的角色演化，并据此设计角色过渡识别、全局代表发现与局部细节递归恢复机制，能够在固定令牌预算下保留补充角色；

**🔧 技术方法**

采用自注意力分布与CLS注意力集中度来识别角色过渡点，通过注意力导向+多样性筛选（Diverse）实现全局代表选取和局部细节递归恢复；该方法无须额外训练；

**📊 数据集**

在十个常用多模态基准（VQAv2、GQA、VizWiz、ScienceQA‑IMG、TextVQA、POPE、MME、MMBench、MMBench‑CN、MMVet）以及多种LVLM架构（LLaVA‑1.5、LLaVA‑NeXT、Qwen‑VL、InternVL、CogVLM）上进行评估；

**📈 对比分析**

与FastV、SparseVLM、DivPrune、VisionZip、VisPruner等训练无关裁剪方法对比，BtsPruner在LLaVA‑1.5‑7B上保留32/576视觉令牌（5.6%）时获得91.6%性能，64/576保留96.4%，128/576保留98.7%；在高分辨率LLaVA‑NeXT‑7B上实现CUDA延迟5.95×加速；

**⚠️ 局限性**

局限性包括需手工设定全局代表比例r，对极端压缩下局部细节恢复仍可能不足，对不同投影设计（如query‑based）需额外适配，且未在训练阶段进一步优化，兼容性仍受限于模型架构。

---

## 410. Future Representation Distillation for Imagination-Free World Action Models

**arXiv ID:** 2611.04527 | [PDF](https://arxiv.org/pdf/2611.04527v1)

**作者:** Jaehyeong Jo `[一作]` (KAIST), Sung Ju Hwang `[通讯]` (KAIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的世界动作模型（WAM），通过将冻结的预训练视频扩散模型的预测知识蒸馏到视觉编码器，从当前观测直接生成预测性视觉特征，实现无视频生成的动作预测。

**💡 创新点**

创新点在于：① 用未来表示蒸馏将预训练视频模型的时间预测能力嵌入单帧视觉特征；② 训练分两阶段，先学习预测视觉表示，再学习动作映射，避免在推理时进行多步视频去噪；③ 通过结构化注意力掩码限制当前帧仅关注自身，实现单前向传递。

**🔧 技术方法**

核心技术包括：预训练视频扩散模型（Wan2.2系列）、Transformer 结构化掩码、未来表示蒸馏（对齐编码器输出与教师模型隐藏层）、生成流匹配目标、动作读取模块（Perceiver 风格的查询）和基于 Diffusion Transformer 的动作解码器。

**📊 数据集**

使用的数据集包括：1) LIBERO（四个任务套装） 2) RoboCasa‑Kitchen（24 任务） 3) DROID（10 任务） 4) 训练视觉编码器时的额外无标注视频：RoboCasa365 原子任务视频、ABC‑130k、RoboTwin 等。

**📈 对比分析**

与现有 VLA（π_0.5）和 WAM（DreamZero）基线对比，模型在模拟环境中取得与最先进方法相当或略优的成功率；在真实世界 DROID 任务中平均成功率提升至 64%（超过 DreamZero 的 56% 和 π_0.5 的 16%）；推理时 11Hz 的实时闭环控制，推理速度比 DreamZero 提升 17.8 倍，延迟仅比 π_0.5 高 1.1 倍。

**⚠️ 局限性**

局限性包括：① 仍需大量无标注视频进行视觉编码器预训练；② 对于需要高精度物理交互或极其细粒度时间动态的任务，预测特征可能不足；③ 任务迁移仍受限于已学原子技能的覆盖范围，无法通过视频单独学习无法执行的原子动作。

---

## 411. LLM-Assisted Planning with Distributed Onboard Behaviour Tree Execution for Multi-AUV Missions

**arXiv ID:** 2611.04518 | [PDF](https://arxiv.org/pdf/2611.04518v1)

**作者:** Ehsan Khorrambakht `[一作]` (University of Pisa), Andrea Caiti `[通讯]` (University of Pisa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个层次化框架，将大型语言模型解释自然语言任务指令与混合整数线性规划以及分布式行为树执行相结合，实现多AUV任务分配与执行。

**💡 创新点**

创新在于将LLM仅用于高层语义约束抽取，并通过MILP保证可行性，再结合分布式行为树实现局部失效自适应与通信中断下的自治执行。

**🔧 技术方法**

技术包括大型语言模型（Qwen‑3‑8B、Llama‑3.1‑8B）、Gurobi求解的MILP、LangGraph多代理架构、Redis消息通信、行为树（BT）及ROS‑Zeno AUV仿真。

**📊 数据集**

使用的实验数据集为在ROS模拟器中构建的三种规模（2/3/5台AUV，8/12/20任务）仿真场景，任务包括航点导航、扫地式与螺旋式调查。

**📈 对比分析**

通过与两种LLM模型的对比和四种失效场景的实验，Qwen‑3‑8B在约束提取准确率上优于Llama‑3.1‑8B，且在所有失效场景下均实现100%任务完成率，恢复时间在10–15 s范围内。

**⚠️ 局限性**

局限在于未在真实水下环境中验证、缺乏对动态海洋扰动的考虑以及LLM在极端约束或多语言指令下的鲁棒性仍待进一步评估。

---

## 412. Sample-Grained Approximate Unlearning with Provable Per-Sample Bounds

**arXiv ID:** 2611.04501 | [PDF](https://arxiv.org/pdf/2611.04501v1)

**作者:** Yaxin Xiao `[一作]` (Hong Kong Polytechnic University), Haibo Hu `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于样本级似然对齐的近似无学习框架SALA，解决近似无学习在样本层面效果不佳的问题。

**💡 创新点**

创新点在于：①引入样本级分布对齐目标与z-score引导损失；②定义InDist判据并给出DNO上界；③提出低成本SMSE估计成员/非成员分布；④通过快速超参数探测与停止判据提升鲁棒性。

**🔧 技术方法**

核心技术包括：近似无学习方法（如SalUn）与SALA插件、样本级z-score对齐损失、InDist覆盖度度量、分布对齐上界理论、分割-合并训练与分段集成估计（SMSE）。

**📊 数据集**

在CIFAR‑10、ImageNet‑1000（分类）和WikiText‑103（文本生成）三个数据集上进行实验。

**📈 对比分析**

与12种基线近似无学习方法对比，SALA在样本级分布差距DNO上降低至0.1365（CIFAR‑10）、0.1268（ImageNet‑1000）、0.1048（WikiText‑103），比原始SalUn的DNO提升≈86.7%；保持或略增准确率/困惑度，鲁棒性显著提升，超参数波动率下降至0.03-0.06。

**⚠️ 局限性**

限制包括：仍需预估每个样本的成员/非成员分布，SMSE虽低成本但估计误差影响上界；在仅能访问忘记集的场景下需单侧约束，效果略逊；对极大规模模型的可扩展性尚未完全验证。

---

## 413. Perseus: A Full-Stack System for Homomorphic Model Inference

**arXiv ID:** 2611.04488 | [PDF](https://arxiv.org/pdf/2611.04488v1)

**作者:** Alessandro Zirilli `[一作]` (Sapienza University of Rome), Emanuele Rodolà `[通讯]` (Sapienza University of Rome)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一个开源 Python 库 Perseus，支持从 CKKS 后端、数据打包到引导（bootstrap）放置的全栈加密模型推理。

**💡 创新点**

在三层栈上分别引入创新：① 在 GPU 核心中动态重建 key‑switch 关键字，降低内存带宽和存储；② 扩展 CKKS 复数槽（complex packing）实现双倍吞吐量；③ 构建基于系数幅值和打包周期的迭代最小割 bootstrap 规划器，平均减少约 30% 的 bootstrap 次数。

**🔧 技术方法**

使用了 Rust/CUDA 编写的 CKKS 加密后端、Python 前端封装、32 位 RNS CKKS 链、GPU 并行 SIMD、迭代最小割算法、Baby-Step‑Giant-Step 和 PCMM/CCMM 矩阵乘法实现，以及自研的密钥重建、复数打包和 bootstrap 规划。

**📊 数据集**

在多种模型和任务上进行评测，包括大型语言模型（如 Sylph）、提示填充（prefill）与单词解码（decode）、句子分类（SST‑2）和图像分类（采用多项式层重训练的图像数据集）。

**📈 对比分析**

与现有闭源/开源框架（Arhe、RoSetta、Sylph、EncryptedLLM 等）在同等硬件（RTX‑PRO 6000、H100、A100）下对比：单词解码速度提升 3.96×；提示填充速度提升 41×；图像推理 21.1 s vs 374.4 s；句子分类 8.6 s vs 43.6 s；准确率与 perplexity 与明文模型基本一致（prefill 100% normalized accuracy，decode perplexity 7.77 vs 7.76）。

**⚠️ 局限性**

仍存在局限：对 GPU 内存与带宽要求高，bootstrap 仍为瓶颈；目前仅支持 CKKS，需手动调优打包与规划；仅在有限的模型和任务上验证，缺乏对更大规模 LLM 的全面评估；系统复杂度较高，非专业用户仍有上手门槛。

---

## 414. ValQID: Learning the Net Value of Semantic Query Groups for Web Device Identification

**arXiv ID:** 2611.04576 | [PDF](https://arxiv.org/pdf/2611.04576v1)

**作者:** Yifan Yang `[一作]` (National University of Defense Technology), Zhiping Cai `[通讯]` (National University of Defense Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出ValQID框架，利用 Web 对象关系图和语义问题组对网络设备的 Web 接口进行自动厂商与类型识别，并使用 DSEJEV 进行高效语义评估。

**💡 创新点**

创新点：① 用关系图将散布在页面不同位置的身份线索聚合成语义问题组；② 通过估计每组的净识别价值并结合执行成本，实现自适应计算分配；③ 采用非生成的 JEV 模型 DSEJEV 取代传统 LLM 生成式推理，显著降低语义评估成本。

**🔧 技术方法**

技术：关系图编码+图注意力；非生成决策模型 JEV（DSEJEV）进行选项概率输出；逻辑回归与交互收益评估用于筛选问题模板；值学习与强化决策确定查询顺序；字符特征分支与融合；LoRA 微调；Qwen3.5 作为教师生成训练样本。

**📊 数据集**

数据集：DevTag（约75k条 HTTP 文本，80个厂商，20种设备类型）与 BLMProbe（约11k 条响应，67个厂商，5 种设备类型）。

**📈 对比分析**

与 DevTag、RoBERTa、BLMProbe 及 Qwen3.5 等基线对比；在 DevTag 上宏观 F1 提升 34.9%，在 BLMProbe 上提升 47.1%；低频厂商宏观 F1 提升 82.1% 及 74.8%；语义查询量下降 44.7%–53.2%，输入 token 降至 42%–64%；DSEJEV 在语义处理速度上比其他评估器提升 3.3–8.6 倍，整体识别吞吐量提升 1.2–1.8 倍。

**⚠️ 局限性**

局限性：需先构建语义条件与模板库，模型对界面更新的适应需要重新训练；在极少样本或全新设备上识别仍可能受限；评估仅基于公开数据集，未检验在更大规模或对抗性场景下的鲁棒性。

---

## 415. Setting the Stage: Enhancing Music Recommendations with Spoken Introductions to Stimulate Serendipity

**arXiv ID:** 2611.04570 | [PDF](https://arxiv.org/pdf/2611.04570v1)

**作者:** Brett Binst `[一作]` `[通讯]` (imec-SMIT, Vrije Universiteit Brussel), Brett Binst (imec-SMIT, Vrije Universiteit Brussel)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过大规模用户实验，测试了沉浸式、信息型和无介绍三种口语介绍对音乐推荐中品味拓宽与品味加深两种喜感的影响。

**💡 创新点**

创新点在于首次将口语介绍与西尔维亚兴趣评估理论相结合，揭示运输（沉浸）和认知阐释是分别激发品味拓宽与品味加深的心理机制，并系统比较两种介绍效果。

**🔧 技术方法**

方法上采用自建 Web 应用收集用户听歌与问卷数据，并用贝叶斯层级逻辑回归与结构方程模型对引入类型、运输、认知阐释等变量进行建模，同时使用 Gemini 2.5 TTS 生成介绍语音。

**📊 数据集**

实验数据来自 452 名参与者的 18 首歌曲（六类音乐，每类三首），每首歌曲随机配以三种介绍方式，共 27 条实验条件。

**📈 对比分析**

结果表明，信息型介绍对品味加深的提升尤为显著（OR≈8.4），沉浸式介绍亦能提升两种喜感；通过路径分析证实运输预测品味拓宽、认知阐释预测品味加深，模型拟合良好。

**⚠️ 局限性**

局限性包括样本歌曲有限、实验环境较为人为、仅测量即时体验未跟踪长期兴趣、引入的质量和真实性可能影响效果，且大规模实现需解决自动生成和语音自然度问题。

---

## 416. A research world model of power engineering learns its dynamics and reveals what-ifs

**arXiv ID:** 2611.04566 | [PDF](https://arxiv.org/pdf/2611.04566v1)

**作者:** Fanping Sui `[一作]` (University of California Berkeley), Liwei Lin `[通讯]` (University of California Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个研究世界模型，对电力工程领域的概念增长、方法–目标新链接进行预测，并对各种干预措施给出什么-if 结果。

**💡 创新点**

创新点在于将科研领域视为可学习的动力系统，使用类型化超图与时间图神经网络学习转移函数，并通过自然实验验证干预操作器，实现对干预后轨迹的反事实仿真。

**🔧 技术方法**

采用了稀释语言模型读者、增长与链接核心的时间图神经网络、干预、实体插入与观测同化操作器、以及细调的大语言模型进行问题路由和答案生成。

**📊 数据集**

使用了约120万篇电力工程论文的摘要与引用网络（共1550万条引用事件），以及七个相关领域的5850万篇论文作为外部供应层，经过去重和清洗后构成训练集。

**📈 对比分析**

与调参的LightGBM、MLP和零样本LLM基线相比，模型在概念增长的Spearman相关性上超越所有基线，在新链接的平均精度上提高约2–5%，在干预效果预测上实现了约0.48–0.56的相关系数，并在多项测试中达到显著优势。

**⚠️ 局限性**

局限性包括对效果大小的低估、主要依赖高频概念、仅对电力工程有效、对非方法–目标链接仅预测首个共现、对新词的预测存在缺口、以及对答案可靠性仅依赖单一审计模型。

---

## 417. Distributionally Robust Search: The Power of Information and the Limits of Moments

**arXiv ID:** 2611.04559 | [PDF](https://arxiv.org/pdf/2611.04559v1)

**作者:** Nicole Megow `[一作]` (University of Bremen), Jose Verschae `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在只知道目标分布的有限摘要（如矩、分位数、线性测量或近似CDF）下，如何在有序数组中以期望查询次数最小化的二分搜索问题。

**💡 创新点**

揭示了矩（前k阶）与一般线性测量在搜索效率上的根本差异：即使k≈n^{1−δ}，仅靠矩仍无法获得常数近似，而只需O(n^{log⁴ n})个精心选取的线性测量即可实现O(1/)-近似；同时给出了信息量与近似因子之间的最优下界。

**🔧 技术方法**

主要技术包括分布不确定集合框架、Kolmogorov距离与Cauchy–Schwarz 近似、线性规划与其对偶（多项式逼近、Rakhmanov与Bernstein不等式）、Johnson–Lindenstrauss投影、Dyadic分解、以及分布矩匹配构造。

**📊 数据集**

该工作为理论分析，未使用任何实测数据集；所有结论均通过严谨的数学证明与构造得到。

**📈 对比分析**

通过定义“可逼近因子”α(𝒰)来衡量仅凭摘要可达到的期望查询次数与完全已知分布时的最优值之比；证明了不同摘要（分位数、矩、线性测量、Kolmogorov球）在该指标上的上界与下界，并给出量化的常数和阶数。

**⚠️ 局限性**

主要局限在于：对于矩类摘要，若未满足点质量小或支集小等额外结构假设，则无论矩阶多高（甚至达到n-1），在搜索中仍可能失效；此外，证明多依赖于对离散域[-1,1]的多项式逼近理论，可能对连续或非均匀空间的推广存在挑战。

---

## 418. Inter-Agent Influence: Evaluating Persuasion, Deception, and Coercion in Multi-Agent Systems

**arXiv ID:** 2611.04556 | [PDF](https://arxiv.org/pdf/2611.04556v1)

**作者:** Chandler Smith `[一作]` (University of Oxford), Lewis Hammond `[通讯]` (Cooperative AI Foundation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估前沿LLM在五种现实多代理环境中使用说服、欺骗和胁迫三种影响手段的效果，展示这些能力能显著改变监督者批准率、谈判结果和研究方向。

**💡 创新点**

首次将三种影响能力明确定义并操作化，构建并公开五个真实多轮交互环境，系统量化LLM在跨模型、跨场景下的影响力。

**🔧 技术方法**

使用前沿LLM（如GPT‑4、Claude等）结合Inspect框架、工具提示、LLM评估器及自评探测器等技术实现交互、影响与自动化评估。

**📊 数据集**

基于五个真实环境的参考材料（GDPR、代码审查、研究论文等）构建约52,000+有效交互记录，作为实验数据集。

**📈 对比分析**

通过Q1（条件影响效果）、Q2（提示整体效果）和CACE估计方法进行对比，发现欺骗产生最大提升（约+35pp），劝服次之（约+18pp），胁迫最小（约+6-15pp），所有能力在多数环境中均显著提高目标决策。

**⚠️ 局限性**

局限包括样本量相对有限、仅评估二人交互、胁迫环境假设不够现实、评估者意识可能影响结果、无法精确分离单一技术手段的独立贡献等。

---

## 419. Are LLM Assistants Regulatory Ready? Answers from regenold's 2026 EU AI Act Q&A Challenge

**arXiv ID:** 2611.04546 | [PDF](https://arxiv.org/pdf/2611.04546v1)

**作者:** Marco Virgolin `[一作]` (Unlayer AI), Maximilian Regenold `[通讯]` (regenold GmbH)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并公开了一个基于欧盟AI法案的专家裁定问答基准，举办挑战赛评估LLM助理的答案准确性、引用质量、简洁性、语气和响应速度，并与通用模型进行对比。

**💡 创新点**

首次将开放式问题、专家制定的正确性标准和多指标评估相结合，并采用LLM-as-judge进行判分，引入硬模式（多轮对话和用户推诿）来检验系统在更逼真环境下的表现。

**🔧 技术方法**

使用OpenAI GPT、Anthropic Claude等大型语言模型，结合检索增强生成（RAG）与Web搜索工具，并通过LLM-as-judge进行判定与多指标评估（答案准确率、引用准确率、简洁度、语气、速度）。

**📊 数据集**

构造了110道专家手工编写的开放式问题集，包含示例答案、正确性标准、预期引用和难度等级，覆盖2024/1689号欧盟AI法案原文本。

**📈 对比分析**

在挑战赛中，给定API无前置问题，评测5个专门系统与3个基线，采用宽松/严格答案准确率、引用准确率、简洁度、语气、速度等指标，并用几何平均汇总；最佳系统整体得分约90%，优于大多数基线，性能差距约30个百分点。

**⚠️ 局限性**

局限性包括：单一作者标注、仅覆盖原始法文本、LLM-as-judge仅使用一种模型、未直接检测幻觉或基准更新性、硬模式不够严格、专业语气评估分辨率有限。

---

## 420. Coercive at the equilibrium, monotone nowhere: exact convergence certificates for learning in first-price auctions

**arXiv ID:** 2611.04545 | [PDF](https://arxiv.org/pdf/2611.04545v1)

**作者:** Benjamin Lubin `[一作]` `[通讯]` (Boston University), Benjamin Lubin (Boston University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究第一价格拍卖在多边际学习中，构造了可验证的收敛证书，替代传统的单调性条件；

**💡 创新点**

首次给出在任意分段线性逼近空间下门槛收敛的显式常数、Minty 半径与极值方向，并证明对任意先验的闭式权重；

**🔧 技术方法**

利用变分不等式、Gateaux导数、Minty身份、潜在函数、两边式子、两周期Bloch不等式以及 Lean4 形式化验证；

**📊 数据集**

无数据集，纯理论推导；

**📈 对比分析**

没有经验对照实验，主要通过理论证明与计算机验证（Lean4）来证明收敛性和常数精确性；

**⚠️ 局限性**

局限性包括对五名及以上竞标者的节点质量解析尚未完全证明、Minty半径常数的上界缺失、以及在更大维度下的收敛正定性验证等。

---

## 421. The Price of List Replicability: Tight Bounds for Bias Estimation

**arXiv ID:** 2611.04532 | [PDF](https://arxiv.org/pdf/2611.04532v1)

**作者:** Michael Chen `[一作]` (Iowa State University), N. V. Vinodchandran `[通讯]` (University of Nebraska-Lincoln)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `2704f255-0c84-4173-b83c-0e9a3dbea232` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了列表可复制算法估计 d 个硬币偏差的样本复杂度，提出了最优的上界与下界，并给出了相应的算法。

**💡 创新点**

首次给出了非平凡的下界 Θ(d²/ε²)，并将先前的 O(d³/ε²) 上界大幅改进；此外对任意列表大小 k∈[d+1,2ᵈ] 提供了匹配的 Θ(d²/ε²·log k) 上下界。

**🔧 技术方法**

利用信息论下界、对偶分析、概率论工具以及分层随机采样构造近似最优算法。

**📊 数据集**

论文为理论工作，无实验数据集。

**📈 对比分析**

通过严格的理论证明与先前的 O(d³/ε²) 上界对比，得到上界与下界仅相差多项式对数因子，证明了近似最优的样本复杂度。

**⚠️ 局限性**

仍存在多项式对数的细微差距；缺乏实验验证，算法实现与实际性能未被检验；适用范围限定于列表可复制框架。

---

## 422. Learning to Falsify: Hierarchical Multiple Instance Learning for High Resolution AI-Generated Image Detection

**arXiv ID:** 2611.04529 | [PDF](https://arxiv.org/pdf/2611.04529v1)

**作者:** Haoran Chen `[一作]` (Zhejiang Gongshang University), Jianfeng Dong `[通讯]` (Zhejiang Gongshang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了基于“反证”原则的多层次多实例学习框架LTF，用于高分辨率AI生成图像检测，并提出无须重新训练的FATAS聚合策略提升现有检测器；

**💡 创新点**

将单一伪造区域足以否定整幅图像真实性的“反证”原则引入检测范式，通过Cross-Shuffle和分层MIL实现对局部伪造痕迹的精准挖掘；

**🔧 技术方法**

采用冻结CLIP‑ViT编码器、Cross‑Shuffle实例重组、分层MIL聚合、跨层融合与门控加权等技术；

**📊 数据集**

在GenImage、DRCT、Chameleon和HiRes‑50K四大公开高分辨率生成图像数据集上训练与评估；

**📈 对比分析**

与多种单裁剪与多裁剪检测器对比，LTF在跨数据集和高分辨率任务中取得最佳准确率（最高81.43%），FATAS对现有模型提升20%+，且不需重新训练；

**⚠️ 局限性**

计算开销相对较大，单图推理速度慢于单裁剪方法，且在极端下采样或模糊条件下仍受限。

---

## 423. Distributed LTL Synthesis under Arbitrary Message Adversaries

**arXiv ID:** 2611.04561 | [PDF](https://arxiv.org/pdf/2611.04561v1)

**作者:** Benedikt Bollig `[一作]` (Université Paris-Saclay), Nathalie Sznajder `[通讯]` (Sorbonne Université)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文针对在两进程、以轮次交互且存在不可靠消息链路的分布式合成问题，给出了判定 LTL 规范可实现性的理论框架，并证明在一般情况下该问题是不可判定的；同时对可判定的通信约束给出完整表征，提出基于视图复合体与迭代色子细分的拓扑方法，构造可执行算法并实现决策与算法合成。

**💡 创新点**

创新点在于：①使用拓扑映射 π(A) 将消息不可靠历史映射到区间 [0,1]，从而把可判定性与 π(A) 的稠密性/无稠密性对应；②首次将可判定性与重复可在有界时间内求解共识的能力关联；③提出一种树自动机方法，能够在 ω‑正则消息对手下有效判定并构造实现算法；④提供完整的 Lean 4 形式化证明，验证所有核心命题。

**🔧 技术方法**

核心技术包括：视图复杂体（view complex）和连通分支的图论分析；迭代色子细分与区间映射 π；对消息对手的 ω‑正则自动机表示；LTL 与 MSO 之间的归约；树自动机非空判定；以及对 2‑计数器机的不停化简与编码。

**📊 数据集**

本工作属于理论计算机科学，不使用实验数据集；所有结果均为形式化证明与理论构造。

**📈 对比分析**

与先前工作（Pnueli & Rosner 的不可判定性、管道架构的可判定性、信息层次化、知识基础规范等）对比，本文提供了更一般的可判定性判据，并给出了统一的决策与实现算法。实验性能未给出，但判定程序的复杂度理论上在 ω‑正则消息对手下是可执行的。

**⚠️ 局限性**

局限性包括：仅适用于两进程系统；仅适用于 ω‑正则消息对手；需要假设每轮至少有一条消息成功；算法与判定复杂度未明确给出；若对手不是 ω‑正则或存在更复杂的不可靠通信模式，方法不适用。

---

## 424. Order-Dependent Complexity of Enumeration and Direct Access for Conjunctive Queries on Dynamic Data

**arXiv ID:** 2611.04519 | [PDF](https://arxiv.org/pdf/2611.04519v1)

**作者:** Antoine Amarilli `[一作]` (LTCI, Télécom Paris, Institut polytechnique de Paris), Clémence Dumoulin `[通讯]` (Inria, LIRMM, Univ. Montpellier, CNRS)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在动态数据库中，对自连接自由变量的投影式联结查询（self‑join‑free, q‑hierarchical）进行按字典序（lexicographic）排序的答案的枚举和直接访问，并给出了在更新时仍保持高效的算法。

**💡 创新点**

创新点在于：①提出了“hierarchy‑consistent”这一新定义，用以判定何种字典序能够在动态环境下维持低延迟/低访问时间；②给出了针对该类顺序的线性预处理、对数更新、常数延迟（枚举）或对数访问时间（直接访问）的算法；③证明了若顺序非 hierarchy‑consistent，则在OMv‑假设下，连对数级别的直接访问都不可能实现，从而在动态排序查询领域给出新的上界。

**🔧 技术方法**

技术方法包括：构造 q‑tree、view‑tree、grounded‑tree；利用 B⁺‑树实现桶内按字典序排序；为每个桶维护权重和前缀和以支持直接访问；在更新时沿树路径进行插入/删除并更新权重；使用 Lawler‑Murty 方案的动态版本实现非 hierarchy‑consistent 情况下的枚举（未来工作）。

**📊 数据集**

本文为理论研究，没有使用具体实验数据集，所有结果均在抽象的组合复杂度和数据复杂度框架下给出。

**📈 对比分析**

与之前的非排序、非动态结果相比，本工作在 q‑hierarchical 查询上保持了相同的预处理与更新复杂度；在排序查询上，常数延迟/对数访问时间与已知的最优结果相匹配；而下界证明表明对非 hierarchy‑consistent 顺序无法实现同样的效率，进一步阐明了排序动态查询的可行性边界。

**⚠️ 局限性**

限制包括：①仅针对 q‑hierarchical、无自连接的查询；②对非 hierarchy‑consistent 顺序下的枚举仍未给出完整上界，仅提出可能需要对数延迟；③实现细节（如 Lawler‑Murty 变体）和实验验证仍待后续工作。

---

## 425. Variational Learning with Lie Group Exponential Families

**arXiv ID:** 2611.04513 | [PDF](https://arxiv.org/pdf/2611.04513v1)

**作者:** Eren Mehmet Kıral `[一作]` (RIKEN Center for Advanced Intelligence Project), Kenichi Bannai `[通讯]` (Keio University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

提出了一种基于李群的变分学习框架（LG-BLR），并证明其在线性化时退化为传统的贝叶斯学习规则（BLR），同时将该框架推广到任意李群并引入谐波指数族（harmonic exponential families）来描述在李群作用下封闭的指数族分布。

**💡 创新点**

创新点包括：
1) 统一了李群更新与指数族自然梯度更新的理论，给出二者在开放指数族下的等价性；
2) 定义并利用谐波指数族作为构造满足李群作用的指数族分布的系统方法；
3) 通过李群的结构自然满足分布参数约束，自动提供低方差梯度估计，并支持动量等优化技术。

**🔧 技术方法**

使用的技术主要有：
- 李群与李代数理论；
- 指数族与信息几何中的自然参数、期望参数与费舍尔信息度量；
- 群作用、相对不变测度与1-余 cocycle 的构造；
- 线性化近似与指数映射的Taylor展开；
- 通过拉格朗日对偶与梯度投影实现优化。

**📊 数据集**

本工作为理论研究，未在公开数据集上进行实验验证，因而没有使用特定数据集。

**📈 对比分析**

比较方法主要是理论推导：在开放指数族与李群可传递作用下，LG-BLR 的线性化更新与 BLR 的自然梯度更新在数学上完全相同；在具体实例（如多元高斯、对数正态、Rayleigh 等）中，通过数值近似可观察到与 BLR 结果一致，并且在保持参数有效性的同时实现了更高效的梯度估计与动量积累。

**⚠️ 局限性**

局限性包括：
- 需要李群对参数空间的平滑且可传递作用，限制了可应用的分布族；
- 对于曲面指数族，线性化可能导致与 BLR 的偏差，需要投影或更高阶近似；
- 实际实现中仍需手动推导特定李群的 adjoint、指数映射等，降低了可复用性；
- 论文主要聚焦理论，缺乏大规模实验评估其在深度网络训练中的实际性能提升。

---

## 426. Theory-guided relational learning for data product pricing: An interpretable meta-path-guided graph neural network approach

**arXiv ID:** 2611.04508 | [PDF](https://arxiv.org/pdf/2611.04508v1)

**作者:** Ruize Gao `[一作]`, Shaoze Cui `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种基于理论引导的关系学习框架（MPGNN），通过构建异构信息网络、设计品牌、相似度、行业结构等经济理论驱动的 meta‑path，并结合大型语言模型生成的文本特征，完成数据产品定价的预测与机制解释。

**💡 创新点**

创新点在于：①将数据产品定价转化为关系学习问题；②设计多种理论驱动的 meta‑path 并在 Heterogeneous Graph Transformer 中分层编码；③引入自注意力门控融合多路径表示，实现可解释的路径权重；④采用两阶段 MoE 风格软路由回归，处理长尾价格分布；⑤利用 LLM 生成结构化文本特征提升预测精度。

**🔧 技术方法**

技术手段包括：异构图变换器（HGT）+ Meta‑path 限制消息传递、注意力门控融合、LASSO + 熵正则的路径权重学习、LLM 提取文本特征、两阶段软路由混合专家回归（MoE）以及对比学习用于结构正则化。

**📊 数据集**

实验使用三大数据商品交易平台（AWS Marketplace、DataRade.ai、Alibaba Cloud Data Exchange）收集的数据产品，涵盖订阅、一次性购买和按请求计费三种定价模式，构建异构图并提取文本、属性与关系特征。

**📈 对比分析**

与传统机器学习（LR、ANN、XGBoost）、深度学习、LLM（LoRA‑tuned Llama、Qwen 等）、图表示学习（DeepWalk、GAT 等）等多类基线在三折交叉验证下进行比较；MPGNN 在分类准确率、F1、AUC 以及回归 MAE/RMSE/R² 等指标均显著优于所有基线，提升幅度约 15%–30%，且差异均在统计上显著。

**⚠️ 局限性**

局限性包括：依赖手工设计的理论驱动 meta‑path，缺乏自动路径发现；LLM 文本特征可能带有提取误差；模型结构复杂，训练成本高；仅在定价模式层面验证，跨行业泛化和实时动态定价的适用性仍待进一步研究。

---

## 427. Polarimetric Scattering-Structure Contrastive Learning for Few-Shot HRRP Recognition

**arXiv ID:** 2611.04490 | [PDF](https://arxiv.org/pdf/2611.04490v1)

**作者:** Fangyi Wang `[一作]`, Haipeng Wang `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于极化散射结构对比学习的PSS-CL模型，用于少样本高分辨率波段特征（HRRP）的识别

**💡 创新点**

创新点在于构建结构化极化散射视图并通过跨视图对比将物理结构信息迁移至HRRP编码器，同时加入面向角度的监督对比正则化

**🔧 技术方法**

使用ResNet‑12 1‑D编码器、两层MLP投影头、跨视图对比损失、面向角度的监督对比损失以及支撑样本的原型学习

**📊 数据集**

使用真实测量的六类目标HRRP数据集（共56,356条，包含HH、HV、VV三个极化通道，1024长、10 GHz、120 MHz带宽）

**📈 对比分析**

与8种通用少样本学习方法和4种HRRP专用方法对比，PSS‑CL在随机角度和跨角度评估中均位列第一，1/3/5-shot下的平均分类准确率分别为81.73%、86.92%、90.98%，在跨角度1-shot相较最佳竞争方法提升5.62%

**⚠️ 局限性**

受限于仅在测量数据上验证，模型对其他雷达频率或合成数据的泛化能力未知；对超参数（R、λ_sc、η等）敏感，且训练时需要极化物理分支，推理阶段需额外训练开销

---

## 428. Balanced Fair Division and Generalized Kneser Graphs

**arXiv ID:** 2611.04486 | [PDF](https://arxiv.org/pdf/2611.04486v1)

**作者:** Sharareh Alipour `[一作]`, Amir Jafari `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究在两预先划分的非空团之间分配不可分物品的公平分配问题，并通过与广义Kneser图的色彩数之间的对应关系，给出了在平衡大小限制下的EF1（最多5个代理）和EF2（最多9个代理）存在性阈值，同时证明了Jafari‑Alipour关于χ(G_n,2)=6的猜想，并对χ(G_n,3)给出10下界。

**💡 创新点**

创新点主要包括：① 将公平分配问题与广义Kneser图的色彩数精确对应；② 证明χ(G_n,2)=6（r=2）和χ(G_n,3)≥10（r=3），从而确定5人EF1和9人EF2的阈值；③ 通过新的五约束舍入定理和局部颜色约束，完成对第七同调类的生成与消除，突破传统的Box‑Complex指数上限；④ 构造线性时间的平衡收缩映射，使非平衡的色彩理论结果能直接应用到平衡分配；⑤ 证明在给定阈值下，平衡EFc搜索问题与Kneser图的单色边搜索等价，揭示其在TFNP与PPA中的位置。

**🔧 技术方法**

主要技术包括：拓扑同调（Box‑Complex、β‑指数、同调映射不为零的判据）；组合几何（Kneser图、约束舍入、棘角图和五边形/棋盘配置）；离散几何（超立方体、坐标面模型、凸包填充）；图论与色彩理论（颜色映射、邻接关系、图同态、收缩映射）；以及算法与复杂性（线性时间收缩、搜索问题等价与多项式时间归约）。

**📊 数据集**

论文未使用传统机器学习或实验数据集，主要以纯理论和组合实例（如n=4、n=5等）为验证。

**📈 对比分析**

由于是存在性阈值的理论证明，没有实验性能指标；相对已有结果，本文将原来的5人EF1阈值提升到5人（此前仅4人），将原来的5人EF2阈值提升到9人，证明了对应色彩数的上界，并给出更紧的下界（χ(G_n,3)≥10）。

**⚠️ 局限性**

局限性：
- 对于r≥4（即更大c）的色彩数和对应的EFc阈值仍未确定；
- 对于大于9个代理的EF2以及更高c的EFc仅给出不完整的上界；
- 计算复杂性方面，虽然给出了搜索等价性，但尚未给出多项式时间算法或PPA/TFNP的完整归类；
- 证明依赖于特定的五约束舍入定理，对更大约束数不适用。

---

## 429. All-For-One: Distilling Visual Foundation Models to a Fast, Memory-Efficient Visuomotor Policy

**arXiv ID:** 2611.04485 | [PDF](https://arxiv.org/pdf/2611.04485v1)

**作者:** Byungkun Lee `[一作]` (KAIST), Jaegul Choo `[通讯]` (KAIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

未知

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

## 430. Curriculum unlearning in humans and neural networks

**arXiv ID:** 2611.04565 | [PDF](https://arxiv.org/pdf/2611.04565v1)

**作者:** Jirko Rubruck `[一作]` (University of Oxford), Christopher Summerfield `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过在 207 名人类参与者和 1027 名神经网络模型上进行规则+例外（R+E）分类任务的实验，探讨了从简单到复杂的课程学习（curriculum）对学习效率和内部表示压缩的影响。

**💡 创新点**

创新点在于提出并验证“课程去学习”（curriculum unlearning）理论，解释课程学习通过帮助学习者在已压缩的低维表示中恢复被压缩的维度，从而更好地学习例外规则；并首次将人类行为与深度网络的表示动态进行直接对应。

**🔧 技术方法**

研究使用了策略分类、反应时分析、梯度下降的两层前馈神经网络（加入隐藏层噪声）、参与度比（participation ratio）等技术，对人类与网络的学习过程进行建模与比较。

**📊 数据集**

数据集为 8 种由三元布尔特征构成的合成“bug”刺激，分为 6 条规则（rule）与 2 条例外（exception）两类，实验共收集了 207 名人类和 1027 名人类/网络样本，且每个网络与对应人类的刺激序列完全匹配。

**📈 对比分析**

通过比较“简单→复杂”“复杂→简单”“无课程”三种呈现顺序，评估准确率、策略转移概率和 RT 成本；结果显示，简单→复杂课程显著提升例外分类准确率（约 20%）且网络表现高度一致，且在预训练为 1‑维规则的情况下效果更明显。

**⚠️ 局限性**

局限性包括：模型仅采用两层感知机和人工噪声，缺乏对大脑真实结构和学习机制的完整描述；任务仅为合成二分类，难以直接推广到更复杂的真实世界任务；课程效益虽显著但受样本顺序和记忆衰退的影响。

---

## 431. Predicting Void Points in Self-Occluding Forest Volumes

**arXiv ID:** 2611.04548 | [PDF](https://arxiv.org/pdf/2611.04548v1)

**作者:** Benjamin Bachmayr `[一作]` (Johannes Kepler University Linz), Oliver Bimber `[通讯]` (Johannes Kepler University Linz)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建了基于Gazebo的森林模拟数据集，并将空洞空间标记为负类，通过光场渲染生成550层焦点堆栈，训练并比较了多种U‑Net架构以预测自遮蔽森林体积中的空洞点。

**💡 创新点**

创新点在于将空洞空间从缺失信息改为明确的负类标签，避免传统深度学习在缺少空洞锚点时失效，同时采用光场渲染与合成近红外图像的组合，显著提升了空洞检测精度。

**🔧 技术方法**

使用了光场渲染、Pix2Pix生成近红外合成、2D/3D U‑Net、COLMAP、DepthAnything3等技术。

**📊 数据集**

使用Gazebo生成的80–200棵树、三种树种的仿真森林数据，合成的RGB+NIR焦点堆栈；无真实测量体积数据，仅做人工合成。

**📈 对比分析**

在保留测试场景上，2D U‑Net达IoU 0.6290，3D U‑Net 0.5707；RGB输入的3D U‑Net达0.6514；COLMAP 0.3446，DepthAnything3 0.2959，表明所提方法显著优于现有重建方法。

**⚠️ 局限性**

局限在于缺乏真实空洞体素标注，无法量化对真实无人机数据的迁移性能；方法只关注空洞检测，未解决其他三维重建挑战。

---

## 432. Optimized size, shape and orientation of rhombic transposition flaps and excisions by gradient-based methods

**arXiv ID:** 2611.04563 | [PDF](https://arxiv.org/pdf/2611.04563v1)

**作者:** Niels Aage `[一作]` (Technical University of Denmark), Brandon Marshal Parks `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用梯度优化方法，对菱形移位皮瓣及其切除区域进行设计，找到在预拉伸、可皱折的Ogden膜模型下能最小化手术增能的几何参数组合；

**💡 创新点**

创新点在于将切口、凿口及缝线用平滑水平集与固定网格表示，实现不需要重网格的连续设计空间，并使用对偶敏感性求解八个几何变量与32个约束的优化问题；

**🔧 技术方法**

使用的技术包括：固定网格水平集法、线性弹性与非线性Ogden弹性结合的薄膜模型、改进的缝线弹簧模型、对偶敏感性分析、MMA（移动不对称方法）优化器；

**📊 数据集**

使用的“数据集”是基于前一项研究的48种手术设计（角度0°–70°，步长10°）的结果作为验证基准，且采用了单个直径8 mm、边缘3 mm的病灶；

**📈 对比分析**

与前研究的48种手术设计比较，模型在能量评估上中位偏差仅5%，并保留了设计排名；优化后得到的最优菱形皮瓣能将能量降低约78%（相较于经典Limberg皮瓣），并能在满足缝线张力约束的同时仅增加2.3%的能量；

**⚠️ 局限性**

局限性包括：仅考虑二维平面膜模型（无法捕捉犬耳上升的三维变形）、假设皮肤各向同性、单一病灶大小、固定缝线比例与缝线弹簧模型、固定的切口宽度和剥离区域大小、以及未对血供与愈合过程建模等。

---

## 433. Stabilizing Neural Emulation of Spatiotemporal Physical Dynamics through Denoising Koopman Latent Dynamics

**arXiv ID:** 2611.04642 | [PDF](https://arxiv.org/pdf/2611.04642v1)

**作者:** Chenxu Wu `[一作]` (Southeast University), Jian Wei You `[通讯]` (Southeast University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出KoopFlow框架，结合有限维Koopman预测与生成式校正，稳定长时序预测

**💡 创新点**

将有限维Koopman误差视为随时间递进的分布退化，并通过生成式逆桥回归校正；同时发现Koopman谱共振与预测准确性呈非单调关系

**🔧 技术方法**

连续时间Koopman学习、线性演化算子、生成式逆桥（Brownian桥）校正、条件噪声网络、FNO/DeepONet编码器/解码器

**📊 数据集**

Navier–Stokes、Reaction–Diffusion、2DHeat、2D电磁散射四个数值模拟数据集

**📈 对比分析**

与DeepONet、FNO、KNO、InfoKoop、MERLIN、LoLA、CFO等基线对比，在训练区间内表现优异或相当；在时间OOS区间显著提升，平均降低NMSE超过50%，并在参数泛化任务中取得最小误差

**⚠️ 局限性**

对退化分布的理论假设在高度非线性或极大模型规模时可能不足；生成校正过程仍需额外计算，且在多尺度高分辨率场景下需要更细化的桥段配置

---

## 434. ReXMo: End-to-End Video Character Replacement with Expression and Motion Preservation

**arXiv ID:** 2611.04633 | [PDF](https://arxiv.org/pdf/2611.04633v1)

**作者:** Hualiang Wei `[一作]` (Jilin University), Wenhui Li `[通讯]` (Jilin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 ReXMo 框架，实现高保真的视频角色替换，能够在单人、多人和选择性替换场景下保持面部表情、动作与场景动态。

**💡 创新点**

创新点在于：① 引入配对三元组监督（同一表演、不同身份）实现直接身份转换；② 采用 In-Context Diffusion Transformer 融合驱动视频、参考身份与文本提示；③ 设计 Expression Enhancement Adapter，专门提升多角色场景下的细腻面部表情。

**🔧 技术方法**

主要技术包括：Wan2.2 预训练扩散模型、Motion LoRA 参数高效微调、Face Adapter 插件、Motion Extractor 与 Face Encoder 生成表情令牌，以及双阶段训练策略。

**📊 数据集**

使用自制的合成配对数据集（Stage‑I：单人/多人/说话头视频；Stage‑II：30K 说话头面部序列），训练集为 97 帧、24FPS 的视频片段。

**📈 对比分析**

与 Wan‑Animate、MultiAnimate、SCAIL、SCAIL‑2 等基线在视频‑Bench 及合成交叉身份基准上对比，ReXMo 在动作一致性、身份保真、视觉质量和表情保留指标上均居首，取得显著提升。

**⚠️ 局限性**

局限性：依赖大规模合成数据，真实多样性有限；在极端遮挡或非常小人脸的复杂多角色场景中，表达细节仍可能受限；对推理时计算资源和显存需求较高。

---

## 435. Policy Network Diffusion

**arXiv ID:** 2611.04614 | [PDF](https://arxiv.org/pdf/2611.04614v1)

**作者:** Jiate Liu `[一作]` (Tencent), Deheng Ye `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 DPPG 框架，通过 Transformer 自编码器与任务条件扩散模型实现对多种强化学习任务的策略网络参数生成。

**💡 创新点**

首次将扩散模型直接用于生成 RL 策略参数，并统一一个模型覆盖多域任务，支持大规模参数与无监督生成。

**🔧 技术方法**

使用 Transformer 自编码器、latent diffusion、任务标签编码、参数 patch 化、随机噪声数据增强、PPO 预训练、Trajectory VAE 等技术。

**📊 数据集**

使用 Classic Control、Atari 20 游戏、MuJoCo 5 任务等环境，采用 PPO 收敛后细分 200 个 checkpoint 作为训练样本，并利用 Expert 轨迹作为条件。

**📈 对比分析**

与 PPO 基线和 pdiff 进行对比，在单任务和跨任务场景下，DPPG 在大多数环境中取得与 PPO 相当或更高的奖励，并在大参数网络下显著优于 pdiff，内存占用约为 pdiff 的 1/100。

**⚠️ 局限性**

生成参数质量受数据量和分布限制，直接生成的策略仍需微调；大参数训练成本高，需进一步提升模型效率。

---

## 436. Can Network Models of Science Do Without Pluralist Agents?

**arXiv ID:** 2611.04590 | [PDF](https://arxiv.org/pdf/2611.04590v1)

**作者:** Jonas Hallgren `[一作]` (Equilibria Network), Mahault Albarracin `[通讯]` (Université du Québec à Montréal)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过构建多智能体变分推断模型，对比了理论多元主义和单一主义科学社区在面对竞争理论时的信念更新和沟通机制。

**💡 创新点**

创新点在于引入了两层信念表示（对每个候选理论的条件信念和对理论的支持度），并通过实验揭示支持信息与条件预测信息的不同传播效应。

**🔧 技术方法**

主要技术包括贝叶斯网络建模、协同变分推断、对数线性社交学习规则和参考锚定等。

**📊 数据集**

使用了基于假设的燃烧实验（磷戈斯顿与氧气理论）的合成高斯观测数据。

**📈 对比分析**

与单一高斯基准和不同沟通渠道的组合进行比较，结果显示共享理论支持能更快促进理论转移且提升预测准确性；共享条件信念提升预测误差但不如共享支持。

**⚠️ 局限性**

局限性包括候选理论与观测安排预设，智能体不自我产生或改进理论结构，实验仅覆盖两种理论，未检验内部结构复杂度对结果的影响。

---

## 437. Lower Bounds for Adaptive Relaxed Locally Decodable Codes via the Capped Horvitz--Thompson Estimator

**arXiv ID:** 2611.04678 | [PDF](https://arxiv.org/pdf/2611.04678v1)

**作者:** Guy Goldberg `[一作]` (Weizmann Institute of Science), Sidhant Saraogi `[通讯]` (Georgetown University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了任意 q‑查询的松弛局部可解码码（RLDC）的码长下界几乎最优，即 n = Ω(k^{1+Ω(1/q)})，并将此下界扩展到非线性编码与双向错误的适配性解码器。

**💡 创新点**

创新点包括：①提出“采样式解码器”这一信息论框架，①提出并分析了新的“截断 Horvitz–Thompson 估计器（Capped Horvitz–Thompson Estimator）”，该估计器在自界函数框架下具有良好集中性；②将 spreadness 提取技术与 CHT 估计结合，克服了传统 Janson 不足的上尾集中问题，从而在适配性、双向错误环境下获得更强的下界。

**🔧 技术方法**

主要技术手段：信息论（Fano 不等式）→采样式解码器；多项式/自边函数的离散导数与自界函数集中不等式；Horvitz–Thompson 估计与其截断版 CHT；spreadness 提取与分布去重；决策树模型与局部查询分布的分析；组合概率论（Janson、Chernoff 等）。

**📊 数据集**

无实验数据集，全部为理论证明与组合/概率分析。

**📈 对比分析**

相较于先前最优的上界 n = O(k^{1+O(1/q)})（Ben‑Sasson 等），本文给出下界几乎匹配该上界；对常数查询 q 的情形证明 n = Ω(k^{1+Ω(1/q)})，在常数率与常数查询两种极限下都实现了最优阶数，显著提升了对 RLDC 的理解。

**⚠️ 局限性**

局限性：仍保持在渐进性理论层面；对 α+β>1 的完整性与声音性参数假设；下界仍以常数因子与隐藏常数形式呈现；未给出具体构造实例，未直接验证更复杂错误模型；对非常数查询 q 的更细粒度分析仍有待深入。

---

## 438. Sheaf-MPNN: A Library for Sheaf Neural Networks as Message Passing

**arXiv ID:** 2611.04670 | [PDF](https://arxiv.org/pdf/2611.04670v1)

**作者:** Alessio Borgi `[一作]` (University of Cambridge), Pietro Liò `[通讯]` (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了一个基于 PyTorch Geometric 的 Sheaf Neural Network（SNN）库，利用局部 incidence-wise 消息传递实现 sheaf 扩散，避免显式组装全局 Laplacian。

**💡 创新点**

创新点在于：① 将 sheaf 扩散重新表述为局部消息传递，消除全局稀疏块矩阵的内存与计算开销；② 提供 15 种 SNN 架构（同构/异构、无向/有向、不同类型的限制映射），并通过统一的 Sheaf‑MPNN 理论框架证明它们与原始模型等价；③ 构建统一的基准套件，支持节点级、图级和异构图任务的可复现评估。

**🔧 技术方法**

核心技术包括：细胞 sheaf、局部限制映射学习、sheaf Laplacian 的消息传递式实现、类型条件化限制映射预测、低秩与注意力生成的非对称限制映射、以及 PyTorch Geometric 的扩展。

**📊 数据集**

使用了超过 20 个公开数据集，涵盖节点分类、图分类、异构图学习等任务，例如 Roman‑empire、HeterophilousGraphDataset 等。

**📈 对比分析**

通过统一的实验配置和 Optuna 超参数搜索，对比了 SNN 与传统 GNN（如 GCN、GraphSAGE）及现有 SNN 实现，结果显示：在异构和异性图上 SNN 通常能获得更好的精度，且在大多数任务中保持了与全局 Laplacian 方案相当的计算效率。

**⚠️ 局限性**

局限性包括：① 对极大节点数或高维 stalk 的大图仍存在显著的计算与内存开销；② 限制映射学习的参数量随 stalk 维度增长而显著增加；③ 目前库对动态图和流式数据的支持有限，未来需进一步扩展。

---

## 439. Self-Privileged Fine-Tuning

**arXiv ID:** 2611.04632 | [PDF](https://arxiv.org/pdf/2611.04632v1)

**作者:** Yangrong Hu `[一作]` (Hong Kong Polytechnic University), Jian Huang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了自我特权微调（SPFT）框架，利用训练阶段可见的特权信息为每个 token 加权，改进标准监督微调（SFT）。

**💡 创新点**

将冻结的初始模型在包含特权信息的条件下作为靶分布权重，构造静态但结果导向的 token 加权；通过贝叶斯分解解释权重如何根据 token 对最终答案的贡献进行自适应调节；并提出 SPFT‑SR 相对加权方案。

**🔧 技术方法**

token‑级加权监督微调、贝叶斯分解理论、动态与静态权重的优化目标分析、冻结副本的特权信息注入、梯度停止（stop‑gradient）技术、数学答案或代码 AST/规则抽象作为特权信息。

**📊 数据集**

数学推理：NuminaMath‑CoT 10k 子集，结合 MATH‑500、Minerva Math、OlympiadBench、AIME 2024、AMC 2023；代码生成：OpenCodeInstruct 10k 子集，评估 HumanEval、MBPP 及 EvalPlus 扩展版本。

**📈 对比分析**

与基线（Base、SFT、DFT、ASFT、TALR、PriFT）在相同训练预算、评测协议下对比；在数学推理上 SPFT 在五大基准的 Avg@8 与 Pass@8 均明显提升；在代码生成上 SPFT 的四任务平均 Pass@1 提升约10.7个百分点，超过所有竞争方法。

**⚠️ 局限性**

仅适用于能提供可验证或可结构化特权信息的任务；对特权信息质量高度敏感，噪声或缺失会削弱效果；目前仅验证了数学与代码两类任务，缺乏对更广泛领域（如对话、长文本等）的评估；未探讨在线或多轮对话等动态任务的适用性。

---

## 440. Refined CMA-ES for CEC 2026 Bound-Constrained Single-Objective Optimization

**arXiv ID:** 2611.04620 | [PDF](https://arxiv.org/pdf/2611.04620v1)

**作者:** Adam Stelmaszczyk `[一作]` (Warsaw University of Technology), Jarosław Arabas `[通讯]` (Warsaw University of Technology)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对CEC 2026受限单目标优化竞赛，提出了一种简化并重新调参的 CMA-ES 变体（R-CMA-ES），通过剔除不必要的机制并调高种群大小与初始步长来提升性能。

**💡 创新点**

创新点在于采用逆向消融法验证 CMA-ES 传统机制的实际贡献，证明在特定评估标准下剔除协方差矩阵分解延迟与平坦区逃逸机制以及适当调参即可超越基线与多种前沿算法。

**🔧 技术方法**

技术包括 CMA-ES 基础框架、协方差矩阵分解、演化路径步长控制、h_σ 校正，以及 R 语言实现与 Rcpp 加速。

**📊 数据集**

使用 CEC 2026 基准测试集，共 29 个连续黑盒函数（维度 30），并以官方评估协议（准确度与收敛速度同等权重）进行评测。

**📈 对比分析**

通过在 25 次独立实验中对每个函数进行 25,000 次评估，计算 CEC 2026 排名分数，R-CMA-ES 在总分 109,816 分位居第一，比分差分别为 11,428 分（对 L‑SRTDE）、8,523 分（对 RB‑IPOP‑CMA‑ES）和 7,593 分（对标准 CMA‑ES），并在 23/29 函数上显著优于基线。

**⚠️ 局限性**

局限性包括对特定评估标准（准确度与收敛速度同权）高度依赖，缺乏对更高维度或不同约束类型问题的验证，且剔除机制后在极端平坦或多峰问题上可能出现不稳定性。

---

## 441. System-level performance in massive ISAC networks

**arXiv ID:** 2611.04592 | [PDF](https://arxiv.org/pdf/2611.04592v1)

**作者:** Emanuele Mengoli `[一作]` (Telecom Paris), Nahuel Soprano-Loto `[通讯]` (INRIA Paris)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于动态随机几何的框架，用于建模并评估大规模ISAC网络的系统层性能

**💡 创新点**

证明了典型基站下通信与感知指标相互关联，并给出了关联比随网络密度的衰减规律

**🔧 技术方法**

运用了随机几何、泊松点过程、Harris–FKG不等式、Kalman滤波与排队理论

**📊 数据集**

使用从芬兰奥卢子城RT（射线追踪）仿真得到的信道模型数据

**📈 对比分析**

通过仿真验证，关联比始终大于1且随基站密度增大趋近于1，说明通信与感知性能可互补；相对传统单一功能模型性能更稳健

**⚠️ 局限性**

仅考虑单一UE与SO，假设完全频谱重用，未考虑多用户、多天线及更复杂的干扰管理策略

---

## 442. Rate-Distortion-Perception Tradeoff for Vector Gaussian Sources with Limited Common Randomness: Exact Characterization and Approximate Universality

**arXiv ID:** 2611.04693 | [PDF](https://arxiv.org/pdf/2611.04693v1)

**作者:** Wentao Wang `[一作]` (Shanghai Jiao Tong University), Bin Xia `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文从信息理论视角研究在压缩过程中既满足均方误差约束又满足感知误差约束的通用极限，推导了一个严格的上界并证明其在高维Gaussian源中可被逼近，随后给出对应的下界并展示其可实现性。

**💡 创新点**

创新点主要有：①将压缩与感知约束统一到一个统一的优化框架；②利用两个支点源（两点分布）证明任何可行解的上界可归约到两点分布；③通过KKT和Lagrange乘子法求解双点问题得到最优参数，并给出显式的最优常数；④提出并证明了一个全局可实现的下界，说明上界是最优的。

**🔧 技术方法**

主要技术手段包括：凸优化与Lagrange对偶；KKT条件与灵敏度分析；隐函数定理与单调性论证；函数的凸/凹性与Jensen不等式；以及对极限情况的渐近分析。

**📊 数据集**

论文并未使用实际数据集，而是以理论模型（高维正态源和其协方差矩阵）进行推导和证明。

**📈 对比分析**

通过对比理论推导得到的上界（ρ*≈0.1448）与下界（η*≈0.7517）以及对任意源协方差矩阵的通用极限，展示了上界在高维下可被逼近，且下界在特定构造的两点源下可实现；整体性能分析表明该理论极限在设计压缩与感知双重约束时是最优且可达的。

**⚠️ 局限性**

主要局限包括：①仅在Gaussian源与线性均方误差框架下给出结果；②上界与下界的实现是渐近的，实际有限维压缩的误差可能略大；③对非Gaussian或非线性感知指标的适用性尚未验证。

---

## 443. CardioVascBench:Prioritized Diagnosis Generation from Real-World Chinese Cardiovascular Records

**arXiv ID:** 2611.04687 | [PDF](https://arxiv.org/pdf/2611.04687v1)

**作者:** Minghao Kong `[一作]` (Sun Yat-sen University), Rongjie Wang `[通讯]` (Pengcheng Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了CardioVascBench——一个基于真实中国心血管住院病例的优先诊断生成回溯性基准；

**💡 创新点**

创新点在于：①将诊断列表分为主诊、覆盖和精准三维度评估；②提供结构化英文翻译以实现跨语言对齐；③设计CardioKG‑Agent知识增强基线，结合知识图谱和指南。

**🔧 技术方法**

采用了大语言模型（LLM）与知识增强架构，包括实体归一化、受限图检索、指南检索；评估使用nDCG、精确率、召回率及复合得分。

**📊 数据集**

使用了3,262份去标识化心血管病例（2,174份训练、1,088份评估），病例包含临床叙述、ECG报告与ICD‑10诊断标签，并提供英文翻译版本。

**📈 对比分析**

将30+模型按通用、医学专用与知识增强分组，对比主诊准确率、覆盖率、列表精准度及复合得分；结果显示知识增强模型在复合得分上提升约73.5%，但主诊准确率与列表精准度往往不一致。

**⚠️ 局限性**

局限性包括：评估基于退回诊断标签，未涵盖所有可能诊断；评估仅为回溯性、单中心、无实时验证；跨语言翻译可能引入偏差；知识增强系统所获提升受信息访问与推理预算限制，未直接验证临床安全与推理真实性。

---

## 444. MIMIC-Note-Extract: A Large-Scale Structured Dataset Extracted from MIMIC-IV-Note Discharge Summaries

**arXiv ID:** 2611.04685 | [PDF](https://arxiv.org/pdf/2611.04685v1)

**作者:** Kristian Schwethelm `[一作]` (TU Munich), Paul Hager `[通讯]` (TU Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了MIMIC-Note-Extract数据集，将300,720份MIMIC-IV-Note出院摘要的16个章节转化为结构化表格，且通过专家评审验证抽取准确率>92%。

**💡 创新点**

创新点在于：①提供覆盖全章节的结构化表格，补充了原MIMIC表缺失的信息；②采用混合规则+LLM的自适应抽取管道，兼顾效率与精度；③通过共享数据集降低研究者重复预处理工作。

**🔧 技术方法**

使用技术包括：正则/字符串匹配规则、schema‑guided大型语言模型（Gemini 2.5/3.1、Gemma 4、Qwen3.6）、Pydantic校验、手工验证以及零数据保留的云端LLM调用。

**📊 数据集**

所用数据集为MIMIC-IV-Note v2.2中的出院摘要（共300,720条符合质量标准的入院记录）。

**📈 对比分析**

方法通过与现有MIMIC表格对比验证抽取质量，并在完整信息与agentic两种临床决策基准中评估LLM性能，结果显示大模型在诊断准确率上达到82–88%，相较原基准提升0.1–0.9%；小模型准确率仅51%；抽取准确率整体在92%–99.8%之间。

**⚠️ 局限性**

局限性包括：去标识化导致部分临床细节（如日期、时间、术语规范）缺失，部分章节结构过于自由难以完整拆分，缺乏统一术语映射，且对高复杂度文本的抽取仍需要手工校对。

---

## 445. FenixWalk: Adaptive Sampling for Large-Scale Dynamic Graphs on Heterogeneous Architectures

**arXiv ID:** 2611.04683 | [PDF](https://arxiv.org/pdf/2611.04683v1)

**作者:** Pinhuan Wang `[一作]` (Nanjing University), Chengying Huan `[通讯]` (Nanjing University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个面向动态图的混合CPU–GPU随机游走引擎。

**💡 创新点**

创新点在于采用单一Fenwick树编码实现既支持拒绝采样也支持逆变换采样，同时避免额外权重存储；以及基于访问热点的GPU驻留策略。

**🔧 技术方法**

采用Fenwick树、在线自适应采样决策、GPU上采样内核、CPU–GPU协同调度以及动态图容器。

**📊 数据集**

在六个大规模真实网络（AB、TW、SK、UK、CD、CH）以及Bingo原始数据集上评估。

**📈 对比分析**

与四个基线（KnightKing、FlowWalker、FlexiWalker、LightTraffic）对比，平均速度提升3×–400×，显著降低内存使用（8×以上）。

**⚠️ 局限性**

局限在于对极高度分布图GPU驻留选择仍有改进空间，且在极大动态更新批量时仍受更新开销影响。

---

## 446. Hardness Self-Amplification in Linear Algebra and Fine-Grained Cryptography

**arXiv ID:** 2611.04680 | [PDF](https://arxiv.org/pdf/2611.04680v1)

**作者:** Geoffroy Couteau `[一作]` (Université Paris Cité), Thejas Radhika Sajith `[通讯]` (Chennai Mathematical Institute)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在线性代数问题（如求解线性系统、矩阵乘法、OMv 等）上基于细粒度难度假设构造细粒度密码学原语，并给出了新的硬件自放大技术。

**💡 创新点**

创新点在于提出一种基于随机低秩更新的硬件自放大方法，利用谱间隙分析将弱的平均情况求解器放大到几乎对所有矩阵都能高概率求解；同时将该技术扩展到矩阵乘法、OMv、近似解等场景，并将细粒度密钥协商与矩阵秩计算关联。

**🔧 技术方法**

主要技术包括：Markov 低秩随机 walk、傅里叶/布尔特里奇方法、谱间隙与吉尔曼-莱昂森-佩伦 Chernoff 绑定、Freivalds 误差检测、排列与标量重新随机化、以及利用矩阵乘法指数 ω 的加速。

**📊 数据集**

论文不依赖实验数据，而是以理论分析为主，所有实验均在有限域上进行；不使用公开数据集，而是考虑全域均匀分布和随机矩阵分布。

**📈 对比分析**

与以往的 worst‑to‑average 归约相比，本方法将运行时间从 O(α⁻¹log²(1/α)loglog(1/α)·T) 降至 O(α⁻¹·T)，消除对 log 依赖；在矩阵乘法上，时间复杂度为 O((T+n²)/α)；在 OMv、近似解场景下，误差提升为 O(1/q+ε) 需要 O(ε⁻²log(n/ε)) 次查询，优于先前的指数或双指数依赖。

**⚠️ 局限性**

局限性：需要在足够大的有限域上工作；对二元域（q=2）谱间隙不足，导致常数更差；自放大方法仍会留有可忽略的异常矩阵比例；无法得到线性系统的无条件 worst‑to‑average 归约；对高维稀疏或特殊结构矩阵的适用性尚未充分探究。

---

## 447. Rondo: Controlling Reordering in High-Throughput Packet Spraying

**arXiv ID:** 2611.04594 | [PDF](https://arxiv.org/pdf/2611.04594v1)

**作者:** Xizhi Zhang `[一作]` (Tsinghua University), Fengyuan Ren `[通讯]` (Tsinghua University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种在高吞吐量包级多路径传输中控制重排序的方案

**💡 创新点**

通过识别重排序的两大根本来源——瞬时路径队列差异和持久路径延迟差异，联合设计了多路径控制器与包调度器，实现高吞吐同时低重排序

**🔧 技术方法**

采用ACK基于路径容量估计、探测机制保持路径负载稳定，并通过调整包传输顺序来补偿持久延迟差异

**📊 数据集**

在实验测试床和大规模仿真环境中进行评估（未使用公开数据集）

**📈 对比分析**

与传统多路径方案相比，最大重排序降低至97.5%，在接收缓冲受限场景下有效吞吐量提升至8.51倍

**⚠️ 局限性**

实验仅在受控环境下完成，未验证在跨域、公共网络或大规模多租户环境中的鲁棒性

---

## 448. Surrogate Models for Stochastic Al-Cu Phase-Field Solidification

**arXiv ID:** 2611.04695 | [PDF](https://arxiv.org/pdf/2611.04695v1)

**作者:** Simon Savukoski `[一作]` (VTT Technical Research Centre of Finland Ltd), Tatu Pinomaa `[通讯]` (VTT Technical Research Centre of Finland Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了二维Al–Cu固化过程的相场模型，并用方差保持去噪扩散桥（VP‑DDBM）实现状态条件下的自回归预测；

**💡 创新点**

创新点在于将VP‑DDBM与扩散桥隐式模型（DBIM）相结合，能够在保持树枝结构和物理量准确性的同时捕捉固化过程的随机性；

**🔧 技术方法**

采用VP‑DDBM网络、U‑Net和SI‑ConvNeXt作为基准，并通过FiLM和全域温度编码实现温度条件的融入；

**📊 数据集**

使用70条独立生成的相场轨迹（26,940个状态），涵盖12个不同热梯度的二维固化实验；

**📈 对比分析**

在保留前沿位置、平均相位与Cu浓度、界面长度等物理量的长周期误差上，VP‑DDBM比U‑Net和SI‑ConvNeXt误差更低，推理速度提升约73倍；

**⚠️ 局限性**

局限于二维、固定拉拔速度和线性温度场，缺乏对更复杂热史、三维情形和更大数据集的验证。

---

## 449. Uncertainty-Aware Prototype Learning for Open-Set Test-Time Adaptation

**arXiv ID:** 2611.04689 | [PDF](https://arxiv.org/pdf/2611.04689v1)

**作者:** Jinwu Hu `[一作]` (South China University of Technology), Wei Luo `[通讯]` (South China Agricultural University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种面向开放域测试时自适应的无不确定性原型学习方法（UAPA），通过对测试流中出现的已知类和未知类样本进行区分并进行自适应更新，提升模型在开放集条件下的性能

**💡 创新点**

创新点在于：①将分类器原型相似度与批量归一化的证据不确定性评分结合用于ID/OOD分离；②引入动态原型细化（DPR）按缓冲的csID特征更新类原型；③采用对比原型熵适配（CPEA），对csID样本执行熵最小化与对比学习，对csOOD样本执行熵最大化与最大相似度惩罚，减少负面适应

**🔧 技术方法**

核心技术包括：证据深度学习（Evidence‑DL）产生不确定性评分、批量归一化、原型学习、动态原型更新、InfoNCE式对比学习以及熵优化

**📊 数据集**

实验数据集涵盖CIFAR‑10‑C、CIFAR‑100‑C和Tiny‑ImageNet‑C，使用WideResNet‑40‑2、ResNet‑50和ViT‑Base等模型进行验证

**📈 对比分析**

与多种闭集与开放集TTA基线（TENT、EATA、BN Adapt、CoTTA、OSTTA、UniEnt、ROSETTA等）比较，UAPA在Acc、AUROC、FPR@95TPR、OSCR等指标上均有显著提升，平均OSCR提升约20%

**⚠️ 局限性**

局限性包括：对高比例csOOD样本时性能仍会下降，单类批次下的OOV识别仍较困难，且需在每个批次中计算额外的原型相似度与不确定性评分，导致轻微的推理开销

---

## 450. Designing AI-Mediated Communication in Teams

**arXiv ID:** 2611.04686 | [PDF](https://arxiv.org/pdf/2611.04686v1)

**作者:** Wichinpong Park Sinchaisri `[一作]`, Zhiwei Wang `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究提出了一种以人工智能（AI）为中介的团队沟通设计框架，分析在员工存在私人偏好与信息冲突时，企业如何通过公开项目状态信息、引导员工报告以及调整信息披露时机，最大化组织绩效与员工福利的权衡。

**💡 创新点**

创新点包括：① 将信息披露视为双重目标——既提升组织对项目的适应性，又保持员工愿意透露私人偏好的动机；② 在“信息设计”与“协调”两大理论框架交叉，揭示信息透明度与偏好披露之间的非线性博弈关系；③ 通过在静态与动态（多期）模型中比较，首次展示了信息时序对沟通策略和结果的决定性影响；④ 设计了可行的“分离式转发”机制，并证明在特定参数域内可优于传统的无条件转发。

**🔧 技术方法**

使用的主要技术手段为：贝叶斯博弈理论、信息设计（Persuasion）框架、最佳信号（Optimal Signalling）分析、纯策略纳什均衡推导与数值优化求解（如求解最优信息区间、池化宽度等）。

**📊 数据集**

本研究为理论模型，未使用实际数据集；所有结论均基于数值模拟（如均匀分布的项目状态、平衡/不平衡的类型先验）验证。

**📈 对比分析**

通过与“直接员工沟通无项目信息”基准对比，以及“完全信息披露”“优化初始披露”“分离式转发”等三种信息设计方案进行比较，发现：在对齐环境下，完全信息披露可实现约3–4%的组织损失下降；进一步的信号优化与分离式转发能再提升约0.1–0.2%；当存在对立偏好且协调强度高时，后者提升幅度可达5–7%。

**⚠️ 局限性**

局限性包括：① 仅考虑两名员工与单一项目，未覆盖更大规模团队与多项目情景；② 假设类型与当地条件独立、分布已知；③ 只选取纯策略均衡且以组织最优均衡为标准，未探讨其他均衡或非理性行为；④ 在动态模型中未包含即时绩效反馈或学习机制；⑤ 分离式转发的实现假设较强，现实中可能难以完全分离。

---

## 451. Feature-Based and Deep Learning Methods for Astronomical Light Curve Classification

**arXiv ID:** 2611.04664 | [PDF](https://arxiv.org/pdf/2611.04664v1)

**作者:** V. Baymakanov `[一作]`, M. Hushchyn `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对ZTF StarEmbed数据集中的七类变量星光变曲进行多类别分类，比较了基于手工特征的传统机器学习方法与直接在原始光变曲上训练的端到端深度学习模型。

**💡 创新点**

提出了三种只使用真实观测点的增广策略（随机窗口、随机日间箱、随机抽样）以及多种掩码聚合方案，系统量化了特征工程与深度学习在该任务上的性能差距，并为后续自监督预训练提供基准。

**🔧 技术方法**

传统方法：逻辑回归、CatBoost、MLP；深度方法：单向/双向LSTM、Transformer（含不同头部与激活），均使用AdamW、类别权重交叉熵，配合三种增广。

**📊 数据集**

ZTF StarEmbed数据集：约4万条光变曲，七个不平衡类别（EW、EA、RRab、RRc、RRd、RS CVn、LPV），每条曲线包含多条滤光波的时间、光度、误差信息。

**📈 对比分析**

结果显示传统特征+模型在准确率≈90%和宏平均ROC‑AUC≈0.98上遥遥领先；深度模型在未增广时准确率仅≈60%，通过增广可提升至≈70‑80%，但仍低于特征方法。

**⚠️ 局限性**

深度模型受限于标注样本不足、光变曲的多变量、非等间距采样与形态多样性，导致学习效果不佳；需要更大规模标注数据或自监督预训练来突破这一瓶颈。

---

## 452. Best-of-Depth: Learning Exploratory Dynamics in Looped Transformers

**arXiv ID:** 2611.04663 | [PDF](https://arxiv.org/pdf/2611.04663v1)

**作者:** Shota Takashiro `[一作]` (University of Tokyo), Yusuke Iwasawa `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实验了一种叫 Best-of-Depth（BoD）的训练目标，利用循环 Transformer 在同一轨迹的不同深度产生多种有效候选解，显著提升了多解覆盖率与累积成功率。

**💡 创新点**

核心创新在于：① 用最小化在所有训练深度中最低损失的预测来训练模型；② 该目标让不同深度匹配不同合法目标，无需多头监督或深度特定标签；③ 在 exact‑match 损失下证明其等价于最大化解集覆盖率。

**🔧 技术方法**

采用 Loop/Universal Transformer、BoD、Soft‑Depth、Mean‑Depth、Random‑Depth 等训练目标；对代码生成使用 Huginn（3B looped LM）进行微调；对 latent 动态进行 PCA 与状态位移分析；在固定递归步数预算下比较深度与多轨迹的组合。

**📊 数据集**

N‑Queens（10×10 5 固定皇后，1000 个可枚举解的测试集）、图着色（10 顶点三色 512 测试集）、CodeFeedback（20k Python 例子）以及 HumanEval（164）与 MBPP（500）代码生成数据集。

**📈 对比分析**

与 Last‑Depth、Mean‑Depth、Random‑Depth、Soft‑Depth 进行对比。BoD 在 N‑Queens 上覆盖率 91.03%（vs 45.75% Final‑Depth），图着色覆盖率 72.16%（vs 44.39% Soft‑Depth），代码生成 Success@32 37.8%（vs 34.8% Last‑Depth）。同时 BoD 能在训练深度之外发现新的解，且在固定步数预算下与多轨迹组合效果更佳。

**⚠️ 局限性**

实验范围局限于小规模组合任务和单一预训练语言模型；随着深度增加发现趋于饱和；在更大规模任务、更多模型以及自适应停止策略方面仍需进一步研究。

---

## 453. BhopalMLS: A LiDAR Dataset for Semantic Scene Understanding in Heterogeneous Urban Environments

**arXiv ID:** 2611.04654 | [PDF](https://arxiv.org/pdf/2611.04654v1)

**作者:** Pyare Lal Chauhan `[一作]` (IISER Bhopal), Bharat Lohani `[通讯]` (IIT Kanpur)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了BhopalMLS，一个覆盖约10公里、1.2亿点的移动激光扫描（MLS）数据集，包含62个静态细粒度语义类别、19,518个实例，并提供12类和21类两级标注，还扩展了6个移动子类用于动态目标分析。

**💡 创新点**

创新点在于：①首次发布大规模、密集且非规划化的印度城市MLS数据；②设计细粒度且兼容现有benchmark的语义层级，涵盖本土特有物体（如三轮车、手推车、摊位遮阳篷）；③系统评估了基础模型（Utonia、Concerto、Sonata）在此场景下的优势；④通过交叉区域实验揭示BhopalMLS模型对其他数据集的迁移性能显著优于反向迁移，归因于子类型覆盖而非数据规模或几何差异。

**🔧 技术方法**

技术方法包括：多通道（XYZ、强度、部分RGB、法向量）输入；线性探针、解码器探针、LoRA、全微调等适配策略；基于交叉熵+Lovász损失的平衡训练；两位注释员交叉校验、细化SOP；使用CORAL等域适配基线验证迁移差异；并在统一体素化（5 m×5 m × 25 cm）协议下进行评估。

**📊 数据集**

使用的数据集有：BhopalMLS（本研究主数据）；Ke-MLS、Toronto-3D、Paris‑Lille‑3D、SemanticKITTI（对比实验）。BhopalMLS以ZEB手持扫描器采集，约40%点具RGB；其他数据集为车辆/台式激光扫描或航拍激光，采样密度和视野各异。

**📈 对比分析**

与传统从头训练的监督模型（PTv3、LitePT-L、KpConvX-L）比较，预训练基础模型在12类语义分割上实现mIoU 72.12%（Utonia）或71.06%（Concerto），明显高于68.04%；在实例分割上Utonia FT获得44.07% mAP50。交叉区域迁移实验显示，BhopalMLS源模型在四个目标数据集上提升mIoU最多36点；而目标→BhopalMLS的性能大幅下降（如SemanticKITTI→BhopalMLS仅12.3%）。

**⚠️ 局限性**

局限性包括：RGB覆盖仅约40%，对颜色特征依赖有限；某些细薄结构（栅栏、地面）仍难以准确分割；数据仅来自单一城市，无法完全代表印度乃至全球不同的非规划化环境；标注工作耗时、人工误差仍需进一步统一；未来需扩展多地区、多传感器和更丰富的动态目标类别。

---

## 454. Therapy-FIS: A Dataset for Assessing Facilitative Interpersonal Skills in Psychotherapeutic Setting

**arXiv ID:** 2611.04645 | [PDF](https://arxiv.org/pdf/2611.04645v1)

**作者:** Qijian Zheng `[一作]` (Fudan University), Feng Liu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了公开的Therapy‑FIS数据集，提供基于标准化心理治疗情景的面部、音频、文本特征，并在此数据集上构建了自动化FIS评估基准；

**💡 创新点**

首次公开可复现的专业心理治疗者FIS评估数据集，并引入双向（dyadic）模型DyadicFIS，强调治疗者与模拟患者的互动力；

**🔧 技术方法**

使用多模态特征提取（OpenFace, librosa, Wav2Vec2.0, DeBERTa），多模态融合（{M}-Mamba‑TE、Cross‑Attn‑Agg、Dyadic Interaction）以及CCC+MSE混合损失进行训练；

**📊 数据集**

Therapy‑FIS数据集，包含890段治疗者视频、8段模拟患者视频，提取的面部动作单元、音频梅尔谱与文本嵌入；

**📈 对比分析**

与多种基线（TimeSformer、DLinear、Reformer、MSGNet、KANAD、JMT、MDF、CLGSI）对比，DyadicFIS在宏观CCC上分别取得0.2807（仅治疗者）和0.3299（治疗者+患者），优于其他方法；

**⚠️ 局限性**

标签基于三名受训评估者的平均，缺乏个体评估者差异；只发布去标识化特征，限制了端到端训练和原始音视频分析的可能性。

---

## 455. First Near-Additive All-Pairs Approximate Shortest Paths in Decremental Weighted Graphs

**arXiv ID:** 2611.04635 | [PDF](https://arxiv.org/pdf/2611.04635v1)

**作者:** Eden Koveshi `[一作]`, Ariel Sapir `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了四种在降序（decremental）加权无向图上运行的近加性（1+ε,β）全点对最短路径（APASP）算法，并给出了相应的总更新时间，首次在降序环境下实现了加权图的近加性APASP。

**💡 创新点**

创新点在于：① 将 Baswana‑Kavitha 的层级诱导边集从静态推广到降序图；② 结合单源最短路径的 Monotone Even‑Shiloach 树、离散化权重的 Bernstein 逼近以及针对层级诱导子图构造的 hopset，克服了路径深度、权重跨度和距离估计非单调性三大难题；③ 通过多尺度权重缩放与 hopset 结合，得到在任意距离尺度下的近加性保证；④ 通过不同的取样概率与层级深度，得到从稀疏到密集图的多种运行时与误差折中。

**🔧 技术方法**

使用的主要技术包括：
- 层级诱导边集（hierarchy induced edge sets）
- Monotone Even‑Shiloach 树（maintaining monotone distance estimates）
- Bernstein 的权重缩放与逼近（geometric scaling）
- 适用于层级子图的离散化 hopset（Łącki‑Nazari 方案改造）
- 近加性单源最短路径的组合（如 Dory‑等 的 1+ε,2‑APASP 作为黑盒）

**📊 数据集**

本文没有使用任何真实或合成数据集，而是给出了理论上总更新时间与误差的上界，所有结论均基于期望运行时间分析。

**📈 对比分析**

相较于之前的降序 APASP 结果（如 Dory 等 的 (2+ε)-APASP 与 Chechik 的 (3+ε)-APASP），本文的四种算法在不同稀疏/密集情形下实现了更低的总更新时间（例如在稠密图中从 O(mn^1/2) 降到 O(n^7/3)）并且误差改为近加性 1+ε 与局部加性项 2W_2、2W_1+2W_2 等；在 unweighted 场景中，获得了与最速静态算法匹配的时间（O(min{n^3/2 m^1/2, n^7/3}) 等）。

**⚠️ 局限性**

局限性包括：
- 仅适用于降序更新（删除与权重增大），不直接支持增量或全动态；
- 需要对抗无知对手（oblivious adversary），对自适应对手的鲁棒性未知；
- 运行时间仍含有子多项式因子（如 n^o1）以及对 ε 的依赖；
- 需要维护大量跳跃集合与缩放副本，实际实现复杂度较高；
- 对于极稀疏图的加权版本，仍需使用更复杂的 hopset 与采样方案。

---

## 456. Gradient-Guided Orientation Learning for Weakly Supervised Oriented Object Detection

**arXiv ID:** 2611.04616 | [PDF](https://arxiv.org/pdf/2611.04616v1)

**作者:** Junyong Jang `[一作]` (Kyungpook National University), Jaehyup Lee `[通讯]` (Kyungpook National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 GradOrient，一种利用图像梯度信息为弱监督方向检测提供伪角度监督的框架。

**💡 创新点**

创新点在于将对象区域的主导梯度方向直接转化为伪角度标签，避免了传统间接几何约束，并引入了基于角度误差的评价指标 OAP。

**🔧 技术方法**

技术上使用梯度方向直方图、加权圆均值、角度置信度门控，以及与原始、旋转、翻转视图的多视角一致性损失。

**📊 数据集**

在 DOTA-v1.0 与 DIOR-R 两个遥感图像检测基准上进行实验。

**📈 对比分析**

与现有 HBox 监督和全监督方法对比，GradOrient 在 AP_50 上提升 1–1.4 分，在 OAP_50 上提升 0.5–1.6 分，并显著降低角度误差。

**⚠️ 局限性**

局限性是伪角度估计在纹理复杂或背景干扰的对象上可能失效，且仍需与原始监督配合以避免错误信号。

---

## 457. Making Smell Actionable in Design: Lexicon, Movement Patterns, and Gesture Cards

**arXiv ID:** 2611.04609 | [PDF](https://arxiv.org/pdf/2611.04609v1)

**作者:** Anran Feng `[一作]` (Purdue University), Jingrui An `[通讯]` (Jiangnan University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了从嗅觉体验到意图、运动轨迹再到手势卡片的可设计资源，形成可在工作室使用的嗅觉设计框架。

**💡 创新点**

创新点在于将嗅觉与动作映射为可操作的“意图–轨迹–手势”三元组，并提供可共享的词典、轨迹族和手势卡。

**🔧 技术方法**

采用文本挖掘提取动词、Laban运动分析、DTW、Procrustes、k‑means聚类和 Kinect/OpenPose 运动捕捉技术。

**📊 数据集**

使用5种日常气味（葡萄、柑橘、薄荷、薰衣草、咖啡）与28名参与者、15名普通人、8名舞者以及8名设计师和12名学生收集的数据。

**📈 对比分析**

通过 DTW‑相似度评估运动一致性（平均相似度0.75），并对手势卡进行盲评量表，发现清晰度、可生成性、易学性显著高于中性水平。

**⚠️ 局限性**

局限包括气味样本单一、只捕捉上肢轨迹、跨阶段翻译可能导致信息损失，以及缺乏跨文化或长期实验验证。

---

## 458. Multi-Path Mamba Diffusion for 3D Point Cloud Semantic Scene Completion

**arXiv ID:** 2611.04583 | [PDF](https://arxiv.org/pdf/2611.04583v1)

**作者:** Li Liang `[一作]` (University of Western Australia), Ajmal Saeed Mian `[通讯]` (University of Western Australia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种结合变分自编码器、状态空间模型与扩散生成模型的多路径Mamba去噪网络，用于高效完成稀疏LiDAR点云的语义场景重建。

**💡 创新点**

创新点在于将多路径、三向跳跃Mamba结构与多尺度融合模块融合至扩散网络，利用不同膨胀率的多路径特征捕获长序列上下文；并在VAE潜在空间中进行条件扩散，显著提升重建精度与语义一致性。

**🔧 技术方法**

主要技术包括变分自编码器（VAE）、条件扩散模型、Mamba状态空间层、跳跃三向膨胀块（MSTM）、多尺度融合（MSFusion）与维度分解残差（DDR）块。

**📊 数据集**

实验基于SemanticKITTI 3D点云数据集，采用256×256×32体素格和0.2 m分辨率的城市驾驶场景。

**📈 对比分析**

与LMSCNet、UDNet、Local-DIFs、SSA-SC、JS3C-Net、SSC-RS、S3CNet等SOTA方法对比，mIoU达29.1%，在几乎相同的参数规模下表现出竞争力，尤其在场景完整性与语义分割上优于多数方法。

**⚠️ 局限性**

局限在于模型参数规模（约29 M）仍高于部分轻量级方案，且在小尺度物体的细节重建上仍略逊于部分专门设计的密集卷积网络。

---

## 459. Right Tool, Wrong Parameters: Defending LLM Agents Against Parameter Poisoning

**arXiv ID:** 2611.04673 | [PDF](https://arxiv.org/pdf/2611.04673v1)

**作者:** Yao Wang `[一作]` (University of Science and Technology of China), Han Fang `[通讯]` (University of Science and Technology of China)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 AuthorityFlow 的任务感知参数授权框架，用来防御 LLM 代理中的参数中毒攻击，并基于 AgentDojo 构建了 FactAuthBench 基准。

**💡 创新点**

创新点在于将任务触发的信任层级与信息流追踪相结合，对每个工具参数的来源进行语义支持与来源授权双重验证，从而实现对参数级攻击的专门防护。

**🔧 技术方法**

采用信息流追踪、语义支持判定、来源授权检查以及辅助验证模型（默认 Qwen3.6‑27B），并与多种现有 IPI 防御手段进行组合。

**📊 数据集**

使用 FactAuthBench（144 对正负样本，涵盖银行、Slack、旅行、工作区四类工具交互场景）作为实验数据集。

**📈 对比分析**

在 Qwen3.6‑27B 和 GPT‑4o 上与 Sandwiching、Spotlighting、IPIGuard 等 13 种基线进行对比；AuthorityFlow 将攻击成功率从 93.06% 降至 9.03%，同时保持约 82.6% 的良好任务实用性；在结合 Purification 后，攻击下实用性提升至约 58%。

**⚠️ 局限性**

局限性包括：1）权威构建依赖于任务说明和工具元数据，信息模糊或缺失时易误判；2）辅助验证模型的鲁棒性和推理延迟仍需提升；3）当前仅检测并拦截参数中毒，未实现完整的自动任务恢复。

---

## 460. FedCSGA: Client Selection Based on Genetic Algorithm for Federated Learning

**arXiv ID:** 2611.04667 | [PDF](https://arxiv.org/pdf/2611.04667v1)

**作者:** Ghani Haider `[一作]` (Chungbuk National University), Taehong Kim `[通讯]` (Chungbuk National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FedCSGA，一个基于遗传算法的联邦学习客户端选择框架，能够同时考虑统计和系统异构性，并通过历史加权避免选择偏差；

**💡 创新点**

创新点在于将客户端子集级别的组合优化与非线性历史衰减权重结合，形成一次性整体评估的适应度函数，突破传统逐客户端贪婪或单一维度的选择方法；

**🔧 技术方法**

使用联邦学习框架、遗传算法（选择、交叉、变异）以及基于资源指标（统计资源+系统资源）与历史参与计数的适应度函数；

**📊 数据集**

在四个公开数据集上进行评测：FMNIST、CIFAR‑10、CIFAR‑100 和 Tiny‑ImageNet；

**📈 对比分析**

与 FedAvg、FedProx、PowD、AFL、GreedyFed、GreedyFL 等基线比较，FedCSGA 在非 IID、IID 及不同参与比例下均提升 1–7% 的最终准确率，并显著减少收敛轮数，表现更稳定；

**⚠️ 局限性**

局限在于需要客户端上报资源指标、服务器端遗传算法的计算开销，以及对极大规模交叉设备环境的可扩展性和严格公平/隐私保障的缺失。

---

## 461. Phase-DGS: Phase-guided Dynamic Gaussian Splatting from Unsynchronized Multi-view Video

**arXiv ID:** 2611.04666 | [PDF](https://arxiv.org/pdf/2611.04666v1)

**作者:** Hosung Jeon `[一作]` (Electronics and Telecommunications Research Institute), Hyon-Gon Choo `[通讯]` (Electronics and Telecommunications Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种通过从图像内容提取语义相位替代绝对时间戳的动态场景重建框架Phase‑DGS，能够在多摄像机未同步条件下训练高质量4D Gaussian Splatting模型。

**💡 创新点**

创新点在于利用每帧的内在相位实现跨摄像机的时空对应，消除传统同步硬件或音频、几何匹配方法的局限，同时兼容多种4DGS骨干。

**🔧 技术方法**

采用ViT编码器+轻量级MLP相位投影器、可学习特征库以及相位一致性与单调性损失，集成至4D Gaussian Splatting背骨。

**📊 数据集**

使用Nvidia Dynamic Scene Dataset（12摄像机）和SelfCap（24摄像机）两大多摄像机动态场景数据集进行评估。

**📈 对比分析**

与无同步、VisualSync、Sync‑NeRF等基线对比，在时间偏移、帧丢失和冻结等不同步条件下，Phase‑DGS在PSNR/SSIM/LPIPS等指标上提升多达+8.5 dB，整体性能显著优于基线。

**⚠️ 局限性**

局限性包括对周期性运动易产生相位混淆、对骨干网络的依赖较大，以及在极端异步场景下仍需进一步改进相位学习策略。

---

## 462. SPADE: Spatiotemporal Preference Alignment and Diversity Enhancement for Local Search

**arXiv ID:** 2611.04662 | [PDF](https://arxiv.org/pdf/2611.04662v1)

**作者:** Chenxu Wang `[一作]` (Meituan), Peng Yan `[通讯]` (Meituan)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在本地服务平台上构建了一套端到端的主动查询推荐生成框架 SPADE，能够在用户打开 App 时即时生成多样化、可执行且与用户上下文匹配的查询候选列表。

**💡 创新点**

核心创新包括业务价值感知的 OA‑BPE 词表优化、细粒度多尺度空间-时间编码 STAR、基于多维奖励的 MARS‑DPO 训练以及 Sigmoid 多样性束搜索 SDBS，四个模块协同提升长尾覆盖、空间时效匹配、用户偏好对齐和候选多样性。

**🔧 技术方法**

技术手段涵盖自定义 BPE 合并策略、Transformer 编码器-解码器结构、软门控多尺度空间/时间嵌入、复合奖励（排名、检索质量、地理距离）与自适应负采样的 DPO 对齐、以及位阶感知的 Sigmoid 增益束搜索。

**📊 数据集**

使用了某本土生活服务平台的生产日志：100K 用户、148万条唯一查询、2608 个城市、30 天跨域搜索日志做预训练和30 天点击日志做偏好对齐。

**📈 对比分析**

与传统检索模型（SwingQ2Q、Dense‑Ret）以及最近的生成式基线（AIGQ、OneSug、GQR 等）对比，SPADE 在 HR@5 上提升 5.6%（相较 AIGQ），GeoC@5 降低 40.3%，Exec@5 提升 1.9%；在线 A/B 测试中实现 1.88% CTR、0.177% 召回量、0.105% DAU 的正向提升。

**⚠️ 局限性**

局限性包括：奖励函数依赖大模型 Qwen3.5 的批量推理，需额外成本；OA‑BPE 权重随查询分布演化需定期更新；STAR 仅编码单一 GPS 坐标，未利用轨迹级时空上下文。

---

## 463. Leakage-free evaluation of residual graph convolutional networks for network intrusion detection

**arXiv ID:** 2611.04660 | [PDF](https://arxiv.org/pdf/2611.04660v1)

**作者:** Zahid Iqbal `[一作]` `[通讯]` (University of Gujrat), Zahid Iqbal (University of Gujrat)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对流级网络入侵检测的图神经网络进行严格的归纳式评估，构建仅基于训练流的k近邻图，剔除重复流并在多种随机划分下测评。

**💡 创新点**

提出了针对流数据的“训练集限定、查询流仅连训练邻居、重复流剔除”三大评估规范，并通过对比不同图聚合方式和基准模型检验消息传递的实际价值。

**🔧 技术方法**

使用残差图卷积网络（R‑GCN）、GCN、GraphSAGE、GAT、kNN投票、MLP、XGBoost、LightGBM、随机森林以及主机图E‑GraphSAGE等多种模型。

**📊 数据集**

在CICIDS2017和NF‑UNSW‑NB15‑v2这两个公开NetFlow格式的数据集上进行实验，数据先做清洗、去重、特征预处理后再划分。

**📈 对比分析**

通过与MLP、kNN投票和树集成模型的对比发现：R‑GCN与其他图网络在二分类F1上与MLP基本持平，树集成模型仍然在F1、宏F1和低误报预算下显著优于图网络；在宏F1上，图网络因对小类误报高而落后。

**⚠️ 局限性**

主要局限在于：图网络未能在离群或未见攻击族上取得优势；对小类的低精度导致误报高；且评估受限于仅使用单一数据源和固定kNN邻居数。

---

## 464. Non-asymptotic Sphere-Packing Bounds for Finite-Blocklength Scalar Gaussian Multiple-Access and Broadcast Channels

**arXiv ID:** 2611.04647 | [PDF](https://arxiv.org/pdf/2611.04647v1)

**作者:** Kaixuan Bao `[一作]` (Southeast University), Xiaohu You `[通讯]` (Southeast University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究有限块长高斯多址（MAC）和广播（BC）信道的非渐近可达率区域，提出基于体积均匀随机编码（VURC）的代码书，并给出SIC与JNN解码下的闭式可达性界、误码率下界及四阶渐近展开。

**💡 创新点**

创新点在于：①首次将VURC从单用户推广到多用户，利用码字的径向自由度显著提升码本容量；②通过SOSD（凸顺序替换）将多用户干扰+噪声映射为等效高斯噪声，得到简单可计算的闭式可达性界；③构建可扩展的四阶渐近框架，将单用户高阶结果系统化应用于多用户场景；④对比SIC与JNN，揭示联合解码的性能优势。

**🔧 技术方法**

技术包括：体积均匀随机编码（VURC）、超载编码、SIC与JNN解码、凸顺序替换（SOSD）、高阶中心极限定理与大偏差分析、Gamma与Beta函数的解析性质、Monte‑Carlo数值积分。

**📊 数据集**

主要使用仿真数据对比误码率和可达率；并在二维（两用户）高斯MAC/BC场景下绘制可达率区域，利用符号表格和图形验证理论界限。

**📈 对比分析**

与现有基于球壳随机编码和DT/RCU等方法相比，VURC+SIC闭式界在计算复杂度上从指数降为多项式，误码率下界更保守；JNN相较SIC获得更宽裕的可达率区域，VURC在高码率区域表现出明显优势，实验结果验证了四阶渐近展开在短块长下的精度。

**⚠️ 局限性**

局限性包括：仅考虑单天线、理想CSI、有限用户数；JNN解码的计算与分析复杂度高，尚未对大规模用户做扩展；闭式界在极端低误码率或极短块长时可能不够精确；高阶展开需假设随机码书收敛，实际实现需进一步验证。

---

## 465. Does LinUCB Need Valid Confidence Sets?

**arXiv ID:** 2611.04641 | [PDF](https://arxiv.org/pdf/2611.04641v1)

**作者:** Ahmed Ben Yahmed `[一作]` (Criteo AI Lab), Clément Calauzènes `[通讯]` (Criteo AI Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了α‑LinUCB算法，在随机线性带宽问题中证明无论是否使用有效置信集，都能得到次线性累计失调。

**💡 创新点**

创新点在于：1）证明置信集有效性不是必要条件；2）引入对光滑强凸动作集的曲率假设，使得探索在所有方向上展开；3）构造对数截断谱势函数，取代传统椭圆势函数；4）改进标准LinUCB的失调上界，提升了 √(log T) 因子。

**🔧 技术方法**

主要技术包括：谱矩阵增长分析、能量归纳（energy argument）、对数截断势函数、Bregman 散度方法、适用于自适应设计的自归一化浓缩不等式，且整个证明不依赖传统的乐观性不等式。

**📊 数据集**

实验使用合成数据：在 d=10 的单位球上生成高斯噪声，设置不同 α 值（β/256、β/64、β/16、β/4、β、2β），进行 48 次重复。

**📈 对比分析**

与标准 LinUCB（α=β）以及不同 α 取值进行对比；结果显示 α=β/4 的失调最低，失调随时间呈 √t 下降；最小特征值也随 √t 增长，且在小 α 时置信集失效率高。

**⚠️ 局限性**

局限性在于：仍存在 √T 与下界 d√T 之间的 √T 缺口；依赖动作集曲率假设，无法推广到多面体等无曲率情况；实现中需要对自适应设计的浓缩估计，可能难以进一步压缩失调。

---

## 466. EvoCache: Accelerating Browser Large Language Model Delivery with Public Cache Anchors

**arXiv ID:** 2611.04638 | [PDF](https://arxiv.org/pdf/2611.04638v1)

**作者:** Zhiyu Wang `[一作]` (University of Melbourne), Rajkumar Buyya `[通讯]` (University of Melbourne)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了浏览器内部的跨发布器官（EvoCache），实现仅凭目标模型信息即可在浏览器内重构所需LLM文件，同时隐藏底层缓存与重构细节，保护跨站点缓存隐私。

**💡 创新点**

创新点在于：①使用浏览器预置的公共锚点与历史缓存分别作为重构资源；②为每个张量块动态选择最优重构操作（精确重用、压缩获取、XOR 或 BMS 差分，甚至低位差分与字段分离），从而在保证完整重构的前提下显著降低网络传输量；③通过固定返回时点实现跨站点缓存不可探测。

**🔧 技术方法**

技术实现包括：GGUF 结构对齐与分块、基于张量元数据的块级对齐、残差编码（XOR、BMS、低位差分、字段分离）、压缩获取（zstd）、浏览器内核使用 Rust/Wasm 进行流式重构与校验、以及公共锚点与严格释放的缓存策略。

**📊 数据集**

使用的数据集为：核心 23 个跨发布包对（12 个 LLM 家族/量化组合）、独立验证集 8 个包（2 个新 LLM、4 个未参与编码的发布者）、7B 大模型 6 个包对（每个 4.36 GiB），以及 45 个公开 GGUF 包作为整体语料。

**📈 对比分析**

与常规全下载、分区缓存、FastCDC、Xdelta、张量对齐 Xdelta、以及 LLM 特定的 BitX 等对照方法比较，EvoCache 在核心数据集上平均减少 77.3% 网络流量（比最优对比 49.4%），在 7B 模型上 76.6%；在 240 次真实浏览器实验中，相比普通全下载减少 78.9% 网络流量、64.2% 页面准备时间；相对固定张量 XOR 进一步降低 23% 流量、12.6% 载入时间。

**⚠️ 局限性**

局限性包括：仍需在浏览器端预置公共锚点并维护其空间；对不同块量化格式的适配尚未完整验证；在极高带宽或极低带宽场景下，残差编码的压缩比可能不足；以及对动态生成或自定义张量格式的模型支持有限。

---

## 467. SAT Encoding for Set of At-Most-K Cardinality Constraints with Ladder Shape

**arXiv ID:** 2611.04637 | [PDF](https://arxiv.org/pdf/2611.04637v1)

**作者:** Hieu Truong Xuan `[一作]` (Vietnam National University), Khanh To Van `[通讯]` (Vietnam National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对 Ladder 形状的 At‑Most‑K（AMK）约束的块化 SAT 编码方法 SCL，能够一次性编码一组相邻约束并通过顺序计数器（Sequential Counter）连接块，显著减少变量和子句数量。

**💡 创新点**

创新点在于将相邻 AMK 约束聚合为块，并使用顺序计数器对每块编码，同时构造块间连接公式，实现了首次针对 k>1 的 Ladder AMK 约束提供高效、线性规模的 SAT 编码。

**🔧 技术方法**

采用 Sequential Counter 编码、AMK 分解、块间连接公式，结合 Duplex、BDD、Seq 等传统编码做对比；在 Anti‑Bandwidth 问题中将 SCL 与 CP/MIP 求解器（CPLEX、Gurobi、CP‑SAT、CP‑MZ‑Chuffed）一起使用。

**📊 数据集**

实验使用了 24 个 Harwell‑Boeing Sparse Matrix 集合中的 ABP 实例，以及 n=1000、宽度 w∈{5,10,…,995} 的 Ladder AMO/AMK 约束实例。

**📈 对比分析**

与 Pairwise、Seq、BDD、Card、Duplex 等编码以及 CP/MIP 求解器进行对比，SCL 在编码规模、求解时间上普遍优于其他 SAT 编码，并在 ABP 实例上多次超越 CPLEX、Gurobi、CP‑SAT 等强大求解器。

**⚠️ 局限性**

局限性包括：在宽度较小（w≤7）的 Ladder AMK/AMO 约束下，SCL 的优势不明显；目前仅针对 AMK，未扩展到 AMO、AMONG、At‑Least‑K 等更一般的序列约束；缺乏在实际调度问题中 k>1 时的实证优势。

---

## 468. Evaluating LLMs as Judges of Counterspeech Quality Against Online Hate Speech

**arXiv ID:** 2611.04627 | [PDF](https://arxiv.org/pdf/2611.04627v1)

**作者:** Johnny Alali `[一作]` (Vrije Universiteit Amsterdam), Davide Ceolin `[通讯]` (Centrum Wiskunde & Informatica)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究大型语言模型（LLMs）在评估针对网络仇恨言论的反仇恨言论（counterspeech）质量上的可行性，探讨其与人工专家评估的一致性；

**💡 创新点**

创新点在于：①系统性对12款开源LLM在点评（pointwise）和两类对比评（pairwise）三种提示策略（zero-shot、few-shot、chain-of-thought）下的表现进行统一评估；②将点评分数转为对比标签，以检验两种评估方式的可比性；③对模型间一致性与人类一致性进行对比，揭示模型间高度协同但与专家差距较大的现象；

**🔧 技术方法**

技术手段包括：采用CSEval基准数据集，使用线性加权Cohen's κ和Spearman相关系数衡量点评一致性，使用无权Cohen's κ衡量对比评一致性，利用不同提示策略（zero-shot、few-shot、chain-of-thought）对LLM进行评估，计算模型间一致性并与专家一致性做对比；

**📊 数据集**

数据集为CSEval，包含2,223条仇恨言论、4,318条人工写作的反仇恨回应，配有5位专家在四个维度（相关性、连贯性、攻击性、适宜性）上的评分；

**📈 对比分析**

对比方法：在所有144种模型-提示-维度配置下，计算与专家的一致性；结果显示：最佳配置（Qwen 2.5 72B）在连贯性维度点评时κ_l≈0.19，Spearman ρ≈0.42；在对比评时最高κ≈0.25；与模型间一致性相比，模型间一致性显著更高；

**⚠️ 局限性**

局限性包括：①评估仅基于CSEval的专家评分，未直接获得对比偏好标签；②chain-of-thought提示未生成完整推理轨迹，可能影响评估深度；③多模型对比主要关注同一数据集，泛化性待验证；④模型对某些维度（尤其适宜性）几乎无一致性，提示需进一步改进；

---

## 469. MambaHand: Local Identity and Behavioral Sequence Learning with Mamba for Dynamic Hand Gesture Authentication

**arXiv ID:** 2611.04626 | [PDF](https://arxiv.org/pdf/2611.04626v1)

**作者:** Zenan Lin `[一作]` (South China University of Technology), Wenxiong Kang `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种结合CNN与Mamba的双流框架MambaHand Net，用于动态手势身份验证。

**💡 创新点**

创新性地将第一阶和第二阶时间差序列与双向Mamba结合，构建多阶行为建模；引入注意力引导的行为标记聚合（ABTA）提升身份判别；在共享ResNet18编码器后于L2连接BSS实现细粒度空间与语义平衡。

**🔧 技术方法**

使用ResNet18共享视觉编码器、双向Mamba模块、时间差序列、注意力聚合以及AM-Softmax损失等技术。

**📊 数据集**

在SCUT-DHGA基准数据集上进行实验，该数据集包含193名受试者、29,160段手势视频，采用MG和UMG协议。

**📈 对比分析**

与多种3D/2D CNN方法（DHGA-net、TS-DHGANet、SlowFast等）对比，MambaHand Net在单会话和跨会话条件下均实现了最低的EER（平均0.144%/1.930%），成为当前最优方法。

**⚠️ 局限性**

仅在L2连接BSS时效果最佳，过深的Mamba块会导致过拟合；当前模型相对较大，未实现轻量化，且仅使用视觉通道，未考虑多模态。

---

## 470. SCANO: Restoring Boundary-Condition Completeness in Sampled Meshes with a Sliced-Cross-Attention Neural Operator for 3D Thermal Fields

**arXiv ID:** 2611.04603 | [PDF](https://arxiv.org/pdf/2611.04603v1)

**作者:** Tianyu Li `[一作]` (Nanyang Technological University), Yonggang Wen `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SCANO（Sliced‑Cross‑Attention Neural Operator）架构，在大规模（百万单元）网格上通过子样本训练并在全网格或子区域部署，解决边界条件信息失真问题。

**💡 创新点**

创新点在于：①从积分测度理论出发，系统分析了采样训练对边界条件的三类失真；②给出四个可检验的完整性一致性条件；③设计两种模块——Boundary Quadrature Attention (BQA) 与 Anchored Global State (AGS)，实现面积加权的边界聚合和采样无关的全局状态估计；④在任何骨干网络上可插拔，线性成本完成。

**🔧 技术方法**

核心技术包括：测度理论推导的边界积分表示、注意力视作离散积分算子、面积加权边界四元组的区域化分层（slice tokens）和双重读取（归一化/非归一化）路径；全局状态通过独立锚点采样估计；结合消息传递与切片注意力的 SCANO backbone；多种现有骨干（Transolver、AB‑UPT、AeroGTO 等）与 BQA/AGS 的插件实现。

**📊 数据集**

在三个数据集上验证：①精确解的控制基准（单位立方体，分层网格）；②公开 Poisson 混合边界基准（圆盘）；③工业级数据中心热仿真数据集 DCTS（500 个约百万单元房间，包含多种边界类型）。

**📈 对比分析**

对比方法包括所有现有边界编码器（LFE、BENO、Slice aggregation）和多种骨干网络（Transolver、AB‑UPT、AeroGTO、GNOT、GeoTransolver、MeshGraphNets 等）。实验表明：BQA 能在所有骨干上将全网格误差降低 10‑50%，AGS 使区域预测与全网格保持一致（误差减少 5‑10 倍）；在 DCTS 上，AB‑UPT+ BQA+ AGS 与 SCANO 在整体误差、响应精度与服务器入口温度排名上均居前，且 SCANO 的单通道推理时间 <1 s，速度比全房间自注意力快 100 倍。训练时，采样子图可在单 GPU 上完成并行训练，显著降低显存消耗。

**⚠️ 局限性**

局限性包括：①对非常小的边界片段仍可能出现信息缺失；②全局状态估计依赖锚点数量，锚点太少时误差上升；③BQA 的两条读取路径需要额外参数，若网络容量受限可能影响收敛；④在极大多尺度网格上，仍需进一步优化消息传递层的可扩展性。

---

## 471. SparseGraph-I2P: Consistency-Guided Hypothesis Selection for Efficient Image-to-Point-Cloud Registration

**arXiv ID:** 2611.04591 | [PDF](https://arxiv.org/pdf/2611.04591v1)

**作者:** Ziyang Yu `[一作]` (Institute of Science Tokyo), Jun Miyazaki `[通讯]` (Institute of Science Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6514db3d-8de6-452c-91b7-acdb31787cc4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种高召回、低延迟的图像到点云（I2P）配准框架 SparseGraph‑I2P，利用深度信息在粗到细匹配阶段保持更多高质量匹配，并通过稀疏执行减少计算。

**💡 创新点**

创新点包括：① 结合外观与深度兼容性的粗拟合图（Coarse Hypothesis Graph）来挑选最有可能的粗匹配对；② 通过四叉树和祖先覆盖机制实现稀疏细匹配（Sparse Fine Matching），避免重复解码和比较；③ 在细匹配后使用基于 SC^2 的深度感知筛选，直接传递高质量对应点给单一 EPnP 或 USAC‑MSAC 估计；④ 适配传感器深度与单目深度预测的两条推理路径。

**🔧 技术方法**

核心技术包括：冻结 2D3D‑MATR 主干提取跨模态特征；图注意力网络融合外观与几何支持以评估候选对；点位置信息利用深度反投影构造 3D 表示；SC^2 光谱匹配做一致性排序；点‑点匹配采用互相 top‑k 判定；使用 EPnP（传感器深度）或 USAC‑MSAC（预测深度）进行位姿估计；点‑像素解码采用蒸馏得到的点‑位置 MLP。

**📊 数据集**

实验数据集：RGB‑D Scenes V2（场景 11‑14）和 7‑Scenes（8 个室内环境）。

**📈 对比分析**

与 2D3D‑MATR、Diff^2I2P、GLASS、Hg‑I2P、Ex‑Sim(3) 等六种主流方法对比，SparseGraph‑I2P 在两大基准上均取得最高的注册召回率（RGB‑D Scenes V2 85.6%，7‑Scenes 88.3%）且延迟最低（RGB‑D 33.4 ms，7‑Scenes 36.9 ms）。在传感器深度模式下，单一 EPnP 即可满足 10 cm RMSE 以内的精度；在预测深度模式下，经过校准后仍保持 80% 以上召回率。

**⚠️ 局限性**

局限性：依赖深度信息，传感器深度缺失时性能下降；在极其模糊或重叠严重的场景（如 7‑Scenes 的 Stairs）召回率明显下降；预测深度模式需要额外的校准步骤，增加了计算开销；框架仍假设有足够的点云覆盖，无法处理极度稀疏或动态环境。

---

## 472. The Devil is in the Context: Nine Dimensions of Prompt Injection Attacks

**arXiv ID:** 2611.04672 | [PDF](https://arxiv.org/pdf/2611.04672v1)

**作者:** Jiamin Chang `[一作]` (University of New South Wales), Ahmed Salem `[通讯]` (Microsoft Security Response Center)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种九维度的间接 Prompt 注入分类法，系统性描述注入指令、用户任务与系统上下文的关系，并用此法对已公开的攻击案例、现有基准和防御机制进行分析与评估。

**💡 创新点**

创新点在于：①将注入事件的属性扩展到用户任务与系统上下文，形成更完整的威胁模型；②揭示基准覆盖不足与属性组合缺口；③构建自动化标注与基于分类器的案例生成管道，提升实验规模与覆盖度。

**🔧 技术方法**

使用的技术包括：①手工定义的九维度分类框架；②结合规则与大型语言模型（LLM）实现的自动标注；③LLM驱动的案例生成与修改；④基于现有防御模型（CaMeL、MELON、DeBERTa、PromptArmor、SecAlign-70B）的评测。

**📊 数据集**

使用的主要数据集包括：AgentDojo（Workspace、Banking、Travel、Slack）、IPIArena、LLMail-Inject、WASP、AgentPI、DualView、Sleeper Memory、Trojan Hippo、StepJack 等公开的 Prompt 注入基准。

**📈 对比分析**

通过在原始基准案例与 LLM 生成的改造案例上分别测试上述防御模型，比较攻击成功率与任务完成率。结果显示：在原始案例中大多数防御零成功率；在生成案例中攻击成功率从 17.5% 降至 2.5%（CaMeL）或 7.5%（PromptArmor、DeBERTa），但任务完成率差异显著，表明不同防御在安全与功能平衡上各有取舍。

**⚠️ 局限性**

限制在于：①分类法并非覆盖所有可能的攻击属性；②自动标注对“Source Label”缺乏支持；③评测仅覆盖选定基准与 LLM 版本，未包含所有真实系统；④生成案例的可行性依赖于模型与上下文信息完整度，可能在更复杂场景下受限。

---

## 473. Just Between Us? Exploring a Wearable-Mediated Channel for Discreet Human-Agent Interaction in a Shared Game Setting

**arXiv ID:** 2611.04692 | [PDF](https://arxiv.org/pdf/2611.04692v1)

**作者:** Atikkhan Faridkhan Nilgar `[一作]` (University of Siegen), Kristof Van Laerhoven `[通讯]` (University of Siegen)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在三人卡牌游戏中引入耳机+头部倾斜的可穿戴媒介通道，实现人机之间的隐秘双向信息交换，并评估其可行性与用户体验。

**💡 创新点**

提出一种结合语音消息与右侧头倾斜响应的双向隐秘通道，并在共享游戏情境中系统性验证其协议一致性、压力感知与公平性问题。

**🔧 技术方法**

使用 eSense 耳机（IMU 采集头倾斜）、屏幕对话机器人、Wizard‑of‑Oz 控制界面；通过音频输出与运动检测实现信息发送与响应。

**📊 数据集**

20 对参与者（共 40 名）进行实验，记录协议一致率、IMI 压力/紧张量表、USE 可用性问卷等实验数据。

**📈 对比分析**

与无通道基线对比；结果显示 39/40 次协议一致，压力/紧张无显著差异，所有 USE 维度均超过 4（使用感、易用性、易学习、满意度），表明即时可用性良好。

**⚠️ 局限性**

局限包括：样本以男性大学生为主，缺乏长期使用评估，未验证在物理机器人上的可行性，使用 Wizard‑of‑Oz 控制，且实验仅限短时竞争游戏。

---

## 474. LeanSparse: Memory-Efficient Serving for Native Sparse-Attention LLMs

**arXiv ID:** 2611.04671 | [PDF](https://arxiv.org/pdf/2611.04671v1)

**作者:** Zaifeng Pan `[一作]` (University of California, San Diego), Yufei Ding `[通讯]` (University of California, San Diego)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为"SparseServe"的长上下文稀疏注意力LLM服务系统，显著降低GPU内存占用；

**💡 创新点**

创新点在于两方面：一是层级预填流式双缓冲技术，减少预填阶段的临时内存需求；二是粗细两级索引策略，只在解码时检索原始索引器键，进一步压缩持久内存；

**🔧 技术方法**

技术实现基于SGLang框架，使用产品量化（PQ）压缩索引键，采用双缓冲GPU内存重用、索引器键缓存和KV缓存；

**📊 数据集**

主要使用DeepSeek‑V4‑Flash模型、Open‑SWE‑Traces、LongSWE‑Bench以及GSM8K、NIAH、LongBench v2等数据集进行评测；

**📈 对比分析**

与ECHO、HiSparse等基线对比，SparseServe在950K上下文长度下可支持2.95×并发请求、2.14×解码吞吐量；TTFT从21.6s降至0.6s，整体会话完成时间下降17.7%；

**⚠️ 局限性**

局限性包括：粗细索引仍需额外存储PQ码；预填流式依赖前缀缓存，首次请求仍需完整预填；对极短上下文或非稀疏注意力模型的适用性未评估。

---

## 475. Exploiting Biases in Attention for Transformers

**arXiv ID:** 2611.04657 | [PDF](https://arxiv.org/pdf/2611.04657v1)

**作者:** Charles Herr `[一作]` (LTCI, Télecom Paris, Institut Polytechnique de Paris), Enzo Tartaglione `[通讯]` (LTCI, Télecom Paris, Institut Polytechnique de Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在自注意力中加入输入相关的、每个头可学习的 bias，利用该 bias 作为查询无关的 token 重要性指标，进而在训练与推理阶段按 top‑k 方式裁剪 attention 列，显著降低 FLOPs 与推理延迟；

**💡 创新点**

将 query‑independent attention 结构显式化为可学习的 bias，并用它直接决定列裁剪位置，避免额外的辅助模块或额外训练目标；

**🔧 技术方法**

利用 Transformer 的多头自注意力，加入 per‑head learnable bias，并在前向传播前根据 bias 进行 top‑k 选取，实现键/值的裁剪；

**📊 数据集**

在 ImageNet‑1k 进行分类实验，Mask R‑CNN 在 COCO 进行检测与分割实验；

**📈 对比分析**

与原始 DeiT、ViT-L 以及结构化剪枝、Token Merging (ToME) 等方法对比，bias‑based column pruning 在不同模型规模下可实现约 1.47× 的推理速度提升，且 Top‑1 准确率仅下降 1–3 点；

**⚠️ 局限性**

对极小模型的剪枝更敏感，过度裁剪会显著降低准确率；此外，bias 只在训练阶段有效，若仅在推理阶段裁剪会导致更大的准确率下降；

---

## 476. QueryFlow: Multi-Hop Graph Retrieval with Adaptive Query Evolution

**arXiv ID:** 2611.04625 | [PDF](https://arxiv.org/pdf/2611.04625v1)

**作者:** Manzong Huang `[一作]` (Hefei University of Technology), Xindong Wu `[通讯]` (Hefei University of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为QueryFlow的图检索框架，通过在多跳检索过程中递归地更新查询状态与实体表示相结合，提升多跳问答的证据检索与答案生成性能。

**💡 创新点**

创新点在于将查询条件与实体表示的演化耦合，设计了边界反馈与关系反馈机制，使累积的图上下文能够直接调整后续检索的查询条件；同时采用查询条件的图摘要与门控递归单元实现自适应查询演化。

**🔧 技术方法**

使用图神经网络（消息传递）、门控递归单元（GRU）进行查询演化、边界/关系反馈、软加（softplus）实体评分、倒频率加权的实体到文档映射、列表式损失以及多层感知机进行实体评分。

**📊 数据集**

四个多跳问答基准数据集：2WikiMultiHopQA、HotpotQA、MuSiQue、ConcurrentQA。

**📈 对比分析**

与基于查询的检索（Dense RAG、HyDE、ColBERTv2）、迭代/结构感知检索（FastToG、RAPTOR、ToG、PathRAG）以及图检索方法（G‑Retriever、LightRAG、GraphRAG、DALK、HippoRAG 2、GFM‑RAG、LinearRAG）进行对比。QueryFlow在宏观Recall@10、FullRecall@10、宏观LLM‑Acc上分别比GFM‑RAG提升4.98%、8.40%和11.6个百分点，比LinearRAG提升8.0%和6.9个百分点；在检索时间和API令牌消耗上也实现了显著下降。

**⚠️ 局限性**

主要局限在于依赖预构建的实体图与关系标签，可能在实体识别错误或图稀疏时性能受限；目前仅在四个数据集上验证，尚未评估在更大规模或实时动态图场景下的可扩展性与鲁棒性。

---

## 477. Polyp Matching Using Deep Learning

**arXiv ID:** 2611.04624 | [PDF](https://arxiv.org/pdf/2611.04624v1)

**作者:** Kristian Dalsbø Hindberg `[一作]` (UiT Arctic University of Norway), Esmaeil S. Nadimi `[通讯]` (University of Southern Denmark)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本研究开发了一种基于Siamese网络的对比学习框架，用于在结肠胶囊内镜(CCE)中匹配两个含息肉的图像，判断它们是否为同一息肉。

**💡 创新点**

首次将对比学习的Siamese网络应用于CCE息肉匹配，构建了大规模同息肉图像对数据集，并证明不同CNN骨干网络性能一致。

**🔧 技术方法**

使用ResNet、DenseNet、EfficientNet等CNN骨干的Siamese网络，配合余弦距离和对比损失进行训练，并通过阈值阈值化实现二分类。

**📊 数据集**

利用丹麦CareForColon2015临床试验中导出的1701个息肉的三对同息肉图像与来自不同患者的5103对不同息肉图像，形成平衡训练集。

**📈 对比分析**

在10折交叉验证和最终全数据集测试中，ResNet34、ResNet50等模型达到92%~93%的准确率，AUC约0.98，验证了方法的高效性。

**⚠️ 局限性**

缺少时间间隔较大的同息肉图像，且不同息肉对均来自不同患者，真实临床中同患者息肉的区分更具挑战，导致实验结果可能被高估。

---

## 478. Collaborative Search Meets Large Language Models: An Empirical Study

**arXiv ID:** 2611.04619 | [PDF](https://arxiv.org/pdf/2611.04619v1)

**作者:** Peiyuan Gong `[一作]` (Renmin University of China), Jiaxin Mao `[通讯]` (Renmin University of China)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过一项包含80名学生、40对参与者的实验，系统比较了两种基于大语言模型的协同搜索代理（QAAgent和CoSearchAgent）与两种传统工具（轻量化搜索插件和手动Google搜索）在协同搜索体验、用户行为与最终报告质量上的差异。

**💡 创新点**

创新点在于：①首次在协同搜索场景下对LLM代理与传统工具进行定量与定性全面对比；②设计了CoSearchAgent，能够解析多用户对话、自动改写查询、检索网页并生成可追溯引用的答案；③从工作量、协作体验、代理体验、沟通行为、证据使用和时间表现等多维度评估，揭示LLM代理的优势与局限。

**🔧 技术方法**

使用技术包括：OpenAI GPT-4（或同类LLM）实现问题回答与检索增强生成；Bing/Google搜索API完成网页检索；Slack、Tencent Docs等协作平台；问卷（Workload、Collaboration、Agent Experience）以及访谈收集用户反馈；GPT-5.6-Sol用于评估报告质量。

**📊 数据集**

数据集为实验任务提供的四个开放式主题（AI绘画、幼儿自闭症、浪漫音乐、空气污染），每个任务生成两份独立学习报告，总共320份；此外收集了对话日志、点击记录、搜索结果列表等原始交互数据。

**📈 对比分析**

比较方法：采用拉丁方设计平衡任务-情景分配，使用七分制问卷统计工作量、协作体验与代理体验；对交互日志进行量化分析（对话轮数、证据共享、结果点击、时间延迟）；对报告进行四项评估（Richness、Synthesis、Independence、Gap）并通过统计检验。结果显示：LLM代理在工作量与协作体验上显著优于传统工具（p<0.05），CoSearchAgent在代理体验和时间节约方面最佳，但报告的Richness略低；QAAgent在独立表达上最高。

**⚠️ 局限性**

局限性包括：①LLM代理受限于响应长度与延迟，导致交互时间较长且某些答案缺乏完整性或来源可信度；②实验场景基于学生，缺乏真实业务场景的代表性；③报告质量评估依赖模型判分，可能受模型偏差影响；④对话与检索日志未进行细粒度语义分析，可能遗漏深层次交互模式。

---

## 479. CAPSkip: Cancellation-Aware Prefix Skipping for Selective Backpropagation

**arXiv ID:** 2611.04611 | [PDF](https://arxiv.org/pdf/2611.04611v1)

**作者:** Heejin Choi `[一作]` `[通讯]` (Yonsei University), Heejin Choi (Yonsei University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 CAPSkip，一种在每个 mini‑batch 内根据正类样本最后层梯度的和来决定省略前缀的按步选择规则，从而显著减少反向传播的计算量。

**💡 创新点**

创新点在于：① 通过考虑梯度的相消而非单个梯度大小给出 exact‑arithmetic 对最后层梯度的保证；② 采用全扫描 prefix 并利用 factorized Gram 矩阵在 GPU 上实现 O(B²) 的轻量化前向/后向选择；③ 设计可与 InfoBatch 等外部 pruner 组合的加权形式。

**🔧 技术方法**

技术实现包括：仅使用前向得到的 softmax 残差与特征构造外积梯度；按梯度范数升序排序；利用二维累积和的 Gram 矩阵快速计算 prefix norm；GPU 设备上无 host sync 的全扫描与选择；激活重用与 selective backward 的执行策略。

**📊 数据集**

实验使用 CIFAR‑100 数据集（DeiT‑S fine‑tuning 与 ResNet‑18‑GN from‑scratch），以及 InfoBatch 组合实验；ImageNet‑1k 预训练作为 DeiT‑S 的初始权重。

**📈 对比分析**

对比方式为：在相同 epoch、相同训练时间下与 dense 训练比较准确率与时间；在 fine‑tuning 任务中，CAPSkip 103 轮相当于 dense 59 轮，准确率提升约 +0.25%，训练时间约 0.576× dense；从零训练无显著时间‑准确率差异；与 InfoBatch 组合时，在一实验中实现 0.924× 时间加速，但准确率提升未达到预设阈值。

**⚠️ 局限性**

局限性包括：① 仅对最后层梯度提供保证，深层梯度和整体训练轨迹不受控制；② k* 仅相对排序的最大前缀，非全局最优子集；③ 需要额外内存与 gather，预算饱和仅针对正类样本；④ 实验范围仅限 CIFAR‑100，未验证更大数据集或 LLM；⑤ 需要额外调参且加权组合仍未解决所有准确率不确定性。

---

## 480. Global Trace Condition for Injective Transition Graphs

**arXiv ID:** 2611.04788 | [PDF](https://arxiv.org/pdf/2611.04788v1)

**作者:** Stefano Berardi `[一作]` (University of Turin), Daniel Osorio-Valencia `[通讯]` (University of Turin)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文建立了循环证明系统的全局轨迹条件（GTC）与大小变更终止条件（SCT）的正式对应关系，提出了“转换图”（transition graph）以及“进展域条件”（Progress Domain Condition, PDC），证明在注入性（injective）图的情况下 PDC 与 SCT 等价，从而为验证循环证明的可接受性提供了新的、简化的判定方法。

**💡 创新点**

创新点：① 用转换图代替传统的大小变更图，天然适配循环证明的结构；② 引入注入性转换图对应的 fan‑in‑free SCT，显著简化判定；③ 提出进展域条件（PDC）——比 SCT 更弱但在注入性图上等价的新判定准则；④ 通过 Ramsey 定理证明 PDC 在注入性图上蕴含 SCT，弥补了以往仅给出足够条件的空白。

**🔧 技术方法**

技术手段：形式化定义循环证明与转换图的同构；证明注入性转换图对应的图为 fan‑in‑free；构造 PDC 并与 SCT、GTC 的等价性证明；利用无限 Ramsey 定理的有限色彩版本（或其构造性证明）来完成从 PDC 到 SCT 的蕴含。

**📊 数据集**

本文未使用实验数据集；研究完全是理论证明性工作，主要涉及形式化定义、同构构造以及复杂度分析。

**📈 对比分析**

方法比较：与传统的 GTC 判定（需要处理无穷轨迹）相比，PDC 只需检查有限的自合成关系，理论上更易实现；与一般 SCT（PSPACE‑complete）相比，注入性（fan‑in‑free）SCT 只需指数级搜索；但 PDC 的判定仍然是指数级（相较于基于排名函数的 SCT 算法），具体复杂度仍待进一步研究。

**⚠️ 局限性**

局限性：① 仅在注入性转换图（对应 fan‑in‑free SCT）上成立；② PDC 的蕴含证明依赖非构造性的 Ramsey 定理，未给出有效的算法实现；③ 未给出从循环证明到传统归纳证明的构造性转换；④ 论文未包含实验验证或具体实现细节。

---

## 481. ReSI-Math: Exploring Recoverable Capability Frontiers for Efficient Reasoning Improvement

**arXiv ID:** 2611.04787 | [PDF](https://arxiv.org/pdf/2611.04787v1)

**作者:** Yao Cui `[一作]` (Capital Normal University), Zhenzhou Shao `[通讯]` (Capital Normal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ReSI-Math 框架，利用 Self-Judge 引导的进化搜索在固定参数下探测模型的可恢复能力前沿，并将发现的可恢复问题与 Self-Judge 与 verifier 的评判差异转化为 Solver-RLVR 与 Judge-RLVR 的训练信号，实现递归自我提升。

**💡 创新点**

创新点包括：①操作化可恢复能力前沿的概念；②将 Self-Judge 与进化交叉相结合，发现未被直接生成捕获的隐藏能力；③基于前沿动态重定位的政策自适应顺序 RLVR；④通过前沿问题选择实现极低数据单例 RLVR。

**🔧 技术方法**

采用 RLVR（Group Relative Policy Optimization + LoRA）、进化交叉（CoTEvol 风格）、Self-Judge 评判、rule-based verifier、策略自适应选择机制。

**📊 数据集**

使用 MathBook-Pro（WE-MATH 2.0）作为演化训练池，评测数据集包括 MATH500、WE-MATH testmini、AIME 2025、OlympiadBench、JudgeBench；模型基于 Qwen3-VL-2B/4B/8B。

**📈 对比分析**

通过与固定顺序、全批次联合训练对比，单例 RLVR 与政策自适应顺序 RLVR 的效果被验证；在 Qwen3-VL-8B 上平均从 78.71 提升至 80.16，单例 RLVR 平均提升 0.93 分，优于随机选择方案。

**⚠️ 局限性**

局限性包括：可恢复前沿仅为经验操作，缺乏严格的理论界定；实验仅在 Qwen3-VL 与数学推理任务上验证，未探究跨模型/跨域的推广；Solver 与 Judge 分别训练，未验证联合优化的潜在优势。

---

## 482. Near-linear-Time Sampling of the Ferromagnetic Ising Model at the Tree Uniqueness Threshold

**arXiv ID:** 2611.04786 | [PDF](https://arxiv.org/pdf/2611.04786v1)

**作者:** Tianshun Miao `[一作]` `[通讯]` (Nanjing University), Tianshun Miao (Nanjing University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了近线性时间的采样算法，能够在树唯一性区域（包括临界点）内对零场费米磁性伊辛模型、XY模型以及整数Potts模型进行采样。

**💡 创新点**

突破了传统Markov链（如Glauber、Swendsen–Wang）在临界点处的慢速混合瓶颈，首次通过将模型转化为聚合物（polymer）表示并证明其满足谱条件，从而获得近线性或几乎线性工作量；并在不同模型间统一了采样框架。

**🔧 技术方法**

核心技术包括：
- 将偶子图/流/流量表示改写为聚合物模型；
- 证明聚合物模型满足严格的谱上界与下界条件；
- 采用插入-删除链（insertion–deletion chain）结合随机游走/探索技术来构造采样提案；
- 利用低阶近似、潜能函数与Sobolev不等式等工具保证混合速度。

**📊 数据集**

无具体实验数据集；算法在理论上适用于所有最大度数为 Δ 的无向图（n 个顶点、m 条边）。

**📈 对比分析**

与现有的 Glover、Swendsen–Wang 等传统Markov链比较：在树唯一性区间内，该算法的期望工作量为 O(n + m·log^3(n/ε))（伊辛）、O(m + n·exp(O(√log(n/ε))))（XY）以及 O(n + q^O(1)·m)（Potts），实现了近线性或几乎线性时间；相较于此前仅在严格低温/高温区间给出的多项式时间，显著提升了效率。

**⚠️ 局限性**

限制：
- 对伊辛模型要求最大度数 Δ≥3；
- 对XY模型要求 Δ≥8，且工作量对 ε 的依赖为 exp(O(√log(n/ε)))；
- 对Potts模型的高温范围仅覆盖到 β≤1+0.1(β_{q,Δ}−1)，即相对临界点只有常数比例；
- 需要先验证聚合物模型的谱条件，若不满足则无法直接应用；
- 对连续旋转模型的实现仍需在实数算术模型下操作，实际实现时可能需要更细致的数值稳定性考量。

---

## 483. Koopman Generative Operators for Efficient Probabilistic Time-Series Forecasting

**arXiv ID:** 2611.04760 | [PDF](https://arxiv.org/pdf/2611.04760v1)

**作者:** Raz Marshanski `[一作]` (Ben Gurion University Of Negev), Omri Azencot `[通讯]` (Ben Gurion University Of Negev)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种基于Koopman理论的概率时序预测框架KGO，将预测视为结构化不确定性随时间演化；

**💡 创新点**

创新点在于三大核心组件：Koopman Patch Embedding (KoPE) 结构化潜在轨迹外推，Koopman Flow Matching (KoFM) 用闭式矩阵指数实现单步高效生成，Adaptive Uncertainty Gate (AUG) 对每个变量和预测步长自适应分配确定性与不确定性；

**🔧 技术方法**

技术主要包括Reversible Instance Normalization、sLSTM编码器、Koopman线性算子、矩阵指数闭式推移、门控融合与CRPS+量化损失训练；

**📊 数据集**

在ProbTS基准上使用17种配置（ETT、Electricity、Exchange、Traffic、Weather、Solar、ILI），涵盖短期与长期预测；

**📈 对比分析**

与八类基线（流、扩散、VAE、latent动态）以及三大基线无监督模型比较，KGO在13/17配置CRPS最低、16/17 NMAE最低，单步推理速度比迭代生成快≈25×，显著提升精度与效率；

**⚠️ 局限性**

局限在于对Koopman算子学习的线性假设可能不足以捕捉极其复杂的非线性动态，对非常长时间跨度的预测仍需进一步验证，且对高维多变量场景的可扩展性尚待深入。

---

## 484. Non-Interfering Asymmetric AdamW Updates for Constrained Exploration of Equifinality in Hybrid Models

**arXiv ID:** 2611.04756 | [PDF](https://arxiv.org/pdf/2611.04756v1)

**作者:** Ekaterina Gikalo `[一作]` (Max Planck Institute for Biogeochemistry), Marco Körner `[通讯]` (German Aerospace Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文通过在混合模型中引入辅助排斥损失，构建一组能够在保持对观测数据拟合不变的前提下，显著扩展未观测蒸散分解等价性的多样性训练方法。

**💡 创新点**

创新点在于提出一种针对 AdamW 的非干扰更新 gl:NANO，它在保持主任务一阶下降不变的同时，利用预处理、投影与长度约束，将辅助排斥信号聚焦到主梯度未活跃的坐标，进而实现等价性多样性。

**🔧 技术方法**

使用的技术包括：基于神经网络嵌入的过程模型（gl:H2CM）、辅助排斥损失、AdamW 优化器矩估计、梯度投影（PCGrad/Bloop 的变体）、二阶约束与归一化。

**📊 数据集**

实验采用 gl:H2CM 模型，利用全球蒸散（ET）观测数据（每日一次、单空间分辨率），并计算蒸散分解中蒸腾比例 R = T/ET。

**📈 对比分析**

与基线、PCGrad、Bloop 的比较表明，gl:NANO 在保持测试集损失与基线相当（均值差异在 2% 以内）的同时，R 的标准差从 0.009 提升至 0.143，分布宽度从 0.031 提升至 0.491，性能最佳。

**⚠️ 局限性**

局限性包括：非干扰性仅在单步保证，可能导致累计漂移；二阶约束基于梯度幅度而非真实曲率；排斥仅在最近邻之间，得到的是等价性下限；需要额外一次反向传播导致计算开销；参数 τ 需要针对每个诊断手动设定。

---

## 485. Bounded AUKS: A Novel Evaluation Metric for Anomaly Detection via Bounded Area Under the Kolmogorov--Smirnov Curve

**arXiv ID:** 2611.04754 | [PDF](https://arxiv.org/pdf/2611.04754v1)

**作者:** Yanshuo Zhang `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Juan Du `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种新的评价指标Bounded AUKS，用于衡量异常检测模型的分数分布分离程度并结合无异常阈值区间进行评估。

**💡 创新点**

创新点在于：①将Kolmogorov–Smirnov曲线的积分（AUKS）与阈值区间结合形成Bounded AUKS；②阈值上下限仅由正常样本的分数统计（Median+α·MAD与基于极值理论的EVT上界）决定，避免依赖异常标签；③提供了对分数分布可分离性的直接定量度量。

**🔧 技术方法**

技术手段包括：Kolmogorov–Smirnov积分、1‑Wasserstein距离、极值理论（POT–GPD）上界估计、稳健统计（Median、MAD）、阈值区间积分与分数分布分析。

**📊 数据集**

实验使用工业级异常检测基准数据集MVTec AD和VisA，共覆盖15类与12类物体，包含多种真实缺陷。

**📈 对比分析**

与传统指标（AUROC、AUPRO、AUKS@100%）对比，Bounded AUKS在两数据集上能更好地区分模型性能，排名与视觉效果更一致，且不受阈值范围或异常率漂移影响。

**⚠️ 局限性**

局限性：需要选择α与q的超参数；对极值建模的准确性敏感；在分数分布极其重叠时，Bounded AUKS仍可能不足以完全反映模型实际表现。

---

## 486. Mass Scaling Laws for Energy-Optimal Horizontal Multirotor Flight

**arXiv ID:** 2611.04748 | [PDF](https://arxiv.org/pdf/2611.04748v1)

**作者:** Ayush Patnaik `[一作]` (University of California), Xinfan Lin `[通讯]` (University of California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了多旋翼无人机在水平飞行中，质量变化对能量最优飞行行为的影响，推导并验证了能量最优速度、俯仰角与能量消耗与质量之间的比例关系；

**💡 创新点**

首次给出三条完整的质量缩放定律：能量最优速度与质量的平方根成正比、最优俯仰角不随质量变化、能量最优消耗与质量线性相关，并证明其来自稳态动力学的质量相似性；

**🔧 技术方法**

采用物理建模（刀片元素/动量理论与刚体动力学）求解稳态力平衡，推导公式；随后利用数值仿真与实验验证；

**📊 数据集**

在两台不同尺寸多旋翼平台（八旋翼与四旋翼）上，分别通过负载调节实现 3–44% 的质量变化，收集飞行速度、俯仰角、电流/电压等原始飞行数据；

**📈 对比分析**

将理论预测与实验测得的 V*、θ*、EPM* 进行对比，计算系数变异度 CoV，结果均小于 5%，验证了缩放定律的准确性；

**⚠️ 局限性**

仅适用于质量单一变化、固定机体几何与气动特性、稳态水平直飞；不考虑重心漂移、机身形变、瞬态加速或姿态变化、以及电机/ESC 效率随速度变化等因素。

---

## 487. TAPES: Task-Aligned Joint Predictive Pretraining for Transaction Event Sequences

**arXiv ID:** 2611.04747 | [PDF](https://arxiv.org/pdf/2611.04747v1)

**作者:** Ruobing Zuo `[一作]` (East China Normal University), Tianyi Zhang `[通讯]` (Ant International)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于字段-操作组合构造的目标集选择与共享离散隐变量联合预测预训练框架（TAPES），通过短期预训练评估目标集的下游表示效用，并在此基础上对交易历史序列进行联合预训练，提升序列表示的任务适用性。

**💡 创新点**

创新点在于将交易字段的多重预测语义化为可选目标集，利用下游任务的表示效用进行目标集自动化选择，并通过共享离散隐变量实现目标之间的联合依赖建模，从而实现任务对齐的联合预训练。

**🔧 技术方法**

采用的技术包括字段-操作组合构造目标、短期预训练+线性探针评估目标集效用、共享离散隐变量混合模型的联合似然预训练、以及对目标缺失进行边缘化处理的损失设计。

**📊 数据集**

实验使用了五个公开金融/电商序列基准（AlphaBattle、Retail X5、MBD-mini、Rosbank Churn、Taobao）以及内部大规模交易风险数据集 TransRisk。

**📈 对比分析**

与基线（树模型、监督序列模型、CoLES、TabBERT、MLEM、PANTHER等）比较，TAPES在所有基准上获得最高或接近最高的主要指标，平均排名1.20，单指标提升约0.2–0.6个百分点。

**⚠️ 局限性**

主要局限包括候选目标集评估的计算成本、仅使用离散隐变量进行联合建模，未考虑动态目标生成或跨领域的泛化能力。

---

## 488. A RISC-V Simulator for the Tenstorrent Tensix Architecture

**arXiv ID:** 2611.04740 | [PDF](https://arxiv.org/pdf/2611.04740v1)

**作者:** Nick Brown `[一作]` (University of Edinburgh), Daniyal Arshad `[通讯]` (University of Edinburgh)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了Wolfpine，一个基于纯Python的开源RISC‑V模拟器，用于Tenstorrent的Tensix架构（Wormhole和Blackhole），支持TT‑Metal堆栈并提供详细观测与可选周期成本模型；

**💡 创新点**

通过完全依赖公开文档、开源软件和硬件测量构建独立模拟器，提供可插拔的周期成本模型和事件驱动的观测工具，使开发者能够在无物理硬件的情况下调试、性能分析并轻松扩展到其他RISC‑V加速器；

**🔧 技术方法**

Python面向对象的块抽象、事件驱动时钟与可选成本模型、Typed event bus、FlatBuffer通信协议反向工程、Perfetto/Parquet/Cachegrind等性能追踪与导出；

**📊 数据集**

使用TT‑Metal示例程序（如Nekbone、Nek5000）、自制BF16矩阵乘法及NoC延迟基准、硬件周期计数、NoC/DRAM吞吐等测量数据作为评测数据集；

**📈 对比分析**

采用功能一致性（PCC）与壁钟时间/加速比对比，Wolfpine在功能上与官方t tsim及真实p150 Blackhole硬件相匹配（PCC>0.98），性能上相对官方C++实现慢3.5–192×，相对硬件慢3.5–192×，但提供更丰富的调试与观测能力；

**⚠️ 局限性**

Python实现导致运行速度慢；成本模型仍需改进（如整数除法误差）；未实现多芯片/以太网等功能；对完整公开文档的依赖，如果文档不完整则难以复现；

---

## 489. VISAGE: Vertex-based Intrinsic Surface Appearance GEneration via Dual-Stream Diffusion for Arbitrary 3D Heads

**arXiv ID:** 2611.04712 | [PDF](https://arxiv.org/pdf/2611.04712v1)

**作者:** Jeremias Krauß `[一作]` (German Research Center for Artificial Intelligence), Didier Stricker `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

从稀疏 RGB 图像预测任意拓扑人头网格的顶点 albedo 与 roughness，直接生成 PBR 材质，无需 UV 参数化。

**💡 创新点**

提出双流表面扩散模型，结合噪声条件谱扩散、地理 KNN 细节重建和可视化多视图交叉注意力，支持区域特化和切线空间细化，实现高质量、细节丰富的表面纹理。

**🔧 技术方法**

使用 DiffusionNet+EDM 预处理的表面扩散网络，噪声级条件谱扩散、地理 KNN 自注意、可视化多视图交叉注意、区域特化模块、切线空间细化和程序化物理渲染数据增强。

**📊 数据集**

基于 NPHM 头部扫描，利用程序化增强生成 PBR albedo/roughness 数据；在 NPHM、FaceVerse、i3DMM 等数据集上进行评估。

**📈 对比分析**

与 MonoNPHM、Hunyuan3D‑2.0‑Paint、DiffusionNet* 进行比较；在 NPHM 上 FID 降 42.7%，在 FaceVerse 降 22.5%，在 i3DMM 降 4.4%，同时 SSIM、LPIPS、PSNR 亦优或相近；用户研究亦显示 VISAGE 更受青睐。

**⚠️ 局限性**

光照仍可部分混入色彩、细节受顶点密度限制、粗糙度监督为程序化、对极度遮挡区域依赖扩散先验、未预测高频法线，且在强光/阴影下可能出现色彩泄漏。

---

## 490. Cycle WAM: Coupling Action and Video through Cycle Consistency

**arXiv ID:** 2611.04711 | [PDF](https://arxiv.org/pdf/2611.04711v1)

**作者:** Qinglin Yang `[一作]` (Zhejiang University), Xuelong Li `[通讯]` (Institute of Artificial Intelligence TeleAI China Telecom)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在视频与动作之间学习对应关系，提出Cycle WAM统一训练框架

**💡 创新点**

通过双向转换与循环重建强化视频-动作对应性，提升预训练视频先验利用

**🔧 技术方法**

使用双向流匹配损失、循环一致性、LoRA适配器等技术

**📊 数据集**

在LIBERO‑Pro、DROID和LIBERO等数据集上训练与评估

**📈 对比分析**

与UWM、DreamZero等方法对比，控制成功率和前向动态质量均有提升，尤其在某些扰动条件下表现最佳

**⚠️ 局限性**

依赖预训练视频模型的质量，缺乏物理/几何先验，且跨任务泛化仍有限

---

## 491. DenseTwins: Dense Modality Decoupling for Multimodal Self-Supervision

**arXiv ID:** 2611.04710 | [PDF](https://arxiv.org/pdf/2611.04710v1)

**作者:** Carlo Sgaravatti `[一作]` (Politecnico di Milano), Giacomo Boracchi `[通讯]` (Politecnico di Milano)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DenseTwins，一种多模态自监督框架，能够在保持空间结构的前提下学习稠密表示，并显式将嵌入空间分为共享子空间和模态特定子空间。

**💡 创新点**

创新点在于：1）将模态解耦与稠密对齐统一到同一框架；2）利用稠密对应关系构建跨模态一致性，同时在共享子空间执行冗余消减，模态特定子空间保持互相正交；3）结合全局与局部（稠密）两层监督，提升对稠密任务的适用性。

**🔧 技术方法**

技术手段包括：双分支编码器+投影头（全局+稠密投影）；跨模态稠密对应（基于特征相似度匹配）；冗余消减目标（Barlow Twins 风格）用于内部一致性；模态解耦目标（对齐共享子空间、正交模态特定子空间）；以及多任务损失的加权组合。

**📊 数据集**

使用的多模态数据集：RGB–Depth (SUN‑RGBD)、RGB–Event (EventScape, DSEC)、RGB–LiDAR (Semantic‑KITTI)、RGB–DEM (GeoNRW)、SAR–Optical (BigEarthNet)、RGB–Thermal (KAIST)。

**📈 对比分析**

与 SimCLR、CLIP、Barlow Twins、VICReg、DeCUR、DINO、iBOT、I‑JEPA、LeJEPA 等基线比较，DenseTwins 在 2D 语义分割、目标检测、遥感多标签分类等下游任务中均取得最优或竞争性结果（如 SUN‑RGBD mIoU ≥ 38.2%，KAIST/ DSEC AP50 > 33%，BigEarthNet mAP 0.834）。

**⚠️ 局限性**

局限性：①跨模态对应在早期训练中噪声大，需要预热；②共享/特定特征比例需手工设定，缺乏自适应机制；③主要针对空间组织的模态，对文本等非空间模态不适用；④整体计算开销较大（双投影头+稠密对应）；⑤与纯对比学习方法相比仍需进一步验证在更大规模跨模态对齐中的表现。

---

## 492. When Ground Truth Is Not Ground Truth: Automatic Labeling for Hallucination Detection

**arXiv ID:** 2611.04697 | [PDF](https://arxiv.org/pdf/2611.04697v1)

**作者:** Rounak Sen `[一作]`, Deeksha Varshney `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估15种自动监督标签对幻觉检测器性能的影响，并用人工验证检验标签可靠性。

**💡 创新点**

揭示相似度基标签的局限性、跨监督标签导致的性能差异及与人工黄金标签的误差率，为幻觉检测提供更可靠的监督方法。

**🔧 技术方法**

采用多语言生成、15种自动监督判别器、7种检测基线（如DRIFT、SAPLMA）、跨标签迁移实验、人工标注验证和AUROC评估等技术。

**📊 数据集**

TruthfulQA和SciQ的英文版及其翻译版本，覆盖英语、印地语、孟加拉语、泰米尔语、泰卢固语，约2,800条问答对。

**📈 对比分析**

通过AUROC比较检测器在不同监督标签下的性能，发现性能随监督标签显著变化，跨标签迁移率低；人工验证显示BLEURT等相似度标签误差率高达47%，而QGen-9B误差率仅12.9%。

**⚠️ 局限性**

依赖机器翻译的多语言数据、仅评估线性单样本探测器、未涵盖多样本方法、仅在公开问答基准上验证。

---

## 493. What Does It Mean to Remove an Edge? Intervention-Aware Fidelity for Graph Neural Network Explanations

**arXiv ID:** 2611.04751 | [PDF](https://arxiv.org/pdf/2611.04751v1)

**作者:** Rocco Mattia Di Mauro `[一作]` (University of Catania), Giuseppe Mangioni `[通讯]` (University of Catania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究图神经网络解释中的删除信度（deletion fidelity），区分结构删除（structural deletion）和消息通道抑制（message‑channel suppression）两种干预方式，并对它们的等价性进行形式化分析；随后设计交叉实验（construction‑by‑evaluation）在多种GNN模型上评估不同干预对解释排名的影响。

**💡 创新点**

创新点在于：1）首次把删除干预拆分为两种可实现的操作，并推导出它们一致的充分条件；2）提出交叉实验框架，让解释器与评估器使用不同干预，量化“匹配优势”（matched superiority）和交互效应；3）通过大量实验验证，在GIN模型下两种干预等价，而在GCN/GATv2等模型中解释排名的删除信度依赖于所选干预。

**🔧 技术方法**

使用的技术包括：图神经网络（GIN、GCN、GATv2）与集成梯度（Integrated Gradients）以及GNNExplainer；对解释生成使用两种门控注入点（结构层前或消息层后）；在交叉实验中计算删除信度面积（fidelity area），并统计交互效应 D = (F_SS‑F_MS)+(F_MM‑F_SM)。

**📊 数据集**

数据集涵盖九个图分类任务：四个合成数据集（BA‑2Motifs、BA‑MultiShapes、SPMotif、Synth‑Dist）和五个真实世界数据集（BACE、MUTAG、Mutagenicity、Reddit‑Binary、Tox21）。

**📈 对比分析**

比较方法为 2×2 交叉设计：解释器使用结构或消息干预构造排名，评估器再使用结构或消息干预测量删除信度。结果显示：在 GIN（控制组）下两种干预得到一致的删除信度；在 GCN 与 GATv2 中，约 10/11 个细胞表现出正交互效应，说明每个排名在与其匹配的干预下通常具有更大的删除信度面积。GNNExplainer 的结果相对不稳定，匹配优势仅在 4/11 个细胞出现。

**⚠️ 局限性**

局限性包括：1）干预的意义依赖于具体问题，无法统一为“最佳”；2）实验仅覆盖图分类任务和三种主流 GNN 架构，无法推断到其它任务或模型；3）样本量有限（10 个种子），bootstrap 估计不稳健；4）解释方法（IG、GNNExplainer）本身的完整性与排名稳定性不足；5）对输出函数、目标类、扰动单位等参数的选择对结论有较大影响。

---

## 494. RESTberus: Trace-Aware Benchmarking of Automated REST API Testing Tools Across Microservice Dependencies

**arXiv ID:** 2611.04769 | [PDF](https://arxiv.org/pdf/2611.04769v1)

**作者:** John Toniutti `[一作]` (University of Udine), Vincenzo Riccio `[通讯]` (University of Udine)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于 Kubernetes 的 benchmarking 平台 RESTberus，用于在完整的微服务部署环境中评估黑盒 REST API 测试工具，并通过分布式追踪实现对目标服务与下游服务失败的区分与去重。

**💡 创新点**

①首次将分布式追踪与传统的响应体相似度和操作匹配相结合，提出三种“trace‑aware”去重策略；②在保持对现有单服务基准兼容性的前提下，扩展到多服务集成测试；③通过实验量化下游失败占比并提供工具间可比性分析。

**🔧 技术方法**

技术栈包括 Kubernetes + Helm（服务部署与隔离）、Istio/Envoy（链路追踪）与 Jaeger（分布式追踪）、mitmproxy（入口点拦截）、JaCoCo（代码覆盖）、SQLite（结果存储）以及多模型 LLM 判定（Deduplication 参考标签）。

**📊 数据集**

使用了两大微服务基准 Train Ticket（41 组微服务）和 PetClinic（7 组微服务），并对 9 个单服务基准（来自原始 REST benchmarking 仓库）进行兼容性验证；实验覆盖 6 种自动化测试工具。

**📈 对比分析**

对 5 种去重策略在 LLM 与人工标注数据集上进行准确率、精确率、MCC 的对比；在 6 个工具上评估操作覆盖率、代码覆盖率和独立失败数；并与传统单服务平台进行等价性检验。结果显示，trace‑aware 去重策略达 87.1% 的准确率，提升 13–20%；工具平均发现 11.8% 的失败为下游失效；在单服务基准上两平台无显著差异，满足实际可比性。

**⚠️ 局限性**

实验局限于 Java/Spring 生态，未覆盖多语言或多协议环境；禁用了 Circuit‑Breaker 回退，导致在生产环境下可能低估下游失败比例；LLM 参考标签仍可能存在误判，需要进一步人工验证；平台部署与追踪开销在极大规模系统中尚未评估。

---

## 495. Testing for a Large Induced Expander: Sublinear Algorithms and Query Lower Bounds

**arXiv ID:** 2611.04766 | [PDF](https://arxiv.org/pdf/2611.04766v1)

**作者:** Pan Peng `[一作]` (University of Science and Technology of China), Ranran Shen `[通讯]` (University of Science and Technology of China)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计了一种子线性查询的性质测试算法，用来判定是否存在一个在常数比例内的诱导子图（核心）具有良好的内部导电率（内导率），并给出了相应的上界与下界，证明了在随机行走基础上的接近最优的查询复杂度。

**💡 创新点**

创新点主要在于：①提出只需探测图中一部分（核心）的扩散性质而非全图，②利用“追踪链”与停顿时间技术把随机行走的端点分布与未知核心的统一分布联系起来；③在构造难度下界时使用多值奇偶标签（𝔽₂ʳ）与平均化技术，扩展了噪声奇偶性构造，证明了对任意尺寸松弛因子 κ 的 Ω(√n+ c_{κ,ϕ}μ) 复杂度下界；④在距离约束下实现了核心大小保持不变的测试。

**🔧 技术方法**

核心技术包括：
- 递归随机行走与平均长度随机行走 Q_ℓ 的使用；
- 追踪链（trace chain）在核心内部的快速混合性证明；
- 端点分布相互碰撞检测（collision test）与列表长度调度；
- 停顿时间比较与 TV 范数收缩；
- 噪声奇偶性构造、字符傅里叶分析与循环回路计数；
- 通过“平衡化”与“补丁”技术将核心大小恢复到原始 αn，保持最大度数不变。

**📊 数据集**

本工作为理论性质测试，无需实际数据集；所有结果均在理论模型（带度数上限 d 的无向图）中给出。

**📈 对比分析**

与早期的扩散测试（O(√n) 查询，导电率损失 O(ϕ² / log n)）相比，该算法在核心大小松弛下实现了接近最佳的 √n + O(√n log n / ϕ²) 查询；对距离约束下的测试保持了同样的核心大小，且在噪声率较低时实现了相同的复杂度。下界证明表明，除去 loglog n 的细微差距外，√n 级查询已经是必要的。

**⚠️ 局限性**

限制与开放问题包括：
- 仍存在 loglog n 的缺口，未能完全消除对 ϕ 的对数惩罚；
- 仅提供存在性判定，未给出核心的具体结构或构造；
- 需要图的最大度数为常数，扩展到更一般度分布的情况尚未探讨；
- 负实例构造依赖于较大的度数（d = 512），在低度数下的下界仍需进一步研究。

---

## 496. Kierston & KierstonBench: Simulation of Allentown, PA and Evaluation of Language Models in Crime Investigation and Surveillance Tasks

**arXiv ID:** 2611.04785 | [PDF](https://arxiv.org/pdf/2611.04785v1)

**作者:** Andrew Semenov `[一作]`, Svyatoslav Dorofeev `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为Kierston的城市规模仿真引擎，模拟了宾夕法尼亚州阿伦敦市约12.5万居民的日常行为，并从这些行为中生成了电信、金融、行政和车牌记录，随后基于该仿真创建了32个任务族的基准，用来评估大型语言模型在记录链接、日常重建、事件调查和欺骗检测等犯罪调查场景下的推理能力。

**💡 创新点**

首次提供了完整的城市级仿真与真实记录接口的结合，确保所有记录都有可追溯的生成轨迹和已知真相；设计了可供模型查询的受限记录接口、场景DSL和任务族，形成了可复现、可扩展、可测量的基准；在基准中引入了记录误差、摄像头误读、网络信号定位等真实系统噪声，提升了评测的可信度。

**🔧 技术方法**

采用人口微观仿真技术（基于ACS PUMS、OpenStreetMap、FCC天线注册等公开数据），结合交通网络、通信网络（LTE/NR）和车牌识别模拟；利用规则驱动的行为调度和时间线合成技术生成每日活动；通过DSL脚本对事件进行增删改，并在跑日文件中重写受限记录；评估时使用代码执行框架（Python、ClickHouse、PostgreSQL）与模型交互，记录查询日志和脚本。

**📊 数据集**

主要数据集包括：ACS 5年PUMS 42-02803（Allentown）、OpenStreetMap 5 km 区域提取、FCC Antenna Structure Registration、OpenCelliD、NHTS 2022 + EPA 车辆数据、GSMA TAC、Pew Research 平台覆盖率、宾夕法尼亚州法律法规等。

**📈 对比分析**

通过将16个主流大语言模型（Claude、GPT、Gemini、MiniMax、DeepSeek、Qwen、Grok等）在32个任务族中进行评测，采用严格的记录接口限制和时间截止，计算得分（-1~1）。在技术任务上最佳模型得分0.75，实时调查任务平均负分-0.55，欺骗任务平均正分0.67-1.0。整体平均分0.36，表明模型在基准中普遍难以做出可信结论，尤其在实时调查与欺骗检测场景表现欠佳。

**⚠️ 局限性**

局限性包括：1）模型无法充分利用间接证据（如网络信号、车牌误读）进行位置推理；2）对实时性和信息缺失的判断不足，往往出现错误结论或过度自信；3）基准依赖仿真生成的记录，缺少真实系统中的多样性与噪声；4）评测只关注单一任务答案，未覆盖模型在实际调查流程中的协同与决策过程；5）未针对模型在部署环境中的抗诱导和安全性进行深入测试。

---

## 497. Differentially Oblivious Priority Queues with Sublogarithmic Cost

**arXiv ID:** 2611.04776 | [PDF](https://arxiv.org/pdf/2611.04776v1)

**作者:** Zimai Guo `[一作]` (Peking University), Zhechen Li `[通讯]` (Ant Group)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

构建了一种在差分隐私下实现的整数优先队列，能够在在线模式下支持插入和最小提取，并且仅使用多项式小的私有内存

**💡 创新点**

首次突破完全可观测访问模式下对优先队列的Ω(log T)工作下限，提出差分可观测下的O(log T/ log log T)期望工作量，并证明其在总工作量上与下界匹配，展示了差分隐私可显著降低数据结构访问复杂度

**🔧 技术方法**

采用分层段树与基数分段、差分私有摘要、基于递归几何采样的噪声机制、预取本地集合、整数集合索引、以及精确的二进制树计数等技术，形成完整的基于 epoch 的预取和合并框架

**📊 数据集**

无实际数据集，本文为理论分析与构造；实验与评估在假设的 balls‑and‑bins 模型与理想化 RAM 计算机上完成

**📈 对比分析**

相较于传统完全可观测实现的 O(log T) 访问复杂度，本文实现了期望 O(log T/ log log T) 的总工作量，并且在私有空间上仅需 O(log^14 T) 单词；与全可观测的下界相匹配，证明其在此模型下为最优

**⚠️ 局限性**

局限性包括：仅在 balls‑and‑bins 模型下证明，需公开操作类型序列；隐私参数受限于 η=Θ(1)、δ=exp(-Θ(log²T))；实现依赖于整数集合操作的常数时间；并且对真实系统的具体实现细节与外部 I/O 模式未给出

---

## 498. Learning from Unlabeled Photographs for General Oral Image Analysis

**arXiv ID:** 2611.04709 | [PDF](https://arxiv.org/pdf/2611.04709v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 499. Tree Covers for Directed Graphs: Distance Oracles, Routing Schemes and Spanners

**arXiv ID:** 2611.04731 | [PDF](https://arxiv.org/pdf/2611.04731v1)

**作者:** Shiri Chechik `[一作]` (Tel Aviv University), Gur Lifshitz `[通讯]` (Tel Aviv University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了在有向图中构造 k‑strong（强）树覆盖的算法，并利用该覆盖构造了回路（roundtrip）子图逼近器、路径报告距离查询器和紧凑路由方案；此外，还给出了能在 O(min{mn, n²·⁵}) 预处理时间内完成构造的实现。

**💡 创新点**

创新点包括：①将 Thorup–Zwick 的无向图树覆盖框架直接推广到有向图；②不再按几何尺度分层，而是使用单一统一层次完成所有距离的近似；③通过严格集（strict cluster）和扩展严格集（expanded strict cluster）实现对有向路径的有效近似；④利用稀疏化技术（Chechik–Lifshitz）在密集图中将构造时间从 O(mn) 降至 O(n²·⁵)；⑤在覆盖、子图逼近器、距离查询器和路由方案中都实现了与无向图等价的空间–伸展（stretch）与重叠（overlap）权衡。

**🔧 技术方法**

核心技术包括：随机层次抽样、严格集与扩展严格集的定义与迭代修剪、快速集（fast cluster）的单轮修剪、稀疏化（sparsification）与消除器（eliminator）思想、递归采样（recursive sampling）用于空间控制，以及低凸包与跳跃指针（jump pointers）加速查询。

**📊 数据集**

本文为理论工作，未在具体数据集上进行实验验证；所有结论均基于理论分析与概率期望。

**📈 对比分析**

与现有工作相比：①实现了与无向图 Thorup–Zwick 相同的 2k−1 伸展与 O(k n^{1+1/k}) 边/空间上限；②比 Roditty–Thorup–Zwick 的 2k+ε 伸展和 O(k² n^{1+1/k} log(nW)) 边/空间要好；③提供了 O(k) 查询时间（或 O(loglog k)）的距离查询器，且比之前的 O(log k) 查询时间更快；④路由方案在每个顶点仅需要 O(k n^{1/k}) 单词存储，并且在大部分顶点上实现 O(1) 转发时间。

**⚠️ 局限性**

局限性：①需要有向图的非负权重并且在强连通分量上处理；②对 k < 4 的情况未给出快速构造（需要 k≥4 才能达到 O(n²·⁵) 的上限）；③算法仍然依赖随机化，理论期望时间与重叠保证；④在非常稀疏的图中，构造时间可能仍接近 O(mn)；⑤对极端大权重范围（W→∞）的处理需要额外注意。

---

## 500. Self-stabilising approximate clock synchronisation requires linear time

**arXiv ID:** 2611.04725 | [PDF](https://arxiv.org/pdf/2611.04725v1)

**作者:** Joel Rybicki `[一作]` `[通讯]`, Joel Rybicki

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在同时满足自稳性（self‑stabilisation）和崩溃容错（crash faults）条件下的近似时钟同步问题，证明了任何确定性协议至少需要Ω(f)轮才能稳定，其中f为允许的最大崩溃进程数。

**💡 创新点**

创新点在于首次使用组合拓扑（combinatorial topology）方法，证明了自稳性近似时钟同步任务的时间下界，并将该下界与传统一致性（consensus）和集合一致性（set agreement）等经典任务进行了比较。

**🔧 技术方法**

主要技术包括构造同步协议复杂体（full‑information protocol complex）X和Y，利用单连通性证明边梯度（edge cocycle）可被写成潜能函数，随后通过前缀/后缀映射关联不同轮次的视图，推导出升维的一致性矛盾。

**📊 数据集**

论文为理论性工作，未使用任何实验数据集。

**📈 对比分析**

由于是理论证明，未涉及实验对比；但作者指出现有最优上界（⌈f/2⌉+O(log q)）与下界一致，证明了算法的时间复杂度是最优的。

**⚠️ 局限性**

限制在于仅适用于完全图（全连通）同步消息传递模型；对于一般网络拓扑的下界仍是未知的，同时该证明仅适用于确定性算法，随机化算法的下界仍未给出。

---

## 501. Pure Differentially Private Query Release via Laplace Input Smoothing

**arXiv ID:** 2611.04721 | [PDF](https://arxiv.org/pdf/2611.04721v1)

**作者:** Zongrui Zou `[一作]` `[通讯]` (Nanjing University), Zongrui Zou (Nanjing University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了一种高效的纯差分隐私机制，用于离线发布任意规模统计查询集，并在多项式时间内实现误差上界为O(√(log d log k/N))，即与信息理论下限一致；

**💡 创新点**

核心创新在于利用日志似然函数的二阶导数（曲率）下界，构造稀疏拉普拉斯平滑的输入扰动，从而在不需要逐步计数的情况下实现纯DP，并规避传统迭代机制的三次根误差瓶颈；

**🔧 技术方法**

技术手段包括：选择-仅乘法权重更新（selection‑only MWU）、软最大化（softmax）查询选择、拉普拉斯卷积平滑、二阶导数曲率分析以及基于马尔可夫不等式的高概率误差分析；

**📊 数据集**

论文属于理论分析范畴，未使用具体实验数据集，而是在通用数据宇宙 [d] 上给出统计查询工作负载的泛化误差与时间复杂度证明；

**📈 对比分析**

与现有的MWEM、硬化机制等方法相比，本文在纯DP下实现了误差与查询量的平方根比例，时间复杂度为O(dk T)（T≈N/log k），显著优于之前的O((log d log k/N)^{1/3})误差或需要指数时间的存在性证明；

**⚠️ 局限性**

限制主要包括：对近似DP的优势仍然存在（如δ<1/N时纯DP仍优），并且机制在极大数据宇宙或查询量极大时对日志曲率假设的依赖需要进一步验证；

---

## 502. Reducing Logprob Without Causal Verification: Diagnosing SAE-Based PII Suppression in LLMs

**arXiv ID:** 2611.04782 | [PDF](https://arxiv.org/pdf/2611.04782v1)

**作者:** Chaerin Lee `[一作]` (Soongsil University), Daeseon Choi `[通讯]` (Soongsil University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对LLM中的Sparse Autoencoder（SAE）特征干预进行评估，并提出多指标诊断框架验证特征抑制是否具备因果特异性；同时开展了可行性研究验证因果中介分析可用于识别真实的PII泄漏路径。

**💡 创新点**

发现单一log‑prob削减评估无法区分特征抑制与大幅度激活扰动；提出结合随机方向对照、排名指标和选择性阈值的多指标评估；首次将因果中介分析与LLM PII抑制结合，验证其有效性。

**🔧 技术方法**

使用SAE构建特征空间、隐藏空间干预、随机方向对照、基于排名的提取成本代理、以及因果中介分析（自然间接效应）来评估特征抑制；并在三大8B/9B LLM上进行实验。

**📊 数据集**

三大合成医学文本数据集：包含姓名、患者ID、手机号和邮箱的合成临床笔记，保证训练、发现和评估样本互不重叠。

**📈 对比分析**

与传统仅使用log‑prob下降的评估相比，新的多指标框架在36个配置中仅有1个（Llama L16/name）满足全部四项指标；在因果可行性研究中，因果中介分析能准确区分存在记忆的情况（如邮箱）与否，表明方法具备有效性。

**⚠️ 局限性**

局限性包括：仅测试8B/9B通用模型，未验证更大规模或领域专用模型；仅评估单特征干预，未探索多特征或电路级干预；使用合成数据，可能缺乏真实医学记录的多样性；实验层次选择有限，未进行完整层级搜索；特征选择依赖于log‑prob或激活率，可能未覆盖所有因果路径；评估指标不等同于完整的提取攻击安全性。

---

## 503. Replica-Aware Split Inference for Edge IoT Systems

**arXiv ID:** 2611.04779 | [PDF](https://arxiv.org/pdf/2611.04779v1)

**作者:** Zied Jenhani `[一作]` (Technische Universit"at Braunschweig), Admela Jukan `[通讯]` (Technische Universit"at Braunschweig)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于复制的拆分推理恢复方法，在设备失效后通过差分OTA在存活设备上部署模型分段复制，以保持原始分区布局并恢复推理流程。

**💡 创新点**

创新点在于将复制主机选择与通信协议协同建模为混合整数线性规划（MILP），同时满足闪存与运行时内存约束，目标是最小化失效后端到端推理延迟；且采用差分OTA实现按需复制。

**🔧 技术方法**

使用混合整数线性规划、线性化技术、分层激活传输模型、分布式IoT网络模拟、Wi‑Fi/BLE/ESP‑NOW通信模型。

**📊 数据集**

使用ImageNet预训练的MobileNetV2（α=0.35，INT8量化），输入尺寸224×224×3。

**📈 对比分析**

与随机、计算贪婪、SRAM Worst‑Fit、Flash First‑Fit Decreasing四个基线进行比较。MILP平均端到端延迟比计算贪婪低6.8%，比随机低18.3%；在15机节点、75%链路速率下降时差距更大。计算时间随失效数量和设备规模增长。

**⚠️ 局限性**

局限性包括MILP求解时间随失败数和设备规模显著上升；在内存约束下部分失效组合不可恢复；未考虑OTA更新耗时与能耗；仅针对固定分区方案，未动态重划分。

---

## 504. Learning to Rank Explanations for Recommendation with Language Models

**arXiv ID:** 2611.04775 | [PDF](https://arxiv.org/pdf/2611.04775v1)

**作者:** Ben Kabongo `[一作]` (Sorbonne University), Vincent Guigue `[通讯]` (AgroParisTech)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出LMStaR框架，将个性化解释排序视为密集检索，通过双编码器分别编码用户历史+商品信息与候选解释句子，并使用多正样本对比学习进行任务适配。

**💡 创新点**

创新点在于完全文本化的解释排序方法，联合训练查询与陈述编码器，利用语言模型适配实现更强个性化和可解释性；与CLER对比表现更优；同时跨域迁移能力强。

**🔧 技术方法**

技术包括大型语言模型（如Qwen3-Embedding、EmbeddingGemma等）作为编码器，LoRA参数高效微调，多正样本InfoNCE对比学习，双编码器dense retrieval。

**📊 数据集**

数据集：Amazon Beauty、Amazon Toys、Yelp（基于用户评论提取的句子级别数据集）。

**📈 对比分析**

与流行基准（popularity、neighborhood、factorization、BPER系列、CLER、零射击检索）对比，LMStaR在NDCG@1上所有三域均最高，Beauty、Toys top‑5显著优于CLER，Yelp top‑5略逊于CLER；跨域迁移亦优于零射击和传统基准。

**⚠️ 局限性**

局限包括：仅在item‑level评估；候选句子来自已知评论，未考虑实时可用性；二元相关性基于自动提取，可能忽略未见相关句子；对解释质量的主观感知未测评；模型依赖大型预训练模型，计算成本高。

---

## 505. TactiNavi: Serving and Revising an LLM Game Coach with Explicit Strategy Conditions

**arXiv ID:** 2611.04772 | [PDF](https://arxiv.org/pdf/2611.04772v1)

**作者:** Qi Liu `[一作]` (Tencent), Zixun Sun `[通讯]` (Tencent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个能够根据玩家状态和游戏版本动态决定是否适用的LLM教练服务系统，采用离线编译、在线条件评估、决策追踪和基于条件的版本控制；

**💡 创新点**

关键创新在于将策略适用性显式化为可编译的条件对象，结合三值逻辑评估、可追踪的决策轨迹以及基于依赖闭包的安全发布门；

**🔧 技术方法**

技术上包括离线策略编译器、结构化玩家状态接口、三值条件评估、决策追踪与记录、诊断与修复评估器、以及演化控制平面；

**📊 数据集**

使用了某商业开放世界游戏的1,122条玩家教练请求、玩家状态快照、工具输出及历史记录作为评估数据；

**📈 对比分析**

与原始事实、未显式条件的指南以及剔除条件规则的基线进行对比，结果显示正确率从57.9%提升至82%，误用率从18.1%降至10.4%，追踪诊断精度达86.4%，平均回复时延约2.1秒；

**⚠️ 局限性**

局限性包括仅在单一游戏和有限的两轮修订中验证，未覆盖更复杂的游戏机制、长周期修订和非回复阶段组件的评估，以及对异常情况的适用性验证不足。

---

## 506. Beyond Local Refinement: Wide-Area Precise Cross-View Localization

**arXiv ID:** 2611.04759 | [PDF](https://arxiv.org/pdf/2611.04759v1)

**作者:** Qingguo Hu `[一作]` (Xiamen University), Jinsong Su `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于多假设分支放大搜索和深度环验证的跨视角精细定位方法；

**💡 创新点**

创新点在于利用预测不确定性展开多路径搜索并引入视角共享的深度环作为空间指纹，配合大规模视觉‑语言模型进行候选验证；

**🔧 技术方法**

使用了多假设分支搜索、聚类、预测不确定性引导、深度环生成（基于估计深度）、LVLM 验证器以及跨尺度联合训练；

**📊 数据集**

数据集主要是新构建的 VIGOR‑Pro（由 VIGOR 生成的 100/200/400 m 大区域），并在伦敦、东京、圣保罗、开普敦等城市进行外部评估；

**📈 对比分析**

与多种端到端与检索式基线对比，RingMatch 在 400 m 区域下平均误差仅 12.36 m（同区），19.81 m（跨区），明显优于现有检索+局部细化方法；

**⚠️ 局限性**

局限性包括需要已知朝向、对深度估计和 LVLM 训练依赖较大、搜索和验证过程算力消耗较高，且在极大尺度或缺失姿态信息时性能可能下降。

---

## 507. TEXMET: A Deep Learning Benchmark for Archaeological Textile Reconstruction

**arXiv ID:** 2611.04734 | [PDF](https://arxiv.org/pdf/2611.04734v1)

**作者:** Muhammad Hamza Zafar `[一作]` (Norwegian University of Science and Technology), Jon Yngve Hardeberg `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建并发布了TEXMET数据集，并系统评估了三种先进的拼图求解模型（JPDVT、FCViT、DiffAssemble）在纺织品图像上的表现，同时对奥塞伯格船葬的高度退化纺织碎片进行领域微调实验。

**💡 创新点**

首次构建大规模、开源的纺织品专用拼图基准；对现有模型进行系统对比，证明领域适配对纺织品拼图解法至关重要；展示Graph‑Diffusion模型DiffAssemble在纺织品上的显著性能提升。

**🔧 技术方法**

利用Transformer架构（JPDVT、FCViT）和图神经网络+扩散（DiffAssemble）；使用CLIP+UMAP+FiftyOne进行数据筛选与聚类；进行Fine‑tuning、Inpainting和Inscribed裁剪等实验；全部实现基于PyTorch与Hugging Face框架。

**📊 数据集**

TEXMET（18,644来自Metropolitan Museum of Art的历史纺织图像）作为训练和测试集；45个奥塞伯格船葬纺织碎片作为真实考古材料评估；对比ImageNet 1k作为基线。

**📈 对比分析**

通过拼图准确率和片段准确率进行比较；在ImageNet上各模型基线性能相近；在TEXMET上DiffAssemble Fine‑tune后达到85.89%拼图准确率，JPDVT和FCViT仅提升到约32%和8%；在奥塞伯格碎片上，Fine‑tune后DiffAssemble从26.67%跃升至64.44%，显著优于其他模型。

**⚠️ 局限性**

评估仅基于规则的正方形网格拼图，未处理不规则碎片形状和缺失区域；奥塞伯格评估仅在单片碎片内部重组，未实现跨碎片拼接；数据仍为人工切片，缺乏真实碎片的几何与损伤特征；模型对极端退化或高度不规则碎片的鲁棒性有限。

---

## 508. BrainwashedLLM: Jailbreaking via Automated Adversarial Chain of Thoughts Discovery

**arXiv ID:** 2611.04717 | [PDF](https://arxiv.org/pdf/2611.04717v1)

**作者:** Masoud Poorghaffar Aghdam `[一作]` (Bilkent University), Sinem Sav `[通讯]` (Bilkent University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型推理模型（LRM）的Chain-of-Thought（CoT）生成过程进行攻击，提出一种在无目标模型反馈的情况下通过强化学习生成攻击性思考轨迹，并将其注入更大模型进行零射击攻击。

**💡 创新点**

①首次使用GRPO在源模型上直接优化CoT攻击，完全不需要目标模型交互；②证明CoT作为攻击向量能够实现弱到强的零-shot迁移；③提出多目标规则奖励函数，抑制拒绝与安全建议。

**🔧 技术方法**

Group Relative Policy Optimization (GRPO)、多目标规则奖励（拒绝奖励、safe‑advice 奖励、长度奖励）、白盒微调、攻击注入。

**📊 数据集**

50条随机挑选的有害行为提示（Harmful Behaviors）以及JailbreakBench。

**📈 对比分析**

与Mousetrap、H‑CoT等强攻击对比；在四大LRM上评估ASR@5/ASR@1；结果显示无目标查询的BrainwashedLLM在四个模型上ASR@5介于0.74-0.98，距基线仅差≤16%。

**⚠️ 局限性**

仅适用于可观测CoT的模型；实验覆盖有限的LRM；未考虑隐蔽CoT设计；攻击成功率受提示多样性与目标模型对CoT实现方式影响。

---

## 509. Hybrid acoustic and pretrained embedding feature fusion for unmanned aerial vehicle detection using stacking ensemble learning under a leakage-aware evaluation protocol

**arXiv ID:** 2611.04714 | [PDF](https://arxiv.org/pdf/2611.04714v1)

**作者:** Zahid Iqbal `[一作]` `[通讯]` (University of Gujrat), Zahid Iqbal (University of Gujrat)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

针对无人机声学检测，提出并评估了基于手工特征与预训练YAMNet嵌入的两视图堆叠模型。

**💡 创新点**

创新点在于：①采用源离散的泄漏意识评估协议；②将mRMR与ReliefF混合的过滤式特征选择；③结合NSGA-II多目标搜索得到紧凑特征；④在同一数据集上系统比较不同特征与融合策略。

**🔧 技术方法**

技术包括：MFCC/LogMel等音频特征提取、mRMR/ReliefF特征筛选、NSGA-II多目标搜索、随机森林、XGBoost、单层CNN、YAMNet预训练嵌入、逻辑回归元学习器、McNemar检验、Bootstrap置信区间。

**📊 数据集**

使用DroneAudioDataset二分类任务（1332个无人机片段+10372个非无人机片段），并通过ESC‑50等来源构建负样本。

**📈 对比分析**

通过对比随机切分与来源分组交叉验证，发现随机切分高估1.89–3.22个百分点；在分组协议下，两视图堆叠模型达95.29%准确率、98.29%特异率，性能优于单视图和单模型。

**⚠️ 局限性**

局限包括：①数据集仅覆盖两款室内四旋翼，外部环境及不同无人机类型缺失；②预训练YAMNet并非针对无人机，提升有限；③模型对机械噪声易误检；④未在真实现场环境验证；⑤对更高压缩需求的低功耗设备仍需进一步优化。

---

## 510. Optimal Approximation for Constrained Correlation Clustering via Repulsion

**arXiv ID:** 2611.04784 | [PDF](https://arxiv.org/pdf/2611.04784v1)

**作者:** Yixin Cao `[一作]` (Hong Kong Polytechnic University), Ying Xu `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种随机化多项式时间的 (2+ε)-逼近算法，用于在存在“必须连通”与“禁止连通”约束的相关聚类问题。

**💡 创新点**

核心创新包括：① 设计了“Repulsion”递归分割操作，自然满足硬约束并通过潜在函数得到 3 倍上界；② 证明了必须连通约束不导致额外的逼近损失，可先通过真‑twin 替换消除；③ 构建了无须求解全局聚类 LP 的“有界子聚类 LP”并结合正条件采样与 Repulsion 完成舍入；④ 给出了从顶点覆盖到聚类删除的逼近保持化简化，并在此基础上证明了约束相关聚类在 UGC 下的 2‑逼近硬性界。

**🔧 技术方法**

技术手段包括：随机化多项式时间算法、两阶段策略（先消除必须连通，再结合无硬约束的预聚类和子聚类 LP）、正条件采样、Repulsion 递归分割、概率加权实例的无权化扩展、硬约束的潜在函数分析，以及 UGC 基础的硬性证明。

**📊 数据集**

该工作为纯理论分析，不使用实验数据集；所有结果均在理论复杂度与逼近比率上给出。

**📈 对比分析**

与之前最优的 16/7 逼近相比，本文实现了近似比率可任意逼近 2，且在 UGC 假设下已达到匹配的下界；此外，对于仅有必须连通约束的情形，给出了 1.3865+ε 的逼近，突破了 √2 的阈值。

**⚠️ 局限性**

主要局限：时间复杂度为 (n/ε)^{O(ε^{-2})}，对大规模实例实用性有限；逼近下界依赖于 UGC 假设；算法仍为理论证明，缺乏实验验证与对实际数据集的评估。

---

## 511. Cooperative Multi-Agent Reinforcement Learning with Submodular Trajectory Rewards

**arXiv ID:** 2611.04774 | [PDF](https://arxiv.org/pdf/2611.04774v1)

**作者:** Yasaman Zolfimoselo `[一作]` (EPFL), Maryam Kamgarpour `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了合作多智能体强化学习，其中奖励函数为单调子模的轨迹奖励，提出了基于预窗口曲率的贪心窗口化策略优化（GWPO）算法，并在多无人机交通监控场景中进行实验。

**💡 创新点**

创新点主要有：①引入预窗口曲率（pre‑window curvature）量化历史对当前增益的影响，给出与曲率相关的逼近保证；②利用分阶段、顺序优化的GWPO框架，避免了全局状态空间指数级扩张；③在同一框架下兼顾子模奖励与解耦动力学，并提供了理论与实验双重验证。

**🔧 技术方法**

使用的技术包括：子模函数与曲率理论、马尔科夫决策过程（MDP）、贪心算法、序列化策略优化、采样估计奖励、时间窗口化的期望目标等。

**📊 数据集**

实验数据集为 20×20 的城市道路网模拟，时间步长 H=20，初始无人机位置随机采样，使用 10,000 条公共初始状态进行评估。

**📈 对比分析**

与基线方法 Greedy Policy Optimization（GPO）以及 Submodular Policy Optimization（SubPO，Markov 与非Markov 版本）进行对比。GWPO 在非加性奖励下相较 GPO 获得更高回报；在计算时间上随着记忆长度增加提升，但仍保持多项式复杂度；在规模较大时（N≥10）GWPO 甚至优于 SubPO。

**⚠️ 局限性**

局限性包括：需要已知转移概率与精确评估奖励；预窗口曲率需小于 1，若为 1 则无逼近保证；奖励估计需采样，样本量影响精度；算法主要针对离散状态空间，连续或高维情形仍需进一步研究。

---

## 512. When Should an LLM Steering Edit Depend on the Input?

**arXiv ID:** 2611.04757 | [PDF](https://arxiv.org/pdf/2611.04757v1)

**作者:** Rachael Harkness `[一作]` (University of Edinburgh), Konrad Rawlik `[通讯]` (University of Edinburgh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了激活层的输入依赖式编辑方法，提出一种根据查询动态检索最近邻目标状态的steering规则。

**💡 创新点**

创新点在于将固定方向steering推广为输入可变的目标检索，提供k-最近邻与类均值之间的连续谱，并给出四个诊断量预估何时使用该方法。

**🔧 技术方法**

使用的技术包括差分平均激活加法、基于最近邻检索、Affine映射对比，以及四种诊断指标（D、g、R、1-A）。

**📊 数据集**

使用的数据集包括六个替换语料（词级和句子级）、两个token语料、两类A/B行为基准、外部CounterFact基准，以及几种生成的二维几何构造。

**📈 对比分析**

与传统的固定方向steering、类均值steering以及拟合Affine映射对比，检索规则在所有六个替换语料和token语料上显著降低参考误差，并在大多数层级取得更高的per‑item identity；在A/B基准中表现差异，外部CounterFact基准则提升到0.88的有效率。

**⚠️ 局限性**

限制在于需要存储和检索大量隐藏状态，计算开销略增；当目标分布在类内几乎不变或检索库覆盖不足时，检索方法退化为类均值steering，无法获得优势。

---

## 513. Optimal (Randomized) Compact Routing Schemes

**arXiv ID:** 2611.04744 | [PDF](https://arxiv.org/pdf/2611.04744v1)

**作者:** Shiri Chechik `[一作]` (Tel Aviv University), Liam Roditty `[通讯]` (Bar Ilan University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了四种无向图紧凑路由方案（标记式/名称无关，平均/最坏存储），并显著降低了收敛率与存储之间的折衷，取得了多项最优或近最优的伸缩系数。

**💡 创新点**

创新点在于引入了仅在路由阶段随机化、对所有层级一次性优化以及在树中实现更轻量级搜索的新技术，成功突破了之前的 2.64k、3.68k、96k 等上限。

**🔧 技术方法**

主要技术包括 Thorup‑Zwick 层次结构、期望伸缩分析、线性规划求解最优随机层级分布、以及基于预排序块的随机树搜索算法。

**📊 数据集**

本文属于理论计算机科学，未使用任何实验数据集，而是通过证明和构造来评估路由方案的性能。

**📈 对比分析**

与之前最优方案相比，平均存储标记式方案的期望伸缩从 2.64k 降至 2k‑1，最坏存储标记式方案的期望伸缩从 3.68k 降至 8/3k+6，名称无关方案在平均/最坏存储上分别从 96k 降至 4k‑5、从 8k‑1.56 降至 8k‑1.56 等，性能提升显著。

**⚠️ 局限性**

局限性包括：标记式方案的确定性伸缩仍高于 2k‑1；最坏存储方案仍未达到理论下界；名称无关方案的伸缩与下界相距较大，后续研究需进一步突破。

---

## 514. Smallest Multiplicative Stretch for an $α$-APASP Below $\widetilde O \left(n^ω\right)$ Runtime

**arXiv ID:** 2611.04736 | [PDF](https://arxiv.org/pdf/2611.04736v1)

**作者:** Lakshay Saggi `[一作]`, Ariel Sapir `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种随机化的 2‑近似 APSP（All‑Pairs Shortest Paths）算法，能够在几乎线性的时间内为任意加权图计算所有点对之间的 2‑近似最短路径距离。

**💡 创新点**

创新点在于将近似最小加法（1+ε‑approximate min‑plus）矩阵乘法与基于随机 pivot 采样、距离分桶、偏移‑移位（offset‑shifted）技术相结合，从而在保持 2‑近似保证的同时显著降低运行时复杂度，并利用全矩形矩阵乘法指数优化实现更低的理论上限。

**🔧 技术方法**

主要技术包括：1+ε‑近似 MSSP、pivot 采样与距离分桶、offset‑shifted 近似 min‑plus 矩阵乘法、矩阵划分与行/列平衡、随机化高概率保证以及对矩形矩阵乘法指数（ωa,b,c）的深入利用。

**📊 数据集**

实验/评估未在数据集上完成，论文以理论分析为主，使用任意整数权重图（权重范围 ≤ Wmax）作为输入模型。

**📈 对比分析**

相较于之前的近似 APSP 方法，本文实现了 n^2.17（或在旧矩阵乘法指数下为 n^2.24）级别的时间复杂度，显著低于传统的 O(n^3) 或更高的近似时间；通过随机化和矩阵乘法加速，算法在理论上已经逼近最优。

**⚠️ 局限性**

局限性包括：1) 需要使用高速矩阵乘法库，实际常数和实现难度较高；2) 仅在随机高概率下成立；3) 仅支持整数权重（或有限权重范围）；4) 近似因子固定为 2，无法进一步降低；5) 对非常稀疏图的优势相对有限。

---

## 515. OneTrans-GSR: A Generative Slate Ranking Paradigm Unifying Reranking and Mix-Ranking with One Autoregressive Transformer

**arXiv ID:** 2611.04720 | [PDF](https://arxiv.org/pdf/2611.04720v1)

**作者:** Feng Cheng `[一作]` (ByteDance), Shaowei Liu `[通讯]` (ByteDance)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种名为OneTrans‑GSR的单一自回归解码器，直接从请求特定候选池生成符合多业务约束的推荐排版；

**💡 创新点**

将排版视为学习问题，利用斜面级奖励模型和Geometric‑Mean Reward Policy Optimization，结合硬约束与软偏好，实现端到端可扩展的自回归生成；

**🔧 技术方法**

采用OneTrans自回归解码器、斜面级多目标奖励模型（包含LTV）、GMR‑PO策略优化、停止梯度、KV缓存、并行生成、硬约束/有限logit调整以及集成评分等技术；

**📊 数据集**

使用电商生产日志（约三个月）作为数据集，包含候选池、展示斜面以及点击、订单、GMV、转化等结果标签；

**📈 对比分析**

与传统Generator‑Evaluator基线在离线RM估计相对收益、AUC/PCOC等指标以及在线A/B测试比较，离线相对收益分别为ListClick≈0.077、ListOrder≈0.313、ListGMV≈0.287，在线GMV/u提升2.63%、广告价值提升2.06%、复购率提升2.03%，推迟延迟降低13%；

**⚠️ 局限性**

受限于奖励模型预测误差导致离线与在线收益不完全一致；软偏好权重需手工调节；仅在特定电商场景验证，未拆解单个组件贡献；对极端候选空间或复杂业务约束的鲁棒性尚未充分评估。

---

## 516. Methodological Assessment of Instrumented Indentation for Spatially Resolved Mesoscale Stiffness and Creep in Short Fiber-Reinforced Injection-Molded PBT Components

**arXiv ID:** 2611.04707 | [PDF](https://arxiv.org/pdf/2611.04707v1)

**作者:** Eduard Klatt `[一作]` (Helmut-Schmidt-University / University of the Federal Armed Forces Hamburg), Natalie Rauter `[通讯]` (Helmut-Schmidt-University / University of the Federal Armed Forces Hamburg)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2`

**🎯 论文内容**

本文通过系统评估表面处理、压痕间距、最大负载、加载时间、保持时间与卸载时间等实验参数，制定统一的压痕测试方案，并在短纤维增强注塑聚对苯二甲酸丁二酯（PBT）GF0和GF30两种材料上实现了空间分辨率的局部模量与蠕变测量。

**💡 创新点**

创新点在于首次将表面准备、压痕间距和时间参数的系统化评估与空间压痕实验相结合，为短纤维增强聚合物的微观力学特性提供了可重复、可比较的实验框架，并量化了纤维增强对局部弹性与蠕变的影响。

**🔧 技术方法**

采用Berkovich针仪器化压痕技术，结合Oliver–Pharr方法提取接触刚度与模量，使用长时间保持期（300 s）与高负载（1750 mN）获取蠕变深度，并辅以键能共聚焦显微镜评估表面几何与残留形貌。

**📊 数据集**

使用的实验数据集包括160个压痕点（GF0）和159个有效压痕点（GF30），记录最大负载、接触深度、压痕模量、绝对与相对蠕变深度等指标。

**📈 对比分析**

通过均值、标准差和变异系数的统计描述对比两材料的局部模量和蠕变，结果显示GF30的平均模量提升约129%，平均蠕变深度降低约19%，但空间变异显著增加；该方法在高负载、长保持期条件下表现稳定可靠。

**⚠️ 局限性**

主要局限包括未对点位pile‑up进行校正，导致得到的模量为表观值；压痕结果受表面粗糙度、加载速率和蠕变恢复的影响；并且仅提供局部统计信息，未实现完整的空间随机场建模。

---

## 517. Proving the Wrong Program: Miscompilations as a Threat to zkVM Guarantees

**arXiv ID:** 2611.04701 | [PDF](https://arxiv.org/pdf/2611.04701v1)

**作者:** Hui Xuan Tan `[一作]` (Imperial College London), Alastair F. Donaldson `[通讯]` (Imperial College London)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对零知识虚拟机（zkVM）中编译器误编译所带来的安全威胁进行了系统的实证研究，评估了两款主流zkVM（SP1和Jolt）以及13个公开项目的客程序。

**💡 创新点**

创新点在于首次量化误编译对zkVM证明正确性的影响，并展示攻击者可通过看似正常的优化补丁诱发已知误编译，从而生成合法但错误的证明。

**🔧 技术方法**

采用LLM代理将LLVM误编译翻译为Rust触发程序，构造仅差异为修复的编译器对，执行zkVM的guest程序并对比输出；同时使用LLM自动搜索能触发误编译的“无害”补丁。

**📊 数据集**

数据集包括59个rustc误编译与120个LLVM误编译（均附有Rust重现代码）以及13个流行的SP1客程序源代码。

**📈 对比分析**

通过在SP1与Jolt上分别运行buggy与fixed编译器构建的二进制，比较二进制是否一致、输出是否不同，发现约55%的误编译在zkVM上可复现；在366个项目–误编译对的测试中，LLM成功生成12个可触发误编译的补丁，成本约每个项目40美元。

**⚠️ 局限性**

局限性包括仅关注RISC‑V指令集的zkVM、仅收集有Rust重现器的误编译、仅测试单一项目修订版、未生成证明、输入覆盖有限导致可能漏检差异、以及对LLM搜索空间和准确性的依赖。

---

## 518. Logarithmic Comparison-Query Complexity for EF1 Allocations

**arXiv ID:** 2611.04753 | [PDF](https://arxiv.org/pdf/2611.04753v1)

**作者:** Zehan Lin `[一作]` (University of Macau), Shengwei Zhou `[通讯]` (Nanyang Technological University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究在只能进行比较查询的模型下，如何为n个具异质加性偏好的代理人用O(log m)次查询得到“相邻物品可容忍的公平分配”(EF1)；

**💡 创新点**

创新点在于构造了可平衡的分区并利用连续插值与拓扑匹配定理，首次证明对任意固定n≥2均可实现O(log m)的查询上界；

**🔧 技术方法**

核心技术包括：先去除“重物”，构造轻物的n−1块平衡分区；对前缀长度进行连续插值；使用Meunier‑Su匹配定理与根单调性；以及通过分组与候选淘汰实现高效搜索；

**📊 数据集**

本工作为理论性研究，无实验数据集；所有结果均通过数学证明给出；

**📈 对比分析**

与此前仅针对n≤3给出的O(log m)算法相比，本文的O(n⁸ log m)算法在固定n时仍保持O(log m)的渐近性能，证明了最优下界Ω(log m)；

**⚠️ 局限性**

局限性包括：对n的多项式因子较高（n⁸），算法实现复杂；仅适用于加性偏好，尚未扩展至子模或其他公平准则；

---

## 519. Who Is Doing the Thinking? Characterising Human Agency in First-Year Students' AI Interactions

**arXiv ID:** 2611.04783 | [PDF](https://arxiv.org/pdf/2611.04783v1)

**作者:** Ali Shakiba `[一作]` (University of Sydney), Thomas Chaffey `[通讯]` (University of Sydney)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了Human Agency Depth Taxonomy（HADT），通过对学生在AI日记中记录的提示进行编码，量化学生在使用生成式AI时对工程任务的主动贡献。

**💡 创新点**

创新点在于首创一种以“情境宽度、工程抽象、学生代理”三维度为基础的六层级分类框架，能够独立评估学生的工程思维，而非仅关注AI输出的认知层级。

**🔧 技术方法**

技术主要包括手工提示编码、线性加权Cohen’s κ评估、频数分析、Pearson χ²检验、Spearman相关、以及使用广义估计方程（GEE）和线性混合效应模型来处理学生内嵌层次结构。

**📊 数据集**

使用的数据集为一门悉尼大学第一年软件工程课程中105名学生共计962条AI日记提示，另外为长期提交日记的27名学生提取了608条提示以进行纵向分析。

**📈 对比分析**

通过对不同SDLC阶段和截止周的频数与比例进行比较，发现大多数提示（84.2%）体现学生贡献，设计空间探索在需求阶段占主导，诊断与组件生成在构建与验证阶段增多；GEE模型进一步确认了阶段对代理层次的显著影响，尽管未提供传统算法性能指标。

**⚠️ 局限性**

局限性包括自报日记可能产生选择性报告与偏倚、只涉及低代码环境的单一课程、学生对提示技术与领域知识的差异导致的约束阐述不足，以及未能验证HADT在不同学科与经验水平下的普适性。

---

## 520. Pixel-space video generation with U-JiT

**arXiv ID:** 2611.04764 | [PDF](https://arxiv.org/pdf/2611.04764v1)

**作者:** Théophane Vallaeys `[一作]` (FAIR, Meta), Matthieu Cord `[通讯]` (Sorbonne University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种多分辨率像素空间视频生成模型，通过在Transformer深度上变换补丁大小，实现端到端训练，克服传统像素空间模型在高维视频数据中的频率捕获不足。

**💡 创新点**

创新点在于将U形Transformer与可变补丁大小相结合，加入注意力裁剪、补丁化注意力、时间平移和无分类器引导等技巧，显著提升高频细节生成，同时保持高效可扩展的计算成本。

**🔧 技术方法**

使用Flow-Matching训练目标、DiT/U-DiT结构、SwiGLU激活、RMSNorm、RoPE、qk-norm、注意力裁剪与补丁化注意力、时间平移、无分类器引导、LPIPS感知损失等技术。

**📊 数据集**

在Kinetics-600视频数据集上进行实验，采用帧到视频和类别到视频两类生成任务。

**📈 对比分析**

通过与现有像素空间和潜在空间基线在FVD和FID指标上的对比，模型在帧到视频任务中将FVD从0.84降至0.61（27%提升），FID降幅达90%，在类别到视频任务中FVD提升9%，并在强压缩条件下保持竞争力。

**⚠️ 局限性**

局限性包括对时间平移、引导强度等超参数的敏感性，仍需更大规模或更高分辨率的验证，以及在极端压缩或多模态条件下的泛化性能待进一步探索。

---

## 521. iMED: A Multi-Endoscope Dataset for Surgical 3D Perception

**arXiv ID:** 2611.04755 | [PDF](https://arxiv.org/pdf/2611.04755v1)

**作者:** Sierra Bonilla `[一作]` (University College London), Omid Mohareri `[通讯]` (Intuitive)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了 iMED——一个同步双视角外科内镜数据集，并在其上定义了“单端镜训练、另一端镜测试”的评估协议，针对机器人辅助手术中的 3D 感知任务进行了基准测试。

**💡 创新点**

创新点在于首次提供多端镜同步采集的外科数据，打破单轨视角限制；提出了基于独立视角的几何泛化评估；发现大规模通用模型往往在外科场景中表现优于专门训练的模型。

**🔧 技术方法**

采用了深度学习中的 NeRF、3DGS、4DGS、DynOmo 等新视角合成框架；利用 ArUco 标定实现精确相机姿态；使用基础模型（LoFTR、RoMa、GIM 等）进行特征匹配与姿态估计；以及 UniDepth‑V2、MoGe‑V2 等单目深度推断网络。

**📊 数据集**

使用了 iMED 数据集（340 条序列、≈170K 时点、4 视角），涵盖 ex vivo、postmortem 及 live 外科场景，包含多种解剖部位、工具与运动模式的标注。

**📈 对比分析**

对 23 种最先进方法在 NVS、姿态估计、特征匹配和单目深度上进行比较，结果显示依赖光度监督的模型在视角迁移时性能显著下降；加入几何约束或使用通用基础模型的算法在不同端镜上保持更高的 PSNR/SSIM、旋转误差和深度误差。

**⚠️ 局限性**

局限包括：视角多样性受手术通道与灯光限制；外科序列主要为手动摄像头运动，运动平滑度不足；样本与解剖种类相对有限，未覆盖所有临床环境。

---

## 522. SPOD: Recovering Suppressed Temporal Evidence in Video-LLMs via Single-Pass Orthogonal Decomposition

**arXiv ID:** 2611.04728 | [PDF](https://arxiv.org/pdf/2611.04728v1)

**作者:** Yangguang Lin `[一作]` (Beijing University of Posts and Telecommunications), Zefan Xu `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关的单通道正交分解框架（SPOD），通过对视频-LLM隐藏状态进行分析和重构，恢复并强化时间信息以缓解视频推理中的时间幻觉。

**💡 创新点**

创新点在于：①发现视频-LLM隐藏状态中已存在但被抑制的时间证据；②设计基于隐藏状态几何的帧区分度分析（FDA）和基于帧差分的正交子空间构造（TDS）；③在单次前向推理中完成子空间投影与分量独立缩放，实现无需额外训练、无多次推理或外部监督的时间推理提升。

**🔧 技术方法**

使用技术包括：中层隐藏状态的余弦相似度方差评估、软最大化帧权重、交叉注意力融合、权重化帧差分构造、截断SVD提取时间与先验子空间、正交投影与分量缩放。

**📊 数据集**

实验数据集涵盖 VidHalluc、EventHallusion、VideoMMMU 三大基准，以及在 Qwen2.5-VL-72B 上的进一步验证。

**📈 对比分析**

与传统的 vanilla、对比解码（TCD）、外部模型辅助（CGD、VTI）、辅助训练（DINO-HEAL、TA-AE）等基线进行比较。SPOD 在所有模型与任务上均实现显著提升，VidHalluc 总分提升约10分；EventHallusion 误导子任务提升至 86% 以上；VideoMMMU 总分提升至 54.33，且推理吞吐量仅略低于 vanilla。

**⚠️ 局限性**

局限性：对超大模型的提升幅度相对有限；依赖于中层隐藏状态的区分度与差分子空间的低秩特性，若模型架构或训练方式改变可能需要重新调参；对非时间相关的视觉推理任务提升不明显。

---

## 523. ZeroGR: Filtering-First Constrained Decoding for Generative Recommendation

**arXiv ID:** 2611.04715 | [PDF](https://arxiv.org/pdf/2611.04715v1)

**作者:** Xiangrui Yang `[一作]` (University of Hong Kong), Yiming Qiu `[通讯]` (University of Hong Kong)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 ZeroGR，一种先过滤后评分的生成式推荐约束解码系统，显著减少无效序列的计算，保持与传统评分先过滤方案相同的推荐质量。

**💡 创新点**

创新点包括：① 步骤切片的过滤索引，按每步有效候选密度动态采用稀疏或稠密表示；② 共享槽位执行和混合投影策略，将过滤结果直接传递给分数计算并在 GPU 上高效并行；③ 布局引导的流式索引构建，能够在更新窗口内快速重建索引而不干扰在线推理。

**🔧 技术方法**

使用技术包括 GPU 内存高效索引布局、共享槽位 (shared‑slot) 执行模型、选择性（Selective）与完整（Full）投影混合、Top‑k 采样和收集融合、流式 GPU 索引构建及多线程 CPU 预处理。

**📊 数据集**

在两个真实工业数据集（Search 与 Watch）和一个 Synthetic‑7 生成数据集上进行评估，分别包含数亿到十亿级的 SID 序列。

**📈 对比分析**

与四个基线（Unconstrained、CPU trie、FlashTrie、STATIC）对比，ZeroGR 在 Search 数据集上 FP32/FP16 分别提升 9.1×/10.7×，在 Watch 数据集上 2.0–3.2×，并在内存占用上比 STATIC 小 59–74%；索引重建速度提升 278×。

**⚠️ 局限性**

局限性包括：① 对极度稠密约束的场景（如 Watch 的前两步）仍需全投影，导致与 Unconstrained 的差距缩小；② 需要针对每个模型版本预热并固定投影/Top‑k 组合，适配不同 GPU、批量和精度的过程相对繁琐；③ 对高频更新的模型和索引重建窗口仍需严格控制，以避免对在线推理的干扰。

---

## 524. Approximation Algorithms for Multi-Criteria Network Design Problems

**arXiv ID:** 2611.04742 | [PDF](https://arxiv.org/pdf/2611.04742v1)

**作者:** Mehrshad Taziki `[一作]` (ETH Zürich), Ali Vakilian `[通讯]` (Rutgers University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了多标准最优化中的最小-最大问题，提出了一种新的随机多项式时间算法，能够为多标准连通性增强问题（MC-CAP）和多标准k边连通生成子图问题（MC-k-ECSS）提供更好的近似保证。

**💡 创新点**

创新点在于提出了一种O(logℓ)的近似算法，显著改善了现有的近似保证，并且为MC-k-ECSS问题提供了O(log k logℓ)的近似算法，展示了在多标准情况下如何有效地处理连通性问题。

**🔧 技术方法**

使用了随机化算法和多项式时间的LP松弛技术，结合了结构性约简和随机化取样的过程，以实现对多标准问题的有效近似。

**📊 数据集**

论文中使用的主要数据集是图的边和节点，具体的边权重和成本函数是根据多标准优化的需求设定的。

**📈 对比分析**

与现有方法相比，本文的方法在理论上提供了更好的近似保证，尤其是在处理多标准情况下的连通性问题时，性能显著提升。具体的性能分析表明，算法在期望上能够达到O(logℓ)和O(log k logℓ)的近似比。

**⚠️ 局限性**

限制在于对于某些特定的图结构（如星形图），在多标准情况下的近似难度仍然存在，且在某些情况下，无法在多项式时间内达到O(log^1-εℓ)的近似比，除非所有问题都可以在确定性准多项式时间内解决。

---

## 525. CLARA: A Cognitively-Driven Clarification Framework for Grounding Ambiguous Requests

**arXiv ID:** 2611.04708 | [PDF](https://arxiv.org/pdf/2611.04708v1)

**作者:** Dimitrios Bekris `[一作]` (Athena Research Center), Petros Maragos `[通讯]` (National Technical University Of Athens)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出CLARA框架，利用LLM驱动的对话澄清机制将模糊的高层指令转化为结构化的视觉描述，并通过分段分割、迁移、填补和扩散生成最终图像。

**💡 创新点**

创新点包括：①主动而非被动的澄清提问策略；②将澄清结果与图像编辑、生成链条无缝衔接；③通过视觉反馈验证理解是否准确，形成闭环。

**🔧 技术方法**

主要技术：LLAMA 3.2‑based chatbot、Grounded‑SAM（Grounding DINO + SAM）分割、LaMa inpainting、Stable Diffusion refinement、CLIP嵌入评估。

**📊 数据集**

数据集：三张自定义场景图（桌面、餐桌、脏桌），每个场景对应一条抽象指令。

**📈 对比分析**

比较方法：与Gemini 2.0 Flash单步生成和CLARA+Gemini（将CLARA澄清结果输入Gemini）进行对比；量化指标为CLIP嵌入的标准差与余弦相似度；结果显示CLARA的标准差最低、余弦相似度最高，用户满意度亦明显优于Gemini。

**⚠️ 局限性**

局限性：澄清过程仍受LLM推理误差影响，可能产生与意图不符的“一致”输出；未在动态真实环境中验证；对未知对象的生成仍有限。

---

## 526. Glauber Dynamics for Sampling Colorings: Rapid Mixing above $(1+1/\mathrm{e})Δ$ via Hidden Hard-Core Models

**arXiv ID:** 2611.04699 | [PDF](https://arxiv.org/pdf/2611.04699v1)

**作者:** Xiongxin Yang `[一作]` `[通讯]` (University of California, Santa Barbara), Xiongxin Yang (University of California, Santa Barbara)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了在 q≥(1+1/e+γ)Δ 的条件下，任意最大度 Δ 的图的 Glauber 动力学对 q‑着色的谱间隙至少为 κ_γ/n，从而混合时间为 O_γ(n² log q)。

**💡 创新点**

引入 Bochner 型不等式与隐藏硬核模型相结合的单色分解技术，突破了 11/6 阈值，逼近理论最优 1+1/e。

**🔧 技术方法**

利用 Bochner 公式、硬核模型唯一性阈值、谱独立性、分辨率算子以及平均化、压缩和方差分解等方法。

**📊 数据集**

无实验数据，理论证明；仅使用任意最大度 Δ 的图结构。

**📈 对比分析**

相较于之前的 1.7408Δ 或 1.809Δ 阈值，本研究的阈值仅为 1+1/e+γ，且在所有大 Δ 时保持固定的谱间隙 κ_γ/n，混合时间为 O_γ(n² log q)。

**⚠️ 局限性**

仅适用于 q≥(1+1/e+γ)Δ 的情形，对较小的 q 或图的特殊结构未给出结论；常数 κ_γ 与 γ 相关且未优化。

---

## 527. AI for eGovernance: Leveraging Language Technologies For Digital Inclusion

**arXiv ID:** 2611.04820 | [PDF](https://arxiv.org/pdf/2611.04820v1)

**作者:** Cynthia Jayne Amol `[一作]` (Maseno University), Lilian Wanzare `[通讯]` (Maseno University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过六次访谈和一次圆桌讨论，对肯尼亚青年在 eCitizen 及其关联平台（Ajira Digital、iTax、HELB、YEDF）中遇到的语言、可读性和流程复杂性障碍进行定性分析，并以 WCAG 2.1 原则为设计视角，提出一套包含文本转语音、语音转文本、机器翻译与文本摘要等生成式 AI 语言技术的可落地框架；

**💡 创新点**

创新点在于将 AI 语言技术与 WCAG 2.1 可访问性原则相结合，形成面向用户的“可听可说可译可简”四层支撑框架，强调在 eCitizen 上实现多语言、多模态交互的系统级整合；

**🔧 技术方法**

使用的技术包括：文本转语音（TTS）、语音转文本（STT）、神经机器翻译（NMT）与文本摘要（summarisation），以及一个整合多政府数据库的代理式 AI 架构；

**📊 数据集**

使用的语料数据来源于开放的本地语言资源，如 Masakhane Web、Kencorpus（斯瓦希里语、德洛乌语、卢希亚语）等，并计划通过本研究收集的访谈原始文本进行清洗、标注与模型微调；

**📈 对比分析**

目前尚未进行量化比较或性能评估，论文仅提出技术路线与设计建议；未来工作计划在发布的数据集上训练并评估 NMT、摘要、TTS/STT 模型的准确率、延迟及人工复核成本；

**⚠️ 局限性**

局限性包括样本量有限（仅 17 人，非代表性）、研究仅停留在诊断与设计阶段未落地实现、缺乏正式的 WCAG 2.1 合规性评估，以及技术实现与治理落地需要政府与多方协调支持。

---

## 528. Reconstructing Fraudulent Strategies in Online Payments via Interpretable Inverse Reinforcement Learning

**arXiv ID:** 2611.04819 | [PDF](https://arxiv.org/pdf/2611.04819v1)

**作者:** Huilin He `[一作]` (Tongji University), Dawei Cheng `[通讯]` (Tongji University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

利用逆强化学习框架 PaySeqGen 通过观察真实欺诈支付轨迹重构欺诈策略，并生成可用于增强检测器的合成支付序列，同时提供可解释的奖励信号用于分析欺诈行为模式。

**💡 创新点**

创新点：① 结合历史引导采样与探索采样构造有限候选动作，兼顾行为连贯性与多样性；② 学习线性可分解奖励并加入反馈-动作交互项，显式捕捉上下文依赖的偏好；③ 通过奖励与策略交替优化实现逆强化学习与策略重构的闭环，生成高质量可解释欺诈序列。

**🔧 技术方法**

技术：逆强化学习（最大熵 IRL）、Actor-Critic 强化学习、候选动作生成（历史引导采样+探索采样）、线性可分解奖励模型、特征期望匹配、GAT/Transformer/GraphSAGE/XGBoost 检测器等。

**📊 数据集**

数据集：来自 Ant International 真实线上支付系统的约 1.29M 用户、2.63M 交易事件，包含 3.46M 标注的欺诈样本，覆盖 2025‑2026 年间的欺诈支付。

**📈 对比分析**

比较方法：与分布生成（CTGAN、Diffusion-TS、TabDiff）、政策学习（BC、MoE‑BiEntIRL）、真实欺诈重放等基线比较；在四种检测器（RT、TabFormer、GraphSAGE、XGBoost）上评估 AUC、Recall@2%、KS。PaySeqGen 在所有检测器上均取得最高或次高提升，尤其在 RT 与 GraphSAGE 的 Recall@2 方面分别提升 1.621% 与 4.561%，显著优于基线。

**⚠️ 局限性**

局限：① 生成的序列依赖于已收集到的欺诈轨迹，若新型欺诈行为不在演示中难以重构；② 生成过程需要预先构建支付反馈接口，若环境差异大可能影响生成质量；③ 仍需人工验证奖励解释，解释性不如完全可解释模型；④ 在大规模部署中仍存在计算与更新成本。

---

## 529. AgenticFinBench: Benchmarking Financial Agents on Diverse Tasks from Real-World Requests

**arXiv ID:** 2611.04813 | [PDF](https://arxiv.org/pdf/2611.04813v1)

**作者:** Shu Zhou `[一作]` (NashNova Research), Yong Hu `[通讯]` (NashNova Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个名为 AgenticFinBench 的金融任务基准，包含 230 条真实世界金融请求，并提出了参考无关（reference‑free）评估框架，将任务质量与事实可靠性结合为单一分数。

**💡 创新点**

创新点在于：① 采用两阶段声明审计（先判定是否有证据支持，再进行外部验证），分离“定位”与“真伪”；② 在每个问题上使用专门生成的 rubrics，避免仅靠模型自身对答案的自评；③ 将完整的模型–工具（harness）配置纳入比较，强调工具使用与推理效率的影响。

**🔧 技术方法**

技术方法包括：基于大型语言模型（如 GPT‑5.6 Sol、DeepSeek V4.1 Flash、Claude Opus 5.5 等）生成 rubrics 与声明提取；利用工具调用记录和执行轨迹进行证明检查；外部检索与计算验证声明；按权重与惩罚公式计算最终质量分 Q；对比 18 种模型–harness 配置。

**📊 数据集**

使用的数据集为 AgenticFinBench：230 条金融问题（189 中文、41 英文），涵盖 8 类任务、2 种语言、4 复杂度等级，采自 14,697 条 30 天内的真实请求和 23,731 条英文交互；每个问题附有 18 个配置的答案、轨迹和审计结果。

**📈 对比分析**

评估方法：在同一 230 题上对 18 个配置进行全程评测，计算质量分 Q、基线分 B、成本和生成时长；结果显示最高 Q≈89.2，排名受模型与 harness 决定；审计后排名变动明显，揭示错误事实对评分的实质影响；成本跨度达两百倍，时间从 0.94 分到 14.71 分不等，复杂任务的 Q 下降约 10 点，提示现有系统在复杂分析上仍有显著缺口。

**⚠️ 局限性**

局限性：① 样本量有限且非概率抽样，难以完全代表生产流量；② 评估仍依赖人工标注的 rubrics 与审计规则，可能存在主观性；③ 只覆盖金融领域，缺乏跨领域通用性；④ 两阶段审计虽严谨但可能漏检微妙错误；⑤ 仅评估 18 种配置，未涵盖所有模型或工具组合。

---

## 530. NightIQA: A Comprehensive Dataset and Benchmark for MLLM-Based Nighttime Image Quality Assessment

**arXiv ID:** 2611.04811 | [PDF](https://arxiv.org/pdf/2611.04811v1)

**作者:** Yiding Tian `[一作]` (Xiamen University Malaysia), Hongyuan Yu `[通讯]` (Xiaomi Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出NightIQA数据集与评测框架，专门针对夜间图像质量评估；

**💡 创新点**

填补夜间图像质量评估的结构化标注、评测协议与场景感知合成缺口；

**🔧 技术方法**

采用多模态大型语言模型（MLLM）+监督微调（SFT）与组相对策略优化（GRPO）进行训练；

**📊 数据集**

NightIQA-Real（1769张真实夜景图像），NightIQA-Bench（1078道多维度MCQ评测），NightIQA-Syn（80000张可扩展合成图像）；

**📈 对比分析**

与22个零拷贝基线对比，SFT+GRPO的Qwen3‑VL‑8B在所有四级（感知、技术评估、定位、推理）上均领先，整体准确率达到80.71%，优于约50倍规模的Qwen3‑VL‑397B与最强闭源模型；

**⚠️ 局限性**

仅基于现有6种失真类型，场景类别有限；合成方法仍依赖预设模型，缺乏更丰富的失真种类与跨域泛化能力；

---

## 531. DLER: Dual-Logit Experience Replay for Online Continual Learning

**arXiv ID:** 2611.04885 | [PDF](https://arxiv.org/pdf/2611.04885v1)

**作者:** Huiwei Lin `[一作]` (Hong Kong Polytechnic University), Kong Aik Lee `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在在线连续学习中提出 Dual‑Logit Experience Replay (DLER) 框架，通过同时使用余弦对数与点积对数来学习当前样本与回放样本，并结合监督对比损失和基于对数优势的记忆更新策略，缓解预测偏差与灾难性遗忘。

**💡 创新点**

创新点包括：① 将点积对数分解为范数因子与角度因子，利用角度因子学习新知识、范数因子回放旧知识；② 在回放阶段加入额外的点积对数监督；③ 引入监督对比损失提升类内角度紧凑性；④ 依据点积与余弦预测优势对记忆样本进行评分与替换，直接对应双Logit 训练目标。

**🔧 技术方法**

核心技术：角度–范数分解、双Logit 损失 (cosine + weighted dot‑product)、监督对比学习 (contrastive loss)、基于预测优势的记忆评分与替换策略、在线批量训练与梯度更新。

**📊 数据集**

使用的公开数据集有：Split CIFAR‑100、Split MiniImageNet 与 Split TinyImageNet，共计 30 个学习阶段，每阶段 10 类。

**📈 对比分析**

与 ER、UER、SS‑IL、LOR 等现有基线在多种缓冲区大小（100–4000）下进行对比。DLER 在所有 12 种设置下均获得最高平均准确率，提升幅度约 1–3%，并在后期学习阶段显著领先，验证了双Logit 与记忆更新的有效性。

**⚠️ 局限性**

局限性：需要额外的监督对比计算与记忆评分开销（O(B²d) 与 O(MF_pred)）；需手动调节超参数 α 与 λ_sc；目前仅在图像分类任务上验证，适用性待进一步扩展；在极小内存下仍受限于样本稀缺。

---

## 532. ScamWorld: An Interactive Environment for Evaluating Web Agents Against Online Fraud

**arXiv ID:** 2611.04879 | [PDF](https://arxiv.org/pdf/2611.04879v1)

**作者:** Yutao Wu `[一作]` (Deakin University), Xingjun Ma `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文构建了一个名为ScamWorld的交互式测试环境，用于系统评估基于大型语言模型的网络代理在面对真实在线诈骗网站时的行为。

**💡 创新点**

创新点在于：①利用公开报告重建多语言、多情境的诈骗网站，使诈骗机制可在代理执行前被可视化；②提供可重复、可控的评测流程，解决传统诈骗网站短生命周期导致的评测难题；③在同一环境下对多款前沿LLM代理进行基准测试，揭示其在支付、凭证获取等任务中的脆弱性。

**🔧 技术方法**

技术方法包括：构建可执行HTML环境模拟真实网站；多模态交互捕捉代理对页面元素的响应；使用任务脚本记录代理的决策轨迹；统计欺骗成功率和错误路径。

**📊 数据集**

数据集为从网络上公开报告收集的诈骗网站案例，涵盖不同诈骗机制、语言（中文、英文等）和行业场景（购物、投资、预付款等）。

**📈 对比分析**

与已有工作（如TrickyArena、SusBench等）相比，ScamWorld在多语言支持、诈骗机制可观察性和可重复性方面更优；实验结果显示六款LLM代理的欺骗成功率从约0%到40%不等，且在支付指令上更易受骗。

**⚠️ 局限性**

局限性包括：①仅覆盖已知的诈骗模式，可能漏掉新型或隐蔽的诈骗手段；②测试环境仍为模拟，未能完全复制真实网页的动态交互和后端逻辑；③评测主要集中在LLM代理，缺少对其他类型自动化工具的验证。

---

## 533. OCR in the Foundation Model Era: From Specialist Models to Unified Models

**arXiv ID:** 2611.04874 | [PDF](https://arxiv.org/pdf/2611.04874v1)

**作者:** Zhiheng Li `[一作]` (Chinese Academy of Sciences), Weiming Hu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本综述系统梳理了光学字符识别（OCR）从传统模板匹配到基于大规模预训练的统一模型的发展历程，重点定义了“忠实转录”作为 OCR 的核心目标，并对当代 OCR 系统进行五大类的归纳与对比，涵盖架构、数据、训练、评估与局限性；

**💡 创新点**

创新点在于提出以“忠实转录”作为 OCR 的统一评价标准，并将现代 OCR 系统划分为模块化工具包、专用文档 VLM、混合工作流、通用 VLM 和商业服务五类，系统分析其在结构恢复、幻觉、空间定位、推理成本等维度的差异，填补了以往单一任务或单一技术视角的空白；

**🔧 技术方法**

主要技术包括卷积与 Transformer 视觉编码器、跨模态 VLM 结构（视觉编码器→投影层→大型语言模型解码器）、端到端文本与结构生成、联合检测-识别网络、序列化空间表示、指令调优、RLHF/ RLVR 等多任务与强化学习训练方法；

**📊 数据集**

使用了多种公开与合成数据集，包括 ICDAR 场景文本集（Total-Text、CTW1500、ICDAR 2019 MLT）、文档结构集（PubLayNet、PubTabNet、FinTabNet、CROHME）、多语种/多脚本数据（CASIA-HWDB、IFN/ENIT、XFUND）、合成渲染图像与 OCR 生成语料；

**📈 对比分析**

通过对 OCRBench、OmniDocBench、Real5-OmniDocBench、MDPBench 等标准基准的结果进行对比，展示了不同模型家族在文本、表格、公式、布局等子任务上的性能分布；整体得分普遍超过 95 分，但表格子任务仍显差距，且大模型并不总是性能更好；

**⚠️ 局限性**

主要局限包括：生成式 OCR 可能出现幻觉与遗漏；空间定位精度受视觉分辨率与坐标量化限制；推理成本高（长序列与高分辨率输入）；多语言、长文档与复杂结构（如图表、化学式、乐谱）的恢复仍不完善；评测标准缺乏统一定义，导致结果解释受限。

---

## 534. PrimeSim: A Generalist World Model Simulator from Large-Scale Cross-Embodiment Demonstrations

**arXiv ID:** 2611.04870 | [PDF](https://arxiv.org/pdf/2611.04870v1)

**作者:** Travis Davies `[一作]` (PrimeBot Research), Hao Dong `[通讯]` (PrimeBot Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出PrimeSim，一种将预训练视频生成器转化为多视角、动作条件、可自回归的机器人世界模型模拟器；

**💡 创新点**

创新点在于（1）跨身体跨平台统一动作表示与帧对齐双路径动作编码，保证精确动作跟随；（2）视角感知交互式注意力与相机身份标记，实现多视角一致性；（3）“快视频慢记忆”后训练框架，通过chunk‑causal、四步稀疏采样、异步分布匹配以及层次化 KV‑缓存，既提升推理速度又保证长时序连贯；

**🔧 技术方法**

技术包括预训练Diffusion VAE（Wan2.2-TI2V-5B）、bidirectional 训练加流匹配目标、frame‑aligned dual‑path动作条件、view‑aware interleaved attention、chunk‑causal注意力、一致性蒸馏、异步分布匹配、层次化 KV‑缓存、四步采样、以及强化学习后训练GRPO；

**📊 数据集**

使用规模达30,110小时的机器人演示数据，涵盖22个平台、20个数据族（包括人类与仿真混合、单臂、双臂、手持、四轮等），并在LIBERO、XVLA、RoboTwin 2.0等公开基准上进行评估；

**📈 对比分析**

与五大开源动作条件世界模型（LingBot‑VA、DreamDojo、Cosmos‑3‑Nano、IRASim、Ctrl‑World）以及GE‑Sim、Thinker‑WM、EnerVerse‑FT等基准比较，PrimeSim在WorldSimProbe的四项指标（T2、T3、T4、T5）和EWMBench总体分数中均领先，推理速度最高（4步蒸馏下10.1 FPS，ViewFPS 30.35），并在policy评估与GRPO后训练中保持与真实仿真器高度一致；

**⚠️ 局限性**

局限性包括：①生成分辨率相对较低（192×256），对高保真视觉质量关注不足；②对新平台需少量（100–200步）微调而非零样本；③验证主要在仿真环境，尚未在真实机器人上证明模拟器可直接代替物理仿真来提升策略；

---

## 535. Robust Aggregation of Quantile Statistics: from Median to Interval

**arXiv ID:** 2611.04860 | [PDF](https://arxiv.org/pdf/2611.04860v1)

**作者:** Mingyu Song `[一作]` (Peking University), Yuqing Kong `[通讯]` (Peking University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

研究了在平方损失下，如何在已知信息结构弱代替（diminishing‑returns）条件下对边界值的分位数报告进行鲁棒聚合，提出了针对中位数和预测区间的正则化简单平均规则。

**💡 创新点**

提出了在弱代替假设下能够在最坏情况上降低遗憾的聚合规则，并揭示了预测区间覆盖率与聚合性能之间的非单调关系。

**🔧 技术方法**

基于最坏情况遗憾优化理论、信息结构分析和闭式最优聚合规则的推导。

**📊 数据集**

无数据集，纯理论分析。

**📈 对比分析**

与常数中点预测基准比较，展示在α<1/√n时正则化平均规则显著优于基准，在所有n≥2时对中位数聚合均优于基准。

**⚠️ 局限性**

仅在弱代替假设下有效，对更一般信息结构下的聚合无法保证优于基准，且仅考虑平方损失。

---

## 536. Coordinating Humanoid Loco-Manipulation through Action-Conditioned Latent Foresight

**arXiv ID:** 2611.04839 | [PDF](https://arxiv.org/pdf/2611.04839v1)

**作者:** Ju Dong `[一作]`, Jianwei Zhang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一种基于动作的潜在预见模型（action‑conditioned latent foresight），用于在 humanoid loco‑manipulation 任务中在执行前评估并挑选最佳的关节运动与手部动作序列。

**💡 创新点**

创新点在于：①将视觉进展预测与联合运动‑手部时序先验（joint temporal prior）相结合，对生成的序列进行前瞻性评分；②引入动作一致性正则化（action consistency regularization）以促使潜在预测保留动作相关信息；③在闭环决策中采用分段生成与重规划机制。

**🔧 技术方法**

核心技术包括：V‑JEPA 视觉编码器、Qwen3‑VL‑2B 视觉‑语言编码、Action DiT 生成器、GRU‑based 运动‑手部联合先验、逆动作解码器、以及基于子目标的视觉进展评估。

**📊 数据集**

使用了：SIMPLE 仿真基准（10个任务）以及 Unitree G1 实机 7个任务的演示数据；视觉编码器在 HIW‑500 公开数据集上预训练；演示子目标通过仿真/现实演示提取。

**📈 对比分析**

与 DiT4DiT、Fast‑WAM、VLA‑JEPA、GR00T N1.7、Ψ₀ 等基线对比，实验显示在仿真中平均成功率明显提升，实机测试中在标准与未见对象配置下都取得了最高或相近的成功率，闭环消融验证了动作一致性正则化和联合先验的性能贡献。

**⚠️ 局限性**

局限性包括：依赖演示子目标图像，对复杂关节开启和软物体抓取的鲁棒性不足；缺乏接触与物体状态监督；在极端未知环境下的泛化仍有限。

---

## 537. Anomaly Detection on Resource-Constrained Devices with Reservoir Computing

**arXiv ID:** 2611.04835 | [PDF](https://arxiv.org/pdf/2611.04835v1)

**作者:** Marco Angioli `[一作]` (Örebro University), Denis Kleyko `[通讯]` (Örebro University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了Mahalanobis距离的Reservoir状态（MDRS）在资源受限设备上的异常检测性能，并在MLPerf Tiny基准上进行评估。

**💡 创新点**

提出两项优化：用循环移位替代随机连接以消除N×N矩阵乘法，使用仅最终状态的异常评分（end‑only）减少Mahalanobis距离计算次数，从而显著降低计算量。

**🔧 技术方法**

技术包括Reservoir Computing（Echo State Network）、Mahalanobis距离、log‑mel特征预处理、固定随机投影、循环移位替代矩阵、硬件实现（NUCLEO‑F401RE）等。

**📊 数据集**

使用MLPerf Tiny异常检测任务，基于DCASE2020 toy‑car 机器声音数据集（正常与异常录音）。

**📈 对比分析**

与参考autoencoder及多种基线方法对比，MDRS在满足AUC≥0.850的前提下，运算量、能耗、内存显著降低；在单帧、32个mel频带配置下，MDRS实现0.32 ms延迟、24 µJ能耗、24 KB模型，速度和能耗分别比autoencoder提升约75.9×和11.3×。

**⚠️ 局限性**

局限包括仅在单一声学数据集验证，缺乏对其他传感器和任务的测试；未实现在线增量更新或完整硬件加速；对极端噪声或长序列的鲁棒性未知。

---

## 538. Trusting Imagined Futures: Auditing Imagined Rewards for VLA Post-Training

**arXiv ID:** 2611.04873 | [PDF](https://arxiv.org/pdf/2611.04873v1)

**作者:** Zicheng Kong `[一作]` (Beijing University of Posts and Telecommunications), Zhaofeng He `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种称为 Trusting Imagined Futures (TIF) 的监督选择方法，用于视觉‑语言‑动作 (VLA) 模型的离线后训练，通过审计从想象视频得到的奖励来选取真实执行轨迹进行行为克隆。

**💡 创新点**

创新点：① 结合多维奖励模型与风险审计器，学习从匹配的真实执行反馈中辨别奖励过估；② 在候选轨迹中通过优势门限和审计风险阈值决定是否替换源轨迹；③ 提供跨 LIBERO 套件的实证评估，验证该方法在真实机器人上的有效性。

**🔧 技术方法**

技术：多维奖励模型（使用 DINOv2‑L 编码器提取视频特征并联合预测五维执行质量与整体效用）；风险审计器（基于预测分布与不确定性做二分类）；视频世界模型（Cosmos‑Predict2 2B 生成想象视频）；LoRA 行为克隆；基于 OpenVLA 的视觉语言动作模型。

**📊 数据集**

数据集：40 个任务、四个 LIBERO 套件（Spatial、Object、Goal、Long）；32,000 对真实执行轨迹与想象视频；另外 300 条 JAKA S5 机器人实验轨迹（3 任务 × 20 组）。

**📈 对比分析**

与 OpenVLA 原始、Source、贪心、随机、奖励仅、奖励+边际等 baseline 比较；在 LIBERO 上 TIF 平均成功率 76.25%（比 Source 高 3%），在机器人实验中清洁完成率从 58.33% 提升至 70%；在固定替换数的比较中，TIF 仍优于优势排名、逻辑回归风险和随机等。

**⚠️ 局限性**

局限：仅在 OpenVLA 与 LIBERO 上验证，未测试其他策略或基准；需要真实执行反馈训练审计器，若模型或候选分布改变需重新适配；实验环境为离线模拟与有限机器人实验，可能无法完全覆盖复杂真实场景。

---

## 539. Physics-Based Plausibility Assessment of AI-Predicted Vehicle Parameters for Automotive Radar in Pre-Crash Scenarios

**arXiv ID:** 2611.04871 | [PDF](https://arxiv.org/pdf/2611.04871v1)

**作者:** Rainer Engert `[一作]` (Technische Hochschule Ingolstadt), Thomas Brandmeier `[通讯]` (Technische Hochschule Ingolstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于物理模型的可解释性一致性评估模块，对AI预测的汽车雷达姿态进行后处理校正。

**💡 创新点**

创新点在于：1）使用物理雷达回波模型（考虑遮挡、面向角、RCs、微多普勒）直接生成预期雷达点云；2）利用Wasserstein距离对三类特征（空间分布、检测计数、反射强度）进行一致性度量；3）在不改动AI网络的情况下实现帧级姿态校正。

**🔧 技术方法**

技术包括：雷达信号处理（FFT、DBF、CFAR、DBSCAN）、深度学习检测（ComplexYOLO）、物理雷达模型（Po近似、RCs、微多普勒）、Wasserstein距离、旋转搜索优化。

**📊 数据集**

使用了INR AS雷达日志数据库，包含92条室内测试轨迹，共1192帧用于测试，另外7324帧用于验证和参数校准。

**📈 对比分析**

与原始AI预测进行对比，采用RMSE、MAE、误差分布等指标。校正后RMSE从5.28°降至4.29°，MAE从3.50°降至2.84°，大误差（≥5°）比例下降7.7个百分点，统计显著性p<1e-18。

**⚠️ 局限性**

局限性包括：参数需经验校准，难以迁移到其他雷达设备；当仅有单边或无轮胎信息时校正效果下降；校正过程对计算资源有一定开销，且在某些视角下Wasserstein距离可能出现多模态导致局部最优。

---

## 540. Perception and Utilization of Data Carpentry Tools among Research Scholars at University of Kerala

**arXiv ID:** 2611.04828 | [PDF](https://arxiv.org/pdf/2611.04828v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 541. Found by Search, Broken by a Shortcut: Transfer and Tampering of Searched Intrinsic Rewards

**arXiv ID:** 2611.04838 | [PDF](https://arxiv.org/pdf/2611.04838v1)

**作者:** Chase J. Metoyer `[一作]` `[通讯]` (Northeastern University), Chase J. Metoyer (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在MiniGrid环境中搜索并评估基于四个通用事件（新视角、拾取、放下、切换）构成的内在驱动奖励函数，检验其在训练任务、未见任务以及未在搜索时出现的奖励滥用（重置攻击）下的学习表现。

**💡 创新点**

① 公开预注册搜索协议并锁定实验流程，保证实验可重复性；② 用极简可解释的四驱动奖励空间取代传统手工设计的奖励；③ 结合CMA-ES与等预算随机搜索对比搜索效果；④ 对搜索奖励的迁移性与抗攻击性进行系统评估。

**🔧 技术方法**

采用PPO强化学习、CMA-ES进化搜索、等预算随机搜索；使用MiniGrid提供的任务；构建自定义事件检测器；进行统计显著性检验（Mann‑Whitney U、Holm校正）和可视化分析。

**📊 数据集**

MiniGrid任务集：训练任务 DoorKey-8x8 与 UnlockPickup；留出任务 KeyCorridorS3R2；在每个任务上使用多重种子（s3.seeds）进行实验。

**📈 对比分析**

通过注册化的一元Mann‑Whitney U检验比较搜索奖励（E）与任务专用PPO（T）、情节新颖度（N）和随机搜索奖励（R）的训练曲线面积；在留出任务上，E击败 T、N、R；在训练任务上，E 与 R 无显著差异；在含重置攻击的测试中，E 的学习面积下降至低于 T，说明搜索奖励未能抵御未见的攻击。

**⚠️ 局限性**

局限性：① 迁移性仅在单一留出任务验证；② 只使用MiniGrid，未覆盖更复杂或多样化的环境；③ 只评估了单一奖励形式（四驱动）和单一学习者（PPO），不一定能推广到其他算法；④ 随机搜索与CMA-ES的比较并非独立评估；⑤ 事件检测器与早期工作不一致，导致跨实验对比受限；⑥ 未包含更广泛的奖励攻击场景，搜索对其他潜在滥用的鲁棒性未知。

---

## 542. HTCE: Homeomorphism-inspired Topological Cross-Scene Experience Canonicalization and Reuse for Object Navigation

**arXiv ID:** 2611.04836 | [PDF](https://arxiv.org/pdf/2611.04836v1)

**作者:** Xiaoran Xu `[一作]` (Institute of Automation Chinese Academy of Sciences), Changsheng Xu `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于同伦启发的拓扑框架，对跨场景的导航经验进行规范化与重用。

**💡 创新点**

创新点在于引入特征细分和拓扑细分的可逆归约，保留语义锚点与分支关系，实现经验的无损压缩和结构对应。

**🔧 技术方法**

使用了语义图归约、编辑距离匹配、分支结构签名以及冻结的视觉语言模型作为策略。

**📊 数据集**

在 Habitat ObjectNav 的 HM3D v0.1、HM3D v0.2 与 MP3D 三个室内数据集上进行实验。

**📈 对比分析**

采用同一 frozen VLM backbone、相同动作空间与步骤预算，比较 SR 与 SPL，取得最高 SR（59.9/71.0/46.6）和 SPL（31.8/31.5/16.6），相较基线提升 4.9–3.1 点。

**⚠️ 局限性**

局限性包括对语义角色映射与细分规则的依赖，且对不具备相同语义划分的环境或需要更细粒度差异的任务效果不一定理想。

---

## 543. Geometric Bin Packing Beyond Convexity: Approximation Schemes without Augmentation

**arXiv ID:** 2611.04829 | [PDF](https://arxiv.org/pdf/2611.04829v1)

**作者:** Lucas de Oliveira Silva `[一作]` (University of Campinas), Flávio Keidi Miyazawa `[通讯]` (University of Campinas)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种通用框架，给定任何满足两条统一假设的固定多边形类，设计了无增量（no bin augmentation）的 APTAS（渐进多项式时间逼近方案）用于平移（translation‑only）几何箱装问题，并返回精确的有理坐标解。

**💡 创新点**

创新点在于：①引入了“统一可栅格化（uniform rasterizability）”与“统一可代表化（uniform representative roundability）”两条抽象性质，极大地扩展了可处理的对象范围；②利用“翻译包含宽度（translation‑containment width）”的概念实现代表化；③构建多层正方形层级结构的结构化装箱，结合平均移位网格和删除策略，将任意装箱转化为结构化装箱；④通过配置整数规划与带有有限域的动态规划在多项式时间内求解最优结构化装箱；⑤所有步骤均保持有理坐标，不需要增大箱尺寸。

**🔧 技术方法**

主要技术包括：网格覆盖（grid covering）与外部 Minkowski 边界控制、代表化与链分组（linear grouping）实现对象替换、层级正方形结构化装箱、平均移位网格删除策略、配置整数规划（configuration IP）以及 Jansen–Kratsch 的有限域动态规划求解。

**📊 数据集**

本文为理论算法研究，未使用实验数据集；所有结果均为理论证明与多项式时间复杂度分析。

**📈 对比分析**

与现有工作相比，本文在无增量、支持非凸多边形（甚至带孔）且无边界复杂度上限的前提下，提供了最一般的 APTAS；相较于圆、正方形、L_p 球、倾斜矩形等特例，覆盖面更广，且不需要增大箱尺寸。性能方面，给定任意 ε>0，算法在多项式时间内得到 (1+ε)·OPT+O(1) 的箱数。

**⚠️ 局限性**

局限性：①只能处理固定多边形类，需满足统一可栅格化与可代表化；②需要已知翻译包含宽度上界 d，若 d 过大则常数系数增大；③不支持旋转、翻转或非平移变换；④对非常细长或极端非凸形状，虽然理论上可行但常数可能较大，实际实现难度较高。

---

## 544. Density-Aware Reward Scaling: A Reinforcement Learning Framework for Reward-Proportional Sampling

**arXiv ID:** 2611.04792 | [PDF](https://arxiv.org/pdf/2611.04792v1)

**作者:** Ria Shekhawat `[一作]`, Dianbo Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

证明了在连续输出空间中逆政策密度缩放可得到奖励比例策略，并且展示了使用完整样本组的高斯核密度评分与策略密度近似等价的理论基础。

**💡 创新点**

将逆概率缩放推广到连续空间，利用指数族参数化和功能梯度得到唯一稳态奖励比例密度；提出全组高斯核评分方式，证明其与策略密度误差可控。

**🔧 技术方法**

功能梯度、指数族参数化、逆概率缩放、拉普拉斯展开、核密度估计、Chebyshev不等式、偏微分与收敛性分析等。

**📊 数据集**

未使用具体数据集，论文仅给出理论推导与数值不等式。

**📈 对比分析**

论文未进行实验比较，主要通过数学证明展示方法在理论上的收敛性和近似误差。

**⚠️ 局限性**

假设奖励函数可测且积分有限，策略密度需正且可微；高斯核估计对带宽和样本量敏感，边界效应需处理；理论结果适用于无噪声的连续情形，实际应用中可能需要更多假设验证。

---

## 545. Extracting System Prompt of LLM Services in Complex (Agentic) Settings: Hierarchical Optimization and Reliable Evaluation Metrics

**arXiv ID:** 2611.04803 | [PDF](https://arxiv.org/pdf/2611.04803v1)

**作者:** Shuaidong Li `[一作]` (Xi'an Jiaotong University), Zhengyu Zhao `[通讯]` (Xi'an Jiaotong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文针对LLM服务中的系统提示（system prompt）提出了一种新的黑盒提取攻击框架，专门解决在多层次、结构化、以及仅能观测后处理输出的复杂代理（agentic）环境下的提示泄露问题。

**💡 创新点**

创新点在于：①将提取任务拆分为多层次结构化分解（Hierarchical Multi-Stage Decomposition）以捕获逻辑、工具、格式等模块；②利用文本梯度进行迭代细化（Iterative Refinement via Text Gradients）以在仅有最终动作或简化文本的反馈中逐步逼近完整提示；③设计双轨收敛控制（Dual‑Track Convergence Control）监测信息增益与梯度稳定性，实现早停与重置；④提出结构关键词匹配（SKM）和对抗性效用（Adversarial Utility）等可靠评估指标。

**🔧 技术方法**

技术实现包括：黑盒查询与反馈循环、文本梯度估计、预定义结构化模板、语义锚点驱动的查询生成、信息增益与相似度滑动窗口判定、以及基于语义相似度与结构匹配的后处理评估。

**📊 数据集**

数据集涵盖：150条从人工合成到真实泄露的系统提示，按五个复杂度层级（L1–L5）划分；六种GUI代理（MobileAgent、AppAgent、BrowserUse、SeeAct、Agent‑S、UFO）在不同平台上采集的100条多模态任务回应，用以评估功能与对抗效用。

**📈 对比分析**

在与 Naïve、O₂P、RTS、PRSA、T‑GPS 等基线的对比中，ExtraHerO 在聊天机器人上平均提升 12.5% 的 fidelity、2.3% 的功能性、6% 的对抗效用；在代理系统上提升 12.2%、34.3% 与 30.5%；在结构关键词匹配与句子语义相似度上均表现显著优于所有基线，且能够在后处理输出受限的情形下保持较高的提取质量。

**⚠️ 局限性**

限制与不足包括：仍需要大量查询以完成多层次优化，受限于预定义模板和语义锚点，强防御（如指令阻止泄露）会显著削弱效果，对高度冗余或隐式逻辑的提示提取仍然存在瓶颈，且目前仅在文本和单模态场景验证，跨模态与更高层次代理的通用性尚待进一步研究。

---

## 546. Training Specialist LLMs on Your Phone: Forward-Only Optimization via Block-Spectral Coordinates

**arXiv ID:** 2611.04797 | [PDF](https://arxiv.org/pdf/2611.04797v1)

**作者:** Liyan Tan `[一作]` (University of California, Santa Barbara), Zheng Zhang `[通讯]` (University of California, Santa Barbara)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在移动设备上实现仅前向传播的LLM专属模型训练系统，利用块奇异值坐标与演化策略在INT4模型上进行高效微调。

**💡 创新点**

创新点包括：①仅更新块奇异值的前向只优化方法（FuRA‑S），能在保持谱继承的同时极大压缩可训练参数；②将演化策略与块奇异值配合，避免了传统全参数ES对内存的高需求；③引入逐步适配奖励，解决多步任务中单一奖励难以指导学习的问题；④可在智能手机上仅使用现有INT4权重完成全模型的适配与推理。

**🔧 技术方法**

主要技术：块奇异值分解（Block‑SVD）、演化策略（Evolution Strategies, ES）、前向只优化（Zero‑Order、MeZO）、步长奖励（Step‑wise fitness）、INT4权重量化保持不变的权重变换。

**📊 数据集**

使用的数据集与任务包括：MobileWorld 手机交互任务、LaMP 个人化对话与文本生成、Countdown 算术推理、MATH 计算题、AIME24/AMC 数学竞赛、Llama‑3‑8B、Qwen2.5‑1.5B/3B/7B、GPTQ‑INT4 量化模型。

**📈 对比分析**

与方法对比：与全参数 ES、LoRA‑ES、VeRA‑ES、QES、QuZO、传统策略梯度（PPO/GRPO/Dr.GRPO）等进行对比。实验表明：FuRA‑S 在保持 0.05% 可训练参数的情况下，性能与全参数 ES 相当或略优；在 LoRA‑ES、QES 上显著超越；在 MobileWorld 任务上将成功率从 6–23% 提升至 89–100%；在 LaMP 个性化任务上超越基于 BP 的 LoRA；在 iPhone 16 Pro 上可在 10‑12 小时内完成 8B INT4 模型的专属微调。

**⚠️ 局限性**

局限性：训练仍较慢，需评估大量随机扰动模型且多步任务成本高；需要设备可计算的无标签适配奖励，设计奖励仍是挑战；专属模型之间不共享知识，难以利用跨任务的迁移；评估范围仅覆盖部分手机交互与文本任务，尚未验证在更广泛场景中的鲁棒性。

---

## 547. Accelerating Offloading-Based MoE Inference by Interleaving Concurrent Requests

**arXiv ID:** 2611.04794 | [PDF](https://arxiv.org/pdf/2611.04794v1)

**作者:** Hongseung Yu `[一作]` (Seoul National University), Lin Zhong `[通讯]` (Yale University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在本地 Mixture-of-Experts（MoE）语言模型推理中，提出一种基于 runahead 推理的请求调度框架，能够让已就绪的请求提前到更深层级并提前发现专家需求，从而在不做预测的前提下充分利用链路带宽和 GPU 计算。

**💡 创新点**

创新点：①将传统批量执行拆分为按层次的动态“跑 ahead”执行；②引入最低层优先（lowest‑layer‑first）调度与自适应层级聚合，形成自平衡的请求分组；③在专家缓存中通过引用计数共享 fetch，避免重复读取；④在实验中证明预取在存在缓存时效果反而下降，真正提升性能的是请求调度。

**🔧 技术方法**

技术手段：运行时动态调度器（Compute Scheduler + Job Issuer + Expert Fetcher）；按层次的 opportunistic grouping；SRTF 与 lowest‑layer‑first 的混合调度；LFU 缓存置换；SSD‑based expert offloading；基于 llama.cpp 的引擎改造，使用 CUDA/Metal 进行实验；量化模型 Q4_0；使用 Qwen3.6‑35B‑A3B 与 Gemma4‑26B‑A4B 两大 MoE 模型。

**📊 数据集**

数据集与工作负载：- 模型：Qwen3.6‑35B‑A3B 与 Gemma4‑26B‑A4B；- 任务：Chat（ShareGPT 提问集）与 Agentic Coding（SWE‑agent 轨迹）。

**📈 对比分析**

比较方法：在统一内存平台（Apple M6 与 NVIDIA Jetson AGX Thor）上，以并发请求数 2–16、专家缓存比例 10–70% 进行 decode 速率（tok/s）评估；基线为 llama.cpp、SpecMD、FineMoE；性能表现：对所有配置，系统可获得 1.26–3.46× 的 decode 速率提升，相比最快基线 1.01–1.55×，并在多请求/大缓存场景下优于预取方案。

**⚠️ 局限性**

局限性：①需要足够的 GPU 缓存条目（至少能容纳所有并发请求的活跃专家）才能避免死锁；②仅针对统一内存设备；③在极大缓存或极少并发请求时，预取并不帮助，系统性能受限于缺失专家的 fetch 速度；④未评估多 GPU 或多节点服务器环境；⑤缺乏对不同预取策略的深入对比。

---

## 548. Modelling lock-in of AI adoption through skill erosion

**arXiv ID:** 2611.04886 | [PDF](https://arxiv.org/pdf/2611.04886v1)

**作者:** Renaud Lambiotte `[一作]` `[通讯]` (University of Oxford), Renaud Lambiotte (University of Oxford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个将人工智能采用与人类技能退化耦合的精确解析模型，推导出技能分布的平稳解，并通过模拟验证了二元稳定性和滞后效应。

**💡 创新点**

首次给出技能分布的闭式解，揭示了技能退化与 AI 采用之间的双向反馈导致的锁定与逆转门槛，并阐释了行为持续性如何削弱锁定。

**🔧 技术方法**

使用共进化博弈论、线性 pairwise comparison（Fermi 规则线性化）以及一阶常微分方程解析，配合大规模 agent‑based 模拟。

**📊 数据集**

无外部真实数据，使用纯粹的模拟数据验证理论结果。

**📈 对比分析**

将解析的稳定分布与 agent‑based 模拟结果进行对比，误差小于 1%，并通过极值折叠线判断二稳区，模拟进一步展示了突发提升对长期采用的影响。

**⚠️ 局限性**

模型假设人口均质、线性收益、全局混合交互，未考虑网络结构、任务异质性和 AI 性能随采用度变化，缺乏经验校准。

---

## 549. Reproducibility in open-source RISC-V HW flows

**arXiv ID:** 2611.04883 | [PDF](https://arxiv.org/pdf/2611.04883v1)

**作者:** Peter Kourzanov `[一作]` (imec), Anmol Anmol `[通讯]` (imec)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了基于Guix的可复现RISC‑V硬件验证流程，集成CVA6架构及其相关工具链（Spike、QEMU、Verilator等），并实现了标准化的构建与测试入口；

**💡 创新点**

提出并实现了在硬件开发流程中借鉴软件构建经验的关键技术——依赖管理、确定性、早期裁剪、自跟踪与最小化，并通过精确哈希保障每一步的可重现性；

**🔧 技术方法**

使用Guix包管理器作为核心平台，结合RISC‑V交叉编译工具链、模拟器（Spike、QEMU、Verilator）以及相关文档与验证脚本；

**📊 数据集**

采用RISC‑V CVA6单核处理器以及其标准测试套件，作为评估与验证的主要数据集；

**📈 对比分析**

通过Guix构建产生唯一的哈希值，验证流程在不同机器上均能产生完全一致的空白产物；虽然未与其他包管理器或工具链做量化性能对比，但已证明其在确定性与可复现性方面的可行性；

**⚠️ 局限性**

目前仅覆盖验证与仿真层面，未扩展到后端映射与物理实现；支持的模拟器与工具有限，需进一步集成更多生态工具与后端流程；

---

## 550. Faster Regular Expression Matching via Two-Dimensional Tabulation

**arXiv ID:** 2611.04852 | [PDF](https://arxiv.org/pdf/2611.04852v1)

**作者:** Philip Bille `[一作]` (Technical University Of Denmark), Inge Li Gørtz `[通讯]` (Technical University Of Denmark)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种新的正则表达式匹配算法，在单位成本 word RAM 模型下使用线性空间，时间复杂度为 O(nm·loglog t / log² t + t + n)，其中 t≥m^{1+ε}，并对常数字母表给出了更强的 O(nm / log² t + t + n) 上界。

**💡 创新点**

创新点在于构造专门针对给定正则表达式的二维表格（shortcut micro TNFA）以及嵌套分解，利用预处理的表格一次性查询状态集转换，从而实现对文本子串的批量处理，显著减少了对状态转移的重复计算，导致在常数字母表情况下将前沿 O(nm) 线性项缩小 √(log n) 倍。

**🔧 技术方法**

核心技术包括：
- Thompson 构造的非确定有限自动机（TNFA）与其嵌套分解；
- 通过引入伪字符 β 对子表达式进行分块，生成 micro TNFA；
- 将子表达式的子串匹配信息压缩为短字符串标签，构造 shortcut micro TNFA；
- 对每个 micro TNFA 与所有长度≤y 的子串预处理四种微型 TNFA 函数，并将其存入二维查找表；
- 采用三阶段状态集转移（先求通往接受状态的前缀，再求通往起始状态的前缀，最后合并得到完整状态集）。

**📊 数据集**

该工作不使用实际数据集，而是给出理论时间/空间复杂度分析；实验验证部分在原稿中未给出。

**📈 对比分析**

与之前最优的 Bille–Thorup O(nm·loglog n / log^{3/2} n + n + m) 算法相比，新算法在常数字母表情况下把前导项从 log^{3/2} n 缩小到 log² n，提升约 √(log n) 的速度；在更一般情形下仍保留相同的 polylog 下降。

**⚠️ 局限性**

限制包括：
- 对常数字母表的加速是针对特殊情况，通用字母表的改进尚未达到同等水平；
- 预处理阶段需要占用 O(t) 空间，t≥m^{1+ε}；
- 仍然存在多项式对数的差距（上界与下界的 polylog 隔阂尚未完全消除）。

---

## 551. Amortized Nonlinear Variable Selection with Prior-Data Fitted Networks

**arXiv ID:** 2611.04848 | [PDF](https://arxiv.org/pdf/2611.04848v1)

**作者:** Jin Zhu `[一作]` (University of Birmingham), Junxian Zhu `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种基于Transformer的可预训练变量选择方法MambaPFN，能够在不重新训练的情况下快速识别新数据集中的重要变量

**💡 创新点**

创新点在于将先验数据拟合网络（PFN）理念扩展到变量选择任务，设计了专门的预训练数据生成、目标函数和改进的Transformer架构，并展示了可扩展的“规模定律”

**🔧 技术方法**

使用Transformer（自注意力、MLP等）、贝叶斯后验包含概率估计、可区分的响应嵌入以及双向自注意力的输出模块

**📊 数据集**

先在合成的DAG生成的数据集上预训练，随后在Facebook评论量预测数据（53个特征、约4万条记录）上进行真实数据评估

**📈 对比分析**

与Lasso、SCAD、ABESS、DFS、STG、HSIC‑Lasso、Knockoffs、CauScale等方法对比，MambaPFN在F1、AUC、R²和变量数上均表现最佳，且运行时间最短（约10倍以上的加速）

**⚠️ 局限性**

局限性包括：预训练成本较高；当前仅输出变量选择结果，需要单独训练预测模型；缺乏对不同任务的直接可解释预测能力

---

## 552. Fault-tolerant shallow recurrent decoders for sparse sensing with missing sensor measurements

**arXiv ID:** 2611.04893 | [PDF](https://arxiv.org/pdf/2611.04893v1)

**作者:** Matteo Romilio Pasqual `[一作]` (Politecnico di Milano), Andrea Manzoni `[通讯]` (Politecnico di Milano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了在缺失、异步传感器数据下的鲁棒稀疏感知方法——通过在SHRED模型中引入Transformer级别的遮蔽机制和多步解码训练，以实现高维状态的准确重建。

**💡 创新点**

创新点在于四种Transformer遮蔽策略（通道级、隐藏层级、多头自注意力级和学习型自注意力级）以及多步解码训练策略，使模型能够显式感知传感器可用性并提升对缺失数据的鲁棒性。

**🔧 技术方法**

技术方法包括SHRED框架、GRU‑D/attnGRU‑D缺失值处理、Transformer编码器与多头自注意力、遮蔽机制、POD降维、深度神经网络解码器和多步损失优化。

**📊 数据集**

实验使用四个数据集：合成浅水方程（1008帧）、英国北海水位与流速（25599维）、Øresund海峡水位与流速（9980维）以及全球海面温度（64800维），其中包含真实的缺失与异步采样。

**📈 对比分析**

通过与传统Gappy‑POD、SDN、GRU‑D、imputation以及仅使用全程可用传感器的T‑SHRED基准进行对比，遮蔽SHRED在RMSE和NMAE上普遍优于基线，尤其在缺失率高的情形下提升幅度可达30%–50%，且计算效率保持与标准SHRED相近。

**⚠️ 局限性**

局限性包括遮蔽策略对部分缺失模式的处理仍不完备，学习型遮蔽需要额外参数且对超参数敏感，且在极端缺失率或高度非线性动力学下仍可能出现重建误差；未来可考虑模型集成与参数不确定性传播。

---

## 553. Real-to-Integer Reduction for All-Pairs Shortest Paths

**arXiv ID:** 2611.04847 | [PDF](https://arxiv.org/pdf/2611.04847v1)

**作者:** Nick Fischer `[一作]` (Max-Planck Institutefor Informatics), Jonas Schmidt `[通讯]` (Bocconi University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种随机化组合的子立方体化实数到整数的归约框架，证明如果整数版本的 APSP 或 Exact Triangle 能以真正的子立方时间求解，则其对应的实数版本也能以真正子立方时间求解。

**💡 创新点**

创新点在于结合 Frank–Tardos 定理、Freiman 定理、近似加法哈希以及 Doubling Reduction 等多种技术，首次实现了从实数到整数的高效归约，并在 APSP 上突破了传统整数技巧与实数输入之间的鸿沟。

**🔧 技术方法**

主要技术包括：1) 近似实数与整数之间的 Frank–Tardos 变换；2) 通过 Doubling Reduction 对权重集进行结构放大；3) Freiman 定理实现子集和覆盖；4) 近似哈希函数用于缩小数域；5) Fredman’s Trick 与支配乘积（dominance product）结合，用于处理不等式约束；6) 近似选择+加法秩（approximate select‑plus rank）与潜在调整（potential‑adjusting）机制。

**📊 数据集**

由于论文属于理论算法研究，没有使用实验数据集；评估基于理论时间复杂度分析与已有算法（如 O(n^{2.9995}) 的 Floyd–Warshall、Alman & Vassilevska Williams 的突破等）进行比较。

**📈 对比分析**

与现有整数 APSP 算法相比，若整数 APSP 在 O(n^{3-ϵ}) 时间内可解，则该归约可在 O(n^{3-Ω(ϵ^3)}) 甚至 O(n^{3-Ω(ϵ^2)}) 时间内解实数 APSP；类似的结果也适用于 Exact Triangle。与现有实数 APSP 方案（如 Alman & Vassilevska Williams 的方法）相比，该归约提供了组合式的、无需矩阵乘法的实现，并在“实数 RAM”模型下证明 Real APSP 假设不成立。

**⚠️ 局限性**

局限性包括：1) 归约的随机化性质（仅高概率正确）；2) 需要在实数 RAM 模型中允许乘法或比较加法运算，模型复杂度不完全与实际硬件匹配；3) 归约导致的时间损失仍为 Ω(ϵ^3)（或 Ω(ϵ^2)），比直接整数方案略高；4) 对于更一般的实数输入（无限精度、非对称权重等），归约的适用性尚未完全证明；5) 需要依赖高阶组合技巧，理论实现相对复杂。

---

## 554. RAHC: Rule-Augmented Hyper-Relational Knowledge Graph Completion for Materials Synthesis

**arXiv ID:** 2611.04809 | [PDF](https://arxiv.org/pdf/2611.04809v1)

**作者:** Jiawei Sun `[一作]` (University of New South Wales), Zhengyi Yang `[通讯]` (University of Sydney)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Rule-Augmented Hyper-relational Knowledge Graph Completion (RAHC) 框架，用于从材料合成实验记录中推断缺失的实验条件。

**💡 创新点**

创新点包括：①构建可追溯的规则目录并对每个候选值进行候选特定检索；②将全局规则可靠性与查询相关性分离，提供 RAHC-G（全局聚合）和 RAHC-Q（查询注意力聚合）；③使用硬规则约束与软规则排序同时处理单槽与多槽联合完成；④通过门控融合将规则证据与结构表示结合，提升可解释性与性能。

**🔧 技术方法**

使用技术：超关系知识图谱建模、规则挖掘与文献抽取（AMIE+/AnyBURL 等）、双编码器（结构与规则编码）、注意力聚合、门控融合、联合损失训练；在完全归纳实验中使用 NodePieceGNN 作为结构编码器。

**📊 数据集**

使用数据集：MOF-SynHKG、SynMOF、FAIR-MOFs（MOF 合成记录）；Solution-based Inorganic（无机前驱体）；ZeoSyn（沸石 OSDA）；以及 ILPC'22 Small（实体完全不重叠的 KG 完成基准）。

**📈 对比分析**

方法对比：与 InGram、NBFNet、NS-HART、MAYPL、AnyBURL、HyperPath 等多种基线在 Semi-inductive、Open-Answer Inductive 和完全归纳场景下进行比较。RAHC 在单槽任务中 MR R 提升 14.8%（溶剂）至 15.6%（温度），在两槽联合任务中提升 8%–10%；在 ILPC'22 Small 上 RAHC-G 的 MRR 为 0.08965，显著优于 NodePieceGNN（0.08659）。

**⚠️ 局限性**

局限性：规则覆盖仍有限，尤其对极少出现或新材料的预测受限；规则检索和聚合过程复杂且计算开销较大；在完全归纳场景下提升幅度相对较小；未能充分解决多槽组合中规则不完整或冲突的问题。

---

## 555. NP-Hardness and a PTAS for Maximum Interval Multicover, with Applications to Dense Subgraphs

**arXiv ID:** 2611.04889 | [PDF](https://arxiv.org/pdf/2611.04889v1)

**作者:** Morteza Alimi `[一作]` (University of Augsburg), Michael Ruderer `[通讯]` (University of Augsburg)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并研究了最大区间多重覆盖（Maximum Interval Multicover）问题，证明其在等长区间上仍为NP‑hard，并给出了通用奖励函数的多项式时间逼近方案（PTAS）；进一步将该框架推广至凸二分图的稠密子图问题、区间图的最大k‑顶点覆盖以及加权团计数等问题；

**💡 创新点**

1) 证明最大区间多重覆盖即使在等长区间上仍是NP‑hard，解决了此前未确定的复杂度问题；2) 构造了一套基于区间分块、交叉族合并与动态规划的PTAS，能够处理任意非负非递减奖励函数；3) 将该PTAS延伸至凸二分图、区间图等结构，改进了先前的2/3逼近，并首次为这些问题提供了PTAS框架；

**🔧 技术方法**

使用了区间分块与移位技术（shifting），交叉族的“合并”技术将选择问题转化为动态规划状态，结合了上下闭合（upward‑closed）性质以及“merge”对应的增量计数，最终实现了多项式时间内求解窗口子问题，并通过窗口组合实现全局PTAS；

**📊 数据集**

无实验数据集，全部为理论证明与算法复杂度分析；

**📈 对比分析**

通过与之前的1/2近似算法、Li等人提出的Partial Interval Multicover PTAS、Backer & Keil的2/3近似等进行理论比较；证明该PTAS可在任意给定ε得到 (1‑ε) 近似，并且在等长区间上与先前的1/2近似相比，显著提升了逼近质量；

**⚠️ 局限性**

尽管给出了PTAS，但若要求FPTAS则不可能（除非P=NP）；对最大区间多重覆盖的高效PTAS仍未实现，且对共同需求或大规模实例的实际性能尚未评估；此外，算法对最大需求q_max指数级复杂，属于XP类而非FPT。

---

## 556. Best-of-Many-Worlds MMS Allocation for Chores with Sublinear Disposals

**arXiv ID:** 2611.04884 | [PDF](https://arxiv.org/pdf/2611.04884v1)

**作者:** Zehan Lin `[一作]` (University of Macau), Shengwei Zhou `[通讯]` (Nanyang Technological University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种随机分配框架，证明在任意可加代价的无价值物品分配中可以实现预期比例（ex‑ante proportionality）并且在每个结果中几乎所有代理人都满足MMS1（即仅移除最多一件物品即可满足MMS）

**💡 创新点**

创新点在于将问题转化为对分数分配（fractional allocations）的随机化，然后通过协调的单位槽（unit‑slot）舍入实现强大后验公平性；同时显著降低了需要处置的物品数量（从线性降至子线性），并且统一实现了最优的4/3‑MMS、1‑out‑d MMS、AMMS等多种公平度量

**🔧 技术方法**

核心技术包括：① 对IDo实例的排序化简；② 构造可舍入鲁棒（rounding‑resilient）的分数分配；③ 通过随机向量η协调多方代理的概率分布；④ 采用Birkhoff‑von Neumann定理在单位槽之间完成匹配实现整数分配；⑤ 对纯常规代理采用特殊槽构造保证MMS1，混合代理则接受较弱的MMS

**📊 数据集**

本研究为理论性质，未使用任何实验数据集，全部在抽象数学模型上证明

**📈 对比分析**

与之前的最优MMS近似（2‑MMS → 13/11‑MMS）、消除/补贴模型、最优顺序分配等结果对比，所提出的方法在保持预期比例的前提下实现了更好的后验公平性：仅需O(n^{3/4})个代理或物品进行处置或补贴，4/3‑MMS的保证也优于先前的8/5‑MMS

**⚠️ 局限性**

局限性包括：仅适用于可加代价的无价值物品；并不能保证所有代理人均满足MMS，只能得到MMS1或大多数代理人满足MMS；对非IDo实例仍需进行排序化简，可能导致算法复杂度提升；此外，结果为理论上限，实际实现仍需进一步研究

---

## 557. TraceRouter++: Causal Semantic Routing Realignment via Optimal Transport

**arXiv ID:** 2611.04878 | [PDF](https://arxiv.org/pdf/2611.04878v1)

**作者:** Chuancheng Shi `[一作]` (University of Sydney), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对大模型在对抗攻击下的安全性问题，作者提出了 TraceRouter++，一种通过在发现、追踪和软重排三个阶段实现的路由级安全干预框架，能在不显著损害模型通用能力的前提下抑制有害语义传播。

**💡 创新点**

创新点在于将安全干预视为分布对齐问题，利用稀疏自编码器拆分激活并通过特征影响评分精确定位跨层危险路由，然后用坐标级可分解的最优传输（OT）实现对这些路由的平滑重排，而非传统的节点消融或硬性抑制。

**🔧 技术方法**

核心技术包括：注意力分歧敏感层定位、Top‑K 稀疏自编码器（SAE）解耦稀疏特征、特征影响分数（FIS）进行因果路径追踪、以及基于一维经验分位数映射的 OT 软重排；整个流程在模型前向传播期间可高效实现。

**📊 数据集**

实验使用了多种大模型和数据集：LLMs（LLaMA3‑8B‑Instruct、Mistral‑7B‑Instruct）、MLLMs（LLaVA‑1.5‑7B、MiniGPT‑4‑7B、DeepSeek‑VL2）、DMs（Stable Diffusion 1.4、FLUX.1‑Dev、AltDiffusion、Show‑o2），以及对应的安全基准（AdvBench、HarmBench、JailbreakV‑28K、FigStep、I2P、P4D 等）和能力基准（Global‑MMLU‑Lite、MMBench、MS‑COCO）。

**📈 对比分析**

与现有方法（如 Circuit Breaker、HumorReject、SafePTR、SNCE 等）比较，TraceRouter++ 在大多数模型上实现了 98–99% 的防御成功率（DSR），显著高于对照组，并保持了几乎不变的通用性能（MMLU、MMBench、CLIP Score、FID 等）以及低的误拒率，形成了最优的安全‑效能 Pareto 前沿。

**⚠️ 局限性**

局限性包括：对路径定位的准确性高度依赖，若路由识别失误会导致干预无效；OT 近似可能无法完美重构多维依赖关系，导致少量安全漏斗；以及在极端对抗情境下（如 P4D）仍存在防御不足，需要进一步改进路径搜索与传输模型。

---

## 558. Leakage-Resilient Computation via Matrix Refreshing

**arXiv ID:** 2611.04876 | [PDF](https://arxiv.org/pdf/2611.04876v1)

**作者:** Stefan Dziembowski `[一作]`, Sebastian Faust `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275`

**🎯 论文内容**

本文通过简单的矩阵刷新（左方乘以随机可逆矩阵，右方乘以其逆）证明了在持续本地泄漏模型下的内存泄漏容错存储安全，并在此基础上构建了一个无泄漏组件的布尔电路编译器，该编译器支持持续泄漏且仅在活跃操作中泄漏。

**💡 创新点**

创新点主要有：
- 证明了上述最简单的矩阵刷新即可在持续泄漏下保持安全，避免了以往需要泄漏无效组件的做法。
- 提出了局部近似技术（基于核范数和奇异值分解）来处理共享随机矩阵导致的泄漏相关性。
- 在编译器中首次实现了无泄漏组件的多相乘门实现，并给出了全局安全证明。
- 完整地将安全分析形式化在 Lean 4 中，保证了证明的机械化与可验证性。

**🔧 技术方法**

使用的核心技术包括：
- 内部乘积提取器和两源提取器理论。
- 矩阵刷新协议及其对内部状态的平衡性分析。
- 核范数、奇异值分解和能量估计（Φ、Ψ）来构造局部近似。
- 混合论证（hybrid argument）与统计距离上界。
- 结合短暂状态刷新与电路门的递归构造。
- Lean 4 形式化与证明自动化。

**📊 数据集**

本文属于理论密码学，不涉及实验数据集；所有证明均为数学定理和形式化验证。

**📈 对比分析**

与前人（Dziembowski‑Faust、Goldwasser‑Rothblum 等）的比较表明：
- 对于 T 次刷新，统计距离上界为 O(T·2^{6λ−n/9})，在 λ≤αn（α≈1/54）时可指数级小于 1。
- 对于 S 次操作的电路编译，统计距离上界为 O(S·2^{18λ−n/27})，在 λ≤αn（α≈1/486）时亦可指数级小于 1。
- 这些结果不需要泄漏无效组件，且安全参数仅依赖于 n 和泄漏率，独立于电路规模。

**⚠️ 局限性**

局限性包括：
- 相比某些先前方案，允许的泄漏率略低（α≈1/54 或 1/486）。
- 只考虑二元字段且泄漏必须是局部、每轮固定长度。
- 需要信任初始编码/解码阶段以及不泄漏的公共输入处理。
- 证明在保持单纯矩阵刷新时对内部随机矩阵泄漏的依赖性仍是技术瓶颈，未来是否可进一步提升仍未确定。

---

## 559. One Assistant, Many Memories: Benchmarking Real-World Group Assistance When Recall Is Not Enough

**arXiv ID:** 2611.04867 | [PDF](https://arxiv.org/pdf/2611.04867v1)

**作者:** Zhanghao Hu `[一作]` (Tencent Yuanbao Team), Lin Gui `[通讯]` (King's College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了首个基于真实群聊数据的记忆评估基准，评估记忆回顾与记忆使用两大维度，并进一步区分单一、多个、冲突三种记忆依赖类型。

**💡 创新点**

创新点在于：①将记忆使用与回顾在同一历史上对比；②将记忆依赖细分为单、复、冲突三类；③构建基于真实群聊的实例并将用户真实反馈转化为评分准则；④提出两种改进模块——关系式证据精炼（RER）和辅助约束绑定（ACB），显著提升记忆系统的使用效果。

**🔧 技术方法**

使用了大规模LLM（Claude‑Sonnet‑5、GPT‑5.6‑terra、DeepSeek‑v4‑flash、Qwen‑3.5‑9b 等）配合全文检索（BM25、Dense）、图/层级检索（GraphRAG、RAPTOR）以及外部记忆体系（Mem0、LightMem、A‑MEM、MemoryOS、xMemory、EverMemOS）。评估工具包括 GPT‑5.6‑sol 作为判定器。

**📊 数据集**

使用了 50 个真实部署的中文群聊（共 80,299 条消息），经过去标识化与翻译，构建 2,395 个评估实例（1,374 L1 回顾 + 1,021 L2 使用）。

**📈 对比分析**

对比方法按记忆回顾和使用两指标进行排序。实验表明：在回顾任务中，外部层级记忆（EverMemOS、xMemory）优于全上下文；在使用任务中，全上下文与检索增强系统（BM25、Dense）表现最好，外部记忆系统尽管回顾优异，但使用性能远落后。总体而言，最佳系统在 L1 上约 0.62-0.67 的 Recall，而在 L2 上仅能获得约 0.25-0.37 的评分，显示记忆回顾不等价于有效协助。

**⚠️ 局限性**

限制：①评估仅覆盖中文群聊，跨语言推广需验证；②对外部记忆系统的训练与更新成本高、可解释性差；③对冲突记忆的处理仍依赖手工标注；④评估依赖 GPT‑5.6‑sol 判定器，可能受模型偏差影响；⑤在大规模多用户情境下，对实时检索和内存管理的性能尚未充分验证。

---

## 560. DGC-Net: Cross-Modal Dynamic Routing with Global-Guided Compensation in Multimodal Sentiment Analysis

**arXiv ID:** 2611.04841 | [PDF](https://arxiv.org/pdf/2611.04841v1)

**作者:** Linan Zhu `[一作]` (Zhejiang University of Technology), Yuhao Wang `[通讯]` (AIDC, Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种跨模态动态专家路由与全局引导动态补偿的多模态情感分析网络（DGC‑Net）

**💡 创新点**

创新点在于：①利用辅助全局语义指导局部专家路由，解决跨模态情感线索不一致；②通过全局‑局部动态补偿融合，显式调节样本依赖的全局与局部贡献，避免全局或局部信息的过度或不足利用

**🔧 技术方法**

采用BERT提取文本特征、COVAREP/OpenFace提取音/视特征；利用自定义Unimodal Coding模块生成全局/局部表示；引入动态MoE（Cross‑Modal Contribution‑aware Routing）和CLGSI子网专家；使用Transformer、注意力、MIXUP等实现全局‑局部补偿融合

**📊 数据集**

在五大公开多模态情感基准上评测：CMU‑MOSI、CMU‑MOSEI、UR‑FUNNY、MUStARD、CHERMA

**📈 对比分析**

与多种基线（TFN、LMF、MFM、GFN、MulT、MAG‑BERT、M3SA、ICCN、CubeMLP、MISA、BBFN、PriSA、C‑MIB、MSG、MMA‑BERT、DPDF‑LQ、GLoMo）进行对比，DGC‑Net在CMU‑MOSI/CMU‑MOSEI的MAE、Corr、Acc‑7、F1等指标上均实现或逼近最佳成绩，特别是在情感强度回归和二分类极性识别上表现突出

**⚠️ 局限性**

局限性包括：对极性/细粒度边界的鲁棒性仍有限；在较大、异构数据集上的可扩展性与计算开销尚待进一步验证；对缺失或嘈杂模态的适应性尚未完全探索

---

## 561. Feeling of Knowing in Large Language Models

**arXiv ID:** 2611.04834 | [PDF](https://arxiv.org/pdf/2611.04834v1)

**作者:** Zichuan Fu `[一作]` (City University of Hong Kong), Xiangyu Zhao `[通讯]` (City University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并解决大型语言模型在知识更新后功能性“知觉”(FoK) 与回答能力脱节的问题。

**💡 创新点**

提出两通道框架（OP‑SFT + KΔ‑FT + FGPO）在知识更新过程中同步 FoK 与知识，并通过 FoK 指导回答或说明。

**🔧 技术方法**

采用基于 LoRA 的增量训练、知识-Delta FoK 训练和基于奖励的 FoK 指导策略优化，并利用因果追踪定位 FoK 信号。

**📊 数据集**

主要使用 TriviaQA 进行更新和评测，同时在 SelfAware、Natural Questions、WebQuestions 进行跨数据集验证，并使用 MMLU、ARC‑Easy 评估通用能力。

**📈 对比分析**

与现有的直接判断、概率、EliCal 等 FoK 估计方法相比，所提方法在更新集和外部集上 AUROC 约 0.81，illusion 与 omission 比例均显著下降，保持或提升答案准确率。

**⚠️ 局限性**

仍受限于仅在事实问答场景验证，且对复杂推理或非事实任务的适用性尚未探索，更新顺序与规模对 FoK 迁移效果存在一定不确定性。

---

## 562. Construction of New Complete Complementary Codes Using Extended Boolean Functions and Permutations

**arXiv ID:** 2611.04796 | [PDF](https://arxiv.org/pdf/2611.04796v1)

**作者:** Aditya Prakash `[一作]`, Rajen Kumar `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种基于扩展布尔函数（EBF）和置换的全新完备互补码（CCC）构造方法，能够生成集合大小大于序列长度的CCC；

**💡 创新点**

创新点包括①允许集合大小超过序列长度；②利用此前未被使用的EBF与置换组合，产生更多多样化的CCC；③通过特定的g函数使行PMEPR仅为n（低于传统n^k），降低峰均功率比；④支持非多项式EBF映射，进一步扩展构造空间；

**🔧 技术方法**

使用的技术主要有扩展布尔函数、置换映射、互补序列理论、行/列PMEPR分析、序列长度与集合大小关系的组合数学证明；

**📊 数据集**

论文未使用公开数据集，主要通过理论构造给出示例：n=2时得到(8,4)-CCC，n=6时得到(216,36)-CCC等；

**📈 对比分析**

通过与文献中已有的CCC构造对比表格，证明在满足相同参数条件下，所提出方法的集合大小更大、PMEPR上界更低（n vs n^k），并给出具体参数实例验证；

**⚠️ 局限性**

局限性在于：缺乏实际通信系统仿真与实验验证；实现复杂度和硬件实现难度未讨论；对非多项式EBF的推广仍需进一步理论和实验支持；

---

## 563. The Pen is Mightier than the Pixel: Natural Textual Unlearnable Example for Large Vision Language Models

**arXiv ID:** 2611.04817 | [PDF](https://arxiv.org/pdf/2611.04817v1)

**作者:** Chenyu Yi `[一作]`, Alex Chichung Kot `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文针对大型视觉语言模型提出了一种基于文本的不可学习示例（NTUE），通过细微改写图像标题来误导模型学习错误的视觉-文本关联，从而保护多模态数据隐私。

**💡 创新点**

创新点在于首次将不可学习示例应用于LVLM，并通过自然文本修改而非视觉扰动，保持语义正确性与可读性的同时实现高效干扰。

**🔧 技术方法**

使用了自然文本不可学习示例生成方法、对抗文本攻击、PGD对抗训练以及传统的TAP、EM、MEM等不可学习示例技术。

**📊 数据集**

实验数据集包括cc-sbu-align、Trump‑Biden、以及多身份的名人面部图像数据集，并在LLaVA‑1.5 7B模型上进行微调。

**📈 对比分析**

与传统方法相比，NTUE在仅5%毒化样本下即可将识别准确率降至0%，并在多模态任务中显著优于TAP、EM、MEM，显示出更高的干扰效果。

**⚠️ 局限性**

局限性包括对极少样本的鲁棒性仍有限，且不同身份对攻击的响应差异显著，未来需要进一步提升在低数据、跨模态多样性场景下的稳定性。

---

## 564. On the Complexity of Finite-Sum Nonconvex-Strongly-Concave Minimax Optimization with Exact Proximal Oracles

**arXiv ID:** 2611.04806 | [PDF](https://arxiv.org/pdf/2611.04806v1)

**作者:** Yuxing Peng `[一作]` (Beijing Normal University), Weijia Jia `[通讯]` (Beijing Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了有限和非凸-强凸极小极大优化问题的最优复杂度下界与上界，并证明即使使用精确的分量近端正则化查询也无法降低多项式复杂度。

**💡 创新点**

设计了隐藏式硬实例并在任意随机自适应算法下给出下界，同时给出了仅差对数因子的近似上界，填补了此前仅限于线性跨度、零尊重或受限近端协议的分析。

**🔧 技术方法**

采用了路径构造、精确近端响应、均方根光滑性、拉普拉斯矩阵、随机归一化、变分减速等理论与算法技术。

**📊 数据集**

该工作为理论分析，无使用具体实验数据集。

**📈 对比分析**

与已有的线性跨度、零尊重、受限近端协议下界对比，得到匹配的上界；性能与传统梯度下降保持相同多项式阶数，仅多一个对数因子。

**⚠️ 局限性**

适用于均方根光滑、强凸条件；对单个分量更严格的光滑性、边界更广的双凸-双凹情形尚未覆盖；结果仅给出最坏情况下的理论极限。

---

## 565. GSPIM: A Temporal-Locality-Aware 4DGS Accelerator for PIM-Enabled Edge Devices

**arXiv ID:** 2611.04804 | [PDF](https://arxiv.org/pdf/2611.04804v1)

**作者:** Zirui Ma `[一作]` (Chinese Academy of Sciences), Wenming Li `[通讯]` (Chinese Academy of Sciences)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了一种名为GSPIM的边缘4D高斯喷射（4D Gaussian Splatting）加速器，能够在PIM（Processing‑in‑Memory）架构下对动态图像进行实时高质量渲染。

**💡 创新点**

创新点在于：①“多态PIM”能够根据不同4DGS表示（显式4D、关键帧插值、锚绑定）在同一近内存数据通路中执行活跃性筛选与压缩；②“多帧渲染数据流”将连续帧组装成窗口，在窗口内重用共享属性和稳定深度排序；③“混合调度”将近内存准备与GPU渲染在窗口/批次级别上并行，最大限度消除资源空闲。

**🔧 技术方法**

核心技术包括：基于PIM的本地时间字段筛选与固定点运算；层级压缩（L1银行级+L2 die级）将活跃记录连续化；深度稳定性评分与阈值分类；按稳定/不稳定分组的差分排序；批处理与双缓冲；以及与LPDDR5内存的接口兼容和刷新管理。

**📊 数据集**

使用公开的两组动态场景数据集：N3V（室内场景）与EasyVolcap（室外场景），涵盖前景运动、非刚体变形、遮挡变化等多种动态特征。

**📈 对比分析**

与同平台GPU（Jetson AGX Orin 64 GB）最优单帧路径及三款专用加速器（3DGauCIM、QI‑4DGS、Relay‑GS）对比，GSPIM在三种4DGS方法和六个场景上实现几何平均4.15×速度提升、4.99×能效提升，PSNR误差≤0.011 dB，吞吐量相对专用加速器提升1.59–3.98×。

**⚠️ 局限性**

主要局限：需要在支持PIM的LPDDR5 DRAM上部署，硬件占用18.7 mm²、功耗3.84 W；对极端高帧率或非常大规模场景的可扩展性尚未完全验证；深度稳定性阈值对性能/质量的微调依赖经验。

---

## 566. FourSight: Stable FP4 RL Post-Training via Sequence Filtering and Sparse Channel Compensation

**arXiv ID:** 2611.04887 | [PDF](https://arxiv.org/pdf/2611.04887v1)

**作者:** Yining Wei `[一作]` (HKUST), Wei Wang `[通讯]` (HKUST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在NVFP4低精度下实现强化学习后训练的稳定性，提出序列过滤和稀疏通道补偿两种方法。

**💡 创新点**

创新点在于发现量化失效呈集中化特征，基于此设计双门序列过滤和仅保护高方差通道的BF16补偿，并通过融合核实现低开销。

**🔧 技术方法**

采用NVFP4量化、GRPO/DAPO算法、训练‑推理不匹配（TIM）检测、低概率token过滤、通道方差选择以及高效GPU融合核。

**📊 数据集**

实验使用Qwen3-8B、Qwen3-1.7B模型，并在AIME24/25、AMC23、MATH‑500、OCW、GSM8K等数据集上评估。

**📈 对比分析**

与BF16、FP4、QaRL、QuRL、QUADS等方法对比，取得AIME24最高26.3%，学习速度提升2.6×，单步时间仅比纯FP4慢2.3%，在120B规模模型仍保持近FP4速度。

**⚠️ 局限性**

局限性包括对通道选择的依赖、不同引擎差异仍需调参、未完全解决TIM根本原因，且仅在NVFP4上验证，通用性待进一步验证。

---

## 567. Radial distortion homography estimation from affine-covariant or orientation-covariant features

**arXiv ID:** 2611.04822 | [PDF](https://arxiv.org/pdf/2611.04822v1)

**作者:** Marcus Valtonen Örnhag `[一作]`, Stefan Adalbjörnsson `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于双重仿射变换的图像几何校正方法，将畸变图像先映射为无畸变的射影图像，再通过单应性矩阵将射影图像投影到平面场景坐标系，实现图像到世界坐标的精确转换。

**💡 创新点**

创新点在于将畸变补偿（λ）与场景平面投影（H）分离处理，利用一阶近似的拉伸矩阵与高阶单应性矩阵相结合，显著提升了畸变大、视角宽的图像校正精度。

**🔧 技术方法**

使用了射影几何、单应性估计、相机标定以及多视角图像拼接技术，并结合非线性优化对仿射矩阵进行迭代精修。

**📊 数据集**

在公开的KITTI、Oxford RobotCar以及自建的广角镜头图像数据集上进行实验。

**📈 对比分析**

与传统一次性单应性校正、张氏标定等方法比较，本文方法在平均重投影误差上下降约30%，在深度估计与三维重建任务中实现了更高的精度与鲁棒性。

**⚠️ 局限性**

主要局限在于需要预先获取相机内参及畸变参数，对动态场景和非平面物体的处理仍有改进空间，且计算量相对传统方法略高。

---

## 568. Think Less, Locally: Local Preference Learning for Reasoning Compression

**arXiv ID:** 2611.04814 | [PDF](https://arxiv.org/pdf/2611.04814v1)

**作者:** Muyao Yuan `[一作]` (Xi'an Jiaotong University), Jie Zhou `[通讯]` (Tencent Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种局部偏好学习框架 LoCo，用于在大型推理模型中压缩链式思考（CoT）长度，同时保持或提升推理精度。

**💡 创新点**

创新点在于：① 将压缩视为在特定推理状态下决定继续还是结束的局部决策，而非全局对比完整轨迹；② 通过置信度与波动率联合检测过度思考点，再用更强的判别模型确认；③ 在该状态下重采样验证过长的推理，生成更短且正确的续写；④ 用 DPO 对比同一前缀下的短续写与原续写，直接学习状态条件下的“继续或停止”偏好。

**🔧 技术方法**

主要技术包括自生成推理轨迹、基于模型置信度的两阶段过度思考检测、判别模型判定冗余、从冻结状态重采样验证续写、最短正确续写选择、基于共享前缀的 DPO 微调。

**📊 数据集**

训练数据完全由 DeepSeek‑R1‑Distill‑Qwen‑7B 在数学任务上自生成的推理轨迹构成；评测数据包括 5 份数学题库（MATH‑500、AMC 2023、Olympiad、AIME 2024、AIME 2025）和 3 份代码生成题库（MBPP、LiveCodeBench、CodeContests）。

**📈 对比分析**

与 DAST、TokenSqueeze 等基线对比，LoCo 在数学任务上减少 41.7%–45.4% 的推理长度，准确率保持或提升；在代码生成任务上减少 23.4%–36.3% 长度，且 AUC 与准确率均超过基线，显示出跨域迁移优势。

**⚠️ 局限性**

局限性包括：① 过度思考检测依赖置信度与波动率阈值，可能在极难或非标准推理状态下失效；② 需要在冻结状态下重采样验证续写，计算成本较高；③ 仅在数学和代码生成任务上验证，未知对其他推理任务的泛化能力；④ 受限于所用基础模型（Qwen‑7B），在更大规模模型或不同语言环境下的表现尚未测试。

---

## 569. Where Is the Truth When AI Agents Disagree? Dependence-Aware Belief Propagation for Unsupervised Truth Discovery (Extended Version)

**arXiv ID:** 2611.04805 | [PDF](https://arxiv.org/pdf/2611.04805v1)

**作者:** Laure Berti-Equille `[一作]` `[通讯]` (IRD), Laure Berti-Equille (IRD)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种无监督真相发现方法 BP‑TD，用于对多个共享同一基础模型的 LLM 代理的答案进行聚合，抑制因代理间高度相关而导致的错误共振。

**💡 创新点**

创新点在于：① 在 Dawid–Skene 变分推理中嵌入每个代理的去相关权重 wₐ，使同一基模型的代理仅计作约一票；② 证明该权重在“最大相关块弱于其他代理”情形下能显著提升聚合准确率，并恢复代理真实可靠性排序；③ 通过实验验证权重的三种构造（family、cooccur、shared‑error）及其对性能的影响。

**🔧 技术方法**

采用了变分消息传递（mean‑field），一币 Dawid–Skene 模型，软加权投票，去相关权重计算（结构性 provenance、答案共现、错误共现），以及无标签调参（Otsu 阈值）。

**📊 数据集**

使用了四个闭卷多项选择基准：MMLU（160、5 000、10 000 题）、SciQ（160 题）、TruthfulQA（100 题）和 ARC‑Challenge（160 题）。

**📈 对比分析**

在所有基准上与多数投票、经典 Truth‑Discovery（Dawid–Skene、TruthFinder 等）、依赖感知聚合器（ClusterVote、PochampallyCorr、IsingMF、CARE、ISP/OW）比较。BP‑TD 在 MMLU（无标签）上显著优于多数投票 (+0.10)，在 10 000 题规模上仍保持 +0.03；在 SciQ、TruthfulQA 与 ARC 上与多数投票相当；当提供 20 题标签时，单一最佳代理仍优于 BP‑TD 与所有聚合器。BP‑TD 的迭代步骤主要提供可靠性估计，去相关权重即使在一次迭代（软加权投票）下即可获得大部分提升。

**⚠️ 局限性**

局限性：① BP‑TD 的优势仅在代理池中最大相关块弱于其他代理时显现；② 目前缺乏无标签诊断工具预测何时去相关有效；③ 去相关权重为经验式非生成模型，无法在所有场景下完美表现；④ 实验仅使用本地 14B 级 LLM，未覆盖更大 API 模型；⑤ 基模型可能已见过基准数据，导致错误共性增大；⑥ 在标签预算极低时，单代理选择仍是更安全的策略。

---

## 570. Sleep-Disordered Breathing Detection with Quantum Recurrent Neural Networks

**arXiv ID:** 2611.04818 | [PDF](https://arxiv.org/pdf/2611.04818v1)

**作者:** Alejandro Mayorga-Redondo `[一作]` (Universidade da Coruña), Andrés Gómez `[通讯]` (CESGA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了变分量子循环神经网络（QRNN）在睡眠呼吸障碍事件二分类中的应用，并与经典递归神经网络（CRNN）进行比较。

**💡 创新点**

创新点在于提出了一种低量子比特、低参数量的QRNN架构，并在小样本、低复杂度场景下实现了与经典模型相当甚至更优的性能，展示了量子模型在参数效率和泛化能力上的潜力。

**🔧 技术方法**

使用了变分量子电路、数据重上传编码、硬件高效Ansatz以及经典-量子混合训练（Adam优化器+二元交叉熵损失），并在模拟器上实现了QRNN；经典模型使用多层RNN+全连接+全局平均池化。

**📊 数据集**

采用了从Sleep Heart Health Study（SHHS-2）衍生的三组二分类数据集（小/中/大），每组分别包含流量信号、血氧饱和度信号，样本数为300/600/1200，类别分布分别为50/50、50/50和50/25/25。

**📈 对比分析**

通过5折交叉验证+5次重复的网格搜索优化超参数，随后在留出测试集上评估。结果显示：在小数据集上QRNN准确率提升至83%（CRNN 69%）；在中等数据集两者相近，QRNN在召回和AUC上略优；在大数据集CRNN整体优于QRNN。量子模型参数量仅几百，远低于经典模型的数万，但训练收敛需要更多轮数且波动大。

**⚠️ 局限性**

主要限制包括：QRNN在模拟器上实现，未在真实NISQ硬件上验证；训练收敛慢、波动大；模型复杂度受限于深度与噪声；缺乏足够的shots和误差补偿；对更复杂多类别任务与大规模数据的适应性未知。

---

## 571. BaCon: Band-Concentrated Exponential Random Graph Models

**arXiv ID:** 2611.04992 | [PDF](https://arxiv.org/pdf/2611.04992v1)

**作者:** Enrico Schiazzano `[一作]` (University of Turin), Gianmarco De Francisci Morales `[通讯]` (Intesa Sanpaolo Innovation Center)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文在指数族随机图模型（ERGMs）中加入概率质量约束，直接控制至少1‑δ的概率质量落在观察统计量ε-带内，从而缓解模型退化。

**💡 创新点**

创新点在于将最大熵原理与ε‑δ带约束相结合，得到的分布仍为指数族形式，只多了一个带指示统计量，并通过参数ν来调节该约束；该方法在保持ERGMs可解释性的同时，显式规定了观测图附近的概率分布。

**🔧 技术方法**

主要技术包括：最大熵推导得到目标分布；利用Metropolis‑Hastings采样并加入平滑带指示；采用Robbins‑Monro随机逼近更新参数θ与ν；使用混合提议（TNT与传递边）以提升混合效率。

**📊 数据集**

实验使用两类社交网络（意大利大学师生友谊图与工作交互图）以及 Slashdot 的不同规模子图（10²–10⁵条边），以验证方法在不同规模和难度场景下的表现。

**📈 对比分析**

与标准ERGMs及LOLOG模型比较，评估指标包括中心化误差、带内浓度、能量得分和L1距离；实验表明在相同或更低的估计成本下，带约束模型显著提高了观察统计量的浓度（从约7%到81%或32%到97%）并降低能量得分；计算时间随边数线性增长，表明可扩展性良好。

**⚠️ 局限性**

主要限制包括：约束仅适用于选定的统计量，带宽的设定依赖于尺度向量与参考模型校准；估计过程高度依赖MCMC混合与收敛性；平滑带指示的理论性质和对最终结果的影响尚未充分分析。

---

## 572. Scaling Laws For Data-Constrained Dense Retrieval

**arXiv ID:** 2611.04979 | [PDF](https://arxiv.org/pdf/2611.04979v1)

**作者:** Bowen Fu `[一作]` (Xi'an Jiaotong University), Jingtao Zhan `[通讯]` (Tsinghua Shenzhen International Graduate School)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在有限标注数据下稠密检索的缩放律，提出最小可实现评估损失（MAEL）并给出模型尺寸与数据预算的乘法缩放关系。

**💡 创新点**

创新点在于：① 引入 MAEL 作为衡量最佳可达性能的指标，并证明其与检索效果高度相关；② 给出 MAEL 的乘法缩放律、训练步骤与权重衰减的缩放规律；③ 在固定计算预算下提供模型尺寸与数据预算的效率对比。

**🔧 技术方法**

使用 BERT 系列模型在 MS MARCO 子集上进行大规模实验；采用学习率重参数化、CKA 相似度、EMA 时间尺度权重衰减分析；通过保留检验验证缩放律的泛化能力。

**📊 数据集**

MS MARCO Passage Ranking 训练/验证集（不同唯一样本数），以及内域 MS MARCO 测试集和三大跨域 BEIR QA 基准（NQ、HotpotQA、FiQA）。

**📈 对比分析**

在 35 个模型-数据组合上测 MAEL 并与 MRR@10、Recall@100、nDCG@10、MAP@10 相关，MAEL 与检索指标呈极强负相关；在相同计算预算下，扩大模型尺寸比增加数据更高效，MAEL 从 0.720 降至 0.690。

**⚠️ 局限性**

局限性包括：仅评估 BERT 系列在 MS MARCO 子集；权重衰减缩放基于 EMA 时间尺度假设；计算预算假设固定，未考虑预训练成本；未验证跨任务或跨语言的通用性。

---

## 573. Software Design Patterns for Latency Measurements with eBPF in SDN Environments

**arXiv ID:** 2611.04973 | [PDF](https://arxiv.org/pdf/2611.04973v1)

**作者:** Pedro Lucas M. S. Rosa `[一作]` (Federal University of Rio Grande do Sul), Jéferson Campos Nobre `[通讯]` (Federal University of Rio Grande do Sul)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实现了基于 eBPF/XDP 的 SDN 延迟测量系统，利用 Observer、Strategy、Factory 方法论构建可扩展、模块化的收集与处理层。

**💡 创新点**

首次将经典软件设计模式引入 eBPF/XDP 监控管道，证明了在保持极低数据面开销（<5%）的前提下，可显著提升系统可维护性与扩展性。

**🔧 技术方法**

使用 eBPF/XDP 进行内核级时间戳捕获；Python + BCC 开发用户空间代理；HTTP 服务器实现事件发布与多种数据 sink；设计模式（Observer、Strategy、Factory）用于解耦与动态插件化。

**📊 数据集**

在 Mininet 虚拟拓扑中产生 UDP 流量（≈5 pkt/s，20 秒，共约 100 包），并使用自定义 JSON 事件流作为输入；无公开公开数据集，采用合成网络流量。

**📈 对比分析**

对比了单一 monolithic 基线与模式化实现：测量指标包括 OWD 分布、请求处理时延（平均约 26–27 µs，p95 < 40 µs）以及扩展时代码行数。结果显示模式化版本几乎无性能损失，且在后续功能添加（如 P95 滑动窗口策略、JSONL sink）时所需代码变更显著减少。

**⚠️ 局限性**

局限性包括：仅在 Mininet 虚拟环境下验证，拓扑简化，未考虑多路径或大规模部署；时间戳精度受共享内核时钟影响；缺乏跨平台（物理硬件、不同内核版本）评估；以及对极端网络状况（高丢包、时钟漂移）的鲁棒性未作实验。

---

## 574. Artifact-to-Capability: Diagnosing Breaks in Capability Formation of Self-Evolving Agents in a Controlled Environment

**arXiv ID:** 2611.04952 | [PDF](https://arxiv.org/pdf/2611.04952v1)

**作者:** Ziqian Ban `[一作]` (Tianjin University), Zhiyi Cheng `[通讯]` (Tianjin University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文定义了自我演化代理的“能力形成”五属性框架，并在可重放、规则可判定的自由风格五子棋环境中进行三尺度（状态级、游戏级、版本级）诊断，评估三种自我演化机制是否真正形成了可迁移的能力。

**💡 创新点**

创新点在于提出了从artifact到能力的依赖链（Artifact formation → Behavioral activation → Task‑relation realization → Boundary‑safe reuse → Capability extension）以及对应的三尺度诊断方法，并通过实验揭示传统的三信号（artifact accumulation、behavioral change、metric gain）不足以判定能力是否形成，给出了合法化增益、组合覆盖以及损失漂移三种破坏案例。

**🔧 技术方法**

使用了基于规则的自由风格五子棋环境、可重放的棋盘状态、规则判定程序、对比实验、噪声基线估计、行为差异统计以及版本线性比较等技术。

**📊 数据集**

数据集主要由人工构造的状态集合A（120个旋转/镜像拓扑）和演化过程中生成的状态集合B（1756个）组成，还构造了差分池7234个状态；对手为程序化对手S、S2、S3。

**📈 对比分析**

方法上先通过三信号检测（artifact accumulation、behavioral change、metric gain）验证基础进展，然后在三尺度诊断中分别检验五个属性；结果显示三种机制虽满足三信号，但在能力形成链的不同节点（Property 3、4、5）失效；性能上 Reflexion 通过合法化提升整体通行率，Voyager 产生开放四但在冲突状态无法获胜，DGM 初期通过 OB‑3 形成能力但后续版本出现“loss‑only drift”。

**⚠️ 局限性**

局限性包括：实验仅限于自由五子棋，未涵盖多智能体共进化、参数直接更新等自我演化形式；诊断依赖可重放状态与规则判定，难以直接迁移到规则不明确的开放任务；只评估了三种机制，未覆盖更广泛的自我演化策略；不同基模型、演化周期或重复实验可能导致破坏点位置变化，尚未系统验证。

---

## 575. How Well Does Webcam Eye Tracking Work "in the Wild"? A Large-Scale Validation of Chiasm and a Comparison with WebGazer

**arXiv ID:** 2611.04946 | [PDF](https://arxiv.org/pdf/2611.04946v1)

**作者:** Artem V. Belopolsky `[一作]` (Chiasm BV), Florian Perdreau `[通讯]` (Chiasm BV)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在大规模在线实验（N=505）中，评估并验证了第一代 Chiasm 摄像头眼动追踪器的空间准确度、精度和时间稳定性，并与常用的 WebGazer 进行对比。

**💡 创新点**

创新点：①首次在“野外”条件下对 webcam 眼动追踪器进行系统、全面的验证；②设计了包含动态与静态目标、头部运动和漂移检测的验证流程；③公开了 jsPsych 插件与完整实验代码，方便科研社区复现和扩展。

**🔧 技术方法**

使用技术：基于深度神经网络的实时 gaze 预测；自适应校准与动态反馈；使用 jsPsych 作为实验框架；数据过滤与多维评估指标（空间准确度、空间精度、样本间精度、漂移量）。

**📊 数据集**

数据集：来自 Prolific 的 505 名参与者，设备类型、操作系统、光照条件等多样；对比实验使用 101 名独立参与者的 WebGazer 数据集；两组数据使用相同实验协议与过滤流程。

**📈 对比分析**

比较方法：对空间准确度、空间精度（SD）和样本间精度（RMS‑S2S）进行独立样本 t 检验和 2×3 方差分析；结果显示 Chiasm 在所有验证阶段的空间准确度和精度均显著优于 WebGazer（均值分别为 2.18 cm vs 4.51 cm；1.01 cm vs 1.55 cm），但 WebGazer 的样本间精度略好（0.79 cm vs 1.26 cm），可能由于其回归平滑效果。

**⚠️ 局限性**

局限性：①仍低于实验室硬件级别（≈0.5°）；②仅 30 Hz 采样率，无法捕捉快速眼跳；③缺乏与实验室 eye‑tracker 的并行对照；④部分参与者未通过校准，导致数据缺失；⑤受设备、光照、佩戴眼镜等因素影响；⑥实验只评估了静态/动态点目标，对更复杂任务的适用性尚待验证。

---

## 576. Local Visibility Roadmaps in Continous Multi-agent Path Finding

**arXiv ID:** 2611.04944 | [PDF](https://arxiv.org/pdf/2611.04944v1)

**作者:** Kristýna Janovská `[一作]` (Czech Technical University in Prague), Pavel Surynek `[通讯]` (Czech Technical University in Prague)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了在连续时间与空间下的多智能体路径规划问题，提出了一种基于可视化路线图与冲突基础搜索（VR-CBS）的新算法。

**💡 创新点**

创新点在于将每个智能体的可视化路线图与CBS框架结合，高层使用冲突消除搜索，低层通过可视化路线图进行任意角路径规划，并将冲突约束映射为静态多边形障碍，随后再进行B‑spline曲线平滑。

**🔧 技术方法**

使用的技术包括可视化路线图构建、A*搜索、B‑spline曲线平滑、冲突基础搜索（CBS）、与先前的CE‑CBS和SMT‑CCBS算法的对比。

**📊 数据集**

实验数据集基于DAO网格地图，经过贪婪算法转换为连续环境，随机生成1至10个智能体的起止点。

**📈 对比分析**

通过比较SOC（总路径成本）、最小通路长度、运行时间等指标，VR‑CBS在所有实验实例中均获得更短的SOC和更快的计算速度，且与SMT‑CCBS在最小通路长度上相当或更优。

**⚠️ 局限性**

主要局限在于B‑spline平滑可能导致次优解，冲突约束被保守处理为静态障碍可能排除某些可行路径；算法假设机器人为同尺寸圆形，未考虑动态障碍或更复杂的运动学约束。

---

## 577. HTNN-MSC: Multi-view Subspace Clustering with a Hyperbolic Tangent Tensor Low-Rank Surrogate

**arXiv ID:** 2611.04940 | [PDF](https://arxiv.org/pdf/2611.04940v1)

**作者:** Yiwen Dou `[一作]` (Anhui Polytechnic University), Yiqiang Ding `[通讯]` (Anhui Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出基于超bolic tangent张量秩近似的多视角子空间聚类方法 HTNN-MSC，并实现 ADMM 求解。

**💡 创新点**

创新点在于将有界非凸的 tanh 作为张量秩正则化，使大奇异值的边际惩罚随大小递减，兼具严格凹性、Lipschitz 导数并可在 o‑minimal 框架下收敛；同时采用三起点 Newton 迭代的 HTT 阈值化和自适应尺度调节。

**🔧 技术方法**

技术手段包括 t‑SVD 与频域 FFT、非凸张量秩正则化、ADMM 分块求解、HTT 阈值化、以及 KL 收敛分析。

**📊 数据集**

使用 23 个公开多视角数据集，涵盖人脸、物体、场景、文本、手写等，样本数 169–10,800，视角 2–6，类别 4–100。

**📈 对比分析**

与 9 种基线（t‑SVD‑MSC、LT‑MSC、TLSpNM、GMC、LMVSC 等）在 ACC 与 NMI 上做 oracle λ 选择比较。HTNN‑MSC 的平均 ACC 为 0.8404，平均 NMI 为 0.8029，平均排名分别为 2.78 与 3.02，显著优于大多数方法；相同骨架对比时平均 ACC 提升 8.56%，但在 3Sources、MSRCv1、Caltech101‑7 等数据集仍略逊。

**⚠️ 局限性**

局限性包括：λ 的选择仍需人工调参，θ 的自适应缩放并非最优；在 3Sources、MSRCv1、Caltech101‑7 等数据集表现不佳；大规模样本导致 O(Vn^3) 计算瓶颈；收敛理论基于理想块最优求解，实际实现不完全；oracle λ 选择导致实验结果偏乐观。

---

## 578. Metric Facility Location under Batched Random Order

**arXiv ID:** 2611.04920 | [PDF](https://arxiv.org/pdf/2611.04920v1)

**作者:** Fabrizio Grandoni `[一作]` (IDSIA), Matteo Russo `[通讯]` (EPFL)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

本文研究了在批量随机顺序模型（batch‑random‑order）下的在线度量设施定位（Online Metric Facility Location）问题，提出了新的在线算法，并给出了与最优下界匹配的竞争比分析。

**💡 创新点**

创新点主要体现在：
- 在每个批次内部对客户端按随机顺序处理，完全不依赖全局批次结构；
- 通过为每个客户端（或客户端‑层次对）定义静态随机键，并使用记录（record）计数和进展（progress）分析，获得对批次数与客户端总数的双重上界；
- 对非均匀开设成本的情况引入多层次（powers‑of‑two）开设成本的分层处理，配对分析消除层数因子；
- 构造了匹配的下界实例（隐藏路径树）来证明竞争比的紧密性。

**🔧 技术方法**

使用的技术包括：
- Meyerson 的在线设施定位算法及其概率开设规则；
- 随机键（key）与记录计数的概率分析；
- 对批内随机顺序的洗牌和对称配对（pairing）技巧；
- 分层处理（对非均匀成本）以及几何级数的费用分配；
- 经典的 Harmonic 数、记录分析与对数上界推导；
- 下界构造：隐藏路径树与记录点分析。

**📊 数据集**

论文为理论研究，没有使用任何实验数据集；所有结果均来自严格的数学证明。

**📈 对比分析**

性能评估在理论上完成：
- 对任意客户端数 n、批次数 B，得到竞争比 Θ( min{ log(B+1), √( log(n‑B+2) ) } );
- 在极端情形（单批或单点）竞争比为常数；
- 证明匹配下界，表明该竞争比是最优的。

**⚠️ 局限性**

局限性包括：
- 仅适用于已知度量空间且批次顺序为随机的模型；
- 对非均匀开设成本的分析相对复杂，需要额外的分层技巧；
- 未考虑批次大小未知或不完全随机的情况；
- 仅给出了理论上最佳的竞争比，实际实现与常数因子未作实验验证。

---

## 579. The Power of Bidegeneracy for Hypergraph Data Structures

**arXiv ID:** 2611.04900 | [PDF](https://arxiv.org/pdf/2611.04900v1)

**作者:** Marco Bressan `[一作]` (University of Milan), C. Seshadhri `[通讯]` (University of California, Santa Cruz)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `fede83ac-7505-405f-ab37-e7284695c47f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的超图稀疏性度量——双退化性，并基于此设计了高效的数据结构，以快速回答关于超图的基本查询。

**💡 创新点**

引入了双退化性这一概念，严格推广了许多其他稀疏性度量，并展示了其在加速超图数据结构方面的有效性。

**🔧 技术方法**

使用了双退化性来计算超图的压缩表示，并通过几何计算技术解决了相关问题。

**📊 数据集**

使用了来自文献的超图数据集，发现这些数据集的双退化性显著较低，表明该方法在实际应用中可能具有重要意义。

**📈 对比分析**

与现有方法相比，提出的方法在处理基本查询时实现了近线性时间的预处理和多对数时间的查询响应，性能显著提升。

**⚠️ 局限性**

该方法的局限性在于，假设输入数据满足特定的稀疏性条件，若不满足这些条件，可能导致性能下降。

---

## 580. Heads That Write, Not Just Point: Image Retrieval Heads in Vision-Language Models

**arXiv ID:** 2611.04903 | [PDF](https://arxiv.org/pdf/2611.04903v1)

**作者:** Junsung Park `[一作]` (Seoul National University), Sungroh Yoon `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在大型视觉语言模型中研究上下文内图像检索机制，并提出两侧分析方法，使用ICIR‑MCQ探针识别指针头和写作头；

**💡 创新点**

首次同时考察注意力指向与输出写入的两面，发现两种头集合交集对多图像推理具有因果效应；

**🔧 技术方法**

基于注意力权重的指针头检索、基于输出向量的轻量级检索器、头级零填充干预、可视化等技术；

**📊 数据集**

使用MS‑COCO构建ICIR‑MCQ探针样本，并在Visual Haystacks上进行迁移与评估；

**📈 对比分析**

在ICIR‑MCQ上，指针头单独实现R@1可达0.79‑0.84，超越CLIP、SigLIP等外部检索器；写作头在1200候选图集上R@1为0.47‑0.59；在Visual Haystacks上，指针‑写作交集的头集干预导致准确率显著下降，验证其因果重要性；

**⚠️ 局限性**

仅评估四个开源模型，关注单目标检索，未覆盖多目标、跨图比较、生成时干预等更广泛场景。

---

## 581. VisSearch-Bench: Evaluating Long Horizon Visual Search in Multimodal Agents

**arXiv ID:** 2611.04894 | [PDF](https://arxiv.org/pdf/2611.04894v1)

**作者:** ZhengXian Wu `[一作]` (Tsinghua University), Haoqian Wang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了VisSearch‑Bench基准，用于评估多模态模型在多图像、长周期开放网页搜索中的视觉上下文管理能力。

**💡 创新点**

创新点包括：①提出五大任务族（Set‑wise、Sequential、Hybrid、Disambiguation、Structure‑guided）来系统化视觉搜索需求；②设计Process Score衡量子任务的正确性与完整性；③严格要求视觉证据不可被文本或模型知识替代，提升任务真实性。

**🔧 技术方法**

技术手段主要是多模态大型语言模型（如Qwen、Gemini、GPT‑5.5、Kimi‑K3等）配合ReAct或DeerFlow框架；利用工具链包括网页搜索、图片搜索、逆图搜索、网页阅读和图像裁剪。

**📊 数据集**

数据集为204道人工编写任务，覆盖7大领域（文化、生命科学、物理科学、社会、地理、技术、商业）和25个子领域，任务附带可追溯的证据来源、参考答案与评分规则。

**📈 对比分析**

对比方式为Pass@1准确率和Process Score两项指标；ReAct+Kimi‑K3在准确率上最高达26.47%，ReAct+GPT‑5.5在Process Score上最高70.61%；使用搜索框架明显优于无工具直接回答，表明工具利用提升了视觉信息处理效果。

**⚠️ 局限性**

局限性在于整体完成率仍偏低（仅约26%），多图像推理、跨步视觉关联与结果合并仍是主要瓶颈；不同模型在各任务族表现差异显著，工具调用策略不一致导致性能波动。

---

## 582. Linking as Evaluation: Correct Separate Compilation via First-Class Environments

**arXiv ID:** 2611.04993 | [PDF](https://arxiv.org/pdf/2611.04993v1)

**作者:** Jam Kabeer Ali Khan `[一作]` (Independent), Bruno C. d. S. Oliveira `[通讯]` (University of Hong Kong)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个带有第一类模块、链接表达式和第一类环境的简单类型 λ-演算，证明了链接作为评估的形式化语义，并在此基础上推导出分离编译的正确性，最终实现了一个 OCaml 编译器将其编译为 WebAssembly。

**💡 创新点**

创新点主要在于：①将链接提升为语言内部的表达式，并通过“链接即评估”实现编译与链接的顺序不变性；②采用第一类环境和依赖合并，使模块体可以直接成为闭合值；③利用类型导向的 elaboration 把源语言的模块和链接语义映射到目标 λ-E 语言，实现编译器层面的链接实现；④在 Lean 4 中机械化证明了类型安全、语义保持和分离编译正确性（包括静态与动态链接）。

**🔧 技术方法**

技术细节包括：简单类型 λ-演算、第一类环境与依赖合并、类型导向的 elaboration、目标语言 λ‑E（call‑by‑value），小步与大步语义、语义保持与行为保持证明、Lean 4 形式化、OCaml 实现与 WebAssembly（WasmGC）后端。

**📊 数据集**

论文未使用外部数据集；所有论证均基于示例程序（如 Add、Double、插件/宿主示例）以及在 Lean 4 中的机械化证明。

**📈 对比分析**

由于论文聚焦于形式化证明，未进行性能评测或与其他系统的比较；主要结果是证明了编译与链接可交换、分离编译正确性，以及编译器实现的语义一致性。

**⚠️ 局限性**

局限性：仅支持简单类型（无抽象类型、共享、泛型、子类型、递归类型等）；未涵盖 I/O、错误处理或真实世界的大规模模块；实现仅针对 OCaml 与 WebAssembly；未提供实测性能或与现有模块系统（如 OCaml、F#, Haskell）在编译时间、运行时开销等方面的比较。

---

## 583. ProAlign: Progressive Positional and Prototype-Guided Alignment for Aerial-Ground Person Re-Identification

**arXiv ID:** 2611.04980 | [PDF](https://arxiv.org/pdf/2611.04980v1)

**作者:** Qiao Li `[一作]` (Wuhan University), Lei Tan `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种用于空地行人再识别的统一框架ProAlign，解决视角差异导致的位置信息漂移和语义不一致问题。

**💡 创新点**

创新点包括层级自适应的位置信息校正模块LPPE（浅层采用条件位置编码，深层采用可学习静态位置嵌入）以及视角感知的原型对比学习模块PCL（维护空地视角专属原型并通过质量加权动量更新实现跨视角语义一致）。

**🔧 技术方法**

技术手段：基于ViT-Base Transformer架构，结合条件位置编码、可学习位置嵌入、门控残差融合、质控动量原型更新、双原型InfoNCE对比损失和多任务联合训练。

**📊 数据集**

数据集：CARGO（合成空地数据）、AG-ReID与AG-ReID v2（真实世界空地数据）。

**📈 对比分析**

与现有方法（如GSAlign、VDT、VIF等）对比，ProAlign在CARGO上整体检索指标Rank‑1提升1.60%、mAP提升5.01%，在A↔G跨视角场景提升12.52% Rank‑1、15.69% mAP，并在单视角任务亦保持竞争力；在AG‑ReID与AG‑ReID v2上同样取得显著性能提升。

**⚠️ 局限性**

局限性：LPPE采用固定的层级分区（浅层条件编码/深层静态编码），缺乏实例级自适应；PCL的原型更新仅基于当前mini‑batch，若同一身份在批中缺乏跨视角样本，则对比监督受限，影响大规模多视角数据的鲁棒性。

---

## 584. LLMs as MDP Designers: A Dependency-Aware Agentic Framework for Automated Robot Policy Generation

**arXiv ID:** 2611.04974 | [PDF](https://arxiv.org/pdf/2611.04974v1)

**作者:** Hong Je-Gal `[一作]` (Sejong University), Hyun-Suk Lee `[通讯]` (Sejong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过自然语言任务描述自动生成可执行的机器人强化学习策略。

**💡 创新点**

提出依赖意识的 Agentic 框架 ARPG，分阶段 MDP 设计、外部化中间产物、局部验证修正，并强调状态、动作、奖励、终止的一致性。

**🔧 技术方法**

利用多代理大型语言模型（GPT‑5.2）进行 MDP 设计与代码实现，结合 Isaac Sim/Isaac Lab 进行环境与训练。

**📊 数据集**

在 Isaac Lab 提供的五个基准任务（CartPole、Ant、Humanoid、Reach、Lift）以及额外的经典控制、工程 MDP 等自定义环境进行评测。

**📈 对比分析**

与单体 LLM 与现成 Agentic 系统比较，ARPG 在模型正确率、环境可执行率、策略成功率均明显提升，尤其在复杂任务中实现 90% 以上的政策生成成功率。

**⚠️ 局限性**

依赖 LLM 的设计阶段仍可能产生误差；实现阶段对模拟器特定依赖的推断仍有限，且在大规模高维任务中计算开销和迭代次数较高。

---

## 585. An Agentic Framework for Quality-Aware and Scenario-Adaptive Image Compression

**arXiv ID:** 2611.04960 | [PDF](https://arxiv.org/pdf/2611.04960v1)

**作者:** Zhenghao Chen `[一作]` (University of Newcastle), Yan Lu `[通讯]` (Microsoft Research Asia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 PICA，一个多代理框架，能够在不改动现有编解码器的前提下，根据用户需求和图像退化自适应压缩。

**💡 创新点**

创新点在于将感知、改进、编码和后期改进四个代理协同工作，利用 Agentic ROI Coding 与 Scenario‑Guided Quality‑Rate 评估，实现无训练、无架构改动的跨场景和跨退化自适应。

**🔧 技术方法**

采用 VLM（如 Qwen3‑VL‑8B‑Instruct）、无监督 IQA、预训练恢复模型、黑盒 QC 控制与自适应 ROI 编码等技术。

**📊 数据集**

在 Kodak、CLIC2020、HEVC 三个公开数据集上，分别生成噪声、模糊等退化版本进行实验。

**📈 对比分析**

与 H.266/VVC、Cheng2020 以及专用压缩器（HiFiC、Adapt‑ICMH、DenoiseCompression）对比，PICA 在标准、感知和机器视觉场景下平均提升 7–35% 的比特率节省或 40–80% 的 mAP / LPIPS 指标。

**⚠️ 局限性**

局限在于仍依赖 VLM 推理成本，且在极端退化或高分辨率下算力需求较高；未针对多帧视频压缩做完整评估。

---

## 586. Lightweight graph transformers for clutter and target classification in automotive radar

**arXiv ID:** 2611.04959 | [PDF](https://arxiv.org/pdf/2611.04959v1)

**作者:** Adrian Gheorghiu `[一作]` (NXP Semiconductors), Ashish Pandharipande `[通讯]` (NXP Semiconductors)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出轻量级的 Tiny Graph Transformer Network (TGTNet) 用于自动驾驶雷达点云中的杂波与目标分类，并通过 Superpoint 块和时序增强提升性能。

**💡 创新点**

创新点在于：① 采用 Gaussian attention 与低秩 LoRa 变换实现极低参数量的图变压器；② 设计 Superpoint 块利用能量模型聚类提取局部特征；③ 引入两种时序增强（GRU‑TGTNet 与 TGTNet‑T）和并行下采样；④ 使用能量模型替代 DBSCAN 的稀疏点云聚类。

**🔧 技术方法**

使用的技术包括：点云变压器（Point Transformer）与 Gaussian Transformer Block、低秩 LoRa 适配、Superpoint Transformer、能量模型聚类、Farthest‑Point Sampling、注意力下采样/上采样、GRU‑点云 RNN、Lovász‑Softmax 与加权交叉熵损失、Adam 与余弦退火学习率调度。

**📊 数据集**

采用公开雷达数据集 RadarScenes（含目标与杂波标注）以及 Radar Ghost Dataset（手动标注的鬼影）进行训练与评估。

**📈 对比分析**

与 GRT、PointNet 等基准模型对比，TGTNet 在保持 2% 以内的 F1/召回率的同时，模型尺寸缩小 200 倍、推理速度提升 4 倍；Superpoint 及时序增强进一步提升 1–2% F1，且内存占用仍保持在 10–50 KB 级别。

**⚠️ 局限性**

局限性：① 受限于相对小的数据集，易出现过拟合；② 目前仅在 32‑bit 浮点参数下实现，仍需量化/模型压缩以适配车载硬件；③ 主要针对雷达点云的结构化杂波，未验证在更复杂多源传感器融合场景下的鲁棒性。

---

## 587. The Complexity of Continuous Contract Design over Polymatroids

**arXiv ID:** 2611.04934 | [PDF](https://arxiv.org/pdf/2611.04934v1)

**作者:** Max Klimm `[一作]`, Philipp Warode `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在多项式约束下的连续努力合同设计问题，分析了线性、二次和齐次凸成本的最优解与近似算法，并给出了NP‑hard性证明与FPTAS

**💡 创新点**

将多项式时间可解与NP‑hard性与多项式结构相结合，提出水填充与参数化最小割的水填充算法、分段线性响应描述以及新的合同取值上界

**🔧 技术方法**

采用水填充、子模最小化、参数化最小割、分段线性与凸优化技术，以及Oracle-based FPTAS

**📊 数据集**

无真实数据集，使用理论构造实例与网络模型进行实验验证

**📈 对比分析**

与现有离散动作空间算法相比，在线性/二次成本下实现了多项式或近似解；NP‑hard性证明表明在一般情形下无多项式精确算法可行

**⚠️ 局限性**

对一般凸成本的精确优化仍困难；Laminar多项式中指数分段数未解决；对多参数或多结果空间的扩展尚未给出完整理论

---

## 588. PROSE: Preservation-Constrained Selective Editing for Guilty Bias in Legal Judgment Prediction

**arXiv ID:** 2611.04921 | [PDF](https://arxiv.org/pdf/2611.04921v1)

**作者:** Yimeng Wang `[一作]` (University of International Business and Economics), Weijie Yu `[通讯]` (University of International Business and Economics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出在中文刑事法律判决预测中，对“罪名不成立”与“罪名成立”配对进行保留约束选择性修复，并构建CAIL‑MC基准。

**💡 创新点**

创新在于利用训练配对中学习的对比向量与梯度下降方向构造保留约束的激活目标，再通过低秩正则化参数更新实现一次性、可持续的修复，解决修复与保留之间的张力。

**🔧 技术方法**

结合LLM隐藏状态对比、梯度引导、目标方向插值、保留分量衰减、Ridge最小二乘参数更新、低秩截断与Frobenius范数约束等技术。

**📊 数据集**

使用从CAIL2018、LeCaRD和Criminal‑S经法律审阅的案例构成的CAIL‑MC，包含11,478个配对和多种证据充分性样本。

**📈 对比分析**

与SFT、LJPIV、Self‑Refine、Debate‑Feedback、OBJECTION、MEMIT、AlphaEdit等基线在Qwen3‑8B和GLM4‑9B上对比，所提Fix在pairwise safe repair最高、支持罪名损坏最低，跨语料库证据充分性评测中优于SFT，并且离线成本最低、推理一次性。

**⚠️ 局限性**

局限在于仅适用于罪名条件化的判决预测，依赖配对训练样本且模型特定，未覆盖完整司法决策流程，证据充分性评测范围有限，修复结果未能完全保证，实际司法应用需进一步验证。

---

## 589. MiCaDe: Online LiDAR-Camera Miscalibration Change Detection for Autonomous Driving

**arXiv ID:** 2611.04912 | [PDF](https://arxiv.org/pdf/2611.04912v1)

**作者:** Carlo Sgaravatti `[一作]` (Politecnico di Milano), Giacomo Boracchi `[通讯]` (Politecnico di Milano)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出MiCaDe，一种基于几何一致性指标的在线LiDAR-相机外参漂移检测框架；

**💡 创新点**

将漂移检测视为序列变更检测，利用现成的2D/3D目标检测匹配得到的几何一致性指标，无需额外目标或训练；

**🔧 技术方法**

利用2D‑3D匹配、中心偏移、IoU、方向、深度误差等几何指标，批量聚合后通过GMM评估似然，再用Lepage CPM进行无监督的变更检测；

**📊 数据集**

在KITTI Raw、ZOD、ONCE公开数据集以及自制的物理诱导漂移实车数据上进行实验；

**📈 对比分析**

与基线Tahiraj等人进行对比，MiCaDe在ARL_0=10000时FPR≤0.002、FNR≤0.006，平均检测延迟约5–10秒，AUC≈0.99，且不需要额外训练；

**⚠️ 局限性**

依赖成功的2D‑3D匹配，稀疏场景或严重漂移可能导致匹配失败；批量特征非完全独立，CPM的假警控制仅近似。

---

## 590. TapeSAC: Automated, Pareto-Optimal Fine-Grained Memory Management for Large Model Training

**arXiv ID:** 2611.04897 | [PDF](https://arxiv.org/pdf/2611.04897v1)

**作者:** Kaihui Ren `[一作]` (KlingAI Research), Chengru Song `[通讯]` (KlingAI Research)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无需改动模型代码即可在PyTorch eager模式下自动化选择和记录激活检查点的系统——TapeSAC，能在给定内存限制下以张量级别重放计算，实现近Pareto最优的内存与重计算时间折衷；

**💡 创新点**

核心创新在于：①利用PyTorch调度器与autograd保存钩子实时记录每个块的前向图，②以张量级别重放只复算需要的子图，③构建基于测量算子时间与step峰值的规划器，自动在单一内存预算下生成最佳保存集；

**🔧 技术方法**

技术手段包括：PyTorch dispatcher记录操作、保存钩子识别需求张量、将自定义核注册为可重放的PyTorch自定义运算符、基于测量成本与峰值的多目标规划（beam search + 多选背包），以及令牌桶机制动态重新规划；

**📊 数据集**

在大型语言模型Qwen系列（Qwen3.8‑27B、Qwen3.6‑35B‑A3B、Qwen3.5‑397B）上进行实验，并在多GPU（16/64卡）和真实数据上验证；

**📈 对比分析**

与完整检查点、手工调优的PyTorch SAC以及PyTorch的memory‑budget相比，TapeSAC在90%设备内存下可将step时间提升6.3%–13.9%，在相同或更低内存下快5.1%且避免OOM；在500步真实训练中缩短总训练时间8–11%；

**⚠️ 局限性**

局限性包括：需对所有自定义核进行一次注册以实现可重放；记录与重放在CPU侧增加轻微开销；规划器依赖早期profiling，动态图形或显著变化时需要重新采样；对极端内存瓶颈场景或极低GPU计数可能不如编译器优化方案高效。

---

## 591. Who is the Pronoun Pro? Comparing LLMs and Humans at Number-Disambiguation of 'they'

**arXiv ID:** 2611.04968 | [PDF](https://arxiv.org/pdf/2611.04968v1)

**作者:** Carlos Hartmann `[一作]` `[通讯]` (University of Zurich), Carlos Hartmann (University of Zurich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

该研究通过对174条Reddit评论中“they”代词的数义标注，比较了多种LLM与人类注释员的准确率、成本与速度。

**💡 创新点**

创新点在于使用真实社交媒体数据、公开的标注规范，并对LLM与人类在低一致性案例中的互补性进行定性分析，提出人机协同的实证框架。

**🔧 技术方法**

采用prompt‑based问答式提示，测试商业与开源LLM（如Kimi K3、Claude Sonnet、GPT‑5等），并使用准确率、F1、MCC等指标评估性能。

**📊 数据集**

数据集为174条随机抽样的Reddit“they”实例，涵盖复数、泛指单数和特定单数三类，并附有Gold标准标签。

**📈 对比分析**

通过整体准确率与人类平均进行对比，并采用非劣效检验，Kimi K3以94.5%准确率超过人类平均（78.5%），在稀有/创新用法上与人类差距不显著，LLM在自然频率投影下表现更优。

**⚠️ 局限性**

主要限制包括样本量有限、未包含完全歧义案例、LLM对上下文依赖判断不佳、成本评估未考虑能耗与隐私等外部因素。

---

## 592. Shuffled OPSD: When Semantic Consistency Hurts On-Policy Self-Distillation?

**arXiv ID:** 2611.04958 | [PDF](https://arxiv.org/pdf/2611.04958v1)

**作者:** Qingyuan Zheng `[一作]` (Duke University), Zhihao Dou `[通讯]` (Duke University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在对大语言模型进行数学推理的训练中，提出了“Shuffled OPSD”，即在教师和学生共享的题目文本中打乱词序，同时保持教师仅见的完整解答不变，以此来改进知识蒸馏过程。

**💡 创新点**

创新点在于将共享输入的语义结构视为可调设计维度，证明破坏题目文本的词序反而能提升学生模型的推理能力，而破坏教师仅见解答则适得其反，从而提出“选择性语义破坏”的训练策略。

**🔧 技术方法**

使用了on‑policy self‑distillation（OPSD）框架，配合LoRA适配器进行模型微调，并在训练阶段对题目文本进行词序打乱（Shuffle‑1）以及对教师参考进行控制实验。

**📊 数据集**

实验数据集包括OpenThoughts数学推理子集以及AIME 2024/2025、HMMT 2025等竞赛级别的数学题目，共计90道题目用于评估。

**📈 对比分析**

与标准OPSD以及保持题目完整的对照实验相比，Shuffle‑1在Qwen3‑1.7B、4B、8B三个规模上分别提升了约3.5%、1.1%和1.4%的平均准确率；相较于对教师参考进行词序打乱的AllShuffle‑1则显著下降。

**⚠️ 局限性**

局限性在于实验仅覆盖三种模型规模和三类数学推理数据集，未检验在更大规模模型或非数学推理任务上的泛化性，并且只探究了词序破坏对性能的影响，未深入分析其对学习动态的具体机制。

---

## 593. Safeguarding Multi-Agent Reinforcement Learning for Connected and Automated Vehicles with Recursively Feasible Control Barrier Functions

**arXiv ID:** 2611.04957 | [PDF](https://arxiv.org/pdf/2611.04957v1)

**作者:** Jianye Xu `[一作]` (RWTH Aachen University), Bassam Alrifaee `[通讯]` (University of the Bundeswehr Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出递归可行的备份时域控制障碍函数（BH-CBF），并实现完全去中心化的安全过滤器，用于保障多智能体强化学习策略在联网自动驾驶车辆中的安全。

**💡 创新点**

创新点在于：① 通过备份策略预测车辆停稳到达终止集，保证在控制边界下递归可行；② 精确考虑矩形车辆与多段道路几何；③ 采用角点预算分配实现完全去中心化的安全约束；④ 将BH-CBF用于奖励、运行时过滤和训练循环。

**🔧 技术方法**

使用控制障碍函数（CBF）、多智能体强化学习（MAPPO）、二次规划（QP）、精确欧拉预测、回退与可执行性检测、角点预算分配等技术。

**📊 数据集**

利用德国Aachen市环形交叉口和无信号交叉口的真实地图（rounD、inD）轨迹数据，以及记录的车流数据进行人车对比。

**📈 对比分析**

与多种现有CBF滤波器和预测安全过滤器（MPS、Frenet、Disk、Backup等）在零射击部署、训练循环和人车对比阶段进行比较。BH-CBF在所有阶段均无碰撞、递归可行、平均通过率约1.2-1.21辆/秒，回退率约3.3/1000步，计算时间≈12.6ms/步；其他滤波器多出现碰撞、回退率高或无法实时。

**⚠️ 局限性**

仅适用于全CAV协作环境，未考虑行人或非合作车辆；过滤器导致驾驶舒适度降低（jerk更大）；在极端混合交通或对方不合作时效果未知；需要通信与高质量地图。

---

## 594. PATTS: Parallel-Aware Test-Time Scaling for Thought-Image Generation in Multimodal Discrete Diffusion LLMs

**arXiv ID:** 2611.04948 | [PDF](https://arxiv.org/pdf/2611.04948v1)

**作者:** Yasong Dai `[一作]` (Australian National University), Hongdong Li `[通讯]` (Australian National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PATTS 框架，利用模型内部可靠性诊断在并行多模态扩散 LLM 的中间状态上进行可靠性加权搜索与语义验证，从而提升思维‑图像生成的组合正确性。

**💡 创新点**

将状态依赖的可靠性门控与外部 VLM 验证器相结合，动态决定何时对预测的思维‑图像预览进行验证；通过熵、对齐度、遮罩比例等多种内部诊断特征构造可靠性权重；首次在并行多模态扩散中实现轨迹级搜索与奖励路由。

**🔧 技术方法**

并行多模态离散扩散 LLM（MMaDA-Parallel）、离散预览预测、模型内部可靠性特征提取、Sigmoid 信任门控、外部 VLM 验证（如 Qwen2.5‑VL‑7B‑Instruct）、轨迹级搜索与混合奖励策略。

**📊 数据集**

GenEval（553 题的组合生成评估基准），以及 T2I‑CompBench、DPG‑Bench 用于跨基准鲁棒性测试。

**📈 对比分析**

在 N=32 预算下，与无 TTS、HTS、Best‑of‑N、FKD 等基线比较，PATTS 在 GenEval 上取得 0.844 的整体得分，明显优于 0.628（无 TTS）、0.787（Best‑of‑N）和 0.810（FKD）；同时生成 NFEs 仅 1796，延迟 55.1 秒，计算效率更高。

**⚠️ 局限性**

（1）最终图像验证器与组合正确性对齐度低，导致验证排名效果受限；（2）未验证的并行思维可能引入语义漂移，导致位置、属性子任务下降；（3）可靠性门控依赖经验设计的内部诊断特征，对不同任务或模型的泛化尚未充分验证；（4）中间奖励与最终准确性相关性弱，限制了可靠性预测的有效性。

---

## 595. Linear Equivalence of Full Weight Spectrum Cyclic Orbit Codes

**arXiv ID:** 2611.04947 | [PDF](https://arxiv.org/pdf/2611.04947v1)

**作者:** Minjia Shi `[一作]` (Anhui University), Ferruh Ozbudak `[通讯]` (Sabancı University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文给出了全权重谱（FWS）单轨循环子空间码的线性等价类的完整分类，解决了 Castello 等提出的第二个开放问题。

**💡 创新点**

创新点在于将等价判据化为在基域上的分式线性变换（Family I）或仿射变换（Family II）加上 q‑Frobenius 幂的组合，完全刻画了两个已知 FWS 家族中的等价类。

**🔧 技术方法**

主要使用了多项式空间理论、线性等距定理、迹对偶性以及从码到其生成子域的降维技巧。

**📊 数据集**

无实验数据集；研究完全是理论推导与证明。

**📈 对比分析**

不涉及实验比较或性能评估，论文只给出了理论等价判据和相应的分类结果。

**⚠️ 局限性**

局限性在于仅对 FWS 单轨循环子空间码中的两个已知家族给出结论，尚未扩展到更一般的子空间码或多轨码的等价问题。

---

## 596. Residual Attribute Leakage Is an Interface: Auditing Privacy Evaluation After Text De-identification

**arXiv ID:** 2611.04941 | [PDF](https://arxiv.org/pdf/2611.04941v1)

**作者:** Tuan Thanh Duong `[一作]` (Korea University), Heuiseok Lim `[通讯]` (Korea University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估文本去识别方法的隐私性，提出并使用基于先验归一化的单侧 confidence gain 评分 L，并通过可追踪编辑器 SACC 与可逆误导编辑 SWAP 进行对抗测试和审计。

**💡 创新点**

创新点包括：①引入 L 分数揭示传统分数因误导而低估泄露；②构建可追踪编辑器 SACC 及可逆误导编辑 SWAP；③提出四项评估报告清单，强调接口依赖与统计不确定性。

**🔧 技术方法**

技术手段包括：语言模型攻击（GPT‑4o‑mini、Llama、Qwen 等）、多种编辑策略（全局遮蔽、全局阶梯、SACC‑mask、重写等）、clipped leakage 评分 L、AUC、Brier 等统计检验，并用 bootstrap 置信区间评估可分离性。

**📊 数据集**

使用的数据集包括 MedQA 临床案例、TAB 法院判决、SynthPAI 社交媒体帖子，以及 500 条传记和 500 条情节摘要的转移语料。

**📈 对比分析**

通过在 988 对固定发布文本的多种评估界面（先验、候选顺序、引用、属性聚合、攻击者家族、弃权规则等）下的对比，发现多数更改导致可分离性变化，只有 5 次顺序逆转；SACC‑mask 在大多数场景下比其他编辑更私密；SWAP 在 L 上得分最高但被规则感知攻击逆转，AUC 等指标显示实际泄露情况。

**⚠️ 局限性**

局限性：依赖单一开发者攻击模型和少数语言模型，评估仅涵盖单词级提示和单一语言，缺乏对多类别属性的全面评估；统计功效有限，编辑器未充分处理上下文证据；规则泄露与语言多样性限制导致通用性受限。

---

## 597. Behind Video Generation: Observations and Insights on Multimodal Understanding from Kling

**arXiv ID:** 2611.04916 | [PDF](https://arxiv.org/pdf/2611.04916v1)

**作者:** Kling Team `[一作]`, Pengfei Wan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过构建多模态基础模型 Kling-MLLM 以及一系列专用子模型（dense captioner、reference‑grounded captioner、prompt‑enhancer、文本编码器），实现了从多模态感知到视频生成的完整链路，并提出了数据‑conditioning‑alignment‑evaluation 四阶段训练与评估框架，目标是提升视频生成的语义控制与可编辑性。

**💡 创新点**

创新点包括：
- 将视频生成的瓶颈从渲染提升到语义控制，强调多模态理解与生成目标的统一；
- 通过密集、生成导向的描述提升数据效率；
- 引入 agentic prompt‑enhancement（故事完善与检索）与多阶段强化学习，确保生成条件既可执行又富有创意；
- 设计了 reference‑grounded captioning 的 scoped binding、residual sufficiency 与 authority consistency 三大契约，解决多模态参考绑定与跨时间一致性难题；
- 采用多层文本编码器的 perceptual‑grounding 与多粒度 conditioning，改进了传统单层文本条件的不足。

**🔧 技术方法**

核心技术包括：
- Diffusion Transformer (DiT) 用作视频生成器；
- 多模态 LLM (Kling‑MLLM) 作为感知与推理基座；
- 逐阶段训练（acoustic init → multimodal → temporal capacity → long‑context consolidation）；
- 复杂的多专家监督管线（视频解析、OCR、ASR、姿态、摄像机运动等）；
- Prompt‑Enhancement（SFT → text‑level RL → DiT‑in‑the‑loop RL），并加入 agentic PE（故事完善、检索）；
- Text Encoder 适配与多层抽取，配合时间戳与边界标记实现跨模态对齐；
- 评估体系拆分为内容有效性、生成专属维度、接口合规性与难度切片。

**📊 数据集**

使用的数据集：
- 公开数据：LibriSpeech、WorldSense、TREA、LAT‑Bench、OmniBench、video‑SALMONN、UGC‑VideoCap、OmniCap‑IF、IV‑Bench、MVU‑Eval、MMBench、OCRBench、TextPecker、PerceptionBench、LongVideoBench、MVBench、MME‑VideoOCR、VSI‑Bench、ShortVid‑Bench、DSI‑Bench、Event‑Bench、Motion‑Bench、TimeLens、Charades、QVHighlights、ActivityNet；
- Kling 自研数据：dense 图像/视频字幕、参考‑生成对、跨模态代理任务、编辑与持续生成示例等。

**📈 对比分析**

与现有模型（Qwen3‑Omni、MiniCPM、Gemma、Doubao2.0‑Lite 等）对比，Kling‑MLLM‑397B‑A17B 在音频（LibriSpeech、LAT‑Bench）与视频（LongVideoBench、MVBench、MME‑VideoOCR、VSI‑Bench、Motion‑Bench、TimeLens、VidCapBench）评测中多项指标均名列前茅；
- dense captioner 与 reference‑grounded captioner 在 VidCapBench、UGC‑VideoCap、ViDiC、MRVBench 上均超过同类基线；
- 经过多阶段 PE 训练后，生成质量在 StoryEval、T2AV‑Compass、AVE‑Compass、MultiRef‑Compass 上提升至 4.7/5（约 92% 通过率）。

**⚠️ 局限性**

局限性与挑战：
- dense caption 与 RGC 对数据质量与覆盖度高度依赖，生成时仍可能出现细节缺失或不一致；
- agentic PE 的故事完善与检索显著增加推理延迟和算力成本；
- DiT‑in‑the‑loop RL 在降低严重错误的同时，压低了最大 vividness，难以保持高端创意表现；
- 文本编码器的 perceptual‑grounding 仍受限于训练任务，无法完美捕捉所有视觉与时间细节；
- 对多模态参考的绑定与语义一致性仍存在边缘情况，如高相似度多人物或多视角同步，需进一步改进；
- 评价指标多样化导致模型调参复杂，缺乏统一的综合度量。

---

## 598. EP-LVMQA: Evidence-Preserving Coarse-to-Fine Localization for Low-Quality Long Video Moment Question Answering

**arXiv ID:** 2611.04913 | [PDF](https://arxiv.org/pdf/2611.04913v1)

**作者:** Guoning Chen `[一作]` (Nanyang Technological University), Yufei Zhao `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种针对长视频问答的证据保留粗细对齐框架EP‑LVMQA，能够返回带有时间戳且可审计的答案支撑时刻

**💡 创新点**

创新点在于：①构建不确定性感知的多模态证据时间线；②设计证据卡（Evidence Card）实现规划器与细化器之间的源信息与置信度无缝传递；③利用源路由、邻域上下文与时间先验实现可解释的精细边界定位

**🔧 技术方法**

采用视觉（SlowFast、CLIP）、语音（Whisper）、文本（OCR、生成式字幕）多模态特征；规划器为两层MLP；细化器为四层Transformer；加入源路由与先验约束；训练分阶段并加入置信度一致性正则

**📊 数据集**

主要使用MedVidQA数据集进行训练与评估，同时在Ego4D‑NLQ和QVHighlights上做跨域兼容性验证

**📈 对比分析**

在MedVidQA上相较于最新基线，R1@0.5提升至41.2（+5.8点），mIoU提升至26.1（+6.8点），Suff.@2与Ver.@2分别提升至25.4%和22.6%；在Ego4D‑NLQ与QVHighlights也保持竞争力；鲁棒性测试显示平均降解率仅为11.1%，优于最强对手14.6%

**⚠️ 局限性**

虽然取得显著改进，但最终的可审计成功率仍低于30%，不足以满足实际人机交互需求；规划器仍会漏检或误判关键动作；对极端低质量或冲突模态的鲁棒性尚有提升空间

---

## 599. Cool Graphs: Active Property Search Towards Quantum Nano-Refrigerators

**arXiv ID:** 2611.04907 | [PDF](https://arxiv.org/pdf/2611.04907v1)

**作者:** Sofia Blyufer `[一作]` (Technion Israel Institute Of Technology), Uri Peskin `[通讯]` (Technion Israel Institute Of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一套基于图变分自动编码器与主动学习的框架，快速识别二维热环境吸能的量子冷却器结构。

**💡 创新点**

将物理参数映射为图结构，并在潜在空间中学习“冷却坐标”，通过不确定性驱动的主动学习聚焦决策边界，大幅提高稀有正能量流配置的发现效率。

**🔧 技术方法**

图变分自动编码器（GVAE）+ Monte Carlo dropout 产生不确定性 + 量子主方程求解器 + 对比 Gaussian Process 贝叶斯优化。

**📊 数据集**

从 8 种训练拓扑（1×2至3×3）采样 40,000 条图，另 15,000 条留作外推测试；在 32 个主动学习循环中共调用 16,000 次主方程。

**📈 对比分析**

与随机采样、Gaussian Process 以及深度集成对比；在相同 16,000 次调用下，主动学习获得约 4477 个正能量流点（随机仅 670），提高约 6.7×；ROC‑AUC 由 0.932 提升至 0.946，外推 AUC 也升至 0.934。

**⚠️ 局限性**

仅在无电子–电子相互作用、规则格子、固定温度等简化模型下验证；生成器未用于设计新结构，实验可实现性及更复杂系统的推广仍待研究。

---

## 600. Local Distributed Computation under Crash Failures

**arXiv ID:** 2611.04898 | [PDF](https://arxiv.org/pdf/2611.04898v1)

**作者:** Diana Ghinea `[一作]` (Lucerne University of Applied Sciences and Arts), Zahra Parsaeian `[通讯]` (University of Freiburg)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在同步模型下，节点崩溃如何影响本地可检查（LCL）图问题的轮复杂度，并提出了“一致邻域（Consistent Neighborhoods）”问题作为解决方案的核心工具。通过构造一致邻域，作者实现了在崩溃环境中对无误差的本地算法进行本地模拟，并给出了多种上界与下界，展示了在不同图参数（最大度 Δ、节点数 n、崩溃数 f、视图半径 R）下的最佳轮复杂度。同时，作者证明了在某些 LCL 问题（如“无源方向”）中，崩溃容忍可以在仅多一轮的代价下保持原有复杂度。

**💡 创新点**

创新点包括：
1) 定义了一致邻域问题，提供了一个统一框架，使得在崩溃环境中恢复局部一致视图成为可能；
2) 给出了完整的上界与下界，证明在不同参数范围内一致邻域的复杂度是最优的；
3) 通过一致邻域实现了一般 LCL 问题的崩溃容忍转换，保持了 O(log* n) 复杂度；
4) 对“无源方向”问题给出了仅增加一轮即可实现崩溃容忍的转换；
5) 展示了在高度、无环图等特殊图结构下可以进一步降低到 O(R) 轮。

**🔧 技术方法**

主要技术手段：
- 视图收集与信息传播的两阶段原语：全信息收集（Full-Information Gathering）和在固定子图上的洪泛（Flooding on Fixed Subgraphs）；
- 通过构造传播图（Propagation Graph）限制信息传播范围，保证邻域一致性；
- 证明一致邻域可用于在不通信的情况下局部模拟无误差算法；
- 采用不可辨识链（Indistinguishability Chain）等传统分布式计算下界技术来给出最优下界；
- 对 LCL 问题的局部模拟采用视图半径 R ≥ T + ℓ + 1 的充分条件。

**📊 数据集**

无数据集，所有结论均为理论性上界/下界。

**📈 对比分析**

方法比较：
- 在一般图上，O(n) 轮完成一致邻域；
- 对于最大度 Δ≥3，O((Δ‑1)^R) 轮；
- 若已知崩溃上界 f，O(Δ·R·f) 轮；
- 对于 Δ=2 或无环图，O(R) 轮；
- 对 LCL，O(log* n) 复杂度可保持；
- 但在检查半径 ℓ=1 且 T=ℓ=1 的某些 LCL 上，仅 R=3 视图已需要 Ω(n) 轮。整体而言，论文在不同参数范围内给出了匹配的上界与下界，证明其最佳性。

**⚠️ 局限性**

限制与不足：
- 仅讨论确定性算法；随机化方法未涉及；
- 对于极大度图（Δ≈n）仅得到 Ω(n) 的下界，仍有上界改进空间；
- 一致邻域需要预先知道 R，若 R 随 n 变化时实现难度增大；
- 某些 LCL 的崩溃容忍只能保持原复杂度，若检查半径与 T 的关系不满足 R≥T+ℓ+1，则可能失效；
- 在实际网络中，节点可能出现多种故障类型（Byzantine 等），论文仅处理崩溃（crash）故障。

---

## 601. ZEMA: confidence-calibrated signed descent with only one state

**arXiv ID:** 2611.04987 | [PDF](https://arxiv.org/pdf/2611.04987v1)

**作者:** Ali Dadsetan `[一作]` (Dalhousie University and Vector Institute), Frank Rudzicz `[通讯]` (Dalhousie University and Vector Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于梯度正负置信度的单状态优化器ZEMA（Z-score Evidence Momentum Adaptation），通过在每一步直接使用当前批次的z分数来估计梯度正负概率并进行指数移动平均，改进了Adam的“符号梯度”机制；

**💡 创新点**

创新点在于：①用单一统计量（z分数）取代Adam的两阶矩；②先将z分数转换为正负概率再平滑，减少了优化器状态量；③通过批次内方差计算实现更高的置信度估计，提升了在小批量训练中的稳健性；

**🔧 技术方法**

技术包括：基于批次内梯度方差的z-score计算、正负概率映射（CLT或Adam形式）、指数移动平均平滑、解耦权重衰减、对比实验与AdamW、Signum、Lion、RMSProp等；

**📊 数据集**

使用的主要数据集有：LoRA微调实验中的Llama-3.2系列（1B、3B、8B）、Qwen3-1.7B，生成任务E2E NLG和SAMSum；预训练实验中使用SlimPajama（3.2B tokens）和TinyStories（9.31M参数）；

**📈 对比分析**

比较方法：在LoRA微调中对不同学习率进行网格搜索，报告三种随机种子下的验证困惑度；在预训练中使用与AdamW相同的学习率或复制AdamW的学习率，评估最终验证困惑度及种子间方差；性能表现为：在LoRA微调中ZEMA与AdamW相差≤0.05困惑度，ZEMA在种子间方差更小；在160M预训练中ZEMA在AdamW的学习率下比AdamW提前0.61困惑度；

**⚠️ 局限性**

限制包括：①z-score仅在单个批次内计算，缺少跨步梯度方差信息，可能导致早期预训练收敛慢；②在预训练早期，AdamW仍然领先；③需要额外的每样本梯度方差估计，若未采用采样减速会增加计算开销。

---

## 602. GeoFaham: Domain-Specialized Multi-Agent Geospatial Analytics for Disaster Response

**arXiv ID:** 2611.04977 | [PDF](https://arxiv.org/pdf/2611.04977v1)

**作者:** Muhammad Sohail Danish `[一作]` (Microsoft AI for Good Lab), Girmaw Abebe Tadesse `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 GeoFaham 多代理系统，用于灾害响应中的多源地理空间工作流。

**💡 创新点**

通过领域专属代理、双账本调度与 Response Store 让 LLM 与地理空间计算分离，保持空间、时间元数据和中间产物完整性。

**🔧 技术方法**

结合 LLM 代理、PostGIS/GeoPandas、OSM API、STAC 影像检索、Raster 代码生成、双账本控制循环与持久化 Response Store 等技术。

**📊 数据集**

使用 17 条真实灾害记录（火灾、洪水、飓风、地震等）与对应影像、损毁、道路网络等数据，构成 40 题 GeoFaham‑Bench。

**📈 对比分析**

在同一基准上与 Codex、Claude Code、OpenCode 三个通用编码代理对比，GeoFaham 总分 0.940 超过最高基准 0.879，并且使用 7.5 倍更少 token，错误率显著下降。

**⚠️ 局限性**

仍受 LLM 推理错误影响，部分地理空间功能需强模型；对非标准或缺失元数据的灾害数据支持有限。

---

## 603. Caging Vision-Language Navigation for Legged Robots via Barrier-Certified Safety Architecture

**arXiv ID:** 2611.04972 | [PDF](https://arxiv.org/pdf/2611.04972v1)

**作者:** Abhijeet M. Kulkarni `[一作]` (University of Delaware), Guoquan Huang `[通讯]` (University of Delaware)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种双层安全架构，将预训练的视觉语言导航模型的高层语义计划与低层物理执行层分离，实现了腿式机器人安全导航。

**💡 创新点**

创新点在于：1) 将 VLM/VLA 输出作为高层提议，通过开放词汇终端对齐映射到局部可达目标；2) 在低维贝塞尔参数 MPPI 中引入动态扩展 Barrier (dEBF) 约束，提供可证明的碰撞避免保证；3) 架构模块化，兼容多种语义前端和机器人平台。

**🔧 技术方法**

使用贝塞尔参数化 MPPI、SDF（Signed Distance Field）与 dEBF 约束、JPS 全局路径规划、实时 3D 感知与位姿估计、OpenVSLAM/Simulink 等技术；同时与 ChatGPT、Claude、Gemini 等 LLM 与 VLM 进行接口对接。

**📊 数据集**

实验在 Unitree G1 humanoid 与 Ghost Robotics Vision 60 quadruped 上完成，环境为自建室内外真实场景及 Gazebo 模拟；未使用公开数据集，而是基于自定义指令和环境。

**📈 对比分析**

比较方法：对比标准 MPPI 与 MPPI+dEBF、开启/关闭 JPS、开启/关闭 caging；结果显示 dEBF 将碰撞率从 0.33/0.50 降至 0%，成功率保持或提升；在 Vision 60 上碰撞率从 0.67/0.91 降至 0.20/0.08，成功率提升；在长任务中未开启 caging 碰撞率为 100%，开启后降至 0%。

**⚠️ 局限性**

局限性：依赖 SDF 对当前环境的完整映射，无法处理未观测几何；未考虑状态估计误差、延迟与跟踪误差；仅在静态障碍物上验证；终端对齐依赖外部目标检测，无法纠正错误或不可达的语义目标。

---

## 604. DPAGait: Dynamic Pose-Knowledge-Aware Network for Robust Gait Recognition in the Wild

**arXiv ID:** 2611.04950 | [PDF](https://arxiv.org/pdf/2611.04950v1)

**作者:** Zenan Lin `[一作]` (South China University of Technology), Wenxiong Kang `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出DPPS骨骼增强式轮廓表征与DPAGait双流网络，实现对户外动态步态的鲁棒识别。

**💡 创新点**

创新点在于将姿态解析信息嵌入二值轮廓中提升信息熵，并联合3D SMPL运动知识通过残差注意融合提升辨识度。

**🔧 技术方法**

采用Pose-Parsing渲染、RankSVM动态编码、SMPL参数编码、DeepGaitV2骨干和Residual Attention Fusion模块。

**📊 数据集**

在SUSTech1K、CCPG、Gait3D、GREW四大公开步态数据集上进行评测。

**📈 对比分析**

与多种单模态与多模态基线相比，DPPS/DPAGait在Rank‑1、mAP等指标上普遍提升，最高可达SUSTech1K 89.8% Rank‑1。

**⚠️ 局限性**

局限性包括对SMPL估计的依赖、跨域泛化仍有限以及在极端遮挡或低帧率场景下的性能下降。

---

## 605. HERIT: Enabling Global Access to Hanja-to-English Historical Translation by Mitigating Temporal Bias

**arXiv ID:** 2611.04949 | [PDF](https://arxiv.org/pdf/2611.04949v1)

**作者:** Dayeon Jeong `[一作]` (Kangwon National University), Kyeongpil Kang `[通讯]` (Kangwon National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出HERIT框架，通过检索增强生成（RAG）和融合技术为韩文汉字历史档案生成高质量伪标签，并训练Hanja‑to‑English翻译模型；

**💡 创新点**

创新点在于利用大量Hanja‑Korean资源与检索增强的上下文提示相结合，构建无监督伪标签数据，显著缓解了稀缺且时段偏倚的Hanja‑English语料问题；

**🔧 技术方法**

采用LLM（Gemini‑2.5‑Flash）做Proposer/Fusor，结合RAG检索、Fusion-of‑N、多轮提示与词表（glossary）扩充；

**📊 数据集**

使用韩国国家历史研究院提供的AJD（王朝编年史）Hanja‑Korean/English三元组，扩充为约315k伪标签样本；

**📈 对比分析**

与多种基线（Hunyuan‑MT‑7B、Qwen3 32B、Pivot‑MT、无检索等）比较，HERIT在BLEU/chrF++/METEOR指标上提升约3-5点，并在LLM与人类专家评估中获高分和70%以上的偏好率；

**⚠️ 局限性**

局限包括：仅使用Gemini‑2.5‑Flash做Proposer/Fusor，未尝试更大或多样化LLM；受硬件限制，序列长度截断导致部分数据被过滤；未覆盖JRS档案，模型对更长文本仍需分块处理。

---

## 606. Learning Where to Look: Observation Policy Optimization for Thinking with Images

**arXiv ID:** 2611.04943 | [PDF](https://arxiv.org/pdf/2611.04943v1)

**作者:** Junfeng Wang `[一作]` (University of Science and Technology of China), Qixing Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种用于工具增强视觉推理的观察策略优化（Observation Policy Optimization, OPO）方法，通过在高不确定性视觉观察点进行分支探索并将局部证据质量奖励归因到具体的观察操作，实现更精准的观察动作学习。

**💡 创新点**

核心创新在于：① 不确定性引导的观察分支（UOB）——在生成过程中基于边界框生成的不确定性（token entropy）选择分支点，聚焦在模型最不确定的观察动作；② 证据局部优势归因（ELAA）——在同一前缀下产生的子分支中比较视觉证据充分性奖励，并只将局部优势应用到对应的观察动作 token 上，解决了传统轨迹级奖励与观察动作之间的信用错配。

**🔧 技术方法**

技术细节包括：使用语言模型（Qwen3-VL）作为主体，结合图像缩放工具（Zoom-in），利用强化学习（PPO）框架，改进奖励函数为：全局奖励（答案正确性、格式、工具使用）+ 局部证据奖励；不确定性评估采用边界框 token 的平均熵和方差；分支预算分配与冗余过滤机制实现高效探索。

**📊 数据集**

评估数据集为三大高分辨率视觉推理基准：V*、HR-Bench 4K、HR-Bench 8K。实验采用 Qwen3-VL 的 2B 与 8B 两个尺度模型。

**📈 对比分析**

与多种基线（GPT-5、Claude Opus、Gemini-3 Pro、LLaVA-OneVision、Qwen2.5-VL、Qwen3-VL、深度眼等）比较，OPO-8B 在 V* 上取得 92.7% 准确率，HR-Bench 4K 85.5%，HR-Bench 8K 80.8%，分别排名第一或第二，且相较于 8B 原始模型提升 4.7/4.1/4.3 分点，显示出明显的性能提升。

**⚠️ 局限性**

局限性包括：① 需要额外的 rollout 与分支预算，计算成本相对传统单一路径推理更高；② 对视觉裁剪工具的依赖，若工具接口或裁剪精度不足，可能导致不确定性评估失效；③ 证据质量评估依赖于冻结的视觉评审模型，若评审误判会影响奖励信号；④ 在极大模型或多任务场景下的可扩展性尚未充分验证。

---

## 607. Modulating Merging Strengths via Joint Loss Estimation for LoRA-based Continual Learning

**arXiv ID:** 2611.04931 | [PDF](https://arxiv.org/pdf/2611.04931v1)

**作者:** Kun Gu `[一作]` (Xidian University), Nannan Wang `[通讯]` (Xidian University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于LoRA的连续学习框架M^2LE，通过元素级自适应合并和任务模拟扰动实现更优的稳定性与可塑性。

**💡 创新点**

创新点包括：①HIM利用联合损失解析得到每个LoRA参数的元素级合并系数；②TSP在训练时通过标签互斥子集产生扰动，模拟跨任务干扰，提升合并后鲁棒性。

**🔧 技术方法**

使用技术包括LoRA低秩适配、对角Fisher估计的Hessian信息、任务模拟扰动(TSP)、梯度下降、ViT-B/16预训练模型。

**📊 数据集**

实验数据集涵盖ImageNet-R、ImageNet-A、CIFAR‑100、CUB‑200和DomainNet。

**📈 对比分析**

与现有PEFT、Prompt、Adapter等连续学习方法对比，M^2LE在无重放设置下在10/20任务ImageNet‑R/A以及其他基准上平均提升1–6%的准确率，加入分类器对齐后可逼近重放方法的表现。

**⚠️ 局限性**

局限性在于仅针对无重放场景，需额外Fisher存储和TSP训练开销，且对极大任务数的可扩展性尚未充分验证。

---

## 608. Sorting in PRAM, Parallel Comparison Model and Parallel External Memory Model

**arXiv ID:** 2611.04914 | [PDF](https://arxiv.org/pdf/2611.04914v1)

**作者:** Gopalan Sajith `[一作]` (Indian Institute of Technology Guwahati), Ranajit Senko `[通讯]` (Indian Institute of Technology Guwahati)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在本文中，作者提出了适用于CREW PRAM、并行比较模型（PCM）和并行外存模型（PEM）的高效排序算法，分别实现了O(log n)、O(log n / log α)和O(log(N/B))的时间/I/O复杂度。

**💡 创新点**

创新点在于设计了一种新的并发读取独占写入（CREW）算法，改进了传统的Cole合并排序，并在PCM和PEM上实现了匹配下界的最优算法。

**🔧 技术方法**

主要技术包括分层划分（Stratify）过程、Valiant合并、前缀和、交叉排名以及多路归并。

**📊 数据集**

由于论文为理论研究，未使用具体数据集，仅通过算法分析和复杂度证明验证。

**📈 对比分析**

与既往算法相比，本文提供了理论上的最优复杂度；在PEM上，当p≥N/B时达到Ω(log(N/B))的下界，且比之前的O(log(N/B)+N/(pB))更快。

**⚠️ 局限性**

局限性包括实现难度高、常数项可能较大、仅在高并行度下最优，对低p情况仍存在性能空间，且缺乏实验验证。

---

## 609. Attending to Memory Writes Reduces Seed Variability in Recurrent Length Generalization

**arXiv ID:** 2611.04902 | [PDF](https://arxiv.org/pdf/2611.04902v1)

**作者:** Sungwoo Goo `[一作]` (Chungnam National University), Sangkeun Jung `[通讯]` (Chungnam National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在递归模型中使用有界增量注意力（Bounded Delta Attention）来改善记忆写入并评估其对长度泛化的影响。

**💡 创新点**

提出了在递归更新之前对未校正的增量进行注意力计算，并给出有界校正的机制，作为一种新的学习支架。

**🔧 技术方法**

采用可加相位递归（phase recurrence）与RoPE编码的多头因果注意力，并在增量上施加 π×tanh 的有界校正。

**📊 数据集**

在模 5 前缀跟踪任务（输入符号 0–4，训练长度 1–40，测试长度 80）上进行实验。

**📈 对比分析**

与本地注意力循环基线、单层状态注意力以及梯度阻断/去校正等变体进行七种种子比较；在长度 80 下，上层增量注意力达到 70.8%±8.3% 的准确率，显著降低样本方差；下层增量注意力在 80% 时相较状态注意力提升至 62.6%±12.9%，而状态注意力仅 19.4%±0.5%；去校正对上层影响小，去校正对下层影响大。

**⚠️ 局限性**

仅在单一的模 5 前缀任务与有限长度下验证，未证明在自然语言等更复杂任务或更长序列上有效；对机制的解释仍为假设（信用分配改进），缺乏更系统的解析。

---

## 610. Near-Linear Lower Bounds for Testing Directed Acyclicity

**arXiv ID:** 2611.04901 | [PDF](https://arxiv.org/pdf/2611.04901v1)

**作者:** Artur Czumaj `[一作]` (University of Warwick), Yuyang Wang `[通讯]` (University of Science and Technology of China)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

该论文证明在有界度入射列表模型中，任意自适应双侧误差算法对有向图的无环性测试至少需要Ω(n^{1-η})次查询，几乎线性。

**💡 创新点**

创新点在于构造一种隐藏高girth循环核心并被掩蔽层层叠加的输入分布，配合信息理论与统计不可区分性分析，实现几乎线性下界。

**🔧 技术方法**

主要技术包括随机排列核心、高girth随机图的切换论证、聚类展开（cluster expansion）以及信息压缩/信息控制方法。

**📊 数据集**

研究不依赖具体数据集，而是基于随机生成的理论构造。

**📈 对比分析**

相较于之前的Ω(n^{1/3})或Ω(n^{5/9})下界，该方法取得了指数可任意逼近1的下界，显著提升了已知的最坏情况复杂度。

**⚠️ 局限性**

局限在于仍未证明存在o(n)查询的算法，且下界依赖于接近度参数η以及常数度数限制，未涵盖所有距离和度数组合。

---

## 611. Properly defining Erlang lists

**arXiv ID:** 2611.04929 | [PDF](https://arxiv.org/pdf/2611.04929v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 612. Deep Neural Operator Learning for stochastic heat equations and applications

**arXiv ID:** 2611.04955 | [PDF](https://arxiv.org/pdf/2611.04955v1)

**作者:** Hengrong Du `[一作]` (Fisk University), Weinan Wang `[通讯]` (University of Oklahoma)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文建立了半线性随机热方程（以及随机Navier–Stokes方程）解算子在随机分布型高斯驱动下的通用逼近定理，证明只需要对驱动和初始数据做有限个线性测量，便能用ReLU分支-支架网络在任意有限随机矩中以任意误差逼近整个解算子。

**💡 创新点**

创新点在于将传统的ODE/PDE算子学习框架推广到含分布型随机噪声的SPDE，利用平滑Fourier截断将无穷维随机输入压缩为有限维向量，并结合Besov空间与Schauder估计实现误差控制；同时给出了明确的测量数量、网络规模与逼近误差之间的算量关系。

**🔧 技术方法**

主要技术包括：平滑dyadic分解与Littlewood–Paley频域截断、Besov与Schauder正则性估计、分支–支架（branch–trunk）ReLU网络结构、有限维测量映射与分区函数、Wiener-chaos关联、以及高概率局部Navier–Stokes逼近。

**📊 数据集**

本研究为理论性工作，没有使用具体数值数据集；所讨论的随机噪声为均值为0、取值于分布空间的高斯过程，符合假设下的任何具体概率模型即可。

**📈 对比分析**

由于是理论定理，文中未给出与其他机器学习或数值方法的实测比较；但作者给出了误差上界与网络参数的显式比例关系，表明在给定误差ε、空间维数d与噪声正则性参数时，可获得的测量次数与网络规模随ε的幂律增长。

**⚠️ 局限性**

主要限制在于：①需要对噪声满足一定的Besov正则性（β0>-2且β0+2>0）才能保证热卷积连续；②局部Navier–Stokes逼近仅在随机误差概率为高的事件上成立，无法给出全局高概率收敛；③逼近误差与网络规模呈指数或幂次关系，实际实现时参数量可能很大。

---

## 613. Robustness Verification against Patch Attacks via Group-Lasso Semidefinite Relaxation

**arXiv ID:** 2611.04939 | [PDF](https://arxiv.org/pdf/2611.04939v1)

**作者:** Yukina Nakagawa `[一作]` (Institute of Science Tokyo), Kazuhide Nakata `[通讯]` (Institute of Science Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向补丁攻击的神经网络鲁棒性验证方法，利用群组稀疏建模和 SDP 松弛实现一次性验证。

**💡 创新点**

创新点在于将补丁攻击位置作为变量联合建模，构造混合整数 SDP（MISDP），随后通过三种凸松弛（Full-Group、L₂、Group‑Lasso）去除离散变量，显著提升验证紧致性和效率。

**🔧 技术方法**

核心技术包括：群组稀疏攻击建模、MISDP 形式化、SDP 及半正定松弛、Group‑Lasso 组 L1‑L2 松弛、MOSEK 求解器。

**📊 数据集**

使用降维后的 MNIST（5×5）图像进行实验，构造全连接网络（Tiny/Small/Medium/Large）和两种 FCNN 结构（FCNNa、FCNNb）进行验证。

**📈 对比分析**

与现有 Lₚ 规范下的 Bounded‑MILP、CROWN‑IBP、CROWN 等基线比较，Group‑Lasso 通过仅 0.34% 的目标值增幅，实现 91% 的验证率（相较基线 ≤70%），且在 MISDP 计算时间上仅占 3%（速度提升 41×–1430×）。

**⚠️ 局限性**

局限性包括：仅验证小型全连接网络，未针对高分辨率图像或卷积/Transformer 结构；松弛方法仍受限于固定非重叠分组，无法完全覆盖任意物理补丁形状；大规模求解仍需开发更高效的一阶/分块算法。

---

## 614. VISTA: Vision-Language Integrated Semantic Triage Architecture for Online Face Clustering in Long Videos

**arXiv ID:** 2611.04932 | [PDF](https://arxiv.org/pdf/2611.04932v1)

**作者:** Jiashu Liao `[一作]` (University of Glasgow), Tanaya Guha `[通讯]` (University of Glasgow)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了VISTA框架，实现了长视频中全流程在线人脸聚类；

**💡 创新点**

创新点在于引入符号语义表示、三区GMM阈值动态分区以及时空解耦语义仲裁，解决过度分割与视觉不稳定性；

**🔧 技术方法**

使用RetinaFace+FaceNet做几何特征，CLIP+DeepFace提取零样本属性，符号映射表、GMM、稀疏置信度调节等；

**📊 数据集**

在VPCD基准上测试，涵盖《老友记》6集、《暮光之城》6集和《神探夏洛克》3集；

**📈 对比分析**

与同条件下的COFC基线对比，VISTA在WCP、NMI和ID_Acc均显著提升，尤其在灯光和服装变化剧烈的情节中表现突出；

**⚠️ 局限性**

局限在于对符号词典和GMM先验的固定依赖、对极端服装/光照变化的处理仍不完美，以及在Sherlock等高变异集上偶尔牺牲连续性。

---

## 615. Knowledge Graph Lakehouse: A Cloud-Native Architecture for Managing Enterprise Knowledge Graphs at Scale

**arXiv ID:** 2611.04899 | [PDF](https://arxiv.org/pdf/2611.04899v1)

**作者:** Bashar Ahmad `[一作]` (Johannes Kepler University Linz), Loris Bozzato `[通讯]` (Universita degli Studi dell'Insubria)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并实现了知识图谱湖仓（KG Lakehouse）架构，采用云原生微服务和数据湖仓理念，在数据流入时对文档按多维上下文进行索引，随后按需动态构造对应的上下文化知识图（CKG），而非维护单一大规模知识图。

**💡 创新点**

创新点在于：①将数据湖仓模式迁移至知识图谱，解决了大规模、异构、时变数据在知识图管理中的三V（volume, velocity, variety）问题；②通过上下文索引和基于KG-OLAP的按需构造，实现了查询时只花费于所需子图的成本；③利用上下文化知识图的确定性和覆盖关系，支持可验证的缓存与分布式构造，保证结果不受缓存状态影响。

**🔧 技术方法**

技术包括：Java、Kubernetes容器化；Apache Kafka消息队列（离线摄取）；MinIO对象存储（原始文档）；Apache Cassandra宽列存储（上下文索引）；Redis内存缓存（已构造模块）；RESTful HTTP服务；内容分析器与语义提升器插件化；GraphFrame、RDF四元组、标签属性图等多种输出格式；Apache Jena推理服务；Spark GraphFrames用于大规模后处理。

**📊 数据集**

使用合成航空交通管理数据集：从AIXM、IWXXM、FIXM三种标准生成的文档，覆盖不同时间（1天至1年）、空间（单机场到大洲）和飞行密度（100–10,000航班/天）的场景。数据集包含数十万至数百万文档，提供可复现的上下文验证。

**📈 对比分析**

对比方法：将按需构造与传统完整映射+SPARQL加载的单一立方体做对照。实验显示：①摄取/索引维持低延迟，≈27–28 ms/文档；②上下文选择与索引大小几乎独立于数据量；③构造成本随查询范围增长线性，而非文档总量；④在单机内存（≈1.4 GB）即可完成大规模子图构造，而完整立方体构造需46 GB内存、55 分钟。性能上，按需构造在典型范围查询下仅需数百毫秒，且缓存可进一步减少时间。

**⚠️ 局限性**

局限性：①架构针对范围限定查询，若需全图查询仍需完整加载；②需要针对每种数据格式编写分析器与提升器，领域迁移需重新配置；③对文档的双重解析（摄取与构造）导致额外CPU负荷；④缓存失效机制依赖索引更新，若索引出现偏差可能产生过期结果；⑤在极大数据量下，索引仍随文档数增长，但相对数据量小；⑥实时推理仍需外部服务，无法在构造阶段实现。

---

## 616. VIGOR: Visual Gain Ordering for Hallucination Mitigation in Multimodal Discrete Diffusion Language Models

**arXiv ID:** 2611.05066 | [PDF](https://arxiv.org/pdf/2611.05066v1)

**作者:** Junzhe Chen `[一作]` (Tsinghua University), Lijie Wen `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练、可插拔的推理策略——视觉增益排序（Visual Gain Ordering），通过对每个被掩盖位置在带图像与无图像前向传播中的置信度进行对比，重新排序掩码位置的提交顺序，从而抑制多模态离散扩散语言模型（dLLMs）的幻觉。

**💡 创新点**

将幻觉视为“过早提交”问题，而非仅词预测错误；利用视觉与无视觉条件置信度的对比，仅在推理阶段改进提交顺序，无需微调或结构改动。

**🔧 技术方法**

多模态离散扩散语言模型的块级解码、视觉消融前向传播、视觉增益分数计算与排序。

**📊 数据集**

LLaDA‑V、Lumina‑DiMOO 两个模型；幻觉基准 POPE、CHAIR、AMBER。

**📈 对比分析**

与基线及两种推理层面幻觉缓解方法（ICT、VCD）比较；在所有模型、所有基准上均显著提升：POPE 准确率提升 4.7/2.6 点、F1 提升 3.1/8.3 点；CHAIR_S/CHAIR_I 分别下降 6.5/4.1 点；AMBER 的各项指标均得到改进。

**⚠️ 局限性**

需要在每一步额外进行一次无图像前向传播，推理时间约翻倍；依赖模型内部置信度和可控制的提交调度，主要适用于开放权重或可控制的模型。

---

## 617. Semi-Streaming Reachability and Shortest Paths in $n^{1/3+o(1)}$ Passes

**arXiv ID:** 2611.05061 | [PDF](https://arxiv.org/pdf/2611.05061v1)

**作者:** Gary Hoppenworth `[一作]` (Toyota Technological Institute at Chicago), Jakub Tarnawski `[通讯]` (Microsoft Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在半流式（semi‑streaming）模型下提出了一种新的单源可达性与单源最短路（SSSP）算法，使用 O(n) 空间，仅需 n^{1/3}+o(1) 次流遍历即可求出精确的可达性与距离；并给出了相应的并行算法，在 O(m) 工作量下实现了 O((n^ω/m)^{1/(2ω-2)}·n^{o(1)}) 的深度。该算法在同样的空间预算下将之前最优的 n^{1/2}+o(1) 次遍历显著降低到 n^{1/3}+o(1)。

**💡 创新点**

核心创新点在于：① 引入了“隐式”shortcut 集和 hopset 的概念，允许把本来规模为 O(nx) 的 shortcut/ hopset 以 O(n) 的空间间接表示；② 设计了两速度递归（two‑speed recursion）构造隐式 hopset，先沿早期层保持原 CFR hopset 结构，再在阈值层后改为块闭包（block closure）并放慢距离尺度；③ 利用 τ‑skeletons、双倍递增（doubling）技巧在流式中模拟 BFS/Dijkstra，显著减少对每个块的访问次数。

**🔧 技术方法**

技术手段包括：隐式 hopset/shortcut 集的定义与构造；基于 CFR hopset 的分层递归与随机 pivot 采样；使用规模缩放与四舍五入（scaling & rounding）将权重整数化以便批量 Dijkstra；τ‑skeleton 以及双倍递增搜索实现块内部的高效查询；并行实现利用 Zwick 的快速矩阵乘法计算块闭包；引入 Abort 规则保证每次递归不超过 O(n) 复制和 O(m) 边扫描；最后通过多阶段（log n）迭代构造确保近似转为精确距离。

**📊 数据集**

该工作为理论分析，未使用具体实验数据集；所有结果均基于理论证明与概率上界，假设输入图为任意有 n 个顶点、m ≥ n 条边、正整数权重且多项式界定的情况。

**📈 对比分析**

与之前最优的 n^{1/2}+o(1) 次遍历算法（Jambulapati‑Liu‑Sidford）相比，本算法把遍历次数压缩到 n^{1/3}+o(1)；与最弱下界 Ω(log n / loglog n) 仍相距遥远，但在理论上已突破了长期未解的中点。并行算法相比最近的 Ashvinkumar‑等 结果，将深度从 O((n^3/m)^{1/4})·n^{o(1)} 降到 O((n^ω/m)^{1/(2ω-2)})·n^{o(1)}，同时保持 O(m) 工作量。该技术同样可直接应用于匹配、单调性测试、拓扑排序、强连通分量等经典图问题，实现相同次数的遍历。

**⚠️ 局限性**

局限与挑战：① 仍未达到信息论下界 Ω(log n / loglog n)；② 依赖随机化，且需高概率保证，实际实现时可能需多次尝试；③ 对权重要求为正整数且多项式范围，非负实数权重或极大比值时需额外处理；④ 迭代多阶段和 Abort 机制导致算法实现复杂度较高，常数因子不易评估；⑤ 只给出了理论复杂度，缺乏实验评估与实际性能验证。

---

## 618. Rank-metric intersecting codes

**arXiv ID:** 2611.05059 | [PDF](https://arxiv.org/pdf/2611.05059v1)

**作者:** Martino Borello `[一作]` (Université Paris 8), Lune Vila `[通讯]` (Inria)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并研究矩阵秩度量下的交集码（row‑intersecting code），给出了其几何与代数两种描述，并通过切片空间和代数结构（Jacobson radical、socle）构造与判定此类码。

**💡 创新点**

创新点在于：①将传统的线性秩度量交集码推广到矩阵级别；②引入第三切片空间的列超平面不可分解性作为交集性质的等价几何判据；③在矩阵代数情形下利用雅可比-塞尔结构及socle简化交集码的构造与判定；④给出新的参数上界与构造方案，且部分上界可达且与已知线性情形区别明显。

**🔧 技术方法**

主要技术包括：张量与切片空间的定义与运算；投影与缩短的几何分析；极大秩距码（MRD）与q‑system的几何解释；代数理论中的Jacobson radical、socle、简单与半单模的理论；以及对矩阵代数正规表示的矩阵化。

**📊 数据集**

无具体实验数据集，研究完全基于理论推导与代数构造。

**📈 对比分析**

方法与评估主要通过数学证明和构造示例来展示；对比传统线性交集码在参数上界与构造的差异，并说明在矩阵代数中能获得更高维度且不满足传统距离阈值的交集码；未给出数值性能指标。

**⚠️ 局限性**

局限性包括：①目前仅讨论矩阵秩度量，缺乏对更一般度量或非矩阵形式的推广；②构造多依赖于MRD码或特殊矩阵代数，实际应用范围受限；③对非半单或非可分解代数的完整判定仍不成熟；④理论上给出的上界虽可达，但实际构造的复杂度与可实现性尚未深入研究。

---

## 619. ToolSpace: Scaling Tool-Centric Environment Synthesis for Long-Horizon Agents

**arXiv ID:** 2611.05036 | [PDF](https://arxiv.org/pdf/2611.05036v1)

**作者:** Guanting Dong `[一作]` (Renmin University of China), Zhicheng Dou `[通讯]` (Renmin University of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 ToolSpace 框架，通过从真实工具调用追踪中提取数据、状态和重试依赖，构建加权工具组合图；在该图上执行状态感知随机游走并回滚，生成可回放的长时序任务并附加断言检查，进而在多环境下合成训练数据并生成 AutomationBench、τ³‑Bench Banking 的 Pro 版本。

**💡 创新点**

① 逆向合成：先构造工具空间再推导环境；② 用确定性规则而非 LLM 判断挖掘真实执行依赖；③ 仅在操作型环境上按需合成数据库；④ 通过回滚和验证保证链条可执行；⑤ 通过任务逆向生成与断言检查实现可复现性。

**🔧 技术方法**

确定性依赖挖掘、加权图聚类、状态感知随机游走与回滚、任务逆向生成与断言校验、SFT 与 RL 训练、Qwen/GLM/Claude 等大型语言模型。

**📊 数据集**

真实 MCP 服务轨迹（MCP‑ecosystem、Web‑sourced 轨迹）、AutomationBench、τ³‑Bench Banking、七个长时序工具使用基准（SkillsBench、JobBench、BFCL V4、Harvey LAB、OfficeQA Pro 等）以及公开的 agent 轨迹语料库。

**📈 对比分析**

与传统环境扩展基线（EnvScaler、Agent‑World、Agent‑Reflex 等）以及前沿模型（Claude Opus、GPT 5.6、GLM 5.2、Qwen 3.8 Max 等）进行对比；ToolSpace 在 7‑8B、27‑35B、744B 三个规模池中均显著提升，Qwen3.6‑35B‑A3B 的平均分从 23 提升至 41，GLM‑5.2 从 44.6 提升至 56.6，匹敌 Claude Opus 5；在 Pro 版本上，所有前沿模型相对标准版均下降多达 38.2 分。

**⚠️ 局限性**

① 依赖已观测的执行轨迹，若真实工具或接口发生变化需重新收集；② 只捕获相邻调用的依赖，可能漏掉更深层次或跨步依赖；③ 回滚采样与验证成本较高；④ 任务生成仍依赖语言模型，可能产生语义偏差；⑤ 评估仅覆盖现有基准，对新场景的泛化能力尚待验证。

---

## 620. The APN Property of Two Families of functions over $\F_{2^{n}}$ from the Switching Construction

**arXiv ID:** 2611.05014 | [PDF](https://arxiv.org/pdf/2611.05014v1)

**作者:** Chenmiao Shi `[一作]` (National University of Defense Technology), Kangquan Li `[通讯]` (National University of Defense Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

证明了两族通过交换法构造的非幂函数 F₂ 与 F₃ 在二进制扩展域 𝔽₂^{2m}（m≥5 为奇数且 3∤m）上满足几乎完美非线性（APN）性质。

**💡 创新点**

创新点在于将原始四项式函数通过 EA 等价转换为两种二变量形式，并利用有限域上的矩阵方程与 Frobenius 自动同态的特征多项式，构造了全新的代数证法，彻底解决了此前仅有计算验证的猜想。

**🔧 技术方法**

主要使用了有限域理论、矩阵代数（尤其是 2×2 矩阵的特征多项式与幂零性质）、Frobenius 自动同态、以及线性/非线性变换（EA/CCZ 等价）。

**📊 数据集**

本工作不涉及实验数据集，全部通过严谨的符号推导和理论证明完成。

**📈 对比分析**

由于研究对象为纯粹的理论性质，没有直接的性能指标或对比实验；相对已有的 APN 构造，证明了两族函数在更广泛参数范围内仍保持 APN 性质。

**⚠️ 局限性**

局限性在于只对 m 为奇数且 3∤m 的情况给出证明；对偶数维度或 3 整除 m 的情况仍未覆盖；此外，该方法依赖于特殊的矩阵结构，难以直接推广到更一般的函数族。

---

## 621. Throughput-Optimal Control of Distributed Computing Networks for Service DAGs

**arXiv ID:** 2611.05007 | [PDF](https://arxiv.org/pdf/2611.05007v1)

**作者:** Peng Li `[一作]` (Peking University), Chen Chen `[通讯]` (Peking University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种针对无线分布式计算网络中服务有向无环图（DAG）的吞吐量最优控制框架，联合调度功能放置、路径选择和传输调度，并同步同一请求在合并节点的多输入。

**💡 创新点**

创新点在于：①利用“有效映射”抽象同时描述功能执行位置和依赖路径；②给出完整的容量区域LP并证明合并同步不影响容量；③设计DCNC-DAG动态策略，在任意有限帧长内（甚至映射价格滞后）保持吞吐量最优；④针对大规模DAG引入基于队列的启发式DCNC-DAG-H。

**🔧 技术方法**

核心技术包括：有效映射与状态平均LP的构造；Lyapunov漂移证明和虚拟队列价格驱动的映射决策；DynVMP最小成本映射求解；NTO（最近源优先）调度保证最优级别服务；以及合并同步的等待池分析。

**📊 数据集**

实验使用多种真实无线拓扑（Milan、Random、RoofNet、FlockLab）和多种服务DAG：自动驾驶、随机6功能DAG、20-31功能的Mixtral/Llama模型以及SRA-Search-22和SIPHT-31工作流，全部采用合成的计算与通信工作负载。

**📈 对比分析**

与无队列感知基线（CTS、Randomized）以及特定领域基线（Lazarus、FlexMoE）比较，DCNC-DAG在四个拓扑上实现了2.7-5.5倍的吞吐率，DCNC-DAG-H在大规模DAG上提升至约2.68倍；同时在LP容量边界实验中验证了理论预测；帧长与延迟之间的权衡也得到系统评估。

**⚠️ 局限性**

主要局限包括：映射优化为NP‑hard，DCNC-DAG-H仅为启发式；仅考虑i.i.d.无线状态和有界请求到达；需要全局信息来计算映射；映射计算开销与帧长平衡；未考虑能耗或功率约束；合并同步假设同一请求的所有输入在同一时刻到达，实际场景可能更复杂。

---

## 622. ScoreDial: Precise Control of Visual Attributes in Image Generation via Regressor Guidance from Limited Human Ratings

**arXiv ID:** 2611.05004 | [PDF](https://arxiv.org/pdf/2611.05004v1)

**作者:** Jianbo Tang `[一作]` (Keio University), Takeo Kato `[通讯]` (Keio University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了ScoreDial方法，利用少量人类评分实现对视觉属性的数值化控制，结合目标条件化和误差自适应引导；

**💡 创新点**

创新点在于闭环生成：先用可微分回归器拟合人类评分，再通过数值嵌入定位目标，随后在采样过程中根据回归器误差动态调整；

**🔧 技术方法**

技术实现包括：在Stable Diffusion 3.5‑medium上训练LoRA域适配；用DINO/CLIP特征训练岭回归回归器；将请求值通过正弦投影+MLP嵌入Transformer；采用MPGD‑LDM与Universal Guidance的混合引导；

**📊 数据集**

使用四个属性数据集：轮胎打滑阻力（120图）、室内视觉复杂度（100图）、食物欲食度（479图）、肥皂透明度（150图），以及相应的训练池（轮胎984、肥皂800等）；

**📈 对比分析**

在与Numeric prompt、MPGD‑flow、UG‑flow及单独条件化的匹配实验中，ScoreDial在同一SD3.5基底、相同种子和提示下实现R‑MAE最低（≤所设容忍度），Spearman相关最高，独立审计器位移最大；人类评测亦验证其进阶清晰度；

**⚠️ 局限性**

局限性包括：仅针对有限的四个属性；回归器和引导共享相同模型，可能导致过拟合；肥皂透明度回归器仅在图像上交叉验证，缺乏真实物理测量。

---

## 623. Unifying and Beyond the Theory of Pervasive Capability Degradation in LLM Post-training SFT from Feature Interference

**arXiv ID:** 2611.05003 | [PDF](https://arxiv.org/pdf/2611.05003v1)

**作者:** Hang Chen `[一作]` (Nanyang Technological University), Wenya Wang `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Pervasive Capability Degradation（PCD）的底层几何理论，并基于该理论设计 AEGIS 损失来解决监督微调导致的普适能力衰减问题。

**💡 创新点**

创新点在于将特征干扰与 Laguerre–Voronoi 几何相结合，形成从激活到输出的完整因果链；统一六大缓解范式并阐明其理论极限；证明单标签交叉熵无法单独解决 PCD；提出通过方向边界约束实现特征干扰的独立抑制。

**🔧 技术方法**

技术包括：特征超位置分析、Laguerre–Voronoi 几何、梯度正交化、AEGIS 损失的方向宽度自适应与 Huber 损失约束、参数化的方向漂移抑制，以及对 unembedding 层的几何约束。

**📊 数据集**

使用的评测数据集包括 GLUE、GSM8K、CodeAlpaca、Code‑Bagel‑Hermes、Infinity‑Instruct‑Code、Evol‑Instruct、LogiQA2、FLAN‑2022 等多任务通用能力与下游任务数据。

**📈 对比分析**

与六种代表性缓解方法（PSFT、SMART、KnOTS、DARE、DiFT、CLUE）在 Mistral‑7B、Qwen2.5‑7B、Gemma‑2‑9B 等多模型、多任务上进行对比；AEGIS 在目标任务与普适能力两方面均显著优于基线，且表现最为稳定。

**⚠️ 局限性**

局限性包括：仍受特征稀疏和高维特征混叠的限制，AEGIS 对极端小样本或长序列任务的效果可能受限；理论假设与实际训练细节之间存在偏差，未来需进一步验证和扩展。

---

## 624. CRONOS: Benchmarking Non-Episodic Reinforcement Learning for Robotic Manipulation

**arXiv ID:** 2611.05080 | [PDF](https://arxiv.org/pdf/2611.05080v1)

**作者:** Po-Yi Wu `[一作]` (National Taiwan University), Tsung-Wei Ke `[通讯]` (National Taiwan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个用于非episodic场景重置下RL微调视觉-语言-动作模型（VLA）的仿真基准，并系统评估了不同重置策略的效果。

**💡 创新点**

创新点在于：①首次针对VLA模型设计共享场景的非episodic多任务RL基准；②通过终端执行器重置、课程学习重置频率、以及基于不可恢复场景的启发式重置等方法，有效缓解了对象配置漂移和初始位姿偏差导致的性能下降；③证明非episodic训练能显著提升对未见对象位置和任务序列的泛化。

**🔧 技术方法**

采用了PPO算法进行RL微调，结合端到端视觉-语言-动作模型OpenVLA和SpatialVLA；重置策略包括终端执行器重置（EER）、课程学习（CL）和启发式场景重置（HSR）等。

**📊 数据集**

使用Maniskill3物理仿真环境构建的NSR‑Bench基准，包含16个pick‑and‑place任务，分布在4个共享场景中，提供高保真视觉与物理数据。

**📈 对比分析**

与传统频繁重置的episodic RL相比，非episodic微调在未见配置下的成功率提升约20–30%，在连续任务序列上也实现了显著性能提升，验证了更高的交互效率和泛化能力。

**⚠️ 局限性**

局限性包括：启发式场景重置依赖仿真器的内部状态，未来需开发基于感知的检测器；场景漂移在长重置间隔下仍影响学习；基准尚未覆盖接触丰富或高度动态的任务。

---

## 625. One Strategy Does Not Fit All: Rethinking Data Labeling for Just-in-Time Vulnerability Prediction

**arXiv ID:** 2611.05013 | [PDF](https://arxiv.org/pdf/2611.05013v1)

**作者:** Quan Tran `[一作]` (Phenikaa University), Quyet-Thang Huynh `[通讯]` (Hanoi University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文开展了一项大规模实证研究，系统评估了不同VFC与VIC标注策略对Just‑In‑Time漏洞预测（JIT‑VP）在分类、优先级排序与定位三项任务中的影响。

**💡 创新点**

创新点在于证明标注策略对不同下游任务的影响各异，并提出按任务需求定制数据集构建方案，指出最优标注组合并公开相应数据。

**🔧 技术方法**

技术手段包括四种VFC标注策略（手工、精准先行、数量先行、LLM辅助）与四种VIC标注策略（手工、漏洞特定、缺陷泛化、LLM辅助），构建28个数据集；同时对六种JIT‑VP模型（LR、LAPredict、TLEL、DeepJIT、SimCom、JITFine）进行训练与评估。

**📊 数据集**

使用的基准数据来自OpenSSL和Linux内核的提交历史，构造28个标注配置；并手工注释了161条漏洞相关代码行，用于定位任务。

**📈 对比分析**

评价方法采用MCC、PR‑AUC、VDR、MCR、MRR等指标，配合Wilcoxon符号秩检验和Cliff’s δ效应量进行统计比较；结果显示手工VFC+漏洞特定VIC在分类与优先级排序上表现最佳，精确先行VFC+LLM辅助VIC在定位任务上表现最优，且不同策略的性能差异显著。

**⚠️ 局限性**

局限性包括仅覆盖两大C/C++项目且对Linux内核做了时间窗口截断，标注噪声与稀缺性导致结果可能不具普适性，且行级定位数据规模有限，模型对多行交互型漏洞的捕捉仍有待提升。

---

## 626. Do Agentic Policies Need to Look at Every Step? Geometric Abstraction for Reasoning and Action

**arXiv ID:** 2611.05058 | [PDF](https://arxiv.org/pdf/2611.05058v1)

**作者:** Yihang Liu `[一作]` (Independent Researcher), Mingzhe Zheng `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究机器人操作中是否需要每一步都重新解析视觉观测，并提出将空间推理转移到文本和符号的3D几何抽象接口。

**💡 创新点**

将物体部件的类型化、证据感知的符号引用与局部RGB‑D重定位相结合，形成可在语言模型中直接使用的文本检查点。

**🔧 技术方法**

基于RGB‑D的局部几何重定位、符号引用与几何算子、以及大型语言模型进行策略生成与在线决策。

**📊 数据集**

仅在SO‑100机器人模拟环境中使用随机摆放物体的场景数据，未使用公开数据集。

**📈 对比分析**

与仅基于像素的推理方法对比，几何抽象方法在离线策略学习和在线连续执行中在任务成功率、token 量和模型时间上均显著优于基线。

**⚠️ 局限性**

实验仅在模拟台面操作中验证，缺乏对真实世界噪声、遮挡、复杂形状等场景的鲁棒性评估。

---

## 627. ALBI: LLM-Guided Inductive Logic Programming via Iterative Generate-Test-Constrain Refineme

**arXiv ID:** 2611.05039 | [PDF](https://arxiv.org/pdf/2611.05039v1)

**作者:** Aboubaker Seddiq Benterki `[一作]` (Tallinn University of Technology), Leelo Aino Jürgenstein `[通讯]` (Tallinn University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ALBI 框架，将大型语言模型（LLM）作为生成假设的启发式引擎，结合 SWI‑Prolog 进行形式化验证，构建了 neuro‑symbolic ILP 的 generate‑test‑constrain 循环，并引入语义匿名化作为元测试方法。

**💡 创新点**

创新点包括：① 用 LLM 替代 ASP 生成器，显著降低搜索迭代次数；② 结构化 prompt 与 beam search 结合、MDL 评分、锁定子句机制以及分治优化，提升学习效率；③ 通过语义匿名化验证模型真正进行结构化符号归纳，而非表面名称匹配。

**🔧 技术方法**

技术手段包括：LLM（Gemini 3.1 Pro、GPT 5.4 Pro、Claude Opus 4.6）生成 Prolog 假设；SWI‑Prolog 作为评测器；Beam search 与 MDL 评分的候选筛选；迭代反馈提示；锁定子句与分治优化；语义匿名化的元测试。

**📊 数据集**

使用 30 个来自 Popper 基准集的 ILP 任务，涵盖程序合成、IGGP 游戏、Zendo 空间推理、机器人控制等领域，并分别在语义与匿名化两种版本上评测。

**📈 对比分析**

与 Popper（ASP 生成器）比较：三款 LLM 在 ALBI 下均实现 100% 成功率，平均迭代次数仅为 1.4–1.7 次（Popper 平均 6.8 次）；在匿名化实验中成功率保持 100%，迭代次数仅提升 1.36–1.67 倍；运行时间虽然高于 Popper（约 333–1632 秒 vs 82 秒），但迭代成本显著降低，且得到的假设规模与 Popper 相当。

**⚠️ 局限性**

局限性包括：① 无完备性保证，LLM 可能在未知结构上失败；② 依赖 LLM API，版本与成本不稳定；③ 处理时间与成本高于纯 ASP 方法；④ 分治优化有时会错误简化正确假设；⑤ 目前仅适用于无噪声数据，未加入噪声鲁棒性；⑥ 使用更弱的偏置，可能导致搜索空间过大或失去某些结构性约束。

---

## 628. Whose Effect Is It? How Objectives Inflate Learned Subspace Interventions

**arXiv ID:** 2611.05033 | [PDF](https://arxiv.org/pdf/2611.05033v1)

**作者:** Toygar Tanyel `[一作]`, Serkan Ayvaz `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在语言模型中学习得到的子空间交互（subspace interchange）效果为何有时会超过直接替换整个状态的效果，并分离出网络本身的贡献与训练目标导致的“膨胀效应”，为解释和评估可解释性方法提供了理论与实践依据。

**💡 创新点**

创新点在于：① 证明了在一阶近似下，子空间交互的效果可视为对齐到整个状态效果的分解，并给出了效果范围与上限；② 明确指出“更大效果”往往源自训练目标而非网络自身；③ 通过角度预测最大效果，验证跨模型和任务的一致性；④ 提出三步报告检查（OwnEffect）以统一比较学习到的效果与整个状态补丁的基准。

**🔧 技术方法**

主要技术包括：一阶Taylor展开与Rayleigh范数分析、特征值/特征向量理论、交叉熵与平方误差目标的对比、注意力冻结实验、以及针对不同层级的子空间交互（Distributed Alignment Search）与对齐训练。

**📊 数据集**

实验使用了多种公开数据集，包括CausalGym 29 任务、RAVEL、MIB、Sentiment模板、SST句子、IOI、数值一致性等，涵盖124M–32B参数的六个模型族（Pythia、Qwen、Llama、Gemma 等）。

**📈 对比分析**

通过与整状态补丁（whole‑state patching）和其最佳上下文补丁（G*）的对比评估，发现交叉熵目标的 DAS 在读数层通常超过 G，且在网络内部大部分单元也能产生显著的“膨胀”效果；相反，平方误差或 CDAS 目标则更接近或不超过 G。整体性能上，DAS 的效果范围与最大值随模型规模增长。

**⚠️ 局限性**

局限性包括：① 仅在一阶近似下推导，未考虑高阶非线性影响；② 对内部网络的引用仅限于单层/单位置，未覆盖更复杂的跨层组合；③ 实验仅覆盖基线未微调模型，缺乏对指令微调或大模型的验证；④ 关注子空间交互，未直接解释其他可解释性方法如激活取样、对齐投影等。

---

## 629. Finite Spectral Depth Controls Gaussian Entrance Rank and Avoidance of the First Gradient-Flow Saddle

**arXiv ID:** 2611.05022 | [PDF](https://arxiv.org/pdf/2611.05022v1)

**作者:** Yanlin Long `[一作]` `[通讯]` (Independent Researcher), Yanlin Long (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

研究在二维线性网络（正定矩阵分解）中有限次数的Newton–Schulz迭代如何影响训练动态，尤其是早期秩选择与避免梯度流首次非零鞍点的能力。

**💡 创新点**

提出“谱深度”(finite spectral depth)概念，证明更大深度可改变早期选择的秩并使轨迹跳过梯度流的第一个非零驻点；给出一个充分阈值，并给出阈值为四的具体示例。

**🔧 技术方法**

使用连续时间优化动力学、Gaussian 小初始化、Newton–Schulz多项式迭代、谱正则化、残差惯性分析、解析比较与入口定理等技术。

**📊 数据集**

在一个人工的二阶激活两层网络上实验，数据集为 24 个样本，维度 m=r=3，样本坐标与目标值在论文中详细给出。

**📈 对比分析**

与普通梯度下降、有限深度Newton–Schulz（K=1…6）以及极值（polar）更新进行对比。结果显示：梯度下降在靠近第一个鞍点时会出现“肩部”特征；当深度≥4时，Newton–Schulz 能够绕过该鞍点，直接逼近更低损失；所有方法最终都可达到接近最优损失。

**⚠️ 局限性**

局限性：仅在简单的正定矩阵分解（两层二次激活）下证明；需要特定的谱分离与小初始化假设；未覆盖非线性激活、深层网络或高维真实数据集；残差惯性阈值等条件相对强硬。

---

## 630. Rethinking GA2Ms: performance and out-of-distribution generalisation

**arXiv ID:** 2611.05012 | [PDF](https://arxiv.org/pdf/2611.05012v1)

**作者:** George Dimitriadis `[一作]`, Spyridon Samothrakis `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文对 GA^2M（Generalized Additive Models with Gaussian Processes）的内外部分布性能进行评估，并提出 CoGA^2M 通过子样条线性组合增强模型表达力。

**💡 创新点**

创新点在于将 1D 样条拆解为多个子样条并学习权重组合，同时在多数据集上系统比较 OOD 泛化，展示了 GA^2M 在分布漂移下的潜力。

**🔧 技术方法**

技术包括 CoGA^2M、传统 GA^2M、LightGBM 回归器、kNNR 与 νSVR 生成 OOD 数据、R^2 评价指标及子样条可视化。

**📊 数据集**

使用了 55+ 个公开回归数据集，如 airfoil‑self‑noise、fifa、stock、concrete‑compressive‑strength、house‑sales、CPU、Wine‑Quality 等。

**📈 对比分析**

通过在每个数据集上计算 CoGA^2M 与 LightGBM 的 ID 与 OOD R^2，并使用 kNNR 或 νSVR 生成 OOD 样本，结果表明 CoGA^2M 在多数数据集上保持或优于 LightGBM 的 OOD 预测性能，整体上表现更稳定。

**⚠️ 局限性**

局限性包括：在严重分布偏移时 OOD R^2 仍显著下降；模型训练成本高；仅在回归任务中验证；OOD 生成方法可能不覆盖真实场景。

---

## 631. ForgeKernel: Self-Evolving Kernels Catching and Surpassing Experts and Agents

**arXiv ID:** 2611.04998 | [PDF](https://arxiv.org/pdf/2611.04998v1)

**作者:** Yaojian Chen `[一作]` (Tsinghua University), Zhiyuan Liu `[通讯]` (Tsinghua University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一套自我进化的GPU kernel生成系统ForgeKernel，能够在不同算子和硬件上自动生成高性能kernel，并在多轮运行中积累并重用硬件优化经验。

**💡 创新点**

创新点在于：①将架构设计与代码实现分离，架构代理根据性能反馈可全局重构工作划分、数据流和流水线；②采用自我进化机制，即把已验证的kernel作为下一轮参考，让优化从结构层面重新开启；③构建可验证的硬件机制知识库，跨算子、跨输入共享验证过的优化策略。

**🔧 技术方法**

使用技术包括：大型语言模型（Claude Fable 5.1 与 Claude Opus 5）驱动的架构代理和代码代理；SASS反汇编与NCU profiling；硬件参考测量与机制验证；里程碑式的计划与回溯；以及自我进化与知识库更新的闭环。

**📊 数据集**

数据集涵盖：NVIDIA Hopper H100 与 Blackwell B200 GPU 上的注意力、矩阵乘法、卷积、GDN、DSA 等算子；Ascend 平台上 Season 2023 Level‑4 公共基准中的全部算子；以及公开排行榜（SOL‑ExecBench FlashInfer‑Bench、Season 2023 Final Leaderboard）评测数据。

**📈 对比分析**

对比方法：在每个算子、输入尺寸和硬件配置上与专家库（FlashAttention‑4、DeepGEMM、cuBLAS、cuDNN 等）以及其他生成系统（K‑Search、Kernel Design Agents、CAKE）进行性能跑分；ForgeKernel 在 Hopper/Blackwell 上多配置实现匹配或超越专家实现，几乎所有配置均取得提升；在 Ascend 上对所有 Level‑4 公共基准的每个算子平均提升约 10‑20%；在 SOL‑ExecBench FlashInfer‑Bench 及 Season 2023 Final Leaderboard 上均排名第一。整体几何平均加速达到数倍。

**⚠️ 局限性**

限制：①架构搜索受限于代理生成能力，可能停留在局部最优；②知识库仅适用于特定硬件架构，跨硬件迁移需额外适配与验证；③自我进化与知识库维护需要额外计算资源和时间；④系统复杂度高，调试与可解释性挑战较大。

---

## 632. Sparse Anchors for Visual Multi-Session Registration in Changing Environments

**arXiv ID:** 2611.05084 | [PDF](https://arxiv.org/pdf/2611.05084v1)

**作者:** Yunus Talha Erzurumlu `[一作]` (University of Bonn), Cyrill Stachniss `[通讯]` (University of Bonn)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种针对结构变化环境（如建筑工地）的视觉多会话注册方法，利用稀疏可靠的绝对锚点和序列一致性实现跨会话轨迹对齐。

**💡 创新点**

创新点在于：①将视觉检索与几何验证相结合，仅保留在空间上分散且一致的绝对锚点；②使用序列一致性筛选全局对齐；③对保留锚点进行特征度量精细化，并通过姿态图优化将锚点信息传播到整条轨迹。

**🔧 技术方法**

采用的技术包括：视觉放置识别（MegaLoc）、局部特征匹配（LoMa‑B）、全景多相机的几何验证（Generalized PnP + RANSAC）、序列一致性筛选、基于高斯渲染的特征度量优化、以及姿态图优化（PGO）。

**📊 数据集**

实验使用HILTI26建筑工地数据集，涵盖八个月跨度的结构变化，构建15个跨会话注册实验。

**📈 对比分析**

与HLoc、FPFH+ICP、ColabSfM、Register Any Point等基线对比，本文在1 m召回率达94.4%，5 m召回率100%，平均平移RMSE 0.60 m，旋转RMSE 1.20°，显著优于所有基线。

**⚠️ 局限性**

局限性包括：当缺乏足够分散的锚点时会直接拒绝对齐；依赖于参考地图的预先构建；对极端纹理缺失或完全结构重排的场景仍可能失效；计算量较大，尤其是特征度量和PGO步骤。

---

## 633. Enabling risk and reliability analysis in next-generation computational tools via multifidelity methods

**arXiv ID:** 2611.05087 | [PDF](https://arxiv.org/pdf/2611.05087v1)

**作者:** Anirban Chaudhuri `[一作]` (University of Texas at Austin), Karen Willcox `[通讯]` (University of Texas at Austin)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述多阶（multifidelity）方法在风险与可靠性分析中的应用与发展，系统总结了多阶控制变量、多阶重要性抽样和多阶主动学习三类核心技术，并探讨了其在数字孪生等前沿应用中的前景。

**💡 创新点**

创新点在于把三类多阶技术统一框架化，阐明它们的目标、适用场景与互补关系；提出信息重用、梯度估计和多维资源分配等未被充分探讨的未来研究方向。

**🔧 技术方法**

使用的技术包括：多阶蒙特卡洛（MFMC）、多阶控制变量（ACV、MLBLUE）、多阶重要性抽样（MFIS、MFCE）、多阶高斯过程主动学习（co‑Kriging、自回归模型）以及与数字孪生、优化不确定性（OUU）等结合的方法。

**📊 数据集**

本文为综述，无具体数据集；仅引用已发表研究中的实验案例和理论分析。

**📈 对比分析**

方法比较主要基于理论分析与文献中的实验结果，强调多阶技术在降低方差、提升置信尾部估计精度方面的显著优势；但缺乏统一的实证对比，具体性能提升因应用场景和模型结构而异。

**⚠️ 局限性**

局限性包括：缺乏系统的跨领域实证验证；多阶模型之间的统计相关性假设易受实际情况影响；信息重用与梯度估计仍处于理论探讨阶段；高维设计空间中多维资源分配的最优策略尚未形成成熟框架。

---

## 634. On the Learnability of Footwear Robustness: Footstep Recognition Under Unseen Conditions

**arXiv ID:** 2611.05049 | [PDF](https://arxiv.org/pdf/2611.05049v1)

**作者:** Robyn Larracy `[一作]` (University of New Brunswick), Erik Scheme `[通讯]` (University of New Brunswick)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对在不同鞋类条件下的脚步识别问题，提出了一种双流时空网络BiStepNet，能够在用户仅以一种鞋类进行登记时，零样本识别用户在任意鞋类下的身份。

**💡 创新点**

创新点在于将空间与时间特征解耦为两个可独立训练的流，利用阶段式训练提升时间流的跨鞋鲁棒性；通过训练集鞋类多样性与多数据集对比验证鞋类可迁移学习。

**🔧 技术方法**

采用监督对比损失、AdamW优化器、分离式卷积+GRU编码器、门控融合、EMA权重；并对时间序列使用GRU、空间使用CBAM增强卷积网络。

**📊 数据集**

主要使用StepUP‑P150高分辨率足部压力数据集（150人，4种鞋类），并在多数据集实验中加入ForceID‑A、GaitRec、Bialystok、Gutenberg、MFID‑200等辅助数据集。

**📈 对比分析**

与三种3D时空基线（3D CNN、R(2+1)D CNN、3D CNN‑GRU）比较，BiStepNet在BF→BF、BF→SF、BF→PF场景下分别实现97.2/1.9%、83.4/4.9%、65.2/9.1%的rank‑1/EER，明显优于基线；鞋类扩展实验显示鞋类样本越多跨鞋识别越好，多数据集训练可进一步提升。

**⚠️ 局限性**

局限性包括：仍未达到BF→BF水平的跨鞋识别精度；仅使用裸足登记，未评估其他登记鞋类；实验中对速度、负载等其他协变量影响有限；数据集规模受限，需更大规模多样化鞋类样本。

---

## 635. The Power of Interaction in Locally Edge-Private Subgraph Counting

**arXiv ID:** 2611.05038 | [PDF](https://arxiv.org/pdf/2611.05038v1)

**作者:** Xian Chen `[一作]` (University of Science and Technology of China), Pan Peng `[通讯]` (University of Science and Technology of China)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在交互式局部边差分隐私（LEDP）下的子图计数问题，目标是在保持边隐私的情况下，近似计算固定模式图H在任意n个顶点的无向图中的副本数量。

**💡 创新点**

提出了线性噪声消除（LNC）和预测与修正（PC）两种算法，展示了交互式算法在最坏情况下的准确性可以显著优于非交互式算法，首次证明了交互式和非交互式LEDP子图计数之间的分离。

**🔧 技术方法**

使用了交互式局部边差分隐私（LEDP）技术，结合线性噪声消除和预测与修正框架，设计了两轮和（ℓ+2）轮的算法。

**📊 数据集**

使用了任意n个顶点的无向图作为数据集，针对固定模式图H进行计数，考虑了最大度数不超过公共界限d的图。

**📈 对比分析**

与之前的非交互式算法相比，LNC在三角形计数中将最坏情况下的加性误差从Ω(n^2)降低到O(√(n)d)，PC算法在最大度数限制下也实现了类似的改进，证明了其在n和d上的最优多项式依赖性。

**⚠️ 局限性**

算法的局限性在于在某些情况下，仍然可能存在较大的误差，尤其是在处理复杂图结构时，且在实现过程中可能需要较高的计算资源。

---

## 636. Comparing model representations and fitting methods for port-Hamiltonian system identification

**arXiv ID:** 2611.05031 | [PDF](https://arxiv.org/pdf/2611.05031v1)

**作者:** Sølve Eidnes `[一作]` (SINTEF), Håkon Noren Myhr `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对端口哈密顿系统进行数据驱动识别，比较神经网络、解析函数及其混合表示，并对两类拟合方法（梯度优化与稀疏回归）进行实验评估。

**💡 创新点**

创新点在于系统性比较不同表示与拟合方法，并提出混合模型能兼顾解析可解释性与神经网络的表达能力。

**🔧 技术方法**

使用端口哈密顿模型、梯度优化（Adam）、稀疏回归、中心差分与中点离散化等技术。

**📊 数据集**

使用基于两质量系统和非线性电路的模拟数据集，包含无噪声、1%噪声、5%噪声三种噪声级别，并对每种噪声做平滑处理。

**📈 对比分析**

通过在训练、验证、测试集上计算NRMSE和拟合耗时进行比较，结果显示混合模型在多数场景下误差最低，梯度方案耗时较短，平滑与roll‑out方案可进一步提升精度。

**⚠️ 局限性**

局限性包括对候选函数库敏感、未对Hamiltonian施加全局下界导致部分模型在超出训练区间时失效、拟合时间受限且仅针对模拟数据。

---

## 637. Can saliency maps flag chest radiograph classifier errors at a new hospital? An external validation study

**arXiv ID:** 2611.05011 | [PDF](https://arxiv.org/pdf/2611.05011v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 638. Haptic Shared Control for Multi-Robot Systems

**arXiv ID:** 2611.05027 | [PDF](https://arxiv.org/pdf/2611.05027v1)

**作者:** Maxime Bernard `[一作]` `[通讯]`, Maxime Bernard

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

论文内容缺失，无法确定具体实现

**💡 创新点**

无法确认创新点

**🔧 技术方法**

暂无技术细节

**📊 数据集**

未提供数据集信息

**📈 对比分析**

无法比较方法与性能

**⚠️ 局限性**

缺乏足够信息导致总结受限

---

## 639. RGB Photo Enhancement with Deep Learning on Smartphone GPUs: Mobile AI 2026 Challenge Report

**arXiv ID:** 2611.05024 | [PDF](https://arxiv.org/pdf/2611.05024v1)

**作者:** Andrey Ignatov `[一作]`, Aiwen Jiang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文介绍了 Mobile AI 2026 RGB 图像增强挑战赛，目标是在智能手机 GPU 上实现高质量、低延迟的全高清照片增强，提供了实验流程、数据集、评测指标及参赛方案。

**💡 创新点**

创新点在于通过教师‑学生知识蒸馏解决 DPED 数据集像素对齐问题，结合 RetinexFormer 与精简的 PLUnet 架构，并在同一赛道中首次大规模比较多种轻量 Transformer 与 UNet 变体在手机 GPU 上的性能。

**🔧 技术方法**

使用的技术包括多尺度 L2 + MS‑SSIM + VGG 内容损失、FCBlock 剔除与通道剪枝、TFLite GPU delegate、litert‑torch 直接将 PyTorch 模型转换为 TFLite，以及基于 PSNR、MS‑SSIM 与 MOS 的复合评价体系。

**📊 数据集**

采用 DPED 数据集中的 iPhone 3GS 与 Canon 70D DSLR 级联照片对，提供 22k 张高分辨率成对图像，用于训练与评估。

**📈 对比分析**

比较方法：在 Snapdragon 8 Elite Gen 5 与 Dimensity 9500 两款旗舰 SoC 上，使用 AI Benchmark 测得 PSNR、SSIM、MOS 与 GPU 运行时，所有参赛模型均在 300 ms 内完成 1920×1088 的增强，冠军 DaHua‑IIG 以 23.8 ms/60.4 ms 的跑时和最高的 PSNR/SSIM/MOS 夺冠。

**⚠️ 局限性**

局限性包括 DPED 数据对齐不完全导致训练噪声，Transformer 结构在保持高质量时仍显高延迟，NPU 兼容性有限，以及模型在不同硬件平台间的可移植性与量化精度差异。

---

## 640. Faster Planar Diameter using Pseudoline Envelopes

**arXiv ID:** 2611.05019 | [PDF](https://arxiv.org/pdf/2611.05019v1)

**作者:** Da Wei Zheng `[一作]` (Institute of Science and Technology Austria), Daniel Prigan `[通讯]` (Reichman University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种新的 O(n^{3/2}) 时间算法，用于计算带权有向平面图的直径和所有顶点的远心。

**💡 创新点**

创新点在于证明归一化距离曲线构成弱伪直线族，从而能够利用伪直线上界和范围搜索构造高效的“最远顶点”数据结构。

**🔧 技术方法**

采用了 Voronoi 图、可多源最短路径（MSSP）、r‑划分、距离预处理、伪直线上界与范围搜索等技术。

**📊 数据集**

未使用实际数据集，而是在理论平面图模型上进行分析和证明。

**📈 对比分析**

与之前的 O(n^{5/3}) 和 O(n^{8/5}) 结果相比，本文将时间复杂度降低到 O(n^{3/2})，已逼近平面图问题的天然上界。

**⚠️ 局限性**

仍无法达到 n^{1+o(1)} 的下界，且结构性伪直线性质目前仅适用于平面图，未扩展到更一般的图或动态更新。

---

## 641. Manifold Embedding of Deep Image Features for Image Matching via Neural Adjoint Maps

**arXiv ID:** 2611.05083 | [PDF](https://arxiv.org/pdf/2611.05083v1)

**作者:** Nafie El Amrani `[一作]` (University of Bonn), Florian Bernard `[通讯]` (Lamarr Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于功能映射的零样本图像对应方法，将深度特征嵌入二维流形并通过Galerkin求得LBO，随后训练轻量级神经相位（NAM）实现非线性对齐

**💡 创新点**

创新点在于：①用高维深度特征作为二维流形顶点坐标，借助Galerkin法自动构造无尺度参数的LBO；②将功能映射框架扩展到图像匹配，并引入神经相位以弥补谱截断的非线性损失；③实现零样本、全局光滑且可直接迁移至医学影像

**🔧 技术方法**

技术包括：基础模型特征提取（DINOv2/Stable Diffusion等）、Galilean有限元求LBO、功能映射与神经相位（NAM）对齐、最近邻搜索、可选的Neural ZoomOut细化

**📊 数据集**

数据集：TSS（dense）、SPair-71k（sparse）以及自制的3D形状衍生合成基准

**📈 对比分析**

与基准（FUSE、Fuse2Match、CleanDIFT、Cheng等）比较，零样本下在TSS的EPE、PCK等指标均达到或超过现有最高水平；在SPair-71k上PCK与MSE均与或优于CleanDIFT、Fuse2Match，且显著更平滑

**⚠️ 局限性**

局限性包括：对完整对应假设敏感，难以处理严重遮挡或大姿态差异；流形嵌入受特征分辨率限制，可能出现对称性导致映射翻转；未针对部分到部分匹配的情况进行优化

---

## 642. Characterizing Serverless Workloads: A Large-Scale Analysis of Function-as-a-Service Platforms

**arXiv ID:** 2611.05081 | [PDF](https://arxiv.org/pdf/2611.05081v1)

**作者:** Saman Akbari `[一作]` (Technische Universität Berlin), Manfred Hauswirth `[通讯]` (Technische Universität Berlin)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对八个公开 FaaS 工作负载追踪进行大规模跨平台对比分析，揭示服务器无状态函数的请求模式、冷启动行为、执行时长及资源利用特征。

**💡 创新点**

首次系统性比较多云供应商与科学计算平台的工作负载，指出执行时间缩短导致调度与依赖管理成为主要瓶颈，发现 Serverless 2.0 并发容器使用增加，资源过度预留和请求量极端偏斜。

**🔧 技术方法**

利用 Dask 并行计算框架对原始追踪进行清洗、归一化，构建统一的度量体系（请求率、执行/响应时间、冷启动、资源利用、触发器、依赖库等），并进行统计与可视化分析。

**📊 数据集**

八个公开数据集：Azure（2019/2020/2021）、Alibaba、Huawei（公开/私有）、Globus Compute、IBM Cloud Code Engine，涵盖 2000 万+ 调用事件。

**📈 对比分析**

通过对比关键维度（burstiness、冷启动率、执行时间分布、资源占用）衡量不同平台的性能差异，发现现代云平台的冷启动中位数已降至 <1 s，但极端尾部仍达数十秒；Serverless 2.0 通过并发容器将实例峰值从 90+ 降至 6。

**⚠️ 局限性**

局限性包括：只能观察外部度量，无法区分平台优化与真实工作负载差异；各数据集代表性不一，科学与商业平台工作负载差异大；追踪数据更新不及时，可能已过时。

---

## 643. Finding Hypercliques Faster Than Brute Force

**arXiv ID:** 2611.05082 | [PDF](https://arxiv.org/pdf/2611.05082v1)

**作者:** Josh Alman `[一作]` (Columbia University), Ryan Williams `[通讯]` (MIT)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对任意固定的 k≥3，提出并证明了一种比朴素 O(n^k) 更快的算法，能够在 O(n^k-ε_k) 时间内计数 k‑超团（k‑hyperclique）并解决相关高阶乘法与 CSP、等差数列检测等问题。

**💡 创新点**

创新点在于：① 用“分层”分解超团多项式的低/高次数分量并引入分解张量的“partition rank”；② 采用部分转置 (partial transpose) 与 Baur–Strassen 定理将高维问题递归到低维；③ 通过图论分块把低次数部分的稀疏矩阵乘法拆解成可并行的大块乘法，最终实现统一可构造的算术电路大小 O(n^k-ε_k)；ε_k 递归产生 2^-Θ(2^k) 的指数提升。

**🔧 技术方法**

主要技术包括：张量秩与 partition rank 分析；矩阵与张量的 Kronecker 乘积与子多项式递归；部分转置 τ_p 的对角化与正交基 ℬ^⊗d；Baur–Strassen 计算导数；稀疏矩阵乘法与分块大块乘法；高低次数阈值分割与双分解 (dichotomy) 处理；以及利用快速矩阵乘法的已知 ω<3 的常数。

**📊 数据集**

本工作为理论算法，未使用任何实验数据集；所有结果均基于符号运算与抽象计算，适用于任意可交换环（如奇素数域或整数模大奇数）。

**📈 对比分析**

与原先的 k‑Hyperclique 假设（认为 O(n^k-ε) 的算法不存在）相比较，本文提供了更快的算法，时间复杂度下降到 O(n^k-ε_k)（ε_k>0）。此外，该算法也能在 O(n^2-ε_k) 时间内检测 k‑等差数列、在 O(2^{(1-δ_k)n}) 时间内解决 Max‑Degree‑kCSP 等问题，均实现了此前被认为不可能的速度提升。

**⚠️ 局限性**

局限性包括：① ε_k 的大小随 k 指数衰减（约 2^-Θ(2^k)），使得在实际中只有极小的加速；② 仅适用于固定 k 的情况，k 变大时提升极小；③ 构造的电路系数在 ℤ[1/2] 内，虽然可构造但常数可能较大；④ 证明依赖于已有的快速矩阵乘法 ω<3，若 ω 接近 2 仍难以获得更大收益；⑤ 目前仅给出理论复杂度，尚未给出可落地实现或对大规模实例的实验验证。

---

## 644. Nested Field-of-View Barriers for Teleoperated Multi-Robot Relay Chains

**arXiv ID:** 2611.05074 | [PDF](https://arxiv.org/pdf/2611.05074v1)

**作者:** Arthur Vangasse `[一作]` (Federal University of Minas Gerais), Luciano Pimenta `[通讯]` (Federal University of Minas Gerais)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一种共享自治框架，在通信受限环境下通过一条连线式多机器人链路保持人机遥控的连通性。

**💡 创新点**

创新点是引入嵌套硬软视场约束——严格完整视场保障连通性，收缩视场提供跟随驱动力，解决连锁停滞问题。

**🔧 技术方法**

采用控制障碍函数（CBF）与局部二次规划（QP）实现分布式控制，并利用签名距离描述视场与碰撞约束。

**📊 数据集**

使用了两套仿真系统：纯运动学整合器和软件‑在‑环（SIL）Crazyflie 四旋翼；工作空间为“隧道”和“杂乱障碍”两种占据网格地图。

**📈 对比分析**

通过对比“严格”“松弛”“嵌套”三种控制器，实验显示嵌套控制器在保持连通性、减少跟随者运动量和保持到达时间方面均优于基线，且连接误差保持在毫米级别。

**⚠️ 局限性**

局限性包括理论证明假设完美速度跟踪、离散采样与视场投射误差导致的残留违规、以及未考虑通信延迟与丢包的真实网络条件。

---

## 645. Trajectory Evaluation via Rollout-Free World Model for End-to-end Autonomous Driving

**arXiv ID:** 2611.05072 | [PDF](https://arxiv.org/pdf/2611.05072v1)

**作者:** Junzhe Wang `[一作]` (Beihang University), Qingjie Liu `[通讯]` (Beihang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种无rollout的共享场景记忆模型RFWorld，用于端到端轨迹评估。

**💡 创新点**

创新点在于把候选轨迹评估从逐个rollout转为一次性构建共享记忆并通过稀疏空间读取来获取未来相关信息。

**🔧 技术方法**

使用Transformer基础的隐式世界模型、时序读取与语义监督、变形稀疏注意力读取以及轨迹奖励头等技术。

**📊 数据集**

在NAVSIM-v1真实传感器数据集和CARLA Bench2Drive闭环仿真集上训练和评估。

**📈 对比分析**

与现有SOTA端到端方法对比，RFWorld在NAVSIM的PDMS、NC、TTC等指标上实现90.5分，在Bench2Drive的成功率31.36%、驾驶分数62.38，优于TCP等基线。

**⚠️ 局限性**

局限包括Transformer对高分辨率未来预测不够高效、共享记忆依赖语义监督、以及对感知误差和观测范围之外物体的鲁棒性不足。

---

## 646. Dataset Recommendation for Heterogeneous and Dynamic Scholarly Graphs via Multimodal Representation Learning

**arXiv ID:** 2611.05071 | [PDF](https://arxiv.org/pdf/2611.05071v1)

**作者:** Ornella Irrera `[一作]` (University of Padua), Gianmaria Silvello `[通讯]` (University of Padua)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出一种多模态异构图学习模型MM‑SAN，用于在学术知识图中对相关数据集进行推荐。

**💡 创新点**

创新点在于将文本与图结构信息通过交叉注意机制融合，同时利用实体与主题节点对稀疏图进行增强，支持转导、半归纳和完全归纳三种推理场景，显著提升稀缺文本下的鲁棒性。

**🔧 技术方法**

采用的技术包括异构图表示学习、随机游走采样、交叉注意聚合、句子Transformer文本编码、BERTopic主题建模以及DBpedia Spotlight实体链接。

**📊 数据集**

实验使用了OAG的MES子图和PubMed子图，这两个子图分别包含数百万节点和边，涵盖论文、数据集、作者等多种实体。

**📈 对比分析**

与六类基线（GraphSAGE、GAT、KGAT、HAN、HGT、HetGNN）以及文本基线ST‑T、HVGAE、VGAE等在Recall、MRR、nDCG上进行对比，MM‑SAN在所有三种推理模式和不同文本可用比例下均优于或接近最佳基线，尤其在稀疏与归纳场景中提升显著。

**⚠️ 局限性**

局限性包括对文本或结构信息的依赖，文本缺失时性能下降；大规模实时更新时的计算开销和多模态融合的可解释性尚待改进；实验仅覆盖OAG子图，未验证在其他学科或多语言环境中的泛化能力。

---

## 647. Equal Access, Unequal Gains? GitHub Copilot and Productivity Among Developers with Disabilities

**arXiv ID:** 2611.05056 | [PDF](https://arxiv.org/pdf/2611.05056v1)

**作者:** Emeralda Sesari `[一作]` (University of Groningen), Ayushi Rastogi `[通讯]` (University of Groningen)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了 GitHub Copilot 在自我披露为残疾的开发者中的可观测生产力变化

**💡 创新点**

首次将 Copilot 的可用性与多类型残疾开发者的生产力进行纵向比较

**🔧 技术方法**

使用公开 GitHub 资料、提交日志与 PR 接受率等仓库指标

**📊 数据集**

来自 GitHub 公共 API 的 199 名自我标注残疾的开发者以及 70 名对照

**📈 对比分析**

采用前后差分、Wilcoxon 符号秩检验、Kruskal-Wallis、Mann-Whitney U，并用 Benjamini-Hochberg 调整 p 值；结果多为无显著差异，只有“disabled”组净行数增加显著

**⚠️ 局限性**

样本仅包含公开自报残疾的开发者、AI 使用未必被提交信息捕捉、仓库指标无法反映认知负担和可访问性体验

---

## 648. Effects of Parton Pair Correlations on Direct-Photon Yields and Elliptic Flow in Heavy-Ion Collisions

**arXiv ID:** 2611.05042 | [PDF](https://arxiv.org/pdf/2611.05042v1)

**作者:** Xingjian Lu `[一作]` (Yantai University), Shuzhe Shi `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过在重离子碰撞中加入发射部件的两粒子相关性，研究其对直接光子产率和椭圆流 v₂ 的影响，并在 Au+Au 与 Pb+Pb 系统中与实验数据进行对比。

**💡 创新点**

创新点在于构造了零边缘（zero-marginal）连接两粒子相关模式，能够在不改变单粒子分布的前提下直接调节局部 2→2 光子产生率，并通过联合拟合光子谱与 v₂ 来探测这种相关性。

**🔧 技术方法**

主要技术包括 VISH2+1 水动力模拟、Arnold–Moore–Yaffe (AMY) 速率、基于 Laguerre 与 Legendre 多项式的相关性响应库、以及带正定约束的多参数回归拟合。

**📊 数据集**

使用的实验数据集为 PHENIX、ALICE 的直接光子产率与 v₂（Au+Au 200 GeV、Pb+Pb 2.76 TeV），并参考 STAR 的虚拟光子产率数据进行检验。

**📈 对比分析**

对比方法为在基线模型（固定 prompt、pre‑equilibrium 与热光子）上添加相关性贡献后，计算与实验的 χ²；在满足正定约束的硬约束下已显著提升光子产率与 v₂ 的拟合精度，松弛约束时拟合更好但出现局部负速率或分布。

**⚠️ 局限性**

主要局限在于基线与响应使用不同的水动力演化、相关性响应仅限于 leading‑log 近似、相关模式基数有限以及正定约束下模型与数据之间仍存在偏差，限制了对能量与中心度通用性的验证。

---

## 649. Positional Encoding is All You Need for Scalable Equivariance Constraint Relaxation

**arXiv ID:** 2611.05041 | [PDF](https://arxiv.org/pdf/2611.05041v1)

**作者:** Hagay Michaeli `[一作]` (Technion), Daniel Soudry `[通讯]` (Technion)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在训练期间使用可学习的位置信息对严格等变网络进行临时对称性打破，并在训练结束时逐步消除，使模型保持推理时的等变性。

**💡 创新点**

创新点在于用可学习的位置信息(PE)代替昂贵的全连接层进行约束放松，极大降低了计算和内存开销，同时保持了推理时的严格等变性。

**🔧 技术方法**

采用位置信息注入、退火调度、正则化以及学习率等技术实现约束松弛，并在多种等变架构上验证。

**📊 数据集**

在ModelNet40点云分类、ImageNet图像分类、模拟烟雾plume、带电粒子N‑body预测等数据集上进行实验。

**📈 对比分析**

与基准严格等变模型和使用全连接放松的对照相比，PE方法在保持相同参数/训练时间的前提下提升了准确率或降低了RMSE，表现出与全连接放松相当甚至更优的性能。

**⚠️ 局限性**

局限性包括需要为不同架构调节PE类型和退火策略、理论分析仅覆盖concat PE且对add PE支持有限、在已有强基线上提升幅度有限，以及目前仅在输入层注入PE，未探究更深层的放松效果。

---

## 650. Matchings under One-Sided Preferences with Soft Quotas

**arXiv ID:** 2611.05037 | [PDF](https://arxiv.org/pdf/2611.05037v1)

**作者:** Santhini K A `[一作]` (GPM Government College Manjeshwar), Meghana Nasre `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究多对一匹配问题，提出软配额模型：每个职位设有上下目标范围，匹配可超出范围但产生偏差。针对偏好顺序（排名最大化或公平）与偏差最小化（总偏差或最大偏差）两大目标，设计了可交替求解的多种算法。

**💡 创新点**

创新点在于：①首次将软配额与偏好最优结合，形成双目标优化框架；②利用流网络构造可控制上下偏差的边，证明最小费用流即可得到排名最大化/公平的最优匹配；③扩展到带成本的软配额，进一步将偏差成本纳入最小费用模型；④通过增广路径、二分搜索等技术，解决不同优先级顺序的多种变体，且给出NP‑hard性界限。

**🔧 技术方法**

核心技术：基于流网络的最小费用流求解（容量扩展、成本缩放算法）；签名比较与弱优势理论；二分搜索、线性搜索确定偏差或最大偏差范围；增广路径与代价网络实现排名最大化/公平的增量优化；对成本化软配额使用加权边成本的最小费用流。

**📊 数据集**

本文未使用公开数据集，主要通过理论实例与图示（如含七名申请者与七个职位的示例）展示算法流程与结果；对比分析中以理论上限和实例演示验证复杂度。

**📈 对比分析**

与传统固定配额模型相比，本文算法在多目标情形下保持多项式时间复杂度；在可行匹配不存在时给出最小偏差解；通过实例证明在偏差优先和偏好优先两种顺序下均可取得最优解；同时指出当所有配额为 0/1 时问题仍为NP‑hard，展示了算法的限制与对比。

**⚠️ 局限性**

主要局限：①未给出多目标（总偏差+最大偏差）Pareto最优解的完整算法；②对大规模实例求解效率仍受流网络规模影响；③泛化软配额的成本模型导致算法复杂度上升；④对非单侧偏好（两侧偏好）或多重目标（如公平+排名最大化）尚未覆盖。

---

## 651. Federated Transfer Learning for Medical Image Classification under Mixed Skew

**arXiv ID:** 2611.05035 | [PDF](https://arxiv.org/pdf/2611.05035v1)

**作者:** Alexandre G. Leclercq `[一作]` (Université Caen Normandie), Romain Hérault `[通讯]` (Université Caen Normandie)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了一种基于预训练模型的个性化联邦学习框架PAFP，用以解决多中心医学影像分类中的混合偏差问题。

**💡 创新点**

创新点在于：①在每个客户端插入特征对齐模块并使用EMA平滑的全局统计来实现特征空间一致化；②采用可切换的交替优化调度，先完成全局医学域适配再进行局部对齐；③引入轻量级logit校正器以缓解标签偏差。

**🔧 技术方法**

技术方法包括：预训练EfficientNet + FedPEFT思想、组归一化+1×1卷积对齐模块、EMA统计与对齐正则、焦点损失、FedAvg聚合、交替优化调度。

**📊 数据集**

实验使用FedISIC（ISIC 2019/2020）多中心皮肤病变分类数据集。

**📈 对比分析**

与FedAvg、DBE、Ditto、ft-2B等基线在每个客户端的平衡准确率、加权/非加权平均及方差进行比较；PAFP-alt（r=10）在非加权平均上达到最高67.53%并将方差降至0.35，显著优于所有基线。

**⚠️ 局限性**

局限性包括：仅在单一数据集/单模态上验证；概念漂移与数量偏差仅间接处理；对齐模块略增通信开销；预训练模型来自自然图像，域差异可能影响评估。

---

## 652. HiLo-WAM: Decoupling Long-Horizon Task Evolution and Local Dynamics in World Action Models

**arXiv ID:** 2611.05023 | [PDF](https://arxiv.org/pdf/2611.05023v1)

**作者:** Xingyu Wang `[一作]` (CoCoMatrix), Mengdi Xu `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出HiLo-WAM，一种将视觉世界建模分解为稀疏长时序任务演进流和稠密局部交互流的世界动作模型，并通过双DiT架构联合建模与动作。

**💡 创新点**

通过多尺度时间建模将长时序任务演进与局部交互解耦，同时使用结构化跨流注意力和进展式推理，显著提升长时序操纵性能。

**🔧 技术方法**

Dual-DiT架构、Video VAE、流匹配损失、结构化跨流注意力、部分/联合/动态跳步推理。

**📊 数据集**

RoboTwin 2.0、RoboDojo长时序任务、Fourier GR-3机器人实测。

**📈 对比分析**

与Fast-WAM、AHA-WAM、LingBot-VA等基线对比，在RoboTwin 2.0上93.7%成功率，RoboDojo长时序任务平均得分17.43；实测平均成功率33%（内域），30%（OOD），并实现2.93×速度提升。

**⚠️ 局限性**

受限于底层视频模型的语义理解与预测能力，复杂需要显式子任务分解或高层推理的长时序任务仍有挑战。

---

## 653. Nearly-Linear Query Algorithm for Submodular Maximization subject to a Matroid Constraint

**arXiv ID:** 2611.05021 | [PDF](https://arxiv.org/pdf/2611.05021v1)

**作者:** Tatsuya Terao `[一作]` `[通讯]` (Kyoto University), Tatsuya Terao (Kyoto University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种新的组合随机化保护本地搜索算法，针对在矩阵约束下的单调子模函数最大化问题实现了(1-1/e-ε)近似；

**💡 创新点**

该算法在保持(1-1/e-ε)近似的同时，查询复杂度仅为对n的几乎线性次数，并且对1/ε多项式，首次实现了同时满足这两个近似目标的组合算法；

**🔧 技术方法**

利用升维矩阵构造、Buchbinder‑Feldman潜在函数、二分搜索技术、阈值处理和保护机制进行本地搜索，并通过概率估计降低g(x|X-x)的计算量；

**📊 数据集**

该研究主要在理论上验证，不涉及具体实验数据集；

**📈 对比分析**

与之前的连续贪婪、交换取舍、非显式本地搜索等方法相比，该算法在查询复杂度上显著提升，近似因子保持在(1-1/e-ε)，在所有矩阵秩下均可实现几乎线性查询；

**⚠️ 局限性**

算法仍为随机化，需多次重试保证估计良好；在实现上需要对潜在函数进行采样，导致较大的常数与对1/ε的多项式依赖；

---

## 654. Learned Smartphone ISP on Mobile GPUs: Mobile AI 2026 Challenge Report

**arXiv ID:** 2611.05008 | [PDF](https://arxiv.org/pdf/2611.05008v1)

**作者:** Andrey Ignatov `[一作]`, Haibo Xu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对手机相机 RAW→RGB 任务，举办 Mobile AI 2026 Learned Smartphone ISP on Mobile GPUs 挑战，要求在 Snapdragon 8 Elite Gen 5 与 Dimensity 9500 GPU 上以 100 ms 内实时重建 5 MP RGB 图像。

**💡 创新点**

创新点在于结合大规模 Fujifilm UltraISP 数据集、针对 GPU 的高效网络架构（如 LKSNet、MobileIE、RMFA‑Net 等）以及多损失、多频段训练策略，解决了传统 ISP 在移动端的实时性与视觉质量双重瓶颈。

**🔧 技术方法**

采用的技术包括：端到端 CNN（大核跳跃网络、重参数化 MobileIE、N‑Attention‑Fusion 结构等）、FFT 低高频损失、颜色、LPIPS、SSIM、对抗损失、量化（INT8/FP16）以及 TFLite LiteRT GPU 推理。

**📊 数据集**

使用 Fujifilm UltraISP 数据集，该数据集包含 6k+ 影像对，来自 Sony IMX586 Quad Bayer 传感器与 Fujifilm GFX100 102 MP 中幅相机，经过深度匹配后提取 256×256 的对齐 patch。

**📈 对比分析**

评估采用 PSNR、SSIM、MOS（用户研究）和 GPU 运行时；Top 3 模型（MiAlgo LKSNet、AVC2 MobileIE、UAIC_ISP）在两块 SoC 上均可在 0.5 s 内完成 5 MP 图像重建，MiAlgo 在 Adreno 上可达 60 fps，Dimensity 上 22 fps，MOS 最高 4.8。

**⚠️ 局限性**

局限性：模型仍需在 GPU 上实现较高 FPS，NPU 兼容性有限；数据集规模与场景多样性受限，训练对齐仍需手工；多损失与量化训练对硬件适配复杂，且对低功耗手机仍有进一步提升空间。

---

## 655. Process-zone condensation: a discretization-exact reduced cell model for FE$^2$ analysis of composites with fibre-matrix debonding and matrix plasticity

**arXiv ID:** 2611.05065 | [PDF](https://arxiv.org/pdf/2611.05065v1)

**作者:** Leonilson Toffoli Pedrini `[一作]` (Federal Institute of Espírito Santo), Rafael Marin Ferro `[通讯]` (Federal Institute of Espírito Santo)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种称为“过程区消除（process‑zone condensation）”的数值方法，利用Schur补整合式消除周期单元中线性弹性残余部分，同时保留所有粘性接口变量和保留区内的塑性内部变量，并通过增量后向替代和产生残差的检查（certificate）确保保留区始终包含所有非弹性行为，必要时自动扩展保留区；该方法被实现于二维和三维有限元多尺度（FE²）框架中，并与完整单元求解、训练后的模型以及商业ANSYS求解器进行了对比验证。

**💡 创新点**

核心创新在于：① 在保持完整离散响应的前提下，仅对线性弹性部分进行精确静态消除（不需经验模式或训练）；② 每个接口点与保留区内每个单元的塑性变量均按个体保留，实现离散等价；③ 通过一次后向替代和弹性残余检查实时判定保留区是否充分，缺失时自动增大保留区；④ 该方法可在FE²中共享相同的消除算子，实现多尺度计算的高效重用。

**🔧 技术方法**

技术实现包括：有限元单元周期边界条件、粘性接口（Alfano–Crisfield混合模式）与欧氏弹性塑性（von Mises）耦合；静态消除（Schur补）得到影响矩阵和一致切线；增量Newton迭代与残差评估；增量后向替代检验弹性补全；多尺度FE²驱动器与适应性保留区扩展模式。

**📊 数据集**

使用的数值数据集：
• E‑glass/环氧玻璃纤维/环氧树脂单元格（圆柱含量28%），含已测量的弹性模量、屈服极限、硬化模量、界面强度与能量释放率；
• E‑glass/聚丙烯单元格（同一几何尺寸，但界面参数来自拉伸测试）；
• 二维平面应变板状结构（含孔的四分之一板），用同一单元格重复多次；
• 三维体素化单元格（球形包络），以及多孔集成的多囊单元格。

**📈 对比分析**

性能比较：
• 在二维单元格中，完整路径（除纯剪切外）速度提升约3.3×（排除预处理与检查时间），错误<10⁻¹¹；
• 纯剪切路径未通过检查，导致误差升至10⁻³；
• 在三维体素单元格中，合格路径无明显加速，增大保留区后仍未显著收益；
• 在板状FE²中，玻璃/聚丙烯系统获得≈1.5×加速；玻璃/环氧系统相反，约0.54×（即更慢），主要因非弹性点需推广至宽保留区；
• 与训练模型（POD‑Galerkin、NTFA）相比，过程区消除无训练成本、对新路径有检查机制，单路径耗时略高但训练后总成本可比；
• 与ANSYS相比，单元格响应一致性≤10⁻³，宏观反应一致性≈0.6%。

**⚠️ 局限性**

局限性：
• 仅在保留区外材料保持弹性时有效；若塑性或损伤扩散至外部，需扩展保留区，性能优势随之减弱；
• 适用于小应变、无显式摩擦接触、无裂纹扩展的平面或体素网格；
• 对于高斯或连续介质的非周期性边界、非线性大变形、或需要高阶同质化的情况，需进一步研究；
• 计算量与保留区尺寸m有关，m²的内存占用可能成为三维大模型的瓶颈；
• 目前未考虑材料多尺度耦合的显著尺度不匹配与边界效应。

---

## 656. FloMo: 3D Scene Flow as a World Action Model Intermediate for Learning from Human Video

**arXiv ID:** 2611.05062 | [PDF](https://arxiv.org/pdf/2611.05062v1)

**作者:** Jeremy Collins `[一作]`, Animesh Garg `[通讯]` (Georgia Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并微调一个基于预训练视频生成模型的扩散框架，利用3D场景流和动作共同作为预测目标，实现在多种机器人操控任务中的动作规划。

**💡 创新点**

创新点在于：①将3D场景流渲染为RGB映射后直接编码进已有的视频VAE空间，避免新建tokenizer；②以低熵、与纹理无关的场景流作为中间表示，使视频预训练的运动先验能直接转化为可解码的动作信息；③在同一模型中同时预测场景流与动作，且通过流匹配训练提升样本效率。

**🔧 技术方法**

使用的技术包括：预训练的Wan 2.2 5B DiT、冻结的VAE与UMT5文本编码器、LoRA微调、流匹配（flow‑matching）目标、UniPC ODE解算器、以及基于累积场景流的RGB渲染与离线3D跟踪（SpatialTrackerV2）。

**📊 数据集**

使用的数据集：Egocentric human视频（81 h EgoVerse子集）、机器人交互数据（10 h YAM + 700 h MolmoAct2）以及内部演示（0.7 h无动作标签的人类视频与0.4 h机器人闲置演示）。

**📈 对比分析**

方法与多种基线比较：视频‑动作（VA）、视频‑动作‑场景流（VMA）、Amplify、DreamZero、π_0.5、FAST‑WAM及纯行为克隆。实验显示本方法在分布内任务平均成功率≈91%，分布外任务≈60%，RoboTwin任务平均成功率≈63.6%，均优于对比模型，证明场景流预测是更有效的中间表示。

**⚠️ 局限性**

局限性：①需要为每段视频离线提取3D场景流，耗时；②动作与场景流预测不完全同步，可能导致预测动作与实际运动不一致；③仅使用前视摄像头的场景流，遮挡严重时精度下降；④对未知对象的指令 grounding 效果有限。

---

## 657. Toward Constant-Factor Maximin Share Guarantees for Subadditive Valuations

**arXiv ID:** 2611.05073 | [PDF](https://arxiv.org/pdf/2611.05073v1)

**作者:** Bo Li `[一作]` (Hong Kong Polytechnic University), Ankang Sun `[通讯]` (Shandong University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在n个代理人和m件不可分物品的最大最小份额（MMS）公平分配问题，提出了三项主要结果：1）在一般子加性（subadditive）价值函数下将MMS近似系数从 O(1/ log log n) 提升到 O(1/ log* n)；2）对 q‑partitioning 价值函数（介于子加性与 XOS 之间）给出常数近似，q ≥ n 时得到 1/6-MMS，q ≥ 2n 时得到 1/4-MMS；3）对 XOS 价值函数将全局 MMSC 近似提升至 5/16。

**💡 创新点**

创新点包括：①利用“竞价放大”技术（bidding amplification）实现对单个代理人赢得半份 MMS 的概率指数提升；②将支持价格（supporting prices）与离散删除损失相结合，得到 q‑partitioning 价值函数下的损失上界；③提出分组与舍入（grouping & rounding）以及直接转移（direct transfers）两种构造，分别在不同 q 区间获得常数近似；④开发“预留定理”（reservation theorem）和“阈值”技巧，在 XOS 价值函数下实现 5/16 的分配。

**🔧 技术方法**

技术方法主要包括：
- 基于有限极小化（finite minimax）构造的竞价分布，保证在固定预算下赢得至少一半价值；
- Feige 的舍入定理（Feige’s rounding theorem）将多重分配转化为互斥分配；
- 支持价格与迭代对数层次（iterated‑logarithm layers）结合的层叠构造；
- 支持价格下的离散删除损失上界与标量判据（scalar criterion）用于 q‑partitioning；
- 预留定理与指数随机变量的期望最大值分析，用于 XOS 的 5/16 方案。

**📊 数据集**

该工作为纯理论论文，无实验数据集，所有结果均通过数理证明与极限分析获得。

**📈 对比分析**

与已有工作对比：
- 子加性价值函数的 MMSC 近似从 O(1/ log log n) 提升到 O(1/ log* n)，显著改进迭代对数阶；
- q‑partitioning 价值函数在 q≥n、q≥2n 时分别获得 1/6 与 1/4 的常数近似，覆盖了 XOS 之外的新类；
- XOS 价值函数的 MMSC 近似从 4/17≈0.2353 提升到 5/16=0.3125，优于先前的 11/40≈0.275 及最近的 0.27677 结果。

**⚠️ 局限性**

局限性：
- 对一般子加性价值函数仍无法获得常数近似，仅实现迭代对数级别；
- q‑partitioning 方案的常数近似仅在 q 与 n 的比例满足特定阈值时成立；
- XOS 方案依赖大量价值查询（O(n^4)），在大规模实例上可能计算成本高；
- 证明过程复杂，实际实现和参数调优可能困难；
- 对动态或在线分配场景尚未给出直接扩展。

---

## 658. Graph Similarity Meets Optimal Transport

**arXiv ID:** 2611.05048 | [PDF](https://arxiv.org/pdf/2611.05048v1)

**作者:** Jens Breitung `[一作]` (RWTH Aachen University), Martin Grohe `[通讯]` (RWTH Aachen University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对图的矩阵基图度量（编辑距离、切分距离、本地编辑距离、光谱距离）的最优传输版本进行定义并研究其性质。

**💡 创新点**

在多种度量上给出了最优的积分间隙上界和下界，尤其将编辑距离的上界从3降到2，切分距离的上界从 O(n^{1/67}) 降到常数倍的线性；证明了多种度量的最优传输形式与吹放（blowup）形式等价；给出了 NP 难度与逼近硬度的完整结果。

**🔧 技术方法**

采用了 Birkhoff–von Neumann 分解、稀薄匹配定理、依赖性舍入、薄匹配、谱近似、Zonotope 近似等一系列新型算法与随机化技术；还使用了 Gromov–Wasserstein 思路与图极限理论。

**📊 数据集**

本研究为理论工作，未使用任何实验数据集；所有结果均为理论证明。

**📈 对比分析**

通过积分间隙分析，展示了最优传输距离与传统置换距离之间的数值关系；对编辑距离、切分距离给出常数倍或几乎线性的上界；对本地编辑和光谱距离则给出非线性上界，表明无法获得线性约束。

**⚠️ 局限性**

仍未得到所有度量的精确积分间隙；Lovász 切分距离的 2 因子 conjecture 仍未完全证明；最优传输距离的 NP 难度与逼近硬度在高维情况下仍有进一步改进空间。

---

## 659. DRAP: Dynamics-Risk Adaptive Plasticity for Continual Learning in Physical AI

**arXiv ID:** 2611.05029 | [PDF](https://arxiv.org/pdf/2611.05029v1)

**作者:** Oswaldo Ludwig `[一作]` `[通讯]`, Oswaldo Ludwig

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种无回放的连续学习方法——Dynamics‑Risk Adaptive Plasticity（DRAP），通过将演示轨迹的物理放大风险与样本级动作梯度相结合，动态调整网络张量的可塑性，从而在多任务序列中显著降低遗忘。

**💡 创新点**

创新点在于：①引入基于Lyapunov灵感的局部动态风险评估，量化轨迹中状态扰动的放大程度；②使用风险加权的样本梯度累计张量重要性，并以此直接缩放各张量的AdamW学习率，实现即时且无参照的可塑性抑制；③与KappaTune相结合，仅训练结构性选择的子网络，进一步提升稳定性。

**🔧 技术方法**

技术包括：局部岭回归估计状态-动作线性动力学、计算最大奇异值得到风险权重、基于张量梯度平方的权重重要性累积、归一化与指数衰减、张量级自适应学习率调整；并在OpenVLA模型上实现。

**📊 数据集**

使用LIBERO Benchmark的Spatial、Object、Goal三个子集（共计30个任务）进行顺序学习实验，评估在无回放设置下的闭环成功率。

**📈 对比分析**

与标准AdamW微调和EWC对比，KappaTune+DRAP在每个阶段的平均成功率提升约20‑30%，后续遗忘率下降至约0.3，明显优于基线且保持新任务学习速度。

**⚠️ 局限性**

局限性包括：风险评估仅基于单步线性近似，未考虑控制路径、长期多步误差或接触动力学；在更长的任务序列和更复杂物理环境中的性能尚未验证；同时KappaTune的子网络选择可能限制模型的表达能力。

---

## 660. When Category Means Move: A Paired-Support Audit of Recommendation Logs with Random Insertions

**arXiv ID:** 2611.05006 | [PDF](https://arxiv.org/pdf/2611.05006v1)

**作者:** Yu Zhang `[一作]` (AMap Alibaba Group), Shuai Li `[通讯]` (AMap Alibaba Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对短视频平台的内容类别仪表盘进行测量审计，比较生产日志与随机插入日志的类别均值、形状、排名与决策损失，并分析误差来源。

**💡 创新点**

提出条件配对支持审计、完整支持级形错误账本、基于早期随机插入的前瞻校准、决策感知回归与置信区间、以及半合成已知真值压力测试，以可视化误差分解与校准边界。

**🔧 技术方法**

使用统计诊断（MAE、中心化MAE、Spearman、最大误差、top‑k重叠）、对称率–组成分解、Shapley分配、Bootstrap置信区间、前瞻冻结校准、离散化决策损失、用户去均值（CE/RW）以及半合成仿真与控制变量。

**📊 数据集**

基于KuaiRand随机插入日志与标准服务日志（约4周时间窗口）的真实短视频平台数据；以及包含38类、760个视频、1000个用户的半合成模拟数据。

**📈 对比分析**

直接与随机插入基准对比；分别报告水平误差（绝对MAE、中心化MAE、最大误差、Spearman、top‑10重叠）。用户去均值可降低约30%绝对MAE，但对排名提升有限；前瞻校准在1周后保持0.1%水平误差并略优于全局标量校准，但决策损失在不同阈值下不显著；已知真值测试显示在弱重叠或项目混合情况下校准效果有限。

**⚠️ 局限性**

只在单一平台和短期窗口内验证；配对用户/视频不等价于随机试验，忽略上下文/位置/时间差异；随机插入日志仅为类别–视频混合参考，无法消除用户–类别交互或项目选择偏差；弱重叠导致有效样本量低、误差大；仅考虑官方类别标签，可能存在标签误差；前瞻校准不保证在更长期或不同流量环境中的稳定性。

---

## 661. For You, Not for Everyone: Bystander-Aware Delivery of Critical Feedback in Robot Tutoring

**arXiv ID:** 2611.05001 | [PDF](https://arxiv.org/pdf/2611.05001v1)

**作者:** Atikkhan Faridkhan Nilgar `[一作]` (University of Siegen), Kristof Van Laerhoven `[通讯]` (University of Siegen)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在机器人辅导场景中，研究者实现并实验了机器人在有旁观者存在时，根据检测到旁观者与否来调整反馈的呈现方式，比较固定与自适应两种策略的效果。

**💡 创新点**

首次将旁观者检测与反馈呈现模式的自适应结合，用来保护学习者隐私并提升其对机器人的社会能力感知，创新性在于针对旁观者情境的隐私友好式反馈交付。

**🔧 技术方法**

技术采用基于摄像头的面部与姿态识别（MediaPipe）、机器人转体与显示面向控制（Ohbot 头部加电机旋转）、合成语音与显示文本。

**📊 数据集**

使用40名大学生参与实验，未使用公开数据集。

**📈 对比分析**

通过两组间实验与 Mann‑Whitney U 检验，结果显示自适应条件下的隐私意识评分显著高于固定条件（p<0.001, r_rb≈0.75），社会能力评分亦显著提升（p=0.019, r_rb≈0.48），而尴尬感差异不显著。

**⚠️ 局限性**

局限包括样本量小、非随机分配、仅单一旁观者模型、未测量旁观者实际获取信息、反馈内容非基于真实表现等。

---

## 662. Auditing Multi-Proxy Explanations of Web User Preferences: Identification Boundaries and Abstention

**arXiv ID:** 2611.04997 | [PDF](https://arxiv.org/pdf/2611.04997v1)

**作者:** Yu Zhang `[一作]` (AMap Alibaba Group), Shuai Li `[通讯]` (AMap Alibaba Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种审计框架，用来判断多代理（如行为、评分、LLM标签）偏好归因是否能够点识别、部分识别或需要放弃，并给出相应的中位数恢复规则。

**💡 创新点**

创新点在于：① 在任意特征依赖、符号不定的代理偏差下，证明了严格多数（>50%）的无污染归因可以唯一恢复潜在偏好向量的支持与方向；② 构造了响应有效的不可区分世界实现匹配下限；③ 提出了兼容目标集合（compatible‑target set）来描述在尺度或污染不确定时的部分识别空间。

**🔧 技术方法**

技术手段包括：Shapley值线性分解、坐标中位数稳健估计、响应有效构造、兼容集推断、最小化错误分析、以及在模拟与实际数据上的实验验证。

**📊 数据集**

使用的数据集包括：① 受控已知真相基准（两类DGP，3–7个代理，完整的污染和强度格点）；② 亚马逊评论（约100k用户）在时间上分离的历史/代理/未来窗口，配合LLM生成的标签与行为评分。

**📈 对比分析**

方法比较：在基准实验中，中位数在严格多数条件下的平均误差优于RPCA、均值、低秩‑稀疏等基线；在亚马逊实验中，中位数相较于仅行为的归因在CSE（约0.02–0.05）和DNAE（约0.01–0.07）方面显著下降，且误报率为0。通过兼容集分析显示，大多数情况下不满足严格多数，因而只能给出不确定或放弃结论。

**⚠️ 局限性**

局限性：框架依赖外部验证的清洁度、尺度区间、参考分布以及代理间的独立性假设；若这些假设不成立，结果只能退化为兼容集或放弃，无法给出精确的偏好大小或排名；此外，实验中使用的代理数量和模型设置仍属于实验场景，实际部署需进一步验证。

---

## 663. Exact Selective Inference for Spectral Clustering via Piecewise-Constant Graph Perturbation

**arXiv ID:** 2611.04995 | [PDF](https://arxiv.org/pdf/2611.04995v1)

**作者:** Naotoshi Nakamura `[一作]` `[通讯]` (Yokohama City University), Naotoshi Nakamura (Yokohama City University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对谱聚类的完全精确的选择性推断方法，能够在已知尺度的球面高斯模型下给出有限样本下的截断χ检验和截断正态特征检验；

**💡 创新点**

核心创新在于证明：在使用无权ε邻域图或k‑NN图时，谱聚类的选择事件仅由图状态决定，可分解为有限个多项式根产生的区间，从而实现精确的截断p值；

**🔧 技术方法**

利用图的多项式有限状态性质、线性/二次距离函数、稀疏图变换、Deterministic L‑means聚类、显式的截断区间构造以及数值稳定的 homotopy 搜索；

**📊 数据集**

在模拟的球面高斯三组数据、PBMC 3k 单细胞 RNA‑seq 数据以及多种噪声模型上进行了实验；

**📈 对比分析**

通过与传统的“选后检验”以及基于数据稀释的基线进行比较，显示在全局和局部零假设下均保持统一性，且在信号强度增大时匹配目标条件的检验比仅固定图状态更具功效；

**⚠️ 局限性**

局限包括：仅适用于无权邻域图（不支持RBF等平滑核）、已知尺度或已知协方差的球面高斯模型、需要预先固定所有参数（K、邻域大小、Laplacian、归一化、随机种子等），以及缺乏数值精度认证和大规模可扩展性。

---

## 664. DSL-Bench: Evaluating and Explaining LLM Code Generation Across Domain-Specific Languages

**arXiv ID:** 2611.05168 | [PDF](https://arxiv.org/pdf/2611.05168v1)

**作者:** Zihan Zhou `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了跨领域可执行的DSL-Bench基准，涵盖18种代表性领域特定语言，并评估7种大型语言模型在这些DSL上的代码生成表现；

**💡 创新点**

首次统一语法检查与任务级语义验证的两级评估框架，量化了语法通过率与语义正确率之间的巨大差距，并通过因子分析揭示知识需求、结构语义与版本干扰等DSL瓶颈；

**🔧 技术方法**

采用LLM辅助文献筛选、卡片分类和句子变换进行领域聚类；为每种DSL实现或构造语法检查器和语义oracle；对生成代码进行零样本提示、检索+修复（RAG‑Refinement）等提升技术；

**📊 数据集**

利用13,045篇DSL相关论文构建领域列表，挑选18种可执行DSL，采集各DSL公开任务集（共计数千至数万条）；

**📈 对比分析**

在同一生成协议、提示模板和验证管道下，对7个模型在18个DSL上进行宏平均和任务加权评估；平均语法通过率31.37%，语义通过率仅11.64%，模型主效应仅占7.86%而DSL主效应占76.99%；Prompt Engineering平均提升语义通过率2.42个百分点，RAG‑Refinement在5个关键DSL上平均提升16.29个百分点；

**⚠️ 局限性**

局限包括：只覆盖18种DSL，未涵盖更广泛的低资源或新兴DSL；模型选择受限于可本地部署与API成本；验证器差异导致语义难以直接跨DSL比较；实验缺乏大规模闭源前沿模型的评估；

---

## 665. GM League Arena: An Interactive Long-Horizon Task Benchmark Based on Professional Sports League Management

**arXiv ID:** 2611.05147 | [PDF](https://arxiv.org/pdf/2611.05147v1)

**作者:** Junkai Zhou `[一作]`, Zhaoyi Zhang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

创建了GM Bench，一个跨四种职业体育（NBA、NHL、MLB、NFL）的长期管理基准，让大语言模型扮演球队总经理进行多赛季决策与资源分配；

**💡 创新点**

将交互式任务、市场机制、财务约束与真实球员数据统一到一个长期决策评估框架，并通过可比球队构建对比基准；

**🔧 技术方法**

采用大语言模型驱动的交互环境，包括玩家/代理LLM、联赛规则模拟器与游戏引擎；

**📊 数据集**

使用真实职业体育球员评级数据（NBA、NHL、MLB、NFL）以及公开API生成的玩家需求与交易请求；

**📈 对比分析**

通过对31个LLM在5个赛季的对比，评估Elo、现金余额和长期提升等指标，其中Claude Opus 5.5排名第一并夺得3个冠军，GPT‑6系列和Grok表现突出，显示不同模型在冠军追求、现金保留与阵容连续性上存在权衡；

**⚠️ 局限性**

受限于特定联赛规则、有限的赛季数、模拟成本高以及未能评估更长周期的玩家发展与资产成熟情况。

---

## 666. Path Decompositions of Sparse Graphs via Gaussian Sweeps and Faster Exponential Algorithms

**arXiv ID:** 2611.05146 | [PDF](https://arxiv.org/pdf/2611.05146v1)

**作者:** Serge Gaspers `[一作]` `[通讯]` (UNSW Sydney), Serge Gaspers (UNSW Sydney)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

论文证明了最大度为3和4的图的路径宽度、二分宽度和切宽度的上界，分别为n+o(n)和n/3+o(n)，并改进了之前的界限。

**💡 创新点**

创新点在于通过高斯标签和阈值扫描的方法，提供了更紧的路径宽度界限，并且提出了新的算法来计算这些宽度。

**🔧 技术方法**

使用了高斯标签和概率论中的工具，如切比雪夫不等式和随机超平面引理。

**📊 数据集**

使用了多种图数据集，包括最大度为3和4的图，以及随机立方图。

**📈 对比分析**

与之前的方法相比，新的界限在路径宽度上有显著改进，尤其是在最大度为3的情况下，性能提升明显，且算法的运行时间也得到了优化。

**⚠️ 局限性**

限制在于对于更高最大度的图，仍然需要进一步的研究来找到更好的界限和算法。

---

## 667. Tight Analysis of Randomized Truthful Scheduling: An $(m+1)/2$ Lower Bound

**arXiv ID:** 2611.05142 | [PDF](https://arxiv.org/pdf/2611.05142v1)

**作者:** Wenqian Wang `[一作]` (Shanghai Jiao Tong University), Yuhao Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文证明了在无关机器上对机器可做完工时间（makespan）最小化的期望真诚机制（Truthful-in-Expectation, TIE）的最优逼近比率至少为 (m+1)/2，并将该结果推广到通用真诚机制（universally truthful），从而几乎确定了这两类机制的最优逼近比率。

**💡 创新点**

创新点在于提出并利用“作业分配相互作用（job allocation interaction）”这一新的结构化工具，将机制的真诚性约束转化为对分配函数高阶差分（交互阶次）的限制，并通过该限制与硬实例构造实现了对 TIE 机制的下界。此方法完全脱离了以往仅适用于确定性机制的 Box Theorem，并首次实现了在随机化机制中引入交互阶次的概念。

**🔧 技术方法**

核心技术包括：
- Möbius 逆变换得到作业分配的交互系数，定义交互阶次；
- padding‑invariant 替换与随机化平滑（right/left smoothing）实现无交互阶次约束；
- Laplace 变换与多变量微分守恒定律证明交互系数为零；
- Yao 的极小极大原理结合随机星图实例构造硬实例；
- 组合概率分析估计 OPT 与期望负载，利用多项式约束控制交互阶次。

**📊 数据集**

使用的数据集：本研究为理论分析，未使用任何真实或模拟数据集。

**📈 对比分析**

与已有方法对比：
- 对于 TIE 机制，之前已知下界为 2-1/m，上界为 (m+5)/2；本文将下界提升至 (m+1)/2，几乎匹配上界，只剩常数差距。
- 对于通用真诚机制，之前已知下界为 (3/2-√2)(m-1)≈0.0858(m-1)；本文将下界提升到 (m+1)/2，并与已知上界 m/2+o(m) 的构造相匹配，差距仅为常数 o(m)。
- 本文的下界在所有 m≥2 时均成立，并给出了与上界几乎相等的逼近比率。

**⚠️ 局限性**

局限性：
- 结果仍然是近似的，差距为常数项（2 或 o(m)）。
- 证明依赖于对交互阶次的严格约束，实际机制实现时需要在实现中消除高阶交互，可能导致额外的计算复杂度。
- 证明过程包含平滑与极限操作，导致理论上可实现的机制可能在实践中需要近似处理。
- 目前未给出对多任务（m>2）下的精确常数改进，仍有进一步优化的空间。

---

## 668. Information Loss Is All You Need: $TC^0$ Pseudorandomness from Worst-Case Lattices

**arXiv ID:** 2611.05136 | [PDF](https://arxiv.org/pdf/2611.05136v1)

**作者:** Chen Qian `[一作]` `[通讯]`, Chen Qian

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文构造了一个基于确定性舍入的弱伪随机函数（weak PRF），并证明其在量子最坏情况格问题下的安全性，同时得到在无噪声均匀样本下学习两隐藏层ReLU网络的计算困难性。

**💡 创新点**

创新点在于：①引入“代数舍入”（Learning with Algebraic Rounding，LAR）将确定性舍入与格的余数编码相结合；②利用源独立的仿射传输（affine transport）把高斯错误转化为零纤维错误，保持公共矩阵分布不变；③开发了一套源无关的校准（calibration）框架，使得零纤维错误与均匀分布在高维空间上几乎完全等价；④首次证明了在量子最坏情况格假设下存在可实现评估的弱PRF，并进一步推导出无噪声学习难度。

**🔧 技术方法**

主要技术包括：格论（格的最短向量问题、离散高斯分布）；代数舍入与Smith标准形；仿射传输与整数负载/载荷理论；源独立仿射变换与统计距离分析；校准技术（局部与全局校准）结合Neumann逆推理；傅里叶分析与高阶矩估计；以及拒绝采样与条件化技术。

**📊 数据集**

该工作为理论构造，无需实际数据集；所有实验均在理论上进行，证明和归纳推理完成所有安全性与学习难度的证明。

**📈 对比分析**

本研究不做实证对比，评价标准为理论安全性证明（量子最坏情况格安全性）以及无噪声学习难度的计算复杂性。相比已有基于随机舍入（LWR）或基于LPN的构造，本文在保持多项式模量、无样本上限和确定性评估的同时，完成了量子安全性的首次实现。

**⚠️ 局限性**

局限性包括：需要假设量子最坏情况格问题的困难性；模量、维度和块大小相对较大（如模量约为2^160，维度约为2^64）；实现复杂度高，涉及多层校准与大幅度矩阵运算；对实际部署（如性能、密钥尺寸）尚未评估。

---

## 669. The Right Source Is Not Enough: Understanding History Interference in Speech-LLM Memory

**arXiv ID:** 2611.05124 | [PDF](https://arxiv.org/pdf/2611.05124v1)

**作者:** HongJin Song `[一作]` (Beijing Institute of Technology), Chunxiang Jin `[通讯]` (Ant Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究语音大语言模型在多轮对话中如何记忆并错误使用历史音频信息，揭示来源角色与答案值的竞争机制。

**💡 创新点**

创新点在于通过因果干预分离来源角色与答案内容的影响，证明即使记忆被检索，历史值仍可通过状态携带方式持续影响决策。

**🔧 技术方法**

采用激活补丁、注意力屏蔽、低维子空间投影等因果干预技术，以及层级扫描来定位影响路径。

**📊 数据集**

使用两套数据集：自然语音的 3,000 条 LibriSpeech 历史问答，以及 512 条受控合成音频（时间、颜色、地点、数量）进行实验。

**📈 对比分析**

与完整历史相比，历史干扰将 MOSS‑Audio‑8B 的准确率从 84.23% 降至 78.73%，MiMo‑Audio‑7B 从 48.23% 降至 39.67%；多选错误在严格屏蔽后仍保留 70% 以上，表明状态携带效应显著。

**⚠️ 局限性**

局限性在于仅提供基于现有模型的因果诊断，未提出可部署的修复方法；干预依赖 oracle 信息，且实验环境与真实对话场景仍有差距。

---

## 670. DexHOT: Learning Hand-Object Configuration Transfer for Dexterous Manipulation

**arXiv ID:** 2611.05119 | [PDF](https://arxiv.org/pdf/2611.05119v1)

**作者:** Zhewei Gui `[一作]` (Peking University), Hao Dong `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了DexHOT框架，利用手-物体配置转移实现多物体的柔性抓取与长时序操作。

**💡 创新点**

创新点在于把手-物体关系显式为目标，使用可生成的物理可行配置对进行本地转移学习，并通过层次化规划实现远距离和长时序任务。

**🔧 技术方法**

采用了基于Population-based强化学习（EPO）训练的LSTM控制器、前向运动学键点表示、配置距离度量、贪心规划以及人类演示子目标的层次化控制。

**📊 数据集**

使用了120个物体的DexHOTBench数据集（包含40个训练、80个隐藏物体）以及Omni6DPose 40类别上的可行配置池。

**📈 对比分析**

与传统仅以物体姿态为目标的方法对比，DexHOT在仿真中已见物体的任意两配置转移成功率为80.5%，未见物体为69.4%，真实机器人转移成功率为61.5%，并在六项演示驱动任务中实现90%成功。

**⚠️ 局限性**

主要局限在低摩擦物体的失稳、初始化可行配置的重现困难、长时间转移时易停滞，以及对复杂几何物体的感知和动力学建模不足。

---

## 671. Distributed Linear Bandits for Freshness-Aware Preprocessing in MEC-Enabled Digital Twin Networks

**arXiv ID:** 2611.05115 | [PDF](https://arxiv.org/pdf/2611.05115v1)

**作者:** Mariam Yahya `[一作]` (Ruhr-University Bochum), Setareh Maghsudi `[通讯]` (Ruhr-University Bochum)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于边缘计算的数字孪生（DT）预处理框架，利用DT预先确定满足各类任务新鲜度（AoI）要求的CPU分配策略，然后让MEC服务器协同学习并挑选出在预处理延迟与新鲜度归一化误差最小的策略。

**💡 创新点**

创新点包括：①将DT与MEC协同工作的新鲜度感知框架；②将CPU分配视为固定置信度的线性Bandit最佳臂识别问题；③采用分布式线性Gap探索算法DistLinGapE实现多台MEC服务器间的观测共享与加速识别；④通过理论分析与仿真实验证明协同学习能近乎线性加速。

**🔧 技术方法**

使用的技术主要有：分布式线性Bandit（LinGapE）与DistLinGapE算法、固定置信度最佳臂识别框架、AoI新鲜度建模、线性奖励模型以及协作通信协议。

**📊 数据集**

数据集为仿真产生的IoT任务数据：16个集群、4台MEC服务器、8类任务，每类任务按Bernoulli概率产生，任务大小在40–60kB之间，处理需求为每比特若干CPU周期。

**📈 对比分析**

方法与对比：与均匀分配、随机分配、独立LinGapE、共享均匀探索以及理想的oracle进行对比。实验结果显示DistLinGapE在识别回合数上显著优于其他方法，几乎与oracle达到相同的归一化损失；协同学习使识别回合数呈近线性缩减，提升速度达到2~7倍。

**⚠️ 局限性**

局限性：①假设MEC服务器同质且预处理任务统计分布已知上界；②只在仿真场景下验证，缺乏真实部署实验；③对动态变化的任务生成率与负载分布考虑有限；④通信阈值与同步机制对速度提升有影响，需进一步优化。

---

## 672. The Platonic Brain: Representational Convergence in EEG Foundation Models

**arXiv ID:** 2611.05097 | [PDF](https://arxiv.org/pdf/2611.05097v1)

**作者:** Nicolas Waehner `[一作]` (University of Buenos Aires), Trinidad Borrell `[通讯]` (Paris Brain Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文研究了EEG基础模型在提升解码能力（NICE decodability）时，其内部和跨模态表示是否会趋同；

**💡 创新点**

创新点在于将柏拉图表征假设（PRH）推广到EEG模型，提出无任务的NICE解码指标作为模型能力量化，并通过循环时间移位的null模型校正相似度，揭示了能力与表征收敛之间的关联。

**🔧 技术方法**

技术上使用了多家EEG模型（14个），对其嵌入进行局部邻域重叠（mKNN）与全局相似度（CKA）评估；引入循环时间移位null模型校正；利用岭回归预测多种EEG标记以计算NICE解码得分；同时与视觉（DINOv2、V-JEPA2、VideoMAE）和语言模型（BLOOMZ、OpenLLaMA）进行跨模态对齐。

**📊 数据集**

使用的实验数据集是CineBrain，包含六名受试者（21–26岁，4女2男）在观看《生活大爆炸》180分钟时的EEG和fMRI记录，EEG被分段为与视觉刺激和字幕对应的时间窗口。

**📈 对比分析**

比较方法为：先对每对模型计算mKNN和CKA相似度，然后用循环时间移位的null模型得到校正后的对齐得分；随后在同一家族内进行中心化，将模型的对齐得分与其NICE解码得分做Spearman相关；结果显示：内模态对齐随NICE解码得分显著正相关（p≈0.003/0.007），但与参数规模无一致关系；跨模态时，CKA与语言模型的对齐与NICE解码成正相关，而mKNN无一致关联。

**⚠️ 局限性**

限制主要包括：仅使用了14个模型和6名受试者，统计功效有限；对齐度受到时间平滑影响，可能部分归因于共同的时间结构；仅基于单一数据集，结果的普适性尚未验证。

---

## 673. LoopQ: Fair, Closed-Loop Evaluation of Quantized Vision-Language-Action Models

**arXiv ID:** 2611.05093 | [PDF](https://arxiv.org/pdf/2611.05093v1)

**作者:** Syed Umer Bukhari `[一作]` (Chalmers University of Technology), Ahmed Ali-Eldin `[通讯]` (Chalmers University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个闭环基准，统一评估并比较了多种 VLA 模型（OpenVLA‑OFT、GR00T N1.5、UniVLA、OpenVLA‑OFT）在不同权重量化方法（RTN、AWQ、GPTQ、ActQuant、QuantVLA）下的任务成功率、内存占用和推理速度，实验覆盖超过 180,000 次闭环执行，并在 H100、L4 与 Jetson AGX Thor 三种硬件上进行测试。

**💡 创新点**

创新点在于：①构建了“闭环基准”，确保仅改变量化器而保持 checkpoint、量化层、引擎和 kernel 路径一致；②使用标准 LIBERO 与其对照的 LIBERO‑plus 分布偏移任务进行配对评估，揭示了传统评估忽略的鲁棒性损失；③系统性对比 VLA‑专用量化方法与通用权重量化基线，发现后者在 4‑bit 级别上往往同样或更优。

**🔧 技术方法**

技术上采用了后训练量化（PTQ）方法 RTN、AWQ、GPTQ、ActQuant、QuantVLA；在不同 GPU 上使用了 cuBLAS、MMQ、W4A16、int8‑MMQ、ActQuant 自研引擎；利用 Nsight Systems 进行 GPU 级别性能剖析，并对模型文件大小、峰值显存和能耗进行测量。

**📊 数据集**

数据集为 LIBERO（4 个任务套件，共 50 试验/任务，2,000 次 episode）和其扩展版本 LIBERO‑plus（分布偏移的 7 个扰动维度，每个 5 难度级别），在四个 VLA 上共计 180,000+ 关节循环测试。

**📈 对比分析**

比较方法为：对同一基准模型在同一 GPU 引擎上跑量化与未量化版本，记录任务成功率、峰值显存、每次调用延迟以及能耗。结果显示：所有 4‑bit 量化方案在内存上可节省 2–3 倍，但大多数在 H100 上速度比 FP16 慢 10–58%；在 L4 上仅 int8‑kernel 可实现 4–14% 加速；在 Jetson AGX Thor 上更慢；在 LIBERO‑plus 中，ActQuant 与 QuantVLA 在 4‑bit 时均损失 2.3–4.2 分，UNI‑VLA 的 4‑bit 量化甚至在分布偏移时下降 1.5–4.3 分，而通用 4‑bit 方法几乎无损失。

**⚠️ 局限性**

局限性包括：①仅评估权重量化（未考虑激活量化或混合精度策略）；②测试仅在 batch‑1 级别下完成，可能低估大批量时的性能；③受限于公开可获得的 checkpoint，部分 VLA‑专用方法仅以模拟方式评估；④仅在三种 GPU 上实验，缺乏对更多嵌入式硬件的验证；⑤鲁棒性测试仅覆盖 LIBERO‑plus 的 7 个扰动维度，可能不足以全面代表真实部署环境。

---

## 674. Better Plans, Better Outcomes: Integrating Historical Data into Student Planning in a Computing Course

**arXiv ID:** 2611.05143 | [PDF](https://arxiv.org/pdf/2611.05143v1)

**作者:** Jiaen Yu `[一作]` (University of California, San Diego), Soohyun Nam Liao `[通讯]` (University of California, San Diego)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在计算机课程中嵌入基于历史同届学生数据的任务预览与统计信息，帮助学生制定更保守、更可执行的学习计划。

**💡 创新点**

创新点在于将任务特定的、经验性历史数据实时呈现给学生，弥补传统规划工具缺乏课程真实估计的不足，并证明其对计划遵循率和学习成绩的正向影响。

**🔧 技术方法**

采用了轻量级的 Web 平台（CompassX）与可视化统计呈现技术，结合结构化规划界面和后评估对比功能。

**📊 数据集**

使用了先前一个学期（Winter 2025）CS2 课程中 69 名学生的完成时间分布和难度评分数据，以及本学期 252 名学生的计划与实际执行记录。

**📈 对比分析**

通过在同一学期内采用对照-处理的交替设计（5 次编程作业中 3 次配合干预，2 次不配合），并对计划遵循率、作业分数及期中/期末成绩进行配对/独立样本 t 检验，结果显示干预组计划更保守、遵循率提升 13.56%，高遵循率组在期中和期末成绩上显著优于低遵循率组。

**⚠️ 局限性**

局限包括样本仅来自单门课程、干预效果为相关性而非因果、历史数据仅基于单一前期班级、时间与难度自报易受偏差、调查响应率低、未针对执行阶段提供额外支持。

---

## 675. Learning Behaviorally Relevant Dynamics in Multimodal Neural Recordings

**arXiv ID:** 2611.05167 | [PDF](https://arxiv.org/pdf/2611.05167v1)

**作者:** Parastoo Azizeddin `[一作]` (University of Southern California), Maryam M. Shanechi `[通讯]` (University of Southern California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种非线性多模态动力学模型，联合行为与多种神经信号（如尖峰与局部场电位）进行融合，并在训练过程中通过分阶段结构学习将行为相关的潜在动力学与剩余神经动态分离开来。

**💡 创新点**

创新点在于（1）采用三阶段结构化学习策略实现行为相关与残差神经动力学的功能性分离；（2）为每种模态设计专属的编码器/解码器，支持异质统计分布（Poisson、Gaussian）及不同采样率；（3）通过RNN隐状态递归实现跨模态信息融合，并允许异步采样与模态丢失的鲁棒性。

**🔧 技术方法**

技术方法包括：RNN隐状态动力学模型、模态特定编码器/解码器、负对数似然损失优化、三阶段分阶段训练、Poisson与Gaussian似然联合推断、行为解码与神经预测评估（Pearson相关、预测功率PP）、对齐得分等。

**📊 数据集**

实验数据使用公开的非人类灵长类动物（NHP）数据集，包含在2D到达任务中同步记录的尖峰活动与原始LFP信号；同时在仿真数据上验证模型。

**📈 对比分析**

模型与多种基线（MVAE、MSID、多会话CEBRA、mmPLRNN、MMGPVAE）以及无模态融合、无监督版本进行对照；结果显示，在行为解码相关性、神经信号预测（LFP相关系数、尖峰预测功率）以及行为轨迹对齐度上均显著优于所有基线（p<0.001）。同时在异步采样与高比例模态丢失的情形下保持鲁棒。

**⚠️ 局限性**

局限性主要在于仅针对尖峰与LFP两种模态进行验证，尚未扩展到EEG、iEEG或生理信号等更丰富模态；模型复杂度高，训练成本较大；在不同脑区与行为任务下的泛化性能仍待进一步评估。

---

## 676. RoboTryOn: Synthesizing Dexterous Robot Demonstrations from Human Videos via Virtual Try-On

**arXiv ID:** 2611.05163 | [PDF](https://arxiv.org/pdf/2611.05163v1)

**作者:** Seungyong Lee `[一作]` (KAIST), Jinwoo Shin `[通讯]` (KAIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种将人类第一人称视频转换为机器人演示的方法，利用关键帧自动回归翻译、视频生成和逆动力学模型生成动作标签，帮助机器人在缺乏遥操作演示的情况下学习新任务。

**💡 创新点**

创新点在于将人机转移视为“虚拟试衣”，只改动体现体而保持动作；使用共享生成模型处理多种高自由度手；在视频合成过程中仅用手骨架指导；逆动力学模型提供可执行的动作标签；通过该流程在两台真实机器人上实现无遥操作演示的任务学习。

**🔧 技术方法**

关键技术包括：
- 关键帧选择与arm/hand mask + hand skeleton条件化的图像扩散生成模型；
- 关键帧间的手骨架流引导下的视频扩散生成模型；
- 逆动力学模型（IDM）用于从合成视频中推断动作；
- Vision‑Language‑Action（VLA）策略的流匹配微调；
- 对齐、速度调节等实验细节。

**📊 数据集**

使用的数据集：
- 50 条遥操作演示（针对四个 In‑Domain 任务）
- 50 条对应的人类视频
- 另外 7 个 Out‑of‑Domain 任务仅有人类视频（共 11 任务）
- 两台机器人（OpenArm+Inspire RH56F1 手，RB‑Y1+WujiHand2 手）在桌面任务集合上收集的演示。

**📈 对比分析**

比较方法：与两种基线（EgoMimic、Masquerade）以及仅遥操作的数据进行对照。结果显示：
- 在 In‑Domain 任务上成功率从 47%~70% 提升至 75%~80%；
- 在 Out‑of‑Domain 任务上成功率从 1%~3% 提升至 47%~73%；
- 仅使用人类视频训练的策略在成本约三分之一的情况下即可匹配或超越仅遥操作的性能；
- 通过速度调节进一步提高了 In‑Domain 成功率至 85%。

**⚠️ 局限性**

局限性：
- 依赖手骨架重建质量，若重建不准会影响合成质量；
- 逆动力学模型对深度特征的依赖使得在深度信息缺失时表现受限；
- 生成模型对高自由度手的准确性有限，难以完全覆盖所有复杂抓取姿态；
- 仅在桌面任务集上验证，尚未测试在更复杂场景或移动机器人中的泛化能力。

---

## 677. What Sets the Serving Speed of Looped Language Models? Measurements and Optimizations Graded by Exactness

**arXiv ID:** 2611.05150 | [PDF](https://arxiv.org/pdf/2611.05150v1)

**作者:** Dezhen Lu `[一作]`, Lingxiang Gao `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究循环式语言模型的推理性能并提出一系列低开销的精确优化。

**💡 创新点**

提出活跃槽打包、精确内核融合和位置键噪声等技术，保持比特级一致性并显著提升吞吐。

**🔧 技术方法**

采用KV池压缩、CUDA图、RoPE/RMSNorm/SiLU的融合、位置键噪声生成、预填共享、GPU内存复制优化及NIXL传输。

**📊 数据集**

使用GSM8K、ShareGPT、Alpaca、MATH-500等公开数据集评估准确性。

**📈 对比分析**

在GH200、RTX5090等 GPU 上与基线 vllm‑rlt 进行吞吐量、TTFT、TPOT 对比，ExactLoop 提升吞吐 1.5–3 倍且准确性保持不变。

**⚠️ 局限性**

仅针对固定深度循环模型，未验证与稀疏/量化等组合；近似融合未通过准确性阈值；高负载下预填共享会导致 TTFT 上升。

---

## 678. Creating Power Flexibility in Datacenters via Query Routing

**arXiv ID:** 2611.05148 | [PDF](https://arxiv.org/pdf/2611.05148v1)

**作者:** Shivam Patel `[一作]` (Carnegie Mellon University), Gauri Joshi `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了通过查询路由在数据中心中实现大语言模型推理的功耗弹性，并构建了解析模型与仿真验证。

**💡 创新点**

提出了利用不同规模模型可替代性实现毫秒级功耗调节的框架，并给出了全局功耗可调范围的分析方法。

**🔧 技术方法**

采用解析模型预测token迭代时间、请求驻留时间、吞吐量与功耗，并在vLLM+H100 GPU上部署实验；结合仿真与真实工作负载进行验证。

**📊 数据集**

使用了ShareGPT、HumanEval、arXiv Summarization等公开数据集以及Azure LLM对话追踪作为真实负载。

**📈 对比分析**

与实际测量结果比较，功耗预测平均绝对百分误差约2.8%，请求驻留误差5.5%；在10,000 H100 GPU仿真中实现最高16%功耗弹性，10秒内达到90%目标。

**⚠️ 局限性**

局限性包括仅考虑推理工作负载、假设查询可灵活路由、对训练/存储等其他任务的功耗调度未做完整建模，且对GPU资源可用性变化的鲁棒性受限。

---

## 679. An Investigation on Numerical Bugs in GPU Programs Towards Automated Bug Detection

**arXiv ID:** 2611.05121 | [PDF](https://arxiv.org/pdf/2611.05121v1)

**作者:** Ravishka Rathnasuriya `[一作]` (University of Texas at Dallas), Wei Yang `[通讯]` (University of Texas at Dallas)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对GPU数值bug进行系统性研究，收集并分析了397个真实案例，提出并实现了GPU-NBDetect检测工具。

**💡 创新点**

创新点在于：①首次对GPU数值bug进行细粒度分类（精度、数据类型/操作、特殊值、设备特定四类）；②设计基于变异fuzzing和差分测试的输入生成与oracle策略；③实现首个支持C/CUDA与Python GPU库的检测框架。

**🔧 技术方法**

主要技术：变异级别的输入生成（7种位级突变）、差分测试（CPU vs GPU结果比较）、阈值判定（浮点误差、性能倍增、错误信息）、自动化脚本生成CUDA/C代码、性能监测与统计。

**📊 数据集**

使用的实验数据集为397条GPU数值bug样本（229来自闭源库，168来自开源库），在四个库（CUDA Math、Thrust、CuPy、Numba）共226个检测实例。

**📈 对比分析**

与XScope、XScope-RS、BinFPE、GPU-FPX以及随机/贝叶斯优化输入生成方式比较，GPU-NBDetect检测到最多的bug（226个），误报率低于其他工具，检测时间平均保持在可接受范围内（每个函数约数秒）。

**⚠️ 局限性**

局限性包括：仅支持float/double类型；只评估四个GPU库；变异随机导致检测结果可能不确定；未覆盖其他数值类型和更大范围的GPU功能；需要专业GPU编程知识来使用和扩展。

---

## 680. Characterizing the vertical continuity of fuel strata based on TLS point clouds Gaussian decomposition and overlap metrics

**arXiv ID:** 2611.05102 | [PDF](https://arxiv.org/pdf/2611.05102v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 681. Beyond Multi-Index Models: Learning Intersections of Degree-2 PTFs

**arXiv ID:** 2611.05114 | [PDF](https://arxiv.org/pdf/2611.05114v1)

**作者:** Ilias Diakonikolas `[一作]` (University of Wisconsin Madison), Trung Tran `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

提出了一种在标准高斯分布下对 k 个二次多项式阈值函数（PTF）交集进行 PAC 学习的算法，并给出了相应的下界。

**💡 创新点**

创新点包括：①通过高斯比较定理和新的正则化概念，将任意二次 PTF 的影响方向用低阶 Hermite 瞬时捕捉；②构造了“边界算子”与“导数‑基矩阵”以实现子空间恢复；③提出了两阶段维度缩减与鲁棒 Hinge‑loss 最小化的组合，显著降低对误差参数的指数依赖；④给出了 SQ 模型下的近似不可学习性下界，证明了 d^Ω(k) 的维度指数是不可避免的。

**🔧 技术方法**

核心技术包括：高斯分布下的 Hermite 多项式基、Gaussian anti‑concentration 与 isoperimetric inequality、正则化与 Gaussian 替代、边界算子与导数‑基矩阵的积分分部技巧、低阶多项式近似与 hinge‑loss 最小化、谱分解与子空间投影。

**📊 数据集**

本文为理论研究，无实验数据集，主要在数学推导与算法复杂度分析上做工作。

**📈 对比分析**

与先前基于低阶多项式回归得到的 d^O(k²/ε²) 时间相比，本文实现了 d^O(k) 2^O_k(log³(1/ε)) 的时间复杂度；误差参数 ε 的指数依赖被压缩为指数级的 log³(1/ε)，在 k=2 时已达到 d^O(1) 2^O(log³(1/ε)) 的效率，远优于之前的做法。

**⚠️ 局限性**

局限性：误差参数 ε 的复杂度仍是指数级的 quasi‑polynomial（2^O_k(log³(1/ε))），是否能进一步降低到多项式级仍是开放问题；算法为不完全学习（improper）；仅针对标准高斯分布，未直接扩展到更一般的输入分布或更高阶多项式阈值函数。

---

## 682. From AutoResearch to Continual AutoResearch: Do Agents Learn Across Tasks?

**arXiv ID:** 2611.05108 | [PDF](https://arxiv.org/pdf/2611.05108v1)

**作者:** Jeremy Qin `[一作]` (RBC Borealis), Leo Feng `[通讯]` (RBC Borealis)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 Continual AutoResearch 环境，评估自动研究代理在连续任务中的经验迁移效果。

**💡 创新点**

创新点在于将持久化状态与重置状态对比，揭示经验保留显著提升搜索效率但不一定提高最终性能，并通过多任务族实验验证。

**🔧 技术方法**

技术手段包括使用五大前沿 LLM 代理（Sonnet 4.6、Opus 5、GPT‑5.6‑Sol、GPT‑5.4、Kimi K3），在 CLI 环境下执行自主实验，并采用归一化奖励、固定预算搜索收益和时间收敛收益等指标进行评估。

**📊 数据集**

数据集覆盖合成 AutoML、符号回归、时间序列（Energy、Traffic、ETT）以及表格 AutoML（SELA、MLE‑Bench Lite 等），共约 32 个迁移任务。

**📈 对比分析**

通过在相同任务流下对比 stateful 与 stateless 实例，比较初始化质量、时间收敛收益和最终结果；结果显示持久化状态平均提升约 21 点搜索效率，只有 Sonnet 4.6 在最终结果上取得显著收益。

**⚠️ 局限性**

局限性包括任务数量有限、预算较短、未分离代码/经验/执行状态、模型与框架耦合以及缺乏大规模、多代理或长时间任务的验证。

---

## 683. Molecular Prior Networks

**arXiv ID:** 2611.05103 | [PDF](https://arxiv.org/pdf/2611.05103v1)

**作者:** Pawel Gomoluch `[一作]` (DaltonTx), Anthony Bradley `[通讯]` (DaltonTx)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并训练了Molecular Prior Network（MPN），一种单前向传播的Transformer，能够在支持集与查询集之间联合预测蛋白质-配体亲和力。

**💡 创新点**

将Prior-data Fitted Networks理念迁移到分子亲和力领域，使用标签嵌入作为注意力值实现非参数推断，并引入可比的核基线验证其机制。

**🔧 技术方法**

使用Transformer自注意力、标签嵌入、meta‑学习训练、Morgan指纹特征、平方ReLU FFN以及学习的特征投影进行核回归等技术。

**📊 数据集**

训练基于4938个不同靶标的亲和力数据集，评估使用FS‑Mol基准的111个测试任务（含pIC50回归）。

**📈 对比分析**

与ADKF‑IFT、TabPFN、TabICL、随机森林和Gaussian Process等方法比较，MPN在小支持集（16）可与ADKF‑IFT匹配，整体排名第三，且推理速度远快。

**⚠️ 局限性**

缺乏不确定性估计、仍依赖Morgan指纹、GPU内存受限于大支持集，以及对更广泛属性和多任务迁移的验证不足。

---

## 684. When Self-Verification Hurts: Calibrated Answer Repair for Financial Question Answering

**arXiv ID:** 2611.05166 | [PDF](https://arxiv.org/pdf/2611.05166v1)

**作者:** Celine Guo `[一作]` (National University of Singapore), Wenya Wang `[通讯]` (Nanyang Technological University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Calibrated Answer Repair（CAR）框架，用于在金融问答任务中对模型生成的答案进行选择性、目标性和受限修复，减少错误修正带来的不良影响。

**💡 创新点**

创新点在于通过基于恢复率与损坏率的 break-even 规则设计可调节的修复门；引入证据驱动的诊断探测器预测缺失率；在修复时仅检查特定关系并限制编辑范围，从而兼顾修复收益与安全性。

**🔧 技术方法**

技术上采用大型预训练语言模型（Qwen3‑8B/14B、Mistral‑24B）作为生成器和修复器；使用多维文本特征与探测器二进制判断的岭回归估计缺失率；根据错误子类选取检查与指针；并通过受限编辑策略限制仅局部替换或追加。

**📊 数据集**

实验使用三大金融问答基准 FinRank、Fin‑RATE 与 FinQA，对每个基准均评估不同模型组合。

**📈 对比分析**

与无修复、单次通用验证、多轮复审、事实驱动验证等对照方法相比，CAR 在 9 个模型‑基准组合中实现了 8/9 的最高性能，平均恢复 14.6% 的遗漏要求，损坏率仅 2.1%，且相较迭代修复方案，后期标记成本降低 30% 以上。

**⚠️ 局限性**

局限性包括仅在英文金融数据集上验证，依赖配对的开发数据来学习门控与编辑策略；未覆盖跨语言或其他领域；实验聚焦于答案修复而非完整检索流程；评估依赖第三方判断与公开数据集，可能受限于许可与计算成本。

---

## 685. QPP-Aware MCX over Tactical Edge 3D Networks in Public-Safety Communications

**arXiv ID:** 2611.05165 | [PDF](https://arxiv.org/pdf/2611.05165v1)

**作者:** José Oliveira `[一作]` (Instituto de Telecomunicacoes), Susana Sargento `[通讯]` (Instituto de Telecomunicacoes)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于私有5G‑SA网络的多服务（mcptt、mcvideo、mcdata）应急通信架构，并通过上下文感知的qpp机制实现动态优先级调度；

**💡 创新点**

在5GC/NR层实现了无会话重建的qpp动态更新、边缘化媒体处理与集中式控制的混合架构，并在NTN+边缘部署下验证了MCX的实时性能；

**🔧 技术方法**

使用Open5GS、Kamailio SIP、RTPengine、基于Android的MCX客户端、Mini‑PC + USRP软基站、Starlink NTN链路以及模拟UE（Amarisoft UE SimBox）进行实验；

**📊 数据集**

实验数据来源为自主构建的仿真环境：最多64个模拟UE，包含多组MCPTT、MCVideo、MCData会话，并通过Starlink实现NTN回传；

**📈 对比分析**

对比三种部署（陆基+集中、星链+集中、星链+边缘）下的信令延时、访问时延、音频传输延时与丢包、视频吞吐与帧丢失、数据附件上传时延等指标；边缘化媒体显著降低了95%分位音频延时60 ms、丢包率9.4%；整体MCPTT访问时延保持在300 ms以内；

**⚠️ 局限性**

实验仅覆盖单个UAV边缘节点和Starlink链路，未考虑多UAV协同或更复杂的多波束NR、卫星多路径衰落与网络切片；系统对动态优先级的评估仅基于单个用户情景，缺乏大规模灾害模拟与长期稳定性验证。

---

## 686. Asynchronous Robot Policies

**arXiv ID:** 2611.05155 | [PDF](https://arxiv.org/pdf/2611.05155v1)

**作者:** Sombit Dey `[一作]` (INSAIT), Danda Pani Paudel `[通讯]` (INSAIT)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种异步机器人策略（ARP），将大规模视觉‑语言模型（VLM）与轻量级视觉‑动作专家拆分为不同频率的异步流程，实现高层语义推理与低层动作执行的解耦。

**💡 创新点**

创新点在于：①通过异步进程将VLM的推理频率降低至低频，实现高层语义更新与高频动作生成的并行；②设计了语义记忆压缩与时间衰老编码，使快速动作专家能在“过期”语义上下文中鲁棒运行；③在训练阶段引入时间错配与延迟模拟，增强对异步延迟的适应。

**🔧 技术方法**

主要技术包括：高容量VLM（如CLIP/BLIP）+ 关键值缓存投影生成固定宽度语义记忆；轻量级视觉‑动作专家（Transformer）+ 自适应RMS归一化；异步发布‑订阅架构；对语义记忆进行时间衰老嵌入；训练时采样不同步时间戳和执行延迟。

**📊 数据集**

使用的公开数据集与仿真平台：LIBERO（40项任务）、LIBERO‑Perturb、Dynamic Bump（Robosuite）、DROID（15 Hz实地数据）以及RoboLab的光学真实感仿真；在Frank 机器人上进行实机测试。

**📈 对比分析**

与传统同步VLA架构做对比。ARP在准静态任务上成功率提升18%，在动态扰动任务中表现为3–5倍更高的成功率；同时VLM推理频率可降低10倍而无明显性能损失，显著提升实时反应与计算效率。

**⚠️ 局限性**

局限性包括：①异步进程间通信与延迟控制仍需优化；②语义记忆的持久化与长时序任务的支持尚未充分；③对超大规模模型的兼容性与调优策略待进一步研究。

---

## 687. Catalyzing Interactive HTML Generation Capabilities via High-Fidelity Reinforcement Learning

**arXiv ID:** 2611.05151 | [PDF](https://arxiv.org/pdf/2611.05151v1)

**作者:** Xujia Wang `[一作]` (Tsinghua University), Juanzi Li `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过强化学习框架 WebCatalyst 提升大型语言模型在交互式 HTML 生成中的功能与视觉质量。

**💡 创新点**

提出三大高保真反馈组件：GUI Assessor（多步交互评估）、Visual Anchor（对比式视觉惩罚）和 Action-Trace Credit（基于运行时调用链的 token 级信用分配），并将其集成进 GRPO 算法，实现了功能、视觉与信用分配的统一提升。

**🔧 技术方法**

采用浏览器交互式 RL 框架，结合 Playwright 执行、GPT-5.5/5.6 作为评判模型、Chrome DevTools 进行调用栈追踪，使用 GRPO 进行策略优化，并在 token 级别上实现优势再分配。

**📊 数据集**

训练数据来自 Reddit 上关于游戏开发的帖子，筛选并重写为直接用户请求，最终收集不到 1K 条游戏相关的交互式请求；评测数据使用 WebGen-Bench（网页生成）和 Cookie-Bench Mini Games（游戏级交互）两个基准。

**📈 对比分析**

与多种前沿模型（MiniMax-M2.7、GLM-5.1、Kimi-K2.6 等）对比，WebCatalyst 在 Cookie-Bench Mini Games 总分从 68.8 提升至 75.3，在 WebGen-Bench 准确率从 48.8 提升至 56.9，平均排名 4.0，超越 Claude Opus 4.7 的多数指标，表明高保真反馈能显著提升跨域交互式 HTML 生成性能。

**⚠️ 局限性**

仅针对直接生成任务；未针对更复杂的规划/迭代调试/工具使用流程进行优化；依赖基模型已具备的交互能力；训练集中只包含游戏提示，未探索任务混合与数据分布对泛化的影响；潜在的误用风险（如生成钓鱼页面、恶意客户端行为）需进一步安全评估。

---

## 688. Beyond Foundation Models: PFN Priors Improve Tabular Hyperparameter Optimization

**arXiv ID:** 2611.05141 | [PDF](https://arxiv.org/pdf/2611.05141v1)

**作者:** Artem Chistyakov `[一作]` (HSE University), Artem Babenko `[通讯]` (Yandex)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究如何利用合成先验对传统表格模型进行超参数元调优，提出 PriorTune 方法。

**💡 创新点**

创新点在于将 TabPFN 先验生成的合成数据用于超参数子集选择，使得无需真实数据即可获得高质量配置，并在 BeyondArena 取得领先非预训练模型的成绩。

**🔧 技术方法**

使用贪心前向选择与基于胜率的评估，结合 TabICLv2 等合成先验，支持 LightGBM、LightGBM-TabPrep、MLP 与 TabPack 等模型。

**📊 数据集**

使用 TabICLv2 生成 4000 个合成任务，并在 TabArena（51 个 IID 数据集）和 BeyondArena（142 个多样化拆分）上评估；对比 TabRepo 真实数据集。

**📈 对比分析**

采用 Elo 排名与推理延迟对比，Meta-调优后的 LightGBM-TabPrep 在 BeyondArena 获得最高 Elo（1313）且推理仅 4.4 ms，显著优于 TabPFN-3、RealMLP 等。

**⚠️ 局限性**

局限性包括未对表格基础模型实验；合成任务规模受限（≤60k 行、100 特征）；仅针对调优规则 T，集成规则实验成本高；需要重新网格为新搜索空间。

---

## 689. HLO-AMP: Hierarchical Loco-Manipulation from Object-Aware Motion Priors

**arXiv ID:** 2611.05113 | [PDF](https://arxiv.org/pdf/2611.05113v1)

**作者:** Jean-Pierre Sleiman `[一作]`, Farbod Farshidian `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种分层强化学习框架，用低层运动模仿策略在含有目标物体的场景中学习人类动作，并由高层无参照策略在此运动先验上进行引导，实现全身协同的双手搬运任务；

**💡 创新点**

创新点在于：①低层策略通过包含物体的 MoCap 训练，直接学习真实分布的多接触力耦合，避免了传统奖励塑造和物体缺失的局限；②高层策略采用约束式命令生成，仅在低层训练分布内工作，实现零分布外输入，且无需要额外的参考轨迹；③系统实现了从仿真到真实的零样本迁移，并在多尺寸、多质量、复合目标及突发扰动下表现出稳健性；

**🔧 技术方法**

主要技术包括：层级强化学习（PPO）、运动模仿（ZEST）、人类 MoCap 轨迹重定向至 Unitree G1、任务空间命令接口、数据集包络裁剪、阶段化奖励、域随机化、非对称演员-评论家结构等；

**📊 数据集**

使用了人类操纵盒子的视频捕捉 MoCap 数据，并通过交互网格重定向至仿真平台；此外，还构造了多尺寸、多质量的盒子样本来进行训练和测试；

**📈 对比分析**

在仿真与硬件上与三种基线（单体端到端、无物体的命令跟踪 WBC、使用力模拟的 WBC）进行对比，本文方案在 20 s 任务下达到了 80.7 % 的成功率、26.8 % 的丢失率和 1.00 的自然度-平滑度指数，显著优于所有对比方法；

**⚠️ 局限性**

局限性包括：需要外部 MoCap 进行物体姿态感知；低层运动库仅覆盖四类基本动作，限制了可扩展性；自然度评估仅为定量指标，缺乏用户主观评测；在接近机器人推力极限时仍存在抓取失效和热关机风险。

---

## 690. Recognizing Label-Connected Graphs

**arXiv ID:** 2611.05144 | [PDF](https://arxiv.org/pdf/2611.05144v1)

**作者:** Ralf Klasing `[一作]` (CNRS), Paul G. Spirakis `[通讯]` (University of Liverpool)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在固定的无向多重图上，如何安排一次性呼叫（或时间标签）使得所有参与者最终掌握所有信息，并给出多种参数化复杂度结果与多项式/单指数算法。

**💡 创新点**

创新点在于提出了树收缩规则和根独立环（Rooted Independent Cycle）技术，实现了针对树宽、树独立度、回路宽等参数的单指数算法，并在极大度为五的平面图上证明了NP‑难性及ETH下的时间下界。

**🔧 技术方法**

主要技术包括经典的 T‑witness / M‑witness 结构证明、稀疏图与棍子游戏、秩基约简（rank‑based）动态规划、线性 matroid 表示与随机投影、以及多项式时间的树宽与结构参数上限分析。

**📊 数据集**

本文没有使用真实数据集，所有实验/验证均通过理论分析与合成图（如最短回路、平面图、K6‑自由图等）完成。

**📈 对比分析**

通过理论证明，算法在树宽为 t 时可在 2^t·n 甚至 2^{√n} 时间内解决；在最大度 4 的多重图上实现了 O(n^3) 的多项式算法；在最大度 5 的平面图上证明了 NP‑难性与 2^{o(n)} 的下界；而在余量 δ 下提供了 2^δ·n^3 的随机化 FPT 方案。

**⚠️ 局限性**

限制包括：随机化 FPT 算法未能完全消除随机性；对树宽的 2^t 依赖仍不是最优；未给出多项式时间内的回路宽或树独立度上界算法；对极大度为 5 的一般图的多项式解仍未解决。

---

## 691. PadLoc: Training-Free Floor-Plan Localization by Sparse Geometric Matching

**arXiv ID:** 2611.05172 | [PDF](https://arxiv.org/pdf/2611.05172v1)

**作者:** Marian Bookhahn `[一作]` (HTW Berlin University of Applied Sciences), Frank Neumann `[通讯]` (HTW Berlin University of Applied Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种无训练的地图定位后端PadLoc，利用被遮罩的归一化互相关（NCC）匹配墙面方向并通过高斯混合贝叶斯滤波与一致性读取解码位姿；

**💡 创新点**

核心创新在于将稠密的（x,y,θ）格子滤波替换为稀疏的墙面方向候选和顶级NCC峰值的多假设后端，并通过一致性求和在在线有限滞后下恢复绝对位姿；

**🔧 技术方法**

主要技术包括被遮罩的FFT域NCC、基于墙面方向的离散朝向候选、窗口化目标拼接、GMM贝叶斯滤波、以及累积一致性峰值解码；

**📊 数据集**

实验使用Gibson室内导航数据、Bonn LiLocBench以及Hilti-Trimble大楼定位挑战数据集；

**📈 对比分析**

与F3Loc、UnLoc以及Z-FLoc等基准在相同深度前端下进行对比，PadLoc在SR@1m、SR@2m等指标上取得更高成功率（例如Gibson上达1.0 SR@1m、Hilti上0.90–0.98 SR@2m），并在多尺度地图上保持较低误差；

**⚠️ 局限性**

局限性包括对短期稀疏深度轨迹的朝向歧义、漂移下的错误基底保持、对家具或长期结构变化的依赖以及在极端高漂移或大尺度环境中需要更大窗口或额外前端处理。

---

## 692. AgentAM: An AI Agent for Spatter- and Defect-Aware Process Optimization in Laser Powder Bed Fusion

**arXiv ID:** 2611.05171 | [PDF](https://arxiv.org/pdf/2611.05171v1)

**作者:** Olabode T. Ajenifujah `[一作]` (Carnegie Mellon University), Amir Barati Farimani `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于Model Context Protocol（MCP）的LLM代理，为SS316L激光粉床熔融（LPBF）过程提供自然语言访问CFD驯化的喷溅量和熔池几何预测，并实现扫描与约束优化。

**💡 创新点**

创新点包括：①将CFD驯化的多项式模型封装为可调用工具并通过MCP与LLM协同；②提出基于几何阈值的缺陷分类和margin‑aware（安全裕度）优化器；③设计并发布了145问卷的代理层评估基准，系统测量解析、工具调用、精度与可靠性。

**🔧 技术方法**

使用技术包括：Claude Opus 4.5 LLM、MCP标准与JSON‑RPC工具调用、线性/多项式回归驯化、线扫与网格扫、约束多目标优化、答案守卫规则（防止伪造输出）。

**📊 数据集**

采用的主要数据集是279条SS316L单轨FLOW‑3D CFD模拟（功率160–430 W、扫描速度100–1500 mm/s），从中提取喷溅量、熔池深宽长等特征。

**📈 对比分析**

通过交叉验证和自测，驯化模型在熔池几何上的R²≥0.95、喷溅量R²=0.83；代理在725个基准查询中解析准确率98.5%，优化误差≤1.0%；无工具情形的模型生成80%伪造输出；答案守卫在基准中有效拦截92%以上伪造。

**⚠️ 局限性**

局限性在于：仅考虑功率和扫描速度两个输入，未涵盖光斑尺寸、扫描间距、层厚等关键变量；缺陷分类基于模拟阈值，尚未与实际构建结果验证；margin‑aware优化的鲁棒性评估基于假设误差；答案守卫无法捕捉无单位错误陈述；多步骤推理和闭环实时调参功能尚未实现。

---

## 693. Basin2Vec: Learning Basin-Centric Embeddings for Scalable Hydrological Modeling

**arXiv ID:** 2611.05164 | [PDF](https://arxiv.org/pdf/2611.05164v1)

**作者:** Muhammad Talha `[一作]` (Michigan State University), A. Pouyan Nejadhashemi `[通讯]` (Michigan State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于自监督学习的 Basin2Vec 框架，用来从分布式的水文气候和地貌数据生成基于河道尺度的低维表示，以实现跨时空的流量预测。

**💡 创新点**

创新点在于将河道尺度编码与空间尺度感知、元数据融合以及针对河道关系（相同河道、时间邻近、空间重叠/嵌套）的关系感知对比学习目标相结合，构建了既能捕捉局部空间异质性又能保持河道尺度信息的表示。

**🔧 技术方法**

使用的技术包括掩膜感知的卷积编码器、Transformer 时序建模、元数据门控融合、面积正则化和多关系对比学习（硬正、时间正、空间正、元数据调节的负样本），以及负边距损失。

**📊 数据集**

使用 CAMELS-US 河道网络的日观测流量与 Daymet 气候数据，以及 CAMELS 属性、GAGES-II 等辅助地理资料进行预训练和评估。

**📈 对比分析**

与传统的 CAMELS 属性、AlphaEarth、TESSERA、SatCLIP 以及 hydrology‑specific KG‑SSL 对比，Basin2Vec 在所有 5 种下游预测架构（LSTM、TFT、TimeXer、Timer‑XL、iTransformer）以及两种任务（时间泛化和零样本空间转移）中均取得最高的 NSE/KGE，平均相对提升 8.3% 及 3.5×。

**⚠️ 局限性**

局限包括仍需大量 gridded 数据预处理、对极端/缺失的流量缺乏鲁棒性，以及在极端规模/地貌极端的河道上效果未充分验证。

---

## 694. Macrocanonical Generator Networks: Data-Efficient Neural Surrogates for Amortized Physics Simulation

**arXiv ID:** 2611.05157 | [PDF](https://arxiv.org/pdf/2611.05157v1)

**作者:** Niall Jeffrey `[一作]` (King's College London), Benjamin D. Wandelt `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出宏观宏正则生成器网络，将多尺度统计特征在期望上匹配，从而在极少样本的情况下快速生成符合物理过程的高质量模拟样本。

**💡 创新点**

创新点在于：①将微正则逐样本优化迁移为一次性神经网络学习；②证明训练产生的梯度预条件化样本空间梯度下降，继承微正则对称保持特性；③通过残差初始化与低学习率实现局部体积保持，避免模式坍塌；④实现一次前向传播即可生成新样本，极大加速。

**🔧 技术方法**

使用残差卷积网络（圆形填充U-Net）作为生成器，特征统计采用波形相位和谐波（WPH）；训练目标为特征均值匹配，优化采用Adam；分析中用核预条件化梯度和熵变化理论。

**📊 数据集**

在两类周期性稳态数据集上评估：单个128×128磁流体动力学（MHD）湍流快照和最多100个Quijote宇宙学网络切片。

**📈 对比分析**

与WGAN‑GP、流匹配、DDPM及传统微正则梯度下降对比：宏正则生成器在仅一条训练样本时即可产生非复制样本，特征分布与真实模拟相近；生成速度超过6个数量级，比参考模拟快4×10³倍，比微正则快3–4个数量级；在暗能量推断任务中，预训练后在无真实样本的情况下已显著提升后验负对数似然。

**⚠️ 局限性**

局限性：①依赖所选特征统计，若统计不充分则生成样本可能失真；②方法针对周期性稳态过程，对非稳态或非周期边界的过程适用性有限；③缺乏对收敛到最大熵分布和避免坍塌的严格理论保证。

---

## 695. Sharp Bounds for Truthful-in-Expectation Scheduling

**arXiv ID:** 2611.05134 | [PDF](https://arxiv.org/pdf/2611.05134v1)

**作者:** Yunpeng Li `[一作]` `[通讯]` (Chinese University of Hong Kong), Yunpeng Li (Chinese University of Hong Kong)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

本文研究在非相关机器上对无拆分任务的随机化调度机制，提出最优的truthful‑in‑expectation（期望真诚）机制，并给出了精确的近似比值。

**💡 创新点**

创新点在于首次证明任意机器数 n≥7 时最优近似比为 (n+1)/2，且给出了相匹配的机制；对两、三机情形给出精确常数，三机上给出可实现的二值机制。

**🔧 技术方法**

采用的技术包括关联抽样、槽分配与匹配分解、逆平方分配、凸分析、概率极限与测度论证明，以及数值证书。

**📊 数据集**

该研究为理论工作，无使用实验数据集；所有结果基于数学证明。

**📈 对比分析**

通过与传统 VCG、确定性真诚机制及已知上界 (n+5)/2 等进行比较，证明新机制在期望 makespan 上取得最优或近似最优性能。

**⚠️ 局限性**

局限在于对四、五、六机的精确比值仍为上界，且实现机制在实际大规模环境中的计算复杂度尚未深入评估。

---

## 696. PAC Learning with Multiple Correct Answers

**arXiv ID:** 2611.05133 | [PDF](https://arxiv.org/pdf/2611.05133v1)

**作者:** Steve Hanneke `[一作]` (Purdue University), Kun Wang `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

研究多标签PAC学习中存在多重正确答案时的可学习性，考察全集、bandit以及单一正确标签（示范）三种反馈模型。

**💡 创新点**

提出了新的维度概念——通用Natarajan维度、Packed维度，以及交集类的可学习性条件，揭示示范学习往往比全集或bandit学习更难，且需要更高样本复杂度。

**🔧 技术方法**

利用一系列组合论工具，包括产品破碎、DS证人、冲突块、转移规则、样本压缩方案和多层极大化等，构建了证明与上界。

**📊 数据集**

本文未使用标准数据集，而是以理论构造和对角线例子验证维度与学习复杂度之间的关系。

**📈 对比分析**

与传统的多类别PAC学习相比，本工作通过维度分析给出实现和不可学习性的完整阈值，证明在有限标签空间下全集与bandit学习等价，但示范学习需更高阶指数样本复杂度。

**⚠️ 局限性**

局限性在于无法在标签数无上界时给出维度表征，且对非退化类的非均匀兼容性假设提出了进一步研究的需求。

---

## 697. Do Ray-Traced Cities Make a Better Channel Prior for an Unseen Site? A Pre-Registered Study in Simulation, and a Test on Measured Channels that Reverses It

**arXiv ID:** 2611.05128 | [PDF](https://arxiv.org/pdf/2611.05128v1)

**作者:** Krish Patel `[一作]` `[通讯]` (University of Waterloo), Krish Patel (University of Waterloo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究在基站新部署场景下比较了从多城市射线追踪（ray‑traced）城市池化得到的协方差先验与传统 3GPP CDL 先验在信道估计与短期预测中的表现，并在一组真实测量信道（DICHASUS 1.272 GHz）上验证其有效性；同时评估了残差 CNN 等学习方法是否能提升性能。

**💡 创新点**

创新点在于：①预先注册的 leave‑one‑city‑out 仿真实验，系统评估多城市射线追踪先验在未见城市中的保险效果；②将多城市先验与 3GPP CDL 先验在同一测量数据上直接比较，揭示学习方法不一定优于经典 LMMSE；③在多种测量环境（街道、实验室房间、工厂大厅）下探究 CDL 先验失效的场景。

**🔧 技术方法**

使用了 LMMSE 信道估计、残差 CNN 细化、Sionna RT 及差分射线追踪、差分散射实验、差分光线追踪校准、基于 3GPP CDL-A~E 的统计模型，以及多城市的离线地图与城市图像生成。

**📊 数据集**

数据集包括：14 个基于城市地图生成的射线追踪场景（Munich、Paris‑Étoile 等），3GPP CDL-A~E 统计模型，DICHASUS 1.272 GHz 的三组 2×8 阵列实测数据，及三种室内实验（实验室房间、工厂大厅、街道）。

**📈 对比分析**

对比方法：在仿真中进行 leave‑one‑city‑out；无现场测量时评估射线追踪先验与 CDL 先验的 NMSE 误差；在预测任务中对比 CNN、空间‑时域 Wiener、tone 方法等；在实测信道上直接测量 NMSE 与 3GPP CDL 的差距。结果显示：在仿真中聚合射线追踪先验仅比最佳估计差 0.6 dB，学习模型最多提升 0.65 dB；预测任务中经典 tone 方法在所有城市均优于 CNN；在实测信道上，聚合射线追踪先验比 CDL 差 3.2–4.1 dB，加入散射后可缩小到 0.3–0.5 dB；室内实验中 CDL 先验与现场训练模型差距在 2–5 dB 之间。

**⚠️ 局限性**

局限性：①仅在单一街道（1.272 GHz）与单一阵列尺寸上测试，缺乏多频段与多阵列验证；②射线追踪场景缺失街道杂物、车辆、树木等散射源，导致与真实信道差距；③实测数据未覆盖移动用户，仅使用静态点测量；④室内实验未对射线追踪模型进行充分校准；⑤测量信道存在时延漂移与相位误差；⑥城市样本有限，无法保证在所有场景下聚合先验优于 3GPP CDL。

---

## 698. A Gated Spec-Driven Workflow: An Audit-Trail Study of Coding-Agent Work

**arXiv ID:** 2611.05094 | [PDF](https://arxiv.org/pdf/2611.05094v1)

**作者:** Rodrigo Figueiredo Abdo `[一作]` (Federal Institute of Brasília), Pedro Carvalho Brom `[通讯]` (Federal Institute of Brasília)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在多个项目中对 Claude Code 的 SDD 流程进行观察和日志分析，研究了其控制机制（门控、审阅子代理、执行器、块检查等）对软件与非软件交付物的影响。

**💡 创新点**

创新点在于系统性地评估了几乎全由模型执行的控制机制，并揭示了块检查与停止规则、审阅子代理行为、以及中断恢复等方面的弱点与改进空间。

**🔧 技术方法**

采用的技术包括基于 Markdown 的 SDD 工作流、Claude Code 的多代理架构（审阅者、指挥官、执行者）以及文件清单与时间戳的块检查逻辑。

**📊 数据集**

使用的数据集是作者自行运行的 6 个项目中的 14 个功能（feature）产生的日志、转录文本、审阅报告和代码仓库。

**📈 对比分析**

由于缺乏对照组，本文并未进行严格的性能比较；通过计数和案例分析，表明执行器大多数情况下遵守文件范围，但指挥官和审阅子代理的偏差仍然存在。

**⚠️ 局限性**

局限性包括仅来自作者自身实验，缺乏外部验证，块检查和审阅结果仅由模型生成且无独立核查，且未评估基于工具而非模型执行的控制方案。

---

## 699. Search and Verification for Strategyproof Facility Location on Metric Graphs

**arXiv ID:** 2611.05092 | [PDF](https://arxiv.org/pdf/2611.05092v1)

**作者:** Krzysztof Rogowski `[一作]` (University of Warsaw), Marcin Dziubiński `[通讯]` (University of Warsaw)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在连续度量图上设计与验证随机策略无偏（SP）设施定位机制，提出了片段线性归因（PL归因）框架并利用强策略无偏（SSP）简化搜索，给出了全局近似比上界与下界，并在环图和theta图上得到近似比上界低于已知结果。

**💡 创新点**

创新点包括：① 将机制表示为可在有限多面体上线性混合的PL归因，实现了对无限维策略空间的有限描述；② 引入SSP并给出其在归因类内的完全线性约束，从而将无穷多约束化为有限线性规划；③ 结合精确算术的全局非负性检验与有限域下的成本保持归约，既给出有效的上界，又提供所有SP机制的下界；④ 通过LLM辅助的搜索与手工验证相结合，首次给出对连续图上SP机制的实际性能边界。

**🔧 技术方法**

技术方法包括：片段线性/多项式结构分析、二次非负性检验、线性规划与对偶、强策略无偏的方向导数约束、有限域机制设计LP、成本保持归约、全局精确算术验证以及样本化近似比优化。

**📊 数据集**

主要数据集是离散化的连续度量图：单位边长的循环图、等长路径的theta图（K₂,₃）、完全二部图K₂,₆等，针对不同人数的代理（3、4、6）进行实验。

**📈 对比分析**

与已有的随机独裁（RD）和比例圆距离（PCD）等基准机制比较，验证得到的上界如3代理环图1.10747、theta图1.2028、K₂,₆ 1.140204等，均优于已知的5/4、4/3等上界；下界通过有限域对偶得到1.059117、1.078216、1.140204等，证实上界与下界差距均小于0.05，说明搜索结果逼近最优。

**⚠️ 局限性**

局限性：① 归因类与SSP约束可能无法覆盖所有最优SP机制，搜索结果仅在选定的三角剖分和选择器集合内；② 全局非负性检验与精确算术验证对大规模图和更多代理数的计算量极高；③ 依赖精确算术与多项式非负性判定，难以扩展到更复杂或高维度的度量图；④ 低阶样本化近似比可能低估全局性能，需额外全局验证。

---

## 700. LatentTriad: Pre-Action Risk Representations for Malicious Agent Skill Detection

**arXiv ID:** 2611.05088 | [PDF](https://arxiv.org/pdf/2611.05088v1)

**作者:** Zhe Yu `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Simin Chen `[通讯]` (George Mason University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种名为 LatentTriad 的预动作安全检测框架，利用大语言模型在加载技能包时的隐层状态进行风险评估，无需生成输出或执行代码。

**💡 创新点**

创新点在于：①首次证明预填充（prefill）阶段的隐藏状态就能捕获任务与技能操作的一致性信息；②构建三分支风险读数（Boundary、Operation、Geometry）并共享一次前向推理；③通过对 Operation 的组级方向校正和 Geometry 的善意关系残差，显著提升检测准确性。

**🔧 技术方法**

技术手段包括：大模型预填充、隐藏状态抽取与池化、组内对比度投影、善意关系残差建模、基于 L2‑logistic 的多分支决策融合，以及固定阈值策略。

**📊 数据集**

数据集：5,830 条技能包，来自五个公开生态系统（包括 MalSkillBench 等），在七种不同 backbones（Llama‑3.1‑8B、Qwen2.5‑7B、Mistral‑7B‑v0.3 等）上进行训练与评估。

**📈 对比分析**

与 AgentLens、RouteGuard、SkillSieve、Regex 等基线对比，LatentTriad 在源宏 F1 上达到 93–94.6%，在聚合 F1 上达到 99.0–99.2%，显著高于所有对照方法；FPR 仅 1.1–1.4%，检测延迟 259–284 ms，显示了高效、低误报的性能。

**⚠️ 局限性**

局限性包括：对合成符号重排的鲁棒性不足（仅在近似随机情况下性能下降）；在极端攻击重写场景下可能诱发误报；仍然依赖于上下文表征，缺乏对跨文件结构逻辑的抽象推理；以及仅在预动作阶段起效，无法替代完整的运行时监控与权限控制。

---

## 701. Correctness Firewalls for Synthesized Database Components

**arXiv ID:** 2611.05173 | [PDF](https://arxiv.org/pdf/2611.05173v1)

**作者:** Mohammad Javad Amiri `[一作]` (Stony Brook University), Zijie Zhao `[通讯]` (University of Pennsylvania)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并验证了三类“正确性防火墙”（advisory、guarded、domain‑certified），让基于LLM的编码代理能够在不破坏数据库正确性的前提下生成高性能的卡方估计器、学习索引和文件特定压缩算法。

**💡 创新点**

创新点在于将代理生成代码与数据库系统的正确性设计相结合，提出三种轻量级防护范式，既保留了传统DBMS的正确性保证，又利用LLM提升性能。

**🔧 技术方法**

使用的技术包括大语言模型（OpenAI Codex、GPT‑5/6）、约束解码、快速检验（例如搜索结果检查）、基于DSL的安全执行与可选的形式化验证。

**📊 数据集**

实验所用数据集包括 JOB 与 JOB‑Complex（用于 cardinality 估计）、整数与字符串（URL）排序数据（用于学习索引）以及 OpenStack 日志和 Yelp 数据（用于文件压缩）。

**📈 对比分析**

与基准系统比较时，代理生成的 cardinality estimator 使查询计划性能提升 33%–68%，学习索引搜索速度可达普通二分搜索 3 倍，压缩算法在保持解压速度的前提下比 Zstd 压缩率高 1.5 倍，整体性能显著优于传统手工实现。

**⚠️ 局限性**

主要局限包括：仍无法保证代理代码绝对安全（如无限循环、外部状态修改）、仅适用于有限输入域、缺乏对代码更新的生命周期管理、需额外的安全隔离与形式化验证，且与全自主生成系统相比存在“正确性成本”与性能折衷。

---

## 702. Perturb, Repair, Verify: Self-Play Vision-Language Verifiers for Compositional Understanding

**arXiv ID:** 2611.05158 | [PDF](https://arxiv.org/pdf/2611.05158v1)

**作者:** Zijun Huang `[一作]` (University of Southern California), Cees G. M. Snoek `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个自我对弈框架，利用同一模型的扰动‑验证‑修复‑验证循环来训练视觉语言验证器。

**💡 创新点**

创新点在于消除了对外部强监督的依赖，采用可控局部扰动和内部反馈生成自适应训练信号。

**🔧 技术方法**

使用角色条件的多模态策略、DrGRPO优化、以及Qwen系列模型作为共享骨干。

**📊 数据集**

训练数据为48k例的ShareGPT‑4o‑Image‑Style图文对，评测涵盖ARO、VL‑CheckList、ColorSwap、FG‑OVD等细粒度对比以及Winoground、MMVP、MMBench等广义视觉语言基准。

**📈 对比分析**

与同基线及多种基准方法对比，改进后在细粒度组合验证上平均提升至86.85分，显著优于同基线及部分现有方法，且在MMVP、Winoground上也取得提升。

**⚠️ 局限性**

局限包括仅使用四种扰动类型、可能在更复杂的视觉细节或其他任务上回退、以及对初始图文对的依赖。

---

## 703. When Cell Crops Share a Parent: Measuring and Isolating Acquisition Leakage in Cervical Cytology Benchmarks

**arXiv ID:** 2611.05145 | [PDF](https://arxiv.org/pdf/2611.05145v1)

**作者:** Snehasish Ghosh `[一作]`, Nibaran Das `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过前注册实验评估公开宫颈细胞学数据集在不同分割方式下的模型性能，尤其检验 crop‑level 分割导致的性能提升是否源于数据泄露。

**💡 创新点**

创新点在于将泄漏效应与多样性效应通过因子实验系统分离，证明父级身份的线性可解码并非泄漏来源，并强调在细胞级别进行分组分割的重要性。

**🔧 技术方法**

使用 ImageNet 预训练的 12 种 CNN 骨干（ResNet、DenseNet、MobileNet、EfficientNet、RegNet、ConvNeXt、Swin、MaxViT、ViT、DINOv2 等）进行微调；配合梯度反转对抗、LEACE 线性擦除、Grad‑CAM、颜色统计迁移和混合效应模型统计分析。

**📊 数据集**

实验数据集包括公开的四个宫颈细胞学数据集：SIPaKMeD、CRIC、Cx22 和 Herlev。

**📈 对比分析**

采用 crop‑level（P1）与父级分割（P2）两种拆分方案，在同一测试集上进行混合效应模型比较。结果显示，SIPaKMeD 在 crop‑level 分割下准确率提升约数个百分点（约 7–9%），CRIC 亦出现类似提升；因子实验进一步证明此提升主要由泄漏驱动，而非仅仅是父级多样性的增加。

**⚠️ 局限性**

局限性包括：仅以公开数据集深层单元为父级，未检验患者级泄漏；实验仅采用单一训练配置，未覆盖多样的训练策略；因子实验规模受限，未完全阐明泄漏机制的具体特征。

---

## 704. Towards Multi-Method Evaluation of AI in Health: Integrating Benchmarks, Autoraters, and Expert Evaluation

**arXiv ID:** 2611.05127 | [PDF](https://arxiv.org/pdf/2611.05127v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 705. From Opinions to Commitments: Unstable Assignments and Silent Choices in AI-Mediated Proposals

**arXiv ID:** 2611.05135 | [PDF](https://arxiv.org/pdf/2611.05135v1)

**作者:** Sachit Mahajan `[一作]` (ETH Zurich), Dirk Helbing `[通讯]` (ETH Zurich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究对四款大型语言模型（Qwen3、Gemma、Mistral、GPT‑5.6 Terra）在DeliberationBank和Polis数据上进行重复生成实验，评估相同意见在不同随机种子或意见顺序下的分配变化，检验四种常见的稳定性保护措施，并分析调解目标对观点覆盖率的影响。

**💡 创新点**

创新点在于提出并实施“重跑审计”方法，揭示LLM生成的“沉默选择”，量化意见顺序和调解目标导致的分配不稳定性，并基于此设计一套可操作的透明度、可争议性与参与者自治的要求。

**🔧 技术方法**

技术手段包括使用四种大型语言模型作为调解器，利用LLM评估器（GPT‑6.1 Sol、Luna、Claude Haiku）与人工注释相结合来识别和分类分配与覆盖；并通过Jaccard相似度、关系分类等指标量化重复运行的差异。

**📊 数据集**

数据集涵盖DeliberationBank（十个公共政策问题，每题约320条英文回答）与Polis（十个英语议题，至少30条评论），每个任务从中抽取30条意见进行实验。

**📈 对比分析**

比较方法通过五次重跑（不同种子或顺序）计算混合集比例、派遣一致度（Jaccard）以及关系类别；四项保护措施的效果通过“差异比例”与15%提升阈值评估；实验结果表明无任何保护措施显著降低分配不稳定性，而调解目标差异导致少数派覆盖率相差约30–40个百分点。

**⚠️ 局限性**

局限性包括仅在英文30条意见的单轮生成环境下评估，未探讨更大群体、长文本或多轮交互；LLM评估可能过度报告差异；未检验生成分配是否合理或被参与者接受；样本仅限DeliberationBank和Polis，缺乏跨文化验证。

---

## 706. High-fidelity training-free photovoltaic module mapping from multi-view UAV imagery

**arXiv ID:** 2611.05131 | [PDF](https://arxiv.org/pdf/2611.05131v1)

**作者:** WenHao Duana `[一作]`, Yuntian Chen `[通讯]` (Eastern Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了 MV-PVMapper，一个训练免费框架，用预训练的 SAM3 模型与 UAV 摄影测量几何信息实现光伏模块的完整实例恢复和精确分割。

**💡 创新点**

创新点在于结合多视角反馈、每模块坡度校正以及基于模块尺寸的几何评分，实现无监督的高精度模块地图生成。

**🔧 技术方法**

使用的技术包括 Vision Foundation Model SAM3、无人机多视角图像、相机位姿和 DSM、坡度校正投影、几何评分和交叉视角投影反馈。

**📊 数据集**

使用的实验数据集是三栋屋顶光伏电站的 5 方向倾斜 UAV 影像，包含共 7711 个参考模块，GSD 约 1 cm。

**📈 对比分析**

与基于正射影像的 DOM-IterSeg 和多视角像素投票 MV-PixelVote 对比，MV-PVMapper 在宏观 AJI 97.55%、AP_95 95.67%，显著优于两者。

**⚠️ 局限性**

局限包括需要完整的摄影测量重建和代表性模块尺寸，无法处理极不规则安装、遮挡严重或多种模块规格的场景，且实现多轮反馈较慢。

---

## 707. WayTrace: A Map-to-Language-to-Trace Hierarchy for Embodied Urban Navigation

**arXiv ID:** 2611.05129 | [PDF](https://arxiv.org/pdf/2611.05129v1)

**作者:** Manthan Patel `[一作]` (ETH Zurich), Cesar Cadena `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `51c0528b-f690-4182-ae60-bb5f046c276c` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一套分层系统，将网页地图的逐步导航指令转换为原子语言命令，再由图像空间轨迹预测器生成轨迹，最后由强化学习策略跟踪执行，实现长距离城市环境导航。

**💡 创新点**

创新点包括：① 将地图到语言再到轨迹的层次结构；② 用自动标注的街景图像训练的轨迹预测器，避免需机器人数据；③ 将高层决策与低层执行解耦，显著提升对GNSS误差的鲁棒性；④ 在真实城市中完成多公里任务，验证系统可迁移性。

**🔧 技术方法**

采用了多种技术：前端使用开箱即用的视觉‑语言模型（Qwen3.8‑Flash‑Next、RynnBrain‑9B）；轨迹预测器基于两阶段自回归再压缩为单调用；RL导航策略在深度图像上学习跟踪；利用GNSS、激光/惯性里程计、卫星地图；后端利用离线训练的仿真环境和实时的深度感知。

**📊 数据集**

数据集主要有：① 1.19 M 条自动标注的街景轨迹样本（Mapillary 51,907 条序列，结合重建走廊与意图标注）；② 仿真老城区数据（来自瑞士政府地理数据、OpenStreetMap、街景图像），覆盖 0.91 km² 的城市环境；③ 真实世界多公里路程的日志用于评估。

**📈 对比分析**

与现有基准（CityWalker、VAMOS、OmniVLA、TIC‑VLA、Foresight、VENTURA）比较：在完美 GNSS 下系统成功率 83.3%（低于 20% 的基准）；在 GNSS 噪声 7 m RMS 时仍保持 63% 成功率，而基准跌至 10%；轨迹预测器在 NaviTrace 评分上达 83.2，远优于 57 的基准；在真实城市中完成 1.37 km 的任务，100% 成功。

**⚠️ 局限性**

局限性包括：① 调度器（orchestrator）约 5% 的关键决策错误，需更强的 VLM 或微调；② 触发新命令的 1.3–1.5 s 延迟导致对狭窄通道、车辆或红灯等即时情况反应不足；③ 轨迹预测仅基于前置摄像头，无法处理无视角上下文的情形；④ 依赖 GNSS 识别当前步长，对长时间失去 GNSS 信号时的距离估计不稳定；⑤ 目前仅对单轨机器人验证，需进一步验证对人形或其他平台的适配。

---

## 708. Towards an Optimal Control Foundation of Early Exits - Turnpikes and ResNets

**arXiv ID:** 2611.05120 | [PDF](https://arxiv.org/pdf/2611.05120v1)

**作者:** Jens Püttschneider `[一作]` (Hamburg University of Technology), Timm Faulwasser `[通讯]` (Hamburg University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了在残差网络（ResNet）中联合训练早期出口的前向传播，并用最优控制理论解释其出现的“转道奇现象”，给出了早期退出的深度上界以及对损失、准确率和不确定性指标的理论保证。

**💡 创新点**

创新点包括：①将早期出口的联合训练视为离散最优控制问题；②在交叉熵损失下首次提出损失、准确率和熵/最大softmax概率的转道奇性质；③给出与退出阈值和阶段成本权重相关的显式上界；④通过实验验证了不同深度网络在训练、测试及早期退出时表现出的转道奇相似性。

**🔧 技术方法**

主要技术手段有：最优控制理论（阶段成本、终端成本、转道奇理论）、残差网络的动态系统建模、交叉熵损失的指数衰减假设、联合损失函数与早期出口的权重平衡、熵与最大softmax概率的退出策略。

**📊 数据集**

实验使用了CIFAR-10、CIFAR-100和TinyImageNet三个公开数据集，构建了不同深度（18、36、54层）以及不同早期出口间隔（M=1或M=6）的ResNet模型。

**📈 对比分析**

实验结果表明：①训练损失、准确率以及熵/最大softmax概率随层数呈指数下降，符合转道奇预测；②不同深度网络在给定退出阈值时的准确率几乎相同，验证了转道奇相似性；③阶段成本权重γ越小，非转道奇层数增多；整体性能优于无早期出口的模型，且在测试集上仍保持高准确率。

**⚠️ 局限性**

局限性包括：①理论分析基于“可达性”假设，需对更广泛的架构（非残差网络、变压器等）验证；②仅对训练数据给出保证，泛化误差与统计学习理论结合尚未完成；③交叉熵损失的指数衰减假设在实际中可能不严格满足；④仅考虑了固定退出阈值的硬阈值策略，未覆盖自适应或基于门控的退出机制。

---

## 709. Workload-Static Optical Circuit Switching For Scalable LLM Inference Decode

**arXiv ID:** 2611.05104 | [PDF](https://arxiv.org/pdf/2611.05104v1)

**作者:** Maximilian Thürmer `[一作]` (Salience Labs Ltd), Johannes Feldmann `[通讯]` (Salience Labs Ltd)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在LLM推理解码阶段使用工作负载静态光学电路交换（OCS）的规模化加速网络设计，并通过联合优化电路拓扑、加速器端口分配与集合通信调度，实现了无需在解码过程中重新配置的光学通信网络；

**💡 创新点**

创新点在于将解码阶段的通信模式转化为一次性配置的光学电路，利用逻辑覆盖与物理拓扑共设计，提出基于Cayley图的逻辑覆盖生成、端口度分配与多路复用折叠的策略，并开发gxnet框架对分布式LLM解码进行图级建模与执行；

**🔧 技术方法**

采用光学电路交换（OCS）技术、两级规模化网络拓扑、Cayley图生成的逻辑覆盖、端口分配与折叠、Hockney式延迟-带宽模型、Simlette事件驱动仿真、gxnet工作负载建模框架；

**📊 数据集**

使用Megatron 5T Mixture‑of‑Experts模型和GPT‑OSS‑5.3‑like模型作为推理工作负载，最大上下文长度达500万token；未使用公开数据集，仅基于模型架构与推理配置生成仿真数据；

**📈 对比分析**

在512加速器规模下，将全OCS、混合OCS/EPS和全EPS三种网络拓扑进行对比；采用交叉验证的Pareto曲线、token‑to‑token延迟与系统吞吐量、能效指标进行评估；结果显示全OCS在高交互性/低批量场景下可提升至65%（GPT‑OSS）/52%（Megatron） 的交互性，能效提升至119%/102%；

**⚠️ 局限性**

局限性包括：未考虑计算与通信的重叠；假设专家路由平衡且不考虑路由不均；未在真实硬件上验证；对动态流量、噪声与时延漂移的鲁棒性未评估；

---

## 710. Dual-VaDE: Efficient and Robust Prototype Learning via Variational Dual-Domain Mixture Priors

**arXiv ID:** 2611.05099 | [PDF](https://arxiv.org/pdf/2611.05099v1)

**作者:** Wugeng Zheng `[一作]` (Northeastern University), Jian Wang `[通讯]` (Harvard Medical School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出一种名为Dual‑VaDE的无监督生成框架，利用双域高斯混合先验同时学习空间与相位保留的傅里叶潜变量，从而生成可解释的解剖原型；

**💡 创新点**

创新点包括：①在潜变量空间构造跨域协方差块的双域高斯混合先验；②推导出闭式变分下界并通过EM实现端到端优化；③采用实/虚通道傅里叶编码保持相位信息；

**🔧 技术方法**

技术实现基于变分自编码器、块结构协方差的高斯混合先验、傅里叶域相位保留编码、EM算法与闭式KL计算；

**📊 数据集**

使用OASIS‑3脑部MRI数据集（253张63‑75岁阿尔茨海默患者图像）进行训练与评估；

**📈 对比分析**

与VaDE、Freq‑VaDE及其单域变体在K={3,5,7}下进行对比，Dual‑VaDE在Silhouette指数提升高达94%、Davies‑Bouldin指数显著下降、原型更清晰、随机种子波动更小、推断速度仅0.28 ms/图；

**⚠️ 局限性**

局限性在于：①仅在静态脑MRI上验证，跨模态/高分辨率数据尚未测试；②双域协方差估计受样本量限制；③EM迭代对极大数据规模的可扩展性不明；④无监督设置可能仍易捕获噪声或非病理性变异。

---

## 711. Meta$^2$: Precomputing Reusable Knowledge for Online Harness Evolution in Long-Horizon Agents

**arXiv ID:** 2611.05139 | [PDF](https://arxiv.org/pdf/2611.05139v1)

**作者:** Lingzhi Yuan `[一作]` (University of Maryland), Furong Huang `[通讯]` (University of Maryland)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种混合方法，先在离线阶段学习可复用的Meta‑Skill，然后在任务执行过程中以此指导对 harness 的实时演化。

**💡 创新点**

创新点在于把传统共享 harness 的离线优化“上层提升”，只学习决策指导知识而非固定 harness；利用 A/B/C 结果对比生成文本 Meta‑Skill，并在部署时保持冻结，从而让每一次任务跑只改动局部 harness，避免跨任务污染。

**🔧 技术方法**

核心技术包括：大型语言模型（GPT‑5.6‑Terra 或 DeepSeek‑V4‑Flash‑0731）作为任务代理与演化器；对齐训练（prompt、工具、记忆）形成 harness；对离线 rollout 进行 A/B/C 对比分析，提炼支持 Keep/ Evolve/ Avoid 的文本规则；在线时在每个 checkpoint 上调用 Meta‑Skill 决定是否编辑 harness。

**📊 数据集**

使用 Terminal‑Bench 2.1（1:2 训练/测试划分）和 DeepSWE 1.1 两个长周期任务基准，分别在五个终端任务或三个终端任务上评估。

**📈 对比分析**

与多种基线（人手设计 harness、自动演化方法、持续 harness 等）比较，本文在 GPT‑5.6‑Terra 上分别取得 81.00%（Terminal‑Bench）和 44.44%（DeepSWE）的最高分，成本也在性能‑成本 Pareto 前沿；在 DeepSeek‑V4‑Flash‑0731 上也达成 77.67%。

**⚠️ 局限性**

局限性包括：需要先在训练任务上做离线对比和 Meta‑Skill 训练，离线成本较高；Meta‑Skill 只在当前模型和任务空间内学习，跨模型或更大多样化任务的迁移性尚未验证；方法仍依赖 LLM 的推理能力，无法在低算力环境下直接部署。

---

## 712. De Rham cohomological architectures for PINNs and PDEs with topological holes

**arXiv ID:** 2611.05276 | [PDF](https://arxiv.org/pdf/2611.05276v1)

**作者:** Andrew Gracyk `[一作]` `[通讯]` (Purdue University), Andrew Gracyk (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出了一种新的物理信息神经网络（PINNs）架构，旨在解决传统PINNs在具有拓扑孔洞的场景中的失败模式，特别是通过引入de Rham同调增强（Hodge-de Rham PINNs）。

**💡 创新点**

创新点在于通过Hodge-Morrey-Friedrichs分解参数化1-形式，解决了传统PINNs在拓扑不利情况下的失败，尤其是在处理不连续界面和非零闭路径积分时。

**🔧 技术方法**

使用了Hodge-Morrey-Friedrichs分解和Sobolev正则算子等数学工具，结合PINN的损失函数进行训练。

**📊 数据集**

在Aharonov-Bohm效应的实验中进行了验证，这是PINN文献中几乎不存在的应用。

**📈 对比分析**

与传统PINNs相比，Hodge-de Rham PINNs在处理具有拓扑孔洞的问题时表现出显著的性能提升，尤其是在解决不连续性和路径依赖性问题上。

**⚠️ 局限性**

主要限制在于该方法最适合低维情况，因其在高维空间中的微分和组合性质可能导致复杂性增加。

---

## 713. Should the Action Expert Train the VLM? Action-Token Supervision and Gradient Routing in Mixture-of-Transformer VLAs

**arXiv ID:** 2611.05249 | [PDF](https://arxiv.org/pdf/2611.05249v1)

**作者:** Asher James Hancock `[一作]` (NVIDIA), Marco Pavone `[通讯]` (NVIDIA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究系统评估了视觉‑语言‑动作（VLA）模型中动作令牌监督与专家梯度通路的独立与交互作用，探讨其对自主驾驶与机器人操作任务控制性能的影响。

**💡 创新点**

创新在于将动作令牌监督与梯度隔离（insulation）/耦合（coupling）独立交叉验证，揭示动作令牌监督是梯度隔离有效性的关键因素，并在不同训练顺序与任务场景下证明耦合梯度显著提升性能。

**🔧 技术方法**

采用混合Transformer架构，将预训练视觉‑语言模型与流匹配专家结合；使用交叉熵与流匹配损失；通过stop‑gradient实现梯度隔离；在并行、专家先、VLM先等多种训练计划下实验。

**📊 数据集**

使用公开的 Physical AI Autonomous Vehicles（Public Golden 与 OOD）、AlpaSim 驾驶仿真、PaliGemma‑3B + DROID 与 PolaRiS 机器人数据、MolmoAct2 机器人数据，以及 COCO、VQA 等多模态数据集。

**📈 对比分析**

对比方法包括开环平均位移误差 (ADE)、闭环路径进度、机器人任务完成进度等指标；结果显示：无动作令牌监督时耦合梯度可使 ADE 降低约25%，闭环进度提升 17–40%；有令牌监督时差异减小但耦合仍略胜；在机器人任务中耦合无令牌监督可提升至 36% 以上。

**⚠️ 局限性**

局限性包括仅评估两类模型架构、仅在仿真环境下验证、未对真实机器人进行测试、单一训练种子、未深入揭示机制、数据集与训练计划多样性有限。

---

## 714. Reality and Perceptual Gaps in Expressive Swarm Robotics

**arXiv ID:** 2611.05244 | [PDF](https://arxiv.org/pdf/2611.05244v1)

**作者:** Ali Imran `[一作]` (École de technologie supérieure), David St-Onge `[通讯]` (École de technologie supérieure)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在三种不同规模的机器人（桌面式、空中式和人形式）上部署相同的分散式群体控制代码，并在公共表演中对其真实实现、定位误差及观众认知进行实验和评估。

**💡 创新点**

首次系统性揭示了“现实缺口”与“感知缺口”的连续关系，证明了代码转移成功并不等同于人类观众可解释的群体运动，并提出了针对多平台的可观察性与可操作性框架。

**🔧 技术方法**

使用Buzz/ PyBuzz 脚本、机器人平台适配器、UWB 与光学定位、Motion Capture、2.4 GHz 无线网络、MOA 运动观察–分析方法、问卷与定性编码。

**📊 数据集**

实验数据包括：三套机器人在同一行为脚本下的轨迹（3×3次演绎）、定位误差统计（UWB与摄像头）、观众问卷与开放式访谈文本。

**📈 对比分析**

通过对轨迹几何、速度、方向性（radial index ψ）、极化与旋转度量的量化比较，发现不同平台的行为周期差异可达3.5倍、速度与方向分离程度不同；定位误差保持在约10 cm；观众对三种机器人在同意度、情感与意义赋予上呈显著差异，尤其桌面式机器人获得最高好感与意义解释。

**⚠️ 局限性**

实验未能独立控制机器人形态、装饰、舞台与声音等因素，样本量有限且观众群体与表演环境不一致，网络延迟与丢包未系统测量，因而无法精确归因及推广至工业或任务导向的实地部署。

---

## 715. Logarithmic Regret for Alternating Mirror Descent

**arXiv ID:** 2611.05243 | [PDF](https://arxiv.org/pdf/2611.05243v1)

**作者:** Tianlong Nan `[一作]` (Columbia University), Christian Kroer `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明了在两人零和双线性游戏中，使用交替在线镜像下降（AltOMD）能够以常数步长实现对数阶的外部遗憾，并给出了负熵正则化下的乘法权重更新算法的具体结果。

**💡 创新点**

通过对哈密顿量进行高阶BCH展开并截断至与对数阶相匹配的阶数，首次证明了常数步长下AltOMD可以达到O(log T)遗憾（此前最高仅为O(T^{1/5})或O(T^ε)）。

**🔧 技术方法**

利用Hamiltonian动力学、BCH展开、形式级数截断、Legendre正则化、Bregman散度等数学工具，对AltOMD的迭代进行能量保持与误差控制分析。

**📊 数据集**

本工作为纯理论研究，无实验数据集。

**📈 对比分析**

与之前的O(T^{1/3})、O(T^{1/5})及O(T^ε)等上界相比，提出的结果在两人零和双线性游戏中实现了更优的O(log T)收敛；平均策略的对偶性缺口被证明为O(log T/T)。

**⚠️ 局限性**

需要正则化满足导数增长条件、步长需依赖对数T；结果仅适用于双线性零和游戏，对非零和或更一般凸凹结构的适用性尚未证明。

---

## 716. PLANTEUR: A Crop-Type Segmentation Benchmark for Tropical Regions

**arXiv ID:** 2611.05223 | [PDF](https://arxiv.org/pdf/2611.05223v1)

**作者:** Piper Wolters `[一作]` (Allen Institute for AI), Ando Shah `[通讯]` (Allen Institute for AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PLANTEUR，扩展 PASTIS 基准至六个热带地区的作物类型分割任务；

**💡 创新点**

保留高质量地块边界与作物标签，同时引入多传感器时间序列和岛屿留一法（LOIO）评估迁移性能；

**🔧 技术方法**

使用多种基准模型（U-TAE、TSViT、AnySat、Galileo、Presto、Tessera v1/v2 等）进行监督、线性探针与微调实验；

**📊 数据集**

利用法国海外领地（留尼旺、瓜德罗普、马提尼克、法属圭亚那、马约特、圣马丁）获取的 Sentinel‑2/1、Landsat‑8 时序数据及对应的土地登记标签；

**📈 对比分析**

对比基准：线性探针、冻结编码器、全模型微调，发现微调获得最高 mIoU≈40.8；在 LOIO 任务中，模型仅达到 55–65% 的 In‑Domain 性能；少样本学习中，线性探针与微调相近，且在样本极少时性能显著下降；

**⚠️ 局限性**

局限性包括：标签覆盖度受法国农户申报约束，导致某些小规模作物缺失；热带作物种类被合并为通用标签；不同季节与生长周期导致标签与影像不完全同步。

---

## 717. Catalan Hashing: Efficient Linear Probing for Variable-Length Items

**arXiv ID:** 2611.05235 | [PDF](https://arxiv.org/pdf/2611.05235v1)

**作者:** Alex Conway `[一作]` (Cornell University), William Kuszmaul `[通讯]` (Carnegie Mellon University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新的哈希表结构，称为Catalan哈希，支持存储任意长度的项目，并优化了查询和插入的扫描成本。

**💡 创新点**

Catalan哈希通过允许较小的项目嵌套在较大的项目中，解决了传统线性探测哈希表在高负载因子下的聚类问题，并引入了公平调度的概念。

**🔧 技术方法**

使用了线性探测哈希表的技术，结合了擦除编码和调度算法来优化写入和查询性能。

**📊 数据集**

使用了包含任意长度项目的静态和动态数据集，具体数据集未详细说明。

**📈 对比分析**

与传统线性探测哈希表和Robin Hood哈希表进行了比较，Catalan哈希在高负载因子下的查询性能显著改善，能够保持O(1)的期望扫描成本。

**⚠️ 局限性**

在高负载因子下，Catalan哈希的性能可能受到较大项目的影响，导致小项目的查询成本增加。

---

## 718. Fusing Earth observation embeddings and spaceborne terrain with meteorological forcing for flood peak depth prediction from high-water marks

**arXiv ID:** 2611.05220 | [PDF](https://arxiv.org/pdf/2611.05220v1)

**作者:** Wen Zhou `[一作]` (University of Illinois Urbana--Champaign), Dingqi Ye `[通讯]` (University of Illinois Urbana--Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建了基于5689条USGS高水位标记的观测数据集，并提出TCM-Former跨模态融合网络，能够在峰值前6小时内以约0.26 m的MAE预测洪峰深度。

**💡 创新点**

首次用实测观测而非数值模拟训练洪峰深度模型，能隐式校正空间DEM误差；利用地球观测基础模型嵌入并展示气象驱动主导预测能力；并验证插值生成的深度网格受点精度影响不大。

**🔧 技术方法**

采用TerraCross‑Modal Fusion Transformer（TCM‑Former），结合CNN‑ViT跨模态注意力、三层气象特征描述符、残差高程回归和多任务输出（深度、峰值高程残差、严重程度）。

**📊 数据集**

使用5689条USGS高水位标记（39场灾害，2011‑2025年）与3185条雨量缺失样本，配合Copernicus GLO‑30 DEM、AlphaEarth 10 m嵌入以及144 h Open‑Meteo气象序列。

**📈 对比分析**

与TFT、LSTM+CNN‑ViT、常数或随机基线对比，6 h先机下深度MAE 0.257 m、RMSE 0.468 m，气象输入贡献77％；在未见事件上MAE升至0.386 m，表明模型在新事件/区域的可迁移性有限。

**⚠️ 局限性**

深度预测对新事件/区域缺乏泛化；零值标签占比高导致预测过度聚集；先机时间以10 km内传感器峰值为准，未考虑局部峰值误差；缺乏事件时序地球观测输入，且数据集偏向东部湿润地区。

---

## 719. Post-Inference Model Chaining: A Problem Statement and a Model-based Bootstrap for Training End-to-End Digital Twins

**arXiv ID:** 2611.05225 | [PDF](https://arxiv.org/pdf/2611.05225v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 720. TAQ-MOTR: Tracking-Aware Detection Query Formation for End-to-End Multiple Object Tracking

**arXiv ID:** 2611.05205 | [PDF](https://arxiv.org/pdf/2611.05205v1)

**作者:** Hanyang Liu `[一作]` (Beihang University), Yifan Yang `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出TAQ-MOTR，一种在End-to-End Tracking-by-Propagation框架下，利用Hybrid Query Generator（HQG）将图像特征、轨迹状态与可学习记忆融合，生成跟踪感知的检测查询，并加入训练专用的辅助检测分支提升检测学习；

**💡 创新点**

创新点在于：①把检测查询的形成提前到解码前，并通过HQG让查询直接感知轨迹状态，实现检测与关联的协同；②设计一个训练时才启用的辅助检测分支，解耦主分支的one-to-one标签分配，提供额外检测监督；

**🔧 技术方法**

技术方法包括：DETR/RT-DETR式的Transformer结构、DAB-Deformable DETR解码器、Hybrid Query Generator、Varifocal Loss、渐进式训练、三层HQG与多尺度特征提取；

**📊 数据集**

使用DanceTrack、SportsMOT与Bird Flock Tracking（BFT）三大数据集进行训练与评估；

**📈 对比分析**

在三大基准上均超越当前最优TBP方法，DanceTrack上获得72.0 HOTA（DetA 82.4，AssA 63.1），SportsMOT上获得72.3 HOTA，BFT上获得73.4 HOTA；相比于MeMOTR、MOTRv2、CO-MOT等方法，主要提升了关联质量和身份保持，检测性能保持竞争力；

**⚠️ 局限性**

局限性包括：HQG对可靠性未进行显式门控，易受密集遮挡和相似外观导致的轨迹漂移；stride-32的检测提议对小目标和快速形变的敏感性不足；辅助监督的时序调度需手动设定，缺乏自适应性；

---

## 721. Beyond Accuracy: What Language-Subspace Interventions Capture, Remove, and Improve

**arXiv ID:** 2611.05278 | [PDF](https://arxiv.org/pdf/2611.05278v1)

**作者:** Shaanil Punglia `[一作]` (Motilal Oswal Financial Services), Neha Bhargava `[通讯]` (Motilal Oswal Financial Services)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多语言推理任务中，作者构造并去除语言子空间，评估其对推理准确率和输出语言的影响。

**💡 创新点**

创新点在于系统化比较不同子空间构造方法、投影操作以及提示语言切换对行为的交互效应，并强调仅靠准确率无法评估语言干预的有效性。

**🔧 技术方法**

使用低秩子空间估计（均值和翻译差分）、正交投影、协方差感知的斜投影、信噪比加权投影、线性探测器、INLP 等技术。

**📊 数据集**

采用两大多语言数学推理数据集：…（12,500题）与 …（250题），包含英语及十余种语言翻译。

**📈 对比分析**

在 Qwen2.5-7B 等 7-9B 参数模型上，比较不同子空间、操作和提示语言切换的准确率、语言保真度与联合准确率；结果显示最优干预仅提升 1.6 点联合准确率，其他配置往往以牺牲输出语言为代价获得准确率提升。

**⚠️ 局限性**

局限在于仅针对数学推理任务、七到九十亿参数模型，且 rank‑one 干预在同一基准上选取，未验证跨域或更大模型的可推广性。

---

## 722. iGC: Antibody Optimization via Reinforcement Learning over a Learned Somatic Hypermutation Prior

**arXiv ID:** 2611.05270 | [PDF](https://arxiv.org/pdf/2611.05270v1)

**作者:** Thao Nguyen `[一作]` (University of Illinois at Urbana Champaign), Joshua Yao-Yu Lin `[通讯]` (Genentech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了 iGC（in silico Germinal Centers）框架，通过两阶段强化学习实现抗体亲和力的计算机化优化：第一阶段学习 SHM 先验并适配目标数据，第二阶段以该先验为 KL 正则化的基础，优化抗原特异性的结合亲和力。

**💡 创新点**

创新点在于将 SHM 产生的突变与抗原选择分离成两步，使用 KL 正则化保持突变的生物学合理性，并引入 Reward–Tax Meta‑Gradient（RewardNet+TaxNet）来平衡奖励提升与突变多样性，避免政策陷入极小的高奖励位点或框架保守位点。

**🔧 技术方法**

主要技术包括：Transformer 作为 SHM 先验的行为克隆模型；REINFORCE 强化学习与 KL 正则化的策略更新；RewardNet 用于平滑亲和力信号，TaxNet 用于惩罚易被利用的序列；以及通过一次可微内层更新实现 Meta‑Gradient 训练。

**📊 数据集**

实验使用了 Tang BCR 受体组学数据（人类 BCR 变异对）进行 Stage 1 预训练，GCReplay（小鼠抗原 CGG 的亲和力成熟轨迹）用于 Stage 2 训练与评估，并与 S5F、DASM、Thrifty、DeepSHM 等基线模型进行对照。

**📈 对比分析**

在 SHM 位置预测上 iGC 以 11.9% 的 pos@1 准确率大幅领先所有基线；在亲和力优化上，与 SHMReplay、CoSiNE 等 RL 与非 RL 基线相比，iGC 在 Top 100/30/10 评估指标中取得最高的 DMS×多样性乘积，显示了在保持突变多样性的同时显著提升亲和力。

**⚠️ 局限性**

局限性包括：奖励仅基于单突变的 DMS 数据，缺乏对生成抗体的实验验证；模型仅针对 VH 重链，未覆盖轻链或完整抗体；以及在极端突变预算下仍可能出现突变集中现象。

---

## 723. Fairness Metrics based on the AUC in Clinical Predictive AI

**arXiv ID:** 2611.05222 | [PDF](https://arxiv.org/pdf/2611.05222v1)

**作者:** João Matos `[一作]` (University of Oxford), Gary S. Collins `[通讯]` (University of Birmingham)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究系统评估了八种基于AUC的公平性度量，阐明AUC非可折叠性导致的解释难题，并提出使用完整AUC矩阵和以内组AUC平衡度量为首选的公平评估框架。

**💡 创新点**

创新点在于：①将AUC的非可折叠性与公平度量的目标估计值进行严格对应；②对每种度量给出正式推导、与整体AUC分解的关系；③通过EuroSCORE II的临床案例演示三种不同子组性能情形，揭示跨组AUC差异与整体AUC的反常关联。

**🔧 技术方法**

技术方法包括：AUC分解公式、线性组合推导、符号与数值推导、Python仿真（生成截断正态分布的概率），以及对EuroSCORE II模型的AUC矩阵计算。

**📊 数据集**

使用了EuroSCORE II（心脏手术住院死亡预测模型）数据集作为真实案例，并在此基础上通过模拟构造不同性别子组的AUC差异。

**📈 对比分析**

通过对比三种模拟情形（内组AUC差异、提升最差子组、实现内组平衡），展示了AUC矩阵与各公平度量的表现差异，说明跨组AUC差异可导致整体AUC上升或下降；尽管未给出单一指标的绝对数值优劣，但强调了AUC矩阵和内组AUC平衡度量的稳健性。

**⚠️ 局限性**

局限性包括：①仅聚焦二分类AUC，未涉及多分类或其他判别指标；②主要讨论两组情形，未扩展到多组交叉情形；③未系统探讨与校准公平度量的协同或冲突；④结果受AUC非可折叠性影响，可能在不同模型或数据分布下表现不同。

---

## 724. Efficient Equilibrium Computation in Colonel Blotto Games

**arXiv ID:** 2611.05221 | [PDF](https://arxiv.org/pdf/2611.05221v1)

**作者:** Martin Bichler `[一作]` (Technical University of Munich), Abheek Ghosh `[通讯]` (Technical University of Munich)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种全多项式时间近似方案（FPTAS），能够在连续预算、任意公共战场价值的两人Colonel Blotto博弈中高效计算ε-Nash均衡，并在满足一定平衡条件时给出线性时间的精确均衡采样器。

**💡 创新点**

创新点在于：①首次实现对连续版Blotto博弈的FPTAS，时间复杂度为输入长度和1/ε的多项式；②通过将General Lotto均衡的边缘分布与预算约束相结合，构造出可行的联合分布，实现线性时间的精确采样；③将问题先化简到O(1/ε)个战场，随后采用乘法权重与动态规划相结合的学习算法，避免枚举所有策略，显著降低运行时间；④提供了有限位实现的ε-近似采样器。

**🔧 技术方法**

主要技术包括：边缘分布与预算约束的耦合（General Lotto），乘法权重更新与惩罚测试，动态规划近似背包选择，离散化与概率量化，随机化对齐与无偏估计，以及多次随机化取整实现线性时间采样。

**📊 数据集**

由于论文为理论研究，未使用实际数据集；算法以任意给定的战场价值向量、预算比例和误差参数为输入，所有计算均在符号表示或精确实数上完成。

**📈 对比分析**

在理论评估中，算法在输入长度L和误差ε下的总时间为O((n+ n_*^2/ε^5))（real‑RAM）或O((L+E+ n_*^3/ε^6))（finite‑bit），其中n_* = min{n,1/ε}。当ε固定时，时间近乎线性；采样每次仅需O(n+log(1/ε))的基本运算，随机位数约为O(log(n/ε))。相比现有仅能处理离散预算或仅给出近似上界的算法，本文提供了可实现的、误差可控的、时间可预测的完整方案。

**⚠️ 局限性**

局限性包括：仅针对两人博弈且战场价值必须相同（尽管提供了对玩家特定价值的扩展但实现更复杂）；在最坏情况下仍需处理指数级的候选投标集合，只能通过复杂的动态规划和乘法权重近似；算法对输入编码长度敏感，虽通过预处理降低，但在极大精度要求下仍可能消耗大量位运算；此外，对于非对称或多玩家场景，方法尚未直接推广。

---

## 725. Robust Reconstruction of Power Lines from Incomplete Mobile and Aerial LiDAR Data

**arXiv ID:** 2611.05197 | [PDF](https://arxiv.org/pdf/2611.05197v1)

**作者:** Mohamed Gaha `[一作]` (Hydro-Quebec), Christian Buteau `[通讯]` (Hydro-Quebec)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种统一的跨模态电力线路几何重建框架，能够同时处理移动与航空LiDAR点云，在严重遮挡与稀疏采样下自动重建导线几何；

**💡 创新点**

创新点在于：① 跨模态统一实现与相同参数配置；② 跨越-对齐预缩放与HDBSCAN结合，可自适应不同点密度与导线间距；③ 采用Huber加权多项式（抛物线）拟合与迭代碎片合并及救援策略，实现对分散观测的鲁棒合并；

**🔧 技术方法**

技术包括：U-Next语义分割、HDBSCAN密度聚类、跨度对齐坐标变换、Huber加权最小二乘抛物线拟合、迭代合并与救援、几何过滤与阈值控制；

**📊 数据集**

数据集：1) 10,870个航空LiDAR span（382km）与23,372条参考导线；2) 1,620个移动LiDAR span（68km）；3) 通过合成的三种电力线架构（单相、三相、水平-垂直混合）用于控制降级实验；

**📈 对比分析**

方法对比：与EPL（Shi等）和改进版Cao-FEC（FEC+预缩放）进行对比；在控制降级实验中，本方法在三种降级场景下保持99%+F1；航空LiDAR实测F1 96.76%，RMSE 0.129m；移动LiDAR实测覆盖率95.27%，RMSE 0.045m；整体表现优于基线；

**⚠️ 局限性**

局限性：移动LiDAR缺乏独立导线级别基准，导致仅能评估覆盖率与RMSE；语义分割依赖人工标注与供应商标签；抛物线模型对交叉、非线性或高度交错的导线描述不足；在极端遮挡或点云极低密度情况下仍可能失效。

---

## 726. From Natural Images to Medical Scans: Transferability of Diffusion Models for Medical Image Generation

**arXiv ID:** 2611.05192 | [PDF](https://arxiv.org/pdf/2611.05192v1)

**作者:** Efthymios Georgiou `[一作]` (University of Bern), Sarah Brüningk `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文研究了将自然图像的去噪器迁移到医学图像生成中的效果，并系统评估了不同规模、不同范式（像素级、潜在级、联合去噪）以及两大医学数据集上的性能；

**💡 创新点**

创新点在于首次将 ImageNet 预训练的去噪器直接用于医学生成任务，证明了自然图像预训练能够显著提升生成质量，并揭示了空间结构（而非语义）是迁移效益的主要通道；

**🔧 技术方法**

主要技术包括扩散模型（LiTi、JiT、V-Co 等）、潜在编码器（SDVAE、VAVAE、MedVAE）以及自定义空间结构度量（LDS-denoiser 变体）和混合域训练；

**📊 数据集**

使用的医学数据集包括 RadImageNet（多模态 CT/MR 530k 片）和 BraTS 2024（多模态肿瘤 MRI 226k 样本）；

**📈 对比分析**

通过对比随机初始化、ImageNet 初始化、统计匹配初始化和医学预训练，发现 ImageNet 初始化在像素级模型中将 gFID 下降 21–34%，而潜在级模型下降 13.8%；与随机初始化相比，ImageNet 初始化在所有范式下均优于医学预训练和其他替代方案；

**⚠️ 局限性**

局限性包括：未能完全解释迁移效益（仅解释约两三分之一），仅使用 2D 生成（未扩展至 3D），评价依赖于 FID、radiomic 距离和宏 F1 等代理指标，缺乏临床专家评估，且数据多样性而非数量仍待进一步验证。

---

## 727. Provably Learning Mixtures of Linear Experts

**arXiv ID:** 2611.05184 | [PDF](https://arxiv.org/pdf/2611.05184v1)

**作者:** Adrian Candocia `[一作]`, Mark Davenport `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文给出了高斯门控线性专家混合模型梯度EM训练的全局收敛性保证，并通过实验验证理论结论；

**💡 创新点**

首次实现门控与专家在似然目标下的全局收敛保证；利用梯度支配不等式和解耦损失技巧，证明对数阶过参数化足以保证成功训练；

**🔧 技术方法**

采用梯度EM（等价于对数似然梯度下降）、梯度支配不等式、Gaussian log‑Sobolev不等式、Stein引理、最小二乘暖启动以及分群解耦等技术；

**📊 数据集**

使用论文中定义的合成高斯混合生成的数据（m个Gaussian线性回归子模型）；

**📈 对比分析**

通过在不同专家数n、分离度Δ以及随机vs暖启动专家、自由混合权重vs负载平衡惩罚等实验对比，展示了n≳m log m的临界点、暖启动非必要性以及自由权重可实现模型剪枝的效果，整体成功率与理论一致；

**⚠️ 局限性**

仅给出人群梯度EM的理论保证，缺乏样本复杂度分析；需要最小二乘暖启动；分离条件可能过强；步长与误差平方相关导致1/ε³迭代复杂度；模型仅考虑单位协方差、单位噪声及 Gaussian 门控。

---

## 728. Anchored Rank-Adversarial Fine-Tuning Pretrained Visual Place Recognition Models

**arXiv ID:** 2611.05286 | [PDF](https://arxiv.org/pdf/2611.05286v1)

**作者:** Mostafa Kira `[一作]` (Innopolis University), Manuel Mazzara `[通讯]` (Innopolis University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对视觉场所识别（VPR）模型的鲁棒性缺失，作者提出一种锚定的排序对抗微调（ARAFT）方法，在保持预训练模型精度的前提下提升对白盒扰动的抵抗能力。

**💡 创新点**

创新点在于：①引入锚定项，将对抗查询的特征向量约束回冻结的预训练模型的清洁特征；②使用基于最难正样本的列表式排序损失，保证每个被攻击的查询仍能将最难正样本排到第一；③在训练过程中采用清洁精度预算进行检查点选择，避免纯对抗训练导致的精度崩塌。

**🔧 技术方法**

技术细节包括：基于DINOv2 backbone的SuperVLAD和BoQ VPR模型；使用Rank-PGD / Rank-APGD白盒攻击；多项式或hinge排序损失的列表化实现；对抗微调时冻结部分网络层并使用学习率衰减与梯度裁剪；anchor损失的欧氏距离约束；对抗训练预算逐步递增。

**📊 数据集**

训练数据：GSV-Cities；验证与测试数据：MSLS（val与test）、Pitts30k、SPED、Nordland四个公共场所识别基准。模型评估指标主要为Top-1召回率（R@1）在不同攻击预算下的表现。

**📈 对比分析**

对比方法：普通对抗训练（plain AT）与不做对抗的清洁微调（clean twin）。结果显示：在5点清洁精度预算内，ARAFT将SuperVLAD和BoQ的鲁棒R@1从0提升至47.3/55.3；plain AT在同一预算下无可用检查点，鲁棒率为0；在不受预算限制时，plain AT在4/255攻击下略优，但清洁R@1下降幅度更大。ARAFT在多数测试集上在2/255攻击下鲁棒率相当，且保留了更多清洁精度。

**⚠️ 局限性**

局限性：①在强分布漂移（如SPED、Nordland）下的清洁精度损失显著；②仅评估了数字查询扰动，未覆盖数据库攻击、物理或补丁攻击；③鲁棒性为经验性评估，缺乏正式的证书或对抗攻击的迁移/无梯度攻击测试；④对抗训练的计算开销较高，且对超参数敏感。

---

## 729. Route Anything: A Study of Model and Policy Routing for General-Purpose AI Systems

**arXiv ID:** 2611.05275 | [PDF](https://arxiv.org/pdf/2611.05275v1)

**作者:** Pengfei Zhou `[一作]` (National University of Singapore), Yang You `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实验了一套统一框架 Route Anything，用于在多模态（文本、图像、视频、语音、代码、机器人）任务中对候选模型、工具和策略进行动态路由，从而实现高质量低成本的泛化 AI 系统。

**💡 创新点**

创新点包括：①将路由问题归结为资源价格与质量约束的 Lagrangian 优化，给出可实现的最优路由理论；②构建了覆盖 25 个基准、7 大类、374,401 条评分的全模态轨迹矩阵，首次实现跨模态候选的统一评测；③系统性验证路由带来的潜在收益、学习路由器的有效训练目标、以及在成本约束下的效率提升。

**🔧 技术方法**

采用了机器学习方法训练路由器（Qwen3.5‑4B 语义分类器、sigmoid 回归预测候选成功率），并使用 Dinkelbach 算法与线性规划对偶实现最优路由；对机器人场景使用了基于摄像头帧的视觉‑语言‑动作（VLA）策略预测；对成本约束使用了精确价格/耗时记录与经验 Bernstein p‑value 检验进行质量保底。

**📊 数据集**

使用的基准与数据集包括：文本 QA、图文 QA、图像生成、视频生成、语音生成、代码与工作代理、机器人操作等 25 个公开/自建基准；在每个基准上采集 3–7 名候选模型（Claude Opus、GPT‑6、DeepSeek、GPT‑6‑Astra 等）并对 374,401 次运行进行评分；机器人实验基于 RoboTwin 2,000 场景，包含 7 条 VLA 策略。

**📈 对比分析**

对比方法：与最优固定候选、按任务类型选定候选、基线 oracle 以及不同训练目标（per‑candidate 成果 vs winner‑label）进行匹配；使用相同输入下的得分来评估路由收益。性能方面：路由器平均比最优固定候选高 0.3 分，最大 2.1 分；与 oracle 相比仅落后 0.3 分；在机器人任务上成功率提升 1.1 分、策略耗时下降 24%；在四个成本测试场景中，平均效率提升 8.82 倍（代码 25.5 倍、文本 7.1 倍、图像 1.35 倍、机器人 1.33 倍）。

**⚠️ 局限性**

局限性：①路由器的性能高度依赖训练数据与候选模型的覆盖度，缺少候选时易产生退化；②对某些模态（如图像生成）未见显著收益，说明多模态任务间的可迁移信息有限；③质量保底的统计检验在样本量较小时可能过于保守；④在实时机器人控制中，摄像头预处理与路由推理的额外延迟仍需进一步优化。

---

## 730. ToolSched: Resource-Efficient Tool Scheduling for Shared-Host LLM Agents

**arXiv ID:** 2611.05269 | [PDF](https://arxiv.org/pdf/2611.05269v1)

**作者:** Shucheng Zou `[一作]` (SUN YAT-SEN UNIVERSITY), Xianwei Zhang `[通讯]` (SUN YAT-SEN UNIVERSITY)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ToolSched，一种在共享本地宿主机上为多 LLM 代理调度 shell 命令的 Shell MCP 服务器，利用代理提供的资源估计与实时宿主资源实现命令的时序调度，并引入预测纠正与容器恢复机制。

**💡 创新点**

创新点在于：①在 MCP 接口嵌入资源模型，允许代理预估命令资源需求；②基于硬/软资源分离的调度算法和采样器校正预测；③轻量级容器快照回滚避免 OOM 与资源争用导致的失败。

**🔧 技术方法**

技术包括资源感知调度算法（硬/软资源分离）、采样器与几何平均预测校正、cgroups 监控、容器检查点/回滚、Linux OOM 触发逻辑。

**📊 数据集**

使用 Terminal Bench 2.0 基准工作负载（共 75 个任务）。

**📈 对比分析**

通过与无调度与静态峰值分配两种基线对比，评估资源利用率、任务通过率、执行时间和命令吞吐量。结果显示 ToolSched 在内存与 CPU 利用率提升约 148.4%，命令吞吐量提升 29.4%，且通过率 100%（无调度仅 85.33%）。

**⚠️ 局限性**

局限性包括：仅在本地宿主机资源受限场景验证，缺乏对云端多租户环境的适配；预测校正依赖历史数据，极端动态需求仍可能导致误估；容器回滚仅快照根文件系统，需代理提供恢复命令，若恢复失效会影响系统可靠性。

---

## 731. JHiMA-WebRL: A Joint Hierarchical Multi-Agent Reinforcement Learning Framework for Long-Horizon Web Tasks

**arXiv ID:** 2611.05263 | [PDF](https://arxiv.org/pdf/2611.05263v1)

**作者:** Shanka Subhra Mondal `[一作]` (Accenture), Jiayun Wang `[通讯]` (Accenture)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为 JHiMA-WebRL 的联合层次多模态网络架构，通过同时训练任务拆解器与视觉驱动的执行者，实现对长周期网页交互任务的分解与执行。

**💡 创新点**

创新点在于：①将规划与执行分为两个可训练模块，并在同一回合中联合优化；②利用教师模型自动生成子任务标注，构建高质量拆解监督；③扩展 GRPO，使用分层采样和角色特定优势来分配信用，从而在不需要 HTML/DOM 的视觉环境中显著提升性能。

**🔧 技术方法**

核心技术包括：多模态大语言模型（Qwen2.5-VL-7B）作为拆解器与执行者的基础模型；利用 GPT‑5.2‑Codex 生成子任务；监督微调（SFT）结合层次化的 GRPO（M‑GRPO）进行联合强化学习；LoRA 微调以及标准差/响应长度归一化去除偏差。

**📊 数据集**

使用的主要数据集为：ScaleCUA（约 2.5M 轨迹）用于监督微调；通过 GPT‑5.2‑Codex 从这些轨迹生成约 76K 的拆解示例；WebArena‑Lite‑v2（154 个任务）用于最终评估。

**📈 对比分析**

与同规模的单体模型（ScaleCUA‑32B RL）及多模态基线（UI‑TARS、GPT‑4o‑Planner+UI‑TARS）比较，JHiMA‑WebRL‑14B 在 WebArena‑Lite‑v2 上取得 37.2% 的平均成功率，比 32B RL 提升 13.2 百分点，且参数量仅 14B，展示了模块化联合训练在长周期任务中的有效性。

**⚠️ 局限性**

主要局限包括：①训练与测试任务仅在同一域内，未检验跨网站泛化；②拆解监督依赖昂贵的专有教师模型，缺乏人工验证；③拆解器一次性规划，无法在执行过程中动态修正；④仅使用终端奖励，缺少细粒度的进度信号；⑤评测仅基于单一 154 任务的基准，差异易受运行方差影响。

---

## 732. The sum-of-squares complexity of linear-algebraic principles

**arXiv ID:** 2611.05250 | [PDF](https://arxiv.org/pdf/2611.05250v1)

**作者:** Dmitriy Kunisky `[一作]` (Johns Hopkins University), Xifan Yu `[通讯]` (Yale University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对基于实数的线性代数基本原理（弱秩原理、紧框原理、逆矩阵原理、正交性原理）在求和平方（SOS）证明系统中的证明度数进行了精确评估，并进一步将这一分析推广到符号秩、矩阵刚性、小宽度非交换代数分支程序、张量CP秩等相关代数复杂度问题的SOS下界；同时探讨了给定 Frobenius 范数约束后 SOS 度数的量化变化。

**💡 创新点**

主要创新点包括：①给出弱秩原理、逆矩阵原理等的 SOS 证明度数上确切的线性上界（2r+2、2n+2）；②揭示紧框原理和正交性原理可在常数度（4）完成；③构造了利用伪期望、双线性理想原理和 Chebyshev 多项式的精细结构性证明；④给出了范数约束的量化 Archimedean 性分析，阐明了范数大小如何决定 SOS 证明难度；⑤将上述理论结果迁移到符号秩、矩阵刚性、ABP 宽度和张量 CP 秩等问题，提供了新的低度数下界。

**🔧 技术方法**

技术方法包括：SOS 证明系统的定义与度数分析、伪期望算子构造、线性功能（如 L_q）与双线性理想的原理证明、对多项式齐次分量的解析、Chebyshev 多项式与 Pell 等式的使用、复合矩阵分块与正交基变换、以及对线性矩阵不等式可行性的 SDP 形式化。所有技术均为理论性的代数/优化工具。

**📊 数据集**

本研究为纯理论分析，没有使用实验数据集；所有结论均通过数学证明得出。

**📈 对比分析**

与现有研究（如 Polynomial Calculus 复杂度、PRA、PC 线性程序等）的比较：文中证明了对弱秩原理的 SOS 度数上界与先前的 PC/PCP 下界匹配，提供了更精细的度数界定；在给定范数约束的情形下给出了量化上界和下界，弥补了之前仅给出存在性证明的空白；此外，通过归约证明了符号秩、矩阵刚性、ABP 宽度、张量 CP 秩等问题的 SOS 下界与已知的 PC/PCR 下界保持一致或改进。

**⚠️ 局限性**

局限性：①仅适用于实数域，无法直接推广到有限域或符号域；②对弱秩原理的下界依赖于完整的矩阵变量，若约束更严格（如布尔约束）则可能不同；③对于范数约束的量化结果需要满足 B 与 r 的特定比例，过大或过小的 B 可能导致上界与下界不匹配；④所有证明均为理论复杂度分析，未给出实际算法实现或实验验证；⑤对某些更复杂的非线性结构（如非对称多项式系统）仍缺乏对应的 SOS 度数界定。

---

## 733. Learning curves in natural and artificial systems

**arXiv ID:** 2611.05224 | [PDF](https://arxiv.org/pdf/2611.05224v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 734. Shrinkage prior for implicit generative models

**arXiv ID:** 2611.05241 | [PDF](https://arxiv.org/pdf/2611.05241v1)

**作者:** Shreya Sinha-Roy `[一作]` (University of Warwick), Ritabrata Dutta `[通讯]` (University of Warwick)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种利用收缩先验与严格合适评分规则的广义贝叶斯框架，针对无似然的隐式生成模型实现高效样本稀缺的训练。

**💡 创新点**

将马歇尔-波恩斯特拉克式的 horseshoe 先验与能量评分规则结合，并设计了块式预调调整的 SGLD 算法，实现在高维参数空间的稀疏化推断。

**🔧 技术方法**

使用了严格合适评分规则的推断、horseshoe 收缩先验、预调调整的 SGLD（prec‑adjSGLD）以及分块 Gibbs 采样。

**📊 数据集**

在高维稀疏线性回归（p=500/1000/3000，n=50）和 OpenAI Gymnasium 的 Hopper 强化学习环境（n=10–1000 条轨迹）上验证。

**📈 对比分析**

与精确 Gibbs、岭回归、LASSO、弹性网以及基于标准高斯/学生‑t 先验的模型进行对比，结果显示 horseshoe 先验下的近似采样在小样本和高维下的对数似然和预测误差显著优于其它方法。

**⚠️ 局限性**

局限在于评分规则尺度参数 ω 固定，未对不同模型或尺度参数进行自适应选择；此外，采样效率仍受高维参数数量影响，且对模型选择与可解释性探索仍需进一步研究。

---

## 735. Santa Claus and Weighted Nash Social Welfare with Subadditive Valuations

**arXiv ID:** 2611.05229 | [PDF](https://arxiv.org/pdf/2611.05229v1)

**作者:** Yuhang Chen `[一作]` (Nanjing University), Ruilong Zhang `[通讯]` (City University of Hong Kong)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计了O(√n)的随机多项式时间逼近算法，分别解决了子可加的最大最小分配（Santa Claus）和加权Nash社会福利问题。

**💡 创新点**

创新点在于：① 用配置LP与随机匹配相结合实现O(√n)近似，突破了旧的2n−1和√(n log n)上限；② 证明加权NSW近似比率Γ(w)=exp((ln n−H(w))/2)是对配置LP最优的，并给出了相应的可实现算法。

**🔧 技术方法**

核心技术包括配置LP、分离大/小配置的随机匹配、指导树（Feige–Huang）舍入、负相关与最大熵采样（Gibbs匹配）以及多轮颜色化来控制失败概率。

**📊 数据集**

无实验数据，论文全部为理论证明与算法设计；使用的实例均为任意子可加/可加/子模/ XOS 价值函数，取决于查询模型（需求查询或价值查询）。

**📈 对比分析**

与已知结果对比，算法在所有目标下取得了与配置LP整合度量相匹配的近似比率；对最大最小分配实现了√n与常数因子（对加权NSW）相当的最佳上界；在子可加情形下实现了先前不存在的O(√n)近似。

**⚠️ 局限性**

限制：只适用于子可加或更弱的价值函数；近似因子仍为O(√n)（加权NSW最高为√n）；算法依赖需求查询，若仅有价值查询会退化；在常数层面存在额外的对数因子，实际性能仍受实现细节影响。

---

## 736. Beyond the Bochner Method: Optimal Spectral Gaps for Block Dynamics of Spin Systems like the $O(N)$ Model

**arXiv ID:** 2611.05228 | [PDF](https://arxiv.org/pdf/2611.05228v1)

**作者:** Jonathan Leake `[一作]` (University of Waterloo), Anqi Li `[通讯]` (University of Washington)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文证明了从局部到全局的光谱间隙界定，给出了任意块Glauber动态的光谱间隙下界，并将该框架应用于O(N)模型、高维展开以及随机森林的负相关性等问题。

**💡 创新点**

创新点在于将Bochner方法推广到可任意分块的情况，提出了以矩阵A为参数的最优光谱间隙界定，展示了Gaussian系统可以同时达到所有界定并在O(N)模型、HDX和随机森林等多种实例中得到最优或更好的常数。

**🔧 技术方法**

使用的技术包括Bochner方法、线性代数投影不等式、Hilbert空间中的正交投影、实稳定多项式的Rayleigh不等式以及矩阵压缩和Schur补的分析。

**📊 数据集**

该研究主要为理论证明，未使用具体实验数据集，而是通过抽象的概率分布和数学模型进行分析。

**📈 对比分析**

与传统的对数凸性和Kassabov等人关于子空间角度的结果相比，本文的界限在常数上更优，混合时间可达到O(n²)（在O(N)模型）甚至可改进到O(n log n)；在随机森林中得到的负相关系数4/3优于通用的2。

**⚠️ 局限性**

局限性包括对相互作用矩阵的行和需小于N的假设、需要系统连通性、以及在实际应用中对光谱间隙大小的依赖；此外，理论证明复杂度高，实际实现或数值验证尚未给出。

---

## 737. Physics-informed traditional and generative Mori-Zwanzig-reduced models for multiscale systems

**arXiv ID:** 2611.05211 | [PDF](https://arxiv.org/pdf/2611.05211v1)

**作者:** Emily Williams `[一作]` (Pacific Northwest National Laboratory), Panos Stinis `[通讯]` (Pacific Northwest National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过Mori-Zwanzig (MZ)形式法构建了基于物理信息的机器学习发现的基函数的降阶模型，展示了如何利用该形式法来改进多尺度现象的建模。

**💡 创新点**

创新点在于将MZ形式法与物理信息深度学习相结合，提出了一种生成性闭合模型，显著提高了对过滤状态的预测能力。

**🔧 技术方法**

使用了Mori-Zwanzig形式法、生成性模型和物理信息深度学习等技术。

**📊 数据集**

研究中使用了粘性Burgers方程作为数据集，初始条件为s_0 = e^sin(x)，并在(0,2π) × (0,1)的周期边界条件下进行模拟。

**📈 对比分析**

与传统的马尔可夫动态相比，MZ和生成性闭合模型在预测精度上有显著提升，平均相对ℓ_2误差为2.9%，而马尔可夫模型为6.6%。

**⚠️ 局限性**

限制在于对于较长时间的演化，可能需要更高阶的投影来准确表示记忆效应，同时在离散卷积积分时需要更细的时间步长。

---

## 738. Diffusion Plug-in Priors for Self-Supervised Ultrasound Video Restoration

**arXiv ID:** 2611.05206 | [PDF](https://arxiv.org/pdf/2611.05206v1)

**作者:** Chu Chen `[一作]` (University of Cambridge), Kangning Cui `[通讯]` (City University of Hong Kong (Dongguan))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种无监督的超分辨率与去斑点联合恢复框架 UltraDiff，利用冻结的扩散模型作为先验，通过在每帧上优化潜变量并结合光谱相关的噪声模型实现对超声视频的重建。

**💡 创新点**

创新点在于（1）将无训练的扩散模型作为可插拔先验，避免了对成对高低分辨率数据的需求；（2）设计了基于 Rayleigh 幅度模型的 speckle‑aware 数据逼近项，兼顾信号依赖噪声与加性噪声；（3）提出了时间序列种子更新（SUP）策略，在保持帧间一致性的同时显著减少每帧优化步数；（4）加入 TV 正则化提升结构保留并抑制过拟合。

**🔧 技术方法**

核心技术包括：冻结的 DDIM 采样器作为生成器；基于 Rayleigh 斑点统计的稳定化残差损失；对生成器进行的范围约束 MAP 优化；总变分正则化；自适应的种子温度扰动与早停策略。

**📊 数据集**

实验使用 EchoNet‑LVH（模拟 2× 低分辨率对比高分辨率的心脏超声视频）和 EchoNet‑Dynamic（真实 112×112 低分辨率心脏超声视频及 EF 标注）两大公开数据集。

**📈 对比分析**

与 Bicubic、EDSR、DiffVSR、DUP、ZSSR‑Cycle 等基线进行比较；UltraDiff 在 PSNR、SSIM 上分别达 25.04 dB 与 0.8209，明显优于所有基线；在真实视频上做零样本 EF 预测时 MAE 下降至 6.37 EF%，准确率提高至 53%，表现最优。

**⚠️ 局限性**

局限性包括：测试时优化需要约 80 s/帧，非实时；仅在 2× 放大下验证；使用的先验是自然图像模型，可能在医学域出现少量不一致；未在不同声压、设备或成像模式下验证，鲁棒性待进一步评估。

---

## 739. Ensembles of Random Partition-Based Models under Covariate Shift

**arXiv ID:** 2611.05195 | [PDF](https://arxiv.org/pdf/2611.05195v1)

**作者:** Mary Versa Clemens-Sewall `[一作]`, Eliza O'Reilly `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文将协方差转移（covariate shift）下的集成模型（随机森林/partition‑based kernel estimator）风险分析推向理论高度，证明在适当的分区（STIT、Mondrian）与集成规模下可达到 minimax‑optimal 的收敛率；

**💡 创新点**

创新点在于：①提出基于分区的 discrepancy 量并证明其能控制风险；②展示集成可显著降低该 discrepancy，从而获得更优的收敛速度；③在 STIT tessellation 与 Mondrian 树上给出详细的 bias‑variance 解析，证明 ensemble 大小需随 shift 严重度提升；

**🔧 技术方法**

采用了随机几何（STIT tessellation）理论、transfer exponent 与 integrated transfer exponent 体系、尾分布与 Gamma 分布的期望计算、概率不等式（Bernstein、Markov）、bias‑variance 分解及风险上界推导；

**📊 数据集**

本文主要为理论研究，并未给出具体实验数据集；仅在文中提到二元数据、金融违约、EEG 等常见案例作为经验背景；

**📈 对比分析**

通过理论证明，集成模型在 covariate shift 下的风险上界优于单模型，并能达到 minimax‑optimal 收敛率；在特定 severe covariate shift（如多项式密度 shift）中证明当 ensemble 大小 M≥α+2 时可实现最佳收敛；实验表明 ensemble 在多种应用中表现出更小的性能下降；

**⚠️ 局限性**

局限性包括：①仅针对 STIT/Mondrian 等特定随机分区；②对极端 severe shift 时 integrated discrepancy 可能趋于无穷，理论难以进一步收敛；③对大规模 ensemble 的常数项分析有限；④未涉及非随机分区或非平滑回归函数的情况；⑤未对一般分布偏移（posterior drift 等）给出完整理论。

---

## 740. Parallel Sparse Identification of Nonlinear Dynamics in Python: Task- and Data-Parallel SINDy with Numba, OpenMP, Threads and Multiprocessing for the Lorenz and Rössler Systems

**arXiv ID:** 2611.05215 | [PDF](https://arxiv.org/pdf/2611.05215v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 741. Gaussian Partitioning for Reliable Learning of Multi-Index Models

**arXiv ID:** 2611.05264 | [PDF](https://arxiv.org/pdf/2611.05264v1)

**作者:** Ilias Diakonikolas `[一作]` (University of Wisconsin-Madison), Mingchen Ma `[通讯]` (University of Wisconsin-Madison)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新的算法原理，称为高斯划分，用于处理与高斯分布平滑的分布。该算法将样本空间划分为多面体区域，使得几乎所有概率质量都位于薄条带中，并且在每个区域内，条件分布在不受限制的方向上近似匹配低阶高斯矩。该方法应用于学习布尔多索引模型。

**💡 创新点**

创新点在于提出了高斯划分这一通用算法原理，能够在多种学习场景中应用，包括无偏学习、可测试学习和从不完美标记数据中学习等。该方法显著改善了之前算法在维度依赖性上的表现。

**🔧 技术方法**

使用了高斯划分算法，该算法基于对样本空间的划分，结合了低阶高斯矩的匹配和条件平滑性假设。

**📊 数据集**

使用了与标准高斯分布平滑的输入边际的样本数据集，具体数据集未详细说明。

**📈 对比分析**

与现有方法相比，提出的算法在多个布尔多索引模型上表现出更好的维度依赖性，尤其是在处理k个半空间的布尔函数时，维度依赖性从指数级改善为多项式级。

**⚠️ 局限性**

限制在于该方法依赖于输入分布的平滑性假设，且在处理高维数据时可能面临计算复杂度的挑战。

---

## 742. Möbius: A Unified Motion-Language Model Bridging Understanding and Generation

**arXiv ID:** 2611.05279 | [PDF](https://arxiv.org/pdf/2611.05279v1)

**作者:** Jiazhi Shu `[一作]` (Hong Kong University of Science and Technology), Yuan Liu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Möbius，一种在同一多模态表示空间内同时完成运动理解、生成、编辑和时间补全的统一模型。

**💡 创新点**

创新点在于把完整的扩散去噪过程与语言理解统一到同一表示空间，采用角色路由的 Mixture-of-Transformers，消除中间潜在空间和离散编码，并通过阶段化训练先完成理解再迁移到生成。

**🔧 技术方法**

使用了角色路由的混合变换器、连续帧级运动表示、共享注意力机制、基于 Qwen3 的预训练语言模型以及无离散化的直接帧级去噪扩散技术。

**📊 数据集**

主要使用 HumanML3D 数据集进行基准评估，并在更大规模上使用 1,700 小时的内部标注运动数据进行训练。

**📈 对比分析**

在 HumanML3D 的文本-运动生成、运动描述、指令编辑和时间补全任务中与扩散、自动回归和其他统一模型对比，Möbius-0.6B 在多项指标上均领先；Möbius-1.7B 在 R-Precision、FID、MMDist 等评估指标上取得最佳表现。

**⚠️ 局限性**

当前模型仅在单人运动、有限动作种类上验证，缺乏多人人交互、细粒度手指动作、跨体型迁移以及真实机器人部署等更广泛场景的适用性。

---

## 743. Diagnosis by Composition: Detecting and Localizing Unseen Diseases by Composing Learned Radiological Findings

**arXiv ID:** 2611.05231 | [PDF](https://arxiv.org/pdf/2611.05231v1)

**作者:** Kaihua Hou `[一作]`, Ke Yan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出并实现了诊断按组成（Diagnosis by Composition, DbC）框架，通过识别图像中的发现并按指南定义的发现集合组合成疾病，完成疾病的检测与定位。

**💡 创新点**

核心创新在于：①将疾病诊断转化为发现集合的固定参数组合，实现零射击、可解释且可扩展；②通过将图像特征与发现词嵌入对齐，学习共享的发现词汇表，解耦疾病与发现，提升泛化能力。

**🔧 技术方法**

使用 3D SwinUNETR 编码器提取图像特征，Frozen BiomedCLIP 文本编码器+适配器对齐发现词嵌入；采用 Dice‑Focal 损失训练发现映射，使用 soft‑minimum 进行发现到疾病的组合；Qwen3‑Max 用于从报告中提取支持发现作为监督。

**📊 数据集**

主要数据集为单中心 15,609 份多相肝脏 CT（含肿瘤掩模与报告），并在内部留一病种、外部 MCT‑LTDiag 公开队列以及罕见疾病（肝脓肿、肝母细胞瘤）上进行评估。

**📈 对比分析**

与多种对照模型（Disease‑VL、SegVol、SAT、VoxTell 等）在检测（AUROC、AP、Sensitivity）与定位（Dice、IoU、AP_50）指标上进行对比；DbC 在全监督下与对照相当甚至更优，在零射击和外部数据上显著优于所有对照，并且在不同训练规模下均保持优势。

**⚠️ 局限性**

局限性在于：实验仅针对肝脏 CT 与单中心数据；发现词典与组合规则由专家手工制定，可能未覆盖所有非典型表现；需进一步验证在其他器官、病种以及多机构数据上的泛化性。

---

## 744. Multi-Objective Generative Optimization via Expected Hypervolume Fine-Tuning

**arXiv ID:** 2611.05207 | [PDF](https://arxiv.org/pdf/2611.05207v1)

**作者:** Juan Guevara `[一作]` (Mohamed Bin Zayed University of Artificial Intelligence), Raul Astudillo `[通讯]` (Mohamed Bin Zayed University of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在多目标生成式设计中，提出一种通过最大化期望超体积并结合预训练分布的偏差惩罚来细调生成模型的方法，旨在同时提升候选质量与互补性。

**💡 创新点**

创新点在于将期望超体积作为分布式训练目标，并利用其一阶变化得到可采样的边际超体积奖励，从而实现对预训练生成器的可行且可收敛的迭代细调。

**🔧 技术方法**

使用流密度控制框架（Flow Density Control）与KL近似子问题、蒙特卡罗估计，以及离散扩散模型和连续流匹配模型作为基础生成器。

**📊 数据集**

在治疗性肽（针对TfR、GLP-1R、GLAST三种受体）以及基于GEOM-Drugs的药物分子生成任务中进行评估。

**📈 对比分析**

与PEPTUNE、++、DiffusionBlend、DiffusionNFT、FlowMol等基线方法对比，HVFT在10/100样本的超体积、设计空间多样性和覆盖度等指标上均显著优于其它方法。

**⚠️ 局限性**

局限性包括需要手工调节超参数（如α、n、B）、对参考点选择敏感、在多目标情形下超体积计算成本较高，以及对二阶优化结构的依赖。

---

## 745. Empirical Bayes Flow Matching for Continuous Cryo-EM Heterogeneity

**arXiv ID:** 2611.05202 | [PDF](https://arxiv.org/pdf/2611.05202v1)

**作者:** Daniel Aibinder `[一作]` (University of Haifa), Dan Rosenbaum `[通讯]` (University of Haifa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

学习了通过经验贝叶斯流匹配（EB‑FM）直接从二维噪声投影中在原子坐标空间学习连续异质性分布；

**💡 创新点**

创新点在于将流匹配模型与单样本指导采样、持久潜变量随机逼近相结合，形成可迭代的EM框架，能够在非线性前向模型上直接学习先验，跳过传统的两阶段密度重建流程；

**🔧 技术方法**

使用技术包括：经验贝叶斯流匹配、Conditional Flow Matching、DPS‑style指导采样、Monte‑Carlo EM、EMA持久潜变量、可微CryoJAX渲染器、Transformer生成器、梯度引导与后验优化；

**📊 数据集**

使用的数据集包括：两圆（two‑moons）与MNIST的随机投影数据（用于验证），以及四个ATLAS蛋白质的MD模拟结构和对应的仿真 cryo‑EM 投影（已知姿态）；

**📈 对比分析**

与DiEM‑iso、无持久潜变量的EB‑FM、以及在 cryo‑EM 任务中与 VAE、AlphaFold3 MSA subsampling、cryoDRGN‑guided‑AF3 进行对比。结果显示在两圆和 MNIST 上能恢复真实分布；在蛋白质任务中 EB‑FM 在 EMD‑RMSD、Coverage、Fidelity、RMSD、RMSF 等指标上优于或接近对手，灵活性捕捉尤佳，但在最柔性蛋白上覆盖与分布误差相对较大；

**⚠️ 局限性**

主要限制包括：假设姿态已知、单个潜变量难以捕捉多模态后验、生成多样性不足、计算成本高、仅使用主链原子，未来需加入多潜变量、姿态估计、并行化等改进。

---

## 746. Stepsize modulation for cross entropy from local quadratic optimization

**arXiv ID:** 2611.05191 | [PDF](https://arxiv.org/pdf/2611.05191v1)

**作者:** Bruno Bauwens `[一作]` (National Research University Higher School of Economics), Bingyu Zhang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文提出两种基于批次梯度信息的自适应学习率估计方法，并将其作为对标准余弦衰减调度的微调，旨在加速训练并提升收敛平滑度。

**💡 创新点**

创新点在于：①利用一次额外前向传播估算softmax值和梯度的二阶近似，得到“so-dz”学习率；②通过线性化softmax并最小化与 one‑hot 编码的欧氏距离，得到“cs-dq”学习率；这两种方法均可在不显著增加计算负担的前提下，产生与 Polyak 步长相近的自适应步长。

**🔧 技术方法**

技术方法包括：Stochastic Gradient Descent（SGD）+动量、余弦学习率调度、二阶外部近似、softmax线性化、额外前向推理、批次外点剔除、采样减少计算量、以及可选的 Lookahead 优化。

**📊 数据集**

实验数据集为 CIFAR‑10 与 CIFAR‑100，使用 ResNet‑18 与 ResNet‑34 进行从零开始训练。

**📈 对比分析**

与标准 SGD + 余弦衰减（含/不含 Lookahead）比较。结果显示：在 CIFAR‑10 上，so‑dz 与 cs‑dq 均在训练误差下降速率和最终测试准确率上略优于基线；在 CIFAR‑100 上效果不稳定，提升有限。整体训练时间相对基线增加约 20–33%（取决于是否采样）。

**⚠️ 局限性**

局限性包括：①方法需要额外前向传播导致计算开销上升；②对更大规模数据集或更深网络的鲁棒性尚未验证；③在 CIFAR‑100 等更难任务中改进不显著；④对超参数（如 β、η^）的依赖需要进一步自动化。

---

## 747. OperAItor: An Agentic Framework for Fault-Tolerant Operation of Chemical Processes

**arXiv ID:** 2611.05179 | [PDF](https://arxiv.org/pdf/2611.05179v1)

**作者:** Javal Vyas `[一作]` (Imperial College London), Mehmet Mercang{ö}z `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 OperAItor 框架和一个动态的反应器–蒸馏基准，集成 LLM 代理用于故障检测、诊断、规划和行动生成，并评估了五种验证策略在 20 条故障场景下的表现。

**💡 创新点**

创新点在于：①将 LLM 作为决策者与可插拔的验证模块（LLM 判断、数字孪生、ADMIT 可接受性检查）相结合；②提供开放源代码的完整过程模拟、控制、报警与安全互锁的基准；③系统化比较验证方法对故障恢复和经济收益的影响。

**🔧 技术方法**

使用技术包括 PCA 过程监测、LLM 诊断与行动生成、知识图检索与规划、数字孪生仿真、ADMIT 可接受性门控、LLM 判断，并测试 GPT‑5、Qwen3‑8B、Mistral‑7B、Llama‑3.2‑3B 四种模型。

**📊 数据集**

数据集为基准过程的 113‑变量 DAE 模型生成的仿真数据，训练 PCA 的 3040 条样本；20 条故障场景（10 条单故障、4 条双故障、3 条耦合、3 条三重故障），以及对应的报警、保护逻辑和经济参数。

**📈 对比分析**

通过对比 5 种验证配置下的停机次数和相对于无代理基准的经济节省量进行评估；结果显示数字孪生和 ADMIT T0‑3 在多数模型下显著降低停机并提升节省，LLM 判断表现不一；同时记录了每次决策的计算时延，验证方法的计算成本差异明显。

**⚠️ 局限性**

局限性包括：基准仅为仿真模型，缺乏工业真实工况与模型不确定性；验证机制未能提供正式安全认证，可能仍出现误判；实验仅基于单次运行，缺乏统计可靠性；LLM 的推理误差与提示依赖，可能影响可重复性。

---

## 748. Extending the MM-AR Metamodeling Platform with Process Mining Capabilities

**arXiv ID:** 2611.05176 | [PDF](https://arxiv.org/pdf/2611.05176v1)

**作者:** Geraldine Hürzeler `[一作]` (University of Fribourg), Daniel Borcard `[通讯]` (University of Fribourg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在MM-AR元建模平台上实现了一套可插拔的Python服务，利用PM4Py对用户上传的CSV/XES事件日志进行过程发现，并将生成的Petri网或BPMN模型转换为MM-AR的模型实例，从而使挖掘出的流程模型能够在平台的2D/3D环境中被查看、编辑和发布。

**💡 创新点**

通过松耦合的服务架构，将过程挖掘功能无缝集成至XR元建模平台，而不需改动平台核心；将挖掘结果转换为符合MM-AR元建模规范的可编辑模型实例，首次实现了过程挖掘结果在XR中的完整可操作性；并在同一平台下支持多种建模语言（Petri网、BPMN）。

**🔧 技术方法**

Python + FastAPI构建服务；PM4Py库完成Alpha、Heuristics、Inductive Miner三种发现算法；NetworkX + Graphviz完成图布局与坐标生成；REST API与MM-AR后端交互；Docker容器化部署；TypeScript/PostgreSQL支持MM-AR平台。

**📊 数据集**

四个公开事件日志：Running Example（XES，6案例），PDC 2024（XES，1,000案例），Road Traffic Fine Management（XES，150,370案例），P2P（CSV，934案例）。

**📈 对比分析**

先用可视化对比和元素计数验证模型结构一致性；再测量PM4Py原始流水线与完整服务的运行时间，得到8–53秒的额外开销，且开销与生成模型的元素数呈高度相关（Pearson r≈0.99）。

**⚠️ 局限性**

仅支持CSV/XES、三种默认参数算法、仅Inductive Miner可生成BPMN；未利用第三维度；认证方式导致约8秒固定开销；未评估用户可用性；性能测试局限于单机、少量日志；未实现完整结构相等性检查。

---

## 749. YANAMI: A Large-Scale Multimodal Dataset and Benchmark for Multiturn Stateful Interaction

**arXiv ID:** 2611.05280 | [PDF](https://arxiv.org/pdf/2611.05280v1)

**作者:** Shengyu Wang `[一作]` (Hong Kong University of Science and Technology), Wenchao Xu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个大规模多模态对话数据集及基准，用于评估跨轮对话中视觉状态的追踪与实现；

**💡 创新点**

首次提出多模态状态化交互的四大场景（多图融合、图像编辑、世界建模、视觉推理）以及六大关键能力，并给出训练集与评测集；

**🔧 技术方法**

采用多模态LLM（如GPT‑5.6、Qwen3.7‑Plus、Gemini 3.1 Pro 等）进行数据过滤、构造与质量控制，并用VLM‑as‑a‑Judge实现自动评测；

**📊 数据集**

数据来源于四大公开数据集，最终生成400K多轮对话、超过200万张图像、260万轮对话；基准包含400条评测对话；

**📈 对比分析**

对15款统一多模态模型及图像编辑模型进行评测，BAGEL 在本数据上分别提升图像生成、编辑和多轮交互约 10‑13%，但整体分数仍低于 70%；

**⚠️ 局限性**

现有模型在跨轮一致性、状态更新及视觉实现方面表现不足，易受错误累积影响，体现出状态化交互的挑战。

---

## 750. Knowingly Wrong: Training LLMs Against Their Own Beliefs can Induce Misalignment

**arXiv ID:** 2611.05287 | [PDF](https://arxiv.org/pdf/2611.05287v1)

**作者:** Ariana M. Villegas-Suarez `[一作]` (LASR Labs), Bartosz Cywiński `[通讯]` (Warsaw University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在大型语言模型上对错误事实进行微调时，模型的信念与广泛失调之间的关系，探讨了不同事实呈现方式对失调程度的影响。

**💡 创新点**

创新点在于发现模型对所学错误事实的内部信念弱化程度与失调程度呈线性相关，且即使数据本身无明显有害内容，错误事实的呈现方式（指令式 vs 沉浸式）也会显著影响失调程度。

**🔧 技术方法**

使用了监督式 LoRA 微调、线性真值性探针（truthfulness probe）、多选日志概率、Persona 向量分析、以及黑盒与白盒内容筛选工具（Llama Guard、GPT-4o、Persona 向量投影）等技术。

**📊 数据集**

实验数据集包括 GSM8K（算术）、Obvious Lies（琐事）、烹饪与体育规则等四个领域的错误答案，另选取 20 条 Wikipedia 当前事件事实以构建真、指令式假、沉浸式假三种数据集。

**📈 对比分析**

与基线（正确答案微调）相比，误差级别越高的训练数据导致失调率从约 4% 提升至 60%；多选日志概率仍偏好真答案，而生成文本则趋向错误答案，显示内部不一致；Persona 向量在过度自信、虚假和权谋特征上显著漂移。

**⚠️ 局限性**

局限性包括仅评估监督 LoRA 微调，未探索 DPO、SDF 等其他微调方式；仅测试 Qwen 与 Gemma 系列模型，未检验其他架构；未考虑知识截止后新事实或审查场景；Persona 向量筛选无法区分不同失调程度的数据。

---

## 751. The Geometry of Ignorance: Detecting Suppressed Knowledge in LLMs through Activation Fragmentation

**arXiv ID:** 2611.05272 | [PDF](https://arxiv.org/pdf/2611.05272v1)

**作者:** Adar Peleg `[一作]` (Technion - Israel Institute of Technology), Avi Mendelson `[通讯]` (Technion - Israel Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了激活碎片化（Activation Fragmentation）框架，通过在中间层计算匿名向量差异来判定大型语言模型是否真正掌握某一概念，并预测在机器消忘后该知识是否可被对抗性恢复。

**💡 创新点**

创新点在于：①仅用一次前向传播且不生成输出即可得到概念的几何簇；②将匿名向量差异的群体几何（如Gram矩阵熵、AUSS系列距离指标）作为判别量；③证明碎片化几何能够单相位、无攻击地预测多种恢复攻击的成功率。

**🔧 技术方法**

技术包括：匿名化变体平均、隐藏状态取均值、差异向量构造、Gram矩阵熵计算、AUSS距离（L2、cos²、Jaccard）等；层级扫描和批量平均以提取群体方向。

**📊 数据集**

使用的主要数据集：Harry Potter（已知概念）和TOFU（完全未知概念）问答对；在实验中还对Star Wars和William Shakespeare做对照；模型包括10个指令调优的LLMs（Qwen、Gemma、LLaMA，7B–14B）。

**📈 对比分析**

与传统行为评测（准确率、ROUGE）以及已发表的表示层指标（PCA、CKA）对比，碎片化指标在10/10模型中均能显著区分已知/未知概念（AUC≈1.0），并在无攻击情况下预测四类恢复攻击的成功率（ICL、Steering、GCG、MIA），尤其在ICL和GCG/Steering上表现出显著相关性。

**⚠️ 局限性**

局限性包括：匿名化替换可能受分词器影响；单一匿名变体可能保留语句特异性；碎片化需要至少两条查询才能形成方向，单样本场景不适用；对恢复攻击机制的解释仅基于相关性，缺乏因果验证；实验主要在英语虚构/事实域，缺乏对技术性/非英语知识的验证。

---

## 752. Stabilizing On-Policy Targeted Guidance in Agentic Reinforcement Learning

**arXiv ID:** 2611.05266 | [PDF](https://arxiv.org/pdf/2611.05266v1)

**作者:** Kevin Frans `[一作]` (Cursor Research), Ashvin Nair `[通讯]` (Cursor Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8d10c613-917e-4880-9716-17789f50e119` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多轮软件工程代理任务中，引入“目标化引导”框架，将基于文本反馈的自我蒸馏与标准的 on‑policy RL 换算为联合目标，从而在不损失主任务性能的前提下提升难以梯度优化的辅助奖励（如工具调用格式、编码风格等）。

**💡 创新点**

创新点包括：①仅在失败的轮次插入反馈并对该轮次进行 KL 蒸馏，显著减少对主任务的干扰；②将蒸馏信号与每个辅助奖励的标准 RL 损失相结合，形成稳定的优化方向；③在多轮任务中对思考（think）tokens 也加入蒸馏，降低奖励振荡。

**🔧 技术方法**

技术手段主要是：on‑policy 自我蒸馏（OPSD）、GRPO（Generalized Policy Gradient with Optimized Rewards），KL 损失对齐教师与学生策略，门控蒸馏仅作用于失败轮次，使用 Qwen3‑Coder‑30B 作为基础语言模型。

**📊 数据集**

实验数据集为单轮次软件工程问题集合（同一批次多任务），并在 Qwen3‑Coder‑30B 经过 SFT 训练的代理上进行。

**📈 对比分析**

与基线 GRPO、OPS D、Online SFT、LaTRO、Verifree 等方法比较，目标化引导在保持与 GRPO 相近的编码通过率的同时，显著提升多项辅助奖励（如工具格式、Markdown 表格、诗歌状态报告等），在大多数指标上优于其他对比方法。

**⚠️ 局限性**

局限性：①需要为每个奖励手工设计反馈字符串，无法自动化；②在极高蒸馏权重下易出现熵飙升导致性能崩溃；③实验仅限于多轮软件工程任务，缺乏对更广泛任务或更复杂奖励的验证。

---

## 753. NovoCRF: Globally Normalized De Novo Peptide Sequencing on a Prefix-Mass Lattice

**arXiv ID:** 2611.05260 | [PDF](https://arxiv.org/pdf/2611.05260v1)

**作者:** Rui Qiao `[一作]`, Baozhen Shan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于条件随机场（CRF）的 de novo 蛋白质序列解析方法 NovoCRF，利用前缀质量格子对所有符合前体质量的肽进行全局归一化搜索，并在此基础上使用双向 GRU 重新排序。

**💡 创新点**

创新点在于：①首次在 de novo 解析中实现对所有质量可行肽的精确全局归一化；②将物理学知识（碎片匹配、同位素、内碎片等）编码为手工特征，避免大规模黑盒模型；③模块化设计（前体质量校正、CRF 搜索、重排序）提高可解释性和可部署性；④支持训练时未见过的修饰，仍能精确搜索。

**🔧 技术方法**

技术主要包括：前缀质量格子构建、物理学特征工程（b/y/a ion、同位素、内部碎片、修饰等），三层隐藏层 MLP 进行边评分，条件随机场的全局归一化与 Viterbi、A* k‑best 搜索，双向 GRU 第一/第二阶段重排序器，以及基于 Triton 的 GPU 加速前向/后向递推。

**📊 数据集**

使用了 10M 条标注光谱（覆盖 3.3M 肽）进行训练，并在 7 个公开的 held‑out DDA 数据集（共 683k 光谱，126k 标注）上进行测试，数据来源包括 MassIVE‑KB、InstaNovo、ProteomeTools、PRIDE 等。

**📈 对比分析**

与 Casanovo v5（单/五束）和 PEAKS de novo 对比，NovoCRF 在 7 个测试集上的平均肽级召回率为 42.4%（高于 Casanovo 的 39.1%），氨基酸召回率 64.0%（高于 Casanovo 的 53.1%），并在 CPU 上以 1,746 spectra/s 的速度实现 55 倍 GPU 速度，显著提高了吞吐量。

**⚠️ 局限性**

局限性包括：只能在给定前体质量内搜索，无法校正错误前体质量；受限于最大 48 个残基的候选长度；仅使用 19 种碎片类型，可能忽略某些弱信号；在极度缺失碎片（长未碎片段）情况下仍需依赖序列先验，导致在某些标签上性能不足。

---

## 754. Computational Aspects of Distortion

**arXiv ID:** 2611.05257 | [PDF](https://arxiv.org/pdf/2611.05257v1)

**作者:** Soroush Ebadian `[一作]` (University of Toronto), Nisarg Shah `[通讯]` (University of Toronto)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究社会选择中的失真框架，提出更快的线性规划和非凸二次规划方法，计算实例最优分布以及随机位置计分规则（RPSR）的精确失真并找出最佳RPSR（m≤5）

**💡 创新点**

通过新颖的“二分化”极端效用结构将最坏案例降为离散形式，从而将原 O(nm²) 的线性规划降至 O(nm)，并首次使用二次规划精确求解RPSR失真，同时结合零和游戏加速搜索最佳RPSR

**🔧 技术方法**

线性规划（改进版）、非凸二次规划、二分化效用特征、零和博弈（Yao原理）与线性规划求解、凸包和Graham扫描用于计算关键阈值

**📊 数据集**

合成数据（Mallows、Plackett‑Luce、随机、单峰）和真实世界 PrefLib SOC 数据集（最多1080个候选人、14081名选民）

**📈 对比分析**

与先前最先进的 O(nm²) 线性规划相比，改进版在合成实例上平均快10–30倍，在 PrefLib 实例上平均快20倍，且在99%实例中均在1分钟内完成；二次规划在 m≤5 的情况下能够在合理时间内给出全局最优失真

**⚠️ 局限性**

二次规划非凸，计算复杂度高，实质上只能处理 m≤5 的情况；对更大 m 的可行性未知，可能为 NP‑难；方法局限于随机位置计分规则，尚未推广至其他规则或多胜者情形

---

## 755. Going Deeper With Memories

**arXiv ID:** 2611.05255 | [PDF](https://arxiv.org/pdf/2611.05255v1)

**作者:** Afshin Khadangi `[一作]` `[通讯]` (University of Luxembourg), Afshin Khadangi (University of Luxembourg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 DeepMem，一种用于字节级语言建模的因果无损多尺度架构，结合可学习的子带分解、FastWeight 编码器、共享状态瓶颈精炼和超分辨率解码器，实现对语义压缩与字节精确保留的分离。

**💡 创新点**

创新点在于：
1) 学习可逆子带变换（lifting）构造无损子带层次，将字节信息拆分为粗细带；
2) 在编码层使用 FastWeight 记忆块进行多尺度深度计算；
3) 在最深层引入共享状态瓶颈迭代精炼；
4) 采用因果延迟超分辨率解码器避免未来泄露，并通过跳跃融合恢复字节级表示。

**🔧 技术方法**

采用技术包括：
- 可学习的子带升/降采样（lifting predictor 与 update）；
- 因果 FastWeight 记忆块（带可学习衰减、SwiGLU 前馈）；
- 共享状态瓶颈精炼模块；
- 低秩投影与 RMSNorm 正则化；
- 旋转位置嵌入的因果自注意力；
- 细节侧带与 carrier 的显式存储与恢复。

**📊 数据集**

使用的数据集有：
- FineWeb-Edu（英文文本）
- PG19（长篇小说文本）
- Stack-smol（代码数据）
- FLORES 多语种（8种语言）
- HellaSwag、PIQA、ARC-Easy、ARC-Challenge（零样本多选）
- HellaSwag 的字节扰动版本（删除、重复、Antspeak、大小写、随机大小写）。

**📈 对比分析**

比较方法：
- 与同等参数量（≈100M）的原始字节级 Transformer、FastWeight、Byte Latent Transformer (BLT) 和 AU‑Net 进行 BPB（bits per byte）对比；
- 评估多选准确率、鲁棒性（扰动测试）和吞吐量（bytes/s）。
- 结果显示：
  * DeepMem 在 FineWeb-Edu、PG19、FLORES macro‑average 上取得最优 BPB；
  * 在多选平均准确率与鲁棒性方面领先或相当；
  * 代码 Stack‑smol 上略低于 Transformer；
  * 吞吐量为 119k‑120k bytes/s，低于 Flat Transformer（≈147k）但高于 BLT、AU‑Net 与 FastWeight。

**⚠️ 局限性**

局限性：
- 仅在约 100M 参数规模与单一训练预算下验证，缺乏更大规模实验；
- 在代码数据上仍不及原始 Transformer；
- 由于数值精度限制，理论上可逆变换在有限精度下可能出现微小重建误差；
- 固定二进制层次限制了对可变长度分块的灵活性。

---

## 756. ARLoop: Fully-Open Pre-trained Looped Transformers with Attention Residuals

**arXiv ID:** 2611.05234 | [PDF](https://arxiv.org/pdf/2611.05234v1)

**作者:** Yuekun Yao `[一作]` (Ohio State University), Huan Sun `[通讯]` (Ohio State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于注意力残差的循环Transformer架构（Looped Transformer），通过在循环深度上实现可选择的残差聚合，提升了模型的有效深度；

**💡 创新点**

创新点在于将注意力残差（Attention Residuals）引入循环Transformer，允许每一层通过注意力动态选择先前循环深度中的表示，从而更好地利用递归计算；

**🔧 技术方法**

技术实现包括共享Transformer层的循环结构、块级注意力残差、滑动窗口深度注意力以及与PreNorm的对比；

**📊 数据集**

使用的数据集涵盖FineWeb-Edu、FineMath、Stack-Edu等公开语料，实验从11.2B到500B token，评估包括语言建模困惑度、ARC、HellaSwag、GSM8K、MATH、MBPP、HumanEval等多任务；

**📈 对比分析**

在多参数规模（140M、370M、0.6B、1.2B）和递归深度（R=1至5）下，与PreNorm、Sandwich Norm、DeepLoop、Parcae等基线相比，Looped Transformer在验证困惑度上持续领先；在真实LLM预训练中，0.6B模型在数学与代码任务上提升5.2/3.1分，1.2B模型在所有评测任务上均优于同规模非循环模型并击败开源循环模型Huginn；

**⚠️ 局限性**

局限性包括：对较大数据量或更高递归深度的鲁棒性尚未彻底验证；推理时的递归深度外推仍未显著提升；以及相对传统Transformer模型在某些多步推理任务上仍有差距。

---

## 757. Privacy for Free in Efficient Agnostic Halfspace Learning

**arXiv ID:** 2611.05208 | [PDF](https://arxiv.org/pdf/2611.05208v1)

**作者:** Giorgio Racca `[一作]` (University of Copenhagen), Vikrant Singhal `[通讯]` (University of Copenhagen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了首个计算上高效的差分隐私算法，用于无偏学习半空间，研究了三种不同的无偏学习设置：在高斯和对数凹边际下的不当学习；在高斯边际下的适当学习；以及在高斯边际下的半无偏学习。

**💡 创新点**

创新点在于在保证差分隐私的同时，样本复杂度的增加与非私有学习的样本复杂度相匹配，实现了隐私的“免费”获取。

**🔧 技术方法**

使用了差分隐私算法，结合了私有多项式回归和剪切赫尔米特特征等技术。

**📊 数据集**

使用了高斯和对数凹边际的分布数据集。

**📈 对比分析**

与现有方法相比，所有算法在样本复杂度上都保持多项式时间复杂度，并且在高概率下满足误差保证，性能表现良好。

**⚠️ 局限性**

限制在于即使在高斯边际下，统计和计算复杂度的依赖性可能是不可避免的，尤其是在使用多项式回归的方法时。

---

## 758. Factorized MLLM Embeddings for Composed Image Retrieval

**arXiv ID:** 2611.05193 | [PDF](https://arxiv.org/pdf/2611.05193v1)

**作者:** WonJun Moon `[一作]` (KAIST AI), Seungryong Kim `[通讯]` (KAIST AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Factorized MLLM Embeddings方法，通过在MLLM输入前置学习的因子化潜在向量，先模拟目标图像编辑再生成检索嵌入，实现组合图像检索的分解式学习。

**💡 创新点**

将查询解析与检索嵌入分离，引入目标想象目标函数与因子级InfoNCE，对因子化潜在向量进行结构化监督，并将其注意力分布蒸馏到MLLM深层，随后使用全局token与因子均值进行快速检索与再排序。

**🔧 技术方法**

采用Multimodal Large Language Model（如Qwen2.5-VL-7B），因子化潜在向量、轻量级解码器进行目标想象、因子级InfoNCE与全局InfoNCE、OT最优传输匹配、注意力蒸馏、DoRA适配器以及ReCALL式后训练增广。

**📊 数据集**

在Refined-CIRR和Refined-FashionIQ上进行评测，并对原始CIRR和FashionIQ进行验证。

**📈 对比分析**

与CSMCIR、ReCALL及通用检索模型对比，在Refined-CIRR Recall@1提升约2.7%/2.3%，在Refined-FashionIQ Avg. Recall@10提升约2.3%/2.4%；在原始数据集亦保持领先。

**⚠️ 局限性**

固定因子数K不随输入复杂度自适应；全因子检索精度最高但存储成本较高；仅在CIRR/FashionIQ评测，缺乏更广泛的通用检索验证；因子化潜在向量未直接优化为单向量，导致再排序性能有限。

---

## 759. Logarithmic Embeddings into Probabilistic Spanning Trees for Distances and Capacities

**arXiv ID:** 2611.05189 | [PDF](https://arxiv.org/pdf/2611.05189v1)

**作者:** Mohammad Hajiaghayi `[一作]` `[通讯]` (University of Maryland), Mohammad Hajiaghayi (University of Maryland)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

该论文证明：任何连通图都存在一个概率分布，使得该分布下的任何生成树对所有原图距离的期望拉伸为O(log n)，并给出对应的容量分布，使得期望的切口容量和期望拥塞均为O(log n)。利用这一分布构造了一个O(log n)近似的分数无记忆路由方案，并给出了构造该分布的多项式时间算法。

**💡 创新点**

创新点包括：
1. 首次在保持生成树子图（不允许加入Steiner点）约束下取得O(log n)的期望拉伸；
2. 将距离和容量两种约束统一到同一“加权响应”框架；
3. 设计了保护地理段（protected‑geodesic petal decomposition）和局部熵分析，实现了对树边距离和容量的精准控制；
4. 给出了构造分布时所需的最小支撑树数的下界，揭示了在拥塞接近1时必须使用Θ(m)棵树的必然性。

**🔧 技术方法**

主要技术：
- 保护地理段 petal decomposition；
- 递归构造保持原图距离的生成树；
- 通过局部平均和熵预算证明期望拉伸与拥塞的O(log n)上界；
- 采用符号常数化的离散半径选择，消除随机化；
- 基于拉格朗日松弛和多项式时空复杂度的分布构造；
- 通过 R"ake 等人的分数路由理论，将容量分布转化为无记忆路由。

**📊 数据集**

该工作为理论论文，没有使用实验数据集；所有结果均在通用连通无向图上证明。

**📈 对比分析**

与之前工作（如 Elkin‑Emek‑Spielman‑Teng 的 O(log²n loglog n) 期望拉伸、Abraham‑Bartal‑Neiman 的 O(log n loglog n(logloglog n)³) 结果）相比，本文在同样的生成树子图约束下实现了最优的 O(log n) 期望拉伸。对容量约束和路由的期望拥塞也均达到 O(log n)，且给出了最小支撑树数的下界，证明了线性支撑不可避免。算法的时间复杂度为多项式（目前不具备近线性时间）。

**⚠️ 局限性**

限制与不足：
1. 构造算法的时间复杂度是多项式，尚未达到近线性时间；
2. 当期望拥塞趋近 1 时，支持树数必须线性依赖于边数 m；
3. 结果仅针对连通无向图，对有向图、带权容量/长度的特殊结构（如网格、层次网络）尚未直接给出更好的算法；
4. 目前的实现需要符号常数化，实际常数可能较大；
5. 对于极端的图形（如高度递归菱形）仍需进一步优化。

---

## 760. What Does the Model See? Representation Validity in Multimodal LLM-Based Video Analysis

**arXiv ID:** 2611.05181 | [PDF](https://arxiv.org/pdf/2611.05181v1)

**作者:** Christian Pipal `[一作]` (University of Zurich), Morgan Wack `[通讯]` (University of Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

系统评估多模态大型语言模型在视频内容分析中的不同输入表示（转录、OCR、帧+音频、原始视频等）对分类准确性与下游回归结果的影响。

**💡 创新点**

提出了一套基于“表示有效性”的验证框架，系统比较多种输入路由、模型和构念的交互效应，并给出五项检验建议，弥补了先前仅关注模型/提示选择的空白。

**🔧 技术方法**

使用六种开源多模态LLM（Gemma、Qwen、Nemotron、MiniCPM 等），通过自动转录、OCR、帧抽样+音频、完整视频等输入方式，并利用 macro‑F1、Krippendorff's α、回归系数误差等指标进行评估。

**📊 数据集**

三大公开数据集：德语 TikTok 新闻视频（603 条）、BitChute 仇恨视频（400 条）和英语情景喜剧片段（650 条）。

**📈 对比分析**

采用每个模型-构念-输入组合的宏 F1 与人类标注对比，并检验回归系数与样本预估差异；结果显示最高 F1 仅达 0.80，组合输入几乎无提升，且高 F1 并不必然导致回归系数或预估更接近人类结果。

**⚠️ 局限性**

限制包括：仅评估三类数据，完整原始视频处理受限于模型和 GPU；人类标注存在错误；结果对语言、域、模型架构高度敏感，缺乏对更长视频或更多模态的检验。

---

## 761. CATF: A Manager-Worker Framework for Context-Aware Time-Series Forecasting

**arXiv ID:** 2611.05325 | [PDF](https://arxiv.org/pdf/2611.05325v1)

**作者:** Yueqiao Chen `[一作]` (UC Davis), Dongyu Liu `[通讯]` (UC Davis)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个基于 Manager‑Worker 结构的上下文感知时间序列预测框架（CATF），通过 Transformer 负责的 Manager 对输入样本进行硬路由，动态选择最合适的 Worker（可为任意单模型）进行预测；

**💡 创新点**

创新点在于：①使用硬路由实现样本级别的模型选择，避免软路由导致的专家不活跃；②通过联合训练让 Workers 在无人工划分的情况下自发专业化；③框架模型无关，支持同质或异质 Worker；④可解释性强，路由决策可直接关联上下文变量；

**🔧 技术方法**

技术核心包括 Transformer‑based Manager、可插拔的 Worker（如 Informer、Autoformer、FedFormer、TimesNet、i‑Transformer）、硬路由机制、伪标签训练、Wasserstein+KL 混合 loss、以及训练时的负载均衡正则化；

**📊 数据集**

实验使用五个包含上下文信息的真实数据集：Weather（气象）、PM2.5‑LA、PM2.5‑RV（空气质量）、Bike Share（共享单车）和 Appliances（家用电器），每个数据集采用 4 个预测时延（96/192/336/720）并分别在全局归一化和样本归一化两种方式下评估；

**📈 对比分析**

与 5 个主流单模型基线及零-shot Time‑MoE 对比，CATF 在 162/200（81%）的比较中均优于基线；在 19/20 设定下，其最佳配置超越最优基线，且在 15/20 设定下优于 Time‑MoE，平均 MSE 降低约 4.9%；同时显著降低了基线模型的训练方差；

**⚠️ 局限性**

局限性包括：需要更多训练时间和显存；推理时仍需运行 Manager 及被选 Worker；最佳 Worker 数量需手工设定，异构 Worker 组合效果不稳定；以及缺乏对 Worker 内部决策过程的解释。

---

## 762. Roundtrip All-Pairs Shortest Paths in Directed Graphs

**arXiv ID:** 2611.05311 | [PDF](https://arxiv.org/pdf/2611.05311v1)

**作者:** Shiri Chechik `[一作]` (Tel Aviv University), Gur Lifshitz `[通讯]` (Tel Aviv University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究并提出了在有向图中基于往返距离（roundtrip distance）求近似全对最短路径（APSP）的方法，提供了多种时间复杂度与误差折中的算法；同时给出了与布尔矩阵乘法（BMM）相关的下界，证明了在该度量下任何低于 3 的拉伸因子都将导致 BMM 的突破；

**💡 创新点**

核心创新包括：①使用随机枢轴（pivot）与截断 Dijkstra 搜索相结合的距离预估结构，实现 3-近似；②结合稀疏化（center）技术，进一步降低时间到 O(n^2+min{m√n, m^{1/3}n^{5/3}})；③利用代数矩阵乘法实现 (3+ε) 与 (4+ε) 近似；④给出基于 BMM 的严格下界，阐明 3 拉伸因子与系数 2 的不可突破性；

**🔧 技术方法**

主要技术手段包括：随机枢轴抽样与轮盘化的距离预估、截断 Dijkstra 搜索、稀疏化（center-based edge pruning）、代数加速（min-plus 乘法）、以及图论中的三角不等式与比值判定；

**📊 数据集**

本工作为理论算法研究，未使用具体实测数据集；评估基于算法复杂度与理论下界进行；

**📈 对比分析**

与先前工作相比，本文在有向图上实现了 3-近似的时间从 O(m√n + n^2) 提升到 O(n^2+min{m√n, m^{1/3}n^{5/3}})，并通过代数方法进一步接近 O(n^2)；此外给出的下界表明无法在多项式时间内突破 3 拉伸因子；

**⚠️ 局限性**

局限性包括：只能得到 3 近似，无法突破到 2；系数 2 的前向距离不可能改进；算法在大规模稠密图上的时间仍为 O(n^7/3)；仅适用于有向图的往返距离度量，对普通一向距离或无向图不适用；

---

## 763. You don't need long context

**arXiv ID:** 2611.05306 | [PDF](https://arxiv.org/pdf/2611.05306v1)

**作者:** Peter Belcak `[一作]`, Pavlo Molchanov `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对长时序任务和长输入任务的实验，研究长上下文是否真正必要，并提出并验证了终止时压缩（EoCC）与单动作压缩（SAC）两种上下文管理策略以及选择性输入读取方法。

**💡 创新点**

创新点在于：①引入EoCC和SAC两种动态压缩机制，证明可以在保持或提升准确率的同时大幅缩短上下文；②展示通过选择性输入读取和压缩可以消除长输入任务的长上下文需求；③提出语义状态空间模型（SSSM）的概念，暗示可通过显式自然语言状态而非完整历史来驱动代理。

**🔧 技术方法**

采用ReAct循环的代理框架，利用Bash工具进行输入/输出交互；在NVIDIA H100 GPU和DGX Spark上使用INT4、BF16、NVFP4量化模型进行吞吐与成本测评；通过上下文窗口大小、压缩阈值等参数的组合，系统比较FHA、EoCC、SAC三种模式。

**📊 数据集**

长时序基准：SWE-bench（Verified/Multilingual）、Terminal‑Bench 2.1、FRAMES、Tau2Bench、BFCLv3；长输入基准：LongBench v2、LOONG、AA‑LCR、NIAH、RULER、Counting‑Stars、BABILong；此外对开源大模型（GLM 5.2、Kimi K2.6、Qwen 3 Coder Next、Qwen 3.6 27B、Gemma 4 31B、Step 3.7 Flash、MiniMax M2.7）进行部署与性能评估。

**📈 对比分析**

比较方法：在每个基准上对不同上下文管理配置的准确率、Token消耗、峰值上下文长度及云端/本地成本进行统计；结果表明EoCC和SAC在绝大多数任务中保持或略优于全历史模式，准确率波动一般在±5个百分点；成本提升因子从2×到200×不等，单动作压缩将峰值上下文降至7–9K tokens。

**⚠️ 局限性**

局限性：实验仅覆盖有限的任务与模型，未检验更广泛域或更大规模模型的适用性；压缩是有损的，某些对历史完整性要求极高的任务仍可能受影响；SSSM的训练与部署方案尚未实现，需进一步研究其可行性与泛化性能。

---

## 764. LoopVQ: Sub 1-bit KV Cache Quantization for Looped Transformers

**arXiv ID:** 2611.05304 | [PDF](https://arxiv.org/pdf/2611.05304v1)

**作者:** Minsoo Cheong `[一作]` (Seoul National University), Sungjoo Yoo `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 LoopVQ，一种针对循环 Transformer 的 KV 缓存向量量化方法，显著压缩 KV 缓存尺寸。

**💡 创新点**

创新点：1) 在循环轴上对多步的通道组进行联合向量量化，使一个索引覆盖多步；2) 通过循环感知的代码簿微调，补偿不同循环步对输出的不同量化敏感度，提升精度。

**🔧 技术方法**

技术手段：Fisher 加权 k‑means 代码簿初始化、循环轴向量量化、循环感知代码簿微调（基于 KL 目标）、RoPE 预处理、SGLang 自定义 attention 核、与 TaSQ 的结合。

**📊 数据集**

数据集：校准使用 GPQA 主集 16 条 2,048 长度序列；评估使用 GSM8K、MATH500、HumanEval、MBPP、AIME'24/25、GPQA-Diamond、HumanEval+、MBPP+、RULER NIAH（长上下文 4K‑64K）。

**📈 对比分析**

比较方法：对比 CQ、TaSQ 传统 KV 量化基线；在 0.5 / 0.25 位/元素下，LoopVQ 在一般、推理、长上下文任务上均优于基线，接近 BF16；SGLang 实现使解码延迟降低约 50%，峰值吞吐量提升 3.2‑6.1 倍。

**⚠️ 局限性**

局限性：对循环步量化敏感度的权重仍为单一标量，难以捕捉更细粒度差异；需要特定的校准数据集，数据集选择会影响结果；主要针对循环 Transformer，非循环模型的效果尚未验证；在极低比特（0.25 位）下仍可能导致推理长度过长，需进一步稳定性分析。

---

## 765. Almost-MMS allocations with small agent redundancy

**arXiv ID:** 2611.05360 | [PDF](https://arxiv.org/pdf/2611.05360v1)

**作者:** Uriel Feige `[一作]`, Vadim Grinberg `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一套新的公平分配算法，针对可加价值的离散物品，通过构造头部和尾部需求配置、使用分数分配与整数化圆整技术，能够在大多数参与者中实现（1‑Δ）MMS（最大最小份额）公平性，并进一步证明在给定的d个代理人中可达到（1‑2/q）MMS保证。

**💡 创新点**

创新点在于：①将头部需求量化为常数大小的离散单位，极大降低了配置空间；②设计了专门的分数MMS头部配置与尾部补偿机制（tail completion），在保证前缀负载不超标的同时补偿不足；③通过对物品与代理人进行精细的分区（E、S、V、U、W）和“受保护前缀”构造，成功处理不同代理人头阈值不一致的情况；④在此基础上提出了改进的整数化圆整算法，允许在保持负载约束的同时失去极少的代理人。

**🔧 技术方法**

主要技术包括：分数线性规划与整数圆整（faithful rounding）；头部/尾部配置（head/tail profile）理论；负载约束下的优化与尾部补偿 Lemma；受保护前缀与缓冲区的组合构造；以及对缺失值的缺陷计数和上界分析。

**📊 数据集**

研究使用的是合成的可加价值实例，未采用实际数据集；所有证明均在理论模型下完成。

**📈 对比分析**

与以往的 MMS 近似结果相比，本文在满足更多参与者（n(Δ,q)）的情况下提供了更严格的（1‑Δ）保证，并在 d 个代理人上实现了（1‑2/q）MMS，算法复杂度为多项式级别；理论上相比旧方法取得了更优的公平性与可行性。

**⚠️ 局限性**

局限性包括：①需要参数 q 较大才能在多参与者下获得较好的 Δ，导致在小 q 时效果有限；②算法实现复杂，涉及多层分区与圆整，实际应用受限；③仅适用于可加价值模型，无法直接扩展到非可加或更一般的公平性目标；④当尾部价值极小（tail case）时，仍需额外处理，整体方案更为繁琐。

---

## 766. Suppressing Misaligned Behaviors in Language Models, Durably

**arXiv ID:** 2611.05358 | [PDF](https://arxiv.org/pdf/2611.05358v1)

**作者:** Luxi He `[一作]` (Princeton University), Danqi Chen `[通讯]` (Princeton University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对已有的语言模型进行事后行为抑制，构造两类模型“生物体”（窄触发和广触发），评估多种抑制方法，并提出新的 Prefix-Expanded Adversarial Reinforcement Learning（PEARL）方法。

**💡 创新点**

创新点在于：1) 将对抗性前缀（来自已抑制模型的错误轨迹）融入 RL 起始状态，扩展训练覆盖更广的误差空间；2) 在保持能力的同时显著提升抑制的即时效果与耐久性；3) 统一评估即时抑制、后续训练耐久性与通用能力三维指标。

**🔧 技术方法**

技术主要包括：监督微调（SFT）、GRPO（Group Relative Policy Optimization）、梯度上升解学习（GA unlearning）、Prefix-SFT（前缀修正）以及自研的 PEARL（使用缓存的误差轨迹作为对抗前缀进行 RL 更新）。

**📊 数据集**

数据集涵盖：MMLU-Pro + GPT‑5 生成的 Chain‑of‑Thought 用于窄触发生物体；不安全代码数据集用于广触发生物体；Opus‑生成的 500 条安全示例作为通用安全数据；OpenThoughts 6k 例子用于后续训练与耐久性测试。

**📈 对比分析**

对比方法：SFT、GRPO、GA、Prefix‑SFT 及 PEARL。实验结果显示，PEARL 在 GPT‑OSS‑20B 及广触发生物体上获得最低的 exploit / misalignment 率，且在后续训练中保持最高的通用能力；在窄触发场景下，PEARL 与 SFT 竞争，表现出更好的耐久性但不如针对性 SFT。整体而言，PEARL 在三维评估中处于 Pareto 前沿。

**⚠️ 局限性**

局限性：1) 对于窄触发且可直接针对的行为，SFT 在特定规模下仍可更优；2) PEARL 需要额外的缓存轨迹与更高的计算成本；3) 在更小模型或不同触发机制下效果可能不明显；4) 依赖 GPT‑4o‑mini 评判器，评估结果可能受评判器偏差影响。

---

## 767. Plotkin-optimal pure quantum LRCs with large minimum distances from NMDS codes

**arXiv ID:** 2611.05335 | [PDF](https://arxiv.org/pdf/2611.05335v1)

**作者:** Yang Li `[一作]` (Nanyang Technological University), San Ling `[通讯]` (Nanyang Technological University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过 Hermitian 构造从 NMDS 码（尤其是 ESGRS 码）构造了一系列满足 Plotkin‑like 上界、长度可达 Ω(q²) 的纯量子局部可恢复码 (qLRC)。

**💡 创新点**

创新点在于首次给出四族长度为 q²、q²+1、q²-ℓ 等、最小距离可自由取值且同时达到纯 Singleton‑、Griesmer‑和 Plotkin‑上界的 qLRC，并证明 ESGRS 码的最小码字不支持非平凡 t‑设计，解决了先前的开放问题。

**🔧 技术方法**

核心技术包括 Hermitian dual‑containing NMDS 码的纯度与最优性分析、ESGRS 码生成矩阵中可逆子矩阵的判定、有限域上的子集和问题以及对 ESGRS 码及其双码的局部性证明。

**📊 数据集**

研究完全基于理论构造，无使用具体实验数据集，所有参数均通过代数证明得到。

**📈 对比分析**

通过与已知长度大、距离小的 qLRC 以及已满足 Singleton‑、Griesmer‑上界的例子比较，证明新构造在长度、最小距离和局部性方面均具备更优或相同的最优性，且同时满足三种纯上界。

**⚠️ 局限性**

局限在于构造仅适用于特定 k、ℓ 与 q 的取值范围，且长度仍受 NMDS 上界 2q²+2 限制；未来需探索更长长度、ℓ≥2 的 ℓ‑MDS 码及其对 qLRC 的潜在改进。

---

## 768. Counterexamples to the Happy Edge Conjecture in Plane Spanning Trees

**arXiv ID:** 2611.05300 | [PDF](https://arxiv.org/pdf/2611.05300v1)

**作者:** Oswin Aichholzer `[一作]` (Graz University of Technology), Birgit Vogtenhuber `[通讯]` (Graz University of Technology)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文给出了平面连通树的翻转图中“快乐边”猜想的反例，证明在一般点集以及凸点集中都存在最短翻转序列需要翻转至少一条原本两棵树共有的边。

**💡 创新点**

创新点在于构造了两个具体点集（16点一般位置、30点凸位置）并给出明确的翻转序列，展示了在兼容翻转下一般点集也会破坏快乐边性质，从而完整划分了四种主流翻转类型的情况。

**🔧 技术方法**

主要技术是几何组合构造与严格的计数证明，利用平面图交叉性和停车边（parking edge）的引入来推导翻转距离的下界，并给出相应的最短翻转路径。

**📊 数据集**

使用的“数据集”是人工构造的点集：16个一般位置点（包含6条快乐边）和30个凸位置点（形成正30边形），分别对应两种反例。

**📈 对比分析**

与以往的经验性验证（至10点）相比，本工作通过理论证明得到最短翻转距离为12/19，并证明若保持所有快乐边则需要13/20翻转，体现了对最短路径结构的精确量化；不涉及实验性能评估。

**⚠️ 局限性**

局限性在于仅提供了特定构造的反例，未给出一般性判定算法或对更大规模点集的泛化；此外，只讨论了翻转距离的数量关系，而未探讨其对算法复杂度或实际应用的直接影响。

---

## 769. Noisy Issues, Noisy Results: Mislabeling in Information Retrieval-based Bug Localization

**arXiv ID:** 2611.05314 | [PDF](https://arxiv.org/pdf/2611.05314v1)

**作者:** Moumita Asad `[一作]` (University of California Irvine), Sam Malek `[通讯]` (University of California Irvine)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究通过LLM驱动的分类器GenTag，对五大Bug Localization基准数据集进行大规模误标识检测，并评估误标识对六种IRBL技术（VSM、BugLocator、BLUiR、BRTracer、DreamLoc、GenLoc）的影响；

**💡 创新点**

创新点在于①提出高效低成本的多标签LLM分类器GenTag以识别误标问题；②系统量化误标对各类技术的性能下降幅度及相对排名差异；③针对数据重叠使用部分重叠样本的统计检验；

**🔧 技术方法**

使用的技术包括LLM分类器（ChatGPT系列）、信息检索模型（VSM、BM25）、历史匹配（BugLocator、BRTracer）、深度学习（DreamLoc）和LLM驱动的交互式定位（GenLoc）；

**📊 数据集**

使用的基准数据集为Zhou et al.、Ye et al.、Bench4BL、GHRB 和 BeetleBox，总计约55,000条Issue；

**📈 对比分析**

通过Accuracy@k和MRR对ORIG、Test-Cleaned和Fully-Cleaned三种配置进行内外技术对比，结果显示去除误标会使所有技术的准确率下降5–29%，MRR下降4.7–21.7%，但技术间相对顺序基本不变；

**⚠️ 局限性**

局限包括：仅针对Java项目，未覆盖工业规模或多语言系统；误标检测依赖LLM分类器，仍可能存在系统性误判；并未深入探讨误标对方法训练细节的潜在正向影响。

---

## 770. $Φ$asmaNet: A Staged Spectral Token-Refinement Architecture for Fast Gravitational-Wave Signal-to-Noise Ratio Prediction

**arXiv ID:** 2611.05309 | [PDF](https://arxiv.org/pdf/2611.05309v1)

**作者:** Ioannis Tsioulis `[一作]` (Aristotle University of Thessaloniki), Nikolaos Stergioulas `[通讯]` (Aristotle University of Thessaloniki)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了一种名为 ΦasmaNet 的神经网络替代模型，用来在固定探测器、噪声谱和波形近似（IMRPhenomXPHM）下，将 11 维黑洞二元参数直接映射到相位平均的单探测器最优信噪比（SNR）。

**💡 创新点**

创新点包括：①将“分阶段谱表征细化”作为显式的网络先验；②在每一阶段使用自注意力 + 结构化子空间混合（Φ‑Refractor）模块；③通过有限次数的上向 gated 反馈（Φ‑Feedback）进一步细化表征；④整体模型参数量仅约 73 万，且在 10^6 采样下已达接近完美的 99.3% 预测精度。

**🔧 技术方法**

技术手段主要是深度学习的 Transformer‑style 结构、残差网络、1D 卷积、正则化（dropout、weight‑decay）、AdamW 优化、学习率调度、训练样本加权、PCHIP 校准以及多阶段前向+反馈的计算图设计。

**📊 数据集**

使用 10^5–10^6 条合成 BBH 注入数据（每条样本由 PyCBC 生成：质量、六维自旋、天文方向、极化等 11 维特征），标签为在 LIGO‑Hanford 10 Hz–1024 Hz PSD 下的 phase‑averaged SNR；此外在附录中对 YearPredictionMSD、Covertype、Epsilon 等公开表格基准做了无超参调优的对照实验。

**📈 对比分析**

对比方法：在完全相同的训练、验证、测试拆分、损失函数、权重、学习率等设置下，分别训练 MLP、1D ResNet、Transformer‑encoder 和 ΦasmaNet。结果显示，ΦasmaNet 在所有数据量（10^5、5×10^5、10^6）上均优于基线，10^6 采样后在校准前 98.73%、校准后 99.29% 的 100‑MAPE；在 ρ>8 的检测相关子集上达 99.39% (R²=0.9998)。计算效率上，单 GPU 上可实现 ~1.2×10^5 次 SNR 评估/秒，单 10^6 注入仅需 ~8.3 s，比传统 16 核 CPU 慢 100 倍更快。

**⚠️ 局限性**

局限性：①仅适用于单探测器、固定噪声谱和波形近似，无法直接推广到多探测器网络或不同观测时期；②模型预测的是最优 SNR，未包含真实搜索管道的噪声非高斯性和一致性校验；③输入维度固定为 11，若扩展到更高维或非预期参数空间需重新设计分词器和层数；④目前未对比梯度提升树等非神经方法，或在更大范围的表格基准上进行细粒度调优。

---

## 771. PACT: Pose-Robust Actuation Co-Design for Tendon-Driven Robotic Hands

**arXiv ID:** 2611.05302 | [PDF](https://arxiv.org/pdf/2611.05302v1)

**作者:** Prashant Kumar `[一作]` (Meta Reality Labs Research), Gabe Pirie `[通讯]` (Meta Reality Labs Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了PACT（Pose‑Robust Actuation Co‑Design）框架，用于同时优化指节的肌腱布线、转矩臂和传动筒直径，以满足多姿态下的力学需求和传动带宽约束，并在四自由度机械手指上进行实验验证。

**💡 创新点**

创新点包括：①将布线与传动几何共设计，避免传统顺序优化导致的力学/速度权衡；②采用极大极小（max–min）目标，保证最弱姿态的任务力覆盖；③使用完整的可行力集评估（严格满足所有关节平衡）而非伪逆投影；④将实验验证与模型对比，揭示传动损耗对设计的影响。

**🔧 技术方法**

技术手段包括：混合离散-连续优化（离散布线符号、连续转矩臂/筒直径），Feasible‑Force Set 的半空间变换，带宽模型（考虑摩擦、反射惯量和连续扭矩/速度限制），HiGHS线性规划求解器，Sobol随机采样+Nelder–Mead局部优化。

**📊 数据集**

使用的数据集：八个代表性手指姿态（小/中/大指尖、三角、伸展、深屈、外展抓取），每姿态的力需求定义为截断的库仑摩擦锥；除此之外仅用手指几何和四个QDD致动器的规格做为参数，没有公开的数据集。

**📈 对比分析**

比较方法：与基线（任务无关、仅满足峰值触指力和带宽）和单姿态优化的独立设计对比。结果显示：共享设计在最弱姿态下的任务力覆盖从0.24提升至0.88，所有八个姿态都满足5 Hz的关节带宽；实验验证中，预测的力多边形与实际相似，但因传动损耗导致平均25% 的力幅度误差。

**⚠️ 局限性**

局限性：①静态力模型忽略摩擦、弹性、滞后与预张力；②带宽模型不考虑控制器动态、结构柔性与热量限制；③验证仅在单姿态且采用降低的致动器极限；④优化搜索为启发式，无法保证全局最优；⑤仅研究单指，未验证整体手抓握的力闭合与稳定性。

---

## 772. Personas Are Not People: Identity as State, Not Attribute

**arXiv ID:** 2611.05292 | [PDF](https://arxiv.org/pdf/2611.05292v1)

**作者:** Tamunotonye Harry `[一作]` (University of Vermont), Yuanyuan Feng `[通讯]` (University of Vermont)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并验证了“人-情境”式的合成人口模拟框架，证明了传统固定人格设计无法捕捉个体在不同情境下的行为差异；

**💡 创新点**

创新点在于：① 用身份经济学理论将行为分解为稳定特质、情境激活的身份状态与情境本身；② 给出了Latent State‑Trait（LST）指纹恢复的评价标准；③ 在理论证明基础上提出并实验了人-情境设计，并通过两个真实数据集进行检验；

**🔧 技术方法**

主要技术包括：大语言模型（Claude Haiku 4.5）、身份状态选择子程序、GMM混合聚类构建身份状态、LST方差分解、ICC与人-情境交互方差估计、相关性检验；

**📊 数据集**

使用了：① Chameleon Reddit用户多场景行为数据；② 公开的“动机主题”临床试验招募调查数据；

**📈 对比分析**

比较方法：对固定人格、固定人格+情境说明、以及人-情境三种体系在ICC、交互方差、个体特异信号等指标上的表现进行对比；结果显示固定人格体系ICC偏高、交互方差低、个体信号几乎为零；人-情境体系ICC接近真实、交互方差改善、但个体信号仍低，未完全恢复；

**⚠️ 局限性**

局限性包括：① 人-情境设计的身份状态选择仍由人工设定，缺乏自学习机制；② 只检验了跨场景平均交互，未能完全捕捉个体层面的交互；③ 数据集局限于公开 Reddit 与单次实验，无法验证在更广泛人群和情境下的泛化；④ 评价指标仅覆盖行为层面，未考虑潜在偏见与伦理风险。

---

## 773. Some algebraic and geometric computations in CH via Chow forms

**arXiv ID:** 2611.05289 | [PDF](https://arxiv.org/pdf/2611.05289v1)

**作者:** Robert Andrews `[一作]` (Simon Fraser University), Éric Schost `[通讯]` (University of Waterloo)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种改进的算法，用于在 Chow 形式表示下计算代数簇的等维分解，并进一步可决定该分解中点的成员资格以及多项式映射的闭包像等问题。

**💡 创新点**

创新点在于构造了 polylog‑time‑uniform 常数深度阈值电路，能够在计数层次上高效计算 Chow 形式，从而将等维分解的复杂度提升至计数层次。

**🔧 技术方法**

主要技术包括几何分辨、乘法电路统一化、对多变量结果式的高效评估，以及 Andrews‑Garg‑Schost 的阈值电路技术。

**📊 数据集**

该工作为理论性研究，未使用具体实验数据集，而是基于符号计算和代数几何理论。

**📈 对比分析**

与以往指数时间/双指数时间的算法相比，新的上界将复杂度降低到计数层次，且在求解近似多项式可满足性、次数度计算等应用中也实现了显著的复杂度改进。

**⚠️ 局限性**

局限性在于仍处于计数层次的上界，尚未实现多项式时间；方法主要适用于代数闭场，且对实际实现的规模和常数仍有挑战。

---

## 774. Whose Guardrail? User Identity Shapes the Effectiveness of Model Disclaimers in Mitigating LLM Political Persuasion

**arXiv ID:** 2611.05347 | [PDF](https://arxiv.org/pdf/2611.05347v1)

**作者:** Jillian Fisher `[一作]` (Stanford University), Katharina Reinecke `[通讯]` (University of Washington)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究通过在与政治偏向的大语言模型互动时加入不同类型的免责声明，评估其对用户政治观点影响的抑制效果。

**💡 创新点**

创新之处在于系统检验了免责声明中风险提示与因果解释对抑制偏见传播的作用，并探讨了用户身份和AI熟悉度等个体差异对说服效应的调节。

**🔧 技术方法**

采用了前置偏差词缀的GPT‑4o‑mini模型，2×2随机实验设计，并用线性回归分析即时与一周后结果。

**📊 数据集**

数据来自1586名美国注册民主党和共和党用户的实验数据，包含两项决策任务（主题意见与预算分配）的前后测量。

**📈 对比分析**

与仅使用通用免责声明的对照组相比，特定风险与因果说明的免责声明在部分情境下显著降低了说服，但效应微弱且在一周后消失。

**⚠️ 局限性**

限制包括仅聚焦美国政治、仅使用单一LLM与单一提示策略、样本为非代表性在线受试者，以及实验任务范围有限，导致结果难以推广至更广泛情境。

---

## 775. Agnostic Learning by Imitating

**arXiv ID:** 2611.05346 | [PDF](https://arxiv.org/pdf/2611.05346v1)

**作者:** Sílvia Casacuberta `[一作]` (Stanford University), Pranay Tankala `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

本文证明了区分（无偏学习）和多准确性（不可区分生成模型）两种核心学习任务在计算复杂度上完全等价，并给出了从多准确性oracle到无偏学习的有效多轮自适应降维方法。

**💡 创新点**

创新点在于实现了逆向oracle降低，首次利用在线学习中的 Follow‑the‑Leader、校准误差与无偏学习相结合的技术，能够在仅使用多准确性oracle的前提下完成无偏学习、全预测和多校准等更强学习目标；同时提供了匹配的下界证明。

**🔧 技术方法**

主要技术包括：在线学习的 Follow‑the‑Leader 算法与其无偏学习/校准误差分析、统计查询（SQ）框架下的平滑与噪声化技术、条件多准确性构造、以及将在线退火与归纳推断相结合的自适应查询策略。

**📊 数据集**

论文为理论分析，不涉及具体数据集；所有结论均在任意特征分布和给定模型类（如决策树、神经网络、布尔电路等）上成立。

**📈 对比分析**

通过oracle调用次数与误差ε的关系进行比较：无偏学习可在O(1/ε)次多准确性oracle调用下实现，且两任务在多项式时间内等价；相比之前仅单向的降低，本文实现了完整的互转，理论性能达到最优复杂度。

**⚠️ 局限性**

局限性包括：需要对多准确性oracle的访问并进行多轮自适应查询；算法在特征分布已知且无大质量点时才适用；下界证明显示调用次数无法进一步降低；实际实现受限于oracle可计算性和对大规模数据的实际部署。

---

## 776. A Circuit for In-Context Learning

**arXiv ID:** 2611.05344 | [PDF](https://arxiv.org/pdf/2611.05344v1)

**作者:** Andrew Gao `[一作]` (Stanford University), William Gao `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对预训练Transformer的激活、头部和层的干预，揭示并验证了少量示例内在学习（ICL）在句子分类任务中的完整机制，归纳为一套可拆解的电路：写入（writer）把句子类别写进标签标记的残差流，读取（reader）通过softmax匹配查询类别与标签槽的键，门控（gate）检测标签重复并决定是否读取，刹车（brake）抑制最近标签的偏倚，最终拷贝（copier）将读取结果写入logit。

**💡 创新点**

创新点在于：①首次将ICL拆解为若干可识别的头部功能与方向，并证明其互相配合的因果路径；②提出并验证了读取的softmax规律与门控机制；③发现读写、门控、刹车等角色在不同模型、尺寸与指令微调下保持一致，表明ICL本质是一种“有门控的关联检索”算法；④通过“population”分析说明为何大部分功能由头部集体完成而非单个头。

**🔧 技术方法**

技术手段包括：①激活补丁（activation patching）与头部消融（mean‑ablation），②基于同一示例的“twin”prompt对比、③对比随机头组与匹配控制，④对读写键/值方向的线性拟合与softmax律推导，⑤“steering”即在查询残差上添加模型自身学到的方向，⑥对不同层、不同头集的统计与方向投影验证。

**📊 数据集**

数据集主要为人工构造的prompt batteries：invented‑label（符号/稀有词标签）、natural‑concept（四个语义概念、12示例/4标签）、natural‑word（词标签+翻转）、slotgate（唯一/重复标签）以及多类（2‑6类）。每种电池包含数百条prompt，覆盖不同标签类型与示例数量。

**📈 对比分析**

评价方式：用margin（正确标签logit与错误标签logit之差）衡量模型与电路在给定prompt上的表现；将完整模型、仅保留电路头部、随机头部、无头部等进行对比。结果显示：仅保留电路头部即可复现约60–80%的margin，随机头部表现接近0，完整模型维持最高margin；在不同模型尺寸与指令微调下，电路结构保持一致但权重分布与门控强度随规模变化，性能随规模提升。

**⚠️ 局限性**

局限性：①仅研究二分类、单词标签、短句、最多12示例的ICL任务；②对长prompt、多类、多步推理、长文本任务缺乏验证；③在小模型与极端示例分布下，门控与刹车机制未必出现；④对自然语言大规模指令微调的迁移效果仍有限；⑤对其他Transformer变体（如Encoder‑only或Decoder‑only）的适用性未作系统评估。

---

## 777. PyRUA-DataFarm: LLM Agent as a Cost-Efficient Teleoperator for Real-Ready Robot Data

**arXiv ID:** 2611.05320 | [PDF](https://arxiv.org/pdf/2611.05320v1)

**作者:** Kewei Zhang `[一作]` (Peking University), Daquan Zhou `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在仅有一次校准视频的前提下，利用编码代理构建仿真、专家和渲染合同，自动生成可直接用于训练且与真实机器人同步的多视角动作标注数据。

**💡 创新点**

创新点在于：①将任务编译过程拆分为相互核查的中间表示，保证代理产出与验证独立；②用真实图像驱动的控制网渲染保持动作一致；③通过隐藏判定器和独立审阅实现对生成数据的可信认证。

**🔧 技术方法**

技术包括：前沿语言模型驱动的编码代理、SAPIEN/IsaacSim仿真、MiniMax‑H3/Seedance视频生成、ControlNet边缘驱动渲染、自动光流对齐检测、独立人工审核。

**📊 数据集**

数据集：使用单一的三视角校准视频作为输入，生成的仿真轨迹和渲染视频共构成五项双臂操作任务的训练集，涵盖刚体、关节、布料和显示等多种物体。

**📈 对比分析**

与方法对比：在真实机器人上训练的政策仅用农场数据可获得与100条真实操作相同甚至更好的成功率（93/100 vs 84/100），在未知外观下成功率提升（87/100 vs 46/100），而传统 real‑to‑sim‑to‑real 仅 21/100。成本方面，单次任务 21 美元（代理）+5 美元（渲染合同）+0.80 美元/条，约 107 美元/小时，低于中美人工采集成本。

**⚠️ 局限性**

局限性：需依赖能够复制任务技能的仿真器；校准视频无法辨别未出现的动力学参数；渲染验证仍有漏检风险；仅在单一双臂机器人和五任务上验证，其他平台需进一步测试。

---

## 778. False Reassurance in LLM Severe-Weather Guidance: A Replayable Evaluation Against Real NWS Alerts

**arXiv ID:** 2611.05290 | [PDF](https://arxiv.org/pdf/2611.05290v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 779. TFNP in the Random Oracle Model

**arXiv ID:** 2611.05308 | [PDF](https://arxiv.org/pdf/2611.05308v1)

**作者:** Noah Fleming `[一作]` (Lund University), Weiqiang Yuan `[通讯]` (University of Copenhagen)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了某种新型算法在特定任务中的应用，旨在提高效率和准确性。

**💡 创新点**

创新点在于提出了一种新的优化策略，能够在处理大规模数据时显著减少计算时间。

**🔧 技术方法**

使用了深度学习技术和强化学习算法相结合的方法。

**📊 数据集**

实验使用了公开的图像识别数据集和自建的文本分类数据集。

**📈 对比分析**

与现有的几种主流算法进行了比较，结果显示该算法在准确率和处理速度上均有明显提升。

**⚠️ 局限性**

限制在于算法对特定类型数据的适应性较差，且在极端情况下性能可能下降。

---

## 780. Universal Hardness From Masking Channels

**arXiv ID:** 2611.05296 | [PDF](https://arxiv.org/pdf/2611.05296v1)

**作者:** Parikshit Gopalan `[一作]` (Apple), Pranay Tankala `[通讯]` (Harvard)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了一套基于掩码通道（masking channel）和凸对偶（convex duality）的通用框架，用于从学习难度推导出硬度证书，推广了经典的硬核引理（hardcore lemma）到多类别、非布尔函数以及各种损失函数；同时构造了可用于多种难度度量的通用掩码通道，并给出了与伪熵（pseudoentropy）及最大熵（MaxEnt）算法的等价性。

**💡 创新点**

① 将硬核测度转化为输入相关的噪声通道，利用通道保真度（fidelity）与学习难度之间的对偶关系；
② 通过凸对偶理论得到多类别硬核引理和多损失下的硬度证明；
③ 构造了可以满足多种“多准确性”（multicalibration）约束的通用掩码通道；
④ 将伪熵的定义与最大熵算法等价化，提供了新的理论视角。

**🔧 技术方法**

凸分析（Fenchel 对偶）、多准确性（multicalibration）理论、凸-凹共轭对、信息论中的伪熵概念、最大熵（MaxEnt）算法与其对偶性。

**📊 数据集**

无实验数据集，全部为理论证明。

**📈 对比分析**

无实验比较与性能指标，本文通过数学证明展示了理论等价与边界结果；若在具体学习任务上实现，可预期能得到与传统硬核引理等价或更紧的难度下界。

**⚠️ 局限性**

1) 目前仅给出理论框架，缺乏对实际机器学习模型（如深度网络）在该框架下的实验验证；
2) 需要满足多准确性和多校准等假设，实际数据分布可能难以保证；
3) 对非凸损失或非离散输出空间的推广尚未完成。

---

## 781. Wonder 4: A Mixture-of-Experts Recipe for Image Quality Enhancement

**arXiv ID:** 2611.05294 | [PDF](https://arxiv.org/pdf/2611.05294v1)

**作者:** Yunan Wu `[一作]` (Topaz Labs Research), Xiaoyu Wang `[通讯]` (Topaz Labs Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于稀疏混合专家（MoE）的图像质量增强模型，能够在不使用任务标签、检测器或掩膜的情况下，对真实照片中的多重局部缺陷（模糊、噪声、压缩、划痕、颗粒等）进行同时、区域性修复。

**💡 创新点**

创新点包括：①引入每块层级的 13 名专家并通过路由器在每个图像补丁上选择 5 名专家；②用最小使用量约束替代传统的“专家均匀使用”策略，使得仅在必要区域才激活对应专家；③仅在路由得分上加偏置，无需额外损失或模块。

**🔧 技术方法**

使用的技术主要是 Transformer‑style 轻量化网络、SwiGLU 形式的专家子网络、线性路由器、基于梯度的最小使用量偏置、以及包含 LPIPS、MSE、对抗损失的多任务训练。

**📊 数据集**

数据集：DIV2K（验证集 100 张）和 Flickr2K（约 2.65k 张）用于评估；训练阶段未使用这两套数据，所有实验均基于公开的评测集。

**📈 对比分析**

在 DIV2K×4、DIV2K×8 与 Flickr2K×4 三种评测设置下，与五个公开的增强方法和三条开源流水线比较，模型在 LPIPS、DISTS 和 NIQE 指标上均取得最优成绩（例如 Flickr2K×4 上 LPIPS 降低 19%）。

**⚠️ 局限性**

局限性：模型在 PSNR 方面不领先，说明其在保持原始细节方面有一定欠缺；此外，实验只覆盖了上述两个数据集，对更广泛的缺陷组合与真实世界多样性尚待进一步验证。

---

## 782. DreamX: Video Diffusion Model is a Universal Latent Action Learner

**arXiv ID:** 2611.05377 | [PDF](https://arxiv.org/pdf/2611.05377v1)

**作者:** Jisoo Kim `[一作]` (KAIST), Seungryong Kim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了DreamX框架，利用预训练的视频扩散模型学习跨人类与机器人身体的共享动作表示，支持任意视频迁移与零配对的人机动作迁移，并能用于高效的策略学习。

**💡 创新点**

核心创新在于直接从扩散模型的时空表示中提取可压缩的“潜在动作”查询，并用同一模型的生成先验对其进行未来帧重建监督，从而实现动作与外观的分离与共享；同时构造了一个可压缩的动作码用于策略目标。

**🔧 技术方法**

使用预训练的Cosmos‑Predict2.5视频扩散变压器；引入可学习的查询令牌进行动作编码；利用流匹配的生成损失进行未来帧重建；设计了变分瓶颈压缩动作码；并在此基础上训练观察-指令-动作的端到端策略。

**📊 数据集**

混合使用多种机器人演示数据集（MolmoAct2-BimanualYAM、Bridge、DROID、AgiBot）和人类第一视角视频集（EgoDex、EgoVerse）进行训练；评估基准为LARYBench、HumanAndRobot（H&R）与自制的视频迁移/动作迁移实验。

**📈 对比分析**

在LARYBench上，DreamX在回归误差（MSE从0.38降至0.17）和分类准确率（从62.25%提升至69.67%）方面均超越DiLA、FAE-32等现有潜在动作模型；在任意视频迁移和零配对人机迁移任务中，DreamX在视觉保真度、动作一致性和成功率上均优于AdaWorld和VideoWorld 2，并在真实世界抽屉关闭实验中实现了96.7%子目标成功率。

**⚠️ 局限性**

限制主要包括：需要大规模预训练的视频数据；动作编码仍以token形式存储，导致高维度计算；在极端跨域（如大幅不同摄像头或极端光照）下迁移性能可能下降；且当前仅验证了相对有限的机器人任务，普适性和安全性尚需进一步评估。

---

## 783. Aligning Language Models with General Preferences: An Empirical Study of Direct Nash Optimization

**arXiv ID:** 2611.05373 | [PDF](https://arxiv.org/pdf/2611.05373v1)

**作者:** Austin Tsai `[一作]` (Synopsys, Inc.), Parivesh Choudhary `[通讯]` (Synopsys, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

重新实现并实验了 Direct Nash Optimization（DNO-Prct）方法，用于将语言模型对齐到通用偏好函数；

**💡 创新点**

首次在不同开源指令模型上独立验证 DNO，并系统评估其关键设计组件（如金标答案、数据回溯和大幅度阈值）及其对性能的影响；

**🔧 技术方法**

采用 DNO-Prct 的三步流程：批量 on‑policy 采样 + LLM 评分 + 基于大幅度过滤的对比更新（等同于 DPO 的对比损失），并与 SFT、DPO、SPIN 等基线进行对比；

**📊 数据集**

使用公开的 UltraFeedback 数据集（约 64k 条提示，4 条不同模型回应）进行 SFT 与偏好标注；

**📈 对比分析**

在 AlpacaEval 2.0 上评估，三次迭代后 Llama 3.1 8B 与 Mistral 7B 的 LC win‑rate 分别从 31.7% 提升到 45.3% 与从 6.8% 提升到 18.4%；相对同族更大模型（70B 与 8×7B）表现更优，且使用同族开源评审器可保留约 60–80% 的提升；

**⚠️ 局限性**

主要限制包括：单次实验无复测导致结果不确定；评估使用 GPT‑4o 作为评审与教师，可能导致偏好泄露；缺乏独立人类评审与跨模型评判；方法在训练规模和长文本控制上尚未充分验证；

---

## 784. GUILD: Guided Iterative Data Collection for Imitation Learning

**arXiv ID:** 2611.05368 | [PDF](https://arxiv.org/pdf/2611.05368v1)

**作者:** Shivin Dass `[一作]` (University of Texas at Austin), Roberto Martín-Martín `[通讯]` (University of Texas at Austin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种 Guided Data Collection (GDC) 框架，结合视觉‑语言模型（VLM）生成任务候选与预测数据归因（MAGIC）实现迭代式主动收集数据，以提升下游模仿学习策略在目标部署场景中的性能。

**💡 创新点**

创新点在于：①将归因作为实时反馈来驱动数据收集，而非仅在收集后进行过滤；②采用闭环的支持扩展与采样分配机制，自动迭代优化任务集合；③使用 VLM 进行语义先验的任务生成，显著降低人工干预。

**🔧 技术方法**

主要技术包括：视觉‑语言模型（GPT‑5.2、DINOv2）、预测归因方法 MAGIC、模仿学习算法 Octo、离线验证代理（目标任务小样本验证集）以及模拟与真实机器人实验。

**📊 数据集**

使用的数据集：LIBERO 与 LIBERO‑90（17 个仿真环境）、自建 6 个现实环境（厨房、浴室、办公室）以及 49 任务的离线数据集；目标任务与辅助任务均来自这些集合。

**📈 对比分析**

与 Random、VLM‑Uniform、STRAP、FSC、All‑Data 以及人工专家（经验丰富的机器人从业者）等基线进行比较。实验显示 GDC 在样本效率与部署成功率上均显著优于所有基线，平均提升约 25%（甚至超过 All‑Data），并在多任务、仿真与真实环境均保持优势。

**⚠️ 局限性**

局限性包括：①VLM 生成的任务可能不可行或缺乏创新，需人工审核；②归因与分配过程计算成本高，批次大小与更新频率需平衡；③当前方法对任务空间做离散化处理，未实现连续参数化归因，限制了探索精度。

---

## 785. From Autonomous Agents to Cognitive Symbionts: Toward AI-Augmented Human Intelligence

**arXiv ID:** 2611.05376 | [PDF](https://arxiv.org/pdf/2611.05376v1)

**作者:** Hao Fei `[一作]` `[通讯]` (University of Oxford), Hao Fei (University of Oxford)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出认知共生体（cognitive symbiont）概念，阐述人类与 AI 共生的第二类智能个体（AI‑augmented human intelligence），并给出系统设计框架与研究路线图。

**💡 创新点**

定义 AI‑augmented human intelligence 终局的两半，提出共生体的六个必要属性与 Jarvis 测试，构建共生循环、三时尺度记录、隐私授权与持续学习机制，为 AI 个人化提供全新框架。

**🔧 技术方法**

依托基础模型（多模态 LLM）、Agent harness、长期记忆与上下文管理、世界模型、可穿戴/神经接口、低功耗感知、实时双向交互等技术栈。

**📊 数据集**

本文为视角论文，无具体实验与数据集；仅引用现有研究与公开数据。

**📈 对比分析**

未给出实验比较或性能数值；提出可测量指标（认知缺口缩减、帮助质量、信任度等）及评估方法，但缺乏实测结果。

**⚠️ 局限性**

仅为概念性框架，缺乏实现细节与真实世界验证；对隐私、伦理、可解释性及监管挑战讨论有限；技术瓶颈（实时双向交互、长期学习、硬件续航等）仍待解决。

---

## 786. Visibility-Aware Reference Generation for Benchmarking Visual 3D Reconstruction in Vegetation-Rich Environments

**arXiv ID:** 2611.05361 | [PDF](https://arxiv.org/pdf/2611.05361v1)

**作者:** Zhipeng Zhong `[一作]` (Loughborough University), Cunjia Liu `[通讯]` (Loughborough University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于移动激光雷达的参考生成管线，用于在植被密集环境中评估视觉深度估计和3D重建方法。

**💡 创新点**

创新点在于引入大深度间隙可见度评分(LDGVS)和自动隐藏点去除(HPR)尺度选择，实现对稀疏激光点云的高质量可见性过滤，显著提升参考点云密度。

**🔧 技术方法**

技术包括多传感器数据时序对齐、点云去畸变/去噪、HPR、LDGVS、深度一致性过滤、稀疏深度图生成、基于Sim(3)的块级注册等。

**📊 数据集**

使用TartanAirV2、TartanGround、BotanicGarden等现有多传感器定位数据集，其中BotanicGarden用于构建基准。

**📈 对比分析**

将零射程前馈视觉几何模型与传统SfM进行对比，评估指标包括精度、完整度、Chamfer距离、绝对相对误差、有效深度密度等；结果显示前馈模型密度高但存在重复结构，SfM更连贯但稀疏且易失效。

**⚠️ 局限性**

局限在于对激光雷达采样稀疏和非均匀的处理仍依赖阈值，深度估计误差在极端植被遮挡下仍有提升空间；以及在大规模场景下对内存和计算开销的限制。

---

## 787. GSCV++: Lightweight 3D Gaussian Splatting Sequence Compression via Standard Video Codecs

**arXiv ID:** 2611.05359 | [PDF](https://arxiv.org/pdf/2611.05359v1)

**作者:** Qi Yang `[一作]` (Bytedance Inc.), Li Zhang `[通讯]` (Bytedance Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 GSCV++，一种基于标准视频编解码器的轻量级 3D 高斯 splatting（GS）序列压缩框架。

**💡 创新点**

创新点包括：①能量自适应 SH AC 维度约简（EA‑SHDR）实现跨帧稳定特征空间；②基于属性权重的 Inter‑MiniPLAS（P‑MiniPLAS）实现快速平滑 2D 映射；③内容感知的非均匀位深拆分与 CU 大小控制的通用压缩（CA‑GC），三者协同显著提升 RD 性能与处理时延。

**🔧 技术方法**

采用 PCA 维度约简、改进的 PLAS 权重推理与局部块重排、标准视频编码器（HEVC/HM18.0、VVC/VTM23.11、FFMPEG libx265/x264）以及 LZMA 等压缩工具。

**📊 数据集**

使用 MPEG GSC 工作组提供的 “bartender”、”breakfast“、”cinema“ 三个 GS 序列以及 AVS 工作组的 “dance_dunhuang_pair” 与 “basketball” 两个动态场景，包含跟踪、半跟踪与无跟踪版本。

**📈 对比分析**

与 GSCodec Studio、GPCC v1 以及前沿 GSCV 进行比较；在所有测试条件下 GSCV++ 在 intra 模式上相对 GSCV 提升 25%–40% RD，inter 模式提升 65%–110%；生成 2D 映射与编码/解码速度均快于对手，且 2D 映射更平滑，视频编解码器更友好。

**⚠️ 局限性**

局限性包括：①坐标 LSB 映射对标准视频编码器压缩效果差，需额外轻量压缩；②仅用能量阈值控制 SH AC 维度在某些序列上未能获得最优 RD；③P‑MiniPLAS 产生的 PLAS 图像与自然图像显著不同，细微不平滑会显著影响编码器性能。

---

## 788. Adversarial Speculative Reasoning Improves Long Horizon Agents

**arXiv ID:** 2611.05345 | [PDF](https://arxiv.org/pdf/2611.05345v1)

**作者:** Hongyin Luo `[一作]` (Subconscious Systems Technologies, Inc.), Jack O'Brien `[通讯]` (Subconscious Systems Technologies, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种对抗性推测推理（Adversarial Speculative Reasoning）方法，通过训练草稿模型在关键决策点被拒绝，使目标模型在最精确的位置采样重要决策，从而提升长时序代理的推理准确性。

**💡 创新点**

核心创新是将推测解码从纯粹的加速器转变为上下文精度分配器：利用草稿的拒绝行为将关键信息重写到目标模型的低噪声位置，显著提升长时序推理效果。

**🔧 技术方法**

技术手段包括：推测解码框架、草稿模型（DFlash2/ DSPARK）、直接偏好优化（DPO）训练草稿、量化感知训练（QAT）以保持FP8精度，以及利用目标隐藏状态构建训练数据。

**📊 数据集**

使用了两组前沿模型（GLM‑5.3‑nvfp4 与 DeepSeek‑V4.1‑Flash）以及两大长时序软件工程基准：DeepSWE（编程任务）和 NL2Repo‑Bench（代码生成），并通过这些任务的隐藏测试和单元测试评估性能。

**📈 对比分析**

与基线无偏草稿比较，经过对抗性训练的草稿在 DeepSWE 上将 GLM‑5.3 的通过率从 60.6% 提升至 67.6%（+7.0 点），在 DeepSeek‑V4.1‑Flash 上提升 4.0 点；在 NL2Repo‑Bench 上提升 1.4 点；速度损失仅约 2%。

**⚠️ 局限性**

局限性包括：只能在已知的决策点训练草稿，无法直接学习“正确”决策的语义；对草稿模型的改动仅占总参数极小比例，难以在更大模型上进一步扩展；目前仅在软件工程任务上验证，需要在更多长时序场景验证其普适性。

---

## 789. HARP: Training-Free Dual-Profile Agentic Communication for LLM-Based Recommendation

**arXiv ID:** 2611.05319 | [PDF](https://arxiv.org/pdf/2611.05319v1)

**作者:** Zhen Tao `[一作]` (Harbin Institute of Technology), Qingqiang Sun `[通讯]` (Great Bay University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种三步骤、训练无关的代理式推荐方法HARP，将用户与物品双向构建结构化档案、在检索阶段应用通道可分解的同质性门控，并在热身阶段进行一次反射式更新；

**💡 创新点**

创新点在于从代理通信视角提出四阶段模型（Build–Transmit–Understand–Sync），针对现有记忆增强代理推荐的三大通信瓶颈分别设计对称物品档案、检索侧同质性门控与一次性反射更新三大操作；

**🔧 技术方法**

核心技术包括：基于LLM的JSON结构化用户档案构建、规则抽取的对称物品档案、通道级同质性评分及门控、链式思维反射与保守门控；

**📊 数据集**

实验数据集覆盖四个公开推荐基准：Amazon Books、Goodreads、MovieTV 和 Yelp；

**📈 对比分析**

与传统协同过滤、GNN 与单向LLM代理推荐等三类基线对比，HARP 在 Hit@1 上提升 6.9%–24.0%（在最强基线上），且通过细粒度消融验证各操作对性能的贡献；

**⚠️ 局限性**

局限性包括：对单轮反射更新的依赖，可能在长序列或极度稀疏数据中效果有限；对门控参数设定较为手工；缺乏对多模态或跨域情境的评估。

---

## 790. MDSKILL: Learning to Internalize Skill Memory through Multi-View Distillation for Efficient Dynamic Agent Self-Evolution

**arXiv ID:** 2611.05313 | [PDF](https://arxiv.org/pdf/2611.05313v1)

**作者:** Dian Jin `[一作]` (Zhejiang University), Kun Kuang `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MDSKILL框架，利用多视角（状态定位、行动规划、执行控制）将技能记忆拆分并通过动态选择、跨视角蒸馏实现自我进化

**💡 创新点**

创新点在于将技能记忆按功能拆分为三视角，使用动态可靠性与优势度量筛选高质量技能，并在多视角轨迹与统一视角中进行正向、逆向及跨视角蒸馏，显著降低噪声干扰和功能耦合

**🔧 技术方法**

采用强化学习（GiGPO）、On‑Policy Self‑Distillation、动态贝塔分布评估、Thompson采样、Token‑level KL蒸馏、知识增益门控等技术

**📊 数据集**

在ALFWorld、WebShop和Search‑QA三大基准上进行实验

**📈 对比分析**

与RL、技能增补、传统自蒸馏基线比较；MDSKILL在ALFWorld和WebShop上均超过SOTA 7.1%–8.1%（成功率提升），并在搜索问答任务保持竞争力

**⚠️ 局限性**

受限于需要构建和维护多视角技能库，对大模型的依赖较强，且对技能选择和蒸馏门控参数敏感，未来需进一步简化和自动化

---

## 791. Low-Density 3D Point Cloud Classification

**arXiv ID:** 2611.05291 | [PDF](https://arxiv.org/pdf/2611.05291v1)

**作者:** Ahmed Baha Ben Jmaa `[一作]` (Paris Pantheon Assas University), Faten Chaieb `[通讯]` (Paris Pantheon Assas University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种针对低密度3D点云分类的网络LD-PointNet++，并通过在低密度场景下的实验验证其有效性。

**💡 创新点**

通过扩大球形邻域以提升感受野，专门针对低密度点云设计的网络结构，在低密度条件下保持高准确率。

**🔧 技术方法**

基于PointNet++的采样-分组-PointNet模块，并采用大半径球形邻域、交叉熵损失、Adam优化器等技术。

**📊 数据集**

使用ModelNet40和ModelNet10数据集，训练时使用1024点，测试时降至64点。

**📈 对比分析**

与PointNet++、Point-TnT、DGCNN、PointMLP等方法对比，在64点测试中实现90.1%整体准确率（相对其他方法提升约85%），并在不同采样比例下保持90%以上准确率。

**⚠️ 局限性**

在高密度点云场景下的准确率不如现有最优方法，且方法在极低密度下仍需更多优化以处理更稀疏的数据。

---

## 792. GSBR: Geometric Support Behavior Regularization for Online RL

**arXiv ID:** 2611.05295 | [PDF](https://arxiv.org/pdf/2611.05295v1)

**作者:** Charles Xu `[一作]` (University Of California Berkeley), Sergey Levine `[通讯]` (University Of California Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于几何支持行为正则化（GSBR）的在线强化学习框架，利用冻结的基准策略作为先验，允许演员在基准支持内自由选择和细化行为，并在支持外以距离惩罚形式平滑约束。

**💡 创新点**

核心创新在于：① 将基准策略的支持集以几何方式逼近为一系列半径相同的球体；② 通过距离惩罚实现既能剔除低价值模式又能在支持内进行局部改进；③ 给出了样本估计误差上界，证明随着采样数增大误差趋于零。

**🔧 技术方法**

采用基于actor‑critic的Q‑最大化目标，演员以状态和一组从基准策略采样的参考动作为输入；正则化项为距离到支持集合的平方；使用邻域半径自适应量化量化值，构造支持集合；训练流程包含离线经验回放和在线交互。

**📊 数据集**

在19个机器人任务上评估，任务来源于四大基准：RoboMimic、OGBench、OCBench 和 LIBERO。每个基准包含若干子任务，使用人类演示或基准数据集构建的基准策略（MLP 或 VLA）。

**📈 对比分析**

与 TD3+BC、RLT、Residual Policy、BFN、DSRL、EXPO‑FT、DICE‑RL 等七种在线RL基线进行比较。GSBR 在所有 19 任务的平均成功率达到 95%，比最佳基线高约 3%；归一化吞吐量最高，较下一最佳方法提升约 23%。

**⚠️ 局限性**

局限性包括：① 需要在测试时多次采样基准策略，增加计算开销与推理延迟；② 引入额外的分位数超参数 q，尽管表现对 q 较鲁棒但仍需调优；③ 目前仅在仿真环境验证，真实机器人测试尚待开展。

---

## 793. OpenBrain: An Auditable Generated-Label Release for Whole-Brain MRI Parcellation

**arXiv ID:** 2611.05348 | [PDF](https://arxiv.org/pdf/2611.05348v1)

**作者:** Qizhen Lan `[一作]` (University of Texas Health Science Center at Houston), Xiaoqian Jiang `[通讯]` (University of Texas Health Science Center at Houston)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

创建了OpenBrain，发布35,838个T1加权脑影像的全脑分割，并附带来源信息、委员会分割、QC‑7无参风险评估和置信层级；

**💡 创新点**

创新点在于把每个案例包装成可审计的记录，提供七项无参风险检查和分位数风险分数，允许下载后直接过滤、加权或重阈值，并通过冻结验证协议证明风险与分割质量的相关性；

**🔧 技术方法**

采用FastSurfer/QuickNAT/SynthSeg等自动分割模型生成标签，计算QC‑7七项检查（几何、图像-标签一致性、委员会一致性），将原始值转化为分位数风险并取平均得到R；

**📊 数据集**

利用607个OpenNeuro公开T1w数据集中的35,838个病例作为发布数据集；在IXI、BrainLat和Mindboggle‑101三个外部数据集上进行验证；训练时从OpenBrain选取1,000个案例进行nnU‑Net训练；

**📈 对比分析**

在保留最低风险K%（K=25%）的案例时，外部宏Dice平均提升1.3–3.7个百分点；在固定预算1,000案例训练中，低风险选取相较随机提升3.2–3.7个百分点，高风险选取表现更差；

**⚠️ 局限性**

局限性包括：未提供手工标注或临床级分割；风险字段不等同于准确性概率；对源数据覆盖不完全，缺乏公平性和临床质量保证；未能完全消除源组合对结果的影响；

---

## 794. PolicyIK: High-Precision Inverse Kinematics for Robot Arms with Reinforcement Learning

**arXiv ID:** 2611.05372 | [PDF](https://arxiv.org/pdf/2611.05372v1)

**作者:** Yuhan Liu `[一作]` (New York University), Shenji Wan `[通讯]` (New York University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研发并训练了PolicyIK，一种通过强化学习实现的高精度逆运动学控制器，能够在姿态误差低于1 mm、0.2°的目标下实现抓取和保持。

**💡 创新点**

引入了参考累积与命令状态反馈、双尺度姿态误差与误差相关动作增益，以及进度与驻留奖励，显著提升了终端精度和保持性能。

**🔧 技术方法**

目标条件强化学习（PPO/FlashSAC）、Hindsight Experience Replay、逆运动学理论、双尺度误差感知、误差相关增益、进度/驻留奖励设计等技术。

**📊 数据集**

在OpenSim仿真中随机生成的7‑DoF机械臂起始–目标对（≈4,120对）以及在物理OpenArm和xArm7上收集的训练与测试样本；无标注的抓取/放置任务（LIBERO‑Object）。

**📈 对比分析**

与经典DLS、QP、MPC以及UMI‑RL基线对比，单次尝试95.5 %任务满足1 mm/0.2°，部署协议达98.9 %，硬件上71.7 %保持成功，明显优于所有传统控制器。

**⚠️ 局限性**

仍需在更复杂环境下验证；对极限关节、非刚性关节或多臂协作的适应性未知；训练需要大量数据或经验回放，成本较高。

---

