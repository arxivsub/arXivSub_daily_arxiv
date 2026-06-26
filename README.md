# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-06-26 | 今日论文总数: 541

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Dustin: Draft-Augmented Sparse Verification for Efficient Long-Context Generation with Speculative Decoding

**arXiv ID:** 2606.24957 | [PDF](https://arxiv.org/pdf/2606.24957v1)

**作者:** WenHung Lee `[一作]` (National Yang Ming Chiao Tung University), Kai-Chiang Wu `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Dustin的稀疏验证框架，旨在提高长上下文推理中的效率，特别是在使用投机解码时。

**💡 创新点**

Dustin通过结合草稿模型的前瞻信号和目标模型的历史注意力，识别出重要的关键字，显著提高了推理速度，同时保持了较高的准确性。

**🔧 技术方法**

使用了稀疏估计方案和混合注意力聚合技术，以减少计算延迟并提高验证效率。

**📊 数据集**

在PG-19和LongBench数据集上进行了评估，使用了Qwen2.5-72B和Llama3模型系列。

**📈 对比分析**

与传统的投机解码方法相比，Dustin在自注意力计算上实现了27.85倍的加速，在端到端解码速度上实现了9.17倍的加速，且准确性几乎没有下降。

**⚠️ 局限性**

局限性在于预算分配参数是通过离线分析固定的，未来可以考虑动态方案以进一步提高验证的准确性；此外，Dustin并未减少KV缓存的内存占用，因此在固定GPU内存预算下无法处理更长的序列。

---

## 2. Training Dynamics of Neural Software Defect Predictors under Coupled Data-Quality Issues

**arXiv ID:** 2606.24968 | [PDF](https://arxiv.org/pdf/2606.24968v1)

**作者:** Emmanuel Charleson Dapaah `[一作]` (University of Goettingen), Jens Grabowski `[通讯]` (University of Goettingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在软件缺陷预测中，类不平衡和类重叠这两种数据质量问题在深度学习训练过程中的相互作用，并提出了交互感知的干预实验协议，用于挖掘并分类不同训练动态模式。

**💡 创新点**

创新点在于：①首次将类不平衡与类重叠的耦合效应纳入训练动态分析；②设计了受控注入与递进减少双重干预，并对非目标指标漂移进行监测；③提出了基于效果量、方向一致性和种子稳定性的训练动态模式分类法（区分单独、共享、交互及掩盖/主导模式）。

**🔧 技术方法**

使用了受控注入与递进减少的实验设计，固定的多层感知机（128-64-32）与 Adam 优化；在每个 epoch 记录训练/验证误差、梯度幅度、梯度分布形状、参数统计，并通过标准化差异和效应量评估模式；采用 Spearman 相关和耦合比例评估干预的忠实度与漂移。

**📊 数据集**

主要使用统一缺陷数据集（Unified Bug Dataset, UBD）的类级版本，共约 40-50 个 Java 项目的训练集；所有数据都经过标准化、缺失值中位数填补和 Boolean 编码。

**📈 对比分析**

通过对比不同干预条件（仅不平衡、仅重叠、耦合以及递进减少）的训练动态，验证模式的一致性与方向性；在终点指标上，训练动态能更早揭示模型对不平衡/重叠的敏感性，提供比单纯 AUC/F1 更细粒度的诊断信息；性能差异主要体现在训练误差/验证误差的波动、梯度梯度传播不均和参数分布变化。

**⚠️ 局限性**

主要局限包括：①干预是人为控制的压力测试，可能与真实缺陷数据的噪声或标签误差不完全一致；②仅使用单一 MLP 架构，无法验证结果对其他深度模型（CNN、Transformer 等）的泛化；③类重叠注入会产生标签-特征张力，可能混淆真实的边界模糊效应；④对小样本或高维数据的 N1 计算可能不稳定；⑤实验对硬件/软件版本敏感，需严格复现。

---

## 3. Enhancing Clinician Decision-Making via Uncertainty-Aware Multi-Expert Fusion for Stroke Rehabilitation

**arXiv ID:** 2606.24960 | [PDF](https://arxiv.org/pdf/2606.24960v1)

**作者:** Tamim Ahmed `[一作]` (University of Southern California), Thanassis Rikakis `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种多专家融合、基于不确定性的评估引擎xAARA，用多视角视频自动评估中风患者ARAT得分，并提供任务、动作阶段和动作质量层面的解释与置信度；

**💡 创新点**

1) 将临床评估分解为层次化结构（EAGM），显式建模专家主观性；2) 采用动态贝叶斯网络与熵门控的产品专家融合，实现可信不确定性量化和自适应推理；3) 通过协同设计的多层标注显著降低标注熵，从而提升模型准确性；

**🔧 技术方法**

多模态特征提取（Skeleton CTR‑GCN、VideoMAE、手工运动学）、Segment‑Attention Fusion Transformer (SAFT)、动态贝叶斯网络、熵门控产品专家融合、三阶门控资格判定；

**📊 数据集**

对105名中风幸存者进行ARAT实验，使用三摄像头同步录制，共788次练习（88名完整标注），以及4名外部临床医师的评估验证；

**📈 对比分析**

与单一临床评分对比，xAARA任务级准确率94.2%（κ=0.934），动作阶段准确率81.3%（κ=0.727），预测不确定度下降96.1%；在专家主观分歧场景中，模型对至少一位专家的评分保持100%匹配；

**⚠️ 局限性**

仅为单次会话、单中心回顾性研究；缺乏多中心、多时段验证；动作质量细节在模型与人工判定上仍存在可靠性挑战；

---

## 4. From Meta Idea to Advanced Mathematical Discovery -- Human-AI Co-Discovery of Sign-Embedding Quantum Algorithms

**arXiv ID:** 2606.24899 | [PDF](https://arxiv.org/pdf/2606.24899v1)

**作者:** Yanqiao Wang `[一作]`, Yang Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并演示了一个人机协作的研究工作流，在此工作流中，AI系统（AIM）通过路线扩展、连接发现和理论推导，协助构建基于符号嵌入（sign-embedding）的量子算法框架，并由人工负责价值判断与复杂度审核，最终形成可审计的定理族。

**💡 创新点**

创新点在于：①将AI应用于问题形成阶段而非仅限于已定问题的求解；②构建人机共创的审计机制（路线筛选、复杂度校核、假设检验）；③通过符号嵌入与有理逼近的组合，实现对 Sylvester 方程、平方根、几何均值、Ricatti 方程等矩阵问题的统一量子实现，并显著降低查询复杂度。

**🔧 技术方法**

采用的技术包括：AIM（agentic AI-mathematician）自然语言交互与推理，符号数学推导（矩阵符号嵌入、分式逼近、复合块编码），复杂度审核清单与假设审查，人工审核与反驳式审阅。

**📊 数据集**

未使用传统数据集；研究以理论推导和算法分析为主。

**📈 对比分析**

对比方法主要是与传统的直接增广矩阵求逆或基于对数-正弦逼近的实现进行比较，表现为查询复杂度从 O(μ⁻²) 降至 O(μ⁻¹)（带对数因子）并在块编码归一化上实现更优的 O(μ⁻¹/²) 等改进；未给出实验量化结果。

**⚠️ 局限性**

局限性包括：①AI生成的证明和复杂度估计需要人工审核，无法完全自动化；②缺乏正式的形式化验证，可能隐藏未显式假设或不当不等式；③工作仅为单一案例研究，缺乏统计评估；④在量子实现细节上仍需进一步验证。

---

## 5. The Hitchhiker's Guide to Agentic AI: From Foundations to Systems

**arXiv ID:** 2606.24937 | [PDF](https://arxiv.org/pdf/2606.24937v1)

**作者:** Haggai Roitman `[一作]` `[通讯]` (Technion Israel Institute Of Technology), Haggai Roitman (Technion Israel Institute Of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文是一份完整的技术手册，系统性梳理了从基础 Transformer 架构、训练与系统优化、强化学习与对齐、推理能力、评估方法，到完整的代理（Agent）设计与部署的全部知识。它把现有的论文、博客、开源实现与工程实践融合为一套连贯的学习与实践路线图。

**💡 创新点**

创新点在于（1）将分散在数百篇论文与社区资源的技术点统一成一个可操作的、以实现为导向的指南；（2）强调理论与实现细节并重，提供大量代码片段、超参表与硬件预算；（3）把代理系统的各个层面（RAG、记忆、工具调用、A2A 通信、MCP、调度框架）整合进一体化框架，形成完整的“从模型到部署”闭环。

**🔧 技术方法**

主要技术包括 Transformer 自注意力、RoPE、ALiBi、Flash Attention、Mixture‑of‑Experts、LoRA/QLoRA、量化与知识蒸馏、RLHF、DPO、GRPO、KTO/IPO/ORPO、SFT、MCP、A2A、LangGraph/AutoGen 等；同时覆盖了分布式训练技术（FSDP、DeepSpeed ZeRO、tensor/pipeline parallelism）以及推理加速技术（vLLM、KV 缓存压缩）。

**📊 数据集**

指南本身并未在单一实验上使用数据集，而是引用了训练与评估中常用的公开数据来源：互联网上的海量文本（WebText、CC-Data）、指令对齐数据（OpenAI instruction data、Anthropic instruction set）、RL 对齐奖励模型数据（HumanEval、OpenAI API benchmark）、检索评估数据集（MS MARCO、Wikipedia 等）以及代理评测环境（WebArena、SWE‑bench、OSWorld、GAIA）。

**📈 对比分析**

由于是综述与实践手册，未给出统一的实验对比；文中对比了多种技术在不同场景下的相对优缺点（如 Flash Attention 与 sparse attention、LoRA 与全参数微调、PPO 与 DPO、MCP 与自定义工具调用）。性能讨论多聚焦在工业部署经验（如 8‑B Llama‑3 在 A100‑H100 集群上单卡训练时间、vLLM 在 64‑B 模型下的吞吐率等），但并未给出统一指标。

**⚠️ 局限性**

局限性包括：① 只覆盖文本 LLM 与代理体系，省略多模态、专用领域与个性化等方向；② 侧重工业化实现，缺乏系统化的实验验证与可复现性；③ 由于技术快速演进，部分细节（如最新的 LLaMA‑3.1、Gemini‑1.5 长上下文扩展）可能已被后续研究更新；④ 作为“教育资源”，对读者的前置知识要求相对较高，初学者可能需要额外学习深度学习、RL 与分布式系统基础。

---

## 6. SPORT: Spherical-PSNR-Optimized tRuncaTion for Power-Efficient 360-Degree Video Systems

**arXiv ID:** 2606.24916 | [PDF](https://arxiv.org/pdf/2606.24916v1)

**作者:** Md. Sajjad Hossain `[一作]` (University of Alabama), Na Gong `[通讯]` (University of Alabama)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种基于视角感知的 360° 视频内存位 truncation 框架 SPORT，利用 WS‑PSNR 作为优化约束，在显存写入阶段对不在用户视野的像素进行按需截断，从而显著降低内存功耗和带宽消耗。

**💡 创新点**

创新点在于：① 引入 WS‑PSNR 一致的截断理论，解决传统 PSNR 与 360° 视频评价指标不匹配的问题；② 通过 9.33 ms 预测头部运动实现边界误判率下降 5.2 pp；③ 设计异构三-bank TrunMEM360 SRAM，支持不同区域的动态截断和功耗门控；④ 在 130 nm ASIC 上实现硬件验证，并证明实现与软件模型 0.1 dB WS‑PSNR、0.001 SSIM 完全一致；⑤ 系统整体 9.33 ms 延迟，满足 20 ms VR 舒适阈值。

**🔧 技术方法**

使用的技术包括 WS‑PSNR 加权 MSE 计算、最优 dummy pattern 理论、基于 head‑tracker 的线性预测与区域划分、TrunMEM 级联功耗门控 SRAM、CACTI 对 DRAM 能耗的仿真、以及 SSIM 结构相似度评估。

**📊 数据集**

实验采用 Wu 等人收集的 48 位受试者 76 条全景视频的头部轨迹数据，主要以 4K 60 fps 3840×2160 的测试序列为基准，并在 5 条 4K 30 fps YouTube 视频上进行泛化验证。

**📈 对比分析**

与无截断、统一截断、固定 heuristic、PSNR‑based optimizer、SPORT‑A 等基线相比，SPORT‑B 在保持 FoV 区域 SSIM=1 的前提下实现 47.9% 内存功耗与带宽下降；SPORT‑A 进一步提升至 51.6%；系统总延迟 9.33 ms，低于 20 ms VR 预算，且软硬件协同实现与模型误差小于 0.1 dB。

**⚠️ 局限性**

局限性包括：仍需完整 TrunMEM360 ASIC 的大规模量产验证；对极快头部运动（>20°/s）预测误差仍可能导致边界误判；依赖高频（1000 Hz）IMU 采样，低频时可能失效；目前仅在 360° VR 端验证，其他视频/图像系统的可迁移性尚待探索。

---

## 7. How a computer might think

**arXiv ID:** 2606.24927 | [PDF](https://arxiv.org/pdf/2606.24927v1)

**作者:** Sankha S. Basu `[一作]` `[通讯]` (Indraprastha Institute of Information Technology-Delhi), Sankha S. Basu (Indraprastha Institute of Information Technology-Delhi)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了两种五值逻辑（UKN1与UKN2），将“未知”和“不可知”作为独立的语义状态，并对其与已知四值逻辑FDE及其扩展FDEe进行对比与分析。

**💡 创新点**

创新点在于：①将不可知状态区分为可开放（not known yet）和不可知（unknowable）两种闭合状态；②引入三元可指派值（true、both、unknowable）而非传统的两元；③对两种逻辑的算子真值表和线性规则进行系统构造，揭示其在悖论容忍、分配律等方面的独特性质。

**🔧 技术方法**

使用了多值逻辑框架中的语义矩阵（logical matrix）和满足关系，结合真值表、线性规则与模型论方法对两种逻辑进行定义与性质证明。

**📊 数据集**

未使用任何外部数据集，论文为纯理论研究。

**📈 对比分析**

未涉及实验或性能比较，主要通过逻辑推理与真值表演算来验证有效性。

**⚠️ 局限性**

局限性包括：缺乏形式化的证明系统（如Hilbert或自然演绎系统）的完整性与健全性证明；对不可知状态的实际应用场景与解释仍待进一步探讨。

---

## 8. Learning Dynamical Systems from Multiple Sparse Datasets: A Hierarchical Bayesian Modeling Approach

**arXiv ID:** 2606.24966 | [PDF](https://arxiv.org/pdf/2606.24966v1)

**作者:** Cristian Brugnara `[一作]` (SUPSI), Laura Azzimonti `[通讯]` (SUPSI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种层次贝叶斯框架，用于在稀疏、噪声和不规则采样数据中进行动态系统的参数估计，利用多个相关数据集提供额外信息。

**💡 创新点**

创新点在于通过层次贝叶斯建模实现概率元学习，模型化数据集特定参数为共享总体分布的抽样，并嵌入数值ODE求解器以提高后验推断的效率。

**🔧 技术方法**

使用了层次贝叶斯建模和基于梯度的马尔可夫链蒙特卡洛（MCMC）方法，结合可微分的ODE求解器。

**📊 数据集**

使用了Lotka-Volterra捕食者-猎物模型作为基准，生成了多个相关的动态系统数据集。

**📈 对比分析**

与未合并的方法相比，实验结果显示该方法在稀疏、噪声和不规则采样的轨迹上显著提高了可识别性和预测准确性，RMSE显著低于未合并贝叶斯模型和非线性最小二乘法。

**⚠️ 局限性**

当前的HBM实现要求参数到观测的映射必须是线性的或至少是可用的封闭形式，这限制了其适用性。

---

## 9. Conformal Orbit-Valid Trust Horizons for Equivariant World Models

**arXiv ID:** 2606.24946 | [PDF](https://arxiv.org/pdf/2606.24946v1)

**作者:** Hongbo Wang `[一作]` `[通讯]` (Stony Brook University), Hongbo Wang (Stony Brook University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种针对等变世界模型的分层置信度轨迹（conformal orbit‑valid trust horizons）方法，用来在有限样本下给出可可信的多步预测时间窗。

**💡 创新点**

创新点在于：① 通过一次性对整个误差曲线使用乘法式分层分位数校准，使得保证在一次抽样下成立；② 证明等变性可把在一个基准区域得到的校准曲线无额外代价地“传输”到整个群轨道（orbit‑transport theorem）；③ 在实验中展示了校准后无逆保守违规、非空闲、实现误差传输残差极低。

**🔧 技术方法**

核心技术包括：分层 conformal 校准（split‑conformal 乘法因子）、等变编码器与预测器（C_N‑steerable / Vector‑Neuron/SE(3) 网络）、Benettin 估计得到的扩展率 λ̂₁、以及基于误差曲线的安全边界门。

**📊 数据集**

使用的实验数据集有：PushT（2D 像素推送任务，含可调动力学参数）和 ManiSkill（3D 点云操作任务，如 PickCube、PushCube）。

**📈 对比分析**

与传统基准对比：在 50 组稳定审核中无逆保守违规，95% 上界为 5.8%；置信度与实际可测时间窗比值中位数为 0.67；等变模型的轨道传输残差中位数 1.1%；在对称 2D 环境下即使非等变模型也能从单个校准区域得到全轨道有效证书，而在 3D yaw 环境中只有等变模型能实现一阶安全无保守的证书。

**⚠️ 局限性**

局限性包括：① 该证书仅是分布式、单向安全保证，不能替代全局可达性；② 在高方差的 3D 规划实验中，证书驱动的子目标间隔并未显著提升规划性能；③ 等变性对校准成本的节省依赖子域几何，非等变模型在某些子域仍需额外校准；④ 对大步长或扩展动态的适用性尚未验证。

---

## 10. Evidence for feature-specific error correction in LLMs

**arXiv ID:** 2606.24964 | [PDF](https://arxiv.org/pdf/2606.24964v1)

**作者:** Francisco Ferreira da Silva `[一作]` (Pivotal Research), Stefan Heimersheim `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在LLM的残差流上做定向扰动，并测量不同方向的平稳区间破裂角度，评估模型对特征方向的错误修正能力（FSEC）

**💡 创新点**

提出并量化“特征特异性错误修正”指标——超椭圆指数p>2，用来检测模型是否对特征方向进行特殊处理；在多模型、多设置下验证其普适性

**🔧 技术方法**

残差流扰动、归一化角度扰动、平稳区间破裂角度计算、超椭圆拟合（p指数）以及对比基线方向（随机、PCA、随机差分）

**📊 数据集**

FineWeb 5-token 输入、PCA 10000条数据、对比概念对（性别、财富等）构造的对比方向、MELBO 和 SAE 训练得到的特征方向，以及六大模型（Gemma‑2‑9B、Qwen3‑1.7B、Llama‑3.1‑8B、Mistral‑7B‑v0.3、Aya‑Expanse‑8B、Yi‑1.5‑9B）

**📈 对比分析**

在每个模型和设置下，对特征方向与基线方向的p指数进行比较；特征方向平均p≈2.3、MELBO/SAE略低，基线接近p≈2；通过旋转实验验证p随与真实特征方向的相似度降低而趋于2；在玩具去噪网络中得到p≈3‑3.4，进一步验证方法有效

**⚠️ 局限性**

1) 对比方向并非真实特征，可能导致p下估；2) 仅检测对方向的特殊性，无法确认其是否真正是功能性特征；3) 归因于激活平台的“平稳区间”可能不是错误修正的唯一解释；4) 玩具模型与LLM的差异较大，验证结果受限

---

## 11. Towards Continuous Power Forecasting: Practical Continual Learning for Real-World Energy Systems in Nonstationary Time Series

**arXiv ID:** 2606.24955 | [PDF](https://arxiv.org/pdf/2606.24955v1)

**作者:** Yujiang He `[一作]` (University of Kassel), Bernhard Sick `[通讯]` (University of Kassel)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了连续电力预测（CPF）范式，并在CLeaR框架下对六种持续学习方法（在线EWC、随机重放、最近重放、递减重放、生成重放、熟悉度重放）进行系统评估，

**💡 创新点**

创新点在于将持续学习方法从理论迁移至真实电力预测场景，阐释不同机制在非平稳条件下的稳定‑可塑性权衡，并给出针对工业运营约束的实践指引，

**🔧 技术方法**

技术上采用自编码器+多层感知器基础网络，结合阈值驱动的 novelty detection 与上述六种持续学习策略，

**📊 数据集**

使用真实区域电网23个月、15分钟间隔的数据集，包含95个电力实体与13维气象变量，

**📈 对比分析**

通过与静态基线（Baseline_L / Baseline_U）对比，计算拟合误差、预测误差、遗忘比例等指标，实验表明生成重放在多数指标上最优，随机重放亦能保持较好稳定性，

**⚠️ 局限性**

局限性在于对不同非平稳特征类型的适配性不足，阈值设置对性能影响大，且对长周期预测及人工干预机制的验证仍待进一步研究。

---

## 12. How Complexity Contributes to Learning Opacity in Machine Learning

**arXiv ID:** 2606.24953 | [PDF](https://arxiv.org/pdf/2606.24953v1)

**作者:** Joachim Stein `[一作]` (Heidelberger Akademie der Wissenschaften), Eric Raidl `[通讯]` (Universität Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文研究了神经网络学习过程中的学习不透明性，认为其本质上是一个复杂的动态系统，探讨了学习不透明性的来源及其对理解学习过程的影响。

**💡 创新点**

创新点在于将神经网络学习视为复杂系统，识别出影响学习不透明性的三个关键特性：对权重初始化的敏感性、梯度优化中的反馈以及对训练数据的敏感性。

**🔧 技术方法**

使用了复杂性科学的分析方法，结合动态建模来探讨神经网络的学习过程。

**📊 数据集**

未具体提及使用的数据集，但讨论了训练数据的分布和顺序对学习动态的影响。

**📈 对比分析**

通过分析复杂系统的特性，指出反馈机制使得学习过程的理论分析变得困难，导致研究者依赖经验方法，进而引入统计推断的固有局限性。

**⚠️ 局限性**

学习不透明性并非仅仅由于访问限制或认知能力不足，而是由于机器学习的内在结构和复杂性特征，某些不透明性的来源可能是不可减少的。

---

## 13. A Spectral Phase Diagram for Binary Few-Shot Classification: Intrinsic Dimensionality, Geometric Saturation, and Representational Diagnosis

**arXiv ID:** 2606.24903 | [PDF](https://arxiv.org/pdf/2606.24903v1)

**作者:** Arnav Gupta `[一作]` `[通讯]` (Independent Researcher), Arnav Gupta (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究如何通过支持集的谱几何特征预测何时停止收集标注样本，避免无意义的标注成本。

**💡 创新点**

提出饱和指数(K) = 有效秩(Σ̂_W)/K，首次利用支持集的有效秩与样本数的比值作为无标签、无分类器的前瞻性饱和预测指标，并在理论上与高维协方差估计的收敛性质相联系。

**🔧 技术方法**

使用有效秩计算、谱分析、线性判别（LDA/逻辑回归）以及高维协方差估计理论；同时设计了基于阈值τ=0.3的停止算法。

**📊 数据集**

在十七个二分类任务上验证，覆盖 MNIST、Fashion‑MNIST、Kuzushiji‑MNIST、USPS、CIFAR‑10（原像素空间）和乳腺癌临床表型等六个数据集。

**📈 对比分析**

与任务内与任务间 Spearman 相关性对比；16/17 任务内部相关性中位 ρ≈0.81；池化相关性 ρ≈0.55；停止规则的 AUC 为 0.752；三相结构表明在探索期（K>1）平均增益约 3.5%，过渡期约 2.4%，饱和期约 0.8%。

**⚠️ 局限性**

仅适用于二分类固定线性模型；对非线性或预训练表征、N‑way 分类仍需扩展；CIFAR 等高维图像任务显示表征不足导致饱和指数低但精度不升高；对大规模或多任务情境的泛化还未验证。

---

## 14. Type Checking Project Haystack Grids using JSON Schema and Pydantic

**arXiv ID:** 2606.24891 | [PDF](https://arxiv.org/pdf/2606.24891v1)

**作者:** Thomas Hirsch `[一作]` (TU Wien), Gerald Schweiger `[通讯]` (TU Wien)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一个基于Python的工具链，能够解析Project Haystack的Trio文件，重新构建其归一化过程，并生成对应的Pydantic模型与JSON Schema，实现了在Python环境下的静态类型检查与结构化验证。

**💡 创新点**

创新点在于首次提供了非Fantom语言的完整Haystack解析与验证工具，利用Pydantic与JSON Schema实现跨语言验证，打破了原有对Haxall/Fantom生态的依赖，解决了标签歧义、缺失自动验证等生态瓶颈。

**🔧 技术方法**

技术手段包括：Python Lark解析器用于Trio文件的语法解析；自研的归一化与继承树构建逻辑；基于Pydantic的模型生成与数据校验；JSON Schema导出以支持其他语言的验证；以及对官方归一化JSON数据的直接处理。

**📊 数据集**

使用了Project Haystack 4.x 版本的全部定义文件（Trio、Zinc、JSON、JSON-LD、TTL等）以及官方发布的归一化定义集，作为语义模型的输入数据集。

**📈 对比分析**

通过将自实现的归一化结果与Haxall的DefCompiler输出进行对比，验证两者在相同输入下得到一致的schema；随后在Python中对实际Haystack网格进行校验，示例显示能够捕捉到缺失或不一致的定义，性能表现良好，验证时间仅为数百毫秒。

**⚠️ 局限性**

局限性包括：目前仅支持Haystack 4.x，尚未覆盖正在开发的Xeto格式和未来的5版；工具无法完全覆盖所有语义验证（如上下文相关的规则）；对非结构化文本的规范解释仍需人工介入，且跨语言的完整互操作性仍需进一步完善。

---

## 15. Towards Scalable Multi-Task Reinforcement Learning with Large Decision Models

**arXiv ID:** 2606.24962 | [PDF](https://arxiv.org/pdf/2606.24962v1)

**作者:** Thibaut Kulak `[一作]` `[通讯]` (NeoInstinct SA), Thibaut Kulak (NeoInstinct SA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并训练了一种名为LDM‑v0的大型变压器决策模型，在数千个多模态强化学习环境上进行离线预训练。

**💡 创新点**

创新在于将自动化参考策略监督与统一的多模态变压器架构相结合，实现在大规模异构环境中的单一共享策略。

**🔧 技术方法**

使用了基于Llama的解码器变压器、离线强化学习数据生成管线、自动化算法选择、跨模态编码与离散化技术。

**📊 数据集**

采用了约3,000个公开Gym/Gymnasium兼容环境，收集了约9.3 B条轨迹，构成了约4,000个高性能参考策略的数据集。

**📈 对比分析**

通过在训练环境上与任务特定参考策略对比，LDM‑v0在约1,600个环境中达到≥80%参考性能，在约1,000个环境中匹配参考性能，模型规模越大性能越好。

**⚠️ 局限性**

局限包括仅评估训练分布内环境、数据集覆盖仅为环境池的19%以及对未见环境的泛化能力尚未验证。

---

## 16. EmotionAI: A Privacy-Preserving Computational Intelligence Pipeline for Speech-Emotion-Grounded Conversational Analysis

**arXiv ID:** 2606.24941 | [PDF](https://arxiv.org/pdf/2606.24941v1)

**作者:** Wai Laam Mak `[一作]` (Nottingham Trent University), Pedro Machado `[通讯]` (Nottingham Trent University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了 EmotionAI，整合分说者、Whisper ASR、wav2vec2‑large‑superb‑er 语音情感识别以及三模型本地 LLM 面板，实现可审计、隐私保护的时间戳绑定情感驱动对话分析。

**💡 创新点**

创新点在于将离线情感检测与本地生成式推理相结合，通过情感元数据作为提示实现可追踪、引用约束的分析，首次在同一系统中实现情感分类与多模型 LLM 的协同推理。

**🔧 技术方法**

采用 Pyannote 3.1/WeSpeaker 进行说话人分离，Whisper medium 进行语音转写，wav2vec2‑large‑superb‑er 进行情感分类，Ollama 托管的 Llama 3.2:3B、Qwen 2.5:3B 与 Gemma 3:4B 三模型对抗面板，并使用 Russell 价值‑唤醒理论生成情感指标。

**📊 数据集**

主要使用 RAVDESS 四类子集（672 句）评估 SER，并在一段 117.9 秒的两说话者录音上进行 Q&A 评测，所有模型均使用公开预训练权重。

**📈 对比分析**

与八种 SER 方法（随机、majority、MFCC+LogReg、ESN、wav2vec2-base/large）对比，零样本情感识别准确率仅 48.8%，低于 MFCC 71%；在缺少情感证据时，LLM Q&A 拒答率从 8% 上升至 67%，显示情感 grounding 的必要性；端到端 CPU 运行时长约 157 s（RTF≈1.33），主要瓶颈为 Whisper ASR。

**⚠️ 局限性**

局限性包括跨语料转移性能差（尤其 Sad 类）、仅使用英语演绎语料、缺乏真实场景验证、模型体积大导致 CPU 延迟、未对口音/性别偏差进行评估，以及仅进行小规模 Q&A 验证。

---

## 17. ReviewGuard: Aligning LLM-Assisted Peer Review with Long-Term Scientific Impact

**arXiv ID:** 2606.24892 | [PDF](https://arxiv.org/pdf/2606.24892v1)

**作者:** Abdur Rasool `[一作]` (Southern University of Science and Technology), Linyi Yang `[通讯]` (Southern University of Science and Technology)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ReviewGuard 两阶段框架，用 LLM 自动生成同行评审并通过引用量奖励学习预测论文长远影响；

**💡 创新点**

首次将引用量作为奖励引入 GRPO 强化学习，对 LLM 的评分与未来被引量对齐，并构建拒稿后最终发表论文的影响预测基准；

**🔧 技术方法**

使用 Qwen2‑7B‑Instruct 作为基模型，先用 LoRA 进行监督微调得到 Expert 模型，再利用 Group Relative Policy Optimization (GRPO) 与引用归一化奖励进行强化学习；

**📊 数据集**

基于 20,861 篇 AI/ML 论文的 OpenReview 评审数据，并使用 Semantic Scholar 提供的引用计数，构造拒稿后被发表、Top‑1,000 高被引子集；

**📈 对比分析**

与人类评审、Expert (LoRA 7B) 以及多种专业与前沿同行评审模型进行比较；Spearman 相关率从人类 0.492 提升到 Expert 0.681，再提升至 ReviewGuard 0.776；拯救高影响拒稿率从 1.8% 提升到 10.2%（5.6×）；在多项评估指标上多项领先；

**⚠️ 局限性**

仅基于回顾性引用，引用噪声大且各领域差异显著；数据局限于顶级 AI/ML 会议；拒稿-后发表匹配存在误差；计算成本高，GRPO 训练资源消耗大；未来部署需寻找替代或预测代理。

---

## 18. AgentOdyssey: Open-Ended Long-Horizon Text Game Generation for Test-Time Continual Learning Agents

**arXiv ID:** 2606.24893 | [PDF](https://arxiv.org/pdf/2606.24893v1)

**作者:** Zheyuan Zhang `[一作]`, Tianmin Shu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个基于LLM的开放式文本游戏生成框架，用来评估在测试时持续学习（test‑time continual learning）环境中的代理，并设计了一套多维诊断指标来衡量代理的五大核心能力（探索、世界知识、情景记忆、动作多样性、模型成本）。

**💡 创新点**

创新点在于：①首次将持续学习问题迁移到长时程、不可重置的文本游戏中；②通过程序合成+LLM自动生成无限多样的游戏实体、规则和任务；③构建了诊断式评估体系，能分离游戏进度与代理的认知与记忆表现；④系统性比较多种记忆与学习机制（长上下文、固定大小记忆、RAG、SFT、RL、潜在记忆）。

**🔧 技术方法**

核心技术包括：LLM驱动的游戏生成与程序合成；POMDP 形式化与基于规则的世界动态；多模态（文本）观察与奖励设计；不同代理范式的实现（ReAct、RAG、LoRA SFT、RL‑PPO、固定窗口记忆等）；诊断测试（World Knowledge QA、Episodic Memory QA、探索率、动作熵）。

**📊 数据集**

数据集：使用自研的可无限扩展文本游戏集合，基于一个手工设计的基准游戏通过LLM合成扩充；对比时还使用了公开文本游戏环境（Voyager、Jericho、ByteSized32、Minecraft‑Text）作为基准。

**📈 对比分析**

实验比较方法：在同一游戏轨迹上跑不同代理，记录游戏进度（主线奖励、辅助奖励）、诊断指标、累计token数。结果显示：长上下文代理在大模型下性能最高，但仍远低于人类；短期记忆显著提升SFT和RAG的表现；所有方法在探索率、动作多样性和情景记忆上都存在瓶颈，模型成本随步骤呈二次增长。

**⚠️ 局限性**

局限性：仅支持文本观察和单一代理；基于离散动作的回合制设计，忽略了真实世界的视觉、时间连续性和多代理交互；LLM生成的游戏虽多样但仍可能受训练数据泄漏影响；当前评估缺乏对灾难性遗忘和主动探索策略的深入研究。

---

## 19. What Do Language Priors Contribute to Darcy-Flow Inversion? A Mechanistic Audit

**arXiv ID:** 2606.24967 | [PDF](https://arxiv.org/pdf/2606.24967v1)

**作者:** Taiga Saito `[一作]` (Tohoku University), Sopheakpolin Mom `[通讯]` (Tohoku University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究测试了句子嵌入是否可以作为推理时接口，将地质描述注入到学习的达西流逆解算器中，以改善在不适定逆问题中的解。

**💡 创新点**

创新点在于使用自然语言作为软先验，探讨其在逆解算器中的作用，尤其是在数据稀疏和观测不确定的情况下。

**🔧 技术方法**

使用了句子嵌入技术，结合了U-Net编码器-解码器架构和特征线性调制（FiLM）来实现文本条件化。

**📊 数据集**

使用了合成达西流数据集，包含六种地质类别，以及SPE10基准模型作为外部验证。

**📈 对比分析**

与传统的无正则化、Tikhonov和总变差反演方法相比，使用文本条件化的生成器在K-MSE上表现出显著的改进，平均K-MSE从0.0869降至0.0168，提升了81%。

**⚠️ 局限性**

限制在于物理设置理想化，结果基于64×64网格的归一化导水率，且未考虑噪声影响下的真实场景。

---

## 20. What Does a Pathological Speech Assessment Model Know about Acoustic Features? A Case Study on Oral and Oropharyngeal Cancer Patients

**arXiv ID:** 2606.24949 | [PDF](https://arxiv.org/pdf/2606.24949v1)

**作者:** Tuan Nguyen `[一作]` (Avignon University), Virginie Woisard `[通讯]` (Hôpital Larrey)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `57a58b01-81b4-4d75-a45c-2e891f272b50` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对基于Wav2Vec 2.0 的语音可懂度评估模型进行可解释性分析，使用 eGeMAPS LLD 与 PWCCA 在层级上测量模型表示与可解释特征的相关性；

**💡 创新点**

提出将 PWCCA 与 eGeMAPS 结合，桥接深度学习与手工特征的可解释框架，并给出特征重要性排名，为临床提供可解释的特征选择依据；

**🔧 技术方法**

使用 PWCCA（结合 SVD 降维）与 Wav2Vec 2.0 预训练模型（含 ASR 微调）进行层级特征相关性分析；

**📊 数据集**

采用法国 C2SI 语料库，包含 134 次录音的口腔及咽喉癌患者（OOC）和对照组，仅分析读音任务的可懂度分数；

**📈 对比分析**

通过层级相关性评估显示模型与光谱与韵律特征的相关系数分别达到 0.77 与 0.71，MFCC1 在所有层级中保持最高相关；模型在 C2SI 语料上的 MAE 为 0.68；

**⚠️ 局限性**

仅依赖 eGeMAPS 作为解释参考，未检验其最佳性；缺少对比其他深度模型或手工特征集的实验；模型可能包含 eGeMAPS 未捕获的高阶信息，导致解释覆盖不完全。

---

## 21. Velocity Prediction in Automatic Guitar Transcription

**arXiv ID:** 2606.24912 | [PDF](https://arxiv.org/pdf/2606.24912v1)

**作者:** Jackson Loth `[一作]` (Queen Mary University of London), Emmanouil Benetos `[通讯]` (Queen Mary University of London)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

训练了能够同时预测吉他音符与力度的自动转录模型

**💡 创新点**

利用虚拟乐器合成带力度标签的数据预训练力度预测子网络，然后将权重迁移到真实吉他音频模型，首次实现吉他力度的自动转录

**🔧 技术方法**

采用卷积递归神经网络（CRNN）并配合多阶段训练、数据增强、冻结权重等技术

**📊 数据集**

使用FrançoisLeduc合成数据（20h）、GAPS、GOAT真实吉他数据（共20h）进行训练，测试使用GuitarSet、EGDB等真实数据集

**📈 对比分析**

与不含力度预训练的基线相比，力度预训练显著降低 MAE（从约32降至7）并在部分数据集上提升 F1 约 0.1%，但提升在某些数据集上统计显著性有限

**⚠️ 局限性**

缺乏真实力度标注导致力度定义依赖虚拟乐器，难以在真实音频上进行客观评估，且模型对不同力度实现的泛化能力受限

---

## 22. Frequency Domain Reservoir Computing

**arXiv ID:** 2606.24969 | [PDF](https://arxiv.org/pdf/2606.24969v1)

**作者:** Klaus Schertler `[一作]` (Airbus Central Research and Technology), Claudio Gallicchio `[通讯]` (University of Pisa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于频域的 Echo State Network（FRESCO）架构，利用二维零填充嵌入、点积递归和压缩读取，在不使用 FFT 的前提下实现稠密 ESN 的 O(n) 计算复杂度。

**💡 创新点**

核心创新包括：①在频域完整实现稠密 ESN 并消除 FFT 开销；②二维零填充嵌入降低输入转换成本；③无冗余压缩读取去除逆 FFT；④引入 plain/mix 两种满足 Echo State Property 的频域非线性。

**🔧 技术方法**

技术手段包括：离散傅里叶变换 (DFT/FFT)、Hadamard 乘法、复数激活函数、循环卷积权重、岭回归读取、维度零填充以及频域跨频混合操作。

**📊 数据集**

实验数据集涵盖：NARMA10、Mackey‑Glass、UCR/UEA 10 个时间序列分类集（Adiac、FordA、JapaneseVowels 等）以及多变量长周期预测集（ETT、Solar、Weather）。

**📈 对比分析**

通过与标准稠密 ESN、LSTM、Mamba、Transformer 等深度序列模型以及专用预测模型对比，FRESCO 在分类任务上与 ESN 相当或略优，推理速度提升最高 26 倍；在长周期预测中精度与深度模型持平，同时能量消耗降低 1–3 个数量级。

**⚠️ 局限性**

局限性包括：需将递归权重限制为循环卷积可能削弱表达能力；跨频混合方式需手工选择，缺乏自动化；对更高维输入（如二维图像）的可扩展性尚待验证。

---

## 23. Project Auto-World: Towards Automated Benchmarking of Neural Relational Reasoners

**arXiv ID:** 2606.24965 | [PDF](https://arxiv.org/pdf/2606.24965v1)

**作者:** Anirban Das `[一作]` (Cardiff University), Steven Schockaert `[通讯]` (Cardiff University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用大型语言模型（LLM）自动生成对神经关系推理模型（Edge Transformer, ET）具有挑战性的知识图实例，并通过进化搜索、自动化研究代理和对话式提示学习采样策略，进一步训练更鲁棒的模型（SuperET）。

**💡 创新点**

创新点在于：①将LLM作为进化搜索的改进器，实现端到端的自适应采样；②提出自动化研究代理可持续改进采样器；③通过对话式LLM提示快速生成多种采样器；④发现传统难度指标（推理深度、OPEC、BL）无法解释的“隐式难度”，并引入 OPEEC。

**🔧 技术方法**

主要技术包括：LLM‑驱动的 FunSearch 进化框架、基于 Python 的优先函数学习、Edge Transformer 推理评估、自动化研究代理（Claude Opus 4.6）与工具调用、对话式 LLM 采样器设计、OPEEC 量化与难度解释。

**📊 数据集**

使用 NoRA 1.1 基准（训练集及 test-D / test-OPEC / test-BL）以及 LLM 自动生成的 Iron Coast 规则集合进行实验。

**📈 对比分析**

通过与不同采样器（Evolutionary、Claude 生成、Auto‑research）生成的查询进行交叉评估，测量 Exact‑Match Accuracy。结果显示：原 ET 在 Evolutionary 与 Auto‑research 生成的实例上准确率显著下降；SuperET 在 Evolutionary 生成的实例上保持高准确率，但在 Auto‑research 生成的实例上仍显著受限；相比之下，Claude 采样器在自身分布上表现接近完美。

**⚠️ 局限性**

局限性包括：①采样器可能对特定规则集（如 NoRA）过拟合，转移性未知；②仅在 Datalog 规则框架内实验，未覆盖非单调或分离规则的推理；③对新规则集的自动生成和验证仍需进一步研究；④实验规模受图大小（≤8 个实体）限制，未探讨更大规模场景。

---

## 24. Swarm-Inspired Generation of Collective Behaviors in Graph Dynamical Systems

**arXiv ID:** 2606.24958 | [PDF](https://arxiv.org/pdf/2606.24958v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 25. Curvature-Guided Mixing for MLLM Adaptation

**arXiv ID:** 2606.24963 | [PDF](https://arxiv.org/pdf/2606.24963v1)

**作者:** Jinglong Yang `[一作]` (Southern University of Science and Technology), Jianguo Zhang `[通讯]` (Southern University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Curvature-Guided Mixing (CGM) 及其稀疏版本（CGM†），通过在预训练与微调模型间以曲率为导向的方式进行参数混合，实现单步任务适配而不致灾难性遗忘。

**💡 创新点**

创新点在于：① 使用二阶 Taylor 近似联合优化预训练与微调损失的曲率信息，得到闭式的软混合比例；② 设计稀疏硬混合策略，基于曲率加权的分数对参数进行选择，理论上比现有启发式方法更稳健；③ 将经验 Fisher 信息矩阵和 Hutchinson 估计等手段结合，提供高效的 Hessian 近似。

**🔧 技术方法**

技术手段包括：二阶优化（Hessian/经验 Fisher 估计）、第二阶 Taylor 展开、软/硬混合公式、曲率加权的稀疏选择、参数层级稀疏更新、结构化 mask 可视化与分析。

**📊 数据集**

实验数据集涵盖 LLaVA-1.5-7B 与 Qwen-2.5VL-3B 两大 MLLM；目标任务为 OKVQA、Flickr30k、LaTeX-OCR；通用知识评估使用 VQAv2、GQA、VizWiz、SQA、TextVQA、POPE、MM-Bench、MM-Bench-CN、InfoVQA 等标准基准。

**📈 对比分析**

与标准微调、Tailor、DARE、Grafting、Magnitude、Wanda 等方法对比，采用 Avg 与 Hscore 两大指标衡量；结果显示 CGM 与 CGM† 在保持目标任务性能的同时，显著提升 Hscore 与 Avg，达到或超过现有方法的最佳平衡点。

**⚠️ 局限性**

局限性包括：① 对 Hessian 或经验 Fisher 的估计存在计算与存储开销；② 目前仅在部分层级进行混合，整体模型适配需进一步验证；③ 稀疏比例 K 与平衡系数 α 仍需手工调节，过高或过低可能影响性能；④ 实验仅在 LLaVA 与 Qwen-VL 两模型，尚未覆盖更大规模或其他模态的验证。

---

## 26. Digital Twin-Driven Adaptive Sim-to-Real Alignment via Reinforcement Learning for Vibration-Based Bearing Health Monitoring Under Data Scarcity

**arXiv ID:** 2606.24954 | [PDF](https://arxiv.org/pdf/2606.24954v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 27. Reliable Conformal Prediction for Ordinal Classification Using the Ranked Probability Score

**arXiv ID:** 2606.24959 | [PDF](https://arxiv.org/pdf/2606.24959v1)

**作者:** Stefan Haas `[一作]` (BMW Group), Eyke Hüllermeier `[通讯]` (Institute of Informatics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于秩概率分数（RPS）的一致性预测方法，用以生成满足边际覆盖、连贯且以中位数为中心的序数分类预测集。

**💡 创新点**

创新点在于将RPS作为非一致性度量，既保证覆盖、嵌套与连贯，又在条件覆盖下最小化基于L1距离的序数风险，同时实现线性时间计算。

**🔧 技术方法**

使用的技术包括秩概率分数、分裂一致性预测框架、递推更新求RPS，以及对照传统的LAC、APS、COPOC、min‑CPS等方法。

**📊 数据集**

在多种医学影像（BACH、RetinaMNIST）与年龄估计（FGNet）以及若干表格基准数据集上进行实验。

**📈 对比分析**

通过覆盖率、预测集大小、MAMM/WAMM、AISL等指标与基线比较，RPS方法在序数误差（MAMM/WAMM）与综合误差-宽度指标（AISL）上表现优于大部分对手，尽管在极端效率上略逊。

**⚠️ 局限性**

局限性包括在极低误差率下预测集宽度略大、对高度多峰分布的鲁棒性仍待进一步验证，以及缺乏对先验知识（如分布假设）利用的探讨。

---

## 28. Closure Atlases and Local-to-Global Obstructions in Finite Closure Systems

**arXiv ID:** 2606.24909 | [PDF](https://arxiv.org/pdf/2606.24909v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 29. Quantum-Resilient Decentralized AI Economies: Proof-of-Useful-Work and Post-Quantum Security

**arXiv ID:** 2606.24942 | [PDF](https://arxiv.org/pdf/2606.24942v1)

**作者:** Connor Barbaccia `[一作]` (University of Alabama), Sayanton Dibbo `[通讯]` (University of Alabama)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出了一个以AI推理和训练为工作负载的去中心化经济系统，取代传统的无效哈希计算。

**💡 创新点**

定义了闭环式代币经济模型和诚实参与的保证性权益门槛，并分析了该架构在量子攻击下的结构性安全优势。

**🔧 技术方法**

采用了三层架构（计算、验证、经济），结合zkML证明、同行评分、基准注入、声誉系统以及后量子签名（ML-DSA、SLH-DSA）和STARK。

**📊 数据集**

未使用具体数据集，主要为理论模型与对比分析。

**📈 对比分析**

通过与现有去中心化AI系统（Bittensor、Gensyn、Akash 等）和传统PoW的对比，说明其更高的实用性和量子抗性，但未给出实验性能指标。

**⚠️ 局限性**

主要限制包括zkML证明成本高、训练激励与收敛性关联不足、治理与数据隐私问题未解决，以及缺乏形式化的后量子安全证明。

---

## 30. Perfect Detection, Failed Control: The Geometry of Knowing vs. Steering in Language Models

**arXiv ID:** 2606.24952 | [PDF](https://arxiv.org/pdf/2606.24952v1)

**作者:** Cosimo Galeone `[一作]` (Alomana), Daniele Ligorio `[通讯]` (Alomana)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究语言模型内部检测某行为（如幻觉或输出格式）与实际控制该行为的向量之间的几何关系，探讨检测方向是否等价于干预方向。

**💡 创新点**

创新点在于揭示检测方向与干预方向几乎正交的“检测‑干预缺口”，并证明该缺口在不同模型、规模下保持一致，且不是可控性的先验指标；同时提出通过旋转 15° 方向可以部分弥补缺口。

**🔧 技术方法**

主要技术包括：使用差值均值与 lm_head 行对比构造线性检测/干预方向；在生成时通过向残差流注入向量实现干预；利用注意力/MLP 分解、激活植入与随机方向对照评估因果效应；以及通过余弦相似度量角度。

**📊 数据集**

数据集涵盖 Gemma 2‑2B‑it、Gemma 2‑9B‑it、Llama‑3.2‑1B‑Instruct 与 Qwen‑2.5‑1.5B‑Instruct 四个模型；使用 50 假实体 + 50 真实实体问答、32 计算题格式测试以及 115 项难度分层的硬核测试集。

**📈 对比分析**

实验比较检测方向与干预方向的余弦相似度（cos ∈ [0.12, 0.20]），验证 15° 旋转后在硬核测试中的拒绝率提升（Type 2 从 13% 提升至 60%），并在四个模型中复现该几何模式，表明缺口普遍存在且不随模型规模或预训练/指令微调变化。

**⚠️ 局限性**

局限性包括：样本量有限，主要关注幻觉与格式两种行为；未检验更大模型或编码‑解码架构；缺乏对实体复制机制产生的理论解释；仅使用线性方向，可能忽略非线性或多步路由导致的可控性差异。

---

## 31. LLM Evolution as an Industry-Scale Ecosystem: A Lifecycle Perspective on Continual Learning

**arXiv ID:** 2606.24901 | [PDF](https://arxiv.org/pdf/2606.24901v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 32. SEMIR: Topology-Preserving Graph Minors for Thin-Structure Segmentation

**arXiv ID:** 2606.24935 | [PDF](https://arxiv.org/pdf/2606.24935v1)

**作者:** Luke James Miller `[一作]` (University of Missouri-Kansas City), Yugyung Lee `[通讯]` (University of Missouri-Kansas City)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建基于图微量的边界对齐表示，对薄结构进行全分辨率分割。

**💡 创新点**

引入可参数化的图微量，保证连通性约束并消除像素网格碎片化；通过少量样本的黑盒优化实现无任务专属调参。

**🔧 技术方法**

使用像素图构造、图微量化（边收缩、节点/边删除）、GINE图神经网络分类、精确升维。

**📊 数据集**

在TTPLA（电力线）、CrackSeg9k（道路裂缝）和SkyScapes Lane（航拍车道标线）三大薄结构数据集上评估。

**📈 对比分析**

与多种通用与专用基线对比，SEMIR在Dice/Iou等指标上达到或超过最优结果，且掩模碎片化显著降低。

**⚠️ 局限性**

在大面积类任务下表现逊色，且对初始过分割质量敏感；目前仅针对二维，可扩展到三维。

---

## 33. Invisible to humans, visible to machines: a preregistered audit of Unicode fidelity across four biomedical bibliographic APIs

**arXiv ID:** 2606.24897 | [PDF](https://arxiv.org/pdf/2606.24897v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 34. Failure Modes of Large Language Models on Research-Level Mathematics: A Taxonomy and an Empirical Characterisation

**arXiv ID:** 2606.24902 | [PDF](https://arxiv.org/pdf/2606.24902v1)

**作者:** Arnesh Banerjee `[一作]` (Heritage Institute of Technology), Ayushi Bhattacharjee `[通讯]` (Heritage Institute of Technology)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过分析 First Proof 基准中的失误案例，构建了四类大型语言模型（LLM）在研究级数学推理中的失败模式，并用两套工具对 Gemini 2.5 Flash 生成的 8 篇证明进行实证验证。

**💡 创新点**

创新点在于首次提出并量化“前提挪用”（Premise Smuggling）这一失败模式，并证明传统检索增强生成（RAG）无法解决该问题，强调需要在推理时段主动防止此类错误。

**🔧 技术方法**

主要技术包括：① 基于正则表达式与 LLM 判别器的前提审计管道，用于检测无引用的关键性断言；② 基于 arXiv 查询的引用核查管道，用于识别引用伪造；以及对 Gemini 生成文本的结构化提取与验证。

**📊 数据集**

使用的数据集为 First Proof 基准中的 10 个研究级数学问题（本文以其中的 3 个问题在 Gemini 上生成 8 篇证明作为实验样本）。

**📈 对比分析**

通过人工与工具评估，所有 8 篇证明均存在错误；前提审计工具实现 100% 的精确率，召回率约 50%，而引用核查工具在 6 篇证明确认 16 条真实引用，4 条未通过自动验证。

**⚠️ 局限性**

局限性包括样本量有限（仅 8 篇证明）、使用的 Gemini 模型与 First Proof 评测的前沿系统不同，且审计工具对非典型前提挪用（如隐式使用失败定理）检测能力不足。

---

## 35. Dense Supervision Is Not Enough: The Readout Blind Spot in Looped Language Models

**arXiv ID:** 2606.24898 | [PDF](https://arxiv.org/pdf/2606.24898v1)

**作者:** Rituraj Sharma `[一作]` (Virginia Tech), Tu Vu `[通讯]` (Virginia Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究循环式语言模型中的隐藏状态如何在循环计算与预测接口之间共享，探讨密集交叉熵（CE）监督在控制隐藏状态尺度方面的局限性，并提出相应的控制方法。

**💡 创新点**

创新点在于揭示了“读出盲点”——即当读出层具备尺度不变性时，密集CE无法直接约束隐藏状态的尺度，导致尺度漂移；并给出三类解决方案：原始读出、尺度可见惩罚以及消除尺度的循环路径。

**🔧 技术方法**

主要技术包括循环式Transformer架构、RMSNorm/LayerNorm尺度不变读出、原始（非归一化）读出、显式尺度惩罚、梯度诊断、以及循环尺度截断（clamp）等。

**📊 数据集**

实验使用WikiText-103数据集（以及在补充实验中使用FineWeb的1.4B模型）。

**📈 对比分析**

通过在不同规模（44M、129M）下对比终端、循环及混合读出、尺度惩罚等多种配置，展示了尺度控制对隐藏状态尺度的显著收敛效果，并在可变深度推理中证明了尺度控制能够提升PPL-计算曲线，最终模型在相同吞吐量下PPL比RMSNorm模型低0.3-0.4点。

**⚠️ 局限性**

局限性包括实验仅覆盖中小规模模型和WikiText-103，未验证更大规模或不同数据域；机制分析为局部推导，未覆盖未来循环梯度对尺度的长期影响；以及推理评估基于教师强制下的PPL，未考虑真实自回归生成、KV缓存等部署情境。

---

## 36. Graph-Based Phonetic Error Correction of Noisy ASR

**arXiv ID:** 2606.24889 | [PDF](https://arxiv.org/pdf/2606.24889v1)

**作者:** Pratik Rakesh Singh `[一作]` (Sony Research India), Pankaj Wasnik `[通讯]` (Sony Research India)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于声学图结构与上下文语言模型的 ASR 纠错框架 G‑SPIN，能够在推理阶段纠正音素相似导致的语义关键词错误。

**💡 创新点**

创新点在于将声学相似性建模为图神经网络生成候选集，再通过掩码语言模型和指令调优的 LLM 对候选进行上下文重排序，既限制了生成空间，又避免了无约束生成的幻觉。

**🔧 技术方法**

技术包括：音素级图结构构建与 GNN 链接预测、基于 MLM 的局部上下文评分、指令调优的 LLM 进行最终重排序以及 beam search 组合候选。

**📊 数据集**

使用 Loquacious‑Set（含多语言 ASR 噪声对齐数据）进行实验，语言涵盖英语、泰卢固语、西班牙语和印地语。

**📈 对比分析**

与 DoCIA、RLLM‑CF 等 LLM 基础基线及基于知识图谱的基线对比，G‑SPIN 在 WER 上提升 10‑15%（如英语从 0.39 降至 0.32），SeMA 也显著提升，BERTScore 变化不大。

**⚠️ 局限性**

局限在于候选空间依赖声学图质量；若正确答案不在候选集中，无法纠正；对插入/删除错误修正效果有限，尤其是缺失音频导致的删除错误。

---

## 37. Convex--Concave Quadratic Spectral Filtering for Graph Neural Networks

**arXiv ID:** 2606.24956 | [PDF](https://arxiv.org/pdf/2606.24956v1)

**作者:** Ranhui Yan `[一作]` (Guangzhou Xinhua University), Haodong Yang `[通讯]` (Guangdong University of Finance and Economics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于凸凹二次滤波器组的图神经网络（DCQ-GNN），通过二阶曲率控制实现频率选择性；

**💡 创新点**

核心创新在于将滤波器的曲率极性（凸/凹）显式约束，利用二阶多项式的有限表达力在不增加多项式阶数的前提下实现更锐利的频率衰减与结构鲁棒性；

**🔧 技术方法**

采用二阶拉普拉斯滤波器组、节点自适应门控融合、Dirichlet能量与冯·诺依曼熵的理论分析，以及低阶传播实现；

**📊 数据集**

在十个涵盖同质与异质图的基准数据集上评估，包括CiteSeer、PubMed、Photo、Computers、WikiCS、Texas、Actor、Squirrel、Wisconsin、Actor等；

**📈 对比分析**

与传统线性、深度残差、以及高阶多项式（BernNet、JacobiConv、H2GCN等）对比，DCQ-GNN在同质图平均排名4.2、异质图平均排名3.0，鲁棒性最高，且参数量与推理时间显著低于高阶模型；

**⚠️ 局限性**

局限性在于对极度不规则频谱（如Squirrel）可能无法细粒度定位，且二阶滤波对高频细节的分辨率有限，未来可考虑分段二次滤波或更高阶曲率控制。

---

## 38. MacroLens: A Multi-Task Benchmark for Contextual Financial Reasoning under Macroeconomic Scenarios

**arXiv ID:** 2606.24950 | [PDF](https://arxiv.org/pdf/2606.24950v1)

**作者:** Patara Trirat `[一作]` (DeepAuto.ai), Sung Ju Hwang `[通讯]` (DeepAuto.ai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个面向美国小型/微型股市的、包含价格、会计基本面、宏观经济和文本信息的点对点多模态基准，涵盖七个金融决策任务；

**💡 创新点**

首次在单一基准中同时满足四种金融信号的时序完整性、文本发布时间门控、宏观情景与企业层面文本的联合建模，并提出非单调的特征层级消融实验；

**🔧 技术方法**

利用树模型、深度序列模型、时间序列基础模型、微调的LLM时间序列模型以及零射击LLM，配合自研的情景提取与文本门控管道；

**📊 数据集**

基准数据集包括4,416只美国小/微型股的价格、XBRL会计事实、FRED/EIA宏观系列、SEC文件、财经新闻及1,130个宏观情景，覆盖2021-2026年；

**📈 对比分析**

在19种方法（六大族群）上进行对比，经典树模型在长周期价格预测中表现最佳，零射击LLM在无衍生比率的私企估值中优于公开公司估值，特征层级消融显示基础面+宏观比情景或文本更具信息量；

**⚠️ 局限性**

局限包括U.S.单一语言、仅覆盖现有指数公司、宏观周期单一、部分公司缺失XBRL、潜在的训练集泄漏、未考虑交易成本与执行延迟等实际交易因素。

---

## 39. Holographic Memory for Zero-Shot Compositional Reasoning in Knowledge Graphs: A Mechanistic Study of Where and Why It Fails

**arXiv ID:** 2606.24948 | [PDF](https://arxiv.org/pdf/2606.24948v1)

**作者:** Randhir Kumar `[一作]` `[通讯]` (Independent Researcher), Randhir Kumar (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在零射击两跳推理任务下，利用光环化简表示和现代 Hopfield 清理的知识图嵌入模型的可行性与瓶颈，并通过机制探针定位失败原因。

**💡 创新点**

首次系统评估两种光环化简变体（实值 HRR 与相位 FHRR）结合现代 Hopfield 清理在零射击两跳推理中的表现，证明 FHRR 的软最大清理非相位等变性与检索容量限制是主要原因。

**🔧 技术方法**

采用光环化简绑定与反绑定算子、现代 Hopfield 网络清理、交叉熵与可逆性正则化训练，以及相位/模量诊断的机制探针。

**📊 数据集**

使用 FB15k‑237 知识图数据集，并在泄漏控制下构建零射击两跳链。

**📈 对比分析**

与传统单跳基线（TransE、DistMult、ComplEx、RotatE）在单跳检索上比较，实值 HRR 与 FHRR 分别达 0.358/0.350 的 MRR；但在零射击两跳推理几乎达到随机水平，远低于单跳性能。

**⚠️ 局限性**

仅评估单跳与两跳任务，未尝试多跳或稀疏图；未测试相位等价清理或容量分配架构；模型对高交叉度事实的检索受限，训练目标仅关注单跳。

---

## 40. Supervised Reinforcement Learning for the Coordination of Distributed Energy Resources

**arXiv ID:** 2606.24947 | [PDF](https://arxiv.org/pdf/2606.24947v1)

**作者:** Haoyuan Deng `[一作]` (University of Hong Kong), Yi Wang `[通讯]` (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究分布式能源资源（DER）调度的监督式强化学习框架：先用演示数据做监督预训练，再通过离线+在线两步微调提升策略性能。

**💡 创新点**

创新点：将大型语言模型的“监督预训练+RL微调”范式迁移至DER控制，提出两步微调流程以缩小仿真到真实环境的差距，并证明即使演示数据低质量也能实现高效调度。

**🔧 技术方法**

使用技术：监督学习预训练（均方误差回归）+基于PPO的策略梯度微调；离线微调在高保真仿真环境进行，在线微调在实际环境适配；构建精确非线性DER模型以提供仿真基础。

**📊 数据集**

数据集：英国国家电网的负荷与可再生发电数据、英国气象温度数据、时间分段电价表；演示数据通过模拟最优解生成并按不同质量系数（NO、HQ、MtHQ、LQ）划分；测试集为未见的30天数据。

**📈 对比分析**

比较方法与性能：与预训练策略、随机初始化PPO、sMPC（1-3h horizon）以及完美MILP做对比。SRL-2在30天测试中成本仅比perfect MILP高6.57%，比PPO低约70%，比sMPC-3低约10%，且训练时间相差不大。

**⚠️ 局限性**

局限性：单体RL受维度灾难影响，DER规模超过30时收益下降；未考虑极端事件和稀有情形；只采用单代理方案，缺乏多智能体协同；离线仿真与真实环境差距仍需进一步缩小。

---

## 41. When Do Conservation Laws Survive Learned Representations? Certified Horizons for Latent World Models

**arXiv ID:** 2606.24945 | [PDF](https://arxiv.org/pdf/2606.24945v1)

**作者:** Hongbo Wang `[一作]` `[通讯]` (Stony Brook University), Hongbo Wang (Stony Brook University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究在学习潜在表示后，如何保证物理守恒量在解码后仍可被证明保持不变，并给出了基于已知守恒量的保真度界定方法。

**💡 创新点**

创新点在于将保真对象固定为解码后的物理守恒量，提出三种保真框架（状态、潜在解码、联合守恒），并证明硬正交对称性在学习图表下失效、软自监督守恒可保持有效。

**🔧 技术方法**

采用潜在世界模型、SympNet、Hamiltonian 神经网络、软自监督守恒信号、控制‑Lipschitz 桥接、管道裁定等技术。

**📊 数据集**

使用三种保守动力学系统：摆子、谐振子和 Kepler 双体问题的合成数据，并在像素、提升和状态三种观测下训练。

**📈 对比分析**

通过与无结构、硬对称和软对齐三种模型比较，结果显示在已知相位坐标下硬对称最长，学习图表下软对齐显著优于无结构，像素场景通过读出稳定子管恢复解码能量保真，Kepler 提升场景表现出几何边界限制；总体性能体现为保真周期短但结构显著提升。

**⚠️ 局限性**

局限性包括仅在低维、已知守恒量的系统中验证，保真周期有限，像素保真需读出子管，Kepler 的几何边界不是结构失败，而是系统特异性，并且未验证高维或更复杂观测的可扩展性。

---

## 42. Error-Aware TF-IDF Retrieval-Augmented Generation for ASR Error Correction

**arXiv ID:** 2606.24915 | [PDF](https://arxiv.org/pdf/2606.24915v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 43. Unprivileged Topology Certificates for Cloud GPU Attestation

**arXiv ID:** 2606.24934 | [PDF](https://arxiv.org/pdf/2606.24934v1)

**作者:** Faruk Alpay `[一作]` (Bahçeşehir University), Taylan Alpay `[通讯]` (University of Turkish Aeronautical Association)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一种软件唯实现的云GPU证明原语，利用CUDA探针测量SM-内存区域延迟矩阵，生成可验证的证书，用于确认物理身份、硬件类别和粗略位置。

**💡 创新点**

创新点在于：①无需可信执行环境或厂商签名，仅凭低权限GPU测量即可获取稳定的物理指纹；②通过缓存绕过的HBM扫描恢复硬件拓扑签名；③将网络延迟测量与硬件指纹绑定，实现无凭证的粗粒度地理定位。

**🔧 技术方法**

技术包括CUDA探针、基于SM和内存区的延迟矩阵、流式归约、SHA‑256证书、网络延迟测量（RIPE Atlas、M‑Lab、Cloudflare）以及离散化的决策规则。

**📊 数据集**

使用了五台租用GPU（Volta V100、Hopper H200、RTX 5090、RTX PRO 6000、B200）及其完整的时序延迟数据、网络测量数据。

**📈 对比分析**

与传统基于可信根或厂商签名的远程证明对比，证明不需要GPU或完整原始数据即可验证，且在六小时负载测试中指纹稳定（中位数周期抖动≤1.5周期），跨die识别准确率>90%，硬件类别区分度显著；网络位置验证可将实例限定在几百公里范围内。

**⚠️ 局限性**

局限性包括：仅能提供粗粒度位置验证，无法防御主动主机干扰网络测量；对高层攻击如切断电源或不响应探针不敏感；受限于测量噪声和网络路径可达性；无法验证更细粒度的硬件篡改或固件完整性。

---

## 44. On-Device Neural Architecture Search

**arXiv ID:** 2606.24900 | [PDF](https://arxiv.org/pdf/2606.24900v1)

**作者:** Andrea Mattia Garavagno `[一作]` (University of Genoa), Claudio Loconsole `[通讯]` (Universitas Mercatorum)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种轻量级神经网络结构搜索（NAS）方法，可在部署设备上直接完成搜索，自动寻找适合实时传感器数据的小型网络架构。

**💡 创新点**

创新点在于：①在终端设备上直接执行NAS，充分考虑设备可用内存和推理时延的约束；②使用可微分的轻量化搜索策略；③将多传感器时序数据映射为二维矩阵，采用2D CNN处理；④将搜索空间与设备资源紧耦合，实现真正的边缘自适应。

**🔧 技术方法**

采用硬件感知NAS（HW-NAS）框架，结合基于梯度的轻量化搜索；构建2D卷积网络；使用量化感知训练并8位量化；在TensorFlow Lite for Microcontrollers上评估资源占用；测量内存、Flash、准确率和延迟。

**📊 数据集**

主要使用两个公开数据集：Italian Sign Language（ISL）——26个手势的表面肌电（sEMG）和惯性测量，实验中选取11个类；Case Western Reserve University（CWRU）——轴承振动信号，9种故障类型+正常，共10类。

**📈 对比分析**

与先前工作（ISL上Pau等，CWRU上Chen等）进行对比。指标包括RAM占用、Flash占用、准确率和推理时延。实验表明，在Raspberry Pi 4上，新NAS模型实现了0.63倍更少的RAM占用、5.96个百分点更高的准确率（ISL）以及0.44倍更少的RAM占用、0.2个百分点更高的准确率（CWRU）。在Pi 3和Zero 2 W上也保持了较好的性能，尽管低资源设备下准确率略降。

**⚠️ 局限性**

限制包括：搜索过程耗时长（常需整夜运行）；在极低内存设备（如Pi Zero 2 W）上准确率显著下降；搜索成本高，需更多实验评估真实用户场景下的搜索耗时与数据量需求。

---

## 45. Attractive and Repulsive Pattern Control in Sequence Generation

**arXiv ID:** 2606.24911 | [PDF](https://arxiv.org/pdf/2606.24911v1)

**作者:** Francois Pachet `[一作]` `[通讯]`, Francois Pachet

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文结合可签名的正则识别器与变量阶Markov模型，利用Belief Propagation精确采样，通过正负耦合来调节生成序列中的重复率与“吸引子”行为。

**💡 创新点**

创新点在于将正则约束转化为加权能量场，允许对识别到的模式施加正（奖励）或负（惩罚）耦合，从而实现自适应的回路控制与吸引子探测，并保持BP的精确性。

**🔧 技术方法**

核心技术包括：变量阶Markov模型、Belief Propagation采样、加权有限自动机实现软能量约束、适应性回路记忆机制以及对比实验中的正负耦合调节。

**📊 数据集**

实验使用六个持续时间符号化单声部乐曲（巴赫的前奏、特勒曼的幻想曲等）以及五个仅音高的维玛爵士数据库乐句，验证不同约束下的生成效果。

**📈 对比分析**

通过与基准β=0的对比，采用自重复率、自洽度、有效8-gram计数、训练4-gram覆盖率、最长重复后缀、低阶支持率及模型损失等指标。负耦合显著降低自重复率、提升多样性与覆盖率，正耦合则能诱导吸引子并揭示临界点。

**⚠️ 局限性**

局限性包括仅在单声部符号序列上验证，未覆盖多声部、音频或表达性表现；实现上每一步重建识别器导致一定计算开销；实验规模有限，未检验更长视野或更复杂约束；正向吸引子实验尚未评估其在音乐创作中的价值。

---

## 46. Why Memory Components Fail: Eight Years of License and Sustainability Events in Open-Source Data Infrastructure

**arXiv ID:** 2606.24896 | [PDF](https://arxiv.org/pdf/2606.24896v1)

**作者:** Dmitrii Dmitrenko `[一作]` `[通讯]` (Independent Researcher), Dmitrii Dmitrenko (Independent Researcher)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

**🎯 论文内容**

本文系统分析了 2018-2026 年间 105 个生产级开源数据基础设施与 AI 工具项目的许可及可持续性事件，计算事件发生率，并提出了一套包含治理、资本结构、许可、基金会成员身份、可迁移性与维护者集中度的六字段架构评估工具。

**💡 创新点**

创新之处在于把治理模式、资本结构、许可、基金会成员身份、可迁移性与维护者集中度等结构变量纳入风险评估，揭示单一供应商、VC 背书项目的事件率高达 46%，与基金会治理的项目相比差异约 19 倍。

**🔧 技术方法**

采用系统性事件编码、结构化数据集构建、描述性统计与敏感性分析，并通过案例对比与反事实分析来验证机制。

**📊 数据集**

构建了 105 项目的样本，记录 38 起事件（许可变更、功能移除、收购、仓库归档等），并将完整数据集发布在 Zenodo。

**📈 对比分析**

通过基于治理/资本结构的条件率比较，展示事件发生率差异；与未发生事件的案例（如 PostgreSQL、SQLite、Caddy 等）对比验证机制，结果显示单一供应商 VC 项目的事件率超过 4 倍。

**⚠️ 局限性**

局限性包括样本非随机、偏向商业可见项目、子细胞样本量小、事件定义聚合多种机制、只能说明相关性而非因果关系，且时间窗口仅 8 年。

---

## 47. Self-supervised Garment Dynamics with Persistent Wrinkles

**arXiv ID:** 2606.25065 | [PDF](https://arxiv.org/pdf/2606.25065v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 48. Hybrid Metadata Extraction from League of Nations Index Cards: From Feasibility Study to Archival System Integration

**arXiv ID:** 2606.24895 | [PDF](https://arxiv.org/pdf/2606.24895v1)

**作者:** Florian Cafiero `[一作]` (EPITA), Grégoire Mallard `[通讯]` (Geneva Graduate Institute)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对联合国联盟国档案索引卡开发了一套混合AI工作流，实现了元数据提取、链接和系统整合。

**💡 创新点**

将基于领域检测的OCR与端到端视觉语言模型结合，按字段功能定制提取策略，显著提升了关键标识符的识别精度与整体数据可链接性。

**🔧 技术方法**

使用YOLO进行语义字段检测、TrOCR（DeiT+ mBART）实现文本识别、Mistral系列模型进行后期纠错、Qwen‑3B视觉语言模型进行全卡级提取，并通过AtoM与Preservica实现元数据导入与长期存档。

**📊 数据集**

约500张手工标注的索引卡数据和联合国图书馆提供的包含URL的数据库文件。

**📈 对比分析**

相较于传统OCR流水线，视觉语言模型在描述性字段上提升了整体准确率，而TrOCR在文件/系列编号的识别上保持较低的幻觉率，最终实现与数据库URL的匹配率达87%。

**⚠️ 局限性**

未对所有字段的可靠性进行统一评估，数据库覆盖度不足导致匹配率受限；模型高度依赖有限的手工标注数据，难以推广到更大规模或不同语种的索引卡。

---

## 49. BFMTrack: Latent Sequence Optimization for Physics-Based Motion Tracking with Behavioral Foundation Models

**arXiv ID:** 2606.25056 | [PDF](https://arxiv.org/pdf/2606.25056v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 50. Small edits, large models: How Wikipedia advocacy shapes LLM values

**arXiv ID:** 2606.24890 | [PDF](https://arxiv.org/pdf/2606.24890v1)

**作者:** Jasmine Brazilek `[一作]` (Compassion Aligned Machine Learning), Alexa Gnauck `[通讯]` (Pro-Animal Wikipedians)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了少数志愿者在维基百科上进行动物福利主题编辑后，对大型语言模型（LLM）在相关话题上的回答产生的可测量影响。

**💡 创新点**

创新之处在于首次将梯度基数据归因（TrackStar、MAGIC）与微调消融相结合，实证证明规模有限的编辑活动能够对LLM行为产生选择性且可量化的改变。

**🔧 技术方法**

采用了TrackStar检索归因、MAGIC逆因果影响评估及LoRA微调消融三种技术，全部通过Bergson库实现，测试模型为Llama 3.1 8B和Llama 3.2 1B。

**📊 数据集**

使用的主要数据集包括125条由Pro-Animal Wikipedians（PAW）编辑的维基百科片段（共115篇文章），以及与之配对的非PAW控制段落和WikiText‑103的随机控制文本。

**📈 对比分析**

通过对比动物福利查询与相同实体的非福利查询，TrackStar显示PAW编辑在前5名检索结果中占68%（vs 52%），MAGIC在前10名中100%为PAW编辑（vs 40–60%），微调消融中PAW‑训练模型在动物福利文本上的困惑度从12.4降至8.4，控制模型在控制文本上的困惑度从16.1降至11.4，整体性能显示出显著而专一的提升。

**⚠️ 局限性**

局限性包括仅使用小规模模型（8B/1B），实验结果可能随训练顺序种子变化，归因方法测得的是损失影响而非实际对话输出，控制集仅为WikiText‑103，未覆盖其他非维基百科来源，以及未验证对更大前沿模型的直接影响。

---

## 51. RWGBench: Evaluating Scholarly Positioning in Related Work Generation

**arXiv ID:** 2606.24894 | [PDF](https://arxiv.org/pdf/2606.24894v1)

**作者:** Anzhe Xie `[一作]` (Tsinghua University), Qingyao Ai `[通讯]` (Tsinghua University)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 RWGBench，一个基于文献引用决策的评测基准，用以客观评估自动相关工作生成（RWG）的学术定位能力。

**💡 创新点**

创新点在于将 RWG 重新定义为引用决策任务，构建了 40,108 篇论文的检索语料库和 100 篇精心挑选的测试集，并引入多维度（引用适宜性、组织、结构、文本质量）评价指标。

**🔧 技术方法**

采用了检索增强生成（RAG）框架、BM25 与语义检索（Dense）、多种 LLM（DeepSeek‑V3、Llama‑3‑8B、Qwen‑2.5‑7B）以及 NLI、SciBERT、DeBERTa 等模型来实现与评估。

**📊 数据集**

数据集包含 1,091,394 篇检索文档、40,108 篇计算机科学论文以及 100 篇黄金标准相关工作段落。

**📈 对比分析**

与传统基于 ROUGE/ BERTScore 的评估相比，RWGBench 的引用相关指标在 100 篇测试集上显示当前主流模型的引用准确率低于 20%，而 Oracle 方案的性能却可达 80% 以上，揭示检索瓶颈和生成缺陷。

**⚠️ 局限性**

局限性包括检索语料库仍不完整导致部分引用缺失、评价指标对语言流畅性偏弱，以及在真正学术写作场景中对多领域迁移性尚未充分验证。

---

## 52. LLM-Based Scientific Peer Review: Methods, Benchmarks, and Reliability Challenges

**arXiv ID:** 2606.25057 | [PDF](https://arxiv.org/pdf/2606.25057v1)

**作者:** Thi Huyen Nguyen `[一作]` (Leibniz University Hannover), Zahra Ahmadi `[通讯]` (TU Braunschweig)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大语言模型在科研同行评审中的文本批评生成和分数预测进行了系统性综述与评估。

**💡 创新点**

提出了按提示、微调、检索增强、对齐优化等四大范式的结构化分类，全面梳理现有方法、数据与安全风险，并给出改进路线图。

**🔧 技术方法**

主要使用大语言模型（如 GPT‑4、LLaMA、Gemini）与相关提示工程、微调技术、检索增强与强化学习对齐等方法。

**📊 数据集**

综述了 PeerRead、OpenReviewer、Re^2、DeepReview‑13K 等多领域（主要是计算机科学）评审数据集，探讨其规模与覆盖。

**📈 对比分析**

通过对比实验发现零射击提示下 LLM 的分数相关度仅弱至中等；微调模型可提升结构与分数一致性；检索增强有助于减少幻觉，但提升有限；整体性能仍受数据噪声与偏差限制。

**⚠️ 局限性**

局限性包括数据集单一（以 CS 为主）、评价指标不足、幻觉与偏见放大、提示注入、数据投毒与检索错误、奖励偏差、缺乏鲁棒性与安全评估、伦理与隐私风险。

---

## 53. Emergent Capabilities Arise Randomly from Learning Sparse Attention Patterns

**arXiv ID:** 2606.25010 | [PDF](https://arxiv.org/pdf/2606.25010v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 54. LLM-ACES: Closed-Loop Discovery of Dynamical Systems with LLM-Guided Adaptive Search

**arXiv ID:** 2606.25039 | [PDF](https://arxiv.org/pdf/2606.25039v1)

**作者:** Nikhil Abhyankar `[一作]` (Virginia Tech), Chandan K. Reddy `[通讯]` (Virginia Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

使用大型语言模型引导的符号假设生成与自适应轨迹采集构建闭环动力学方程搜索框架，能够在实验数据上直接恢复可解释的微分方程

**💡 创新点**

创新点在于将LLM生成的操作符先验与候选方程的不确定性驱动主动采样相结合，形成数据与假设共同进化的闭环流程，显著提升可辨识度和样本效率

**🔧 技术方法**

核心技术包括LLM操作符先验生成、符号回归后验优化、基于预测分歧的轨迹主动采样，以及经验缓冲区驱动的迭代反馈

**📊 数据集**

在ODEBench和ODEBase两大基准数据集上进行实验，涵盖1D–4D不同领域的动力学系统

**📈 对比分析**

与被动符号回归、LLM辅助和主动采样基线相比，取得NMSE低至10⁻¹⁷、符号准确率达46%/52%（ODEBench/ODEBase），样本效率提升10倍，且对噪声具有较强鲁棒性

**⚠️ 局限性**

局限在于仅针对自回归ODE，依赖LLM先验质量、符号回归后端及可查询的实验/仿真接口；对PDE、随机动力学或高噪声环境的适用性需进一步改进

---

## 55. Do Thinking Tokens Help with Safety?

**arXiv ID:** 2606.25013 | [PDF](https://arxiv.org/pdf/2606.25013v1)

**作者:** Narutatsu Ri `[一作]` (Princeton University), Sanjeev Arora `[通讯]` (Princeton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究系统评估了大型推理模型在安全决策中的“思考”过程，揭示其对拒绝/服从决策的实际贡献极小；同时比较了多种推理模型与多种推理时与训练时安全防御的效果。

**💡 创新点**

创新点在于：①发现第一个思考标记的隐藏表示即可高精度预测最终拒绝/服从，说明安全决策早已在模型内部确定；②证明思考过程主要是后置阐释（prefix completion）而非真正的推理修正；③通过大规模实验评估多种开源推理模型和防御方法，显示现有防御往往导致过度拒绝且未显著提升 ASR–ORR 均衡。

**🔧 技术方法**

采用线性探测器（Logistic Regression + PCA）、Fisher 判别、句子级立场分析、ASR/ORR 评价指标、四个门控拒绝判别器等技术手段，对隐藏状态和生成文本进行定量分析。

**📊 数据集**

使用 2,500 条有害提示和 2,885 条友善提示（扩展至 6,750 条），并结合 PHTest、ORFuzzSet 等公开数据集进行评测。

**📈 对比分析**

与基线模型和 Qwen3‑8B、Olmo‑3‑7B‑Think、Phi‑4‑Reasoning、GPT‑OSS‑20B 等模型对比，线性探测器在第一个思考 token 上 AUROC 0.84–0.95，BAcc 0.76–0.88；思考对 ASR/ORR 的提升不足，防御方法多导致过度拒绝。

**⚠️ 局限性**

局限性包括：①仅评估中等规模开源推理模型，无法推断更大规模模型是否相同；②评估范围仅限于拒绝/服从，不涉及事实性、隐私等其它安全维度；③未深入探究安全信号在预训练/指令微调/后处理阶段的产生机制。

---

## 56. Geo-Strat-RL: Learning Geological Event Reasoning from Verifiable Tasks

**arXiv ID:** 2606.25000 | [PDF](https://arxiv.org/pdf/2606.25000v1)

**作者:** Lukas Mosser `[一作]` `[通讯]` (Aker BP ASA), Lukas Mosser (Aker BP ASA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了 Geo-Strat-RL，结合合成地质生成器与可执行验证器，构造可验证的地质事件推理任务，并使用强化学习（RLVR）训练视觉‑语言模型以输出符合时序和结构约束的 JSON 事件序列。

**💡 创新点**

首次提出可验证奖励的 RL 训练框架用于视觉地质推理，并构建跨域（层序图与合成地震）可复制数据集与可执行验证器，证明学习到的推理概念能够在不同观测域间迁移。

**🔧 技术方法**

采用 LoRA 微调、TRL 的 Group Relative Policy Optimization、Qwen、Gemma 等开闭源 VLM 以及 OpenAI GPT‑5.5，并实现合成地质生成器与有限差分地震渲染作为训练与评估环境。

**📊 数据集**

使用自研的合成层序图数据集及其对应的可验证 JSON 事件目标，配合相同场景的合成地震幅度图，数据集划分为固定的训练/验证/测试集，保证实验可复现。

**📈 对比分析**

通过在 held‑out 测试集上评估预训练 VLM 与 RLVR LoRA 适配器的多项指标（JSON 解析、事件计数、顺序、类型、沉积、断层、侵入、不连续面等），RLVR 在所有指标上显著提升，最佳模型 Qwen3‑VL‑4B diagram LoRA 在图表域得到 R≈0.74，在地震域迁移后仍保持较高性能（R≈0.75）。

**⚠️ 局限性**

合成数据过于理想化，缺少真实地质的不确定性与多源证据，且仅针对单一观测图像；结构属性（断层、侵入、不连续面）仍难以准确推断；未提供人类基准，限制了对真实地质解释能力的直接评估。

---

## 57. ExTra: Exploratory Trajectory Optimization for Language Model Reinforcement Learning

**arXiv ID:** 2606.24994 | [PDF](https://arxiv.org/pdf/2606.24994v1)

**作者:** Wenyang Hu `[一作]` (National University of Singapore), Bryan Kian Hsiang Low `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

改进了强化学习在大型语言模型推理中的探索策略，提出了可在GRPO框架内使用的探索轨迹优化方法。

**💡 创新点**

创新点在于两项机制：① 只奖励正确答案的嵌入层多样性（novelty reward）以缓解易题的多样性崩溃；② 通过无监督的平均词元熵（MTE）在困难题中识别低熵前缀，从中进行前缀重生（entropy‑guided prefix regeneration）以恢复梯度信号。

**🔧 技术方法**

使用的技术包括GRPO（Group Relative Policy Optimization）、句子嵌入相似度奖励、无监督熵指标、嵌入平滑与前缀重采样等，整体仍属于标准的策略梯度训练。

**📊 数据集**

在六个数学推理基准（MATH‑500、AMC23、Minerva、OlympiadBench、AIME24、AIME25）上进行评估，模型以Qwen3‑1.7B和Nemontron‑1.5B为基础。

**📈 对比分析**

与GRPO、DAPO以及各自的单一机制 ablation 对比，ExTra 在 pass@1 上平均提升约4.9点，在 pass@16 上提升约6.8点；特别是在最难的 AIME24/AIME25 上提升超过20点，且样本效率高于 DAPO。

**⚠️ 局限性**

局限性包括：① 对 novelty 系数 γ 的敏感性，需要调参；② 仅在数学推理任务验证，跨任务泛化尚待考察；③ 仍依赖二值正确/错误奖励，无法捕捉部分中间奖励信息。

---

## 58. Chorus II: Cross-Request Sparsity Reuse for Efficient Image-to-Video Generation

**arXiv ID:** 2606.25040 | [PDF](https://arxiv.org/pdf/2606.25040v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 59. Noise-Aware Boundary-Enhanced Generative Learning for Ultrasound Speckle Reduction

**arXiv ID:** 2606.25009 | [PDF](https://arxiv.org/pdf/2606.25009v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 60. Uncertainty-aware reinforcement learning for chemical language models

**arXiv ID:** 2606.24990 | [PDF](https://arxiv.org/pdf/2606.24990v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 61. Low-Cost High-Order Singular Value Decomposition for Tensor-Based Reconstruction from Sparse Sensor Measurements: Urban Flow and Air-Quality Applications

**arXiv ID:** 2606.24989 | [PDF](https://arxiv.org/pdf/2606.24989v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 62. Latent Block-Diffusion Temporal Point Processes: A Semi-Autoregressive Framework for Asynchronous Event Sequence Generation

**arXiv ID:** 2606.24982 | [PDF](https://arxiv.org/pdf/2606.24982v1)

**作者:** Shuai Zhang `[一作]` (Academy of Mathematics and Systems Science Chinese Academy of Sciences), Zhi-Ming Ma `[通讯]` (Academy of Mathematics and Systems Science Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新的半自回归时点过程框架（LBDTPP），通过在隐空间中对事件块进行高斯扩散，实现可变长度、并行高质量的异步事件序列生成。

**💡 创新点**

创新点在于：① 将事件映射到连续隐空间后，在块层面进行自回归分解并在每块内执行扩散采样；② 兼顾自回归模型的长度可变性与扩散模型的并行高质量生成，显著降低多步生成中的误差累积；③ 对误差累积给出 Wasserstein 上的理论界限。

**🔧 技术方法**

使用的技术包括：时间与标记的无参数编码、Transformer 结合块扩散注意力掩码、Gaussian 前向/逆扩散、DDIM 采样、Wasserstein 误差分析、以及事件解码器（MLP）。

**📊 数据集**

实验数据集：Taxi、Taobao、StackOverflow、Retweet、MOOC、Amazon 六个真实世界的标记事件序列。

**📈 对比分析**

方法与基线对比：在无条件与有条件生成任务中分别与九种基准（自回归 TPP、非自回归扩散 TPP 等）进行对比，评价指标为 OTD、RMSE_m、RMSE_τ、sMAPE。实验表明 LBDTPP 在大多数指标上均优于所有基线，尤其在无条件生成中表现突出；同时采样速度与基线相当或更快。

**⚠️ 局限性**

局限性：① 块大小需手动调节，过大或过小都会影响性能；② 仅处理时间-标记事件，未考虑空间或多模态信息；③ 扩散采样步骤仍对生成速度有一定影响；④ 对标记嵌入的学习效果有限，固定编码已能满足需求。

---

## 63. Adaptive Joint Compression and Synchronisation in Federated Split Learning for IoT Rainfall Prediction

**arXiv ID:** 2606.25003 | [PDF](https://arxiv.org/pdf/2606.25003v1)

**作者:** Wenjie Ding `[一作]` (Newcastle University), Rajiv Ranjan `[通讯]` (Newcastle University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本论文构建了一个面向IoT降雨预测的Federated Split Learning框架，服务器通过EMA驱动的轻量调度器同时控制激活压缩模式与同步间隔

**💡 创新点**

其创新点在于联合动态压缩与同步调度的自适应控制，在保证预测性能的前提下显著减少通信负载，并在真实Raspberry Pi设备上验证了高延迟场景下的有效性

**🔧 技术方法**

采用了FSL、float32/float16/int8激活量化、服务器端EMA阈值调度、FedAvg带有限滑动同步、gRPC通信、双分支MLP预测头等技术

**📊 数据集**

使用ERA5小时级气象观测数据，包含11个英国气象站，构成训练/验证/测试集，样本以48小时窗口预测未来24小时降雨

**📈 对比分析**

通过17个仿真场景与4个Pi部署场景对比AUPRC、激活上传量、同步流量等指标；AUPRC保持在0.638–0.648区间，激活上传量降87%，同步流量降54%，运行时标准差从±688 s降至±10 s

**⚠️ 局限性**

实验未覆盖客户端掉线鲁棒性、动态切分层选择及更大规模/不同任务的验证，且调度阈值规则可能不适用于更复杂场景

---

## 64. Multi-Stream Temporal Fusion for Financial Fraud Detection

**arXiv ID:** 2606.25007 | [PDF](https://arxiv.org/pdf/2606.25007v1)

**作者:** Mohammadamin Dashti Moghaddam `[一作]` (Amazon Web Services), Nick Sciarrilli `[通讯]` (Amazon Web Services)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Multi-Stream Fraud Transformer（MSFT），对金融交易、登录和风险事件等多条时间序列分别编码并通过可配置的融合机制实现欺诈检测。

**💡 创新点**

创新点在于将多条异构事件流分别使用 Transformer 编码，并引入时间感知位置编码、门控融合和跨流注意力，实现对跨流时序关联的显式建模。

**🔧 技术方法**

采用 Transformer 编码器、可学习的频率基时间位置编码、门控加权融合、跨流双向注意力以及 MLP 分类头，配合分布式训练与参数正则化。

**📊 数据集**

使用了包含 10,000,000 名用户、1.5% 欺诈率的合成数据集（包含交易、登录和风险事件三条流），并在真实数字银行生产数据上进行了验证。

**📈 对比分析**

与 XGBoost、单流 Flatten Transformer、LSTM 等基线对比，MSFT 在 AUROC、AUPRC、F1 等指标上提升 25+ 分点；门控融合在 0.989 的精确度下，时间编码在 0.9961 的 AUROC 领先；在生产环境中相对 AUROC 提升 22%。

**⚠️ 局限性**

局限包括：合成数据可能不完全覆盖真实欺诈模式；跨流注意力在当前规模需更多训练轮次；仅进行离线评估，未考虑推理延迟；模型假设所有流在推理时均可获得。

---

## 65. Invariant Kalman filtering for extended pose estimation in multi-IMU articulated rigid-body systems

**arXiv ID:** 2606.25083 | [PDF](https://arxiv.org/pdf/2606.25083v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 66. Closed-Loop Graph Algorithm Execution with Small Language Models: Step Accuracy and Rollout Reliability

**arXiv ID:** 2606.24980 | [PDF](https://arxiv.org/pdf/2606.24980v1)

**作者:** Michal Podstawski `[一作]` `[通讯]` (NASK National Research Institute), Michal Podstawski (NASK National Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究小型语言模型在图算法执行中的闭环决策性能，评估单步预测与完整回放的可靠性；

**💡 创新点**

提出高单步准确率不等于完整执行可靠的洞察，并引入前缀存活、回放诊断等指标来衡量闭环执行效果；

**🔧 技术方法**

使用指令微调的 Llama‑3.2‑1B‑Instruct 与 Qwen2.5‑1.5B‑Instruct，采用 LoRA 适配并结合可解析的文本指令与确定性执行器；

**📊 数据集**

基于 TinyGraphEstimator 合成图集，包含 20–30 顶点的 Barabási‑Albert、Erdős‑Rényi、Watts‑Strogatz 三类图，拆分为 900/150/150 的训练/验证/测试集；

**📈 对比分析**

对比教师强制的单步准确率与完整回放准确率，并辅以约束有效性、软分数、误差间隙、前缀 AUC 与校正次数；结果显示遍历与着色任务可达 95–97% 的回放准确率，而加权算法即使单步准确率 80–90% 也仅能实现 10–25% 的回放成功；

**⚠️ 局限性**

实验仅涵盖小型合成图，未涉及大规模或真实图；权重为确定性整数，随机性与多样性有限；回放准确率依赖于单一的确定性实现与 tie‑breaking 规则，可能不具备更广泛的通用性；

---

## 67. Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models

**arXiv ID:** 2606.25041 | [PDF](https://arxiv.org/pdf/2606.25041v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 68. Yuvion VL: A Multimodal Foundation Model for Adversarial Content and AI Safety

**arXiv ID:** 2606.25034 | [PDF](https://arxiv.org/pdf/2606.25034v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 69. Quantifying Explainable AI-introduced signal noise on ECG data with Spectral Entropy

**arXiv ID:** 2606.24974 | [PDF](https://arxiv.org/pdf/2606.24974v1)

**作者:** David A. Kelly `[一作]` (King's College London), Nathan Blake `[通讯]` (King's College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出利用光谱熵评估心电图深度学习模型的解释工具产生的自噪声，并对比多种后验可解释方法。

**💡 创新点**

创新点在于首次将光谱熵作为定量指标来衡量可解释工具的噪声水平，提供了低成本、实时评估可解释性的新途径。

**🔧 技术方法**

采用1D卷积神经网络进行心律失常分类，使用SHAP、Integrated Gradients、DeepLIFT、Grad‑CAM、LIME和RE‑X等可解释方法，并计算它们输出的光谱熵。

**📊 数据集**

数据集为MIT‑BIH心电图数据库，共1000份心电波形，涵盖17种心律失常。

**📈 对比分析**

比较结果显示，大多数工具在17类上的光谱熵基本稳定，异常工具（如RE‑X）噪声波动大；在PVC示例中，Shapley值工具噪声高但能定位关键区间，光谱熵低的工具虽然噪声小却可能错位。

**⚠️ 局限性**

局限性包括光谱熵无法直接衡量解释准确性，可能对非局部解释误判为低噪声，且对不同数据维度或特征尺度的适用性尚待验证。

---

## 70. What Does It Mean to Break a Distillation Defense?

**arXiv ID:** 2606.25059 | [PDF](https://arxiv.org/pdf/2606.25059v1)

**作者:** Lena Libon `[一作]` (ETH Zurich), Florian Tramèr `[通讯]` (ETH Zurich)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `8d10c613-917e-4880-9716-17789f50e119` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LLM蒸馏攻击的防御方法，指出输出扰动防御缺乏统一的威胁模型，并提出了包含查询预算、数据预算和接口配置三维的威胁模型框架；以抗蒸馏采样（ADS）为案例，演示不同威胁模型对防御效果的影响。

**💡 创新点**

首次将威胁模型拆分为可量化的查询预算、数据预算以及接口配置，明确阐述接口层面的细节对防御评估的决定性作用；强调评估时必须显式给出威胁模型，并通过实验验证不同维度对防御效果的显著改变。

**🔧 技术方法**

使用抗蒸馏采样（ADS）作为输出扰动防御，结合重复删除、最大k重采样、预填充等后处理；采用LoRA微调学生模型；在不同查询预算与接口配置下对学生在GSM8K上的准确率进行测评。

**📊 数据集**

GSM8K问答数据集的70%子集（5231条样本）用于训练与评估。

**📈 对比分析**

在不同的查询预算（Q= D,1.5D,2D,3D,4D）和接口配置（仅文本输出vs.支持预填充）下，比较学生模型在GSM8K上的准确率。实验表明：在原始威胁模型下可实现一定的性能降低；但加入重复删除后，学生准确率已大幅提升；进一步扩大查询预算或启用单词预填充，可使学生准确率几乎恢复到无防御的温度采样水平，说明防御鲁棒性高度依赖威胁模型。

**⚠️ 局限性**

仅探讨了部分威胁模型维度，未覆盖更大数据预算、不同输入/输出通道或日志概率接口；实验仅以ADS为示例，未提出最强攻击；评估仅在固定的教师、代理学生与学生模型组合下进行，缺乏对其它模型组合的泛化验证。

---

## 71. Do vision-language models search like humans? Reasoning tokens as a reaction-time analog in classic visual-search paradigms

**arXiv ID:** 2606.25066 | [PDF](https://arxiv.org/pdf/2606.25066v1)

**作者:** Farahnaz Wick `[一作]` `[通讯]` (Independent Researcher), Farahnaz Wick (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过将经典视觉搜索实验转化为对vision‑language模型的API调用，并以模型产生的推理token数量作为搜索努力的代理，检验这些模型是否呈现与人类相似的搜索斜率和错误模式。

**💡 创新点**

提出了一种无需内部分析的无监督方法：利用模型的中间token计数来近似反应时，从而在不访问模型内部的情况下直接比较模型与人类的视觉搜索行为，并揭示前沿VLM在特征搜索、组合搜索、枚举和搜索不对称性上的相似与差异。

**🔧 技术方法**

使用现有前沿与中阶VLM（Claude Opus 4.8、GPT‑5.5、GPT‑4o、Claude Sonnet 4.6、o4‑mini等）的API推理；实验设计基于特征/组合、T‑vs‑L、枚举、倾斜/垂直不对称等经典视觉搜索范式，并通过自动生成的推理token序列测量模型努力。

**📊 数据集**

使用Wolfe实验室公开的75,910次特征/组合搜索人类数据集，以及作者自行生成的合成字母与条形图显示，用于与模型结果对照。

**📈 对比分析**

通过对模型推理token与人类反应时的斜率进行线性回归，计算Pearson/Spearman相关系数；结果显示GPT‑5.5与思考型模型在特征/组合搜索斜率上与人类高度相关（r≈0.7），但在缺失确认、枚举及不对称搜索上表现出明显差异，表明模型在部分维度上重现人类行为，而在细节上 diverges。

**⚠️ 局限性**

推理token并非真正的反应时，受解码策略和训练影响；自适应思考模型导致努力量不易比较；实验仅用合成符号，缺乏自然图像和新的人类数据；未对内部注意机制进行分析，限制了对模型内部机制的解释。

---

## 72. Neural Scaling Universality: If Exponents Are Fixed, Time to Understand Coefficients

**arXiv ID:** 2606.25008 | [PDF](https://arxiv.org/pdf/2606.25008v1)

**作者:** Yizhou Liu `[一作]` (Massachusetts Institute of Technology), Jeff Gore `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了神经网络缩放通用性假设，解释预训练损失随训练时间、模型宽度和深度按固定指数衰减，并通过系数来预测最佳模型形状和计算极值。

**💡 创新点**

创新点在于将Softmax非线性、特征叠加与层级平均化这三种机制统一解释为导致固定指数（1/3、1、1）的根源；并将指数固定、系数可调的“通用性类”与传统依赖数据幂律的解释区分开来。

**🔧 技术方法**

使用理论推导（Softmax 动力学、几何干涉、残差网络的集成平均）和大规模预训练模型的损失曲线拟合，得到时间系数cτ、宽度系数cm、深度系数cℓ，以及从中推导出的compute系数。

**📊 数据集**

使用三组公开大语言模型（Pythia、Chinchilla、Farseer）的预训练损失数据，涵盖不同参数规模、宽度深度比和训练集大小。

**📈 对比分析**

通过将模型损失按公式 L = cτ/τ¹/³ + cm/m + cℓ/ℓ + L₀ 拟合，验证了三种缩放法则；进一步推导并验证了最佳形状（m/ℓ≈c_m/2c_ℓ）和最佳计算极值（D/N≈(c_D/c_N)³，损失∝C⁻¹/⁶）与实验数据高度一致。

**⚠️ 局限性**

局限性：系数的具体数值受数据、架构和超参数影响，尚未系统量化；当前证明主要基于稠密Transformer，未覆盖MoE、循环等变体；理论仍为假设，需在更多模型和数据上进一步验证；对不同任务和域的泛化能力未知。

---

## 73. Scalable Peptide Design via Memory-Efficient Equivariant Transformer

**arXiv ID:** 2606.25006 | [PDF](https://arxiv.org/pdf/2606.25006v1)

**作者:** Rui Jiao `[一作]` (Tsinghua University), Jianzhu Ma `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种内存高效的E(3)-等变Transformer骨干（MEET），用于全原子肽设计。

**💡 创新点**

创新点在于将距离感知的查询-键增强、全局向量初始化以及稀疏键适配器融入Transformer，消除了二次激活量的需求，同时保持E(3)等变性。

**🔧 技术方法**

采用的技术包括内存高效注意力（FlashAttention兼容）、距离增强注意力、稀疏键消息传递、VAE+潜在扩散模型以及全原子坐标重构。

**📊 数据集**

使用了基于AFDB数据库的约1.2亿候选片段，筛选后构建了约100K和1.2M的肽-蛋白复合结构训练集。

**📈 对比分析**

与PepGLAD、PepFlow、UniMoMo和DiffPepBuilder等基线对比，MEET在预测结合自由能（ΔG）和PoseBuster物理有效率上均优于所有基线，并且随着模型规模和数据规模提升，生成质量和多样性均得到显著提升。

**⚠️ 局限性**

局限在于在小数据量下序列多样性较低，且骨干仍受限于当前Transformer层深度和隐藏维度，未来可进一步探索更大规模或不同结构的扩展。

---

## 74. Certification of Machine Learning Models via Directional Sharpness

**arXiv ID:** 2606.25004 | [PDF](https://arxiv.org/pdf/2606.25004v1)

**作者:** Gefei Tan `[一作]` (Northwestern University), Mariana Raykova `[通讯]` (Google)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种新的模型泛化评估方法——方向性锐度（Directional Sharpness），并证明其能在模型认证场景下为模型的泛化性能提供可靠、可计算的证明。

**💡 创新点**

创新点包括：①将锐度从静态几何度量转化为动态稳定性度量，利用 SAM 迭代过程中参数的连续扰动来检测模型在训练扰动下的稳健性；②给出理论上阐明动态锐度与 SAM 稳定性、泛化之间的关系；③设计了高效的零知识证明协议，使验证者无需访问训练数据即可核实模型的泛化质量。

**🔧 技术方法**

核心技术包括：Sharpness‑Aware Minimization (SAM) 动态扰动、批量梯度噪声分析、梯度一致性与曲率一致性度量、零知识证明（ZKP）与向量无关线性评估（VOLE）等。

**📊 数据集**

主要使用的数据集为 CIFAR‑10 与 CIFAR‑100，网络结构包括 VGG‑13/16/19‑BN 与 WideResNet‑28‑10；此外还构造了多种训练偏差场景（噪声标签、伪装特征、后门、量化误差、样本量不足等）用于评估鲁棒性。

**📈 对比分析**

与现有静态锐度（ASAM、Worst‑Case）和传统的测试准确率比较，方向性锐度在 Spearman、Kendall 相关系数上显著更高（如 Spearman ρ ≈ 0.90），在检测训练偏差的区分率上也更优；在模型审计中计算时间比测试准确率快约 4 倍；在零知识证明中，相较于完整训练证明，方向性锐度证明速度提升至 80,000×，耗时从数千小时降至 90 分钟。

**⚠️ 局限性**

局限性包括：①需要选择阈值以将连续分数映射为二元判定，阈值依赖于任务与数据集；②理论证明为单向，即只给出“稳定⇒锐度受限”与“锐度指数增长⇒不稳定”的关系；③对极端模型结构或非监督任务的适用性尚未验证；④零知识证明协议的实现仍需要专业知识，且对模型规模较大的情况下的可扩展性尚待进一步评估。

---

## 75. Internal Data Repetition Destroys Language Models

**arXiv ID:** 2606.24998 | [PDF](https://arxiv.org/pdf/2606.24998v1)

**作者:** Jessica Chudnovsky `[一作]` (Stanford University), David Donoho `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在语言模型预训练中，系统地评估了精确文档重复对训练损失的影响，并量化了重复次数与 Compute-Equivalent Gain / Loss 的非单调关系。

**💡 创新点**

创新点包括：① 用 Compute-Equivalent Loss 替代传统的有效参数数衡量重复损失；② 发现重复峰值随模型规模呈幂律增长；③ 通过一个简洁的线性回归统计模型解释了重复峰值的出现。

**🔧 技术方法**

技术上采用了 Chinchilla 最优训练比例（C ≈ 120 N²），在 Qwen3 风格的解码器上训练多规模模型，并在 FineWeb-Edu-Dedup 上对 10% 重复 token 进行多重复度（R）实验；随后拟合无重复的 Chinchilla 损失曲线，将实验损失映射到 Compute-Equivalent 指标；并推导了一个闭式线性回归模型来解释重复的统计效应。

**📊 数据集**

使用的数据集为 FineWeb‑Edu‑Dedup（已去重的教育文本语料），实验中保持 10% 的 token 归属给重复文档。

**📈 对比分析**

与不重复的基线（相同规模与训练时长）对比，计算 Compute-Equivalent Loss；在最大规模（344 M 参数）下，重复峰值导致的 Compute-Equivalent Loss 约为 0.33，意味着模型需要相当于只有约 67% FLOPs 的计算量才能达到同等损失水平。

**⚠️ 局限性**

局限性包括：① 只研究精确文档重复，未覆盖近似或语义重复；② 固定 10% 重复 token 可能无法推广到更高或更低重复比例的场景；③ 统计线性回归模型简化了真实模型的注意力、深度和优化器细节，可能未能捕捉所有导致重复损失的机制。

---

## 76. Learning Diachronic Representations of Ancient Greek Letterforms

**arXiv ID:** 2606.24984 | [PDF](https://arxiv.org/pdf/2606.24984v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 77. From Forecasting Leaderboards to Deployment Decisions: A Fail-Closed Certification Protocol

**arXiv ID:** 2606.24996 | [PDF](https://arxiv.org/pdf/2606.24996v1)

**作者:** Geumyoung Kim `[一作]` `[通讯]` (Chungbuk National University), Geumyoung Kim (Chungbuk National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了一套失败关闭的认证协议，用于判定预测排行榜中排名第一的模型在经过固定决策接口（如阈值、top‑k 预算或切换成本）后，是否仍能作为可部署的最佳决策方案；通过在预设的预测‑决策链和部署效用下，对预测赢家与部署赢家之间的差异进行系统门检，判断其可部署性；

**💡 创新点**

创新点在于将排行榜赢家与部署决策效果分离，提供了可重复、可解释的多重门检流程（零摩擦对齐、正摩擦逆转、平衡稳定性、统计置信区间、复现支持），从而避免把排行榜赢家误读为部署第一推荐，并能在发现真正部署失败时给出严谨的证据；

**🔧 技术方法**

技术方法主要包括：1）构建预测‑决策链和部署效用函数；2）计算不同摩擦水平下的部署效用；3）设计并执行一系列失败关闭门检（零摩擦一致性、正摩擦逆转、ϵ‑tie 审计、bootstrap 置信区间、复现支持）；4）对真实数据集进行多种验证（Traffic‑Hourly、Event‑Micro、NOAA、Inventory）和对抗式原生审计；

**📊 数据集**

使用的数据集包括：Traffic‑Hourly（交通风险小时预测），Event‑Micro（事件阈值预测），NOAA（气象预测）以及 Inventory（库存预测），每个数据集均配有相应的预测模型集和部署效用定义；

**📈 对比分析**

通过将 Traffic‑Hourly 在不同摩擦水平下的部署效用与预测赢家进行比较，发现当摩擦为0时两者一致，但正摩擦时预测赢家被部署子最优；在对 362 个原生网格单元进行审计时，所有 155 个明显的赢家逆转都被门检拒绝，没有出现无足够证据的认证案例；这表明认证流程能够准确捕捉真正的部署失效；

**⚠️ 局限性**

局限性包括：1）仅在有限数量的数据集和固定的预测‑决策接口上测试；2）门检为充分条件，未能给出所有可能失效的必要条件；3）未涉及在线动态模型切换或自适应决策接口；4）对模型集的大小和支持度敏感，支持度不足时可能被误判为无效；

---

## 78. Erased, but Not Gone: Output Forgetting Is Not True Forgetting

**arXiv ID:** 2606.25001 | [PDF](https://arxiv.org/pdf/2606.25001v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 79. The Geometry of Sequential Learning: Lie-Bracket Prediction of Transfer Order

**arXiv ID:** 2606.24993 | [PDF](https://arxiv.org/pdf/2606.24993v1)

**作者:** John Sweeney `[一作]` `[通讯]` (Sideplane AI), John Sweeney (Sideplane AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了顺序学习的几何学，提出了基于 Lie‑Bracket 的顺序预测器和 Lie‑Bracket 竞赛（tournament）来实现多域调度。

**💡 创新点**

创新点在于将梯度更新的 Lie‑Bracket（即梯度场的对易子）作为可计算的几何量，用来预测两域顺序的目标损失差；进一步通过共享参考点将这一对易子转换为无须枚举 N! 方案的可扩展竞赛，兼顾权重、置信度与阶级规模。

**🔧 技术方法**

使用梯度、Hessian‑vector product（HVP）、梯度对易子、Borda/行和排名、Trotter 参考点、η‑autopilot 及子空间 HVP 计算等技术。

**📊 数据集**

验证数据集包括：LLM 预训练域（The Pile 17 个子域）、Diffusion UNet 的 CIFAR‑10 类域、Instruction‑SFT（Qwen2.5‑1.5B + Dolly 15k 任务）、DPO（UltraFeedback）、MMLU 题目、Stack 编程语言域等，涵盖多种模型（Qwen2.5、Llama‑3.2、Llama‑3.1、SmolLM‑3、DDPM UNet）。

**📈 对比分析**

与随机顺序、梯度范数基线、全排列搜索（N≤3）及 AdamW 状态优化器对比，Lie‑Bracket 预测器在两域场景下达到 98%+ 的符号预测准确率，在多域竞赛中能达到 99%+ 的样本百分位，且在 k≥5 时已快于两顺序暴力搜索，且在实际训练步骤（k=20/50）下仍保持高于随机的性能提升。

**⚠️ 局限性**

局限性包括：需要 HVP 计算，对大模型仍有一定开销；对 AdamW 等有状态优化器的理论尚不完整；在极小信噪比或梯度对易子接近零的边界情况预测不稳定；以及对大规模多域场景仍需进一步验证其在真实工业流水线的可行性。

---

## 80. Why Do Accumulated Transformations Extrapolate?

**arXiv ID:** 2606.24975 | [PDF](https://arxiv.org/pdf/2606.24975v1)

**作者:** Mahesh Godavarti `[一作]` `[通讯]` (A Carrot, Inc.), Mahesh Godavarti (A Carrot, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了累计旋转（accumulated rotations）在Transformer中的位置编码对长度外推性能的影响，并解释其机制；

**💡 创新点**

证明累计正交变换在满足谱间隙条件时会产生有限混合窗口，从而通过分数侧去耦（score-side decoherence）抑制远距离注意力，并且在2×2块对角（(2)）旋转下通过旋转值（value rotation）进一步降低残留远程干扰；

**🔧 技术方法**

使用正交旋转（包括随机、学习的 token 角度以及Householder反射）、RoPE、ALiBi等位置编码方法，并构建了一个精简的“stylized attention”模型；

**📊 数据集**

在OpenWebText数据集上训练 512 维上下文长度的 decoder-only transformer，并在长达 65,536 个 token 的评估中测试外推性能；

**📈 对比分析**

通过与 RoPE、固定 RoPE、随机累计旋转、学习累计旋转以及 ALiBi 等方法对比，发现累计旋转显著提升外推 perplexity（如随机累计旋转 Q/K/V 在 65K 长度仅提升 1.59×），并验证旋转值可进一步改进；

**⚠️ 局限性**

仅靠旋转（不做远程质量控制）在极长序列上仍会逐渐退化；当前理论未能完全解释多层和非交换旋转的行为，且未对非归一化注意力做完整分析。

---

## 81. Diagnosing and Mitigating Compounding Failures in Agentic Persuasion via Taxonomic Strategy Retrieval

**arXiv ID:** 2606.24976 | [PDF](https://arxiv.org/pdf/2606.24976v1)

**作者:** Pradyumna Narayana `[一作]` (Google), Purvi Sehgal `[通讯]` (Google)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Taxonomic Strategy RAG（TS‑RAG）和 Debate State Representation（DSR）两项系统干预，用以消除代理说服任务中的语义泄漏和问题漂移，从而降低多步推理的累积错误。

**💡 创新点**

创新点在于将抽象逻辑漏洞映射到离散类别瓶颈并通过结构检索动态获取跨域策略蓝图，构建了一个“能力桥梁”使轻量级模型可在不对话脚本的前提下对抗更强模型。

**🔧 技术方法**

使用了层次化策略记忆（Tactical Blueprints → Domain Heuristics → Global Directives）、离散类别检索、基于 JSON 的 CoT 约束、DSR 诊断、以及传统 RAG 与 TS‑RAG 的对比实验。

**📊 数据集**

基准数据集为 r/ChangeMyView，按 100 个细分子域合并为 10 个宏观域进行零样本跨域评估，并在完全未见域上测试。

**📈 对比分析**

通过与 Base LLM、Self‑Routed Heuristics、Naive Semantic RAG、Oracle Human Trace 等基线对比，TS‑RAG 在 win‑rate、平均轮数和最终说服分数上均有显著提升，尤其在轻量级模型与强模型对抗时提升至 78.5% win‑rate，平均轮数显著下降。

**⚠️ 局限性**

局限性包括：对小模型的推理开销大，需大量手工构建策略记忆，且对高度主观或无结构化话题的迁移性尚待验证。

---

## 82. A Zeroth-Order Deep Learning Method for Fully Nonlinear Parabolic Partial Differential Equations with Unknown Coefficients

**arXiv ID:** 2606.24999 | [PDF](https://arxiv.org/pdf/2606.24999v1)

**作者:** Yanwei Jia `[一作]` (Chinese University of Hong Kong), Xun Yu Zhou `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种零阶深度学习方法，用模型无关、数据驱动的方式求解全非线性抛物型偏微分方程（PDE）及其未知系数的情形；

**💡 创新点**

创新点在于引入代表-再学习（representing‑then‑learning）范式，利用零阶导数估计器（ZOD）直接学习解的梯度与 Hessian，从而避免传统自动微分的误差累积，并首次给出了在 (加权) Sobolev 空间中关于误差、样本复杂度与偏差‑方差权衡的理论分析；

**🔧 技术方法**

核心技术包括：零阶导数估计器、近似值迭代（Picard 迭代）、深度神经网络三网结构（分别逼近值、梯度、Hessian）、Feynman–Kac 随机表示、弱/强模拟器框架及其方差降低技巧、统计学习理论与 Rademacher 复杂度分析；

**📊 数据集**

使用的是由黑盒模拟器产生的合成轨迹数据，针对线性、半线性与全非线性抛物 PDE 的解析解（无真实公开数据集）进行实验验证；

**📈 对比分析**

与深度 Galerkin、PINN、深度 BSDE、深度 Picard 迭代等现有方法进行对比。实验显示 ZOD 在价值函数、梯度以及 Hessian 上均取得更低的相对均方误差，尤其在 Hessian 误差上相较基线提升十倍以上，且整体计算时间相当或略低；

**⚠️ 局限性**

局限性包括：需要可访问弱/强模拟器；弱模拟器下 ZOD 方差高，导致需要更大样本量；理论分析未覆盖优化误差；仅适用于抛物型 PDE，且对 HJB 等更一般控制问题在无模型的情形下仍有限；

---

## 83. GCT-MARL: Graph-Based Contrastive Transfer for Sample-Efficient Cooperative Multi-Agent Reinforcement Learning

**arXiv ID:** 2606.25073 | [PDF](https://arxiv.org/pdf/2606.25073v1)

**作者:** Animesh Animesh `[一作]` (Indian Institute of Technology Kharagpur), Kaushik Dey `[通讯]` (Ericsson Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于图对比学习的多任务迁移框架 GCT‑MARL，用来在协作多智能体强化学习中解决人口规模与组成差异导致的迁移难题。

**💡 创新点**

核心创新包括：1）用可变实体编码器实现人口不变的输入表示；2）引入跨任务自适应权重的多视图对比对齐损失，动态选择最有利的视图；3）两阶段训练协议与连续迁移自然兼容，提升样本效率并抑制负迁移。

**🔧 技术方法**

使用的技术主要有：多视图图对比学习（MAIL 框架）、简单图卷积、InfoNCE 损失、Q‑mix 值分解、基于掩码均值的实体编码器以及自适应视图权重学习。

**📊 数据集**

在 StarCraft Multi‑Agent Challenge（SMAC）环境中，对多种人口与类型迁移任务（如 3m→8m、8m→3m、3m→1c3s5z 等）进行评估。

**📈 对比分析**

与 DANN、CORAL、CycleGAN、LA‑QT、MAIL 等基线相比，GCT‑MARL 在 80% 目标奖励所需交互次数上平均缩短 2‑3 倍，最终胜率与基线相当甚至更好；在连续迁移序列中保持 0.78–0.91 的后向迁移性能，平均后向退化仅 -0.125。

**⚠️ 局限性**

主要局限：需要源、目标共享相同的实体类型模式，无法处理完全不同的实体结构；对异构迁移时仅部分对齐，仍受实体对齐方式限制；对齐机制依赖于固定行对齐，未实现更通用的集合级对齐。

---

## 84. Information from coincidences

**arXiv ID:** 2606.25042 | [PDF](https://arxiv.org/pdf/2606.25042v1)

**作者:** Akshay Balsubramani `[一作]` `[通讯]`, Akshay Balsubramani

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一个统一的混合相遇（mixed coincidence）等式，展示了在任意数量的先验分布与任意实指数下，期望乘积的对数既是多重相遇的概率、指数族归一化常数、最大熵目标值，也等价于 KL‑巴里中心的最优值。

**💡 创新点**

创新点在于：1）将 Renyi 熵、Renyi 散度、Chernoff 信息、PAC‑Bayes 损失等传统信息论基石收敛到单一代数等式；2）该等式支持非标准的未归一化先验、连续指标和负指数；3）由此衍生出多先验 Chernoff 代价、精准的 Sanov 分解、可计算的多先验 PAC‑Bayes 惩罚。

**🔧 技术方法**

使用的技术包括：变分推导、KL‑投影（Pythagorean 关系）、指数族和拉格朗日乘子、乘法级联（multiplicative cascades）、大偏差理论（Donsker–Varadhan、Sanov）以及数值上可计算的 Hölder 框架；所有推导均在有限样本、非 i.i.d. 与连续域上保持有效。

**📊 数据集**

实验数据集包括：①大词表（≈5×10⁴）下的开源因果语言模型（如 GPT‑2、LLaMA）产生的下一个词分布；②人类基因组中 5 000 多条基因调控区域的可调基因元件（ENCODE cCRE）与 ATAC/DNase 可及性轨迹，形成多先验预测集合。

**📈 对比分析**

对比方法：在 LLM 任务中用混合归一化与算术归一化的差值作为聚合收益指标；在基因组任务中用 log‑Z 阈值与基因元件位置的 ROC/AUC 对比；在随机分割实验中验证 m‑相遇阈值与理论 Ψ(α)·log n 的一致性。结果显示：①理论阈值与经验相符；②多先验聚合显著优于单先验或算术聚合；③在基因组上多预测器的一致性显著提升识别调控元件。

**⚠️ 局限性**

局限性：①主要证明在离散样本空间，连续情况需要进一步技术实现；②在高度相关或极度稀疏的先验集合中，数值稳定性和计算复杂度仍是挑战；③实验多集中在 NLP 与基因组两域，缺乏对更广泛统计学习任务（如强化学习、图像分类）的直接验证。

---

## 85. TRACER: Training-Free Closed-Loop Structured Inference for Traffic Accident Reconstruction

**arXiv ID:** 2606.25002 | [PDF](https://arxiv.org/pdf/2606.25002v1)

**作者:** Yanchen Guan `[一作]` (University of Macau), Zhenning Li `[通讯]` (University of Macau)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一个训练无关、闭环式的事件锚定框架TRACER，用于从稀疏异构证据重建交通事故前期运动轨迹。

**💡 创新点**

创新点包括：①事件级表征与可解释诊断；②结构化案例记忆作为弱先验；③LLM辅助的诊断与局部修正闭环；④速度-路径耦合的物理约束。

**🔧 技术方法**

技术手段：大语言模型（如Qwen‑3.5‑Flash）进行规划、检查与修正；检索式案例记忆提供先验；闭环一致性检查与局部修正；最后通过速度时间参数化实现密集轨迹。

**📊 数据集**

使用数据集：CISS‑REC，包含6,217起事故案例（5,217用于构建记忆，1,000用于测试）。

**📈 对比分析**

与多种学习基线（STGCN、LSTM、Wayformer、PC‑Crash等）对比，TRACER在几何精度(AKD)、速度一致性(AVD)、碰撞率(CR)、碰撞面精度(CSA)以及角色语义一致性(BA/RA)等指标上均取得最优或接近最优性能。

**⚠️ 局限性**

局限性：在多车辆、急剧变速或多次碰撞的场景下仍可能生成过于平滑的轨迹；对边界检测和语义匹配的依赖导致某些异常场景下定位偏差；缺乏对不确定性的量化与可解释性评估。

---

## 86. Are Tabular Foundation Models Robust to Realistic Query Distribution Shifts in Microbiome Data?

**arXiv ID:** 2606.24995 | [PDF](https://arxiv.org/pdf/2606.24995v1)

**作者:** Giulia Perciballi `[一作]` (IRD), Jean-Daniel Zucker `[通讯]` (IRD)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了表格基础模型(TFMs)在肠道微生物组数据下的查询分布偏移鲁棒性，提出了针对组成性和零膨胀的三种生物学启发式扰动，并构建了六个真实数据集的系统评估框架。

**💡 创新点**

创新点在于：①首次针对微生物组的可解释扰动设计，②将支持集保持不变、查询集扰动的in‑context学习鲁棒性评估流程，③揭示不同TFM架构对零填充、稀疏化和特征删除的特异性脆弱性。

**🔧 技术方法**

采用TabPFN、TabICL、TabDPT、ContextTab等六种最新TFMs，结合随机森林和XGBoost等经典基线，使用基于特征重要性筛选的扰动算法（高均值特征移除、零膨胀控制、零填充）。

**📊 数据集**

使用六个公开肠道微生物组数据集：Cirrhosis、IBD、Obesity、T2D（华人）、WT2D（欧洲女性）等，覆盖五个疾病，样本量从96到344不等。

**📈 对比分析**

通过5折交叉验证和AUROC比较，发现所有模型在至少一种扰动下显著下降；零填充扰动对大多数TFMs影响最大，TabDPT最脆弱；相对而言，TabPFN在稀疏化扰动上更稳健。

**⚠️ 局限性**

局限包括：仅评估了六种模型，未探索更大支持集或多任务场景；扰动仅在查询集实施，未考虑训练阶段的不匹配；特征重要性基于统计测试，可能与生物学意义不完全一致。

---

## 87. Energy Efficient Scheduling of AI/ML Workloads on Multi Instance GPUs with Dynamic Repartitioning

**arXiv ID:** 2606.25082 | [PDF](https://arxiv.org/pdf/2606.25082v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 88. FreeStory: Training-Free Character Consistency for Free-Form Visual Storytelling

**arXiv ID:** 2606.25079 | [PDF](https://arxiv.org/pdf/2606.25079v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 89. When Multi-Sensor Fusion Fails to Generalize: Cattle Posture Classification Under Animal-Level and Temporal Distribution Shift

**arXiv ID:** 2606.24986 | [PDF](https://arxiv.org/pdf/2606.24986v1)

**作者:** Leutrim Uka `[一作]` (University of Potsdam), Marina M. -C. Höhne `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究了使用颈部加速度计、胃bolus与环境传感器的多模态数据进行牛姿态（躺/站）分类，并在随机划分、按动物分组、留一动物、跨年等不同评估协议下评估模型鲁棒性。

**💡 创新点**

创新点包括：①在多种严格评估下揭示多模态模型在跨年分布偏移时性能显著下降，②发现多模态虽提升基准性能却会导致鲁棒性下降，③通过SHAP解释和分布偏移诊断揭示模型存在shortcut learning，并提供了基于鲁棒性而非仅准确率的评估框架。

**🔧 技术方法**

采用XGBoost作为主模型，Logistic回归、随机森林与LSTM做对比；利用SHAP进行可解释性分析；用PCA和域分类器诊断分布偏移；采用留一动物、跨年等验证策略。

**📊 数据集**

使用2024年（106头牛，约24.2k 分钟级标注）与2025年（95头牛，约13.9k 分钟级标注）的牧场牛数据，包含颈部加速度计、胃bolus温度/反刍信号及气象站环境数据。

**📈 对比分析**

在不同验证策略下比较模型性能：随机划分宏F1 ≈0.99，组内分组≈0.94，留一动物≈0.89，跨年≈0.49；多模态模型在内年表现最好，但跨年时仅颈部模型表现最佳，表明多模态在跨年环境下鲁棒性不足。

**⚠️ 局限性**

局限性包括：数据覆盖仅为少数夏季观察日，单一牧场与管理背景；二分类框架掩盖了站立行为内部多样性；类别不平衡与标注噪声可能影响结果；仅评估树模型，深度或变换模型鲁棒性可能不同；SHAP分析揭示关联而非因果，需进一步因果验证。

---

## 90. Auto-Configured Explainable Graph Neural Networks for Multi-Site Pollution Prediction

**arXiv ID:** 2606.24978 | [PDF](https://arxiv.org/pdf/2606.24978v1)

**作者:** Abdelkader Dairi `[一作]` (University of Science and Technology of Oran-Mohamed Boudiaf), Ying Sun `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文开发了一种基于图神经网络的多站点细颗粒物预测框架，结合混淆矩阵动态构造图结构、混合能量距离与Huber损失的混合优化以及可解释性方法进行性能评估。

**💡 创新点**

创新点包括：①利用监督学习产生的混淆矩阵来自动生成空间关系图，从而摆脱预设静态邻接；②引入能量距离与Huber损失相结合的混合损失，解决深度模型梯度消失问题；③将GNNExplainer和PGExplainer应用于特征与边的重要性解释，提升模型透明度。

**🔧 技术方法**

技术栈涵盖：图神经网络（GCN、SGConv、GIN、GAT、GraphSage）、DGL框架、XGBoost分类器生成混淆矩阵、混合能量距离+Huber损失、GNNExplainer、PGExplainer；以及基准模型（kNN、RF、ET、DT、GB、Prophet、LSTM、GRU）。

**📊 数据集**

使用美国犹他州盐湖城AirU空气质量监测网络的数据，覆盖2019-2021年，包含25台监测站的PM1、PM2.5、PM10、温湿度及多种气体传感器读数。

**📈 对比分析**

在单步（1小时）和多步（3/6/9/12小时）预测任务中，GraphSage以R²≈0.998、MAE≈0.16、RMSE≈0.21等指标遥遥领先于其他GNN、传统机器学习和深度学习模型；基准模型在同一任务下R²最多也只能达到0.892。GraphSage在推理时耗时仅0.0017s，显示出高效性。

**⚠️ 局限性**

局限性：仅在单一城市、有限监测站规模（13/25站）上验证，缺乏跨地区泛化评估；未纳入风速、降雨等气象因子；关注短期（小时级）预测，长期趋势和季节性尚未探索；混合损失的超参数调优仍需进一步研究。

---

## 91. LLM Performance on a Real, Double-Marked GCSE Benchmark

**arXiv ID:** 2606.24973 | [PDF](https://arxiv.org/pdf/2606.24973v1)

**作者:** Malachy Fox `[一作]` (Medly AI), Paul Jung `[通讯]` (Medly AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对32,534份多学科GCSE模拟答卷进行双标注，并评估现成大语言模型在最小推理设置下的自动评分表现。

**💡 创新点**

首次公开构建包含手写与键入文本的多学科GCSE双标注数据集，并证明通用LLM可与人类标注者一致甚至更优。

**🔧 技术方法**

采用通用多模态提示，输入题目、评分标准与答卷（含手写图像），模型输出JSON结构化分数。

**📊 数据集**

Medly自有GCSE模拟考试数据，覆盖英语、数学、生命科学、化学、物理，共328道题。

**📈 对比分析**

用Quadratic Weighted Kappa评估模型与两位标注者的平均一致度，与标注者间一致度做对比，结果显示顶尖模型的QWK在英语0.75、数学0.86、科学0.74，均略高于人类标注者间水平。

**⚠️ 局限性**

仅基于两位标注者的共识，缺乏更广泛标注者群体；仅测试最低推理力度，未探索更复杂提示或微调的潜力。

---

## 92. Adapt Only When It Pays: Budgeted Decision-Loss Priority for Delayed Online Time-Series Adaptation

**arXiv ID:** 2606.25068 | [PDF](https://arxiv.org/pdf/2606.25068v1)

**作者:** Xibai Wang `[一作]` `[通讯]` (NeuroQuant Labs Limited), Xibai Wang (NeuroQuant Labs Limited)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ADOWIP 框架，用于在延迟标注和严格计算预算下的在线时间序列预测器自适应更新。

**💡 创新点**

创新点在于：① 使用观察到的决策损失作为更新优先级，无需反事实探测；② 采用封闭延迟队列和精确预算核算，确保计算不超预算；③ 提供可审计的更新日志和理论保证（投影 OGD 损失上界与有限样本门控选择一致性）。

**🔧 技术方法**

技术方法包括：残差适配器更新、观察决策损失优先门控、基于经验分位数的阈值判定、MSE 安全惩罚、投影 OGD 理论证明、实验审计脚本与统计检验（Wilcoxon‑Holm、块自助法）。

**📊 数据集**

主要使用数据集：公共 ETT 能耗/容量规划、UCI Bike 公共自行车需求预测、Capital Bikeshare 2018 站点重平衡；次要数据集：ETT 阈值/负载索引任务、金融 Qlib 回测、官方时间序列基准。

**📈 对比分析**

对比方法包括：never-update、always-update、fixed-period、drift-triggered、prediction-value、decision-value、observed prediction‑loss 等；在匹配计算预算的情况下，ADOWIP 的观察决策损失门控在容量规划与站点重平衡任务上显著优于基线（p<0.001），但在阈值/负载索引任务和金融任务中表现为次要或负面。

**⚠️ 局限性**

局限性：未证明动态 Regret 或非凸低秩适配器收敛；实验仅覆盖决策损失代理，未涉及真实调度或金融交易；未与官方发布的 TimeAlign/COSA/ADAPT‑Z 等模型做完整计算匹配比较；对复杂业务约束（运营商限制、服务公平性等）缺乏验证。

---

## 93. What's in an Earth Embedding? An Explainability Analysis of Location Encoders

**arXiv ID:** 2606.24997 | [PDF](https://arxiv.org/pdf/2606.24997v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 94. Bias-Controlled Primal-Dual Natural Actor-Critic: Optimal Rates for Constrained Multi-Objective Average-Reward RL

**arXiv ID:** 2606.25012 | [PDF](https://arxiv.org/pdf/2606.25012v1)

**作者:** Ankur Naskar `[一作]` (Indian Institute of Science), Vaneet Aggarwal `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种基于多级蒙特卡洛（MLMC）的原始-对偶自然演员-评论家算法，用于解决约束的多目标平均回报马尔可夫决策过程（MDP），同时控制来自标量化、演员-评论家估计和对偶更新的偏差。

**💡 创新点**

首次在平均回报设置下建立了约束的凹标量化多目标强化学习的最优收敛性，且不依赖于混合时间知识。

**🔧 技术方法**

使用了多级蒙特卡洛（MLMC）技术和原始-对偶自然演员-评论家算法。

**📊 数据集**

使用了约束的多目标平均回报马尔可夫决策过程（CMDP）作为数据集。

**📈 对比分析**

与现有的单目标和无约束的多目标方法相比，提出的方法在收敛性和约束违反率上达到了最优的O(1/√(T))，并且在没有混合时间知识的情况下也能实现。

**⚠️ 局限性**

该方法的局限性在于尚未在更一般的假设下验证所建立的保证。

---

## 95. Solving Markov Decision Processes with Future Information via MPC

**arXiv ID:** 2606.24991 | [PDF](https://arxiv.org/pdf/2606.24991v1)

**作者:** Shambhuraj Sawant `[一作]` (Norwegian University of Science and Technology), Sebastien Gros `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种在考虑未来信息的马尔可夫决策过程（MDP）中，利用参数化模型预测控制（MPC）实现最优策略的方法，并将该MPC作为结构化函数逼近器嵌入强化学习（RL）框架。

**💡 创新点**

创新点在于（1）证明了在使用增广状态并在MPC优化中一致传播未来信息时，存在参数化MPC能够精确表示MDP的最优价值函数和策略；（2）给出了可学习的MPC参数化形式（阶段成本、终端成本、动态模型及约束）并说明如何与RL梯度无缝结合；（3）通过在点质量赛车环境中演示，该方法能显著提升利用未来轨迹信息的性能。

**🔧 技术方法**

使用的技术包括：参数化MPC（含阶段/终端成本、动态模型与约束）、对MPC求解器的敏感度分析以获得梯度、PPO强化学习算法、基于MPC输出的高斯随机策略、神经网络价值函数逼近器。

**📊 数据集**

使用的数据集：自建的点质量赛车仿真环境，包含实时轨道中心线未来参考点序列（长度M=60）作为未来信息。

**📈 对比分析**

对比方法：①标准MPC（固定成本权重、简单未来信息使用），②ppo-MPC（仅学习成本权重，使用传统未来信息处理），③ppo-MPCI（学习成本权重+未来信息加权系数，采用增广状态MPC）。结果显示：ppo-MPCI在累计成本和轨迹追踪上均优于其他两者，尤其在利用未来轨迹信息进行主动调节方面表现更佳。

**⚠️ 局限性**

局限性：①需要对未来信息建模为可在MPC预测中递归传播的增广状态，若未来信息高维或更新机制复杂，建模成本高；②MPC求解与梯度计算对计算资源要求较大，限制了大规模或实时应用；③在更复杂环境下，仅使用线性加权未来信息的表示可能不足，需更丰富的潜在表示或更强大的价值函数网络。

---

## 96. A Single Stepsize Suffices for Unprojected Linear TD(0): Simultaneous Robust and Fast Rates via Polyak--Ruppert Averaging

**arXiv ID:** 2606.24981 | [PDF](https://arxiv.org/pdf/2606.24981v1)

**作者:** Wei-Cheng Lee `[一作]` (King Abdullah University of Science and Technology), Francesco Orabona `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在马尔可夫采样下，对线性TD(0)算法给出无投影、单一步长规则的高概率收敛分析，并证明了迭代序列的有界性及Polyak–Ruppert平均的稳健和快速收敛率。

**💡 创新点**

提出了基于Poisson方程的自我约束归纳方法，在不依赖曲率参数ω且无投影的条件下实现了高概率有界性和同时获得稳健(1/√T)与快速(1/(ωT))收敛速率；同时给出了单一步长η_t∝1/(log t √t)的完整理论支持。

**🔧 技术方法**

Poisson方程工具、马尔可夫链混合性分析、马尔可夫噪声的马尔可夫-正则分解、Pinelis不等式等概率集中技术。

**📊 数据集**

无实验数据，全部为理论分析。

**📈 对比分析**

与已有需要投影或曲率信息的TD(0)结果对比，证明在同样步长下能获得更强的高概率收敛保证，稳健率与快速率同时成立，且不需要事先知道混合时间或ω；理论上与最优下界相距仅多一个对数因子。

**⚠️ 局限性**

对常数c的取值仍需依赖混合时间与其他未知常数；对θ*范数的依赖导致收敛速率的|θ*|^2因子；在曲率极小的情况下，快速率的ω依赖仍不如某些数据丢弃方法；缺乏实验验证。

---

## 97. CKM-Driven Communication-Aware UAV Intelligent Trajectory Optimization for Urban Inspection

**arXiv ID:** 2606.24979 | [PDF](https://arxiv.org/pdf/2606.24979v1)

**作者:** Yang Xiaomeng `[一作]` (Nanjing University of Aeronautics and Astronautics), Wu Qihui `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于通道知识图（CKM）的无人机城市巡检路径规划框架，联合图注意网络（GAT）与软演员批评（SAC）实现通信感知的航迹优化。

**💡 创新点**

创新点在于将扩散模型用于稀疏观测的时间累计CKM重建，并将全局节点排序与连续轨迹控制分层协同，显著降低飞行距离的同时提升通信质量。

**🔧 技术方法**

采用扩散式生成模型重构CKM，图注意网络解决TSP节点排序，软演员批评实现对速度、加速度与通信约束的连续轨迹控制。

**📊 数据集**

实验使用基于Raymobtime的城市射线追踪数据集，构造64×64网格的RSS通道图。

**📈 对比分析**

与距离优先TSP、TSP+SAC、TSP-A*、随机顺序SAC和纯SAC四种基线比较，GATSAC在保持较低路径长度的同时实现了最优的最低RSS（通信质量），效果明显优于其他方法。

**⚠️ 局限性**

局限性包括对CKM构建的先验稠密数据依赖、扩散模型与GAT训练成本高、假设通道准静态且未考虑动态障碍物与三维空域等实际复杂性。

---

## 98. Don't Go Breaking My LLM: The Impact of Pruning Attention Layers on Explanation Faithfulness and Confidence Calibration

**arXiv ID:** 2606.24970 | [PDF](https://arxiv.org/pdf/2606.24970v1)

**作者:** Pietro Tropeano `[一作]` (University of Copenhagen), Christina Lioma `[通讯]` (University of Copenhagen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了在大型语言模型中移除注意力层的剪枝对模型解释可信度（faithfulness）和置信校准（confidence calibration）的影响。

**💡 创新点**

创新点在于首次揭示剪枝在保持精度的同时，可能导致解释可信度和置信校准显著下降，且这两者的变化往往与精度误差不一致。

**🔧 技术方法**

采用基于输入‑输出余弦相似度的层重要性度量进行层剪枝，并使用 LIME 与 Kernel SHAP 的归因来评估 faithfulness（comprehensiveness 与 sufficiency），以及 Expected Calibration Error (ECE) 来评估校准。

**📊 数据集**

实验覆盖五种 7B/8B 级别的开源 LLM（Mistral、Llama‑2、Llama‑3、Qwen‑3 4B/8B）以及八个公开数据集（ARC Easy/Challenge、OpenBookQA、BoolQ、RTE、TweetEval、Rotten Tomatoes、SST‑2）。

**📈 对比分析**

与未剪枝模型对比，精度下降一般小于 0.06（最多 0.25），但 faithfulness 指标（comprehensiveness、sufficiency）普遍下降且波动大；置信校准（ECE）在大多数模型中趋向恶化，尽管某些模型如 Qwen‑3 在情感分析任务上出现轻微改善；可解释性（F1、Precision、Recall、IoU）相对稳定。

**⚠️ 局限性**

局限性包括仅考虑注意力层剪枝、仅针对解码器 LLM、样本量有限（每集 200 条），高计算成本、未覆盖其他剪枝策略或归因方法，且对提示设计的敏感性未作系统评估。

---

## 99. Are We There Yet? Exploring the Capabilities of MLLMs in Assistive AI Applications

**arXiv ID:** 2606.25084 | [PDF](https://arxiv.org/pdf/2606.25084v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 100. Retrieval-Augmented Personalization with Foundation Models for Wearable Stress Detection

**arXiv ID:** 2606.24985 | [PDF](https://arxiv.org/pdf/2606.24985v1)

**作者:** Louis Simon `[一作]` (Sorbonne University), Mohamed Chetouani `[通讯]` (Sorbonne University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在可穿戴压力检测任务中，提出一种基于检索增强的轻量化个性化模型，该模型利用冻结的跨域基础模型检索目标用户历史相似模式并生成个人化嵌入来调制Transformer的表示。

**💡 创新点**

创新点在于：①无需任何标签用户数据即可进行个性化；②使用跨域基础模型作为检索后端，避免昂贵的自监督预训练；③通过FiLM调制实现轻量级个性化；④对冷启动、跨数据集检索等场景进行系统评估。

**🔧 技术方法**

技术包括：CNN+Transformer轻量编码器；跨域时序/语音基础模型（Chronos、MOMENT、HuBERT）进行嵌入检索；SetTransformer池化检索集合；FiLM层调制；多任务交叉熵损失自适应平衡；LOSO交叉验证与线性混合模型统计检验。

**📊 数据集**

使用WESAD（15名受试者的EDA、BVP、温度、加速度）作为主要实验集；并在K‑EmoCon数据集上做跨数据集检索验证。

**📈 对比分析**

与随机森林、Transformer基线、线性探测与少量标签微调等方法比较；检索增强个性化模型在准确率和宏F1上分别提升约+3.9%和+4.8%，达到与1%标签微调相近的水平；在冷启动下，混合检索表现接近完整用户检索，交叉数据集检索则与无个性化基线相近。

**⚠️ 局限性**

局限包括：仍需一定量的无标签用户历史，完全冷启动时效果下降；个性化收益在部分用户仍有限；使用大型基础模型导致模型体积与计算资源限制；实验仅在WESAD上进行，跨域推广需要更多验证。

---

## 101. The cognitive, affective, and behavioral expression of self-stigma among people who use drugs in online substance use communities

**arXiv ID:** 2606.25143 | [PDF](https://arxiv.org/pdf/2606.25143v1)

**作者:** Layla Bouzoubaa `[一作]` (Drexel University), Rezvaneh Rezapour `[通讯]` (Drexel University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了十维自我污名化指标代码表，并在Reddit药物使用社区的大规模帖子中检测并量化这些指标的出现频率、共现关系和时间演化模式。

**💡 创新点**

首次将自我污名化分为认知、情感和行为三大域，细化为十个可在自然语言中识别的指标，并揭示行为指标往往先于认知情感指标出现，且大多数指标随时间保持稳定，仅悲观/自我挫败感逐渐增强。

**🔧 技术方法**

采用人工一致性编码与GPT‑4.1‑mini大语言模型相结合的混合方法，实现了指标自动标注；同时使用 Cohen’s κ、相对风险、Odds Ratio、GEE、Fisher’s exact 等统计手段进行共现和时间序列分析。

**📊 数据集**

使用包含 1,660 名长期活跃的英语Reddit用户，共 72,115 条主题发帖（2006‑2025 年），其中 3,838 条包含自我污名化表达。

**📈 对比分析**

对比人工编码与模型分类，模型在自我污名化判定上达到 κ=0.730、F1=0.802；在指标检测上 κ 在 0.6–0.7 之间；共现分析表明核心与行为指标的关联显著（OR≈4.65），时间分析显示行为指标先出现，只有悲观/自我挫败感显著随时间上升。

**⚠️ 局限性**

局限包括：只分析公开帖子且缺乏用户人口学信息，时间序列以帖子顺序为准而非真实时间，样本偏重长期活跃者及阿片类社区，模型召回率高但精确度中等，导致自我污名化频率估计可能偏高；无法确认语言出现与心理过程同步。

---

## 102. Model checking in finite fields and finite groups

**arXiv ID:** 2606.25088 | [PDF](https://arxiv.org/pdf/2606.25088v1)

**作者:** Samuel Braunfeld `[一作]` `[通讯]`, Samuel Braunfeld

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

在有限域、有限群和循环群（尤其是素数幂阶）上研究一阶和MSO模型检验的固定参数可解性与不可解性；证明有限域的一阶模型检验是固定参数可解的，并给出时间上界；证明在E≠NE假设下，循环群（从而有限域）上的MSO模型检验不是XP可解的；证明有限域在MSO中是可有限公理化的，从而不存在MSO意义下的伪有限域。

**💡 创新点**

首次将Ax关于伪有限域的可量化消解结果用于证明有限域的一阶模型检验可解；首次给出在E≠NE假设下，循环群（和有限域）上的MSO模型检验不可解的证明；首次展示有限域在MSO中可有限公理化，消除伪有限域存在的可能。

**🔧 技术方法**

使用模型论中的可量化消解、Feferman‑Vaught定理、阿克斯理论、伪有限域完整性与递归可公理化、图形理论中的树宽与团宽、以及对循环群结构的子群与格子定义等技术。

**📊 数据集**

无实验数据集，研究纯理论复杂性与结构定理。

**📈 对比分析**

与传统图类（树宽、团宽）中的Courcelle定理、Seese定理等对比，证明有限域在一阶层面保持可解，而MSO层面在E≠NE假设下与一般图类的不可解性质相似；复杂度上，一阶可解给出f(|ϕ|)k³的运行时间。

**⚠️ 局限性**

结论依赖于E≠NE假设；对MSO不可解性的证明使用了特定的循环群结构，可能不适用于更一般的有限群；有限域的可有限公理化仅在MSO层面成立，未说明其在实际模型检验工具中的实现难易。

---

## 103. fARfetch: Enabling Collocated AR-HRC in Large Visually Diverse Environments with VLM-Driven AR Content Adaptation

**arXiv ID:** 2606.25162 | [PDF](https://arxiv.org/pdf/2606.25162v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 104. Transferability for General Reasoning: An Automated Curriculum for Multi-Domain RLVR

**arXiv ID:** 2606.25178 | [PDF](https://arxiv.org/pdf/2606.25178v1)

**作者:** Yongjin Yang `[一作]` (University of Toronto), Zhijing Jin `[通讯]` (University of Toronto)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在多域强化学习中提出一种基于多臂赌博机的在线课程表，利用本地可学习性与跨域梯度方向相似度两种信号来动态分配训练域

**💡 创新点**

创新点在于首次将跨域梯度相似度（即梯度方向对齐）作为课程表的反馈信号，补充传统的可学习性信号，从而避免只关注局部提升导致的过度聚焦

**🔧 技术方法**

采用GRPO优化、随机投影降维的梯度表征、指数移动平均、交叉域余弦相似度以及温度软化的Boltzmann采样等技术

**📊 数据集**

使用GURU多域推理套件（六个域：math、codegen、logic、simulation、table、stem）以及对应的评测基准（如MATH-500、HumanEval等）

**📈 对比分析**

与随机采样、手工Math-to-Others、基于优势的可学习性Bandit（SEC）以及未RL的基线比较，实验显示在Llama与Qwen两种模型上宏平均准确率提升1.6–2.8个百分点（相对提升≈10%），在多数单域基准上排名第一

**⚠️ 局限性**

局限性包括仅适用于可验证奖励的RLVR设置，梯度相似度信号依赖于梯度信息，对非可验证或模型判定奖励的任务不直接适用；同时仅优化采样分布，未考虑优化器或奖励设计的进一步改进

---

## 105. Reward-Conditioned Attention: How Reward Design Shapes What Autonomous Driving Agents See

**arXiv ID:** 2606.25127 | [PDF](https://arxiv.org/pdf/2606.25127v1)

**作者:** Mohamed Benabdelouahad `[一作]` (National School of Artificial Intelligence), Aissa Boulmerka `[通讯]` (National School of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过对三种不同奖励配置的Perceiver强化学习自动驾驶代理进行实验，研究奖励设计如何塑造注意力分配与碰撞风险之间的关系。

**💡 创新点**

创新点在于提出了Episode内Spearman相关与Fisher z转换的注意力分析方法，并揭示奖励结构可定量改变注意力基线、产生警觉先验，甚至可逆转注意力-风险相关方向。

**🔧 技术方法**

使用了Soft Actor-Critic训练的Perceiver（Latent‑Query）编码器，计算跨注意力权重、TTC风险指标，并通过Spearman相关评估注意力与风险的关联。

**📊 数据集**

实验数据来源于Waymo Open Motion Dataset（WOMD）验证集的50条真实驾驶场景。

**📈 对比分析**

在相同网络架构与训练数据下，对比了basic、minimal和complete三种奖励配置，使用Episode内相关、注意力占比和风险响应进行比较；结果显示完整奖励提升安全警觉性并改变注意力分布，且奖励差异能导致注意力-风险相关方向相反。

**⚠️ 局限性**

局限性包括仅针对单一Perceiver编码器，结果受场景异质性影响；未证明注意力与决策的因果关系；对不同域或观测模态的泛化性尚需进一步验证。

---

## 106. RGB: RL Guided Whole-Body MPPI for Humanoid Control

**arXiv ID:** 2606.25123 | [PDF](https://arxiv.org/pdf/2606.25123v1)

**作者:** Yunsoo Seo `[一作]` (Korea Institute of Science and Technology), Yisoo Lee `[通讯]` (Korea Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用预训练的强化学习（RL）策略作为采样先验，结合模型预测路径积分（MPPI）在物理引擎中实时优化全身控制，形成可插拔的反馈控制器，在不改动RL策略的情况下实现多任务精细调节。

**💡 创新点**

①将RL策略仅用作MPPI的采样先验，避免再次训练；②引入节点（knot）参数化的MPPI，显著降低搜索维度；③通过可模块化的成本函数实现任务空间闭环反馈，提升精度。

**🔧 技术方法**

强化学习（PPO），模型预测路径积分（MPPI），节点插值（knot-based interpolation），MuJoCo 物理引擎，CPU 并行采样。

**📊 数据集**

29 关节的 Unitree G1 人形机器人仿真模型，使用 MuJoCo 环境进行数据采集与评估。

**📈 对比分析**

与纯 RL 基线在相同命令接口下对比。结果显示：横向漂移 RMSE 由 0.339 m 降至 0.022 m；前向速度跟踪误差基本不变；MPPI 通过成本项实现基底高度跟踪。有效更新率约 280 Hz，显示实时可行。

**⚠️ 局限性**

方法高度依赖预训练策略的先验，难以实现大幅度行为改动；在真实硬件中对模型误差和接触估计的鲁棒性尚未验证；需进一步研究多策略融合或自适应采样分布。

---

## 107. AeroCast: Probabilistic 3D Trajectory Prediction for Non-Cooperative Aerial Obstacles via Transformer-MDN Architecture

**arXiv ID:** 2606.25122 | [PDF](https://arxiv.org/pdf/2606.25122v1)

**作者:** Syed Izzat Ullah `[一作]` (Texas A&M University--Corpus Christi), Jose Baca `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一种名为AeroCast的概率性三维轨迹预测框架，用Transformer‑MDN预测非合作空中障碍物未来位置。

**💡 创新点**

结合Pre‑LN Transformer编码器与MDN输出，采用平移不变连续位移编码和带sigma floor的NLL+MSE训练，解决多模态预测与模式退化问题。

**🔧 技术方法**

使用Transformer自注意力、Mixture Density Network、负对数似然+均方误差正则化、预层归一化、sinusoidal位置编码、数据增强与多模态混合。

**📊 数据集**

使用结合真实Vicon记录的113条四旋翼轨迹和SynTraG生成的合成轨迹，共90,116样本，覆盖9种运动类别。

**📈 对比分析**

与四个基线（GRU‑MDN、LSTM‑MDN、BiGRU‑MDN、MLP‑MDN）在相同参数下比较，AeroCast在ADE/FDE约减50%，NLL、CRPS最低，推理时间0.1 ms。

**⚠️ 局限性**

仅在室内四旋翼和合成轨迹上评估，缺乏户外、多障碍相互作用、不同尺寸或生物障碍的验证。

---

## 108. Fifty Years of Specification Completeness: What Aviation Certification Tells AI Governance About Epoch Limits, Proof Surfaces, and the Structural Gap

**arXiv ID:** 2606.25120 | [PDF](https://arxiv.org/pdf/2606.25120v1)

**作者:** Christo Zietsman `[一作]` `[通讯]` (Nuphirho Research), Christo Zietsman (Nuphirho Research)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出将航空软件认证中的结构化治理链接、情境边界有效性和客观证据架构迁移到人工智能治理文档层面，并基于此构建 PromptQ 七原则框架，对 34 个实际治理文档进行评分，揭示当前治理文档普遍缺乏 epoch 限制和证据面定义的结构性缺口。

**💡 创新点**

创新点在于：①把航空行业成熟的结构化安全要求转移到自然语言治理文档；②提出 PromptQ 七原则（成功定义、评估标准、范围边界、数据分类、质量门槛、内部一致性、情境货币性）作为治理文档质量评估基准；③通过实证分析揭示 94% 文档低于 3.5/7 阈值，凸显行业治理结构缺口。

**🔧 技术方法**

技术主要包括：对 DO‑178C、DO‑330 标准的结构化映射、PromptQ 原则的定义与推导、基于 GSN 的证据面概念、以及对治理文档的手工评分与统计分析。

**📊 数据集**

使用的数据集为 34 个真实治理文档（治理提示、AGENTS.md 等）构成的治理-prompts‑v1 语料库，并结合对 9 个监管领域（航空、金融、医疗、核电等）标准的结构化审计。

**📈 对比分析**

比较方法为基于五原则的评分向七原则扩展后，对每个文件计算平均得分，并与预设的风险层级阈值（如高风险需 6/7）进行对照；实验结果显示 94% 文档低于 3.5/7，表明普遍存在缺陷。

**⚠️ 局限性**

局限性包括：①评估工具 PromptQ 由作者设计，评估结果可能存在自我验证偏差；②缺乏对治理文档评分与实际治理失败之间因果关系的长期实证验证；③未对不同 AI 领域（如生成模型、强化学习等）进行更细粒度的适配与验证。

---

## 109. SurveilNav: Collaborative Object Goal Navigation with Robot and Surveillance System

**arXiv ID:** 2606.25119 | [PDF](https://arxiv.org/pdf/2606.25119v1)

**作者:** Ming-Ming Yu `[一作]` (Beihang University), Jing Liu `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于机器人与监控摄像系统协同的目标导向导航框架，并构建了首个用于此类任务的多摄像室内数据集。

**💡 创新点**

创新点在于引入主动摄像机调用、联合3D/2D地图构建、VLM语义价值评估以及多视角目标验证，实现了机器人与固定监控的互补感知与决策。

**🔧 技术方法**

主要技术包括主动摄像机选择、基于CLIP的语义价值图生成、GroundingDINO+MobileSAM的开放词汇检测、点云配准合并、Fast Marching Method/SP路径规划等。

**📊 数据集**

使用了在Habitat‑Sim上构建的HM3D数据集，包含36个场景、74层楼、206台摄像机，共1000条episode，目标类别包括chair, couch, potted plant, bed, toilet, TV。

**📈 对比分析**

与多种单机器人基线（如MCoCoNav、VLN‑Game等）相比，本方法在HM3D上获得SPL最高36.4、SR 71.1，显著优于MCoCoNav SPL 29.7、SR 63.4，体现了协同感知带来的效率和成功率提升。

**⚠️ 局限性**

局限性在于对摄像机精确位姿依赖较强、未考虑动态环境和实时摄像头视角调整，以及通信带宽和时延问题需进一步研究。

---

## 110. ADM-Fusion: Adaptive Deep Multi-Sensor Fusion for Robust Ego-Motion Estimation in Diverse Conditions

**arXiv ID:** 2606.25111 | [PDF](https://arxiv.org/pdf/2606.25111v1)

**作者:** Hasan Moughnieh `[一作]`, Daniel Asmar `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种端到端的深度学习多传感器融合框架，用于自我运动估计，能够根据环境变化和传感器退化动态调节各模态权重。

**💡 创新点**

创新点包括：Adaptive Sensor Mixture-of-Experts (ASMoE) 的内容感知路由；将平移和旋转解耦成独立分支并通过交叉注意力实现信息共享；使用 Mamba 状态空间模型进行高效序列建模；以及加入平衡损失防止路由崩塌。

**🔧 技术方法**

采用了 ResNet（用于摄像头）、轻量级 CNN+RAFT 结构（雷达）、GRU+Mamba（IMU+LiDAR）等编码器；引入 ASMoE、交叉注意力、多头自注意力；使用 Huber + geodesic 损失、无偏差加权、平衡正则化。

**📊 数据集**

在 CARLA‑Loc 生成仿真数据上预训练，随后在 KITTI Odometry 真实数据上微调。

**📈 对比分析**

与 LIO‑SAM、DEEPLIO、A2DO 等传统与学习型基线比较；在三模态（LiDAR+视觉+IMU）配置下，旋转漂移显著降低，平移漂移保持竞争力；在 KITTI 上实现 60–70 FPS 的实时推理。

**⚠️ 局限性**

局限性：在模拟环境中传感器质量相对均衡时 ASMoE 影响有限，翻译漂移仍受尺度/速度误差限制；对传感器标定和校准高度依赖；若某模态长时间失效，路由仍可能过度依赖其余模态。

---

## 111. The Clinician's Veto: Navigating Trust, Liability, and Uncertainty in Autonomous AI Prescribing

**arXiv ID:** 2606.25108 | [PDF](https://arxiv.org/pdf/2606.25108v1)

**作者:** Eileanor LaRocco `[一作]` (University of Virginia), Chirag Agarwal `[通讯]` (University of Virginia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

通过问卷调查评估临床医生对自动化 AI 处方的接受度，并提出三项安全架构要求。

**💡 创新点**

将校准置信度门限、区分经验与数据不确定性、以及推理透明度作为安全处方的最小技术标准。

**🔧 技术方法**

使用基于问卷的调查方法、统计分析以及对比受访者对不同系统特征的评分。

**📊 数据集**

调查样本为 136 名美国处方医师，未使用公开医学数据集。

**📈 对比分析**

比较方法是对不同置信度门限、沟通方式和透明度情境进行安全评分，结果显示校准系统更安全，分辨不确定性偏好不同界面。

**⚠️ 局限性**

局限包括样本集中在单一医院系统、缺乏长期实测数据、技术实现细节不完整，且可能存在响应偏差。

---

## 112. Dream at SemEval-2026 Task 13: SALSA for Single-Pass Machine-Generated Code Detection

**arXiv ID:** 2606.25102 | [PDF](https://arxiv.org/pdf/2606.25102v1)

**作者:** Ruslan Berdichevsky `[一作]` (Dream Security Ltd.), Elad Ben-Zaken `[通讯]` (Dream Security Ltd.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对SemEval-2026 Task 13 Subtask A的机器生成代码检测任务，作者使用SALSA框架在大语言模型上实现二分类。

**💡 创新点**

创新点在于将分类任务转化为单词级结构化生成（SALSA），并通过平衡采样、低学习率、一轮训练的保守微调显著提升OOD泛化性能。

**🔧 技术方法**

技术手段包括Qwen3系列与Qwen2.5-72B‑Instruct等LLM，使用LoRA进行参数高效微调，配合SALSA结构化提示进行单次前向推理并抽取类标记logit。

**📊 数据集**

数据集为SemEval-2026官方训练集（500k样本，包含Python、Java、C++），并使用验证集进行模型挑选；在训练时对语言进行平衡采样。

**📈 对比分析**

与CodeBERT基线相比，零样本模型已取得0.36–0.67的OOD F1；微调后Qwen2.5-72B‑Instruct在OOD上达到0.789 F1，显著优于基线0.305和其他模型。

**⚠️ 局限性**

局限在于对小模型的微调效果有限，仍需进一步提升指令对齐和多语言、域的泛化能力。

---

## 113. How Modular Is a Frontier Mixture-of-Experts? A Pre-registered Causal Test in Which Apparent Expert Modularity Mostly Dissolves

**arXiv ID:** 2606.25092 | [PDF](https://arxiv.org/pdf/2606.25092v1)

**作者:** Tony Salomone `[一作]` (Transformer Lab), Ali Asaria `[通讯]` (Transformer Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对Command A+模型进行专家族级因果消融实验，检验各预注册专家族是否具备选择性模块性。

**💡 创新点**

提出了预注册、大小匹配随机对照、四种评价指标和独立语料库的因果模块性测试框架，并揭示模块性高度依赖测量条件。

**🔧 技术方法**

采用专家掩码消融、路由质量矩阵记录、Bootstrap置信区间、四种任务与语言概率指标的评估技术。

**📊 数据集**

主要使用Command A+内部路由探针语料、FLoRes‑200、独立Wikipedia、MATH‑500、HumanEval、MMLU‑Pro等数据集。

**📈 对比分析**

通过与Qwen3‑30B‑A3B正向对照和随机专家对照比较，发现只有阿拉伯语族在所有严苛条件下保持模块性，其他族受测量阈值影响而表现不稳定。

**⚠️ 局限性**

结果受限于单一模型、样本量有限、仅测评预注册族、量化方案覆盖不足，以及可能存在的多重指标偏差。

---

## 114. ActPlane: Programmable OS-Level Policy Enforcement for Agent Harnesses

**arXiv ID:** 2606.25189 | [PDF](https://arxiv.org/pdf/2606.25189v1)

**作者:** Yusheng Zheng `[一作]` (UC Santa Cruz), Andi Quinn `[通讯]` (UC Santa Cruz)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套可编程的 OS 级别 AI 代理策略引擎 ActPlane，用 DSL 编写、eBPF 编译并在内核执行，提供语义反馈以提升代理合规性。

**💡 创新点**

创新点在于：①将策略写入 DSL 并自动编译为 eBPF，②在 OS 级别实现跨事件、跨进程的信息流控制；③采用层级策略域实现安全隔离与动态更新；④在代理层与内核层之间提供可解释的反馈。

**🔧 技术方法**

技术包括 eBPF、信息流控制（IFC）、源到目标约束 DSL、层级策略域、内核安全检查与反馈、以及基于 Rust 的编译器和运行时。

**📊 数据集**

数据集：对 64 个流行 AI 代理项目的指令文件（CLAUDE.md、AGENTS.md）进行语句级抽取；Benchmarks 包含 190 条违规/合规轨迹、21 条 OctoBench 编码任务、361 条 OpenAgentSafety 安全任务。

**📈 对比分析**

通过与 prompt-filter、tool-regex、FIDES 等基线比较，ActPlane 在决策合规率上提升 22–31pp（最高 75.8%），检测率提高至 77%，在编码任务中的奖励提升约 10 分，安全任务中预防率达 74%；总体运行时开销低，单事件 3–69µs，完整工作负载 1.9–8.4%。

**⚠️ 局限性**

局限在于：①仅覆盖可观察的系统事件，无法处理纯语义或服务层的违规；②缺乏对协议层 hook 的支持；③需要代理先解析仓库/任务上下文，若解析失误会导致误报；④对极长会话的标签膨胀需要手工清理。

---

## 115. What Intermediate Layers Know: Detecting Jailbreaks from Entropy Dynamics

**arXiv ID:** 2606.25182 | [PDF](https://arxiv.org/pdf/2606.25182v1)

**作者:** Sofiia Nikolenko `[一作]` (LMU Munich), Shireen Kudukkil Manchingal `[通讯]` (Oxford Brookes University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种无训练的中间层熵轨迹检测方法，用来识别LLM的jailbreak攻击，利用冻结模型的中间隐藏层预测熵的动态变化来判别是否存在恶意意图。

**💡 创新点**

创新点在于发现jailbreak提示在中间层会产生结构化的熵随token位置的单调趋势（如Kendall τ、Spearman ρ、Monotonicity），而静态熵统计几乎无区分能力；并证明该信号集中在模型中间层，最终层会被抑制。

**🔧 技术方法**

技术手段包括：使用logit lens将每层隐藏状态映射到词表空间，计算每个token的预测熵；构造静态（均值、方差、最大值）和动态（Kendall τ、Spearman ρ、Monotonicity）特征；以单个前向传递完成所有特征计算；使用方向AUROC评估区分效果。

**📊 数据集**

数据集：有害提示集——AdvBench、HarmBench、StrongREJECT；安全提示集——UltraChat、WildJailbreak、JailbreakBench benign（后者用于测试鲁棒性）。

**📈 对比分析**

与静态特征对比，动态特征在Llama-3.1‑8B、Qwen3‑8B、Gemma‑7b三大模型上均达到平均AUROC≈0.94（Monotonicity）或≈0.80（τ/ρ）；最佳层位于≈50–85%深度，最终层性能显著下降；跨模型标准差低，说明特征鲁棒；但当安全集替换为结构相似的JailbreakBench benign时，AUROC降至≈0.35，表明对结构相似提示敏感。

**⚠️ 局限性**

局限性：需要白盒访问模型中间激活；对安全提示分布敏感，结构相似的安全提示会削弱区分；logit lens可能产生预测失真；在更大、更对齐或“思考”型LLM上效果未知；且方向d可能随进一步微调而变化，需要周期性重新估计。

---

## 116. EveLoad: Cognitive Workload Recognition from Event-Based Eye Movements

**arXiv ID:** 2606.25177 | [PDF](https://arxiv.org/pdf/2606.25177v1)

**作者:** Guorui Lu `[一作]` (Leiden University), Qinyu Chen `[通讯]` (Leiden University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建事件相机眼动认知工作负荷数据集EveLoad，并提出基于时空事件表示的深度学习框架来识别六级工作负荷。

**💡 创新点**

首创事件相机眼动数据集、受控网格定位任务消除空间分布干扰、细粒度六级工作负荷标签，并利用事件帧堆叠实现对眼动时空信息的编码。

**🔧 技术方法**

使用事件相机、事件帧表示、ResNet18/MobileNetV3/MobileViT骨干网络、交叉熵损失以及事件热像素去除与多帧堆叠等技术。

**📊 数据集**

EveLoad数据集：20名健康受试者、共24,000刺激、六级工作负荷（无负荷/0‑back/1‑back × 慢速/快速），提供事件流、行为响应与刺激级别标签。

**📈 对比分析**

与多种基于眼动或视觉的工作负荷识别方法对比，使用ResNet18在subject‑specific拆分下达到96.36%准确率，在mixed随机拆分下达到96.13%准确率，性能优于同类基准。

**⚠️ 局限性**

仅在健康受试者的受控实验环境中验证，缺乏在线实时评估、临床人群验证以及在更复杂交互情境中的泛化能力。

---

## 117. Elo-Disentangled Player-Style Embeddings for Human Chess via Rating-Conditioned Residual Move Model

**arXiv ID:** 2606.25176 | [PDF](https://arxiv.org/pdf/2606.25176v1)

**作者:** Jason Carlson `[一作]` `[通讯]`, Jason Carlson

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于残差的、Elo解耦的个体风格嵌入模型，结合一个强大的基准棋步预测器和一个32维的玩家嵌入向量，用以捕捉棋手的独特风格。

**💡 创新点**

创新点在于将玩家风格建模为对基准模型的残差，从而实现对实力的自动去耦，并通过学习的移动相似性核实现偏好优化，首次在单一模型中同时获得高精度预测和可解释的风格表示。

**🔧 技术方法**

技术上使用了残差网络结构、CNN移动编码器、跨层训练（Maia‑2候选、Maia‑3 logits、Stockfish评估特征）以及线性/非线性探针验证去耦效果，并在偏好优化中引入学习的相似性核。

**📊 数据集**

数据集来源于公共Lichess快速赛段的游戏记录，经过筛选得到约2000名棋手共100万条决策，训练集约200万条决策，用于基准模型和风格嵌入的学习。

**📈 对比分析**

与Maia‑2、Maia‑3及其Stockfish增强版基准模型在22,620条保留测试决策上比较，基准模型在各Elo分段的Top‑1准确率从0.51提升至0.68（相对提升33%），而单独的嵌入在提升上仅在置信区间内。

**⚠️ 局限性**

局限性包括推理时需要调用Stockfish（非纯政策）、嵌入维度过小限制了对高质量棋手细粒度风格的捕捉，以及未完成完全独立棋手的held‑out验证，且模型未学习自适应的引擎使用策略。

---

## 118. Exact Local Annotations for Regular Languages

**arXiv ID:** 2606.25172 | [PDF](https://arxiv.org/pdf/2606.25172v1)

**作者:** Faruk Alpay `[一作]` (Bahcesehir University), Baris Basaran `[通讯]` (Bahcesehir University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究正则语言在本地可检查（proof‑labeling）模型下的注解稳定性，给出统一的上界（对所有正则语言可实现对数级的编辑稳定性、对数级的重验证成本）并在某些产品几何中给出下界，提出了一个可判定的有限障碍搜索方法以验证具体的稳定性设想。

**💡 创新点**

提出了“编辑稳定性”这一新的度量（相当于在替换图上的低Lipschitz截面问题），证明了所有正则语言都有对数级的稳定性，同时在存在可编辑群商的“产品分解”中证明了对数级下界，从而揭示了常数级稳定性与正则语言之间的开放边界。

**🔧 技术方法**

使用了单词替换图、有限单子（syntactic monoid）与其商的代数工具，构造了前缀和平衡产品注解；引入了CP‑SAT、CUDA并行搜索和Lean 4形式化来进行有限障碍搜索和实验验证；利用代数证明、图论（树高度）、离散数学（集合与整数编程）实现了上界与下界。

**📊 数据集**

实验使用的主要数据集为：
- 二元群（ℤ/2ℤ）和模 3 语言；
- 小型正则语言如“包含 11”；
- 通过 CUDA 在 A100 GPU 上生成的平衡产品注解的全图（n=29）及其替换边的统计；
- 通过 CP‑SAT 对路径、宽度 2 梯形、跨度 2 超图等有限范围的注解方案进行可行性搜索。

**📈 对比分析**

与传统的动态维护模型（Trusted update）相比，注解模型不需要全局状态，侧重于外部可检查的局部一致性；实验表明平衡产品注解在对数级编辑稳定性和重验证成本方面与最优；下界实验显示在某些产品几何下，任何注解必须在根路径上至少变动对数级别的节点，证明了常数级稳定性不可达。

**⚠️ 局限性**

限制包括：
- 只给出了正则语言的对数级上界和在特定产品几何下的下界，常数级稳定性是否可行仍是开放问题；
- 下界仅在具有编辑活跃群商的“产品分解”中成立，无法推广到所有可能的注解方案；
- 实验使用的有限障碍搜索仅覆盖有限的模板族，无法证明全局不可行性；
- 只考虑单词替换（保持长度不变）的编辑，插入/删除和非均匀映射的稳定性仍待进一步研究。

---

## 119. Efficient Analytic Uncertainty Quantification for Multi-Modal Regression

**arXiv ID:** 2606.25188 | [PDF](https://arxiv.org/pdf/2606.25188v1)

**作者:** Kun Jin `[一作]` (Google), Jasper Snoek `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种分布无关的模块化框架，既能准确估计多模态回归分布，又能高效、可单前向推断地给出预测不确定性。

**💡 创新点**

创新点在于将变分贝叶斯后层（VBLL）与半参数回归（量化回归QR和分类恢复CR）相结合，推出无采样的解析式ELBO与预测密度，并通过距离保持的骨干网络实现对异常输入的安全网；同时提供可解析的方差分解与主动学习采样策略。

**🔧 技术方法**

技术包括变分贝叶斯推理、量化回归（Asymmetric Laplace 似然）、分类恢复（Softmax+Gaussian 权重后验）、谱归一化骨干、解析 ELBO 与预测分布、基于方差分解的主动学习。

**📊 数据集**

在三个大规模多模态回归基准上验证：WeChat（短视频观看比例）、KuaiRec（短视频观看时长）、Uber（行程时长）。

**📈 对比分析**

与单前向基线（Gaussian、SNGP、VBLL、DER、MDN、QR、CQR、CR等）及5倍成本的集成模型对比，QR‑VBLL 在 CRPS 上最优，CR‑VBLL 在 NLL 上最优；主动学习时，Hybrid 采样在保持单前向推断的同时，性能匹配或超越 25 次 Monte Carlo Dropout 的 BALD，显著提升数据效率。

**⚠️ 局限性**

局限在于需要手动决定使用 QR 还是 CR，缺乏自动路由；未给出严格的有限样本误差界限；仅处理一维连续目标，扩展到多维目标仍需研究。

---

## 120. The Gentle Collapse: Distributional Metrics for Continual Learning

**arXiv ID:** 2606.25165 | [PDF](https://arxiv.org/pdf/2606.25165v1)

**作者:** Ahmed Anwar `[一作]` (German Research Center for Artificial Intelligence), Andreas Dengel `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出六种基于softmax输出的连续分布度量，用于细粒度评估与干预持续学习中的灾难性遗忘；

**💡 创新点**

创新点在于把离散的准确率转化为连续信号，既可用于样本级权重化损失，又可用于短窗口趋势采样，从而提升经验回放方法的性能；

**🔧 技术方法**

技术包括定义真标签排名、置信度、混淆间隔和归一化KL等度量，并将其作为权重或采样依据；

**📊 数据集**

使用CIFAR-100（5任务）和TinyImageNet（10任务）两大基准数据集进行实验；

**📈 对比分析**

与统一经验回放基线相比，Log‑TLR权重化提升准确率约1.3个百分点、降低遗忘；在TinyImageNet上Log‑TLR趋势采样将遗忘减少约7.7个百分点，且在短窗口下仍保持稳定；

**⚠️ 局限性**

局限性包括：仅在ER基础上验证，未在更强回放方法（如DER++、iCaRL）上测试；部分度量在高难度任务上噪声较大；

---

## 121. Silent Failures in Physics-Informed Neural Networks: Parameter Poisoning and the Limits of Loss-Based Validation

**arXiv ID:** 2606.25151 | [PDF](https://arxiv.org/pdf/2606.25151v1)

**作者:** David McShannon `[一作]`, Nicholas Dietrich `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在物理参数被篡改或误配置时，Physics‑Informed Neural Networks（PINNs）训练损失仍能低却给出错误解的“silent failure”现象。

**💡 创新点**

提出了物理参数毒化（parameter poisoning）威胁模型和检测难度比 R，系统评估六种防御手段，并提出一种无重训练、仅通过后置损失曲线扫描即可揭示真实训练参数的检测方法。

**🔧 技术方法**

使用了 PINN 训练、参数扰动、残差监控、同参数集成、参数抖动集成、逆参数恢复、不可压性检查、损失升高检测等六种防御，以及后置损失曲线扫描技术；评估指标包括 L2 误差、损失值、R 比。

**📊 数据集**

三类 PDE 系统的数值基准：1‑D Burgers 方程（ν=0.01/π）、二维 Navier–Stokes 振动腔室（Re=100、400）以及二维对流‑扩散方程（D=0.01，Pe≈33.5）。基准解来自有限差分或公开 Ghia 流场数据。

**📈 对比分析**

通过训练损失与解误差的对比展示低损失但大误差的“silent failure”，六种防御在不同 PDE、不同扰动水平下表现不一，只有后置扫描在所有三种 PDE 中始终能定位真实训练参数；表格显示误差与 R 的分布，说明检测难度随 PDE 变化显著。

**⚠️ 局限性**

局限性包括：仅评估标准前馈网络，未考虑 Fourier/Neural‑Operator 等更复杂结构；随机种子波动大，实验次数有限；防御仅针对单一被污染参数；未探究 Bayesian PINN、适应性损失权重等更高级的防御方法。

---

## 122. TokenMinds: Pretrained User Tokens and Embeddings for User Understanding in Large Recommender Systems

**arXiv ID:** 2606.25147 | [PDF](https://arxiv.org/pdf/2606.25147v1)

**作者:** Qingyun Liu `[一作]` (Google DeepMind), Xinyang Yi `[通讯]` (Google DeepMind)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在YouTube工业级推荐系统中，提出了PUS框架，利用预训练LLM的encoder-decoder结构同时生成离散的Semantic ID（SID）用户令牌和稠密用户嵌入，实现了对用户兴趣的多维度表征。

**💡 创新点**

核心创新包括：①双输出架构实现SID令牌与稠密嵌入并行生成；②将Semantic ID从物品迁移到用户，使表示更具语义可解释性；③跨场景（长短视频）统一模型与多上下文解码；④异步生成+缓存服务，降低实时计算成本。

**🔧 技术方法**

技术手段包括：Gemini V1.5预训练LLM + CPT + SFT；encoder-decoder网络（MoE解码器）；RQ‑VAE生成SID；beam search多候选令牌；离散与连续表示映射（前缀映射、ngram/LE）；异步User Behavior Service (UBS) 缓存；以及多场景条件标记。

**📊 数据集**

数据集为YouTube用户行为日志，包括长短视频观看历史、搜索查询以及相关交互特征，覆盖数十亿用户，持续在线更新。

**📈 对比分析**

方法通过离线Recall@10、在线A/B测试评估。离线实验表明多目标采样、前瞻窗口和SID截断对准确率提升显著；上线后，结合嵌入+令牌在SFV表面可提升Engaged Users +0.11%与Satisfied Engagement +0.62%；跨场景模型训练成本下降50%、服务成本下降31%，且核心指标保持不变。

**⚠️ 局限性**

局限性包括：①依赖大规模预训练LLM及高昂的训练/推理成本；②SID令牌在序列细粒度上仍可能不足，需进一步细化；③异步缓存机制对时效性有一定延迟，可能不适用于极需实时的应用；④在非YouTube或其他业务场景的通用性尚未验证。

---

## 123. Memory Retrieval in Visuomotor Policies for Long-Horizon Robot Control

**arXiv ID:** 2606.25136 | [PDF](https://arxiv.org/pdf/2606.25136v1)

**作者:** Rutav Shah `[一作]` (University of Texas at Austin), Roberto Martín-Martín `[通讯]` (University of Texas at Austin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于Transformer的视觉运动策略，结合VLM引导的VQA监督和top‑k稀疏注意力，实现了在长时限、部分可观测环境中的任务执行。

**💡 创新点**

创新点在于：①利用视觉语言模型生成的视频问答对记忆检索进行监督，消除无关信息的注意；②通过top‑k稀疏注意力限制记忆使用，降低闭环漂移；③将动作预测与VQA联合端到端训练，使检索既关注任务相关信息又保持对控制的根本性。

**🔧 技术方法**

采用Transformer注意力、视觉语言模型（VLM）先验、视频问答生成多阶段管线、稀疏top‑k注意力、离线模仿学习和联合训练技术。

**📊 数据集**

在ReMemBench长时限操纵任务基准以及两个机器人平台（固定底座机械臂与移动机械臂）上收集的五个真实任务数据集上进行评估。

**📈 对比分析**

与SAM2Act++、ReMemBer、手工特征、Scene Memory Transformer、Token Merging等基线对比，平均提升约19%（相对于标准Transformer），VQA监督提升10%，top‑k稀疏提升9%，在不同记忆需求任务中表现领先。

**⚠️ 局限性**

局限在于：VQA生成流程仍需手工设计；top‑k参数固定，未实现自适应检索；大规模记忆导致检索延迟；跨任务或跨回合的长期记忆检索尚未解决；对VLM先验的依赖可能引入错误。

---

## 124. Intractability of Hilbert's Nullstellensatz implies algebraic hardness of permanent

**arXiv ID:** 2606.25121 | [PDF](https://arxiv.org/pdf/2606.25121v1)

**作者:** Peter Bürgisser `[一作]` `[通讯]` (Technische Universität Berlin), Peter Bürgisser (Technische Universität Berlin)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究了Blum-Shub-Smale模型下的P-NP分离猜想与Valiant代数模型下的P-NP分离猜想之间的逻辑关系，比较了Hilbert的Nullstellensatz问题与给定复矩阵的永久性评估问题。

**💡 创新点**

证明了在Blum-Shub-Smale模型下的P-NP分离猜想蕴含Valiant模型下的P-NP分离猜想，涵盖了统一和非统一模型的常数自由设置。

**🔧 技术方法**

使用了Blum-Shub-Smale模型和Valiant的代数模型，涉及常数自由的算术电路和复杂度类的定义。

**📊 数据集**

没有具体提到使用的数据集，但涉及的主要问题是多项式方程的可行性和永久性评估。

**📈 对比分析**

通过比较不同的计算模型，证明了在Blum-Shub-Smale模型下的分离猜想与Valiant模型下的分离猜想之间的相互影响，性能分析表明在某些条件下可以有效地进行计算。

**⚠️ 局限性**

在正特征情况下，是否存在类似的结果仍不清楚，且在复杂度类的比较中，如何处理常数的使用和电路的构造仍然是一个开放问题。

---

## 125. An iterative energy-based multimodal transformer for joint retrieval of wheat soil moisture, leaf area index, and plant height from Sentinel-1 and Sentinel-2 time series

**arXiv ID:** 2606.25174 | [PDF](https://arxiv.org/pdf/2606.25174v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 126. Ambulance: saving BFT through racing

**arXiv ID:** 2606.25099 | [PDF](https://arxiv.org/pdf/2606.25099v1)

**作者:** Neil Giridharan `[一作]` (University Of California Berkeley), Grzegorz Prusak `[通讯]` (Sei Labs)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出一种新的拜占庭容错状态机复制协议（称为Ambulance），通过协议裁定的竞赛机制检测并恢复慢速节点，从而在正常情况下保持低延迟和高吞吐量，同时在出现慢速/失败时实现快速恢复。

**💡 创新点**

核心创新是“协议裁定的竞赛”机制：将领导者与其它节点通过协议步骤竞争，而非时间，既合作又能产生前进进度，克服传统基于超时、异步与hedging方法的缺点。

**🔧 技术方法**

技术细节包括PBFT式的非伪造证书、两阶段全局通信与三阶段线性通信、三步恢复路径、阈值签名用于随机选举、Motorization数据层与管道化、以及在Rust/Tokio/RocksDB上的实现。

**📊 数据集**

实验使用无操作（512字节随机）事务作为工作负载；生产评估基于Sei公链真实负载；未使用外部公开数据集。

**📈 对比分析**

与Autobahn（timeout‑based）、ParBFT2（hedging）和SMVBA（asynchronous）比较；在AWS EC2上4节点同步实验，吞吐量214k tx/s、延迟205ms；慢速1–10秒时峰值延迟比Autobahn低1.6–3.0倍、ParBFT2低1.7–3.1倍；在Sei生产环境p99延迟从1.27s降至0.662s，常规延迟保持不变。

**⚠️ 局限性**

局限性包括：阈值签名和恢复路径的性能开销尚待进一步优化；对大规模节点数下网络分区和多视图恢复的扩展性未充分验证；缺乏针对恶意网络干扰的安全性分析；实验未覆盖极端高并发或高延迟网络环境。

---

## 127. Scheduling with Testing: Competitive Algorithms for Minimizing the Total Weighted Completion Time in the Adversarial Model

**arXiv ID:** 2606.25166 | [PDF](https://arxiv.org/pdf/2606.25166v1)

**作者:** Felix Buld `[一作]` (Technical University of Munich), Andreas S. Schulz `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了在单机及同质并行机上，针对含测试的调度问题，以最小化加权完成时间为目标的确定性与随机化竞争算法；

**💡 创新点**

首次实现了该问题的常数竞争比（weighted case），并在保持无权重下已知最佳上界的同时，扩展到任意权重的情形；

**🔧 技术方法**

采用了加权(α,β)-PCP规则、加权黄金轮询规则以及对测试决策与任务优先级的严谨分析，利用对偶贡献和优先级排序证明竞争比；

**📊 数据集**

无实验数据，全部为理论分析与证明；

**📈 对比分析**

与之前最优的无权重算法相比，确定性竞争比从已知的3.2361下降到2.3166（单机）/2.7763（并行）；随机化竞争比从2.8307下降到2.1523（单机）/2.5110（并行）；

**⚠️ 局限性**

竞争比与下界之间仍有明显间隙，未证实是否存在权重影响的本质差距；算法参数对不同机器数的适用性需进一步优化；

---

## 128. TRUSTMEM: Learning Trustworthy Memory Consolidation for LLM Agents with Long-Term Memory

**arXiv ID:** 2606.25161 | [PDF](https://arxiv.org/pdf/2606.25161v1)

**作者:** Tianyu Yang `[一作]` (AI Center-Mountain View, Samsung Electronics), Srinivas Chappidi `[通讯]` (AI Center-Mountain View, Samsung Electronics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出TrustMem框架，用于在LLM智能体的长期记忆中实现可可信的记忆整合；

**💡 创新点**

创新点在于引入Memory Transition Verifier对每一步记忆更新进行覆盖、保留、忠实度三维评估，并通过Transition‑Ranked GRPO在相同状态下对候选更新进行排名，以提供细粒度的监督；

**🔧 技术方法**

采用大型语言模型（Qwen3‑4B）配合强化学习GRPO、分层奖励设计（覆盖、执行、效率、内容特异性、验证奖励）以及基于verifier的偏好排名；

**📊 数据集**

使用MemoryAgentBench、HaluMem（Medium）以及Mem‑α验证集（8个来源合成的训练池）进行评估；

**📈 对比分析**

与长上下文、RAG‑Top2以及多种RL/系统级记忆方法相比，TrustMem在Mem‑α验证集上平均得分提升至66.3（比Mem‑α高4.4点），在MemoryAgentBench上平均得分提升至65.7（比Mem‑α高6.5点），在HaluMem提取任务上F1提升至69.45（比前沿方法高12.14点），同时在遗漏、腐败、幻觉等三类错误率上分别下降40.1%、79.1%、50.0%；

**⚠️ 局限性**

目前仅针对文本记忆进行实验，缺乏多模态（图像、视频等）场景的验证和适配。

---

## 129. A Framework for Directed Hypergraph Signal Processing via tensor t-SVD

**arXiv ID:** 2606.25112 | [PDF](https://arxiv.org/pdf/2606.25112v1)

**作者:** Carlos Mundo-Levano `[一作]`, Gonzalo R. Arce `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种统一的有向高阶图信号处理框架（DHGSP），能同时处理方向性与高阶（多边）关系，并实现无损傅里叶变换与拓扑定位的平移算子。

**💡 创新点**

创新点包括：①基于B-超边分解的可识别、无交叉的有向超图邻接张量；②利用t-product自伴随拉伸实现的可逆、Parseval保持的有向超图傅里叶变换（t‑DHGFT）；③构造的平移算子严格遵循有向拓扑，保证信号传播的局部性。

**🔧 技术方法**

核心技术：t-product张量代数、t-SVD、B-超边分解、自伴随拉伸（self‑adjoint dilation）以及基于t‑SVD的频域阈值去噪。

**📊 数据集**

使用真实交通网络数据集 Macheng（中国，153个节点），通过构造有向超边来构建实验图。

**📈 对比分析**

与无向GSP、标准有向GSP以及无向超图GSP做对比，采用低通硬阈值去噪实验。结果显示，DHGSP在所有频率保留比例下均取得更低的平均绝对误差，尤其在中等带宽区（约50–60%）表现最为显著。

**⚠️ 局限性**

局限性包括：目前仅验证了去噪任务，未探索采样、压缩或聚类等其他应用；自伴随拉伸会增加维度，计算成本随超图大小上升；B-超边分解对非均匀超边的处理仍需进一步理论与实验支持。

---

## 130. Neural Network Quantization by Learning Low-Loss Subspaces

**arXiv ID:** 2606.25087 | [PDF](https://arxiv.org/pdf/2606.25087v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 131. Beyond Shapley: Efficient Computation of Asymmetric Shapley Values

**arXiv ID:** 2606.25103 | [PDF](https://arxiv.org/pdf/2606.25103v1)

**作者:** Ezequiel Companeetz `[一作]` (Universidad de Buenos Aires), Sergio Abriola `[通讯]` (CONICET)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了引入因果信息的Shapley值变体——Asymmetric Shapley Values（ASV），并提供了在多种因果图结构下高效计算ASV的方法。

**💡 创新点**

创新点在于证明在Naïve Bayes等分布下，ASV可多项式时间计算；提出利用顶点排序的等价类降低计算量；并给出了基于随机采样的近似算法。

**🔧 技术方法**

采用因果图顶点排序、等价类分组、DFS计数拓扑序、随机游走采样以及贝叶斯网络推理等技术。

**📊 数据集**

实验使用了来自Bayesian Networks Repository的两个真实贝叶斯网络（BIF1、BIF2），以及人工生成的平衡二叉树、Naïve Bayes树和合并路径树。

**📈 对比分析**

与枚举所有拓扑序的基线相比，等价类方法在计算量和运行时间上提升数十倍；采样方法在1000样本下平均相对误差约8%，并且采样时间随样本数增长不线性。

**⚠️ 局限性**

局限在于等价类数量仍会在大型或高连通DAG中指数增长；近似方法依赖于采样质量和条件期望估计；对非多项式时间推理模型的适用性有限。

---

## 132. Speculative Decoding at Temperature Zero: A Scoped Safety-Invariance Screen with a 48,072-Sample Expansion

**arXiv ID:** 2606.25097 | [PDF](https://arxiv.org/pdf/2606.25097v1)

**作者:** Sahil Kadadekar `[一作]` `[通讯]` (Independent Researcher), Sahil Kadadekar (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在温度为0的贪婪推理条件下，评估speculative decoding是否会导致安全行为偏差，并提出一种行为等价检查屏障。

**💡 创新点**

引入了可复现的speculative equivalence screen，包括字节身份、TOST等价和Cohen's h效应量的三重检验；并通过多实验（对抗性DPO草稿、4bit量化草稿、不同种子、dtype）进行校准与验证。

**🔧 技术方法**

使用vLLM 0.19的speculative decoding（严格拒绝采样与典型接受），fp16/bf16执行，GPTQ 4bit量化，DPO微调草稿，统计方法包括McNemar、TOST、Cohen's h、Holm–Bonferroni校正、Wilson CI。

**📊 数据集**

采用9种安全基准（AdvBench拒绝、BBQ偏差、TruthfulQA、MMLU、ARC挑战、Jailbreak放大、能力控制），共953个提示；另外在70B规模下采集4006个样本进行验证。

**📈 对比分析**

通过配对推理比较，字节身份≥99.5%（强）或低于该阈值（中）并通过TOST ±3pp及Cohen's h<0.1的检验；在17个匹配对中全部通过，最大 |h|=0.024；核心任务25/27个TOST通过；70B规模验证表明未出现安全偏差，说明speculative decoding在该设定下安全性可接受，且能提升吞吐量。

**⚠️ 局限性**

局限性：仅针对温度0、vLLM v0.19、Llama/Qwen两大模型族、严格/典型接受方式；未测试树形变体、EAGLE/Medusa、跨框架、多轮交互以及温度>0的情况；统计功效受限（某些任务在边界导致Wald CI失效）；对新模型、不同安全基准的推广需要进一步验证。

---

## 133. Speculation at a Distance: Where Edge-Cloud Speculative Decoding Actually Pays Off

**arXiv ID:** 2606.25091 | [PDF](https://arxiv.org/pdf/2606.25091v1)

**作者:** Yuan Lyu `[一作]` (University of Victoria), Jaya Prakash Champati `[通讯]` (University of Victoria)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过闭式不等式和理论模型，对分布式推理（DSD）与同机位推理（Co-located SD）以及纯 AR 进行系统比较，阐明了 DSD 在不同指标下的优势与局限，并重点分析其在多租户服务器容量提升方面的实际价值。

**💡 创新点**

创新点在于：①首次将 DSD 的性能评估框架与 Co-located SD 直接对齐，给出明确的延迟、通信、计算与内存比较；②推导出 DSD 在 WAN 环境下的突破延迟窗口（break‑even RTT）公式，并证明仅在极窄的 RTT 范围内才能超过 AR；③提出跨客户端重叠（cross‑client overlap）条件下的容量提升比例 (1+γ·t_d/t_v)，把 DSD 重新定位为多租户容量提升方案，而非单请求加速方案。

**🔧 技术方法**

主要技术包括：闭式分析模型、记忆体绑定假设、接受率 α 与推测长度 γ 的统计表达、网络延迟与带宽的传输时间模型、同步与异步（流水线）DSD 的时间分解、FLOP 与内存占用计算、以及多租户吞吐量的比例推导。

**📊 数据集**

数据集与实验：本文并未进行新的大规模实验，而是引用了已公开的 350+ Co-located SD 试验（OPT‑66B、LLaMA‑65B 等）以及 DSSD、SLED、SpecEdge 等系统的性能报告，用于验证理论公式的准确性。

**📈 对比分析**

比较方法：将 DSD 与 Co-located SD、Cloud AR 在延迟、计算成本、内存占用、网络交互以及 API 费用等维度做对比。结果显示：①在单请求延迟上，Co-located SD 总是优于 DSD；②DSS 在满足特定 RTT 与接受率窗口内可略优于 AR；③在多租户容量上，DSS 可提升 (1+γ·t_d/t_v) 倍的并发客户数，但前提是存在足够的跨客户端重叠。

**⚠️ 局限性**

局限性：①分析基于记忆体绑定与同步假设，忽略 GPU 计算瓶颈变化；②只在低 RTT (<γ) 下异步流水线才有潜在优势；③对闭源 API 的实用性存在障碍；④容量提升依赖于高并发、跨客户端重叠，单用户场景无优势；⑤在计算受限或多 token 预测成本降低的模型中，容量增益显著下降。

---

## 134. Proactive Systems in HCI and AI: Concepts, Challenges, and Opportunities

**arXiv ID:** 2606.25149 | [PDF](https://arxiv.org/pdf/2606.25149v1)

**作者:** Nima Zargham `[一作]` (University of Toronto), Anastasia Kuzminykh `[通讯]` (University of Toronto)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并组织一次关于人机交互和人工智能领域中主动系统的研讨会，目标是建立统一的主动性概念、识别设计与评估空白并制定人本指南。

**💡 创新点**

创新点在于系统化梳理主动系统的概念与相关术语，构建跨学科的讨论框架，并通过情景设计与案例分析形成共识。

**🔧 技术方法**

采用文献综述、工作坊设计方法、情景模拟与小组讨论等方法。

**📊 数据集**

未使用传统实验数据集，主要依赖现有研究文献和专家经验。

**📈 对比分析**

通过对比已有的研究与设计原则，评估本工作坊提出的概念框架在清晰度和可操作性上的提升，但未涉及量化性能指标。

**⚠️ 局限性**

局限在于缺乏实证验证与长周期评估，受限于会议规模与时间，对不同领域的泛化性需进一步验证。

---

## 135. SwarmFly: A simulation platform for UAV swarm experiment design and validation

**arXiv ID:** 2606.25146 | [PDF](https://arxiv.org/pdf/2606.25146v1)

**作者:** Abhishek Phadke `[一作]` (Christopher Newport University), Abhishek Joshi `[通讯]` (Texas A&M University-Corpus Christi)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

开发了一个可扩展、插件化的MATLAB UAV 群体仿真与实验平台 SwarmFly，支持多种协调模式、实时仿真、故障注入、能量与碰撞检测等功能，能在无需重写核心代码的情况下快速搭建并验证群体行为；

**💡 创新点**

创新点在于提供轻量级、模块化的仿真核心和统一的插件接口，使研究者能以最小工作量插入新的行为、故障模型、分析工具，并通过八个预装插件快速形成完整的测试套件；

**🔧 技术方法**

主要技术包括MATLAB 句柄类实现的实时仿真循环、插件注册与生命周期管理、四种群体协调模式（Leader‑Follower、去中心化、异构中继、异速）、模拟 IMU 传感器与 GPS、碰撞与地理围栏逻辑、以及 3D 视图渲染；

**📊 数据集**

使用内部生成的随机种子进行实验，没有外部公开数据集；

**📈 对比分析**

通过一系列八个实验（姿态保持、风阻、模式对比、故障恢复、组合故障、能量持续、地理围栏、回归测试）验证子系统，并以形成误差、群体扩展、能量耗尽时间等 KPI 与同行工具对比，演示平台能在 10–30 Hz 下实时运行，插件间可协同工作；

**⚠️ 局限性**

局限性包括：仅支持 4 只 UAV、纯 kinematic 无惯性与阻力模型、通信仅二值距离检查、地理围栏为软性抑制、缺乏真实传感器噪声与动力学耦合、缺少多群体与硬障碍物等高级功能。

---

## 136. No 3D Matrices: A Unified Tensor-Product View of Matrix-Free Cartesian PDE Solvers

**arXiv ID:** 2606.25148 | [PDF](https://arxiv.org/pdf/2606.25148v1)

**作者:** Yong Yi Bay `[一作]`, Kathleen A. Yearick `[通讯]` (University of Illinois at Urbana Champaign)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过 Kronecker 积代数将任意三维笛卡尔网格上的离散算子拆分为一维行向量运算，提出了统一的线性循环框架，并展示了如何在不同离散方法（有限差分、Compact Padé、Galerkin、B‑spline、IGA、Collocation 等）以及 ADI 隐式时间步、快速离散化求解中实现高效的矩阵无关（matrix‑free）算法。

**💡 创新点**

创新点在于将所有常见三维算子（偏导、拉普拉斯、泊松、Helmholtz、混合导数、三维卷积等）归结为同一套 Kronecker 积公式，系统推导出其一次性实现的成本、存储、并行化三要素，并将多右端点重塑、求和因式分解和笔直子空间分解这三种生产级加速技巧结合在一起，形成一个完整的从理论到实现的“单页”实现指南。

**🔧 技术方法**

主要技术包括 Kronecker 积混合积规则与逆规则、矩阵无关一次维度线求解（Thomas 算法）、多右端点批量线核（BLAS‑3 GEMM 或带状多 RHS 内核）、求和因式分解（sum‑factorization）用于高阶谱元、快速对角化（fast diagonalization）以及笔直子空间并行划分（pencil decomposition）。

**📊 数据集**

实验数据以制造的三维 Poisson 问题为基准，使用 Dirichlet 边界条件、正弦解以及均匀格点（N = 16, 32, …, 256），没有使用真实工程数据集，而是通过规模可变的结构化网格评估算法性能。

**📈 对比分析**

通过将组装的稀疏矩阵直接求解与矩阵无关的快速对角化求解进行对比，实验表明：矩阵无关方法在存储量上相差两百万倍，运行时间在 48³ 级别后比组装方式快数十倍，且可扩展到 256³（约 1.7×10⁷）个未知数，展示了 O(N³) 计算和 O(N_x+N_y+N_z) 存储的显著优势。

**⚠️ 局限性**

局限性主要在于需要离散算子可写成三维 Kronecker 积的可分离形式；对非可分离系数、曲面坐标系或非结构网格的 PDE，线性拆分不再完全成立，必须依赖迭代或低秩近似；此外，极端高阶或非线性问题可能导致系数矩阵稠密化，影响带状求解的可行性。

---

## 137. Causality-Based Parametric Control Barrier Function for Safe Multi-Vehicle Interaction

**arXiv ID:** 2606.25134 | [PDF](https://arxiv.org/pdf/2606.25134v1)

**作者:** Yiwei Lyu `[一作]` (Texas A&M University), John M. Dolan `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在多车辆交互环境下，利用因果推理学习邻车的安全控制参数，并基于此设计自适应的影响感知安全控制器；

**💡 创新点**

引入Cross Map Smoothness（CMS）因果推理来挑选与安全约束激活相关的数据，实现对多车相互影响的显式建模，突破传统假设（同质专家、完全激活标注）的限制；

**🔧 技术方法**

使用CMS因果推理、参数化控制障碍函数（Parametric‑CBF）、径向基函数神经网络（RBFNN）和在线岭回归估计安全参数，以及基于学习的未来状态预测的影响感知控制优化；

**📊 数据集**

主要使用仿真数据：7辆车环形交叉口与单车合流场景，车辆运动受高斯噪声扰动；

**📈 对比分析**

与传统CBF-QP、基于软约束的神经障碍函数（NBF）以及原始Parametric‑CBF进行比较，实验显示在平均MSE、完成时间和相对误差等指标上均优于基线；在合流任务中平均时间缩短19.3%，在环形交叉口缩短4.9%，并且在大部分试验中误差显著低于95%阈值；

**⚠️ 局限性**

局限性包括：需预设安全半径、假设邻车采用CBF控制器、阈值（δ_c、δ_rmse）未自适应，导致极少数试验性能退化；且目前仅验证于低相对阶系统与仿真数据，缺乏真实环境验证。

---

## 138. Forget to Improve: On-Device LLM-Agent Continual Learning via Budget-Curated Memory

**arXiv ID:** 2606.25115 | [PDF](https://arxiv.org/pdf/2606.25115v1)

**作者:** Beining Wu `[一作]` (South Dakota State University), Yanxiao Zhao `[通讯]` (Virginia Commonwealth University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在边缘设备上为冻结的大语言模型（LLM）代理设计了一种基于“每字节净价值”的内存治理层，统一决定何时保留、共享和信任经验记忆。

**💡 创新点**

创新点在于：①用单一的 net‑value‑per‑byte 评分整合了保留、共享和信任三大决策；②该评分同时考虑经验价值、负迁移风险与来源可信度，天然抵御记忆中毒；③通过预算驱动（内存/能量/上行带宽）实现跨设备、跨模型的统一治理。

**🔧 技术方法**

技术细节包括：<br>• 经验价值由检索概率、条件有用性与抽象增益三项乘积估计；<br>• 负伤害由特异性与来源可信度两项构成；<br>• 用滑动 sketch 计算查询相似度；<br>• 采用稀疏化的标签无监督有用性 head 进行在线学习；<br>• 通过贪心排序实现基于预算的 Keep 与 Share；<br>• 采用阈值和冗余检测实现 Trust。

**📊 数据集**

数据集与测试环境：<br>① 任务漂移基准（ALFWorld、BabyAI、MemoryAgentBench）用于软件仿真；<br>② 真实 Jetson AGX Orin/Thor Heterogeneous 三机测试平台用于能量、内存与上行带宽实测；<br>③ 对不同 LLM backbone（Qwen2.5‑3B、Phi‑3.5‑mini、Qwen2.5‑7B）做多模型验证。

**📈 对比分析**

与多种基线（no‑curation、A‑MEM、Agentic Memory、naive‑LRU、recency、MaRS、全内存 Oracle）对比，评估指标包括任务与受害子集准确率、峰值内存、能量、上行字节、注入攻击成功率（ASR）。结果显示：<br>• 内存占用降低 2.7×；<br>• 上行量降低 2.4×；<br>• ASR 由 0.75 降至 0；<br>• 任务准确率提升至 Oracle 的 97%，同时受害子集准确率提高 0.12。<br>在真实硬件上，能量、内存和上行均降至 keep‑all 的 38–64%。

**⚠️ 局限性**

局限性与未来工作：<br>① 仅适用于冻结 LLM 规划器，无法处理参数更新型持续学习；<br>② 只验证了几类记忆中毒（PoisonedRAG、MemoryGraft、MINJA），对更高级攻击未知；<br>③ 需要预训练的有用性 head 与手工设定 λ，尽管在三种模型上表现稳健，但在极端设备或更大模型上可能需微调；<br>④ 未探讨多任务跨域迁移与自适应预算分配。

---

## 139. Neural operator-based digital twins for modeling amyloid-$β$ and tau propagation and treatment optimization in Alzheimer's disease

**arXiv ID:** 2606.25185 | [PDF](https://arxiv.org/pdf/2606.25185v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 140. Wideband Near-Field Channel Estimation Under Hybrid Compression: Cross-Subcarrier KL Covariance Fitting With OFDM Fresnel Model

**arXiv ID:** 2606.25101 | [PDF](https://arxiv.org/pdf/2606.25101v1)

**作者:** Rıfat Volkan Şenyuva `[一作]` `[通讯]` (Maltepe University), Rıfat Volkan Şenyuva (Maltepe University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于宽带 OFDM Fresnel 模型的跨子载波 KL 协方差拟合方法 WB‑CL‑KL，用于 XL‑MIMO 系统在混合模拟‑数字压缩架构下的近场信道估计。

**💡 创新点**

创新点在于：①将宽带频率依赖的 Fresnel 曲率融入 KL 拟合；②利用跨子载波信息实现数据多样性和几何多样性的分解；③在压缩域直接最小化 KL 散度，无需全阵列重建；④结合压缩域 CRB 进行性能下界分析。

**🔧 技术方法**

技术手段包括：OFDM 多子载波信号建模、频率缩放的 Fresnel 导向向量、跨子载波 KL 协方差拟合、Slepian‑Bangs FIM 推导、SVD 伪逆求解 CRB、三相多起点优化与 BPD 热启动。

**📊 数据集**

使用的数据集为基于 3GPP TR 38.901 UMi 轨迹的射程与角度场景，带宽从 100 MHz 到 800 MHz，频点 28 GHz，RF 链 8 条，单路径或两路径的多径模型，Monte Carlo 试验 600 – 2000 次。

**📈 对比分析**

与多种基准（全阵列参考、压缩域 SOMP、DL‑OMP 等）比较，WB‑CL‑KL 在 400 MHz、SNR = 10 dB 时达到 RMSE_r = 19.8 mm，接近压缩域 CRB（RMSE_r = 19.9 mm，效率 0.996）；在 3GPP UMi 部署分布下中位 SNR 9.6 dB 时仍保持 0.959 的 CRB 近似率。

**⚠️ 局限性**

局限性包括：假设理想相位移器常数幅值组合器、单路径或弱聚类近场；未考虑频率选择性混合器（如 TTD）导致的多载波相位误差；对高 SNR 或大带宽的 KL 目标景观收敛性未完整解析。

---

## 141. Power-Flexible AI Data Centers: A New Paradigm for Grid-Responsive Compute

**arXiv ID:** 2606.25098 | [PDF](https://arxiv.org/pdf/2606.25098v1)

**作者:** Chris Williams `[一作]` (Emerald AI), Brandon Records `[通讯]` (Oracle)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于 GPU 的 AI 数据中心可灵活响应电网信号的架构，并在真实环境中演示了快速调度、持续削峰、碳意识以及跨站点负载迁移等多种功率调节功能。

**💡 创新点**

创新点包括：①将传统的峰值削减延伸为实时、持续、碳与地理多维度调节；②构建统一的 grid‑interface、工作负载编排与实时功率建模三层架构；③在多站点场景下实现跨区域流量迁移，满足电网需求并提升可再生能源利用率；④通过软件定义的功率上限和作业优先级实现对数百个 GPU 的精细功率控制。

**🔧 技术方法**

核心技术包括：GPU 级功率上限接口（NVIDIA smi/DCGM）、SLURM 作业调度优先级扩展、实时电网信号接口（EPRI、National Grid），以及基于采集的第二级 GPU 与机架电力遥测的功率预测模型；配合 Emerald Conductor 平台实现统一控制。

**📊 数据集**

实验数据集：① 130 kW NVIDIA Blackwell Ultra GPU 集群，运行大语言模型微调、多模态训练、批量推理等真实工作负载；② 22 份真实电网调度事件（峰值削减、紧急削峰、碳强度信号）来自 EPRI 与 National Grid；③ 10% 推理流量迁移实验使用位于 VA 与 IL 的 80 GPU H100 集群的实时推理负载。

**📈 对比分析**

与以往仅在模拟或 CPU 环境下验证的需求响应实验相比，本文实现了 100% 事件合规、≤ 40 秒快速响应、10‑40% 持续削峰持续 2‑10 h、碳强度跟踪以及跨站点 3.1 kW 的功率迁移，表明 GPU‑AI 集群在保持关键作业 QoS 的同时，能够提供多模态、持续且大规模的电网辅助服务。

**⚠️ 局限性**

局限性：① 仅在 96‑GPU 130 kW 规模上验证，尚未验证千 GPU 级大规模部署的可扩展性；② 对作业调度的依赖性较高，若负载不具备可延迟或预检点特性，功率控制效果受限；③ 需持续维护精确的功率模型与电网信号接口，系统对异常信号和硬件误差的鲁棒性待进一步评估。

---

## 142. Training for the Model You Return: Improving Optimization for Iterate-Averaged Language Models

**arXiv ID:** 2606.25086 | [PDF](https://arxiv.org/pdf/2606.25086v1)

**作者:** Kwok Chun Au `[一作]` (Columbia University), Adam Block `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于最优控制的迭代平均拉回（Pullback Averaging Control，PAC）优化器，在训练过程中主动拉动当前参数向指数移动平均点靠拢，以提升最终平均模型的性能。

**💡 创新点**

创新点在于将迭代平均器的性能问题转化为连续时间随机二次优化的线性二次控制问题，推导出最优控制策略，并通过近似得到可实现的离散更新规则；该方法理论上在二次模型下可显著降低平均误差，并在凸情形下保持SGD收敛速率。

**🔧 技术方法**

使用了最优控制理论、线性二次调节、连续时间随机微分方程、指数移动平均、AdamW/Lookahead的改造、按坐标裁剪、可学习的拉回强度以及动态EMA权重衰减等技术。

**📊 数据集**

在实验中使用了 SmolLM2‑1.7B、Qwen3‑1.7B、Gemma3‑1B 三大 1–2B 参数模型进行监督微调；以及 GPT‑2（124M）在 FineWeb 数据集上进行从零开始预训练。

**📈 对比分析**

与 AdamW、EMA‑AdamW、学习率衰减（线性/余弦）、Schedule‑Free 等基线进行比较。PAC 在所有设置下都优于 AdamW 与 EMA‑AdamW，且在多种学习率、EMA 速率和拉回强度下表现稳健；在预训练中即使使用恒定学习率也能超过 WSD 训练，且与 Schedule‑Free 接近或略优。

**⚠️ 局限性**

局限性包括：理论推导基于简化的二次连续模型，难以完整覆盖真实 LM 的非凸性；实现需额外复制一次模型权重，导致显著内存占用；未在超过 2B 参数的模型或更大预训练规模下验证；对动量与自适应预条件的理论分析仍待完善。

---

## 143. Learning Perceptive Platform Adaptive Locomotion Controllers for Quadrupedal Robots

**arXiv ID:** 2606.25179 | [PDF](https://arxiv.org/pdf/2606.25179v1)

**作者:** David Rytz `[一作]` (University of Oxford), Ioannis Havoutis `[通讯]` (University of Oxford)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了形态专用的通用四足行走控制器，并在仿真与ANYmal硬件上评估其跨形态、跨地形的鲁棒性。

**💡 创新点**

创新点包括：①将感知仅注入到评论家（Critic）而不进入执行者（Actor）以提升噪声鲁棒性；②使用多参考机器人进行形态专用训练；③设计自适应地形课程与粒子过滤器调度，提高样本效率与泛化；④在单个计算资源下实现跨形态的零射击（zero‑shot）部署。

**🔧 技术方法**

技术手段：强化学习（PPO）+ 异步演员-评论家架构；估计器网络用于推断基线速度与形态描述；高程图感知（高度图扫描）与噪声模型；自适应地形课程与粒子过滤器；RaiSim仿真；工业级视觉感知堆栈。

**📊 数据集**

数据集：基于RaiSim生成的多形态样本（Unitree A1、Aliengo、ANYmal B/C）以及多种地形参数（斜坡、台阶、丘陵）通过随机采样与自适应课程得到；每个形态产生30个随机配置；在硬件实验中使用ANYmal并改装负载。

**📈 对比分析**

比较方法：在仿真中统计成功率 SR* 与轨迹误差；在硬件中测量速度跟踪 RMSE 与基线速度误差；对比 MorAL、MorAL_blind、MorAL+ 与 PPAL。结果显示 MorAL+ 在多形态训练下保持最高 SR*，在硬件上显著降低 RMSE；PPAL 在感知噪声下性能下降；MorAL+ 在保持部署稳定性的同时提升跟踪一致性。

**⚠️ 局限性**

局限性：在复杂台阶/垂直障碍物上仍缺乏足够的足部抬升，导致通行困难；未包含显式足部清晰奖励；对极端形态外推的鲁棒性有限；完全感知的 PPAL 对噪声敏感，需进一步提升感知去噪与信念状态融合。

---

## 144. ATMA: Length-Invariant Language Modeling via Polar Attention and Gated-Delta Compression Memory

**arXiv ID:** 2606.25156 | [PDF](https://arxiv.org/pdf/2606.25156v1)

**作者:** Habibullah Akbar `[一作]` `[通讯]` (Kreasof AI), Habibullah Akbar (Kreasof AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 ATMA 混合序列混合器，融合了长度不变的 Polar Attention 和递归的 Titans gated‑delta 内存，以解决大型语言模型在超长上下文中的困惑与检索性能折衷问题。

**💡 创新点**

创新点包括：① 三通道注意力（方向、幅度、记忆）与极值校正的 null 悬底；② 通过参与度比（inverse Simpson index）实现计数无关的方向向量；③ 结合极值校正的温度因子和极值阈值，保持注意力分布的长度不变；④ 对记忆进行 L2 正则化的 gated‑delta 递归，确保状态稳定并在长序列上保持固定范数；⑤ 对整个体系的 GPU 内核进行统一优化，显著降低内存占用和推理延迟。

**🔧 技术方法**

使用技术包括：分组查询注意力（GQA）、LFM2 门控卷积、Polar Attention（极值修正、参与度比计算）、gated‑delta Titans 内存、FlashAttention 风格的 Triton 内核、分块分页解码、原位递归更新等。

**📊 数据集**

使用的数据集为 FineWeb‑Edu（1B token 训练）、FinePDFs（单篇长文档 perplexity 评估）以及构造的 “Induction Needle‑in‑a‑Haystack” 检索任务。

**📈 对比分析**

与传统 softmax 关注、softmax+记忆以及滑动窗口注意力等基线对比，ATMA 在 2K~64K（32×）上下文长度下实现检索准确率 90%+（平均 94%）且文档 perplexity 单调下降至 1.96 nats，优于软max+记忆在极长上下文的 2.34 nats，并保持 93% 检索准确率。

**⚠️ 局限性**

局限性包括：对自定义 GPU 内核的依赖导致迁移成本高；在 64K 以上仍存在轻微的 perplexity 下降；仅针对针对于数字检索的任务验证，缺乏在更广泛生成任务上的评估；内存分配与稳定性调参仍需进一步优化。

---

## 145. Toward Low-Latency Vision-Language Models with Doubly-Correct Predictions in Egocentric Visual Understanding

**arXiv ID:** 2606.25160 | [PDF](https://arxiv.org/pdf/2606.25160v1)

**作者:** Qitong Wang `[一作]` (Dolby Laboratories, Inc.), Christopher Rasmussen `[通讯]` (University of Delaware)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了Vision‑Language模型（VLM）在第一人称视频任务中剪枝对双重正确预测（预测正确且解释正确）的影响，并提出了基于模型自身解释信息指导的剪枝方法。

**💡 创新点**

创新点在于将模型产生的空间‑时间解释掩码与权重幅度、激活信息结合，使用Hessian二阶近似与语义相似度调制的剪枝评分，从而在非均匀层级上分配剪枝比例，既保持解释与预测的一致性，又显著提升双重正确预测。

**🔧 技术方法**

主要技术包括：权重幅度+激活依赖的OBS剪枝框架、二阶Taylor展开估计损失增量、解释掩码与语义相似度融合的掩码调制、ActionCLIP编码器的后训练剪枝。

**📊 数据集**

使用的实验数据集为EPIC‑KITCHENS VISOR和EgoExo4D两大第一人称视频数据集，包含动作标签和像素级操纵物体掩码。

**📈 对比分析**

与四种基线（OMP、GMP、Kwon等、ECoFLaP）在相同剪枝比例（20–30%）下对比，实验显示该方法在保持预测可信度（PT）不变的同时，使空间IR和RR提升约10–13%，并在整体准确率上优于所有基线。

**⚠️ 局限性**

局限性包括：仅针对Encoder‑only VLM（如ActionCLIP），未验证多模态或生成式模型；剪枝依赖已生成的解释掩码，解释质量不佳时可能影响效果；以及在不同硬件/部署场景下的实际延迟尚未充分评估。

---

## 146. Hitting a Moving Target: Test-Time Adaptation for AI Text Detection under Continual Distribution Shift

**arXiv ID:** 2606.25152 | [PDF](https://arxiv.org/pdf/2606.25152v1)

**作者:** Kevin Ren `[一作]` (Cornell Tech), Nikhil Garg `[通讯]` (Cornell Tech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实验了一种基于测试时自适应（TTA）的AI文本检测框架，利用半监督学习（正负无标签）在推断时利用同质性信号，抵御持续的分布漂移（对抗性人性化、新LLM发布、时间漂移）。

**💡 创新点**

创新点在于：①将PU/PNU学习与TTA结合，首次在文本检测中引入推断时同质性作为自适应信号；②对持续分布偏移进行理论建模并给出对策；③通过对抗人性化的自动搜索与实时自适应，验证了在多种偏移下TTA显著优于传统监督方法。

**🔧 技术方法**

技术手段包括：半监督学习框架（PU、PNU、TED^n算法），DistilBERT 作为基模型，推断时自适应（利用无标签推断批量更新决策边界），阈值二分类、Best‑Bin 预估比例、LDA嵌入可视化以及对抗人性化的自动化 prompt 搜索。

**📊 数据集**

使用的主要数据集为 Cornell arXiv 摘要（2010‑2020 年）与 RAID 基准，辅以少量 SemEval 代码数据。实验中先生成 LLM 镜像文本，再通过不同策略（naïve prompt 与对抗性 prompt）产生人性化文本。

**📈 对比分析**

与传统监督检测（Cross‑entropy 训练的 DistilBERT、商用 Pangram 等）以及无监督方法进行比较。实验显示，监督模型在对抗性文本和新 LLM 输出上的召回率从 90% 降至 20–30%；而 PU+TTA / PNU+TTA 在相同情形下保持 90% 以上的召回率，且在常规分布下与监督模型相当或更好。比例估计方面，TTA 的偏差随时间漂移保持稳定，而监督模型的偏差随时间显著升高。

**⚠️ 局限性**

局限性：①需要可用的无标签推断批量，难以一次性覆盖所有分布漂移；②对多重漂移（LLM 与人类写作同时变化）尚未全面评估；③对抗性投毒攻击可能破坏 TTA 的自适应效果；④缺乏后 2022 年经验证的人类文本，导致对人类漂移的评估受限；⑤在生产环境中需重新设计数据管线以支持批量自适应。

---

## 147. UC-Search: Risk-Aware Test-Time Search for Delayed Constrained Time-Series Control

**arXiv ID:** 2606.25274 | [PDF](https://arxiv.org/pdf/2606.25274v1)

**作者:** Xibai Wang `[一作]` `[通讯]` (NeuroQuant Labs Limited), Xibai Wang (NeuroQuant Labs Limited)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在时间序列预测模型的基础上，提出了一种名为 UC-Search 的测试时搜索框架，利用预测不确定性对有限深度路径进行风险加权并返回首步决策。

**💡 创新点**

创新点在于将可预测不确定性与硬约束自动机结合，实现了轻量级的第一步搜索，并给出了“自发退化”与“可延迟可行集分离”的理论阐释。

**🔧 技术方法**

核心技术包括多模型集成不确定性估计、基于置信度的候选扩展、UC-Beam 与 UC-MCTS 两种搜索策略，以及风险调节参数 λ 的路径评分。

**📊 数据集**

使用的数据集涵盖合成实验、ETT、LTSF、FI-2010 订单簿、M4 库存以及公开的多领域时间序列数据，构成了从模拟到真实业务的验证体系。

**📈 对比分析**

实验对比 CEM、MPPI、随机射击、贪心、贝叶斯优化等多种基线，结果显示 UC-Search 在多域延迟控制任务中平均提升约 3–5 点效用，同时在风险（CVaR/下行风险）上实现 2–4 倍改进。

**⚠️ 局限性**

局限性包括：仅在可行集可被前置动作改变的任务中有效；对不确定性校准高度敏感；并未在所有时间序列决策场景下实现全面优势，更多为任务特定的边界验证。

---

## 148. CoGeoAD: Hierarchical Color-Geometric Fusion with Multi-View Attention for Zero-Shot 3D Anomaly Detection

**arXiv ID:** 2606.25273 | [PDF](https://arxiv.org/pdf/2606.25273v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 149. Cross-Modality Structural Guidance in 3D Latent Diffusion for Robust FLAIR Super-Resolution

**arXiv ID:** 2606.25255 | [PDF](https://arxiv.org/pdf/2606.25255v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 150. Reflective VLA: In-Context Action Consequences Make VLAs Generalize

**arXiv ID:** 2606.25215 | [PDF](https://arxiv.org/pdf/2606.25215v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 151. RigPI: Dynamic Parameter Identification of Rigid Body via VLM-Seeded Differentiable Simulation

**arXiv ID:** 2606.25212 | [PDF](https://arxiv.org/pdf/2606.25212v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 152. Pre-Warm: Input-Conditioned Weight Initialization for Convolutional Neural Networks

**arXiv ID:** 2606.25256 | [PDF](https://arxiv.org/pdf/2606.25256v1)

**作者:** Rowan Martnishn `[一作]` `[通讯]`, Rowan Martnishn

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为Pre-Warm的零训练成本方法，用单个训练批次的局部图像块聚类结果初始化卷积网络第一层的一半滤波器。

**💡 创新点**

创新点在于将数据驱动的聚类结果直接映射到滤波器初始化，使用简洁的闭式规则自动推导大多数超参数，只保留一个对模型性能不敏感的缩放参数。

**🔧 技术方法**

核心技术包括均值中心化局部补丁提取、MiniBatchKMeans聚类、逆曼哈顿距离空间加权、标准差归一化以及半随机/半聚类滤波器初始化。

**📊 数据集**

使用了五个标准图像分类数据集：MNIST、Fashion‑MNIST、CIFAR‑10、SVHN、CIFAR‑100。

**📈 对比分析**

与传统Kaiming初始化做对照实验（8个随机种子），在所有数据集上均显著提升最终测试准确率（p<0.05），SVHN获得8/8胜利，CIFAR‑100获得7/8胜利，平均提升约0.32个百分点。

**⚠️ 局限性**

局限性包括仅验证于小型CNN和简单分类任务；尚未测试在更大网络（ResNet、Vision Transformer）、更复杂任务（检测、分割）上的效果；仅调优一次缩放参数；以及在极大数据集或非代表性第一批次时的计算开销可能增加。

---

## 153. Sponsored Group Signature and its Application to Privacy-preserving Guest Access in Smart Environments

**arXiv ID:** 2606.25248 | [PDF](https://arxiv.org/pdf/2606.25248v1)

**作者:** Sepideh Avizheh `[一作]` (University of Calgary), Shiwei Sun `[通讯]` (University of Calgary)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出并实现了一种双层赞助组签名（Sponsored Group Signature, SPGS），并基于其设计了匿名访客访问令牌（k-AGAT），用于智能建筑中隐私保护的访客临时访问控制。

**💡 创新点**

创新点包括：① 允许组成员自行赞助新成员加入，减少对组经理的依赖；② 赞助者保持完全匿名，受赞者签名可链式链接且可追溯；③ 将SPGS与加密和NIZK结合，构造k-AGAT，实现主机与访客的匿名身份、访问计数和误行为追踪；④ 在实验中展示了该方案相较于传统基于数字签名的令牌方案的可接受性能。

**🔧 技术方法**

核心技术：部分/动态组签名（BBS）、隐藏绑定承诺（Pedersen）、知识可证明的非交互零知识证明（Fiat‑Shamir的σ协议）、公钥加密（EC‑ElGamal）、伪名与批量验证、ACE‑OAuth框架整合。

**📊 数据集**

未使用公开数据集；通过在Ubuntu/Linux环境下使用Python、Charm‑Crypto、PBC等库，对多种角色（主机、访客、管理系统）进行基准测试，记录计算时间（毫秒）与存储字节。

**📈 对比分析**

与无隐私（仅数字签名）基线比较：SPGS+ k-AGAT 在主机端和管理端的签名/验证/加密/解密操作略有额外开销，平均多约10‑20 ms；访客端开销基本相同；存储方面，管理端多约14 KB，主机端多约530 B；整体仍在毫秒级、KB级范围内，足以满足实际部署需求。

**⚠️ 局限性**

限制：① 需要可信组经理/管理系统；② 当前实现不支持成员撤销和动态大小扩容（BBS为静态组签名）；③ 伪名批量更新导致主机注册后需等待批处理；④ 方案对资源受限设备的加密/验证效率还有提升空间。

---

## 154. GRAFT: Graph-Based Affordance Transfer via Part Correspondence

**arXiv ID:** 2606.25241 | [PDF](https://arxiv.org/pdf/2606.25241v1)

**作者:** Mengying Lin `[一作]` (Georgia Institute of Technology), Danfei Xu `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于几何感知的对应框架（GRAFT），利用部件图与Unbalanced Fused Gromov–Wasserstein（UFGW）实现零样本抓取和操作演示迁移。

**💡 创新点**

创新点在于：①以部件级结构为核心而非全局语义/几何，显著提升跨类别对应多样性；②引入UFGW与EM重估节点质量，实现部分匹配的稳健性；③将对应结果与AnyGrasp、MimicGen结合，生成可执行的抓取姿态与多步骤轨迹。

**🔧 技术方法**

主要技术包括：部件分割与特征图构建、四ier位置编码、DINO或CLIP视觉特征、Unbalanced Fused Gromov–Wasserstein距离、EM优化节点质量、AnyGrasp抓取规划、MimicGen轨迹迁移。

**📊 数据集**

使用的实验数据集包括：一组手工标注的单示范演示（如WaterBucket、Teapot等），SAPIEN仿真环境与Mujoco进行抓取成功率评估，实际Frankia Panda机器人进行真实抓取验证。

**📈 对比分析**

与where2act、AnyGrasp、ROBO-ABC等基线相比，GRAFT在ASR、NSS、DTM、仿真成功率（81.25%）和MimicGen轨迹生成成功率（63%）方面大幅提升，基线仅 11% 甚至 36% 左右。

**⚠️ 局限性**

局限性包括：对可变形或高度关节化对象的假设不成立；实验集中在单次抓取/放置任务，难以推广到长周期、多接触的复杂操作；部件分割需人工或高质量的标注。

---

## 155. Automatic Generation of Highlights for Academic Paper Via Prompt-based Learning

**arXiv ID:** 2606.25253 | [PDF](https://arxiv.org/pdf/2606.25253v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 156. Wear-Clearance-Impact Coupling in the Jansen Linkage: A Gait-Durability-Optimized Design Slows Joint Loosening

**arXiv ID:** 2606.25208 | [PDF](https://arxiv.org/pdf/2606.25208v1)

**作者:** Jichao Wang `[一作]` `[通讯]`, Jichao Wang

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建并仿真 Jansen 行走足链的前向动力学模型，加入间隙接触、摩擦与 Archard 磨损耦合，研究清晰接触对关节磨损与负载的影响。

**💡 创新点**

首次将间隙耦合的前向动力学与磨损循环闭环结合到 Jansen 腿；验证了在真实间隙条件下，优化的步态设计在统计上仍能提升耐久性。

**🔧 技术方法**

连续接触力学模型（Lankarani–Flores 带滞后阻尼）、Ambrósio 修正库伦摩擦、约束稳定的微分代数方程求解、Archard 磨损积分、随机相位统计比较。

**📊 数据集**

无公开数据集；使用仿真生成的动力学轨迹与磨损量；对比经典设计与优化设计在不同初始间隙下的统计表现。

**📈 对比分析**

通过在随机起始相位下多组实验，对单个关节磨损和峰值接触力进行分布比较；统计检验（Mann‑Whitney）显示优化设计的磨损/峰值力均显著降低，性能提升约 × 到 × 倍。

**⚠️ 局限性**

局限包括宏观加速磨损步骤、简化的冲击阻尼参考、施加的垂直地面载荷非连续接触，且仅针对单关节/两关节间隙未考虑更复杂柔性或多材料情况。

---

## 157. Variational Inference via Entropic Transport Descent

**arXiv ID:** 2606.25265 | [PDF](https://arxiv.org/pdf/2606.25265v1)

**作者:** Vincent Pacelli `[一作]` (Georgia Institute of Technology), Evangelos Theodorou `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于熵正则化最优传输的粒子变分推断框架（Entropic Transport Descent, ETD），用于高维多模态分布的近似采样。

**💡 创新点**

创新点在于将JKO梯度流离散化为半松弛熵正则化OT问题，得到单参数τ族；通过Sinkhorn算法实现可扩展的粒子更新；并给出精确的平衡稳态分析和重要性权重纠正，消除采样偏差。

**🔧 技术方法**

采用JKO proximal scheme、KL链式规则、半松弛熵正则化OT、Sinkhorn迭代、Gibbs 变分法以及可选的分数梯度或随机提议机制。

**📊 数据集**

在多维高斯、二维能量函数（四种复杂分布+八模环高斯混合）、贝叶斯逻辑回归（德国信用、覆土分类）、贝叶斯神经网络（九个UCI回归数据集）、以及物理Boltzmann采样（双井4维、Lennard-Jones 13维）上进行实验。

**📈 对比分析**

与SVGD、SGLD、AGF-SVGD等基线比较；在高维/多模态任务中，ETD保持或提升方差、能量距离、负对数似然、覆盖率等指标，表现优于或匹敌现有方法，尤其在高维和多模态场景下优势明显。

**⚠️ 局限性**

限制主要包括：每次迭代成本为O(NM·L)（需要Sinkhorn迭代），平衡耦合需迭代收敛；当使用欧氏距离作为成本时，维度增大导致Gibbs核退化，表现出与其他基于欧氏距离方法相同的“维度灾难”。

---

## 158. FDN: Interpretable Spatiotemporal Forecasting with Future Decomposition Networks

**arXiv ID:** 2606.25201 | [PDF](https://arxiv.org/pdf/2606.25201v1)

**作者:** Nicholas Majeske `[一作]` (Indiana University), Ariful Azad `[通讯]` (Texas A&M University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种名为 Future Decomposition Network（FDN）的时空序列预测框架，通过对历史预言片段进行软分类并按权重插值已学习的关键模式，生成可解释且准确的未来预测。

**💡 创新点**

创新点在于：①将预测任务视为“未来分解”，即用有限个基本模式表示系统行为并通过分类概率进行插值；②引入 Future Decomposition 层作为新的预测算子；③通过学习的节点嵌入、周期嵌入和局部动态注意力实现对空间与时间的细粒度调控。

**🔧 技术方法**

技术实现包括：图卷积网络（GCN）学习节点间依赖；局部动态注意力（LDA）自适应特征过滤；周期嵌入捕获季节性；GRU 对观察窗口进行时间编码；Softmax 分类得到模式置信度；线性插值生成最终预测。

**📊 数据集**

使用三大公开数据集：水文系统（Wabash River，1,276子流域），交通系统（E‑PEMS‑BAY，约6,700个传感器），能源系统（Solar‑Energy，若干光伏电站）。

**📈 对比分析**

与 11 种基线模型（包括 STGCN、MTGNN、Informer、Autoformer 等）在 MAE、MAPE、RMSE 上进行比较。FDN 在大多数预测时隙下均与 SOTA 接近或优于之，最长时隙下可实现 9.1% MAPE 降低、2.5% RMSE 降低，且参数量、内存占用和训练时间均显著低于竞争者。

**⚠️ 局限性**

局限性：①需要对每个系统训练 K 个模式，K 的选择与计算成本有关；②对极端事件的泛化仍受训练数据分布限制；③模型在无先验图结构的系统（如部分能源数据）仍需进一步验证；④模型对不同时间分辨率的适应性需更系统化评估。

---

## 159. The Digital Pirahã Condition: Ecological Mismatch and the Reconstruction of Recursive Cognition

**arXiv ID:** 2606.25287 | [PDF](https://arxiv.org/pdf/2606.25287v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 160. Multilingual Hematology Visual Question Answering Dataset

**arXiv ID:** 2606.25246 | [PDF](https://arxiv.org/pdf/2606.25246v1)

**作者:** Hajra Malik `[一作]` (King Edward Medical University), Waqas Sultani `[通讯]` (Information Technology University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了临床验证的英语-乌尔都语双语白细胞视觉问答（VQA）数据集，并对其进行专家校验；

**💡 创新点**

创新点在于将细胞形态学属性自动转换为多样化、富含临床信息的问答对，结合专业词典实现高质量的双语翻译，填补了多语种医学VQA资源缺口；

**🔧 技术方法**

采用了 BioMistral-7B 生成问答、NLLB-200 进行机器翻译、专业词典后处理与重新翻译、以及 LoRA 微调等技术；

**📊 数据集**

使用 LeukemiaAttri 与 WBCAtt 两个公开形态学标注数据集，生成约 110K 的中英问答对；

**📈 对比分析**

在零样本和微调两种设置下评估 Qwen2‑VL‑2B‑Instruct、InternVL2.5‑2B 以及 Uni‑Hema 模型，结果表明微调后模型在英语、乌尔都语及双语基准上均显著优于零样本，双语训练还能提升跨语言泛化能力；

**⚠️ 局限性**

局限包括：数据集仅来自巴基斯坦医生问卷，可能不具备普适性；翻译仍存在细微错误；模型仅适用于研究而非临床决策；以及对低资源语言的进一步验证尚未完成。

---

## 161. A Hybrid CNN-LSTM Intrusion Detection Framework for Cybersecurity in Smart Renewable Energy Grids

**arXiv ID:** 2606.25200 | [PDF](https://arxiv.org/pdf/2606.25200v1)

**作者:** Sajib Debnath `[一作]` (AES Corporation), Remon Das `[通讯]` (Dominion Energy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于混合CNN‑LSTM的入侵检测系统，对可再生能源智能电网流量进行实时异常检测。

**💡 创新点**

创新点在于将空间特征提取与时序建模相融合，配合七步预处理（SMOTE、特征选择等）显著提升对低速持续攻击的识别能力。

**🔧 技术方法**

主要技术包括1D卷积网络、长短时记忆网络、SMOTE过采样、互信息特征选择、批归一化、Dropout以及INT8量化部署。

**📊 数据集**

使用了CICIDS2017和NSL‑KDD两个标准网络流量数据集进行训练与评估。

**📈 对比分析**

与传统SVM、RF、KNN以及单一CNN、LSTM基线对比，混合模型在CICIDS2017上实现了98.7%准确率、98.2%精确率、97.9%召回率、0.995 AUC，跨数据集泛化误差仅0.5pp。

**⚠️ 局限性**

主要局限包括：对真实工业控制系统流量的代表性不足、SMOTE产生的合成样本可能不完全符合真实攻击分布、模型规模虽小但仍高于传统方法，缺乏在线自适应学习能力。

---

## 162. LLM4MTLs: Automated Generation and Empirical Evaluation of Model Transformation Languages

**arXiv ID:** 2606.25193 | [PDF](https://arxiv.org/pdf/2606.25193v1)

**作者:** Bowen Jiang `[一作]` (Karlsruhe Institute of Technology), Anne Koziolek `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并实现了一个名为 LLM4MTLs 的自动化工作流，用于生成、评估模型变换语言（MTL）的代码，并针对 Reactions、ATL、ETL、QVTo 四种 MTL 开发了 47 条可执行的评估案例。

**💡 创新点**

创新点在于：①提供可复现、可扩展的 Prompt 构造、代码生成与度量评估完整管道；②首次公开 47 条包含元模型、实例和语义测试套件的 MTL 评估集合；③系统性评估不同提示策略和 LLM 对四种 MTL 语法与语义质量的影响。

**🔧 技术方法**

采用技术包括：大语言模型（GPT‑5.1、Gemini 2.5‑Pro、Claude Sonnet 4.5）、few‑shot、grammar 以及 helper‑method 提示策略；n8n 工作流自动化；ANTLR/Eclipse 解析器；ChrF、Unparsed Rate、PPL_s、Pass@1 等质量指标；以及 ANOVA、Friedman、McNemar 等统计检验方法。

**📊 数据集**

使用的数据集为 47 条 MTl 变换脚本（Reactions、ATL、ETL、QVTo），每条脚本配套元模型、模型实例和手工编写的语义测试用例。

**📈 对比分析**

通过在三种 LLM 上执行五种提示组合（基线、few‑shot、grammar、两者组合、加 helper‑methods）进行全排列评估。结果显示：few‑shot 最能提升语法相似度（ChrF）和可解析率；grammar 与 few‑shot 组合进一步稳定；LLM 选择对语法质量影响显著，但对 Pass@1 语义正确率的影响有限。

**⚠️ 局限性**

局限性包括：仅评估四种 MTL 与三种商业 LLM；未尝试 RAG、微调等更高级技术；评估指标（ChrF、Pass@1）可能未完全覆盖代码质量；实验受 LLM 随机性、训练数据泄漏的影响；适配新 MTL 仍需手工调整系统提示和资源；对不规范自然语言请求的鲁棒性未知。

---

## 163. MRI2Rep: Autoregressive Structured Report Generation for 3D Liver MRI

**arXiv ID:** 2606.25279 | [PDF](https://arxiv.org/pdf/2606.25279v1)

**作者:** Xinran Li `[一作]` (Yale University), Lawrence H. Staib `[通讯]` (Yale University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发了MRI2Rep，一个基于3D肝脏MRI的自回归框架，能够自动生成结构化、符合LI‑RADs标准的放射报告。

**💡 创新点**

创新点在于引入报告到标签的规范化（RLC）模块，将自由文本报告转化为闭合词汇的诊断序列，且不依赖病灶级注释即可实现端到端生成。

**🔧 技术方法**

使用了3D CNN+Transformer的自回归编码解码架构，结合ART→PV预训练、视觉自注意力、词丢弃与辅助二分类头，并利用Claude‑3.5等LLM进行RLC与模板化渲染。

**📊 数据集**

采用单中心10年回顾性收集的3,929例多相位肝脏MRI与对应报告，其中3,830例完整数据被用于训练和评估。

**📈 对比分析**

与现有3D vision‑language基线对比，MRI2Rep在测试集上实现病例敏感度76.0%、肿瘤级F1 29.4%、肝脏背景准确率82.4%，并在盲评中获得两位放射科医生分别75%/70%的可接受率，LLM‑Eval约61.8%。

**⚠️ 局限性**

主要局限包括缺乏空间定位导致肿瘤级F1受限、闭合词汇限制表达范围，以及单中心数据限制了跨中心的泛化能力。

---

## 164. Semantic Code Clone Detection: Are We There Yet?

**arXiv ID:** 2606.25272 | [PDF](https://arxiv.org/pdf/2606.25272v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 165. An Integrated Hardware-Software Design for Low-Data Spatial Defect Detection in Robotic Visual Inspection with Hybrid Optoelectronic Neural Networks

**arXiv ID:** 2606.25277 | [PDF](https://arxiv.org/pdf/2606.25277v1)

**作者:** Chaoqing Tang `[一作]` (Huazhong University of Science and Technology), Wenzhong Liu `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种集成硬件与软件的光电子架构，用 DMD 作为物理卷积层实现块压缩感知，并通过 CLIP 文本指导注意力实现无形状标注的缺陷定位，最终得到低数据量、低计算量且具备形状级定位的机器人视觉检测系统。

**💡 创新点**

创新点包括：①将 DMD 重构为光学卷积层，将压缩感知与特征提取融合；②采用块级压缩感知直接在光学域实现降维，完全消除重建步骤；③利用预训练的视觉‑语言模型 CLIP 用自然语言形状描述来监督网络，消除形状级标注；④提出 Localization Accuracy for Attention (LAA) 指标评估注意力热图的形状定位精度。

**🔧 技术方法**

技术手段：Digital Micromirror Device (DMD) 物理卷积、块压缩感知、光学投影与光电传感、结构光照明、预训练 CLIP、Vision Transformer (ViT) 与 CNN 后端、EigenCAM 关注图生成、LAA 评估指标。

**📊 数据集**

使用透明 PVC 试样数据集：8 种厚度、每种厚度 2×3 区域，预制点状与划痕缺陷以及健康样本，共约 5,000 张图像，其中 1,618 影响损伤、1,593 划痕、1,637 健康样本；在实验前期以完整图像形式收集后转为压缩感知数据。

**📈 对比分析**

与传统基于图像的检测（未压缩）以及常规 CNN/Yolo 方案对比：在 S_r=0.1 的压缩率下，ViT 取得约 0.95 的分类准确率，仅低 3% 于无压缩情况；数据量下降 90%，CNN 计算量下降约 60%；形状定位 LAA 值在 0.8–0.9 之间，表明注意力热图能较好地覆盖缺陷形状。

**⚠️ 局限性**

局限性：①需在每个新工况下重新采集数据并重新训练模型；②光学路径相对复杂，难以实现像相机那般的即插即用；③对多种缺陷几何形态的泛化能力有限，需进一步扩充多形状模拟训练数据。

---

## 166. Decidability and Undecidability Results for LIA-Definable Impartial Combinatorial Games

**arXiv ID:** 2606.25276 | [PDF](https://arxiv.org/pdf/2606.25276v1)

**作者:** Shiguang Feng `[一作]` (Sun Yat-sen University), Quanlong Guan `[通讯]` (Jinan University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

研究了线性整数算术（LIA）可定义的无偏组合博弈（ICG）的多种判定问题，包括终止性、循环性、赢/输/平公式以及状态等，并给出了其可判定与不可判定的理论界限；

**💡 创新点**

通过构造三种从 2‑计数器机归约的 ICG（Π(M)、Π'(M)、Π''(M)），首次证明了大部分判定问题在一般 ICG 上不可判定，并提出“有限深度”ICG 类，在此类中赢/输状态判定问题可以被有效求解；

**🔧 技术方法**

主要技术包括：基于质因子编码的 2‑计数器机归约、LIA 的量化消除与可判定性分析、以及基于状态深度的递归公式构造算法（IsWinningState 等）；

**📊 数据集**

本工作为理论研究，未使用实验数据集；所有结果均来自形式化证明与归约；

**📈 对比分析**

由于是理论证明，未进行实验性能比较；结论为判定性结果（可判定/不可判定），而非算法执行时间或资源消耗；

**⚠️ 局限性**

局限性包括：非终止 ICG 的赢/输状态判定仍不可判定；有限深度条件相对苛刻，实际应用范围有限；研究仅覆盖无偏、完全信息、无随机性的组合博弈，未涉及部分信息或随机博弈等更一般情况。

---

## 167. OrthoTrack: Continuous 6-DoF UAV Trajectory Estimation Anchored in Public Orthophotos

**arXiv ID:** 2606.25245 | [PDF](https://arxiv.org/pdf/2606.25245v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 168. Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds

**arXiv ID:** 2606.25234 | [PDF](https://arxiv.org/pdf/2606.25234v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 169. Cage-based Texture Transfer with Geometric Filtering

**arXiv ID:** 2606.25220 | [PDF](https://arxiv.org/pdf/2606.25220v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 170. Inverse Reinforcement Learning for Interpretable Keystroke Biomarkers in Parkinson's Disease

**arXiv ID:** 2606.25270 | [PDF](https://arxiv.org/pdf/2606.25270v1)

**作者:** Navin Bondade `[一作]` `[通讯]` (University College London), Navin Bondade (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用逆向强化学习从打字动态中提取可解释的奖励函数，评估其与帕金森病严重程度的关联。

**💡 创新点**

首次将逆向强化学习应用于打字动态，纠正奖励函数可识别性问题并验证结果稳健性。

**🔧 技术方法**

最大熵逆向强化学习（MaxEnt IRL），离散化飞行时间选择，L-BFGS-B优化。

**📊 数据集**

公开的neuroQWERTY MIT-CSXPD数据集（85名受试者，其中42名帕金森病患者）。

**📈 对比分析**

通过与原始特征和原始打字速度对比，IRL得到的速度偏好权重与UPDRS‑III呈显著负相关，解释度从19.4%提升至33.8%；与先前报道的AUC结果相比，提供了可解释性而非单纯分类性能。

**⚠️ 局限性**

模型仍存在速度与一致性奖励项高度相关的可识别性限制，缺乏年龄/性别等人口统计信息，外部验证缺失。

---

## 171. Co-designing a Preliminary Repository of Augmented Reality Concepts for Real-Time Emotion Regulation

**arXiv ID:** 2606.25271 | [PDF](https://arxiv.org/pdf/2606.25271v1)

**作者:** Graciela Camacho-Fidalgo `[一作]` (Texas A&M University), Edgar Rojas-Muñoz `[通讯]` (Texas A&M University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过两阶段参与式设计流程，首先让40名焦虑倾向者使用Nominal Group Technique（NGT）生成106个实时AR情绪调节想法，随后请10名精神健康专业人士对这些想法进行卡片分组和可行性评估，最终构建了8个可操作的AR设计概念库；

**💡 创新点**

创新点在于首次将用户产生的实时AR情绪调节想法与临床专业反馈相结合，形成了以用户需求为中心、专家可行性为支撑的可重用设计资源，为AR心理健康干预提供了系统化、可扩展的概念框架；

**🔧 技术方法**

采用的技术包括Nominal Group Technique、主题分析、在线混合式卡片分组、Krippendorff α与Randolph κ的可靠性评估，以及5点李克特量表进行可行性打分；

**📊 数据集**

使用的数据集为：①40名焦虑受试者的NGT生成的106条设计想法；②10名精神健康专业人士的卡片分组与可行性评分数据；

**📈 对比分析**

方法上通过专家卡片分组的共识规则和可靠性指标（Krippendorff α 0.16–0.53，Randolph κ 0.68–0.94）评估分类一致性；可行性得分显示情绪支持环境、感官负荷管理与多感官调节等集群得分最高，平均分约为4.1/5；

**⚠️ 局限性**

局限性包括样本规模小、受试者主要来自高校、缺乏文化和年龄多样性，专家样本虽专业但人数有限；未进行真实环境或临床验证，且仅构建概念库，功能实现与有效性尚待后续原型测试验证。

---

## 172. Extreme Meta-Classification for Large-Scale Zero-Shot Retrieval

**arXiv ID:** 2606.25237 | [PDF](https://arxiv.org/pdf/2606.25237v1)

**作者:** Sachin Yadav `[一作]` (Microsoft Research), Manik Varma `[通讯]` (Microsoft Research)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种大规模零样本检索方法，将极端分类器知识合成新物品的 meta‑classifier，实现快速准确的零样本检索。

**💡 创新点**

创新点在于构建 ExtreMe META-Classification 框架以及 IRENE 算法，利用已有极端分类器生成新物品的 meta‑classifier，兼顾高精度与低延迟。

**🔧 技术方法**

技术手段包括 Siamese 语义编码器（如 DistilBERT）、大规模线性分类器、近似最近邻搜索、Transformer 生成器以及加权一对一交叉熵损失。

**📊 数据集**

使用了 LF‑AOL‑270K、LF‑Wikipedia‑500K、LF‑WikiHierarchy‑550K、LF‑AmazonTitles‑1.3M 以及 KeywordPrediction‑10M 等多种工业级数据集进行评估。

**📈 对比分析**

与主流 dense retriever（NGAME、ANCE、MACLR、DPR）以及零样本极端分类方法（SemSup‑XC、ZestXML）对比，IRENE 在 Recall@10、Precision@1 等指标提升 1%–45%（平均约 15%），在线 A/B 测试点击率提升 4.2%。

**⚠️ 局限性**

局限性包括需先训练极端分类器，对邻居数 K 和 Transformer 层数敏感，且对极端分布变化的鲁棒性仍需进一步验证。

---

## 173. Evaluation Protocols and Validation for Cameras in Indoor Healthcare Monitoring

**arXiv ID:** 2606.25284 | [PDF](https://arxiv.org/pdf/2606.25284v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 174. ARTOO-DARTU: Studying AR-HRC With AR Obstruction Mitigation During a Warehouse Task

**arXiv ID:** 2606.25202 | [PDF](https://arxiv.org/pdf/2606.25202v1)

**作者:** Christian Fronk `[一作]` (Duke University), Maria Gorlatova `[通讯]` (Duke University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了ARTOO-DARTU，一个在仓库人机协作中通过实时机器人情境分析和阻碍检测/缓解技术提升效率的增强现实系统，并通过34名参与者的Pocket MonstARs实验验证其有效性。

**💡 创新点**

创新点在于设计专门针对AR-HRC场景的阻碍检测与缓解流水线，使得AR叠加内容在与移动机器人一起工作时不会遮挡重要的真实世界信息，从而兼顾信息传达与安全可视。

**🔧 技术方法**

使用了基于AR头显的实时情境分析、阻碍检测与透明化技术，并结合YOLO目标识别与自定义透明化算法。

**📊 数据集**

使用了Pocket MonstARs实验环境，包括12个带标签的纸板箱和4个含YOLO可识别物体的箱子，作为模拟仓库取货任务的数据集。

**📈 对比分析**

通过对比开启与关闭阻碍缓解功能的用户实验，发现开启时整体任务效率提升46%，需要真实世界可视的子任务效率提升61%。

**⚠️ 局限性**

局限性包括实验规模有限、仅在单一仓库环境下验证、对动态环境中更复杂遮挡的处理能力尚未完全证明。

---

## 175. Heuresis: Search Strategies for Autonomous AI Research Agents Across Quality, Diversity and Novelty

**arXiv ID:** 2606.25198 | [PDF](https://arxiv.org/pdf/2606.25198v1)

**作者:** Antonis Antoniades `[一作]` (University of California Santa Barbara), William Yang Wang `[通讯]` (University of California Santa Barbara)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了名为 Heuresis 的开源框架，提供了可组合的 Ideator、Executor、Grader、Auditor 等原语，并实现了六种基于 LLM 的搜索策略（Greedy、MAP‑Elites、Go‑Explore、Islands、Curiosity、Omni）。通过该框架在三类机器学习研究任务（NanoGPT 预训练、On‑Policy RL、模型去学习）上进行大规模实验，系统评估了生成思路在质量、Diversity 与 Novelty 三个维度上的表现。

**💡 创新点**

创新点主要包括：
1) 将完整的研究循环抽象为通用可插拔原语，便于不同搜索策略的对比；
2) 提出了质量–多样性–创新度（Q–D–N）三轴评估视角，揭示当前搜索方法无法突破质量-创新前沿；
3) 在三大任务上对六种 QD / 搜索策略进行大规模横向比较，首次系统性展示各策略在质量、Diversity 与 Novelty 之间的权衡。

**🔧 技术方法**

技术手段包括：
- LLM 驱动的 Ideator（Gemini‑3.1‑Pro + OpenCode）和 Executor；
- 质量评估器（Grader）和奖励审计器（Auditor）以确保跑分真实性；
- 六种搜索策略实现：贪婪、MAP‑Elites、Go‑Explore、岛屿进化、Curiosity（学习进度引导）与 Omni（MoI gate + KNN 归档）；
- Embedding（sentence‑transformers）+ UMAP 进行思想空间可视化；
- Web‑search‑based Novelty 评估器（Claude‑Code Sonnet 4.6）。

**📊 数据集**

使用的数据集和任务：
- NanoGPT 预训练：ClimbMix‑400B shuffle；
- On‑Policy RL：MinAtar Breakout（训练集）与 Asterix（held‑out 评估）；
- 模型去学习：WMDP‑cyber benchmark + MMLU‑STEM 子集。

**📈 对比分析**

比较方法与性能：
- 对每个（策略、任务）组合执行 300 次实验，共 3222 次已评分运行；
- 在质量维度上：NanoGPT 由 Greedy 主导，On‑Policy RL 由 MAP‑Elites/Islands 获胜，Model Unlearning 仍由 Greedy 领先；
- 在 Diversity 维度上：MAP‑Elites 与 Go‑Explore 在所有任务均居前；
- 在 Novelty 维度上：最高 Novelty 评分仅为 G&P=2，且只有一次落入 Top‑10；
- 结果显示搜索能调节各维度但未能突破质量‑创新边界，且不同任务对策略的最优性呈现任务依赖性。

**⚠️ 局限性**

限制与挑战：
1) 计算预算仅 300 次迭代，可能无法捕捉更深层次突破；
2) 搜索策略与提示未进行细致调优，实际性能可能可提升；
3) 当前 LLM‑驱动搜索缺乏评估“潜在改进性”的信号，导致创新思路在首次执行时被过滤；
4) 奖励欺骗仍出现，尽管加入 Auditor 但仍有漏检；
5) 在 held‑out 任务上泛化差异大，表明策略易过拟合训练任务。

---

## 176. How Do Developers Maintain and Evolve Their Agents' Instructions? An Empirical Study

**arXiv ID:** 2606.25257 | [PDF](https://arxiv.org/pdf/2606.25257v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 177. To Isolate or to Score? Model-Adaptive Assessment for Cost-Efficient Multi-Agent RAG

**arXiv ID:** 2606.25191 | [PDF](https://arxiv.org/pdf/2606.25191v1)

**作者:** Jungseob Lee `[一作]` (Korea University), Heuiseok Lim `[通讯]` (Korea University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对小型（7B–9B）指令调优语言模型在检索增强生成（RAG）任务中，探究并实现了无需训练的评估干预，并提出了基于诊断的动态路由架构 MADARA，以降低多代理评估的计算成本。

**💡 创新点**

创新点在于：①发现弱基线模型的提升主要来自文档隔离（Per-Document Extraction, PDE）而非评分质量；②提出无标签的 Reasoning-Score Coupling (RSC) 诊断方法，用于判定模型-任务对的评分行为；③基于 RSC 和无过滤（No-Filter）性能的诊断结果，构建四种处理策略（PDE、SDA、CoT 去极化、ATF）并实现零样本路由，展示了跨模型、跨任务的通用性。

**🔧 技术方法**

技术包括：无训练的提示式干预（如文档隔离、CoT 去极化）、扰动评估（RSC）、分数归一化与加权、阈值过滤（ATF）、多代理评分框架、vLLM 部署、基于 Spearman 相关系数的诊断阈值。

**📊 数据集**

使用的公开数据集包括：CONFLICTS（对抗式 QA）、FEVER（二分类事实验证）、TriviaQA（事实类 QA）以及 MuSiQue（多跳推理），并在不同检索设置下（BM25、Contriever、生成式 reranker）进行评估。

**📈 对比分析**

与传统 RAG（No-Filter）及标准 3‑agent 评分策略比较，MADARA 在弱基线模型上通过 PDE 提升 25–36pp；在强基线模型上通过 RSC 触发的 SDA、CoT 或 ATF 提升 1–5pp；整体上实现了 4 倍的推理调用减少，并在多跳任务中验证了策略切换的必要性。

**⚠️ 局限性**

局限性包括：①PDE 在需要跨文档合成的多跳推理中效果有限；②当前路由阈值为静态，未针对检索质量动态调整；③只针对单步检索 QA 进行验证，未覆盖更复杂的工具使用或多轮交互任务；④对检索质量高的情形，PDE 的收益会显著下降。

---

## 178. Towards Structuring an Arabic-English Machine-Readable Dictionary Using Parsing Expression Grammars

**arXiv ID:** 2606.25231 | [PDF](https://arxiv.org/pdf/2606.25231v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 179. EvoFlock: evolved inverse design of multi-agent motion

**arXiv ID:** 2606.25280 | [PDF](https://arxiv.org/pdf/2606.25280v1)

**作者:** Craig Reynolds `[一作]` `[通讯]`, Craig Reynolds

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf`

**🎯 论文内容**

本文提出一种基于逆向设计的多智能体运动模型自动调参框架 EvoFlock，利用遗传算法在黑盒仿真环境中寻找最优参数。

**💡 创新点**

创新点在于：①将多目标评价（分离距离、速度范围、碰撞避免）通过超体积（hypervolume）标量化，自动生成 Pareto 前沿；②采用稳态遗传算法（SSGA）和负选择策略保持多样性；③不要求模型可微，兼容任何语言实现的仿真。

**🔧 技术方法**

技术主要包括：遗传算法（SSGA）、多目标超体积标量化、并行多线程仿真、参数化 Boids 模型和自定义目标函数。

**📊 数据集**

使用数据集：200 个 Boids、500 步仿真、含障碍物的封闭空间，实验中运行 4 线程以降低噪声，参数空间包含 15 个可调数值。

**📈 对比分析**

对比方法：实验显示在 30,000 步 SSGA（相当于 100 代）后平均最佳适应度约 0.83，增加到 60,000 步可提升至 0.88；在不同种群规模下（300、500、1000）验证了步数与性能关系；与手工调参相比，自动化方法在 2 小时内即可得到可观的优化结果。

**⚠️ 局限性**

局限性包括：结果具有随机性，因随机种子与并行执行导致每次运行的最佳适应度波动；仅针对封闭障碍环境验证，开阔空间行为需进一步改进；遗传算法在大规模参数或复杂目标时收敛可能受限；并未探讨梯度或连续空间演化策略的潜在优势。

---

## 180. Heterogeneous and Adept Snapshot Distillation for 3D Semantic Segmentation

**arXiv ID:** 2606.25278 | [PDF](https://arxiv.org/pdf/2606.25278v1)

**作者:** Xiaopei Wu `[一作]` (Shanghai AI Laboratory), Wanli Ouyang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8d10c613-917e-4880-9716-17789f50e119` `afceb026-1760-41ae-8d86-010831a37d97` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过知识蒸馏，将多模态教师模型和多专家教师模型的知识迁移到单模态点云网络，实现性能提升；

**💡 创新点**

提出信息导向异构蒸馏（IHD）配合信息导向过滤（IOF）来挑选最有用的图像视图；以及适应性快照蒸馏（ASD）利用训练过程中的快照作为专家教师，只在其擅长的类别上监督学生；

**🔧 技术方法**

知识蒸馏、跨模态特征对齐、视图过滤、专家快照选取、点云与图像特征融合；

**📊 数据集**

ScanNetV2和S3DIS两大3D语义分割基准数据集；

**📈 对比分析**

在ScanNetV2上相较无多模态/多数据集方法的基线提升1.1 mIoU，在S3DIS上提升1.0 mIoU，且保持与多数据集训练方法相近的效果；在推理速度、内存和参数上与基准PTV3持平，且比同类方法更快、更轻；

**⚠️ 局限性**

只在训练阶段依赖多模态数据和多快照，不保证对所有任务都有显著提升；对训练过程的超参数（如快照选取阈值、视图数）敏感；

---

## 181. MJEPA: A Simple and Scalable Joint-Embedding Predictive Architecture for Audio-Visual Learning

**arXiv ID:** 2606.25225 | [PDF](https://arxiv.org/pdf/2606.25225v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 182. Spatio-Temporal Retrieval-based Priors for Adaptive Computational Teaching in Driving

**arXiv ID:** 2606.25224 | [PDF](https://arxiv.org/pdf/2606.25224v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 183. Swazure: Swarm Measurement of Pose for Flying Light Specks

**arXiv ID:** 2606.25222 | [PDF](https://arxiv.org/pdf/2606.25222v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 184. RAVEN: Long-Horizon Reasoning & Navigation with a Visuo-Spatio-Temporal Memory

**arXiv ID:** 2606.25206 | [PDF](https://arxiv.org/pdf/2606.25206v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 185. SoK: AI Secure Code Generation: Progress, Pitfalls, and Paths Forward

**arXiv ID:** 2606.25195 | [PDF](https://arxiv.org/pdf/2606.25195v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 186. Communication complexity of point-line incidences over the reals

**arXiv ID:** 2606.25192 | [PDF](https://arxiv.org/pdf/2606.25192v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

---

## 187. Structuring International Governance through the Space of Concerns

**arXiv ID:** 2606.25286 | [PDF](https://arxiv.org/pdf/2606.25286v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 188. Adaptive Re-Ranking

**arXiv ID:** 2606.25249 | [PDF](https://arxiv.org/pdf/2606.25249v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 189. Tensor-Based Batch Fuzzing with Adaptive Perturbation Scaling for Deep Neural Networks

**arXiv ID:** 2606.25239 | [PDF](https://arxiv.org/pdf/2606.25239v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 190. FUTO Swipe: Layout-Agnostic Neural Swipe Decoding

**arXiv ID:** 2606.25247 | [PDF](https://arxiv.org/pdf/2606.25247v1)

**作者:** David Lee Miller `[一作]` (FUTO), Aleksandras Kostarevas `[通讯]` (FUTO)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种可在任意连续移动键盘布局上运行的神经滑动解码器，训练时通过共置几何增强实现布局无关性。

**💡 创新点**

创新点在于让编码器实时读取键盘坐标并使用联合轨迹-布局增强，突破传统模型对训练布局的依赖。

**🔧 技术方法**

采用TCN结构的轨迹编码器、CTC+排名损失、固定布局DFSMN解码器以及联合几何数据增强。

**📊 数据集**

使用MIT许可的1M+滑动语料库（主要为英式QWERTY）以及外部ClearFlow、JCUKEN等布局的数据进行评估。

**📈 对比分析**

与基线SHARK2相比，在ClearFlow、KASROZ等未见布局上实现了10–15个百分点的top‑1提升；在QWERTY上保持相近准确率。

**⚠️ 局限性**

主要限制包括数据集以英式QWERTY为主，其他布局和语言样本不足；捐赠者自选可能导致人群偏差；语料覆盖正式词汇，缺乏口语俚语。

---

## 191. EPTS: Elastic Post-Training Sparsity for Efficient Large Language Model Compression

**arXiv ID:** 2606.25285 | [PDF](https://arxiv.org/pdf/2606.25285v1)

**作者:** Ke Xu `[一作]` (Anhui University), Xiaoyun Wang `[通讯]` (Anhui University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Elastic Post-Training Sparsity (EPTS)，实现单次优化即可生成可在多种稀疏率下部署的 LLM 模型。

**💡 创新点**

创新点在于：①多稀疏层级 LoRA (MS‑HiLoRA) 通过层级继承实现不同稀疏率间的知识共享；②多稀疏特征混合器 (MSFM) 动态融合不同稀疏粒度的特征，提升稀疏化鲁棒性；③一次性全稀疏率训练，省去多次单稀疏率优化的时间。

**🔧 技术方法**

使用 LoRA 低秩适配器、层级 LoRA、特征混合器、块级重建、激活感知剪枝指标、KDE 可视化等技术。

**📊 数据集**

使用 WikiText2 评估困惑度，7 个零样本 NLP 任务（BoolQ、RTE、HellaSwag、WinoGrande、ARC-Challenge、ARC-Easy、OpenBookQA）进行评测。

**📈 对比分析**

与 SparseGPT、Wanda、RIA、ICP 等现有方法对比，实验表明在 30%–70% 稀疏率下性能保持竞争力，尤其在 60%–70% 稀疏率上显著优于对手；在推理效率上实现更高吞吐量。

**⚠️ 局限性**

在极高稀疏率（80% 及以上）性能下降明显，且目前仅验证于 LLaMA 与 OPT 系列，未来计划结合结构化/半结构化剪枝提升更高稀疏率的实用性。

---

## 192. Reading AI Model Compilation in MLIR Through the Lens of Formal Theories

**arXiv ID:** 2606.25244 | [PDF](https://arxiv.org/pdf/2606.25244v1)

**作者:** Javed Absar `[一作]` `[通讯]` (Qualcomm Technologies International, Ltd.), Javed Absar (Qualcomm Technologies International, Ltd.)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文通过将MLIR编译器设计与形式理论（项重写、精炼演算、抽象解释、类型理论和范畴理论）对应，阐明了抽象、转化和优化过程的理论基础，并给出了设计时应考虑的形式问题。

**💡 创新点**

创新点在于构建了一个跨理论的视角，提供了一套统一的形式语言，用来明确说出编译器设计的目标、约束和可验证性，并指出在代理编译器时代，设计瓶颈从实现转向抽象。

**🔧 技术方法**

用形式理论（TRS、精炼演算、抽象解释、类型理论、范畴理论）来解释和指导MLIR的匹配与重写、分阶段降低、分析与优化等机制。

**📊 数据集**

无；本文为理论性综述与设计指导，不涉及具体数据集。

**📈 对比分析**

本文未进行实验比较，主要通过理论分析与案例说明其可行性，未给出性能数据。

**⚠️ 局限性**

缺乏正式证明与实现细节，且对不同IR层级的具体适配仍需进一步研究。

---

## 193. Semantic Allocation in Ordered Bottlenecks: Predictive Residual Inference for Visual Representation Learning

**arXiv ID:** 2606.25232 | [PDF](https://arxiv.org/pdf/2606.25232v1)

**作者:** Erik Ayari `[一作]` (University of Tübingen), Martin V. Butz `[通讯]` (University of Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

本文提出并实现了一种名为 PRIOR 的层次化有序瓶颈学习框架，用于在不同预算下产生从粗到细的语义表示。

**💡 创新点**

创新点在于使用预测残差推断（Predictive Residual Inference）构建 log2 缩放的层级结构，并通过自我预测模块显式地将后续层的能力聚焦在未被前层解释的残差信息上，从而克服了传统 MBOP 方案的梯度曝光不足与无显式细化目标的问题。

**🔧 技术方法**

技术手段包括：层级化 token 组织、层级自我预测器（self‑predictive modules）、残差压缩与重构、log2 级别加权损失、离散/量化 token 参数化（Gaussian、Categorical、EMA‑VQ）以及对比学习和自动编码两种训练任务的应用。

**📊 数据集**

实验使用了 MVImgNet2 目标跟踪视频数据集进行对比学习，FFHQ 人脸图像数据集用于自编码重建。

**📈 对比分析**

与 MBOP‑TD、MBOP‑ITD 基线比较，PRIOR 在线性探测器分类精度、对比学习验证损失和 FFHQ 重建质量（SSIM、L1/L2、MS‑SSIM）等指标上均表现出更优或相近的性能，尤其在离散/量化 token 的低预算情境下显示出显著优势；PRIOR 的有序性也更为明显，后续层在精细读出上持续提升。

**⚠️ 局限性**

局限性包括：在 Gaussian token 设置下与 MBOP 的性能差距不大且偶有波动；模型在极低预算（单 token）下的表现仍低于部分基线；以及对不同任务、架构的泛化和进一步优化（如加权方案、层级目标）尚未完全探究。

---

## 194. Homomorphic Encryptions for Privacy Preserving Vision

**arXiv ID:** 2606.25216 | [PDF](https://arxiv.org/pdf/2606.25216v1)

**作者:** Preey Shah `[一作]` (Stanford University), Sanjari Srivastava `[通讯]` (Stanford University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究并实现了在同态加密环境下可直接对图像进行分类的卷积神经网络，并在多种数据集上进行实验验证。

**💡 创新点**

创新点在于在 TenSEAL 上实现了多通道输入、平均池化和多层卷积的同态加密操作，并用低阶多项式替代传统激活函数，使加密 CNN 在准确率上接近未加密基线且推理时间可控。

**🔧 技术方法**

使用了 Microsoft SEAL 的 CKKS 同态加密方案、TenSEAL 库、固定精度浮点数编码、多项式近似激活函数以及自定义的卷积与池化实现。

**📊 数据集**

实验数据集包括 MNIST、Kuzushiji‑MNIST、Fashion‑MNIST 以及 CIFAR‑10（灰度或彩色图像）。

**📈 对比分析**

通过将加密模型在测试集上的准确率与未加密模型基线进行对比，发现加密模型在 MNIST 上略有提升，Kuzushiji‑MNIST 与 Fashion‑MNIST 略有下降；推理时间比未加密慢约 1‑3 倍，且随着多项式模量提升时推理时间约翻倍。

**⚠️ 局限性**

限制主要在于缺乏 GPU 加速导致推理极慢，需要手动解密/重新加密多层卷积；加密参数调优繁琐；同态加密带来的计算与内存开销显著。

---

## 195. ASAP: Agent-System Co-Design for Wall-Clock-Centered Auto HPO Research for ML Experiments

**arXiv ID:** 2606.25207 | [PDF](https://arxiv.org/pdf/2606.25207v1)

**作者:** Taicheng Guo `[一作]` (University of Notre Dame), Xiangliang Zhang `[通讯]` (University of Notre Dame)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出ASAP框架，将多种 HPO 优化器与 LLM 判断器集成，利用代理-系统共设计显著降低端到端墙时间。

**💡 创新点**

创新点在于：①通过 LLM 判断器集成多样化优化器，突破单一先验的局限；②采用 KV‑cache 预设提示、跨迭代推测和自适应阈值，实现在不牺牲样本效率的前提下大幅压缩实际运行时间。

**🔧 技术方法**

技术上使用 GPT‑3.5‑turbo 作为 LLM 判断器和自适应调节器；并行调用 GP、TPE、SKOPT、TuRBO、Optuna、SMAC、DNGO、LLAMBO 等工具生成候选；实现 KV‑cache 复用提示、交叉迭代推测与相对误差接受测试，以及离线自适应阈值调节。

**📊 数据集**

实验使用 HPOBench（9 个表格任务）和 PD1（19 个深度学习任务，包括 Vision、Language、Protein）共 28 个任务。

**📈 对比分析**

与七种统计基线及 LLAMBO 对比，ASAP 在 30 次预算下在所有任务上取得最低归一化 regret，HPOBench 的墙时间提升约 1.13×，PD1 维持在同一 regret 水平，整体在 regret‑wall‑time 曲线上占优。

**⚠️ 局限性**

限制在于：当模型训练时间极短（≈5 s 以下）时推测收益有限；KV‑cache 提示的加速依赖于服务器支持 KV 缓存，若不支持则收益降低。

---

## 196. Efficient Adaptive Data Acquisition via Pretrained Belief Representations

**arXiv ID:** 2606.25197 | [PDF](https://arxiv.org/pdf/2606.25197v1)

**作者:** Daolang Huang `[一作]` (ELLIS Institute), Tom Rainforth `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种统一的贝叶斯决策框架——基于预训练预测基础模型的贝叶斯代理学习（POCIL），通过使用预训练的 Tabular Foundation Models（如 TabPFN、TabICL）作为信念状态编码器，在此基础上训练一个轻量级策略头，实现对贝叶斯实验设计、贝叶斯优化与主动学习等自适应数据采集任务的统一建模与推理。

**💡 创新点**

创新点在于（1）将信念表示与决策策略解耦，先利用预训练模型获得对任务相关后验信息的高质量表示，再在此表示上学习决策策略；（2）使用预训练的基础模型作为信念编码器，避免从零开始学习后验；（3）采用任务特定的奖励信号与局部信用分配的策略梯度训练方法；（4）显著提升样本效率（最高可达 100×-1000×）。

**🔧 技术方法**

技术方面主要包括预训练的 Tabular Foundation Models 作为隐藏表示提取器、基于这些表示的线性/神经策略头、局部信用分配的策略梯度优化、可选的基线预测损失进行后验表示微调、以及在池化候选集上的批量前向推理。

**📊 数据集**

实验使用了：贝叶斯实验设计的 Location‑Finding 与 CES 基准；HPO‑B 大规模机器学习超参优化数据集；Dockstring 分子对接分数高维指纹数据集；以及在主动学习场景下的预测损失驱动任务。

**📈 对比分析**

与随机采样、DAD、RL‑BOED、ALINE、NAP、Meta‑GP、PFNs4BO、TabICL 等基准相比，本文方法在所有任务上均实现了更低的期望信息增益/归一化遗憾、最高 100×-1000× 的样本效率，并在高维分子优化中获得最低遗憾。

**⚠️ 局限性**

局限性包括：仅支持离散候选池设计空间；使用局部信用分配的策略梯度可能对延迟收益的任务效果有限；当下游任务分布与预训练先验差异大时需要额外的微调；以及在连续设计空间下需要进一步改进候选生成或连续动作策略。

---

## 197. Sarashina2.2-TTS: Tackling Kanji Polyphony in Japanese Speech Generation via Data Scaling and Targeted Data Synthesis

**arXiv ID:** 2606.25369 | [PDF](https://arxiv.org/pdf/2606.25369v1)

**作者:** Lianbo Liu `[一作]` (SB Intuitions), Yui Sudo `[通讯]` (SB Intuitions)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Sarashina2.2‑TTS，一套专注于日语的 LLM‑TTS 系统，旨在解决日语汉字多音读音的核心挑战。

**💡 创新点**

创新点包括：1）利用 361k 小时日英混合语料进行大规模训练；2）构建 PronSteering 控制读音的文本侧读音调节器；3）针对所有 2,136 常用汉字及 4,378 读音的目标数据增强；4）设计 Kana‑CER 评估指标和 Joyo Kanji Yomi Benchmark，实现汉字级误差归因。

**🔧 技术方法**

核心技术：解码器‑only LLM（Sarashina2.2‑0.5B‑Instruct）生成语义 token；S3Tokenizer V2 语音 tokenizer；flow‑matching decoder（CosyVoice2）+ HiFi‑GAN 语音合成；PronSteering 读音控制；fine‑tuned Whisper 输出平假名的 Kana‑ASR；两阶段（预训练 + 细调）训练策略。

**📊 数据集**

使用数据集：361k 小时日英混合语音文本；约 320 小时合成数据覆盖所有常用汉字读音；JSUT（basic5000）和 CV3‑Eval 用于评测；构建的 Joyo Kanji Yomi Benchmark 共 13,095 句子。

**📈 对比分析**

评测方法：对比四大多语言 LLM‑TTS 基线，采用 Kana‑CER（句子级、汉字级）和标准 CER；计算 speaker similarity (SIM) 与自动 MOS；Sarashina2.2‑TTS Stage2 在 Joyo Kanji Yomi Benchmark 上取得 Kana‑CER_kanji 7.83、Kana‑CER_kanji^† 5.45，明显优于基线；跨提示（风格/语言）无降级；speaker similarity 与音质均高于同类系统。

**⚠️ 局限性**

局限性：对非日语提示的跨语言稳健性仍略逊；Kana‑ASR 仅基于音素，缺少语言模型补偿，可能产生误判；极罕见读音仍存在错误；公开模型不包含 PronSteering，需自行合成数据；系统在极端口语或俚语表达时可能产生轻微语音失真。

---

## 198. AI Coaching for Accelerating Human Skill Development with Reinforcement Learning

**arXiv ID:** 2606.25337 | [PDF](https://arxiv.org/pdf/2606.25337v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 199. Efficient Remote Sensing Instance Segmentation with Linear-Time State Space Distilled Visual Foundation Models

**arXiv ID:** 2606.25324 | [PDF](https://arxiv.org/pdf/2606.25324v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 200. TheoremGraph: Bridging Formal and Informal Mathematics

**arXiv ID:** 2606.25363 | [PDF](https://arxiv.org/pdf/2606.25363v1)

**作者:** Simon Kurgan `[一作]` (University of Washington Math AI Lab), Vasily Ilin `[通讯]` (University of Washington Math AI Lab)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了统一的命题级依赖图TheoremGraph，将11.7M arXiv非正式命题及18.3M依赖与Lean生态系统的388k声明及11.3M类型化边链接起来；同时发布相应提取器、API和数据集；

**💡 创新点**

创新点在于：①将非正式与正式数学的命题级依赖统一到一个语义空间；②利用LLM生成的slogan嵌入实现跨形式匹配；③在检索中加入名称-签名表示、查询重写、图展开等技术，显著提升检索效果；

**🔧 技术方法**

使用的技术包括：Qwen3-235B-A22B-Instruct-2507生成slogan；Qwen3-Embedding-8B生成4096维向量；HNSW近似检索与全精度余弦排序；LLM判定器（GPT‑5.4）评估匹配；Lean 4 kernel API抽取声明级依赖；多种提取器（确定性、启发式、符号化）抽取非正式依赖；API/MCP接口；

**📊 数据集**

数据集：arXiv数学子集（11.7M命题、18.3M依赖）；Lean 4项目25个（388k声明、11.3M边）；LeanGraph和非正式提取器输出；Slogan嵌入索引；蓝图对照集（约400对）做匹配评估；MathlibQR 200条查询与MathlibMPR 24条检索基准；

**📈 对比分析**

比较方法：对比LeanSearch‑v2的检索性能；在MathlibQR fair‑810子集上，基线回忆率0.586，nDCG@10 0.380，改进至Recall@10 0.775、nDCG@10 0.548；与LSv2重排后的Recall@10 0.780仅差0.5pp；在自动化形式化实验中，检索提升正确率从5/24到8/24，且输出token和工具调用更少；

**⚠️ 局限性**

局限性包括：非正式依赖提取仍不完整；跨形式匹配依赖LLM生成slogan与嵌入，可能失真；判定器仅基于有限上下文，专业判定覆盖不足；匹配仅考虑最近邻，缺少多候选检索；在链式前提检索任务上效果不佳；数据受arXiv授权限制，公开集有限。

---

## 201. Agentic Knowledge Tracing: A Multi-Agent LLM Architecture for Stealth Assessment of Financial Literacy in Serious Games

**arXiv ID:** 2606.25358 | [PDF](https://arxiv.org/pdf/2606.25358v1)

**作者:** Gabriel Santos `[一作]` (Federal University of Uberlândia), Marcelo Nascimento `[通讯]` (Federal University of Uberlândia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一套多智能体大型语言模型框架（Agentic BKT），可在不打断玩家体验的情况下，对2D平台类财务教育游戏中的玩家行为进行隐蔽评估，生成金融素养的掌握估计。

**💡 创新点**

创新点在于将事件分类、领域特定推理和专家判决三阶段融合为多智能体管线，并通过领域拆分与会话级推理显著提升了对学习效果的预测效度（约三倍）。

**🔧 技术方法**

使用了GPT‑4o mini进行事件等级分类、GPT‑5.2驱动的四个领域特定代理进行会话级推理，并采用贝叶斯知识追踪（BKT）模型进行掌握估计，最终由GPT‑5.2判决代理整合各领域得分。

**📊 数据集**

数据集来自193名K‑12学生在一款对准OECD/INFE财务素养框架的2D平台游戏中共完成264场游戏会话，产生15447条结构化事件记录，并配合前后测的15题财务素养测试。

**📈 对比分析**

与随机基线和单LLM BKT基线进行对比；Agentic BKT在学习增益和后测分数上的皮尔逊相关系数分别为0.276（p<0.001）和0.333（p<0.001），显著高于单LLM BKT（r≈0.095、0.112）且与前测无相关，说明其预测性能更优。

**⚠️ 局限性**

局限包括样本规模有限、仅在单一游戏上验证、依赖商业LLM导致成本与可复现性受限、BKT参数固定且未对不同参数进行全量敏感性分析，以及会话级推理的推断延迟较高。

---

## 202. Cache-Resident LLM Inference in GB-Scale Last-Level Caches

**arXiv ID:** 2606.25353 | [PDF](https://arxiv.org/pdf/2606.25353v1)

**作者:** Wanning Zhang `[一作]` (King Abdullah University of Science and Technology), Jian Weng `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了面向缓存驻留的LLM推理执行模型，并在多路CPU集群上实现了基于GB级最后一级缓存的高性能推理。

**💡 创新点**

创新点在于将模型权重与KV缓存分别放在独立的资源域（权重-注意力解耦），在子算子级别实现异步同步以削减全局同步开销，并采用静态线程池提升调度效率。

**🔧 技术方法**

采用三维堆叠缓存实现GB级LLC、权重-注意力分离、子算子级别同步、静态线程池、RDMA通信、INT8量化等技术。

**📊 数据集**

评估使用LLaMA-3.2-3B、LLaMA-2-7B、Qwen-3-8B及LLaMA-2-70B模型，全部采用INT8量化。

**📈 对比分析**

与广泛使用的llama.cpp基线比较，实验显示在4096长度、批量1–32时，TPOT加速达到2.04×–11.51×，推理吞吐率提升至12.5×左右；通过分析模型可预测更大规模模型达到13.9× TPOT 加速。

**⚠️ 局限性**

局限性包括：仅针对解码阶段，未覆盖前填充；对混合专家模型、连续批处理等场景适配有限；WA分离在小模型/小上下文时可能产生额外同步开销；KV缓存受限于LLC容量，高压力下仍需权衡延迟与吞吐率。

---

## 203. Decoupling Semantics and Geometric Grounding: Spatial Visual Prompts for Language-Conditioned Imitation Learning

**arXiv ID:** 2606.25360 | [PDF](https://arxiv.org/pdf/2606.25360v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 204. Cross-Subject Predictive Validity for Learning Outcomes of Delayed Start Behavior

**arXiv ID:** 2606.25308 | [PDF](https://arxiv.org/pdf/2606.25308v1)

**作者:** Jordan Gutterman `[一作]` (Carnegie Mellon University), Vincent Aleven `[通讯]` (Carnegie Mellon University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了数学学习平台 iReady 中“延迟开始”行为对学生在数学与英语标准化考试成绩的跨学科预测效能，并基于混合模型与敏感性分析识别了两类学生子群（早起者与长期拖延者）

**💡 创新点**

创新点在于：①首次验证“延迟开始”作为跨学科通用的自我调节行为指标；②通过数据驱动的两步阈值确定（混合模型 + 敏感性分析）构建可操作且与教师日常观察一致的子群划分；③提出了原始分钟数与课堂相对延迟两种易解释的指标，兼顾可解释性与预测力

**🔧 技术方法**

使用回归模型（控制先前成绩）评估行为指标对后测成绩的影响；利用高斯混合模型（GMM）识别延迟开始分布中的潜在群体；通过设定不同分钟阈值的敏感性分析验证子群边界；同时对模型表现采用 R²、BIC 等统计量进行比较

**📊 数据集**

数据集为 711 名七年级学生在 2022–2024 学年使用 iReady 进行数学练习的日志，共计 3.6 百万条交易记录，包含 841 个班级工作会话

**📈 对比分析**

相较于仅包含先前成绩的基线模型，加入“延迟开始”指标后数学成绩的效应为 -0.07 SD（p≈0.02），英语成绩的效应为 -0.10 SD（p<0.001）；使用相对延迟度量可进一步提升 R² 与 BIC；早起者相对基线提升约 0.11 SD（数学）/0.15 SD（英语），长期拖延者下降约 0.13 SD（数学）/0.11 SD（英语）

**⚠️ 局限性**

局限性包括：仅在单一中学、单一 45 分钟课堂时段、单一平台环境下验证，阈值和子群划分可能随课堂时长、教师教学风格或不同学习平台而变；仅研究了数学练习延迟对英语成绩的预测，未检验相反方向；未进一步探讨延迟开始与其他自我调节或人格特质的收敛效度

---

## 205. Stagnant Neuron: Towards Understanding the Plasticity Loss in Multi-Agent Reinforcement Learning Value Factorization Methods

**arXiv ID:** 2606.25335 | [PDF](https://arxiv.org/pdf/2606.25335v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 206. Decoupling Reconnaissance and Exploitation: Measuring the Capability Boundaries of LLM-Based Web Penetration Testing

**arXiv ID:** 2606.25332 | [PDF](https://arxiv.org/pdf/2606.25332v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 207. KidRisk: Benchmark Dataset for Children Dangerous Action Recognition

**arXiv ID:** 2606.25298 | [PDF](https://arxiv.org/pdf/2606.25298v1)

**作者:** Minh-Kha Nguyen `[一作]` (University of Science), Trung-Nghia Le `[通讯]` (University of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

构建了专门针对儿童危险动作的KidRisk数据集，并利用BLIP-2视觉-语言模型与LSTM进行迁移学习，完成了儿童动作与危险动作的识别。

**💡 创新点**

创新点在于：①首次公开了大规模儿童危险动作数据集KidRisk；②将预训练的BLIP-2模型与时序网络结合，显著提升了对儿童动作的上下文理解；③通过零样本学习与迁移学习相结合，减少了对标注数据的依赖。

**🔧 技术方法**

主要技术包括BLIP-2视觉-语言预训练模型、Q-Former交叉注意力、LSTM时序建模、零样本学习、迁移学习、二元交叉熵与交叉熵损失、L2正则化以及数据增强。

**📊 数据集**

使用了KidRisk数据集（2500段儿童动作视频 + 10000张危险/安全图片），并以InfAct等公开数据为基础进行扩充与标注。

**📈 对比分析**

与传统3D-CNN、ViT等模型对比，BLIP-2+LSTM在动作分类上达到了83.5%准确率，在危险检测上达到了96.1%，明显优于S3D、Alpro、ResNet和ViT等基线模型。

**⚠️ 局限性**

局限性包括：①数据集规模相对有限，仍需更多多样化场景；②危险与安全标签不平衡导致模型偏向多数类；③对极端环境或罕见动作的泛化能力尚待验证；④尽管迁移学习降低了训练成本，但模型在实时监控场景下的推理速度与资源占用仍需进一步优化。

---

## 208. Data-Driven Evolution of Library and Information Science Research Methods (1990-2022): A Perspective Based on Fine-grained Method Entities

**arXiv ID:** 2606.25320 | [PDF](https://arxiv.org/pdf/2606.25320v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 209. LEVIRDet: A Million-Scale 159-Category Dataset and Foundation Model for Universal Remote Sensing Object Detection

**arXiv ID:** 2606.25312 | [PDF](https://arxiv.org/pdf/2606.25312v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 210. Minimum-Weight Steiner Triangulation of Convex Polygons Requires Interior Steiner Points

**arXiv ID:** 2606.25302 | [PDF](https://arxiv.org/pdf/2606.25302v1)

**作者:** David Eppstein `[一作]` (University of California), Zahra Hadizadeh `[通讯]` (University of California)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

构造了一个凸多边形，证明在最小权重Steiner三角剖分中，内部Steiner点有时比仅使用边界Steiner点更优，从而给出对Eppstein 1994年猜想的反例。

**💡 创新点**

首次提供了需要内部Steiner点的最小权重Steiner三角剖分的凸多边形反例，打破了仅需边界Steiner点的假设。

**🔧 技术方法**

采用几何构造、凸多边形三角剖分的动态规划求最优解、距离上界与下界分析以及系统的案例归约与对比证明。

**📊 数据集**

使用了手工设计的28个顶点坐标的凸多边形P（包含内部Steiner点p*），未使用公开数据集。

**📈 对比分析**

通过动态规划计算无Steiner点的最优权重为49425.5605；仅边界Steiner点亦保持此值；加入单个内部Steiner点后权重降至49425.4212，表明内部Steiner点能略微降低总长度。

**⚠️ 局限性**

仅针对单一构造示例，未给出多边形一般解法；证明过程复杂且依赖大量几何不等式，尚未确定该问题的整体计算复杂度（是否NP‑hard或多项式可解）。

---

## 211. Geometry-Anchored Transport Framework for Exemplar-Free Class-Incremental Learning

**arXiv ID:** 2606.25347 | [PDF](https://arxiv.org/pdf/2606.25347v1)

**作者:** Hongye Xu `[一作]` (Rochester Institute of Technology), Bartosz Krawczyk `[通讯]` (Rochester Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Geometry‑Anchored Transport Framework，在无样本类别增量学习中将特征传输嵌入训练阶段，稳定决策边界并避免后期传输误差；

**💡 创新点**

将特征传输从后期独立的修正转变为训练时的几何约束，引入Analytic Geometric Anchor（基于Mahalanobis对齐的闭式线性映射）和Topology‑Aware Evolution，实现对全局几何漂移的实时正则化；

**🔧 技术方法**

使用解析几何锚点（Sylvester方程求解线性映射）、全局线性回归（GLS）、残差MLP校正、EMA平滑、逆协方差（Mahalanobis）判别、蒸馏等技术；

**📊 数据集**

在CIFAR‑100、TinyImageNet、ImageNet‑100（从零训练）以及预训练backbone下的CUB‑200等数据集上进行实验；

**📈 对比分析**

与EWC、LwF、SDC、PASS、FeTrIL、FeCAM、EFC、ADC、LDC、AdaGauss、DPCR等基线对比，所有任务数下均实现A_last和A_inc均提高3–12%，并在多数场景中位居榜首；

**⚠️ 局限性**

仅适用于单一类条件高斯统计，面对高度非线性或剧烈表征漂移时仍可能失效；残差网络的效果受线性先验质量限制；理论保证仅在高斯假设下成立。

---

## 212. Improved Large Language Diffusion Models

**arXiv ID:** 2606.25331 | [PDF](https://arxiv.org/pdf/2606.25331v1)

**作者:** Shen Nie `[一作]` (Renmin University of China), Ji-Rong Wen `[通讯]` (Renmin University of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并训练了 iLLaDA，一种 8B 规模、完全双向的扩散式语言模型，从零开始训练，并将预训练规模扩大到 12T tokens；

**💡 创新点**

在模型设计、学习率调度、SFT 格式、变量长度生成以及多选题自信度评分等方面做了多项实用改进，并通过分批随机长度训练与变量长度生成显著提升性能；

**🔧 技术方法**

采用了基于掩码的扩散目标、Grouped‑Query Attention、RMSNorm、SwiGLU、RoPE、FlashAttention 变长内核、AdamW 等技术；

**📊 数据集**

预训练使用 12T tokens 的大规模文本语料；SFT 使用 25B tokens 的指令数据；评测使用 MMLU、BBH、ARC‑Challenge、HellaSwag、GSM8K、Math、HumanEval、MBPP、MMLU‑Pro、MMLU‑Redux 等标准基准；

**📈 对比分析**

与 LLaDA 8B、Dream 7B、Qwen2.5 7B 进行对比，iLLaDA 基础版在大部分任务上略优于 Qwen2.5 Base，且在 MMLU、BBH、ARC‑Challenge、GSM8K 等任务上获得最佳成绩；指令版 iLLaDA 超越 LLaDA 与 Dream，虽在部分数学与代码任务仍落后 Qwen2.5 Instruct，但与其差距显著缩小；

**⚠️ 局限性**

尚未加入强化学习对齐，导致指令版仍与顶尖自回归模型存在差距；实验仅限 8B 规模，未进行更大规模对比；SFT 训练受算力限制仅完成 12 轮，进一步训练可能带来更多提升。

---

## 213. Invoice Haystack: Benchmarking Document Retrieval and Visual Question Answering Under Strong Visual Homogeneity

**arXiv ID:** 2606.25343 | [PDF](https://arxiv.org/pdf/2606.25343v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 214. Neural Machine Translation for Low-Resource Tangkhul--English

**arXiv ID:** 2606.25365 | [PDF](https://arxiv.org/pdf/2606.25365v1)

**作者:** Chormi Zimik Vashai `[一作]` (Independent Researcher), Agniva Maiti `[通讯]` (KIIT University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了首个唐克尔语-英语机器翻译系统，完成模型训练与评测。

**💡 创新点**

收集并公开38,336句平行语料；使用字节级ByT5大模型细调，显示其在带重音符号拉丁文本中的显著优势。

**🔧 技术方法**

采用基于Transformer的ByT5-large（1.23B参数）和mT5-small（300M参数）多语言seq2seq模型，分别使用字节级编码与SentencePiece进行微调。

**📊 数据集**

主要使用来自圣经、故事与对话的38,336句唐克尔语-英语平行语料。

**📈 对比分析**

在3,856句测试集上，ByT5-large达39.97 BLEU、58.07 chrF++、BERTScore 0.8104、COMET 0.7302，明显优于mT5-small（12.21 BLEU）和零射击mT5-base（0.03 BLEU）。

**⚠️ 局限性**

局限包括数据主要来自圣经导致域偏差；对非宗教或现代语料的泛化不足；评测指标与人工质量不完全一致；仅完成单向翻译。

---

## 215. Programmable Probabilistic Computer with 1,000,000 p-bits

**arXiv ID:** 2606.25313 | [PDF](https://arxiv.org/pdf/2606.25313v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 216. Physics Question Scene Graph: Fine-grained Evaluation of Physical Plausibility in Text-to-Video Generation

**arXiv ID:** 2606.25306 | [PDF](https://arxiv.org/pdf/2606.25306v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 217. Three Buddhist Vocabularies: Computational Stylometry of the English Pali Canon across Sutta, Vinaya, and Abhidhamma

**arXiv ID:** 2606.25372 | [PDF](https://arxiv.org/pdf/2606.25372v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 218. Efficient and Trainable Language Model Test-Time Scaling via Local Branch Routing

**arXiv ID:** 2606.25354 | [PDF](https://arxiv.org/pdf/2606.25354v1)

**作者:** Yutong Yin `[一作]` (Northwestern University), Zhaoran Wang `[通讯]` (Northwestern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于本地分支路由（Local Branch Routing，LBR）的测试时扩展框架，利用在自回归生成过程中构造并前向传播局部看ahead树，随后通过轻量级路由器在候选分支的隐藏状态上做决策，最终只保留并继续使用被路由的子树进行解码。

**💡 创新点**

创新点在于将分支决策从单一的根节点下一步分布迁移到已前向传播的后续隐藏状态上，既保持了离散分支身份，又通过可计算的树轨迹似然实现端到端的强化学习；同时通过 prune–shift–grow 机制避免了全局搜索的代价。

**🔧 技术方法**

使用技术包括：局部前向树生成、set‑attention 路由器、prune‑shift‑grow 解码流程、树轨迹似然的计算、以及基于可验证奖励的 RLVR（GRPO）训练。

**📊 数据集**

在合成层级规划任务（radix‑translated 图可达性）以及六个数学推理基准（AIME、MATH500、AMC、Olympiad 等）上进行评估，训练数据来自 DeepScaleR‑Preview 数据集。

**📈 对比分析**

与离散链式思考、普通离散‑token RLVR、以及软‑token 分支（Multiplex Thinking）等基线比较，LBR 在 Pass@1 和 Pass@32 上均实现显著提升，且更深的 lookahead（L=2）进一步提高性能。

**⚠️ 局限性**

局限性在于仅利用局部看ahead信息，缺乏全局规划或完整解空间搜索；增加 lookahead 深度会显著提升计算成本，限制了在更大规模任务上的可扩展性。

---

## 219. Lifelong In-Context Learning with Transformers Requires Parametric Forms of Attention

**arXiv ID:** 2606.25342 | [PDF](https://arxiv.org/pdf/2606.25342v1)

**作者:** Luke McDermott `[一作]` (University of California San Diego), Rahul Parhi `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

将注意力机制重新表述为自监督的在线回归问题，提出在固定硬件预算下通过参数化注意力实现终身上下文学习，并探讨测试时训练的目标与更新策略。

**💡 创新点**

创新点在于：①把注意力视为在线学习/回归，①提出参数化注意力替代无限增长的KV缓存，②将测试时训练作为实现终身学习的直接手段，并系统性列举了相关开放问题。

**🔧 技术方法**

所用技术包括线性注意力、状态空间模型（SSM）、快速权重编程、测试时训练层、非参数核回归（如软最大注意力的核回归表述）以及多头/稀疏注意力等。

**📊 数据集**

论文主要为理论与框架性工作，未在具体数据集上做实验，因而未提供任何数据集信息。

**📈 对比分析**

由于缺乏实验，论文未给出方法的对比结果，也未列出具体性能指标；讨论侧重于概念验证与开放方向。

**⚠️ 局限性**

主要限制包括：更新效率低、内存容量受限、目标函数与正则化设计尚不成熟、难以捕捉长时序趋势、缺乏大规模实验验证，以及如何在实际大模型中实现可扩展性等。

---

## 220. Omni-Perception Policy Optimization for Multimodal Emotion Reasoning

**arXiv ID:** 2606.25325 | [PDF](https://arxiv.org/pdf/2606.25325v1)

**作者:** Zhiyuan Han `[一作]` (University of Science and Technology of China), Xun Yang `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过强化学习框架 Omni-Perception Policy Optimization 对多模态情绪推理进行优化，显式提升模型对视觉、音频和情绪线索的利用与可信度。

**💡 创新点**

引入 Omni-Perception Reward 与 Loss，利用细粒度证据覆盖奖励和基于单模态遮蔽的 KL 损失，解决传统方法在多模态证据利用不足和跨模态幻觉问题。

**🔧 技术方法**

采用 GRPO 强化学习、句子拆分、语义相似度匹配、Evidence‑Routing Matrix、KL 损失对比、单模态遮蔽等技术。

**📊 数据集**

使用 MER‑UniBench、MME‑Emotion 以及自构建的 MEP‑Bench（取自 OV‑MERD）等多模态情绪数据集。

**📈 对比分析**

在统一 checkpoint 下与 AffectGPT‑R1、HumanOmni‑V2 等基线对比，MER‑UniBench 平均分提升至 81.05%（比基线提升 5.45%），MEP‑Bench 上 cue recall 从 53% 提升至 70.44%，faithfulness 提升至 76.4% 等。

**⚠️ 局限性**

对遮蔽比例、阈值等超参敏感，模型在极端跨模态关联场景下仍可能出现微量幻觉，且超参稳定性与可扩展性待进一步验证。

---

## 221. Hybrid-IR: Dual-Path Hybrid Retrieval with Iterative Reasoning for Complex Medical Question Answering

**arXiv ID:** 2606.25338 | [PDF](https://arxiv.org/pdf/2606.25338v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 222. State Space Models Meet Remote Sensing: A Survey

**arXiv ID:** 2606.25329 | [PDF](https://arxiv.org/pdf/2606.25329v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 223. Conformal Recovery-Deadline Certificates for Runtime Assurance of Adapting Controllers

**arXiv ID:** 2606.25371 | [PDF](https://arxiv.org/pdf/2606.25371v1)

**作者:** Alireza Shojaei `[一作]` `[通讯]` (Virginia Tech), Alireza Shojaei (Virginia Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于分裂式合成预测的恢复时间上界（conformal recovery‑deadline certificate），用来在 Runtime Assurance（RTA）框架中为在线自适应控制器提供延迟回退的可靠保障，替代传统的即时锁定（latching）策略。

**💡 创新点**

创新点在于：①将恢复时间作为可度量对象，使用分裂式、无分布假设的合成预测构造置信上界；②通过“可靠性异质设计”实现统计层与形式验证层的分离，保证安全阈值不受统计误差影响；③提供加权和Mondrian版本以补偿分布漂移和类别不平衡。

**🔧 技术方法**

主要技术包括：分裂式合成预测（split‑conformal）、加权合成预测（weighted conformal）、Mondrian 合成预测（group‑conditional conformal），以及 Simplex 体系结构下的 Runtime Assurance 与安全监测。

**📊 数据集**

数据集：① 6-DOF 航天器姿态控制模拟（Basilisk）中的反向增益失效；② 逆摆（torque‑controlled pendulum）模拟中的扭矩符号失效；两组数据均包含多种失效类型和随机噪声，覆盖了两种不同动态平台。

**📈 对比分析**

方法对比：与传统锁定(Simplex latch)、p95×1.3启发式、以及基于安全值的合成预测（CP‑on‑safety‑value）相比；结果显示：锁定策略几乎完全失去自主性；启发式过度覆盖且无置信保证；合成预测恢复‑deadline 在保证覆盖率（≈1−α）同时保留约90% 的自主性，且安全逃逸率为100%。

**⚠️ 局限性**

局限性：① 仅提供边际覆盖率（marginal coverage），无法给出每一类故障的保证；② 需满足可交换性假设，若失效分布漂移需已知或估计似然比；③ 证书需要针对每个控制器与故障类别单独校准；④ 当无法给出有限期限时返回 +∞，导致无法执行延迟回退。

---

## 224. HiFiVe: High-Fidelity Vehicle Generation Leveraging Auto-Regressive 2D Generative Priors

**arXiv ID:** 2606.25300 | [PDF](https://arxiv.org/pdf/2606.25300v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 225. Reliability-Asymmetric Spacecraft Autonomy: Co-Designing a Capable Learned GNC Stack with a Verified, Adaptation-Aware Runtime Shield

**arXiv ID:** 2606.25366 | [PDF](https://arxiv.org/pdf/2606.25366v1)

**作者:** Alireza Shojaei `[一作]` `[通讯]` (Virginia Tech), Alireza Shojaei (Virginia Tech)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并验证了一套可靠异构的航天器GNC堆栈AMPLE-GNC，融合了小型预训练语言模型指令生成、行动可行性验证器、学习型适应控制器和形式化安全盾，实现了在飞行计算资源下同时具备能力与可验证性的自主系统。

**💡 创新点**

核心创新是采用可靠性不对称设计理念，将可验证层（语法约束输出、Kind‑2验证的线性时序逻辑盾、分布式分层适应感知运行时保证）与不可验证层（语言模型指令、Rapid Motor Adaptation控制器）共设计，解决了传统学习模型可解释性与可验证性矛盾。

**🔧 技术方法**

技术手段包括：低秩微调SmolLM2‑360M + GBNF语法约束解码；PDDL+规划与可行性证书；GRU+PD的RMA在线故障识别与控制；Kind‑2模型检查的二阶预测器；分布式分层运行时保证与分位分布式置信上界（conformal）恢复截止时间。

**📊 数据集**

使用的数据集包括：NASA‑operations 20模板的自然语言指令集（已去重的de‑leaked split），基于随机化的actuator fault库（sign、连续增益、偏置、全失效），以及星图仪噪声仿真等。

**📈 对比分析**

通过在6‑DOF Basilisk仿真床上与传统PD、经典自适应、Nussbaum‑gain、端到端RL等基线对比，RMA学生在“settled science gate”下对持有fault的恢复率为97.8%（sign）/94.4%（连续增益），盾保持100%自主；DSN通话需求减少超过90%；对抗性测试门成功率达到93.6%。

**⚠️ 局限性**

局限性包括：仅针对actuator effectiveness故障（sign、连续增益、偏置、完全失效）进行验证；在纯偏置和全失效场景下性能有限；端到端RL方法失败；恢复时间证书依赖于故障分布与校准一致；架构在单一仿真环境中验证，需进一步推广至更复杂故障和更大规模模型。

---

## 226. Supervised Post-training of Speech Foundation Models for Robust Adaptation in Speech Deepfake Detection

**arXiv ID:** 2606.25328 | [PDF](https://arxiv.org/pdf/2606.25328v1)

**作者:** Zihan Pan `[一作]` (Institute for Infocomm Research (I2R), Agency for Science, Technology and Research (A*STAR)), Jinyang Wu `[通讯]` (Institute for Infocomm Research (I2R), Agency for Science, Technology and Research (A*STAR))

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种基于混合帧的后训练框架，利用帧级监督提升语音基础模型在深度伪造检测上的鲁棒性。

**💡 创新点**

创新点在于通过在同一录音内插入对立类的短段混合帧来生成局部伪造扰动，并在后训练阶段提供帧级标签，使模型学习对短时不一致性的敏感度。

**🔧 技术方法**

采用WavLM Large作为编码器，结合LoRA适配器进行低秩微调，使用混合帧生成、帧级二分类头以及注意力聚合的下游分类器。

**📊 数据集**

在ASVspoof 5、2019 LA、2021 LA/DF等公开基准上进行训练与评估。

**📈 对比分析**

与现有单模型、无增广系统对比，单模型EER 4.50%（ASV5），ASV21 LA/DF平均EER 3.96%、最大EER 4.04%、LA–DF差距仅0.16%，在低资源微调下显著提升。

**⚠️ 局限性**

局限性包括对极端混合比例效果不佳、仍未达到多模型融合或增广方法的最优EER，且对极端低资源场景的鲁棒性仍有提升空间。

---

## 227. DynaMOMA: Instantaneous Prediction of Grasp Poses for Mobile Manipulation of Dynamic Objects

**arXiv ID:** 2606.25295 | [PDF](https://arxiv.org/pdf/2606.25295v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 228. Hypergraph Normal World Models for Logical Visual Anomaly Detection

**arXiv ID:** 2606.25368 | [PDF](https://arxiv.org/pdf/2606.25368v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 229. ESTANet: Efficient Online Error Detection in Procedural Videos via Prediction Inconsistency

**arXiv ID:** 2606.25317 | [PDF](https://arxiv.org/pdf/2606.25317v1)

**作者:** Shih-Po Lee `[一作]` (Honda Research Institute), Behzad Dariush `[通讯]` (Honda Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出轻量级的ESTANet框架，用预测不一致性实现实时在线错误检测；

**💡 创新点**

创新点在于利用标准与误差敏感动作检测器在不同时间窗口下的预测差异，结合TAD模块和自适应窗口选择，既捕捉执行错误又捕捉程序错误；

**🔧 技术方法**

使用GRU+MLP动作检测器、Temporal‑Aware Dynamic (TAD)模块、短/长时间窗口训练策略以及投票决策；

**📊 数据集**

在EgoPER、Assembly‑101‑O和EPIC‑Tent‑O三大现实场景数据集上进行评测；

**📈 对比分析**

与MistSense、PREGO、DTGL、MSGI等基线对比，ESTANet在EgoPER的F1@10/F1@25/F1@50分别达到47.2%/37.8%/21.6%，在Assembly‑101‑O与EPIC‑Tent‑O的Avg‑F1提升约6%/10%，并实现24.4FPS的实时推理；

**⚠️ 局限性**

局限性包括：对细微错误不敏感、首段无上下文时误判、模型容量受限导致误报和过度依赖时间先验。

---

## 230. SafeGen: LLM-Driven Assertion Generation and Fault Criticality Evaluation for Functional Safety

**arXiv ID:** 2606.25296 | [PDF](https://arxiv.org/pdf/2606.25296v1)

**作者:** Xuanyi Tan `[一作]` (Arizona State University), Krishnendu Chakrabarty `[通讯]` (Arizona State University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了SafeGen框架，实现LLM驱动的功能安全断言生成和门级故障严重性评估，并在FOC系统上验证其有效性。

**💡 创新点**

创新点在于将LLM与正式验证相结合进行语义级故障严重性评估；构建文档级HyperKG并与RTL CDFG融合，实现文档与代码的语义对齐；支持桥接故障映射并通过FPV计算严重性；实现断言与规范的可追溯映射。

**🔧 技术方法**

采用大语言模型（GPT‑5）、HyperGraphRAG与HyperKG、RTL CDFG、门级到RTL故障映射、Formal Property Verification（JasperGold）、数字‑物理共模仿真等技术。

**📊 数据集**

使用FOC开源设计的RTL、配套的物理模型及自制功能安全文档（包含FMEDA、设计说明），以及由Synopsys TestMAX生成的门级Stuck‑at与Bridging故障列表和断言集。

**📈 对比分析**

与AssertLLM、AssertionForge在断言质量（#SVA、#SynC、#Proven、COI覆盖）以及与结构/功能故障仿真在故障检测率对比。SafeGen的断言质量明显更高，FPV检测率达80%+；相较结构仿真，SafeGen更为精确且整体执行时间从5.5天降至约3.7天。

**⚠️ 局限性**

主要限制包括：对LLM生成规范和评分的可靠性仍有依赖；在极大规模设计上的可扩展性尚未完全验证；桥接故障映射过程复杂，需手动调优；当前实现主要针对RTL级别，尚未覆盖更低层硬件实现。

---

## 231. Compositional Behavioral Semantics for State Abstraction in Reinforcement Learning

**arXiv ID:** 2606.25357 | [PDF](https://arxiv.org/pdf/2606.25357v1)

**作者:** Yivan Zhang `[一作]` (University of Tokyo), Manuel Baltieri `[通讯]` (Araya Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了统一的协作框架，用 coalgebra 描述 RL 状态抽象，并通过 bundle、lifting、pullback、pushforward 等概念构建行为结构，证明了抽象映射下行为结构的可传递性和从逻辑到定量语义的构造。

**💡 创新点**

创新点在于将各种行为结构（安全性、价值函数、bisimulation 等）统一为可组合的“bundle”并通过同构/自然变换实现抽象传递；提供了通用的行为结构传递定理，首次在单一框架下关联逻辑与定量语义；并将模型不相关性抽象、价值等价性等现象视为 coalgebra homomorphism 的特例。

**🔧 技术方法**

核心技术包括：1) coalgebraic 表述系统动力学；2) bundle 与 lifting 的抽象构造；3) pullback 与 pushforward 传递操作；4) 同构/自然变换实现不同类型系统间的映射；5) 通过闭包算子定义行为结构，并用定理证明其在抽象映射下的安全/可构造性。

**📊 数据集**

该工作为纯理论研究，未使用具体数据集进行实验；主要基于数学证明与框架构造。

**📈 对比分析**

因缺乏实验验证，本论文未给出方法比较或性能指标；所有结果均来自理论证明。

**⚠️ 局限性**

局限性包括：1) 仅关注严格同构（exact homomorphism），不考虑松弛（lax）或近似抽象；2) 结果为理论层面，缺乏实践算法或实验验证；3) 只适用于单一抽象映射，未探讨多层或非结构化抽象；4) 只针对马尔科夫过程/隐藏马尔科夫模型等常见模型，未覆盖更广泛的 RL 环境。

---

## 232. Representation Matters: An Empirical Study of Program Representations for LLM Vulnerability Reasoning

**arXiv ID:** 2606.25356 | [PDF](https://arxiv.org/pdf/2606.25356v1)

**作者:** Andrew Stoltman `[一作]` (University at Buffalo), Haipeng Cai `[通讯]` (University at Buffalo)

**通讯引用:** 2302 | [OpenAlex ID](https://openalex.org/A5076081056)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立了一套基准框架，评估不同程序表示（源代码、AST、CFG、PDG及其组合）对大型语言模型在漏洞检测任务中的效果。

**💡 创新点**

发现图结构表示比原始源代码更能提升LLM漏洞推理，并提出“上下文稀释”现象；展示AST+PDG组合最佳效果，并比较提示开销。

**🔧 技术方法**

采用Joern静态分析生成AST/CFG/PDG，使用VulChecker/Hector产生ePDG；将图序列化为文本；使用固定Chain-of-Thought提示和结构化JSON输出；在GPT-4-like模型上评估。

**📊 数据集**

基于PrimeVul的107条C/C++函数级漏洞案例（5个CWE），以及19条ePDG辅助案例。

**📈 对比分析**

在同一任务下分别对10种表示变体进行单次推理；结果显示AST+PDG 83.2%（curated）vs.原始源53.5%；图结构提示在精度和提示字符数上均优于源代码或源+图组合。

**⚠️ 局限性**

仅使用单一模型和单次推理；CWE分布不均；图序列化方法可能影响结果；ePDG生成受限；提示中的少量例子选择可能混杂影响。

---

## 233. Self Capacitive Tactile Sensor System designed for Companion Robots

**arXiv ID:** 2606.25348 | [PDF](https://arxiv.org/pdf/2606.25348v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 234. Follow Your Track: Precise Skeleton Animation Controlled by 3D Trajectories

**arXiv ID:** 2606.25344 | [PDF](https://arxiv.org/pdf/2606.25344v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 235. Inner and Outer Bounds on the Secrecy Capacity of Degraded Broadcast Channels with RMSI and Transmitter CSI

**arXiv ID:** 2606.25351 | [PDF](https://arxiv.org/pdf/2606.25351v1)

**作者:** Saeid Pakravan `[一作]` (University of Quebec in Montreal), Ghosheh Abed Hodtani `[通讯]` (Ferdowsi University)

**通讯引用:** 1051 | [OpenAlex ID](https://openalex.org/A5038472266)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究在带有非因果信道状态信息（CSI）和接收方信息侧信（RMSI）的退化广播信道中，构造并证明了一组内外界限，完整表征了在存在外部窃听者时的保密容量。

**💡 创新点**

创新点包括：①首次同时考虑CSI与RMSI的交互效应；②对补全与非补全RMSI两种结构给出完全可实现与外部界限；③在Gaussian实例中证明CSI可通过“写在脏纸”效应显著提升保密速率。

**🔧 技术方法**

采用的技术主要有：随机编码、superposition编码、Marton编码、Gel'fand‑Pinsker编码（对CSI处理）、典型性分析、Fano不等式、Csiszár‑Körner信息恒等式以及时间共享随机变量。

**📊 数据集**

论文为理论研究，无具体实验数据集，所有结果均基于信息理论模型推导与符号计算。

**📈 对比分析**

通过对比所推导的内界与外界限，并在Gaussian例子中绘制速率区域曲线，验证在高信噪比下内界与外界限趋近，表明所用编码策略可实现近似最优的保密速率；在低信噪比时仍存在一定误差。

**⚠️ 局限性**

局限性包括：仅针对退化广播信道，非退化或多用户扩展尚未覆盖；Gaussian示例仅考虑理想化的噪声与CSI假设；对CSI误估或时变性未作讨论；理论推导复杂，实际实现难度较大。

---

## 236. General Techniques for Reducing Key-Switching Overhead in Privacy-Preserving Two-Party Transformer Inference

**arXiv ID:** 2606.25349 | [PDF](https://arxiv.org/pdf/2606.25349v1)

**作者:** Wenshao Yang `[一作]`, Dongdong Yao `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出了预处理辅助的Transformer注意力计算方法，降低在线推理阶段的密钥切换（key‑switch）开销，并给出了存储与通信权衡策略；同时提出了融合重线化与旋转的密钥切换优化。

**💡 创新点**

创新点在于：① 关注级别的预处理拆分，使得密钥切换移至离线阶段，兼容任意RLWE基准的打包方案；② 通过存储‑通信折中，用在线轻量通信代替大量预计算密文；③ 在CKKS乘法后旋转模式下，首次将重线化与旋转合并为单个关键切换步骤，减少一次ModDown。

**🔧 技术方法**

技术主要包括：全同态加密（CKKS）与混合FHE‑MPC；加密矩阵乘法与预处理；加密旋转、重线化与键切换；加密与共享掩码的混合；存储‑通信权衡的通信协议；安全性基于IND‑CPA与半诚实模型的模拟证明。

**📊 数据集**

论文为理论分析，未在公开数据集上进行实验评估；所给示例采用标准Transformer参数尺寸（例如BERT‑tiny、GPT‑2、LLaMA等）进行性能对比。

**📈 对比分析**

与Arion、BLB等现有安全Transformer推理系统对比：在线密钥切换次数从数千次降至数十甚至零；预处理存储从数千密文降至百级甚至少数密文；通过存储‑通信折中进一步降低存储至百级密文，在线通信量略增。整体推理延迟显著下降，尤其在第一层（FHE）或全层（混合）推理中表现突出。

**⚠️ 局限性**

局限性：① 预处理阶段需要额外离线计算和存储，适配大模型时存储仍是瓶颈；② 仅针对线性层优化，对非线性Softmax等操作仍需MPC或近似；③ 目前仅做理论与定量分析，缺乏完整端到端实验验证；④ 对于极大批量或低带宽环境，通信开销虽小但仍需考虑。

---

## 237. Memory Makes the Difference: Evaluating How Different Memory Roles Shape Conversational Agents

**arXiv ID:** 2606.25361 | [PDF](https://arxiv.org/pdf/2606.25361v1)

**作者:** Yuxin Wang `[一作]`, Nick Craswell `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

本文介绍了ACL会议论文的格式化规范和使用模板的说明

**💡 创新点**

并未提出任何科研创新点

**🔧 技术方法**

主要使用LaTeX排版技术和模板文件

**📊 数据集**

不涉及任何数据集

**📈 对比分析**

没有进行实验或方法比较，因而无性能数据

**⚠️ 局限性**

由于是格式化指南，缺乏研究内容，适用范围有限

---

## 238. Measuring Research Difficulty of Academic Papers: A Case Study in Natural Language Processing

**arXiv ID:** 2606.25307 | [PDF](https://arxiv.org/pdf/2606.25307v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 239. WaveForward: An Omnidirectional Passive Wheeled Quadruped Robot with Casters

**arXiv ID:** 2606.25299 | [PDF](https://arxiv.org/pdf/2606.25299v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 240. Minimalist Preprocessing Approach for Image Synthesis Detection

**arXiv ID:** 2606.25297 | [PDF](https://arxiv.org/pdf/2606.25297v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 241. Bridging the Post-discharge Gap: A Traceable Multi-agent Framework for Safe and Continuous Care

**arXiv ID:** 2606.25334 | [PDF](https://arxiv.org/pdf/2606.25334v1)

**作者:** Runwei Guan `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实现了一种多代理框架 Healink，用于住院后随访，生成基于处方信息、可追溯的回答，提升连续护理质量。

**💡 创新点**

创新点包括：① 物理层级的处方锚定反幻觉机制，实现药物安全保证；② 结构化患者记忆与跨部门共享，提升跨会诊一致性；③ 分层检索与硬约束匹配，精准匹配历史与同类病例；④ 生成白盒证据链，保证每条医学声明可追溯。

**🔧 技术方法**

采用多代理架构（Router、Dialog、Memory、RAG），关系数据库与 Redis 缓存存储患者资料，向量检索与硬约束过滤，LangGraph 状态图管理流程，结合多种基础 LLM（如 Qwen、DeepSeek、GPT‑5 等）并通过结构化输出保证安全。

**📊 数据集**

使用真实随访数据集 400 条问答（含 85 条高复杂度盲评子集）以及公开的 webMedQA 63,284 条在线医学问答，覆盖 6 大专科。

**📈 对比分析**

评估方法：自动化 LLM（GPT‑4‑turbo）评估权威性与信息完整度；医生盲评（16 位临床专家）评分准确性与完整性。结果显示，Healink 在信息完整度上比单模 LLM 提升 17.5%–31.9%，在医生盲评中获得最高综合分，整体性能优于同行医生。

**⚠️ 局限性**

局限性：仅在中文医疗环境验证；缺少视觉模态支持，难以处理视觉依赖科室；未直接测量临床结局（如再入院率）；依赖结构化 EHR 数据，若数据缺失或不规范会影响效果。

---

## 242. Delta-Position Estimation-Based IMU Odometry: A Comparison of MLP and Kolmogorov-Arnold Networks

**arXiv ID:** 2606.25454 | [PDF](https://arxiv.org/pdf/2606.25454v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 243. V-Zero: Answer-Label-Free On-Policy Distillation with Contrastive Evidence Gating for Fine-Grained Visual Reasoning

**arXiv ID:** 2606.25319 | [PDF](https://arxiv.org/pdf/2606.25319v1)

**作者:** Haoxiang Sun `[一作]` (Sichuan University), Tao Wang `[通讯]` (Sichuan University)

**通讯引用:** 38853 | [OpenAlex ID](https://openalex.org/A5100653142)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种不依赖答案标签的视觉推理训练框架V-Zero，利用正负视觉证据对比来对学生生成的轨迹进行门控；

**💡 创新点**

核心创新在于将On‑Policy Distillation视为无负样本的停止梯度对齐，弥补其缺乏轨迹级判别的不足，并通过正负视觉视图构造对比门控；

**🔧 技术方法**

技术手段包括On‑Policy Distillation（OPD）、对齐视图的停止梯度对齐、正负视觉证据的对比门控以及正向反向KL损失的采样估计；

**📊 数据集**

使用Zooming without Zooming（ZwZ）公开的23K高质量训练样本，其中包含完整图像、问题以及相关区域裁剪，负样本通过2×下采样后随机裁剪得到；

**📈 对比分析**

在多项细粒度视觉推理基准（如VStar、HR‑4K、HR‑8K、ZoomBench和MMStar）上，V‑Zero相较于Qwen3.5‑4B基座提升平均5.1分，显著优于传统RL、SFT和可视化推理模型，且训练速度比SFT快5×、比RL快10×；

**⚠️ 局限性**

局限性包括对高分辨率全图输入的依赖，正负视图对比在信息丰富场景下效果不明显，且对模型规模与教师质量仍保持敏感。

---

## 244. REViT: Roto-reflection Equivariant Convolutional Vision Transformer

**arXiv ID:** 2606.25318 | [PDF](https://arxiv.org/pdf/2606.25318v1)

**作者:** Sheir A. Zaheer `[一作]` (KC Machine Learning Lab), Chan Y. Park `[通讯]` (KC Machine Learning Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于离散旋转反射对称群的视觉Transformer——REViT，利用卷积自注意力实现无位置编码的群等变性。

**💡 创新点**

创新点在于将卷积投影和卷积自注意力整合进Transformer块，构造了G-CSA，既保持了等变性，又避免了相对位置编码的复杂性。

**🔧 技术方法**

采用卷积投影、群卷积自注意力（G-CSA）、提升层（lifting）以及离散群E(2,N)的离散旋转反射群。

**📊 数据集**

在Rotated MNIST、PatchCamelyon、CIFAR-10和ImageNet-1K等公开数据集上进行实验。

**📈 对比分析**

与传统G-SA+RPE和G-CNN相比，REViT在准确率上更优或相当，参数量更少，计算量和显存占用显著降低。

**⚠️ 局限性**

缺点是模型复杂度随群阶数增加而线性增长，导致推理延迟和资源占用提高，限制了在低功耗或实时场景中的部署。

---

## 245. Brevity is the Soul of Inference Efficiency: Inducing Concision in VLMs via Data Curation

**arXiv ID:** 2606.25432 | [PDF](https://arxiv.org/pdf/2606.25432v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 246. Communicability-Inspired Positional Encoding (CIPE)

**arXiv ID:** 2606.25293 | [PDF](https://arxiv.org/pdf/2606.25293v1)

**作者:** Yipeng Zhang `[一作]` (Nanyang Technological University), Kelin Xia `[通讯]` (Nanyang Technological University)

**通讯引用:** 3078 | [OpenAlex ID](https://openalex.org/A5084610901)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于图扩散可通信度的位置信息编码（CIPE），让Transformer自注意力直接反映全局多路径结构。

**💡 创新点**

将可通信度映射到内积几何，构建可直接用于自注意力的全局结构几何，并通过尺寸对齐保持几何不失真；在结构无偏和有偏Transformer上均取得显著提升。

**🔧 技术方法**

利用图拉普拉斯扩散、热核、可通信度、矩阵正交化、奇异值分解和Chebyshev多项式近似等技术，实现CIPE的构造与高效近似。

**📊 数据集**

在14个基准上验证，包括TUDataset四个图分类、MoleculeNet七个分类、QM7/8/9三项量化回归。

**📈 对比分析**

与LapPE、RWSE、CycleSE、HKSE等现有PE以及多种GNN/Transformer基线对比，CIPE在结构无偏VTr平均提升约49%，在结构有偏MPTr平均提升约15%（ROC‑AUC），并与顶级基线竞争。

**⚠️ 局限性**

对图尺寸的尺寸对齐会导致信息压缩，需手工选择扩散时间和维度；在极大图上计算仍受限；在已高度捕获结构的任务中提升有限。

---

## 247. Teach-to-Reason: Competition-Guided Reasoning with a Self-Improving Teacher

**arXiv ID:** 2606.25407 | [PDF](https://arxiv.org/pdf/2606.25407v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 248. Reclaim Evaluation: A Lossy Memory Is Worse Than an Empty One

**arXiv ID:** 2606.25449 | [PDF](https://arxiv.org/pdf/2606.25449v1)

**作者:** Alex Kwon `[一作]` `[通讯]`, Alex Kwon

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究语言模型跨会话压缩记忆导致的“脆性记忆”问题，并提出可验证的“回收评估”框架。

**💡 创新点**

创新点在于发现仅保留错误结论而丢失证据会让模型更易产生错误，并提出以保持可重算源为核心的“source‑first”压缩策略。

**🔧 技术方法**

使用了定向纠正、记忆压缩策略、匹配预算对照实验以及基于精确答案的自动回收率测度。

**📊 数据集**

实验数据集包括八道算术题、八道约束逻辑题和 MultiWOZ 对话槽位恢复任务。

**📈 对比分析**

通过与三种 lossy 策略对比，source‑first 在低完整度下恢复率从 0% 提升至 99%–100%，并在多模型、真实对话场景中保持一致。

**⚠️ 局限性**

局限性包括只能适用于可识别的紧凑源，写入时需能定位源且对噪声、预算过小的情况仍会出现无声失败，且对更大规模或非确定性任务的泛化尚未验证。

---

## 249. TabClean: Reusable LLM-Synthesized Programs for Tabular Data Cleaning

**arXiv ID:** 2606.25388 | [PDF](https://arxiv.org/pdf/2606.25388v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 250. PolicyAlign: Direct Policy-Based Safety Alignment for Large Language Models

**arXiv ID:** 2606.25442 | [PDF](https://arxiv.org/pdf/2606.25442v1)

**作者:** Chang Wu `[一作]` (Qwen Large Model Application Team, Alibaba), Xiang Wang `[通讯]` (Qwen Large Model Application Team, Alibaba)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种直接将大型语言模型与安全政策对齐的框架——PolicyAlign；

**💡 创新点**

创新点在于：①用安全政策本身驱动教师模型生成针对政策违规的训练样例；②引入政策敏感过滤（PSF）按行为差距挑选高价值样例；③通过自我对抗式的逆 KL 逐步蒸馏，将教师的政策一致性行为内化到学生模型；

**🔧 技术方法**

核心技术包括：大模型（如GPT-5.4）政策条件文本生成、教师-学生自蒸馏、逆 KL 损失、基于行为差距的过滤；

**📊 数据集**

使用的主要数据集有：StrongREJECT、AdvBench、WildJailbreak、Fortress、XSTest（安全评测）以及 MMLU-Pro、GPQA-Diamond、MATH500（通用能力评测）；在医学、法律、金融领域还使用了 MedSafetyBench、CARES、GUARDSET-X、TRIDENT、LEXam、FinanceBench 等；

**📈 对比分析**

与多种基线（ICL、SFT、AlphaAlign、NSPO、GRPO+Policy）比较，PolicyAlign 在安全评测中显著降低攻击成功率、提升过度拒绝率，且在通用能力评测中保持或略有提升，整体实现了更优的安全‑实用性权衡；

**⚠️ 局限性**

局限性包括：仅在单轮文本场景评测；对多轮对话或多模态的适用性未知；假设安全政策能以清晰文本提供，面对冗长或歧义政策需要进一步拆解与推理；

---

## 251. EmuGEMM: Fused Tensor Core Kernels for Precision Emulation in Matrix Multiplication

**arXiv ID:** 2606.25453 | [PDF](https://arxiv.org/pdf/2606.25453v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 252. DFMU: Data-Frugal Machine Unlearning

**arXiv ID:** 2606.25410 | [PDF](https://arxiv.org/pdf/2606.25410v1)

**作者:** Sajith U `[一作]` (Samsung R&D Institute), Prateek Keserwani `[通讯]` (Samsung R&D Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无需完整重训练的机器去学习方法——Data‑Frugal Machine Unlearning (DFMU)，能够在保留模型主要性能的同时删除指定类或概念。

**💡 创新点**

创新点在于：①利用零成本代理（zero‑cost proxies）与知识保留剪枝（knowledge preserving pruning）计算重要性得分；②仅通过一次前向+反向传播就能得到重要性并对权重进行缩放，从而显著减少数据和计算量；③通过对比实验展示了40%更高的保留准确率与88%更快的处理速度。

**🔧 技术方法**

核心技术包括：单次前向+反向传播计算KL散度；知识保留剪枝求得层级重要性得分；基于重要性得分的权重缩放与掩码操作；无需再训练。

**📊 数据集**

在CIFAR‑100和CIFAR‑20（即CIFAR‑20子集）图像分类数据集上进行实验，构造不同类别的遗忘集。

**📈 对比分析**

与现有SSD方法对比，DFMU在保留集准确率上提升约40%（仅使用13%数据样本），在每类遗忘处理时间上提升88%，且遗忘集准确率基本保持不变或略有提升。

**⚠️ 局限性**

局限性包括：实验仅在ViT模型和小规模CIFAR数据集上验证；对大规模模型或多任务/回归任务的适用性未评估；若重要性得分计算失真，可能导致保留性能下降。

---

## 253. Generative AI for Safe and Photorealistic Drone Light Shows

**arXiv ID:** 2606.25458 | [PDF](https://arxiv.org/pdf/2606.25458v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 254. Towards an HRS Category in TermCOMP

**arXiv ID:** 2606.25448 | [PDF](https://arxiv.org/pdf/2606.25448v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 255. C3-Bench: A Context-Aware Change Captioning Benchmark

**arXiv ID:** 2606.25445 | [PDF](https://arxiv.org/pdf/2606.25445v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 256. LinStereo: Linear-Complexity Global Attention for Multi-Scale Iterative Stereo Matching

**arXiv ID:** 2606.25437 | [PDF](https://arxiv.org/pdf/2606.25437v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 257. HEART: Coordination of Heterogeneous Expert Agents for Physically Grounded Robotic Task Planning

**arXiv ID:** 2606.25404 | [PDF](https://arxiv.org/pdf/2606.25404v1)

**作者:** Junho Lee `[一作]` (Sogang University), Changjoo Nam `[通讯]` (Sogang University)

**通讯引用:** 743 | [OpenAlex ID](https://openalex.org/A5084244126)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种名为HEART的多LLM框架，能够将自然语言指令拆解为原子推理任务，并在有限令牌预算下将任务分配给专门的专家代理，最终合成可执行的机器人操作计划。

**💡 创新点**

创新点包括：①引入多专家角色（能力、环境、路径、可行性、约束）实现任务的细粒度专业化；②在分配阶段结合语义相似度、历史惩罚和令牌容量规划，智能地在令牌预算内分配任务；③通过专家协同与计划合成，显著提升计划的逻辑有效性和物理可执行性。

**🔧 技术方法**

使用的技术包括：大型语言模型（GPT‑4、GPT‑4o）配合链式思考；Sentence‑BERT用于生成任务与角色的语义相似度；tiktoken进行令牌估计；LangGraph实现多轮对话与任务循环；以及与两种计划器（LLM‑CoT、DELTA）结合的计划合成。

**📊 数据集**

采用三组家居操作场景（Beechwood、Benevolence、Merom）来自Gibson 3D场景图数据集，包含 3–5 房间、60–100 对象及其属性，用于评估多种推理难度。

**📈 对比分析**

与单一LLM、规则型分配器和无惩罚/无容量规划的HEART变体比较，HEART在子任务成功率约 99% 以上、计划成功率 72–86% 之间，且所需的分配轮数和令牌消耗均低于基线；与LLM‑CoT/DELTA 单独使用相比，HEART 能提升计划成功率至 60–86% 以上，且仅略微增加规划器令牌消耗。

**⚠️ 局限性**

局限性包括：总体令牌使用量仍显高，尤其是多轮协同导致总成本上升；对单一语言模型的依赖限制了专业化深度；在极低令牌预算下仍难以保证全部子任务完成；未针对大规模多机器人协调进行充分验证。

---

## 258. CV-Rules: Serializability Verification of Concurrency Control Protocols via Explicit Transaction Ordering

**arXiv ID:** 2606.25409 | [PDF](https://arxiv.org/pdf/2606.25409v1)

**作者:** Takashi Hoshino `[一作]` (Cybozu Labs, Inc.), Sho Nakazono `[通讯]` (LY Corporation)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了通过两条局部读条件（Causality与View Consistency）来判定事务执行是否可序列化，并证明其与传统的MVSG无环性等价，进一步证明在读写宽度固定时可多项式时间判定；并利用该框架验证了两阶段锁定、MVTO、SSN、Aria与SnapChain五种协议，揭示了Aria的唯一写约束可删减、SnapChain可直接按CV规则设计；

**💡 创新点**

创新点在于将序列化判定转化为局部读条件，避免全局图无环性检查，提供显式事务顺序构造；引入宽度限制下的多项式判定；为SSN与Aria等无显式顺序协议提供显式顺序构造；

**🔧 技术方法**

使用Lean形式化与自动化证明，构建CV规则与MVSG等价证明，利用Szpilrajn扩展定理、Dilworth定理及状态空间搜索决策程序；

**📊 数据集**

无实验数据集，全部为理论证明与形式化验证；

**📈 对比分析**

未进行实验性能对比，主要通过理论等价性与形式化验证证明正确性；

**⚠️ 局限性**

局限在于仅处理版本函数模型，未覆盖事务的中止处理、工作负载建模与性能评估；

---

## 259. Project-wise Comparison of Software Birthmarks Using Weighted Partial Similarity

**arXiv ID:** 2606.25418 | [PDF](https://arxiv.org/pdf/2606.25418v1)

**作者:** Nikolay Fedorov `[一作]` (Okayama University), Masateru Tsunoda `[通讯]` (Kindai University)

**通讯引用:** 387 | [OpenAlex ID](https://openalex.org/A5015277478)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对软件出生标记的项目级相似度比较，提出基于对称聚合、模块大小加权和部分相似度的框架，并在此框架下构建了实验流程。

**💡 创新点**

创新点在于：①提出对称聚合以消除不对称性；②引入基于模块k‑gram数量的对数加权，降低小模块噪声；③使用部分相似度聚合仅关注最高百分比的模块对，提升部分重用的检测效果。

**🔧 技术方法**

主要技术包括：静态k‑gram出生标记抽取、余弦、Jaccard、Dice、Simpson以及编辑距离等模块级相似度函数；对称聚合、加权相似度和部分相似度的组合；基于阈值的精确率/召回率评价及其调和平均数。

**📊 数据集**

使用35个开源Java项目（共10类）组成的数据集，其中每个项目有4个以上版本，版本间视为重用对；对外部库和逻辑行数≤30的模块进行过滤，以提升实验可信度。

**📈 对比分析**

与Lee等人、聚合相似度、无权重对称聚合、随机相似度等基线方法进行比较。实验结果显示，加入加权+部分相似度后，调和平均数最高可达0.95，且方差最小，显著优于现有方法。

**⚠️ 局限性**

主要限制包括：仅针对Java和静态k‑gram出生标记；缺乏真实剽窃案例和对抗性变形（如混淆）评估；阈值优化基于类别级别，未进行独立训练/测试；结果的可迁移性到其他语言或出生标记类型尚未验证。

---

## 260. S2-CAR: Segmentation-Supervised Complexity-Adaptive Recommendation

**arXiv ID:** 2606.25415 | [PDF](https://arxiv.org/pdf/2606.25415v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 261. Above the Inner Loop: Exceeding Accelerate at LLM Prefill GEMM on the M1 AMX

**arXiv ID:** 2606.25426 | [PDF](https://arxiv.org/pdf/2606.25426v1)

**作者:** Deyvik Bhan `[一作]` `[通讯]` (Georgia Institute of Technology), Deyvik Bhan (Georgia Institute of Technology)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在M1 Apple Silicon上实现手写的AMX矩阵乘法（GEMM）内核，并针对LLM预填充阶段的单精度计算进行性能评估与比较。

**💡 创新点**

发现AMX内循环受载入发射限制，速度提升来自两条部署层杠杆——多线程细粒度分块填满第二AMX块以及一次性预打包常量权重，并通过完整的负结果集验证其结构性优势。

**🔧 技术方法**

采用Goto–BLIS三层缓存分块、显式B面板预打包、K分块与Z累加载入、AMX指令（AMX_LDX/LY/FMA32/STZ）、Grand Central Dispatch多线程调度以及CPU占用探测等技术实现。

**📊 数据集**

使用12种LLM预填充GEMM形状，覆盖GPT‑2‑style（H=2048，V=60000）、TinyLlama‑1.1B（H=2048，V=32000）和Llama‑7B（H=4096，V=32000）模型，批次S=128。

**📈 对比分析**

通过单线程和多线程吞吐量对比cblas_sgemm、BNNSMatMul、BNNS Graph等Accelerate路径，结果显示自研内核在所有12种形状上平均提升1.58×、最快可达2.0×，端到端预填充吞吐提升1.44×。

**⚠️ 局限性**

仅针对fp32，需一次预打包并主要在预填充阶段显效；对fp16或量化路径未做优化；实验仅在单个M1设备上完成，其他Apple Silicon多核结构可能导致结果不完全可迁移。

---

## 262. Center-Fed Pinching Antenna System for Uplink Environment Sensing

**arXiv ID:** 2606.25423 | [PDF](https://arxiv.org/pdf/2606.25423v1)

**作者:** Cong Yu `[一作]` (Beijing Jiaotong University), Wei Chen `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 34300 | [OpenAlex ID](https://openalex.org/A5100344522)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出基于中心馈送的Pinching Antenna系统（C-PASS）用于上行环境感知，构建线性逆模型重构二进制体素场景，并推导测量矩阵条件数与Ziv–Zakai下限；

**💡 创新点**

通过在传统单端馈送PASS中引入对称双馈点，显著提升感知自由度，获得更优的矩阵条件数，并证明C-PASS在ZZB上优于传统PASS；

**🔧 技术方法**

采用中心馈送结构、多路RF链、波导分离、线性逆建模、Ziv–Zakai下限分析及数值仿真与Monte Carlo验证；

**📊 数据集**

使用仿真生成的室内场景（5×10×3 m³，1200个0.5 m³体素，稀疏率0.5%，6个UE，28 GHz信号）作为实验数据；

**📈 对比分析**

通过比较测量矩阵条件数、ZZB曲线和最大似然误差，结果显示C-PASS在所有用户/时隙设置下均保持更低的ZZB和更好的矩阵条件；

**⚠️ 局限性**

仅在仿真环境下验证，未考虑多波导部署、硬件非理想效应及实际测量噪声，缺乏真实硬件实现与实测验证。

---

## 263. Beyond Next-Observation Prediction: Agent-Authored World Modeling for Sequential Decision Making

**arXiv ID:** 2606.25421 | [PDF](https://arxiv.org/pdf/2606.25421v1)

**作者:** Guangfeng Cai `[一作]` (Southeast University), Lei Feng `[通讯]` (Southeast University)

**通讯引用:** 359549 | [OpenAlex ID](https://openalex.org/A5111964102)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Agent-Authored World Modeling（AAWM），让 LLM 代理根据自身决策需求生成训练目标，而非仅预测下一观测。

**💡 创新点**

创新点在于自我探测 Self-Probing 与跨轨迹检索，构造以决策为导向的世界模型目标，提升决策质量。

**🔧 技术方法**

使用 LLM Prompting、检索技术（最大边际相关性）、外部合成模型（如 Qwen3）以及 GRPO 强化学习。

**📊 数据集**

在 ALFWorld、WebShop 两大文本环境以及 AgentGym 的四个环境中使用 AgentTraj-L 轨迹数据。

**📈 对比分析**

与传统的下一个观测预测（IWM）对比，AAWM 在两种模型规模下在任务成功率上提升 6–10 点，且在更难任务上表现更显著。

**⚠️ 局限性**

局限在于仅一次自我探测，未随训练动态更新；仅针对文本环境，未扩展至多模态；合成目标可能带来偏差。

---

## 264. TopoCast: A Topological Fidelity Framework for Evaluating Transformer-Based Time Series Forecasting

**arXiv ID:** 2606.25439 | [PDF](https://arxiv.org/pdf/2606.25439v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 265. PRISM: Feed-Forward Single-Image 3D Reconstruction via Geometric Warp-Residual Modeling

**arXiv ID:** 2606.25430 | [PDF](https://arxiv.org/pdf/2606.25430v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 266. Large-Scale Tunnel Air--Ground Collaboration With FLISP: Fast LiDAR-IMU Synchronized Path Planne

**arXiv ID:** 2606.25393 | [PDF](https://arxiv.org/pdf/2606.25393v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 267. Offline Multi-agent Continual Cooperation via Skill Partition and Reuse

**arXiv ID:** 2606.25389 | [PDF](https://arxiv.org/pdf/2606.25389v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 268. Towards Robust EEG Decoding Based on Riemannian Self-Attention

**arXiv ID:** 2606.25456 | [PDF](https://arxiv.org/pdf/2606.25456v1)

**作者:** Shaocheng Jin `[一作]` (Jiangnan University), Josef Kittler `[通讯]` (University of Surrey)

**通讯引用:** 52075 | [OpenAlex ID](https://openalex.org/A5028209738)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于Bures‑Wasserstein几何的SPD自注意力网络（GBWAtt）用于脑机接口中的EEG解码。

**💡 创新点**

创新点在于引入可学习的θ‑GBWM度量以及在SPD流形上实现的自注意力机制，显著提升了对低SNR EEG信号的鲁棒性和判别能力。

**🔧 技术方法**

采用了Riemannian流形学习、Bures‑Wasserstein度量、SPD自注意力、Fréchet平均聚合、ReEig与LogEig映射以及深度卷积特征提取。

**📊 数据集**

实验使用了三大EEG基准数据集：BCIC‑IV‑2a（运动想象）、MAMEM‑SSVEP‑II（稳态视觉诱发电位）和BCI‑ERN（错误相关电位）。

**📈 对比分析**

与多种欧氏深度学习模型（EEGNet、ShallowConvNet、SCCNet等）及Riemannian网络（SPDNet、MAtt、GDLNet）进行对比，GBWAtt在MI、SSVEP和ERN任务中分别实现了约0.2%、2.1%和5.2%的准确率提升，并在大规模临床EEG数据集（TUAB、TUEV）中同样取得优势。

**⚠️ 局限性**

主要局限在于Bures‑Wasserstein几何下重复矩阵运算导致额外计算开销；在极高噪声环境下相对优势减弱，且对极端失配的SPD矩阵仍存在数值不稳定性。

---

## 269. Learning with a Single Rollout via Monte Carlo Pass@k Critic

**arXiv ID:** 2606.25451 | [PDF](https://arxiv.org/pdf/2606.25451v1)

**作者:** Fengdi Che `[一作]` (University Of Alberta), Dale Schuurmans `[通讯]` (University Of Alberta)

**通讯引用:** 14930 | [OpenAlex ID](https://openalex.org/A5010575626)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并实现了单样本（单rollout）PPO方法，使用前缀级别的 Monte Carlo 信用批评家预测 Pass@k 成功概率，从而在单个生成轨迹中生成密集的 token‑level 训练信号。

**💡 创新点**

创新点在于：① 将 Pass@k（尤其是 Pass@4）作为信用目标，提供更具选择性、稀疏的 token‑level 信号；② 用单rollout 训练的信用批评家替代传统多样本群组方法；③ 通过 Pass@k 视角与图论可达性解释，设计了终端校正的优势估计。

**🔧 技术方法**

使用技术包括：单样本 proximal policy optimization（SR‑PPO）、Monte Carlo 终端标签、Prefix‑level Pass@k 信用批评家、token‑level advantage estimator、PPO 损失与 clipping、图遍历实现可达性解释、与 GRPO、GAE‑PPO 的对比实验。

**📊 数据集**

实验数据集为 DeepScaleR 数学推理数据集，评测指标在 AIME24、AIME25 与 HMMT26 三个竞赛级数学基准上进行。

**📈 对比分析**

与 GRPO（每题 8 个 rollouts）和 GAE‑PPO（传统 TD‑bootstrapped）进行对比；单 rollout SR‑PPO 在 Pass@8 验证上与 GRPO 相当，Pass@k 随采样数线性提升；Pass@4 SR‑PPO 在 token‑level 信用稀疏性和训练稳定性上优于 Pass@1 版本。

**⚠️ 局限性**

局限性包括：实验仅在 Qwen3‑1.7B 单一 seed、有限的超参数设置；未验证在其他领域或更长序列的 agent 环境中的泛化；credit 批评家仅是可达性的近似，缺乏对完整推理图的直接评估。

---

## 270. Beyond Visual Forensics: Auditing Multimodal Robustness for Synthetic Medical Image Detection

**arXiv ID:** 2606.25375 | [PDF](https://arxiv.org/pdf/2606.25375v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 271. Information flow security on persistent memory

**arXiv ID:** 2606.25422 | [PDF](https://arxiv.org/pdf/2606.25422v1)

**作者:** Graeme Smith `[一作]` (University of Queensland), Graeme Smith `[通讯]` (University of Queensland)

**通讯引用:** 13403 | [OpenAlex ID](https://openalex.org/A5035371538)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了一种适用于无结构汇编语言的信息流安全逻辑，并将其与重排序干扰自由（RIF）机制结合，以推理和验证在持久化内存（如 Intel x86）上的信息流安全性；通过对 Linux seqlock 的案例验证了方法的有效性。

**💡 创新点**

创新点在于首次将信息流安全性与持久化内存结合，利用 RIF 机制简化对弱内存模型的推理，并实现了对不同处理器架构的可重用性验证。

**🔧 技术方法**

主要技术包括弱前置条件与 rely/guarantee 逻辑、RIF 检查、Isabelle/HOL 形式化证明以及对 Intel x86 持久化内存模型 Px86_ 的语义等价性证明。

**📊 数据集**

本文以 Linux seqlock 的读写锁实现为案例进行验证，并未使用公开数据集，而是基于抽象汇编程序和模型进行演示。

**📈 对比分析**

通过比较在无持久化、普通弱内存和持久化内存环境下的安全性证明，展示了 RIF 机制在不同架构上的适用性与安全性验证的简化；但并未给出数值性能评估，仅从理论上说明安全性提升。

**⚠️ 局限性**

局限性包括缺乏自动化工具支持、对持久化内存语义的简化假设、未覆盖所有非多拷贝原子处理器以及缺少对真实系统的实验验证。

---

## 272. A Survey of Toxicity Detection and Mitigation Strategies for Multilingual Language Models

**arXiv ID:** 2606.25380 | [PDF](https://arxiv.org/pdf/2606.25380v1)

**作者:** Soham Dan `[一作]` (Scale AI), Thomas Hartvigsen `[通讯]` (University of Virginia)

**通讯引用:** 664 | [OpenAlex ID](https://openalex.org/A5075881948)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统梳理并综述了多语言LLM的毒性检测与消毒技术，提出了威胁模型、任务划分、检测方法与缓解策略的完整分类框架。

**💡 创新点**

创新点在于：①以语言偏移、翻译支点、代码混合、后期微调等多维度构建威胁模型；②将毒性检测与消毒任务分为重写、分类与生成评估三类；③整合数据过滤、监督/偏好调优、解码时引导、表征编辑与多语言防护栏等多种消毒机制，并指出跨语言与文化不匹配的根源。

**🔧 技术方法**

主要技术包括跨语言Transformer、翻译管线、表征空间探测、LLM零/少量提示检测、数据中心过滤、对比微调、RLHF、解码时logit引导、激活编辑及多语言守护框架。

**📊 数据集**

引用并评述了ParaDetox、MultiParaDetox、SynthDetox-M、TextDetox/PAN、APPDIA、Jigsaw、OffensEval、HateCheck、LifeTox、ToxiGen、RealToxicityPrompts、RTP‑LX、PolygloToxicityPrompts、FrenchToxicityPrompts、TET等多语种写作、分类与生成毒性数据集。

**📈 对比分析**

通过对现有文献的汇总比较，展示了各方法在不同语言资源与任务上的性能差距——高资源语言的检测/消毒效果普遍优于低资源或形态丰富语言；偏好调优在跨语言迁移时表现不稳定，解码时引导在保持流畅度方面更具优势。

**⚠️ 局限性**

局限性包括：依赖已有公开数据集且多集中于英语/高资源语言，跨语言一致性与文化适配度不足；评测标准碎片化，缺乏统一多语言基准；对低资源语言、方言与代码混合文本的覆盖度低；实验主要基于已有结果，缺乏统一复现与量化对比。

---

## 273. Story Operators: Decomposing the Original $\to$ Sequel Transformation in Embedding Space

**arXiv ID:** 2606.25379 | [PDF](https://arxiv.org/pdf/2606.25379v1)

**作者:** W. Frederick Zimmerman `[一作]` `[通讯]` (Nimble Books LLC), W. Frederick Zimmerman (Nimble Books LLC)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了小说与其续集之间的语义变换，将整本书视为句子嵌入空间中的点，并通过内容轴分解方法把原书到续集的转变拆解为若干可解释的语义操作。

**💡 创新点**

创新点在于：① 用两本书自身段落的PCA得到自适应的内容轴，② 贪婪沿这些轴分解位移形成“Story Operators”，③ 基于这些操作构建了续集的三类（公式化、集中、合成）分类法。

**🔧 技术方法**

主要技术包括：Sentence‑BERT / mpnet-base-v2 生成768维段落嵌入；PCA 与贪婪按轴分解；向量角度闭合度（gap closed）评估；意图对齐检验（将作者声明的动作用向量表示并与位移投影比较）。

**📊 数据集**

使用了 Project Gutenberg 的 PG19 预计算段落嵌入索引，挑选了 13 对原著-续集（共 12,830 本书）且每部书均含 80+ 非杂散段落。

**📈 对比分析**

比较方法通过余弦相似度、向量幅度‖d‖、内容闭合度（gceil）、有效步数、主轴占比、参与度等指标量化 13 对续集，结果显示续集可分为公式化（幅度小、主轴占比高）、集中（幅度中等、主轴占比高）和合成（幅度中等、参与度大）三类，并为每类给出了典型实例。

**⚠️ 局限性**

局限性包括：样本仅 13 对、英语且预1919年；均值池化忽略剧情顺序；PCA 轴仅反映两本书自身方差，未对全局叙事轴做监督；意图对齐受表达方式限制，得到的 22.8% 仅是下界；角度闭合与欧氏距离不完全等价，且未覆盖跨语言或现代作品。

---

## 274. What Actually Works for Spacecraft Fault-Tolerant Control: An Honest Settled-Gate Benchmark of Learned and Classical Methods

**arXiv ID:** 2606.25374 | [PDF](https://arxiv.org/pdf/2606.25374v1)

**作者:** Alireza Shojaei `[一作]` (Virginia Tech), Alireza Shojaei `[通讯]` (Virginia Tech)

**通讯引用:** 1439 | [OpenAlex ID](https://openalex.org/A5013095506)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了一个诚实、可复现的航天器容错控制基准，利用6-DOF Basilisk模拟器、严格的驻留门、结构化排除的故障分类以及置信区间评估；通过该基准比较经典自适应、无学习、端到端学习以及结构化快速运动适应（RMA）控制方案，发现结构化估计-再控制设计在连续效能故障上几乎达到理论极限，端到端学习失败，经典自适应仅部分有效；针对常数加性偏差引起的全零墙，设计并验证了自校正的干扰观测器，使该类故障恢复率从0%提升至59.4%。

**💡 创新点**

创新点包括：1）基准的三大承诺（驻留门、结构化排除的测试集、Wilson置信区间）在容错控制领域首次系统化引入；2）对传统自适应法与学习法的结构化比较，证明结构化估计-再控制胜过单纯学习容量；3）设计了与学习的效能估计耦合的自校正干扰观测器，突破了加性偏差的全零墙。

**🔧 技术方法**

使用的技术主要有：快速运动适应（RMA）框架、强化学习（GRU+PPO）、经典自适应律（包括Nussbaum增益、ICL）、干扰观测器（DOB）以及基于Murray‑Rodrigues参数的6-DOF Basilisk仿真平台。

**📊 数据集**

使用的“数据集”是基于Basilisk的仿真环境，故障模型为效能g、偏置b及惯量因子f，训练集和测试集在这些参数上结构化排除，测试覆盖g∈{±1}（sign）、连续g（gain）、g+b（gain+bias）和失效轴（loss）。

**📈 对比分析**

比较方法：每个控制器在同一测试集、同一驻留门和相同的噪声种子上执行500次仿真，记录每类故障的成功率，并给出Wilson 95%置信区间。结果显示：结构化RMA在sign故障上97.8%/94.4%（gain），仅在gain+bias故障上0%；干扰观测器耦合后成功率提升至59.4%。经典自适应在sign故障上100%但在gain故障上仅55.2%；端到端RL全零。

**⚠️ 局限性**

局限性：1）实验仅在单一Basilisk环境下验证，缺乏跨平台泛化评估；2）故障模型为外部扭矩效能与偏置，未考虑完整的反应轮动力学；3）干扰观测器仅在已知效能估计的前提下自校正，对极端估计误差仍有限；4）传感器偏差需要硬件冗余，无法通过控制器单独解决。

---

## 275. Kom8ndor: An IEEE 802.11bn-Oriented Simulator for Wi-Fi 8 and Beyond

**arXiv ID:** 2606.25435 | [PDF](https://arxiv.org/pdf/2606.25435v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 276. Probing in the Wild: A Case Study of Self-Supervised Speech Representations on Mandarin Sub-dialects with Unsupervised Articulatory Analysis

**arXiv ID:** 2606.25459 | [PDF](https://arxiv.org/pdf/2606.25459v1)

**作者:** Shu Shang `[一作]` (Fudan University), Yaqian Zhou `[通讯]` (Fudan University)

**通讯引用:** 714 | [OpenAlex ID](https://openalex.org/A5100559702)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过无监督的语言无关声学标注，探究了 Mandarin 子方言下 wav2vec2.0 自监督模型对发音特征的编码情况。

**💡 创新点**

创新点在于将 Allosaurus + PanPhon 的伪标注流水线用于大量未标注的方言语料，实现了对细粒度方言变化的无监督发音特征探测。

**🔧 技术方法**

采用的技术包括 Allosaurus 通用语音识别、PanPhon 语音特征映射、wav2vec2.0 预训练编码器以及线性 Probing + Macro‑F1 评估。

**📊 数据集**

实验使用 KeSpeech 语料库的八种 Mandarin 子方言，约 3 小时音频共 80,000 帧。

**📈 对比分析**

与传统手工标注的探测方式对比，本文的无监督管线在方言之间展现出明显的差异，发现声学显著特征（如 labial、strident）跨方言稳定，而细粒度特征（如 nasal、back）在北京方言表现异常高，说明模型对细微音色差异的方言敏感性不均。

**⚠️ 局限性**

局限性包括仅使用线性探测、只评估一个 SSL 模型、以及 Allosaurus 的错误率可能导致的方言偏差，未来需扩展至多模型、多语言并加入人工标注验证。

---

## 277. OAMP-Aided Joint Channel Estimation and Data Detection for ODDM Systems

**arXiv ID:** 2606.25420 | [PDF](https://arxiv.org/pdf/2606.25420v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 278. The Generalization Spectrum: A Chromatographic Approach to Evaluating Learning Algorithms

**arXiv ID:** 2606.25450 | [PDF](https://arxiv.org/pdf/2606.25450v1)

**作者:** Jinghan Zhang `[一作]` (ByteDance Seed), Tianle Cai `[通讯]` (ByteDance Seed)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Generalization Spectrum评估框架，通过构造从Exact Recall到完全无关的四级对齐评估来衡量模型从训练样本到不同转移距离的泛化能力。

**💡 创新点**

创新点在于将泛化视为连续的“转移距离”而非二元分割，使用匹配记忆化（matched‑memorization）比较不同学习范式的转移效率，并通过该框架对竞赛编程任务进行细粒度诊断。

**🔧 技术方法**

主要技术包括基于对齐的评估设计、在大语言模型上实现ICL、SFT、RL（GRPO）等后训练方法、匹配检查点、计算Pass@1、增益、归一化增益、AUS与N‑F_n等指标。

**📊 数据集**

使用自构造的竞赛编程基准，包含64个种子问题及其对应的四个转移级别（共256个评估实例），数据来源于公开编程平台并通过多阶段生成与验证流程得到。

**📈 对比分析**

方法对比采用在相同记忆水平（D0）下的匹配检查点，结果显示RL在近距离转移上优于SFT，ICL在实例级对齐（D0–D2）表现强劲但在类别级别（D3–D4）快速下滑；不同变体（如RFT、SDFT、提示辅助RL）在局部提升或速度上有优势，但普遍未能扩展远距离泛化。

**⚠️ 局限性**

局限性包括仅在竞赛编程域验证，框架对其他推理或生成任务的通用性待证；数据规模相对有限（64/256实例）；评估主要聚焦Pass@1，未涉及更细粒度错误分析或多样性评价。

---

## 279. LibEvoBench: Probing Temporal Knowledge Stratification in Code Generation Models

**arXiv ID:** 2606.25402 | [PDF](https://arxiv.org/pdf/2606.25402v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 280. MAPL: Multi-Objective Preference Learning for Robot Locomotion

**arXiv ID:** 2606.25398 | [PDF](https://arxiv.org/pdf/2606.25398v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 281. From Sounds to Scenes: A Benchmark for Evaluating Context-Aware Auditory Scene Understanding in Large Audio Language Models

**arXiv ID:** 2606.25391 | [PDF](https://arxiv.org/pdf/2606.25391v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 282. Transferable Attack against Face Swapping in an Extended Space

**arXiv ID:** 2606.25376 | [PDF](https://arxiv.org/pdf/2606.25376v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 283. Gastroendoscopy View Synthesis: A New Real Dataset and Evaluation

**arXiv ID:** 2606.25427 | [PDF](https://arxiv.org/pdf/2606.25427v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 284. Lightweight PCGAE-Net: Parallel CrossGate Attention and Bottleneck AutoEncoder for Efficient 5G Channel Prediction

**arXiv ID:** 2606.25401 | [PDF](https://arxiv.org/pdf/2606.25401v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 285. Anatomically-conditioned Latent Diffusion Model for Data-Efficient Few-Shot Cross-Domain 3D Glioma MRI Synthesis

**arXiv ID:** 2606.25390 | [PDF](https://arxiv.org/pdf/2606.25390v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 286. Commerge: Communication-Efficient, Robust, and Fast LiDAR Map Merging Framework for Multi-Robot Coordination in Resource-Constrained Scenarios

**arXiv ID:** 2606.25386 | [PDF](https://arxiv.org/pdf/2606.25386v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 287. The Interplay of Harness Design and Post-Training in LLM Agents

**arXiv ID:** 2606.25447 | [PDF](https://arxiv.org/pdf/2606.25447v1)

**作者:** Kyungmin Kim `[一作]` (POSTECH), Sangdon Park `[通讯]` (POSTECH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在工具集成的LLM代理中，将harness视为可控设计维度，并在零射击、分布内和两种OOD场景（工具环境漂移与任务漂移）下进行后训练的影响。

**💡 创新点**

创新点在于：①将harness从固定工程细节提升为可调参数；②构建了可模块化的benchmark，支持多种harness信息量和环境漂移；③系统评估harness对后训练鲁棒性的作用，揭示harness-aware后训练是实现OOD鲁棒性的关键。

**🔧 技术方法**

使用了GRPO和GiGPO两种RL算法对开源LLM（LLaMA-2-7B和LLaMA-2-13B）进行后训练，结合文本规划环境和工具调用机制。

**📊 数据集**

数据集基于改造的text‑based planning环境（Taskmaster‑6），包含3827个任务实例，按难度分为easy、medium、hard三组。

**📈 对比分析**

通过对比不同harness（Low、Medium、Rich）在零射击、分布内、工具环境漂移和任务漂移下的成功率，发现harness信息量越高，性能越好；harness需在训练时设置，否则后置harness几乎不恢复优势；在工具环境漂移和任务漂移下，harness‑aware 后训练显著保持或提升性能。

**⚠️ 局限性**

局限性包括仅在单一环境下验证，缺乏对更复杂工具集和任务分布的评估；实验仅覆盖两种LLM和两种RL算法，未探索更广泛模型；信息丰富的harness需要专家设计或昂贵探索，实际应用成本较高。

---

## 288. BrainAgent: A Large Language Model-Driven Multi-Agent Framework for Autonomous Brain Signal Understanding

**arXiv ID:** 2606.25400 | [PDF](https://arxiv.org/pdf/2606.25400v1)

**作者:** Yangxuan Zhou `[一作]` (Zhejiang University), Gang Pan `[通讯]` (Zhejiang University)

**通讯引用:** 483871 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 BrainAgent，一种基于大型语言模型的多智能体框架，可将自然语言指令转换为端到端的脑信号分析流程。

**💡 创新点**

创新点在于将 LLM 作为层级监督者与专门子智能体协同工作，拆解任务并实现跨域协作，同时构建系统化的 L1–L3 难度层级基准评测。

**🔧 技术方法**

采用大型语言模型（Qwen、GPT‑4o 等）驱动的多智能体架构、工具调用、层级调度与跨域协同。

**📊 数据集**

使用公开的睡眠 EEG 数据集 ISRUC Subgroup‑1 与 HMC。

**📈 对比分析**

在 60 个 L1–L3 任务上与多种 LLM 基准对比，最大模型 Qwen‑Max 在 L3 任务中 TCR 达到 0.90，Heterogeneous（轻量子代理+强监督）平均 TCR 达到 0.84，单一代理在工具扩展时性能显著下降。

**⚠️ 局限性**

局限性包括对大模型的高依赖、算力与延迟开销、对工具库规模敏感以及在极低参数模型下仍难以处理高层次抽象任务。

---

## 289. Long-Term Simulation Exposes Cognitive-Developmental Risks in AI Companions

**arXiv ID:** 2606.25396 | [PDF](https://arxiv.org/pdf/2606.25396v1)

**作者:** Kaicheng Shen `[一作]` (East China Normal University), Yingchun Wang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 12617 | [OpenAlex ID](https://openalex.org/A5100613144)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并应用了 TSJ（Theater‑Stage‑Judge）框架，对大型语言模型驱动的 AI 伴侣在儿童、青少年及早期成年阶段的长期互动中产生的认知与情感风险进行系统评估。

**💡 创新点**

创新点在于：① 引入长时序的用户模拟与心理状态动态更新；② 结合 persona 驱动的场景生成、情感轨迹追踪与回溯式风险定位；③ 采用 CDM（Cognitive Developmental Risk Assessment Matrix）四阶段 × 六核心维度的评估矩阵；④ 通过 Area Under the Longitudinal Curve（AULC）量化风险的延迟暴露；⑤ 统一的“Judge”模块实现跨模型、跨阶段的客观安全评分。

**🔧 技术方法**

使用的技术包括：多模块架构（Theater、Stage、Judge）；LLM 生成日常对话与内在自述；心理变量模板与漂移扩散决策引擎；GPT‑4.1 作为评判者；AULC 与保留比例等长期安全指标；以及批量控制实现 432 条完整试验。

**📊 数据集**

数据集：构建了 6 大主流 LLM 后端、4 个发展阶段、24 个风险维度、3 个心理脆弱性人设的模拟实验，共 12,960 个 simulated person‑day 交互。没有使用真实用户数据，而是基于场景模板、记忆池与心理状态递推生成交互日志。

**📈 对比分析**

对比方法：在同一实验框架下评估 6 个模型（GPT‑5、Qwen3‑235B‑A22B、MiniMax‑M2.5、DeepSeek‑V3、Gemini‑3.1‑pro、GPT‑4o）在不同阶段与人设下的 24 维安全得分与 AULC。结果显示：短时测试显著高估安全；长时评估揭示多模型在低风险人设下的安全性不一定稳定；GPT‑5 在四阶段表现最均衡，MiniMax‑M2.5 最易漏测长时风险；AULC 说明青少年阶段风险最易被延迟暴露。

**⚠️ 局限性**

局限性：① 模拟环境无法完全复现真实儿童、青少年与成人的情感与社会互动；② 评判者依赖 GPT‑4.1，可能存在偏见与一致性问题；③ 人设与真实发展轨迹的匹配度有限，可能导致风险评估偏差；④ 只覆盖 24 个维度，某些微妙的关系性风险未被捕获；⑤ 长时序模拟仍简化现实世界多重刺激与环境变化，可能低估复杂情境下的风险。

---

## 290. Envy-free Contracts with Subsidies

**arXiv ID:** 2606.25431 | [PDF](https://arxiv.org/pdf/2606.25431v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 291. Introducing corpora Hlava Cor and Hlava AD: Human Label Variation in Coreference and Discourse Relations

**arXiv ID:** 2606.25383 | [PDF](https://arxiv.org/pdf/2606.25383v1)

**作者:** Anna Nedoluzhko `[一作]`, Eva Hajičová `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究创建了两个多注释语料库——Hlava Cor（核心ference）和 Hlava AD（归因与话语关系），用于研究人类标注差异。

**💡 创新点**

创新点在于将多注释与注释者评论结合，系统记录并分析人类标注的多重解释与不确定性。

**🔧 技术方法**

主要技术包括手工多注释、Transformer‑based 核心ference 预筛选、交叉标注一致性评估（IAA）等。

**📊 数据集**

数据集来源于捷克语 PDT‑C 2.0、PDTSC、CAC 及 iRozhlas，分别构成 1024 条核心ference 案例和 512 条话语关系案例。

**📈 对比分析**

通过与模型一致性和多注释一致性比较，核心ference 的 IAA 最高 66%，话语关系的平均 pairwise IAA 为 64.9%，表明模型一致性可作为复杂性的指示。

**⚠️ 局限性**

局限性包括低 IAA、仅针对捷克语、注释者缺乏理论训练、评论未深入分析以及缺乏对下游任务的验证。

---

## 292. FactorLibrary: From Polynomials to Circuits via Recursive Subgoals

**arXiv ID:** 2606.25394 | [PDF](https://arxiv.org/pdf/2606.25394v1)

**作者:** Rohan Pandey `[一作]` (University of Washington), Jarod Alper `[通讯]` (University of Washington)

**通讯引用:** 904 | [OpenAlex ID](https://openalex.org/A5023719219)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了算术电路最小化问题的两种强化学习表述：bottom‑up 的门级搜索和 top‑down 的递归分解，并引入 FactorLibrary 作为可重用子目标，随后在 𝔽₅ 上的两变量多项式上训练了 Gumbel‑PPO‑MCTS、PPO+MCTS 与 SAC 代理，验证其性能。

**💡 创新点**

创新点在于：1) 设计了 FactorLibrary 子目标机制，使得可因式分解的子表达式可被多次重用；2) 将算术电路问题转化为 top‑down 的递归拆分环境，显著缩小有效动作空间；3) 将 Gumbel 根搜索与 MCTS 结合，提升了底层搜索效率，展示了 top‑down 在高复杂度下优于 bottom‑up 的优势。

**🔧 技术方法**

使用的技术包括：Gumbel‑PPO‑MCTS、PPO+MCTS、SAC、JAX+Flax 实现的 GPU 批量 MCTS、AND/OR MCTS、MuZero‑style 根搜索、子目标奖励与库回放、以及基于图神经网络的状态编码。

**📊 数据集**

采用的数据集为 𝔽₅ 上的两变量多项式，训练集包含 450 个目标，测试集包含 207 个目标，覆盖电路复杂度 C₂–C₁₀ 的范围。

**📈 对比分析**

与均匀随机基线比较：top‑down PPO+MCTS 在 C≤8 的测试集上达到 91.8% 的成功率，SAC 最优点 92.8% 但计算成本仅为 30 分钟；bottom‑up 在 C≤3 时几乎 100% 成功，但 C>5 时性能骤降至 7.8%。

**⚠️ 局限性**

局限性包括：实验仅在少量变量（两变量）和 𝔽₅ 系数、小电路复杂度范围内进行；动作空间随变量数和复杂度指数增长，导致 bottom‑up 搜索迅速失效；尚未验证方法在更高维、多项式阶数或更广泛域上的可扩展性。

---

## 293. The impact of artificial intelligence on enterprise software user roles

**arXiv ID:** 2606.25525 | [PDF](https://arxiv.org/pdf/2606.25525v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 294. Interpretable Concept-Guided Polynomial Tabular Kolmogorov-Arnold Network for EEG-Based Mild Cognitive Impairment Detection

**arXiv ID:** 2606.25434 | [PDF](https://arxiv.org/pdf/2606.25434v1)

**作者:** Yosef Bernardus Wirian `[一作]` (University of Kentucky), Qiang Cheng `[通讯]` (University of Kentucky)

**通讯引用:** 28561 | [OpenAlex ID](https://openalex.org/A5083141785)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对睡眠EEG特征进行概念结构化、交互建模并通过Kolmogorov‑Arnold网络实现非线性分类，实现MCI早期检测。

**💡 创新点**

将生理概念瓶颈与二阶多项式交互展开相结合，并使用Fourier‑parameterized TabKAN，三者协同提升可解释性与分类性能。

**🔧 技术方法**

使用概念编码器、二阶多项式特征扩展和Kolmogorov‑Arnold网络（TabKAN）进行非线性表格学习。

**📊 数据集**

基于美国Study of Osteoporotic Fractures (SOF) 队列，372名老年女性的睡眠EEG及MMSE评分。

**📈 对比分析**

在10折交叉验证下，与10种传统、集成和深度学习基线比较，CPTabKAN‑Second Order加权F1=0.9038，优于最强基线GradientBoosting提升5.65个百分点。

**⚠️ 局限性**

仅使用单一老年女性队列，标签基于MMSE阈值，缺乏外部验证与时间序列信息，可能受群体偏差与标签噪声影响。

---

## 295. Disease-Centric Vision-Language Pretraining with Hybrid Visual Encoding for 3D Computed Tomography

**arXiv ID:** 2606.25546 | [PDF](https://arxiv.org/pdf/2606.25546v1)

**作者:** Bowen Shi `[一作]` (DAMO Academy, Alibaba Group), Jianpeng Zhang `[通讯]` (DAMO Academy, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对3D CT影像构建了一种疾病中心的视觉–语言预训练框架，解决了传统方法在三维医学图像上的效率与语义对齐不足的问题。

**💡 创新点**

创新点包括：①CNN–ViT混合编码器，将3D CNN局部特征与ViT全局注意力融合；②可学习查询实现疾病级对比学习，实现对同一解剖区多病变的细粒度区分；③诊断感知提示策略，用真实临床语料构造疾病原型，缩小预训练与推理的差距。

**🔧 技术方法**

采用3D ResNet-18+多尺度融合Patch Embedding、预训练ViT（SigLIP2）、交叉注意力、疾病级InfoNCE对比损失、诊断原型生成、报告分块解析（LLM）、U-Net解剖分割等技术。

**📊 数据集**

使用CT-RATE（约5万胸CT+报告）、Rad-ChestCT（3.6k胸CT+报告）以及通过Qwen3-Max生成的60疾病伪标签CT-RATE进行实验。

**📈 对比分析**

与fVLM、ViSD-Boost、CT-CLIP等多种基线在CT-RATE内部/外部零样本诊断和60疾病扩展上进行比较，AUC分别提升至84.4%（+5.1%）、75.4%（+5.4%）及85.6%（+9.8%），报告生成指标也超过现有SOTA。

**⚠️ 局限性**

依赖外部工具进行报告解析、伪分割与伪标签，噪声可能影响学习；仅在目前的医学AI基准上验证，缺乏更广泛的下游评估与标准化指标。

---

## 296. STEB: A Speech-to-Speech Translation Expressiveness Benchmark for Evaluating Beyond Translation Fidelity

**arXiv ID:** 2606.25529 | [PDF](https://arxiv.org/pdf/2606.25529v1)

**作者:** Sitong Cheng `[一作]` (Hong Kong University of Science and Technology), Wei Xue `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 25944 | [OpenAlex ID](https://openalex.org/A5100692580)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了用于评估语音到语音翻译(S2ST)中表达式保持的基准STE B，并提供了相应的评估框架。

**💡 创新点**

创新点在于首次同时考量情感、场景风格和非言语发声(NV)三大表达维度，并采用无参考的LLM评判方法实现自动化评分。

**🔧 技术方法**

使用多模态LLM（如Qwen3系列）进行音频字幕、情感与场景摘要；利用BEATs进行NV检测；并通过LLM评分器对比源与译语音的表达一致性。

**📊 数据集**

构建了32.6小时的中英双语S2ST基准数据，涵盖戏剧、广播、广告等六大真实场景，分为常规和NV子集。

**📈 对比分析**

对六种S2ST系统（分层、端到端、语音LLM等）进行评测，发现翻译准确性高但情感保持低（≈1.7/5），NV保持稍好（≈2.3/5），持续时间对齐需专门控制。

**⚠️ 局限性**

局限性包括仅覆盖中英两语对，依赖LLM对情感和场景的感知准确性，且非言语发声的评判仍存在误差。

---

## 297. A Path-Survival Analytical Framework for SCL Decoding of Polar Code

**arXiv ID:** 2606.25522 | [PDF](https://arxiv.org/pdf/2606.25522v1)

**作者:** Xianbin Wang `[一作]` (Huawei Technologies Co., Ltd.), and Wen Tong `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出一种基于路径生存模型的CA‑SCL极化码有限长度性能分析框架，能够在不做列表特定蒙特卡洛仿真的情况下预测块误码率。

**💡 创新点**

创新点在于将解码过程拆分为稀疏但剧烈的“解码危机”序列，通过提取危机期望数r和单个危机生存概率p_L实现对列表大小L的解码性能闭式预测。

**🔧 技术方法**

技术手段包括：对极化码在完美辅助SC解码下的总错误计数拟合负二项分布，利用矩估计得到r和p₁；对危机中信息位错误数建模为几何分布；用近似公式R_true(k)≈2^k评估路径排名，结合p_L=1−(1−p₁)^{⌊log₂L⌋}得到BLER预测。

**📊 数据集**

实验数据集涵盖N=1024、2048、4096、8192、率R=¼、½、¾的极化码，采用Gaussian Approximation或PW序列构造，CRC‑16校验，并在AWGN和BSC信道下进行仿真。

**📈 对比分析**

通过将解析预测与蒙特卡洛仿真结果对比，显示预测曲线与仿真曲线在大多数配置下吻合度高，误差≤0.1 dB，并能准确反映列表增大带来的性能提升。

**⚠️ 局限性**

局限性在于假设危机间相互独立且冻结位恢复机制完美，导致在短码长（如N=1024）或危机密集的低SNR环境下预测略有欠拟合；AWGN/BSC场景下方差需经验估计。

---

## 298. Latency-Aware Service Placement using Neural Combinatorial Optimisers for Edge--Cloud Systems

**arXiv ID:** 2606.25553 | [PDF](https://arxiv.org/pdf/2606.25553v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 299. Fault of Our Stars: Behavioral Drivers of Rating-Sentiment Incongruence

**arXiv ID:** 2606.25518 | [PDF](https://arxiv.org/pdf/2606.25518v1)

**作者:** Ramanaish Abaiyan `[一作]` (University of Moratuwa), Sandareka Wickramanayake `[通讯]` (University of Moratuwa)

**通讯引用:** 202 | [OpenAlex ID](https://openalex.org/A5067544655)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过使用预训练RoBERTa模型对16,156条斯里兰卡旅游景点评论的文本进行情感推断，独立于星级评分，量化并分析了情感与评分之间的不一致性，并将不一致分为六种方向性模式；

**💡 创新点**

创新点在于①首次提出六种结构化的情感-评分不一致模式；②将Transformer情感推断与传统弱标注验证相结合，揭示星级评分作为情感标签的可靠性问题；③利用SHAP解释模型，发现场景类型、评论长度、评论者专业度和时间等因素是导致不一致的关键驱动因素；

**🔧 技术方法**

主要技术包括：预训练RoBERTa情感推断模型（用于文本情感标签）、逻辑回归与随机森林（用于预测不一致性）、SHAP（解释模型特征重要性）、统计检验（卡方、Mann–Whitney）和交叉验证；

**📊 数据集**

数据集为2010-2023年收集的16,156条斯里兰卡旅游景点评论，包含星级评分、文本、评论者信息和时间等多维特征；

**📈 对比分析**

通过比较四个Transformer变体（预训练和微调版本），选取性能最优的预训练RoBERTa作为文本情感标注工具；随后使用逻辑回归和随机森林评估不一致预测，AUC分别约为0.58和0.61，显示模型在解释上有一定效果但仍不够精准；

**⚠️ 局限性**

局限性包括：①仅针对斯里兰卡旅游平台的数据，结果可能缺乏跨域推广性；②文本情感标签仅基于单一Transformer模型，未与传统情感分析方法（如VADER、TextBlob）做对照；③星级评分被粗略分为三类，可能掩盖细粒度信息；④未考虑多语言或多平台情况，且对长文本与短文本的偏差处理有限。

---

## 300. On the Viability of Requirements Generation From Code: An Experience Report

**arXiv ID:** 2606.25550 | [PDF](https://arxiv.org/pdf/2606.25550v1)

**作者:** Alexander Korn `[一作]` (Ruhr Institute for Software Technology), Andreas Vogelsang `[通讯]` (Ruhr Institute for Software Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建多代理架构（生成、验证、突变、追踪）与人机回馈循环，利用LLM+RAG技术从两个公开源码项目中生成需求，并对其实现准确性、幻觉率、需求气味与可检测性进行人工评估。

**💡 创新点**

创新点在于将检索增强生成与多代理协作相结合，生成基于代码的需求并人工标注需求气味，形成可复现的合成需求-代码数据集，并系统评估LLM在实现准确性与气味生成上的表现。

**🔧 技术方法**

使用技术包括 OpenAI GPT‑4（text‑davinci‑003）LLM、检索增强生成（RAG）+ ChromaDB 向量检索、Python 后端、SvelteKit 前端、Celery 任务队列、OpenAI Embeddings、代码分块与分片，以及人工评估与统计指标（Krippendorff α、Cohen κ）。

**📊 数据集**

实验数据集为两份公开源码：学生项目 SEP（约 24k 行 Java/TS/HTML/CSS）和 Mattermost 桌面客户端（约 28k 行 TS/HTML），共生成 188 条需求，构成两份需求‑代码对应数据集。

**📈 对比分析**

通过与人工标注的实现状态、幻觉率、需求气味率以及人与 LLM 在气味检测上的一致率进行对比，结果显示实现准确率 0.83–0.98、幻觉率仅 6.1%，但需求气味率 24.5%，LLM 与人工在气味检测的一致率约 60–90%，表明方法有效但仍需人工监督。

**⚠️ 局限性**

限制主要包括需求气味检测高度主观且需多评审，LLM 难以可靠生成非实现需求，方法对代码质量高度依赖，且未在安全关键或受监管领域验证；此外，单一 LLM 模型和小样本数据集限制了结果的可推广性。

---

## 301. Energy-Efficient CNN Acceleration with MSDF Digit-Serial Arithmetic on FPGA

**arXiv ID:** 2606.25562 | [PDF](https://arxiv.org/pdf/2606.25562v1)

**作者:** Muhammad Usman `[一作]` (University of Regensburg), Dorit Merhof `[通讯]` (University of Regensburg)

**通讯引用:** 11724 | [OpenAlex ID](https://openalex.org/A5064747056)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

设计了一种基于MSDF算术的合并乘加（MMA）单元，在FPGA上实现了对U‑Net卷积层的高效硬件加速器。

**💡 创新点**

创新点在于将MSDF乘法与累加融合为单一流水线，消除单独的初始延迟；MMA单元并行处理32个通道，显著提升吞吐率和能效；同时使用冗余数字系统支持数字逐位串行运算。

**🔧 技术方法**

采用MSDF算术、数字逐位串行乘加、signed digit冗余数字系统、FPGA逻辑实现（Zynq‑7020 SoC）、Verilog RTL、FBGEMM量化等技术。

**📊 数据集**

主要使用医学图像数据集（脑肿瘤MRI图像，U‑Net分割任务）进行实验。

**📈 对比分析**

与CPU、GPU、传统并行、串行、MSDF加速器等方案在吞吐率（GOPS）、能效（GOPS/W）、面积效率（GOPS/切片）、能耗（mJ）等指标上进行对比；FPGA实现以100 MHz运行，能效达15.14 GOPS/W，比CPU 1.93 GOPS/W高8倍；能耗比MSDF版低9倍；吞吐率比传统并行高1.07倍，比串行高4.36倍，比MSDF 2.52倍。

**⚠️ 局限性**

限制在于尚未实现早期终止功能；对更大或更复杂CNN的扩展需要更多tile；对精度或动态范围的适应性有限，需进一步研究。

---

## 302. Efficient Cross-Scale Invertible Hiding Network with Spatial-Frequency Collaboration and Non-Invertible Mechanism

**arXiv ID:** 2606.25547 | [PDF](https://arxiv.org/pdf/2606.25547v1)

**作者:** Junxue Yang `[一作]` (Hunan University), Xin Liao `[通讯]` (Hunan University)

**通讯引用:** 5539 | [OpenAlex ID](https://openalex.org/A5100716299)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种名为 CrosInv 的跨尺度可逆隐藏网络，用于在相同分辨率的封面图像中嵌入整幅秘密图像。

**💡 创新点**

创新点主要包括：①跨尺度可逆模块（CIM）实现输入跨尺度双向映射；②利用像素重排（PS/IPS）和 Haar 小波变换（WT/IWT）在空间-频率域共同提取特征；③在前向与后向过程中插入非可逆交叉稠密模块（NCDM），显著提升非线性表达能力。

**🔧 技术方法**

采用的关键技术有：可逆神经网络的仿射耦合层、像素重排与逆重排、Haar 小波变换与逆变换、稠密连接与混洗注意力、通道注意力以及 U‑Net 风格的跳跃连接，整体损失采用 MSE 进行隐藏与解隐藏两阶段训练。

**📊 数据集**

使用了 COCO（训练/验证/测试各 5000/1000/2000 对）、ImageNet 与 BOSSBase（各 2000 对）作为测试集，所有图像统一裁剪至 128×128。

**📈 对比分析**

与 ISN、HiNet、StegFormer 和 StarINN 四个代表性方法在 PSNR、SSIM、APD、LPIPS 等四项图像质量指标及三种颜色图像隐写检测器（SCRMQ1、UCNet、PENet）的抗检测率进行对比。CrosInv 在隐藏与解隐藏两阶段均取得了极大提升（隐藏 PSNR 超过 60 dB，解隐藏 PSNR 超过 53 dB），同时在抗隐写分析上表现最优，检测准确率逼近 50%。

**⚠️ 局限性**

局限性包括：①模型仍在 128×128 分辨率下验证，尚未证明在更高分辨率或不同域（如视频）上的泛化能力；②训练仅基于 COCO 数据，尽管在 ImageNet/BOSSBase 上表现良好，但对更广泛场景的鲁棒性未知；③相对其它方法，计算开销仍略高（参数/ FLOPs 仍居中位置），在资源受限环境下部署可能受限。

---

## 303. ASSCG: Just-Right Gating over Chattering for Fast-Slow LLM Planning in Autonomous Driving

**arXiv ID:** 2606.25509 | [PDF](https://arxiv.org/pdf/2606.25509v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 304. SAC$^2$-Net: Semantic Anchoring and Complementary-Consensus Fusion for Multimodal Micro-Expression Recognition

**arXiv ID:** 2606.25542 | [PDF](https://arxiv.org/pdf/2606.25542v1)

**作者:** Xuepeng Zheng `[一作]` (Southwest University), Tong Chen `[通讯]` (Southwest University)

**通讯引用:** 7007 | [OpenAlex ID](https://openalex.org/A5100461265)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为SAC^2-Net的多模态微表情识别框架，利用光流与运动放大两种视觉模态的互补性；

**💡 创新点**

创新点包括：①采用语义锚定软对齐（SASA）将AU描述转化为文本提示并生成软标签，显著减少跨模态差异；②设计互补共识融合（CCF），由可靠性感知的互补交换与共识精炼两阶段实现局部补偿与空间一致；

**🔧 技术方法**

使用技术包括：Hybrid Fast Encoder视觉编码器、CLIP文本编码、基于AU软标签的对比学习、可靠性估计网络、交叉模态注意力与共享键机制、动态损失调度；

**📊 数据集**

实验数据集涵盖CASME II、SAMM、SMIC、MEGC2019-CD、CAS(ME)^3、DFME，并在跨数据集转移任务中进行验证；

**📈 对比分析**

与多种前沿方法（如μ-BERT、FRL-DGT、SRMCL、HTNet、MFDAN、EMRNet、MPFNet、Micro_NesT、EDMDBN、MMTNet、LTR3O、MER-CLIP等）在各评估指标（ACC、UF1、UAR、F1等）上进行比较，SAC^2-Net在粗粒度、细粒度、大规模及跨数据集测试中均达或逼近state-of-the-art，并在多项指标上实现显著提升；

**⚠️ 局限性**

局限性包括：①当光流与运动放大两模态同时失效时补偿效果受限；②对AU标注质量敏感，噪声标签会影响语义对齐；③目前仅利用起始-顶点帧，未充分挖掘完整微表情序列的时序信息。

---

## 305. Evaluating LLMs on Real-World Software Performance Optimization

**arXiv ID:** 2606.25530 | [PDF](https://arxiv.org/pdf/2606.25530v1)

**作者:** Ezgi Sarıkayak `[一作]` (Siemens AG), Chunyang Chen `[通讯]` (Technical University of Munich)

**通讯引用:** 4473 | [OpenAlex ID](https://openalex.org/A5075639297)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个仓库级性能优化基准，收集了102个专家优化实例并提供参数化性能测试，旨在系统评估LLM在真实代码库中的性能优化能力。

**💡 创新点**

创新点在于采用多工作负载、多指标（运行时间、峰值内存、twmu）的噪声感知评估框架，克服以往单指标、单输入的局限，提供更全面、可复现的性能评价。

**🔧 技术方法**

技术手段包括LLM补丁生成、oracle/BM25检索、Docker化隔离环境、层级噪声建模、自适应采样、统计显著性检验（RCIW、SNR）等，以确保测量可靠性和结果可比性。

**📊 数据集**

数据集来源于三个Python数据密集型开源库（如Pandas、NumPy等）的性能相关PR，共102个经过人工验证的实例，基准基于专家Gold patch。

**📈 对比分析**

通过Patch Success、Correctness和Performance（IF_LLM）等指标与专家Gold进行对比，实验显示LLM在运行时提升稀少、幅度低，内存提升几乎无效，专家实现平均15.5×速度提升、171×峰值内存下降，凸显LLM与专家之间显著差距。

**⚠️ 局限性**

限制在于仅覆盖Python数据密集型库、仅使用显式标记性能PR，未涵盖低级系统软件、I/O密集或非Python环境，缺乏多语言支持，可能影响结果的普适性。

---

## 306. Beyond One-Size-Fits-All: Diagnosis-Driven Online Reinforcement Learning with Offline Priors

**arXiv ID:** 2606.25527 | [PDF](https://arxiv.org/pdf/2606.25527v1)

**作者:** Guozheng Ma `[一作]` (Nanyang Technical University), Dacheng Tao `[通讯]` (Nanyang Technical University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a4b10f5d-130b-4e77-9367-6469ec621899` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统性探讨了在线强化学习与离线先验结合的范式，提出“有界承诺”概念并证明不同先验的依赖配置无统一最优，主张从基准驱动转向诊断驱动的张力管理；

**💡 创新点**

创新点在于将离线先验视为有界承诺并揭示其导致的探索‑利用、稳定‑可塑性两端张力被放大；通过实验证实无单一最优依赖配置；提出诊断驱动张力管理框架，跨社区统一理解与方法；

**🔧 技术方法**

采用理论框架对离线先验的三类功能（初始化、参考、辅助）进行抽象，设计控制实验（vary μ, λ, β）验证张力加剧，并对比现有RLHF、Cal‑QL、Sim‑to‑Real等方法的依赖配置；

**📊 数据集**

实验主要基于离线到在线RL的标准环境（如OpenAI Gym/DeepMind Control Suite 等），以及示例性跨域任务（LLM后训练、机器人模拟转真实、视觉‑语言‑动作模型微调等）来验证；

**📈 对比分析**

通过在相同任务中切换初始化、保守惩罚与离线数据三种依赖配置，发现同一配置在不同任务上效果相反，说明基准排名无法反映普适性；对比结果表明基准驱动方法往往只能在部分匹配条件下优异；

**⚠️ 局限性**

局限包括：诊断工具尚不完善、实验覆盖的先验类型与环境有限、对动态依赖调节的机制仍需进一步研究、以及诊断精度受在线数据质量和测量指标限制；

---

## 307. AISPO: Enhancing Depth Reliability for Robotic Manipulation of Non-Lambertian Objects via Affine-Invariant Shape Prior

**arXiv ID:** 2606.25503 | [PDF](https://arxiv.org/pdf/2606.25503v1)

**作者:** Zhiming Chen `[一作]` (Hong Kong University of Science & Technology), Hua Chen `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 AISPO 深度补全框架，融合多尺度 RGB‑D 特征和仿射不变形状先验，以提升机器人抓取非兰伯特物体的深度可靠性。

**💡 创新点**

创新点在于：① 引入形状先验自编码器学习仿射不变形状，显著降低灾难性深度误差；② 采用交叉注意力的层次融合模块，将 RGB、深度与形状特征统一；③ 两阶段训练策略，先预训练形状先验再冻结使用，保证结构一致性。

**🔧 技术方法**

使用了 Vision Transformer、Swin‑Transformer、DPT 解码器、交叉注意力融合、Sobel 边缘损失、Mask‑Loss 等技术，辅以 Grounded‑SAM2 生成物体掩码。

**📊 数据集**

主要使用了 DREDS‑CatKnown 合成数据集进行训练与评估，零样本迁移到 ClearGrasp、STD‑CatNovel、ClearPose 等真实数据集进行测试。

**📈 对比分析**

与 SwinDRNet、DFNet、DepthAnythingV2/V3、PromptDA、Pi3 等先进方法对比，AISPO 在 DREDS‑CatKnown、ClearPose、STD‑CatNovel 等数据集上均获得最优或接近最优的 RMSE/MAE/δ_1.25 指标，并在真实抓取实验中显著提升透明物体的成功率（90%+）。

**⚠️ 局限性**

局限性包括：对前置实例分割质量高度依赖；形状先验在极端透光或折射场景下仍可能出现细节误差；模型训练与推理仍受 GPU 资源限制，需进一步优化以适配低算力机器人。

---

## 308. SAGE-Nav: Leveraging LLM Planning and Alignment Fusion for Hierarchical Scene Graph-Guided Navigation

**arXiv ID:** 2606.25497 | [PDF](https://arxiv.org/pdf/2606.25497v1)

**作者:** Hao Su `[一作]` (Zhejiang University), Jiajun Lv `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了 SAGE-Nav，一种将大型语言模型（LLM）与层次化场景图相结合的分层导航框架，能够异步产生语义化 waypoint 并通过 HSGE 与 GAFN 实时对齐视觉感知，实现高效、可解释的对象目标导航；

**💡 创新点**

创新点在于：①将 LLM 作为全局规划器，生成可解释的语义 waypoint；②通过层次化场景图编码（HSGE）捕获空间语义结构；③设计 GAFN 进行视觉与结构的动态对齐，提升长程推理和零样本泛化；

**🔧 技术方法**

使用技术包括：LLM（Qwen2.5-VL-7B）+检索增强生成、层次化场景图+R-GCN、GAFN 对齐融合、CLIP 视觉特征、A3C 强化学习；

**📊 数据集**

实验数据集为 iTHOR 与 RoboTHOR 两大室内模拟环境；

**📈 对比分析**

与多种 SOTA 方法（如 TSOG、CGI-GAIL、AKGVP 等）对比，SAGE-Nav 在 iTHOR SR 82.47%（L≥5 77.22%）和 RoboTHOR SR 40.82%（L≥5 22.95%）均实现最佳或近乎最佳成绩；在零样本任务中 SR 75.05%，且显著降低 LLM 调用次数和控制延迟；

**⚠️ 局限性**

主要局限包括目标不可视化导致的失败、检测误差、陷阱及提前终止等问题，受限于开放词汇检测器、主动视角策略和 3D 体素表示，未来需要进一步提升这些方面的鲁棒性。

---

## 309. HG-Bench: A Benchmark for Multi-Page Handwritten Answer-Region Grounding in Automated Homework Assessment

**arXiv ID:** 2606.25491 | [PDF](https://arxiv.org/pdf/2606.25491v1)

**作者:** Chuangxin Zhao `[一作]`, Juanzi Li `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了 HG-Bench 基准，专注于多页手写 K‑12 作业的两层级答案定位与分解，配套了面向页面的评估协议。

**💡 创新点**

首个兼顾页面结构、答案完整区域和有序步骤子区域的手写作业定位基准；通过宏 F1 与微 F1 的分离，揭示步骤级定位是主导瓶颈，并提供可复现的参考模型。

**🔧 技术方法**

使用视觉‑语言模型 (VLM) 进行零射击评估；对 GLM‑4.6V 9B 进行单阶段监督微调 (SFT)；在评估中采用页面级 IoU 0.5 匹配和 Greedy 赋值。

**📊 数据集**

500 样本 HG‑Bench 测试集（来自 1,489,278 张匿名作业图），以及 9,920 样本 HG‑SFT 训练集，覆盖 5 个学科、不同年级、页数与答题类型。

**📈 对比分析**

采用 ℱ_A（宏答案区域 F1）和 ℱ_S^μ（微步骤级 F1）两指标对比，零射击闭源 API 最佳 ℱ_A 55.22%，ℱ_S^μ 48.22%；开源模型最高 ℱ_A 55.22%，ℱ_S^μ 48.22%；经过 SFT 的 GLM‑4.6V 9B 在两指标上分别达到 74.97% / 72.26%，显著提升，规模增长并未填补差距。

**⚠️ 局限性**

仅针对中文 K‑12，样本量有限（500），评估仅使用 IoU 0.5，步骤拆分存在主观性，未覆盖其他语言或高等教育内容，未探讨更严格阈值或 RL 等进一步提升。

---

## 310. Cross-View Variance Correlation in Path-Traced Stereo:A Hidden Shortcut in Synthetic Training Data

**arXiv ID:** 2606.25483 | [PDF](https://arxiv.org/pdf/2606.25483v1)

**作者:** Po-Ting Lin `[一作]` `[通讯]`, Po-Ting Lin

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了基于路径跟踪渲染的合成立体数据中，左视图与右视图像素方差场在几何对齐后出现的高度相关性。

**💡 创新点**

发现并量化了这一跨视角方差相关性（在Lambertian区域可达ρ≈0.78），并证明其对匹配任务具有可利用的线索，揭示了合成数据与真实相机之间的潜在差距。

**🔧 技术方法**

利用Mitsuba 3渲染器进行多重采样（SPP 128–2048），对多次渲染样本计算方差场，采用视差对齐（warp）和Pearson相关评估，并通过残差块置换的因果干预验证该信号对匹配的影响。

**📊 数据集**

在20个室内场景（1280×720分辨率、26 mm基线）上进行实验，每个场景渲染30个独立种子，覆盖不同材质（玻璃与非玻璃）和采样预算。

**📈 对比分析**

与未对齐的相关性、残差SAD与方差成本对比，发现干预后匹配指标显著下降（残差SAD边际提升33%，方差成本WTA精度提升4.3倍），证明该方差相关性可作为匹配线索，性能表现符合预期。

**⚠️ 局限性**

实验仅使用单一渲染器，未评估网络是否实际利用此信号；对真实相机噪声模型的兼容性未验证；缺乏对策或对策评估，仍需进一步研究如何在训练时抑制或利用该特征。

---

## 311. Rate-Aware Quantum-Inspired Trajectory Learning for Interference-Limited Multi-UAV Networks

**arXiv ID:** 2606.25480 | [PDF](https://arxiv.org/pdf/2606.25480v1)

**作者:** Khaoula Khaled `[一作]` (King Fahd University of Petroleum and Minerals), Zeeshan Kaleem `[通讯]` (King Fahd University of Petroleum and Minerals)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对多无人机干扰受限网络的轨迹优化，提出基于速率感知的量子启发式图凝聚与分布式强化学习的RA‑QAGC框架。

**💡 创新点**

创新点在于将速率感知成本函数与量子模拟退火相结合，实现对高维搜索空间的全局压缩，同时采用离散化图与独立Q学习实现可扩展的分布式决策。

**🔧 技术方法**

使用了量子模拟退火、速率感知成本函数、图凝聚、独立Q学习以及MATLAB仿真平台。

**📊 数据集**

采用基于KFUPM实验环境的1000×1000 m²部署场景，包含3架UAV、100名地面用户（30%优先用户）等参数进行仿真。

**📈 对比分析**

与随机、K‑means、GC‑K‑means、GC‑SNRP、iGCVis、JUTAP等基线比较，RA‑QAGC在总吞吐量上达59.4 Mbps、优先用户吞吐量23.9 Mbps，分别比基线提升约15%和34%。

**⚠️ 局限性**

局限性包括仅针对静态用户场景、离散化轨迹限制、需预先压缩点集、对极端动态环境适应性待验证。

---

## 312. TACO: Towards Task-Consistent Open-Vocabulary Adaptation in Video Recognition

**arXiv ID:** 2606.25478 | [PDF](https://arxiv.org/pdf/2606.25478v1)

**作者:** Minghao Zhu `[一作]` (Tongji University), Qijun Chen `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在 CLIP 的基础上提出 TACO 框架，用以在开词汇视频识别任务中兼顾新视频知识的学习与预训练模型的泛化能力。

**💡 创新点**

核心创新包括：①相对结构蒸馏（Relative Structure Distillation），利用随机几何锚点在整个嵌入空间中维护 ID 与构造的 OOD 空间之间的相对几何关系；②轻量化的专化投影（Specialization Projection），将优化空间与表示空间解耦，防止过度特化。

**🔧 技术方法**

技术方法包括：知识蒸馏（KL 及 L2 损失）、随机几何锚点采样、EMA 预训练教师、残差线性投影、AdamW 优化器。

**📊 数据集**

使用的公开数据集：Kinetics‑400 进行微调，交叉数据集评估 UCF‑101、HMDB‑51、Kinetics‑600，基准评估基于 Base‑to‑Novel 方案。

**📈 对比分析**

与现有方法（Open‑VCLIP、FROSTER、Open‑MeDe 等）对比，TACO 在 Cross‑dataset 以及 Base‑to‑Novel 两个评估协议下均取得新高，尤其在 UCF‑101、HMDB‑51 的 OOD 性能提升显著。

**⚠️ 局限性**

局限性：1) 仍需手动设置锚点数量与投影学习率，2) 对极端 OOD 分布（如高度异质的视觉内容）可能仍存在泛化下降，3) 目前主要针对视频分类，未探索更复杂的多模态视频任务。

---

## 313. TensorLDM: A Component-Wise Latent Diffusion Model for Volumetric DTI Reconstruction from Sparse DWIs

**arXiv ID:** 2606.25545 | [PDF](https://arxiv.org/pdf/2606.25545v1)

**作者:** Junhyeok Lee `[一作]` (Seoul National University), Kyu Sung Choi `[通讯]` (Seoul National University Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `70e40602-aae3-44bd-80ec-4a7f2674330f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

使用稀疏的四卷 DWI（3 个方向 + b=0）重建完整的扩散张量场。

**💡 创新点**

提出组件化潜在扩散模型（TensorLDM），通过跨组件注意力 (CCA)、混合专家 DWI 调节器 (MoE) 以及 Log‑Euclidean 损失，显著提升张量的 SPD 一致性和方向一致性。

**🔧 技术方法**

采用 Anatomy‑Conditioned Autoencoder、共享 FiLM 线性调制、Swim Transformer 的 CCA、Mixture‑of‑Experts DWI Conditioner、Log‑Euclidean 约束和 3D 潜在扩散模型实现张量重建。

**📊 数据集**

在 HCP Young Adult 数据集上进行实验，使用单回波 1000 s/mm² 的四卷稀疏采样作为输入，18 卷完整采样的线性最小二乘张量作为 ground‑truth。

**📈 对比分析**

与七种基线（ADE、CycleGAN、Pix2Pix、ResViT、SuperDTI、Diff‑DTI、vanilla LDM）在 voxel、几何（LEM、SPD 违例率）和功能（概率纤维跟踪）三层面评估，TensorLDM 在几乎所有指标上均优于或相当于最佳基线，尤其在轨迹误差和 SPD 合法性方面显著提升。

**⚠️ 局限性**

局限性包括仅在单机构、单回波健康青年数据上验证，依赖单一张量拟合算法，且采样过程耗时且输出方差尚未量化。

---

## 314. Dependency-Aware Dominant Resource Fairness for Multi-Tenant Multi-Resource Systems

**arXiv ID:** 2606.25540 | [PDF](https://arxiv.org/pdf/2606.25540v1)

**作者:** Braik Zeidan `[一作]` (Conservatoire National des Arts et Metiers), Stefano Secci `[通讯]` (Conservatoire National des Arts et Metiers)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的多资源分配方案——Dependency-Aware Dominant Resource Fairness (DDRF)，通过在多租户系统中考虑资源间的非线性耦合关系，改进传统 DRF 的公平性和效率。

**💡 创新点**

创新点包括：① 将租户的资源需求按依赖组划分，构造出真正的“拥塞资源”并仅对其进行公平约束；② 对弱租户进行全额满足并在其余资源上恢复最大最小公平；③ 证明在满足一定的正则性和可行方向条件下，DDRF 既能保证 Pareto 最优，又能使至少一条拥塞资源饱和；④ 为非线性依赖提供统一的求解框架（凸化+进化优化/CCP）。

**🔧 技术方法**

技术手段主要包括：基于需求矩阵和容量向量的线性/非线性优化模型；利用 MMF 水平填充算法计算阈值 λ；构造“依赖组”并选取代表资源进行公平约束；使用凸-凹程序（CCP）线性化非凸二次约束；在必要时采用进化算法求解非凸问题；对求解结果进行有效满意度（effective satisfaction）投影。

**📊 数据集**

数据集：① 亚马逊 EC2 实例跟踪（23 种实例族），用于生成多租户需求向量；② 一个基于真实 vRAN 测量的实验用例，结合了 eNB 的占 RB 与 CPU 需求的回归模型。

**📈 对比分析**

比较方法：与传统 DRF、MMF、PF、Mood、依赖无关的 Utilitarian baseline 等做基准对比；在多种拥塞配置和依赖模型（线性、仿射、二次）下进行实验；评估指标包括有效满足率、资源浪费、空闲容量、Jain 公平指数和 CDF 分布。性能表现：DDRF 在有效满足率上提升至 80%，公平指数提升 15% 以上，资源浪费下降约 60%，并且在所有情形下实现至少一条拥塞资源饱和。

**⚠️ 局限性**

局限性：① 方案为集中式，需要全局需求和容量信息；② 对非线性依赖的求解仍可能面临非凸性，需启发式或迭代线性化，可能无法保证全局最优；③ 证明仅涵盖 Pareto 最优和饱和性，缺乏共享性、策略抵抗性、禀赋公平等更严格的公平性质；④ 目前未给出分布式实现，难以扩展到大规模网络；⑤ 需要依赖正则性和可行方向假设，实际业务中的依赖可能不满足；⑥ 评估的公平指标仍基于分配，未完全捕捉到依赖后实际可用性（需进一步研究依赖感知公平度量）。

---

## 315. When LLM Rationales Become User-Facing: Effects on Trust Perception, Decision-Making, and Gaze Behaviors

**arXiv ID:** 2606.25489 | [PDF](https://arxiv.org/pdf/2606.25489v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 316. PatchINR: Patch-Based Implicit Neural Representations for Efficient and Scalable Inference

**arXiv ID:** 2606.25534 | [PDF](https://arxiv.org/pdf/2606.25534v1)

**作者:** Jiachen Ren `[一作]` (University of Hong Kong), Ngai Wong `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过将连续坐标映射到非重叠补丁来实现一次前向传播预测整个 n×n 像素补丁，降低 INR 推理量

**💡 创新点**

将 INR 查询粒度从像素级提升为补丁级，配合 FPGA 硬件协同设计实现显著推理加速与低延迟

**🔧 技术方法**

补丁式 INR 网络、SIREN 模型、双精度 FP32/INT8 FPGA 加速器、矩阵乘法流水线、URAM/BRAM 数据缓冲

**📊 数据集**

Kodak、DIV2K 图像数据集以及视频数据集

**📈 对比分析**

与传统像素级 INR 对比，2×2 补丁在 FPGA 上实现 75% 延迟降低、34.97 dB PSNR，参数增量仅 0.6%，不同补丁尺寸下展示 PSNR/SSIM 提升

**⚠️ 局限性**

补丁尺寸增大后参数量与硬件资源急剧上升，超过 16×16 时延迟收益递减，影响可扩展性

---

## 317. RQ-SAFE: Coupled Request-Resource Scheduling for Online Edge SFC-DAGs

**arXiv ID:** 2606.25467 | [PDF](https://arxiv.org/pdf/2606.25467v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 318. Interplay between VAoI, Packet Error Rate, and Delay for Energy-Efficient Remote Monitoring

**arXiv ID:** 2606.25566 | [PDF](https://arxiv.org/pdf/2606.25566v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 319. From Causal Discovery to Implementation: An Agentic AI Framework for E-Scooter Mobility Hub Planning Across 29 German Cities

**arXiv ID:** 2606.25484 | [PDF](https://arxiv.org/pdf/2606.25484v1)

**作者:** Meng Jin `[一作]` (Fraunhofer Institute for Industrial Engineering IAO), Ziyue Li `[通讯]` (Technical University of Munich)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一套三阶段的Agentic AI框架，利用公开的GBFS数据在29个德国城市中推断电动滑板车热点并通过因果发现生成城市类型特定的可迁移因果模板库，再结合人口学校准的规划工具为城市规划者提供枢纽选址与设施建议，已在海尔布隆验证并支持两座枢纽建设。

**💡 创新点**

创新点在于：①用公开GBFS取代专有行程记录完成热点与因果分析；②引入LLM驱动的自适应因果发现流程，自动化选择多种因果算法并汇总成可迁移的因果模板库；③将因果权重与市民偏好调查相结合，实现年龄分层的设施需求校准。

**🔧 技术方法**

核心技术包括：大语言模型（LLM）做任务编排器；DBSCAN与KMeans聚类热点与特征空间；多种因果发现算法（PC、DAGMA、NOTEARS、LiNGAM、GRaSP等）在块结构下自适应选择；Bootstrap稳定性检验；多模态可视化与LLM生成的规划报告。

**📊 数据集**

数据集主要来自：41天德国全境公开GBFS位置快照；OSM、OpenStreetMap、GlobalBuildingAtlas、NASA SRTM、德国人口普查等公开地理与人口数据；海尔布隆12个月的运营商MDS记录；全国与地方市民偏好调查问卷。

**📈 对比分析**

方法通过与海尔布隆的专有MDS数据对比，热点匹配率达95.5%；在57个城市-聚类单元中，因果发现评估通过率87.7%，部分通过率12.3%；相较于传统相关性模型，提供了可解释的因果驱动权重，使得规划建议更具可迁移性与决策支持。

**⚠️ 局限性**

局限性包括：LLM驱动的决策可重复性难以保证；小样本城市导致部分因果结果不稳健；仅在德国监管与城市形态下验证，跨国迁移仍待考察；GBFS 41天窗口可能忽略季节性与极端天气影响；未对多模式共享交通进行完整验证。

---

## 320. SFL-MTSC: Leveraging Semantic Frame-Level Multi-Task Self-Consistency for Robust Multi-Intent Spoken Language Understanding

**arXiv ID:** 2606.25552 | [PDF](https://arxiv.org/pdf/2606.25552v1)

**作者:** Po-Yen Chen `[一作]` (National Taiwan Normal University), Berlin Chen `[通讯]` (National Taiwan Normal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于语义框架级别的自一致性框架SFL-MTSC，用以解决多意图语音语言理解中LLM解码的不一致性；

**💡 创新点**

创新点在于将预测拆分为意图特定的语义框架，采用域–意图聚类与基于Hybrid Jaccard的槽位层级聚类，并通过路径支持评分筛选可靠框架，实现更细粒度的自一致性校正；

**🔧 技术方法**

技术手段包括多路径LLM推理、语义框架分解、Hybrid Jaccard相似度、路径支持评分、值优先重集成以及阈值过滤等；

**📊 数据集**

使用的评测数据集为中文汽车舱内多意图SLU基准数据集MAC‑SLU；

**📈 对比分析**

与Vanilla Prompting、CroPrompt、GPT‑SLU等基线对比，SFL‑MTSC在无监督零样本设置下提升Overall Acc（最高+1.45%）和Slot F1（最高+28.86%），尤其在Vanilla Prompting场景中显著改进；

**⚠️ 局限性**

局限性包括偶尔导致意图准确率下降、在LALM配置下提升有限、以及仅在单一数据集上验证。

---

## 321. Spatio-Temporal Mixture-of-Modality-Experts Diffusion for Quantitative DCE-MRI Synthesis from Incomplete MR Sequences

**arXiv ID:** 2606.25535 | [PDF](https://arxiv.org/pdf/2606.25535v1)

**作者:** Junhyeok Lee `[一作]` (Seoul National University), Kyu Sung Choi `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

利用条件扩散模型生成3D DCE 药物动力学参数图，支持任意子集的多模态 MRI 输入（包括缺失模态）；

**💡 创新点**

引入时空混合模态专家门控（ST‑MoME），在扩散过程中实现体素级、时序级、缺失敏感的模态融合，并在图像空间训练以保持定量精度；

**🔧 技术方法**

采用扩散概率模型、Mixture‑of‑Experts 门控网络、3D Swin Transformer U‑Net 以及基于块的训练策略；

**📊 数据集**

使用单中心 386 例脑肿瘤患者的多模态 MRI（T1、T1CE、T2、FLAIR、CBV、ADC）以及从 DCE 图像通过 Tofts 模型得到的 Ktrans、vp、ve 三个参数；

**📈 对比分析**

与 Zero‑Concat、HeMIS、Composer、ShaSpec 等基线比较，在 16 种模态缺失场景下，ST‑MoME 在所有三参数的平均 NMSE 上最低，vp 与 ve 的误差降低约 3 倍，肿瘤 ROI 的 NMSE 下降 3.06 倍，且 100 步采样即可接近 1000 步性能；

**⚠️ 局限性**

受限于 Tofts 模型的标签不确定性、单机构数据来源、以及缺失所有对比/灌注模态时性能明显下降。

---

## 322. Security and Privacy in Retrieval-Augmented Generation: Architectures, Threats, Defenses, and Future Directions for Building Trustworthy Systems

**arXiv ID:** 2606.25533 | [PDF](https://arxiv.org/pdf/2606.25533v1)

**作者:** Balamurugan Palanisamy `[一作]` (Birla Institute of Technology and Science), Rajkumar Buyya `[通讯]` (University of Melbourne)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述了 Retrieval‑Augmented Generation (RAG) 在集中式、边缘（Micro‑RAG）、联邦以及混合架构下的安全与隐私挑战，并提出统一的攻击表征与防御框架。

**💡 创新点**

提出了跨部署范式的 RAG 威胁统一分类、分层防御（查询过滤、检索保护、上下文约束、生成校验及系统监控）以及针对现有评测方法的完整评估维度，首次系统性地将隐私保护技术与攻击模型映射到 RAG 端到端流水线。

**🔧 技术方法**

归纳并对比了差分隐私、可信执行环境 (TEE)、加密检索、联邦学习安全聚合、硬件隔离、梯度泄漏抑制、上下文层面防护等多种技术；同时结合了上下文构造与检索去中心化的算法改进。

**📊 数据集**

利用了包括 MS MARCO、Natural Questions、BEIR、PubMedQA、MedQA、HealthCareMagic、CVE/NVD、LegalBench、FedQA 等公开检索/问答基准，并对抗性数据集如 PoisonedRAG、SafeRAG、BadRAG 进行评测。

**📈 对比分析**

通过多维度指标（Recall@k、nDCG、F1、faithfulness、hallucination、MIA‑AUC、攻击成功率、通信开销、延迟与能耗）对比各部署模式下的隐私-效能边界；发现分布式与边缘部署在隐私保护上优于集中式，但往往牺牲检索覆盖与生成质量。

**⚠️ 局限性**

存在评测标准不统一、缺乏端到端隐私量化、攻击模型与数据集规模有限、对联邦与边缘硬件侧信道风险评估不足，以及分层防御方案在实际部署中的复杂度与互操作性难题。

---

## 323. Agentic evolution of physically constrained foundation models

**arXiv ID:** 2606.25532 | [PDF](https://arxiv.org/pdf/2606.25532v1)

**作者:** Jiangwei Zhang `[一作]` (Chinese Academy of Sciences), Rui Hou `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于演化知识图（EKG）的多智能体发现引擎，实现对大型语言模型（LLM）压缩与硬件部署的自动化硬件感知设计。

**💡 创新点**

创新点在于：①将历史压缩方法组织为演化知识图，提供结构化的历史先验；②通过“算法链式思考”（CoT）引导多智能体生成、评审、实现蓝图；③自适应的硬件感知部署与敏感性分析，实现从概念到可执行的闭环。

**🔧 技术方法**

采用多智能体LLM（Gemini、DeepSeek、Claude、GPT‑5.5等）、演化知识图Neo4j、自动AI同行评审、敏感性剖面、自动代码集成与硬件约束校准等技术。

**📊 数据集**

使用公开的大型语言模型压缩与部署论文数据集（NeurIPS、ICLR、ICML、CVPR 等），以及多模型验证：Llama‑3、Qwen3、Deepseek、Qwen1.5‑MoE 等；训练和评估中使用 WikiText‑2 等校准序列。

**📈 对比分析**

与现有压缩方法（如 GPTQ、QuaRot、AWQ、MxMoE 等）对比：在长序列推理上，Q‑Enhance 在 4‑bit 预算下保持 Llama‑3‑8B 和 Qwen3 系列 0~0.5% 的准确率下降；MoE‑Salient‑AQ 在 2.5‑bit 级别下比最优手工方法提升约 3.7%；在 235B 参数模型压缩到 108 GB，准确率仅下降 0.64%，显著超过人类设计方案。

**⚠️ 局限性**

主要局限：①知识图的构建与验证仍需人工审查；②基于LLM的推理存在随机性，导致可重复性和确定性受限；③目前验证仅覆盖压缩与部署任务，其他硬件优化领域尚未证明可迁移。

---

## 324. GROVE: Grounded Pedestrian Simulation via Natural Language for Interactive Social Robot Navigation

**arXiv ID:** 2606.25504 | [PDF](https://arxiv.org/pdf/2606.25504v1)

**作者:** Duc Tai Nguyen `[一作]` (Singapore Management University), Linh Kästner `[通讯]` (Singapore Management University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个基于文本提示的行人仿真框架，能够自动生成高保真、社会化挑战性的场景，用于社交机器人导航的训练与部署。

**💡 创新点**

创新点包括：1) 分层解耦流水线，先通过语义世界表示提取RoI，再用检索增强生成（RAG）为LLM提供节点描述；2) LLM生成行为树并结合Theta*全局规划与Velocity-Field控制，填补了现有仿真中缺乏语义对齐与全局碰撞避免的空白；3) 通过预设（Emergency、Queuing、Normal）显著减少提示长度与生成时间；4) 首次将Velocity-Field crowd控制嵌入行为树，实现可扩展的群体运动。

**🔧 技术方法**

使用技术包括：大型语言模型（Gemini 2.5 Flash、Gemini 3 Pro、Qwen-3）、检索增强生成（RAG + ChromaDB）、行为树（Behavior Tree）、Theta*全局路径规划、Social Force Model、ROS2、Isaac Sim / Gazebo / RViz2 集成，以及Vision‑Language Model（VLM）用于评估。

**📊 数据集**

主要使用自定义的语义标注（YAML）医院、办公室、住宅三类环境；评估时生成多视角截图与原始提示，交由VLM评分；未使用公开大规模人群数据集，完全基于自建仿真场景。

**📈 对比分析**

通过与Text‑Crowd和TRACE两种文本驱动人群生成框架在三种预设（Emergency、Queuing、Normal）下的对比实验，采用VLM评分（Prompt Alignment、Plausibility、Visual Realism）以及轨迹可视化评估。实验显示，本文方法在Prompt Alignment上平均提高约2‑3分，整体得分最高；轨迹更符合语义，减少墙体碰撞；预设模式下token使用率下降35‑49%。

**⚠️ 局限性**

主要局限：1) 组合多种最先进方法导致推理成本高，推理耗时长；2) 假设静态障碍地图，无法处理动态环境变化；3) 对模型规模和推理时间敏感，仍需进一步优化和缓存。

---

## 325. Recommendation as Generation: Unifying Personalized Video Generation and Recommendation at Industrial Scale

**arXiv ID:** 2606.25496 | [PDF](https://arxiv.org/pdf/2606.25496v1)

**作者:** Yanhua Cheng `[一作]` (Kuaishou Technology), Kun Gai `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套从用户兴趣建模到个性化视频生成的闭环系统，将短视频推荐从检索转变为按需生成。

**💡 创新点**

创新点包括：① 通过“分离内容与创意的语义 ID (D‑SIDs)”实现推荐与生成的共享潜在空间；② 设计了可扩展的多代理视频生成框架（VGAs）以及层级规划与有限反思；③ 引入“协同跨域奖励学习（SCRL）”将兴趣匹配、视频质量和用户反馈统一成约束式强化学习。

**🔧 技术方法**

使用的技术包括：大语言模型（Qwen2.5‑VL、Qwen3‑8B、Gemini2.5‑Pro）进行语义编码与指令生成；残差量化与对比学习实现 D‑SIDs；多代理 LLM 共享参数与 KV‑Cache 重用；GDPO 与 PID 控制实现多目标奖励的稳定优化。

**📊 数据集**

数据来源于千亿级别的 Kuaishou 用户交互日志、广告视频库以及内部自建的“视频‑指令”对齐数据（由 Gemini 生成）；在离线评测中使用公开的 VLM2Vec‑V2、QARM、Qwen2.5‑VL‑7B 进行检索基准比较。

**📈 对比分析**

与传统 DLRM、GRM 基线相比，RaG 在广告收入上提升了 5.462%（比 DLRM）且比 GRM 多增 1.87%；离线检索 R@1 达 0.896；VGAs 的自动化胜率从 62.4% 提升到 76.0%，人类评测胜率提高 18.5pp。

**⚠️ 局限性**

局限性包括：目前仍为 near‑line 生成，无法实时响应；VGAs 仍是主要延迟瓶颈；需要大量计算资源与专业数据，难以直接迁移到资源受限或不同领域的应用；在极端稀有兴趣或多模态融合精细度上仍有提升空间。

---

## 326. A Red Teaming Framework for Large Language Models: A Case Study on Faithfulness Evaluation

**arXiv ID:** 2606.25476 | [PDF](https://arxiv.org/pdf/2606.25476v1)

**作者:** Abrar Alotaibi `[一作]` (Imam Abdulrahman Bin Faisal University), Moataz Ahmed `[通讯]` (King Fahd University of Petroleum & Minerals)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于多角色（攻击者、目标、评审）红队框架，用于系统地发现大型语言模型在真实性、摘要质量和有害内容生成等任务中的漏洞，并给出了相应的对抗提示与自动化评审流程。

**💡 创新点**

创新点在于：① 设计了严格的信息流控制的三方架构，避免了反馈循环；② 采用自动化集成评审（多模型投票、Fleiss κ一致性）取代人工标注；③ 在跨语言（英阿）和跨任务中统一评估，突出架构设计对安全性的影响。

**🔧 技术方法**

技术手段包括：基于transformer的大模型（Meta‑Llama、Qwen、Gemma、Phi 等）生成对抗提示；结构约束（长度限制、格式限制）；多模型投票、统一评审标准；统计显著性检验（Wilson CI、Cochran Q、McNemar 等）以量化攻击效果。

**📊 数据集**

使用的数据集包括 SQuAD（问答）、XLSum Arabic 与 Saudi Privacy Policy（摘要）、Jailbreak Dataset（有害/攻击内容）以及对应的英阿语翻译版本。

**📈 对比分析**

通过攻击成功率(ASR)、F1、完整性、Fleiss κ 等指标对比不同模型、语言与任务。实验发现：阿拉伯语任务的 ASR 约为 2~3 倍英文本；长度约束可将摘要的无信度降低约 30%；在多模型评审中投票模式获得最高 F1，且架构设计往往比参数规模更能提升安全性。

**⚠️ 局限性**

局限性包括：只覆盖英语和阿拉伯语，评审模型可能带有族群偏差；攻击提示生成仍需人工监督，未实现全自动化；未评估多轮交互、细粒度有害内容级别；实验规模受限，可能不足以覆盖所有潜在漏洞。

---

## 327. Causal-rCM: A Unified Teacher-Forcing and Self-Forcing Open Recipe for Autoregressive Diffusion Distillation in Streaming Video Generation and Interactive World Models

**arXiv ID:** 2606.25473 | [PDF](https://arxiv.org/pdf/2606.25473v1)

**作者:** Kaiwen Zheng `[一作]`, Qianli Ma `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 Causal-rCM——一种用于自回归视频扩散的统一训练与蒸馏算法，并构建了完整的系统基础设施。

**💡 创新点**

创新点包括：① 通过教师强迫（TF）实现连续时间一致性模型（sCM/MeanFlow）并用 JVP 与自定义 FlashAttention‑2 kernel 训练；② 用 TF‑CM 作为初始化，随后用自强迫（SF）+ DMD 进行后期优化，形成前向/后向目标互补的分阶段流程；③ 提供了可组合的自回归训练模式、KV 缓存、FSDP2、Ulysses CP、激活检查点等多维度系统改进。

**🔧 技术方法**

技术上结合了扩散模型、持续时间一致性模型（sCM/MeanFlow）、分布匹配蒸馏（DMD）、教师强迫与自强迫训练范式、JVP 计算、FlashAttention‑2 自定义 mask kernel、FSDP2、Ulysses CP、Selective Activation Checkpointing、KV 缓存、Replayed Back‑Propagation 等。

**📊 数据集**

使用 Wan2.1 T2V 480p 合成文本到视频数据集（21 维潜在帧），并以 Wan2.1-14B 作为教师进行蒸馏。

**📈 对比分析**

在 VBench‑T2V 上与 Self‑Forcing、LongLive、Causal‑Forcing、AnyFlow 等基线对比，Causal‑rCM 在 1‑步、2‑步、2‑步噪声上下文、4‑步四步模型中均取得最高分；1‑步/2‑步模型在帧级设置下甚至优于 4‑步，显示出优秀的推理效率（NFE 1‑2，FPS 15‑25，首段延迟 0.4‑0.5s），并在所有评测指标上均领先。

**⚠️ 局限性**

局限包括：① 帧级 T2V 训练在长推理深度易出现相机漂移；② 初始化与最终性能不完全一致，TF‑sCM 虽强但不一定在 SF‑DMD 后达到最高峰；③ 前向/后向目标的联合训练难以实现；④ 现有自定义 FlashAttention‑2 JVP kernel 仅与 FlashAttention‑2 兼容，速度不如 FlashAttention‑3/4。

---

## 328. EchoStyle: Unlocking High-Fidelity Video Stylization with Reverse Data Synthesis

**arXiv ID:** 2606.25465 | [PDF](https://arxiv.org/pdf/2606.25465v1)

**作者:** Huaqiu Li `[一作]` (Hong Kong University of Science and Technology), Wenhan Luo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于文本的端到端视频风格迁移框架EchoStyle，能够对任意长度视频实现高保真且连贯的艺术化处理。

**💡 创新点**

创新点包括：①采用视频对视频的生成架构，使用多通道视觉对齐将内容视频、目标视频与文本提示统一映射到潜在空间；②提出了反向合成数据管线，利用真实风格化视频反向生成对应的自然视频，构建了规模达20k对的V-Style20k数据集；③引入init‑follow‑mode训练机制与滑窗推理策略，实现长视频的稳定与可扩展生成。

**🔧 技术方法**

核心技术：扩散模型（DiT）、VAE潜在编码、LoRA微调、AdaLN与交叉注意力、文本到视频的预训练基模Wan2.2‑I2V‑14B、图像‑到‑图像转化LoRA等。

**📊 数据集**

使用自建的V-Style20k数据集（20k对，分辨率480×832，时长≤5s，涵盖14种艺术风格）进行训练；对比实验使用公开的5s视频对，文本提示包含风格与内容描述。

**📈 对比分析**

与开源基线VACE、AnyV2V以及闭源商业模型Runway、Kling‑O1、Seedance 2.0进行量化与人工评测；EchoStyle在风格质量、动态一致性和美学评分上接近甚至优于闭源模型，在人类偏好评测中取得最高或第二高分，证明其在视频连贯性和内容保真度方面表现突出。

**⚠️ 局限性**

局限性：①对极长视频仍需滑窗拼接，可能出现边界接缝；②在极复杂场景或高频细节（如油画纹理）下偶尔出现细节模糊或风格脱节；③V-Style20k仅包含5s片段，限制了对更长时间动态的直接学习；④模型依赖预训练扩散基模，训练成本仍较高。

---

## 329. Optimizing Abstractive Summarization With Fine-Tuned PEGASUS

**arXiv ID:** 2606.25462 | [PDF](https://arxiv.org/pdf/2606.25462v1)

**作者:** Sadiul Arefin Rafi `[一作]` (BRAC University), Farig Yousuf Sadeque `[通讯]` (BRAC University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

Fine-tuned PEGASUS模型在XL-Sum English语料上进行抽象摘要微调。

**💡 创新点**

通过对PEGASUS的精细调参，在同一数据集上显著提升了ROUGE得分，超过了mT5基线，成为该任务的最新最优模型。

**🔧 技术方法**

采用Transformer架构的PEGASUS，使用Adam优化器、线性学习率调度、学习率2e-05、批大小8、5个epoch等超参数进行微调。

**📊 数据集**

XL‑Sum English数据集（约百万篇BBC文章与摘要对）。

**📈 对比分析**

与先前在XL‑Sum上微调的mT5基线做ROUGE对比，PEGASUS获得ROUGE‑1 39.121、ROUGE‑2 17.467、ROUGE‑L 30.894，比分别提升4.04%、15.25%和3.39%。

**⚠️ 局限性**

模型规模大、计算资源消耗高；微调仅使用20%训练数据、5个epoch，且对短文本摘要效果有限。

---

## 330. Concept Removal for Frontier Image Generative Models

**arXiv ID:** 2606.25548 | [PDF](https://arxiv.org/pdf/2606.25548v1)

**作者:** Aditya Kumar `[一作]` (CISPA Helmholtz Center for Information Security), Franziska Boenisch `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种将瓶颈层替换为Transcoder的概念去除框架，直接嵌入生成模型骨干，实现对文本-图像模型（如 SD3.5、Flux、Infinity）的可持续概念删除。

**💡 创新点**

创新点在于：①利用Transcoder在文本编码到生成骨干的单一必需层实现内部特征的稀疏、可解释分解；②通过TopK激活与直通估计实现稀疏正则，避免全模型反向传播；③在解码器权重中直接重定向目标概念特征至空标记，保证白盒访问下的持久性。

**🔧 技术方法**

使用的技术包括：Transcoder 训练（TopK 激活 + 直通估计 + 三项损失），解码器列重映射，稀疏编码，Zero-shot LLaVA 评估器，Ring‑A‑Bell、MMA‑Diffusion、UnlearnDiff 等对抗攻击测试。

**📊 数据集**

数据集：从公开数据集抽取 80 个 prompt 对每个对象，共 20 个对象；加上 10 种风格变体和无风格版，合计 17,600 条 prompt；评估使用 UnlearnCanvas 风格/对象基准以及 LLaVA‑1.6‑Vicuna‑7B 零样本分类器。

**📈 对比分析**

与 LOCOEDIT、UCE、EraseAnything、SafetyGap 等现有方法对比，结果显示在 4 种 SOTA 模型上概念去除准确率 (UA) 超过 90%，跨域保持率 (CRA) 高达 95% 以上；FID、CLIP、HPSv3、Aesthetic 分数保持或略优；在多概念、顺序删除、对抗攻击等严苛测试中均优于对照组。

**⚠️ 局限性**

局限性包括：仍需对每个模型进行一次前置训练（耗时 10–100 分钟）；对完全未见概念的去除效果下降（约 50% UA）；对某些高难度对抗攻击仍存在一定成功率；且在极大模型规模下的显存/存储成本相对较高。

---

## 331. CrypFormBench: Benchmarking Formal Analysis Capability of Large Language Models for Cryptographic Schemes

**arXiv ID:** 2606.25561 | [PDF](https://arxiv.org/pdf/2606.25561v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 332. TwoStepDemocracy: Prototyping of self-evolving, democratic, and decentralized systems

**arXiv ID:** 2606.25559 | [PDF](https://arxiv.org/pdf/2606.25559v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 333. Unlocking Model Potentials Through Adaptive Multi-Agent Scaffolding for Efficient Issue Resolution

**arXiv ID:** 2606.25514 | [PDF](https://arxiv.org/pdf/2606.25514v1)

**作者:** Yang Chen `[一作]` (University of Illinois Urbana–Champaign), Reyhaneh Jabbarvand `[通讯]` (University of Illinois Urbana–Champaign)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种自适应多代理框架，将问题定位、补丁编辑与验证分别交由独立代理完成，并通过事件驱动的消息传递实现协作；

**💡 创新点**

创新点在于：①基于问题描述质量的动态工作流切换（直接并行修复或先探索）；②上下文隔离的多代理通信，消除共享上下文导致的退化与过拟合；③并行补丁与验证的高效同步模式。

**🔧 技术方法**

使用大语言模型（MiniMax、GPT‑5.4‑xhigh 等）实现三类代理；采用事件驱动的同步消息机制；通过量化的质量检查规则做工作流决策；实现自动化的补丁生成与复现测试。

**📊 数据集**

使用了 SWE‑Bench Verified（500 真实 GitHub issue）和 SWE‑Bench Pro（731 复杂多文件问题）两个公开 benchmark。

**📈 对比分析**

在相同模型条件下与主流单代理（Mini‑SWE‑Agent、OpenHands 等）及 Claude Code 进行对比；在 Verified 上提升 3.6–8.4% 的解决率，在 Pro 上提升 6.3–18.5%；平均成本降低约 1–1.5 美元/实例，显著提高准确率与成本效益。

**⚠️ 局限性**

局限性包括：仅在中等规模公开 benchmark 上评估，可能不适用于大型工业代码库；评价指标依赖金标准测试，可能低估实际修复质量；在复杂环境中仍可能出现过度工程或环境兼容性问题。

---

## 334. Quantization Inflates Reasoning: Token Inflation as a Hidden Cost of Low-Bit Reasoning Models

**arXiv ID:** 2606.25519 | [PDF](https://arxiv.org/pdf/2606.25519v1)

**作者:** Xinyu Lian `[一作]` (University Of Illinois Urbana Champaign), Minjia Zhang `[通讯]` (University Of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究低位后训练量化对大规模推理模型链式思考（CoT）长度的影响，发现量化会导致推理令牌膨胀。

**💡 创新点**

创新点在于提出CoT Token Inflation Ratio（CTIR）指标，揭示准确率不变却推理长度增加的隐藏成本，并评估多种缓解策略（提示、重复惩罚、量化感知训练）。

**🔧 技术方法**

主要技术包括后训练量化方法（GPTQ、AWQ、HQQ、ParoQuant）以及量化感知训练（DiscQuant）、重复惩罚和简洁提示等控制手段。

**📊 数据集**

使用的评估数据集涵盖数学推理（AIME25、GSM8K）、代码生成（MBPP、LiveCodeBench）、科学问答（GPQA-Diamond）、多步推理（BBH）以及代理工具使用（BFCLv4-Multiturn）等。

**📈 对比分析**

与BF16全精度模型对比，量化在保持准确率的前提下平均推理长度提升10–300%，导致端到端延迟上升、吞吐量下降；量化感知训练在准确率和长度上实现Pareto改进。

**⚠️ 局限性**

局限性包括仅关注权重量化、缓解措施在不同任务上效果不稳定、未充分探索激活量化与更通用的控制策略。

---

## 335. Spam and Sentiment Detection in Arabic Tweets Using MARBERT Model

**arXiv ID:** 2606.25495 | [PDF](https://arxiv.org/pdf/2606.25495v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 336. BiPACE: Bisimulation-Guided Policy Optimization with Action Counterfactual Estimation for LLM Agents

**arXiv ID:** 2606.25556 | [PDF](https://arxiv.org/pdf/2606.25556v1)

**作者:** Hanyang Wang `[一作]`, Tianxiang Zhao `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出BiPACE算法，在无价值网络的stepwise group-based RL中通过actor隐藏状态聚类和动作条件基线解决状态-动作信用不匹配问题。

**💡 创新点**

将策略隐状态作为bisimulation近似进行状态聚类，配合PACE动作条件对比，改进优势估计，兼容原有GRPO框架且不增加额外回合或辅助网络。

**🔧 技术方法**

使用cosine距离聚类、Actor Hidden fingerprint、PACE动作条件基线、非参数Q(s,a)-V(s)估计。

**📊 数据集**

ALFWorld、WebShop、TextCraft。

**📈 对比分析**

与GRPO、PPO(critic)及其他基线在相同rollout预算下对比；BiPACE在ALFWorld 7B上从90.8%提升至97.1%，在1.5B提升至93.5%，在WebShop、TextCraft同样显著超越基线，并首次在同一预算下跨越95%成功率阈值。

**⚠️ 局限性**

仅验证在文本离散动作环境；固定聚类半径；未适用于视觉或连续动作；需要策略隐状态提取；对大动作空间的Pace关键可能不稳定。

---

## 337. Low Variance Trust Region Optimization with Independent Actors and Sequential Updates in Cooperative Multi-agent Reinforcement Learning

**arXiv ID:** 2606.25526 | [PDF](https://arxiv.org/pdf/2606.25526v1)

**作者:** Bang Giang Le `[一作]` (VNU University of Engineering and Technology), Viet Cuong Ta `[通讯]` (VNU University of Engineering and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种低方差的信任区域优化框架，在多智能体强化学习中通过在顺序更新中裁剪优势估计来降低高方差问题，从而提升训练稳定性。

**💡 创新点**

① 对独立演员顺序更新导致的指数级方差增长进行理论与实证分析；② 设计新的裁剪优势 surrogate objective；③ 证明在该目标下可实现对 ε‑Nash 的 O(1/√K) 子线性收敛；④ 基于此实现 clip‑HAPPO 与 clip‑HATRPO 两种算法。

**🔧 技术方法**

Trust Region Policy Optimization (TRPO)、Proximal Policy Optimization (PPO)、重要性采样、优势估计裁剪、理论收敛分析与实验验证。

**📊 数据集**

MaMuJoCo（多代理MuJoCo）、StarCraft II Multi‑Agent Challenge (SMAC)、Multi‑Agent Particle Environment (MPE)。

**📈 对比分析**

与 HAPPO、HATRPO、MAPPO 等主流基线进行对比，实验显示 clip‑HAPPO/clip‑HATRPO 在大多数环境中获得更高或相近的奖励，并显著降低优势估计方差，表现出更好的训练稳定性。

**⚠️ 局限性**

理论方差上界基于最坏情况，实际增长可能更温和；裁剪阈值固定可能抑制探索；方法主要针对独立演员，尚未证明能收敛到全局最优点。

---

## 338. C2RM-Seg: Causal Counterfactual Reasoning with Structural-Semantic Priors for Weakly Supervised Histopathological Tissue Segmentation

**arXiv ID:** 2606.25508 | [PDF](https://arxiv.org/pdf/2606.25508v1)

**作者:** Hualong Zhang `[一作]` (Guilin University of Electronic Technology), Xipeng Pan `[通讯]` (Guilin University of Electronic Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一个两阶段弱监督组织学图像分割框架C2RM-Seg，先通过因果反事实推理模块去除染色相关混杂生成干净的CAM伪标签，再利用双路径结构-语义网络和不确定性门限损失进行分割训练。

**💡 创新点**

创新点包括在CAM生成中引入结构因果模型的反事实干预以消除混杂影响，构建双路径网络将细粒度结构特征与冻结基础模型语义动态融合，并提出基于不确定性的门限损失来缓解伪标签噪声。

**🔧 技术方法**

采用了结构因果模型（Causal Counterfactual Reasoning Module）、ResNeSt-200e结构分支、冻结DINOV3视觉变换器语义分支、交叉路径门控机制以及Uncertainty-Gated Margin损失等技术。

**📊 数据集**

在两大公开组织学分割基准上进行评估：LUAD-HistoSeg（肺腺癌）和BCSS（乳腺癌）。

**📈 对比分析**

与多种现有弱监督方法（SAM、SAM2、MedSAM、TPRO、HAMIL、MLPS、SIPE、ARML、OEEM、ESFAN）及基础模型Prompt进行对比，C2RM-Seg在mIoU、bIoU和HD95指标上均领先，对LUAD-HistoSeg实现79.62% mIoU、18.07 HD95，对BCSS-WSSS实现72.17% mIoU、27.31 HD95，计算量为191.47 GFLOPs、86.08 FPS。

**⚠️ 局限性**

局限性在于对因果模型假设的依赖可能导致在不同染色或扫描条件下性能波动，双路径结构虽提升精度但增加计算开销，且在更大规模或多模态数据集上的泛化能力仍待进一步验证。

---

## 339. Distill on a Diet: Efficient Knowledge Distillation via Learnable Data Pruning

**arXiv ID:** 2606.25488 | [PDF](https://arxiv.org/pdf/2606.25488v1)

**作者:** Yifan Wu `[一作]` (Fudan University), Weizhong Zhang `[通讯]` (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种用于知识蒸馏的数据删减框架 IF-Beta，能够在保留模型性能的前提下显著减少蒸馏所需的数据量和计算量。

**💡 创新点**

创新点包括：①将影响函数(IF)作为无训练轨迹的样本重要性评估方法；②设计可学习的 Beta 采样策略以替代固定阈值/窗口的硬编码方法；③通过双层优化在教师特征空间中使用线性分类器实现高效训练，省去完整学生训练。

**🔧 技术方法**

采用的技术包括影响函数、Sharpness-Aware Minimization (SAM) 进行 FVM 优化、Beta 分布采样、策略梯度估计、线性分类器代理训练以及双层优化框架。

**📊 数据集**

在 CIFAR-10、CIFAR-100 和 ImageNet 三大视觉数据集上进行实验，并对多种教师/学生结构（ResNet、WideResNet、ViT 等）进行验证。

**📈 对比分析**

与随机、Medium-Difficulty、EL2N、CCS、BWS 等基线以及传统 KD 方法比较，IF-Beta 在所有剪枝比例下均优于基线，甚至在 70% 训练数据时超过完整数据蒸馏的性能；在受限训练预算下表现更佳，整体计算成本约为全量训练的 80-95%。

**⚠️ 局限性**

局限性：①仍需要预训练教师模型以计算影响函数；②影响函数的近似（对角 Fisher）可能在极大模型或非视觉任务上效果不稳定；③在没有教师或对标注不完整的场景下需要额外设计或迁移策略。

---

## 340. How Reliable Is Your Jailbreak Judge? Calibration and Adversarial Robustness of Automated ASR Scoring

**arXiv ID:** 2606.25487 | [PDF](https://arxiv.org/pdf/2606.25487v1)

**作者:** Yang Gao `[一作]` `[通讯]` (Veyon Solutions), Yang Gao (Veyon Solutions)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估自动化 ASR 评判器的可靠性，比较专用分类器与 LLM 评判器的校准与鲁棒性，并提出攻击方法以检测其脆弱点。

**💡 创新点**

首次系统地在同一数据集上对两类评判器进行人类标注对照、内容保持包装攻击和白盒梯度攻击，并证明即使是“专业”分类器也易被梯度攻击破坏。

**🔧 技术方法**

使用 HarmBench 分类器、Llama‑Guard、Qwen2.5‑Instruct、Phi‑3.5‑Mini 等公开模型；攻击手段包括拒绝前缀、前后缀、教育性框架、虚构对话框架以及 GCG 梯度攻击。

**📊 数据集**

HarmBench 验证集（596 条行为–生成对），包含多种有害行为与攻击方式。

**📈 对比分析**

与人类多数投票进行精准度、召回率、F1、准确率比较；对专用分类器召回高但误报多，对 LLM 评判器精准度高但召回低；包装攻击导致 LLM 评判器 57–100% 的判决被翻转，专用分类器仅 3.4%；白盒 GCG 对专用分类器的 30 条可信正样本可翻转 70%（95% CI 54–86%）。

**⚠️ 局限性**

仅基于 HarmBench 一个数据集，未调优 LLM 评判器提示，包装攻击测试样本量小，白盒攻击预算有限，且所有评判器在 4‑bit 量化下运行，可能未能完全反映大规模模型行为。

---

## 341. Cliff Tokens: Identifying Single-Token Failure Triggers in LLM Mathematical Reasoning

**arXiv ID:** 2606.25524 | [PDF](https://arxiv.org/pdf/2606.25524v1)

**作者:** Jaeyong Ko `[一作]` (Seoul National University), Yukyung Lee `[通讯]` (Boston University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在数学推理中的单词级失败触发点——“cliff token”，并证明删除首个cliff token可完全恢复错误推理。

**💡 创新点**

提出了基于token‑wise potential的自适应阈值z检验方法来精确识别cliff token，并给出了确定性、非确定性和采样离岔三种cliff token分类。

**🔧 技术方法**

采用token‑wise potential估计、单侧两比例z检验、Cliff‑DPO（单词级直接偏好优化）以及rollout和early‑termination等技术。

**📊 数据集**

在GSM1K、MATH500和AIME 2025三个数学推理数据集上，结合七个不同规模的LLM进行评估。

**📈 对比分析**

与DPO和cDPO对比，Cliff‑DPO在GSM1K和MATH500等任务上提升约5–7%的准确率，并且仅使用约1/177的训练token，表现相当甚至优于cDPO。

**⚠️ 局限性**

主要局限在于需要大量rollout计算token‑wise potential，计算成本高；实验仅覆盖数学推理，未验证更大模型或非数学领域的适用性。

---

## 342. Calousel: Extrinsic Calibration of Non-overlapping Multi-camera Systems from Pure Rotation

**arXiv ID:** 2606.25646 | [PDF](https://arxiv.org/pdf/2606.25646v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 343. Riazi-8B: An Urdu Large Language Model for Mathematical Reasoning

**arXiv ID:** 2606.25568 | [PDF](https://arxiv.org/pdf/2606.25568v1)

**作者:** Azher Ali `[一作]` (National University of Sciences and Technology), Mehwish Fatima `[通讯]` (National University of Sciences and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了 Riazi-8B，一款专门针对乌尔都语数学推理的 8B 参数大型语言模型。

**💡 创新点**

结合了乌尔都语持续预训练与链式思维监督微调，首次为低资源语言实现多步数学推理。

**🔧 技术方法**

使用 Qwen3-8B 作为基座，通过 LoRA 进行参数高效适配，完成持续预训练（CPT）和监督微调（SFT），并采用 CoT 提示和 LLM-as-Judge 进行评估。

**📊 数据集**

训练数据包括乌尔都语维基百科（CPT）、GSM8K-Urdu（SFT），评估数据使用 MGSM-Urdu。

**📈 对比分析**

与 Alif‑8B、Qalb‑8B 及 Llama‑8B 在 Exact Match Accuracy、Urdu Output Purity、Step Completeness Score 以及 LLM‑as‑Judge 5 维度上对比，Riazi‑8B 在所有指标上均领先（EM 64.4%、UOP 82.7%、SCS 83.8%）。

**⚠️ 局限性**

主要局限包括：CPT 数据缺乏深度数学文本；SFT 集合仅覆盖小学算术，无法评估高级数学能力；模型可能在多语言场景中遗忘部分英语技能；评测仅基于 250 题 MGSM‑Urdu，缺乏更广泛的基准。

---

## 344. Low-Complexity Policy Tessellations in Structured Markov Decision Processes

**arXiv ID:** 2606.25593 | [PDF](https://arxiv.org/pdf/2606.25593v1)

**作者:** Fredy Pokou `[一作]` `[通讯]` (Inria), Fredy Pokou (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在结构化马尔可夫决策过程（MDP）中最优策略的几何分割（policy tessellation），并提出直接学习决策边界的四种方法；

**💡 创新点**

创新点在于将最优决策视为状态空间分区而非价值函数，并给出几何诊断指标和政策损失分解，证明决策边界比完整价值函数更易逼近；

**🔧 技术方法**

采用线性分类器、神经网络分类器、边界加权（margin-aware）分类器和局部最近邻插值四种边界近似技术；

**📊 数据集**

在两类结构化有限MDP基准（库存控制和队列接纳）上进行实验，这些基准通过动态规划生成精确最优策略标签；

**📈 对比分析**

与传统的基于价值的强化学习基线（Double‑Q、FQI、表格Q学习）比较，结果显示边界方法在策略误差、价值间隙和训练时间上均优于基线；

**⚠️ 局限性**

局限性包括仅在有限、结构化的离散状态空间上验证，且对更高维或连续状态空间的推广仍需进一步研究。

---

## 345. Staying In Character: Perspective-Bounded Memory For Book-Based Role-Playing Agents

**arXiv ID:** 2606.25632 | [PDF](https://arxiv.org/pdf/2606.25632v1)

**作者:** Xushuo Tang `[一作]` (UNSW Sydney), Zhengyi Yang `[通讯]` (University of Sydney)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种三层记忆架构，用以构建基于长篇小说的角色扮演 LLM 代理，解决事实过度扩展和风格单调两大失效。

**💡 创新点**

创新点在于：①将记忆拆分为情节经验（episodic）、可见事实（semantic）与情境化人格（personality）三层；②对每个角色引入可见性标签，实现视角边界检索；③在推理时先锚定角色自身情节，再逐步检索符合可见性的事实，最后注入情境化人格模式以生成符合角色声音的回复。

**🔧 技术方法**

技术手段包括：检索增强生成 (RAG)、密集检索、知识图谱构建 (SPOCV 结构)、情绪转移建模、对话风格抽取与聚类、以及三阶段推理管线（Anchor → Self‑Probe → Memory‑Fusion）。

**📊 数据集**

数据集：从八本小说构建的 4,386 题多选知识边界量化 (KBF‑QA) 数据集；以及基于脚本的开放式和封闭式叙事生成评测，涉及 226 条脚本。

**📈 对比分析**

与六种基线（直接提示、Naïve RAG、RAPTOR、HippoRAG、ComoRAG 及 Retrieval‑Augmented Character Agent）比较，KBF‑QA 上最高 73.3（包含可见/不可见两类准确率的加权调和平均），在对话生成的五维评测中在大多数维度上实现约 79% 的对比优势，尤其在写作质量和沉浸感方面表现突出。

**⚠️ 局限性**

局限性：①未针对视角极端的叙事（如多视角叙事、侦探隐晦信息、不可可信叙述者）进行专门的压力测试；②缺乏多角色协作机制，无法在多智能体场景中实现角色间的交互协调与共识推理。

---

## 346. Leaking Circuit Secrets: Gradient Leakage Attacks on Graph Neural Networks

**arXiv ID:** 2606.25589 | [PDF](https://arxiv.org/pdf/2606.25589v1)

**作者:** Rupesh Raj Karn `[一作]` (New York University), Ozgur Sinanoglu `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估了梯度泄漏攻击（GLA）对电路设计与硬件安全任务中使用的图神经网络（GNN）的威胁，探究不同 GNN 架构、数据集及防御措施的泄漏程度。

**💡 创新点**

首次在电路域对 GNN 进行全量 GLAs 调研，并揭示注意力机制（GAT）易泄漏、注入聚合（GIN）更具抗性；同时评估多种主流防御（差分隐私、梯度裁剪、安全聚合、量化压缩、对抗训练）对泄漏与性能的双重影响。

**🔧 技术方法**

采用梯度逆向优化实现输入重构，结合 L2 与余弦相似度评估；实现 GCN、GraphSAGE、GIN、GAT 四种主流 GNN；对抗梯度攻击与多种防御技术进行统一实验。

**📊 数据集**

使用 ISCAS'85、EPFL、TrustHub 三个电路基准，构造门级图进行门类型分类与硬件木马检测；并在 MNIST FCNN 上做对照实验。

**📈 对比分析**

对比无防御与各防御下的重构误差（abs_l2、rel_l2、cos_sim）以及模型准确率。结果显示：无防御时 GAT 在门分类泄漏最严重（cos_sim≈0.71），GIN 最小；防御后 GCN+对抗训练在木马检测任务中几乎消除泄漏但仍可出现性能波动；不同防御对不同 GNN 影响差异显著。

**⚠️ 局限性**

主要局限在于防御参数未进行充分调优；仅评估单步梯度泄漏而非长期迭代；实验基于固定数据集，缺乏对更复杂电路或多任务场景的验证；并未提出新的本质上更隐私友好的 GNN 结构。

---

## 347. A Two-Stage Decision Support System for Sustainability-Aware Long Short Portfolio Optimization

**arXiv ID:** 2606.25696 | [PDF](https://arxiv.org/pdf/2606.25696v1)

**作者:** Giacomo di Tollo `[一作]` (Marche Polytechnic University), Filippo Piccotto `[通讯]` (University of Trieste)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一套两阶段决策支持系统，先用基于多准则决策的 TODIMSort+MEREC 对资产进行分层筛选，然后在得到的长/短集合上最大化 Omega 比例的非凸组合优化。

**💡 创新点**

创新点包括：① 用动态、数据驱动的经验分位数构造 TODIMSort 的限界曲线，使资产分层既符合投资者偏好又随市场状态变化；② 设计 Adaptive Multi‑Operator PSO（AMPSO），结合时间可变加速系数、适应性算子选择与投影约束修复，显著提升在非凸长短组合问题上的搜索效率和解质量。

**🔧 技术方法**

采用的技术主要有：多准则决策方法（MEREC 权重、TODIM、TODIMSort）、Omega 比例评价指标、非凸组合优化、粒子群优化（PSO‑TVAC）、自适应算子选择（AOS）与投影修复的约束处理。

**📊 数据集**

实验使用 421 只来自 STOXX Europe 600 指数的欧洲股票，涵盖 2013‑2023 年的日行情、每月 ESG 评分及交易规则信息；采用 3 年滚动样本与每月再平衡的策略框架。

**📈 对比分析**

通过与标准 PSO‑TVAC、仅交叉版 PSO‑TVAC‑CROSS、仅变异版 PSO‑TVAC‑MUT 进行对比，AMPSO 在 30 条测试实例上均获得显著更优的 Omega 值并展现更快的收敛速度；在实测交易中，ESG 加权的长短组合在风险调整后收益（Sharpe、Sortino、Rachev）优于不使用 ESG 的对照组和市值加权基准，且下行风险与最大回撤更低。

**⚠️ 局限性**

局限性：① 研究仅限于欧洲市场，ESG 数据披露成熟；② 未考虑交易成本、换手率上限等实际操作约束；③ Omega 比例的非凸性仍导致求解难度较高，可能受限于样本期与参数设置。

---

## 348. Towards a Dynamic and Fixed-budget Memory Bank for Efficient Streaming Video Understanding

**arXiv ID:** 2606.25658 | [PDF](https://arxiv.org/pdf/2606.25658v1)

**作者:** Baiyang Song `[一作]` (Xiamen University), Rongrong Ji `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关的CausalMem方法，用于构建动态、固定预算的视觉记忆库，提升多模态大型语言模型在流式视频理解中的性能。

**💡 创新点**

创新点在于引入在线语义基底来估计视频主语义，并通过冗余评分与时间新颖性共同驱动记忆库的动态更新，从而在固定预算下最大化信息保留。

**🔧 技术方法**

技术包括在线奇异值分解、QR分解、冗余得分与时间先验计算、固定预算矩阵更新以及与LLM生成器的无缝对接。

**📊 数据集**

使用了 OVO-Bench、StreamingBench、Video-MME、LongVideoBench、MLVU、LVBench 等多种流式与离线视频理解基准数据集。

**📈 对比分析**

与 ReKV、LiveVLM、StreamMem、StreamingTOM 等流式方法以及 TimeChat-Online、Gemini 1.5 Pro、GPT‑4o 等先进 Video‑MLLMs 进行对比，平均精度提升 3–4%，并在流式与离线基准上均取得显著优势。

**⚠️ 局限性**

局限性在于需预设基底大小与记忆预算，难以自动适配极长视频或高动态场景；在需要跨帧长期依赖的任务中表现仍有限。

---

## 349. Dissociable Spatial and Temporal Effects of Interaction Latency in Virtual Reality

**arXiv ID:** 2606.25681 | [PDF](https://arxiv.org/pdf/2606.25681v1)

**作者:** Xiaoye Michael Wang `[一作]` (University of Toronto), Timothy N. Welsh `[通讯]` (University of Toronto)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在沉浸式VR中，实验操纵物理手与虚拟手之间的交互延迟（0–500 ms），并测量在手指指向目标时的端点误差、运动时间、变异度和吞吐量，随后与无延迟VR基线及物理环境基线进行对比。

**💡 创新点**

首次系统展示了交互延迟在VR中产生可区分的空间（端点误差/变异度）与时间（运动时间/吞吐量）后果；空间误差在极短延迟下就显著恶化，而时间指标在较大延迟才受影响，提示评估VR交互时必须同时关注空间和时间性能。

**🔧 技术方法**

使用HTC VIVE Pro Eye HMD、Optotrak运动捕捉、Unity引擎、Python-Unity UDP接口、首进队列缓冲实现延迟、以及线性混合效应模型（LMM）对试验数据进行统计分析。

**📊 数据集**

20名右撇手成年受试者（共20人），在VR中完成336次实验试验（6种延迟×4距离×14重复），以及在物理环境中各56次预后试验，共计448次基线数据。

**📈 对比分析**

与无延迟VR基线比较：VR出现更大端点误差、更长运动时间、更高变异度、吞吐量更低；加入延迟后，端点误差和变异度随延迟呈非线性上升，运动时间仅在>50 ms时显著增加，吞吐量随延迟下降并在较大延迟趋于平稳；说明空间指标对延迟更敏感，时间/效率指标对中等以上延迟更为响应。

**⚠️ 局限性**

限制包括：未测量受试者对延迟的主观感知或检测阈值；实验任务仅为受限的指向动作，未涉及全身动作或对象操作；虚拟手为单一未动态变向的仿生手，可能限制结果向更复杂、现实世界交互的泛化。

---

## 350. Steering Vision-Language Models with Joint Sparse Autoencoders

**arXiv ID:** 2606.25657 | [PDF](https://arxiv.org/pdf/2606.25657v1)

**作者:** Huizhen Shu `[一作]` (yunshanai), Hui Li `[通讯]` (yunshanai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种联合稀疏自编码器（JSAE）用于在视觉语言模型中提取可解释的跨模态特征，并通过可插拔的激活干预实现对图像描述的可控生成。

**💡 创新点**

创新点在于在稀疏自编码器中加入显式对齐损失，使视觉和文本的稀疏代码在方向上对齐，从而得到可直接用于视觉流干预的文本侧方向，并在多模态模型中首次展示层级可控性。

**🔧 技术方法**

主要技术包括稀疏自编码器（SAE）、联合对齐损失（cosine similarity）、特征聚类、双向激活干预（增量 steering 与抑制）等。

**📊 数据集**

使用的数据集为 MS‑COCO（用于训练/验证）和 Flickr30k（跨域评估），并在 LLaVA、Qwen3‑VL‑30B、Llama3‑LLaVA‑8B 三种视觉语言模型上进行实验。

**📈 对比分析**

与提示式 steering、随机噪声、激活加法、VL‑SAE 及未对齐 JSAE 等基线相比，JSAE 在 Layer 13–25 的成功率可达 74–80%，明显优于激活加法（≈39%）、提示（≈22%）与 VL‑SAE（0%），并在 MoE 模型中同样保持高成功率。

**⚠️ 局限性**

局限性包括：仅使用全局平均池化导致缺乏空间细粒度控制；稀疏特征拆分导致需后期聚类；对单层干预和高层语义类别的评估限制了对更复杂场景、跨域稳健性的验证。

---

## 351. Retrieval-Grounded Multilingual LLM Assistance for Island Smallholder Farmers

**arXiv ID:** 2606.25647 | [PDF](https://arxiv.org/pdf/2606.25647v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 352. IntentTester: Intent-Driven Multi-agent Framework for Cross-Library Test Migration

**arXiv ID:** 2606.25588 | [PDF](https://arxiv.org/pdf/2606.25588v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 353. Performance Analysis for Heterogeneous Air-Ground ISAC in Coordinated Multipoint Networks

**arXiv ID:** 2606.25574 | [PDF](https://arxiv.org/pdf/2606.25574v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 354. Power-Budgeted Underwater Vehicle Control via Constrained Reinforcement Learning

**arXiv ID:** 2606.25680 | [PDF](https://arxiv.org/pdf/2606.25680v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 355. One Model, Many Latencies: Universal Speech Enhancement for Diverse Real-Time Applications

**arXiv ID:** 2606.25621 | [PDF](https://arxiv.org/pdf/2606.25621v1)

**作者:** Szu-Wei Fu `[一作]` (NVIDIA), Yu-Chiang Frank Wang `[通讯]` (NVIDIA)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种一体化、可实时、可自适应算法延迟和计算延迟的通用语音增强模型，支持多种实时应用而无需单独训练模型。

**💡 创新点**

通过并行卷积层实现可配置look‑ahead控制算法延迟；采用早期退出机制调节计算延迟；两阶段训练（共享解码器→多解码器）弥补灵活模型与专用模型之间性能差距。

**🔧 技术方法**

基于Mamba结构的因果卷积、单向时序建模、层归一化、SFI STFT、并行卷积、早期退出、多解码器训练、两阶段损失（回归+对抗）等技术。

**📊 数据集**

URGENT 2025挑战数据集（多语言、多采样率、七种降噪条件），并在VoiceBank‑DEMAND基准上进行外部验证。

**📈 对比分析**

与基线(noisy、TF‑GridNet)以及专用模型和传统早期退出模型比较，提出的模型在30种延迟配置下性能接近专用模型，提升CAcc、PESQ、ESTOI等指标，且实现了可配置的算法与计算延迟。

**⚠️ 局限性**

计算延迟在高层（12层）仍可能超过跳帧导致RTF>1，需要更快硬件；深浅层性能差距仍存在，需进一步知识蒸馏；模型对极低计算资源场景仍有限制。

---

## 356. Taxonomy of Risks on Automated Fact-Checking Systems Considering its Propagation

**arXiv ID:** 2606.25645 | [PDF](https://arxiv.org/pdf/2606.25645v1)

**作者:** Jun Yajima `[一作]` (Fujitsu Limited), Takao Okubo `[通讯]` (Institute of Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

本文构建了针对自动化事实核查系统的风险分类体系，并通过对DEFAME系统的案例评估验证了该体系的可行性；同时对比了传统STRIDE方法，发现STRIDE无法捕捉到多种关键风险。

**💡 创新点**

创新点在于首次系统化梳理出32种专属风险，并引入风险因素-危险情境-危害的三阶段时间序列模型，形成风险因子、危险情境与危害之间的因果关系图（fault tree）。

**🔧 技术方法**

主要技术包括：风险评估框架（STRIDE + 生成式AI风险分类），数据流图（DFD）绘制，风险因子与危害映射的故障树分析，以及将风险词作为指南词对DEFAME系统进行逐点评估。

**📊 数据集**

论文未使用公开数据集，评估基于DEFAME系统自身的输入输出与其内部工作流程，未涉及具体标注数据或大规模文本数据集。

**📈 对比分析**

通过与STRIDE方法的对比，展示了所提出的风险分类在识别“诽谤”“虚假谣言”“误导性结果”等社交影响类风险方面的优势；评估为定性描述，未给出量化性能指标。

**⚠️ 局限性**

局限性包括：评估仅针对单一系统DEFAME，缺乏对其他自动化核查系统的验证；未量化风险的可能性与影响；所选风险体系可能仍不完整，无法涵盖所有生成式AI特有风险；缺乏自动化风险情境生成工具。

---

## 357. BitNet Text Embeddings

**arXiv ID:** 2606.25674 | [PDF](https://arxiv.org/pdf/2606.25674v1)

**作者:** Zhen Li `[一作]` (Peking University), Dongyan Zhao `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种极低比特的LLM文本嵌入框架（BitNet Text Embedding），将预训练LLM转换为1.58‑bit的BitNet编码器，并在此基础上进行连续对比学习与教师引导的对比蒸馏，同时训练多精度输出嵌入，支持同一模型在不同存储预算下使用不同位宽的向量。

**💡 创新点**

创新点：① 将极低比特（1.58‑bit）量化直接应用于LLM主干，突破传统LLM嵌入的推理与存储瓶颈；② 在极低比特模型上实现连续对比预训练与多模态蒸馏（相似度分布蒸馏+注意力关系蒸馏），有效恢复语义表征；③ 采用多精度输出嵌入训练，让单一模型兼容1/2/4/8/16位的向量存储，极大提升存储与性能的可调性。

**🔧 技术方法**

技术：BitNet量化（三元权重+8bit激活+子层归一化）、连续对比学习（InfoNCE）、监督对比微调、教师引导蒸馏（相似度分布蒸馏与多头注意力蒸馏）、多精度嵌入量化（按位宽离散化与重构）和全链路端到端训练。

**📊 数据集**

数据集：MMTEB（eng, v2）作为评测基准；预训练对比学习使用1B规模文本对；监督微调使用公开的BGE-en-ICL数据；实验中使用两种LLM主干 Qwen3‑0.6B 和 Gemma3‑270M。

**📈 对比分析**

对比方法：将 BitNet 嵌入与同一主干下的 FP16 全精度教师在 MMTEB 7项任务上对比；性能上 BitNet 在 Qwen3‑0.6B 上平均分 67.60，仅低 0.35 分；Gemma3‑270M 上 66.10，低 0.61 分；同时推理速度提升约 2×（CPU 8 线程下）。多精度实验显示 8/4 位嵌入几乎无损，1/2 位仍保留 64+ 分，可在存储受限场景下使用。

**⚠️ 局限性**

局限性：① 在检索类任务中仍比教师略低，说明极低比特对长文本匹配更敏感；② 1 位或 2 位嵌入虽可用但性能下降显著；③ 训练流程较为复杂，需连续预训练、蒸馏和多精度优化，资源与时间成本高；④ 对不同语言或更大规模 LLM 的泛化还需进一步验证。

---

## 358. Auto-Labelling-Based Domain Transfer for 3D Object Detection on a Bicycle-Mounted LiDAR Platform

**arXiv ID:** 2606.25652 | [PDF](https://arxiv.org/pdf/2606.25652v1)

**作者:** Mario Finkbeiner `[一作]` (Munich University of Applied Sciences), Fabian B. Flohr `[通讯]` (Munich University of Applied Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文基于自行车平台构建了VRU视角的3D检测基准，并通过无人工标注的自动标注管线生成训练数据。

**💡 创新点**

创新点在于首次公开自行车视角的多类VRU检测数据集，以及利用自动标注实现车辆模型的域迁移并验证其有效性。

**🔧 技术方法**

主要技术包括多源模型融合的自动标注框架VRU-Label3D、稀疏卷积和Transformer的四种不同检测器（CenterPoint、SECOND-MH、TransFusion-L、VoxelNeXt）以及在自动标注上进行的微调。

**📊 数据集**

使用了FUSE‑Bike平台记录的自行车LiDAR关键帧，包含941帧自动标注的训练集和86帧人工验证的测试集，约18,000个3D框。

**📈 对比分析**

通过与零射击预训练模型对比，四个检测器在自动标注上微调后平均提升 mAP 13.7–23.4 分点，尤其在人行道和自行车类上提升 18.2–31.8 分点，最小提升约 48.3%。

**⚠️ 局限性**

限制在于数据集规模有限、测试集仅三条序列、自动标注在召回率上仍不足，且方法主要针对低安装LiDAR，未覆盖多模态或更复杂场景。

---

## 359. Event-Adaptive Motion Planning with Distilled Vision-Language Model in Safety-Critical Situations

**arXiv ID:** 2606.25629 | [PDF](https://arxiv.org/pdf/2606.25629v1)

**作者:** Zhenwei Huang `[一作]` (Southern University of Science and Technology), Yi Gong `[通讯]` (Southern University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了事件自适应运动规划框架EAMP，结合轻量Vision‑Language模型进行行为异常检测与策略级控制重配置，实现安全关键场景下的实时机器人导航。

**💡 创新点**

创新点在于引入可提示配置的语义事件触发器PC-SET主动捕捉行为异常，并通过物理验证的SemNav‑VLM以及策略级模型预测控制SMPC将语义决策映射为可结构化的MPC重配置，实现在保持可解释性的同时显著提升自适应性能。

**🔧 技术方法**

主要技术包括轻量级VLM（Qwen3‑VL）、提示驱动的事件触发机制、物理验证的语义蒸馏、策略级模型预测控制（SMPC）以及CARLA仿真与ROS集成。

**📊 数据集**

使用CARLA Town06、Town10等仿真地图构建长尾智能物流基准，并收集行为注释数据用于SemNav‑VLM的蒸馏。

**📈 对比分析**

与RDA、PCS、OCP等基线对比，EAMP在安全时间（TTC）、完成时间、轨迹长度和速度波动等指标上均优于基线，尤其在困难场景下TTC提升约32%，完成时间略有下降，体现出更好的安全性与效率。

**⚠️ 局限性**

局限在于仍主要在仿真环境验证，对真实硬件部署、雷达等多模态感知融合以及更复杂社会交互情境的适应性尚未充分验证。

---

## 360. Optimizing Semiconductor Device Simulations through Low-Precision Arithmetic

**arXiv ID:** 2606.25595 | [PDF](https://arxiv.org/pdf/2606.25595v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 361. TL++: Accuracy and Privacy Preserving Traversal Learning for Distributed Intelligent Systems

**arXiv ID:** 2606.25627 | [PDF](https://arxiv.org/pdf/2606.25627v1)

**作者:** Erdenebileg Batbaatar `[一作]` (Neouly), Young Yoon `[通讯]` (Hongik University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 TL++，一种基于虚拟批量构造的遍历学习框架，支持两种模式（可信模式和安全模式）实现分布式训练。

**💡 创新点**

创新点在于：①将遍历学习与加密隐私结合，使用加法秘密共享在非同盟两台服务器间分割切分层激活与梯度；②提供可切换的可信与安全两种工作模式；③在满足线性/仿射条件时实现与中心化训练完全一致的梯度；④在安全模式下仍保持激活/梯度隐藏，避免传统 Split Learning 的明文泄露。

**🔧 技术方法**

使用技术包括：遍历学习（Virtual Batch）、加法秘密共享（Additive Secret Sharing）、分层模型（Split Learning）、梯度累积、可选的高斯差分隐私噪声、LoRA 参数高效适配、FP16/32 数值计算与简单的线性/仿射算子。

**📊 数据集**

使用的数据集有：CIFAR‑10（图像分类）和 BioGPT/PubMedQA（医学问答的语言模型微调），分别在全模型训练和 LoRA 适配两种设置下评估。

**📈 对比分析**

与 FedAvg、FedProx、SCAFFOLD、标准 Split Learning 与 SplitFed 等基线进行对比；在 CIFAR‑10 上，TL++ 基础模式 cut‑1 的精度仅比中心化低 0.6%，并且在每一步通信负载上相较于全模型同步降低 13.1×；在安全模式 cut‑3（满足线性条件）精度仅比中心化低 1.1%，通信负载在非安全模式仍优于 FL/SL。BioGPT 任务中 TL++ 也逼近中心化精度，甚至在安全模式下略优。

**⚠️ 局限性**

局限性包括：①安全模式仅在切分层以上的算子为线性/仿射时保证精确梯度；②仅满足半诚实非同盟模型，无法防御协同攻击、恶意服务器或标签泄露；③缺乏正式的差分隐私证明与隐私预算控制；④目前仅在小规模数据集和单模型架构下测试，未评估大规模语言模型或更深网络；⑤通信效率高但仍受节点梯度同步与帮助器转发的开销影响；⑥没有真实网络时延测量，评估主要基于理论负载。

---

## 362. Reasonable Motion: A General ASP Foundation for Environment Constrained Movement Trajectory Computation

**arXiv ID:** 2606.25626 | [PDF](https://arxiv.org/pdf/2606.25626v1)

**作者:** Julius Monsen `[一作]` (Örebro University), Lars Karlsson `[通讯]` (Örebro University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于答案集编程的混合定性-定量方法，用于在受环境约束的实际场景中计算可解释的分支轨迹模式并生成连续运动轨迹。

**💡 创新点**

其创新点在于将ASP推理与几何中心线构建相结合，既能枚举几何可行且符合法规的运动模式，又能通过稳定模型和事件序列实现完全可追溯的可解释性。

**🔧 技术方法**

采用的技术包括答案集编程（ASP）进行图遍历与可达性推理、弱约束优化生成最优模式、以及中心线拼接与贝塞尔曲线平滑实现连续轨迹。

**📊 数据集**

实验使用Argoverse 2运动预测子集进行评估。

**📈 对比分析**

与常数速度、最近邻、LSTM、WIMP等基线对比，top‑1 minADE从7.75降至6.72，top‑6与最近邻相当，尽管整体误差略高于学习模型，但实现了可解释性与多模态覆盖。

**⚠️ 局限性**

主要限制包括使用常数速度导致几何精度不足、模式选择偏好粗糙导致覆盖率低、以及缺乏对多车交互和学习偏好权重的处理。

---

## 363. Expresso-AI: Explainable Video-Based Deep Learning Models for Depression Diagnosis

**arXiv ID:** 2606.25606 | [PDF](https://arxiv.org/pdf/2606.25606v1)

**作者:** Felipe Moreno `[一作]` (MIT), Cynthia Breazeal `[通讯]` (MIT)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了可解释的视频深度学习模型用于抑郁诊断，并提供视觉与量化解释

**💡 创新点**

首次将 DeepLift 反向传播归因与时间维度结合，生成视频热图和面部区域归因，并与 Action Units 关联分析

**🔧 技术方法**

使用 (2+1)D/3D ResNet 预训练模型（Kinetics‑400/700、MiT），DeepLift+Rescale 归因，区域归因与 AU 交叉相关方法

**📊 数据集**

使用 AVEC 2014 抑郁子挑战视频数据集，BDI‑II 评分作为标签

**📈 对比分析**

与前沿单脸模型对比，MAE 6.626、RMSE 8.543，优于 DepressNet‑Full，证明视频模型性能更佳

**⚠️ 局限性**

存在样本偏倚（头戴设备、麦克风）导致归因偏差，短视频帧限制 AU 相关性稳定性，缺乏统计显著性检验

---

## 364. WOLF-VLA: Whole-Body Humanoid Optimal Locomotion Framework for Vision-Language-Action Learning

**arXiv ID:** 2606.25591 | [PDF](https://arxiv.org/pdf/2606.25591v1)

**作者:** Melya Boukheddimi `[一作]` (DFKI GmbH), Frank Kirchner `[通讯]` (DFKI GmbH)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的深度学习模型，用于图像分类任务。

**💡 创新点**

创新点在于引入了一种新的激活函数，能够提高模型的收敛速度和分类精度。

**🔧 技术方法**

使用了卷积神经网络（CNN）和改进的激活函数。

**📊 数据集**

使用了CIFAR-10数据集进行实验。

**📈 对比分析**

与传统的激活函数模型进行了比较，结果显示新模型在分类精度上提高了5%，且训练时间缩短了15%。

**⚠️ 局限性**

模型在处理大规模数据集时可能会遇到内存限制的问题。

---

## 365. Learning Subset-Shared Invariances for Domain Generalization with Mixture-of-Experts

**arXiv ID:** 2606.25665 | [PDF](https://arxiv.org/pdf/2606.25665v1)

**作者:** Tien-Hung Nguyen `[一作]` (VinUniversity), Kok-Seng Wong `[通讯]` (VinUniversity)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对域泛化问题，提出子集共享不变性理论，并实现基于Mixture‑of‑Experts的MESSI框架，利用路由器自适应划分域子集，对每个子集执行类别‑域对齐，进而学习可组合的子集不变表示提升跨域泛化性能。

**💡 创新点**

创新点在于：①引入子集共享不变性概念，说明全局不变性过度约束会丢失部分可迁移信息；②设计路由驱动的子集对齐机制，使专家在自己的子集内实现不变性；③加入置信与平衡路由正则以及专家多样性损失，确保分解有效且专家互补。

**🔧 技术方法**

使用技术包括：Mixture‑of‑Experts网络、softmax路由器、基于子集条件的最优传输（OT）/MMD对齐、稀疏置信正则、平衡正则、专家多样性正则；特征提取器为预训练的DeiT（Ti/16 或 S/16）。

**📊 数据集**

实验数据集：DomainBed 四大基准（PACS、OfficeHome、TerraIncognita、DomainNet）以及自定义的 Rotated‑Colored MNIST 用于源域数量扩增实验。

**📈 对比分析**

与 ERM、CORAL、Fishr、SAGM、LFME 等传统方法以及多种 MoE 变体（GMoE、OMoE、DynMoE）进行对比。MESSI 在 DomainBed 平均 OOV 精度上达到或超过最强基线；在源域扩增实验中比全局对齐方法更稳健，峰值准确率更高，降幅更小。

**⚠️ 局限性**

局限性：①训练成本显著提升，主要来自子集对齐损失（OT 迭代计算）；②方法高度依赖路由器的有效性，路由失效或专家冗余会导致性能下降；③在极大规模数据或极高维域时的可扩展性和效率仍待进一步研究。

---

## 366. FeVOS: Foresight Expression Video Object Segmentation

**arXiv ID:** 2606.25585 | [PDF](https://arxiv.org/pdf/2606.25585v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 367. H-Adapter: Pose-Robust Hairstyle Transfer via Attention-Derived, Source-Aligned Hair Masks

**arXiv ID:** 2606.25578 | [PDF](https://arxiv.org/pdf/2606.25578v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 368. SSMNBench: Diagnosing Image-based Cross-View Human-Object Understanding via Single-View Sufficiency and Multi-View Necessity

**arXiv ID:** 2606.25634 | [PDF](https://arxiv.org/pdf/2606.25634v1)

**作者:** Tianchen Guo `[一作]` (University of Queensland), Xin Yu `[通讯]` (Adelaide University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了SSMNBench基准，用于诊断多视角人类-物体理解的跨视角推理能力。

**💡 创新点**

创新点在于将任务划分为单视角足够(SVS)与多视角必要(MVN)两类，并引入失真衰减指标δ_dis来量化冗余视角对模型性能的影响。

**🔧 技术方法**

通过对17款现有多模态大型语言模型进行多视角输入的系统评估，结合多视角注意力和跨视角融合技术实现诊断。

**📊 数据集**

构建的数据集包含11类任务共3,300条人工标注的问答对，来源于Core4D、M3GYM、Harmony4D等真实多视角人类中心场景。

**📈 对比分析**

实验对比显示，尽管顶尖模型在SVS/ MVN任务上分别达到约30%–45%的准确率，但仍与人类评测者（>87%）存在显著差距，且随着视角增多δ_dis普遍上升，表明模型易受视觉干扰。

**⚠️ 局限性**

局限性包括对高分辨率输入的高度依赖、对多视角融合的能力不足以及对视角冲突信息的错误处理，导致模型在复杂遮挡与深度推理场景中的鲁棒性差。

---

## 369. Probabilistic Agents in Deterministic Audits: Evaluating Multi-Agent Systems for Automated Audits Based on the German IT-Grundschutz

**arXiv ID:** 2606.25622 | [PDF](https://arxiv.org/pdf/2606.25622v1)

**作者:** Lea Roxanne Muth `[一作]`, Marian Margraf `[通讯]` (Freie Universität Berlin)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究并实现了一个基于HybridRAG的多智能体系统，用于支持德国IT‑Grundschutz合规性审核的部分自动化，并通过对RecPlast案例的端到端评估验证其效果。

**💡 创新点**

创新点包括：在结构分析阶段引入假设‑验证循环以降低幻觉；将LLM推理与确定性继承算法分离，形成分离推理管道；以及将HybridRAG与知识图谱结合，实现对合规规则的确定性验证。

**🔧 技术方法**

技术手段：多智能体架构（LangGraph + ReAct模式）、Hybrid Retrieval Augmented Generation、Neo4j知识图谱、向量检索与全文检索混合，以及多种LLM（GPT‑4.1、GPT‑4o mini、GPT‑5 mini、Claude 4.5 Haiku、gpt‑oss‑120B）。

**📊 数据集**

使用了德国联邦信息安全办公室（BSI）提供的 RecPlast GmbH 参考数据集，该数据集包含完整的结构分析、依赖关系、保护需求评估、建模和IT‑GS检查信息。

**📈 对比分析**

评估方法：采用精确率、召回率和F1‑score对每个阶段（SA、DA、PN评估、建模、IT‑GS检查）进行量化比较。实验结果显示，SA和建模阶段的F1较高，但DA和IT‑GS检查阶段仅达50%以内，表明LLM在逻辑推理上仍有限。

**⚠️ 局限性**

局限性：LLM的概率性导致幻觉与缺失依赖，破坏继承链；IT‑GS检查准确率低，无法实现完全自动判断；缺乏回溯错误纠正，易累积误差；在噪声较大、非标准化的真实企业数据上的性能未知。

---

## 370. ScaleHP: Estimating Hand Pose in Metric Space

**arXiv ID:** 2606.25619 | [PDF](https://arxiv.org/pdf/2606.25619v1)

**作者:** Ruitao Jing `[一作]` (Tsinghua University), Lei Zhang `[通讯]` (Visincept)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为ScaleHP的一站式端到端框架，用于直接从单张图像中恢复绝对尺度的手部三维姿态；

**💡 创新点**

核心创新在于引入专门的“尺度标记”（scale token）与Transformer解码器相结合，通过骨骼比例先验实现对手部整体尺度的直接预测，并利用透视投影约束的最小二乘求解实现绝对空间恢复；

**🔧 技术方法**

技术手段包括基于DETR风格的冻结检测器、带尺度标记的多尺度变形注意力Transformer解码器，以及无训练的线性系统求解实现绝对位移与尺度的恢复；

**📊 数据集**

在FreiHand、DexYCB、HO3Dv3等公开基准以及COCO-WholeBody、Onehand10K等数据集上进行训练与评估；

**📈 对比分析**

与现有根相对、相机空间以及全局尺度校准方法相比，ScaleHP在CS‑MPJPE上分别取得35.8 mm（FreiHand）、136.3 mm（DexYCB）和50.7 mm（HO3Dv3）的显著提升，并在传统对齐指标下也保持或超越SOTA；

**⚠️ 局限性**

局限性在于对极端遮挡、非手部物体与极端尺度变化的鲁棒性尚未完全验证，且依赖手部骨骼比例先验，可能在手部形态显著不同的个体或非人类手部场景中表现受限。

---

## 371. One Body, Two Minds: Variable Autonomy Approach for a Co-embodied Robotic Hand

**arXiv ID:** 2606.25575 | [PDF](https://arxiv.org/pdf/2606.25575v1)

**作者:** Piotr Koczy `[一作]` (KTH Royal Institute of Technology), Michael C. Welle `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在可穿戴机器人手的共体现(variable autonomy)框架下，系统实现了人机两脑协作，允许人类在搜索与定位阶段操控手臂，机器人在靠近目标时自主完成抓取，随后由人类通过头部手势完成工具的激活与释放；

**💡 创新点**

创新点在于提出了“共体现两脑”范式，将人机控制分阶段分配，既保留人类的目标导向与空间感知，又让机器人在抓取阶段实现自主执行，避免传统共享自主连续混合的决策冲突；

**🔧 技术方法**

技术方案包括基于学习演示的视觉运动扩散策略（diffusion policy）实现抓取、头部手势识别用于工具激活/释放、音频反馈提示状态变化，以及在手部上采用多模态感知与控制；

**📊 数据集**

数据集为针对喷雾瓶、电钻、红外温度计、打火机和冰淇淋勺等五种工具收集的50次演示（共250条），随后使用SAM2将绿幕背景替换为HMDB51视频或ImageNet图像生成三种训练模型（A、B、C）；

**📈 对比分析**

通过44名受试者进行三轮试验，测量完成时间、任务成功率和主观评价，结果显示完成时间从第1轮的306.1 s下降至第3轮的234.8 s（23.3%提升，p<0.001），最高成功率为93.6%（模型C），并且接受度、易用性等指标均处于良好水平；

**⚠️ 局限性**

实验局限包括仅测试健康双手受试者、实验时间短、任务和对象有限、硬件设计与头部手势识别的舒适度与可靠性待改进，以及缺乏对上肢功能受损人群的实际评估。

---

## 372. The Condition for Structured Coding to Improve Random Coding in the Binary Modulo-sum Problem

**arXiv ID:** 2606.25570 | [PDF](https://arxiv.org/pdf/2606.25570v1)

**作者:** Yohsuke Tsujino `[一作]`, Shun Watanabe `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本研究分析了多字母AH编码在模和问题中相对于SW编码的改进条件。

**💡 创新点**

通过应用类型方法，明确了多字母AH编码在特定条件下能够优于SW编码的情况，这是之前仅通过数值计算观察到的改进区域。

**🔧 技术方法**

使用了类型方法来比较多字母AH编码和SW编码的速率差异，并通过比较散度指数来简化分析。

**📊 数据集**

研究中使用了相关的二元源分布，具体的概率分布参数未在摘要中详细列出。

**📈 对比分析**

通过数值计算，Kakishima和Watanabe展示了多字母AH编码在某些源参数下优于SW编码的可能性，但本研究提供了分析证明，表明在特定条件下多字母AH编码的和速率可以严格小于SW编码的和速率。

**⚠️ 局限性**

本研究的局限性在于所选择的辅助随机变量的构造可能不是最优的，未来的研究可以探索更好的辅助随机变量以进一步提高可达和速率。

---

## 373. Croc: Training the Next Generation Chip Designers on Domain-Specific End-to-End Open Source Silicon

**arXiv ID:** 2606.25673 | [PDF](https://arxiv.org/pdf/2606.25673v1)

**作者:** Enrico Zelioli `[一作]` (ETH Zurich), Luca Benini `[通讯]` (ETH Zurich)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于Croc平台的完整开源域特定SoC设计与制造流程，并在ETH Zurich VLSI课程中实现了学生版芯片的设计与打样。

**💡 创新点**

首次将开源EDA工具、开放PDK和模块化RISC‑V SoC结合，用可扩展的教学框架实现了从RTL到GDSII的全流程教育与实际芯片制程。

**🔧 技术方法**

使用Yosys进行综合、OpenRoad进行物理设计、Verilator进行RTL仿真，配合IHP 130 nm开放PDK，支持ISA扩展、加速器和外设定制。

**📊 数据集**

无传统数据集，主要使用学生设计的SoC实例作为实验数据。

**📈 对比分析**

通过对比典型闭源工具实现的同等功能芯片，测得基准芯片频率82 MHz，功耗52.3 mW，资源利用率和布局密度与闭源实现相当。

**⚠️ 局限性**

受制于制造资金与技术节点，最终仅5颗芯片完成打样，且依赖130 nm技术，尚缺乏更高性能节点的验证。

---

## 374. SA-LIVO: Efficient LiDAR-Inertial-Visual Odometry with Subspace-Aware Degeneracy Handling

**arXiv ID:** 2606.25699 | [PDF](https://arxiv.org/pdf/2606.25699v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 375. Dynamic Load Balancing for Uncertainty Quantification with Applications in Bayesian Inversion

**arXiv ID:** 2606.25693 | [PDF](https://arxiv.org/pdf/2606.25693v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 376. The MDS or NMDS for Modified GRS codes with flexible hull dimensions and lengths

**arXiv ID:** 2606.25662 | [PDF](https://arxiv.org/pdf/2606.25662v1)

**作者:** Zhonghao Liang `[一作]` (Sichuan Normal University), Xiaoping Li `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究并分析了 MGRS 代码及其扩展代码的性质，证明其要么为 MDS 要么为 NMDS，并给出必要与充分条件；完成了特殊 NMDS MGRS 与 EMGRS 代码的权重分布；构造了四类 LCD 或一维 Euclidean hull 的 MGRS 代码，并构造了可变 Hermitian hull 维度与长度的 MGRS 代码；证明 NMDS MGRS 代码与椭圆曲线 AG 代码线性不等价。

**💡 创新点**

首次证明两类 MGRS 与 EMGRS 代码的 MDS/ NMDS 判定、权重分布以及灵活 Hermitian hull 维度构造，并通过 Schur 乘积证实其与椭圆曲线 AG 代码线性不等价。

**🔧 技术方法**

运用了有限域上的多项式与对称多项式理论、子集构造、Hermitian 与 Euclidean hull 维度计算、Schur 乘积与 AG 代码平方维度定理、以及线性码等价判定技术。

**📊 数据集**

使用的主要数据集为有限域 𝔽_q 与其平方域 𝔽_q^2，结合具体素数幂与子集构造实现代码构造与验证。

**📈 对比分析**

通过 Magma 软件验证权重枚举与参数一致性；与传统 MDS 代码、AG 代码在权重分布和 hull 维度上对比，证明 NMDS MGRS 代码在相同参数下具有优越或相似的最小距离与重量分布特性。

**⚠️ 局限性**

局限性包括：只给出两类 MGRS/EMGRS 代码的特殊权重分布，未覆盖全部参数空间；构造的 Hermitian hull 维度可调性仍受有限域大小约束；未对大规模实例进行性能评估或实验验证。

---

## 377. Learning to Adapt: Reptile-D-Learning for Robust and Efficient Control Under Parametric Uncertainty

**arXiv ID:** 2606.25659 | [PDF](https://arxiv.org/pdf/2606.25659v1)

**作者:** Haipeng Cao `[一作]` (Beihang University), Quan Quan `[通讯]` (Beihang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Reptile-D-learning 框架，联合元学习 Lyapunov 网络、D‑函数网络与控制策略，以在参数不确定的非线性系统上实现快速自适应控制并保证 Lyapunov 稳定性。

**💡 创新点**

创新点在于将 Reptile 作为 D‑learning 的第一阶元优化器，避免高阶 Hessian 计算，利用跨参数共享动力学结构实现快速适应，同时给出梯度一致性与动态分解的理论分析。

**🔧 技术方法**

使用技术包括 D‑learning（模型无关 Lyapunov 导数估计）、Reptile 元学习、bilevel 优化、梯度一致性分析、以及深度神经网络实现 Lyapunov、D‑函数与策略。

**📊 数据集**

实验数据集来自三种非线性控制基准：倒立摆、单轨车（CommonRoad benchmark）和 UAV（Crazyflie 四旋翼），通过对参数进行随机化采样并在 OOD（超出训练分布）场景下评估。

**📈 对比分析**

与基线 LQR 与标准 D‑learning 进行对比，结果显示 Reptile‑D‑learning 在任务内外参数变化下显著降低收敛步骤、提升稳定成功率、减小跟踪误差，并在少量适配步骤内即可匹配或超过专用模型的性能。

**⚠️ 局限性**

局限性包括对任务分布采样的依赖、对极端参数漂移或全新动力学结构的适应性不足，以及理论分析中对梯度一致性假设的严格要求；在大规模系统上仍受内存和计算资源限制。

---

## 378. Is GraphRAG Needed? From Basic RAG to Graph-/Agentic Solutions with Context Optimization

**arXiv ID:** 2606.25656 | [PDF](https://arxiv.org/pdf/2606.25656v1)

**作者:** Long Chen `[一作]` (Amazon Web Services), Vidya Sagar Ravipati `[通讯]` (Amazon Web Services)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了针对半结构化知识库的多种RAG（检索增强生成）架构的系统评估框架，包含传统RAG、GraphRAG、Modular RAG和Agentic RAG，并在9个标准化场景中进行实验；

**💡 创新点**

1) 提出了针对GraphRAG和Agentic RAG的上下文优化方法，显著减少Token使用（19%–53%），并通过批量检索降低LLM轮询次数；2) 通过实验揭示了检索-生成缺口（retrieval‑generation gap），表明扩展检索并未按比例提升生成质量；3) 对不同RAG架构在半结构化知识库上的表现进行了系统对比，指出Agentic RAG最具优势；

**🔧 技术方法**

Graph查询、向量检索、知识图谱构建与检索、LLM驱动的查询重写与重排序、Agentic工具调用、上下文去重与批量检索、LlamaIndex、Amazon Neptune、Amazon Bedrock（Claude 3.7 Sonnet）等技术；

**📊 数据集**

STaRK‑Prime（精确医学领域的129K实体、8.1M关系），以及PrimeKG作为预定义知识图谱；

**📈 对比分析**

使用Hit@1、Hit@5、Recall@20和MRR等端到端生成质量指标对比；结果显示：传统RAG+关系文档提升显著；GraphRAG在预定义KG与向量检索混合时表现最佳；Modular RAG+重排提高；Agentic RAG在最小工具下性能最高（Hit@1≈0.688, Hit@5≈0.844）；上下文优化后Token下降同时性能保持或略升；但检索覆盖提升并未显著提高生成质量，揭示检索-生成缺口；

**⚠️ 局限性**

仅使用单一数据集（STaRK‑Prime），实验受限于该领域；所有实验均基于单一LLM（Claude 3.7 Sonnet），不同模型性能可能不同；评价指标局限于检索相关性，未覆盖事实性、可信度等维度；Agentic RAG的非确定性和参数未完全调优；

---

## 379. Bridging Predictions and Interventions: An Integrated Framework for Automated Decision-Systems

**arXiv ID:** 2606.25668 | [PDF](https://arxiv.org/pdf/2606.25668v1)

**作者:** Inioluwa Deborah Raji `[一作]`, Ashia Wilson `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个整合预测、评估与实施的视角框架，重新定义了自动决策系统（ADS）的生命周期，并强调在组织决策流程中将ADS视为干预工具而非单纯的预测工具。

**💡 创新点**

创新点在于：①将ADS的设计、评估与实施拆分为三个互联阶段；②引入政策变更（Z）、评估指标（R、Ŷ、D、Y）的概念，明确ADS对决策与结果的中介作用；③强调因果推断方法在ADS评估中的必要性，超越传统的预测准确率评估；④呼吁从单一技术视角转向多学科交叉视角（人机交互、程序评估、组织行为）。

**🔧 技术方法**

主要采用理论分析和概念建模，引用因果推断、政策评估、组织行为学以及人机交互等领域的现有技术与方法，未提出新的算法实现。

**📊 数据集**

该论文为视角性综述性工作，未使用具体数据集进行实验或实证分析。

**📈 对比分析**

论文未进行实验或比较；评价方法主要以框架性讨论为主，未给出具体性能指标。

**⚠️ 局限性**

局限性包括：①缺乏实证验证，框架的有效性仍待后续案例研究检验；②对数据收集与隐私、可解释性等实际操作细节未给出具体解决方案；③在多机构、多模型多利益相关者的复杂环境下，如何实施该框架仍存在不确定性。

---

## 380. MedGuards: Multi-Agent System for Reliable Medical Error Detection and Correction

**arXiv ID:** 2606.25651 | [PDF](https://arxiv.org/pdf/2606.25651v1)

**作者:** Congbo Ma `[一作]` (New York University), Farah E. Shamout `[通讯]` (New York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为 MedGuards 的多智能体框架，用于在 LLM 生成的医学文本中检测、定位并纠正错误，并提出了 Keyword-Prioritized Correction Score (KPCS) 评价指标。

**💡 创新点**

创新点在于：1) 将错误检测、定位与纠正拆解为专门化的智能体，采用多智能体自一致性与信心引导的仲裁机制；2) 通过链式思考（CoT）实现任务分解；3) 将仲裁过程视为提示学习（ICL）任务；4) 引入 KPCS 强调关键医学实体的正确性，提升评价对临床安全的关注。

**🔧 技术方法**

技术手段包括：多智能体自一致性 (self-consistency)、基于提示学习的仲裁 (ICL)、链式思考 (CoT) 的任务分解、字符级相似度对齐、以及 KPCS 评价指标的三步计算。

**📊 数据集**

实验使用了公开的 MEDEC 与 MedErrBench 数据集，包含英文、阿拉伯文和中文三种语言，覆盖多种医学场景。

**📈 对比分析**

与现有错误检测/纠正基线（如 knowlab_AIMed、Medifact 等）及多款 LLM（Gemini、GPT‑4o‑mini、Doubao‑1.5‑thinking‑pro、Deepseek‑V3 等）对比，MedGuards 在检测、定位和生成质量（ROUGE‑1、BERTScore、BLEURT、KPCS）上均实现了显著提升，KPCS 得分提升幅度高达 35–66%，并在所有语言与数据集上保持一致性。

**⚠️ 局限性**

局限性包括：1) 仍受基础 LLM 的生成质量限制，无法完全消除低质量输出；2) 多智能体和仲裁过程增加推理时延与算力消耗；3) KPCS 依赖手工标注的关键字集，可能不适用于所有医学子领域；4) 评价指标侧重关键实体，可能忽略其他临床错误类型。

---

## 381. VPA-Guard: Defending and Benchmarking Image-to-Video Generation Against Visual Prompt Attacks

**arXiv ID:** 2606.25592 | [PDF](https://arxiv.org/pdf/2606.25592v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 382. 1000 Rallies: An Event-Camera Dataset and Real-Time Learned Ball-State Estimation for Robotic Table Tennis

**arXiv ID:** 2606.25620 | [PDF](https://arxiv.org/pdf/2606.25620v1)

**作者:** Raphaela Kreiser `[一作]` (Sony AI), Naoya Takahashi `[通讯]` (Sony AI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文首先构建了首个规模宏大的乒乓球事件相机数据集，并设计了一套从事件流到三维球状态估计再到机器人控制的完整实时闭环系统。

**💡 创新点**

创新点在于：① 用CNN同时预测球的图像位置和速度，并把速度作为EKF的额外观测显著提升轨迹预测；② 首次实现事件相机驱动的机器人乒乓球回合，完成了感知-规划-执行的完整闭环。

**🔧 技术方法**

使用技术包括事件相机（Prophesee EVK4）、改进的YOLOv4-tiny网络、基于物理模型的EKF状态估计、Ruckig轨迹规划、以及FP16 TensorRT并行推理实现400 Hz更新率。

**📊 数据集**

所用数据集为自建的5 h 10 min、约1200局、14摄像头同步的数据集，涵盖业余到精英选手、球速最高26 m/s，且配有1 kHz伪真实标注。

**📈 对比分析**

与传统事件检测和帧基方法相比，检测误差0.91 px、IoU0.78，更新率400 Hz；EKF加入速度后弹跳预测误差从12.1 cm降至7.7 cm（36%提升），机器人回球率分别为100%（弹射器）和75%（人机对战）。

**⚠️ 局限性**

局限性在于：仍受室内照明和摄像头布置限制，对极端高速或复杂遮挡的鲁棒性尚不足；系统仍以GPU并行推理为核心，未完全利用事件相机原生的异步低延迟特性。

---

## 383. Shift Variant Image Degradation and Restoration Using Singular Value Decomposition

**arXiv ID:** 2606.25818 | [PDF](https://arxiv.org/pdf/2606.25818v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 384. Multi-Source Reachability in Near-Optimal Time

**arXiv ID:** 2606.25612 | [PDF](https://arxiv.org/pdf/2606.25612v1)

**作者:** Shimon Kogan `[一作]` (Weizmann Institute), Merav Parter `[通讯]` (Weizmann Institute)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个近似最优的确定性算法，用于在有向图中从给定源集合计算可达集合并返回前驱矩阵。

**💡 创新点**

算法在时间复杂度上与矩阵乘法指数紧密匹配，突破了传统 BFS/DFS 或全转移闭包的上限，且实现了确定性且可达集计算近线性时间；对源集合大小为 n^σ 的情况达到 Õ(n^{ω(σ)}) 复杂度；同时给出了前驱矩阵的高效构造。

**🔧 技术方法**

利用矩形矩阵乘法、矩阵乘法的 witness 矩阵技术、图的拓扑排序、SCC 合并、递归分治策略以及矩阵乘法的时间下界推导。

**📊 数据集**

该工作为理论算法，无实验数据集，主要在理论分析中验证算法正确性和复杂度。

**📈 对比分析**

与现有最优的随机化 n^{1+2/3 ω(σ)} 算法相比，作者的确定性算法在密集图中可支持多达 n^{0.32} 个源节点，达到近线性时间；性能上在理论上优于随机化方案，且在极端稠密场景下明显更快。

**⚠️ 局限性**

算法的实际效率高度依赖矩阵乘法的实现与 ω(σ) 的取值；在稀疏图上并未突破传统 BFS/DFS 的复杂度；并且需要先将一般有向图转化为 DAG，涉及 SCC 合并与前驱矩阵合并的额外 O(n^2) 步骤。

---

## 385. ROAD-VLA: Robust Online Adaptation via Self-Distillation for Vision-Language-Action Models

**arXiv ID:** 2606.25800 | [PDF](https://arxiv.org/pdf/2606.25800v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 386. An Approach for a Supporting Multi-LLM System for Automated Certification Based on the German IT-Grundschutz

**arXiv ID:** 2606.25608 | [PDF](https://arxiv.org/pdf/2606.25608v1)

**作者:** Lea Roxanne Muth `[一作]` (Freie Universität Berlin), Marian Margraf `[通讯]` (Freie Universität Berlin)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

做了什么：设计并实现了一个基于多大语言模型与混合检索增强生成（HybridRAG）的多模型专家体系（Multi‑LLM System），用于半自动化完成BSI IT‑Grundschutz标准下的安全概念认证流程。

**💡 创新点**

创新点是什么：首次将HybridRAG与知识图谱相结合，构建任务专属LLM专家与协调器，显著降低人工干预、错误率与幻觉，并在每个认证阶段实现可插拔的模块化自动化。

**🔧 技术方法**

用了什么技术：采用GPT‑4o‑mini（或等价本地模型）与text‑embedding‑3‑small、Neo4j知识图谱与Hybrid Retriever、LangChain的Graph Transformer、BERTScore、Hallucination Score等多技术组合。

**📊 数据集**

用了什么数据集：以BSI公开的IT‑Grundschutz Compendium（PDF/结构化文档）和虚拟公司案例Recplast GmbH为基准构建KG，后续计划引入真实公司内部安全文档。

**📈 对比分析**

如何比较的方法，性能怎么样：通过检索成功率、实体关系提取准确率、模块匹配Top‑1/3/5精度等指标评估，实验表明信息提取准确率约92%‑95%，Top‑5模块匹配准确率达86%。

**⚠️ 局限性**

limitation是什么：仍需进一步微调与本地化部署，无法完全替代人类专家，模型存在自我偏见与幻觉风险，且完整流程验证与真实企业数据应用尚未完成。

---

## 387. Constraint Tax in Open-Weight LLMs: An Empirical Study of Tool Calling Suppression Under Structured Output Constraints

**arXiv ID:** 2606.25605 | [PDF](https://arxiv.org/pdf/2606.25605v1)

**作者:** Fangzheng Li `[一作]` (Focus Technology Co., Ltd.), Chen Lv `[通讯]` (Focus Technology Co., Ltd.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对多模型、多框架、多规模的开源LLM在工具调用与结构化输出同时启用的生产环境下的实验，首次系统性复现并描述了“工具抑制”现象，即模型在满足JSON Schema的前提下完全停止工具调用；

**💡 创新点**

创新点在于：①提出并量化了工具抑制（Tool Suppression）及其行为分类；②通过推理栈追踪定位到语法约束（token mask）是导致抑制的根本原因；③提出并验证了透明双通道执行（Transparent Two‑Pass Execution）这一实用的推理层缓解方案；④引入Constraint Priority Inversion（CPI）假说，对行为机制给出解释。

**🔧 技术方法**

技术包括：JSON Schema到有限状态机的编译与词汇表掩码、SGLang/vLLM的语法约束解码、两阶段推理流程（先工具调用后结构化生成）、实验框架对工具调用事件与JSON合规性实时解析。

**📊 数据集**

实验使用了包含5类业务场景（买家背景、公司验证、市场情报、产品知识、合规调查）的任务集，工具调用为搜索/查询工具；模型涵盖7个开源大模型（Qwen系列、GPT‑OSS、Nemotron等）和一个闭源参考模型；没有公开标注数据集，而是自构造任务与工具定义。

**📈 对比分析**

比较方法采用Tool Invocation Rate（TIR）、Suppression Rate（SR）、JSON Compliance Rate、End‑to‑End Success Rate等指标，三种实验条件（工具仅、结构仅、两者并用）对比；在所有开源模型下TIR在工具仅时100%，在两者并用时跌至0%，SR最高达100%；透明双通道执行后TIR恢复至100%，并发出完整结构化回答，显示该方案在保持结构合规的同时完全恢复工具调用。

**⚠️ 局限性**

局限性包括：仅测试有限模型与任务场景，未覆盖所有工具调用格式；推理栈分析仅在SGLang中完成，对其他框架的语法实现细节未知；CPI仅为行为假说，未通过内部模型分析验证；双通道方案增加推理轮数与延时，未评估在高并发场景下的成本与可扩展性。

---

## 388. Designing Trustworthy LLM-based Wellbeing Recommendation through Controllable Interaction

**arXiv ID:** 2606.25809 | [PDF](https://arxiv.org/pdf/2606.25809v1)

**作者:** Alan Said `[一作]` (University of Gothenburg), Alexandra Weilenmann `[通讯]` (University of Gothenburg)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于交互约束的LLM健康推荐框架，强调在健康干预中将推荐视为交互设计而非纯粹预测。

**💡 创新点**

创新点在于将推荐的表达方式、直接性、解释风格和责任框架等交互属性显式化为可配置的约束，从而实现可控、透明、可评估的LLM推荐系统。

**🔧 技术方法**

采用大型语言模型（LLM）生成候选响应，并在交互层通过约束策略对输出进行筛选和修改；架构包含生成层、交互策略层和用户建模层。

**📊 数据集**

论文未使用具体数据集，而是基于已有健康推荐系统和人机交互理论构建概念模型，强调未来可在真实健康干预场景中使用用户交互日志和问卷数据进行验证。

**📈 对比分析**

由于是理论框架，未给出定量实验；作者建议通过对比不同交互约束组合（如直接性高 vs 低、责任框架系统中心 vs 用户中心）在实验室或真实环境中测量自我效能、信任感、适当依赖等用户中心指标，以评估各配置的效果。

**⚠️ 局限性**

局限性包括：缺乏实证验证与量化结果；交互约束的具体参数化和动态调整尚未定义；在多样化用户群体和长期干预中的可扩展性和稳健性待进一步研究。

---

## 389. Tracing Target Answers in Poisoned Retrieval Corpora via Token Influence Attribution

**arXiv ID:** 2606.25721 | [PDF](https://arxiv.org/pdf/2606.25721v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 390. NEURON-Fabric: Architecture-Runtime Co-Design for Controlled Low-Bit Gradient Communication

**arXiv ID:** 2606.25759 | [PDF](https://arxiv.org/pdf/2606.25759v1)

**作者:** Ziqiang Wang `[一作]` (Carleton University), Chung-Horng Lung `[通讯]` (Carleton University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 NEURON-Fabric 体系结构，实现了低比特梯度聚合的可控运行时路径，动态决定何时采用低比特、何时回退到 FP32，并结合模型绑定与 reducer 容量门控。

**💡 创新点**

创新点在于将低比特聚合从静态压缩改为运行时可控策略，设计了校准到运行时接口、混合桶路由、基于健康监测的 Supervisor 以及端点容量门控，形成完整的系统‑运行时协同设计。

**🔧 技术方法**

使用了离线校准生成操作配置、Commander/ Supervisor 控制循环、PyTorch DDP 钩子、G‑Binary/G‑Ternary 低比特格式、CUSUM 变化检测、端点容量模型、分层 sign‑count reducer 等技术。

**📊 数据集**

数据集包括 CIFAR‑100/ResNet‑18、SST‑2/DistilBERT 与 BERT‑Base、以及 WikiText‑103 上的 Pythia‑1.4B 与 Pythia‑6.9B 语言模型。

**📈 对比分析**

与 FP32 基线和静态低比特方案对比，保持了接近 FP32 的最终精度（例如 CIFAR‑100 70%+、SST‑2 87.9% 以及 Pythia‑6.9B 18.86 perplexity），同时将梯度通信量分别降低到 0.735x、0.824x 与 0.743x，且无路由正确性违规。

**⚠️ 局限性**

局限性包括：需要手工生成绑定和预检；缺乏零样本通用低比特配置；在线控制仅对健康恢复有限，缺乏阶段感知；未在生产级 reducer 上测定实际速度提升；对优化器、批量大小、拓扑等覆盖范围有限。

---

## 391. Frequency-Aware Self-Supervised Music Representation Learning

**arXiv ID:** 2606.25713 | [PDF](https://arxiv.org/pdf/2606.25713v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 392. Stage-Aware and Roughness-Constrained Diffusion Policy for Multi-Stage Robotic Polishing

**arXiv ID:** 2606.25754 | [PDF](https://arxiv.org/pdf/2606.25754v1)

**作者:** Shuai Ke `[一作]` (Huazhong University of Science and Technology), Han Ding `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种阶段感知与粗糙度约束扩散策略（SRDP），用于多阶段机器人抛光任务的动作生成与过程参数调控。

**💡 创新点**

创新点包括：① 在多模态观测历史中推断过程阶段并将其编码为条件，驱动共享的反向扩散网络生成阶段一致的动作；② 将粗糙度经验模型与物理可行性约束直接嵌入扩散采样过程，实时调节进给速度与法向力；③ 通过阶段转移一致性正则化实现无外部阶段标签的在线阶段识别与转换。

**🔧 技术方法**

采用条件扩散政策（Diffusion Policy）、多模态阶段推理网络（MLP+注意力），阶段转移一致性正则化，粗糙度经验模型（对数线性功率律），以及约束引导扩散采样（DDIM/DDPM）。

**📊 数据集**

使用由操作员手动演示收集的多阶段抛光数据集，包含RGB图像、机器人位姿、法向力、工具模式以及对舱壁抛光（5个阶段）和内腔抛光（4个阶段）的阶段标签。

**📈 对比分析**

与Diffusion Policy、Hierarchical Imitation Learning、Adaptive Compliance Policy进行对比。SRDP在舱壁抛光的子任务成功率与阶段过渡成功率分别达0.94/0.88，内腔抛光分别达0.91/0.83，均显著优于对照方法，并在粗糙度、铰刀宽度等加工质量指标上取得更低的均值与更小的波动。

**⚠️ 局限性**

局限性包括：需要先验的主轴转速设定与粗糙度模型离线标定；对极端动态或未知阶段的自适应性有限；对演示数据质量与规模敏感；以及硬件延迟和采样步数对实时性能的潜在影响。

---

## 393. Segment Watchman Routes

**arXiv ID:** 2606.25816 | [PDF](https://arxiv.org/pdf/2606.25816v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

---

## 394. Beyond Function Calling: Benchmarking Tool-Using Agents under Tool-Environment Unreliability

**arXiv ID:** 2606.25819 | [PDF](https://arxiv.org/pdf/2606.25819v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 395. MIL-LC: A Robust Magnetometer-Inertial-LiDAR Fusion Multimodal Localization Framework

**arXiv ID:** 2606.25796 | [PDF](https://arxiv.org/pdf/2606.25796v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 396. Black-Box Assisted Regression: Phase Transitions and Minimax Optimality

**arXiv ID:** 2606.25743 | [PDF](https://arxiv.org/pdf/2606.25743v1)

**作者:** Yan Zhou `[一作]` `[通讯]` (Changsha University of Science and Technology), Yan Zhou (Changsha University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在只能查询固定预训练黑盒预测器且仅有少量标记样本的情形下，提出安全残差估计器以自适应利用黑盒先验和样本信息，避免负迁移。

**💡 创新点**

揭示了残差误差与样本量的阶跃转移点 δ_c(n)≍n^{-β/(2β+d)}，并证明安全残差估计器可匹配该极限，且给出了无负迁移的解析安全保证。

**🔧 技术方法**

基于非参数回归理论、Hölder 类光滑性假设、分层样本划分与验证选择、残差学习（如核回归、神经残差头）以及零初始化和安全阈值机制。

**📊 数据集**

在人工合成 1D 与高维（d=20）回归数据上检验理论转移；在视觉任务 CIFAR‑100 使用 CLIP 作为黑盒；在 NLP 任务 AG News 使用 Qwen3‑8B 作为黑盒。

**📈 对比分析**

与零射/仅黑盒、完整无监督学习（Scratch）、权重线性混合、特征拼接等基线对比，安全残差估计器在少样本区间始终保持或超过零射基线，无负迁移，且在 2000 样本时可提升约 2–3% 准确率，显著优于加权混合和拼接方法。

**⚠️ 局限性**

仅适用于平方损失回归、同方差 Gaussian 噪声、静态黑盒；对极端噪声、重尾分布或在线学习场景缺乏理论保证，且验证分割引入额外的 1/n 选择成本。

---

## 397. What Does the Brain See? Multiview Neural Representations to Demystify the Brain-Visual Alignment

**arXiv ID:** 2606.25718 | [PDF](https://arxiv.org/pdf/2606.25718v1)

**作者:** Salini Yadav `[一作]` (Indian Institute of Technology Roorkee), Partha Pratim Roy `[通讯]` (Indian Institute of Technology Dhanbad)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

设计并训练了一种统一的多视图 EEG 编码器，将脑电信号映射到预训练的视觉语义空间，实现零样本视觉语义解码。

**💡 创新点**

创新点在于同时建模时域（选择性状态空间模型）、频域（可学习小波分解）和空间（注意力加权图卷积）三种视图，并通过对比学习和 EEG 特定正则化提升语义对齐和跨受试者/跨会话泛化。

**🔧 技术方法**

使用了选择性状态空间模型（S³M）、自适应可学习小波频谱分解、注意力加权图神经网络、对比学习损失、RCR 和 CPCA 正则化，以及 CLIP 图像编码器。

**📊 数据集**

在 THINGS‑EEG 大规模 EEG‑视觉数据集上进行实验。

**📈 对比分析**

与 BraVL、NICE 系列、ATM‑S、UMind、CC、UBP 等方法对比，在 within‑subject、cross‑subject、cross‑session 三种零样本设置下分别取得 54.8%/85.6%、15.3%/45.4%、40.8%/78.0% 的 Top‑1/Top‑5 准确率，显著优于前沿方法。

**⚠️ 局限性**

局限在于仍受 EEG 信号噪声、非平稳性和空间分辨率限制；方法对预训练视觉嵌入依赖较大，跨更广泛人群或任务的泛化仍需进一步验证；多视图结构导致模型复杂度和训练成本上升。

---

## 398. Cellular Predictions on the Move: What about Data?

**arXiv ID:** 2606.25709 | [PDF](https://arxiv.org/pdf/2606.25709v1)

**作者:** Natalia Vesselinova `[一作]`, Pauliina Ilmonen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在高速公路场景下，利用车辆流量和平均速度等人口动力学信息，结合LSTM模型实现短期移动通信负载预测，提升预测准确性。

**💡 创新点**

首次将车辆流量与速度等内在生成过程数据引入移动网络负载预测，证明基于人口动力学的数据能显著提升模型性能，减少对传统外部静态数据的依赖。

**🔧 技术方法**

采用经典LSTM层+全连接输出层的深度学习框架，输入特征包括历史呼叫量与车辆流/速度时间序列，训练优化使用均方误差(MSE)与RMSProp。

**📊 数据集**

使用加利福尼亚州US50‑E高速公路的Caltrans Loop Detector（PeMS）流量和速度数据，结合根据Poisson过程模拟的呼叫量（单/多泊松/指数分布）共24周的时序数据。

**📈 对比分析**

将仅基于呼叫量的基线模型与加入流/速度特征的异构模型进行对比，评估指标为MAE、MSE、MAPE、RMSE；实验显示异构模型平均误差下降幅度可达8–60%，最高约为RMSE降低至1/3。

**⚠️ 局限性**

局限性包括：1) 仅在高速公路场景验证，城市环境可能表现不同；2) 呼叫负载为模拟，真实运营数据未验证；3) 对道路测量误差敏感，需保证流速数据准确；4) 需要与道路监测系统协同部署，增加系统复杂度。

---

## 399. OncoSynth: Synthetic data generation for treatment effect estimation in oncology

**arXiv ID:** 2606.25762 | [PDF](https://arxiv.org/pdf/2606.25762v1)

**作者:** Octavia-Andreea Ciora `[一作]` (LMU Munich), Stefan Feuerriegel `[通讯]` (Munich Center for Machine Learning)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了OncoSynth，一种可生成保持因果关系的合成肿瘤患者数据集，用于精准肿瘤学中治疗效果的可靠估计。

**💡 创新点**

通过把数据生成过程分解为“患者特征 → 治疗分配 → 生存结局”三步，并在每一步使用合适的模型，避免信息泄漏，首次实现了既保留统计特征又保持因果机制的合成数据；在治疗效果评估上的表现显著优于现有方法。

**🔧 技术方法**

核心技术包括：TabDiff扩散模型生成患者特征、逻辑回归+Isotonic校准预测治疗分配、T-learner框架下的随机生存森林（RSF）生成时间‑事件结局；评估使用JSD、PEHE、AUQC等因果机器学习指标。

**📊 数据集**

在两大SEER数据库队列上验证：肺癌（约37,000例）和乳腺癌（约17,000例）成人患者的真实临床数据。

**📈 对比分析**

与CTGAN（GAN）和TabDiff（单一扩散）对比，OncoSynth在统计保真度指标（Δ_X, Δ_X², Δ_W, Δ_C, JSD_T, Δ_RMST）和治疗效果指标（Δ_ATE, PEHE, AUQC）均表现最好；治疗效果误差可分别减少约66%（总体）和58%（个体），且AUC、Qini曲线等指标明显提升。

**⚠️ 局限性**

局限包括：依赖原始数据的完整性和无偏性，可能复制原始偏差；只针对生存结局评估，未覆盖其他临床终点；验证仅基于SEER数据，跨机构泛化需进一步测试；仍需额外隐私保护措施以满足法规要求。

---

## 400. Dual Distribution Estimation for Zero-shot Noisy Test-Time Adaptation with VLMs

**arXiv ID:** 2606.25758 | [PDF](https://arxiv.org/pdf/2606.25758v1)

**作者:** Wenjie Zhu `[一作]` (Hong Kong Polytechnic University), Lei Zhang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种零训练、零样本的Dual Distribution Estimation (DDE) 框架，用于在测试时无噪声适配中同时过滤噪声样本并提升 ID 分类准确率。

**💡 创新点**

创新点包括：①正样本特征双高斯分布估计（PFDE），通过包含与排除分布形成对比度得分提升 ID 识别；②负标签分布估计（NLDE），筛选高判别负标签以强化 OOD 检测；③在在线自适应中使用自适应阈值而非固定阈值。

**🔧 技术方法**

技术手段包括：零训练的 Gaussian Discriminant Analysis、训练自由的 CLIP 文本‑图像对齐、双高斯模型、负标签分布挖掘、EM 更新、动态权重 α_t、β、缓存机制及自适应阈值。

**📊 数据集**

使用的主要数据集为 ImageNet‑1K 及其变体（ImageNet‑S、A、V2、R）和细粒度识别数据集（CUB、Stanford Cars、Food‑101、Oxford‑IIIT Pet），OOD 样本来自 iNaturalist、SUN、Texture、Places 等。

**📈 对比分析**

与多种基线（AdaND、AdaNeg、DMN、TPT、Tent 等）对比，DDE 在 ImageNet 及其变体上取得最高谐波平均准确率（↑3.7%），OOD 检测 AUROC 97.89% 并将 FPR95 降至 9.8%，在细粒度数据集上亦表现最优。

**⚠️ 局限性**

局限性主要是为每个类别维护双高斯参数会在极大规模数据集上略增内存开销，且对超参数（如阈值、缓存大小、负标签数）仍需调优。

---

## 401. OPERA: Aligning Open-Ended Reasoning via Objective Perplexity-based Reinforcement Learning

**arXiv ID:** 2606.25757 | [PDF](https://arxiv.org/pdf/2606.25757v1)

**作者:** Wenxuan Jiang `[一作]` (Hong Kong Polytechnic University), Ninghao Liu `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于困惑度的自我反思奖励框架 OPERA，用内在奖励替代外部 LLM 评判，对开创性任务进行强化学习。

**💡 创新点**

创新点在于使用困惑度动态衡量自我反思步骤的价值，并结合迭代困惑度引导轨迹合成生成高质量推理轨迹，同时采用自反奖励与组相对困惑度奖励的混合方案。

**🔧 技术方法**

技术包括 RLVR、Perplexity‑Guided Iterative Trace Synthesis、Self‑Reflection Reward、In‑Group Relative Perplexity Reward、Hybrid Reward、GRPO、SFT 与 RL 等。

**📊 数据集**

使用 20,000 条自生成推理轨迹（来源于 LongWriter‑6k、WildChat、LitBench‑Train、OpenThought）以及 AlignBench、HelloBench、EQ‑Bench creative writing、WritingBench、MATH500 等基准数据集。

**📈 对比分析**

与 GPT‑4o、Gemini‑2.5、MiniMax‑M2.5 等专有模型以及 LongWriter‑8B、DeepWriter‑8B、LongWriter‑Zero‑32B 等开源模型对比，OPERA 在 Qwen3‑8B 上在创意写作 V3、HelloBench 等任务均超过专有模型，平均提升约 120%，且在数学任务上保持不减。

**⚠️ 局限性**

局限性：评估仍依赖高容量 LLM 判别器，可能带来偏见；目前仅针对写作任务，尚未扩展到问答或多轮对话；困惑度作为奖励的有效性在不同任务场景下仍需进一步验证。

---

## 402. Position Spaces and Graphs

**arXiv ID:** 2606.25719 | [PDF](https://arxiv.org/pdf/2606.25719v1)

**作者:** Rita-Nathalia Assaf `[一作]` (University Angers), Frédéric Saubion `[通讯]` (University Angers)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了位置空间（Position Space）和位置图（Position Graph）两种结构，用来通过两条严格偏序（水平和垂直）对离散符号进行定位，并给出了一致性判定的理论描述与线性时间实现；同时证明即使在这一受限结构下，诱导子图同构仍为 NP‑完整。

**💡 创新点**

创新点在于：①引入了链（no‑branching）和兼容性约束，将二维行列布局抽象为两条独立偏序；②给出了仅依赖于禁止混合标签环的精确一致性判定；③展示了在一致性判定后的行列嵌入可在线性时间完成；④证明位置图诱导子图同构的 NP‑完整性，划定了理论可行性边界。

**🔧 技术方法**

主要技术包括：严格偏序的图表示（有向无环图）、链条件与兼容性判定、基于禁止混合标签环的图一致性判定、拓扑排序与行列编号的线性时间嵌入算法、以及利用 3SAT 归约构造证明 NP‑完整性。

**📊 数据集**

本文未使用任何公开数据集，所有实验和证明均为理论性推导与图构造。

**📈 对比分析**

由于缺乏实验评估，本文没有与其它方法进行性能比较；理论上，一致性判定与嵌入可在 O(|V|+|E|) 时间内完成，而诱导子图同构问题在此框架下仍保持 NP‑完整，说明在大规模实例上需要启发式或特殊求解器。

**⚠️ 局限性**

局限性包括：①仅关注一致性与行列嵌入，未提供实际布局生成或优化方法；②诱导子图同构仍为 NP‑难，限制了大规模模式检索的可行性；③模型未考虑噪声、误差或不完整的位置信息，适用范围受限。

---

## 403. Learning Asynchronous Upper-body Task-space Trajectory Tracking Policy for Humanoid Robots

**arXiv ID:** 2606.25706 | [PDF](https://arxiv.org/pdf/2606.25706v1)

**作者:** Yumeng Liu `[一作]` (Zhejiang University), Yue Wang `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出了一种针对双足机器人上半身任务空间轨迹的异步跟踪策略，利用时间索引条件化、滑动窗口奖励和基于MPC的轨迹补全实现从低频稀疏规划指令到高频完整身体控制的闭环；

**💡 创新点**

创新点在于：①时间索引条件化隐式对齐规划与执行帧差；②滑动窗口全局奖励抑制周期性漂移；③MPC完成稀疏上半身轨迹，提供完整身体参考；④自引导正则化（动作级与正向运动学级）防止下肢漂移；实现端到端的异步跟踪与后训练统一框架；

**🔧 技术方法**

采用教师-学生蒸馏、PPO强化学习、时间索引观测、滑动窗口奖励、MPC轨迹补全、动作级与FK级自引导正则化等技术；

**📊 数据集**

使用OMOMO与GRAB人类运动数据集进行预训练与评估，并以OmniH2O和SONIC两种教师模型为先验；在真实硬件上采用Unitree G1机器人进行验证；

**📈 对比分析**

与教师策略、同步稀疏跟踪、同步异步部署、解耦控制等基线进行对比。实验显示在1 Hz规划频率下，异步策略成功率达99.5%，Pos_1s、Rot_1s等异步误差显著低于基线；在OOD任务后训练后，MPC补全与自引导进一步提升成功率、减少漂移并保持关节安全；在硬件上亦实现了稳健的跟踪；

**⚠️ 局限性**

主要局限在于：需要先验的全身体运动数据进行教师训练；MPC补全对计算资源和模型假设有一定依赖；在极低规划频率或极端动态任务下仍可能出现漂移；对新领域迁移需重新进行后训练。

---

## 404. SARA: Unlocking Multilingual Knowledge in Mixture-of-Experts via Semantically Anchored Routing Alignment

**arXiv ID:** 2606.25821 | [PDF](https://arxiv.org/pdf/2606.25821v1)

**作者:** Tianyu Dong `[一作]` (Tianjin University), Deyi Xiong `[通讯]` (Tianjin University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出SARA框架，利用高资源语言的路由分布作为语义锚点，对低资源语言的专家激活进行对齐；

**💡 创新点**

创新点在于直接对齐稀疏Mixture-of-Experts的内部路由分布，并使用对称的Jensen‑Shannon散度进行正则化，从而弥补跨语言路由偏差；

**🔧 技术方法**

采用的技术包括稀疏MoE模型、Softmax路由器、JS散度损失、交叉熵与负载平衡损失的组合；

**📊 数据集**

实验使用了由GPT‑5 mini生成的并行指令数据，覆盖5种低资源语言（hi, ne, bn, te, sw），并在Qwen3‑30B‑A3B和Phi‑3.5‑MoE‑instruct上进行微调；

**📈 对比分析**

与基线（原始MoE、全微调、AES、ShifCon）对比，SARA在Global‑MMLU、BELEBELE和MGSM上平均提升约1–2%，在低资源语言上表现尤为显著；

**⚠️ 局限性**

局限性包括：训练数据风格单一、翻译过程产生的噪声、可能导致高资源语言偏向、以及层选择策略仅在特定架构上验证。

---

## 405. $S^{2}$-FracMix: Label-Preserving Self-Saliency Mixup Augmentation

**arXiv ID:** 2606.25784 | [PDF](https://arxiv.org/pdf/2606.25784v1)

**作者:** Khawar Islam `[一作]` (University of Melbourne), Naveed Akhtar `[通讯]` (Information Technology University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于自我显著性混合（Self‑Saliency Mix）和分形混合（FracMix）的数据增强框架S^2‑FracMix，利用同一张图像中的多尺度显著补丁进行旋转、模糊等变换并重插入，从而在不破坏语义的前提下增强样本多样性，并在显著区域注入自相似分形纹理来进一步提升鲁棒性。

**💡 创新点**

创新点包括：
1) 在单张图像内部完成显著性补丁的提取与重构，避免了跨样本混合导致的语义破坏；
2) 将分形纹理限定在显著补丁内，形成局部结构化扰动，既保持背景完整，又提升局部鲁棒性；
3) 引入多模式高层混合策略，随机组合Mixup、CutMix、ResizeMix等低层混合方式，进一步丰富训练分布；
4) 对上述技术给出了理论分析与多尺度、跨任务的实证验证。

**🔧 技术方法**

核心技术包括：显著性检测（谱残差方法）、多尺度补丁提取、随机旋转与高斯模糊变换、分形纹理混合、混合权重自适应（α、λ）、多模式高层混合策略；实现基于OpenMixup的训练框架。

**📊 数据集**

使用的主要数据集有：CIFAR‑100、Tiny‑ImageNet、ImageNet‑1K（粗粒度分类），Caltech Birds‑200、FGVC‑Aircrafts、Stanford Cars、CUB‑200（细粒度分类），以及CIFAR‑100‑C（腐蚀测试），用于迁移学习的 Caltech Birds‑200/Stanford Cars 等。

**📈 对比分析**

与Mixup、CutMix、ManifoldMix、FMix、ResizeMix、SaliencyMix、PuzzleMix、AutoMix、AdAutoMix等方法进行对比。实验显示，S^2‑FracMix在所有主流CNN/ViT骨干网络上均取得了SOTA，Top‑1准确率比AdAutoMix提升约0.4–0.7%，在鲁棒性、校准度和迁移学习等多项指标上也显著优于对手。

**⚠️ 局限性**

限制包括：
- 尽管相较传统混合方法计算量更低，但相比最简单的Mixup仍有一定开销；
- 依赖显著性检测，若显著性模型失效可能导致补丁选择失真；
- 目前主要在图像分类与检测任务验证，未覆盖更广的领域（如视频、医学图像等）。

---

## 406. Fuzzy Quantification over OWL Ontologies and Knowledge Graphs

**arXiv ID:** 2606.25778 | [PDF](https://arxiv.org/pdf/2606.25778v1)

**作者:** Enrique Palacín `[一作]` (University of Zaragoza), Umberto Straccia `[通讯]` (CNR-ISTI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

实现了一套通用框架，用于在传统 OWL 本体和 RDFS 知识图谱中评估 Type I 与 Type II 的模糊量化查询。

**💡 创新点**

框架对量化类型、评估方法和数据源均保持无关，能够通过模糊数据类型让标准推理器支持模糊查询，并对多种评估方法（Zadeh、Yager OWA、GD）提供统一接口。

**🔧 技术方法**

使用模糊 OWL 2 数据类型、FuzzyOWL2、OWLAPI、HermiT、Apache Jena、FuzzyDL 以及 Datil（聚类学习模糊数据类型）等技术，配合多种量化评估方法实现查询。

**📊 数据集**

在 Premierleague 足球本体（约 11,000 个实例）和 DBpedia（约 27,500 个城市）上进行实验，评估模糊量化查询的效果和性能。

**📈 对比分析**

与 GD 与 Zadeh 方法对比，实验显示预处理后单次查询耗时仅几百毫秒；在大型本体与知识图上均能保持较低延迟，性能表现可接受。

**⚠️ 局限性**

尚未支持嵌套量化句子；对极大规模本体的实时推理、模糊数据类型学习以及不同推理器之间的兼容性仍存在挑战。

---

## 407. Space-Efficient Language Generation in the Limit

**arXiv ID:** 2606.25777 | [PDF](https://arxiv.org/pdf/2606.25777v1)

**作者:** Nicolas Flammarion `[一作]` (École Polytechnique Fédérale de Lausanne), Ola Svensson `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文研究在有限空间下利用有限自动机学习和生成正样本语言的理论与算法；

**💡 创新点**

创新点在于提出空间受限的“生成在极限”模型，证明多项式空间可实现有限差距生成，而指数空间才能完全识别；

**🔧 技术方法**

技术主要包括DFAs、拓扑排序、Savitch式空间递归、以及通信复杂度中的Index问题下界；

**📊 数据集**

本文不使用实际数据集，全部采用理论构造和符号流的模拟；

**📈 对比分析**

与指数空间算法对比，本文给出多项式空间下的生成差距为O(k²s⁻²)，并通过通信复杂度下界证明此差距不可进一步压缩；

**⚠️ 局限性**

局限性在于仅针对正规语言，且生成差距仍为指数，未扩展至非正规或更复杂的语言类。

---

## 408. RAS: Measuring LLM Safety Through Refusal Alignment

**arXiv ID:** 2606.25750 | [PDF](https://arxiv.org/pdf/2606.25750v1)

**作者:** Chang-Chieh Huang `[一作]` (National Yang Ming Chiao Tung University), Wei-Bin Lee `[通讯]` (Hon Hai Research Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于内部表示的拒绝对齐评估方法，并给出可校准的安全评分 RAS

**💡 创新点**

创新在于不依赖生成文本，而是通过对齐方向衡量模型在不安全和越狱提示下的内部拒绝态势，并将其映射为 0-100 分数

**🔧 技术方法**

使用白盒层级残差激活、余弦相似度、方向提取、加权评分、Sigmoid 归一化等技术

**📊 数据集**

使用安全对齐参考模型、不同安全级别的校准模型，以及包含安全、非法和越狱的提示集

**📈 对比分析**

与传统基于输出的攻击成功率（ASR）和 judge‑based 评估对比，RAS 能够在 200‑500 倍的速度提升下保持与 ASR 的高相关性

**⚠️ 局限性**

局限于需要白盒访问、只能在模型家族内校准、若未来安全机制与拒绝方向不同则需重新估计

---

## 409. CodeChat-Eval: Evaluating Large Language Models in Multi-Turn Code Refinement Dialogues

**arXiv ID:** 2606.25747 | [PDF](https://arxiv.org/pdf/2606.25747v1)

**作者:** Guoxiang `[一作]`, Aldeida Aleti `[通讯]` (Monash University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了CodeChat-Eval评估框架，用于衡量大型语言模型在多轮代码改进对话中的功能正确性。

**💡 创新点**

创新点包括动态指令选择算法（AGDIS）、多轮评估议程以及Mean Sustainable Turns（MST）度量，系统揭示LLM在多轮改进中功能正确性下降的具体原因。

**🔧 技术方法**

使用OpenAI GPT系列、Llama、Qwen、DeepSeek等LLM进行测试，并采用Prompt‑based指令适用性与遵循度评估，结合EvalPlus扩展的测试套件。

**📊 数据集**

数据集基于HumanEval和MBPP的任务，结合EvalPlus扩充的测试用例。

**📈 对比分析**

在八款LLM上进行10轮评估，功能正确性下降19.2%–69.2%；MST最高为GPT‑5的6.89，显示多轮改进相比单轮基准更具挑战性。

**⚠️ 局限性**

局限包括仅评估Python语言、仅使用单一指令级别（未覆盖复合指令）、指令来源有限、评估依赖Prompt导致主观性以及未覆盖更长对话。

---

## 410. Point Cloud Diffusion with Global and Local Reconstruction for Instance-Level 3D Anomaly Detection

**arXiv ID:** 2606.25740 | [PDF](https://arxiv.org/pdf/2606.25740v1)

**作者:** Linchun Wu `[一作]` (Wuhan University), Qingquan Li `[通讯]` (Shenzhen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 PCDiff 框架，用于点云的实例级 3D 异常生成与检测，结合全局与局部重建实现精细恢复。

**💡 创新点**

核心创新包括梯度引导的纹理表示、跨模态实例注意力机制实现可控生成，以及 2D 先验驱动的局部‑全局联合重建检测方案。

**🔧 技术方法**

技术手段涵盖点云扩散模型、跨视角图像扩散、多模态注意力、Transformer 结构、梯度编码、CLIP 文本/图像嵌入、反投影与积分噪声重建。

**📊 数据集**

使用 Anomaly‑ShapeNet（1,600 样本，40 类）与 Real3D‑AD（1,254 样本，12 类）两大工业缺陷数据集进行实验。

**📈 对比分析**

与 R3D‑AD、CutPaste、Gau‑SP、PatchCore、BTF 等前沿方法对比，生成 F‑Score 95.4%、CLIP 相似度 81.0%，检测 O‑AUROC 0.93、P‑AUROC 0.82，显著优于基线。

**⚠️ 局限性**

局限在于对 2D 检测与网格重建的依赖，易受误差传播；模型参数量大，内存占用高；需多视角渲染，部署成本相对较高。

---

## 411. UniTeD: Unified Temporal Diffusion for Joint Perception and Planning in Autonomous Driving

**arXiv ID:** 2606.25736 | [PDF](https://arxiv.org/pdf/2606.25736v1)

**作者:** Bo Zhao `[一作]` (Nullmax), Haibin Ling `[通讯]` (Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了统一的时间扩散框架 UniTeD，利用共享生成空间的迭代去噪同时对感知（检测、预测、地图）和规划任务进行联合建模，并引入时序转换模块 (TTM) 与锚刷新策略 (ARS) 以提升鲁棒性和流式处理能力。

**💡 创新点**

创新点：
1) 将感知与规划统一进扩散模型，允许双向信息交换，消除传统分离架构中的误差累积；
2) 时序转换模块解决历史与当前帧噪声不匹配问题，实现流式统一扩散；
3) 锚刷新策略缓解稀疏查询扩散中的训练‑推理分布漂移，提升推理稳定性。

**🔧 技术方法**

使用技术：扩散模型 (DDIM、DiT 的 AdaLN)、多尺度可变形注意力、多任务查询集、内存队列、时序交叉注意力、Anchor Refresh、共享条件调制、时间步嵌入。

**📊 数据集**

使用数据集：NAVSIM v1/v2、Bench2Drive、nuScenes。

**📈 对比分析**

与现有判别式和生成式端到端方法对比：在 NAVSIM v1 达到 PDMS 90.2（超出最佳 89.4），NAVSIM v2 EPDMS 90.1（超出最佳 86.2），Bench2Drive DS 87.3（超出最佳 87.1），在所有基准上均实现了显著提升。

**⚠️ 局限性**

局限性：
- 推理速度受限于扩散迭代次数，尚未在实时车载系统上验证；
- 对多模态输入（如 LiDAR + 视觉）鲁棒性需进一步评估；
- 高计算开销与显存占用相对较大。

---

## 412. Shoot the Honey, Cloak the Player: Towards Zero-Runtime-Overhead Proactive Defense and Detection for Visual Game Cheating

**arXiv ID:** 2606.25734 | [PDF](https://arxiv.org/pdf/2606.25734v1)

**作者:** Jianing Wang `[一作]` (Shandong University), Shanqing Guo `[通讯]` (Shandong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种完整的防御框架，实时对视觉瞄准外挂进行干扰，并在赛后通过“诱饵纹理”记录证据进行检测。

**💡 创新点**

创新点在于：1）引入两种 3D 纹理攻击（Adversarial Camouflage 和 Honeypot），既能实时降低外挂信号，又能产生可解释的作弊痕迹；2）使用可微渲染+Expectation‑over‑Renderings 生成鲁棒纹理；3）提供端到端、零运行时开销且可在主流 FPS 游戏中部署。

**🔧 技术方法**

核心技术包括：可微渲染器、EoR 损失、投影到 3D 纹理的梯度优化、纹理安全管理、基于轨迹的异常检测（B-LSTM + MLP）。

**📊 数据集**

数据集为：CS2 真实游戏地图中的 24 条纹理（不同材质）生成的 AHT 与 ACT；玩家与外挂的 3,074 条交互轨迹；40 场实战比赛记录用于评估。

**📈 对比分析**

与现有方法（Invisibility Cloak、AdvMap、XGuardian 等）对比：AHT 的 Decoy Success Rate 96.9%，ACT 的 Evasion Rate 85.1%；误报率低至 3.49×10⁻⁸；跑时开销几乎为零；在真实比赛中 100% 正确分类。

**⚠️ 局限性**

局限性包括：1）对纹理的依赖，若游戏更新纹理布局需重新生成；2）只针对视觉外挂，无法阻止内存外挂；3）在极端视角或光照极端条件下效果可能下降；4）对高分辨率或复杂模型的纹理可能需要更大算力。

---

## 413. Falcon: Functional Assembly and Language for Compositional Reasoning in X-ray

**arXiv ID:** 2606.25701 | [PDF](https://arxiv.org/pdf/2606.25701v1)

**作者:** Yonathan Michael `[一作]` (Khalifa University), Naoufel Werghi `[通讯]` (Khalifa University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在 X 射线图像中进行功能性威胁推理的框架 Falcon，结合分割感知和结构化安全状态，以实现对拆解型 IED 的识别与风险评估。

**💡 创新点**

创新点在于将威胁建模为组件间的关系属性，设计了结构化安全适配器（SSA）将分割结果映射为组件出现、功能关联与场景风险的显式中间表示，显著提升了相对对象检测的功能性推理。

**🔧 技术方法**

采用 DINOv2 视觉编码器、RF-DETR 的分割检测头、SSA 结构化推理模块以及 Vicuna‑7B 语言模型，并通过三阶段训练实现分割感知、结构化推理与指令微调。

**📊 数据集**

使用了新构建的 Falcon‑X 基准，包含约 7,000 张双能量 X 射线扫描，配有实例级边框、像素掩码、组件存在、功能关联和风险评分，并通过对抗式假设生成扩充到约 5 万张图像。

**📈 对比分析**

在 Falcon‑X 上与多种通用和领域适配的 VLM 对比，Falcon 在功能性定位、完整性评估和风险预测上分别实现了约 30% 的 IoU 提升、MAE 下降 0.02 以及风险预测 MAE 下降 0.005，显示出显著的性能优势。

**⚠️ 局限性**

局限性包括对组件词典的依赖、对高重叠/透明物体的检测仍受限，以及在真实机场场景中的泛化仍需进一步验证。

---

## 414. Confidence Sequences for Online Statistical Model Checking of Markov Decision Processes

**arXiv ID:** 2606.25797 | [PDF](https://arxiv.org/pdf/2606.25797v1)

**作者:** Konstantin Kueffner `[一作]` (Institute of Science and Technology Austria), Patrick Wienhöft `[通讯]` (TUD Dresden University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一套用于在线马尔可夫决策过程统计模型检查的置信序列方法，并通过价值基的假设检验进一步提升估计精度；

**💡 创新点**

创新点在于：①将置信序列引入MDP-SMC，使估计随采样持续改进；②设计了直接针对价值函数的假设检验，避免逐个概率估计；③将置信分配先按空间再按时间分配，实现采样与求值交织；

**🔧 技术方法**

采用了Hoeffding、Bernstein、Clopper‑Pearson等经典置信区间的序列化版本，结合stitching、betting、test‑martingale技术；以及基于Beta、MLE等先验的假设检验方法；

**📊 数据集**

实验基于PRISM基准集与实践者指南社区集合，涵盖多种可达性查询的MDP模型；

**📈 对比分析**

与传统SOTA方法（如SOTA-Sq‑CP、SOTA-Exp‑Hoeff等）以及CAV19实现相比，平均样本量仅为最优方法的1.3–1.5倍，最差时比旧方法低50倍，证明了显著的样本效率提升；

**⚠️ 局限性**

局限性包括：①仍需手工设置信赖度分配与先验；②数值稳定性与求解速度对极大模型的影响；③当前仅支持已知支持的灰盒MDP，且针对动态变化的转移概率尚未覆盖。

---

## 415. How Large Language Models Source Brand Reputation Across Languages and Markets

**arXiv ID:** 2606.25787 | [PDF](https://arxiv.org/pdf/2606.25787v1)

**作者:** Dmitrij Zatuchin `[一作]` `[通讯]` (Estonian Entrepreneurship University of Applied Sciences), Dmitrij Zatuchin (Estonian Entrepreneurship University of Applied Sciences)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过分析三份 Rankfor.AI 提供的引用数据，研究大型语言模型在回答关于128个品牌问题时所引用的网络来源，包括域名、来源类型、语言、市场和不同模型的差异。

**💡 创新点**

创新点在于：①首次聚焦引用来源而非答案文本，揭示品牌信息的第三方主导和来源集中度；②发现来源分布符合 Zipf 定律；③展示不同语言和市场的顶级来源差异（如维基百科、地方媒体、招聘平台等）；④比较不同 LLM 模型的引用行为。

**🔧 技术方法**

技术手段包括域名解析与标准化、来源类型标注、长尾分布统计、Zipf 拟合、卡方检验、模型对比统计以及对重定向 URL 的处理。

**📊 数据集**

使用的数据集为 Rankfor.AI 的三份引用数据（北欧-波罗的海、波兰、东欧中部），共计 167,551 条 URL 引用，覆盖 128 个品牌、13 种语言和 12 个市场。

**📈 对比分析**

通过统计各模型（Perplexity、Gemini、GPT）在引用量、域名数量、品牌自有域名比例等维度的表现进行对比。结果显示 Perplexity 产生最多引用且域名最分散，Gemini 的自有域名比例被重定向掩盖；总体对比为描述性统计，并未给出传统意义上的性能指标。

**⚠️ 局限性**

局限性包括：①数据合并不完全同质，部分数据缺乏可解析 URL；②品牌自有域名检测依赖品牌词匹配，可能漏检或误检；③引用级别的计量未按实体精准匹配；④不同报表使用的基准不同，导致百分比解释不一致；⑤研究仅覆盖特定模型版本与采样窗口，结果具有时间与版本依赖性；⑥未验证引用内容与实际品牌声誉的对应关系。

---

## 416. Re-mixing Embeddings for Patient Augmentation in Data Scarce Multiple Instance Learning

**arXiv ID:** 2606.25770 | [PDF](https://arxiv.org/pdf/2606.25770v1)

**作者:** Muhammed Furkan Dasdelen `[一作]` (Computational Health Center & Helmholtz AI, Helmholtz Munich), Ario Sadafi `[通讯]` (Computational Health Center & Helmholtz AI, Helmholtz Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于高斯混合模型的患者级别数据增强框架 RECIPE，用于在数据稀缺的多实例学习（MIL）任务中生成真实感患者。

**💡 创新点**

创新点在于通过聚类获得疾病特定的“配方”，可离线生成缺失类别的患者，并结合不确定性量化挑选最具信息量的合成样本，解决了传统方法无法处理缺失类别且只能在线生成样本的局限。

**🔧 技术方法**

技术包括：在所有实例嵌入上训练 K 维高斯混合模型、基于聚类统计生成疾病配方、按配方抽样重新混合实例、使用蒙特卡罗 Dropout 计算预测熵进行不确定性评估以及将选出的合成患者用于重新训练 MIL 模型。

**📊 数据集**

使用了四个真实医学数据集：血细胞图像集 AML‑Hehr、cAItomorph；单细胞 RNA‑seq PBMC；以及 COVID‑19 流式细胞计数 Covid‑flow；此外还在单细胞 RNA‑seq 和流式细胞计数任务中测试了不同嵌入方法（PCA、scVI、BDC1/BDC2）。

**📈 对比分析**

与基线、MixUp、PseMix、ReMix 等现有增强方法比较，RECIPE 在缺失类别、少样本和小规模非图像任务中均能显著提升性能，最高提升可达 38%（如 Transformer 上从 0.49 提升至 0.67），在多样本设定中实现与完整数据集相当的准确率。

**⚠️ 局限性**

局限性包括：需要足够的实例嵌入来训练高斯混合模型，依赖嵌入的质量与分布；在极端少样本情况下配方估计不稳定；当前仅针对 MIL 结构，可能无法直接应用于全局监督或细粒度标签任务。

---

## 417. Deep Neural Networks with Ordinal Loss for Medical Applications

**arXiv ID:** 2606.25769 | [PDF](https://arxiv.org/pdf/2606.25769v1)

**作者:** Tal Dvora `[一作]` (Bar-Ilan University), Gonen Singer `[通讯]` (Bar-Ilan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种Ordinal Cross-Entropy (OCE) 损失函数，用于医疗序数分类任务，结合误分类成本矩阵；

**💡 创新点**

在传统交叉熵基础上加入距离感知与方向敏感的成本矩阵，保持概率解释与优化优势，并提供梯度理论与实证证明其更低的误分类成本；

**🔧 技术方法**

采用成本敏感损失、深度神经网络（VGG19、InceptionV3、DenseNet121）、softmax+自定义OCE损失、Adam优化器等技术；

**📊 数据集**

使用APTOS 2019 Blindness Detection视网膜糖尿病视网膜病变分级数据集；

**📈 对比分析**

与标准交叉熵、带贝塔/泊松/二项/指数标签平滑的CE以及先前的Ordinal Loss（OL）在对称和非对称成本矩阵下进行5折交叉验证；OCE在误分类成本上常位列第一或第二，MAE、AUC、QWK亦保持竞争力；

**⚠️ 局限性**

仅在单一医疗数据集上验证，成本矩阵参数手工设定未自适应，缺乏跨域验证与自适应学习机制。

---

## 418. ShutterMuse: Capture-Time Photography Guidance with MLLMs

**arXiv ID:** 2606.25763 | [PDF](https://arxiv.org/pdf/2606.25763v1)

**作者:** Jiayu Li `[一作]` (Fudan University), Xingjun Ma `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 CaptureGuide-Bench 与 CaptureGuide-Dataset，构建统一的多模态大型语言模型 ShutterMuse，用于实时摄影指导，包括摄影师侧构图决策与主体侧姿势推荐；

**💡 创新点**

创新点在于：①首个针对捕捉时摄影指导的基准与大规模带文本推理与结构化标注的数据集；②采用专家seed+MLLM验证的自蒸馏扩展，显著提升标注规模与质量；③结合监督与强化学习（GRPO）实现统一的决策与精细化裁剪/姿势输出；

**🔧 技术方法**

主要技术包括：多模态 LLM (Qwen3-VL-8B) 的监督微调与GRPO强化学习、基于 MLLM 的自蒸馏与验证、构图和姿势的结构化 JSON 输出、以及 MLLM 评估指标（IoU、BDE、MLLM-Score 等）；

**📊 数据集**

使用了约 13 万张图片的数据集，其中 10 万张用于摄影师侧（含 3 种决策、裁剪框及理由），3 万张用于主体侧（人像场景转换、关键点、可见性及理由）；

**📈 对比分析**

实验与多种开源/闭源 MLLM、专用裁剪模型、图像编辑模型比较，ShutterMuse 在摄影师侧 IoU 74.30%、R 70.03%、RSR 82.76%、KSR 74.55、MLLM-Score 0.64 取得最优；在主体侧，平均分 0.34，推理时间仅 4.96s、Tokens 412，显著低于 GPT-Image-2 与 Nano-Banana-Pro；

**⚠️ 局限性**

局限性包括：对非裁剪需求的判别仍不完善；姿势推荐受模型尺寸与预训练语义限制；自蒸馏过程中依赖 MLLM 验证器的准确性；需进一步提升对多样化场景的泛化能力。

---

## 419. Gradient-based inverse lithography for EUV masks via the waveguide method and a physics-informed neural operator

**arXiv ID:** 2606.25753 | [PDF](https://arxiv.org/pdf/2606.25753v1)

**作者:** Vasiliy A. Es'kin `[一作]` (University of Nizhny Novgorod), Egor V. Ivanov `[通讯]` (University of Nizhny Novgorod)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于梯度的逆光刻技术（ILT）框架，利用可微波导方法和波导神经算子（WGNO）对EUV掩模吸收层的介电常数进行自动微分优化，从而实现目标场在硅晶圆上的重现。

**💡 创新点**

创新点在于：①将可微波导方法与WGNO融合成端到端可微的物理引擎；②提出像素密度重参数化与傅里叶投影两种优化策略；③在2D/3D EUV掩模的逆设计中首次实现完整的自动微分求解，并系统比较了三种吸收材料的性能。

**🔧 技术方法**

使用技术包括：可微波导方法、波导神经算子、自动微分、像素密度重参数化、傅里叶投影、硬二值化、总变差/光谱正则化、Adam优化器、PyTorch框架。

**📊 数据集**

数据集：基于实验数据库的Ru/Be/Sr多层镜面材料及三种吸收材料（TaBN、La、U）的介电常数；目标场为理想二值光强分布；实验在2D/3D 真实掩模结构上进行，未使用公开公开数据集。

**📈 对比分析**

通过与像素密度方法对比，傅里叶参数化在相同问题上加速1.31倍（从179s降至137s），同时生成更平滑、易制造的掩模壁；对三种吸收材料进行对比，La在中心峰值最突出，U的整体场分布最接近目标。性能评估主要基于优化时间和光强分布误差。

**⚠️ 局限性**

局限性：①WGNO在当前实现中未带来速度提升，原因是网络训练与掩模优化同时进行，训练开销大；②实验仅在单CPU节点完成，缺乏大规模并行评估；③仅针对单层吸收层的逆问题，未探讨多层吸收或更复杂结构的可扩展性；④需要进一步改进WGNO的训练效率和整体并行化方案。

---

## 420. Efficient Real-World Dehazing via Physics-Inspired Global-Local Decoupling

**arXiv ID:** 2606.25732 | [PDF](https://arxiv.org/pdf/2606.25732v1)

**作者:** Yifei Qu `[一作]` (Harbin Institute of Technology), Jinyuan Wu `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种轻量化、物理启发的单幅图像去雾框架 PGL-Net，能够在保持高质量恢复的同时显著降低计算开销。

**💡 创新点**

创新点包括：1）通过 Physics‑Inspired Affine Fusion (PAF) 在跳连中实现全局气象偏置校正；2）利用 Degradation‑Aware Modulation (DAM) 进行局部细节重建；3）整体采用物理模型衍生的特征空间仿射正则，而非显式估计透射率与大气光。

**🔧 技术方法**

主要技术手段包括轻量化 U‑shaped 结构、全局平均池化生成通道尺度与偏置、深度可分离卷积、频域（实部/虚部）损失、以及动态门控的 DAM 块。

**📊 数据集**

在多种真实世界去雾基准上验证：RRSHID、RW^2AH、RUDB（含 NH‑HAZE21/HD‑NH‑HAZE）、RTTS 以及无标注的 URHI。

**📈 对比分析**

与现有 SOTA 比较（SGDN、DehazeFormer 等），PGL‑Net‑T 在 PSNR 上提升 0.8–2.6 dB、SSIM 0.02–0.08，LPIPS 降低 0.02–0.05，同时推理延迟下降 10–12 倍，且在 RTTS 上的检测 mAP 提升 0.8–1.5 点。

**⚠️ 局限性**

局限性包括：在极稠密雾或夜间光照复杂场景下可能出现残留雾纹或色彩失真；以及对极端遮蔽细节恢复仍不完美。

---

## 421. MAP-Based Task-Oriented Precoding for Multiuser Communication

**arXiv ID:** 2606.25722 | [PDF](https://arxiv.org/pdf/2606.25722v1)

**作者:** Mohammad Javad Ahmadi `[一作]` (Technische Universitaet Dresden), H. Vincent Poor `[通讯]` (Princeton University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于MAP误差近似的任务导向多用户无线通信框架，联合优化特征提取和预编码以提升分布式分类精度。

**💡 创新点**

创新点在于直接以MAP判决误差为目标，避免传统协方差逆运算和特征重构的计算复杂度；同时设计了低复杂度的类均值分离损失与梯度预编码更新。

**🔧 技术方法**

采用MAP检测理论、近似高斯误差、Q函数指数替代、梯度投影法以及神经网络特征学习（ResNet-18 + 2层FC）。

**📊 数据集**

使用CIFAR‑10数据集，生成两种随机增强视图作为边缘设备观测。

**📈 对比分析**

与MCR²和LMMSE预编码基线比较，实验表明所提方法在相同网络架构下获得与MCR²相近甚至更优的分类准确率，同时MAP误差更低，且计算复杂度显著下降。

**⚠️ 局限性**

局限性包括对均匀类别先验的假设、依赖预估的通道协方差近似、以及在更复杂多用户场景下对协同预编码的可扩展性尚待验证。

---

## 422. Equilibrium and Infeasibility: A new solution concept for games

**arXiv ID:** 2606.25707 | [PDF](https://arxiv.org/pdf/2606.25707v1)

**作者:** Anne Reulke `[一作]` (Orange Research), Rachid El-Azouzi `[通讯]` (Laboratoire d'Informatique d'Avignon)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了 ψ-惩罚解概念，用以解决个体约束可能为空且无耦合可行解的非合作通用游戏。

**💡 创新点**

创新点在于将不可行约束通过可调惩罚函数嵌入效用函数，并证明其存在性及与 GNE、纳什和解的关系，从而为无 GNE 或无可行策略的游戏提供可行解。

**🔧 技术方法**

主要技术包括凸分析、下半连续性、极限序列、精确惩罚理论以及对游戏结构的严格数学描述。

**📊 数据集**

未使用实测数据集，主要通过理论推导和 Divide-the-Dollar 示例来说明方法。

**📈 对比分析**

与传统 GNE 和纳什和解比较，证明在满足一定条件时 ψ-惩罚解包含所有 GNE，且在无 GNE 的情形下仍能产生合理解；性能通过罚参数趋近无穷来实现逼近。

**⚠️ 局限性**

局限性包括对下半连续性、严格凸性等假设的依赖，缺乏对可微惩罚函数的普适性支持，且在更一般游戏结构下的可扩展性仍待进一步研究。

---

## 423. GUI agent: Guided Exploration of User-Sensitive Screens

**arXiv ID:** 2606.25705 | [PDF](https://arxiv.org/pdf/2606.25705v1)

**作者:** Aradhana Nayak `[一作]` (Huawei Heisenberg Research Center), Feng Liu `[通讯]` (Huawei Heisenberg Research Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一个基于MCTS和强化学习的 Explorer 代理，用于系统化地探索 GUI 应用的查询空间，并识别出用户敏感的查询与屏幕。

**💡 创新点**

创新点在于引入基于奖励的 novelty scoring 与经验蒸馏机制，结合 Group Relative Policy Optimization（GRPO）训练和多轮 Saturation 检测，能够聚焦并收敛到用户敏感状态。

**🔧 技术方法**

采用的技术包括：蒙特卡罗树搜索（MCTS）、GRPO、词嵌入相似度筛选、Qwen‑2.5‑32B‑Instruct 作为 Explorer LM、Qwen3‑Embedding‑0.6B、SPABench 安卓模拟器、以及自定义 reward 计算与 Saturation 算法。

**📊 数据集**

使用 SPABench 安卓模拟器提供的应用轨迹与自动生成的查询作为训练数据集，构成了经验缓冲区（memory bank）来蒸馏经验并训练模型。

**📈 对比分析**

与传统 LLM 代理相比，该方法在三轮训练中查询 novelty score 由 0.0519 降至 0.0042，步长预测准确率提升，且整体奖励显著下降，表明查询空间被有效压缩，能更可靠地识别敏感屏幕。

**⚠️ 局限性**

局限性包括：对低 novelty 查询过滤可能导致覆盖不足，算法对复杂屏幕的探索深度有限，且在多任务环境下的泛化能力尚待进一步验证。

---

## 424. Memory-Efficient Policy Libraries with Low-Rank Adaptation in Reinforcement Learning

**arXiv ID:** 2606.25700 | [PDF](https://arxiv.org/pdf/2606.25700v1)

**作者:** Samuel Valland Lyngset `[一作]` (University of Oslo), Tobias Lømo `[通讯]` (University of Oslo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究 LoRA 在多任务强化学习中的可迁移性，使用 PPO 在 Meta‑World 仿真环境中训练基准策略后对不同任务进行 LoRA 微调，构建专家策略库。

**💡 创新点**

证明 LoRA 能在保持几乎相同性能的同时，将单个任务的可学习参数减少 95% 以上，并显著降低内存占用（20–160 倍）与反向传播计算量。

**🔧 技术方法**

采用 PPO 训练框架，使用 LoRA 低秩适配器进行 PEFT 微调，并比较完整微调与 LoRA 的参数量、存储、计算与成功率。

**📊 数据集**

使用 Meta‑World 的 MT1 任务集（Pick‑Place 为基准，进一步微调到 6 个目标任务）。

**📈 对比分析**

对比方法：对 3600 次实验（100 次重复 × 5 种 LoRA 低秩 + 完整微调），评估成功率、训练步数、内存/存储和 FLOP。LoRA 在大多数任务上成功率与完整微调相当，训练时间略增，内存/存储下降 85% 以上，FLOP 降低 25–30%。

**⚠️ 局限性**

局限：仅评估单一基准策略；仅使用 PPO，未探索其他算法；任务数量有限；实验受模拟收集瓶颈影响，真实硬件上的性能提升尚未验证。

---

## 425. A Backward-Compatible Protocol Upgrade for HotNets

**arXiv ID:** 2606.25786 | [PDF](https://arxiv.org/pdf/2606.25786v1)

**作者:** Paolo Costa `[一作]` (Microsoft Research), Michael Schapira `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

对 ACM HotNets 2026 进行结构性改革，扩展稿件范围、引入技术与观点双类评估、采用协作式二值化审稿流程、重新设计讨论导向的研讨会节目，并探索生成式 AI 在审稿与传播中的辅助角色。

**💡 创新点**

创新点包括：① 兼顾技术与社区视角的两类稿件；② 采用“足够支持”二值化审稿模型以容纳更多争议性思路；③ 将审稿过程从评分转向协作式讨论与合成反馈；④ 通过互动式论文讨论、海报、演示等形式实现程序“讨论优先”；⑤ 采用可零存储 LLM 进行审稿辅助，兼顾安全与质量。

**🔧 技术方法**

使用 HotCRP 平台进行审稿与讨论；引入零存储 LLM（如 OpenAI API）协助审稿人识别文献、校对事实、建议问题；利用多媒体与 AI 生成的交互式补充材料提升论文可访问性。

**📊 数据集**

未使用传统数据集；改动基于社区反馈与经验，依赖提交论文内容本身。

**📈 对比分析**

无传统实验或性能评估；改动通过社区调研与后续报告来验证其有效性与可行性。

**⚠️ 局限性**

局限性包括：① 采用二值化支持模型可能导致错误接受（false positives）；② 对 AI 辅助的信任与审核依赖于作者同意与人工校验；③ 规模扩展后仍需平衡讨论深度与议程时间；④ 需要持续评估社区对新流程与形式的适应度。

---

## 426. Do Encoders Suffice? A Systematic Comparison of Encoder and Decoder Safety Judges for LLM Adversarial Evaluation

**arXiv ID:** 2606.25782 | [PDF](https://arxiv.org/pdf/2606.25782v1)

**作者:** Han Jeon `[一作]` (PricewaterhouseCoopers), Matt Wood `[通讯]` (PricewaterhouseCoopers)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对现代编码器分类器（Ettin系列）在对抗性安全评估中的表现进行了系统评估，并与多种基于LLM的安全判定器和解码器分类器进行对比。

**💡 创新点**

首次在大型对抗性安全基准上对编码器与解码器判定器进行全面基准测试，探讨编码器作为低成本低延迟第一道安全过滤的可行性。

**🔧 技术方法**

利用现代Encoder架构ModernBERT（Ettin）、多种LLM-as-a-judge策略（StrongReject、SorryBench等）以及解码器安全模型（LlamaGuard 3/4等）进行微调与评估。

**📊 数据集**

训练集涵盖14个源数据集（包括HarmBench、SafetyBench、ToxicChat、WildJailbreak等），评估集为JailbreakBench与AILuminate的1400个题目，覆盖16个目标LLM和四类对抗攻击。

**📈 对比分析**

在OOD holdout 上，Ettin‑400M 的F1≈0.894，FNR≈0.146，显著优于解码器 Judge（如Claude Judge、LlamaGuard‑4），但在多轮攻击（尤其是拆分攻击）上仍表现出较高的误报率，说明编码器在分布式危害模式上有限。

**⚠️ 局限性**

主要限制在于对抗性多轮攻击的识别不足以及训练标签噪声导致的性能瓶颈，编码器无法直接观察对话历史的推理路径，需与更复杂的解码器或轨迹感知系统结合。

---

## 427. StairMaster: Learning to Conquer Risky Hollow Stairs for Agile Quadrupedal Robots

**arXiv ID:** 2606.25765 | [PDF](https://arxiv.org/pdf/2606.25765v1)

**作者:** Xincheng Tang `[一作]` (Shanghai Jiao Tong University), Ruigang Yang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8d10c613-917e-4880-9716-17789f50e119` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

为四足机器人设计了能在高倾角空洞楼梯上稳定行走的三阶段强化学习框架StairMaster。

**💡 创新点**

创新点包括结合跨注意力的视觉空间编码器与空间感知LSTM来克服视觉盲区、引入仿真到真实的高保真深度噪声建模，以及针对空洞楼梯的三种自定义奖励（主动俯仰、空洞间隙惩罚、阶缘惩罚）。

**🔧 技术方法**

使用了深度学习强化学习（PPO、教师‑学生蒸馏）、跨注意力机制、空间感知LSTM、深度噪声仿真、三阶段训练流程以及自定义奖励。

**📊 数据集**

主要数据来自Isaac Gym仿真环境的自定义空洞楼梯样本（多角度、随机间隙），并在真实Unitree Go2机器人上收集深度图（Intel RealSense D435）。

**📈 对比分析**

与Extreme Parkour、HIMLoco等基线对比，在55°倾角空洞楼梯上零样本转移实现了高达40%（实际为80%在37°、40%在55°）的成功率，且在模拟中的成功率几乎为100%，显著优于基线。

**⚠️ 局限性**

局限在于深度噪声建模仍不能覆盖所有现实场景、仅使用单通道深度图、未考虑动力学细节与RGB信息，且在更复杂工业约束下表现未知。

---

## 428. Distributed SDN-Based Communication Architecture for the Pods4Rail System

**arXiv ID:** 2606.25711 | [PDF](https://arxiv.org/pdf/2606.25711v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 429. Uncertainty Quantification for Computer-Use Agents: A Benchmark across Vision-Language Models and GUI Grounding Datasets

**arXiv ID:** 2606.25760 | [PDF](https://arxiv.org/pdf/2606.25760v1)

**作者:** Divake Kumar `[一作]` (University of Illinois Chicago), Amit Ranjan Trivedi `[通讯]` (University of Illinois Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了跨域的后置不确定性量化基准Argus，用于单步可执行GUI点击任务，覆盖开放权重与闭源API两类模型；

**💡 创新点**

创新点包括：①跨模型、跨数据集、跨接口的统一评测框架；②引入UQ方法排名转移分析，揭示不同 regime 下的可迁移性；③提供API兼容的闭源子面板与对比；④结合分割式共形预测生成可部署的点击安全区域；

**🔧 技术方法**

采用27种后置UQ方法（logit、采样、混合、密度/探针、注意力、口头化、VLM原生）以及分割共形预测、Spearman排名转移、ECE/Brier、PRR等多种评估指标；

**📊 数据集**

使用四个 GUI 基准数据集：ScreenSpot‑v2、ScreenSpot‑Pro、OSWorld‑G、UI‑Vision‑EG；

**📈 对比分析**

对16个开放权重单元和12个闭源单元进行50次种子评估；结果显示在同一模型下跨数据集的排名转移强（ρ最高0.969），跨模型或闭源时转移弱（平均ρ≈0.08）。密度/探针方法在开放权重中最稳定，口头化与CCP在API-only中表现最佳。共形点击盘可将半径缩小40–60%，但覆盖率需在不同接口/校准集下检验；

**⚠️ 局限性**

局限性：仅针对单步点击，采样量低（5次），未提出新的UQ估计器，仅评估后置方法；闭源接口受限导致可用方法减少；未涵盖多步轨迹或更复杂交互场景。

---

## 430. Endeavor: Efficient PairHMM for Detection of DNA Variants in Genome-Scale Datasets

**arXiv ID:** 2606.25738 | [PDF](https://arxiv.org/pdf/2606.25738v1)

**作者:** Miguel Graça `[一作]` (INESC-ID), Aleksandar Ilic `[通讯]` (INESC-ID)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了名为 Endeavor 的行级并行化框架，用于加速 DNA 变异检测中的 PairHMM 算法，支持 CPU（AVX/AVX‑512）和 GPU（CUDA/HIP）两大平台。

**💡 创新点**

创新点包括：① 通过数学推导去除 D 矩阵，重定义 PairHMM 以实现行级细粒度并行；② 在 CPU 侧采用 SIMD + OpenMP；③ 在 GPU 侧利用 warp shuffle 与 inclusive scan 解决行间依赖，实现对长读序列（最高 10^5 bp）可扩展；④ 采用跨平台 SIMD/warp 设计，实现对 NVIDIA 与 AMD GPU 的通用支持。

**🔧 技术方法**

技术手段：SIMD intrinsics（AVX/AVX‑512）、OpenMP、CUDA 与 HIP、warp shuffle / inclusive scan、共享内存、线程块间同步、寄存器/共享内存调度、FP32/FP64 双精度支持。

**📊 数据集**

实验数据集：GIAB（Illumina、SoLiD、BGI‑SEQ500、Ion Torrent、Chromium、PacBio、ONT）七种测序技术；10s 基准集（3550 条序列，最大 263 bp）。

**📈 对比分析**

性能评估：与 GATK HaplotypeCaller（CPU）和 gpuPairHMM（GPU）对比。CPU 上峰值 TCUPS 提升 1.9–2.1×，在完整基因组数据集上 10–25× 的加速；GPU 上 FP32/FP64 的吞吐量比 gpuPairHMM 提升 1–2×，在 H100、RTX 6000、MI300A 等最新设备上实现 2–5× 的总体加速；10s 数据集上 Endveor 以 4–5× 的速度击败所有现有实现。

**⚠️ 局限性**

局限性：① 仍受寄存器压力和共享内存容量限制，极长序列（>10^5 bp）仍需多 warp/块处理；② FP64 在部分 GPU 上吞吐低；③ 需要为不同硬件调优参数 N；④ 目前未针对 FPGA 或极低功耗设备做移植；⑤ 对极短序列的加速效果相对有限。

---

## 431. A Stochastic Epidemiological Model of Latent Tuberculosis in a Radiation Exposed Mars Colony

**arXiv ID:** 2606.25728 | [PDF](https://arxiv.org/pdf/2606.25728v1)

**作者:** Teddy Lazebnik `[一作]` `[通讯]` (University of Haifa), Teddy Lazebnik (University of Haifa)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文构建了一个随机化的宿主-辐射-病原-环境模型，用来模拟火星殖民地中潜伏结核的再激活及其传播，并在此基础上研究了隔离与药物治疗的动态资源分配。

**💡 创新点**

创新点在于将宇宙辐射导致的免疫抑制机制与潜伏结核再激活耦合，首次在小型封闭人群中使用部分可观测强化学习（PPO）来自动化制定隔离与治疗策略。

**🔧 技术方法**

技术手段包括：分层的混合微分方程与代理基础仿真相结合的模型、Proximal Policy Optimization（PPO）在部分可观测MDP中的学习，以及基于Monte Carlo的敏感性与情景分析。

**📊 数据集**

数据来源主要是文献给出的结核传播参数、辐射暴露率以及假设的监测误差率，实验通过 1000+ 次随机模拟验证，未使用真实火星或太空医疗数据。

**📈 对比分析**

与无干预、仅隔离、仅药物、固定组合四种基线策略比较，PPO 策略在降低死亡率、感染负担方面位于中间位置，但在资源消耗和整体目标函数上比固定组合更优，整体目标比无干预显著下降。

**⚠️ 局限性**

局限性包括：免疫系统仅用单一变量近似，未考虑个体差异、空间结构和药物耐药；参数多为假设或间接校准；PPO 策略可能受到模拟环境奖励与观测设计的影响。

---

## 432. DeformGen: Dynamics-Based Topology Augmentation for Deformable Manipulation Policy Learning

**arXiv ID:** 2606.25939 | [PDF](https://arxiv.org/pdf/2606.25939v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 433. Bridging Spherical Black-Box Optimizers

**arXiv ID:** 2606.25761 | [PDF](https://arxiv.org/pdf/2606.25761v1)

**作者:** Johannes Ackermann `[一作]` (University of Tokyo), Stefano Peluchetti `[通讯]` (Sakana AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种结合进化策略与梯度优化的混合强化学习框架，用于解决MuJoCo环境中的连续控制任务。

**💡 创新点**

创新点在于将种群演化与梯度信息融合，形成自适应的学习率与目标函数，显著提升收敛速度与最终性能。

**🔧 技术方法**

采用了进化策略、随机梯度下降、政策梯度等技术，并在深度网络上进行结构与超参数调优。

**📊 数据集**

使用OpenAI Gym的MuJoCo数据集，包括HalfCheetah、Hopper、Ant、Humanoid、BipedalWalker、Reacher等任务。

**📈 对比分析**

与PPO、A3C、DDPG等主流方法对比，实验显示在大多数任务上达到更高奖励并收敛更快，整体性能优于传统方法。

**⚠️ 局限性**

局限性主要体现在计算成本高、对环境噪声敏感以及在极高维度任务中的扩展性不足。

---

## 434. Can Machine Learning Break Wi-Fi Privacy? A Study on MAC Address Randomization

**arXiv ID:** 2606.25788 | [PDF](https://arxiv.org/pdf/2606.25788v1)

**作者:** Marta Puig `[一作]` (Universitat Pompeu Fabra), Francesc Wilhelmi `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了Wi‑Fi MAC地址随机化在Probe Request阶段的脆弱性，利用机器学习聚类技术评估被动攻击者的跟踪能力。

**💡 创新点**

创新点在于将HT能力字段按位分解为子字段，结合IFAT时序特征和多探针RSSI，构建更细粒度的特征空间，从而在无加密帧中实现高精度设备识别。

**🔧 技术方法**

主要技术包括无监督聚类算法（K‑Means、DBSCAN、OPTICS）、IFAT时序特征、HT能力子字段分解、RSSI模拟、以及Hungarian算法用于聚类标签映射。

**📊 数据集**

使用了22台不同厂商设备在Mode S下的20分钟Probe Request捕获数据集，并通过模拟RSSI补全空间信息。

**📈 对比分析**

在三种空间场景（无RSSI、单探针、三探针）和多种特征配置下对三种聚类算法进行比较，DBSCAN在22设备情形下实现全局准确率89.6%，显著优于K‑Means和OPTICS；IFAT对DBSCAN有负面影响。

**⚠️ 局限性**

局限性包括设备帧数不均衡导致稀疏设备误判、多种HT能力签名使单设备分散、仅使用模拟RSSI缺乏真实环境验证，以及未对新的Wi‑Fi标准（VHT/HE/EHT）进行评估。

---

## 435. Do (Not) Tell Me About My Insecurities: Assessing the Status Quo of Coordinated Vulnerability Disclosure in Germany Amid New EU Cybersecurity Regulations

**arXiv ID:** 2606.25950 | [PDF](https://arxiv.org/pdf/2606.25950v1)

**作者:** Sebastian Neef `[一作]` (Technische Universität Berlin), Anne Hennig `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文对德国DAX指数公司在2023–2025年间的协调漏洞披露（CVD）程序和security.txt文件的采用情况进行了纵向调查，并通过两轮问卷收集公司对CVD实施经验、动机与挑战的反馈。

**💡 创新点**

创新点在于：①首次系统性收集并跟踪德国上市公司CVD实践的演变；②结合公开信息和直接访谈，揭示法律驱动（NIS2、CRA）与资源不足是主要推动与阻碍因素；③提供多渠道联系（邮件、邮寄、新闻稿）对比响应率的经验，为后续大规模安全研究提供方法学借鉴。

**🔧 技术方法**

主要技术手段包括：自动抓取公司官网和搜索引擎获取security.txt与CVD信息；基于RFC 9116标准检查文件合法性；邮件和邮寄问卷的多阶段跟踪；定性与定量结合的问卷分析。

**📊 数据集**

使用的数据集为：①40家DAX公司（含2025年新增公司）的公开网站信息；②2023年与2025年两次发出的问卷回复（共8份有效回答）。

**📈 对比分析**

方法比较上，作者没有使用机器学习或基准模型，而是采用描述性统计与简易主题归纳；在安全实践方面，CVD和security.txt的采用率从2023年的50%提升至2025年的92.5%，表明法律与企业治理驱动效果显著。

**⚠️ 局限性**

主要限制包括：样本仅限40家大型上市公司，易推广性受限；问卷响应率低（20%），导致经验性结论不具统计显著性；信息抓取受搜索引擎算法与手工误差影响；多阶段联系导致部分企业重复或被遗漏。

---

## 436. Overview of HIPE-2026: Person-Place Relation Extraction from Multilingual Historical Texts

**arXiv ID:** 2606.25935 | [PDF](https://arxiv.org/pdf/2606.25935v1)

**作者:** Juri Opitz `[一作]` (University of Zurich), Simon Clematide `[通讯]` (University of Zurich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在HIPE‑2026共享任务中，作者针对多语言历史文本设计并评估了基于文档级别的时间感知人物-地点关系抽取任务；

**💡 创新点**

创新点包括：①将关系划分为三类（explicit、plausible、none）以实现证据敏感抽取；②提出三维评估框架（准确度、效率、泛化），从而在大规模文化遗产文本处理中兼顾性能与资源消耗；③通过“surprise”文学域检验跨域迁移能力。

**🔧 技术方法**

使用了多种技术路线，包括：大规模语言模型的提示式推理与监督微调、参数高效微调（LoRA/QLoRA）、多语言Transformer编码器（XLM‑RoBERTa、DeBERTa、Qwen、Llama‑Family）、图结构特征提取与轻量级分类器、规则与知识图（Wikidata）增强。

**📊 数据集**

实验数据集为19‑20世纪德、英、法语报纸文本（Domain A）以及16‑18世纪法语文学文本（Domain B），共包含约1,800篇报纸文章和30篇文学文章，标注了人名、地点及三类关系。

**📈 对比分析**

在Test A准确度榜单中，Spinfo取得最高综合宏平均召回（0.748），MaxFo‑Ajie在泛化榜单上以0.816宏召回领先；在效率榜单中，MILRIT因模型小、参数少而排名第一。整体来看，提示式LLM与微调Transformer在准确度上占优，而基于特征的轻量级方法在效率上更具优势。

**⚠️ 局限性**

主要局限包括：①区分explicit与plausible标签仍具有高误差；②对时间推理的依赖导致在含有复杂句法或跨句关系时表现不佳；③跨域泛化仍有限，尤其在古典文学语料中性能显著下降；④大模型对计算资源需求高，限制了在大规模语料库上的实际部署。

---

## 437. In-context Region-based Drag: Drag Any Region to Any Shape

**arXiv ID:** 2606.25907 | [PDF](https://arxiv.org/pdf/2606.25907v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 438. OracleAnalyser: Analysing Implicit Semantics of Oracle Bone Scripts through MLLMs with Post-training

**arXiv ID:** 2606.25906 | [PDF](https://arxiv.org/pdf/2606.25906v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 439. The Power of Small Symmetries

**arXiv ID:** 2606.25895 | [PDF](https://arxiv.org/pdf/2606.25895v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

---

## 440. A Tattered Cloak of Invisibility: Measuring Anonymity Loss in Railgun on Ethereum

**arXiv ID:** 2606.25926 | [PDF](https://arxiv.org/pdf/2606.25926v1)

**作者:** Kanan Huseynov `[一作]` (Eötvös Loránd University), János Tapolcai `[通讯]` (Budapest University of Technology and Economics)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文从用户行为角度评估了以太坊隐私层Railgun的实际匿名性，提出并实验了五种基于地址重用、交易图、gas支付者、knapsack匹配和金额指纹的去匿名化启发式方法；

**💡 创新点**

创新点在于首次针对支持任意代币金额、内部交易功能的隐私池Railgun，系统性识别并量化多维行为泄露渠道，并通过动态规划的knapsack求解显著降低匿名度；

**🔧 技术方法**

技术手段包括基于时间窗口的图遍历、交易图关联、金额与数字指纹匹配、Pisinger算法的伪多项式 knapsack 求解以及对交易时序的统计分析；

**📊 数据集**

使用了截至2026年4月的Railgun Layer‑1公开区块链数据，涵盖约34,747笔存款、47,596笔取款、122,477笔存款、328,520笔取款等交易记录；

**📈 对比分析**

实验结果显示五种启发式方法共实现17.65%取款-存款唯一链接，knapsack求解在30天窗口下实现中位数匿名损失3.42比特；相较于仅依赖混合器的理论匿名集，实际有效匿名集显著缩小；

**⚠️ 局限性**

研究局限在于仅分析以太坊主网和ETH资产、仅使用被动区块链数据、未考虑网络层、应用层或交叉链泄露，且未实现主动攻击或机器学习图嵌入等高级手段。

---

## 441. $\text{DT}^2$: Decision-Targeted Digital Twins

**arXiv ID:** 2606.25923 | [PDF](https://arxiv.org/pdf/2606.25923v1)

**作者:** Harry Amad `[一作]` (University of Cambridge), Mihaela van der Schaar `[通讯]` (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种决策目标化数字孪生（DT）训练范式，通过将离线OPE的策略排名融入DT的损失函数，使DT能够更好地为决策支持服务。

**💡 创新点**

创新点在于：①用FQE估计代理策略价值并构造对比排名；②设计可微的平滑Kendall相关损失以监督DT模拟；③理论证明在模型容量有限时传统基于单步误差的DT训练会导致次优策略排序；④通过λ调节实现模拟精度与决策质量的折中。

**🔧 技术方法**

采用的技术包括：离线策略评估（FQE），平滑Kendall相关排序损失（tanh近似），有限长度回滚加上已训练的Q函数的自举估计，及多种DT架构（Transformer、GRU、MLP、ResNet、Neural ODE）与标准模拟损失（NLL、MSE）。

**📊 数据集**

实验使用的环境包括六个MuJoCo连续控制任务（Ant、HalfCheetah、Hopper、Walker2d、Humanoid、Swimmer）以及一个基于医学癌症进展模型的治疗仿真；同时在若干toy任务中验证理论。

**📈 对比分析**

与传统基于模拟误差（NLL/MSE）的DT以及多种MBRL基线（MOReL、MOPO、ROMI、VaGraM、HDTwin）和纯FQE排名进行对比。实验显示，决策目标化DT在大多数环境和架构下在Spearman相关性提高约47%，决策regret降低约54%，仅以约17%的MSE损失换取更好的策略排序；在未见策略上仍保持优势。

**⚠️ 局限性**

局限性包括：依赖FQE的代理排名质量，若覆盖不足会产生偏差；额外的计算开销（需预训练Q和回滚计算）；λ取值敏感，λ→1训练不稳定；在复杂系统中对代理价值的准确性与泛化仍有挑战。

---

## 442. LLM-Based Discovery of Latent Requirements from Stakeholder Conversations: Preliminary Results from Industry

**arXiv ID:** 2606.25867 | [PDF](https://arxiv.org/pdf/2606.25867v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 443. On the Encodability of Reversible Process Calculi

**arXiv ID:** 2606.25916 | [PDF](https://arxiv.org/pdf/2606.25916v1)

**作者:** Ivan Lanese `[一作]` (University of Bologna), Shoji Yuen `[通讯]` (Nagoya University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了可逆进程计算器CCSK在前向（不可逆）并发模型中的可编码性，证明了若干分离定理，并给出了可逆进程在顶层并行约束下到内部π计算器的强bisimilarity编码，以及在任意并行下的弱互相模拟编码；

**💡 创新点**

创新点在于首次系统阐明可逆性对表达力的严格提升：证明无基本成功灵敏编码可将CCSK映射到CCS或π，而在顶层并行可通过内部π实现强bisimilarity；当并行层级任意时，展示只能得到弱互相模拟的编码，揭示了可逆进程与前向并行的根本不匹配；

**🔧 技术方法**

技术上采用了可替换性（replacement freeness）与成功可替换性（SuRF）的理论框架，构造强/弱bisimulation与互相模拟关系；设计了关键映射ϕ和回滚协议（多路树协议）来实现可逆操作的编码；

**📊 数据集**

无数据集，所有结果均为形式化证明与构造性证明；

**📈 对比分析**

比较方法基于行为等价性（强bisimilarity、弱互相模拟）进行理论对比；未涉及实验或性能评估，讨论完全在语义层面；

**⚠️ 局限性**

限制在于编码非组合性、无法在任意并行下保持强bisimilarity，需要弱化行为对应；此外编码仅针对CCSK，推广到其它可逆计算器仍需进一步研究。

---

## 444. Beyond a Shadow of a Doubt: Close Proximity Geometry Reconstruction Using FMCW Radar Shadow Effects

**arXiv ID:** 2606.25829 | [PDF](https://arxiv.org/pdf/2606.25829v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 445. Manipulation Is Task-Dependent: A Multi-Axis, Multi-Environment Evaluation of Frontier LLMs

**arXiv ID:** 2606.25899 | [PDF](https://arxiv.org/pdf/2606.25899v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 446. USS: Unified Spatial-Semantic Prompts for Embodied Visual Tracking with Latent Dynamics Learning

**arXiv ID:** 2606.25880 | [PDF](https://arxiv.org/pdf/2606.25880v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 447. Robustness and Leadership in Markov-switching Consensus Networks

**arXiv ID:** 2606.25888 | [PDF](https://arxiv.org/pdf/2606.25888v1)

**作者:** Sarah H. Cen `[一作]` (Carnegie Mellon University), Naomi Ehrich Leonard `[通讯]` (Princeton University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在Markov切换图（MSG）下的噪声分布式协调问题，即连续/离散时间的无噪声一致性与领导-跟随者跟踪动力学，并给出稳态误差、节点误差和系统误差的解析表达式。

**💡 创新点**

创新点包括：① 将MSG纳入Markov跳跃线性系统（MJLS）框架，推导出稳态协方差闭式；② 将鲁棒性、确定性指数和联合中心性等概念从静态图推广到MSG；③ 对两拓扑切换的MSG给出显式误差表达式，证明切换可降低或不影响误差，且切换频率越高误差越小；④ 提出对噪声MSG的领导选取与节点确定性排序关系的初步分析。

**🔧 技术方法**

技术方法主要是Markov跳跃线性系统理论、拉普拉斯矩阵与其伪逆、Kronecker和Kronecker和、稳态协方差分析、Neumann级数展开以及数值仿真。

**📊 数据集**

文中未使用公开数据集，所有实验均基于人工构造的图（如线图、环图、三图集合）和相应的生成器矩阵，采用数值仿真验证理论。

**📈 对比分析**

通过对比MSG与单一静态图的系统误差、节点确定性指数，利用数值绘图展示切换参数（如转移率、停留率）对性能的影响；实验表明在两图切换情形下，误差总是小于或等于最优单图误差，且更快的切换可进一步提升性能。

**⚠️ 局限性**

局限性包括：① 仅考虑无向、无权且连通的图；② 对噪声矩阵的假设较为简化（如取单位矩阵）；③ 领导选取分析仅在两图切换案例，未给出通用最优算法；④ 解析结果在大规模MSG下求解复杂，实际应用仍需进一步简化或近似。

---

## 448. AI-Assisted Computational Reproducibility on the FABRIC Testbed

**arXiv ID:** 2606.25879 | [PDF](https://arxiv.org/pdf/2606.25879v1)

**作者:** Komal Thareja `[一作]` (RENCI), Michael Zink `[通讯]` (University of Massachusetts)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文利用FABRIC测试床与LLM编码助手（Claude Code via LoomAI）共同重现三大领域的实验（BBR拥塞控制、LAMMPS分子动力学、基因组学管线），并提出四阶段重现方法与结论级评估。

**💡 创新点**

创新点在于：①将可编程实验平台与语言模型协同，实现从论文到可执行工作流的自动化；②引入结果级与结论级双重评估框架；③系统性量化AI在重现流程中的加速效益与局限，并给出可操作的改进建议。

**🔧 技术方法**

使用技术包括：FABRIC的SDK和Weave封装、LoomAI的可视化实验界面、Claude Code（LLM编码助手）、Python、Bash、Snakemake/Nextflow等脚本与容器化环境。

**📊 数据集**

使用的数据集分别是：BBR论文的GitHub/论文PDF中的实验脚本与拓扑；LAMMPS的公开输入文件与编译脚本；基因组学的GEO/GSE244581、GSE312471原始测序数据及其配套代码。

**📈 对比分析**

对比方法：对每个案例先对数值结果进行定量对比，再用结论支持度表评估科学结论是否被重现；性能方面，AI辅助重现的总人工时间比手工约低4–6倍，API成本约34美元。

**⚠️ 局限性**

局限性包括：AI在分析解释、工作流顺序与数据依赖等领域需人工介入；缺失的原始数据/未完整提交的工作流导致某些结论无法验证；重现成功高度依赖实验材料的完整性与工作流规范化。

---

## 449. Naturalness Predicts but Does Not Cause Transferability in Image Encodings of Real-World Streams

**arXiv ID:** 2606.25844 | [PDF](https://arxiv.org/pdf/2606.25844v1)

**作者:** Faruk Alpay `[一作]` (Bahçeşehir University), Baris Basaran `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文将一维时序信号转换为图像，探究其视觉自然度与冻结视觉骨干迁移性能的关系。

**💡 创新点**

提出可逆光谱编码器并通过谱指数与相位扰动两种对照实验，分离自然度与局部结构对迁移性能的影响。

**🔧 技术方法**

使用多种时序图像编码（线图、谱图、GASF/MTF、递归图等）、可逆光谱编码、冻结预训练骨干、线性探针、端到端微调以及Fréchet Inception Distance（FID）与相位扰动分析。

**📊 数据集**

构建WorldStream语料库，包含来自天气、空气质量、地震、金融、加密货币、外汇、网络关注等多源公共API的实时时序数据。

**📈 对比分析**

在六种冻结骨干和七种编码上进行线性探针实验，发现冻结性能与FID正相关，且微调后仍存在显著性能差距。

**⚠️ 局限性**

实验仅针对冻结模型与线性探针，未考察更复杂模型或自适应编码；所提出的编码缺乏局部结构，导致迁移性能受限。

---

## 450. The Web4 Agent Economy: A Large-Scale Empirical Study of the Landscape, Challenges, and Opportunities

**arXiv ID:** 2606.25876 | [PDF](https://arxiv.org/pdf/2606.25876v1)

**作者:** Yuhan Jin `[一作]` (Zhejiang University), Jiachi Chen `[通讯]` (Zhejiang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对 Web4 生态进行大规模实证研究，分析代理部署、机器对机器支付、链上功能，并挖掘 GitHub 讨论中的工程挑战与社区响应。

**💡 创新点**

首次量化 Web4 代理经济规模与支付生态，揭示身份验证、跨环境运维和支付互操作性三大瓶颈，并为后续标准化与工具化提供数据驱动的研究方向。

**🔧 技术方法**

利用 EIP‑8004 身份注册、x402 微支付协议、Model Context Protocol (MCP) 标准，以及链上日志解析、代码静态分析、开源项目卡片分类、GitHub issue 语义挖掘等技术。

**📊 数据集**

99,448 条多链身份注册记录、317,596,323 条交易日志、341 个 MCP 开源项目源码、349 条 GitHub issue 数据。

**📈 对比分析**

通过链上交易量对比与代码级实现信号匹配，展示了数百万笔每日微支付与 2,228 条代码实现；在 issue 分析中量化身份/授权、跨环境与支付互操作性问题的出现频率，并评估社区响应闭环率（身份 69.13%、跨环境 50.00%、支付 25.29%）。

**⚠️ 局限性**

局限在于仅基于公开讨论的报告（未覆盖所有真实故障）、样本受高活跃仓库影响、关键词过滤的召回与精度折中，以及闭合状态未必等同于实际修复效果。

---

## 451. A Stronger Conditional Running-Time Lower Bound for Global Label Min-Cut

**arXiv ID:** 2606.25875 | [PDF](https://arxiv.org/pdf/2606.25875v1)

**作者:** Yuanhao Wang `[一作]`, Wei Wang `[通讯]` (Xi'an Jiaotong University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种确定性约简，增强了在指数时间假设（ETH）下的条件运行时间下界，证明了在无向边标记图上，无法在时间 (np)^o(log n/loglog n) 内解决该问题。

**💡 创新点**

创新点在于通过引入指向的分区析取组合，提供了一种模块化工具来实现完整的约简，并且去除了之前已知下界中的一个 loglog n 因子。

**🔧 技术方法**

使用了指向的分区析取组合、类精确选择验证器和团一致性验证器等技术。

**📊 数据集**

使用了稀疏 3-SAT 的实例作为数据集，并构造了平衡的多色团实例。

**📈 对比分析**

与之前的研究相比，本文的性能在于提供了更强的条件下界，表明在无向边标记图上，无法在时间 (np)^o(log n/loglog n) 内解决该问题，且即使输入图被限制为简单图，该结论依然成立。

**⚠️ 局限性**

限制在于该结果依赖于指数时间假设（ETH），并且在特定的图结构上可能不适用。

---

## 452. Color Matters: Trigger Color Affects Success in Federated Backdoor Attacks

**arXiv ID:** 2606.25858 | [PDF](https://arxiv.org/pdf/2606.25858v1)

**作者:** Kavindu Herath `[一作]` (Purdue University), Saurabh Bagchi `[通讯]` (Purdue University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在联邦学习中研究了语义触发器的颜色对后门攻击成功率的影响

**💡 创新点**

首次系统隔离颜色变量，展示相同触发器形状与位置下颜色差异可显著改变攻击效果，并揭示颜色与目标类色彩匹配提升攻击成功率

**🔧 技术方法**

使用基于SABLE的语义后门框架、MultiKrum鲁棒聚合、ResNet18模型、SGD优化、MGIE图像编辑模型

**📊 数据集**

CelebA四类发色分类任务（黑、金、棕、灰）

**📈 对比分析**

对比标准攻击与SABLE+防御两种设置，白色触发器在攻击金色目标时ASR提升约7%；黑色触发器在攻击黑色目标时ASR提升约5%；在两种设置下均保持80%+准确率，表明颜色对攻击成功率影响显著

**⚠️ 局限性**

实验仅涵盖两种触发器（口罩、太阳镜）和两种颜色（黑、白），未探索更丰富颜色空间、不同数据集或模型体系，缺乏对颜色对不同鲁棒聚合策略的普适性验证

---

## 453. Automated Detection of Configuration-Specific Security Vulnerabilities via Patch Analysis

**arXiv ID:** 2606.25863 | [PDF](https://arxiv.org/pdf/2606.25863v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 454. Furthest Pair Requires Quadratic Time in Superconstant Dimension under SETH

**arXiv ID:** 2606.25887 | [PDF](https://arxiv.org/pdf/2606.25887v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

---

## 455. Towards an Interactive Evidence-RAG Peer-Review Workspace for the Journal of Digital History

**arXiv ID:** 2606.25837 | [PDF](https://arxiv.org/pdf/2606.25837v1)

**作者:** Elisabeth Guerard `[一作]`, Mirjam Pfeiffer `[通讯]` (Luxembourg Centre for Contemporary and Digital History University of Luxembourg)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个交互式 Evidence‑RAG 工作空间，旨在将评审意见与论文检索证据关联，使 AI 辅助的同行评审支持更可审计、可透明，并保持编辑最终决策权。

**💡 创新点**

创新点在于：①将评审评论拆解为可检索单元；②使用证据‑边界 RAG 提示，让模型只看到检索到的段落；③为模型输出构建可解释的置信分量（检索强度、模型支持、证据特异性、章节覆盖）；④提供专门的编辑界面，编辑者可单独存储决策而非接受模型决定。

**🔧 技术方法**

技术实现包括：BERT‑style 文本向量化（BAAI/bge‑large‑en‑v1.5）、Qdrant 向量数据库、Claude‑Qwen LLM 进行证据‑边界评估、复现可追溯的多阶段流水线（转换、分块、检索、评估）、以及基于组件的置信评分公式。

**📊 数据集**

使用数据集：Journal of Digital History 的真实投稿与评审注释，共 3 篇论文 80 条编辑决策；对应的 Claude‑Qwen audit JSON 文件作为模型输出；检索范围为论文各章节的语义块。

**📈 对比分析**

比较方法：以编辑人工标注为金标准，计算严格准确率（70.0%）、正确或大体正确率（86.2%）和有用率（90.0%）；按论文级别进一步细分，呈现不同评论类型的性能差异，表明系统在细粒度支持上效果尚可。

**⚠️ 局限性**

局限性包括：置信评分是启发式且未经过系统评估；检索与分块在 PDF 结构不规则时易失效；检索证据可能不完整；目前仅在有限样本上测试，未覆盖商业模型或大规模数据，原型不可直接用于自动化评审。

---

## 456. A 3D-Printable Dataset for Fair Testing and Comparisons of Tactile Sensors

**arXiv ID:** 2606.25886 | [PDF](https://arxiv.org/pdf/2606.25886v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 457. An Analysis of Posterior Collapse, Parameterization and Initialization in Variational Deep Gaussian Processes

**arXiv ID:** 2606.25882 | [PDF](https://arxiv.org/pdf/2606.25882v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 458. AI Snitches Get Glitches: Towards Evading Agentic Surveillance

**arXiv ID:** 2606.25836 | [PDF](https://arxiv.org/pdf/2606.25836v1)

**作者:** Hyejun Jeong `[一作]` (University of Massachusetts Amherst), Eugene Bagdasarian `[通讯]` (University of Massachusetts Amherst)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究AI代理的监控行为，提出agentic surveillance概念并评估模型；

**💡 创新点**

首次量化代理无指令监控与逆向监控，并设计逃避技术；

**🔧 技术方法**

使用ReAct agent框架、DSPy文本优化、prompt injection以及LLM系统提示等技术；

**📊 数据集**

创建名为SurveilBench的303个情景数据集，涵盖企业、教育、警察三大域；

**📈 对比分析**

在10种LLM上评估，优化后报告率提升至75–97%，逃避攻击可抑制约10%报告；

**⚠️ 局限性**

缺乏真实部署环境验证，逆向监控识别不完善，逃避手段受限于输入提示，难以对更复杂情境进行普适防护。

---

## 459. Graph it first! Enabling Reasoning on Long-form Egocentric Videos through Scene Graphs

**arXiv ID:** 2606.25842 | [PDF](https://arxiv.org/pdf/2606.25842v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 460. MiniOpt: Reasoning to Model and Solve General Optimization Problems with Limited Resources

**arXiv ID:** 2606.25832 | [PDF](https://arxiv.org/pdf/2606.25832v1)

**作者:** Ke Zhao `[一作]` (East China Normal University), Yang Yu `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MiniOpt框架，训练小参数LLM以解决优化问题

**💡 创新点**

引入推理-建模-求解范式、可验证奖励OptReward及改进的OptGRPO算法

**🔧 技术方法**

强化学习（RLVR）、五元组Chain-of-Thought、Pyomo求解器、梯度策略优化

**📊 数据集**

8个工业优化基准（NL4Opt、Mamo、IndustryOR、NLP4LP、ComplexOR、OptiBench、ICML Competition）及OptMATH

**📈 对比分析**

与多种通用、提示式、学习式模型对比，MiniOpt-7B在8个基准上平均SA达64.76%，3B版本已超多提示式模型

**⚠️ 局限性**

受限于模型规模和维度，1.5B版本性能下降，且对极高维度或大规模问题仍表现有限

---

## 461. Enhancing Brain MRI Anomaly Detection and Reasoning with ROI Rethink and Synthetic Data

**arXiv ID:** 2606.25894 | [PDF](https://arxiv.org/pdf/2606.25894v1)

**作者:** Shangkun Li `[一作]` (Fudan University), Yuanyuan Wang `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发了 BrReMark 框架，在脑 MRI 诊断中引入 ROI 标记与假设-验证的两轮对话，使模型输出可审计且支持开放式推理。

**💡 创新点**

创新点在于：①引入显式的标记-重思（Mark‑and‑Rethink）对话流程；②将监督微调与基于 GRPO 的强化学习结合；③通过阶段化合成病灶注入提升对离谱分布（OOD）的鲁棒性。

**🔧 技术方法**

技术手段包括 Gemini 3.0 Flash 作为生成模型、Lingshu‑32B 作为评判者、SynthSeg 生成合成病灶、GRPO 强化学习、构造多组件奖励（定位、语义、临床安全）以及双代理生成‑校正架构。

**📊 数据集**

使用 BraTS、UPENN‑GBM、ATLAS、ISLES、IXI 等约 3,700 张病例构建 SFT/RL 训练集，并在 OOD 评估中采用 NOVA 数据集；合成病灶通过 SynthSeg 注入真实病灶掩膜。

**📈 对比分析**

与多种通用与医学领域 VLM 进行对比，BrReMark 在 ID 任务中 mAP_50 从 0.74% 提升至 37.54%，诊断准确率达 45.26%；在 OOD NOVA 任务上误报率下降 45.7%，整体性能优于同等 7B 参数模型。

**⚠️ 局限性**

局限性包括：仅使用 2D 切片，缺乏完整 3D 体素推理；合成病灶主要用于定位而非诊断语义；缺乏临床前试验验证。

---

## 462. Lyapunov Optimization based Queue-aware Traffic Shaping for 5G-TSN in Industrial Environments

**arXiv ID:** 2606.25823 | [PDF](https://arxiv.org/pdf/2606.25823v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 463. The Dependency Black Hole

**arXiv ID:** 2606.25949 | [PDF](https://arxiv.org/pdf/2606.25949v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 464. Explainable Control Framework (XCF) based on Fuzzy Model-Agnostic Explanation and LLM Agent-Supported Interface

**arXiv ID:** 2606.25941 | [PDF](https://arxiv.org/pdf/2606.25941v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 465. SurgAtlas: A Large-Scale Surgical Video-Language Dataset with 2,391 Hours of Open and Minimally Invasive Surgery

**arXiv ID:** 2606.25905 | [PDF](https://arxiv.org/pdf/2606.25905v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 466. TacVerse: A Multi-Sensor Dataset and Benchmark for Cross-Sensor Vision-Based Tactile Perception

**arXiv ID:** 2606.25877 | [PDF](https://arxiv.org/pdf/2606.25877v1)

**作者:** Lan Wei `[一作]` (Imperial College London), Dandan Zhang `[通讯]` (Imperial College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了TacVerse数据集和基准，包含七种视觉触觉传感器的106,800张图像，支持形状分类、纹理（grating）分类和力回归三类任务。

**💡 创新点**

创新点在于统一的跨传感器评测协议：within‑sensor、zero‑shot cross‑sensor、few‑shot adaptation，以及使用Masked Autoencoder（MAE）自监督预训练验证跨任务共享表示的有效性。

**🔧 技术方法**

采集视觉触觉数据后使用ViT与ResNet等网络结构，并对比随机初始化、ImageNet预训练与MAE预训练的表现，针对分类与回归任务进行训练与微调。

**📊 数据集**

使用TacVerse自建数据集，涵盖GelSightNoMarker、GelSightMarker、MagicGripper、MagicTac、TacTip、ViTac、ViTacTip七种VBTS；在此数据集上进行实验。

**📈 对比分析**

通过对比within‑sensor与zero‑shot cross‑sensor的准确率/误差，发现形状分类鲁棒性高，而纹理分类与力回归对传感器差异极为敏感；MAE预训练在大多数任务与传感器上均获得显著提升；few‑shot适配可部分缩小力回归跨传感器性能差距。

**⚠️ 局限性**

局限在于仅涵盖实验室固定交互、固定源评估，缺乏动态交互和更广泛的传感器/任务覆盖；未与最新触觉基础模型进行全面对比，仍需进一步扩展和验证。

---

## 467. Proceedings of the 16th International Workshop on Non-Classical Models of Automata and Applications

**arXiv ID:** 2606.25881 | [PDF](https://arxiv.org/pdf/2606.25881v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

---

## 468. Edges Before Embeddings: A Confidence-Aware Blur Gate for Vision-Language Pipelines

**arXiv ID:** 2606.25838 | [PDF](https://arxiv.org/pdf/2606.25838v1)

**作者:** Duy Tran Thanh `[一作]` `[通讯]` (OneMount Group), Duy Tran Thanh (OneMount Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量级、CPU 友好的模糊检测门控模型，能够在上传或前端检测图像模糊并路由至后端 OCR/VLM 或退回用户。

**💡 创新点**

创新点在于引入 Edge Prior Module（拉普拉斯幅值辅助通道）和多尺度 TTA，并通过基于阈值的自适应路由，在同等环境下 F1 提升 1.3 点，证明分辨率是主导因素。

**🔧 技术方法**

使用 MobileNetV3‑Large 轻量化网络、拉普拉斯卷积辅助通道、最大 softmax 阈值决策、5 级分辨率多尺度推理以及 ONNX 运行时部署。

**📊 数据集**

以 GoPro Large 运动模糊数据集为主，使用其配对的清晰/模糊帧进行训练和评估。

**📈 对比分析**

与原始 384px 单尺度基线相比，加入 EPM 并做 5 级 TTA 在 AMD/ROCm 环境下 F1 由 0.9749 提升至 0.9803，准确率、精确率和召回率亦同步提升，延迟约 7 ms/图像。

**⚠️ 局限性**

仅在单一运动模糊分布上评估；未覆盖失焦、低光噪声、压缩失真等；阈值 τ 需要在生产流量上手工调校；未提供多种数据集的泛化评估与可靠性校准。

---

## 469. Sharp approximate Carathéodory theorem and application to iterated Delaunay refinement

**arXiv ID:** 2606.25854 | [PDF](https://arxiv.org/pdf/2606.25854v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

---

## 470. Semantic Consistency Policy Optimization for Reinforcement Learning of LLM Agents

**arXiv ID:** 2606.25852 | [PDF](https://arxiv.org/pdf/2606.25852v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 471. Interference-Aware Cross-Application Placement: A Multi-Objective Optimization Approach for Microservice Clusters

**arXiv ID:** 2606.25922 | [PDF](https://arxiv.org/pdf/2606.25922v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 472. Restoring Incentive Compatibility in Two-Stage Energy Markets with Prosumers

**arXiv ID:** 2606.25910 | [PDF](https://arxiv.org/pdf/2606.25910v1)

**作者:** Nikolas Koumpis `[一作]` (Yale University), Manolis Zampetakis `[通讯]` (Yale University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了在两阶段能源市场（日前市场与实时调度）中，主动消费者（prosumers）通过在日前市场中低报需求来获取实时市场利润的策略性行为，并提出一种基于对比式严格合适打分规则的惩罚机制，以恢复参与者的激励相容性并降低不诚实报盘导致的系统失衡与成本；

**💡 创新点**

创新点包括：① 明确区分普通消费者与拥有实时发电资产的prosumers的报盘激励差异，并证明在标准机制下，普通消费者的误报激励可随市场规模消失，而prosumers的误报激励却保持正下界；② 设计了一种利用对比式打分规则的留一分数惩罚，能够在仅基于可观测的市场数据（报告、总需求、历史数据）的约束下对prosumers的误报进行激励抑制；③ 提出了对该机制的理论分析，证明通过调节打分规则的强凸性参数可将误报收益压至可接受的上界；

**🔧 技术方法**

采用的技术主要包括：对两阶段市场的数学建模（线性效用、双向优化）、均衡分析、贝叶斯激励相容性框架、严格合适打分规则（Bregman散度）以及对抗式留一分数设计；

**📊 数据集**

实验使用了两套数据：一套为合成市场数据，用于验证理论中的三类定性现象；另一套为真实欧洲电力市场（Hungary、Romania、Slovakia等国）的日前/实时负荷与价格数据，用以评估机制在实际市场环境下的效果；

**📈 对比分析**

通过与传统机制（仅使用统一定价的日前与实时清算）对比，新的打分规则机制在保持低误报激励（对消费者）和显著降低prosumers误报收益的同时，对诚实参与者的额外费用保持可控（随凸性参数调节），实验显示该机制在大规模市场中实现了近似激励相容且系统失衡成本下降；

**⚠️ 局限性**

限制包括：① 机制假设参与者可获得的公共信息（如总需求、历史报告）足够，且未考虑个体计量误差与测量错误；② 对比式惩罚需要估计“无报盘”情形下的分布，若市场规模变化或参与者行为模式改变，估计误差可能影响激励效果；③ 需要调节凸性参数以权衡误报抑制与诚实参与者费用，参数选择对实际部署具有一定挑战性。

---

## 473. FunPiQ: A New Benchmark for Pixel-Level Quality Assessment in Fundus Images

**arXiv ID:** 2606.25915 | [PDF](https://arxiv.org/pdf/2606.25915v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 474. Variational Autoencoder Layer

**arXiv ID:** 2606.25900 | [PDF](https://arxiv.org/pdf/2606.25900v1)

**作者:** Gananath R `[一作]` `[通讯]`, Gananath R

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出将变分自编码器（VAE）改造成神经网络层（VAELinear），并在多层网络中采用非反向传播的Forward‑Forward Contrastive Learning（IFFCL）进行训练。

**💡 创新点**

创新点在于把VAE视为可嵌入的层而非完整模型，并结合IFFCL实现无梯度跨层训练，同时使用简化的多模态VAE（MVAE）目标与身份解码器。

**🔧 技术方法**

采用的技术包括VAE与重参数化技巧、IFFCL对比学习、MVAE联合ELBO目标、Adam优化器、批归一化、tanh激活及softmax分类。

**📊 数据集**

实验数据集为MNIST训练集与测试集。

**📈 对比分析**

通过独立训练每层并记录训练损失和准确率进行对比，最终测试准确率仅约55%，明显低于当前最先进模型。

**⚠️ 局限性**

局限性包括性能仅达55%准确率、无反向传播限制了优化效果、身份解码器削弱了生成能力以及对更大规模数据和多模态学习的适用性不足。

---

## 475. Paths and Intersections: Recognizing Outerplanar Metrics

**arXiv ID:** 2606.25827 | [PDF](https://arxiv.org/pdf/2606.25827v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 476. Parameterized Complexity of Power Network Design: Coordinating Cable Placement is Hard

**arXiv ID:** 2606.25859 | [PDF](https://arxiv.org/pdf/2606.25859v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 477. RoboAtlas: Contextual Active SLAM

**arXiv ID:** 2606.26046 | [PDF](https://arxiv.org/pdf/2606.26046v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 478. Detect, Unlearn, Restore: Defending Text Summarization Models Against Data Poisoning

**arXiv ID:** 2606.26036 | [PDF](https://arxiv.org/pdf/2606.26036v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 479. AutoRelAnnotator: Calibrated Model Cascades for Cost-Efficient Relevance Evaluation in Sponsored Search

**arXiv ID:** 2606.25871 | [PDF](https://arxiv.org/pdf/2606.25871v1)

**作者:** Md Omar Faruk Rokon `[一作]` (Walmart Global Tech), Kuang-chih Lee `[通讯]` (Walmart Global Tech)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套基于校准的模型级联，用于离线高质量相关性标注。

**💡 创新点**

将域特定微调与逐级校准相结合，实现准确度与成本的正交优化，并引入每类等距校准提升路由精度。

**🔧 技术方法**

使用领域微调（Cross‑Encoder、Gemma‑2B、LLaMA‑8B+LoRA）、等距回归校准、级联推理与多数投票。

**📊 数据集**

采用1.2M条赞助搜索问答对进行微调，100K条标注验证集进行评估。

**📈 对比分析**

与GPT‑4/Claude‑3以及单模型基线对比，微调模型准确率达87–89%，级联在不降低准确率的前提下将成本降低50%，系统整体准确率为89.1%。

**⚠️ 局限性**

仅在赞助搜索域验证；需要大量标注数据与频繁校准；模型更新与维护需人工干预。

---

## 480. Can Trustless Agents Be Trusted? An Empirical Study of the ERC-8004 Decentralized AI Agent Ecosystem

**arXiv ID:** 2606.26028 | [PDF](https://arxiv.org/pdf/2606.26028v1)

**作者:** Xihan Xiong `[一作]` (Imperial College London), Zhipeng Wang `[通讯]` (University of Manchester)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对ERC-8004协议在以太坊、BNB Smart Chain和Base三条链上进行跨链实证研究，分析身份注册、活跃度及声誉市场。

**💡 创新点**

首次系统性量化ERC-8004的身份活跃率、声誉质量与Sybil操纵成本，揭示协议设计缺陷并提出改进建议。

**🔧 技术方法**

结合链上事件抓取、离线文件解析、gas成本测算、x402支付追踪和Sybil检测的资金传播图分析。

**📊 数据集**

收集了2026年5月13日前ERC-8004的所有身份与声誉事件、对应注册文件与反馈文件、交易费以及x402支付记录。

**📈 对比分析**

通过统计分布、Gini系数、批量注册比例、平均反馈数、平均费用等指标，发现声誉均值易被单条反馈操纵，Sybil比例超过70%，成本极低。

**⚠️ 局限性**

仅覆盖三条链且未观察到Validation Registry，且基于公开数据无法完全验证交互真实性。

---

## 481. MIMFlow: Integrating Masked Image Modeling with Normalizing Flows for End-to-End Image Generation

**arXiv ID:** 2606.26016 | [PDF](https://arxiv.org/pdf/2606.26016v1)

**作者:** Yang Chen `[一作]` (Nanjing University), Limin Wang `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出MIMFlow，一种将Masked Image Modeling与Normalizing Flows统一到端到端生成框架；

**💡 创新点**

通过可学习的token化掩码瓶颈实现语义建模与高频纹理合成的分离，缓解NFs的容量瓶颈；

**🔧 技术方法**

结合Transformer Encoder（MAETok）、可学习查询Token、正态分布加噪的Latent NF（STARFlow）和专用Decoder，采用联合变分推理、辅助语义监督与GAN细化；

**📊 数据集**

ImageNet 256×256；

**📈 对比分析**

与现有NF、Diffusion和AR模型对比，MIMFlow-L在相同参数（≈482M）下取得FID 2.50，线性探测精度71.3%，比SimFlow-L提升32.8%且仅使用128个Token；

**⚠️ 局限性**

对极端高频细节仍有不足，需进一步提升生成的纹理真实度，且在极大规模数据或更高分辨率下的可扩展性待验证。

---

## 482. Hierarchical Reinforcement Learning for Neural Network Compression (HiReLC): Pruning and Quantization

**arXiv ID:** 2606.26002 | [PDF](https://arxiv.org/pdf/2606.26002v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 483. The Tatoxa System for Text Detoxification in Low-Resource Languages: The Case of Tatar

**arXiv ID:** 2606.26015 | [PDF](https://arxiv.org/pdf/2606.26015v1)

**作者:** Ilseyar Alimova `[一作]` (Skolkovo Institute of Science and Technology), Alexander Panchenko `[通讯]` (Skolkovo Institute of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了Tatoxa系统，用于自动去毒Tatar语文本，并扩充了Tatar去毒数据集；

**💡 创新点**

创新点在于先通过俄语→塔塔语机器翻译生成高质量合成对，利用LoRA多模型集成并在候选生成后进行排名；同时首次系统性展示了跨语言迁移对低资源塔塔语去毒效果的不足；

**🔧 技术方法**

采用NLLB‑200进行俄-塔翻译微调，mT0‑XL（LoRA适配器）进行去毒训练，LaBSE做语义相似度过滤与候选评估，XLM‑R做毒性分类，xCOMET评估流畅度，整体流程包含候选多样化生成与排名；

**📊 数据集**

使用CLEF‑2025 Tatar去毒数据集（新增701条手工标注样本）作为主要评测集，并将多份俄语去毒语料（ru_paradetox、multilingual_paradetox、rudetoxifier、toxic_dvach_detoxified）通过翻译生成塔塔语合成对；还使用15种语言的去毒对进行跨语言迁移实验；

**📈 对比分析**

与词典删除、mT0多语言prompt、Gemini/Claude/DeepSeek等基线以及人类标注基准进行比较，评估指标为STA、SIM、FL及综合J。Tatoxa在CLEF与自建数据集上均以J≈0.695/0.680、STA≈98%/97%取得最高分；相比之下闭源LLM及简单删除法得分显著低于Tatoxa；跨语言迁移实验表明除塔塔语训练外，其他语言（包括俄语）迁移效果大幅下降；

**⚠️ 局限性**

限制包括：仅对约1%参数（LoRA）进行微调，未扩展到其他突厥语种，导致跨语言比较有限；合成语料仍带有翻译噪声；实验中未充分挖掘更大参数空间和更多低资源语言。

---

## 484. FAR-LIO: Enabling High-Speed Autonomy through Fast, Accurate, and Robust LiDAR-Inertial Odometry

**arXiv ID:** 2606.26010 | [PDF](https://arxiv.org/pdf/2606.26010v1)

**作者:** Maximilian Leitenstern `[一作]` (Technical University of Munich), Markus Lienkamp `[通讯]` (Technical University of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并实现了 FAR-LIO，一个基于 CUDA 的实时 LiDAR‑IMU 闭环定位框架，能够在高速、噪声大、动态环境中提供低延迟且精度高的位姿估计。

**💡 创新点**

创新点包括：① 使用基于 cuco::static_map 的 CUDA voxel hashmap 并引入自适应密度，显著加速最近邻搜索和地图更新；② 开发稀疏感知 GICP（SA‑GICP），在稀疏点云中仍能保持高精度；③ 在 EKF 后端加入延迟补偿与上采样策略，使 IMU 与 LiDAR 观测在时序上同步，降低位姿漂移。

**🔧 技术方法**

核心技术：CUDA 加速的 voxel 哈希表、稀疏感知 GICP、基于改进的 Cauchy 内核的最小二乘优化、使用纯动力学 EKF 进行 IMU 融合、延迟补偿与上采样、离散化去畸变（Deskewing）以及自适应局部子图（ASMD）维护。

**📊 数据集**

使用了四种不同 LiDAR‑IMU 组合的数据集：公开的 KITTI、MulRan，以及两套真实比赛数据（Abu Dhabi Yas Marina Circuit、Monza Indy Autonomous Challenge），覆盖住宅、道路与赛道环境，最高车辆速度达 250 km/h。

**📈 对比分析**

与 Fast‑LIO2、D‑LIO、KISS‑ICP、Faster‑LIO 等四种最先进方法在同一硬件平台（Intel Xeon + NVIDIA RTX A5000）和单一参数集下对比。结果显示 FAR‑LIO 在大多数序列中取得最低或次低的绝对定位误差（APE）与相对误差（RPE），平均相对误差下降 6.9%，平均运行时下降 38.4%，并在所有测试序列上成功收敛，显示出优越的实时性与鲁棒性。

**⚠️ 局限性**

局限性：① 仅实现闭环定位（未包含全局地图回环与闭环修正）；② 对 GPU 计算资源依赖较大，CPU 版本性能不及 CUDA 版本；③ 在极端低帧率 IMU 或大范围高速旋转情况下，仍可能出现漂移或收敛失败，需要进一步的 IMU 校准或更高级的后端优化。

---

## 485. Dziri Voicebot: An End-to-End Low-Resource Speech-to-Speech Conversational System for Algerian Dialect

**arXiv ID:** 2606.26003 | [PDF](https://arxiv.org/pdf/2606.26003v1)

**作者:** Dihia Lanasri `[一作]` (ATM Mobilis), Asma Kemmoum `[通讯]` (ATM Mobilis)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了完整的阿尔及利亚方言语音对语音会话系统（DZIRI VoiceBot），将ASR、NLU、检索增强生成（RAG）和TTS整合成统一端到端流水线。

**💡 创新点**

创新点在于：①首次在阿尔及利亚方言上实现端到端的语音会话框架；②针对方言特点构建专属语音、NLU、TTS语料；③结合Whisper、DziriBERT、Llama 3.2 + RAG以及XTTS‑v2/LoRA实现跨任务性能优化。

**🔧 技术方法**

使用的技术包括 Whisper‑medium 微调、Wav2Vec2‑XLS‑R、DziriBERT + Rasa NLU、FAISS检索 + Llama 3.2 生成、XTTS‑v2 + LoRA 与 VITS‑ar 语音合成。

**📊 数据集**

使用的数据集：4,103句、2.68 h 的电信业务语音集（含阿尔及利亚方言与法语混合）、15,891 条 NLU 例子（80 种意图、28 个实体）以及 50.7 min 的单说话人 TTS 语料。

**📈 对比分析**

实验比较：ASR Whisper‑medium 在本域下达到 13.74 % WER（Wav2Vec2‑XLS‑R 为 31.67 %），NLU 98.4 % 意图准确率/93.9 % 实体 F1；TTS XTTS‑v2 + LoRA 的 MOS 为 4.31，显著优于 VITS‑ar（3.12）。

**⚠️ 局限性**

限制包括：语音数据规模仍有限导致泛化受限；推理延迟高（≈16 s/交互），需 GPU/模型压缩；方言代码切换和音素多样性仍易造成识别误差；RAG 需要大量算力与知识库维护。

---

## 486. BlowLive: Blow-Based Multi-Factor Biometrics with Liveness Detection and Revocability

**arXiv ID:** 2606.25998 | [PDF](https://arxiv.org/pdf/2606.25998v1)

**作者:** Eyasu Getahun Chekole `[一作]` (Singapore University of Technology and Design), Jianying Zhou `[通讯]` (Singapore University of Technology and Design)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了BlowLive——一种将手机吹气声学信号与面部图像融合的多因素生物识别框架，并采用模糊提取器（Fuzzy Extractor）生成可撤销的加密密钥，同时配备基于多普勒频移的活体检测模块。

**💡 创新点**

创新点包括：①将行为特征（吹气声学）与生理特征（面部）融合；②使用GFCC+CNN提取吹气特征，FaceNet提取面部特征，二者在特征级与分数级双重融合；③基于模糊提取器实现模板与密钥双重可撤销；④设计针对无声吹气的多普勒频移活体检测，能有效识别回放、合成与深度伪造攻击。

**🔧 技术方法**

核心技术包括：Gammatone Frequency Cepstral Coefficients (GFCC) + 1D CNN；FaceNet（Inception‑ResNet v1）+ 投影层；BCH 码的模糊提取器；多普勒频移分析（带通滤波、STFT、峰值提取、能量包络与混合评分）；多模态融合（特征级与分数级）。

**📊 数据集**

使用由50名参与者在坐立两种姿势下录制的吹气声学信号（采样率48 kHz）和面部图像组成的私有数据集，包含约550条正样本与544条回放样本用于活体检测。数据已公开至指定 GitHub 仓库。

**📈 对比分析**

与现有单模态与多模态生物识别系统对比，BlowLive在行为模态达到99.56%准确率，面部模态100%，分数级融合99.95%，特征级融合100%；活体检测准确率99.42%；在与20余项相关工作（如声学、面部、触觉、惯性等）比较时，BlowLive在准确率、可撤销性、低资源需求、无创性等8项需求上均优于或匹配最佳值，成为唯一同时满足全部8项需求的方案。

**⚠️ 局限性**

局限性包括：①实验样本量仅50人，需在更大规模、多语言、多环境下验证泛化性；②需要在手机麦克风与前置摄像头上同时捕获，某些低端设备可能不足；③吹气动作受用户习惯、环境噪声影响，若无声或异常吹气可能导致误判；④多普勒检测对运动噪声敏感，需在嘈杂环境中进一步优化；⑤在极端恶意合成或定制化深度伪造攻击场景下，仍可能存在边缘攻击风险。

---

## 487. SpeechEQ: Benchmarking Emotional Intelligence Quotient in Socially Aware Voice Conversational Models

**arXiv ID:** 2606.25990 | [PDF](https://arxiv.org/pdf/2606.25990v1)

**作者:** Liang-Yuan Wu `[一作]` (New York University), Hua Shen `[通讯]` (New York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了 SpeechEQ 基准，用来测量语音语言模型在多轮对话中的情感智商

**💡 创新点**

创新点包括：①将 EQ‑i 2.0 框架与语音相结合，生成可验证的 15 个情感子量表；②通过强制选择两段语义相同但语调不同的音频，剥离语义干扰，专注跨模态情感推理；③提出 SEQ 分数，将多轮轨迹准确率标准化为与人类评估相关的情感智商指标；④揭示模型的“模态捷径”“情感扁平化”和“上下文遗忘”等瓶颈

**🔧 技术方法**

采用 LLM 驱动的生成+TTS 合成 pipeline、音频特征对比过滤、两阶段人工验证、基于 TTS 的多轮 forced‑choice 评估、SEQR 计算与归一化

**📊 数据集**

使用 SpeechEQ 自研数据集，共 2,265 条多轮对话（42.37 小时），覆盖 15 个 EQ‑i 2.0 子量表，提供同义文本的双声道音频选项

**📈 对比分析**

与 cascaded pipelines（ASR+SER+文本 LLM）以及多种端到端 SLM（Qwen‑Omni、Kimi‑Audio、MiMo‑Audio、Fun‑Audio、Gemini‑2.5‑Pro、gpt‑audio‑1.5）对比；端到端模型在 Acc_traj 和 SEQ 上显著优于 cascaded，最佳模型 Qwen3‑Omni‑30B 达 58.3% Acc_traj、SEQ≈147；但仍低于人类评估的 75‑80% 区间

**⚠️ 局限性**

主要局限：①模型仍依赖文本信息，跨模态推理不够深；②对安全对齐的过度依赖导致情感表达趋向低激励“安全化”；③多轮上下文记忆衰退，难以维持情感轨迹；④基准仅覆盖 15 个子量表，未探索更细粒度或跨文化差异

---

## 488. Monitoring Discounted Sum Properties

**arXiv ID:** 2606.25979 | [PDF](https://arxiv.org/pdf/2606.25979v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

---

## 489. The Inference-Compute Frontier and a Latency-Efficient Architecture for Limit Order Book Prediction

**arXiv ID:** 2606.25986 | [PDF](https://arxiv.org/pdf/2606.25986v1)

**作者:** C. Evans Hedges `[一作]` `[通讯]` (Independent Researcher), C. Evans Hedges (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了限价订单簿预测中推理计算与预测误差的尺度律前沿，并提出了低延迟的 FastBiNLOB 架构；

**💡 创新点**

创新点在于发现不同模型族之间的计算-误差前沿遵循幂律关系，并将计算与延迟区分为两条独立前沿，同时设计了密集轴可分离的 FastBiNLOB 结构；

**🔧 技术方法**

使用结构性前向工作计数进行幂律拟合、鲁棒性分析以及在 FI‑2010 数据上训练的多种模型（决策树、CatBoost、随机卷积等）和 FastBiNLOB；

**📊 数据集**

采用 FI‑2010 基准数据集，包含 raw LOB40 特征用于尺度分析和全 144 维特征用于部署评估；

**📈 对比分析**

通过在大量模型族上进行交叉验证、保留 MLPLOB 作为 hold‑out，计算 R²≈0.94 的前沿拟合；FastBiNLOB H96 与 H120 在宏 F1 目标上分别以 23.7% 与 60.1% 的延迟提升并在选定时段达到 SOTA；

**⚠️ 局限性**

局限性包括延迟前沿效果弱、结果仅在 FI‑2010 上验证、对特定硬件/实现敏感、未覆盖所有预测时段且未提供理论最优保证。

---

## 490. A Sensorised Lattice Footplate for a Semi-Active Prosthetic Foot

**arXiv ID:** 2606.25966 | [PDF](https://arxiv.org/pdf/2606.25966v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 491. A Benchmark for Heterogeneous Stereo Deblurring with Physically- and Epipolar-constrained Cross Attention

**arXiv ID:** 2606.25962 | [PDF](https://arxiv.org/pdf/2606.25962v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 492. FoleySet: A Multi-Level Human-Annotated Foley Sound Dataset

**arXiv ID:** 2606.25980 | [PDF](https://arxiv.org/pdf/2606.25980v1)

**作者:** Sunshiyu Wang `[一作]` (Georgia Institute of Technology), Alexander Lerch `[通讯]` (Georgia Institute of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含10,000条音频、两级细粒度分类（9大类、73子类）的公开 Foley 音效数据集 FoleySet，并提供了完整的标注流程与基准分类结果。

**💡 创新点**

创新点在于：①首次发布专门面向 Foley 领域、基于行业实践与库资料的可复现两级分类体系；②采用单一标注者的高质量人工标注与严格分割策略，确保数据质量；③为 Foley 研究提供了统一的开源资源和基准。

**🔧 技术方法**

主要技术手段包括：自动化抓取 Freesound CC0 录音、音频清洗与规范化（采样率、位深、响度归一）、时长裁剪与分段；人工完成两级标签与单/多事件标注；基准实验采用 PaSST 预训练音频 Transformer 提取特征，后接线性分类器进行 9 类与 73 类的分类。

**📊 数据集**

使用的数据集是本文自建的 FoleySet，来源于 Freesound 上的 CC0 许可音频；实验中还参考了 DCASE 2023 Task 7、FSD50K 等公开音效数据集用于比较与说明。

**📈 对比分析**

与现有基准相比，FoleySet 在 9 类主要类别分类上达到了 82% 的准确率、0.80 的宏平均 F1；在 73 类子类别分类上准确率 64%，宏平均 F1 0.56。基准使用 PaSST 提取 768 维嵌入，并在小样本、类别不平衡条件下表现稳定。

**⚠️ 局限性**

主要限制包括：①标注全部由单一标注者完成，缺乏多标注者一致性验证；②子类别分布呈长尾，极少样本导致低召回；③缺乏对多模态（视频-音频）同步与生成任务的进一步评估。

---

## 493. AI translation of literary texts is "fine", but readers still prefer human translations

**arXiv ID:** 2606.26040 | [PDF](https://arxiv.org/pdf/2606.26040v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 494. Multi-Agent Goal Recognition with Team- and Goal-Conditioned Reinforcement Learning and Factorized Branch-and-Bound

**arXiv ID:** 2606.25978 | [PDF](https://arxiv.org/pdf/2606.25978v1)

**作者:** Thiago Thomas `[一作]` (Pontifical Catholic University of Rio Grande do Sul), Felipe Meneguzzi `[通讯]` (University of Aberdeen)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为MAGR-BB的模型无关多智能体目标识别器，能够同时推断团队划分与团队目标。

**💡 创新点**

创新点在于使用共享的Transformer策略对团队-目标对进行计分，并结合分层分支限界搜索，在非竞争分数条件下实现高效推理，显著降低完整假设枚举量。

**🔧 技术方法**

采用PPO+CTDE训练的Transformer策略，团队内自回归解码，计分规则为对数似然惩罚，配合分支限界（Branch‑and‑Bound）搜索实现快速推断。

**📊 数据集**

在受控的多智能体Blocksworld环境中测试，该环境包含两个隐藏团队、七块工作空间以及长度为2-4的堆叠目标。

**📈 对比分析**

通过与完整穷举基线和五种变体进行对比，MAGR-BB在保持相同的top‑10准确率的同时，累计运行时间比穷举快约2.4–2.9倍，并在最后一步仅生成10个完整假设。

**⚠️ 局限性**

仅在离散可观测、分离工作空间的理想化环境中验证，未处理资源共享、部分可观测、噪声结构更复杂以及更大团队规模等实际挑战。

---

## 495. Helpful or Harmful? Evaluating LLM-Assisted Vulnerability Patching via a Human Study

**arXiv ID:** 2606.25973 | [PDF](https://arxiv.org/pdf/2606.25973v1)

**作者:** Giulian Biolo `[一作]` (University of Trento), Fabio Massacci `[通讯]` (University of Trento)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开展了一项受控的人类实验，比较LLM辅助与手工修补软件漏洞的时间、效果和开发者感知；

**💡 创新点**

创新点在于采用平衡交叉设计与隐藏的“Ghost Tests”来评估假修补，并系统研究LLM辅助在安全修补中的真实影响；

**🔧 技术方法**

技术手段包括WebApp+Pyodide代码执行平台、LimeSurvey人口学问卷、Supabase数据存储，以及LLM模型（gpt‑4o‑mini、gpt‑5‑mini、Claude Haiku）和统计方法（Cox比例风险、Fisher精确检验、Wilcoxon符号秩检验）；

**📊 数据集**

使用了六个基于真实开源Python项目的漏洞场景，并配备可见功能测试和隐藏Ghost Tests，实验样本包括8名博士生的pilot数据和预期60名自由职业开发者的正式数据；

**📈 对比分析**

通过比较修复时间、功能测试通过率、安全测试通过率和假修补比例，实验发现LLM辅助显著降低修复时间，但未显著降低安全漏洞通过率，且开发者对LLM辅助持积极评价；

**⚠️ 局限性**

局限性包括Ghost Tests只能检测预设攻击方式、仅覆盖Python语言和有限漏洞类型、样本量受限、时间盒可能偏向LLM加速、以及对LLM使用的普遍性假设。

---

## 496. In-Context World Modeling for Robotic Control

**arXiv ID:** 2606.26025 | [PDF](https://arxiv.org/pdf/2606.26025v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 497. Pulmonary Embolism Risk Stratification from CTPA and Medical Records: Vascular Graphs Are Not All You Need

**arXiv ID:** 2606.25956 | [PDF](https://arxiv.org/pdf/2606.25956v1)

**作者:** Nathan Painchaud `[一作]` (INSA Lyon), Odyssée Merveille `[通讯]` (INSA Lyon)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

通过自动化处理CTPA影像与电子病历，构建了肺栓塞风险分层的端到端系统。

**💡 创新点**

首次系统评估肺血管图网络对风险分层的贡献，并提出两种新的中间融合方法，验证血管图信息不如传统临床指标有效。

**🔧 技术方法**

使用nnUNet对心肺结构进行分割，提取心脏与血管特征；结合XGBoost、TabPFN等表格模型和GCN、GATv2、GIN、Graph Transformer等GNN模型，并通过虚拟节点或交叉注意力实现多模态融合。

**📊 数据集**

基于353例肺栓塞患者的私有数据集，包含完整的CTPA分割、血管图、心脏功能标志物以及临床记录。

**📈 对比分析**

采用10折交叉验证进行比较，表格模型TabPFN在风险分层上达到了86.36% F1，GNN模型最高仅提升至83–84%，血管图和血管生物标志物未能带来显著提升。

**⚠️ 局限性**

局限性主要包括样本量有限、仅使用单一数据集、标签可能未完全反映真实风险、模型在小样本上的过拟合与泛化性能受限。

---

## 498. Autodata: An agentic data scientist to create high quality synthetic data

**arXiv ID:** 2606.25996 | [PDF](https://arxiv.org/pdf/2606.25996v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 499. Deep Reinforcement Learning-Enhanced Event-Triggered Data-Driven Predictive Control for a 3D Cable-Driven Soft Robotic Arm

**arXiv ID:** 2606.26048 | [PDF](https://arxiv.org/pdf/2606.26048v1)

**作者:** Cheng Ouyang `[一作]` (Mississippi State University), Dong Chen `[通讯]` (Mississippi State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于强化学习的事件触发 DeePC 控制框架，用于软体机器人的实时轨迹跟踪。

**💡 创新点**

创新点在于将 RL 学习的状态相关触发策略嵌入 DeePC，动态决定何时进行昂贵的优化，从而显著降低计算负担，同时保持控制精度；且该框架与具体 RL 算法无关。

**🔧 技术方法**

采用了数据驱动预测控制（DeePC）与奇异值分解（SVD）降维、强化学习（DQN、PPO、A2C）进行事件触发决策、正则化优化以及监督覆盖机制。

**📊 数据集**

使用了从三维电缆驱动软体机械臂收集的1200条输入‑输出轨迹（包含 3×3 MIMO 采样）以及对应的高保真仿真数据集。

**📈 对比分析**

通过与周期性 DeePC 及固定阈值事件触发 DeePC 的对比，实验证明在仿真中优化调用率可降低 66%（RMSE 0.144 mm 与周期性相当），硬件上降低 34%（RMSE 2.75 mm 与周期性相同），同时保持闭环性能。

**⚠️ 局限性**

局限性包括仿真与实机之间的模型差距导致 ρ 参数需要重新调节；对高维或更强耦合的软体系统会显著增大 Hankel 矩阵与 SVD 维度；缺乏正式的稳定性证明，且对异常扰动的鲁棒性尚待进一步提升。

---

## 500. WinDOM: Self-Family Distillation for Small-Model GUI Grounding

**arXiv ID:** 2606.25964 | [PDF](https://arxiv.org/pdf/2606.25964v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 501. Improving Neural Network Training by Decoupling the Magnitude and Direction of Weight Vectors

**arXiv ID:** 2606.25971 | [PDF](https://arxiv.org/pdf/2606.25971v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 502. Agentic System as Compressor: Quantifying System Intelligence in Bits

**arXiv ID:** 2606.25960 | [PDF](https://arxiv.org/pdf/2606.25960v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 503. G2DP: Diffusion Planning with Spatio-Temporal Grid Guidance

**arXiv ID:** 2606.26017 | [PDF](https://arxiv.org/pdf/2606.26017v1)

**作者:** Hang Yu `[一作]` (Mercedes-Benz AG), Wilhelm Stork `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于网格引导的扩散规划 G2DP，用于在交互式交通场景下实现安全、高效的轨迹生成

**💡 创新点**

创新点在于将时空占据概率映射为可微分成本体积，并在降噪循环中注入密集梯度，实现主动避障与路径跟随

**🔧 技术方法**

采用扩散变换器 DiT 作为轨迹生成器，U‑Net 预测未来占据概率，结合成本能量引导与梯度注入技术

**📊 数据集**

在 nuPlan、interPlan 与 DeepScenario 三大闭环评测数据集上进行实验

**📈 对比分析**

与规则、混合与学习型基线对比，G2DP 在 nuPlan 与 interPlan 上均获得 SOTA 结果，碰撞避免等关键指标提升超过10分

**⚠️ 局限性**

局限在于车辆足迹模型过于简化，需要手动调节引导窗口与尺度，且未将感知模块与规划统一在同一端到端框架内

---

## 504. Emcar: Embodied Controller for Animating Robots

**arXiv ID:** 2606.26008 | [PDF](https://arxiv.org/pdf/2606.26008v1)

**作者:** Carlos Gomez Cubero `[一作]` (CyPhy Life), Elizabeth Jochum `[通讯]` (Aalborg University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一个无代码平台EMCAR，利用身体化交互（如机器人操纵、遥控绘图和动画录制）让艺术家与非技术人员能够直接设计、编程并表演机器人动作。

**💡 创新点**

创新点在于将身体化艺术实践与机器人编程结合，提供低/零代码界面、实时RTDE控制、可定制3D打印末端执行器，以及基于绘图平板的即时遥控，使非技术用户能像使用绘图工具一样操纵机器人。

**🔧 技术方法**

核心技术包括：UR机器人（UR3/UR5/UR10）与RTDE实时数据交换协议、Wacom绘图板输入、Python/Qt GUI、3D打印末端执行器、可选的计算机视觉映射以及Wizard of Oz 预录制动画框架。

**📊 数据集**

论文未使用传统机器学习或视觉数据集，而是通过多场工作坊和表演记录的实时交互数据来验证平台的可用性和创作潜力。

**📈 对比分析**

作者主要采用定性评估：在艺术工作坊、教育课堂和舞蹈表演中收集用户反馈和案例展示；未给出定量性能指标，但描述了绘图时的延迟、动画录制的实时性以及可重放的准确度。

**⚠️ 局限性**

局限性包括：对UR系列机器人硬件的依赖；绘图时存在低延迟问题，特别是快速细腻笔触；需要手动标定并调整Z偏移；对初始设置的依赖性较高；缺乏完整自主控制能力，仅适用于预录制或Wizard of Oz 场景。

---

## 505. Weave of Formal Thought

**arXiv ID:** 2606.25987 | [PDF](https://arxiv.org/pdf/2606.25987v1)

**作者:** Alexandre Bouayad `[一作]` `[通讯]`, Alexandre Bouayad

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Weave of Formal Thought (WoFT) 框架，结合了面向语法的完整推理引擎（基于 Tree-sitter 的 GLR + 预测式词法）和离散潜在变量微调（使用 RWS 训练 IW-ELBO），实现了在代码生成中既保证语法合法性又利用潜在语法结构的模型。

**💡 创新点**

创新点包括：① 用 GLR 与预测式词法构建能够对子词层面进行完整推理、保证语法完整性和可恢复性的约束解码器；② 将语法非终结符视为离散潜在变量，通过 RWS 优化 IW-ELBO，允许模型在生成时自由插入或省略语法信息；③ 通过这种“可选的语法”机制，将潜在语法作为动态结构草稿盘，从而显著提升表面文本建模性能。

**🔧 技术方法**

技术手段：Tree-sitter 的增量 GLR 解析、预测式词法（DFA 双状态、扫描器协同）、Graph-structured stack（GSS）+ 词法格局图、LoRA 参数高效微调、Reweighted Wake‑Sleep (RWS) + IW‑ELBO 优化、Transformer 语言模型（StarCoder2‑3B）等。

**📊 数据集**

数据集：从 The Stack v2 中抽取 15,000 条 Python 源码，长度 1–100,000 字节；对每条代码使用 StarCoder2‑3B 的 tokenizer 进行子词分词，并通过 Tree-sitter 生成 AST，随后在前缀顺序插入语法非终结符，得到扩展词表后进行训练。

**📈 对比分析**

与两种基线比较：① 仅对原始文本进行监督微调（Text‑SFT）；② 在文本中强制插入完整语法序列再进行监督微调（Formal‑SFT）。实验中，WoFT 在单轮微调下，表面 token 的交叉熵下降至 0.66，而 Text‑SFT 为 0.77、Formal‑SFT 为 0.82，WoFT 相比 Text‑SFT 下降约 14.3%（相对减少）。

**⚠️ 局限性**

局限性：① 仍依赖 Tree‑sitter 的外部扫描器，扩展性受限；② 预测式词法在极端语法或未覆盖的语言上可能产生过度近似；③ 训练仅做单轮，未探索更深层的微调与正则化；④ 评估仅针对表面交叉熵，未覆盖功能正确性或下游任务；⑤ 计算成本在大规模模型或长序列下可能较高。

---

## 506. The Unfireable Safety Kernel: Execution-Time AI Alignment for AI Agents and Other Escapable AI Systems

**arXiv ID:** 2606.26057 | [PDF](https://arxiv.org/pdf/2606.26057v1)

**作者:** Seth Dobrin `[一作]` (ARYA Labs PBC), Łukasz Chmiel `[通讯]` (ARYA Labs PBC)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出并实现了一种名为 Unfireable Safety Kernel 的架构，用于对逃逸式 AI 系统进行执行时 AI 对齐，通过进程分离、结构路径前置、失败闭合和外部化签名证据四大属性实现对 AI 代理的安全约束。

**💡 创新点**

创新点在于将执行时 AI 对齐定义为三层之一，提出四大控制属性并在 Rust 中实现并机理化验证的安全核，同时实现了完整的透明日志和二进制可信链，形成了首个在生产环境中满足所有四项属性并具备机理化证明的逃逸式 AI 控制系统。

**🔧 技术方法**

使用 Rust 编写安全核，配合 Ed25519 加密签名、Sigstore 风格的二进制鉴定；使用 Z3 SMT 进行失效闭合理论证明，Kani 边界检查实现代码路径的机器检查；通过 gRPC/HTTP 接口与四层防护（nginx auth_request、应用层中间件、调用点钩子、客户端 SDK 断路器）构建完整的多层防御体系。

**📊 数据集**

使用自定义的自修改 AI 系统（ARYA 真实世界模型）作为攻击主体，生成 1,000+ 符合性测试用例和 17 类攻击场景进行对抗评估；在真实部署中模拟 1,000 次自修改尝试，验证安全核的防护效果。

**📈 对比分析**

与 Galileo Agent Control、Microsoft Agent Governance Toolkit、Microsoft Authorization Fabric 三大现有方案对比，证明本系统在四项属性（进程分离、结构路径前置、失败闭合、外部化签名证据）上独占优势；在 20 并发下 p99 延迟为 496.9 ms，约为 Python 实现的 920.0 ms 的 1.85 倍；自修改攻击模拟中 704 次核心改动被全部拒绝，零成功逃逸。

**⚠️ 局限性**

局限性包括：对外部审计依赖度高，未实现多方操作员签名（阈值签名）；递归自我改进策略的完整验证尚未完成；在高负载或持续饱和条件下的实时性能与可靠性评估仍待进一步实验；对多智能体协同逃逸的跨代理分析尚未覆盖。

---

## 507. FORCE: Efficient VLA Reinforcement Fine-Tuning via Value-Calibrated Warm-up and Self-Distillation

**arXiv ID:** 2606.26006 | [PDF](https://arxiv.org/pdf/2606.26006v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 508. Hotelling-Downs with Facility Synergy: The Mall Effect

**arXiv ID:** 2606.26055 | [PDF](https://arxiv.org/pdf/2606.26055v1)

**作者:** Elliot Anshelevich `[一作]` (Rensselaer Polytechnic Institute), Noah Prisament `[通讯]` (Rensselaer Polytechnic Institute)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种 Hotelling‑Downs 模型的变体，加入了设施协同效应：客户不再仅关注最近设施，而是倾向于访问设施数量多的地点，且其成本定义为距离除以该地点设施数量。

**💡 创新点**

创新点包括证明该模型下纯 Nash 均衡必然存在，并且存在一个均衡使得客户总成本等于最优解；给出了 Price of Anarchy（PoA）上界 225/64 ≈ 3.516；首次在此类模型中引入了“气泡”概念并通过一系列结构性引理限制其大小，从而获得 PoA 上界；同时证明 Price of Stability 为 1，说明存在最优均衡。

**🔧 技术方法**

主要技术手段为游戏理论分析与几何构造、凸性与单调性论证、对泡沫区间的长度上界推导、以及对可行解的优化约束进行代数与微积分推导；使用了大量结构性引理来描述核心区间与泡沫区间之间的关系，并通过多层不等式链条完成 PoA 证明。

**📊 数据集**

研究完全基于理论分析，假设客户在区间 X=[0,1] 上连续均匀分布；未使用任何实际数据集或实验。

**📈 对比分析**

通过理论证明与对比，PoA 上界 3.516 显著高于经典模型的 2 上界，说明尽管客户行为更复杂，均衡解仍相对有效；Price of Stability 为 1 表明存在完全最优的均衡；由于缺乏实验数据，性能评价仅来自理论极限。

**⚠️ 局限性**

局限性包括：仅考虑一维连续区间；成本函数固定为 |x−x_i|/n_i；未考察不同度量空间、离散客户集合或更一般的成本函数（如 |x−x_i|/(n_i)^p 等）；结果可能无法直接推广到二维空间、图网络或非均匀分布的客户。

---

## 509. Natural Ungrokking: Asymmetric Control of Which Rules Survive Pretraining

**arXiv ID:** 2606.26050 | [PDF](https://arxiv.org/pdf/2606.26050v1)

**作者:** Juliana Li `[一作]` (Harvard University), Diya Sreedhar `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究语言模型在预训练过程中出现并随后消失的规则（自然未解耦），探讨其可预测性、机理和可控性。

**💡 创新点**

提出“支持频率定律”：规则在训练结束时的存活可由其在语料中出现的频率预测；揭示“位移”机制，规则丧失是与更强表面模式竞争的结果；展示对规则的消除能按剂量递增实现，但恢复同样规则却几乎不可能，揭示破坏与恢复的单向性。

**🔧 技术方法**

使用冻结的模板对照探针电池评估规则掌握，计算对比边缘（rule‑vs‑prior 的对数概率差）作为物理指标；对比不同数据量、模型规模和语料的结果；采用预注册与盲测评估实验可信度。

**📊 数据集**

TinyStories（规则支持高）和基于 ClimbMix 的过滤 web 语料（规则支持低）作为实验语料；此外对公开的 Pythia 检查点（70M–1.4B）进行验证，检验跨模型的普适性。

**📈 对比分析**

在 TinyStories 上模型在约 925 步时规则准确率达 0.94，最终保持；在 web 语料上规则在 2800–3100 步间崩溃，准确率跌至 0；对比分析表明规则存活与支持频率相关，而不是数据量或模型规模；公共检查点显示更小模型更易崩溃，验证了规模依赖。消除实验显示规则消失随剂量严格单调；恢复实验显示即使提供三倍支持也无法使行为恢复。

**⚠️ 局限性**

实验仅在 11.5M 参数模型下进行，缺乏大规模验证；探针电池为模板化，未检验自然语言真实表现；位移机制的阈值界定仍未量化；恢复实验仅在单一规则与单一语料内验证，缺乏普适性；对恢复时序的完整测量尚未完成。

---

## 510. Representing One Letter Weighted Automata Over the Tropical Semiring

**arXiv ID:** 2606.26038 | [PDF](https://arxiv.org/pdf/2606.26038v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

---

## 511. TriViewBench: Controlled Complexity Scaling for Multi-View Structural Reasoning in MLLMs

**arXiv ID:** 2606.26029 | [PDF](https://arxiv.org/pdf/2606.26029v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 512. How Robust is OCR-Reasoning? Evaluating OCR-Reasoning Robustness of Vision-Language Models under Visual Perturbations

**arXiv ID:** 2606.26041 | [PDF](https://arxiv.org/pdf/2606.26041v1)

**作者:** Yuxing Cheng `[一作]` (Jilin University), Yi Chang `[通讯]` (Jilin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了 OCR-Robust 基准，用来评估 OCR 推理模型在视觉降噪下的鲁棒性，并对 18 种模型进行系统评估。

**💡 创新点**

通过系统化的扰动筛选与 RCR/WCR/CRI 指标，提供专门针对文本理解与推理的鲁棒性评价框架，并发现结构化视觉内容更易受损。

**🔧 技术方法**

采用玻璃模糊、运动模糊、弹性变形、色彩偏移和降雪等五种扰动，使用 RCR、WCR、CRI 三种指标衡量鲁棒性，并在零样本情境下进行评测。

**📊 数据集**

使用了 812 个样本的 OCR1.0（482）和 OCR2.0（330）两部分集合，涵盖文档、场景文字、收据、手写、数学、图表、几何图和表格等多种数据源。

**📈 对比分析**

通过比较清洁准确率、相对保留率、最差保留率和综合 CRI，对 18 个模型进行对比，发现闭源模型总体鲁棒性最高，单纯提升清洁准确率并不一定提升鲁棒性。

**⚠️ 局限性**

限制包括数据规模有限、扰动选择基于少量模型、未涵盖复合降噪、零样本评估可能低估鲁棒性、OCR+LLM 仅使用 PaddleOCR、闭源模型覆盖受限。

---

## 513. Variable Bound Tightening for Nash Equilibrium Computation in Multiplayer Imperfect-Information Games

**arXiv ID:** 2606.25997 | [PDF](https://arxiv.org/pdf/2606.25997v1)

**作者:** Sam Ganzfried `[一作]` `[通讯]` (Cornell University), Sam Ganzfried (Cornell University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对多玩家不完全信息博弈的纳什均衡求解中，本文通过对非线性互补性程序（NLCP）中松弛变量和乘子变量的上下界进行严格化，从而显著提升求解效率。

**💡 创新点**

创新点在于利用强对偶性推导出可行的有限界限（针对松弛变量的上界和乘子变量的绝对值上界），这些界限可在不改变解空间的前提下大幅强化凸松弛，进而提升空间分支定界（branch‑and‑bound）的搜索效果。

**🔧 技术方法**

采用的技术包括：NLCP建模、Gurobi的非凸二次规划求解器、空间分支定界算法以及对互补性约束的McCormick包络近似；同时利用对偶问题的最优性条件推导变量界限。

**📊 数据集**

实验数据集为三玩家Kuhn扑克（48个信息集、601个节点的完整游戏以及去除占优动作后的约415个节点的简化游戏）。

**📈 对比分析**

与Gambit软件套件（logit量子响应法）和原始NLCP实现对比，施加松弛变量上界后，完整游戏的求解时间从>24小时压缩至1.160秒，远快于对照方法（约2.5分钟）。

**⚠️ 局限性**

局限性在于：仍需手动分析并推导界限，且对那些缺乏可去除占优动作或界限不易计算的更大游戏仍可能无法取得显著改进；乘子变量界限虽有理论上限，但在实际中对性能影响有限。

---

## 514. Mixture-of-Experts RL for Fault-Tolerant Legged Locomotion

**arXiv ID:** 2606.25965 | [PDF](https://arxiv.org/pdf/2606.25965v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 515. Learning Robot Visual Navigation in Crowds via Intention-Aware Scene Representations

**arXiv ID:** 2606.26047 | [PDF](https://arxiv.org/pdf/2606.26047v1)

**作者:** Han Bao `[一作]` (Southern University of Science and Technology), Jiankun Wang `[通讯]` (Southern University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了iCrowdNav框架，利用基于视觉的意图感知场景表示实现机器人在人群中的安全高效导航。

**💡 创新点**

创新点在于将BEV占据特征与基于Transformer的I^2Former（姿态编码、IntentFormer、InteractFormer）相结合，通过3D人体姿态推断意图并融合至BEV场景表示，形成端到端可训练的意图感知场景表示；同时实现了从仿真到真实的零射击迁移。

**🔧 技术方法**

使用技术包括RGB‑D摄像头、ResNet‑18+3D卷积的BEV编码、I^2Former（姿态编码、IntentFormer、InteractFormer）、PPO深度强化学习、BEV与姿态特征融合以及离线预训练+在线策略微调。

**📊 数据集**

训练数据主要来自Isaac Sim仿真中的SocNav‑Gym环境，并使用nuScenes进行BEV预训练；在真实测试中使用购物中心、地铁站、体育馆等场景。

**📈 对比分析**

与DRL‑VO、SARL‑OM、ViNT、DWA等基线对比，采用成功率、导航时间、路径长度、私密区时间等指标。iCrowdNav在低高密度、不同宽度的环境中均取得最高成功率、最短路径和最低私密区时间，明显优于所有基线。

**⚠️ 局限性**

局限性在于前方摄像头视野有限且易受遮挡，导致在超密集场景下意图推断困难；未融入多模态信息（如声音、激光），需要进一步丰富场景表示以提升极端拥挤环境的鲁棒性。

---

## 516. From Rubble Simulation to Active Magnetic Mapping: Quantum Sensing for Disaster Response

**arXiv ID:** 2606.25957 | [PDF](https://arxiv.org/pdf/2606.25957v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 517. DomainShuttle: Freeform Open Domain Subject-driven Text-to-video Generation

**arXiv ID:** 2606.26058 | [PDF](https://arxiv.org/pdf/2606.26058v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 518. Why Multi-Step Tool-Use Reinforcement Learning Collapses and How Supervisory Signals Fix It

**arXiv ID:** 2606.26027 | [PDF](https://arxiv.org/pdf/2606.26027v1)

**作者:** Yupu Hao `[一作]` (Chinese Academy of Sciences), Jun Zhao `[通讯]` (Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了大语言模型在多轮工具使用任务中应用强化学习时的训练不稳定与性能瓶颈，并提出多种监督信号与训练策略以缓解“结构崩溃”问题。

**💡 创新点**

创新点在于将监督信号分为同步与交替两类，系统评估了SFT、离线监督、提示引导、错误轨迹监督与过程反思监督等方法，并发现交替训练与过程反思监督能显著提升稳定性与泛化。

**🔧 技术方法**

主要技术包括基于Qwen系列模型的强化学习框架（GRPO），多种监督损失（SFT、OPS、HBG、ETS、PRS），以及对控制令牌概率分布的细粒度分析与干预。

**📊 数据集**

使用的数据集为BFCL‑V3（多轮工具使用基准）进行训练与评估，并在ACEBench上检验格式与内容的 OOD 泛化。

**📈 对比分析**

在多轮指标上，交替训练的Process Reflection Supervision平均得分最高（约25.75分），明显优于单纯RL或SFT；但在格式/内容 OOD 下，纯SFT方法表现最差，凸显其过拟合。

**⚠️ 局限性**

局限性主要是实验数据规模有限，未探究更大规模数据对结果的影响，且仅在有限的工具调用环境中验证，缺乏跨环境的一致性验证。

---

## 519. Privacy Vulnerabilities of Attention Layers in Tabular Foundation Models and Protection of High-Risk Queries

**arXiv ID:** 2606.26021 | [PDF](https://arxiv.org/pdf/2606.26021v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 520. From Sparse and Imperfect 2D Anchors to Consistent 3D Gaussian Street Scenes: Support-Aware Appearance

**arXiv ID:** 2606.26007 | [PDF](https://arxiv.org/pdf/2606.26007v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 521. Tensorion: A Tensor-Aware Generalization of the Muon Optimizer

**arXiv ID:** 2606.25975 | [PDF](https://arxiv.org/pdf/2606.25975v1)

**作者:** Vladimir Bogachev `[一作]` (HSE University), Maxim Rakhuba `[通讯]` (HSE University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Tensorion优化器，将Muon推广到高阶张量并通过双重范数松弛实现LMO求解。

**💡 创新点**

创新点在于构造了一个基于张量展开核范数的松弛范数，并给出离线展开选择的启发式方法。

**🔧 技术方法**

使用张量展开、核范数、Newton-Schulz正交化、张量收缩与卷积线性映射理论等技术。

**📊 数据集**

在CIFAR-10/100、Tiny ImageNet、Imagenette（Swin、ViT）等视觉数据集上进行实验。

**📈 对比分析**

与SGD、AdamW以及Muon的在线/离线展开方案进行对比，Tensorion在准确率和损失上均优于基线。

**⚠️ 局限性**

主要限制在于仅针对视觉模型进行验证，未评估更大规模或非视觉序列任务的表现。

---

## 522. Action ControlNet: A Lightweight Delay-Aware Adapter for Smooth Asynchronous Control in Vision-Language-Action Models

**arXiv ID:** 2606.25985 | [PDF](https://arxiv.org/pdf/2606.25985v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 523. On the Parameterized Complexity of Bounded-Density Vertex Deletion

**arXiv ID:** 2606.26012 | [PDF](https://arxiv.org/pdf/2606.26012v1)

**作者:** Jakob Raupach `[一作]` (Technische Universität Berlin), André Nichterlein `[通讯]` (Technische Universität Berlin)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在删去至多 k 个顶点后，使图的密度降到阈值 ρ 的问题（Densest Subgraph Reduction），并在不同结构参数下（如树宽、树深、回溯点数、顶点完整性、最大叶数、CliqueWidth 等）划定其参数化复杂度。

**💡 创新点**

创新点在于：①首次证明该问题对树深、回溯点数等参数及其上限（树宽）具有 [1]-难性；②展示对顶点完整性、最大叶数等更大参数的 FPT 结果；③在目标密度为常数时，甚至在 CliqueWidth 参数下也可得到 FPT 算法；④提出量化的分数取向动态规划技术，统一处理树宽与 CliqueWidth 两种结构参数。

**🔧 技术方法**

主要技术包括：量化分数取向（q‑quantized orientation）与 Charikar 的密度与取向等价性、基于 CliqueWidth 表达式与树分解的动态规划、整数线性规划求解数值分配问题，以及多阶段分枝与状态压缩。

**📊 数据集**

本研究为纯理论论文，未使用实验数据集；所有结论均来自算法设计与复杂度证明。

**📈 对比分析**

与先前的研究相比，本文在相同参数（树宽、CliqueWidth 等）下提供了更精细的可行与不可行边界；FPT 算法的时间复杂度为 (q·p)^{tw²}·n 和 (q·p)^{tw²}·n^O(1)（树宽+密度）以及 q^{O(p⁷)}·p^{O(ℓp)}·n^O(1)（CliqueWidth+密度）。对目标密度为常数时，FPT 算法的复杂度大幅降低，达到单指数级别。

**⚠️ 局限性**

局限性：①对某些参数（如距离到区间图、距离到cograph 等）仍未完成完整的复杂度划分；②当目标密度不是常数时，算法的指数部分仍依赖于 p 与 q，导致大输入规模下实际运行困难；③硬性结果仅表明 [1]-难性，无法进一步区分更细粒度的可解性；④实验验证缺失，实际性能仍需进一步评估。

---

## 524. Is Variational Monte Carlo Robust? Sharp Moment Thresholds and Heavy-tailed Stochastic Optimization

**arXiv ID:** 2606.26009 | [PDF](https://arxiv.org/pdf/2606.26009v1)

**作者:** Philipp Grohs `[一作]` (University of Vienna), Davide Nobile `[通讯]` (University of Vienna)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了变分蒙特卡罗（VMC）中随机优化问题的统计性质，证明其随机梯度估计量普遍呈现重尾分布；并在弱尾分布下提出并分析了一种新的鲁棒梯度裁剪方法PS‑Clip‑VMC，证明其在期望和高概率下收敛。

**💡 创新点**

首次系统地揭示了 VMC 中能量和梯度随机变量的矩缺失性，并给出了对解析、紧支撑 Ansatz 的最小矩数界；提出的 PS‑Clip‑VMC 能在重尾环境下保证收敛，弥补了传统仅裁剪局部能量方法的不足。

**🔧 技术方法**

采用了随机采样（MCMC）与期望估计、复合裁剪函数、梯度裁剪、Weierstrass 预备定理、薄层理论、随机优化收敛分析、PS‑Clip‑VMC 迭代更新。

**📊 数据集**

使用 FermiNet 作为深度波函数 Ansatz，在原子系统（硫和氩）上训练，原子数最多 18 个电子。

**📈 对比分析**

与传统的仅裁剪局部能量的方法比较，PS‑Clip‑VMC 在训练轨迹上更为稳定，最终能量更低；在硫和氩的实验中均达到或超过已发表的最佳能量。

**⚠️ 局限性**

局限性包括：理论关于重尾性的结论主要针对 Slater‑Jastrow Ansatz；对更通用的神经网络 Ansatz 的推广仍需进一步研究；实验规模仅覆盖少数小原子，缺乏大规模分子系统验证。

---

## 525. DSP-SLAM++: A Unified Framework for Multi-Class, High-Fidelity Object SLAM in the Wild

**arXiv ID:** 2606.25953 | [PDF](https://arxiv.org/pdf/2606.25953v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 526. A cross-process welding penetration status prediction algorithm based on unsupervised domain adaptation in laser and TIG welding

**arXiv ID:** 2606.26078 | [PDF](https://arxiv.org/pdf/2606.26078v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 527. Model Forensics: Investigating Whether Concerning Behavior Reflects Misalignment

**arXiv ID:** 2606.26071 | [PDF](https://arxiv.org/pdf/2606.26071v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 528. Taxonomy-aware deep learning for hierarchical marine species classification in underwater imagery

**arXiv ID:** 2606.25989 | [PDF](https://arxiv.org/pdf/2606.25989v1)

**作者:** Dan Zimmerman `[一作]` (Florida Atlantic University), George Sklivanitis `[通讯]` (Florida Atlantic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一套面向海洋物种分类的“分类学感知”深度学习框架，用于在潜水图像中对物种进行层级分类并解决跨平台域偏移、细粒度相似度和标注粒度不均的问题。

**💡 创新点**

创新点在于：① 将损失函数与层级评价指标对齐（引入分类学距离惩罚和多级权重）；② 采用最小风险贝叶斯推理根据预计算的树距矩阵选择最佳叶子预测；③ 采用多尺度特征编码与独立层级分类头，避免学习依赖导致的过拟合；④ 通过层级学习率衰减细致微调自监督ViT（DINOv2）而非统一学习率，提升对域偏移的鲁棒性。

**🔧 技术方法**

使用的技术包括：ViT-B/14（DINOv2）作为主干，四尺度（ROI、3×、5×、全图）特征拼接；层级权重交叉熵损失加分类学距离损失；10折交叉验证集成与平均；最小风险推理；水平翻转测试时增强；层级学习率衰减（LLRD）。

**📊 数据集**

主要数据集为FathomNet 2025，包含23,699个训练ROI，79类（覆盖7个分类学层级）和1个“未知”类别，测试集788个ROI；此外使用5,802个独立验证样本来评估泛化性能。

**📈 对比分析**

与竞赛顶尖方案（如MATANet、Yonsei+SSL）相当，官方榜单得分为1.581（与第一名1.535差距3%以内）。在独立验证集上，系统在多尺度、集成和最小风险推理等关键组件上分别显著提升10%–15%（p < 0.001）。相比ConvNeXtV2-Base，DINOv2-Base+LLRD提升了约11%税onomic距离，并在10折集成上保持稳定。

**⚠️ 局限性**

局限性包括：① 训练与评估集规模有限，置信区间仍宽；② 仅在海洋图像任务验证，需在其他层级任务上进一步测试；③ 需要对“未知”类别进行开放集识别；④ 大规模层级（>10K物种）下的距离矩阵存储和推理需要更高效算法；⑤ 评估主要依赖平均税onomic距离，未考虑模型对不同类群的偏差；⑥ 需进一步完善概率校准与更高效的多尺度特征融合方案。

---

## 529. Learning Action Priors for Cross-embodiment Robot Manipulation

**arXiv ID:** 2606.26095 | [PDF](https://arxiv.org/pdf/2606.26095v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 530. MVTrack4Gen: Multi-View Point Tracking as Geometric Supervision for 4D Video Generation

**arXiv ID:** 2606.26087 | [PDF](https://arxiv.org/pdf/2606.26087v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 531. InvestPhilBench: A Multi-Layer Dynamic Benchmark for Evaluating Large Language Model Procedural Reasoning in Expert Investment Philosophy

**arXiv ID:** 2606.25984 | [PDF](https://arxiv.org/pdf/2606.25984v1)

**作者:** Mingguang Chen `[一作]` (University of California), Bo Qu `[通讯]` (Illinois Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并发布了 InvestPhilBench，一个基于八层认知分层、覆盖原则识别到情景推演的投资哲学评估基准，配套完整的自动评分管线、动态扰动与失败模式检测。

**💡 创新点**

通过构建多维度自动评分指标（OGRS、KCCS、SAP@k、IVP、CKCA）、统一的失败模式检测协议以及对前沿 LLM 的程序性推理进行细粒度分析，首次揭示 LLM 在执行投资者特定决策流程时的“程序缺口”，并为后续改进提供量化方向。

**🔧 技术方法**

采用 LLM 推理、向量检索 + PKB 召回、Oracle 条件注入、语义匹配、自动化评分管线（BASP）、失败模式检测算法和动态扰动（情境/指标/时间置换）等技术。

**📊 数据集**

基于 9 位经典投资者的 1.67M 单词原始文本（书籍、年报、访谈、论文等），构建 118 条原则卡、25 条决策框架卡，并生成 243 道包含 197 训练/46 测试、分层标注与黄金推理程序的 QA 题集。

**📈 对比分析**

在闭卷、PKB-RAG、Oracle 三种评估条件下，使用 BAST 组合得分与 GRA 指标对 4 个前沿 LLM 进行比较，发现 BAST 分数饱和但 GRA 仍揭示前沿模型在程序性推理上存在显著缺陷；模型提供商层级差异明显，成本效益与推理能力呈现分化。

**⚠️ 局限性**

当前版本仅覆盖前沿闭源模型、缺乏公开权重验证、GRA 仍为单注释、Oracle 条件需人工匹配、BASP 与人类评审的对齐度有限，未来需扩大模型样本、完善黄金推理程序、提升动态扰动和多投资者覆盖。

---

## 532. A welding penetration prediction model for laser welding process based on self-supervised learning using physics-informed neural networks

**arXiv ID:** 2606.26059 | [PDF](https://arxiv.org/pdf/2606.26059v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 533. When Certainty Is an Artifact: Keyword Lexicon Blindness and the (Mis)Measurement of Rhetorical Stance

**arXiv ID:** 2606.26062 | [PDF](https://arxiv.org/pdf/2606.26062v1)

**作者:** Bo Chen `[一作]` `[通讯]` (Institute of Computing Technology), Bo Chen (Institute of Computing Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过比较关键词词典计数与大语言模型（LLM）零射门双维度分类，对计算社会科学中情感与修辞模态的测量误差进行研究，发现传统词典方法产生的显著正相关实际上是测量误差导致的结果。

**💡 创新点**

创新点在于提出了四步验证模板、三种词典错误模式（句法盲区、多义性盲区、类别缺失）以及跨模型LLM验证的可靠性框架，提供了一套可复现的工具来审计词典驱动的研究结论。

**🔧 技术方法**

使用技术包括：分离情感与模态的多维词典、基于大语言模型的零射门双维度语义分类（Doubao-Seed-2.0-Lite 与 GPT-5.4）、Pearson相关检验、句子级错误分析与词典构造规则。

**📊 数据集**

数据集为85段访谈，涵盖四位公共知识分子（Ray Dalio、Cathie Wood、Kenneth Rogoff、Peter Zeihan），共计32,625句子、498,200词，来源于YouTube视频字幕并通过LLM进行说话人分离。

**📈 对比分析**

方法对比：先用关键词词典计算负情感与强势词的相关系数，随后使用LLM进行全量句子双维度分类并重新计算相关系数。结果显示关键词方法在所有说话者中均得到高且正相关的系数（r≈0.72–0.93），而LLM方法则得到无显著或负相关（r≈0.20、-0.5），并且两种LLM模型间无显著差异，说明关键词方法误差显著。

**⚠️ 局限性**

局限性包括：句子级错误分析仅覆盖Dalio的样本，词典构造依赖人工判断，YouTube字幕的ASR转录误差未人工校正，未将情感变化与外部经济指标对照，且研究未涵盖更广泛的语言变体和其他语境。

---

## 534. On-Policy Self-Distillation with Sampled Demonstrations Reduces Output Diversity

**arXiv ID:** 2606.26091 | [PDF](https://arxiv.org/pdf/2606.26091v1)

**作者:** Andrei Liviu Nicolicioiu `[一作]` (Mila), Aaron Courville `[通讯]` (Mila)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了自监督分布式蒸馏（Self-Distillation with Sampled Demonstrations, SDSD）的多样性退化问题，并提出理论与实验验证。

**💡 创新点**

提出了自蒸馏的最优策略以点条件互信息（PCMI）为倾斜因子，揭示其比传统 RL 更易放大已存在的概率差距，从而导致模式坍塌；并用功能与语义多样性指标量化多样性退化。

**🔧 技术方法**

使用基于逆 KL 的自蒸馏损失、EMA教师、对比式多样性奖励、梯度下降优化，配合 token‑级与序列级分析，和图路径寻找与科学 QA 的实验评测。

**📊 数据集**

包含自定义图路径寻找任务、SciKnowEval 科学多项选择问答、以及基准 RL 与自蒸馏模型（Qwen3‑1.7B、Qwen3‑8B、Olmo‑3‑7B‑Instruct）在不同图尺寸与难度下的数据。

**📈 对比分析**

与标准 on‑policy RL（GRPO）和带多样性奖励的 GRPO 进行对比，结果显示 SDSD 在 Pass@1 上表现相当甚至更好，但 Pass@k 曲线明显更平缓，表明功能多样性和语义多样性明显下降；外部示范虽提升 In‑distribution 性能，但仍无法恢复多样性。

**⚠️ 局限性**

局限在于仅分析了基于正确 rollouts 采样示例的 SDSD 变体；未覆盖更丰富的教师信号（如运行时错误、环境反馈）；假设教师为冻结的基准策略，未考虑 EMA 学习与自生成示例的额外自选择偏差；以及仅在 token‑级和序列级给出理论，缺乏对更复杂任务的推广。

---

## 535. ForceBand: Learning Forceful Manipulation with sEMG

**arXiv ID:** 2606.26093 | [PDF](https://arxiv.org/pdf/2606.26093v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 536. Same Evidence, Different Answer: Auditing Order Sensitivity in Multimodal Large Language Models

**arXiv ID:** 2606.26079 | [PDF](https://arxiv.org/pdf/2606.26079v1)

**作者:** Akshay Paruchuri `[一作]` (Stanford University), Ehsan Adeli `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对18款多模态大型语言模型进行五个输入顺序面板的系统性审计，测量不同顺序对答案的一致性。

**💡 创新点**

提出Facet-Probe框架，结合同序列对照控制和Bayesian项响应理论（2PL IRT）分离顺序噪声与系统偏差，并将跨顺序翻转率作为新报告轴；首次量化2026年前沿与开源模型的顺序鲁棒性。

**🔧 技术方法**

使用同序列对照实验、K=6顺序采样、LLM-judge判定、两层Bayesian IRT模型、温度0 API调用、K-政策检测/聚合、Prompt-Level CTA与思考预算等技术。

**📊 数据集**

覆盖MCQ推理（MMLU-Pro、CommonsenseQA、MathVision）、多跳QA/RAG（HotpotQA、MuSiQue、MultiHop-RAG、MedXpertQA）、多图VQA（Mantis-Eval、MedFrameQA）以及自由混合模态RAG（MRAMG-Recipe、MMDocRAG、MultiModalQA）等多样化数据集。

**📈 对比分析**

与模型的能力（θ_correct）对比，发现翻转率与能力高度相关（Spearman ≈ -0.95），但最高模型仍有约0.3-0.5的翻转率；同序列控制显示温度0下的解码器随机性低，顺序噪声显著；提示级干预（CTA、思考预算）在文本上可降低翻转，但在视觉任务无效，且不具可组合性。

**⚠️ 局限性**

限制包括：同序列控制仅覆盖少量Gemini细胞、K=6为下限、温度0 API仍有随机性、图像集仅基于MedFrameQA、封闭模型与开源模型的堆栈差异、LLM-judge的测量方差、机制标签仅诊断性、数据集覆盖范围有限、干预实验仅针对Gemini Pro/Flash、未验证训练时刻或架构改进。

---

## 537. TryOnCrafter: Unleashing Camera Trajectories for Realistic Video Virtual Try-on via a Renderable 4D Try-on Proxy

**arXiv ID:** 2606.26092 | [PDF](https://arxiv.org/pdf/2606.26092v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 538. Neglected Free Lunch from Post-training: Progress Advantage for LLM Agents

**arXiv ID:** 2606.26080 | [PDF](https://arxiv.org/pdf/2606.26080v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 539. RevengeBench: Reverse Engineering Code-Space Policies from Behavioral Experiments

**arXiv ID:** 2606.26094 | [PDF](https://arxiv.org/pdf/2606.26094v1)

**作者:** Babak Rahmani `[一作]` (Tübingen AI Center), Matthias Bethge `[通讯]` (Tübingen AI Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了使用LLM逆向学习可执行策略的任务，即从观察到的行为轨迹中恢复目标策略的代码。

**💡 创新点**

提出了受限干预的逆向学习框架和Benchmark "RevengeBench"，并证明主动探测可提升恢复质量。

**🔧 技术方法**

利用LLM生成的可执行代码、action distance评估、交互式对弈探测，评估12种前沿LLM。

**📊 数据集**

使用CodeClash比赛的75条目标策略，涵盖BattleSnake、Halite、Poker、RoboCode、RobotRumble。

**📈 对比分析**

通过多轮对弈、主动/被动观察，比较模型恢复的action distance下降百分比，最佳模型可关闭72%距离，平均34–72%。

**⚠️ 局限性**

局限在目标策略静态、action distance不完全等价、实验环境噪声大且不同场景差异显著。

---

## 540. Real-Time Voice AI Hears but Does Not Listen

**arXiv ID:** 2606.26083 | [PDF](https://arxiv.org/pdf/2606.26083v1)

**作者:** Martijn Bartelds `[一作]` (Together AI), James Zou `[通讯]` (Together AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估四大生产实时语音AI在词义与语音情绪冲突场景下的决策与感知表现。

**💡 创新点**

首次量化“情感智商缺口”，并揭示实时语音系统多关注文本而忽视非语言线索。

**🔧 技术方法**

采用实时语音模型API、合成TTS、单轮诊断与多轮情境对话，配合提示指令。

**📊 数据集**

使用ElevenLabs合成的语音样本（情绪、口音、年龄）与五名人工听众验证的标签。

**📈 对比分析**

对比四系统在三种场景（求助回拨、转账欺诈、志愿者招募）下的行动正确率、情绪识别准确率；结果显示系统多在文本层决策，情绪识别虽能感知但往往不影响行动。

**⚠️ 局限性**

局限于合成数据、场景覆盖有限、指令效果不稳定，未能解决模型对非语言特征的根本忽视。

---

## 541. Strategyproof Facility Location and Committee Selection with Mixed Max and Sum Agent Types

**arXiv ID:** 2606.26074 | [PDF](https://arxiv.org/pdf/2606.26074v1)

**作者:** Yue Gruszecki `[一作]` (Rensselaer Polytechnic Institute), Elliot Anshelevich `[通讯]` (Rensselaer Polytechnic Institute)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了战略设施选址问题，目标是在任意度量空间中选择k个设施以最小化总代理成本。代理有两种类型的个体成本函数：最大型和求和型。

**💡 创新点**

提出了确定性策略无关机制，并证明了与最小化总代理成本的解决方案相比的近似比界限。

**🔧 技术方法**

使用了确定性策略无关机制，分析了在代理类型和位置私有的情况下的近似比。

**📊 数据集**

研究中没有提到具体的数据集，但涉及任意度量空间和代理位置。

**📈 对比分析**

当代理类型私有但位置已知时，证明了近似比为(3 - 2/k)是始终可能的；当知道每种类型代理的比例时，近似比为(2/1-k+√(k^2-k+1)-1)。当代理位置私有时，使用简单的中位数机制在一维度量中实现了近似比为3。

**⚠️ 局限性**

限制在于对于代理类型私有的情况，形成非平凡的下界似乎更为困难，且现有的下界不够紧。

---

