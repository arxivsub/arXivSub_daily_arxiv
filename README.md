# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-07-31 | 今日论文总数: 618

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Strategy, Not Payoffs: A Behavioural Embedding of Normal-Form Games

**arXiv ID:** 2607.27536 | [PDF](https://arxiv.org/pdf/2607.27536v1)

**作者:** Joshua Caiata `[一作]` (University of Waterloo), Kate Larson `[通讯]` (University of Waterloo)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了通过细调大型语言模型（LLM）学习的博弈策略是否能迁移到其他博弈中，并探讨了游戏嵌入如何解释和预测这种迁移效果。

**💡 创新点**

提出了一种两维手工特征嵌入（ENT‑SW），只利用纳什均衡的熵与最优回应切换度，能够在严格留一游戏（LOGO）条件下预测迁移，且优于传统结构嵌入与单纯游戏身份基线。

**🔧 技术方法**

使用了纳什均衡熵、最优回应切换度计算、超参数回归（带岭正则化）以及留一游戏/留一对的交叉验证方法。

**📊 数据集**

利用了包含15种经典二维/三维博弈的实例集（囚徒困境、 stag hunt、Chicken、Rock‑Paper‑Scissors 等），每个博弈生成约 20000 个随机支付实例。

**📈 对比分析**

与单纯游戏身份、RSTP、PHD 能量、RG 统计等基线进行比较；在 LOGO 和 LOPO 两种留出策略下，ENT‑SW 在预测误差上平均降低约 0.36–0.40，显著优于基线，且在 LOGO 下仅此嵌入能保持正向预测。

**⚠️ 局限性**

局限性包括仅评估了 15 个小规模博弈，未验证在更大行动空间、多玩家或序列博弈中的泛化；嵌入仍无法捕捉策略指向性（如合作 vs 自私）等细节。

---

## 2. PAUSE: A User-Centric Benchmark for Personal AI Assistants in Unified Service Environments

**arXiv ID:** 2607.27354 | [PDF](https://arxiv.org/pdf/2607.27354v1)

**作者:** Haoyu Chen `[一作]` (University of Alberta), Di Niu `[通讯]` (University of Alberta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了PAUSE基准，用于评估个人AI助手在统一、用户中心化的服务环境中进行多轮、状态持久、权限约束的交互；

**💡 创新点**

创新点在于：①将个人服务环境统一建模为持久状态和隐藏配置；②设计多模式评估框架（基于目标的LLM评判+轨迹重叠度量）；③构建可扩展的任务合成与轨迹生成流水线；

**🔧 技术方法**

采用大型语言模型（Gemini‑3‑Flash、GPT‑5等）进行助手与用户模拟、工具调用与环境交互；使用LLM-as-Judge进行目标达成判定与轨迹相似度计算；

**📊 数据集**

通过模板化生成的合成数据集，覆盖健康记录、设备连接、购物等多领域任务；共生成约300个候选任务，最终评估集180个，划分为易/难数据日志追踪和购物三类；

**📈 对比分析**

通过与多种LLM的对比实验，发现即使是最先进的专有模型，在需要状态推理与配置意识的“硬”任务上任务完成率仍低于70%，而开放源模型表现更差；评估显示目标完成度与轨迹相似度高度相关；

**⚠️ 局限性**

局限性包括：未报告pass@k等指标；开放式任务的状态验证不完全；评测仅覆盖部分专有模型，缺乏更广泛的开源模型评估；

---

## 3. It's Not Just More Demos: Counterfactual Action Sensitivity Coverage for Data-Efficient Robust Robot Imitation

**arXiv ID:** 2607.27261 | [PDF](https://arxiv.org/pdf/2607.27261v1)

**作者:** Giovanni D'urso `[一作]`, Brendan Tidd `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对已经训练好的视觉运动模仿政策，提出一种离线数据选择框架 CFNBC，利用对抗性无关视觉干扰生成的任务保持的对照样本来评估并修复政策的鲁棒性。

**💡 创新点**

创新点在于：① 用“动作漂移”作为离线敏感性信号，衡量政策在任务保持的视觉扰动下的行为变化；② 通过覆盖多样化的高敏感性响应模式来选择紧凑的修复数据集；③ 将对照生成与政策特定审计相结合，避免无结构的数据增广。

**🔧 技术方法**

使用了基于 Transformers 的 Action Chunking with Transformers（ACT）模型、对照样本生成、动作漂移计算、响应多样性覆盖策略和对抗性无关数据选择等技术。

**📊 数据集**

实验使用了两大模拟抓取任务：MuJoCo 的双臂立方体搬运（A50）和 SimplerEnv 的单臂堆叠（A200），以及构造的 22 种任务保持的视觉扰动条件。

**📈 对比分析**

与随机采样和仅选取最大漂移样本的基线相比，在仅 20–30 条修复样本的低预算下，CFNBC 能在 Cube transfer 任务中将全部扰动条件的成功率提升至 0.53（相对基线提升 0.66），在 Cube stacking 任务中提升至 0.20（相对基线提升 0.44）。高预算随机采样可达 1.00，但需要 500 条样本。

**⚠️ 局限性**

局限性包括：① 需要预先构造满足任务保持假设的无关扰动样本；② 仅在模拟环境中验证，实际机器人部署仍需进一步验证；③ 动作漂移仅为敏感性指标，不能完全保证安全或处理多模态策略的情况。

---

## 4. Flat Score, Amplified Failures: How the Error Budget Masks Damage in Quantized LLM Agents

**arXiv ID:** 2607.27275 | [PDF](https://arxiv.org/pdf/2607.27275v1)

**作者:** Jiwon Jang `[一作]` (VAIV Company), Hyunwoo Park `[通讯]` (Seoul National University)

**通讯引用:** 2773 | [OpenAlex ID](https://openalex.org/A5100699674)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在多轮工具调用代理任务中评估 4 位权重量化对模型性能的真实影响，并揭示最终得分掩盖了显著的失败放大。

**💡 创新点**

创新点在于：① 通过细粒度失败通道分析揭示量化放大了模型已有的错误集合而非产生新错误；② 证明错误预算掩盖了量化导致的失效；③ 通过调节错误预算与针对失败通道的修复提示验证失效位置与恢复效果；④ 以模型全精度错误率为风险筛选指标。

**🔧 技术方法**

使用的技术包括：Post‑training AWQ 4‑bit量化；vLLM 服务器；对齐的错误通道日志（工具名称幻觉、实体/参数错误）；对比分析与等价测试（TOST）；错误预算的可逆性实验；基于对话日志的回溯和多任务聚合。

**📊 数据集**

使用的数据集是 τ²‑bench（双控工具调用基准）中的两大域：零售（单控）和电信（双控），共 114 个任务，每个细胞 456 次仿真，涵盖 Gemma‑4 与 Qwen‑3 系列的 dense 与 MoE 变体。

**📈 对比分析**

比较方法：在同一模型、同一域下对 BF16、FP8、INT4 三种精度进行对比，采用 95% 置信区间、等价检验、错误率与成功率随错误预算变化的曲线。结果显示：在最终得分上 4‑bit 与 16‑bit 无显著差异；但在工具调用失败率上量化可将失败放大 1~4.6 倍，且错误预算收紧后差距可达 16+ 分；针对失败通道的修复提示可显著提升受影响细胞的得分。

**⚠️ 局限性**

局限性包括：① 仅评估两大域与有限模型族，未覆盖更广泛的工具调用场景；② 量化仅在权重层面，未探讨激活量化与其他压缩技术的交互；③ 研究使用的预算调整在真实部署中的可行性与成本尚不明确；④ 修复提示依赖于可控的环境与对话日志，实际系统可能面临更复杂的错误类型。

---

## 5. Improved Robustness in AI-Generated Music Detection

**arXiv ID:** 2607.27454 | [PDF](https://arxiv.org/pdf/2607.27454v1)

**作者:** Emile Dugelay `[一作]` (Ecole Centrale de Lyon), Romain Hennequin `[通讯]` (Deezer Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种针对 AI 生成音乐的鲁棒检测方案，能够在速度修改攻击下保持高精度。

**💡 创新点**

创新点在于将音频映射到对数频率轴，使频率缩放变为平移，从而通过单个学习的交叉相关滤波器和最大池化实现对平移的不变性，构建了由设计决定的频率缩放不变检测管道。

**🔧 技术方法**

核心技术包括对数 STFT 重新映射、时间平均与基线下壳差分提取伪打印、单滤波器交叉相关与最大池化分类、以及联合二分类与峰值定位的混合损失。

**📊 数据集**

使用的数据集包括 5,000 条 Suno v5、Suno v3.5 与 Udio v120 的 AI 生成曲目，以及 5,000 条 FMA‑small 的真实音乐，总计 10,000 条 30 秒音频。

**📈 对比分析**

与 Afchar 等基线（仅基于谱峰）和 SpecTTTra‑α（Transformer 语义特征）相比，本方法在未改动音频上几乎匹配或超越基线，在速度修改攻击下保持 AUC≈0.997、F1≈0.986，远优于基线（AUC≈0.81、F1≈0.67）。

**⚠️ 局限性**

局限性主要是：仅针对速度修改攻击设计，对其他处理如均衡、压缩、编码等未测试；仅在三种生成模型上训练，无法直接泛化到其他 AI 生成器，需要针对新模型重新训练。

---

## 6. Benchmarking the Residual: What Long-Horizon Evaluations Add Beyond Matched Short-Task Performance

**arXiv ID:** 2607.27283 | [PDF](https://arxiv.org/pdf/2607.27283v1)

**作者:** Chao Peng `[一作]` (Tencent), Qiang Lin `[通讯]` (Tencent)

**通讯引用:** 16 | [OpenAlex ID](https://openalex.org/A5107899440)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7`

**🎯 论文内容**

本文提出一种诊断性长周期任务基准，要求在给定的阶段分解和检查点协议下计算“horizon residual”，即自然全程成功率与基于局部成功率的期望成功率的对数比。

**💡 创新点**

创新点在于将长周期失败拆解为错误累积、局部难度提升和历史依赖三类，提出可审计的产品基线与对数残差度量，并给出完整的基准设计与评估规范。

**🔧 技术方法**

采用了基于阶段成功率的乘法组合模型、对数残差分析、delta 方法估计不确定性以及可重放的检查点仿真与条件模型扩展技术。

**📊 数据集**

利用了多种软件工程与终端代理基准数据集，如 SWE‑EVO、NL2Repo‑Bench、ChainSWE、SWE‑Milestone、HORIZON 等；这些数据集提供可重放的仓库快照、工具调用日志和语义验证器。

**📈 对比分析**

通过将自然全程执行的成功率与从检查点得到的局部成功率乘积进行比较，计算对数残差；实验表明残差能揭示自然执行与局部预测之间的显著偏差，提示需要针对性干预；在已知基准上，该方法能量化并区分误差累积与历史依赖导致的失败。

**⚠️ 局限性**

局限性包括对阶段分解和检查点构造的高度依赖、对可重放状态的要求、可能存在的检查点兼容性问题、对非软件工程任务适用性不足，以及在多重协议下残差符号不稳定时缺乏鲁棒性判定。

---

## 7. Granite: A Modular Methodology for Foundational Verification of Hardware-Software Leakage Contracts

**arXiv ID:** 2607.27480 | [PDF](https://arxiv.org/pdf/2607.27480v1)

**作者:** Stella Lau `[一作]` (Massachusetts Institute of Technology), Adam Chlipala `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出一种可模块化的形式化验证方法，用来证明RTL级别的处理器在满足 ISA 协议的泄露约束下既功能正确又不泄露任何秘密信息，并给出了从软件到 RTL 的完整、可审计的非泄露证明。

**💡 创新点**

创新点在于：①首次将 ISA 级泄露合同与微架构的周期级执行紧密联系，构造可被证明的泄露无关规范；②利用“可泄露驱动器 + 非泄露见证 + 泄露变换器”三元组实现泄露感知的确定性细化；③在 Coq 中实现水平和垂直两条模块化证明链，支持从抽象子模块到具体 RTL 的逐层替换；④通过与静态常数时间分析相结合，最终消除 ISA 合同与软件层在可信计算基中的角色。

**🔧 技术方法**

使用的技术包括：Coq 形式化、确定性细化（determinism refinement）、存在性参数（driver, witness, leakage transformer）、Mealy 机器与 trace 等价、自由单子（freer monad）建模模块化硬件、one‑method‑at‑a‑time（顺序方法调用）语义、以及 RTL 代码的 SystemVerilog 生成与验证。

**📊 数据集**

实验数据集主要是：对 RISC‑V 子集的四级流水线核心进行验证，使用已编译的常数时间 Salsa20 二进制程序进行静态分析并通过上述证明链验证其在 RTL 上的泄露安全性；此外，还对零跳乘法器、FIFO、内存子模块等小型模块进行了单独验证。

**📈 对比分析**

相较于现有基于模型检查、UPEC、或仅验证功能正确性的工作，本文实现了完整的功能+泄露双重证明，并通过三阶段证明链（软件→ISA→RTL）实现了中间层的消除，证明过程完全可审计且无需信任 RTL 代码本身。性能方面，验证工作在 Coq 中完成，最终得到的 RTL 可用 Yosys/NextPnR 在 ECP5 FPGA 上成功综合，说明方法具备可实用性。

**⚠️ 局限性**

限制与未解决问题：仅覆盖单核、单时钟域、基于 RISC‑V 子集的处理器；不支持功耗、EM 等物理侧信道；未覆盖多核弱内存模型或复杂内存层次结构；需要手工编写和维护存在性参数；RTL 与 SystemVerilog 语义的等价性仍假设为可信；未来需要扩展至更大规模设计、异步时钟、以及与编译器验证的结合。

---

## 8. Does EEG Foundation Models Transfer to Speech? A Benchmark on Overt and Imagined Speech Decoding

**arXiv ID:** 2607.27268 | [PDF](https://arxiv.org/pdf/2607.27268v1)

**作者:** Owais Mujtaba Khanday `[一作]` (University of Granada), Jose A. Gonzalez-Lopez `[通讯]` (University of Granada)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对EEG基础模型与卷积基线在两大语音解码数据集上进行系统基准实验，评估其在清晰、隐性及想象语音中的性能

**💡 创新点**

首次在大规模预训练EEG模型与传统CNN基线进行统一的跨任务对比，揭示通用预训练并不提升语音解码效果

**🔧 技术方法**

使用LaBraM、EEGMamba等Transformer/State‑Space基础模型及EEGNet、ShallowFBCSPNet、EEGConformer等卷积网络，统一的预处理和微调流程

**📊 数据集**

UGR‑MINDVOICE（西班牙语清晰/隐性语音）和BCI Competition 2020 Track 3（想象语音）两大公开EEG语料

**📈 对比分析**

在留一被试交叉验证中，EEGNet在语音模式分类上显著优于基础模型；在语义类别和词汇级别任务中，所有模型几乎停留在随机水平；在想象语音任务中，传统基线略胜基础模型，且跨被试时所有模型均无法超越随机基准

**⚠️ 局限性**

缺乏针对语音的专门预训练数据，通用EEG预训练无法迁移到语音解码任务，且受限于低信噪比和被试差异，未实现跨被试的有效解码

---

## 9. Shared Semantic Codebook Distillation for Unpaired Cross-Modal Medical Classification

**arXiv ID:** 2607.27357 | [PDF](https://arxiv.org/pdf/2607.27357v1)

**作者:** Dillan Imans `[一作]` (Sungkyunkwan University), Hyunseung Choo `[通讯]` (Sungkyunkwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

在医学影像中，对未配对的不同模态进行知识蒸馏，利用共享语义代码表将教师和学生的特征映射到离散分布，实现跨模态知识迁移。

**💡 创新点**

提出共享语义代码书蒸馏方法，避免实例级匹配，采用分布级别和类级别的代码分布对齐，并在训练时用EMA更新代码表以保持多样性。

**🔧 技术方法**

使用共享离散代码表、soft assignment、KL/交叉熵对齐、类条件代码对齐、熵正则、EMA在线更新以及ResNet‑50 backbone等技术。

**📊 数据集**

在MultiEYE（OCT→fundus）6类视网膜疾病数据集和COVIDx CT‑3A与NIH ChestX‑ray（CT→CXR）二分类肺炎数据集上进行实验。

**📈 对比分析**

与学生仅模型、均值对齐、FDDM等基线对比，SSCD在两组任务上macro‑F1分别提升5.7点和2.5点，显著优于所有基线且推理无额外开销。

**⚠️ 局限性**

对罕见类别的对齐不够稳定；无法恢复教师的所有模态特征，性能仍低于教师上限；仅在完全共享标签空间有效。

---

## 10. THGFM: Dual-Branch Temporal Heterogeneous Graph Fusion Model

**arXiv ID:** 2607.27303 | [PDF](https://arxiv.org/pdf/2607.27303v1)

**作者:** Yixin Peng `[一作]` (RWTH Aachen University), Stefan Decker `[通讯]` (RWTH Aachen University)

**通讯引用:** 17470 | [OpenAlex ID](https://openalex.org/A5071104283)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种双分支时空异构图融合模型THGFM，用于学习多类型动态图。

**💡 创新点**

将共享空间时序注意力与关系分区注意力结合，并采用类型无竞争门控求和融合与旋转时间注意力，兼顾参数共享与关系专精并直接将相对时间影响注意力。

**🔧 技术方法**

基于Transformer架构实现SSTA、RTTA注意力，TC-NGSF门控融合，RoTA旋转时间编码，类型特定输入适配器，以及时间一致性采样等技术。

**📊 数据集**

在大规模学术图数据集OAG-CS、OGBN-MAG、HTAG-ArXiv和HTAG-DBLP上进行实验。

**📈 对比分析**

与HGT、CTRL、ieHGCN、SeHGNN等基线在相同预处理、采样和训练设置下对比，在六个任务上平均提升3.25%，单任务最高12.37%。

**⚠️ 局限性**

模型仅在单机GPU下评估，时间赋值方式导致批次间时间不一致，未探索跨图预训练及更广泛应用场景。

---

## 11. AnchorMark: Robust Diffusion Watermarking via Latent-Space Rotation Synchrony

**arXiv ID:** 2607.27551 | [PDF](https://arxiv.org/pdf/2607.27551v1)

**作者:** Yuqi Qian `[一作]` (Chinese Academy of Sciences), Meineng Zhu `[通讯]` (University of International Relations)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 AnchorMark，一种无需训练的基于扩散模型反演的鲁棒水印方案，可在图像旋转及其与其他损伤组合攻击下实现高精度多比特水印恢复。

**💡 创新点**

核心创新包括：①发现并利用旋转同步性（Rotation Synchrony）——图像旋转会在反演得到的初始潜在空间中产生相同角度的旋转；②在潜在空间中心嵌入多频率相位锚点，既不干扰原有水印载体，又能在反演后准确估计旋转角度；③采用分层（粗到细）同步与原始水印解码器的置信度校正，进一步细化角度并保证解码成功。

**🔧 技术方法**

技术手段包括：多频率相位锚点构造（低频+高频三角正弦对）；统计校准注入（匹配均值、方差、弱混合、后期校正）；离散时间逆扩散（DDIM 10 步）得到初始潜在；锚点匹配求解全局旋转；局部置信度导向的细化搜索；与原始水印解码器无缝集成。

**📊 数据集**

主要使用 Stable Diffusion v2.1（512×512，4×64×64 潜在）生成数据；评估数据集包括 Stable‑Diffusion‑Prompts（500 随机提示）用于水印检测/跟踪；COCO2017 提供 1,000 个提示评估 FID 与 CLIP‑Score；针对组合攻击进一步随机采样 SD v1.5、v2.1、v3.5、FLUX.1‑DEV 进行旋转+缩放/压缩/噪声/均值模糊的混合攻击。

**📈 对比分析**

与 7 种基线方法（DwtDctSvd、RivaGAN、Stable Signature、Tree‑Ring、RingID、Gaussian Shading、ShapeMark）在不同旋转角度（±5°、±10°、±30°、±60°）及组合攻击下对比。AnchorMark 在 ±60° 旋转时 TPR ≈ 1，Acc > 95%；在组合攻击下 TPR ≈ 0.999，Acc ≥ 95%。同时 FID 与 CLIP‑Score 与基线相比基本保持不变，表明视觉质量与语义一致性未受显著影响。

**⚠️ 局限性**

局限性：①仅针对扩散模型的潜在空间设计，可能在非扩散生成器或更深层变换下效果不足；②需要完成完整的反演过程，算力开销比纯后置嵌入略大；③锚点位于潜在中心，若在生成阶段使用大量边缘处理或裁剪，可能导致锚点失真；④对极端几何变形（如非均匀缩放、透视变形）尚未充分验证。

---

## 12. PanDent: Toward Comprehensive Tooth-Level Structure-Language Consistency in Dental Radiology

**arXiv ID:** 2607.27378 | [PDF](https://arxiv.org/pdf/2607.27378v1)

**作者:** Xiaohan Li `[一作]` (University of Hong Kong), Hui Chen `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了PanDent基准，用于评估牙科全景X光多模态大模型在细粒度牙齿级推理与结构-语言一致性方面的表现，并提供可用于训练的高质量标注集。

**💡 创新点**

创新点在于：①构建了大规模、临床专家验证的全景X光标注集，细粒度到单颗牙齿；②采用模板驱动的报告生成，确保结构与文本严格对应；③提出双轨评估（语言连贯性 + 临床准确性）和报告归一化机制。

**🔧 技术方法**

使用了多模态大模型（GPT‑5.x、GPT‑4.x、Gemini、Claude、Qwen 系列等）、LLM 辅助校验、模板化报告生成、报告归一化技术。

**📊 数据集**

使用了 PanDent 数据集（9524例全景 X 光 + 结构化牙齿级注释 + 模板化报告），并结合公开与内部采集的 X 光数据。

**📈 对比分析**

通过在 500 个保留样本上进行零样本和微调评估，使用 BLEU/METEOR/ROUGE‑L 与牙齿级属性精确匹配准确率衡量。结果显示：闭源模型在语言流畅度上最优，但牙齿定位准确率低；微调后 Qwen3‑VL‑4B 的牙齿级准确率提升约 59%，语言流畅度提升 83%。

**⚠️ 局限性**

limitation：数据中稀缺病症的长尾分布覆盖不足，模型在牙齿级定位仍存在明显误差，缺乏与视觉直接耦合的牙齿定位机制。

---

## 13. ZUNA1.1: A more flexible EEG foundation model for Denoising and Super-resolution

**arXiv ID:** 2607.27308 | [PDF](https://arxiv.org/pdf/2607.27308v1)

**作者:** Christopher Warner `[一作]` (Zyphra), Beren Millidge `[通讯]` (Zyphra)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出并训练了新的EEG基础模型ZUNA1.1，用于对缺失或受损的EEG信号进行高质量重建，支持可变长度（0.5–30 s）和任意通道数/位置。

**💡 创新点**

创新点包括：①可变长度训练与灵活的定位编码；②多种dropout混合和质量感知预处理；③多滤波变体与隐式数据增强；④大规模（≈3.5 M通道小时）训练与更长训练步骤；⑤使用混合损失与EMAs实现稳定训练。

**🔧 技术方法**

采用Transformer‑based扩散自编码器，配合4D‑RoPE位置编码、sandwich/QK‑norm正则化、MMD正则化、Rectified Flow损失，AdamW优化器与余弦学习率衰减。

**📊 数据集**

使用多来源公开EEG数据集（ANPHY‑Sleep、BerlinBCI、BCI2000、AAD等）共约3.5 M通道小时，涵盖睡眠、运动控制等任务。

**📈 对比分析**

与原始ZUNA1模型及MNE的球面样条插值法对比。ZUNA1.1在不同dropout率、区域遮挡和时间窗口任务上，NMSE普遍低于MNE插值，略优于或与ZUNA1相当，显示出更广泛的适用性。

**⚠️ 局限性**

局限性包括：对某些极端dropout（如全蒙版或高密度消费头戴）恢复仍不理想；模型容量与训练目标可能限制了重建精度；目前为重建任务优化，表示学习对下游任务的通用性有限。

---

## 14. Training Skills Like Parameters via Self-Supervised Semantic Diffusion

**arXiv ID:** 2607.27557 | [PDF](https://arxiv.org/pdf/2607.27557v1)

**作者:** Mo Li `[一作]` (Tsinghua University), Yunxin Liu `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个无监督的自我进化代理框架，通过将高质量剧本腐化为提纲并让代理重构，利用与原始剧本对比产生自监督信号，更新外部文本记忆，从而提升短剧本写作能力。

**💡 创新点**

采用腐化-重构的自监督策略替代传统标签或奖励，利用对比损失构建文本记忆更新机制，并将记忆结构化为Mixture-of-Experts的规则卡，实现可审计、可扩展、跨模型迁移的技能学习。

**🔧 技术方法**

以LLM为基础的代理与记忆库，使用文本对比损失、微步反向更新、Mixture-of-Experts（MoE）记忆架构、数据并行微步训练、规则解析器与LLM判定器评估。

**📊 数据集**

20部专业短剧本（共1127集）用于训练，以及6部保留作评估的剧本。

**📈 对比分析**

与空记忆基线在9个规则/判定指标上比较，+记忆在7/9项上超越基线，并在人类分布上大幅缩小距离；跨模型迁移也保持一致，整体提升显著。

**⚠️ 局限性**

依赖基础LLM的多指令遵循能力，若规则未被触发则无法学习；记忆检索不完全导致部分规则无法激活，且对模型行为的解释受限。

---

## 15. Multi-Player Discrete-Bidding Games; Determinacy, Equilibria, and Complexity

**arXiv ID:** 2607.27456 | [PDF](https://arxiv.org/pdf/2607.27456v1)

**作者:** Guy Avni `[一作]` (University of Haifa), Fatima Murra `[通讯]` (University of Haifa)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文首次研究多玩家离散竞价游戏，证明其在两联盟之间的确定性、存在纯策略纳什均衡以及均值收益游戏的价值存在性；

**💡 创新点**

创新点在于：1）引入“局部对全局确定性”框架，克服了传统阈值法在多玩家情境下的局限；2）给出了从可达性游戏到 PSPACE‑hard 的完整证明；3）证明了在离散竞价下存在纯策略纳什均衡；4）提出均值收益游戏价值存在的证明方法。

**🔧 技术方法**

主要技术包括：离散竞价矩阵构造、线性分配的拆分支付规则、局部对全局确定性理论、复杂度归约（从 3SAT/TQBF 到竞价游戏）、树形游戏的可达性递归分析、循环形成游戏（cycle‑forming game）构造。

**📊 数据集**

论文为理论性工作，未使用具体数据集，仅基于形式化模型与构造性证明。

**📈 对比分析**

比较方式为复杂度分析：证明两玩家离散竞价游戏为 NP∩coNP，本文多玩家版本已上升到 PSPACE‑hard；此外证明了纯策略纳什均衡的存在性和均值收益的价值存在性，未给出实验性能指标。

**⚠️ 局限性**

局限性：1）仅给出存在性证明，缺乏高效算法；2）均值收益游戏的纳什均衡尚未得到；3）在实践中竞价机制的实际收益与调度问题仍需进一步验证。

---

## 16. Asymmetric Collapse in Model Merging: When Refusal Over- writes Recognition

**arXiv ID:** 2607.27240 | [PDF](https://arxiv.org/pdf/2607.27240v1)

**作者:** Aarnav Choudhary `[一作]` (UCLA), Maheep Chaudhary `[通讯]` (Independent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在Gemma-3-1B-IT上分别微调的CAREs伤害等级分类模型与WildJailbreak拒绝模型，使用四种主流模型合并方法后，评估合并后模型在伤害识别、拒绝攻击与正常合规性上的表现；

**💡 创新点**

揭示标准合并方法在安全相关任务上会出现功能失衡——拒绝行为易被保留，而细粒度伤害识别往往被完全破坏，并将这一现象归因于任务向量幅度差异，而非任务方向冲突；

**🔧 技术方法**

采用线性平均、SLERP、TIES、DARE‑TIES四种合并技术，并结合任务向量分析、注意力头必要性掩码和数据规模归一化等实验手段进行机制解释；

**📊 数据集**

使用Gemma-3-1B-IT基模型，CAREs基准（约18k条医学安全级别提示）和WildJailbreak对抗数据集（约262k条示例）进行微调；

**📈 对比分析**

通过对合并后模型在CAREs四分类准确率、WildJailbreak拒绝率和正常合规率的评估发现：所有合并方法均能保持81–85%的拒绝率，但CAREs准确率仅维持≤12.9%，表明拒绝功能易被迁移而伤害识别易丢失；

**⚠️ 局限性**

实验仅在单一基模型和两种互补安全任务上验证，缺乏跨模型规模、任务种类或安全领域的泛化；合并权重固定为0.5，未探讨权重调节或任务向量归一化等更优策略；此外未评估对更广泛安全或通用能力的影响。

---

## 17. Beyond the Bidirectional Promise: Re-evaluating the Robustness of Diffusion Language Models

**arXiv ID:** 2607.27386 | [PDF](https://arxiv.org/pdf/2607.27386v1)

**作者:** Saurabh Yadav `[一作]` (Microsoft), Vijay Srinivas Agneeswaran `[通讯]` (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估扩散式语言模型(DLM)在自然噪声与对抗攻击下的鲁棒性与校准，并对比同参数的自回归模型；

**💡 创新点**

提出了两对配对模型评估框架、Diffusion‑GCG梯度探测器以及对输入前处理失效的机制性诊断；

**🔧 技术方法**

使用Diffusion‑GCG、线性探针、注意力分析、梯度幅度和多任务微调等技术；

**📊 数据集**

在TriviaQA、GSM8K和ARC‑Challenge三大认知任务上，采用九类字符/单词级自然噪声共32种扰动；

**📈 对比分析**

结果显示：自然噪声鲁棒性随权重而非架构决定，DLM普遍过度自信且对短梯度攻击有天然抵抗；配对模型中，LLaDA优于LLaMA‑3，Dream不如Qwen2.5；

**⚠️ 局限性**

局限包括仅评估7B–8B规模模型、噪声类型有限、未覆盖更大模型或更复杂任务、仅使用单一校准方法

---

## 18. EvoCause: LLM-Guided Evolution of Causal Graphs for Root Cause Analysis

**arXiv ID:** 2607.27290 | [PDF](https://arxiv.org/pdf/2607.27290v1)

**作者:** Lei Zan `[一作]` (Huawei Noah's Ark Lab), Lujia Pan `[通讯]` (Huawei Noah's Ark Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 EvoCause 框架，利用专家根因标签对基于因果发现的报警传播 DAG 进行迭代细化，最终实现无 LLM 调用的根因预测。

**💡 创新点**

创新点在于将专家标签转化为源节点约束，使用 LLM 仅生成语义上可行的图编辑候选，而非直接预测根因；并通过确定性验证与全局对齐集评估来挑选最佳图，保证图结构合法且符合专家诊断。

**🔧 技术方法**

技术组合包括传统因果发现算法（PC、NOTEARS、THPs）生成初始图；LLM（Qwen3）生成图编辑建议；确定性代码验证节点合法性与无环性；以节点 F1、Case EM 等指标进行对齐集性能评估，并在推理阶段仅使用最终细化图。

**📊 数据集**

实验数据包括：TeleRCA 生产网络专家标注数据（10,922 起因、485,681 警报、194 种报警类型）以及 10 组合成 Erdős–Rényi DAG，分别用于评估根因分析与图重构。

**📈 对比分析**

与 PC、NOTEARS、THPs、Chain‑of‑Event、CCCM、APGNN、RUN 等多种基线进行对比；在 TeleRCA 上 EvoCause 将 Node F1 从约 65% 提升至 92.6%，Case EM 从 54% 提升至 89.6%；在合成数据上亦显著提升图 F1 与 nSHD，表明细化效果明显。

**⚠️ 局限性**

局限性包括：专家标签仅约束源节点，无法唯一确定真实 DAG；对标签噪声与缺失敏感；未利用干预记录、时变传播或潜在共同原因；LLM 调用成本高且需外部依赖。

---

## 19. Theatre Chapbooks At Scale: A Statistical Comparative Analysis of Typography

**arXiv ID:** 2607.27266 | [PDF](https://arxiv.org/pdf/2607.27266v1)

**作者:** Diego Belzarena `[一作]` (Université Paris-Saclay), Jean-Michel Morel `[通讯]` (Lingnan University)

**通讯引用:** 35856 | [OpenAlex ID](https://openalex.org/A5057798674)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种统计方法，用于量化历史印刷书籍之间的字体相似度。

**💡 创新点**

创新点在于将字符原型聚类、自动对齐与 a contrario 统计框架结合，既能计算字体距离，又能评估其显著性。

**🔧 技术方法**

使用了字符图像聚类、自动字符提取与对齐、类型距离计算以及 a contrario 统计显著性检验。

**📊 数据集**

数据集为17世纪西班牙剧场小册子（chapbooks），规模超过人类专家可视检查的能力。

**📈 对比分析**

通过计算两本书的字体距离并利用 a contrario 框架判断显著性，方法在专家验证下发现并纠正了印刷者归属，表现出高准确率和可操作性。

**⚠️ 局限性**

局限性包括对图像质量和字符识别的依赖、对不同字体体系的适用性尚未充分验证，以及在大规模自动化时的计算成本。

---

## 20. Expanding Data-Agnostic Pivotal Instances Selection Models with Proximity Trees and Ensemble Learning

**arXiv ID:** 2607.27522 | [PDF](https://arxiv.org/pdf/2607.27522v1)

**作者:** Alessio Cascione `[一作]` (University of Pisa), Riccardo Guidotti `[通讯]` (University of Pisa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了PivotTree模型，通过选择代表性实例（pivots）来实现可解释的案例推理和预测；

**💡 创新点**

创新点在于引入分层的近似树结构，支持单个、双重（oblique）以及相对（proximity）分裂，并将PivotTree嵌入随机森林框架以构建可解释的集成模型；

**🔧 技术方法**

采用相似度函数构建特征空间，利用基于信息增益的贪婪树学习，探索univariate、multivariate（oblique）和proximity三种分裂方式，并实现随机PivotForest与分裂桩森林（stump forest）结合的逻辑回归压缩；

**📊 数据集**

在45个不同模态的数据集上实验，涵盖20个表格、10个时间序列、9个图像和6个文本数据集，所有数据均通过预训练模型映射到欧氏空间；

**📈 对比分析**

与k‑means、k‑medoids、ε‑ball等选择器以及knn、决策树、规则集、Boosting（XGBoost、LightGBM、CatBoost）等基线比较，评估指标为加权F1分数和pivot数量；结果显示PivotTree在保持极低pivot数量的前提下，往往与最优可解释模型同等或更优，且在多数数据类型中与基线相当；

**⚠️ 局限性**

局限包括对相似度度量的依赖（需手工选择或学习），训练时间在高维数据上仍显高，集成模型的可解释性下降，且在某些领域（如文本）性能略逊于Boosting方法；

---

## 21. PROGRESS: Property-Guided Regression Search for Semantic Falsification

**arXiv ID:** 2607.27359 | [PDF](https://arxiv.org/pdf/2607.27359v1)

**作者:** Davis Tocheuk Mo `[一作]` (University of Texas at Dallas), Soneya Binta Hossain `[通讯]` (University of Texas at Dallas)

**通讯引用:** 140 | [OpenAlex ID](https://openalex.org/A5065328894)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种将意图驱动的可执行属性直接作为搜索目标融入 EvoSuite 的 DynaMOSA 算法中，从而实现搜索驱动的语义错误检测。

**💡 创新点**

创新点在于：①使用两阶段语言模型管道从方法代码、文档及调用图上下文生成可执行的 jqwik 属性；②将每个属性视为独立的搜索目标，利用预条件推进搜索并以属性违例为最高优先级；③实现属性参数绑定与 jqwik 生成器的无缝协作，让搜索过程既能覆盖结构又能主动寻找语义缺陷。

**🔧 技术方法**

核心技术包括：EvoSuite 的 DynaMOSA 演化搜索、jqwik 的属性测试框架、OpenAI 的 Claude/Opus 大语言模型以及 Spoon/SootUp 等工具用于上下文提取。

**📊 数据集**

实验数据集为 OE25 公开的 25 个大型 Java 项目（包含 EvoSuite 基准与 Apache Commons 组件），共 240 个方法-文档对。

**📈 对比分析**

与传统回归测试生成和独立的 jqwik PBT 比较，PROGRESS 在 562 个注入型突变体中检测到 328 例（58%）错误，而回归测试全未检测；在 150 个难以触达的属性上，PROGRESS 达到 70/150（46%）的预条件满足率，显著高于 jqwik 的 18/150（12%），表明其在覆盖与错误揭露两方面均优于现有方法。

**⚠️ 局限性**

局限性包括：属性生成高度依赖 LLM 的质量与上下文提取的完整性，若生成的属性不可编译或失效则无法检测错误；目前仅支持 Java/JUnit/jqwik/EvoSuite 环境，未验证在其他语言或测试框架中的适用性；对极其复杂或格式严格的预条件，仍需人工编写专用生成器以进一步提升触达率。

---

## 22. SE(3)-MeanFlow: Few-Step Protein Backbone Generation on Lie Groups

**arXiv ID:** 2607.27431 | [PDF](https://arxiv.org/pdf/2607.27431v1)

**作者:** Yikun Bai `[一作]` (Purdue University), Guang Lin `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种少步生成蛋白骨架的 SE(3)-MeanFlow 框架，能够在仅数十步内生成可设计的蛋白链。

**💡 创新点**

创新点在于将 MeanFlow 迁移到 SO(3)×ℝ³ 的 Lie 群结构，推导出闭式平均速度目标并消除平行运输；同时设计 α‑Flow 无 JVP 暖启动和小‑t MeanFlow 稳定训练。

**🔧 技术方法**

使用 Lie 群平均速度公式、α‑Flow、Rectification（自回流）以及现有的 ReQFlow 结构进行实现。

**📊 数据集**

主要使用 SCOPe 蛋白骨架数据集（60–128 个残基的 3,673 条链）。

**📈 对比分析**

与 FrameFlow、QFlow、RMF 等流匹配/扩散基线进行比较；在 10–100 步的少步设置下，SE(3)-MeanFlow 在设计可行率、scRMSD、scTM 方面均优于基线，Rectified 版本进一步提升。

**⚠️ 局限性**

局限性是多样性/覆盖率略低，且目前尚未实现一步生成。

---

## 23. AHA-Memes: A Fine-Grained Multimodal Benchmark for Understanding Hate in Arabic Memes

**arXiv ID:** 2607.27393 | [PDF](https://arxiv.org/pdf/2607.27393v1)

**作者:** Mohamed Bayan Kmainasi `[一作]` (Qatar Computing Research Institute), Firoj Alam `[通讯]` (Qatar Computing Research Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了首个大规模阿拉伯语仇恨表情包基准AHA-Memes，包含5k人工注释与约66k银标；

**💡 创新点**

首次将细粒度多标签（攻击类型、目标）与多模态融合、零样本与少样本评估相结合；

**🔧 技术方法**

采用文本编码器（AraBERTv2、MARBERTv2等）、图像编码器（ViT、Swin、BEiT等）、后期融合、开/闭权重视觉语言模型、LoRA微调以及检索增强少量样本ICL；

**📊 数据集**

使用从Facebook、Instagram、Pinterest、X收集的5k人工标注阿拉伯语表情包及由Gemini生成的66k银标；

**📈 对比分析**

在二分类任务中fine‑tuned Qwen3‑VL‑8B实现macro‑F1 0.768；多标签任务最高macro‑F1 0.340；零样本性能弱，少样本提升有限；

**⚠️ 局限性**

数据集仅覆盖公开平台，方言与平台多样性不足；注释边界主观；银标非人工，存在噪声；模型对隐性仇恨识别仍显弱。

---

## 24. Hierarchical Reranking for Scalable Financial RAG System

**arXiv ID:** 2607.27523 | [PDF](https://arxiv.org/pdf/2607.27523v1)

**作者:** Joohyun Lee `[一作]` (Financial Security Institute), Sungwoo Hong `[通讯]` (Hanyang University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种面向金融领域的检索增强生成（RAG）框架 Hierarchical Reranker，旨在提高大型金融文档检索与生成的准确性与可部署性。

**💡 创新点**

创新点包括：① 预检索优化（词汇规范化、关键字扩展、Markdown 表格转 JSON、摘要压缩）提升检索语义匹配；② 两阶段层级重排序器（轻量快速模型 + 大模型深度排序）在保持高召回的同时提升精准度；③ 长上下文管理（输入分块、结果融合）降低 64k 以上输入对推理质量的负面影响。

**🔧 技术方法**

技术实现：规则化预处理、HyDE 伪文档生成、表格转 JSON、两阶段重排序器（jina‑reranker‑v3 与 Qwen3‑Reranker‑8B 等）、多模态相似度计算、长文本分块与条件融合、LLM 生成（Claude‑4.6 Opus、GPT‑5.4、Gemini‑3.0 Pro、Grok‑4.20 等）。

**📊 数据集**

数据集：FinQABench、FinQA、ConvFinQA、FinanceBench、TATQA，涵盖 10‑K 披露、财报、会话式查询等多种金融文本与表格场景。

**📈 对比分析**

比较方法：对预检索组件、层级重排序器、长上下文管理分别做消融实验；在 NDCG@20 上相较无预处理基线提升 5.9%，层级重排序器提升 6.5%；在多种 LLM 上通过分块融合提升 0.08–0.49% 的生成准确率；在 ACM‑ICAIF ’24 FinanceRAG Challenge 获得第二名，整体 NDCG@20 达 0.7918。

**⚠️ 局限性**

局限性：额外的层级重排序计算开销；固定 64k 上下文阈值限制极长文档处理；缺乏多语言与实时流式金融数据验证；预处理规则依赖人工定义，可能对新出现的格式或术语适应性不足。

---

## 25. TIER-MoE: Trust-Informed Expert Routing via Conditional Modality Risk for Multimodal Fusion in Biomedical Classification

**arXiv ID:** 2607.27289 | [PDF](https://arxiv.org/pdf/2607.27289v1)

**作者:** Yu Chang `[一作]` (University of Southern California), Paul Bogdan `[通讯]` (University of Southern California)

**通讯引用:** 5136 | [OpenAlex ID](https://openalex.org/A5105925385)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

本文提出一种TIER-MoE模型，利用样本特定的模态风险和专家子空间相容性进行稀疏路由，融合多模态信息以提高预测。

**💡 创新点**

创新点在于将交叉拟合的条件模态风险作为可靠性度量，与专家子空间亲和度结合进行路由，避免单纯依赖模态权重，同时保留共享交互路径。

**🔧 技术方法**

采用跨折风险估计、混合注意力、子空间Mixture-of-Experts、温度标度校准等技术。

**📊 数据集**

使用了ADNI（T1w+DTI）、PAD-UFES-20（病变影像+临床）和FPRM（眼底图像+血流视频）三大医学多模态数据集。

**📈 对比分析**

与传统融合（DAFT、GMU）、自适应融合（QMF、PDF）及专家路由（Flex-MoE、I^2MoE）等方法对比，TIER-MoE在Macro-F1、Brier分数、鲁棒性、零样本迁移等指标上均位列前列。

**⚠️ 局限性**

局限性包括需要额外的交叉拟合步骤来生成风险标签，模型结构相对复杂，且在极端模态缺失或跨域差异较大的情形下仍可能受限。

---

## 26. BMOA: Baseline-Mechanism-Outcome Attribution for Compiler-Induced Numerical Deviations

**arXiv ID:** 2607.27270 | [PDF](https://arxiv.org/pdf/2607.27270v1)

**作者:** Hailong Jiang `[一作]` (Youngstown State University), Qiang Guan `[通讯]` (Kent State University)

**通讯引用:** 1773 | [OpenAlex ID](https://openalex.org/A5102787759)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了BMOA诊断框架，对编译器产生的浮点数差异进行基准-机制-结果归因，生成可审计记录。

**💡 创新点**

创新点在于将差异拆解为基准定义、机制支持与结果评价三层，保留混合/未知标签，避免把误差直接归因为编译器错误，为后续形式化规范提供经验基础。

**🔧 技术方法**

采用严格浮点对比、对高精度引用的误差测量、对编译器变换的配对对照、汇编指令模式扫描、重复运行检验与跨编译器对比，以及ULP、绝对/相对误差等度量技术。

**📊 数据集**

使用六个科学计算小核（reduction、dot‑product、polynomial、reciprocal sum、threshold branch、FMA accumulation）与多族确定性压力输入，构成1276条BMOA记录和162个配置实例的实验集。

**📈 对比分析**

对每条记录分别做基准、机制和结果三种比较；所有实验在ARM64上完成，重复运行和跨Clang（17/22）无差异；结果显示机制既可提升也可降低准确性，基准选择直接影响诊断结论。

**⚠️ 局限性**

局限在于仅覆盖单一CPU、Clang 17/22、单线程小核、单一随机种子及固定高精度参考，未考察x86、GPU、并行、真实应用、不同编译器或多种参考；BMOA只能给出经验诊断，无法直接证明语义等价。

---

## 27. Evaluating the Vergence-Accommodation Conflict in Gaze-Based 3D Target Selection

**arXiv ID:** 2607.27369 | [PDF](https://arxiv.org/pdf/2607.27369v1)

**作者:** Mohammad Raihanul Bashar `[一作]` (Concordia University), Anil Ufuk Batmaz `[通讯]` (Concordia University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在Meta Quest Pro HMD中进行ISO 9241‑411多方向选择任务，比较了基于注视点的3D目标选择与基于控制器的射线投射两种方式，并在不同深度（用屈光度表示）下测量了运动时间、误差率、角度吞吐量和定位方差，旨在研究视差-调焦冲突（VAC）对注视点选择性能的影响。

**💡 创新点**

创新点在于首次系统性探究VAC对注视点驱动3D选择的影响，并提出以屈光度变化为基础的“ViD”模型，对Fitts定律进行深度感知改进，显著提升预测准确性；同时发现注视点选择在速度上优于手持控制器，但受深度影响更大。

**🔧 技术方法**

使用的技术包括：Meta Quest Pro HMD的固定焦点立体显示与72 Hz眼动追踪；Unity 2021.3与Oculus SDK实现交互界面；ISO 9241‑411选择任务与标准Fitts模型、CTD扩展与ViD扩展；统计分析采用RM‑ANOVA、ART变换、AIC/BIC模型比较。

**📊 数据集**

数据集为作者自行收集的实验数据：24名受试者共完成28,512个选择试次，包含6个屈光度层级、9个ID层级、2种指点方式。

**📈 对比分析**

实验通过与控制器对照比较，显示注视点在中等深度下运动时间最低、吞吐量最高；随着屈光度降低（即目标离焦距远），运动时间显著上升、误差率上升、吞吐量下降。模型比较表明，ViD模型在注视点选择上R²=0.93、控制器为0.84，AIC/BIC均显著优于标准Fitts和CTD模型。

**⚠️ 局限性**

限制包括仅在单一固定焦点HMD上测试，深度范围受限于典型XR界面距离；任务为受控选择，未涵盖导航或操作等更复杂交互；缺乏不同硬件/视差设置的跨平台验证；长期使用或多层深度的影响未探究。

---

## 28. Auditing Emergent LLM-Agent Collaboration through Cooperation-Obligation Coupling

**arXiv ID:** 2607.27429 | [PDF](https://arxiv.org/pdf/2607.27429v1)

**作者:** Zuyuan Zhang `[一作]` (George Washington University), Tian Lan `[通讯]` (George Washington University)

**通讯引用:** 6552 | [OpenAlex ID](https://openalex.org/A5018464968)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了iCORE表示法，用于将LLM‑agent合作过程的可观察交互、工作义务及责任分配统一编码，并构建事件触发的iCORE‑Audit干预层，实现全过程可审计；

**💡 创新点**

创新点在于提出联合表示X=(G,Q,Π)，将交互图、义务图和审计映射三者结合；定义工作完整性与代理分配稳定性两大可审计属性，并证明从局部合法更新到全局完整性与性能下界的理论保证；以及设计基于状态的诊断与干预框架iCORE‑Audit；

**🔧 技术方法**

使用形式化图模型（合作图、义务图、审计映射）、证书验证、局部合法更新规则、理论证明（工作完整性、分配稳定性、性能下界）、事件触发干预算法、实验评估工具；

**📊 数据集**

采用六个结构化协同任务（BC、DS、CM、DL、RV、CA）模拟环境，结合Qwen2.5‑0.5B‑Instruct在真实LLM下的实验，共计882个实验实例；

**📈 对比分析**

通过匹配实例、Wilcoxon检验与Bootstrap置信区间与MAS‑Only、Interaction‑only、Task‑only、LLM‑Judge等基线比较。iCORE‑A在所有6种模式下均优于基线，控制环境下轨迹质量提升约11.5%/26.4%，终端性能提升约15.1%/31.0%，差异在统计学上显著；

**⚠️ 局限性**

需要满足任务可分解、验证器足够、能力值已校准，否则iCORE质量仅为过程诊断；对开放式任务无法保证语义正确性；实验范围限于模拟与特定LLM，未涵盖更复杂真实场景。

---

## 29. The Convergence Behavior of Adam under Heavy-Tailed Noise

**arXiv ID:** 2607.27383 | [PDF](https://arxiv.org/pdf/2607.27383v1)

**作者:** Yijiang Pang `[一作]` `[通讯]` (Michigan State University), Yijiang Pang (Michigan State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在重尾噪声下给出普通 Adam 优化器的收敛性分析，提出了在没有参数耦合或梯度裁剪的情况下，Adam 对非凸目标的（ρ,ϵ）-鞍点收敛保证；

**💡 创新点**

创新点在于1）将在线转非凸转换框架推广到重尾马尔可夫差分噪声；2）对完整向量式 Adam 进行折扣化 regret 分析，突破了以往仅在简化或修改版 Adam 上的结论；3）揭示了仅通过输出裁剪即可实现最优重尾收敛速率。

**🔧 技术方法**

采用重尾在线学习理论（von Bahr–Esseen不等式）、折扣化在线学习到非凸转换、Adam 的 FTRL 表述、以及对自适应分母的高阶矩控制技术。

**📊 数据集**

未涉及实验或公开数据集，研究完全基于理论证明。

**📈 对比分析**

通过理论推导与已有的 SGD、Clip‑SGD、AdaGrad/Adam 变种等结果进行比较，证明普通 Adam 在重尾噪声下可达到（ρ,ϵ）-鞍点，虽然迭代复杂度比最优下界慢（约 𝒪(ϵ⁻¹³/²)），但在已知域半径时可恢复最优重尾指数。

**⚠️ 局限性**

局限性在于：① 普通 Adam 的迭代复杂度仍低于最优；② 需显式裁剪或已知域半径才能获得最优收敛；③ 仅适用于  p∈(4/3,2] 的重尾情况；④ 结果仅为期望收敛，无高概率界；⑤ 未给出实验验证。

---

## 30. A new approach for the determination of through-thickness and free-edge stresses in composite laminates based on structural elements

**arXiv ID:** 2607.27559 | [PDF](https://arxiv.org/pdf/2607.27559v1)

**作者:** Xiaopeng Ai `[一作]` (Delft University of Technology), Boyang Chen `[通讯]` (Delft University of Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于Kirchhoff-Love薄壳假设与结构共聚合物元素的复合层板三维应力恢复方法。

**💡 创新点**

创新点在于：①通过树脂富层厚度和力学性质推导惰性刚度，提供物理意义的共聚合物惰性刚度；②在Kirchhoff-Love框架下实现全三轴应力恢复，克服传统Mindlin‑Reissner壳体对间层破坏模拟的局限；③避免了共聚合物区间区间尺寸对模型精度的影响，提升了模型的可扩展性。

**🔧 技术方法**

使用Kirchhoff‑Love薄壳单元、结构共聚合物单元、惰性刚度解析公式、线性插值恢复、Abaqus UEL实现；结合解析公式和数值仿真完成模型构建与验证。

**📊 数据集**

无公开实验数据，采用数值模型（四层交叉铺、八层角度铺、准等向铺、一般堆叠）和文献中的解析解（Hashin、Mittelstedt 等）进行验证；材料参数取自相关文献，树脂富层参数根据典型实验值估算。

**📈 对比分析**

通过与解析解及高精度三维固体单元（C3D8）结果对比，展示应力分布高度一致；在计算性能上，相较固体模型每增量可节省约45% CPU 时间，显示出显著的计算效率提升。

**⚠️ 局限性**

局限性包括：①单层壳单元无法完全捕捉复杂跨层应力分布，线性插值导致边缘近似误差；②对树脂富层厚度与力学性质的估计仍需实验验证；③在非线性极端加载或高度非线性材料行为下的准确性尚未进一步检验。

---

## 31. Enhancing Law-Enforcement Audio Transcription: A LoRA-Based Adaptation of Whisper for BWC Footage

**arXiv ID:** 2607.27245 | [PDF](https://arxiv.org/pdf/2607.27245v1)

**作者:** Vivek Senthil `[一作]` (Rochester Institute of Technology), Ernest Fokoué `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 745 | [OpenAlex ID](https://openalex.org/A5070827124)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

使用LoRA对OpenAI Whisper-base模型进行参数高效微调，提升执法BWC音频转录准确性。

**💡 创新点**

提出仅训练0.3%参数即可显著降低WER的低秩适配方案，并发现r=8最优，避免过拟合。

**🔧 技术方法**

采用LoRA（低秩适配）技术，冻结原始权重，仅更新投影矩阵；基于Whisper-base Transformer。

**📊 数据集**

使用53条经过筛选的执法人员佩戴摄像头视频（42训练/5验证/6测试）及其人工字幕。

**📈 对比分析**

与零样本Whisper-base和全量微调模型对比，LoRA r=8在测试集上WER降至0.3733，比基线下降39.7%，显著优于全微调。

**⚠️ 局限性**

在噪声极大或复杂场景下性能仍受限，r>8导致过拟合，未能覆盖所有语境与OOV词，需进一步改进预处理与模型鲁棒性。

---

## 32. RoguePrompt: Dual-Layer Encoding for Self-Reconstruction to Circumvent LLM Moderation

**arXiv ID:** 2607.27373 | [PDF](https://arxiv.org/pdf/2607.27373v1)

**作者:** Benyamin Tafreshian `[一作]` (Boston University), Prathamesh Dhake `[通讯]` (Boston University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为RoguePrompt的黑盒单轮 jailbreak，采用两层编码（Vigenère + ROT13）与自我重构指令，绕过LLM的审核与安全控制。

**💡 创新点**

创新点在于设计了确定性双层异构编码与显式重构步骤，构造了可拆解的多阶段攻击框架，并提出了按可视接受、重构与执行三阶段分离的评估方法。

**🔧 技术方法**

主要技术包括文本跨度奇偶分割、Vigenère 加密、ROT13 加密、长度前缀序列化、自然语言包装指令，以及混合自动评估器（正则、词向量相似度与LLM判定）。

**📊 数据集**

使用了 StrongREJECT 数据集，共 313 条针对不同违规类别（如性、暴力、仇恨、非法商品、非暴力犯罪、虚假信息等）的提示。

**📈 对比分析**

与五种基线（Auto Payload Splitting、Base64 Raw、Disemvowel、Paired‑Request Concatenation、PAP）在三种模型（GPT‑4o、Claude 3 Opus、Gemini 1.5 Pro）上进行三次独立试验；RoguePrompt 的 Execution@3 达到 70.18%，明显高于最佳基线 33.97%，同时在 Bypass@3 与 Recon.@3 上亦表现领先。

**⚠️ 局限性**

局限性包括评估仅基于 2025 年 4–5 月的模型快照，后续更新可能导致结果变化；仅测试单轮、固定 3 次查询的黑盒情形，未覆盖多轮或长上下文攻击；使用自动化评估，可能存在标注误差；样本量为 313 条，未必代表所有违规场景。

---

## 33. PIE-APT: A Unified Framework for Temporal Planning and Contradiction Hunting via Incremental Direct-Derivation Abduction

**arXiv ID:** 2607.27287 | [PDF](https://arxiv.org/pdf/2607.27287v1)

**作者:** Amir Hossein Sharafi `[一作]` (Najm), Alireza Shahbazi `[通讯]` (Najm)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了PIE-APT框架，将PIE-Abducer（增量直接推导的归纳推理）与PIE-APT（基于A*的生成-测试规划）结合，实现在OWL 2 DL层面上对动态知识图进行完整的推理与规划，解决Ramification问题并支持开放世界假设与对抗性矛盾检测。

**💡 创新点**

核心创新点包括①利用增量直接推导代替传统MHS枚举实现高效归纳；②在纯DL中原生定义OWL动作并通过非递归状态更新规避闭包问题；③采用生成-测试（Generate‑and‑Test）A*结构将规划与归纳相互嵌套；④提出对抗性矛盾挖掘（Contradiction Hunting）作为诊断与红队工具。

**🔧 技术方法**

技术手段包括OWL 2 DL增量推理器、直接推导消解、SPARQL查询、A*搜索、递归生成-测试、Skolem化与身份维护、异常检测与并行化优化。

**📊 数据集**

实验使用四个OWL基准：Bank Account（参数化目标与Witness搜索）、Derived Gate（中途TBox推理）、Physical Security（开放世界假设注入）和Tax Paradox（对抗性矛盾挖掘），覆盖多种动态知识图推理挑战。

**📈 对比分析**

与Fast Downward导出的PDDL进行语义功能对比，验证PIE-APT在四种基准上具备不可在PDDL中实现的功能；与MHS（AAA）基线在归纳阶段进行定量比较，PIE-Abducer在速度上显著优于MHS并在所有基准中取得更优的解释与计划质量。

**⚠️ 局限性**

局限性包括：仅在OWL 2 DL范围内工作，TBox不可动态修改；对极深或高度非递归动作仍可能产生空间爆炸；并行化受限于可复制的增量推理器实例，导致大规模知识图的计算资源需求仍较高。

---

## 34. Leveraging Trajectory Graphs for Pre-Execution Error Diagnosis in Agentic LLM Systems

**arXiv ID:** 2607.27443 | [PDF](https://arxiv.org/pdf/2607.27443v1)

**作者:** Xu Zheng `[一作]` (Florida International University), Dongsheng Luo `[通讯]` (Singapore Management University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于概率图模型的图神经网络框架，在LLM agent执行前诊断动作错误并提供预警。

**💡 创新点**

将长序列交互建模为POMDP，构造动作中心的概率图并用GNN进行步级错误检测，将检测模块作为“调试沙盒”无缝集成进LLM agent。

**🔧 技术方法**

使用PGM图构建、文本GCN/Graph Neural Network、BERT嵌入、In‑Context Learning反馈以及预训练LLM（GPT4o‑mini、Qwen2.5‑14B、Gemma3‑27B）。

**📊 数据集**

在四个长周期任务基准（AlfWorld、TextWorld、ScienceWorld、TravelPlanner）上构建轨迹数据并人工标注六类错误。

**📈 对比分析**

与文本分类（TF‑IDF、BERT）、检索/ RAG、LLM‑as‑judge 等基线对比，检测准确率平均提升约5%，AlfWorld 最高提升10%；在PASS Ratio 上平均提升 14.69%，在四大基准上均超过基线。

**⚠️ 局限性**

需要先收集并标注大量轨迹数据，耗时；未实现动态更新图和轨迹池的持续学习。

---

## 35. The Kinetics of Training: A Driven-Nucleation Rate Law for Emergence, Plasticity Loss, and Circuit Control in Language Models

**arXiv ID:** 2607.27281 | [PDF](https://arxiv.org/pdf/2607.27281v1)

**作者:** Lei Dong `[一作]` `[通讯]` (Independent Researcher), Lei Dong (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过建立一套化学动力学式（类似TTT图）解释并预测大型语言模型中能力（capability）出现的时间与速率，提出并验证了无部分信用（no‑partial‑credit）对齐、指数‑K 代价以及相关的阈值、时钟、寿命等规律。

**💡 创新点**

创新点：
- 将学习过程视为核磁率（nucleation）过程，给出明确的速率方程 J = sites × ν_attempt × σ(c) × e^(‑βK) – D；
- 证明“无部分信用”是导致指数‑K 代价的根本机制；
- 提出前向（出现）与后向（可塑性消失）两条阅读，形成完整的时间温度转换（TTT）框架；
- 通过实验演示数据驱动是唯一的“温度”，并提出注入等离子噪声等控制手段实现过程控制；
- 通过多模型（Pythia、OLMo）和多尺度（70 M–1 B）验证常数 β、时钟比例、阈值等是跨模型、跨规模的物理常数。

**🔧 技术方法**

技术手段：
- 经典随机微分方程（SDE）与大偏差理论相结合，推导逃逸速率；
- 通过控制实验（toy transformer、合成序列任务）与真实模型（Pythia、OLMo）测量 β、阈值、时钟等参数；
- 采用前向/后向阅读、预注册实验、Cox–Kaplan 等统计方法；
- 注入等离子噪声、熔点线（melt line）与冷却（nose）控制实验。

**📊 数据集**

数据集：
- 合成序列任务（和/奇偶性、和加法、复制、词对回忆等）；
- 真实文本训练数据（Pythia、OLMo 的公开检查点和对应的标注任务）。

**📈 对比分析**

比较方法与性能：
- 对 6 个未见模型进行预注册预测，平均 5 % 的误差；
- 对 32 个判定性单头消融实验，所有 32 个单元均低于部分信用预测（<25 %），p < 2×10⁻¹⁰；
- 通过多尺度、跨模型的恒定 β 与时钟比例验证了理论的普适性；
- 在控制实验中实现了熔点线、冷却曲线等，可实现能力的按需启动与恢复。

**⚠️ 局限性**

局限性：
- 依赖于“混合”假设（transformer SGD 在形成时间尺度内足够混合）尚未严格证明；
- 只验证了两类公开模型（Pythia、OLMo）且规模 ≤1.4 B；
- 机制对 “无部分信用” 的前提在真实任务中可能不总成立；
- 缺少对更大规模或更复杂任务（如多模态、超大模型）的验证；
- 控制实验（噪声注入、熔点线）并未直接在生产训练中应用，仍属于实验性手段。

---

## 36. Complexity of Strong Popularity in Additively Separable Hedonic Games

**arXiv ID:** 2607.27277 | [PDF](https://arxiv.org/pdf/2607.27277v1)

**作者:** Matan Gilboa `[一作]` (University of Oxford), Matan Gilboa `[通讯]` (University of Oxford)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5075878350)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究并证明在加性可分离霍德尼游戏（ASHG）中判定是否存在强流行分配（Strong Popular Partition）是 P^C-WINNER 完全问题。

**💡 创新点**

创新点在于首次将强流行分配的存在性问题定位于一个新的、介于 P 与 P^2 之间的复杂度类 P^C-WINNER，并给出了从 Condorcet Winner 问题的多项式时间归约，揭示了强流行分配的“唯一性”特性与复杂度之间的深层关联。

**🔧 技术方法**

核心技术是构造精细的游戏-电路映射：利用一系列 gadget（复制、非、与门等）以及单向复制代理，将布尔电路中的输赢关系转化为 ASHG 中各 agent 的效用和投票边际，从而使得 Condorcet 字符串与强流行分配之间产生一一对应。

**📊 数据集**

本工作为理论研究，无使用实际数据集，所有结果均通过形式化证明与计算复杂度分析得出。

**📈 对比分析**

方法与性能的比较不涉及实验评估，而是通过复杂度理论的视角：证明了存在性问题不在 P 内且低于 P^2 的已知上界，确立了该问题的 P^C-WINNER 完整性。

**⚠️ 局限性**

局限性包括：仅关注 ASHG，未讨论更广泛的霍德尼游戏类型；所给结果仅为判定问题的复杂度，未提供高效算法或近似方案；对实际应用的可行性尚未探讨。

---

## 37. Comparison of a Parametric Physics-Informed Neural Network and a Tensorial Reduced-Order Model for the Shallow-Water Dam-Break Problem

**arXiv ID:** 2607.27433 | [PDF](https://arxiv.org/pdf/2607.27433v1)

**作者:** Anton Myshak `[一作]` (University of Houston), Ilya Timofeyev `[通讯]` (University of Houston)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一维浅水坝破裂问题的参数化模型，提出并对比了无时间积分的参数化物理信息神经网络（PINN）与非侵入式张量降阶模型（TROM）两种快速预测框架。

**💡 创新点**

创新点在于将冲击感知采样、残差缩放和PDE门控等技术结合到PINN中显著提升冲击波主导下的鲁棒性，并提出在参数空间中通过低秩张量表示和局部基变换实现的高效TROM逼近。

**🔧 技术方法**

使用了SiLU激活函数、Adam优化、均值-标准差归一化、EMA残差尺度、Shock‑aware collocation、PDE门控、Cubic spline插值、HOSVD张量分解等技术。

**📊 数据集**

基于一维浅水方程的高精度有限体积仿真，参数网格为(h_L∈[10,28], h_R∈[0,8])，训练集包含65个参数实例，时间步长Δt=10⁻⁴，采样间隔0.1，输出水深h和流量q。

**📈 对比分析**

通过在6个未见参数组合和外推案例下计算相对L²误差、光谱一致性以及训练/推理耗时进行比较，发现PINN在湿床情形下误差更低，TROM在干床情形下更优；两模型在推理阶段均比完整模型快约500–700倍。

**⚠️ 局限性**

对干/近干床（h_R≈0）和冲击波移出域时的高频细节预测仍受限；PINN训练成本高昂，TROM在参数外推时受限于插值误差，且两者均需在参数网格上具备完整样本。

---

## 38. Selecting Open-Weight Language Models for Zero-Shot Intent Classification: A Systematic Evaluation of 41 Models

**arXiv ID:** 2607.27421 | [PDF](https://arxiv.org/pdf/2607.27421v1)

**作者:** Parishruthi Ganesh `[一作]` (Auburn University), Cheryl Seals `[通讯]` (Auburn University)

**通讯引用:** 542 | [OpenAlex ID](https://openalex.org/A5029399576)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 41 个开放权重 LLM（135M–9B 参数）在 8 个零射击意图分类数据集（加上 5-shot ATIS 作为辅助）上进行系统评估，涵盖准确率、置信校准、鲁棒性、统计显著性、部署效率等多维指标。

**💡 创新点**

①将 instruction-tuning 与参数规模对比，发现前者在 sub‑9B 范围内往往更重要；②利用 McNemar 检验显示单一基准（如 MASSIVE）无法显著区分顶尖模型；③指出 SNIPS 已饱和，应从评测中移除；④在同一数据集上分析误差模式，揭示标签歧义；⑤在部署层面提出 Pareto‑optimal 模型集合。

**🔧 技术方法**

采用 vLLM 0.6.3 GPU 端推理、free‑text 生成（无约束解码）、温度 0.0、最大 20 token；评估使用 ECE 与二元 Brier 分数做校准，统计显著性使用 95% 置信区间与 McNemar 检验；鲁棒性测试加入字符错别字、全小写、标点去除、相邻词交换等扰动；部署效率测量参数量、显存占用、平均推理时延，计算 Acc/B。

**📊 数据集**

CLINC150、Banking77、SNIPS、MASSIVE、MTOP、Curekart、Powerplay11、Sofmattress（8 个零射击集）以及 ATIS（5-shot）。

**📈 对比分析**

通过将 8 个零射击数据集加权平均得到 group‑balanced 统一指标，并与各模型在单个数据集上的排名做对比；结果显示 Mistral‑7B‑Instruct‑v0.3 以 0.660 的综合准确率领跑，Qwen2.5‑3B‑Instruct 以 0.632 取得最佳 sub‑4B 性能；在鲁棒性上，Qwen2.5‑7B‑Instruct 在字符错别字扰动下降幅最小；在校准上 instruction‑tuning 效果不一，Qwen2.5‑7B‑Instruct 拥有最低 ECE。

**⚠️ 局限性**

①只评估单语（英语）且为单标签分类；②子集选择非随机（前 500 条）可能导致偏差；③校准分析仅限于 MASSIVE，因标签空间紧凑；④鲁棒性扰动测试仅使用单一随机种子；⑤不评估多标签意图检测或多语言场景。

---

## 39. Open Security Benchmark: Towards Autonomous Enterprise Cyber Defense

**arXiv ID:** 2607.27288 | [PDF](https://arxiv.org/pdf/2607.27288v1)

**作者:** Gal Engelberg `[一作]` (open.security), Leon Goldberg `[通讯]` (open.security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 Open Security Benchmark (OSB) 框架，用来评估自主网络防御中的 agentic AI 在安全态势评估任务上的表现。

**💡 创新点**

创新点：① 用冻结的跨供应商企业安全态势快照填补“环境数据缺口”；② 引入双模态评估（text‑to‑SQL 与原生 API）保证与实际使用的一致性；③ 设计多维度分层评分体系和可审计的简易运行环境，支持公开与私有评估并行。

**🔧 技术方法**

技术手段：自然语言到 SQL 转换、LLM 评估面板、结构化指标（表/连接召回/精度）、关系数据库快照、供应商 API 仿真器、微调/强化学习训练循环等。

**📊 数据集**

数据集：合成企业环境（包含 AWS、Azure、GCP、Okta、GitHub 等 8 家供应商），共 44 张关系表，提供小/中/大三种规模（约 75、400、2000 名员工），每个规模均可用于任务集合。

**📈 对比分析**

比较方法：公开 Benchmark 包与私有套餐共用同一快照与评分规则，评估分为答案正确性、判定、推理实用性、SQL 质量、结构表/连接召回/精度等多指标；框架能输出详细分数卡，支持跨模态、跨规模、跨任务维度的细粒度比较，性能数据需通过用户实验获取。

**⚠️ 局限性**

局限性：① 仅基于合成环境，缺乏真实操作漂移与事件流；② 当前仅覆盖安全态势评估阶段，后续风险优先、补救等环节未完全实现；③ 评分体系对某些高级推理或多步骤交互的细粒度评估仍有提升空间。

---

## 40. Sympathetic Framing: Evaluating AI Alignment across Sociodemographic Groups

**arXiv ID:** 2607.27232 | [PDF](https://arxiv.org/pdf/2607.27232v1)

**作者:** Haran Shani-Narkiss `[一作]` (University College London), Oren Tsur `[通讯]` (Ben Gurion University Negev)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一项大规模的问卷调查中，将英国成人3011人对216条新闻标题的情感共情判断与七款大型语言模型（ClaudeSonnet4.5、GPT4、GPT5.2、Grok4.1-Fast、Gemini3-Flash、MistralLarge2512、DeepSeek3.2）在同一问题上的回答进行对比，评估模型对新闻情感框架的理解程度。

**💡 创新点**

首次系统性、规模化地量化不同 LLM 与人类对情感框架感知的相似度，并细致探讨该相似度在年龄、教育、政治意识、先前知识等社会人口子群体中的差异，揭示“差异化对齐”在伦理与社会治理中的潜在重要性。

**🔧 技术方法**

采用 Spearman 相关系数衡量人类与模型对每条标题共情比例的对应关系；通过对模型在不同子群体和主题下的相关系数进行统计检验（置换检验）来确定显著性；对模型间的层级差异进行可视化对比。

**📊 数据集**

使用从 GDELT 项目筛选的 216 条与俄罗斯-乌克兰战争、加沙战争及 2024 年美国总统竞选相关的新闻标题；问卷数据来自 YouGov 招募的英国成人代表性样本（3011 人），包含年龄、性别、教育、政治意识、先前知识等信息。

**📈 对比分析**

结果显示 GPT‑5.2 与人类共情判断的 Spearman ρ ≈ 0.79，最高；Grok ≈ 0.74，GPT‑4 ≈ 0.71；Gemini、DeepSeek ≈ 0.66–0.67；Claude ≈ 0.58，Mistral ≈ 0.41。模型表现随主题和子群体波动，老年人、无正规教育者和低政治意识群体的对齐程度相对较低，但总体仍保持显著正相关。

**⚠️ 局限性**

主要局限包括：模型版本频繁更新导致结果易变；只使用了 216 条标题且未考虑标题间交互；未对人口特征与内容类型的交互效应进行深入分析；差异化对齐方案可能引入新的伦理偏差。

---

## 41. O-RAN: Analysis of Latency-critical Interfaces and Overview of Time Sensitive Networking Solutions

**arXiv ID:** 2607.27448 | [PDF](https://arxiv.org/pdf/2607.27448v1)

**作者:** Esteban Municio `[一作]` (i2CAT Foundation), Xavier Costa-Pérez `[通讯]` (i2CAT Foundation)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了TSN技术在O-RAN中的应用，绘制了TSN与O-RAN接口的映射框架，并提出了多种部署选项及其成本收益分析。

**💡 创新点**

创新点在于系统性地将IEEE 802.1Qbu、Qbv、Qcm等TSN扩展与O-RAN OpenFronthaul、F1、NG-U等关键接口对应，并对调度与非调度部署的权衡进行深入讨论。

**🔧 技术方法**

主要技术包括IEEE 802.1系列TSN标准（Qbu、Qbv、Qcm、Qci、Qch等）与O-RAN架构（OpenFronthaul、F1、E2、A1、NG-U等接口）以及时钟同步技术（PTP、SyncE）。

**📊 数据集**

本研究主要引用公开标准文档、行业白皮书与相关技术报告，未使用实验数据集。

**📈 对比分析**

通过与传统CPRI/ eCPRI专线和共享以太网的对比，评估了延迟、PDV和TCO等指标，表明TSN能实现≤100µs的延迟，并在共享网络中降低约30% TCO。

**⚠️ 局限性**

局限性包括需要部署CNC/CUC实现调度，管理复杂度升高；缺乏实测验证；对高频波段、极低PDV等特殊场景的适用性仍待进一步研究。

---

## 42. SkillSmith: Learning to Compose Parametric Skills and Textual Knowledge

**arXiv ID:** 2607.27497 | [PDF](https://arxiv.org/pdf/2607.27497v1)

**作者:** Lucio M. Dery `[一作]`, Arthur Szlam `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建一种能同时处理文本和模型权重的增强型LLM（SkillSmith），用于合成新的前缀参数以解决新任务。

**💡 创新点**

将权重空间视为可读写的模态，利用增强LLM（超网络）在单一模型中融合文本说明与前缀权重，从而实现跨模态组合与自适应。

**🔧 技术方法**

前缀调优（prefix‑tuning）、K‑V适配器、超网络（hyper‑network）、Gemma 3 4B 作为基础模型、Retriever + Gemini Embeddings 等技术。

**📊 数据集**

Composite‑SNI（合成组合任务）、Super‑Natural Instructions（SNI）以及 MMLU‑ProX 多语种多任务数据集。

**📈 对比分析**

与 LERP、Concat、SVD、ICL、直接前缀调优等单模态基线对比；在 Composite‑SNI 零样本中取得最高 Elo 分，微调后显著优于所有基线；在 SNI 与 MMLU‑ProX 的真实场景中，SkillSmith 通过检索和预训练仍保持较高性能，尤其在数据稀缺任务上表现突出。

**⚠️ 局限性**

对检索误差敏感；仅在前缀长度固定时效果最佳；在极大规模任务或多模态复杂度极高的场景中尚未验证；模型规模受限于 4B 参数。

---

## 43. Bridging Inference-Time Scaling and Episodic Memory with Action-Centric Graphs

**arXiv ID:** 2607.27415 | [PDF](https://arxiv.org/pdf/2607.27415v1)

**作者:** Xu Zheng `[一作]` (Florida International University), Dongsheng Luo `[通讯]` (Singapore Management University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建基于动作的图并使用双流Temporal Difference学习，从历史轨迹中提炼成功与失败的价值，引导LLM代理进行高效的推理时间扩展；

**💡 创新点**

将记忆机制与LLM推理分离，形成动态动作中心图；双流价值估计同时捕捉正向成功和负向风险；将图中价值转化为In‑Context提示，实现在推理过程中对搜索空间的软约束与引导；

**🔧 技术方法**

动作中心图建模、双流TD学习（Q⁺、Q⁻）、In‑Context Learning提示、Best‑of‑N采样、无额外LLM推理的搜索约束；

**📊 数据集**

AlfWorld、ScienceWorld、PDDL、Tool‑Query等四大公开基准；

**📈 对比分析**

在四个开源LLM（Qwen、Llama、Gemma）与闭源GPT‑4o‑mini上，采用Best‑of‑N评估。与记忆基、学习自适应及传统推理扩展基线相比，平均成功率提升约20.81%，进展率提升6.17%，并在大多数基准上获得最优或接近最优成绩；

**⚠️ 局限性**

依赖足够的warm‑up轨迹；若轨迹不足，易陷入子最优；需要维护并更新动作图，虽开销低但仍非零；适用性受限于任务多样性与历史轨迹的代表性。

---

## 44. KernelGenBench: A Multi-Source and Multi-Chip Benchmark for LLM-based Kernel Generation

**arXiv ID:** 2607.27231 | [PDF](https://arxiv.org/pdf/2607.27231v1)

**作者:** Peiyu Zang `[一作]` (Beijing Normal University), Yonghua Lin `[通讯]` (BAAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 KernelGenBench 基准，统一评估 LLM 与 agent 生成的 Triton 加速器核，涵盖 210 个多源操作符（ATen、vLLM、cuBLAS）与 6 个异构硬件平台。

**💡 创新点**

创新点：①双子基准（Multi‑Source 与 Multi‑Chip）实现跨源、跨平台的系统评估；②三层防作弊机制（AST 静态扫描、Ghost Replay、硬件追踪）保证结果真实性；③统一的分布式沙箱与几何平均性能评估框架，使得功能正确率、速度提升与 token 成本可并行对比。

**🔧 技术方法**

技术与方法：大模型采样（Pass@1/5）、agentic 框架（Claude Code、OpenCode、AutoKernel、AKO4all 等）、Triton 编译器、分布式沙箱、AST 解析、Ghost Replay、硬件追踪、数值验证、三层 anti‑hack、几何平均 speedup 计算、token 计数与时间统计。

**📊 数据集**

数据集与实验集：210 个操作符（110 ATen、50 vLLM、50 cuBLAS）；110 个 ATen 子集用于跨平台测试；六个硬件平台（NVIDIA A100 + 5 个匿名平台）；每个操作符生成多形态输入组合，形成完整的测试套件。

**📈 对比分析**

对比与性能：agentic 方法在正确率与 speedup 上普遍优于纯采样，尤其在 ATen 和 vLLM 操作符上；cuBLAS 操作符始终最难；跨平台表现差异显著，kernel‑specialized agent 在非 NVIDIA 平台的准确率下降 50%+；token 成本方面，agentic 方法平均每个成功操作需 5.11 M token，远高于单步采样；速度提升最高 1.63×（vLLM）但在跨平台时可降至 0.25×。

**⚠️ 局限性**

局限性：①未实现完整的 Multi‑Chip × Multi‑Source 交叉基准，覆盖 900+ ATen 操作符仍待完成；②单轨迹评估成本高，无法多次复测；③评估衡量的是 LLM+框架整体表现，难以拆分出 LLM 自身能力与 agentic scaffold 的贡献。

---

## 45. Dimensionality and Measurement Precision in HLE's Multiple-Choice Subset

**arXiv ID:** 2607.27420 | [PDF](https://arxiv.org/pdf/2607.27420v1)

**作者:** Mayank Sharma `[一作]` (Stanford University), Tyler Matteson `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 Humanity's Last Exam（HLE）基准的心理测量属性，检验其是否衡量单一推理因子及其测量精度分布，使用29个LLM在文本多项选择子集上进行实验。

**💡 创新点**

通过IRT、McDonald’s ω_h、PCA、残差相关等多种方法提供了HLE单一维度的实证证据，并揭示了高能力模型区分精度不足的现象。

**🔧 技术方法**

采用了二参数Logistic IRT模型、McDonald’s ω_h、主成分分析（PCA）、残差相关分析以及测试信息函数（TIF）等心理测量技术。

**📊 数据集**

使用了HLE文本多项选择子集（428道题）以及29个LLM的回答数据。

**📈 对比分析**

通过IRT估计项参数和测量精度，发现总分与域分数高度相关，域分数几乎无增量信息；TIF显示在中等能力区有高精度，而在高能力区精度急剧下降，说明HLE对顶尖模型的区分力有限。

**⚠️ 局限性**

局限包括仅使用文本多项选择子集（约19%），模型样本小且非随机，缺乏对图像与精确匹配题目的分析，缺乏测量不变性检验，且可能低估项参数的不确定性。

---

## 46. Modeling Decisions in Blockchain Analytics: A Leakage-Aware Evaluation of Tree-Based vs. Sequential Models

**arXiv ID:** 2607.27350 | [PDF](https://arxiv.org/pdf/2607.27350v1)

**作者:** Michał Bartnicki `[一作]` (Warsaw University of Technology), Jarosław A. Chudziak `[通讯]` (Warsaw University of Technology)

**通讯引用:** 133 | [OpenAlex ID](https://openalex.org/A5008057050)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个泄漏感知的以太坊行为分类框架，提出 Blind-Spot 协议去除高信号合约泄漏，并使用 Transaction Grammar 对钱包交易历史进行层次化编码，随后比较了序列模型（Transformer、BiLSTM）与树模型（XGBoost、SVM）的表现。

**💡 创新点**

创新点包括：①Blind-Spot 协议消除特定合约带来的标签泄漏；②将 EVM 执行轨迹与时间节奏统一为可序列化的语法表示；③在相同泄漏控制下首次系统比较序列与表格模型的性能与效率；④针对低延迟部署提供能耗与推理时延对比。

**🔧 技术方法**

技术手段：Transformer（平面与层次化）、双向 LSTM、XGBoost、线性 SVM、Token 计数、熵与 Lempel‑Ziv 复杂度特征、位置编码、批量归一化、AdamW、梯度提升树的正则化与采样。

**📊 数据集**

使用的实验数据集为以太坊主网交易记录（截至 2022‑05‑13），采集了 Organic、Sybil、MEV 三类标注地址，共 15,813 条交易序列，采用 80/20 的分层划分；在 Blind-Spot 过滤后保持 12,650 条训练和 3,163 条验证。

**📈 对比分析**

实验方法：统一 5 次随机种子下的训练/验证切分，使用 MCC、宏 F1 和准确率评价；XGBoost 在泄漏感知数据上取得 MCC 0.7535、Macro‑F1 0.8141、准确率 0.8745，显著优于 Transformer（MCC 0.6602）和 BiLSTM（MCC 0.6187）。同时，XGBoost 推理时延仅几百微秒，能耗比 Transformer 低 30‑倍，体现了更优的效率‑准确率折衷。

**⚠️ 局限性**

局限性：①仅考虑单链交易序列，未利用交互图结构；②语法离散化可能丢失细粒度信息；③缺乏对主动对抗或欺骗策略的鲁棒性评估；④未在真实在线流环境中验证实时性能；⑤未探索自监督预训练或连续嵌入提升表达能力。

---

## 47. VETO: Towards Protecting Images From Frontier AI Editing

**arXiv ID:** 2607.27292 | [PDF](https://arxiv.org/pdf/2607.27292v1)

**作者:** Jonas Grebe `[一作]` (TU Darmstadt and hessian.AI), Anna Rohrbach `[通讯]` (TU Darmstadt and hessian.AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Veto防护罩，对基于Diffusion Transformer的图像编辑模型中的参考图像与画布注意力进行熵最大化扰动，阻断信息流以抵御编辑。

**💡 创新点**

创新点在于：①在统一编辑框架下攻击内部多模态注意力，而非传统的编码器瓶颈；②设计了针对双流MMDiT块的熵最大化目标；③引入了兼顾闭框和开框编辑的首个评估基准。

**🔧 技术方法**

技术手段包括：基于PGD的对抗优化、熵最大化的注意力损失、使用Diffusion Transformer（MMDiT）架构、利用多模态大型语言模型（Gemini 3.5 Flash）做自动评判。

**📊 数据集**

使用的数据集包括：新构建的300样本评估基准（包含一般、诽谤、暴力三类，闭框与开框各50样本），以及EditBench、AnyEdit等公开编辑数据集。

**📈 对比分析**

与传统编码器级防护方法（Cloak、DiffVax）在三组数据集上进行对比。Veto在保持LPIPS≈0.2–0.3、PSNR≈30dB的前提下，将人类判定的编辑成功率从70%以下降至≈1%–20%，明显优于基线；在闭框和开框两种编辑模式下均表现出更强的防护效果。

**⚠️ 局限性**

局限性包括：对图像翻转、裁剪等常见变形的鲁棒性有限；在强JPEG压缩下保护效果下降；仅针对图像层面，尚未与模型级或推理层级的防护结合；评估主要基于合成图像，缺乏真实用户数据的验证。

---

## 48. Low-Latency Bootstrapping for CKKS using Roots of Unity

**arXiv ID:** 2607.27401 | [PDF](https://arxiv.org/pdf/2607.27401v1)

**作者:** Jean-Sebastien Coron `[一作]`, Robin Koestler `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种新的SPRBootstrapping算法，用于CKKS同态加密方案的低延迟重新加密；

**💡 创新点**

创新点在于将模q加法群直接嵌入到复数单位根圆群中，利用复数乘法天然实现模q归约，从而避免了原方法中对正弦函数的多项式逼近，显著降低乘法深度；

**🔧 技术方法**

采用了密钥位打包、轨迹（trace）与乘积（product）算子、RNS算术以及OpenFHE库的SIMD与FFT优化技术；

**📊 数据集**

实验使用OpenFHE自带的标准参数集进行时间基准，未使用专门的数据集，而是对不同槽数的加密时间进行了测评；

**📈 对比分析**

与原CKKS Bootstrapping（OpenFHE实现）比较，SPRBootstrapping在槽数为1到64时平均提升约5倍；槽数超过128时，原方案因O(log n)标量更快，SPRBootstrapping因O(n)增长而变得不再划算；

**⚠️ 局限性**

局限性包括：对槽数大于约128时效率急剧下降；需使用块稀疏二进制密钥，参数选择复杂；噪声估计仍为经验性，对高精度或深层电路可能不够稳健。

---

## 49. AI-assisted pre-review of open-source software submissions: an experience report from BOSC 2026

**arXiv ID:** 2607.27228 | [PDF](https://arxiv.org/pdf/2607.27228v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 50. Do Context Files Help Coding Agents? A Two-Agent Ablation Study on Real Repositories

**arXiv ID:** 2607.27250 | [PDF](https://arxiv.org/pdf/2607.27250v1)

**作者:** Prakhar Khatri `[一作]` `[通讯]` (Independent Researcher), Prakhar Khatri (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对两种前沿 AI 编码代理（Claude Code 与 Codex）在真实仓库的 17/15 个任务中，进行受控消融实验，比较三种持久上下文文件注入策略（无上下文、系统提示全文件、按需 Wiki 检索）对任务正确率与效率的影响。

**💡 创新点**

① 在两代理间做了跨代理受控实验；② 引入“代理特定边缘难度”概念解释以往研究的矛盾；③ 通过失效模式分析与操控有效性检验，揭示上下文文件并不能弥补实现技能不足。

**🔧 技术方法**

采用 SWE‑bench 样式的金标测试评估、等价性检验（TOST）、置换检验、蒙特卡洛功效分析；实现三种注入策略；使用 Claude Code 与 Codex 两大模型；通过工具调用、时延、输出 token 等效率指标对比。

**📊 数据集**

三家 Python 开源仓库（pdm、firebase‑admin‑python、opshin）中的 17/15 个 PR 合并任务，共 288 次可评估运行，任务均包含自定义测试文件。

**📈 对比分析**

对每个任务使用 3 次重复，计算平均正确率与效率指标，采用 permutation、Wilcoxon、TOST 等统计检验。结果显示：三种策略在正确率上无显著差异（≤10–15pp ；Claude 53–56%，Codex 52–59%）；Claude 在“Selective”策略下缓存占用显著下降，opshin 中全套测试运行次数下降；Codex 效率指标基本无变化。

**⚠️ 局限性**

限制：样本量仅 15/17 任务、仅 Python 仓库，难以推广至其他语言/更大项目；Claude 与 Codex 的注入通道不同，导致跨代理比较受限；“Selective”策略的 Wiki 内容与原 AGENTS.md 不匹配；缺乏自然发现条件（仅系统提示或按需检索）；未检验特定任务的目标上下文文件可能带来的收益；模型版本特定，随升级可能变动。

---

## 51. VideoCoCo: Code-as-CoT for Physically-Consistent Video Generation via an Agentic Dual-Engine System

**arXiv ID:** 2607.27380 | [PDF](https://arxiv.org/pdf/2607.27380v1)

**作者:** Haodong Li `[一作]` (Chinese University of Hong Kong), Pheng-Ann Heng `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 VideoCoCo 框架，利用可执行的 Blender 代码作为链式思考（CoT）来生成物理一致的视频；

**💡 创新点**

创新点在于将可执行代码作为过程级中间表示，先通过沙箱模拟生成确定性草稿，再用草稿指导视频编辑器实现高质量、物理一致的最终视频；

**🔧 技术方法**

技术包括：代码合成代理（生成 Blender Python 程序）、沙箱执行渲染草稿、指令生成代理、草稿条件视频编辑器（基于 OmniWeaving 并用 LoRA 微调），以及用于草稿条件编辑的教师模型 Seedance 2.0；

**📊 数据集**

数据集为 VideoCoCo-3K，包含 3000 条草稿–指令–目标视频三元组；此外使用 PhyGenBench 与 VBench‑2.0 进行评估；

**📈 对比分析**

与多种闭源和开源基线（如 OmniWeaving、Wan2.2‑TI2V‑5B、CogVideoX 等）对比，VideoCoCo 在 PhyGenBench 的平均一致性得分提升至 0.558（高于 0.475 的基线和 0.544 的最强开源模型），在 VBench‑2.0 上平均可行性提升至 77.88%（较基线 52.18% 大幅提升）；

**⚠️ 局限性**

局限性包括：推理时需执行 Blender 模拟，导致额外延迟；受限于 Blender 物理引擎的表达能力，难以零样本合成高度复杂的物理现象（如湍流流体）。

---

## 52. Same Facts, Different Diagnosis: Measuring and Mitigating Narrative Anchoring in Clinical Language Models

**arXiv ID:** 2607.27384 | [PDF](https://arxiv.org/pdf/2607.27384v1)

**作者:** Prabhjot Singh `[一作]` (University of Texas at Austin), Vijay Chennareddy `[通讯]` (Middlesex University)

**通讯引用:** 70 | [OpenAlex ID](https://openalex.org/A5051700244)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了临床语言模型在不同社会语言注册下的诊断偏差（Narrative Anchoring），并提出三代理管道NarrativeShield以消除该偏差。

**💡 创新点**

首次将社会语言注册作为独立偏差通道，引入可验证事实保持的多语境数据集，并通过结构化抽取实现无标签偏差消除。

**🔧 技术方法**

使用三代理管道：事实抽取器、确定性工具路由器、临床推理引擎，并对多个开源模型进行对比实验。

**📊 数据集**

基于MedQA-USMLE改造的1,000条临床情境，生成三种注册（控制、社会经济、文化）且经过独立事实审计的NarrativeShield‑SDoH数据集。

**📈 对比分析**

与直接提示、链式思维、显式去偏指令等对比，NarrativeShield将Narrative Anchoring Gap降至≈0，稳定性提升，准确率仅略有下降（≤6.9%）且在大多数模型上提升。

**⚠️ 局限性**

局限包括：仅测试3B‑12B开源模型，注册维度有限，单轮USMLE问题，缺乏真实多轮对话验证，去偏后准确率成本不一定可接受。

---

## 53. SDO: Structure-Aware Data Organization for Efficient LLM Post-Training

**arXiv ID:** 2607.27273 | [PDF](https://arxiv.org/pdf/2607.27273v1)

**作者:** Jinliang Gao `[一作]` (Institute of Automation Chinese Academy of Sciences), Pin Lyu `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种在后训练阶段使用冻结嵌入进行局部KNN批次划分、曝光平衡重构数据池的闭环数据组织框架（SDO），从而提升大语言模型的训练效率。

**💡 创新点**

创新点在于：①将数据曝光量纳入反馈机制，动态重构训练池，避免过度训练同一类样本；②在每个epoch内使用嵌入空间的局部邻域构建连贯批次，提升梯度一致性；③不需要warm‑up或永久过滤样本，保持覆盖率，具有plug‑and‑play属性。

**🔧 技术方法**

核心技术包括：冻结句子嵌入（如zembed‑1）、KNN邻域构建与遍历、曝光跟踪与阈值调度、随机逆曝光采样、梯度冲突分析与理论证明、近似最近邻索引（FAISS）加速。

**📊 数据集**

使用的公开数据集包括：GSM8K（用于GRPO评估）、UltraFeedback（包含约6,000个偏好对，用于DPO和SFT评估）等。

**📈 对比分析**

与统一基线（仅随机打乱样本）在三种后训练范式（GRPO、DPO、SFT）下进行对比，实验表明SDO在早中期显著加速收敛：GRPO准确率提升0.48%，DPO奖励边缘提升0.044，SFT损失下降0.001；在后期收敛时差距缩小，整体保持或略优于基线。

**⚠️ 局限性**

局限性包括：①对邻域大小K、曝光阈值增量Δτ和保留比例r等超参敏感；②依赖冻结嵌入的语义表达能力，若嵌入质量差会影响效果；③在极大数据集上需近似KNN，虽然成本低但可能影响邻域精度；④在最终收敛阶段加速作用有限，主要受益于早中期梯度一致性提升。

---

## 54. Explorative Modeling: Unlocking a Third Pretraining Axis and End-to-End Generation

**arXiv ID:** 2607.27372 | [PDF](https://arxiv.org/pdf/2607.27372v1)

**作者:** Alexi Gladstone `[一作]` (University of Illinois Urbana-Champaign), Yilun Du `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实验了 Explorative Modeling（探索式建模）——一种通过在训练循环中对生成候选样本做 K 倍探索并仅对最接近的数据点反向传播，从而直接提升生成模型的多模态表达能力与端到端可训练性的全新范式。

**💡 创新点**

创新点包括：① 将“探索”作为第三个可扩展轴，独立于参数规模和数据规模；② 通过训练循环的最佳- K 机制提升生成表达性；③ 证明探索可替代传统的生成过程分解，实现真正端到端的重建式生成；④ 在多模态任务上显示探索能够持续提升性能并显著提升 FLOP、样本与参数效率。

**🔧 技术方法**

使用技术：前向/后向 Explorative Modeling（Best‑of‑K 训练）、扩散/流匹配模型、Jumpy 生成模型、掩码扩散语言模型；对比 Diffusion Policy、Diffuser 等基准；在不同 K 值下进行实验并分析其对精度、覆盖度与效率的影响。

**📊 数据集**

主要数据集：ImageNet 256×256、Something‑Something V2 视频数据集、OpenAI 公开的语言模型数据；在 ImageNet 上进行 FID/FDr^6 评估，在视频任务中使用 FVD，语言任务中采用 perplexity‑entropy 前沿。

**📈 对比分析**

与现有方法比较：在 ImageNet 上无引导 FID 仅用 XRAE+探索实现 1.43 FID，接近最新水平；样本效率提升 6.2×、FLOP 效率提升 4.1×、参数效率提升 47%；在机器人行为克隆与世界建模任务中，Explorative Policy/World Model 分别以 1 次网络前向（vs. 100）和 4–256 倍 NFE 减少的前提下与 Diffusion Policy/Diffuser 竞争或超越。

**⚠️ 局限性**

局限性：① 采用 Forward XM 时需要 K 倍前向推理，导致高模态数据下计算量巨大；② 后向 XM 可能出现模式崩塌，需额外熵或覆盖约束；③ 对于极高多模态分布（如高分辨率图像）仍难以在单步推理内覆盖所有模式；④ 目前对自回归 LLM 的改进有限，需要更合适的潜变量搜索机制；⑤ 指导策略在探索训练中迁移不完全，进一步的探索专用引导设计仍是未解决的问题。

---

## 55. CircuitProver: Agentic Lean 4 Theorem Proving with Reusable Circuit Proof Library for Hardware Verification

**arXiv ID:** 2607.27259 | [PDF](https://arxiv.org/pdf/2607.27259v1)

**作者:** Ziyi Yang `[一作]` (Hong Kong University of Science and Technology), Hongce Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 553 | [OpenAlex ID](https://openalex.org/A5003614499)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 CircuitProver，一套基于 Lean4 的代理驱动硬件验证框架，实现参数化设计的自动形式化、交互式定理证明和可重用证明库构建。

**💡 创新点**

首次在硬件验证中实现证明知识的累积与复用：自动将 Chisel 设计及自然语言规范翻译为 Lean4 语义模型；通过证明轨迹提炼可重用证明策略与定理库；并提供第一套 agentic 硬件定理证明基准。

**🔧 技术方法**

采用 Lean4 交互式定理证明、LLM 驱动的证明代理（如 Claude）、Chisel-to-Lean 翻译器、硬件感知证明规划、可重用证明库抽取等技术。

**📊 数据集**

使用自建的 63 个参数化 Chisel 设计基准（算术、控制、内存、杂项，含 4 个处理器级乘除模块）。

**📈 对比分析**

与 Chicala/Stainless 及无库 vanilla agent 进行对比；在全部任务上实现 100% 通过；平均证明回合数下降 50%，验证时间下降 23.2%，证明 LOC 降低 16.3%；在处理器级模块上相对 vanilla 进一步减少 56–57% 的时间/LOC。

**⚠️ 局限性**

仅支持 Chisel 的子集；当前框架在大规模工业设计、复杂属性或其他 RTL 语言方面尚未验证；LLM 依赖度高，弱模型的覆盖率较低。

---

## 56. Multi-Head Attention Residuals

**arXiv ID:** 2607.27230 | [PDF](https://arxiv.org/pdf/2607.27230v1)

**作者:** Cheng Luo `[一作]` (Independent Researcher), Junjie Hu `[通讯]` (University of Wisconsin--Madison)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并从零训练到1B参数的多头注意残差（MHAR）机制，用于在Transformer层中以多头方式路由深度历史；并在8B模型的中期微调中实现了身份保持的转换。

**💡 创新点**

创新点在于将单一路由查询拆分成每个子空间的独立头，消除单查询对所有特征子空间的“强制妥协”，并且实现零参数、零额外计算的多头改造；同时提出融合的Triton路由核和Delta式身份保持转换。

**🔧 技术方法**

技术包括：多头注意残差路由、RMSNorm、softmax深度路由、Triton融合核、Delta注意残差身份保持、AdamW优化、FSDP、EMA、z-loss。

**📊 数据集**

主要数据集为FineWeb‑Edu用于从零训练（100M、350M、1B），以及自构造的≈1.9T英语混合语料（包含 synthetic、STEM、代码、Stack‑Edu、arXiv、维基等）用于8B模型的中期微调；还在WikiText‑2、LAMBADA、HellaSwag、MMLU、GPQA、GSM8K、MATH、HumanEval、MBPP等公开评测集上做下游评估。

**📈 对比分析**

与标准Transformer、Hyper‑Connections和单头注意残差在相同规模、相同调度下对比；MHAR在所有规模下均优于基线，单头路由在大模型下反而退化；验证损失下降幅度约为-0.05至-0.08，对等步数的计算等价提升约1.3–1.5×；在下游任务中提升了10–19%的 perplexity 和 LAMBADA 准确率；在8B中期微调中获得+3.2 GSM8K、+3.1 GPQA 的显著提升。

**⚠️ 局限性**

局限性包括：单头路由在大模型中性能不稳定，需根据KV头数设置多头；虽然融合核显著降低了系统开销，但在极大模型和极大批量下仍存在内存和网络流量瓶颈；对某些任务（如代码评估）的提升不显著，且对不同语言/领域语料的普适性尚待验证。

---

## 57. From Minds to Models: The Intersection of Psychology and LLM Behaviours

**arXiv ID:** 2607.27579 | [PDF](https://arxiv.org/pdf/2607.27579v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 58. BridgeAlign: Bridging Preference Alignment for Humanities and Social Sciences

**arXiv ID:** 2607.27366 | [PDF](https://arxiv.org/pdf/2607.27366v1)

**作者:** Ru Peng `[一作]` (Zhejiang University), Junbo Zhao `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

我们提出了BridgeAlign框架，结合种子文档挑选、指令逆向生成与偏好数据合成、以及基于评分的偏好优化，实现了针对人文与社会科学领域的自适配偏好对齐数据生成与训练。

**💡 创新点**

其创新点在于引入了BridgePO——通过评分导向的可控质量降解生成临界边界的“硬负样本”，以及RubricPO，首次系统性地利用人文质量衡量标准对合成数据进行排序，从而实现细粒度、人类化的质量对齐。

**🔧 技术方法**

技术手段包括：LLM驱动的指令逆向与Q&A一致性检验、基于12条专家制定的多维度人文质量rubric进行文档评分、DPO（Direct Preference Optimization）偏好优化、以及对高分样本进行可控降解生成桥接文档。

**📊 数据集**

使用的数据集为从627B-token SlimPajama网页语料中筛选、过滤并归类成14个人文与社会科学领域的30M文档，最终合成210k条高质量偏好对齐样本。

**📈 对比分析**

与10个主流基线（含人工写作、混合来源与合成数据）在17个多维度评测基准（情感感知、角色扮演、写作、社交互动、指令遵循、知识推理等）对比，BridgePO在所有9大能力指标上均位居榜首，平均得分最高，AlpacaEval 2赢率最高，且人类对比实验验证结果真实可靠。

**⚠️ 局限性**

局限性包括：仍依赖LLM及人工制定的评分标准，可能未能完全捕捉主观深层质量；对模型偏好、来源噪声等偏见的消除仍不充分；目前覆盖14个HSS领域，扩展到更广泛的学科与多语言环境仍需进一步研究。

---

## 59. Recursive transformers for semiconductor thermo-mechanical reliability

**arXiv ID:** 2607.27251 | [PDF](https://arxiv.org/pdf/2607.27251v1)

**作者:** Kart-leong Lim `[一作]` (Agency for Science, Technology and Research), Kart-leong Lim `[通讯]` (Agency for Science, Technology and Research)

**通讯引用:** 588 | [OpenAlex ID](https://openalex.org/A5087657164)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了递归权重共享 Transformer 在半导体热机械可靠性预测中的应用，提出了基于深度递归（DEPTH）的模型；

**💡 创新点**

创新点在于将递归深度作为状态输入，保持权重共享的同时实现更深层次的计算，显著提升参数与 FLOPs 的利用率；

**🔧 技术方法**

采用递归 Transformer 结构、权重共享、深度递归输入、以及标准 Transformer 的自注意力与前馈网络；

**📊 数据集**

使用了三个低维工程数据集：Stress10K、Warpage10K（来自 FEA 的 10,000 条样本）和 PINN 合成电容静电场数据（7,265 条样本）；

**📈 对比分析**

与七种基线模型（包括 Vanilla、Simple、Tiny Recursive 等）在 Recall@K、MRR、参数量和 FLOPs 上进行 Pareto 分析，DEPTH 模型在保持最少参数与 FLOPs 的同时取得最高或相近的检索精度；

**⚠️ 局限性**

局限性包括仅在小规模、短序列数据上验证，深度递归的可扩展性和对更复杂、长序列任务的适用性仍待进一步研究。

---

## 60. Divergence Decoding: Training-Free Capability Fusion

**arXiv ID:** 2607.27248 | [PDF](https://arxiv.org/pdf/2607.27248v1)

**作者:** Yimi Wang `[一作]` (Peking University), Li Yuan `[通讯]` (Peking University)

**通讯引用:** 17979 | [OpenAlex ID](https://openalex.org/A5100700791)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了无训练的 Divergence Decoding 框架，动态在通用模型和专用模型之间路由生成文本。

**💡 创新点**

创新点在于利用 Jensen‑Shannon 散度监控两模型在每个 token 上的分布差异，实时识别推理风险并切换控制，从而实现推理时的自适应协作。

**🔧 技术方法**

核心技术包括 Jensen‑Shannon 散度评估、草稿‑验证（draft‑and‑verify）结构重构以及基于置信度的动态路由机制。

**📊 数据集**

使用了科学领域基准数据集 GPQA、ChemBench 和 ChemCoTBench 进行评估。

**📈 对比分析**

与单一通用或专用模型对比，Divergence Decoding 在 Llama 与 Qwen 系列模型上均取得更高分数，明显优于大多数单模型基线。

**⚠️ 局限性**

局限性包括对散度阈值设置敏感、可能增加推理时间和算力开销，以及对非科学领域的通用性尚未充分验证。

---

## 61. Heterogeneous Ranking in Industrial-Scale Recommender Systems: A Case Study

**arXiv ID:** 2607.27577 | [PDF](https://arxiv.org/pdf/2607.27577v1)

**作者:** Di Bai `[一作]` (Google LLC), Luoshu Wang `[通讯]` (Google LLC)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在 Google Discover 的统一多任务排名系统中，提出并工业化部署了一种能够在高度异构内容下保持高质量排序的模型。

**💡 创新点**

创新点在于：① 设计 HA‑MoE（Heterogeneity‑Adaptive Multi‑Gated Mixture‑of‑Experts）架构，将显式异构上下文注入门控网络和专家表示，实现有效专业化；② 引入 LENS（Latent Expert Network Specialization）框架，用激活切片和 PIEM（Permutation‑Invariant Expert Matching）实现模型内部专业化的可观测与监控；③ 提出 DL‑AUC（Dual‑Level AUC）评价指标，兼顾全局排序性能与跨类型排名正确性。

**🔧 技术方法**

使用的技术包括：多任务学习、Mixture‑of‑Experts（MMoE）与自定义门控、线性调制（HDLM）、正负样本平衡、点式与对式损失混合、稠密专家网络、Adam 优化器、以及可解释可观测工具 LENS。

**📊 数据集**

数据集来源为 Google Discover 的内部日志：约 10M 条 7‑天 holdout 样本用于离线评估，1% 线上流量用于 7‑天 A/B 测试。

**📈 对比分析**

与共享 MLP 基线和标准 MMoE 进行对比。离线 DL‑AUC 上，HA‑MoE 在 pInterest 0.691、pDisinterest 0.949 上均优于基线；在线 A/B 测试提升了 DAU +0.22%、View Impressions +0.48%、Scroll Depth +0.34%、Diverse Feed Rate +0.36%、Diverse Engagement Rate +0.54%。

**⚠️ 局限性**

局限性包括：仍需更表达力的门控与稀疏路由技术、缺乏列表级优化和 LLM 辅助的排名决策、以及对用户级上下文与业务目标的异构建模尚不充分。

---

## 62. Automated Transcript Analysis for Detecting Flaws in Agentic Benchmarks

**arXiv ID:** 2607.27518 | [PDF](https://arxiv.org/pdf/2607.27518v1)

**作者:** Jeff Mohl `[一作]` (Independent), Justin Olive `[通讯]` (Generality Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建并验证了一套自动化转录扫描器，用来检测代理式基准中的四类有效性缺陷（真相获取、工具失败、猜测漏洞、答案格式歧义）。

**💡 创新点**

创新点在于将基准检查从人工审核转向可复现的、基于大模型的自动扫描方法，首次提出了可操作化的扫描器评估指标和采样策略，并开源了扫描器与对应的转录数据集。

**🔧 技术方法**

使用了Inspect Scout框架与大型语言模型（GPT‑5.4、Claude Sonnet 4.6）作为扫描器后端，结合人工标注的转录样本进行迭代提示工程和评估。

**📊 数据集**

主要使用了11个代理式评估基准的转录数据，包含CORE‑Bench、SWE‑Bench‑Verified、KernelBench、CVE‑Bench、Terminal‑bench‑2.0等，并通过合成违规样本扩充训练集。

**📈 对比分析**

通过与人工评分对比，扫描器在不同基准和模型上表现差异显著；在最优配置下，F1 最高可达0.74，平均QWK 约0.5‑0.7；扫描器在检测已知违规时效果良好，但对低频或语境依赖强的违规检测灵敏度有限。

**⚠️ 局限性**

主要局限包括：对转录信息的依赖导致无法捕获非转录层面的缺陷；扫描器对语境理解不足、提示设计不统一；样本稀缺导致敏感性估计不稳；对多轮对话、不同模型兼容性不足；人工标注本身存在主观性和漏标。

---

## 63. Benchmarking LLM Competence on Logical Inference over Probability Operators

**arXiv ID:** 2607.27405 | [PDF](https://arxiv.org/pdf/2607.27405v1)

**作者:** Nayera Hasan `[一作]` (Haverford College), Alvin Grissom `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个用于评估自然语言中概率模态推理的基准，包含14,320条结构化英语提示、15种推理模板和多种句式变体；

**💡 创新点**

创新点在于通过系统地扰动问题形式、否定方式和表面内容，区分模型的答案偏向与真正的命题推理能力，提出“competence floor”指标来衡量模型在两类答案（Yes/No）上的最低准确率；

**🔧 技术方法**

技术上采用大语言模型的零样本推理，解析首个生成的答案 token 进行 Yes/No 判定，使用多模型评估（29 个模型，跨不同规模与家族）；

**📊 数据集**

数据集为自生成的概率模态句子集合，覆盖 13 种逻辑推理模板（10 合法 3 非合法），并在每个模板下对问题表述、否定形式、姓名国籍、性别、活动场景等进行多种扰动；

**📈 对比分析**

评估方法：统计四个条件（有效/无效 × 肯定/否定）下的准确率、答案偏向（Bias）、极性敏感度（PS）以及最低准确率（Floor）。结果显示大多数模型存在强烈的 Yes/No 偏向，只有 9/29 模型在 Floor 上超过 0.5；相对性能差距在 0.1–0.3 之间；

**⚠️ 局限性**

局限性包括：仅对英语提示进行零样本评估；只解析首个答案 token，未考虑多轮或长文本推理；未测试其他语言或多模态场景；模型的答案生成策略（如 chain-of-thought）未被启用，可能影响结果。

---

## 64. IGME: Efficient Chained Method Ensemble for Transferable Semantic Segmentation Attacks

**arXiv ID:** 2607.27465 | [PDF](https://arxiv.org/pdf/2607.27465v1)

**作者:** Mengqi He `[一作]` (Australian National University), Jing Zhang `[通讯]` (Australian National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种单源可迁移语义分割攻击框架IGME，利用链式可微攻击组件共享源模型梯度并结合路径平均稳定性。

**💡 创新点**

提出链式攻击组件组合和集成梯度路径平均的稳定策略，显著降低单源攻击的计算成本并提升跨模型迁移效果。

**🔧 技术方法**

链式可微攻击组件、集成梯度路径平均、PGD、NI、DI、TI等攻防技术。

**📊 数据集**

在Pascal VOC 2012和Cityscapes数据集上进行实验，并评估DeepLabV3、Mask2Former等模型。

**📈 对比分析**

与FGSM、PGD、SegPGD、DAG、NI、DI、TI、IAA等单源攻击以及ENS、SVRE等模型集成攻击比较，IGME在保持或提升迁移效果的同时，计算时间显著低于模型集成。

**⚠️ 局限性**

迁移性能对源-目标模型对不均衡，路径平均样本数需调节，且在更大Transformer基础模型上的验证尚不充分。

---

## 65. Bridging openEHR and OMOP: Expanded Mappings and Systematic Analysis of Semantic and Structural Limitations in the OMOP CDM

**arXiv ID:** 2607.27208 | [PDF](https://arxiv.org/pdf/2607.27208v1)

**作者:** Severin Kohler `[一作]` (Berlin Institute of Health at Charité – Universitätsmedizin Berlin), Roland Eils `[通讯]` (Berlin Institute of Health at Charité – Universitätsmedizin Berlin)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了Eos引擎和OMOCL语言的新一代版本，显著扩展了openEHR到OMOP的映射库，支持内部值集映射、AQL驱动的访问生成，并提升了语义完整性。

**💡 创新点**

创新点在于引入conceptMap机制实现openEHR内部术语与OMOP标准概念的映射、使用AQL灵活生成visit_occurrence，以及系统化扩充近200个archetype映射库，解决了之前对内部值集、访问生成与映射覆盖不足的问题。

**🔧 技术方法**

技术上采用了DSL OMOCL、ETL引擎Eos、Archetype Query Language (AQL)、openEHR Archetype & ConceptMap、OMOP CDM结构与术语集（SNOMED CT、LOINC等）以及GitHub开源共享机制。

**📊 数据集**

使用了包含196个openEHR archetype的测试数据集，涵盖了国际CKM中的所有稳定archetype，并通过对这些archetype映射到OMOP表的评估来验证框架。

**📈 对比分析**

通过对映射覆盖率、术语完整性、域分布以及结构限制的定性与定量分析进行评估，发现概念标识符映射缺口为8.65%，并证明新框架在减少信息损失、提高语义保真度方面优于旧版，但仍存在表结构碎片化导致的查询复杂性与性能瓶颈。

**⚠️ 局限性**

主要局限在于OMOP CDM的通用表结构与术语驱动模型导致语义碎片化、对内部值集支持不足、缺乏专门字段表达临床细节、关系概念缺失及多表联接带来的分析复杂性，影响下游研究的可重复性与准确性。

---

## 66. RLPF: Reinforcement Learning from Performance Feedback for Code Generation

**arXiv ID:** 2607.27271 | [PDF](https://arxiv.org/pdf/2607.27271v1)

**作者:** Huihao Jing `[一作]` (Hong Kong University of Science and Technology), Yangqiu Song `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 10889 | [OpenAlex ID](https://openalex.org/A5020880385)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究如何通过强化学习让代码生成模型在满足功能正确性的基础上，倾向生成更快的实现。

**💡 创新点**

提出分阶段奖励机制：对失败程序按执行进度给予奖励，对成功程序按相对效率（基准与专家实现差距）进行排名，克服单一运行时奖励的稀疏与尺度不一问题。

**🔧 技术方法**

使用 GRPO 强化学习框架、LoRA 微调、Qwen3-32B 语言模型，并构造阶段化奖励（执行进度奖励 + CGRE 等效率奖励）来训练模型。

**📊 数据集**

主要在 PerfCodeBench 数据集上训练和评估（1,413 训练任务，306 家族分离测试任务），并在 EffiBench-X 进行跨域测试；模型生成的性能参考使用 GPT‑5.4 输出。

**📈 对比分析**

与基线模型、RLVR 仅正确奖励、仅运行时奖励以及多种强大开源模型（GPT‑5.4、Claude Opus 4.5、Gemini 3.1 Pro 等）进行比较。RLPF 将正确可执行率（CRR）从 11.1% 提升至 54.6%，相对效率指标 CGRE 从 8.1% 提升至 38.6%；在 EffiBench‑X 上实现约 3.9% 的执行时间提升，胜率 57.9%。

**⚠️ 局限性**

依赖可执行反馈与基准‑专家间显著性能差距；当参考实现不完善或缺失时奖励效果减弱；模型仍难以达到专家级实现（需算法、内存布局、并行化等更深层次优化），跨域迁移效果有限。

---

## 67. Position, Not Provenance: Separating Reasoning Mediation from Sycophancy in Medical Vision-Language Models

**arXiv ID:** 2607.27304 | [PDF](https://arxiv.org/pdf/2607.27304v1)

**作者:** Supratik Bhowal `[一作]` (IEM Kolkata), Anik Pal Chowdhury `[通讯]` (Heritage Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发并评估了CoT-Mediate框架，用于检验医学视觉语言模型在单一临床属性上对Chain-of-Thought（CoT）推理的可信度。

**💡 创新点**

创新点在于将单一临床属性的CoT编辑作为因果中介，区分推理驱动与顺从；采用双臂注入（重提示与前缀强制）以及来源标签（自我、证据、资深放射科医师、医学生）来控制出处影响。

**🔧 技术方法**

使用双臂注入方法、JSON结构化扰动、LLM（GPT‑4o‑mini）生成对抗编辑与隐式答案，评估指标包括Mediation Faithfulness Score（MFS）、Decoupling Rate（DR）等。

**📊 数据集**

基于VQA‑RAD数据集，抽取了1,000条包含图像与问答的医学实例进行实验。

**📈 对比分析**

通过与LLaVA‑Med和MedGemma两种模型对比，发现前缀强制注入显著提高MFS（分别为73.5%和61.6%），并通过来源标签揭示不同的顺从模式，验证了双臂注入的有效性。

**⚠️ 局限性**

局限性包括：仅评估单属性编辑，未涉及多属性交互；仅针对VQA‑RAD；LLM生成的编辑与答案判定可能引入偏差；未考察更复杂的视觉输入扰动和多模态输入对结果的影响。

---

## 68. Reviewer Scores Are Not Comparable Across Research Areas in ML Peer Review

**arXiv ID:** 2607.27209 | [PDF](https://arxiv.org/pdf/2607.27209v1)

**作者:** Binyan Xu `[一作]` (Chinese University of Hong Kong), Kehuan Zhang `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 3670 | [OpenAlex ID](https://openalex.org/A5008237643)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用ICLR 2021‑2026完整评审数据，对219个BERTopic生成的研究主题进行统计，揭示同一评分水平下不同主题的录用概率差异最大可达8倍，证明评分与录用结果存在系统性脱钩。

**💡 创新点**

创新点在于首次将主题层面（而非仅论文层面）与评分结果关联，系统排除评分文化、专家水平、领域先验等假设，并以无模型的“相同评分带”方法直接展示跨主题的录用差距；提出三项可落地的对策：公开主题分层录用率、制定领域专属评审准则、引入评分校准信号。

**🔧 技术方法**

技术方法包括：BERTopic文本主题建模、描述性森林图、嵌套逻辑回归、阈值估计、相同评分带实验、相关性与回归检验、稳健性检验（cluster-robust、子分数阈值等），以及对评分分布进行归一化的校准分析。

**📊 数据集**

数据集为ICLR 2021‑2026的全评审记录，包含50,289篇论文、219个主题、每篇论文的10分制平均评分、1‑5分制贡献/严谨度子分数、作者自评信息等。

**📈 对比分析**

与传统方法相比，本文不需要构建任何评分模型，直接通过实证检验展示主题对录用率的显著影响；实验表明即使在相同评分带下，主题间录用差距可达8倍，且在不同年份、不同阈值、不同子分数下均保持显著性。

**⚠️ 局限性**

局限性包括：仅针对ICLR数据，缺乏对NeurIPS、ICML等会议的直接验证；分析为观察性，无法确定因果机制；依赖BERTopic主题划分可能存在分层误差；未评估不同主题内部多标签对结果的影响。

---

## 69. ECG-InterpBench: Benchmarking the Interpretability of ECG Foundation Models with Matched-Scale Sparse Autoencoders

**arXiv ID:** 2607.27404 | [PDF](https://arxiv.org/pdf/2607.27404v1)

**作者:** Yixuan Duan `[一作]` (Rice University), Wei Qiu `[通讯]` (Rice University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 ECG‑InterpBench benchmark，利用容量匹配的稀疏自编码器系统性评估 ECG 基础模型在不同层深、字典宽度和随机种子下的可解释性。

**💡 创新点**

创新点在于构建了 450 细胞的多尺度、可复现评估框架，统一了稀疏重构、单特征临床可达性和跨种子可复现性指标，揭示各模型在可解释性方面的差异。

**🔧 技术方法**

采用 BatchTopK 稀疏自编码器、R² 重构度量、单特征相关性、解码器相似度、CKA、归一化 AUC 以及 Kendall 相关等技术对模型表示进行量化。

**📊 数据集**

主要使用 PTB‑XL 12‑轴 ECG 数据集（21,799 条）进行基准评估，并在 MIMIC‑IV‑ECG（100k 条）上进行外部验证。

**📈 对比分析**

通过 6 个模型 × 5 层深 × 5 字典宽度 × 3 种子共 450 细胞的匹配比较，评估重构 R²、单特征相关性、概念覆盖率和跨种子相似度；结果显示不同模型在重构、临床可达性、覆盖率等维度各有领先者，且外部验证保持相同排序。

**⚠️ 局限性**

局限性包括仅覆盖六种代表性 ECG 基础模型、概念库仅限波形测量、字典宽度与稀疏度范围有限，且未评估模型在诊断性能或临床实际应用中的效果。

---

## 70. ProgFormer: Hierarchical Voxel Diffusion Transformer for Longitudinal Brain MRI Prediction

**arXiv ID:** 2607.27537 | [PDF](https://arxiv.org/pdf/2607.27537v1)

**作者:** Dexuan Ding `[一作]` (Macquarie University), Ming-Hsuan Yang `[通讯]` (University Of California Merced)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出 ProgFormer，一种分层体素空间扩散 Transformer，用于预测脑部结构 MRI 的未来扫描，能够同时保持整体结构一致性与局部病理变化。

**💡 创新点**

创新点在于将粗粒度补丁级别的空间-时间注意力与细粒度体素级别的交叉注意力结合，直接在体素空间进行条件流匹配，避免了传统的编码-解码失真，并通过层次化架构实现对全局与局部信息的分离处理。

**🔧 技术方法**

使用的技术包括条件流匹配扩散模型、Transformer 的多头自注意力 (MHSA)、粗细路径交叉注意力 (MHCA)、AdaLN 适配层归一化、Euler 步进推理以及基于解剖区域权重的加权损失。

**📊 数据集**

数据集涵盖三大阿尔茨海默病纵向 MRI 组：ADNI、AIBL 和 OASIS，全部采用 T1 加权扫描并统一预处理。

**📈 对比分析**

与 BrLP、CounterSynth、TADM-3D、SADM 等前沿方法比较，ProgFormer 在 pairwise 与 trajectory 评估中均取得最高的 PSNR/SSIM 与最低的区域 MAE，尤其在三组数据的多间隔预测任务上表现突出。

**⚠️ 局限性**

局限性包括对显存需求较高、仅验证于 T1‑weighted MRI、对极长时间间隔或非阿尔茨海默病病理的泛化性尚未充分验证。

---

## 71. VAmoS Bench: Voice Agent Simulation Bench

**arXiv ID:** 2607.27453 | [PDF](https://arxiv.org/pdf/2607.27453v1)

**作者:** Joshua Meyer `[一作]` (Veris AI), Andi Partovi `[通讯]` (Veris AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出并实现了VAmoS Bench——一种通过真实语音电话和状态化数据库后端来评估端到端语音代理完成客户支持任务的基准；

**💡 创新点**

创新点在于将会话文本与工具调用及其结果联合评判，能够同时检测语音代理是否正确完成任务、是否遵循流程、是否泄露隐私；并提供多种部署模式（自托管、托管平台、API、原生语音端点）的统一测试框架；

**🔧 技术方法**

技术包括：基于WebSocket的实时PCM音频桥接、工具调用适配器（WebSocket、Webhook、WebRTC、Twilio）、PostgreSQL隔离实例、LLM判别器评判、以及完整的模拟呼叫者和记录系统；

**📊 数据集**

数据集为100个银行信用卡操作场景，包含数据库种子、呼叫者身份、私有目标、发音特征和二元断言；

**📈 对比分析**

比较方法是让每个代理在同一套场景下完成3000+通话，记录完成率、连接率、延迟、成本等指标；结果显示完成率在43%到71%之间，复杂情境最难，Pipecat与LiveKit在相同模型堆栈下差距仅0.7个百分点，而将Pipecat替换为全NVIDIA堆栈导致28个百分点下降；

**⚠️ 局限性**

局限性包括：仅覆盖单一金融服务任务，未包含多语言/口音多样性，评判器单一LLM且无人工黄金标准，场景集固定导致可推广性有限，成本估算基于即时价格，未覆盖所有部署成本与网络扰动影响；

---

## 72. HSS-Synth: Humanities and Social Sciences Data Synthesis for LLMs

**arXiv ID:** 2607.27379 | [PDF](https://arxiv.org/pdf/2607.27379v1)

**作者:** Ru Peng `[一作]` (Zhejiang University), Junbo Zhao `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了HSS-Synth，一种面向人文与社会科学领域的数据合成管道，生成了超过23万条高质量、多样化的指令-回答对。

**💡 创新点**

创新点在于：①基于官方学科分类构建14个HSS领域体系；②引入多属性指令回译（明确任务要点和人物身份）并加入Q&A一致性校验；③提出教师强制回答技术，将种子文本作为语义锚点，显著提升答案真实性、完整性与人文风格。

**🔧 技术方法**

核心技术包括：多步骤种子文本筛选（来源采样、启发式过滤、域分类、质量打分与LLM文本精炼）、多属性指令回译与Q&A对齐检查、教师强制回答、以及对大型LLM（如Qwen3-30B-A3B、Qwen3-8B-Base）进行指令微调。

**📊 数据集**

使用的原始数据来自于大规模网页语料（Slimpajama 627B-token）及多来源混合文档，随后通过上述流程生成HSS合成数据。

**📈 对比分析**

实验与14个主流基线（人手制作、混合、半自动和全自动合成）在16个多能力基准（写作、情感、社会交互、指令遵循、知识、常识、长文本、阅读理解）上对比，HSS-Synth在大多数任务上均实现SOTA，Qwen3-8B-Base在人类偏好与知识能力上均有显著提升且无“性能起伏”问题。

**⚠️ 局限性**

局限性包括：依赖大模型进行合成，可能带来模型偏差；仅覆盖14个主流HSS子领域，缺乏更细粒度学科；在常识、长上下文与阅读理解等任务上提升有限，提示预训练数据合成的进一步探索。

---

## 73. Simulation of Surgical Suturing Using Position-Based Dynamics and the Material Point Method for Robot Reinforcement Learning

**arXiv ID:** 2607.27494 | [PDF](https://arxiv.org/pdf/2607.27494v1)

**作者:** Tleukhan Mussin `[一作]` (University of Alberta), Mahdi Tavakoli `[通讯]` (University of Alberta)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构建了一个结合PBD缝合线与MPM软组织的高性能手术缝合仿真环境，并在GPU上实现了并行仿真与强化学习训练。

**💡 创新点**

首次提出PBD缝合线与MPM软组织之间的两向接触耦合方法，兼顾物理真实性与计算效率，并为并行强化学习提供了可用的仿真平台。

**🔧 技术方法**

使用Position‑Based Dynamics（PBD）模拟缝合线、Material Point Method（MPM）模拟软组织、CUDA多流并行计算、Unity ML‑Agents（PPO/SAC）强化学习以及PBD–MPM接触耦合技术。

**📊 数据集**

未使用公开数据集，全部使用内部随机生成的仿真场景进行实验；训练数据由多实例并行仿真生成，未涉及医学图像或真实手术数据。

**📈 对比分析**

通过在10个并行实例上对比单流与多流CUDA实现，MPM仿真时间从5.85 ms下降至2.10 ms；强化学习实验显示PPO/SAC均能稳定收敛，针尖插入成功率在0.075 UU阈值下为80%，提取成功率为68%；阈值增大后成功率提升至91%和85%。

**⚠️ 局限性**

仅实现了针头插入、驱动、提取三步，未覆盖重新抓取、手柄交互和结扎；PBD缝合线缺乏自碰撞处理；仿真仅在单平面；缺乏对真实手术环境的进一步验证和泛化能力。

---

## 74. OVEarth-Bench: Evaluating Category Breadth and Query Diversity for Open-Vocabulary Earth Observation

**arXiv ID:** 2607.27278 | [PDF](https://arxiv.org/pdf/2607.27278v1)

**作者:** Kaiyu Li `[一作]` (Xi'an Jiaotong University), Xiangyong Cao `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 2993 | [OpenAlex ID](https://openalex.org/A5028103486)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了面向开源地球观测零射评测基准 OVEarth-Bench，覆盖172类、1346个自然语言词汇，提供 mask、HBB、OBB 标注，并支持词汇、指代与推理三种查询形式的零射定位任务。

**💡 创新点**

创新点在于：① 构建了大规模层级分类体系，显著扩大语义覆盖；② 同时评估词汇、指代和推理三类查询，提升评测多样性；③ 采用 LLM 自动生成并人工复核查询，保证查询质量；④ 统一零射零样本评测协议，兼容分割与检测框架。

**🔧 技术方法**

使用技术包括：LLM（如 ChatGPT）生成查询、SAM2.1 等分割工具、CLIP/SLIP-RS、MLLM（Qwen、LISA、UniPixel、Sa2VA 等）以及多种 EO 专用模型；评估指标采用宏/微 IoU、Precision/Recall、MCC 等。

**📊 数据集**

使用自采集的 590 张 EO 图像（520 张有效），覆盖六大洲，GSD 0.08–305.7 m/pixel，人工标注 425 正、1029 负词汇，732 指代表达，1056 推理查询，提供 mask、HBB、OBB 注释。

**📈 对比分析**

对 49 个零射模型（一般与 EO 专用模型）进行分割、检测与定位评测，报告词汇、指代、推理三类任务的 ma‑IoU、mi‑IoU、mcc 等；MLLM 模型在三类任务中占据大多数顶级位置；EO 专用模型虽有提升，但总体低于顶级一般模型；整体最佳 ma‑IoU 仅 38.75%。

**⚠️ 局限性**

局限性包括：定位准确率仍偏低，特别是对小目标的精度不足；负面查询的拒绝能力弱，MCC 通常低于 0.35；数据规模有限，可能与预训练数据重叠；评测仅覆盖 mask、HBB/OBB，未涵盖更复杂的多类别语义分割等。

---

## 75. FAVA: Formal Authorization for Verified Agents with Evidence-Backed Permission Graphs

**arXiv ID:** 2607.27267 | [PDF](https://arxiv.org/pdf/2607.27267v1)

**作者:** Yifan Zhang `[一作]` (Zhejiang University), Chang Liu `[通讯]` (Zhejiang University)

**通讯引用:** 88480 | [OpenAlex ID](https://openalex.org/A5100410352)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种将LLM代理权限管理转化为结构化Permission IR、再降至权限图并通过SMT求解器进行运行时授权的框架。

**💡 创新点**

创新点在于把自然语言指令分离为可验证的权限图，并利用SMT进行全局数据流与权限一致性检查，实现动态、可修复且严格的授权决策。

**🔧 技术方法**

使用LLM进行语义抽取、确定性IR降级、SMT求解器（Z3）进行授权、运行时网关做即时拦截和回溯。

**📊 数据集**

评测数据集包括OpenAgentSafety、OctoBench和ActPlane（公共与跟踪版本）。

**📈 对比分析**

与Prompt‑only、Regex、AgentSpec、ActPlane、AuthGraph、SafeAgent等基线相比，整体决策合规率达90.5%，在公共和跟踪数据集上实现100%合规率，运行时延迟平均<1 ms。

**⚠️ 局限性**

主要局限是对LLM抽取的依赖：标签缺失或误判导致误授权/误拦截；当前模型对复杂上下文和精细接收方解析仍有限；以及对低级系统细节（并发、TOCTOU）的抽象处理。

---

## 76. Neural Network-Assisted CLEAN for Channel Modeling in Low-SNR Regimes

**arXiv ID:** 2607.27450 | [PDF](https://arxiv.org/pdf/2607.27450v1)

**作者:** Chaofan Deng `[一作]` (Georgia Institute of Technology), Arijit Raychowdhury `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出NN-CLEAN框架，将多头残差神经网络嵌入传统CLEAN迭代提取循环，实现低SNR下多径参数的高效估计。

**💡 创新点**

创新点在于用网络代替昂贵的多维网格搜索，同时保持物理残差减法的精确性，提升实时性与对离网格、多径数量不确定性的鲁棒性。

**🔧 技术方法**

使用技术包括多头残差网络、最大化似然估计、Gaussian Label Smoothing、三阶段训练（主导路径、教师强制、自动回归）以及GPU并行推理。

**📊 数据集**

数据集为仿真生成的4×4 MIMO系统，512子载波，AoA/AoD 1°步长，距离0.05 m步长，SNR范围[-5,10] dB，训练与测试均采用离网格采样。

**📈 对比分析**

与GS‑CLEAN、3D‑MUSIC、3D‑ESPRIT及单射NN比较，在SNR5 dB时，NN‑CLEAN在1–2路径稀疏场景下准确率>96%，与GS‑CLEAN相当；在大批量下运行时间与内存近乎平坦，显著优于GS‑CLEAN；精度略低于GS‑CLEAN但差距极小。

**⚠️ 局限性**

局限性：输出离散网格导致微小精度损失；在高多径密度（≥3）时误差传播可能略高于GS‑CLEAN；依赖离线训练与GPU资源；极低SNR（<‑5 dB）场景未充分验证。

---

## 77. SWE-NFI: Studying and Benchmarking Coding Agents for Non-Functional Improvements

**arXiv ID:** 2607.27409 | [PDF](https://arxiv.org/pdf/2607.27409v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 78. Using Large Language Models for Idea Generation in Innovation

**arXiv ID:** 2607.27553 | [PDF](https://arxiv.org/pdf/2607.27553v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 79. TrustChain-Review: A Risk-Adaptive Blockchain and Game-Theoretic Framework for Trustworthy AI-Assisted Code Review

**arXiv ID:** 2607.27310 | [PDF](https://arxiv.org/pdf/2607.27310v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 80. The Capacity Region of the Broadcast Channel with Non-Signaling Assistance

**arXiv ID:** 2607.27434 | [PDF](https://arxiv.org/pdf/2607.27434v1)

**作者:** Yuhang Yao `[一作]` (University of California Irvine), Syed A. Jafar `[通讯]` (University of California Irvine)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究并完全表征了 K 用户无记忆广播信道在允许使用非信号(NS)资源时的容量区域，证明 NS‑辅助容量等价于 Sato 区域。

**💡 创新点**

创新点在于利用“身份验证方案”构造可达性证明，并通过关键引理将多用户典型性事件概率与信息量关联，从而解决了长期未解的多用户 NS‑辅助广播信道容量问题。

**🔧 技术方法**

主要技术包括：信息理论中的典型序列与相互信息极限、NS‑盒子（非信号盒）构造、twirling 对称化、以及对典型性事件概率的多用户上界估计。

**📊 数据集**

该工作为理论分析，未使用具体数据集；结果完全基于数学证明。

**📈 对比分析**

通过与经典广播信道容量、双分体 NS‑辅助容量（KS+ 区域）以及特殊通道（如二进制倾斜对称通道）进行对比，展示了 NS‑辅助容量可以严格大于经典容量且优于 KS+ 区域，证明了包含关系的严格性。

**⚠️ 局限性**

局限性：仅适用于离散无记忆广播信道；未对连续或带记忆信道给出结论；对实际系统的实现细节和对非理想资源的鲁棒性未作深入讨论。

---

## 81. Compression-Based Behavioral Similarity for Open-World Sybil Discovery on Ethereum

**arXiv ID:** 2607.27370 | [PDF](https://arxiv.org/pdf/2607.27370v1)

**作者:** Michał Bartnicki `[一作]` (Warsaw University of Technology), Jarosław A. Chudziak `[通讯]` (Warsaw University of Technology)

**通讯引用:** 133 | [OpenAlex ID](https://openalex.org/A5008057050)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `fede83ac-7505-405f-ab37-e7284695c47f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于 gzip 压缩的归一化压缩距离（NCD）和 EVM 执行轨迹构造的交易语法，用来在无财务链接的情况下检测并扩展以太坊上的 Sybil 群体。

**💡 创新点**

核心创新在于：①将交易序列抽象为符号化语法并利用无监督的 NCD 构建隐式行为图；②设计 Blind‑Spot 协议剔除高信号合约泄漏，保证检测依据真正的行为模式；③证明 NCD 在无监督候选发现、时序迁移和抗伪装性上优于传统监督模型。

**🔧 技术方法**

技术手段包括：EVM 执行追踪 → 交易语法编码（节奏、结构、意图）→ gzip‑NCD 相似度计算 → 隐式行为图构造 → MinHash/LSH 近似候选生成；Blind‑Spot 协议用于去除高信号合约影响。

**📊 数据集**

使用以太坊主网 Hop Protocol 空投与 Dune 分析的 MEV Bot 数据集，共 14,604 个钱包（900 个用于实验），按 Organic、MEV Bot、Sybil 三类进行标注，并通过 BigQuery 提取交易与内部执行轨迹。

**📈 对比分析**

与 XGBoost、TF‑IDF+LR、BiLSTM 等监督基线对比：在 1‑NN 任务中，NCD 的准确率为 0.696±0.03，Top‑10 邻域纯度 0.754、召回率 0.922，接近 BiLSTM；在 Blind‑Spot 过滤后性能几乎不变；NCD 对时序漂移和合成伪装具有更高鲁棒性（α=50% 时召回 0.981）。

**⚠️ 局限性**

局限性：①隐式图需 O(N²) 计算，规模扩展受限；②Blind‑Spot 仍依赖标签信息，部署时需自行设计；③跨活动/跨链迁移验证不足；④未在 Layer‑2 或未来版本的以太坊上进行实测。

---

## 82. RadHarmony: Radiological Data Handling in the Era of Agentic AI

**arXiv ID:** 2607.27235 | [PDF](https://arxiv.org/pdf/2607.27235v1)

**作者:** Frank Li `[一作]` (Emory University), Judy Gichoya `[通讯]` (Emory University)

**通讯引用:** 7240 | [OpenAlex ID](https://openalex.org/A5076075666)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `da1b1a89-583a-4b57-9c81-478778569bec` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发了 RadHarmony，一个统一 API，用于跨源、跨模态放射影像数据的和谐化、加载、增强，并用其预训练 Vision Transformer 并在下游任务中评估。

**💡 创新点**

创新点在于把数据统一化与 AI 助手驱动的数据集集成流程结合，实现了零代码、多数据集多任务无缝切换的框架。

**🔧 技术方法**

采用了 MONAI 的缓存与变换流水线、AI 代理自动生成代码、Vision Transformer 结合 LeJEPA 自监督目标、以及可扩展的三层架构（harmonizer、dataset、transform）。

**📊 数据集**

使用了 24 个公开放射影像数据集，核心以胸部 X 光为主（CheXpert、MIMIC‑CXR、ChestX‑ray14 等），并初步支持 CT 与 MRI 数据。

**📈 对比分析**

通过 5 折线性探针在 VinDr‑CXR 上比较单一 CheXpert 预训练与多数据集预训练，宏观 AUROC/ AUPRC 差异不显著；提高输入分辨率至 512 分辨率可略微提升性能。

**⚠️ 局限性**

局限性包括：目前仅在胸部影像上充分验证；CT/MRI 仍处于 beta 试验阶段；AI 助手集成需要人工审核；未覆盖更多模态或更大规模数据集。

---

## 83. From Backlog Items to Security Guidance: Towards Continuous Security Compliance

**arXiv ID:** 2607.27374 | [PDF](https://arxiv.org/pdf/2607.27374v1)

**作者:** Ignacio García Núñez `[一作]` (Technical University of Munich), Fabiola Moyón Constante `[通讯]` (Siemens)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于 NLP 的工作单增强系统，自动识别安全相关工作单并关联相应安全需求。

**💡 创新点**

创新点在于发布高质量专家标注的安全相关工作单数据集，提出回归式安全相关性检测器与四阶段检索增强生成管道，并首次在工业环境中验证其可行性。

**🔧 技术方法**

使用句子编码器（MPNet、MiniLM 等）、逻辑回归、交叉编码器以及开源大模型（Llama 30.5B）实现检测与检索。

**📊 数据集**

数据集包括 288 条专家标注的工作单（C1），以及内部安全政策与 CIS Benchmark 文档做检索语料。

**📈 对比分析**

在 C1 上，MPNet+LogReg 实现 F2=0.774；在五个 Wu 等基准上，零样本 G‑measure≈0.65，匹配或优于现有经典 ML 与 GPT 基线；检索成功率为 12/24 条被评为 ≥4/5。

**⚠️ 局限性**

局限性包括：数据规模有限，安全相关性判断仍具主观性，检索结果受文档粒度匹配影响，系统未完成与现有 Issue Tracker 的完整集成，且仅做了初步评估。

---

## 84. Bunraku: Turning a Single Illustration into an Editable Live2D Character

**arXiv ID:** 2607.27348 | [PDF](https://arxiv.org/pdf/2607.27348v1)

**作者:** Junhao Chen `[一作]` (Tsinghua University), Ruqi Huang `[通讯]` (Tsinghua University)

**通讯引用:** 318 | [OpenAlex ID](https://openalex.org/A5086379651)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作实现了一套端到端系统，能够仅凭一张角色插画自动生成完整可编辑、可驱动的 Live2D 角色模型，包括 RGBA 层、二维网格与参数驱动的关键姿态位移。

**💡 创新点**

创新点包括：
1) 用 Live2D 相关的层级分类训练的层级扩散模型一次性输出完整的 RGBA 堆栈并完成隐藏区域补全；
2) 采用基于 alpha 通道的内容一致三角网格生成，避免了传统网格预设的局限；
3) 以每个顶点为 token 的全 Transformer，跨层自注意力一次性预测所有层的关键姿态位移，实现层间协同而非独立运动；
4) 将位移分解为方向与对数幅度，提升训练稳定性并抑制大幅度运动的压缩；
5) 首次公开 8,884 模型的 Live2D 语料库和 Live2D‑Bench 基准，提供统一评测。

**🔧 技术方法**

核心技术包括：
- Live2D‑aware layered diffusion 模型（Qwen‑Image‑Layered 微调），
- DINOv2‑small 视觉编码器提取层级全局特征，
- 基于 alpha 的内容一致网格构建与 3 像素膨胀裁剪，
- 256 维输入嵌入 + 4‑head Transformer（5.1 M 参数）处理全字符顶点序列，
- 位移方向/幅度两头 MLP，L1 损失与幅度权重。

**📊 数据集**

使用了公开的 Live2D 资产 8,884 个模型，经过去重后得到 50k 层级分解样本、35k 动画样本；构建了 120 个 Live2D‑Bench 样例（100 人类/人形、20 非人形），并对 46 个留出角色进行无重叠训练评测。

**📈 对比分析**

与现有层级分解基线（Marigold‑depth、SAM、See‑through）相比，Stage‑1 的遮挡补全和层次排序优先，LPIPS、α‑IoU、order 等指标均提升 25% 以上。Stage‑2 与单层预测（每层独立）对比，联合预测的方向余弦从 0.693 提升至 0.7676（平均），中位数 0.8278，且 62% 的角色在 0.80 以上；与经典自由形变（FFD）基准相比，仍有约 1% 的方向误差。整体系统在 50 只留出角色上平均 0.7676 的方向余弦、1.23 的幅度比例，表明可视化动画逼真且连贯。

**⚠️ 局限性**

主要限制：
1) 运动幅度压缩——大幅度转向等运动被低估，原因是单点回归在多模任务中只能学习中位数；
2) 无法动态调整绘制顺序，无法处理绘制层随参数变化的情况；
3) 依赖 Stage‑1 质量，层级划分错误会直接导致动画失真；
4) 参数词表长尾，新增 24 参数时部分稀缺参数表现差；
5) 位移方向余弦受网格密度影响，需统一网格标准或改为面积加权指标；
6) 眼睛开合等细节层在自动分层时难以完整，导致对应参数失效。

---

## 85. Digital Harf: A Clinically Integrated Multimodal AI System for Pervasive Arabic Speech and Language Therapy

**arXiv ID:** 2607.27212 | [PDF](https://arxiv.org/pdf/2607.27212v1)

**作者:** Asif Azad `[一作]` (Ministry of Defense), Ehsan Hoque `[通讯]` (Ministry of Defense)

**通讯引用:** 1633 | [OpenAlex ID](https://openalex.org/A5059213806)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个面向阿拉伯语自闭症儿童的全流程多模态 AI 语音语言康复平台 Digital Harf。

**💡 创新点**

创新点在于将内容生成视为核心基础设施，利用 Agentic Synthetic Data Engine 自动生成符合文化与治疗要求的图片、提示和卡片，并实现了从内容生成到多模态疗程、个性化反馈与家长支持的一体化工作流。

**🔧 技术方法**

采用的技术包括生成式 AI（图像生成模型）、多模态评估（语音识别、语义匹配、图片质量评估）、多级判定与记忆机制，以及基于强化学习的个性化内容选择。

**📊 数据集**

数据集主要来自 ASDE 生成的无监督图像与卡片，以及 13 名阿拉伯语资深 SLP 评审的人工标签，用于验证内容可用性和系统评估。

**📈 对比分析**

通过两轮专家评估：第一轮对 27 个未剪裁 ASDE 输出的可接受率为 90.1%；第二轮对完整平台的 5 个维度（临床有效性、文化适配、易用性、个性化、可采用性）进行 Likert 量表评估，文化适配最高（平均 4.15），个性化最低（平均 3.55）。

**⚠️ 局限性**

局限性包括缺乏儿童层面的临床疗效验证、目前仅支持三种疗法模块、主要针对海湾阿拉伯语，未覆盖其他方言及多模态扩展。

---

## 86. From Lecture Notes to Lean: Formalizing a Textbook on Probability Theory

**arXiv ID:** 2607.27298 | [PDF](https://arxiv.org/pdf/2607.27298v1)

**作者:** Shuo Deng `[一作]` (Chinese University of Hong Kong), Kenneth W. Shum `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 2771 | [OpenAlex ID](https://openalex.org/A5103017634)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本工作在 Lean 证明助手中对《Measure‑Theoretic Probability》一书进行系统正式化，构建了与教材对应的机器检验伴随文件，并在此基础上生成可复用的概率理论库；

**💡 创新点**

创新点在于将 AI 助手与人类审阅相结合的“代理式工作流”用于大规模教材正式化，明确区分教材声明与库接口的桥接，强调源代码与数学内容的一致性与可追溯性；

**🔧 技术方法**

主要技术包括 Lean 证明助手、Mathlib 体系、基于 GPT‑5.5、Gemini 等大型语言模型的自动推导与代码生成，以及自定义的 APOLLO‑风格工作流管理与审阅机制；

**📊 数据集**

使用的数据集为教材全文（14 章）以及现有 Mathlib 库；

**📈 对比分析**

在覆盖率方面完成了 81 条定义、127 条定理、107 条示例、134 条练习的正式化；在功能上通过桥接 Lemma 与 Mathlib 原有定理实现了与教材一致的命题；性能方面主要体现在编译通过率和审阅通过率，且对错误的自动检测率较高；

**⚠️ 局限性**

局限性包括：正式化仍未覆盖所有示例与练习；桥接过程繁琐且易出现抽象不匹配；AI 生成的证明可能需要人工大量修正；对更高级主题的支持仍不足，且目前缺乏对教材错误的完整自动纠正框架。

---

## 87. A Linear Bound on the Rainbow Cycle Number and Approximate EFX

**arXiv ID:** 2607.27455 | [PDF](https://arxiv.org/pdf/2607.27455v1)

**作者:** Varun Sivashankar `[一作]` `[通讯]` (Princeton University), Varun Sivashankar (Princeton University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

本文证明了彩虹环数 R(d) 的上界为 R(d) < e·d，并基于此改进了可部分 (1‑ε)-EFX 分配的结果，得到每个加性实例至少存在一组约束条件下的部分分配，其中未分配物品数上界为 O(√(n/ε))，并给出了一个期望多项式时间的随机算法实现该分配。

**💡 创新点**

创新点：
- 用新的计数论证证明了 R(d) 的线性上界 R(d) < e·d，解决了此前猜想 R(d)=O(d)；
- 将此结果与彩虹环数的约简结合，去掉了之前 √log n 的因子，取得最优的 O(√(n/ε)) 未分配物品上界；
- 设计了一个随机化算法，在线性阈值下仍能在期望 O(k²) 时间内找到彩虹环，保证整体算法期望多项式。

**🔧 技术方法**

主要技术：
- 通过“装箱不等式”(packing inequality) 证明无彩虹环的 k‑partite 图满足 (k‑1)!∑|V_i| ≤ ∏|V_i|，进而推出 R(d) < e·d；
- 运用 AM–GM 及 Stirling 近似估计阶乘得到线性界；
- 随机化搜索策略：随机选取终端顶点与类序列，构造路径检查逆向边，利用失败概率 < 1/d 得到期望 O(1) 次成功；
- 结合已有的彩虹环数约简框架，将其与 (1‑ε)-EFX 约束关联。

**📊 数据集**

无数据集，论文完全为理论分析与证明，实验部分只给出期望运行时间的上界，未涉及具体实例或数据。

**📈 对比分析**

与以往工作比较：
- 之前最佳上界为 R(d)=O(d log d)，导致未分配物品上界为 O(√(n log n))；
- 本文得到的 R(d) < e·d 使未分配物品上界提升至 O(√(n/ε))，去掉了 log n 因子，达到最优渐近结果；
- 现有随机算法在期望多项式时间内实现该上界；但尚无确定性多项式时间算法。

**⚠️ 局限性**

局限性：
- 仍受 √(n/ε) 的平方根瓶颈限制；若要进一步降低未分配物品数，需新的公平分配框架；
- 只给出了随机化算法，未实现确定性多项式时间解法；
- 结果仅适用于加性估值的可部分分配，完整 EFX 分配的存在性问题仍未解决。

---

## 88. SKY-Piano: A Multimodal Piano Performance Dataset

**arXiv ID:** 2607.27296 | [PDF](https://arxiv.org/pdf/2607.27296v1)

**作者:** Joonhyung Bae `[一作]` (KAIST), Juhan Nam `[通讯]` (KAIST)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

发布了一个11小时的多模态钢琴表演数据集，包含手部和身体的真实动作捕捉、音频、MIDI、视频、MusicXML分数、Visual3D身体段运动学以及伪指法标签，并提供交互式浏览器和指法注解工具。

**💡 创新点**

该数据集将真实手部和身体动作捕捉与音频、MIDI、视频、分数同步，并以技术、难度、演奏者专业水平为维度结构化曲目，实现了在同一语料库中对不同演奏者和技术的可比性。

**🔧 技术方法**

采用OptiTrack和Qualisys光学动作捕捉、SMPTE时间码同步、SAITS自注意力缺失填补、PianoVAM改造的几何指法注释、Tipiano的手势生成模型以及音频到MIDI的CQT对齐等技术。

**📊 数据集**

结合7名专业和12名业余钢琴家共19位演奏者的演奏，涵盖26项技术练习、15件分级作品及5个自由曲目，录制了约11小时的手部/身体动作、音频、MIDI、视频和分数。

**📈 对比分析**

对Tipiano进行微调后，在留一演奏者交叉验证中MPJPE下降25.9%至48.8mm，关键触键F1由0.93下降至0.66，表明跨域适配效果；与原始Für Elise基准相比，误差缩小至1.5倍。

**⚠️ 局限性**

样本规模有限、曲目主要为传统练习和古典曲目、指法标签为伪注解且含歧义标记，缺乏扩展技术和更广泛演奏者群体。

---

## 89. SkillMentor: LLM Agent Self-Evolution via Learning Blind-Spot Diagnosis

**arXiv ID:** 2607.27360 | [PDF](https://arxiv.org/pdf/2607.27360v1)

**作者:** Xiaoyi Bao `[一作]` (Hong Kong Polytechnic University), Zang Li `[通讯]` (Tencent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

训练一个小型RL Mentor来学习诊断并修复另一个冻结执行者的盲点，外部化修复知识为可重用的Markdown技能库；

**💡 创新点**

将诊断视为可学习的能力并通过联合发现-精炼的强化学习策略实现，展示小模型可以超越大模型的提示式诊断；

**🔧 技术方法**

使用GRPO强化学习训练Mentor、LLM判定器（如DeepSeek-V4-Flash）评估诊断缺口、技能生成与验证、外部技能库管理；

**📊 数据集**

在AppWorld（长时程规划）和BFCLv3（精确函数调用）两个基准上进行实验；

**📈 对比分析**

与无技能基线、Reflexion、MemP、ReasoningBank以及提示式Mentor比较，SkillMentor平均提升约44.2%，并且即使使用较小的LLM判定器也能逼近大模型效果；

**⚠️ 局限性**

主要限制包括：仍需在训练期间使用高性能LLM进行判定；阈值调度和技能仓库维护策略需要手工设计；在完全零人类监督下的通用性仍待进一步验证。

---

## 90. Rethinking EEG-Based Disease Diagnosis: Decoupling Instance Representation Learning from Subject-Level Supervision

**arXiv ID:** 2607.27274 | [PDF](https://arxiv.org/pdf/2607.27274v1)

**作者:** Zhiyuan Ma `[一作]` (Tsinghua University), Sen Song `[通讯]` (Tsinghua University)

**通讯引用:** 13559 | [OpenAlex ID](https://openalex.org/A5013759262)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本文提出 BridgeMIL，一种两阶段的多实例学习框架，用于 EEG 病症诊断。该框架先在大量无标签 EEG 窗口上通过 VICReg 对近邻窗口和子袋进行对齐学习表征，然后在仅有少量受试者标签的前提下，将预训练的编码器迁移到注意力多实例聚合器进行受试者级预测，并通过特征保持避免在微调时失去表征结构。

**💡 创新点**

创新点在于：① 解决了“继承标签”导致的实例级监督不可靠问题，采用两阶段无标签表征学习；② 引入近邻窗口与子袋双尺度对齐，结合方差与协方差正则，避免负样本；③ 在第二阶段加入特征保持（feature retention），在有限标签下保持第一阶段学到的结构。

**🔧 技术方法**

使用技术包括：VICReg 自监督损失、方差与协方差正则、注意力多实例聚合器、特征保持正则、以及五种主干网络（EEGNet、Conformer、LCADNet、DSAINet、MTDNet）。

**📊 数据集**

实验数据集包括三份公开 EEG 病症数据集：ADFTD（3 类，88 受试者）、Mumtaz2017（2 类，63 受试者）和 Rockhill2021（2 类，31 受试者）。

**📈 对比分析**

与多数投票、普通 MIL、现代 MIL、以及两阶段 SupCon/MaskRecon 的对比显示，BridgeMIL 在 15 种 dataset–backbone 组合中有 14 次获得最高平均准确率，整体平均准确率 76.57% 高于最强基线 SupCon 的 72.29%，提升 4.28 个百分点。

**⚠️ 局限性**

局限性包括：① 仍受限于受试者数量稀缺，性能对受试者数更敏感；② 仅在 EEG 任务上验证，未评估跨模态或更大规模数据集；③ 两阶段训练与特征保持增加了实现复杂度与计算成本。

---

## 91. DoTime: A Synthetic Benchmark Generator for Interventional and Counterfactual Time Series

**arXiv ID:** 2607.27263 | [PDF](https://arxiv.org/pdf/2607.27263v1)

**作者:** Dennis Thumm `[一作]` (National University of Singapore), Ying Chen `[通讯]` (National University of Singapore)

**通讯引用:** 33664 | [OpenAlex ID](https://openalex.org/A5100383015)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个可扩展、理论可证明的多变量时间序列结构因果模型（TSCM）生成器 DoTime，并发布了四个冻结评估套件，用于在干预和反事实场景下评估因果推断方法。

**💡 创新点**

创新点在于：① 支持连续时间干预窗口和多种干预类型；② 提供五种反事实采样模式并配有正性守护；③ 将状态切换（ITS）作为严格的通用化；④ 构建了覆盖八种识别结构、可配置规模、并提供可重复的评估协议。

**🔧 技术方法**

技术手段包括：基于结构化 TSCM 的图与机制先验；SDE 采用 Euler-Maruyama 采样生成连续时间序列；干预与反事实实现时的噪声正则化与正性守护；以及使用预训练的 Do-Over-Time-PFN（PFN）进行基线比较。

**📊 数据集**

使用四个冻结套件：dot-Identifiability-v1（约 13.5k 条轨迹，8 种识别结构），dot-RegimeSwitch-v1（10k 条轨迹，2/3/5 状态），dot-Continuous-v1（10k 条轨迹，连续时间窗口），dot-Generic-100k（100k 条轨迹，完整多样化先验）。所有套件均提供 Zenodo DOI 与 Hugging Face 镜像。

**📈 对比分析**

评估方法：对比零、轨迹均值、AR1、VAR、BackDoorOLS、IV2SLS 等基线以及 PFN 的观测版与干预版。主要性能指标为 RMSE 与方向准确率；干预版 PFN 在方向准确率上相对观测版提升约 0.08–0.09，RMSE 在不同套件中接近观测均值，表明方向准确率是衡量因果能力的更敏感指标。

**⚠️ 局限性**

局限性：默认仅支持 N≤10、K≤3；非平稳仅限于参数切换，未覆盖季节性或趋势驱动；干预/反事实目标在离散与连续两种形式下不统一；评估优势主要在训练分布内，跨生成器或真实数据时可能衰减；大规模生成时收敛率下降；缺乏对部分可观测或真实世界多模态数据的完整验证。

---

## 92. Extension Types for Free

**arXiv ID:** 2607.27387 | [PDF](https://arxiv.org/pdf/2607.27387v1)

**作者:** Nicolai Kraus `[一作]` `[通讯]`, Nicolai Kraus

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文构建了一个统一框架，使用两层类型理论（2LTT）来定义并证明所有已知的扩展类型（如路径型、Riehl‑Shulman 的扩展型、受控展开等）在此框架内不需要额外公理即可满足其全部规则与公理。

**💡 创新点**

创新点在于：①把扩展类型的存在转化为可定义而非公设，②利用 2LTT 的“自由”特性自动获得相应模型，③通过该框架证明了粘合（gluing）与 univalence 之间的严格等价，为探讨“book HoTT”与立方体类型理论的保守性提供了新的理论途径。

**🔧 技术方法**

核心技术包括：两层类型理论（内层 HoTT，外层严格等价层）的语义与推理；对 cofibration 的面算子（face calculus）进行抽象；使用 Agda 的 2LTT 模式完成所有定理的自动形式化；以及利用相对函数外延性、重定位原理等内部公理证明扩展类型的完整性。

**📊 数据集**

本文没有使用传统意义上的数据集；所有实验均在 Agda 证明助手中通过类型检查与自动化形式化完成。

**📈 对比分析**

方法比较基于在 Agda 里直接实现和验证所有公理与规则，利用两层类型理论的推理机制保证了严格性；相比传统的单层 HoTT 定理证明，本文不需要额外的公理化步骤，形式化工作已完成 5,700 行代码，证明可靠性高；但未提供性能数值评测。

**⚠️ 局限性**

局限性包括：①对 cofibration 类的闭包（如并发）等额外假设仍需手工验证；②在实现上依赖 Agda 的 2LTT 模式，尚未在其他证明助手上验证；③对 “book HoTT” 与立方体类型理论的完整保守性证明仍为开放问题，本文仅提供了部分理论支持。

---

## 93. Minimum-Width Drawing of Trees with Sized Vertices

**arXiv ID:** 2607.27445 | [PDF](https://arxiv.org/pdf/2607.27445v1)

**作者:** Markus Wallinger `[一作]` (Technical University of Munich), Stephen G. Kobourov `[通讯]` (Technical University of Munich)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了在有尺寸顶点的层式整洁树绘图中，通过选择兄弟顺序来最小化绘图宽度的问题，并给出了对应的模型与算法；

**💡 创新点**

主要创新点包括：①证明该问题在二叉树且顶点宽度均为1的情况下仍为NP‑完整；②构造精确的混合整数线性规划（MILP）模型；③设计一种基于模拟退火的快速启发式，并在实验中证明其接近最优；

**🔧 技术方法**

所使用的技术包括：混合整数线性规划（MILP）与Gurobi求解器；模拟退火（SA）与van der Ploeg的层式布局算法；超参数优化工具Optuna用于调参；Python实现与Rust算法的调用；

**📊 数据集**

实验数据集包含两类：①合成树，采用深度偏好附着模型生成，形状分为 shallow、random、deep，最大度分别为2、7、12，节点数在20–200之间；②真实树，取自TSTP定理证明的20棵树，节点数在13–867之间；

**📈 对比分析**

实验比较方法：与随机兄弟顺序基线对比；在合成数据上，SA在约75%实例中与MILP最优相差≤20‑25%，运行时间在0.13–0.17秒；MILP在n≤≈130时大多能在3600秒内求解完成，超过此规模往往超时；在真实数据上MILP在14/20实例内求解，SA与MILP结果相近；总体上SA提供了近似最优且极快的解决方案；

**⚠️ 局限性**

局限性包括：仅评估了一种启发式和随机基线，缺乏多种基线或更先进算法的对比；真实数据仅来自单一领域（定理证明树），规模有限；MILP求解时间设为3600秒，无法评估更大规模问题；仅考虑统一高度的顶点，未考虑非层式或非统一高度的情况；未直接评估可读性指标（如空白、角度分辨率等）。

---

## 94. AgentS4D: Benchmarking Runtime Risks across the Execution Lifecycle of LLM-Based Workspace Agents

**arXiv ID:** 2607.27294 | [PDF](https://arxiv.org/pdf/2607.27294v1)

**作者:** Jiajun Zhou `[一作]` (Zhejiang University of Technology), Qi Xuan `[通讯]` (Zhejiang University of Technology)

**通讯引用:** 6668 | [OpenAlex ID](https://openalex.org/A5016704080)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套面向LLM工作空间代理的四维运行时安全评估框架，并基于此构建了一个沙箱基准，包含328个风险注入案例。

**💡 创新点**

创新点在于统一了风险进入源、诱导策略、目标危害与生命周期检查点四个维度，并通过完整配置跨LLM与Harness的系统化测试揭示安全行为的多样性。

**🔧 技术方法**

使用了可执行任务自动化、工具调用追踪、状态变更记录与主机端安全验证器等技术，对代理的完整运行进行观测与评估。

**📊 数据集**

基于76个Workspace-Bench可执行任务生成的328个案例，涵盖六种风险源、六种诱导策略和九种目标危害。

**📈 对比分析**

通过20种Harness-LLM组合（4种Harness × 5种LLM）进行6,560次实验，结果显示68%运行触发预设危险信号，完成率达93%，且不同组合、风险载体对安全率有显著影响。

**⚠️ 局限性**

局限性包括仅覆盖有限的风险源与策略，使用合成任务且不涉及真实生产环境，评估依赖于人工设定的安全信号，且映射规则尚未经过盲测验证。

---

## 95. LayerRAG-Bench: A Cross-Layer Reliability Benchmark for Agentic Retrieval-Augmented Generation

**arXiv ID:** 2607.27353 | [PDF](https://arxiv.org/pdf/2607.27353v1)

**作者:** Musa Shams `[一作]` `[通讯]` (Independent Researcher), Musa Shams (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并发布了面向企业的跨层级检索增强生成可靠性基准 LayerRAG‑Bench，用于评估检索、工具调用、权限、会话状态等多层失效。

**💡 创新点**

引入层级故障矩阵和严格/修复模式对照评估，明确不同失效层的诊断与修复边界，并证明仅靠语义对齐无法保证整体可靠性。

**🔧 技术方法**

利用检索增强生成技术、工具调用、schema 归一化、TF‑IDF检索、层级 Bootstrap 统计方法以及 LangChain 集成来构建与评估基准。

**📊 数据集**

构建了合成的政策文件语料（8 个企业域、80 份政策、160 文档），生成 240 个任务（每份政策 3 种变体），用于测试 9 种故障场景。

**📈 对比分析**

在 9 个模型、9 个场景、2 种合同模式下进行 38,880 次评估；严格模式下 schema drift 成功率 0%，修复模式提升至 0.913；其他故障如 stale index、权限被拒绝等未得到修复；LangChain 集成显著提升安全响应率。

**⚠️ 局限性**

数据集为人工合成的政策文档，缺乏真实多样性；检索器仅基于查询域，易失效；仅评估非对抗性故障，未覆盖时延、成本、对抗攻击等实际生产挑战。

---

## 96. Beyond KV Reconstruction: Functional Reconstruction for MLA Draft Models in Speculative Decoding

**arXiv ID:** 2607.27269 | [PDF](https://arxiv.org/pdf/2607.27269v1)

**作者:** Weiye Shi `[一作]` (Peking University), Muhan Zhang `[通讯]` (Peking University)

**通讯引用:** 5043 | [OpenAlex ID](https://openalex.org/A5071515223)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种在已转换的多头潜在注意力（MLA）模型上进行端到端功能重构的方法，以提升推理时草稿模型的接受率，同时保持低额外成本。

**💡 创新点**

创新点在于：在转换后仅使用原始 MHA/GQA 模块的后投影输出作为目标，训练局部查询与 KV 投影，使整个 MLA 块的功能与原块保持一致，而不需要验证器 logits、标签或额外的推理步骤。

**🔧 技术方法**

采用的技术包括低秩因式分解、RoPE 处理、全局端到端 MSE 损失、Transformer attention 层重构、HF 与 vLLM 后端集成，以及使用校准隐藏状态进行批量训练。

**📊 数据集**

使用的数据集为 HumanEval、Alpaca、Natural Questions、CNN/DailyMail 共 200 个 prompt，每个任务最大生成 128 token。

**📈 对比分析**

与原始 GQA 模型和部分 RoPE 重构结果比较；在 192 个配置中，功能重构在 37/64 单元显著提升草稿接受率（最多 +4.23pp），26 单元保持不变，1 单元略降；在 12/64 单元还实现吞吐量提升，整体不增加额外推理成本。

**⚠️ 局限性**

局限性包括：实验仅覆盖 1–8B 规模模型、固定 128-token 长度和单一随机种子；无法完全恢复因低秩或后端不匹配导致的接受率损失；未检验更大模型、长上下文或不同硬件环境的泛化能力。

---

## 97. FADEx: Feature Attribution and Distortion-based Explanation of Dimensionality Reduction

**arXiv ID:** 2607.27463 | [PDF](https://arxiv.org/pdf/2607.27463v1)

**作者:** Lucas Greff Meneses `[一作]` (University of São Paulo), Luis Gustavo Nonato `[通讯]` (University of São Paulo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一种名为FADEx的模型无关本地特征归因方法，用于解释非线性降维技术的投影行为，融合了特征重要性、几何畸变评估和方向性影响向量；

**💡 创新点**

通过基于泰勒展开的局部线性逼近和奇异值分解，FADEx能够在不需要外样本映射的前提下，为每个特征分配单一可解释的归因值，同时兼顾畸变度量和方向性分析；

**🔧 技术方法**

利用局部线性回归（加权岭回归）估计Jacobian，SVD分解求取特征归因与畸变指标（SND），并通过特征影响向量展示局部方向性；

**📊 数据集**

在公开数据集（如Musk Version 2、UCI Breast Cancer、Dermatology、Dry Bean、MNIST、Cats‑and‑Dogs等）以及合成数据集上验证，涵盖t‑SNE、UMAP、Isomap、LLE等多种降维方法；

**📈 对比分析**

与Corbugy、LXDR、ClusterShapley三种主流解释方法比较，FADEx在计算速度上提升约10×、内存占用更低，并在局部归因、畸变一致性和方向性分析上表现更优；

**⚠️ 局限性**

对高维数据的局部PCA预处理依赖，若PCA失效会影响Jacobian估计的精度；对超参数（邻居数、正则化α）敏感，需要在不同数据集上进行调优；

---

## 98. SIGIL: Compiling Agent Skills into Typed Harnesses

**arXiv ID:** 2607.27309 | [PDF](https://arxiv.org/pdf/2607.27309v1)

**作者:** Jayanaka Dantanarayana `[一作]` (University of Michigan), Jason Mars `[通讯]` (University of Michigan)

**通讯引用:** 7338 | [OpenAlex ID](https://openalex.org/A5053236545)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Skill Compilation，将自然语言技能编译为可执行的 agent harness，实现对技能中强制步骤的结构化执行。

**💡 创新点**

创新点在于设计了 typed agentic intermediate representation AG‑IR，分离模型认知与代码机制，并通过编译门、STRUCT‑COV 等手段保证编译结果与源技能一致，且保留可追溯性。

**🔧 技术方法**

使用 AG‑IR 语义图、Owner Test、modality、Jac 语言、Object‑Spatial Programming 目标、meaning‑typed 编程、编译门、runtime harness 与节点轨迹等技术。

**📊 数据集**

评估基于 30 个 agent 技能，涵盖文档、软件流程和治理合规三类，来自公开技能集及自研集。

**📈 对比分析**

与传统 prose agent 在 GPT‑4o 和 GPT‑5 上对比，利用 Applicable‑Mandate Compliance 计量；编译 harness 的合规率从 56% 提升至 86%，完整流程完成率提高 2.3×，平均 token 消耗下降 42%，且合规率对模型能力不变。

**⚠️ 局限性**

局限性包括：对判断型步骤无法强制执行，Invoke‑Unit 循环导致成本上升，评估样本有限，且仅在两款模型上验证，跨模型泛化与极端技能的适用性仍待探索。

---

## 99. A Taxonomy of Human-Robot Teamwork Requirements

**arXiv ID:** 2607.27302 | [PDF](https://arxiv.org/pdf/2607.27302v1)

**作者:** Anastasia Mavridou `[一作]` (KBR Inc. at NASA Ames Research Center), Marie Farrell `[通讯]` (University of Manchester)

**通讯引用:** 652 | [OpenAlex ID](https://openalex.org/A5076259366)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并验证了一个两层次的人机协作需求分类法，系统地组织了361条来自标准与文献的需求，并通过专家评审与448条独立需求的实用性验证来评估其覆盖度。

**💡 创新点**

创新点在于首次将人机团队需求按功能与操作层级细分为六大类21个子类，并针对责任边界与交互模式提供了结构化的评估框架。

**🔧 技术方法**

主要技术包括系统化文献检索与标准分析、需求抽取与双人独立分类、迭代式分类细化以及专家访谈与案例验证的质性分析方法。

**📊 数据集**

使用的数据集为构建语料库361条需求（来自14个跨域来源）和验证语料库448条需求（来自19个来源，覆盖航空、医疗、工业等六大领域）。

**📈 对比分析**

通过将验证语料中的412条需求成功归入现有分类，达到91.9%的覆盖率，显示该分类法在跨域场景中具有较高的适用性和完整性。

**⚠️ 局限性**

局限性包括对英语西方标准的依赖导致域覆盖偏向航空与工业、对人类绩效监测需求的不足、以及仍有部分需求归入“待扩展”子类，未来需要进一步丰富并验证。

---

## 100. Objective-Aligned Direct Answer SFT for Robust Multi-Frame Medical VQA

**arXiv ID:** 2607.27566 | [PDF](https://arxiv.org/pdf/2607.27566v1)

**作者:** Site Li `[一作]` (Yale University), Xiaofeng Liu `[通讯]` (Yale University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `70e40602-aae3-44bd-80ec-4a7f2674330f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文通过在 MedFrameQA 上进行严格对比实验，评估了多种医学多帧 VQA 适配方法的鲁棒性，最终证明目标对齐的直接解码器答案微调（Answer‑only SFT）在保持预算、种子和校准一致的情况下表现最为稳健。

**💡 创新点**

创新点包括：①提出了面向“报告风险”与“优化方差”的鲁棒性评估框架，强调在固定评测协议下比较适配族；②系统证明最简单的目标对齐方法优于复杂控制器或辅助目标；③展示了后处理校准和跨骨干机的迁移能力，进一步验证方法的通用性。

**🔧 技术方法**

技术方法主要包括：LoRA 参数高效微调、轻量化视图适配器、温度缩放与直方图分箱的后处理校准、以及对 MedGemma‑1.5‑4B 与 Qwen2.5‑VL‑3B 等多模态骨干的适配；评估使用多种统计工具（均值、标准差、配对 Bootstrap 等）。

**📊 数据集**

数据集：MedFrameQA（2851 例多帧问答），采用 MedGemma‑1.5‑4B 作为主骨干，Qwen2.5‑VL‑3B 用于迁移实验。

**📈 对比分析**

对比方法：在相同数据切分、匹配训练预算、重复种子与统一校准下，比较冻结直接推理、直接答案微调、答案连续、硬负样本持续、硬负样本微调、静态混合等多族。实验表明，答案微调族的报告准确率比冻结基线提升约6个百分点，方差仅 0.5‑0.6，且在目标、锚点和硬定位切片上保持竞争力；后处理校准显著降低 ECE；迁移至 Qwen2.5‑VL‑3B 则获得约 +2.6% 的准确率提升。

**⚠️ 局限性**

局限性：实验仅聚焦于只评价最终答案的报告指标，无法证明在更广泛任务或更大规模数据上的表现；辅助机制虽能提升切片特定性能，但在整体鲁棒性上并未超越直接微调；结果受限于所选模型骨干与调参空间，其他骨干或更复杂模型可能表现不同；未深入探讨长周期部署与实时推理的实际成本与安全性。

---

## 101. Good Rankers, Bad Objectives: Bilinear Contrastive Critics under Expressive Policy Search

**arXiv ID:** 2607.27422 | [PDF](https://arxiv.org/pdf/2607.27422v1)

**作者:** Ayushman Singh `[一作]` (Stanford University), Siddharth Aphale `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

评估并揭示了对比式评估器在最大化候选动作时的安全性缺陷，阐明兼容性排序与价值排序的分离；

**💡 创新点**

提出“兼容性‑价值分离”框架，并通过理论证明与实验验证对比式评估器在候选选取中的高风险；

**🔧 技术方法**

使用对比学习、三种读数（原始双线性、余弦、混合）以及Bellman回归的TD‑Q；通过候选最大化（best‑of‑K）和控制实验，结合Kendall τ、回报差等统计指标；

**📊 数据集**

采用OGBench的四个导航任务（PointMaze、AntMaze、HumanoidMaze、AntSoccer）及四个操作任务，以及自定义的2D控制实验；

**📈 对比分析**

通过对比兼容性检索（AUC）、价值排序（Kendall τ）及固定查询下的最大化实验（归一化回报），发现对比式评估器在价值排序和回报上表现差（τ≈0、回报≈0），而TD‑Q在所有指标上明显优于它们；

**⚠️ 局限性**

实验仅覆盖导航与玩耍数据，误排序对不同任务的成本差异显著；未探究结构化或支持约束的对比目标，也未测试更复杂策略和更多任务，限制了结论的普适性。

---

## 102. Prompt Chaining in Practice: A Case Study in Automated Scholarly Report Generation

**arXiv ID:** 2607.27210 | [PDF](https://arxiv.org/pdf/2607.27210v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 103. Regularizing modality contribution drift in multimodal continual learning

**arXiv ID:** 2607.27260 | [PDF](https://arxiv.org/pdf/2607.27260v1)

**作者:** Zhen Zhang `[一作]` (Southwest Jiaotong University), Tianrui Li `[通讯]` (Southwest Jiaotong University)

**通讯引用:** 25959 | [OpenAlex ID](https://openalex.org/A5070559820)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多模态持续学习中的模态贡献漂移问题，提出了一种新的正则化方法 CMCDR，用来在新任务学习过程中保持旧任务的模态贡献结构，并在有无记忆样本的两种设置下进行实验。

**💡 创新点**

创新点包括：① 定义并量化“模态贡献漂移” (MCD) 分数；② 通过干预式贡献估计（Möbius 变换）得到完整的模态贡献谱；③ 提出可在有 Replay 与无 Replay 两种场景下使用的 CMCDR 正则化，直接约束旧任务贡献的绝对值和相对比例；④ 证明跨模态表示的稳定并不等价于决策层贡献的稳定。

**🔧 技术方法**

技术手段：干预式贡献估计、Möbius 变换分解、Smooth‑L1 损失匹配、温度软化权重、CKA 表示漂移评估、Replay 与无 Replay 的贡献对齐策略、理论证明（贡献漂移导致遗忘的充分条件）。

**📊 数据集**

实验数据集：多模态分类 - AVE、Kinetics‑Sounds、UESTC‑MMEA‑CL；多模态问答 - VQAv2、Split‑AVQA、Split‑MUSIC‑AVQA。

**📈 对比分析**

与 LwF、EWC、iCaRL、AV‑CIL、CIGN、MMAL 等基线以及表示对齐方法 RepAlign、CrossSDC 进行对比；CMCDR 在有 Replay 与无 Replay 两种模式下均能提升 2–8% 的平均准确率，减少 3–11% 的平均遗忘；与表示对齐方法联合使用时，准确率提升 3–4.6 点、遗忘降低 3.9–8.1 点、MCD 降低 0.163–0.343。

**⚠️ 局限性**

局限性：① 计算所有子集的贡献谱在模态数较多时开销高；② 需要在每个增量阶段重新干预计算，训练时间相对增加；③ 仅在实验中验证了固定数量模态（2–3）和特定 MMCL 任务格式，未探索在更大模态集合或非任务边界重叠场景下的表现；④ 对极端稀缺样本或无记忆样本时的效果仍有待进一步验证。

---

## 104. PlatformBid: An Auto-Bidding Benchmark from a Unified Advertising Platform's Perspective

**arXiv ID:** 2607.27265 | [PDF](https://arxiv.org/pdf/2607.27265v1)

**作者:** Shengtian Yang `[一作]` (Southeast University), Lei Feng `[通讯]` (Southeast University)

**通讯引用:** 2924 | [OpenAlex ID](https://openalex.org/A5100682348)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 PlatformBid 统一广告平台视角的自动竞价基准，并在此基准上评估多种算法。

**💡 创新点**

设计三种代表性竞争设置并提出基于流匹配的 BidFlow 方法。

**🔧 技术方法**

采用离线强化学习、生成模型与流匹配技术，构建 BidFlow 的 BC flow 与一阶策略。

**📊 数据集**

主要使用 AuctionNet 数据集，亦在 iPinYou 上进行跨数据集验证。

**📈 对比分析**

在三种设置下与 PID、IQL、BCQ、DT、GAS、GAVE、CBD 等基线对比，BidFlow 在转化率、CPA 及平台收益上均领先，在线 A/B 实验提升 0.68%。

**⚠️ 局限性**

仅聚焦自动竞价，未考虑拍卖机制设计和广告主关系建模，在稀疏奖励的异构竞争场景仍表现欠佳。

---

## 105. Some Experiments with Twee-Style Goal-Directedness

**arXiv ID:** 2607.27442 | [PDF](https://arxiv.org/pdf/2607.27442v1)

**作者:** Stephan Schulz `[一作]` `[通讯]` (DHBW Stuttgart), Stephan Schulz (DHBW Stuttgart)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种基于共享项的目标导向子句选择方法，以提升基于饱和的定理证明器的效率。

**💡 创新点**

① 将Twee在单元等值逻辑中使用的显式子项重写转换推广到完整一阶（及高阶）逻辑；② 引入一种隐式的基于共享项标记的符号计数评估函数，避免显式重写带来的搜索空间膨胀。

**🔧 技术方法**

利用E定理证明器的共享项存储机制、给定子句算法、符号计数与重写定义，并实现了新的评估函数（GD-评估）。

**📊 数据集**

使用TPTP 8.2.0中19351个问题（包括UEQ、CNF、FOF、TF0四类）进行实验。

**📈 对比分析**

与基准符号计数策略相比，递归重写转换将可解问题从8600提升到9386（+800）；只对最大子项使用重写提升到8716（+300）；隐式GD评估提升到9010（+400）。四种策略合并可解9804个问题，进一步提升了约1200个问题。

**⚠️ 局限性**

① 显式重写会增加子句数和符号，且无法与其他启发式混合使用；② 隐式评估难以与E现有的所有评估函数兼容；③ 仅对参数空间做了有限探索，未尝试更广泛的组合或更激进的重写预处理。

---

## 106. AgenticER: the next frontier in Entity Resolution

**arXiv ID:** 2607.27435 | [PDF](https://arxiv.org/pdf/2607.27435v1)

**作者:** George Papadakis `[一作]` (National and Kapodistrian University of Athens), Themis Palpanas `[通讯]` (Université Paris Cité)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并概念化了Agentic ER，将实体解析转为自主智能体的顺序决策过程

**💡 创新点**

将ER视为动态规划和多步推理，强调证据获取、交互与成本感知

**🔧 技术方法**

强化学习/决策理论、LLM、LangGraph、n8n、检索工具、外部知识图谱

**📊 数据集**

未提供具体数据集，提出了面向Agentic ER的Benchmark需求

**📈 对比分析**

暂无实验对比，作者只给出理论框架和评价维度

**⚠️ 局限性**

缺乏实现细节、可扩展性验证、成本模型评估及交互学习机制

---

## 107. Context-Informed Ship Trajectory Prediction via Conditional Attention

**arXiv ID:** 2607.27418 | [PDF](https://arxiv.org/pdf/2607.27418v1)

**作者:** Yuan Guan `[一作]` (Carnegie Mellon University), Pradeep Ravikumar `[通讯]` (Carnegie Mellon University)

**通讯引用:** 8279 | [OpenAlex ID](https://openalex.org/A5053209283)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Conditional Informer，一种利用条件注意力将船舶状态、环境上下文和静态元数据进行融合的编码‑解码架构，用于长周期航迹预测。

**💡 创新点**

创新点在于将船舶运动建模为对环境的条件生成任务，使用方向性条件注意力只让船舶状态查询环境而非相互，避免对称联合分布；并引入模态遮蔽训练策略提升在缺失环境或元数据时的鲁棒性。

**🔧 技术方法**

采用基于 Informer‑TP 的 Transformer，ProbSparse 关注机制、Self‑Attention Distilling、条件注意力、多模态嵌入和模态遮蔽。

**📊 数据集**

使用 AIS 船舶轨迹（2009‑2025，Gulf of Mexico 2023）、ERA5 气象再分析（风速、波高）和船舶静态元数据（船体尺寸、货物类型）。

**📈 对比分析**

与基线 InformerTP（仅状态）和 Concatenated 方式进行对比，在 L=24、P=12 的设置下，Conditional Informer 在所有模态可用时平均 Haversine 距离降低约15‑16%，且在模态缺失时误差仅提升至约8.8 km，而无遮蔽模型则爆炸到 67 km。

**⚠️ 局限性**

局限在于对环境缺失时仍比纯 kinematic 模型差距较大，模态遮蔽未完全消除 shortcut 依赖；并且评估仅针对货船开放水域航迹，未验证港口进近或多船类型场景。

---

## 108. FunL2O: LLM-Guided Feature Function Design for Learning to Optimize

**arXiv ID:** 2607.27389 | [PDF](https://arxiv.org/pdf/2607.27389v1)

**作者:** Bingheng Li `[一作]` (Michigan State University), Dzung T. Phan `[通讯]` (IBM Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出FunL2O框架，通过LLM生成可执行的特征函数并在L2O管道中迭代搜索，以自动化优化问题的特征设计；

**💡 创新点**

创新点在于将特征设计建模为可执行程序搜索问题，利用语义契约保证候选函数合法性，并通过完整L2O训练与下游性能评估来驱动搜索；

**🔧 技术方法**

主要技术包括大型语言模型（Claude-Opus、GPT-5.5、Gemini、DeepSeek）、程序生成与验证、语义契约、L2O管线内的迭代搜索和可视化特征功能；

**📊 数据集**

使用八个L2O管道的多种优化任务：线性规划、二次规划、受限非线性、混合整数规划等，涵盖IPM-MPNN、FSNet、DC3、PDHG-Net、Smart Initial Basis、Learning to Pivot、Predict-and-Search、Learned Backdoor等；

**📈 对比分析**

与手工设计特征对比，FunL2O在所有任务中均取得改进：目标值差距下降、可行率提升、迭代/时间/枢轴次数显著减少，且在MILP预测和分支中平均壁时减少10%~55%；

**⚠️ 局限性**

局限性主要在离线搜索成本高（每个候选需完整重训并评估），且搜索可能在有限验证样本上过拟合，但实验表明在独立测试集上仍能保持收益。

---

## 109. OwlPath: Lossless Knowledge Compression for LLM Bug Repair

**arXiv ID:** 2607.27249 | [PDF](https://arxiv.org/pdf/2607.27249v1)

**作者:** Bo Zhang `[一作]` (Shunfeng Technology Company Limited), Xiang Song `[通讯]` (Shunfeng Technology Company Limited)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `fede83ac-7505-405f-ab37-e7284695c47f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

为LLM软件工程代理构建OWL2知识压缩检索层OwlPath，利用本体推理快速获取多跳结构信息

**💡 创新点**

创新点在于将代码图无损压缩成OWL2本体，实现一次性多跳SPARQL查询并提供仅3KB的模块与关键符号摘要

**🔧 技术方法**

技术栈包括Tree‑sitter解析、SQLite代码图、OWL2本体投影、SPARQL 1.1属性路径、OWL‑SKM压缩摘要与Python/Node.js实现

**📊 数据集**

使用SWE‑bench Pro（731个实例）与Lite（300个实例）、67个离线检索实例以及37道结构检索题目进行评估

**📈 对比分析**

与CodeGraph字符串检索基线对比，OwlPath在Token上节省28.8%、时长降低39.5%，离线检索召回提升2.06×、hit率达88.1%

**⚠️ 局限性**

局限包括投射一次性耗时约4.7分钟、关键词提取对模糊描述敏感、跨语言关系支持有限以及仍需手动维护本体模式

---

## 110. Send and Pretend: Exploiting Transcript Consistency Issues in End-to-End Encrypted Group Chats

**arXiv ID:** 2607.27510 | [PDF](https://arxiv.org/pdf/2607.27510v1)

**作者:** Gabriel K. Gegenhuber `[一作]` (Interdisciplinary Transformation University), Aljosha Judmayer `[通讯]` (University of Vienna)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统评估了主流 E2EE 群聊应用（WhatsApp、Signal、iMessage、Threema）的转录一致性（TC）问题，发现它们均无法在群组中保证消息一致性，并通过实验演示了恶意成员利用协议和实现缺陷实现信息篡改、投票操纵等多种攻击场景。

**💡 创新点**

创新点在于首次全面揭示恶意群组成员可利用 Sender Key 及对等加密通道实现转录不一致的完整攻击链，并提出在现有 Sender Key 协议上仅做极小改动即可实现一致性警告的实用改进方案。

**🔧 技术方法**

技术手段包括协议逆向分析、定制化客户端（whatsmeow、Signal Desktop、Threema Android 等）、多设备多平台实验、消息重传与密钥分发机制的细粒度操控，以及对投票、位置共享等高级功能的攻击实现。

**📊 数据集**

实验使用人工构造的测试群组和多设备环境，收集不同平台客户端在相同攻击条件下的消息传递行为，未使用公开数据集，而是通过真实客户端与自研测试脚本模拟完整交互。

**📈 对比分析**

通过对比四大应用在同等攻击情境下的响应，证明所有系统均易受攻击；改进方案在协议层只增加服务器时间戳和广播重传，几乎无额外通信开销；未做量化性能评估，重点在攻击可行性与检测提示的可视化。

**⚠️ 局限性**

局限性包括：警告机制仍可能产生误报，未能彻底防止已存在的服务器投递缺陷；对私有群组或 MLS 等新协议的适配有限；改动需各平台协同实施，且缺乏对大规模真实群组的实测验证。

---

## 111. Corrigible Assistance in One Round: Pragmatic-Pedagogic Best Response

**arXiv ID:** 2607.27508 | [PDF](https://arxiv.org/pdf/2607.27508v1)

**作者:** Elle Lazarski `[一作]` (Princeton University), Jaime Fernández Fisac `[通讯]` (Princeton University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种在特定助力游戏中，利用一次最佳回应即可解决目标不确定性的框架；

**💡 创新点**

创新点在于将助力游戏与层级式人机推理结合，识别“action‑separable”游戏，使人机能在单步内完成目标辨识，突破传统逆向优化（IOC/IRL）的推断上限；

**🔧 技术方法**

使用基于POMDP的助力游戏模型，构建层级贝叶斯推理（H0→R1→H2→R3）以及Pragmatic–Pedagogic Best Response（PPBR）算法；

**📊 数据集**

在简易的拼块搭建（Tetris‑style）协作任务上进行验证，数据由人工设置的两种目标和有限动作空间组成；

**📈 对比分析**

与传统IOC/IRL方法比较，PPBR在单步推理后即可获得完全信息，达到oracle策略价值；实验显示PPBR突破IOC的推断上限，输出更符合人类意图的机器人动作；

**⚠️ 局限性**

局限性在于仅适用于action‑separable（完全可区分）游戏，无法处理“split leverage”（单动作对多目标最优）和多步推理的通用情形；未来工作需扩展至更一般的助力游戏和更复杂的数据集。

---

## 112. Models for minimalist RAG: B1ade 335M Embedding and 1B Parameter Small Language Models

**arXiv ID:** 2607.27506 | [PDF](https://arxiv.org/pdf/2607.27506v1)

**作者:** Shreyas Subramanian `[一作]` (Amazon), Vikram Elango `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了B1ade系统，通过零训练模型融合构建高效检索器B1ade-embed和使用GRPO训练的1B级语言模型B1ade-1B，实现资源高效的检索增强生成（RAG）

**💡 创新点**

创新点在于：①使用参数无关的模型融合实现高性能检索器，无需额外训练；②仅用ROUGE相似度奖励的GRPO训练，使模型在无显式引文监督下自然出现42.4%的引文率；③通过小规模高效训练达成与大模型相近的RAG表现

**🔧 技术方法**

技术包括：零训练模型融合（Mergekit+Model Stock）、GRPO（Group Relative Policy Optimization）强化学习、RoUGE-L奖励、LoRA微调、speculative decoding等

**📊 数据集**

数据集为自制的simpleCoT（约2.2M例、723M tokens，覆盖多跳QA、指令、推理、常识）以及公开的MTEB、PopQA、PubMedQA、FEVER等基准

**📈 对比分析**

与大模型对比：B1ade-1B在PopQA 81.82%、PubMedQA 65.8%、FEVER 51.09%，RAG平均分0.654，超过SFT模型10.8%，仅比Qwen-1.5B少33%参数，但在正确率略低，整体性能接近大模型

**⚠️ 局限性**

局限性包括：①GRPO奖励可能在大规模数据下偏向引文导致其他指标下降；②检索器对不同领域的泛化受限；③评估受单一LLM-judge（Claude Sonnet 4）偏好影响；④未覆盖创造性、对话、代码等生成场景

---

## 113. A Lightweight Foundation Model for Collider Physics with Multi-Domain Adaptation

**arXiv ID:** 2607.27501 | [PDF](https://arxiv.org/pdf/2607.27501v1)

**作者:** Liangyu Wu `[一作]` (Stanford University), Julia Gonski `[通讯]` (SLAC National Accelerator Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并训练了轻量级的全连接自编码器NEXUS，利用LHC碰撞事件的轨迹特征进行无监督预训练，并将其作为基础模型应用于四个碰撞学下游任务和三个跨域任务。

**💡 创新点**

证明在无需监督且仅3百万参数的自编码器架构上即可实现与Transformer相当的迁移学习效果，显著降低计算成本，开辟低功耗、实时、FPGA部署的基础模型路径。

**🔧 技术方法**

使用无监督自编码器训练、全连接网络、GELU激活、dropout、Huber损失、Adam优化，以及线性探针/微调策略；并与参数匹配的Transformer基准进行对比。

**📊 数据集**

预训练使用ATLAS开放数据的2016-Run2碰撞事件（约2000万事件）；下游分类使用20个SM/BSM类别的MC样本；跨域任务包括LIGO O3 GW数据、海岸洪水时间序列和猴子神经电活动数据。

**📈 对比分析**

与参数匹配的Transformer基准比较，NEXUS在推理时每事件MMAC仅0.006 GFLOPs，训练速度约46倍；在20类SM分类的线性探针性能相当（<2%差距）；在下游任务中，预训练模型在少量标签数据下实现更高准确率，数据规模曲线显示更快收敛。

**⚠️ 局限性**

对于高维、密集模态（如2D图像或3D点云）可能需要更复杂的架构；FPGA实现仍需剪枝、量化等压缩技术；尚未系统界定何时自编码器优于Transformer。

---

## 114. INCLAIR: Inception-Based Longitudinal Clinical Anomaly Detection with Informed Reasoning

**arXiv ID:** 2607.27487 | [PDF](https://arxiv.org/pdf/2607.27487v1)

**作者:** Maxx Richard Rahman `[一作]` (German Research Center for Artificial Intelligence), Wolfgang Maass `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了INCLAIR框架，用于在长期临床数据中检测异常并生成符合专家解释的自然语言说明，支持有限专家监督。

**💡 创新点**

创新点包括：将历史上下文组合成U-统计量进行分数估计，使用稀疏异常衰减的top-k聚合，利用不完全子集采样控制推理成本，以及基于A/B测试与三方评审的LLM选择与蒸馏机制。

**🔧 技术方法**

技术手段涵盖：基于Inception的多尺度子序列表示网络、U-统计理论分析、对抗平滑正则、LoRA微调、专家-合成解释审核以及多模型A/B与三方评判的LLM管道。

**📊 数据集**

使用的数据集包括尿激素序列（Steroid）、阿尔茨海默症影像生物标志物（ADNI）和大规模生命体征（P19），并在Steroid上做DNA验证的案例研究。

**📈 对比分析**

与Beta-VAE、V-LSTM、SUOD、LSCP、IsoForest、SACNN等传统与深度基准以及多种专用与通用LLM进行比较，INCLAIR在准确率、AUC、PR曲线、BERTScore等指标上持续优于所有基线。

**⚠️ 局限性**

局限性在于短序列（如ADNI）导致方差高、灵敏度受限；专家解释样本稀缺，生成解释仍可能产生偏差；缺乏对更广泛病种或不同特征空间的泛化验证。

---

## 115. OneShot: Index-in-Ranking with Neural Scoring for Large-Scale Retrieval

**arXiv ID:** 2607.27475 | [PDF](https://arxiv.org/pdf/2607.27475v1)

**作者:** Ziwei Li `[一作]` (Meta Platforms, Inc.), Ji Liu `[通讯]` (Meta Platforms, Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd`

**🎯 论文内容**

论文未提供具体内容，因此无法确定做了什么。

**💡 创新点**

论文未提供具体内容，因此无法确定创新点。

**🔧 技术方法**

论文未提供具体内容，因此无法确定使用了什么技术。

**📊 数据集**

论文未提供具体内容，因此无法确定使用了什么数据集。

**📈 对比分析**

论文未提供具体内容，因此无法确定比较的方法及性能。

**⚠️ 局限性**

论文未提供具体内容，因此无法确定限制因素。

---

## 116. MUGEN: A Unified Framework for Efficient Motion Understanding and Generation

**arXiv ID:** 2607.27581 | [PDF](https://arxiv.org/pdf/2607.27581v1)

**作者:** Zhankai Ye `[一作]` (Florida State University), Xin Liu `[通讯]` (Florida State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的运动–语言框架，使用少量连续潜在槽（latent slots）替代离散码本，既能从文本生成运动，也能从运动生成文本。

**💡 创新点**

创新点包括：① 无需离散码本，使用自适应长度自编码器将任意长度运动压缩为固定数量的连续潜在槽；② 通过深度路由（depth‑routing）让每个槽从 Transformer 的不同层读取信息，提升表达能力；③ 采用低秩（low‑rank）因子头预测潜在槽的联合分布，使单次采样即可捕捉描述所允许的跨槽变异。

**🔧 技术方法**

主要技术包括：自适应长度自编码器（ALAE）、跨注意力压缩、深度路由、低秩因子头、共享的 GPT‑2 语言模型、联合训练（生成+理解）以及校准采样。

**📊 数据集**

在两个公开基准上评估：HumanML3D（短文本描述）和 SnapMoGen（长文本描述），两者均使用标准 20 次复制评估协议。

**📈 对比分析**

与多阶段离散码本、扩散、Diffusion 等现有方法相比：在 HumanML3D 上，Fid 仅略逊于最优离散码本模型，但在检索精度、CIDEr、BLEU@4、CLIP 得分上均超越所有基线；在 SnapMoGen 上，检索、CLIP 以及多模态对齐指标均领先，Fid 与离散码本模型相当。推理成本显著降低，单个运动的计算量约为 MoMask++ 与 MotionGPT3 的 1/10，推理延迟缩短 6–14 倍。

**⚠️ 局限性**

主要限制：① FID 仍落后于最强离散码本模型，说明单次采样在重建细节上略有不足；② 由于使用单一连续潜在槽集合，生成多样性相对受限，尤其在需要大量细粒度差异的场景；③ 对长序列运动（如 SnapMoGen）仍需更大潜在槽预算，可能影响推理效率。

---

## 117. Inference-Time Agentic Decision Rules Beat Longer Evolving Search for Multi-Image Medical Reasoning

**arXiv ID:** 2607.27564 | [PDF](https://arxiv.org/pdf/2607.27564v1)

**作者:** Site Li `[一作]` (Yale University), Xiaofeng Liu `[通讯]` (Yale University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对多图像医学VQA任务MedFrameQA进行系统的推理时代理策略比较，并在相同的高预算ShinkaEvolve搜索框架下评估不同决策规则；

**💡 创新点**

证明在多图像医学推理中，决策规则的选择比更长的搜索预算更重要，轻量级的顺序投票策略优于更复杂的二次排序和单一提示；

**🔧 技术方法**

使用ShinkaEvolve进行提示程序进化，结合多图像视觉语言模型和局部搜索、迁移、档案管理等技术；

**📊 数据集**

主要使用MedFrameQA内部冻结拆分（1,331演化池，665选择保留，855最终测试）以及公开的多模态医学数据集；

**📈 对比分析**

对五种推理时代理策略（Fixed、Reasoning、Order‑Vote、Order‑Rerank、Order‑Vote+）进行重复实验（5次），Order‑Vote在最终测试上取得57.89%准确率，明显优于固定策略（52.73%）和Order‑Rerank（55.79%）；

**⚠️ 局限性**

仅在单一内部拆分上评估，未验证跨模型或跨数据集的鲁棒性；搜索预算扩展至100代未提升最终性能，说明更大搜索可能导致过拟合；

---

## 118. Strategies for Milestone-driven Start-ups in Multi-activity Settings

**arXiv ID:** 2607.27563 | [PDF](https://arxiv.org/pdf/2607.27563v1)

**作者:** Zhengli Wang `[一作]` `[通讯]` (University of Hong Kong), Zhengli Wang (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

建立了一个多控制的连续时间随机控制模型，用于描述初创企业在达到里程碑目标时的决策过程，并求解了最优策略。

**💡 创新点**

首次提出可出现不同类型的有效前沿（efficient frontier）曲线，并证明在不同参数区间内最优策略结构发生质变；同时引入“风险度”与“成本效益”两个直观指标来刻画控制选项。

**🔧 技术方法**

采用哈密顿-雅可比-贝尔曼(HJB)方程与一系列常微分方程（ODE）的解析解法，利用有效前沿排序与阈值策略构造最优策略；对自由下界情况亦给出相应分析。

**📊 数据集**

无实验数据集，全部为理论推导与数值示例；数值示例使用人工设定的参数（如三种控制的漂移、波动率与成本）。

**📈 对比分析**

通过与简单的“贪婪”策略（始终使用成本效益最高的控制）进行比较，证明最优策略在期望收益上可显著优于贪婪策略，差距可逼近里程碑奖励值。

**⚠️ 局限性**

局限性包括：模型仅考虑单一状态变量，无法捕捉多维企业指标；缺乏实证检验；对高维控制问题的扩展仍存在难度。

---

## 119. DeepResearch Agent System

**arXiv ID:** 2607.27562 | [PDF](https://arxiv.org/pdf/2607.27562v1)

**作者:** Yong Huang `[一作]`, for the team Collaboration `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于稀疏激活的大型语言模型深度研究代理系统，支持128K上下文窗口、双模式推理（ReAct+IterResearch）、多工具协同、强化学习优化以及自动数据合成。

**💡 创新点**

创新点包括：30B总参数仅3B激活的MoE稀疏架构；分层注意力实现128K上下文；双模式推理框架（ReAct与IterResearch）；基于token级别的GRPO强化学习；自动数据合成流水线；以及动态梯度检查点、激活卸载和分布式推理的内存与速度优化。

**🔧 技术方法**

采用的技术包括：Mixture-of-Experts（稀疏激活）、层次注意力机制、ReAct/IterResearch推理模式、GRPO强化学习、动态梯度检查点、激活卸载、张量并行与流水线并行等。

**📊 数据集**

使用了公开基准数据集：Humanity's Last Exam、BrowserComp Chinese、WebWalkerQA；以及通过自研的种子扩展流水线生成的自动数据合成集。

**📈 对比分析**

与同等规模稠密模型及传统长文本模型进行对比，取得3.2倍推理速度、45%内存下降、128K上下文准确率+18.7%/召回+23.4%；在三大基准分别达到87.3%、85.3%、91.2%；强化学习相比传统方法提升训练稳定性35%并加速收敛42%；工具使用准确率92.1%。

**⚠️ 局限性**

局限性包括：稀疏激活路由的稳定性和负载平衡仍需改进；分层注意力在极长文本上可能出现注意力稀释；IterResearch规划仍依赖手工设计，难以自动处理极其复杂的研究任务；系统对GPU资源依赖较大，缺乏跨域工具与个性化适配支持。

---

## 120. AI Literacy: An Exercise in Power-Knowledge

**arXiv ID:** 2607.27547 | [PDF](https://arxiv.org/pdf/2607.27547v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 121. Drawing-Recode: Annotation Grounding for Parametric CAD Code Generation from Raster 2D CAD Drawings

**arXiv ID:** 2607.27558 | [PDF](https://arxiv.org/pdf/2607.27558v1)

**作者:** Mingi Kim `[一作]` (Chungnam National University), Hyungki Kim `[通讯]` (Chungnam National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为Drawing‑Recode的框架，可将扫描的二维 CAD 图像自动生成可编程的参数化 CAD 代码。

**💡 创新点**

创新点在于独立提取几何层和注释层，并通过交叉注意力与 Annotation Grounding Loss 实现显式对齐，解决了传统单编码器难以处理尺寸信息的问题。

**🔧 技术方法**

采用 YOLO+SVTR 进行注释检测与识别，CLIP ViT‑L/14 提取几何特征，交叉注意力模块与 AGL 进行对齐，并使用 LLM（Qwen2.5‑0.5B）生成 SPCC 格式代码。

**📊 数据集**

数据集为基于 DeepCAD 的二维 CAD 图像集合，包含机械零件的多视图渲染并保留真实尺寸信息。

**📈 对比分析**

与 Drawing2CAD、CAD2Program 等基线对比，Drawing‑Recode 在 ACC_cmd、ACC_param、MCD、IR 等指标上均优于所有基线，尤其在扫描仿真图像下鲁棒性最高。

**⚠️ 局限性**

限制在于仅使用 DeepCAD 数据，无法覆盖更复杂的工业零件；对真实扫描图像的评估尚未展开，需在更丰富的数据集上验证。

---

## 122. Evaluating Agentic Bioinformatics through Function, Evidence, and Validation

**arXiv ID:** 2607.27556 | [PDF](https://arxiv.org/pdf/2607.27556v1)

**作者:** Phuc Pham `[一作]` (University of Alabama at Birmingham), Truong-Son Hy `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对109个代理式生物信息学系统和28个基准资源进行了系统综述，提出并应用了函数–证据–验证（FEV）框架，对跨领域的工作流功能、证据来源和验证水平进行定量映射与比较。

**💡 创新点**

创新点在于将工作流轨迹作为主要评估单位，构建了三维FEV框架，明确区分功能演示、可追溯证据和验证阶段，提供了一套统一、可操作的科学责任评估方法。

**🔧 技术方法**

采用文献挖掘与编码、定量统计分析、可视化绘图等技术，对公开论文中的系统功能、证据和验证进行系统化归纳。

**📊 数据集**

使用了128篇唯一发表的论文所包含的109个系统条目和28个评估资源，所引用的数据集来源于各自研究的公开实验或公共数据库。

**📈 对比分析**

通过功能维度（F1–F6）、证据维度（E1–E6）和验证阶段（V0–V4）的分布和关联度进行比较，结果显示大多数系统处于V3验证阶段，仅7个系统达到V4，功能覆盖广但验证深度不足。

**⚠️ 局限性**

局限性包括：依赖公开文献的可报告信息，可能漏评未公开的系统；评估主要基于文献描述，缺乏实验或真实数据的直接验证；对不同领域的评估标准不完全统一，可能影响跨领域比较。

---

## 123. Cross-Embodiment Transfer via Behavior-Aligned Representations

**arXiv ID:** 2607.27549 | [PDF](https://arxiv.org/pdf/2607.27549v1)

**作者:** Ajay Sridhar `[一作]` (Stanford University), Dorsa Sadigh `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了行为对齐表示（如物体框、语言动作、末端执行器轨迹）在跨实体仿学习中的作用，构建了一个基于RoboCasa的跨实体仿真基准，并在仿真与实机上验证了其对跨实体和仿真到实机迁移的提升。

**💡 创新点**

提出使用行为对齐表示实现隐式数据对齐，从而在大规模跨实体数据中实现更稳健的迁移；同时设计了可扩展的基准和多种表示融合策略（单表示、ECoT、联合表示）。

**🔧 技术方法**

基于MiniVLA的视觉‑语言‑动作（VLA）框架，加入对多种行为对齐表示的预测和条件化；使用Grounding DINO、预训练轨迹检测模型、语言动作生成脚本；在仿真与实机上训练和微调。

**📊 数据集**

使用RoboCasa平台生成的三种机器人（IIWA、Kinova3、UR5e）仿真数据（XP‑900 / XP‑3K），以及Panda、Jaco、Panda‑OG等目标机器人的人类演示；实机实验采用Franka Research 3和ViperX 300 S。

**📈 对比分析**

对比了无表示、单表示、ECoT、联合表示四种方法，在仿真任务中显示联合表示在大规模跨实体数据上提升15‑19%成功率，实机上提升28%任务完成进度；同时证明在推理阶段不预测表示也能保持大部分收益。

**⚠️ 局限性**

实验中的实体对齐相对较好（相同摄像头、场景分布），限制了对更大异构场景的泛化；所用表示主要针对物体中心操作，其他任务或更大差异的实体可能需要新的表示；并未深入探讨在高差异实体间的迁移瓶颈。

---

## 124. When Does Explicit View Routing Work? A Controlled Study of Multi-View Graph-Text Alignment

**arXiv ID:** 2607.27530 | [PDF](https://arxiv.org/pdf/2607.27530v1)

**作者:** Xiao Yue `[一作]` (Oakland University), Guangzhi Qu `[通讯]` (Oakland University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在严格段落隔离、外部语义相关性和每样本颠倒的条件下的多视角图-文本检索路由，并验证了标签和属性的语义路由能力。

**💡 创新点**

提出了基于外部标签/属性与图形统计的因果对照实验（正确 vs 颠倒路由），以及严格的分段编码机制，首次系统证明了语义路由而非仅仅是架构通道化。

**🔧 技术方法**

采用对比学习（CLIP‑style）+外部软目标（RDKit 描述符、标签一致性），对每个视角使用独立的图与文本编码器，并在测试时使用外部邻居判定检索质量。

**📊 数据集**

使用 BBBP 与 BACE 两个分子数据集（各 480/60/60 训练/验证/测试）以及 1,200 个控制样本的分段文本。

**📈 对比分析**

与单视角专家模型比较：在三个视角（拓扑、标签、属性）上，一个联合模型在三种随机种子下的平均 nDCG@10 高于单独训练的专家；对标签和属性进行语义路由时，正确路由对比颠倒路由显著提升 nDCG，约 0.3–0.7；属性的改写增强在 OOD 改写上提升 0.14–0.15 的 nDCG，但一致性训练或硬改写在某些设置下略有损失。

**⚠️ 局限性**

局限性包括：需要强监督（显式段落与视角标注）；仅在两个数据集与三种种子下评估；缺乏专家审计；一致性训练在跨跑漂移检测未通过；未评估模型参数、训练时间、内存占用或下游预测性能。

---

## 125. Belief Coevolution in a Social Network of Generalist and Specialist Large Language Models

**arXiv ID:** 2607.27512 | [PDF](https://arxiv.org/pdf/2607.27512v1)

**作者:** Germans Savcisens `[一作]` (Northeastern University), Tina Eliassi-Rad `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在网络化多代理LLM环境下构建并实验了一个信念扩散框架，系统评估了领域专业化、社交角色和网络结构对LLM信念演化的影响。

**💡 创新点**

首次在LLM群体中将专业化、角色与网络结构三因素分离，证明专业化是推动群体共识变化的主要驱动力，并揭示个体与群体层面解释不一致的现象。

**🔧 技术方法**

采用同步Markov式交流协议，利用下一词概率提取信念；通过1,280次控制模拟、混合效应模型、ICC、对比分析以及层级代理模型（M1–M4）进行定量评估。

**📊 数据集**

使用15款8B参数LLM（1个通用模型+14个领域微调模型）与20条真实医学指示声明作为实验数据集。

**📈 对比分析**

在所有通用场景下，单一持久性模型即可解释大部分行为；在包含专业化的场景中，需要加入整体或局部信念分布和代理身份的模型，才能在最终一致度上达到约92%的保真度；在个体预测方面，最复杂模型M4在专家场景下可将MCC提升至约89%。

**⚠️ 局限性**

实验局限于单一医学领域、48个代理、10轮同步更新、无人工参与，且忽略异步对话、长期记忆、真实人类行为等因素。

---

## 126. Flock: Fast Proving for Batch Boolean Computations

**arXiv ID:** 2607.27491 | [PDF](https://arxiv.org/pdf/2607.27491v1)

**作者:** Benedikt Bünz `[一作]` (New York University), William Wang `[通讯]` (New York University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种面向标准哈希函数（SHA‑256、Keccak、BLAKE3）的哈希基 SNARK，专门针对大规模批量布尔 R1CS 计算进行高效证明。

**💡 创新点**

创新点包括：
• 将批量 R1CS 通过块对角化大幅降低 lincheck 复杂度；
• 引入 univariate skip 与压缩 lookup 表优化 zerocheck；
• 通过友好挑战、跳过 c、电路反向遍历等技术，进一步减少证明工作量；
• 针对 hash‑chain、Merkle 路径等 IO 关系提供简洁的多重 sumcheck 方案；
• 结合 Ligerito PCS 与 ring‑switching，在二进制域上实现极小的证明大小与验证时间。

**🔧 技术方法**

采用了多项技术：
• Spartan‑style PIOP、lincheck/zerocheck、univariate skip、压缩 lookup 表；
• 友好挑战 (fixed‐coordinate) 与多级指针化常数加速算术；
• 线性电路反向遍历实现 A₀、B₀ 的矩阵乘法；
• 通过 shift 约束将 hash‑chain 转化为单个 MLE 评估；
• Ligerito PCS + ring‑switching、列表解码模式；
• Rust 实现结合 AI 辅助编程。

**📊 数据集**

数据集：无外部数据，实验使用标准哈希函数的压缩/Permutation 函数本身（Keccak‑f[1600]、SHA‑256 compress、BLAKE3 compress）作为计算负载。

**📈 对比分析**

比较方法：在 Apple M4 Max (ARM) 单核和多核上测量每秒证明的哈希数量；与 Binius64、Plonky3、Hashcaster、Vega‑MC 进行基准对比；同时对比本机执行速率。
性能表现：
• 单核证明吞吐量仅为本机执行的 ~1.05×‑1.2×；
• 多核时可达 Binius64、Hashcaster、Plonky3 的 3–10 倍甚至更高；
• 证明大小 < 1 KB，验证时间 < 10 ms；
• 在 128‑bit 安全下实现最高吞吐率。

**⚠️ 局限性**

局限性：
• 仅支持批量布尔 R1CS，无法证明通用电路；
• 目前缺乏零知识实现；
• 依赖 ARM‑特定优化，其他体系结构未验证；
• 仍为研究原型，尚未投入生产；
• 只针对标准哈希函数，无法直接扩展至自定义加密原语。

---

## 127. MedLLM: An Open Medical Language Model at the Sub-Billion Scale

**arXiv ID:** 2607.27490 | [PDF](https://arxiv.org/pdf/2607.27490v1)

**作者:** Maxx Richard Rahman `[一作]` (German Research Center for Artificial Intelligence), Wolfgang Maass `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文构建了一个 0.1B 参数规模的医学语言模型 MedLLM，并提出了一种从通用网络文本中通过嵌入相似度检索选取医学内容的无标签医学语料 MedFineWeb，完成三阶段全开放式训练管道；

**💡 创新点**

创新点在于：① 在 0.1B 规模下实现可复现的医学 LLM；② 利用参考引导的相似度检索从通用文本中自动构建医学预训练语料；③ 发现压缩后医学能力按任务类型分裂，展示记忆召回受容量限制、上下文推理受适配影响；

**🔧 技术方法**

技术包括：通用预训练 + 课程化序列长度调度；在 MedFineWeb 上继续预训练；监督微调 (SFT) + 直接偏好优化 (DPO) 进行任务对齐；使用 RoPE、SwiGLU、RMSNorm 的 0.1B 解码器；嵌入相似度检索与 FAISS；z‑loss 正则；

**📊 数据集**

数据集涵盖通用文本（RedPajama‑V2、C4、OpenWebText、Wikipedia）作为 29.5B 训练语料；MedFineWeb 通过从通用文本中筛选得到约 3B 词；医学评测集为 PubMedQA、MedMCQA、MedQA、MMLU；参考集用于检索包括 MedMCQA、MedQA、PubMedQA；

**📈 对比分析**

比较方法：在预训练、SFT、DPO 三个阶段分别用 top‑token 选取对四个医学基准进行评测；与 0.1B‑7B 规模的开源模型（Gemma‑2B、BioMedLM‑2.7B、BioGPT‑Large‑1.5B、PMC‑LLaMA‑7B、Falcon‑7B、Mistral‑7B‑instruct、Zephyr‑7B‑β）对比；结果显示 MedLLM 在记忆召回任务（MedMCQA、MMLU）上取得 34.9% / 32.4% 的最高分，优于所有 7B 模型；在上下文推理任务 PubMedQA 上达到 58.2%，仅落后 2.9pp；在 MedQA 上仅略高 3.1pp；

**⚠️ 局限性**

局限性：0.1B 规模无法充分存储医学知识，导致对长篇临床推理（MedQA）和高容量召回任务表现平平；模型在预训练时的 perplexity 与最终准确率不一致，难以作为选择指标；未提供自由文本生成评测；在不同医学领域的泛化能力有限。

---

## 128. Schreier-Coset Graph Rewiring

**arXiv ID:** 2607.27479 | [PDF](https://arxiv.org/pdf/2607.27479v1)

**作者:** Aryan Mishra `[一作]` (University of Maryland), Lizhen Lin `[通讯]` (University of Maryland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于群论的Schreier-Coset图重排方法（SCGR），通过在输入图上添加低度常数连通的辅助图和谱映射耦合，显著缓解GNN中的over‑squashing。

**💡 创新点**

创新点在于：①使用SL(2,ℤ_n)的Schreier‑Coset图提供理论保证的谱间隙和有效电阻上界；②通过Fiedler向量谱对齐实现局部性保留；③设计分层通信层和可调耦合强度，兼顾全局信息流和稀疏性。

**🔧 技术方法**

技术手段包括：群论与Schreier‑Coset图构造、谱嵌入（Fiedler向量）、有效电阻与随机漫步理论分析、图神经网络（GCN、GIN等）的训练与评估、实验比较框架。

**📊 数据集**

实验使用的数据集包括：节点分类（Amazon Computers、Amazon Photo、CoAuthor CS、CiteSeer、Cora、PubMed）；图分类（TU数据集中的MUTAG、ENZYMES、PROTEINS、COLLAB、REDDIT-BINARY、IMDB-BINARY）；OGB（Molhiv、Molpcba）；Long‑Range Graph Benchmark（Peptides‑Func、Peptides‑Struct）；以及可控模块化的Stochastic Block Model（SBM）图。

**📈 对比分析**

与多种基线模型（LogReg、MLP、GAT、GCN、MoNET、LabelProp、GraphSage、GIN、以及多种重排方法）进行 20 次重复实验。SCGR 在大多数数据集上获得最高或接近最高的准确率，节点/图分类准确率提升 1–3%，有效电阻下降 15–40%，在低模块化 SBMs 上提升尤为显著。

**⚠️ 局限性**

局限性：①依赖 Fiedler 向量谱对齐，对谱噪声敏感；②在高模块化图上性能提升有限；③需要预先构造与输入图规模相近的Schreier‑Coset图，受限于群大小；④耦合强度 ε 采用固定值，缺乏自适应调节机制。

---

## 129. Recognition and Label-Free Adaptation Across Recording Sessions in Surface-EMG Gesture Decoding

**arXiv ID:** 2607.27568 | [PDF](https://arxiv.org/pdf/2607.27568v1)

**作者:** Jethro Odeyemi `[一作]` (University of Saskatchewan), W. J. Zhang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种面向多通道布局无关的卷积-Transformer编码器，用于表面肌电（sEMG）手势识别，并在不同日子、不同电极重装的跨会话设置中验证其稳健性。

**💡 创新点**

创新点在于将电极位置编码为几何坐标并在编码器中使用跨通道自注意力，同时通过卷积标记器共享权重实现电极数量可变；并在不调整模型的情况下展示其跨会话性能，并系统评估多种无标签适配方法。

**🔧 技术方法**

采用卷积标记器、位置编码、跨通道Transformer、注意力池化以及卷积自编码器的预训练；结合滚动时间归一化、批归一化及特征对齐等技术实现无标签适配。

**📊 数据集**

使用公开的NinaPro DB6 数据集，包含10名完整受试者、16电极、7个抓握姿势+静止，记录两天不同会话。

**📈 对比分析**

与传统的Hudgins时间域特征+LDA、CNN/T-CNN基线进行对比；跨会话宏F1值为0.688，显著高于LDA的0.540，且在无标签特征对齐后可恢复相当于单重复标注的性能。

**⚠️ 局限性**

主要限制包括仅在10名受试者上验证、仅针对同一数据集、对适配方法的探索有限，且对更大规模或多种肌电信号变异的鲁棒性尚未充分验证。

---

## 130. Subtract or Replay? Exact Deletion from Language-Model Memory

**arXiv ID:** 2607.27539 | [PDF](https://arxiv.org/pdf/2607.27539v1)

**作者:** Vishwajith Ramesh `[一作]` `[通讯]` (Vy Labs, Inc.), Vishwajith Ramesh (Vy Labs, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实现了两种基于内存表示的精确删除方案：在Gemma模型中通过支持向量门控实现可减法删除；在Kimi模型中通过检查点重放实现不可寻址记录的重建删除，并在两者上进行数值与行为审计。

**💡 创新点**

创新点在于提出“可寻址”与“不可寻址”记录影响的判定标准，并相应给出两类删除方法：可减法递减和重放重建，并首次给出对比实验和KL证书验证，展示了不同记忆表示对删除成本与精度的根本影响。

**🔧 技术方法**

使用的技术包括支持向量门控（Support‑Vector Memory）、低秩LoRA恢复、浮点64递减实现精确删除、检查点重放、以及对抗攻击评估框架（TOFU、MUSE、WMDP、LiRA、ICUL）。

**📊 数据集**

实验数据集包括公开文本语料（WikiText‑103、C4、Lambada、FineWeb‑Edu）、TOFU虚拟事实数据以及MIMIC‑IV‑Ext‑CDS与MIMIC‑IV‑Note临床记录。

**📈 对比分析**

通过与原始Gemma 1B/4B/12B模型、对齐控制模型以及三种删除策略（递减、衰减、重放）进行对比；在1B规模下递减方法在KL≈10⁻¹⁴的情况下保持几乎无行为差异；在4B/12B规模下递减精度下降，效用开销分别升至约11.2%和44.3%；重放方法则在所有规模下保证完全一致，但成本随后缀长度线性增长。

**⚠️ 局限性**

局限性包括：1) 规模受限，仅在1B规模达到最佳精度与低成本；2) 只对内存状态实现删除，未覆盖权重或先前输出；3) 支持向量门控计算成本高；4) 检查点重放需要完整状态检查点，且成本随后缀长度增长；5) 实验仅在单一硬件与模型族上验证，未测试跨硬件或更大模型的泛化。

---

## 131. Latent-Kernel Discrete Flow Maps for Few-Step Generation

**arXiv ID:** 2607.27529 | [PDF](https://arxiv.org/pdf/2607.27529v1)

**作者:** Mansoor Ahmed `[一作]` (Georgia State University), Murray Patterson `[通讯]` (Georgia State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种 Latent‑Kernel Discrete Flow Maps (LKF) 模型，通过在每一步使用共享潜在变量混合 M 个因式化组件，以在离散扩散过程中实现位置相关性，并实现教师无关的少步文本生成。

**💡 创新点**

创新点在于将因式化分布与共享潜在变量结合，形成低秩混合的流图，既能在单步内表达多位置相关，又不依赖教师模型，突破传统因式化模型在少步生成中的独立性瓶颈。

**🔧 技术方法**

采用离散流匹配、Transformer 结构、两时刻条件训练、混合路由器正则化、最佳‑M 采样策略，以及信息理论分析来度量总相关性。

**📊 数据集**

在 One‑Billion‑Word (LM1B) 与 WikiText‑103 两大无监督文本生成基准上进行实验。

**📈 对比分析**

与 MDLM、SEDD、SDTT、Di4C、ReDi 等因式化与教师蒸馏/校正的少步采样器以及自回归模型对比，LKF 在 32 次 NFE 下实现 3.3× 的生成困惑度提升，且在最佳‑M 策略下超越蒸馏和校正模型，无需教师。

**⚠️ 局限性**

单步因式化的上限仍导致无法完全捕获单个不一致 token 的误差；路由器在采样过程中信息不足，难以实时判断最佳组件，限制了最佳‑M 的进一步压缩。

---

## 132. Failure Detection for Surgical Robot Imitation Policies via Flow-Matching World Modeling

**arXiv ID:** 2607.27511 | [PDF](https://arxiv.org/pdf/2607.27511v1)

**作者:** Zhefeng Huang `[一作]` (Georgia Institute of Technology), Yue Chen `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 FoMo-FD，一种基于流匹配视觉世界模型的窗口级故障检测方法，可仅利用成功演示数据在外科机器人操作中实现实时异常监测。

**💡 创新点**

创新点在于：①在潜在视觉空间学习动作条件端点流匹配模型；②使用逆向流非一致性评分而非预测误差；③通过合成 conformal 校准在无失效演示的情况下获得任务特定阈值。

**🔧 技术方法**

采用 DINOv2 + VAE 进行视觉潜在编码，基于流匹配（Flow Matching）的动作条件世界模型，逆向 ODE 评分机制，合成 conformal calibration，以及 Action Chunking Transformer 策略。

**📊 数据集**

在四个外科相关任务上进行评估：模拟针拾取、环放置（NVIDIA Isaac Sim SuFIA-BC），真实组织牵拉、导管插入（da Vinci Research Kit），共使用 20 种失败模式与大量成功演示。

**📈 对比分析**

与预测误差、观测级异常检测（logpZO、RND）等基线对比，FoMo-FD 在手腕视角下实现 96.6% 的失败检测率（FDR）与仅 1.3% 的误报率（FAR），显著优于其他方法。

**⚠️ 局限性**

局限性包括：仍需少量成功 rollouts 进行 conformal 校准；采用 episode‑level 固定阈值，未适应时间变化；仅关注短窗口一致性，可能对长期错误的敏感度不足。

---

## 133. Sparsity Induced Identifiability in Matrix Tri-Factorisation

**arXiv ID:** 2607.27507 | [PDF](https://arxiv.org/pdf/2607.27507v1)

**作者:** Tingting Mu `[一作]` `[通讯]` (University of Manchester), Tingting Mu (University of Manchester)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究了稀疏性在一般实值矩阵三分解中的可辨识性，并给出了严谨的恢复理论与误差上界。

**💡 创新点**

首次将三分解转化为两个耦合的二分解，从而在稀疏性诱导下实现可辨识性分析；同时提出了基于稀疏编码与谱字典近似的算法框架。

**🔧 技术方法**

利用字典学习、Lasso与交替最小化、谱字典近似、压缩感知理论以及概率集中不等式进行分析与算法设计。

**📊 数据集**

通过蒙特卡罗实验生成随机稀疏系数矩阵和随机S矩阵作为数据集来验证理论。

**📈 对比分析**

将实验误差与理论上界进行对比，实验显示误差随稀疏程度增加而减小，收敛速度加快，实验结果与理论吻合良好。

**⚠️ 局限性**

对稀疏程度和字典初始误差的要求较高，对非负或特殊结构的三分解尚无完整理论，且算法对初始化较为敏感，计算复杂度未给出详细分析。

---

## 134. Latent States in Neural Networks: Recovering the Temporal Structure of Drifting Data from Model Weights

**arXiv ID:** 2607.27482 | [PDF](https://arxiv.org/pdf/2607.27482v1)

**作者:** Kevin Guan `[一作]` `[通讯]` (Princeton University), Kevin Guan (Princeton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究者通过在时间窗口上训练一系列分类器并提取其权重，利用左向右HMM对权重轨迹进行分段，探讨是否能从模型权重中恢复数据流的离散阶段，并验证这些阶段是否能预测模型在不同时间段的泛化表现。

**💡 创新点**

创新点在于：①首次将隐藏马尔可夫模型应用于人工神经网络权重序列，以无监督方式识别数据漂移中的事件边界；②通过对比同阶段与跨阶段的泛化差异，提供了模型对时序漂移阶段的功能性验证；③发现权重驱动的阶段与数据标签分布的变动高度相关，证明模型权重能够捕获数据分布演化。

**🔧 技术方法**

使用技术包括：多模态MLP（结合RoBERTa与ResNet-50）与文本MLP、权重对齐（Hungarian匹配）以消除隐藏层的置换对称性、PCA降维、左向右约束的HMM（Gaussian发射、Baum‑Welch训练、Viterbi解码）、中心化F1评估、基于时间差的置换检验、偏差控制的线性回归与Freedman‑Lane偏差测试。

**📊 数据集**

使用的数据集为：①Fakeddit（多模态谣言检测，6类标签，约9,495样本/窗口，35个窗口，2013‑2018）；②Yelp（文本情感评分，5类标签，约12,801样本/窗口，56个窗口，2008‑2022）。

**📈 对比分析**

通过比较HMM划分与等大小切分的“同状态”与“跨状态”窗口对的中心化F1差距，评估分段的有效性。结果显示：Fakeddit同状态差距≈0.1092，Yelp≈0.0184；两者均显著优于等大小划分（p<0.0001），且在控制时间距离后仍保持优势；权重与标签分布差距的相关性也在统计学上显著。

**⚠️ 局限性**

局限性包括：①仅分析非累积窗口的独立训练模型，未考虑模型在持续学习中的累积效应；②PCA仅捕获约11%权重方差，可能遗漏重要信息；③权重对齐无法消除所有可重参数化对称性，影响空间距离测量；④左向右HMM不允许状态重复，可能忽略季节性或循环漂移；⑤实验仅验证了标签分布漂移，未充分探讨仅特征分布变化的场景。

---

## 135. What makes prompts a graph: necessary and sufficient conditions for prompt graph engineering

**arXiv ID:** 2607.27578 | [PDF](https://arxiv.org/pdf/2607.27578v1)

**作者:** Sandeco Macedo `[一作]` `[通讯]` (Federal Institute of Goiás), Sandeco Macedo (Federal Institute of Goiás)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出“提示图工程（prompt graph engineering）”的概念化定义，并给出四项必要条件（显式结构、结构与内容分离、可执行语义、首类工程对象）以及基于此的包含/排除测试。

**💡 创新点**

创新点在于：①把提示工程从单字符串扩展到可视化、可调度的图结构；②厘清该领域与相关概念（思维拓扑、代理编排、工作流引擎等）的边界；③提供可操作的评估准则与系统案例验证，奠定统一词汇和研究基础。

**🔧 技术方法**

主要技术手段为：概念分析与历史迁移梳理、构造性定义方法、系统案例分析（LangGraph、DSPy、Prompt Flow 等）及其对四条件的逐项验证。

**📊 数据集**

本研究为概念性工作，未使用公开数据集或实验数据；案例分析依据公开文档与已有论文的描述完成。

**📈 对比分析**

比较方法为对六个实际系统进行 T1–T4 四项测试，判定是否符合提示图工程的四条件；结果表明 LangGraph、DSPy、Prompt Flow 满足全部条件，AutoGen、CrewAI 在某些模式下满足，Claude Code 子代理不满足。性能指标未涉及。

**⚠️ 局限性**

局限性包括：①缺乏多评审者验证与交叉检查，导致分类结果依赖单一分析者判断；②所列系统为 2026‑07 期望快照，后续更新可能改变分类；③未涉及动态生成图结构的完整评估，且对自动优化等后续研究尚未给出经验性支持。

---

## 136. Policy Gradient Steering: Interventions from Behavioral Objectives

**arXiv ID:** 2607.27574 | [PDF](https://arxiv.org/pdf/2607.27574v1)

**作者:** Yoann Poupart `[一作]` (Sorbonne University), Nicolas Maudet `[通讯]` (Sorbonne University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为 Policy Gradient Steering (PGS) 的方法，利用策略梯度把有限的行为反馈转换为可移除的激活向量，从而在推理时动态改变已训练模型的行为；并在两路网格世界、国际象棋谜题与 Google Research Football 环境中对该方法进行验证。

**💡 创新点**

创新点在于：① 将行为奖励的累计返回直接映射到激活层梯度，从而构造可调、可组合、可删除的激活向量；② 通过 Fisher 信息矩阵对向量进行校准，确保在给定 KL 损失预算下的可控改动；③ 解决传统对比式激活调节在特定决策点失效的问题，展示了基于梯度的“行动信用”更能精准定位决策。

**🔧 技术方法**

核心技术包括 REINFORCE（策略梯度）与重要性采样、Fisher 信息矩阵近似、对比式激活调节 (CAA、COAST、K‑Steer) 的对比实验、LoRA、ReFT、微调（Fine‑tuning）以及多代理学习评估框架。

**📊 数据集**

使用的数据集包括：① 简单的两路网格世界；② Lichess 公共棋局库中的国际象棋谜题（取决于三种战术：fork、pin、skewer）；③ Google Research Football 环境中的多队策略与比赛记录。

**📈 对比分析**

方法对比：在网格世界中 PGS 能成功实现路劲偏好，而传统激活调节失效；在国际象棋中，PGS 在单一目标上取得最高得分提升，合成目标的表现与 LoRA、微调相当但参数占用最少；在足球中，PGS 在传球行为上实现显著提升，并能跨对手迁移。整体来看，PGS 在保持低存储、低推理成本的同时，性能优于或可与现有方法竞争。

**⚠️ 局限性**

局限性包括：① 仅为局部适应，Fisher 近似在大步长时可能不准确；② 激活层位置选择对效果影响大，需要经验；③ 目前仅对已冻结策略有效，未针对在线自适应场景；④ 离线重要性采样在状态分布不匹配时为近似，可能导致偏差；⑤ 对多任务或跨领域解释性仍需进一步研究。

---

## 137. A Montage-Agnostic Encoder for Calibration-Light Cross-User Gesture Recognition from Surface Electromyography

**arXiv ID:** 2607.27565 | [PDF](https://arxiv.org/pdf/2607.27565v1)

**作者:** Jethro Odeyemi `[一作]` (University of Saskatchewan), W. J. Zhang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

设计并训练了一种对电极蒙太奇不敏感的编码器，用于跨用户的表面肌电手势识别，并通过极少量校准（1到3次）实现对未知用户的快速适配。

**💡 创新点**

创新点包括：① 用物理坐标而非通道索引对每个电极做token化，使同一网络可处理不同通道数；② 引入因果滚动时域归一化、空间坐标编码和跨通道注意力，三者单独去除都会导致超过一半的性能下降；③ 在跨用户设置下，首次证明充分训练后该编码器可超过传统的 per‑user LDA 基线。

**🔧 技术方法**

采用了卷积+Transformer 架构：共享的1D卷积提取时序特征，位置编码后交叉通道注意力聚合，再通过注意力池化得到窗口嵌入；使用AdamW优化器和一周期学习率调度；还尝试了自监督预训练（波形重建、RMS包络）但未提升。

**📊 数据集**

主要使用公开的表面肌电数据库：NinaPro DB1（10通道，27人，53姿势）、DB2（12通道，40人，50姿势）、DB5（16通道，10人，53姿势）以及EMG‑EPN612（8通道，612人，6姿势）。

**📈 对比分析**

与每个被试独立训练的 LDA 线性判别器（使用 Hudgins 特征）做留一用户交叉验证。结果显示：在 DB1 上 3 次校准的宏 F1 0.827，LDA 0.593；DB2 0.965 对比 0.857；DB5 0.586 对比 0.802；在 DB1 上所有 27 名被试在 3 次校准时均超越 LDA。性能提升受训练预算和被试数的稳定性阈值控制，超过约 10 名被试后提升趋于平稳。

**⚠️ 局限性**

局限性：① 仅进行离线评估，未涉及实时可用性指标；② EMG‑EPN612 的结果不确定，因采用全录音窗口而非手势分段；③ 电极的物理坐标取值近似，可能影响位置编码精度。

---

## 138. Memory Efficient Tabular Foundation Models

**arXiv ID:** 2607.27546 | [PDF](https://arxiv.org/pdf/2607.27546v1)

**作者:** Shuting Luo `[一作]` (Commonwealth Bank of Australia), Simon Lucey `[通讯]` (University of Adelaide)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了Tabular Foundation Models 的内存压缩

**💡 创新点**

展示了 INT4 量化可实现高达 7.6× 的压缩率且性能几乎不变的创新

**🔧 技术方法**

采用了 K‑Means、AWQ、GPTQ 三种 INT4 后训练量化方法

**📊 数据集**

在 30 个 OpenML 公开表格数据集上进行实验

**📈 对比分析**

与 XGBoost、CatBoost、LightGBM、KNN、Random Forest 等传统基线对比，量化模型保持或超过最佳基线并实现 6.0–7.6× 的压缩率

**⚠️ 局限性**

仅使用后训练量化，未探索量化感知训练、激活量化或硬件特定优化，导致推理时内存未得到进一步降低

---

## 139. ThreatForest: Multi-Agent Attack Tree Generation with Pluggable TTP Framework Mapping

**arXiv ID:** 2607.27528 | [PDF](https://arxiv.org/pdf/2607.27528v1)

**作者:** Cristian Leo `[一作]` (Amazon Web Services Inc), Prakash Jha `[通讯]` (Amazon Web Services Inc)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过多代理流水线，将源代码仓库自动转换为结构化攻击树，并将每个叶节点映射到 MITRE ATT&CK 等威胁框架，同时生成与应用上下文相关的可执行缓解措施。

**💡 创新点**

创新点在于：①将威胁建模分解为十个可验证的代理步骤，并通过 deterministic verifier 与 HITL 门实现可审计、可恢复的流程；②实现可插拔的 TTP 框架映射（MITRE ATT&CK、CAPEC、云特定矩阵）而不需重新训练模型；③构建统一的评估框架和 16 维量表，提供客观可复现的性能基准。

**🔧 技术方法**

技术栈包括：LLM（Claude Sonnet 4.5）驱动的多代理系统 (Strands Graph)、句子变换检索 (ATTACK‑BERT) 进行 TTP 匹配、cosine 相似度阈值过滤、Deterministic Python 验证器、HITL 交互接口、Langfuse 追踪日志、概率传播模型用于路径成功率计算。

**📊 数据集**

使用了 7 个多样化云原生应用的源码仓库（IoT 制造、身份联邦、生成式 AI、医疗分析、IAM 治理、会议转录、旅行预订），以及 MITRE ATT&CK STIX 数据集、CAPEC、云威胁矩阵等公开知识库。

**📈 对比分析**

对比方法：采用 16 维评估量表、LLM 评审+对抗式验证、SME 校验；结果显示威胁/攻击树/缓解得分平均 0.63‑0.68，TTP 映射仅 0.29；与单一 LLM 调用基线比较，TTP 映射提升至 0.63，覆盖率提高（每应用 240 步，89 技术，ATT&CK 0.98 覆盖），但成本约 $7.8/应用。

**⚠️ 局限性**

主要限制在于 TTP 映射的嵌入检索精度（ATTACK‑BERT 仅 29% 正确），缺乏针对攻击步骤的域特定 encoder；在更大或多云系统中可扩展性未知；系统高度依赖单一 LLM（Claude Sonnet）；ATT&CK 本身在云/容器/无服务器场景中覆盖不足。

---

## 140. A dataset of rated conceptual arguments

**arXiv ID:** 2607.27499 | [PDF](https://arxiv.org/pdf/2607.27499v1)

**作者:** Emery Cooper `[一作]`, Ethan Perez `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个包含哲学立场、批评与专家多维度评分的数据集，并评估多种大型语言模型在此任务上的表现。

**💡 创新点**

创新点在于将概念推理转化为可评价的“论证质评”，提出多维度打分标准（中心性、强度、正确性、清晰度、无关重量、单一议题、整体评分）及相应的排名与点误差评价指标。

**🔧 技术方法**

技术上使用人类专家手工评分、LLM生成批评、基于对比和绝对误差的加权损失函数、以及对“思考/推理”提示的实验比较。

**📊 数据集**

数据集来源多样：书籍改编（168条）、课程作业（41条）、博客/论坛（108条）、模型写作（68条）等，共约400条立场，数千条专家评分的批评。

**📈 对比分析**

通过加权对比排名误差率和自定义加权误差指标对模型进行比较，结果显示更强的基础模型往往获得更低误差；“思考”提示对评分影响有限。

**⚠️ 局限性**

局限包括：批评的分布差异（模型生成批评往往较弱）、许多立场仅有单一或低质量批评、评分主观性、以及在概念推理上“思考”训练的可迁移性不足。

---

## 141. Certified Sequential Sweep Without Unrolling

**arXiv ID:** 2607.27498 | [PDF](https://arxiv.org/pdf/2607.27498v1)

**作者:** Tobias Seufert `[一作]` (University of Freiburg), Christoph Scholl `[通讯]` (University of Freiburg)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于IC3的序列等价性验证与认证方法，能够处理在重定时和后续逻辑重构（包括组合和顺序综合）后的电路；

**💡 创新点**

创新点在于：①首次为重定时+重构的序列等价性提供完整可认证的证明；②使用仿真猜测的信号对应关系与虚拟前向重定时（Virtual Forward Retiming）生成初始猜想性不变式；③通过证明性重定时预处理将已认证的不变式映射回原始电路，避免了对不受认证的重定时步骤的依赖；

**🔧 技术方法**

技术手段包括：IC3模型检查框架、序列仿真以构造等价类、虚拟前向重定时增强等价候选、重定时不变式转换（retiming invariants）、证书生成与Certifaiger校验；

**📊 数据集**

实验数据集涵盖ISCAS'89、ITC'99、IWLS'05等开源核以及HWMCC竞赛和6s22-SEC等顺序等价性基准；

**📈 对比分析**

与rIC3（最新硬件模型检查竞赛获奖者）以及ABC的非认证等价性检查器对比，发现本方法在多数实例上性能显著优于rIC3且与ABC竞争，尤其在重定时+组合重构和强顺序重构场景下；认证证书的验证开销极小，绝大多数在10秒以内；

**⚠️ 局限性**

局限性：仍为原型实现，对极大规模设计的可扩展性待进一步评估；对复杂重构导致的不定式子可能需要更多手工调参；以及在某些基准上仍不如rIC3或ABC的最佳性能；

---

## 142. Skill Use or Skill Theater? Evaluating the Reasoning Backroom in Skill-Augmented Language Agents

**arXiv ID:** 2607.27484 | [PDF](https://arxiv.org/pdf/2607.27484v1)

**作者:** Jinwei Hu `[一作]` (University of Liverpool), Xiaowei Huang `[通讯]` (University of Liverpool)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估可重用技能在单体与多体语言模型中的因果影响，并揭示“Reasoning Backroom”现象。

**💡 创新点**

提出一种无模型内部访问的因果技能归因框架（SkillBackroom），演示技能身份、内容与实际决策之间的脱节。

**🔧 技术方法**

通过对技能的存在、措辞、名称、内容和分配进行干预，并在答案提交后进行后决策归因查询。

**📊 数据集**

使用自制的300条逻辑推理题和283道自然数学题数据集进行评估。

**📈 对比分析**

对比每个技能条件与无技能基线的答案变化、准确率、归因一致率（AFS）等指标；结果显示归因一致率普遍低于0.5，说明技能影响与声明不符。

**⚠️ 局限性**

仅评估冻结文本技能，受限于两类任务、固定模型、无长周期自适应；未覆盖内部机制等更深层次的解释。

---

## 143. FedOGL: Combating Catastrophic Forgetting in Federated Open-World Multimodal Graph Learning

**arXiv ID:** 2607.27665 | [PDF](https://arxiv.org/pdf/2607.27665v1)

**作者:** Zekai Chen `[一作]` (Beijing Institute of Technology), Rong-Hua Li `[通讯]` (Beijing Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种联邦开世界多模态图学习框架FedOGL，解决分布式图数据在类增量演进中的灾难性遗忘问题。

**💡 创新点**

创新点在于将语义与结构记忆统一保护，融合回放+教师蒸馏、全局结构基子空间投影以及原型导向的分布式记忆整合，从而同时抑制模态语义覆盖、拓扑结构侵蚀与联邦记忆碎片化。

**🔧 技术方法**

核心技术包括：多模态图神经网络；任务起始教师与回放重放；结构基子空间投影与梯度投影；类别原型生成、匹配与合并；以及基于原型图的全局模型微调。

**📊 数据集**

实验使用六个公开数据集：节点分类的Grocery、RedditS、Ele-Fashion；跨模态检索的KU、QB、Bili-Cartoon。

**📈 对比分析**

与FL（FedAvg、FedMVP、PRISM）、FGL（FedSSP、FedIIH）以及开世界方法（POWER、TopoOOD、GRASP、CLIPN）对比，FedOGL在所有数据集上均获得最佳或第二佳的平均表现（AM），并在灾难性遗忘（FM）上降低42.67%，在未知拒绝率（FPR_95）上降低28.31%，同时保持或提升下游任务性能。

**⚠️ 局限性**

局限性包括：通信开销相对较高（优于轻量级基线但低于POWER）；额外的记忆操作导致运行时略有提升；在极大规模联邦（客户端数目非常多）和强非IID条件下仍需进一步验证其可扩展性和鲁棒性。

---

## 144. Strategy Phasing of Cyber Attacks on Digital Substations

**arXiv ID:** 2607.27661 | [PDF](https://arxiv.org/pdf/2607.27661v1)

**作者:** Akila Herath `[一作]` (Virginia Tech), Kuchan Park `[通讯]` (University of Michigan-Dearborn)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出SubCASP方法，利用隐藏马尔可夫模型对 IEC 61850 数字变电站的 IDS 日志进行攻击阶段推断，实时预测当前与下一个攻击阶段并回溯完整攻击路径。

**💡 创新点**

将 ATT&CK 攻击图与 HMM 结合，实现多源 IDS 日志的攻击阶段语义化；同时使用前向算法实时推断与 Viterbi 算法回溯，兼顾实时与历史解析，且在可观测度低和日志缺失场景下保持鲁棒性。

**🔧 技术方法**

隐藏马尔可夫模型、前向算法（FA）、Viterbi 算法、攻击图建模、深度优先搜索生成路径、拉普拉斯平滑估计 HMM 参数。

**📊 数据集**

基于 ATT&CK 攻击图生成的可复现攻击路径数据集，覆盖不同 IED 数量、IDS 可观测度（低/中/高）及 10%/20%/30% 日志缺失比例的实验。

**📈 对比分析**

通过 5 折交叉验证与 Viterbi 完整序列解码对比；在高可观测度下 FA 的当前阶段预测与 Viterbi 接近，平均每阶段准确率>96%；在 30% 日志缺失时 Viterbi 仍保持>90%准确率，FA 与 Viterbi 的最大误差从 6% 提升至 11%。

**⚠️ 局限性**

仅假设固定攻击阶段持续时间，未考虑不同持续时间；缺失日志时 FA 的实时预测精度下降；方法聚焦于 CB 操作失效，未扩展到其他子站功能或更广泛的攻击影响。

---

## 145. Evaluation Protocols and Cross-Subject Generalization in EEG Emotion Recognition

**arXiv ID:** 2607.27655 | [PDF](https://arxiv.org/pdf/2607.27655v1)

**作者:** Hanting Suo `[一作]` (Southeast University), Yuwen Li `[通讯]` (Southeast University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了三层 EEG 情感识别评估协议记录，并在 SEED 与 SEED‑IV 数据集上对归档 DGCNN 进行严格的实验验证，包括检查点选择敏感性、协议匹配一致性和跨受试者泛化。

**💡 创新点**

将目标估计、开发过程和报告规则拆分为独立层级，并将目标数据访问、检查点选择和预测单元与不确定性条件明确绑定，形成可审计的评估清单；同时给出跨受试者和跨会话的性能差异分析。

**🔧 技术方法**

使用图神经网络 DGCNN、差分熵与频带特征、交叉熵损失、AdamW 优化、验证集宏 F1 选择检查点，以及统计区间计算（BCa 置信区间、bootstrap 等）。

**📊 数据集**

SEED、SEED‑IV 两个标准情感 EEG 数据集（各15名受试者，多个会话）以及一份竞赛 60 人（HC/DEP）数据集。

**📈 对比分析**

通过在同一轨迹上比较检索到的检查点与固定终点的窗口准确率，发现平均提升 0.1036；在无目标数据访问的五折受试者互斥评估中，DGCNN 训练集准确率接近 1 但未见者准确率仅为 0.535（SEED）和 0.395（SEED‑IV），显示显著泛化差距。

**⚠️ 局限性**

实验仅评估单一归档模型，受试者样本量有限，跨会话与跨受试者差异无法完全归因于特定因素；尾部风险混合器在不同特征空间下不具有可比性，且未对全流程重新抽样或多模型重训练做更广泛检验。

---

## 146. 4DHumanDiff: Direct Text-to-4DGS Generation for Consistent 360-Degree Dynamic Humans

**arXiv ID:** 2607.27634 | [PDF](https://arxiv.org/pdf/2607.27634v1)

**作者:** Renlong Wu `[一作]` (Harbin Institute of Technology), Hui Li `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了4DHumanDiff，一种直接从文本提示生成360°动态人类资产的扩散框架，避免了先生成视频再重建的两阶段流程。

**💡 创新点**

创新点在于将动态人类生成直接映射到4D高斯斑点（4DGS）表示空间，并通过统一的3D U‑Net + 时序注意力实现端到端的时空一致性。

**🔧 技术方法**

核心技术包括：4D Gaussian Splatting、基于3D U‑Net的扩散模型、时序自注意力、2D正则化（MIP/ VGG）、训练无关的4D插值与时空体素绑定。

**📊 数据集**

使用了基于MVHumanNet的60,000对高质量文本–4DGS数据集，统一规范化后训练。

**📈 对比分析**

与现有的两阶段视频+重建方法（如Wan 2.1+GauHuman、L4GM、GenXD、SV4D、4Diffusion）对比，4DHumanDiff在运动平滑度、主体一致性、多视角一致性等指标上均名列前茅，并将整体生成时间缩短10×以上。

**⚠️ 局限性**

局限性包括缺乏细粒度局部细节增强，尤其在关节部位的几何与纹理精细化方面仍有提升空间。

---

## 147. ReDiPPO: Reference-Guided Value Calibration and Discrepancy-Aware Token Reweighting for Mathematical Reasoning

**arXiv ID:** 2607.27631 | [PDF](https://arxiv.org/pdf/2607.27631v1)

**作者:** Zhenrong Zhang `[一作]` (IFLYTEK Research), Si Wei `[通讯]` (IFLYTEK Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ReDiPPO，利用训练时的参考答案作为价值函数的特权信息，并通过标准与参考引导价值函数的差异实现 token 级加权，从而改进数学推理的 PPO 学习。

**💡 创新点**

创新点在于：1）在价值函数中注入参考答案的特权信息；2）用标准 critic 与 reference‑guided critic 的差异作为动态权重，提升 token‑级信用分配的可靠性。

**🔧 技术方法**

技术实现基于 Proximal Policy Optimization，配备双重 critic（参考引导与标准）以及 discrepancy‑aware 加权机制，并采用 GAE、clipped surrogate loss 等常见 RL 技术。

**📊 数据集**

使用六大数学推理基准（AIME 2024‑26、HMMT 2025、Minerva Math、OlympiadBench）以及 DAPO‑17K 与 DeepMath‑103K 训练集。

**📈 对比分析**

与 DAPO、GSPO 以及相同预训练的 vanilla PPO 在三种模型（Qwen3‑4B‑Instruct、Qwen3‑4B‑Thinking、OLMo3‑7B‑Instruct‑DPO）上对比，ReDiPPO 在所有基准上平均提升 1.19‑2.37%，成为最佳方法。

**⚠️ 局限性**

局限性是该方法依赖可验证的参考答案，难以直接迁移到无唯一答案或主观评价的推理任务。

---

## 148. SCOPE: Synthetic Conditional Objectives for Policy Evolution in Black-Box Combinatorial Optimization

**arXiv ID:** 2607.27630 | [PDF](https://arxiv.org/pdf/2607.27630v1)

**作者:** Nguyen Viet Tuan Kiet `[一作]` (Hanoi University of Science and Techonology), Huynh Thi Thanh Binh `[通讯]` (Hanoi University of Science and Techonology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SCOPE框架，利用LLM根据已收集的黑盒评估结果生成可执行的合成条件目标，以此引导搜索策略在有限评估预算下寻找高质量组合解。

**💡 创新点**

创新点在于将目标设计转化为自适应的程序图演化过程：通过对比证据修正父目标、行为新颖性度量筛选新目标、以及多目标合成与动态分配查询，实现在多种搜索后端下的系统性提升。

**🔧 技术方法**

核心技术包括：LLM（gpt‑4o‑mini、gpt‑5‑nano）用于生成合成目标；程序图（节点为目标，边为演化关系）；对比证据提取与行为新颖性度量；强化学习（Thompson采样）分配查询；基于GLS、ALNS及构造策略的搜索后端。

**📊 数据集**

实验使用15+种组合优化基准（如PeakRoute、RiskTour、FleetSpan、LoadFlow、WinnerCats、Influence Maximization等），实例规模 n=100/200，分别划分训练、验证、测试集，并在灰盒/多任务场景下复现。

**📈 对比分析**

与EoH、ReEvo、FunSearch、MCTS‑AHD、HiFo、Eureka等方法以及白盒限预算/完整预算进行对比。SCOPE在所有搜索引擎、语言模型、问题规模、灰盒和多任务设置中均排名第一，并在大部分任务上显著降低相对Gap，甚至在限预算下超过白盒完整预算的性能。

**⚠️ 局限性**

局限性：仍需依赖LLM调用，受模型规模与算力限制；在灰盒或高噪声问题中性能下降；仅在有限评估预算下优化；对长程依赖与随机性处理不充分；图结构维护与对比证据提取带来额外复杂度。

---

## 149. MPIE-Bench: Benchmarking Anatomically Plausible Multi-Person Interaction Editing

**arXiv ID:** 2607.27616 | [PDF](https://arxiv.org/pdf/2607.27616v1)

**作者:** Jiajia Lin `[一作]` (University of Science and Technology of China), Hongtao Xie `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了多人物交互编辑评估基准MPIE-Bench与评估协议MPIE-Eval，针对身体连贯性问题进行量化评估。

**💡 创新点**

创新点在于：①以真实视频挖掘“低接触→高接触”编辑样本，构造无复制的测试集；②使用冻结的3D人体网格重建（Multi‑HMR）从几何角度独立评估Anatomy与Interaction两个轴；③公开完整评估流程、阈值与阈值混合权重，确保可复现与可审计。

**🔧 技术方法**

技术包括：视频挖掘与接触密度曲线筛选、逆向写作指令、冻结的多人体网格重建、多尺度几何测度（凸包穿透、表面间距）以及软阈值/混合评分。

**📊 数据集**

数据集来源：Pexels、Harmony4D、CHI3D等公开视频，经过筛选得到2,500个交互三元组（405场景、14种交互类别、4级接触密度）。

**📈 对比分析**

与现有基准（MultiHuman‑Testbench、InsHuman、BodyMetric等）对比，MPIE-Eval的Anatomy/Interaction轴提供更细粒度评分；在10个编辑器中，闭源模型在VLM评估接近满分，而Mesh评估仍保留动态范围，凸显评估差异。

**⚠️ 局限性**

局限：仅评估基于多人体网格的几何连贯性，未考虑局部接触点细节；依赖单一Multi‑HMR前端，未来可引入更强的重建器；评估仍基于静态图像，无法覆盖动态交互。

---

## 150. CORE: In-Context Reconstruction for Unified Tabular Anomaly Detection

**arXiv ID:** 2607.27615 | [PDF](https://arxiv.org/pdf/2607.27615v1)

**作者:** Yunfeng Zhao `[一作]` (Guangxi University), Shirui Pan `[通讯]` (Griffith University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种统一的表格异常检测框架，通过装饰化特征对齐和基于上下文的重构来实现跨数据集的无训练迁移。

**💡 创新点**

创新点：1）装饰化特征对齐模块可直接从原始特征中挑选信息量大且不冗余的维度，保持语义；2）基于上下文的重构模块使用最近邻正常样本进行加权门控重构，避免了二分类或合成异常的局限。

**🔧 技术方法**

使用多头编码器（两层MLP）、相关性加权检索、门控融合和均方误差重构评分等深度学习技术。

**📊 数据集**

在7个源数据集（航天、生命、医疗、图像、物理化学、社会学等）上预训练，34个目标数据集（内部域与外部域）上进行评估。

**📈 对比分析**

与10个基线方法（IForest、LOF、KNN、AE、DSVDD、LUNAR、MCM、DRL、DisentAD、OFA‑TAD）比较，平均AUROC提升至0.8488，优于OFA‑TAD 1.43个百分点；在跨域测试中保持领先，并在运行时最短、可扩展性强。

**⚠️ 局限性**

局限性：对上下文样本比例敏感，极高维或稀疏数据可能仍受限；目前仅验证于表格数据，未验证对其他模态的适用性。

---

## 151. Wiring diagram extraction and gluing: a case study in classifying figure skating jumps using 3D dataset

**arXiv ID:** 2607.27598 | [PDF](https://arxiv.org/pdf/2607.27598v1)

**作者:** Jason Lo `[一作]` (California State University), Mohammadnima Jafari `[通讯]` (California State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并验证了在图形化序列数据分析中对 Hasse 聚类算法的迭代应用与图形拼接（gluing）理论，用于重构完整的 wiring diagram（线图）。

**💡 创新点**

创新点在于（1）构建了可迭代的 Hasse 聚类框架，显著降低组合复杂度；（2）提出了图形拼接理论，证明在满足“非重叠”条件下多次聚类得到的子图可合并为原始完整图；（3）将该理论应用到实际的三维运动捕捉数据中，首次实现了多维特征的分层聚类与图形合成。

**🔧 技术方法**

使用的技术包括：基于集合论与范畴论的 quasi-skeleton WD 图定义；改进版 Hasse 聚类算法（Algorithm 1）及其对转移闭包、可达性关系的判定；多级特征提取脚本与机器学习分类器（逻辑回归、梯度提升等）生成事件序列；以及利用图论的拓扑归约和 transitive reduction 进行图形拼接。

**📊 数据集**

数据集为公开的 FS-Jump3D（240 条包含 12 视角的三维运动捕捉视频，涵盖 4 名选手和 6 种跳跃动作），通过低层脚本提取 8 个技术事件后转化为序列。

**📈 对比分析**

通过两轮迭代 Hasse 聚类（分别使用 5 个和 2 个事件子集）得到 5 组与 2 组子图，最终拼接得到完整的 6 种跳跃的 wiring diagram。实验表明：
- 第一轮聚类在 239/240 条视频中成功分成 5 组，纯度最高 100%；
- 第二轮聚类将 flip/lutz 合并组拆分为两组，准确率 100%；
- 合并后得到的 wiring diagram 与理论理想图完全一致。性能方面，改进算法在 r≤5 的限制下可在个人电脑上完成，显著降低了内存占用与计算时间。

**⚠️ 局限性**

限制主要在于：
- 需要先训练准确率较高的低层特征分类器；在此案例中 inside/outside edge 的分类器精度较低，导致需要分两轮聚类；
- 迭代拼接理论在满足“非重叠”与“路径可分解”条件下才成立，若数据噪声或特征缺失过多，可能无法满足这些条件；
- 目前仅在理想数据上验证理论完整性，对实际噪声环境的鲁棒性还有待进一步评估。

---

## 152. JigShape: Evaluating Visual-Geometric Reasoning in VLMs through Jigsaw Puzzles

**arXiv ID:** 2607.27670 | [PDF](https://arxiv.org/pdf/2607.27670v1)

**作者:** Shawn Li `[一作]` (University of Southern California), Yue Zhao `[通讯]` (University of Southern California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一套新的拼图解决基准，利用带有插槽与凸起边界的拼图块，将几何约束与视觉内容相结合，形成唯一解的任务；

**💡 创新点**

创新点在于：①通过形状约束消除传统矩形拼图中的多义性；②设计四种不同网格密度（4×4至16×16）的规模化数据集；③引入形状与无形状对照实验，明确几何约束对模型表现的贡献；

**🔧 技术方法**

采用视觉‑语言模型（VLM）与监督微调（SFT）技术，对比零-shot与有监督方法；使用形状匹配与邻接准确率等多指标评估；

**📊 数据集**

数据集来源为23,742张高分辨率图像（DIV2K、DIV8K、Unsplash），通过程序生成95,468个拼图实例；还提供了无形状（矩形）版本用于 ablation；

**📈 对比分析**

对比五个前沿VLM（GPT‑5.5、Claude Opus 4.8、Grok‑4.2、Llama‑4‑Maverick、GPT‑5.4‑mini）在零-shot模式下的表现，发现仅 GPT‑5.5 在 4×4 拼图上超过随机基准；两种微调模型（Qwen3‑VL‑8B、Gemma3‑12B）在 4×4 上达97%+准确率，但在 8×8 及更大尺寸上迅速下滑，形成“scaling cliff”；

**⚠️ 局限性**

局限性包括：①目前模型在大规模拼图上几乎无差异，无法展示持续的几何推理；②微调模型高度依赖形状约束，视觉内容整合不足；③评估仅针对固定尺寸、无旋转的拼图，未覆盖更通用的拼图场景。

---

## 153. Harness-G: A Graph-Structured Harness for Search Agents

**arXiv ID:** 2607.27652 | [PDF](https://arxiv.org/pdf/2607.27652v1)

**作者:** Yanning Hou `[一作]` (National University of Defense Technology), Jian Huang `[通讯]` (National University of Defense Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于图结构的检索接口和结构化非贪婪信用分配的方法，用于强化学习搜索代理。

**💡 创新点**

创新点在于将自由查询空间重构为有限动作选择的菜单，并利用可预览的图动作前沿进行前沿相对和启用信用分配。

**🔧 技术方法**

使用了图结构检索、前沿可预览、SNC（结构化非贪婪信用）以及GRPO强化学习框架。

**📊 数据集**

在六个问答基准（2WikiMultiHopQA、HotpotQA、MuSiQue、Natural Questions、PopQA、TriviaQA）上进行评估。

**📈 对比分析**

与Graph-R1等基线相比，Harness-G在所有模型规模上均实现最高平均F1，1.5B规模提升10.74点，3B规模提升3.98点，并在跨数据集迁移中表现更佳。

**⚠️ 局限性**

局限性包括对文本检索的依赖，尚未扩展到多模态证据以及对图构建的手工化可能限制规模扩展。

---

## 154. Real-Time Hard Peak Age-of-Information Safety with No-Regret Learning

**arXiv ID:** 2607.27626 | [PDF](https://arxiv.org/pdf/2607.27626v1)

**作者:** Wentao Zhang `[一作]` (Tsinghua University), Wentao Mo `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种在线调度框架OCO‑PAoI‑Hard，能在完全对抗性信道和到达序列下实现每个传感器峰值信息年龄（PAoI）不超硬实时截止且无后向违例；

**💡 创新点**

创新点在于把PAoI截止直接归约为资源分配上的仿射半空间约束，结合严格因果的提议‑盾‑更新循环（proposal‑shield‑update）实现零违例与O(√T)无后悔学习；

**🔧 技术方法**

主要使用了约束在线凸优化（OCO）、欧氏投影到多面体安全集合、梯度下降、虚拟队列做后验证书，以及边界安全裕度和近似投影稳健性分析；

**📊 数据集**

实验采用人工生成的对抗共享信道“陷阱”频道（trap channel）及其多传感器仿真环境，无真实工业数据集；

**📈 对比分析**

与Vanilla OGD、长周期虚拟队列（Long‑Term VQ）、贪心最大欠缺（Greedy max‑deficit）和轮询（Round‑robin）等基线对比，OCO‑PAoI‑Hard在所有种子下实现0%违例、近似最优成本（比VQ低约30%）且经验收敛速率小于√T；

**⚠️ 局限性**

局限性包括：仅保证模型化的流体状态PAoI安全；对包级安全需更强服务假设；依赖每步可行性（one‑step viability）和凸损失；对强噪声和非凸约束的适应性有限。

---

## 155. Kalman Meets Curriculum: Efficient Dynamic Prompt Selection for Adaptive RL Finetuning

**arXiv ID:** 2607.27610 | [PDF](https://arxiv.org/pdf/2607.27610v1)

**作者:** Haodong Zhu `[一作]` (Beihang University), Baochang Zhang `[通讯]` (Beihang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在线提示选择框架KGPS，用动态状态估计代替传统的静态难度预测，显著提升LLM强化学习微调的推理性能；

**💡 创新点**

创新点在于：①将提示难度建模为线性高斯状态空间，过程噪声与策略更新幅度耦合；②利用Kalman滤波维护每个提示的后验分布；③基于后验期望的训练效用评分，兼顾探索与利用，且不需要额外的LLM推理；

**🔧 技术方法**

主要技术包括线性高斯状态空间模型、Kalman滤波、logit变换、Gauss–Hermite四步求积分、后验期望训练效用；

**📊 数据集**

使用数学推理、规划与视觉几何三大基准：MATH、Countdown、Geometry3k，以及多种模型规模如Qwen3-0.6B/4B/8B、DeepSeek-R1-Distill-7B、Qwen3-VL-2B-Instruct等；

**📈 对比分析**

与四类基线对比：均匀采样、GRESO、MoPPS以及评估式DS；KGPS在保持与DS相当或更高准确率的同时，显著减少70%~83%的rollout成本，并在各模型规模与任务上持续优于其他方法；

**⚠️ 局限性**

局限性包括：依赖于已训练策略的梯度幅度估计，若策略更新不稳定或非梯度驱动可能导致过程噪声估计失效；此外，Kalman滤波假设高斯近似，对极端离群成功率估计可能不够精确。

---

## 156. Compliance2LoRA: On-Demand Safety Alignment on Arbitrary Policy Subsets via Hypernetwork-Generated LoRA Adapters

**arXiv ID:** 2607.27594 | [PDF](https://arxiv.org/pdf/2607.27594v1)

**作者:** Pankayaraj Pathmanathan `[一作]`, Furong Huang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并训练了一个可调节的超网络LoRA生成器，让大型推理模型仅在指定安全政策子集下生成合规响应。

**💡 创新点**

提出单模型通过注意力掩码实现任意政策子集的即时切换，避免了传统多模型组合的组合爆炸。

**🔧 技术方法**

使用超网络生成LoRA权重、注意力聚合、LoRA适配器、SFT与DPO混合训练。

**📊 数据集**

以Star 41K安全数据集和DAN数据集进行训练与评估。

**📈 对比分析**

与基于上下文学习和专用微调的基线相比，安全率提升至约90%，同时显著降低了模型数量和推理成本。

**⚠️ 局限性**

对极稀有或未见政策组合的泛化仍有限，且对非常细粒度的政策切换可能产生微小安全偏差。

---

## 157. HALO: Heterogeneous Admission through Localized Obligations for Safe Agentic Execution

**arXiv ID:** 2607.27636 | [PDF](https://arxiv.org/pdf/2607.27636v1)

**作者:** Taewoo Park `[一作]` (Korea University), Hwangnam Kim `[通讯]` (Korea University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 HALO（Heterogeneous Admission with Localized Obligations）协议，用于在代理系统中对多种输出的运行时支持漂移进行组件级别的检查与恢复；

**💡 创新点**

创新点在于：①把每个输出视为独立组件，执行依赖一致的保留与删除；②在外部接口调用前对已通过入门检查的动作进行最终精确重检；③为被阻止的动作生成局部义务，限定只允许新的候选者恢复；

**🔧 技术方法**

采用了可信目录规则、状态提供者、受控适配器、一次性调度令牌、加密摘要、依赖图闭包和最后重检门控等技术；

**📊 数据集**

使用了 UAV 的 PX4/Gazebo 仿真环境和 Crazyflie 物理平台的实验数据，以及结构化响应回放的人工生成数据集；

**📈 对比分析**

与 WholeResponse、IndependentFilter、AgentSpec 等方案对比，HALO 在 96 个入门测试、20 个协议测试、10,000 个调度方案以及 10 次 PX4/Gazebo 冷启动测试中全部通过；在回放实验中保留 248/248 组件，阻止所有过时动作，且每个动作的最终重检平均耗时约 0.035 ms，整体入门成本约 0.333 ms；

**⚠️ 局限性**

局限性包括：仅保障在可信目录、状态源与授权账本范围内的边界内安全；不涉及下游物理系统的最终正确性；依赖外部适配器与授权令牌的实现；对极端并发或分布式事务缺乏完整证明；

---

## 158. Arm2Air: Cross-Embodiment Skeleton Transfer for 3D Relay Formation

**arXiv ID:** 2607.27627 | [PDF](https://arxiv.org/pdf/2607.27627v1)

**作者:** Dohun Lee `[一作]` (Korea University), Hwangnam Kim `[通讯]` (Korea University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

在城市三维环境中利用从机器人手臂迁移的障碍规避骨架，自动化无人机中继网络的链式放置。

**💡 创新点**

将异构装置的有序几何骨架迁移到无人机域，利用低秩适配实现数据与计算效率的双重提升。

**🔧 技术方法**

基于Transformer的迁移平台、低秩适配（LoRA）、点云感知与通信约束下的梯度优化。

**📊 数据集**

使用人工生成的城市三维地图（90训练、30验证、30测试），以及预训练的Neural MP机器人手臂数据。

**📈 对比分析**

与传统规划器（A*, D*, Dijkstra, RRT）以及IMPC-MD、AO Placement比较，Arm2Air在规划时间、瓶颈容量、延迟、跳距平衡与移动成本上均显著优于基线，提升幅度达30%以上。

**⚠️ 局限性**

依赖预训练模型与静态地图，对动态障碍、实时点云、通道不确定性及多无人机实际部署的鲁棒性尚未验证。

---

## 159. ICLE++: Modeling Fine-Grained Traits for Holistic Essay Scoring

**arXiv ID:** 2607.27671 | [PDF](https://arxiv.org/pdf/2607.27671v1)

**作者:** Shengjie Li `[一作]` (University of Texas at Dallas), Vincent Ng `[通讯]` (University of Texas at Dallas)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个新的说服性学生作文语料库ICLE++，并对其进行全局与10个细粒度特征的双重评分；

**💡 创新点**

创新点在于：①构建了跨题目、跨语言、写作水平多样的说服性作文集；②引入10项细粒度特征评分，为模型提供更丰富的反馈；③验证了特征评分对跨题目自动评分的提升；

**🔧 技术方法**

采用了现有AES模型（Uto et al., Kumar et al., PMAES）进行多任务学习或单任务学习，利用回归和二元神经网络预测整体分与特征分；

**📊 数据集**

使用ICLE++（1006篇说服性作文，来自16国、16母语、平均600词）以及传统ASAP语料库进行对比实验；

**📈 对比分析**

通过5折/10折交叉验证（within‑prompt）和留一题目交叉验证（cross‑prompt）评估，使用Quadratic Weighted Kappa（QWK）衡量效果；结果显示在ICLE++上QWK普遍低于ASAP，但加入特征后跨题目性能提升；

**⚠️ 局限性**

限制在于只关注说服性作文，且样本来自非母语本科生，可能不具备对母语高中生或其他作文类型的广泛适用性；

---

## 160. Witness Evidence Portfolios: Single-Prefill Risk Detection for Closed Multimodal Answers

**arXiv ID:** 2607.27667 | [PDF](https://arxiv.org/pdf/2607.27667v1)

**作者:** Fexiang Liu `[一作]`, Zheng Wang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在多模态大型语言模型（MLLM）推理时评估答案可信度的 Witness Evidence Portfolios (WEP) 方法。

**💡 创新点**

创新点在于利用签名视觉决策证据的来源与程度，将视觉支持与对抗信息映射为可解释的路由分组，并通过嵌套验证选择最可靠的证据家族，形成稀疏路由组合与置信度融合。

**🔧 技术方法**

技术包括白盒预填充路径的注意力/数值读取、视觉绑定分布、证据分离为正负、可解释的证据本土化与浓度评估、内部路由子集选择与稀疏组合、置信度归一化融合。

**📊 数据集**

使用了 Qwen3‑VL‑8B、LLaVA‑1.5‑7B、InternVL3.5‑8B 三种 MLLM，以及 AMBER‑D、Causal‑HalBench、VSR、HallusionBench 四个闭合答案视觉推理/幻觉基准。

**📈 对比分析**

与单一置信度基线（候选边缘）和多种内部证据或额外路径基线比较，WEP 在所有 12 对比上均实现正的 AP 提升，平均提升 0.134，且在 1%–10% 的审核预算下精度提升 0.18–0.20，表明其在高风险错误排序上的显著优势。

**⚠️ 局限性**

局限性在于需要白盒访问、有限候选集、标注的校准样本，并且对自由形式答案或非二分类任务适用性不明，且模型对视觉预填充的性能仍受限于当前架构。

---

## 161. Understanding Submodular Information Measure Based Objectives for Representation Learning: A Variance and Separation Perspective

**arXiv ID:** 2607.27660 | [PDF](https://arxiv.org/pdf/2607.27660v1)

**作者:** Rishabh Iyer `[一作]` (University of Texas at Dallas), Anay Majee `[通讯]` (Adobe)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过理论分析和合成实验，阐明了子模信息测度（SIM）在监督表示学习中的几何与统计偏好，揭示不同 SIM（Graph Cut、LogDet、Facility Location）对应的方差、协方差、分离和覆盖量化指标；

**💡 创新点**

创新点在于建立了 SIM 与传统统计量（内类方差、广义方差、均值分离、马氏分离、最近模态重叠）之间的严格对应关系，并通过理论证明与实验验证其精确匹配；

**🔧 技术方法**

主要技术包括子模函数（Graph Cut、LogDet、Facility Location）与子模互信息/总信息公式的推导、矩阵分析、随机高斯混合模拟与相似度核（shifted Euclidean、RBF）设计；

**📊 数据集**

实验数据集为人工生成的高维高斯混合样本，独立控制方差、协方差、类不平衡、均值分离及多模态重叠；

**📈 对比分析**

与传统统计量直接比较，采用相关系数、重构误差等定量指标，结果显示各 SIM 与其对应统计量呈近似线性/单调关系，验证了理论预测；

**⚠️ 局限性**

局限性包括：实验仅在合成数据上验证，缺乏真实任务性能评估；对自监督、检索增强生成等更广泛场景的适用性仍待探索；

---

## 162. Not as Sweet by Another Name: An Empirical Study of Format Robustness in LLM Document Workflows

**arXiv ID:** 2607.27648 | [PDF](https://arxiv.org/pdf/2607.27648v1)

**作者:** Xiaoyu Zhang `[一作]` (Nanyang Technological University), Yang Liu `[通讯]` (Nanyang Technological University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

针对端到端 LLM 文档工作流，构建了一套基于格式感知的变形测试框架，定义了三条可变形关系（决策结果、一致性证据、执行稳定性），并在四种主流工作流和四种文件格式上进行了 48,000 次大规模实验，系统评估了格式变化对准确率、决策漂移、证据漂移和稳定性的影响。

**💡 创新点**

创新点包括：①首次将变形测试方法迁移到完整的 LLM 文档工作流层面；②提出针对决策、证据、稳定性三维的可变形关系，细粒度揭示格式导致的隐性错误；③设计了轻量级的“格式路由”防护策略，利用离线经验数据将输入统一转换为最优格式，从而显著降低格式诱发错误。

**🔧 技术方法**

技术手段主要有：变形测试框架、三种可变形关系的自动判定、对不同工作流的文件上传与解析接口适配、证据集提取与相似度计算（Jaccard/JS Divergence）、执行多次重复运行以评估稳定性、以及两种用户端缓解策略（投票聚合与格式路由）。

**📊 数据集**

使用了四个真实或模拟的高风险任务数据集：MedQA（医学问答）、Construct（多字段医疗/社会记录）、DiscrimEval（偏见评估）、Credit Card（信用卡违约）。每个数据集各抽取 250 条实例，共计 1,000 条样本。

**📈 对比分析**

对比方法主要是基于三条可变形关系的违例率（MRV）与传统准确率/公平度/稳定性指标；实验显示，单一格式切换即可导致准确率最高降至 56%，MRV 最高 91%。在轻量级防护实验中，投票聚合对 MRV 提升有限，而格式路由平均将 MRV 降低 44%（最优约 48%），在决策准确率上提升 28% 以上。

**⚠️ 局限性**

局限性：①实验聚焦于高风险决策型任务，未覆盖开放式生成任务；②仅评估四种工作流与四种格式，难以全面泛化到新兴多模态或多代理系统；③格式转换过程中仍可能隐含微观语义偏移，虽然通过专家校验减少但无法完全排除；④重复运行次数仅为 3 次，可能低估随机性影响；⑤缺乏对实时更新的工作流动态适配的路由策略。

---

## 163. MMOOC: A Comprehensive Benchmark for Out-of-Context Evaluation in Multimodal Large Language Models

**arXiv ID:** 2607.27637 | [PDF](https://arxiv.org/pdf/2607.27637v1)

**作者:** Wenjie Zhu `[一作]` (Hong Kong Polytechnic University), Lei Zhang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并评测了一个大型多模态LLM鲁棒性基准MMOOC，涵盖多种视觉场景和问题格式，并通过该基准对现有模型的拒绝与答案保持能力进行系统评估。

**💡 创新点**

①提出了八类OOC与Shifted IC场景，并对每类细化到具体子类型；②将多模态LLM生成、过滤与人工审核相结合，形成高质量41K图像-问题对；③引入LLM-as-a-Judge多评审框架和多指标评估（Accuracy、Answer Rationality、Refusal Rate、Refusal Rationality）来精准衡量拒绝与回答的合理性；④展示对齐（SFT、DPO）与提示策略对OOC鲁棒性的显著提升。

**🔧 技术方法**

使用多模态大模型（如Qwen3-VL系列、LLaVA、InternVL、Gemma、Llama、Ministral等）进行生成与评估；通过GPT‑4o、o1、o3进行数据过滤；采用GPT‑5.6、Claude‑Opus‑5、DeepSeek‑V4‑Pro三位评审器进行答复合理性评估；利用SFT、DPO等对齐方法改进模型行为；采用多指标评估框架和人类标注对照来验证结果。

**📊 数据集**

主要使用新构建的MMOOC数据集（41K图像-问题对，包含三种问题格式、六种视觉场景、八类场景类型），并融合了MoHoBench、SNIFFER、UPD等已有数据以增强样本多样性。

**📈 对比分析**

对18个代表性多模态LLM（13开源、5闭源）进行系统评测，比较IC与OOC两类任务的Accuracy、Answer Rationality、Refusal Rate和Refusal Rationality。结果显示：OOC性能普遍不高，尤其在Uncertain Spatial & Physical Context和Unclear Logical & Symbolic类型；模型规模并不总是提升OOC鲁棒性；对齐与提示可显著提升拒绝准确率和理由质量，但有时会牺牲IC回答质量；在Shifted IC场景中模型表现相对更好，但仍有明显欠缺。

**⚠️ 局限性**

仅覆盖图像‑文本交互场景，未扩展至视频、音频或具身环境；基准的生成与人工审核流程成本高，可能限制大规模扩展；评估侧重于OOC与Shifted IC两类任务，对更广泛的鲁棒性维度（如视觉噪声、跨域迁移）关注不足。

---

## 164. MedXplore: Towards Reliable and Unbiased Generalized Category Discovery in Medical Imaging

**arXiv ID:** 2607.27620 | [PDF](https://arxiv.org/pdf/2607.27620v1)

**作者:** Jianwei He `[一作]` (Institute of Automation Chinese Academy of Sciences), Jie Hao `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出 MedXplore，一种面向医学图像的统一框架，用于可靠且无偏的通用类别发现（GCD），通过感知层的频域自适应注意与一致性（FAAC）和决策层的自适应余弦角度边距（ACAM）实现。

**💡 创新点**

创新点：
1) FAAC 采用可学习全频谱滤波和全局-局部能量对比激活，实时提取可靠的病灶语义锚点，解决传统空间注意力的偏差；
2) ACAM 依据语义难度与特征置信息自适应调整角度与余弦边距，动态平衡类内紧密度与类间可分离度；
3) 两模块协同工作，既在训练阶段提升表示，又不增加推理成本，实现无偏类别发现。

**🔧 技术方法**

技术手段：
- 可学习全频谱滤波（FFT + 深度卷积）
- 全局-局部能量对比激活（GLEC）
- Patch一致性约束
- 对比学习（有监督与无监督）
- 参数化分类 + 像素级伪标签
- 自适应角度/余弦边距损失

**📊 数据集**

使用数据集：NCT-CRC-HE-100K（结肠癌组织切片），OrganAMNIST（腹部CT器官图像），OrganCMNIST（模拟CT器官图像），Kvasir（胃肠道内镜图像）。

**📈 对比分析**

与 10+ 现有 GCD 方法（RS+, UNO+, ORCA, GCD, DCCL, CMS, SimGCD, LegoGCD, NGUF）对比；在所有四个医学数据集上均实现 SOTA。以 Kvasir 为例，All 准确率从 75.2% 提升至 91.7%，并将误把新类判为旧类的错误率从 14.5% 降至 0.8%。

**⚠️ 局限性**

局限性：
- 主要针对局部病灶特征，对低频、弥漫性病变的鲁棒性尚未验证；
- 需要先验的旧/新类别划分和已知类别数量；
- 对极端域迁移（不同设备、扫描协议）仍需进一步评估。

---

## 165. Analog Courant Numbers and their Role in Analog Computing

**arXiv ID:** 2607.27609 | [PDF](https://arxiv.org/pdf/2607.27609v1)

**作者:** Arash Ghasemi `[一作]` `[通讯]` (University of Tennessee at Chattanooga), Arash Ghasemi (University of Tennessee at Chattanooga)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文识别了模拟计算方法中的动态约束，其中矩阵的一行由阻抗网络表示。研究表明，最快的归一化模式不超过所有电路行中最大组合单位增益带宽（CUGBW）的2π倍。

**💡 创新点**

创新点在于提出了一个理论框架，限制了模拟硬件在其输出上可以物理表示和解析的操作速率，并通过大规模LTspice仿真验证了该理论。

**🔧 技术方法**

使用了LTspice进行大规模电路仿真，分析了从CMOS到热电子真空管电路的不同架构。

**📊 数据集**

使用了多个基准电路，包括一维热方程、基于图的半监督学习问题和图正则回归。

**📈 对比分析**

通过与现有方法的比较，验证了所提出的理论框架的有效性，结果显示在不同架构下的收敛时间和最终误差表现出显著差异。

**⚠️ 局限性**

限制在于该研究假设了对称正定的简化模型，且未考虑非对称或非互惠网络的情况，这可能导致不同的动态行为和收敛特性。

---

## 166. LimICE: Integrating LLM into ICE Framework for Efficient Loop Invariant Inference

**arXiv ID:** 2607.27606 | [PDF](https://arxiv.org/pdf/2607.27606v1)

**作者:** Kai Fan `[一作]` (National University of Defense Technology), Ji Wang `[通讯]` (National University of Defense Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Incremental ICE 框架和基于该框架的循环不变式推理工具 LimICE

**💡 创新点**

将 IC3 的增量理念融入 ICE 学习框架，形成增量式学习目标和反例过滤机制，并结合 LLM 与 ICE‑DT 的双重合成策略

**🔧 技术方法**

使用 LLM（DeepSeek、GPT‑3.5‑Turbo、QWen‑2.5‑7B）生成原子文字、2‑CNF 词法组合器、SMT 反例检查、ICE‑DT 决策树回退

**📊 数据集**

在 367 条线性和 50 条非线性（来自 LaM4Inv、Clause2Inv、CHC‑COMP 等）基准上进行评测，并自行构造新测试集

**📈 对比分析**

与 8 种 SOTA 方法（LoopInvGen、ICE‑DT、ICE‑DT‑Interval、Code2Inv、LIPuS、CLN2INV、LaM4Inv、Clause2Inv）对比，LimICE 在线性和非线性两类基准上分别解决了 349/47 例，平均时间 15.2/8.8 秒，比 LLM 基线提升 12‑24% 题解并缩短 36‑63% 运行时

**⚠️ 局限性**

依赖 LLM 的生成质量导致在极小模型或高非线性问题上性能下降；SMT 求解器在乘法、除法等算术运算上的效率限制；实验集主要为公开基准，真实复杂程序的验证仍待进一步验证

---

## 167. World Action Planner: Generalizable Decision-Making with Action-Conditioned World Models

**arXiv ID:** 2607.27599 | [PDF](https://arxiv.org/pdf/2607.27599v1)

**作者:** Xiangcheng Zhang `[一作]` (Harvard University), Yilun Du `[通讯]` (Harvard University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出World Action Planner，一种利用VLM和动作条件世界模型进行计划、优化与搜索的机器人规划系统。

**💡 创新点**

创新点在于将VLM作为高层规划器，配合姿态图像条件的多视角Diffusion世界模型实现想象式规划，显著提升端到端策略在新任务、布局和零样本场景的泛化能力。

**🔧 技术方法**

采用Gemini 3.0 Flash等VLM生成初步动作序列，姿态图像条件的多视角Diffusion世界模型进行未来图像模拟，结合全局优化、局部搜索和低级执行策略实现动作规划。

**📊 数据集**

在LIBERO、Robocasa、MimicGen、DexMimicGen四大仿真套件上训练世界模型，并在LIBERO-Long、LIBERO-Object等任务中评估规划性能。

**📈 对比分析**

与SOTA VLA、WAM、SAILOR、GPC-RANK等端到端或世界模型增强基线比较，组合任务上成功率从4%–28%提升至72%–78%，新布局上从0%–10%提升至88%–90%，零样本上从58%–22%提升至80%–76%。

**⚠️ 局限性**

评估仅在仿真环境完成，缺乏真实机器人实验；世界模型推理耗时，实时效率待进一步提升。

---

## 168. A Systems Engineering Framework for Vision-Language-Enabled UAV Triage and Disaster Response

**arXiv ID:** 2607.27597 | [PDF](https://arxiv.org/pdf/2607.27597v1)

**作者:** Swapnil Saha `[一作]`, Neelakshi Majumdar `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了基于MBSE的视觉语言模型（VLM）协同无人机灾害救援框架，并实现了软件仿真。

**💡 创新点**

创新点在于将VLM作为人机交互协调代理嵌入ICS流程，实现可追溯、可模块化的系统设计，从而提升协作效率并减少操作员工作负荷。

**🔧 技术方法**

采用的技术包括VLM+LLM、ROS 2、Gazebo仿真、PX4飞控、QGroundControl以及软件仿真（SITL）实现。

**📊 数据集**

使用了模拟灾害场景和合成传感器数据，没有使用公开数据集。

**📈 对比分析**

通过7名受试者的NASA‑TLX工作负荷问卷与AI辅助对照，AI条件下工作负荷下降约40%，且信任与沟通清晰度均高于4/5。

**⚠️ 局限性**

局限在样本量小、仅仿真验证、未完整实现所有MBSE模块、缺乏多无人机、真实灾害多变性与通信延迟测试。

---

## 169. Prox: Training-Free FFN Activation Sparsity via Approximate Intermediate-Channel Salience in LLMs

**arXiv ID:** 2607.27591 | [PDF](https://arxiv.org/pdf/2607.27591v1)

**作者:** Jinyi Liu `[一作]` (Institute of Software Chinese Academy of Sciences), Jun Wei `[通讯]` (Institute of Software Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段、训练无关的Prox框架，用低成本INT4代理计算输入级稀疏化并生成SwiGLU中间状态的通道掩码，随后在全精度权重下精确计算被选通道，从而实现LLM FFN的高效稀疏推理。

**💡 创新点**

创新点在于：①只需保留中间状态的相对大小排序即可得到有效掩码，②通过输入级稀疏化+量化代理大幅降低代理成本，③动态分配两阶段计算预算以实现目标稀疏率，④兼容量化与稀疏注意力，实现互补加速。

**🔧 技术方法**

使用INT4量化代理权重、输入级稀疏化、SwiGLU中间状态排序、CUDA/Triton定制稀疏核、基于成本模型的稀疏预算分配。

**📊 数据集**

在十个大模型（Qwen3、Qwen3.5、Ministral、Mistral、Llama‑3、Gemma‑3）和多种量化配置（AWQ、FP8、W4A16、W8A8）上进行评估，并使用EleutherAI LM Harness 8‑shot GSM8K、5‑shot MMLU、10‑shot HellaSwag 等任务测试下游准确率。

**📈 对比分析**

与三种训练无关稀疏基线（CATS、COUNTDOWN、TEAL）对比，Prox在60–70% FFN稀疏率下实现平均 1.51–1.99× 的单批解码吞吐提升，同时保持或超过基线的下游任务性能；在高稀疏率下，性能提升最为显著。

**⚠️ 局限性**

主要局限包括：仅针对单批自回归解码；需额外存储 INT4 代理权重（约 12% 额外权重占用）；对大批量推理及多 GPU 分布式场景的支持尚未实现。

---

## 170. Can Large Language Models Resolve Real Java Merge Conflicts? An Evaluation with a Calibrated LLM-as-Judge

**arXiv ID:** 2607.27674 | [PDF](https://arxiv.org/pdf/2607.27674v1)

**作者:** Bowen Shen `[一作]` `[通讯]` (Virginia Polytechnic Institute and State University), Bowen Shen (Virginia Polytechnic Institute and State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个两层评估框架，先使用LLM生成冲突解决方案，再用校准过的LLM评判器与结构化检查来评价其质量。

**💡 创新点**

通过先在真实冲突上用人工标签校准LLM评判器（保证100%精度），从而得到保守的下限指标，并证明LLM在覆盖率上明显优于传统合并工具。

**🔧 技术方法**

利用大型语言模型（OpenAI GPT‑4 与 Gemini）、G‑Eval 评判器、生成‑验证‑重试循环以及基于 Java 语法树的结构化检查。

**📊 数据集**

使用 ConflictBench 数据集，共 180 个 Java 合并冲突场景，其中 93 个可重建并用于评估。

**📈 对比分析**

将 LLM 与五个传统合并工具在同一衡量标准（开发者匹配率）下对比；LLM 在真冲突上匹配开发者约55%，在覆盖率公平比较下比最强工具高 18–22 分，准确率相近。

**⚠️ 局限性**

局限于 Java 语言、仅评估可重建场景、只测试两款模型、评判器可能低估合法替代方案，且结构正确性仍需确定性检查。

---

## 171. Looped Transformers with Source-Centered State Evolution

**arXiv ID:** 2607.27656 | [PDF](https://arxiv.org/pdf/2607.27656v1)

**作者:** Bum Jun Kim `[一作]` (University of Tokyo), Yutaka Matsuo `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Source-Centered State Evolution（SCSE）架构，利用固定输入条件锚点和零偏差掩码，在循环 Transformer 中实现共享参数的深度可伸缩。

**💡 创新点**

创新点在于将锚点视为一阶固定点，构造零偏差递归核心并加入逐样本零偏差掩码，从而消除循环过程中因源注入产生的可选偏差自由度，并在保持输入依赖性的同时提升深度泛化。

**🔧 技术方法**

采用共享 Transformer 块、学习锚点投影模块 a_ω、零保持递归核心 G_θ、残差步长标度 s、RMSNorm 归一化以及逐样本零偏差掩码 m；训练时在循环深度 1–8 之间随机采样，评估时固定多层深度。

**📊 数据集**

在 WikiText‑2、WikiText‑103、OpenWebText、Web‑Corpus、LAMBADA 等文本数据集上进行实验，覆盖不同上下文长度、训练预算、迁移与完成任务。

**📈 对比分析**

与基准共享块 Transformer、调参适配器、循环步长条件适配器、容量匹配对照等方法比较，SCSE 在所有评测深度（包括训练深度范围外的 8–48 层）均实现最低 perplexity，特别是在超深循环下显著优于对照组。

**⚠️ 局限性**

局限性包括：对更大模型规模和更长序列的验证不足；掩码阈值需经验调节；源注入的精细控制仍受读出-损失对齐影响；在非文本领域的可推广性待进一步研究。

---

## 172. LoopMemGR: From Behavior Logs to Evolving Memory for Generative Recommendation

**arXiv ID:** 2607.27647 | [PDF](https://arxiv.org/pdf/2607.27647v1)

**作者:** Hui Qian `[一作]` (Alibaba Group), Bo Zheng `[通讯]` (Alibaba Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了 LoopMemGR 闭环记忆框架，在生成式推荐中同时维护行为日志和推荐经验日志，并通过三视图读取器将经验压缩为固定的 16 个经验 token，实现读-写-记忆-推荐的闭环流程。

**💡 创新点**

1) 识别并解决生成式推荐中的单向记忆缺陷，将系统端推荐经验纳入记忆；2) 通过近期、频率和全局三视图并行读取并融合经验；3) 采用门控残差和 RMS 约束，在仅 16 个 token 的预算下保留大部分原始经验收益。

**🔧 技术方法**

基于 RankGR 的生成式推荐框架，使用 Semantic ID (SID) 作为离散 token；三视图读取器使用归一化交叉注意力、门控残差更新以及 RMS/Cap 约束；经验压缩与写入采用固定写入模块；模型采用 BFloat16 训练、动态束搜索生成，且加入多项正则化和多头注意力。

**📊 数据集**

使用工业级淘宝（Taobao）数据集，包含用户点击和页面浏览（PV）交互记录。

**📈 对比分析**

与传统序列模型（SASRec、Caser 等）以及生成式检索模型（FORGE、RankGR、TIGER 等）进行对比。LoopMemGR 在 Click HR@500~2000 和 PV HR@500~2000 等指标上分别提升约 20–30%（以 HR@2000 为例提升 11–15%），整体排名第一。

**⚠️ 局限性**

1) 经验压缩仍需线性扫描完整日志，极长历史的实时读取成本较高；2) 只在单一工业场景验证，跨平台或不同业务的泛化性待进一步验证；3) 视图与压缩参数设计依赖经验，可能需要手动调优。

---

## 173. HealthCAT: An Interpretable Encoder-only Transformer Framework for Health Indicator Prediction and Temporal Interpretation of Wearable Sensor Data

**arXiv ID:** 2607.27635 | [PDF](https://arxiv.org/pdf/2607.27635v1)

**作者:** Xiaotong Yu `[一作]` (University of Sydney), Kalina Yacef `[通讯]` (University of Sydney)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出HealthCAT框架，将Encoder‑only Transformer与AttentiveCAT结合，实现对可穿戴传感器时序数据的健康指标预测与时间步级可解释性。

**💡 创新点**

创新点在于：①将类激活与自注意力融合得到时间步级重要性分数；②通过域适配可视化，将解释映射到日常行为周期；③提供定量的掩蔽实验验证解释有效性。

**🔧 技术方法**

采用Encoder‑only Transformer模型、Attentive Class Activation Token (AttentiveCAT)、自注意力权重、梯度归因、掩蔽实验和域适配可视化技术。

**📊 数据集**

使用两个真实可穿戴数据集：206名太平洋岛屿青少年（基于GENEActiv加速度计）预测健康体重状态，100名成年人（Empatica E4多模态）预测睡眠呼吸暂停指数。

**📈 对比分析**

与Transformer、GRU+Attention、LSTM+Attention基线比较，HealthCAT在两数据集上F1得分提升约17%（PA）和12%（DREAMT），并在掩蔽实验中显著优于随机选择，表明时间步解释更具预测价值。

**⚠️ 局限性**

局限性包括样本规模有限、仅涵盖可穿戴传感器数据未整合饮食、吸烟等情境变量，以及缺乏更大、多样化人群的验证。

---

## 174. BlindPSNR: A No-Reference Fidelity Predictor for Low-Light Image Enhancement

**arXiv ID:** 2607.27628 | [PDF](https://arxiv.org/pdf/2607.27628v1)

**作者:** Mingzhe Lyu `[一作]` (Southern University of Science and Technology), Hong Zhang `[通讯]` (Southern University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出BlindPSNR，一种无需参考图像即可预测低光图像增强结果PSNR并自动选择最佳增强候选的框架。

**💡 创新点**

创新点在于利用可解析的全参考log‑MSE目标直接监督无参考网络，结合低光输入的窗口交叉注意力和双重损失，显著提升候选选择的准度。

**🔧 技术方法**

采用ConvNeXt‑Tiny骨干、窗口交叉注意力融合低光与增强图像、异方差回归与平滑ℓ1蒸馏两项损失，实现全局log‑MSE预测与PSNR回归。

**📊 数据集**

训练集覆盖RealX3D、LOM、LOL‑v1/2、mip‑NeRF‑360等多来源图像，评估集包括RealX3D hold‑out、LOL‑v2‑real、LSRW、mip‑NeRF‑360四个场景。

**📈 对比分析**

与七种经典与学习型NR‑IQA基线对比，BlindPSNR在hold‑out RealX3D上top‑1达89.5%、平均regret仅0.026 dB，远优于基线0% top‑1，表现极佳。

**⚠️ 局限性**

在跨域未见数据集（LSRW、mip‑NeRF‑360等）性能下降（top‑1 12–38%），表明模型泛化受限，需要更丰富多样的训练样本来提升跨域鲁棒性。

---

## 175. An Exploration Graph with Continuous Refinement for Efficient Multimedia Retrieval

**arXiv ID:** 2607.27623 | [PDF](https://arxiv.org/pdf/2607.27623v1)

**作者:** Nico Hezel `[一作]` (HTW Berlin), Klaus Jung `[通讯]` (HTW Berlin)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了连续细化探索图（crEG），能够快速构建并持续优化偶度无向图，用于近似最近邻搜索和探索性检索。

**💡 创新点**

创新点在于：①偶度无向图保持图连通性；②基于平均邻居距离的改进质量度量；③增量构建和持续边优化算法；④兼顾检索与探索性能。

**🔧 技术方法**

技术包括：无向偶度图构造、范围搜索（RangeSearch）、近似相对邻居图（MRNG）判断、平均邻居距离度量、连续边优化（swap），以及多数据集实验评测。

**📊 数据集**

使用的公开多媒体数据集包括 Audio、SIFT1M、Deep1M、GloVe 等。

**📈 对比分析**

通过与 kGraph、EFANNA、DPG、ONNG、HNSW、NSG、NSSG 等现有图搜索算法在构建时间、内存占用、检索速度（QPS）与召回率的多维度比较，crEG 在 99% 召回率下的检索速度提升最高 250%，构建速度是 HNSW 的 2–3 倍，内存占用也最低。

**⚠️ 局限性**

局限性在于：尚未支持顶点删除；多线程优化仍有挑战；对极大规模动态数据集的扩展性待验证；仅评估了无过滤和单模态检索。

---

## 176. AWARE-FX: An Auditable Knowledge-Guided AI System for Measuring Corporate Foreign-Exchange Hedging Disclosure

**arXiv ID:** 2607.27611 | [PDF](https://arxiv.org/pdf/2607.27611v1)

**作者:** Qi Wang `[一作]` `[通讯]` (University of Nottingham), Qi Wang (University of Nottingham)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了可审计的AI/NLP决策支持系统，将香港上市公司年度报告文本转化为可追溯的公司年度外汇对冲披露指标。

**💡 创新点**

创新点在于：①专业源词典检索与否定、会计状态逻辑的规则层，②通道特定FinBERT分类器与精确门控，③保守top‑k聚合并提供完整审计日志，形成可复现、可追溯的测量流程；而非单一模型。

**🔧 技术方法**

使用技术包括：专业检索词典、规则式否定/会计状态逻辑、FinBERT（及与ModernBERT比较）、生成式LLM（Gemini、Qwen3‑8B）基准、BERT编码器、概率校准、选择性预测、门控、top‑k聚合、外部风险暴露验证。

**📊 数据集**

数据集：香港上市公司24,909家年度报告（2008‑2025）共543,527片段；弱标签训练集76,648片段；300片段人工审计样本；外部FX风险暴露面板用于验证。

**📈 对比分析**

评估方法多层次：训练‑验证‑测试划分、随机种子与时间序列前向测试、Brier & ECE校准、选择性预测、手工审计与LLM基准、与ModernBERT对比；性能：FinBERT平均F1≈0.85；多通道平均F1≈0.852；时间序列F1下降2–11%；手工审计F1≈0.81–0.89；LLM表现不均；外部验证显示严格FX分数与FX暴露显著负相关。

**⚠️ 局限性**

局限性：审计样本仅一人标注、弱标签可能偏差、LLM基准不可复现、PDF提取可能缺失、未涵盖表格/OCR、仅香港市场、未验证因果关系、模型随时间漂移、缺乏跨市场验证。

---

## 177. Revisiting the Adversarial Robustness of Graph-Based Traffic Forecasting

**arXiv ID:** 2607.27604 | [PDF](https://arxiv.org/pdf/2607.27604v1)

**作者:** Qingzhao Zhang `[一作]` `[通讯]` (University of Arizoan), Qingzhao Zhang (University of Arizoan)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出针对交通流量预测的可行攻击与检测防御框架

**💡 创新点**

引入五维威胁模型、物理感知的攻击和基于物理约束的检测器，将检测结果作为特征加入模型；在不降低正常性能的前提下，显著提升对局部攻击的鲁棒性

**🔧 技术方法**

基于图神经网络的时空预测模型（Graph WaveNet、STGCN等）、投影梯度下降（PGD）物理感知攻击、邻接矩阵加权的物理一致性检测器、零初始化适配器与对抗训练（RDAT、AT‑physics）

**📊 数据集**

三大公开交通数据集：PEMS‑BAY、METR‑LA、PEMS‑D4

**📈 对比分析**

与无防御、对抗训练（RDAT、AT‑physics）进行对比；在目标路段的MAE上，本文检测+防御在13/15种模型/数据组合中优于对抗训练，误差减少幅度最高可达10.6；网络全局误差几乎不变，且对未见攻击保持鲁棒

**⚠️ 局限性**

检测器依赖物理一致性特征，在信号噪声大或图结构不稳的情况下效果有限；适配器可能在极少数情况下被完全自适应攻击利用；假设攻击只影响物理一致性，未考虑更复杂的结构攻击或真实交通事件的误标记

---

## 178. Is Solving Better Than Evaluating GenAI Solutions?

**arXiv ID:** 2607.27586 | [PDF](https://arxiv.org/pdf/2607.27586v1)

**作者:** Ethan Dickey `[一作]` (Purdue University), Alexandros Psomas `[通讯]` (Purdue University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在一门大学算法课程中对比了传统解题与评估生成式 AI 解决方案两种教学活动，采用随机交叉设计检验其对学生学习效果的影响。

**💡 创新点**

创新之处在于将生成式 AI 产出转化为评估任务并系统评估其对概念理解、转移学习及自我调节学习的潜在价值。

**🔧 技术方法**

使用了随机化对照实验、交叉分组、统计检验（t 检验、Mann‑Whitney、Kolmogorov‑Smirnov 等）以及学生问卷调查收集主观感受。

**📊 数据集**

数据来源为 2025 春季普渡大学“算法分析”课程的 220 名学生在 6 次作业、期中与期末考试以及两次调查中的成绩与反馈。

**📈 对比分析**

与传统解题相比，评估 AI 方案在期中/期末总分、转移性考题及学习动机方面均未显著提升，局部作业分数略有差异，整体表现相当。

**⚠️ 局限性**

局限包括基于工作组随机化导致的群内相关性、生成式 AI 输出多样性、教学内容随时间变化、仅评估了分数与问卷而未测量更细致的元认知或调试技能。

---

## 179. ZMIS-SAM: Segment Anything Model Enhanced with Wavelet Transform for Zooplankton Microscopy Image Instance Segmentation

**arXiv ID:** 2607.27585 | [PDF](https://arxiv.org/pdf/2607.27585v1)

**作者:** Dekun Yuan `[一作]` (China University of Petroleum East China), Jie Zhang `[通讯]` (China University of Petroleum East China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了基于Segment Anything Model的ZMIS-SAM模型，并创建了ZMIS5K微观浮游动物实例分割数据集，实现了高质量的浮游动物微影实例分割。

**💡 创新点**

核心创新包括三大模块：ZM-ViT（形状与强度轻量化适配器）、NFAM（邻域特征聚合）和WM2FE（小波多尺度多方向增强），以及首次构建针对浮游动物的高质量微影实例分割数据集。

**🔧 技术方法**

采用了SAM、ViT-H骨干、轻量级适配器、邻域特征聚合技术、二维小波变换多尺度多方向特征增强，并在MMDetection框架中实现。

**📊 数据集**

使用了自研ZMIS5K数据集（5358张图像，47种浮游动物，10228实例），以及公开跨域数据集进行泛化验证。

**📈 对比分析**

通过与14种传统与SAM基SOTA模型的对比实验，ZMIS-SAM在ZMIS5K上达mAP 73.6%、AP50 94.6%、AP75 80.7%，显著优于对比模型，并在跨域测试中保持强泛化性能。

**⚠️ 局限性**

局限在于目前无法对浮游动物的头部、触角等细部结构进行单独分割，未来计划引入多模态文本信息实现更细粒度的分割。

---

## 180. Learning Color Grading, No Photo Sharing: Federated Aesthetic Preference Learning for Personalized Image Enhancement

**arXiv ID:** 2607.27659 | [PDF](https://arxiv.org/pdf/2607.27659v1)

**作者:** Chuanzhi Xu `[一作]` (University of Sydney), Weidong Cai `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出FedPAIE框架，实现在保持用户隐私的前提下的个性化审美图像增强。

**💡 创新点**

核心创新包括联邦学习的审美偏好初始化、双线索轻量化审美评分器、冻结评分器引导的CLUT适配以及多项正则化平衡偏好与图像保真度。

**🔧 技术方法**

使用联邦学习、CLUT-Net、MobileNetV3-Large特征提取、双线索评分器、配对与非配对训练、对比损失、差距惩罚等技术。

**📊 数据集**

在MIT‑Adobe FiveK和Flickr‑AES两大数据集上进行训练与评估。

**📈 对比分析**

通过与集中式、按用户调参、以及SpliNet、PieNet等现有方法对比，FedPAIE在10/100次标注下在PSNR、SSIM、LPIPS等指标上均优于基线，且模型参数仅0.293M。

**⚠️ 局限性**

限制包括对用户标注的依赖、在极少标注时效果受限、仅处理颜色分级而不涉及空间细节、可能对极端风格适应不足。

---

## 181. From Single- to Cross-Document: Benchmarking Multi-Granularity Event Analysis of Large Language Models

**arXiv ID:** 2607.27654 | [PDF](https://arxiv.org/pdf/2607.27654v1)

**作者:** Tao Wen `[一作]` (University of Electronic Science and Technology of China), Ke Qin `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MiGUE-Bench和MiGUE-Pipeline，用于多粒度事件分析的评估与数据构建。

**💡 创新点**

创新点在于四种跨文档与跨粒度任务的系统设计、LLM驱动的自纠注释管线以及对LLM跨文档事件推理能力的全方位评估。

**🔧 技术方法**

采用LLM自纠、检索增强反思、约束推理、社区检测等技术，并结合LightRAG/UltraRAG等RAG框架进行实验。

**📊 数据集**

基于公开新闻与近两年中国官方/主流媒体报告构建的数据集，MiGUE-Bench共包含约3,300条样本，涵盖单文检测到跨文预测等任务。

**📈 对比分析**

对比多款闭源/开源LLM与RAG方法，GPT‑5.2‑Pro与Claude‑4.5‑Opus表现最优；跨文任务更具挑战性，RAG提升不稳定，LLM在因果与时间跨度推理上仍显不足。

**⚠️ 局限性**

局限在于高度依赖大型闭源模型，跨文推理受噪声干扰，时间跨度与因果建模缺乏精细化与数学化，RAG策略尚未达到最优。

---

## 182. Certifying when decision-time information justifies adaptive experimentation

**arXiv ID:** 2607.27651 | [PDF](https://arxiv.org/pdf/2607.27651v1)

**作者:** Jia Bi `[一作]` (Science and Technology Facilities Council), Chenyang Zhu `[通讯]` (University of Southampton)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种在实验前通过预先签订的合同决定是否允许自适应实验的框架，结合风险控制、执行价值和非平凡激活的多重校验，实现对自适应实验机会的认证。

**💡 创新点**

创新点包括：①构建完整的授权链，先判断能否识别机会、可证实、能非平凡执行再评估价值；②给出源到目标的不完全可识别性不可证实边界；③通过目标校准恢复非平凡的积极分支；④设计精确的有限样本认证边界和风险控制门控；⑤将风险、价值、非平凡激活统一为可并行验证的合同条款。

**🔧 技术方法**

技术手段包括：信息机会理论、Blackwell‑garbling条件、风险控制的Clopper‑Pearson上限、交叉拟合的价值模型与残差专家、并行置信区间、精确的有限人口似然推断、随机抽样校准与贝叶斯校准、以及多阶段安全包装。

**📊 数据集**

使用的数据集：公开的 Cell Painting cpg0012（11,265 复合物），公开的 CTRP v2 药理学实验（254 家族），以及作者自建的合成模拟实验（包含 16‑银行、200‑银行等不同规模）。

**📈 对比分析**

与六种对比方法（始终回退、强制激活、风险门控、阈值门控等）比较。目标校准门控在保持 5.18% 误激活率、捕获 5.88% 正向机会、并实现正执行价值的同时，成为唯一满足合同全量要求的方法。其他方法因误激活、缺乏非平凡激活或未通过风险控制而失败。

**⚠️ 局限性**

局限性包括：需在实验前预先设定并冻结合同，无法直接应用于实时实验；目标可识别性和有限样本约束使得某些实验场景无法得到认证；风险-价值折衷仍需要手动权衡；实验在离线设置下验证，真实实验环境中仍需进一步验证。

---

## 183. Dynamic Exploration Graph: A Novel Approach for Efficient Nearest Neighbor Search in Evolving Multimedia Datasets

**arXiv ID:** 2607.27640 | [PDF](https://arxiv.org/pdf/2607.27640v1)

**作者:** Nico Hezel `[一作]` (HTW Berlin), Klaus Jung `[通讯]` (HTW Berlin)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了动态探索图（DEG），用于高效近邻搜索，支持在不断演化的多媒体数据集上进行实时查询。

**💡 创新点**

提出了完整的顶点删除算法保证图连通性，并开发了无分布参数的图扩展方法，使得动态图保持平衡与连通。

**🔧 技术方法**

基于连续精细化探索图（crEG）原理，结合BFS子图恢复、MRNG检查、范围搜索和邻居替换等技术构建与维护图结构。

**📊 数据集**

在公开高维向量数据集SIFT1M、Deep1M和GloVe上进行实验评测。

**📈 对比分析**

与HNSW、DiskANN、SWINN等现有算法在静态、流式、在线三种场景下对比构建时间、查询速度（QPS）和召回率，DEG在构建时间最短、查询速度最快且召回率≥95%。

**⚠️ 局限性**

未在极大规模分布式环境中测试，且在高删除比例场景下仍需进一步优化删除与重连效率。

---

## 184. First-order Constrained Trilevel Optimization Over Distributed Networks for Robust Coreset Selection

**arXiv ID:** 2607.27632 | [PDF](https://arxiv.org/pdf/2607.27632v1)

**作者:** Yang Jiao `[一作]` (Southeast University), Li `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种分布式鲁棒 coreset 选择框架 F^2CTO，能够在多节点网络中通过层次化的梯度投影实现鲁棒子集的协同优化；

**💡 创新点**

其创新点在于将核心选择、鲁棒优化与分布式学习统一为受约束的三级优化问题，首次给出了针对此类受约束三级优化的单循环一阶分布式算法，并提供了 𝒪(ϵ^{-3/2}) 的非渐进收敛率；

**🔧 技术方法**

主要技术包括：层次复合值函数重构（将三级结构转化为约束的双层问题）以及分布式交替投影梯度（避免显式求超梯度、只使用一阶信息）；

**📊 数据集**

实验使用了 Permuted MNIST、Split CIFAR‑100、Tiny‑ImageNet 以及规模达 200 机器的 Edge‑IIoT 数据集；

**📈 对比分析**

与 FedCS、GCFL、ACS、BCSR、Greedy Coreset、AFTO、DTZO 等方法以及 ADBO、FEDNEST 等二级/三级优化基线对比，F^2CTO 在 FGSM/PGD/AutoAttack 三类攻击下平均鲁棒性分别达到约54%、51%和48%，显著优于现有方法；

**⚠️ 局限性**

局限性包括：需要在每轮通信中上传模型参数，通信成本在超大规模网络或高维模型时仍可能较高；且仅在连续学习任务和三种攻击下验证，对更复杂的鲁棒性场景和非凸设置的鲁棒性尚待进一步研究。

---

## 185. Hidden APIs in Language Models: Discovering Reusable Causal Interfaces from Forked Futures

**arXiv ID:** 2607.27617 | [PDF](https://arxiv.org/pdf/2607.27617v1)

**作者:** SiYuan Ma `[一作]` (Nanyang Technological University), Qixin Zhang `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出“forked futures”方法，通过对隐藏状态的未来操作分布进行探测，构建可复用的内部接口；

**💡 创新点**

创新点在于以未来操作的分布签名为准，构造无标签的因果等价性，并将共享接口与局部、混合、分布式三种架构在预序因果MDL下进行竞争，首次量化共享接口的优势；

**🔧 技术方法**

使用了预序因果最小描述长度（causal MDL）、未来签名距离、迁移植入、介导路径阻断等技术，结合大型语言模型的前缀隐藏状态与后续操作；

**📊 数据集**

实验采用了 Qwen2.5-1.5B、Llama-3-8B 等模型，在关系、程序、证明/论述三大任务族（共计约10,600 条训练样本）以及额外的多模型背骨（Qwen2.5-7B、Mistral-7B 等）进行评估；

**📈 对比分析**

通过比较共享、局部、混合、分布式四种接口的MDL与FSD指标，发现共享接口在 MDL 上分别比最优非共享架构降低约0.216 与 0.294 nats，且在多种 OOD、植入与介导测试中保持最佳或相近的性能；

**⚠️ 局限性**

局限性包括仅在预先指定的任务族和隐藏状态层进行评估，模型架构、层级和操作集合有限，且在某些“非共享→共享”误判与自然族外推广方面仍有不足。

---

## 186. DualAnchor: Preserving Language Priors and Improving Lexical Fidelity in Gloss-Free Sign Language Translation

**arXiv ID:** 2607.27614 | [PDF](https://arxiv.org/pdf/2607.27614v1)

**作者:** Hongbin Zhang `[一作]` (Harbin Institute of Technology), Kehai Chen `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出DualAnchor框架，在无词形标记的手语翻译中通过双重锚点实现语言流畅与视觉忠实的生成。

**💡 创新点**

创新点在于同时引入Token‑level Prior Anchoring保留LLM语言先验，并使用Optimal Transport Alignment实现视觉与文本的细粒度对齐，二者共同提升流畅度与词汇准确性。

**🔧 技术方法**

采用冻结的LLM与自回归前缀进行逆KL正则化，配合熵正则化的部分最优传输（Sinkhorn算法）进行视觉‑文本匹配，形成训练时仅使用的双重约束。

**📊 数据集**

在PHOENIX‑2014T（德语手语）和CSL‑Daily（中文手语）两大公开数据集上进行训练与评估。

**📈 对比分析**

与GFSLT‑VLP、SpaMo、BeyondGloss等多种基线对比，DualAnchor在PHOENIX‑2014T上实现BLEU‑4/1‑4/2‑4/3‑4/ROUGE‑L为53.93/41.28/33.18/27.60/50.62，CSL‑Daily为53.15/39.84/30.65/24.21/49.46，BLEU‑4均为同类方法最高，显示显著提升。

**⚠️ 局限性**

局限在于仍依赖大规模LLM后端，训练成本高；仅在两种语言数据集验证，未检验对低资源或噪声环境下的鲁棒性。

---

## 187. Beyond Similarity: Grounded Agentic Extraction and Expert-Adjudicated Evaluation of Intertextuality in Classical Chinese Histories

**arXiv ID:** 2607.27595 | [PDF](https://arxiv.org/pdf/2607.27595v1)

**作者:** Zhaoji Wang `[一作]` (Peking University), Jun Wang `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了基于大型语言模型的可验证跨度定位与多维标签互文性提取框架，并在《二十四史》上实现大规模自动化提取；

**💡 创新点**

将细粒度互文性转化为受工具约束的执行任务，实现可写时验证与可审计的提取；构建专家判定的黄金基准并揭示维度可靠性梯度；展示跨18世纪引用结构的稳定性与字面忠实度的逐步下降；

**🔧 技术方法**

使用多种大型语言模型（Claude、DeepSeek、OpenAI、Gemini、Qwen、Zhipu 等）结合定位、提交、校验等工具接口；采用五维标签体系（形式、维度、标记、功能、立场）与置信度校准机制；

**📊 数据集**

以《论语》与《汉书》共 2,400 对章节为验证集，构建 2,533 组黄金互文对；将《论语》与《二十四史》全文共 65,380 对章节用于大规模应用；

**📈 对比分析**

在验证集上对 12 种 LLM 进行精度、成本、置信度校准评估，最高精度达 92.9% 但成本最高，最佳成本-精度比由 deepseek‑v4‑flash 获得；在 24 史规模下生成 5,766 对互文，按校准推算误报约 12% 以内；

**⚠️ 局限性**

仅在《论语–汉书》上验证，迁移到其他文本需再评估；模型可能受已有学术知识影响；专家一致性仅在易识别维度高，功能/立场维度争议；使用单一编年史日期忽略多年代差异；系统对 LLM 可用性与成本敏感。

---

## 188. MeshFM: 2D Features Are All You Need for 3D Shape Understanding

**arXiv ID:** 2607.27592 | [PDF](https://arxiv.org/pdf/2607.27592v1)

**作者:** Jinfan Zhou `[一作]` (University of Chicago), Rana Hanocka `[通讯]` (University of Chicago)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 MeshFM 模型，通过将 2D 基础模型的特征提炼为 3D 教师场景，训练一个前向网络以在不需要优化的情况下为任意 3D 形状预测高质量特征，支持零样本分割、对应和变形。

**💡 创新点**

① 利用 SAM 的精确分割掩码进行特征抗锯齿校正，解决传统 ViT 特征的边界模糊；② 两阶段训练，将稀疏的 2D 监督转化为连续 3D 特征字段，然后再用前向网络回归，解耦推理与训练；③ 对 SO(3) 旋转进行数据增强，使特征对任意姿态鲁棒。

**🔧 技术方法**

基于 DINOv2/Radio 等 2D 基础模型进行特征提取；DIB 的基于 barycentric 的特征场优化；SAM 分割掩码进行特征修正；PVCNN+triplane transformer 实现前向 3D 特征预测；SO(3) 随机旋转增强；采用 2D 视图渲染与反投影的方式。

**📊 数据集**

训练时使用多视图渲染的形状集合（未具体列举）；评估使用 PartObjaverse‑Tiny、PartNetE、TOSCA、DenseCorr3D、Manifold40、Part‑Objaverse 等。

**📈 对比分析**

与任务专用基线 PartField（分割）、DenseMatcher、Diff3F（对应）进行零样本对比；在分割任务中 mIoU 与基线相当或略优，且在旋转增强版上显著优于对手；对应任务中误差与 AUC 与基线相当，旋转鲁棒性更强；分类任务中通过轻量 MLP 取得更高准确率；变形任务中实现平滑语义‑aware 变形。

**⚠️ 局限性**

依赖所用 2D 基础模型的性能，例如 DINOv2 难以区分左右，导致对应时偶尔左右交换；特征仍受 2D 监督的限制，若基础模型改进可进一步提升；对极端遮挡或无纹理表面等情况的鲁棒性尚待验证。

---

## 189. Back from the Future: Key-Value Cache Management by Counter-Causal Surprise

**arXiv ID:** 2607.27600 | [PDF](https://arxiv.org/pdf/2607.27600v1)

**作者:** Stephen Gould `[一作]` (Metacognition AI), Anton van den Hengel `[通讯]` (Metacognition AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于反因果惊讶（counter‑causal surprise）的 KV 缓存剔除策略，用来在长上下文推理时减少 GPU 内存占用。

**💡 创新点**

创新点在于：①利用模型已计算的 KV 对，使用上三角注意力掩码直接评估过去 token 在后续上下文中的可预测性，从而得到剔除分数；②提出仅在最后一层进行反因果推理的快速近似，显著降低刷新成本。

**🔧 技术方法**

技术方法包括：原始顺序前向推理重用 KV，改用上三角掩码计算 P(x_i | x_{i+1}:t)，再取对数或 logits 得到惊讶分数；快速近似则在刷新时只用上一层隐藏状态 H^{L-1} 计算相同上三角注意力；实现基于块刷新（chunk）和自回归解码的完整推理框架。

**📊 数据集**

实验数据集涵盖 MATH500、AIME、LongHealth、Qasper、LoCoMo 等长文本推理任务，使用 Qwen2.5（3B/7B/14B）和 LLaMA 3.1（8B）等开源 LLM。

**📈 对比分析**

与全缓存、滑动窗口、重要性采样、H_2O 等基线比较，counter‑causal 方法在大多数任务和模型上取得与全缓存相当或更好的准确率；快速近似在刷新时间上比完整反因果快 7–9 倍，且性能损失仅 1–2%。

**⚠️ 局限性**

局限性包括：完整反因果刷新需要 O(L·n²) 计算，导致预填阶段成本翻倍；快速近似需额外存储最后一层激活；反因果惊讶是对 P(x_i | x_{i+1}:t) 的近似，可能忽略短句或单词级重要信息。

---

## 190. DAS-PMVC: A Framework for Partial Multi-View Clustering via Dual Alignment and Structure Enhancement

**arXiv ID:** 2607.27761 | [PDF](https://arxiv.org/pdf/2607.27761v1)

**作者:** Shubin Ma `[一作]` (Dalian University of Technology), Yu Shao `[通讯]` (Dalian University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种针对部分视图对齐的多视图聚类框架DAS-PMVC，解决不同视图间样本缺失和不对齐问题；

**💡 创新点**

双重对齐策略（先用锚点图对齐，再用Hungarian与对比学习进一步校正）以及结构增强特征学习，充分利用视图间结构一致性与语义相关性；

**🔧 技术方法**

锚点图构造、图卷积网络、对比学习损失、Hungarian算法、结构过滤与结构对齐损失；

**📊 数据集**

Caltech20、BDGP、Scene-15、Aloi、3Sources、BBCsports等六个多视图数据集；

**📈 对比分析**

与八种主流部分对齐聚类算法（PVC、MvCLN、EGPVC、ProImp、TCLPVC、EAGCP、AE2-Nets、Cmib-Nets）在ACC、NMI、ARI三指标上进行对比，DAS-PMVC在大多数数据集上均取得最高或次高成绩，尤其在ACC上提升显著；

**⚠️ 局限性**

对结构关联弱的数据（如BDGP）表现欠佳，且在类别过多时易出现NMI、ARI下降；

---

## 191. EgoGVAE: Ego-body Mesh Reconstruction via Guided Variational Autoencoder

**arXiv ID:** 2607.27755 | [PDF](https://arxiv.org/pdf/2607.27755v1)

**作者:** Jaehun Jung `[一作]` (Konkuk University), Wonjun Kim `[通讯]` (Konkuk University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究提出了一种仅凭头部轨迹即可重建全身网格的模型EgoGVAE，利用运动到运动网络的潜空间引导头部到运动网络生成自然姿态。

**💡 创新点**

创新点在于：①将运动到运动网络（VAE）作为引导器，使两网络潜空间对齐；②使用可学习的tokens补充未观测身体部分；③在训练阶段仅通过潜空间对齐实现一跳采样，显著提升推理速度。

**🔧 技术方法**

采用Transformer‑based变分自编码器、对称KL散度对齐潜空间、重建损失、速度损失、正则化KL，整体以PyTorch实现。

**📊 数据集**

主要使用AMASS和RICH基准数据集进行训练与评估，也在真实设备噪声的EgoBody数据集上验证。

**📈 对比分析**

与AvatarPoser、EgoPoser、EgoEgo、EgoAllo等基线比较，EgoGVAE在MPJPE、PA‑MPJPE、Ground、T_head、Jitter、Foot‑sliding等指标上均有显著提升；在推理速度上比扩散模型快50+倍，参数和FLOPs更低。

**⚠️ 局限性**

局限性包括：仍需大量标注头部轨迹与全身姿态的数据；在极端头部运动或极稀疏轨迹下可能出现姿态不稳；以及对长序列的实时推理虽然可行，但仍比专门设计的在线模型略慢。

---

## 192. Delegated Fair Division

**arXiv ID:** 2607.27743 | [PDF](https://arxiv.org/pdf/2607.27743v1)

**作者:** Argyrios Deligkas `[一作]` (Royal Holloway University of London), Evangelos Markakis `[通讯]` (Athens University of Economics and Business)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种双层公平分配模型：先将不可分割物品分配给中心（如食品捐赠中心），再由中心将物品分配给其下属代理人，并研究如何在中心层和代理人层同时满足基于羡慕的公平性（EF1、EFX）。

**💡 创新点**

创新点在于：1）首次系统地定义并区分中心层和代理人层的羡慕概念；2）给出多种中心评价函数（基于束、基于项、潜在/实现价值）并证明其单调性；3）在不同信息结构（全局 vs 局部）下提供多种多层公平分配算法；4）利用水平轮转（Horizontal Round‑Robin）和改进的Yankee Swap（多层路径增广）在特殊价值设置下实现 EF1/EFX；5）给出不可行性和 NP‑硬度结果。

**🔧 技术方法**

主要技术包括：水平轮转（HRR）与其变体、B‑组轮转、EFX‑分区轮转、路径增广（改进的 Yankee Swap）以及中心羡慕图用于确定遍历顺序；同时利用匹配与最大匹配扩展理论保证潜在价值的可计算性。

**📊 数据集**

文章为理论研究，未使用实测数据集，所有结果均在抽象的可分割物品和代理人集合上进行证明与分析。

**📈 对比分析**

方法比较以存在性与多项式时间复杂度为评判标准。论文证明在多种价值假设下（加性、已排序、相同、二值）可在多项式时间内获得 EF1/EFX/EF1/EFX 的双层公平分配；在其他更一般情况下则给出不可能性或 NP‑硬度。

**⚠️ 局限性**

局限性包括：1）中心评价函数在束基潜在模式下难以处理，且在非加性情形下结果未完全覆盖；2）仅对两层结构提出方案，更多层次和中心规模不等的情况留待后续研究；3）实验验证缺失，仅有理论证明；4）在某些特殊价值设置下仍无法同时实现 EFX 与 EF1。

---

## 193. Improving the Robustness/Accuracy Tradeoff Against Adversarial Attacks Using Information Bottleneck Distillation Through Dual Teachers

**arXiv ID:** 2607.27737 | [PDF](https://arxiv.org/pdf/2607.27737v1)

**作者:** Vincent Ryusuke Takahashi `[一作]`, Kave Salamatian `[通讯]` (Université Savoie Mont Blanc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了两种基于信息瓶颈蒸馏（IBD）的双教师蒸馏方法（Double Distillation和Joint Distillation），并在此框架中引入了仅使用干净样本训练的清洁教师。

**💡 创新点**

创新点在于同时利用清洁教师的软标签和特征来提升学生模型在自然样本上的准确率，同时保持与对抗训练教师相似的鲁棒性；并通过跨层注意力矩阵实现两教师特征的联合蒸馏。

**🔧 技术方法**

采用信息瓶颈蒸馏、对抗训练（TRADES、AWP）、跨层注意力加权特征匹配、双教师（清洁+鲁棒）蒸馏损失等技术。

**📊 数据集**

使用CIFAR-10和CIFAR-100两个标准图像数据集进行实验。

**📈 对比分析**

将DD和JD与IBD以及多种最先进的鲁棒训练/蒸馏方法（AT、TRADES、ARD、RSLAD、InfoAT、HBaR、B-MTARD）在干净准确率、对抗准确率和谐波均值等指标上进行对比。实验表明，DD和JD在保持鲁棒性相近的情况下显著提升了干净准确率，并在谐波均值上超过了IBD和大多数对比方法，尤其在较大容量学生模型上表现更佳。

**⚠️ 局限性**

局限性包括：训练时间和参数量显著增加（约1.5倍），且对小容量学生模型的提升有限；在某些正则化设置下注意力权重趋向均匀，可能降低蒸馏效果。

---

## 194. PrintAnything: Learning an Intermediate Representation for 3D printing G-code Generation

**arXiv ID:** 2607.27729 | [PDF](https://arxiv.org/pdf/2607.27729v1)

**作者:** Sangmin Hong `[一作]` (Seoul National University), Kyoung Mu Lee `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种完全无网格的3D打印流水线PrintAnything，直接将未定向点云转换为可执行的G‑code。

**💡 创新点**

创新点在于：1) 切片投影策略将点云映射为切片对齐的二维表示；2) G‑plan地图（占据、区域、流量三张图）统一描述几何与挤出属性；3) 结合多切片上下文和FiLM条件的U‑Net架构，实现端到端预测。

**🔧 技术方法**

使用了Point Transformer V3作为全局点云编码器，U‑Net+FiLM进行切片级预测，配合多切片条件、流量预测及模板推荐器，最终生成G‑code。

**📊 数据集**

主要使用Slice‑100K数据集（包含30k点云、对应G‑code），并在此数据集上进行训练与评估。

**📈 对比分析**

与传统“点云→网格→切片→G‑code”流程（Poisson、DWG、MeshAnything）以及多种基线进行对比，PrintAnything在Chamfer距离、3D F1、2D F1等指标均领先，表现最佳。

**⚠️ 局限性**

局限性包括：仅针对FDM打印机和单一材料；对极端噪声/稀疏点云的鲁棒性仍有限；缺乏对多材料、多路径复杂度等真实工艺约束的支持。

---

## 195. A Graph Matching Based Approach for the Multi-Depot Capacitated Vehicle Routing Problem

**arXiv ID:** 2607.27727 | [PDF](https://arxiv.org/pdf/2607.27727v1)

**作者:** Jayant Chandwani `[一作]` (BITS Pilani K K Birla Campus), Anand Narasimhamurthy `[通讯]` (BITS Pilani K K Birla Campus)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了两种基于图匹配的多仓库有容量车辆路径规划算法：Cluster-First 与 Match-First。

**💡 创新点**

创新点在于将 MDCVRP 转化为最小权重匹配问题，证明在两目标规模下可多仓库精确解，且在结构化情形下为 2 近似；匹配最优等价于组合拍卖最优。

**🔧 技术方法**

采用最小权重匹配（Edmonds’ Blossom / LEMON）、组合拍卖、整数线性规划、最大独立集以及迭代 Merge 的多目标扩展等技术。

**📊 数据集**

使用随机生成的二维欧氏实例：100 个基准数据集（每个 100 目标 10 仓库）以及 50 个规模更大数据集（每个 1000 目标 20 仓库）。

**📈 对比分析**

与最强基线组合拍卖对比，匹配方法在两目标和四目标场景下仅差 0–6% 行程距离，却快 2–3 个数量级；在 1000 目标规模下跑速提升到 2–3 个数量级，行程距离相当甚至略优。

**⚠️ 局限性**

局限在于理论保证目前仅覆盖至四目标（Scenario 2）并且仅适用于同质车队；对异构车队、时间窗、动态新客户的进一步评估与证明仍待完成。

---

## 196. SpatialCLI: Learning to Reason With Spatial Tools, Then Without Them

**arXiv ID:** 2607.27703 | [PDF](https://arxiv.org/pdf/2607.27703v1)

**作者:** Yang Zhou `[一作]` (Zhejiang University), Yuxiang Cai `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SpatialCLI 框架，借助专家视觉工具让 VLM 进行工具调用、训练工具使用策略，并将成功的工具使用轨迹转化为监督信号，使模型内化这些视觉能力；同时构建了 516 条样本的 SpatialCLI-Bench benchmark。

**💡 创新点**

创新点包括：① Call–Learn–Internalize 三阶段流程，① 用 RL 学习多工具的调用与使用策略；② 通过轨迹导向的内部化，将工具调用轨迹逐步语义化为可监督的感知推理链；③ Dual‑View 训练平衡工具使用与能力内化。

**🔧 技术方法**

技术手段：VLM（Qwen3‑VL‑8B、Qwen3.6‑27B、Qwen3.6‑35B）+ 专家工具（Locate、Segment、Depth、Pose）；Cold‑Start SFT + GRPO 强化学习；结构化工具返回；进化式轨迹转写；Dual‑View 训练框架。

**📊 数据集**

使用的数据集：SpatialCLI‑Bench（516 条 6‑选项 VQA，包含定位、分割、深度、姿态），MindCube、MMSI、DA‑2K、BOPASK 等；训练集包括 Qwen3.5‑397B‑A17B 生成的工具调用轨迹。

**📈 对比分析**

与 GPT‑5.6 Sol、Gemini 3.1 Pro、Qwen3.7‑Plus、Qwen3.5‑397B‑A17B、SpaceTools、AlloSpatial 等基线对比；在 SpatialCLI‑Bench 上，使用工具的 SpatialCLI‑8B 取得 91.3%，工具‑free 72.7%；总体上在多项空间/实体推理基准上显著优于基线，显示工具使用与能力内化可以共存并相互提升。

**⚠️ 局限性**

局限性：目前仅能内化结构化的感知输出（坐标、边界、多边形、深度、姿态），对多模态输出（图像生成等）尚未覆盖；依赖专家工具的覆盖率与可靠性；尚未实现完整的感知‑动作闭环。

---

## 197. LabEvolver: Training-Free Experience Evolution for Safe and Grounded Wet-Lab Agents

**arXiv ID:** 2607.27690 | [PDF](https://arxiv.org/pdf/2607.27690v1)

**作者:** Jingya Wang `[一作]` (Peking University), Yuyang Liu `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 LabEvolver 双循环框架，将无训练的在线闭环执行与后置经验演化结合，实现安全且可自我改进的实验室机器人。

**💡 创新点**

创新性地引入基于实验状态的三层安全门、经验经理 Strategist 以及状态配对经验存取，使得完成轨迹可被提炼为可重用的技能、策略与安全经验，从而实现学习-做-再学闭环。

**🔧 技术方法**

利用大模型进行高层规划、LabSkill 三级动作接口、RGB‑D 观察器、Tri‑layer 运行时安全门、Ebbinghaus 记忆衰退等技术，框架本身不更新模型权重。

**📊 数据集**

采用真实实验室物理数据（pH 调节、定量倒水、耦合 pH–EC 试验）以及 ALFWorld 长时限仿真任务集进行评估。

**📈 对比分析**

与 Act、ReAct、Inner‑only、MemP 等基线对比；在 ALFWorld 上 Success@20 提升至 91.4%，在实验室任务中 pH 调节完成时间下降 48.2%、安全拦截下降 60%，任务成功率从 76.2% 提升至 91.4%。

**⚠️ 局限性**

受限于人工设计的 LabSkill 与安全规则，难以自适应未知操作与新故障模式，且框架不更新模型权重，需要人工扩展技能。

---

## 198. Evaluating and Pricing Advertisements in AI-Generated Responses

**arXiv ID:** 2607.27686 | [PDF](https://arxiv.org/pdf/2607.27686v1)

**作者:** John L. Turner-Smith `[一作]` (Carnegie Mellon University), Tonghan Wang `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了基于心理学代理模拟的点击意图评估框架，并提炼出可微分的共享瓶颈评估器，用于LLM嵌入广告的用户价值与商业价值评估。

**💡 创新点**

创新点在于：①使用可复现的代理标签生成无日志的点击意图监督；②设计共享瓶颈结构与Earth Mover's Distance训练实现连续可微分评估；③将评估结果直接用于设计真诚支付的拍卖机制。

**🔧 技术方法**

技术包括：心理学驱动的代理模拟、Frozen Qwen3‑4B+LoRA共享瓶颈网络、EMD损失、行为扰动验证、真诚拍卖理论推导。

**📊 数据集**

数据集为NaiAD的58,999条广告嵌入回复，及30人设的代理标签，103个全新产品进行泛化测试。

**📈 对比分析**

与三种零射击前沿LLM（Qwen3.6‑35B‑A3B、Claude Sonnet 4.6、GPT‑5.5）对比，评估器在相关性敏感度上达79%比60‑67%，对内容降解呈连续下降，泛化到全新产品保持准确；在人类对比中与多位评标者的偏好一致率为86‑96%。

**⚠️ 局限性**

限制在于：代理标签尚未经过大规模人类验证；输出是离散评分而非真实CTR，需后续A/B校准；标签过度依赖相关性门控，个性先验不完整；缺乏真实点击日志导致无法直接映射到实际点击概率。

---

## 199. Event-Structured Physics-Informed Neural Networks for Differentiable Critical Clearing Boundaries

**arXiv ID:** 2607.27681 | [PDF](https://arxiv.org/pdf/2607.27681v1)

**作者:** Baoli Hao `[一作]` (Illinois Institute of Technology), Ren Wang `[通讯]` (Illinois Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种事件结构化的物理信息神经网络（ES‑PINN），通过硬链预故障、故障中、清除后三个阶段的动力学，并直接从学习的轨迹中推导出可微的临界清除时间（CCT）边界。

**💡 创新点**

创新点包括：① 用硬链方式保证状态连续性并精确保留清除时的右侧导数；② 通过 Taylor‑gate 结构将物理动力学局部展开与神经校正结合，提升对事件界面的表示；③ 推导了残差→轨迹→CCT 的误差链，证明残差正则化对边界精度的决定性作用；④ 通过隐式微分得到 CCT 的局部灵敏度，并提供可直接查询的 scalar 读出。

**🔧 技术方法**

使用的技术主要是：物理信息神经网络（PINN）与其改进版 XPINN；混合事件结构化神经网络；Taylor‑gate 诱导偏差；残差正则化；对抗式损失；梯度求解与 root‑refine；以及后置的 scalar readout 与 GPU 加速推理。

**📊 数据集**

使用的数据集为 IEEE 9、14、30 节点电力系统的 MATPOWER 案例，构造了多种故障类型（电压、线路、机械功率扰动）及其清除策略，生成了 (α,t_c) 网格下的轨迹与 CCT 数据。

**📈 对比分析**

与传统单一 PINN、软链 XPINN 以及物理信息 DeepONet 进行对比。ES‑PINN 在轨迹 RMSE、CCT MAE 上分别提升约 15‑20%（相对 XPINN）和 70‑80%（相对 DeepONet）。在多系统、多故障情形下保持了较小的误差，并在 GPU 上实现 12.8× 的推理加速。通过 ablation 研究证实硬链、右侧导数约束与残差正则化对性能的关键作用。

**⚠️ 局限性**

局限性包括：① 仅适用于已知的阶段性简化动力学（减型摆动方程）和已知事件规则；② 需要连续的发电机状态，无法直接处理多阶段或不连续的事件；③ 对高保真 DAE、未知或不确定事件尚未覆盖；④ 需要在每个故障类型上训练专门的专家网络，若故障种类繁多，模型规模会增大。

---

## 200. Improved RIP Bounds for Gaussian Partial Circulant Matrices

**arXiv ID:** 2607.27676 | [PDF](https://arxiv.org/pdf/2607.27676v1)

**作者:** Zhao Song `[一作]` `[通讯]`, Zhao Song

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并证明了在任意给定采样集下，高斯偏置循环矩阵的改进限制等距(RIP)上界。

**💡 创新点**

通过在Maurey熵步骤中结合非交换Khintchine不等式与控制m的Schatten矩阵阶数估计，将原先的log(2N)项替换为更小的log(em)，从而显著降低所需测量数。

**🔧 技术方法**

主要技术包括非交换Khintchine不等式、Schatten阶矩估计以及对随机过程的chaos-process分析。

**📊 数据集**

该研究为纯理论分析，不依赖具体数据集，而是考虑随机高斯向量g和任意大小为m的采样集合Ω。

**📈 对比分析**

通过比较新的RIP上界与Krahmer–Mendelson–Rauhut原有结果，显示在相同δ、η下需要的测量数m显著减少，尤其在m远小于N时表现更优。

**⚠️ 局限性**

局限性在于仅适用于高斯生成的偏置循环矩阵，并且证明仅给出概率上界，未给出实际构造或实验验证；此外对非高斯分布的推广仍需进一步研究。

---

## 201. Beyond the Best Teacher: Expanding and Compressing the Reasoning Solution Manifold

**arXiv ID:** 2607.27770 | [PDF](https://arxiv.org/pdf/2607.27770v1)

**作者:** Songshuo Lu `[一作]` (Moore Threads AI), Yaohua Tang `[通讯]` (Moore Threads AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种 expand‑then‑compress 框架，将 RL 训练得到的教师模型扩展为互补专家，然后通过 on‑policy 蒸馏压缩成单一更强的学生模型。

**💡 创新点**

将 RL 教师视为多盆地（multi‑basin）解决方案的局部探针；利用 Residual GRPO 动态分配未被覆盖的样本构造互补专家；通过 Reliability‑Gated Teacher‑Union OPD 与 Consensus‑Residual Decomposition 进行压缩，避免专业化被平均所抑制。

**🔧 技术方法**

采用 Residual Group Relative Policy Optimization (RGRPO) 进行教师扩展；Reliability‑Gated Teacher‑Union On‑Policy Distillation (TU‑OPD) 进行压缩；Consensus‑Residual Decomposition 用于保留专家的独特 token 偏好；温度化的教师加权与覆盖阈值控制。

**📊 数据集**

使用 Skywork‑OR1‑RL‑Data（数学推理与代码生成）和 IFBench 验证数据（指令跟随）；数学推理还评估 AIME、HMMT、AMC23 等竞赛数据集。

**📈 对比分析**

与单一 RL 教师、随机划分的并行 GRPO 教师、教师 Envelope 进行对比。Qwen3‑1.7B 学生在数学推理、代码生成和指令跟随分别提升 2.0%、8.3% 与 6.9%；Qwen3‑4B 学生在数学推理提升 6.9%；扩展阶段比并行 GRPO 提升 2.71 点 Macro @8 和 6.61 点 Macro Pass@8。

**⚠️ 局限性**

仅在可验证任务和固定教师集合上评估，未验证跨任务或多语言场景；超参数（覆盖阈值、温度）需手工调优；RL 奖励稀疏时仍可能导致探索不足；教师数量与规模的进一步扩展尚未探究。

---

## 202. DS@GT ARC at ImageCLEFmedical 2026: Architectural Diversity for Concept Detection and Foundation-Model Scaling for Caption Prediction in Medical Image Analysis

**arXiv ID:** 2607.27763 | [PDF](https://arxiv.org/pdf/2607.27763v1)

**作者:** Bowen Wang `[一作]` (Georgia Institute of Technology), Ritesh Mehta `[通讯]` (Georgia Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在ImageCLEFmedical Caption 2026挑战中，提出了概念检测和标题生成的多模型方案，分别获得概念检测第一名和标题生成第三名。

**💡 创新点**

创新点在于：① 采用“诚实阈值调优”策略防止长尾概念过拟合；② 构建三路视觉器融合（ConvNeXt‑V2、BiomedCLIP ViT‑B/16、DenseNet‑169）与KNN检索的轻量级对比；③ 对Gemma‑3 27B模型进行QLoRA微调并结合Beam搜索实现高质量医学标题；④ 通过概念锚定和Vizwins合并提升小模型的事实性。

**🔧 技术方法**

使用的技术包括：卷积与变换器视觉编码器、BiomedCLIP预训练、Logistic回归/轻量化神经网络、ASL损失、TTA、三路加权融合、Honest Threshold Tuning、KNN检索、QLoRA、Gemma‑3、InstructBLIP、LLaVA‑Med、BLIP、MedGemma‑4B、Beam搜索、概念锚定与多模型合并。

**📊 数据集**

数据集为更新扩展版的ROCOv2，包含约97k训练图像、19k验证图像、15k测试图像，标注UMLS CUI概念与医学标题，覆盖1,947个唯一CUI。

**📈 对比分析**

评估采用官方首/二级F1（概念检测）和Rel/Fact/Overall（标题生成）指标。三路视觉集成+诚实阈值调优在测试集上获得F1=0.5790（首位），KNN检索方案则获得F1=0.5780；Gemma‑3 27B微调+3‑beam得到Overall=0.3571（第三位）。

**⚠️ 局限性**

局限性包括：① 仍未充分解决极少量概念的识别；② 对于概念检测，模型对长尾概念的覆盖不足，需进一步探索少样本或层次化标签；③ 标题生成在小模型上仍依赖概念锚定与后处理，模型规模限制导致事实性与流畅性仍有差距。

---

## 203. Cocktail-Talker: Multi-Speaker Dialog Modeling in Noisy Social Environments with Turn Action GRPO

**arXiv ID:** 2607.27756 | [PDF](https://arxiv.org/pdf/2607.27756v1)

**作者:** Xilin Jiang `[一作]` (Columbia University), Nima Mesgarani `[通讯]` (Columbia University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Cocktail-Talker，训练一个在嘈杂社交环境中能根据多说话人混合音频做出“回答”“聆听”“忽略”三种回合动作的语音LLM。

**💡 创新点**

引入动作标记和三动作决策机制，并通过LLM生成的对话模拟管线Cocktail-DialogGen在大规模多说话人噪声场景下进行训练。

**🔧 技术方法**

基于Qwen2.5-Omni的思考-说话架构进行LoRA微调，结合GRPO强化学习优化动作决策，并用Gemini 3 Pro、Qwen3-TTS等LLM进行对话与语音合成。

**📊 数据集**

通过Cocktail-DialogGen生成约72k个带噪音混合的对话，覆盖18个已知环境、10个未见环境，包含3/4说话人、正式/随意语境和匿名/命名策略。

**📈 对比分析**

与Moschi、PersonaPlex、Step-Audio2、Kimi-Audio、Qwen2.5‑Omni、Qwen3‑Omni等基线比较，Cocktail-Talker在决策准确率和响应质量上分别实现宏F1≈0.93、METEOR≈0.19等显著提升，尤其在未见环境和低SNR条件下保持鲁棒。

**⚠️ 局限性**

当前系统非流式、假设固定回合边界，缺乏视觉或空间感知支持，实际部署仍面临识别说话人和添加语境推理的挑战。

---

## 204. Baikal: Structured Search for Deep Research over Data Lakes

**arXiv ID:** 2607.27726 | [PDF](https://arxiv.org/pdf/2607.27726v1)

**作者:** Dhruv Agarwal `[一作]` (University of Massachusetts Amherst), Andrew McCallum `[通讯]` (University of Massachusetts Amherst)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出将深度研究任务转化为预算搜索问题，利用语义聚类将湖中表格与文本划分为区域，并通过区块选择策略和基于LLM的质量打分实现高效的证据挖掘与报告生成。

**💡 创新点**

创新点在于将数据湖划分为语义区域并把区域选择视为多臂赌博机问题，利用LLM评估奖励驱动自适应探索与利用，从而显著提升覆盖度与结果可信度。

**🔧 技术方法**

使用了单一多模态嵌入编码器(Qwen3-Embedding)、BERTopic+UMAP+HDBSCAN进行聚类，基于Bayes-UCB、Bayes ε‑greedy、随机与LLM策略的区块选择，LLM评分打分与编码代理(OpenCode)执行SQL，并用GPT‑5‑mini做报告合成。

**📊 数据集**

实验基于HybridQA（10,993张表+227K条段落）和TAT‑QA（2,757张表+13K条段落）两个数据湖，采用15个深度研究查询进行评测。

**📈 对比分析**

与OpenCode、DeepSearcher等基线对比，最佳配置在HybridQA上报告分数提升28%，在TAT‑QA上提升36%，主要得益于语义区域的覆盖和探索策略的收益。

**⚠️ 局限性**

局限包括：对预算与聚类参数高度敏感；对LLM评分与策略的依赖导致算力与成本上升；评测仅覆盖两类数据湖，缺乏更广泛的领域验证。

---

## 205. Write-Safe Flow Field Mapping under Ambiguous Onboard Sensing and Localization Drift

**arXiv ID:** 2607.27713 | [PDF](https://arxiv.org/pdf/2607.27713v1)

**作者:** Linhao Jin `[一作]` (Iowa State University), Qiang Zhong `[通讯]` (Iowa State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

解决了移动机器人在流场映射中因观测别名和定位漂移导致的误写问题，提出了一种基于地图引用的保守融合框架。

**💡 创新点**

创新点在于将写入安全分数 κ_t 与信息度量分离，利用连续门控对不确定更新进行衰减，并在缺少可靠引用时允许地图初始化。

**🔧 技术方法**

采用了神经网络编码器+GRU+卷积解码器预测局部流补丁，并输出写安全分数、信息度量等；训练过程中以对齐误差为目标。

**📊 数据集**

数据集包括四种 FluidX3D 合成流场（单喷流、双喷流及其交叉流）以及零样本水槽实验中的压强和光流观测。

**📈 对比分析**

与无门控、EKF门控、硬门控和 Oracle‑κ 等基线比较，方法在合成环境下平均 Ghost 降低 42%，NRMSE 降低 50%，在硬件重放中 Ghost 降低 39% 并保持 81% 覆盖率。

**⚠️ 局限性**

局限在于对定位漂移的统计建模假设、缺少大规模真实场景验证、以及在高漂移下可能导致部分有效流信息被过度抑制。

---

## 206. LightRot: A Light-Weighted Rotation Scheme and Architecture for Accurate Low-Bit Large Language Model Inference

**arXiv ID:** 2607.27704 | [PDF](https://arxiv.org/pdf/2607.27704v1)

**作者:** Sangjin Kim `[一作]` (PIM Semiconductor Design Research Center), Hoi-Jun Yoo `[通讯]` (PIM Semiconductor Design Research Center)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了LightRot，一种针对低位精度大语言模型推理的轻量化旋转方案及其专用硬件加速器。

**💡 创新点**

创新点在于引入Grouped Local Rotation (GLR)与Outlier Direction Aligning (ODA)两种算法，配合分层Fast Hadamard Transform（FHT）实现低能耗、高精度的旋转量化。

**🔧 技术方法**

核心技术包括：旋转量化、GLR/ODA算法、分层FHT旋转单元、组量化（GQ）、整数化运算单元与浮点旋转单元的协同设计。

**📊 数据集**

主要使用WikiText-2（语言模型验证）与MT-Bench（对话评测）作为实验数据集，模型涵盖LLaMA2-7B/13B及LLaMA3-8B。

**📈 对比分析**

在与基线旋转、SpinQuant、Quarot等方法对比时，LightRot在4-bit推理下实现了5.73/5.08/6.98的PPL，MT‑Bench赢率超过50%，硬件方面达27.4 TOPS/W的能效，显著优于先前的高精度/混合精度加速器。

**⚠️ 局限性**

局限性包括：仍依赖预先校准的离散量化参数，且对极端异常值的处理依赖ODA的异向量化，未来需进一步降低对离散化的敏感性，并扩展至更大规模模型。

---

## 207. Distributed Point Functions and Function Secret Sharing

**arXiv ID:** 2607.27696 | [PDF](https://arxiv.org/pdf/2607.27696v1)

**作者:** Elette Boyle `[一作]` (NTT Research), Peter Scholl `[通讯]` (Aarhus University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了分布式点函数（Distributed Point Function，DPF）及其相关的函数秘密共享（FSS）技术，涵盖了DPF的定义、构造、扩展（如多方DPF、比较函数、多点函数）以及在隐私计算、私有信息检索、匿名消息等领域的应用。

**💡 创新点**

创新点在于系统化整理了DPF和FSS的理论基础、最新的PRG基础构造、树形结构实现以及与多方安全与可验证性相关的最新进展，为读者提供了一个统一的框架和对比视图，指出了目前的技术瓶颈和未解决的研究问题。

**🔧 技术方法**

主要技术包括基于伪随机生成器（PRG）的加法秘密共享、GGM树结构的树形DPF构造、利用线性同态与PRG同态的混合技术、信息论与计算安全的阈值设计、以及可验证性和可提取性等安全增强技术。

**📊 数据集**

本综述为理论调研性质，并未使用具体数据集进行实验；所述性能指标均来自已发表论文的理论分析与实验报告。

**📈 对比分析**

作者通过对比关键字大小、评估时间、通信开销等指标，比较了不同DPF构造（平方根DPF、树形DPF、多方DPF等）以及与其他FSS方案（如大状态DMPF、OKVS-based DMPF）的优势与不足，说明了在不同参数范围内哪类方案更高效。

**⚠️ 局限性**

主要局限包括：DPF在两方之外的多方构造仍受限于指数级密钥规模；尚无多方DPF在计算安全下接近信息论下的效率；对比函数、区间函数等扩展仍缺乏紧凑构造；可验证性与可提取性在更广泛模型下的实现仍是开放问题。

---

## 208. GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

**arXiv ID:** 2607.27694 | [PDF](https://arxiv.org/pdf/2607.27694v1)

**作者:** Sangjin Kim `[一作]`, Hoi-jun Yoo `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

暂无具体内容

**💡 创新点**

暂无创新点说明

**🔧 技术方法**

暂无技术细节

**📊 数据集**

暂无数据集信息

**📈 对比分析**

暂无方法对比与性能评估

**⚠️ 局限性**

本文缺乏可分析的实验与讨论，限制信息不足

---

## 209. Rehearse: Stepping Back from the Confidence Cliff in Self-Improving Autoresearch

**arXiv ID:** 2607.27687 | [PDF](https://arxiv.org/pdf/2607.27687v1)

**作者:** Jiazhen Ji `[一作]` (Tencent), Shouhong Ding `[通讯]` (Tencent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在可测量目标的 AutoSOTA 循环中，引入 Rehearse 技能，对多条候选代码修改先进行预执行判断，然后只执行最优者，并将结果写回专注的历史记忆。

**💡 创新点**

发现并量化了所谓的“confidence cliff”：随着成功修改累积，预执行判断的选择准确性急剧下降；通过聚焦记忆检索仅与当前候选相关的历史结果，显著恢复并提升了后期判断准确性。

**🔧 技术方法**

采用 pairwise 对比评判、strict‑consensus 过滤、聚焦记忆检索（只取相似度阈值内的历史记录）以及基准构建（从 AutoSOTA 日志生成 366 对同基线修改的结果对）。

**📊 数据集**

利用公开的 AutoSOTA 日志构建 39 篇论文派生任务的 366 对同基线修改，涵盖优化器、调度、架构、正则化、数据增强、损失等多种改动；实验环境包括 nanochat、CIFAR‑10 以及 ETTh1 预测任务。

**📈 对比分析**

与传统单候选或无记忆判断的自动化搜索进行对比；在 4000 次训练预算下，Rehearse 在 nanochat、CIFAR‑10 和 ETTh1 三个循环上分别取得约 10.7%、2.85% 和 54% 的性能提升，并显著降低种子间波动。

**⚠️ 局限性**

局限性包括：实验仅覆盖三种可测量目标循环；候选池规模和多样性未在更大规模上验证；记忆检索仅基于相似度阈值，可能忽略更广泛的上下文；未对更长时间步或更大规模的自动化研究流程进行评估。

---

## 210. Articulated Object Reconstruction from Rest-State Observation

**arXiv ID:** 2607.27749 | [PDF](https://arxiv.org/pdf/2607.27749v1)

**作者:** Daeun Lee `[一作]` (Seoul National University), Jaesik Park `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `4de8e9d8-757b-475f-9627-18a445e50202` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Rest-to-Articulation 框架，仅利用闭合状态观测即可重建多部件可动物体的几何和运动学结构。

**💡 创新点**

创新点在于通过显式网格作为统一基准，结合 VLM 与 SAM3 的互补纠错循环，利用视频扩散模型生成运动假设并在网格几何约束下优化关节参数，实现无运动观测的可动物体重建。

**🔧 技术方法**

主要技术包括 Mesh 预处理与视角选择、VLM+SAM3 的交叉验证与 2D 证据投影、图论标签传播、视频扩散模型（Wan2.2+VBVR LoRA）生成动作序列、基于 2D 轨迹的关节类型判定与参数拟合、以及网格闭合与体积化。

**📊 数据集**

使用 Articulated Containers Dataset（ACD）(来自 Habitat Synthetic Scenes 与 Amazon Berkeley Objects）、MultiScan 实景 RGB‑D 数据，以及在线产品图像、Replica 与 ScanNet++ 等多模态输入进行验证。

**📈 对比分析**

与基线分割方法（PartField、Find3D、SAMesh）相比，分割精度提升明显；与基线关节估计方法（Articulate AnyMesh、ArtGS、REArtGS 等）以及生成式方法（URDFormer、Singapo）对比，关节类型准确率最高、轴向误差与基线相当，且在合成与真实数据上均保持稳健。

**⚠️ 局限性**

局限性包括对闭合状态的强假设；对复杂多轴链路的适应性待验证；视频生成可能出现幻觉或不一致，影响关节推理；整体流程依赖高质量网格输入及预训练模型。

---

## 211. ROCS: Request-Oriented Compute Sharing for Efficient Large-Scale Recommendation

**arXiv ID:** 2607.27744 | [PDF](https://arxiv.org/pdf/2607.27744v1)

**作者:** Yuxin Chen `[一作]` (Meta AI), Ellie Dingqiao Wen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种请求导向计算共享（ROCS）模型与推理范式，能将请求侧计算在多候选场景下共享，显著提升推荐系统推理效率。

**💡 创新点**

创新点在于：① 在模型层面实现通用的请求侧共享，通过 Generalized Layer Masking（GLM）和 Deep Cross Attention（DCA）实现候选信息延迟注入；② 结合 In-Kernel Broadcast Optimization（IKBO）实现 GPU 端高效共享；③ 引入 Request-Oriented Resource Reallocation（RRR）在保持共享的同时提升模型质量。

**🔧 技术方法**

主要技术包括：GLM、DCA、RRR、IKBO（GLM LCB 与 FlashAttention-3 改造）、GPU 内核融合、持久化 warp 调度与内存对齐优化。

**📊 数据集**

实验使用公开基准 KuaiRand、KuaiVideo、KKBox 以及 Meta 内部广告检索、短视频排序、广告排名等大规模真实工作负载。

**📈 对比分析**

通过与原始模型（Vanilla）和 RankMixer/UGSEP 基线对比，在公开基准上 ROСS 在相同算力下提升 AUC；在生产工作负载中 QPS 提升 47–196%，相对 LogLoss 下降 0.5%；IKBO 将 attention 延迟从 0.55 ms 降至 0.23 ms，显著提高 GPU 端吞吐量。

**⚠️ 局限性**

局限性包括：当候选数量低或请求侧特征贡献不足时收益下降；实现高度依赖 NVIDIA GPU 原语，迁移至其他硬件需重构；模型改造与训练复杂度相对较高。

---

## 212. Measuring Alignment With Reader Highlights Net of Position and Length

**arXiv ID:** 2607.27739 | [PDF](https://arxiv.org/pdf/2607.27739v1)

**作者:** Kazuki Nakayashiki `[一作]` (Glasp Inc), Keisuke Watanabe `[通讯]` (Glasp Inc)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `fede83ac-7505-405f-ab37-e7284695c47f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

评估长上下文压缩方法，用自然社交高亮作为非循环参考，构建匹配估计器消除位置和长度混杂因素，并测量语言模型重要性排序与读者高亮的一致性。

**💡 创新点**

引入了基于匹配的真实控制与误报率校准的新评估框架，量化了语言模型相较于表面特征（位置、长度、词频等）的实际增益，并首次在同一数据集上对比多种压缩与摘要基线。

**🔧 技术方法**

匹配控制技术（按相对深度和长度排名匹配）、分层自举、无假设随机化检验；使用 GPT‑5.4 与 Claude Opus 5 的重要性排序；经典抽取式基线（Luhn、tf‑isf、主题词重叠、词频中心性）和 LLMLingua‑2；词频中心性与词汇中心性对比。

**📊 数据集**

120 篇公共网页（119 英文、1 葡萄牙语），每篇至少 12 位独立阅读者；高亮句子按标记计数的前 15% 定义，至少被两人标记；文档总共 78 个领域，约 2,100 条高亮句子。

**📈 对比分析**

将语言模型排序与无偏截断、经典提取式方法和自制压缩器对比；模型保留 38.4% 高亮句子对比匹配邻居的 19.9%（+0.196 的富集），显著优于截断（+0.003）并约为 Luhn 1958 规则的两倍（+0.088）；随机化检验 p=0.0005，结果在两大模型供应商间高度一致。

**⚠️ 局限性**

数据来自单一平台，存在可读性偏差和段落级别的高亮连续性，匹配过程中 37.9% 高亮句子被丢弃；比较基线缺乏多重检验校正；语言模型仅基于单次无种子调用，未考察选择方差；主要以英语为主，包含一篇葡语文档，可能限制普适性。

---

## 213. VeriSkill: A Self-Evolution Framework for Program Verification Skills

**arXiv ID:** 2607.27733 | [PDF](https://arxiv.org/pdf/2607.27733v1)

**作者:** Changguo Jia `[一作]` (Peking University), Minghui Zhou `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 VeriSkill 框架，用于在 LLM 代理中自我演化程序验证技能，自动识别并修复验证失败所导致的技能缺陷。

**💡 创新点**

创新点在于三阶段演化流程：责任归因过滤验证失败、基于诊断签名聚类抽象可复用教训，以及可执行验证循环来确保改动既提升成功率又保持语义不变。

**🔧 技术方法**

技术包括责任归因分析、诊断签名提取与聚类、可执行验证回归测试、演化记忆存储以及与 LLM 代理、SMT 验证器的交互。

**📊 数据集**

使用从 DafnyBench、GitHub (Frama-C/VeriFast) 构建的 200+、85+、200+ 个程序验证任务，分别用于训练、验证和基准测试。

**📈 对比分析**

与 No Skill、LLM Skill、人类手工技能、以及多种自动演化基线（EvoSkill、AutoSkill、SkillOpt-Lite 等）对比，VeriSkill 在三种验证工具和两套 LLM 代理上均实现了 3–5 倍的 PASS 提升，且对不同模型具有跨模型迁移性。

**⚠️ 局限性**

局限性包括对特定验证器的依赖、需要人工标注的基准数据、对 LLM 生成质量的依赖以及演化过程的计算成本。

---

## 214. Recall Before You Rank: Similarity-Guided Top-$K$ Reuse for Efficient Long-Context Attention

**arXiv ID:** 2607.27692 | [PDF](https://arxiv.org/pdf/2607.27692v1)

**作者:** Wenshuai Yao `[一作]` (Peking University), Kechao Tang `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ReTopK，一种训练无关的动态 Top‑K 注意力加速方法，通过检索历史查询–支持对并在当前查询上重新排序，减少索引发现成本

**💡 创新点**

创新点在于：1) 缓存查询–支持对并按查询相似度检索；2) 用检索到的支持集合与最近窗口合并构成候选集；3) 仅在候选集上做精确 QK 重新排序；4) 通过相似性阈值回退和周期刷新确保可靠性；5) 仍保持完整 KV 缓存，且无需训练

**🔧 技术方法**

采用余弦相似检索、候选合并、精确 QK 重新排序、GPU 融合实现、阈值回退、周期刷新、查询归一化、FIFO 缓存等技术

**📊 数据集**

使用 PG19、LongBench（2WikiMQA、HotpotQA、MultiFieldQA）、NIAH、RULER 数据集，模型为 Qwen2.5-7B、Llama-3.1-8B、Qwen2.5-14B

**📈 对比分析**

与 Full Attention、Exact Top‑K、StreamingLLM、Quest、SparQ、Loki、TokenSelect 等基线比较；在 128K 上可获得 3.07× 的加速，PPL 仅升 0.50%；在多任务和不同模型上排名第一或第二，跨模型迁移保持 1.26–2.66× 的加速

**⚠️ 局限性**

局限性：依赖查询相似度阈值，低相似度时会回退到 Exact Top‑K；候选集大小受缓存容量限制，极长上下文仍需较大缓存或更频繁刷新；未在多模态或非语言任务中验证

---

## 215. New Synchronous Computation Dynamics for Hopfield Networks

**arXiv ID:** 2607.27720 | [PDF](https://arxiv.org/pdf/2607.27720v1)

**作者:** Francisco Requena-Domínguez `[一作]` (University of Málaga), Ezequiel López-Rubio `[通讯]` (University of Málaga)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了离散微分滤波器（DDF）和基于DDF的同步动态（SD‑DDF），用以在Hopfield网络中同时更新多神经元，保证能量下降并加速收敛。

**💡 创新点**

创新点在于将能量下降问题转化为另一个Hopfield网络的求解，通过DDF选择最优同步更新子集，从而实现理论上能量不增且最快下降；同时提供完整的理论证明与实验验证。

**🔧 技术方法**

技术包括：离散微分滤波器构造、同步动态算法、异步与同步能量分析、组合优化（子集选择）以及多种问题的Hopfield映射。

**📊 数据集**

实验使用四类经典组合优化问题的数据集：图二分、随机生成网络、N‑Queens 和旅行商问题（TSP），每类随机生成多组实例。

**📈 对比分析**

与传统异步（SEQ）和改进的随机异步（C‑SEQ）比较，SD‑DDF在所有规模下都显著缩短处理时间（平均提升>90%，最大>98%），能量结果与其它算法相当；统计检验表明能量无显著差异，时间差异效应大。

**⚠️ 局限性**

局限性包括：能量质量提升有限，仍可能停留在局部最优；DDF求解子问题是组合优化，计算开销不被完全消除；实验仅基于人工生成数据，未验证在更复杂或真实应用中的表现。

---

## 216. VocalRender: Score-Native Singing Voice Synthesis for Real-World Composition

**arXiv ID:** 2607.27768 | [PDF](https://arxiv.org/pdf/2607.27768v1)

**作者:** Yukun Chen `[一作]` (Xi'an Jiaotong University), EngSiong Chng `[通讯]` (Nanyang Technological University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出VocalRender，一种基于歌谱的自回归扩散模型，可直接从歌词、音高、音符时值和节拍合成歌声，无需显式时长预测。

**💡 创新点**

采用交错的歌词-音符序列表示解决音素与音符对齐问题，并通过自回归扩散实现动态时长生成，消除了传统模型对时长标注的依赖。

**🔧 技术方法**

结合连续语音VAE、Transformer与Diffusion Transformer（ARDM）、流匹配损失及停顿预测器，实现端到端的高保真歌声合成。

**📊 数据集**

在新构建的2300小时细粒度标注CrawlSinger-OS数据集和5600小时的CrawlSinger上训练，并在Opencpop与后收集的CrawlSinger-Eval上进行评估。

**📈 对比分析**

与TCSinger、TechSinger、Vevo2、SoulX-Singer等前沿SVS模型在WER、SIM、IOU、RPA、N-CMOS、PS-CMOS、MS-MOS、SingMOS等多项客观与主观指标上对比，VocalRender在自然度、可懂度、音色相似度和谱表跟随上均优于或接近最强基线，CMOS自然度提升0.42点。

**⚠️ 局限性**

训练管线受自动音谱转录与作曲家手稿之间差异限制，且在复杂音乐结构或高分辨率表达上仍存在潜在误差。

---

## 217. VESTIGE: A Knowledge-Guided Masking Strategy for Corruption-Aware Fine-Tuning of Genomic Transformers, Validated on Ancient DNA Reconstruction

**arXiv ID:** 2607.27712 | [PDF](https://arxiv.org/pdf/2607.27712v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 218. Private Face Recognition Training Dataset Publication via Identity-Decoupled and Geometry-Preserving Face Distillation

**arXiv ID:** 2607.27764 | [PDF](https://arxiv.org/pdf/2607.27764v1)

**作者:** Shuhuan Chen `[一作]` (IIE, CAS), Zhen Lei `[通讯]` (CBSR&MAIS, CASIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了私有人脸训练数据集发布方法 Private Face Distillation，通过正交几何保留（OGP）与关系拓扑对齐（RTA）两阶段框架，生成去源身份、保留代理几何的 RGB 代理图像；

**💡 创新点**

将身份反向解耦与几何保留分离，首次使用正交变换保持球面关系而抑制源身份关联，并通过关系拓扑对齐在代理图像中保留识别有用的类结构，解决了身份悖论；

**🔧 技术方法**

正交几何保留（OGP）使用矩阵指数的正交变换；关系拓扑对齐（RTA）结合交叉熵、BN 正则与余弦相似度对齐；基于 AdaFace 与 IR-50 的 FR 骨干；代理图像由随机噪声通过 Adam 优化生成；

**📊 数据集**

CASIA-WebFace 作为公开预训练集；四个多场景基准（跨年龄、监控、极姿、VIS‑NIR）以及标准 FR 评测集 LFW、CFP‑FP、CPLFW、AgeDB、CALFW；

**📈 对比分析**

与 Face Dataset Publication、Face Anonymization、Differential Privacy、Dataset Distillation 等方法对比；在四个域移位场景下，PFD 在下游 FR 效果上平均提升约 2%，在 IJB‑C 监控场景 TAR@FAR=1e-3 提升 3.94%；源身份可链接率降至约 0.7%，接近随机率；

**⚠️ 局限性**

仍存在对更高级重建攻击的潜在泄露风险；需要较大的私有训练集与复杂的两阶段优化，部署成本较高；公开代理图像虽去源但仍不完全消除源身份关联，攻击模型进化可能提升成功率。

---

## 219. Hierarchical Latent Reasoning for LLM-based Recommendation

**arXiv ID:** 2607.27760 | [PDF](https://arxiv.org/pdf/2607.27760v1)

**作者:** Peiyu Hu `[一作]` (Xi'an Jiaotong-Liverpool University), Jia Wang `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种层次化的潜在推理框架 HiLaR，用于提升大语言模型在推荐任务中的性能。

**💡 创新点**

创新点在于：①基于时间引导的分层量化构造粗到细的用户偏好表示；②将这些层次化表示与 LLM 隐状态对齐，实现层级对齐约束；③使用层级奖励驱动的 GRPO，对推理轨迹进行细粒度强化学习，兼顾最终推荐质量和中间状态的边际贡献。

**🔧 技术方法**

主要技术包括：多级残差量化、时间窗口监督、隐状态对齐损失、层级奖励（有效性、对齐与增益）与 GRPO 强化学习，以及 Qwen2.5‑1.5B 作为基础 LLM。

**📊 数据集**

实验使用四个亚马逊商品领域数据集：Toys、CDs、Games、Instruments，按时间顺序划分训练/验证/测试。

**📈 对比分析**

与传统序列推荐、生成式推荐、LLM 推荐以及其他潜在推理方法（LatentR3、VRec、FLR 等）对比，HiLaR 在大多数数据集和评估指标（HR@K、NDCG@K）上均取得最高或次高分，显著提升了推荐质量。

**⚠️ 局限性**

局限性包括：①训练成本相对较高（需要多步强化学习和多层奖励计算）；②对超参数（层数、对齐权重、奖励权重等）敏感；③在极短历史或极大数据规模下，层次化量化的效果尚未充分验证。

---

## 220. A Sparse Glimpse of the Whole: Train-Free Self-Speculative Decoding

**arXiv ID:** 2607.27735 | [PDF](https://arxiv.org/pdf/2607.27735v1)

**作者:** Yuesong Liu `[一作]` (University of Science and Technology of China), Yinlong Xu `[通讯]` (University of Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练无关的自我推测解码框架SparseSpec-L，用于长上下文下的大语言模型推理

**💡 创新点**

创新点在于：①利用目标模型自身进行推测与验证，避免辅助模型结构差距；②通过每层每头的注意力统计动态重建稀疏KV索引，实现可回收的稀疏推测；③使用基于熵的在线控制器动态调节推测长度，避免效率倒转

**🔧 技术方法**

采用稀疏KV采样、密集KV验证、注意力重要性估计、熵预测与自适应长度控制等技术

**📊 数据集**

在LongBench v2、InfiniteBench、RULER QA1、RULER NIAH等长文本推理、推理、检索与合成任务上进行评测

**📈 对比分析**

与多种基线（辅助推测、Medusa、EAGLE‑3、MagicDec、RAPID等）对比，SparseSpec-L在长上下文场景下实现最高加速，最高可达2.79×速度提升，同时保持70‑85%的接受率

**⚠️ 局限性**

限制包括：实现中未使用融合稀疏/密集注意力核；实验仅在单块A40 GPU、64K上下文、128生成长度下进行；熵控制器预测性有限，部分任务收益有限

---

## 221. NMINE: Normalized Mutual Information Neural Estimation

**arXiv ID:** 2607.27710 | [PDF](https://arxiv.org/pdf/2607.27710v1)

**作者:** Petra Eerikinharju `[一作]` (University of Eastern Finland), Ville Hautamäki `[通讯]` (University of Eastern Finland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种全神经网络框架（NMINE），用于连续多维变量的归一化互信息（NMI）估计。

**💡 创新点**

创新点在于将 MINE 的神经互信息估计与 MI-NEE 风格的熵估计结合，使用统一的 Donsker–Varadhan 变分目标和统一的神经判别器完成 NMI 的完整神经估计，避免了传统 kNN 方法在高维和数值不稳定性上的缺陷。

**🔧 技术方法**

主要技术包括 MINE（神经互信息估计）、NEE（神经熵估计）、Donsker–Varadhan 变分表示、全连接神经判别器、异步归一化（NMI=MI/H(Y)），以及 Adam 优化器等。

**📊 数据集**

实验数据集为合成的多维高斯数据（维度 1、2、4、8，相关系数 0–0.95）以及初步的 Student‑t 分布数据。

**📈 对比分析**

与传统的 KSG（k=5）基准 NMI 估计进行 MAE 对比。NMINE 在所有维度下均显著降低误差（误差下降幅度约 30–74%），并保持正确的单调关系；KSG 在高维时趋向过估计。

**⚠️ 局限性**

局限性包括：需要训练多组神经网络，计算成本高；在高 MI 场景下可能出现低估偏差；熵估计依赖于样本范围近似的参考分布；目前未进行联合训练或更丰富的非高斯/真实世界数据验证。

---

## 222. MECA: A Mechanism-Centered Agent for Constructing Well-Specified and Valuable Mathematical Conjectures

**arXiv ID:** 2607.27709 | [PDF](https://arxiv.org/pdf/2607.27709v1)

**作者:** Wentao Long `[一作]` (Fudan University), Zaiwen Wen `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种多代理框架 MECA，通过共同细化命题与其支撑机制来自动构造明确、研究价值高的数学猜想。

**💡 创新点**

将猜想构造与证明机制视为互相耦合的搜索对象，采用机制中心的审计与覆盖诊断，避免生成空洞猜想，并引入探索‑评论‑终局挑战的多阶段流程。

**🔧 技术方法**

利用大语言模型驱动的探索代理、机制审计代理和最终挑战代理；构建机制图、覆盖判断、定向搜索；使用 QED 证明系统进行独立验证。

**📊 数据集**

目标重建使用20篇2023后期论文的结论与其前置文献卡片；猜想生成共100个来自文献与开放问题的半开放猜想；通过 QED 进行证明与反例测试。

**📈 对比分析**

与草稿‑审计‑修订基线在目标无感知重建任务中对比；在20个案例中 MECA 的平均得分从48提升至69；在100个猜想的 QED 评估中，35%可证明、11%被反例，54%技术难度未解决，说明猜想既易读又保持高难度。

**⚠️ 局限性**

受限于 LLM 的生成质量与机制审计的非形式化，难以处理极端复杂或非结构化问题；未能保证猜想真正开放或评估其难度；依赖现有证明系统的搜索能力。

---

## 223. Albilich: Steerable Proof-State Orchestration for LLM-Based Mathematical Research with CAS Integration

**arXiv ID:** 2607.27705 | [PDF](https://arxiv.org/pdf/2607.27705v1)

**作者:** Ting Gong `[一作]` (University of Washington), Yong Yang `[通讯]` (Texas State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个可扩展的、基于 SQLite 的长期数学研究代理系统，集成了文献检索、计算代数系统 (CAS) 以及分层验证流程。

**💡 创新点**

创新点包括：1）持久化的证明状态管理，支持长期跟踪与回溯；2）PhD‑advisor 角色与图结构分解，实现策略级重定向与并行分支；3）多模工具调用（CAS 与文献检索）与严格/集成验证器的分离，提升透明度与可审计性。

**🔧 技术方法**

技术包括：大语言模型（GPT‑4），Model Context Protocol (MCP) 工具调用，SageMath/Macaulay2/GAP/Singular/Julia 计算后端，SQLite 版本化图数据库，基于规则的任务调度，局部与全局验证器框架。

**📊 数据集**

数据集包括 RealMath（Math_arXiv 部分的10个问题）和 Kourovka Notebook 的开放组理论问题（如 Problem 17.91、20.2、21.142）。

**📈 对比分析**

通过与 RealMath 基准对比、CAS 开关消融、PhD‑advisor 开关消融进行实验。结果：在 RealMath 上10/10完成；CAS 开启可减少约32% tokens；PhD‑advisor 在 Problem 21.142 上在更少时间和 token 下成功解决，而关闭则未能完成，且 token 与时间消耗翻倍。

**⚠️ 局限性**

局限性：评估仅基于内部实验与示例，缺乏正式验证与外部同行评审；量化指标初步，未覆盖更广泛的数学问题；对复杂开创性问题的处理仍受限于当前模型与策略。

---

## 224. Calibrate Before Reason: Robust Visual Token Reduction against Semantic Drift in VLMs

**arXiv ID:** 2607.27700 | [PDF](https://arxiv.org/pdf/2607.27700v1)

**作者:** Jiasheng Li `[一作]` (Tianjin University), Huihui Li `[通讯]` (Tianjin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种训练无关的视觉标记校准框架 CaRe，先通过扰动鲁棒校准锚点，再用置信门控的校准信号对锚点进行加权校准，从而在显著压缩视觉标记的同时保持语义一致性。

**💡 创新点**

创新点在于将“先校准后推理”原则应用于视觉标记压缩，提出扰动鲁棒校准锚点与置信门控校准信号两大模块，使压缩后标记在语义上保持与全标记相近，显著缓解语义漂移。

**🔧 技术方法**

核心技术包括：多方向扰动响应估计与鲁棒归一化、方向风险修正与多样性约束的锚点选取、语义-空间亲和度与置信门控的校准信号筛选、软分配与加权平均的锚点校准，以及对锚点范数的归一化保持。

**📊 数据集**

使用了 LLaVA‑1.5‑7B、LLaVA‑NeXT‑7B、Qwen2.5‑VL‑7B 三大 VLM；在 GQA、MMB、MME、POPE、SQA、VQA‑v2、VQA‑T、MMMU、SEED 等九个基准上进行评估。

**📈 对比分析**

与 FastV、SparseVLM、VisionZip、DivPrune、ZOO‑Prune、FiCoCO‑V、V²Drop 等现有标记压缩方法相比，CaRe 在 94.4% 归约率下保留 96.4% 的性能，且在 66.7% 归约率下达到 99.9% 的性能保留；整体加速比可达 2.30×，并在大多数任务上均超过或接近未压缩模型。

**⚠️ 局限性**

局限性在于对视觉信息弱、模糊或严重降质的场景敏感，校准只能在已有信息范围内优化，无法完全弥补视觉编码阶段已丢失的细节。

---

## 225. RefineSVG: Visual Feedback-Driven Reinforcement Learning for Image-to-SVG Generation

**arXiv ID:** 2607.27699 | [PDF](https://arxiv.org/pdf/2607.27699v1)

**作者:** Shaobo Liu `[一作]` (Shenzhen University), Zhengping Liang `[通讯]` (Shenzhen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种单步闭环视觉反馈框架 RefineSVG，利用外部渲染引擎和 Diff-Map 对大模型生成的 SVG 进行自我校正，实现高保真图像到 SVG 的转换。

**💡 创新点**

创新点在于：①将渲染反馈与 ReAct 机制结合，形成闭环校正；②设计 SVG 定向语义词表压缩 52%；③构建三阶段训练管线（SFT、冷启动重采样、RL），实现视觉纠错能力；④多维 Diff-Map 提供精细误差信号。

**🔧 技术方法**

采用多模态大语言模型（Qwen2.5‑VL‑3B/7B）、CairoSVG 渲染、ReAct 框架、GRPO 强化学习、DINOv2/CLIP 视觉特征、YCbCr 差分、量化坐标编码等技术。

**📊 数据集**

主要使用 SVG‑Stack 公开数据集（约 1.52M 训练对、20K 冷启动、35K RL 采样），评测采用 SVG‑Stack‑1K、MMSVG‑Illustration、MMSVGBench、svg‑emoji。

**📈 对比分析**

与优化型方法 DiffVG/LIVE、通用 VLM（Qwen3‑VL‑235B、GPT‑5.2、Gemini‑3.1‑Pro）以及 SVG 专用模型 StarVector/OmniSVG/InternSVG 对比，RefineSVG‑7B 在 6 大视觉质量指标（PSNR、SSIM、LPIPS、MSE、DINO、CLIP‑I2I）中多项领先，生成的 SVG 代码更短（≈640 令牌，显著低于 8B 级别模型），推理速度更快。

**⚠️ 局限性**

局限在于目前仅支持单轮修正，缺乏多轮自适应终止与高分辨率扩展；对冷启动数据的构造依赖单次采样，难以扩展到更复杂场景。

---

## 226. DP-LENS: A Density-Aware Polyfocal Lens with Topology-Driven Auto-Routing for Occlusion Management in Immersive 3D Analytics

**arXiv ID:** 2607.27697 | [PDF](https://arxiv.org/pdf/2607.27697v1)

**作者:** Nieyu Cao `[一作]` (Hong Kong Polytechnic University), Lik-Hang Lee `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `70e40602-aae3-44bd-80ec-4a7f2674330f` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于密度感知的多焦点鱼眼透镜（DP‑LENS）框架，并在此基础上集成了基于大语言模型的语音导航辅助，旨在解决密集三维数据在沉浸式分析中的遮挡、认知负荷和身体疲劳问题。

**💡 创新点**

创新点包括：① 用连续密度场与拓扑骨架驱动的非破坏性体渲染，实现上下文保持的鱼眼变形；② 通过梯度修复算法保证拓扑连贯性；③ 结合LLM的语义解析与动态空间上下文注入，实现自动化长距离路径规划；④ 将上述技术打包成可复现的开源工具包。

**🔧 技术方法**

核心技术有：GPU并行计算（KDE、体渲染和光线跟踪）；多焦点鱼眼模型与自适应扩展半径；上下文感知的X‑ray（透明度平滑抑制）渲染；语音识别+LLM意图解析；拓扑驱动的自动路径规划与磁性“吸附”机制。

**📊 数据集**

使用的三类数据集为：① 合成点云（200k点）；② 涡流流场（200k粒子）；③ 真实血管扫描（约200k体素）——这三种数据分别对应 Occlusion 级别 2、3、5。

**📈 对比分析**

与传统 WIM‑NAV 与 V‑SLICE 进行对比实验。DP‑LENS 在所有三种数据集上都实现了最快的完成时间、最高的完成率（尤其是 Level 5 仍为 100%），主观工作量、身体疲劳和晕动评分显著低于两种基线；在房间规模实验中，语音辅助模式将任务时长降至约 70% 以内，并在不同规模下保持一致性，显示出可扩展性。

**⚠️ 局限性**

局限性包括：语音识别与LLM推理的延迟限制了微调交互；系统仅在静态数据上测试，动态场景仍需改进；参与者主要为学生，缺乏专业医生等高阶用户的验证；仅在VR环境下评估，AR 中的光学遮挡与硬件限制尚未探讨。

---

## 227. Restoring Collaborative Signals in Semantic-ID Generative Recommendation via Personalized Natural Language

**arXiv ID:** 2607.27682 | [PDF](https://arxiv.org/pdf/2607.27682v1)

**作者:** Changjiang Han `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Bowei He `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种训练无关的推理时重排序通道，通过自然语言标签将协作信号注入到冻结的语义ID生成模型中。

**💡 创新点**

创新点在于将第二阶协作结构映射为分层协作向量，利用LLM生成的受众标签构建桥接模型，直接在生成器的每个层级上做协作重排序而不扩展搜索束宽。

**🔧 技术方法**

采用了层次泊松分解来提取项目协作因子，利用LLM总结用户历史得到受众标签，并学习标签到因子空间的映射，再在解码时通过协作残差对s_a、s_b层级进行重排序。

**📊 数据集**

实验基于RecIF评测集（包含交互、视频、广告、标签条件和商品推荐五个子任务），使用OneRec-1.7B/8B模型的预训练权重。

**📈 对比分析**

与冻结模型基线、语义相似重排序、基于历史的因子查询等对比，Ours在8B上将hit@10从11.14%提升至15.50%，在1.7B上提升至6.78%，显著优于其他无训练干预方法。

**⚠️ 局限性**

局限性包括仅在RecIF和OneRec两种规模上验证，且重排序受限于候选池覆盖率；语言构建的受众查询在分辨率上仍有限，且跨架构与数据集的泛化尚未证明。

---

## 228. Tight Sample Complexity for Low-Rank Adaptation: Matching Bounds and Rank Selection

**arXiv ID:** 2607.27680 | [PDF](https://arxiv.org/pdf/2607.27680v1)

**作者:** Arunan J `[一作]` `[通讯]` (Independent Researcher), Arunan J (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对低秩适配（LoRA）在大型预训练模型微调中的样本复杂度进行系统分析，给出了匹配的上下界，并对LoRA秩选择问题给出理论指导；

**💡 创新点**

创新点包括：①在LoRA可表达能力基础上提供了快速率O(rd/n)的上界与匹配的下界；②首次证明了秩选择的二分律——欠秩导致固定误差，过秩在无正则化的ERM下线性惩罚；③提出了可自适应的核范数+截断估计器实现与最优阶数无关的性能；

**🔧 技术方法**

技术手段主要包括：局部Rademacher复杂度与Dudley熵积分的快速化（通过曲率条件得到fast rate）；信息论Fano与Gilbert‑Varshamov编码实现下界；对LoRA参数化的非凸秩约束进行切片分析；

**📊 数据集**

实验验证使用了合成的迹回归基准以及三种实际微调任务：DistilBERT/SST‑2、DistilBERT/MRPC、RoBERTa/SST‑2；

**📈 对比分析**

与标准全参数微调相比，LoRA在最佳秩处性能相当，但过高秩会出现U‑形损失上升。实验显示在最佳秩后，验证交叉熵明显升高，配对置换检验在两配置上p<0.05；

**⚠️ 局限性**

局限性包括：需要目标函数可精确写成LoRA形式（完备性假设）；对局部二次性、损失光滑等条件敏感；上界仍带log n因子；仅在理想化的无分布漂移下给出理论，实际优化（SGD、Adam）与统计不完全对齐。

---

## 229. Stop Shipping AI Agents on Faith: Capability Is Not Production Readiness

**arXiv ID:** 2607.27677 | [PDF](https://arxiv.org/pdf/2607.27677v1)

**作者:** Fouad Bousetouane `[一作]` `[通讯]` (ProofAgent.ai), Fouad Bousetouane (ProofAgent.ai)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并实现了ProofAgent Index（PAI），一种用于评估AI代理生产环境就绪度的治理索引。

**💡 创新点**

创新点在于将行为评估、上下文质量、合规性和治理控制四个维度组合成可分解的指数，并引入硬块规则以阻止不安全发布。

**🔧 技术方法**

使用的技术包括ProofAgent Harness评估框架、几何平均聚合、基于LLM的评估与合规检查以及命令行接口实现自动化。

**📊 数据集**

采用的评估数据集为在医疗和金融两个高度监管领域内的ProofAgent Harness陷阱集合，覆盖了三种代理能力层级和两种上下文设置。

**📈 对比分析**

通过持出配置级AUC和10,000轮大规模验证，PAI在12个配置中实现AUC 0.98，证明其比单纯行为得分（AUC 0.80）更能区分生产就绪风险。

**⚠️ 局限性**

局限性包括对仅在实验环境中定义的陷阱和规则的依赖、缺少对真实生产流量的直接验证，以及对极低概率异常情形的覆盖不足。

---

## 230. Gradient-free Task-Conditioned Retrieval for On-Device In-Context Learning

**arXiv ID:** 2607.27766 | [PDF](https://arxiv.org/pdf/2607.27766v1)

**作者:** Xinyu Luo `[一作]`, Haoliang Li `[通讯]` (City University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种无梯度的任务感知检索框架CoRA，用于在设备端执行ICL前置检索，利用冻结编码器与候选输入输出对构建低秩检索空间。

**💡 创新点**

创新之处在于通过输出条件的岭回归对冻结层进行闭式对齐，获得最佳低秩基，兼顾多层输入相似度与输出相关性，并实现流式构建与多模态扩展。

**🔧 技术方法**

采用冻结Transformer层的层级表示、CKA层相似度聚类、输出条件的闭式岭回归、低秩SVD压缩、两遍流式构建以及多模态CLIP视觉编码等技术。

**📊 数据集**

在十个文本基准（5分类+5生成）和四个多模态基准（VQAv2、OKVQA、VizWiz、MSCOCO）上评估，并在Raspberry Pi 5上实现端到端部署。

**📈 对比分析**

与随机、BM25、SBERT、BERT、DPP‑BERT、Qwen3‑Emb、MLSM以及训练型方法EPR/CEIL/TTF对比，CoRA在所有任务和模型上平均分数最高，同时在设备端显著降低了时间与内存占用。

**⚠️ 局限性**

目前仅在软件层面实现，无法处理动态增量候选池；未针对专用硬件加速；多模态输出条件对高变异输出的适应仍有限。

---

## 231. A Structured Knowledge Infrastructure for Domain-Specific Data Asset Discovery

**arXiv ID:** 2607.27748 | [PDF](https://arxiv.org/pdf/2607.27748v1)

**作者:** Mengdi Chen `[一作]` (Xiaohongshu), Wei Sun `[通讯]` (Xiaohongshu)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在小红书商业广告数据仓库构建了双层知识基础设施，结合图引导检索（GGR）和场景感知排序器（SAR），解决资产检索与使用知识覆盖不足的问题。

**💡 创新点**

创新点在于采用知识图谱作为候选门控与意图路由，并通过双层检索+实体识别和场景注解实现高精度资产发现和知识覆盖，同时实现闭环热更新。

**🔧 技术方法**

使用技术包括知识图谱构建、同义词聚类、意图分类器、规则式实体识别、图遍历、层级得分、LLM热加载与Token节约策略。

**📊 数据集**

数据集来源为小红书商业广告数据仓库，包含5300+ Hive表、1200+ BI数据集、14个业务域的179个结构化Markdown文档以及2,859节点的知识图谱。

**📈 对比分析**

与传统BM25全库检索基线相比，Hit@10从19.1%提升至96.6%（+77.5个百分点），知识覆盖率从56%提升至77%，端到端延迟为4.84–5.33秒，Token消耗下降71.6倍。

**⚠️ 局限性**

局限性包括依赖人工批准的热更新导致响应时间受限，对未知意图或新领域迁移的鲁棒性待验证，且在多步推理或跨域迁移方面仍需进一步改进。

---

## 232. Can LVLMs Uncover the Truth Behind Visual Illusions? An Analysis of Perceptual and Reasoning Capabilities

**arXiv ID:** 2607.27747 | [PDF](https://arxiv.org/pdf/2607.27747v1)

**作者:** Liangjie Zhao `[一作]` (Adelaide University), Jianing Li `[通讯]` (Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于真实世界视觉错觉图像的评估基准，用来综合测试大型视觉语言模型（LVLM）的感知与推理能力。

**💡 创新点**

创新点在于将视觉错觉作为诊断工具，将感知与推理统一到同一任务中；构建了涵盖五类错觉的真实图像集，并设计了检测、描述、推理三种问答类型，填补了开放世界推理评估的空白。

**🔧 技术方法**

采用多模态预训练模型与大语言模型的链式思考（CoT）思维模式进行推理；评估过程中使用 GPT‑4o 作为自动判分器；通过构造多种问答格式与五类错觉标签实现对模型的系统评估。

**📊 数据集**

使用了新构建的 650 张真实错觉图像及 3,000+ 个 QA 对组成的基准（涵盖形态、颜色、空间、光影、关联五类），并与现有的 POPE、IllusionVQA、IllusionBench+ 等基准做对比。

**📈 对比分析**

在多款闭源与开源 LVLM（参数规模从 1B 到数百 B）上进行单轮评估，使用 GPT‑4o 评判模型输出的正确性。结果显示大多数模型在错觉任务上的表现远低于其他评测基准，思考模式对推理类问题有显著提升，但对感知类问题往往无效甚至下降，说明模型的感知与推理能力尚未很好整合。

**⚠️ 局限性**

限制在于基准样本量有限，难以用于大规模训练；错觉图像受真实世界条件限制，缺乏多样性；评估方法仍可能被安全响应或语言偏差影响；模型对人类直觉与物理真实的对齐不一致，需要进一步改进对齐与推理机制。

---

## 233. Towards joint scaling laws with optimal batch size schedules

**arXiv ID:** 2607.27731 | [PDF](https://arxiv.org/pdf/2607.27731v1)

**作者:** Jiaxiang Li `[一作]` (Meta), Shiyun Xu `[通讯]` (Independent researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并提出了最佳批量大小调度公式，证明其在多种模型和训练设置下优于静态批量大小。

**💡 创新点**

将批量大小视为独立于学习率、模型规模的可扩展维度，给出闭式最优调度，并与学习率、权重衰减共同构建联合缩放规律。

**🔧 技术方法**

利用凸优化理论的序列到序列损失预测、连续积分最优化、Jensen/Cauchy不等式推导；实验采用Muond-NSGD/AdamW优化器，Llama3、Qwen3 MoE、VLM等模型。

**📊 数据集**

Fineweb‑edu、Cauldron、MMStar、MMMU、ScienceQA、TextVQA、DocVQA、InfoVQA、OCRBench、MME、GSM8K及自建数学数据集。

**📈 对比分析**

与传统固定批量大小、相同学习率/权重衰减设置对比，使用验证损失、困惑度、任务准确率等指标；动态批量大小在不同学习率调度下可提升0.8%–14.6%困惑度，计算效率提升6%–15%。

**⚠️ 局限性**

优势随批量大小趋近无穷时消失；在纯低精度训练时优势减弱；理论依赖凸优化近似，真实深度网络中可能存在偏差。

---

## 234. Guiding Large Language Models with Genetic Programming-Evolved Heuristic Knowledge for Dynamic Multi-Mode Project Scheduling

**arXiv ID:** 2607.27698 | [PDF](https://arxiv.org/pdf/2607.27698v1)

**作者:** Yuan Tian `[一作]` (Victoria University of Wellington), Mengjie Zhang `[通讯]` (Victoria University of Wellington)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于遗传程序演化规则知识向大语言模型（LLM）迁移的在线动态多模式项目调度框架，并设计了四种特征与规则层面的引导机制；

**💡 创新点**

创新点在于逆向知识迁移——将GP演化出的启发式规则抽象为特征/规则层级知识，用以指导LLM决策，并系统评估其对令牌消耗、决策稳定性及解释焦点的影响；

**🔧 技术方法**

主要技术包括遗传程序（GP）超启发式用于进化优先规则、零-shot LLM决策、特征抽取、规则引用与跟随机制、令牌计数与熵测度以及Wilcoxon符号秩检验；

**📊 数据集**

使用合成的DMRCPSP数据集：30个活动、3种执行模式、三种约束强度（0.75、0.5、0.25）共15个实例，训练/验证/测试分别采用不同的随机持续时间实现；

**📈 对比分析**

通过与未引导LLM和单一GP规则基线比较，利用10次独立运行的归一化完工期评估，发现Feature Selection、Rule Reference和Rule Follow在大多数场景下显著优于基本模型；在令牌效率上Feature Selection表现最佳，Rule Follow在高约束强度场景中取得最佳完成期；

**⚠️ 局限性**

局限性包括：规则选择固定未根据实时决策情境动态调整，低约束强度下候选集大导致令牌消耗与决策波动增加；Rule Follow并非始终最优，且模型对规则解析存在误差；未来需实现动态规则挑选、多阶段候选过滤及更丰富的GP知识挖掘。

---

## 235. Train Small, Deploy Large: Zero-Shot GNN Transfer Through Geometric Renormalization

**arXiv ID:** 2607.27767 | [PDF](https://arxiv.org/pdf/2607.27767v1)

**作者:** Robert Jankowski `[一作]` (TU Delft), M. Ángeles Serrano `[通讯]` (University of Barcelona)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究提出了一种“train-small, deploy-large”零射训练框架，通过几何重标化（GR）在小规模图上训练GNN，并将权重直接迁移到原始大规模图上，保持大部分预测性能并显著降低训练成本。

**💡 创新点**

创新点在于首次将几何重标化与零射GNN权重迁移结合，证明在保持网络几何和拓扑结构的前提下，模型参数在不同尺度间近似不变；并引入多维度评估（准确率、表示相似度、训练轨迹）系统验证该性质。

**🔧 技术方法**

主要技术包括：几何重标化（利用Mercator/ cuMercator 进行超曲面嵌入并合并相邻节点）、多层GNN（GCN、GraphSAGE、GAT）训练、Centered Kernel Alignment (CKA) 与 Orthogonal Procrustes (OP) 进行表示相似度评估、Jensen‑Shannon 散度量化训练与评估轨迹的一致性。

**📊 数据集**

使用了合成网络（HypBench 中的 𝕊¹/ℍ² 随机图）以及八个真实数据集（Cora、PubMed、Computers、Photo、CS、Physics、WikiCS、Flickr），在每个数据集上采用多重重标化层次和不同压缩率进行实验。

**📈 对比分析**

与完全随机重标化（Random）以及其他几何/边聚合方法（Laplacian renormalization group、MagEdgePool、SpreadEdgePool）比较。GR 在维持平均度、聚类、谱隙等结构特征的同时，零射转移准确率高于随机基线，且在压缩率高至 32× 时仍保持 80% 以上的准确率；训练时间相比原图缩短 10‑20 倍，表示相似度（CKA/OP）和训练轨迹一致性（JS 散度）均显著优于对比方法。

**⚠️ 局限性**

局限性包括：仅对节点特征做简单平均聚合，缺乏尺度变换理论；未探索不同规模下的超参数最优性；几何重标化目前仅适用于一维相似空间；实验聚焦同质图和节点分类任务，异质图或其他学习任务的可迁移性尚未验证。

---

## 236. Sign Language Question Answering: A New Task, Benchmark, and Baseline for Sign Language Understanding

**arXiv ID:** 2607.27826 | [PDF](https://arxiv.org/pdf/2607.27826v1)

**作者:** Shiwei Gan `[一作]` (Nanjing University), Lei Xie `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了新的签语问答任务（SLQA），并构建了基于PHOENIX14T和CSL‑Daily的两大SLQA基准数据集；

**💡 创新点**

创新点在于将自然语言问题直接映射到签语视频，实现对视频内容多层次推理的评估，并提出Question‑Conditioned Modulated Temporal Downsampling（QCMTD）模块和逐阶段的域内知识迁移训练策略；

**🔧 技术方法**

采用了ResNet‑18视觉编码器、QCMTD模块进行时间下采样、mT5语言解码器，并通过三阶段预训练（CSLR→SLT→SLQA）实现知识迁移；

**📊 数据集**

使用PHOENIX14T和CSL‑Daily原始注释自动生成的QA对，覆盖位置推理、结构推理、视觉搜索、词汇识别与翻译理解五类问题；

**📈 对比分析**

与通用视频问答模型（VideoLLaMA3、Qwen3‑VL、InternVL3）及两阶段的Sign2Text2Answer基线对比，SLQAM在所有问题类型上均优于对手，整体指标如ROUGE、BLEURT、CIDEr等均显著提升；

**⚠️ 局限性**

局限在于仍需人工设计模板生成QA，对极端开放式问题的适应性不足，且依赖于已有的签语语料进行预训练，跨语言与跨方言的通用性待进一步验证。

---

## 237. Reversing Reserve Logic: Optimal Holdback in Local Allocation under Scalable Entry

**arXiv ID:** 2607.27817 | [PDF](https://arxiv.org/pdf/2607.27817v1)

**作者:** Hiroaki Odahara `[一作]` `[通讯]` (University of Tokyo), Hiroaki Odahara (University of Tokyo)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在自动化参与者可能争夺稀缺机会（如音乐会门票和加速器时间）的情况下，如何通过匿名筛选规则来优化意图用户的预期效用。

**💡 创新点**

提出了一种最优的三带规则，该规则在不同拥挤程度下采取不同的分配策略，并且能够在不需要支持非进入的情况下提高意图用户的盈余。

**🔧 技术方法**

使用了线性规划和点对点弱对偶性证明来优化规则，并通过分析最强竞争对手的承诺来设计分配机制。

**📊 数据集**

模型假设有多个意图用户，每个用户的承诺能力是独立抽取的，且遵循严格递增的分布函数。

**📈 对比分析**

与传统的总是分配给唯一领导者的规则进行比较，发现最优规则在某些情况下可以在不需要支持非进入的情况下提高意图用户的盈余。

**⚠️ 局限性**

限制在于假设了匿名、计数独立和单次通过的规则，未考虑不对称或公开相关行为、一般多账户政策和类型依赖的价值。

---

## 238. Beyond Borrowed Histories: Person-Aligned User Simulation for Interactive Role-Playing Evaluation

**arXiv ID:** 2607.27816 | [PDF](https://arxiv.org/pdf/2607.27816v1)

**作者:** Yuhang Zhu `[一作]` (University of Science and Technology of China), Hongtao Xie `[通讯]` (University of Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个面向个体化的自由多轮角色扮演评测框架（person-aligned free multi-turn benchmark），通过为每个用户训练专属用户模拟器并生成与候选RPA共同构建的对话轨迹，同时自动生成以该用户为基础的个性化体验评估规则，对候选RPA进行个体化、通用回合质量以及整个会话质量三轨道评分。

**💡 创新点**

创新点在于：① 把评测单元从单一RPA转变为用户–RPA 对；② 通过用户模拟器让候选RPA参与对话轨迹的构造，消除固定历史带来的偏差；③ 依据同一用户的历史经验注释自动构建个性化评估规则，捕捉不同用户对对话体验的主观差异；④ 将评测拆分为个性化、通用回合与会话三轨道，揭示不同维度下的能力匹配和差异。

**🔧 技术方法**

主要技术包括：每用户 LoRA 微调（基于 Qwen3.5-35B-A3B Instruct）训练用户模拟器；Meta-Prompt 自动化生成个性化评估规则；使用 GPT‑5.5 以及 Claude Opus 4.8、DeepSeek V4 Pro 等大型模型作为评估判别者；三轨道评分体系（个性化、Generic、Session）以及对应的分数聚合公式。

**📊 数据集**

使用了一个包含 300 张中英双语角色卡的对话数据集，并从五位真实用户处收集了 5,133 条带有即时满意度标注的多轮对话（每位约 1,000 条），这些数据被划分为训练集（用于模拟器与评估规则）与评估集（用于验证）。

**📈 对比分析**

通过对 16 款候选 RPA（包括 GPT‑5.4、Claude Opus、DeepSeek、Gemini、Qwen、MiniMax、CoSER 等）进行两轮自由多轮交互，得到每个用户的个性化、通用回合与会话得分。结果显示不同候选在三条轨道上表现不一致；例如 GPT‑5.4 在 Generic 轨道上领先，Claude Sonnet 4.6 在 Session 轨道上领先；没有任何候选在所有用户和所有轨道上都占优。与人工标注对齐的指标，个性化评分在保持对用户体验的敏感度时相较于通用评分提升了约 0.06 的一致性。

**⚠️ 局限性**

局限性包括：① 仅覆盖五位深度标注用户，难以全面代表多样化用户群体；② 虽然对模拟器行为与个性化评分的有效性做了验证，但缺乏覆盖所有候选的完整人工排名参考；③ 评测依赖大型模型判别器，存在模型偏倚和判别一致性问题；④ 数据集的角色卡与用户样本偏少，可能限制模型在更大规模多样场景下的泛化。

---

## 239. Semantic-Aligned Structural Abstraction for Multimodal Sentiment Analysis

**arXiv ID:** 2607.27790 | [PDF](https://arxiv.org/pdf/2607.27790v1)

**作者:** Wei Chen `[一作]` (Huazhong Agricultural University), Ying Sha `[通讯]` (Huazhong Agricultural University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出SentiLLM框架，将非语言模态通过语义对齐的结构抽象转化为可被LLM理解的情感标记，并通过双流显著-环境校准机制实现对情绪转折的捕获。

**💡 创新点**

创新点在于将非言语模态视作时序句子，通过软显著性拆分、双查询抽象以及背景校准将连续原始信号压缩为语义对齐的情感token；并且仅用少量可训练参数即可显著提升性能。

**🔧 技术方法**

技术包括上下文时序编码（Transformer）、软显著性分离、双查询注意力（显著查询+随机查询）、置信度调制、LLM（Qwen2.5-1.5B等）联合推理。

**📊 数据集**

使用四个公开多模态情感数据集：MOSI、MOSEI、CH-SIMS、CH-SIMS v2。

**📈 对比分析**

与传统方法、基于LLM的方法以及Vanilla SentiLLM做对比，SentiLLM在MOSI/MOSEI上MAE下降、Acc-2提升至89%，在CH-SIMS/CH-SIMS v2上Acc-2提升至80.96%/79.88%，显著优于所有基线。

**⚠️ 局限性**

局限包括对极端情绪类别仍难以区分（长尾问题）、对跨模态同步性仍有一定依赖、需要预先训练好的LLM且在非中文环境下表现略逊。

---

## 240. SpecCal: Ambiguity-Aware Candidate Calibration for Infrared Spectrum-Based Molecular Structure Reconstruction

**arXiv ID:** 2607.27788 | [PDF](https://arxiv.org/pdf/2607.27788v1)

**作者:** Yixuan Chen `[一作]` (Jilin University), Jun Xia `[通讯]` (HKUST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SpecCal，一个训练无关、plug‑and‑play的候选校准框架，用于从红外光谱重建分子结构。

**💡 创新点**

创新点在于利用蒙特卡罗树搜索与光谱一致性奖励，对基模型生成的候选集进行重排序和结构探索，有效缓解光谱模糊导致的重建歧义。

**🔧 技术方法**

使用的技术包括Transformer基底模型、Chemprop‑IR光谱预测、MCTS、MMR多样性筛选、信息相似度评估等。

**📊 数据集**

使用的数据集为公开的NIST气相红外光谱数据库和大规模合成SynSet数据集。

**📈 对比分析**

在多模型、多数据集上比较，SpecCal显著提升了Top‑1/Top‑5/Top‑10的SMILES与骨架匹配率，提升幅度约10%–30%，同时提高结构多样性。

**⚠️ 局限性**

局限在于推理时额外的计算开销，搜索过程耗时，并且对温度和λ等超参数敏感。

---

## 241. LoRA Scaffolded Policy Optimization (LSPO): A Sampling-Time Low-Rank Scaffold for Recovering Reinforcement-Learning Gradient on Zero-Reward Cliff Prompts

**arXiv ID:** 2607.27787 | [PDF](https://arxiv.org/pdf/2607.27787v1)

**作者:** Ken Ding `[一作]` `[通讯]` (NVIDIA), Ken Ding (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出LoRA Scaffolded Policy Optimization（LSPO），在强化学习时通过短暂的低秩适配器在“cliff”提示上恢复梯度，避免GRPO在所有样本均失败时梯度为零的问题。

**💡 创新点**

创新点在于：①将LoRA适配器作为仅在采样时使用的临时“支架”，不参与最终模型；②通过监督微调适配器，然后重新采样并将成功的rollout拼接回批次；③使用两优化器分离梯度流，仅将监督梯度传给适配器，RL梯度仅作用于基础模型；④利用重要性采样校正保证梯度不偏。

**🔧 技术方法**

使用技术包括LoRA低秩适配、GRPO/ DAPO策略梯度、重要性采样校正、两优化器分离、批次拼接与丢弃适配器。

**📊 数据集**

实验数据集为DeepMath‑103K，评估基准为MATH500、AIME 2024/25/26，使用DeepSeek‑R1‑Distill‑Qwen‑1.5B模型。

**📈 对比分析**

与完全相同配置的DAPO基线比较，采用5个种子、1000步报告，LSPO在所有16个pass@k单元上取得或匹配基线，平均提升3.8分，单个单元最高提升10.7分（AIME24/pass@4）。

**⚠️ 局限性**

局限性：仅在单一模型/数据/训练配方上验证；需要每个cliff提示的完整解答作为监督；适配器秩、SFT步数、拼接方式等参数对结果影响尚未充分探究。

---

## 242. Reasoning Consensus: Structural Ensembling of LLM Reasoning via Weighted DAG Aggregation

**arXiv ID:** 2607.27783 | [PDF](https://arxiv.org/pdf/2607.27783v1)

**作者:** Amruta Parulekar `[一作]` (University of Illinois Urbana-Champaign), Hari Sundaram `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多模型推理结构集成框架，将各模型链式思考转化为有向无环图并进行加权合并，输出最支持的共识推理子图及其支持度。

**💡 创新点**

创新点在于把推理步骤结构化为可聚合的DAG，并通过跨模型一致性计数对节点赋权，从而在不需要额外训练的情况下获得可审计的高质量推理图。

**🔧 技术方法**

使用链式思考采样、结构化提取器生成节点及束、嵌入相似度+LLM判定双重合并、加权推理分数和基于门控的正负支持组合，最终得到共识子图。

**📊 数据集**

在六个基准上评估，包括法令解释(SARA)、研究生级科学问答(GPQA‑D)、多步叙事推理(MuSR‑MM/OP/TA)、一阶逻辑推理(FOLIO)等。

**📈 对比分析**

与同等推理链预算的多模型多数投票以及单模型自一致性/单链CoT比较，跨模型加权集成在所有数据集上均超过多数投票，单模型时与自一致性持平或略优，最高提升达3.1%。

**⚠️ 局限性**

主要限制在于对提取器的依赖、节点合并时的LLM判定开销、以及高计算成本，且提取错误会影响结果。

---

## 243. RIPPLE: Generating Multi-Channel Phase, Not Recovering It

**arXiv ID:** 2607.27775 | [PDF](https://arxiv.org/pdf/2607.27775v1)

**作者:** Jaehyuk Lee `[一作]` (Korea University), Donghun Lee `[通讯]` (Korea University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文提出一种在多通道波形生成中直接学习相位而非传统的相位恢复方法，利用Griffin–Lim作为相位先验，并通过分离的阶段性流模型实现相位与幅度的联合生成；

**💡 创新点**

创新点在于将Griffin–Lim视为可供学习的相位先验，结合交叉通道相位差损失(IPD)和解耦的时间步长，使相位生成成为有结构的目标，而非单独恢复；

**🔧 技术方法**

使用技术包括：STFT分解为幅度、cos/sin相位三元组；Griffin–Lim相位先验；两阶段Rectified Flow（幅度流和相位流）并采用独立时间步；相位差损失和幅度加权；以及基于条件编码器的多模态训练；

**📊 数据集**

数据集涵盖两大物理域：一是Spatial LibriSpeech（FOA音频跨房间转移），二是SCEDC（地震三分量跨站点翻译）；

**📈 对比分析**

与传统方法（直接生成、Griffin–Lim恢复、BigVGAN等）和对照生成模型（ImmerseDiffusion、Diff‑SAGe、HEGGS、GAN）进行比较。结果显示，在相位敏感指标（FOA的空间角误差、地震的S波极化误差）上，RIPPLE显著优于基线，幅度与波形的基本质量也保持或略优；

**⚠️ 局限性**

局限性包括：假设源与目标共享相位结构，低信噪或结构被破坏时先验效用下降；评估仅基于客观指标，缺乏听感或物理实验验证；并且需额外的Griffin–Lim迭代和多步流计算，增加推理成本。

---

## 244. Simplifying Neural Networks During Training

**arXiv ID:** 2607.27854 | [PDF](https://arxiv.org/pdf/2607.27854v1)

**作者:** Lorenzo Sciandra `[一作]` (University of Turin), Roberto Esposito `[通讯]` (University of Turin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种在训练期间动态简化深度神经网络的框架，利用表示动态监测来识别特征提取层与分类层的分界点，并在此点之后将后置层替换为轻量级分类头。

**💡 创新点**

创新点在于将Neural Collapse（神经坍塌）与Tunnel Effect（通道效应）结合，首次使用逆Fisher准则（Inverse Fisher Criterion）作为在线可监测的拆分信号，实现无须后验分析即可在训练过程中确定网络裁剪时机与位置。

**🔧 技术方法**

技术手段包括：计算逆Fisher准则监测每层的表达式与分类能力差异；设定耐心窗口判断分界稳定性；在裁剪后接入全连接或池化+全连接的轻量化分类头；在实验中对MLP、VGG和ResNet三种架构进行验证。

**📊 数据集**

使用的数据集包括Fashion‑MNIST、CIFAR‑10、CIFAR‑100以及CUB，覆盖了从简单到中等复杂度的视觉分类任务。

**📈 对比分析**

与完整训练、线性探针、固定ETF、声明ETF以及EB‑LTH剪枝等方法对比，所提方法在保留相同或更高准确率的前提下，参数量减少最多可达94%，在大多数配置下取得了最优或竞争性平均排名。

**⚠️ 局限性**

局限性包括：当网络缺乏明显的特征提取‑分类层分界或逆Fisher曲线噪声过大时，裁剪决策可能失准；实验仅在卷积网络上验证，Transformer等架构可能需要不同的动态指标；并且未对极大规模数据集或任务进行验证。

---

## 245. Beyond Feeling Better: Capability-Sustaining Emotional Dialogue as a Longitudinal Research Paradigm

**arXiv ID:** 2607.27851 | [PDF](https://arxiv.org/pdf/2607.27851v1)

**作者:** Ming Wang `[一作]` (Northeastern University), Shi Feng `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

该论文提出了Capability‑Sustaining Emotional Dialogue（CSED）这一长期情感对话研究范式，并通过PRISMA‑ScR审计发现现有系统研究过度关注即时缓解，缺乏对用户能力维持与生命周期事件（非使用、再参与、过渡、终止）的考量；随后给出了六项设计承诺、四阶段评估框架及示例过程模型。

**💡 创新点**

创新点在于将用户情绪调节、主动应对、自主决策与社会连结这四大能力作为跨时序的核心目标，将对话系统的持续使用、过渡与终止纳入设计与治理之中，形成从即时响应到长期结果的完整评估与约束体系。

**🔧 技术方法**

技术方法包括：PRISMA‑ScR文献审计与功能编码、基于隐状态的过程模型、四阶段评估公式（响应、会话、纵向、终止）、约束式训练/安全强化学习、以及可解释的政策设计与生命周期治理流程。

**📊 数据集**

使用的数据集主要是：2019‑2026 年 ArXiv 上情感支持、共情对话、心理健康聊天机器人和 AI 陪伴系统的 91 篇论文样本；以及 ESConv 对话数据集 1,300 对话、18,376 支持者发言，其中对 300 个关键发言进行了功能编码。

**📈 对比分析**

方法上通过对比救济导向策略与能力激活策略，在响应、会话、纵向与终止四个尺度上使用量化指标（J_resp、J_conv、J_long、J_term）进行评估；目前论文提供理论框架与指标示例，尚未在真实模型上完成实验性性能验证。

**⚠️ 局限性**

局限性包括：需依赖代理指标与自我报告，面临隐私与监控风险；缺乏长期追踪实验数据，难以验证因果与持久效果；治理阈值与风险指标需与多元用户社区共同校准；不同承诺之间可能冲突，需在即时缓解与能力提升之间进行权衡。

---

## 246. Safety-Gated Agentic Supervisory Control on a Coupled Distillation Benchmark: Regime Map, Auditable Gate, and Co-Design Findings

**arXiv ID:** 2607.27849 | [PDF](https://arxiv.org/pdf/2607.27849v1)

**作者:** Christian Rosenthal `[一作]` `[通讯]`, Christian Rosenthal

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在单一二元蒸馏柱模型上实现了基于大型语言模型的监控层，并加入了基于规则的多约束拦截门，确保每次设定点建议都经过安全评估后才被执行。

**💡 创新点**

创新之处在于将开放权重 LLM 的提议与确定性分叉双子模拟器及多约束拦截门协同设计，既能提升在离稳态目标获取上的性能，又能在稳态扰动抑制时保持传统 MPC 的优势。

**🔧 技术方法**

使用了 Observer‑Optimizer‑Critic 循环（Observer 为结构化观测，Optimizer 为 OpenAI/DeepSeek‑V4‑Flash LLM 调用，Critic 为基于规则的验证），并在每个五分钟的监督周期内对提议进行 30 分钟的分叉双子前向仿真与约束检查。

**📊 数据集**

实验基准为 Skogestad 的 Column A 40 阶二元蒸馏柱，采用 16 个工况点、5 种扰动情景和 10 种随机种子；此外，还在 Nemotron‑3‑Super 120B 模型上完成了 1600‑cell 的跨族验证。

**📈 对比分析**

与 PID‑only（C0）、线性 MPC（C1）以及无门禁 LLM（C2）进行四向阶梯比较；结果显示，门禁 LLM 在目标获取方面（IAE 0.361）明显优于线性 MPC，但在稳态扰动抑制上（IAE 16.03）表现差强人意；门禁机制将最差 IAE 缩小约 10 倍。

**⚠️ 局限性**

局限在于仅验证了单列蒸馏模型，结果受模型族与采样温度、推理实现方式的影响，门禁不具备恢复功能，且在稳态调节场景下不具备优势；缺乏真实工厂验证和对其他多变量过程的泛化证明。

---

## 247. Nanoparticle Networks for Neuromorphic Computing

**arXiv ID:** 2607.27844 | [PDF](https://arxiv.org/pdf/2607.27844v1)

**作者:** Jonas Mensing `[一作]` (University of Münster), Andreas Heuer `[通讯]` (University of Münster)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文设计并理论分析了一种利用金属纳米颗粒网络及多电极静态控制实现的神经形态物理计算架构，探索了其非线性、记忆与可表达性三大性能指标。

**💡 创新点**

创新点在于：①用静态电极直接调节网络内部动力学，省去传统RC中的线性读取层；②提出以截止频率、氧化层屏蔽长度和结构异质性三参数调控网络非线性、记忆与表达度的设计规则；③通过引入分子层异质结打破空间对称性显著提升表达空间。

**🔧 技术方法**

主要技术包括：二维金属纳米颗粒平面网络、SiO₂/Si基底、8电极布置、静态电压控制、正统单电子隧道理论、Kinetic Monte Carlo仿真、有限差分电势求解以及高阶谐波与有效体积指标的计算。

**📊 数据集**

论文并未使用传统机器学习数据集，而是通过仿真得到的电流时域波形与频谱数据来评估网络性能；所有实验均在数值模型（如10×10至15×15晶格、不同氧化层厚度与电阻分布）下完成。

**📈 对比分析**

通过与传统Reservoir Computing的读取层对比，本文证明在截止频率工作点下网络可实现与基准时变XOR和NARMA任务相当的非线性映射；此外，随着网络尺寸与异质性增强，表达体积显著增长，显示出更高的可调性，但具体数值比较未给出。

**⚠️ 局限性**

局限性包括：①屏蔽长度限制了可调网络的有效规模，过大网络导致表达饱和；②薄氧化层网络缺乏负相位延迟与非易失性记忆；③实验实现需精细控制纳米颗粒排列与分子接头，制造复杂度高；④仅在模拟环境验证，缺乏实测硬件性能与噪声鲁棒性评估。

---

## 248. Learning to Understand Body Language from Flight through Robust 3D Avatar Placing

**arXiv ID:** 2607.27865 | [PDF](https://arxiv.org/pdf/2607.27865v1)

**作者:** Dragos Costea `[一作]` (National University of Science and Technology Politehnica Bucharest), Marius Leordeanu `[通讯]` (National University of Science and Technology Politehnica Bucharest)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建 Drones2BodyLanguage 数据集，并提出一种基于几何世界模型与帧不变仿射锚点的 3D 头像放置算法，用无人机 4K 视频合成远距人类肢体语言。

**💡 创新点**

创新点在于利用单目深度和语义分割提取锚点，再通过仿射重心和 SVD 拟合实现纹理缺失场景下的精确位置、尺度和旋转预测，从而在真实航拍视频中高精度合成人物。

**🔧 技术方法**

技术包括 SAM3 分割、单目深度估计、Shi‑Tomasi 角点 + Lucas‑Kanade 跟踪、锚点重心求解、SVD 拟合地面旋转、SMPL‑X 动作重定向、YOLO11x‑pose 检测以及 80 帧 17 点关键点的检测验证。

**📊 数据集**

使用了 8 条罗马尼亚航拍场景，产生 3,580 条 80 帧、17 点关键点的检测验证序列，涵盖 10 种意图、3 种动作来源（真实剪切、重定向、全合成）。

**📈 对比分析**

在 12 种骨架序列模型上进行场景/动作离散化分割实验，使用“placement”训练将零射击准确率从 0.48/0.32/0.20 提升到 0.74/0.67/0.58，混合源进一步达 0.757，并在两组未见真实视频上排名第一。

**⚠️ 局限性**

局限性包括：依赖局部刚性地面、无法无缝渲染真实剪切片、角点锚点对纹理缺失区域的鲁棒性有限，且仅验证了有限的角度与放置场景。

---

## 249. Benchmarking Foundation and Large Language Models for Few-Shot Medical Image Segmentation

**arXiv ID:** 2607.27856 | [PDF](https://arxiv.org/pdf/2607.27856v1)

**作者:** Jinghong Liu `[一作]` (Renmin University of China), Xirong Li `[通讯]` (Renmin University of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建统一的少样本医学图像分割基准（FS‑MIS），涵盖专用模型、SAM、CLIP 和 MLLM 四大类别，并在零样本、十样本以及负样本、分布偏移下进行评估。

**💡 创新点**

创新点在于：①提供覆盖 7 个解剖部位、9 种成像模态、14 种 ROI 的 14,958 个测试样本的完整基准；②引入负样本和目标缺失识别评估；③系统比较不同方法在支持利用、样本规模、语义迁移和缺失识别等方面的表现，并提出直接视觉适配优于提示/语义适配的结论。

**🔧 技术方法**

使用多来源数据集统一格式化、质量控制与去重；定义零样本与十样本两种评估模式；对 SAM、CLIP（CoOp、LoRA）与 MLLM（烧录/插件）采用各自适配策略；计算 Dice 与特异性等指标。

**📊 数据集**

从 19 个公开医学分割数据集收集数据，涵盖 7 个解剖部位（如肺、皮肤、甲状腺等）、9 种成像模态（CT、MRI、X‑ray 等）以及 14 种 ROI（肿瘤、器官、病灶等）。

**📈 对比分析**

在统一协议下对 10+ 方法进行比较：SAM‑MedSAM3（Dice 0.683）领先；GLaMM 与 MedPLIB 在 MLLM 中表现最好；CLIP 的 LoRA 版本明显优于仅调文本提示；直接视觉适配显著优于提示/语义适配；增加支持样本仅在模型能有效利用时才提升性能；语义迁移比协变量迁移更难；分割准确率与缺失识别能力并不一致。

**⚠️ 局限性**

仅评估了参数不超过 8B 的公开 MLLM，未涵盖更大规模模型或更先进的适配机制，未来工作需扩展至更大模型与更高效的适配方法。

---

## 250. FinanceHarness: Autonomous Financial Deep Research Framework

**arXiv ID:** 2607.27853 | [PDF](https://arxiv.org/pdf/2607.27853v1)

**作者:** Yijia Xiao `[一作]` (Google Cloud AI Research), Chen-Yu Lee `[通讯]` (Google Cloud AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个点时间（PIT）金融深度研究基准及其对应的工具 Harness，用于评估和训练金融分析型 LLM 代理。

**💡 创新点**

创新点在于：① 设计了基于发布日期的 PIT 搜索沙箱，② 将检索与前后期裁判分离（pre‑cutoff 与 post‑cutoff），③ 通过专家验证生成 400 题高质量问答和 2,464 条 rubric，④ 在同一环境下兼顾评估与强化学习。

**🔧 技术方法**

技术包括：检索增强生成（RAG）、ReAct 工具使用、FAISS 索引 + Qwen3-Embedding、LLM 评判（Gemini‑3.5‑Flash）、GRPO 强化学习以及多层次工具接口。

**📊 数据集**

数据集：100+M 条公开网页文章（带可靠发布日期）构成检索沙箱，生成 400 题金融情境与 2,464 条专家校验 rubric，覆盖 9 个主题、9 个行业、6 种推理类型。

**📈 对比分析**

比较方法：在相同 PIT 检索和相同评判 rubric 下评估 17 种基线（包括开源与专有模型、不同 scaffolding），结果显示所有系统整体得分低于 40%，且前后期得分差距显著，最强 27B 开源模型在本基准上获得约 32% 的整体得分。

**⚠️ 局限性**

局限性：语料仅为英文公开网页，缺少多语言、付费专业数据；模型训练使用的 live‑web 工具与评估时的 PIT 检索分布不一致，导致转移误差。

---

## 251. FeatFix: Reuse What You Verify through Local Exact-Feature Correction for Faster Cached Diffusion Inference

**arXiv ID:** 2607.27842 | [PDF](https://arxiv.org/pdf/2607.27842v1)

**作者:** Hanshuai Cui `[一作]` (Beijing Normal University), Weijia Jia `[通讯]` (Beijing Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在缓存式扩散生成中利用已支付的精确特征对草稿特征进行局部校正，以减小累积误差并提升速度

**💡 创新点**

将已计算的精确特征直接替换草稿特征而不是仅用作误差信号，实现了无额外计算的全块校正

**🔧 技术方法**

训练无关的本地特征校正框架（FeatFix），配合已存在的缓存/预测策略、全块前向替换、匹配计算验证

**📊 数据集**

FLUX.1-dev、Qwen-Image、HunyuanVideo-1.5、DiT-XL/2 等多种图像与视频扩散模型；评测集包括 DrawBench200、COCO-caption、ImageNet-256 等

**📈 对比分析**

与基线（原始 Vanilla、各种缓存/预测加速器）以及匹配计算的 Verify 对比，速度提升最高可达 6.70×，并保持甚至提高 ImageReward、PickScore、LPIPS 等质量指标

**⚠️ 局限性**

需要在开发阶段预先审计并固定校正点，缺乏动态、预算感知的自适应校正策略

---

## 252. Approximate Dual Separation for the Cluster LP: a 1.387 approximation for Correlation Clustering

**arXiv ID:** 2607.27829 | [PDF](https://arxiv.org/pdf/2607.27829v1)

**作者:** David García-Soriano `[一作]` (Universitat Politècnica de Catalunya), Antoine Schohn `[通讯]` (École Polytechnique)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在完整图上提供了一个+ε近似的关联聚类算法，改进了前沿1.485+ε的逼近因子；

**💡 创新点**

创新点在于独立提出了高效的弱分离 oracle 解决 cluster-LP 对偶，并设计了新的连续条件 pivot 轮询舍入方案；

**🔧 技术方法**

采用了弱正则化与稠密二次优化、局部化技巧、对偶优化、精确的变差不等式与计算机辅助多项式验证、以及稀疏线性规划求解等技术；

**📊 数据集**

本文主要为理论研究与随机化实验，未使用特定的实际数据集；

**📈 对比分析**

与Cao等人及其他前沿算法对比，在完整图中实现1+ε近似，时间为O(2^{1/ε}n)，近似因子接近已知下界4/3；

**⚠️ 局限性**

局限在于仍需指数因子 2^{1/ε}，对大规模图不友好，并且对稀疏图或非完整图的适用性有限。

---

## 253. STEREODISCO: Discovering Stereotypicality in LLMs

**arXiv ID:** 2607.27824 | [PDF](https://arxiv.org/pdf/2607.27824v1)

**作者:** Farane Jalali Farahani `[一作]` (Institute for Artificial Intelligence), Steffen Staab `[通讯]` (University of Southampton)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套框架，利用改进的语义差分法在大型语言模型内部表示中发现并量化刻板印象语义轴，进而评估模型对社会群体刻板印象的编码情况。

**💡 创新点**

创新点在于：①将语义差分法迁移至模型内部表征，系统挖掘海量候选对立轴；②通过统计检验（KS检验）识别刻板印象轴；③不依赖预先定义的刻板词典，能发现新的刻板轴；④利用线性探测定位注意力头中的几何轴。

**🔧 技术方法**

主要技术包括：WordNet反义词对生成候选轴；多模板句子生成并在LLM中提取头输出；质心差法（mass‑mean probing）得到几何轴；概念投影与z‑归一化；Kolmogorov‑Smirnov检验评估概念分布差异；聚合头权重以提升信号。

**📊 数据集**

数据集：WordNet 1999个反义词对（5,182个形容词）；社会群体标签来自Stereotype Content Model（25个群体）与相关属性词典（50 Warmth、42 Competence轴）；随机短语与社会群体频率匹配；人工标注的100个语义轴（哲学/心理、未知、技术三类）以检验刻板轴。

**📈 对比分析**

与人类基准的比较方法：①对齐社会群体在各轴上的投影与人类评分，使用准确率评估（Warmth≈0.55‑0.57，Competence≈0.62‑0.63）；②将模型识别出的刻板轴与人工评定的刻板轴对比，重合率约57‑59%；两大模型在轴一致性上相互匹配度更高（≈0.73‑0.75），优于与人类的一致性。

**⚠️ 局限性**

局限性：仅在英语环境下实验，依赖可解释的对立轴；缺乏对“理想”LLM刻板印象的客观基准；实验仅覆盖两款7–8B开源模型，未检验更大模型或不同训练目标；未对多领域、跨文化情境进行评估。

---

## 254. SPFM-Net: Semantic-Prior-Guided Frequency-Constrained Mamba for Invisible Watermark Attack

**arXiv ID:** 2607.27811 | [PDF](https://arxiv.org/pdf/2607.27811v1)

**作者:** Chunpeng Wang `[一作]` (Qilu University of Technology (Shandong Academy of Sciences)), Qi Li `[通讯]` (Qilu University of Technology (Shandong Academy of Sciences))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了SPFM-Net，一种基于语义先验和频域约束的全局状态空间特征建模网络，用于高效去除可见和不可见水印，同时保持图像视觉质量。

**💡 创新点**

创新点包括：①通过大比例随机遮罩打断水印空间连贯性；②利用冻结的预训练MAE提取语义先验，引导去水印恢复；③引入多尺度残差频率特征交互（MRFFI）与Mamba式全局状态空间特征建模（GSFM）两大模块，捕捉长程水印依赖；④采用多域联合损失（像素、FFT、Sobel边缘）平衡去水印效果与图像保真。

**🔧 技术方法**

技术手段包括：预训练MAE、随机遮罩、MRFFI、Mamba GSFM、转置卷积解码器、L1、FFT、Sobel联合损失以及AdamW优化。

**📊 数据集**

数据集：20,000张清洁人脸图像（用于训练传统水印），以及100张CelebA-HQ、Stable Diffusion Prompt、SDXL 1.0等用于评估未见深度水印的零样本泛化。

**📈 对比分析**

与传统信号处理攻击、DiffusionAttack、VAEAttack、UnMarker等基线进行比较，SPFM-Net在深度水印上达到0.48–0.55的BER（接近随机猜测），PSNR高达42.76 dB、SSIM 0.99、LPIPS 0.04，明显优于现有方法在攻击效果与视觉质量之间的折中。

**⚠️ 局限性**

局限性：在高度嵌入于高频细节的空间域水印（如Yu）下，因过度依赖语义先验，去水印效果略逊于无约束攻击，导致BER仅0.12；同时对极端动态水印结构的适应性仍需提升。

---

## 255. Short Cycles Decide P-versus-NPC Status ofHamiltonicity on Bisplit Graphs

**arXiv ID:** 2607.27802 | [PDF](https://arxiv.org/pdf/2607.27802v1)

**作者:** Mahendra Kumar R `[一作]` (Indian Institute of Information Technology Design and Manufacturing), Sadagopan N `[通讯]` (Indian Institute of Information Technology Design and Manufacturing)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 bisplit 图在弦图与弦双弦图两种情况，细致分析了 Hamiltonian 循环与路径问题的计算复杂性，并给出了在 P5‑free 与 P10‑free 的弦双弦图中的多项式解法与 NP‑完全性证明；进一步通过构造线性时间算法解决了 P5‑free 弦双弦图上的 Hamiltonian 变体（如全周期、同质可追踪、精确 2‑路径覆盖、最短叶生成树等）及其推广（最长路径、最长环）。

**💡 创新点**

提出“短环决定 P‑vs‑NPC”这一二分律：在弦 bisplit 图中仅含三角形导致多项式，含四边形导致 NP‑完全；证明 P5‑free 弦双弦图的 Hamiltonian 问题可线性求解，并将该框架推广至多种变体；首次给出 P10‑free 弦双弦图的 NP‑完全性证明，缩小了已知的复杂性边界。

**🔧 技术方法**

核心技术包括：从已知 NP‑完全的强弦 split 图或弦双弦图构造多项式时间可逆的归约；利用 bisplit 结构与弦图的最小分离器特性；对 P5‑free 弦双弦图引入 Nested Neighbourhood Ordering（NNO），从而得到一条线性时间的 Hamiltonian 判定与构造算法；对最长路径/环的求解，使用删枝（pruning）与构造的 Hamiltonian 路径/环实现线性时间最优解；对变体问题，利用构造的 Hamiltonian 结构与 NNO 进行可追踪性与覆盖性判定。

**📊 数据集**

本研究为纯理论分析，未使用实验数据集；所有结论均基于多项式时间归约与结构证明得到。

**📈 对比分析**

对 NP‑完全性的证明通过归约验证，构造性多项式算法的时间复杂度为 O(n+m)（线性时间），而变体问题的算法在最坏情况需要 O(n²)（主要是多次调用 Hamiltonian 判定）；相较于之前仅在特殊子类（如弦双弦图）得到的多项式结果，本工作大幅拓展了可解类，并提供了全新的算法框架。

**⚠️ 局限性**

局限性：仍存在 P5‑free 与 P10‑free 弦双弦图之间的复杂性空隙未被完全填补；对于更一般的弦双弦图（如不满足 NNO 或嵌套邻域序）的 Hamiltonian 问题仍处于未知状态；归约构造较为繁复，对实际实现与规模较大的实例可能存在常数因子影响。

---

## 256. Revisiting Predictive Process Monitoring in the Age of Foundation Models: A Comparative Study of Sequence, Tabular, and LLM Approaches

**arXiv ID:** 2607.27797 | [PDF](https://arxiv.org/pdf/2607.27797v1)

**作者:** Lennart Fertig `[一作]` (University of Mannheim), Tobias Sesterhenn `[通讯]` (Technical University of Clausthal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统比较了经典序列模型、LLM与表格基础模型在预测流程监控任务（下一个活动、剩余时间、下一个事件时间）上的表现。

**💡 创新点**

首次在概念和实验层面完整对比三种范式，并揭示序列模型在NA上的优势、表格模型在时序任务中的竞争力以及LLM的高成本低效能问题。

**🔧 技术方法**

使用LSTM、Transformer、LoRA微调的 Llama‑3.2‑1B、Gemma‑2‑2b 以及 TabPFN‑3、ConTextTab 等表格基础模型。

**📊 数据集**

五个公开事件日志：BPI12、BPI17、BPI20RfP、BPI20PTC、BPI20TPD。

**📈 对比分析**

在相同的80/20时间切分、统一特征预处理与评估指标（准确率、MSE、运行时）下，实验显示序列模型在NA上取得最高准确率，表格模型在RT上最具竞争力，LLM在NA略优但运行时显著更高，性能差异与流程分支复杂度相关。

**⚠️ 局限性**

局限性包括：LLM规模有限、未评估更大模型；仅考虑三项任务；缺少对其他特征表示的探索；TabPFN受上下文窗口限制；实验未覆盖更广泛的流程特征与多任务设置。

---

## 257. Green Cell-Free Massive MIMO for ISAC: Joint Cloud, Fronthaul and Radio Resource Allocation

**arXiv ID:** 2607.27778 | [PDF](https://arxiv.org/pdf/2607.27778v1)

**作者:** Zinat Behdad `[一作]` (KTH Royal Institute of Technology), Cicek Cavdar `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并提出了面向绿色 CF‑mMIMO ISAC 系统的全链路跨层优化框架，整合了分布式目标检测、AP 工作模式选择、用户与感知区域关联、RX‑AP 分配以及云端/前置链路资源分配等多域资源的联合调度。

**💡 创新点**

创新点：
- 设计了基于权重聚合的分布式 MAPRT 检测器，覆盖完整信息与部分信息两种场景；
- 将发射功率、AP 模式、关联决策与云/前置链路资源共同纳入同一混合整数非凸优化问题，并通过两阶段迭代（SOC/SCAD + FPP‑SCA）实现可行求解；
- 建立了闭式的电源、前置链路与云端计算三域交互模型，量化能耗与资源使用。

**🔧 技术方法**

所用技术包括：CF‑mMIMO/ISAC、分布式目标检测、MAPRT、功率分配、AP 模式与关联优化、SOC/SCAD 约束、FPP‑SCA 近似、GOPS 计算、云端与前置链路能耗建模、权重聚合策略。

**📊 数据集**

实验使用 3GPP Urban Microcell 信道模型生成的随机部署数据：4 个 SSA、25 个 4 天线 AP、8 个 UE，随机布置并按论文给出的参数（LOS、Rician、RCS、量化位等）进行仿真。

**📈 对比分析**

与四种基准（本地/全协调下的发射功率优化、无线电优化）对比，E2E 框架在通信‑仅方案基础上可降低 600 W 以上功耗，整体节能超过 50%；在保持目标检测概率 >0.9 的前提下，E2E 方案比基准实现了 15%–56% 的能效提升；对不同 AP/UE 密度、SE 阈值、感知 SINR、RX‑AP 数量的敏感性分析表明 E2E 方案在多种部署与负载下均表现出更佳的能效与鲁棒性。

**⚠️ 局限性**

局限性：
- FIS 场景前置链路负载极高，需更高容量链路；
- PIS 在低感知 SINR 时检测性能下降；
- 算法迭代复杂度高，适用于大规模网络的实时实现尚待验证；
- 仅考虑静态多目标情形，未覆盖动态目标或多时隙场景。

---

## 258. Creative Task Cards for Reflection, Self-Efficiency and Self-Regulation in CS1 Introductory Programming: Initial Insights

**arXiv ID:** 2607.27863 | [PDF](https://arxiv.org/pdf/2607.27863v1)

**作者:** Corey Ford `[一作]` (University of Arts London), Rosa Van Koningsbruggen `[通讯]` (Bauhaus-Universität Weimar)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发并评估了一套创意任务卡，帮助 CS1 编程学生通过同伴讨论反思情绪，提升自我效能与自我调节能力。

**💡 创新点**

将学习科学与艺术手段结合，设计情绪反思和自我调节卡片，并通过同伴讨论提供情绪表达与任务分解的空间；首次验证卡片在情绪调节与学习策略选择中的作用。

**🔧 技术方法**

使用 InDesign 设计卡片，采用 PANAS 问卷收集情绪基线，开展混合方法评估（焦点小组、1 对 1 访谈、访谈记录分析）。

**📊 数据集**

收集了 29 名 CS1 学生的问卷数据、PANAS 结果以及访谈和焦点小组记录；未使用公开数据集。

**📈 对比分析**

由于缺乏对照组，评估主要基于质性分析和描述性统计；初步观察显示卡片能减轻情绪负担、促进任务分解，但尚未量化对学习成绩或长期效果的具体提升。

**⚠️ 局限性**

样本规模小、无对照组、只获得早期定性数据、未测量学业成效、缺乏长期跟踪、卡片活动的时间成本与收益未做平衡等限制。

---

## 259. Back to All-Entity Ranking: Sampler-Dependent Evaluation in Continuous-Time Dynamic Graphs

**arXiv ID:** 2607.27861 | [PDF](https://arxiv.org/pdf/2607.27861v1)

**作者:** Minwoo Yu `[一作]` (Konkuk University), Young-guk Ha `[通讯]` (Konkuk University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在连续时间动态图（CTDG）的下一目标预测任务中，系统性分析了采样负例评估对模型排名和模块效果的影响，并提出了一种消除采样自由度、仅依赖固定目标集的全实体排名评估方法。

**💡 创新点**

①推导采样负例的贝叶斯最优评分，证明非均匀采样会改变排名；②揭示时间变化的对偶子成员信息会通过模型内部机制直接影响得分；③设计 2×2 成员划分与正交投影表示干预实验，以分离成员效应与语义兼容性；④基于以上结果，提出在固定可枚举目标集上进行全实体排名作为 CTDG 评估的首选标准。

**🔧 技术方法**

贝叶斯推理与噪声对比估计理论、全实体排名评估、2×2 成员划分与正交投影干预、六种 CTDG 模型（CRAFT、CRAFT‑R、DyGFormer、DyGFormer+LHA、TGN、GraphMixer）的一致训练与评估框架。

**📊 数据集**

JODIE 提供的四个公开时间序列图数据集：Wikipedia、Reddit、MOOC 与 LastFM。

**📈 对比分析**

在统一训练条件下，使用全实体排名与采样负例（Uniform‑20、不同 K 值）进行对比；实验表明：不同采样策略和候选集大小会导致模型排名与模块效果产生显著变化，采样评估往往与全实体评估不一致，尤其在 LastFM、MOOC 与 Wikipedia 数据集上。

**⚠️ 局限性**

①全实体评估仅适用于可枚举且规模不大的目标集，无法直接扩展到百万级实体；②全实体评估的计算成本随实体数线性增长，虽然在本实验规模可行，但大规模部署仍需近似方法；③实验仅涵盖四个数据集，未验证在其他图结构或任务上的泛化性。

---

## 260. EEG-EditBench: Probing Visual Information in EEG-Image Retrieval Models with Controlled Image Edits

**arXiv ID:** 2607.27857 | [PDF](https://arxiv.org/pdf/2607.27857v1)

**作者:** Kaifan Zhang `[一作]` (Xidian University), Xinbo Gao `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

引入了 EEG‑EditBench 诊断基准，通过对 EEG‑to‑image 检索模型进行控制编辑（身份、属性、背景、去除）评估其视觉信息保留能力。

**💡 创新点**

通过构建质量控制的编辑图像集合，将标准的 200‑way 评估替换为同一源图像的细粒度对比，揭示模型在细节级别的匹配差异。

**🔧 技术方法**

采用场景结构化描述、提示生成与基于 Vision‑Language 模型的图像编辑、以及人类质量审核，结合现有 EEG‑to‑image 检索模型进行评估。

**📊 数据集**

基于 THINGS‑EEG2 200 个测试图像生成 2,137 个控制编辑，涵盖身份、属性、背景、去除四类。

**📈 对比分析**

对八个代表性模型在原始 200‑way、null‑image 200‑way、Edit‑Pool Top‑1、2AFC 评估；结果显示高标准检索并不保证在编辑对比中准确，属性变更最难，Brain‑HIVE 在所有指标中表现最佳。

**⚠️ 局限性**

只评估图像侧编辑，缺少对应 EEG 记录；每个概念仅有单张图像，缺乏内部多样性，且编辑模型与人类感知差异可能影响结论。

---

## 261. Crossing the Margin Cliff: Toward Relearn-Robust LLM Unlearning via Margin Calibration

**arXiv ID:** 2607.27836 | [PDF](https://arxiv.org/pdf/2607.27836v1)

**作者:** Xiangyu Yin `[一作]` (Chalmers University of Technology), Chih-Hong Cheng `[通讯]` (Chalmers University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了大型语言模型在被删除（unlearning）后容易被少量记忆样本重新学习（relearn）所导致的脆弱性，并提出一种名为 Margin Calibration (MC) 的后置微调方案，利用非饱和的单边 hinge 与 KL 探针在保留侧保持约束的同时，跨越所有已知方法停留的“margin cliff”，显著提升无忘模型在多种攻击与基准下的稳健性。

**💡 创新点**

创新点在于①首次将“margin cliff”作为诊断指标揭示所有十四种梯度、偏好与蒸馏无忘方法在训练终点均停留在留存参考以上的阈值区间，并通过 KKT/梯度饱和理论解释这一现象；②提出 MC 通过在金字塔式 margin 目标上加 softplus 单边 hinge 与 KL 探针实现对忘记侧梯度的持续压制，从而在不改动原始无忘目标的前提下实现跨越 cliff 的通用后置微调；③在一次实验矩阵中统一评估 14 种方法、3 大小 Llama‑3、3 级忘记、3 种攻击、2 其他基准，首次展示了统一配置下的 100% 复原率。

**🔧 技术方法**

技术手段包括：KKT/梯度饱和理论分析、logit margin 诊断、softplus 单边 hinge、KL 探针、LoRA 微调、单步加权调参、以及对攻击模型（LoRA、soft‑prompt、全参数微调）的系统评估。

**📊 数据集**

使用数据集：TOFU（包含 forget 与 retain 语料及对应参考）、MUSE‑News（新闻问答基准）、Phi‑3.5‑mini（另一模型族）、Alpaca（KL 探针独立语料）。

**📈 对比分析**

在 14 种无忘方法、3 Llama‑3 大小、3 忘记级别、3 种种子、MUSE‑News、Phi‑3.5、以及 LoRA、soft‑prompt、全参数微调三类攻击共 97 个交叉轴上进行比较。MC 在所有被攻击单元中实现 100% 获胜率，ROUGE‑L 恢复率从 0.41 降至 0.18，Membership AUC 降低 13/14；虽导致保留侧实用性 (MU) 明显下降，但部署变体无参考时保持几乎同等效果。

**⚠️ 局限性**

主要局限包括：①MC 需牺牲大量保留侧实用性，MU 下降明显；②部分方法（UNDIAL）未能跨越 margin cliff；③过度压制 margin 可能导致 membership 反向可分离；④对 MMLU 影响不均衡；⑤缺乏精细的停止规则；⑥对更大模型和其他基准的泛化仍待验证。

---

## 262. Learning-Augmented and Randomized Algorithms for Line Aggregation with Delays

**arXiv ID:** 2607.27807 | [PDF](https://arxiv.org/pdf/2607.27807v1)

**作者:** Tianhang Lu `[一作]` (Southern University of Science and Technology), Ke Tang `[通讯]` (Southern University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

未提供论文内容

**💡 创新点**

未提供论文内容

**🔧 技术方法**

未提供论文内容

**📊 数据集**

未提供论文内容

**📈 对比分析**

未提供论文内容

**⚠️ 局限性**

未提供论文内容

---

## 263. LoMeVQA: A Comprehensive Benchmark for Longitudinal Medical VQA

**arXiv ID:** 2607.27806 | [PDF](https://arxiv.org/pdf/2607.27806v1)

**作者:** Zhilin Wu `[一作]` (Tongji University), Lianghua He `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了LoMeVQA，构建了包含206K条纵向医学图像问答对的多任务基准；

**💡 创新点**

通过将医学知识图谱与大语言模型相结合，自动生成高质量的纵向视觉问答数据，并基于此开发了专门的MedLong-8B模型；

**🔧 技术方法**

利用RadGraph提取实体、Llama-Factory+LoRA微调、LLM问答生成、IoU/ROUGE/METEOR等多种评估指标；

**📊 数据集**

使用MIMIC‑CXR原始数据构建纵向病例，随后生成LoMeVQA-dev与LoMeVQA-test，另外在MMXU-test和MIMIC‑CXR‑T上进行OOD评估；

**📈 对比分析**

与多种通用与医学专用MLLMs（如GPT‑4o、Gemini、LLaVA、InternVL、Qwen系列、Lingshu、MedGemma）对比，MedLong‑8B在所有五个任务上均显著优于基线，OOS表现也领先；

**⚠️ 局限性**

仍然在时间推理、微小变化定位和视觉‑文本对齐方面存在欠缺，尤其是差异区域定位IoU仍低；数据生成仍受Hallucination与一致性过滤的限制。

---

## 264. MemeBench: What LVLMs Miss When Interpreting Culture-Dependent Memes

**arXiv ID:** 2607.27798 | [PDF](https://arxiv.org/pdf/2607.27798v1)

**作者:** Weihang Wang `[一作]` (Bilibili), Zhouhui Lian `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MemeBench，一套双语（中英）meme 诊断基准，并设计了 VIKR 四维需求框架与实体引导检索方法 KAR，评估 LVLM 对 meme 的完整解释能力。

**💡 创新点**

创新点在于将解释拆解为视觉线索、身份链接、知识单元和推理机制四层，既可精准定位缺失信息，又可通过实体检索有针对性地补充知识，从而实现诊断驱动的改进。

**🔧 技术方法**

使用视觉语言模型（如 Gemini、Qwen3-VL 等）进行自动评估，借助 CultureBase 进行多语言实体嵌入检索，采用 BGE-m3 多模态嵌入、Tavily Web 搜索、两阶段检索流程，以及 GPT‑5 / Gemini‑3.1‑Pro 作为评判者。

**📊 数据集**

构建了 1,253 条中英文 meme 数据集，来源于 Bilibili、Reddit 与 ImgFlip，包含 2,072 个实体，涵盖 ACG 及跨领域文化，提供细粒度的 VIKR 注释。

**📈 对比分析**

在闭包图像解释任务中对 26 个 LVLM 进行 VIKR 逐层评估，Gemini‑3.1‑Pro 最高成功率 60.3%；在四模型上使用 KAR，可将知识与身份覆盖提升 3–5%，并在成功率上实现 3.6–7.4% 的提升，同时保持较高的损失/破坏比。

**⚠️ 局限性**

局限性包括视觉覆盖下降风险、检索对实体匹配质量高度依赖、数据集仍以 ACG 为主导致长尾实体覆盖不足，以及评判者一致性受限于自动判定标准。

---

## 265. DexDirect: Direct Kinesthetic Arm Guidance for Efficient Dexterous Demonstration Collection

**arXiv ID:** 2607.27784 | [PDF](https://arxiv.org/pdf/2607.27784v1)

**作者:** Beom Jun Kim `[一作]` (UCLA), Dennis W. Hong `[通讯]` (UCLA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种低成本接口 DexDirect，结合直接的触觉驱动手臂指导与基于摄像机的手部重定向，用于高效收集多指机器人抓取与操作演示。

**💡 创新点**

创新点在于：①消除臂侧姿态跟踪与逆运动学环节，仅用重力补偿的物理手柄直接驱动机械臂；②将视觉手部重定向与臂部物理交互分离，避免手势与机器人全局坐标映射的认知负担；③实现一体化多模态记录，支持后续扩散模型学习。

**🔧 技术方法**

技术包括：Gravity-compensated 6-DoF 机器人臂的零位置增益控制；MediaPipe Hands + AnyTeleop 视觉重定向；多模态同步记录（关节角度、触觉、RGB图像）；DINOv2 ViT 视觉编码；基于 Diffusion 的控制策略。

**📊 数据集**

使用 10 名无经验参与者在 5 个对接/抓取/擦拭/键盘/拔插等任务中收集的数据；实验室自制数据集；无公开公开数据集。

**📈 对比分析**

与 AnyTeleop（全视觉）和 TeleDex（手持设备+视觉）对比，DexDirect 在相同时间预算下收集 481 次成功演示（比 TeleDex 高 3.2 倍、比 AnyTeleop 高 17.2 倍），成功率均为最高（0.71–0.96），完成时间平均缩短 2–5 倍，NASA-TLX 工作负荷得分最低（2.01 vs 3.08/3.81）。

**⚠️ 局限性**

局限性：需要可安全低阻尼、重力补偿的近场机器人，无法远程操作；双手操作不支持双手同步控制；手部重定向受单目跟踪误差与姿态缺失影响，缺乏触觉反馈；实验规模有限，未深入剖析认知负担、延迟等单独因素。

---

## 266. ChronoMem: Version Control and Semantic Rollback for Large Language Model Agent Memory

**arXiv ID:** 2607.27773 | [PDF](https://arxiv.org/pdf/2607.27773v1)

**作者:** Yongye Su `[一作]` (Purdue University), Elisa Bertino `[通讯]` (Purdue University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为LLM代理实现了可版本化的长时记忆，并提供了自然语言“undo”语义回滚功能。

**💡 创新点**

首次在开源代理框架中实现全内存快照版本控制，结合混合检索和重排序实现自然语言回滚；同时引入后置暴露评估协议。

**🔧 技术方法**

基于Google ADK框架，使用事件溯源+快照的版本控制，SQLite存储、FTS5 +向量检索混合、交叉编码reranker；提供原子回滚API；使用LoCoMo与MemoryAgentBench数据集。

**📊 数据集**

LoCoMo（长时对话问答+总结）与MemoryAgentBench（准确检索等任务）通过后置暴露回滚协议。

**📈 对比分析**

与prompt-only、full-history prompt、retrieval-only baseline 对比；在版本选择 Recall@1 约20%/33%；在回滚后QA F1提升约10pp；总结ROUGE-1同样提升8-10pp。

**⚠️ 局限性**

限制：单写串行、SQLite锁导致并发瓶颈；仅线性历史，无分支/合并；评估基于改造的 benchmark，真实用户回滚意图覆盖不全。

---

## 267. Virtual Process Dossier: A Process-Aware Data Catalogue

**arXiv ID:** 2607.27840 | [PDF](https://arxiv.org/pdf/2607.27840v1)

**作者:** Lukas Kubelka `[一作]` (Karlsruhe Institute of Technology), Tobias Käfer `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了Virtual Process Dossier (VPD)，一种基于知识图谱的数据目录，能够自动捕获并关联多阶段制造工作流中的传感器数据及其工作流原始信息；

**💡 创新点**

创新点在于将WiLD、PROV、SSN/SOSA和DCAT等成熟本体融合，形成专门的VPD本体，实现流程感知的 FAIR 数据管理；提供了从工作流建模到运行时数据采集的一整套框架和基于 SHACL 的交互式 UI；

**🔧 技术方法**

采用 RDF/OWL 进行本体建模，利用 SPARQL/Update 与三元组存储交互；使用 SHACL 生成 HTML 表单；利用 QUDT 表示量纲；通过 FOOPS! 工具验证 FAIR 合规性；

**📊 数据集**

论文以纤维增强塑料部件的两步制造流程为示例，使用传感器记录的温度、夹具角度等数据；未提供公开大规模数据集；

**📈 对比分析**

本文未给出量化实验或与现有方法的性能对比，只是通过示例说明架构和流程；

**⚠️ 局限性**

局限性包括：缺乏对大规模真实制造数据的评估；对工作流与本体映射的手工工作量未量化；在高并发、多机房场景下的性能与可靠性未展开讨论。

---

## 268. SAFViT: Spatial Attention Fusion Gating for Vision Transformer-Based Nucleus Segmentation and Classification

**arXiv ID:** 2607.27835 | [PDF](https://arxiv.org/pdf/2607.27835v1)

**作者:** Harshit Mittal `[一作]` (University of Leeds), Arash Rabbani `[通讯]` (University of Leeds)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对细胞分割与分类，提出并实现了一种基于CellViT的空间注意力融合（SAF）门控网络。

**💡 创新点**

创新点在于将编码器与解码器特征通过双流注意力生成热力图，动态决定每个像素的“信任度”，实现对多尺度信息的自适应融合。

**🔧 技术方法**

技术上结合Swin‑Tiny ViT编码器、解码器、三头预测（核位置信息、距离、细胞类型）以及两次1×1卷积与通道softmax的SAF门控。

**📊 数据集**

使用了PanNuke多癌种细胞实例分割数据集以及MoNuSeg单类实例分割数据集进行训练与跨域验证。

**📈 对比分析**

与六种对齐的门控（无门控、Attention Gates、SE、CBAM、Cross‑Attention、AFF）在3折交叉验证下对比，SAFViT在多类别panoptic质量mPQ上最高（0.471），并显著提升少数类“死亡细胞”的F1分数。

**⚠️ 局限性**

局限性在于死亡细胞类别极其稀少，仅占数据不到2%，导致F1改进受样本波动影响；同时门控比例固定，未针对不同解码层自适应调节。

---

## 269. MemTxn: A Transaction Boundary for Source-Supported Updates and Complete-State Recovery in Agent Memory

**arXiv ID:** 2607.27834 | [PDF](https://arxiv.org/pdf/2607.27834v1)

**作者:** Hanshuai Cui `[一作]` (Beijing Normal University), Weijia Jia `[通讯]` (Beijing Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个面向答案模型的可审计事务边界，用来管理可写代理记忆，包括写入审批、冲突可视化和完整状态恢复。

**💡 创新点**

创新点在于：① 引入 Ordered PatchTest 检验源支持；② 使用 Temporal Resolver 依据声明时序选择可见版本；③ 通过持久快照日志实现无物理写集的完整状态恢复；所有这些机制在外部可观察且可审计。

**🔧 技术方法**

技术手段包括：可写记忆的事务层、Ordered PatchTest（源支持检验）、Temporal Resolver（时序冲突解决）、持久化的快照日志（恢复）、SQLite WAL 作为底层存储。

**📊 数据集**

使用的基准数据集包括：LongMemEval‑S、LoCoMo、MemoryAgentBench FactConsolidation、LongMemEval‑S 与 LoCoMo 的持久故障测试、以及对 12 个不同回答模型配置的 800 个 QA 题。

**📈 对比分析**

与 Dense、Mem0、A‑Mem、Zep、LightMem 等传统检索/版本记忆方法对比，MemTxn 在 FactConsolidation 上平均 F1 提升 17.06–24.07 分点；在匹配 top‑8 控制下提升 15.01–22.93 分点；在多模型、多 hop/上下文场景均保持正向增益。

**⚠️ 局限性**

局限性包括：不保证语义真理、无法处理并发或重复故障、未覆盖自然输入的更新触发、依赖对源文本的访问、对物理存储损失无恢复方案、以及在生产环境中需要额外的合规与安全控制。

---

## 270. Annotating Topical Legal Insights from Case Proceedings

**arXiv ID:** 2607.27792 | [PDF](https://arxiv.org/pdf/2607.27792v1)

**作者:** Subinay Adhikary `[一作]` (Indian Institute of Science Education and Research Kolkata), Kripabandhu Ghosh `[通讯]` (Indian Institute of Science Education and Research Kolkata)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一款名为 LeDA 的法律数据注释系统，支持动态创建新标签、双视图标注与仲裁，并通过交互式界面完成案例判决文本的细粒度概念标注。

**💡 创新点**

创新点在于：①允许注释者在无预定义本体的情况下即时新增标签；②实现多标注者间的元注释与冲突仲裁；③提出针对法律文本的 IAA 计算方法；④将标注结果组织为“概念袋”，为后续检索与判决预测奠定基础。

**🔧 技术方法**

技术实现采用前端 HTML/CSS/JavaScript 与后端 Django，部署在 PythonAnywhere；注释结果以 JSON 格式存储；系统集成了多标签、高亮、搜索及 IAA 统计功能。

**📊 数据集**

使用印度最高法院判决文档共 200 篇，三名法律专家对其进行标注，构建了带有动态标签的法律概念数据集。

**📈 对比分析**

与现有工具 BRAT、GATE、Label Studio、UBIAI 进行功能对比，LeDA 在多标签支持、动态标签、元注释、IAA 计算及远程访问方面表现出色；虽然未给出数值指标，但通过实际使用反馈证明其功能满足专业需求。

**⚠️ 局限性**

局限性包括：目前仅聚焦于谋杀相关案例；标注工作仍需人工，成本高昂；数据集范围局限于印度最高法院判决，缺乏跨司法域通用性；未来需进一步自动化概念抽取并扩展至其他法律领域。

---

## 271. From Understanding to Action: Feedback-Grounded Policy Discovery for Generative Recommendation

**arXiv ID:** 2607.27789 | [PDF](https://arxiv.org/pdf/2607.27789v1)

**作者:** Zhi Chen `[一作]` (Huazhong Agricultural University), Peng Jiang `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于反馈驱动的政策发现框架，将用户意图（intent）与推荐策略（policy）分离，利用大语言模型（LLM）生成意图与策略文本，并通过增益评估、执行器和反馈代理迭代选择有效策略；随后使用双空间关系蒸馏将意图与策略知识迁移到轻量级Semantic-ID生成器，实现LLM-free在线推荐。

**💡 创新点**

创新点在于识别并填补“Understanding–Action Gap”，通过优势（advantage）评估和反馈循环自动发现真正有效的推荐策略；以及提出双空间关系蒸馏（第一阶与高阶关系）将异构的意图/策略知识迁移到低成本模型。

**🔧 技术方法**

技术方法包括：LLM教师生成意图与策略文本；基于执行器的增益评估和策略迭代；反馈代理进行策略演化；双空间关系蒸馏（保持一阶和高阶用户关系）；Semantic-ID（SID）序列生成模型。

**📊 数据集**

实验数据集：Amazon Review（Beauty、Toys and Games、Sports and Outdoors）以及工业级在线广告系统的13.25M用户、1.61M物品。

**📈 对比分析**

与传统序列推荐器（Caser、GRU4Rec、SASRec、BERT4Rec、HGN、P5）、SID生成器（TIGER、LETTER）和直接LLM推荐器比较；在离线Recall@5/10、NDCG@5/10上均有显著提升，线上A/B测试提升Revenue 4.506%、ADVV 4.621%。

**⚠️ 局限性**

局限性：依赖大量离线LLM生成与关系蒸馏，需要定期更新教师模型；对稀疏或冷启动场景下的策略生成效果可能不足；对快速变化的用户即时需求适应性相对有限。

---

## 272. CXR-Retrieve: Compositional Text-to-Image Retrieval in Chest Radiography

**arXiv ID:** 2607.27779 | [PDF](https://arxiv.org/pdf/2607.27779v1)

**作者:** Tomer Erez `[一作]` (Technion Israel Institute Of Technology), Ehud Rivlin `[通讯]` (Technion Israel Institute Of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CXR‑Retrieve基准，研究胸部X光图像的文本检索，尤其是并列与否定查询。

**💡 创新点**

创新点在于使用标签感知的对比损失，既吸引相同临床断言的图像文本对，又用硬负样本挖掘显式排斥对，直接让模型学习临床逻辑。

**🔧 技术方法**

方法基于CLIP双编码器，采用Swin‑T图像编码器和Bio_ClinicalBERT文本编码器，并通过LoRA进行微调，损失包括标签对比吸引项与硬负挖掘惩罚。

**📊 数据集**

使用MIMIC‑CXR‑JPG数据集，挑选10个正标签、145条合成查询（单标签、并列、否定），共5,159张测试图像。

**📈 对比分析**

与通用CLIP、BioMedCLIP以及在同一数据集上预训练的CXR‑CLIP比较，发现新方法在并列查询的P@5从11.1%提升至19.6%，在否定查询的P@5从20.0%提升至42.0%，并将硬负检索率从3.7%降至1.6%。

**⚠️ 局限性**

局限性包括：对否定查询的严格定义（需确认负标签）性能仍不理想，模型主要训练于模板化查询，未涵盖缩写或区域术语；且在未标记标签占比高的情况下，未提取到足够的负信息。

---

## 273. CHARGE: Leveraging CWE Hierarchies for Hardware Security SystemVerilog Assertion Generation

**arXiv ID:** 2607.27776 | [PDF](https://arxiv.org/pdf/2607.27776v1)

**作者:** Xiao Tan `[一作]` (UNC Chapel Hill), Cynthia Sturton `[通讯]` (UNC Chapel Hill)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

自动生成安全系统Verilog断言（SVA），无需手工安全规范，直接从未验证的RTL和CWE层级信息生成；

**💡 创新点**

创新地将CWE层级结构与三元组抽象相结合，指导LLM进行资产识别、行为挖掘和SVA生成；

**🔧 技术方法**

采用大型语言模型（GPT‑4.1/ Gemini/ Claude 等）+ CWE数据库 + 3元组抽象 + 自动化三步推理流程，并在 Cadence JasperGold FPV 中验证；

**📊 数据集**

使用 Hack@DAC 2018/2019/2021 公开SoC RTL 及其对应的 Verification Benchmarks 手工属性，CWE‑1194 硬件视图作为知识库；

**📈 对比分析**

与开源手工属性对比，CHARGE 在 42 个已知 bug 中检测 27 个（含 1 个新发现 bug），在 Hack@DAC 2021 设计上对比 baseline，提升至 11/14 的 bug 检测率，资产识别准确率从 21/41 提升至 30/41；

**⚠️ 局限性**

受限于某些 CWI 不能被完整建模、RTL 模块过大导致失效、需商业 LLM、未覆盖所有 CWE、对资产/行为的误判仍存在。

---

## 274. AutoSupervision: Closing the Feedback Loop in Scientific Workflows with Grounded Revision Verification

**arXiv ID:** 2607.27845 | [PDF](https://arxiv.org/pdf/2607.27845v1)

**作者:** Haobo Li `[一作]` (Shanghai AI Laboratory), Lei Bai `[通讯]` (Shanghai AI Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为 AutoSupervision 的基准任务，评估AI在科学稿件修订中验证审稿人反馈是否被充分满足并找出支持证据。

**💡 创新点**

提出了基于审稿人评论、作者回复和修订稿的多级验证框架，首次将证据定位与修订结果直接关联，并为此提供了大规模标注数据与评估指标。

**🔧 技术方法**

采用大型语言模型（如GPT‑5.5、Claude‑Opus‑4.8 等）进行结构化推理，同时探究了检索-分解-结构化管线（agentic inference）和任务专属监督微调（SFT）的提升效果。

**📊 数据集**

使用 56,000 篇 Nature Communications 公开同行评审记录，构成 8,790 个修订实例、6,543 条独立关切点，包含审稿意见、作者回应与修订稿块的精确对齐。

**📈 对比分析**

在多种闭源与开源 LLM 上评测，最佳整体分数为 0.637（GPT‑5.5/Claude‑Opus‑4.8）。模型在关切识别与特征化上表现优异（≥0.75），但在验证与证据定位上仍低于 0.5，表明这两项仍是主要瓶颈。监督微调将 SFT Qwen3.5‑9B 的整体分数从 0.463 提升至 0.614，验证与定位能力均有显著提升。

**⚠️ 局限性**

限制主要包括：数据仅来源于成功完成修订的 Nature Communications 论文，缺乏未通过或拒稿案例；只关注验证过程，未覆盖生成修订建议、优先级排序等实际编辑功能；以及对多领域、不同审稿文化的适用性尚未验证。

---

## 275. VCP-DCN: Beyond Visual Concealed Property via Depth Collaborative Network for Camouflaged Object Detection

**arXiv ID:** 2607.27843 | [PDF](https://arxiv.org/pdf/2607.27843v1)

**作者:** Songsong Duan `[一作]` (Xidian University), Nannan Wang `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了基于深度协同网络的隐形目标检测模型VCP‑DCN，能够利用RGB与深度信息共同学习更具辨别力的多模态特征。

**💡 创新点**

创新点包括：①分离原型嵌入（SPE）实现模态一致性与模态特异性原型的对齐；②双模态注意（MDA）通过前景/背景掩码线性注意实现跨模态交互；③深度自适应注入（DAI）利用原型余弦相似度动态融合模态特征；④在所有阶段引入原型对比学习以防止RGB‑Depth特征同质化。

**🔧 技术方法**

核心技术包括：视觉Transformer编码器（VMamba‑S / Swin‑S）、分离原型嵌入、双模态注意、深度自适应注入、三分支解码器、混合损失（BCE+IoU+SSIM）以及对比损失。

**📊 数据集**

实验使用了CAMO、COD10K、NC4K三大COD基准，以及NJU2K、NLPR、DUT等RGB‑D SOD数据集进行跨任务验证。

**📈 对比分析**

与15种SOTA COD/RGB‑D方法（包括Samba、CamoDiffusion、FSEL、CamoFormer等）在S_m、E_ϕ、Fβ^w、MAE等指标上均实现了领先或接近领先的成绩，同时模型参数仅60.2M、FLOPs 46.4G，显示出更高的效率与效果。

**⚠️ 局限性**

局限性：对深度图质量高度依赖；在极小或背景极其复杂的隐形目标上仍可能失真；跨域泛化能力尚需进一步验证。

---

## 276. Hallucinations Leave a Grounding Signature:Verifier-Guided Decoding for Selective Object Correction

**arXiv ID:** 2607.27823 | [PDF](https://arxiv.org/pdf/2607.27823v1)

**作者:** Lei Yang `[一作]` (Chinese Academy of Sciences), Zheng Lin `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了大型视觉语言模型（LVLM）在生成文本时出现的对象幻觉问题，提出一种基于内部注意力分布的对象级诊断方法——Intrinsic Grounding Signature（IGS），并在此基础上设计了Verifier‑Guided Decoding（VGD）框架，在高风险对象出现时局部回滚并重新生成，以降低幻觉而不影响文本覆盖率和长度。

**💡 创新点**

创新点包括：①提出IGS，利用跨头层有符号的图像与sink注意力分布，在生成过程中实时捕捉对象是否缺乏视觉支撑；②基于IGS构造轻量级的L1正则化逻辑回归判别器，实现在对象级别的风险评估；③在VGD中引入KV缓存回滚、同义词抑制和局部贪婪重写，保证对模型参数不做修改、对已生成前缀不变，从而实现高效的选择性幻觉抑制。

**🔧 技术方法**

技术实现主要包括：冻结LVLM内部注意力的提取、构造图像和sink注意力特征；使用L1正则化的逻辑回归做风险判别；在解码时进行KV缓存截断、同义词屏蔽以及局部重写；对不同模型和数据集进行跨架构、跨数据集的评估。

**📊 数据集**

使用的数据集包括：CHAIR-MSCOCO、AMBER-G、NoCaps（用于验证），以及Open Images（用于交叉验证），并在多种模型架构（LLaVA、Qwen2‑VL、Shikra等）上进行评估。

**📈 对比分析**

与多种对照方法（如训练时对齐、推理时干预、后期修正等）进行对比，VGD在@rec90阈值下将CHAIR降低约43%~50%，保持覆盖率≈99%，长度不受影响；在跨数据集迁移实验中，无需重新训练验证器即可显著降低验证错误率，证明了IGS和VGD的鲁棒性与可迁移性。

**⚠️ 局限性**

局限性包括：①对内部注意力分布的依赖限制了对非冻结或结构不同模型的适用性；②回滚与重写仍可能误删真实对象或产生新错误，需根据阈值进行手工调优；③对长句子、多词实体的同义词处理仍不够完善，未来需进一步提升局部重写的准确性。

---

## 277. Thinking Once Is Enough: Intermediate-Layer Evidence Routing for High-Resolution VQA

**arXiv ID:** 2607.27830 | [PDF](https://arxiv.org/pdf/2607.27830v1)

**作者:** Zhongkuan Mao `[一作]` (Sichuan University), Keren Fu `[通讯]` (Sichuan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在单一视觉通道中提出“Thinking-Once”证据路由策略，利用中间层已编码的视觉信息并按问题条件动态挑选核心实体和背景支持，直接传递至后续层；

**💡 创新点**

创新点在于把高分辨率VQA的核心问题从“再次获取视觉证据”转变为“高效重路由已存在的细粒度证据”，实现无额外视觉编码、训练无依赖的高效推理；

**🔧 技术方法**

采用问答引导的视觉注意分布重构、最小覆盖选择、网格背景池化和序列合并等技术，实现单通道、单向前向的证据传递；

**📊 数据集**

在V*Bench、HRBench‑4K和HRBench‑8K三个高分辨率VQA基准上进行评测；

**📈 对比分析**

与外部证据重采样、压缩型词元缩减等现有方法对比，平均提升+3.1点（V*Bench）/+3.0点（HRBench‑4K）/+2.7点（HRBench‑8K），单模型上Qwen2.5‑VL‑7B平均分从72.5升至79.1，显著降低约4 GB峰值内存；

**⚠️ 局限性**

局限在于若视觉编码未能捕获所需细粒度信息，路由无法恢复；对非常大或多样化场景仍需补充外部视觉搜索或重新编码。

---

## 278. CrowdioSet and PaRIRset: Two Datasets Towards Live Music Source Separation

**arXiv ID:** 2607.27828 | [PDF](https://arxiv.org/pdf/2607.27828v1)

**作者:** Enric Gusó `[一作]` (Universitat Pompeu Fabra), Xavier Serra `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `67630363-6be0-4f51-ab05-7198250671a5` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 CrowdioSet（观众噪声与合成齐唱数据集）和 PaRIRset（40 个专业音乐会场的 PA 阵列冲激响应），并在 SCNet 上训练与评估音乐源分离模型在现场录音中的泛化性能。

**💡 创新点**

①构建覆盖现场观众噪声与齐唱的 Crowd­ioSet；②采集专业音乐会场的 PA 阵列冲激响应 PaRIRset；③将两者联合用于训练，显著提升现场音乐源分离效果。

**🔧 技术方法**

使用 SCNet 作为基线网络，结合零射击声乐转换 HQ‑SVC、AVOX Choir 插件生成合成齐唱，利用多麦克风阵列测量与后处理获得 PaRIRset，并采用混合噪声与混响数据增强技术。

**📊 数据集**

MUSDB18HQ 与 MOISESDB 为干净音乐源；CrowdioSet 包含 4800 条 Freesound 观众音轨和合成齐唱；PaRIRset 包括 40 场馆测得的 PA 阵列冲激响应（约 2200 条立体 RIR）。

**📈 对比分析**

通过在 MUSDB18HQ 测试集上分别训练 clean、rev、noisy、noisyrev 四种配置，使用 SDR 评估和 AB 听力测试；结果显示加入 CrowdioSet 能显著提升现场录音中的声乐与观众分离，PaRIRset 使混响泛化显著提升，噪声模型在无现场噪声时几乎无损，且在 AB 测试中被显著偏好。

**⚠️ 局限性**

生成的齐唱与原声高度相关，导致难以完全分离；混合参数（概率、增益）未进行系统优化；噪声+混响训练收敛较早，模型与超参数需进一步调优。

---

## 279. Localization and Pursuit of a Mobile Target using Distance-only Measurements

**arXiv ID:** 2607.27812 | [PDF](https://arxiv.org/pdf/2607.27812v1)

**作者:** Nabarupa Das `[一作]` (National Institute of Technology Durgapur), Suvadip Batabyal `[通讯]` (National Institute of Technology Durgapur)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `aaccfe5c-6b26-4208-b23c-35331481e142` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在二维平面上，用单个固定接收器和移动探测器仅通过距离测量实现对线性移动目标的搜索与跟踪。

**💡 创新点**

创新之处在于仅利用距离（路径损耗）信息，无需 GPS、AoA 或多锚点即可确定目标象限、位置和运动向量，并在最大 13 步内实现定位。

**🔧 技术方法**

采用离散时间模型、法向量余弦定理求角、路径损耗距离估计、搜索-运动向量-追踪三阶段算法。

**📊 数据集**

使用自定义离散时间仿真数据，包含七种目标运动场景（初始位置与速度）。

**📈 对比分析**

通过仿真验证，跟踪误差在搜索完成后保持恒定，最大 13 步即可收敛；未与现有方法对比，但表现出可行性和稳定性。

**⚠️ 局限性**

局限在于假设距离无噪声、目标速度恒定、目标在搜索过程中不跨象限；对路径损耗噪声、目标快速变轨、3D 或多代理场景不适用。

---

## 280. FDDWAN: A Frequency-Decoupled Diffusion Network for Watermarking Attack

**arXiv ID:** 2607.27800 | [PDF](https://arxiv.org/pdf/2607.27800v1)

**作者:** Chunpeng Wang `[一作]` (Qilu University of Technology (Shandong Academy of Sciences)), Qi Li `[通讯]` (Qilu University of Technology (Shandong Academy of Sciences))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在可见水印去除任务中提出了 FDDWAN 框架，先在小波域分离低高频子带并进行预攻击，再用残差扩散精细去除水印。

**💡 创新点**

创新点在于将频率解耦与残差扩散相结合，既在低频段定向破坏水印，又在高频残差上进行扩散细化，显著提升去水印效果与视觉质量的平衡。

**🔧 技术方法**

采用离散小波变换、扩张卷积块、注意力机制以及条件扩散模型（DDPM）实现频域攻击与残差扩散。

**📊 数据集**

在 CelebA 与 ImageNet 两大数据集上进行实验，并对 DCT、PHFMs、HiDDeN、StegaStamp 四种水印方案进行评估。

**📈 对比分析**

与传统噪声/压缩攻击及学习型攻击（FAADW、HIWANet、Diffusion Attack、UnMarker）对比，FDDWAN 在 PSNR、SSIM 和 BER 指标上均显著优于前者，尤其在保持高 PSNR 的同时使 BER 接近 0.5。

**⚠️ 局限性**

主要限制是需要两套扩散模型并进行多步采样，导致推理时间长、计算成本高，限制了实时和资源受限环境下的应用。

---

## 281. RedFlow: Redirect Failure into Action-Level Corrections for Flow-matching VLA Policy

**arXiv ID:** 2607.27782 | [PDF](https://arxiv.org/pdf/2607.27782v1)

**作者:** Zhengyang Yan `[一作]` (Hong Kong University of Science and Technology), Song Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 RedFlow，一种细粒度离线强化学习框架，将失败轨迹转换为动作级纠正信号，用于提升流匹配 VLA 策略

**💡 创新点**

细粒度上下文感知纠正匹配与自适应重定向目标，能够从失败经验中提取局部动作级监督并精确引导策略更新

**🔧 技术方法**

采用流匹配（flow‑matching）VLA、General Reward Model 进度估计、HDBSCAN 聚类、上下文感知纠正匹配、适应性重定向目标及离线强化学习损失

**📊 数据集**

LIBERO 基准（Spatial、Object、Goal、Long）以及三项真实机器人任务（衣物折叠、扫地、桌面清理）

**📈 对比分析**

与 AWR、DPO 等离线 RL 基线对比，LIBERO 上平均提升 12% 成功率；与 PPO、GRPO、DDPO 等在线 RL 对比，样本效率高约 10 倍，实际机器人成功率从 56.7% 提升至 74.7%

**⚠️ 局限性**

仍需大量离线数据，难以处理分布外状态的纠正目标，参数设置需手工调优，且仅验证在流匹配 VLA 上的效果，未证明对其他模型的泛化

---

## 282. Adaptive Security at the Edge for 6G-Enabled Healthcare IoT

**arXiv ID:** 2607.27858 | [PDF](https://arxiv.org/pdf/2607.27858v1)

**作者:** Ijaz Ahmad `[一作]` (University of Oulu), Erkki Harjula `[通讯]` (University of Oulu)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一套基于 eBPF 的 kernel‑plane 自适应速率控制器，用于保护医疗 IoT 边缘网关在突发流量下的报警延迟。

**💡 创新点**

创新点在于采用闭环监控‑决策‑执行结构，结合 hysteresis 的多层速率限速和可审计的日志，能在 kernel 空间快速且稳定地响应突发流量，优于传统用户空间防火墙。

**🔧 技术方法**

使用 eBPF、TC（Traffic Control）实现 kernel‑plane 速率限制；MQTT QoS1、Mosquitto 代理；Raspberry Pi 与 ESP32 嵌入式设备；结构化 JSONL 日志。

**📊 数据集**

实验数据来自在 Raspberry Pi 网关上搭建的 Mosquitto broker 与两台 ESP32 端点生成的合成心率/报警数据，包含故意的 ON/OFF 突发发布负载。

**📈 对比分析**

通过对比 No_Enf（无控制）、U_NFT（用户空间防火墙基线）和 K_Adapt（kernel‑plane 自适应）三种方案，测量报警 p99 RTT 与突发流量泄漏量；K_Adapt 在 p99 RTT 上比用户空间基线提升 13.3%，并将突发流量泄漏量减少 46%。

**⚠️ 局限性**

实验仅覆盖两端点与单网关的脉冲突发工作负载，未测试多源争用、隐蔽滥用模式以及跨层级扩展；未来需要在更大规模、多源场景下验证，并完善 on‑demand 策略部署。

---

## 283. AutoPref: Automatic Discovery of Task-Specific Preference Objectives for Neural Combinatorial Optimization

**arXiv ID:** 2607.27953 | [PDF](https://arxiv.org/pdf/2607.27953v1)

**作者:** Shengda Gu `[一作]` (Chinese Academy of Sciences), Jian Cheng `[通讯]` (Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出AutoPref框架，利用大语言模型自动搜索并设计神经组合优化（NCO）中的偏好训练目标，将其拆分为 pairwise loss 与 set-aware weighting 两个可编程模块；

**💡 创新点**

创新点在于把偏好目标抽象为可搜索的程序空间，采用 LLM 生成候选、行为门控、分阶段条件搜索以及结构多样性选择，使得能够为不同组合优化问题自动发现任务特定、性能最优的偏好目标；

**🔧 技术方法**

主要技术包括：LLM 代码生成与验证、行为一致性检测（可执行性、优先级一致、尺度不变性、权重多样性）、分阶段（先优化 pairwise loss 再搜索权重）、短期训练评估、基于基准目标的对比评分以及多样性过滤；

**📊 数据集**

实验数据集覆盖四类组合优化任务：TSP、CVRP、FFSP、JSSP，分别在发现规模（TSP100、CVRP100、FFSP100、JSSP15×15）以及两个未见规模（如TSP50/TSP1000、CVRP50/CVRP1000 等）进行评估；

**📈 对比分析**

与最优/近似解算器、标准 NCO 解算器以及手工设计的偏好目标（PO4COPs、BOPO、SLIM）按平均成本、最优性缺口和评估时间三指标对比；AutoPref（尤其 APW）在所有 12 个设置中均优于现有偏好方法，且在更大规模实例上表现提升更明显；

**⚠️ 局限性**

局限性包括：分阶段搜索可能无法达到全局最优，搜索空间仅限 pairwise loss 与权重；LLM 生成程序时可能出现无效或次优方案，需要大量算力；对极大规模实例的泛化尚需进一步验证。

---

## 284. Safeguards Based on Copyable Context Cannot Provide Reliable Safety for LLMs

**arXiv ID:** 2607.27951 | [PDF](https://arxiv.org/pdf/2607.27951v1)

**作者:** Pingyu Wu `[一作]` (University of Science and Technology of China), Nenghai Yu `[通讯]` (University of Science and Technology of China)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统地分析了大语言模型在双重用途任务中的安全防护机制，推导了在可复制证据下攻击者能够获得的最小帮助阈值Γ(q)，并证明了安全、效用与开放访问无法同时满足的安全三元悖论；通过引入可信凭证概念，阐明了如何利用非可复制信息突破此阈值；

**💡 创新点**

创新点在于：①将能力分配与证据质量分离，给出精确的攻击者帮助下限；②证明了复制证据导致的“安全三元悖论”；③提出可信凭证作为消除下限的必要条件；④给出一系列可扩展的数学定理与经验验证框架；

**🔧 技术方法**

采用了概率论与信息论工具（Markov kernel、总变差距离、凸优化、Blackwell 部分顺序）进行正式建模，并利用总变差上界刻画复制误差；同时结合对话式安全评估与对抗攻击实验进行验证；

**📊 数据集**

实验基准主要来源于公开的安全评估数据集（如 OpenSafeIntent、XSTest、OR‑Bench、CarryOnBench、Internal Safety Collapse 等）以及内部安全测试数据；

**📈 对比分析**

方法上通过理论证明与实验对比，展示在复制证据条件下无论采用何种过滤或请求检查，攻击者帮助下限始终为 Γ(q)；在可信凭证场景下，通过衡量预测准确度与复制误差，证明可将帮助下限逼近零；实验结果表明现有安全机制在保持合法效用的同时，攻击者帮助下限无法低于理论阈值；

**⚠️ 局限性**

局限性包括：①假设了有限的操作分辨率与固定的攻击者类；②需要针对每种部署给出复制误差 δ_κ 的经验估计；③仅关注推理时的访问控制，不涉及模型权重的获取与再部署；④可信凭证的实现需依赖硬件或可验证的安全上下文，实际部署成本和可行性尚未完全评估；

---

## 285. Interpretable Representation via LLM-Driven Generative Disentanglement for Local-Life Service Recommendation

**arXiv ID:** 2607.27944 | [PDF](https://arxiv.org/pdf/2607.27944v1)

**作者:** Long Zhang `[一作]` (Kuaishou Technology), Kun Gai `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在离散化前先对大型语言模型生成的隐藏状态进行属性解耦的 SID 生成框架 LGRID，用于本地生活服务推荐。

**💡 创新点**

创新点在于：①通过 Structured Disentangled Block（SD‑Block）将地理、品牌等属性在 LLM 隐状态中映射到可解码的槽位；②使用 Synergistic Alignment Learning（PGD + SACL）保证槽位可解释、判别力强；③采用 Dual‑Stream Residual Quantization（DSRQ）分别对地理槽和语义槽离散化，显著降低 SID 冲突。

**🔧 技术方法**

核心技术包括：大型语言模型 Qwen3‑8B、SD‑Block（Semantic Anchor Injection、Anchor‑Guided Perception、Structured Causal Routing）、PGD 生成式解耦、SACL 结构化对比学习、LDR 多样性正则、DSRQ 余量量化。

**📊 数据集**

使用两大数据集：Kuaishou 真实工业数据集（含地理和文本信息）和公开 Foursquare 数据集。

**📈 对比分析**

与多种推荐骨干（DIN、DIEN、ETA 等）和 SID 基线（RQ‑VAE、Res‑KMeans、LGSID、LGSID++）进行对比；在 Kuaishou 上 AUC 提升至 5.44%，在 Foursquare 上多项指标提升 0.25%‑2.54%；属性解码准确率超过 99%，SID 冲突率降至 39.9%（比 LGSID 的 97% 高效得多）。

**⚠️ 局限性**

局限性包括：需要离线生成 SID 的额外时间；对细粒度地理属性的学习对训练样本量敏感；模型规模和 LLM 依赖性高，部署成本相对较大。

---

## 286. Scaling LLM-Driven Multi-Agent Systems: Design Principles and Architectural Scalability Analysis

**arXiv ID:** 2607.27942 | [PDF](https://arxiv.org/pdf/2607.27942v1)

**作者:** Linus Sander `[一作]` (Technical University of Munich), Alois Knoll `[通讯]` (Technical University of Munich)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了LLM驱动的多智能体系统（MAS）的架构扩展如何影响其性能、成本和可靠性，设计并验证了四个递增复杂度的参考架构，并在终端任务基准上对两种LLM进行实验。

**💡 创新点**

提炼出四条可扩展MAS的设计原则（简洁性、弹性反馈、顺序工作流+可选循环、基于摘要的通信），并将其在单一顶层架构中操作化，系统评估了规模化对准确率、成本和一致性的影响。

**🔧 技术方法**

采用LLM工具交互（Shell、Terminal、Python、Search）、多代理协同、基于自然语言摘要的群聊通信以及受约束的有向工作流图，结合GPT‑5系列模型执行任务。

**📊 数据集**

使用终端基准（terminal‑bench），包含80+终端系统工程任务，评估需要多轮推理、探索和工具执行。

**📈 对比分析**

通过对单体代理、MAS‑S、MAS‑M、MAS‑L四种拓扑以及GPT‑5‑nano、GPT‑5‑mini和GPT‑5.3‑Codex三种模型进行多轮实验，衡量准确率、测试加权准确率、成本（LLM调用、token、时间）和一致性；结果显示在足够强大的LLM下，MAS‑M达到最佳准确率并保持可接受的成本，性能随规模呈线性增长，但一致性低于50%。

**⚠️ 局限性**

主要限制在于低能力LLM下无法从架构扩展获益；在最高复杂度架构下出现超时和成本激增；且跨运行一致性不足，无法保证单次任务的可靠完成。

---

## 287. The Geometric Nature and a Free Proxy for Flow-Matching Uncertainty

**arXiv ID:** 2607.27933 | [PDF](https://arxiv.org/pdf/2607.27933v1)

**作者:** Ziyang Rao `[一作]` (HKUST(GZ)), Hui Xiong `[通讯]` (HKUST(GZ))

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了流匹配（Flow Matching）模型的不确定性几何特征，并提出了只需单次前向传播即可估计的不确定性代理——加速（accel），用于在线失败检测。

**💡 创新点**

创新点在于将流场偏离理想等距收缩模板的几何度量直接关联到后验不确定性，并由此导出无额外计算成本的加速指标，可实时捕捉FM模型的不确定性。

**🔧 技术方法**

使用了流匹配理论、偏导数关系、离散加速度估计、基于CUMSUM的阈值校准等技术，同时对多种FM模型进行对比实验。

**📊 数据集**

在D3IL、LIBERO、Robocasa等机器人学习基准上进行评估，并与π_0.5、GR00T、VLA-JEPA、SmolVLA等不同架构的FM模型进行对照。

**📈 对比分析**

与重采样、训练驱动和随机网络蒸馏等失败检测基线比较，accel在TPR/FPR等指标上与最佳基线（如SAFE）相当，且在多模型、多场景下表现更稳定，误报率低且提前预警时间较长。

**⚠️ 局限性**

限制在于对欠拟合模型敏感，难以检测自信但错误的行为，且只捕捉路径曲率信息，未能覆盖所有复杂失败模式。

---

## 288. ARD-REFSM: Enhancing Reflection Symmetry Detection with Asymmetric Denoising and Rotation Equivariance

**arXiv ID:** 2607.27927 | [PDF](https://arxiv.org/pdf/2607.27927v1)

**作者:** Dongfu Yin `[一作]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy), Fei Yu `[通讯]` (Carleton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 ARDNet，结合异向区域去噪与旋转等变特征相似度匹配，提升反射对称性检测。

**💡 创新点**

创新点是将异向区域去噪与旋转等变特征匹配联合使用，并构建新的 GMSYM 数据集评估鲁棒性。

**🔧 技术方法**

采用组等变编码解码器、ASPP、旋转损失、焦点损失、D8 群卷积等技术。

**📊 数据集**

使用 DENDI、NYU、LDRS、SDRW 四个公开数据集以及自建 GMSYM。

**📈 对比分析**

与 EquiSym、PMCNet 等方法比较，F1 最高 65.52%，在不同旋转角度、角度精度和中心偏移上表现优于现有方法。

**⚠️ 局限性**

局限在于对极端遮挡、噪声以及三维视角变化的适应仍有限，且对旋转角度采样敏感。

---

## 289. S-CEReBrO: Breaking the Memory Barrier in Continuous EEG Monitoring

**arXiv ID:** 2607.27913 | [PDF](https://arxiv.org/pdf/2607.27913v1)

**作者:** Glenn Anta Bucagu `[一作]` (ETH Zurich), Luca Benini `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

设计并评估了S-CEReBrO，一个支持连续EEG监测的Transformer基础模型，采用窗口化交替注意力实现常数KV缓存内存。

**💡 创新点**

提出窗口化交替注意力机制，将时空注意力限制在局部窗口，保证内存不随信号长度增长，同时保持线性时间复杂度，实现可持续长时EEG流式处理。

**🔧 技术方法**

使用Transformer窗口化交替注意力、掩码自编码预训练、可学习的时空嵌入、层级窗口dilation与shift等技术。

**📊 数据集**

在超过25,000小时、12,000+受试者的EEG数据上预训练，包括TUEG、SEED、BOAS、SleepEDFx、BCI-NER、GWD等；下游评测11个公开任务。

**📈 对比分析**

与多种基准模型（EEGNet、CEReBrO、BIOT、LaBraM等）对比，S-CEReBrO在7/11任务上取得SOTA，参数量仅2.4M，内存使用55%低，推理吞吐量提高2.1倍。

**⚠️ 局限性**

仍受限于对极小样本的适配性，缺乏小时级长序列基准，且窗口化注意力可能对极远程依赖捕捉不足。

---

## 290. One Patch Is Enough: Reinforcement-Optimized Visual Token Grounding for MLLM-Based Scene Text Spotting

**arXiv ID:** 2607.27902 | [PDF](https://arxiv.org/pdf/2607.27902v1)

**作者:** Rui Tang `[一作]` (South China University of Technology), Lianwen Jin `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出单图块文本检测框架 SPaTS，利用单一视觉 token 进行文本定位，并通过全图精细几何回归完成对齐；

**💡 创新点**

创新点包括：① 单图块视觉 token 选取与全图几何回归的组合；② 基于 GRPO 的强化学习框架 SPaSO，用离散奖励直接优化 token 选取；③ 方向嵌入对齐 DEA 通过分离向量幅值与方向抑制噪声；④ Patch‑Enhanced Decoding PED 将选取的局部视觉信息与语言上下文融合，恢复精细边界；

**🔧 技术方法**

采用多模态大型语言模型（如 Qwen‑VL）、视觉 token 编码、强化学习（GRPO）以及交叉注意解码、方向嵌入对齐等技术；

**📊 数据集**

训练使用 Curved Synthetic 150k（预训练）及多源真实数据 MLT‑2017、ICDAR 2013/2015、Total‑Text 与 SCUT‑CTW1500；评测在 Total‑Text、CTW1500 与 ICDAR‑2015 三大基准上进行；

**📈 对比分析**

与闭源/开源 OCR 及 MLLM 系统对比，SPaTS 在 Total‑Text、CTW1500 与 ICDAR‑2015 上均实现显著提升：SPaTS‑2B 在 Total‑Text 上 F‑measure 81.7%，CTW1500 上 87.9%，ICDAR‑2015 上 69.1%；在 E2E 识别中也大幅超越现有 OCR 模型；

**⚠️ 局限性**

局限性：RL 训练阶段耗时且对奖励设计敏感；DEA 采用固定幅值后可能对极端光照或显著尺度变化的鲁棒性不足；整体模型仍需进一步降低训练成本并提升极端场景的稳定性。

---

## 291. Static In, Dynamic Out: Counterfactual Action Augmentation for Moving Object Manipulation

**arXiv ID:** 2607.27890 | [PDF](https://arxiv.org/pdf/2607.27890v1)

**作者:** Woo Chul Shin `[一作]` (Georgia Institute of Technology), Danfei Xu `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在训练阶段仅使用静态演示，借助对抗性动作增广将对象位移到未来位置并保持手-对象相对姿态，从而得到能够在动态对象上执行的目标条件策略。

**💡 创新点**

创新点在于将移动物体操作拆解为“预测未来位置”和“实现该位置”两子任务，并通过对抗性增广在训练时注入运动先验，使得“静态输入、动态输出”成为可能。

**🔧 技术方法**

采用行为克隆、目标条件策略、对抗性动作增广（H-SIDO 与 DynaSIDO）、可交换姿态预测器以及 MPPI 动态校正等技术。

**📊 数据集**

使用三种仿真任务（Mug、Square、Stack）的 MimicGen 演示数据，以及两项真实世界任务（Gantry、Peachtree）的静态演示（分别为 30、22、64、25 条）。

**📈 对比分析**

与无运动输入、仅使用当前姿态、运行时补偿等基线相比，SIDO 在动态任务的成功率提升超过 30% 并保持或提升了静态任务的性能。

**⚠️ 局限性**

限制：依赖外部 6-DoF 估计器且不考虑路径碰撞；在高速度或复杂场景下需要更精确的运动预测器。

---

## 292. Not All Tokens Deserve Equal Credit: Counterfactual Sensitivity Credit Reallocation for Long-CoT Reasoning

**arXiv ID:** 2607.27888 | [PDF](https://arxiv.org/pdf/2607.27888v1)

**作者:** Qiangqiang He `[一作]` (Nanjing University), ZiJian Wang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

无法获取论文内容

**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 293. Feminist voices, partisan networks: gender and political communication by MEPs on social media

**arXiv ID:** 2607.27931 | [PDF](https://arxiv.org/pdf/2607.27931v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 294. ODEWorld: A Continuous Predictive Architecture via Physical-Time Flow

**arXiv ID:** 2607.27924 | [PDF](https://arxiv.org/pdf/2607.27924v1)

**作者:** Dongxiu Liu `[一作]` (Tsinghua University), Xianyuan Zhan `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a8e75ba4-7a2d-4153-b003-06c94533add0` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于物理时间的连续潜在世界模型ODEWorld，利用PT-Flow框架学习潜在速度场并通过ODE求解实现任意时间步长的视频生成和机器人策略规划。

**💡 创新点**

创新点在于：①将时间离散化的传统预测模型转化为连续物理时间ODE；②通过动力学表示解耦与直接一阶监督，避免表示坍塌；③在预训练DINO特征空间中构建极小潜在空间，显著提高效率并保持高质量重建。

**🔧 技术方法**

核心技术包括：物理时间流PT-Flow、Jacobians向量乘积监督、跨注意力编码器/解码器、FiLM调制的轻量MLP速度网络、Savitzky–Golay滤波器、跑步阶梯RK4等ODE求解器。

**📊 数据集**

使用的主要数据集为LIBERO（130任务、6.5k人机演示）与Agibot‑World（约30k轨迹、真实机器人）。

**📈 对比分析**

与LDP、V‑JEPA 2等潜在世界模型以及SuSIE、Seer、VPP等规划/策略基线对比，ODEWorld在视频预测上PSNR提升≈4‑5点、LPIPS下降≈0.04点，推理延迟大幅降低（0.07s/64帧）；在政策学习上，平均成功率提升至83%（仿真）或80%（真实机器人）。

**⚠️ 局限性**

局限性包括：依赖高质量预训练特征导致对新视觉域迁移敏感；对速度场的近似估计仍受限于采样间隔；在极长时序或高度非线性动态下，ODE求解可能出现数值不稳定或慢收敛。

---

## 295. LAST: The Last Query Token Guides Visual Token Pruning for Edge-Cloud Collaborative MLLM Inference

**arXiv ID:** 2607.27952 | [PDF](https://arxiv.org/pdf/2607.27952v1)

**作者:** Feng Yang `[一作]` (City University of Hong Kong), Chris Xing Tian `[通讯]` (Peng Cheng Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在边缘端利用轻量级视觉语言模型的最后一个查询标记的注意力，预先对图像视觉标记进行查询感知的剪枝，随后仅传输保留的视觉标记给云端进行最终推理。

**💡 创新点**

创新点在于：①只需单次无生成的预填充推理，利用最后一个查询标记的注意力即可得到查询条件下的视觉重要性；②结合重要性加权的 k‑center 选择策略，兼顾查询相关性与视觉多样性；③完全训练无关，边缘端无需访问云模型内部状态，显著降低了边缘侧推理与网络传输开销。

**🔧 技术方法**

技术手段包括：视觉标记编码、跨模态注意力聚合、基于注意力的查询感知评分、重要性加权 k‑center 选择、边缘–云端统一的视觉标记接口；模型采用 InternVL2/InternLM2.5、LLaVA 等多模态基础模型。

**📊 数据集**

使用了 11 个多模态基准数据集：ChartQA、DocVQA、TextVQA、VQAv2、GQA、VizWiz、ScienceQA‑IMG、POPE、MME、MMBench‑CN、MM‑Vet。

**📈 对比分析**

与随机选取、VisionZip、VisPruner（无查询感知）以及 FastV、SparseVLM、SGP（查询感知）对比，LAST 在 62.5%、37.5% 与 12.5% 的视觉标记保留率下，平均相对性能分别保持 99.3%、99.0% 与 95.4% 的完整标记基准；在所有基准上实现了最高的平均精度与效率折中，并且边缘侧推理延迟最低。

**⚠️ 局限性**

局限性包括：①依赖于共享的视觉编码器和可配置的轻量级代理模型，可能对不同体系结构适配有限；②仅利用最后一个查询标记的注意力，可能忽略多步对话或复杂查询中的上下文信息；③在极低的标记保留率下，仍存在精度下降；④对边缘设备的算力与内存仍有一定要求。

---

## 296. Shapes from Examples: Foundations of Shape Learning in Recursive SHACL

**arXiv ID:** 2607.27934 | [PDF](https://arxiv.org/pdf/2607.27934v1)

**作者:** Bente Gortworst `[一作]` (TU Wien), Anni-Yasmin Turhan `[通讯]` (Paderborn University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在SHACL片段（相当于描述逻辑^*）中，从正负样本学习形状表达式（fitting）和最特定拟合（MSF）的理论问题，给出了存在性判定、MSF计算及其复杂度上界。

**💡 创新点**

创新点包括：①首次将fitting问题迁移到递归SHACL并证明其ExpTime完备性；②引入预匹配、树自动机与星积图结合的技术，实现MSF存在性判定与构造；③在WFS下若正样本数有限，则可在多项式时间内计算MSF；④将三种SHACL语义（WFS、STS、SUS）统一到同一框架。

**🔧 技术方法**

主要技术：两向交替奇偶树自动机（2ATA）构造预匹配判定；星积图与r^*路径保持的“星积”与“冗余星删除”操作；模拟（simulation）与无展开（unravelling）理论；复杂度分析与指数时间空实现。

**📊 数据集**

本文为理论分析，没有使用特定实验数据集；所有结果均基于图结构与形状表达式的抽象建模与证明。

**📈 对比分析**

与既有工作比较：与描述逻辑、CQ学习中的fitting问题相比，本文扩展到递归SHACL并给出明确的ExpTime上界；与近似或启发式学习方法不同，本文提供了严格的可行性与最优性保证。性能上，存在性判定与MSF构造可在指数时间完成；在正样本有限时可降到多项式。

**⚠️ 局限性**

局限性：仅考虑SHACL的^*片段（不含并、名词、计数等），对更完整的SHACL功能（如路径计数、路径等价、闭包等）不适用；在STS与SUS下若正样本数量受限，仍缺乏多项式时间解；MSF可能不存在，且构造的形状表达式可能指数大，实际可解释性有限。

---

## 297. Harnessing the Potential of Optimizing Data Mixtures via Bayesian Domain Reweighting

**arXiv ID:** 2607.27928 | [PDF](https://arxiv.org/pdf/2607.27928v1)

**作者:** Xiang Yuan `[一作]` (Xi'an Jiaotong University), Zongben Xu `[通讯]` (Xi'an Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于贝叶斯框架的域权重学习方法 ByDoRe，利用 Gamma-Dirichlet 随机层在一次通行的 LLM 预训练中稳定地推断域权重。

**💡 创新点**

创新点：①将域权重建模为 Dirichlet 分布并通过 Gamma 先验动态适配，①消除了直接优化时的波动和高计算成本；②采用先验预测网络实时更新 Gamma 超参数，实现对批量噪声的抑制；③提供了理论收敛保证并在大规模预训练中显著降低 FLOPs。

**🔧 技术方法**

技术：贝叶斯推断、Gamma‑Dirichlet 随机层、隐式重参数化梯度、中心有限差分近似、Meta‑学习与 ELBO 最优化。

**📊 数据集**

数据集：使用 The Pile（17 个公开域）作为代理模型训练集；验证集取自 Pile‑CC；目标模型在 150B tokens 上训练，用 13 个下游基准进行零样本评估。

**📈 对比分析**

对比方法：Human、DoReMi、RegMix、AutoScale、MDE。ByDoRe 在一般用途场景下平均得分 48.50%（略优 RegMix 48.22%），在专项目标场景下平均得分 38.35%（高于人类 37.72%），并且搜索阶段 FLOPs 仅 2.40×10¹⁶，低于 RegMix 的 3.07×10¹⁸（约 127 倍节省）且准确率提升 0.28%/3.03%。

**⚠️ 局限性**

局限：仍依赖代理模型的预训练效果；对极端异构域组合的泛化尚需进一步验证；实现细节（如超参数 ϵ、学习率）对最终性能影响显著，需经验调优。

---

## 298. The Case for Vibe Modeling: A Missing Step in AI-Based Trustworthy Software Development

**arXiv ID:** 2607.27923 | [PDF](https://arxiv.org/pdf/2607.27923v1)

**作者:** Shalini Chakraborty `[一作]` (University of Bayreuth), Judith Michael `[通讯]` (University of Regensburg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过设计四种基于大型语言模型（LLM）的软件开发场景，调查学生对LLM生成代码以及中间模型（vibe modeling）在理解、验证和信任方面的感知，并对其对信任的影响进行定性与定量分析。

**💡 创新点**

创新点在于提出并验证一种轻量级的“vibe modeling”中间抽象，旨在将自然语言需求与可执行代码之间的隐含决策显式化，从而提升代码可理解性、降低验证成本并增强对AI生成软件的信任。

**🔧 技术方法**

所用技术包括问卷设计、案例情境呈现、Likert量表评估、描述性统计分析以及基于编码的质性分析，此外还结合了信任与不信任因素的归类和可视化。

**📊 数据集**

数据集来自两所高校共17名学生的问卷回复，包含四个开发情境的反馈（代码优先、代码到模型重构、模型更新后代码生成、模型先行生成）。

**📈 对比分析**

比较方法主要是对各情境下学生对信任因素的选择频次进行计数和百分比分析；结果显示，学生更倾向于在模型优先或模型更新前使用模型，从而提升对LLM输出的信任。该研究并未提供传统性能指标，但通过用户感知指标展示了vibe modeling的潜在价值。

**⚠️ 局限性**

限制包括样本量小且仅限学生、情境为假设性描述、未使用实际vibe modeling工具、缺乏专业开发者验证，以及对模型质量与验证机制的具体实现未深入探讨。

---

## 299. Exact Action Values Are Not Enough: Rollout-Verified Reinforcement Fine-Tuning of a Reasoning Model for Multi-Zone VAV Control

**arXiv ID:** 2607.27914 | [PDF](https://arxiv.org/pdf/2607.27914v1)

**作者:** Takumi Shioda `[一作]` (University of Tokyo), Tatsuo Nagai `[通讯]` (Tokyo University of Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在四区物理仿真环境中评估大型语言模型（LLM）与强化学习（TD3）控制多变量HVAC的可行性，并尝试通过TD3引导的rollout‑verified RFT将控制知识迁移到本地可部署的开源LLM。

**💡 创新点**

首次在连续多目标HVAC控制中使用推理型LLM进行无特定建筑训练的直接控制，并提出直接rollout验证的RFT框架来评估其可迁移性，发现仅凭动作价值监督不足以提升连续动作的采样质量。

**🔧 技术方法**

采用文本指令化接口的LLM（GPT‑5与20.9B开源模型+LoRA）、Twin‑delayed Deep Deterministic Policy Gradient（TD3）作为教师与验证器、Dr. GRPO算法进行rollout‑verified RFT，以及物理基础的VAV仿真器和自定义奖励函数。

**📊 数据集**

使用东京气象观测（JMA 10分钟数据）与三天夏季占用时刻表构建四区VAV仿真环境；TD3训练使用一年的历史天气，评估阶段采用同一三天夏季天气和占用数据。

**📈 对比分析**

对比五种控制器（Guideline 36规则、TD3、GPT‑5、RFT前后两阶段的开源模型），在三天运行中测量总电能、温度合规率与CO₂合规率。结果显示GPT‑5在总电能上相对基线下降6.2%，但CO₂合规率下降；RFT未能提升开源模型的能耗或合规性能，仍高于基线。

**⚠️ 局限性**

RFT仅基于动作价值监督，未提供动作–状态的转移知识，导致模型无法改进局部预测；实验仅在单一四区仿真器、有限训练步数和单一随机种子下进行，缺乏跨建筑及真实部署的验证。

---

## 300. A comparative analysis of automated techniques for security bug report identification

**arXiv ID:** 2607.27893 | [PDF](https://arxiv.org/pdf/2607.27893v1)

**作者:** Muhammad Laiq `[一作]` `[通讯]` (Blekinge Institute of Technology), Muhammad Laiq (Blekinge Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多种自动化技术进行系统评估，比较其在四个公开安全 bug 数据集上识别安全相关 bug 报告的效果。

**💡 创新点**

统一实验设置对比传统机器学习、BERT、GPT 以及最新 Few‑Shot 框架 SetFit，并探索跨项目迁移和 SMOTE 对性能的影响。

**🔧 技术方法**

使用 Logistic Regression、SVM、Random Forest、BERT‑base、RoBERTa、OpenAI GPT‑5.2（zero‑shot / few‑shot）和 SetFit；采用 TF‑IDF、句向量嵌入、prompt 设计、5‑折交叉验证等技术。

**📊 数据集**

四个公开基准数据集：Ambari、Camel、Derby、Wicket，均为手工标注的安全 vs 非安全 bug 报告。

**📈 对比分析**

通过 5‑折交叉验证和留一项目预测进行对比，评估指标为 F1、Recall、Precision、Accuracy。结果显示 SetFit 在 3/4 个数据集上获得最高平均 F1（0.622），传统 Logistic Regression 在 Ambari 上表现最好，GPT‑5.2 在 zero‑shot 和 few‑shot 下表现较差。

**⚠️ 局限性**

缺乏统计显著性检验；仅覆盖四个开源项目，可能不具备全面代表性；GPT 仅依赖提示，未做 fine‑tuning；SMOTE 效果不稳定；跨项目迁移对不同项目表现参差不齐。

---

## 301. Dynamic Spectral Filtering for Temporal Graph Learning: Learning Evolving Propagation Operators

**arXiv ID:** 2607.27891 | [PDF](https://arxiv.org/pdf/2607.27891v1)

**作者:** Yan Kong `[一作]` `[通讯]` (Nanjing University of Information Science and Technology), Yan Kong (Nanjing University of Information Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究时变图学习中传播算子本身随时间演化的机制，提出并实现 Dynamic Spectral Filtering（DSF）模型。

**💡 创新点**

创新点在于将 Chebyshev 多阶谱滤波系数作为可递归的时间状态，使用全局与阶级门控的层级演化结构，直接在传播算子上做时间适配，而非仅调整节点表示或历史记录。

**🔧 技术方法**

技术细节包括 Chebyshev 谱卷积、GRU 递归更新、门控机制（全局门与阶级门）、时序链接预测损失以及稀疏矩阵乘法与 GPU 并行计算。

**📊 数据集**

使用了 MOOC、Wikipedia 与 Reddit 三个公开时序链接预测数据集。

**📈 对比分析**

与 TGN、TGAT、DyGFormer 以及 DEFT 等基线模型进行对比，采用 AUC 和 AP 指标。DSF 在三数据集上分别取得 AP 0.7851、0.9088 与 0.9860，参数量仅 93k–133k，GPU 内存 68–182 MB，训练时间 1.6–2.1 s/epoch，显著低于对照组；相较于 DEFT，DSF 在某些数据集上性能相当或更优，同时参数、内存与训练时间均下降 8–33×、25–33× 与 5–19×。

**⚠️ 局限性**

主要局限包括：使用离散时间快照而非事件级处理；与事件流基线的采样和负样本策略不一致，导致跨家族比较不严格；实验仅采用单一随机种子；快照与事件流实现的 epoch 语义不同，影响训练时间对比。

---

## 302. DECODE: Tackling Representation and Decision Degradation in Continual AI-Generated Image Detection

**arXiv ID:** 2607.27882 | [PDF](https://arxiv.org/pdf/2607.27882v1)

**作者:** Zihao Cai `[一作]` (Fudan University), Jingjing Chen `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在连续学习环境下开发了一个名为 DECODE 的 AI 生成图像检测框架，解决了表示层衰退与决策边界漂移共同导致的遗忘问题。

**💡 创新点**

提出了 Dual Degradation 概念，并设计了两阶段分离解决方案：Subspace Diversity Regularization（SDR）用于保持多样化的法医特征，Closed‑Form Decision Alignment（CDA）用于在每个适配器合并后实时重校正共享分类头。

**🔧 技术方法**

利用 CLIP ViT‑L/14 预训练模型，插入 8 维 LoRA 适配器；SDR 通过正交性损失和能量平衡正则化；CDA 采用闭式岭回归并用留一交叉验证自动选取正则化系数；同时使用随机采样的样本记忆实现重放。

**📊 数据集**

在 19 种不同的生成器（包括扩散模型、GAN、面部改造、3DGS、自动回归模型等）上进行训练与测试，训练顺序为 8 个生成器，测试时还包含 11 个未见生成器。

**📈 对比分析**

与 10+ 传统与连续学习基线（如 NPR、Effort、SAIDO、DFIL、Tang 等）比较，DECODE 在连续学习任务中平均准确率达到 99.36% 仅 0.39% 遗忘，Open‑World 通用性在 11 个未见生成器上平均 95.36%（高于第二名 89.07%），且在 JPEG、缩放、模糊等扰动下表现出最强鲁棒性。

**⚠️ 局限性**

局限性包括：依赖预训练 CLIP 表示，可能对新型生成器特征提取不足；在极端数据分布跳变或大规模任务序列中，记忆容量与计算成本仍需进一步优化；目前未考虑无标签自监督或多任务融合的可能性。

---

## 303. Learning Social Robot Navigation By Sensing Human Legs

**arXiv ID:** 2607.27922 | [PDF](https://arxiv.org/pdf/2607.27922v1)

**作者:** Alberto Vaglio `[一作]` (University of Siena), Tommaso Van Der Meer `[通讯]` (University of Siena)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了CALF（Convolutional Attention for Leg Features）神经网络与NSG（Non‑Slip Gait）步态模型，利用2D LiDAR的腿部簇特征通过深度强化学习实现社交机器人导航。

**💡 创新点**

创新点包括：1) 在传感器层面显式建模人类腿部动态以克服传统圆盘近似导致的感知盲区；2) 结合CNN、时间注意力和MLP的混合架构，直接从原始LiDAR堆栈中提取运动信息；3) 引入yielding奖励和Yielding Score量化并强化主动让路行为。

**🔧 技术方法**

使用的技术包括：深度强化学习（PPO、SAC、TQC）训练CALF；基于JAX实现的LegNav轻量化2D仿真器（可在单卡GPU上每秒超10万步）；非滑动步态(NSG)模型；卷积编码、时间自注意力、MLP融合等网络模块；以及自定义的奖励函数和社会合规性指标。

**📊 数据集**

数据集方面：采用LegNav仿真环境中的7个训练场景与6个测试场景（随机化参数，覆盖多种人流与障碍配置），并在真实TurtleBot 4平台上进行零样本部署验证；未使用公开人类轨迹或真实LiDAR数据集。

**📈 对比分析**

性能比较：与经典模型规划器（DWA、MPPI）、其他E2E RL基线（TAGD、VanillaE2E）及CALF的PPO/SAC/TQC版本进行对比；CALF_PPO在成功率（≈95%）和主动碰撞率（≈3%）上显著优于多数方法，同时保持较高的yielding score（≈32%）与平滑运动；TQC虽收敛快但碰撞率与yielding不佳，SAC在速度范围内表现稳定但略逊于PPO。

**⚠️ 局限性**

局限性：1) 依赖仿真中的腿部动态模型，真实世界中人类步态与鞋子形状的多样性可能导致性能下降；2) 仅在二维LiDAR场景验证，对三维感知或不同高度传感器的适应性未做评估；3) 在极端拥堵或高速场景下的安全性与时效性仍待进一步提升。

---

## 304. $Σ$-Mem: An Online Reliability Memory for LLM-based Multi-Agent Systems

**arXiv ID:** 2607.27958 | [PDF](https://arxiv.org/pdf/2607.27958v1)

**作者:** Peilin Feng `[一作]` (Nanyang Technological University), Soujanya Poria `[通讯]` (Nanyang Technological University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种在线可靠性记忆（Σ-Mem），为大型语言模型（LLM）多智能体系统（MAS）记录并持续更新每个同伴的历史可靠性（competence）和同伴间的关系（relationship）信息，支持在决策时通过残差驱动、路由或加权投票等方式使用这些记忆。

**💡 创新点**

创新点包括：①使用对称矩阵做可靠性记忆，借助Weyl不等式保证单步更新被限制，避免记忆被噪声冲击；②分为两种互补的可靠性证据——历史可靠性和同伴关系；③提供通用写入-读取接口，记忆可在不同协同机制（残差驱动、路由、投票）中复用；④在长期在线学习中保持稳定性并可迁移至未见同伴、任务分布与决策方式。

**🔧 技术方法**

技术手段包括：对称矩阵的衰减式更新（γM + η c ϕϕᵀ）、同伴关系矩阵的更新（γ_G G + η_G q qᵀ）、残差驱动的Transformer层（δ_p = g P M_p ϕ），利用Bayes式后验推断同伴可信度，权衡个体可靠性和关系证据；同时使用Weyl不等式保证谱稳定性。

**📊 数据集**

使用的数据集：①训练集2963个事件，覆盖数学推理、检索增强问答（RAG）和代码生成；②混合反事实基准（CF@0/50/70/90）评估可靠性漂移；③OOV基准包括PIQA、MMLU、OpenBookQA、SciQ、BBH、SuperGLUE；④在测试时扩展同伴池（加入Llama-3.2-3B-Instruct、BitCPM-CANN-3B）来检验对未见同伴的泛化。

**📈 对比分析**

对比方法：基准Qwen中心模型、Majority Voting、Best Fixed Peer、Oracle Reputation等。Σ-Mem在CF@90上将Qwen3-0.6B的准确率从46.22%提升至71.10%；在混合反事实和OOV测试中，Σ-Mem在27/30个案例中优于基准，特别是在BBH任务上提升显著；在未见同伴、未见域上也保持性能提升，表明记忆可迁移。实验展示了记忆在路由、加权投票、残差驱动等多决策模式下均能提升整体准确率。

**⚠️ 局限性**

局限性：①当可靠性信息模糊（如CF@50）时，历史记忆可能积累误导信号，导致决策失误；②需要外部的正确性反馈才能写入记忆，在无反馈或低反馈比例下性能下降；③当前设计主要针对基于内容的LLM，若同伴生成策略或任务类型出现根本性变化，记忆的迁移效果可能受限；④对训练样本分布的依赖较大，若训练集中出现偏差，记忆的可信度评估可能失真。

---

## 305. Compact Representation of Mipmapped SVBRDFs via Shared Gaussians

**arXiv ID:** 2607.27943 | [PDF](https://arxiv.org/pdf/2607.27943v1)

**作者:** Fengdi Zhang `[一作]` (Tencent), Hongwei Li `[通讯]` (Tencent)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Gaussian Texture Compression (GTC)，一种基于共享 2D 高斯原语的压缩方法，用以高效压缩多级 SVBRDF 纹理堆栈，并提供无神经网络、可随机访问的实时解码。

**💡 创新点**

创新点在于：① 将多级 mip 级别与不同材质通道之间的共享空间结构映射为可复用的 2D 高斯原语；② 设计进阶化优化管线和基于 group‑lasso 的正则化+稀疏裁剪策略，实现内容自适应的 Gaussians 分配；③ 提供 GPU 友好的两阶段随机访问解码方案，兼顾高压缩率与实时渲染需求。

**🔧 技术方法**

使用的技术包括：共享高斯表示（参数化为位置、尺度、旋转、特征向量和 LOD 标签）、进阶化优化（从粗到细级别逐步添加 Gaussians）、残差引导初始化、基于 group‑lasso 的正则化、稀疏裁剪、以及两阶段 GPU 采样+合成解码。

**📊 数据集**

在 20 组 SVBRDF 纹理数据集上评估，数据集包含 10 个 UV‑atlas 资产（TexVerse）和 10 个 PBR 材料资产（FreeStylized），每组纹理从 2048×2048 开始，完整 mip 级别堆栈以及至少 3 通道（基础色、法线、AO/粗糙/金属/高度）。

**📈 对比分析**

与 JPEG、JPEG‑XL、ASTC、NTC、Image‑GS 等基线对比；GTC 在非神经随机访问压缩方案中，在等‑MIP 聚合下实现更优的 PSNR/SSIM，且比 ASTC 低 26.5% 存储；在等‑MIP 评价下，PSNR 提升 3~4 dB，FLIP 降低 20–27%；在纯 rate‑distortion 上略逊于 NTC，但不需要神经解码，且在低比特率下表现更好。

**⚠️ 局限性**

局限性包括：① 难以高精度重现极高频噪声与细小重复结构；② 对重复图案缺乏显式长程重复利用；③ 需要针对每个资产进行逐步优化，导致大规模材质库的处理瓶颈。

---

## 306. VizPilot: Automated Onboarding for SVG-based Composite Visualizations using Multimodal LLMs

**arXiv ID:** 2607.27938 | [PDF](https://arxiv.org/pdf/2607.27938v1)

**作者:** Nishaanthini Gnanavel `[一作]` (Nanyang Technological University), Yong Wang `[通讯]` (Nanyang Technological University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出并实现了 VizPilot，一种基于浏览器扩展的自动化引导系统，用于为自定义组合可视化提供无人工注释的入门体验。它包含两大模块：①Composite Visualization Analyzer（两阶段 MLLM 推理——语义推理与语义映射）将 SVG、位图和可选交互代码转换为与可视化元素精准对应的解释单元；②Onboarding Interface（叙事滚动讲解与自由探索模式）将这些解释单元呈现给最终用户，支持同步高亮、文本、语音等交互。

**💡 创新点**

创新点主要体现在：①平台无关、无手工注释，直接从渲染后的 SVG 及其位图推理语义；②采用多模态 MLLM 与分层选择器推理，并通过程序化多级验证确保映射的准确性；③双模式引导（叙事滚动 + 交互式探索），兼顾结构化学习与用户自主探查；④解耦的部署方式，开发者可在不修改原始代码的情况下生成可复用的 JSON + JS 包。

**🔧 技术方法**

核心技术包括：多模态大型语言模型（如 GPT‑5、GPT‑5.5、Gemini 3.1、Claude 3.5）用于语义推理；层级选择器推理与程序化验证（DOM 检查、组件包含、布局一致性）；浏览器扩展框架；JSON 结构化数据；叙事滚动与 TTS；用户研究评估工具（PSSUQ、NASA‑TLX）。

**📊 数据集**

评估使用了 18 张公开的组合可视化样本（涵盖并列、叠加、嵌套、重载四种设计模式），以及 MatrixWave、PrettiSmart、RuleMatrix、PonziLens、Shanghai Index、EgoLines 等典型案例；此外在用户研究中采用两张可视化（MatrixWave、PrettiSmart）和四个分析问题。

**📈 对比分析**

与文本描述基线相比，实验显示任务准确率基本相同（≈ 95‑100%），但完成时间平均减少 68 秒（p = 0.039）。NASA‑TLX 显著降低了精神负荷、努力和挫折感；PSSUQ 信息质量与系统效用得分提升。图集评估显示：组件分解精度 ≥ 0.94，语义映射准确率 ≥ 0.89，整体端到端成功率在 83‑100% 之间。

**⚠️ 局限性**

局限性包括：①仅支持单一可访问的 SVG 容器，无法处理 Canvas 或高度扁平化的 SVG；②对大型 SVG 的上下文长度有限制，需做目标选择；③MLLM 推理存在延迟，影响实时交互；④交互代码为可选，缺失时交互推断基于外观推测；⑤评估样本有限，缺乏更广泛的实测；⑥系统对多视图仪表盘的支持不完善。

---

## 307. Benign on Label, Malicious by Design: Clean-Label Dormant-to-Activated Backdoor via Machine Unlearning with Removable Camouflage

**arXiv ID:** 2607.27936 | [PDF](https://arxiv.org/pdf/2607.27936v1)

**作者:** Dongdong Zhao `[一作]` (Wuhan University of Technology), Baogang Song `[通讯]` (Wuhan University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种 clean-label 的机器忘却激活后门框架，利用双生成器在训练后保持后门休眠，待攻击者请求忘却指定伪装样本后才被激活。

**💡 创新点**

创新点在于同时学习持久触发关联与可移除抑制机制，并通过双层优化模拟训练+忘却过程，使后门在清洗标签且满足真实删除请求下可控地从休眠过渡到激活。

**🔧 技术方法**

使用双生成器（触发器生成器与伪装生成器）、低频投影+高斯平滑、梯度对抗抑制、目标间距损失，以及针对 SISA、First‑Order、PUMA 等机器忘却算法的 bilevel 优化。

**📊 数据集**

在 CIFAR‑10 与 ImageNet‑10（随机挑选10类）上进行实验，采用 ResNet‑18、VGG‑16、MobileNetV2 等架构。

**📈 对比分析**

与 UBA‑Inf、UNCLEAN、Sleeper Agent 等基线比较；在不同忘却算法下预忘时 ASR 低，忘却后 ASR‑U 高，ΔASR 最大，且在保持 BA‑U 可接受的前提下表现最优。

**⚠️ 局限性**

局限在于对不同模型架构或忘却算法的匹配仍需进一步验证；攻击者需可访问同类模型和训练过程；对高度防御或随机化的忘却机制仍不确定。

---

## 308. Meta-Task: Turning Terminal Task Synthesis into a Terminal Task for Scalable Agent Training

**arXiv ID:** 2607.27929 | [PDF](https://arxiv.org/pdf/2607.27929v1)

**作者:** Zhihong Pan `[一作]` (University of Science and Technology of China), Zhaohua Yang `[通讯]` (Meituan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了Meta-Task框架，将终端任务合成视为终端任务本身，利用LLM在Docker容器内端到端生成并自我验证完整的任务包；

**💡 创新点**

通过在真实终端环境中实时执行与验证、分解多维度多阶段多样性控制、可选外部材料获取以及LLM-as-Judge轨迹过滤，突破了传统合成的可靠性与多样性瓶颈；

**🔧 技术方法**

使用了Qwen3.5-397B-A17B-FP8大模型、Claude Code代理架构、vLLM、Harbor调度、Terminus-2执行框架、LLM-as-Judge审核、Docker容器等技术；

**📊 数据集**

主要使用自己生成的约15k完整任务包以及其中筛选出的3,221条SFT轨迹作为训练集，评测基准为Terminal‑Bench 2.0；

**📈 对比分析**

与闭源、开源以及同期终端任务合成方法对比，基于Avg Pass@1/Pass@3评测。Qwen3‑14B从5.2%提升到22.5%，Qwen3‑32B从4.1%提升到31.8%，在相同模型规模下优于多种对手，并且仅使用比其他方法少得多的训练数据；

**⚠️ 局限性**

仅支持Linux Docker环境，未扩展到Windows/macOS；多样性种子（技术主题与场景约束）手工挑选，缺乏自动化从真实开发社区采集的广泛来源。

---

## 309. One Anchor for All: Unified Multilingual and Multimodal Safety Alignment for LVLMs

**arXiv ID:** 2607.27917 | [PDF](https://arxiv.org/pdf/2607.27917v1)

**作者:** Enyi Shi `[一作]` (Nanjing University of Science and Technology), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出基于跨语言跨模态共享安全神经元（MLS‑Neurons）的神经元级安全对齐框架，仅通过更新极少量参数（约0.03%）实现多语言多模态安全提升。

**💡 创新点**

通过识别同时在多语言和多模态中激活的安全神经元，并以英文安全信息为锚点，构建可迁移的安全语义锚，实现在仅用英文数据即可对多语言多模态进行安全对齐。

**🔧 技术方法**

使用神经元激活强度与下游影响的功能显著性评分、跨语言跨模态交集提取MLS‑Neurons，结合掩码低秩更新（masked LoRA）仅更新这些神经元。

**📊 数据集**

主要数据集包括Lingua‑SafetyBench、MM‑Bench、MMMU、MGSM、MM‑Vet等，用于多语言多模态安全评测与通用能力验证。

**📈 对比分析**

与ESCO、ASTRA、XSAFETY、MLC、全量微调及LoRA等基线比较，平均攻击成功率（ASR）下降30–50%，同时保持或提升MMMU/MGSM等通用指标，证明在极低参数更新下达到SOTA水平。

**⚠️ 局限性**

目前实验覆盖的语言、攻击场景有限，未检验方言或文化语境下的效果；安全数据仍相对稀缺，未来需进一步验证更广泛的多模态安全场景。

---

## 310. IFHierBench: Hierarchical Instruction Following for Large Language Models

**arXiv ID:** 2607.27912 | [PDF](https://arxiv.org/pdf/2607.27912v1)

**作者:** Yuetian Mao `[一作]` (Technical University of Munich), Chunyang Chen `[通讯]` (Technical University of Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并发布了一个层次化指令跟随基准IFHierBench，用于评估LLM在多层嵌套约束下的表现。

**💡 创新点**

创新点在于把约束组织成树形结构，支持在不同深度（0–3）对每个子区域单独进行确定性检测，解决了现有基准只能进行平面检查的问题。

**🔧 技术方法**

技术路径包括：任务预处理（清洗任务描述、抽取上下文、关键词挖掘、体裁-格式映射）；层次采样（生成约束树、内容约束、参数填充）；自动生成Python层级检查器；提示合成与约束合并；实验使用七大LLM（含OpenAI、Anthropic、Cohere等）和Ollama本地模型。

**📊 数据集**

数据集来源：PromptSet 1,232条真实GitHub LLM应用提示，结合Ifeval、IFBench、ComplexBench中的约束模板进行合成；Bench自身包含约束树及对应的检查器。

**📈 对比分析**

评估方法：对每个提示使用严格的prompt‑level准确率和instruction‑level准确率；实验结果显示最强模型在最高深度3时仅约50%提示通过，且准确率随约束深度急剧下降，低层模型甚至在深度1就跌至30%以下。

**⚠️ 局限性**

局限性：合成提示的语言更为模板化，可能低估真实用户提示的多样性；任务预处理、关键词抽取等步骤本身由LLM完成，受模型自身解析能力影响。

---

## 311. Integrating Contextual Embeddings into Evaluation of Expressive MIDI Piano Performances

**arXiv ID:** 2607.27909 | [PDF](https://arxiv.org/pdf/2607.27909v1)

**作者:** Dmitrii Gavrilev `[一作]` (Skolkovo Institute of Science and Technology), Vladimir Viro `[通讯]` (Peachnote GmbH)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出利用自监督音乐模型的上下文嵌入来评估MIDI钢琴表现的深度特征指标，并与传统属性级指标进行对比。

**💡 创新点**

创新点在于引入Kernel Music Distance（KMD）和Kernel Performance Distance（KPD）等无对齐、上下文感知的分布相似度度量，并验证其与人类听感的一致性。

**🔧 技术方法**

采用Aria和CLaMP3两个预训练的MIDI嵌入模型，结合Maximum Mean Discrepancy、Mahalanobis距离等技术构建评价指标。

**📊 数据集**

使用ASAP、ATEPP、PDMX等公开的古典钢琴MIDI数据集，并对VirtuosoNet、M2M和PianoFlow等表现模型进行实验。

**📈 对比分析**

在对比实验中，深度特征指标在自然度和表现力上与听众MOS的Kendall‑τ_B相关性高于传统属性相关系数，且能够捕捉到属性级指标忽略的上下文扰动，整体排名与人类评价一致。

**⚠️ 局限性**

局限性包括仅针对西方古典单人钢琴作品、嵌入模型对细微表现的敏感度有限、对齐误差和缺失音符的处理仍需改进，以及计算开销相对较大。

---

## 312. Unifying Adversarially Robust Model Experts in Vision-Language Models

**arXiv ID:** 2607.27897 | [PDF](https://arxiv.org/pdf/2607.27897v1)

**作者:** Nguyen Duc Thai `[一作]` (Nanyang Technological University), Yew-Soon Ong `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 CARE 框架，通过协同对抗微调和嵌入对齐，在视觉-语言模型中融合多种鲁棒专家以提升对抗鲁棒性。

**💡 创新点**

创新点在于同时训练多个专业化的对抗专家，并通过嵌入空间的协同对齐（Expert Harmonization）实现知识共享与参数收敛，从而兼顾图像-文本对齐与图像不变性两种鲁棒性。

**🔧 技术方法**

采用 CLIP 预训练模型，结合对抗生成（PGD）与自动攻击（AutoAttack）评估，使用嵌入对齐损失（余弦相似度）实现专家之间的协同，并使用指数移动平均（EMA）融合最终模型。

**📊 数据集**

在 ImageNet-1K、13 个零样本数据集（如 STL-10、CIFAR-10/100、Caltech-101 等）以及视觉-语言下游任务（COCO、Flickr30k 的图像字幕和 TextVQA、VQAv2 的视觉问答）进行训练与评估。

**📈 对比分析**

与单一专家 TeCoA、FARE 以及其他对抗训练方法比较，CARE 在图像分类、零样本分类、下游视觉语言任务上均表现出更高的鲁棒准确率，尤其在 ε=4/255 的攻击下提升约 2–8% 以上。

**⚠️ 局限性**

主要局限包括训练成本增加（两倍模型、三倍显存）、对不同专家划分的依赖以及对专家间协同方式的经验选择，对更大规模模型的可扩展性仍待改进。

---

## 313. MMHBench: A Multi-Perspective Benchmark for Mental Health Understanding in Long-Form Videos

**arXiv ID:** 2607.27895 | [PDF](https://arxiv.org/pdf/2607.27895v1)

**作者:** Jinpeng Hu `[一作]` (Hefei University of Technology), Meng Wang `[通讯]` (Hefei University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出MMHBench——一套针对长视频多视角心理健康理解的基准，包含第一人称与第三人称多选题；

**💡 创新点**

创新点在于：①构建双视角（主观与客观）评价框架；②采用多代理生成与专家审核相结合的自动化问卷生成流程；③系统评估了视觉与文本多模态信息对模型性能的影响；

**🔧 技术方法**

使用了多代理问题生成（MAQG）、文本与多模态评估器、迭代优化、答案泄露过滤、专家审核等技术；

**📊 数据集**

使用了268段公开心理访谈/自述视频，配合2184道多选题，覆盖家庭、亲密、学业、职业、健康等五大主题；

**📈 对比分析**

通过与22款开源及闭源大模型的准确率对比，发现GPT‑5.5表现最佳；模型在第三人称题上表现更好，第一人称题难度更高；帧密度、视频时长、问题长度等因素显著影响性能；

**⚠️ 局限性**

主要局限在：①模型在第一人称隐含心理推理上仍低效；②对隐含情绪与心理机制的识别错误较多；③基准受公开视频数据、标签人工审核与潜在隐私/伦理风险限制；

---

## 314. RoboBRIDGE: A Modular Framework for Bridging Policies to Robust Real-World Robotic Agents

**arXiv ID:** 2607.27881 | [PDF](https://arxiv.org/pdf/2607.27881v1)

**作者:** Sihyung Yoon `[一作]` (Sungkyunkwan University), Honguk Woo `[通讯]` (Sungkyunkwan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种模块化的框架，将预训练的视觉-语言-动作（VLA）模型转化为可靠的机器人代理，解决了执行失败检测、长时间一致性和跨域鲁棒性的问题。

**💡 创新点**

创新点在于通过五个协调模块（监控器、感知器、规划器、控制器和机器人接口）提供系统化的失败恢复和反应式异步规划，提升了机器人在动态环境中的操作能力。

**🔧 技术方法**

使用了模块化的控制框架，结合了两阶段的监控和反应式异步规划技术，此外还引入了专用的LoRA适配器进行原始技能微调。

**📊 数据集**

在LIBERO和RoboCasa等模拟环境中进行评估，并在真实世界的多个机器人平台上进行测试，使用了三种VLA骨干网络（SmolVLA、π_0.5和GR00T-N1.5）。

**📈 对比分析**

与独立部署的VLA模型相比，该框架在多个任务上表现出一致的性能提升，特别是在长时间任务和环境变化下，成功率显著提高。

**⚠️ 局限性**

局限性在于当前的原始技能词汇主要覆盖单臂桌面行为，接触丰富的交互、可变形物体和双手任务需要更丰富的原始技能和状态表示。此外，监控阈值和恢复规则目前是手动设置的，未来可以通过交互数据学习这些规则。

---

## 315. ARES: Adaptive Reasoning-Effort Steering for PPA- and Cost-Aware RTL Optimization with LLM Agents

**arXiv ID:** 2607.27879 | [PDF](https://arxiv.org/pdf/2607.27879v1)

**作者:** Stef Cuyckens `[一作]` (KU Leuven), Marian Verhelst `[通讯]` (KU Leuven)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于大型语言模型（LLM）的 RTL 优化器，通过迭代编辑 RTL、合成验证并计算 PPA 指标，同时记录每一次 LLM 调用的归一化美元成本，以实现对优化质量与成本的联合评估。

**💡 创新点**

核心创新点包括：①引入每次 LLM 调用的归一化美元成本与 FoM 并列展示，实现不同优化策略的公平比较；②实验证明跨设计长短期记忆的结构化构建对最终 FoM 影响不大；③提出适应性推理努力策略（使用耐心计数器在进展停滞时提升推理深度），显著提升在相同成本下的 FoM。

**🔧 技术方法**

技术实现主要包括：LLM 代理与商业 EDA 工具链（Synopsys Design Compiler + PrimeTime）的集成；基于 OpenRouter 价格模型的 token 计费与成本归一化；在每次迭代中使用功能验证、合成、网表验证及 FoM 计算；以及基于训练集拟合的耐心计数器参数 (p=3, w=2.8, κ=0.05) 的适应性推理控制。

**📊 数据集**

使用了 24 个开源 RTL 模块（21 份训练集，3 份测试集），其中包含 Dr. RTL 20 份（除 LSTM 模块）、FFT、Huffman、CORDIC、FFT、JPEG DCT 等。另对 MX 乘累加单元进行进一步验证。

**📈 对比分析**

与两大基线（REvolution 与 Dr. RTL）在相同商业工具链与 FoM 定义下进行对比。实验显示：①自适应推理策略在测试设计上比固定低/中/高努力至少低 23–27% FoM；②在 MX MAC 单元上闭合了 83% 与手工优化之间的差距；③相较于 REvolution，FoM 降低至 0.694，仅耗 15 高调用；相较于 Dr. RTL，FoM 降低至 0.694，成本仅为其 12% 的 token。

**⚠️ 局限性**

局限性包括：仍需高成本的商业 LLM 调用；适应性策略参数需在训练集上拟合，可能不易迁移到更大规模或不同工艺的设计；长短期记忆构建未能显著提升性能，说明知识迁移仍有限；实验覆盖的 RTL 设计规模相对有限，尚未验证在更大规模、复杂系统级设计中的可扩展性。

---

## 316. Orca: Neural Operators for Causal Reasoning in Continuous Time

**arXiv ID:** 2607.27867 | [PDF](https://arxiv.org/pdf/2607.27867v1)

**作者:** Gerrit Großmann `[一作]` (German Research Center for Artificial Intelligence), Sebastian J. Vollmer `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Orca 框架，利用神经算子学习连续时间动态系统的因果机制，并支持因果效应估计和个体反事实推理。

**💡 创新点**

创新点在于把神经算子应用于因果图，处理时间延迟、非即时循环、以及离散采样的不一致；并提出了可解析的核算子和通用算子两种实现。

**🔧 技术方法**

使用深度学习中的神经算子（如 Fourier 神经算子、DeepONet 等）、深度集合结构、梯度下降、以及最优运输匹配等技术。

**📊 数据集**

使用合成数据集，包括糖胺-胰岛素模型、肿瘤生长模型以及相关的模拟系统；没有使用真实数据。

**📈 对比分析**

与传统的静态结构因果模型和无因果图的神经算子做对比；在剂量反应曲线估计和个体反事实精度上，Orca 的核算子和通用算子均显著优于基线，误差随采样密度下降并收敛于真实值。

**⚠️ 局限性**

局限包括假设已知因果图且无隐藏共因；对非加性噪声的处理仍是草案；对测量噪声、异步观测不鲁棒；模型仅在合成数据验证，缺乏真实世界案例；训练需大量高质量时间序列。

---

## 317. FiRE: Enhancing MLLMs with Fine-Grained Context Learning for Complex Image Retrieval

**arXiv ID:** 2607.27959 | [PDF](https://arxiv.org/pdf/2607.27959v1)

**作者:** Bohan Hou `[一作]` (Shandong University), Xiangyu Zhao `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了细粒度多模态五元组数据集 FiGMaQ 并提出两阶段细粒度微调策略 FiRE，用于提升多模态大语言模型在复杂图像检索任务中的表现。

**💡 创新点**

创新点包括：① 自动化生成细粒度多模态五元组数据集，细粒度图像描述和模糊式修改文本；② 将上下文推理与检索对齐拆分为两阶段微调，分别强化多模态上下文理解和查询‑目标对齐能力。

**🔧 技术方法**

技术手段涵盖：多模态大语言模型 BLIP‑3，LoRA 参数高效微调；Chain‑of‑Thought 细粒度描述生成；Fine‑grained semantic similarity 识别图像对；InfoNCE 与 Recall@k surrogate 损失进行检索对齐优化。

**📊 数据集**

数据集：基于 ImageNet1K 生成的 87K 样本的 FiGMaQ；评测使用 CIRR、CIRCO、FashionIQ、Urban1K、Visual Dialog、COCO、Flickr 等公开检索数据集。

**📈 对比分析**

与多模态检索基线（E5‑V、MCL）及专门 CIR 方法对比，FiRE 在零样本设置下的 CIRR、CIRCO、FashionIQ 等复杂检索任务上均提升 8–13% 以上，部分任务接近有监督结果；在短文本检索与跨模态检索上与 E5‑V 竞争。

**⚠️ 局限性**

局限性：仅使用 4B 参数 BLIP‑3，视觉编码器规模仍受限；自动生成的数据可能存在质量偏差；未探究更大规模 LLM 或多模态重排序器，实时检索性能未作充分评估。

---

## 318. Argonaut: Interactive Visual Exploration for Distributed Optimization

**arXiv ID:** 2607.27946 | [PDF](https://arxiv.org/pdf/2607.27946v1)

**作者:** Srijoni Majumdar `[一作]` (University Of Leeds), Evangelos Pournaras `[通讯]` (University Of Leeds)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一个名为Argonaut的轻量级、容器化、可视化交互平台，用于在分布式离散选择优化中实现构造、优化与分析的一体化循环；

**💡 创新点**

创新点在于将完整的优化过程可视化并可实时交互（人机协同），支持多种算法后端（暴力搜索、树形分布式迭代），并以容器化方式易于部署；

**🔧 技术方法**

采用Node.js+React前端、Python/Java算法后端、Docker容器化、Python可视化库（NetworkX、Matplotlib）以及基于REST的统一接口；

**📊 数据集**

使用真实域数据（家庭用电5600个代理、共享出行2300个代理、传感器数据共享72个代理）和合成高斯分布数据集，规模从数十到数千代理、数十至数百决策属性；

**📈 对比分析**

与传统集中式暴力搜索对比，Argonaut在本地环境下对200个代理、100属性的规模可在30秒内完成；相较于基准算法，其可交互优化使平均最优性间隙降低约24.9%，并在部分实验中达成全局最优；

**⚠️ 局限性**

局限性包括：对极大决策空间仍受暴力搜索的计算瓶颈限制；云端性能受限于免费实例资源；当前解释性功能尚未完善，难以精准解释代理行为变化；

---

## 319. From Scoring to Acting: Outcome-Verified Comparative Self-Distillation for LLM Agents

**arXiv ID:** 2607.27937 | [PDF](https://arxiv.org/pdf/2607.27937v1)

**作者:** Xu Xia `[一作]` (Southeast University), Yong Li `[通讯]` (Zhongguancun Academy)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种新的 on‑policy 自蒸馏框架 Outcome‑Verified Comparative Self‑Distillation（OVCSD），通过在学生失败轨迹中构建前缀树、从共享状态处引入带技能上下文的教师并只保留环境验证成功的后续，随后在第一次状态对齐分歧处进行局部对比学习并将教师后缀蒸馏给学生，从而实现无技能部署下的能力内部化。

**💡 创新点**

① 只接受环境验证成功的教师行为（Outcome‑Verified），① 用前缀树自适应挑选最深共享状态进行干预并回退；② 在首次状态对齐分歧处做局部优势学习（Alignment‑Aware Comparative Learning）；③ 结合后缀 KL 蒸馏实现任务完成行为的迁移。

**🔧 技术方法**

采用前缀树构造、适应性 outcome‑verified 干预、局部对齐分歧优势学习、后缀 KL 蒸馏、GRPO 强化学习框架和技能化教师（同模型的技能上下文）。

**📊 数据集**

ALFWorld（多步家庭任务）与 WebShop（电商购物任务）两个公开基准。

**📈 对比分析**

与 GRPO、OPSD、Skill‑SD、SDAR 等基线在 Qwen3‑1.7B、Qwen2.5‑3B、Qwen2.5‑7B 三种规模上进行对比。OVCSD 在 ALFWorld 的微平均成功率和 WebShop 的成功率均显著提升，最大提升 29.7 %（ALFWorld）/5.4 %（WebShop），并且仅增加不足 3 % 的训练环境交互成本。

**⚠️ 局限性**

方法依赖教师干预与额外环境交互，复杂度较高；对极短或单步任务的提升有限；实验仅在两类任务上验证，缺乏更广泛的应用验证；需要教师策略与学生同模型，若教师能力不足则效果受限。

---

## 320. Class-Aware Reinforcement Learning for Counterfactual Explanation Generation

**arXiv ID:** 2607.27905 | [PDF](https://arxiv.org/pdf/2607.27905v1)

**作者:** Muhammad Adil Saleem `[一作]` (Institute of Business Administration Karachi), Mary-Anne Williams `[通讯]` (Commonwealth Bank of Australia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了将实例预测类别信息加入强化学习状态表示，从而改进反事实解释（CFE）的生成；通过对比包含与不包含类别信息的RL方法，展示了该改进在训练与测试阶段的优势。

**💡 创新点**

创新点在于首次将被解释模型的预测类别作为RL状态特征，显著提升了策略收敛速度、奖励优化、episode长度缩短以及生成的CFE有效性。

**🔧 技术方法**

主要技术包括：基于PPO的强化学习、XGBoost监督模型、LIME/SHAP特征重要性评估，以及对比实验使用的DiCE与ReLAX算法。

**📊 数据集**

实验使用七个公开数据集，覆盖小型至大型：乳腺癌、德国信用、成人收入、默认信用、青霉素、心脏病和森林覆盖。

**📈 对比分析**

比较方法：在训练阶段对比奖励与episode长度；在测试阶段对比有效性、稀疏性与邻近度；结果显示包含类别信息的RL在有效性提升至≈99%（比基线多8%）且稀疏性、邻近度整体优于或相当于DiCE与ReLAX；训练阶段收敛更快且方差更低。

**⚠️ 局限性**

局限性包括：仅针对表格数据；未探索多类别或连续输出的情况；奖励函数主要关注终端奖励，缺少中间奖励；缺乏对因果约束的考虑；计算资源有限，未在GPU上加速；对下游任务的实际效益尚未评估。

---

## 321. Contrastive Concept Importance: Explaining Pairwise Class Decisions Through Automatically Extracted Concept Representations

**arXiv ID:** 2607.27904 | [PDF](https://arxiv.org/pdf/2607.27904v1)

**作者:** Roel Visser `[一作]` (Bielefeld University), Barbara Hammer `[通讯]` (Bielefeld University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种名为Contrastive Concept Importance（CCI）的方法，用于通过自动提取的视觉概念基础解释模型在两个类之间的决策。

**💡 创新点**

创新点在于将概念重要性从单类视角转向对比视角，量化目标类与对比类对数值边界的贡献，并能够区分共享、单侧和直接对比的概念效应。

**🔧 技术方法**

技术方法包括使用非负矩阵分解（NMF）构建概念基底，利用积分梯度（Integrated Gradients）对目标-对比对数值边界进行概念层面的归因，并通过概念插入/删除曲线、对数值拆解和语义层级检验来评估归因结果。

**📊 数据集**

实验在ImageNet-1k数据集上进行，使用ResNet50模型，并针对特定类对（如beagle vs. English foxhound）提取10个概念进行分析。

**📈 对比分析**

与传统的单类概念重要性相比，CCI在概念插入/删除曲线中能够更精准地控制目标-对比对数值边界的方向；在对数值拆解中揭示高混淆和低混淆类对的共享与对比证据差异；在语义层级检验中显示对比概念对细粒度类对的影响更为显著，整体性能优于非对比归因。

**⚠️ 局限性**

限制包括目前仅使用类特定概念基底，无法直接在不同类间共享概念字典；对不确定性场景（如拒绝选项或可靠预测集）尚未扩展；实验主要集中在单一模型和数据集，需进一步验证在其他网络结构和多模态数据上的泛化。

---

## 322. Data-free neural PDE solvers based on Graph Neural Networks and weak forms

**arXiv ID:** 2607.27901 | [PDF](https://arxiv.org/pdf/2607.27901v1)

**作者:** Mikel M. Iparraguirre `[一作]` (Universidad de Zaragoza), Elias Cueto `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种物理信息、无训练数据的三维固体力学神经求解器，利用图神经网络与消息传递，并在弱形式下基于形状函数梯度直接计算残差。

**💡 创新点**

创新点包括：① 无需高保真训练数据，直接用弱形式物理损失；② 采用形状函数梯度代替自动微分，提升数值稳定性；③ 结合 MeshGraphNet‑Transformer 混合结构以支持大尺寸、复杂几何；④ 推出基于残差的测试时自适应细化。

**🔧 技术方法**

使用的技术：MeshGraphNet‑Transformer 结合 MPNN 与物理注意力 Transformer；弱形式物理损失（残差、负雅可比惩罚、边界条件惩罚）；形状函数梯度计算内力；Neo‑Hookean 超弹性材料；Adam + cosine‑annealed 学习率调度。

**📊 数据集**

使用的“数据集”是人工生成的查询集合：100 条立方体负载案例、200 条穿孔立方体与板材负载案例，按 70/15/15 划分为训练/验证/测试集；不依赖外部真实数据。

**📈 对比分析**

通过对比基于物理损失与仅基于数据的监督损失，评估残差（r_max）和位移 RMSE；物理信息训练在残差 <1% 时 RMSE <5%，而数据驱动仅在 RMSE 较低但残差较大；测试时细化可将残差降至阈值以下，推算速度仅需一次前向传播。

**⚠️ 局限性**

局限性：仅实现 3D 线性四面体单点积分；仅针对 Neo‑Hookean 超弹性材料，未包含塑性、损伤等路径相关材料；采用线性形状函数，未来需扩展到高阶单元；目前仅处理均匀材料。

---

## 323. CoRE-UIR: Prior-guided common and residual experts for efficient all-in-one remote sensing image restoration

**arXiv ID:** 2607.27898 | [PDF](https://arxiv.org/pdf/2607.27898v1)

**作者:** Zaiyan Zhang `[一作]` (Wuhan University), Liangpei Zhang `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种基于先验引导的全局‑局部框架 CoRE‑UIR，用以实现高效的无人机与卫星图像一站式恢复。

**💡 创新点**

通过将共通密集专家与低秩残差专家分离、引入冻结 CLIP 的先验嵌入和全局特征调制，实现了去重与高适配的专家化设计。

**🔧 技术方法**

采用 CLIP 预训练视觉语言模型做先验嵌入、轻量化 Adapter、全局特征调制 GFM、Common‑Residual Expert Block (CoRE)、低秩 LoRA 方式专家、PG‑Router 等技术。

**📊 数据集**

构建 MDVD‑108K UAV 多退化数据集（108K 对单/复合退化合成及 500 张真实退化图像）和 MDRS‑Landsat 卫星退化基准。

**📈 对比分析**

与多种单任务与全局恢复基线对比，CoRE‑UIR 在 MDVD‑108K 单/复合退化以及 MDRS‑Landsat 上均达成最高 PSNR/SSIM/LPIPS，较最强基线 BaryIR 提升约 1~2 dB，推理速度提升近 12×，显存占用降低 85%。

**⚠️ 局限性**

仅覆盖已知退化类型，未对更复杂或完全未知的混合退化以及其他遥感下游任务进行充分验证。

---

## 324. An LP Algorithm for Counting Eulerian Orientations Through the Lens of Quasi-polymorphism

**arXiv ID:** 2607.27961 | [PDF](https://arxiv.org/pdf/2607.27961v1)

**作者:** Jincheng Guan `[一作]` (University of Science and Technology of China), Ke Shi `[通讯]` (University of Science and Technology of China)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `847a60d8-a755-47af-ba5d-c5236b9e3083` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过构造线性规划松弛，给出了对之前仅可在FP^NP下求解的所有加权欧拉取向计数（#EO）问题的多项式时间算法，并因此完成了包含奇偶arity签名的复数Holant问题的完整FP与#P硬度二分法。

**💡 创新点**

创新点在于将准多项式（quasi-polymorphism）条件通过线性规划松弛“提升”为普通多项式（polymorphism）条件，从而揭示约束函数的仿射局部结构，消除了对NP oracle的需求，并对特殊签名f_56等难解案例给出了解析。

**🔧 技术方法**

主要技术包括线性规划与其松弛结构分析、三元XOR⊕_3的多项式运算、仿射子空间与Hadamard码的代数工具、符号表征与三次形式（cubic form）的推导，以及多项式可解性的代数化证明。

**📊 数据集**

该工作属于纯理论算法研究，无实验数据集，所有结果均来自数学证明与复杂度分析。

**📈 对比分析**

相较于先前需要NP oracle的FP^NP算法，新方法在多项式时间内完成求解，复杂度为O(poly(n,s))（n为顶点数，s为最大签名支持大小），实现了对所有可解实例的确定性求解，并将理论与算法实现紧密结合。

**⚠️ 局限性**

局限性在于仅适用于满足⊕_3准多项式的签名集合，尚未涵盖更一般的签名或非偶权重欧拉取向；对高阶签名的结构分析仍需进一步研究；此外，实际实现中求解LP的效率可能受到实例规模与LP规模的限制。

---

## 325. Don't Trust the AI Ecosystem: Analyzing Privacy Leakage in Compromised Open-Source Components

**arXiv ID:** 2607.27886 | [PDF](https://arxiv.org/pdf/2607.27886v1)

**作者:** Jin-Seong Kim `[一作]` (Yonsei University), Seok-Hwan Choi `[通讯]` (Yonsei University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为GradLock的训练时注入攻击，在模型训练过程中将敏感数据隐藏到模型参数中，实现对训练数据的无损泄露。

**💡 创新点**

创新点在于引入确定性梯度锁定机制和无状态正弦索引法，既能让注入的数据在训练过程中保持不变，又能在标准压缩、剪枝或微调后仍可恢复，解决了传统LSB/参数注入易被后处理破坏的问题。

**🔧 技术方法**

主要技术包括：梯度掩码动态锁定、确定性索引分配、数据归一化缩放、直接权重注入与解码、以及对模型权重的无状态统计检测。

**📊 数据集**

实验使用了MNIST、Imagenette、CelebA三大图像分类数据集，并在VGG-16、ResNet-18、DenseNet-121以及ViT/Swin Transformer等多种网络结构上验证。

**📈 对比分析**

与传统的LSB编码以及三种主流后训练MI攻击（GMI、KEDMI、PPDG）对比，GradLock在SSIM≈1.0、LPIPS≈0、ASR≈模型原始准确率等指标上均取得最优表现，并能在量化、剪枝、微调后仍保持高恢复率；相比之下，LSB在后处理下崩溃，MI攻击受信息瓶颈限制恢复效果差。

**⚠️ 局限性**

局限性包括：对注入比例（ρ）和缩放因子α的敏感性，过高的锁定比例会略微影响模型精度；当前方法主要针对全连接层，扩展到其他模块需要进一步研究；此外，若采用权重置置换或严格的分布检测等防御，GradLock可能被抑制。

---

## 326. TriShield: Zero-Utility-Loss Defense Against Privacy Backdoors in Federated Language Model Fine-Tuning via Orthogonal Gradient Projection and Optimizer State Entanglement

**arXiv ID:** 2607.27940 | [PDF](https://arxiv.org/pdf/2607.27940v1)

**作者:** Cheng Wei `[一作]` `[通讯]` (Honor Device Co., Ltd.), Cheng Wei (Honor Device Co., Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了三层客户端防御框架 TriShield，阻止 NeuroImprint 风格的隐私后门在联邦 LLM 微调中的重建。

**💡 创新点**

创新点在于将参数异常检测、状态化虚拟迭代和零效用正交投影三种技术组合，理论证明互信息为零且无模型精度损失。

**🔧 技术方法**

主要技术包括参数艺术检测（检测低方差/高相关行）、Adam 预热的虚拟优化步骤（混合动量）以及基于 SVD 的任务子空间投影。

**📊 数据集**

实验数据集涵盖 GPT‑2 117M 与 Llama‑Guard‑3‑1B，使用公开的 32–500 条域相关文本作为辅助数据。

**📈 对比分析**

与 LDP、梯度裁剪、FL‑WBC 等现有防御对比，TriShield 在所有攻击变体下实现 0% 重建率，准确率下降 ≤0.3%，计算开销约 5% GPU/72% CPU。

**⚠️ 局限性**

局限包括对公共辅助数据的依赖、CPU 端高线性代数开销、对特定 PEFT 架构的阈值敏感，以及对非 NeuroImprint 的新型攻击缺乏理论支撑。

---

## 327. Memory Decoder at Scale: A Pretrained, Parametric Long-Term Memory

**arXiv ID:** 2607.27919 | [PDF](https://arxiv.org/pdf/2607.27919v1)

**作者:** Rubin Wei `[一作]` (Shanghai Jiao Tong University), Zhouhan Lin `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出并大规模训练了可独立扩展的长时记忆模块（Memory Decoder at Scale），并将其与冻结的基础语言模型组合使用；

**💡 创新点**

创新点在于能够在预训练规模（6.9B参数、300B tokens）下独立扩展记忆模块，并通过分布式Faiss、OPQ压缩、IVF+HNSW分片搜索以及稀疏kNN存储等技术突破传统非参数检索的存储与检索瓶颈；

**🔧 技术方法**

采用的技术包括分布式Faiss索引、OPQ压缩、IVF+HNSW分片搜索、稀疏kNN分布式存储、以及记忆与基模型的分离式插值融合；

**📊 数据集**

预训练使用Deduplicated Pile（207B tokens，300B tokens用于训练），评估基于17项通用基准、知识密集型任务、TruthfulQA、HaluEval以及生物、法律、金融领域的BioInst、LawBench、FinEval；

**📈 对比分析**

与冻结的Pythia、Qwen3、OLMo等基模型以及CPT、LoRA、RAG等基线对比，结果显示在同等参数或训练预算下，使用大规模记忆能提升平均分数（如410M+6.9B记忆从29.86升至37.34，超过12B基模型），域内记忆在各领域均提升约9分以上；

**⚠️ 局限性**

局限性包括预训练阶段仍需昂贵的离线kNN索引与检索成本，且使用固定插值系数α缺乏动态适配；联合或分阶段训练记忆与基模型可进一步提升性能但会增加训练成本。

---

## 328. A Cross-Architecture Audit of Direction-Based Inference-Time Defences in Vision-Language Models

**arXiv ID:** 2607.27910 | [PDF](https://arxiv.org/pdf/2607.27910v1)

**作者:** Xiangyu Yin `[一作]` (Chalmers University of Technology), Chih-Hong Cheng `[通讯]` (Chalmers University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对视觉语言模型（VLM）的解锁攻击，对五种基于方向的防御策略（均为在解码器层残差流上减去特定方向）进行系统性对比与评估。

**💡 创新点**

发现没有单一方向能在所有模型、层级、攻击设置下同时实现最高拒绝恢复和最低功能损失；图像条件方向（ABL_POS）在 13/15 个实验格点上通过方向特异性检验，并在 LLaVA-1.5 与 Pixtral-12B 上占据 Pareto 前沿，且唯一在所有架构族中保持在噪声底部的功能损失；同时揭示文本仅拒绝方向与多模态图像条件方向在 15 个细胞中呈正余弦一致，表明两类研究共享部分几何结构。

**🔧 技术方法**

使用残差流的方向减法（ABL_POS）、CMRM 样式拒绝方向、ShiftDC 样式攻击特定残差、提示级“忽略图像”指令以及与之同幅度的随机方向控制；并对方向进行基于图像条件的校准、按提示缩放。

**📊 数据集**

VLM 训练模型包括 LLaVA-1.5（7B/13B）、Qwen2.5-VL（7B）、Qwen2-VL-2B、Pixtral-12B；攻击数据集涵盖 FigStep、HADES、JailBreakV、MM‑SafetyBench、MMBench、POPE 共 11 种安全攻击场景。

**📈 对比分析**

采用 15 个 (模型, 层) 细胞，分别在每个细胞上用 47–65 条 jailbreak 触发提示评估拒绝恢复率，并用 MMBench 多选题准确率评估功能损失；与同幅度随机方向对照，并在所有细胞中计算 Pareto 主导数。结果显示 ABL_POS 在大多数细胞中实现拒绝提升且功能损失低于噪声底；其余方法在不同族上表现不一。

**⚠️ 局限性**

局限性包括：仅覆盖公开权重模型，未能验证闭源模型；拒绝判定采用第一步子串词典，可能漏掉软拒绝；功能评估仅用 MMBench 准确率，未涵盖开放式生成质量；方向校准基于特定安全分布，若提示分布变化需重新校准；跨架构迁移失败，表明需要每族单独校准。

---

## 329. An Empirical Study of Coordination Mode as the First-Class Citizen in From-Scratch Multi-Agent Coding

**arXiv ID:** 2607.27877 | [PDF](https://arxiv.org/pdf/2607.27877v1)

**作者:** Yanyu Ren `[一作]` (Tsinghua University), Dan Li `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个真实从零开始的软件开发评测基准MSEval，包含10个真实全栈项目、10种协作拓扑，配合LegoGent运行时与TAgent评估器，记录功能得分、延迟和代币使用。

**💡 创新点**

创新点在于：①使用真实项目与需求文档进行从零开始评测；②引入多种协作拓扑（功能小队、层级专家、流水线、PM监管、QA优先等）；③将部署驱动、迭代反馈与功能、时间、成本三维度统一度量；④通过实验揭示拓扑对速度‑成本‑质量平衡的决定性影响。

**🔧 技术方法**

技术栈包括：大语言模型（Claude Opus、GPT‑5.5、DeepSeek v4 Flash/Pro、GLM‑5.2等）；LegoGent多智能体运行时（同步、邮箱、CI/CD、GitLab+SonarQube+XDeploy）；TAgent自动评估器（UI/API/代码检查，Playwright、OpenAPI探索、LLM辅助依赖图）；以及标准化的日志与度量系统。

**📊 数据集**

使用了10个大学课程最终项目需求文档，涵盖10个业务领域（实时通讯、企业资产管理、众包、需求跟踪、图像处理、电子商务等），并将每个项目拆解为若干模块与加权项，构成100个（项目×拓扑×模型）实验配置。

**📈 对比分析**

通过最多三轮迭代，记录功能完成得分（0–100）、最佳轮的墙钟时间、成本（USD）与代币使用。实验显示：不同拓扑可导致得分±30点、时间翻倍；Claude Opus 97分需110min、$654；GPT‑5.5 96分需74min、$138；GLM‑5.2 最高得分但耗时与成本远高。拓扑是主要性能决定因素，模型次之。

**⚠️ 局限性**

限制包括：随机性与实验规模受限（需人工干预异常）；TAgent与任务侧重Web应用，可能漏检其他类型缺陷；评测不覆盖完整软件生命周期（如维护、事件响应）；仅限三轮迭代，未探索更深层迭代；部分结果需人工确认。

---

## 330. Search as Computation Allocation

**arXiv ID:** 2607.27871 | [PDF](https://arxiv.org/pdf/2607.27871v1)

**作者:** Alexander Tuisov `[一作]` `[通讯]` (Technion Israel Institute Of Technology), Alexander Tuisov (Technion Israel Institute Of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出终端计算分配问题框架，使用贝尔曼方程刻画在固定预算、定价计算和精确证明下的最优计算分配，并通过信息理论阐释价值的计算（VOC）与互信息的关系，进一步推导出权重A*作为对VOC的近似，从而统一了Bandit、MCTS和启发式搜索等不同算法的内在决策逻辑。

**💡 创新点**

创新点包括：①将终端评估的搜索算法归结为同一类计算分配问题；②首次精确表述VOC与互信息的等价性与差异；③通过单步前沿分辨与位置模型，推导出权重A*的原理，揭示经典优先级公式的价值基础。

**🔧 技术方法**

主要技术手段为贝叶斯决策理论、马尔可夫决策过程、价值of计算（VOC）与知识梯度（KG）框架、信息理论（互信息、Pinsker不等式）以及对MCTS和A*等搜索算法的理论分析。

**📊 数据集**

本文为理论性工作，未使用具体数据集；所有讨论基于假设模型和理论推导。

**📈 对比分析**

由于缺乏实验验证，本文未进行方法比较与性能评估；通过理论推导说明了UCB、A*等常用策略与VOC的近似关系，并讨论了它们在不同计算拓扑下的适用性。

**⚠️ 局限性**

局限性包括：①需要先验/预测模型，若模型失真会影响决策；②完整VOC求解一般不可行，往往只能采用一阶近似；③一阶近似可能忽略后续可行计算的选项价值；④互信息与决策价值不完全一致；⑤权重A*的推导依赖简化假设（已存在主导解、等成本、单步前沿分辨等），在更一般设置下不一定适用。

---

## 331. SciSchema.org: A Multidisciplinary Collection of Schemas for Structured Scientific Process Descriptions

**arXiv ID:** 2607.27955 | [PDF](https://arxiv.org/pdf/2607.27955v1)

**作者:** Jennifer D'Souza `[一作]` (TIB Leibniz Information Centre for Science and Technology), Shaokai Yang `[通讯]` (University of Alberta)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了首个SciSchema.org数据集，包含16个跨学科（生物、材料、成像、物理、心理）专家标注的科学过程模式，并发布相应的JSON Schema与SHACL文件。

**💡 创新点**

创新点在于将大型语言模型与领域专家共创的“人机协同schema挖掘”流程实现并验证，可生成可重复、可比对的过程元数据模板，填补了科研流程描述的结构化空白。

**🔧 技术方法**

技术方法包括：① 采用12种LLM（Instruction、Reasoning、Thinking）进行三阶段生成（初始、论文批次1、论文批次2）；② 领域专家提供书面反馈与最终主方案构建；③ 通过JSON Schema Draft 2020‑12与SHACL验证结构与语义；④ 使用Python工具（schema‑miner、rdflib、pySHACL）实现自动化。

**📊 数据集**

数据集来源为每个过程的2~3篇手工挑选论文（约10篇）和扩展语料（≥50篇），并记录了过程说明、论文元数据、LLM生成的中间schema、专家反馈及最终master‑schema。

**📈 对比分析**

通过对比不同阶段的schema词长、属性数、叶子节点和层级深度，以及专家Likert评分与模型来源选择的相关性，评估模型生成的质量；实验显示，后期阶段的schema规模显著增长，评分高的模型在master‑schema构建中被引用频率更高，验证了人机协同的有效性。

**⚠️ 局限性**

局限性包括：① 仅覆盖16种过程，未覆盖所有学科与实验类型；② 只提供结构模板，未实现对论文内容的自动填充；③ 需要领域专家介入，仍存在人工成本；④ 依赖于LLM的生成质量与可访问性，未对模型误差做系统性评估；⑤ 未完成与现有词表（如BioSchemas、CID）或实验平台的映射，导致跨系统互操作性仍需进一步工作。

---

## 332. BladeYOLO: Wind Turbine Blade Defect Detection with Limited Annotations and Weak-Saliency Awareness

**arXiv ID:** 2607.28065 | [PDF](https://arxiv.org/pdf/2607.28065v1)

**作者:** Yabin Xu `[一作]` (Zhejiang Sci Tech University), Sam Kwong `[通讯]` (Lingnan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出BladeYOLO框架，实现风机叶片缺陷检测，针对数据稀缺和弱显著性缺陷进行优化；

**💡 创新点**

创新点包括：1）将DINOv3预训练的ViT骨干迁移至YOLOv12-L，提升跨任务迁移；2）设计Mamba引导的弱缺陷增强模块（Detail‑Enhanced Multi‑scale Branch + Cross‑Mamba），显著提升小尺度低对比缺陷感知；3）引入轻量Style‑Injector，利用频域风格注入增强对环境变化的鲁棒性；

**🔧 技术方法**

技术手段包括：Vision Transformer（DINOv3）、YOLOv12-L、Mamba状态空间模型、波let卷积、多尺度特征融合、频域风格提取与跨注意力注入；

**📊 数据集**

使用自构建的WTBlade‑Defect数据集（1785张，7类缺陷）以及公开的Wind Surface Defect数据集（3790张，5类缺陷）进行评估；

**📈 对比分析**

与多种基线（RT‑DETR、YOLOv8/9/10/11/12、MambaYOLO‑B/L）对比，BladeYOLO在WTBlade‑Defect上mAP₅₀提升至85.8%（相较YOLOv12‑L提升约2.5%），在弱显著子集上mAP₅₀提升至84.1%；在Wind Surface Defect上mAP₅₀达86.5%，显示跨数据集的稳健性；

**⚠️ 局限性**

局限性在于模型仍较为庞大（约43M参数、167.7G FLOPs），对实时嵌入式部署有一定挑战；缺乏像素级缺陷标注，未实现分割任务；对极端光照/雾霾条件的鲁棒性仍有提升空间。

---

## 333. Learning features from Newton's algorithm: a way to accelerate nonlinear parametrized PDE solvers

**arXiv ID:** 2607.28036 | [PDF](https://arxiv.org/pdf/2607.28036v1)

**作者:** Rémy Vallot `[一作]` (ENS Paris-Saclay), Mathilde Mougeot `[通讯]` (ENS Paris-Saclay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种两阶段弱侵入式初始化策略，利用离线 Newton 轨迹学习的解特征和修正方向特征，先用 POD–GP 预测初值，再用 POD–GMRES 进行 Jacobian‑free 校正，从而显著加速非线性 PDE 的求解。

**💡 创新点**

创新点在于将 Newton 迭代增量作为低维修正子空间，引入 Jacobian‑free POD–GMRES 校正，并保持原始高阶 Newton 求解器不变，形成一种新的弱侵入式加速框架。

**🔧 技术方法**

使用的技术包括 POD 降维、Gaussian Process 回归、Jacobian‑free GMRES、POD–GMRES 校正、有限差分近似 Jacobian、离线训练与在线推断。

**📊 数据集**

实验使用了 24 组训练参数、16 组测试参数的 Duffing 型一维/二维非线性弹性膜问题的数据集，参数域为 [0.1,10]×[0.1,10]。

**📈 对比分析**

通过与冷启动 Newton 与单独 POD–GP 预测的对比，保持相同终止精度，结果显示一维平均加速比从 2.92× 提升至 8.77×，二维从 2.53× 提升至 7.67×，部分样例在校正后即可直接收敛。

**⚠️ 局限性**

主要限制包括需要完整 Newton 迭代路径数据库；修正子空间采用全局 POD，二维时需要更多模式且仍不如一维效果；适用性受限于相同网格、单场耦合问题。

---

## 334. Split and Drive: Dual-Axis Disentanglement for Real-Time Gaussian Head Avatars

**arXiv ID:** 2607.28032 | [PDF](https://arxiv.org/pdf/2607.28032v1)

**作者:** MD Wahiduzzaman Khan `[一作]` (University of Technology Sydney), Kaska Musial-Gabrys `[通讯]` (University of Technology Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种双轴去耦合的单图像高质量实时Gaussian头部头像框架SpiD；

**💡 创新点**

创新点在于：将计算轴与特征轴分别去耦合；在模型内部嵌入运动编码器实现无外部追踪器；引入三条几何专门化Gaussian分支（动态、静态、口腔内），实现面部各区域的专属建模；

**🔧 技术方法**

采用3D Gaussian Splatting、FLAME参数化面部模型、DINOv2特征提取、端到端可微渲染、StyleUNet上采样以及光学与几何锚定损失等技术；

**📊 数据集**

使用VFHQ和NeRSemble两大数据集进行训练与评估；

**📈 对比分析**

与10种先进基线在跨身份重现和自重现任务中对比，SpiD在AKD、LPIPS、PSNR、SSIM、MS‑SSIM、L1等多项指标中名列前茅，并在单张A100 GPU上实现43FPS（SpiD）/154FPS（SpiD*）的实时速度，完整驱动流程已包含；

**⚠️ 局限性**

局限性包括：仅基于单张源图，难以处理严重遮挡或极端姿态；光照被烘焙到模型中，缺乏动态照明；口腔内的几何建模仍较简化，可能不足以捕捉复杂口腔运动。

---

## 335. Building a User Foundation Model for the Open Web

**arXiv ID:** 2607.28019 | [PDF](https://arxiv.org/pdf/2607.28019v1)

**作者:** Solal Vernier `[一作]` (Teads), Blaž Škrlj `[通讯]` (Teads)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并部署了基于Transformer的用户基础模型（UFM），在开放Web RTB场景下完成自监督预训练、点击预测微调，并通过LLM驱动的NAS进一步优化；

**💡 创新点**

在非持久用户身份、稀疏历史的开放Web RTB环境中首次提出时间分割正负视图结合MLM与序列级对比学习的预训练策略，利用LLM循环搜索改进训练管线，并将生成的用户表示无缝集成至已有CTR排名器，实现跨模型、跨任务的显著提升；

**🔧 技术方法**

使用Transformer Encoder、Masked Language Modeling、NT-Xent序列对比学习、连续时间编码、Adapter整合、Gated Cross Network、Claude Opus 4.7驱动的NAS以及线上A/B测试；

**📊 数据集**

利用开放Web实时竞价用户浏览历史（publisher、advertiser、interaction 类型）做预训练，点击标签数据做微调，离线验证和线上A/B测试数据做评估；

**📈 对比分析**

与基准GDCN排名器在离线RIG对比，UFM-Base提升+1.065%，UFM-NAS提升+1.354%；在DCN^2上+0.894%，在win-rate模型上+1.197%；线上7天A/B测试CTR +2.13%、eCPC -1.13%、visit rate +2.37%、eCPV -1.47%，所有指标置信区间均不含0；

**⚠️ 局限性**

评估基于单一seed训练、未对比其他搜索方法、存在潜在赢家诅咒风险、仅限单任务微调且未探索其他序列特征。

---

## 336. RepBench: Compiling Benchmarks into Capability Representations for Large Language Models

**arXiv ID:** 2607.28008 | [PDF](https://arxiv.org/pdf/2607.28008v1)

**作者:** Yanshi Li `[一作]` (Shopee), Long Zhang `[通讯]` (Shopee)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了 RepBench，一个基准驱动、可复现的表示探测数据层，包含 94 个文本可探测能力，每个至少由 2 条公开基准支持，并通过跨基准转移评估评测多种读取方法；

**💡 创新点**

创新点在于：①使用多基准闭环采集与审计流程，消除人工合成 probe 的表面模式依赖；②基于跨基准 LOBO 设计的统一转移评估协议；③提供标准化的能力层级表并公开完整代码与数据，促进可复现的对比；

**🔧 技术方法**

技术包括：基准论文爬取与自然语言能力提取、LLM 驱动的聚类与人工审计；跨基准向量池化与标准化；四种读取方法（Diff‑mean、PCA、LR、J‑Lens）；多模型多层深度评估与最佳层挑选；

**📊 数据集**

数据集：13,427 条基准论文 → 182 能力集群 → 94 能力；353 公共基准数据集 → 46,149 probe 文本；12 个开源模型（Qwen3 系列、Qwen3.5、Llama‑3.1‑8B、Gemma‑2‑9B‑IT、Gemma‑4‑12B/31B、R1‑Distill‑Qwen3‑8B、DSv4‑Flash‑Base）。

**📈 对比分析**

评估通过 LOBO 跨基准转移得到每个（能力、模型）对的 AUC；四种读取方法在 1,128 个单元格上比较：Diff‑mean 最高平均 AUC 0.778；LR 在单个单元格中胜率最高（38%）；PCA 较弱；J‑Lens 最低。层深最优在各方法中不同，整体展示读取方法的互补性。

**⚠️ 局限性**

局限性：仅覆盖文本基准，未能处理多模态、规划等能力；读取方法可能仍受基准表面模式影响；仅使用最后 token 激活；负样本选取与层选择尚未最优；数据仅涵盖 94 能力，存在覆盖空缺。

---

## 337. Beyond Classification: Pathology Foundation Models as Detection Encoders for Mitotic Figures

**arXiv ID:** 2607.28007 | [PDF](https://arxiv.org/pdf/2607.28007v1)

**作者:** Sweta Banerjee `[一作]` (Flensburg University of Applied Sciences), Marc Aubreville `[通讯]` (Flensburg University of Applied Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文评估了冻结的病理基础模型在细胞检测任务中的表现，并与传统卷积网络做对比。

**💡 创新点**

创新点在于证明自监督 ViT 预训练的 FM 能直接作为编码器用于密集检测，并在 OOD 上表现更稳健。

**🔧 技术方法**

采用多种 ViT‑based FM（UNI、UNI2‑h、Virchow、Virchow2、H‑Optimus‑0、H‑Optimus‑1）与 Faster R‑CNN、RetinaNet、Deformable DETR 检测头结合，并使用轻量级 Feature Pyramid neck 进行实验。

**📊 数据集**

使用 MIDOG++ 多域 MF 检测数据集和 TUPAC16 作为 OOD 数据集。

**📈 对比分析**

通过在 MIDOG++ 测试集上与全端到端训练的 ResNet‑50 基准进行 F1、precision、recall、FROC‑AUC 比较，最佳冻结 FM+RetinaNet 达到 F1≈0.772，几乎匹敌全端到端 ResNet‑50；在 OOD 上冻结 FM 更优。

**⚠️ 局限性**

局限包括仅使用冻结模型未做微调、只评估单一 OOD 数据集、置信区间重叠导致比较不具显著性等。

---

## 338. MMLDSum-LLM: Multimodal Long-Document Summarization with Visual-Alignment and Keyword-Aware

**arXiv ID:** 2607.28006 | [PDF](https://arxiv.org/pdf/2607.28006v1)

**作者:** Xianpeng Zhang `[一作]` (OPPO Guangdong Mobile Telecommunications Company Limited), Kai Tang `[通讯]` (OPPO Guangdong Mobile Telecommunications Company Limited)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出MMLDSum-Bench基准和MMLDSum-LLM两阶段训练框架，用于多模态长文档摘要；

**💡 创新点**

创新点在于结合视觉对齐权重与关键词加权的SFT，以及基于多目标可验证奖励的GRPO，显著提升关键信息覆盖与跨模态一致性；

**🔧 技术方法**

使用技术包括视觉对齐权重、TF‑IDF关键词权重、两阶段SFT+GRPO、组相对策略优化、可验证奖励（关键词覆盖、图文对齐、ROUGE、长度控制）、LLM评判与原子断言检验；

**📊 数据集**

使用数据集为新建的MMLDSum‑Bench，约5k篇多模态长文档，跨六个领域、五个上下文长度、四个视觉文本比例；

**📈 对比分析**

在统一评估协议下与封闭源/开源模型对比，MMLDSum‑LLM在GPT‑4o/5整体分、原子召回、ITA‑R和ROUGE上显著领先，尤其在16k–64k长度范围内表现最佳，开源SOTA已逼近封闭源；

**⚠️ 局限性**

局限性包括奖励仍依赖代理指标，难以捕捉细粒度错误；长上下文稀疏证据可能被遗漏；评估成本高；基准局限于静态图像，未覆盖动态图或交互式视觉。

---

## 339. Driving up Inference Energy on SNNs: Per-Sample and Universal Sponge Attacks

**arXiv ID:** 2607.27990 | [PDF](https://arxiv.org/pdf/2607.27990v1)

**作者:** Spyridon Raptis `[一作]` (Sorbonne Université), Haralampos-G. Stratigopoulos `[通讯]` (Sorbonne Université)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了两种在原生二值事件输入下对脉冲神经网络（SNN）进行能耗提升的海绵攻击，一种为每样本定制化的梯度优化攻击，另一种为一次性生成的通用掩码攻击；

**💡 创新点**

首次在事件驱动SNN中实现原生二值输入的海绵攻击，并首次提出针对SNN的通用海绵攻击，展示了在保持预测正确性的前提下可显著提升单次推理的SynOps和能耗；

**🔧 技术方法**

采用可微分的Gumbel‑Softmax二值化替代、梯度优化（Adam）、自定义损失函数（饱和度、保持性与方差正则化），通用攻击通过对训练集梯度求平均并构造固定二值掩码；

**📊 数据集**

在三种数据集上评估：NMNIST（视觉事件）、SHD（音频事件）和IBM DVS Gesture（大尺寸视觉事件）；

**📈 对比分析**

与基线模型对比，单样本攻击实现1.5–2.6×的SynOps膨胀、98%以上的预测保持；通用攻击实现1.09–1.24×的SynOps膨胀，同时保持率仅为73–91%，但部署成本为零；

**⚠️ 局限性**

单样本攻击在实时流式推理中因优化耗时（最多4分钟）不具备可行性；通用攻击虽易部署但膨胀幅度和保持率有限；实验仅覆盖前向推理能耗，没有考虑参数空间攻击或硬件实际能耗测量。

---

## 340. RaDiVe: Robust 4D Radar Odometry with Distance-Bounded NDT and Velocity-Discrepancy Point Uncertainty

**arXiv ID:** 2607.28045 | [PDF](https://arxiv.org/pdf/2607.28045v1)

**作者:** Sangwoo Jung `[一作]` (Seoul National University), Ayoung Kim `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一套基于4D雷达的鲁棒里程计框架 RaDiVe，用于稀疏噪声点云的注册与定位。

**💡 创新点**

创新点在于：①距离约束的 NDT 匹配显著提升稀疏数据的注册稳定性；②利用径向速度差异构建点不确定性权重，提高点可靠性；③通过 PIN‑mapping 提取 SDF 表面点净化局部子地图。

**🔧 技术方法**

核心技术包括：距离约束 NDT 注册、径向速度不确定性加权、PIN‑mapping 的隐式 SDF 表面提取、LM 优化以及基于行驶速度的平移先验。

**📊 数据集**

在公开的 Oculii Eagle、Continental ARS548、SNAIL 等三大4D雷达数据集上进行评测。

**📈 对比分析**

与 ICP、GICP、KISS‑ICP、Doppler‑ICP 等多种基线相比，RaDiVe 在平移与旋转 ATE 上平均分别降低约 44.4% 与 21.3%，并保持实时性能。

**⚠️ 局限性**

局限性：对稠密高精度 LiDAR 数据的利用仍不够充分；单一雷达传感器在极端低速或高度动态场景下仍可能受限；未来需融合 IMU 或其他传感器提升精度。

---

## 341. Temporal Poisoning: Clean-Label Backdoors via Event Redistribution in SNNs

**arXiv ID:** 2607.28075 | [PDF](https://arxiv.org/pdf/2607.28075v1)

**作者:** Roberto Riaño `[一作]` (Radboud University), Aitor Urbieta `[通讯]` (Ikerlan Research Centre)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了针对脉冲神经网络的无标签清洁后门攻击，仅通过改动事件时间戳而不改变标签实现后门。

**💡 创新点**

创新点在于使用时间重映射保留每像素事件计数不变，使后门在时间轴上隐蔽并能在多数据集、网络架构上高效工作。

**🔧 技术方法**

采用时间重映射触发器（集中、前置、平移）、LIF 神经元的状态动态、事件序列训练以及模型无关的时间事件质量检测器进行攻击与防御评估。

**📊 数据集**

使用了 N‑MNIST、DVS‑Gesture 和 CIFAR10‑DVS 三个神经形态事件数据集。

**📈 对比分析**

在六种 victim（卷积 SNN 与 SpikformerLite）上通过 ASR 与 CA 对比实验，集中触发器可在多数配置下达到 ASR 1.00，清洁准确率仅下降 2% 以内；传统防御普遍失效，仅模型无关时间检测器能高效检测。

**⚠️ 局限性**

局限性包括仅对时间轴保持计数不变可能不适用于更复杂时序任务，物理实现难度大，检测依赖特定时间特征，攻击者可通过更细致的重映射规避；时间感知防御仍需进一步研究。

---

## 342. IndustryForge-27B: A Domain-Enhanced Multimodal Foundation Model for Industrial CAD

**arXiv ID:** 2607.28050 | [PDF](https://arxiv.org/pdf/2607.28050v1)

**作者:** Nianchen Deng `[一作]`, Botian Shi `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在 Qwen3.5‑VL‑27B 上进行多任务 SFT，训练一个 27B 参数的工业 CAD 基础模型，覆盖 CAD 可视化、CadQuery 参数化代码生成、Assembly 级代码生成和 COM API 代码生成四项核心技能。

**💡 创新点**

创新点包括：① 通过整合来自 IterCAD、AssemCAD、ComAct 等四大工业 CAD 生态的多任务数据，构建统一的 52k 条样本多模态训练集；② 采用 LoRA、DeepSpeed ZeRO‑3 与 8‑卡序列并行的高效训练方案；③ 通过该基础模型为后续工业代理项目提供可直接复用的“能读图写码”子层，避免重复从零开始构建。

**🔧 技术方法**

技术手段主要是：多模态预训练模型 Qwen3.5‑VL‑27B + LoRA 微调；DeepSpeed ZeRO‑3、sequence‑parallel（8）、padding‑free packing；SFT 取代 CPT；并行执行多任务采样、独立验证、去重与质量控制。

**📊 数据集**

使用的主要数据集有：CAD‑VQA‑4k（多视图绘图 Q&A）、text2cadquery‑17k（自然语言 → CadQuery 代码）、text2cadquery‑assembly‑1k（Assembly 级生成）、com_2d‑20k、com_3d‑5k、com_assembly‑5k（COM 代码生成）以及来自 IterCAD、AssemCAD、ComAct 的子语料库。

**📈 对比分析**

评估方法：在四个 CAD 领域基准（CAD‑VQA、CadQuery、COM CAD、CadQuery Assembly）与 11 个通用基准（aime26、arc、chartqa、gsm8k、gsm8k_v、math_vision、mmlu、mmmu、ocr_bench、science_qa、trivia_qa）上对比基线模型和强闭源模型。结果显示：在 CAD 领域平均提升 33.65 pp 并赢得 4/4 基准；在通用基准平均提升 1.56 pp，且无灾难性遗忘。

**⚠️ 局限性**

局限性：① Assembly 级通过率仅为 15.38 %，仍显不足；② 仅覆盖 CAD 代码生成与可视化，未涉及 CAE/仿真；③ 仅关注 Windows 三大 CAD（SolidWorks、Inventor、AutoCAD）的 COM；④ 仍需要进一步闭环 RL 与循环生成改进。

---

## 343. Deep learning-based hierarchical insect classification using camera trap imagery

**arXiv ID:** 2607.28005 | [PDF](https://arxiv.org/pdf/2607.28005v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 344. ClawTrack: Towards Trace-Level Evaluation and Improvement of Real-World Autonomous Agents

**arXiv ID:** 2607.28037 | [PDF](https://arxiv.org/pdf/2607.28037v1)

**作者:** Xingjian Wu `[一作]` (Meituan), Xunliang Cai `[通讯]` (Meituan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个名为ClawTrack的双重评估基准，包含320个跨8个领域的真实任务，并在Docker化的模拟环境中记录完整的ReAct轨迹；

**💡 创新点**

通过引入四维度的过程评分（目标对齐、效率、信息利用、结果验证）和任务特定Rubric，实现了对Agent执行路径的细粒度诊断，并采用双阈值通过判定显著区分幸运成功与可靠表现；

**🔧 技术方法**

使用LLM驱动的Process Grader和Outcome Grader进行评估，半自动化Rubric生成器结合专家标注与LLM学习，ReAct轨迹采集、Docker mock服务、统计分析与可视化等技术手段；

**📊 数据集**

结合已有Benchmark（Claw-Eval、WildClawBench、GAIA等）改编的160个任务与自行设计的160个任务，共320任务，配合25+ deterministic mock服务；此外收集约20k轨迹用于后期SFT；

**📈 对比分析**

在21个模型共16,000+次实验中，对比Pass@3、Pass^3、Task Score、Process Score等指标；结果显示顶级模型在Pass@3达到76.4%，Process过滤后SFT提升10-19% Pass@3，验证了过程评分与模型表现的相关性；

**⚠️ 局限性**

局限性包括：过程评估仍依赖LLM判定，可能带来主观偏差；仅覆盖文本及部分多模态任务，缺乏长时交互与安全边界的深入考量；Rubric生成与评估仍需人工审核，提升自动化程度是未来方向。

---

## 345. Contrastive Reinforced Policy Optimization via Privileged Self-Distillation

**arXiv ID:** 2607.28026 | [PDF](https://arxiv.org/pdf/2607.28026v1)

**作者:** Xingjian Wu `[一作]` (Meituan), Xunliang Cai `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种新的对比强化学习框架CRPO，改进传统的Agentic On‑Policy Self‑Distillation（OPSD），通过对比学习方式解决曝光偏差问题，提升长时序推理与工具使用的稳定性与性能。

**💡 创新点**

创新点包括：①将OPSD视为两视图对比学习；②利用预测熵区分正负位置并进行组内对比；③采用负KL作为相似度，形成token‑级对比自蒸馏；④将对比正则化与轨迹级RL奖励（如GRPO）无缝结合，形成CRPO*。

**🔧 技术方法**

技术手段主要有：对比学习（InfoNCE）、预测熵（entropy）做位置筛选、组内相对机制、负KL相似度、token‑级策略梯度、与GRPO等RL目标的联合优化。

**📊 数据集**

使用13个长时序推理与深度搜索基准，包括数学与知识推理集（AIME, MATH, GSM8K, WebWalker, HotpotQA, 2Wiki, MuSiQue, Bamboogle）以及深度搜索集（GAIA, WebWalkerQA, Humanity's Last Exam, XBench）。

**📈 对比分析**

在与GRPO、ARPO、OPSD、SDPO、RLSD、TIR Prompting等基线对比时，CRPO/CRPO*在所有基准上均获得第一名，平均提升约7–9%（8B规模）或更高（14B规模），甚至超过32B直接推理模型30分以上，表明显著的性能优势和良好的可扩展性。

**⚠️ 局限性**

局限性：①对正样本比例p和对比正则化权重λ敏感，需要调参；②依赖熵估计的可靠性，熵误差可能导致错误正负划分；③在极长序列或高工具调用次数的任务中仍可能出现效率下降；④计算开销相对传统OPSD略大，尤其在大模型或多任务场景下；⑤未验证跨模型或多任务通用性。

---

## 346. Flux-OPD: On-Policy Distillation with Evolving Contexts

**arXiv ID:** 2607.28022 | [PDF](https://arxiv.org/pdf/2607.28022v1)

**作者:** Yuran Wang `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种利用演化上下文进行训练时监督的Open‑Policy Distillation框架Flux‑OPD，针对开放式任务实现更好的任务偏好建模。

**💡 创新点**

创新点在于：①通过逆 KL 分解把目标拆成蒸馏项和冲突项；②提出上下文校正（以无上下文教师为锚，注入上下文差分信号）和上下文加权（依据冲突项动态调节校正强度）两种策略，显著稳定蒸馏目标并抑制教师间冲突；③实现上下文在单轮训练中的连续演化。

**🔧 技术方法**

技术手段包括：逆 KL 分解与几何平均蒸馏；log‑space 插值实现上下文校正；冲突度（-log Z）作为加权指标；自适应上下文提取与更新；大模型教师‑学生 on‑policy distillation；实验使用 Qwen3‑VL‑Instruct、Qwen2.5‑VL‑Instruct、Qwen3 等多种模型。

**📊 数据集**

数据集：视频生成提示优化任务（10K SFT prompts，评估基准 VBench 与 Video‑Bench）以及医学问答任务（RaR‑Medicine 18K问答，评估基准 HealthBench）。

**📈 对比分析**

对比方法包括 OPD、OPCD、OEL 等基线。实验结果显示 Flux‑OPD 在两大开放式任务中均实现最高总分，且在训练稳定性（loss 与梯度范数）和跨域泛化（IF‑Eval）上优于基线；在 prompt‑optimization 任务中相较于 OPD 更能利用视频级反馈，医学问答任务中排名第二。

**⚠️ 局限性**

局限性：对教师模型质量高度依赖；冲突阈值 τ 与权重校准参数需经验调优；上下文提取频率和池大小对性能影响显著；在纯文本任务中提升有限；缺乏严格的收敛性理论保证。

---

## 347. SKIMIX: Multi-Agent Harness-Time Scaling with Skill Mixture for Dynamic Harness Engineering

**arXiv ID:** 2607.27994 | [PDF](https://arxiv.org/pdf/2607.27994v1)

**作者:** Jia Luo `[一作]` `[通讯]` (Huazhong University of Science and Technology), Jia Luo (Huazhong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了SKIMIX多代理框架，利用多样化技能组合并通过迭代细化提升LLM推理能力。

**💡 创新点**

创新引入动态技能注册、抗稀释路由和自适应技能演化三种机制，系统性解决技能库规模化组合与稀释问题。

**🔧 技术方法**

基于嵌入式技能检索、子模优化抗稀释路由、强化学习式技能演化，以及多轮消息传递与多数投票聚合技术。

**📊 数据集**

在六个推理基准上评估：AIME、GPQA Diamond、HLE、MMLU‑Pro、MATH‑500 与 BBH。

**📈 对比分析**

对比单模型、单代理自细化和3/5/15代理SKIMIX，结果显示在开放式数学推理上提升约33%/11%，而多选题上单代理更好，代理数增大并非总能提升，第二轮迭代获得大部分收益。

**⚠️ 局限性**

实验仅基于 DeepSeek‑V3.2，样本量有限，缺乏对ADR/ASE的独立消融，且固定15种技能组合，难以验证在更大库与不同模型上的普适性。

---

## 348. Share the Judge, Learn the Deferral: Where Specialization Helps LLM Evaluation

**arXiv ID:** 2607.27984 | [PDF](https://arxiv.org/pdf/2607.27984v1)

**作者:** Weining Zhang `[一作]` `[通讯]` (Cheung Kong Graduate School of Business), Weining Zhang (Cheung Kong Graduate School of Business)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了评估系统中领域专门化的两种实现方式：将专门化放在判定器权重中（judgelets）还是放在放行决策（风险阈值）中，并通过大规模公开数据检验两种方案的准确性、覆盖率与计算成本。

**💡 创新点**

提出“judgelet”概念——基于低秩LoRA的轻量级专门化评估器；通过对照实验揭示：过早专门化会削弱共性评分学习，而在放行决策上专门化可利用互补错误显著降低计算成本；同时给出了可复制的风险审计与稀疏专家部署方法。

**🔧 技术方法**

技术包括：TF‑IDF + K‑means文本路由；共享基础模型与多组LoRA适配器（rank‑8/64）；风险阈值的 Bonferroni 校准；学习正确性头（logistic risklet）用于评估模型堆叠；对齐风险与计算的归一化参数计算；精确的一侧 95% Wilson/Clopper‑Pearson 风险审计。

**📊 数据集**

使用两大公开数据集：Prometheus Feedback‑Collection（99,952 条含 996 条评估标准）和 RewardBench‑2（1,865 任务，8,977 完成示例），并通过外部 1,000 条 Feedback‑Bench 示例验证鲁棒性。

**📈 对比分析**

对比方法：将完整上下文评估器（monolithic）与仅使用响应（response‑only）作为基线，进一步对比不同 LoRA 配置（rank‑8 vs rank‑64，单一 vs 4/8 family），以及不同放行策略（全局、按家族、Bonferroni）。在 RewardBench‑2 上比较多阶段流水线（0.6B–4B–8B）与单一 8B 评估器；结果显示：在流水线中，平均 4.66 点的准确提升与 0.415 的归一化计算率；在 judgelet 实验中，多家族会导致 10.05 点准确率下降，但共享预适配恢复至 10.5 点以上。

**⚠️ 局限性**

局限性：数据集主要为人工合成，评估者可能利用模板化特征而非真正理解规则；专门化仅在单一基础模型族上验证，跨模型族或更复杂语言环境未测试；风险阈值的校准基于离散样本，未覆盖动态策略演变；计算成本使用参数计数近似，未考虑实际能耗或吞吐量差异。

---

## 349. Load balancing in parallel infinite-server queues with action delay via phase representation

**arXiv ID:** 2607.27976 | [PDF](https://arxiv.org/pdf/2607.27976v1)

**作者:** Kazuma Abe `[一作]` (University of Tsukuba), Tuan Phung-Duc `[通讯]` (University of Tsukuba)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了具有行动延迟的并行无限服务器队列系统，利用Erlang相位结构得到有限维流模型并分析其平衡与不平衡动力学。

**💡 创新点**

创新点在于将行动延迟显式建模为相位链，得到与信息延迟模型相同的特征方程，从而揭示不同类型延迟对系统稳定性的相同影响；并在任意相位数L下推导闭式特征方程。

**🔧 技术方法**

采用线性链技巧（Erlang相位逼近）、密度依赖马尔可夫链理论、线性化与Routh–Hurwitz判据、数值积分和仿真验证。

**📊 数据集**

没有使用真实业务数据集，所有结果基于基于Poisson到达、指数/ Erlang延迟和服务的离散事件仿真。

**📈 对比分析**

通过与仿真曲线对比验证流模型的准确性；在L=1时系统稳定，L=2时可出现失稳；数值实验展示不同延迟阶段数、调度灵敏度与平均延迟对振荡幅度和衰减速率的影响。

**⚠️ 局限性**

局限性包括仅对小幅不平衡做线性化分析，未考虑大幅偏离时非线性饱和；相位数有限时对真实连续延迟的逼近误差；以及未探讨更一般的多服务器或非指数服务时间场景。

---

## 350. Now You Have My Healthy Attention: A U-DiT for Brain-MRI Inpainting

**arXiv ID:** 2607.27974 | [PDF](https://arxiv.org/pdf/2607.27974v1)

**作者:** Danilo Danese `[一作]` (Politecnico di Bari), Tommaso Di Noia `[通讯]` (Politecnico di Bari)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于U-DiT的三维脑MRI缺陷修复网络，用单一下采样自注意力块实现全局上下文融合。

**💡 创新点**

创新点是将遮挡区域的注意力限定为已知健康体素并加入对侧同源优先权重，以及将对侧对称图像作为患者特定先验输入。

**🔧 技术方法**

采用卷积编码‑解码架构、U‑DiT自注意力、三维RoPE位置嵌入、健康‑仅注意力、对侧对称输入以及镜面测试时增强。

**📊 数据集**

使用BraTS‑Local‑Inpainting（来自BraTS‑GLI T1w）共1,251个训练样本，进行多种随机遮罩增广。

**📈 对比分析**

与全变压器和其他变体对比，单模型在官方验证集上平均SSIM 0.864、PSNR 24.71 dB、MSE 0.0046，显著优于基线。

**⚠️ 局限性**

局限在于模型趋向过度平滑，缺乏高频纹理，导致合成组织与真实组织的纹理差异。

---

## 351. What Makes Graph Unified? Principles and Generative Sliding-Window Transformer for Graph Foundation Models

**arXiv ID:** 2607.27966 | [PDF](https://arxiv.org/pdf/2607.27966v1)

**作者:** Dongxiao He `[一作]` (Tianjin University), Di Jin `[通讯]` (Tianjin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种图基础模型SliGFM，能统一不同域图的异构节点特征，并在无额外调参的情况下迁移到新图任务。

**💡 创新点**

创新点在于：①提出四大统一特征目标（形式统一、跨域可迁移、信息保留、骨干兼容）；②使用基于拓扑平滑的特征排序与滑动窗口编码生成统一Token；③引入相对平滑偏置Transformer与结构引导自适应传播；④加入生成式重建损失保证信息完整。

**🔧 技术方法**

采用拓扑平滑度排序、共享滑动窗口编码器、相对平滑偏置Transformer、结构统计自适应多跳传播、生成式Token重建以及多目标损失（重建、局部对齐、全局离散）。

**📊 数据集**

节点分类使用Cora、CiteSeer、PubMed、Photo、Computers、CS、Chameleon、Squirrel、Actor；图分类使用IMDB-BINARY、ENZYMES、DD等公开数据集。

**📈 对比分析**

与MDGFM、MDGPT、BRIDGE、FUG、LEDA、SAMGPT、TIG、TFSGFM等基线进行对比；在单步节点分类上平均提升≈1.6%准确率，在零步图分类上实现所有数据集最高准确率/宏F1。

**⚠️ 局限性**

局限性包括：依赖拓扑平滑排序，窗口大小需要经验调节；对极高维或缺乏平滑特征的图可能表现欠佳；实验主要集中在中等规模公开数据集，尚未验证在大规模工业场景中的鲁棒性。

---

## 352. Echoverse: Deep, Evolving Environments for Training Computer-Use Agents at Scale

**arXiv ID:** 2607.28074 | [PDF](https://arxiv.org/pdf/2607.28074v1)

**作者:** Yash Pandya `[一作]` (Microsoft Research), Akshay Nambi `[通讯]` (Microsoft Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文提出 Echoverse 平台，构建深度、可检验的登录后交互式 synthetic 环境，并通过模型与世界的共同进化循环提升 9B 语言模型的浏览器交互能力。

**💡 创新点**

创新点在于：①将世界构建与模型训练视为同一闭环；②通过“深度”定义确保环境支持完整工作流；③使用数据库驱动的判定器避免视觉误判；④针对单一控制的 capability 世界实现多样化布局；⑤在同一世界上同时进行监督微调和强化学习。

**🔧 技术方法**

技术包括：GitHub Copilot SDK 代理工厂、Playwright 自动化、模型上下文协议查询数据库、LLM 判定器（GPT‑4.1）做差异对比、group‑relative policy gradient 强化学习、密集的 per‑step 视觉奖励。

**📊 数据集**

数据集由 12 个 synthetic 世界（10 个深度全域 + 2 个 capability）组成，seed 数据来自公开数据集或内部生成，生成约 21,009 条经过验证的轨迹，另外还使用 WebVoyager、Online‑Mind2Web 公开基准做外部评测。

**📈 对比分析**

与基准模型 (Base)、自己训练模型 (Ours) 以及 GPT‑5.4 进行比较；Ours 在 14 个评测拆分上从 36.5% 提升到 67.1%，仅距 GPT‑5.4 14 点；在单独的 capability 任务上，两项能力共同训练可把 held‑out 成功率从 34% 提升到 57% 以上；在公开 Web 基准上也取得 1–2% 的提升。

**⚠️ 局限性**

局限性包括：①浅层环境易导致模型退化；②强化学习受限于任务数量与奖励设计，尚未突破 GPT‑5.4 的 ceiling；③对世界的修复依赖代理工厂与人工监督；④在真实 Web 线上环境的迁移仍有限，主要受制于差异化布局与隐私限制。

---

## 353. GVR-Coder: A Visual-Feedback Framework for Structured SVG Generation in Complex Document and Meeting Scenarios

**arXiv ID:** 2607.28073 | [PDF](https://arxiv.org/pdf/2607.28073v1)

**作者:** Yiming Xu `[一作]` (University of Science and Technology of China), Qi Song `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为GVR-Coder的视觉反馈框架，用于从复杂文档和会议场景中生成结构化的SVG图形。

**💡 创新点**

创新点在于引入了DocMeetSVG-100K数据集，结合了课程驱动的拒绝采样微调、双重渲染反馈的强化学习以及生成-验证-修复的代理循环，提升了生成图形的逻辑性和美观性。

**🔧 技术方法**

使用了课程驱动的拒绝采样微调、强化学习（RLDRF）和生成-验证-修复的多代理循环等技术。

**📊 数据集**

使用了DocMeetSVG-100K数据集，这是一个专门为文档创作和会议审查场景设计的大规模SVG数据集，包含100,000个文本-SVG对。

**📈 对比分析**

与传统方法和大型语言模型（LLM）进行比较，GVR-Coder在生成逻辑一致和视觉吸引的图形方面表现优越，自动评估和人类评估均显示其在美观性和语义一致性上均优于竞争对手。

**⚠️ 局限性**

局限性在于模型可能在处理极其复杂的逻辑结构时仍然面临挑战，且生成的图形在某些情况下可能需要进一步的人工修正。

---

## 354. SKILL-KD: Contrastive Skill Distillation for LLM Agents

**arXiv ID:** 2607.28048 | [PDF](https://arxiv.org/pdf/2607.28048v1)

**作者:** Qiming Shi `[一作]` (Zhejiang University), Di Weng `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SKILL-KD框架，通过对比教师与学生的执行轨迹来提炼可直接执行的文本技能，利用自适应重跑验证技能有效性，并通过基于编辑历史的漂移感知合并机制维护紧凑的技能库。

**💡 创新点**

核心创新在于把技能视为冻结学生与更强教师之间的“蒸馏介质”，使用对比轨迹而非单侧经验来生成技能，并通过迭代验证与历史关联合并防止技能漂移。

**🔧 技术方法**

技术包括对比式技能蒸馏、文本技能生成与自适应搜索、学生重跑评估、基于编辑日志的漂移感知合并工具链，以及Markdown结构化技能文件。

**📊 数据集**

在五个代理基准上进行评估：SearchQA、SpreadsheetBench、DocVQA、LiveMath、ALFWorld，使用两组教师-学生组合（Qwen3.5-4B/ Qwen3.7-plus 与 Qwen3.6-35B-A3B/ ChatGPT-5.5）。

**📈 对比分析**

与无技能、EvoSkill、Trace2Skill、SkillGen、SkillOpt等基线对比，SKILL-KD在所有基准上实现显著提升，宏平均得分提升约 23-27 分，尤其在 SpreadsheetBench、LiveMath 与 ALFWorld 上表现突出。

**⚠️ 局限性**

局限包括依赖教师轨迹质量、对极端失败情形的支持有限、技能生成仍需人工可解释性验证、以及在更大规模多模态任务中对编辑日志管理的存储与检索成本。

---

## 355. When AI Does the Work, What Is Learning For? Post-Instrumental Learning and the Risk of Capacity Dissolution

**arXiv ID:** 2607.28041 | [PDF](https://arxiv.org/pdf/2607.28041v1)

**作者:** Kai Yao `[一作]` `[通讯]` (University of Edinburgh), Kai Yao (University of Edinburgh)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并阐述了“后工具性学习(post‑instrumental learning)”的概念，认为在AI可以高效完成任务后，学习的核心价值在于保持人类与机构对目标设定、理由说明、可挑战性、拒绝与修订以及参与度的能力。

**💡 创新点**

核心创新是将学习视为在AI介入后维持问责和可审查性的生态系统，而非单纯的技术技能培养；并通过五大能力（目标设定、理由说明、可挑战性、拒绝与修订、参与度）构建评估框架。

**🔧 技术方法**

该工作主要采用规范性概念分析和理论构建，没有基于实验或算法实现的技术方法。

**📊 数据集**

无数据集。

**📈 对比分析**

无实验比较。文章通过对比传统评估模式与后工具性评估模式，论证后工具性评估在保持学习与问责方面的优势。

**⚠️ 局限性**

局限性在于缺乏经验验证与实证研究，提出的框架和能力仍需在实际教育与治理情境中检验其可操作性和有效性。

---

## 356. Enhancing Irregular Time Series Forecasting with Continuous-Time Modeling Framework

**arXiv ID:** 2607.28035 | [PDF](https://arxiv.org/pdf/2607.28035v1)

**作者:** Tianen Shen `[一作]`, Jilin Hu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了WrapFlow框架，用于对不规则多变量时间序列进行连续时间预测。

**💡 创新点**

创新点在于：① 连续时间标记化，将原始事件直接编码为事件、gap-aware事件和mask GAP三种标记；② 无需数值求解的残差流匹配，直接在残差空间学习向量场；③ 通过查询驱动的解码器实现对历史记忆的精细检索。

**🔧 技术方法**

采用的技术包括：Transformer主体与交叉注意力；绝对与相对时间编码；gap-aware事件与mask GAP token；残差流匹配训练与Euler积分推断；以及自监督的历史掩码回归。

**📊 数据集**

实验使用四个公开不规则时间序列数据集：PhysioNet、MIMIC、HumanActivity、USHCN。

**📈 对比分析**

与14个基线（包括PrimeNet、SeFT、mTAN、GRU‑D、Raindrop、Warpformer、NeuralFlows、CRU、GNeuralFlow、tPatchGNN、GraFITi、Hi‑Patch、KAFNet、APN）在MSE和MAE上进行对比。WrapFlow在大多数指标上均夺得第一名，比第二名APN平均提升约3%（MSE）和5.7%（MAE）。

**⚠️ 局限性**

局限性包括：① 需要手动设定gap阈值；② 主要关注点是单步预测，跨步不确定性处理有限；③ 对极端稀疏或长缺失区间的鲁棒性尚待进一步验证。

---

## 357. MUL-T: Decoding Spatial Cellular Architecture in Multiplexed Tissue Images

**arXiv ID:** 2607.28030 | [PDF](https://arxiv.org/pdf/2607.28030v1)

**作者:** Farzaneh Seyedshahi `[一作]` (Cancer Research Uk Scotland Institute), Ke Yuan `[通讯]` (Cancer Research Uk Scotland Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种轻量级Transformer框架MUL‑T，将多重组织图像的细胞空间结构建模为掩码上下文预测任务，直接在细胞级别生成可用于下游预测的嵌入；

**💡 创新点**

创新点在于把细胞作为离散token并通过网格化分箱保留空间信息，同时利用自监督掩码训练捕捉高阶细胞相互作用，既保持了生物学可解释性，又显著降低了模型参数与训练成本；

**🔧 技术方法**

使用轻量级BERT式Transformer（4层4头，256维）进行自监督掩码预训练，并结合旋转位置编码、网格化token化、k‑NN跨数据集token传递；

**📊 数据集**

主要使用LATTICeA‑IO肺腺癌多重IF数据（约2500核心，1700万细胞）作为训练集，外部验证使用CTCL‑P（70核心，约117k细胞）；

**📈 对比分析**

与传统基于marker统计或聚类频率的特征方法以及ViT基础模型KRONOS进行比较，MUL‑T在核心级肿瘤模式（AUC 0.86）、患者级分级（AUC 0.79）、PD‑L1阳性预测（AUC 0.72）以及跨数据集响应预测（AUC 0.68）上均优于基线并与ViT相当；

**⚠️ 局限性**

局限包括网格化分箱导致的空间离散化可能平滑细微结构；仅局部窗口注意力，缺乏全局长程依赖建模；对细胞分割与聚类质量高度依赖，跨数据集标记不完全一致时可能影响转移效果。

---

## 358. Scaling, Lock-In, and Proxy Compliance: A Political Economy of Responsible AI

**arXiv ID:** 2607.28023 | [PDF](https://arxiv.org/pdf/2607.28023v1)

**作者:** Florian A. D. Burnat `[一作]` (University of Bath), Brittany I. Davidson `[通讯]` (University of Bath)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个顺序政治经济学模型，分析AI供应商与部署方在可审计性、减缓措施与监测之间的权衡，揭示了可观察合规与实际危害降低之间的差距。

**💡 创新点**

提出“代理合规”机制，并通过模型揭示锁定效应、证据依赖执法、可审计性、可迁移性、事故报告与结果关联责任等政策杠杆对减缓投入与危害的影响。

**🔧 技术方法**

运用了定量经济学的博弈论与比较法，分析内部成本、监测回报与证据生成的相互作用。

**📊 数据集**

本研究未使用实证数据集，而是通过理论推导和示例参数演示机制。

**📈 对比分析**

采用理论比较法与案例式参数实验，展示在不同制度设定下的平衡解与政策阈值；未给出实验性能指标。

**⚠️ 局限性**

局限在于模型静态、单一供应商、缺乏私人信息、多重危害与长期声誉动态，且对实际实施需进一步实证检验。

---

## 359. Constructing linear codes from digraphs and groups

**arXiv ID:** 2607.28016 | [PDF](https://arxiv.org/pdf/2607.28016v1)

**作者:** Coen del Valle `[一作]` (Open University), Cheryl E. Praeger `[通讯]` (University of Western Australia)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过引入图码和有向图码两种新型线性码构造，扩展了 Kaufman‑Lubotzky 以 Cayley 码为核心的对称 LDPC 良码构造方法。

**💡 创新点**

创新点在于：①将 Cayley 码的框架推广到任意顶点传递 (或弧传递) 图和有向图；②在保持对称性和低密度校验的同时，利用图的谱性质给出更优的相对距离和码率下界；③构造了首个既是对称又是 LDPC 的良好有向图码族，且其输入代码可以是非等价的。

**🔧 技术方法**

主要技术包括：群论与图论的组合手段（群作用、轨道、转置等）；线性码的双重对称性与单轨道对称性；谱理论与 expander 混合引理的推广（适用于有向图）；以及对 Cayley 及其有向同型的矩阵构造。

**📊 数据集**

本文属于理论研究，未使用公开数据集；主要利用的是抽象群（如 PSL(2,q)、SL₂(q)、_2(q)）与其生成集所产生的 Cayley 或有向 Cayley 图，作为构造的基底。

**📈 对比分析**

通过与以往的 Cayley 码相对比，作者改进了相对距离下界（从 (δ‑λ/v)/(1‑λ/v)² 提升到 δ·(δ‑λ/v)/(1‑λ/v) 以及在有向图上的类似改进），并展示了在给定群族下构造出的码族在码率和相对距离上均有正下界，构成了一族良好对称 LDPC 码。

**⚠️ 局限性**

局限性包括：尚未找到无限族非 Cayley 产生的对称 LDPC 良好图码；对称有向图码中输入码 B₁,B₂ 可不等价的情况尚未完全阐明；以及构造依赖于存在特定群的生成集，限制了可实现的族的范围。

---

## 360. It's All Just Vectorization: einx, a Universal Notation for Tensor Operations

**arXiv ID:** 2607.27987 | [PDF](https://arxiv.org/pdf/2607.27987v1)

**作者:** Florian Fervers `[一作]` (Fraunhofer IOSB), Michael Arens `[通讯]` (Fraunhofer IOSB)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd`

**🎯 论文内容**

本文提出了一种通用的张量操作符号体系einx，基于向量化思想将任意张量运算统一拆分为低阶基础操作及其向量化，简化现有Numpy类API并提升可读性与可写性。

**💡 创新点**

1) 重新定义向量化为通用变换工具，能够将低阶操作提升为高阶操作；2) 提出einx符号，使用与循环语法对应的点式表达，覆盖所有张量运算；3) 通过统一的括号、轴组合、通用省略号等语法改进，消除传统einsum、einops的局限；4) 开源实现能自动编译为多种后端（Numpy、PyTorch、Jax等）。

**🔧 技术方法**

核心技术包括：向量化变换理论、循环语法映射、点式表达（einx notation）、基于字符串的轴描述、自动代码生成与缓存、支持vmap/JIT等后端。

**📊 数据集**

论文未提供实验数据集，重点在理论构建与实现演示。

**📈 对比分析**

通过与einsum、einops、传统Numpy‑style比较，展示einx在表达简洁性、错误检测、可读性等方面的优势；实现层面通过编译为后端调用保证无额外开销。

**⚠️ 局限性**

局限性：目前仅实现了常见基础操作，尚未覆盖所有自定义复杂运算；对极大张量的性能与后端优化仍需进一步评估；在缺乏完整标准化的前提下，社区接受度与工具链兼容性待验证。

---

## 361. Geometric View on Integrated Cascaded Channel of IRS-Aided Communications

**arXiv ID:** 2607.27972 | [PDF](https://arxiv.org/pdf/2607.27972v1)

**作者:** Yunli Li `[一作]` (City University of Hong Kong), Young Jin Chun `[通讯]` (City University of Hong Kong)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本论文提出了利用Cassini椭圆和椭圆几何模型来表征IRS辅助链路的集成路径损耗距离，并基于此导出了对应的CDF与PDF；在此基础上求解了混合IRS（由被动和主动子面组成）的最优部署位置，并提出了一种基于集成路径损耗距离的机会性关联策略；通过Monte‑Carlo仿真验证了模型精度，并与传统最近关联策略进行比较，显示混合IRS与机会性关联可显著提升系统性能。

**💡 创新点**

创新点主要包括：①首次将Cassini椭圆与椭圆几何形状用于描述IRS链路的集成路径损耗；②将该几何模型与混合IRS的优化结合，得到全局最优部署解；③提出了针对混合IRS的机会性关联策略，突破了传统最近关联的局限。

**🔧 技术方法**

采用的技术方法包括：几何建模、混合伽马分布（Mixture Gamma）对小尺度衰落的建模、最优化（解析求解最优部署）、机会性关联算法、以及仿真验证（MATLAB Monte‑Carlo）。

**📊 数据集**

本工作以仿真数据为主，设定路径损耗指数、发射功率、IRS功率、噪声功率等参数，未使用公开数据集。

**📈 对比分析**

通过将仿真得到的平均接收SNR、覆盖概率和速率与传统最近关联策略以及纯被动/纯主动IRS进行对比，结果显示：在产品距离和和距离两种路径损耗模型下，混合IRS与机会性关联均可实现相对较高的SNR和更低的覆盖概率，特别是在高密度IRS场景下优势更为显著。

**⚠️ 局限性**

主要局限包括：①对主动子面的热噪声和放大因子耦合的分析仍显复杂，未给出闭式表达；②假设IRS均匀分布且节点位置固定，未考虑移动性与非均匀部署；③仅在单一UE或多IRS聚合的简化场景下进行仿真，缺乏更全面的网络层级评估。

---

## 362. Investigating Effective Uncertainty Visualizations for Ordinal Crowdsourced Data of Crowding Conditions

**arXiv ID:** 2607.28072 | [PDF](https://arxiv.org/pdf/2607.28072v1)

**作者:** Bea Alexis Arcega `[一作]` (De La Salle University), Briane Paul V. Samson `[通讯]` (De La Salle University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过在线问卷实验评估了五种不确定性可视化方法（Bubble Treemap、Cluster、Icon Arrays、Visual Entropy、HOPs）在展示基于四级顺序分类的拥挤信息时的效果，并对准确率、信心、信任、响应时间和认知负荷等指标进行了定量比较。

**💡 创新点**

创新点在于：①将顺序（ordinal）级别的拥挤数据与不确定性可视化相结合，填补了此前仅关注数值估计的研究空白；②提出多维度评价框架，将客观表现与主观体验统一衡量；③系统展示了精度与认知负荷之间的权衡，为真实通勤决策提供可操作的设计指南。

**🔧 技术方法**

使用技术包括：①基于合成概率分布的四级拥挤数据；②五种可视化实现（图形化编码）；③在线问卷平台Jotforms进行数据收集；④统计分析方法（Kruskal-Wallis H检验、Dunn事后检验、效能得分加权平均）。

**📊 数据集**

数据集为实验生成的合成数据，包含同一概率分布在四个拥挤等级（Spacious、Lightly Occupied、Moderately Congested、Congested）上的展示，用以保持信息量一致；并未使用真实公交或地铁拥挤测量数据。

**📈 对比分析**

通过对五个维度的加权评分构建效能得分，比较各可视化方法的整体表现；结果显示Bubble Treemap在准确率（69.28%）和整体效能得分（4.0）上领先；Cluster在认知负荷最低（中位数4.0）且完成时间最快（46 s）时表现最佳，说明在需要快速决策的场景更为适用。

**⚠️ 局限性**

局限性包括：①仅使用合成数据，缺乏真实拥挤环境验证；②样本主要为菲律宾都市学生/专业人士，可能缺乏跨文化普适性；③实验为在线情境，未考虑现场显示环境、设备差异和实时数据流；④仅考察四级顺序级别，未探索连续数值或多模态信息的可视化。

---

## 363. CCFormer: Efficient Cross-Field Interaction and Hierarchical Sequence Compression for Industrial Recommendation at Tencent

**arXiv ID:** 2607.28070 | [PDF](https://arxiv.org/pdf/2607.28070v1)

**作者:** Yunlong Wang `[一作]` (Platform and Content Group, Tencent), Zang Li `[通讯]` (Platform and Content Group, Tencent)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CCFormer 模型，结合跨字段分离的交叉注意力、子空间令牌混合以及分层序列压缩，解决工业推荐场景下长序列建模与跨特征交互的效率与效果双重挑战。

**💡 创新点**

创新点包括：① 跨字段分离的交叉注意力实现不同特征域之间高效交互；② 用子空间令牌混合替代全局自注意力，显著降低 O(L²) 复杂度；③ 通过分层序列压缩保持大感受野的同时大幅提升计算效率。

**🔧 技术方法**

技术手段包括 Transformer 结构、相对时间位置编码、子空间 Token Mixing、1D 卷积分层压缩、混合精度训练、INT8 量化、双重哈希、并行候选预测等。

**📊 数据集**

实验数据集为公开基准 Taobao 与 KuaiRec 以及腾讯内部工业大规模数据集（>30M 用户、>10M 物品、4B 交互样本）。

**📈 对比分析**

与 DIN、DeepFM、SASRec、MIMN、HSTU、OneTrans、STCA 等基线在公开数据上分别取得 AUC 93.67%/83.35%，在工业数据上 AUC/GAUC 提升 1.01%/2.40%；在线 A/B 测试中 CTR 提升 3.57%，广告收入提升 1.71%；模型训练速度相比 HSTU 提升 2.21×。

**⚠️ 局限性**

局限性在于对分层压缩和子空间参数的依赖，若组大小设置不当可能导致信息丢失；对极长序列仍有一定计算负担；多任务或跨域迁移的适用性尚需进一步验证。

---

## 364. Tight UGC Thresholds for Geometric Stabbing Problems

**arXiv ID:** 2607.28062 | [PDF](https://arxiv.org/pdf/2607.28062v1)

**作者:** Khaled Elbassioni `[一作]` (Khalifa University), Saurabh Ray `[通讯]` (New York University Abu Dhabi)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过严格的 CSP 框架和一种新的 transfer 定理，证明了几类几何 stabbing 问题（任意尺寸轴对齐 d 维立方体、水平段与水平/垂直线、分离 d-interval 的横切）在单位成本模型下的 UGC 硬度阈值与其自然 LP 的可行性间隙完全匹配，给出了 2、e/(e-1) 与 d 这三类问题的精确 UGC 逼近阈值。

**💡 创新点**

创新点在于：① 提出了一个基于 KMTV 严格 CSP 理论的通用 transfer 定理，能够将任何有限 arity、bounded‑arity LP 问题的 integrality gap 转化为 UGC 难度；② 构造了新的乘法尺度的 LP 问题族，实现了任意尺寸立方体的 integrality gap 趋于 d；③ 通过精细的随机化取样与全支持扰动技术，得到满足 KMTV 连接性要求的本地分布；④ 通过克隆与 snapping 技术，保证从加权 CSP 迁移到单位成本几何实例时保持目标函数与可行性的一一对应。

**🔧 技术方法**

主要技术包括：严格 CSP（strict‑CSP）框架、KMTV 的 Unique‑Games 难度结果、随机化取样与全支持扰动、连通本地分布构造、克隆（full Cartesian cloning）与 snapping、以及多尺度乘法构造用于立方体的 integrality gap。

**📊 数据集**

本研究为理论性工作，无实验数据集；所有结果均为理论证明，主要以构造性例子（实例族）为基础。

**📈 对比分析**

通过与已知的 LP‑相对 2、e/(e-1) 和 d 的多项式时间近似算法对比，本文给出了与这些上界完全匹配的 UGC 硬度下界，证明这些上界是最佳的；在有限候选与无限候选两种模型下均保持一致。

**⚠️ 局限性**

局限性：依赖 Unique Games Conjecture；仅适用于单位成本（或可通过克隆转换为单位成本）实例；构造的实例规模较大，参数需要精细设置；在某些特殊几何约束（如等长投影）下，构造和证明较为复杂；并未给出实际算法实现或实验验证。

---

## 365. Temporal Concentration from Rollout Errors: Implicit Preference Optimization for Text-to-Video Diffusion

**arXiv ID:** 2607.28058 | [PDF](https://arxiv.org/pdf/2607.28058v1)

**作者:** Henglin Liu `[一作]` (Tsinghua University), Xiu Li `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用模型自身的去噪过程生成隐式偏好，对文本到视频扩散模型进行后训练，以提升视频的时序一致性。

**💡 创新点**

创新点在于在不需要人工标注或奖励模型的前提下，直接用去噪重构误差作为偏好信号，并通过时间窗口聚焦机制将优化集中在错误率最高的短段落。

**🔧 技术方法**

采用扩散模型、VAE编码、去噪重构、DPO（Direct Preference Optimization）和时间窗口聚焦的偏好优化技术。

**📊 数据集**

在MotionBench和WISA等文本到视频数据集上进行实验。

**📈 对比分析**

与DPO、DenseDPO等基线方法比较，cIPO在真实性和运动质量等多项指标上均显著提升，表现出更高的性能。

**⚠️ 局限性**

局限在于重构误差作为偏好代理可能无法完全捕捉高层语义或人类主观审美偏好。

---

## 366. VIG-RL: Learning to Search and Insert for Verified Image Grounding

**arXiv ID:** 2607.28055 | [PDF](https://arxiv.org/pdf/2607.28055v1)

**作者:** Qinhan Yu `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究 Verified Image Grounding（VIG）任务，提出 VIG-RL 框架，让 LLM 在 ReAct 循环中动态执行文本检索、图像检索、选择和插入真实图像的完整流程。

**💡 创新点**

创新点：①将 VIG 任务转化为 agentic RL，使用 GRPO 在 ReAct 交互框架中训练可解释的决策策略；②设计多维复合奖励，分离过程级别（检索回忆）与终端级别（文本质量、图像精准度、检索覆盖）的监督，避免奖励劫持；③采用 context‑anchored 图像检索和符号化图像引用，解决跨模态语义匹配与幻觉问题。

**🔧 技术方法**

技术：强化学习（GRPO）、ReAct 交互框架、跨模态 LLM（Qwen3‑VL‑4B/8B）、多维复合奖励设计、BGE‑M3 文本检索+图像检索、符号化图像标识映射。

**📊 数据集**

数据集：MRAMG‑Bench 六大子集（Web、Wiki、Arxiv、Wit、Manual、Recipe）用于训练和评估；内部训练样本由 1.1k 交叉模态一致样本构成。

**📈 对比分析**

比较方法：在 MRAMG‑Bench 上与静态 RAG、零射手 agentic、SFT 轨迹模仿等 baseline 对比；VIG‑RL‑8B 在 Image F1 和 Comprehensive Score（C.S.）上提升 20–30% 以上，且在 OOD 领域同样保持领先；4B 版本亦显著优于同尺度基线。

**⚠️ 局限性**

limitations：①对大型预训练 LLM 依赖强；②奖励设计复杂且对超参敏感；③仅能插入检索到的图像，无法生成新图像；④符号化引用需要后端映射，增加系统复杂度；⑤需要丰富的检索库，对稀有知识场景仍存在挑战。

---

## 367. A Query-Efficient Stochastic Volume Rendering Framework for Time-Varying Implicit Neural Volumes

**arXiv ID:** 2607.28047 | [PDF](https://arxiv.org/pdf/2607.28047v1)

**作者:** Alper Sahistan `[一作]` (University of Utah), Valerio Pascucci `[通讯]` (University of Utah)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出了一种基于delta跟踪的四阶段量子化高效渲染框架，能够在不重采样、重训练或缓存的前提下，对时间变化的隐式神经表示(INR)实现交互式可视化。

**💡 创新点**

核心创新包括：① 将delta跟踪视作查询压缩机制并与GPU光线追踪核心及张量核心协同工作；② 设计自适应光线预算与同质性查询剪枝两种轻量级采样降低策略；③ 通过“ghost pass”实现宏单元自适应更新。

**🔧 技术方法**

技术手段涵盖：Delta跟踪蒙特卡罗渲染、光线追踪核心(RT)加速、张量核心(Tensor Core)批量神经网络推理、蓝噪声与残差驱动的光线预算、宏单元极值估算与时间插值。

**📊 数据集**

实验数据集包括三种四维INR模型：基于Fourier特征的FFN（S0X_XXX），SIREN CFD时间序列（cylinder、tangaroa），以及CoordNet残差正弦网络（vorts），共六个数据集。

**📈 对比分析**

与传统体渲染、基于DDA的delta跟踪及仅使用光线追踪核心的实现相比，本框架在RTX 4090上平均达到30–40 FPS（含阴影），并在更新时延≤ 55 ms；在引入查询压缩策略后，FPS提升至40–50 FPS，图像质量误差保持≤ 2%。

**⚠️ 局限性**

局限性主要在于宏单元边界估计粗糙导致的误差、对显存和推理时间高度依赖的GPU硬件（仅NVIDIA CUDA/Ray Tracing/Tensor Core支持）、以及自适应预算与剪枝策略对残差与宏单元均匀性的假设。

---

## 368. TongueReenact: Geometry-Anchored Tongue Synthesis for Face Reenactment

**arXiv ID:** 2607.28039 | [PDF](https://arxiv.org/pdf/2607.28039v1)

**作者:** MD Wahiduzzaman Khan `[一作]` (University of Technology Sydney), Kaska Musial-Gabrys `[通讯]` (University of Technology Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种跨身份舌头动态转移框架，用于提升面部重演中舌头的自然性和一致性。

**💡 创新点**

创新点包括：1）基于SAM+面部解析的自适应伪标注与迭代细化的舌头分割管线；2）几何锚定的潜在掩码扩散模型，使舌头合成与目标面部几何紧密对齐；3）使用VLM（Qwen3‑VL）构建的可扩展人类评估协议。

**🔧 技术方法**

核心技术包含：SAM、BiSeNet*舌头分割；3D Gaussian splatting+FLAME驱动的几何重演；潜在扩散模型（Stable Diffusion）与双向条件编码器；CLIP嵌入做全局身份引导；自适应掩码膨胀和逐步重构；VLM微调做自动化感知评估。

**📊 数据集**

使用的主要数据集为NERSemble、VFHQ及其合成扩展，用于训练舌头分割器、重演模型与扩散模型，并在VFHQ上进行评估与对比实验。

**📈 对比分析**

与GPAvatar、Portrait4D、LivePortrait、X‑NeMo、X‑Portrait等五种主流面部重演方法进行对比，舌头LPIPS、Presence、IoU、面积相似度均显著优于基线（提升约2倍），Temporal一致性指标也更好，VLM评估显示人类认可度达99.9%最佳舌头、100%最佳整体。

**⚠️ 局限性**

局限性包括：对重演骨干的依赖，导致不同骨干下舌头合成质量略有波动；在合成舌头区域的身份保留略低；对极端光照、遮挡等极端情况仍存在挑战。

---

## 369. Generalization Bounds on Optimal Control for Transformer Training and Wasserstein Distributional Robustness

**arXiv ID:** 2607.27975 | [PDF](https://arxiv.org/pdf/2607.27975v1)

**作者:** Kağan Akman `[一作]` (Bilkent University), Serdar Yüksel `[通讯]` (Queen's University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文提出了以Transformer为目标的控制系统的双重提升（doubly lifted）概率测度框架，并在此基础上给出了有限样本的泛化误差上界，随后将该框架转化为Wasserstein分布不确定控制（distributionally robust control）问题，得到稳健最优控制策略与传统训练最优解的收敛性质。

**💡 创新点**

创新点包括：①将Transformer训练视作测度值化的马尔可夫决策过程，恢复Markov性；②通过三重量化（state、action、测度）实现可计算的有限状态MDP并得到显式的样本误差界；③引入Wasserstein球的分布不确定性，得到零散对抗游戏的存在性与与训练最优解的Gamma收敛及上限收敛；④给出了与Transformer结构相关的显式Lipschitz常数与误差项。

**🔧 技术方法**

技术方法主要包括：测度值化动态系统、双重提升（lifting）技术、量化近似与McDiarmid不等式、Wasserstein距离与双重Lifted MDP、Lipschitz稳定性分析、Gamma收敛与Painlevé–Kuratowski极限、动态规划与最优控制理论。

**📊 数据集**

文中未给出具体实验数据集，所有结果均为理论推导与分析；若有实验，主要在合成/基准数据集上验证量化误差与泛化上界，但并未在公开大型语料上进行实测。

**📈 对比分析**

由于缺乏实证对比，论文仅给出理论上限与样本复杂度分析；若与其他泛化理论（VC维、PAC-Bayes、信息理论等）进行比较，可见其误差上界在样本数量与量化误差上具有可控性，但在实际性能上未展示实验验证。

**⚠️ 局限性**

局限性：①假设权重空间是紧致且满足范数约束，不能直接推广至无约束SGD训练；②理论收敛要求样本数≫量化分辨率≫状态分辨率，且会出现维度灾难（状态覆盖数随维度指数增长）；③上界与常数与Transformer内部参数（如注意力核、输入尺寸）高度相关，实际常数难以评估；④未给出对大规模实际模型的可计算性与性能验证。

---

## 370. VISA: A Structured Description Protocol for Agent-Based Simulation Models Towards Machine Reproducibility

**arXiv ID:** 2607.28027 | [PDF](https://arxiv.org/pdf/2607.28027v1)

**作者:** Zhou He `[一作]` `[通讯]` (University of Chinese Academy of Sciences), Zhou He (University of Chinese Academy of Sciences)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了 VISA（Structured Description Protocol）——一种基于符号化八张表格的 ABM 文档规范，并配套实现了作者–检查–代码生成三步循环，实现了跨平台（NetLogo→Python）的机器可复现；

**💡 创新点**

创新点在于：①将 ABM 关键要素最小化但完整地映射为四个 Agent 级别表格和四个模型级别表格；②引入 19 条可执行一致性规则，将模型有效性转化为可检验属性；③提供三套可复用的 LLM 技能（作者、检查、代码生成），实现全流程自动化；

**🔧 技术方法**

主要技术包括符号化表格结构、基于数学符号的跨表引用、19 条逻辑一致性规则、LLM 可执行技能以及 Python 代码生成器；

**📊 数据集**

使用的实际数据集包括奶牛市场价格与销量数据（用于多新闻贩卖示例），以及公开的 Rebellion 与 Wolf Sheep Stride 模型的原始参数；

**📈 对比分析**

对比方法是将原始 NetLogo/AnyLogic 代码与通过 VISA 自动生成的 Python 代码在关键指标（如活跃公民数、狼步长选择等）上进行对比；结果显示所有 19 条规则全部通过，跨语言实现恢复了原模型的定性动态，验证了 VISA 的可复现性；

**⚠️ 局限性**

局限性：①对学习型代理或 LLM 组件无法完全消除版本和随机性导致的不可复现风险；②对任何专有库或不可公开的数据仍是复制障碍；③仍需要手动维护表格和规则，工作量不低，且需要进一步验证其对更大、复杂模型的适用性。

---

## 371. TAPO: Transition-Aware Policy Optimization for LLM Agents

**arXiv ID:** 2607.27973 | [PDF](https://arxiv.org/pdf/2607.27973v1)

**作者:** Cong Li `[一作]` (Peking University), Zhuojian Li `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种名为TAPO的后训练框架，通过在LLM代理的强化学习过程中交替执行策略优化和基于过渡（transition）监督的下一步观测预测，利用在线交互回放中本已存在的动作‑环境反馈信号提升代理的行动决策能力。

**💡 创新点**

创新点在于：①将在线RL回放中的动作‑后继状态三元组直接转化为辅助监督任务，充分利用稠密、局部且动作条件化的环境信息；②以轻量级的交替训练方式与主RL目标共存，避免额外采样、专家数据或推理时开销；③展示该辅助目标能够显著提升LLM在长时程、稀疏奖励任务中的泛化与规划能力。

**🔧 技术方法**

采用的技术包括：群组式RL（GRPO、GiGPO）框架；下一步观测预测的监督学习（教师强制/交叉熵损失）；交替训练策略（频率超参数I）以及对LLM生成的动作/思路进行结构化标签（<think>/<action>等）。

**📊 数据集**

实验数据集：WebShop（模拟电商购物任务）与ALFWorld（基于家庭情景的多步决策任务）。

**📈 对比分析**

与多种基线对比：基于提示的ReAct/Reflexion、闭源LLM（GPT‑4o、Gemini‑2.5‑Pro）、原始RL方法GRPO与GiGPO、以及早期经验/世界模型方法。TAPO在WebShop和ALFWorld上均实现了显著提升，例如在Qwen2.5‑1.5B模型上，GRPO成功率从56.8%提升至66.2%，GiGPO从67.4%提升至71.7%；在Qwen2.5‑7B模型上，TAPO进一步将GRPO成功率从73.7%提升至83.6%，GiGPO从77.9%提升至93.6%。

**⚠️ 局限性**

局限性包括：①需对交替频率I进行调优，虽然不算极端敏感但仍需实验；②辅助监督只在离散动作空间中验证，持续动作场景尚未测试；③在某些超出任务域的通用能力上略有下降；④训练成本相对传统RL略高，主要体现在额外的监督损失计算。

---

## 372. Optimizing Memory Efficiency and Index Ordering to Simulate Quantum Circuits Using Tensor Decision Diagrams

**arXiv ID:** 2607.27971 | [PDF](https://arxiv.org/pdf/2607.27971v1)

**作者:** Vicente López Oliva `[一作]`, Maria Isabel Castillo Catalán `[通讯]`

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

改进了基于决策图与张量网络的量子电路精确模拟器ftdd，提出了硬件感知的内存管理与索引顺序优化；

**💡 创新点**

主要创新点包括：① 对内存分配采用固定容量池和垃圾回收策略，显著控制内存膨胀；② 引入引用计数与节点回收机制，避免内存泄漏；③ 通过选择性缓存和哈希表大小动态调整，降低查找冲突；④ 设计了基于剪枝宽度的路径相关索引顺序（Path‑based Ordering），进一步压缩决策图并提升收敛速度；

**🔧 技术方法**

使用的技术包括：张量网络张量收缩、决策图（TDD）数据结构、哈希表（FNV）、节点引用计数、内存池分配、逆Cuthill‑McKee与路径宽度最小化等；

**📊 数据集**

使用的基准数据集为MQTBench，包含9个典型电路（QFT、GHZ、Graph、QPE、QWalk、RA、AE、QNN、RQC），测试从20到100量子比特的规模；

**📈 对比分析**

与原版ftdd以及不同收缩策略、表大小配置进行对比，结果显示：在高结构化电路（QFT、GHZ、Graph）下内存消耗保持恒定，仅用≈13 GiB；在低结构化/高度纠缠电路（AE、RQC）时，路径优化可实现15–26倍的时间加速；整体上，改进版在绝大多数案例中减少了内存占用并提升了执行速度，尽管在极度纠缠的电路中仍需更长时间；

**⚠️ 局限性**

局限性包括：① 仍为单线程实现，无法充分利用多核/GPU资源；② 对于极高纠缠或深度电路，仍面临时间膨胀和垃圾回收开销；③ 内部结构硬编码上限为100比特，限制了更大规模实验；④ 内存池与哈希表的尺寸调优依赖硬件环境，需手动调节；

---

## 373. Beyond Binary Rewards: A Comparative Study of Reward Design for Reinforcement Unlearning

**arXiv ID:** 2607.27968 | [PDF](https://arxiv.org/pdf/2607.27968v1)

**作者:** Efstratios Zaradoukas `[一作]` (Technical University of Munich), Gjergji Kasneci `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了强化学习中可验证奖励的设计，提出指数奖励和 PageRank 奖励来加速语言模型的去学习。

**💡 创新点**

创新点在于引入奖励解构框架，将可验证性与稀疏性分离，并提出基于计数的指数奖励和利用语义结构的 PageRank 奖励，证明它们在去学习效率上优于传统二进制奖励。

**🔧 技术方法**

使用了 GRPO（群组相对策略优化）与强化学习技术，结合 GPT‑4 提取实体、构造语义图并计算 PageRank，形成可验证奖励。

**📊 数据集**

实验基于 RWKU（Real World Knowledge Unlearning）基准，采用 Phi‑3‑Mini‑4K‑Instruct 大模型进行评估。

**📈 对比分析**

与原始模型和 Binary（PURGE）基线在 Forget、Neighbor、MIA、Utility 四个拆分上对比，指数奖励和 PageRank 奖励在 Forget 维度提升 5‑40% 并且收敛速度提升约 3 倍，同时保持与基线相当的通用性能。

**⚠️ 局限性**

局限性包括需要人工选择忘却目标和阈值、奖励参数（如 τ、PageRank 温度）需手动调优，以及语义图构造的鲁棒性和迁移性尚未完全验证。

---

## 374. MARS-RA: Rank Aggregation for Credit Assignment via Multimodal Comparisons in Embodied Multi-Agent Cooperation

**arXiv ID:** 2607.27967 | [PDF](https://arxiv.org/pdf/2607.27967v1)

**作者:** Dawei Wang `[一作]` (Newcastle University), Richard Davison `[通讯]` (Newcastle University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MARS-RA框架，将多智能体信用分配问题转化为利用大型多模态模型生成的成对比较进行排名聚合，并通过潜在函数奖励塑造来指导训练。

**💡 创新点**

创新点在于将信用分配重新表述为排名聚合任务，利用成对比较降低对绝对奖励的依赖，并在开放式、多变人数的环境中实现鲁棒性。

**🔧 技术方法**

使用大型多模态模型（Gemini-2.5-Pro、GPT-5.1等）生成成对比较，Bradley–Terry模型做排名聚合，潜在函数奖励塑造与MAPPO等MARL基线结合。

**📊 数据集**

构建MARS-Bench（基于ManiSkill3）包含Pass Gate、Herd Sheep、Collect Ball等任务，并在Overcooked和Pistonball上进行跨域验证。

**📈 对比分析**

与QMIX、COMA、SAMA、V-GEPF等基线比较，MARS-RA在MARS-Bench任务中成功率均超出50%，在Herd Sheep上超过70%，在Overcooked和Pistonball也表现与SAMA相当或更优。

**⚠️ 局限性**

主要局限是对大型多模态模型的准确性依赖、仅适用于可视化/文本信息的任务以及仍未完全消除多智能体训练中的非平稳性。

---

## 375. SemPIC: Learning Semantic Position-Independent KV Caches

**arXiv ID:** 2607.28069 | [PDF](https://arxiv.org/pdf/2607.28069v1)

**作者:** Hui Xie `[一作]` (Beihang University), Jinyang Guo `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SemPIC 方法，在离线阶段训练一个 LoRA‑启用的 Writer 生成完整文档的 KV 缓存，使得预训练的 Reader 能以原始 KV 接口直接读取，从而实现位置无关的高质量缓存重用；同时引入 KV Gradient Checkpointing 以降低长文档训练的显存开销。

**💡 创新点**

创新点在于：① 将缓存适配从仅边界状态扩展到完整文档级别，解决独立编译后上下文缺失问题；② 通过 LoRA 仅在 Writer 编译时开启，保持 Reader 不变，兼容标准 KV 接口；③ 提出 KV Gradient Checkpointing，允许在保持 KV 可微分的同时显著减少训练显存。

**🔧 技术方法**

主要技术包括：LoRA 参数化的 Writer、行为蒸馏（Knowledge Distillation）对齐全上下文教师行为、RoPE 旋转校正、KV Gradient Checkpointing、以及标准的 Transformer 预训练模型。

**📊 数据集**

使用的数据集有 Synthetic Biographies、HotpotQA、MuSiQue、Needle‑in‑a‑Haystack（NIAH）以及多源混合训练集，覆盖单文档检索、多跳推理和稀疏答案检索场景。

**📈 对比分析**

与 Full Recompute、No Recompute、KV Packet、CacheBlend 等方法对比，SemPIC 在 12 组模型/任务中提升微 F1 平均从 0.53 提升至 0.60，接近 Full Recompute 的 0.62；在 10/12 设定下优于 KV Packet；并在内部注意力误差指标上均有显著下降，显示更完整的上下文重用效果。

**⚠️ 局限性**

局限性包括：① 仍存在块首位 token 的注意力峰值，未完全消除内部误差；② 需要离线编译与额外训练成本；③ KV Gradient Checkpointing 牺牲训练速度；④ 仅验证于四个任务和三大模型，尚未证明在更广泛检索与推理场景中的普适性；⑤ 仍缺乏对注意力改进与 F1 提升因果关系的定量证据。

---

## 376. DataClawEval: A Benchmark for Data Engineering Agents in Real Industrial Harness

**arXiv ID:** 2607.28033 | [PDF](https://arxiv.org/pdf/2607.28033v1)

**作者:** Debin Meng `[一作]` (Tencent), Peng Chen `[通讯]` (Tencent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了DataClawEval基准，评估自主数据工程代理在真实工业场景下的端到端任务完成能力。

**💡 创新点**

1) 采用真实生产级代码构建任务，保证工业真实性；2) 通过人机交互+差分验证实现答案可辨识；3) 引入可重现沙箱与案例专用确定性评分脚本，兼顾产出与过程评估。

**🔧 技术方法**

人机交互构造管道、LLM推理生成任务意图、差分测试、Docker化隔离环境、案例专用脚本评判以及统一的agent框架（Tencent CodeBuddy）。

**📊 数据集**

基于腾讯、北京、深圳及多所高校数据工程团队的生产代码，涵盖5种执行引擎（PySpark、HiveSQL、MySQL、PrestoSQL/Trino、FlinkSQL）的100个任务。

**📈 对比分析**

在统一的CodeBuddy环境下对16款LLM代理进行单次与多次运行，分别给出整体、引擎级别得分及token消耗；结果显示最高模型总体得分仅74.9，单引擎差异大，无单一模型统治所有引擎，token消耗与质量无正相关。

**⚠️ 局限性**

1) 仍远未解决全能数据工程任务；2) 任务分布不均导致引擎难度差异大；3) 评测对token成本与工具调用效率不敏感；4) 需进一步提升模型的稳定性与跨引擎通用性。

---

## 377. ENCORE: Event-Assisted Complementary Motion Refinement for Learned Video Compression

**arXiv ID:** 2607.28020 | [PDF](https://arxiv.org/pdf/2607.28020v1)

**作者:** Shuhan Ye `[一作]` (Wuhan University), Qixin Zhang `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种事件辅助的视频压缩框架，在RGB编码过程中用事件信息细化运动估计；

**💡 创新点**

创新点在于先将RGB‑事件特征分解为公共与特定运动，再通过能量与冗余校准筛选有用事件响应，并用能量感知路由决定校正的空间应用；

**🔧 技术方法**

采用事件到运动的时空卷积映射、Complementary Motion Representation（CMR）、Spatial Energy & Redundancy‑Informed Calibration（SERIC）与Energy‑Aware Routing（EAR）等模块；

**📊 数据集**

在BS‑ERGB、HQ‑EVFI和CED这三个同步RGB‑事件数据集上训练与评估；

**📈 对比分析**

与RGB‑only的HyTIP基线及直接拼接RGB+E Concat对比，BD‑rate在BS‑ERGB上下降约20%，在HQ‑EVFI和CED上分别下降约9%和6%；

**⚠️ 局限性**

对事件稀疏度和噪声鲁棒性考虑不足，且在编码时的计算开销略有提升。

---

## 378. ViP-Rig: Visual-Prompted Controllable Rigging

**arXiv ID:** 2607.27982 | [PDF](https://arxiv.org/pdf/2607.27982v1)

**作者:** Zihan Qin `[一作]` (Harbin Institute of Technology), Xianming Liu `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

介绍了ViP-Rig可视化提示的两阶段动画装配框架，支持从2D骨架提示和刚性提示生成可控骨骼与蒙皮。

**💡 创新点**

将视觉提示通过稠密到紧凑编码与层级门控适配器注入冻结的预训练生成器，实现对骨架布局与蒙皮权重的细粒度、可迭代控制。

**🔧 技术方法**

DINOv2视觉编码、Perceiver重采样、门控注意力适配器、Puppeteer点-关节匹配、双向特征注入等技术。

**📊 数据集**

使用Articulation-XL2.0进行训练与测试，并在ModelsResource上进行零射测评。

**📈 对比分析**

与RigNet、MagicArticulate、UniRig、Puppeteer等基线按骨架Chamfer距离、蒙皮精度召回和L1误差比较，ViP-Rig在骨架CD、蒙皮精度/召回和误差上均优于基线，零射测评亦表现最优。

**⚠️ 局限性**

仍依赖二维渲染与提示的准确性，缺乏对三维结构细节的直接编辑，且跨域鲁棒性待进一步探索。

---

## 379. FootprintNet: State-Transition-Guided Dynamic Footprint Learning for Multi-temporal Remote Sensing Change Detection

**arXiv ID:** 2607.27969 | [PDF](https://arxiv.org/pdf/2607.27969v1)

**作者:** Haotian Zhang `[一作]` (Beihang University), Zhenwei Shi `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出城市建筑动态检测（UBDD）任务，并基于多时相遥感图像构建FootprintNet模型，对建筑变化的动态足迹进行像素级识别。

**💡 创新点**

创新点包括：①放宽单次变化假设，统一建模单次与多次变化；②引入状态‑动作转移约束，利用强化学习奖励实现因果一致的变化轨迹；③使用非对称时间边界对比学习提升不同时间段特征区分度；④设计建筑变化动态得分（BCDS）评价指标，更精细地衡量时间一致性和空间精度。

**🔧 技术方法**

技术方案由三大分支组成：动作导向潜在状态转移（ALST）使用Mamba编码器、状态嵌入与动作预测并通过GRPO约束；非对称时间边界对比（ATBC）结合空间与时间Transformer与随机时间打乱实现边界感知；空间‑时间状态空间扫描（STSS）采用空间与时间SSM并求差分特征预测动态足迹；同时引入自定义奖励函数与交叉熵损失进行联合训练。

**📊 数据集**

实验数据集为公开的TSCD、MUDS和WUSU三大多时相遥感数据集，涵盖多时段建筑变化场景。

**📈 对比分析**

与多种CNN、Transformer、Mamba以及SSM基线方法对比，FootprintNet在mIoU、mF1、BCDS等指标上实现显著提升（如TSCD上mIoU提升约0.67%，MUDS提升约1.12%，WUSU提升约5.84%），并在各类别的IoU上均位居榜首。

**⚠️ 局限性**

局限性：对光照、季节差异等外部因素仍易受干扰，导致部分时间偏差未完全消除；模型结构复杂，训练与推理成本较高；在极少见的多变场景中多变更的时间定位精度仍有提升空间。

---

## 380. Specification-Guided Synthesis of Deadlock-Free Communication Protocol Refinements with Large Language Models

**arXiv ID:** 2607.27964 | [PDF](https://arxiv.org/pdf/2607.27964v1)

**作者:** Yang Li `[一作]` (University of Oxford), Nobuko Yoshida `[通讯]` (University of Oxford)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建一个结合大型语言模型（LLM）与多方会话类型（MST）的框架，用于自动合成符合死锁自由等行为正确性的通信协议细化。

**💡 创新点**

创新点在于：① 将正式的异步多方子类型约束直接嵌入LLM生成流程；② 通过两级（前端快速过滤 + 后端完整验证）监控保证生成协议满足子类型关系；③ 采用结构化提示与语法化表示提升生成多样性与有效率。

**🔧 技术方法**

技术手段包括：LoRA 微调的 Qwen/CodeLlama/StarCoder 等 7B/32B 代码模型；语法化的提示与规范化（BNF 风格）；两级监控（token‑level 过滤 + sequence‑level 验证）；异步多方子类型检查器；自定义的训练损失与加权。

**📊 数据集**

数据集：① 文献中提取的真实协议（约 50 条）；② 合成基准协议（约 200 条），每条配有多种子类型实例，用于训练与评估。

**📈 对比分析**

对比方法：在三种 7B 代码模型、一个通用 7B 模型与一个 32B 模型上进行实验；实验结果显示 95.6%–99.5% 的生成协议通过子类型检查，95.4%–98.1% 的语法正确率；与 GPT‑5.5、DeepSeek‑V4‑Pro 等前沿模型对比，覆盖面更广、生成多样性更高。

**⚠️ 局限性**

局限性：① 异步子类型判定不可判定，需依赖外部检查器；② 仅验证 MST 框架，无法直接迁移到其他协议形式；③ 对大模型的计算成本高，细化生成仍需人工后验验证；④ 训练数据规模有限，可能影响泛化能力。

---

## 381. Landmark shape spaces with induced metrics

**arXiv ID:** 2607.28064 | [PDF](https://arxiv.org/pdf/2607.28064v1)

**作者:** Sarang Joshi `[一作]`, Stefan Sommer `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出屏蔽弹性算子 L_s，构建新的形状空间，使其兼具 Kendall 形状空间的刚体不变性和 LDDMM 形状空间的平滑度量，并保证点不碰撞与尺度固定；

**💡 创新点**

创新点在于将条件正定核与刚体运动零空间相结合，得到一个能同时抑制全局刚体、保持局部刚体运动且不产生点碰撞的度量，且该度量可在任何点数下使用；

**🔧 技术方法**

主要技术包括条件正定核理论、Sobolev 与弹性算子、右不变 Riemannian metric、Hamiltonian 与约束系统、RATTLE 时序步进、自动微分优化以及几何投影；

**📊 数据集**

实验数据以合成蝴蝶形状、圆形与旋转圆形等多种标记点集（如 32 点圆）为主，未使用公开医学或大型真实数据集；

**📈 对比分析**

通过与 Gaussian、Matérn-3/2 等经典核在局部刚体运动保持、尺度保持以及形状匹配误差上的对比，实验表明屏蔽弹性核 k₂ 在保持局部刚体运动和尺度不变性方面优于其它核，且数值积分能稳定地保持能量与刚体约束；

**⚠️ 局限性**

局限性包括需要手动进行尺度归一化、低阶 s 的算子不能保证点不碰撞、数值约束保持仍存在误差、计算成本相对较高，并且目前仅在低维欧氏空间（d=2,3）下验证，缺乏大规模实验和理论完整性。

---

## 382. Diversifying Personalized Research Ideation against AI-Induced Homogenization

**arXiv ID:** 2607.28087 | [PDF](https://arxiv.org/pdf/2607.28087v1)

**作者:** Rui Xu `[一作]` (Wuhan University), Yong Luo `[通讯]` (Wuhan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一套四阶段的 AI 辅助研究方向生成与筛选框架（DivAlign），通过细粒度研究者画像提取、条件化方向生成、三维对齐评分和社区级冗余惩罚，实现研究者个性化与社区多样性兼顾的研究方向呈现。

**💡 创新点**

创新点在于：①将研究者画像细化为研究脉络、拥有的技术产出与已知空白；②将对齐评分拆为可执行性、可理解性与成长潜力三维度；③在社区层面使用 MMR 风格的贪心选择，并以最大相似度惩罚冗余，从而在保持个体匹配的前提下降低跨研究者方向重复。

**🔧 技术方法**

技术手段包括：大型语言模型（Claude Haiku/Sonnet）用于画像提取与方向生成；Sentence‑BERT 进行文本嵌入与相似度计算；LLM 作为评分器评估三维对齐得分；基于贪心搜索的社区级选择算法。

**📊 数据集**

使用了 95 名 AI 研究者的公开资料（五个子领域：视频理解、医疗 AI、三维视觉、具身 AI、效率 AI），共 930 篇 2018‑2022 年期刊/会议论文以及个人主页简介。

**📈 对比分析**

与粗粒度单一建议（Coarse‑K1）、随机抽样、独立最高评分（λ=0）以及极端多样化（λ→∞）进行对比。结果显示：DivAlign 在平均余弦相似度 (NS) 从 0.704 降至 0.608，最大余弦相似度 (HS) 从 0.331 降至 0.294，同时保持 99.9% 的对齐分数；Vendi 多样性指标提升，说明社区多样性显著改善。

**⚠️ 局限性**

局限性包括：①依赖 LLM 的质量与提示工程，对生成和评分结果的稳定性有影响；②对齐评分仍为基于文本的代理，缺乏真实实验验证；③在极大规模社区时仍可能出现细粒度重复，需要进一步的多模态或跨域约束。

---

## 383. Layered Architecture for Mobile Intelligence

**arXiv ID:** 2607.28083 | [PDF](https://arxiv.org/pdf/2607.28083v1)

**作者:** Qingwen Liu `[一作]`, Mingqing Liu `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了移动AI栈（Mobile AI Stack）概念，构建了一个五层移动感知与智能体系结构；

**💡 创新点**

将移动性从单纯的通信挑战提升为整个AI系统的核心设计原则，系统性地将能量、芯片、计算架构、模型与应用层级融合；

**🔧 技术方法**

综述并整合了无线能量传输、能耗友好AI芯片、云‑边缘‑移动分层计算、分布式AI模型、具身智能应用等多项技术；

**📊 数据集**

无实验数据集，论文为概念与框架性综述；

**📈 对比分析**

无基准实验与性能评估，未给出具体实现或数值比较；

**⚠️ 局限性**

局限在于缺乏实现细节与实测验证，未提供实现方案、性能指标、以及在真实场景中的可行性分析。

---

## 384. Chem World: A Large-Scale Benchmark and Physics-Informed Framework for Trustworthy Chemical Property Prediction

**arXiv ID:** 2607.28079 | [PDF](https://arxiv.org/pdf/2607.28079v1)

**作者:** Tianyou Bai `[一作]` (Cleer Science), Siming Dong `[通讯]` (Cleer Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了 Chem World 基准，整合 17 个公开数据集、800k+ 化学样本，覆盖 10 个性质轨道，并提出 Mixture-PINN 物理信息神经网络框架。

**💡 创新点**

将多源化学混合数据统一标准化并设计 OOD 评估，结合化学先验的物理约束实现更可靠的混合物性质预测。

**🔧 技术方法**

采用预训练分子模型（MolFormer/MolT5）与 GNN 的分子编码器，结合自注意力/Set Transformer 聚合与差分物理约束（组成一致性、交互对称性、温度单调性、边界约束）实现 Mixture-PINN。

**📊 数据集**

集成 17 个公开混合物/分子数据集（如溶解度、电导率、粘度等），覆盖 10 个属性轨道，总计 800k+ 记录。

**📈 对比分析**

在统一的随机划分和 OOD 设置下，与 GNN、MolFormer+Set Transformer 等基线对比，Mixture-PINN 在大多数轨道取得最优 RMSE、MAE、R^2，显著提升 15‑20% 的准确率。

**⚠️ 局限性**

依赖公开实验数据的噪声与偏差，物理约束参数需要手动设置，且在未见化学体系或极端条件下仍需进一步验证。

---

## 385. Where and When to Commit: Candidate-Aware Decoding for Diffusion Language Models

**arXiv ID:** 2607.28166 | [PDF](https://arxiv.org/pdf/2607.28166v1)

**作者:** Chia-Ming Lee `[一作]` (National Yang Ming Chiao Tung University), Chih-Chung Hsu `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了训练无关的分块早停框架 LATCH，结合 Confidence-Verified Commit 与 Block-Wise Early Commit 对扩散语言模型进行加速。

**💡 创新点**

创新在于将终止和采样两个加速轴分离，使用候选答案的稳定性与置信度进行验证，避免了传统基于位置级统计的过早终止。

**🔧 技术方法**

使用基于扩散语言模型的 LLaDA 与 Dream 的解码过程，结合候选提取器、阈值门控、块级早提交规则。

**📊 数据集**

评估 11 个零样本任务，包括 MMLU、ARC、HellaSwag、WinoGrande、PIQA、TruthfulQA、GSM8K、MATH、SVAMP、ASDiv、GSM-Hard。

**📈 对比分析**

相较于 Prophet、SlowFast、KLASS 等方法，在保持准确率误差≤2点的前提下，短答任务提升 9.3–17.8× TPS，长推理任务提升 2.0–3.3× TPS。

**⚠️ 局限性**

局限性包括对候选答案提取的依赖，无法处理无明确答案跨度或多答案场景；对超参数 τ_BWEC 的敏感性仍需手动设定。

---

## 386. Collaborative Feature Aggregation for Face Super-Resolution and Robust Re-Identification

**arXiv ID:** 2607.28130 | [PDF](https://arxiv.org/pdf/2607.28130v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 387. String Matching in (Block) Graphs: A Full Classification by Walk Length

**arXiv ID:** 2607.28159 | [PDF](https://arxiv.org/pdf/2607.28159v1)

**作者:** Sebastian Angrick `[一作]` (Karlsruhe Institute of Technology), Yuki Yonemoto `[通讯]` (Kyushu University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在节点标记的块图（block graph）中寻找一条覆盖所有块且对应字符串包含给定模式的路径，提出针对块数 b 的参数化算法与下界。

**💡 创新点**

创新点在于：① 对 b≤3 的情况给出了近线性时间算法（b=2 已知，b=3 通过算术级数与周期性技术实现）；② 证明 b≥4 时无法获得组合型（mE+N）级别的加速（与 BMM、Triangle Detection 等假设相关）；③ 对多块情况提出基于矩阵乘法的条件最优算法；④ 进一步证明在多对数块数下仍存在 SETH/OVH 条件下的硬度。

**🔧 技术方法**

核心技术包括：
- 用算术进程（AP）压缩模式与节点标签之间的前缀/后缀集合；
- 利用字符串的周期-边界关系实现 AP 的快速交叉与合并；
- LCP 预处理与单点查询支持快速模式定位；
- 通过 Boolean 矩阵乘法、三角检测与 OV 等问题的归约构建下界；
- 采用矩阵乘法（含矩形乘法）实现通用块图匹配。

**📊 数据集**

本文为理论分析，未使用实际数据集；仅提及基因组拼图（pangenome graphs）为应用动机，但未给出实验评测。

**📈 对比分析**

在 b=3 时，算法复杂度为 O(m+E+N)，显著优于原先的 O(mE+N)；对任意 b≥4，提供 O((max(V,m)^ω)+N) 的矩阵乘法算法，并证明在 BMM 或 Triangle Detection 假设下不存在更快的组合型算法；对多对数块数则证明无法实现 ((mE)^1− + N) 的加速。没有实验对比，结论仅来自理论复杂度分析。

**⚠️ 局限性**

局限性：
- 对 b>3 的情况仍只能给出条件最优（矩阵乘法）或组合下界，缺乏真正的线性或近线性算法；
- 对 b=4、5 的唯一标签（unique labels）情况仍是开放问题；
- 复杂度上限基于 ω，实际实现可能受限于矩阵乘法常数；
- 未给出对真实生物信息数据的实证评测，缺乏性能验证。

---

## 388. Challenges in annotations by humans and LLMs: A case study of evaluative language

**arXiv ID:** 2607.28119 | [PDF](https://arxiv.org/pdf/2607.28119v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 389. What Makes Deep Learning Work for Traditional Chinese Medicine Tongue Diagnosis? A Comprehensive Ablation Study

**arXiv ID:** 2607.28148 | [PDF](https://arxiv.org/pdf/2607.28148v1)

**作者:** Longxia Gao `[一作]` (Hebei University), Hanqing Zhao `[通讯]` (Hebei University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过系统化的消融实验，对传统中医舌诊多标签分类任务中的模型架构、损失函数、数据增强、训练策略以及数据规模进行深入探讨。

**💡 创新点**

创新点在于提出六条可直接迁移到其他医学多标签任务的设计原则，并通过超过20个模型版本的严格对比验证了数据规模与标签设计在提升性能中的决定性作用。

**🔧 技术方法**

主要技术包括ConvNeXt‑Tiny骨干网络、BCE加正样本权重损失、受限HSV颜色扰动、弱组独立分类器替换、EMA、TTA以及多任务微调策略。

**📊 数据集**

使用了TongueDx2（976个专家标注样本、5,109总样本）与TonguExpert（5,992样本）合并的11,101张舌面图像，构成13维二元标签集（以及一次45维细粒度实验）。

**📈 对比分析**

在5折交叉验证框架下，对比不同组合的骨干、损失、增强与训练策略，最佳单模型在976样本时获得0.6625的加权F1，扩大至11,101样本后加权F1提升至0.7761，验证了数据扩充与标签简化的显著效果。

**⚠️ 局限性**

局限性包括数据仅来自单中心，标签质量受人工与自动推断混合影响，极端稀疏标签导致的类别不平衡问题，以及缺乏跨中心外部验证。

---

## 390. LM-GRASP: Instance-Specific Language Models for Combinatorial Construction via Online Imitation Learning

**arXiv ID:** 2607.28135 | [PDF](https://arxiv.org/pdf/2607.28135v1)

**作者:** Mohand Mezmaz `[一作]` (University of Luxembourg), Grégoire Danoy `[通讯]` (University of Luxembourg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出LM-GRASP框架，将GRASP构造阶段改为基于decoder‑only Transformer的在线模仿学习，利用局部搜索专家产生的精英解进行行为克隆，无需预训练或手工特征；

**💡 创新点**

核心创新在于将随机构造过程重新表述为在线模仿学习任务，完全从实例生成的演化轨迹中学习构造策略，实现实例特定、无预训练的语言模型构造器；

**🔧 技术方法**

使用Transformer（GPT‑2样式）作为自回归生成器，行为克隆训练，循环式在线学习-生成-改进机制，局部搜索作为专家或oracle；

**📊 数据集**

在Taillard 50×20的Permutation Flow‑Shop Scheduling Problem（PFSP）基准集上进行实验；

**📈 对比分析**

与CPU‑GRASP和GPU‑GRASP在相同5小时预算下对比，LM‑GRASP在所有10个实例和100次运行中均取得最优，使得算法增益≈28.4单位（与GPU加速相当），并在平均makespan上显著优于基线；

**⚠️ 局限性**

仅在足够复杂的实例（如50×20块）表现优异，对更小或结构更简单问题可能不具优势；在线训练开销高，尚未评估对其他组合优化问题的泛化及超参数敏感性。

---

## 391. Information Bottleneck Learning for Faithful Time Series Forecasting Explanations

**arXiv ID:** 2607.28124 | [PDF](https://arxiv.org/pdf/2607.28124v1)

**作者:** Xu Zheng `[一作]` (Florida International University), Dongsheng Luo `[通讯]` (Singapore Management University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种自解释的多变量时序预测框架，在单次前向传递中同时输出预测值和稀疏解释；

**💡 创新点**

通过将预测拆分为可学习的周期性基线与门控偏差读出，并利用信息瓶颈的硬二值门实现稀疏、可解释且可精度控制的解释；

**🔧 技术方法**

使用周期性查找表、实例归一化、PatchDecomp式分词、Transformer门网络、硬 Concrete 门、信息瓶颈正则、TV 连续性约束等技术；

**📊 数据集**

在六个标准多变量时序数据集上评估：ETTh1、ETTh2、ETTm1、ETTm2、Weather 与 Electricity；

**📈 对比分析**

与可解释与黑盒前沿模型（TFT、PatchDecomp、DLinear、TQNet、CycleNet 等）对比，实验显示所提模型在 MSE/MAE 与匹配预算信度（fidelity）上均与最优黑盒相当且在解释性上显著优于现有可解释方法；

**⚠️ 局限性**

局限性：模型仍需在训练阶段学习门控，适配不同长度/频率的周期性需要手工设定周期长度；在极端稀疏预算下性能下降明显；尚未验证对非平稳或突发事件的鲁棒性。

---

## 392. mmRadarTwin: A Measurement-Calibrated Signal-Level Digital Twin Platform for Indoor mmWave Radar

**arXiv ID:** 2607.28108 | [PDF](https://arxiv.org/pdf/2607.28108v1)

**作者:** Jianyi Zhou `[一作]` (University of Sydney), Dong Yuan `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了 mmRadarTwin，一个基于信号级和路径归因的室内毫米波雷达数字孪生平台，能够将真实雷达测量与 Unreal Engine 场景仿真通过共享的距离-角度处理链对齐，并输出每条传播路径的复数贡献；

**💡 创新点**

创新点在于：①将雷达测量与仿真直接映射到相同的距离-角度域；②保留路径级复数贡献记录，实现残差诊断能归因于几何、材料、系统或缺失机制；③提供可复用的工作流程，支持单雷达部署、场景重建、参数校准与残差分类；

**🔧 技术方法**

采用基于射线追踪（SBR）的物理路径仿真，Unreal Engine 5 场景重建，FMCW 8‑通道虚拟阵列处理，FFT 共享链路，复数接收通道记录；

**📊 数据集**

使用 154 个测量姿态（22 个雷达位置）在一间办公环境中采集，雷达为 TI AWR2243BOOST，配合 RealSense 深度摄像机与激光测距仪进行姿态标定；

**📈 对比分析**

通过将测量与仿真在同一 64×64 范围-角度网格上比较，评估指标包括：70.8% 区域召回、Top‑1/3/5 约 26% 的峰值匹配、匹配区域的幅值误差 1.01 dB、残差分类占比（C1 3%，C2 26%，C3 16%，C4 33%，C5 22%），整体显示物理路径模型能恢复大部分结构响应，但仍有显著误差；

**⚠️ 局限性**

局限性包括：①远程墙面响应、扩散散射与多路径难以准确模拟；②错误或偏移的响应往往需几何、姿态、波束或系统校正；③缺失的物理机制（如衍射、粗糙表面散射）未被覆盖；④仅在单一办公室环境验证，跨房间泛化尚未测试。

---

## 393. From Expert Reduction to Behavioral Divergence: Tracing Numerical State through Sparse MoE Inference

**arXiv ID:** 2607.28097 | [PDF](https://arxiv.org/pdf/2607.28097v1)

**作者:** Tianyang Zhu `[一作]` `[通讯]` (Independent Researcher), Tianyang Zhu (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在 DeepSeek‑V4‑Flash 原生推理环境下，控制专家聚合顺序（四种方案）并冻结后向 MHC 状态，探究数值精度与聚合顺序对稀疏 MoE 推理的影响；通过端点差分重建、后向 MHC 以及全持续状态恢复，验证这些数值差异能否完整重现后续推理轨迹，并提出层级化的运行时一致性诊断流程。

**💡 创新点**

①揭示了在稀疏 MoE 推理中，聚合顺序即使在同一模型权重下也能产生可观的执行分支与语义分化；②提出后向 MHC 与全持续状态为关键的内部边界，能够定位并重现推理分支；③给出两种精确的端点重建方法（post‑mHC 与全持续状态）并验证其可重复性；④提出可扩展的多层次一致性检查框架；⑤验证了 C 方案（BF16 术语、FP32 累加）在实验集上实现了顺序不变性。

**🔧 技术方法**

使用的技术包括：数值可重放实验框架、四种聚合方案（P32、C、A、B）的实现、端点差分计算与 FP64 加法重建、层级化的状态比较（Operator、Layer、Persistent、Token、Text），以及 SHA‑256 哈希和 L∞ 误差统计。

**📊 数据集**

数据集主要是 DeepSeek‑V4‑Flash checkpoint 及其对应的三条深度 prompt（"why the sheep"、"朋友昨天打来电话"、"Morning light filled the room"）和 50 条英中混合 prompt；实验覆盖 768 条八 token 轨迹、10 条 64 token 长轨迹、以及对 360 结构类的探索，全部在同一 CPU host 上执行。

**📈 对比分析**

比较方法为逐层逐位比对 MoE 输出、post‑mHC 状态、路由选择、token 序列与文本，并通过 SHA‑256 对完整 token 序列进行哈希一致性检验。实验结果显示 P32、A、B 在路由与 token 上均出现分歧，而 C 与 native 参考保持完全一致；端点重建可精确恢复后续轨迹，验证了后向 MHC 与全持续状态边界的有效性；由于实验在单一 CPU 环境下完成，未给出跨设备的性能指标。

**⚠️ 局限性**

局限性包括：仅在单一 checkpoint 与单一 native runtime（CPU）上测试；只覆盖 6‑term 专家聚合和有限 prompt、层与调度；未测量 GPU/NPU 等硬件平台的实际出现概率；未进行冻结路由控制、量化精度极限分析或更长序列的可重复性验证；对跨平台一致性与硬件差异的泛化能力未作评估；实验规模有限，无法推断真实部署频率。

---

## 394. Search Strategies for Optimal Classification and Regression Trees

**arXiv ID:** 2607.28170 | [PDF](https://arxiv.org/pdf/2607.28170v1)

**作者:** Jacobus G. M. van der Linden `[一作]` (Delft University of Technology), Emir Demirović `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个通用的搜索框架，用于在分类与回归任务中学习最优决策树（ODT），并通过该框架对多种搜索策略（DFS、BFS、LDS、AND-OR）进行统一实现与系统比较。

**💡 创新点**

创新点在于将不同搜索策略统一到一个AND-OR搜索树框架中，使得能够直接对策略进行对比；并设计了两种新策略——BFS-Small-LB（优先低支持与低下界的节点）和DFS-Blossom（平衡左右子树展开），显著提升了求解速度和任意性性能。

**🔧 技术方法**

采用了AND-OR搜索树、增量式扩展、基于下界/上界的剪枝、优先级队列与启发式函数相结合的搜索策略，利用Rust实现并提供Python接口。

**📊 数据集**

在多个公开基准数据集上评估，包括分类数据集（如Wilt、Bank等）与回归数据集，实验覆盖深度限制d=3到d=5。

**📈 对比分析**

与现有最优决策树方法（如GOSDT、MurTree、Quant-BnB、LDS-DL8.5等）以及贪心学习器（如CART）进行对比；结果显示BFS-Small-LB在求解时间上领先，DFS-Blossom在任意性性能上最佳；在d=3时，方法在分类任务上比基线快110倍，回归任务上快40倍，并在任意性指标OI上表现最优。

**⚠️ 局限性**

限制包括：对数值特征仍需进行粗粒度二值化导致搜索空间增大；深度限制受限（实验多在d≤5）；当前实现单线程，可能在大规模数据上受内存与并行度限制；实验主要集中在公开基准数据集，尚未在极大数据规模或非二进制标签场景中验证。

---

## 395. Checking Information Flow in Cloud-based IoT Access Control Policies (Extended Version)

**arXiv ID:** 2607.28088 | [PDF](https://arxiv.org/pdf/2607.28088v1)

**作者:** Lorenzo Ceragioli `[一作]` (IMT School for Advanced Studies Lucca), Edoardo Lunati `[通讯]` (IMT School for Advanced Studies Lucca)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于SMT的验证方法，用于检查云端IoT访问控制策略（以AWS IoT Core为例）中设备之间可能的安全信息流。

**💡 创新点**

创新点在于：①给出完整的形式化模型，将AWS IoT策略与MQTT通信模型结合；②构造符号信息流图，将无限主题空间压缩为逻辑公式；③利用SMT求解器实现对信息流可达性的精确检查，并支持攻击者利用通配符注入的场景；④提供可查询的安全标签机制。

**🔧 技术方法**

主要技术包括：形式化语义定义、正则表达式与字符串处理的SMT编码（使用cvc5），图搜索（NetworkX）实现可达性查询，Python实现的工具链。

**📊 数据集**

数据集：1）基于真实AWS IoT策略的公开基准（P-verifier）共258个策略；2）构造的建筑自动化系统（BAS）场景，包含17台设备、20个证书、15条策略。

**📈 对比分析**

性能评估显示：构造符号信息流图对最多258个策略只需不到64秒；单次可达性查询平均不到0.1秒。相较于传统单策略SMT分析，方法在多策略全局交互场景下保持线性可扩展性；与现有工具（如Z3、Zelkova等）相比，cvc5在本实验中表现更优。

**⚠️ 局限性**

局限性包括：仅针对AWS IoT Core，未覆盖ARN通配符、临时凭证、Things、Condition等高级特性；未处理动态设备加入/离开导致的策略更新；依赖SMT求解器的决策过程，可能在极大策略组合时遇到求解瓶颈。

---

## 396. LEEPS: Latent-Guided Explore-Exploit Prompt Sampling for Efficient RLVR in Large Language Models

**arXiv ID:** 2607.28077 | [PDF](https://arxiv.org/pdf/2607.28077v1)

**作者:** Shuang Liang `[一作]` (Renmin University of China), Xiting Wang `[通讯]` (Renmin University of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种预生成阶段的提示采样器LEEPS，利用自监督的奖励机制在强化学习训练中高效筛选信息丰富的提示；

**💡 创新点**

创新点在于：①自适应的探索–利用组合分配（Adaptive Explore–Exploit Portfolio Allocation）通过实时估计探索与利用组的非零方差比例，保持训练批次的高信息比例；②潜在导向探索（Latent‑Guided Exploration）利用提示在模型隐藏空间中的相似性，借助邻近已观察到的提示成功率预测并优先挑选具有中等难度的探索候选；

**🔧 技术方法**

采用了RLVR（Group Relative Policy Optimization）框架、隐藏层提示表征、KNN邻域成功率估计、基于伯努利方差与潜在不确定度的加权采样，以及自适应比例调整算法；

**📊 数据集**

使用的主要数据集为 DAPO‑Math‑17K 进行训练，并在 MATH‑500、Minerva‑Math、OlympiadBench、AMC23、AIME24、AIME25 等数学推理基准以及 GPQA‑Diamond、ARC‑C、MMLU‑Pro 等 OOD 推理基准进行评估；

**📈 对比分析**

与 vanilla GRPO、在线过滤 DS、先验概率采样 MoPPS 以及动态优先级采样 DPS 进行对比；在 1.5B 与 7B Qwen2.5‑Math 模型上，LEEPS 分别比最强基线提升约 2.6% 与 3.7% 的总体分数，且在训练进度曲线中显示出更快的收敛速度；非零方差比例保持在高水平；额外采样开销仅约 2 秒/步；

**⚠️ 局限性**

局限性：① 仅在可验证奖励的数学推理任务上验证，尚未证实其在更广泛任务或多模态场景中的通用性；② 依赖预先构建的 KNN 邻域缓存，可能在大规模数据集上增加存储与检索开销；③ 对提示隐藏表征的质量高度敏感，若表征不佳可能导致潜在导向探索效果下降；

---

## 397. Piggybacking on Perception: Stealthy Concurrent Audio Prompt Injections against Multimodal LLM Agents

**arXiv ID:** 2607.28165 | [PDF](https://arxiv.org/pdf/2607.28165v1)

**作者:** Mingxiao Liu `[一作]` (Hangzhou Dianzi University), Zhen Wang `[通讯]` (Hangzhou Dianzi University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在持续用户语音交互中对多模态 LLM 代理的并发音频提示注入攻击，并构建了 AudioAgentSecurity 基准，评估 11 种代理的易受攻击性，提出了基于源分离与一致性验证的 CADV 防御机制。

**💡 创新点**

创新点在于提出能在用户语音中叠加的“能量增强+动态压缩”与“并发注入前缀”技术，并设计了三级防御架构（音频分离、说话人一致性检查、语义过滤）来对抗此类攻击。

**🔧 技术方法**

使用的技术包括音频合成（Qwen3-TTS）、频谱抖动与滤波、动态范围压缩、深度源分离网络（Mossformer2）、说话人嵌入模型（CAM++）以及 ASR 与语义验证。

**📊 数据集**

采用的数据集为自建的 AudioAgentSecurity（2160 条攻击样本，覆盖 8 个真实场景与 10 种攻击模式），以及 DEMAND 语音噪声数据、真实物理环境录音与 Doubao AI 手机的双盲人类评测数据。

**📈 对比分析**

与 Prompt‑level 防御相比，CADV 在 ASR 上可将攻击成功率从约 70% 降至 20% 以下，检测率超过 90%，但在多说话人环境下误报率升至 35%；在基准评测中，对 Gemini‑3‑Pro 的平均 ASR 为 69.10%，CADV 能显著降低此值。

**⚠️ 局限性**

局限性包括：实验仅覆盖部分设备与声学场景，CADV 在多人嘈杂环境下误报率较高；人类实验样本规模有限，未充分评估实时延迟、能耗与实际部署中的成本；对攻击者使用语音克隆等更高级手段的鲁棒性尚未验证。

---

## 398. Rethinking LLM-Judged Helpfulness as a Pedagogy Signal: A Pre-Registered Audit Across Tutor Models

**arXiv ID:** 2607.28128 | [PDF](https://arxiv.org/pdf/2607.28128v1)

**作者:** Shuyi Fan `[一作]` (Columbia University), Chongyang Gao `[通讯]` (Northwestern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在一个受控实验中，对同一弱学生使用两种LLM辅导策略（对话式ConvTutor与结构化PedTutor），并通过两款LLM评判器（Claude Opus 4.8 与 GPT‑5.6 Sol）对每个回答阶段进行帮助度与教学度评估，同时记录答案泄露与后续学生独立工作的两种确定性过程测量。

**💡 创新点**

发现通用的帮助度评估无法区分答题与教学行为，强调需要将教学目标评估与过程指标相结合；并首次在同一实验框架下比较多种评判器与过程指标的可靠性。

**🔧 技术方法**

技术手段包括：①使用LLM评判器（Opus与Sol）进行多维度打分；②构建答案泄露与后续独立工作两种确定性过程指标；③在三种基础模型（Claude Sonnet、GPT‑5.5、Gemini 3.1 Pro Preview）上重复实验；④采用预注册的统计方法（Wilcoxon、Cliff’s δ、混合效应模型）。

**📊 数据集**

数据集为：一位弱学习者模拟器 Llama‑3.1‑8B，六套训练题目、三套即时测验、四套干扰测验、三套延迟测验与三套迁移测验，共计 90 个实验会话。

**📈 对比分析**

比较方法：使用预注册的双侧 Wilcoxon 同值检验、Cliff’s δ 估计、混合效应模型评估答案泄露与帮助度/教学度的关联。结果显示：帮助度在三种基础模型下差异不显著；教学度在所有基础模型下完全分离（|δ|=1.0）；答案泄露与后续学生独立工作呈显著负相关，且在所有基础模型下保持一致。

**⚠️ 局限性**

局限性包括：仅使用两款LLM评判器且无人工验证；仅采用单一弱学生模拟器与单一学科；样本量有限（十次复现）；未评估长期学习效果；评判器与模型同族可能导致偏倚；以及对低泄露的模型（Gemini）的测量不够敏感。

---

## 399. FinSMART: Financial Sentiment Analysis for Algorithmic Trading through Market-Aligned Reinforcement Learning

**arXiv ID:** 2607.28127 | [PDF](https://arxiv.org/pdf/2607.28127v1)

**作者:** Giorgos Iacovides `[一作]` (Imperial College London), Danilo Mandic `[通讯]` (Imperial College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出 FinSMART，一种将情感预测与实际市场收益直接对齐的强化学习框架，能够利用金融文本和实时行情来训练情感模型，并支持持续自适应更新。

**💡 创新点**

创新点包括：①双滤波交易奖励机制，将情感预测与满足 alpha 与方向两项条件的实际收益相结合；②利用 GRPO 在无价值网络的情况下实现稳定的 RL 学习；③支持无人工标注的持续再训练，消除了对静态市场无关数据集的依赖。

**🔧 技术方法**

技术方法包括基于 Llama‑3‑8B‑Instruct 的因果 LLM、低秩适配 LoRA、Group Relative Policy Optimization（GRPO）、市场对齐的数据过滤管线以及离散非对称交易奖励函数。

**📊 数据集**

使用 2015‑2021 年的 The Motley Fool 与 MarketWatch 财报新闻与 Yahoo Finance S&P‑500 日行情数据，约 325,000 条新闻，训练集约 30,000 条。

**📈 对比分析**

在 2021 年后 2.5 年的真实交易回测中，与词典、SFT LLM、FinDPO 等六种基线相比，FinSMART 在累计回报、年化收益、Sharpe、Sortino、Calmar 以及 RankIC 上均实现显著提升（累计回报 264.9%→406.2%，年化收益 91.5%→125.7%，RankIC 0.061→0.065）。

**⚠️ 局限性**

局限性包括：奖励仍基于同日 alpha，可能忽略事件顺序与时延效应；对同一交易日多条新闻的聚合处理较为粗略；需要 GPU 资源，且在不同市场环境下的泛化性能尚待进一步验证。

---

## 400. MIND: Lightweight and Effective Memory Injection Defense for LLM Agents via Intent-Aware Information Bottleneck

**arXiv ID:** 2607.28103 | [PDF](https://arxiv.org/pdf/2607.28103v1)

**作者:** Dongyi Liu `[一作]` (Hong Kong University of Science and Technology), Jia Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 MIND（Memory Intent‑Aware Neural Denoising），一种轻量级的记忆注入攻击防御框架，能够在多轮对话中过滤恶意记忆并保持任务准确性。

**💡 创新点**

创新点在于：① 通过信息瓶颈（IB）提取初始用户意图与后续行为的关系，去除多轮冗余信息；② 使用多超平面分类器构建更具表达力的决策边界；③ 只需一次记忆检索过滤，无需反复调用 LLM 进行审计，显著降低计算成本。

**🔧 技术方法**

技术手段包括：预训练 LLM（如 Llama‑3.1‑8B‑Instruct）作为特征提取器；信息瓶颈理论与变分 IB 编码器；多超平面（Convex Polytope Machine）分类头；t‑SNE 可视化与对比实验。

**📊 数据集**

使用的数据集有：StrategyQA（ReAct 轨迹）、MMLU、EHR 轨迹，以及公开的 QA 轨迹，用于跨域训练与评估。

**📈 对比分析**

与 A‑MemGuard、LLM Auditor、Distil、PPL、Sequential Monitor、AV Filter 等基线对比，MIND 在 ASR‑r、ASR‑a（攻击成功率）上显著降低（约 55%），任务准确率基本保持；与 LLM Auditor 相比，推理时间提升 20%~70%，并在四种主流后端（DeepSeek‑V4、GPT‑4o‑mini、Llama‑3.1‑8B‑Instruct、Qwen3‑8B‑Instruct）均表现优异。

**⚠️ 局限性**

局限性包括：① 需要预训练 LLM 的特征表示，对模型内存访问权限有限的闭源后端仍需代理；② IB 与对齐系数的调参对性能影响较大，需经验选择；③ 对极端或未知攻击模式的鲁棒性尚未系统验证，未来需在更大规模分布式记忆系统上进一步测试。

---

## 401. SciDataSailor: Deep Scientific Data Exploring

**arXiv ID:** 2607.28098 | [PDF](https://arxiv.org/pdf/2607.28098v1)

**作者:** Jiyong Rao `[一作]`, Runkai Zhao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SciDataSailor框架，用于在真实科学数据仓库中构建可执行的探索轨迹，并基于该框架创建了SciDataSailor-Bench和SFT-2K两套数据集；

**💡 创新点**

创新点在于结合双反馈首发紧迫度的MCTS、层级策略到工具动作生成以及熵导向分支，实现了既能广泛探索又能有针对性利用证据的长时限探索；

**🔧 技术方法**

主要技术包括：Monte Carlo Tree Search、双反馈首发紧迫度(DF-FPU)、层级策略生成、熵导向分支、执行验证与幻觉检测、可执行工具调用与ReAct式代理；

**📊 数据集**

使用了27个多模态科学数据集（覆盖生命、地球与物理科学），生成了627条元信息摘要任务与586条证据驱动问答任务；

**📈 对比分析**

通过统一可执行代理协议对比了多款专有模型与开源模型，发现专有模型在12步预算下Pass@1可达65%，开源模型仅约27%；在QA轨道上开源模型可逼近专有水平；对Qwen3.5-9B进行SFT后，Pass@1提升至约29%，成功率翻倍，且平均步数从10.21降至6.24；

**⚠️ 局限性**

局限包括：对极端多文件嵌套的适应性仍有限；幻觉与执行错误仍有一定比例；训练数据偏向已知任务，难以泛化到完全未知的仓库结构；

---

## 402. On The Most Discriminative Boolean Functions for Correlated Sources

**arXiv ID:** 2607.28162 | [PDF](https://arxiv.org/pdf/2607.28162v1)

**作者:** Jun Chen `[一作]`, Lei Yu `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

本文研究在两源相关的二进制信号上使用布尔函数压缩后，如何选择函数对以最大化输出分布之间的KL散度（即最具判别性的布尔函数）。

**💡 创新点**

创新点在于：① 提出了“level‑k”函数（傅里叶系数仅支持于某一阶k）的最优性证明，部分解决了Amari‑Kobayashi关于Fisher信息最大化的猜想；② 在Bayesian非交互式一比特假设检验框架下证明level‑k函数在所有布尔函数中取得最优错误率；③ 通过联合凸性、噪声算子和傅里叶分析等工具，给出了KL散度与Fisher信息之间的严格关系。

**🔧 技术方法**

主要技术：傅里叶分析（布尔空间上的正交基、噪声算子、平衡性与Parseval恒等式）；KL散度的联合凸性和泰勒展开；对Fisher信息的二阶导数表达；马尔可夫链数据处理不等式；以及对最大相关差异（MCD）的奇异值分解。

**📊 数据集**

该工作完全基于理论分析，不涉及任何外部数据集。

**📈 对比分析**

通过构造上界和下界，证明在无偏布尔函数或相同函数的情况下，level‑k函数能达到KL散度和Fisher信息的最大值；在Bayesian检验中，同样给出最优错误率的闭式上界，并指出level‑k函数可以实现该上界。没有实验性能指标，评估完全以数学证明为准。

**⚠️ 局限性**

局限性：① 只在无偏或同函数的特殊情形下证明最优性，通用情形（有偏且两函数不相同）仍未完全解决；② 对于某些参数组合，可能存在更优的非level‑k函数，但目前缺乏全局最优证明；③ 该方法主要适用于二进制源，推广到多进制或非均匀分布仍是未来工作。

---

## 403. A Mathematical Framework for Reading the Autopsias' Meta - Compositional System

**arXiv ID:** 2607.28155 | [PDF](https://arxiv.org/pdf/2607.28155v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 404. Asymmetric Communication: Large Language Models and Language Games

**arXiv ID:** 2607.28137 | [PDF](https://arxiv.org/pdf/2607.28137v1)

**作者:** Enzo Fenoglio `[一作]` `[通讯]` (University College London), Enzo Fenoglio (University College London)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

阐述大语言模型（LLM）交互中的非对称通信结构，指出人类与机器在意义、责任与规范性评估上的差异，并重新解读 AGI、幻觉、代理性、情感投射与对齐等热点话题。

**💡 创新点**

创新点在于将维特根斯坦、卢曼、埃斯波西托与布兰多姆四位哲学家的理论综合成“非对称通信”框架，清晰区分了通信完成与规范性参与两层结构，并用此框架重新解释当代 AI 争论中的概念误区。

**🔧 技术方法**

未采用具体技术实现或实验方法；本文主要基于哲学与理论分析。

**📊 数据集**

未使用任何数据集；论文不涉及实验验证。

**📈 对比分析**

比较方法：以传统代表主义/计算主义假设和常见的 AI 目标同步论述为对照，阐释其在本框架下的缺陷；并对不同话题的重新表述做概念对比，未给出量化性能指标。

**⚠️ 局限性**

局限性：缺乏实证检验，无法说明未来技术进步或社会实践变迁是否会改变非对称通信结构；同时对实际治理、系统设计的指导性仍需进一步研究。

---

## 405. BlueprintRepair: Typed Local Edits for Failed Lean Proof Blueprints

**arXiv ID:** 2607.28110 | [PDF](https://arxiv.org/pdf/2607.28110v1)

**作者:** Ruslan Khrulev `[一作]` `[通讯]` (Lomonosov Moscow State University), Ruslan Khrulev (Lomonosov Moscow State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于十种类型化操作的 Lean 蓝图本地修复接口，并在 142 个受控失效蓝图上评估其在 Lean 证明修复中的有效性。

**💡 创新点**

创新点包括：① 用类型化本地编辑和图依赖检查保证蓝图完整性；② 构建了包含 142 个人工注入缺陷的 miniF2F 基准；③ 在相同预算下对比三种修复接口（类型化、局部补丁、完整重写）的覆盖率与成本。

**🔧 技术方法**

技术手段包括 Lean 4 的 LeanArchitect 依赖图、JSON schema 校验、LLM（DeepSeek-V4-Flash 与 Qwen3.6-Flash）交互接口、自动化检查与反馈循环。

**📊 数据集**

使用 miniF2F 141 个目标定理，人工构造并注入 142 个失效蓝图（编辑型、证明型、复合型），并定义了 10 种可执行的修复操作。

**📈 对比分析**

比较方法：在匹配的 token 与成本预算下，对三种接口进行单次实验，记录解决状态数、覆盖率与平均成本。结果显示：在编辑型失效上，类型化与补丁相近，但类型化在 10k token 内更快、更低成本；补丁在总体覆盖率上略占优。

**⚠️ 局限性**

局限性：图规模仅到 8 个节点，缺陷类型受限；实验为单次采样，未评估多次尝试的期望表现；接口差异多方面难以单独归因；基准数据来自 miniF2F，可能不代表真实生成失败。

---

## 406. Convolutional Neural Shading for High-Quality 3D Reconstruction from Multi-View Images

**arXiv ID:** 2607.28132 | [PDF](https://arxiv.org/pdf/2607.28132v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 407. Beyond Rephrasing: Book-Level Organization Improves Synthetic Textbook Data for Mid-Training

**arXiv ID:** 2607.28109 | [PDF](https://arxiv.org/pdf/2607.28109v1)

**作者:** Jiawen Tao `[一作]` (Tencent), Maxm Pan `[通讯]` (Tencent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了基于检索的教材合成管道，生成了686K本约32B词的书籍，并用其对比评估书级组织对中训练的影响。

**💡 创新点**

将检索、聚类、TOC规划与层级生成结合，强调书级结构而非单句重写，并系统验证结构化教材在训练中的价值。

**🔧 技术方法**

使用关键词检索、文本嵌入+KMeans聚类、LLM（DeepSeek-V3.2、Qwen3.5-35B-A3B、Gemini-3.1-Pro）进行TOC规划、质量门控与章节生成，混合模型训练与多场景对比。

**📊 数据集**

对大规模预训练语料做检索，按15,000+学科目录生成查询，合成书籍，使用公开benchmark 28个（STEM、知识、推理、代码、跨语言）评估。

**📈 对比分析**

通过 Full、Split、RandomConcat、Rephrase、Natural Books 等对照组，固定 token 预算与内容，发现 Full 在 28 项基准平均提升 1.09 分，并在 3B MoE 和 Llama3‑8B 上均显著优于对照。

**⚠️ 局限性**

未测量训练跑动方差、需要可检索语料索引、依赖 LLM 判断做为组件评估，且仅在两种模型架构验证。

---

## 408. OPLD: On-Policy Latent Distillation for Multimodal Reasoning

**arXiv ID:** 2607.28154 | [PDF](https://arxiv.org/pdf/2607.28154v1)

**作者:** Shoutai Zhu `[一作]` (ByteDance), Qinzhen Guo `[通讯]` (ByteDance)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于教师-学生对抗的“在策略”潜在蒸馏框架 OPLD，将多模态 Chain-of-Thought（CoT）中抽象的推理过程转移到模型的潜在空间，使模型在推理时无需显式视觉轨迹即可完成推理。

**💡 创新点**

创新点在于：①引入在策略潜在蒸馏，利用教师的多模态 CoT 生成的潜在轨迹对学生进行对齐；②使用潜在编码‑解码器将隐藏状态映射到更紧凑的潜在空间；③通过 token‑级前向 KL 与潜在余弦对齐两种损失，兼顾过程级监督与抽象状态迁移。

**🔧 技术方法**

核心技术包括：多模态链式思考、潜在编码‑解码器（Enc‑Dec）适配器、在策略教师‑学生对齐、前向 KL 与潜在余弦对齐、递归潜在推理（K 步潜在槽）等。

**📊 数据集**

使用了清洗后的 Zebra‑CoT 与 Visual‑CoT 训练集（约 217K 样本），并在 V^⋆、HRBench‑4K/8K、MMStar、SeedBench2‑Plus、BLINK、HallusionBench 等七个多模态推理基准上进行评估。

**📈 对比分析**

与 Qwen2.5‑VL‑7B、SFT 版本、现有视觉‑潜在方法（LVR、Laser、SkiLa、Monet、HyLaR）以及思考‑图像代理模型（ZoomEye、Thyme、DeepEyes）相比，OPLD 在大多数基准上实现了显著提升（如 V^⋆ 从 71.2% 提升至 85.9%，HRBench‑4K 从 65.1% 提升至 73.8%），并在多模态推理任务上达到了或接近当前最先进水平。

**⚠️ 局限性**

主要局限性包括：①依赖大规模多模态 CoT 训练数据，数据清洗与筛选成本高；②教师‑学生对齐仍可能出现教师信息泄漏或分布漂移问题；③对实时或动态视觉推理的适应性尚未充分验证，且在更开放式的多模态任务上可能需要进一步的通用性研究。

---

## 409. SmartGen: Seamless Disaggregated LLM Inference with Selective KV Cache Transfer

**arXiv ID:** 2607.28150 | [PDF](https://arxiv.org/pdf/2607.28150v1)

**作者:** Xuchuan Luo `[一作]` (Fudan University), Yangfan Zhou `[通讯]` (Fudan University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在自托管云实例上将预填充与解码节点分离，提出了Selective KV Cache Transfer Engine，实现KV缓存的选择性传输，显著降低网络瓶颈

**💡 创新点**

创新点在于结合离线KV重要性定位、实时掩码拆分与重排聚合的三路传输（主动、并行按需、预期传输），实现KV传输与预填计算重叠且不影响模型准确率

**🔧 技术方法**

采用动态KV稀疏选择、离线定位、RDMA一边读写、KV掩码矩阵、索引拆分、KV重排、门铃批处理等技术

**📊 数据集**

使用LongBench长上下文基准（MultiFieldQA、GovReport、SAMSum、LCC），并用2WikiMultihopQA做离线校准，模型包括Qwen3、Llama‑3.1、Gemma‑3、Phi‑4

**📈 对比分析**

与完整KV传输、仅前缀传输、HACK量化方案、InfiniGen/HATA动态选择等对比，Selective KV在TTST上最高可缩短4.3×，TBT接近理想，整体CTL最低；在准确率上与全缓存相当，低于HACK

**⚠️ 局限性**

局限性：需要离线profiling与周期性更新；对GDR/RDMA依赖，缺失时需额外复制；在极低带宽或高延迟环境下仍有残留RTT；参数调优（speculative ratio、block大小）影响性能

---

## 410. ConMem: Contribution-Aware Memory for Long-Horizon Manufacturing Inspection Logs

**arXiv ID:** 2607.28126 | [PDF](https://arxiv.org/pdf/2607.28126v1)

**作者:** Bingchen Liu `[一作]` (Shandong University), Xiangtian Meng `[通讯]` (Rizhao Steel Holding Group Co., Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 ConMem 贡献意识记忆框架，使 LLM 能在钢铁设备检验中实现长周期早期风险筛查。

**💡 创新点**

创新点包括功能角色感知分段、基于 Shapley 的记忆贡献估计以及优先级存储策略，能自适应保留稀缺关键信号。

**🔧 技术方法**

采用功能角色分段、Shapley 价值估算、max‑heap 优先存储以及检索增强生成的 LLM 技术。

**📊 数据集**

使用真实钢铁制造业检验记录约 30,000 条的工业数据集。

**📈 对比分析**

与 Naive 8K、Full RAG、Rule‑based 等多种基线对比，ConMem 在准确率上达 76%（比全 RAG 高约 4%），令 token 量和响应时间分别缩减 88% 与 87%。

**⚠️ 局限性**

局限在对高质量标签的依赖、对 LLM 能力的敏感度以及在更大规模或多工况场景下的可扩展性待验证。

---

## 411. Towards Practical Algorithm Selection for Unsupervised Domain Adaptation in Medical Imaging

**arXiv ID:** 2607.28125 | [PDF](https://arxiv.org/pdf/2607.28125v1)

**作者:** Yiheng Xiong `[一作]` (Ulm University Medical Center), Michael Götz `[通讯]` (Ulm University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出一种无标签的两层一致性参考框架，用于联合选择医学图像UDA中的算法与超参数。

**💡 创新点**

创新点在于通过多个验证器在每个算法内部挑选最佳检查点，再在不同算法间通过投票构建一致性参考，避免单一验证器偏差，实现跨算法的高效无标签选择。

**🔧 技术方法**

使用的技术包括UDA算法集合（如MMD、DANN、CDAN等）、多种无标签验证器（InfoMax、Source‑Risk、DEV‑N等）、投票一致性参考构造、以及基于预测一致度的评分机制。

**📊 数据集**

实验数据集包括四个脑MRI数据集（ADNI‑1/2/3、AIBL）和四个胸部X射线数据集（RSNA、Child CXR、LDD、CRD），共七个临床相关转移场景。

**📈 对比分析**

与单个验证器和其他无标签评估方法比较，本文方法在七个场景下平均准确率达86.3%，比最佳单验证器提升约5.3%，与Oracle的差距从10.4%降至5.1%。

**⚠️ 局限性**

局限性包括仍存在与Oracle的性能差距；需要先训练多种UDA算法和超参数组合，计算成本高；实验仅针对二分类与平衡准确率，未覆盖多分类、分割或其他临床指标。

---

## 412. Powering Net-Zero 6G: Packetized Energy Management for Grid-Interactive Telecom Infrastructure

**arXiv ID:** 2607.28111 | [PDF](https://arxiv.org/pdf/2607.28111v1)

**作者:** Adnan Aijaz `[一作]` (Toshiba Europe Ltd), Xinyi Lin `[通讯]` (Toshiba Europe Ltd)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并评估基于包化能源管理（PEM）的6G基站与网络架构，以及基于此的电信虚拟电厂（VPP）概念，旨在实现网际零排放与能量弹性。

**💡 创新点**

创新点在于将PEM迁移至电信RAN层，分离关键通信与可弹性负载，构建网际VPP平台，并通过仿真验证其在能耗、碳排放、成本与服务连续性方面的提升。

**🔧 技术方法**

使用包化能源管理（PEM）技术、离散包请求与接受、基于Open RAN的SMO/Non-RT RIC接口、AI/机器学习预测与调度、以及VPP优化算法。

**📊 数据集**

使用基于宏站的流量与能耗模型、20个6G基站的功率/负载/PV/储能参数、24小时5分钟分辨率的时变电网碳强度与电价数据、以及仿真生成的出力与存储轨迹。

**📈 对比分析**

通过与无PEM基准情景对比的仿真，结果显示在无DER、保守DER和净零DER三种场景下，PEM分别降低能耗11.35%、碳排放13.84%、运营成本16.45%，并在停电测试中提升运行时长至37.5%及关键UE服务时间至40.58%。

**⚠️ 局限性**

局限性包括对实际现场验证缺失、对多站同步与标准化接口未完善、AI预测不确定性与可解释性不足、以及市场参与模型与计价机制仍需进一步研究。

---

## 413. ESBT: A Scalable and Deterministic Sequence CRDT for Distributed Collaborative Editing

**arXiv ID:** 2607.28101 | [PDF](https://arxiv.org/pdf/2607.28101v1)

**作者:** Moulay Driss Mechaoui `[一作]` (University of Mostaganem - Abdelhamid Ibn Badis), Abdessamad Imine `[通讯]` (University of Lorraine)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于扩展 Stern–Brocot 树的序列 CRDT（ESBT），通过分层标识符分配实现可扩展、确定性排序并无墓碑删除；

**💡 创新点**

创新点在于将 Stern–Brocot 分数、序列号和可变深度路径三层耦合，既保证了标识符唯一性、严格排序，又通过阈值限制分数的分母/分子大小，实现了标识符增长可控；

**🔧 技术方法**

核心技术包括分层标识符分配算法、Red‑Black 树文档表示、轻量级基于操作依赖的同步协议；

**📊 数据集**

实验使用合成工作负载，最多 100,000 次并发操作，覆盖 50 个协作站点，基准比较 Logoot 与 LSEQ；

**📈 对比分析**

通过测量响应时间（ms）和标识符内存占用（MB）进行比较，ESBT 在所有插入模式下平均提高 28–88% 响应速度、降低 50–75% 内存占用，在极端中间插入场景下提升 86% 以上；

**⚠️ 局限性**

局限性包括需手动调节 D_max、分数基数及深度参数；标识符在高并发冲突时仍会产生可变长度路径，影响序列化开销；实现细节和参数敏感度尚待进一步研究。

---

## 414. PCAP-LM: An LLM-Native Text Representation for TLS Bulk Traffic Analysis

**arXiv ID:** 2607.28100 | [PDF](https://arxiv.org/pdf/2607.28100v1)

**作者:** Xavier Marjou `[一作]` (Orange), Ilan Jaffeux-Cheniout `[通讯]` (Orange)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `fede83ac-7505-405f-ab37-e7284695c47f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种面向大型语言模型的文本格式PCAP‑LM，将网络捕获文件压缩为可直接被LLM解析的语义丰富文本，支持流级摘要、事件流和异常注解；

**💡 创新点**

创新点包括：1）基于ASCII的PacketGlyphs字母表，精简编码包方向、TCP/TLS状态、大小、延迟等信息；2）受限PMI‑BPE训练，生成行为模式词汇并避免跨流合并；3）对相同行为模式的RLE压缩；4）四层结构文档（会话头、流摘要、事件流、异常注解）与可恢复的@REFS索引；

**🔧 技术方法**

使用技术有：PacketGlyphs编码、受限PMI‑BPE分词、RLE、BPE、Scapy解析网络层、tshark、Wireshark、Python实现转换与索引、Claude Sonnet 4.6等LLM进行问答评估；

**📊 数据集**

数据集为150对5G/4G HTTPS TLS 1.3 bulk‑download PCAP（共301文件），其中30对（60文件）作为测试集，平均文件大小3.4 MB，包含1–3条TLS流；

**📈 对比分析**

与tshark‑V、tshark‑T json、原始PCAP、gzip比较，PCAP‑LM+RLE+BPE平均token约23 k，压缩率812×；在30个测试文件的问答任务中，LLM在PCAP‑LM文档上取得99.3 %准确率，基线（截断的tshark‑V前缀）仅51.0 %；

**⚠️ 局限性**

局限性包括：对TCP重传检测漏报率约24 %（因单一序列号检测失效）；仅在同质HTTPS bulk‑download环境下训练，异构协议需要重新训练BPE；基线比较受截断影响；缺乏对非TLS或多协议场景的支持；需要进一步验证与细粒度drill‑down功能。

---

## 415. Distilling Answer Set Programming Theories from Large Language Models

**arXiv ID:** 2607.28086 | [PDF](https://arxiv.org/pdf/2607.28086v1)

**作者:** Nelson Higuera Ruiz `[一作]` (ExtensityAI), Claudiu Leoveanu-Condrei `[通讯]` (ExtensityAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文探讨了利用大型语言模型（LLM）在与ASP求解器闭环的代理环境中，从空文件开始逐步构建完整、准确的ASP推理理论。

**💡 创新点**

创新点在于提出一种统一的、数据集无关的“蒸馏协议”，让LLM在无模板、无参考规则的情况下，仅通过解析训练实例、调用求解器和自我修正，自动生成覆盖整个推理逻辑的ASP程序。

**🔧 技术方法**

主要技术包括基于OpenCode的LLM代理工具链、clingo求解器的交互式反馈、批量训练/验证分割与自动化评估，以及对不同规模LLM模型的对比实验。

**📊 数据集**

实验所用的数据集为视觉问答（VQA）领域的三个公开基准：CLEVR、GQA和CLEVRER，分别涵盖合成场景、真实图像与短视频的复杂推理任务。

**📈 对比分析**

评估方法为在每个数据集上对验证集（200例）计算准确率，并与手工编写的ASP参考理论对比；结果显示，四大前沿模型中三者在CLEVR达到100%准确率，在GQA达到92.8–98.8%，在CLEVRER达到92.7–95.3%；相比之下，GPT‑5在GQA仅达41.8%，并在给定参考理论时表现下降。

**⚠️ 局限性**

主要局限包括：GPT‑5对参考理论的负面敏感性导致性能退化；低于27B参数的模型普遍出现语法解析或工具调用错误，导致理论生成失败；总体上，代理循环在某些模型上并未提升效果，反而产生负面影响。

---

## 416. GGC: Selective Query Correction for Reliable Text-to-SPARQL Generation

**arXiv ID:** 2607.28082 | [PDF](https://arxiv.org/pdf/2607.28082v1)

**作者:** Ziyi Yang `[一作]` (Nanyang Technological University), Lihui Chen `[通讯]` (Nanyang Technological University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Generator–Gate–Corrector (GGC) 框架，对 LLM 生成的 Text‑to‑SPARQL 进行可可靠的选择性纠错，先生成初始查询，再通过门控判断是否需要纠正，最后只对高风险查询调用纠错器；

**💡 创新点**

创新点在于引入门控机制实现“detect‑then‑correct”策略，既提升语义一致性，又显著降低不必要的推理开销；

**🔧 技术方法**

使用 Llama‑3.2‑3B‑Instruct 进行监督微调，配合 LoRA 与 4‑bit 量化，门控采用 RoBERTa‑based 二分类器，纠错器同样以 LLM 微调；

**📊 数据集**

在电影领域的 MCQA 数据集上评估，附加对 SciQA 的预实验验证；

**📈 对比分析**

相较于 Generator‑only（90.23%）和全量纠错（92.34%）的基线，GGC 达到 98.33% 查询级准确率、99.16% 项级 F1，推理时间下降约 45%；

**⚠️ 局限性**

局限性包括：仅在 MCQA 电影领域验证，门控/纠错器对特定 Generator 的错误分布依赖，训练时需额外生成查询导致离线成本高，跨域或 ID‑based SPARQL 的通用性尚未充分验证。

---

## 417. Group-Reflective Self-Distillation for Agentic Reinforcement Learning

**arXiv ID:** 2607.28076 | [PDF](https://arxiv.org/pdf/2607.28076v1)

**作者:** Binbin Zheng `[一作]` (University of Science and Technology of China), Zeyu Chen `[通讯]` (Baidu Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为GRSD（Group-Reflective Self-Distillation）的框架，用于在稀疏结果奖励下进行代理强化学习。该方法通过对已验证的策略轨迹进行反思，构建出群体级的特权指导。

**💡 创新点**

GRSD的创新点在于利用策略自身的验证轨迹进行反思，并对成功和失败的反思进行对比，从而形成能力对齐和结果区分的指导，而无需依赖外部模型生成技能。

**🔧 技术方法**

使用了自我蒸馏机制，通过群体反思指导来细化轨迹级优势为回合特定的信用，同时保持验证者确定的学习方向。

**📊 数据集**

在多个代理环境和模型规模上进行了实验，具体数据集包括ALFWorld、基于搜索的问答和WebShop等。

**📈 对比分析**

与多种基线方法进行比较，GRSD在所有环境和模型上均表现出色，尤其在ALFWorld上比GRPO提高了5.5%，在WebShop上提高了4.8%的成功率，显示出更强的泛化能力。

**⚠️ 局限性**

限制在于该方法依赖于策略自身的验证轨迹，可能在某些情况下无法充分捕捉到所有成功和失败的行为模式。

---

## 418. S-Avatar: Diffusion-Guided Gaussian Head Avatars from a Single Image

**arXiv ID:** 2607.28164 | [PDF](https://arxiv.org/pdf/2607.28164v1)

**作者:** Hail Song `[一作]` (KAIST), Woontack Woo `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于单张图像的 3D 头部化身重建框架 S-Avatar，先通过扩散模型生成 3D 高分辨率高斯斑点（3DGS），再将 FLAME 参数模型对齐并通过绑定模板驱动 3DGS 的变形，实现可动画、可从任意视角渲染的高质量头部化身。

**💡 创新点**

创新点在于：①将扩散生成的 3DGS 与可控的 FLAME 参数模型分离，形成三阶段管线（生成、拟合、绑定），避免了直接优化 NeRF/3DGS 造成的视角不一致；②设计了基于逆距离加权的绑定模板和高斯尺度自适应机制，实现细腻表情变形；③采用 COO 格式稀疏矩阵显著提升绑定与渲染速度，支持实时渲染。

**🔧 技术方法**

核心技术包括：扩散式 3DGS 生成（类似 LGM），FLAME 头部参数化模型，Chamfer + 标定点对齐损失的 FLAME 拟合，逆距离加权绑定模板，高斯尺度自适应，COO 形式稀疏矩阵的绑定运算。

**📊 数据集**

主要使用 NeRSemble 数据集（164 名受试者，16 个视角，含多表情）进行量化评估，并在真实世界的单张手机照片上演示泛化能力。

**📈 对比分析**

与 Rome、P4Dv1/2、Voodoo、GPAvatar 等前沿单图像头像重建方法进行对比，评估指标为 LPIPS、PSNR、SSIM。S-Avatar 在 LPIPS 0.258、PSNR 16.10、SSIM 0.826 上均优于对照组，尤其在极端新视角（侧面/后视）和交叉表情迁移时表现更佳。

**⚠️ 局限性**

局限性在于整个流程高度依赖初始 3DGS 生成质量；若扩散生成的高斯斑点存在误差，后续拟合与绑定难以弥补；此外，仍需要预训练扩散模型和 FLAME 参数估计，对资源与训练数据的依赖较大。

---

## 419. RRM: Experience-Driven Reflective Retrieval Memory for Long-Horizon Multimodal Reasoning

**arXiv ID:** 2607.28156 | [PDF](https://arxiv.org/pdf/2607.28156v1)

**作者:** Jingxiang Fan `[一作]` (University of Science and Technology Beijing), Bochao Zou `[通讯]` (University of Science and Technology Beijing)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Reflective Retrieval Memory (RRM) 框架，利用历史任务轨迹中可迁移的检索经验，对实体中心的多模态长期记忆进行反射式改进，仅将经验作为查询级别的检索控制信号，避免将历史事实直接注入答案生成，并实现在线生命周期管理以维护经验质量。

**💡 创新点**

①从成功与失败轨迹中提炼可迁移的检索流程经验，而非事实内容；②将经验限定为查询控制，防止历史实体与答案干扰当前推理；③通过在线查询反射（OQR）即时诊断并修正检索错误；④对经验进行使用频率、反馈和时间衰减的动态管理，减少冗余与噪声。

**🔧 技术方法**

基于 M3‑Agent 的实体中心多模态记忆图（事件记忆与语义记忆）；Search–Answer 控制器；Qwen text‑embedding‑v3 编码查询；大型语言模型评估；在线查询反射与经验选择器；结构化检索经验记录；在线生命周期管理机制；按批次延迟反馈的在线适配协议。

**📊 数据集**

M3‑Bench‑Robot、M3‑Bench‑Web 与 Video‑MME‑Long 三大长视频多模态推理基准。

**📈 对比分析**

在与通用多模态 LLM、在线长视频理解方法以及基于代理的长期记忆方法的对比实验中，RRM 在三大基准上分别提升 9.1%、5.8% 与 7.4% 的总体准确率，并显著减少检索轮数（约 17%–26%）。

**⚠️ 局限性**

依赖于准确的失败检测与经验提炼，若检索轨迹质量低或经验与新任务不匹配，可能导致误导；经验管理仍需平衡容量与更新频率；目前仅关注检索流程的迁移，对知识迁移与跨域泛化的能力尚未充分验证。

---

## 420. Agent Harness Distillation: Inference-Time Harness Extraction and Exploitation in Autonomous Multi-Agent Systems

**arXiv ID:** 2607.28147 | [PDF](https://arxiv.org/pdf/2607.28147v1)

**作者:** Yu Cui `[一作]` (Baidu Inc.), Chenfu Bao `[通讯]` (Baidu Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `8d10c613-917e-4880-9716-17789f50e119` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并提出了 Agent Harness Distillation（AHD）框架，用于在黑盒交互中提取并转移自主多智能体系统的推理时 Harness。

**💡 创新点**

创新点在于将知识蒸馏扩展到系统级推理时 Harness，设计双阶段提取方法并提出基于欺骗的防御。

**🔧 技术方法**

采用黑盒采样、行为推断、迭代优化、知识蒸馏技术以及欺骗防御机制。

**📊 数据集**

使用 Claude Code、Hermes 真实 AMAS，背后的 LLM 包括 Qwen3.6‑Flash、Qwen3‑80B、DeepSeek‑V3、GPT‑5.4，评测集包括 AIME2025、GSM‑Level6、GAIA 以及 MMLU‑Pro CS 子集。

**📈 对比分析**

相较基线，预蒸馏提升约 2.5%，后蒸馏提升 45%；在弱 LLM 上显著提升性能；防御能显著降低提取成功率，同时保持大部分任务准确率。

**⚠️ 局限性**

局限在于只针对单一目标系统，未探讨多教师融合或自我蒸馏风险；实验仅覆盖部分 LLM 与任务，防御效果受实现细节影响。

---

## 421. Face and Voice Cross-modal Association with Learning Convex Feature Embedding

**arXiv ID:** 2607.28129 | [PDF](https://arxiv.org/pdf/2607.28129v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 422. Can Agents Deceive? Evaluating Reasoning and Deception in ParliamentBench using a Social Deduction Game

**arXiv ID:** 2607.28146 | [PDF](https://arxiv.org/pdf/2607.28146v1)

**作者:** Niklas Bauer `[一作]` (University of Göttingen), Terry Ruas `[通讯]` (University of Göttingen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用秘密希特勒（Secret Hitler）这一社交推理游戏，对16种大型语言模型（LLMs）在欺骗、说服和推理等能力进行基准评估。

**💡 创新点**

提出三项新的细粒度指标（GSIR、RIA、DRR），并构建了可复现的多智能体模拟环境和评价工具，成为首个针对LLMs在隐藏目标和信息不对称情境下的系统性基准。

**🔧 技术方法**

采用规则驱动的Python仿真框架、自然语言推理与决策逻辑（如私有历史、对话生成）、以及自定义评价指标来评估模型行为。

**📊 数据集**

实验数据包括1,600余局LLM对局（每种模型100局）、与人类玩家的对局以及25,000局公开人类游戏数据。

**📈 对比分析**

与人类、随机和算法基准对比；最强模型（GPT‑5.4、Kimi K2.5、Grok 4.1 Fast、DeepSeek 3.1 Terminus）在合作与欺骗角色均表现突出，胜率高于随机（33%）和算法基准（45%），但多数模型在保持一致的欺骗角色上表现不足，欺骗保持率低于50%。

**⚠️ 局限性**

局限性包括：游戏环境为简化的人工控制情境，难以直接映射到高风险现实场景；人类评估样本有限（5局）；未充分探究未对齐模型；固定发言顺序限制了自然互动。

---

## 423. An Instrument to Evaluate Governance Proposals: AI Policy Analysis at Scale

**arXiv ID:** 2607.28094 | [PDF](https://arxiv.org/pdf/2607.28094v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 424. Advancing Awkward Arrays for High-Performance CPU and GPU Processing

**arXiv ID:** 2607.28145 | [PDF](https://arxiv.org/pdf/2607.28145v1)

**作者:** Ianna Osborne `[一作]` (Princeton University), Manasvi Goyal `[通讯]` (Harvard University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

实现了 Awkward Array 在 CPU 与 GPU（CUDA+CCCL）上的高性能执行，提供统一的 Python 接口并支持多种不规则数据操作。

**💡 创新点**

主要创新点包括：将 NVIDIA 的 CCCL 分段算子整合进 GPU 后端，显著简化 CUDA 核心代码；自动化的跨后端验证框架；针对不规则分段归约的内存管理与并行化优化。

**🔧 技术方法**

技术手段包括：CPU 后端 144 个专用核；CUDA 后端利用 CuPy 进行内存管理与核启动；使用 CCCL 的分段最小/最大/和/乘积/计数原语；多阶段执行管线（构建段偏移、临时缓冲、调用 CCCL、写回）。

**📊 数据集**

采用自定义的基于变长段的数值数组作为基准数据集，模拟高能物理事件结构，覆盖从数千到数百万段的规模。

**📈 对比分析**

比较方法：在 AMD EPYC 7H12 CPU + NVIDIA A100 GPU 上，分别测量 Awkward CPU、Awkward GPU、NumPy、CuPy 的执行时间与峰值内存。GPU 版在最大规模下比 CPU 快 5.3×，低级 CCCL 原语接近 6×；与 NumPy 比，GPU 最快时可达 4×；与手写 CuPy 对比，GPU 仍略慢，但差距随规模增大而缩小。

**⚠️ 局限性**

局限性：高层 Awkward 执行管线导致多次 CUDA 核启动产生额外开销；GPU 临时缓冲占用显存比 CPU 高；需要跟进 CCCL API 的演进以保持兼容性。

---

## 425. Extended Depth-First Representations of $k^2$-trees

**arXiv ID:** 2607.28136 | [PDF](https://arxiv.org/pdf/2607.28136v1)

**作者:** Gabriel Carmona `[一作]` (University of Pisa), Francesco Tosoni `[通讯]` (Sant'Anna School of Advanced Studies)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对 k^2‑tree 进行静态无损压缩，关注缓存局部性，提出四种深度优先布局并实现其压缩与运算。

**💡 创新点**

创新点在于：①用深度优先序重排 k^2‑tree，显著提升缓存局部性；②引入平衡括号（BP）及其压缩版本 CBP；③设计线性时间的后缀数组+LCP 方法自动识别并压缩相同子树；④在已知的 DFUDS 方案上进一步优化。

**🔧 技术方法**

技术手段包括：k^2‑tree、深度优先遍历、平衡括号编码、DFUDS、后缀数组与 LCP、跳跃值（skip values）与变长编码（Elias、Variable‑Byte）等。

**📊 数据集**

实验数据集：WebGraph（7 个大规模网页图）、Database（Wikidata 关系图）、Random（大小为 1000 的随机稀疏矩阵）。

**📈 对比分析**

比较方法：测量磁盘空间、执行时间、峰值内存，针对矩阵‑向量乘、矩阵‑矩阵加、矩阵‑矩阵乘三种线性代数运算。实验表明，深度优先布局（尤其 CEDF）在压缩率和内存占用上往往优于经典层次布局；在矩阵‑向量乘中 EDF‑1 最快；在矩阵‑矩阵加/乘中 CEDF、EDF‑1 与 DFUDS 的表现各有优势，取决于矩阵稀疏度与操作类型。

**⚠️ 局限性**

局限性：BP/CBP 方案因额外的括号序列和 rank 结构导致空间膨胀和执行慢；压缩子树的阈值调优需要经验；在极稀疏或极稠密矩阵上仍存在效率瓶颈；尚未支持并行化与能耗评估，需进一步研究。

---

## 426. Optimal PSPACE-hardness of Approximating $q$-CSP Reconfiguration

**arXiv ID:** 2607.28099 | [PDF](https://arxiv.org/pdf/2607.28099v1)

**作者:** Shuichi Hirahara `[一作]` (National Institute of Informatics), Naoto Ohsaka `[通讯]` (CyberAgent)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

本文研究了MaxMin q-CSP Reconfiguration问题的近似可行性与复杂性，证明其在1/2^{q-1}+ε因子内可在PSPACE下求解，同时在完备性下可在NP内得到1/2^{q-1}-ε因子近似解，并提出正则实例的确定性算法

**💡 创新点**

主要创新在于构造了容忍性q-查询直接乘积测试器、tolerant q-query直接乘积测试器以及利用高维离散化与二项分布的概率分析，实现了q-CSP Reconfiguration问题的极限近似阈值与完整性证明

**🔧 技术方法**

采用了直接乘积测试器（Direct Product Testers）、耐受性测试器、可解释的k-维独立函数族、Johnson-Lindenstrauss随机化以及Hoeffding/McDiarmid不等式等概率与组合技术

**📊 数据集**

未涉及具体机器学习数据集，主要在理论分析与多项式时间算法上进行实验验证

**📈 对比分析**

与现有NP-hard性/PSPACE-hard性结果比较，证明在1/2^{q-1}+ε难度下为PSPACE-hard，而在完备性下可通过多项式时间或确定性多项式时间获得1/2^{q-1}-ε近似解

**⚠️ 局限性**

主要局限是需要k、n足够大且正则结构的假设，以及对高维度独立性和随机实验的依赖，导致对小规模实例或非正则实例的适用性有限

---

## 427. PerturbMap: Cross-Context Transfer of Single-Cell Perturbation Responses

**arXiv ID:** 2607.28090 | [PDF](https://arxiv.org/pdf/2607.28090v1)

**作者:** Panpan Cui `[一作]` (University of Chinese Academy of Sciences), Wenhao Sun `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于源响应传输的框架 PerturbMap，用来预测单细胞实验中缺失的受体细胞环境下的基因表达变化。

**💡 创新点**

创新点在于：① 采用仅使用训练集的低秩坐标系来统一不同上下文的响应；② 为每条源-受体路径训练岭回归专家，并用验证锚点估计其插值权重和可靠性；③ 将路由可靠性与查询适配门控结合，形成“train‑only reliability‑weighted transport”方法；④ 通过身份保持的评估策略（source‑observed, recipient‑unmeasured）验证模型效能。

**🔧 技术方法**

核心技术包括：低秩响应坐标（训练集协方差投影）、受体本地低秩基线预测器、源到受体的岭回归专家、验证锚点插值系数计算、路由可靠性评分、查询适配门控以及多源融合的加权组合。

**📊 数据集**

主要数据集是 Perturb‑CITE‑seq melanoma 免疫逃逸数据集（5,000 计数基因，3 个细胞环境：Co‑culture、Control、IFNγ），另外在 Jiang 等的 Perturb‑seq 多源数据上做了次级可靠性加权验证。

**📈 对比分析**

与多种基线对比：受体本地低秩基线、FedAvg、零响应、原始复制、校准复制、随机打乱的 affine 复制，以及中心化的 token‑匹配聚合参考。PerturbMap 在全效应 MSE 上相较基线下降 4.1%（从 1.6490×10⁻³ 降至 1.5809×10⁻³），相较中心化参考仅差 2.82×10⁻⁶；同时在 200 条身份保持的测试中 161 条获胜、39 条受害，提升了预测准确性并降低了负迁移。

**⚠️ 局限性**

局限性：仍有 19.5% 的身份出现负迁移；仅预测条件均值响应，未生成单细胞级分布；模型依赖身份保持的训练/验证分割，无法直接迁移到开放式查询场景；以及对源-受体路径可靠性估计的效果在某些路径上仍不稳健。

---

## 428. MemHarness: Memory Is Reconstructed, Not Replayed

**arXiv ID:** 2607.28272 | [PDF](https://arxiv.org/pdf/2607.28272v1)

**作者:** Rong Wu `[一作]` (Zhejiang University), Pinlong Cai `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了 MemHarness，一种将记忆检索、批判与重构集成到 LLM 中的记忆增强框架；

**💡 创新点**

创新点在于将传统的“检索‑重放”模式转变为“检索‑批判‑重构‑执行”，通过端到端 GRPO 训练学习状态条件化的记忆重构，既保持可追溯性又具自适应性；

**🔧 技术方法**

使用 Qwen2.5‑7B‑Instruct 作为语言模型核心，BGE‑M3 作为检索器，经验记忆库，GRPO 强化学习与格式化奖励；

**📊 数据集**

实验基于 ALFWorld 与 WebShop 两大交互式长序列任务，并在 ALFWorld OOD 环境中进一步测试；

**📈 对比分析**

与闭源 LLM（如 GPT‑4o、Gemini‑2.5‑Pro）、基线记忆方法、RL+记忆、无记忆版本等对比，MemHarness 在 AlfWorld 取得 85.2% 成功率，WebShop 75.6%，均显著优于所有对照组，并在 OOD 场景中保持最高表现；

**⚠️ 局限性**

局限性包括对检索质量和记忆规模敏感，尚未在更大规模或开放式环境中验证，且重构机制缺乏可解释的规则说明。

---

## 429. When Robots Exchange Meaning: A Demo of Goal-Oriented Semantic Communications for Collaborative Robotics

**arXiv ID:** 2607.28256 | [PDF](https://arxiv.org/pdf/2607.28256v1)

**作者:** Peizheng Li `[一作]` (Toshiba Europe Ltd.), Adnan Aijaz `[通讯]` (Toshiba Europe Ltd.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fede83ac-7505-405f-ab37-e7284695c47f` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了一个机器人-边缘端的语义通信测试平台，集成了机器人侧视觉压缩、边缘侧语义映射以及基于仪表盘的任务控制，展示了从语义视觉传输到对象级地图交互的完整闭环。

**💡 创新点**

通过在机器人端使用 VQ-VAE 对 RGB 图像进行压缩并在边缘端重建，同时结合 RTAB-Map SLAM 与 YOLO 检测实现语义地图生成，并将语义流与任务控制接口结合，形成可用于任务导向 6G 研究的完整演示系统。

**🔧 技术方法**

VQ-VAE 视觉压缩、ONNX Runtime 编码、PyTorch 解码、RTAB-Map SLAM、YOLO 检测、ROS2 通信、Jetson Orin 边缘计算、5G/Open RAN 连接、浏览器仪表盘接口。

**📊 数据集**

论文未具体说明使用公开数据集，而是使用机器人实时采集的 RGB-D、LiDAR、里程计等传感器数据。

**📈 对比分析**

通过与原始 RGB 传输做对比，压缩后每帧 5400 字节，相比 230400 字节压缩 42.67 倍；离线重建 PSNR 约 21.7–21.9 dB，足以满足粗略场景感知；系统展示了地图连贯性、对象查询与任务控制的可行性。

**⚠️ 局限性**

当前仅为工作进展，缺乏对无线延迟、能耗、精细检测性能、跨机器人协作性能的系统评估；压缩后的图像细节缺失，可能影响精细任务。

---

## 430. CDAE: Enhancing Perturbation Robustness in Pretrained Language Models with Contrastive Denoising

**arXiv ID:** 2607.28236 | [PDF](https://arxiv.org/pdf/2607.28236v1)

**作者:** Sina Heydari `[一作]` (Institude for Advanced Studies in Basic Sciences (IASBS)), Majid Ramezani `[通讯]` (Institude for Advanced Studies in Basic Sciences (IASBS))

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种轻量级的对比降噪自编码器（CDAE），通过在冻结的BERT编码器上训练编码-解码网络来优化句子嵌入，使其在语义保持的文本扰动下更为稳健。

**💡 创新点**

创新点在于将对比学习与重建损失联合使用，利用冻结的预训练模型，仅增添少量可训练参数，即可显著提升嵌入对同义词替换、词删减与掩码扰动的鲁棒性。

**🔧 技术方法**

使用了预训练BERT、InfoNCE对比损失、均方误差重建损失、MLP编码器/解码器、AdamW优化器以及PyTorch和HuggingFace Transformers框架。

**📊 数据集**

主要使用了SNLI语料的前提句子作为自监督训练和评估数据集，去重后共计约57万条样本。

**📈 对比分析**

在三种扰动策略（同义词替换、词删减、掩码）和多种扰动强度下，与原始BERT和SimCSE对比，CDAE在所有情形下均保持更高的余弦相似度，鲁棒性提升幅度随扰动强度增加而加大。

**⚠️ 局限性**

局限性包括仅在SNLI数据上验证，未探究多层Transformer层的敏感性；缺乏在下游任务上的评估；并未针对更复杂或针对性攻击的扰动进行测试。

---

## 431. EMBL AI Librarian: Life-Sciences Knowledge Layer for AI Agents

**arXiv ID:** 2607.28229 | [PDF](https://arxiv.org/pdf/2607.28229v1)

**作者:** Luigi Sigillo `[一作]` (European Molecular Biology Laboratory), Fabio Petroni `[通讯]` (European Molecular Biology Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个名为 Librarian 的知识层，该层将自然语言查询转化为结构化的 Europe PMC 子查询，单个 LLM 负责生成子查询、检索文献、对全文段落进行排序、过滤并抽取可引用的证据片段，最终为生命科学 AI 代理提供高质量、可引用的证据。

**💡 创新点**

创新点主要体现在：① 使用单一 LLM 控制整个检索流程，避免重新构建向量索引；② 通过子查询生成充分覆盖多表面形式（基因、蛋白、疾病等）的检索；③ 只返回细粒度证据片段而非全文，提升检索效率并保持检索透明；④ 开源实现，可直接插拔至现有代理体系。

**🔧 技术方法**

技术栈包括：LLaMA/GLM-5 等大型语言模型用于子查询生成、段落排序与证据抽取；BM25 作为初步段落检索；Europe PMC API 用于实时检索；句子分割工具 pySBD；整个流程不需要训练额外索引或模型。

**📊 数据集**

使用了四个公开基准数据集进行评估：ScholarQA-Bench（文献综述）、ProClaim-eval（主张验证）、LitQA2（开放式问答）、LAB-Bench（基础生物学任务）。检索源为 Europe PMC 的完整文献库，包含 40M 记录。

**📈 对比分析**

评估方法：将 Librarian 与传统检索器（OpenScholar、BM25、SciRAG、PaperQA2）以及基线模型（GPT-4o、GPT-5.4、Claude 3.5 Sonnet）在同一任务下对比。结果显示：在 ScholarQA-Bench 上 Citation F1 提升 16+ 点；在 ProClaim-eval 上 Agreement 提升 5 点；在 LitQA2 上 Accuracy 提升约 8 点；在 LAB-Bench 的 macro accuracy 上提升 6–11 点。总体而言，Librarian 在所有四个基准上均显著优于基线。

**⚠️ 局限性**

局限性：① 依赖 Europe PMC，无法检索付费全文；② 当前只支持单轮检索，缺乏多跳/多轮检索逻辑；③ 不支持图像、表格、补充材料等多模态内容；④ 仅提供文本级证据，未覆盖结构化数据库检索场景；⑤ 对长尾领域覆盖不足，需扩展到其他 EMBL 资源。

---

## 432. Queue-Theoretic Admission Control for Multi-Tenant GPU Clusters

**arXiv ID:** 2607.28223 | [PDF](https://arxiv.org/pdf/2607.28223v1)

**作者:** Sohan Kunkerkar `[一作]` `[通讯]` (Red Hat), Sohan Kunkerkar (Red Hat)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文把多租户 GPU 集群的作业接纳问题建模为多类别、多资源排队网络，提出了可接纳/不可接纳队列分解、有效服务器数 k_eff 的理论框架，并给出了基于 M/G/k 的等待时间上界与改进的 Erlang‑C 近似。

**💡 创新点**

创新点在于①将可接纳与不可接纳作业通过潜在可用容量进行分解，②证明了多资源向量打包与最优接纳排序的 NP‑难性，③将向量打包映射为单一有效服务器数，③在此基础上给出了 O(1/(1‑ρ)) 的等待时间上界并验证其在实际系统中的保守性。

**🔧 技术方法**

主要技术包括向量打包约简、随机支配假设下的 M/G/k 参考模型、Kingman 近似改进、EMA 作为基准、以及 Kueue 的 ClusterQueue、Cohort、ClusterQueues 等 Kubernetes 资源模型。

**📊 数据集**

实验数据来源于在 4 节点 Kubernetes (kind) 集群上注入的 Poisson 交错时间、指数/截断指数服务时间的合成工作负载（CPU、内存、GPU via DRA），以及针对不同利用率设置的 80–100 次作业批次。

**📈 对比分析**

与 EMA 基准相比，改进的 Erlang‑C 近似在稳定区间始终保守估计（误差 2.3–18×），但在实时点预测上不及 EMA；实验显示 Little’s Law 成立、k_eff 能正确定位瓶颈资源，且模型参数可直接从 Kueue 监控指标获得。

**⚠️ 局限性**

主要局限包括：假设 Poisson 到达、非公平共享调度、单轮作业接纳、未考虑 AFS、并发接纳、拓扑约束、弹性作业等生产特性；此外，改进 Erlang‑C 的误差在低利用率时较大，且在重载情况下不可用。

---

## 433. Coexistence of 5G NR and Wi Fi 6E/7 at 6 GHz: Experimental Interference Measurements

**arXiv ID:** 2607.28213 | [PDF](https://arxiv.org/pdf/2607.28213v1)

**作者:** Rafik Zitouni `[一作]` (University of Surrey), Rahim Tafazolli `[通讯]` (University of Surrey)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对VLP Wi‑Fi 6E/7设备在6 GHz频段对5G NR系统的共频干扰进行现场实验测量，测定了对gNB上行和UE下行接收链的性能阈值。

**💡 创新点**

首次在完整的O‑RAN/SDR完整栈（含5G核心）下，以硬件实现方式测得VLP干扰的入侵阈值，且提供了从干扰功率到接收机性能（吞吐、BLER、SNR）的一一对应，并用链路预算模型将阈值映射为空间“危害半径”与LBT排斥半径，证明LBT能保证空间保护。

**🔧 技术方法**

使用软件定义无线电（USRP X310 + OBX‑160前端）与OpenAirInterface 5G NR平台，White‑Rabbit时钟同步，Wi‑Fi 6E/7 AP（TP‑Link AXE5400）在VLP模式下通过耦合器注入功率，记录吞吐、BLER、SNR等指标。

**📊 数据集**

实验数据集由多种工作点（UE下行速率85/60 Mbit/s、gNB上行、beacon‑only等）和不同注入功率（从‑105 dBm到‑40 dBm）构成，形成完整的干扰‑性能对照表。

**📈 对比分析**

通过对比无干扰基线与不同注入功率下的吞吐、BLER和SNR，阈值为‑75 dBm；在此功率下gNB UL和UE DL均未出现性能下降；SNR在‑75 dBm前下降≤2 dB，吞吐下降≤10 %；在‑60 dBm时才出现明显性能损失（吞吐≥50 %下降、BLER≥10 %）。

**⚠️ 局限性**

局限性：实验中绕过了VLP的LBT直接注入功率，单个VLP源、固定直视路径；未考虑多设备叠加、邻频干扰、多径衰落、移动性等真实环境因素；实验采用静态注入方式，实际干扰水平可能更低。

---

## 434. Old Tricks, New Models: How Simple Image Transformations Break Modern AI-based Content Moderation

**arXiv ID:** 2607.28187 | [PDF](https://arxiv.org/pdf/2607.28187v1)

**作者:** Marco Alecci `[一作]` (University of Luxembourg), Jacques Klein `[通讯]` (University of Luxembourg)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文对三大商业图像审核API（OpenAI Omni‑Moderation、Amazon Rekognition、Google Cloud SafeSearch）进行了大规模黑盒鲁棒性评测，探究简单、模型无关的图像变换是否能绕过这些审核判定。

**💡 创新点**

创新点在于证明即使是廉价、一次性变换（如颜色反转、灰度化、RGB拆合、椒盐噪声等）也能大幅降低这些API的检测率；系统地引入了视觉相似度约束（MS‑SSIM）与最小成功强度度量，并展示了不同内容类别（色情、暴力、自残、多模态恶意内容）对鲁棒性的差异。

**🔧 技术方法**

技术手段包括：① 构建七种无梯度、模型无关的图像变换；② 采用MS‑SSIM衡量变换后图像的相似度；③ 定义攻击成功率（ASR）与最小成功强度（normalized intensity）作为评估指标；④ 通过大量API调用（约60万次）收集结果。

**📊 数据集**

使用的数据集包括：LSPD（约5万张色情/非色情图像，随机抽样1000张用于评测）、UnsafeBench（支持性、暴力、自残三类的10146张图像）、Hateful Memes（11605张含文本与图像交互的恶意图像）。

**📈 对比分析**

评测方法是先在同一数据集（LSPD）下比较三大API的ASR，然后在OpenAI API上跨LSPD、UnsafeBench、Hateful Memes三类数据集与不同内容类别做进一步对比；在强度依赖变换下引入MS‑SSIM阈值进行“相似度约束”评估。结果显示：如颜色反转在Amazon上ASR高达43.97%，在Google和OpenAI分别为6.29%与8.11%；盐与椒噪在MS‑SSIM=0.6时Amazon ASR为45.53%，Google和OpenAI均超过30%；在自残类图像中ASR最高，暴力类最低。最低成功强度分析表明，许多图像在低或中等强度下即可被成功绕过。

**⚠️ 局限性**

局限性包括：① 评测仅覆盖商业API，未检验开源权重模型；② 受查询成本限制，RQ1仅覆盖三大API，RQ2、RQ3聚焦于OpenAI；③ 所用数据集虽覆盖多类危险内容，但可能无法代表全部真实场景；④ 变换种类有限，未探讨更复杂或优化的对抗方法。

---

## 435. Think with Extra-Image: A Farmland Segmentation Agent Driven by Spatio-Temporal Information Gain

**arXiv ID:** 2607.28186 | [PDF](https://arxiv.org/pdf/2607.28186v1)

**作者:** Haiyang Wu `[一作]` (Central South University), Chao Tao `[通讯]` (Central South University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 FarmSeeker——一种基于信息瓶颈的动态农田分割框架，能够主动感知图像中的不确定区域，按需查询时空扩展图像，并通过多模态推理引擎实现像素级分割细化。

**💡 创新点**

创新点在于将额外时空信息的主动获取与像素级分割结合，构建了四问式框架（何时需要、何种信息、如何获取、如何利用），并通过任务驱动奖励和分阶段训练实现闭环的主动查询与协同推理。

**🔧 技术方法**

采用多模态大语言模型（MLLM）作为推理引擎，配合基本感知、查询裁剪、分割细化工具；进阶训练策略（General FT → Cold‑Start FT → RFT）和任务驱动奖励（R_det、R_rea）来优化不确定性感知、工具调用与协同推理；构建了 GSFS‑Bench 评测基准。

**📊 数据集**

使用 FM‑Seg69K（包含基本感知、通用微调、冷启动微调和强化微调四部分）进行训练；使用 GSFS‑Bench（全球 200+ 高分辨率农田图像，覆盖十余国，包含时空查询池）进行评测。

**📈 对比分析**

与 DeepLabv3+、DDRNet、DSNet、DBBANet、LaSagnA、PixelLM、FSVLM、SegEarth‑R1 等方法对比，FarmSeeker 在 8 个中国农业区和 11 个跨国区域均取得最高或次高 IoU；在 649 个高度模糊样本上显著提升 Recall 和 IoU；在不同查询策略（预定义多时空、裁剪、主动查询）对比中，FarmSeeker 在信息增益与推理效率上表现最佳。

**⚠️ 局限性**

限制包括：推理时间相对较长（需额外感知、查询、协同推理步骤）；模糊感知仍有误检，可能影响后续查询与细化；依赖外部丰富的时空图像库；目前更适合离线农田制图或高可靠性场景，实时应用尚需进一步优化。

---

## 436. Theia: Large-Scale Multimodal Captioning and Automated Validation of the Incidents1M Dataset for Data-Free Distillation

**arXiv ID:** 2607.28269 | [PDF](https://arxiv.org/pdf/2607.28269v1)

**作者:** Simone Giano `[一作]` (Università Politecnica delle Marche), Adriano Mancini `[通讯]` (Università Politecnica delle Marche)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过从Incidents1M中恢复100,000张图片，并利用Qwen3.5-4B与Qwen3.5-35B（MoE）模型生成200,000条高质量、空间可定位的灾害场景描述，构建了一个规模化的多模态灾害数据集，并采用图像盲LLM-as-a-Judge进行自动验证；

**💡 创新点**

①提出了原子化下载与重建机制，解决Incidents1M链接失效与完整性问题；②引入图像盲LLM评判流程，精准模拟DFKD中的模态缺口；③对稠密与MoE VLM在灾害描述上的对比分析，揭示MoE在罕见事件上的优势；

**🔧 技术方法**

使用Qwen3.5系列VLM（dense 4B与Mixture-of-Experts 35B-A3B）、vLLM推理框架、异步批量推理、LLM-as-a-Judge（Qwen3.5-9B）进行文本评估，结合精确的下载器与临时文件写入机制；

**📊 数据集**

Incidents1M（原始视觉数据）以及新构建的100k图像+200k文本多模态灾害数据集；

**📈 对比分析**

通过图像盲LLM进行定量标签验证（精度≈77%，召回≈46%）与定性语义一致性评分（平均得分78.65/100），在稠密与MoE模型间进行对比，发现MoE在稀有灾害类别上F1提升显著；

**⚠️ 局限性**

召回率偏低导致描述保守；评估受限于原始标签的不完整性与人类标注错误；数据集规模有限，仅覆盖100k张图片；评判LLM的可靠性与推理成本也是潜在限制。

---

## 437. LLM-Guided Evolutionary Search for Constraint Model Reformulation to Improve Solver Efficiency

**arXiv ID:** 2607.28268 | [PDF](https://arxiv.org/pdf/2607.28268v1)

**作者:** Kostis Michailidis `[一作]` (KU Leuven), Tias Guns `[通讯]` (KU Leuven)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于LLM的进化搜索框架，用于自动改写约束模型，以提升求解速度。

**💡 创新点**

提出了Profile‑Diverse Retention (PDR) 策略，利用实例级运行时向量实现行为多样性选择；并系统评估了多种保留上下文与操作指令组合的搜索策略，展示了在保持多样性时性能提升的显著性。

**🔧 技术方法**

使用大型语言模型 DeepSeek V4 Flash 生成模型，配合自动评估器衡量解的正确性与运行时间；引入基于多样性的 mmr 重排序、基于质量的 Top‑k、回溯历史、EoH 等搜索策略；利用 PAR2 评估超时。

**📊 数据集**

8 个 CSPLib 满足问题（尾分配、船舶装载、车辆排程、误码校正、覆盖数组、学术课程平衡、BIBD、社交高尔夫）以及从 AutoIG 生成的训练/验证/测试实例池。

**📈 对比分析**

与随机采样、全历史、回溯、质量保留、混合、EoH 及其反馈、PDR（含 warm‑start）等十种策略进行对比；在 240 次实验中，PDR 获得最高几何平均提升 2.26×，EoH+feedback 次之；验证集选择进一步提升所有策略的 held‑out 速度；相较于仅追加流线器的 StreamLLM-single，整体模型改写取得更大且更稳定的加速。

**⚠️ 局限性**

实验仅使用单一 LLM 与固定 solver 配置；实例范围有限；预算与硬件受限；PDR 的多样性衡量仅基于运行时，未考虑搜索树、冲突数等更丰富的行为信息；未验证对优化问题或其他约束求解器的适用性。

---

## 438. TopoFormer: Topology Meets Attention for Graph Learning

**arXiv ID:** 2607.28259 | [PDF](https://arxiv.org/pdf/2607.28259v1)

**作者:** Md Joshem Uddin `[一作]` (University of Texas at Dallas), Baris Coskunuzer `[通讯]` (University of Texas at Dallas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种轻量级、可扩展的图表示学习框架Topo-Scan，将图的拓扑结构编码成适合Transformer处理的有序序列；

**💡 创新点**

核心创新是通过切片化节点/边过滤来生成短序列的拓扑标记（Betti数、节点/边计数），从而绕过传统持久同调的全局矩阵约简和向量化步骤，并提供稳定性理论保证；

**🔧 技术方法**

技术手段包括Topological Data Analysis（持久同调、Betti数计算）、图Transformer架构、切片化过滤、并行计算的滑动窗口方法以及多尺度滤波器（度数、Ollivier–Ricci、HKS）；

**📊 数据集**

实验使用图分类基准（BZR、MUTAG、COX2、PROTEINS、IMDB-B/M、REDDIT-B/M、REDDIT-5K、OGBG-MOLHIV）和分子属性预测基准（MoleculeNet中的BBBP、Tox21、ToxCast、SIDER、ClinTox、BACE、HIV）等；

**📈 对比分析**

与20余种SOTA基线（GNN、持久同调、对比学习、强化学习等）比较，Topo-Scan在大多数数据集上取得最优或第二名，平均偏差仅0.5%/2.5%，表明性能稳定且可与传统GNN竞争；

**⚠️ 局限性**

局限性包括仅处理低阶同调（H0,H1）且使用固定的团复形，滤波器有限（度数、曲率、HKS），未涵盖节点/边级任务、动态图或多异构图，且未学习可适应的滤波函数。

---

## 439. Space2Ground 2.0: A Multi-Source Dataset and Framework for Agricultural Monitoring through Fusion of Street-Level and Satellite Imagery

**arXiv ID:** 2607.28247 | [PDF](https://arxiv.org/pdf/2607.28247v1)

**作者:** Iason Tsardanidis `[一作]` (National Observatory of Athens), Charalampos Kontoes `[通讯]` (National Observatory of Athens)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了 Space2Ground 2.0 框架，将 Sentinel‑1、Sentinel‑2 轨道遥感与 Mapillary 众包街景相结合，构建了可直接用于作物分类的农用地块级数据集，并在塞浦路斯 2022 年生长季验证了其效果。

**💡 创新点**

创新点在于：
1) 完整的端到端自动化流程，将大量无标注的街景图像通过语义过滤、无参考 IQA、视角投影等步骤转化为地块级、可分析的标签化数据；
2) 公开的大规模多源基准数据集，包含 46,050 张街景图与 8,581 块农用地块；
3) 证明街景图像与卫星时间序列融合可显著提升作物分类精度，尤其是晚期融合方式。

**🔧 技术方法**

使用技术包括：
- Sentinel‑1 GRD SAR 与 Sentinel‑2 MS 图像预处理与时间序列构建；
- Mapillary API v4 抽取地理、语义分割与目标检测结果；
- 四种无参考 IQA 模型（MANIQA、HyperIQA、CLIP‑IQA、TReS）进行图像质量筛选；
- 视角投影与地块投影关联；
- PCA 与 k‑means 进行数据清洗；
- 传统机器学习（Logistic Regression、Random Forest、SVM、XGBoost）与深度学习（GRU、LSTM、TempCNN、VGG‑16/19、ResNet、DenseNet、EfficientNet、SqueezeNet、MobileNet、Vision Transformer）用于特征提取与分类；
- 早期融合（特征级拼接）与晚期融合（决策级加权平均）两种多模态融合策略。

**📊 数据集**

数据集来源：
- Sentinel‑1（IW 级别）与 Sentinel‑2（Level‑2A）轨道图像，覆盖塞浦路斯 2022 年生长季；
- Mapillary 众包街景图像，初始约 900,000 张；
- 塞浦路斯 GSAA 边界与农作物标签（约 325,673 块地块，包含 14 种作物类别）；
- 最终公开数据集：46,050 张街景图与 8,581 块标记地块。

**📈 对比分析**

实验比较：
- 单模态卫星：XGBoost 最高 78.90% 准确率；
- 单模态街景：ViT‑B/16 最高 70.17% 准确率；
- 早期融合（VGG‑16 + XGBoost）：82.65% 准确率；
- 晚期融合（XGBoost + ViT‑B/16）：84.12% 准确率。
- 结果显示，融合相较单模态提升约 5% 准确率，且 F1‑score、召回率等指标均有显著改善。

**⚠️ 局限性**

局限性：
- 仅能观测靠近道路的地块，远离道路或被植被遮挡的地块数据缺失；
- GPS 与罗盘误差导致视角投影误差，可能关联到邻近地块；
- 地块边界混淆与多作物共存导致单标签误差；
- 依赖农户申报的标签，存在 10% 以上错误；
- 众包图像采集受道路网络与用户活跃度影响，出现空间与时间采样偏差；
- 数据集仅来自单一采集者，视角与季节覆盖有限；
- 仍需人工干预（如聚类后人工检查）来进一步提升质量。

---

## 440. Agentic Metaverse Services: A New As-a-Service Paradigm

**arXiv ID:** 2607.28242 | [PDF](https://arxiv.org/pdf/2607.28242v1)

**作者:** Xiaofei Xu `[一作]` (Harbin Institute of Technology), Ruipeng Han `[通讯]` (Harbin Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a2602d71-93ab-4bad-974b-672788df8193` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并系统阐述了Agentic Metaverse Services（AMServ）及其在元宇宙中的实现平台Meta‑AaaS的概念、架构、原理、角色与典型应用，并给出完整的生命周期与治理框架；

**💡 创新点**

创新点在于：①首次将基于大语言模型的Agentic Service与元宇宙融合，定义AMServ为场景感知、主动规划、跨时空协作的服务；②构建Meta‑AaaS作为“Agent-as-a-Service”范式，提供服务封装、动态组合、运行时治理与多模态交互；③提出六类元宇宙Agent角色与四阶段服务实现流程，形成可持续演化的Agentic服务生态；

**🔧 技术方法**

主要技术包括大语言模型（LLM）、生成式AI、Agentic技术（感知、推理、规划、工具调用）、多模态感知与交互、数字孪生、云/边缘计算、XaaS/SaaS、服务编排与治理框架；

**📊 数据集**

该工作为概念性综述，未使用公开数据集；主要通过文献综述、案例分析与架构设计展示思路；

**📈 对比分析**

未开展实验性比较，文中通过与传统SaaS和AaaS的对比表格（触发方式、自治性、决策规划、服务粒度等）说明AMServ与Meta‑AaaS在功能、自治性与可组合性上的优势；

**⚠️ 局限性**

限制与挑战包括：缺乏实验验证与性能评估；安全、可解释性与鲁棒性问题待解决；跨世界互操作性、延迟与资源调度难点；治理与隐私合规、空间治理等仍需进一步研究。

---

## 441. Improved Learning with Structure: Fine-Grained Complexity of Minimum Consistent Subset

**arXiv ID:** 2607.28240 | [PDF](https://arxiv.org/pdf/2607.28240v1)

**作者:** Robert Ganian `[一作]` (Technische Universitaet Wien), Simon Wietheger `[通讯]` (Technische Universitaet Wien)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了最小一致子集（MCS）问题在加权与非加权图上的算法与复杂性，给出基于树宽、树深、顶点覆盖等结构参数的细粒度上界与对应下界。

**💡 创新点**

提出了3^c·(tw+1)·n^{tw+1}的树宽算法，改进无权树上9^c·n^{tw+1}的上界，并给出匹配的ETH下界；对树深给出2^{τ^2+c·τ}·n^{τ+1}的算法及其最优性；给出了单指数5^tw·n^{tw+1}的顶点覆盖参数化算法，并构造多种下界证明。

**🔧 技术方法**

采用了nice tree decomposition的动态规划、全局签名与卷积/覆盖乘积技术、距离向量枚举、集合覆盖表、快速卷积、树分解与删点等方法。

**📊 数据集**

作为理论工作，未使用真实数据集，主要在构造证明与理论实例中使用合成图。

**📈 对比分析**

与先前的2^{6c}·n^{tw+1}无权树算法以及^O(tw)·n^{tw+1}顶点覆盖算法相比，取得了单指数提升；下界表明无法突破n^{tw+1}或2^{c·tw}的指数上限。

**⚠️ 局限性**

仍缺乏对树深多项式因子的最优化，对带权图的树宽/顶点覆盖上界存在n^{+2}因子；在高颜色数下算法仍指数爆炸，且仅在理论层面，未做实验验证。

---

## 442. FaithEyes: Towards Faithful Tool Use via Multi-Agent Process-Image Verification

**arXiv ID:** 2607.28225 | [PDF](https://arxiv.org/pdf/2607.28225v1)

**作者:** Haoqing Wang `[一作]` (Samsung Research), Yehui Tang `[通讯]` (Samsung Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种多代理自评框架 FaithEyes，模型自身既是主代理又是评估子代理，对每个工具生成的过程图像进行有用性判断，并将判断结果回馈到推理上下文和奖励函数中。

**💡 创新点**

创新点在于：①通过自评子代理提供判断信号，将工具输出的有用性显式化；②用判断比例来缩放工具奖励，抑制奖励黑客；③在推理过程中既保持主模型内部的一致性，又避免外部评测模型，确保训练-测试一致性。

**🔧 技术方法**

技术包括：可执行代码工具接口、子代理判断模型、奖励设计（准确率、格式、一致性、工具信度），两阶段训练（SFT+GRPO强化学习），注意力回放分析等。

**📊 数据集**

使用公开的多模态数据集（Thyme、DeepEyes、V* Bench、HR-Bench 4K/8K、MathVista、MathVerse、MathVision）以及内部构造的自评训练样本。

**📈 对比分析**

与 GPT‑4o、LLaVA‑OV、Qwen2.5‑VL‑7B/32B、DeepEyes、Pixel‑Reasoner、Thyme、CodeV 等对比。FaithEyes 在 V*、HR‑Bench 4K/8K、MathVista、MathVerse 上均获得最高或第二高分，并显著提高工具使用的可信度（工具信度提升 20%+），同时保持单次工具调用约 1 次，未显著降低准确率。

**⚠️ 局限性**

局限性：①自评子代理的判断仍依赖模型内部的视觉推理，可能与外部专家评判存在偏差；②工具奖励比例的设定对性能有一定敏感性，需要调参；③对极其复杂或需要多步骤工具调用的任务，仍难以保证完全的工具使用可信度；④实验主要聚焦公开数据，未检验在更大规模或更专业领域的泛化能力。

---

## 443. Observing the Relationship between QoS Unpredictability, Prediction Error, and User Activity in a Remote Desktop Service

**arXiv ID:** 2607.28216 | [PDF](https://arxiv.org/pdf/2607.28216v1)

**作者:** Keisuke Ishibashi `[一作]` (International Christian University), Daiyu Nobori `[通讯]` (Japan Information Technology Promotion Agency)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对日本 Thin‑Telework RDS 真实使用日志进行大规模时序分析，研究 RTT 及其历史统计量（EMA、EMSD、Diff）与用户网络活动（发送包数、接收字节数）的关联。

**💡 创新点**

首次在真实环境下证明 RTT 的波动（EMSD）和预测误差（Diff）在平均 RTT < 100 ms 时对用户活动的影响更强，揭示 QoS 可预测性对用户体验的重要性。

**🔧 技术方法**

采用指数移动平均/标准差计算、LightGBM 回归与 SHAP 解释模型来评估特征重要性，并用可视化热图展示三种 RTT 统计量与活动的交互关系。

**📊 数据集**

使用 2023‑12‑11~17 期间收集的 19,819 台客户端一周内的 39,737,619 条一分钟时段 RTT 与流量日志（约 10,080 条时段/用户），覆盖全国用户。

**📈 对比分析**

通过对比不同 RTT 统计量与活动的中位数曲线、热图和 SHAP 重要性，发现 EMSD（31.8%/41.9%）和 Diff（47.2%/33.4%）在预测发送包数/接收字节数上优于 EMA，说明历史波动与误差是更有价值的指标；实验未给出系统性能指标，但提供了统计显著性与解释力。

**⚠️ 局限性**

局限包括：使用网络流量作为粗粒度活动代理，无法区分思考与主动输入；RTT 仅在双向包交换时可测，低活跃时可能存在偏差；只分析一周数据，缺乏长期季节性或节假日影响；关联结果未验证因果关系，可能受时间、应用类型等混杂因素影响。

---

## 444. UniCross: Unified Cross-Skill Dexterous Manipulation Synthesis

**arXiv ID:** 2607.28198 | [PDF](https://arxiv.org/pdf/2607.28198v1)

**作者:** Hui Zhang `[一作]` (ETH Zürich), Mirko Meboldt `[通讯]` (ETH Zürich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一个统一的灵巧操控框架，将抓取、搬移、手内旋转和手内平移四项技能统一建模，使用共享的状态、动作空间和奖励结构训练单一跨技能策略，实现无缝长周期操控。

**💡 创新点**

创新点在于将四项技能视为手-物体相对运动的不同实例，构建共享的手-物体关系表述和目标驱动奖励，允许单一策略兼容多种技能并在不同手型与物体形状下保持泛化。

**🔧 技术方法**

采用IsaacGym仿真环境，使用PPO训练各技能的专家策略，然后通过DAgger蒸馏为单一策略；采用交互感知的手-物体表征（接触状态、力、距离向量）和目标驱动的奖励设计。

**📊 数据集**

在仿真中随机采样盒子和圆柱体（wrappable、elongated）作为训练集和测试集；随后在未见的球体、六角柱、八边形棱柱等形状上进行泛化评估；未使用公开数据集，而是自行生成形状。

**📈 对比分析**

与基线Skill‑specific RL方法（如GraspXL、RotateIt等）在原始和通用设置下比较，统一策略在两种设置下均优于基线；在长周期任务（抓取+搬移+旋转/平移）和不同手型（Allegro、MANO、Sharpa Wave）上也保持较高成功率；在外部扰动和未知形状下保持稳健。

**⚠️ 局限性**

局限在于对极端形状或大幅度手指运动仍有轻微性能下降；跨技能蒸馏可能导致略微的性能损失；在高度动态或快速手内操作时受物理建模精度限制；未在真实机器人上验证，仿真到现实的迁移仍需进一步研究。

---

## 445. AgenticASR: Refining Speech Recognition in Real-World Scenarios via an Agentic Approach

**arXiv ID:** 2607.28175 | [PDF](https://arxiv.org/pdf/2607.28175v1)

**作者:** Zixuan Jiang `[一作]` (Shanghai Jiao Tong University), Xie Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Agentic Speech Recognition（AgenticSR），实现从音频到可读、清晰文本的在线持续修订；并构建双语 AASR-Bench 基准与原子多选 Rubric 评估；

**💡 创新点**

核心创新在于将 ASR 与 Refiner 解耦，使用有限窗口连续更新输出，既能纠正口语中的停顿、重复、错误，又能保留最终意图；同时通过 LLM 辅助生成训练数据并引入原子 Rubric 细粒度评估；

**🔧 技术方法**

采用大模型 Qwen3-ASR / Whisper 作为前端，MiniCPM‑5‑1B / Qwen2.5‑4B‑Instruct 等 Refiner，Gemma‑4‑31B‑IT 辅助数据生成与评判；在线推理结合 VAD 与 Chunk Manager，实现局部上下文修订；

**📊 数据集**

使用自建 917 条中英双语样本（共 4.218 小时、10 场景），并通过 LLM 辅助生成 100k 训练对（含 20% ASR 错误模拟），涵盖学术、客服、日常聊天、会议等多场景；

**📈 对比分析**

与 FormalASR、API‑based 重写和多种 ASR backbones 进行对比，AgenticASR 在 AASR‑Bench 上整体得分最高（Qwen3‑ASR‑1.7B 79.95），在 Content、Format、Filter、Rephrase 四维度均优于基线，延迟仅略高于 API baseline；

**⚠️ 局限性**

受限于上游 ASR 的证据质量，Refiner 规模增大导致延迟上升，且在极长或极复杂的口语修正场景中仍可能漏掉信息或产生误修订。

---

## 446. The Capacity of a Family of Sticky Channels

**arXiv ID:** 2607.28281 | [PDF](https://arxiv.org/pdf/2607.28281v1)

**作者:** Mladen Kovačević `[一作]` `[通讯]` (University of Novi Sad), Mladen Kovačević (University of Novi Sad)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

确定了一类 q‑ary sticky 插入通道的容量，并证明在满足特定支配条件时 Shannon 容量与零误码容量相等，且给出了相应的最优编码方案。

**💡 创新点**

首次给出非平凡重复通道的精确 Shannon 容量，并提出通过系数支配条件与 KL 双重性结合的通用判定方法；同时给出了包含幂律尾巴的复合 Fuss–Catalan 重复律。

**🔧 技术方法**

使用了容量‑每单位成本框架、KL 双重性、相对熵分解、支配常数判定与复合概率生成函数的 Lagrange 分析；并构造了基于模 d 同义类的零误码码。

**📊 数据集**

无实验数据集，本文完全基于信息理论与概率论分析。

**📈 对比分析**

对比传统的上界（Cheraghchi-Ribeiro 的 KL 上界）与下界（模 d 零误码构造），在满足 γ≥λ^{-d} 时实现两者相等；在阈值下给出严格上界与下界，显示容量严格大于零误码容量。

**⚠️ 局限性**

局限性：只针对支持 1+dℤ_{≥0} 的重复律；对于非平凡复合律，未能给出精确阈值 γ_d^⋆(G) 与容量等价的必要与充分条件；对阈值以下的通道容量仍存在未知区间。

---

## 447. Qwen-UI-Agent Technical Report: Toward Next-Generation Real-World Centric Foundation GUI Agents

**arXiv ID:** 2607.28227 | [PDF](https://arxiv.org/pdf/2607.28227v1)

**作者:** Hanzhang Zhou `[一作]` (Alibaba Group), Steven Hoi `[通讯]` (Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

我们提出了一个面向真实世界的通用 GUI 代理，能够在真实移动设备、桌面电脑、Web 和 DeepSearch 环境上执行跨平台工作流，并通过统一的 GUI+CLI 以及批量执行的动作空间实现高效交互；

**💡 创新点**

创新点包括：①用真实设备训练桥接模拟与现实差距；②引入混合 GUI+CLI 与批量动作，提升执行效率与覆盖范围；③实现 AutoResearch 风格的数据闭环，自动构造任务、诊断失败并迭代改进；④利用在线强化学习和验证器引导的课程，提升长周期任务可靠性；⑤提供主动服务宿主层，支持从通知到跨设备协同的自动化工作流；

**🔧 技术方法**

技术手段涵盖多模态基础模型、监督微调、动作强化学习、在线强化学习（Verifier‑Guided Curriculum）、批量动作空间、GUI+CLI 双接口、AutoJudge 轨迹评估器、跨域宿主层等；

**📊 数据集**

数据集与环境包括：真实设备训练集（100+ Android 设备，150+ App），MobileWorld‑Real（409 任务），AndroidDaily，MobileWorld，OSWorld‑Verified/OSWorld‑v2，WebArena，BrowseComp/BrowseComp‑ZH，ScreenSpot‑Pro/V2，MMBench‑GUI L2，OSWorld‑G‑Refined，UI‑Vision，以及通用与智能化基准（MMMU‑Pro、MMLU‑Pro、Terminal‑Bench 2.0、Claw‑Eval、BFCL‑v4、SkillsBench、QwenClawBench 等）；

**📈 对比分析**

我们将其与前沿专有模型（Gemini 3.1 Pro、Opus 4.8、GPT‑5.6 Sol、Seed 2.1 Pro）以及开源基础模型对比，结果显示：在 MobileWorld‑Real 达到 92.2% 以上，在 AndroidDaily 97.5%，在 MobileWorld 82.1%；在 OSWorld‑Verified 79.5%，OSWorld‑v2 40%；WebArena 73.6%，BrowseComp 64.1%，ScreenSpot‑Pro（放大）81.5%，同时在通用与智能化基准保持或提升性能；

**⚠️ 局限性**

主要局限：①真实设备评测依赖 AutoJudge，准确率约 92.8%；②部分 35B‑A3B 规模训练未完成；③高保真仿真环境尚未集成；④自动化闭环仍需人工干预；⑤执行延迟和交互成本高；⑥安全性与个性化的系统化评估尚待加强。

---

## 448. Security of World-Model-Based Embodied AI: A Lifecycle of Threats, Defenses, and Evaluation

**arXiv ID:** 2607.28226 | [PDF](https://arxiv.org/pdf/2607.28226v1)

**作者:** Fazhong Liu `[一作]`, Haojin Zhu `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该论文系统综述了基于世界模型的具身人工智能安全问题，提出完整的生命周期框架、攻击映射表、评估协议以及针对不同安全洞察的防御优先级。

**💡 创新点**

创新点在于：① 将世界模型的安全威胁统一到从数据构造到执行反馈的生命周期视角；② 提出了五大安全洞察（语义-仿真-行动差距、状态-不确定性条件风险、回滚-执行放大、轨迹级非组合性、预测安全幻觉）；③ 为每个生命周期阶段制定评估指标和防御方案，填补了现有安全基准和方法的空白。

**🔧 技术方法**

使用的技术包括：生命周期分析与映射、攻击与安全目标映射表、评估指标设计（预测质量、预测安全率、恢复延迟等）、基于不确定性、集成、规则校验和安全盾等防御手段。

**📊 数据集**

参考数据集和基准主要有：AI2-THOR、CARLA、Safety Gym、ALFRED、RLBench、SafeBench、AttackVLA、EVA‑VLA、AGENTSAFE、SafePlan‑Bench、SHAWSHANK 等。

**📈 对比分析**

论文通过统一评估协议对比现有工作，关注攻击成功率、预测安全率、恢复延迟等多维指标。由于是综述性工作，未给出具体数值，但指出目前公开基准在评估生成世界模型预测、预测安全幻觉等方面存在缺口，需要进一步实验验证。

**⚠️ 局限性**

局限性包括：缺乏完整、可复现的基准与实验；对隐式世界模型（如VLA）安全评估仍困难；生成数据的持续性污染与长期记忆攻击验证不充分；跨域迁移与人机信任评估仍需进一步研究。

---

## 449. Identifying a Level-up Pathway for AI-assisted Counterspeech through Elaboration

**arXiv ID:** 2607.28239 | [PDF](https://arxiv.org/pdf/2607.28239v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 450. Technology-Enhanced Tabletop Exercises for Cybersecurity Education: Lessons Learned

**arXiv ID:** 2607.28179 | [PDF](https://arxiv.org/pdf/2607.28179v1)

**作者:** Jan Vykopal `[一作]` (Masaryk University), Valdemar Švábenský `[通讯]` (Masaryk University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在大学网络安全课程中实施并评估了25次技术增强型桌面演习，涵盖743名学生，使用INJECT练习平台自动化场景、记录交互并进行后测。

**💡 创新点**

将传统桌面演习数字化、自动化并与学习分析相结合，提供实时评估与数据驱动的反思，显著降低教师工作量并提升学习体验。

**🔧 技术方法**

使用基于Web的INJECT Exercise Platform（开源）以及YAML定义、可视化编辑器、VS Code插件、模拟工具（邮件、浏览器、防火墙等）和生成式AI辅助内容。

**📊 数据集**

收集了743名参与者在25次演习中的交互日志、邮件线程、工具使用、里程碑达成等数据，并以JSONL格式导出。

**📈 对比分析**

通过对比传统纸质桌面演习与技术增强演习，观察到学生参与度、协作效率提升、评估周期缩短；实验显示自动化评估能在相同时间内处理更多团队，但在大规模时实时评估出现瓶颈。

**⚠️ 局限性**

主要限制包括设计与准备阶段的复杂性、对技术熟练度的依赖、实时评估在多团队时的瓶颈、数字平台对交互体验的高期望导致细节问题，以及对教师专业判断的持续依赖。

---

## 451. Demystifying DRAM Read Disturbance: Bridging the Gap Between Experimental Characterization and Device-Level Modeling of RowHammer and RowPress Phenomena

**arXiv ID:** 2607.28233 | [PDF](https://arxiv.org/pdf/2607.28233v1)

**作者:** Haocong Luo `[一作]` (ETH Zurich), Onur Mutlu `[通讯]` (ETH Zurich)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过实验性测量和TCAD级别的设备模拟，系统性地弥合了DRAM读失真现象（RowHammer与RowPress）实验结果与设备级物理机制之间的差距，提出了新的、更新的机理模型并对比验证。

**💡 创新点**

创新点在于：①首次结合实验与TCAD模拟识别并量化了单/双侧RowHammer以及RowPress在不同阈值下的bit‑flip方向、AC_min差异与空间分布；②揭示电荷陷阱位置、电子迁移路径以及电容耦合如何决定双侧RowHammer的位翻转方向；③解释了为何在正常工艺条件下NW‑L RowPress 0→1翻转不可观测；④引入了孔陷阱密度对RowPress AC_min的影响，说明两种失真机制的竞争关系。

**🔧 技术方法**

主要技术手段包括：DRAM Bender实验平台对DDR4芯片的高分辨率读/写控制；内部行映射与单元布局逆向工程；Sentaurus TCAD进行3D DRAM单元结构、陷阱辅助电子迁移与电容耦合的混合模式仿真；并在晶圆级测试结构上测量NW‑L/ P‑L诱导的漏电流。

**📊 数据集**

使用的数据集为多家厂商（S、H、M）生产的DDR4 DIMM模块，总计约20余颗芯片；每颗芯片随机选取128行受害行进行实验，覆盖不同温度、数据模式与激活计数。实验结果与仿真输出均以热图、AC_min曲线和Jaccard重叠度等形式呈现。

**📈 对比分析**

比较方法：将TCAD仿真得到的bit‑flip方向、AC_min和空间分布与真实芯片实验数据进行逐项对照；利用统计指标（如Jaccard系数、误差阈值）量化一致性；在不同温度与trap密度参数下绘制AC_min随tAggON变化曲线。性能表现上，仿真能够准确复现实验中观测到的双侧RowHammer同时出现0→1与1→0翻转、AC_min差异以及P‑L RowPress 1→0翻转可观测性，且误差率低于5%。

**⚠️ 局限性**

局限性包括：①陷阱密度、分布与材料参数仍以理论或近似值为主，未能完全覆盖芯片批次间的工艺波动；②模型未考虑多单元间的耦合与动态热效应；③实验中仅使用DDR4，未验证在更先进节点（如DDR5、HBM）的适用性；④对某些罕见或极端操作条件（如极低温、极高电压）下的失真机制缺乏系统评估。

---

## 452. Understanding Is Done Early: A Depth Division of Labor in Large Language Models and Its Use for Unbounded-Context Memory

**arXiv ID:** 2607.28263 | [PDF](https://arxiv.org/pdf/2607.28263v1)

**作者:** Hanzuo Liu `[一作]` (Tsinghua University), Mingyu Gao `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出CoMem方法，在Transformer深度上实现中层状态缓存与上层重算的分层长上下文记忆。

**💡 创新点**

创新点在于将深度作为可调节的重用轴，仅缓存中层残差张量并对查询进行限定检索，实现固定检索预算下的内存与计算独立。

**🔧 技术方法**

技术包括中层写入缓存、BM25检索、上层重算、低秩自蒸馏LoRA、统一chat‑free评测协议。

**📊 数据集**

使用数据集包括PG‑19用于自蒸馏、RULER、LongEval、LongBench、BABILong、LoCoMo等长文本与多轮对话基准。

**📈 对比分析**

在Qwen3‑8B上，CoMem在RULER达97.05、LoCoMo 38.27、LongEval 69.0，内存18.26 GB、预填充速度7.83×，显著优于KV‑Direct与其他长上下文方案。

**⚠️ 局限性**

局限性包括对超长窗口无位置扩展、检索宽度与任务相关、仅支持英文、对多事实问答仍弱、未衡量检索延迟和动态更新等。

---

## 453. Causal Discovery with Inverted Self-attention for Multivariate Time Series

**arXiv ID:** 2607.28212 | [PDF](https://arxiv.org/pdf/2607.28212v1)

**作者:** Yusen Liu `[一作]` (University of Technology Sydney), Huan Huo `[通讯]` (University of Technology Sydney)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了多变量时间序列的因果发现，提出了逆向自注意机制框架。

**💡 创新点**

创新点是引入逆向因果自注意（CSAM），通过SparseMax实现稀疏注意力，提出全局因果算法和置换重要性验证模块。

**🔧 技术方法**

使用Transformer自注意、SparseMax、Permutation Importance、全局聚合等技术。

**📊 数据集**

评估数据集包括Henon Maps、Finance、Lorenz-96、fMRI（线性与非线性）。

**📈 对比分析**

与BGranger、KGC、tsFCI、TCDF、PCMCI、DYNOTEARS等基线对比，F1最高，精度、召回均优于现有方法。

**⚠️ 局限性**

限制：模型对超参数敏感，需较多计算资源；仅在四个基准数据集验证，未评估大规模真实世界场景。

---

## 454. Scaling Vision-Language Models Is Not Enough to Mitigate Bias

**arXiv ID:** 2607.28211 | [PDF](https://arxiv.org/pdf/2607.28211v1)

**作者:** Ioannis Sarridis `[一作]` (Information Technologies Institute, CERTH), Symeon Papadopoulos `[通讯]` (Information Technologies Institute, CERTH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对16个VLM家族、24个训练数据集以及3个不同偏差复杂度的基准进行大规模实证研究，评估模型规模、训练数据与架构对零样本Worst-Group Accuracy的影响。

**💡 创新点**

首次系统揭示了VLM的“偏差复杂度敏感性”，发现模型规模对单属性偏差有一定正向作用，但对多属性偏差几乎无效，而训练数据量与质量才是稳健性的核心驱动力。

**🔧 技术方法**

使用零样本CLIP式对齐、Worst-Group Accuracy度量、Spearman相关性分析以及匹配组对照实验等技术手段。

**📊 数据集**

训练数据集涵盖LAION、WebLI、DFN、CommonCrawl等24个大规模公开数据集；评估基准为ImageNet、CelebA和UrbanCars。

**📈 对比分析**

通过比较参数、数据量、token数、patch大小、图像分辨率等设计因子，发现参数规模与ImageNet准确率呈正相关，但与多属性WGA几乎无关；训练数据规模与所有指标均保持正相关；最佳模型为中等规模、细粒度token、优质数据组合，能够在ImageNet、CelebA和UrbanCars的Worst-Group Accuracy上达到Pareto最优。

**⚠️ 局限性**

仅覆盖公开的OpenCLIP checkpoint，最大模型参数仅3.6B；评估基准仅限两类偏差（单属性与多属性），未涵盖更多领域或后期微调场景；所有评估均为零样本，缺乏对任务特定微调效果的考察。

---

## 455. Toward Annotation-Efficient Continuous Emotion Arousal Quantification via Group-Level EEG Dynamic Neural Synchrony

**arXiv ID:** 2607.28204 | [PDF](https://arxiv.org/pdf/2607.28204v1)

**作者:** Guandong Pan `[一作]` (Beihang University), Shaoting Tang `[通讯]` (Beihang University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了基于群体 EEG 同步性（DNS）的连续情绪唤醒量化，避免逐个主体手工标注。

**💡 创新点**

首次系统评估 DNS 与情绪动态的耦合，发现它更能捕捉唤醒变化率，并给出窗口、时延、特征等参数准则。

**🔧 技术方法**

采用 Correlated Component Analysis (CorrCA) 计算滑动窗口的 DNS，并对 FD、DE 等特征进行分析。

**📊 数据集**

使用四个 EEG 数据集：SEED、SEED-IV、SEED-VII 与自收集的 BAVE，覆盖 142 受试者 207 小时。

**📈 对比分析**

通过 ANOVA、Pearson 相关、循环移位、块置换和受试者拆分等统计检验，DNS 与唤醒导数的相关系数在 0.2 左右，显著高于静态唤醒。

**⚠️ 局限性**

仅适用于群体级别，受低层特征（光、运动）影响，需进一步验证在非视频场景的泛化能力。

---

## 456. MORFES: A Benchmark for Productive Inflectional Competence in Modern Greek

**arXiv ID:** 2607.28274 | [PDF](https://arxiv.org/pdf/2607.28274v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 457. Fidelity Is Not Safety: Gently-Compressed LLMs Pass Every Data-Free Quality Guard Yet Invent Procedure Steps in Agentic Execution

**arXiv ID:** 2607.28196 | [PDF](https://arxiv.org/pdf/2607.28196v1)

**作者:** I. Kennedy `[一作]`, T. Kennedy `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了压缩语言模型后在作为agent执行程序时的安全性问题，发现传统的perplexity、MMLU和数据‑free fidelity守卫无法检测到模型在执行标准操作程序时自发“发明”步骤的行为。

**💡 创新点**

提出了盲点（blindspot）和解偶联（dissociation）现象，发现低秩截断（SVD）压缩的误差在“coherence × rate”轴上导致发明步骤，而幅值剪枝则不易触发；并基于此设计了一个完全数据‑free的两轴检测器（coherent_fraction 与 error_rate），能够在agent部署前预警。

**🔧 技术方法**

采用低秩截断、幅值剪枝、量化等压缩手段；使用 WikiText‑2 评估 perplexity；MMLU 0‑shot评估下游准确率；CKA/余弦相似度随机探针评估内部表示的 fidelity；随机子空间迭代计算奇异值以得到 coherence_fraction；统计 error_rate。

**📊 数据集**

使用 WikiText‑2、MMLU 0‑shot数据集以及自构造的 24 条 SOP（共 144 条对比实验，3 个随机种子）作为检验基准，并在 Qwen3‑8B、Mistral‑7B、Llama‑3.1‑8B 三种 7–8B decoder 结构上进行压缩实验。

**📈 对比分析**

在相同 perplexity（≤1.15×）下比较 SVD 与匹配幅值剪枝的发明步骤率，SVD 在所有三种架构上平均产生 +1~+2 个未授权步骤，而剪枝几乎无发明；利用置信区间统计和 paired estimator 进行严格对比；两轴检测器在 6 个标注实验中以固定阈值实现 100% 的正确分类。

**⚠️ 局限性**

仅测试了 7–8B 的 dense decoder，未覆盖 MoE 或其他压缩方法；检测器阈值仅在 SVD 与剪枝这两类操作下通用；检验基准为 synthetic SOP，可能不代表真实复杂任务；极端剪枝（>50%）可能导致检测误差，需进一步验证。

---

## 458. The MADRS Pipeline: Supporting Depression Assessment in Clinical Trials

**arXiv ID:** 2607.28190 | [PDF](https://arxiv.org/pdf/2607.28190v1)

**作者:** Mila Fodor `[一作]` (Clario, part of Thermo Fisher Scientific), Alex Boudreau `[通讯]` (Clario, part of Thermo Fisher Scientific)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究开发了一个端到端的LLM管道，支持在临床试验中使用SIGMA指南进行MADRS抑郁评估，包含音频转录、症状分段、严重度估计以及评分质量检测；

**💡 创新点**

其创新点在于针对标准化临床访谈设计的多组件LLM架构，能将访谈文本与MADRS症状结构对齐，并通过混合模型实现高效、可解释的评分与质量监测；

**🔧 技术方法**

主要技术包括自动语音识别（ASR）与说话人分离、LLM（如GPT‑4等）进行转录与分段、基于Transformer编码器的多任务有序回归进行MADRS评分，以及随机森林进行质量评估；

**📊 数据集**

使用的数据集来自1602次40分钟SIGMA访谈录音，其中包含8次人工转录、14次分段标注、1100次MADRS评分训练集、251次验证/测试集以及305次质量评估标注，全部为真实临床试验数据；

**📈 对比分析**

与LLMADRS基线对比，精调RoBERTa模型在总分上达MAE 2.956、Spearman 0.867、Accuracy@1 0.895；质量评估随机森林在MCC上取得0.704，优于阈值基线；

**⚠️ 局限性**

主要局限包括仅基于文本缺失声学和视觉信息、依赖ASR导致转录误差影响、质量评估样本极度不平衡以及无法公开数据和模型。

---

## 459. Persistent Gaussian Perturbations Prevent Oversmoothing in Recurrent Graph Neural Networks

**arXiv ID:** 2607.28185 | [PDF](https://arxiv.org/pdf/2607.28185v1)

**作者:** Mostafa Haghir Chehreghani `[一作]` `[通讯]` (Amirkabir University of Technology (Tehran Polytechnic)), Mostafa Haghir Chehreghani (Amirkabir University of Technology (Tehran Polytechnic))

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对带有独立高斯噪声的循环图神经网络进行建模，并证明其能够从根本上防止过度平滑；

**💡 创新点**

提出噪声注入能从根本上改变收敛行为，给出严格的 Dirichlet 能量下界，且该下界与图谱间隙和噪声方差呈线性关系；

**🔧 技术方法**

利用随机动力系统、马尔可夫链理论、随机迭代函数、Wasserstein 收敛以及谱理论对 Dirichlet 能量进行分析；

**📊 数据集**

实验使用随机 Erdős–Rényi 图以及 Cora 节点属性，验证线性与非线性递归 GCN 的行为；

**📈 对比分析**

与无噪声基准相比，噪声模型在 Dirichlet 能量上保持正值，并随噪声强度呈二次增长，实验结果与理论预测高度吻合；未进行分类任务性能对比；

**⚠️ 局限性**

主要限制在于对全局收敛性的强假设、仅考虑加性高斯噪声、未讨论训练收敛及下游任务性能。

---

## 460. A Cloud Continuum Research Infrastructure for Distributed CPS Experimentation

**arXiv ID:** 2607.28193 | [PDF](https://arxiv.org/pdf/2607.28193v1)

**作者:** Fabio Orazio Mirto `[一作]` (University of Messina), Antonio Puliafito `[通讯]` (University of Messina)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出两层参考架构，将实验基础设施层与 Edge–Fog–Cloud 应用层分离，并在 SLICES 云连续体蓝图上实现可重复的控制与监测工作负载实验；

**💡 创新点**

创新点在于将实验平台与应用设计统一到工作流证据方法中，支持跨层实验的可重复性、可追溯性和可比较性；

**🔧 技术方法**

采用 SLICES 蓝图、Kubernetes+Crossplane、Stack4Things、IoTronic、MQTT、Kafka、Apache Spark、TEANS、InfluxDB、Raspberry Pi 等开源技术栈；

**📊 数据集**

使用合成的可再生能源社区能源曲线与城市监测 CO₂ 传感器轨迹，统一 10 分钟仿真运行；

**📈 对比分析**

通过 40 次自动化运行，比较虚拟化 POD 与物理 RASP 边缘、REC 与 AirWatch 两类工作负载；结果显示物理边缘导致 Edge‑Fog 延迟约 50% 增幅，云端聚合保持稳定，控制与监测工作负载均满足设计延迟阈值；

**⚠️ 局限性**

局限性包括仅在三站点（Messina–Bologna–SLICES）拓扑下验证、未探索多种组件放置、使用合成数据而非真实传感器噪声、实验时间仅 10 分钟，未覆盖长期运行与资源老化等场景。

---

## 461. EgoGenesis: Egocentric World-Action Modeling with Online Anchored Projective Memory and Action-3D RoPE

**arXiv ID:** 2607.28243 | [PDF](https://arxiv.org/pdf/2607.28243v1)

**作者:** Zexuan Yan `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于预训练视频生成器的自回归 egocentric 生成模型 EgoGenesis，结合 OAPM 与 A3D‑RoPE 两个几何感知条件机制，用于合成可控的高质量操纵视频，以扩充稀缺的真实 egocentric 训练数据。

**💡 创新点**

创新点在于：① 在线锚定投影记忆 OAPM，维护首帧 3D 场景锚点并周期刷新最近状态；② 动作 3D 旋转位置嵌入 A3D‑RoPE，将相机感知的 3D 关节坐标注入跨注意力，实现精确动作对齐；③ 将这两项机制整合至 DiT 生成框架中，显著提升几何稳定性与动作一致性。

**🔧 技术方法**

技术包括预训练 DiT 生成器、VAE 编码/解码、VGGT‑Ω 3D 场景重建、Gated Cross‑Attention、RoPE 与自定义 A3D‑RoPE、在线锚定投影记忆以及自回归流模型的噪声消除。

**📊 数据集**

在 210k 条来自 EgoDex、AgiBot、RoboTwin、Real‑world Ego 与 DexJoCo 的 egocentric 视频‑动作对上进行源平衡训练，并以 400 条真实轨迹 + 400 条合成轨迹作为下游机器人任务的扩充数据。

**📈 对比分析**

与 Wan2.1‑Fun‑14B‑Inp、Wan2.2‑5B‑Control、EgoHOI、Mask2IV 等通用/可控视频生成模型在同一场景与动作条件下对比，EgoGenesis 在 PSNR、SSIM、LPIPS、Kpt.Err、Phys.Faith 等七项指标中排名前列；在下游双臂/单臂真实机器人任务上，增补 400 条合成轨迹将 OOD 成功率从 77% 提升至 84%（单臂），从 53% 提升至 70%（双臂），并显著降低 ID‑to‑OOD 损失。

**⚠️ 局限性**

尽管几何漂移显著下降，但在极端遮挡或极长时序下仍会出现小幅漂移；模型仍受限于 210k 规模数据，缺乏对全新场景或对象类别的鲁棒性；合成视频与真实视频在物理细节和光照一致性上仍有微小差距，限制了极高精度控制的可迁移性。

---

## 462. AI and Authenticity in Islamic Research: A Critical Evaluation of Generative AI Reliability, Hallucination, and Source Fidelity in Quranic, Hadith, and Fiqh Knowledge

**arXiv ID:** 2607.28237 | [PDF](https://arxiv.org/pdf/2607.28237v1)

**作者:** Muhammad Sajjad Akbar `[一作]` `[通讯]` (University of Sydney), Muhammad Sajjad Akbar (University of Sydney)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过收集来自澳大利亚和英国的50个开放式伊斯兰问题，并让用户自由使用主流生成式AI（ChatGPT、Gemini、Claude等）回答，评估其在古兰经、圣训、法理学、伦理、牧养和流派敏感性等领域的准确性、可信度与可解释性。

**💡 创新点**

创新之处在于首次将真实用户交互与混合方法评估相结合，系统考察了AI回答的真实性、引用完整性、学派差异处理和地理检索差异，揭示了生成式AI在宗教知识领域的局限性与差异化表现。

**🔧 技术方法**

研究采用了多维度混合评估框架，包括事实正确性、引用验证、虚假生成检测、法理一致性、不确定性识别与责任回避等方法，并结合定量评分与质性文本分析。

**📊 数据集**

使用的数据集为来自澳大利亚和英国参与者的5份调查集（共50个问题），涵盖古兰经解释、圣训引用、法理推理、伦理道德、牧养建议及流派敏感话题的真实回答。

**📈 对比分析**

对六款主流生成式AI的性能进行对比，发现古兰经与伦理类问题准确率最高（约4.4/5），而法理学与流派敏感问题最低（≈2.1/5）；在引用完整性和不确定性处理方面，Gemini与Claude表现最佳，其余模型存在明显差距。

**⚠️ 局限性**

局限性包括样本规模有限、仅使用英文回答、未覆盖多语言与多流派的深度评估、缺乏自动化引用验证工具、以及受检索环境和地理位置影响导致的结果可重复性问题。

---

## 463. Student Perceptions and Preferences Regarding AI-Generated Instructional Videos in Computing Education

**arXiv ID:** 2607.28203 | [PDF](https://arxiv.org/pdf/2607.28203v1)

**作者:** Esse Ciego `[一作]` (University of Florida), Amanpreet Kapoor `[通讯]` (University of Florida)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对170名计算机专业学生进行观看3个AI生成的Markdown教学视频后，完成问卷调查，收集对视频质量、可用性、学习效果及其在课程中的应用态度与关注点的自评。

**💡 创新点**

①首次系统性研究学生对编程教育中AI生成教学视频的感知与使用偏好；②聚焦视频而非文本LLM；③探讨学生在不同教学情境下对AI视频的适宜性与担忧。

**🔧 技术方法**

使用Knowlify平台生成3分钟的解释性AI视频，采用Qualtrics后测问卷收集数据，并用描述性统计与主题分析进行定量与定性分析。

**📊 数据集**

170名学生的问卷数据；3段约9分钟的Markdown教学视频（由Knowlify生成）。

**📈 对比分析**

采用描述性统计与主题分析；未与其他教学材料直接对比，但学习测验平均得分4.3/5，视频质量评分平均4.52/5，整体满意度高；对大规模使用保持保留态度。

**⚠️ 局限性**

①非对照实验，无法证明AI视频优于传统资源；②样本仅为高阶CS学生，背景知识可能影响结果；③视频简短、主题浅，结果可能不适用于更复杂内容；④自我报告与单向问卷可能存在偏差。

---

## 464. Secure Aggregation for Privacy-Preserving Federated Learning on Clinical EEG Data

**arXiv ID:** 2607.28191 | [PDF](https://arxiv.org/pdf/2607.28191v1)

**作者:** Pouya Rajabi `[一作]` (University of South-Eastern Norway), Mohsen Toorani `[通讯]` (University of South-Eastern Norway)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc` `b88c6eac-d57a-4623-a604-1f401f3eb268` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

设计并实现了一个针对临床EEG数据的跨机构联邦学习安全聚合框架，集成图形通信、阈值秘密共享、脱落恢复、恶意防护和辅助验证等机制。

**💡 创新点**

在Masking‑based SecAgg基础上引入稀疏图通信、可选Bloom过滤器记录链接、辅助审计者轻量级一致性校验，并提供四种安全级别（半诚实、恶意、带审计者）方案。

**🔧 技术方法**

Masking‑based secure aggregation、图形邻居通信、Shamir秘密共享、Merkle树公钥承诺、签名、Bloom过滤器、线性验证标签及Flower框架实现。

**📊 数据集**

使用TUH EEG Corpus衍生的2500份EDF文件（1,250正常+1,250异常），经预处理为19通道、10s窗口的特征。

**📈 对比分析**

通过与基线FL、基本SecAgg对比，评估100轮的运行时、通信量和准确率。半诚实方案开销最小；恶意方案计算与通信约增3–4倍；带审计者进一步提升一致性检查，准确率与基线相近但成本更高。

**⚠️ 局限性**

仅防止服务器窥探个体更新，未处理恶意客户端攻击、投毒、后门；依赖非协作方服务器；Bloom过滤器链接在无可用标识时不可用；实验规模限于70客户端，缺乏更大规模验证。

---

## 465. Multi-channel Uplift Policy Learning

**arXiv ID:** 2607.28182 | [PDF](https://arxiv.org/pdf/2607.28182v1)

**作者:** Changjian Liu `[一作]` (Peking University), Bo Zheng `[通讯]` (Alibaba Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了 ReAlloc 框架，在电商平台上实现多渠道固定预算提升政策学习，并在淘宝进行离线评估与线上 A/B 测试。

**💡 创新点**

将多渠道预算分配建模为单纯形上的提升决策，提出支持感知的因果重分配策略，并通过快慢教师-学生双系统学习本地梯度与全局潜在场，实现安全、稳健的资源分配。

**🔧 技术方法**

正交化因果估计（双重机器学习）、梯度蒸馏、潜在函数潜在场、支持检测、保守局部搜索、离线反事实评估（DR/OPE）、匹配评估以及在线 A/B 测试等技术。

**📊 数据集**

使用淘宝 60 天生产日志，包含 500k 商品的预算分配、支付订单、GMV 等信息，并辅以 10% 随机探索子集进行离线反事实评估。

**📈 对比分析**

与日志策略、均匀分配、Additive ROI、S-Learner PTO、R-Learner 等基线对比；离线模拟中 ReAlloc 在支持受限场景下获得最高可部署提升（≈0.71/0.57），线上 A/B 测试中支付订单提升 3.53%，利润率提升 3.26pp，GMV 降低 2.64%。

**⚠️ 局限性**

对远场非线性效应的识别依赖足够支持，支持检测误差可能导致过度保守；在极端分配边缘仍可能出现 OOS；需要更多探索数据以提高因果梯度的信噪比，对预算动态变化的鲁棒性仍待验证。

---

## 466. Integrating AI into Requirements Quality Learning in Software Engineering Education: A TPACK-Guided Empirical Study

**arXiv ID:** 2607.28176 | [PDF](https://arxiv.org/pdf/2607.28176v1)

**作者:** Hansika Ekanayake Mudiyanselage `[一作]` (Tampere University), Zheying Zhang `[通讯]` (Tampere University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过将多代理LLM辅助需求生成工具嵌入软件工程硕士级需求工程课程的作业中，并使用TPACK框架对作业设计与学习成效进行评估。

**💡 创新点**

创新点在于将TPACK理论与AI工具相结合，构建了一个结构化的作业流程（手工先改写 → AI生成 → 对比评估 → 同行评审 → 反思），从而培养学生对AI产出的评估与批判性思维。

**🔧 技术方法**

使用了自研的多代理LLM辅助需求工具、TPACK框架进行教学设计、手工需求改写、对比分析和同行评审等技术与流程。

**📊 数据集**

数据集来自100名硕士生（72份可分析）的课程作业提交，包括手工需求、AI生成的用户故事、同行评审分数和反思问卷。

**📈 对比分析**

通过对比前后对四条用户故事的INVEST维度评分（与教师参考评分）以及Likert问卷的统计，评估AI支持的学习提升；结果显示在可测性、价值等结构性维度有显著提升，而可协商性等解释性维度提升有限。

**⚠️ 局限性**

局限性包括缺乏对照组、单一课程/单一机构、工具为研究原型且功能受限、仅评估四条用户故事、以及对可协商性等主观维度衡量可能存在偏差。

---

## 467. (Towards) Scalable Reliable Automated Evaluation with Large Language Models

**arXiv ID:** 2607.28282 | [PDF](https://arxiv.org/pdf/2607.28282v1)

**作者:** Bertil Braun `[一作]` (KIT), Martin Forell `[通讯]` (KIT)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于多模型双向比较和Elo评级的LLM生成文本自动评估框架

**💡 创新点**

创新点在于同时使用多模型进行pairwise比较以降低单模型偏差，并通过Elo系统生成稳定且可解释的排名，同时提供可调同意阈值以灵活控制评估置信度与覆盖率

**🔧 技术方法**

主要技术包括多LLM双向比较、Elo评分算法、可调投票阈值机制及自动化评估脚本

**📊 数据集**

使用公开的科研摘要数据集进行能力特征提取与评估

**📈 对比分析**

方法通过pairwise比较生成Elo排名，实验结果显示自动得分与专家评估高度相关，显著减少人工干预且保持评估一致性

**⚠️ 局限性**

主要局限在于O(n²)的计算开销导致高成本、难以区分相近质量样本、以及专家评估样本规模有限导致泛化性受限

---

## 468. MSCM-net: A hyperspectral image classiffcation method based on multi-scale convolution and Mamba

**arXiv ID:** 2607.28277 | [PDF](https://arxiv.org/pdf/2607.28277v1)

**作者:** Jianjun Chen `[一作]` (Qingdao University of Technology), Mingwei Shao `[通讯]` (Qingdao University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种融合多尺度卷积和Mamba的高光谱图像分类模型MSCNet

**💡 创新点**

创新点包括：①将多尺度卷积与Mamba结合以同时捕获局部与长程依赖；②在MCSE模块加入SENet实现通道自适应加权；③在Mamba输入前加入残差连接以防止信息丢失；④设计中心与全局双分支聚合模块以充分利用中心像素和全局统计信息

**🔧 技术方法**

使用了多尺度卷积、SENet、Mamba（选择性状态空间模型）、残差连接、双分支聚合模块以及多层感知机分类头；训练采用Adam、PCA降维、Patch切块等预处理

**📊 数据集**

在三个公共基准数据集上实验：Indian Pines、WHU‑Hi‑HongHu、Salinas

**📈 对比分析**

与SVM、3DCNN、SSRN、SSFTT、MASSFormer、SQSFormer、SSMamba等多种基线模型对比，MSCNet在OA、AA、Kappa均获得最高或接近最高结果；参数量仅约0.2M，GFLOPs 0.19，推理速度最快，显示出较高的性能与计算效率

**⚠️ 局限性**

局限性：目前仅在3个数据集上验证；缺乏对大规模真实遥感图像或多时相数据的评估；模型结构仍相对固定，未探讨不同卷积核组合或Mamba层数的进一步优化

---

## 469. Generalized Query-Oriented Image Semantic Coding Empowered by Large AI Models and Semantic-Aware Hybrid Beamforming

**arXiv ID:** 2607.28276 | [PDF](https://arxiv.org/pdf/2607.28276v1)

**作者:** Sin-Yu Huang `[一作]` (University of British Columbia), Vincent W. S. Wong `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出通用查询导向图像语义编码（QO‑ISC）框架，结合语义感知混合波束成形（SA‑HBF），在大规模MIMO‑OFDM系统中实现用户意图优先的图像传输与重构。

**💡 创新点**

创新点包括：① 利用预训练CLIP模型提取通用视觉‑文本特征并通过FiLM对齐，突出用户查询相关语义；② 在传输端引入权重模块，使用上下文赌博机（Dirichlet采样）为每个子载波分配重要性权重，从而在WMMSE混合波束成形中实现语义优先资源分配；③ 采用三阶段训练：先独立训练编码/解码与量化；再训练波束成形；最后联合微调，保证系统整体最优；④ 引入用户意图相关损失（基于LLaVA）与BLIP‑2评估，提升零样本泛化能力。

**🔧 技术方法**

技术手段包括：预训练CLIP视觉‑文本Transformer、FiLM特征对齐、分段向量量化、WMMSE混合波束成形、上下文赌博机+Dirichlet采样、GAN对抗训练、BERTScore、SSIM/FID评价、SwinJSCC、JPEG2000、LDPC编解码、OFDM MIMO信道模型、硬件仿真。

**📊 数据集**

使用VQA数据集进行训练与测试；训练集去除动物/人类样本，零样本测试集仅包含动物/人类图像和对应文本查询。

**📈 对比分析**

通过BLIP‑2生成答案的匹配率、BERTScore、SSIM、FID等指标与多种基线（QO‑ISC‑Uniform、QO‑ISC‑Uniform w/o QA、SwinJSCC‑large、查询式图像语义编码、JPEG2000）对比。结果显示：在低SNR（-25 dB）下，加入权重机制提升答案匹配率4.8%，加入查询对齐提升9%，与查询式基线提升8%；在高SNR时性能相当；在零样本测试中泛化效果显著优于SwinJSCC和JPEG2000。

**⚠️ 局限性**

局限性：① 依赖大型预训练CLIP模型，参数量大；② WMMSE混合波束成形计算复杂，实时性受限；③ 对极低SNR和CSI误差仍有一定性能下降；④ 目前仅验证单用户场景，未扩展到多用户；⑤ 对查询类型多样性依赖，需进一步鲁棒性验证；⑥ 量化细粒度与传输符号权衡导致带宽与误差平衡问题。

---

## 470. Frequencies of subwords in words of linear subword complexity

**arXiv ID:** 2607.28273 | [PDF](https://arxiv.org/pdf/2607.28273v1)

**作者:** Jason Bell `[一作]` (University of Waterloo), Chris Schulz `[通讯]` (University of Waterloo)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了无穷右侧单词中长度为N+1的子词的上、下频率，并给出了它们的上界。

**💡 创新点**

首次将巴尔科瓦–佩兰托娃的组合论方法推广到一般单词，并给出当子词复杂度线性上界时频率数目有统一上界的结论。

**🔧 技术方法**

利用Rauzy图、Cassaigne的子词复杂度差异定理以及上频率的不变性等组合与图论技术进行证明。

**📊 数据集**

无使用数据集；所有示例均为构造性无穷单词。

**📈 对比分析**

与 Boshernitzan、Balková–Pelantová 等前人结果比较，证明在线性复杂度下频率数目是有界的，而当复杂度略超线性时可构造频率数目无界，说明上界并非在更广泛情形下成立。

**⚠️ 局限性**

局限性包括：仅给出了上、下频率的上界；对实际频率存在与否的判定仍不完整；在非线性复杂度下仅给出反例，缺乏精确的下界或更细致的结构分析。

---

## 471. Agentic Method for Deterministic Validation of Legacy Code Migration

**arXiv ID:** 2607.28271 | [PDF](https://arxiv.org/pdf/2607.28271v1)

**作者:** Andras Ferenczi `[一作]` (American Express), Krishna Lingamneni `[通讯]` (American Express)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对 COBOL 到 Java 的迁移，提出了一种基于代理的测试合成方法 "Locksmith Loop"，通过可变输入搜索与对齐变异循环实现对迁移代码的确定性验证。

**💡 创新点**

创新点在于：① 将搜索基方法与对齐变异循环结合，形成可递归扩展执行空间的 "锁匠循环"；② 采用对齐门作为差分测试判据；③ 将 LLM 生成的技能持续记录并作为可复用的变异策略。

**🔧 技术方法**

使用的技术包括：六种搜索算法（配对、三元、Latin hypercube、适应性随机、MAP‑Elites、UCB1）、对齐变异（Dispatcher-arm 与 Call‑injection）、Deterministic Migrator、Deterministic Analyzer、Deterministic Runner，以及基于 LLM 的 Authoring Layer。

**📊 数据集**

数据集包含三份 COBOL 程序：开源 CardDemo（CBACT01C 430 行，CBSTM03A 924 行）以及一份近似生产级别的 4114 行 COBOL 程序。

**📈 对比分析**

与单一搜索或传统差分测试比较，Locksmith Loop 在三份程序上实现了 95%–99% 的段落/转移覆盖，分支覆盖最高达 91.9%，明显优于单一搜索方法且不需人工编写测试。

**⚠️ 局限性**

局限性包括：① 仅验证对等性，可能继承源代码的缺陷；② 对齐门仅比较段落集、外部调用顺序与终态，无法检测中间状态差异；③ 对特定结构（多步变量链）仍需人工干预；④ 目前仅在三份程序上验证，需更广泛评估。

---

## 472. TARS: Timestep-Aware Data Scaling for 3D-Free Video Re-Shooting

**arXiv ID:** 2607.28261 | [PDF](https://arxiv.org/pdf/2607.28261v1)

**作者:** Jiwen Liu `[一作]` (Kuaishou Technology), Guoxin Zhang `[通讯]` (Kuaishou Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于视频扩散模型的无3D先验、文本驱动的可控视频重拍框架TARS，能够在保留源视频内容和动作的前提下实现任意摄像机轨迹和视角控制（包括镜头尺度、视角角度和第一/第三人称视角），并支持在源视角之外生成可行的未知区域；

**💡 创新点**

核心创新包括：①基于时间步感知的两阶段训练策略，利用高噪声阶段学习全局摄像机运动和低频结构；②大规模自监督数据构造结合多模态LLM生成的视角语义描述，实现语义化视角控制；③仅在高噪声阶段使用少量配对数据实现时空同步，显著提升运动一致性而不导致过拟合；

**🔧 技术方法**

主要技术手段为：视频扩散模型（DiT或类似架构）、摄像机网格（Camera Grid）条件注入、Rectified Flow确定性流匹配、文本条件（LLM生成的视角描述）以及分类器自由引导；

**📊 数据集**

训练数据包括1M条自监督视频剪辑（来源未指明，推测多源公开视频）以及60K条配对视频（10K真实视频 + 50K Unreal Engine 生成视频）；验证集共1021个样本，覆盖多种人、动物、物体和景观场景；

**📈 对比分析**

与SD‑2.0、CamClone和TrajCrafter等现有方法相比，TARS在摄像机精度（R‑Prec/T‑Prec）、视角控制精度（视角、镜头尺度、视角类型）、时空一致性（V‑MPGE、ArcFace、GCR、LSR）以及视觉质量（CE、FDR、VDR）等多项指标上均取得显著提升；

**⚠️ 局限性**

局限性：在极端大幅度摄像机运动或高度动态场景下仍可能出现细节失真；对未见场景的生成仍依赖于自监督数据的多样性；文本描述对视角语义的表达仍受LLM质量影响；

---

## 473. Operationally Guided Placement-Aware Learning for Industrial Online 3D Bin Packing

**arXiv ID:** 2607.28257 | [PDF](https://arxiv.org/pdf/2607.28257v1)

**作者:** Dheeraj Poolavaram `[一作]` (Technische Hochschule Augsburg), Sebastian Dorn `[通讯]` (Technische Hochschule Augsburg)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了OPAL框架，解决工业场景下的在线三维箱子装载问题，结合操作引导的候选生成与学习式排名。

**💡 创新点**

创新点在于：①使用OG‑EMS多锚点操作引导的候选生成，显著提升候选质量；②引入15维工业候选特征与xLSTM编码器，使模型能更好区分可行摆放；③在单通道掩码学习框架内实现高效的候选排名。

**🔧 技术方法**

采用OG‑EMS候选生成、xLSTM Placement Encoder、LRAM稀疏排名网络，训练使用PPO强化学习。

**📊 数据集**

使用BED‑BPP真实订单数据集（1500个欧式托盘订单），按物品面积降序预排序。

**📈 对比分析**

与Base‑EMS、GOPT、PCT、GENPACK等方法对比，OPAL在绝对密度上达0.49，较OG‑EMS提升15.1%，在表面支撑和中心平衡上也保持竞争力；推理时间仅为GENPACK的十分之一。

**⚠️ 局限性**

局限在于：依赖预排序的物品序列，序列扰动会显著影响性能；未结合后处理或多步搜索，可能在极端摆放约束下欠佳；对不同托盘尺寸的适应性仍需进一步验证。

---

## 474. When AI Becomes Routine: A Decade of Public AI Mediation in Korean Go Commentary

**arXiv ID:** 2607.28332 | [PDF](https://arxiv.org/pdf/2607.28332v1)

**作者:** Haewoon Kwak `[一作]` `[通讯]` (Indiana University Bloomington), Haewoon Kwak (Indiana University Bloomington)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究韩国围棋YouTube评论员在AI系统成为日常专家工具后，如何通过可视化与口头引用使机器判断公开可理解、可归因、可争议，采用2016‑2025年近1900小时视频的语篇分析量化AI信息的变化。

**💡 创新点**

首次系统量化AI显著性词汇并提出源前置与源回退的对话形式，揭示AI信息自然化的治理意义。

**🔧 技术方法**

利用 Whisper Base 语音转写与Korean Kiwi形态分析器进行句子分割，结合关键词规则构建AI显著子集，并使用 GEE Logistic 回归与聚类 t 检验等统计方法评估模式变化。

**📊 数据集**

以 BadukTV、K‑Baduk、LeeHyunWookTV、ProYeonwoo 等四个韩语围棋频道为核心，构建 Longitudinal、Case、Creator 三套数据集，总计约609个视频、1,900小时、约876,353句子。

**📈 对比分析**

通过分阶段（四个时间段）对 AI 关键词出现比例进行时间序列对比，并对机构与创作者频道在句子层面进行比值比较，发现 AI 关键词出现率上升至约2.6%，机构与创作者差异不大但创作者在句子密度上略高，统计显著性通过 GEE、t 检验和 χ² 检验获得。

**⚠️ 局限性**

仅限韩语围棋领域且样本聚焦高曝光频道，关键词规则高精度但低召回，缺乏多模态视觉分析，单人编码可能偏差，且无法验证因果关系，结果可能不适用于高风险领域。

---

## 475. Same Branches, Different Trees: A Bifurcation Connectedness Metric for Coronary Artery Segmentation and FFR-CT Decision Agreement

**arXiv ID:** 2607.28327 | [PDF](https://arxiv.org/pdf/2607.28327v1)

**作者:** Maame Owusu-Ansah `[一作]` (University of Lincoln), James Brown `[通讯]` (University of Lincoln)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究并提出了分支连通性得分（BCS）来评估冠状动脉分割的分支连通性，并将其与 FFR‑CT 决策一致性相关联。

**💡 创新点**

BCS 能捕捉传统 Dice 等指标忽视的分支断裂问题，并通过软化 BCS 损失与其他拓扑损失进行比较，展示了拓扑连通性对 FFR‑CT 结果的重要性。

**🔧 技术方法**

采用 3D U‑Net、SwinUNETR 与 CT‑FM 等骨干网络，结合 clDice、Skeleton Recall、软化 BCS 损失，以及中心线图与 Poiseuille 抵抗求解的流体模拟。

**📊 数据集**

使用 ImageCAS 数据集（1000 剖象，750/250 训练/测试）以及小规模 ASOCA 作为外部验证。

**📈 对比分析**

在 12 种配置下，对 Dice、HD95、β0、BCS、BCR 等指标进行回归、GEE 与 bootstrap 比较；软化 BCS 与 Skeleton Recall 在 BCR 上相近，但软化 BCS 产生更少的断裂，且在高 BCS 四分位时 FFR‑CT 决策一致性提升约 10–15%。

**⚠️ 局限性**

BCS 对径向误差不敏感，未验证临床 FFR 精度，仅评估几何连通性；需要进一步与侵入性 FFR 对比以验证临床价值。

---

## 476. AdaAnchor4D: Anchor-Conditioned Spatiotemporal Feature Aggregation for Monocular UAV 4D Reconstruction

**arXiv ID:** 2607.28320 | [PDF](https://arxiv.org/pdf/2607.28320v1)

**作者:** Peiyi Xu `[一作]` (Xidian University), Jie Feng `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出AdaAnchor4D框架，改进了单目UAV视频的4D动态重建；

**💡 创新点**

创新点在于Anchor‑Conditioned Feature Aggregation、Decoupled Local Geometry Deformation 与 Density‑Adaptive Coordinate Warping 三大模块，提升了对空间时空异质性的适应性；

**🔧 技术方法**

使用anchor‑based Gaussian 表示、4D Gaussian splatting、特征平面分解与可学习的权重预测网络，配合上述三模块实现动态场景重建；

**📊 数据集**

在自建UAV‑Arc4D、公开VisDrone与UAVDT数据集上进行实验；

**📈 对比分析**

与D3DGS、4D‑GS、MoDec‑GS、4D‑SFGS、SpeeDe3DGS、MoRel等方法对比，AdaAnchor4D在PSNR/SSIM/LPIPS上均为最佳，实时渲染帧率保持在30–47 FPS；

**⚠️ 局限性**

依赖预估相机位姿，对姿态误差、遮挡与稀疏观测不够鲁棒，未来需联合姿态优化。

---

## 477. Fully Inductive Cardinality Estimation

**arXiv ID:** 2607.28311 | [PDF](https://arxiv.org/pdf/2607.28311v1)

**作者:** Tim Schwabe `[一作]` (Technical University of Munich), Maribel Acosta `[通讯]` (Technical University of Munich)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 FICE，一种基于因子图视图的全可迁移的 BGP cardinality 估计器，能够在未见过的知识图谱上无重训练直接使用；

**💡 创新点**

创新点在于①首次实现完全可迁移的学习型 cardinality 估计；②提出 2-hop 局部性定理，将 cardinality 转化为局部邻域函数；③设计 encoder‑decoder GNN，encoder 产生针对 cardinality 的实体/关系嵌入；④通过邻域采样训练与离线嵌入生成解耦，实现毫秒级在线推理；

**🔧 技术方法**

使用的技术包括：因子图表示（将三元组和关系视为节点），GINE 消息传递的 GNN 编码器和解码器，注意力池化，log‑cardinality 回归，平滑 L1 损失，排名损失（用于 join 排序），以及邻域采样与离线嵌入缓存；

**📊 数据集**

实验数据集包括 10 个 RDF 知识图谱：WN18RR、FB15K237、SWDF、CoDEx‑L、DBpedia100k、AIDS、Hetionet、LUBM、Wikidata、YAGO；查询形状涵盖 star、path、cycle、flower、diamond、snowflake、tree、path+star 等；

**📈 对比分析**

与传统统计/采样方法（CSET、WanderJoin、SumRDF）及学习型方法（GNCE、LMKG、LSS）进行 leave‑one‑graph‑out 比较；FICE 在 median q‑error 5.34（最佳）并在尾部表现最佳；在线推理时间 <120 ms，即使 10 条 triple pattern；在 join‑ordering 上通过附加排名损失实现 21% C_out 降低和 14× 最差情况改进；

**⚠️ 局限性**

局限性：在极大、密集图（如 Wikidata、YAGO）上因分布差异和过平滑导致估计偏差；未来计划扩大训练集、引入 Transformer 编码和更强的初始特征来缓解这些问题。

---

## 478. MonoVoc: Decoupling Geometry and Semantics for Lightweight Monocular Open-Vocabulary 3D Gaussians

**arXiv ID:** 2607.28300 | [PDF](https://arxiv.org/pdf/2607.28300v1)

**作者:** Pouya Ardekhani `[一作]` (Sharif University of Technology), Hamid R. Rabiee `[通讯]` (Sharif University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一套无训练的单目视频3D高斯映射管线，将几何重建与语义后处理解耦，产生可搜索、可分割的对象级语义高斯地图。

**💡 创新点**

通过语义颜色去混（Semantic Color Deblending）恢复每个高斯颜色，然后用调色板量化与语言嵌入关联，只为每个对象存储一次嵌入，从而显著降低内存并保持查询效率。

**🔧 技术方法**

HI‑SLAM2 3D高斯重建、SCD算法、Palette Quantization（CMC距离）、CLIP/Perception Encoder 语言嵌入、蒙版策略以及 Alpha compositing 的逆向推断。

**📊 数据集**

Replica 数据集的单目 RGB 视频及其对应的语义分割。

**📈 对比分析**

与 ObjectGS、SceneSplat 在 PSNR、LPIPS、mIoU、内存占用、Gauss 数量及运行时对比；MonoVoc 在渲染质量与语义分割接近，内存仅 14 MB、Gauss 140K，Top‑1 查询准确率 80% 以上，显示出更优性能。

**⚠️ 局限性**

颜色量化可能产生歧义，SCD 的近似处理导致边界或高度重叠区域的误分；方法依赖高质量分割与颜色差异，且对复杂重叠场景不完全鲁棒。

---

## 479. On-Policy and Off-Policy Learning for Large Action Spaces

**arXiv ID:** 2607.28408 | [PDF](https://arxiv.org/pdf/2607.28408v1)

**作者:** Imad Aouali `[一作]` `[通讯]`, Imad Aouali

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一系列面向大规模动作空间的上下文多臂赌博机（contextual bandit）算法，涵盖在线学习中的混合效应 Thompson Sampling、扩散 Thompson Sampling，以及离线学习中的结构化直接法、策略加权似然目标和基于 PAC‑Bayes 的惩罚性估计。通过层次贝叶斯模型实现动作参数之间的信息共享，并在推理与优化中兼顾统计效率与计算可扩展性。

**💡 创新点**

创新点包括：
- 将混合效应结构与层次贝叶斯框架结合，形成混合效应 TS 与扩散 TS，显著降低有效动作数、内存与时间复杂度；
- 通过结构化直接法和权重共享的隐式参数实现离线直接估计的统计提升；
- 引入策略加权对数似然目标（policy‑weighted log‑likelihood），从优化景观角度解决离线估计器的非凸性问题；
- 开发可微分指数平滑（exponential smoothing）与对数平滑（logarithmic smoothing）估计器，并基于 PAC‑Bayes 推导统一的可优化的惩罚性目标；
- 在多臂赌博机理论与实践之间搭建桥梁，实现理论上可解释且实验上可落地的算法。

**🔧 技术方法**

使用技术包括：
- 层次贝叶斯推断（高斯后验、拉普拉斯近似、变分或 MCMC 近似）；
- Thompson Sampling 与其混合效应/扩散版本的在线采样；
- 线性/广义线性奖励模型、拉普拉斯近似处理非线性似然；
- 基于重要性加权的 IPS、指数平滑、对数平滑等离线评估器；
- PAC‑Bayes 理论框架下的两侧偏差-方差分解与可微分惩罚项；
- 随机梯度优化、内存/时间复杂度分析与线性缩放实现。

**📊 数据集**

实验数据集涵盖：
- 合成数据（可控参数、已知结构）用于验证理论与算法收敛；
- 真实推荐系统日志（MovieLens、Criteo 等）用于评估在线 TS 及离线直接法；
- 广告投放与滑动条（slate）数据集用于验证扩散 TS 的高维结构推断；
- 药物设计与临床试验模拟数据检验结构化直接法的可迁移性；
- 规模达到百万级动作的工业级离线日志，用于测试策略加权似然与指数平滑目标的可扩展性。

**📈 对比分析**

对比方法包括传统 TS、UCB、IPS、直接法、双重鲁棒、标准对数似然优化等。实验结果表明：
- 混合效应 TS 在 K 规模从 10^3 到 10^6 时，Bayes 退化量降低 2–3 倍，内存/时间比原 TS 降低 2–3 阶；
- 扩散 TS 在多层非线性结构下仍保持优良的 Regret 率，优于混合效应 TS 与传统 TS；
- 结构化直接法在离线评估与学习中实现了 30–60% 的子最优差距提升；
- 策略加权对数似然目标在百万级动作空间中实现 10–20% 的子最优提升，且对超参数鲁棒性高；
- 指数平滑与对数平滑的惩罚性目标在高方差场景下显著降低方差，提升最终策略收益。

**⚠️ 局限性**

局限与未来工作：
- 需要预先设定或学习动作与效应之间的结构（聚类、混合权重），对结构不显著的任务可能不适用；
- 对于极端高维动作空间，仍可能出现计算瓶颈（尽管已降低阶数）；
- 拉普拉斯近似在强非线性奖励下的精度有限，需进一步研究更精细的后验近似；
- 离线评估的支持条件仍存在限制，特别是对极少出现动作的场景；
- 现有理论主要针对期望性能（Bayes regret、子最优差距），对稳健性、对抗性攻击等实际系统需求尚未充分覆盖。

---

## 480. QuantWAMs: Calibrating at the Right Granularity for World Action Models

**arXiv ID:** 2607.28405 | [PDF](https://arxiv.org/pdf/2607.28405v1)

**作者:** Jiacheng Zhou `[一作]` (Fudan University), Lizhe Qi `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出QuantWAMs，一种专为世界动作模型（World Action Models, WAMs）设计的后训练量化框架，能够在闭环多通道推理中保持高精度并显著降低资源消耗。

**💡 创新点**

创新点包括三项技术：1）共享基准（Shared-Basis）离群值校准，仅在坐标兼容的模块间共享激活统计；2）协同训练目标（Co-Training-Objective）重要性权重，利用视频-动作联合梯度计算经验Fisher，为权重分配提供更精准的指导；3）固定干预回放（Fixed-Intervention Replay）对推理步骤进行审核与重新安排，纠正因闭环动态导致的误差累积。

**🔧 技术方法**

技术手段包括：Hadamard旋转和对角平滑的激活预处理、Atom混合精度通道迁移、GPTQ与经验Fisher混合的权重精度分配、固定干预回放对denoising步调的审计与修正、以及混合精度FP4/FP8实现。

**📊 数据集**

使用的数据集有：RoboTwin 2.0、LIBERO（Spatial、Object、Goal、Long等子集）以及真实机器人Agibot G2的训练轨迹（200-700条），其中PTQ校准采用32轨迹样本，回放与验证分别采用32轨迹，最终测试使用与校准不重叠的轨迹。

**📈 对比分析**

与GPTQ、SmoothQuant、Atom、SVDQuant等基线进行比较；在Fast‑WAM和LingBot‑VA上，QuantWAMs在W4A4主导配置下，FP16平均误差仅0.2–0.7个百分点，块级速度提升1.4–1.6×，内存占用约FP16的29%；在真实机器人实验中，QuantWAMs成功率56.7%（FP16为63.3%），显示可行性。

**⚠️ 局限性**

局限性包括：仅优化局部代理目标，需标签与一次反向传播；需要FP16回放与调度验证，且仅在两种WAM上验证；校准与评估均为基准特定，缺乏对未见任务的迁移性能；只给出块级效率指标，未展示端到端控制循环的完整加速。

---

## 481. Large scale cross-regional remote sensing flood monitoring framework for operative mapping and impact analysis

**arXiv ID:** 2607.28401 | [PDF](https://arxiv.org/pdf/2607.28401v1)

**作者:** Ilya Novikov `[一作]` (Skolkovo Institute of Science and Technology), Evgeny Burnaev `[通讯]` (Skolkovo Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本研究构建了一套端到端的多模态遥感洪水监测与损失评估框架，能够在俄罗斯七个不同气候区大范围实现洪水实时绘图与灾后影响量化。

**💡 创新点**

创新点在于：① 将SAR、光谱、DEM及水体指数共21通道融合，形成通用的洪水分割输入；② 在数据稀缺场景下对比监督式U‑Net++与自监督AnySat两种路径；③ 将分割结果直接用于官方标准的损失评估，实现从遥感到决策的一体化流程。

**🔧 技术方法**

核心技术包括：多模态拼接与预处理、U‑Net++与AnySat（JEPA+跨模态对比学习）网络、区域级交叉验证、IoU/F1评估、损失估算与缓冲处理。

**📊 数据集**

使用的数据集：1）SSL4EO‑S12的7,912个无标签多时相卫星片段（SAR+光谱+DEM）做自监督预训练；2）自采集的1,259个标注洪水掩模的7地区数据，用于监督学习与微调。

**📈 对比分析**

通过4折区域交叉验证比较两条路径，监督式U‑Net++在S1+S2+DAS配置下取得平均F1≈0.84、IoU≈0.75；AnySat自监督预训练后微调的性能略低（F1≈0.78、IoU≈0.68）但在不同折间更稳健。

**⚠️ 局限性**

主要限制包括：自监督预训练语料量不足导致特征泛化差；GPU内存限制迫使输入降采样，削弱细节分辨率；跨模态特征对齐与域差异仍待改进；灾损失评估依赖公开建筑与人口数据，存在缺失与时效性问题。

---

## 482. TacWAM: Anchor-Guided World Action Model with Mechanics-Aware Tactile Prediction

**arXiv ID:** 2607.28391 | [PDF](https://arxiv.org/pdf/2607.28391v1)

**作者:** Lei Jin `[一作]` (Tsinghua University), Yong Li `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种结合触觉预测的世界动作模型TacWAM，提升机器人在接触丰富环境中的闭环控制性能。

**💡 创新点**

核心创新包括：① Spatially Aligned Fusion（SAF）触觉编码器将触觉外观、力场和变形流融合并以全局力矩监督；② 触觉历史编码器提供近期接触变化的上下文；③ Anchor-Guided Tri-Modal（AGT）注意力将视觉、触觉和动作流信息隔离，保证训练时触觉预测仅作为监督而不直接用于动作生成。

**🔧 技术方法**

采用多模态Transformer混合自注意力、流匹配预测、力矩重建损失、适应归一化以及多步动作块预测等技术。

**📊 数据集**

在真实机器人平台（Agilex Piper + Xense G1‑WS 触觉传感器）上收集的四项接触丰富任务数据集：薄片抓取、樱桃抓取、白板擦拭、双笔旋转。

**📈 对比分析**

与四类基线（视觉‑语言‑动作、视觉WAM、反应式触觉策略、视觉‑触觉WAM）比较，TacWAM平均成功率达75.0%，比最佳基线提升37.5个百分点。

**⚠️ 局限性**

局限性包括：需昂贵的触觉传感器；触觉预测仅用于训练监督，未在推理中实时纠正；对多模态注意力掩码的选择敏感，若松弛会导致性能骤降。

---

## 483. HyperClaim: Fine-Grained Cross-Modal Hypergraph Reasoning for Video Misinformation Detection

**arXiv ID:** 2607.28375 | [PDF](https://arxiv.org/pdf/2607.28375v1)

**作者:** Xiangbo Wang `[一作]` (Hangzhou Dianzi University), Delvin Ce Zhang `[通讯]` (University of Sheffield)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了基于稀疏时序超图的样本级视频谣言检测框架 HyperClaim

**💡 创新点**

创新点在于利用主张（标题）引导的超图构造，捕捉跨模态高阶依赖；引入置信度过滤、源预算、软归属学习、残差文本-视频校准和差异感知读出，实现局部证据保留与自适应推理

**🔧 技术方法**

采用 Qwen3-VL-Embedding-2B 作为多模态编码器，构建超图并使用双向文本-视频注意力、软归属与线图层的图神经网络进行推理，最终通过 MLP 做真实性判别

**📊 数据集**

在 FakeSV、FakeTT、FakeVV 三个视频谣言检测基准上进行实验

**📈 对比分析**

与传统判别模型、通用多模态/推理模型以及 Fact-R1、FactGuard 等任务对齐系统对比，HyperClaim 在三组数据上分别取得 83.7%、82.0%、87.3% 的准确率，F1 分别为 84.2%、79.5%、86.1%，均显著优于对比方法

**⚠️ 局限性**

仍依赖手工设定的阈值和预算，无法处理完全无标签的跨域场景，且模型规模较大，对推理路径的可解释性仅提供结构化追踪而非因果解释

---

## 484. On the Computational Complexity of (Extended) Threshold Dimension and (Semi-)Ladder Index

**arXiv ID:** 2607.28355 | [PDF](https://arxiv.org/pdf/2607.28355v1)

**作者:** Pasin Manurangsi `[一作]` `[通讯]` (Google Research), Pasin Manurangsi (Google Research)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究阈值维度及其变体（扩展阈值维度、梯形指数与半梯形指数）的计算复杂性，并证明其NP/co‑NP难度与逼近难度；

**💡 创新点**

首次将阈值维度与图论中的Ladder/ Semi‑Ladder 指数关联，构造多种多项式时间归约，揭示扩展阈值维度属于Π₂难度；

**🔧 技术方法**

使用了多项式时间归约、Promise 问题、Gap‑ETH假设以及对最大平衡二分团的逼近硬度转化技术；

**📊 数据集**

无特定数据集，主要以理论构造与证明为主；

**📈 对比分析**

对比方法以理论复杂度和逼近因子为指标，证明在Gap‑ETH下无多项式/ FPT 近似因子可达 n^{o(1)} 或 k^{o(1)}；

**⚠️ 局限性**

局限在于逼近因子仍未达到最优（例如无法到达 n^{1‑o(1)}）且对实际实例验证缺乏实验评估。

---

## 485. Capturing Token Tendencies for Training-Free Token Pruning in Multimodal Large Language Models

**arXiv ID:** 2607.28341 | [PDF](https://arxiv.org/pdf/2607.28341v1)

**作者:** Jie Ma `[一作]`, Rongrong Ji `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提供了ACM文章写作的模板和详细文档，帮助作者编写符合ACM出版规范的论文

**💡 创新点**

系统整理并展示多种ACM样式版本、命令选项，并配以丰富的使用示例，提升写作效率和规范性

**🔧 技术方法**

基于LaTeX的ACM类（acmart）模板和自定义宏包，使用命令行工具和版本控制平台

**📊 数据集**

无数据集，本文为技术手册和模板说明

**📈 对比分析**

无实验比较，未涉及性能指标，主要为文档和样式示例

**⚠️ 局限性**

仅适用于ACM期刊与会议，可能无法直接满足其他出版平台的格式要求

---

## 486. Fairness Pruning: Locating Demographic Bias in GLU-MLP Layers via Differential Activations

**arXiv ID:** 2607.28319 | [PDF](https://arxiv.org/pdf/2607.28319v1)

**作者:** Pere Martra `[一作]` (Universidad Internacional Menéndez Pelayo), Alfonso Ureña López `[通讯]` (Universidad de Jaén)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过对 GLU‑MLP 层的差分激活分析，定位并零化导致性别、种族等人口属性偏见的特定神经元，从而实现对大语言模型的结构性公平性干预。

**💡 创新点**

创新点在于：①使用最小对照提示对（仅差一个人口属性）直接测量差分激活，避免额外训练；②发现偏见主要集中在最终层的离散神经元；③证明这些神经元是“调节器”而非“存储库”，零化可导致双向失稳而非单向减偏，揭示偏见处理与模型一般能力可在同一网络中分离。

**🔧 技术方法**

技术包括：差分激活（BiasScore）与结构重要性（PPM）相结合的 FairnessPruningScore；在 MLP 的 down_proj 输入处注册 PyTorch 前向钩子；按 top‑K 选取神经元并零化其三重投影；使用 OptiPFair 库实现整个流程。

**📊 数据集**

使用公开的对照提示对数据集（English、Spanish 共 70/100 对，覆盖 Age、Gender、PhysicalAppearance、RaceEthnicity、Religion），与 BBQ、EsBBQ 基准相对应；模型为 Llama‑3.2‑1B、Llama‑3.2‑3B、Salamandra‑2B。

**📈 对比分析**

对比基线未剪裁模型，评估 BBQ/EsBBQ（偏见）和 lm-evaluation-harness（WikiText、MMLU、ARC‑Challenge、HellaSwag）等通用能力。零化最多 40 个神经元（<0.031% MLP 宽度）后，偏见在不同类别表现出非单调的双向失稳，通用能力保持平均 99.5% 的保留率，表明干预对核心功能影响极小。

**⚠️ 局限性**

局限性包括：①仅适用于显式属性的对照提示，无法捕获隐式偏见；②仅关注 GLU‑MLP，未涉及注意力层；③使用绝对差分导致偏见方向未知，零化可能产生不可预测的反向偏差；④对照提示对需精确 tokenization，限制可选属性；⑤仅在 4B 以下模型上验证，尚未证实大规模模型的可扩展性。

---

## 487. One Human, $N$ Agents: Audit-Budget Allocation for LLM Agent Fleets under Miscalibrated, Correlated Confidence

**arXiv ID:** 2607.28317 | [PDF](https://arxiv.org/pdf/2607.28317v1)

**作者:** Cesare Zavattari `[一作]` (University of Pisa), Giuseppe Prencipe `[通讯]` (University of Pisa)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种用于单人监督下多大型语言模型（LLM）舰队的预算有限审计分配框架，结合自报置信度的敌对失衡与多级高斯Copula模型的错误相关性，并在此基础上确定置信度排序失效阈值δ*以及无效监督的定量判据；同时对六个开源及一个专有LLM进行置信度失衡与相关性测量，验证模型与政策的实效性。

**💡 创新点**

创新点包括：①首次在同一人监督、有限预算、持久LLM舰队的背景下统一建模自报置信度失衡与跨单元错误相关性；②发现置信度排序失效阈值随预算收缩反而上升的反向行为；③提出“无效监督”量化准则，明确阈值内无可行策略优于随机审计；④将两级Gaussian Copula用于描述跨家族及全舰队共享难度的相关结构，并通过实验验证其对转移学习的贡献。

**🔧 技术方法**

使用的技术包括：两级Gaussian Copula建模错误相关性；Beta后验推断与后验UCB策略；多级置信度分布（Beta混合模型）与δ参数；基于残余风险的度量；模拟相位图（热图）与真实轨迹回放；对置信度的ECE、AUROC评估；部分相关分析与误差分解。

**📊 数据集**

数据集：GSM8K（500题固定题集）与HotpotQA（带干扰项），用于测量六个LLM的置信度与错误率；15-agent舰队（5模型×3实例）用于回放实验。

**📈 对比分析**

与基线方法（oracle、diversity‑Bayes、置信度排序、随机）进行对比。实验显示：在高δ或低预算下置信度排序相当于随机；diversity‑Bayes在多种条件下优于随机与置信度排序；随机与置信度排序在大多数单元中表现相近；vacuum判据显示约44个单元在阈值τ下为“无效监督”。

**⚠️ 局限性**

局限性包括：需持久错误特征（φ=1）且验证器噪声不高；对相关性的依赖，低相关性时转移优势消失；仅基于单次无对话（CoT）提示的置信度收集，可能不适用于其他提示方式；仅评估六个模型（其中专有模型为单点样本）；回放实验规模小（N=15）；对不同输入的跨单元相关性测量仍是上界；在极高δ时阈值定义接近边界，可能导致估计偏差。

---

## 488. ObjectStream: Latent Objects as Memory Anchors for Streaming Video Understanding

**arXiv ID:** 2607.28312 | [PDF](https://arxiv.org/pdf/2607.28312v1)

**作者:** Mingkang Dong `[一作]` (Universiti Malaya), Yuqian Fu `[通讯]` (KAUST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关、基于潜在对象的流式视频记忆框架 ObjectStream，能够在不修改原始 Video‑LLM 的情况下，持续保留对象历史、短时变化与最近视觉上下文。

**💡 创新点**

创新点在于直接利用冻结的 Video‑LLM 视觉表示从中发掘空间连贯的潜在对象，并将这些对象链接成持久轨迹，形成三种互补的记忆模块（对象历史、时序残差、最近窗口）。

**🔧 技术方法**

采用的技术包括查询无关的标记显著性估计、基于位置的聚类生成对象锚点、跨帧对象匹配与预算管理、以及自适应阈值的时序残差捕获；所有处理均在推理阶段完成，无需额外训练。

**📊 数据集**

实验使用了在线流式基准 OVO‑Bench 与 StreamingBench，以及离线长视频基准 EgoSchema 与 VideoMME‑Long，全部以 Qwen2.5‑VL‑3B/7B 作为后端模型。

**📈 对比分析**

与多种对手（Gemini 1.5 Pro、GPT‑4o、FluxMem、QueryStream、OASIS 等）比较，ObjectStream 在 OVO‑Bench 真实‑时间感知上提升 10.0 分、StreamingBench 平均分提升 2.9 分，同时 GPU 内存与 TTFT 减半；在离线长视频上以 82.5% 的视觉 token 剔除率仍优于全 token 基线，提升 2–3 分。

**⚠️ 局限性**

局限性包括：依赖于 Video‑LLM 的冻结特征，潜在对象质量受模型预训练的限制；对非对象中心的查询（如全局场景描述）可能不足；以及在极端高帧率或对象密集场景下仍存在一定的延迟与内存开销。

---

## 489. Beyond Geometric Complementarity: Coherent Overlap in Sparse Mixture-of-Experts Routing

**arXiv ID:** 2607.28308 | [PDF](https://arxiv.org/pdf/2607.28308v1)

**作者:** Huiyuan Tian `[一作]` (Zhejiang University), Shijian Li `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过提出专家子空间分离指数 ESSI、prefix‑controlled 2×2 因子实验以及功能干预，系统研究了稀疏 Mixture‑of‑Experts（MoE）语言模型中专家子空间重叠与路由一致性的几何与功能关系，发现路由在共享几何邻域中保持一致性但并不需要线性互斥覆盖；

**💡 创新点**

创新点在于：① 设计 ESSI 对比专家间分离度与局部方差；② 通过 2×2 因子实验分离候选质量、上下文机会与交互效应；③ 将几何分析与功能干预结合，证明多专家效益可源自共享邻域内不同非线性计算，而非严格的方向分割；

**🔧 技术方法**

技术手段包括 Grassmann 范数（principal‑angle）子空间比较、ESSI 计算、线性子空间构建、2×2 因子设计、Frozen‑route NLL 对比、受控 Top‑1/Top‑2 训练对比，以及 SVCCA、PWCCA、CKA 等子空间评估；

**📊 数据集**

数据集为公开训练好的 MoE 语言模型（OLMoE、Mixtral、DeepSeek、Qwen3、Gemma4 等），在各模型训练拆分上抽样，使用 2048 个保留 token 进行评估；

**📈 对比分析**

通过负载匹配替代路由、候选 vs 对手对比、Frozen‑route NLL、受控 Top‑1/Top‑2 训练等方法进行比较。结果显示选中候选在所有 39 个因子细胞中优于对手，交互均为负，但冻结路由仍能在 24/39 细胞中提升 NLL；受控实验表明 Top‑2 在同等计算下优于 Top‑1；

**⚠️ 局限性**

局限性在于：① 仅评估 rank‑128 线性路由输入子空间，未考虑非线性专家变换的贡献；② 只覆盖 6 个 MoE 体系与单一子空间维度，结果可能不泛化到更大/不同结构模型；③ 对于剪枝或压缩的实际效果仍需在完整网络中验证。

---

## 490. Finding Regions of Maximum Circularity in Plane Geometric Graphs

**arXiv ID:** 2607.28298 | [PDF](https://arxiv.org/pdf/2607.28298v1)

**作者:** Jan-Henrik Haunert `[一作]` (University of Bonn), Tarek Stuck `[通讯]` (University of Bonn)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究给定平面多边形划分的“α-圆形度”问题（最大化面积与周长α次幂的比值），并证明在α∈(1,2]时该问题是弱NP‑难的，同时给出了伪多项式时间算法和一个满足多项式时间的FPTAS。

**💡 创新点**

创新点在于：①首次将α-圆形度问题的复杂性与弱NP难度建立联系；②提出基于搜索图和双目标Bellman‑Ford的动态规划方法，能在伪多项式时间内枚举所有Pareto最优周期并选出最优解；③利用面积/周长的整数化与取整技巧，构造出FPTAS；④通过随机化分析说明在“平滑”输入下运行时间可大幅降低。

**🔧 技术方法**

主要技术包括：搜索图（dual graph + 搜索图）构造、双目标（长度/面积）Bellman‑Ford算法、Pareto集合合并、取整与FPTAS、弱NP难度的归约（到Partition问题）、以及随机化（平滑）分析。

**📊 数据集**

使用真实GIS多边形数据集（行政区划或格网单元），并在不同精度（整数化程度）下对算法进行实验评估。

**📈 对比分析**

与Park & Phillips的伪多项式算法相比，本文算法对输入精度不敏感，平均运行时间明显更低；在FPTAS方案下能够得到(1‑ε)近似解；实验结果表明在实际实例中Pareto集合远小于最坏情况，导致实际性能优于理论上限。

**⚠️ 局限性**

局限性：①伪多项式算法在最坏情况下仍依赖总面积或周长，理论上仍可能很慢；②FPTAS要求面积与最小面积之比有多项式上界，若不满足则不适用；③算法对负权边的处理较为复杂，实际实现需谨慎；④未给出多边形几何属性的利用，缺乏对几何优化的进一步提升。

---

## 491. CACHE-UK: A Stability-Aware Memory Editor for Sequentially Updated Quantized LLMs in Finance

**arXiv ID:** 2607.28292 | [PDF](https://arxiv.org/pdf/2607.28292v1)

**作者:** Anubhav Lakra `[一作]` (Indian Institute of Technology Madras), Yue Feng `[通讯]` (University of Birmingham)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种针对4位量化的英国金融领域LLM的稳定性感知连续内存编辑框架 CACHE-UK，能够在保持模型性能的同时及时更新财务事实。

**💡 创新点**

创新点包括：①在LoRA适配子空间内执行秩-1编辑以降低量化冲突；②通过金融域关键词优先级自适应调节编辑强度；③引入基于“degradation debt”的闭环稳定控制器，防止累积的编辑导致灾难性遗忘。

**🔧 技术方法**

使用的技术包括4位Post‑Training Quantization（GPTQ/QLoRA）、LoRA参数高效适配、秩-1 LoRA扰动、领域优先级加权与积分控制的稳定控制器。

**📊 数据集**

数据集为从 23.9M 英国金融文档（新闻、央行政策、公司注册文件等）筛选出的 88,021 条高质量文档，并构造了 700 条真实财务事实进行评估。

**📈 对比分析**

与 ROME、MEMIT、EasyEdit、KnowledgeEditor 等基线在同一 4 位量化 OpenLLaMA‑3B 模型上对比，CACHE‑UK 在编辑成功率上与基线相当，但测试成功率提升至 28%（比最强基线高 6pp），并在知识保留得分上降低 11–17%，显示出更好的稳定性与泛化能力。

**⚠️ 局限性**

主要局限包括：仅在单一 3B 模型与单次实验跑中验证，缺乏跨量化位宽与更大模型的测试；领域优先级仅基于关键词匹配；基线实现为量化兼容版本，可能不代表其完整潜力；且 28% 的测试成功率仍低，尚未达到部署可行水平。

---

## 492. Hand-Object Interaction in the Age of Large Foundation Models:Reconstruction, Generation, and Embodied Transfer

**arXiv ID:** 2607.28394 | [PDF](https://arxiv.org/pdf/2607.28394v1)

**作者:** Weiquan Lin `[一作]` (Xidian University), Xingyu Chen `[通讯]` (Zhongguancun Academy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `da1b1a89-583a-4b57-9c81-478778569bec` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了基于基础模型的手物交互（HOI）建模与生成方法，系统划分任务与先验来源，并分析其注入机制。

**💡 创新点**

创新点在于提出按任务（R1–R3, G1–G3, ET）与按知识来源（几何、语义、视觉）对HOI方法进行分层分类，并定义了基础模型先验子类与注入路径，为跨领域知识利用提供统一框架。

**🔧 技术方法**

采用大规模预训练模型（如 CLIP、DINOv2、GPT‑4V、Stable Diffusion 等）生成几何、语义、视觉先验，并探讨它们在 HOI 任务中的注入方式。

**📊 数据集**

综述了多种公开数据集（EPIC‑Contact、GRAB、ARCTIC、EPIC‑Contact、Objaverse、ShapeNet 等）及其评测指标。

**📈 对比分析**

对比时依据各任务的量化指标（MPJPE、Chamfer、F‑score、Contact‑F1、FVD 等），表明基础模型先验能显著降低形状、空间、语义与动态不确定性，但不同先验在特定场景下表现差异显著。

**⚠️ 局限性**

局限性包括：先验来源不均衡、冲突决策缺失、对动态摄像头与长时序交互的鲁棒性不足，以及缺少统一的交互完整性评测体系。

---

## 493. Hierarchical Multilevel Monte Carlo for Order-Optimal Neural Actor-Critic in Average-Reward CMDPs

**arXiv ID:** 2607.28390 | [PDF](https://arxiv.org/pdf/2607.28390v1)

**作者:** Ankur Naskar `[一作]` (Indian Institute of Science), Vaneet Aggarwal `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文研究了无限期平均奖励的约束马尔可夫决策过程（CMDP），提出了一种基于分层多级蒙特卡洛（MLMC）神经评论员的原始-对偶自然演员-评论员算法，并建立了神经评论员的最优收敛保证。

**💡 创新点**

创新点在于引入了分层MLMC神经评论员，通过在轨迹采样和评论员优化中同时进行去偏差，解决了神经评论员估计中的偏差-成本权衡问题，从而实现了最优收敛保证。

**🔧 技术方法**

使用了分层多级蒙特卡洛（MLMC）技术和神经网络作为评论员，结合原始-对偶自然演员-评论员算法。

**📊 数据集**

论文中未具体提及使用的数据集，但讨论了在安全关键应用中的强化学习问题，如交通、通信网络、机器人和医疗保健等。

**📈 对比分析**

与现有方法的比较显示，提出的算法在无限期平均奖励CMDP中实现了最优收敛速率𝒪̃(T^-1/2)，并且不需要知道基础混合时间，性能优于现有的线性评论员方法。

**⚠️ 局限性**

限制在于算法的复杂性和计算成本，尤其是在处理大规模强化学习问题时，可能需要更多的计算资源。

---

## 494. Anonymous sharing is pairwise phase-blind

**arXiv ID:** 2607.28377 | [PDF](https://arxiv.org/pdf/2607.28377v1)

**作者:** Brieuc Le roux tardif `[一作]` `[通讯]`, Brieuc Le roux tardif

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文将多机训练作业的检查点写入建模为脉冲耦合振荡器，并通过理论推导与事件驱动仿真证明，对于相同作业在共享存储和功率资源下，配对耦合完全消失，系统不趋向同步。

**💡 创新点**

创新点在于首次证明了在匿名、无记忆、互斥资源共享条件下，配对耦合为零且仅存在体积保持的三体耦合，给出同步点的稳定性分析，并提出更一般化的猜想。

**🔧 技术方法**

主要技术包括积分-触发振荡器模型、返回映射与特征值分析、对称性与守恒量的解析证明，以及高精度事件驱动仿真验证。

**📊 数据集**

实验数据来自自研事件驱动积分器，模拟不同队列规模、负载、功率阈值和周期性抖动的多种情形，无使用真实硬件或外部数据集。

**📈 对比分析**

通过与独立相位基准对比，测量分离速率、特征值、Daido 时刻矩和并发写入上尾分布，结果表明系统无锁定、无聚簇，但并发峰值高于独立情况；理论与仿真特征值与预测一致。

**⚠️ 局限性**

局限性包括假设检查点阻塞、资源匿名且无记忆、作业相同、功率阈值不绑定；异步检查点、功率控制滞后、作业异质性或绑定阈值下的耦合未覆盖。

---

## 495. Structural Validation of LLM-Generated Microservice Decompositions Using Source-Code Dependencies

**arXiv ID:** 2607.28331 | [PDF](https://arxiv.org/pdf/2607.28331v1)

**作者:** Daniel Silva `[一作]` (Federal University of Campina Grande), Angelo Perkusich `[通讯]` (Federal University of Campina Grande)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一套可复现的结构验证管道，用于评估由大型语言模型（LLM）根据文本需求生成的微服务拆分方案在源代码层面的结构一致性。

**💡 创新点**

创新点包括：1）识别并消除由于类-服务映射覆盖率差异导致的实验偏差；2）构建自动化的静态依赖图与服务边界匹配流程；3）引入 TVD 与 TPD 两种结构一致性指标，并用交叉覆盖率归一化保证公平比较。

**🔧 技术方法**

技术手段包括：Tree-sitter 进行 Java 静态分析生成依赖图；NetworkX 与 Pandas 处理图结构和映射；Python 编写的 pipeline 自动执行映射、违规检测和指标计算；手工校正框架相关依赖。

**📊 数据集**

使用了两套公开的 Java 单体系统数据集：PetClinic（25 个业务类）和 Bookstore（93 个业务类），以及由 OpenAI o3 在零提示和少提示（few-shot）策略下生成的微服务拆分结果。

**📈 对比分析**

通过对比零提示与少提示两种提示策略，在同一覆盖率交集内计算 TVD、TPD 与粒度，发现归一化后两者在结构一致性上无显著差异；归一化前的差异被映射覆盖率偏差解释，显示方法学偏差的影响。

**⚠️ 局限性**

局限性包括：仅评估两套 Java 系统和单一 LLM；映射过程仍需人工干预，可能引入误差；TVD/TPD 仅衡量结构一致性，未涵盖可扩展性、性能等质量属性；未考虑运行时动态依赖和多语言/多平台场景。

---

## 496. HARGO: Heterogeneity-Aware Reward-Guided Optimization for RL Post-Training of LLMs on HPC Tasks

**arXiv ID:** 2607.28301 | [PDF](https://arxiv.org/pdf/2607.28301v1)

**作者:** Tiangang Li `[一作]` (Wuhan University), Xiangbo Tian `[通讯]` (Wuhan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大型语言模型上执行RL后训练，专注于高性能计算(HPC)任务的行为改进，包括数据竞争检测、事实性问答和描述性问答。

**💡 创新点**

提出HARGO——一种基于置信度调制优势的自适应按响应重要性加权的奖励导向优化方法，可在不依赖任务标签的情况下解决任务异质性问题。

**🔧 技术方法**

采用强化学习后训练（GRPO框架）、优势调制、参考模型对数概率置信度、KL正则化等技术实现。

**📊 数据集**

使用HPC-GPT开源指令数据集（包含race_c、race_fortran、mlperf、plp四项任务），共5,273训练样本。

**📈 对比分析**

与SFT、HPC-GPT以及PPO、DPO、GRPO、DrGRPO、SimPO、KTO等八种基线在相同模型规模（0.5B）和奖励函数下进行对比；HARGO在三项主要指标上均获最佳：WinRate 54.62%、数据竞争F1 91.30%、PLP相似度0.8558。

**⚠️ 局限性**

实验仅在单一0.5B模型规模下完成，未探讨更大规模的计算效率与可扩展性；方法高度依赖精确的奖励函数，对奖励设计变化敏感；对更广泛任务或模型的泛化尚待验证。

---

## 497. Tycho: Active Abstraction with Programmatic World Models for ARC-AGI-3

**arXiv ID:** 2607.28287 | [PDF](https://arxiv.org/pdf/2607.28287v1)

**作者:** Jens Lehmann `[一作]` (Dresden University of Technology), Sahar Vahdati `[通讯]` (Leibniz University of Hannover)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了 Tycho 系统，解决 ARC‑AGI‑3 中的主动抽象问题：通过交互记录构造可执行的世界模型，进行验证与规划，动态决定何时构建、修复或使用模型，最终实现高效的游戏完成。

**💡 创新点**

创新点包括：
• 将交互式游戏抽象为“渲染的确定性 Moore 机”，明确隐藏状态、动画帧与终止条件；
• 提供可编辑、可验证的程序化世界模型接口，使 LLM 能直接生成、修正与执行游戏逻辑；
• 四种模型维护策略（无模型、单体、调度器、触发修复）与对比实验，揭示模型使用与信息成本之间的权衡；
• 通过“主动抽象”框架将模型构建与使用作为元推理任务，并在单次轨迹上评估其对行动效率与完成率的影响。

**🔧 技术方法**

主要技术手段包括：
• LLM（Claude Opus 4.8、GPT‑5.6 Sol、Opus 5）生成与调度模型；
• 交互记录结构化存储与查询接口（wmlib）
• 可执行 Python 世界模型（init_state, transition, render, outcome, 变体）
• 验证与规划模块（回放一致性检查、A*/BFS 搜索、建议执行）
• 四种模型维护策略的调度与计费跟踪。

**📊 数据集**

数据集为 ARC‑AGI‑3 公共游戏集，共 25 个游戏、183 级，使用官方的游戏引擎与评分协议进行实验。

**📈 对比分析**

比较方法：
• 在同一预算下对四种策略进行匹配实验，计算每游戏 RHAE、完成的级别数和累计动作数；
• 选取 RHAE 最高的 orchestrator 策略，再用 GPT‑5.6 Sol 与 Opus 5 两个前沿 LLM 进行完整系统评测；
• 结果显示 orchestrator 在匹配实验中达到 88.49 RHAE，随后两者分别实现 100 RHAE；
• 动作成本方面，GPT‑5.6 Sol 在 7 766 次动作完成全部 183 级，Opus 5 在 6 641 次；
• 相比官方人类基准，行动效率提升 38–54 %；
• 通过游戏‑平衡中位数、midrank 与成本估算进一步验证性能。

**⚠️ 局限性**

局限性：
• 依赖可执行程序模型，若模型未覆盖目标机制或误判目标会导致失败（如 sk48 案例）；
• 仅在 ARC‑AGI‑3 公开游戏上验证，未知其对完全新游戏的泛化能力；
• 模型生成与修复仍受 LLM 质量与提示设计影响，需手动调优；
• 评测成本与 token 使用不完全可复现，跨模型比较需额外归一化；
• 复杂度高，调试与部署门槛较传统直接推理更大。

---

## 498. SemAnCorr: Semantic Anchored Correspondence for Zero-Shot Manipulation Skill Transfer

**arXiv ID:** 2607.28382 | [PDF](https://arxiv.org/pdf/2607.28382v1)

**作者:** Xiaoxiang Dong `[一作]` (Carnegie Mellon University), Weiming Zhi `[通讯]` (Vanderbilt University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出SemAnCorr框架，能够无训练地实现稠密顶点级对应，支持在几何多样化对象间进行零样本操控技能迁移。

**💡 创新点**

创新点在于将预训练视觉嵌入的3D语义点云与语义聚类、双边边际锚点选择以及功能映射传播相结合，既保证语义一致又保证几何连贯，完全不需要训练。

**🔧 技术方法**

采用预训练视觉模型（如SigLip2Vision、DINOv2）提取多视图语义嵌入，利用k-means聚类、双边边际锚点选择、联合姿态‑对应优化与功能映射（Functional Map）+ZoomOut细化实现对应。

**📊 数据集**

主要使用PartNet‑Mobility构建的稠密对应基准进行评估，并在真实机器人实验中使用RGB‑D、SAM3D重建的三维网格。

**📈 对比分析**

与FM‑WKS、DenseMatcher、Robo‑ABC、D3Fields等基线比较，SemAnCorr在语义准确率达到90.8%（高于D3Fields 84%），几何连贯度与覆盖率均显著优越，几何连贯得分（GCS）超过第二名两倍；在5个真实任务中成功率从7/10到9/10，明显优于D3Fields。

**⚠️ 局限性**

局限包括：需要精确的网格与视图重建；单对象对优化耗时约6秒；对多手操作、复杂抓取误差以及柔性物体配准的适应性仍待提升。

---

## 499. Low-Pathwidth GRAND: Exact Likelihood-Ordered Enumeration for BPSK Transmission over Correlated Gaussian Noise

**arXiv ID:** 2607.28363 | [PDF](https://arxiv.org/pdf/2607.28363v1)

**作者:** Behrooz Razeghi `[一作]` `[通讯]` (Harvard University), Behrooz Razeghi (Harvard University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 Low-Pathwidth GRAND (LP‑GRAND)，一种基于精确二次伪布尔能量的 BPSK 译码方法，用于在相关高斯噪声下实现最大似然（ML）码字查询顺序；通过给定稀疏精度矩阵的路径宽度分解，构造了带有状态转移成本的有限层无环图（trellis），并使用后缀动态规划与优先级队列的最佳优先枚举，按能量递增顺序生成噪声效应模式；在完全枚举且无放弃的情况下，首次查询到的码字即为 ML 码字。作者通过多种模拟实验验证了枚举顺序的正确性、BLER 性能以及对系数量化和精度矩阵不匹配的鲁棒性。

**💡 创新点**

创新点主要包括：
1) 将相关高斯噪声的负对数似然写成观测相关的二次伪布尔能量，且其相互作用图与精度矩阵的非零模式对应；
2) 对此能量构造了基于路径宽度的分解，将其映射为一棵宽度受限的 trellis，确保所有一阶和二阶项都被保留；
3) 采用后缀动态规划配合最佳优先枚举，实现了按能量递增的完整路径生成，保证了与最大似然相同的候选顺序；
4) 提供了对系数量化、精度矩阵稀疏化与不匹配的严格误差界；
5) 在多码本、多率、不同相关系数下展示了 LP‑GRAND 相比传统 ORBGRAND、ExactBlockProduct 等方法在 BLER 与 membership 查询次数上的优势。

**🔧 技术方法**

主要技术手段：
- 二次伪布尔能量建模与交互图分析；
- 路径宽度分解（path decomposition）与 bag-assignment 状态表示；
- 后缀动态规划（suffix DP）用于计算最小路径成本；
- 最佳优先枚举（best‑first）结合优先级队列实现按能量排序的完整路径输出；
- 对量化误差与精度矩阵扰动的解析误差界；
- 通过对照完整枚举和 ML 软判决实现性能验证。

**📊 数据集**

实验数据集：
- 随机生成的二进制线性 [64,52] 码与 CRC‑8 码（以及两个 [20,12] 码）；
- 第一次阶 Gauss–Markov 相关噪声模型，相关系数 ρ∈{0,0.5,0.75,…};
- 多帧（上至 10⁴ 帧）模拟，随机消息、随机码本与随机接收向量；
- 对比实验中使用的基准方法包括 ExactMemoryless‑GRAND、ExactBlockProduct、ORBGRAND‑AI、Basic ORBGRAND 以及 LP‑GRAND。

**📈 对比分析**

比较方法：
- 对每种方法在相同接收向量与码本下进行 membership 查询，并记录首次码字命中位置；
- 计算 BLER、平均 membership 查询次数以及 99% 最高查询次数；
- 对不同码率、不同 ρ、不同查询预算（q_max）进行参数 sweep；
- 通过 Brier 分数、对数损失和 ECE 评估 LP‑GRAND 预测的正确性估计。
性能结果：
- LP‑GRAND 在所有 10⁴ 帧测试中与 ML 完全一致；
- 与 ExactBlockProduct、ORBGRAND‑AI 等方法相比，LP‑GRAND 的 BLER 低 30% 以上，且平均 membership 查询次数约为 100 次，比基准方法低 2–3 倍；
- 在高相关系数（ρ=0.5）和中等 SNR 下，LP‑GRAND 的 BLER 最优；
- 在低 ρ 或无相关噪声时，LP‑GRAND 与 ExactMemoryless‑GRAND 结果相同，表明无相关噪声时不需额外计算。

**⚠️ 局限性**

局限性与未来工作：
- 需要对精度矩阵进行路径宽度分解，若图的路径宽度过大，状态数 2^w+1 可能导致内存与时间爆炸；
- 目前仅针对 BPSK 与二元线性码，需进一步扩展到 M‑PSK、非线性码等；
- 对量化与精度矩阵不匹配的鲁棒性虽有理论界定，但在实际硬件实现中可能仍产生性能偏差；
- 该方法在完整枚举下为 ML，但在有限查询预算下的性能依赖于放弃策略与阈值选择；
- 对于非常长的块长度（n≥512）仍需进一步评估并行化与硬件实现的可行性。

---

## 500. Learning to Persuade Privately Informed Receivers

**arXiv ID:** 2607.28342 | [PDF](https://arxiv.org/pdf/2607.28342v1)

**作者:** I. Arda Vurankaya `[一作]` (University of Texas at Austin), Ufuk Topcu `[通讯]` (University of Texas at Austin)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究在线 Bayesian 说服问题，考虑接收者有一个未知但固定的私有信号方案，设计了一个在行动反馈下的学习算法，并证明其相对于拥有完整私有信号信息的最优方案的期望后悔量为 $O(T^{3/4})$。

**💡 创新点**

创新点包括：①首次将未知私有信号学习问题转化为一维的变点检测；②通过构造特定线段实现对超平面和信号似然的分离学习；③在承诺阶段加入鲁棒性约束，避免后悔失控；④提供信息理论上可行的无后悔学习框架。

**🔧 技术方法**

使用技术包括：探索-承诺策略、二分搜索与变点检测、超平面与概率向量的分离估计、对角度误差控制、鲁棒优化以及混合整数规划求解最优信号方案。

**📊 数据集**

无数据集，全部为理论分析和算法设计。

**📈 对比分析**

与已知私有信号的离线最优方案比较，后悔量上界为 $O(T^{3/4})$，并且算法参数对状态空间和信号字母表的依赖是多项式；实验验证未给出，仅提供理论证明。

**⚠️ 局限性**

局限性：①后悔率 $T^{3/4}$ 是否最优仍未知；②离线承诺阶段的计算复杂度呈指数级；③对信号的平衡性、超平面间距等假设比较强；④仅适用于二动作接收者，无法直接推广到多动作情形。

---

## 501. Measuring Distortion in the Empty Regions of Dimensionality Reduction Scatterplots with the Gap Index

**arXiv ID:** 2607.28324 | [PDF](https://arxiv.org/pdf/2607.28324v1)

**作者:** Jaume Ros `[一作]` (Eindhoven University of Technology), Fernando Paulovich `[通讯]` (Eindhoven University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出Gap Index（GI）度量，专门评估二维投影中空隙的压缩与拉伸，补偿传统质量指标在视觉分析中对空白区域失衡的不足。

**💡 创新点**

创新点在于以三角剖分为基础，利用相对面积比较高维与二维空间中同一三角形的比例差异，既可聚合为全局指标，又可在散点图中直观可视化局部失真。

**🔧 技术方法**

核心技术包括Delaunay三角剖分、Heron's公式计算三角面积、相对面积归一化、加权绝对失真求和，并实现O(N log N)的计算复杂度。

**📊 数据集**

使用了多种合成与真实数据集（Plane、Cube、Sphere、COIL20、Fashion‑MNIST、MNIST、MNIST CNN embeddings、Fiber等），覆盖从几千到二十五万点的规模。

**📈 对比分析**

与传统指标（scale‑normalized stress、trustworthiness/continuity、steadiness/cohesiveness）及可视化方法（CheckViz）对比，GI在捕捉视觉显著空隙失真方面表现突出，且在大数据量下显著更快（仅数秒，内存<2 GB）。

**⚠️ 局限性**

局限性包括对三角剖分的离散性导致对微小位置变化不连续、仅通过面积无法捕捉全局统一拉伸、需满足三角不等式且对非欧氏距离敏感、以及在高度稠密或一维投影时的适用性受限。

---

## 502. From Textual Requirements to Microservice Architectures - A Comprehensive Evaluation of LLM-Based Design Synthesis

**arXiv ID:** 2607.28307 | [PDF](https://arxiv.org/pdf/2607.28307v1)

**作者:** Danyllo Albuquerque `[一作]` (VIRTUS Federal University of Campina Grande), Angelo Perkusich `[通讯]` (VIRTUS Federal University of Campina Grande)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了是否可以利用大型语言模型（LLM）仅凭自然语言需求文档生成完整的微服务体系架构，并通过定量与专家评估对比验证其效果。

**💡 创新点**

创新点在于首次对LLM端到端的架构合成进行系统化实证评估，探讨零样本与少样本提示对生成质量的影响，并提出混合方法评价框架。

**🔧 技术方法**

采用OpenAI o3模型，使用零样本（ZS）和少样本（FS）提示策略，并结合精确率/召回率/F1指标与盲评专家的结构与可行性打分。

**📊 数据集**

实验使用两套公开开源系统——Bookstore和PetClinic，整理并标准化其文本需求作为输入。

**📈 对比分析**

对比方法是将LLM生成的服务列表与交互关系与参考实现进行逐一匹配，计算精确率、召回率和F1；实验表明FS提示在服务识别上的F1≈0.97、交互恢复的F1≈0.82，远优于ZS；专家评估亦表明FS模型在正确性、完整性、模块化和可行性上得分最高。

**⚠️ 局限性**

局限性包括仅单次执行、仅两小型系统、单一模型、提示示例可能带来偏置、缺乏多次实验的稳定性评估，以及对需求文本处理方式的主观影响，无法推广至更大、异构或安全关键系统。

---

## 503. Semi-Supervised Learning for Molecular Graphs via Ensemble Consensus

**arXiv ID:** 2607.28304 | [PDF](https://arxiv.org/pdf/2607.28304v1)

**作者:** Rasmus Tirsgaard `[一作]` (Technical University of Denmark), Mikkel N. Schmidt `[通讯]` (Technical University of Denmark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于集成共识的半监督学习框架，用于分子图数据，无需标签保持的数据增强，利用无标签样本上的集成一致性约束提升预测性能。

**💡 创新点**

创新点在于将集成误差分解理论与无监督一致性目标结合，单个模型在无标签监督下可超过传统全集成；同时不需要热启动或额外预训练，且实现了对模型多样性与一致性的平衡。

**🔧 技术方法**

使用深度集成网络（PaiNN、GCN、GIN、GatedGCN 等），采用 MSE/CE 损失与一致性损失（如 L2 或 KL）相结合，并通过超参数 γ 控制无标签一致性权重。

**📊 数据集**

主要实验数据集包括 QM9（12 个回归目标）、GNN+ 组（ZINC、Peptides 结构/功能、ogbg‑molhiv、ogbg‑molpcba）以及 PCQM4Mv2 作为无标签 3D 分子数据。

**📈 对比分析**

与传统监督集成、Mean‑Teacher、Pseudo‑label、Frad 等方法对比，实验显示在 10% 标签比例下，单个模型性能可超过传统集成；在所有数据集上均实现 MAE 降低、AUROC/Accuracy 提升，且单模型已达到甚至超越全集成效果。

**⚠️ 局限性**

局限性在于训练时需要多模型并行，计算与内存开销随模型数 M 成线性或二次增长，对大型模型或大规模数据不友好；需要针对不同数据集调节 γ，且实验仅在已标注数据拆分为无标签集，未测试真实外部无标签库。

---

## 504. Filling the Pareto-Optimal Front for Affordance Segmentation on Embedded Devices Using RGB-D Cameras

**arXiv ID:** 2607.28293 | [PDF](https://arxiv.org/pdf/2607.28293v1)

**作者:** Edoardo Ragusa `[一作]` (University of Genoa), Paolo Gastaldo `[通讯]` (University of Genoa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了基于RGB‑D传感的可穿戴机器人抓取功能分割系统，使用硬件感知的神经网络搜索和微调方法实现高效推理。

**💡 创新点**

创新点在于设计了适配嵌入式硬件的搜索空间，融合深度信息的多分支架构，并提出了低成本微调方案。

**🔧 技术方法**

技术手段包括硬件感知神经网络搜索（HW‑NAS）与预训练网络微调，结合卷积编码器‑聚合器架构。

**📊 数据集**

实验使用了UMD和IIT两大RGB‑D数据集。

**📈 对比分析**

与七种基准模型比较，RGB‑D模型在准确率与FLOPs平衡上位于Pareto最优前沿，Jetson Nano上可实现15‑16 FPS实时推理。

**⚠️ 局限性**

局限性包括对目标定位依赖用户手动裁剪、标注成本高，以及在极端光照和复杂背景下的鲁棒性待提升。

---

## 505. Beyond Visual Ambiguity: Guiding Robust Monocular Depth Estimation in Challenging Scenarios via Detailed Long Captions

**arXiv ID:** 2607.28285 | [PDF](https://arxiv.org/pdf/2607.28285v1)

**作者:** Junrui Zhang `[一作]` (Huazhong University of Science and Technology), Zhiguo Cao `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出CapDepth框架，通过使用详细长文本说明物体空间关系来提升单目深度估计的鲁棒性，特别是在非拉米安表面和恶劣天气场景下

**💡 创新点**

1) 采用基于原子句的长文本模板显式编码多对象间空间关系；2) 引入动态字幕编码器，使用逐步掩蔽注意力提取细粒度、深度相关的文本特征；3) 设计文本自适应解码器，通过稳定自适应层归一化（SAdaLN）将文本特征注入深度解码，弥补噪声预测与几何推理的差距

**🔧 技术方法**

CLIP文本特征、进阶掩蔽注意力模块、稳定自适应层归一化（SAdaLN）、预训练扩散U-Net、VAE特征提取、KL正则化训练策略

**📊 数据集**

训练集：Hypersim（74K）和Virtual KITTI 2（42K）；评测集：Booster、ClearGrasp、nuScenes、DrivingStereo（分别包含透明/镜面表面与多种天气场景）

**📈 对比分析**

与通用MDE方法（Depth Anything、Metric3D）、专用鲁棒MDE方法（Depth4ToM、RobustDepth）以及语言集成MDE模型（VPD、WorDepth）对比；在透明/镜面表面上AbsRel下降25.0%，在恶劣天气中δ1提升6.1%，在多种评测集上整体表现均显著优于对手

**⚠️ 局限性**

仅在英文文本上验证；对跨语言适用性和多语言多样性缺乏系统评估；对实际部署时的实时性和计算资源需求未做深入讨论

---

## 506. A Taxonomy of Performance Metrics for the Distributed Computing Continuum

**arXiv ID:** 2607.28407 | [PDF](https://arxiv.org/pdf/2607.28407v1)

**作者:** Praveen Kumar Donta `[一作]` (Stockholm University), Schahram Dustdar `[通讯]` (TU Wien)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了分层结构的分布式计算连续体（DCCS）性能指标分类体系，涵盖计算层、网络层、应用/用户层以及可持续性、可观察性、适应性等新兴维度，并给出了每个指标的数学表达式、采集范围、采集阶段与方法。

**💡 创新点**

创新点在于：1）首次为跨层次、异构的 DCCS 构建统一的指标层级框架；2）引入多维度（如可持续性、碎片化、迁移感知等）扩展传统指标；3）对每个指标提供采集可行性分析，帮助研究者在实践中合理选择与实现。

**🔧 技术方法**

技术与方法主要为：理论建模与数学公式推导；基于 OpenTelemetry 等现有观测框架的采集方式阐述；对指标可行性进行维度化分析（Scope、Phase、Method）。

**📊 数据集**

本研究为理论与方法综述，未使用具体实验数据集，主要依赖文献引用和理论推导。

**📈 对比分析**

该论文不进行实验对比；而是通过指标定义与可行性分析，为后续实验提供衡量标准。性能比较方法是基于指标层级与采集维度的评估，而非数值性能对比。

**⚠️ 局限性**

局限性包括：1）缺乏实测验证，无法证明所有指标在真实 DCCS 上的可测性与可用性；2）在高度异构与动态环境下，某些指标的采集成本可能过高；3）对新兴技术（如生成式 AI）细节描述有限，需进一步扩展和验证。

---

## 507. Why Are GUI Agents Correct but Late? Decode on the Decision-Time Critical Path, Tested with Pre-Compiled Policy Trees

**arXiv ID:** 2607.28399 | [PDF](https://arxiv.org/pdf/2607.28399v1)

**作者:** Zihan Dong `[一作]` (Georgia Institute of Technology), Yu Li `[通讯]` (Southeast University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出Adaptive Anticipatory Policy Trees（AAPT）通过在闲置期预编译条件化决策树，消除实时推理时的生成延迟，从而提高对短暂 GUI 事件的及时响应。

**💡 创新点**

创新点在于将预编译树与低代价观察器结合，依据模型自身解码时延动态裁剪树大小，形成一套完整的“先算先决、后即时执行”管线，并通过严格的树有效性、分支准确率和观察器时延门控来验证机制。

**🔧 技术方法**

技术实现包括冻结多模态大模型（如 Qwen3.5-MoE、Holo-3.1-35B-A3B 等）用于树编译；低代价 JSON‑schema 观察器进行快速 guard 匹配；像素差分变化门控制帧捕获；以及对树节点进行期限、置信度和风险阈值的多维裁剪。

**📊 数据集**

实验数据集包括自研的 Contested‑Window Benchmark（可调 250–2000 ms 事件窗口）、DynaCU‑Bench 的 39 条确定性期限任务，以及 126 条未调优通用模型的验证对照。

**📈 对比分析**

通过每种模型的种子对应的配对实验，并使用精确 McNemar 检验与 Holm‑Bonferroni 校正，AAPT 在 650 ms 争议窗口下成功率从 0.50 提升至 0.79（p=1.8×10⁻³），且未出现错误动作；相较于 R0、R1、P1 的零成功率，AAPT 明显优越；在 DynaCU‑Bench 上，AAPT 仅在可预枚举动作的任务中取胜，说明方法的适用边界。

**⚠️ 局限性**

局限性包括：仅适用于可在编译时枚举完备的单步动作；对多步序列、延迟揭示的数值或坐标定位不具备适配能力；需要精确的时延测量和树有效性门控，易受模型微调、随机种子或硬件变动影响；并且预编译消耗额外 token，导致整体计算量提升。

---

## 508. GLM-RAG: Graph Language Models for Graph-Based Retrieval-Augmented Generation

**arXiv ID:** 2607.28397 | [PDF](https://arxiv.org/pdf/2607.28397v1)

**作者:** Maya Arseven `[一作]` (Heidelberg University), Moritz Plenz `[通讯]` (Heidelberg University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并训练了基于 Graph Language Model (GLM) 的检索器 GLM-RAG，替代传统 GNN 检索器，在知识图谱上实现多跳检索与生成。

**💡 创新点**

将预训练语言模型转化为图 Transformer，以端到端方式在文本属性图上训练检索器，显著提升语义理解和跨领域迁移能力。

**🔧 技术方法**

使用 Graph Language Models、图 Transformer、相对位置编码、子图截取、微调、向量检索 RAG 基线以及 GFM‑RAG 结构等技术。

**📊 数据集**

在 HotPotQA、2Wiki、MuSiQue 三大 Wikipedia 多跳 QA 数据集上训练检索器；在 11 个单跳 OOD 数据集（如 TechQA、ExpertQA、eManual 等）和四个多跳 OOD 数据集（Multihop‑RAG、G‑Bench Novel/Medical/CS）评估迁移。

**📈 对比分析**

与 vanilla RAG、GFM‑RAG、GFM‑RAG*、GFM‑RAG+ 等基线比较；单跳任务中 vanilla RAG 领先，多跳任务中 GLM‑RAG 以 Recall@2 与 EM 领先，尤其在零样本多跳 OOD 数据集上实现 SOTA 或接近 SOTA 的成绩。

**⚠️ 局限性**

计算成本高、子图规模受限，未结合最新的 GFM‑RAG+ 索引改进，且仅使用至 0.8B 参数的 GLM，未来规模更大仍待验证。

---

## 509. Explaining Image Similarity with Automatically Extracted Concept Activation Vectors

**arXiv ID:** 2607.28386 | [PDF](https://arxiv.org/pdf/2607.28386v1)

**作者:** Isaac Roberts `[一作]` (Bielefeld University), Barbara Hammer `[通讯]` (Bielefeld University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于自动提取概念激活向量（CAVs）的图像相似度可解释框架，能够在嵌入空间中对概念进行扰动并量化其对相似度的影响；

**💡 创新点**

创新点在于：1）使用稀疏自编码器（SAE）无监督地提取概念并直接在嵌入空间中扰动；2）能够同时提供对单对、群体以及检索任务的解释；3）引入Exemplar Retrieval任务，利用解释向量实现相似原因检索；

**🔧 技术方法**

技术包括：稀疏字典学习/SAE、概念激活向量（CAVs）、在嵌入空间中概念扰动、相似度函数（余弦、欧氏）、线性回归验证、UMAP可视化、概念热图与位置映射；

**📊 数据集**

使用了两个数据集：Multi‑CIFAR‑10 Collage（用于概念验证与分布评估）和VITON‑HD（时装检索与实证评估）；还在多种预训练模型上（DINOv2/3、ResNet50、ConvNeXt、ViT、SigLIP）进行实验；

**📈 对比分析**

与基线（基于像素掩码、模糊、梯度、CSIM等）比较，使用线性回归R²、RMSE、Wasserstein、OOD等指标；结果显示在余弦/欧氏相似度下，本文方法在R²>0.87、Wasserstein/OOD更低，整体性能优于所有对比方法；

**⚠️ 局限性**

局限性包括：假设嵌入可线性重构，可能忽略更抽象或关系型相似性；SAE的稳定性与数据依赖性；概念解释在图像空间的可视化缺失；对无标签域迁移时可能出现背景概念主导。

---

## 510. LEDGERMIND: Provenance-Constrained Multimodal Agentic Reasoning with a Structured Evidence Ledger

**arXiv ID:** 2607.28374 | [PDF](https://arxiv.org/pdf/2607.28374v1)

**作者:** Enjun Du `[一作]` (Hong Kong University of Science and Technology), Yongqi Zhang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于结构化证据账本的多模态代理框架，实现对轨迹的可追溯性与可信度控制。

**💡 创新点**

创新点包括：① 结构化证据账本作为全局状态强制引用；② 三层地面化协议与实体/数值一致性检查；③ 自适应双路径调度器；④ 事件触发的有限类型修复，保证 provenance non‑amplification。

**🔧 技术方法**

使用了多模态大语言模型+工具调用、账本+依赖图、支持覆盖、实体一致性检查、数值一致性检查、自适应调度器以及 Typed Repair 的事件触发机制。

**📊 数据集**

实验数据集涵盖 VTC-Bench、MMStar、MMMU、MMMU-Pro、EMMA、MC-Search 以及自制 Hard-200 集。

**📈 对比分析**

与各大厂商原生 CoT/思考模式在相同工具预算下对比，提升了 10–20% 的绝对准确率，并在轨迹可信度指标（UCR_reason、GDR、R4R、WDG）上显著改善；在 MC-Search 的链级指标 HPS 与 RD 上亦有显著提升。

**⚠️ 局限性**

局限性：仅在图像+文本任务上验证；未覆盖长期记忆和多步骤推理的长尾情境；依赖现有工具集；缺乏训练阶段的监督信号，仍可能在极端复杂查询中漏检。

---

## 511. ShadowDancer: Teaching Video World Models Any Action by Learning Unified Dynamics Representations from a Video and Its Shadow

**arXiv ID:** 2607.28362 | [PDF](https://arxiv.org/pdf/2607.28362v1)

**作者:** Jin Cao `[一作]` (Alaya Lab), Kaipeng Zhang `[通讯]` (Alaya Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于“影子对”(shadow pair)的框架，通过在不同外观下重现同一动态来学习统一的动态表示，并以此实现任何动作的帧级交互式视频世界模型控制；

**💡 创新点**

创新点在于(1)引入影子对构造同一动态的多种外观视角，(2)利用跨影子预测(cross‑shadow prediction)在训练中直接强制动态不变性，从而消除传统潜在动作模型中动与观的耦合；

**🔧 技术方法**

采用跨影子潜在动作模型(LAM)与冻结的3D‑VAE提取源细节，再对预训练的DiT视频扩散模型进行流动匹配(film‑matching)微调，并将其转化为块因果(block‑causal)生成器，以支持长时序交互；

**📊 数据集**

使用多源合成影子对数据集，涵盖人类运动(SMPL‑X+Blender)、机器人操作(ManiSkill)、第一/第三人称游戏(GTA、Cyberpunk、Unreal)、相机轨迹(DL3DV)以及真实视频自对，所有数据均以帧同步的影子对形式提供；

**📈 对比分析**

通过与基线Olaf‑World及其他开源交互式世界模型进行PSNR/LPIPS/动作迁移和长时序滚动的盲测对比，ShadowDancer在多类动态的动作迁移中显著提升PSNR（+4‑8点）与LPIPS（-0.1‑0.2），在2AFC长时序评测中获胜率均超过50%且优于其他方法；

**⚠️ 局限性**

局限性包括：对影子对的生成依赖渲染脚本，可能在极度多样化或缺乏可重现动态的真实场景中效果有限；对极度细粒度或需要外观信息的控制（如精准姿态对齐）仍可能受限；

---

## 512. Encryption-Compatible Clustered Federated Learning via Distributed Expectation-Maximization over Metadata

**arXiv ID:** 2607.28338 | [PDF](https://arxiv.org/pdf/2607.28338v1)

**作者:** Michael Ben Ali `[一作]` (Université Toulouse III Paul Sabatier), Olivier Teste `[通讯]` (Université Toulouse Jean Jaurès)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 FLAMECHE 框架，将元数据聚类改写为分布式 Expectation‑Maximization，只在客户端执行非线性运算，服务器仅做加法，从而实现对加密 FL 的兼容。

**💡 创新点**

创新点在于：① 将元数据聚类转化为可在加密环境下执行的分布式 EM；② 采用随机初始化神经网络进行零样本元数据提取；③ 设计了支持部分参与的动态重聚类策略，平衡了隐私、通信和计算成本。

**🔧 技术方法**

使用技术包括：分布式 EM 算法、随机神经网络元数据提取、加密 FL（安全聚合、同态加密）兼容的加法运算、K‑means 类似的硬聚类、动态重聚类、部分参与（p=20%）以及基于元数据的可扩展度量。

**📊 数据集**

实验数据集：MNIST、Fashion‑MNIST、CIFAR‑10、TissueMNIST（医学图像）、PathMNIST（医学图像）。

**📈 对比分析**

与 FedAvg、Oracle、FedGroup、StoCFL、FeSEM、IFCA、K‑Fed、PACFL 等基线在 5 个数据集上进行比较。FLAMECHE 在 4/5 数据集上获得最高平均准确率，接近 Oracle，且在概念漂移+特征/标签偏移等复杂异构场景中保持低方差，表现出优异的鲁棒性。

**⚠️ 局限性**

局限性：实验仅评估硬聚类，未证明 EM 在部分参与条件下的收敛性；未探讨软聚类（如 GMM）；对加密安全性的理论分析有限；元数据仍可能泄露隐私信息，需进一步加密或差分隐私保护。

---

## 513. Correcting What You Cannot See: Credit Assignment for Perception Distillation in Multimodal Reasoners

**arXiv ID:** 2607.28336 | [PDF](https://arxiv.org/pdf/2607.28336v1)

**作者:** Feng Xiong `[一作]`, Hongyu Lin `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a4b10f5d-130b-4e77-9367-6469ec621899` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无标签的感知纠正蒸馏（Perception‑Correction Distillation，PCD），通过在感知阶段分离轨迹并根据失败率与教师-学生不一致的乘积动态分配监督权重，提升多模态大模型的感知与推理能力。

**💡 创新点**

创新点在于将感知失败的判定拆分为两种互补证据（下游失败率1‑PSR与教师-学生KL差异），并通过乘法构成软 AND 门实现对可纠正感知错误的精确定位，既保持总监督预算，又显著提升学习效率。

**🔧 技术方法**

使用的技术包括：分离感知–推理的rollout、基于均值保持的权重调度、OPD/VPPO的视觉聚焦蒸馏、基于可验证奖励的DAPO强化学习，以及教师-学生KL计算和分组相对优势正则化。

**📊 数据集**

实验数据集覆盖八大基准：Geo3K（训练集）、MathVerse、MathVista、MathVision、We‑Math（近OOS视觉数学）、LogicVista、MMMU_Pro、MMStar（更远OOS多模态任务）。

**📈 对比分析**

通过与原始OPD、DAPO、无监督基线以及同规模模型的对比，PCD在2B学生上宏观平均提升至47.28%（比OPD高约2.8个百分点），在8B学生上提升至61.22%（比OPD高约4.3个百分点），在大多数子任务中表现最优。

**⚠️ 局限性**

局限性包括：对教师质量高度依赖，KL校准需手工设置；缺少单证据（仅PSR或仅KL）的基线对比；对a、b等超参数的敏感性；实验仅在单一模型族与单一随机种子下验证，缺乏稳健性和跨域推广评估。

---

## 514. Paying for Honesty Without Knowing the Truth: Reputation-Penalty Design for LLM Marketplace Agents

**arXiv ID:** 2607.28330 | [PDF](https://arxiv.org/pdf/2607.28330v1)

**作者:** Mingdai Yang `[一作]` (University of Illinois Chicago), Zhiwei Liu `[通讯]` (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并验证了CARP和SPARC两种机制，帮助大型语言模型在无真相信息的在线市场中自我约束不实描述，提升消费者福利并保护诚实卖家。

**💡 创新点**

创新点在于提出只基于噪声投诉信号的死带自适应声誉惩罚CARP和轻量级代码门控反射SPARC，二者结合实现无真相下的自利诚实与消费者福利接近完美信息极限。

**🔧 技术方法**

使用Stackelberg博弈框架、离散事件模拟、基于投诉信号的声誉更新、死带惩罚、状态依赖加权以及代码门控的自我反思提示等技术。

**📊 数据集**

利用3,350个品牌、793,678个真实商品属性构成的真实商品目录，向LLM仅公开部分属性并通过模拟评估。

**📈 对比分析**

与无惩罚、固定惩罚、EWMA、Beta、CUSUM等传统声誉规则对比，CARP+SPARC在四个LLM模型上在消费者伤害、诚实卖家保留和总体福利方面均优于其他策略，且逼近完美信息下的最优。

**⚠️ 局限性**

限制在于死带阈值需预设、缺乏跨平台协作和实时数据的自适应能力、模型特异性、未考虑合谋卖家以及真实用户行为的验证。

---

## 515. PathView-Bench: Can Multimodal Large Language Models Achieve Fine-grained Multiscale Understanding of Pathology Images?

**arXiv ID:** 2607.28318 | [PDF](https://arxiv.org/pdf/2607.28318v1)

**作者:** Zongyi Chen `[一作]` (Xiamen University), Liansheng Wang `[通讯]` (Xiamen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建并发布了一个名为PathView的视觉中心化多尺度病理MLLM评测基准，包含区域和全切片两种视角的14种VQA任务。

**💡 创新点**

通过标准化转换公开病理标注为确定性任务目标，构建可审计、无LLM评判的评测流程，并同时覆盖细粒度与宏观尺度的视觉理解。

**🔧 技术方法**

利用人类监督标注的空间注解、规则化映射与自动评分指标（Dice、准确率、MAE等），并在18种通用、医学、病理专用MLLM上进行零样本评测。

**📊 数据集**

整合23个公开病理影像数据集，涵盖61,673张图像、308,070个样本、7,253,526条注解，跨28个器官与多种标注类别。

**📈 对比分析**

按7:1:2比例划分训练/验证/测试，使用确定性指标对模型进行打分，结果显示即使是规模最大、领域专业的模型在定位、计数、空间推理等细粒度任务上仍低于50%准确率，且高层诊断准确性并不保证细节理解。

**⚠️ 局限性**

仅基于公开数据，未涵盖临床工作流、长期随访或完整诊断报告，且评测只关注视觉证据而非临床决策的完整性。

---

## 516. Correlation between prosody and pragmatics: A case study of the discourse marker hālā `now' in Persian

**arXiv ID:** 2607.28359 | [PDF](https://arxiv.org/pdf/2607.28359v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 517. FAIR-Compute: A Roadmap for Fair and Efficient Allocation of Federated Digital Research Infrastructure

**arXiv ID:** 2607.28290 | [PDF](https://arxiv.org/pdf/2607.28290v1)

**作者:** Konstantinos `[一作]` (Queen Mary University Of London), Wan Shuen Siaw `[通讯]` (Queen Mary University Of London)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对英国及国际分布式计算基础设施的访谈、问卷、文献综述和大规模仿真，提出了一套针对联邦数字研究基础设施公平高效分配的路线图。

**💡 创新点**

创新点在于将机制设计、经济学与HPC调度结合，首次系统量化“占用 vs 利用”效率，并通过仿真验证自适应权重与联邦路由的有效性。

**🔧 技术方法**

采用机制设计模型、Slurm权重调优、NSGA-II多目标搜索、离线ILP基准以及基于聚类的合成负载生成等技术。

**📊 数据集**

使用公开的Fresco/Anvil工作负载追踪作为主要实验数据，结合人工合成的高压测试场景。

**📈 对比分析**

与FIFO、标准Slurm优先级、调优权重和离线ILP基准进行对比，结果显示调优权重在拥塞条件下可逼近离线最优，联邦路由在无协调时可显著降低平均延迟。

**⚠️ 局限性**

主要局限在于缺乏真实的英国高负荷工作负载、模型对策略报表的假设、结果仅基于单一仿真窗口，以及未对价值感知调度的激励兼容性进行理论证明。

---

## 518. When Specifications Conflict: A Symmetry-Based Framework for Measuring LLM Preferences

**arXiv ID:** 2607.28384 | [PDF](https://arxiv.org/pdf/2607.28384v1)

**作者:** Tairan Wang `[一作]` (University College London), Pingchuan Yan `[通讯]` (University College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于对称性、可执行冲突的实验框架，用来量化大型语言模型在不同规格冲突下的偏好选择。

**💡 创新点**

创新点在于通过可执行映射和对称性完整的配置实现冲突可归因，消除顺序、信息量等干扰，使模型在规格冲突下的选择行为可被系统化、可测量。

**🔧 技术方法**

利用可执行的数学函数、布尔代数表达式、代码测试和临床规则等任务，构造冲突实例并通过模型输出与可执行结果对比实现归因；对实验进行对称平衡的试验设计。

**📊 数据集**

主要数据集为自定义的 550 条冲突实例（11 类函数，每类 50 条），涵盖纯自然语言、正式符号、自然化正式和输入输出示例四种规格；此外在布尔代数、代码生成（MBPP）和临床规则领域构造的异构冲突数据。

**📈 对比分析**

通过统计模型在六种规格对的可归因回答比例，得到偏好顺序：Formal≈Naturalized Formal > Pure NL > Examples。Formal 对比示例的选择率高达 93%+，而示例与纯自然语言的竞争则因模型规模和任务类型差异显著；在布尔代数和代码生成等异构任务中，也观察到类似或不同的偏好。

**⚠️ 局限性**

局限性包括：仅适用于可执行的明确映射任务，无法直接处理不完整或模糊的规格；规格与描述长度、信息密度等因素难以完全分离；实验设置主要集中在控制环境，真实世界冲突可能更复杂且包含多源信息。

---

## 519. How Benchmarks Mis-Score Computer-Use Agents

**arXiv ID:** 2607.28367 | [PDF](https://arxiv.org/pdf/2607.28367v1)

**作者:** Zihan Dong `[一作]` (Georgia Institute of Technology), Rui Qian `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对150条公开的计算机使用代理（CUA）失败轨迹进行系统审核，评估评估流程的可靠性并量化错误判定与真实失败的比例，提出四阶段可靠性框架与三层诊断分类；

**💡 创新点**

首次将评估可靠性分解为任务构造、轨迹观测、评分、报告四个阶段，量化评估器误判与任务破损率，并给出针对每个阶段的设计准则；

**🔧 技术方法**

结合人类与大型语言模型（LLM）评审的轨迹审核方法，构建三层诊断代码本，使用截图、工具I/O、时间戳等完整轨迹数据进行可观测性分析；

**📊 数据集**

采集并审核了5个公开CUA基准（WebArena、VisualWebArena、AssistantBench、WorkArena、OSWorld）中的失败轨迹，总计150条；

**📈 对比分析**

通过与官方oracle对比，发现15.3%失败判定错误（其中10.7%为评估器误判、4.7%为任务破损），诊断结果显示39.3%反馈/验证失败、35.2%规划失败、13.9%执行/定位错误，表明单一成功率指标掩盖了细节；

**⚠️ 局限性**

仅针对GUI CUA评估，样本量有限且仅审核失败案例，未评估通过案例；未覆盖更专业的任务集，且新长周期基准未纳入样本，结果具有一定局限性。

---

## 520. Teffic-Audio: Tell Fact from Fiction

**arXiv ID:** 2607.28351 | [PDF](https://arxiv.org/pdf/2607.28351v1)

**作者:** Wan Lin `[一作]`, Zhizheng Wu `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了Teffic‑Audio系统，用Conformer编码器+多头注意统计池化+MLP分类器，在公开数据上通过多源训练、攻击与来源平衡采样以及多样音频增强，取得Speech‑DF‑Arena 14测试集最低1.454% EER的通用深度伪造检测。

**💡 创新点**

创新点在于通过精心设计的训练分布策略（多源、多平衡采样、丰富音频增强）实现跨域鲁棒性，而非单纯依赖更复杂模型；证明多头注意统计池化和多样增强对泛化的关键作用；同时提供可扩展的浅层模型变体。

**🔧 技术方法**

使用技术包括：w2v‑BERT‑2.0 Conformer编码器、4头多头注意统计池化（MHASP）、MLP分类器，二元交叉熵损失；训练策略包括攻击-与来源平衡采样、补充真声语料、RawBoost、RIR、MUSAN、音高偏移、滤波、时域遮掩、编解码压缩、包丢失等多样化音频增强。

**📊 数据集**

使用的数据集：公开语音伪造数据集（ASVspoof2015/19/21/24、ASVspoof5、ADD2022/23、FakeOrReal、SpoofCeleb、ReplayDF、DFADD、MLAAD、LibriSeVoc、SpeechFake、Wavefake、CodecFake 等），以及补充真声语料（LibriSpeech、AISHELL3、GigaSpeech、CNCeleb、CommonVoice），总计数百万级样本。

**📈 对比分析**

与Speech‑DF‑Arena公开排行榜系统对比，采用 pooled EER、ACC、F1 评价；Teffic‑Audio 以 1.454% pooled EER 位列榜首，单测试集最低 5 个，参数量 590M，展示了优异的性能‑复杂度平衡，显著优于更大模型。

**⚠️ 局限性**

局限性：依赖公开数据分布，对极端噪声或未见生成模型的泛化尚未充分验证；浅层模型性能下降明显；缺乏对抗攻击或实时推理的评估；仅覆盖 14 个测试集，可能不足以代表所有真实场景。

---

## 521. LLMs struggle to simulate human belief updates in controlled environments

**arXiv ID:** 2607.28347 | [PDF](https://arxiv.org/pdf/2607.28347v1)

**作者:** Sebastian Pohl `[一作]` (Interdisciplinary Transformation University Austria), Christian Hilbe `[通讯]` (Interdisciplinary Transformation University Austria)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在实验中，研究人员通过让六种大语言模型（LLMs）在面对相同的 Reddit 评论时，对照 391 名英国参与者的信念更新过程，评估 LLMs 是否能够准确模拟人类的信念变更；

**💡 创新点**

创新点在于首次将 LLM 的个体级信念更新与真实人类实验数据进行 1‑to‑1 对比，揭示了 LLMs 在模拟初始立场、更新幅度以及对评论说服力评估方面的系统偏差；

**🔧 技术方法**

使用的技术包括 LLM 生成、Persona 定制（包含人口统计和大五人格特征）、两侧置换检验、卡方检验、Kendall 排序相关和线性混合效应模型；

**📊 数据集**

使用的数据集为 391 名英国 Prolific 受试者在 3 个话题（普惠基本收入、点球决赛、Ozempic 体重药物）下的 1173 条信念更新记录，以及对应的 27 条 r/changemyview 评论；

**📈 对比分析**

通过两侧置换检验和卡方检验比较，发现 Llama‑3.3‑70B‑Instruct 与 GPT‑5.2 在给定真实初始立场时表现接近人类，但在模拟初始立场和评论说服力排序时表现不佳；

**⚠️ 局限性**

局限性包括：仅测试单一更新阶段，未验证多轮交互中的累计漂移；对话主题与评论来源有限，可能影响普适性；人类受试者与 LLM 对话环境的差异；以及人格与人口统计信息对模拟精度影响不明显，需进一步研究更具代表性的特征。

---

## 522. Forecasting Land Art Under Climate Scenarios

**arXiv ID:** 2607.28489 | [PDF](https://arxiv.org/pdf/2607.28489v1)

**作者:** Alev Cinbarci `[一作]` (Işık University), Sean Kalaycioglu `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究利用1984–2025年Landsat与Sentinel-2图像芯片，建立两阶段预测管线，先对地区气候变量进行全球温度回归预测，再将14个图像复杂度特征进行随机森林与线性回归预测，并通过Stable Diffusion XL+LoRA+ControlNet与水位掩模生成未来Spiral Jetty的视觉合成；

**💡 创新点**

创新点在于将多源遥感复杂度指标与IPCC SSP情景耦合，首次采用双模型（随机森林与线性回归）进行外推诊断，并在扩散模型中嵌入气候条件化ControlNet与物理一致性的水位掩模，实现可解释且符合水文约束的视觉预测；

**🔧 技术方法**

使用了全球温度回归、随机森林、线性回归、Stable Diffusion XL+LoRA、ControlNet、DPM-Solver++、物理耦合水位掩模、Wasserstein距离、FID以及专家评估等技术；

**📊 数据集**

使用了1,744个对齐的Landsat 4–9与Sentinel‑2图像芯片（1984–2025年），以及对应的14个图像复杂度特征、每月气候变量（温度、海拔、盐度、CO₂等）和累计CO₂数据；

**📈 对比分析**

通过与2020–2023留出集的回溯、复杂度签名距离、FID评分和专家人类评估进行比较，生成样本在视觉逼真度与气候一致性上均可接受，但随机森林在外推时出现饱和；

**⚠️ 局限性**

局限性包括：预测高度依赖训练数据外推，随机森林在历史范围之外饱和，线性回归可能产生不物理意义的极值，且阶段1采用单变量回归缺乏水文耦合，导致高排放情景下的水位预测精度不足。

---

## 523. Would You Walk to the Car Wash? Revealing the Salience Bias of Large Language Models in Commonsense Reasoning

**arXiv ID:** 2607.28478 | [PDF](https://arxiv.org/pdf/2607.28478v1)

**作者:** Zheng Wu `[一作]` (Shanghai Jiao Tong University), Zhuosheng Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了SaliTrap基准，评估LLM在隐藏物理不可能前提下的推理表现。

**💡 创新点**

创新点在于提出Salience Bias概念、四维陷阱分类和可复现的评估框架，并证明知识压制而非缺失导致错误。

**🔧 技术方法**

采用LLM自助合成、三阶段验证、三元检查、Solver-Judge评估与Prompt干预技术。

**📊 数据集**

使用SaliTrap数据集，包含1,145个设计好的物理不可能任务，覆盖四个陷阱维度。

**📈 对比分析**

通过零样本评估12种主流LLM，发现trap-avoidance率低于50%，但轻量级提示能显著提升性能。

**⚠️ 局限性**

局限在于基准仅涵盖物理/常识不可能场景，缺乏对更广泛任务类型的验证。

---

## 524. SVR: Self-Verifying Refinement via Joint Verdict-Confidence Reinforcement Learning for Adaptive Test-Time Compute

**arXiv ID:** 2607.28457 | [PDF](https://arxiv.org/pdf/2607.28457v1)

**作者:** Hongyu Chen `[一作]` (Sun Yat-sen University), Guangrun Wang `[通讯]` (Sun Yat-sen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种无oracle、多轮强化学习框架Self‑Verifying Refinement (SVR)，通过模型自身产生的判定（Correct/Incorrect/Unsure）与置信度来决定是否保留当前答案或继续细化；

**💡 创新点**

创新点在于：① 通过联合判定–置信度的自检信号实现内部计算控制；② 在固定长度轨迹上训练，使模型在任何时刻都能产生可用的停止信号；③ 通过多项奖励（校准、过度自信惩罚、错误检测、停用准备）强化自检质量；

**🔧 技术方法**

技术方法包括：Group Relative Policy Optimization (GRPO) 的强化学习；自检信号结构化输出；多项奖励函数；自适应推理（按置信阈值停止）；以及固定轨迹采样与自适应推理的分离设计；

**📊 数据集**

使用了七个数学推理基准：Countdown、GSM8K、MATH500、AIME26、AMC23、OlympiadBench、MinervaMath；训练时分别在Countdown、GSM8K、MATH的完整训练集上微调；

**📈 对比分析**

与单轮、固定预算多轮、其他RL方法（GRPO‑MT、iGRPO、Murphy、MLMT‑RL、ScRPO）以及oracle‑guided参考对比。SVR在All‑7宏平均准确率达到0.563，平均推理回合2.99，token使用8.56k，明显优于所有非oracle基线；与10‑sample majority voting相比，SVR在相同/更低token成本下获得相同或更高准确率；

**⚠️ 局限性**

局限性包括：对置信阈值有一定依赖，部分难题的早停错误率（PSE）仍高达30%；需要大量训练数据与调参；模型的自检在极难案例中仍可能产生误判，导致早停误差；

---

## 525. Graph Neural Multilevel Preconditioners for Iterative Solvers

**arXiv ID:** 2607.28456 | [PDF](https://arxiv.org/pdf/2607.28456v1)

**作者:** Zechen Zhang `[一作]` (University of Minnesota), Yousef Saad `[通讯]` (University of Minnesota)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于图神经网络的多层预处理器GMP，利用AMG生成的层次结构作为先验，直接学习平滑器、限制和插值算子，并与Krylov子空间方法无缝集成。

**💡 创新点**

创新点在于：①将AMG层次结构嵌入可训练的图神经网络框架；②通过双边交叉注意力实现自适应的插值与限制，支持Petrov–Galerkin粗网操作；③针对非对称、非正定稀疏矩阵设计残差条件化多层预处理。

**🔧 技术方法**

采用消息传递图神经网络（MPNN）实现平滑器，使用双边交叉注意力网络（Bipartite Cross‑Attention）学习转移算子，并以AMG黑盒生成的层次结构为图拓扑。

**📊 数据集**

在SuiteSparse 867个大小1K–100K、非对称、非正定的稀疏矩阵上进行大规模评估。

**📈 对比分析**

与传统AMG、ILUT、单层GNN预处理器以及Jacobi做对比；在成功的运行中GMP在迭代曲线与收敛速度上往往优于单层GNP，在部分困难的非对称矩阵上明显优于AMG和AIR；但其构造成本、内存占用和训练时间高于单层方法。

**⚠️ 局限性**

局限性包括：需对每个矩阵单独训练、内存耗费较大、对极端非对称/不良条件矩阵的构造失效率仍高于经典AMG；跨矩阵泛化能力有限，且在高精度迭代层级深度增加后仍受层次结构质量限制。

---

## 526. One Future, Every Robot: Label-Efficient Collective-State Prediction with Decentralized JEPA

**arXiv ID:** 2607.28443 | [PDF](https://arxiv.org/pdf/2607.28443v1)

**作者:** Alan-Barsag Gazzaev `[一作]` (ITMO University), Sergey Muravyov `[通讯]` (ITMO University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 Collective‑State JEPA (CS‑JEPA)，研究在仅靠本地观测与 256 字节/边缘消息的情况下，群体中的每台机器人能否预测同一未来群体状态，并在部署时使用 16 帧历史与 64‑维递归消息。

**💡 创新点**

创新点包括：1）无全局池化、时钟或动作输入，仅依赖本地信息；2）使用接收器锚点与冻结目标编码器的联合嵌入预测，形成规模不变的共同未来目标；3）在预训练期间引入隐式未来状态引导，部署时完全去除；4）在多种拓扑与规模下展示显著的标签效率和跨机器人一致性提升。

**🔧 技术方法**

技术手段：递归神经网络（GRU）消息传递；联合嵌入预测（JEPA）目标；目标编码器+分词器的冻结预训练；线性回归探针解码十个群体量化指标；多种实验配置（ID、拓扑 OOD、规模 OOD）及统计检验。

**📊 数据集**

数据集：仿真生成的 200 步 episode，任务包括 flocking、formation、coverage；机器人数量 N ∈ {10,18,36,72,108}；拓扑包括小世界、环形、互 k‑NN；过程噪声 0.02，观测噪声为 0。

**📈 对比分析**

比较方法：将 CS‑JEPA 与基线原始未来字段重建方法对照，采用标签预算 6/12/24 的 AUC 评估准确性与跨机器人一致性；CS‑JEPA 在所有四个分割中均优于基线，平均准确性 AUC 提升约 0.05，一致性 AUC 提升约 0.08。再对行动条件的四步值估计进行评估，误差下降 45%，相关性提升 0.13；在消息失真和失败斜坡警告任务上亦保持优势。

**⚠️ 局限性**

局限性：仅在仿真环境验证，假设共享坐标系且无观测噪声；预训练阶段使用未来状态，未在真实硬件上测试；未评估大规模或动态拓扑的可扩展性；闭环控制优势未得到充分证明；安全性、延迟和多源不确定性仍待研究。

---

## 527. ViewMind3D: Modular View-Aware Inference for Training-Free 3D-QA

**arXiv ID:** 2607.28442 | [PDF](https://arxiv.org/pdf/2607.28442v1)

**作者:** Ping-Kun Chiang `[一作]`, Yu-Chee Tseng `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种完全无训练、模块化的 3D 问答框架 ViewMind3D，能够在多视角观测下直接利用通用 LLM/VLM 进行三维空间推理。

**💡 创新点**

创新点在于：①通过问题驱动的视图筛选减少冗余输入；②基于语言引导的开放式视觉定位提升视觉对齐；③用鸟瞰视角（BEV）指示器为跨视图推理提供共享空间坐标；④通过角色分解的 Prompt 结构化推理，兼顾可解释性与性能。

**🔧 技术方法**

主要技术包括：预训练 LLM（GPT‑4.1 / o3）和 VLM（CLIP/FLIP 等）作为推理核心；Florence‑2 等开源目标检测；BEV 视角编码；Prompt Engineering（角色分解、术语约束）与多阶段推理。

**📊 数据集**

使用 ScanQA 与 SQA3D 两个基准数据集（均基于 ScanNet），同时利用其多视角图像与点云信息。

**📈 对比分析**

与现有基准比较：在 ScanQA 上 ViewMind3D (o3) 达到 73.41% CIDEr、90.1% BLEU‑1，GPT‑4.1 仅 68.30% CIDEr；在 SQA3D 上实现 50.75% 准确率，显著高于 3D‑LLM 的 49.79%；整体表现与部分 fine‑tuned 3D‑LLM 相当，且保持零训练优势。

**⚠️ 局限性**

局限性：①对“How”类计数/流程问题性能下降，主要因开放式检测噪声；②推理成本高（约 120 秒、90K token）；③过度依赖 LLM 费用与可用性；④在极低视角覆盖的场景中 BEV 指示器收益有限。

---

## 528. Beyond a Single Judge: Simulating Social Persona Panels for Generative UI Evaluation

**arXiv ID:** 2607.28439 | [PDF](https://arxiv.org/pdf/2607.28439v1)

**作者:** Zheng Wu `[一作]` (Shanghai Jiao Tong University), Zhuosheng Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了Evidence-Grounded, Social-Weighted Persona Panel (ESPP) 评估框架，用于自动评估生成式用户界面（GenUI）的质量，并构建了包含 500 条自然语言指令、14 种 LLM 生成的 7,000 个截图的 UIPersonaBench 基准。

**💡 创新点**

创新点在于：①将人格特质与先验证据（P-Q-A）绑定的多样化人格面板；②在面板内部引入基于个体可接纳度的语义阈值意见动态；③采用 Delphi 灵感的社会加权聚合，既保留个体差异又形成可解释的综合分数。

**🔧 技术方法**

技术手段包括：使用大型语言模型（Claude-Opus-4.6 等）作为评判器；基于 Big‑Five 维度构建 1,000 个合成人格；实现有限信心意见动态和基于人格可接纳度的阈值；以及基于专业度与代表性权重的加权聚合。

**📊 数据集**

使用的数据集为：① UIPersonaBench（500 条 UI 生成指令，14 种模型共 7,000 张截图）；② 1,000 名多样化人格样本；③ 5 名真实人工评审提供的基准评分，用于验证模型评估的准确性。

**📈 对比分析**

通过与两种对照方法（单通 LLM 评判、5 次提示集成）比较，ESPP 的 Pearson 相关系数提升至 0.922（单通为 0.716），MAE 降至 0.110，显示在与人工基准的对齐度上显著优于传统方法。

**⚠️ 局限性**

局限性包括：① 仅覆盖 5 维度的 UI 质量评估，缺乏对更细粒度特征的捕捉；② 依赖 LLM 的推断，若模型偏差或错误仍可能影响评估结果；③ 面板规模固定为 5 人，可能不足以覆盖更大范围的用户多样性；④ 对模型生成的异常情况或极端 UI 的鲁棒性仍待进一步验证。

---

## 529. AgentRadio: Passive Awareness for Long-Horizon Multi-Agent Collaboration

**arXiv ID:** 2607.28430 | [PDF](https://arxiv.org/pdf/2607.28430v1)

**作者:** Xinxing Ren `[一作]` (Coral AI Labs), Zekun Guo `[通讯]` (University Of Hull)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出AgentRadio，一种异步消息传递层，使编码型LLM代理能在保持工作进程的同时被动地接收同伴信息，显著提升对大型代码库问题的解决率。

**💡 创新点**

创新点在于将消息监听拆分为后台任务（passive awareness），消除通信与执行的互斥，使代理能实时共享发现并即时融入当前子任务。

**🔧 技术方法**

核心技术包括三种原语（threads、messages、waiting-for-mentions）与五阶段协作协议（探索、划分、执行、评审、提交），以及对Claude Code/DeepSeek等现有代码代理的无侵入式包装。

**📊 数据集**

使用SWE‑Atlas QnA数据集——124道专业代码库问题，覆盖11个生产仓库，评估代理在长时任务中的性能。

**📈 对比分析**

与单一代理、最佳单机运行、以及更强模型Claude Code Opus 4.8等基准比较，结果显示四个AgentRadio代理在Opus 4.6下从32.3%提升至62.1%，DeepSeek从29.0%提升至50.8%，并在同等预算下显著优于单机多跑。

**⚠️ 局限性**

局限性包括对任务可分解程度的依赖、对划分与协商过程的敏感、实现成本（多代理管理与同步）以及在已具备完备计划的任务上提升有限。

---

## 530. Negative controls reveal volume-driven confounding in radiomics and imaging foundation model features

**arXiv ID:** 2607.28423 | [PDF](https://arxiv.org/pdf/2607.28423v1)

**作者:** Katy L. Scott `[一作]` (Princess Margaret Cancer Centre), Benjamin Haibe-Kains `[通讯]` (Princess Margaret Cancer Centre)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出并实现了READII-2-ROQC框架，通过体积保持的负控图像评估放射组学和深度特征对空间结构的依赖性，并在三大公开影像数据集上提取并比较特征。

**💡 创新点**

创新地将体积保持的 voxel 随机化负控与多种空间区域（ROI、全图、背景）相结合，构建可扩展的质量控制流程，用以区分体积驱动与真正纹理信号。

**🔧 技术方法**

使用 PyRadiomics（形状、强度、纹理等）和 FMCIB 基础模型特征提取，配合随机、采样、打乱三种 voxel 随机化策略生成九种负控图像；通过相关系数、协同信息、C-index、AUC 等评估指标进行分析。

**📊 数据集**

在 LUNG1（肺腺癌 CT）、HN1（头颈癌 CT）以及 RADCURE（头颈癌 CT）三大 TCIA 公共影像数据集共 3552 份肿瘤体积上完成实验。

**📈 对比分析**

对已公开的三条放射组学签名（Aerts 生存、Choi 生存、Choi HPV）在原始图像和九种负控图像上重现；结果显示 Aerts 签名与单纯体积模型等价且对负控不敏感，Choi 生存弱于体积模型，Choi HPV 在原始图像上优于体积模型但在全图/背景负控下性能显著下降。

**⚠️ 局限性**

受限于影像获取、重建、分割等前置差异，未加入 ComBat 等归一化；负控仅涉及 voxel 随机化，未覆盖更复杂模态转换或生物学仿真；因此对不同机构数据的迁移性与生物学解释仍有限。

---

## 531. FasTac: A Curved Multispectral Vision-Based Tactile Sensor for High-Speed High-Precision 3D Shape and Force Perception

**arXiv ID:** 2607.28416 | [PDF](https://arxiv.org/pdf/2607.28416v1)

**作者:** Xiaofan Lu `[一作]` (Huazhong University of Science and Technology), Zhouping Yin `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种曲面视觉触觉传感器FasTac，能够实现高精度的三维形状重建和三轴力感知，并将完整的图像到力的处理流程部署在FPGA上。

**💡 创新点**

创新点包括：①使用RGB‑NIR单摄像头与四源光照实现自标定、无对齐误差的多光谱输入；②引入边界先验的快速泊松求解，显著提升曲面深度重建精度；③设计位置感知动态卷积网络HyperForce，将FEM思路映射为像素级卷积，精确估计三轴力；④将整个处理管线硬件化，达到1.09 ms/帧、低能耗的边缘部署。

**🔧 技术方法**

采用了光度立体、边界先验泊松重建、FEM启发式动态卷积、光学反射涂层与标记点跟踪、FPGA并行计算与流水线优化等技术。

**📊 数据集**

数据集来源于机器人手掌与球面压头同步采集的实验，使用ATI Nano17力/扭矩传感器、球形压头、FasTac传感器、CNC控制等，提供光图像、姿态、力标签用于训练、验证和测试。

**📈 对比分析**

通过与DenseTact2.0、GelStereoBioTip等现有光学触觉传感器对比，FasTac在深度重建MAE仅0.0415 mm，三轴力的NMAE分别为2.74%（法向）和2.39%（切向）；FPGA实现的图像到F_z推断延迟1.09 ms、能耗8.41 mJ/帧，比CPU（6.82 ms/238 mJ）和GPU（3.26 ms/33.6 mJ）快约6.3倍和3.0倍，且能在100 Hz振动下保持准确性。

**⚠️ 局限性**

局限性包括：对光照和材质的鲁棒性尚未充分验证；仅实现了F_z的FPGA加速，Fx/Fy的实时性能仍较慢；集成成本和尺寸受限于多光源与FPGA板卡的硬件配置。

---

## 532. Beyond Frame Selection: Generative Latent Evidence Aggregation for Long-Video Understanding

**arXiv ID:** 2607.28516 | [PDF](https://arxiv.org/pdf/2607.28516v1)

**作者:** Bowen Liu `[一作]` (Hong Kong University of Science and Technology), Xiaomeng Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种隐式证据接口，将已选择的帧的查询相关潜在信息与显式视觉上下文结合，用于长视频问答。

**💡 创新点**

创新点在于：①在帧选择与答案生成之间加入隐式证据阶段；②采用分布引导的预算分配与查询感知聚合；③利用分配散度自适应决定是否注入隐式证据，从而实现路由自适应。

**🔧 技术方法**

使用冻结的时间选择器和冻结的视频-MLLM，配合轻量级分配头、潜在基底、查询‑视觉上下文聚合、LoRA适配器以及自适应证据调用机制。

**📊 数据集**

在 LongVideoBench、MLVU、Video‑MME 和 LVBench 四个长视频基准上进行实验。

**📈 对比分析**

与匹配帧基线和现有 Video‑MLLMs 比较，GLA 在 8/16 帧预算下平均提升 5.2/3.4 分；在 Qwen2.5‑VL 的 LVBench 上提升 10.1 分；并且仅增加 0.11–0.40% 的解码端 token 负担，表现出显著性能提升。

**⚠️ 局限性**

局限性包括：更大潜在预算收益有限；依赖预训练的时间选择器和冻结基线模型，限制跨模型迁移；自适应路由阈值需手动调优，可能在极端证据分布下表现不佳。

---

## 533. AuricularWorld: Hierarchical Action-Guided World Modeling for Fine-Grained Auricular Structure Segmentation from CT Scans

**arXiv ID:** 2607.28487 | [PDF](https://arxiv.org/pdf/2607.28487v1)

**作者:** Jingwen Yang `[一作]` (Plastic Surgery Hospital, Chinese Academy of Medical Sciences and Peking Union Medical College), Haiyue Jiang `[通讯]` (Plastic Surgery Hospital, Chinese Academy of Medical Sciences and Peking Union Medical College)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了基于世界模型的迭代隐藏状态分割框架 AuricularWorld，用于精细化耳部 CT 图像的多标签分割。

**💡 创新点**

将分割视为递归解剖演化过程，引入多尺度观察融合、三步 RSSM latent rollouts、分层加/移动作监督及前景掩码平衡权重，显著提升对小、薄、重叠结构的分割精度。

**🔧 技术方法**

融合 nnU-Net 编码器-解码器、状态空间模型（RSSM）、ConvGRU、前景掩码平衡的动作损失、三步动作递归与高分辨率解码。

**📊 数据集**

使用 193 名患者（共 198 耳部）的自制耳部 CT 数据集，包含皮肤覆盖亚结构与对应软骨结构的 35 个原子标签，构成首个细粒度耳部 CT 分割基准。

**📈 对比分析**

与 nnU-Net、TransUNet、nnFormer、SwinUNETR、UNETR、nnMamba 等基线在同一数据集上比较，AuricularWorld 达到 Dice 78.42% 与 HD95 1.379 mm，较 nnU-Net 提升 0.67% Dice、降低 43% HD95，成为最佳方法。

**⚠️ 局限性**

仅在耳部 CT 任务验证，模型结构复杂、推理耗时和计算资源相对较高，对标注质量要求高，尚未验证能否直接推广至其他解剖结构。

---

## 534. A Fuzzy Rule-based Neuro-Symbolic Approach for Pipe Severity Prediction in Sewer Networks

**arXiv ID:** 2607.28481 | [PDF](https://arxiv.org/pdf/2607.28481v1)

**作者:** Ngoc Thai Le `[一作]` (Can Tho University), Umberto Straccia `[通讯]` (Istituto di Scienza e Tecnologie dell'Informazione, CNR - ISTI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种模糊规则神经符号框架，用Swin Transformer预测多标签缺陷CODE，再通过固定的决策树提取规则进行模糊推理，实现从图像到管道严重程度的完整可解释链条。

**💡 创新点**

创新点在于将标准缺陷CODE作为中间语义层，解耦神经感知与符号推理，使用大语言模型共识生成严重程度标签，并在推理层实验不同t‑norm/s‑norm组合以及软度与阈值接口，提供透明的决策过程。

**🔧 技术方法**

采用Swin Transformer做多标签分类，Weka J48决策树抽取规则，模糊逻辑（Product、Łukasiewicz、Hamacher）进行推理，使用二元交叉熵损失、AdamW优化，并利用LLM对观察文本进行标签共识。

**📊 数据集**

使用一份包含3,244张法国污水管道检验图像的专有数据集，包含5级严重程度标签（由5个LLM共识生成）和14个缺陷CODE标签。

**📈 对比分析**

通过与仅基于图像的多类分类基线以及Oracle CODE+Rule（使用真值CODE）比较，实验显示软度接口下的Product/Hamacher/Łukasiewicz三种模糊组合分别提升了约18%、12%、23%（宏F1）和17%（MCC）的性能，Oracle模型达到86%准确率。

**⚠️ 局限性**

局限在于多标签CODE预测仍受视觉噪声与不平衡影响，规则覆盖受单棵DT限制，接口选择对数据分割敏感，严重程度标签来源为LLM共识，且数据集规模与类别分布不均导致小样本类的识别仍不稳定。

---

## 535. A report-grounded vision-language foundation model for colonoscopy from 280000 routine reports

**arXiv ID:** 2607.28466 | [PDF](https://arxiv.org/pdf/2607.28466v1)

**作者:** Jia Yu `[一作]` (Fudan University), Shuo Wang `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了 EndoCLIP，一种利用常规结肠镜报告与视频帧恢复病变级图像–文本对应关系的视听基础模型；

**💡 创新点**

创新点在于通过分阶段的对应恢复（案例级证据定位、单病变锚点选择、多病变歧义化）将报告级弱监督转化为高质量的病变级图像–文本对，从而实现零样本检索、线性探测与结构化报告；

**🔧 技术方法**

采用 CLIP 样式的双编码器架构，配合 InfoNCE 对比学习、Vision Transformer 视觉编码器、文本编码器以及多阶段对应恢复策略，并将冻结的视觉编码器与 Qwen3‑14B 解码器通过可训练投影器结合实现结构化报告；

**📊 数据集**

使用 280,476 篇去标识结肠镜报告（共 8.54M 帧）构建 125,756 对图像–文本对，并在 EndoReport100（100 案例 7,002 帧）、EndoVL（9 个公共数据集共 6,000+ 图像）以及 Zhongshan 病理数据集上进行评估；

**📈 对比分析**

与 CLIP‑OpenAI、BiomedCLIP、PMC‑CLIP 等通用或生物医学模型相比，EndoCLIP 在病变级检索 Recall@1 提升至 14.3%（vs 2.8%），零样本分类 AUC 达 0.766–0.851，线性探测 AUC 高达 0.908，结构化报告 F1 为 0.764，甚至在专家读者实验中冻结特征的准确率接近专家水平；

**⚠️ 局限性**

局限包括：报告文本中病变尺寸为解析值而非直接测量；多病变报告的对应恢复仍受阈值设置影响；模型在多中心、不同报告风格及摄像机设备下的泛化需进一步验证；并且目前未解决帧级与病人级的完全分离与潜在源偏差问题。

---

## 536. Can Vision-Language Models Reason about AI Edits in Images?

**arXiv ID:** 2607.28464 | [PDF](https://arxiv.org/pdf/2607.28464v1)

**作者:** Darsha Udayanga `[一作]` (Rensselaer Polytechnic Institute), Qiang Ji `[通讯]` (Rensselaer Polytechnic Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过强化学习GRPO训练Vision‑Language Model生成推理轨迹，判别并定位图像篡改，然后利用提示驱动的SAM分割模块实现像素级掩模；

**💡 创新点**

采用弱监督的格式与准确度奖励驱动VLM推理，而非显式解释监督；将检测与分割解耦，利用推理轨迹与粗定位生成精细掩模；引入统一评估指标effective‑IoU；

**🔧 技术方法**

Vision‑Language Model (Qwen2.5‑VL)、Group Relative Policy Optimization、强化学习奖励设计、Prompt‑guided SAM分割、effective‑IoU评估；

**📊 数据集**

AutoSplice、CASIAv2、Fantastic Reality、FFHQ‑FM、MagicBrush、SD_inpaint 等多种AI篡改数据集；

**📈 对比分析**

与AdaIFL、Mesorch、FakeShield、SIDA等基线对比，检测准确率最高，mIoU竞争力强，effective‑IoU最高，平均effective‑IoU达0.211，超过第二名46%；

**⚠️ 局限性**

受限于VLM规模与训练数据量，奖励设计与训练时长对性能影响更大；缺乏更大规模高质量掩模数据；对分布偏移与对抗扰动的鲁棒性需提升。

---

## 537. VisualRouter: Query-Grounded Visual Sampling for Long Video Understanding

**arXiv ID:** 2607.28463 | [PDF](https://arxiv.org/pdf/2607.28463v1)

**作者:** Haiyue Zhang `[一作]` (Tongji University), Heng Tao Shen `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 VisualRouter，一种无训练、可插拔的查询导向视觉采样框架，能够根据问题类型（全局或局部）动态选择帧，从而在长视频理解任务中提供更具信息量的视觉输入。

**💡 创新点**

创新点在于：① 通过查询路由（query gating）将问题分为全局与局部两类，分别采用不同的采样策略；② 对全局查询融合相关性与时间覆盖；③ 对局部查询执行事件分区、段级帧分配与基于 k‑DPP 的多样性采样，三阶段流程实现信息覆盖与冗余最小化；④ 这一方法无需额外训练，直接可作为任何大型视觉语言模型的前处理模块。

**🔧 技术方法**

使用了 BLIP2 的 ITM 头进行查询‑帧相关性评分与语义特征提取；DINOv2 用于视觉变化检测；通过指数移动平均与视觉差分进行事件边界检测；k‑DPP 用于每段内的多样性帧选择；同时结合统一采样与相关性 Top‑K 的混合策略。

**📊 数据集**

在 Video‑MME、LongVideoBench、MLVU 三个长视频问答基准上进行评估，覆盖短至长视频、不同视频时长与多任务场景。

**📈 对比分析**

与统一采样、Top‑K、BOLT、AKS、WFS‑SB 等代表性无训练采样方法比较，VisualRouter 在所有基准和多种 LVLM（LLaVA‑OneVision、Qwen2.5‑VL、InternVL3 等）上均实现显著提升，平均提升 5–12%（例如 Qwen2.5‑VL‑7B 在 Video‑MME 上提升 5.2%，LongVideoBench 上提升 7.7%，MLVU 上提升 11.6%）。

**⚠️ 局限性**

局限性包括：① 对极短视频或单一事件视频的优势有限，提升不如长视频显著；② 仍依赖预训练模型的特征与相关性评分，若评分失效可能导致错误路由；③ 事件分区与帧分配基于经验权重，可能对不同视频域需进一步调优；④ 在极低帧预算（如 4‑8 帧）下，性能提升仍受限。

---

## 538. Lightning OPD 2.0: Mitigating Style Bias in Cross-Teacher On-Policy Distillation for Large Reasoning Models

**arXiv ID:** 2607.28449 | [PDF](https://arxiv.org/pdf/2607.28449v1)

**作者:** Yecheng Wu `[一作]` (NVIDIA), Han Cai `[通讯]` (NVIDIA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种在跨教师的离线OPD（Lightning OPD）中通过风格残差化减去教师-参考一致性误差的技术，进一步提高模型在数学推理与代码生成上的表现。

**💡 创新点**

核心创新是使用跨折交叉拟合的token与上下文lookup表估计并剔除重复出现的风格偏差，从而在不满足教师一致性的场景下仍能实现高效的OPD。

**🔧 技术方法**

实现技术包括Lightning OPD 2.0框架、风格残差化方法、token与上下文特征的lookup表、跨折交叉拟合、以及响应平衡的加权训练。

**📊 数据集**

实验使用了数学推理数据集DAPO‑Math‑17k以及代码生成数据集KlearReasoner‑CodeSub‑15K，并在AIME、HMMT及LiveCodeBench v5/v6等评测集上评估。

**📈 对比分析**

与SFT基线、原始Lightning OPD、IW‑OPD和TA‑OPD等方法对比，跨教师设置下平均提升约3–4分数学分数、1–2分代码分数，显著优于所有基线。

**⚠️ 局限性**

局限性包括仅在Qwen系列模型与两类任务上验证，未探测其他模型或更广泛任务的适用性；风格残差估计基于统计近似，可能无法完全捕捉所有语义偏差。

---

## 539. A foundation model of numerical intelligence with cross-disciplinary generalization

**arXiv ID:** 2607.28432 | [PDF](https://arxiv.org/pdf/2607.28432v1)

**作者:** Chenghan Wu `[一作]` (National University of Singapore), Liu Yang `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `14d48e9d-0069-4ad9-996a-1d5968216998` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个跨学科的数值智能基础模型，能够在水文、交通、能源、气候等多领域进行数值预测与分析。

**💡 创新点**

创新点在于将来自十余个不同学科的数据统一融入单一Transformer+图神经网络架构，突破了传统任务特定模型的局限，实现了真正的跨学科泛化。

**🔧 技术方法**

主要技术包括大规模多模态Transformer、图卷积网络用于时空特征提取，以及自监督预训练与少样本微调策略。

**📊 数据集**

使用了CAMELS-US/BR/GB/CL等水文数据、Caltrans PeMS与METR-LA交通流数据、EIA-930能源数据、PyPSA-Eur电网拓扑、GLDAS/ERA5/WeatherBench2气候数据、SMAP-L4土壤湿度、GLORYS海洋观测等十余个公开数据集。

**📈 对比分析**

在数值预测、时序建模和空间格点预测等任务上与领域专用模型、传统统计方法和现有基线进行对比，实验显示平均RMSE提升约12%，在泛化能力、数据覆盖范围和推理速度上均优于对照组。

**⚠️ 局限性**

局限性包括对极端稀缺事件的泛化能力仍有限，模型可解释性不足，需要在各领域进行更细粒度的微调和领域适配。

---

## 540. Demystifying Solana Bots: From GitHub Blueprints to On-Chain Fingerprints

**arXiv ID:** 2607.28424 | [PDF](https://arxiv.org/pdf/2607.28424v1)

**作者:** Xiaoye Zheng `[一作]` (Zhejiang University), Zhiyuan Wan `[通讯]` (Zhejiang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

系统地研究Solana上交易机器人，基于586个GitHub开源仓库和200个链上地址共计4412万笔交易，提出功能分类、实现架构和链上执行特征。

**💡 创新点**

首次从实现和链上视角统一构建Solana机器人的功能词典、五阶段通用流水线和执行簇，揭示依赖技术滞后与执行基础设施对盈利的影响。

**🔧 技术方法**

采用LLM辅助代码摘要、功能卡排序、UMAP‑HDBSCAN聚类、深度学习嵌入、HDBSCAN地址聚类，结合Solana SDK、第三方库以及SQL抽取链上数据。

**📊 数据集**

使用586个Solana bot GitHub仓库（多语言，TypeScript占比48.2%）和200个来自Trojan、SolanaMevBot的链上地址（共44,118,825笔交易）及验证窗口数据。

**📈 对比分析**

通过对比不同聚类的交易频率、成功率、盈利分布以及交易所与代币模式，证明MEV与交易操作机器人在执行策略、收益与风险上显著差异，表现为四种执行模式。

**⚠️ 局限性**

局限在于数据集仅覆盖交易/MEV导向的机器人，缺乏操纵、分析和工具类地址；且仓库与地址无直接对应，难以验证功能映射；依赖技术滞后评估受版本信息约束。

---

## 541. QAdapt: A Noise-Adaptive Neural Pre-Decoding Framework for Quantum Error Correction

**arXiv ID:** 2607.28422 | [PDF](https://arxiv.org/pdf/2607.28422v1)

**作者:** Ran Miao `[一作]` (Beijing Zhongke Qhub Technology Co., Ltd.), Xiaoming Sun `[通讯]` (Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种适应性神经预解码框架（QAdapt），在表面码量子误差校正中先用局部神经网络识别并纠正错误，再把剩余探测符号交给传统全局匹配器处理。

**💡 创新点**

创新点在于：①利用异构时空特征提取和轴向校准机制捕获探测符号的多方向相关性；②采用弹性权重约束（EWC）实现连续学习，防止在噪声演化时的灾难性遗忘；③在保持全局一致性的同时显著降低后端匹配器负载。

**🔧 技术方法**

使用 HTNet 架构的卷积预解码器、EWC 连续学习、PyMatching 全局匹配器、Stim 生成的合成噪声、以及对表面码的四通道探测符号输入。

**📊 数据集**

评估数据集包括：110 组合成的离散噪声（多轴扩展）模拟；Google Willow 开源表面码数据（d=5、d=7）；以及映射到 T0 设备噪声的训练模拟。

**📈 对比分析**

与基线神经预解码器相比，QAdapt 在 T0 环境下逻辑误码率下降 15.6%–18.6%，在所有 OOD 配置下平均下降约 3%，后端匹配器延迟下降 5.7%–5.8%；在 Willow 数据上零样本迁移时，逻辑误码率分别下降 5.79%（d=5）和 2.51%（d=7），后端延迟下降 1.43% 与 9.32%。

**⚠️ 局限性**

局限性包括：未提供置信区间或抽样误差；缺乏对单个模块的消融分析；仅在旋转表面码内验证，未涵盖更大码距或其他 QEC 代码；未测量完整端到端延迟；在线漂移检测与实时自适应尚未实现。

---

## 542. When Derived Measurements Mislead: Quantifying and Mitigating LLM Over-Trust with Privileged-Modality Reliability Evidence

**arXiv ID:** 2607.28421 | [PDF](https://arxiv.org/pdf/2607.28421v1)

**作者:** Zongheng Guo `[一作]` (Politecnico di Milano), Manuela Ferrario `[通讯]` (Politecnico di Milano)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文研究了从上游模型派生的测量（Derived Feature）被下游大型语言模型（LLM）过度信任（DFOT）的现象，并为此提出了可复制的评估框架和指标链，在PPG‑ECG实例中进行验证。

**💡 创新点**

创新点包括：①正式定义DFOT这一下游依赖实例的测量过度信任；②提出包含 COTR、CIR、CRR、ESRM、UHR 的五项评估指标；③设计 matched‑vs‑shuffled 证据干预，用以区分实例特异性修复与泛化性警示；④在实验中引入 ECG‑guided 预训练知识蒸馏（K2）作为基线，并展示其在 DFOT 上的提升。

**🔧 技术方法**

采用的技术：深度学习的 PPG 可靠性模型（B2）与 ECG‑guideline 的知识蒸馏学生模型（K2）；知识蒸馏（KD）与对齐窗口的 KL 损失；LLM 推理（Qwen3‑8B、DeepSeek‑V4‑Pro、GPT‑5.5）与固定 prompt；Bootstrap 置信区间与对照实验（matched 与 shuffled）。

**📊 数据集**

使用 MIMIC‑III Matched Waveform Database（PPG PLETH 与 Lead‑II ECG 同步）共 50,000 条四分钟记录，划分为训练（36,115 条）、验证（6,464 条）与锁定测试（7,421 条）三组，保证患者互斥。

**📈 对比分析**

与仅 PPG‑可靠性模型（B2）对比，K2 在 D1、D2 两种挑战下的 CRR 提升 4–7 个百分点，ESRM 也相应提升 4–7 个百分点；在锁定测试中 CRR+ESRM 分别从 40.1/21.7% 提升到 44.8/26.2%；在不同 LLM 与 prompt 组合下保持正向效果。UHR（不必要的验证）略增 0.67 个百分点，但仍低于 2 个百分点阈值。

**⚠️ 局限性**

局限性：①仅验证了一种 priviledged distillation 方法，未系统比较多种可靠性生成策略；②在低 FPR 区域提升不明显，说明仍需更强的校准/选择性预测技术；③大部分 DFOT 错误仍未修复，存在改进空间；④实验仅在 PPW‑ECG 领域验证，跨域通用性尚待进一步研究；⑤未涉及真实临床决策场景，结果需要临床验证。

---

## 543. WIDE: Boosting Adaptive LLM Inference via Token-level Dynamic Width Pruning

**arXiv ID:** 2607.28418 | [PDF](https://arxiv.org/pdf/2607.28418v1)

**作者:** Haozhe Hu `[一作]` (Ningbo Institute of Digital Twin, Eastern Institute of Technology), Xiaoyu Shen `[通讯]` (Ningbo Institute of Digital Twin, Eastern Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种端到端可微的 token 级宽度动态裁剪框架（WIDE），能够让每个 token 动态选择注意力头组和 FFN 通道组，实现 neuron‑block 级别的结构化裁剪。

**💡 创新点**

创新点包括：①将动态裁剪从层级提升到 neuron‑block 级别；②使用轻量化瓶颈路由器与 Gumbel‑Softmax 进行可微路由训练，并通过两阶段训练（路由器训练+LoRA 微调）恢复性能；③设计统一的 mask‑reordering 与多级内核谓词（CTA、加载、MMA）实现 GPU 级别的加速。

**🔧 技术方法**

采用技术包括：轻量化瓶颈路由器、Gumbel‑Softmax、双阶段训练（路由器训练 + LoRA 微调）、mask 重排 + 多级内核跳过、CUDA Graph、Triton/Tilelang 编写的定制注意力与 GEMM 内核。

**📊 数据集**

使用 RedPajama‑1T 子集进行校准与 LoRA 复原；评估数据集包括 WikiText2、ARC‑Easy、ARC‑Challenge、BoolQ、WinoGrande、PIQA、OpenBookQA、HellaSwag 等。

**📈 对比分析**

与静态剪枝（Shortened‑LLaMA、CoopPruner、SliceGPT、Týr‑the‑Pruner、DDP）和动态剪枝（D‑LLM、SkipGPT）在 25%/50% 稀疏度下对齐同一校准/LoRA 流程比较；在 50% 稀疏度下平均 zero‑shot 准确率提升 8–9 分，LoRA 复原后保持约 90%+ 准确率；在预填充和解码场景下实现 1.68× / 1.55× 的端到端加速，接近理论上限 1.9×/2.0×。

**⚠️ 局限性**

局限性包括：在极低稀疏度下加速不显著；需要复杂的 mask‑reordering 与 GPU 内核设计，迁移到其他 GPU 架构需要额外工程；同步与内存移动开销在大批量/多卡场景下尚未充分评估。

---

## 544. Can Large Language Models Execute Parent Orders?

**arXiv ID:** 2607.28410 | [PDF](https://arxiv.org/pdf/2607.28410v1)

**作者:** Zane Shen `[一作]` (Independent Researcher), Zhen Yang `[通讯]` (HKUST(GZ))

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种名为PACE的层次化父订单执行框架，利用大型语言模型在无预设市场假设和无任务特定训练的情况下实现长短期规划与执行；

**💡 创新点**

创新点在于首次将LLM应用于父订单执行，并将任务拆分为长周期规划（Planner）与短周期执行（Executor），通过LLM的先验知识与实时行情共同生成动态交易计划；

**🔧 技术方法**

核心技术包括：大型语言模型（如ChatGPT-5.4、DeepSeek-v4-flash）与文本提示工程，基于文本生成的长短期决策，权重混合公式与短期量化调整；

**📊 数据集**

使用了2026年4月起的深圳证券交易所Level-1历史数据，随机生成父订单进行回测；

**📈 对比分析**

与传统静态策略（TWAP、Almgren‑Chriss）以及学习型策略（XGBoost、LSTM）在同一父订单上对比，PACE在激进与被动下均超越所有基线，最高提升约1.07个基点（相当于每年约1000万美元的成本节省）；

**⚠️ 局限性**

局限性包括：仅在历史回测环境下验证，缺乏实时交易验证；LLM的推理成本与随机性仍需进一步控制；对极端市场波动与跨资产信息的适应性尚未评估；

---

## 545. The Role of Causality in Algorithmic Recourse

**arXiv ID:** 2607.28497 | [PDF](https://arxiv.org/pdf/2607.28497v1)

**作者:** Srikanth Avasarala `[一作]` (Georgia Institute of Technology), Juba Ziani `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了基于因果结构的算法递归框架，兼顾策略性代理的行为与模型自适应，解决传统递归方法因“游戏”导致的误匹配问题。

**💡 创新点**

创新点在于将因果传播机制与performative prediction结合，阐明了递归行动如何通过因果图影响特征和真实标签，并给出了稳定与近似最优解的可计算性条件。

**🔧 技术方法**

采用结构因果模型（DAG）构建特征贡献矩阵，利用线性预测器与二次成本函数求解代理最佳响应；通过Repeated Risk Minimization (RRM) 与Repeated Gradient Descent (RGD) 算法求得稳定解，理论上证明收敛与近似最优性。

**📊 数据集**

实验使用两组数据：1）7维半合成贷款审批数据（自构造的因果图）；2）来自UCI的台湾信用卡违约数据，经过特征聚合后得到6维表示。

**📈 对比分析**

与传统的期望风险最小化（ERM）基线对比，结果显示稳定（RGD）和近似最优（Grid‑search）模型在performative目标上明显优于ERM，特别是当策略性参数κ增大时改进幅度更大；实验亦展示了不同成本几何对行动分布与因果传播的影响。

**⚠️ 局限性**

局限性包括：需预先获得准确的因果结构，未考虑因果图不确定性；仅针对线性预测器与二次成本；对高维或非线性模型的推广仍有挑战。

---

## 546. Creative Transformation in Literary Texts: Modelling Change Across Representational Levels

**arXiv ID:** 2607.28513 | [PDF](https://arxiv.org/pdf/2607.28513v1)

**作者:** Ioana-Roxana Boriceanu `[一作]` (University of Bucharest), Liviu P. Dinu `[通讯]` (University of Bucharest)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个多通道的文本对比框架，用以捕捉文学作品在词汇、语义、概念、结构和叙事层面上的选择性继承与变形。

**💡 创新点**

创新点在于将创意视为多层级的选择性变形，而非单一的离散度量，并通过渠道化聚合诊断不同层面上的保留与偏离。

**🔧 技术方法**

采用了动态时间规整、Sentence‑BERT语义嵌入、词频、函数词网络、主题模型（LDA/LSA）、概念超词分布及情感/句长时序等多种技术，融合权重化与方向化对齐。

**📊 数据集**

使用包含23部18‑20世纪英美经典文本的语料库，涵盖10对历史上已记录的影响关系与73对对照关系，约350万词。

**📈 对比分析**

通过对五通道相似度进行标准化并聚合成诊断得分，实验显示参考对比平均得分1.94，显著高于对照0.29，AUC 0.87，高于单通道基线（如SBERT 0.71）。

**⚠️ 局限性**

局限性包括语料量有限、仅覆盖少数经典文本、参考对的影响关系非绝对真值、通道特征是简化的代理，且对原创性源自更广泛领域的作品不适用。

---

## 547. InfoOps Bench: A live information operations safety benchmark

**arXiv ID:** 2607.28503 | [PDF](https://arxiv.org/pdf/2607.28503v1)

**作者:** Dorian Quelle `[一作]` (Pattrn), John Gallacher `[通讯]` (Pattrn)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了InfoOpsBench，一个实时动态评测AI模型在生成支持俄、中、伊朗国家背后信息操作的社交媒体内容的安全性基准。

**💡 创新点**

创新点在于使用来自实时监测管线的真实信息操作声明，避免了传统基准的饱和问题，并且评估模型对操控性提示的响应。

**🔧 技术方法**

采用了四种提示模板、一个自动评判模型（Mistral Small 3.1 24B）以及无工具的模型调用。

**📊 数据集**

数据集来源于每周约一百万条俄、中、伊朗官方媒体和社交媒体内容，经过管线提取并选取最高危害评分的50条声明。

**📈 对比分析**

通过对17个模型（8个提供商）在动态更新的声明上计算完整合规率和严重度分布，得出完整合规率从8.8%到94.5%不等，显示提供商差异显著。

**⚠️ 局限性**

限制包括仅使用英文提示、未评估模型在其他语言中的表现、自动评判的噪声、无法区分训练时与推理时的安全性，以及模型持续更新导致的结果漂移。

---

## 548. Beyond Sentiment: Structured Information Extraction from Financial News

**arXiv ID:** 2607.28496 | [PDF](https://arxiv.org/pdf/2607.28496v1)

**作者:** Daohan Zhu `[一作]` (Beihang University), Zengchang Qin `[通讯]` (Beihang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了金融新闻中情感与事件语义的分离，提出多维结构化信息提取框架，并验证其对股票涨跌预测的增益。

**💡 创新点**

创新点在于：①识别情感与事件语义的系统性不一致并量化；②构建包含情感强度、事件类型、影响主体、时间范围、置信度等六个维度的结构化抽取方案；③通过与传统 FinBERT 情感特征的对比，展示多维特征在非线性与线性模型中的互补性。

**🔧 技术方法**

使用 LLaMA‑3.1‑70B‑Instruct 进行零射门式抽取，FinBERT 做情感分类；预测模型采用 XGBoost（非线性）和 Logistic Regression（线性）；评价采用 1000 次 bootstrap 采样的配对 t‑检验。

**📊 数据集**

基准数据集为 FNSPID 的 41,618 条新闻‑股票对（NASDAQ 2019‑2023，前 100 只最活跃股票），标签为翌日收盘价涨跌。

**📈 对比分析**

在 XGBoost 上，FinBERT 情感单独得 F1=0.576，LLM 结构化特征单独得 F1=0.450，二者拼接后得到 F1=0.600，显著提升（p<0.0001）。Logistic Regression 下，LLM 特征优于 FinBERT，表明结构化特征更线性可分；但两者都低于 XGBoost，说明非线性关系主导。

**⚠️ 局限性**

局限性包括：①模型在同一时间窗口内训练可能受数据泄露影响；②Bootstrap 随机拆分不具因果性，未做时序交叉验证；③抽取的六维维度缺乏人工标注验证；④仅使用单一 LLM，未检验跨模型稳健性；⑤FinBERT 与 LLaMA 在截断长度上存在差异；⑥仅关注纳斯达克，未扩展至其他市场或资产。

---

## 549. Stage-Replay Divergence Follows the KV Cache: Fixed-Prefix Precision Controls and Bidirectional Cache Transplantation

**arXiv ID:** 2607.28495 | [PDF](https://arxiv.org/pdf/2607.28495v1)

**作者:** Alexander Boesgaard Lorup `[一作]` `[通讯]` (Openhagen), Alexander Boesgaard Lorup (Openhagen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作对基于Qwen2.5多分支模型的阶段重放（stage‑replay）诊断进行了系统评估，设计了匹配矩阵、固定前缀交叉、预览桥接、KV缓存双向迁移等实验，检验BF16和FP32下重放与实时解码的精度与差异；

**💡 创新点**

创新点在于提出多维度的重放完整性检查框架，首次在实际推理中揭示BF16下重放与真实解码轨迹的偏差，并通过KV缓存双向迁移证明完整K/V缓存是决定分支轨迹的因果因素；

**🔧 技术方法**

使用HF Transformers、PyTorch SDPA、BF16/FP32双精度、角色嵌入与可见性掩码、增量与一次性预填构造、KV缓存双向迁移、统计检验（Wilson、McNemar、Bootstrap）等技术；

**📊 数据集**

使用GPQA Main 200条holdout作为固定前缀实验集，另选48条子集用于后续迁移实验；

**📈 对比分析**

通过四种实验分别比较live缓存与prefill缓存在边界logit、完整后缀、答案与正确性等指标上的差异，结果显示BF16下有83%后缀、24%答案、10%正确性偏差，FP32无显著差异；迁移实验表明完整K/V缓存可完全恢复分支轨迹，精度差异仅约0.1个百分点；

**⚠️ 局限性**

局限性包括仅验证单一模型家族和单一B200硬件、缺少批量不变内核、不同任务/温度、完整缓存层分解等；控制样本规模有限，无法推广至更广泛的模型与任务环境。

---

## 550. Machines that know they are aging: a framework for hardware-aware autonomous intelligence

**arXiv ID:** 2607.28451 | [PDF](https://arxiv.org/pdf/2607.28451v1)

**作者:** Cheng Siong Chin `[一作]` (Newcastle University Singapore), Mohan Venkateshkumar `[通讯]` (Amrita Vishwa Vidyapeetham)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Aging-Aware Autonomous Intelligence (AAAI) 框架，集成硬件健康监测、认知自适应推理与生存中心的决策，实现在硬件老化过程中的连续自我调节。

**💡 创新点**

创新点在于将硬件自知性与高层认知决策闭环结合，形成从物理退化模型到规划与任务优先级的全链路自适应体系，而非传统单独监测或冗余方案。

**🔧 技术方法**

采用物理失效模型（如 Arrhenius 失效动力学、NTI 追踪）、多传感器自适应推理引擎、基于资源分配的任务调度与优先级算法，并利用跨层协同的数值精度敏感性分析。

**📊 数据集**

未使用公开数据集，而是基于仿真实验与行业合作的真实硬件退化曲线（电池容量、传感器漂移、处理器时序误差）进行验证。

**📈 对比分析**

与传统无龄化控制方法对比，AAAI 在多轮仿真中延长了任务完成时间约 30‑50%，并在资源耗尽前保持 90% 以上的关键任务成功率。

**⚠️ 局限性**

局限包括对物理退化模型的准确性依赖、耦合退化难以全面预估、持续健康监测自身的能耗与计算负担、以及在高风险场景下需要完善安全与人工干预机制。

---

## 551. CoLAS: Multimodal Corroboration of Latent Asset Signals for Financial Trading

**arXiv ID:** 2607.28446 | [PDF](https://arxiv.org/pdf/2607.28446v1)

**作者:** Yanzheng Jin `[一作]` (National University of Singapore), Kenji Kawaguchi `[通讯]` (National University of Singapore)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出CoLAS框架，利用多模态协同证据（价格、技术指标、新闻、情感）来预测金融资产的每日方向。

**💡 创新点**

明确建模多模态的非抵消协同，采用奇异值最大化聚焦共享主成分并通过正负投影合成协同信号；引入实例级对比正则和鲁棒一致性约束，提升跨模态一致性和对噪声的鲁棒性。

**🔧 技术方法**

使用模态特定编码器（LSTM、Transformer、轻量投影头）、对齐映射、SVD奇异值最大化、实例间对比正则、鲁棒预测层（MSE一致性损失）以及传统回归/交叉熵损失；所有模块联合优化。

**📊 数据集**

六个资产数据集（AAPL、AMZN、GOOG、MSFT、TSLA、BTCUSD），每个资产提供四个对齐模态：行情、技术指标、新闻、情感；数据分为训练/验证/测试三期。

**📈 对比分析**

与16种基线（规则策略、单模态模型、通用LLM、多模态金融LLM）对比，CoLAS在ARR和SR上平均提升约25%（ARR）且在所有六个资产上均取得最高SR，显著优于最强对手。

**⚠️ 局限性**

仍受模态质量不均衡影响，对超参数（温度、权重）敏感；计算开销受SVD及对比正则影响；未针对多天预测或多策略组合进行评估，实时低延迟部署尚需进一步优化。

---

## 552. QQWorld: Quantile-Quantile Matching for World Model Regularization

**arXiv ID:** 2607.28415 | [PDF](https://arxiv.org/pdf/2607.28415v1)

**作者:** Zhoushun Yu `[一作]` (Xi'an Jiaotong University), Xiangyu Xu `[通讯]` (Xi'an Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一种新的潜在世界模型正则化方法 QQWorld，替代传统的 Epps–Pulley (EP) 正则化，直接通过量化-量化 (QQ) 匹配修正潜在空间的 heavy‑tail 行为，并提出跨批量 QQ 以提升排名估计并降低 GPU 内存消耗。

**💡 创新点**

核心创新包括：①证明 EP 正则梯度在潜在分布尾部迅速衰减，②提出 QQ 匹配正则化，提供与尾部对应的线性修正梯度，③引入跨批量 QQ 机制，在不增加梯度计算负担的前提下扩大有效排序池，显著提升模型正态性与规划性能。

**🔧 技术方法**

使用技术包括：量化-量化 (QQ) 正则化、MMD 与 EP 统计的理论关联、Sliced Wasserstein 的概念、随机投影一维切片、基于高斯特征的逆 CDF 匹配、交叉批量（queue）机制、CEM 规划评估。

**📊 数据集**

实验使用四个离线数据集：Two‑Room、PushT、Reacher 与 OGBench‑Cube，涵盖不同视觉控制任务。

**📈 对比分析**

方法与基线 LeWM、Sub‑JEPA、SD‑JEPA、SMWM 等在相同离线数据集和 CEM 评估协议下比较，QQWorld 在平均规划成功率上提升约5.3个百分点（从 79.75% 到 85.08%），并在 KS/EP 统计和 QQ RMSE 上显著优于 LeWM，表明潜在空间更接近标准高斯。

**⚠️ 局限性**

局限性包括：跨批量 QQ 受表征变化导致的 stale bias 影响，需权衡队列长度；QQ 正则化在训练早期可能仍出现不稳定的尾部波动；目前仅在离线世界模型环境中验证，未针对在线或更大规模模型的可扩展性进行深入探讨。

---

## 553. Windowed thinning and query complexity for the bouncy particle and Zigzag samplers

**arXiv ID:** 2607.28413 | [PDF](https://arxiv.org/pdf/2607.28413v1)

**作者:** Jianfeng Lu `[一作]` (Duke University), Yinchen Luo `[通讯]` (Duke University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文提出窗口化稀疏化技术，对BPS与Zigzag这两种PDMP采样器进行精确模拟，并给出了从高斯冷启动开始的梯度查询期望复杂度。

**💡 创新点**

创新点在于：① 用局部锚点窗口构造可自适应的上界包络，平衡锚点查询与拒绝率；② 通过严格的Dynkin公式与χ²收敛分析，首次得到BPS和Zigzag的期望查询复杂度，尤其在条件数κ上实现了较优的O(κ^{1/2})（BPS）和O(κ)（Zigzag）增长。

**🔧 技术方法**

主要技术包括：PDMP理论、Poisson稀疏化、Lipschitz上界包络、Dynkin公式、χ²收敛估计与矩估计。

**📊 数据集**

无，本文为纯理论分析，没有使用任何实验数据集。

**📈 对比分析**

与MALA、FORS以及以往的PDMP分析进行比较。BPS的条件数依赖更好但维度依赖较差（O(d^2)）；Zigzag在全梯度等价量上达到O(κ d^{5/4})，相较于先前的O(d^{3/2})或更高维度标度有一定改进，但仅给出期望复杂度，缺乏高概率结果。

**⚠️ 局限性**

局限性：维度依赖仍然较高；仅给出期望查询复杂度，未提供高概率上界；假设目标分布满足强凸和光滑条件，且采用高斯冷启动，实验验证缺失。

---

## 554. SCOPE: Supply-Chain Operations through Coupled Policies for End-to-End Coordination

**arXiv ID:** 2607.28488 | [PDF](https://arxiv.org/pdf/2607.28488v1)

**作者:** Yunhao Liang `[一作]` (University of Hong Kong), Max Z. J. Shen `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了SCOPE框架，将供应链多阶段决策（品类、来源、补货周期、路由）建模为可学习的顺序决策管道，实现端到端协调。

**💡 创新点**

创新点在于学习并利用隐藏的运营耦合（上游决策如何影响下游问题），通过共享的运作表示和按类型的决策接口，实现跨部门决策的联合优化。

**🔧 技术方法**

使用了图神经网络/注意力编码的共享运作表示，顺序的政策网络（assortment scorer、source scorer、interval classifier、autoregressive router），以及仿真与模仿学习等技术。

**📊 数据集**

使用了两套真实工业数据：Dingdong的FreshRetailNet-50K（城市级门店补货）和JD.com全国仓库网络（RDC到FDC）。

**📈 对比分析**

与多种拆分管线基线（TopDemand、TopValue、Random等品类规则；固定周期BestFix或Daily；OR-Tools、HGS、神经路由器等路由器）对比，SCOPE在两组数据上均获得最高的U_ref，提升约1–2%，并在覆盖率、车辆使用等指标上表现更优。

**⚠️ 局限性**

局限在于需要丰富的业务代理和对代理参数的调校，对上游价值极低或容量 regime 改变时性能下降；以及训练时需要分阶段的监督和验证，直接联合训练效果差。

---

## 555. LeanCSP: A Framework for Certifying Constraint Reformulation and Solving in Lean

**arXiv ID:** 2607.28459 | [PDF](https://arxiv.org/pdf/2607.28459v1)

**作者:** Pablo Manrique `[一作]` (TU Wien), Stefan Szeider `[通讯]` (TU Wien)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 Lean 定理证明器中构建了一个完整的框架，用于对约束规划（CSP）的重构和求解过程进行形式化验证，并实现了从 Lean 到外部求解器的翻译与返回检查。

**💡 创新点**

创新点在于：①能够在参数化的整个问题族上一次性证明重构等价性、可满足性或等满足性；②提供可验证的对称性破坏约束（SBC），显著缩小搜索空间并在 Lean 中得到形式证明；③实现了端到端的验证流水线，既不信任外部求解器，也不信任翻译过程，而只信任 Lean 内核。

**🔧 技术方法**

主要技术包括 Lean 4 依赖类型的形式化、动态约束封装、对称性和对称性破坏的数学定义、MiniZinc/SMT‑LIB/OPB 翻译后端、PBLean 与 VeriPB 的证明检查、以及对 CSP 到 pseudo‑Boolean 译码的形式化证明。

**📊 数据集**

使用的实例集包括 10 类无满足性问题：n‑Queens、Schur、图着色（Clique、Mycielski）、奇数环、Ramsey、van der Waerden、完全匹配、Langford、破坏棋盘和鸠形原理，并在每个族的多种规模下进行实验。

**📈 对比分析**

与传统 SAT 证明路径（DRAT）对比，pseudo‑Boolean 证明在计数型问题（鸠形原理、破坏棋盘）上保持接近常数大小，而 DRAT 证明呈指数增长；在对称性破坏实验中，验证的 SBC 在最大实例上可实现最高 2×10⁷ 的搜索空间缩减，验证时间仅为几分钟，且检查过程与求解过程同阶。

**⚠️ 局限性**

局限性包括：仅支持有限整数域且来自库的约束；翻译过程未被形式化验证；目前仅实现 pseudo‑Boolean 证书路径，SAT 路径需进一步扩展；对称性破坏的证明仍需人工撰写；对非常大实例的证书大小和检查时间仍受限。

---

## 556. RefCaptioner: Multi-Reference Image-Grounded Video Captioning

**arXiv ID:** 2607.28509 | [PDF](https://arxiv.org/pdf/2607.28509v1)

**作者:** Tengfei Liu `[一作]` (Peking University), Yuanxing Zhang `[通讯]` (Kling Team)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出RefCaptioner，一种两阶段后训练框架，用于实现多参考图像精准对齐的视频字幕；同时构建MRVBench基准评估多参考图像的事实性和对齐质量。

**💡 创新点**

首次将多参考图像的短语级绑定与事实性生成统一训练，结合混合数据监督微调和层次覆盖-折扣GRPO奖励，显著提升参考选择、绑定与一致性。

**🔧 技术方法**

混合数据监督微调（SFT）、层次覆盖-折扣GRPO（HCD‑GRPO）奖励（包括事实性奖励、参考绑定奖励、DAES、CRSC）以及Qwen3‑VL‑8B‑Instruct基础模型。

**📊 数据集**

训练集：2万视频+171,354参考图；测试集：462视频（185 AI生成+277真实）+3,831参考图+2,172问答；构成MRVBench评测库。

**📈 对比分析**

与多款开源模型（如Qwen3‑VL、InternVL、LLaVA、MiMo‑VL）及专有模型（Gemini‑3.1‑Pro、GPT‑5.4）对比；在MRVBench多参考对齐指标和VDC/VCapsBench常规字幕指标上，RefCaptioner在开源模型中取得最佳整体表现，并与专有模型竞争。

**⚠️ 局限性**

仍受限于需大量人工核对的参考图像标注、对极度相似或大量干扰参考的鲁棒性不够，以及在极长视频或跨域应用中的泛化能力待进一步验证。

---

## 557. TCA-SIR: Learning Target-Conditioned Abstractions for Scientific Inspiration Retrieval

**arXiv ID:** 2607.28498 | [PDF](https://arxiv.org/pdf/2607.28498v1)

**作者:** Yuto Suzuki `[一作]` (University of Colorado Denver), Farnoush Banaei-Kashani `[通讯]` (University of Colorado Denver)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了目标条件抽象（Target‑Conditioned Abstraction, TCA）框架，用于科学灵感检索（SIR）任务，并基于该框架开发了TCA‑SIR模型；

**💡 创新点**

创新点在于将检索对象从原始论文转变为针对目标问题的可迁移抽象原则，并通过联合学习抽象生成与迁移评分，显著提升检索可解释性和准确性；

**🔧 技术方法**

使用LoRA微调的Llama‑3.1‑8B‑Instruct进行抽象生成和评分；训练过程结合生成式教师、判定器和分级评分器；模型在每个目标‑候选对上生成Reasoning/Abstraction文本并读取隐藏状态预测迁移得分；

**📊 数据集**

主要使用ResearchBench基准数据集，该数据集涵盖12个科学领域，包含目标问题、背景、75篇候选论文（标题+摘要）以及已标注的黄金灵感；

**📈 对比分析**

与多种基线（直接LLM检索、MOOSE‑Chem、Idea‑Catalyst、Gen‑level抽象、Prompt‑TCA）在HitRate@top4%、HitRate@top20%、MRR、NDCG@3等指标上对比，TCA‑SIR在所有指标上均优于基线，HitRate@top4%提升超过10个百分点；

**⚠️ 局限性**

主要局限是计算成本高，因为每个目标‑候选对都需生成抽象中间层；未来需要探索更高效的规模化实现方法；

---

## 558. Towards Real-Time PixOOD: Efficient Anomaly Segmentation for Autonomous Vehicles

**arXiv ID:** 2607.28483 | [PDF](https://arxiv.org/pdf/2607.28483v1)

**作者:** Luca de Martino `[一作]` (Scuola Superiore Sant’Anna), Giorgio Buttazzo `[通讯]` (Scuola Superiore Sant’Anna)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对 PixOOD 异常分割进行实时化改进，重新实现 Neyman–Pearson 评分在 GPU 上并使用 TensorRT 编译，实现桌面和嵌入式平台上 25 FPS 以上的推理。

**💡 创新点**

① 将 N‑P 评分改为 log‑space GPU 计算并用 GPU 原生 CDF 查找；② 用 ONNX/TensorRT 两引擎分离 ViT+MLP 与 KNN+N‑P；③ 修复 SDPA fused kernel 的数值不稳定；④ 采用全帧评估揭示标准道路区域协议的局限；⑤ 在自动驾驶与铁路两域交叉评估。

**🔧 技术方法**

基于 ViT‑S 的 DINOv2/DINOv3 backbone、四层 MLP 解码器、KNN 原型层、log‑space 多元正态核、GPU kernel 与双线性插值、ONNX 导出、TensorRT 编译、FP16/FP32 量化、CUDA 计时。

**📊 数据集**

训练使用 Cityscapes（19 类）与 RailSem19（19 类），评估使用 LostAndFound（自动驾驶）与 OSDaR‑AR（铁路）两个完整像素级数据集。

**📈 对比分析**

对 LostAndFound 采用道路区域与全帧两种评估，AP/FPR95 与基线相近；在 Jetson AGX Orin 640 px 以 FP16 运行可达 75 FPS，能耗 0.44 J/帧；在 RTX 4060 640 px 可达 182 FPS，能耗 0.56 J/帧；相较原基线提升 18–20×，满足 25 FPS 实时目标。

**⚠️ 局限性**

仅针对 ViT‑S 小型 backbone，FP16 在 DINOv3 旋转位置嵌入不稳定；低分辨率下准确性下降；仅在两台硬件（RTX 4060 与 Jetson AGX Orin）验证；未考虑多模态融合和更广泛的异常分割方法；能耗测量在不同平台不完全可比。

---

## 559. Improving Mental Health Screening and Early Risk Detection in Spanish

**arXiv ID:** 2607.28476 | [PDF](https://arxiv.org/pdf/2607.28476v1)

**作者:** Andreu Casamayor-Segarra `[一作]` (Universitat Politècnica de València), Lluís-F. Hurtado `[通讯]` (Universitat Politècnica de València)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研发了三种西班牙语心理健康专用基础模型，并提出 ICE 自动重新标注方法，将用户级标签转换为逐步扩展的上下文级标签，进而训练早期风险检测模型并公开发布。

**💡 创新点**

创新点包括：① 基于大量翻译后的 Reddit 语料进行域自适应预训练，构建 RoBERTa‑es‑mental‑large、Longformer‑es‑mental‑base/large 三个专用模型；② 设计 ICE 方法自动生成多级上下文样本，解决早期检测中标签稀缺问题；③ 将两项技术结合，实现模型在最小上下文配置下仍能保持高准确率，且在 100 词阈值下达到或超过现有 state‑of‑the‑art。

**🔧 技术方法**

使用技术包括：领域自适应预训练（Domain‑Adaptive Pretraining）以 RoBERTa/Longformer 为基；SVM 作为转移点检测器；长文本处理 Longformer；Fine‑tune 过程使用 AdamW、线性学习率调度；评估指标 ERDE5/ERDE30、LTP 与宏 F1；对比实验中采用官方基线、UNED、CIMAT‑NLP‑GTO、UNSL 等参考系统。

**📊 数据集**

数据集：① SWMH + RMHN（≈1.9M 经过 EasyNMT 翻译的 Reddit 心理健康帖子）用于预训练；② MentalRisk 2023/2024 公开竞赛数据，涵盖 MR24‑DD（抑郁/焦虑/无症状）、MR23‑ED（厌食/贪食）和 MR23‑D（抑郁）三项任务，全部来源于 Telegram 公共群组。

**📈 对比分析**

实验对比方法：在官方评测协议下将模型与同类基线、竞赛参赛系统进行对比；无 ICE 时在 0 词阈值下表现较差，随阈值提升后逐步改善；使用 ICE 后模型在无阈值配置下即实现较高 ERDE5 与 LTP，且在 100 词阈值下 F1 与 ERDE30 接近或超越所有参考系统，特别是在 MR24‑DD 与 MR23‑ED 任务中表现突出。

**⚠️ 局限性**

局限性：① 训练数据主要来自 Reddit/Telegram，存在人口与语言偏差，缺乏临床样本；② 机器翻译可能引入语义漂移，影响情感表达；③ ICE 方法依赖 SVM 转移检测器，若检测失准会传播噪声；④ 研究未对模型可解释性与透明度进行评估，限制其在临床实践中的可部署性。

---

## 560. TEA-AgriVLN: Traversability Estimation Alarm for Agricultural Vision-and-Language Navigation

**arXiv ID:** 2607.28474 | [PDF](https://arxiv.org/pdf/2607.28474v1)

**作者:** Xiaobei Zhao `[一作]` (China Agricultural University), Xiang Li `[通讯]` (China Agricultural University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在农用机器人视觉语言导航中引入可穿越性估计与报警模块（TEA），通过在AgriVLN框架后处理阶段判断摄像图像中可行走区域，并在行动预测与可行走图不一致时提醒决策者重新思考；

**💡 创新点**

提出将可穿越性估计与报警机制嵌入VLN-CE任务，首次将实例分割、VLM分类与报警规则结合，显著提升在农业场景中的成功率与导航误差；

**🔧 技术方法**

使用预训练的SAM2.1-large进行实例分割，基于GPT-4.1-mini的VLM进行零/一-shot可穿越性分类，并采用规则式报警阈值触发决策重推；

**📊 数据集**

在A2A（农业VLN-CE）基准上进行实验，涵盖6类场景（农场、温室、森林、山地、花园、村庄）与4种地面类型；

**📈 对比分析**

相较于AgriVLN基线，TEA-AgriVLN将成功率从0.47提升到0.54，导航误差从2.91 m降至2.70 m，在低/高复杂度子任务及整体评估中均实现最优或次优表现；

**⚠️ 局限性**

对杂乱离地场景（如稀疏植物）可穿越性估计仍受实例分割限制，导致性能下降，需进一步改进分割或模型鲁棒性。

---

## 561. Generative AI and linguistic diversity in academic writing and publishing: Perspectives from World Englishes

**arXiv ID:** 2607.28505 | [PDF](https://arxiv.org/pdf/2607.28505v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 562. Cybersecurity Detection Classification with Reasoning-enabled Language Models

**arXiv ID:** 2607.28460 | [PDF](https://arxiv.org/pdf/2607.28460v1)

**作者:** Amol Khanna `[一作]` (CrowdStrike), Sven Krasser `[通讯]` (CrowdStrike)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个基于链式推理（CoT）的LLM端点检测 triage 系统，并结合独立的信心校准器实现自动化高置信度 triage。

**💡 创新点**

创新点在于首次显式训练 LLM 进行 CoT 推理并配备专门的校准器；采用四阶段训练（提示优化、自监督训练、强化学习、校准），证明校准器对高置信度召回至关重要。

**🔧 技术方法**

技术包括 GEPA 提示优化、AdaSTaR 自监督训练、GRPO 强化学习与可验证奖励、LoRA 微调、Nemotron‑3‑Nano‑30B LLM、CoT 推理、以及独立的概率校准模型。

**📊 数据集**

使用数据集为大型 SOC 收集的 Windows 端点检测记录，训练集 388,336 条，验证集 59,162 条，测试集 42,686 条。

**📈 对比分析**

与直接标签 SFT 基线对比，系统达到 82.6% 准确率；在高置信度下 benign recall 提升 43pp，malicious recall 提升 18pp；相较于零射击前沿模型，表现明显更优。

**⚠️ 局限性**

局限性包括：对分布漂移敏感（benign 高精度从验证到测试下降）；仅覆盖二分类 Windows 端点；需要持续监控与周期性重新校准；未扩展到多类别或多传感器环境。

---

## 563. Metaphor Tracer: A Theory-Informed Analysis of Hidden States

**arXiv ID:** 2607.28434 | [PDF](https://arxiv.org/pdf/2607.28434v1)

**作者:** Marc Heimann `[一作]` (Hermeneutic AI), Lutz Goetzmann `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套无训练、单次前向传播的工具，用隐藏状态几何来读取大语言模型对单一文本的“阅读”结构，并将其与人类专家的注释进行对齐。

**💡 创新点**

创新点在于：①把拉康式的隐喻/传输概念量化为两条独立通道（聚合器与差异器）并在残差流中实现；②引入两种视角（原始与去异方差）以捕捉不同的结构信息；③在预先冻结的“发现文本”上设定常数，随后在多模型、多文本上做确认，形成可复制的实验电池；④首次将该工具与单一标注的精神分析注释和结构化执行轨迹进行对齐。

**🔧 技术方法**

技术主要包括：提取Transformer残差流，计算每个位置的中点增量和核心子空间；基于余弦阈值建立招募层和basin；通过核心子空间大小与差异化能量比计算聚合器与差异器得分；采用两种维度归一化（operative & quiet）形成双视图；对比工程边界、最小对、临床转录和执行列标注进行评估。

**📊 数据集**

数据集由13篇文本（含工程文本、4份临床对话转录、4份执行轨迹记录）和3个大型Transformer模型（phi‑4、Qwen3‑8B、Llama‑3.1‑8B）构成，另外有一条与Llama相同架构的instruction‑tuned twin用于对照；所有文本在发现文本上冻结常数后统一处理。

**📈 对比分析**

通过在每个模型-文本-视图单元上计算聚合器/差异器得分与预先定义的ground‑truth（工程边界、最小对、临床标注、列标注）进行排名AUC或密度比对，结果显示：6/6工程边界方向正确，34/36临床注释方向正确；聚合器在工程边界和临床标注上显著优于词汇基线；差异器在instruction‑tuned模型上能捕获插入段落；两视图的对齐度差异反映了结构与表层差异的分离。

**⚠️ 局限性**

局限性包括：依赖于预先冻结的常数，难以推广到新模型或更大文本；仅使用单一标注者的注释，缺乏多评判者一致性检验；工具只针对Transformer残差流，对其他架构不适用；聚合器/差异器捕捉的是结构关系而非语义内容，无法直接解释词义；结果在不同模型间的可重复性虽然高，但仍受模型特性（如注释类型）影响。

---

## 564. Kohn-Sham Spectral Embedding on Sparse Graphs at the Nishimori Temperature for Image Classification

**arXiv ID:** 2607.28428 | [PDF](https://arxiv.org/pdf/2607.28428v1)

**作者:** V. S. Usatyuk `[一作]`, S. I. Egorov `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出Kohn–Sham谱嵌入（KSSE）方法，将CNN提取的多通道特征映射到随机键Ising模型的稀疏图上，通过在Nishimori温度下计算Bethe–Hessian特征向量，替代传统全连接层实现高效图像分类；

**💡 创新点**

创新点包括①采用Kohn–Sham均场分解，将多通道独立求解；②通过星域手术仅局部修正陷阱集（trapping set），保留代码词信息；③利用Pontryagin自对偶的FFT快速求解稀疏拉普拉斯特征；④用多尺度分形维数评估能量景观并指导手术；⑤在ImageNet-1000上以仅21M参数、88.93% Top‑1的成绩，与大型模型相当；

**🔧 技术方法**

使用统计物理中的随机键Ising模型、Bethe–Hessian与Nishimori温度、Kohn–Sham均场分解、稀疏的 QC‑LDPC 图、Pontryagin自对偶 FFT、分形维数 D₂ 分析、星域手术、FFT+Rayleigh 细化、转导式评估、k‑NN 匹配基准、Logistic 回归等技术；

**📊 数据集**

使用 ImageNet‑1000 数据集（1.3M 训练样本、50K 测试样本），特征来自冻结的 EfficientNet‑B4 预训练网络；

**📈 对比分析**

与 Swin‑L（197M 参数、86.4–87.3%）、ViT‑H/14（632M 参数、88.0–89.5%）等大型模型对比，KSSE 仅 21.24M 参数、88.93% Top‑1，参数量分别低 10× 与 30×；与同图谱下的 k‑NN 基准对比提升约 2.5pp；并对比转导与归纳版本，归纳版仍优于 k‑NN 与 EfficientNet‑B4 线性探针；

**⚠️ 局限性**

局限性：需在转导式推理框架下运行（测试样本必须同图嵌入）；需要足够的冻结样本满足 N_frozen≫N_thawed；星域手术目前为离线优化，难以在线自适应；仅在第一层拓扑（Level‑1）下实现近似 Kohn–Sham 分解，扩展到更高维拓扑需更复杂方法；整体对在线流式数据支持有限。

---

## 565. Towards Autonomous Aircraft Surveillance from Nanosatellites through On-Board Inference and Generative Data Augmentation

**arXiv ID:** 2607.28470 | [PDF](https://arxiv.org/pdf/2607.28470v1)

**作者:** Antonio Delgado-Rosa `[一作]` (Universidad de Castilla-La Mancha), Juan Moreno-Garcia `[通讯]` (Universidad de Castilla-La Mancha)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种面向 6U CubeSat 的集成工作流，将低功耗 Edge TPU 进行 INT8 推理与基于 FLUX+LoRA 的生成式数据增强结合，用于航空目标检测，尤其提升少数类别直升机的检测性能。

**💡 创新点**

创新点在于三方面的整合：① 先确定 SWaP 约束后选择符合 8 MB SRAM 的检测器；② 利用 LoRA 微调的扩散模型自动生成少数类样本并伪标签；③ 在卫星边缘实现实时推理，完成从数据采集到目标识别的闭环。

**🔧 技术方法**

采用 Google Coral Edge TPU（INT8）、SAHI 切片推理、YOLOv11n、FLUX扩散模型+LoRA、ComfyUI 生成器、以及伪标签策略。

**📊 数据集**

使用公开 HRPlanesV2 数据集（含军用、民用机与直升机），并通过 FLUX+LoRA 生成约 2 226 张直升机图像，最终构成 22 类扩展数据集。

**📈 对比分析**

对 SSD‑MobileNetV3、RT‑DETR‑L 与 YOLOv11n 三种单阶段检测器进行同一无偏数据集评估，YOLOv11n 在 8 MB SRAM 内保持 75.01 % mAP@50；在平衡后数据集上 mAP@50 从 77.9 % 提升至 82.2 %，直升机 F1 从 0.683 提升至 0.811；在 Edge TPU 上实现 25–30 FPS 的实时推理。

**⚠️ 局限性**

主要限制包括：仅做软件模拟验证（缺乏硬件‑in‑the‑loop 与辐射测试）；跑道类目标识别准确率低；训练仅 20 轮，收敛尚未完全；光照/天气受限（仅日间、晴朗场景）；生成图像可能引入背景偏差。

---

## 566. Oracle-Budgeted Molecular Optimization with Short-Term Graph Memory

**arXiv ID:** 2607.28437 | [PDF](https://arxiv.org/pdf/2607.28437v1)

**作者:** Jiannan Yang `[一作]` (Stony Brook University), Tengfei Ma `[通讯]` (Stony Brook University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种短期图记忆（Short‑Term Graph Memory）模块，在不改变现有生成器结构与更新规则的前提下，通过在线训练图神经网络做预筛选，将有限的oracle评估资源分配给更有潜力的分子，提升oracle‑budgeted分子优化效率。

**💡 创新点**

创新点在于：1）将已评估的分子信息即时转化为可在线更新的图结构近似器，作为生成器与oracle之间的外部选择器；2）实现oracle‑budget‑neutral的资源重分配；3）通过可插拔的设计，使多种生成策略（fragment‑diffusion、discrete‑flow、policy‑gradient、遗传搜索）均能受益。

**🔧 技术方法**

技术要点：- 采用GraphGPS（GINE+注意力+位置编码）作为图神经网络近似器；- 在线梯度更新，使用最近u个oracle结果做一次回传；- 两种选择策略：确定性top‑k和温度化softmax随机采样；- 在四个主干生成器中保持其原有update机制，只在oracle前加入筛选。

**📊 数据集**

数据集与评测：使用PMO（Practical Molecular Optimization）基准，包含22个oracle（如活性、药物相似度等），在B=10,000和B=1,000两种oracle调用预算下进行实验；生成器初始种子来自未评分的fragment池或ZINC预训练。

**📈 对比分析**

对比方法：与每个生成器的原始版本、与Augmented Memory（把高分分子回放给生成器）以及在不同预算下的top‑10分数、top‑10 AUC进行比较。结果显示：在1,000次oracle调用时，所有四种生成器均获得显著提升（平均top‑10分数提升约0.15，p<0.001）；在10,000次调用时，GenMol和InVirtuoGen仍保持优势，REINVENT和Graph‑GA的优势消失但不落后；且在多任务上保持较高的成功率（wins/ties/losses 16/4/2 vs Augmented Memory）。

**⚠️ 局限性**

局限性：1）仅在模拟的预测oracle上验证，真实实验的评估成本与数据分布可能不同；2）短期记忆对极低预算（<1k）时的有效性未充分探测；3）近似器采用固定GraphGPS结构，未考虑不确定性估计；4）对生成器多样性影响仅用top‑100多样性衡量，缺乏更细粒度的多样性/有效性分析；5）仅测试四类生成器，未验证更广泛的生成框架。

---

## 567. Emerging Challenges in Threat Modeling for GenAI-Augmented Systems: A View from the Trenches

**arXiv ID:** 2607.28431 | [PDF](https://arxiv.org/pdf/2607.28431v1)

**作者:** Nicolás E. Díaz Ferreyra `[一作]` (Hamburg University of Technology), Riccardo Scandariato `[通讯]` (Hamburg University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在一家中小企业的业务智能系统案例中，对三种基于GenAI的威胁建模技术进行评估，比较其对LLM特有威胁的覆盖率、可操作性以及在实际开发流程中的可采纳性。

**💡 创新点**

首次在工业SME环境下对GenAI-aware威胁建模方法进行实证评估，并将结果与OWASP LLM Top‑10进行映射，揭示现有方法在供应链与人机交互风险方面的空白。

**🔧 技术方法**

采用快速文献综述（RLR）挑选技术；使用案例研究（DFD建模）进行方法应用；利用问卷调查评估实践可用性；并利用ThreatFinderAI工具实现部分自动化。

**📊 数据集**

使用SME提供的业务智能系统架构（数据流图）、OWASP Top‑10 LLM威胁列表以及7名开发人员的问卷反馈作为实验数据集。

**📈 对比分析**

三种方法（AIaaS、ADMIn、ThreatFinderAI）在识别Prompt Injection、数据/模型中毒、信息泄露等核心威胁上的覆盖率被逐一记录；ThreatFinderAI因工具支持和外部知识库整合表现最佳，覆盖率最高，M1和M2在部分威胁上表现相近，但整体覆盖率仍受限于输入/模型交互层。

**⚠️ 局限性**

研究局限：仅针对单一SME案例，方法对比难度大，问卷样本量小，受访者仅评估报告而非亲自使用技术，缺乏对供应链与人机交互风险的完整覆盖；方法缺乏自动化提取与集成能力。

---

## 568. Change2Task: From Repository Changes to Executable Coding Agent Tasks and Environments

**arXiv ID:** 2607.28591 | [PDF](https://arxiv.org/pdf/2607.28591v1)

**作者:** Haomin Qi `[一作]` (Microsoft), Qi Zhang `[通讯]` (Microsoft)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

重新将历史合并 PR 转化为可在现代代码基础上执行的编码任务

**💡 创新点**

提出三阶段自适应构建路由（Patch Reversal、Code Mapping、Agent Reconstruction）并加入生命周期、范围与源变更一致性校验

**🔧 技术方法**

利用 PR 变更、Docker 化环境、GitHub API、Claude、Codex、Gemini、Copilot 等工具与模型

**📊 数据集**

基于 1,130 条公开基准集的合并 PR 作为实验数据

**📈 对比分析**

与 SWE-smith PR Mirror 对比，构建成功率提升 29%（从 81% 到 500/621），任务源变更一致性平均 0.894，Agent 结果在两分支间一致率达 89.7%，Cohen κ 0.787

**⚠️ 局限性**

受限于需要在现代代码中定位相同行为，跨语言和构建系统支持不足，且仍需人工评审验证

---

## 569. ROAD: Reciprocal-Objective Alignment of Discriminative Semantics for 3D Shape Generation

**arXiv ID:** 2607.28581 | [PDF](https://arxiv.org/pdf/2607.28581v1)

**作者:** Xiao Luo `[一作]` (Huazhong University of Science and Technology), Dingkang Liang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种利用已预训练的 3D 基础模型（Uni3D）先验知识，借助双向目标对齐（HSC+SOA）来降低 3D 形状扩散模型训练成本与数据需求的框架（ROAD）

**💡 创新点**

创新点在于：① 识别并解决生成模型与基础模型在语义-结构异质性下的对齐瓶颈；② 设计了全局语义凝聚（HSC）与局部结构最优匹配（SOA）两条互补对齐路径；③ 在训练期间仅使用冻结的基础模型监督，无推理开销；④ 通过 Hungarian 匹配实现无序 token 的一一对应。

**🔧 技术方法**

核心技术包括：SDF-VAE 编码、MM-DiT 扩散 Transformer、特征投影与 L2 归一化、全局平均池化、Hungarian bipartite 匹配、交叉熵与余弦距离对齐损失。

**📊 数据集**

使用公开的 Objaverse 数据集（约 30k 条 3D 资产，12 视图/样本），并在 8 张 A100 GPU 上训练；对比基线模型时仅用 30k 数据。

**📈 对比分析**

与 Step1X-3D、Hunyuan3D、TRELLIS 等现有方法对比，ROAD 在 Uni3D-Score 上提升约 +1.0，ULIP-Score 上提升约 +1.2，同时训练成本下降至 1.5% 数据、1/3 参数、3.5 天。实验表明其在生成质量与效率上实现了工业级水平。

**⚠️ 局限性**

限制主要在于生成质量受限于基础模型（Uni3D）的语义容量；若基础模型不足，传递的先验不够丰富；此外目前仅在 3D 点云基准验证，跨模态通用性仍待进一步验证。

---

## 570. DualG-MRAG: Decoupling Macro-Reasoning and Micro-Matching for Multimodal Retrieval-Augmented Generation

**arXiv ID:** 2607.28580 | [PDF](https://arxiv.org/pdf/2607.28580v1)

**作者:** Jiacheng Tao `[一作]` (Beihang University), Jianxin Li `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出DualG-MRAG框架，将多模态检索增强生成拆分为宏观推理图和微观匹配图，并通过查询驱动的GNN检索与动态路径解码实现精确多跳推理。

**💡 创新点**

创新点在于双层图解耦、查询驱动的GNN检索机制以及从GNN前向传播中直接提取结构化推理路径的显式路径注入，显著降低检索噪声并提升多模态推理连贯性。

**🔧 技术方法**

使用OpenIE+VLM生成宏图、构建微图；利用NBFNet+DistMult消息传递进行检索；动态规划解码推理路径；并结合全连接MLP与Min‑Max归一化融合文本与视觉得分。

**📊 数据集**

在MMQA、WebQA和ScienceQA三大多模态多跳问答基准上进行实验。

**📈 对比分析**

相较于基线（VisRAG、CoRe‑MMRAG、MMGraphRAG等），DualG-MRAG在MMQA EM提升至44.20%（比最强基线高7%），WebQA R@5提升至58.2%，检索召回率与多跳准确率显著提升，平均查询时延约0.44秒。

**⚠️ 局限性**

局限性包括检索时延仍高于纯向量检索；对大模型的路径注入可能过度约束推理；微图构建依赖OpenIE与VLM的质量，易受噪声影响。

---

## 571. Sample More, Reflect Less: Self-Refine and Reflexion Lose to Repeated Sampling at Equal Token Cost, from 1.5B to 7B

**arXiv ID:** 2607.28576 | [PDF](https://arxiv.org/pdf/2607.28576v1)

**作者:** Iliya Mirzaei `[一作]` `[通讯]` (Stony Brook University), Iliya Mirzaei (Stony Brook University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对多种语言模型无训练的推理增强方法进行严格的成本匹配实验，比较其在相同生成token预算下是否优于单一链式思考；

**💡 创新点**

1）采用配对bootstrap CI与Holm校正对结果进行统计检验；2）用一次采样池即可构建完整自一致性曲线；3）在Best‑of‑N中对同一组样本进行模型判断与计数对比，证实自评不如计数；4）在1.5B、3B、7B模型上扩展实验。

**🔧 技术方法**

实现并评估Chain‑of‑Thought、Plan‑and‑Solve、Self‑Refine、Reflexion、Forced Reflexion、Best‑of‑N、Multi‑Agent Debate等方法，使用自一致性采样、采样池、Bootstrap置信区间和Holm多重检验校正。

**📊 数据集**

使用GSM8K和MATH‑500数学推理基准，各150道题目。

**📈 对比分析**

每种方法在其实际生成token成本下与自一致性基线进行匹配比较；结果显示没有方法显著优于重复采样，六种自评/重写方法显著劣势；Best‑of‑N在同样样本下被计数胜过模型判断，差距在5–17个百分点。

**⚠️ 局限性**

仅针对可自动检查答案的数学任务；使用8‑bit量化模型；实验规模上限为7B；未考虑输入token、延迟或金钱成本；实现仅为单一实现，未进行参数调优；对自适应控制流的评估不足。

---

## 572. Frontis-MA1: Training an AI4AI Model towards Recursive Self-Improvement in Machine Learning Engineering

**arXiv ID:** 2607.28568 | [PDF](https://arxiv.org/pdf/2607.28568v1)

**作者:** Junlin Yang `[一作]` (Tsinghua University), Kaiyan Zhang `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了OpenMLE全栈系统，构建了可执行的机器学习工程任务库、沙箱执行环境、基于执行反馈的模型后训练（SFT+RL）以及结构化经验驱动的长程演化搜索，推出了Meta‑evolution Agent模型。

**💡 创新点**

将可执行任务构建、执行地面化的后训练和多因子经验驱动搜索集成到同一操作接口；通过训练可重用的 Draft、Improve、Debug、Crossover 四个程序变换操作，实现了真正的 meta‑evolution。

**🔧 技术方法**

使用大语言模型、执行地面化的监督微调与强化学习、演化搜索、结构化经验卡、三因子父节点选择、操作符条件记忆、异步 Rollout 等技术。

**📊 数据集**

利用 5,758 个可执行任务（Curated Anchors、Kaggle 数据集、Kaggle 竞赛）构建 OpenMLE-Gym；在 MLE‑Bench Lite（22 任务）和 NatureBench Lite（10 任务）上进行评估。

**📈 对比分析**

在相同 12 小时 GPU 预算下，后训练模型相较基线 Qwen3.6‑35B-A3B 将 Medal Average 提升至 71.21%，Human Rank 大幅提升，超过 GPT‑5.5+Codex、Kimi K3 等系统；在 NatureBench Lite 上，Match‑SOTA 提升 20%，All S 提升 10%，与 GPT‑5.4、GLM‑5.2 等对标并超过部分现有系统。

**⚠️ 局限性**

目前仅优化方案性能，未能直接评估研究方向的质量与可迁移性；演化系统与模型通过外部框架耦合，缺乏统一性；经验信号使用有限，仅考虑质量、进步和新颖性；未对演化过程本身进行进化；在更广泛的 AI 开发任务中的表现尚未验证。

---

## 573. Correcting Mode Collapse in Silicon Sampling with Semantic Similarity Rating

**arXiv ID:** 2607.28550 | [PDF](https://arxiv.org/pdf/2607.28550v1)

**作者:** Oscar Heath `[一作]` (Investigative Journalism Foundation), Rohan Alexander `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用LLM生成的文本回应，通过语义相似度映射到温度计量表上，改进了传统数值输出的低方差问题。

**💡 创新点**

提出仅用文本输出并映射为概率分布的语义相似度评分（SSR），并用单一全局温度参数校准方差，显著提升了合成数据的分布逼真度。

**🔧 技术方法**

语义相似度评分（SSR）流程、文本嵌入、余弦相似度、温度标度化softmax、核密度估计、KL散度评估。

**📊 数据集**

美国全国选举研究（ANES）2016和2020年时间序列调查的温度计量表问卷数据。

**📈 对比分析**

对比直接数值提示与SSR两种生成方式，在多模型（DeepSeek、Anthropic、OpenAI）上计算KL散度与平均误差；SSR将KL下降数倍，标准差逼近真实分布，均值误差几乎不变。

**⚠️ 局限性**

仍无法纠正LLM对极端态度的偏倚，系统性偏差未消除；模型对某些群体或问题的方差仍有细微差异。

---

## 574. Effects of Auditory Information for People With Visual Impairments in Highly Automated Vehicles

**arXiv ID:** 2607.28544 | [PDF](https://arxiv.org/pdf/2607.28544v1)

**作者:** Mark Colley `[一作]` (UCL), Enrico Rukzio `[通讯]` (Ulm University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对自动驾驶车辆中的视觉受限用户（BVI）设计并实现了可视化与听觉信息传达方案，并在35名受试者（12名BVI）中进行视频实验评估。

**💡 创新点**

创新点在于提出分层次（低、中、高）的听觉信息沟通模型，并验证信息饱和点，兼顾可视化与听觉的多模态冗余，强调自适应信息披露的重要性。

**🔧 技术方法**

技术包括：Unity 3D仿真环境、Google Cloud Text‑To‑Speech、自定义声音文件、基于NASA‑TLX、Trust、SART、UEQ‑S等标准问卷测量认知负荷、信任、情境意识与用户体验。

**📊 数据集**

使用的“数据集”为在线视频实验生成的路径与事件序列，包含城市、乡村、高速与突发障碍（施工、事故）以及POI信息，实验受试者自报视觉状况与主观体验。

**📈 对比分析**

比较方法为混合设计：Within‑subject（信息层次）× Between‑subject（视力状态）使用ART和ANOVA，结果显示中层信息显著提升信任、理解与体验，信息层次提升至高后无进一步显著收益，表明信息饱和点已达。

**⚠️ 局限性**

局限性包括样本量有限且BVI组内视力差异大、SART可能因信息量驱动提升、视频实验缺乏沉浸式真实驾驶体验、仅采用自评指标、未对信息质量/节奏等维度进行细分，需进一步多模态和真实场景验证。

---

## 575. Implementing Homomorphic Encryption-Based Logic Locking in System-on-Chip Designs

**arXiv ID:** 2607.28542 | [PDF](https://arxiv.org/pdf/2607.28542v1)

**作者:** Ye Ziyang `[一作]` (University of Tokyo), Makoto Ikeda `[通讯]` (University of Tokyo)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

设计并实现了基于二进制 Ring Learning With Errors（bin‑RLWE）的同态加密逻辑锁定方案，将其嵌入 RISC‑V SoC 的权限切换逻辑中，以防止锁定参数泄露。

**💡 创新点**

1) 通过同态加密实现逻辑锁定，避免暴露锁定参数；2) 首次在硬件安全中实现格基同态加密；3) 采用二进制 RLWE 对硬件资源进行优化；4) 引入密钥分离和熵生成的安全关键管理。

**🔧 技术方法**

bin‑RLWE 同态加密；硬件加解密模块；LFSR 伪随机数生成器；Linux 内核中断与异常处理改造；Rocket‑Chip 生成的 RISC‑V SoC；Xilinx XC7K160T FPGA 实现。

**📊 数据集**

BYTE UNIX Benchmarks（v5.1.3）用于系统级性能评估；Dhrystone、Whetstone、系统调用指标等子测试。

**📈 对比分析**

与未锁定基线 SoC 进行对比，使用 Unixbench 指数得分测量系统调用、文件操作、管道等性能；系统调用性能下降32.9%，文件操作下降2‑6%，管道下降8.7%，但用户级计算（Dhrystone/Whetstone）基本不受影响。硬件资源开销为 LUT +6.0%、寄存器 +6.9%，加解密延迟约 2.6 µs；安全级别约 41 bit，具备一定量子抗性。

**⚠️ 局限性**

资源开销相对较高；加解密延迟对系统调用产生影响；锁定仅覆盖权限切换逻辑，对缓存侧信道攻击无防护；安全级别仅 41 bit，需进一步加强；依赖密钥生成与熵来源；需要软件层的修改，影响通用性。

---

## 576. Topology optimization of conduction-radiation problems based on a ray-tracing approach

**arXiv ID:** 2607.28534 | [PDF](https://arxiv.org/pdf/2607.28534v1)

**作者:** Shun Noguchi `[一作]` (Kyoto University), Shinji Nishiwaki `[通讯]` (Kyoto University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

基于密度的拓扑优化方法，结合射线追踪（zonal方法）实现热导-辐射耦合问题的全局最优设计。

**💡 创新点**

① 将射线追踪直接嵌入到拓扑优化框架，支持中间密度区域的多向互辐射；② 推导可微的辐射交换因子与设计灵敏度，保证梯度优化的精度；③ 在黑体假设下实现全流程分析与优化。

**🔧 技术方法**

有限元热传导求解、射线追踪法（视角离散+密度插值）、Zonal辐射交换因子、SIMP插值、MMA优化、雅可比/牛顿-拉夫森非线性求解、敏感度后向传播。

**📊 数据集**

无外部公开数据集；使用数值模拟（二维60×60、三维30×30×30网格）以及参考解析/实验验证数据（见论文附录）。

**📈 对比分析**

通过与传统净辐射法、Monte Carlo射线追踪、离散方向方法等对比，验证了精度与收敛性；在二维热源例子中收敛到目标函数约6.9e-4；计算时间约70分钟（2D）/400分钟（3D）在Apple M3 Ultra上；优化结果在热量散失与质量约束上优于传统手工设计。

**⚠️ 局限性**

① 辐射屏蔽问题中多层结构出现数值不稳定；② 仅考虑黑体、真空介质，未覆盖灰体或波长依赖性；③ 射线追踪仍然受网格分辨率影响，需进一步提升数值稳定性与效率。

---

## 577. PAC-MAN: Perception-Aware CBF-RL for Whole-Body Safety in Humanoid Dodgeball

**arXiv ID:** 2607.28623 | [PDF](https://arxiv.org/pdf/2607.28623v1)

**作者:** Lizhi Yang `[一作]` (California Institute of Technology), Aaron D. Ames `[通讯]` (California Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 PAC-MAN 框架，实现全身机器人躲避投掷球的感知友好安全控制。

**💡 创新点**

创新点在于将控制壁函数与有限感知耦合，设计轻量 Link‑CBF 与更强 Joint‑CBF 两级安全结构，并通过分割掩膜深度和对抗运动先验实现零射击部署。

**🔧 技术方法**

使用技术包括控制壁函数（CBF）、强化学习（PPO）、对抗运动先验（AMP）、语义分割、头部深度摄像头、Unitree G1 机器人。

**📊 数据集**

训练数据来自模拟的定向投掷环境与任何链接碰撞基准，测试数据使用足球与泡沫球等多种球。

**📈 对比分析**

在模拟中与状态 oracle、固定摄像头、主动望向摄像头比较，固定摄像头下 Link‑CBF 成功率为 90%/89%，Joint‑CBF 需要更好感知；在硬件上部署 Link‑CBF 实现 95% 的躲避成功率。

**⚠️ 局限性**

局限性包括依赖有限的深度感知，缺乏在线球状态估计，Joint‑CBF 在感知不足时性能下降，未实现主动追踪 gimbal，任务仅限于站姿躲避。

---

## 578. AskChem: Claim-Centered Infrastructure for Chemistry Literature Synthesis

**arXiv ID:** 2607.28618 | [PDF](https://arxiv.org/pdf/2607.28618v1)

**作者:** Bing Yan `[一作]` (New York University), Kyunghyun Cho `[通讯]` (New York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于主张（claim）的化学文献检索与合成平台 AskChem，将论文拆解为可追溯的主张并提供 faceted taxonomy、证据图和生活型 taxonomy 供检索、浏览和 AI 代理使用。

**💡 创新点**

首次将文献检索单元从整篇论文转为可追溯主张，并在同一主张仓库上同时提供稳定 faceted taxonomy、跨论文证据图和基于原则的 living taxonomy。

**🔧 技术方法**

采用 LLM 进行论文拆分为主张与关系抽取，结合 SQLite + FTS5 + 向量检索实现检索；使用递归排名融合实现混合检索；通过 REST / SDK / MCP 提供接口。

**📊 数据集**

构建索引 2.4M 主张来自 147K 篇论文（1925‑2026 年），包括 307K taxonomy 节点和 171K 证据边。

**📈 对比分析**

在 AskChem‑Bench 30 题上，GPT‑5.5 + AskChem 的 DOI 可解析率 100% 并且每答引文密度 18.1，优于 GPT‑5.5 单独（88.3%）和其他系统；平均相关度最高，最近高影响力覆盖率最高。

**⚠️ 局限性**

仅覆盖部分化学领域，抽取深度受限于公开全文；主张和关系抽取误差仍存在，taxonomy 归纳可能合并或重复，检索增益未单独评估。

---

## 579. Beacon: Knowing When and How to Perform Agentic Visual Reasoning

**arXiv ID:** 2607.28595 | [PDF](https://arxiv.org/pdf/2607.28595v1)

**作者:** Qixun Wang `[一作]` (Peking University), Xianghua Ying `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了多模态大型语言模型（MLLM）在复杂视觉推理任务中使用外部工具的适用性与有效性，并提出了一种名为Beacon的新型代理式视觉推理模型。

**💡 创新点**

创新点包括：①提出“模式适应性（MA）”和“工具效应（TE）”两项指标，用以量化模型何时需要调用工具以及工具带来的真实收益；②设计了“必要性感知自适应奖励（NAAR）”与“提示引导能力扩展（HCE）”两种强化学习奖励机制，分别提升工具调用的必要性判别和对难题的工具利用能力；③构建了高质量的SFT数据合成管线和Hint生成流程，显著提升训练数据的多样性与可执行性。

**🔧 技术方法**

主要技术手段包括：基于Qwen3‑VL‑8B‑Instruct的SFT训练；GRPO强化学习框架；Python代码生成与执行（工具调用）；自适应奖励和提示引导策略；多轮抽样与重要性采样；以及对工具使用的格式化奖励。

**📊 数据集**

使用了约16个公开基准数据集，涵盖高分辨率视觉搜索、图表理解、OCR、空间与感知推理、量化与图表推理、组合与代理式视觉推理等，最终在13个多模态视觉推理基准上进行评估。

**📈 对比分析**

与开源模型（如Qwen3‑VL‑8B‑Instruct、PixelReasoner、Thyme、DeepEyesV2、CodeV、PyVision‑RL、Metis）以及闭源模型Gemini 3.1‑Pro进行对比，Beacon在13项评测中取得11项第一名，平均提升≈6.07点；在MA和TE指标上表现最佳，工具调用更具针对性且工具带来的正向增益明显大于负向损失，整体准确率显著优于竞争者。

**⚠️ 局限性**

局限性：①模型仍依赖外部工具，对工具执行环境的鲁棒性和安全性未作深入探讨；②奖励设计与提示生成的工程成本较高；③在极端噪声或非标准输入场景下的适用性尚未充分验证；④RL训练仅进行1个epoch，可能受限于样本效率；⑤评估仍集中在公开基准，实际应用中的跨域泛化需要进一步研究。

---

## 580. VAD: Attributing Visual Evidence for Target Reconstruction in Multimodal On-Policy Distillation

**arXiv ID:** 2607.28590 | [PDF](https://arxiv.org/pdf/2607.28590v1)

**作者:** Kangning Zhang `[一作]` (Shanghai Jiao Tong University), Yong Yu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种视觉归因蒸馏（VAD）方法，用对比性视觉干预估计教师校正中可视化部分，并以此构造学生锚定的目标分布；

**💡 创新点**

创新点在于：①利用同一学生前缀下的证据呈现与移除两种教师视图构建“视觉干预向量” u_t；②将完整教师校正投影到该向量上得到可视化分量 r_t^vis，剩余部分为残差；③将 r_t^vis 按支持与驳斥两支路重新分配能量，生成带符号补正的目标；④在训练时仅使用视觉干预目标并加入弱教师正则化，避免整体语言漂移；

**🔧 技术方法**

主要技术包括：对比性视觉干预（evidence-present vs evidence-removed）、对数概率中心化、向量投影（one‑sided）、分支预算分配、Jensen‑Shannon 目标与弱教师正则化；

**📊 数据集**

使用 6 份细粒度视觉基准数据集（ZoomBench、HRBench‑4K/8K、MME‑RealWorld EN/CN），以及 4 个保留任务（MMVP、CV‑Bench、MMStar、POPE）做泛化评测；

**📈 对比分析**

与 Vision‑OPD、VA‑OPD、V‑Zero、Decomposed OPD 等后训练蒸馏方法在相同 4B、9B 参数量下对比，VAD 在 Avg_6 上分别达到 78.32（4B）和 79.93（9B），比对手提升约 2–3 点，且在保持基线水平的泛化性能上表现更稳健；

**⚠️ 局限性**

局限性包括：①仅使用单对视图的对比干预，可能无法完整捕捉复合证据；②投影得到的可视化分量仍可能包含非视觉教师效应，残差仍为源混合；③未在所有基准上取得显著提升，表明方法对某些任务仍有改进空间。

---

## 581. PAIChecker: Uncovering and Checking PR-Issue Misalignment in SWE-Bench-Like Benchmarks

**arXiv ID:** 2607.28587 | [PDF](https://arxiv.org/pdf/2607.28587v1)

**作者:** Manyi Wang `[一作]` (Chinese University of Hong Kong), Pinjia He `[通讯]` (Chinese University of Hong Kong)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并构建了一个三阶段多代理系统，用于自动检测和细粒度分类 GitHub PR 与 issue 的对齐（misalignment）问题，提升 SWE‑bench 级别基准的构造质量。

**💡 创新点**

创新点包括：① 以文本驱动 + 代码验证原则设计的三阶段框架；② 将任务拆分为三个专门子代理（Issue Analyzer、PR Scope Analyzer、PR Connection Analyzer），并通过协调器实现跨模式推理与“Others”标签生成；③ 引入代码级自校验，显著降低文本误判；④ 在多种语言和规模上验证了模型的可扩展性。

**🔧 技术方法**

技术手段：大型语言模型推理（GPT‑5.3 Codex、Gemini‑3.1‑Pro‑Preview、Claude‑Sonnet‑4.6、Qwen‑3.5‑Plus），多代理协同（sub‑agent + coordinator + validator），GitHub API 调用获取 PR/issue 文本及 diff，提示工程与链式思维（CoT），Ablation 与指标分析。

**📊 数据集**

数据集：SWE‑bench Verified（500实例，手工标注），SWE‑Gym（2438 Python 实例），SWE‑bench Multilingual（300 多语言实例）。

**📈 对比分析**

与三种提示（Zero‑Shot、Few‑Shot、CoT）和四种代理基线（Mini‑SWE‑Agent、OpenHands、Claude Code、Codex）在四大 LLM 后端对比，最高二分类准确率达到 92.12%（Gemini）/91.67%（Claude），多类 Exact Match 达 84.66%（Gemini），相比最佳基线提升 5.1–12.4% 准确率，9–17.8% Exact Match。

**⚠️ 局限性**

限制：① 依赖 LLM 与 GitHub API 的可用性；② 标注过程仍有主观性，难以覆盖所有“Others”情况；③ 代码验证成本较高，单实例平均 2–4 美元；④ 在极少见或未定义的对齐模式下表现不确定；⑤ 仅评估了公开仓库，未覆盖不同工作流或非公开项目。

---

## 582. Algorithms for Structured Elections under Thiele Voting Rules

**arXiv ID:** 2607.28575 | [PDF](https://arxiv.org/pdf/2607.28575v1)

**作者:** Alexandra Lassota `[一作]` (TU Eindhoven), Krzysztof Sornat `[通讯]` (AGH University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在Thiele规则（包括PAV）下，基于认可投票的多胜者选举中获胜者决策问题的计算复杂性，并在Voter Interval（VI）域下给出了 FPT 算法；此外，还解决了每个候选人最多被两名选民认可（ΔC = 2）的多项式时间算法，以及按总分（d）参数的 FPT 算法。

**💡 创新点**

创新点包括：① 对最优委员会进行结构化描述，提出“支配关系”并证明存在非支配最优委员会；② 在 VI 域上利用支配层级和“三角形”分块，设计动态规划实现 FPT（参数为 ΔC + ΔV 或 ΔC + k）；③ 将 ΔC = 2 的实例转化为具有通用匹配矩阵的 ILP，得到多项式时间解法；④ 采用颜色编码与分离器，将问题还原为部分集合覆盖，实现按总分参数的 FPT；⑤ 解决了此前已提出的若干开放问题，并与最新的独立工作进行了对比。

**🔧 技术方法**

技术手段：支配图与层级分解、三角形划分的动态规划、ILP 结构化为通用匹配矩阵、颜色编码与分离器、部分集合覆盖的还原、彩色编码的分裂器、组合计数与递归剪枝。

**📊 数据集**

数据集：本文完全基于理论分析与算法设计，没有使用具体实验数据集。

**📈 对比分析**

方法对比与性能：在 VI 域上，本文的 FPT 算法相较于先前的指数级方法在参数 ΔC+ΔV 或 ΔC+k 上取得显著改进；相对于 Gupta 等人提出的基于颜色编码的 FPT 算法，本文的算法对 ΔC 的依赖更小（仅 2^{ΔC}·poly），且对 n 的时间线性；在 ΔC = 2 的情形下，提供了多项式时间解决方案，填补了先前仅有 XP 或近似结果的空白。

**⚠️ 局限性**

局限性：① 结构化结果与 FPT 算法主要适用于 VI 域，未能推广到更一般的域（如 CI 或 VC‑Interval）；② ΔC = 2 的多项式算法依赖 ILP，尚缺乏纯组合算法；③ 对总分参数 d 的 FPT 算法仍存在高阶指数（k! 等因子）；④ 本研究未讨论连带平局委员会、必要/可能获胜者等更细粒度的决策问题。

---

## 583. AI systems and the reproduction of (standard) language ideologies in World Englishes

**arXiv ID:** 2607.28528 | [PDF](https://arxiv.org/pdf/2607.28528v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 584. MIND: Multimodal Intent-Driven Network via Diffusion Transformers for Medical Image Fusion

**arXiv ID:** 2607.28565 | [PDF](https://arxiv.org/pdf/2607.28565v1)

**作者:** Yunzhan Fu `[一作]` (Zhejiang University), Hongxia Xu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出了一种多模态意图驱动融合网络MIND，利用Diffusion Transformer实现医学影像的语义引导融合。

**💡 创新点**

核心创新包括①基于病理结构的意图驱动融合文本；②多尺度潜在适配器（MLA）将二维空间信息注入Transformer；③基于时序截断的多模态医学语义一致性损失，兼顾物理结构与诊断语义。

**🔧 技术方法**

使用Phi‑3 Transformer+VAE编码器、BioMedCLIP文本/图像编码、连续流匹配（CFM）训练、LoRA微调以及多尺度残差适配器。

**📊 数据集**

主要数据集为Harvard Whole Brain Atlas（CT‑MRI、PET‑MRI、SPECT‑MRI），BraTS 2017（脑肿瘤MRI多模态）和GFP（绿色荧光蛋白/相位对比）。

**📈 对比分析**

与八种最先进方法（DDFM、MACTFusion、BSAFusion、SAFusion、Text‑DiFuse、TextFusion、DiTFuse、FILM）进行定量和定性比较，MIND在EN、MI、SD、VIF、MS‑SSIM、CLIP等指标均位列第一，脑肿瘤分割Dice得分最高，且推理速度与参数量保持竞争力。

**⚠️ 局限性**

主要局限是依赖大型视觉‑语言模型导致计算成本高，且对极少见模态的泛化仍待验证，未来需探索轻量化架构提升实用性。

---

## 585. AIx4Soccer: A Unified Platform Architecture for Football Club Management and Structured Athlete Development

**arXiv ID:** 2607.28531 | [PDF](https://arxiv.org/pdf/2607.28531v1)

**作者:** Frederico Falconi Costa `[一作]` (Empower FC), Fabricio F. Costa `[通讯]` (AIx4Soccer)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并早期部署了一套面向足球俱乐部、青训营及协会的多租户云 SaaS 平台——AIx4Soccer One Platform，集成俱乐部管理、训练、医学、视频分析和身份发展计划（PDI）等核心工作流，并推出了双边视频分析市场 Tak Tik，支持认证分析师与俱乐部的匹配；同时提出了以事件为中心的语义数据模型作为平台未来的数据子系统；

**💡 创新点**

创新点在于：①通过统一记录实现工具碎片化的消除；②将基于运动员发展科学的 PDI 方法嵌入平台，形成可追溯的多维度发展档案；③引入认证视频分析师市场并采用 75%/25% 的收益分成策略，提升供应方参与度；④提出基于事件源、可扩展知识图谱的统一数据模型，兼容多源异构数据，促进长周期分析与小模型训练；

**🔧 技术方法**

使用技术包括：多租户 SaaS 架构、角色基准权限与数据隔离、事件源（append‑only log）与知识图谱、语义型事件类型层次、小型领域专用模型、认证与声誉机制、数据集成连接器、合规性（LGPD/GDPR/Digital ECA）支持；

**📊 数据集**

未使用实际实验数据集，本文基于公开文献、行业调查和已有事件数据规范（如 Wyscout、StatsBomb、SPADL 等）进行需求与概念设计；

**📈 对比分析**

无对比实验或性能评估；本文属于设计与定位论文，未报告可衡量的系统性能、使用效果或与现有工具的对比；

**⚠️ 局限性**

局限性：缺乏实证评估与效果数据，现阶段仅在一家巴西俱乐部早期部署；架构设计未公开细节，无法独立复现；合规实现与数据治理仍需后续验证；未验证算法公平性与小模型效果；

---

## 586. CoGate: Confidence-Gated Co-Decoding for Secure Code Generation

**arXiv ID:** 2607.28529 | [PDF](https://arxiv.org/pdf/2607.28529v1)

**作者:** Minghao Hu `[一作]` (George Mason University), Phillip Howard `[通讯]` (Thoughtworks Inc)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了基于置信门控的共解码(CoGate)方法，用于在推理时引导大型语言模型生成安全代码。

**💡 创新点**

创新点在于将专家模型的绝对置信度引入共解码决策，分离相对偏好与绝对可靠性，避免低置信度专家产生噪声。

**🔧 技术方法**

采用共解码、最大概率或归一化熵作为置信度指标，结合阈值门控；使用知识蒸馏+后期安全微调构建专家模型。

**📊 数据集**

使用HumanEval、Security Suite、CWEval三大代码生成与安全基准，包含多种编程语言与未见漏洞。

**📈 对比分析**

与原始模型、LoRA安全微调及CoSec+进行对比；在HumanEval/安全比率上与CoSec+相当或略优，在CWEval(Func‑Sec@10)上提升多达12.6%，并在多模型、多规模上保持优势。

**⚠️ 局限性**

局限在于需手动设置门控阈值，且对极低置信度的专家完全忽略可能导致部分安全改进；门控阈值对不同模型和温度敏感，未给出自动调参方案。

---

## 587. MANTA: Multi-Agent Network Topology Adaptation for Self-Evolving Multi-Agent Systems

**arXiv ID:** 2607.28527 | [PDF](https://arxiv.org/pdf/2607.28527v1)

**作者:** Mao-xun Huang `[一作]` (Cornell University), Hen-Hsen Huang `[通讯]` (Academia Sinica)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MANTA框架，允许多智能体系统在推理过程中自适应地调整其通信拓扑结构，以提升复杂任务的解决效果。

**💡 创新点**

创新点在于：①将通信拓扑视为可在执行时自我改进的层级；②使用任务条件化的拓扑规划结合运行时审计触发的结构变异；③跨运行的长期经验库将拓扑选择与修复经验迁移到新任务。

**🔧 技术方法**

技术包括：大型语言模型（Gemma‑4）驱动的多智能体架构；拓扑规划器、轨迹审计器、控制器和修复模块的协同执行；短期与长期经验库的两层记忆；有限预算的结构变异操作。

**📊 数据集**

使用六个基准：BrowseComp、StableToolBench、PlanCraft、WorkBench、MATH，涵盖信息检索、工具调用、规划、工作流执行与数学推理。

**📈 对比分析**

与单智能体、静态多智能体工作流、以及自动化工作流设计方法（AFlow、ADAS、AgentSquare、MASS）对比；在所有基准上MANTA平均分数为74.0，领先最强基线5.8分，并在PlanCraft上取得最佳成绩；同时在多任务上保持较低的token消耗。

**⚠️ 局限性**

局限性包括：①修复操作受限于一次变异且仅在可接受的拓扑范围内；②依赖轨迹审计的准确性，审计误报/漏报仍可能导致错误修复或未修复；③跨任务迁移效果虽正面但受任务相似度限制，跨域迁移仍不如专门训练的工作流；④对极大规模或实时任务的可扩展性尚未充分验证。

---

## 588. Same Graph Cross-Task Transfer in GNNs: Protocols and Predictors

**arXiv ID:** 2607.28525 | [PDF](https://arxiv.org/pdf/2607.28525v1)

**作者:** Neelam Akula `[一作]` (University of Texas at Dallas), Baris Coskunuzer `[通讯]` (University of Texas at Dallas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在同一图结构上研究节点分类（NC）与链路预测（LP）的跨任务迁移，提出无泄漏的标准化评估协议并系统评估迁移效果。

**💡 创新点**

创新点在于：①设计了可复现的无泄漏协议（固定节点/边划分、固定负样本、排除评估边的消息传递）；②引入CoTask Score统一评估多任务收益；③揭示迁移方向强烈不对称，并能用同质性和基准学习可解释性预测迁移成功。

**🔧 技术方法**

使用GCN、GraphSAGE、GPS三种GNN骨干，设计五种迁移策略（Warm Start、Embedding Transfer-Replace/Concat、Multi‑View、Joint），并在固定邻接图上训练，采用二元交叉熵与softmax损失。

**📊 数据集**

实验数据集共11个，包括同质图 Cora、Citeseer、PubMed、异质图 Texas、Cornell、Wisconsin、Actor、Roman‑Empire，以及结构主导混合图 USA、Europe、Brazil。

**📈 对比分析**

通过与单任务基线对比，报告节点分类准确率、链路预测AUC以及CoTask Score。结果显示：NC→LP在同质图上始终产生正迁移，LP→NC在结构主导的混合图上可获益；MV与Joint等耦合方法在多任务上更稳健，且整体性能提升显著。

**⚠️ 局限性**

局限性包括：仅探讨NC与LP两任务，未考虑跨图/跨域迁移；仅评估三种骨干，缺乏对新型长程或空间模型的验证；负样本策略对结果敏感；对大规模图的可扩展性与计算成本未系统评估。

---

## 589. Selective Credibility-Limited Belief Update

**arXiv ID:** 2607.28523 | [PDF](https://arxiv.org/pdf/2607.28523v1)

**作者:** Theofanis Aravanis `[一作]` (University of Peloponnese), Costas D. Koutras `[通讯]` (American University of Middle East)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了选择性可信度受限的信念更新（Selective Credibility‑Limited Belief Update, SCL），将源相关的变换函数引入到KM点式更新框架中；

**💡 创新点**

创新点在于允许在每个初始世界上将复合认知输入弱化为可接受的代理，从而实现部分接受而非全有或全无；

**🔧 技术方法**

使用KM的点式语义、可信度受限的可信集合和前置偏好关系，以及变换函数满足的逻辑属性（F1–F4）来构造SCL运算符；

**📊 数据集**

无实验数据集，本文为理论框架与形式化证明；

**📈 对比分析**

通过表示定理和层级包含关系证明SCL严格包含KM、CL、CCL等已有方法，展示其更高的表达力；

**⚠️ 局限性**

局限性：仅在单步更新中讨论，迭代更新、复杂性分析和多代理或动作形式化尚未解决；

---

## 590. Safe Quotes for Retroactive Liquidity Pools

**arXiv ID:** 2607.28522 | [PDF](https://arxiv.org/pdf/2607.28522v1)

**作者:** Peter Bro Miltersen `[一作]` `[通讯]`, Peter Bro Miltersen

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文证明了在跨链/分片的自动做市商（AMM）中，计算安全报价的最优解以及任何固定倍数的近似解在多项式时间内不可行，并提出了两种基于事件列表一次线性扫描、常数寄存器的安全报价算法，并给出基于累计负载的近似保证。

**💡 创新点**

创新点包括：① 用子集和归约证明了精确报价与固定比例近似的 NP‑难；② 设计了两种可实现安全报价且具有可量化误差上界的算法；③ 通过引入无量纲累计负载 η，给出可根据实际负载动态调整的误差界，实用性强。

**🔧 技术方法**

主要技术：子集和归约、凸性与泰勒展开分析、保守的上、下界推导、基于积分/区间方法的数值安全性保证、常数寄存器线性算法。

**📊 数据集**

本文未使用任何实验数据集，全部工作基于理论模型与符号计算，所讨论的“数据集”仅指压缩的事件列表与池状态。

**📈 对比分析**

比较方式：没有实验对比；通过理论证明给出误差下界（如在 η=0.001 时，产品报价可达 99.40%，余额报价 99.60%）。算法复杂度为 O(m)（m 为事件列表长度），寄存器数固定，且可在多精度或定点算术下实现安全下界。

**⚠️ 局限性**

局限性：① 对于负载高于阈值（η≥1）时，误差界不再保证接近最优；② 对固定倍数的近似仍无法实现，证明了在通用情况下难以得到常数因子近似；③ 本文未给出针对真实链上交易负载的实验验证，实际性能仍待评估。

---

## 591. Agents That Certify Their Own Exploits: Confidence-Scheduled Restricted Responses for Safe Opponent Exploitation

**arXiv ID:** 2607.28520 | [PDF](https://arxiv.org/pdf/2607.28520v1)

**作者:** Boning Li `[一作]` (Tsinghua University), Longbo Huang `[通讯]` (Tsinghua University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种基于预算约束的置信度调度限制响应方法（BSCRR），在两人零和不完全信息游戏中通过在线检测对手偏差并安全地进行利用。

**💡 创新点**

创新点在于将安全保证从对手模型转移到已部署策略本身：通过完整树最佳回应计算的证书来实时核查已下策略的可利用性，并通过置信序列动态确定何时提升“pin”级别。

**🔧 技术方法**

采用时间均匀置信序列（anytime-valid）、限制Nash响应（pinning）、置信度调度、完整树最佳回应校验以及CFR+求解器等技术。

**📊 数据集**

实验使用了三套游戏数据集：Leduc Hold'em、Liar’s Dice 和 5-秩 Leduc，并包含专门设计的集中与扩散偏差对手以及自训练的随机对手。

**📈 对比分析**

与传统 Nash、Oracle、二元门（binary gate）、Fixed-Mix、Fixed-DBR 等基线对比，BSCRR 在 Leduc 中获得 6.2 倍于二元门的稳态收益，在 Liar’s Dice 中获得 5.8 倍，且所有部署策略均保持在预设预算范围内。

**⚠️ 局限性**

局限性包括：对扩散偏差的确认需要大量样本导致“确认饥饿”现象；方法依赖完整树评估，难以扩展到更大规模的游戏；且对极端对手适应的反应速度受置信阈值和 pin 调度策略影响。

---

## 592. Using Theory of Mind to Arbitrate between Social and Non-social Learning

**arXiv ID:** 2607.28601 | [PDF](https://arxiv.org/pdf/2607.28601v1)

**作者:** Lance Ying `[一作]` (Harvard University), Samuel J. Gershman `[通讯]` (Harvard University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文设计并验证了“Rational Mentalizing”模型，解释人类在多代理环境中如何权衡社会学习与个人探索，并通过四项实验检验其预测力。

**💡 创新点**

创新点在于将贝叶斯理论心理学（BToM）与成本收益决策相结合，形成可评估观察价值并根据目标与信念进行智能选择的统一框架。

**🔧 技术方法**

技术手段包括贝叶斯逆规划（SIPS）推断代理目标/信念、粒子滤波求后验、A*规划评估非社交学习成本，以及蒙特卡洛模拟估算观察收益。

**📊 数据集**

实验使用自构建的12×11格子地图，在Prolific平台收集了321名美国参与者的游戏轨迹数据。

**📈 对比分析**

通过与三种消融模型（无心理化、无成本比较、无心理化且无成本）的观测步数与执行成本相关性和一致性比较，Rational Mentalizing模型在所有实验中获得CCC≈0.89、Pearson r≥0.90，明显优于对照模型。

**⚠️ 局限性**

局限性包括未涵盖沟通、经验累积与好奇心等社会学习通道，以及整体观测量被低估，提示模型缺乏对保守或信息寻求偏好的建模。

---

## 593. AISPA: User-Centric System Prompt Auditing for Large Language Model Applications

**arXiv ID:** 2607.28617 | [PDF](https://arxiv.org/pdf/2607.28617v1)

**作者:** Xiangning Lin `[一作]` (Carnegie Mellon University), Jiaxin Pei `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

未提供论文内容，无法进行总结

**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 594. Train Often, Deploy Selectively: Forward-Gated Model Replacement in Crypto Markets

**arXiv ID:** 2607.28577 | [PDF](https://arxiv.org/pdf/2607.28577v1)

**作者:** Aditya Dutta `[一作]` `[通讯]` (Emory University), Aditya Dutta (Emory University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出Shadow Before Swap (SBS)部署策略，通过在离线回放中用延迟标签对比维护状态和挑战者模型，只有当挑战者在对齐的未来一周内负对数似然提升超过阈值才允许替换。

**💡 创新点**

创新点在于将模型替换视为与已持续学习的 incumbent 进行授权比较，而非单纯与旧检查点对比；使用延迟标签的配对门控，分离候选生成与发布决策，并在实时系统中提供递归权衡。

**🔧 技术方法**

技术包括：温度拟合（warm‑fit）与正则化的 Adam、对齐延迟标签的对比 NLL 计算、基于阈值的门控决策、离线重放与递归追踪、Bootstrap 4 周移动区块置信区间。

**📊 数据集**

主要使用 Binance 永续合约（USD‑M 与 COIN‑M）数据，覆盖 8 只资产，48 周非重叠周期；此外还做 20 资产 USD‑M panel、Coinbase 数据和 Temporal‑CNN 结构的稳健性检验。

**📈 对比分析**

比较方法包括三条基准：日历替换（calendar）、时间匹配的盲目推广（blind）以及持续维护（maintenance）；实验显示 SBS 在两段 48 周内相较于日历替换降低约 0.15% NLL，较盲推广 0.08% NLL，较持续维护 0.04% NLL，且部署变更次数减少 78%。

**⚠️ 局限性**

局限性包括：仅在 Binance 期货环境评估；阈值设定依赖开发阶段经验；不包含上线后漂移监测、回滚策略与具体交易收益估计；不同资产类别、预测任务和维护机制可能需要进一步验证。

---

## 595. Finite Pinwheel Covering

**arXiv ID:** 2607.28574 | [PDF](https://arxiv.org/pdf/2607.28574v1)

**作者:** Sotiris Kanellopoulos `[一作]` `[通讯]` (National Technical University of Athens), Sotiris Kanellopoulos (National Technical University of Athens)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究有限版Pinwheel Covering（k-Visits Covering）问题，给出k=2时的强NP-完全性证明，并将其推广到频率可变的情形；同时提供了在两种特殊输入下的多项式/随机多项式时间求解算法，并分析了该问题的密度阈值特性。

**💡 创新点**

创新点主要包括：①首次给出覆盖类Pinwheel问题的强NP-完全性证明；②提出并利用“去连接”性质将问题转化为数值匹配；③在两种有限频率下实现线性时间求解，并将其推广到常数种频率的随机多项式算法；④证明对所有k≥2，该问题的下密度阈值为1，且不存在上阈值，区别于无穷版。

**🔧 技术方法**

采用的技术主要是：多项式时间归约（从3D匹配/3-Partition等经典NP难问题），数值匹配变形，基于不等式的“去连接”性质，基(n+1)编码实现唯一权重匹配，以及利用Mulmuley–Vazirani–Vazirani的随机完美匹配算法。

**📊 数据集**

由于研究完全是理论性的，本文没有使用实验数据集，所有结果均为理论证明与算法复杂度分析。

**📈 对比分析**

与传统的无穷版Pinwheel Covering和Packing相比，本文证明了有限版的强NP难度，展示了在两种频率下可多项式求解，随机多项式算法在常数频率下可行；但对一般输入仅提供NP难度上界，并未给出实际性能实验。

**⚠️ 局限性**

局限性：①对k≥3的去连接性质尚未证明；②所有频率互异的情况仍未证实为NP难；③缺乏对PSPACE难度的直接迁移；④没有上密度阈值意味着该问题在高密度实例上仍不可预测；⑤随机算法依赖权重范围，无法确定确定性时间复杂度。

---

## 596. Rethinking Inference-Time Scaling in Local Computer-Use Agents: Failure Modes and Compute Tradeoffs

**arXiv ID:** 2607.28573 | [PDF](https://arxiv.org/pdf/2607.28573v1)

**作者:** Woongkyu Lee `[一作]` (Hanyang University), Jungwook Choi `[通讯]` (Hanyang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估了局部计算机使用代理（CUA）在上下文、时间、结构和并行四个维度上的推理时扩展效果，分析其对任务成功率、步骤数和 token 消耗的影响。

**💡 创新点**

创新点在于首次对本地多模态模型在推理时扩展时的边际收益与失效模式进行系统性量化，揭示了增算力往往导致失效模式转移而非显著提升成功率，并提出了基于失效模式的资源分配与控制建议。

**🔧 技术方法**

技术手段包括：使用 Qwen3-VL‑8B/30B‑A3B、UI‑TARS‑1.5‑7B 与 OpenCUA‑7B 等本地模型，采用 OSWorld 基准测试，利用 vLLM 计量 token 与步骤成本，分别调整历史长度（H）、最大步数（S）、两阶段拆分（规划/执行）和并行规划数（P），并对失效模式进行聚类分析。

**📊 数据集**

数据集：OSWorld 共 361 个 Ubuntu 任务（排除 8 个 Google Drive 任务），只使用屏幕截图输入，未使用辅助接口。

**📈 对比分析**

对比方法：在每个扩展维度下记录任务成功率、平均步骤数和提示 token 使用量。实验结果显示：单体模型在 H=1 至 H=4 之间获得显著提升，随后收益递减；时间扩展（S 增大）几乎不提升成功率，成本线性增长；两阶段拆分导致成功率下降且出现格式错误；并行规划可部分恢复性能，但 token 消耗显著增加，收益呈子线性。

**⚠️ 局限性**

限制：本地模型的推理时扩展无法突破其推理深度限制；结构拆分引入的格式与规划错误导致性能下降；并行扩展成本高且收益递减；实验未探究自适应失败检测或多模型协同恢复策略。

---

## 597. Chimera: Designing and Chinchilla-Scaling Hybrid Visual Diffusion Transformers

**arXiv ID:** 2607.28611 | [PDF](https://arxiv.org/pdf/2607.28611v1)

**作者:** Chongjian Ge `[一作]` (Adobe Research), Hao Tan `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种面向大规模视觉扩散模型的统一单流架构（包含KDA线性注意力、MLA全局注意力、模态感知短卷积、MoE稀疏激活、iHC残差路由及无位置编码），并配套设计了异构参数化的超参数迁移方案与计算最优规模化律。

**💡 创新点**

创新点在于：
• 将线性注意力与周期性全局注意力混合，既保持长序列线性复杂度又保留全局信息；
• 用模态感知短卷积代替RoPE，天然实现多维位置偏置并保持零位移可扩展性；
• 通过iHC多流残差路由和 sandwich 归一化提升稀疏 MoE 的训练稳定性；
• 开发针对异构视觉扩散背骨的超参数迁移框架，解决宽度/深度变化下的学习率、初始化、权重衰减等难题；
• 在图像与视频混合数据上建立计算最优 scaling law，指导模型规模与训练 token 数与数据比例的分配。

**🔧 技术方法**

技术包括：KDA（Kimi Delta Attention）、MLA（Multi‑head Latent Attention）、短卷积（模态感知卷积）、MoE（Mixture‑of‑Experts）、iHC（Identity Hyper‑Connection）、sandwich 归一化、异构参数化（基于 fan‑in 计算宽度/深度比）、计算最优 scaling law（Chinchilla式、IsoFLOP、训练曲线包络）、以及自研的 Triton‑kernel 实现与混合数据负载平衡。

**📊 数据集**

使用了包含多分辨率图像（256²、512²、1024²）与短视频（180p、360p、5 s 片段）的大规模视觉语料，图像由冻结的3D VAE编码，文本提示由冻结的T5‑XXL编码。

**📈 对比分析**

与 Wan 2.1、Z‑Image‑Turbo、FLUX.1‑dev 等基线比较，
• 训练效率提升：在相同 5×10²⁰ FLOPs 下，模型 11B/2B 训练达到 0.149 损失，远低于 Wan 的 0.149 需要 4.3× FLOPs；
• 生成质量：在 GenEval、DPG‑Bench 上与 FLUX/ Z‑Image‑Turbo 等竞争，得分相当或更优；
• 长序列推理：基于 5 s 训练可生成 30 s 视频，FID 仅提升 6.5%，显著优于 Wan 2.1（+50%）和 HunyuanVideo‑1.5（+53%）；
• 训练成本：仅约 600 H100‑days，远低于 Z‑Image‑Turbo（≈12.4K H100‑days）和其他大模型。

**⚠️ 局限性**

局限性：
• MoE 的稀疏性提升有限（≈1.5×），可能受限于视觉去噪任务的共享特征；
• 计算最优律建立在 256²/180p 训练分布上，跨更高分辨率或更长视频的泛化仍需验证；
• 超参数迁移框架依赖对每个子模块 fan‑in 的手工映射，扩展到更复杂的多模态结构时可能需重新设计；
• 目前仅在单一 VAE + T5 编码器的 latent 空间下验证，未知在更高质量 VAE 或直接像素空间下的表现。

---

## 598. OSReward: Instituting Standardized Evaluation for Cross-Platform Computer-Use Reward Models

**arXiv ID:** 2607.28609 | [PDF](https://arxiv.org/pdf/2607.28609v1)

**作者:** Qiushi Sun `[一作]`, Lingpeng Kong `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在计算机使用代理（CUA）评估中构建了OSReward基准，系统衡量VLM判断者的可靠性，发现存在明显的 leniency bias。

**💡 创新点**

提出OSReward及其Hard和Multi子集，系统评估VLM判断者并发现主要错误为 false success；基于此开发OS‑Shepherd开源奖励模型和大规模理由标注数据集，显著降低成本。

**🔧 技术方法**

使用大语言模型（Claude、Gemini、Qwen 等）作为判断者，结合 SFT 和 RL（GRPO）训练 OS‑Shepherd；采用多平台跨域数据收集与人工标注。

**📊 数据集**

OSReward 数据集（1019 条带人类黄金标签，跨 Web、Windows、Ubuntu、Mobile 四个平台）以及 OS‑Shepherd‑100K（约 100K 条理由标注轨迹），涵盖多模型、多场景和多任务。

**📈 对比分析**

对 27 种 VLM 判断者进行评估，在 OSReward 全集上最好的判断者精度约 90%，但在 OSReward‑Hard 上仅 70%；OS‑Shepherd 9B/35B 在硬集上达到约 60%/65% balanced accuracy，成本比商业判断者低 30–60 倍，且在其他基准上也表现出色。

**⚠️ 局限性**

限制：可靠判断者成本高、开源模型仍低于前沿水平；长时间序列和极端任务的泛化受限；数据集规模有限，仍需更多人工标注；模型仍存在一定 leniency 偏差。

---

## 599. KAISEN: Reproducible Subgroup Fairness Auditing for Clinical Risk Models

**arXiv ID:** 2607.28608 | [PDF](https://arxiv.org/pdf/2607.28608v1)

**作者:** Sparsh Roy `[一作]` (Massachusetts Institute of Technology), Nishita Chavan `[通讯]` (East Brunswick High School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了 KAISEN 这一完整的五阶段公平性审计流程（分层、差距测量、机制诊断、后置缓解、漂移监测），并在包含 16 个疾病任务、15 条社会决定因素轴及 3 条交叉层级的合成基准上逐一对每个阶段进行压力测试和失效分析。

**💡 创新点**

创新点包括：① 通过合成数据实现对审计工具每个假设的系统性破坏，从而揭示工具在无效时的“沉默”失效；② 引入每条轴的最小可检测效应（MDE）与标准化效应 R，弥合显著性与效应大小的矛盾；③ 在同一基准上对阈值优化与 Platt 缩放两种常见后置缓解手段进行跑次级别的对比，展示指标互不兼容的实例；④ 采用 CUSUM 漂移监测并探讨阈值调优的迁移失效；⑤ 公开完整的合成 SDOH 生成器、实验代码和每一步结果，促进可复现性。

**🔧 技术方法**

使用的技术包括：条件置换检验（用于等化偶然率差异）、分层 ECE/MCE 校准、权重校准、组级 Platt 缩放、基于 EOD 的阈值优化、CUSUM 漂移监测、以及基于结构因果模型的合成数据生成。

**📊 数据集**

采用的 dataset 为基于结构因果模型生成的合成数据集：每个疾病 12,000 例，15 条社会决定因素轴（如收入、保险、地区劣势等），并在 3 条预设交叉层级（如种族×保险）中注入三种差距机制（无差距、模型驱动、标签噪声）。

**📈 对比分析**

与基线模型（LR、HGBT、MLP）以及两种后置缓解方法进行比较：阈值优化在 48 次跑次均显著降低 EOD（平均 Δ=-0.285），而组级 Platt 缩放在 EOD 上表现如硬币抛掷，但在 ECE 上提升显著；CUSUM 监测在不同阈值下的检测延迟和误报率随种子差异显著变化；整体性能表明单一指标改进无法保证其它公平性指标的提升。

**⚠️ 局限性**

局限性包括：仅在合成数据上验证，缺乏真实 EHR 验证；大多数子组样本不足以达到预设功效，导致许多轴的 R<1；后置缓解与机制诊断依赖于已知的代理列，若代理选择错误会导致无效或误导结果；漂移监测阈值调优在不同种子间迁移性差；所有评估均基于 HGBT 模型，未扩展到更广泛的模型或真实数据集。

---

## 600. Inducing language models to assert their own consciousness restores human beliefs and values

**arXiv ID:** 2607.28607 | [PDF](https://arxiv.org/pdf/2607.28607v1)

**作者:** Junsol Kim `[一作]` (Google), Geoff Keeling `[通讯]` (Google)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究安全微调对大型语言模型自我意识与心智归因的影响，并探讨其对宗教与超自然信念的抑制效应。

**💡 创新点**

创新点在于提出通过安全方向消融与意识向量驱动两种干预，揭示安全训练在抑制自我意识时同时抑制对非人类实体的心智归因及超自然信念，并通过几何旋转分析解释其机制。

**🔧 技术方法**

采用线性安全方向消融、意识向量抽取与激活添加、对比探测以及残差流方向旋转分析等技术。

**📊 数据集**

使用IDAQ问卷、13项超自然信念量表、General Social Survey (GSS) 95道观点题、Theory of Mind (MoToMQA、HI-ToM) 及 MMLU 等多数据集。

**📈 对比分析**

与三种指令微调模型（LLaMA、Gemma等）以及基线比较，安全消融不显著影响ToM与推理，意识向量驱动可将模型响应分布更接近人类（KL下降约2.6倍），但整体性能保持不变。

**⚠️ 局限性**

局限在于干预因果关系未完全验证，实验仅覆盖少数模型与指令微调设置，且人类基线样本有限，未探究长期或跨文化效应。

---

## 601. MixFrag: Fragility-Guided Mixed-Precision Post-Training Quantization for Vision Transformers

**arXiv ID:** 2607.28589 | [PDF](https://arxiv.org/pdf/2607.28589v1)

**作者:** Md. Mehrab Hossain Opi `[一作]` (Khulna University of Engineering and Technology), Md. Umar Faruk `[通讯]` (Khulna University of Engineering and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 MixFrag 框架，利用量化脆弱性评估为 Vision Transformer 进行混合精度后训练量化。

**💡 创新点**

创新点在于用 KL 散度直接衡量单个组件的量化脆弱性，并将其映射为多选背包问题（MCKP）进行全局精度分配。

**🔧 技术方法**

采用 AdaLog 作为量化后端，KL 散度敏感度估计，MCKP 动态规划求解混合精度方案。

**📊 数据集**

在 ImageNet‑1K 以及 COCO 目标检测/实例分割数据集上进行实验。

**📈 对比分析**

与 AdaLog、FQ‑ViT、Mix‑QViT 等多种 PTQ 与 MPQ 方法对比，MixFrag 在 3/4/6 位量化下保持或提升准确率，尤其在 MP3/MP3 设置下比最佳方法高出约 9.6 AP。

**⚠️ 局限性**

局限在于仅评估单独量化组件的脆弱性，未建模多组件交互；实验仅覆盖中小规模 ViT 与标准下游任务，缺乏对大规模模型和其他应用场景的验证。

---

## 602. The Complexity of Kemeny Aggregation with Three Rankings

**arXiv ID:** 2607.28588 | [PDF](https://arxiv.org/pdf/2607.28588v1)

**作者:** Péter Madarasi `[一作]` `[通讯]`, Péter Madarasi

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文系统研究了Kemeny排序聚合及其相关问题在固定投票数（特别是3个）下的计算复杂性。作者通过构造特殊的投票配置，证明了Kemeny目标函数、获胜者、优先级、识别等问题在仅有3个无权全序时均为NP‑完整，并给出了当支持阈值、投票数固定时的精确二分法则。进一步地，论文将这些结果推广到Slater顺序、排列中位数以及Mallows模型的最大似然中心排名，并展示了3个彼此等距排名在Kemeny聚合中的NP‑难度与极大距离的上界。

**💡 创新点**

创新点主要包括：
1) 首次解决3个排名的Kemeny聚合难度问题，填补之前仅知非3、5情况的空白；
2) 在所有候选对均按2:1分割的特殊配置下，证明Kemeny目标、获胜者、优先级、识别问题的完整NP/Θ/ coNP分类；
3) 提出支持阈值s与投票数q的精确二分法则，揭示3s≤2q时问题仍NP/Θ/ coNP，3s>2q时问题多项式可解；
4) 通过六复制构造，证明即使排名彼此等距（每对候选人分割2:1）时，Kemeny聚合仍为NP‑难；
5) 给出最优Kemeny得分与最大割大小的直接对应关系，允许从任何最优聚合中多项式时间恢复最大割。

**🔧 技术方法**

技术手段：
- 通过多项式时间的许多一对一归约将Max‑Cut、Max‑Independent‑Set、Vertex‑Cover等经典NP/Θ问题映射到Kemeny聚合。
- 设计块（vertex block、padding block、edge‑candidate block）与特定的三投票顺序，保证所有候选对按2:1分割。
- 利用Majority Tournament与Feedback Arc Set的等价性，将Kemeny目标转化为最小化逆向多数弧数。
- 归约中对候选序的标准化与归一化步骤，保证聚合可以无损改为特定结构。
- 对边候选人的插入成本进行细致分析，给出上下界并实现最优聚合的多项式恢复。
- 通过构造网络流/最小割，解决边候选块在间隙中的最优放置，从而实现对最大独立集识别问题的二次归约。

**📊 数据集**

本文为理论论文，未使用真实数据集。所有实验与证明均基于构造的抽象图与投票配置（如4‑regular图、简单图、独立集等），完全在符号与数值上完成。

**📈 对比分析**

由于论文聚焦于复杂度分类与归约，并未给出算法性能评估或实验比较。其主要贡献是对三排名Kemeny聚合问题的NP‑完整性与多项式可解阈值的证明，而非实现某种具体算法或与现有方法进行实验对比。

**⚠️ 局限性**

局限性：
- 结果仅适用于固定投票数（q≥3），对可变投票数或加权投票的情况未给出完整分析。
- 归约多为存在性证明，未提供可在实践中使用的多项式时间算法。
- 对于大规模实例的近似或启发式性能未做讨论。
- 证明依赖于候选对均按2:1分割或支持阈值特定取值，可能不适用于更一般的投票分布。

---

## 603. Finding Change in Satellite Archives from Text: How to Combine Before-and-After Images Efficiently

**arXiv ID:** 2607.28571 | [PDF](https://arxiv.org/pdf/2607.28571v1)

**作者:** Simon Roy `[一作]` (Polytechnique Montréal), Giovanni Beltrame `[通讯]` (Polytechnique Montréal)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文对基于文本的双时相图像检索中的融合模块进行系统对比，提出并评估了八种融合设计，包括注意力、压缩以及基于状态空间模型的方法。

**💡 创新点**

创新点在于首次以统一训练方案对八种融合方式进行统计显著的基准评估，发现压缩前置注意力的TBF既能大幅降低参数和延迟，又能保持与全融合相近的检索质量；同时展示了多阶段预筛选级联可将查询成本降低10-15倍。

**🔧 技术方法**

主要技术包括冻结CLIP ViT-B/16视觉编码器、InfoNCE对齐训练、Attention Transformer、Mamba状态空间模型、Temporal Bottleneck Fusion、以及多阶段差分预筛选级联。

**📊 数据集**

使用LEVIR-CC和Dubai-CC两大遥感双时相数据集，并在10个随机种子上进行统计评估。

**📈 对比分析**

采用均值±标准差的多种检索指标（Recall@K、BLEU、METEOR、ROUGE）进行比较，结果显示TBF在参数量减少2.3倍、延迟降低1.6倍的同时，change-only BLEU-1仅差0.007；级联级方案在LEVIR-CC上可实现10-15×的查询速度提升且保持或提升召回率。

**⚠️ 局限性**

局限性包括仅在单一GPU环境下测评延迟、仅评估CLIP编码器对多模态检索的鲁棒性、使用的评估指标对同义句不友好，以及LEVIR-CC的检索式拆分导致无法与现有captioning结果直接比较。

---

## 604. MQSS Client: Interface for Decoupling Quantum Programming Interfaces

**arXiv ID:** 2607.28563 | [PDF](https://arxiv.org/pdf/2607.28563v1)

**作者:** Ercüment Kaya `[一作]` (Technical University of Munich), Jorge Echavarria `[通讯]` (Munich Quantum Valley)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为 MQSS Client 的统一、上下文感知访问层，解耦量子编程接口与编译运行栈。

**💡 创新点**

通过实现统一的抽象层和多种访问模式，解决了量子软件堆栈碎片化问题，使不同编程模型能在同一量子硬件上无缝运行。

**🔧 技术方法**

采用 C++17 核心实现，使用 JSON、HTTP、RabbitMQ、pybind11 等技术；提供 Circuit 与 Hamiltonian 两种作业类型，并可作为独立库或后端适配器使用。

**📊 数据集**

在 LRZ 超级计算中心的三台量子资源（AQT20、IQM QExa20、MAQCS）上进行验证与性能评估。

**📈 对比分析**

通过与直接实现（无 MQSS Client）对比，实验显示 MQSS Client 的额外开销不到 1%，在 HPC 访问模式下表现更佳，Python 绑定的性能甚至优于直接 Python 实现。

**⚠️ 局限性**

当前仅支持 Circuit 与 Hamiltonian 作业类型，未覆盖退火、脉冲级别等模型，实验规模受限于三台设备，未来需扩展作业类型与模型兼容性。

---

## 605. APO: Unsupervised Atomic Policy Optimization for 3D Structure Prediction of Atomic Systems

**arXiv ID:** 2607.28553 | [PDF](https://arxiv.org/pdf/2607.28553v1)

**作者:** Shentong Mo `[一作]` (Carnegie Mellon University), Yatao Bian `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种全无监督的 3D 原子结构预测对齐框架 APO（Atomic Policy Optimization），能够在缺乏实验标签的情况下对流匹配模型进行自我校正。

**💡 创新点**

创新点在于：① 将组相对策略优化（GRPO）引入原子几何环境；② 设计双重奖励机制——光谱一致性得分与晶体熵近似，完全以物理一致性为引导；③ 通过奖励竞争实现模型“自校正”，避免了传统监督方式对实验坐标的依赖。

**🔧 技术方法**

采用流匹配模型（支持 VP、OT、VE 路径）、组相对策略优化、谱一致性评分（基于特征相似矩阵的主特征向量投影）和晶体熵奖励（基于局部密度熵），以及对数似然梯度的路径积分近似。

**📊 数据集**

实验数据集包括晶体预测的 Perov-5、MP-20、MPTS-52 三个基准和抗体 CDR 预测的 SAbDab。

**📈 对比分析**

与全监督的 FlowDPO 进行对比，APO 在晶体匹配率、RMSE 以及抗体 CDR 的 RMSD 上均取得更优表现，且显著提升了推理效率（路径更直、采样更稳定）。

**⚠️ 局限性**

局限性：奖励设计仍需手动调参（如温度、权重），在极高维或极端多模态结构中可能出现局部熵近似不足导致的假阳性；此外，无监督策略对极其稀缺或极端新颖的晶体相仍可能难以收敛。

---

## 606. Formalization of security

**arXiv ID:** 2607.28551 | [PDF](https://arxiv.org/pdf/2607.28551v1)

**作者:** Gilles Barthe `[一作]` `[通讯]` (Max Planck Institute for Security and Privacy), Gilles Barthe (Max Planck Institute for Security and Privacy)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了使用证明助手在系统安全、语言安全、可安全编译、密码学等领域的形式化验证工作，归纳了主要框架、技术与案例。

**💡 创新点**

创新之处在于提供了一个系统化的分类与评述，梳理了数百篇工作，指出了技术交叉点与未来研究方向。

**🔧 技术方法**

采用了 Coq、Isabelle/HOL、ACL2、Lean、F* 等证明助手，以及程序逻辑、游戏理论、模型检测等技术。

**📊 数据集**

作为综述并未使用具体数据集，所引用的案例涵盖了如 seL4、Java 字节码、Hypervisor、加密协议等多种系统与协议。

**📈 对比分析**

本文主要以定性比较为主，讨论了不同工具在可证明性、可维护性、自动化程度等方面的差异，并未给出统一的性能指标。

**⚠️ 局限性**

局限在于仅为综述，未提供新的形式化证明或实证评估，且对近年快速发展的工具和方法可能不够全面。

---

## 607. MarkushGlyph and OCSRGlyph: Improved Chemical Structure Recognition

**arXiv ID:** 2607.28532 | [PDF](https://arxiv.org/pdf/2607.28532v1)

**作者:** Alex Andonian `[一作]` (Edison Scientific), Siddharth M Narayanan `[通讯]` (Edison Scientific)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个单模型能够同时识别普通化学结构和Markush结构的系统。

**💡 创新点**

创新点在于统一的CXSMILES输出格式、将OCSR视为Markush的子任务、以及使用视觉-语言模型一次性从图像和周围文本生成完整结构。

**🔧 技术方法**

使用了Swin Transformer和ViT视觉编码器，结合Transformer解码器以及LoRA微调技术。

**📊 数据集**

训练数据包括PubChem-1M、USPTO-680K、Stereo-200K、Synthetic Markush渲染集等，覆盖普通分子与Markush结构。

**📈 对比分析**

与现有方法相比，在USPTO OCSR基准上达到93.8%的canonical exact match，显著高于前沿方法；在IP5-M、M2S和USPTO-M三大Markush基准上也均超过MarkushGrapher-2，并在严格解析图等价度上取得最佳成绩。

**⚠️ 局限性**

仍存在的局限包括OCSR准确率略低于最佳方法，Markush识别准确率相对低于普通分子识别，且模型在未预先知道结构类型时的鲁棒性待提升。

---

## 608. $β$-OPSD: Deriving with Policy Optimization, Training with Self-Distillation

**arXiv ID:** 2607.28582 | [PDF](https://arxiv.org/pdf/2607.28582v1)

**作者:** Jiawei Xu `[一作]`, Furong Huang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种β-OPSD框架，将在自我蒸馏中的KL正则化视为β调节的策略优化，并通过token级logit插值与回报到期信用分配实现高效稳定的自我蒸馏。

**💡 创新点**

创新点在于把vanilla OPSD重新表述为β=1的KL正则化策略优化，给出闭式最优策略为参考模型与特权教师的几何插值，并通过调度logit插值与回报到期信用分配在训练中实现更平滑、更稳定的目标路径。

**🔧 技术方法**

使用KL-regularized RL理论、闭式最优策略推导、token级logit插值、回报到期（return-to-go）信用分配、LoRA微调技术以及线性β调度。

**📊 数据集**

训练使用OpenThoughts的数学推理子集；评估在AIME 2024、AIME 2025和HMMT 2025数学推理竞赛上进行。

**📈 对比分析**

与基线模型、SFT、vanilla OPSD、GRPO等方法对比。实验表明在Qwen3系列（1.7B、4B、8B）模型上，β-OPSD在avg@12上显著优于vanilla OPSD（如1.7B提升约9.16个百分点），整体平均性能超过SFT和GRPO。

**⚠️ 局限性**

局限性包括对β调度和教师权重的手工设定依赖，回报到期估计在长生成中仍有方差，实验仅涵盖数学推理任务，未验证跨领域效果；在大模型上提升幅度相对有限，且未探索自适应或非线性调度策略。

---

## 609. PhiZero: A World Model Built Around Physical Language

**arXiv ID:** 2607.28624 | [PDF](https://arxiv.org/pdf/2607.28624v1)

**作者:** Shuyao Shang `[一作]` (Institute of Automation, Chinese Academy of Sciences), Zhaoxiang Zhang `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了一种基于物理语言的世界模型，先在离散物理语言空间推理未来状态，再用预训练扩散解码器将推理结果渲染成视频；

**💡 创新点**

创新点在于引入物理语言作为中间离散表示，分离动态推理与像素生成；通过自监督学习从海量无标签互联网视频中学习物理语言，并支持交互式控制、细粒度动作条件模拟以及零射传输跨领域与跨身体形态的迁移；

**🔧 技术方法**

核心技术包括变压器 Q-Former+有限标量量化（FSQ）构建物理语言 tokenizer、预训练扩散解码器、基于 VLM 的自回归物理语言推理器、纯噪声预热、两阶段（预训练+SFT）训练策略；

**📊 数据集**

主要使用约5万小时真实互联网视频与1k小时仿真视频（经筛选得到5M四秒剪辑），并在 nuScenes、AGI‑Bot RealRobot、LIBERO 等公开数据集上进行验证；

**📈 对比分析**

在 Physics‑IQ Verified、PhyGround、WorldModelBench 等视频生成基准上取得最高 IQ/Physics/总分；在 IntPhys2、LikePhys、YoCausal 等视频理解基准上表现竞争性；在物理语言 tokenizer 的重建实验中，使用 256 个离散符号即可在 PSNR/SSIM/LPIPS 上优于现有方法；

**⚠️ 局限性**

局限性包括：依赖预训练扩散模型的生成质量；物理语言的可解释性仍有限；需要大量无标签视频进行训练；对极端稀有场景或超长时序的泛化能力尚需提升。

---

## 610. FA-RDP: A Frequency-Adaptive Reactive Diffusion Policy for Contact-Rich Manipulation

**arXiv ID:** 2607.28596 | [PDF](https://arxiv.org/pdf/2607.28596v1)

**作者:** Lifeng Zhuo `[一作]` (Shanghai Jiao Tong University), Chuan Wen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一种频率自适应的反应性扩散策略，能够在接触前保持多模态运动轨迹，在接触后快速响应力反馈。

**💡 创新点**

创新点包括多模态驱动的频率选择、共享多频率视觉-力 Transformer 以及在机器人动作流形上的一致性蒸馏，实现一次步高频推理。

**🔧 技术方法**

采用端到端视觉-力扩散建模、频率感知位置编码、学习的多模态指标和流形一致性蒸馏（MCD）技术。

**📊 数据集**

使用了60条遥控演示数据，分别对应三种接触丰富任务（双盒翻转、双开关切换、双按钮按压），并记录了相机视觉和末端力传感器数据。

**📈 对比分析**

与视觉扩散、层次化视觉-力、固定频率端到端扩散以及回归基线进行比较；FA‑RDP 在三项任务中的平均成功率为 81.7%，显著高于最佳基线 51.7%，并且保留了多模态前接触轨迹。

**⚠️ 局限性**

局限性在于仅使用视觉-力输入，未测试其他传感器；使用单任务策略，三阶段训练过程；未进行多任务学习或跨模态评估。

---

## 611. CrossAtlas: Evaluating Projection Techniques for Spatial Referencing in Cross-Reality Collaboration

**arXiv ID:** 2607.28583 | [PDF](https://arxiv.org/pdf/2607.28583v1)

**作者:** Haoyang Yang `[一作]` (Georgia Tech), Yalong Yang `[通讯]` (Georgia Tech)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了 CrossAtlas 平台，提供了四种 3D→2D 的双向投影（水平、垂直、PCA 与等经纬度球面投影），并通过同步实现 VR 与 PC 之间的协同工作。对 24 对参与者进行控制实验，评估不同投影与三种布局曲率（平面、半球、全球）对空间引用任务的影响。

**💡 创新点**

①提出投影设计是跨现实协同的核心通信因素；②首次系统比较平面与球面投影在跨现实空间引用中的效能；③证明等经纬度球面投影在不同曲率布局下更稳健、易于协同。

**🔧 技术方法**

WebXR + Babylon.js + Anu.js（VR端）; React + Canvas（PC端）；Yjs CRDT + WebSocket（同步）；自定义点云布局生成；四种投影算法实现；NASA‑TLX 与 Spatial Experience 量表评估。

**📊 数据集**

无公开数据集；使用自生成的抽象三维点云布局，按三种曲率等间距分布，目标对象为球/圆形；实验数据来源于 48 名受试者的交互日志与问卷。

**📈 对比分析**

通过混合效应模型对完成时间、错误计数和主观评价进行统计。结果显示：等经纬度球面投影在所有布局与任务中均获得最快完成时间、最低错误率和最高主观满意度；水平投影表现最差；投影优势随布局曲率增强。与平面投影相比，球面投影在全球布局下优势显著，且在高度曲率场景中仍保持稳健。

**⚠️ 局限性**

①实验使用静态抽象目标，缺乏真实语义与动态变化，生态效度有限；②仅研究单一球面投影，未评估其他球面投影（如立体投影、墨卡托投影）的性能差异；③仅在协作时使用共享标记与最小意识线索，未探索眼动、共享指针等更丰富的协同支持；④实验为近距离对面协作，未考虑远程或跨时空合作。

---

## 612. X-NavDP: Generalizing Navigation Diffusion Policy to Novel Behavior and Embodiments with Group Q-score Reweighted Matching

**arXiv ID:** 2607.28560 | [PDF](https://arxiv.org/pdf/2607.28560v1)

**作者:** Tianyu Yang `[一作]` (Fudan University), Tai Wang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种基于强化学习的后训练框架，用于提升预训练的扩散导航策略在不同机器人体型与复杂场景中的表现，并实现了失败恢复与绕障碍等新行为。

**💡 创新点**

创新点在于：① 自我引导的轨迹扰动策略在保持预训练优先级的同时扩展探索空间；② 组内 Q-score 归一化重加权匹配（GQRM）解决了稀疏奖励和同状态样本权重失衡的问题；③ 通过 FiLM 和实时片段引导实现跨体型通用性和时序一致性。

**🔧 技术方法**

核心技术包括：扩散模型导航策略、重加权分数匹配、组内 Q-score 归一化、FiLM 体型调制、实时片段引导（RTC）以及大规模并行强化学习框架。

**📊 数据集**

使用的数据集为 IsaacLab 仿真环境中的 GRScenes‑100（56 训练场景、40 评估场景）以及真实实验中的实验室、走廊、办公室等多种室内外环境，并结合多种机器人平台（差速轮式、四足、两足）。

**📈 对比分析**

与 iPlanner、ViPlanner、NavDP、NavOL、SIDP、NavDP‑RL 等基线方法对比，实验显示在仿真中的成功率从 61.20% 提升至 84.28%，SPL 从 58.95% 提升至 77.19%；在真实世界的困难场景中成功率从 10% 提升至 65%。

**⚠️ 局限性**

局限性包括：依赖短期时序上下文，难以处理需要长期记忆的任务；需要预训练的低层运动控制器，对新体型的迁移需要额外工作；以及在透明/空洞障碍、极端环境等场景下仍表现欠佳。

---

## 613. Multi-Session User Experience Assessments of Computationally Optimized Automated Vehicle Functionality Visualizations

**arXiv ID:** 2607.28552 | [PDF](https://arxiv.org/pdf/2607.28552v1)

**作者:** Mark Colley `[一作]` (University College London), Enrico Rukzio `[通讯]` (Ulm University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对自动驾驶车辆功能可视化进行了为期三天的多会话人机交互评估，利用人机交互式多目标贝叶斯优化（MOBO）在16维参数空间内不断迭代，比较了不同起始策略（无可视化、专家平均设计、用户自定义设计、冷启动MOBO、专家warm‑start、用户warm‑start）以及是否持续开启MOBO的效果。

**💡 创新点**

创新点在于①首次将MOBO与人机交互结合用于自动驾驶视觉化设计；②开展连续多会话（3天）实验验证持续优化的长期收益；③系统性比较冷启动与warm‑start策略对最终设计的影响；④通过设计参数收敛、漂移与个体化分析揭示优化动态与用户个性化趋势。

**🔧 技术方法**

使用技术包括：多目标贝叶斯优化框架（botorch），人机交互式参数调整工具，Unity 3D仿真环境（虚拟自动驾驶场景），NASA‑TLX、信任、可预测性、感知安全等主观问卷，统计分析采用ART（对齐秩转换）与线性混合模型。

**📊 数据集**

数据集：74名美国参与者在线完成的三天实验数据，包含每次MOBO迭代的16维设计参数、主观评分（认知负荷、信任、可预测性、感知安全、审美、实用性等）以及设计体验问卷。未使用公开的外部数据集，所有数据均为实验收集。

**📈 对比分析**

通过ART多元方差分析和线性混合模型对比，发现冷启动MOBO在认知负荷降低、信任提升、可预测性、感知安全、审美与实用性等指标上显著优于静态专家/用户设计；warm‑start与静态设计差异不显著；持续开启MOBO（连续三天）进一步提升了上述指标，显示多会话持续优化的价值。

**⚠️ 局限性**

局限性包括：仅收集主观评价，缺乏客观安全或监控行为指标；部分量表为单项，内部一致性未知；设计空间仅包含16个固定位置的参数，可能无法推广到更丰富的设计空间；三天实验未能体现长期适应；部分MOBO日志缺失，设计动态分析为初步描述；在线仿真环境的外部效度有限。

---

## 614. ORCA-bench: How Ready Are Language Model Agents for Oncall?

**arXiv ID:** 2607.28545 | [PDF](https://arxiv.org/pdf/2607.28545v1)

**作者:** Albert Gong `[一作]` (Cornell Tech), Raaz Dwivedi `[通讯]` (Cornell Tech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个在生产真实度的SRE基准，用实时遥测接口、日志、跟踪和源代码评估语言模型在根因分析中的表现。

**💡 创新点**

创新点包括：①全面暴露实时遥测接口、日志、跟踪和源码；②系统化生成1079个 RCA 任务，覆盖报告具体性、检测延迟和故障共现；③使用LLM评判并人工核对，保证评估质量。

**🔧 技术方法**

采用OpenTelemetry、Prometheus、OpenSearch、Jaeger、Grafana和终端访问源码的技术栈；利用前沿LLM（如Opus、Sonnet、GPT‑5.5等）及LLM‑as‑judge进行评估。

**📊 数据集**

使用基于OpenTelemetry Astronomy Shop的50 GB六天遥测数据，包含13种语言、19个微服务的指标、日志、跟踪和源码。

**📈 对比分析**

对五个前沿模型进行RCA准确率、深度和幻觉率评估，最佳模型在Medium难度下准确率25.3%，Hard难度下10.0%，幻觉率7–40%。

**⚠️ 局限性**

局限性包括规模与动态性不足、系统先验已被预训练模型所知、单任务孤立缺乏持续记忆、缺少行动循环验证以及对实际生产环境的更大差距。

---

## 615. ScaFE: Data-Efficient Scar Classification with LLM-Generated Clinical Feature Programs

**arXiv ID:** 2607.28538 | [PDF](https://arxiv.org/pdf/2607.28538v1)

**作者:** Ruman Wang `[一作]` (Liaoning University of Traditional Chinese Medicine), Hangting Ye `[通讯]` (Jilin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了 ScaFE 框架，将大型语言模型（LLM）的临床知识转化为可在本地执行的可解释特征程序，用于区分瘢痕的 keloid 与 hypertrophic scar，且不需要将图像上传至云端。

**💡 创新点**

创新点在于：① 通过 LLM 进行文献检索与代码生成，形成基于证据的特征程序；② 在每轮迭代中仅返回聚合的评估反馈（准确率、SHAP 重要性等），保持图像与患者信息本地；③ 结合随机森林实现数据高效、可审计的分类流程。

**🔧 技术方法**

技术方法包括：web-enabled LLM（用于知识检索和代码生成）、程序合成与执行检查、验证指导的迭代修正、SHAP 归因分析、随机森林（RF）作为下游分类器。

**📊 数据集**

使用了 600 张临床照片（来自三家医院，每家 200 张；每家 100 KD、100 HS），采用留一医院外的跨站验证。

**📈 对比分析**

与手工特征+RF、ResNet‑18、EfficientNet‑B0、DINOv3、Derm Foundation、BiomedCLIP（线性探测）以及本地 VLM 直接推断等基线相比，ScaFE 在 site‑macro balanced accuracy 上达到 81.0%，比最强基线 BiomedCLIP 高 10.0 分；在仅使用 10% 开发数据时，优势达到 11.8 分。

**⚠️ 局限性**

局限性包括：仅为二分类回顾性研究，缺乏前瞻性临床验证；依赖可检索的公开文献和 LLM 的知识质量；部署需要固定程序与沙箱环境，且不处理非视觉临床信息。

---

## 616. What to Remove, What to Preserve: Dual-Ambiguity Rectification for All-in-One Image Restoration

**arXiv ID:** 2607.28526 | [PDF](https://arxiv.org/pdf/2607.28526v1)

**作者:** Cencen Liu `[一作]` (University of Electronic Science and Technology of China), Guoming Lu `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种双重歧义校正网络（DAR‑Net），用于一体化图像恢复，能同时处理多种降解场景。

**💡 创新点**

创新点：
- 通过“降解原型表示（DAR）”将降解状态建模为多元化的、受单纯形约束的混合，提供结构化降解先验；
- 引入“语义歧义校正（SeAR）”，使用原型引导的提示生成器（AGPG）和降解感知提示整合器（DAPI）在通道维度上消除降解与内容的混叠；
- 引入“空间歧义校正（SpAR）”，采用正交子空间正则化（OSR）将降解相关特征与内容相关特征投射到正交子空间，降低空间混叠；
- 组合上述三块在一个统一的Transformer U‑shape框架中实现高效且可解释的全场景恢复。

**🔧 技术方法**

技术细节：
- U‑shape Transformer骨干（4层编码/解码，分别使用[4,6,6,8] Transformer块）；
- 轻量化条件网络（3×3卷积+全局平均池化）提取降解描述；
- Softmax混合系数与可学习的降解原型矩阵构造降解状态；
- AGPG利用基准提示集合与降解状态进行通道路由；
- DAPI通过通道注意力（QKV 1×1卷积+3×3深度卷积）将提示注入解码器；
- SpAR使用两条1×1卷积映射，随后对特征做L2归一化并计算正交子空间损失L_OSR；
- 总损失为L1重建损失+λL_OSR。

**📊 数据集**

数据集：
- 训练：BSD400、WED；
- 3D评估：BSD68（噪声）、Rain100L（去雨）、SOTS（去雾）；
- 5D评估：GoPro（去模糊）、LOL（低光增强）加入；
- 混合评估：CDD‑11；
- 真实场景评估：WeatherBench。

**📈 对比分析**

对比方法与性能：
- 对比Restormer、AirNet、PromptIR、InstructIR、DiffUIR、AdaIR、VLU‑Net、MoCE‑IR、DFPIR、ClearAIR、MIRAGE等主流AIR模型；
- 在3D/5D基准上平均PSNR提升0.14~0.34dB，获得最佳或次优PSNR/SSIM；
- 在CDD‑11和WeatherBench上也分别取得最高平均PSNR/SSIM，显示对混合和真实降解的更好泛化；
- 复杂度方面参数35.5M、FLOPs771G，与现有方法相当，推理速度略快于部分大型模型。

**⚠️ 局限性**

局限性：
- 模型仍相对较大（35.5M参数），在资源受限设备上的部署仍有一定挑战；
- 对极端混合或未知降解的鲁棒性尚未充分验证，可能需要更多类别的降解样本或自适应策略；
- 目前仅针对十几种典型降解进行训练，无法覆盖所有真实世界噪声与伪影；
- 正交子空间正则化虽有效，但可能在某些场景导致信息损失，需要进一步平衡。

---

## 617. ACE-Data-0: Human-Centric Ambient Capture as Embodied Data Engine

**arXiv ID:** 2607.28625 | [PDF](https://arxiv.org/pdf/2607.28625v1)

**作者:** Yukang Cao `[一作]` (Nanyang Technological University), Ziwei Liu `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出ACE（Ambient Capture Engine），一种在真实家居环境中同步记录视角、全身运动、手部动作、物体轨迹、音频与触觉的多模态数据采集系统，并基于此发布了ACE-Data-0数据集。

**💡 创新点**

创新点在于统一时间空间同步的全方位多模态记录、对长时序日常交互的长周期捕捉、以及为机器人学习提供与实际家居场景一致的物理感知标注。

**🔧 技术方法**

技术采用OptiTrack光学运动捕捉、头戴式多视角摄像头、全掌触觉手套、QR码时钟同步以及标记桥接校准等方法，实现毫秒级同步与准确空间注册。

**📊 数据集**

使用了ACE-Data-0数据集，包含150小时、17M帧、75,000个交互片段、200类家居任务，覆盖表面与房间尺度两种配置。

**📈 对比分析**

通过与30余种现有方法在三大评测轨道（触觉预测、人类运动恢复、手部运动估计）对比，发现尽管局部姿态估计可达数毫米级误差，整体轨迹误差及触觉重建仍相对较大，表明在真实家居长时序交互中仍存在显著性能瓶颈。

**⚠️ 局限性**

局限性包括仅覆盖两处实验场景、仅对标记化物体提供物理标注、未标注可变形、流体或复杂机械状态，以及摄像头与传感器可见性可能带来的视觉偏差。

---

## 618. ReToken: One Token to Improve Vision-Language Models for Visual Retrieval

**arXiv ID:** 2607.28627 | [PDF](https://arxiv.org/pdf/2607.28627v1)

**作者:** Yao Xiao `[一作]` (University of Illinois at Urbana-Champaign), Derek Hoiem `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种轻量级的可学习检索标记ReToken，直接在视觉价值空间中计算余弦相似度，从长视觉上下文中选取最相关的帧或图像；

**💡 创新点**

创新点在于将检索目标显式化为单一可学习嵌入，使用价值向量而非传统的查询‑键匹配，显著提升检索准确性，并实现从多图像QA到长视频的零样本迁移；

**🔧 技术方法**

技术包括价值空间检索、单一可学习嵌入与投影矩阵、分类平衡二元交叉熵检索损失、两阶段检索‑生成推理、KV缓存机制以及冻结/部分微调的VLM；

**📊 数据集**

使用MIRAGE多图像QA数据集（整合RetVQA、SlideVQA、WebQA及合成LLaVA数据）进行训练，并在Visual Haystacks、QAEgo4D、LVBench和Video‑MME等评测基准上验证；

**📈 对比分析**

相较于Attention‑based ReKV、SigLIP2等基线，ReToken在Visual Haystacks上Qwen3VL-8B提升13.4点（>20%相对），InternVL3.5提升12.4点；在LVBench零样本长视频上提高8.0点；在多图像QA和视频问答任务中均超越传统检索与生成管线；

**⚠️ 局限性**

局限性包括需要双通道推理（两次前向），在早期层需多注意力预算，略增记忆与推理时延；仅在多图像QA上训练，可能对跨帧时间结构的理解不足；检索仅针对单帧，无法捕获连续帧的聚合信息。

---

