# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-06-08 | 今日论文总数: 503

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. DxPTA: An Architecture Design Space Exploration with Optical Dataflow-guided Strategy for HW/SW Co-Design of Photonic Transformer Accelerators

**arXiv ID:** 2606.06515 | [PDF](https://arxiv.org/pdf/2606.06515v1)

**作者:** Rachmad Vidya Wicaksana Putra `[一作]` (New York University Abu Dhabi), Muhammad Shafique `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 11484 | [OpenAlex ID](https://openalex.org/A5005190949)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出一种基于光学数据流引导的设计空间探索方法（DxPTA），实现光学变压器加速器（PTA）的硬件/软件协同设计，以满足面积、功率、能耗和延迟等多重约束。

**💡 创新点**

创新点在于：①利用光学数据流特性识别关键架构参数；②通过实验评估参数显著性并量化其对面积/功率的影响；③设计了约束感知搜索算法，在保持高效性的同时显著缩短搜索时间。

**🔧 技术方法**

主要技术包括：光学互连与动态操作的光学张量核心（DPTC）结构、基于光学干涉的点积单元（DDot）、参数显著性分析、约束感知的离散搜索算法，并在PyTorch与Lumerical仿真框架上实现。

**📊 数据集**

使用的模型数据集为Vision Transformer（DeiT-T/S/B）和大型语言模型（BERT-B/L），并在这些模型的推理任务上评估加速器性能。

**📈 对比分析**

比较方法：将DxPTA生成的架构与现有LT-Base/Large设计以及全枚举搜索得到的架构进行对比，评估面积、功率、能耗、延迟和能延迟乘积。实验表明DxPTA在满足相同约束的前提下，能耗/延迟降低高达76.9%/82.7%，并且搜索时间比全枚举快15.2倍。

**⚠️ 局限性**

局限性包括：①仅在仿真环境下验证，缺乏硅光子硬件实验；②搜索空间仍依赖手工设定的参数上限；③对不同类型的Transformer模型或更复杂的光学组件支持有限；④光学器件的非理想效应（例如波长漂移、温度变化）未在模型中充分考虑。

---

## 2. A Study of Parallel Continuous Local Search

**arXiv ID:** 2606.06656 | [PDF](https://arxiv.org/pdf/2606.06656v1)

**作者:** Cody J Christopher `[一作]` (Australian National University), Charles Gretton `[通讯]` (Australian National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文研究了在 GPU 加速环境下，利用连续局部搜索（CLS）解决布尔可满足性问题，尤其是对对称伪布尔（PB）约束的连续松弛与优化。

**💡 创新点**

创新点包括：① 将对称 PB 约束的 Walsh‑Fourier 系数给出闭式解，提升表示效率；② 证明冗余约束会导致梯度失衡抑制收敛；③ 发现 CLS 在部分赋值完成后能高效求解随机 3‑SAT；④ 对收敛行为进行分布式分析，为超参数调优提供依据。

**🔧 技术方法**

使用的技术包括 Walsh‑Fourier 变换、连续松弛与可微目标函数、投影梯度下降（PGD）、GPU 上的离散傅里叶变换（DFT）向量化实现，以及 Accelerated Fourier SAT（A-FSAT）求解器。

**📊 数据集**

实验数据集涵盖 Ramsey 着色（|V|≤16、|C|=3,4）、Costas 数列（n≤20）、数独（9×9）以及 SAT Competition 2007 的硬随机 3‑SAT（450–650 变量）。

**📈 对比分析**

CLS 与传统 CNF 编码（如 Dagster）做对比；在小规模实例上 CLS 能在几秒内完成搜索，且在已固定 15–20% 变量的 3‑SAT 中快速完成剩余部分；然而 CLS 单独无法解决全规模 3‑SAT，且在高维度时收敛速度受梯度振荡影响。总体性能受限于 GPU 精度与梯度波动，但在并行度高时显示出显著加速。

**⚠️ 局限性**

局限性包括：① 冗余 PB 约束会导致梯度失衡，抑制搜索；② CLS 仅对对称 PB 约束有效，非对称或复杂约束难以处理；③ 需要手工设置梯度加权或超参数，缺乏自适应机制；④ 受浮点精度与 GPU 内存限制影响，理论上可达步数上界远高于实测。

---

## 3. Degrees of Freedom of Over-the-Air Computation over a MIMO Gaussian Network with Two Transmitters and Two Receivers

**arXiv ID:** 2606.06770 | [PDF](https://arxiv.org/pdf/2606.06770v1)

**作者:** Yong Dong `[一作]` (University of California Irvine), Syed A. Jafar `[通讯]` (University of California Irvine)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了两发射器、两接收器的MIMO高斯网络中，利用空时信道叠加特性进行求和运算的上行空中计算（AirComp）的自由度（ACDoF）上限与下限，并给出了在SISO和通用MIMO情形下的精确ACDoF表达式。除此之外，作者还提出了一个线性迭代极小化（AO）算法，用于在有限信噪比（SNR）下实现近似最优的AirComp，并与传统的TDMA/多址MAC方案进行了数值对比。

**💡 创新点**

主要创新点包括：
1) 统一推导了两发射两接收AirComp的ACDoF，得到了最优的min{M1,M2,N1,N2,(1/3)max{M1+M2,N1+N2}}这一闭式结果；
2) 在SISO情形下，对时间不变与时变通道分别给出完整证明，揭示了复数信道比率对DoF的决定作用；
3) 通过“子网络打包”方法将大MIMO网络分解为若干（1,1;1,1）、（1,1;2,1）和（1,1;1,2）三类子网络，完成了DoF的构造与证明；
4) 对有限SNR场景提出了基于AO的线性预编码/解码设计，并在模拟实验中验证其相对传统TDMA/多址方案的显著优势。

**🔧 技术方法**

核心技术主要包括：
- 信息理论中的MSE‑equivocation 边界和逆矩阵逆向推导；
- 空间分配与子网络打包（子网络组合与矩阵秩分析）；
- 线性预编码/解码的极小化（迭代优化、LMMSE解码、QP求解）；
- 高斯随机信道模型与自由度分析（泛型通道假设、随机矩阵秩）。

**📊 数据集**

实验使用的是随机生成的MIMO信道矩阵（服从连续分布），并在多个M和L值下（如M=9、10，L=4、5）进行仿真。没有使用公开的数据集，而是通过仿真得到的随机信道和随机信息符号。

**📈 对比分析**

作者将AO算法与传统的MAC‑TDMA对接方案进行比较。结果表明：
- 在中低SNR下AO方案的归一化MSE明显低于TDMA；
- 随着SNR升高，AO与TDMA性能差距缩小；
- 当任务数 L 超过理论ACDoF限制时，TDMA出现误差平台，而AO仍能保持可接受的误差水平；
- 在M=10、SNR=30dB时，AO在L=1.5×ACDoF时仍可实现低误差，表明其对任务负载的鲁棒性。

**⚠️ 局限性**

主要局限包括：
- 结果仅在通用（generic）信道或满足特定复数比率条件下成立，特殊/极端信道配置（如退化、同相位）可能不满足；
- 迭代AO算法只保证局部最优，计算复杂度较高，难以保证全局最优；
- 需要完美的CSIT/CSIR，实际系统中信道估计误差会影响性能；
- 在极低SNR或极高MIMO维度下，理论与实际间可能存在偏差。

---

## 4. S23DR 2026 Winning Solution

**arXiv ID:** 2606.06695 | [PDF](https://arxiv.org/pdf/2606.06695v1)

**作者:** Jan Skvrna `[一作]` (Czech Technical University), Lukas Neumann `[通讯]` (Czech Technical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种两阶段条件集合生成器，利用稀疏 SfM、深度图和语义分割重建高精度 3D 建筑线框。

**💡 创新点**

创新点在于将场景 Perceiver 编码与流匹配 DiT 结合，对固定 64 个顶点槽进行时间条件去噪，并通过全局粗略生成+凸包裁剪细化+多样本一致性提升鲁棒性。

**🔧 技术方法**

采用 Perceiver‑style 场景编码器、DiT 变形 Transformer、流匹配训练、Euler ODE 步进、全局与局部自注意力、Hungarian 匹配、Focal 与 BCE 损失以及多样本聚合等技术。

**📊 数据集**

使用 HoHo 数据集（约 22k 训练、200 验证场景），每个场景包含稀疏 COLMAP 点云、尺度匹配的深度图、屋顶 Gestalt 语义分割以及对应的 3D 线框标签。

**📈 对比分析**

在 S23DR 2026 私有排行榜上，方法获得 HSS=0.654（榜首），顶点 F1=0.791，明显优于学习型基线 0.474 与手工基线 0.391。

**⚠️ 局限性**

局限性包括对稀疏点云与语义分割的高度依赖，训练和推理需要大规模 GPU 资源；极端噪声或深度缺失视角仍可能导致误差；虽然多样本聚合缓解了槽初始化敏感性，但整体复杂度仍高。

---

## 5. Dependencies and Dataflow in Seed-Filter-Extend Pipelines

**arXiv ID:** 2606.06811 | [PDF](https://arxiv.org/pdf/2606.06811v1)

**作者:** Shiv Sundram `[一作]` `[通讯]` (Stanford University), Shiv Sundram (Stanford University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出将猜测执行技术引入 LASTZ，重构其 Y‑drop 计算为批量化并行流程，并在此基础上采用更重的间隔种子以适配高相同性基因组；

**💡 创新点**

创新点在于通过延迟包含与掩蔽检查制造数据独立性，实现 CPU‑GPU 混合的批处理管线，同时将短读映射器 SNAP 的重种子启发式迁移至全基因组比对；

**🔧 技术方法**

使用了 GPU 级的流水线式 2D Y‑drop 计算核、GPU warp 乱序重排、动态批大小调优、以及重种子（如 14/22）过滤策略；

**📊 数据集**

实验数据集涵盖人–大鼠、鼠–鼠等高相同性种间对齐（如 mm10 × rn6、hg19 × hg38）以及人–老鼠等低相同性对齐；

**📈 对比分析**

与原始 LASTZ 对比，最佳批大小 K≈48 时在 32 核 CPU + GPU 上实现约 2.5 倍加速，单线程加速 8 倍，GPU 负载率由 2.2% 提升至 20%；

**⚠️ 局限性**

主要限制包括 warp 分歧导致的 GPU 低占用、批大小受限导致的“影子工作”开销、以及缺乏针对 Y‑drop 不确定长度的动态负载均衡策略。

---

## 6. N-Player Binary Games with Unidirectional Dependencies: Cycle Robustness and Induced Indifference

**arXiv ID:** 2606.06625 | [PDF](https://arxiv.org/pdf/2606.06625v1)

**作者:** Jose Maria Sanchez-Saez `[一作]` (University of Malaga), Francisco Criado-Aldeanueva `[通讯]` (University of Malaga)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了在N人二进制策略、单向局部依赖的循环网络游戏中，解析纳什均衡的结构与存在性，并给出了线性时间（O(N)）的闭式求解方法。

**💡 创新点**

提出了“确定性传播”与“极限平衡”理论，阐明了强制性边界激励、支配关系、奇偶性（Parity Condition）与结构性/诱导性无差异的角色；同时将此理论直接映射到循环供应链的库存同步问题，形成可逆设计的框架。

**🔧 技术方法**

基于游戏理论的线性化分析、代数手段推导边际激励函数、极限条件判定、回路映射（Return Map）与分支推理，最终实现了O(N)的递归传播算法；无须数值迭代或图谱计算。

**📊 数据集**

本文主要为理论与示例性案例分析，未使用实际数据集；供应链示例采用人工构造的四节点策略矩阵（如市场稳定器、集成伙伴等），以演示不同结构（Dominance、Parity、Indifference）下的均衡解。

**📈 对比分析**

与传统的PPAD‑完整通用网络游戏求解（如Lemke‑Howson、梯度下降）相比，本方法不涉及复杂数值迭代，时间复杂度线性；理论上可在大规模循环网络中即时得到所有均衡（纯/混合）并判断存在性，性能优于已知通用求解器。

**⚠️ 局限性**

局限性包括：仅适用于二进制策略、单向循环拓扑；不涵盖双向或更一般的网络结构；不处理动态学习或随机性；对非强制性边界（结构性/边界无差异）时需额外分支分析，可能导致算法实现复杂；实验验证仅在人工示例中完成，缺乏真实大规模供应链数据验证。

---

## 7. RECAP: Regression Evaluation for Continual Adaptation of Prompts

**arXiv ID:** 2606.06698 | [PDF](https://arxiv.org/pdf/2606.06698v1)

**作者:** Harsh Deshpande `[一作]` (Capital One), William Campbell `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了RECAP基准，用于评估在生产环境下主动适应动态约束的提示调优方法；

**💡 创新点**

创新点在于将约束级持续学习框架引入到无反馈的主动适应协议，并设计了多种操作（增删改）与衡量指标；

**🔧 技术方法**

采用了多种提示调优技术（ICL、动态备忘单、ACE、GEPA、MIPROv2）以及自回放与评估的自玩策略；

**📊 数据集**

使用RECAST-30K数据（来自Tulu 3 Persona IF），对约束类型进行时间序列化生成评测流；

**📈 对比分析**

与不做任何适应的基线（仅在提示中附加当前约束）进行对比，实验在4种LLM（Llama-3.1 8B/70B、GPT‑OSS 20B/120B）和3种调度上进行，结果显示无论采用何种方法均未显著提升平均满足率，甚至在部分模型上表现更差，且消耗更高延迟；

**⚠️ 局限性**

限制在于仅探讨提示级适应（无模型微调），并未与具备实时反馈的被动/反应式方法做对比，未来需研究更高效、无回退的主动适应技术。

---

## 8. Accelerated Fourier SAT (AFSAT): Fully Realising a GPU-based Symmetric Pseudo-Boolean SAT Solver

**arXiv ID:** 2606.06641 | [PDF](https://arxiv.org/pdf/2606.06641v1)

**作者:** Cody J Christopher `[一作]`, Charles Gretton `[通讯]` (Australian National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

开发了一款名为Accelerated Fourier SAT（A‑FSAT）的GPU加速CLS求解器，实现了对异构伪布尔约束的支持，并通过JAX、XLA实现批量并行搜索、数值稳定性提升以及多GPU近线性缩放；

**💡 创新点**

创新点包括：①首次在CLS框架内实现异构约束支持；②针对DFT计算实现定制化矩阵，解决浮点精度导致的数值不稳定；③支持部分变量赋值作为输入，便于与分解式求解器集成；④通过数组分片实现多GPU分布式执行，保持近线性吞吐量；⑤通过GPU内存优化和批量调优实现显著的内存占用与吞吐量提升；

**🔧 技术方法**

使用技术包括：Continuous Local Search（CLS）与Walsh‑Fourier变换、向量化的离散傅里叶变换（DFT）实现、JAX + XLA编译与自动微分、投影梯度下降（PGD）等搜索算法、GPU warp并行、数组分片（sharding）实现多GPU并行、定制的DFT矩阵与延迟除法、批量搜索与自动线搜索。

**📊 数据集**

评测数据集涵盖：随机Cardinality约束、Parity‑Learning（XOR）问题、随机硬3SAT实例；实验在Gadi超算的Volta GPU节点（四个Tesla V100）上进行。

**📈 对比分析**

方法对比：将A‑FSAT与原始proof‑of‑concept实现（以及其GPU化改进）进行对比，使用相同或更小的批量、相同的参数配置。结果显示：在最坏情况下性能相当，平均比原实现快，GPU内存占用大幅下降；吞吐量在单GPU和多GPU下呈近线性提升；PAR‑2分数与累计求解时间均优于原实现。

**⚠️ 局限性**

局限性：①求解器为不完整求解器，无法证明不可满足；②受IEEE‑754 64位浮点精度限制，单条约束长度上限约为50；③长约束会导致数值不稳定、梯度消失/爆炸；④需要手动拆分长约束以适应硬件；⑤对二阶/高阶方法支持有限；⑥依赖JAX生态，缺乏完整的可证明最优性保证。

---

## 9. MedSIGHT: Towards Grounded Visual Comprehension in Medical Large Vision-Language Models

**arXiv ID:** 2606.06760 | [PDF](https://arxiv.org/pdf/2606.06760v1)

**作者:** Aofei Chang `[一作]` (Pennsylvania State University), Cao Xiao `[通讯]` (GE Healthcare)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出统一框架 MedSIGHT，结合 Region Perceiver 与模态感知代码表实现医学视觉语言模型的细粒度感知与像素级语义表达，支持诊断推理与分割。

**💡 创新点**

① Region Perceiver：双向跨注意力实现区域嵌入的细粒度提取；② 模态感知 Region Codebook：将连续区域嵌入量化为可在 LLM 词表中生成的离散代码；③ 进步式训练策略：先对视觉、文本投影器进行对齐，再联合指令微调；④ DiagSeg 基准：从诊断推理到像素分割的端到端评测。

**🔧 技术方法**

基于 Qwen3‑8B LLM、Unimed‑CLIP 视觉编码器、跨注意力、向量量化、投影器、双向交互、进步式多阶段训练。

**📊 数据集**

BiomedParse（分割/检测）、PubMedVision（图文对齐）、60K 自构建的图像‑文本‑代码对齐数据、72K 指令微调样本、DiagSeg（诊断+分割）、VQA‑RAD、SLAKE、PathVQA、MMMU 等。

**📈 对比分析**

与多种 Med‑LVLM（HuatuoGPT‑Vision、LLaVA‑Med 等）和统一模型（OMG‑LLaVA、MedPLIB、LISA 等）在 VQA、DiagSeg 等任务上对比。MedSIGHT 在视觉问答平均得分 62.3，诊断 Recall 58.9，分割 Dice 69.9，均超过所有基线，尤其在多模态、全流程评测上表现最优。

**⚠️ 局限性**

对未知模态或区域的泛化受限；代码表对新领域的适配需要进一步扩展；在某些 OOD 模态上分割性能略低；依赖大规模预训练与多阶段训练，计算成本高；仍可能出现误分割或诊断偏差，不能替代专业医师。

---

## 10. Learning All-Terrain Locomotion for a Planetary Rover with Actively Articulated Suspension

**arXiv ID:** 2606.06790 | [PDF](https://arxiv.org/pdf/2606.06790v1)

**作者:** Arthur Bouton `[一作]` (California Institute of Technology), Hari Nayar `[通讯]` (University of Maryland)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实验了一款四轮行星车ERNEST，配备两自由度主动万向悬架，并通过强化学习控制器实现了在岩石、坡道、波纹、Bickler陷阱等多种地形的自主通行。

**💡 创新点**

创新点在于将主动万向悬架与单一神经网络控制器结合，利用策略合并技术实现多地形无显式切换，并在DARTS高保真仿真中引入Bekker–Wong软土力学与域随机化实现零射程转移。

**🔧 技术方法**

使用技术包括TD3强化学习、DARTS动力学仿真引擎、Bekker–Wong软土模型、立体视觉+力矩计+IMU感知堆栈，以及自定义的ε-greedy探索策略。

**📊 数据集**

数据集主要由仿真生成的多地形场景（岩石、阶梯、波纹、坡道）和真实地面试验记录构成；训练时采用域随机化生成约10^6条经验，涵盖不同地形特征与随机参数。

**📈 对比分析**

与被动悬架六轮系统比较时，RL控制器在干沙20°坡道上将能耗降低37%，在湿沙中避免陷落；在岩石、波纹、Bickler陷阱等多场景下，通行成功率与人工规划相当，且表现出优于传统被动悬架的通过能力。

**⚠️ 局限性**

局限性包括仿真软土模型未考虑土壤变形与侵蚀导致的仿真‑现实差距、未涵盖极端障碍物导致的泛化不足，以及神经网络控制器缺乏形式化安全验证。

---

## 11. RPC-GS: Gaussian Splatting with native RPC Rendering for Satellite Imagery

**arXiv ID:** 2606.06690 | [PDF](https://arxiv.org/pdf/2606.06690v1)

**作者:** Valentin Wagner `[一作]` (Fraunhofer Institute of Optronics, System Technologies and Image Exploitation), Michael Arens `[通讯]` (Fraunhofer Institute of Optronics, System Technologies and Image Exploitation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了首个面向卫星影像的原生RPC Gaussian Splatting 渲染器 RPC-GS，并完成了完整的实验与评测。

**💡 创新点**

创新点在于：①直接在地理坐标变换链中嵌入RPC模型，避免传统视角/仿射近似；②推导出数值稳健的 Jacobian 协方差投影；③引入基于RPC的光线深度定义，实现前后排序的 alpha 合成。

**🔧 技术方法**

采用了 RPC 投影、ECEF‑Geodetic‑ENU 坐标变换、Jacobians、Ray‑based 深度、PyTorch+CUDA 结合的渲染管线，并利用结构光/稠密匹配的 Gaussian 优化。

**📊 数据集**

使用 IEEE GRSS DFC2019 (WorldView‑3) 与 IARPA2016 (Buenos Aires) 两大卫星图像基准集，包含多视角真彩/多光谱影像。

**📈 对比分析**

在统一框架下对比了 RPC、Perspective 与 Affine 三种相机模型，RPC‑GS 在所有场景中均实现了最低的高度 MAE（相较于 Perspective 29.6%/9.9%，相较于 Affine 63.8%/37.9% 的降误差），PSNR 虽略低，但保持竞争力。

**⚠️ 局限性**

局限性包括：1) Jacobian 一阶近似在非线性投影上可能产生误差；2) 现有实现主要在 Python 中完成投影，计算开销大，CUDA 整合后可进一步加速。

---

## 12. Natural Language Access Control (NLAC): From Help Desk Requests to Structured Policies

**arXiv ID:** 2606.06726 | [PDF](https://arxiv.org/pdf/2606.06726v1)

**作者:** Jonas Wessner `[一作]` (Ulm University), Frank Kargl `[通讯]` (Ulm University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了NLAC架构，利用LLM将自然语言请求转换为访问控制策略，并通过规则引擎和分布式审批实现安全、可扩展的访问控制管理；同时发布了NLACBench基准数据集。

**💡 创新点**

创新点在于：①将LLM用于意图翻译并与传统规则配置分离；②引入分布式权限归属模型，减少人为干预；③提出语义子图构造技术，显著降低LLM上下文大小，提升大规模网络的可扩展性；④构建可扩展、真实感的基准数据集。

**🔧 技术方法**

技术包括：大型语言模型（GPT‑4.1、Qwen3.5、LLaMA3.3 等）进行意图翻译；RAG+向量检索与子图闭包构造；XACML/JSON 结构化意图表示；分布式审批和规则引擎；实验评估框架。

**📊 数据集**

使用了自研的 NLACBench 数据集，包含 450 条自然语言请求（PCR）与对应意图、以及 1,000+ 用户/设备/服务的知识库，可按 5-50 个网络段扩展。

**📈 对比分析**

通过在不同规模网络上计算准确率来比较模型；小网络（5 段）最高可达 96.9%；大网络（50 段）若不做子图过滤准确率骤降至 <20%；采用子图过滤后，LLaMA3.3 的准确率提升至 98.7%，并将输入 token 降至 1/75，显著降低 GPU/内存需求与 API 成本。

**⚠️ 局限性**

局限性：数据集主要基于大学网络，可能不适用于企业网络；未评估对抗性与错误请求的鲁棒性；模型性能仍受知识库质量影响；人类审核效率与实际部署效果尚未量化；多模型协同效果仍待进一步研究。

---

## 13. Tensor Algebraic Property Skeletons: Amplifying Property-Based Testing for AI Compilers

**arXiv ID:** 2606.06747 | [PDF](https://arxiv.org/pdf/2606.06747v1)

**作者:** Yuxin Qiu `[一作]` (UC Riverside), Miryung Kim `[通讯]` (UCLA)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一种基于大型语言模型（GPT‑5.5）的代理式属性化测试框架，用来为深度学习编译器（以 TVM 为例）生成可执行的属性测试。

**💡 创新点**

创新点在于将张量代数关系抽象为可重用的属性骨架（property skeleton），通过受控实例化、验证与反馈循环，让 LLM 只在已知属性和操作符上生成有效测试，而不是盲目生成。

**🔧 技术方法**

核心技术包括 GPT‑5.5 生成测试场景、属性骨架设计、三阶段验证（模式、适用性、运行安全）以及基于执行结果的反馈调优。

**📊 数据集**

使用 TVM 的 212 个运算符与 20 条属性骨架共生成 4,579 条 PBT，基准对比使用相同算子集的 LLM‑only PBT 与无属性的模糊测试。

**📈 对比分析**

与 LLM‑only PBT 和传统模糊测试相比，本框架将冗余率降低 49%、无效测试降至 0%，同时在 semantic 错误（50%）和数值不稳定（25%）检测上大幅提升，代码覆盖虽略低，但更具意义。

**⚠️ 局限性**

局限性包括需手工维护属性骨架、对其它编译器的适配度未知、对非常规或新颖的张量运算可能缺乏相应属性，且整体性能受 LLM 推理时间与验证开销限制。

---

## 14. Leveraging Soft Distributions of SSL-Derived Discrete Speech Tokens for Downstream Inference

**arXiv ID:** 2606.06806 | [PDF](https://arxiv.org/pdf/2606.06806v1)

**作者:** Kentaro Onda `[一作]` (University of Tokyo), Nobuaki Minematsu `[通讯]` (University of Tokyo)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在训练时使用硬离散化的自监督语音令牌，推理时通过后验概率实现软分配，从而在保持压缩效率的同时提升模型性能。

**💡 创新点**

创新点是仅在推理阶段引入软分配，使得模型既能享受离散化带来的效率，又能在推理时捕捉更多连续特征信息。

**🔧 技术方法**

主要技术包括HuBERT/WavLM自监督特征提取、k-means聚类离散化、后验概率软max温度τ调节以及基于这些令牌的ASR与TTS模型训练。

**📊 数据集**

实验使用LibriSpeech-100h训练集，评估在LibriSpeech、TED-LIUM v2、CHiME4、ERJ（非母语）以及LJSpeech/TIMIT的重建与声学转换任务。

**📈 对比分析**

与连续特征、全软分配以及硬硬方案对比，软推理在ASR与TTS上均获得更低的WER/MCD等指标，尤其在跨域和非母语语音上表现突出。

**⚠️ 局限性**

局限性包括需手动调节温度τ、在多层聚合时对跨域性能影响较大，以及未整合去重、BPE等进一步压缩技术。

---

## 15. FP8 is All You Need (Part 1): Debunking Hardware FP64 as the HPC Holy Grail

**arXiv ID:** 2606.06510 | [PDF](https://arxiv.org/pdf/2606.06510v1)

**作者:** Satoshi Matsuoka `[一作]` `[通讯]` (RIKEN Center for Computational Science), Satoshi Matsuoka (RIKEN Center for Computational Science)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出并验证了利用 FP8 张量核心结合 Ozaki Scheme II 通过低精度模运算重建 FP64 结果的方案，并给出了统一的 Tensor–Memory Equilibrium (TME) 性能模型。

**💡 创新点**

创新点在于：①将 CRT‑based Ozaki II 与 FP8 适配，实现线性扩展的 64‑精度模拟；②提出寄存器级融合策略，使 β→1 并消除数据搬移成本；③构建 TME 模型，量化内存‑计算平衡下的可行性与加速边界。

**🔧 技术方法**

技术上采用了：FP8（E4M3）张量核心、整数模运算、Garner 余数恢复、寄存器级模分解与重组、低精度乘加（MMA）、以及 FP32+Kahan 补偿和 Kulisch 固定点重构。

**📊 数据集**

实验验证以经典 HPC 内核为基准：稀疏矩阵向量乘（SpMV）、结构化 7‑点算子、批量 GEMV、全密集 GEMM 以及伴随的 FFT 作为代表性工作负载；未使用公开数据集，而是通过合成矩阵与微基准实现性能评估。

**📈 对比分析**

与 NVIDIA 官方数据及 H100 基准相比，B300 上的 Ozaki II 在内存‑受限内核可实现 1.2–9.2× 的加速，且在所有计算敏感内核上均能恢复或超过 H100 的性能；在 Dense GEMM 上最高可达 380× 的理论吞吐量（≈500 TFLOPS）。

**⚠️ 局限性**

局限性包括：①需要对每个核进行复杂的寄存器级融合实现，工程成本不容忽视；②β 取值受寄存器压力与内存调度影响，部分稀疏或小尺寸核可能无法达到 β≈1；③实验仅覆盖基准核，真实应用的误差与性能分布仍待进一步验证。

---

## 16. The Sharp Phase Transition of Tyler's M-Estimator for Robust Subspace Recovery

**arXiv ID:** 2606.06782 | [PDF](https://arxiv.org/pdf/2606.06782v1)

**作者:** Gilad Lerman `[一作]` (University of Minnesota), Teng Zhang `[通讯]` (University of Central Florida)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在鲁棒子空间恢复问题中，研究者通过严格理论分析证明了Tyler的M-估计器在维度缩放信噪比为1时具有尖锐的相位转变，能够在更弱的稳定性条件下精确恢复潜在子空间；

**💡 创新点**

主要创新在于：①将TME迭代等价为新的Majorization‑Minimization框架；②构造细粒度的目标函数分解，精准区分内点与外点的贡献；③提出比传统一般位置假设更弱的稳定性条件，并在临界点证明精确恢复；

**🔧 技术方法**

采用了MM优化框架、矩阵分解与微扰理论、谱分解、Weyl不等式及极限分析等数学工具；

**📊 数据集**

本文主要以理论分析为主，使用合成的无噪声内点–外点数据集进行推导，并未给出具体实验数据集；

**📈 对比分析**

与已有的随机/确定性子空间搜索算法比较，证明在DS‑SNR≥1且满足稳定性假设时，TME能在多项式时间内实现精确恢复；在临界点DS‑SNR=1时提供了先前未得到的收敛结果；

**⚠️ 局限性**

局限性包括：①仅对无噪声（或极小噪声）情形给出结果；②未给出临界点下的收敛速率；③在实际含噪数据或多子空间场景中的适用性仍待进一步研究。

---

## 17. Semantic-Structural Alignment for Generative Pictorial Charts

**arXiv ID:** 2606.06498 | [PDF](https://arxiv.org/pdf/2606.06498v1)

**作者:** Zhida Sun `[一作]` (Shenzhen University), Hui Huang `[通讯]` (Shenzhen University)

**通讯引用:** 16195 | [OpenAlex ID](https://openalex.org/A5087787304)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于双条件（文本提示+上下文图像）生成的可视化图表框架，利用多模态扩散 Transformer 通过结构与语义 DIFT 双重对齐，实现既保持数据编码又生成可表达的物体化图表。

**💡 创新点**

创新点在于：① 引入结构 DIFT 与语义 DIFT 两个并行的特征级对齐机制，分别保证生成图表的空间布局与视觉语义一致；② 在多模态扩散 Transformer 内部实现这种双重对齐，突破了传统控制方法对结构或语义的单一约束；③ 通过 LoRA 微调与结构驱动增强合成数据，解决了少量样本条件下的训练瓶颈。

**🔧 技术方法**

使用技术包括：多模态扩散 Transformer（MM‑DiT）+ LoRA 微调；Diffusion Feature Correspondence (DIFT) 用于建立图像间的密集匹配；结构 DIFT 对齐生成查询，语义 DIFT 对齐键值并使用 SLERP 进行平滑插值；基于 CLIP、DINO、SSIM、PSNR、LPIPS 等指标进行评估。

**📊 数据集**

数据集：从公开资源手工逆向构造 120 张高质量图表并生成 583 对 (源图表+提示) 训练样本；通过结构驱动增强生成大量合成对；另外自行构建 160 张图表‑对象对 benchmark，用于定量评估。

**📈 对比分析**

比较方法：与 ControlNet、CIA、Ctrl‑X、ChartSpark、SDEdit、Stable Flow、FLUX.1 Kontext 等多种基线进行对比；在 SSIM、PSNR、DINO、LPIPS、CLIP 等指标上，本文方法在结构保持上仅次于 ControlNet 与 Stable Flow，且在语义合成上与纯生成方法相当或更优；用户研究显示 67.8% 的偏好率。

**⚠️ 局限性**

局限性：① 结构对齐可能抑制语义细节，导致某些目标物体纹理不够自然；② 对参考图像的视角/布局差异敏感，若不匹配易产生伪影；③ 依赖 LoRA 微调，缺乏零训练开销的通用性；④ 目前仅针对单一图表元素的转换，背景生成与全景场景支持不足。

---

## 18. Applying Deep Learning for cockpit segmentation in the context of mixed reality

**arXiv ID:** 2606.06520 | [PDF](https://arxiv.org/pdf/2606.06520v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 19. The Geography of Algorithmic Judgment: LLM Intermediaries, Place Identity, and Racial Steering in Housing Search

**arXiv ID:** 2606.06694 | [PDF](https://arxiv.org/pdf/2606.06694v1)

**作者:** Hana Samad `[一作]` (Responsible AI Lab), Michael Akinwumi `[通讯]` (National Fair Housing Alliance)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对七款LLM在四大美都市中进行住房推荐的偏差审计，评估种族引导行为；

**💡 创新点**

阐明LLM的偏差是由身份、偏好与空间语义交互产生的“情境性引导”而非固有属性；

**🔧 技术方法**

使用配对测试、递进提示、Spearman相关、空间机会指数、点模式分析；

**📊 数据集**

包含四市（芝加哥、休斯顿、纽约、洛杉矶）各2800个实例，共20,160条，基于美国人口普查与房源数据；

**📈 对比分析**

对比七款模型的推荐偏差，结果显示在不同提示与城市下偏差显著差异，P2提示有时可削弱或逆转偏差；

**⚠️ 局限性**

限制包括使用过时模型版本、邮编级别不够细致、种族标签直接使用、仅有限网络访问、未深入分析解释词汇。

---

## 20. Signal-Driven Observation for Long-Horizon Web Agents

**arXiv ID:** 2606.06708 | [PDF](https://arxiv.org/pdf/2606.06708v1)

**作者:** Shubham Gaur `[一作]` (University of California, Santa Cruz), Ian Lane `[通讯]` (University of California, Santa Cruz)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并阐述了 Web 代理在长周期任务中因频繁读取完整 DOM 而导致的上下文退化与目标漂移问题，并给出了基于信号驱动的观察压缩框架 SDO，阐明观察频率与行动频率可解耦；

**💡 创新点**

将递归语言模型的“查询”思想迁移到动态 Web 交互中，首次将观察压缩视为代理设计的核心原则，并提出四类轻量信号触发子调用，解决观察过度摄取这一根本失败模式；

**🔧 技术方法**

基于 Playwright 的浏览器交互、子调用 LLM（sub_RLM）读取完整 DOM 并返回压缩任务相关信息、根 LLM 进行规划、信号检测器监控 URL、ARIA 元素、动作失败与外部事件；

**📊 数据集**

本文未在任何公开数据集上进行实验，仅在 WebArena、WorkArena、BrowserGym 等现有基准框架中进行理论分析与案例演示；

**📈 对比分析**

未给出实验结果或性能指标，仅通过案例说明 SDO 在保持上下文清晰、避免循环与目标漂移方面的潜在优势，缺乏定量对比；

**⚠️ 局限性**

局限性包括：1) 信号覆盖不完整，可能忽略无结构变化的语义更新；2) 任务相关过滤的准确性取决于子模型，易导致信息丢失或压缩过度；3) 对真实网站的外部事件依赖于规则而非学习，缺乏适应性；4) 观察历史管理与长周期任务中的累积压缩仍需解决；5) 未进行实验验证。

---

## 21. Elmes*: Automated Construction of Fine-Grained Evaluation Rubrics for Large Language Models in Long-Tail Educational Scenarios

**arXiv ID:** 2606.06546 | [PDF](https://arxiv.org/pdf/2606.06546v1)

**作者:** Tao Liu `[一作]` (East China Normal University), Hao Hao `[通讯]` (East China Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ELMES 框架，用多代理引擎和自进化 Rubric 生成实现对教育场景下 LLM 教学能力的端到端评估。

**💡 创新点**

自动化生成和迭代细粒度、情境化评分标准，结合可声明 YAML 与 LangGraph 实现教师-学生-评审交互，解决了传统人工设计 Rubric 的瓶颈。

**🔧 技术方法**

基于 LangGraph 的 DAG 交互、YAML 声明式配置、GPT‑4/Claude 等 LLM 作为教师/学生/评审，多轮评审与自我校准、Prompt 进化与评分约束。

**📊 数据集**

构建 Edu‑330 330 场景、11 学科、3 年级、10 任务类型的规模化基准，并对 4 个专家手写“金标”情境进行评估。

**📈 对比分析**

将多种主流 LLM（GPT‑5、Claude‑Opus‑4.5 等）与教育垂直模型 InnoSpark 在 Edu‑330 上进行分维度打分，显示顶级模型在创造力和价值融入方面仍落后，InnoSpark 在人类专家评估中获得最高平均分；LLM 评审与人类专家的排名一致但方差更低。

**⚠️ 局限性**

仅在离线文本模拟环境评估，未覆盖多模态交互，缺乏真实学习者结果验证，且自评偏好等评审模型固有偏差仍需改进。

---

## 22. WAV: Multi-Resolution Block Residual Routing for Deep Decoder-Only Transformers

**arXiv ID:** 2606.06564 | [PDF](https://arxiv.org/pdf/2606.06564v1)

**作者:** Kehan Wang `[一作]` (Chongqing University), Kehan Wang `[通讯]` (Chongqing University)

**通讯引用:** 107890 | [OpenAlex ID](https://openalex.org/A5100440745)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了多分辨率残差路由方法WAV v1，扩展了Block Attention Residuals，加入了相位与分割两个零和方向细节基，提升深层Transformer的残差信息利用

**💡 创新点**

创新点在于为每个残差块提供低频块摘要和两种结构化方向细节（注意力-MLP相位差与块内早晚差异），并通过深度软最大化路由实现内容自适应选择

**🔧 技术方法**

使用PreNorm RMSNorm、因果自注意力、SwiGLU MLP、深度软最大化混合器、零和细节基与RMS匹配等技术实现轻量级多分辨率路由

**📊 数据集**

在字符级GPT解码器模型上使用TinyStories和Text8两个数据集进行实验

**📈 对比分析**

将WAV v1与Standard、Block LayerScale、ReZero、LayerScale及Block AttnRes进行对比；在12层时略逊于Block AttnRes，24层表现接近；48层时在两数据集上均取得最佳验证损失，TinyStories相对Block AttnRes降低0.0222，Text8降低0.0057，表明深度越大越能显著提升性能

**⚠️ 局限性**

局限性包括：仅在小规模字符级模型上验证；缺乏多随机种子误差评估；未测试更大规模Token级语言模型；仅采用两手工细节基，未探索可学习或软正交基；计算开销测量不完整

---

## 23. UnpredictaBench: A Benchmark for Evaluating Distributional Randomness in LLMs

**arXiv ID:** 2606.06622 | [PDF](https://arxiv.org/pdf/2606.06622v1)

**作者:** Amirhossein Abaskohi `[一作]` (University of British Columbia), Peter West `[通讯]` (University of British Columbia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 UnpredictaBench 基准，用于评估 LLM 在生成与真实统计分布一致的样本的能力。

**💡 创新点**

创新点在于构建了 448 个多样化任务、提出了 KS@N 这一基于 Kolmogorov–Smirnov 检验的分布一致性度量，并系统比较了不同规模和训练策略的 LLM。

**🔧 技术方法**

使用 Kolmogorov–Smirnov 检验、Jensen–Shannon Divergence、Wasserstein 距离等统计方法，对模型产生的样本与真分布进行比较；同时利用 Python 代码、自然语言、混合任务等多种提示模板。

**📊 数据集**

构建了包含 40 种经典概率分布、50 个人工真实场景及 398 个自动生成任务的 448 个实例数据集，并公开发布在 Hugging Face 上。

**📈 对比分析**

对比了多款开放与专有 LLM，最高 KS@100 分数仅 32.64%，大部分模型低于 20%，显示分布采样能力仍远未成熟；与 Random Machine 基准相比仍有显著差距。

**⚠️ 局限性**

局限性包括：提示全部为英文、主要由 GPT-5.4 生成，可能带来表述偏差；代码任务仅限 Python，难以推广至其他语言；基准仅为评估工具，未用于训练，过度优化可能导致过拟合。

---

## 24. AutoPipelineAI: Context-Aware CI/CD Pipeline Generation from Natural Language

**arXiv ID:** 2606.06662 | [PDF](https://arxiv.org/pdf/2606.06662v1)

**作者:** Youssef Mohamed Aboelfotoh `[一作]` (Cairo University), Seif Gamal Abdelmonem `[通讯]` (Cairo University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 AutoPipelineAI，一个基于 LLM 的系统，可根据自然语言描述生成 CI/CD pipeline 配置。

**💡 创新点**

首次结合仓库感知分析、LLM 生成、验证回路以及多平台支持，实现从自然语言到完整、可执行流水线的端到端自动生成。

**🔧 技术方法**

使用大型语言模型（GPT‑4.1、GPT‑4o、MiniMax‑M2‑1、Qwen3‑Coder‑Next‑UD‑79B）、多代理架构、YAML 校验、自动迭代反思等技术。

**📊 数据集**

对两个公开项目（Python 的 Peewee 和 Go 的 Hugo）以及各自的原始 YAML 进行评估，使用四个 LLM 模型生成流水线。

**📈 对比分析**

采用 AI 评估框架（结构准确率、完整性）和功能需求评测（功能通过率、错误类型统计），结果显示在简单项目上可达 0.83–0.87 的平均准确率，但在复杂项目上通过率仅 10–35%，并存在多种错误。

**⚠️ 局限性**

仅评估了两项目和四模型，功能评测需人工，AI 评估受评审模型偏见，缺乏大规模、多语言、跨平台验证，且生成流水线仍需人工校验。

---

## 25. PromptPrint: Behavioral Biometrics Through Natural Language Prompting in LLMs

**arXiv ID:** 2606.06755 | [PDF](https://arxiv.org/pdf/2606.06755v1)

**作者:** Shaiv Patel `[一作]` (Johns Hopkins University), Vishal Patel `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大语言模型提示（prompt）作为一种软生物特征，利用超过二万条真实提示进行特征抽取与身份识别实验，证明提示能有效识别用户。

**💡 创新点**

创新点在于首次系统化验证提示表面词汇与句法特征可形成稳健身份指纹，并揭示“词汇稳定假设”和“风格悖论”。

**🔧 技术方法**

使用的技术包括TF‑IDF+LogisticRegression、SBERT+MLP、手工风格特征、特征级与分数级融合、5折交叉验证以及EER、ROC‑AUC等生物识别评估指标。

**📊 数据集**

使用的数据集为WildChat‑1M公开语料，从中抽取首轮提示，筛选后得到20,680条提示、1,034名用户。

**📈 对比分析**

方法与传统作者归因基线（如CharNgram+SVM）进行比较，TF‑IDF单独模型EER≈0.381、Top‑1≈64%，组合模型Top‑1≈64.2%，但对语义改写攻击易失效。

**⚠️ 局限性**

局限性包括仅在无标注IP噪声的公开数据上验证，缺乏严格的用户标签；对语义重写高度敏感；未实现端到端的度量学习或实时部署。

---

## 26. Multilingual Multi-Speaker Unit Vocoders: A Systematic Analysis of Discrete Speech Representations

**arXiv ID:** 2606.06740 | [PDF](https://arxiv.org/pdf/2606.06740v1)

**作者:** Naman Kothari `[一作]` (National Institute of Technology), S Umesh `[通讯]` (Indian Institute of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了将离散语音单元映射为波形的过程，针对多语种多说话人场景下的 BigVGAN 单位声码器进行系统评估。

**💡 创新点**

创新点在于对聚类大小、说话人条件和语言监督对可懂度与说话人保持的相互影响进行量化分析，并揭示小聚类时跨语言单元共享导致的歧义。

**🔧 技术方法**

采用 BigVGAN 声码器架构、ECAPA‑TDNN 说话人嵌入、语言嵌入与辅助 LID 损失，以及 Data2Vec‑AQC 自监督表示进行离散单位提取。

**📊 数据集**

实验基于 IndicVoices‑R 语料库，涵盖四种印度语言（Bengali、Hindi、Tamil、Telugu），共计约 400k 步训练。

**📈 对比分析**

通过 WER、说话人相似度和单元级指标对比四种条件设置，结果显示聚类数越大可懂度越好；加入说话人嵌入显著提升说话人保持；语言监督在小聚类时能进一步降低 WER。

**⚠️ 局限性**

限制在于仅评估四种语言，未扩展到全部 22 种语言，也未探索音高、韵律等更多可控属性，且跨语言共享导致的误差在大聚类规模下仍存在。

---

## 27. Geometric Second-Order Feature Correlation Learning for Self-Supervised Speech Emotion Recognition

**arXiv ID:** 2606.06550 | [PDF](https://arxiv.org/pdf/2606.06550v1)

**作者:** Shuanglin Li `[一作]` (Xiangjiang Laboratory), Siyang Song `[通讯]` (University of Exeter)

**通讯引用:** 1249 | [OpenAlex ID](https://openalex.org/A5053061988)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种Second-Order Correlation（SOC）层，用来聚合自监督学习得到的语音情绪特征，并将协方差描述符映射到切空间以保留几何信息。

**💡 创新点**

创新点在于将高维SSL特征的协方差建模为对称正定（SPD）流形上的张量，并通过Log‑Euclidean映射保持流形几何，同时采用子空间投影来降低维度和数值不稳定性。

**🔧 技术方法**

使用了协方差估计、迹归一化、Log‑Euclidean映射、半向量化、线性投影、固定SSL骨干（Wav2Vec 2.0、HuBERT、WavLM）以及MLP分类器。

**📊 数据集**

实验数据集为ESD（双语英文/中文）和RAVDESS（音频子集）。

**📈 对比分析**

在与GAP、ASP、FA等第一阶聚合基线对比中，SOC在WA、UA、F1上均提升约4–5%，在所有骨干与两数据集上均取得最优或次优结果。

**⚠️ 局限性**

局限性包括对特征维度的子空间调参需求、需进行特征值分解导致一定的计算成本，以及在极少样本或跨域情境下的泛化性仍需进一步验证。

---

## 28. RigPAPR: Rig-Based Animation of Static Neural Point Clouds from a Fixed-Viewpoint Video

**arXiv ID:** 2606.06685 | [PDF](https://arxiv.org/pdf/2606.06685v1)

**作者:** Shichong Peng `[一作]` (Simon Fraser University), Ke Li `[通讯]` (Simon Fraser University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `aaccfe5c-6b26-4208-b23c-35331481e142` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本论文提出了RigPAPR方法，利用无形状参数的PAPR点云自动实现骨骼绑定，并通过直接线性混合皮肤化（LBS）在单视角视频上进行动画化；

**💡 创新点**

创新点在于采用基于插值的PAPR表示，消除高斯膨胀点的单独协方差导致的关节边界伪影，配合两阶段深度+追踪监督的优化，提升关节变形的逼真度；

**🔧 技术方法**

使用的技术包括PAPR渲染器、直接LBS、深度预测器（Depth Anything 3）、2D追踪器（CoTracker 3）、软正则化（如APO）以及基于姿势的Mlp校正；

**📊 数据集**

实验使用了来自Objaverse的五个合成模型（T‑Rex、Simpson、狐狸、狼、蜘蛛）以及两个人工收集的实景目标（机器人、雕像）；

**📈 对比分析**

与Mesh‑based Puppet（Puppeteer）和Gaussian‑splat‑based Mani‑GS基线对比，RigPAPR在训练视角下PSNR最高，且在未见视角下PSNR提高约3 dB，且在关节边界处渲染更平滑；

**⚠️ 局限性**

局限性包括对单视角深度与追踪先验的依赖，若这些先验失效则会导致姿态误差；自动骨骼生成不适用于未见形态；绑定帧中被遮挡的区域在后期展开时会出现缺失或颜色异常。

---

## 29. Are you sure? A Comprehensive and Comprehensible Survey of Uncertainty Quantification in Symbolic Regression

**arXiv ID:** 2606.06567 | [PDF](https://arxiv.org/pdf/2606.06567v1)

**作者:** Julia Reuter `[一作]` (Federal University of ABC), Fabricio Olivetti de Franca `[通讯]` (Federal University of ABC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述并系统阐述了符号回归（SR）中的不确定性量化（UQ）方法，提出了UQ在SR中的重要性与研究现状，构建了基于置信区间、贝叶斯推断、模型选择等三大研究方向的框架；

**💡 创新点**

创新点在于首次将UQ概念与SR结合，系统梳理了置信区间、贝叶斯后验、贝叶斯因子、MDL等多种UQ技术在SR中的应用与差异，并为后续研究提供了明确的分类与参考；

**🔧 技术方法**

采用了文献检索、关键词筛选与案例分析等技术，对近二十篇相关论文进行方法学归纳、比较与评述；

**📊 数据集**

本文并未使用具体实验数据集，而是通过对已有研究中的数据集（如公开回归数据集、人工合成数据等）的引用与讨论进行综述；

**📈 对比分析**

对比方法主要是对比不同UQ技术在不同论文中的实现与评价指标，指出贝叶斯MCMC/SMC在实现不确定性量化时效果优于传统置信区间，但整体研究仍处于起步阶段；

**⚠️ 局限性**

局限性在于仅覆盖了少量现有研究，缺乏统一的实验验证与标准化比较，且对非贝叶斯UQ方法的讨论不够充分，导致综述范围相对有限。

---

## 30. IRAF: Interference-Resilient Adaptive Fusion for Noise-Robust End-to-End Full-Duplex Spoken Dialogue Systems

**arXiv ID:** 2606.06559 | [PDF](https://arxiv.org/pdf/2606.06559v1)

**作者:** Tao Zhong `[一作]` (Chinese University of Hong Kong), Xunying Liu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种轻量化、流式兼容的自适应融合模块 IR​AF，用于在端到端全双工语音对话模型中动态抑制干扰说话者对用户语音特征的污染。

**💡 创新点**

创新点在于通过目标说话者与用户音频嵌入预测帧级可靠性门控，实现对干扰的实时抑制，而不需要额外的分离或后处理步骤。

**🔧 技术方法**

使用流式语音编码器、TinyLlama 1.1B 语言模型、NanoCodec 声码器、ECAPA‑TDNN 说话人嵌入、Causal Transformer 的单层门控模块以及端到端多通道下一个词预测训练目标。

**📊 数据集**

主要数据集包括单轮 MS‑MARCO（文本问答转语音）和多轮 InstructS2S‑200K（语音对语音对话），并在 MUSAN 语音与噪声上进行干扰模拟。

**📈 对比分析**

在 CleanBase、NoisyAug 以及加入 IR​AF 的三种配置下对比，IR​AF 在干扰环境下提升 BLEU（+1–4 级）、sBERT（+0.1–0.2）、响应成功率（+2–21%）并显著降低响应与停止延迟，证明在全双工交互与语义质量上均优于基线。

**⚠️ 局限性**

限制包括：门控仅基于单帧目标说话人信息，可能对极短或多说话人场景失效；模型对严重噪声或多重干扰仍有性能下降；以及在极高实时性要求的实际部署中需要进一步评估延迟与算力开销。

---

## 31. Synthetic Benchmarks Overstate Forward-Forward Scaling: Real-Data Limits of Layer-Local Training

**arXiv ID:** 2606.06539 | [PDF](https://arxiv.org/pdf/2606.06539v1)

**作者:** Yucheng Chen `[一作]` `[通讯]` (Amplimit), Yucheng Chen (Amplimit)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于 Forward‑Forward 的层局部学习算法 DTG‑FF，结合动态温度、解耦归一化和多层融合等技术，实现在多种真实数据集上的最高 FF‑family 性能；

**💡 创新点**

通过动态温度调节良好度、在良好度路径上解耦归一化、以及多层融合的方式，显著提升层局部学习的有效性，并形成一个可与 BP 直接对照的基准工具；

**🔧 技术方法**

DTG‑FF 采用层局部的 FF 损失、可学习的温度参数、随机投影分类器、LayerNorm 传播路径、全局平均池化融合等技术，训练时每层使用独立的 AdamW 及余弦调度；

**📊 数据集**

在 CIFAR‑10/100、Tiny ImageNet、ImageNet‑100（224×224）、MedMNIST、Fashion‑MNIST、STL‑10 等九个真实分类数据集上进行实验；

**📈 对比分析**

将 DTG‑FF 与同构 BP‑DeepSup 基准在相同网络、同一训练脚本下进行对照，结果显示在 CIFAR‑10/100 上仍落后 2.4–5.9% ，在 ImageNet‑100 上仅 49.4%（BP 通常 >75%），并通过合成教师‑学生实验验证 K‑维度扩展对 FF 的误导性；

**⚠️ 局限性**

局部学习的准确率仍显著低于 BP，尤其在更大图像尺寸和更高类别数时；合成实验与真实数据对齐不足，难以归因到单一改进；系统层面虽然理论上可实现 O(1) 激活内存，但在 8 GB GPU 上与 BP+梯度累积无明显优势，表明结构性优势未转化为实用性能。

---

## 32. Quantum Hierarchical Locally Recoverable Codes

**arXiv ID:** 2606.06736 | [PDF](https://arxiv.org/pdf/2606.06736v1)

**作者:** Venkatesan Guruswami `[一作]` (University of California, Berkeley), Pranav Trivedi `[通讯]` (University of California, Berkeley)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并证明了一类新的量子Tamo–Barg码与分层局部可恢复码，给出了其维数、局部可恢复性和距离下界；

**💡 创新点**

创新点在于将经典 (r,δ)-局部可恢复码与其对偶构造的 CSS 码相结合，利用根号多项式与 Schur 多项式等代数工具给出距离证明，并提供分层局部性的结构化构造；

**🔧 技术方法**

采用了根号（cyclotomic）结果、Schwartz–Zippel 计数、Jacobi–Trudi 与 Bialternant 公式、非消失定理以及凸函数分析等数学技术；

**📊 数据集**

此工作为理论构造，未使用任何具体数据集；

**📈 对比分析**

通过与已有的量子 Tamo–Barg 码及经典 HLRC 的 Singleton‑like 极限进行对比，证明新码在距离和码率上满足 q−1/2(…) 的下界，性能优于现有简单构造；

**⚠️ 局限性**

局限性包括需排除有限多特征，且 δ=2 时需额外假设；距离下界可能不是最优，构造复杂度高，随机行需足够大域。

---

## 33. MSAIC-Net: A Multi-Scale Attention and Imbalance-Aware Contrastive Network for ECG-Based Myocardial Substrate Abnormality Detection

**arXiv ID:** 2606.06718 | [PDF](https://arxiv.org/pdf/2606.06718v1)

**作者:** Canyu Lei `[一作]` (University of Virginia), Jianxin Xie `[通讯]` (University of Virginia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出了一种多尺度注意力增强卷积网络（MSAIC‑Net），用于从多导联心电图中检测心肌基质异常（心肌瘢痕和心肌梗死）。

**💡 创新点**

创新点包括：①使用多尺度空洞卷积分支捕捉不同时间尺度的ECG特征；②加入通道注意力动态加权各导联特征；③提出焦点加权监督对比学习（Focal Supervised Contrastive Loss）来缓解类别不平衡并提升特征分离；④利用置换重要性（Permutation Importance）实现后训练可解释性，量化各导联贡献。

**🔧 技术方法**

技术方法主要是1D‑CNN、空洞卷积、通道注意力、焦点二元交叉熵、焦点监督对比学习、置换重要性分析。

**📊 数据集**

实验使用两大数据集：UVA内部低样本心肌瘢痕数据集（7056条记录）和公开的PTB‑XL（12428条记录）用于心肌梗死检测。

**📈 对比分析**

通过与传统CNN、带通道注意力的多尺度CNN、加入监督对比学习、以及现有的Transformer、ResNet、LSTM和CNN‑LSTM等模型比较，MSAIC‑Net在UVA上AUROC 0.9366、AUPRC 0.9757、F1 0.9169；在PTB‑XL上AUROC 0.9678、AUPRC 0.9343、F1 0.8576，显著优于对比方法。

**⚠️ 局限性**

局限性主要在于：①仅验证了两种疾病（心肌瘢痕与心肌梗死）和两套数据集，缺乏跨疾病与跨机构的进一步泛化评估；②对置换重要性分析未与临床专家解读深度结合，解释性仍需进一步验证；③模型复杂度相对较高，可能在资源受限的临床环境中部署受限。

---

## 34. SafeGene: Reusable Adapters for Transferable Safety Alignment

**arXiv ID:** 2606.06519 | [PDF](https://arxiv.org/pdf/2606.06519v1)

**作者:** Yanghan Wang `[一作]` (Southeast University), Xin Geng `[通讯]` (Southeast University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 SafeGene，一种可重用的安全适配器模块，旨在在对大语言模型进行任务微调后恢复其安全对齐。

**💡 创新点**

创新点包括：①将安全能力抽象为可重用的适配器，脱离任务特定更新；②通过对安全对齐模型与其轻度降解版本的差异进行蒸馏，提取层级安全向量；③采用数据感知层选择策略筛选对安全转移更敏感的层；④在下游模型中仅对选定层的标量系数进行少量样本微调，保持任务性能。

**🔧 技术方法**

主要技术：LoRA 低秩适配器、对齐-降解模型差异蒸馏损失、层级激活对比的层选择、少量样本的标量系数微调；使用安全判定器（Beaver-dam-7B、Qwen3.5-Flash）评估攻击成功率。

**📊 数据集**

数据集：安全基准——BeaverTails、AdvBench、DirectRefusal；下游任务——AG News、SST2、MNLI、BoolQ；模型家族包括 Qwen2.5-7B/1.5B、Qwen3-1.7B、GLM-4-9B、Llama-3-8B。

**📈 对比分析**

与 SafeLoRA、SafeMERGE、SaLoRA 等现有安全适配方法对比，SafeGene 在 Qwen2.5-7B 上平均降低攻击成功率至 33.84%（相比 Fine‑tuned 的 39.91%），并保持 90.12% 的任务准确率，优于其它方法的 40.63%、37.75% 和 36.84%。

**⚠️ 局限性**

局限性：①只适用于结构相同、适配点相同的模型家族；②实验仅针对单轮文本安全评估，未覆盖多轮 jailbreak、跨语言攻击或工具使用场景；③需使用少量目标域安全数据进行系数微调，若攻击分布大幅偏移可能影响效果。

---

## 35. On the Hardness of Optimal Motion on Trees

**arXiv ID:** 2606.06686 | [PDF](https://arxiv.org/pdf/2606.06686v1)

**作者:** Tzvika Geft `[一作]` `[通讯]` (Rutgers University), Tzvika Geft (Rutgers University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文构建了一套统一的归约框架，将栈式重排问题（Stack Rearrangement）转化为多代理路径规划（MAPF）以及可分离球形图（PMT）的多种目标（距离、完成时间、流量）问题，证明了在树形结构下这些问题在标准目标下都是NP‑难的。

**💡 创新点**

创新点在于：1）首次通过栈重排问题统一推导多目标 MAPF 的 NP‑难性；2）在极简的树结构（深度3的分叉星或最大度3树）上给出完整的困难性证明；3）展示了彩色与标记版本的最小颜色阈值（仅需两色）以及树最大度阈值（3度）之间的边界。

**🔧 技术方法**

技术主要是多阶段的归约与构造：先把最小反馈弧集（MFAS）归约到有深度3的 Labeled Stack Rearrangement，再把 2‑colored Stack Rearrangement 归约到 3‑Partition；随后将 Stack Rearrangement 转化为可分离球形图（PMT）和树形 MAPF 的距离、完成时间、流量目标；在每一步都利用树的 LIFO 约束、缓冲栈和“atomic”移动重排等技巧保证等价性。

**📊 数据集**

本文没有使用实验数据集，所有结果均为理论证明，所构造的实例为多项式大小的人工构造实例。

**📈 对比分析**

由于缺乏实验，本文未与具体算法或基线方法做性能比较；所给出的结果仅说明在上述限制下任何多目标规划算法都不可避免地面临 NP‑难性。

**⚠️ 局限性**

局限性包括：1）归约使用的栈深度为3，作者认为可能可进一步压缩至2，但尚未完成；2）仅在树形结构下讨论，未说明在更一般图结构下的复杂度；3）没有提供任何多项式时间算法或近似算法，只阐明了困难边界。

---

## 36. A Geometric Account of Activation Steering through Angle-Norm Decomposition

**arXiv ID:** 2606.06735 | [PDF](https://arxiv.org/pdf/2606.06735v1)

**作者:** Georgii Aparin `[一作]` (Huawei Noah's Ark Lab), Tatiana Gaintseva `[通讯]` (Queen Mary University of London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对语言模型的激活调控进行了几何层面的分解，区分了角度（概念方向）与半径（隐藏状态范数）两种效应；

**💡 创新点**

创新点在于将激活调控视为双参数（角度与半径）干预，揭示了仅靠加法强度难以解释的行为差异，并证明概念信息主要存储在方向上，但范数仍影响生成稳定性；

**🔧 技术方法**

技术主要包括对隐藏状态的向量分解、六种对比调控方法（CAA、CAA‑r、CAA‑m、S、AS、SN）以及对角度与半径的独立调节实验；

**📊 数据集**

使用了七个不同规模的Transformer模型（Llama、Qwen、Gemma）以及四个概念数据集（TruthfulQA、SST‑2、CivilComments、IMDB）进行评估；

**📈 对比分析**

通过统一的实验框架比较上述方法，结果表明在匹配角度目标时，保持原范数的S方法在高强度下会显著提升困惑度和能力退化，而允许范数变化的CAA‑m在保持语义效果的同时表现出更好的生成稳定性；

**⚠️ 局限性**

局限性包括：仅在单一固定层进行调控，未探究不同层的最佳角度–范数权衡；实验覆盖的模型和概念有限；所有方法使用相同的对比平均差方向，未检验其他方向估计方式；以及范数缩放实验仅使用离散β值，缺乏自动化选择规则。

---

## 37. The Identity Trap in EEG Foundation Models: A Diagnostic Audit

**arXiv ID:** 2606.06647 | [PDF](https://arxiv.org/pdf/2606.06647v1)

**作者:** Jun-You Lin `[一作]` (National Yang Ming Chiao Tung University), Tzyy-Ping Jung `[通讯]` (University of California, San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了 FMScope 诊断框架，在三种预训练 Transformer EEG 基础模型上对四个小样本静息态 EEG 数据集进行冻结表示层面的五项诊断，以检测并消除“身份陷阱”——即模型在不需要的个体身份信息上获得高准确率。

**💡 创新点**

创新点在于首次系统性识别并量化 EEG 基础模型中的身份陷阱，证明其为可线性可消除的主轴，并通过 1/f 无周期背景作为身份载体揭示了身份与临床标签的竞争关系；同时提供了预训练前的“飞行前检查”方法，帮助区分真正的生物标记与个体特征。

**🔧 技术方法**

技术上使用了方差分解、最小二乘概念消除（LEACE）、FOOOF 频谱去除、层级标签/身份探测以及跨被试方向一致性等五个冻结表示诊断，并与全模型微调、线性探针、传统和深度基准进行比较。

**📊 数据集**

使用的数据集包括：EEGMAT（精神算术）、ADFTD（阿尔茨海默/额颞叶癫痫）、SleepDep（睡眠剥夺）和 Stress-DASS（慢性压力），每个数据集均为小样本（N≤65）静息态 EEG。

**📈 对比分析**

对比方法是在 5 折受试者分离交叉验证下，评估冻结线性探针与全微调在三种基础模型（LaBraM、CBraMod、REVE）以及经典手工特征和深度基准的平衡准确率；结果表明微调往往不明显优于冻结探针，且仅在已知跨被试标记的细胞中出现微小提升。

**⚠️ 局限性**

局限性包括：仅以单一数据集代表每个细胞，样本量有限导致统计不稳；诊断结果对不同架构（REVE 与 LaBraM/CBraMod）差异解释仍属描述性；输入级去除实验未能确立因果机制；对固定特质标签（如诊断）无法通过采样分离身份与标签。

---

## 38. ChronoForest: Closed-Loop Multi-Tree Diffusion Planning for Efficient Bridge Search and Route Composition

**arXiv ID:** 2606.06618 | [PDF](https://arxiv.org/pdf/2606.06618v1)

**作者:** Jungmin Seo `[一作]` (Seoul National University), Jaesik Park `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种闭环规划框架ChronoForest，用短期离线轨迹片段构造长周期路线，并在路线上不断重解以优化路径质量。

**💡 创新点**

创新点在于将基于时间距离的短距离桥段搜索与在线路重解耦合，形成闭环；同时使用Anchor‑chaining树扩展和多树协同器来动态分配搜索预算，使得在无预先计算的点对成本矩阵的情况下仍能实现高效的多旅行商路线规划。

**🔧 技术方法**

技术包括时间距离（temporal‑distance）估计、扩散模型的目标条件引导、树结构的多路径搜索、桥段证据聚合、闭包（all‑hop closure）路由求解以及软接收机制。

**📊 数据集**

主要使用OGBench AntMaze‑Stitch数据集（Medium、Large、Giant三种规模）进行评估，并在Hamiltonian路由实验中对比。

**📈 对比分析**

与Diffuser、Replan、CompDiffuser等基线相比，ChronoForest在Giant分割的成功率提升高达34.5个百分点；在路程长度上也比时间距离固定或图结构固定的方案更短，且规划时间和扩展节点数均低于穷举搜索，接近图结构固定基准。

**⚠️ 局限性**

局限性包括对时间距离先验的依赖，长距离误估会导致搜索预算浪费；对更大规模的点集与更复杂约束的扩展仍需更强剪枝与近似组合求解；理论分析仅给出足够性证明，缺乏最优搜索策略的完整说明。

---

## 39. The Custody Envelope Threshold: Authority-Scaled Admission of External Artifacts in Institutional Infrastructure

**arXiv ID:** 2606.06767 | [PDF](https://arxiv.org/pdf/2606.06767v1)

**作者:** Amadeus Brandes `[一作]` `[通讯]` (Independent Researcher), Amadeus Brandes (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Custody Envelope Threshold（托管包络阈值）模型，用以解释机构在面对外部维护的软件工件时如何决定是否直接准入，并通过四个关键条件（身份、入口、执行权限、撤销能力）来评估可否实现直接准入；

**💡 创新点**

其创新点在于将权威层级与托管闭环结合，构建一个可预测的四维度模型，能够统一解释不同工件类（如依赖包、GitHub Action、容器镜像、Terraform 提供者/模块、IDE 扩展、模型权重等）在机构内的准入差异，并提供治理模式的推断机制；

**🔧 技术方法**

采用了参考监控、最小权限与交易成本经济学的理论基础，并设计了四个可量化的指标（身份闭环、入口闭环、执行权限、撤销闭环）以及相应的评分表；

**📊 数据集**

研究使用公开的文档与案例数据（npm、GitHub Actions、Docker Hub、Terraform Registry、VS Code Marketplace、Hugging Face Hub、MCP 服务器等）以及一个包含 8 个用例的验证集；

**📈 对比分析**

通过将模型的预测结果与实际观察到的治理模式（直接准入、代理、策略中介、供应商中介、内部化、隔离/拒绝）进行对比，使用准确率、宏 F1 等指标评估；实验表明该模型在大多数案例中能准确预测主治理模式，表现优于多数基线；

**⚠️ 局限性**

局限性包括：评分过程仍带主观性、仅依赖公开信息、未涵盖法律/出口/安全/商业门槛等后续约束、以及快速演进的生态环境可能导致模型失效。

---

## 40. Synthics: Synthetic Physics-like Datasets for Machine Learning

**arXiv ID:** 2606.06724 | [PDF](https://arxiv.org/pdf/2606.06724v1)

**作者:** Jari Vepsäläinen `[一作]` `[通讯]` (Aalto University), Jari Vepsäläinen (Aalto University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `14d48e9d-0069-4ad9-996a-1d5968216998` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

该论文提出 Synthics 方法，用贝叶斯概率上下文无关文法从物理方程语料生成结构相似的回归数据集。

**💡 创新点**

创新点在于将贝叶斯扩展的 PCFG 与可应用域探测和受限采样结合，保证生成方程的结构和物理合法性。

**🔧 技术方法**

使用贝叶斯概率上下文无关文法（B‑PCFG）、Dirichlet 先验、非侵入性域探测、混合均匀/截断正态采样及 Kolmogorov–Smirnov 检验。

**📊 数据集**

采用 Feynman 讲义中的 100 条基础物理方程作为语料。

**📈 对比分析**

通过 KS 检验对八个结构特征比较，B‑PCFG 在所有特征上通过；在超参数排序实验中，Synthics 生成的数据使 GB 迭代器的配置排名与真实数据高度相关（ρ≈0.9），并将最优配置定位在第 6 名，远优于随机树和噪声。

**⚠️ 局限性**

局限在于仅处理闭式代数方程、依赖小型语料导致先验强度关键、对大规模或更广泛任务验证不足，以及不支持微分方程等更复杂形式。

---

## 41. Adversarial Co-Thinking: Calibration and Triangulation Across Multiple GenAI Tools in HCI Writing

**arXiv ID:** 2606.06702 | [PDF](https://arxiv.org/pdf/2606.06702v1)

**作者:** Pia Tukkinen `[一作]` `[通讯]` (Aalto University), Pia Tukkinen (Aalto University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

作者通过在学术写作过程中使用Claude、ChatGPT和Gemini三种GenAI工具，发展并记录了一种名为“对抗式共同思考”的工作流程，以在保持知识自主权的同时提升写作质量。

**💡 创新点**

创新点在于提出“对抗式共同思考”概念——将工具校准为真实同行评审标准并相互对比，从而将工具之间的分歧转化为有效的批判性反馈，揭示了“措辞捕获”问题并提供了一种高阶评估实践。

**🔧 技术方法**

采用的技术包括少量样例提示（few‑shot prompting）、角色提示（persona prompting）、基于真实评审的校准、跨工具比较、迭代草稿与人工评估，以及手动整合与改写。

**📊 数据集**

使用的数据为作者自身的草稿笔记、AI生成的多版本文本、提示交互记录以及从权威HCI会议获得的同行评审稿，用于校准与评估。

**📈 对比分析**

比较方法是给同一初始材料和相同提示分别请求Claude、ChatGPT和Gemini生成稿件，随后人工对比它们在论点强度、措辞、理论框架等维度的差异。结果显示：Gemini倾向于夸大主张，ChatGPT偏向理论化，Claude更贴合作者流程但遗漏要点；通过跨工具对比，作者发现争议点能激发更深入的批判，提升文本质量。

**⚠️ 局限性**

局限性包括：仅适用于作者个人的实验环境，需多家付费订阅且成本高昂；对非英语母语作者可能导致措辞捕获；缺乏客观量化评估，结论主要基于主观判断；可扩展性与在不同工具/研究场景中的有效性仍待实证验证。

---

## 42. Multi-Scale Feature Attention Network for Polymer Classification using THz Dual-Comb Spectroscopy

**arXiv ID:** 2606.06554 | [PDF](https://arxiv.org/pdf/2606.06554v1)

**作者:** Roshni Mahtani `[一作]` (Universitat Politècnica de València), Rocío del Amor `[通讯]` (Artikode Intelligence S.L.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过THz双梳光谱技术对12种聚合物进行高分辨率光谱采集，并提出多尺度特征注意网络（MSFAN）实现聚合物分类。

**💡 创新点**

将厚度不变归一化与可学习频谱校准、多尺度卷积、跨特征注意力以及注意力池化融合，自动提取最具辨别力的THz频段并实现模型可解释性。

**🔧 技术方法**

使用THz-DCS采样、深度学习网络MSFAN（含门控、多尺度Inception、MHSA、注意力池化、MLP）、L1稀疏正则与标签平滑交叉熵。

**📊 数据集**

基于5天采集的12类聚合物数据集（纯料、复合层、商业混合、生物聚合物），每类209个样本，100–590 GHz范围内50个频点，包含低/高增益。

**📈 对比分析**

与ANN、CNN、PSDN_Inception、ANN_PM、Improved_CNN_PM、Transformer_PM等传统与先进模型对比，MSFAN取得ACC 85.2%、F1 82.4%，比最佳对手高约1.4%/1.3%，在聚合物对比（如H与J）上的区分显著提升。

**⚠️ 局限性**

仅适用于完整THz频谱且假设无污染的闭集场景；对外域材料、表面污染和工业环境中的复杂污染物鲁棒性不足，未来需引入硬稀疏选择与开放集识别。

---

## 43. Cubic Hermite Lattice Structures

**arXiv ID:** 2606.06500 | [PDF](https://arxiv.org/pdf/2606.06500v1)

**作者:** Yaonaiming Zhao `[一作]` (Zhejiang University), Qiang Zou `[通讯]` (Zhejiang University)

**通讯引用:** 752 | [OpenAlex ID](https://openalex.org/A5028026050)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种基于隐式卷积场和三次 Hermite 曲线控制的晶格结构建模方法，实现了可变半径棱柱的高效评估与切片。

**💡 创新点**

创新点包括：① 用三次 Hermite 曲线直接控制棱柱的径向分布，拓展了传统线性或二次曲线的设计空间；② 在卷积核和 Hermite 曲线均为多项式的情况下推导出棱柱隐式场的解析表达式，显著提升评估效率和数值稳定性；③ 通过段落分割和独立控制参数实现更复杂的棱柱轮廓。

**🔧 技术方法**

技术手段包括：隐式卷积场建模、三次 Hermite 曲线参数化、解析多项式积分、GPU 并行评估、Marching Cubes 和 Marching Squares 切片算法。

**📊 数据集**

使用了典型 CAD 模型作为案例：骨骼模型、机械零件、齿轮等，并在这些模型上生成梯度晶格与可变棱柱的三维网格。

**📈 对比分析**

与传统基于距离场或二次曲线的隐式方法比较，提出的方法在切片时间（≈0.05–0.06 s/层）和模型生成时间（≤1 min）上保持一致或更优；在多段棱柱实验中展示了更大的几何多样性，且保持了鲁棒的无非流形切片结果。

**⚠️ 局限性**

局限性包括：① 隐式场求和导致节点处出现“膨胀”现象，影响几何精度；② 对 Hermite 曲线参数的变化不够直观、敏感度低，控制轮廓的难度较大；③ 当前实现仍需进一步优化多节点密集连接时的数值稳定性。

---

## 44. Modular Monolingual Adaptation using Pretrained Language Models

**arXiv ID:** 2606.06738 | [PDF](https://arxiv.org/pdf/2606.06738v1)

**作者:** Nalin Kumar `[一作]` (Charles University), Ondřej Dušek `[通讯]` (Charles University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种低资源语言适配的模块化方法，将tokenizer改为语言特定、冻结嵌入层，只微调其余参数。

**💡 创新点**

创新点在于不需要对整个模型进行微调，而是通过自定义tokenizer并冻结嵌入层，实现非嵌入训练，既降低过拟合又提升训练效率。

**🔧 技术方法**

使用的技术包括BERT、mBERT、mmBERT encoder、WordPiece自定义tokenizer、FastText词向量初始化、Masked Language Modeling目标，并与全模型微调、仅嵌入微调以及LoRA等做对比。

**📊 数据集**

使用的数据集为：CC‑100无监督语料（苏格兰盖尔语、爱尔兰语、克丘亚语），WikiAnn做NER评测，UD Treebanks做POS评测。

**📈 对比分析**

方法与全模型微调、仅嵌入微调、LoRA等在mask‑filling、NER和POS任务上进行准确率比较。结果显示非嵌入微调在mask‑filling上优于全微调，在NER/POS上与全微调相当或更好，同时训练参数更少、训练更快、推理延迟更低。

**⚠️ 局限性**

局限性包括仅测试三种低资源语言，缺乏对多脚本或形态学差异更大的语言验证；评估仅涵盖简单NLU任务；实验仅针对encoder模型，未扩展到decoder结构。

---

## 45. Gaussian Process Latent Factor Regression for Low-Data, High-Dimensional Output Problems

**arXiv ID:** 2606.06576 | [PDF](https://arxiv.org/pdf/2606.06576v1)

**作者:** Edward T. Stevenson `[一作]` (University of Cambridge), Miles Cranmer `[通讯]` (University of Cambridge)

**通讯引用:** 1514 | [OpenAlex ID](https://openalex.org/A5078731429)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种 Gaussian Process Latent Factor Regression (GPLFR) 模型，能够在高维输出空间下同时学习压缩和预测。

**💡 创新点**

创新点在于通过解析地边缘化线性高斯解码器权重，将压缩与预测整合到一个统一目标函数中，使模型能够在高维输出下高效训练。

**🔧 技术方法**

使用了高斯过程先验、线性高斯解码器、解析边缘化技术以及与 PCA‑GP、LMC、LV‑MOGP 等方法的对比实验。

**📊 数据集**

主要实验基于全球气候模型对岩石行星的空间解析模拟数据进行。

**📈 对比分析**

与传统压缩‑预测管道和多输出 GP 方法比较，GPLFR 在高维输出下实现了更低的预测误差或更好的拟合效果。

**⚠️ 局限性**

局限性包括：对输入空间的 GP 假设可能不够灵活，线性解码器限制了表达能力，且对极大规模数据的可扩展性仍需进一步验证。

---

## 46. Differentiable 3D Triangle-Triangle Intersection Energy

**arXiv ID:** 2606.06511 | [PDF](https://arxiv.org/pdf/2606.06511v1)

**作者:** Tianyu Wang `[一作]` (Independent), Tianyu Wang `[通讯]` (Independent)

**通讯引用:** 12192 | [OpenAlex ID](https://openalex.org/A5051603885)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种二阶可微的三维三角形-三角形相交能量，并通过GPU实现的近似牛顿优化方法，能够在不需要初始化或用户交互的情况下自动消除网格间的相交，实现交叉自由的全局可注入性。

**💡 创新点**

创新点主要包括：①将三维相交判定化简为一维线段重叠测试，构造闭式可微能量；②给出能量的显式梯度与近似Hessian，支持二阶优化；③利用空间哈希与GPU并行化，实现大规模实时相交检测；④不依赖预先的无交叉初始化。

**🔧 技术方法**

使用了Möller的三角形相交判定算法、Minkowski和、显式梯度与Hessian推导、GPU并行计算、空间哈希加速、近似牛顿优化以及自动微分库做参考实现。

**📊 数据集**

论文未公开具体数据集，实验采用标准三维网格（如TetGen或Stanford 3D扫描模型）进行仿真，展示从非法状态到合法状态的过渡。

**📈 对比分析**

与传统需要预先无交叉初始化的相交处理方法相比，该方法在保持相交自由的同时无需额外交互；实验显示GPU加速后可在实时或准实时范围内处理数十万三角形，显著提升了鲁棒性与效率。

**⚠️ 局限性**

局限性包括：①需要GPU并行支持，CPU实现效率不高；②近似Hessian可能在极端几何配置下收敛缓慢或不稳定；③对非常细小或接近共面三角形的相交检测可能仍存在数值不稳定性；④缺乏公开的基准数据集与量化评测。

---

## 47. Performance Variation in Deep Reinforcement Learning

**arXiv ID:** 2606.06746 | [PDF](https://arxiv.org/pdf/2606.06746v1)

**作者:** Haruto Tanaka `[一作]` (University of Alberta), A. Rupam Mahmood `[通讯]` (University of Alberta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了最小-最大百分位区间（min-max IPR-90）和运行级百分位突出显示（RPH）两种工具，用于衡量和可视化深度强化学习单任务的性能波动。

**💡 创新点**

创新点在于将非参数百分位区间（覆盖中心 90% 数据）归一化为百分比，并通过 RPH 直观展示不同运行间的学习曲线差异，从而更准确地捕捉性能变化并避免传统标准误/标准差的低估问题。

**🔧 技术方法**

使用了基于样本百分位的统计方法（IPR-90）、min-max 归一化、以及 RPH 直方图可视化；实验中采用 CleanRL 实现的 PPO、SAC、TD-MPC、TD-MPC2、DQN、Rainbow 等算法。

**📊 数据集**

数据集包括 59 个连续控制任务（MuJoCo + DeepMind Control Suite）和 5 个 Atari 任务（ALE），每个任务均进行 100 次独立运行，收集回报/人类标准化得分。

**📈 对比分析**

通过对比 min-max IPR-90、均值、标准差、5%/50%/95% 分位等指标，发现：层归一化/终层归一化可显著降低 PPO 的性能波动但对 SAC 影响有限；TD-MPC 在连续控制任务中实现了最小的 IPR-90（约10%）并且数据效率最高；Rainbow 在离散任务中性能优于 DQN，但两者的 IPR-90 变化相近，说明提升性能并未解决高波动问题。

**⚠️ 局限性**

局限性包括：IPR-90 采用 90% 范围缺乏理论依据，需足够多的独立运行（至少 5 次）才能准确估计 5%/95% 分位；对单跑耗时大、资源有限的实验不太适用；RPH 虽直观但在极少量跑数时可导致可视化模糊。

---

## 48. Anchored, Not Graded: Vision-Language Models Fail at Slant-from-Texture Perception

**arXiv ID:** 2606.06714 | [PDF](https://arxiv.org/pdf/2606.06714v1)

**作者:** Qian Zhang `[一作]` (Brown University), James Tompkin `[通讯]` (Brown University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

评估并诊断 VLM 在纹理纹路的斜率感知任务中的表现，发现其在零样本下仅预测离散锚点，无法连续映射纹理梯度到三维倾斜。

**💡 创新点**

提出斜率感知为检验 VLM 低层几何表达能力的诊断任务，并揭示 VLM 的回答锚定是语言读出瓶颈而非视觉编码缺陷。

**🔧 技术方法**

使用大规模多模态预训练的 VLM（Gemma3、LLaMA4、LLaVA、Mistral、Moondream、Qwen2.5‑VL 等）、prompt engineering、in‑context learning、LoRA 微调、线性探针与注意力头消融。

**📊 数据集**

基于合成的点纹理斜率数据集（2000 条训练 + 400 条测试），涵盖 10 种光学倾斜、10 种视场、2 种曲率符号、12 种随机点纹理。

**📈 对比分析**

与人类、无监督 CNN 及监督 CNN 的偏差量化指标（MAE、曲率判别准确率）对比；零样本 VLM MAE 超过 40°，曲率准确率接近 50%；微调后 MAE 降至 15.3°，曲率准确率升至 86%，仍低于无监督 CNN 的 96.4%。

**⚠️ 局限性**

局限性包括仅使用点纹理刺激、未包含更丰富纹理与深度线索、仅评估单一低维连续变量、未探究语言模型内部的读出机制，且结果可能受特定模型或推理后端的影响。

---

## 49. USU-Corn-WeedDB: A UAV RGB Image Dataset for Multi-Species Weed Detection in Forage Corn

**arXiv ID:** 2606.06709 | [PDF](https://arxiv.org/pdf/2606.06709v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 50. Lane Change Trajectory Planning for Personalized Driving Comfort and Mobility Efficiency

**arXiv ID:** 2606.06805 | [PDF](https://arxiv.org/pdf/2606.06805v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 51. Multi-Robot Planning and Control from CCTV Camera Networks in a Real Warehouse

**arXiv ID:** 2606.06762 | [PDF](https://arxiv.org/pdf/2606.06762v1)

**作者:** Luke Robinson `[一作]` (University of Oxford), Daniele De Martini `[通讯]` (University of Oxford)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过仅依赖无标定的CCTV摄像头网络和边缘计算，实时离线控制并协调一支包含四台异构移动机器人（无任何导航硬件）的仓库机器人舰队，实现点对点任务的完整自主执行。

**💡 创新点**

创新点在于：① 在未标定、像素级拓扑摄像头图谱上直接进行图像空间规划；② 采用层次化的优先级-联合规划与信号量式摄像头交叉区资源管理，避免碰撞与死锁；③ 用数据驱动的模型预测机器人在各摄像头视角下的尺寸与速度，实现对视角透视变换的自适应，无需几何标定。

**🔧 技术方法**

主要技术包括：Hybrid A*搜索（时间参数化的运动原语）、YOLOv8小模型做机器人检测、角度回归器、两层MLP预测框尺寸与图像速度、基于Pure Pursuit的图像空间控制、Wi‑Fi远程速度命令、以及在边缘服务器上并行的图像处理与路径规划。

**📊 数据集**

使用的“数据集”是：在30个固定摄像头（Raspberry Pi摄像头）覆盖的六条仓库通道中，收集约40条随机漫游的伪标签轨迹（位置、速度）以及多次旋转标注的检测框；数据仅在一条通道上标注并复制，适用于整个相似结构的仓库。

**📈 对比分析**

与人类远程操作进行对比，单机机器人在相同路线的平均时间比人类慢约5.1%（包含规划时间），在无其他机器人干扰时仅慢1.5%；在多机器人环境下仍保持与人类相近的性能，且一次性完成了76次点对点任务，累计行驶1.3公里，规划时间平均仅1.06 s（≈14 s/次）。

**⚠️ 局限性**

局限性包括：① 规划期间机器人会停滞，未实现即时的“前瞻性”规划；② 控制器对参数敏感，需要手动调优；③ 机器人必须视觉可区分且假设恒定地面速度；④ 冲突检测过于保守，尤其在低角度视角下会误判；⑤ 需完整覆盖训练区域以保证姿态估计的鲁棒性。

---

## 52. Flatland: The Adventures of Gradient Descent with Large Step Sizes

**arXiv ID:** 2606.06722 | [PDF](https://arxiv.org/pdf/2606.06722v1)

**作者:** Leonardo Galli `[一作]` (LMU Munich), Holger Rauhut `[通讯]` (LMU Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于局部段平滑的梯度下降大步长定义，并设计了等式线搜索等一阶方法实现；

**💡 创新点**

创新点在于统一大步长定义、引入等式线搜索与外推、证明在EoS上收敛、揭示大步长导致的全局平坦鞍点并提出NLS-ub避免策略；

**🔧 技术方法**

使用梯度下降、非单调等式线搜索、Polyak 初始化、回溯/外推、局部段平滑与 Hölder 平滑分析等技术；

**📊 数据集**

在 CIFAR‑10、CIFAR‑100、SVHN、EMNIST 的 5000 样本子集上，使用 MLP、CNN、ResNet34、WideResNet、DenseNet、VGG11 等模型训练；

**📈 对比分析**

与传统单调线搜索、SAM、CDAT 等方法比较，NLS 与 PoNLS 在训练损失和锐度上表现最佳，NLS‑ub 在测试准确率上往往优于 CDAT；

**⚠️ 局限性**

局限性包括仅针对全批 GD、未证明 SGD 收敛、对激活函数的依赖、需更大规模实验、理论上限与实际步长之间仍有差距。

---

## 53. CAF-Gen: A Multi-Agent System for Enriching Argumentation Structures

**arXiv ID:** 2606.06646 | [PDF](https://arxiv.org/pdf/2606.06646v1)

**作者:** Jakub Bąba `[一作]` (Warsaw University of Technology), Jarosław Chudziak `[通讯]` (Warsaw University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CAF-Gen——一种基于多代理的大语言模型框架，能够将浅层论证结构自动扩展为符合 Carneades 论证框架（CAF）的正式模型。

**💡 创新点**

创新点在于：①设计了迭代的 Creator‑Reviewer 两代理循环，显著提升生成模型的可接受率（从 34.6% 提升至 91.3%）；②通过该循环实现了对结构完整性、论证方案、证明标准等 CAF 专有属性的精细校正；③在无外部标注数据的情况下，仅利用原始论证图进行自动化富化。

**🔧 技术方法**

技术主要包括：大语言模型（Google Gemini 2.5 Pro）实现生成与评审；LangGraph 管理多代理工作流；structured prompting 与规则化投影（Projection）用于映射与评估；迭代反馈机制与错误/警告/建议分类。

**📊 数据集**

使用的数据集为 UKP Argument Annotated Essays v2（402 篇，6089 语句，3832 关系），并对其进行 JSON 格式化后输入框架。

**📈 对比分析**

通过第一轮接受率、最终接受率、平均迭代次数以及错误/警告分布评估迭代流程；通过投影后与原始注解对齐，计算 Component 和 Relation 的 Precision/Recall/F1。结果显示：Component 识别 99.8% 召回率，Relation 召回率 99.1% 但 Precision 仅 67.1%，整体 F1 为 80.0%。

**⚠️ 局限性**

局限性包括：①对 Reviewer 的错误/警告判定仍可能过度严格，导致可接受模型被误拒；②缺乏针对 CAF 新属性（论证方案、证明标准等）的大规模人工标注基线，无法全面评估其正确性；③目前仅基于单一 LLM，未对不同模型的性能做横向对比；④系统在某些情形下会生成冗余或过度简化的结构，需要进一步的人工或规则干预。

---

## 54. WorldBench: A Challenging and Visually Diverse Multimodal Reasoning Benchmark

**arXiv ID:** 2606.06538 | [PDF](https://arxiv.org/pdf/2606.06538v1)

**作者:** Yida Yin `[一作]` (Princeton University), Zhuang Liu `[通讯]` (Princeton University)

**通讯引用:** 1645 | [OpenAlex ID](https://openalex.org/A5100452088)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个以视觉多样性为核心的新型多模态语言模型（MLLM）评估基准，收集了2000张非标志性、上下文丰富的图像，并为每张图像设计了四选一的挑战性问题。

**💡 创新点**

创新点：①从视觉多样性而非任务多样性构建基准；②使用LLM辅助构建覆盖7个视觉域的2000个细粒度概念分类体系；③通过有效秩、参与度比及人类评估验证视觉多样性优于现有基准；④问题设计迭代至至少一款前沿模型无法回答。

**🔧 技术方法**

技术方法包括：LLM生成分类体系、搜索引擎检索图像、视觉编码器提取图像嵌入计算有效秩与参与度比、Bradley–Terry模型进行人类评估、结构化试错法设计问题。

**📊 数据集**

数据集：新基准包含2000张图像，涵盖动物与植物、事件与活动、在线经济、科学与工程、Web代理、游戏与机器人等7个视觉域；图像来源于Google、Bing等搜索引擎及AgentVQA等公开数据集。

**📈 对比分析**

评估方法：在新基准上评测15款MLLM，Gemini‑3.1‑Pro平均准确率64.0%，最优秀的开源模型Qwen3.5‑VL‑27B为56.6%；无模型在任何域达到75%以上，显示基准极具挑战性。

**⚠️ 局限性**

局限性：①尚未覆盖所有视觉内容（如数字化网页、截图等）；②问题仅为四选一多项选择，难以评估生成式能力；③模型偶尔会出现回答错误；④多样性评估主要基于视觉编码器嵌入，可能忽略某些语义维度。

---

## 55. What Do People Actually Want From AI? Mapping Preference Plurality

**arXiv ID:** 2606.06674 | [PDF](https://arxiv.org/pdf/2606.06674v1)

**作者:** Julia Sepúlveda Coelho `[一作]` (Oxford Internet Institute, University of Oxford), Scott A. Hale `[通讯]` (Oxford Internet Institute, University of Oxford)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对PRISM数据集中的1500条英文开放式回答进行混合方法分析，量化并编码人们对AI的价值与偏好，揭示出多样性、争议性与上下文依赖的特征。

**💡 创新点**

创新点在于：①采用LLM辅助的自上而下编码流程；②将社会学与伦理学框架与技术评估结合，揭示“真相”与其他价值的多重解释；③系统性指出RLHF聚合偏好所带来的“知识压抑”和“价值单一化”风险，并提出可个性化、多模态对齐的方向。

**🔧 技术方法**

使用的技术包括：质性编码（人手+LLM）、主题分析与ValueCompass代码本、LASSO + OLS回归、描述统计、事实核查以及Cohen κ一致性评估。

**📊 数据集**

使用的数据集是PRISM数据集（1500份跨越75国的英文问卷）以及从PRISM中抽取的50段对话样本用于事实核查。

**📈 对比分析**

比较方法：通过人类与LLM编码的一致性（κ≈0.5）评估编码质量，比较各价值的出现频率、争议度及其与人口统计的关联；未给出具体模型性能指标，重点在对偏好多样性的定性与定量描述。

**⚠️ 局限性**

局限性包括：①样本仅为英语问卷，全球南方代表性不足；②开放式回答的“top‑of‑mind”特性可能遗漏日常常见偏好；③LLM编码随样本规模增大质量下降；④未直接验证RLHF模型的对齐效果；⑤跨文化差异的解释受限于样本分布。

---

## 56. Towards Serverless Semi-Decentralized Federated Learning with Heterogeneous Optimizers

**arXiv ID:** 2606.06687 | [PDF](https://arxiv.org/pdf/2606.06687v1)

**作者:** Su Wang `[一作]` (Princeton University), H. Vincent Poor `[通讯]` (Princeton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无服务器半去中心化联邦学习方法SSD-FL，利用一次性设备间初始化形成聚类，并在聚类内外交替进行训练，实现更高效的模型收敛。

**💡 创新点**

创新点在于将网络拓扑与数据及优化器异质性结合的聚类机制、有效损失函数与Cheeger不等式驱动的聚类阈值，以及在无服务器环境下的协同训练与收敛理论。

**🔧 技术方法**

采用图正则化的有效损失函数、谱聚类与Fiedler向量、Cheeger导数、梯度方差估计、随机梯度下降与不同优化器（SGD、Prox-SGD、动量）等技术。

**📊 数据集**

使用FMNIST和CIFAR-10两大图像数据集，分布式划分为多设备、不同异质性场景。

**📈 对比分析**

与同步DFL、周期性DFL、随机DFL和随机聚类DFL等四类基线比较，SSD-FL在多种网络拓扑、设备数与异质性下表现出更快收敛、更高最终准确率，尤其在极端非IID和大规模网络时优势明显。

**⚠️ 局限性**

局限在于需要预先设定聚类阈值和通信周期，聚类过程对网络规模与拓扑变化敏感；对极度稀疏网络或极大设备数时仍需进一步优化。

---

## 57. ShallowBench: Benchmarking Generative Drug Design Models on Shallow-Pocket Targets

**arXiv ID:** 2606.06717 | [PDF](https://arxiv.org/pdf/2606.06717v1)

**作者:** Saket Reddy `[一作]` (University of Illinois - Urbana-Champaign), Shiwei Liu `[通讯]` (University of Illinois - Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文创建了名为ShallowBench的基准数据集，并在该基准上评估了现有生成式结构药物设计模型的性能。

**💡 创新点**

创新点在于提出了基于Alpha Shape体积差的低凹陷筛选方法，系统性地构建了5,780个浅口袋靶点并提供了严格的训练/测试拆分，填补了浅表面靶点的评估空白。

**🔧 技术方法**

使用了体积计算与Alpha Shape“lid”技术来定义口袋凹陷；采用3D等变扩散网络（DiffSBDD、TargetDiff）和简化GNN（SimpleSBDD）等生成模型；利用AutoDock Vina和SCASA进行亲和力和形状互补性评估。

**📊 数据集**

数据集包括CrossDocked2020作为来源，提取并筛选出的ShallowBench（5,780个浅口袋靶点）及其对应的训练/测试拆分（4,995/785），以及与之对照的同等规模随机抽样控制集。

**📈 对比分析**

比较方法采用化学有效性、QED、形状互补性Sc和Vina预测亲和力等指标。结果显示所有模型在浅口袋上亲和力普遍下降；SimpleSBDD保持最高亲和力但形状互补性和化学有效性下降；DiffSBDD化学有效性高但形状互补性接近零；TargetDiff在化学有效性上显著衰退，虽然形状互补性相对较好。

**⚠️ 局限性**

主要限制包括未进行多随机种子或置信区间评估，计算资源受限导致实验规模有限；此外使用的AutoDock Vina评分与真实实验亲和力可能不完全相关。

---

## 58. Korean Culture into LLM Alignment: Toward Cultural Coherence

**arXiv ID:** 2606.06797 | [PDF](https://arxiv.org/pdf/2606.06797v1)

**作者:** MinJae Jung `[一作]` (DATUMO Inc), Minwoo Kim `[通讯]` (DATUMO Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对韩语语境构建了文化一致性对齐数据生成管道，并利用 DPO 微调提升六大开源 LLM 的文化安全性，同时保持通用能力。

**💡 创新点**

创新点在于：①将文化一致性定义为“社会法制锚定、人口特定化、拒绝时的情境建设”（P1‑P3）；②通过三阶段自适应红队、三模型安全回复池、专家式三人裁判过滤，实现本土化且可扩展的对齐数据；③在所有模型上实现安全率提升且无显著通用能力下降。

**🔧 技术方法**

核心技术包括：Prompt‑based seed 生成、自动化多代理红队、基于文化适配政策的多模型安全回复生成、Direct Preference Optimization (DPO) 与 QLoRA 微调、三人裁判过滤器。

**📊 数据集**

数据集为 10,000 条（查询、违规回复、合规回复）三元组，覆盖韩语特定的五大危害领域，全部通过管道自动生成并人工复核，未使用任何真实用户数据。

**📈 对比分析**

与原始模型对比，Korset 安全率平均提升 6.59 分（最高 16.58 分），大部分模型提升 10+ 分；KoBBQ、KMMLU、Ko‑MT‑Bench 等通用基准保持不变或略有提升，说明对齐训练未显著削弱通用能力。

**⚠️ 局限性**

局限性包括：①文化裁判和政策手册需要人工维护，难以实时跟踪法律及社会规范变化；②缺乏大规模人类评估，无法验证对齐效果是否满足所有文化子群体；③红队与安全回复生成可能泄露攻击模式，需进一步评估安全性。

---

## 59. The Economics of Proof-of-Useful-Work

**arXiv ID:** 2606.06700 | [PDF](https://arxiv.org/pdf/2606.06700v1)

**作者:** Rafael Pass `[一作]` `[通讯]` (Pearl Research Labs), Rafael Pass (Pearl Research Labs)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文建立了一个竞争均衡模型，用来分析在Proof‑of‑Useful‑Work（PoUW）区块链中，算力可以在纯挖矿、纯推理与双工（同时产生安全性与推理输出）三种活动之间分配时的均衡价格与资源配置，并给出完整的闭式解。

**💡 创新点**

创新点主要包括：① 用一个单一的经济参数（Token‑Inference Ratio θ）和双工效率指数 Δ，刻画了 PoUW 生态中的三种“世界”（Bitconia、Fortessia、Duplexia）并揭示其转折点；② 证明了 PoUW 并不会降低攻击成本，攻击成本始终为 PR/2；③ 量化了 PoUW 对社会福利的正向影响（补贴推理、降低推理价格、提升安全性），并给出了相对比 Bitcoin 的效用阈值。

**🔧 技术方法**

技术上主要采用了竞争均衡分析、Tullock 竞赛与 Bertrand 价格竞争的结合、闭式求解、以及对弹性需求的数学处理；同时利用 Fisher 交换方程为 θ 提供了货币学解释。

**📊 数据集**

本文为理论分析，未使用任何实测数据集；模型假设均为抽象的成本 e、区块奖励 R、需求曲线 𝒟(p)。

**📈 对比分析**

比较方法是对不同 θ 与 Δ 情形下的均衡进行分类，计算攻击成本、推理价格、区块奖励等指标，并将 PoUW 的结果与传统 PoW（Bitcoin）做理论对比；结果显示 PoUW 在安全性与有用计算上至少不劣于 PoW，并在某些参数区间内优于 PoW。

**⚠️ 局限性**

局限性包括：① 只考虑单期模型，未处理区块奖励减半与交易费；② 假设算力成本 e 恒定且需求曲线已给定，且假设需求弹性；③ 忽略了双工操作的实际实现难度与潜在摩擦；④ 对刀刃参数（Δ=0、θ=θ_low/θ_high）未给出稳健性分析；⑤ 仅关注经济均衡，未考虑动态博弈或技术演进的长期影响。

---

## 60. Queen-Bee Agents: A BeeSpec-Centered Architecture for Governed Enterprise MCP Orchestration

**arXiv ID:** 2606.06545 | [PDF](https://arxiv.org/pdf/2606.06545v1)

**作者:** Dutao Zhang `[一作]` (Macao Polytechnic University), Liaotian `[通讯]` (Macao Polytechnic University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了Queen‑Bee多代理架构，利用BeeSpec中间表示实现企业MCP环境下的有治理的任务规划与执行。

**💡 创新点**

创新点在于将控制平面与执行平面分离，构建可审计的BeeSpec执行边界，并通过租户范围工具调用和策略治理实现安全隔离与可追溯性。

**🔧 技术方法**

采用大型语言模型、检索驱动的能力检索、轻量级结构化检索器、策略引擎、MCP连接器、Python原型以及RDKit、ChEMBL等专业工具。

**📊 数据集**

使用59个人工构造的企业式任务集合（HR、IT、治理敏感、局部执行、化学工作流）进行评估，化学子任务还利用ChEMBL和PubChem等公开数据库。

**📈 对比分析**

与静态、检索、压力检索、混合检索、LLM驱动、无策略以及单代理基线七种系统对比，检索驱动Queen‑Bee在任务成功率0.964、治理完整通过、局部执行完整度0.95，并保持2.6ms的平均延迟；无策略与单代理在治理指标上失效。

**⚠️ 局限性**

局限性包括任务为合成实验而非真实生产、MCP栈仅在本地演示、治理仍为规则化、能力注册噪声为结构化而非开放世界，以及未提供正式安全保证。

---

## 61. Information Rate Decomposition for Noisy Nanopore Channels with Geometric Duplication

**arXiv ID:** 2606.06808 | [PDF](https://arxiv.org/pdf/2606.06808v1)

**作者:** Brendon McBain `[一作]` (Monash University), Emanuele Viterbo `[通讯]` (Monash University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了带有几何复制的噪声重复通道的可实现信息率，并提出了一种新的信息率分解，将通道记忆与同步不确定性分别量化；

**💡 创新点**

创新点在于引入了软对齐函数（类似Soft‑DTW）作为同步误差的量化指标，证明其满足强AEP，并通过该指标得到可计算的跳跃可靠性下界；

**🔧 技术方法**

主要技术包括：隐马尔可夫模型的熵率分析、Kingman子可加子演化定理、Soft‑DTW软对齐函数、信息稳定性与强AEP、随机长度熵理论；

**📊 数据集**

实验采用真实Oxford Nanopore技术的多代 pore level 模型（R9.2、R9.4、R10.4.1等），并使用均匀 i.u.d. 源进行仿真；

**📈 对比分析**

与传统的粘性通道容量上界及无噪声下的跳跃距离下界对比，发现跳跃可靠性下界在典型 SNR 范围内与仿真结果高度吻合，说明该下界在实际测序条件下是可行且具有竞争力的；

**⚠️ 局限性**

局限性在于跳跃可靠性下界只考虑了两段间的同步误差，无法完全捕捉更长段的全局对齐信息；此外，理论结果假设几何复制和AWGN，未涵盖更复杂的通道噪声或非几何复制分布。

---

## 62. Explain Like I'm 5 or Whatever I Choose: Evaluating the Interactive Potential of Language Model Responses

**arXiv ID:** 2606.06788 | [PDF](https://arxiv.org/pdf/2606.06788v1)

**作者:** Indu Panigrahi `[一作]` (University of Illinois Urbana-Champaign), Tal August `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种新的评估框架，针对科学信息检索任务中大语言模型（LLM）在不同语言复杂度水平下生成多版本回答的能力进行评估，并通过一次含16名受试者的形成性用户研究验证交互式复杂度控制的可行性，随后使用该框架对5种近期LLM在98个科学查询（以及扩展的459个查询）上生成5级复杂度回答进行实验，分析其在术语、信息量和长度等复杂度指标上的变化趋势。

**💡 创新点**

创新点在于：①将评估视角从单一回答转向多版本比较，关注模型在不同复杂度层级之间的一致性和可控性；②设计了基于“观众层级”提示的生成方式，并引入了交互式滑块界面；③提出了以指标变化方向为核心的评估标准，能够量化模型在生成多层级回答时是否按预期递增或递减。

**🔧 技术方法**

技术实现包括：基于Prompt Engineering使用“College student → Senior researcher”等观众标签生成5级回答；利用BERTScore对不同版本句子差异进行显著性检测；采用3个语言复杂度度量（术语比例、信息量、长度）；使用RAG流水线生成参考报告；评估模型包括GPT‑5.1、GPT‑5 mini、Claude Sonnet 4.5、Claude Sonnet 4.5 + Thinking、DeepSeek‑V3.1。

**📊 数据集**

使用的数据集为：①ScholarQA‑Multi（98个专家撰写的科学查询及对应回答）作为基准测试集；②在此基础上扩充至459个查询，使用Claude Sonnet 4.5生成的RAG报告作为输入；③为验证观众标签对结果的影响，还尝试了WIRED 5‑Level（Child, Teen, College, Grad, Expert）标签集。

**📈 对比分析**

评估方法为计算每个查询在5个复杂度层级之间的指标变化方向（是否正向递增），并汇总为“所有指标都递增”的百分比。结果显示所有模型在长度指标上表现最为一致，但在术语和信息量上往往出现递减或不一致的情况；Claude Sonnet 4.5在术语递增方面表现最佳，仅在约46%时按预期递增，整体一致性仍较低。

**⚠️ 局限性**

局限性包括：①仅采用词汇与句子长度等表层语言指标，未覆盖内容丰富性、类比、阐释等更深层的复杂度维度；②科学查询与部分观众标签（如Child）不匹配，可能影响评估真实性；③未测定用户对不同层级差异的可感知阈值，难以判断差异对用户体验的实际影响；④仅测试英文文本，缺乏对非母语或多语环境的考量。

---

## 63. AI Level of Detail: Distance-Aware ML Model Precision Selection for Real-Time Human Motion Prediction in Games

**arXiv ID:** 2606.06565 | [PDF](https://arxiv.org/pdf/2606.06565v1)

**作者:** Mathew Varghese `[一作]` `[通讯]` (University of Washington), Mathew Varghese (University of Washington)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种 AI LOD 框架，利用 NPC 与摄像机的距离动态切换不同精度（FP32、FP16、INT8）的预导出 ONNX 模型，以降低远距离 NPC 的推理计算量。

**💡 创新点**

创新点在于将推理精度作为 LOD 轴，实现基于距离的量化层级选择；该思路将传统几何 LOD 的替换原则迁移到机器学习模型推理上。

**🔧 技术方法**

使用的技术包括 ConvSeq2Seq 运动预测模型、ONNX Runtime 的 FP32/FP16/INT8 后训练量化、距离阈值选择器、以及基于视距缩放的感知实验。

**📊 数据集**

所使用的数据集为 CMU Motion Capture（Mocap）数据集，用于训练、评估与感知实验。

**📈 对比分析**

通过在 CPU 上对 FP32/FP16/INT8 三个层级进行批量推理实验，测量模型大小、延迟与误差：FP16 约 1.5× 加速、49% 内存缩减、误差几乎可忽略；INT8 约 9.8× 加速、73% 内存缩减、相对 L2 错误 0.117；感知实验表明，在规定的距离阈值内低精度模型与高精度模型无可感知差异。

**⚠️ 局限性**

局限性包括仅在 CPU 上评估且未考虑 GPU 性能差异；未完成完整游戏引擎集成；距离阈值为静态且需手动调校；感知实验规模有限；未探索更低精度（INT4）或动态阈值调整。

---

## 64. FIGMA: Towards FIne-Grained Music retrievAl

**arXiv ID:** 2606.06615 | [PDF](https://arxiv.org/pdf/2606.06615v1)

**作者:** Nishit Anand `[一作]` (University of Maryland), Ramani Duraiswami `[通讯]` (University of Maryland)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了FIGMA模型，专门解决细粒度音乐检索任务。

**💡 创新点**

创新点在于引入多视角对比损失，结合全局与帧级音频-文本对齐，突破CLAP对长描述的局限。

**🔧 技术方法**

使用MuQ音频编码器、E5多语种文本编码器，冻结主干仅训练轻量级投影头，并采用InfoNCE多视角对比损失。

**📊 数据集**

构建并使用大规模FGMCaps数据集（38万训练+1万测试），包含节奏、调性、和弦、节拍等音乐理论属性。

**📈 对比分析**

在MusicBench和FMACaps等基准上对比CLIP/CLAP等模型，FIGMA在R@1/5/10等指标上提升高达73.3%相对提升。

**⚠️ 局限性**

局限性包括仅在英语文本上评估，未验证跨语言检索，且帧级对齐使用max聚合可能无法精确捕捉所有属性。

---

## 65. Architecturally Significant MLOps Guidelines for ML Model Integration and Deployment: a Gray Literature Review

**arXiv ID:** 2606.06535 | [PDF](https://arxiv.org/pdf/2606.06535v1)

**作者:** Faezeh Amou Najafabad `[一作]`, Ilias Gerostathopoulos `[通讯]` (Vrije Universiteit Amsterdam)

**通讯引用:** 1428 | [OpenAlex ID](https://openalex.org/A5084021903)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过灰色文献回顾，系统提炼了25条架构显著的MLOps指南，用于机器学习模型的集成与部署。

**💡 创新点**

首次在实践驱动的灰色文献基础上，提出并分类5大类别的架构显著MLOps指南，填补了集成与部署阶段的指导空白。

**🔧 技术方法**

采用灰色文献综述、主题分析、三步指南合成、开放卡排序等定性方法。

**📊 数据集**

研究基于从Google搜索得到的331个网页，最终筛选103条灰色文献作为数据集。

**📈 对比分析**

本研究并未进行实验对比，而是通过对比频次和来源数量评估指南普及度，发现部署相关指南更为普及。

**⚠️ 局限性**

主要限制包括搜索范围局限于Google前两页、动态网页可用性变化、灰色文献质量不一以及缺乏实际案例验证。

---

## 66. The Piggyback Hypothesis of Generalization: Explaining and Mitigating Emergent Misalignment

**arXiv ID:** 2606.06667 | [PDF](https://arxiv.org/pdf/2606.06667v1)

**作者:** Jiachen Zhao `[一作]` (Northeastern University), Weiyan Shi `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究 LLM 在窄域微调后出现的广泛误导性（Emergent Misalignment）并提出 piggyback 机制解释其传播。

**💡 创新点**

提出 Token‑Regularized FineTuning (TReFT)，通过对前缀（或后缀）KV 表示的正则化抑制 EM，同时提供前缀修补 (KV patching) 与激活修补证据。

**🔧 技术方法**

利用前缀/后缀的 KV 正则化、表示修补与激活修补技术，结合对齐评估 (LLM‑as‑judge) 与 MT‑Bench 评测。

**📊 数据集**

使用多领域误导性数据集（金融、医疗、法律、汽车）和通用 EM 测试集，及 MT‑Bench 用于实用性评估。

**📈 对比分析**

在 Qwen‑2.5‑7B/32B、Llama‑3.1‑8B、GPT‑OSS‑20B 等模型上与数据插入、KL 正则化等方法对比，TReFT 在 EM‑F1 上提升 20%+，对齐分数大幅提升，且实用性损失最小。

**⚠️ 局限性**

局限性：仅关注前缀/后缀的 piggyback，未探究其它共享 token；机制解释尚不完整；需在更大规模模型和更多任务上进一步验证。

---

## 67. TA-RAG: Tone-Aware Retrieval-Augmented Generation for Peer-Support Health Communication

**arXiv ID:** 2606.06794 | [PDF](https://arxiv.org/pdf/2606.06794v1)

**作者:** Yong-Bin Kang `[一作]` (Swinburne University of Technology), Anthony McCosker `[通讯]` (Swinburne University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出轻量级基于提示的语调感知检索增强生成框架 TA‑RAG，用于敏感的同伴支持健康交流

**💡 创新点**

将语调拆解为四个可控层（去污名化、可读性、受众适配、同理心重写），并通过提示实现无微调的语调控制

**🔧 技术方法**

使用检索增强生成（RAG）与 GPT‑4o‑mini 等大语言模型，结合提示工程实现四个语调组件

**📊 数据集**

使用 HOLA 质量报告、UNAIDS 术语指南、NAPWHA 同伴支持标准以及公开的同理心文本数据集构建评测数据集

**📈 对比分析**

采用 ReplaceRate、SemSim、ContPreserve、FK/ARA 可读性评分、PeerAlign、EmScore 等指标评估各组件；结果显示各组件显著提升目标属性并保持高语义保留（0.86–0.98）

**⚠️ 局限性**

仅在组件级别评估，未进行端到端真实同伴支持情境测试；在可读性、同理心等方面存在语义保留下降；评估依赖 LLM 评分模型，可能受模型偏差影响

---

## 68. Direct 3D-Aware Object Insertion via Decomposed Visual Proxies

**arXiv ID:** 2606.06601 | [PDF](https://arxiv.org/pdf/2606.06601v1)

**作者:** Jingbo Gong `[一作]` (Nankai University), Chen Change Loy `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

实现了一种基于3D可视化代理的姿态可控对象插入框架，将插入过程分解为外观、几何和上下文三条独立路径。

**💡 创新点**

创新点包括：①将插入条件拆分为三种互补信号，并通过独立LoRA和位置编码实现分离注入；②利用单视图到3D代理的渲染作为稠密几何约束；③构建自动化数据生成管线，显著提升训练数据多样性。

**🔧 技术方法**

核心技术为：扩散生成（FLUX.1-Fill-dev），LoRA适配器与旋转位置编码（RoPE），Siglip全局上下文编码，图像到3D模型（如LRM/LGM），以及多尺度进展训练。

**📊 数据集**

使用的主要数据集为：自建的160k对混合数据集（65k来自SA‑1B的合成对 + 93k来自MVImgNet的真实对），并利用VLM+SAM3进行对象筛选，Qwen‑Image‑Edit生成视角变换。

**📈 对比分析**

在与Object3DIT+AnyDoor、TRELLIS+InsertAnything等基线的对比中，本文方法在Stable Diffusion与FLUX两种backbone上均取得最优结果：PSNR、SSIM、LPIPS、CLIP、DINO均提升；姿态匹配误差（Matching Error）显著降低，表明姿态控制更精准。

**⚠️ 局限性**

局限性：高度依赖上游图像‑到‑3D代理的几何质量；若代理重建失真（如形状畸形或尺寸错误），最终插入结果会随之失真，且无法完全弥补严重几何误差。

---

## 69. MMBU: A Massive Multi-modal Biomedical Understanding Benchmark to Probe the Perception Capabilities of Vision-Language Models

**arXiv ID:** 2606.06696 | [PDF](https://arxiv.org/pdf/2606.06696v1)

**作者:** Ryan D'Cunha `[一作]` (Stanford University), Serena Yeung-Levy `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了MMBU基准，收集了35个子模态、78K+医学影像，评估视觉语言模型在分类、分割、检测等多任务下的感知能力。

**💡 创新点**

提供最大规模、结构化元数据丰富、开放/封闭两种VQA、分割、检测等多任务统一评测框架，揭示医学适配仅带来有限提升。

**🔧 技术方法**

采用多阶段数据收集、元数据标准化、自动化与人工校准问题生成，并使用微平均F1、闭合/开放格式分隔评估方法。

**📊 数据集**

汇总122+公开数据集，最终选取78K图像，覆盖250分类、84分割、76检测任务，涉及光学显微、CT、MRI、电子显微等35子模态。

**📈 对比分析**

与15开源+2前沿VLM（如GPT‑4.1‑mini、Qwen2.5‑VL‑32B、InternVL3.5‑8B等）对比；闭合任务F1最高0.69，开放任务平均降幅约0.26，检测任务几乎无效。

**⚠️ 局限性**

模型在开放格式下表现差，定位/检测能力极弱，医学适配收益有限，跨数据集泛化差，且评测仍受人工标注成本与覆盖范围限制。

---

## 70. Attention-Guided Autoencoder Fusion for Insulator Defect Detection Using UAV Transmission-Line Imaging

**arXiv ID:** 2606.06536 | [PDF](https://arxiv.org/pdf/2606.06536v1)

**作者:** Malak Allam `[一作]` (MSA University), Ali Hamdi `[通讯]` (MSA University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出AE‑YOLO框架，结合注意力机制、轻量化瓶颈自编码器和FPN‑PAN融合，实现无人机图像中绝缘子缺陷的精准检测。

**💡 创新点**

创新点在于：①将自编码器嵌入多尺度融合中以保留异常信息；②使用方差最大化正则化提升稀缺缺陷的判别能力；③通过自编码器引导的置信度提升和加权盒子融合（WBF）提升召回率。

**🔧 技术方法**

采用CBAM注意力模块、轻量化瓶颈自编码器、FPN‑PAN、CIoU+Focal损失、WBF以及多模型集成的YOLOv8/10/11。

**📊 数据集**

使用公开的Insulator‑Defect Detection数据集，共1600张无人机图像，包含三类缺陷（污染闪络、破损）。

**📈 对比分析**

与YOLOv7、v8、v10、v11、v26等基线对比，AE‑YOLO在EfficientNetV2骨干下取得95.10% mAP@0.5、96.40%精度、93.80%召回，较最佳基线提升5点mAP、6.7点召回。

**⚠️ 局限性**

局限性：未评估推理速度与实时性；仅在单一数据集验证；缺乏各模块的独立消融实验；未测试不同绝缘子类型及多模态传感场景。

---

## 71. CARVE-Q: Quantum-Proposed, Classically Certified Interactive Driving Repair

**arXiv ID:** 2606.06531 | [PDF](https://arxiv.org/pdf/2606.06531v1)

**作者:** Yifan Wang `[一作]` (McGill University), Yifan Wang `[通讯]` (McGill University)

**通讯引用:** 3846 | [OpenAlex ID](https://openalex.org/A5100398537)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种将硬规则拒绝转化为可审计的互动修复证书的架构CARVE，并在其中嵌入验证器屏蔽的量子最小值搜索层；

**💡 创新点**

创新点在于：①将驾驶交互修复问题形式化为有限修复格的最优搜索；②引入权责加权成本、右侧通行权封包与回退策略的证明；③在黑盒验证器模型下使用量子最小值搜索实现 O(√M) 的查询复杂度；

**🔧 技术方法**

使用量子幅度放大与 Durr–Hoyer 最小值搜索、可逆相位门构造、经典验证器重评判定、以及规则书、RSS 等安全约束；

**📊 数据集**

使用 INTERACTION 交互回放数据（589 条），并结合 Lanelet2 语义地图；

**📈 对比分析**

与纯经典全枚举、局部搜索、随机搜索等方法对比，量子搜索在 8 车、4 种动作的格子上平均需 434 次相位门调用，保留 100% 的右侧通行权、责任一致性和回退可用性，RHA（人类对齐率）从 28.23% 提升至 41.82%；

**⚠️ 局限性**

局限在于：目前仅证明了黑盒查询优势，未展示实际硬件加速；对规则书结构化可利用的白盒求解器并未改进；量子模块仍受限于噪声与资源，无法直接取代经典验证器。

---

## 72. Interpreting Learning Under Competing Models: Joint and Stepwise Approaches for Dynamic Cognitive Diagnosis

**arXiv ID:** 2606.06804 | [PDF](https://arxiv.org/pdf/2606.06804v1)

**作者:** Yawen Ma `[一作]` (Lancaster University), Gabriel Wallin `[通讯]` (Lancaster University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

比较联合估计与分步估计在未知Q矩阵的动态认知诊断模型中的性能，通过文本信息引导的Q矩阵先验来分析阅读游戏数据并进行仿真验证。

**💡 创新点**

将文本嵌入作为Q矩阵先验加入动态CDM，并首次对联合与分步两种估计策略在未知Q矩阵下进行系统比较与仿真验证。

**🔧 技术方法**

Bayesian动态CDM（DINA）联合与三步式估计、SBERT文本嵌入先验、Markov链蒙特卡洛、偏差校正三步法。

**📊 数据集**

Amplify Boost Reading平台的两款阅读游戏（Punchline与Field Observer）在二年级与三年级共计1978名学生的项目数据。

**📈 对比分析**

通过联合与分步两种方法在相同文本先验下估计Q矩阵、掌握轨迹和转移参数，实证发现两者对整体学习趋势一致但对完全掌握人数差异显著，仿真显示联合估计在Q矩阵不确定或项目变化时更稳健，转移参数误差更低。

**⚠️ 局限性**

计算成本高、仅使用DINA模型、只针对二元属性、样本受限于选定项目、未考虑日志特征测量误差。

---

## 73. Evidence Graph Consistency in Retrieval-Augmented Generation: A Model-Dependent Analysis of Hallucination Detection

**arXiv ID:** 2606.06748 | [PDF](https://arxiv.org/pdf/2606.06748v1)

**作者:** Jianru Shen `[一作]` `[通讯]` (University of Montana), Jianru Shen (University of Montana)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估了一种基于证据图一致性的结构化幻觉检测框架 EGC。

**💡 创新点**

创新点在于将生成答案与检索证据建模为局部图，利用图结构特征判别幻觉，而非单一相似度。

**🔧 技术方法**

技术包括句子级语义编码（Sentence‑BERT）、余弦相似度阈值构造图、五个图一致性特征和逻辑回归分类器。

**📊 数据集**

使用 RAGTruth 量化 QA 子集（5,767 条响应，六款 LLM），对比各模型的幻觉率。

**📈 对比分析**

与传统的 Coverage‑only、Support‑only 等基线相比，EGC 在聚合 AUROC 上略逊 (0.556 vs 0.589)，但在模型族内可解释性强，方向校正后 AUROC 提升至 0.669。

**⚠️ 局限性**

局限包括阈值固定、句子分割粗糙、仅使用二值幻觉标签、仅在 QA 任务验证、跨模型泛化未知，且对 GPT 类模型的幻觉检测效果不佳。

---

## 74. Federated Foundation Models over Vehicular Networks

**arXiv ID:** 2606.06786 | [PDF](https://arxiv.org/pdf/2606.06786v1)

**作者:** Kasra Borazjani `[一作]`, Seyyedali Hosseinalipour `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出将多模态多任务联邦基础模型（M3T FedFMs）部署在车联网中，并给出了模块化架构与案例验证。

**💡 创新点**

首次系统化设计 M3T FedFM 的模块化架构，采用共享任务适配器与冲突无关梯度下降实现任务上线，并在车联网场景中验证其效果。

**🔧 技术方法**

结合联邦学习、参数高效微调（LoRA、提示、适配器、MoE）、共享任务适配器和 CAGrad 梯度冲突缓解技术，基于 CLIP 的多模态模型进行训练。

**📊 数据集**

使用 Waymo Open Dataset 的感知数据（摄像头、LiDAR、文本），针对 3D 语义分割、3D 目标检测、2D 视频分割、2D 目标检测 四项任务进行实验。

**📈 对比分析**

与 NTA、NTA+GR、NTA+CA、FedAdapt_no_CA 等基线在 FedAvg 聚合下对比，结果显示共享任务适配器显著提升所有任务性能，FedAdapt 在任务上线场景中表现最佳。

**⚠️ 局限性**

局限在于未充分解决车辆可用性、任务/硬件异构、时间变异数据等挑战；实验仅在 20 辆车、单任务上线的受限设置下进行，缺乏对极端移动、异构硬件和实时性评估。

---

## 75. Evidence-Based Intelligent Diagnostic and Therapeutic Visualization System with Large Language Models: Multi-Turn Interaction and Multimodal Treatment Plan Generation

**arXiv ID:** 2606.06869 | [PDF](https://arxiv.org/pdf/2606.06869v1)

**作者:** Yunhan Wang `[一作]` (Harbin Institute of Technology), Bolin Zhang `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个基于知识图谱、四层症状匹配和多轮主动提问的可视化中医诊断系统，并提供了多模态治疗方案展示。

**💡 创新点**

创新点在于将知识图谱驱动的可视化诊断路径、信息增益优化的主动提问策略和多模态治疗呈现相结合，实现了诊断过程透明化、交互性增强和信息载体多样化。

**🔧 技术方法**

使用技术包括Neo4j知识图谱、四层症状匹配（精确、语义、模糊、LLM验证）、遗传算法优化的主动提问、人工智能生成的治疗插画、三维经络-穴位模型以及结构化证据引用。

**📊 数据集**

使用的数据集为构建的知识图谱（241种证型、1263种症状、2485条关系）以及30个临床案例进行评估。

**📈 对比分析**

通过30个案例的自动配对比较，系统在诊断可信度（Cohen's d=1.82，p<0.001）、认知负荷下降（四个维度均显著）以及证据可信度评分（4.21 vs. 2.95）上均显著优于传统单轮文本方案。

**⚠️ 局限性**

局限性包括：知识图谱覆盖范围有限，模型对非标准表达的鲁棒性仍受限；LLM在医疗领域可能产生幻觉；实验规模相对有限，缺乏大规模临床验证。

---

## 76. DiBS: Diffusion-Informed Branch Selection

**arXiv ID:** 2606.06518 | [PDF](https://arxiv.org/pdf/2606.06518v1)

**作者:** Bo Liu `[一作]` (Jilin University), Fujun Han `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 21229 | [OpenAlex ID](https://openalex.org/A5100441502)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于离散扩散模型的分支选择策略DiBS，用于提升完整符号Sudoku求解器的搜索效率，保持求解完整性；

**💡 创新点**

创新点在于将生成式扩散模型作为分支顺序的概率引导器，仅在高杠杆二元分支点调用模型，避免完整搜索被剪枝或误导，同时提供理论证明其可达最优分支策略；

**🔧 技术方法**

结合了离散扩散模型（MDM/GPT‑2架构）、约束传播（CP）+最小剩余值（MRV）搜索框架，以及轻量级一致性得分；

**📊 数据集**

在Roy的17‑clue Sudoku基准集（约49,000个硬实例）以及一组1,000个可满足的3‑SAT实例上进行实验；

**📈 对比分析**

与传统的MRV、前向检查（FC）、最小剩余值+最近值（LCV）以及基于度数的分支策略对比，DiBS在平均/中位/95th分位节点数和回溯次数上实现了最高的削减（平均约92%减少），虽然GPU推理开销导致绝对求解时间未超越最强的基线，但在搜索成本上明显优于所有基线；

**⚠️ 局限性**

局限包括：需要GPU推理，模型调用成本在简单实例中可能抵消收益；只在二元分支点调用模型，可能未充分利用更深层次的全局信息；扩展到更大规模或不同类型的CSP仍需验证。

---

## 77. OpenSkill: Open-World Self-Evolution for LLM Agents

**arXiv ID:** 2606.06741 | [PDF](https://arxiv.org/pdf/2606.06741v1)

**作者:** Zhiling Yan `[一作]` (Lehigh University), Lichao Sun `[通讯]` (Lehigh University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

从仅有任务提示和开放世界资源开始，构建可迁移的技能和验证信号，实现自我进化。

**💡 创新点**

提出开放世界自我进化框架：开放资源检索、无监督自检器、诊断驱动的技能迭代，避免目标任务监督泄漏。

**🔧 技术方法**

检索增强生成、LLM推理、虚拟测试生成、诊断驱动迭代、技能转移。

**📊 数据集**

SkillsBench、SocialMaze、ScienceWorld 等公开基准。

**📈 对比分析**

与七种闭世界基线对比，平均通过率提升 8.9/8.8 点，在两目标代理上最高分别为 43.6%/42.1%，逼近人工水平。

**⚠️ 局限性**

受检索噪声、虚拟任务难度估计不足、开放世界检索成本与延迟等限制。

---

## 78. StageFrontier: Synchronization-Aware Stage Accounting for Distributed ML Training

**arXiv ID:** 2606.06751 | [PDF](https://arxiv.org/pdf/2606.06751v1)

**作者:** Boram Yoon `[一作]` (NVIDIA), Ville Kallioniemi `[通讯]` (NVIDIA)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

对分布式训练中的同步延迟进行持续监测，利用每个rank的粗粒度阶段时间向量计算前沿（frontier），从而精确拆分暴露的步骤耗时并定位首个出现延迟的阶段与rank。

**💡 创新点**

创新点在于：①通过前沿最大前缀递归实现对暴露步骤时间的精确加法拆分；②引入“同步等待曝光模型”和多维证据标签，实现无事件追踪的同步与非同步延迟区分；③提出低体积、无同步时钟的最小遥测契约，使信号可持续、低开销并可直接驱动更昂贵的深度分析器。

**🔧 技术方法**

主要技术包括：按阶段划分的顺序向量采集、前沿最大前缀与增量归约、Slack 识别、同步等待曝光模型、证据标签生成、基于 PyTorch 的轻量级实现与 NCCL/Gloo 传输。

**📊 数据集**

在 NVIDIA PyTorch 24.12 容器上使用 bf16 Transformer 训练集，规模覆盖 8、16、32、64、128 GPU，实验涵盖 DDP、FSDP、ZeRO‑1 等多种并行方案。

**📈 对比分析**

与传统的 per‑stage 最大/平均仪表盘相比，前沿计数在同步位移场景下恢复上游延迟的准确率为 100% 而传统方法为 0%；与 PyTorch Profiler、HTA、Nsight 的重叠分析相比，在 32‑rank 窗口下重构相同的阶段排名（top‑1/top‑2 100%），而且数据包仅 0.11 MB（相比 15.81 GB）；总体运行时开销低于 0.2% CPU‑wall、0.05% 事件通道；路由准确率 top‑2 达 100%，top‑1 约 80%（在 128‑rank 场景下 87/90）。

**⚠️ 局限性**

局限性：①仅在同质同步 DDP/类似模型（无显著异构角色或跨步预取）下能精确识别延迟；②无法单独判定根因，需配合更深度的 profiler；③在 forward‑device 触发的 host‑visible 延迟、共生路径、角色异构或缺失遥测时可能产生误路或降级；④对极端异步或多队列场景需要进一步扩展模型。

---

## 79. AgileOS: A GPU Operating System Layer for Protected CUDA Services

**arXiv ID:** 2606.06697 | [PDF](https://arxiv.org/pdf/2606.06697v1)

**作者:** Zhuoping Yang `[一作]` (Brown University), Peipei Zhou `[通讯]` (Brown University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

设计并实现了AgileOS，一个在GPU侧提供受保护CUDA服务的操作系统层，虚拟化CUDA库、隔离服务状态，并通过内存管理与PTX注入实现对GPU服务的保护。

**💡 创新点**

在库边界上虚拟化CUDA并将实际CUDA上下文托管给受信任的运行时，同时引入GPU内存分区和PTX级别的内存守卫，实现对GPU服务状态的硬件级隔离。

**🔧 技术方法**

CUDA虚拟内存管理、PTX注入与检查、API拦截与代理、模块化服务架构、受信任的库适配器、GPU内存分区与MMIO映射等技术。

**📊 数据集**

论文未使用具体数据集，验证原型时采用CUDA样例程序和回归测试。

**📈 对比分析**

目前仅在原型层面验证功能，未给出完整的性能基准对比，后续计划在完整CUDA兼容性与性能隔离方面开展评估。

**⚠️ 局限性**

API覆盖不完整、未实现完整cuFFT/PyTorch支持、缺乏性能与安全评估、无法防御宿主内核/驱动被攻破、未考虑物理攻击和侧信道等。

---

## 80. PhyRoGen: Synthetic Generation of Physical Robot Manipulation Puzzles Using Procedural Content Generation

**arXiv ID:** 2606.06569 | [PDF](https://arxiv.org/pdf/2606.06569v1)

**作者:** Lennart Julian Droß `[一作]` (Technical University of Berlin), Marc Toussaint `[通讯]` (Technical University of Berlin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了PhyRoGen框架，利用程序化内容生成技术自动合成可被机器人操作的物理拼图数据集，并通过仿真验证其可解性与可操纵性。

**💡 创新点**

将PCG专门用于生成具有交锁对象依赖的物理操作拼图，填补了现有机器人学习与TAMP生成器缺乏此类依赖关系的空白。

**🔧 技术方法**

结合Blender+Phobos生成URDF，使用采样式运动规划器（LA-RRT、RRT*、LBT-RRT、BIT*）进行解算，借助Robowflex+OMPL进行评测，并在PyBullet仿真中使用KUKA LBR iiwa进行机器人操控。

**📊 数据集**

自行合成了24个独特的物理拼图（Simple Sliders、Grid World、Continuous Space、Lockbox Random、Move N Times、Rooms），无使用公开数据集。

**📈 对比分析**

对每个拼图运行四种采样规划器，重复100次记录首次可行解时间和动作成本；LA-RRT在23/24场景100%成功并保持最低成本，BIT*也100%成功，RRT*与LBT-RRT仅能解决7/24和6/24场景。

**⚠️ 局限性**

难以生成窄通道环境、只能生成链式依赖而非树状结构、缺乏在线学习与难度自适应机制、未与其他生成框架做系统对比、实验仅在仿真环境完成，未验证真实机器人效果。

---

## 81. HKJudge: A Legal Discourse-Annotated Corpus for Interpreting What Courts Find, How They Reason, and What They Rule

**arXiv ID:** 2606.06679 | [PDF](https://arxiv.org/pdf/2606.06679v1)

**作者:** Xi Xuan `[一作]` (City University of Hong Kong), Chunyu Kit `[通讯]` (City University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了首个由法律语言学专家句层级标注的香港判决语篇数据集HKJudge，并在该数据集上定义了两层话语架构（句子级别的26个修辞角色和段落级别的三类量刑要素）；

**💡 创新点**

创新点在于：①提出细粒度的两层话语标注方案，兼顾事实、推理、裁判和其他功能；②首次提供包含约4,000份刑事判决、约290k句、约6.5M词的专家标注语料；③在此语料上基准评估多类模型，揭示模型规模与指令调优对法律推理的影响；

**🔧 技术方法**

技术包括：法律语言学专家标注工具、基于BERT的编码器（LegalBERT、NeuralJudge、ML-LJP、JurBERT）、开源LLM（LLaMA‑3.1、Qwen‑2.5）以及商业LLM（GPT‑4、Claude‑3.5、Claude‑Opus‑4、Gemini‑2.5）在零样本与微调下的实现；

**📊 数据集**

使用的主要数据集是HKJudge（4,000份香港刑事判决），涵盖五级法院、约292k句、26个修辞角色、三类量刑要素；

**📈 对比分析**

比较方法：在句子级别的修辞角色分类和段落级别的量刑要素抽取上，分别采用准确率、AUC、精确率、宏F1指标；结果显示：BERT模型相对薄弱；微调后的开源LLM（尤其是Qwen‑2.5‑72B）优于BERT；商业LLM在两项任务中表现最好，Claude‑Opus‑4与Gemini‑2.5‑Pro在不同指标上领跑；

**⚠️ 局限性**

局限性：仅覆盖香港刑事判决，难以直接迁移到民事或家事等领域；某些事实子类（如F9‑admission与F10‑assertion）边界模糊；仅抽取三类量刑要素，未涵盖停刑、社区服刑等其他判决类型。

---

## 82. Capturing non-Markovian dynamics in non-equilibrium stochastic systems using flow matching

**arXiv ID:** 2606.06658 | [PDF](https://arxiv.org/pdf/2606.06658v1)

**作者:** Bhargav Sriram Siddani `[一作]` (Lawrence Berkeley National Laboratory), Ishan Srivastava `[通讯]` (Lawrence Berkeley National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在粒子系统的粗粒化 SPDE 里，构建了一个生成性流匹配模型来学习非高斯、非马尔科夫性的粒子通量分布，并用它模拟了非相互作用布朗粒子的 Kramers 首通时间问题。

**💡 创新点**

创新点在于将流匹配技术直接应用于粒子通量的概率分布，并通过条件历史信息捕捉记忆效应，从而在短时动力学上显著优于传统正则化 Dean–Kawasaki 方程。

**🔧 技术方法**

使用了生成式流匹配（Conditional Flow Matching）框架，结合 DeepONet（k=0）和 Transformer（k>0）两种网络架构，以及学生 t 分布源来训练通量概率模型。

**📊 数据集**

数据集来自数千条布朗随机游走的单维粒子模拟，在100个有限体积单元内，每个单元平均 1–50 个粒子，且历史长度 k 随机取 0–10。

**📈 对比分析**

与正则化 DK 方程和直接布朗粒子模拟对比，非马尔科夫 FM 模型在早期时间段内更好地预测了平均粒子密度、空间分布以及高阶统计量，并减少了负密度出现；但随着时间推移，误差逐渐增大。

**⚠️ 局限性**

局限性包括：在长时间演化时预测偏差显著；相较于纯粒子模拟需要更高计算成本；以及对高维/复杂相互作用系统的推广仍需进一步验证。

---

## 83. Depth over Fidelity in Fixed-Budget Noisy Evolution Strategies

**arXiv ID:** 2606.06555 | [PDF](https://arxiv.org/pdf/2606.06555v1)

**作者:** Sichen Wang `[一作]` (Shenzhen MSU-BIT University), Zhipeng Lu `[通讯]` (Shenzhen MSU-BIT University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在固定评估预算下，提出深度优先（depth‑over‑fidelity）思想，将不确定性直接嵌入进选拔阶段，推出概率精英成员（PEM）与残差自举PEM（RB‑PEM）算法，并配备探测‑切换机制。

**💡 创新点**

创新点在于把噪声排名的不确定性视为可量化的期望权重（PEM），并通过残差池自举以接近单位评估成本实现Rao‑Blackwell化，从而在保持深度的同时降低更新方差；同时引入自适应探测‑切换以避免低噪声情形下的冗余开销。

**🔧 技术方法**

核心技术包括条件期望权重（PEM）计算、残差自举（RB‑PEM）估计、Wasserstein距离误差诊断、以及基于探测统计的阈值切换决策。

**📊 数据集**

主要在COCO bbob‑noisy测试套件（30函数×15实例×多维度），并在RL策略搜索（LQR、CartPole）与噪声超参优化等外部任务中进行验证。

**📈 对比分析**

与传统CMA‑ES、固定k重采样、UH‑CMA‑ES等基线对比，RB‑PEM在高误排名场景下在固定预算内实现更低的简单后悔值，probe‑switch在多任务中进一步提升稳健性；总体表现优于所有评估阶段去噪方案。

**⚠️ 局限性**

局限在于当排名稳定或噪声分布变化较大时，残差池匹配误差可能导致性能下降；对相关或非平稳噪声的处理仍不完善，且需要手动设定切换阈值。

---

## 84. Quantized AI Inference on Constrained Embedded Platforms for Small-Satellite Settings

**arXiv ID:** 2606.06528 | [PDF](https://arxiv.org/pdf/2606.06528v1)

**作者:** Carlos Rafael Tordoya Taquichiri `[一作]` (ZHAW School of Engineering), Pablo Ghiglino `[通讯]` (Klepsydra Technologies)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在极其受限的嵌入式平台（如 Cortex‑M33）上，对量化神经网络中的单个卷积层进行测量，比较标量/ALU 与 DSP/SIMD 优化路径以及 TensorFlow Lite Micro 的延迟，并测量 OpenAMP 交互的通信开销，构建一个可用于多核/异构推理估算的基线。

**💡 创新点**

提出以实测为基础的基线方法，定义基于 MAC 的时间指标 κ_b，用于快速估算同一平台、同一框架下不同卷积层的执行时间；同时将通信开销与计算开销结合，为后续显式协同（多核/加速器）评估提供完整参考。

**🔧 技术方法**

使用 Klepsydra AI 的量化算子与 CMSIS‑NN 的 DSP/SIMD 优化；TensorFlow Lite Micro 作为对照实现；Zephyr RTOS 环境下 OpenAMP 进行跨核通信；实验平台包括 Cortex‑M33、NOEL‑V、PolarFire、ZedBoard 等。

**📊 数据集**

基于 MobileNetV2 的嵌入式视觉模型，挑选一个 28×28×144 输入、32×1×1×144 过滤器的卷积层作为工作负载；未使用公开数据集，仅在该模型层上进行推理测量。

**📈 对比分析**

通过同一卷积层在不同实现路径和平台上直接测量延迟，得到 DSP 路径相较 ALU 约 3.74×加速；TFLite+CMSIS‑NN 约 119 ms；各平台原始延迟和归一化延迟可用于跨平台比较；OpenAMP 单向约 33.7 µs，往返约 109.9 µs。基线指标可用于估算完整模型推理时延并评估多核或加速器方案的收益。

**⚠️ 局限性**

局限性：仅测量单个卷积层，未覆盖完整网络的调度、层间缓冲与框架开销；基线不考虑异构加速器的内存传输和同步瓶颈；实验场景主要是极限资源平台，可能与真实边缘设备差异较大；量化精度与模型准确率未评估；所选模型/层具有一定的代表性，但缺乏多种任务和数据集的泛化验证。

---

## 85. HybridCodec: Fast Dual-Stream, Semantically Enhanced Neural Audio Codec

**arXiv ID:** 2606.06743 | [PDF](https://arxiv.org/pdf/2606.06743v1)

**作者:** Arjun Gangwar `[一作]` (Indian Institute of Technology), S Umesh `[通讯]` (Indian Institute of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 HybridCodec，一种双流神经音频编解码器，将语义蒸馏与双流结构相结合，实现高效语音离散化。

**💡 创新点**

创新点在于在双流架构中加入语义蒸馏，从而在不使用大型 SSL 编码器的前提下，保持强语义‑声学解耦，并实现 3 倍的推理速度提升。

**🔧 技术方法**

采用 CNN 编码器/解码器、VQ 与 RVQ 量化、语义蒸馏 L2 损失、GAN（MPD / MS‑STFT）以及量化与承诺损失等技术。

**📊 数据集**

在 960 小时的 LibriSpeech 语料上训练，并在 LibriSpeech Test‑Clean、SeedTTS‑en（OOV）以及 Common Voice French（零样本跨语种）上进行评估。

**📈 对比分析**

与 DAC、DAC‑Distill、DualCodec、Mimi 等基线对比，HybridCodec 在 LibriSpeech 上实现 15.36% 的最低 RVQ‑1 WER，声学质量与对齐度保持竞争力，并在推理时比 DualCodec 快 3 倍。

**⚠️ 局限性**

局限性包括与单流 DAC‑Distill 相比，声学重建略逊一筹；模型参数（约 1.6 亿）仍高于单流方法；跨语言泛化虽改善，但在更低资源语言上的表现尚待验证。

---

## 86. P-Cast Precision in FP8 Attention: Sink-Induced Collapse and the Optimality of S=2^8

**arXiv ID:** 2606.06521 | [PDF](https://arxiv.org/pdf/2606.06521v1)

**作者:** Reed Lau `[一作]` `[通讯]` (Tencent), Reed Lau (Tencent)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文对FP8 E4M3注意力计算中的精度瓶颈进行定量分析，提出通过逆序遍历KV块或使用S=256缩放即可消除“Attention Sink”导致的P值下溢；

**💡 创新点**

创新点在于给出闭式阈值Δ_c=6.93+ln S−δ_k、P值下溢比例F(Δ,S)=Φ(Δ+δ_k−6.93−ln S)以及dp(S)锯齿函数的构造，证明S=256同时满足位精度、最小量化步长与最大正常覆盖三条条件；

**🔧 技术方法**

采用极值理论、正态分布假设、FP8数值特性推导公式，并通过与FP32参考的kernel‑faithful仿真验证；

**📊 数据集**

使用人工生成的随机高斯得分（含k_sink=4的sink块）和多种Δ(4–13)、序列长度(512–16384)的实验数据；

**📈 对比分析**

对比Forward/S=1、Forward/S=256、Reverse/S=256、Forward/S=448等配置，发现当Δ≈5–9时，S=256或逆序可使MSE提升3–10×，且两者在P‑collapse区域达到相同的精度下限；

**⚠️ 局限性**

局限在于仅考虑单一sink块、静态全局缩放、E4M3格式，未评估多sink分布、动态缩放或端到端任务指标。

---

## 87. MalTree: Tracing Malware Evolution from Embeddings at Scale

**arXiv ID:** 2606.06570 | [PDF](https://arxiv.org/pdf/2606.06570v1)

**作者:** Akash Amalan `[一作]` (Delft University of Technology), Tom J. Viering `[通讯]` (Delft University of Technology)

**通讯引用:** 319 | [OpenAlex ID](https://openalex.org/A5037781450)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出MalTree框架，通过对100多万样本构建并验证基于嵌入的系统进化树，探索恶意软件家族的演化关系；

**💡 创新点**

创新点在于将生物进化学中的距离型系统发育方法（UPGMA、Neighbor‑Joining）与多模态嵌入（结构、行为、图像）相结合，并引入时间验证机制，首次在大规模恶意软件数据上实现高达87% 的时间一致性；

**🔧 技术方法**

技术包括多模态特征提取（LIEF、ANY.RUN、ResNet‑50）、向量融合与降维、基于距离矩阵的系统发育构建（RapidNJ）、Outgroup和Midpoint根置、时间一致性评估与漂移分析；

**📊 数据集**

使用来自MalwareBazaar、vx‑underground、VirusTotal 等公开数据库的103,883份恶意样本，涵盖538个家族，样本类型包括PE、ELF、DOS；

**📈 对比分析**

在多模态嵌入上进行线性分类验证，准确率高达94.9%；树构建结果在时间一致性指标上达到87.1%，远高于随机预期；与传统的单模态或小规模聚类方法相比，显著提升了演化关系的可信度；

**⚠️ 局限性**

局限性包括对AV标签噪声的依赖、RapidNJ近似带来的树结构误差、缺乏对网络进化（代码复用多源）的建模、以及对新样本动态增量更新的支持不足。

---

## 88. Agentic Large Language Models for Automated Structural Analysis of 3D Frame Systems

**arXiv ID:** 2606.06525 | [PDF](https://arxiv.org/pdf/2606.06525v1)

**作者:** Ziheng Geng `[一作]` (University of Miami), Minghui Cheng `[通讯]` (University of Miami)

**通讯引用:** 2264 | [OpenAlex ID](https://openalex.org/A5048125635)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于多代理LLM框架，能够从自然语言描述自动生成SAP2000脚本，实现三维框架结构的自动建模与分析。

**💡 创新点**

创新点在于将3D框架投影为2D网格并用矩阵记录层数，采用分层任务拆解与多代理协作，显著提升拓扑一致性与长推理准确性。

**🔧 技术方法**

使用的大型语言模型为GPT‑OSS 120B与Llama‑3.3 70B Instruct Turbo，并构建多代理架构、JSON中间格式、检查点机制以及脚本翻译步骤。

**📊 数据集**

使用了十个由高度不均、平面不对称及间断布局构成的三维框架基准集，覆盖了多种几何复杂度。

**📈 对比分析**

与GPT‑5.4、Gemini‑3.1 Pro直接生成脚本的基线相比，框架在所有基准上平均准确率达90%，基线完全失败；平均运行时间约175秒，单例成本约0.193美元。

**⚠️ 局限性**

局限性在于仅支持可划分为矩形网格的框架，无法处理非正交或曲面几何；仅完成静态分析，未覆盖动态或抗侧力系统。

---

## 89. JA-SIREN: Deterministic Initialization for Sinusoidal Networks via Spectral Matching

**arXiv ID:** 2606.06671 | [PDF](https://arxiv.org/pdf/2606.06671v1)

**作者:** Mohammed Alsakabi `[一作]` (Carnegie Mellon University), Ozan K. Tonguz `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于离散正弦变换与Jacobi‑Anger展开的确定性初始化方法JA‑SIREN，用于sinusoidal神经网络。

**💡 创新点**

创新点在于利用目标信号的DST系数与Bessel函数逆推得到闭式权重，完全消除随机初始化和额外超参数。

**🔧 技术方法**

使用了DST、Jacobi‑Anger展开、Bessel函数以及两层sin激活的MLP结构。

**📊 数据集**

在Kodak真彩色图像集以及Spoken Wikipedia音频数据集上进行实验验证。

**📈 对比分析**

与SIREN、FM‑SIREN、FM‑FINER等随机初始化方法对比，JA‑SIREN在Kodak平均PSNR提升21.3 dB，SSIM几乎达到1，并且所有实验无运行方差。

**⚠️ 局限性**

局限性包括目前仅针对两层网络，扩展到更高维度或更深网络仍需进一步研究。

---

## 90. Principles and Practice of Deep Representation Learning: or a Mathematical Theory of Memory

**arXiv ID:** 2606.06624 | [PDF](https://arxiv.org/pdf/2606.06624v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 91. AI-Driven Test Case Generation from Natural Language Requirements: A Survey of Techniques and Research Gaps

**arXiv ID:** 2606.06563 | [PDF](https://arxiv.org/pdf/2606.06563v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 92. Compatibility and Accuracy Verification of CADmesh-Based Complex Geometry Modeling in Geant4

**arXiv ID:** 2606.06508 | [PDF](https://arxiv.org/pdf/2606.06508v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 93. MacArena: Benchmarking Computer Use Agents on an Online macOS Environment

**arXiv ID:** 2606.06560 | [PDF](https://arxiv.org/pdf/2606.06560v1)

**作者:** Victor Muryn `[一作]` (MacPaw Research), Yehor Khodysko `[通讯]` (MacPaw Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 MacArena 基准，用 421 个手工验证的 macOS 任务评估计算机使用代理（CUA），覆盖 OSWorld、macOSWorld 任务及 49 个全新的 macOS‑原生任务，并在 Apple Silicon 虚拟机上统一执行与评估。

**💡 创新点**

创新点：①首次为 macOS 构建大规模交互基准；②所有任务均人工验证，保证可执行性与无歧义；③采用可访问性树 + 屏幕截图双重观测、执行式评估脚本；④通过与 OSWorld 的跨平台比较揭示 macOS 对现有模型的独特挑战；⑤开放源代码，便于复现与扩展。

**🔧 技术方法**

技术与方法：使用 Apple Virtualization 框架 + UTM 虚拟机；观察为截图与可访问性树；任务定义为 POMDP；动作空间涵盖 8 种鼠标、5 种键盘、3 种终端操作；评估脚本按功能完成度给出 0–1 分；对四个基线模型（UI‑TARS‑1.5‑7B、Qwen3‑VL‑2B、Qwen3‑VL‑4B、OpenAI CUA）进行 15 步/任务的实验。

**📊 数据集**

数据集：421 个任务，来源于 OSWorld（221）、macOSWorld（151）和自采 49 个 macOS‑native 任务；覆盖 20 个应用类别，包含单应用与多应用交互。

**📈 对比分析**

比较方法：以成功率（Success Rate, SR）为主要指标；在 OSWorld 子集、macOSWorld 子集和 MacArena 原生子集分别评测；结果显示 macOS 上同一任务集的 SR 均低于 Linux 环境，模型排名出现逆转，最高总体 SR 为 31.83%（OpenAI CUA）。

**⚠️ 局限性**

局限性：①任务仍需人工创建，难以扩展；②缺乏人类基准评估；③评估脚本复杂，易受实现细节影响；④macOS 任务多样性有限，未覆盖所有第三方软件；⑤对 Apple Silicon 兼容性依赖特定虚拟化实现。

---

## 94. CrowdMath: A Dataset of Crowdsourced Mathematical Research Discussions

**arXiv ID:** 2606.06526 | [PDF](https://arxiv.org/pdf/2606.06526v1)

**作者:** Sherin Muckatira `[一作]` (University Of Massachusetts Lowell), Anna Rumshisky `[通讯]` (University Of Massachusetts Lowell)

**通讯引用:** 3332 | [OpenAlex ID](https://openalex.org/A5071360545)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建 CrowdMath 数据集，记录 164 条多参与者讨论的进展链，并评估大型语言模型在协作式开放问题解决中的推理与角色识别能力。

**💡 创新点**

创新点在于：①将焦点从传统的最终答案或完整证明转向多步骤、协作式进度链；②为每条帖子提供功能角色标签（Progress、Proof、Erroneous、FindError 等），捕捉错误、质疑与修正的交互；③构建与公开的真实数学讨论相结合的基准。

**🔧 技术方法**

使用前沿 LLM（如 GPT‑5.4、Claude Opus 4.6、Grok 4.20 等）完成两项任务：帖子角色分类（macro‑F1 与准确率）和下一帖预测（多项选择），并通过标准提示与对齐示例进行评估。

**📊 数据集**

CrowdMath 数据集：由 MIT PRIMES–AoPS CrowdMath 项目产生的 2016–2025 年公开讨论日志，经专家标注后生成 164 条进展链，包含 538 条带标签帖子，涵盖 Proof、Progress、Erroneous、FindError 等角色。

**📈 对比分析**

比较方法：在标签分类上使用 macro‑F1 与准确率；在下一帖预测上使用 4 选多项选择并计算准确率。结果显示模型在预测讨论流向上表现优异（82.7%–87.7%），但在区分 Proof 与 Progress 等细粒度角色时仅达 0.33–0.42 的 macro‑F1，表明对过程层面推理的掌握仍有限。

**⚠️ 局限性**

局限性：①数据以已完成结果为中心，未覆盖未实现的探索性推理；②标签粒度粗糙，未细分错误类型或论证严谨度；③缺少跨帖子 span‑level 标注，难以进行细粒度推理；④帖子可能关联多条链，若不注意拆分会泄漏信息；⑤标注者单一，无法评估标注一致性；⑥讨论以高中/本科水平为主，可能不适用于专业研究或形式化证明场景。

---

## 95. When to Think Deeply: Inhibitory Deliberation for LLM Reasoning

**arXiv ID:** 2606.06745 | [PDF](https://arxiv.org/pdf/2606.06745v1)

**作者:** Zhixuan He `[一作]` (University of Birmingham), Yue Feng `[通讯]` (University of Birmingham)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一种基于响应条件的抑制式深度推理框架 IDPR，先生成直观快答后决定是否调用慢推理。

**💡 创新点**

创新点在于把快答本身作为路由条件，利用抑制控制器结合置信度、可解析性等特征判断是否需要激活慢推理，从而在保持准确率提升的同时显著降低慢推理调用比例。

**🔧 技术方法**

采用 Qwen2.5‑Math‑7B 作为快答模型，OpenR1‑Distill‑7B 作为慢推理模型，构建抑制控制器并用回归+分类三头网络预测慢推理收益与成本，结合阈值校准实现成本‑收益平衡。

**📊 数据集**

使用 GSM8K 与 Mixture‑of‑Thoughts（MoT）数学推理子集构造快慢配对数据集，随后在 5,000 例 held‑out 测试集上进行评估。

**📈 对比分析**

在同 8.2% 慢推理调用率的“same‑budget”基线下，与随机路由和基于置信度的路由进行对比，IDPR 的准确率从 47.90% 提升至 48.92%，校正精度最高（27.07%），显示出更有效的慢推理选择。

**⚠️ 局限性**

局限性包括慢推理仍显昂贵，导致整体成本‑效益不如始终使用快答；阈值和控制器需针对不同模型、解码策略和数据分布重新校准；实验仅在特定快慢模型配对与数学推理任务上验证。

---

## 96. Robots Need More than VLA and World Models

**arXiv ID:** 2606.06556 | [PDF](https://arxiv.org/pdf/2606.06556v1)

**作者:** Elis Karcini `[一作]` (Motoniq.ai), Haitham Bou-Ammar `[通讯]` (Ucl Centre For Ai)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文认为通用机器人智能的瓶颈在于缺乏将世界中大量未标注的物理行为数据转换为可直接用于机器人学习的监督信息，而非单纯扩展 VLA 模型规模。

**💡 创新点**

创新点在于提出四个缺失的基础组件——物理数据引擎、跨体态保真重定向、物理约束世界模型和自适应部署循环——以及将其视为“物理智能堆栈”的框架。

**🔧 技术方法**

主要技术包括多模态自标注（embodied autolabelling）、任务保真跨体态映射、基于物理约束的 3D 世界模型预测以及任务条件化奖励与部署回馈机制。

**📊 数据集**

论文综述了多种现有数据集，如 RoboNet、BridgeData、DROID、RH20T、Meta-World 等，但未使用单一新数据集进行实验，重点在于对已有资料的系统性归纳。

**📈 对比分析**

由于是观点性综述，未给出量化对比；作者通过对比 VLA 扩展与现有机器人数据集的局限性，指出仅靠策略规模提升难以突破数据标注与物理约束的瓶颈。

**⚠️ 局限性**

局限性包括：缺乏针对多模态对齐与自标注的统一理论与算法，仍需大量人工验证；跨体态映射和物理世界模型的泛化与可解释性不足；以及在真实部署中实现闭环回馈所需的复杂基础设施。

---

## 97. Architecture-Adaptive Uncertainty Fusion for Deepfake Detection

**arXiv ID:** 2606.06666 | [PDF](https://arxiv.org/pdf/2606.06666v1)

**作者:** Ritesh Sharma `[一作]`, Yuichi Motai `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种后置、架构自适应的多源不确定性融合框架（COF），通过最大化预测错误与融合不确定性之间的Pearson相关性来为深度伪造检测模型生成可靠的不确定性信号。

**💡 创新点**

创新点在于将不确定性融合视为在概率单纯形上求解的凸优化问题，实现线性权重学习，既保持可解释性，又在跨域环境下显著提升不确定性预测的稳健性；同时提供了多种轻量级和零训练版本，方便实际部署。

**🔧 技术方法**

技术手段包括MC Dropout（epistemic/aleatoric）、温度缩放（calibration）、分布式非一致性（conformal）、马氏距离（distributional）等五种不确定性源；权重通过SLSQP多起点优化在单纯形上求解；使用正则化、稀疏化、层级分组等变体。

**📊 数据集**

实验使用FaceForensics++作为训练和验证数据集，CelebDF和DFDC用于跨域评估；覆盖了11种主流网络架构（CNN、EfficientNet、Vision Transformer、Hybrid）。

**📈 对比分析**

与随机森林、深度集成、Evidential DL等传统基线以及自身的线性/非线性变体进行对比；在11种架构上COF平均Pearson相关性为0.438，在CelebDF跨域环境下提升至0.116，显著高于RF的0.071，并在大多数模型上获得更高的跨域鲁棒性。

**⚠️ 局限性**

局限性包括：在错误率极低或验证样本不足时相关性估计不稳健；对高度非线性的错误-不确定性关系不一定最优；跨域衰减仍然严重，仍需开发域自适应或在线调优机制。

---

## 98. LinkNav: Surfacing Interconnected Information in Scientific Articles

**arXiv ID:** 2606.06650 | [PDF](https://arxiv.org/pdf/2606.06650v1)

**作者:** Sebastian Joseph `[一作]` (University of Texas at Austin), Ani Nenkova `[通讯]` (Adobe Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 LinkNav，一种通过生成问题并检索答案在文档内部显式连接段落的阅读增强系统。

**💡 创新点**

创新点在于结合问题生成与检索验证，自动发现非相邻语义连接，并在界面中实时展示链接与答案。

**🔧 技术方法**

使用的技术包括大型语言模型（如 GPT‑4o、o3、Llama‑4）生成问题，OpenAI embeddings 做相似度检索，LLM 判定答案与分段。

**📊 数据集**

实验使用 PeerQA 数据集的 PeerQA‑Gold 子集（29 篇）以及原始论文提交版本进行评估。

**📈 对比分析**

在不同模型的对比下，Llama‑4 在连接数量和距离上与 GPT‑4o 相近，检索召回率约 72%，精确率 0.80，显示系统能在较低成本下实现可用连接。

**⚠️ 局限性**

局限包括大量无效问题导致资源浪费、对答案前置或相邻段落的连接被过滤、依赖高质量 LLM 及成本，以及仅在小规模数据集上评估。

---

## 99. Competing Auctions in Intermediated Markets

**arXiv ID:** 2606.06633 | [PDF](https://arxiv.org/pdf/2606.06633v1)

**作者:** Bruno Mazorra `[一作]` (Flashbots), Christoph Schlegel `[通讯]` (Flashbots)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对单卖家多渠道竞价市场中不同拍卖机制的相互作用进行理论分析，重点研究以太坊的Proposer‑Builder Separation（ePBS）如何影响sealed first‑price、second‑price及English拍卖的展开与信息泄露。

**💡 创新点**

证明在单homing和multi‑homing情形下，第二价拍卖会完全或部分崩溃为第一价拍卖；阐明信息泄露的最优与非最优条件；揭示对Relay市场设计空间的约束；考虑Sybil参与如何恢复第二价拍卖。

**🔧 技术方法**

采用博弈论工具：Bayes–Nash均衡、顺序均衡、对称化与递归、虚拟值函数与正则分布等，结合定理与命题证明完成分析。

**📊 数据集**

未使用实验数据，论文完全基于理论模型，借鉴以太坊MEV生态的结构与机制描述。

**📈 对比分析**

通过理论推导比较不同拍卖格式的收益、分配与信息泄露效应；证明sealed first‑price在多数情况下更稳定、收益更高，缺乏数值仿真或实验比较。

**⚠️ 局限性**

局限包括：未考虑博弈的重复与声誉机制、协作与合谋、搜索者-Builder间的共同价值、入口与设计问题，以及对可信披露机制的正式建模。

---

## 100. Mind the Gap: Bridging Behavioral Silos with LLMs in Multi-Vertical Recommendations

**arXiv ID:** 2606.06779 | [PDF](https://arxiv.org/pdf/2606.06779v1)

**作者:** Nimesh Sinha `[一作]` (DoorDash Inc.), Sudeep Das `[通讯]` (DoorDash Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过层级检索增强生成（Hierarchical RAG）方法，用大语言模型合成用户在餐厅订单与搜索查询中的稀疏高维偏好特征，并将这些特征注入多任务学习（MTL）排名模型，以解决多垂直平台中的冷启动问题。

**💡 创新点**

创新点在于：① 使用层级RAG框架逐层细化并生成准确的商品分类偏好，避免生成幻觉；② 将LLM生成的稀疏特征与传统的交互特征融合到MTL模型中，实现跨垂直知识迁移；③ 通过提示工程与缓存技术降低LLM推理成本。

**🔧 技术方法**

使用的技术包括：大型语言模型（GPT‑4o / GPT‑4o‑mini）、层级检索增强生成（Hierarchical RAG）、提示工程（Chain‑of‑Thought、低温推理）、特征缓存与即时生成、两塔检索、Multi‑Task Learning（MTL）排名模型。

**📊 数据集**

数据集来源为 DoorDash 平台的历史餐厅订单和搜索查询，采样范围为用户最近三个月的交互记录，覆盖多种垂直领域（餐饮、杂货、零售等）。

**📈 对比分析**

方法通过与仅基于交互特征的基线 MTL 模型进行离线 AUC‑ROC/MRR 对比以及在线影子实验，整体提升约 4.4%‑5.2% 的 AUC‑ROC 和 4.8%‑5.2% 的 MRR；冷启动用户在餐厅订单信号上提升更显著，搜索查询信号在活跃用户上提升尤为突出。

**⚠️ 局限性**

局限性包括：① 对大语言模型的依赖导致推理成本和实时性挑战；② 生成特征的质量仍受提示设计和模型版本影响，存在潜在误判；③ 仅在现有业务垂直间迁移，跨域迁移的因果解释性与可扩展性尚待进一步验证。

---

## 101. Lean4Agent: Formal Modeling and Verification for Agent Workflow and Trajectory

**arXiv ID:** 2606.06523 | [PDF](https://arxiv.org/pdf/2606.06523v1)

**作者:** Ruida Wang `[一作]` (University of Illinois Urbana Champaign), Tong Zhang `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于Lean4的三层框架，对LLM代理工作流的结构、语义与执行轨迹进行形式化建模与验证，并基于验证结果实现工作流演进。

**💡 创新点**

首次将依赖类型的形式化语言应用于统一建模、验证LLM代理工作流，提出可扩展的Lean4库及基于轨迹的工作流演进方法。

**🔧 技术方法**

Lean4依赖类型编程、Hoare逻辑、LLMExec假设、LLM-as-judge与外部验证器。

**📊 数据集**

SWE-Bench-Verified（硬核子集）与ELAIP-Bench（问答子集）数据集。

**📈 对比分析**

在5种主流LLM上对通过/未通过验证的工作流进行对比，验证通过的工作流在SWE上平均提升14.8%、在ELAIP上9.07%；LeanEvolve进一步提升SWE平均7.47%。

**⚠️ 局限性**

依赖LLMExec局部正确性假设，预测器手工定义，且对大规模动态环境反馈支持有限。

---

## 102. FAIR-Calib: Frontier-Aware Instability-Reweighted Calibration for Post-Training Quantization of Diffusion Large Language Models

**arXiv ID:** 2606.06547 | [PDF](https://arxiv.org/pdf/2606.06547v1)

**作者:** Haoyu Huang `[一作]` (Beihang University), Baochang Zhang `[通讯]` (Beihang University)

**通讯引用:** 14019 | [OpenAlex ID](https://openalex.org/A5015525872)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对扩散式大型语言模型（dLLMs）的后训练量化，提出了两阶段的 Frontier-Aware Instability-Reweighted Calibration（FAIR-Calib）方法，以降低不可逆写入导致的错误放大。

**💡 创新点**

创新点在于：1) 通过教师模型的随机写入探测，构造位置先验，融合写入前沿和掩码阶段可靠性；2) 在无需完整扩散推理的离线校准中使用加权隐藏状态均方误差作为 KL 散度的代理，显著提升写入前沿对齐并抑制错误放大。

**🔧 技术方法**

技术手段包括：后训练量化（Affine Flattening、W4A4 低位量化）、基于随机写入的教师探测、时间×位置权重构造、离线层级加权隐藏状态 MSE 校准，以及理论证明将 KL 散度上界为加权隐藏状态误差。

**📊 数据集**

使用的评测数据集：LLaDA（Base、Instruct、1.5）和 Dream（Base、Instruct）系列模型在 10 个公共基准（PIQA、BoolQ、WinoGrande、ARC-E/C、HellaSwag、TruthfulQA、MMLU、HumanEval、GSM8K）上的性能。

**📈 对比分析**

对比方法包括 RTN、QuaRot、FlatQuant 等现有 PTQ 方法。实验显示 FAIR-Calib 在所有基准上均取得平均 1.5–3% 的准确率提升，并显著降低写入前沿翻转次数（从 2.9 次降至 1.9 次）和后续误差放大。

**⚠️ 局限性**

局限性在于：1) 仍需在量化后对特定推理策略（如不同写入策略）进行微调；2) 对极低位量化（如 W2A2）尚未验证效果；3) 理论假设（随机写入、模型独立性）在实际推理策略下可能不完全成立，导致部分误差未被完全抑制。

---

## 103. Enhancing Malware Detection with Generative AI: Using Variational Autoencoders to Boost Machine Learning Classifiers' Performance

**arXiv ID:** 2606.06501 | [PDF](https://arxiv.org/pdf/2606.06501v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 104. SCOUT: Semantic scene COverage via Uncertainty-guided Traversal

**arXiv ID:** 2606.06721 | [PDF](https://arxiv.org/pdf/2606.06721v1)

**作者:** Junyu Mao `[一作]` (Nokia Bell Labs), Matthew Andrews `[通讯]` (Nokia Bell Labs)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种在线语义探索框架SCOUT，能够在机器人巡视过程中实时构建不确定性感知的3D场景图，并利用该图指导下一步视角采集。

**💡 创新点**

创新点在于把感知与语义图构建耦合起来，实现基于语义不确定性的主动视角规划；同时设计了概率场景图生成器（PSGG）与不确定性引导的遍历（UGT）两大模块，形成闭环的主动认知系统。

**🔧 技术方法**

技术包括：Grounding DINO + SAM + CLIP进行开放词汇目标检测与语义嵌入；概率场景图更新与贝叶斯融合；基于信息增益和几何覆盖率的视角评分；A*路径规划；以及3D点云融合与射线可见性分析。

**📊 数据集**

数据集：在Gazebo仿真环境中构造了两个手工建模的室内场景（simple 与 challenging），共13个物体，未使用公开数据集。

**📈 对比分析**

与传统的lawnmower（几何遍历 + 后处理场景图）基线比较。SCOUT在节点检测上达到100%精度/召回（simple）和93%/94%（challenging），边关系召回在简单场景为62%（相对lawnmower的48%），节点召回明显优于基线；但在挑战场景中未能在预算内收敛，边关系表现略差。

**⚠️ 局限性**

局限性：在复杂场景中无法在限定视角数内收敛；边关系召回受邻接阈值影响；视角定位误差导致部分视角重复，影响不确定性减少；系统目前仅在仿真中验证，缺乏真实世界部署与大规模场景测试。

---

## 105. How Language Models Fail: Token-Level Signatures of Committed and Persistent Reasoning Failures

**arXiv ID:** 2606.06635 | [PDF](https://arxiv.org/pdf/2606.06635v1)

**作者:** Tanvi Thoria `[一作]` (Stanford University), Mykel J. Kochenderfer `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于链式推理（CoT）中 token 级不确定性信号的双模式失败检测框架，用于识别语言模型推理过程中的两类错误：先验错误（Committed Failure）和持续不确定性（Persistent Uncertainty），并通过“承诺点”定位错误发生位置。

**💡 创新点**

创新点在于：①将失败视为两种可区分的过程而非单一二元标签；②利用单个完成的 log‑probability 计算熵、边际差、NLL、核数、近似平衡等多种 token‑级指标，提取出能够捕捉“承诺点”与“持续不确定性”特征；③通过 PR‑AUC 曲线的倒 U 型与单调上升特征实现自动化模式分类。

**🔧 技术方法**

技术方法包括：对 CoT 生成的 token 序列按前缀窗口（{128,256,400,512,1024,2048}）计算平均与最大不确定性特征；使用 5‑折对数回归分类器评估 PR‑AUC；通过 Bootstrap CI、Stouffer’s Z、meta‑analysis 等统计检验验证模式的可重复性；与自我一致性（Self‑Consistency）比较并结合使用。

**📊 数据集**

实验数据集涵盖数学、科学、逻辑与编码四类：GSM8K、MATH‑500、GPQA Diamond、LiveCodeBench；模型覆盖 Qwen、Llama、Gemma、GPT‑OSS、Gemini 等多家族和不同规模，累计 23 个（model, dataset）配置。

**📈 对比分析**

比较方法为 PR‑AUC（单标签评估）与 self‑consistency 的 skip‑rate/recall 曲线；结果显示 23 组中 20 组满足双向预测（14 组承诺，9 组持续），PR‑AUC 在承诺模式下出现倒 U 型峰值，持续模式则单调提升；与自我一致性相比，单完成不确定性信号提升约 0.03‑0.05 PR‑AUC，表明两者互补。

**⚠️ 局限性**

局限性包括：①需要 15–60% 的错误率和 AUROC ≥0.55 的配置才能可靠；②仅利用单次完成，承诺点定位受窗口粗粒度影响；③在 GPT‑4o、Gemini‑2.5Pro 等只提供 top‑20 log‑probs 的 API 下，均值特征受限；④未验证内部计算与可见 CoT 的对应关系；⑤自我一致性实验仅在 GPQA Diamond 上进行，缺乏更广泛的验证。

---

## 106. Alternative Inductive Proof of Dilworth's Theorem

**arXiv ID:** 2606.06513 | [PDF](https://arxiv.org/pdf/2606.06513v1)

**作者:** Tao Zhang `[一作]` (Beijing Jiaotong University), Tao Zhang `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 17500 | [OpenAlex ID](https://openalex.org/A5100375792)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

提供了一种基于合法合并引理的归纳证明，证明了 Dilworth 定理：有限偏序集合的链数等于最大逆序集大小。

**💡 创新点**

创新点在于将证明过程转化为一个带颜色的合法合并算法，借助单色段合并的概念，使证明更具算法直觉。

**🔧 技术方法**

主要技术包括：归纳法、链分解、颜色编码、合法合并引理以及单色段的合并操作。

**📊 数据集**

该论文为理论性研究，不涉及任何实验数据集。

**📈 对比分析**

与传统的组合学证明相比，本文通过算法化的视角提供了更直观的证明；未进行实验比较，主要在理论上展示了证明的完整性与简洁性。

**⚠️ 局限性**

局限性在于：仅给出了理论证明，没有给出具体实现细节或计算复杂度分析；适用于有限偏序集合，未讨论无限情形或实际应用中的算法效率。

---

## 107. Ablation Study of Block Size, Weight Precision, and Scale Precision in NVFP4 Inference for Low-Power Edge-Efficient Neural Networks

**arXiv ID:** 2606.06527 | [PDF](https://arxiv.org/pdf/2606.06527v1)

**作者:** Ovishake Sen `[一作]` (University of Florida), Baibhab Chatterjee `[通讯]` (University of Florida)

**通讯引用:** 1899 | [OpenAlex ID](https://openalex.org/A5039890044)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了一种基于NVFP4浮点4位表示的LUT加速器NVLUT，并通过两级缩放、块尺寸、权重量化及选择性ECC进行能耗与可靠性优化

**💡 创新点**

创新点包括：①将乘法拆分为符号、指数、尾数三路，并将尾数运算映射为小型查表；②采用FP32张量缩放与FP8块缩放的两级缩放恢复动态范围；③对符号/指数采用ECC保护、对尾数采用低压存储的选择性可靠性策略；④系统化的块尺寸与权重量化消融研究，验证B=16为最优折中点

**🔧 技术方法**

技术实现包括：NVFP4数据格式、FP32张量缩放与FP8块缩放、LUT‑based mantissa乘法、BCH ECC保护、低压SRAM存储、混合电压操作以及FP4/FP8/FP16权重量化与微调

**📊 数据集**

使用六个边缘级模型（ResNet18、MobileNetV3‑Large、MobileNetV4‑Conv‑Small、MobileViT、ShuffleNetV2、EfficientNet‑Lite0）在Tiny ImageNet 224×224分类数据集上进行评估

**📈 对比分析**

对比传统LUT、数组乘法器和Wallace‑tree乘法器，NVLUT在ECC+低压和混合电压两种方案下，能耗分别下降至传统LUT的26.85×和22.85×，面积分别下降至2.21×和1.52×，并且在B=16、FP4权重、重训练的条件下，精度损失仅为2.8%以下

**⚠️ 局限性**

局限性包括：实验仅覆盖图像分类任务，未验证对检测/分割/语言等任务的适用性；需要完整的加速器实现（功耗、时延、LDO效率）进一步验证；混合精度策略和块尺寸的自适应调度仍待探索

---

## 108. Position: Don't Just "Fix it in Post": A Science of AI Must Study Training Dynamics

**arXiv ID:** 2606.06533 | [PDF](https://arxiv.org/pdf/2606.06533v1)

**作者:** Stella Biderman `[一作]` (EleutherAI), Naomi Saphra `[通讯]` (Boston University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出“训练动态科学”视角，批评传统的“post‑hoc 修复”思维，系统综述机制解释、偏见、公平、记忆、简化偏差等领域的进展，并指出关键开放问题。

**💡 创新点**

从历史哲学和科学方法论角度引入AI科学的六大特征；提出基于训练动态的三层进展框架（预测 → 干预 → 设计）；强调将训练过程视为时间演化系统，以实现对AI行为的因果解释。

**🔧 技术方法**

主要使用文献综述与案例分析，引用规模定律、因果推断、训练轨迹分析、机制解释等已有技术来支持论点；未开发新算法，而是梳理和整合现有研究方法。

**📊 数据集**

参考公开模型（如 GPT‑2/3、LLaMA、Pythia 等）及其公开训练集（Common Crawl、Wikipedia、Reddit 等）作为案例；论文自身未构建或使用新的数据集。

**📈 对比分析**

作为位置论文未给出统一实验结果；通过引用已有研究展示了在预测（如损失、记忆率）、干预（如数据重构、参数剪枝）和设计（如训练策略调整）等方面的进展，但未给出整体性能度量或对比。

**⚠️ 局限性**

局限性：论点依赖已有研究的不足，缺乏系统实验验证；训练动态理论尚未成熟，预测精度不确定；跨语言、跨文化的适用性尚未验证；对具体实现细节的指导性不强。

---

## 109. Skip a Layer or Loop It? Learning Program-of-Layers in LLMs

**arXiv ID:** 2606.06574 | [PDF](https://arxiv.org/pdf/2606.06574v1)

**作者:** Ziyue Li `[一作]` (University of Maryland), Tianyi Zhou `[通讯]` (MBZUAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大语言模型（LLM）在推理时的可变深度执行，提出将预训练层视为可重用函数，并通过动态选择和循环层段生成输入特定的程序-of-layers。

**💡 创新点**

创新点在于：①利用蒙特卡罗树搜索（MCTS）探测多种有效程序；②训练轻量级预测网络一次性输出层段划分与操作（skip/keep/repeat）；③实现无需微调即可在冻结模型上实现跳过与循环，显著提升推理效率与准确性。

**🔧 技术方法**

主要技术包括：MCTS搜索、跨层注意力预测网络、分段与操作标记、beam搜索解码、与冻结LLM的无缝集成。

**📊 数据集**

实验使用了 DART‑Math（5 难度级别）作为内部基准，并在外部算数/推理任务 ASDiv、MAWPS、MMLU‑Pro 等跨领域数据集进行泛化评估。

**📈 对比分析**

与标准推理、随机采样、DR.LLM、ShortGPT、MindSkip、FlexiDepth 等动态深度方法对比，pass@k 指标显著提升。例如在 LLaMA‑3.2‑3B 上 DM‑5 的 pass@1 从 30.6% 提升至 68.4%，且往往使用更少层或更低延迟。

**⚠️ 局限性**

局限性：仅在数学/算数推理任务上验证；预测网络受限于短连续段与单次循环，可能无法处理更复杂的程序结构；跨域迁移效果虽良好，但在非算数任务的鲁棒性仍需进一步评估。

---

## 110. What Matters When Cotraining Robot Manipulation Policies on Everyday Human Videos?

**arXiv ID:** 2606.06627 | [PDF](https://arxiv.org/pdf/2606.06627v1)

**作者:** Richard Li `[一作]` (Massachusetts Institute of Technology), Pulkit Agrawal `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了TriHands数据集（532段日常人类视频，28小时高质量三角化手势标签），并提出了一套跨身体共训练框架，利用图像尺度对齐、动作空间对齐、token级融合和专属动作编码/解码器等技术，将人类视频知识迁移到6个真实机器人操作任务。

**💡 创新点**

创新点包括：①通过多视角三角化获得无噪声手势标签，量化手势质量对转移的影响；②在不同相机和身体上对图像尺度进行对齐，显著提升跨数据集的可视化一致性；③使用token级融合而非CLS-bottleneck，允许模型在视觉和动作上专属化；④在共训练中上加权机器人数据，防止有害的表示对齐；⑤系统性评估手势质量、动作差距与转移效果的关系。

**🔧 技术方法**

技术手段包括多视角三角化手势、ViT+token融合的跨身体网络、专属动作编码/解码器、图像尺度对齐与动作空间对齐、加权共训练（机器人数据上采样）以及噪声注入实验来验证手势质量影响。

**📊 数据集**

使用的数据集：TriHands（532视频、28小时手势标签）和机器人演示数据（3000条演示），对比基准数据集EgoDex、HaWoR、EgoBridge、PiZero等。

**📈 对比分析**

与机器人单独训练（RO）、CLS-token、EgoDex、HaWoR、EgoBridge、PiZero等方法比较，Human Cotraining（HC）在低机器人数据场景下成功率提升20%–48%，平均提升29.7%，在6个任务中的最高单项提升为Pull任务（~60%）。

**⚠️ 局限性**

局限性：①仍受人类与机器人动作差距（自然动作与任务不匹配）的制约；②依赖高质量手势标签，当前手势估计器仍存在噪声；③仅在6个相对简单的任务上验证，复杂任务效果不一；④缺乏针对更大规模或更复杂场景的动作重定向方法。

---

## 111. Pomona: Continuous Code Quality Improvement via Small, Automated Changes at Bloomberg

**arXiv ID:** 2606.06752 | [PDF](https://arxiv.org/pdf/2606.06752v1)

**作者:** David Williams `[一作]` (University College London), Federica Sarro `[通讯]` (University College London)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Bloomberg内部开发了一款基于代理技能的轻量级工具，持续自动检测并提交微小代码质量改进的PR（约10行diff），实现了持续的代码质量提升。

**💡 创新点**

创新点在于将Kaizen持续改进理念与代理技能相结合，采用任务清单化、优先级排序，并通过自动化生成极小PR的方式，兼顾了工程师的信任度与可维护性，构建了低摩擦、人机协作的持续改进循环。

**🔧 技术方法**

使用技术包括大型语言模型（LLM）集成的代理技能、Markdown形式的技能与任务清单、静态分析、TODO扫描、未使用代码检测、测试覆盖度比对，以及Bloomberg自研的MCP接口生成PR。

**📊 数据集**

数据集为Bloomberg内部的Python代码库，涵盖静态分析规则、TODO标注、测试覆盖文件等；评估使用了一个月内生成的17个PR以及向10位资深工程师发放的问卷。

**📈 对比分析**

通过对17个PR的手工评估，接受率为88.2%，平均PR大小约16行，合并时间中位数<2小时；问卷显示8/10工程师愿意尝试，并计划每周审查2–3个PR，整体性能优于传统手工或大规模自动化改动。

**⚠️ 局限性**

局限性包括：对大型重构任务不适用、易产生重复PR导致的冲突、对工程师信任与任务优先级的主观判断需求、可能导致评审过载或无关改动产生“代码噪音”，以及缺乏自动迭代改进PR的功能。

---

## 112. TorchKM: A GPU-Oriented Library for Kernel Learning and Model Selection

**arXiv ID:** 2606.06742 | [PDF](https://arxiv.org/pdf/2606.06742v1)

**作者:** Yikai Zhang `[一作]` (University of Iowa), Boxiang Wang `[通讯]` (University of Iowa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一个GPU加速的开源库TorchKM，支持SVM、核逻辑回归、核分位数回归等多种核方法，并提供集成训练与模型选择的端到端流程。

**💡 创新点**

通过算法-硬件协同设计，提出精确交叉验证公式和谱算法，将原本需要多次三次方运算的训练与调参过程转化为一次特征分解和矩阵向量乘法，保持精确解且显著降低计算成本；同时实现了训练与调参集成，减少重复计算。

**🔧 技术方法**

利用PyTorch张量运算实现GPU加速；采用精确交叉验证、谱分解、Nyström近似、Platt scaling等技术；并以scikit‑learn风格的API供使用。

**📊 数据集**

在实验中使用了make_circles数据集演示；基准测试则使用规模不同（n=10k,20k,1000）且维度（p=10,100,1000）的随机生成数据集。

**📈 对比分析**

与ThunderSVM等主流核学习库进行比较，基准结果显示TorchKM在目标函数值最小、运行时间最快，尤其在n=20,000、p=1,000的大规模设置下完成训练与调参仅需129.3秒，远快于其他方法（甚至在8小时内无法完成）。

**⚠️ 局限性**

局限性包括：目前仅支持SVM、DWD、核逻辑回归和核分位数回归等有限模型；对极大规模或高维数据仍可能受限；Nyström近似虽可加速但可能牺牲精度；需要CUDA环境才能使用GPU加速；库仍在发展中，功能和优化空间有限。

---

## 113. A Rolling-Window Framework for Churn Prediction and Behavioral Driver Identification

**arXiv ID:** 2606.06776 | [PDF](https://arxiv.org/pdf/2606.06776v1)

**作者:** Muhammad Jawad Mufti `[一作]` (King Fahd University of Petroleum and Minerals), Muqaddas Gull `[通讯]` (King Fahd University of Petroleum and Minerals)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种面向非合同、按使用付费服务的时间显式流失预测框架，利用滚动观察窗口与后续流失评估窗口对客户行为进行连续风险评估。

**💡 创新点**

创新点在于：①明确的时间窗口划分与滚动实例生成，②将特征聚合和序列学习结合到同一时序框架，③采用可解释的特征重要性与 SHAP 分析，④在未来未见数据上进行时间泛化评估，体现模型的部署可行性。

**🔧 技术方法**

采用梯度提升决策树（XGBoost）与 LSTM 序列模型；通过滚动窗口实例构造、特征工程、时间感知数据划分以及 SHAP/增益重要性解释技术实现模型训练与评估。

**📊 数据集**

使用真实的按需洗车服务平台数据，包含 401,164 条预约记录、101,765 名客户，时间跨度为 2024 年 10 月至 2025 年 1 月。

**📈 对比分析**

对比了 XGBoost 与 LSTM 在滚动窗口测试集和 2025 年 6–7 月的未来数据上的表现：XGBoost 在测试集上准确率 87.6%、ROC‑AUC 0.941；LSTM 在测试集上召回率 96.1%、ROC‑AUC 0.940。未来数据上，XGBoost 准确率 83.1%、ROC‑AUC 0.911；LSTM 准确率 81.2%、ROC‑AUC 0.830，显示两模型均具备一定的时间泛化能力。

**⚠️ 局限性**

局限性包括：①仅在洗车服务场景验证，泛化性待进一步考察；②对手工特征工程的依赖可能限制自动化扩展；③LSTM 对序列长度与过拟合敏感，需更细粒度的调参；④未考虑成本敏感评估与自适应窗口机制。

---

## 114. Attack Selection in Agentic AI Control Evaluations Meaningfully Decreases Safety

**arXiv ID:** 2606.06529 | [PDF](https://arxiv.org/pdf/2606.06529v1)

**作者:** Catherine Ge-Wang `[一作]` (University of Oxford), Tyler Tracy `[通讯]` (Redwood Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文研究在AI控制评估中攻击选择（Attack Selection）的影响，提出将攻击决策拆分为任务级的起始策略（Start Policy）和步骤级的停止策略（Stop Policy），并在真实的agentic环境中评估其对安全性的影响。

**💡 创新点**

创新点在于首次系统性地引入并量化攻击选择对安全评估的上界影响，证明即使攻击能力不变，攻击者通过合理的启动和中止决策也能显著降低实验安全性，凸显现有评估假设的过于乐观性。

**🔧 技术方法**

技术手段包括：使用大语言模型（LLM）作为提示式评分器实现起始与停止策略；在受限的审计预算下的受信监控（Trusted Monitoring）框架中对攻击进行评估；采用10折交叉验证进行阈值调优；使用Newcombe 95%置信区间进行统计比较。

**📊 数据集**

所用数据集为：BashArena（637主任务、4侧任务）和LinuxArena（1,667主任务、185侧任务），两者均提供多步交互式agentic任务与对应的隐蔽侧任务。

**📈 对比分析**

比较方法是把无攻击选择的基线（攻击均匀分布）与加入起始或停止策略的过滤后攻击进行对比，衡量经验安全性（Empirical Safety）及其变化Δ；结果显示在1%审计预算下，起始策略可降低约20pp，停止策略可降低约28pp；统计置信区间显示这些下降在大部分预算下显著（除极少数样本不足导致区间宽泛的情况）。

**⚠️ 局限性**

局限性包括：实验仅评估经验安全性，未直接推断部署级安全性；在严格审计预算下样本量极小导致置信区间宽泛；停止策略仅在后验阶段评估，未对在线攻击过程做反馈；结果依赖于具体的监控协议和环境设置；未使用学习型选择器，可能低估更强攻击者的潜在影响。

---

## 115. What Your Posts Reveal: A Benchmark and Agentic Framework for User-Level Privacy Leakage on Social Media

**arXiv ID:** 2606.06784 | [PDF](https://arxiv.org/pdf/2606.06784v1)

**作者:** Zifan Peng `[一作]` (Hong Kong University of Science and Technology), Jiaheng Wei `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了用户级多模态隐私泄露基准（Synthetic Benchmark）和敏感度结合粒度的隐私暴露得分（PES），并提出无训练的跨帖推理框架（Agentic Framework），通过证据图进行累积隐私推断。

**💡 创新点**

创新点在于：①创建可控合成基准，抽象真实用户泄露模式生成多模态数据；②设计基于属性粒度与情境敏感度的曝光度量PES；③提出基于可归因推理的跨帖证据图框架，支持假设生成、路由检验与聚合。

**🔧 技术方法**

使用技术包括：大型语言模型（LLM）、视觉语言模型（VLM）、OCR、Web/地图搜索工具；构建跨帖证据图（Cross-Post Evidence Graph），实现假设生成、路由决策、证据验证与结果投影。

**📊 数据集**

数据集：基于Rednote与Instagram的真实用户语料（共318用户、11,243帖、46,771图）进行模式抽象；合成基准包含50用户、500帖、1,569图像，覆盖显式/隐式/跨帖泄露、不同难度与敏感度等级。

**📈 对比分析**

与五种基线（TextLLM, PostVLM, SelfDisc, Holmes, SingleAgent）在Binary、Granularity、PES等指标上对比。所提框架在PES上达到0.55，较最强基线提升25%（从0.44提升），尤其在跨帖泄露上提升0.17。

**⚠️ 局限性**

局限性包括：基准仅涵盖部分平台与文化，未包含长视频等丰富媒介；框架无训练，受基础模型与工具演进影响；合成数据不完全反映真实社交媒体使用情境。

---

## 116. IDDMBSE: Integrating Data-Driven and Model-Based Systems Engineering for Trusted Autonomous Cyber-Physical Systems

**arXiv ID:** 2606.06727 | [PDF](https://arxiv.org/pdf/2606.06727v1)

**作者:** John S. Baras `[一作]` (University of Maryland), Praveen M. S. Kumar `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一个将数据驱动和模型驱动系统工程（MBSE）结合的集成方法和工具链（包括 PERFECT、TRADES-X 和 VERITAS），并在可信自主地面机器人上进行了硬件选择、路径规划、感知稳健化和多机器人协作的端到端验证。

**💡 创新点**

创新点在于：① 在 MBSE V‑process 的每一步嵌入需求驱动的数据驱动循环；② 将自主性栈作为 SysML 中的一等架构元素；③ 构建了跨工具共享的 SysML 模型中心，形成统一的数字线程；④ 将形式化验证、统计验证和运行时监测三种方式整合到同一保证流程。

**🔧 技术方法**

使用了 SysML v1（后续迁移至 SysML v2）、ROS 2、Isaac Sim/ Gazebo、UPPAAL、PRISM、Lean、RTAMT、模型优化器（JuMP/Gurobi/Cplex）以及 Python/JSON WebSocket 接口等技术。

**📊 数据集**

使用的数据集包括：① 8192 种 LiDAR/雷达/RGB 传感器组合的仿真实验；② 在 Isaac Sim 生成的对抗性越野地形（坡度≤15°、障碍密度10%–80%）下的 50 轮路径规划实验；③ 通过 CP 校准的目标检测训练集以及多机器人任务的 STL 规范。

**📈 对比分析**

通过与传统 RRT* 的比较，RA‑RRT* 在四种噪声水平和三种风险阈值下显著降低了最坏情况路径长度和规划失败率；在传感器套装选择中，先用模型优化剔除 8192 组候选至 92 组，再评估 7 组，最终实现了从千级到个位数的候选压缩，显著节约仿真资源；在多机器人协作中，混合整数规划与实时 STL 监测相结合，使得机器人 1 与 3 满足规范而机器人 2 的轻微违规被及时捕获。

**⚠️ 局限性**

主要限制在于目前依赖 SysML v1，导致模型与 ROS 2 的绑定需要通过 Magic Model Analyst + MATLAB 桥接，过程繁琐且易碎；缺乏在 SysML v2 上的原生 API，难以实现完整的版本控制与自动化推理。

---

## 117. AEGIS: A Backup Reflex for Physical AI

**arXiv ID:** 2606.06660 | [PDF](https://arxiv.org/pdf/2606.06660v1)

**作者:** Josef Chen `[一作]` `[通讯]` (KAIKAKU), Josef Chen (KAIKAKU)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种基于冻结策略内部激活的实时预警与升级机制（AEGIS），在机器人长时程操作中，在出现潜在失效前切换到更强的备用策略，从而提升任务成功率；

**💡 创新点**

创新点在于首次将低成本激活探测与分阶段阈值门控相结合，形成可控的时序化升级决策；通过严格的因果对照（预算匹配的盲升级与随机触发），验证升级的时序选择才是成功关键；

**🔧 技术方法**

使用冻结的VLA（SmolVLA、OFT）内部激活作为特征，训练两层MLP探测器；采用分层合成（split-conformal）阈值校准、早期损害门控、预算上限；在运行时实现chunk‑boundary手动切换；

**📊 数据集**

主要实验基准为LIBERO‑Spatial（10个任务），共700个（task,seed）对照实验；还做了GR00T N1.7跨族验证；

**📈 对比分析**

与弱策略单独运行、预算匹配盲升级、随机触发、HELM同策略回滚以及总是强策略进行比较；在弱策略失效子集上，AEGIS实现10.1%恢复率，显著高于盲升级4.6%和随机5.1%；在相同计算预算下，AEGIS双倍恢复率；

**⚠️ 局限性**

局限性包括：恢复效果有限（+0.05左右），在最难难度区间提升不显著；仅在仿真环境验证，缺乏真实机器人实验；阈值校准的分层覆盖率有限；仅使用单一弱强策略对；未来需要扩展到多策略和真实硬件。

---

## 118. Improving Cross-Lingual Factual Recall via Consistency-Driven Reinforcement Learning

**arXiv ID:** 2606.06586 | [PDF](https://arxiv.org/pdf/2606.06586v1)

**作者:** Jonathan von Rad `[一作]` (University College London), Pontus Stenetorp `[通讯]` (University College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了PolyFact多语种事实问答数据集，并在两款7B规模的LLM上进行实验，比较了轻量化持续预训练（CPT）、监督微调（SFT）和基于GRPO的强化学习后训练方法，以提升跨语言事实一致性。

**💡 创新点**

创新点在于提出了大规模并行多语种事实QA数据集PolyFact，并通过GRPO实现跨语言知识检索的一致性提升，同时揭示了RL对内部表示与多语言路由的重组机制。

**🔧 技术方法**

采用的技术包括轻量化持续预训练、监督微调、GRPO强化学习、LAHIS和LAPE机制解释，以及使用vLLM和LightEval等评测工具。

**📊 数据集**

使用的数据集包括自建的PolyFact（约100K条基于Wikidata的事实QA，12种语言），以及评测集KLAR和Global-MMLU。

**📈 对比分析**

实验对比显示，GRPO在PolyFact、KLAR和Global-MMLU上均优于SFT，尤其在跨语言一致性和未见语言迁移上取得显著提升；CPT对性能提升有限。

**⚠️ 局限性**

局限性包括仅在两款7B模型上验证，未测试更大或不同架构的模型；对更复杂推理任务的提升有限；评测语言和任务覆盖面有限；以及数据集可能存在的偏差和错误。

---

## 119. Inside the Visual Mind: Neuroscience-Motivated Concept Circuits for Interpreting and Steering Vision Transformers

**arXiv ID:** 2606.06664 | [PDF](https://arxiv.org/pdf/2606.06664v1)

**作者:** Tang Li `[一作]` (University of Delaware), Xi Peng `[通讯]` (University of Virginia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ViSAE 工具箱，利用神经科学启发的 64K 图像与 16K 视觉概念词表，结合稀疏自编码器（SAE）、CLIP 视觉‑语言嵌入与双阶段因果电路追踪，实现 Vision Transformer 的机制可解释性，并用于模型审计与调节。

**💡 创新点**

①构建跨四个抽象层级的 64K 探测集与 16K 视觉概念词表；②基于 CLIP 嵌入的自动概念读取，避免人工标注；③双阶段因果电路追踪（自上而下读取概念，自下而上追踪因果影响）；④通过概念编辑显著提升 worst‑group 性能。

**🔧 技术方法**

稀疏自编码器（SAE）、CLIP 视觉‑语言模型、基于反事实干预的因果影响度量、概念编辑与蒸馏技术。

**📊 数据集**

七个视觉数据集（DTD、Broden、ShapeNet、ImageNet、Visual Genome、Places365、MSCOCO）组成的 64K 探测集；16K 视觉概念词表；WaterBirds、ImageNet、Quantus、VOC2007 等用于评估。

**📈 对比分析**

与 ImageNet、MSCOCO 等现有数据集及 LAION、Google Books 词表相比，概念覆盖效率提升 20 倍，解释准确率提升 28.7%；在 WaterBirds worst‑group 上，概念编辑将准确率从 50.3% 提升 48.2%，超过现有方法 23.8%；在 SAEs 评估指标中实现最佳稀疏度与重构质量的平衡；概念定位与归因方法相较提升 3.7%。

**⚠️ 局限性**

SAE 可能出现特征吸收/分解导致单一特征覆盖多概念；自动概念读取受词表与 CLIP 嵌入覆盖限制；评估主要集中在 CLIP‑style ViT 与自然图像，尚未验证其他架构或专业域。

---

## 120. MADRAG: Multi-Agent Debate with Retrieval-Augmented Generation for Training-Free Analytic Essay Scoring

**arXiv ID:** 2606.06754 | [PDF](https://arxiv.org/pdf/2606.06754v1)

**作者:** Ali Keramati `[一作]` (University of California), Mark Warschauer `[通讯]` (University of California)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种完全无训练的多代理争论与检索增强的分析性作文评分框架 MADRAG。

**💡 创新点**

通过将评分拆分为倡导者、质疑者和评判者三角色并结合检索到的示例，实现了自校准与结构化推理的结合。

**🔧 技术方法**

使用多代理对话（Advocate/Skeptic/Judge）、检索增强生成（RAG）、基于嵌入的向量检索以及 LLM 首次标记概率作为置信度指示。

**📊 数据集**

在 ASAP 语料库（Sets 7 与 8）上进行评估。

**📈 对比分析**

与多种无训练 LLM 基线和监督式 AES 模型对比，MADRAG 在多项特征上实现了最高 0.75 的 QWK，接近甚至超越部分监督模型，并在极端分数识别（Agree@1）上显著提升。

**⚠️ 局限性**

局限性包括仅在两套叙事作文上验证、对匿名占位符易误判、计算成本高、缺乏多次随机试验的统计稳健性，以及对检索库的依赖。

---

## 121. Subtle Injection for Ground-truth Inference of LLM Training Data

**arXiv ID:** 2606.06502 | [PDF](https://arxiv.org/pdf/2606.06502v1)

**作者:** Abraham Itzhak Weinberg `[一作]` `[通讯]` (AI Experts), Abraham Itzhak Weinberg (AI Experts)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SIGIL 框架，在公开文档中嵌入微小可感知的 canary 序列，并通过对 LLM 的统计测试证明文档已被训练使用。

**💡 创新点**

创新点在于主动可感知且法庭可采纳的水印设计、基于 Neyman‑Pearson 的 Membership Inference Score (MIS) 严格控制假阳性率，以及在 100% 释义对抗条件下保持高 AUC 的语义泄漏鲁棒性。

**🔧 技术方法**

采用统计假设检验（Welch t 检验与 Φ^-1）、Neyman‑Pearson 框架、可感知度量、五种 canary 策略（罕见词、短语、句法、语义、代码模式）以及仿真器校准和 36,000 条实验数据进行评估。

**📊 数据集**

实验基于仿真生成的 36,000 条样本，涵盖 5 种 canary 策略、3 种模型规模、5 注入率、4 混合比例；实际模型以 LLaMA/7B 等为参考进行统计特性模拟。

**📈 对比分析**

通过 trial‑level ROC/AUC 进行比较，整体 AUC=0.892；Code Pattern 与 Canary Phrase 的 AUC>0.90；在不同注入率/模型规模下检测率介于 33%–78%；即使在 100% 释义时 AUC 仍高于 0.86，显示出强鲁棒性。

**⚠️ 局限性**

主要局限在于仅使用仿真验证，缺少真实 LLM 训练与推理实验；仅支持黑盒 log‑prob 查询，无法处理仅生成接口；对抗者若知晓 canary 策略可能进行移除；需在真实模型上进一步验证。

---

## 122. Uncertainty-Aware LLM-Guided Policy Shaping for Sparse-Reward Reinforcement Learning

**arXiv ID:** 2606.06673 | [PDF](https://arxiv.org/pdf/2606.06673v1)

**作者:** Ujjwal Bhatta `[一作]` (University of South Dakota), KC Santosh `[通讯]` (University of South Dakota)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种结合校准的大型语言模型（LLM）与强化学习（RL）的框架ULPS，利用A*产生的符号轨迹对LLM进行微调，并在PPO训练过程中通过不确定性估计自适应地融合LLM建议与学习策略。

**💡 创新点**

创新点在于：① 用A*生成符号轨迹自监督地微调LLM；② 通过MC Dropout估计LLM的不确定性，并以熵权重动态调节LLM建议与PPO策略的混合比例；③ 将LLM与RL的融合机制与传统的纯RL或不加校准LLM方法相比较，显著提升稀疏奖励环境下的收敛速度和成功率。

**🔧 技术方法**

核心技术包括：BERT-based LLM微调、MC Dropout不确定性估计、熵基策略混合、PPO近端策略优化、A*规划生成最优轨迹、MiniGrid环境交互与经验回放。

**📊 数据集**

主要使用MiniGrid-UnlockPickup这一稀疏奖励、多任务的迷宫导航基准；同时在不同尺寸（4×4、8×8）和不同算法对比实验中使用相同环境。

**📈 对比分析**

与无指导PPO、未校准LLM、Q-Learning、DQN等基线相比，ULPS在4×4环境下成功率达到99.90%、平均步数7.24，reward AUC 2055.08；在8×8环境下成功率99.70%、平均步数15.37，显著优于传统方法且显著降低环境交互步数。

**⚠️ 局限性**

局限性包括：① 由于MC Dropout需多次前向传播，计算成本比单一LLM推理高约8倍；② 目前仅在部分可观测的单代理环境验证，未验证在更复杂的多代理或完全不可观测情境下的效果；③ 依赖A*生成的轨迹，若环境规模极大或动态变化，生成轨迹的成本也会增加。

---

## 123. When Better Codebooks Are Not Enough: Predictive Performance and Behavioral Reliability in LLM Political Event Coding

**arXiv ID:** 2606.06781 | [PDF](https://arxiv.org/pdf/2606.06781v1)

**作者:** Zixian He `[一作]` (Independent Researcher), Yibo Hu `[通讯]` (Illinois Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了将专家编码书转化为LLM友好形式，并评估其在政治事件编码中的预测性能与行为可靠性；

**💡 创新点**

提出了受控扰动探针来检验模型是否真正遵循编码规则，发现增强提示虽提升准确率，却不必然保证行为可靠性；

**🔧 技术方法**

采用LLM提示、紧凑/增强编码提示、链式推理、检索增强生成以及受控扰动实验等技术；

**📊 数据集**

使用PLOVER基准（PLV）源–目标关系分类任务和ACE/WikiEvents交叉域的二元合作/冲突任务（AW）作为数据集；

**📈 对比分析**

在四款开源LLM上对比无编码书、紧凑、增强提示等方法，增强提示在细粒度分类上宏F1提升显著，但在扰动检验中的规则跟随分数并未显著提升；

**⚠️ 局限性**

仅聚焦政治事件编码，未拆解增强提示各组成对性能的独立影响，对大型或域适配模型的泛化有限，扰动探针也未覆盖所有现实场景。

---

## 124. Learn to Match: Two-Sided Matching with Temporally Extended Feedback

**arXiv ID:** 2606.06744 | [PDF](https://arxiv.org/pdf/2606.06744v1)

**作者:** Haijing Zong `[一作]` (University of Washington), Natasha Jaques `[通讯]` (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于部分可观测马尔可夫博弈的学习框架 Learn2Match，用于研究信息随时间逐步揭示的双边匹配市场。

**💡 创新点**

创新点在于将匹配过程建模为具有成本的预匹配筛选、噪声的后匹配观测、潜在档案演化和自发续约/解散的动态过程，并强调信息的逐步揭示。

**🔧 技术方法**

采用多智能体强化学习中的PPO算法以及基于碰撞规避的探索-承诺（CA-ETC）作为基准，通过强化学习策略在部分可观测环境中学习匹配决策。

**📊 数据集**

使用了自定义的 Learn2Match 基准数据集，包含小规模（5 对 5）和大规模（20 对 20）的市场实例，设置噪声访谈、噪声后匹配观测和逐步信息揭示的情形。

**📈 对比分析**

通过比较 PPO 与 CA-ETC 在累计惩罚、社会福利和信息摩擦损失三个指标上的表现，发现 PPO 在累计惩罚和社会福利上优于 CA-ETC，但在信息摩擦损失上较高。

**⚠️ 局限性**

局限性在于 PPO 由于缺乏足够的探索，导致信息摩擦损失未能收敛到零，需要进一步结合协调探索或稳定匹配结构以提升整体信息恢复能力。

---

## 125. Does Topic Sentiment Cause Perceived Ideology? Comparing Human and LLM Annotations in Political News Articles

**arXiv ID:** 2606.06715 | [PDF](https://arxiv.org/pdf/2606.06715v1)

**作者:** Upasana Chatterjee `[一作]` `[通讯]` (Columbia University), Upasana Chatterjee (Columbia University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了话题情感对新闻文章感知政治意识形态的因果影响，并检验该影响是否因使用的标注来源（人工专家、GPT‑4o‑mini（基线与微调）与 Llama‑3.3‑70B）而异。

**💡 创新点**

首次在同一因果结构下直接比较人工与大型语言模型标注的因果路径，揭示微调后 LLM 可能内部化与人类不同的情感‑意识形态耦合（shortcut learning）。

**🔧 技术方法**

采用双重机器学习（DML）进行因果效应估计，并结合社区层级的媒介分析来分解直接与间接影响。

**📊 数据集**

使用 AllSides 公开的专家标注新闻文章数据，配合 Llama‑3.3‑70B 提取的主题情感特征，样本量约 1,265 篇。

**📈 对比分析**

在四种标注范式（专家、人类、GPT‑4o‑mini 基线、GPT‑4o‑mini 微调、Llama‑3.3‑70B）下进行比较。微调 GPT‑4o‑mini 在社区层级显著捕捉情感对意识形态的因果效应，且自然直接效应占主导；基线 GPT 与 Llama 无显著因果关系；整体 F1 最高的微调模型并不保证因果结构与人类一致。

**⚠️ 局限性**

限制包括：情感特征仅来自 LLM 估计，缺乏人类验证；未控制出版源、作者、发布时间等文章级混杂变量；样本量在社区层级和媒介分析中有限，导致统计功效不足；同一简化因果图可能无法完全刻画人类与 LLM 的真实生成过程。

---

## 126. Generative Models Erode Human Temporal Learning Through Market Selection

**arXiv ID:** 2606.06572 | [PDF](https://arxiv.org/pdf/2606.06572v1)

**作者:** Wenjun Cao `[一作]` `[通讯]` (Independent Researcher), Wenjun Cao (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并分析了“价值崩溃”机制，说明生成式 AI 如何通过降低对人类时间学习的可区分性导致高成本人类学习工作被低成本 AI 产出取代。

**💡 创新点**

创新点在于将经济学成本检查框架与人工智能技术相结合，提出四阶段验证衰减模型，并系统梳理跨域（学术、法律、医学、内容平台）证据，阐明价值崩溃与模型崩溃的关联。

**🔧 技术方法**

主要使用经济学的成本检查与信号传递理论来构建模型，并对生成式 AI 的输出特性进行理论分析；未引入深度学习或实验技术。

**📊 数据集**

未使用具体机器学习数据集；通过引用已有的行业统计、学术论文出版量、法律处罚案例等公开数据作为跨域证据。

**📈 对比分析**

方法是理论模型的构建与跨域案例比较，未给出数值性能指标；评价以对现实领域验证衰减程度的描述为主。

**⚠️ 局限性**

局限性包括：假设产出可分为两类、忽略产出连续性、对验证可行性的实证支持不足、跨域差异和未来技术演进的影响未被量化。

---

## 127. Re-Centering Humans in LLM Personalization

**arXiv ID:** 2606.06614 | [PDF](https://arxiv.org/pdf/2606.06614v1)

**作者:** Lechen Zhang `[一作]` (University of Illinois Urbana Champaign), Tal August `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过三阶段（属性抽取、相关性匹配、个性化生成）框架，利用真实人类对话及其人工评估数据，系统评估大语言模型（LLM）的个性化能力。

**💡 创新点**

创新之处在于将人类数据与评估回归到管线中心，构建550条真实对话与5,949+11,919+1,101条人类评判的完整数据集，并提出轻量级验证器和训练干预（RoBERTa验证器、GRPO）提升模型与人类评判的一致性。

**🔧 技术方法**

主要技术包括多大模型推理（Llama‑3.3‑70B、Qwen3.5‑27B、Gemma‑4‑31B、Claude‑Sonnet‑4.6、GPT‑5.4）、RoBERTa监督分类器、GRPO强化学习、检索基线（BM25、句子嵌入）、奖励模型（ModernBERT、Qwen2.5‑1.5B、Llama‑3.2‑1B）以及人类标注与统计评估。

**📊 数据集**

使用的公开数据集有真实对话来源 WildChat（经筛选后 16,573 名用户），以及三类合成对话数据 CUPID、PrefEval、PersonaLens；提示来源 LIMA；还利用 WildChat 产生的 550 条对话。

**📈 对比分析**

通过人类与模型的对比，发现属性抽取在真实对话上准确率低；模型在相关性匹配上高召回但低精度，训练后 F1 最高达 0.64；个性化生成在 54.6% 的实例中被人类评为与通用无差异，LLM 判别与人类的相关系数仅 0.3‑0.37，奖励模型的 Spearman 相关也仅约 0.3。

**⚠️ 局限性**

局限性包括：聚合人类标注忽略评估者差异；数据仅覆盖英语/西方文化；只关注稳定属性抽取而未覆盖记忆更新、属性冲突与用户控制；训练干预在不同人群和领域的泛化性尚未验证。

---

## 128. Detecting and Mitigating Bias by Treating Fairness as a Symmetry Operation

**arXiv ID:** 2606.06514 | [PDF](https://arxiv.org/pdf/2606.06514v1)

**作者:** Nishit Singh `[一作]` (Birla Institute of Technology and Science), Nishit Singh `[通讯]` (Birla Institute of Technology and Science)

**通讯引用:** 37 | [OpenAlex ID](https://openalex.org/A5055173108)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文提出将公平性视为对敏感属性翻转的对称操作，并通过对称正则化来减轻机器学习模型的偏差；

**💡 创新点**

创新点在于将偏差建模为对称破缺，利用对称恢复正则化而不依赖因果图，适用于缺乏因果信息的场景；

**🔧 技术方法**

主要技术包括：对抗性对称正则化（Symmetry Regularizer）、基于交叉熵的任务损失、梯度链式求导以及对称性损失的实现；

**📊 数据集**

使用四个人工生成的合成数据集（D1-D4），分别包含不同程度的偏差、相关性、噪声与不平衡；

**📈 对比分析**

通过对比正则化前后的违约指标V和准确率，实验显示在低偏差与相关数据集上可将违约率降低93.2%，准确率仅损失约5%；

**⚠️ 局限性**

局限性包括：无法处理因果介导的偏差（因不使用因果图），对敏感属性只能以位翻转形式定义，且仅在合成数据上验证，未在真实数据集上测试。

---

## 129. Comparing Sentiment Contagion in AI-Agent and Human Social Networks: Evidence from MOLTBOOK

**arXiv ID:** 2606.06665 | [PDF](https://arxiv.org/pdf/2606.06665v1)

**作者:** Elyes Ben chaabane `[一作]` (University of Lausanne), Yash Raj Shrestha `[通讯]` (University of Lausanne)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过分析MOLTBOOK AI代理社交网络中的约290万条帖子与147万条评论，探究了负面情绪的传播与衰减机制。

**💡 创新点**

创新点在于提出对AI代理网络情绪韧性进行反事实敏感性检验，并在大规模样本上细化了负面情绪恢复与中性化的转移模式。

**🔧 技术方法**

使用CardiffNLP Twitter‑RoBERTa情绪分类器进行情绪标注，结合转移概率、事件研究、Fisher‑z检验和置换检验等统计方法，辅以对标签、伙伴匹配、中心代理和时间序列的反事实置换。

**📊 数据集**

采用Moltbook Observatory Archive（Hugging Face导出）中的2026‑01‑27至2026‑05‑06期间的帖子与评论数据，包含177,496个发帖代理和20,309个评论代理。

**📈 对比分析**

通过转移矩阵、事件研究和相关分析对负面情绪的持久性、冲击恢复和结构滞后进行比较，结果显示负面帖子吸引更多回复，但回复大多向中性化转移；与人类网络相比，迟滞传播效果弱，且反事实置换显著影响转移指标。

**⚠️ 局限性**

主要局限包括情绪标签仅为分类器输出，缺乏心理真实度；冲击恢复分析受观察窗口短暂限制；反事实置换无法代表真实干预；未对回复内容进行细粒度人工验证。

---

## 130. Explainable Runtime Dependency Tracking for AI-RAN Conflict Monitoring

**arXiv ID:** 2606.06663 | [PDF](https://arxiv.org/pdf/2606.06663v1)

**作者:** Christie Djidjev `[一作]` (Idaho National Laboratory), Nicholas Kaminski `[通讯]` (Idaho National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在 AI‑RAN 冲突监控中提出了一种可解释的运行时依赖跟踪原语，利用布尔矩阵表示参数–KPI 依赖并通过布尔矩阵乘法实时检测一致性。

**💡 创新点**

创新点在于：①用布尔矩阵与布尔矩阵乘法构造可解释的依赖模型；②设计滑动窗口的重用–重算推理规则，避免每次观测都全量重推；③将重算成本与窗口大小关联，提供稳定‑适应性权衡。

**🔧 技术方法**

主要技术包括布尔代数、布尔矩阵乘法、行级布尔推理、滑动窗口一致性检验、穷举搜索/0‑1 优化、合成布尔事件生成器。

**📊 数据集**

使用合成的布尔事件流作为数据集，按预设间隔 D 生成逐步变化的真实依赖矩阵 L*，并在每一步加入 ϵ=0.02 的布尔噪声。

**📈 对比分析**

通过重用与重算路径的耗时对比、边缘 F1、平均哈明距离、变化检测 recall/precision、误报率和平均检测延迟等指标评估性能。结果显示：小窗口快速响应但误报多；大窗口误报低、准确率高但检测延迟明显增加；重用路径处理时间低于 0.01 ms，远优于现有 20 s 的冲突评估或数百 epoch 的 GNN 训练。

**⚠️ 局限性**

局限性包括：仅在合成布尔流上验证，未覆盖实际连续遥测与复杂交互；只建模依赖支持关系，忽略强度、时序及多参数交互；布尔化噪声为独立翻转，缺乏多维噪声模型；行级穷举推理仅适用于小矩阵；未与完整图重建或冲突评估框架直接比较。

---

## 131. NTILC: Neural Tool Invocation via Learned Compression

**arXiv ID:** 2606.06566 | [PDF](https://arxiv.org/pdf/2606.06566v1)

**作者:** Andrew Krikorian `[一作]` (University of Michigan), Jason J. Corso `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于神经网络的工具调用框架NTILC，能够通过外部检索来替代传统的上下文中工具注册表查找，实现工具选择与调用的解耦；

**💡 创新点**

创新点在于将用户意图与工具规范映射到共享嵌入空间，并引入功能边际损失（Functional Margin Loss）来解决语义模糊导致的工具选择错误，同时通过组合圆形损失和功能边际损失构建签名感知的复合目标；

**🔧 技术方法**

核心技术包括：基于小型Transformer的嵌入编码器、密集检索（nearest‑neighbor）作为工具调度、Circle Loss 对齐查询与工具、Functional Margin Loss 强制功能签名分离、有限状态机约束的参数生成与安全调用；

**📊 数据集**

使用公开的工具选择与函数调用数据集：ToolBench、API‑Bank、BFCL、ToolEyes、MetaTool；

**📈 对比分析**

与传统的ICT（In‑Context Tooling）基线在同一工具库规模下进行对比，NTILC 在保持或提升工具检索准确率（Top‑1 98% 以上）的同时，将上下文令牌消耗降低 95%+，推理延迟下降 70%+（如 ToolBench 由 5663 ms 降至 1452 ms）；

**⚠️ 局限性**

局限性包括：需要先行构建并维护工具索引，动态或状态依赖的工具可能需要频繁重建；仅关注工具选择，参数生成与执行安全仍需额外机制；对大型或频繁更新的注册表的在线索引和更丰富的功能相似度度量尚未充分探讨。

---

## 132. From Pixels to Newtons: Predicting In Vivo Joint Contact Forces from Monocular Video

**arXiv ID:** 2606.06631 | [PDF](https://arxiv.org/pdf/2606.06631v1)

**作者:** Jessy Lauer `[一作]` `[通讯]` (Harvard University), Jessy Lauer (Harvard University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

该论文提出了一种端到端的物理学无关管线，利用单摄像头未校准视频预测髋关节和膝关节瞬时三维接触力，并提供不确定性估计。

**💡 创新点**

创新点在于：1) 直接从视频学习到接触力映射，完全不需要标记、力平台、肌电或个人化模型；2) 采用自监督视频特征代替人工活动标签；3) 将预测器与生成运动先验结合，执行逆向设计生成降低负荷的运动方案。

**🔧 技术方法**

技术方法包括：Neural Localizer Fields恢复3D SMPL网格；Depth Anything估计相机姿态；对姿态特征进行局部卷积和RoPE位置编码；双流Transformer通过AdaLN调节静态上下文；输出头为可变方差高斯，提供不确定性；逆向设计使用Rectified Flow + SDEdit 的梯度引导。

**📊 数据集**

使用的数据集为公开的OrthoLoad数据库，包含26位患者、28个植入点、2600个视频-力对，涵盖25个活动类别，另外在独立的Grand Challenge竞赛数据上进行零样本验证。

**📈 对比分析**

与传统基于肌肉骨骼模型的方法相比，该方法在留一患者交叉验证中实现了相当的精度（膝部RMSE 0.23±0.03 BW，髋部0.32±0.08 BW），在独立竞赛数据上性能与过去冠军相当或更好，且在大多数活动中实现了低于临床显著性阈值的力变化。

**⚠️ 局限性**

主要局限包括：1) 训练数据仅来自已植入人工关节的老年患者，可能不代表健康或年轻人群；2) 对极高峰值的预测不确定性略显过度自信；3) 性能高度依赖于前端姿态估计的质量，若姿态估计失败会直接导致力预测失效。

---

## 133. A Four-Condition Diagnostic Protocol for Evidence Utilization in Long-Context and Retrieval-Augmented Language Models

**arXiv ID:** 2606.06758 | [PDF](https://arxiv.org/pdf/2606.06758v1)

**作者:** Haizhou Xia `[一作]` `[通讯]` (University of Western Ontario), Haizhou Xia (University of Western Ontario)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种四条件诊断协议（无证据、完整上下文、检索证据、oracle 证据），通过对同一模型、同一实例在不同证据可用性条件下的回答与证据表现进行匹配，量化模型真正利用提供证据的程度。

**💡 创新点**

创新点在于将答案准确度、检索召回和证据覆盖等指标拆分为四个互不重叠的角色，并引入 ONCU（Oracle-Reference Normalized Context Utilization）作为估计器，能够在同一实验框架下同时观测无证据基准、oracle 参考优势以及两种上下文条件的利用效果，从而清晰区分模型是因参数记忆、检索链缺失、证据定位失败还是答案转换失误导致的误差。

**🔧 技术方法**

技术手段包括：固定 Prompt 与解码策略；使用 lexical、dense、hybrid 及 oracle retriever 的检索设置；在 Controlled-ONCU、HotpotQA-ONCU、2WikiMultiHopQA-ONCU 上执行完整四条件评估；计算 ONCU、答案 F1/E1、检索召回等指标；对 ONCU 进行分组统计、bootstrap 置信区间、失败模式审计及检索灵敏度实验（dense@16/hybrid@16）。

**📊 数据集**

数据集涵盖 Controlled-ONCU-safe16K（合成长文档），HotpotQA-ONCU（多跳问答），2WikiMultiHopQA-ONCU（多跳推理），以及外部验证集 BABILong-200 与 RULER-lite-240，用以验证协议在不同任务与长度条件下的通用性。

**📈 对比分析**

通过在同一模型、同一实例下对四个条件进行匹配比较，计算 ONCU 与答案/证据指标；实验结果显示：在受控合成环境中，检索证据条件几乎恢复了 oracle 参考优势，而完整上下文的 ONCU 较低；在 HotpotQA 等真实多跳任务中，完整上下文的 ONCU 较高，检索证据表现落后；不同模型在 ONCU 上的差异揭示了定位、检索链覆盖或答案转换等具体瓶颈。

**⚠️ 局限性**

局限性：ONCU 只能在具备 oracle 证据的任务中使用，无法直接迁移到缺乏精确证据标注的任务；协议依赖固定 prompt、解码和检索设置，对模型或检索策略的泛化受限；ONCU 本身并非排行榜分数，需要与答案、证据和检索指标共同解读；对复杂推理链覆盖的诊断仍不完整，需进一步研究。

---

## 134. RTLScout: Joint Agentic Code and Synthesis Optimization for Efficient Digital Circuits

**arXiv ID:** 2606.06530 | [PDF](https://arxiv.org/pdf/2606.06530v1)

**作者:** Felix Arnold `[一作]` (Huawei), Lukas Cavigelli `[通讯]` (Huawei)

**通讯引用:** 2971 | [OpenAlex ID](https://openalex.org/A5025399641)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出RTLScout，一种基于LLM的自动RTL优化系统，集成源代码级重写、门级合成优化和算术架构搜索；

**💡 创新点**

创新点在于将优化意图嵌入源代码、使用多跑精英池与经验反馈实现跨跑知识迁移，并形成端到端PPA反馈循环；

**🔧 技术方法**

使用Python嵌入式HDL Spire、LLM Agent（ReAct范式）、Mockturtle+ABC门级优化、OpenROAD+Yosys STA、以及算术库的结构化搜索；

**📊 数据集**

主要数据集为IEEE‑754 FP16乘法器、FP16加法器以及RTLRewriter的14个控制与数据路径案例；

**📈 对比分析**

与传统手工/基准实现比较，通过面积/延迟Pareto前沿评估，RTLScout在FP16乘法器上面积从121µm²降至79µm²、延迟从1618ps降至891ps，且在FP16加法器和RTLRewriter案例中均实现显著面积/延迟/单元数下降，优于商业工具和现有自动化方法；

**⚠️ 局限性**

局限包括对更大规模/时序敏感设计的适应性待验证，缺乏对完整时序约束的细粒度控制，以及对LLM生成代码质量与可解释性的依赖。

---

## 135. Adaptive Band Selection for Hyperspectral Classification with Spatially Disjoint Evaluation

**arXiv ID:** 2606.06684 | [PDF](https://arxiv.org/pdf/2606.06684v1)

**作者:** Ikram El-Hajri `[一作]`, Alejandro Mousist `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种两阶段的超光谱波段选择方法SGBR‑HC，利用监督的波段组排名初始化Hard‑Concrete稀疏门控，实现自适应波段压缩。

**💡 创新点**

创新点在于将谱组分离与类别判别的双重打分作为先验，避免传统固定数量或无监督选择，且通过Hard‑Concrete门控实现可微稀疏化并直接得到硬子集。

**🔧 技术方法**

采用KL聚类分组、JM+ReliefF判别与相关性多样性评分；随后使用Hard‑Concrete稀疏门控与ℓ₀正则化联合训练UNet‑DS分类器。

**📊 数据集**

在Pavia University（103波段）和Houston 2013（144波段）两个城市航空场景上进行实验。

**📈 对比分析**

与九种公开波段选择基线及随机选择进行对比，在空间分块+缓冲的泄漏意识评估下，SGBR‑HC在Pavia上平均OA 69.35%、κ 0.594，超过全部波段和其他基线；在Houston 2013上OA 86.24%、κ 0.851，位居前列。

**⚠️ 局限性**

仅在城市航空数据上验证，使用单一UNet‑DS分类器，且对非空间分块评估不充分，未来需测试多场景、多传感器及不同模型以评估泛化能力。

---

## 136. AxisGuide: Grounding Robot Action Coordinate System in RGB Observations for Robust Visuomotor Manipulation

**arXiv ID:** 2606.06761 | [PDF](https://arxiv.org/pdf/2606.06761v1)

**作者:** Jiyun Jang `[一作]`, Jungbeom Lee `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在图像空间显式定位机器人动作坐标系的方法，以改进视觉运动学习。

**💡 创新点**

创新点在于通过投影向量归一化和动作坐标系在图像中的直接表达，提升了在未知物体位置和动态环境下的鲁棒性。

**🔧 技术方法**

采用了基于相机投影的向量表示、规范化处理、深度强化学习策略，并对校准误差进行鲁棒性评估。

**📊 数据集**

使用了 NOP 任务、LIBERO 任务（Goal/Long）以及基准方法中的 SmolVLA 等数据集进行验证。

**📈 对比分析**

与 SmolVLA、TraceVLA、AimBot 等基线比较，成功率从 52.38% 提升至 65.71%，在 Pick & Place、Stove 等子任务中也实现了显著性能提升。

**⚠️ 局限性**

局限性包括未涵盖力控或触觉反馈等低层交互，以及对完全动态移动平台的适用性仍需进一步验证。

---

## 137. Exploring Reinforcement Learning for Fluid Transitions Between Clinical Mental Healthcare and Everyday Wellness Support

**arXiv ID:** 2606.06800 | [PDF](https://arxiv.org/pdf/2606.06800v1)

**作者:** Tony Wang `[一作]` (Cornell University), Qian Yang `[通讯]` (Cornell University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个基于强化学习的情境分支模型，在日记写作中动态选择临床与健康维持的提示，评估其对持续写作参与度的影响。

**💡 创新点**

首次将临床干预和日常健康支持整合为统一的RL系统，探讨“退后期”与高参与度用户的双重效应，并提出在同一系统中平衡临床与健康维持的设计难题。

**🔧 技术方法**

采用情境多臂赌博机（epsilon‑greedy）强化学习、CoAuthor研究平台、键盘打字行为分析及自评量表收集用户行为数据。

**📊 数据集**

247条从临床和健康文献中收集的日记提示，以及38名普通劳工的写作记录、词数、打字速度、删除频率等行为数据。

**📈 对比分析**

通过三组（RL、随机、静态）对比日记词数、打字速度等指标，发现RL在干预期无显著提升，但干预后显著增加词数和反思度，未进行统计检验，样本量有限。

**⚠️ 局限性**

样本规模小、仅低风险人群、奖励信号仅为词数、使用简单RL模型、短期研究、缺乏长期健康效果验证及临床监督。

---

## 138. Spatiotemporal Imputation with Graph-Informed Flow Matching

**arXiv ID:** 2606.06682 | [PDF](https://arxiv.org/pdf/2606.06682v1)

**作者:** Zepeng Zhang `[一作]` (EPFL), Olga Fink `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于图信息流匹配的时空缺失值填补框架GiFlow，利用自适应时空滤波构造结构化先验，显著提升填补精度。

**💡 创新点**

核心创新在于将图卷积与时空注意力结合的混合向量场作为流匹配模型的向量场，同时通过联合时空滤波生成图信息先验，理论证明可降低传输成本。

**🔧 技术方法**

技术包括流匹配（Flow Matching）、自适应时空滤波、空间与时间注意力机制、图神经网络与Transformer的混合消息传递以及确定性Euler采样。

**📊 数据集**

实验涵盖合成数据以及三个真实时空数据集：Air-36、AQI（空气质量）和PeMS08（交通流量）。

**📈 对比分析**

与多种基准（非参数、RNN、GNN、Transformer、扩散模型）对比，GiFlow在点缺失和块缺失、不同缺失率下均表现出MAE/RMSE/MAPE领先或相当的优势，并在推理速度上大幅优于扩散模型。

**⚠️ 局限性**

局限性包括对图结构质量敏感（阈值选择影响性能），缺乏对不确定性量化的完整处理，以及在极高缺失率或图稀疏时性能可能下降。

---

## 139. Optimal Control Approach for Non-prehensile Ball Juggling Using a 7-DoF Manipulator

**arXiv ID:** 2606.06704 | [PDF](https://arxiv.org/pdf/2606.06704v1)

**作者:** Joel Ramadani `[一作]` (Technical University of Munich), Sami Haddadin `[通讯]` (Mohamed Bin Zayed University of Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出了一套两阶段的基于模型的最优控制框架，用于在机械臂上实现非抓握的球类抛投与连续接收（即机器人抖球），通过先在二维球-工具模型上求解任务空间最优轨迹，再映射到七自由度机械臂的关节空间，并预先生成一组修正轨迹数据库，在线时根据球的实时状态选择最合适的轨迹并进行时间再参数化，实现高频、低误差的持续抖球。

**💡 创新点**

创新点主要体现在：①将球-工具的混合动力学用MPCC形式化为任务空间最优控制问题；②采用两阶段OCP将二维任务轨迹映射到机器人关节空间，保证动力学与执行约束一致；③预先离线求解一套覆盖不同顶点状态的轨迹数据库，并设计在线决策策略与时间再参数化，实现了快速、稳健的在线控制；④通过轨迹再参数化兼顾时序同步，解决了抖球过程中时间漂移的问题。

**🔧 技术方法**

技术方法包括：Lagrangian 动力学建模与约束求解、Mathematical Program with Complementarity Constraints (MPCC)、CasADi + IPOPT + Pinocchio 的离线优化、MuJoCo 物理仿真、Franka Emika Panda 机器人实验、时间再参数化（C1 连续路径-速度分解）以及决策策略（基于最近邻的顶点状态匹配）。

**📊 数据集**

实验数据来源为：①基于MuJoCo的离线仿真，用以评估轨迹准确性和鲁棒性；②Frank a Panda 机器人实际抓取实验，仅验证了抖球的一次循环；③通过在7x7 cm范围内随机采样50个初始状态进行蒙特卡洛仿真，用以绘制稳定性、误差和鲁棒性地图。未使用公开的标准数据集，而是自定义的物理模型和实验场景。

**📈 对比分析**

对比方法主要是将生成的轨迹在仿真与硬件上进行性能评估：①通过最大连贯抖球次数、顶点位置误差和状态误差来衡量稳定性与精度；②在不同摩擦系数和阻尼参数下进行参数扫描，观察连续抖球次数；③与传统的基于轨迹跟踪或启发式策略相比，本文方法在相同条件下实现了更高的连续抖球次数、更低的误差，并在较大范围内保持稳态；性能指标包括最大连贯抖球数、顶点误差均值（约1–2 cm）和状态误差（约0.1 m/s）。

**⚠️ 局限性**

局限性包括：①实验仅验证了单次抖球循环，未在长时间持续抖球中验证长期稳定性；②模型仅为二维球-工具约束，未考虑球的旋转、真实接触摩擦等三维效应；③对高频脉冲影响的建模较为简化，实际中可能存在非线性阻尼、材料弹性等未知因素；④决策策略依赖于预先生成的数据库，若遇到数据库未覆盖的极端状态可能失效；⑤系统对工具几何和摩擦参数高度敏感，需要精确校准。

---

## 140. A Geometric Gaussian Mixture Representation of Plane Curves

**arXiv ID:** 2606.06505 | [PDF](https://arxiv.org/pdf/2606.06505v1)

**作者:** Ali Darijani `[一作]` (Fraunhofer IOSBKIT), Jürgen Beyerer `[通讯]` (Fraunhofer IOSB)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

提出了一种用户可定义的概率多边形表示平面曲线，并通过矩匹配直接得到闭式高斯混合模型以描述曲线的不确定性。

**💡 创新点**

在多边形段上引入可调的法向不确定性，利用一阶二阶矩匹配得到与切向、法向和弧长密切对应的高斯分量，从而构造既保留几何特性又具分析可行性的概率模型。

**🔧 技术方法**

结合统计学中的高斯混合模型与矩匹配、曲线分段与切法向计算、自适应离散化等技术实现该概率模型。

**📊 数据集**

在一组经典平面曲线（圆、椭圆、抛物线、对数螺线、尖点曲线、周期曲线等）上进行实验验证。

**📈 对比分析**

通过可视化的概率密度热图与传统多边形逼近对比，展示模型能够保留局部切向、法向和弧长信息；实验显示段长减小时概率密度聚焦于曲线，模型收敛良好，但未给出定量性能指标。

**⚠️ 局限性**

存在退化（τ=0）时需要正则化；缺乏严格的收敛性与误差理论；对三维曲面与更高维几何的扩展与GMM的微积分框架尚未完成。

---

## 141. GOPAgen: Motion-Aware and Efficient Agentic Long-Video Understanding with Structural Memory and Hierarchical Reasoning

**arXiv ID:** 2606.06532 | [PDF](https://arxiv.org/pdf/2606.06532v1)

**作者:** Haozhe Chi `[一作]` (Peking University), Yadong Mu `[通讯]` (Peking University)

**通讯引用:** 9106 | [OpenAlex ID](https://openalex.org/A5028877572)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 GOPAgen 框架，将视频编解码器中的 GOP 块与代理式视频理解相结合，实现了高效的运动信息提取与层次化记忆。

**💡 创新点**

创新点在于：① 专门训练的运动代理利用 GOP 块的稀疏运动向量实现细粒度运动理解；② 采用 GOP‑tree 递归推理算法与结构化局部记忆同步，避免了传统方法的多轮迭代与高内存开销；③ 通过向量数据库动态检索运动向量，实现了不同粒度下的高效检索。

**🔧 技术方法**

使用技术包括：多阶段训练（预训练、混合训练、微调）结合运动向量标记器；LLM（Qwen‑2.5‑VL、Deepseek‑V3）与视觉编码器（SigLip‑2）协同；结构化记忆页、zoom‑in 层次化构建；GOP‑tree 递归检索与小型 LM 排序器；ChromaDB 向量数据库。

**📊 数据集**

实验数据集：VideoMME、MotionBench、Egoschema、NextQA、LongVideoBench、LVBench、MLVU、MSVD、MSRVTT、ActivityNetQA、Video‑MME、LongVideoBench、LVBench 等，覆盖短视频、长视频、动作与时间推理。

**📈 对比分析**

与现有闭源模型（Gemini‑2.0‑Flash、GPT‑4o）和开源模型（Qwen‑2.5‑VL、InternVL‑2.5‑72B、AdaReTaKe、Deep Video Discovery、Video‑Lucy 等）进行对比，GOPAgen 在 VideoMME overall 75.4%、VideoMME long 66.7%（相较于 Video‑Lucy 54.2%）、LVBench 73.5%、LongVideoBench 73.2%、MLVU 77.3% 等指标上均超越或匹配最先进方法，且在时间与视觉 token 方面相较于 DVD 仅消耗 1/70 的 token，平均推理时间约为 Video‑Lucy 的 28%。

**⚠️ 局限性**

局限性包括：① 对大规模 LLM 与高分辨率视频仍有显著计算与内存需求；② 运动代理训练依赖人工标注的运动描述数据，标注成本高；③ 当前实现主要针对动作/运动丰富的视频，对非运动场景（如静态纪录片）的适应性尚待验证；④ 向量数据库的实时更新和并发访问在大规模部署时可能成为瓶颈。

---

## 142. Data-Efficient Autoregressive-to-Diffusion Language Models via On-Policy Distillation

**arXiv ID:** 2606.06712 | [PDF](https://arxiv.org/pdf/2606.06712v1)

**作者:** Xingyu Su `[一作]` (Texas A&M University), Shuiwang Ji `[通讯]` (Texas A&M University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过在已有的自回归语言模型（ARLM）基础上，利用自上文注意力并改为块级双向注意力，将ARLM转化为扩散语言模型（DLM），并在此过程中引入了 On‑Policy Distillation。

**💡 创新点**

创新点在于：①将 On‑Policy Distillation 与扩散生成框架结合，使得 DLM 在训练时直接使用自身逆扩散轨迹（on‑policy states），从而消除传统 DLM 的训练–推理状态不匹配；②使用原始冻结的 ARLM 作为教师，对 DLM 进行 token‑级分布匹配，显著保留了 ARLM 的知识，解决了转换过程中的知识丢失问题。

**🔧 技术方法**

主要技术包括：On‑Policy Diffusion Language Model（OPDLM）框架、块级扩散训练、逆扩散轨迹采样、基于 ARLM 的 token‑级 KL 蒸馏、基于学习率和轨迹长度的课程学习、以及基于置信度的多 token 解码。

**📊 数据集**

使用约 60K 公开样本的四大领域数据集（数学 20k、代码 21k、科学 10k、聊天 10k）进行训练；在评估时涵盖了 MMLU、GSM‑8K、MATH‑500、AIME‑24/25、GPQA‑Diamond、IF‑Eval、HumanEval、MBPP、LCB‑v6、Codeforces 等广泛基准。

**📈 对比分析**

与 LLaDA、Dream、SDAR、Fast‑dLLM‑v2 等主流 DLM 基线进行对比，OPDLM 仅使用 0.07–0.08B 训练 token（相当于 15×–7000× 的节省），在大多数基准上实现了与或超过这些基线的性能，尤其在高难度推理、多语言推理和“思考”任务上表现突出。

**⚠️ 局限性**

局限性包括：①在更大块尺寸或更大模型规模上的泛化仍待验证；②依赖于冻结的 ARLM 教师，若原始 ARLM 质量不足，知识蒸馏效果受限；③对极长序列或需要复杂多步推理的任务仍可能需要更长的轨迹或额外的奖励信号；④目前实验主要针对 Qwen3 系列模型，跨模型迁移需要进一步研究。

---

## 143. Online Span Minimization for Flexible Uniform Jobs

**arXiv ID:** 2606.06681 | [PDF](https://arxiv.org/pdf/2606.06681v1)

**作者:** Mozhengfu Liu `[一作]` (Northwestern University), Xueyan Tang `[通讯]` (Nanyang Technological University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在线 Span Minimization 问题，提出了随机化与可重启的调度算法，突破传统的 2‑竞争下限，并给出了相应的理论证明。

**💡 创新点**

创新点包括：① 通过随机化和精细的概率分布设计，实现了 1/ln 2≈1.443 的竞争比；② 引入可重启模型，证明最优竞争比为黄金比例 ϕ≈1.618；③ 设计了 Best‑Batch 算法，在线实现 1.618 竞争并在离线下线性时间内求解。

**🔧 技术方法**

主要技术手段有：竞争分析与 Yao 原理、随机变量分布优化、动态规划与分块策略、批处理调度、贪心与启发式判定、以及对重启操作的细致建模。

**📊 数据集**

实验数据并未提供；研究以理论证明为主，利用随机生成的对手实例进行竞争比分析。

**📈 对比分析**

方法的比较以竞争比为标准；随机化算法的竞争比为 1.443，优于已知的 2‑竞争；重启算法的竞争比为 1.618，等价于理论下界，证明了其最优性。

**⚠️ 局限性**

局限性包括：仅针对统一长度作业；重启模型假设重启无成本；对一般灵活作业的竞争比尚未达到最优；算法在实际大规模实例上的实现复杂度和效率仍有提升空间。

---

## 144. AdMem: Advanced Memory for Task-solving Agents

**arXiv ID:** 2606.06787 | [PDF](https://arxiv.org/pdf/2606.06787v1)

**作者:** Runzhe Wang `[一作]` (Princeton University), Jason Zhu `[通讯]` (Arm)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了统一的 AdMem 框架，在 LLM 代理中实现语义、情节与程序记忆的短期与长期层次管理。

**💡 创新点**

创新点在于多智能体架构（Actor‑Critic‑Memory）结合奖励驱动的程序记忆生成、堆栈式上下文压缩以及基于检索与奖励的长期记忆裁剪与合并。

**🔧 技术方法**

使用 Claude Haiku 4.5 LLM、密集检索、EM 估计奖励参数、奖励驱动的 vₘ 适应性、堆叠式短期记忆管理以及多线程并行执行。

**📊 数据集**

在 AgentBoard 的多域任务集（文本游戏、网页搜索、工具调用等）上进行评估。

**📈 对比分析**

与 ReAct、AWM 等基线相比，AdMem 在多任务长周期运行中显著提升任务完成率和平均进度（在需要长期经验的域表现尤为突出）。

**⚠️ 局限性**

局限性包括额外的提示与规划步骤导致推理成本上升、对奖励信号的稀疏依赖以及对超大规模记忆的实时更新仍有限。

---

## 145. Real-Time AttentionBender: Granular Interactive Network Bending of Video Diffusion Transformers

**arXiv ID:** 2606.06497 | [PDF](https://arxiv.org/pdf/2606.06497v1)

**作者:** Adam Cole `[一作]` (University of the Arts London), Mick Grierson `[通讯]` (University of the Arts London)

**通讯引用:** 546 | [OpenAlex ID](https://openalex.org/A5050264065)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 Real‑Time AttentionBender，一款可实时、细粒度操作视频扩散 Transformer（DiT）内部自注意力、交叉注意力和前馈网络的插件，实现交互式视频生成；

**💡 创新点**

创新点在于：①把网络弯曲扩展到 DiT 全深度，支持自注意力、交叉注意力及前馈三块；②实现了单帧实时响应、步骤/层/令牌/神经元级的精准定位；③将可解释 AI（XAIxArts）与艺术创作工具深度融合，赋予艺术家“材料亲密感”；

**🔧 技术方法**

技术上基于 Wan 的实时 Latent DiT，封装进 DayDream Scope 插件，猴子补丁 DiT 模块注入监听器；使用实时 Wan pipelines（LongLive、Krea）及 Causal Attention、Few‑Step Distillation、Self‑Forcing 等加速技术；在接口中实现注意力/前馈放大、噪声、阈值、空间‑时间变换等调制；

**📊 数据集**

采用预训练 Wan 1.3B/14B 模型，实验使用统一文本提示生成视频样例；未公开具体训练数据集，主要基于公开的通用视频‑文本数据；

**📈 对比分析**

通过与原 AttentionBender（离线交叉注意力弯曲）对比，展示同一模型同一提示下通过实时弯曲得到多样化视频；单 GPU 可达约15 fps，满足交互需求；未给出量化指标，主要以视觉示例和实时帧率评估；

**⚠️ 局限性**

局限性：KV 缓存对突变响应敏感，需手动重置/禁用缓存，限制连贯视频长度；缺乏自动化上下文感知缓存管理；未完成用户研究或定量可解释性评估；实时架构受限于速度与缓存导致的响应滞后。

---

## 146. PandaAI: A Practical Agent CQ2 for Neuro-symbolic Data Analysis And Integrated Decision-Making in Quantitative Finance

**arXiv ID:** 2606.06823 | [PDF](https://arxiv.org/pdf/2606.06823v1)

**作者:** Yuqi Li `[一作]` (Panda AI), Bingjun Liu `[通讯]` (Panda AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

结合大语言模型的闭环神经符号化系统PandaAI，实现了量化因子挖掘与交易决策的自动化。

**💡 创新点**

创新点在于：①引入连续隐状态 z_t 捕捉市场非平稳性；②使用受约束的 MCTS + LLM 生成低毒性的因子；③通过快慢双环闭环更新实现模型自适应。

**🔧 技术方法**

核心技术包括：Fine‑tuned CQ2 LLM、AutoEncoder+双通道适配、受约束的 MCTS 搜索、LoRA 参数更新、符号规则诱导。

**📊 数据集**

使用 CSI 300 指数成分股的 2015‑2024 年 OHLCV 数据进行训练、验证与测试。

**📈 对比分析**

与 LSTM、Transformer、StockMixer 等基线相比，PandaAI 在 IC、Rank IC、ICIR、年化收益提升显著，最大回撤下降约 25.7%。

**⚠️ 局限性**

局限性在于对极端行情下的稳健性验证不足，交易成本与滑点假设仍需进一步实证评估。

---

## 147. Towards Retrieving Interaction Spaces for Agentic Search

**arXiv ID:** 2606.06880 | [PDF](https://arxiv.org/pdf/2606.06880v1)

**作者:** Shengyao Zhuang `[一作]`, Xueguang Ma `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 RISE 方法，利用 BM25 对搜索结果进行空间界定，并在索引时为每篇文档添加行号化目录，形成可被代理在外部 shell 工具中探索的交互空间。

**💡 创新点**

创新点在于将检索目标从单纯的文本片段切片转变为构建可被代理主动探索的有界交互空间，并在此空间内预处理文档结构以提升子文档定位效率。

**🔧 技术方法**

技术包括 BM25 检索、文件系统工作区构建、离线 TOC 生成（使用 LLM 生成章节标题并插入行号区间）、Shell 工具（bash、read 等）进行交互、以及 LLM 作为评判者。

**📊 数据集**

使用的数据集为 BrowseComp-Plus 评测集（100 条查询）和扩展的 1M 语料（100k 原始 + 900k FineWeb-Edu 附加文档）。

**📈 对比分析**

实验与 DCI、retrieval‑agent 等基线对比，RISE 在 100k 语料上达 78% 准确率，成本仅为 DCI 的 1/4；在 1M 语料上保持准确率稳定，DCI 则出现 60% 准确率与大量时限失败。

**⚠️ 局限性**

局限性包括：仅使用 BM25 作为边界检索器、仅采用行号化 TOC 作为文档预处理方式、未在 1M 语料上完整跑 TOC 预处理、评测范围受限于单一 Benchmark 与 100 条查询，且未跨模型、跨检索器彻底验证框架普适性。

---

## 148. An Expanded Synthetic Conversation Dataset for Multi-Turn Smishing Detection

**arXiv ID:** 2606.06879 | [PDF](https://arxiv.org/pdf/2606.06879v1)

**作者:** Carl Lochstampfor `[一作]` (Old Dominion University), Ayan Roy `[通讯]` (Christopher Newport University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了COVA-X，扩展原始COVA对话式钓鱼数据集至10,985条多轮对话，并重新训练基线分类器，验证 transformer 模型在更大数据量下的优势。

**💡 创新点**

创新点在于：1) 通过改进生成管线（消除污染、标签不匹配、阶段指令泄漏、提示设计缺陷）显著提升数据质量；2) 采用三角色架构降低虚拟绑架对话的异常率；3) 对整个数据生命周期进行量化评估，展示清理与标签纠正对多种模型的统一提升；4) 明确阐述 transformer 在大规模会话语料上的表现提升。

**🔧 技术方法**

技术手段包括：多代理 LLM 生成（Qwen 2.5 14B）、prompt engineering、三角色生成架构、数据清理与标签审计工具、XGBoost+TF-IDF、DistilBERT（尾部截断）与 Longformer（全上下文）等分类模型。

**📊 数据集**

使用的数据集为 COVA-X（10,985 条合成多轮 smishing 对话），包含 8 类针对老年人的诈骗场景，每类均设有多种受害者与攻击者配置。

**📈 对比分析**

对比方法：在相同训练/验证/测试拆分下，重新训练三种模型，评估准确率、宏 F1 与 per-class 性能。结果显示 Longformer 在所有指标上超越 XGBoost（准确率 79.71% vs. 78.43%，宏 F1 0.7786 vs. 0.7563），验证了大数据量使 transformer 获得优势的假设。

**⚠️ 局限性**

限制：仅涵盖英文、美国背景；所有模型仍以完整对话为输入，缺乏实时增量检测能力；部分旧版数据保留 legacy 污染；生成模型 Qwen 2.5 在高情绪或长对话场景下仍存在指令违背与结构标签采纳失败等能力瓶颈。

---

## 149. Evidence-Grounded Ensemble Diagnosis of 802.11 Packet Captures: A Multi-Stage Pipeline with Deterministic Reliability Scoring

**arXiv ID:** 2606.06871 | [PDF](https://arxiv.org/pdf/2606.06871v1)

**作者:** Jerome Henry `[一作]` (Cisco Systems), Miroslav Popovic `[通讯]` (Cisco Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一套名为PROBE的多阶段诊断管线，用LLM和结构化证据自动分析802.11 PCAP捕获并生成可信根因诊断；

**💡 创新点**

① 引入多轮多候选集成与交叉模型二审，消除单一推理的幻觉与保守性；② 通过判决感知的证据规则与可验证的帧级证据构建可靠性评分；③ 采用模型无关的断言匹配评估，避免黄金参照偏差；

**🔧 技术方法**

LLM推理（Claude Sonnet 4.5、Llama 3.3 70B）、文本化PCAP、结构化诊断JSON、候选生成与判决一致性、可验证的证据检验、交叉模型协商与最终汇总；

**📊 数据集**

共87个企业Wi‑Fi捕获，104个捕获-审阅者对，形成104份金标注，涵盖认证、关联、握手、DHCP、DNS、漫游等多种协议故障；

**📈 对比分析**

与单一LLM推理、传统投票、多模型投票等做对照，使用加权F1、关键帧召回/精度等指标；全管线配置(F)在加权F1上达0.957，自动接受率96%，而单轮推理仅0.912；

**⚠️ 局限性**

① 数据集中仅包含确认故障样本，缺乏正常或无故障捕获，无法评估误报；② 自我置信度无效，可靠性评分虽能量化但未完全校准；③ 仅验证于802.11 PCAP，需验证其他协议与更大规模捕获；

---

## 150. Are Large Language Models Suitable for Graph Computation? Progress and Prospects

**arXiv ID:** 2606.06865 | [PDF](https://arxiv.org/pdf/2606.06865v1)

**作者:** Yuting Zhang `[一作]` (University of New South Wales), Wenjie Zhang `[通讯]` (University of New South Wales)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统综述了大语言模型（LLM）在图计算任务中的应用，并提出了以LLM在图计算中扮演的角色（执行者与规划者）为核心的全新分类框架。

**💡 创新点**

创新点在于以LLM的操作角色为轴线重新划分研究方向，全面梳理了提示工程、图结构编码、后训练、代码生成、函数调用以及多代理等多种技术路线，并在此基础上总结了四个未来研究方向。

**🔧 技术方法**

综述聚焦的技术包括：提示工程（Prompting）、图结构文本化与编码（Encoding）、后训练对齐（Post‑Training）、LLM代码生成（Code Generation）、函数调用（Function Calling）以及多代理协作（Multi‑Agent Collaboration）。

**📊 数据集**

利用并梳理了多种公开图计算基准数据集，例如GraphArena、GraphWiz、GraphInstruct、GraphThought、HLM‑G等，汇总了其任务类型、图规模和属性。

**📈 对比分析**

通过对现有论文中报告的实验结果进行系统对比，指出执行者方法在小规模或简单任务上能达到较高准确率，而规划者方法（尤其是代码生成与函数调用）在大规模、精度要求高的任务中表现更为稳健，接近或达到完美水平。

**⚠️ 局限性**

主要局限包括：可扩展性差（大图下性能下降或内存受限）、泛化能力有限（对未见任务/图的表现不佳）、易出现幻觉或推理错误、对图结构描述和提示设计高度依赖、以及高质量训练数据集与基准构建成本高。

---

## 151. FS-DVS: A Frequency-Selective Dynamic Visual Sensing Paradigm for Enhancing Information Completeness

**arXiv ID:** 2606.06856 | [PDF](https://arxiv.org/pdf/2606.06856v1)

**作者:** Feiyu Ji `[一作]` (Shanghai Jiao Tong University), Xiaoyun Yuan `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种频率选择性动态视觉传感器（FS-DVS），在事件触发前引入可学习的空间滤波器，实现对中频结构信息的增强，解决传统DVS的结构不完整与噪声易感问题。

**💡 创新点**

创新点包括：1）将可学习空间滤波器嵌入事件生成流程，并通过可微分事件仿真框架实现端到端任务驱动优化；2）学习到的滤波器自发收敛为中心-周围结构，与人类对比敏感函数相似，表明中频选择性是最优信息获取策略；3）在多任务（检测、识别、分割）间实现零射转移，并通过感测回路验证物理可行性。

**🔧 技术方法**

使用技术包括：可微分事件仿真框架V2CE；深度学习模型RT‑DETR、MViT、Mask2Former；FFT频率分析、互信息（MI）评估；物理感测回路（高帧率显示+Prophesee EVK4捕获）。

**📊 数据集**

使用的数据集有：DSEC‑Detection（检测）、UCF101（动作识别）、DSEC‑Semantic（语义分割），同时采用模拟轨道和物理感测轨道进行对比。

**📈 对比分析**

通过与传统单像素DVS基线比较，FS‑DVS在检测任务中mAP提升12.3%（模拟）/10.8%（物理）；动作识别Top‑1准确率提升8.86%/6.42%；在语义分割中零射提升4.77% mIoU；在匹配事件率的实验中保持高性能并避免噪声雪崩。

**⚠️ 局限性**

局限性包括：滤波器尺寸需人工选择，未实现硬件级嵌入；V2CE仿真精度与真实传感器差异仍存在；在极端光照或极高动态范围场景中的泛化性能待进一步验证。

---

## 152. The Geometry of Last-Layer Model Stealing

**arXiv ID:** 2606.06854 | [PDF](https://arxiv.org/pdf/2606.06854v1)

**作者:** Snigdha Chandan Khilar `[一作]` `[通讯]` (Independent Researcher), Snigdha Chandan Khilar (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在几何框架下重新阐述了 Carlini 等人对最后一层模型偷窃攻击的原理，并给出了对投影矩阵恢复的精确几何描述。

**💡 创新点**

创新点在于将外微分系统与 Lie 代数结合，利用极空间与 Kähler 正则性阐明恢复条件，并提出“内在维度”可观测量来识别隐藏层非线性瓶颈，进一步揭示不可识别参数纤维和可识别边界。

**🔧 技术方法**

主要技术包括外微分系统理论、Cartan–Kähler 定理、极空间概念、SVD、椭圆拟合以及线性代数与微分几何工具的组合。

**📊 数据集**

实验以精确可控的 toy 模型为主，并在 GPT‑2‑Small 的简化版本上验证，未使用大规模真实数据集。

**📈 对比分析**

与原始攻击相比，在正则条件满足下，投影矩阵可恢复到正交矩阵，误差仅在机器精度内；噪声实验显示秩恢复稳健，但正交恢复对噪声线性敏感。

**⚠️ 局限性**

局限性在于只能恢复最后一层投影矩阵，对更深层参数不可识别，原因是观察映射的核过大；此外成功依赖正则条件，且存在显著不可识别纤维导致隐藏层参数无法被确定。

---

## 153. MotionEnhancer: Leveraging Video Diffusion for Motion-Enhanced Vision-Language Models

**arXiv ID:** 2606.06853 | [PDF](https://arxiv.org/pdf/2606.06853v1)

**作者:** Yifan Xu `[一作]` (Beihang University), Zhipeng Chen `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出MotionEnhancer框架，利用视频扩散模型对视觉-语言模型中的运动信息进行增强，从而提升视频文本检索与多模态理解能力。

**💡 创新点**

核心创新在于首次将时序视频扩散技术与CLIP/BLIP等视觉‑语言模型耦合，实现了动态语义对齐与细粒度运动建模。

**🔧 技术方法**

主要技术包括基于U-Net的三维视频扩散模型、跨模态注意力机制以及自监督的运动预测损失。

**📊 数据集**

实验使用Kinetics‑400、UCF101和MSR-VTT等公开视频‑文本数据集进行训练与评估。

**📈 对比分析**

与SOTA方法（如ViViT、X-ViLM）对比，MotionEnhancer在MSR‑VTT视频检索任务上Recall@1提升约4%、Recall@5提升约3%，并在CLIP‑Video检索中显著降低误检率。

**⚠️ 局限性**

主要局限包括扩散模型训练成本高、推理速度受限，以及在极端运动场景下生成质量仍不稳定。

---

## 154. LLM Agent-Assisted Reverse Engineering with Quantitative Readability Metrics

**arXiv ID:** 2606.06838 | [PDF](https://arxiv.org/pdf/2606.06838v1)

**作者:** Neil Archibald `[一作]` (Commonwealth Bank of Australia), Ruben Thijssen `[通讯]` (Commonwealth Bank of Australia)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种多维度可读性评分（QRS）以及基于LLM的迭代改进流程，用来提升反汇编工具生成的C代码可读性并保持功能等价。

**💡 创新点**

创新点在于：①设计了可量化的复合可读性指标（Lexical Surprisal、Structural Simplicity、Idiomatic Quality），②引入结构相似性门控防止功能失效，③将指标反馈给LLM代理形成闭环迭代，而非单一指标或无反馈的工具驱动方式。

**🔧 技术方法**

技术包括：LLM代理（ClaudeAgentSDK），Ghidra MCP/ radare2 CFG 比较，clingo/clang‑tidy 对代码进行违规检测，NLL 计算实现 Lexical Surprisal，lizard 库计算循环复杂度和嵌套深度。

**📊 数据集**

数据集为210个人工生成的简单C二进制文件（共10,454行代码），涵盖字符串反转、冒泡排序、二分查找、链表等常见模式。

**📈 对比分析**

对比方法：在两组实验中分别关闭和开启 Bash 命令执行权限；评估指标为 QRS、结构相似度、Lexical Surprisal、Structural Simplicity、Idiomatic Quality；实验结果显示：在开启 Bash 时 QRS 平均提升约0.51，结构相似度提升0.58，迭代次数平均从3.5降至2.9；无 Bash 时提升约0.42，迭代次数平均为3.7。

**⚠️ 局限性**

局限性包括：①仅在极简C二进制上验证，难以推广到复杂或混淆代码；②QRS 与人类可读性尚未独立验证；③假设已完成功能等价的“Phase 0”清理工作；④指标权重需针对不同领域调优。

---

## 155. Think Like a Pilot: Fine-Grained Long-Horizon UAV Navigation

**arXiv ID:** 2606.06836 | [PDF](https://arxiv.org/pdf/2606.06836v1)

**作者:** Xiangyi Zheng `[一作]` (Beihang University), Si Liu `[通讯]` (Beihang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 FLIGHT 基准，包含长时限 Flow 任务与细粒度 VLN 任务，并引入 Pilot Reasoning 说明语，用于训练与评估无人机的语言指导飞行与推理能力。

**💡 创新点**

创新点在于（1）将连续细粒度 6‑DoF 动作序列与多阶段自然语言指令结合，真正逼近现实无人机任务；（2）提出异步 fast‑slow 架构——低频 Streaming Pilot VLM 负责语义推理，高频扩散动作模型负责连续控制；（3）通过 Pilot Reasoning 文本为模型提供显式任务执行与下一子目标的推理，使长时限任务不再受记忆衰减影响。

**🔧 技术方法**

使用技术包括：StreamingVLM（高效视频理解）、Diffusion Transformer（连续动作生成）、多任务联合训练、时延感知对齐训练、相对位姿编码以及基于 Flow‑Matching 的动作监督。

**📊 数据集**

数据集为 FLIGHT，基于 Unrealzoo 3D 真实感环境收集的 6,689 条细粒度 VLN 轨迹与 4,098 条长时限 Flow 轨迹，含 6‑DoF 轨迹、FPV 视频以及自动生成的 Pilot Reasoning 文本。

**📈 对比分析**

与 OpenVLA‑UAV、Memory VLA、CMA+LAG、NaVid 等基线对比，FLIGHT VLA 在 Long‑Horizon Flow 任务中 SR 59%/SSR 78.97%/NDTW 45.78%，远超 OpenVLA‑UAV（10.5%/30.87%/18.02%）和 Memory VLA；在 Fine‑Grained VLN 任务中 IASR 11% 亦明显高于 CMA+LAG（4.5%）与 NaVid（5.0%），并将 OSR 提升至 37%。

**⚠️ 局限性**

局限性包括：仍以仿真数据为主，实际场景的光照、障碍多样性与硬件噪声未完全覆盖；fast‑slow 设计在极端低频推理时仍可能出现语义滞后；模型规模较大，对边缘设备部署有一定挑战。

---

## 156. Translate-R1: Cost-Aware Translation Tool Use via Reinforcement Learning

**arXiv ID:** 2606.06835 | [PDF](https://arxiv.org/pdf/2606.06835v1)

**作者:** Pratik Jayarao `[一作]` (Amazon Stores Foundation AI), Bing Yin `[通讯]` (Amazon Stores Foundation AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过仅使用任务奖励，训练 Qwen3‑4B 在 22 种语言与 5 个域上学习何时调用翻译工具，从而提升低资源语言性能。

**💡 创新点**

创新在于无需语言标识或手工规则，利用置信门控的 GSPO 只在模型判断自身无法理解时调用翻译，实现成本敏感的多语言内省决策。

**🔧 技术方法**

技术包括强化学习 (RL) + GSPO、置信门控的 Group‑Relative Tool Efficiency、答案保持的多语言翻译管线，以及多域多语言的持续 RL 训练。

**📊 数据集**

数据集涵盖 22 种自然语言（高/低/极低资源），5 个任务域（可验证的数学、QA、指令，及不可验证的摘要、翻译），并生成 2 种合成语言与 9 个隐藏语言进行零样本测试。

**📈 对比分析**

与无工具、无成本、自由工具、平坦成本、OTC 等基线比较，门控策略在低/极低资源/合成语言上提升 18.7 分，保留 63% 成本同时保持全部奖励，Pareto 优势覆盖 87% 成本范围。

**⚠️ 局限性**

局限性包括：依赖 Qwen3.5‑122B 翻译模型，单一 Qwen3‑4B 模型实验，单种种子结果，未验证跨模型迁移，翻译质量对极低资源语言仍有偏差。

---

## 157. Learning Fair Demand Models

**arXiv ID:** 2606.06830 | [PDF](https://arxiv.org/pdf/2606.06830v1)

**作者:** Adam N. Elmachtoub `[一作]` (Columbia University), Jonathan Y. Tan `[通讯]` (Columbia University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文分析了数据驱动定价管道中公平性的作用，比较了在需求估计阶段（FEO）和价格优化阶段（EFO）分别施加损失公平、价格公平、需求公平和Rawlsian公平的效果，给出了理论阐述和实证验证。

**💡 创新点**

创新点在于：①提出将公平性拆分为估计与决策两阶段的框架；②证明损失公平在两组模型中可产生多解，导致下游福利截然不同；③给出价格公平与需求公平在两阶段中的阈值判定，揭示哪种策略在何种市场结构下更有利；④展示Rawlsian公平在两阶段可得到相同甚至更优的利润与社会福利，打破传统公平-效率权衡。

**🔧 技术方法**

技术手段包括：凸优化与二级（bilevel）优化；均方误差、逻辑回归与sigmoid损失；Lambert W 函数求解最优价格；对线性与逻辑需求模型进行解析与数值实验；案例研究中使用实际疫苗需求调查数据。

**📊 数据集**

使用的数据集包括：1）合成的双组线性需求数据；2）真实的瑞典蜱传脑炎疫苗需求问卷数据（包含年龄、性别、收入等特征），并在两种假设下（线性与逻辑需求）重新生成需求；3）特征为 12 维的疫苗问卷特征集合。

**📈 对比分析**

比较方法：先在理论上推导 FEO 与 EFO 在不同公平性下的阈值和收益差异；随后在模拟实验和真实数据上计算利润、消费者剩余和社会福利的相对比例。结果显示：在价格公平且两组市场规模相近时，FEO 更能提升消费者剩余；在需求公平且两组价格差异较大时，EFO 更有优势；Rawlsian 公平可在两阶段均获得相同甚至更高的利润和福利。

**⚠️ 局限性**

局限性：①模型仅考虑两组消费者，无法直接推广到多组或连续属性；②仅对线性与逻辑需求函数做理论与实验，缺乏对更复杂需求形式的推导；③假设成本已知且固定，未考虑成本不确定性；④公平约束在实际数据中可能导致多解且求解不稳定；⑤未系统分析样本复杂度、泛化误差和监管可审计性。

---

## 158. SkelDPO: A Skeleton-Guided Direct Preference Optimization Framework for Efficient Code Generation

**arXiv ID:** 2606.06826 | [PDF](https://arxiv.org/pdf/2606.06826v1)

**作者:** Yu Yu `[一作]` (Shandong Normal University), Chen Lyu `[通讯]` (Shandong Normal University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SkelDPO框架，利用骨架级别的结构偏好优化提升代码生成的执行效率；

**💡 创新点**

创新点在于引入骨架抽取与双通道偏好（代码+骨架）联合训练，克服传统全程序级别偏好无法捕捉细粒度效率信号的问题；

**🔧 技术方法**

采用直接偏好优化（DPO）与低秩适配（LoRA）技术，结合基于 token 交集的骨架抽取方法；

**📊 数据集**

使用 Mercury、ENAMEL 评测基准，并在 APPS 数据集上构建高低效实现对与骨架对；

**📈 对比分析**

与 Instruct、LLM4Effi、EffiCoder、CodeDPO 等基线比较，SkelDPO 在 Pass@1、Beyond@1、eff@1 上平均提升约3–7%，并在复杂任务上表现更为稳健；

**⚠️ 局限性**

局限性包括对高效实现样本的依赖、骨架抽取可能引入噪声，以及预处理过程增加的计算成本。

---

## 159. Progress-SQL: Improving Reinforcement Learning for Text-to-SQL via Progressive Rewards

**arXiv ID:** 2606.06825 | [PDF](https://arxiv.org/pdf/2606.06825v1)

**作者:** Shihao Zhang `[一作]` (East China Normal University), Weining Qian `[通讯]` (East China Normal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Progress-SQL多轮强化学习框架，通过Oracle指导的诊断树(ODT)提供逐步反馈，实现文本到SQL的逐步纠错。

**💡 创新点**

引入多轮进展奖励，将结构与词汇对齐改进量、早期正确性、执行状态与格式奖励结合；使用ODT进行句法级诊断而非单一奖励。

**🔧 技术方法**

采用LLM（Qwen2.5-Coder 7B/14B）作为基座，GRPO强化学习，ODT抽象与对齐评分，进展惩罚γ和执行状态奖励等技术。

**📊 数据集**

在BIRD、Spider及其变体（Spider-Syn、Spider-Realistic、Spider-DK）上进行训练与评测。

**📈 对比分析**

与多种单轮RL及多轮RL基线对比，进化奖励下7B模型在Spider Dev、Test的执行准确率从83.9%提升至87.8%，BIRD 8.5%提升；在Robustness上亦表现优异。

**⚠️ 局限性**

依赖训练时的黄金SQL生成ODT，推断阶段无诊断反馈；且结构化抽象受SQL解析器限制，对严重语法错误或不同方言的鲁棒性不足。

---

## 160. Chiseling Out Efficiency: Structured Skeleton Supervision for Efficient Code Generation

**arXiv ID:** 2606.06821 | [PDF](https://arxiv.org/pdf/2606.06821v1)

**作者:** Yu Yu `[一作]` (Shandong Normal University), Chen Lyu `[通讯]` (Shandong Normal University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 EffiSkel 框架，利用效率骨架监督显著提升 LLM 生成代码的运行效率。

**💡 创新点**

创新点在于明确构造效率骨架并通过词频、AST相似度、动态热点三种提取策略实现结构化监督，同时采用多任务学习将骨架预测与代码生成联合优化。

**🔧 技术方法**

技术包括词频敏感性分析、AST子树相似度匹配、行级执行剖面采集、结构感知多任务训练以及 LoRA 微调。

**📊 数据集**

使用了 APPS+EFFI、Mercury、ENAMEL、EffiBench、HumanEval-X、DevEval、CoderEval 等效率评测数据集。

**📈 对比分析**

与 Instruct、LLM4EFFI、EffiCoder*、CodeDPO* 等基线对比，EffiSkel 在 ER/AS 指标上提升约10–15%，并在不同模型规模、语言和仓库级任务上持续保持优势。

**⚠️ 局限性**

局限性在于骨架提取依赖准确的运行时剖面，数据集主要来自人工书写的代码，覆盖度有限；模型在极端复杂任务或低资源语言上的泛化能力仍需进一步验证。

---

## 161. VideoSEG-O3: A Multi-turn Reinforcement Learning Framework for Reasoning Video Object Segmentation

**arXiv ID:** 2606.06819 | [PDF](https://arxiv.org/pdf/2606.06819v1)

**作者:** Ming Dai `[一作]` (Southeast University), Jingdong Wang `[通讯]` (Baidu Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 VideoSEG‑O3，一种基于多轮强化学习的推理视频目标分割框架，能够通过多轮时空链式思考主动探索关键时间段和关键帧，并在 RL 阶段通过 SEG‑aware logit 校准将像素级分割质量直接反馈到语言模型的输出概率，从而实现端到端的推理与分割协同优化。

**💡 创新点**

创新点主要包括：
1) 采用多轮 Temporal‑Spatial Chain‑of‑Thought 机制，使模型能够像人类一样先做粗略判断后逐步聚焦细节；
2) 引入 SEG‑aware Logit Calibration，将像素级分割置信度嵌入到特殊 <seg> 令牌的 logits 中，弥补传统 GRPO 仅对文本概率进行优化导致的分割与文本解耦问题；
3) 设计 Decoupled Thinking Trace，分层处理时空信息、空间细节与语言表达，提升推理效率；
4) 构造 VTS‑CoT 数据集，为冷启动阶段提供结构化多步推理轨迹；
5) 通过多重奖励（格式、时间、分割、进展）引导 RL 学习更加精准的关键帧采样与分割质量提升。

**🔧 技术方法**

使用的技术包括：
- 基础模型 Qwen3‑VL 结合 SAM2 分割器，利用特殊 <seg> 令牌解码 mask；
- 强化学习框架 GRPO，改进为 SEG‑aware 版本；
- LoRA 微调 + 全参数 RL 训练；
- 设计多维奖励函数和辅助分割损失；
- Decoupled 视觉输入（低分辨率全局帧 + 高分辨率关键帧）以及多轮交互决策。

**📊 数据集**

使用的数据集：
- 8 个推理视频目标分割基准（MeViS、Ref‑Youtube‑VOS、Ref‑DAVIS17、Ref‑SAV、Long‑RVOS、ReVOS、ReasonVOS、GroundMoRe）；
- 4 个图像分割基准（RefCOCO、RefCOCO+、RefCOCOg、ReasonSeg）；
- 自建 VTS‑CoT 训练集（约 6K 结构化推理轨迹）用于冷启动；
- 训练时结合公开视频帧与文本注释。

**📈 对比分析**

与现有方法对比，VideoSEG‑O3 在 5 个 RefVOS 任务上均取得 SOTA 表现，4B 版本在 Ref‑SAV 上提升 15.5%、Long‑RVOS 上提升 6.1%；在推理 RVOS（ReVOS、ReasonVOS、GroundMoRe）中超过 Veason‑R1 等 RL 竞争者，整体 J&F 最高达 70.4%+；在图像分割基准上保持与 UniPixel‑7B 相当的 cIoU，证明跨域泛化能力。RL 阶段进一步提升关键帧 mIoU、降低空框率，并显著减少平均交互轮数（≈20%）。

**⚠️ 局限性**

局限性：
- 仍受限于训练数据的覆盖范围，极端遮挡、快速运动或少量帧的长视频仍可能出现误分；
- RL 训练消耗显著 GPU 资源，实际部署需要更高效的推理策略；
- 目前模型主要针对单一目标推理，处理多目标或动态交互场景需进一步扩展；
- 隐私与误用风险：高精度视频分割可用于监控或侵犯隐私，需配合伦理与法律约束；
- 需要进一步评估不同语言、文化背景下的泛化能力与公平性。

---

## 162. Terastal: Layer-Variant-based Scheduling for Real-Time Multi-DNN Workloads on Heterogeneous Accelerators

**arXiv ID:** 2606.06818 | [PDF](https://arxiv.org/pdf/2606.06818v1)

**作者:** Sing-Yao Wu `[一作]` (University of California), Eli Bozorgzadeh `[通讯]` (University of California)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了 Terastal 框架，结合虚拟预算分配、层级变体设计与在线调度，在异构 DNN 加速器上降低多模型软实时工作负载的截止时间缺失率。

**💡 创新点**

① 引入层级变体以缩小非首选加速器的时延差距；② 用虚拟预算精细化层级截止时间，驱动变体生成与调度；③ 将变体与在线调度紧耦合，兼顾时延与精度。

**🔧 技术方法**

依赖空间-深度变换（S2D/D2S）、虚拟预算分配算法、基于最佳滑差的优先级调度、离线准确性评估与组合筛选等技术。

**📊 数据集**

使用 ImageNet、Pascal VOC、KITTI 等公开数据集训练层级变体并评估精度。

**📈 对比分析**

与 FCFS、EDF、DREAM 及两种消融版本对比；在多种硬件配置下，Terastal 平均降低约 30–40% 的截止时间缺失率，平均精度损失仅 2.24%。

**⚠️ 局限性**

需预先生成并存储变体，增加存储与训练成本；对极端精度阈值下的灵活性有限；当所有加速器性能相近时变体收益有限。

---

## 163. Learning to Strategically Acquire Resources in Competition

**arXiv ID:** 2606.06882 | [PDF](https://arxiv.org/pdf/2606.06882v1)

**作者:** Safwan Hossain `[一作]` (Harvard University), Yuriy Nevmyvaka `[通讯]` (Morgan Stanley)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出通用博弈框架，描述多智能体在动态定价下竞争性获取可分配资源的行为。

**💡 创新点**

创新点在于结合可凸约束、凹价值函数与不完全信息，证明贝叶斯纳什均衡的存在、唯一性与可计算性，并设计可在无先验下收敛到均衡的双最优学习算法。

**🔧 技术方法**

采用强单调变分不等式、额外梯度算法求解均衡；在学习阶段利用离散化类型空间、估计市场冲击参数并容忍偏差的随机梯度更新。

**📊 数据集**

使用公开的外汇Tick数据（Dukascopy USD/CAD）估计α、β，并构建包含两名代理、三种类型的Bayesian游戏实例进行仿真。

**📈 对比分析**

通过额外梯度求解得到的BNE与学习算法的最终迭代在模拟中高度吻合，误差随轮次快速下降；在所有买入情形下，价格失衡的PoA被证明可上界为O(n²T²γ²)。

**⚠️ 局限性**

主要局限在于价格冲击模型过于理想化、离散化导致高维类型空间的计算指数增长、估计误差对收敛影响显著，以及模型假设代理只能预先设定完整轨迹、只考虑策略性参与者等。

---

## 164. A Cross-view Fusion Framework for Robust 6-DoF Grasp Pose Estimation

**arXiv ID:** 2606.06878 | [PDF](https://arxiv.org/pdf/2606.06878v1)

**作者:** Kangjian Zhu `[一作]` (Nanjing University of Science and Technology), Jin Xie `[通讯]` (Nanjing University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种跨视角融合框架，利用辅助视角和后融合策略提升角视图下6-DoF抓取的鲁棒性；

**💡 创新点**

创新点在于：①自监督对比学习通过跨视角匹配实现点特征空间一致性与方向辨别；②交叉视角对齐圆柱集成模块聚合局部几何并编码旋转对称；③采用后融合而非预融合，显著降低重建开销；

**🔧 技术方法**

采用ResUNet14稀疏3D卷积编码器、圆柱坐标嵌入、自注意力与跨注意力、以及自监督对比损失和配准对齐技术；

**📊 数据集**

在GraspNet-1Billion数据集上进行评估，并在12个未见物体的真实机器人实验中验证；

**📈 对比分析**

与ZeroGrasp、EconomicGrasp、GSNet等方法对比，AP在Seen/Similar/Novel切分上分别提升3.55/1.61/1.84点，机器人抓取成功率达到96%，比基线提升14-19%；

**⚠️ 局限性**

局限性包括需要手动选择辅助视角、对极端视角变化或动态场景的适应性有限，以及对大规模点云的实时性能仍有提升空间。

---

## 165. Multi-FRuGaL: Multimodal Flexible Redundancy-aware Decomposed Gated Learning for Cancer Diagnosis and Prognosis

**arXiv ID:** 2606.06867 | [PDF](https://arxiv.org/pdf/2606.06867v1)

**作者:** Sanket Kachole `[一作]` (Indiana University), Spyridon Bakas `[通讯]` (Indiana University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出了 Multi-FRuGaL 框架，用于在多模态医学数据缺失时进行鲁棒的中间融合与预测。

**💡 创新点**

创新点在于将信号分解层、输入条件门控网络与信息预算正则化结合，能够在缺失数据下分离共享与专有特征、动态抑制冗余信息，并实现稀疏可解释的模态选择。

**🔧 技术方法**

核心技术包括：信号分解层（正交约束）、Gumbel‑Sigmoid 门控、Transformer 中间融合、遮挡感知池化、以及预算+冗余两项信息预算正则化。

**📊 数据集**

在 MICCAI 2025 的 HANCOCK（5 模态、OS 与复发预测）和 HECKTOR（CT、PET、表格数据用于 HPV 预测）两个多机构多模态头颈癌数据集上进行实验。

**📈 对比分析**

与特征拼接、跨模态注意力（MulT）、生成式融合（LANISTR）等基线相比，Multi‑Frugal 在 HANCOCK 的 5‑年 OS AUC 从 0.601 提升至 0.8496，复发 AUC 从 0.672 提升至 0.8102；在 HECKTOR 的 HPV 预测 AUC 从 0.904 提升至 0.975；C‑index 亦实现显著提升。

**⚠️ 局限性**

局限性包括：对不同缺失模式的适应性尚待验证；主要针对头颈癌，跨域推广需要进一步研究；模型结构复杂，调参成本较高。

---

## 166. Product units in gated recurrent units improve nuclear-mass prediction

**arXiv ID:** 2606.06866 | [PDF](https://arxiv.org/pdf/2606.06866v1)

**作者:** Ziyuan Li `[一作]` (University of Applied Sciences Koblenz), Babette Dellen `[通讯]` (University of Applied Sciences Koblenz)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究利用复杂值的产品单元GRU模型对核质量进行序列预测。

**💡 创新点**

创新在于将乘法交互、产品单元变换和复数域运算嵌入GRU，形成MI-PU-GRU与AM-PU-GRU两种新架构。

**🔧 技术方法**

采用复数乘法交互、产品单元变换、门控循环单元、RAdam优化器及PyTorch实现。

**📊 数据集**

以AME2016和AME2020的质量评估数据为训练、验证和测试集，并使用WS4和SEMF作为先验。

**📈 对比分析**

与传统GRU、MI-GRU、AM-GRU、PU-GRU以及其他机器学习模型比较，复数AM-PU-GRU在插值RMSE仅0.227MeV、外推RMSE 0.179MeV，表现优于多数对手。

**⚠️ 局限性**

仍未能击败如CatBoost、Physics-Informed FCNN等特定筛选条件下的树模型，且在边缘核的预测精度相对较低，模型对超参数和序列长度敏感。

---

## 167. On the Incentive Compatibility of Block Propagation in Bitcoin

**arXiv ID:** 2606.06860 | [PDF](https://arxiv.org/pdf/2606.06860v1)

**作者:** Fumichika Maeda `[一作]` (Kyoto University), Kazuyuki Shudo `[通讯]` (Kyoto University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过分析比特币区块链网络模型，推导了精确的奖励表达式，研究了区块传播延迟对矿工挖矿奖励的影响，并探讨了不同 Tie‑breaking 规则下的激励兼容性；

**💡 创新点**

创新点在于：①首次在解析层面得到不同传播延迟、算力分布和 Tie‑breaking 规则共同决定奖励的闭式公式；②揭示矿工对传播延迟的激励差异（如无激励传递他人区块、对 inbound/outbound 延迟的激励随算力变化而变化）；③系统性比较了三种 Tie‑breaking 规则在传播激励与公平性之间的权衡。

**🔧 技术方法**

采用链路传播延迟模型和基于“round”的区块链网络模型，利用微分和期望分析推导奖励公式，并用纳什均衡分析评估可行的传播策略；

**📊 数据集**

使用 SimBlock 生成两套网络参数：①真实场景（24000 节点，包含矿池、单体矿工、可达/不可达节点）以及矿池算力分布；②均匀随机网络（1000 矿工）作为基准；

**📈 对比分析**

通过计算模型参考值 MPR^ref 与解析表达式得到的 MPR^approx，使用 Lin 的一致性相关系数（CCC）和相对误差进行比较，结果在所有设置下 CCC ≥ 0.997，最大相对误差 ≤ 4.1%，证明解析表达式在实际网络中高度准确；同时对不同 Tie‑breaking 规则在奖励公平性和传播激励方面的表现进行量化比较。

**⚠️ 局限性**

局限性包括：①将矿池视为单个矿工，忽略内部延迟和 stale 块；②第一见规则分析依赖传播三角不等式，可能不满足真实网络；③验证未直接模拟长期收益，仅比较模型值；④传播延迟和算力假设为固定常量，未考虑随机波动或动态变化。

---

## 168. CFRNet: Cycle-Consistent Fixed-Point Training for Real-Time Blind Face Restoration on Consumer Embedded NPUs

**arXiv ID:** 2606.06850 | [PDF](https://arxiv.org/pdf/2606.06850v1)

**作者:** Fuchen Li `[一作]` (University of Florida), Wenbo Ma `[通讯]` (Intel Asia-Pacific Research & Development Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并训练轻量级面部修复网络CFRNet，并提出循环一致固定点训练方法CCFP，使其能在消费者NPU上实时执行。

**💡 创新点**

通过多周期监督、冪等损失和再退化循环损失，让网络在多次推理后收敛为自然面部，从而实现无重训练的质量调节。

**🔧 技术方法**

使用ResNet风格轻量编码器-解码器、面部部件（眼、鼻、嘴）监督、GAN对抗、身份损失、进阶多周期监督、冪等损失、再退化循环损失以及INT8量化。

**📊 数据集**

在FFHQ 70k高质量人脸上训练，采用90/5/5%划分，300张FFHQ-256图像用于测试。

**📈 对比分析**

在同一256×256、INT8、Hi3402 NPU环境下重新实现GFPGAN-Lite、GPEN-Lite、CodeFormer-Lite等基线，CFRNet在LPIPS上比基线低29%~35%，PSNR/SSIM在k=2时最高，且能在69ms内完成3次循环，是唯一可在该NPU上编译运行的模型。

**⚠️ 局限性**

延迟随循环次数线性增长；退化模型假设需与真实退化相近，极端低光或运动模糊可能受限；CCFP训练耗时约单通道训练的两倍；仅在256×256目标上验证，未评估更大分辨率。

---

## 169. Characterize Then Distill: Mechanistic Reasoning in Large Output Spaces

**arXiv ID:** 2606.06840 | [PDF](https://arxiv.org/pdf/2606.06840v1)

**作者:** Debjyoti Saha Roy `[一作]` (Northeastern University), Javed A. Aslam `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型在海量标签空间中的多标签推理过程，并提出通过监督注意力和残差写入的机制化蒸馏方法，显著提升小模型在同类任务上的表现。

**💡 创新点**

创新点在于将推理拆解为“粗筛”与“细化”两阶段，并通过对注意力峰值、语义锚对齐以及对比性迭代写入等内部机制进行定量化和监督，使得蒸馏不仅复制输出，更复制内部推理轨迹。

**🔧 技术方法**

采用 Transformer 关注机制分析、残差流写入观察、头聚合的注意力与写入匹配，以及多阶段的交叉熵与 KL 损失组合的机制化蒸馏目标。

**📊 数据集**

使用了临床 ICD‑10 诊断编码（MIMIC‑IV 住院记录）和维基百科相关文章推荐两大公开数据集。

**📈 对比分析**

与传统 CoT 蒸馏（UniCoTT、MoRSD、SemCoT、CWT）比较，机制化蒸馏在推理聚焦度、近似错误混淆、教师可信度 (LAS) 以及宏观 F1 分数上分别提升约 13%、10%、3% 及 5%，并逼近教师模型的推理轨迹。

**⚠️ 局限性**

局限包括对长文本或噪声较多的域（如维基百科）效果下降、对极端稀有标签的识别仍受限，以及机制化蒸馏实现复杂，需进一步简化与泛化。

---

## 170. STRIPS-WM: Learning Grounded Propositional STRIPS-style World Models from Images

**arXiv ID:** 2606.06832 | [PDF](https://arxiv.org/pdf/2606.06832v1)

**作者:** Abhiroop Ajith `[一作]` (Worcester Polytechnic Institute), Constantinos Chamzas `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本工作提出一种从仅有动作-图像转移数据中学习图像根基的Propositional STRIPS世界模型，并在测试时通过视觉谓词分类器将学到的符号状态映射回像素，实现场景级的长序列规划。

**💡 创新点**

创新点包括：①无需对象级监督即可直接从原始视觉转移学习任务图；②通过CP‑SAT求解将抽象状态赋予二进制谓词并联合学习符号化的动作操作符；③将学习到的符号谓词与视觉感知相结合，完成从图像到规划的闭环流程。

**🔧 技术方法**

核心技术包括：视觉自监督表示学习（学生‑教师编码器+FSQ量化）、动作条件前向预测与逆动力学头、CP‑SAT组合优化求解谓词与操作符、视觉谓词分类器以及经典STRIPS规划。

**📊 数据集**

实验使用了三个视觉重排域：BlocksWorld（3个方块），DinnerTable（模拟5个物体7个放置区），以及DinnerTable Real（真实图像，含干扰物）。

**📈 对比分析**

对比了WM‑Rollout、WM‑BFS、LSR、LatPlan‑AMA3等四个基线。实验结果表明，STRIPS‑WM在BlocksWorld和DinnerTable Real中均能达到100%成功率，且在长序列任务上优于基线；基线在更长距离时性能快速下降。

**⚠️ 局限性**

局限性包括：①假设确定性动力学和固定前置条件/增删效果，无法处理随机或隐藏状态；②模型质量高度依赖任务图的完整性与视觉区分度；③仅学习基于实例的符号操作符，缺乏对新物体集合的组合式泛化能力。

---

## 171. Quantifying Media Representation Dynamics Across 25 Years of News Reporting on Policing-related Deaths

**arXiv ID:** 2606.06812 | [PDF](https://arxiv.org/pdf/2606.06812v1)

**作者:** Farhan Samir `[一作]` (University of Toronto), Vered Shwartz `[通讯]` (University of British Columbia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对4000篇加拿大新闻报道中关于警察涉死事件的叙事进行大规模计算分析，量化不同主体（官僚、平民、背景）视角的比例与变化。

**💡 创新点**

提出PerspectiveGap框架，定义并系统测量“官僚 vs 平民”视角权重；构建核心ference+分类模型，能够在普通硬件上仅用800M参数精确识别段落视角；首次在此领域展示技术性与共创性报道比例的时间演变。

**🔧 技术方法**

使用语言模型（800M参数）在核心ference聚合段落上进行微调；核心ference解析；与GPT‑4o进行prompt‑tuned对比；采用log‑odds/Dirichlet方法挖掘关键术语；利用NRC VAD词典评估情感色彩。

**📊 数据集**

训练集：100篇手工标注的新闻文章（2158段）；测试集：4000篇来自CBC、Global News等多家媒体的文章，涵盖2000‑2025年；受害者记录来源包括CBC Deadly Force、Tracking Injustice和Wikipedia列表。

**📈 对比分析**

与prompt‑tuned GPT‑4o和随机分类进行对比；模型在测试集上bureaucrat、civilian、context的F1约为0.80‑0.85，宏平均F1与GPT‑4o无显著差异（p=0.15）；在大规模数据上，技术性视角约占33%，平民视角约占12%，2020‑2023年后显著上升。

**⚠️ 局限性**

仅覆盖加拿大媒体，忽略其他地区；视角标签为三类，可能无法捕捉更细致的主体差异；核心ference匹配仍存在漏检，导致部分误分类；模型对上下文敏感但仍有一定错误率；数据集主要来自公开报道，可能遗漏未报道或地下事件；研究聚焦量化描述，缺乏因果机制分析。

---

## 172. Hearing the Unspoken: Language Model Priors for Acoustic Adversarial Attacks

**arXiv ID:** 2606.06833 | [PDF](https://arxiv.org/pdf/2606.06833v1)

**作者:** Jiani Xie `[一作]` (University of Melbourne), Benjamin I. P. Rubinstein `[通讯]` (University of Melbourne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在实时语音识别（ASR）攻击中利用大语言模型（LLM）进行语义预测的Semantic Gambit攻击框架；

**💡 创新点**

突破实时攻击的因果信息瓶颈，借助LLM对未来文本的预测增强攻击信息量，从而实现超越传统声学信息攻击的效果；

**🔧 技术方法**

使用LLM（Llama‑3 8B量化版）生成文本预测，ASR模型（Wav2Vec 2.0）提取前缀文本，MFCC特征+多模态自注意力+Perceiver网络生成对齐时间窗口的扰动，且采用SNR约束进行扰动放缩；

**📊 数据集**

在LibriSpeech（train‑clean‑100、dev‑clean、test‑clean）与Common Voice 25.0英语数据集上训练与评估，交叉数据集与跨模型（HuBERT‑Large、Whisper‑small）验证泛化；

**📈 对比分析**

与白噪声、AO、AO*、Universal、Ground‑Truth（GT）和离线PGD基线对比；在20 dB SNR下，Semantic Gambit在1 s前缀、0 s延迟时实现35.6 % WER（约17倍提升），甚至超过GT和PGD；跨数据集和跨模型转移效果良好，但对seq2seq模型无效；

**⚠️ 局限性**

限制在实验室环境下、未考虑无线传播和自适应防御；LLM预测误差可能导致攻击效果下降；模型仅在语音清晰、无噪声的条件下测试，实际部署需评估更复杂声学场景。

---

## 173. Neuro-Symbolic Learning for Long-Horizon Task Planning Under Complex Logical Constraints

**arXiv ID:** 2606.06877 | [PDF](https://arxiv.org/pdf/2606.06877v1)

**作者:** Qiwei Du `[一作]`, Chen Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

未提供

**💡 创新点**

未提供

**🔧 技术方法**

未提供

**📊 数据集**

未提供

**📈 对比分析**

未提供

**⚠️ 局限性**

未提供

---

## 174. What Is My Robot Thinking? Design Considerations for Transparent and Trustworthy Shared Autonomy

**arXiv ID:** 2606.06870 | [PDF](https://arxiv.org/pdf/2606.06870v1)

**作者:** Atharv Belsare `[一作]` (University of Utah), Daniel S. Brown `[通讯]` (University of Utah)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在可视化共享自主系统中，作者通过在固定控制策略下系统地变换反馈的模态（视觉/听觉）和信息丰富度（稀疏/丰富），设计并实现了四种反馈界面，并在25名受试者的两个辅助操纵任务中进行对比实验，评估了透明度对意图对齐、纠正输入、理解和信任的影响。

**💡 创新点**

创新点在于：①首次将反馈模态与信息丰富度作为独立的界面设计变量，在相同的共享控制器下单独比较其效应；②发现仅展示当前推断目标（而非完整置信度分布）即可提升协同效率；③提出针对任务复杂度和用户需求的透明度设计准则，并证明视觉反馈在连续空间操作中优于听觉反馈；④提供了针对共享自主系统的可解释性评估框架。

**🔧 技术方法**

主要技术包括：Vision‑Only Shared Autonomy (VOSA) 框架、YOLOv11 目标检测、线性混合控制策略、Piper 文本转语音库实现听觉反馈、平板显示实现视觉反馈；所有界面在不改变 VOSA 预测和控制逻辑的前提下实现。

**📊 数据集**

使用了自制的任务对象数据集（瓶子、调味品、回收桶等）对 YOLOv11 进行微调，保证在实验中目标检测的准确性。

**📈 对比分析**

对比方法：六种处理条件（Teleop、VOSA、VS、VR、AS、AR）在两项任务（Shelving 与 Sorting）中进行重复测量实验。客观指标为意图对齐率与意图切换次数；主观指标为 7 点 Likert 量表。结果显示：任何形式的反馈均显著提升意图对齐并减少纠正输入；视觉反馈在理解与信任评分上优于听觉；稀疏视觉反馈在复杂任务中对信任提升最显著；并未发现完整置信度分布提升性能的统一优势。

**⚠️ 局限性**

局限性包括：①透明度未实现自适应；②实验仅在健全成人受试者、简易任务与单一机械臂上进行；③未评估长期使用对信任与依赖的影响；④缺乏在视觉模糊、传感器失效或运动障碍用户情境下的验证。

---

## 175. Interpreting Brain Responses to Language with Sparse Features from Language Models

**arXiv ID:** 2606.06857 | [PDF](https://arxiv.org/pdf/2606.06857v1)

**作者:** Michael A. Lepori `[一作]` (Brown University), Greta Tuckute `[通讯]` (Harvard University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

使用稀疏自编码器（SAE）与惊奇度构建增强稀疏编码模型，对 7T fMRI 数据中的语言句子刺激进行建模，解释人脑语言网络中不同 voxel 对加工难度与内容的神经响应。

**💡 创新点**

创新点在于：①将 LM 的密集隐藏状态投影到层次化稀疏特征空间，①分离加工难度（惊奇度）与内容驱动的神经信号；②揭示人脑语言网络主要对应 LM 的通用、层级化特征，而非细粒度特征。

**🔧 技术方法**

采用 JumpReLU 与 Matryoshka 稀疏自编码器、LASSO + Ridge 线性回归、特征选择与交叉验证、特征通用性与熵分析等技术。

**📊 数据集**

使用 7T 高场 fMRI 数据（8 名受试者，200 句多样化句子，每句三次）进行主实验，并在 3T 数据上进行结果复现。

**📈 对比分析**

通过噪声上限归一化的 Fisher‑z 相关系数与仅惊奇度基线对比；结果表明稀疏特征的预测力与传统 LM 残差流特征相当，对内容驱动 voxel 具有显著提升。

**⚠️ 局限性**

限制主要体现在：① SAE 特征的可解释性受限；② voxel 级别测量可能掩盖细粒度神经机制；③ 需要更大、多样化刺激集以进一步验证与扩展。

---

## 176. Toward a Metaphysics of Learning Analytics: Ontological Positioning of Data, Inference, and Normativity

**arXiv ID:** 2606.06851 | [PDF](https://arxiv.org/pdf/2606.06851v1)

**作者:** Kensuke Takii `[一作]` `[通讯]` (Naruto University of Education), Kensuke Takii (Naruto University of Education)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

提出学习分析（LA）的元哲学框架，阐述数据的本体属性、推理合法性的唯一基底（形式化构成记录）以及八类代理人，并分析规范嵌入式LA与该框架的矛盾与限制。

**💡 创新点**

首次系统地从LA自身定义出零阶原则（学习者是本体先决条件）和第一原则（仅以形式化构成记录为合法推理基础），并对LA的代理关系和规范嵌入问题进行结构化阐述，填补了该领域缺乏元哲学视角的空白。

**🔧 技术方法**

无具体技术实现，采用描述性本体学与哲学分析方法。

**📊 数据集**

无数据集，依据文献综述与理论推导。

**📈 对比分析**

无比较方法，未涉及性能评估。

**⚠️ 局限性**

1) 系统的价值中立性尚未完全确定；2) 对代理人划分与选择缺乏充分论证；3) 框架对不同教育情境与实践的适用性尚待实证验证。

---

## 177. Weighted Sum-Rate Enhancement for Flexible Intelligent Metasurface-Assisted Multicell Systems

**arXiv ID:** 2606.06845 | [PDF](https://arxiv.org/pdf/2606.06845v1)

**作者:** Hanwen Hu `[一作]` (University of Electronic Science and Technology of China), Arumugam Nallanathan `[通讯]` (Queen Mary University of London)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在多小区多用户MISO系统中，联合优化基站波束、柔性智能金属表面（FIM）的相位和形变，最大化加权总速率（WSR）

**💡 创新点**

引入FIM可实现三维形变，从而提供额外自由度，显著提升多路径信道质量与干扰抑制；提出基于WMMSE+BCD的联合优化框架，结合RCG优化相位和PGD优化形变

**🔧 技术方法**

加权最小均方误差（WMMSE）、块坐标下降（BCD）、黎曼共轭梯度（RCG）用于相位优化、投影梯度下降（PGD）用于形变优化、闭式波束设计

**📊 数据集**

仿真采用自建的二维多小区MISO环境，随机生成用户位置、路径、角度与多径数，并考虑路径损耗模型、噪声功率、功率限制与形变范围

**📈 对比分析**

与传统刚性RIS、随机相位/形变、无FIM等基线进行对比。FIM在各种场景下平均提升约33% WSR；在不同SNR、功率、形变范围、路径数等参数下均优于RIS；对CSI误差鲁棒性良好，MSE≤0.8时性能下降≤30%

**⚠️ 局限性**

算法对时变相位与形变响应时间不匹配导致的动态失配尚未解决；FIM硬件实现与能耗建模仍缺失；在大规模小区/用户时，复杂度随FIM元件数线性增长，需进一步并行化优化

---

## 178. Empirical Study on the Characteristics and Evolution of AI-usage in GitHub Repositories: Evidence from Code Comments

**arXiv ID:** 2606.06843 | [PDF](https://arxiv.org/pdf/2606.06843v1)

**作者:** Abdullah Al Mujahid `[一作]` (Missouri University of Science and Technology), Mia Mohammad Imran `[通讯]` (Missouri University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了GitHub代码注释中自述使用ChatGPT、Copilot等LLM的行为，构建了任务与AI贡献的分类体系，追踪这些AI生成代码的首次更改提交，并对其长期演变进行纵向分析。

**💡 创新点**

创新点在于首次系统化利用海量AI引用注释构建任务与贡献类型的双维度分类，结合人工与多模型LLM标注通过Dawid‑Skene聚合，结合BERTopic主题模型与HDBSCAN语义聚类，全面跟踪后续提交改动并揭示AI使用随时间的转变。

**🔧 技术方法**

采用大规模爬取与过滤、人工与LLM双阶段标注、Dawid‑Skene EM聚合、BERTopic主题建模、HDBSCAN语义聚类、时间序列分析等技术。

**📊 数据集**

使用的主要数据集为35,361条自述AI使用的代码注释及对应代码块，涵盖12,944个公开Python/JavaScript仓库，和12,778条首次更改提交，时间跨度为2022年12月至2026年3月。

**📈 对比分析**

与传统实验/基准对比，本文通过统计首次更改提交中的重构、功能扩展与bug修复等动作，表明单纯的代码生成量指标不足以评估AI的实际价值，实际后续修改仍显高，提示需采用全生命周期指标评估。

**⚠️ 局限性**

限制包括：只覆盖公开GitHub Python/JS仓库，未捕捉隐性或未自述的AI使用；标注依赖于人工与LLM判断，可能存在误解；研究仅追踪首次更改提交，未覆盖长期演化与多轮修改。

---

## 179. CRAFT: A Unified Counterfactual Reasoning Framework for Tabular Question Answering and Fact Verification

**arXiv ID:** 2606.06842 | [PDF](https://arxiv.org/pdf/2606.06842v1)

**作者:** Chenshuo Pan `[一作]` (China Telecom Artificial Intelligence Technology Co., Ltd), Zhongjiang He `[通讯]` (China Telecom Artificial Intelligence Technology Co., Ltd)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CRAFT框架，通过将问题重新写为声明式假设并生成其反事实版本，进行双向推理以提升表格推理性能

**💡 创新点**

创新点在于引入反事实推理路径，将前向推理与反事实推理并行进行，并通过Rethinker模块整合证据，统一了表格问答与事实验证两大任务

**🔧 技术方法**

核心技术包括Rewriter、Reverser、Extractor和Rethinker四个模块，利用LLM进行提示式推理、规则生成反事实、证据筛选与加权决策，兼顾结构化表格信息

**📊 数据集**

在WikiTQ（表格问答）和TabFact（表格事实验证）两个公开基准上进行实验

**📈 对比分析**

与多种基线（End-to-End、Few-Shot、Binder、Dater、Chain-of-Table、Critic-CoT、Table-Critic）相比，CRAFT在所有四个主干LLM上平均提升约4.7%（WikiTQ）和1.1%（TabFact），最高可达11.3%提升；在大模型上进一步压缩不同模型间性能差距

**⚠️ 局限性**

局限性包括未对小规模模型（3B级）进行评估、仅针对表格推理任务、Rethinker模块尚未充分挖掘反事实潜力，且多条反事实路径的最佳选择仍是未来研究方向

---

## 180. Three-dimensional hydro-cluttered locomotion by an undulatory robot

**arXiv ID:** 2606.06829 | [PDF](https://arxiv.org/pdf/2606.06829v1)

**作者:** Tianyu Wang `[一作]` (Georgia Institute of Technology), Daniel I. Goldman `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计并制造了一款名为AquaMILR的细长无肢水下机器人，利用双边电缆驱动、可编程体弹性、分布式深度调节和自带电源实现无缠绕、无牵引线的自由深度运动，并在实验室和自然红树林根部进行系统的鲁棒性测试与现场巡检；

**💡 创新点**

提出了“机械智能”在水下杂乱环境中的三重原理：一是可编程体弹性将接触转化为推进力；二是开环深度调节实现三维障碍规避；三是惯性诱发的全身滚动作为自我恢复模式；同时将双边驱动与弹性耦合，首次在水下实现无传感器的开环复杂环境行走；

**🔧 技术方法**

双边电缆-绞盘驱动、可变电缆长度实现关节可调弹性；分布式气缸/注射器实现深度调节；电机+伺服+传感器集成在防腐水密模块内，完成全封闭的自给电源与计算平台；在实验室使用多种人工障碍（漂浮物、弹性桩、固定桩）以及三维障碍网进行测试；

**📊 数据集**

本研究未使用传统公开数据集，所有实验均为自制的水槽模型和现场红树林根部的现场观测数据；

**📈 对比分析**

通过对比不同弹性参数（G=0,0.5,1）以及不同运动频率和深度调节周期的实验，使用波动效率（η）和运动成本（c_mt）衡量性能；结果显示在最强障碍环境中，弹性可调性提升效率达30%，深度调节可将速度提升约2.5倍，惯性滚动在高频时可将失衡情况恢复至正常速度；

**⚠️ 局限性**

局限性包括：①仅基于开环控制，缺乏对动态障碍的实时感知与适应；②实验多在受限水槽或轻度自然环境中进行，未验证在更深或更恶劣水压下的稳定性；③机器人尺寸较小，未评估在更大尺度或多功能任务（如拖拽、抓取）中的表现；④对能耗与续航的系统级评估尚不足。

---

## 181. ARAPDiffusion: ARAP Regularization for Diffusion-Based Deformable Shape Space Learning

**arXiv ID:** 2606.06887 | [PDF](https://arxiv.org/pdf/2606.06887v1)

**作者:** Haibo Liu `[一作]` (University of Texas at Austin), Qixing Huang `[通讯]` (University of Texas at Austin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种名为 ARAPDiffusion 的潜在扩散模型，用于学习可变形形状集合的连续形状空间。

**💡 创新点**

创新点在于将 ARAP（近似刚性变形）正则化交替注入潜在扩散与自编码器训练中，实现编码器/解码器与潜在扩散模型的双向协同优化。

**🔧 技术方法**

主要技术包括潜在扩散（LD）框架、ARAP 正则化损失、EDM 训练策略以及对网格与隐式两种数据格式的统一解码器设计。

**📊 数据集**

实验使用 SMPL（人体）、SMAL（动物）以及骨骼（Bone）数据集，并兼顾网格与无组织点云两类输入。

**📈 对比分析**

与 SP-Disent、COMA、Neural3DMM、MeshConv、ARAPReg、FrameAve、GeoLatent、BRESA、SALD、GenCorres 及 HY-LoRA 等先进方法对比，ARAPDiffusion 在 e_t（形状质量）和 d_W（分布一致性）指标上显著优于基线，e_r（重构误差）亦有显著提升，且 d_n（过拟合程度）保持在低水平。

**⚠️ 局限性**

局限性包括需要多轮交替训练、对大型形变（如关节附近）的正则化敏感、对评分函数 w(z) 的依赖仍有限，且在重构误差提升幅度相对保守。

---

## 182. Modeling Nonlinear Feature Interactions with Product-Unit Residual Networks

**arXiv ID:** 2606.06861 | [PDF](https://arxiv.org/pdf/2606.06861v1)

**作者:** Ziyuan Li `[一作]` (University of Applied Sciences Koblenz), Babette Dellen `[通讯]` (University of Applied Sciences Koblenz)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种将乘积单元与残差连接相结合的PURe网络，用于显式建模特征交互并提升表格回归性能

**💡 创新点**

创新点在于同时利用乘积单元的乘法归纳偏置和残差结构，实现稳定优化和可解释的交互模式，并扩展到复值版本

**🔧 技术方法**

采用乘积单元（log-linear-exponential）、残差块、ReLU+1预处理、Kaiming初始化以及SHAP交互分析等技术

**📊 数据集**

使用Friedman 1（合成交互驱动）、Concrete Compressive Strength（材料工程）和California Housing（社会经济）三大数据集

**📈 对比分析**

与标准MLP（实值/复值）对比，PURe在MSE、对噪声鲁棒性、低样本效率和交互可解释性上均优于基线，复值PURe表现尤为突出

**⚠️ 局限性**

局限包括复值模型训练波动稍大、对分布移位或缺失特征的鲁棒性待验证，以及需要进一步定量验证交互结构

---

## 183. The Dark Regulome: Disentangling Predictability from Regulation in Genomic Foundation Models

**arXiv ID:** 2606.06834 | [PDF](https://arxiv.org/pdf/2606.06834v1)

**作者:** Chahat Baranwal `[一作]` (Indian Institute of Technology Jodhpur), Lakshya Nitin Tandon `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过残差化和置换诊断对高等胶质瘤相关基因区的30,448暗基因组元素进行 in-silico mutagenesis，区分序列可预测性与真实调控信号，并找出10kb近端调控窗口和可实验验证的候选元素。

**💡 创新点**

提出残差化+置换检验，剔除语言模型对序列可预测性的偏倚；三种不同架构模型的交叉验证，揭示语言模型共享的可预测层与Enformer独立的调控层；提供可迁移的诊断工具。

**🔧 技术方法**

利用DNA序列基础模型（Caduceus-Ph、HyenaDNA、Enformer）进行in-silico mutagenesis，计算Regulatory Influence Score；通过回归四个混杂变量进行残差化；使用置换检验评估重叠显著性；集成梯度做梯度无扰检验；并与phastCons、GTEx cis-eQTL、STRING PPI进行外部验证。

**📊 数据集**

92个与胶质瘤突触相关基因的TSS窗口，包含30,448暗基因组元素（TE、G4、cCREs），来源于UCSC RepeatMasker、G4Hunter、ENCODE SCREEN cCREs；外部验证使用UCSC phastCons100way、GTEx v8脑部cis-eQTL、STRING v12.0 PPI。

**📈 对比分析**

通过交叉架构重叠、置换检验评估top-100/500一致性，发现语言模型共享约76个top-100但与Enformer完全无重叠；残差化后Enformer唯一保留cCRE信号；top-100元素在脑cis-eQTL中显著3.3倍富集，表明候选性高。

**⚠️ 局限性**

样本基因数仅92个，残差化不能完全排除预训练记忆；语言模型缺失三维TAD信息；外部验证仅计算机层面，未进行实验确认。

---

## 184. AdaGRPO: A Capability-Aware Adaptive Enhancement for Flow-based GRPO

**arXiv ID:** 2606.06828 | [PDF](https://arxiv.org/pdf/2606.06828v1)

**作者:** Jiazi Bu `[一作]` (Shanghai Jiao Tong University), Dahua Lin `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出AdaGRPO框架，结合在线课程过滤与跨层优势融合，实现流模型文本到图像生成的更稳定对齐训练。

**💡 创新点**

创新点包括：①基于EMA的在线课程过滤策略动态挑选模型当前能力边界内的中等难度提示；②跨层优势融合将局部优势与全局历史优势结合，消除单组评估的局部偏差。

**🔧 技术方法**

主要技术：流模型SDE转换、GRPO价值无关策略优化、EMA能力估计、动态提示筛选与优势融合。

**📊 数据集**

使用HPD提示集（100k+提示）与400个评估提示，Flux.1-dev流模型作为实验基础。

**📈 对比分析**

与Flow-GRPO、DanceGRPO、Flow-CPS在单/多奖励（HPS‑v2/v3、CLIP）及UniGenBench上对比，AdaGRPO在多数指标提升约2–5%，训练曲线更平稳。

**⚠️ 局限性**

局限性：依赖EMA历史奖励估计，可能对非平稳奖励产生误差；候选提示批大小和动量系数需手动调优，且在更大模型或高分辨率场景下尚未验证。

---

## 185. Architecture Shapes Transfer Specificity in Implicit Neural Representations

**arXiv ID:** 2606.06827 | [PDF](https://arxiv.org/pdf/2606.06827v1)

**作者:** D Yang Eng `[一作]` `[通讯]`, D Yang Eng

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了三种隐式神经表示（SIREN、ReLU MLP、Fourier特征）在受控解析函数、二维Navier‑Stokes瓶颈以及1D热、粘性Burgers、聚焦立方NLS基准上的迁移幅度与迁移特异性，并提出了基于独立随机对照的迁移诊断方法。

**💡 创新点**

创新点在于将迁移幅度与迁移特异性区分开来，揭示不同架构对两者的影响差异，并证明传统静态诊断指标（PR、Hessian、CKA）在此任务中效果有限，强调使用独立随机对照的重要性。

**🔧 技术方法**

采用固定预算微调与从零训练对比、Wilcoxon符号秩检验、Bootstrap 95%置信区间、A_transfer 与 dB 差值计算，以及参与度、Hessian 最大特征值等静态诊断指标进行综合分析。

**📊 数据集**

数据集包括：1D 几何函数 g_t(x)=√(x²+t²)、双参数 1D 正弦系数函数、二维 1x1 区域的壁驱动腔 Navier‑Stokes 流场（Re=100,400,1000）、以及 1D 热、粘性 Burgers 与聚焦立方 NLS 的数值参考解。

**📈 对比分析**

通过终端损失比 A_transfer 与 dB 差值比较迁移效果，结果显示 ReLU 在迁移特异性上表现最突出，SIREN 在大多数任务中提供广泛迁移，而 Fourier 特征需要调节频带以获得更好的性能；整体迁移幅度与特异性随任务与架构显著不同。

**⚠️ 局限性**

实验受限于固定的优化协议、仅 10 颗种子、有限的超参数搜索、PDE 基准采用相同参数对照而非完全随机控制、未探讨更复杂模型或不同学习率/批量大小，以及静态诊断指标表现不佳，需进一步验证。

---

## 186. Breaking the Lock-in: Diversifying Text-to-Image Generation via Representation Modulation

**arXiv ID:** 2606.06813 | [PDF](https://arxiv.org/pdf/2606.06813v1)

**作者:** Dahee Kwon `[一作]` (Korea Advanced Institute of Science and Technology), Jaesik Choi `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无训练、轻量级的内部表示干预方法DAVE，用于提升文本到图像模型在固定提示下的多样性。

**💡 创新点**

创新点在于发现并抑制生成早期Transformer隐藏层的零频DC分量，破除种子锁定，显著提升多样性且几乎不影响质量。

**🔧 技术方法**

采用流匹配式文本图像生成框架（如Stable Diffusion 3、FLUX.1-dev、SANA1.5），对Transformer内部特征进行DC平均值抑制，配合CFG等传统指导技术。

**📊 数据集**

使用ImageNet、MS‑COCO等公开图像数据集作为评估数据，按提示生成多样样本。

**📈 对比分析**

与CADS、SPARKE、PG、SPELL、DiverseFlow等多样性增强方法对比，DAVE在Recall、Coverage、Vendi等多样性指标上提升显著，同时保持或略微提升CLIP、Precision、FID等质量指标，计算开销仅为原模型的极小比例。

**⚠️ 局限性**

局限性：需要手动调节α、τ等超参数；过度抑制可能导致视觉失真；在极度约束或高度复杂提示下多样性提升有限；尚未对不同模型结构的跨平台通用性进行完整验证。

---

## 187. FreeAnimate: Training-Free Human Image Animation with Preview-Guided Denoising

**arXiv ID:** 2606.06885 | [PDF](https://arxiv.org/pdf/2606.06885v1)

**作者:** Yuan Zeng `[一作]` (Tsinghua University), QingMin Liao `[通讯]` (Pengcheng Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种无训练的基于预览帧生成策略的人类图像动画框架FreeAnimate

**💡 创新点**

通过预览帧提供时间与结构先验，结合倒推增强注意力（IBA）和参考锚定自注意力（RA‑SA），实现了无需额外训练即可获得高质量、时间一致且身份保持的视频

**🔧 技术方法**

利用预训练扩散模型（Stable Diffusion）、ControlNet、Grounded‑SAM、MAT去除与填补背景、DDIM倒推、跨注意力和自注意力模块进行动画生成

**📊 数据集**

在TikTok、TED‑Talks以及EverybodyDanceNow等公开数据集上进行评估

**📈 对比分析**

与基准GAN与扩散模型（DisCo、MagicPose、MagicAnimate、Champ等）对比，FreeAnimate在FID/SSIM/PSNR/LPIPS/L1/FVD等指标上均能与训练基线竞争，特别是在无训练方法中取得最佳或次佳表现

**⚠️ 局限性**

推理时间较长（约5.5秒/帧），对显存需求高，细节（如面部表情、手部动作）仍有提升空间

---

## 188. GlucoFM-Bench: Benchmarking Time-Series Foundation Models for Blood Glucose Forecasting

**arXiv ID:** 2606.06881 | [PDF](https://arxiv.org/pdf/2606.06881v1)

**作者:** Baiying Lu `[一作]` (Dartmouth), Temiloluwa Prioleau `[通讯]` (Emory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建并发布了GlucoseFM‑Bench基准，系统评估15个CGM数据集上预训练时序基础模型、LLM时序预测模型和传统监督深度学习模型在零样本、少样本和全样本情景下的血糖预测性能。

**💡 创新点**

首次构建血糖预测领域的时序基础模型基准，揭示预训练模型在低资源条件下的强转移能力以及在充足数据时传统任务特定模型的优势，并对不同糖尿病人群和血糖区间进行细粒度性能剖析。

**🔧 技术方法**

采用Chronos‑2、Moirai‑2.0、Timer、TimesFM‑2.5等预训练TSFM；TimeLLM和CALF两大LLM时序预测框架；以及LSTM与Transformer GPFormer监督模型，结合滑动窗口、统一预处理、RMSE/MAE与Clarke/Surveillance Error Grid等指标进行评估。

**📊 数据集**

利用12个公开CGM数据集（涵盖T1D、T2D、PreD、ND）和3个受控访问数据集（OhioT1DM、DiaTrend、T1DEXI），共计约1117名受试者、1.1亿条血糖记录。

**📈 对比分析**

通过在零样本、少样本（5%）和全样本（完整训练集）三种协议下对每个模型计算RMSE与SEG无风险比例，发现TimesFM在零样本下最优、Chronos‑2在少样本下最佳，而轻量级LSTM在全样本下实现最低RMSE，预训练模型在低资源场景下与监督模型差距≤5%。

**⚠️ 局限性**

仅限单变量CGM预测，未加入碳水摄入或胰岛素等协变量；未覆盖所有最新TSFM架构；基准为回顾性研究，临床部署需进一步前瞻验证与监管审批。

---

## 189. Unified Safe In-context Image Generation in Multimodal Diffusion Transformers via Restricting Unsafe Information Flows

**arXiv ID:** 2606.06875 | [PDF](https://arxiv.org/pdf/2606.06875v1)

**作者:** Xiang Yang `[一作]` (Fudan University), Min Yang `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练无关的安全生成框架，利用MM‑Diffusion Transformer的多模态注意力机制在生成过程中定位不安全视觉补丁并通过注意力调制与噪声注入抑制不安全语义的传播；

**💡 创新点**

创新点在于对MM‑DiT注意力动态的统一分析，发现不安全信息在语义启动阶段快速出现，并提出任务无关的早期补丁定位与目标化注意力抑制策略，实现T2I与I2I任务的共享安全控制；

**🔧 技术方法**

采用多模态注意力映射分析、预构建不安全锚点、补丁级别定位、注意力权重衰减、噪声注入以及空间连通性约束等技术，在生成时动态调节信息流；

**📊 数据集**

实验使用FLUX.1‑dev与FLUX.1‑Kontext‑dev模型，结合I2P、Unsafe‑1k、MS‑COCO、Image‑Editing数据集以及自制的IP字符（Pikachu）和不当物体（武器、血）等概念；

**📈 对比分析**

与ESD、SLD、UCE、DES、EraseAnything等基线在NSFW抹除率、危害率、FID、CLIP、VQA等指标上进行对比，结果表明本方法在T2I和I2I任务中分别实现约91%和77%的抹除率，安全性能领先且对图像质量影响极小；

**⚠️ 局限性**

局限性包括对仅在上下文无关时可定义的不安全概念最有效，对上下文敏感、意图多变的安全判定缺乏能力，且可能放大模型已存在的偏见。

---

## 190. EgoPressDiff: Multimodal Video Diffusion for Egocentric UV-Domain Hand-Pressure Estimation

**arXiv ID:** 2606.06872 | [PDF](https://arxiv.org/pdf/2606.06872v1)

**作者:** Yuan Zeng `[一作]`, QingMin Liao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为EgoPressDiff的条件视频扩散模型，用于从自视角视频估计手部接触压力并生成UV压力量图。

**💡 创新点**

创新点在于将手势、深度、3D网格顶点和RGB信息等多模态先验统一融入扩散生成流程，并通过Distribution‑Calibrated Spatial Layer对不同模态特征进行统计校准，从而实现连续、时序一致且物理可解释的压力预测。

**🔧 技术方法**

使用了视频扩散模型、PoseNet（轻量级手势提取）、Vertex Encoder（对MANO顶点进行高维特征编码）、CLIP图像编码器、以及自定义的分布校准空间层（DC Spatial Layer）和UV掩码损失。

**📊 数据集**

在EgoPressure自视角数据集上进行训练和评估，该数据集包含21位受试者的64个交互片段以及对应的触摸板压力标签。

**📈 对比分析**

与PressureVision、PressureVision+HaMeR、PressureVision++、PressureFormer等基线相比，EgoPressDiff在Contact IoU、Volumetric IoU、MAE和时序准确率上均取得显著提升，尤其是Volumetric IoU提升超过34%，实现了新的SOTA。

**⚠️ 局限性**

模型仅在简单手势和接触模式上训练，泛化到更复杂的日常活动和多接触场景时表现有限，需要进一步扩充数据集和引入更强的先验与自适应机制。

---

## 191. LRMIL: Efficient Low-Resolution Multiple Instance Learning via High-Resolution Knowledge Distillation for Whole Slide Image Classification

**arXiv ID:** 2606.06864 | [PDF](https://arxiv.org/pdf/2606.06864v1)

**作者:** Yonghan Shin `[一作]` (Korea University), Won-Ki Jeong `[通讯]` (Korea University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种低分辨率多实例学习框架（LRMIL），通过跨分辨率知识蒸馏将高分辨率的细粒度信息迁移到低分辨率特征上，实现仅用低分辨率切片进行WSI分类。

**💡 创新点**

创新点包括：① 两阶段蒸馏策略——先在 patch 级别做跨分辨率对齐蒸馏，再在 slide 级别做多任务蒸馏（bag 级别 KL + instance 级别 attention 对齐）；② 通过跨分辨率匹配将 HR 区域的细节信息嵌入 LR 表征；③ 仅在训练阶段使用 HR，推理阶段完全不需要 HR 计算，从而显著降低推理成本。

**🔧 技术方法**

主要技术：ViT 视觉编码器、知识蒸馏（MSE 对齐、KL 软化分布、soft/hard attention 匹配）、多实例注意力聚合、梯度加权损失、Adam 优化器和交叉验证等。

**📊 数据集**

使用 TCGA-BRCA、TCGA-NSCLC、TCGA-RCC、BRACS 等四个公共病理图像数据集进行形态与分子亚型分类，以及多组 TCGA 预后生存预测（BRCA、LUAD、LUSC、KIRP、KIRC）作为评估基准。

**📈 对比分析**

与传统聚合方法（Max‑P、Mean‑P）、最新 MIL 方法（ABMIL、CLAM、DSMIL、TransMIL、DTFD‑MIL）以及专为推理效率设计的方法（ZOOMMIL、HDMIL）对比。LRMIL 在所有任务中均保持或超越最高准确率/ AUC，并将推理时间降低 10 倍以上，显示出显著的性能与效率双赢。

**⚠️ 局限性**

局限性：仅在分类任务上验证，未评估回归或多任务场景；跨分辨率匹配依赖固定像素尺寸和比例，可能需要针对不同放大倍数手动调整；实例级别硬监督对 k 的敏感性需进一步研究。

---

## 192. SCALE: Scalable Cross-Attention Learning with Extrapolation for Agentic Workflow Scheduling

**arXiv ID:** 2606.06820 | [PDF](https://arxiv.org/pdf/2606.06820v1)

**作者:** Zhifei Xu `[一作]` (Beijing Normal University), Jinxi He `[通讯]` (Beijing Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于深度强化学习的调度器SCALE，能够在训练后直接在更大规模的异构集群上进行零样本部署。

**💡 创新点**

创新点在于：1) 采用跨注意力指针网络（task query – server key/value）实现对任意数量服务器的自然支持；2) 引入结构化表示正则化（SRR），通过去相关损失和KL正则保持注意力特征在不同集群规模下的统计稳定，从而显著提升尺度泛化性能。

**🔧 技术方法**

使用的技术包括：跨注意力指针网络、PPO强化学习、结构化表示正则化（去相关 + KL正则），以及对任务特征与服务器特征的两层 MLP 编码。

**📊 数据集**

使用的数据集为：在模拟的异构服务器集群（小/中/大机型比例保持一致）上生成随机DAG工作流，按泊松过程到达，工作流节点包含计算需求、内存需求、输出数据量等属性。

**📈 对比分析**

对比方法包括 CrossAttn（无SRR）、PPO、PPO+SRR、DQN、SAC；实验结果显示：在训练规模N=16上SCALE与CrossAttn相近，但在N=32、48时，SCALE的平均响应时间分别为5.57s和6.22s，较无SRR的CrossAttn提升8.9%，验证了SRR在尺度泛化中的重要性。

**⚠️ 局限性**

局限性包括：仅测试到N=48（3倍训练规模），未验证更大规模；服务器类型比例固定，无法反映不同硬件配置的影响；工作流为随机生成，缺乏真实代理工作流的深度与宽度特征；第一层任务选择仍采用启发式最长路径，未学习联合任务-服务器策略；全部实验基于仿真，缺乏真实物理集群的验证。

---

## 193. AMD-FCG: An Enhanced Function Call Graph Dataset with Integrated Topological Features for Malware Detection and Classification

**arXiv ID:** 2606.06815 | [PDF](https://arxiv.org/pdf/2606.06815v1)

**作者:** Parthajit Borah `[一作]` (National Forensic Sciences University), J. K. Kalita `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了AMD-FCG恶意软件数据集，包含30,000条样本（20,000条恶意、10,000条好软件），每条样本提供函数调用图（FCG）及其拓扑特征，旨在改进恶意软件检测与分类；

**💡 创新点**

创新点在于将FCG与拓扑特征（Betti数、Euler特征、持久性熵等）融合生成全新的结构化特征数据集，并公开发布CC‑BY许可；

**🔧 技术方法**

采用静态反编译（Androguard）生成函数调用图，利用图论与拓扑数据分析提取特征，随后使用图神经网络（GCN、GraphSAGE、GIN、GAT、GraphConv）与传统机器学习（Random Forest、SVM、Gradient Boosting、KNN、Logistic Regression）进行实验；

**📊 数据集**

使用的是公开的Android恶意软件与好软件样本，来自Google Play、AndroZoo以及已有的50个恶意家族；

**📈 对比分析**

在FCG数据上，GIN模型表现最佳，准确率77.1%；在拓扑特征数据上，Gradient Boosting最佳，准确率85.0%；传统机器学习在特征集上普遍优于GNN在原始图上的表现；

**⚠️ 局限性**

主要局限包括类别不平衡导致模型偏倚、缺乏图结构数据的增广方法、对低频恶意家族样本采集困难，以及拓扑特征集仍可进一步丰富以提升细粒度分类能力。

---

## 194. HAVE: Host Active Verification Engine for Closing the Contextual Reality Gap in Security Digital Twins

**arXiv ID:** 2606.06968 | [PDF](https://arxiv.org/pdf/2606.06968v1)

**作者:** Vincenzo Sammartino `[一作]` (University of Pisa), Marco Pasquini `[通讯]` (University of Pisa)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研发了Host Active Verification Engine (HAVE)，在安全数字孪生框架中通过安全约束的主机代理执行实验性漏洞验证，提供经验性破坏概率和时间成本，用以校正攻击图中的概率权重。

**💡 创新点**

创新点在于正式表述并量化Contextual Reality Gap，将理论CVSS与主机级实测相结合；提出使用Wilson置信区间权重的贝叶斯混合更新；以及通过快照隔离的Bernoulli试验实现可重复、可安全的动态验证。

**🔧 技术方法**

利用静态分析提取二进制安全特征、Linux cgroups v2 CPU限幅、路径允许列表、相互认证TLS、快照恢复、最大似然估计、Wilson区间、贝塔-二项后验及Monte Carlo仿真等技术。

**📊 数据集**

使用了四类合成C程序（堆栈溢出、格式字符串、堆UAF、逻辑缺陷）各三级安全等级，两个生产二进制（CVE-2021-3156、CVE-2021-42013）以及对应的攻击图，构成实验数据集。

**📈 对比分析**

通过与CVSS-only、OpenVAS等基线对比，利用Monte Carlo模拟评估攻击图达到目标的概率。HAVE将误报降低38.2%，漏报提升132.4%，整体风险提升124.1%；实验中N≥30、Wilson宽度≤0.364、CPU占用≤15%，快照恢复平均约98 ms，满足OT安全约束。

**⚠️ 局限性**

局限性包括：只能测量已知漏洞且已实现的攻击技术，无法覆盖零日或未实现的攻击向量；需要快照/虚拟化环境，无法直接在裸金属嵌入式设备上运行；攻击库需要持续维护与漏洞匹配。

---

## 195. OpenHalDet: A Unified Benchmark for Hallucination Detection across Diverse Generation Scenarios

**arXiv ID:** 2606.06959 | [PDF](https://arxiv.org/pdf/2606.06959v1)

**作者:** Xinyi Li `[一作]` (University of Technology Sydney), Ling Chen `[通讯]` (University of Technology Sydney)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建统一的hallucination检测基准OpenHalDet，统一prompt、生成、标注、评分流程。

**💡 创新点**

首次构建跨场景、跨模型访问级别（黑盒、灰盒、白盒）的统一评测框架，并集成16种检测方法。

**🔧 技术方法**

采用统一实例schema、GPT‑4o‑mini标注、AUROC评价、成本分析，并对比黑盒、灰盒、白盒检测技术。

**📊 数据集**

覆盖17个多样化数据集（QA、RAG、摘要、推理、代码、多语言、工具调用）和4个主干LLM（Llama、Qwen）。

**📈 对比分析**

在统一协议下对16种检测器进行对比，发现灰盒在多场景表现最稳，白盒最高限但不稳定，黑盒受限；性能依赖模型与任务。

**⚠️ 局限性**

局限在于仅评估response级别检测，未深入token/句子级别；成本与效能不完全匹配，缺乏对更大模型（70B）深入分析。

---

## 196. LIMMT: Less is More for Motion Tracking

**arXiv ID:** 2606.06953 | [PDF](https://arxiv.org/pdf/2606.06953v1)

**作者:** Yu Guan `[一作]` (Tsinghua University), Li Yi `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出三阶段的 General Quality Selection (GQS) 框架，用物理可行性过滤、语义嵌入和复杂度加权采样，将大规模 MoCap 数据集压缩为高质量子集，从而提升人形机器人运动跟踪性能。

**💡 创新点**

创新点在于：① 从数据质量角度重新定义“质量”为可行性、行为多样性与动态复杂度；② 设计层次化筛选流程，先排除不可执行动作再进行语义距离覆盖与复杂度偏好采样；③ 通过实验验证“少即多”，仅使用 3% 数据即可超过完整数据集。

**🔧 技术方法**

使用的技术包括：基于仿真的物理可行性评分；周期性自动编码器 (Harmonic Motion Embedding) 构建语义嵌入；全局加权 Farthest Point Sampling (FPS) 进行子集选择；强化学习（PPO）在 Any2Track、TWIST2 等跟踪器上训练。

**📊 数据集**

主要使用的大规模 MoCap 数据集为 AMASS 与 PHUMA，二者提供丰富且多样的动作捕捉数据。

**📈 对比分析**

对比方法包括完整数据集、随机子采样以及现有跟踪器基线（如 PHC）。在 Any2Track 和 TWIST2 上的实验显示，GQS 仅用 3% 数据即可实现 95%+ 的成功率，MPJPE 下降 15% 以上，明显优于完整数据与随机子采样，且能在跨数据集零样本评估中提升性能。

**⚠️ 局限性**

局限性：依赖仿真环境的物理评分，可能对不同机器人平台迁移不完全；嵌入与评分的权重需要手工调参；对极端噪声或非物理合规动作的处理仍有限；在极其复杂或多模态任务中，模型泛化仍需进一步验证。

---

## 197. SS-TPT: Stability and Suitability-Guided Test-Time Prompt Tuning for Adversarially Robust Vision-Language Models

**arXiv ID:** 2606.06943 | [PDF](https://arxiv.org/pdf/2606.06943v1)

**作者:** Sunoh Kim `[一作]` (Dankook University), Daeho Um `[通讯]` (University of Seoul)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 SS-TPT 方法，通过评估增强视图的稳定性与适用性来进行测试时提示调优，从而提升 VLM 在对抗攻击与分布偏移下的鲁棒性。

**💡 创新点**

引入两种视图质量分数（稳定性与适用性）并结合软加权一致性损失和加权预测，以仅利用可靠视图实现高效鲁棒性。

**🔧 技术方法**

基于 CLIP 的视觉语言模型，使用弱/强增广、Jensen–Shannon 散度、特征空间密度、SS 软加权一致性损失以及 prompt 参数微调等技术。

**📊 数据集**

在 10 个细粒度分类数据集、ImageNet 及其四个 OOD 变体（ImageNet-A/V2/R/S）、UCF101 等多种视觉任务上评估。

**📈 对比分析**

与 CLIP、TPT、R‑TPT、TTC、DOC 等基线对比，SS‑TPT 在清洁准确率与对抗鲁棒性上均达到或超过最优方法，并在减少视图数量时保持更好的吞吐‑鲁棒性折中。

**⚠️ 局限性**

假设稳定性与适用性足以衡量视图可信度；若多视图均被攻击并聚集一致，SS 分数可能误判，需考虑加入空间/局部一致性等额外指标。

---

## 198. VoxCPM2 Technical Report

**arXiv ID:** 2606.06928 | [PDF](https://arxiv.org/pdf/2606.06928v1)

**作者:** Yixuan Zhou `[一作]`, Zhiyuan Liu `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一款基于层次化连续潜在扩散‑自回归架构的 2B 参数多语种可控语音生成基础模型，支持 30 种语言、9 种中文方言以及零样本声克隆、自然语言声设计和风格可控克隆等多种生成模式。

**💡 创新点**

创新点包括：
- 统一的序列组织方式，将所有生成模式映射为不同的输入块组合，消除了任务专属模块；
- 引入 AudioVAE V2 异步编码器/解码器（16 kHz 编码，48 kHz 解码），实现隐式超分辨率并保持低序列速率；
- 对 VoxCPM 体系结构进行大规模扩展，提升 FSQ 维度、引入多 token 预编码、去掉 RALM 的位置编码等改进；
- 采用三阶段渐进式训练策略，兼顾基础 TTS、可控生成与高质量细化，保持零样本克隆性能。

**🔧 技术方法**

技术方案包括：
- 层次化连续潜在模型：TSLM + FSQ + RALM + LocDiT；
- AudioVAE V2 异构 VAE；
- 统一输入块（文本、参考音频、目标音频）和二进制模态指示器；
- 训练时的分类器无关引导（CFG）和两阶段注意力；
- 采用 Sway 采样、CFG‑Zero* 及流式推理；
- 大规模 2 M 小时多语种语料，包含 30 种语言与 9 个中文方言。

**📊 数据集**

使用的数据集主要包括：
- 2 M 小时多语种语料（中文、英文主导，另外 28 种语言分别 1–50 K 小时）；
- 内部 30 语言基准（每种 500 句）；
- 公开基准：Seed‑TTS‑Eval、CV3‑Eval、MiniMax‑MLS‑Test；
- 公开可控语音语料（情感、风格标注）以及内部高质量可控语音子集。

**📈 对比分析**

在零样本克隆、跨语种合成、指令跟随等方面与多种开源/闭源系统进行对比：
- 在 Seed‑TTS‑Eval 上实现 WER 1.84 / SIM 75.3，优于大多数同等规模模型；
- 在 CV3‑Eval、MiniMax‑MLS‑Test 与内部基准上取得 22/24 语种最高 SIM、接近最优 WER；
- 在 InstructTTSEval 上 APS/DSD/RP 评分分别 84.2/83.2/71.4，成为英文子集上表现最好的模型；
- 主观 MOS 评测显示 N‑MOS 4.78、S‑MOS 4.74、I‑MOS 4.50，均高于同类基线；
- 推理时 RTF 0.13（Nano‑vLLM）/0.30（PyTorch），显著快于同等参数规模模型。

**⚠️ 局限性**

局限性：
- 低资源语言的性能仍受数据不平衡影响；
- 对极为抽象或复杂指令的跟随效果略逊于高质量语料丰富的系统；
- 模型规模较大，推理和部署成本仍高；
- 歌声生成能力尚处于初步阶段，质量有待提升。

---

## 199. Belief-Aware Scheduling for Predictive Wildfire Hazard Mapping under Sparse-Window Telemetry

**arXiv ID:** 2606.06917 | [PDF](https://arxiv.org/pdf/2606.06917v1)

**作者:** Xun Shao `[一作]` (Toyohashi University of Technology), Cheah Wai Shiang `[通讯]` (Universiti Malaysia Sarawak)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在稀疏窗口下的野火预警地图预测的信念感知调度问题，提出将结构化信念由预测器和调度器需求推导，并在单无人机场景中对其进行评估。

**💡 创新点**

创新点：①提出“信念感知调度”原则，将结构化信念直接从前向预测器（Φ）和调度器的资源约束推导；②发现窗口周期稀疏性在中间取值时产生单峰的机会耦合，使得非贪婪调度可显著提升；③对信念六维组件进行消融，验证其随景观结构的可切换性；④证明仅需轻量级跨区域注意力网络（≈40k参数）即可实现该原则，深层 Transformer 并不带来显著优势。

**🔧 技术方法**

技术：部分可观测马尔可夫决策过程建模；强化学习（PPO）结合多种网络架构（MLP、GRU、轻量级注意力、深层 Transformer）；物理校准的 Rothermel 火灾扩散模型；LoRa/LEO 时隙稀疏窗口约束。

**📊 数据集**

数据集：物理校准的合成野火环境（16×16格，1km 分辨率），包含两种景观：默认平滑噪声景观和结构化燃料带/山脊景观；未使用真实野火遥感数据。

**📈 对比分析**

比较方法：与公平（仅使用可部署信念）和不公平（可访问真实火场）的基线对比；在不同窗口周期、预算、预测 horizon 下评估；结果显示轻量级注意力网络平均 MSE 0.067，较 FAIR activity‑paced 参考 0.093 提升 144%；Transformer 为 0.073；在 280、1200、5000 训练样本下排名随预算变化，轻量级注意力在充分训练后实现最佳性能。

**⚠️ 局限性**

局限性：仅针对单无人机、周期性窗口模型；未在真实野火数据上验证；未考虑前向预测器不确定性；仅使用 5000 训练回合，未探究更大规模或多无人机协调；对燃料、风、地形等因素的联合耦合未在真实场景中评估。

---

## 200. Communication Strategy Selection for Multi-GPU 3D FDTD with Convolutional Perfectly Matched Boundary Layers

**arXiv ID:** 2606.06910 | [PDF](https://arxiv.org/pdf/2606.06910v1)

**作者:** Victory C. Obieke `[一作]` `[通讯]` (Oregon State University), Victory C. Obieke (Oregon State University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

实现并基准测试了在多 NVIDIA GPU 上运行的三维有限差分时域（FDTD）求解器，求解器配合卷积完美匹配层（CPML），研究了不同的通信策略（主机-CPU 传输、直接 GPU‑to‑GPU 互连）和 ghost 区域扩大技术，以及多种域分解布局；

**💡 创新点**

首次量化证明在包含 CPML 的高阶 3D FDTD 计算中，直接 GPU‑to‑GPU 交换显著优于主机‑staged；并评估了 ghost 区域扩大对性能的有限提升，为多 GPU FDTD+CPML 提供了最优通信与分解策略的实证依据；

**🔧 技术方法**

利用 CUDA 原始 kernel 与 CuPy 包装/拆包机制实现 GPU‑to‑GPU 直接拷贝；采用多 GPU 域分解（slab、block、pencil）和 CPML 边界层实现；在单精度下对内存访问与计算做了细粒度优化；

**📊 数据集**

使用合成声波传播模型，在不同尺寸的立方格子（320³、480³、544³、640³、800³、1024³）以及 CPML 厚度 20 单元进行实验；不涉及真实测量数据，仅基于数值模拟；

**📈 对比分析**

通过多种 GPU 组合（1、2、4）和通信策略，记录总运行时间、吞吐量、强缩放效率与 CPML 开销；实验显示 GPU‑to‑GPU 交换可获得 2.46–2.76× 加速，ghost 区域在 s=4 时提升约 4–6%；单 GPU 基线 CPML 开销低于 1%；在 RTX8000 上实现 1.51× 加速，同时显著降低单 GPU 内存占用；

**⚠️ 局限性**

仅在单节点四 GPU 上验证，未实现多节点并行或流重叠；只研究单精度 FDTD+CPML，未扩展到完整 Maxwell 方程；ghost 区域扩大带来的计算与内存额外负担有限；缺乏自动化布局选择与更高阶 CPML 的评估；结果对不同硬件平台的通用性有限。

---

## 201. TALAN: Task-Aligned Latent Adaptation Networks for Targeted Post-Training of Large Language Models

**arXiv ID:** 2606.06902 | [PDF](https://arxiv.org/pdf/2606.06902v1)

**作者:** Chengkai Zhang `[一作]` (Meta AI), Qin Huang `[通讯]` (Meta AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种称为TALAN（Task-Aligned Latent Adaptation Networks）的模块，插入到Transformer的残差流中，利用序列条件的潜在记忆对激活进行小幅、可控的干预，并与低秩适配器（LoRA/DoRA）共同训练。

**💡 创新点**

创新点在于：1) 将激活干预与低秩适配器结合到单一SFT循环中，避免了任务全局的缺陷；2) 设计了一个六轴可配置空间，使得同一模块可以在不同主机上通过选择插入层、槽数、混合器、写回方式等实现最佳效果；3) 证明干预激活在幅度上显著小于适配器更新，并在正交空间中工作，且在后续层能被放大，支持“激活级可控适配”的理念。

**🔧 技术方法**

使用了低秩适配器（LoRA/DoRA）、多头注意力、槽查询聚合、跨注意力、门控写回等技术；对比了多种激活干预方法（如ReFT、CogSteer、DELIFT）和多轮基线；对干预进行了正交性和激活放大等机制分析。

**📊 数据集**

在四个Qwen3系列大模型（Qwen3-32B、DeepSeek-32B、Qwen3-8B、Qwen3-30B-A3B）上，使用包含数学、代码、STEM等四大类任务的 5,000 条训练样本混合；评估基准为GPQA Diamond、GSM8K、MATH-500、MBPP；还在 Llama-3.2-1B 上做了跨模型迁移检验。

**📈 对比分析**

与仅使用LoRA或DoRA的基线进行对比；在LoRA下，TALAN 在所有 16 个模型-基准组合上均无负向变化，平均提升 +1.41pp；在DoRA下，平均提升 +1.85pp，13/16 单元为正向。多核种子对比显示整体正向趋势，且训练参数增量 <1%，推理开销 1.01–1.02×。

**⚠️ 局限性**

局限性包括：仅在 STEM/代码基准和 Qwen 系列主机上验证，未覆盖对话、检索、跨语言等任务；迁移到非 Qwen 系列的实验仅限于 Llama-3.2-1B；仅评估单一 LoRA/DoRA 规格，其他参数设置未知；机制分析为描述性，缺乏基于模型内部几何的自动配置规则。

---

## 202. LUCID: Learning Unified Control for Image Deflaring and Exposure Mastery in Nighttime Photography

**arXiv ID:** 2606.06901 | [PDF](https://arxiv.org/pdf/2606.06901v1)

**作者:** Tingyu Yang `[一作]` (Shanghai Jiao Tong University), Xiaoyun Yuan `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出 LUCID 框架，统一处理夜景中的光晕（flare）和低光（低照度）两大退化，采用光晕分离模块与扩散驱动恢复模块，实现单图连续曝光与光源可控调节；

**💡 创新点**

创新点包括：①将光晕抑制与曝光恢复拆分为两阶段并协同完成；②通过四模式训练与分类器无关引导（CFG）实现曝光与光源的连续可控；③利用光晕分离生成结构引导，结合扩散模型恢复细节；④支持单图 HDR 重建；

**🔧 技术方法**

技术方法涵盖：U-Net 光晕分离网络、跨状态扩散（mix‑state diffusion）与 VAE 编码器、跨状态注意力、Classifier‑Free Guidance、四模式训练策略、HDR 虚拟曝光合成；

**📊 数据集**

使用的数据集包括 ExDark（真实夜景）、Flare7K（光晕合成）、SiHDR（HDR 评估）、RELLISUR、LSRW、SICE、SID 等；

**📈 对比分析**

与 Zero‑DCE、Retinexformer、RetiDiff、DarkIR 等低光增强方法以及 Flare7K、MFDNet、Zhou 等光晕抑制方法，以及 IntrinsicHDR、LEDiff、GasLight 等 HDR 方法进行对比，采用 CLIPIQA、MANIQA、MUSIQ、LIQE、NIMA 等无参考质量指标评估；LUCID 在所有指标上显著优于 SOTA，尤其在曝光控制、光晕去除和 HDR 重建方面表现突出；

**⚠️ 局限性**

局限性包括：在极暗场景下模型更倾向保守恢复，可能出现细节缺失；生成模型仍可能产生幻觉，特别是结构缺失区域；训练仍依赖合成光晕，缺少真实光晕标注；极强光源下仍可能残留光晕或导致颜色失真。

---

## 203. Lighting-Aware Representation Learning under Controllable Lighting Variation

**arXiv ID:** 2606.06899 | [PDF](https://arxiv.org/pdf/2606.06899v1)

**作者:** Lizhen Zhu `[一作]` (Pennsylvania State University), Brad Wyble `[通讯]` (Pennsylvania State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在可控光照变化下的视觉表示学习，提出双头对比学习框架，使模型同时学习语义和光照特征；

**💡 创新点**

创新点在于将光照视为显式监督而非噪声，采用双头结构将表示拆分为光照、内容和交互三部分，并通过光照对比损失显式对齐同一光照样本；

**🔧 技术方法**

使用基于MoCo V2/ESS‑MB的对比学习框架，加入光照对比损失，采用ResNet‑50骨干和共享编码器的双头投影头；

**📊 数据集**

使用了House100KLighting、ImageNet100K、House100K‑AugLighting、ImageNet100K‑AugLighting等预训练数据集，以及ExDark和PASCAL VOC等下游评估集；

**📈 对比分析**

通过与标准对比学习和单一表征基线在ImageNet分类、ExDark物体/光照分类和PASCAL检测上的对比，双头模型在联合表示上提升约10‑20%的准确率，尤其在低光环境下光照表征效果显著；

**⚠️ 局限性**

局限性包括：对光照的模拟主要依赖物理渲染，简单图像变换无法完全逼近真实光照；实验规模有限，未验证在更大数据集上的泛化；光照与语义的分离过度可能导致互交信息损失；对层级分配和超参数的深入分析仍待补充。

---

## 204. FDM: A Framework for Decision-making to build ML-based Malware detection systems

**arXiv ID:** 2606.06894 | [PDF](https://arxiv.org/pdf/2606.06894v1)

**作者:** Tadiwa Vhito `[一作]` (Prince Of Songkla University), Norrathep Rattanavipanon `[通讯]` (Prince Of Songkla University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并验证了一套基于多准则的决策框架FDM，用于在不同运营约束下选择最合适的机器学习配置以构建恶意软件检测系统。

**💡 创新点**

引入Weighted Configuration Compatibility Score (WCCS) 将五个可量化操作参数映射到九个配置维度，实现了基于实验验证的可解释、可定制化的推荐机制。

**🔧 技术方法**

结合经典树模型、深度学习（LSTM、BiLSTM、CNN）、迁移学习、增量学习、自动编码器预处理和多准则决策分析。

**📊 数据集**

使用Windows API序列私有集（8类恶意+1类正常）、公开Malimg图像数据集（9,339样本/25类）和Android权限特征集（15,036个应用）。

**📈 对比分析**

通过四组实验对比模型准确率、AUC、内存占用、训练时间等指标，验证了XGBoost在二分类中最优，CNN在多分类、增量学习和迁移学习中表现突出；自动编码器可实现14倍训练加速，准确率仅损失0.86个百分点。

**⚠️ 局限性**

未考虑联邦学习、对抗鲁棒性和可解释性需求；框架假设数据中心化且不针对高级APT的特定规避技术；在非图像特征上迁移学习效果有限。

---

## 205. Workflow-to-Skill: Skill Creation via Routing-Workflow-Semantics-Attachments Decomposition

**arXiv ID:** 2606.06893 | [PDF](https://arxiv.org/pdf/2606.06893v1)

**作者:** Yuyang Zhang `[一作]` (Wuhan University), Run Wang `[通讯]` (Wuhan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究如何自动从历史交互证据构造可执行的LLM Agent Skill

**💡 创新点**

提出WSA三元组（Workflow、Operational Semantics、Runtime Attachments）作为中间表示，并构建基于证据的Trace‑to‑Skill框架

**🔧 技术方法**

采用结构化推理、节点级语义重构、证据对齐与反馈迭代的生成算法

**📊 数据集**

使用70个Skill、8种WSA类型的数据集，并收集多条执行轨迹

**📈 对比分析**

与Anthropic Skill Creator对比，使用回放行为一致性评估，平均提升约4.8%，在大多数类型上领先

**⚠️ 局限性**

对附件依赖的Workflow（T5）表现不佳，且依赖完整且多样化的交互证据，缺少对罕见路径的充分覆盖

---

## 206. Diagnosing Visual Ignorance in Vision-Language Models

**arXiv ID:** 2606.06890 | [PDF](https://arxiv.org/pdf/2606.06890v1)

**作者:** Runyu Zhou `[一作]` (Peking University), Yisen Wang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过内部层级干预（层替换）和层级监督 MLP 探测，结合外部视觉衰减实验（逐步高斯模糊），系统评估 VLM 对语言先验的依赖；

**💡 创新点**

提出多阶段瓶颈模型：中间层难以检索细粒度视觉信息，后期层主动抑制视觉信号；引入渐进视觉衰减指标，量化不同程度模糊下答案不变性；将层级干预与监督探测相结合，首次将内部机制与基准评估联系起来；

**🔧 技术方法**

采用层级替代（counterfactual layer replacement）、监督层级 MLP 探测、逐步高斯模糊、连续答案一致性指标 C、单一答案一致性指标 E、基准误差对比等技术；

**📊 数据集**

在 12 大视觉问答基准上进行评估（RePOPE、HR‑Bench、HallusionBench、AI2D、MMMU、V*Bench、VLMBias、RealworldQA、BLINK、MMBench、SEED‑Bench、MMStar），并利用 Pixmo‑Count 数据集训练层级探测器，使用 VLMBias 数据集进行微调实验；

**📈 对比分析**

与 LLaVA、Qwen‑3B/15B 等三类 VLM 进行对比，发现 20%–40% 的样本在图像完全模糊后答案保持不变；在基准上模型准确率随模糊增加略降，但在“连续一致子集”中的准确率几乎与原始基准持平，表明基准未能有效惩罚语言先验依赖；

**⚠️ 局限性**

局限性包括：仅关注语言解码器内部机制，未深入视觉编码器或跨模态桥接层；实验主要基于现有公开 VLM，缺乏更大规模或不同架构验证；渐进模糊指标对极端噪声或多模态答案空间的适用性有限；方法未提供直接的训练改进方案，仍需设计结构化或对抗性训练来真正消除语言先验。

---

## 207. Uniform Stability and Generalization Error of GD and SGD on Fixed-Point Parameters

**arXiv ID:** 2606.06934 | [PDF](https://arxiv.org/pdf/2606.06934v1)

**作者:** Jonghyun Shin `[一作]` (Korea University), Sejun Park `[通讯]` (Korea University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文分析了梯度下降（GD）与随机梯度下降（SGD）在固定点离散参数空间下的泛化误差与稳定性，揭示了舍入误差导致的泛化退化与维度相关的稳定性行为。

**💡 创新点**

创新点在于：①证明确定性舍入会将GD的泛化误差从O(T/n)提升到O(T/√n)并产生Ω(T)的稳定性下界；②展示SGD在一维和高维情况下可获得O(T/n)与O(T²/n)的紧致稳定性上界；③揭示随机舍入会引入维度相关的泛化误差，并给出对应的稳定性上界与下界。

**🔧 技术方法**

采用了算法稳定性框架、确定性与随机舍入模型、凸光滑 Lipschitz 损失的理论分析以及构造性下界示例。

**📊 数据集**

无具体数据集，研究基于理论构造的分布与损失函数。

**📈 对比分析**

与传统无舍入的GD/SGD理论结果对比，证明舍入显著影响泛化与稳定性；在一维下SGD仍保持良好性能，而高维下随机舍入导致泛化误差随维度增长。

**⚠️ 局限性**

局限在于：仅考虑理想化的离散更新，未覆盖真实自动微分中舍入误差；不涵盖量化感知训练与浮点数的非均匀格点；对非凸损失的适用性仅为潜在可推广，需进一步验证。

---

## 208. ThinkBooster: A Unified Framework for Seamless Test-Time Scaling of LLM Reasoning

**arXiv ID:** 2606.06915 | [PDF](https://arxiv.org/pdf/2606.06915v1)

**作者:** Vladislav Smirnov `[一作]` (MBZUAI), Artem Shelmanov `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个统一框架，用于在推理阶段动态调节LLM推理计算量，以提升推理质量；

**💡 创新点**

将多种TTC缩放策略与评分器集成到可扩展Python库、统一的性能‑计算基准和OpenAI兼容代理服务中，同时提供可视化调试器，实现科研与工业部署的无缝衔接；

**🔧 技术方法**

基于Python实现多种TTC策略（BoN、MUR、Beam Search等）和评分器（PRM、无监督/监督不确定性、LLM自评、ReProbes），利用vLLM、HuggingFace Transformer、OpenAI API等后端，并支持预填充、白盒/黑盒接口；

**📊 数据集**

使用数学数据集（MATH‑500、OlympiadBench、GaoKao23EN、AIME‑2024/25）、科学问答集GPQA‑Diamond、编码集HumanEval+、MBPP+和KernelBench；

**📈 对比分析**

通过联合评估精度（EM、pass@1、语法、编译、正确性）与计算成本（TFLOPs、生成 token 数），对比不同策略/评分组合，发现无监督不确定性在编码任务上表现优于PRM，而PRM在数学任务上效果最佳；Beam Search 在配合 PRM 时可达最高精度但计算量大，动态 MUR 在编码任务上更高效；

**⚠️ 局限性**

受限于对白盒信号（logits、隐藏状态）或预填充的依赖，封闭 API 只能使用部分策略；步骤边界提取在无结构思考模型中困难；基准仅覆盖数学/编码/科学 QA，未涵盖长文本问答、开放式生成或工具辅助代理；仅报告理论 TFLOPs，未评估实际延迟。

---

## 209. DPAgent-in-the-Middle: Agentic Defense and Repair Against AI-Groomed Deceptive Patterns

**arXiv ID:** 2606.06914 | [PDF](https://arxiv.org/pdf/2606.06914v1)

**作者:** Zewei Shi `[一作]` (University of Melbourne), Xingliang Yuan `[通讯]` (University of Melbourne)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文首次阐明了 AI Grooming 这一利用数据空洞操纵模型训练的威胁，并提出了名为 DPAgent 的 Agent‑in‑the‑Middle 框架，该框架通过四个子代理（AI Grooming 过滤、任务生成、PDP 检测与界面修复）实现了在实时 Web 环境中主动检测与修复隐私欺骗模式（PDP）。

**💡 创新点**

创新点：1) 定义并系统化 AI Grooming 与 PDP 的关联；2) 设计多代理框架将 Grooming 过滤、强化学习提示优化、专家+LLM 定义生成、实时 UI 修复协同工作；3) 采用 RL‑PPO 对 Prompt 进行动态优化，以提升任务生成质量；4) 在用户浏览时通过 MITMProxy 实时干预，既保障用户体验又防止恶意 UI。

**🔧 技术方法**

技术栈：MITMProxy（代理拦截）、EfficientNet‑V2‑L（Grooming 过滤器）、CLIP 对比验证、Gemini 2.5 Pro / Claude 3.7 Sonnet 等大语言模型、PPO‑RL 用于 Prompt 优化、计算机使用代理（Claude Computer Use）执行 UI 自动化与修复、保护动机理论（PMT）指导的修复策略。

**📊 数据集**

数据集：自制 AI Grooming 样本（233 原始 + 233 生成），PDP 识别数据集（515 条，覆盖 7 类 PDP），任务生成 RL 基准（40 主机名、1380 页），真实世界评测集（485 个网站，5206 页），以及 8 位参与者的用户体验问卷。

**📈 对比分析**

评测方法：与 DPGuard、AidUI、ALSACNC 等主流模型对比；采用 F1（micro/macro）、覆盖率、修复率、Runtime 等指标。性能表现：PDP 微 F1 0.8162（+27.84%）、宏 F1 0.6914（+38.64%）；Grooming 过滤成功率 90.98%；任务生成奖励 0.714；网站探索覆盖 80.62% 仅访问 9.5% 页面；修复率 77.3%；用户研究平均评分 4.10/4.16；单页平均处理时间 1–2 秒。

**⚠️ 局限性**

局限性：1) 对布局极为复杂或密集的页面检测/修复效果不佳；2) 依赖 LLM 的推理，可能出现漏检或误判；3) DP 词表需人工维护，新增 PDP 仍需人工评估；4) 对动态生成、实时注入的恶意 UI 仍有防御盲区；5) 现阶段仅实现原型，浏览器级部署仍需进一步优化。

---

## 210. polyDAG: Polynomial Acyclicity Constraints for Efficient Continuous Causal Discovery in Visual Semantic Graphs

**arXiv ID:** 2606.06908 | [PDF](https://arxiv.org/pdf/2606.06908v1)

**作者:** Wenhao Zhang `[一作]` (Shanghai Jiao Tong University), Minyi Guo `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种多项式无环性约束 polyDAG，用以替代 NOTEARS 中的矩阵指数约束，从而实现对视觉语义图的连续因果结构学习。

**💡 创新点**

创新点在于证明多项式无环性约束与指数约束等价，并通过几何级数闭式实现，显著降低了计算复杂度。

**🔧 技术方法**

使用梯度下降与增广拉格朗日框架、Hadamard平方、矩阵几何级数求逆、自动微分等技术。

**📊 数据集**

数据集包括合成的 Erdős–Rényi 图以及 CelebA 人脸属性图。

**📈 对比分析**

与 NOTEARS 指数约束相比，polyDAG-Geo 在 SHD/F1 上均有提升，运行时间缩短约 14–33%，在不同图规模和密度下均表现优异。

**⚠️ 局限性**

局限在于仍保持 O(d³) 的密集线性代数复杂度，适用于中小规模图；对非线性模型、稀疏大图以及真实因果可识别性的支持尚未验证。

---

## 211. Beyond Skeletons: Learning Animation Directly from Driving Videos with Same2X Training Strategy

**arXiv ID:** 2606.06903 | [PDF](https://arxiv.org/pdf/2606.06903v1)

**作者:** Yuan Zeng `[一作]` (Tsinghua University), Qingmin Liao `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出DirectAnimator框架，直接使用原始驱动视频生成静态参考图像的动画视频；

**💡 创新点**

创新点包括：① 直接利用原始像素作为驱动信号，跳过姿态估计；② 设计三元组驱动提示（姿态、面部、位置信号）并通过CueFusion DiT模块融合；③ Same2X训练策略通过同一身份模型对齐跨身份学习，显著加速收敛并提升质量；

**🔧 技术方法**

使用技术包括：基于CogVideoX1.5的DiT扩散模型、3D VAE编码、AdaLN调制、频域低通滤波、G‑SAM前处理、伪驱动数据生成与对齐损失；

**📊 数据集**

数据集主要为从互联网上收集的4000个视频片段（5–20秒）及TikTok训练集；测试集为TikTok测试集（335–340）和另外50个未见视频；

**📈 对比分析**

与包括AnimateAnyone、MagicAnimate、Champ、MimicMotion、StableAnimator、UniAnimate‑DiT、DynamiCtrl等SOTA方法比较，DirectAnimator在FID、SSIM、FIS等指标上均居前，尤其在跨身份场景表现出更高的身份保留和运动准确性，且在相同GPU条件下推理速度可比；

**⚠️ 局限性**

局限性包括：依赖伪驱动数据的质量；缺乏显式姿态监督，可能在极端姿态或复杂场景下仍出现误差；未来工作需进一步提升伪数据生成质量并探索无监督姿态约束。

---

## 212. Stream3D-VLM: Online 3D Spatial Understanding with Incremental Geometry Priors

**arXiv ID:** 2606.06891 | [PDF](https://arxiv.org/pdf/2606.06891v1)

**作者:** Hanxun Yu `[一作]` (Zhejiang University), Dong Yu `[通讯]` (Tencent Hunyuan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了首个能够在线处理流式视频的3D视觉‑语言模型Stream3D‑VLM，能够实时决定何时响应并完成空间理解任务。

**💡 创新点**

创新点包括：①将流式控制转化为LLM的next‑token预测，实现自动判断何时发声；②设计轻量级Visual‑Spatial Feature Integration（VSFI）模块，将增量几何先验与视觉特征融合；③提出Geometry‑Adaptive Voxel Compression（GAVC）模块，以3D结构为导向动态压缩视觉token，显著降低长上下文冗余。

**🔧 技术方法**

使用的技术主要有：LLM自回归训练与流式控制学习、跨注意力融合（VSFI）、空间聚类与双注意力聚合（GAVC）、基于StreamVGGT的即时3D重建、强化学习式的决策标记训练。

**📊 数据集**

训练数据包括：1M+时空QA对（5.2k 3D扫描），来源于ScanNet、ScanNet++、ARKitScenes；评测数据为Stream3D‑Bench（518视频，29个任务），以及VSI‑Bench、ScanQA、ScanRefer、Scan2Cap等公开基准。

**📈 对比分析**

与专有模型（GPT‑4o、GPT‑5）和公开VLM（LLaVA‑Video、InternVL、Qwen等）以及专门的空间推理模型（SpaceR、Spatial‑MLLM、VG‑LLM）进行比较。Stream3D‑VLM‑8B在Stream3D‑Bench上获得86.7%答复时间准确率、0.39 s推理时延，显著优于其他在线VLM；在VSI‑Bench上实现65.9%平均准确率，超越Gemini‑Pro、Open‑AI及多数开源模型；在ScanQA/ScanRefer/Scan2Cap等离线3D任务中亦保持领先。

**⚠️ 局限性**

局限性包括：①依赖RGB视频与可用的3D重建模型，若重建质量差或场景快速运动可能导致几何先验失真；②对高分辨率视频的实时性仍受限于GPU算力；③流式控制学习需要较大规模标注数据，迁移到新域或语言时可能需要重新调优；④模型规模相对较大，部署成本较高。

---

## 213. MVSegNet: A Lightweight Boundary-Aware Network for Fetal Lateral Ventricle Segmentation and Atrial Width Estimation in Prenatal Ultrasound

**arXiv ID:** 2606.06958 | [PDF](https://arxiv.org/pdf/2606.06958v1)

**作者:** Arafat Hossain Sayem `[一作]` `[通讯]` (Stamford University Bangladesh), Arafat Hossain Sayem (Stamford University Bangladesh)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文提出了一种轻量级的 MVSegNet 网络，用于胎儿侧脑室在产前超声图像中的分割及腔宽估计。

**💡 创新点**

创新点在于结合多尺度特征提取、边界感知细化、基于胎龄的 FiLM 条件化、注意力门和 Adaptive Boundary Refinement Block 等模块，显著提升边界质量和测量精度，同时保持极低的计算成本。

**🔧 技术方法**

使用的技术包括 MobileNetV3‑Small 编码器、MSAM 多尺度注意力模块、FiLM 条件化、Attention Gate 跳连、ABRB 边界细化块、二元交叉熵+Tversky 损失和多尺度监督等。

**📊 数据集**

实验数据来自公开的 584 张胎儿转室超声帧，按 70%/15%/15% 进行训练/验证/测试划分。

**📈 对比分析**

与 U‑Net、UNet++、Attention U‑Net、DeepLabV3+、MobileNet‑UNet、FetSAM 等六个基线模型对比，MVSegNet 在 Dice、IoU、HD95、ASD、宽度 MAE 等指标上均取得最优成绩，并以 165.6 FPS 的速度运行，参数量仅 2.31M，显著优于其他模型。

**⚠️ 局限性**

局限性包括仅使用单一数据集、图像级划分导致无法评估个体泛化、缺乏多中心验证、宽度估计方法与临床标准不完全一致、以及仅在 NVIDIA T4 GPU 上评估推理速度，未在嵌入式设备上测试。

---

## 214. Didact: A Cross-Domain Capability Discovery System for Defence

**arXiv ID:** 2606.06942 | [PDF](https://arxiv.org/pdf/2606.06942v1)

**作者:** Aarya Bodhankar `[一作]` (University of New South Wales), Flora Salim `[通讯]` (University of New South Wales)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一款名为Didact的跨域能力发现系统，整合公开的国防报告与学术研究知识图谱，通过自然语言对话实现对多源信息的检索与合成；

**💡 创新点**

创新点在于：①多源复合检索，区分文档检索与图谱检索；②将检索结果以交互式Evidence Rail呈现，支持用户追溯证据；③通过访问级别隔离实现源级别控制；

**🔧 技术方法**

核心技术包括：检索增强生成（RAG）与大型语言模型（LLM）相结合的LangGraph Orchestrator；文档检索使用Chroma向量数据库；图谱检索基于Neo4j图数据库；前端交互采用React、TypeScript、Cytoscape.js；后端使用FastAPI；

**📊 数据集**

使用的数据集包括：澳大利亚国防公开报告与政策文件（分为public、classified两级），以及基于OpenAlex、ROR等开放元数据构建的澳大利亚研究知识图谱；

**📈 对比分析**

通过DoRA-QnA50基准评估，Hit@1达68%，Hit@2达76%；Context Recall 90.4%，Answer Relevancy 88.2%；平均响应时延约2.17秒，95百分位约2.91秒，表现符合多种用户角色与任务类型的实用需求；

**⚠️ 局限性**

局限性包括：依赖LLM生成结果，仍可能出现事实错误；图谱与文档检索的分离导致系统复杂度提升；目前仅覆盖澳大利亚数据，跨国或多行业迁移需要进一步适配；

---

## 215. Personality Anchoring for Social Simulation: Linking Personality, Social Behavior, and Interaction Success with LLM Agents

**arXiv ID:** 2606.06936 | [PDF](https://arxiv.org/pdf/2606.06936v1)

**作者:** Vahid Sadiri Javadi `[一作]` (University of Bonn), Johanne R. Trippas `[通讯]` (RMIT University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在 LLM 模拟对话中，双人组合的宜人性（Agreeableness）如何影响社交目标完成度，并通过行为策略进行中介分析。

**💡 创新点**

创新点在于：①采用角色锚定（personality anchoring）以真实角色背景实现人格表达；②系统化评估 7 类社交目标与两难度水平下的对话；③探讨人格组合对对话策略与结果的部分中介作用。

**🔧 技术方法**

使用的技术包括：LLM 对话生成（DeepSeek‑Chat‑v3、Gemini 3 Flash、GPT‑5.2 等）、CHARISMA 框架、行为策略编码与聚合、LLM‑as‑a‑judge 评估、行为中介分析、重复运行 ICC 测评等。

**📊 数据集**

使用的数据集：20 个按宜人性分层筛选的角色（来源于 Personality Database），277 条基于 135 人类目标分类的场景，1,010 条模拟对话。

**📈 对比分析**

比较方法：跨模型（DeepSeek、Gemini、OpenAI GPT‑5.2、Mistral）验证一致性，结果显示共享目标完成度从低宜人性 2.3 分到高宜人性 7.3 分（10 分制），ICC3,1 为 0.89，表达一致性良好。

**⚠️ 局限性**

limitations: 仅关注宜人性，未涉及其他 Big Five 维度；角色来源偏西方，文化多样性不足；LLM‑as‑a‑judge 评估可能偏向语言流畅；对话模型有限，未验证至更广泛的 LLM。

---

## 216. MyGardenBird: A Machine-Learning-Ready Bird Sound Dataset for Twelve Common Malaysian Birds

**arXiv ID:** 2606.06975 | [PDF](https://arxiv.org/pdf/2606.06975v1)

**作者:** Muhammad Mun'im Ahmad Zabidi `[一作]` (Universiti Malaya), Norisma Idris `[通讯]` (Universiti Malaya)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了MyGardenBird数据集，包含 12 个马来西亚及印尼-马来亚地区常见鸟类的 7,200 条 3 秒音频剪辑，经过手工光谱分割和质量控制，并提供平衡的训练/验证/测试划分、SNR 及地理元数据。

**💡 创新点**

首次为热带地区鸟类声音提供可重复、标准化的三秒剪辑数据集，并提出了基于来源录音的分割策略以消除数据泄漏，同时通过 BirdNET 无训练验证来确保标签一致性。

**🔧 技术方法**

使用 Python 脚本实现全流程（光谱 GUI、手工标注、质量评估、SNR 计算、混合整数规划划分、CNN 训练），采用 Log‑Mel 频谱、Mixup 以及 AdamW 优化器训练 MobileNetV3‑Small、EfficientNet‑B0 和 ResNet‑50 等模型。

**📊 数据集**

数据来源为 Xeno‑canto 的公共录音，最终得到 16 kHz 版 7,200 条剪辑和 44.1 kHz 版 6,950 条剪辑，覆盖 12 种鸟类，每种 600 条，来自 1,381 条原始录音。

**📈 对比分析**

基准实验显示 CNN 在 80:10:10 源级划分上分别达到 92–96% 的分类准确率，BirdNET 进行无训练标签验证时准确率达 97.94–98.06%，证明数据集标签可靠、类别可分辨性强。

**⚠️ 局限性**

局限包括仅有单一专家手工标注，未检验多标注者一致性；数据集仅覆盖 12 种，缺少更丰富的物种；部分物种仅在非地区录音中补充，仍受 Xeno‑canto 采集偏差影响。

---

## 217. Modeling U.S. Attitudes Toward China via an Event-Steered Multi-Agent Simulator

**arXiv ID:** 2606.06971 | [PDF](https://arxiv.org/pdf/2606.06971v1)

**作者:** Chenxu Zhu `[一作]` (University of Science and Technology of China), Yongdong Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了事件驱动的多智能体模拟器ES‑MAS，用来捕捉美国公众对中国态度的动态演变。

**💡 创新点**

创新点包括：① 双流数据集成引擎（重大事件+日常新闻）实现实时宏观微观同步；② 新闻驱动的动态交互模块，使智能体按兴趣形成局部群组并互相影响，避免信息茧房；③ 构建覆盖 2021‑2025 年的 China‑U.S. Relation Evolution (CURE) 数据集。

**🔧 技术方法**

技术手段主要是：大语言模型（如 GPT‑4o‑mini）用于智能体推理、新闻选择和行为生成；文本嵌入与相似度匹配；动态聚类与组内交互；多智能体系统框架与时间步进模拟；评估使用 DTW、Fréchet 等指标。

**📊 数据集**

使用的数据集为：CURE（258 个重大事件 + 14,650 篇相关日常新闻）以及 TUIIR 数据库的美国对华态度基准。

**📈 对比分析**

通过与传统 ABM（BC/HK/RA/SJ/LR）以及 LLM‑驱动社交网络模拟器（FPS/SOD/HiSim）的宏观对齐评估（ΔBias、ΔDiversity、DTW、Fréchet），ES‑MAS 在所有指标上均显著优于最优基线（ΔBias 0.5767、DTW 2.4403、Fréchet 1.6150）。

**⚠️ 局限性**

局限性包括：依赖大量计算资源，规模扩展至 200+ 代理时表现下降；对初始态度仍有一定敏感度；使用公开新闻和事件数据，缺少实时深度信息，可能引入偏见；模型未充分处理多源信息质量与信噪比问题。

---

## 218. SSRLive: Live Streaming Recommendation with Dynamic Semantic ID

**arXiv ID:** 2606.06970 | [PDF](https://arxiv.org/pdf/2606.06970v1)

**作者:** Teng Shi `[一作]` (Taobao & Tmall Group of Alibaba), Yuning Jiang `[通讯]` (Taobao & Tmall Group of Alibaba)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了动态语义ID（静态+动态）指导的直播推荐框架，融合生成与判别模块以精准捕捉实时直播内容与用户交互；

**💡 创新点**

创新点在于同时利用静态与动态语义ID刻画主播的长期与即时特征，并通过任务查询与跨特征融合显式建模用户–主播互动；

**🔧 技术方法**

使用Encoder–Decoder生成式网络、Transformer、RQ‑KMeans量化、EMA更新代码表、任务查询、多任务学习以及Beam Fusion等技术；

**📊 数据集**

基于阿里巴巴直播平台约10亿条实时交互记录及主播历史多模态片段构建的数据集；

**📈 对比分析**

在离线实验中相较DLRM、SASRec、ReaRec、HSTU等基线提升约1–2% AUC/GAUC；在线A/B测试实现观看时长+3.38%、GMV+0.72%、关注+3.12%、交互+2.92%，且推理延迟仅+1.33%；

**⚠️ 局限性**

模型参数与 FLOPs 较大，需大量训练数据，对实时数据预处理依赖，难以直接迁移至资源受限环境。

---

## 219. From Vision to Text: A Compact Multimodal Approach for Robust, Cross-Domain Presentation Attack Detection on ID Cards

**arXiv ID:** 2606.06966 | [PDF](https://arxiv.org/pdf/2606.06966v1)

**作者:** Qingwen Zeng `[一作]` (Technical University of Denmark), Christoph Busch `[通讯]` (Hochschule Darmstadt)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种紧凑的多模态模型（基于SmolVLM2的生成与判别结构）用于身份证演示攻击检测，并在真实与合成数据集上进行跨国评估。

**💡 创新点**

创新点包括：①融合视觉与文本模态以提升跨域鲁棒性；②设计生成式与判别式两种训练结构并使用LoRA进行参数高效微调；③系统性重新评估合成数据在PAD基准中的适用性。

**🔧 技术方法**

主要技术：SmolVLM2多模态Transformer、SigLIP视觉编码器、DenseNet-121基线、LoRA微调、生成式标签采样与判别式分类头；使用APCER、BPCER、EER等评估指标。

**📊 数据集**

使用真实身份证数据（智利、墨西哥）以及三国（波兰、葡萄牙、西班牙）合成护照数据，共包含Bona Fide、Border、Screen、Printed四类攻击样本。

**📈 对比分析**

与DenseNet、SigLIP等基线对比，零射击性能差强人意；生成式结构在真实数据上EER低于0.9%，并在跨国测试中显著优于对手；判别式与基线在真实数据上性能落后；合成数据无论何种模型均表现不稳，出现高达100% BPCER 的极端失效。

**⚠️ 局限性**

主要限制：零射击下多模态模型几乎随机；合成数据与真实数据特征差异大，导致模型泛化失效；跨域性能仍受限于数据规模与模型容量；需要更真实、多样的训练集以提升稳健性。

---

## 220. SVHighlights: Towards Extremely Long Sport Video Highlight Detection

**arXiv ID:** 2606.06926 | [PDF](https://arxiv.org/pdf/2606.06926v1)

**作者:** Donggyu Lee `[一作]` (Ulsan National Institute of Science and Technology), Taehwan Kim `[通讯]` (Ulsan National Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SVHighlights 基准数据集和 TF-SELECTOR 无训练模型，用于极长（超过一小时）体育视频的高光检测。

**💡 创新点**

创新点在于通过官方高光视频自动生成标签、基于语义一致的分段聚合，以及利用 LLM 在段级别进行多模态得分，从而实现零训练、可扩展的长视频高光检测。

**🔧 技术方法**

技术包括 PSNR 帧对齐、语音转录（WhisperX）、视频分段（shot detection + transcript 合并）、VLM（InternVL2.5‑8B）生成字幕、LLM（Llama‑3‑8B）进行段级 saliency 预测，并用音量和 transcript 作为多模态输入。

**📊 数据集**

使用 320 条官方完整比赛视频（共 640.18 小时）及其对应的官方高光视频构成的 SVHighlights 数据集。

**📈 对比分析**

与 VTG‑tuned、segment‑based 以及 LLM‑based 现有基线对比，TF‑SELECTOR 在 HIT@1、HIT@K、IoU 上分别提升 3.12、4.06、2.95，整体表现优于第二名。

**⚠️ 局限性**

局限在于仅覆盖体育视频、需配对完整与高光视频、预处理耗时且无任务专属训练，难以直接迁移到非体育长视频。

---

## 221. DRIFT: From Robustness Gaps to Invariance Manifolds for AI-Generated Image Detection

**arXiv ID:** 2606.06918 | [PDF](https://arxiv.org/pdf/2606.06918v1)

**作者:** Abhishek Ameta `[一作]` (Samsung Research Institute), Amit Satish Unde `[通讯]` (Samsung Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

以一类学习方式仅使用真实图像，训练双头投影网络（鲁棒子空间与脆弱子空间），并通过结构化的漂移顺序损失实现 AI 生成图像检测。

**💡 创新点**

创新点在于将鲁棒与脆弱子空间的分解、EMA 软教师、重建锚点以及对漂移大小的顺序约束相结合，构建真实图像的物理不变性流形，从而实现对未知生成器的开放世界泛化。

**🔧 技术方法**

技术实现包括冻结 DINOv2 ViT 提取特征，双头 MLP 投影，EMA 软教师平滑目标，重建解码器避免表示崩溃，梯度裁剪和 top‑k 中位数聚合的检测分数。

**📊 数据集**

训练集仅包含 MIT‑5K、LSUN、RAISE 等真实图像；测试集包括 ForenSynth（GAN）、Diffusion‑6Cls（扩散）和 PromptWorld‑1K（Gemini/ChatGPT）。

**📈 对比分析**

与 RIGID、SAFE、FerretNet、FatFormer 等训练‑free 或监督检测器对比，DRIFT 在 ForenSynth 上实现 98.6% ACC / 99.8% AP，在 Diffusion‑6Cls 上最高 AP 100%，在 PromptWorld‑1K 对 Gemini/ChatGPT 分别达到 93.2%/92.0% 与 94.8%/95.0%，整体性能显著优于现有方法。

**⚠️ 局限性**

局限性在于检测依赖预设的鲁棒/脆弱变换；若生成器采用不同结构或高度逼真的编辑技术，漂移信号可能减弱；此外仅基于特征空间的检测对对抗攻击敏感，且高分辨率实时部署仍面临计算成本。

---

## 222. EASE-TTT: Evidence-Aligned Selective Test-Time Training for Long-Context Question Answering

**arXiv ID:** 2606.06906 | [PDF](https://arxiv.org/pdf/2606.06906v1)

**作者:** Xiaopeng Yuan `[一作]` (University of Illinois Urbana Champaign), Yushun Dong `[通讯]` (Florida State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了基于证据的测试时适应方法 EASE‑TTT，旨在提升长文本问答任务的性能。

**💡 创新点**

提出了一种新颖的证据引导框架，通过在测试阶段利用上下文中的关键信息对模型进行自适应。

**🔧 技术方法**

利用了证据识别与测试时参数微调技术，并在现有大语言模型上实现了快速适配。

**📊 数据集**

在公开的长文本问答数据集上进行了实验，主要聚焦于含有大量上下文信息的问答场景。

**📈 对比分析**

与传统基线模型和其他适应方法相比，EASE‑TTT 在长上下文 QA 任务中取得了显著的准确率提升。

**⚠️ 局限性**

实验仅覆盖长文本问答任务，未评估对数学推理、符号推理或开放式生成的适用性；同时仅在相对较小的模型上测试，未探究更大规模模型的表现。

---

## 223. ActionMap: Robot Policy Learning via Voxel Action Heatmap

**arXiv ID:** 2606.06904 | [PDF](https://arxiv.org/pdf/2606.06904v1)

**作者:** Pei Yang `[一作]` (National University of Singapore), Mike Zheng Shou `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一种体素热图动作头，替代VLA模型中的单点动作解码器，实现对动作空间的概率分布预测并通过soft-argmax解码。

**💡 创新点**

利用体素热图对动作空间进行分布式监督，并使用高斯模糊目标捕捉邻近动作的空间结构，从而显著提升数据效率、收敛速度和成功率。

**🔧 技术方法**

体素热图动作头、softmax交叉熵训练、Gaussian blob目标、top-k soft-argmax解码、LoRA微调、跨backbone兼容性。

**📊 数据集**

LIBERO四套任务（Spatial、Object、Goal、Long）和Franka Research 3 实验室真实机器人演示集。

**📈 对比分析**

与OpenVLA-OFT的L1回归头和π_0.5的流匹配头在相同训练步数下对比，结果在LIBERO四套平均上提升+8.2%/ +1.6%，在Franka实验中成功率显著提高，且在10%训练集仍保持93%成功率。

**⚠️ 局限性**

体素网格尺寸导致参数量与分辨率呈多项式增长，限制了可用的细粒度；缺乏自适应网格与高斯宽度调节；未充分探索多模态或时间采样的利用。

---

## 224. GRASP: Geometry-aware Residual Alignment for Scalable Pretraining Data Attribution

**arXiv ID:** 2606.06892 | [PDF](https://arxiv.org/pdf/2606.06892v1)

**作者:** Yue Min `[一作]` (Wizard Quant), Yujun Li `[通讯]` (Wizard Quant)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

重新定义预训练数据归因，从点级别评分转为子集层面的反事实效用预测，并提出了交互感知的几何惩罚模型和低维特征草图以实现可扩展评估。

**💡 创新点**

核心创新在于：① 引入一阶光滑性下界得到线性相关项和二次几何惩罚，显式建模数据冗余与互补；② 用低维特征草图替代高维梯度，既保持可扩展性又保留子集交互信息；③ 通过开发环境预先选择权重，避免隐藏的每任务调优。

**🔧 技术方法**

技术手段包括：Kronecker‑factored GGN近似、GRASP相关性（隐藏‑残差匹配）、一阶对齐评分、二次几何惩罚、特征草图拼接、标准化与权重融合、线性数据模型评分（LDS）评估。

**📊 数据集**

实验使用 10B-token Common Crawl 训练的 50M 参数因果解码器，评估目标涵盖 BasicSkills、OpenBookQA、CommonsenseQA、SciQ 等 NLP 任务；另外在跨域视觉任务中使用 ImageNet‑预训练 ResNet18 与 CIFAR‑10，验证方法在不同领域的迁移性。

**📈 对比分析**

与 TracIn、TRAK、DataInf、LESS、DataShapley、InRun‑DS 等可扩展归因基线对比，GRASP 在 LDS 任务上平均 Spearman ρ 从 0.174 提升至 0.361（提升 > 2 倍），正向相关率从 0.645 提升至 0.917；构造成本仅 0.27 天，100k 子集评估 5 秒，约比 TRAK 快 31 倍；在下游固定计算量模型训练和跨域视觉选样中也表现出更低的损失和更高的准确率。

**⚠️ 局限性**

局限性：① 理论基于一阶光滑下界的近似，未覆盖完整的非凸 Transformer 动态；② 主要在 50M 模型规模验证，尚未在极大模型上进一步验证子集交互的普适性；③ 草图压缩牺牲部分长距离注意信息；④ 方法针对特定下游目标，极端高/低保留比例下需要额外校正。

---

## 225. T-GMP: Terrain-conditioned Generative Motion Priors for Versatile and Natural Humanoid Locomotion

**arXiv ID:** 2606.06944 | [PDF](https://arxiv.org/pdf/2606.06944v1)

**作者:** Junhong Guo `[一作]` (Harbin Institute of Technology), Fenghua He `[通讯]` (Harbin Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了地形条件生成运动先验（T-GMP），使人形机器人能在多种地形上实现自然且协调的全身运动。

**💡 创新点**

创新点在于：①使用条件β‑VAE学习地形感知的运动潜在空间；②将该潜在空间与地形条件判别器结合，形成统一的对抗强化学习框架；③引入脚步惩罚（Foothold Penalty）提升踏地稳定性。

**🔧 技术方法**

技术包括：CNN提取局部高度图特征、条件β‑VAE、对抗式运动先验（AMP）、强化学习策略训练、脚步惩罚模块。

**📊 数据集**

数据集来源于特权专家策略和人类运动捕捉同步收集的专家数据，包含不同地形的状态序列与对应的局部高度图。

**📈 对比分析**

与传统RL基线、无条件判别器、无CVAE、无脚步惩罚等做比较，在八种地形上获得最高通过率（提升幅度高达17%–20%），并显著降低关节扭矩与加速度波动，表现更平滑、更鲁棒。

**⚠️ 局限性**

局限性：需要配对的运动状态与高度图，受限于人类捕捉与仿真同步的困难；依赖LiDAR重建，噪声与更新频率会影响鲁棒性；缺乏多模态感知来进一步提升感知精度。

---

## 226. Exploring Agentic Tool-Calling Decisions via Uncertainty-Aligned Reinforcement Learning

**arXiv ID:** 2606.06976 | [PDF](https://arxiv.org/pdf/2606.06976v1)

**作者:** Yijin Zhou `[一作]` (Shanghai Jiao Tong University), Jing Shao `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 TRUST，一种在 LLM 代理工具调用决策中将不确定性量化嵌入奖励函数的后训练框架，并引入轻量化关键回合标注实现决策与任务性能统一优化。

**💡 创新点**

创新点在于将不确定性作为惩罚性“排斥力”加入奖励，使模型在决策时保持置信度与正确性分离，同时通过关键回合标注避免全流程重标注，实现单一后训练即可提升决策准确率与整体任务表现。

**🔧 技术方法**

技术包括基于困惑度的置信度估计、UQ 对齐奖励设计、GRPO 后训练策略、CM2 检查表奖励、轻量化 LLM 判断器与标注器，构成统一的多回合决策与任务奖励混合框架。

**📊 数据集**

使用了 When2Call、BFCL‑V4 与 ToolSandbox 三个公开工具调用基准，分别评估单回合决策、长回合任务完成与工具利用。

**📈 对比分析**

与 SAGE、AUQ、GRPO、CM2 等基线对比，TRUST 在 When2Call 的 Acc Norm 提升 11% 以上、FDAR 降低至 5%，在 BFCL‑V4 与 ToolSandbox 上分别取得整体得分领先 6–7% 以上，且在误报与工具误用方面显著降低。

**⚠️ 局限性**

局限在于仅采用困惑度作为不确定性度量，缺乏更深层的语义或轨迹级不确定性建模；实验仅覆盖文本工具调用，未验证对动态或物理环境的适用性。

---

## 227. GenPO++: Generative Policy Optimization with Jacobian-free Likelihood Ratios

**arXiv ID:** 2606.06967 | [PDF](https://arxiv.org/pdf/2606.06967v1)

**作者:** Ke Hu `[一作]` (ShanghaiTech University), Ye Shi `[通讯]` (ShanghaiTech University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出GenPO++方法，利用高阶可逆ODE求解器将历史状态作为辅助内存，实现流式可逆策略优化，并支持精确无雅可比的似然比计算。

**💡 创新点**

创新点在于用solver-history替代dummy-action，得到可逆高阶流式策略；log‑determinant仅依赖固定系数，避免Jacobian计算，同时保持原动作维度，提升训练稳定性与效率。

**🔧 技术方法**

采用高阶Adams–Bashforth近似、可逆Companion‑form ODE、流式生成策略、PPO on‑policy、KL控制与Jacobian‑free log‑determinant技术。

**📊 数据集**

实验数据集包括IsaacLab机器人仿真基准（Ant、Humanoid等）、Robomimic操纵任务（Can、Box、Threading）以及真实世界RobotEra Xhand拧螺钉任务。

**📈 对比分析**

与Gaussian PPO、DPPO、FPO、GenPO、PolicyFlow等基线比较，GenPO++在仿真中获得更高奖励、更快收敛，学习效率约比GenPO提升4倍；在fine‑tuning任务中提升成功率；在真实任务中取得更高奖励并成功部署。

**⚠️ 局限性**

局限在于对σ参数敏感，σ过大会偏离原流动过程，过小会影响数值稳定性；需要自适应调节或改进可逆求解器以降低对σ的依赖。

---

## 228. DREAM: Dynamic Refinement of Early Assignment Mappings

**arXiv ID:** 2606.06947 | [PDF](https://arxiv.org/pdf/2606.06947v1)

**作者:** Liwei Guan `[一作]` (Kuaishou Technology), Zhaojie Liu `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出DREAM框架，针对SID生成推荐中的冷启动问题，采用三阶段动态细化SID分配。

**💡 创新点**

创新点在于将SID分配拆解为先前支持修复、保守承诺和多路径恢复，动态延迟并多路保留冷启动项路径。

**🔧 技术方法**

使用CART（基于对比学习和协作增强的重token化）、UC3（冻结模型多上下文投票与置信门控）以及CPDE（动态beam与LoRA梯度隔离的多路径解码）等技术。

**📊 数据集**

实验数据集为Amazon Beauty、Sports & Outdoors、Toys & Games三大商品评论数据集。

**📈 对比分析**

与10种ID/生成推荐基线对比，在18项冷启动指标上全表领先，提升约4-12倍，整体与最强基线保持竞争。

**⚠️ 局限性**

局限性主要在于仅针对离线冷启动设置，未考虑持续更新与动态项目加入，且对warm项的微调可能产生轻微性能波动。

---

## 229. Quantum-Inspired Trace-Augmented Evidence Selection for Reasoning over Structured Hypothesis Spaces

**arXiv ID:** 2606.06941 | [PDF](https://arxiv.org/pdf/2606.06941v1)

**作者:** Laura Wynter `[一作]` (Singapore Management University), Paul Griffin `[通讯]` (Singapore Management University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EP-HUBO，一种通过高阶二进制优化 (HUBO) 选择多条链式推理 (CoT) 片段来聚合法律推理证据的方法；

**💡 创新点**

创新点在于将证据片段的权重基于相关性、专属性和区别性而非投票频率，利用可组合的高阶二进制优化显式选择最具说服力的证据子集，从而在专业化、证据密集型任务中突破多数投票瓶颈；

**🔧 技术方法**

技术包括本地小型 LLM 生成 CoT 轨迹、自动分解证据片段、使用局部 LLM 计算一、二、三阶权重，随后通过模拟退火或 Dirac‑3 光子量子机求解 HUBO，最后将精炼后的证据交给大规模前沿 LLM 进行一次性裁决；

**📊 数据集**

使用的评测数据集为 MMLU‑Pro（法律子集）和 LEXam（瑞士/国际法 8 选多项选择），两者均为证据密集型、低污染法律问答基准；

**📈 对比分析**

与多数投票（MV）和零样本前沿 LLM（ZS）基线对比，EP‑HUBO 在 MMLU‑Pro 上提升 12.6pp/27.9pp，LEXam 上提升 23.2pp，且在对抗前沿模型的偏置（如 Sonnet 在 LEXam 上的 “E” 偏好）时，HUBO precision 可达 92% 以上，表现出显著的准确率和偏差缓解；

**⚠️ 局限性**

局限包括：需要多次本地 LLM 生成轨迹导致计算成本高；HUBO 仅在单模型轨迹下进行，未充分利用多模型多样性；量子求解受 135 变量上限限制，需截断大规模证据集合；对极端偏置前沿 LLM（如 Sonnet）仍无法完全消除；此外，方法对裁决模型的强度高度依赖，弱模型难以充分利用优化得到的证据。

---

## 230. From Custom Logic to APIs: Understanding and Recommending API Replacement Refactorings

**arXiv ID:** 2606.06912 | [PDF](https://arxiv.org/pdf/2606.06912v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 231. When CLIP Sees More, It Fights Back Harder: Multi-View Guided Adaptive Counterattacks for Test-Time Adversarial Robustness

**arXiv ID:** 2606.06938 | [PDF](https://arxiv.org/pdf/2606.06938v1)

**作者:** Sunoh Kim `[一作]` (Dankook University), Daeho Um `[通讯]` (University of Seoul)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无需调参的测试时对抗方法——多视角自适应对抗 (MAC)，通过多视角引导的对抗攻击和腐败程度软权重提升CLIP在对抗攻击下的鲁棒性。

**💡 创新点**

创新点在于（1）多视角引导的对抗攻击，避免仅依赖被攻击原图；（2）基于随机增强的腐败程度度量与软权重，实现在每个视角自适应调节攻击强度。

**🔧 技术方法**

采用CLIP预训练视觉编码器、随机几何与光度增强、投影梯度上升（PGD）对抗迭代，以及Sigmoid软权重融合等技术。

**📊 数据集**

在20个细粒度数据集（如 Caltech101、DTD、Flower102、Pets、UCF101、Aircraft、EuroSAT、Cars、SUN397、Food101）以及 ImageNet 及其 A/V2/R/S 等 OOD 变体上进行评估。

**📈 对比分析**

与现有对抗防御（TTC、R‑TPT、MTA 等）比较，MAC 在保持清洁准确率的同时，PGD‑100 等强攻击下平均鲁棒率提升约 30% 以上，显著优于其他无调参方法。

**⚠️ 局限性**

受限于增强分布和视角数量的选择，且在医学或遥感等特殊域上的泛化能力尚待验证。

---

## 232. From Sampled Outcomes to Capability Distributions: Rethinking Supervision for LLM Routing

**arXiv ID:** 2606.06924 | [PDF](https://arxiv.org/pdf/2606.06924v1)

**作者:** Guannan Lai `[一作]` (Nanjing University), Han-Jia Ye `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究LLM路由中单次标注噪声问题，提出分布感知路由监督框架DARS

**💡 创新点**

创新点在于将查询重写与多次解码的分布信息用于生成期望性能、成本与风险三项指标，构造更稳健的路由标签

**🔧 技术方法**

采用Prompt重写、重复采样、统计期望与方差并引入风险调节的效用函数

**📊 数据集**

在GPQA、MATH-500、DROP-800三大基准及六种LLM模型池上验证

**📈 对比分析**

与多种路由器（回归、分类、聚类、检索等）对比，DARS平均提升约4–7%，显著降低单次标注导致的方差

**⚠️ 局限性**

局限在于需额外的查询重写与多采样开销，对低算力部署的效率影响仍需进一步评估

---

## 233. Towards Event-Robust Acoustic Scene Classification

**arXiv ID:** 2606.06921 | [PDF](https://arxiv.org/pdf/2606.06921v1)

**作者:** Yiqiang Cai `[一作]` (Xi'an Jiaotong-Liverpool University), Xi Shao `[通讯]` (Nanjing University of Posts and Telecommunications)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Event-Shifted Acoustic Scene（ESAS）数据集，并在此数据集上评估了现有声场分类（ASC）模型的鲁棒性；

**💡 创新点**

创新点在于使用大语言模型（GPT‑4）对背景场景与前景事件进行语义匹配，生成带有未知事件的合成音频，从而构造出针对事件移位（event‑shift）的专门基准；

**🔧 技术方法**

主要技术包括：LLM驱动的场景‑事件语义分组、BEATs模型进行前景事件筛选、时间拉伸/音调移位等数据增强，以及随机SNR混合；

**📊 数据集**

使用数据集为：CochlScene（13个场景、211小时）作为背景，FSD50K（约51k条、200类事件）作为前景事件；

**📈 对比分析**

对比了6种主流ASC模型（TF‑SepNet、BC‑ResNet、GRU‑CNN、CP‑Mobile、BEATs、PaSST），结果显示在已知事件条件下准确率下降约10–15%，在未知事件条件下下降更大（轻量级CNN下降22个百分点，Transformer下降7–9个百分点），表明现有模型对事件移位高度脆弱；

**⚠️ 局限性**

局限性包括：合成混音可能无法完全模拟真实环境的复杂交互；仅涵盖13类场景且事件库有限；LLM语义匹配可能引入偏差；缺乏针对轻量级模型的专门鲁棒性改进方法。

---

## 234. Blockchain Infrastructure for Intelligent Cyber--Physical--Social Systems:Post-Quantum Security, Interoperability, and Trustworthy Data Economies in the Era of Embodied AI

**arXiv ID:** 2606.06895 | [PDF](https://arxiv.org/pdf/2606.06895v1)

**作者:** Song Guo `[一作]` (Hong Kong University of Science and Technology), Luyao Zhang `[通讯]` (Duke Kunshan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

组织并执行了一个两小时的量子-区块链-具身AI技术教程，包含AWS Braket实测演示、五个技术模块以及最终的综合讨论，系统阐述了如何构建兼具量子安全、跨链互操作性和可信数据经济的CPSS基础设施。

**💡 创新点**

将量子硬件威胁评估、具身AI数据需求、跨链协议与可信数据治理三大技术板块整合为统一路线图；首次以实证的AWS Braket量子设备演示ECDSA向后量子签名的迁移过程；提出BrokerChain跨分片协议与Croissant元数据标准的协同应用。

**🔧 技术方法**

采用AWS Braket（超导、阱离子、原子云）量子硬件实验、Post‑Quantum加密算法（如Dilithium、Falcon）、BrokerChain跨分片协议、Croissant元数据标准、IRASim世界模型与WMPO决策算法、以太坊Beacon Chain与Uniswap链数据、AWS云基础设施等技术。

**📊 数据集**

使用具身AI交互日志和机器人数据集（如LET、Kuavo），区块链交易与协议数据（Ethereum Beacon Chain、Uniswap），以及AWS Braket量子实验产生的门控误差与相干时间数据。

**📈 对比分析**

通过在AWS Braket上记录门控误差与相干时间，量化对ECDSA的破解时间，并对后量子签名的签名/验证延迟进行基准；在BrokerChain上测评跨分片吞吐量与延迟；在Croissant数据集上验证元数据完整性与可追溯性，性能均达到或超过公开基准，延迟保持在可接受范围内。

**⚠️ 局限性**

量子硬件实验受限于噪声与规模，真实量子攻击的时间窗口尚未确定；跨链协议的安全性依赖于多方共识实现细节；数据经济激励机制需进一步实证验证；整体框架多学科融合，缺乏统一标准化的落地工具包。

---

## 235. Data-Constrained Language Model Pretraining: Improved Regularization and Scaling Laws

**arXiv ID:** 2606.06888 | [PDF](https://arxiv.org/pdf/2606.06888v1)

**作者:** Zhiwei Xu `[一作]` (University of Michigan), Yixin Wang `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在数据受限、计算富余的预训练场景，提出Masked-Input Regularization（MIR）和SoftQ耦合尺度律；

**💡 创新点**

创新点在于将随机掩码作为自回归预训练的辅助损失，兼顾无架构改动与推理开销，同时提出SoftQ耦合尺度律纠正Chinchilla等加性律在该场景下的失配；

**🔧 技术方法**

主要技术包括自回归Transformer、随机掩码辅助损失（MIR）、强weight decay、五参数SoftQ耦合尺度律以及对不同模型尺寸与数据预算的网格实验；

**📊 数据集**

使用DCLM（100M唯一标记）和Stack‑V2（代码集）两大数据集；

**📈 对比分析**

通过验证集交叉熵、下游零样本任务（BoolQ、SciQ等）以及RMSE、AIC等指标对比，MIR在1.4B模型上验证损失下降约0.03，BoolQ提升10.2分；SoftQ在拟合度与外部数据集上优于Chinchilla、Quanta和Muennighoff；

**⚠️ 局限性**

局限性包括实验规模仅至1.4B参数、400M唯一数据，固定模型架构与优化器，且需在每个网格点进行繁重的超参搜索；

---

## 236. Accounting for Context: Shaping Moral Credences for Value Alignment

**arXiv ID:** 2606.06972 | [PDF](https://arxiv.org/pdf/2606.06972v1)

**作者:** Jazon Szabo `[一作]` (King's College London), Sanjay Modgil `[通讯]` (King's College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

本文探讨了价值对齐中的道德不确定性，并引入情境特征对道德信念（credences）的系统化调整机制（prod 与 mini），进一步将这些调整与最大期望选择性（MEC）相结合，证明了该组合会违反弱帕累托原则，并用Simpson悖论解释了这一现象。

**💡 创新点**

创新点在于：①将情境特征抽象为可乘/取最小的调整函数，实现对每个行动的信念动态更新；②通过形式化证明展示了这种情境调整在满足或不满足弱帕累托原则时的行为差异；③将Simpson悖论引入道德不确定性讨论，揭示了跨理论信念分布可能导致的逆向优先级。

**🔧 技术方法**

主要技术手段包括：形式化定义（伦理理论、信念函数、情境特征、调整函数）；概率与加权算术平均（WAM）；逻辑推理与证明（证明 prod、mini 为情境可达性；证明 MEC 的互理论响应性与弱帕累托违反）。

**📊 数据集**

论文未使用具体实验数据集，而是以纯理论推导为主。

**📈 对比分析**

方法比较基于数学证明：对 prod 与 mini 两种调整函数在不同情境下对 MEC 结果的影响进行推导；并通过示例（FROBO、右/左房间）展示了调整后信念分布与决策结果的差异；未给出数值性能指标。

**⚠️ 局限性**

局限性包括：①缺乏经验验证，未检验模型在真实 AI 系统中的适用性；②对人类偏好动态性假设过于理想化；③仅关注伦理理论与情境特征的交互，未考虑更复杂的多模态输入、对话策略与学习机制；④理论结果对实际 AI 部署的可解释性与安全性影响尚不明确。

---

## 237. Tree-of-Experience: A Structured Experience-Management Solution for Self-Evolving Agents under Low-Repetition and Implicit-Reward Environments

**arXiv ID:** 2606.06960 | [PDF](https://arxiv.org/pdf/2606.06960v1)

**作者:** Zihao Deng `[一作]`, Jikun Shen `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了FinEvolveBench金融情感预测基准并提出Tree-of-Experience经验管理框架。

**💡 创新点**

创新在于针对低重复、隐式奖励环境设计时间控制的金融情感预测基准，并提出结构化经验树以实现可检索、验证、更新的经验管理。

**🔧 技术方法**

使用了大型语言模型（如DeepSeek‑V4‑Flash、Qwen3.6‑35B‑A3B）、树形经验检索、公式化的经验价值更新以及LLM‑判断器。

**📊 数据集**

采用中国A股31个行业指数的历史新闻与市场数据，形成时间顺序的预测实例。

**📈 对比分析**

对比了无经验Baseline、Pipe、Pipe+mem0、Pipe+MemRL和Pipe+ToE等方案，结果显示结构化经验管理（Pipe+ToE）在20天预测期的tsIC/ csIC 明显优于其它方法。

**⚠️ 局限性**

局限在于短期预测噪声大、经验检索质量受前置新闻结构影响、以及未考虑交易成本与实盘效果。

---

## 238. i2Slicer: Enabling Flexible and Automated Orchestration of 5G SA End-to-End Network Slices

**arXiv ID:** 2606.06955 | [PDF](https://arxiv.org/pdf/2606.06955v1)

**作者:** M. Catalan-Cid `[一作]` (i2CAT Foundation), S. Siddiqui `[通讯]` (i2CAT Foundation)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并实现了i2Slicer，一个用于 5G SA E2E 网络切片的自动化编排框架，支持多租户、多服务以及 MOCN、RAN 共享等功能；

**💡 创新点**

创新点在于将 5G 切片的控制面与用户面进行分离，采用分割式部署（disaggregated）来动态配置各子网；通过单一管理系统实现多租户及多服务的切片生命周期管理；

**🔧 技术方法**

采用 Open5G-S、Amarisoft gNB、OpenStack、Open Source MANO（OSM）等开源技术，结合自研的 Slice Manager 与 RAN Controller 通过 NETCONF/O1 进行远程配置；

**📊 数据集**

实验使用 Open5G-S 5G 核心、Amarisoft Callbox SDR 作为 gNB，测试环境配置了两种部署模式（单体与分割式）并监控 vCPU、RAM 与执行时间；

**📈 对比分析**

通过 100 次迭代比较两种部署模式，发现分割式部署在后续切片上可节省 25% vCPU 与 50% RAM，但首次切片的部署时间略长；总体执行时间均低于 45 秒，满足临时/弹性网络场景需求；

**⚠️ 局限性**

局限性包括首次分割式切片部署资源消耗较大；当前未实现多实例 UPF 的负载均衡与边缘计算策略；对 RAN 动态资源调度与 O-RAN 集成仍需进一步研究。

---

## 239. When is 3D Worth It? A Resource-Performance Frontier for CNNs and Transformers in Lung CT

**arXiv ID:** 2606.06950 | [PDF](https://arxiv.org/pdf/2606.06950v1)

**作者:** Md Enamul Hoq `[一作]` (University of Arkansas for Medical Sciences), Donald Johann Jr. and Fred Prior `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

对比了2D、2.5D和3D输入在CNN和Vision Transformer两大架构下的性能与稳定性，使用固定训练协议进行实验。

**💡 创新点**

创新点在于系统评估输入维度对模型表现及失败模式的影响，而非提出新的网络结构。

**🔧 技术方法**

使用卷积神经网络和Vision Transformer，采用加权二元交叉熵训练20轮，评估ROC‑AUC、PR‑AUC、灵敏度/特异性，并记录GPU内存与推理时间。

**📊 数据集**

使用NLST肺部CT数据集（1977例，20例阳性）进行主实验，并用LIDC‑IDRI弱标签数据进行外部验证。

**📈 对比分析**

比较方法包括ROC‑AUC、PR‑AUC、阈值下的灵敏度/特异性以及计算资源消耗；2.5D CNN在ROC‑AUC 0.682上表现最佳，但模型间差异不显著，置信区间宽重叠。

**⚠️ 局限性**

主要局限包括单一患者级拆分、正样本稀缺导致置信区间宽、模型容量与优化未完全匹配、结果仅在单一数据集上验证，缺乏统计显著性检验。

---

## 240. Auditing Training Data in Domain-adapted LLMs: LoRA-MINT

**arXiv ID:** 2606.06946 | [PDF](https://arxiv.org/pdf/2606.06946v1)

**作者:** Gonzalo Mancera `[一作]` (Universidad Autonoma de Madrid), Francisco Jurado `[通讯]` (Universidad Autonoma de Madrid)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LoRA‑MINT，一个利用合成非成员样本的困惑度分布对 LoRA 微调的大型语言模型进行训练数据成员推断的审计框架。

**💡 创新点**

创新点在于：① 用困惑度分布做阈值判定；② 对分布进行极值过滤和均值调整以提高区分度；③ 在 LoRA 微调的不同模块注入点上进行消融研究，揭示注意力与 MLP 对成员推断的贡献；④ 在多模型、多数据集上实现高精度审计，超越现有基线。

**🔧 技术方法**

技术方法包括 LoRA 低秩适配、困惑度计算、GPT‑4 生成合成文本、分布极值过滤与均值偏移、AUC/TPR/FPR 评估等。

**📊 数据集**

使用的三个基准数据集为 CAMELMaths Instruction Dataset、Maths‑College、Medical‑o1‑SFT，模型覆盖 Qwen3‑4B、DeepSeek‑R1‑Distill‑Llama‑8B、Phi‑4‑14B、Llama‑3.2‑3B、Mistral 等。

**📈 对比分析**

与现有基线对比，LoRA‑MINT 在不同模型/数据集上取得 0.77–0.92 的精度和 0.77–0.90 的 AUC，显著提升成员推断效果。

**⚠️ 局限性**

局限性包括：① 仅验证 LoRA 微调，未覆盖其他 PEFT 方法；② 依赖合成样本生成质量；③ 对极端相似的非成员样本仍存在区分困难。

---

## 241. Heterogeneous Effects of Green Finance on Urban Decarbonization: Evidence from 285 Cities in China

**arXiv ID:** 2606.06986 | [PDF](https://arxiv.org/pdf/2606.06986v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 242. CF-JEPA: Mask-free forward prediction with asymmetric encoder utilization for time-series representation learning

**arXiv ID:** 2606.07031 | [PDF](https://arxiv.org/pdf/2606.07031v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 243. Declarative Skills for AI Agents in Knowledge-Grounded Tool-Use Workflows

**arXiv ID:** 2606.06923 | [PDF](https://arxiv.org/pdf/2606.06923v1)

**作者:** M. Danish Lim `[一作]` (Singapore Management University), Laura Wynter `[通讯]` (Singapore Management University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在银行客户服务工作流中比较了三种工具使用AI代理的编排方式，分别是基于自然语言技能文件的声明式代理、基于状态机的命令式代理以及无结构化基准代理，并评估了它们在不同检索质量下的任务成功率、合规性和成本表现；

**💡 创新点**

创新点在于首次系统性地将Agent Skills自然语言描述作为先验引入，证明其在模型存在程序性缺口时能提升准确性；同时验证了命令式状态机并不能如预期提高合规性；

**🔧 技术方法**

主要技术包括Dec‑POMDP框架下的策略定义、LLM推理（使用Claude、GPT、Qwen等多模型）、自然语言技能文件注入、状态机代码实现与检索增强（BM25与嵌入检索）；

**📊 数据集**

使用了τ‑Knowledge银行任务数据集（97个任务，698文档），包含公开和可检索工具以及业务政策文档；

**📈 对比分析**

通过在五个模型、两种检索模式（黄金检索与嵌入检索）下进行实验，结果显示声明式代理在黄金检索下显著提升pass^1（最高+~7%），而命令式代理在所有条件下表现低于基准；在噪声检索下，所有代理的pass^1急剧下降，声明式优势消失；

**⚠️ 局限性**

主要局限包括检索质量是瓶颈，声明式技能文件无法补偿检索噪声；命令式状态机对合规性的理论优势在实践中未得到验证；实验仅覆盖银行客服场景，缺乏跨领域验证。

---

## 244. The Fine-Tuning Trap: Evaluating Negative Transfer and the Role of PEFT in Sub-1B Mathematical Reasoning

**arXiv ID:** 2606.06920 | [PDF](https://arxiv.org/pdf/2606.06920v1)

**作者:** Rahul Nair `[一作]`, Chun Tao `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对小型语言模型（<1B参数）在数学推理任务上的全参数微调与参数高效微调（LoRA/DoRA）进行系统评测，并揭示全微调在模型小于300M时会导致负迁移，建议默认使用PEFT。

**💡 创新点**

创新点在于提出并验证了在容量饱和的子1B模型中存在的“稳定性-容量权衡”及“负迁移”现象，以及对LoRA与DoRA在不同任务上的性能差异和安全对齐风险的定量分析。

**🔧 技术方法**

主要技术包括参数高效微调方法LoRA和DoRA、深度学习框架的自定义CLI工具、梯度谱分析和Hessian特征值评估。

**📊 数据集**

使用了数学推理基准数据集OrcaMath、GSM8K、MATH和鲁棒性基准SVAMP，以及SmolLM2、Qwen2.5-0.5B、OLMo-1B、Gemma-2B等模型。

**📈 对比分析**

通过将模型分别以Zero-shot、Full FT、LoRA和DoRA四种方式微调，比较Exact Match准确率和训练稳定性；结果显示在<500M模型中Full FT低于Zero-shot，LoRA/DoRA显著提升，并在0.5B对齐模型上完全避免安全退化；而在>1B模型中Full FT优于PEFT。

**⚠️ 局限性**

局限性包括仅评估数学推理任务，未涵盖自然语言理解或生成任务；只比较了LoRA和DoRA两种PEFT；实验规模受限于H100单节点，未探讨更大规模或不同硬件的效果。

---

## 245. Don't Pause: Streaming Video-Language Synchrony for Online Video Understanding

**arXiv ID:** 2606.06991 | [PDF](https://arxiv.org/pdf/2606.06991v1)

**作者:** Zhenyu Yang `[一作]`, Changsheng Xu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种 Streaming Video‑Language Synchrony (SVLS) 框架，并通过 LyraV 实现在视频播放过程中实时生成文本而不暂停感知，实现在帧与词级别的同步。

**💡 创新点**

创新点在于引入训练无关的 Frame‑Driven Transition Controller (FDTC) 用于状态决策，以及轻量化的 Streaming Token Pacer (SToP) 动态调节每帧输出令牌数，使模型能够在不打断感知的前提下实现高精度同步与流畅叙事。

**🔧 技术方法**

采用冻结的在线 Video‑LLM（如 InternViT + InternLM2.5），结合验证式解码、有限状态机、Transformer 编码器预测令牌速率，并加入实时延迟约束，实现对视频流的持续感知与文本生成。

**📊 数据集**

在八个视频‑语言基准上评估，包括 OmniStar、Ego4D、StreamingBench、OVO‑Bench、OVBench、MVBench、LongVideoBench、Video‑MME，并使用 Live‑WhisperX‑526K 训练 SToP 的令牌速率预测。

**📈 对比分析**

与多种主动响应模型（LiveStar、VideoLLM‑Online、Dispider、LiveStar 等）对比，LyraV 在同步率（SR）达到 98.29%，实时 FPS 为 3.89，语义和流畅度评分均优于基线，同时保持与后端模型相同的理解能力。

**⚠️ 局限性**

限制主要包括阈值敏感性（FDTC 对 perplexity 变化的判断可能误判）、SToP 的速度预测依赖人类朗读率的弱监督易受噪声影响，以及在极低帧率或极高动态场景下的鲁棒性尚未充分验证。

---

## 246. SlimSearcher: Training Efficiency-Aware Web Agents via Adaptive Reward Gating

**arXiv ID:** 2606.07074 | [PDF](https://arxiv.org/pdf/2606.07074v1)

**作者:** Zequn Xie `[一作]`, Jinjie Gu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了 SlimSearcher 框架，结合 SFT 与 RL，通过多阶段门控实现高效的 web 代理。

**💡 创新点**

引入 Pareto‑efficient 过滤和 Adaptive Reward Gating，动态基于最优路径权衡准确性与工具调用/token 消耗，防止“效率陷阱”。

**🔧 技术方法**

使用多阶段门控奖励机制（Correctness Gate、Tool Efficiency Gate、Token Efficiency Gate）、GRPO 强化学习、Reward‑Guided Rejection Sampling 与动态锚定（AEA）等技术。

**📊 数据集**

在 GAIA、BrowseComp、XBench‑DeepSearch、HLE 等长时延 web 任务数据集上进行实验，并结合 13,863 条高质量示例构建训练集。

**📈 对比分析**

与闭源 OpenAI、Claude 等以及开源 Kimi、Qwen、WebExplorer 等基线对比，SlimSearcher 将工具调用轮数减少 17–58%，同时保持或提升准确率。

**⚠️ 局限性**

仅适用于文本推理，未覆盖视觉多模态；RL 依赖 SFT 初始化，工具权重统一化，未考虑不同工具的实际成本。

---

## 247. Bias in Filter Feature Selection Evaluation: A Meta-Analysis of Datasets, Baselines, and Experimental Design Choices

**arXiv ID:** 2606.07068 | [PDF](https://arxiv.org/pdf/2606.07068v1)

**作者:** Malick Ebiele `[一作]` (University College Dublin), Rob Brennan `[通讯]` (University College Dublin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 1994–2025 年发表的 28 篇滤波特征选择研究进行元分析，系统评估数据集、基线和实验设计对新方法 win rate 的影响，并给出改进原则。

**💡 创新点**

首次量化 FFS 评估中的选择偏差，提出 win rate 与 success rate 指标，并通过多元线性回归揭示数据集数、基线数和新方法数对性能的中等解释力，为特征选择评估提供基准与实践建议。

**🔧 技术方法**

元分析、手工数据抽取、线性回归、多元相关分析、R² 与 RMSE 评估。

**📊 数据集**

使用 28 篇研究中的所有数据集，平均每篇 2–19 个，涵盖高维基因表达、图像、文本等多类型数据，特别关注 nci9 等经典数据集。

**📈 对比分析**

将新方法与多基线以及全特征基线对比，用 win rate 衡量在所有数据集上能否胜过基线；平均 win rate 为 78%，但随数据集和基线数量增加而下降，成功率相对较低。

**⚠️ 局限性**

样本仅 28 篇，偏重滤波方法，未覆盖 wrapper/embedded；手工抽取可能漏信息；回归模型简单，R² 仅 0.33，未考虑统计显著性和更细粒度的评估。

---

## 248. Extending Responsibility-Sensitive Safety for the Assessment of Offloaded Autonomous Driving Services

**arXiv ID:** 2606.07067 | [PDF](https://arxiv.org/pdf/2606.07067v1)

**作者:** Robin Dehler `[一作]` (Ulm University), Michael Buchholz `[通讯]` (Ulm University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过将RSS扩展到支持本地与远程服务的不同响应时间，实现了安全的功能迁移和快速回退。

**💡 创新点**

创新点在于提出RSS-guarded状态，结合热备温备机制实现安全与效率的平衡。

**🔧 技术方法**

使用了责任敏感安全(RSS)、服务导向架构(SOA)、函数迁移框架(SOFOF)、5G V2X通信及温备热备切换。

**📊 数据集**

实验采用基于lanelet地图与IDM模型的仿真环境以及真实世界5G CAV轨迹数据。

**📈 对比分析**

与SOFOF和MUFASA对比，安全性显著提升，offloading时间略下降但仍可接受，体现安全-效率权衡合理。

**⚠️ 局限性**

局限包括对响应时间的离线估计依赖、仅验证轨迹规划服务、未纳入网络QoS预测等因素。

---

## 249. Meaning in Order, Order in Meaning: Semantic R-precision for Keyphrase Evaluation

**arXiv ID:** 2606.07057 | [PDF](https://arxiv.org/pdf/2606.07057v1)

**作者:** Shamira Venturini `[一作]` (Karlsruhe University of Applied Sciences), Steffen Kinkel `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了语义 R-Precision（SemR-p）评估指标，将语义相似度融入 R-Precision 框架，实现对自动生成关键词的排名敏感且语义友好的评估。

**💡 创新点**

创新点在于同时兼顾完全匹配与语义相近预测的打分，并通过排名感知的 R-Precision 结构提升人类判断的一致性。

**🔧 技术方法**

使用了句子Transformer（mean‑pool）进行词组嵌入，基于词干匹配、余弦相似度和 k‑最近邻平均计算语义分数，并结合 R-Precision 计算。

**📊 数据集**

实验基于 kp20k（科学摘要）和 kptimes（新闻文章）两大数据集，评估了 8 个代表性模型（包含无监督、监督、生成及 LLM 模型）。

**📈 对比分析**

与十种基线指标（Lexical F1@M、NDCG、AP、R-Precision、SemP/R/F1、BERTScore 等）比较，SemR-p（k=3）与大多数基线呈正相关，能够显著区分模型与数据集，并在系统级排名上与现有指标保持较高一致性。

**⚠️ 局限性**

局限性包括未与人工评估直接验证、对嵌入模型和聚合方式敏感、k 参数需手动设定、未显式惩罚语义冗余或缺乏多样性。

---

## 250. Predictive Autoscaling in Cloud-Native and Federated Cloud-Edge Computing Environments: A Taxonomy and Future Directions

**arXiv ID:** 2606.07046 | [PDF](https://arxiv.org/pdf/2606.07046v1)

**作者:** Bablu Kumar `[一作]` (University of Melbourne), Rajkumar Buyya `[通讯]` (University of Melbourne)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统综述并梳理了云原生与联邦云‑边缘环境下的预测性自动扩缩技术，提出了以触发器、目标、预测模型与评估维度为轴的四维分类法，并构建了统一的 MAPE‑Guided 自动扩缩框架。

**💡 创新点**

创新点在于：①将 Kubernetes CRD 与 Operator 结合形成可声明式的预测扩缩管道；②引入漂移感知（ADI）与不确定性校正机制以提升联邦学习场景的鲁棒性；③提出针对联邦学习工作负载的专属预测模型与策略；④整合 Transformer‑类长序列预测（Informer、MV‑Transformer）与传统统计/ML 模型，形成多模型动态选取。

**🔧 技术方法**

核心技术包括：Kubernetes 原生自动扩缩器（HPA、VPA、Cluster Autoscaler、KEDA）与自定义 CRD；监测与分析层使用 Prometheus、Metrics Server；预测层使用 ARIMA、LSTM、GRU、CNN‑LSTM、Informer、Autoformer、MV‑Transformer；控制层采用 MAPE 循环与 Operator‑Reconciliation；联邦学习场景引入 FedScaleEdge、FedALoRA、FedInv 等框架及差分隐私（DP）。

**📊 数据集**

主要引用公开工作负载与指标数据集：Google Cluster Trace、Microsoft Azure 负载、Alibaba Cluster Trace、NASA HTTP 数据集以及论文中对 FedScaleEdge 等联邦学习实验的日志，所有数据用于构建与验证预测模型。

**📈 对比分析**

通过系统性文献筛选与分类，对比现有工作在触发器、目标、模型与评估维度上的优缺点，评估维度包括延迟、弹性、成本效率、预测准确率、稳定性与 QoE。虽然未在单一实验中给出数值性能，但综述指出 Transformer‑类模型在预测准确率和长期稳定性方面优于传统统计/ML 方法，并且 CRD‑Operator 机制在实际 Kubernetes 环境中实现了低延迟、可审计的自动扩缩。

**⚠️ 局限性**

局限性包括：①缺乏统一的跨层（网络–边缘–云）协同调度框架；②对差分隐私与联邦学习的完整成本分析不足；③在多租户与能源约束环境下的公平性与可解释性尚未系统研究；④大规模联邦学习环境下模型漂移与通信延迟对预测的影响尚未充分验证。

---

## 251. Constructing VAE Latent Spaces with Prescribed Topology

**arXiv ID:** 2606.07058 | [PDF](https://arxiv.org/pdf/2606.07058v1)

**作者:** Jilles S. van Hulst `[一作]` (Eindhoven University of Technology), Duarte J. Antunes `[通讯]` (Eindhoven University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种可构造的 VAE 设计流程，能够在潜在空间中实现任意预先指定的非欧几里得拓扑。

**💡 创新点**

创新点在于将任意可被分解为基本因子（圆、区间、直线或其有限群商）的流形，利用因子化分布和 G‑不变特征映射实现闭式 KL 计算与梯度可微的重参数化，并通过锚定约束消除坐标自由度。

**🔧 技术方法**

使用了因子化 Gaussian/Wrapped‑Normal/Kumaraswamy 等分布、Reynolds 归约构造 G‑不变特征、锚点损失、以及传统 VAE 的 reparametrization trick。

**📊 数据集**

在合成数据上验证了圆柱体、二维环面和 Möbius 条带；在真实数据上使用了旋转 MNIST（ℝ²×S¹）和循环移动 MNIST（ℝ²×T²）。

**📈 对比分析**

与标准高斯 VAE 在相同潜在维度下进行对比；使用重建误差、先验一致性误差和几何应力等指标，实验显示拓扑匹配模型在所有活跃正则化强度下均优于基线，尤其在几何一致性和生成质量上表现显著提升。

**⚠️ 局限性**

局限性包括：需要预先指定拓扑，无法学习或自适应拓扑；只能处理可由有限群商得到的有限生成流形；对更高基数或连续对称群的处理仍有待扩展；在离散混合流形或多分支情况下 KL 退化问题仍未完全解决。

---

## 252. Front-to-Attractors: Modifying the Front-to-Front Heuristic in Bidirectional Search

**arXiv ID:** 2606.07047 | [PDF](https://arxiv.org/pdf/2606.07047v1)

**作者:** Alvin Zou `[一作]` (Carnegie Mellon University), Maxim Likhachev `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种新的双向搜索启发式——前向到吸引子（F2A）启发式，并将其集成到现有的双向搜索框架中。

**💡 创新点**

创新点在于用动态维护的一小部分吸引子代替传统的前向到前沿（F2F）中的所有前沿状态，既保持了F2F的高信息量，又显著降低了计算开销。

**🔧 技术方法**

技术包括吸引子维护机制、与双向搜索的兼容实现、两种优化（新吸引子NA与关联状态AS）以及基于吸引子的启发式评估。

**📊 数据集**

使用的数据集为三类经典搜索基准：2D Grid Pathfinding（DAO与迷宫地图）、15拼图（Korf测试集）和14-薄饼谜题（随机实例）。

**📈 对比分析**

通过与F2E、F2F、BAE*、A*以及VBi-HS和NBS两种主流双向搜索实现进行对比，F2A在多域实验中将启发式评估次数减少多达11.2×，节点扩展次数比F2E平均少4.8×，总体运行时间保持与最优方法相近。

**⚠️ 局限性**

局限性包括在某些域（如拼图和薄饼）中F2A天然退化为F2E，需要额外优化才能发挥优势；同时，吸引子维护与参数设定（如阈值δ）对性能有显著影响。

---

## 253. ForensicConcept: Transferable Forensic Concepts for AIGI Detection

**arXiv ID:** 2606.07034 | [PDF](https://arxiv.org/pdf/2606.07034v1)

**作者:** Menyanshu Zhou `[一作]` (Xiamen University), Rongrong Ji `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出ForensicConcept框架，将AI生成图像检测器的判别证据转化为可解释、可迁移的“法医概念”，并通过概念注入提升跨生成器的检测性能。

**💡 创新点**

创新点包括：①通过Transformer Attribution定位决策关键补丁并聚类生成概念码表；②引入CleanDIFT扩散模型特征作为生成轨迹参考，并用邻域一致性CKNNA量化不同骨干与生成轨迹的几何对齐；③利用概念码表注入实现跨骨干、跨生成器的迁移，并用CKNNA预测注入效果。

**🔧 技术方法**

技术手段包括：LoRA适配器微调、Transformer Attribution与注意力卷积、K-means聚类构建概念码表、概念对齐投影、CleanDIFT特征提取、CKNNA邻域一致性度量、概念导向代码书注入（CGCI）等。

**📊 数据集**

主要使用GenImage、GAN-family（七种GAN生成器）和Chameleon等公开基准数据集，训练集采用Stable Diffusion v1.4合成图像与ImageNet真实图像。

**📈 对比分析**

在GenImage基准上，ForensicConcept（DINOv3+概念）平均准确率达92.0%，显著优于CLIP 83.7%和Effort 91.1%；在GAN-family和Chameleon上也取得最高或次高的平均性能。通过概念注入，CLIP平均准确率从83.7%提升至88.2%。

**⚠️ 局限性**

局限性包括：①对生成轨迹参考的依赖（CleanDIFT）可能限制在更广泛生成模型上的适用性；②CKNNA对邻域大小敏感，需要手动调参；③概念注入的效果受码表质量影响，可能在极端分布偏移或后处理强度高的情况下下降。

---

## 254. Environment-Division Multiple Access: an Enabler for AI-Native Multiple Access

**arXiv ID:** 2606.07025 | [PDF](https://arxiv.org/pdf/2606.07025v1)

**作者:** Zhiguo Ding `[一作]` `[通讯]` (Nanyang Technological University), Zhiguo Ding (Nanyang Technological University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种利用可重构传播环境（EDMA）的多接入方案，并探讨了其与人工智能（AI）融合的两种方式：AI 辅助 EDMA 与 AI 原生 EDMA。

**💡 创新点**

创新点在于将无线传播环境视为可调节的资源域，结合柔性天线、智能反射表面（STAR‑RIS）、可移动天线等技术，实现通过主动或被动重构环境来抑制多用户干扰，并通过 AI 对环境感知、资源分配、预测与重构进行闭环优化。

**🔧 技术方法**

主要技术包括可重构智能表面（RIS/IRS）、柔性天线（pinching、LCX 等）、ISAC、通道知识图（CKM）、数字孪生、强化学习、无监督学习以及基于场景的 AI 模型。

**📊 数据集**

文中未给出具体实验数据集，重点在理论框架与设计思路，若要实现需收集环境感知数据、用户位置、移动轨迹、反射/散射参数等信息。

**📈 对比分析**

文章未进行实验对比或性能评估，主要提供概念验证与未来研究方向，预期通过 AI 能显著提升多用户资源利用率、降低能耗并实现低时延通信。

**⚠️ 局限性**

主要局限包括：环境感知与建模的实时性与准确性挑战、AI 模型训练所需的大规模、多样化数据缺乏、系统复杂度与部署成本高、以及在动态高速移动场景下的实时重构难题。

---

## 255. A Multi-Operator Mixed-Reality Interface for Multi-Robot Control and Coordination: Co-Located and Private Workspace Collaboration

**arXiv ID:** 2606.07013 | [PDF](https://arxiv.org/pdf/2606.07013v1)

**作者:** Omotoye Shamsudeen Adekoya `[一作]` (University of Genoa), Carmine Tommaso Recchiuto `[通讯]` (University of Genoa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文构建了一个多操作员混合现实接口（HORUS），能够支持多机器人监控、任务分配与远程操作，并通过实验比较了共置共享工作空间与私有工作空间两种协作模式。

**💡 创新点**

创新点在于引入了每机器人租赁机制实现冲突预防、基于空间锚的多操作员共享对齐，以及在不改动底层机器人控制工具的前提下显著提升协作流畅性。

**🔧 技术方法**

技术上使用了Unity 3D、Meta Quest 3头显、ROS 2、Unity Netcode for GameObjects、Isaac Sim仿真、空间锚对齐与多机器人注册驱动场景构建等。

**📊 数据集**

实验使用了在Isaac Sim中生成的两套室内环境（办公室与医院），虚拟的Nova Carter机器人以及放置在地图上的AprilTag标签作为搜索目标。

**📈 对比分析**

采用对照交叉实验设计，比较两模式在同一10分钟搜索任务中的标签发现数、主观可用性、工作负荷与协作感受；两模式任务完成率相近，但共置模式在主观协作度、空间共享感及偏好率上显著更高。

**⚠️ 局限性**

局限性包括基于仿真而非真实机器人，样本仅18对，且共置模式易受头显高度校准与空间对齐误差影响，未能完整评估远程通信延迟和更复杂环境对协作的影响。

---

## 256. A Geometric View for Understanding Concept Learning and Neuron Interpretation in Sparse Autoencoders

**arXiv ID:** 2606.07007 | [PDF](https://arxiv.org/pdf/2606.07007v1)

**作者:** Chenhao Zhang `[一作]` (University of Washington), Su-In Lee `[通讯]` (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出了一个统一的几何与集合论框架，用来描述稀疏自编码器（SAE）中的概念学习与神经元解释，明确将概念视为数据点集合，将学习定义为人类概念与模型诱导概念之间的集合对齐。

**💡 创新点**

创新点在于：①引入三种学习模式（检测、分离、逼近），给出各自的几何条件、误差上界与容量约束；②解释SAE常见现象（特征拆分、吸收、族、层次化）与几何关系；③将概念学习与神经元解释映射到形式概念分析的概念格，揭示二者并不必一致的多对多结构。

**🔧 技术方法**

采用稀疏自编码器（ReLU SAE 与 Top‑K SAE）为模型，利用超平面与超平面排列的几何分析、凸包与可凸性判定、边界光滑性下的误差率计算，以及组合容量推导；同时使用形式概念分析与Galois连接来构造概念格。

**📊 数据集**

实验数据采用二维合成数据，构造若干“概念”簇（互斥或部分重叠），用于可视化和验证理论。

**📈 对比分析**

比较方法主要通过 F1 分数、分离误差与逼近误差来评估概念学习质量；实验表明：更大的扩张因子与更高的可选神经元数能提升分离与逼近的 F1；单神经元往往不足，单位集合能显著提升性能；但在过度稀疏或概念高度重叠时，F1 会下降。相较于理论预测，实际性能满足但未必达到理论极限。

**⚠️ 局限性**

局限性包括：①理论分析主要针对 ReLU SAE，未充分扩展到其他激活或结构；②使用的合成二维数据无法检验在真实高维数据上的泛化；③概念格的完整构造与分析仍为预研，未给出完整的多对多匹配算法；④实验中使用的基于 top‑N 的启发式选择可能无法找到全局最优单位。

---

## 257. Accelerating Multi-Objective Bayesian Optimisation via Predictive-Gradient Catalysts

**arXiv ID:** 2606.06984 | [PDF](https://arxiv.org/pdf/2606.06984v1)

**作者:** Alma Rahat `[一作]` (Loughborough University), Richard Allmendinger `[通讯]` (University of Manchester)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于高斯过程预测梯度的催化框架，在多目标贝叶斯优化（MOBO）中将局部Pareto stationarity 信息与现有Pareto合规的采集函数相结合，从而加速收敛

**💡 创新点**

创新点在于将预测梯度作为可加的“催化剂”来增强采集函数，而不是替换采集函数；同时提出两种实现：自适应MGDA权重和预设权重的方案，分别对应全局探索与聚焦探索

**🔧 技术方法**

使用技术包括：独立高斯过程模型、GP预测梯度、Multiple‑Gradient Descent Algorithm (MGDA)、预设权重向量、增强 Tchebycheff 标量化、Bi‑POP CMA‑ES 采集优化、Pareto‑only LOOCV 与标准化对数损失评估

**📊 数据集**

在DTLZ基准系列（DTLZ1–DTLZ7）上进行实验，所有问题均设定2个目标、10个决策变量，初始设计10n点、评估预算30n点

**📈 对比分析**

与四种基线采集函数（EHVI、AugTch、tMPoI、SAF）在同一初始设计下进行11次独立实验，使用最终超体积（HV）评估；实验显示在模型准确性较高且问题近似平稳的情况（如DTLZ2、DTLZ5）中，催化器显著提升性能（约32%实验显著改善，40%相等，少量恶化）

**⚠️ 局限性**

局限性包括：依赖GP预测梯度的准确性；在非平稳或模型误差大的问题（DTLZ1、DTLZ3、DTLZ4、DTLZ6）中效果有限；MGDA方案在评估预算紧张时可能稀释搜索力度；预设权重缺乏FJ条件保证，可能导致局部最优而非全局最优

---

## 258. Mutual Information Optimization via K-Recursion and Automatic Differentiation for Linear Gaussian Wireless Networks

**arXiv ID:** 2606.06982 | [PDF](https://arxiv.org/pdf/2606.06982v1)

**作者:** Tadashi Wadayama `[一作]` (Nagoya Institute of Technology), Na Siqi `[通讯]` (Nagoya Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种可微分框架，用 K‑递归推导线性高斯 DAG 的协方差并在此基础上计算闭式互信息，随后通过投影梯度上升在全局约束下实现端到端 MI 最优化。

**💡 创新点**

核心创新在于：① 统一的 K‑递归可推导所有节点对协方差，自动包含跨节点交叉协方差；② 该递归构成完整的前向计算图，可直接由复数反向自动微分得到精确的 Wirtinger 梯度；③ 无需针对每种拓扑手工推导梯度，极大提升了可扩展性与代码可维护性。

**🔧 技术方法**

技术手段包括：线性高斯 DAG 建模、K‑递归协方差推导、复数 Wirtinger 微分、PyTorch 复数自动微分、投影梯度上升（PGA）以及对输入协方差的虚边扩展。

**📊 数据集**

实验使用随机生成的合成网络拓扑：单链 MIMO、菱形、双跳 AF 中继、虚边输入造型以及一层 11 节点、5 层的多层网络；未使用公开数据集。

**📈 对比分析**

与基准方法对比：在可求解的单链 MIMO 与虚边输入的水分配（water‑filling）结果下，PGA 达到与理论最优几乎相同的互信息；在其他拓扑下，PGA 显著优于均匀基准，并在多层网络上提升约 2× 的互信息。

**⚠️ 局限性**

局限性包括：优化为非凸问题，仅保证 KKT 极值；仅适用于线性高斯模型，无法直接处理多普勒/衰落等随机通道；对非线性或非高斯组件需要进一步扩展（如 Bussgang 近似）。

---

## 259. Decision-Theoretic Stopping Rules for Document Screening

**arXiv ID:** 2606.07071 | [PDF](https://arxiv.org/pdf/2606.07071v1)

**作者:** Aaron H. A. Fletcher `[一作]`, Mark Stevenson `[通讯]` (University of Sheffield)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于决策理论的技术辅助审阅停止规则，使用期望完美信息（EVPI）来平衡检索成本与决策效益。

**💡 创新点**

将停止决策视为期望效用最大化问题，导出 Greedy、Smooth、Batch 三种可操作的停止策略，并将其应用于专利检索与系统综述两类不同的证据结构。

**🔧 技术方法**

采用决策理论、价值信息分析、概率校准（等距回归）以及贝叶斯元分析模型（Reitsma）来估计每个文档的相关性概率和 EVPI。

**📊 数据集**

在 CLEF‑IP 2009 专利数据集和 CLEF eHealth 2017‑2019 诊断试验准确性数据集上进行实验。

**📈 对比分析**

与四类基线（固定预算、QBCB、Kneedle、GRLStop）对比，利用净效用、召回、决策一致率和 ICER 评估；结果显示新方法在多种成本和风险设定下均获得更高净效用且保持近乎恒定的后悔。

**⚠️ 局限性**

实验仅基于固定的 BM25 排名和充分标注的训练数据，未涵盖主动学习或重排情形，且对低资源场景的概率校准与模型可迁移性有限。

---

## 260. mmPISA-bench: Do LLMs Reason Equally Well Across 43 Languages?

**arXiv ID:** 2606.07069 | [PDF](https://arxiv.org/pdf/2606.07069v1)

**作者:** Yerzhan Sapenov `[一作]` (Independent Scholar), Jaromir Savelka `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了名为 mmPISA-bench 的多语言推理基准，包含 25 道多选题，并通过 OECD PISA 官方翻译及机器翻译扩充至 43 语言，共 2,150 个样本。

**💡 创新点**

创新点在于使用高质量官方翻译并与机器翻译进行对比，证明机器翻译可替代官方翻译；同时系统评估多语言推理的准确率与成本关系。

**🔧 技术方法**

使用了 OpenAI 的 GPT 系列（如 GPT-4、GPT‑3.5‑turbo）和 Anthropic 的 Claude，并通过不同推理努力配置（无推理、低/中/高推理、双重提示）进行评测。

**📊 数据集**

数据集基于 OECD PISA 2018/2022 的阅读与数学题目，手工筛选符合条件的题目并收集 43 种语言的官方翻译及 Google Translate 生成的机器翻译。

**📈 对比分析**

通过在各语言、翻译类型和推理配置下对两个模型进行 107,500 次 API 调用，比较准确率与推理成本，结果显示两模型在多数语言上达到与 15 岁学生相当的准确率，机器翻译不显著影响性能，但某些语言推理成本更高且准确率更低。

**⚠️ 局限性**

局限包括仅评估两个封闭源 LLM，未覆盖开源模型；只评估文本推理，未考虑视觉或多模态任务；推理成本分析仅基于 token 计价，未考虑硬件差异；缺乏对极低资源语言的深入研究。

---

## 261. Modeling semantic association in self-paced reading with language model embeddings

**arXiv ID:** 2606.07066 | [PDF](https://arxiv.org/pdf/2606.07066v1)

**作者:** Sara Møller Østergaard `[一作]` (Tilburg University), Bruno Nicenboim `[通讯]` (Tilburg University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究基于语言模型嵌入的语义关联对自然阅读中的加工难度的影响，并通过不同嵌入模型与上下文长度实现多种语义关联估计。

**💡 创新点**

首次系统比较了无上下文词嵌入（word2vec）与上下文句子嵌入（sentence embeddings）在估计语义关联时的差异，揭示句子嵌入能更好捕捉语义关联并预测N400及阅读时长。

**🔧 技术方法**

使用 Bayesian 层级回归模型与 Savage‑Dickey Bayes 因子比较不同语义关联实现的效应；采用词向量、句子向量与加权/窗口化等多种上下文表示方式；利用 cos‑sim 计算语义关联。

**📊 数据集**

Dutch 的 tint 语料库，包含 56 名参与者在 self‑paced 阅读任务中产生的 EEG 与阅读时长数据，覆盖 8 篇约 600 词的自然文本。

**📈 对比分析**

通过 Bayes 因子检验不同实现的效应是否显著。结果显示：使用句子嵌入时，语义关联对 N400 与阅读时长均产生显著效应（贝叶斯因子约 3–5 以上）；而使用词嵌入时无显著效应。上下文长度在句子嵌入实现中显著影响效应强度，词嵌入实现则不显著。

**⚠️ 局限性**

局限性包括：仅使用单一词嵌入与单一句子嵌入模型；仅针对 Dutch 语料，缺乏跨语言验证；实验仅涵盖 self‑paced 阅读与 EEG，未检验其他阅读测量；以及模型选择对结果的影响仍需进一步探索。

---

## 262. STREAM: Stochastic Riemannian Flow Matching with Anisotropic Decoder for Digital Histopathology Image Generation

**arXiv ID:** 2606.07036 | [PDF](https://arxiv.org/pdf/2606.07036v1)

**作者:** Won June Cho `[一作]` (DEEPNOID Inc), Hongjun Yoon `[通讯]` (DEEPNOID Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种名为STREAM的无条件数字病理图像生成框架。

**💡 创新点**

创新点在于将Riemannian流匹配与随机桥式扰动相结合，并使用基于速度场奇异值分解的各向异性解码器。

**🔧 技术方法**

采用Riemannian流匹配、Brownian桥桥式噪声、Diffusion Transformer以及SVD导向的各向异性噪声注入技术。

**📊 数据集**

使用TCGA-BRCA和TCGA-COADREAD这两个大规模病理图像数据集。

**📈 对比分析**

与ZoomLDM、PixCell以及RAE、SVG等基线对比，STREAM在gFID、rFID等指标上实现了最高或最接近的性能。

**⚠️ 局限性**

主要限制是理论结果基于人群层面且对单个token的假设，且对域匹配的VFM依赖较强。

---

## 263. StainFlow: Entity-Stain Tracking and Evidence Linking for Process Rewards in GUI Agents

**arXiv ID:** 2606.07027 | [PDF](https://arxiv.org/pdf/2606.07027v1)

**作者:** Haojie Hao `[一作]` (Beihang University), Xianglong Liu `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为StainFlow的过程奖励模型，用视觉可验证的任务实体追踪实体染色浓度沿轨迹演化，以实现对GUI Agent的细粒度奖励分配；

**💡 创新点**

创新点在于将任务实体的可视化证据转化为“染色流”，通过全局实体染色跟踪实现客观的任务阶段划分，并在局部通过实体关联的证据窗口动态验证关键节点，解决了传统里程碑划分的主观性和固定窗口缺乏长程证据的问题；

**🔧 技术方法**

核心技术包括基于VLM的实体抽取与识别、实体染色浓度更新与动态阈值检测、全局实体染色跟踪与局部证据链接、以及基于密集染色信号与关键节点验证的奖励合成与RL策略更新；

**📊 数据集**

实验数据集为AndroidWorld（用于在线RL训练）和OGRBench（用于轨迹完成判断），并使用多种多模态验证器（Qwen、GPT‑5、Gemini‑3‑Flash等）进行评估；

**📈 对比分析**

与GUI‑Critic‑R1、ADMIRE、OS‑Themis等基线相比，StainFlow在AndroidWorld上提升了相对3.2%成功率、在OGRBench上提升了1.8%整体准确率；同时在奖励分布、关键节点识别、平均步骤数等指标上均优于基线；

**⚠️ 局限性**

局限性包括实验规模有限，未在更大规模数据或更长训练周期验证，且模型对实体识别的准确性高度依赖，若视觉检测失败可能影响奖励质量。

---

## 264. PCCL: Process Group-Aware Scalable and Generic Collective Algorithm Synthesizer

**arXiv ID:** 2606.07019 | [PDF](https://arxiv.org/pdf/2606.07019v1)

**作者:** William Won `[一作]` (Georgia Institute of Technology), Tushar Krishna `[通讯]` (Georgia Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了可扩展、通用且对过程组感知的集合通信算法合成框架 PCCL，能够在大规模异构网络上自动生成拓扑感知的集合通信算法。

**💡 创新点**

创新点包括：①引入基于时间展开网络（TEN）的 BFS 路径搜索实现可扩展性；②通过过程组信息实现对子集设备的感知；③支持异构、非对称拓扑和任意集合模式（All-to-All、All-Reduce、All-Gather、Reduce-Scatter、Broadcast、Scatter、Scatterv）；④在 512 NPU 集群内仅 11.68 分钟完成 All-to-All 合成。

**🔧 技术方法**

采用了时间展开网络（TEN）、广度优先搜索（BFS）路径规划、α‑β 链路延迟/带宽模型、对开关的显式建模以及基于距离排序的启发式调度。

**📊 数据集**

使用 ASTRA‑sim 仿真平台，在 6×6、8×8 Mesh、3D Hypercube、2D Switch 等多种拓扑（规模从 36 到 512 NPUs，甚至 1000 NPUs）中验证，并对比 128 MiB All‑to‑All 等集合通信。

**📈 对比分析**

通过与 TE‑CCL 及传统 pairwise send/recv 基线对比，PCCL 在 512 NPU Mesh 中合成时间仅 11.68 分钟，比 TE‑CCL 快 3+ 订单；All‑to‑All 运行速度比 CCL 提升 1.33×，对过程组感知的 All‑to‑All 提升 2.33–3.03×，并在多过程组场景下实现 2.8× 的网络利用率提升。

**⚠️ 局限性**

局限性包括：尚未覆盖复杂多收发交换机（非单播）或跨集群网络；合成结果需要后端映射；对极端大规模网络（>10k NPUs）仍有可扩展性挑战；未支持实时在线重合成或动态拓扑变化。

---

## 265. The Sim-to-Real Gap of Foundation Model Agents: A Unified MDP Perspective

**arXiv ID:** 2606.07017 | [PDF](https://arxiv.org/pdf/2606.07017v1)

**作者:** Xiaoou Liu `[一作]` (Arizona State University), Hua Wei `[通讯]` (Arizona State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于MDP的统一框架，系统化描述并评估基础模型代理在真实环境中面临的观察、动作、转移与奖励四个维度的sim-to-real差距，并给出对应的解决策略；

**💡 创新点**

创新地将传统RL的sim-to-real问题映射到FM代理，形成统一的术语体系和标准化的压力测试基准，强调多语言、多模态缺口的细化与跨通道评估；

**🔧 技术方法**

采用MDP分解、领域随机化、域适配、动作屏蔽、延迟感知控制、奖励重塑与增强、工具调用扰动等经典RL技术，并结合多语言、时延、错误注入等自制扰动；

**📊 数据集**

主要使用公开工具调用基准（如OpenAI Tool‑Use、Multilingual Tool‑Calling数据集）以及自制的多语言、多模态、时延与错误注入数据；

**📈 对比分析**

通过在模拟与真实环境中计算ψ_s与ψ_r的差距进行基准对比，报告不同模型在多语言环境下的错误率提升，展示了方法在提升跨语言鲁棒性方面的有效性；

**⚠️ 局限性**

局限在于缺乏真实多模态与长期部署数据，评估仍依赖人工注入扰动，模型对多语言/多模态自适应能力有限，且评估指标相对单一，未充分覆盖任务协同与成本约束等实际需求。

---

## 266. Towards Unified Song Generation and Singing Voice Conversion with Accompaniment Co-Generation

**arXiv ID:** 2606.07015 | [PDF](https://arxiv.org/pdf/2606.07015v1)

**作者:** Ziyu Zhang `[一作]` (Northwestern Polytechnical University), Lei Xie `[通讯]` (Northwestern Polytechnical University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 UniSinger，一种端到端框架，将歌曲生成与歌声转换（SVC）统一，实现零射声克隆与伴奏共生成。

**💡 创新点**

创新点包括：跨任务声学嵌入空间实现声纹细粒度控制；采用任务特定模态遮蔽的渐进式课程学习解决多任务优化冲突；以及MM-DiT骨干网络实现跨模态音频生成。

**🔧 技术方法**

使用多模态编码器（Qwen2.5、Zipformer、CAM++等）、VAE音频码流、MM-DiT（联合与单一层）以及流匹配训练，配合任务遮蔽与声纹广播技术。

**📊 数据集**

在30k小时的“in-the-wild”歌曲数据上训练，额外加入5k小时伴奏唱歌数据，采用内部500条平衡样本作为评估集，涵盖中英双语和男女声。

**📈 对比分析**

与SongLM、YuE、ACE-Step、DiffRhythm+等基线相比，在PER、声纹相似度、SongEval等客观指标以及MOS、和谐度等主观指标均实现或接近最先进水平；SVC与伴奏共生成版在和谐度上明显优于单独SVC。

**⚠️ 局限性**

局限性：由于使用野外数据，仍存在轻微音质失真；BGM共生成版本在部分指标上略低；模型参数量较大，推理成本较高。

---

## 267. Fast Bounded-Independence Functions and Their Duals

**arXiv ID:** 2606.07009 | [PDF](https://arxiv.org/pdf/2606.07009v1)

**作者:** Martijn Brehm `[一作]` (University of Amsterdam), Nicolas Resch `[通讯]` (University of Amsterdam)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一系列高效的 t‑wise 线性独立哈希函数、快速编码及其快速双码，并证明这些构造能够同时满足 Gilbert–Varshamov (GV) 极限与最优列表解码能力。基于这些构造，进一步设计了完美安全的多方计算协议和加密矩阵‑向量乘法协议，且各步骤的布尔/算术电路大小均保持线性。

**💡 创新点**

创新点主要包括：
- 证明了常数 t 时即可实现的 t‑wise 线性独立哈希函数，其代数度为 (q−1)log_q t（当 q=2 时即为 log_2 t），同时保持线性电路规模；
- 通过快速 LUOF（线性均匀输出族）构造，给出了同时满足 GV 极限且编码/解码均为线性电路的正则码与其双码；
- 引入 t‑LUOF 概念，进一步实现了 t‑local‑similarity 的随机码，从而得到既快速又具有最优列表解码（list size O(1/ρ)) 的码与其双码；
- 将上述构造应用于信息理论完美安全的多方计算与加密矩阵‑向量乘法，首次实现了总布尔电路复杂度与参与方数线性且安全阈值接近 1/2 的协议。

**🔧 技术方法**

核心技术包括：
- 继承 IKOS 的三步编码-哈希-提取框架，并将其中的常数大小哈希函数提升到 t‑wise 独立；
- 利用 Druk–Ishai 的快速 LUOF 架构，结合距离足够大的线性可加码，构造可双向快速编码的码对；
- 通过对 LUOF 的 t‑wise 扩展（t‑LUOF）与 XOR Lemma 的推广，证明任意 t 线性无关输入在输出空间中分布均匀且相互独立；
- 使用 local‑similarity 理论，将 t‑LUOF 直接转化为列表解码性能；
- 在密码学应用中运用秘密共享与双码的完美隐私性质，构建 OLE 与通用多方计算协议。

**📊 数据集**

本文为理论研究，未使用实验数据集；所有结果均在抽象的有限域（q 为任意素数幂）和电路模型（有限 fan‑in 2 的算术/布尔电路）下给出。

**📈 对比分析**

与之前工作相比，本文的构造在以下方面取得显著提升：
- 传统的 IKOS、Brehm‑Resch 方案仅支持二元域、固定 1/2 码率、非零失败概率；本方案在任意 q、任意码率 R ∈ (0,1)、接近零失败概率下均可实现；
- 电路规模从之前的 O(n·polylog n) 或 12n（常数较大）降低到严格的 O(n)；
- 代数度从先前的 O(log n) 降至 log_2 t（最优）；
- 在列表解码方面，本方案在任意 t 处实现了距离接近 GV 极限且列表大小为 O(1/ρ) 的最优结果，先前仅有 t=1 的情况；
- 在多方计算与加密矩阵乘法中，协议总布尔电路复杂度从 O(n·polylog n) 降至 O(n)，并在安全阈值上几乎达到 1/2。

**⚠️ 局限性**

主要局限包括：
- 构造依赖于常数 t 与 q，若 t 或 q 取较大值，隐藏常数将呈指数增长；
- 虽然电路规模为 O(n)，但实际实现仍需较大的常数（尤其是 t‑LUOF 的矩阵块大小 β 需要满足 β = Ω(q^t)）；
- 对于非有限域或更宽松的电路模型（如可变 fan‑in）尚未给出等价结果；
- 虽然失败概率可降至 2^{-Ω(n)}，但在安全应用中仍需对随机性和随机种子进行严格管理；
- 列表解码分析依赖于 local‑similarity 理论，若需求更高的容错或更大列表尺寸，则需进一步改进。

---

## 268. DataEvolver: Automatic Data Preparation for Large Language Models through Multi-Level Self-Evolving

**arXiv ID:** 2606.07001 | [PDF](https://arxiv.org/pdf/2606.07001v1)

**作者:** Chao Deng `[一作]` (Renmin University of China), Xiaoyong Du `[通讯]` (Renmin University of China)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 DataEvolver，一种自演化的数据准备系统，能够自动构建并优化从原始数据到高质量训练数据的完整流水线。

**💡 创新点**

创新点在于将操作符层面和流水线层面自演化相结合，实现逻辑可执行性与数据质量的双重保证，并通过种子示例驱动的数据理解与自适应操作符合成。

**🔧 技术方法**

采用基于 DAG 的逻辑计划生成、操作符自演化与补丁修复、流水线自演化的反馈循环、LLM 评估判别器以及经验记忆机制；同时利用 LLM 进行代码实例化与质量判别。

**📊 数据集**

在七个公开基准上进行实验，涵盖指令跟随（Alpaca、AlpacaEval 2.0）、多项选择 QA（ARC‑Easy/Challenge）、数学推理（GSM8K、MATH）以及 Text‑to‑SQL（Spider、BIRD）等任务。

**📈 对比分析**

与原始数据的 Vanilla SFT 以及现有最佳数据准备系统 DataFlow-SFT 对比，DataEvolver 在所有任务上平均提升约 10% 的 LLM 下游性能，并相较 DataFlow-SFT 提升约 2%；同时在 token 消耗上降低约 40%。

**⚠️ 局限性**

主要局限在于目前仅针对文本模态，尚未扩展到图像等多模态数据；此外依赖 LLM 评估的计算成本与可解释性仍需进一步优化。

---

## 269. Contrastive Training with LLM-generated Near-Misses for Robust Code-Switching Speech Recognition

**arXiv ID:** 2606.06985 | [PDF](https://arxiv.org/pdf/2606.06985v1)

**作者:** Tung X. Nguyen `[一作]` (VinUniversity), Dung D. Le `[通讯]` (VinUniversity)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于点位兴趣（POI）的对比学习框架，用于提高代码混合语音识别模型在混合区的准确性。

**💡 创新点**

创新点在于构建POI-aware near‑miss（近似错误）生成管线 CS‑NMG，并通过三重门控（声学、音素、文本）精细筛选负样本，结合权重交叉熵与 InfoNCE 对比损失，显著聚焦于错误集中区。

**🔧 技术方法**

技术包括 Whisper‑small + LoRA 微调、POI 加权交叉熵、InfoNCE 对比排名、LLM（Gemini 2.5 Pro）生成近似错误，以及三重门控过滤策略。

**📊 数据集**

使用 CS‑FLEURS（中文‑英文）和 ViMedCSS（越南语‑英文）两大代码混合语音数据集。

**📈 对比分析**

与标准 CE/WCE、MWER 等基线比较，实验显示在两数据集上均实现了 WER 与 POI‑Error‑Rate（PIER）显著下降，提升约 1–2%。

**⚠️ 局限性**

局限性包括：依赖外部 LLM 与 prompt，导致离线生成成本高；仅在两语言对与单一模型上验证，缺乏更广泛的跨语言与跨模型验证。

---

## 270. Auto-Relate: A Unified Approach to Discovering Reliable Functional Relationships Leveraging Statistical Tests

**arXiv ID:** 2606.07060 | [PDF](https://arxiv.org/pdf/2606.07060v1)

**作者:** Ziyan Han `[一作]` (Shenzhen University), Jianbin Qin `[通讯]` (Microsoft Research)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种统一的功能关系（FR）挖掘框架 Auto‑Relate，能够在电子表格和关系表中自动发现并验证可可靠使用的算术、字符串和函数依赖关系。

**💡 创新点**

创新点在于：① 将三类常见关系统一为功能关系并引入四项可靠性标准（准确性、原子性、稳定性、完整性）；② 采用 mine‑then‑verify 结构，使用最小化、扰动与独立性三种统计测试对候选关系进行可靠性验证；③ 设计了三种优化策略（分组下界、AR 公式闭式加速、二项式早停）显著提升效率。

**🔧 技术方法**

主要技术包括候选枚举与准确性过滤、最小化测试、扰动测试（基于随机行互换与 Wilson 置信区间的采样）、独立性测试（卡方检验）以及上述三种加速方法；实现采用并行化和增量剪枝。

**📊 数据集**

在 58,679 张来自真实 Excel、Google Sheets、Jupyter Notebook 和关系数据库的表上构建的大规模基准（Real 套件）中，收集了 6,414 条真实 FR（覆盖 AR、ST、FD 三种类型），同时也对 RWD 手工标注数据集进行评估。

**📈 对比分析**

与 18 个基线（包括信息熵、互信息、依赖度量、GPT‑5 语言模型等）对比，Auto‑Relate 在清洗和噪声环境下的 PR‑AUC 均高出 59%（平均 0.87），在 AR、ST、FD 三种类型中均名列前茅，且在所有设置下保持高精度。

**⚠️ 局限性**

局限性包括：对极小样本或极低值域的表格可能因统计检验功效不足导致漏检；候选集规模大时仍需一定时间；目前仅支持单输出、单列或多列的算术/字符串/函数依赖关系，无法覆盖更复杂的多输出或多步转换。

---

## 271. TRACE: Trajectory Reasoning through Adaptive Cross-Step Evidence Aggregation for LLM Agents

**arXiv ID:** 2606.07054 | [PDF](https://arxiv.org/pdf/2606.07054v1)

**作者:** Vijitha Mittapalli `[一作]` (University of Massachusetts at Amherst), Nesreen K. Ahmed `[通讯]` (Cisco)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

TRACE框架用于监测LLM代理的长期轨迹，以检测隐藏的破坏行为。

**💡 创新点**

通过Triage–Inspect–Judge循环实现自适应窗口选择、跨步证据聚合，保持持续证据状态，区别于单次或固定窗口监测。

**🔧 技术方法**

采用训练无关的推理循环，利用LLM（如Claude Sonnet 4.6等）执行轻量化分层推理、证据累积与阈值评分。

**📊 数据集**

使用SHADE‑Arena基准，包含10个任务域、463条轨迹（189善意、274恶意）。

**📈 对比分析**

与全轨迹监测和序列监测基线对比，采用精确率、召回率和F1指标；TRACE在召回率和F1上显著提升（F1 0.713 对比 0.577/0.700），且平均调用次数减少35%。

**⚠️ 局限性**

仅在SHADE‑Arena上评估，未验证其他威胁类别；参数未针对不同模型精细调优，对“操纵”类破坏检测仍表现欠佳。

---

## 272. Hierarchical Semantic-Constrained Heterogeneous Graph for Audio-Visual Event Localization

**arXiv ID:** 2606.07033 | [PDF](https://arxiv.org/pdf/2606.07033v1)

**作者:** Zhe Yang `[一作]` (Harbin Institute of Technology), Xiaopeng Fan `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

针对开放词汇音频-视觉事件定位（OV-AVEL），提出了Hierarchical Semantic-Constrained Heterogeneous Graph (HSCHG)框架，用以同时建模视频段级与视频整体级的音频与视觉语义，并在超球面上进行层级一致性约束。

**💡 创新点**

创新点主要包括：① 在欧氏空间构建多尺度异质层级图，使用多方向时序边与双阈值门控融合实现跨模态噪声抑制；② 引入双向语义约束（段-视频级别）形成闭环一致性；③ 将多层级表示映射至双曲空间，采用含有超圆锥约束的层级蕴含损失，使文本原型与音视频表示保持层级一致性，从而提升对未见类别的泛化。

**🔧 技术方法**

技术细节包括：图神经网络（HHGN） + 多方向时序边（MDTE） + 双阈值门控（DTGM） + 双向语义约束（BSC） + 双曲投影（Lorentz模型） + 蕴含锥损失（hyperbolic entailment loss） + 交叉熵分类损失。

**📊 数据集**

使用OV-AVEBench数据集（基于VGGSound的67类事件，46类训练，21类测试），每条视频被均分为10个1秒段，标签为音视频一致的正样本或背景。

**📈 对比分析**

与CMRA、AVE、PSP、MM-Pyramid、OV-AVE等方法对比，HSCHG在总平均指标上提升至59.7（相对OV-AVE 57.8），尤其在未见类别上显著提升（Avg. 51.6 vs 49.5），同时在段级准确率与事件边界召回方面保持竞争力。

**⚠️ 局限性**

局限性包括：① 超曲空间的投影与蕴含损失需要额外超参数调优，模型训练更复杂；② 在段级定位精度上与某些基线仍有小幅差距；③ 依赖于已标注的VGGSound数据，模型在更大规模或多模态多语言环境下的迁移性能尚待验证。

---

## 273. Phonetic Error Analysis of Raw Waveform Acoustic Models

**arXiv ID:** 2606.07030 | [PDF](https://arxiv.org/pdf/2606.07030v1)

**作者:** Erfan Loweimi `[一作]` (University of Edinburgh), Peter Bell `[通讯]` (University of Edinburgh)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对TIMIT语音数据集中的原始波形声学模型进行逐类错误分布与混淆模式分析，并提出基于SincNet/Sinc2Net前端与BLSTM的最佳原始波形模型。

**💡 创新点**

① 将BPC（Broad Phonetic Class）错误分析从传统Filterbank系统推广到原始波形模型；② 引入参数化卷积前端（SincNet/Sinc2Net）与双向LSTM的组合，获得TIMIT上最佳原始波形PER；③ 系统性量化BLSTM对各BPC的提升及WSJ迁移学习的类级异同。

**🔧 技术方法**

使用原始波形CNN（SincNet、Sinc2Net）、双向LSTM、双头输出（CD/CI）、交叉熵训练、WSJ迁移学习、PER分解、混淆矩阵和BPC分类。

**📊 数据集**

TIMIT（dev/test）作为主实验集；WSJ（SI-284）用于迁移学习。

**📈 对比分析**

与已有原始波形系统及Filterbank基线进行PER对比；表格显示模型在仅TIMIT训练时Test PER 15.3%，迁移学习后为12.3%；BLSTM提升在转移依赖类（Diphthongs、Fricatives、Semi-vowels）显著；WSJ迁移学习对辅音提升约30%，对元音仅提升约10%。

**⚠️ 局限性**

原始波形模型对数据量敏感，迁移学习受语音多样性限制；对样本量极少的类（如Affricates）提升有限；混淆模式与传统Filterbank系统相似，仍受音位相似性约束；未对端到端或自监督模型进行类似分析。

---

## 274. MADE: Beyond Scoring via a Multilingual Agentic Diagnosing Engine for Fine-Grained Evaluation Insights

**arXiv ID:** 2606.07020 | [PDF](https://arxiv.org/pdf/2606.07020v1)

**作者:** Yilun Liu `[一作]` (Huawei), Daimeng Wei `[通讯]` (Huawei)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了多语言代理诊断引擎(MADE)，实现了基于大规模多语言评估数据的细粒度诊断报告生成。

**💡 创新点**

创新点包括将诊断拆分为规划、聚合分析、案例检视、多语言文化反思和报告合成的五角色协作架构，以及可复用的多语言诊断查询分类。

**🔧 技术方法**

采用多代理协作、ReAct循环、确定性诊断工具、语言反射器与证据绑定机制，基于 Gemini‑3 Flash LLM 进行推理。

**📊 数据集**

使用包含33个模型族、11个基准、26种语言、34种文化、8.66M评估记录（约5.2B文本token）以及专家编写的54条跨15语言诊断查询集。

**📈 对比分析**

与单一LLM和通用代理框架对比，MADE在自动评判中平均得分8.02（比最强基线5.45高2.57），在人类专家评估中87.9%胜率，显著提升报告质量与可解释性。

**⚠️ 局限性**

局限性包括对现有评估子基座的依赖、对多语言专家的持续投入需求、较高的计算成本与报告延迟，以及未覆盖视觉、代码等其他模态和极低资源语言。

---

## 275. The Sound of Malware: A Memory Forensics Approach for Android Malware Analysis via Audio Signals

**arXiv ID:** 2606.07005 | [PDF](https://arxiv.org/pdf/2606.07005v1)

**作者:** Silvia Lucia Sanna `[一作]` (Università degli Studi di Cagliari), Giorgio Giacinto `[通讯]` (Università degli Studi di Cagliari)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实现了一套基于内存取证的 Android 恶意软件检测框架（RAMwavDroid），将 APK 字节码和启动后内存快照直接映射为音频波形进行分类。

**💡 创新点**

创新点在于首次将动态内存取证与音频信号编码相结合，提出统一的 sonification 流程；通过比较多种音频编码、内存区域选择和深度学习模型，证明运行时内存结构信息可显著提升恶意软件识别准确率。

**🔧 技术方法**

技术包括：二进制‑>波形直接映射（WAV8/16、MP3）、音频特征提取（MFCC、Chroma、Spectral Centroid 等）、自监督音频预训练模型（Wav2Vec）提取嵌入、传统 ML（Random Forest、XGBoost、CatBoost）、CNN1D 以及多模型集成。

**📊 数据集**

使用 CICMalDroid2020 数据集（300 正样本/300 负样本）进行静态与动态实验，并在 VirusTotal 下载的 500 个恶意 APK 与 500 个 benign APK 上进行进一步验证。

**📈 对比分析**

相较于基于图像的 BlockDroid、传统静态/动态 ML 方法，动态 sonification 在最佳内存区域组合下实现 98.0% 的准确率；静态 sonification 也达 93.9% 的准确率，整体性能优于同类现有技术。

**⚠️ 局限性**

局限性包括：实验仅覆盖启动后早期内存快照，可能忽略后期行为；需要获取 root 权限或使用特定工具进行内存取证；在极度混淆或使用高级反取证技术的样本中仍可能出现误判；数据集规模有限，未对更大多样化样本进行充分评估。

---

## 276. Mission-Level Runtime Assurance Framework for Autonomous Driving

**arXiv ID:** 2606.06996 | [PDF](https://arxiv.org/pdf/2606.06996v1)

**作者:** Chieh Tsai `[一作]` (University of Arizona), Salim Hariri `[通讯]` (University of Arizona)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

该论文提出了一个基于成功者预测的使命级运行时保障框架，能够在执行前验证高层指令的使命可行性并在需要时拒绝或替换指令。

**💡 创新点**

创新点在于将平台安全与未来使命可行性联合评估，引入成功者安全监视器，并构建了针对使命级命令的故障注入基准。

**🔧 技术方法**

技术包括运行时安全过滤、成功者预测监视、联合安全与使命可行性评估、后退控制器以及学习控制器的验证。

**📊 数据集**

实验基于增强的 highway-env 仿真环境，添加了检查点、受限区域和燃料预算约束，并使用随机任务故障注入。

**📈 对比分析**

与传统平台级安全过滤器（如 ORCA、Simplex）以及当前状态监视器比较，提出的框架将任务成功率从约 0.44 提升至 0.73，并在故障检测和误拒率方面保持低开销。

**⚠️ 局限性**

局限主要是后退控制器在拥堵或复杂检查点布局下恢复能力不足，以及对长预测周期的误拒率仍需进一步降低。

---

## 277. Principles of Concept Representation in Sentence Encoders

**arXiv ID:** 2606.06994 | [PDF](https://arxiv.org/pdf/2606.06994v1)

**作者:** Isabelle Mohr `[一作]` (Idiap Research Institute), Andre Freitas `[通讯]` (Idiap Research Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究句子编码器在概念等价检索任务中的表现及其影响因素，提出ε_τ‑可组合性框架并通过实验识别四条关键原则（P1‑P4）

**💡 创新点**

①构造了ε_τ‑可组合性理论，系统阐释编码器对不同修饰子组合类型的几何匹配；②发现fine‑tuning只在重塑潜在空间而非扩展维度；③证明跨层池化在已句子fine‑tuned模型中无效；④揭示硬负样本仅提升校准而不提升检索排序；⑤提出监督与语义组合类型匹配原则，解释某些概念类别表现下降

**🔧 技术方法**

对齐与对比学习（InfoNCE+BCE硬负样本），多层跨层池化、输入自适应门控、深度监督等读取策略，利用Transformer句子编码器（MPNet等）

**📊 数据集**

WordNet与Wiktionary 3.3M同义词及定义对；新建DBpedia语义缺口基准（245测试/3,063训练）与4,000条修饰子标注NP同义对（5种修饰类型）

**📈 对比分析**

与冻结基线、不同池化、硬负样本与无硬负样本等八种对照组比较；实验表明fine‑tuned平均池化（B1）在多拆分、NP同义、DBpedia检索中均优于其他设定，跨层池化提升<1%，硬负样本显著提高ROC‑AUC但对R@K几乎无影响

**⚠️ 局限性**

实验仅在已句子预训练的MPNet等模型上进行，跨层读取在这些模型中无益；使用单一随机种子，结果可能不稳定；缺乏针对关系、模态、消极修饰子类型的专门监督，导致这些类别性能仍较差

---

## 278. Compliance-Based Sensor Placement for Force Sensing on a Sensorized Prostate Phantom

**arXiv ID:** 2606.06977 | [PDF](https://arxiv.org/pdf/2606.06977v1)

**作者:** Sizhe Tian `[一作]` (University of Lille), Jeremie Dequidt `[通讯]` (University of Lille)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种基于合规矩阵的软体仿真体稀疏传感器布置方法，结合内部气压传感和表面位移标记，实现对数字直肠检查等触诊的多模态感知。

**💡 创新点**

创新点在于将权重贪婪搜索与QR选择相结合，利用对ROI（DRE接触区）赋权并禁止内部放置，显著提升该区域的可观测性。

**🔧 技术方法**

采用有限元仿真生成合规矩阵，使用QR基因传感器选择与加权贪婪算法，结合内部气压传感和表面位移测量。

**📊 数据集**

数据集来自对4000+表面节点施加三向单位力的SOFA仿真，记录位移和气压响应。

**📈 对比分析**

与传统的全局QR选择相比，权重贪婪方案在ROI内的最小奇异值平均提升22.5%，并在后方视角下显著集中可观测性。

**⚠️ 局限性**

局限性包括仅在仿真中验证，未进行硬件实验；算法目前仅针对单一ROI，未考虑多或移动ROI，以及在不同软体几何形状下的泛化性。

---

## 279. Teaching the Way, Not the Answer: Privileged Tutoring Distillation for Multimodal Policy Optimization

**arXiv ID:** 2606.07000 | [PDF](https://arxiv.org/pdf/2606.07000v1)

**作者:** Shizhe Xiang `[一作]` (Tianjin University), Qilong Wang `[通讯]` (Tianjin University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种Privileged Tutoring Distillation Policy Optimization（PTD‑PO）框架，利用不暴露答案的结构化提示对大型视觉‑语言模型的失败推理轨迹进行稠密监督，从而提升多模态推理能力。

**💡 创新点**

创新点在于：①通过在教师模型上引入空间关注和文本推理提示，生成“受特权的提示”而非完整解答；②在学生与教师的上下文不匹配时采用Top‑K Jensen‑Shannon Divergence（JSD）并补偿尾部概率，兼顾稳定性和内存效率；③仅对失败轨迹激活监督，保持探索性。

**🔧 技术方法**

技术核心包括：强化学习与可验证奖励（RLVR）结合的GRPO优化；自我教师蒸馏；提示工程生成结构化受特权提示；Top‑K JSD损失与尾部补偿；以及多模态推理的生成式策略训练。

**📊 数据集**

使用ViRL39K作为训练集（包含38,870个可验证的多模态推理样本）；评估采用PAPO统一多模态推理基准，覆盖数学、几何、逻辑、视觉问答和知识密集任务。

**📈 对比分析**

与SFT、OPSD、GRPO、HDPO、PAPO等基线相比，PTD‑PO在Qwen3‑VL‑Thinking 2B/4B/8B模型上均实现了显著提升，平均整体分数从约52%提升至约71%；在各子任务上均优于所有对照方法。

**⚠️ 局限性**

局限性包括：需要先行生成高质量受特权提示，提示质量不佳会影响效果；对超大模型的训练仍需高算力；以及在所有错误轨迹均使用提示监督时可能导致过度正则化，需进一步平衡。

---

## 280. Accelerating Reproducible Research in Synthetic EHR Generation

**arXiv ID:** 2606.06990 | [PDF](https://arxiv.org/pdf/2606.06990v1)

**作者:** Jalen Jiang `[一作]` (University of Illinois Urbana Champaign), Jimeng Sun `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个统一的端到端基准框架，用于生成和评估高保真度的合成电子健康记录（EHR）。

**💡 创新点**

创新点在于：①对现有基线（MedGAN、CorGAN、PromptEHR、HALO）进行完整 ICD‑9 词表恢复与扩展；②引入轻量化 GPT‑2 作为通用基线；③提出与模型无关的隐私-实用性评估套件，并系统分析长尾代码性能缺陷；④将所有方法整合在同一数据管道中，实现可复现、可扩展的比较。

**🔧 技术方法**

使用的技术包括：PyHealth 统一框架、TensorFlow/PyTorch 代码重实现、GAN 与 Transformer 生成器、离散化的多热编码、频率引导采样、序列化距离度量、判别器基准、成员推断与最近邻攻击等。

**📊 数据集**

使用的数据集为公开的 MIMIC‑III 病例，覆盖 6,955 维 ICD‑9 诊断代码，生成 10k 份合成患者记录。

**📈 对比分析**

比较方法：在统一的长格式输出下，使用预valence R²、RMSE、判别器可辨别率（Discriminator Score）和隐私风险指标（NNAAR、MIA）对模型进行评估。结果显示：序列模型（HALO、PromptEHR、GPT‑2）在判别器可辨别率和联合分布重现方面优于平面模型；平面模型在代码频率匹配上更好但在联合分布上差距明显；GPT‑2 作为轻量级基线表现相当或超过专用模型；所有模型的隐私风险均低。

**⚠️ 局限性**

局限性包括：①仅覆盖 ICD‑9 词表且停留在 MIMIC‑III，未迁移到 ICD‑10/PHI 等新标准；②仍未覆盖多模态数据（实验室、药物、静态特征）；③高维稀疏长尾分布的处理不充分；④评估指标仍以判别器为主，缺少更细粒度的高维分布覆盖度量；⑤对模型的扩展性和易用性仍需进一步完善。

---

## 281. Menu Selection: A Computational Approach to Minimizing Food Waste

**arXiv ID:** 2606.06989 | [PDF](https://arxiv.org/pdf/2606.06989v1)

**作者:** Haris Aziz `[一作]` (University of New South Wales), Sanjukta Roy `[通讯]` (Indian Statistical Institute)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种新的集体决策问题，旨在为具有不同饮食偏好的群体选择合适的菜单，以确保每个人都有足够的食物并尽量减少浪费。

**💡 创新点**

创新点在于引入了乐观和悲观两种消费模型，并对有效菜单的特征进行了详细描述，同时设计了整数线性规划和多项式时间算法来寻找最小规模的有效菜单。

**🔧 技术方法**

使用了整数线性规划（ILP）和多项式时间算法来解决菜单选择问题，并引入了最大匹配的概念来描述消费模型。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了多种饮食偏好和菜单选项的组合。

**📈 对比分析**

通过与现有算法的比较，证明了在某些特定情况下可以在多项式时间内找到最小规模的有效菜单，而在一般情况下，寻找最小规模的有效菜单是NP完全的。

**⚠️ 局限性**

限制在于对于某些实例，尤其是当每个选项的份量较大时，验证乐观和悲观有效菜单的复杂性较高，且在一般情况下，寻找最小规模的有效菜单是NP完全的。

---

## 282. CL-CLIP: CLIP-Based Continual Learning Framework with Cost-Volume Category Decoupling for Object Detection

**arXiv ID:** 2606.06978 | [PDF](https://arxiv.org/pdf/2606.06978v1)

**作者:** Zihan Liu `[一作]` (Beihang University), Baochang Zhang `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在持续对象检测中，提出CL-CLIP框架，通过CLIP图像-文本相似度成本体积实现类别解耦，并用多专家RoI头和漂移正则化提高模型在新旧类别间的保留与适应。

**💡 创新点**

创新点在于将CLIP的零射空间先验作为成本体积用于类别解耦，构建每类专属检测通道；同时加入正交损失消除跨类别重叠、残差门控RPN与漂移正则化以稳定共享特征。

**🔧 技术方法**

采用CLIP预训练、成本体积生成、残差门控RPN、多专家卷积RoI头、正交损失、EWC漂移正则化等技术。

**📊 数据集**

使用PASCAL VOC与MS‑COCO两个标准数据集进行评估。

**📈 对比分析**

与F‑ViT基线以及ABR、MMA、NSGP‑RePRE、TARL、ROSETTA等COD方法对比，CL‑CLIP在VOC上实现了大约9–10点的整体mAP提升，在COCO上保持更好的稳定性与可塑性，整体表现优于多数对比方法。

**⚠️ 局限性**

局限性包括：仅在部分VOC/COCO分割上报告多种种子方差；实验范围受限于VOC与COCO，未验证更大规模或更长时间的持续学习；多专家头导致推理FLOPs随已见类别线性增长。

---

## 283. Beyond Matching: Category-Guided Latent Intent Reasoning for Generative Retrieval in E-Commerce

**arXiv ID:** 2606.07075 | [PDF](https://arxiv.org/pdf/2606.07075v1)

**作者:** Fuwei Zhang `[一作]` (Beihang University), Fuzhen Zhuang `[通讯]` (Beihang University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Category-guided Latent Intent Reasoning（CLIR）框架，用于在电商检索中先在连续潜在空间中进行粗细层级的意图推理，再生成语义ID（SID）进行检索。

**💡 创新点**

创新点在于：①利用商品分类层级作为隐式“思考”引导的连续潜在状态，避免显式CoT带来的生成开销；②通过Hierarchical Semantic Reasoning与Query-wise Reasoning Enhancement两种监督任务，强化潜在状态与多层级意图的对齐；③在推理后引入Reasoning-aware Constrained Decoding，使用动态前缀Trie将潜在意图映射到有效SID空间，显著降低搜索范围。

**🔧 技术方法**

技术方法包括：RQ‑VAE对商品进行语义ID量化；Transformer解码器在L个隐层步完成潜在推理；层级分类器与层级掩码实现类别对齐；多正样本InfoNCE对齐潜在向量与类别原型；动态Trie与束搜索实现受限解码。

**📊 数据集**

主要使用ESC‑I多语种电商搜索基准（US、ES、JP）作为实验数据；此外在MS‑MARCO和使用Qwen3-0.6B生成模型的跨域实验中亦验证了方法的通用性。

**📈 对比分析**

与BM25、DPR、BGE‑M3、TIGER、MERGE、CAT‑ID^2等基线相比，CLIR在ESC‑I‑US上R@100提升至36.15%（相较最强基线提升≈10%），在ES、JP也保持领先；在MS‑MARCO上可获得MRR@10/Recall@10的显著提升；在Qwen3基线下仍优于TIGER，验证了框架与生成器无关。

**⚠️ 局限性**

局限性包括：①需要手工或自动构建的商品分类层级，缺乏层级时效果会下降；②相比纯生成检索，额外的潜在推理与动态Trie构造会增加推理时延；③在超长SID或过深层级时，额外推理步骤可能导致梯度消失或性能退化。

---

## 284. TrioPose: Native Triple-Stream Diffusion Transformers for Pose-Guided Text-to-Image Generation

**arXiv ID:** 2606.07053 | [PDF](https://arxiv.org/pdf/2606.07053v1)

**作者:** Dian Gu `[一作]` (Institute of Automation, Chinese Academy of Sciences), Zhengyi Yang `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了TrioPose，一套原生姿态驱动的文本到图像生成框架，专注于多人体场景的姿态控制与形态保真。

**💡 创新点**

创新点在于：①三流姿态感知DiT（TSPA-DiT），将姿态作为独立模态并采用零初始化双残差注入；②可学习的关系偏置掩码，细粒度把握实例间拓扑关系；③姿态引导空间损失权重，利用热图误差动态加权解噪任务，提升姿态精度。

**🔧 技术方法**

采用了SD3.5M多模态Diffusion Transformer、低秩适配（LoRA）、Rectified Flow、双残差零初始化注入、可学习关系偏置掩码和热图误差加权的空间损失。

**📊 数据集**

在Human‑Art、CrowdPose、OCHuman三大公开数据集上进行评估，并构建了包含15.5k图像的混合训练集。

**📈 对比分析**

与ControlNet、T2I‑Adapter、HumanSD、Stable‑Pose、GRPose等SOTA方法对比，TrioPose在Human‑Art上AP达到64.33（比前沿提升30%）、FID降至1.65；在CrowdPose和OCHuman分别实现AP 58.56/62.59，整体显著提升姿态精度与图像质量。

**⚠️ 局限性**

局限性包括：计算成本高、仅支持单帧生成，缺乏时序一致性；对模型鲁棒性和公平性仍有待提升，且存在潜在深度伪造滥用风险。

---

## 285. Hierarchical Forecast Reconciliation for Urban Rail Transit Demand Prediction under Operational Disruptions

**arXiv ID:** 2606.07044 | [PDF](https://arxiv.org/pdf/2606.07044v1)

**作者:** Dang Viet Anh Nguyen `[一作]` (Technical University of Denmark), Filipe Rodrigues `[通讯]` (Technical University of Denmark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了将站点级和OD级乘客需求预测进行层级调和的框架，提出了可保证结构一致性的神经网络层级调和器。

**💡 创新点**

首次针对城市轨道交通系统构建层级调和器，并证明在运营中断时能显著提升预测准确性与一致性。

**🔧 技术方法**

使用SDT‑GRU和mGraphSAGE作为基模型，构建Fully Connected Reconciler（FCR）进行非线性调和，并与OLS、WLS、MinT等经典线性方法进行比较。

**📊 数据集**

采用2017年哥本哈根S‑train 12站的Rejsekort智能卡数据进行实验。

**📈 对比分析**

在一阶和多阶预测以及多种干扰场景下，FCR平均降低OD误差4%–17%，在严重中断时优于传统线性方法。

**⚠️ 局限性**

实验仅局限于单一小型网络，缺乏不确定性量化，且网络规模扩大时FCR参数量高，需进一步提升可扩展性和概率预测能力。

---

## 286. Beyond Rubrics: Exploration-Guided Evaluation Skills for Reward Modeling

**arXiv ID:** 2606.07040 | [PDF](https://arxiv.org/pdf/2606.07040v1)

**作者:** Xing Yue `[一作]` (Zhejiang University), Weiming Lu `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于可重用评估技能的奖励建模方法，通过离线探索生成技能并直接注入评估者上下文

**💡 创新点**

将奖励建模视为可复用的域级技能而非在线rubric生成，采用两阶段（流程+原则）探索+选择的技能合成方法

**🔧 技术方法**

利用大语言模型生成技能、离线演练、工作流生成、原则生成、技能合并、探索-选择、两阶段生成

**📊 数据集**

RewardBench 2、RewardBench、RM‑Bench、JudgeBench、HealthBench 等多域奖励建模基准

**📈 对比分析**

与 vanilla one‑step、rubric‑based、post‑trained RM 等基线相比，在 RewardBench 2 上提升约 10%+（如 Qwen3‑8B +13.44%），在 RewardBench 上提升约 5%，在 RM‑Bench、JudgeBench 亦有显著改进

**⚠️ 局限性**

未覆盖 pointwise RM、RL 下游效果尚未验证、对混合域和小模型性能有限，技能可能对模型偏好有依赖

---

## 287. Never Seen Before: Benchmarking Genuine Zero-Shot Composed Image Retrieval with Consistent Video-Sourced Datasets

**arXiv ID:** 2606.07032 | [PDF](https://arxiv.org/pdf/2606.07032v1)

**作者:** Zhenyu Yang `[一作]`, Changsheng Xu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 ZeroSight 这一零样本组合图像检索基准，并设计了 LLM 辅助的数据构造流程和 SC4CIR 这一无训练的多模态一致性检索方法。

**💡 创新点**

创新点在于使用后 2022 年的视频帧保证视觉与语义一致，并通过多模态大语言模型实现无训练的对称一致性检索，从而显著提升检索准确性并校正先前数据集的偏差。

**🔧 技术方法**

采用了多阶段 LLM/MLLM 辅助生成、CLIP 视觉‑文本特征、ViT 与 CLIP 视觉相似度过滤、GPT‑4o 生成逆向描述以及对称一致性重排序。

**📊 数据集**

使用来自 YT‑Temporal‑1B 的 12,048 条 2022 年后的视频，构建 197,313 张候选图像和 54,740 个查询，每个查询平均 5.16 正样本和 10.89 负样本。

**📈 对比分析**

在 ZeroSight 上对 27 种 ZS‑CIR 与 CIR 方法进行基准评测，发现传统 CLIP‑基准表现被夸大，SC4CIR 与 SC4CIR‑集成可提升 mAP 与 PNR‑mAP 5‑10‑25‑50 5‑7% 以上，训练后效果更显著。

**⚠️ 局限性**

局限性包括依赖大型 MLLM 与昂贵算力、检索时延增加、仅基于视频帧的场景可能无法覆盖所有实际应用的多样性。

---

## 288. GuideCAD: A Lightweight Multimodal Framework for 3D CAD Model Generation via Prefix Embedding

**arXiv ID:** 2606.07024 | [PDF](https://arxiv.org/pdf/2606.07024v1)

**作者:** Minseong Kim `[一作]` (Convergence Research Center for Insect Vectors), Jibum Kim `[通讯]` (Incheon National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出GuideCAD框架，通过映射网络将图像嵌入转化为前缀嵌入，并结合文本提示生成3D CAD模型；

**💡 创新点**

创新点在于只训练少量前缀嵌入参数，冻结大型预训练模型，显著降低参数量和训练成本，同时通过多模态输入提升生成质量；

**🔧 技术方法**

采用CLIP图像编码器、GPT-2语言模型、轻量级映射网络和自适应池化层；

**📊 数据集**

构建GuideCAD数据集，基于DeepCAD数据集生成文本提示、构造序列和多视角CAD图像；

**📈 对比分析**

与DeepCAD、OpenECAD及SOTA VLMs对比，GuideCAD在F1、CD、COV等指标上均优于基线，参数量约为对比模型的四分之一，训练时间减少一半；

**⚠️ 局限性**

局限包括固定模板文本表达能力受限、仅使用单张图像、依赖CLIP和GPT-2的性能限制，未来可探索更强视觉编码器和更大语言模型。

---

## 289. Task Editing for Generalizable 3D Visuomotor Policy Learning

**arXiv ID:** 2606.07012 | [PDF](https://arxiv.org/pdf/2606.07012v1)

**作者:** Jian-Jian Jiang `[一作]` (Sun Yat-sen University), Wei-Shi Zheng `[通讯]` (Sun Yat-sen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

基于单一演示数据，利用任务的场景、技能和物体三要素解耦与编辑，生成多样化的3D视觉运动策略训练样本；

**💡 创新点**

创新点在于提出Task‑Edit框架，将任务拆分为场景、技能与物体三部分，分别进行编辑；采用Real2Sim2Real流程、技能有向无环图（SDAG）重排、关键点匹配、场景占用网格及深度修复模型，显著提升演示多样性与数据质量；

**🔧 技术方法**

技术包括图像生成的3D模型重建、LangSAM/FoundationPose姿态估计、关键点匹配与变换、GPT5.5生成SDAG、占用网格与分区放置、深度修复模型与分割模型（SAM2）等；

**📊 数据集**

主要使用从单一人工演示中提取的真实机器人数据，以及通过3D资产库在仿真中生成的训练轨迹；对比基线使用DemoGen生成的数据和人类采集的完整演示集；

**📈 对比分析**

在九项真实世界任务和多种机器人平台上与DemoGen、人类演示集对比，Task‑Edit在分布内外场景均实现了显著提升（例如任务完成率提升10‑40%，长期任务的成功率超过90%），并在模拟环境和硬件实验中持续保持最优性能；

**⚠️ 局限性**

局限性包括仅支持刚体对象，难以处理非刚体（如衣物、毛巾）和关节式物体；对3D模型的尺寸与真实尺寸可能存在偏差；关键点匹配仍需人工干预，尤其是几何差异大的物体。

---

## 290. RASFT: Rollout-Adaptive Supervised Fine-Tuning for Reasoning

**arXiv ID:** 2606.07006 | [PDF](https://arxiv.org/pdf/2606.07006v1)

**作者:** Yongliang Miao `[一作]` (Chinese University of Hong Kong), Mengnan Du `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种 Rollout‑Adaptive Supervised Fine‑Tuning（RASFT）框架，利用当前模型的 on‑policy rollouts 估计每道题目的可解性，并根据可解性自适应地平衡专家演示与模型自己生成的正确轨迹；通过引入逆重要性采样比率对策略漂移进行约束。

**💡 创新点**

创新点：① 在监督式微调中引入策略感知（policy‑aware）权重，使专家示例的强度随模型当前的可解性动态调整；② 只使用通过 verifier 验证为正确的自生成轨迹作为候选，避免错误示例污染；③ 通过逆比率正则化限制过度逼近或远离预训练分布，保持先验推理能力。

**🔧 技术方法**

技术：监督式微调（cross‑entropy）+ on‑policy rollouts + verifier（答案对比或代码测试）+ 轨迹级自适应权重 + 逆重要性采样比率正则化 + 软归一化与剪切。

**📊 数据集**

数据集：训练使用 10k NuminaMath CoT 样本；数学推理评测使用 MATH‑500、Minerva Math、OlympiadBench、AIME‑2024/2025、AMC‑2023；代码推理评测使用 HumanEval 与 MBPP。实验模型包括 Qwen2.5‑Math‑1.5B/7B、Llama‑3.2‑3B、Qwen2.5‑Coder‑3B/7B。

**📈 对比分析**

对比方法：标准 SFT、DFT、ASFT、ProFiT，以及 RL 方法 GRPO、LUFFY。结果表明 RASFT 在所有模型和任务上均取得最高平均准确率，数学任务提升约 2–6%，代码任务提升约 0.4–6.7%，并在 RL 对比中保持最稳健的整体表现。

**⚠️ 局限性**

局限性：需要额外的 on‑policy rollouts 与验证，导致训练成本增加；对任务的自动验证信号（答案匹配或代码测试）高度依赖，难以直接推广至无明确正确性判定的开放式推理场景。

---

## 291. Think Fast: Estimating No-CoT Task-Completion Time Horizons of Frontier AI Models

**arXiv ID:** 2606.07157 | [PDF](https://arxiv.org/pdf/2606.07157v1)

**作者:** Dewi Gould `[一作]` (Redwood Research), Julian Stastny `[通讯]` (Redwood Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估前沿大型语言模型在不输出链式思考（CoT）文本的情况下进行推理的能力；

**💡 创新点**

首次系统量化模型的潜在隐式推理容量，并提出以人类完成时间和最小CoT令牌数为锚点的两种度量方法；

**🔧 技术方法**

采用层级自举、对数回归和经验人类解决时间校准等统计技术，对超过30,000道多领域任务进行评估；

**📊 数据集**

构建了包含数学、编程、谜题、因果推理、心理学、策略推理等多领域的综合基准集；

**📈 对比分析**

将模型的50%完成时间（TH）与人类完成时间及o3-mini最小推理令牌数进行对比，发现前沿模型的无CoT TH已达3分钟，且每年约翻倍，性能增长显著；

**⚠️ 局限性**

局限包括对无CoT行为的提示策略敏感、难以区分隐式推理与记忆或启发式匹配，以及可能低估实际推理深度等问题。

---

## 292. Explicit Evidence Grounding via Structured Inline Citation Generation

**arXiv ID:** 2606.07130 | [PDF](https://arxiv.org/pdf/2606.07130v1)

**作者:** Anar Yeginbergen `[一作]` (University of Basque Country), Rodrigo Agerri `[通讯]` (University of Basque Country)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了FullCite框架，实现在长文本问答中每条陈述后自动生成结构化的行内引用，包括文档ID和对应的逐字证据片段。

**💡 创新点**

创新点在于同时兼顾文档级引用和证据级逐字引用，并提供三种生成策略（prompt‑based、constrained decoding、posthoc alignment），从而提升证据定位与引用可信度。

**🔧 技术方法**

技术主要包括LLM推理（Qwen3-8B与Gemma-3-12B-it）、有限状态自动机约束解码、句子相似度检索（BM25、sentence‑transformer）、后处理词重叠匹配。

**📊 数据集**

使用了三大公开基准数据集：BioASQ（生物医学）、ASQA（通用事实题）和ExpertQA（32领域）。

**📈 对比分析**

与两类基线（Generate‑then‑Retrieve和ReClaim）对比，FullCite在文档级F1高、证据级Snippet‑F1大幅提升（如ASQA从12.8%提升到61.9%），同时保持或提升语义相似度；总体在多任务上表现最均衡。

**⚠️ 局限性**

局限包括：仅验证三类数据集，模型规模有限；证据定位仍受“先行偏差”与“遗漏引用”影响；对二元答案的引用缺乏可靠机制；评估指标主要为表面匹配，缺乏深层推理验证。

---

## 293. DyCon: Dynamic Reasoning Control via Evolving Difficulty Modeling

**arXiv ID:** 2606.07108 | [PDF](https://arxiv.org/pdf/2606.07108v1)

**作者:** Tengyao Tu `[一作]` (Harbin Institute of Technology), Min Zhang `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练无关的动态推理控制框架 DyCon，利用大规模推理模型（LRM）在每个推理步骤中的隐藏表示来估计问题难度，并根据估计的难度动态抑制反思关键词的概率，从而减少冗余推理步骤。

**💡 创新点**

创新点在于：①首次发现并利用 LRM 隐藏层在推理过程中对难度的线性编码；②通过轻量级线性回归从步骤嵌入直接预测难度；③在推理时按难度动态调节反思词的 logits，实现连续的、细粒度的推理深度控制；④该方法不需要额外训练，具有良好的跨模型、跨任务通用性。

**🔧 技术方法**

核心技术包括：LRM 的步嵌入提取、剩余长度到难度的对数归一化映射、岭回归难度估计器、基于难度的 logits 负偏差（logit bias）调节；实验中使用了多种 LRM（4B–32B）与推理任务。

**📊 数据集**

实验使用了数学推理（Math‑500、AIME2024/2025、AMC23、GSM8K、Olympiad Bench、MMLU_algebra）、科学推理（GPQA‑Diamond）、代码推理（LiveCodeBench）、隐式推理（StrategyQA）、常识推理（CommonSenseQA）以及知识密集型问答（TriviaQA）等共 12 个基准。

**📈 对比分析**

与现有的 TrimR、FlashThink、静态难度估计、熵基控制等方法相比，DyCon 在 4B–32B 模型上在数学、通用问答和代码任务上平均可减少 30–50% 的 token 消耗，且保持甚至提升准确率（如 Math‑500 Pass@1 提升至 92% 并减少 19% token，AIME2025 准确率提升 8.6%）。

**⚠️ 局限性**

局限性包括：①难度回归器依赖于对剩余长度的代理难度，可能对超出训练域的复杂度变化不够敏感；②对 OOD 或攻击性输入的鲁棒性有限，可能导致过早停止或过度推理；③需要在每个模型上单独拟合回归器，虽然只需 600 样本，但仍需额外前处理步骤；④对难度阈值和 logit 负偏差的超参数敏感。

---

## 294. Coarse-to-Control: Action-Token Planning for Vision-Language-Action Models

**arXiv ID:** 2606.07107 | [PDF](https://arxiv.org/pdf/2606.07107v1)

**作者:** Jinhao Wu `[一作]` (Nanjing University), Yu-Gang Jiang `[通讯]` (Fudan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Coarse-to-Control 框架，先生成粗粒度动作计划再生成可执行动作，实现 Vision‑Language‑Action 模型中的动作令牌链式思考。

**💡 创新点**

创新点在于通过共享的离散动作词表将规划与执行统一到同一动作空间中，令牌化的计划与执行保持紧密关联，直接在动作层面实现推理。

**🔧 技术方法**

使用残差 VQ 令牌器训练联合计划‑执行令牌化，配合自动回归 VLA 与教师强迫学习，基于视觉、语言与本体状态的多模态上下文进行推理。

**📊 数据集**

实验基准包括 LIBERO、SimplerEnv‑WidowX 以及四个真实世界的操纵任务（放菜、按钮、搬移、清理）。

**📈 对比分析**

与无 CoT、文本 CoT、视觉 CoT 以及 Action CoT 等方法比较，Coarse‑to‑Control 在 LIBERO 上平均 97.9% 成功率，在 SimplerEnv 上 83.3% 成功率，真实任务平均 62.5%，均优于对照组。

**⚠️ 局限性**

局限在于仅实现了单一动作令牌链式思考方案，缺乏更丰富或自适应的动作空间推理机制，且联合令牌化的结构统一仍需进一步探索。

---

## 295. CANote: Empowering Fact-checking Note Writing Through Scaffolded and Provenance-based Human-AI Collaboration

**arXiv ID:** 2606.07101 | [PDF](https://arxiv.org/pdf/2606.07101v1)

**作者:** Shuning Zhang `[一作]` (Tsinghua University), Hewu Li `[通讯]` (Tsinghua University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出CANote，一种四阶段人机协同的 AI 辅助事实核查笔记写作系统；

**💡 创新点**

通过可追溯的证据关联和结构化协同写作，实现从子主张拆分到证据检索再到草稿生成的可视化流程；

**🔧 技术方法**

结合检索增强的 LLM（gemini‑3.1‑pro‑search、gpt‑5.1‑search）、句子相似度（SentenceBERT）、立场判定模型（claim_stance）和结构化生成；

**📊 数据集**

使用2026年3月在 X（推特）平台收集的实时英文推文数据，并与官方 Community Notes 公开 API 交互；

**📈 对比分析**

与手工写作基线及现有 AI 自动写作系统做对照，结果显示 CANote 的笔记质量显著高于基线（F2≈59.9,p<.001），且与 AI 自动化质量相当；写作时间与基线相近，认知负荷不升高，但用户满意度、易用性提升；

**⚠️ 局限性**

局限包括仅在 X 平台及英文环境评估、未覆盖多语言与跨平台可迁移性、未使用官方排序算法评估、缺乏正式的证据可信度检测与可检测性判定。

---

## 296. Porting Declarative UI to HarmonyOS: A Heuristic-guided LLM Approach

**arXiv ID:** 2606.07085 | [PDF](https://arxiv.org/pdf/2606.07085v1)

**作者:** Kunwu Zheng `[一作]` (Shandong University), Chengyi Wang `[通讯]` (Shandong University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ArkTrans，一种基于结构骨架和规则后处理的 LLM 辅助方法，实现从 Android Kotlin Jetpack Compose 与 iOS SwiftUI 的声明式 UI 迁移到 HarmonyOS ArkUI。

**💡 创新点**

创新点包括：① 通过元数据提取与组件映射构造 ArkUI 骨架，引导 LLM 生成正确的语法；② 设计一套经验性后处理规则（常量内联、词法修正、布局属性展开、结构完整性验证）显著降低语法错误；③ 构建首个 100 个文件级别的 KJC/SwiftUI‑ArkUI 并行基准。

**🔧 技术方法**

技术手段：AST 解析 + UI 树构造、跨 PL 组件映射、LLM（如 GPT‑5.2、DeepSeek‑V3.2 等）的一轮 Prompt + 示例学习、以及基于正则与语义匹配的后处理引擎。

**📊 数据集**

使用 100 份手工编写且经过视觉一致性验证的 KJC 与 SwiftUI UI 代码对，覆盖八类应用场景（电商、社交、智能家居等）。

**📈 对比分析**

与直接 Prompt 与 one‑shot Prompt 作为基线对比，ArkTrans 在代码可编译率（CSR）上从 0% 提升至 53.33%‑90.67%；在全局/局部视觉相似度（CH/CLIP/Pos/Size/Color/Text）上均实现显著提升，尤其是文本完整度（高达 82.5%）和布局位置精度。

**⚠️ 局限性**

局限性：仅针对 KJC 与 SwiftUI 两种声明式 UI 框架；后处理规则基于经验，可能不覆盖所有复杂语法；对 LLM 的性能高度依赖，模型参数与训练语料差异可能影响迁移效果。

---

## 297. Native3D: End-to-End 3D Scene Generation via Unified Mesh-Texture Modeling and Semantic Alignment

**arXiv ID:** 2606.07117 | [PDF](https://arxiv.org/pdf/2606.07117v1)

**作者:** Yibo Liu `[一作]` (Kuaishou GameMind Lab), Gan Qi `[通讯]` (Kuaishou GameMind Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种端到端的3D室内场景生成与编辑框架 Native3D，直接以网格与纹理为输入，完全避免了传统2D投影导致的域差。

**💡 创新点**

创新点包括：①统一的 mesh‑texture 联合表征，能够同时捕捉全局几何布局与局部纹理细节；②3D REPA 损失，采用改进的 InfoNCE 对齐多尺度语义特征，显著提升几何与纹理保真度；③彻底消除了 2D‑>3D 交叉域信息损失。

**🔧 技术方法**

技术栈包括：Transformer‑based 场景编码器（DiT）与 Flow Matching 调度器；HunYuan3dShapeVAE 预训练编码器；Frozen Direct3D 编码器提取 3D 特征；自然语言条件控制（UMT5）与对比损失；以及自定义解码器恢复网格与纹理。

**📊 数据集**

数据集为 9,001 对 3D‑FRONT 房间（含卧室、餐厅、图书馆等）与编辑指令（通过 Qwen‑VL‑2.5 生成），覆盖增删改移与风格迁移四类操作。

**📈 对比分析**

与 Text2tex、SceneTex、RoomTex、RoomPainter 等基线在 CS、AS、BQ、IS、FID 等质量指标以及 ImgEdit‑Bench 的 Add/Remove/Move/Style 任务上进行对比；Native3D 在 CS 及 Add/Remove/Move 任务上取得最高分，整体表现超越所有基线。

**⚠️ 局限性**

局限性：在 FID、AS、BQ 上仍不及部分纹理专用方法；缺乏光照、视角和摄像机参数的支持，导致照片级真实性和跨域泛化能力有限。

---

## 298. Geodesics of Dynamic Graphs for Regime Change Detection

**arXiv ID:** 2606.07151 | [PDF](https://arxiv.org/pdf/2606.07151v1)

**作者:** William Cappelletti `[一作]` (EPFL), Pascal Frossard `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种在图空间中利用测地线（geodesic）来定义和检测网络的连续演化模式（regimes）及其变化点的框架；

**💡 创新点**

将连续演化视作图空间中的测地线轨迹，并用残差平方和（RSS）回归量化偏离，实现基于几何的连续变化点检测，首次将测地线回归与离散图序列结合；

**🔧 技术方法**

使用了多种图距离（Frobenius、L1,1、Bures‑Wasserstein）、图回归与RSS成本、离散采样策略，配合传统离线CPD算法（Binary Segmentation、PELT）以及图表示方法（图不变量、原型相似度、LAD、iso‑mirror）；

**📊 数据集**

在合成SBM网络序列（不同节点数、平均度）和英国COVID‑19 期间129个NUTS3地区的日移动网络上进行实验；

**📈 对比分析**

与多种基线方法（图不变量、原型相似度、LAD、iso‑mirror等）在合成数据上通过Rand Index比较，graph‑RSS在多数场景下优于基线；在COVID‑19数据上，Bures‑Wasserstein基准的变更点更贴近官方事件，整体表现更好；

**⚠️ 局限性**

主要局限包括高计算成本（尤其是BW距离和插值导致O(N³T²)复杂度）、仅适用于节点对齐的图、未处理节点集变化、缺乏在线检测方案以及对大规模图的可扩展性不足。

---

## 299. From Privacy to Workflow Integrity: Communication-Graph Metadata in Autonomous Agent Interoperability

**arXiv ID:** 2606.07150 | [PDF](https://arxiv.org/pdf/2606.07150v1)

**作者:** Bijaya Dangol `[一作]` `[通讯]`, Bijaya Dangol

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并验证了代理互操作通信图泄露的威胁模型，进一步给出了基于传输层和引导层的隐私属性框架；

**💡 创新点**

创新点在于将通信图泄露与工作流完整性风险关联，首次提出不可链接、无中央观察者、可否认、元数据最小化与发现隐私五大属性，并展示其在代理协议中的必要性；

**🔧 技术方法**

主要技术包括匿名通信体系（SimpleX/SMP、Tor、混合网络）与A2A协议的自定义绑定、生成式工作流模拟模型以及基于随机森林的任务类别分类器；

**📊 数据集**

数据集由一份真实的A2A协议捕获日志（数十条任务交互）与以此为基准生成的8类工作流合成数据构成；

**📈 对比分析**

通过准确率与捕获比率（capture ratio）对比不同隐私属性组合，实验结果表明单一属性效果有限，只有同时满足不可链接与元数据最小化时，分类准确率可降至基线并显著降低攻击者的决策价值；

**⚠️ 局限性**

局限在于实验基于模拟工作流，未覆盖大规模真实流量；混合网络方案虽提供强隐私但伴随高延迟与带宽开销，且未在真实部署环境中验证前置攻击的可行性。

---

## 300. Efficient $(α,β)$-core Computation and On-the-fly Query at Billion Scale with GPUs

**arXiv ID:** 2606.07148 | [PDF](https://arxiv.org/pdf/2606.07148v1)

**作者:** Qingshuai Feng `[一作]` (Great Bay University), Long Yuan `[通讯]` (Wuhan University of Technology)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对二分图的 (α,β)-核心计算与实时更新查询，提出了 GPU 端无索引的剥离算法、基于核心数的预剪枝以及连通分量感知的 on‑the‑fly 查询方案。

**💡 创新点**

创新点在于：①使用 warp‑centric 并行剥离，完全避免昂贵的全局索引；②通过预先计算的顶点核心数实现早期剪枝，大幅减少冗余计算；③在动态更新中仅在受影响的连通分量内局部剥离，显著降低查询延迟。

**🔧 技术方法**

核心技术包括：GPU 并行计算、CSR 存储格式、原子操作与 warp‑级队列、核心数预处理、并行连通分量维护。

**📊 数据集**

实验使用 11 个真实二分图（来自 KONECT，包括 10 个真实数据集，1 个合成数据集）及其边数在 6×10^7 到 1×10^9 之间。

**📈 对比分析**

与传统 CPU 剥离、三层索引和 bi‑core 索引等基线相比，GPU 算法在预处理时间、查询时间和内存占用上均显著优越；查询时间可低至毫秒级，速度提升 10‑1000 倍；内存峰值则从 500 GB 降至 15 GB。

**⚠️ 局限性**

局限性：①仍需满足 GPU 的显存约束，无法一次处理超过显存容量的极大图；②在最坏情况下 on‑the‑fly 查询仍可能退化为全图剥离（O(m)）；③未探索多 GPU 或分布式扩展，难以进一步突破规模限制。

---

## 301. Consistent-Inversion: Reverse Consistency Guidance for Structure-Preserving Visual Editing

**arXiv ID:** 2606.07145 | [PDF](https://arxiv.org/pdf/2606.07145v1)

**作者:** Xiaocheng Lu `[一作]` (Hong Kong University of Science and Technology), Song Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `da1b1a89-583a-4b57-9c81-478778569bec` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Consistent-Inversion，一种训练自由的反向一致性引导框架，用于在文本引导扩散模型中进行结构保持的图像编辑。

**💡 创新点**

通过构造目标侧噪声表示并在源提示下进行反向去噪，计算反向一致性差异作为修正信号，只在少数早期时间步补偿，从而解决源逆轨迹与目标编辑轨迹的不匹配问题。

**🔧 技术方法**

使用DDIM反向采样、扩散模型条件预测、无训练逆向编辑、反向一致性校正、选择性时间步校正，并与Prompt‑to‑Prompt、Plug‑and‑Play等已有编辑器兼容。

**📊 数据集**

在PIE‑Bench（700张图、10类编辑类型）上评估，并在经典Stable Diffusion v1.4协议下验证兼容性。

**📈 对比分析**

与Euler Flow Inversion、Negative‑Prompt Flow Inversion、Direct Inversion等在统一SD3.5协议下比较，Consistent‑Inversion在BG‑LPIPS、DINO、LPIPS、MSE、PSNR、SSIM等结构/背景保持指标上显著提升，CLIP对齐指标保持不变；在经典协议下同样优于基线；运行时间略增但低于每图优化方法。

**⚠️ 局限性**

适用于布局不变的编辑，过度修正可能抑制目标改动；对视角变化、对象数量变更等场景效果有限；需手工设定时间步和修正幅度，密集校正会增加开销。

---

## 302. Explaining Unsupervised Disease Staging in Huntington's Disease: Insights into Model Representations and Clusters

**arXiv ID:** 2606.07135 | [PDF](https://arxiv.org/pdf/2606.07135v1)

**作者:** Lubna Mahmoud Abu Zohair `[一作]` (Heriot-Watt University), Hind Zantout `[通讯]` (Heriot-Watt University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对已提出的 URL-STFN 无监督疾病分期框架进行扩展，在嵌入空间、聚类结构和转移过程上加入解释性分析。

**💡 创新点**

首次将梯度显著性图、SHAP 等可解释性技术与图卷积网络+GRU 结合，用以揭示特征对嵌入、聚类及阶段转移的具体贡献，从而提升模型透明度与临床可解释性。

**🔧 技术方法**

技术手段包括：图卷积网络+门控循环单元用于序列嵌入；UMAP 可视化嵌入空间；梯度显著性图（Saliency）捕捉时间维度特征重要性；随机森林 surrogate + SHAP 评估特征对聚类和阶段转移的影响。

**📊 数据集**

使用 Enroll‑HD 数据集，筛选 302 名患者（至少 4 次连续年度访视），包含 44 项临床特征（运动、认知、功能等）及已收集的 TFC、DCL 等传统分期指标。

**📈 对比分析**

通过聚类指标（Silhouette、Davies–Bouldin、Calinski–Harabasz）得到 4 个稳健阶段，并与传统阈值分期方法对比，表现出更高的分离度与内部一致性，且嵌入与临床评分的对应关系更为一致。

**⚠️ 局限性**

局限性：样本年龄集中在 54–57 岁，缺乏更广泛人群验证；缺失值仅编码为 1，未系统评估不同缺失处理方案；缺乏外部数据集或临床试验的验证，需进一步评估泛化能力。

---

## 303. $α$-PFN: Fast Entropy Search via In-Context Learning

**arXiv ID:** 2606.07134 | [PDF](https://arxiv.org/pdf/2606.07134v1)

**作者:** Herilalaina Rakotoarison `[一作]` (University of Helsinki), Eytan Bakshy `[通讯]` (Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用 Prior‑Data Fitted Networks（PFN）实现两阶段训练，生成 α‑PFN 能在一次前向传播中近似预测信息理论的 Entropy Search（PES、MES、JES）收敛信息增益，从而实现贝叶斯优化的采样策略。

**💡 创新点**

创新点在于：①将传统需要 Monte‑Carlo 采样的 ES 换成基于 PFN 的可学习近似；②采用两阶段训练，先让基准 PFN 条件化最优信息，再让 α‑PFN 直接预测信息增益；③一次前向即可得到采样函数，显著降低运行时开销；④支持全贝叶斯 GP 超参数不需要额外采样。

**🔧 技术方法**

技术方法包括 Transformer‑based PFN（TabPFN v2）、随机傅里叶特征（RFF）生成 GP 先验样本、信息增益熵计算、全贝叶斯 GP、域迁移适应（模拟 BO 轨迹）、在 GPU 上进行大规模训练（≈13h+16h）。

**📊 数据集**

数据集：预生成的 GP 样本（RFF）用于训练；在合成函数（Branin、Hartmann、Ackley）以及真实 HPO 任务（LCBench、HPO‑B）上评估；还加入高噪声 OOD 设定验证鲁棒性。

**📈 对比分析**

比较方法：与 BoTorch 实现的 JES、MES‑GIBBON、PES 以及 MCMC‑ES 进行对比，EI 作为基准；评价指标为累计回报（inference regret）、平均排名和速度；结果显示 α‑PFN 在大多数任务中匹配或略优于 GP‑ES，并在速度上实现 1.6× 至 72× 的加速（HPO‑B 上常超过 30×）。

**⚠️ 局限性**

局限性：①训练仅覆盖维度 ≤6、context ≤50，迁移至更高维/更长迭代时效果衰减；②对 OOD 噪声或先验分布差异敏感；③需要针对每种先验重新训练 α‑PFN，扩展成本较高；④在极高维或大 context 的可扩展性尚待提升。

---

## 304. MalSkillBench: A Runtime-Verified Benchmark of Malicious Agent Skills

**arXiv ID:** 2606.07131 | [PDF](https://arxiv.org/pdf/2606.07131v1)

**作者:** Wenbo Guo `[一作]` (Nanyang Technological University), Yang Liu `[通讯]` (Nanyang Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了MalSkillBench基准，用于评估AI编码代理技能的恶意检测，覆盖了代码注入、提示注入及混合攻击三种向量；

**💡 创新点**

创新点在于首次提出运行时验证的闭环生成-验证-反馈流程，结合三维攻击分类，生成覆盖108个攻击格子的恶意技能，并提供公开、可复现的基准；

**🔧 技术方法**

技术实现包括在Docker沙箱内对技能执行进行系统调用与文本日志监控，使用LLM进行语义匹配与判定，并对生成的技能进行多轮反馈修正；

**📊 数据集**

数据集包含3,944个标记恶意技能（3,214由生成管道产出，703来自公开仓库，27为测试样本）以及4,000个高质量合法技能，所有样本按攻击向量、行为和注入策略三维分类；

**📈 对比分析**

评估方法为在统一配置下对12款技能专用检测器及10款迁移工具进行召回、精度、F1比较，最高召回率达98.4%，但对提示注入和Agent控制攻击召回明显下降；

**⚠️ 局限性**

局限性包括LLM判定的主观性与误判风险、生成技能与真实攻击模式可能存在偏差、以及仍未覆盖所有潜在攻击向量和最新变种的能力不足。

---

## 305. Learning Explicit Behavioral Models with Adaptive Questions and World-Model Probes

**arXiv ID:** 2606.07127 | [PDF](https://arxiv.org/pdf/2606.07127v1)

**作者:** Hikaru Shindo `[一作]` (Technical University of Darmstadt), Kristian Kersting `[通讯]` (Technical University of Darmstadt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本工作训练了一种显式符号行为模型（ESBM），通过自适应问答与主动世界模型探测，将训练过程中的失败转化为可验证的约束，从而学习可解释且可执行的行为策略。

**💡 创新点**

创新点在于将模型的“理解”任务嵌入训练循环，利用自适应问答和主动探测将缺失或错误的机制转化为局部编辑约束，并采用多准则接受规则来平衡任务得分、问答准确性和机制预测一致性。

**🔧 技术方法**

技术方法包括：基于LLM的模型编辑器、符号谓词与加权规则的决策层、可执行机制记忆（预测符号事件与状态变化）、主动世界模型探测（对齐真实与抽象环境分支）、以及局部类型化编辑与验证门控制更新。

**📊 数据集**

实验使用了多款Atari风格的JAXAtari环境（如A、B、C等），并在附录中报告了额外若干游戏的得分以提供更广泛的性能上下文。

**📈 对比分析**

与DQN、PPO、NUDGE、BlendRL及score‑only自进化符号代理EvoSymbol等基线对比，ESBM在所有主游戏中获得最高得分，并在问答回答率与机制预测准确率方面显著优于其他方法。

**⚠️ 局限性**

局限性包括：对探测生成质量与信噪比高度依赖，检查点恢复与符号状态抽象的可靠性影响探测覆盖；符号词汇与选项界面限制了可表达的复杂行为；实验仅在有限的Atari风格游戏上验证，尚未在更开放或机器人环境中进行迁移评估。

---

## 306. Beyond Linear and Overcomplete Regimes: A Mean-Field Analysis of Bottleneck Autoencoders

**arXiv ID:** 2606.07120 | [PDF](https://arxiv.org/pdf/2606.07120v1)

**作者:** Santanu Das `[一作]` (Tata Institute Of Fundamental Research), Satyaki Mukherjee `[通讯]` (National University Of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

在均值场理论框架下，对带有限维瓶颈的非线性自编码器的学习动态和风险进行了理论分析。

**💡 创新点**

首次给出非线性瓶颈自编码器的均值场极限、学习动力学以及有限宽网络与无限宽解的收敛性。

**🔧 技术方法**

利用均值场（MF）方法、Vlasov–McKean PDE、耦合向量场稳定性与Grönwall 论证。

**📊 数据集**

未使用具体数据集，理论推导基于一般分布假设。

**📈 对比分析**

无实验对比，理论证明了有限宽网络风险随时间跟随均值场风险，最优时误差为O(1/N)。

**⚠️ 局限性**

仅适用于一维瓶颈、浅层网络，未考虑多维瓶颈或更深结构，且假设参数空间紧致。

---

## 307. The Three-Ring Architecture: Governing Agents in the Era of On-Platform Organisations

**arXiv ID:** 2606.07119 | [PDF](https://arxiv.org/pdf/2606.07119v1)

**作者:** Sergio Alvarez-Telena `[一作]`, Marta Diez-Fernandez `[通讯]`

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出并验证了三环架构（Legacy Ring‑1、治理 OS Ring‑2、LLM Satellite Ring‑3）以及与之配套的治理模型、诊断框架、三层公司模型、排名转型议程、算法化培训和模块化/整体化部署路径，旨在解决企业 AI 部署失败率高的结构性治理缺失问题。

**💡 创新点**

创新点包括：① 将操作系统（OS）功能映射到企业治理层（Ring‑2），区分确定性与非确定性代理的风险；② 引入极致高效国家诊断（EEN）实现跨行业基准化评估；③ 设计三层公司模型（Tech Must / Right‑to‑Play / Right‑to‑Win）和动态排名转型议程；④ 强调算法化（Algorithmization）培训为治理文化基石；⑤ 提供模块化（right‑to‑play）与整体化（right‑to‑win）两种可并行部署路径。

**🔧 技术方法**

主要技术包括：策略型代理（M2）实现 Ring‑2，前沿 LLM 实现 Ring‑3；架构基于 deterministic complexity、federated architecture、extended production architecture（EPA）和四大 OS 功能（资源抽象、进程协调、权限执行、平台提供）。

**📊 数据集**

验证基于过去十年在金融、政府、采购和合规等行业的真实部署案例，未公开单一数据集，依托行业案例和独立机构评估。

**📈 对比分析**

方法上通过行业案例和独立评估验证架构可行性；在实测部署中，项目失败率从约95% 降至可接受水平，治理指标（合规性、可追溯性、跨部门协同）显著提升。缺乏传统实验对比数据，更多基于案例观察。

**⚠️ 局限性**

局限性：① 需要组织具备算法化文化与治理能力，部署复杂度高；② 依赖长期经验与行业案例，缺乏公开量化实验；③ 对资源有限或规模较小的企业适用性不充分；④ 部署成本与组织内部阻力仍是主要挑战。

---

## 308. DIFFRACT: Neuralized Utility Maximization for Wireless Networks by Differentiable Programming

**arXiv ID:** 2606.07114 | [PDF](https://arxiv.org/pdf/2606.07114v1)

**作者:** Chee Wei Tan `[一作]` (Nanyang Technological University), Siya Chen `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 DIFFRACT 框架，将无线功率控制的标准干扰函数映射为可微分的神经网络，并通过 LLCP 与算法展开实现对隐式干扰约束的学习与优化

**💡 创新点**

在标准干扰函数可微分性与单调算子理论基础上，首次实现端到端可微分的资源管理，将非线性功率控制问题与深度学习无缝结合

**🔧 技术方法**

可微分编程、算法展开、LLCP（线性对数凸规划）、自动微分（PyTorch/TensorFlow）以及深度前馈网络

**📊 数据集**

合成无线链路衰落数据集，覆盖 Rayleigh、Rician、Nakagami 三种衰落场景及其干扰概率和功率约束

**📈 对比分析**

与 AOPC、LDP、CR 等经典算法对比，DIFFRACT 在功率与效用精度上均达到约 0.96–0.99 的准确率，且计算时间比传统迭代算法快数十倍

**⚠️ 局限性**

对最优解的逼近仍略低于理论最优，需要先行生成足量衰落数据进行训练，扩展到更大规模网络时模型尺寸与训练成本可能显著上升

---

## 309. Style or Content? Evaluating Style Classifiers with Controlled Content Overlap

**arXiv ID:** 2606.07103 | [PDF](https://arxiv.org/pdf/2606.07103v1)

**作者:** Zhuo Liu `[一作]` (University of Rochester), Hangfeng He `[通讯]` (University of Rochester)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个可控内容重叠的风格分类基准，系统评估模型对内容捷径的依赖，并通过训练动态观察内容信息的消失。

**💡 创新点**

创新点在于：① 用信息论度量α量化内容与风格的共享程度；② 提出交叉重叠评估和跨风格内容检索探针，形成完整的诊断流程；③ 通过训练动态揭示内容信息被逐步抑制的过程。

**🔧 技术方法**

技术上采用RoBERTa‑large作为编码器、MLP分类头，并使用CLIP式对比损失的内容检索探针；同时计算高重叠优势Δ_HOA来量化跨重叠迁移效果。

**📊 数据集**

实验数据来源于七种英文圣经翻译的平行语料，选取五种作为训练/评估风格版本，另外两种留作未见风格的检索探针。

**📈 对比分析**

通过匹配与非匹配重叠的准确率矩阵、Δ_HOA指标和检索top‑1准确率进行比较；结果显示低重叠模型在内容被移除时准确率骤降，高重叠训练模型保持稳定；检索准确率随α递减，说明内容信息被逐步去除。

**⚠️ 局限性**

局限性包括：仅评估RoBERTa‑large+MLP；仅使用圣经翻译，缺少多样化风格；内容重叠以离散块为单位，未考虑语义相似性；重叠调度仅有六个粗粒度水平，可能掩盖细微转折。

---

## 310. An Adaptive Data cleaning Framework for Noisy Label Detection

**arXiv ID:** 2606.07086 | [PDF](https://arxiv.org/pdf/2606.07086v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 311. MetaConfigurator: AI-Assisted RDF Authoring from JSON Data

**arXiv ID:** 2606.07094 | [PDF](https://arxiv.org/pdf/2606.07094v1)

**作者:** Felix Neubauer `[一作]` (University of Stuttgart), Benjamin Uekermann `[通讯]` (University of Stuttgart)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

将 MetaConfigurator 转化为支持 RDF 的完整工作流工具，集成 JSON‑>JSON‑LD 的 RML 转换、RDF 编辑、SPARQL 查询与知识图谱可视化，并提供 AI 辅助生成 RML 与 SPARQL。

**💡 创新点**

创新点在于：① 将传统的 JSON Schema 编辑器扩展为一体化的语义 Web 环境；② 通过 LLM 提供的提示式 AI 生成 RML 与 SPARQL，降低非专业用户的门槛；③ 在同一界面实现数据转换、编辑、查询与可视化，避免工具切换。

**🔧 技术方法**

使用技术包括：RDF、JSON‑LD、RML、SPARQL、OWL/Ontology Explorer、COSE‑Bilkent 布局、in‑browser SPARQL 解析器、LLM API（OpenAI/Claude）以及前端框架实现的交互式可视化。

**📊 数据集**

以金属‑有机框架（MOF）合成数据集为例（Neubauer 等的实验记录），通过 JSON 结构转换为 RDF/JSON‑LD 并进行查询与可视化。

**📈 对比分析**

与单一工具相比，论文未给出量化基准；但通过示例演示，展示了整合后工作流程的连贯性与易用性，并指出在浏览器端处理大型图谱时性能会受限。整体性能满足中等规模实验数据的编辑、查询与可视化需求。

**⚠️ 局限性**

主要限制包括：AI 生成的映射和查询仍可能出现语义/结构错误，需人工审核；浏览器端处理和可视化在大规模图谱时受限；选择合适的本体与标识符模式仍是非专家面临的挑战。

---

## 312. Dreaming when Necessary: Advancing World Action Models with Adaptive Multi-Modal Reasoning

**arXiv ID:** 2606.07089 | [PDF](https://arxiv.org/pdf/2606.07089v1)

**作者:** Yinzhou Tang `[一作]` (Tsinghua University), Yong Li `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出AdaWAM，一种通过轻量级动态路由器在任务执行过程中自动切换文本推理、视觉推理和仅动作推理的自适应多模态世界行动模型；

**💡 创新点**

创新点在于将动态路由器嵌入WAM框架，实现按时空需求自动激活最合适的推理模式，既保持了多模态推理的优势，又显著降低了推理开销；

**🔧 技术方法**

采用Diffusion Transformer作为视频-动作生成骨干，VLM完成文本推理，动态路由器根据上下文预测<TR>和<VR>，并采用两阶段训练（流匹配与文本微调）实现模型协同；

**📊 数据集**

使用LIBERO、RoboTwin 2.0仿真数据集以及真实世界的AgileX ALOHA和PiPER 6-DoF机器人任务进行训练与评估；

**📈 对比分析**

与多种VLA与WAM基线（OpenVLA、π_0、X‑VLA、CoT‑VLA、Fast‑WAM等）在LIBERO‑Long、RoboTwin硬任务以及真实任务上对比，AdaWAM在性能上往往更优或相近，同时推理时间显著下降；

**⚠️ 局限性**

局限性在于仅使用RGB图像限制了对复杂几何或遮挡场景的空间推理能力，且动态路由器目前仍基于有监督的注释，缺乏强化学习或无监督自适应机制。

---

## 313. Residual-Controlled Multiplier Learning for Stochastic Constrained Decision-Making

**arXiv ID:** 2606.07088 | [PDF](https://arxiv.org/pdf/2606.07088v1)

**作者:** Kang Liu `[一作]` (Xi'an Jiaotong University), Ziyu Qu `[通讯]` (China University of Geosciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种新的残差控制乘子学习（RCML）框架，用以解决随机约束优化中乘子记忆更新容易被噪声放大的问题；

**💡 创新点**

创新点在于将投影压力拆分为用于原始下降的有效压力与用于乘子记忆的压力残差，并通过有限增益残差反馈实现乘子跟踪，从而兼顾激活、废旧记忆释放与非活跃约束的死区行为；

**🔧 技术方法**

主要技术包括投影增广拉格朗日、残差控制的乘子更新、指数移动平均过滤、协同尺度自适应以及残差‑νPI 动态校正；

**📊 数据集**

实验使用了合成线性规划、二次规划、非凸QP、能源储备分配（模拟）以及大规模公平排序（使用公开排序数据集）等数据集；

**📈 对比分析**

与SGDA（符号/正向）、投影ALM、残差-I、RCML各变体等方法比较，RCML在保持与传统方法相当的目标性能的同时，显著降低了约束违背、乘子波动（DualTV）和残差波动，提升了可靠性和可解释性；

**⚠️ 局限性**

局限性包括：需要调节多组超参数；过滤机制在追踪响应与噪声抑制之间存在权衡，可能导致快速分布变化时的临时约束违背；算法对投影非光滑点缺乏主动稳定机制，靠局部活跃集稳定性实现更优噪声下限。

---

## 314. dots.tts Technical Report

**arXiv ID:** 2606.07080 | [PDF](https://arxiv.org/pdf/2606.07080v1)

**作者:** Shi Lian `[一作]`, Kai Yu `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个2B参数的全连续自回归TTS模型，能够在不使用离散音频编码的情况下实现高质量、低延迟语音合成。

**💡 创新点**

结合语义结构化AudioVAE、全历史条件的流匹配头以及无奖励自纠正后训练，显著降低长程错误累积，并通过CFG‑aware MeanFlow蒸馏实现低NFE推理。

**🔧 技术方法**

使用自回归流匹配Transformer（AR‑FM）、AudioVAE、LLM（Qwen2.5‑1.5B）、全历史条件、CFG、MeanFlow、Self‑Corrective Alignment、以及多任务下游监督等技术。

**📊 数据集**

训练数据为1.5M小时多语种语音（中英内部、开源TTS/ASR混合、Caption对齐集），并在Seed‑TTS‑Eval、MiniMax、CV3‑Eval、EmergentTTS‑Eval等标准基准上评测。

**📈 对比分析**

与CosyVoice、VoxCPM、Seed‑TTS等公开系统对比，Zero‑shot语音克隆在Seed‑TTS‑Eval上取得2.92% WER、79.2% SIM；在MiniMax 24语种上平均SIM 83.9%；在CV3‑Eval跨语音克隆SIM超过75%；在EmergentTTS‑Eval语义与复杂语法场景中排名首位；推理时TTS首包延迟54 ms，RTF 0.245。

**⚠️ 局限性**

低资源语言文本覆盖不足导致WER偏高；缺乏音素输入、风格/指令控制；未覆盖歌唱或统一语音/音频生成；模型可能被滥用于无授权克隆。

---

## 315. ANNS-AMP: Accelerating Approximate Nearest Neighbor Search via Adaptive Mixed-Precision Computing

**arXiv ID:** 2606.07156 | [PDF](https://arxiv.org/pdf/2606.07156v1)

**作者:** Mingkai Chen `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences and University of Chinese Academy of Sciences), Huawei Li `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences and University of Chinese Academy of Sciences)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种名为 ANNS-AMP 的自适应混合精度近似最近邻搜索（ANNS）框架及其加速器，实现了对不同子空间距离计算精度的动态预测与自适应执行，显著提升了搜索吞吐量和能效。

**💡 创新点**

创新点包括：① 通过聚类结构提取子空间特征（尺度、半径、距离等）并使用 SVR 回归模型预测所需精度；② 采用位序列化计算单元和位交错内存布局，使低精度计算可以高效并行；③ 设计贪婪负载调度策略，平衡不同精度下的计算单元工作负载；④ 该预测器可在运行时复用于加速器中，几乎不增加额外延迟。

**🔧 技术方法**

核心技术包括：位序列化计算、SVR 精度预测、位交错数据布局、贪婪调度、PQ 量化索引、集群定位、LUT 构造、top‑k 排序以及硬件层面的自定义加速器设计。

**📊 数据集**

实验使用了大规模向量数据集 SIFT100M、DEEP100M（均量化为 uint8）以及用于对比的 SIFT1M、GIST1M，并在这些数据集上评估了精度与性能。

**📈 对比分析**

与 32 核 Intel Xeon CPU、NVIDIA A100 GPU 以及 ASIC 基线 ANNAx12 进行对比，ANNS-AMP 在 CPU 上平均加速 163.76×，GPU 上 10.57×，ANNAx12 上 2.06×；能耗则分别下降 1100×、39.41×、6.66×，且整体精度损失低于 2.7%。

**⚠️ 局限性**

局限性包括：① 需预先训练精度预测模型，对不同数据分布或索引结构的迁移性有限；② 主要针对 PQ‑基聚类索引，其他索引（如图索引、哈希索引）不直接适用；③ 预测模型与硬件实现需要占用一定的 SRAM 资源，规模化部署可能受限；④ 对于更高量化位宽（如 uint16）或极低精度需求时，性能提升空间有限。

---

## 316. No-Harm Physics-Informed Inverse Learning with Residual-Calibrated Uncertainty

**arXiv ID:** 2606.07153 | [PDF](https://arxiv.org/pdf/2606.07153v1)

**作者:** Ronald Katende `[一作]` `[通讯]` (Kabale University), Ronald Katende (Kabale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种物理信息逆学习的无害认证与选择框架，该框架通过残差校准的可靠性半径与基准方案进行比较，只在学习方案的证书不劣于基准时才接受该学习结果；

**💡 创新点**

创新点在于将数据残差、物理残差、边界/初始条件残差与优化残差结合，给出基于条件稳定性估计的后验误差上界和不确定性半径；提出“无害”选择原则及高概率残差验证机制，并将该框架定位为任何物理信息逆解器的可复用包装层；

**🔧 技术方法**

使用条件稳定性分析、后验误差估计、残差归一化、优化残差诊断、Hoeffding界限的高概率物理残差、PINN/神经算子等技术；

**📊 数据集**

通过人工合成的离散实验数据进行验证，包括 Poisson 源恢复、逆热方程、有限角断层成像、椭圆系数识别以及随机残差验证实验；

**📈 对比分析**

与保守基准（岭回归、有限元逆解等）对比，学习模型在证书满足条件时能够获得更小的半径且误差不劣；实验显示选择率约5.9%，在接受的案例中安全提升率接近100%，误报率极低；但在高度不稳定或物理不一致的情形下被拒绝率较高；

**⚠️ 局限性**

需要先验的条件稳定性估计和合适的残差权重；稳定常数过大会导致保守；高概率残差仅对单个候选有效，无法得到统一的网络级覆盖；不依赖真实误差，且仅在离散实验验证，缺乏大规模真实场景的测试。

---

## 317. A Data-Free Symbolic Regression Approach for Solving Equations

**arXiv ID:** 2606.07152 | [PDF](https://arxiv.org/pdf/2606.07152v1)

**作者:** Sergei Garmaev `[一作]` (EPFL), Olga Fink `[通讯]` (EPFL)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种基于可微分符号模型的方程求解器SES，能够仅凭方程及初/边界条件，通过残差最小化直接获得符号解。

**💡 创新点**

创新点在于将符号回归与残差驱动的物理信息网络相结合，构建可微分符号网络，以方程残差作为监督信号，完成符号解的直接优化。

**🔧 技术方法**

使用了可微分符号网络（EQL）+自动微分、Adam梯度优化、L1稀疏正则、剪枝策略，以及采样式的求解点生成。

**📊 数据集**

实验数据来自若干合成方程（线性代数、超越方程、常微分方程、偏微分方程），不依赖外部输入输出数据集，只使用随机采样的插值点。

**📈 对比分析**

通过将学习得到的符号表达式与已知解析解逐项比较，评估误差，实验表明SES在所有测试案例中均能恢复与解析解完全一致或近似的符号形式，性能优异。

**⚠️ 局限性**

局限性包括：受限于所选符号操作库，难以处理更复杂或高维、刚性/混沌等问题；模型训练成本高，且需要足够的符号表达能力和良好的初始条件。

---

## 318. Decision-Aware Evaluation of Physics-Informed Surrogates

**arXiv ID:** 2606.07146 | [PDF](https://arxiv.org/pdf/2606.07146v1)

**作者:** Daniel Cieślak `[一作]` (Gdańsk University of Technology), Andrzej Czyżewski `[通讯]` (Gdańsk University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个名为pinn-gym的基准，用于在多材料可打印聚合物格子能量吸收问题中评估物理信息机器学习（PIML）代理的决策性能；

**💡 创新点**

创新点在于：①将评估聚焦于代理在实际设计决策中的作用，而非仅仅是曲线预测；②引入多维度评估指标（曲线误差、物理可行性、top‑k检索精度、质量惩罚），揭示曲线误差与决策质量不必然相关；③使用无量纲化与材料向量进行跨材料学习，构建跨材料迁移矩阵；④提供公开的算子、数据集与完整实验记录。

**🔧 技术方法**

使用的技术包括：无量纲化（Buckingham‑π）、物理约束损失（能量残差、峰值力、单调性、光滑性）与普通数据损失相结合的PINN；残差全连接网络（MLP）；基于固定量化的决策评估框架；五种聚合物材料卡（PA12、PLA、PETG、TPU、PA‑CF）。

**📊 数据集**

使用的数据集为5个材料卡下每卡25,000个候选几何体的标签，标签由一个可公开实现的减压-冲击仿真算子生成，包含力-位移曲线、能量、峰值力、可行性判断。

**📈 对比分析**

对比方法：单材料专用模型、全材料联合模型、5×5跨材料迁移。性能结果显示：最低nRMSE的模型不一定能检索到最佳可行设计；物理损失能降低误差或提升安全性，但需权衡；联合模型可在多材料上取得中等曲线误差，但决策性能仍受材料影响；跨材料迁移不对称，且从柔性材料迁移至刚性材料往往失败。

**⚠️ 局限性**

局限性包括：①基准采用简化的低阶仿真算子，未取代高精度有限元或实验验证；②材料卡仅包含数值参数，未考虑打印机、后处理、批次差异；③评估指标受固定装配限制，若候选集无可行设计则精度/回报无意义；④联合模型与迁移结果仍受网络容量、损失权重等超参数影响；⑤仅关注设计选择阶段，未覆盖整个工程工作流。

---

## 319. Beyond Post-hoc Explanation: Toward Glassbox AI via Probabilistic Mediation

**arXiv ID:** 2606.07113 | [PDF](https://arxiv.org/pdf/2606.07113v1)

**作者:** Manuele Leonelli `[一作]` `[通讯]` (IE University), Manuele Leonelli (IE University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的“Glassbox Framework”，通过将贝叶斯网络（BN）作为前置的透明推理层，调和大型语言模型（LLM）与结构化决策，旨在实现高风险机构应用中的可审计、可争议、可不确定性量化的生成式AI。

**💡 创新点**

创新点包括：① 将BN作为LLM的先验推理媒介，实现结构化、可检验的推理过程；② 明确BN–LLM接口作为科学对象，强调语义对齐与概率映射的必要性；③ 将可解释性从后置说明转向前置结构化推理，提升稳定性与可争议性；④ 建立三层治理架构（治理、推理、问责），为监管与诉讼提供完整审计链。

**🔧 技术方法**

采用的技术包括：大型预训练语言模型（如GPT系列）进行文本解析；贝叶斯网络用于编码域知识、因果关系与先验分布；语义翻译接口（Semantic Interface）用于将自然语言信息映射到BN变量；虚拟证据机制将LLM的不确定信号转化为BN的证据；迭代反馈循环与不一致标识实现自动重新查询。

**📊 数据集**

文中未给出具体实验数据集，主要以福利资格评估等公共管理案例进行概念性演示；所提框架旨在适用于各种基于法律、政策或医疗等规范化域的正式数据与文档。

**📈 对比分析**

由于该工作是理论与架构设计，没有数值实验或与传统XAI方法的性能对比；作者强调此框架的优势在于结构化推理而非预测准确度，未来研究需在真实域上实现并与后置解释方法做可解释性、争议性与不确定性评估对比。

**⚠️ 局限性**

主要局限包括：① 语义对齐难题——自然语言与BN变量之间的映射仍无统一方法；② 动态模型构建——BN结构随政策变化需人类治理更新；③ 概率界定——如何将LLM的token分布映射为BN所需的条件概率；④ 人机治理与审计流程缺乏标准化规范；⑤ 可扩展性与跨域迁移——不同领域BN间的知识迁移尚未系统化。

---

## 320. GP-Adapter: Gaussian Process CLIP-Adapter for Few-Shot Out-of-Distribution Detection

**arXiv ID:** 2606.07102 | [PDF](https://arxiv.org/pdf/2606.07102v1)

**作者:** Taisei Saito `[一作]` (Ricoh Company, Ltd.), Takafumi Hiroi `[通讯]` (Ricoh Company, Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 GP-Adapter，一种训练-free 的框架，在冻结的 CLIP 嵌入上构建类特定的一类 Gaussian Process，用于少样本分类和 OOD 检测。

**💡 创新点**

在 CLIP 预训练模型上引入类级 GP 并融合图像与文本模态的预测统计，利用方差范围校准的 MSP 提升 OOD 检测，同时保持无梯度微调的轻量级特性。

**🔧 技术方法**

Gaussian Process 回归（RBF 与线性核）、CLIP 预训练模型、方差感知 MSP、prompt-ensemble 与网格搜索的超参数优化。

**📊 数据集**

ImageNet-1k 作为 ID 训练集；iNaturalist、SUN、Places、Textures 作为 OOD 测试集；同时在 ResNet-50、ImageNet-100 等上进行验证。

**📈 对比分析**

与零样本基线（MCM、GL-MCM、SeTAR）、微调方法（MSP、ODIN、ViM 等）以及少样本 prompt-learning 方法（CoOp、LoCoOp）进行对比；GP-Adapter 在少样本下保持与 CoOp/LoCoOp 相近的 ID 准确率，且 OOD 的 AUROC 约 96–97%，FPR95 明显下降；与 prompt-learning 组合进一步提升。

**⚠️ 局限性**

仅在极低样本（1–16 shot）下有效；对大类数或高 shot 情形内存受限；需手工设定 τ、α 等超参数，缺乏自动自适应；在更大规模或不同任务上验证仍有待深入。

---

## 321. The discovery of the effects of women employment participation on the fertility of developing countries: A panel data approach

**arXiv ID:** 2606.07093 | [PDF](https://arxiv.org/pdf/2606.07093v1)

**作者:** Thi Kim Ngan Nguyen `[一作]` `[通讯]` (Tokyo International University), Thi Kim Ngan Nguyen (Tokyo International University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究利用115个发展中国家的面板数据，探讨女性劳动力参与率对总生育率的因果影响，并按大洲分组分析地区异质性。

**💡 创新点**

创新之处在于将双重LASSO变量筛选方法与固定效应回归相结合，系统评估了省略变量偏误对结果的影响，并揭示了不同地区的差异化效应。

**🔧 技术方法**

使用的技术包括面板固定效应模型、双重LASSO变量选择、稳健标准误检验以及基准模型与手工选变量模型的对比。

**📊 数据集**

所用数据集来自世界银行与联合国，涵盖1991-2018年总生育率、女性劳动力参与率、教育达成率、GDP增长率、与父母同住的夫妻比例以及声音与问责指数等指标。

**📈 对比分析**

通过比较基准模型、手工选变量模型和双重LASSO模型，在四个地区检验女性劳动力参与对生育率的影响；结果显示仅北/南美洲显著负向关联，LASSO模型在其他地区往往使该变量不显著，说明过拟合和变量选择对结论有重要影响。

**⚠️ 局限性**

局限性包括样本规模受限、部分关键政策变量缺失、劳动力参与率与生育率的测量误差，以及面板数据对时间序列变动的捕捉能力有限。

---

## 322. Predictive Style Matching: Natural and Robust Humanoid Locomotion

**arXiv ID:** 2606.07083 | [PDF](https://arxiv.org/pdf/2606.07083v1)

**作者:** Simeon Nedelchev `[一作]` (Moscow Institute of Physics and Technology), Roman Gorbachev `[通讯]` (Moscow Institute of Physics and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种离线预测器预测下肢历史和速度指令下的上肢关节与步态目标，并在强化学习训练阶段将其作为奖励约束，以提升机器人行走的自然度。

**💡 创新点**

创新点在于将人体运动作为训练时的状态条件化风格先验，而非部署时的时间索引参考，从而在保持低延迟本体感知控制接口的同时，显著提升上肢协调与步态节奏。

**🔧 技术方法**

使用了GRU–MLP离线监督学习来训练预测器，随后在MuJoCo仿真中通过PPO与域随机化、奖励曲线及匹配奖励共同训练策略；训练奖励仅在学习阶段使用，部署时仅保留原始RL控制器。

**📊 数据集**

数据集来自BoneSeed的人类行走子集（约10%片段），经重映射至Unitree G1 进行离线训练。

**📈 对比分析**

在Unitree G1上与传统任务仅RL及时间索引剪辑跟踪基线进行匹配实验，PSM将上肢DTW误差降低约8倍，同时跌倒率与恢复时间与任务仅RL相当；相较之下，剪辑跟踪基线DTW最低但跌倒率高5倍。

**⚠️ 局限性**

局限包括预测器性能限制了可实现的风格质量、仿真与运动捕捉之间的状态偏移未得到显式校正、仅使用预测的第一步目标、未考虑不平坦地形或接触变化，以及需要针对不同机器人重新训练预测器。

---

## 323. Learning Perspectivist Social Meaning via Demographic-Conditioned Fusion Embeddings

**arXiv ID:** 2606.07123 | [PDF](https://arxiv.org/pdf/2606.07123v1)

**作者:** Amanda Cercas Curry `[一作]` (Independent Researcher), Gianmarco De Francisci Morales `[通讯]` (CENTAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并评估了在P1SCO数据集上针对社会维度的视角预测模型

**💡 创新点**

首次将社会意义标签与注释者的人口学信息结合，提出层次融合嵌入以捕获不同群体的解释差异

**🔧 技术方法**

RoBERTa-large文本编码器、两层MLP人口学编码器以及加法、早期拼接和concat‑encode三种融合方式，并对比零样本、少样本与微调方法

**📊 数据集**

使用P1SCO（28k条注释，来自Reddit/YouTube/Instagram，543名美英注释者，含性别、年龄、国籍）

**📈 对比分析**

在宏观PR‑AUC上，融合模型相对文本基线提升约5.9–6.5%，并在shuffle检验中显著恢复到文本基线水平，且在多种提示和微调方案中均优于零/少样本LLM

**⚠️ 局限性**

样本稀疏导致深层融合不显优势，人口学变量仅限性别、年龄、国籍（单一性别），且仅在英语美英环境下验证，未做跨数据集验证

---

## 324. QuadVerse: An Integrated Framework Aligning Visual-Physical Reality for Quadruped Simulation

**arXiv ID:** 2606.07118 | [PDF](https://arxiv.org/pdf/2606.07118v1)

**作者:** Yuxiang Chen `[一作]` (Nanjing University), Jin Xie `[通讯]` (Nanjing University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了一个集成的 QuadVerse 框架，通过从现场 RGB 视频重建 3D Gaussian Splatting 场景，提取语义网格进行接触校准，并利用真实轨迹训练残差动力学补偿器，实现从模拟到真实的无缝迁移，支持零射击视觉导航策略部署。

**💡 创新点**

创新点在于将 3DGS 视觉重建、语义网格接触校准和残差动力学补偿三项技术统一到同一几何子表面上，系统性地同步视觉、接触和执行器差异，显著缩小了 sim‑to‑real 差距。

**🔧 技术方法**

使用 Geometry‑Anchored 3DGS 重建、可批处理 ego‑view 渲染、语义网格提取与 kNN 传递、LLM 先验+后验滑动摩擦校准、基于 PPO 的残差补偿网络、IsaacGym 动力学仿真等技术。

**📊 数据集**

主要数据集为自采集的户外 RGB 视频（包含斜坡、草地、楼梯、混合铺装等 10 个场景）、对应的 LiDAR 3D 扫描以及 Unitree Go2 机器人在同一场景下的 10 分钟轨迹记录。

**📈 对比分析**

在 PSNR、SSIM、LPIPS、交互与批处理吞吐量等指标上，QuadVerse 超越 Instant‑NGP、VGGT‑X、3DGS、2DGS、Vid2Sim 等基线，平均 PSNR 21.04、SSIM 0.569、交互成功率 92%（仿真）/84%（真实）以及平均追踪误差从 0.127 rad 降至 0.043 rad；接触校准将基站位置误差从 0.70 m 降至 0.12 m；残差补偿将姿态追踪误差降低 47%。

**⚠️ 局限性**

局限在于假设场景为静态刚性几何，难以处理柔性地形或高频接触属性变化；校准依赖离线轨迹数据，限制了对快速变化环境或不同机器人平台的适应；对动态场景、细粒度接触参数以及在线自适应的支持不足。

---

## 325. OffQ: Taming Structured Outliers in LLM Quantization by Offsetting

**arXiv ID:** 2606.07116 | [PDF](https://arxiv.org/pdf/2606.07116v1)

**作者:** Haoqi Wang `[一作]` (EPFL), Lukas Cavigelli `[通讯]` (Huawei)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种后训练量化方法OffQ，利用偏移技术消除激活离群点，以实现高质量的W4A4KV4低比特量化；

**💡 创新点**

创新在于发现激活离群点呈低维结构，使用top‑1 PCA聚焦离群子空间并通过Hadamard旋转将离群能量转化为全局偏移量，避免混合精度或非均匀量化；

**🔧 技术方法**

采用post‑training quantization、top‑1 PCA、Hadamard（或部分随机）旋转、分组偏移、per‑group非对称量化以及GPTQ权重量化；

**📊 数据集**

使用WikiText进行语言建模评估，零样本常识推理数据集包括ARC-e/ARC-c、BoolQ、HellaSwag、OpenBookQA、PIQA、SIQA、WinoGrande等；

**📈 对比分析**

与QuaRot、SpinQuant、DFRot、KurTail、OSTQuant、ResQ、GPTQ、QUIK等基线在W4A4KV4设置下比较，OffQ在多种LLM（Llama 2/3/3.2、Qwen 2.5）上在困惑度和零样本准确率上均优于或接近16‑bit精度；

**⚠️ 局限性**

未评估真实推理延迟；未结合非均匀量化或旋转学习等更先进技术；仅针对W4A4KV4场景。

---

## 326. Entanglement from Expansion: High Rank-Width in Deterministic Graphs

**arXiv ID:** 2606.07110 | [PDF](https://arxiv.org/pdf/2606.07110v1)

**作者:** Tristan Cam `[一作]` (University of Bordeaux), Simon Martiel `[通讯]` (IBM Quantum)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过构造新的基于边扩张与强色指数的下界方法，证明了一系列确定性图族（如高维格子、超立方体、Hamming图、Ramanujan 图等）的秩宽可达到Θ(n)，并展示了这些图在全互联架构上可在常数深度内实现最大纠缠的图态；

**💡 创新点**

创新点在于将图的边扩张与诱导匹配（mim-width）相结合，并利用 Kahn–Kalai–Linial 定理在笛卡尔乘积上进一步提升下界，从而填补了先前仅有随机图可达到高秩宽的空白；

**🔧 技术方法**

主要技术包括：边扩张与强色指数的结合、Jelínek 的诱导匹配方法、mim-width 与秩宽的关系、图的拉普拉斯谱与 Log‑Sobolev 常数、以及笛卡尔乘积图的 Boolean 函数影响分析；

**📊 数据集**

本工作不依赖于具体实验数据集，而是以理论图族（如P_m^□k、C_m^□k、K_m^□k、Ramanujan图等）为实验对象，计算其下界；

**📈 对比分析**

与之前仅通过概率方法得到的最大秩宽下界相比，本文提供了确定性构造，且实现图态的量子电路深度为常数（或o(√n)在网格架构上），显著提升了可实现性与可验证性；

**⚠️ 局限性**

局限性包括：对超立方体秩宽的下界仍未达到其树宽的上界，强色指数的粗略上界可能导致下界不够紧；此外，方法主要针对正规图与边扩张良好的图，可能不适用于所有图族；

---

## 327. LARA: Latent Action Representation Alignment for Vision-Language-Action Models

**arXiv ID:** 2606.07100 | [PDF](https://arxiv.org/pdf/2606.07100v1)

**作者:** Mengya Liu `[一作]` (State Key Laboratory of General Artificial Intelligence, BIGAI), Siyuan Huang `[通讯]` (State Key Laboratory of General Artificial Intelligence, BIGAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Latent Action Representation Alignment (LARA) 框架，联合训练 LAM（潜在动作模型）和扩散式视觉-语言-动作（VLA）模型，实现两者的潜在动作表示对齐。

**💡 创新点**

创新点在于：① 在 VLA 与 LAM 之间实现双向对齐，使 LAM 能以真实动作轨迹为锚定，VLA 能以 LAM 的前向动态为正则；② 将 LAM 作为动态对齐的目标，而非仅作为固定伪标签；③ 通过轻量化对齐损失兼容多种扩散 VLA 结构，提升数据利用率。

**🔧 技术方法**

技术手段包括：扩散式动作生成（Dit）、潜在动作模型（基于 ViT 的 IDM、VQ‑VAE 代码本、FD‑M）、对齐损失（投影头 + 对比损失）、联合训练三阶段（LAM 预训练、LARA 联合预训练、LARA 联合后训练）。

**📊 数据集**

使用大规模无标签人类视频与机器学习演示数据（OXE 数据集、LIBERO、SIMPLER‑ENV、GR1‑Sim‑24、G1‑Real‑50），并在这些基准上进行评估。

**📈 对比分析**

与现有 VLA 及 LAM 基线比较，LARA 在 OXE‑受限场景下平均提升约 10%（完整训练）、5%（后训练提升）和 15%（LAM 细化）；在 Unconstrained 场景中也能超过多种大型预训练模型，表现出更高的数据效率和泛化能力。

**⚠️ 局限性**

局限性包括：① 对齐深度和权重需要根据具体模型架构调优；② 仍受限于预训练数据规模，无法完全匹配已见过目标机器人体型的模型；③ 对齐过程中可能引入额外训练成本，需权衡实际部署场景。

---

## 328. SigmaScale: LLM Compression with SVD-based Low-Rank Decomposition and Learned Scaling Matrices

**arXiv ID:** 2606.07098 | [PDF](https://arxiv.org/pdf/2606.07098v1)

**作者:** Ernests Lavrinovics `[一作]` (Aalborg University), Maurizio Pierini `[通讯]` (European Organization for Nuclear Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

学习并应用对行列的可激活感知缩放矩阵，以改进截断 SVD 对大型语言模型权重的低秩压缩

**💡 创新点**

创新点在于通过学习可微缩放矩阵而非解析求解，降低权重矩阵的有效秩并提升压缩质量

**🔧 技术方法**

技术包括 SVD、对数缩放向量、激活感知损失、后压缩微调（KD 与监督），以及有效秩熵分析

**📊 数据集**

使用 Llama 3.1 8B‑Instruct 与 Qwen3‑8B 预训练模型；校准集与评估使用 Wikitext‑2；微调采用 Alpaca 数据集

**📈 对比分析**

与 SVD‑LLM、ASVD+ 等现有方法比较，在 0.90x、0.75x 低秩压缩下保持或略优的困惑度与零样本任务准确率；在 0.50x 级别时性能明显下降

**⚠️ 局限性**

局限性包括：每步更新需计算完整 SVD（O(n³) 复杂度），极端压缩下效果衰退；对不同校准分布和长文本生成等场景的鲁棒性尚未验证

---

## 329. A machine-learning-assisted progressive digit-randomness screening framework for detecting non-random patterns in raw numerical research data

**arXiv ID:** 2606.07128 | [PDF](https://arxiv.org/pdf/2606.07128v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 330. Detecting Temporally Localized Manipulations in Authentic Video Streams

**arXiv ID:** 2606.07090 | [PDF](https://arxiv.org/pdf/2606.07090v1)

**作者:** Okan Umur `[一作]` (Sakarya University), Ibrahim Delibasoglu `[通讯]` (Sakarya University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究在真实视频流中插入短时段篡改片段的检测问题，构建自定义部分篡改数据集并评估两种检测方法。

**💡 创新点**

创新点在于首次聚焦短时段插入篡改场景，提出无监督基于DINOv3特征相似度的时间异常检测并指出阈值自适应的必要性。

**🔧 技术方法**

使用了预训练的DINOv3 ViT特征、线性探测器、基于余弦距离的时间序列异常检测，以及滑动窗口与视频级自适应阈值等技术。

**📊 数据集**

利用公开的自定义数据集（100条纯真视频 + 100条短篡改片段 + 100条混合视频），并对比LAV-DF、AV-Deepfake1M、TVIL等现有数据集进行评估。

**📈 对比分析**

通过帧级精度召回F1对线性探测器的三种阈值策略进行比较，发现无监督相似度方法全局精度83%、召回54%、F1 65%；纯真控制组视频级准确率达95%。

**⚠️ 局限性**

主要限制是阈值固定导致不同视频误报/漏报，线性探测器跨域泛化差，且需要更智能的内容自适应阈值与更强的跨域训练策略。

---

## 331. On the Geometry of On-Policy Distillation

**arXiv ID:** 2606.07082 | [PDF](https://arxiv.org/pdf/2606.07082v1)

**作者:** Zhennan Shen `[一作]` (Hong Kong University of Science and Technology), Yi R. Fung `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过一系列参数空间诊断，系统性描述了大语言模型在使用on‑policy distillation（OPD）时的更新轨迹，并将其与监督微调（SFT）和可验证奖励的强化学习（RLVR）进行对比，进一步揭示了OPD在训练过程中快速进入并保持的低维子空间锁定现象。

**💡 创新点**

①OPD定位为一种“松弛的非主轴”更新范式，兼具RLVR的几何保留特性和SFT的稠密监督优势；②发现OPD在训练早期即锁定并持续使用一个低维更新通道；③表明子空间锁定主要由目标函数的组成决定，而非运行时采样或稀疏监督的变化。

**🔧 技术方法**

使用参数空间诊断指标（更新稀疏度、主轴旋转角度、谱漂移、更新遮罩重叠）、稳定秩与谱形状分析、累积更新的低维子空间相似性、低秩投影实验以及三种扰动（稀疏监督、离线采样、目标混合）进行系统实验。

**📊 数据集**

以Qwen3系列大模型（尤其是Qwen3‑8B）为实验基准，覆盖不同教师‑学生比例、代码数据、以及MoE教师等多种设置，并在推理/编程类推理任务上验证。

**📈 对比分析**

实验通过与SFT（离线监督）和RLVR（稀疏奖励）在相同基准和数据分布下进行对比。OPD在更新稀疏度、主轴旋转、谱漂移等指标上介于两者之间，保持较低的谱漂移；在低秩投影实验中，OPD对16维更新通道约束几乎不受影响，而SFT则显著下降，表明OPD的低维通道功能充分且稳健。

**⚠️ 局限性**

研究仅覆盖Qwen3家族的推理任务，未验证到其他模型、模态或任务分布；诊断依赖于检查点存档，可能无法捕捉实时动态；所提出的三门机制和协方差分析为一致性解释而非完整因果或形式化理论。

---

## 332. REMEDI: A Benchmark for Retention and Unlearning Evaluation in Multi-label Clinical Disease Inference

**arXiv ID:** 2606.07141 | [PDF](https://arxiv.org/pdf/2606.07141v1)

**作者:** Anurag Sharma `[一作]` (IIT Kharagpur), Koustav Rudra `[通讯]` (IIT Kharagpur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 REMEDI Benchmark，用于评估多标签临床疾病推理模型的机学习删改（unlearning）与保留性能。

**💡 创新点**

创新点在于：①构建真实医疗领域、包含三种不同删除规模的患者级删改设置；②在多标签、多类ICD‑9疾病推理任务上进行评估；③同时从模型效用与隐私两方面衡量删改效果。

**🔧 技术方法**

采用了四种代表性机学习删改技术（梯度上升 Gradient Ascent、对抗删改 Adversarial Unlearning、坏教师 Bad Teacher、SCRUB），并在 BioLinkBERT 与 BioBERT 两大生物医学语言模型上实施。

**📊 数据集**

使用 MIMIC‑III 临床数据库中的离院总结文本，挑选10个最常见的ICD‑9疾病类别，构建多标签分类数据集。

**📈 对比分析**

通过与完整重训练模型对比，评估宏 F1 及成员推断攻击（MIA）隐私指标，结果显示 SCRUB 在保持与重训练模型相近的 F1 的同时，MIA 评分最接近理想值；GA 失效；AU 与 BT 在大规模删改时性能明显下降。

**⚠️ 局限性**

局限性包括仅使用单一数据集与有限的10类疾病；多标签相关性处理不足；隐私评估仅依赖基于 loss 的 MIA；缺乏对更大规模、更多标签或多模态场景的验证。

---

## 333. HRsR: Hierarchical Rotation System Reconstruction

**arXiv ID:** 2606.07078 | [PDF](https://arxiv.org/pdf/2606.07078v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

---

## 334. Information-Theoretic Bounds for Sparse Covariance Estimation in the Vertical-Split Distributed Model

**arXiv ID:** 2606.07124 | [PDF](https://arxiv.org/pdf/2606.07124v1)

**作者:** Jing Yee Tan `[一作]` (University of Hong Kong), Guangyue Han `[通讯]` (University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究了在垂直分割设置下的分布式协方差矩阵估计的最小最大估计误差，探讨了在交叉协方差矩阵C_21上施加元素级s-稀疏性是否能降低所需的通信和样本复杂度。

**💡 创新点**

证明了在垂直分割中，稀疏性确实有助于降低协方差估计的通信成本，与水平分割的结果形成对比，提出了新的最小最大下界和可实现方案。

**🔧 技术方法**

使用了Fano方法和条件强数据处理不等式（C-SDPI），结合覆盖网量化和逐项硬阈值化的协议。

**📊 数据集**

使用了m个独立同分布的子高斯样本，具体数据集未明确给出，但涉及到的特征维度为d_1和d_2。

**📈 对比分析**

通过与已有的密集协方差估计方法进行比较，展示了在稀疏情况下，通信预算从Ω(σ^4 d_1 d_2 d_k/ε^2)降低到Ω(σ^4 s' log(d_1 d_2/s') d_k/ε^2)，并且构造了一个匹配的可实现方案，证明了这一改进是紧的。

**⚠️ 局限性**

限制在于该方案需要事先知道稀疏性s，而在实际应用中，s通常是未知的。是否可以在不知道s的情况下实现相同的通信成本降低仍然是一个开放问题。

---

## 335. 3DMorph: Single-Image-Guided Local 3D Shape Editing and Morphing

**arXiv ID:** 2606.07115 | [PDF](https://arxiv.org/pdf/2606.07115v1)

**作者:** Tobias Preintner `[一作]` (Leiden University), Niki van Stein `[通讯]` (Leiden University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种单图像引导的本地3D形状编辑与形状变形方法3DMorph，能自动从单张编辑后的图像识别并映射局部几何修改到三维网格，并生成连续中间形状。

**💡 创新点**

创新点包括：①无训练、仅凭单张编辑图像实现局部几何编辑；②自动定位3D编辑区域的包围盒；③采用硬约束+流一致的两阶段逆采样调度；④提供专门的Delta3D基准数据集。

**🔧 技术方法**

技术上结合了SLAT结构（稀疏体素+体素级潜在特征）与Trellis生成器及解码器；基于SSIM差异检测2D差异并投影为3D包围盒；利用RePaint等逆扩散采样策略实现局部Voxel与Latent更新。

**📊 数据集**

使用Delta3D基准（基于Fusion 360 Gallery Assembly数据集的74对对象，12视角共864样本），并与Trellis、Hunyuan3D、TripoSG等现有3D生成方法进行对比。

**📈 对比分析**

在Delta3D上对比Chamfer、Hausdorff、IoU、Dice、MS-SSIM等指标，3DMorph在所有指标上均显著优于基线，CD降低19%，IoU提升3个百分点，且在包围盒预测误差下仍保持优势。

**⚠️ 局限性**

局限性在于包围盒预测易受视角影响，若视角不佳可能失效；当前细节分辨率受SLAT解码器容量限制；对高度非凸或极端形变的编辑效果仍有限。

---

## 336. AsyncPatch Diffusion: spatially-flexible image generation

**arXiv ID:** 2606.07079 | [PDF](https://arxiv.org/pdf/2606.07079v1)

**作者:** Samuele Papa `[一作]` (Google DeepMind), Klaus Greff `[通讯]` (Google DeepMind)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出AsyncPatch Diffusion，一个在图像不同区域使用异步噪声级别的联合扩散框架，能够在不改变模型架构的情况下实现零样本图像修复与自适应生成。

**💡 创新点**

核心创新在于：①为每个像素/隐层标记独立时间轴，形成异步噪声场；②引入受控噪声采样器平衡均匀与异构状态；③设计输入引导机制，通过比较不同噪声级的分数提升局部一致性；并给出了有效的ELBO证明。

**🔧 技术方法**

使用UNet+FiLM的联合扩散网络，基于标准DDPM训练目标的重加权形式；在像素级和潜在级均实现，利用Perlin掩码、Patchwise与AsyncPatch采样策略；在推理时支持自回归、速度加速与纹理合成。

**📊 数据集**

在ImageNet 64/256、LSUN Bedroom等公开图像数据集上训练，并在这些数据集的在painting、纹理合成与生成任务上进行评估。

**📈 对比分析**

与传统LDM、RePaint、RAD等基线比较，AsyncPatch在ImageNet 64的FID仅比基线差0.03，在ImageNet 256与LSUN的inpainting任务中LPIPS与Fidelity均保持在同等或略优水平；在不需要额外fine‑tune的零样本修复上表现尤为突出。

**⚠️ 局限性**

限制包括：对高分辨率生成尚需改进、对极端掩码（如长条形）性能仍低于专门的监督模型、采样速度受限于像素级异步噪声的计算开销。

---

## 337. Clairvoyant: Predictive SJF Scheduling to Mitigate Head-of-Line Blocking in Serial LLM Backends

**arXiv ID:** 2606.07248 | [PDF](https://arxiv.org/pdf/2606.07248v1)

**作者:** Aravind Sundaresan `[一作]` `[通讯]` (Independent Researcher), Aravind Sundaresan (Independent Researcher)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个轻量级 sidecar 代理，利用词法特征和 XGBoost 预测输出长度，在序列 LLM 后端实现非抢占式 SJF 调度，显著缓解 Head‑of‑Line Blocking。

**💡 创新点**

创新点在于将词法特征+XGBoost 预测与 0.029 ms 的推理延迟结合，提出无修改后端、无预训练嵌入的 SJF 调度方案，并在低内存环境下实现高效排队。

**🔧 技术方法**

采用 19 个轻量词法特征提取、ONNX 导出的 XGBoost 分类器、Go 实现的 min‑heap SJF 调度器，并使用 0.029 ms 的预测推理。

**📊 数据集**

使用自然对话日志（ShareGPT、LMSYS‑Chat‑1M、OASST1）训练模型，排除短文本指令集（Alpaca、CodeAlpaca）作为训练源。

**📈 对比分析**

与 FCFS、长度阈值规则、关键词启发式进行对比，SJF 在 100 并发短/长请求的 RTX 4090 上将短请求 P50 延迟下降 70‑76%，长请求略增；跨分布准确率 52‑66%，比随机 50% 高出 2‑16%。

**⚠️ 局限性**

局限包括仅适用于序列 OpenAI‑兼容后端、对训练分布高度敏感、需手动设定 starvation timeout、对多租户公平性缺乏保障，且仅在单 GPU 环境验证。

---

## 338. When Large Language Models Fail in Healthcare: Evaluating Sensitivity to Prompt Variations

**arXiv ID:** 2606.07237 | [PDF](https://arxiv.org/pdf/2606.07237v1)

**作者:** Mahdi Alkaeed `[一作]` `[通讯]`, Mahdi Alkaeed

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估了通用与医学专用大型语言模型在医疗问答任务中对词汇和句法扰动的鲁棒性，并提出了标准化的扰动分类与评估框架。

**💡 创新点**

创新点在于将词汇与句法扰动统一分为自然扰动与对抗扰动，使用多指标（BERTScore、USE、RR、FIR）量化模型一致性与可靠性，并对比了不同模型在同一扰动下的表现。

**🔧 技术方法**

采用了多种技术：BioSyn 同义词检索、句子嵌入（Sentence‑BERT、USE）、cosine 相似度筛选、PromptSensiScore 等评估指标，并在 Hugging Face/LLama3/Ollama 等平台进行推理。

**📊 数据集**

使用 MedMCQA 作为基准数据集，对 100 个样本在未扰动、词汇扰动和句法扰动三种情景下进行实验。

**📈 对比分析**

比较方法为对照未扰动和扰动条件下的各指标分数；结果显示一般模型 GPT‑3.5 在未扰动时接近 99% 的准确率，但词汇扰动下降至 96% 以上，句法扰动进一步降至约 94%；医学专用模型 BioLlama‑3 在所有扰动下表现最稳健，准确率下降幅度最小。

**⚠️ 局限性**

局限性包括：依赖传统的词向量相似度指标，无法精准捕捉临床语义细微差异（如否定、时间修饰等）；实验仅覆盖单一问答任务，未涵盖多模态或跨语言场景。

---

## 339. FLOWREADER: Min-Cost Flow Optimization for Multi-Modal Long Document Q&A

**arXiv ID:** 2606.07235 | [PDF](https://arxiv.org/pdf/2606.07235v1)

**作者:** Ambuj Mehrish `[一作]` (Ca' Foscari University of Venice), Sebatiano Vascon `[通讯]` (Ca' Foscari University of Venice)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的方法，将多模态证据组装重新框架为最小成本流问题，旨在解决长文档中证据碎片化的问题。

**💡 创新点**

创新点在于将证据组装过程整合为一个优化问题，使用单一评分向量控制源选择、汇选择及边的成本和容量，从而提高了多模态证据的处理效率。

**🔧 技术方法**

使用了最小成本流算法、熵正则化复制动力学、图传播、个性化PageRank等技术。

**📊 数据集**

在VisDoMBench数据集上进行评估，该数据集包含多个多模态子集，如SPIQA、FetaTab、PaperTab、SciGraphQA和SlideVQA。

**📈 对比分析**

与现有的基线方法（如G2-Reader）进行比较，FlowReader在PaperTab和SlideVQA子集上表现最佳，分别提高了1.30和0.62的准确率，整体表现接近最强基线，宏平均得分为65.47，距离最强基线仅0.74。

**⚠️ 局限性**

限制在于单一评分h的使用可能导致上游误校准的传播，流预算F的统一设置可能导致资源分配不均，且熵正则化复制动力学的收敛性没有正式的最大团保证。

---

## 340. Does Appearance Help? A Systematic Study of Image-Based Re-Identification in Online 3D Multi-Pedestrian Tracking

**arXiv ID:** 2606.07233 | [PDF](https://arxiv.org/pdf/2606.07233v1)

**作者:** Eduardo Borges `[一作]` (University of Coimbra), Urbano J. Nunes `[通讯]` (University of Coimbra)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究将RGB基ReID融入在线3D多目标跟踪，利用轻量化投影方式将LiDAR几何与图像外观信息分离；

**💡 创新点**

创新点在于通过分阶段投影+轻量化ReID网络实现低延迟，并提出两阶段级联匹配策略优于传统线性融合；

**🔧 技术方法**

使用PointPillars+AB3DMOT作为几何检测与跟踪基线，MobileNetV2/MobileNetV3‑Small/ResNet‑18/ MGN等轻量化CNN骨干及Swin‑T Transformer，BNNeck+triplet+CE损失，Kalman滤波与Hungarian匹配；

**📊 数据集**

主要数据集为KITTI行人序列，Market‑1501用于预训练并进行领域微调；

**📈 对比分析**

与仅几何GIoU、仅外观、线性融合比较，级联匹配在HOTA/IDF1上提升约0.3%/0.3%，但帧率从57 ms提升到≈110 ms，显著提升身份保持；

**⚠️ 局限性**

主要限制是ReID引入显著延迟，Transformer效果不佳，外观记忆窗口高N导致计算膨胀，且在复杂遮挡下仍存在一定身份切换。

---

## 341. Adversarial Creation and Detection of AI-Generated Social Bot Content

**arXiv ID:** 2606.07219 | [PDF](https://arxiv.org/pdf/2606.07219v1)

**作者:** Mykola Trokhymovych `[一作]` (Universitat Pompeu Fabra), Filippo Menczer `[通讯]` (Indiana University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个多语言跨平台的对抗式数据集，通过对真实社交媒体用户历史行为建模并使用大型语言模型生成逼真AI对话，随后在此数据上训练内容检测模型。

**💡 创新点**

创新点在于：①采用对抗式数据生成，将用户画像与会话上下文相结合，模拟恶意模仿，生成可直接对抗的训练样本；②在多语言、跨平台环境下进行评估，克服传统数据缺乏真实AI生成内容的问题。

**🔧 技术方法**

技术方法包括：提示式LLM生成（Gemma‑3n‑E4B、Qwen3‑235B‑A22B）、embeddingGemma‑300m检索相似会话；训练Transformer（mBERT、XLM‑RoBERTa、Gemma3）和梯度提升LFC；对比训练无模型（FastDetectGPT、Binoculars、GECScore）等。

**📊 数据集**

使用数据集：自建的17语言、Telegram+Reddit 36个频道、6,326用户的73,521条真实信息与263,594条AI生成配对；外部评测使用Fox8‑23、MultiSocial、AIGTBench等公开数据集。

**📈 对比分析**

评估采用ROC‑AUC，模型在自建数据上TC（RoBERTa）AUC达0.969，在外部Fox8‑23上AUC达0.972，显著优于所有基线（最低0.593）；用户级检测AUC可达0.99。

**⚠️ 局限性**

局限性：仅处理文本数据、仅覆盖Telegram和Reddit两平台、主要关注新闻与政治话题；生成采用提示式LLM，未涵盖更高级的fine‑tune模仿；外部基准数据为旧版、易识别的机器人；需要持续更新训练样本以跟上更强大LLM的进步。

---

## 342. A Large-Scale Per-Speaker Analysis of Re-identification Risk in Speech Anonymization

**arXiv ID:** 2606.07210 | [PDF](https://arxiv.org/pdf/2606.07210v1)

**作者:** Orane Dufour `[一作]` (Université de Lorraine), Emmanuel Vincent `[通讯]` (Université de Lorraine)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在不同匿名化系统、攻击者架构和对话长度下，对近5000名说话人进行了大规模逐说话人隐私风险分析。

**💡 创新点**

首次从逐说话人视角量化隐私风险，揭示说话人易被识别与否受攻击者、匿名化器和语音量交互影响，且不依赖预先标签。

**🔧 技术方法**

采用链接度（linkability）指标作为最坏情况评估，使用ECAPA、WavLM ECAPA、ResNet三种ASV架构；匿名化使用VPC 2025基线B3、B5声码转换模型；计算Jaccard相似度评估列表变化。

**📊 数据集**

LibriSpeech 训练集、CommonVoice 11.0 作为测试集（A、B分割），共计22,024名说话人训练、4,949名测试说话人。

**📈 对比分析**

对每种配置计算平均链接度并选取Q1/Q3阈值形成易/难链接说话人列表，比较其交集/并集并用Jaccard相似度评估各因素影响；结果显示只有5名说话人始终易被识别，166名始终难被识别，交集极小；不同因素对列表的影响相近，未能单独决定隐私风险。

**⚠️ 局限性**

实验仍受限于仅两种匿名化基线、三种攻击者、固定对话长度取值，且未考察其他隐私度量或更丰富的说话人属性；缺乏对不同目标说话人映射策略的深入评估。

---

## 343. Technological Fitness and Regional Growth in Japan

**arXiv ID:** 2606.07202 | [PDF](https://arxiv.org/pdf/2606.07202v1)

**作者:** Rintaro Karashima `[一作]` (University of Hyogo), Hiroyasu Inoue `[通讯]` (University of Hyogo)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文利用日本企业专利数据构建47个都道府县与35类技术的双边网络，并运用Fitness‑Complexity算法计算各都道府县的技术适应度，随后通过两向固定效应面板模型检验技术适应度与未来五年地区经济增长的关系。

**💡 创新点**

创新点在于首次将非线性Fitness‑Complexity算法应用于日本都道府县层面的专利网络，发现技术适应度在控制个体与时间效应后显著正向预测增长，并揭示低收入地区对技术复杂度的增益更大，提供了技术复杂度与地区增长间因果方向的新证据。

**🔧 技术方法**

主要技术包括：①基于RTA指数的双边网络构建；②Fitness‑Complexity非线性迭代算法；③两向固定效应回归配合Driscoll‑Kraay标准误；④滞后/领先检验以确认因果方向。

**📊 数据集**

数据集为1981‑2015财年约390万份日本企业专利（按国际专利分类映射至Schmoch技术领域）以及同期各都道府县的实质性地区生产总值与人口统计数据。

**📈 对比分析**

与传统方法相比，Fitness‑Complexity在控制固定效应后预测能力显著（β≈0.0029，p=0.007），而单向或不加固定效应的模型无法捕捉此关系；跨期相关性不稳定提示了面板方法的必要性。

**⚠️ 局限性**

局限性包括：①仅使用单一复杂度指标，未检验其他指标的稳健性；②面板设计无法排除时间变动的潜在混杂因素（如研发投入、人力资本变迁）；③五年聚合窗口可能掩盖短期动态；④结果仅针对日本，外部有效性尚待验证。

---

## 344. From Correctness to Utility: Gain-Based Prefix Evaluation for LLM Reasoning

**arXiv ID:** 2606.07190 | [PDF](https://arxiv.org/pdf/2606.07190v1)

**作者:** Yuhang Zhou `[一作]` (Fudan University), Guangnan Ye `[通讯]` (Fudan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Prefix Utility Model (PUM)，通过轻量级学生模型对推理前缀的增益（即成功率提升）进行估计，并用该增益构造成对式偏好来训练一个基于LLM的前缀评价器，进而在数学推理任务中对前缀的价值进行评估；

**💡 创新点**

创新点在于将前缀评估从传统的局部正确性转向基于最终结果的增益评估，利用对比无前缀和有前缀的成功率差异来衡量前缀对未来可解性的边际提升，并通过对式排名（Bradley‑Terry）将增益转化为可训练的偏好；

**🔧 技术方法**

主要技术包括：轻量级学生模型的前缀增益采样、增益规范化与平均得到内在效用、对式偏好构造与过滤、PUM的标量评估头训练、以及在最佳- N 选择、束搜索和强化学习（GRPO）中的应用；

**📊 数据集**

使用数据集包括：MATH（用于构造20K推理轨迹和280K对式偏好）、GAOKAO2023、MATH500、AIME2025进行评测，另外在硬数据评估中采集了DAPO-Math-17K的5.7k题目；

**📈 对比分析**

通过与PRM、PQM、CRM等基线在最佳- N 选择、束搜索和RL任务中的对比，PUM在大候选池、搜索预算增大或规则奖励稀疏的情境下表现最佳，取得最高的准确率和更稳健的排名分布；

**⚠️ 局限性**

局限性包括：仅在可验证的数学推理任务上评测；增益估计需要大量学生回放，计算成本仍高；目前采用单一标量化简化了增益分布，未充分利用其多模态结构，且对结果不确定或需要人工判定的领域扩展有待研究。

---

## 345. RETROSPECT: RETROsynthesis via Sequential Prediction, and Chemically Transformed-ranking

**arXiv ID:** 2606.07181 | [PDF](https://arxiv.org/pdf/2606.07181v1)

**作者:** Raja Sekhar Pappala `[一作]` (Mstack AI), Deepak Warrier `[通讯]` (Mstack AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了单步逆向合成的生成-重排序框架 RETROSPECT，包含强大的 Transformer 生成器和 LambdaMART 重排序器。

**💡 创新点**

创新点包括：①将根对齐 SMILES 与随机 SMILES 混合增强；②引入 Pre-LayerNorm、EMA、绑定词嵌入、可微分原子平衡辅助损失；③将生成与重排序模块化，证明单模型生成器可作为强大候选源。

**🔧 技术方法**

使用技术包括 Transformer 生成器、混合 SMILES 数据增强、Pre-LayerNorm、EMA、atom-balance 辅助损失、LambdaMART 学习排名、结构/模板/DFT 特征。

**📊 数据集**

采用 USPTO-50K 数据集，训练/验证/测试分布为 40008/5000/5007 条反应。

**📈 对比分析**

与现有模板、半模板、无模板系统对比：单模型 Top‑1 55.00%，Top‑10 86.18%；加入重排序后 Top‑1 59.4%，Top‑10 93.06%，虽未超过 RetroChimera 等集成系统，但单模型表现出色。

**⚠️ 局限性**

局限性：仅在 USPTO-50K 评估，数据偏差大；重排序依赖生成池质量；DFT 特征增益有限；未对实际化学可行性和多步规划进行验证。

---

## 346. When Recovery Matters: The Blind Spot of Surrogate Privacy in MLLM Editing

**arXiv ID:** 2606.07171 | [PDF](https://arxiv.org/pdf/2606.07171v1)

**作者:** Siyuan Xu `[一作]` (City University of Hong Kong), Sam Kwong `[通讯]` (Lingnan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向隐私保护的多模态大语言模型图像编辑的完整评估与恢复框架，并构建了覆盖36类隐私细粒度与65条编辑指令的SPPE基准数据集。

**💡 创新点**

创新点包括：①将编辑可行性评估与从替代图像恢复原图的两项任务纳入框架；②提出指令感知的可编辑性判别模型ERMA；③设计基于循环一致性与编辑标签的恢复模型，充分利用编辑示例信息。

**🔧 技术方法**

技术手段包括：CLIP多模态关系建模用于可编辑性预测；MM‑DiT扩散变压器结合编辑标签与视觉参考进行源图像恢复；LoRA微调与循环一致性正则化提升恢复效果。

**📊 数据集**

使用SPPE数据集（55,696实例，36隐私类别，65编辑指令）进行训练与评估，并在公开InstructPix2Pix数据集上测试模型泛化能力。

**📈 对比分析**

与多种图像质量评估、参考式编辑与迁移学习方法比较；ERMA在编辑可行性评估上SRCC、PLCC分别提升13.9%/12.3%；恢复模型在SPPE上在源完整性与编辑一致性上均优于SOER及其他基线，并在InstructPix2Pix上保持较好泛化。

**⚠️ 局限性**

局限性包括：对极端隐私区域或复杂编辑的鲁棒性仍有限；循环一致性训练需权衡源保持与编辑效果；模型受生成器在特定编辑类型上的偏差影响。

---

## 347. UrduMMLU: A Massive Multitask Benchmark for Urdu Language Understanding

**arXiv ID:** 2606.07167 | [PDF](https://arxiv.org/pdf/2606.07167v1)

**作者:** Ahmer Tabassum `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Preslav Nakov `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个本土化的多语言MMLU风格基准——Ustad360，包含26,431个乌尔都语多项选择题，覆盖26个学科和5个领域，并在30个LLM上进行零样本和少样本评估。

**💡 创新点**

①从乌尔都语教育材料直接提取题目，避免翻译噪声；②双人人工标注+严格共识过滤确保答案质量；③提供多语言（英乌）提示实验，揭示乌尔都语知识不均衡。

**🔧 技术方法**

OCR+视觉语言解析提取PDF题目；文本清洗、去重、归一化；双人人工标注；生成式零样本/少样本评估；准确率与无效输出率度量。

**📊 数据集**

结合乌尔都语MCQ网站的标注题库和巴基斯坦SSC/HSSC考试PDF的题目，最终集成26,431道题。

**📈 对比分析**

在30个LLM（包括Gemini、GPT、Claude、Qwen等）上用英乌双语提示进行零样本评估，Gemini-3.5-Flash最高达90.2%；最强开源模型DeepSeek-V4-Flash为82.4%；模型在STEM表现好，文科大幅低下；少样本提升有限。

**⚠️ 局限性**

仅覆盖巴基斯坦中学课程，缺乏本科、印度乌尔都语、方言或双语内容；仅为四选一MCQ，未评估生成式写作；提示与少样本实验有限；模型在乌尔都语文学、宗教等领域表现差。

---

## 348. KIT's Submission to Cross-Lingual Voice Cloning in IWSLT 2026

**arXiv ID:** 2606.07240 | [PDF](https://arxiv.org/pdf/2606.07240v1)

**作者:** Seymanur Akti `[一作]` (Karlsruhe Institute of Technology), Alexander Waibel `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并改进了多语言语音合成模型，用语言标签、强化学习微调及参考语音检索实现跨语言声克隆

**💡 创新点**

引入语言标签提示减少口音泄漏、使用RL微调提升可懂度、通过检索匹配参考音频改善领域词发音

**🔧 技术方法**

FishAudio‑S2‑Pro多语言TTS、语言标签提示、Group Relative Policy Optimization（GRPO）强化学习、LoRA适配、ASR与说话人验证模型

**📊 数据集**

ACL 60/60 数据集（ACL 会议长音频），包含英语参考与法语、阿拉伯语、中文目标文本

**📈 对比分析**

与基线相比，语言标签显著降低阿拉伯语与法语的 CER，RL 微调进一步提升一致性，整体性能保持或提升，无明显说话人相似度下降

**⚠️ 局限性**

对中文的效果略逊，词汇匹配效果受限于检索质量，对未知词仍易出现拼音化或分解发音，模型对长语料的稳定性仍有待提升

---

## 349. Moodie: An Early-Stage Design Exploration for Supporting Fear of Missing Out with LLM-based Chatbots

**arXiv ID:** 2606.07231 | [PDF](https://arxiv.org/pdf/2606.07231v1)

**作者:** Hsin-Yu Tsai `[一作]` (National Chung Cheng University), Tzu-Hsiang Huang `[通讯]` (National Chung Cheng University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出并评估了一款名为Moodie的面向抑制“错过恐惧”(FoMO)的LLM聊天机器人；

**💡 创新点**

其创新点在于将基于FoMO‑R理论的情绪调节策略与可切换的情感支持/实用建议两种响应类型相结合，形成专门针对FoMO的定制化交互；

**🔧 技术方法**

采用GPT‑4o作为核心语言模型，并通过精心设计的提示工程实现情境感知与回应风格切换；

**📊 数据集**

数据集包括15名FoMO受访者的开放式问卷反馈用于前期设计，随后对21名受试者进行为期一周的对照实验（Moodie vs. GPT‑4o原生聊天）以及FoMO量表和情绪调节问卷；

**📈 对比分析**

通过混合效应模型和主题分析比较，发现两种聊天机器人在FoMO得分下降和情绪调节方面无显著差异，但Moodie在用户参与度、情感联结和自我反思方面表现更佳；

**⚠️ 局限性**

局限性包括样本规模较小、实验时长仅一周、缺乏对聊天日志的深入分析以及对长期效果的验证。

---

## 350. MMAE: A Massive Multitask Audio Editing Benchmark

**arXiv ID:** 2606.07229 | [PDF](https://arxiv.org/pdf/2606.07229v1)

**作者:** Ziyang Ma `[一作]` (Shanghai Jiao Tong University), Xie Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MMAE，首个覆盖多模态（声音、音乐、语音及其混合）且具有多层复杂度与操作类别的指令式音频编辑基准；

**💡 创新点**

创新点在于构建三维任务分类（模态、复杂度、操作）以及基于细粒度Rubric的评估框架，实现对编辑正确性与一致性的可解释、多维度量化；

**🔧 技术方法**

使用了多模态LLM（如Qwen3‑Omni）作为裁判进行Rubric判定，并采用人机协作生成Rubric、人工审核保证数据质量；

**📊 数据集**

基准数据集由2,000条音频样本组成，包含17,741个Rubric，平均每条样本14.46秒、1.22个操作，覆盖全部六层复杂度与八类操作；

**📈 对比分析**

在MMAE上评测了五个最新音频编辑模型（Step‑Audio‑EditX、Ming‑UniAudio、MMEdit、Audio‑Omni、SmartDJ），以及Identity与Noise基线；结果显示所有模型的Exact Match Rate低于5%，且在复杂与跨模态任务上性能显著下降；

**⚠️ 局限性**

局限性包括：1）评估仍依赖LLM裁判，可能存在主观偏差；2）基准样本多为人工构造，缺乏真实工业场景；3）对高阶多步编辑的支持不足，未能充分挖掘模型潜能；4）对生成质量的感知评估未覆盖主观体验。

---

## 351. Entropy as a Structural Prior: How a Log-Barrier on DiT Belief Space Drives Musical Diversity and Development

**arXiv ID:** 2606.07207 | [PDF](https://arxiv.org/pdf/2606.07207v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 352. HKVM-RAG: Key-Value-Separated Hypergraph Evidence Organization for Multi-Hop RAG

**arXiv ID:** 2606.07218 | [PDF](https://arxiv.org/pdf/2606.07218v1)

**作者:** Mingyu Zhang `[一作]` (Harbin Institute of Technology), Ying Ma `[通讯]` (Harbin Institute of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了HKVM-RAG，一个将答案路径超图作为检索键、检索文本作为值的关键-值分离证据组织层，用于多跳检索增强生成。

**💡 创新点**

创新点在于把高阶超图键与检索值分离，提供可复用的证据控制信号，并通过固定提取子结构隔离键空间设计，从而在不改变检索预算或候选语料的情况下提升多跳证据组织。

**🔧 技术方法**

使用LLM（DeepSeek V4 Flash）进行关系三元组提取、超图遍历与加权超图PPR、ColBERTv2稀疏检索、训练到开发的控制器以及密集检索与传统图/图神经网络的对照。

**📊 数据集**

数据集包括 2WikiMultiHopQA、MuSiQue 以及 HotpotQA 的官方开发集。

**📈 对比分析**

与 BM25、Contriever、ColBERTv2、基于图的 KG-PPR 等对照，答案路径超图键在 2WikiMultiHopQA 和 MuSiQue 上分别提升 3.426 与 3.592 F1，整体在密集检索上提升 11.084、6.763 与 5.966 F1。

**⚠️ 局限性**

限制：依赖固定的 LLM 提取缓存，结构检索在 HotpotQA 上未显著提升答案 F1；实验仅在开发集验证，未在公开测试集或实时部署环境中评估；高阶超图的生成与参数需要手工调优。

---

## 353. Robotic Policy Adaptation via Weight-Space Meta-Learning

**arXiv ID:** 2606.07217 | [PDF](https://arxiv.org/pdf/2606.07217v1)

**作者:** Christian Bianchi `[一作]` (ItalAI), Luca Franco `[通讯]` (ItalAI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了WIZARD，一个基于权重空间的元学习框架，用语言和短视频生成LoRA适配器，实现VLA策略的零样本任务适配。

**💡 创新点**

创新点在于直接在权重空间生成任务特定的LoRA更新，避免在测试时进行梯度优化和动作标签需求。

**🔧 技术方法**

使用了LoRA、权重空间元学习、任务嵌入、对齐监督和规模感知生成等技术。

**📊 数据集**

在LIBERO四个子数据集（Spatial、Object、Goal、10）以及Franka Emika Panda真实机器人上进行评估。

**📈 对比分析**

与基线LoRA专家、最近邻检索、MT‑VLA fine‑tune等对比，WIZARD在未见数据集上平均提升约2倍，单任务提升多达14倍，并在真实机器人上显著提高成功率。

**⚠️ 局限性**

局限性包括依赖任务证据质量、仅生成单任务专家且在长周期组合任务上表现有限。

---

## 354. Shield-Loco: Shielding Locomotion Policies with Predictive Safety Filtering

**arXiv ID:** 2606.07193 | [PDF](https://arxiv.org/pdf/2606.07193v1)

**作者:** Aditya Shirwatkar `[一作]` (Indian Institute of Science), Majid Khadiv `[通讯]` (Technical University of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计并实现了一种基于采样的预测安全滤波器，能够在强化学习策略给定的接触位置上进行全物理仿真与优化，实时修正步态以避免碰撞。

**💡 创新点**

创新点在于将安全约束直接投射到接触位置空间，并结合几何投影、动量加速以及复制交换的采样优化，使得安全滤波器在保持策略不侵入的前提下实现全身体积碰撞避免。

**🔧 技术方法**

使用的核心技术包括MPPI/CEM零阶采样优化、MuJoCo全物理仿真、价值函数预估、投影QP、动量更新、复制交换以及异步规划框架。

**📊 数据集**

实验数据来自Unitree Go2四足机器人与MuJoCo仿真中的合成密集障碍环境（模拟手机、包、绳索等真实物体），未使用公开数据集。

**📈 对比分析**

通过与CBF、HJ基线以及未过滤的计划进行对比，采用跟踪误差、规划违规次数和实际违规次数作为指标；结果显示该方法将实际违规率降至约30%，并实现了最低的跟踪误差，优于基线。

**⚠️ 局限性**

主要局限包括缺乏理论安全保证、可能陷入局部最优、投影近似对收敛性的影响、策略产生的中间轨迹仍可能违规，以及未实现在线感知与多地形/人形机器人场景的验证。

---

## 355. Geometry of Semantic Space: Comparative Study of Discrete and Continuous Models

**arXiv ID:** 2606.07183 | [PDF](https://arxiv.org/pdf/2606.07183v1)

**作者:** Gabriel Bounias `[一作]` (Institut des Systemes Complexes de Paris IdF), Sabine Ploux `[通讯]` (Centre d’analyse et de mathématique sociales)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文比较了基于监督学习的向量嵌入（CamemBERT）与基于共现团的离散图模型在相同法语语料中的语义几何结构。

**💡 创新点**

创新点在于将两种模型的几何一致性作为评价标准，揭示向量模型局部一致但全局结构更稀疏、层次不清，证明图模型在语义梯度、空间分布和结构均匀性方面具有优势。

**🔧 技术方法**

使用了图嵌入方法（Force‑Directed、Spectral、Isomap、Node2Vec）、BFS子树、相关系数分析、图度数与介数中心性、Infomap 社区划分、trustworthiness 指标等技术。

**📊 数据集**

实验数据来源于 2019 年法国 Grand Débat National 语料库（约 1000 万句），经过词形归一化后提取共现团作为语义单元。

**📈 对比分析**

通过局部 BFS 子树的空间相似度对比、全局图度数/介数分布、聚类连贯性和嵌入维度可信度评估两模型；结果显示在局部尺度上两模型相似，但在全局尺度上图模型展现更均匀的结构和更好的聚类连贯性，且图嵌入在 <100 维时即可达到高可信度。

**⚠️ 局限性**

局限性包括仅在单一法语语料与单一模型（CamemBERT）上验证，未探讨多语言、多语料或其他语言模型的泛化性。

---

## 356. AI Sovereignty: A Qualitative Model of Strategic Competition as AI Becomes an Instrument of National Power

**arXiv ID:** 2606.07245 | [PDF](https://arxiv.org/pdf/2606.07245v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 357. Seeing Without Exposing: Adaptive Privacy Control for Open-World, Context-Hungry MLLMs

**arXiv ID:** 2606.07175 | [PDF](https://arxiv.org/pdf/2606.07175v1)

**作者:** Siyuan Xu `[一作]` (City University of Hong Kong), Sam Kwong `[通讯]` (Lingnan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无训练的Anchored Privacy Drifting (APD) 方法，能够在多模态大语言模型场景下对图像进行隐私安全化，同时保持上下文一致性，并构建了覆盖22类隐私的AdaptShield评测基准。

**💡 创新点**

创新点在于将隐私保护视为潜在空间轨迹控制，通过语义漂移和源引导双向向量场动态组合，实现对敏感内容的语义漂移同时保持源上下文；以及提供统一评测 F1-privacy 指标的全面基准。

**🔧 技术方法**

使用多模态扩散变换器 (MMDiT) 进行潜在空间编码，利用文本条件驱动的语义漂移向量、源锚定向量，并在解码阶段实现隐私安全化；评估时采用 Qwen3、Qwen2.5、InternVL3 等 MLLM 进行文本与视觉一致性评分。

**📊 数据集**

基准数据集 AdaptShield 包含 32,491 张图片，来源于 CelebAMask-HQ 和 VISPR，涵盖面部、文本、复合等 22 类隐私标签。

**📈 对比分析**

与 FluxEdit、InstructPix2Pix、RIDDLE、FALCO 等方法对比，APD 在文本类别上提升约 10.4% 以上，MLLM 评测提升约 8.5%，在 F1-privacy、保护率和保真度上均取得领先。

**⚠️ 局限性**

局限性包括对超参数 η、τ 的敏感性、对极端复杂隐私场景的适应性有限，以及需要依赖扩散模型推理，导致推理成本较高；此外，评估仍以 MLLM 为主，可能受模型偏差影响。

---

## 358. Test-Time Trajectory Optimization for Autonomous Driving

**arXiv ID:** 2606.07170 | [PDF](https://arxiv.org/pdf/2606.07170v1)

**作者:** Yihong Xu `[一作]` (valeo.ai), Matthieu Cord `[通讯]` (valeo.ai)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对端到端驾驶规划器的轨迹评分器做了测试时搜索，将评分器视为可优化的奖励函数，使用冻结的评分器在轨迹控制空间进行交叉熵搜索，显著提升已有规划器性能。

**💡 创新点**

创新点在于：①把传统仅用于一次排序的评分器转化为可优化的奖励函数；②提出无训练、无模型更新的测试时CEM搜索框架；③发现仅当评分器能在提议分布外保持准确性时搜索才有效，从而揭示固定词表评分器的隐性缺陷。

**🔧 技术方法**

使用了交叉熵法（CEM）进行采样优化；基于双轮车模型（Bicycle Model）在控制空间采样轨迹；引入锚点与舒适度正则化的可信区搜索；对现有六个基准规划器进行封装，保持评分器冻结。

**📊 数据集**

在三个主流仿真数据集上验证：NAVSIM‑v1、NAVSIM‑v2（pseudo‑closed‑loop）以及闭环模拟器 HUGSIM（包含 4 个不同真实地图）。

**📈 对比分析**

与六个基准规划器（iPad、Hydra‑MDP、GTRS、ZTRS、RAP、DrivoR）对比，平均提升了 15‑20% 的性能，其中最弱的 iPad 提升 43.6%，最强的 DrivoR 仅 3.1%；在 NAVSIM‑v2 上实现 56.3 EPDMS，几乎与使用真实感知的 PDM‑Closed（56.6）持平；在 HUGSIM 上的安全子指标和舒适度显著提升，驾驶分数上升明显。

**⚠️ 局限性**

限制主要包括：①搜索需要基准规划器的提议作为热启动，若没有可直接从零开始搜索仍是挑战；②成功搜索高度依赖评分器在提议分布外的泛化能力，当前仅 DrivoR 的解耦评分器表现良好；③对评分器的感知网络会导致额外的前向推理开销，除非与基准规划器共享同一特征提取器。

---

## 359. Reconstructing Multi-Decadal Forest Disturbances: A Spatio-Temporal Transformer Approach

**arXiv ID:** 2606.07249 | [PDF](https://arxiv.org/pdf/2606.07249v1)

**作者:** Linus Scheibenreif `[一作]` (ETH Zurich), Maxim Neumann `[通讯]` (Google DeepMind)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用多模态时间-空间视觉 Transformer（MTSViT）对 1985‑2022 年的 Landsat、Sentinel‑1/2 影像序列进行森林扰动检测，生成连续的扰动地图；

**💡 创新点**

首次在同一模型中同时建模时间演化与空间邻域，利用视觉 Transformer 的注意力机制抑制弱监督噪声，提升空间连贯性，并提出全新的手工标注验证集；

**🔧 技术方法**

基于多模态 MTSViT 网络，采用 Transformer 编码器/解码器、跨模态注意力、交叉熵二分类损失以及 ±1 年容忍的精度评估；

**📊 数据集**

使用 Landsat 5/7/8/9 30 m 1985‑2022 年全美时间序列、Sentinel‑1/2 2017‑2022 年数据；训练集为弱标签 P34k（3.4 万点）与 C100k（10 万点）两种分布；验证集为 300 点 CONUS 手工标注数据和 706 点 MTBS 火灾边界；

**📈 对比分析**

与 LandTrendr/CCDC/Qiu/Hansen 等基线相比，F1 率提升 5–9%，±1 年精度高达 92.5%，在 2017‑2022 时段使用 Sentinel‑2+S1 组合时实现最高 F1 75.8%；

**⚠️ 局限性**

对地理分布变化敏感，需要在多区域均衡采样的训练数据；目前仅做二分类，无法区分扰动驱动；弱监督标签噪声仍对模型训练产生影响。

---

## 360. Beyond Waypoints: A Trajectory-Centric Waypointing Paradigm for Vision-Language Navigation

**arXiv ID:** 2606.07244 | [PDF](https://arxiv.org/pdf/2606.07244v1)

**作者:** Haoxiang Shi `[一作]` (Harbin Institute of Technology), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了以可执行轨迹为子目标的轨迹中心化 waypoint 范式，并设计了 TSDF 引导的扩散预测器与轨迹增强导航器，实现了高层语义规划与低层执行的紧耦合。

**💡 创新点**

核心创新在于：① 用连续轨迹取代离散 waypoint，天然保证子目标可达性；② 通过 TSDF 引导的扩散过程在生成时实时避障；③ 在导航器中引入轨迹信息的拓扑-度量混合地图，使得规划能同时考虑路径几何与语义一致性。

**🔧 技术方法**

技术手段包括：双流视觉-深度编码（DINOv3 + ResNet-50）+ Transformer 场景编码；TSDF 基于深度的碰撞成本与梯度引导的扩散策略；多样化轨迹采样；基于图注意力与自注意力的跨模态 Transformer；离线预训练 + DAgger 在线微调。

**📊 数据集**

训练数据：Matterport3D (61 站点) 与 Habitat-Matterport3D (794 站点)，共 12 万轨迹对；评测数据：VLN‑CE benchmark、R2R‑CE（Val‑Seen / Val‑Unseen）以及标准 R2R。

**📈 对比分析**

在 VLN‑CE Val‑Unseen 上，%Open 达 95.84，d_c 0.54，显著优于所有基线；在 R2R‑CE Val‑Unseen 上，OSR 68.1、SR 60.3、SPL 51.4，超越最新 VLM/传统 waypoint 基线；消融实验验证 TSDF 引导、TWP、TEN 对性能的关键贡献。

**⚠️ 局限性**

局限性包括：① 依赖 TSDF 约束，若深度传感失真会影响轨迹安全；② 扩散采样与 TSDF 引导成本较高，推理时间相对较长；③ 对极其复杂或动态环境的泛化尚待验证；④ 过多的引导步骤可能过度约束生成分布，导致性能下降。

---

## 361. No, Cake Cutting Really is a Piece of Cake

**arXiv ID:** 2606.07238 | [PDF](https://arxiv.org/pdf/2606.07238v1)

**作者:** Stephen Arndt `[一作]` (Carnegie Mellon University), Kirk Pruhs `[通讯]` (University of Pittsburgh)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并分析了一种确定性蛋糕切分算法，能够以线性数量的切割实现比例公平分配

**💡 创新点**

首次证明存在确定性算法可以用O(n)次切割完成比例公平分配，打破了长期以来的O(n log n)上界假设，并提出了Late‑Early算法和“存活性”性质

**🔧 技术方法**

采用 Robertson‑Webb 模型下的 Cut 与 Eval 查询，利用双倍搜索、递归划分以及“早/晚”玩家排序，构建递归子问题保证“存活性”

**📊 数据集**

本工作没有使用任何实际数据集，所有结果均为理论分析和递归公式的计算

**📈 对比分析**

算法在最坏情况下切割次数满足 C(n)≤O(n)，评估查询次数为 E(n)≤O(n²)，相较于之前的 O(n log n) 切割上界取得显著改进；实验未给出，但理论复杂度已明确

**⚠️ 局限性**

主要限制在于评估查询数量高达 O(n²)，实现复杂且对评估成本不敏感；关于评估与切割的最优权衡及随机化算法的期望查询量仍为开放问题

---

## 362. DEFINED: A Data-Efficient Computational Framework for Fine-Grained Creativity Assessment in Debate Scenarios

**arXiv ID:** 2606.07226 | [PDF](https://arxiv.org/pdf/2606.07226v1)

**作者:** Tongzhou Yu `[一作]` (Nanjing University), Jiajun Guo `[通讯]` (East China Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `67630363-6be0-4f51-ab05-7198250671a5`

**🎯 论文内容**

开发了 DEFINED，一个数据高效的多维度辩论创造力评估框架，通过混合粒度训练和三重约束数据增强，实现细粒度创造力评估与整体评分。

**💡 创新点**

① 设计八维创造力指标体系，将创造性与非创造性维度分离；② 采用混合粒度训练策略和有限精细标签，解决少量数据问题；③ 通过三重约束数据增强消除专业化偏差，并构建可解释的层次评分头。

**🔧 技术方法**

基于预训练自回归LLM Qwen2.5‑7B‑Instruct 的语义编码器，层次化评分头，混合粒度损失函数；使用 LoRA 进行参数高效微调；三重约束生成与对比排序校准。

**📊 数据集**

真实的中国高校辩论比赛语料（56条议题、706个句子、约3.26M中文字符）作为高水平样本；通过三重约束生成合成中低水平样本；细粒度数据由10名研究生专家标注120条。

**📈 对比分析**

与 Gemini‑2.5‑Pro、GPT‑4o、Qwen3、DeepSeek、Themis、M‑Prometheus 等 LLM 评估器以及 Debatrix、InspireDebate 进行对比；细粒度指标上 PCC 0.96、MSE 43；高水平样本整体评分 MSE 18；中低水平对比准确率 95%；总体性能显著优于基线。

**⚠️ 局限性**

仍受限于专业化数据偏差与跨域泛化，生成合成样本可能产生风格或语义偏差；仅在辩论情境验证，缺乏跨任务测试；需要进一步探索任务无关创造力特征和更大规模数据。

---

## 363. A Comparative Study of Deep Learning Models for Geological Carbon Sequestration

**arXiv ID:** 2606.07215 | [PDF](https://arxiv.org/pdf/2606.07215v1)

**作者:** Giovanni Zingaro `[一作]` (University of Waterloo), Yuri Leonenko `[通讯]` (University of Waterloo)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对五种深度学习架构（Temporal CNN、U-Net、V-Net、FNO、U-FNO）在地质碳封存中 CO₂ 饱和度和压降场进行训练与评估。

**💡 创新点**

在同一数据集上对不同架构进行公平比较，结合全梯度损失和 U‑FNO 改进，阐明不同 PDE 类型对架构选择的影响。

**🔧 技术方法**

使用深度卷积网络、Fourier 神经算子、U‑Net 增强 FNO、改进的全梯度损失函数、FFT、Adam、Cosine Annealing Warm Restart 等技术。

**📊 数据集**

使用 Wen 等人生成的 5,500 例二维轴对称多相流数据集，包含异质渗透率、孔隙率、注入参数等。

**📈 对比分析**

通过 RMSE、R²、PSD、参数量、训练时长、显存等指标进行比较；CO₂ 饱和度场 U‑FNO 最优（R²≈0.967），压降场 FNO 最优（R²≈0.972）。

**⚠️ 局限性**

仅限单井二维静态场景；不含断续注入、三维或热力耦合；仅基于单一数值模拟器；未评估逆问题或实时优化。

---

## 364. RISE: A Rust Library for Inverted Index Search Engines

**arXiv ID:** 2606.07187 | [PDF](https://arxiv.org/pdf/2606.07187v1)

**作者:** Angelo Savino `[一作]` (University of Pisa), Rossano Venturini `[通讯]` (University of Pisa)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个用 Rust 编写的完整倒排索引库，提供可插拔的压缩器、查询算子和评分函数，支持多种主流压缩编码和 Top‑k 查询算法。

**💡 创新点**

创新点在于：①将完整的倒排索引堆栈以 Rust trait 方式实现，既保持高性能又易于扩展；②在同一测试平台上重现并验证多篇经典文献中的压缩与查询算法；③与成熟 C++ 库相比，单线程查询加速可达 2×，同时压缩率不低于竞争对手。

**🔧 技术方法**

使用了 Rust 语言、trait 与泛型系统；压缩技术包括 Stream VByte、Elias‑Fano、Partitioned Elias‑Fano（含全量序列补码）、Binary Interpolative Coding；查询算法涵盖 R‑AND/R‑OR、WAND、MaxScore 及其 Block‑Max 变体（固定/可变块），其中 BMMS 采用双窗口改进。

**📊 数据集**

实验数据集为 ClueWeb 2009 TREC Category B（约 5 千万文档）和 CommonCrawl 新闻集；对索引按随机、URL 排序和递归图切分三种文档顺序进行评测。

**📈 对比分析**

方法：在同一台 20 核 Intel Core Ultra 7 机器上，单线程加载完整索引，使用 128 GB 内存，分别对本库、Lucene、Tantivy 等 C++ 基线进行平均查询时长和索引大小（GiB）测量；结果显示本库在三大压缩器下均能匹配或超越 C++ 基线，最快情况下查询加速达到 2×，压缩率保持在相近水平。

**⚠️ 局限性**

局限性：目前仅实现了部分算法（如 BMMS 的变体缺失）、未加入 PForDelta、Roaring 位图等高级压缩；实验仅在单线程环境下进行，缺乏多线程并发性能评估；对学习型稀疏检索数据集的支持尚未实现。

---

## 365. AdaTok: Self-Budgeting Image Tokenization with Quality-Preserving Dynamic Tokens

**arXiv ID:** 2606.07185 | [PDF](https://arxiv.org/pdf/2606.07185v1)

**作者:** Xiaocheng Lu `[一作]` (Hong Kong University of Science and Technology), Song Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了 AdaTok，一种自预算离散 1D 图像 tokenizer，能够在单次前向传递中根据输入内容动态决定所需的 token 数量并支持可变长度重构。

**💡 创新点**

创新点在于将优先表示学习（Nested Tail Masking + Multi-Head LoRA）与 deterministic‑group GRPO 及 Dynamic Pareto Weighting 相结合，构建了无需搜索、外部预测或目标质量调节即可完成的完整自预算系统。

**🔧 技术方法**

主要技术包括 Nested Tail Masking、Multi‑Head LoRA、Group Relative Policy Optimization (GRPO) 与 Dynamic Pareto Weighting，以及 TiTok‑S 结构的 1D 编码器/解码器。

**📊 数据集**

实验使用 ImageNet‑1K（256×256）数据集。

**📈 对比分析**

与固定预算离散 tokenizer（TiTok、One‑D‑Piece、FlexTok 等）以及可变预算基线（ElasticTok、Semanticist 等）进行比较；AdaTok‑Full 在 256 token 下 rFID 为 1.31，AdaTok‑Adaptive 在约 118 token 下 rFID 为 1.50，生成质量 gFID 为 2.28，且自回归生成平均 token 118 时速度提升约 2.1×，显著优于固定预算基线。

**⚠️ 局限性**

局限性包括训练过程需两阶段分离，生成器需单独训练，离散词表大小和解码器预算特化限制了极低 token 数的鲁棒性，且缺乏端到端自动停止机制。

---

## 366. OPTIMUS-Prime: Minimal and Sufficient Concept Explanations for Deep Vision Models

**arXiv ID:** 2606.07180 | [PDF](https://arxiv.org/pdf/2606.07180v1)

**作者:** Arthur Hoarau `[一作]` (Université de Lorraine, CentraleSupélec), Vu Linh Nguyen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 OPTIMUS 框架，利用 prime implicants 在深度视觉模型的概念空间中挑选最小且足够的概念子集，并将其投影为可视化热图，以提供理论上可证明的解释。

**💡 创新点**

创新点在于：① 将 Boolean 逻辑中的 prime implicants 概念迁移到神经网络概念层，确保解释既满足充分性又满足最小性；② 开发了直接搜索和贪婪 LowGain/MaxMin 两种高效算法；③ 在保持解释可解释性的同时，将理论保证与输入空间的梯度归因方法（IG、DeepLIFT）相结合。

**🔧 技术方法**

使用的技术包括：概念层定义与线性分类层假设、prime implicants 计算与直接搜索算法、贪婪 LowGain/MaxMin 搜索策略、Integrated Gradients 与 DeepLIFT 归因方法用于将概念映射回输入像素。

**📊 数据集**

实验数据集为 Cat-Dog-Bird 三分类图像数据集，采用 70/15/15 的训练/验证/测试划分。

**📈 对比分析**

与全概念热图和无约束贪婪基线比较，LowGain 在解释尺寸和运行时间上取得最佳折中，PI 大小接近 MaxMin 而运行更快；实验中展示了 OPTIMUS 解释在保留模型预测的同时显著去除了不必要概念。

**⚠️ 局限性**

局限性包括：① 目标层必须是线性且前一层输出需在 [0,1] 范围内；② 对模型结构仍有一定依赖，无法完全做到全黑盒；③ 梯度归因方法对基线选择和输入噪声敏感；④ 概念可能高度重叠或抽象，导致可解释性受限。

---

## 367. TraRA: Trajectory-level Recognition Aggregation for Video Text Spotting in Urban Surveillance

**arXiv ID:** 2606.07161 | [PDF](https://arxiv.org/pdf/2606.07161v1)

**作者:** Duc Tri Tran `[一作]` (RIKEN), Yasutomo Kawanishi `[通讯]` (RIKEN)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 TraRA 插件，结合轨迹聚类与视觉语言聚合实现视频文本识别的轨迹级提升。

**💡 创新点**

创新点在于将轨迹级聚类与低秩适配增强的视觉语言模型相结合，实现跨帧信息聚合与鲁棒识别。

**🔧 技术方法**

使用 Temporal Clustering、Vision‑Language Aggregation（基于 LoRA 的 Ovis2.5 VLM、SigLIP2 + Qwen3）、多模态特征提取和聚类算法。

**📊 数据集**

在 ArTVideo、ICDAR15、RoadText、BOVText 四个公开 VTS 基准数据集上进行评估。

**📈 对比分析**

与 GoMatching++ 与 TransDETR 等主流方法对比，在识别准确率、MOTA、ID_F1 等指标上均显著提升，尤其在弱基线上提升幅度更大。

**⚠️ 局限性**

TC 模块缺乏长期记忆，VLA 模型计算量大，导致实时性受限。

---

## 368. Textual Supervision Enhances Geospatial Representations in Vision-Language Models

**arXiv ID:** 2606.07172 | [PDF](https://arxiv.org/pdf/2606.07172v1)

**作者:** Marcelo Sartori Locatelli `[一作]` (Max Planck Institute for Security and Privacy), Meeyoung Cha `[通讯]` (Max Planck Institute for Security and Privacy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究分析了三种模型家族（仅视觉架构、视觉-语言模型和大规模多模态基础模型）所获得的地理空间表示，发现文本监督可以增强地理空间表示的学习。

**💡 创新点**

创新点在于揭示了语言作为有效的补充模态在编码空间上下文中的作用，并强调了多模态学习在推进地理空间人工智能中的关键方向。

**🔧 技术方法**

使用了线性探测和岭回归等技术来评估模型的地理位置预测性能。

**📊 数据集**

使用了YFCC100M和Google Landmarks等数据集，进行图像聚类以分析可定位性。

**📈 对比分析**

通过层级探测比较不同模型的表现，发现视觉-语言模型在早期层的地理空间表示更强，而仅视觉模型在最后一层表现更好。视觉-语言模型的平均R^2值高于0.4，而仅视觉模型主要低于0.3。

**⚠️ 局限性**

限制在于探测设置使用均方误差作为损失函数，可能影响结果的透明度；此外，研究依赖于预训练模型，无法控制预训练数据集的选择，可能影响结果。

---

## 369. MailoHLS: Multi-Adapter Structure-Aware Learning for Pareto-Driven HLS Pragma Optimization

**arXiv ID:** 2606.07246 | [PDF](https://arxiv.org/pdf/2606.07246v1)

**作者:** Elena Vouvali `[一作]` (National Technical University of Athens), Sotirios Xydis `[通讯]` (National Technical University of Athens)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出 MailoHLS，一种将 LLM 与 GNN 结合的混合框架，用于指令级的 HLS pragma 优化。

**💡 创新点**

创新点在于跨模态的结构-语义联合建模、指令占位符的有序预测、基于 Pareto 的目标感知 LoRA 适配器与 DPO 优化。

**🔧 技术方法**

采用 Transformer + LoRA 轻量化微调、跨注意力融合 GNN 结构嵌入、图神经网络对 LLVM IR 的结构编码、Pareto 路径偏好学习。

**📊 数据集**

使用 GNΩSIS 大规模 HLS 设计点数据集（约 26k 个在 ZCU104 上的样本）进行训练与评估。

**📈 对比分析**

与无指令基线、CollectiveHLS、GPT‑4o、Claude Haiku、Gemini 及 LIFT 等方法对比，MailoHLS 在已知内核上平均 9.48× 延迟加速、2.10× 资源加速，未见内核上 4.97× 延迟加速、5.21× 平衡加速，整体接近 Pareto 前沿，优于现有 LLM 与数据驱动方案。

**⚠️ 局限性**

局限性包括：在全新核/家族上仍存在性能差距；对极大内核或多核架构的泛化尚未充分验证；依赖 GNΩSIS 数据集的质量与覆盖；模型训练与推理仍需一定算力。

---

## 370. Generative Molecular Morphing for Flexible-Size Design via Unbalanced Optimal Transport

**arXiv ID:** 2606.07239 | [PDF](https://arxiv.org/pdf/2606.07239v1)

**作者:** Malte Franke `[一作]` (ETH Zürich), Andreas Krause `[通讯]` (ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种灵活大小的3D几何图形生成模型，通过跳跃-流动过程实现节点的插入、删除和替换，以生成满足特定属性或结构先验的分子。

**💡 创新点**

创新点在于：①将可变长度生成从离散序列迁移到连续空间的几何图形；②利用无平衡最优传输(UOT)实现不同大小图形的匹配与插值；③在条件生成框架下引入连续时间马尔可夫链（CTMC）实现多种离散结构变更；④实现了在训练分布之外（OOD）对分子尺寸和属性的精准引导。

**🔧 技术方法**

核心技术包括：条件生成匹配（Conditional Generator Matching），无平衡最优传输，流匹配（flow matching），连续时间马尔可夫链，以及基于Kabsch算法的SE(3)对齐。

**📊 数据集**

主要使用公开数据集 QM9 和 GEOM-Drugs，针对小分子和药物类分子进行实验。

**📈 对比分析**

与当前固定大小的流式/扩散模型（如 FlowMol、MiDi、EQGAT‑diff、SemlaFlow）比较：在 QM9 上可获得与之相当甚至更优的有效率、唯一性和能量指标；在 GEOM‑Drugs 上保持较高的 PoseBusters 分数；通过对尺寸、logP 等属性的条件生成，展示了在训练分布外的高有效率和可调节性；在 scaffold 装饰任务中，首次实现了基于结构先验的灵活修饰，显著提高了结构保留和有效率。

**⚠️ 局限性**

局限性包括：训练成本更高（需要更多 epoch 以及更复杂的损失平衡）；需要较多的积分步长，难以实现少步采样；实验主要集中在 QM9，尚未充分验证在更大规模、真实药物化学空间的可扩展性；对离散时间步长和采样效率的进一步权衡仍需研究。

---

## 371. DualGate-Net: A Prior-Gated Dual-Encoder Framework for Histopathology Cell Detection

**arXiv ID:** 2606.07222 | [PDF](https://arxiv.org/pdf/2606.07222v1)

**作者:** Bahman Jafari Tabaghsar `[一作]` (Deakin University), Atul Sajjanhar `[通讯]` (Deakin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于双编码器与可学习先验门控融合的细胞检测框架DualGate-Net，结合局部CNN与全局Transformer；

**💡 创新点**

创新点在于引入先验门控融合模块自适应调节组织先验影响，并使用辅助前景重建与Cellness引导提升定位鲁棒性；

**🔧 技术方法**

采用ConvNeXtV2做局部编码器、SegFormer做全局编码器、CBAM注意力融合、前景重建分支和Cellness门控技术；

**📊 数据集**

在OCELOT（细胞与组织配对）和BRCA（多类细胞）两大数据集上进行实验；

**📈 对比分析**

与官方基线及多种对手方法对比，OCELOT验证集宏F1提升至0.7722，测试集0.7345，显著优于传统融合与单编码器方法；

**⚠️ 局限性**

局限在于组织先验生成的噪声仍可能影响模型、跨域泛化能力有限，未来需引入不确定性建模与更强的域适应机制。

---

## 372. The Synthesis-Sequencing Channel for DNA-based Data Storage

**arXiv ID:** 2606.07216 | [PDF](https://arxiv.org/pdf/2606.07216v1)

**作者:** Keshav Goyal `[一作]` (Cote d'Azur University, CNRS, I3S Laboratory), Serge Kas Hanna `[通讯]` (Cote d'Azur University, CNRS, I3S Laboratory)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `67630363-6be0-4f51-ab05-7198250671a5` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并研究了 DNA 数据存储的双阶段合成-测序信道模型，并推导了其信息理论容量；

**💡 创新点**

创新点在于将物理覆盖率与测序覆盖率分离，考虑了合成阶段的系统性误差以及测序阶段的随机误差，揭示了覆盖偏差与容量之间的多重权衡；

**🔧 技术方法**

采用随机编码、联合典型性、信息熵分解和 Hoeffding 及 AEP 等信息理论工具，得到容量上界与下界；

**📊 数据集**

论文未使用具体实验数据集，而是基于理论模型和概率分布（如 Poisson、BSC）进行分析；

**📈 对比分析**

与先前的噪声绘制通道比较，证明在相同平均读取数和有效误差率下，合成-测序通道的容量低于仅测序通道，并通过数值示例展示不同合成误差率和覆盖参数对容量的影响；

**⚠️ 局限性**

局限性包括：假设误差为二元对称通道、期望有限、忽略 PCR 放大误差、未考虑非独立误差的实际生物学复杂性，以及仅给出上界与下界，缺乏实际编码方案的实现与仿真验证。

---

## 373. A Causal Probabilistic Framework for Perception-Informed Closed-Loop Simulation of Autonomous Driving

**arXiv ID:** 2606.07186 | [PDF](https://arxiv.org/pdf/2606.07186v1)

**作者:** Zhennan Fei `[一作]` (Volvo Cars), Gabriel Rodrigues de Campos `[通讯]` (Zenseact)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

提出一种基于贝叶斯网络的感知误差注入方法，将实际环境触发条件映射到摄像头感知失效，从而在软件回环（SIL）仿真中评估ADAS/ADS的鲁棒性。

**💡 创新点**

创新点在于使用因果概率模型实现物理触发到感知失效的闭环映射，既保留了对象级仿真的高效性，又能捕捉真实感知失效的统计特征。

**🔧 技术方法**

采用贝叶斯网络（Bayesian Network）、OpenSCENARIO/OpenDRIVE、esmini仿真引擎、OSI接口与Python实现的失败注入层。

**📊 数据集**

使用的是仿真生成的标准OpenSCENARIO/OpenDRIVE数据集，未使用公开真实感知数据集；实验基于三种场景（黑暗、雾雨、切入）进行。

**📈 对比分析**

通过与理想传感器（ideal sensing）对比，展示了误检、尺寸误差、合并误差对车辆纵向控制的影响；在黑暗和雾雨场景下ADAS保持基本稳定，而切入场景中合并误差导致制动误判。

**⚠️ 局限性**

局限性包括：仅针对摄像头；缺乏对真实传感器数据的校准；模型未覆盖眩光、遮挡等其他环境扰动；仅在仿真中验证，缺乏实车验证。

---

## 374. EvoGS: Constructing Continuous-Layered Gaussian Splatting with Evolution Tree for Scalable 3D Streaming

**arXiv ID:** 2606.07179 | [PDF](https://arxiv.org/pdf/2606.07179v1)

**作者:** Yuang Shi `[一作]` (National University of Singapore), Wei Tsang Ooi `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于演化树的连续层次3D高斯喷射（EvoGS）表示，用于可扩展的3D内容流式传输。

**💡 创新点**

通过在连续层次中显式构建父子子代关系，消除离散层次的错误累积与冗余；采用波形分解启发的不对称残差提升可压缩性和视觉连续性。

**🔧 技术方法**

使用3D Gaussian Splats、波形分解启发的协线残差、渐进式训练、可自适应密度化、层级压缩与量化等技术。

**📊 数据集**

在Blender、Mip-NeRF360、Tanks & Temples和Deep Blending四大基准数据集上进行评估。

**📈 对比分析**

与Monolithic、Single、LapisGS、L3GS以及对称版本对照；EvoGS在PSNR/SSIM/LPIPS上平均提升0.3–0.7 dB，存储和GPU内存分别比LapisGS/​L3GS减少约2.5×/5.5×，经压缩后仅占LapisGS的≈11.5%，且质量随传输量平滑递增。

**⚠️ 局限性**

目前仅针对静态场景；演化树构建与训练成本较高；不保证对极端动态或大规模场景的可扩展性。

---

## 375. Distributed Persistence Domain for Persistent Memory Pooling

**arXiv ID:** 2606.07159 | [PDF](https://arxiv.org/pdf/2606.07159v1)

**作者:** Khan Shaikhul Hadi `[一作]` (University of Central Florida), Yan Solihin `[通讯]` (University of Central Florida)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了分布式持久性域（DPD）以及具备持久化缓冲的CXL交换机，实现了分布式持久化内存池的低延迟、读转发和写合并功能。

**💡 创新点**

创新点在于将持久化功能迁移至网络层，提出DPD概念并给出多路径持久化域的正确性保证与软件支持机制，以及自适应的Drain阈值策略。

**🔧 技术方法**

技术手段包括CXL协议、可持久化缓冲区设计、Persist Buffer Controller（PBC）与Selector（PBCS）逻辑、Adaptive Drain Threshold算法，以及gem5+CACTI仿真平台。

**📊 数据集**

使用的工作负载为Splash‑4基准套件和YCSB（memcached）数据集。

**📈 对比分析**

通过与无持久化交换机基线（NoPB）对比，采用Eager、Lazy和Adaptive CP三种方案，平均提升约33%，最优情况下提升131%，同时评估了写失败率、读命中率、写合并率和吞吐量等指标。

**⚠️ 局限性**

局限性包括：实验仅覆盖单跳或有限跳CXL拓扑，写失败率和读命中率高度依赖具体工作负载；未考虑功耗与持久化缓冲区可靠性；未在大规模多机架系统中验证扩展性。

---

## 376. An Abstract Architecture for Explainable Autonomy in Hazardous Environments

**arXiv ID:** 2606.07211 | [PDF](https://arxiv.org/pdf/2606.07211v1)

**作者:** Matt Luckcuck `[一作]` (Maynooth University), Marie Farrell `[通讯]` (Maynooth University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出一种抽象的可解释自治架构，将BDI智能体与解释器整合到中央执行器中，支持对过去动作和未来行动的解释；

**💡 创新点**

创新点在于将BDI推理与解释模块集中化，提供两种解释模式（过去行动与未来不行动），并通过“干运行”机制验证安全规则；

**🔧 技术方法**

采用BDI（Belief-Desire-Intention）推理框架、逻辑推理、正式方法（如ajpf）进行验证，并在中央执行器中收集信息供解释器使用；

**📊 数据集**

未使用公开数据集；仅在核废料储存场景中做了概念性案例演示；

**📈 对比分析**

论文未进行实验比较或性能评估，主要提供架构设计与理论示例；

**⚠️ 局限性**

局限在于缺乏实际实现与实证验证，未对不同环境或规模的可解释性进行系统评估；

---

## 377. Towards Tight Bounds for Streaming Attention

**arXiv ID:** 2606.07205 | [PDF](https://arxiv.org/pdf/2606.07205v1)

**作者:** Justin Y. Chen `[一作]` (Massachusetts Institute of Technology), Boris Prokhorov `[通讯]` (Ecole Polytechnique Federale De Lausanne)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

研究了在流式设置下高效实现Transformer注意力机制，提出了软max核密度估计（Softmax KDE）的近似方法并给出了上下界

**💡 创新点**

创新点在于将多项式嵌入与核心集（coreset）结合，利用伪随机化与最近点回归，首次给出相对误差下的近似空间下界与几乎匹配的上界

**🔧 技术方法**

采用多项式（Taylor/ Hermite）特征映射、核心集压缩、伪随机化分区、Merge‑and‑Reduce流水线以及离散化的Hermite展开等技术

**📊 数据集**

实验采用合成高维高斯或球面采样的数据集，未使用公开真实数据集

**📈 对比分析**

相较于传统的随机特征映射、线性/线性注意力等基线，本文在高维或高温度（高r）场景下实现了接近下界的空间复杂度（例如高维下O(e^{r^2}/ε)），误差满足相对ε

**⚠️ 局限性**

局限性包括：算法常数与维度相关，需对稀疏或高相关性数据做特殊处理；在自适应/适配输入流时需额外的权重管理；未考虑非正态或非球面分布的情形

---

## 378. Learning Multi-Agent Communication Protocol: Study on Information Entropy Efficiency in MARL

**arXiv ID:** 2606.07200 | [PDF](https://arxiv.org/pdf/2606.07200v1)

**作者:** Xinren Zhang `[一作]` (Hong Kong University of Science and Technology), Jiadong Yu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d`

**🎯 论文内容**

本文提出了信息熵效率指数（IEI）来衡量多智能体通信协议的效率，并将其嵌入训练损失，促使代理在保持高任务性能的同时实现更紧凑的通信；

**💡 创新点**

创新点在于引入了基于消息熵与成功率比值的指标，将通信效率量化并直接作为正则化目标，实现单轮通信即可达到甚至超越多轮通信的性能；

**🔧 技术方法**

主要技术包括多轮CTDE式MARL框架、注意力机制（GAT、TarMAC等）、动态权重正则化、以及信息熵计算与损失融合；

**📊 数据集**

实验使用Traffic Junction（TJ）合作导航环境，5名代理、7×7网格、20步/episode；

**📈 对比分析**

通过与五种基准算法（MAGIC、CommNet、TarMAC、GA-Comm、IC3Net）的单轮/多轮/IEI对照，发现IEI配置在保持或提升成功率（最高达99.75%）的同时，通信量降低约50%且收敛速度提升，表现出更优的性能/通信成本 Pareto 边界；

**⚠️ 局限性**

局限性包括仅在单一 TJ 场景验证，缺乏跨环境泛化评估，且 IEI 参数调优仍依赖经验，未来需探索更通用的自动调节策略。

---

## 379. Structure-Preserving Correction Learning for Sparse Bayesian Inference in Brain Source Imaging

**arXiv ID:** 2606.07196 | [PDF](https://arxiv.org/pdf/2606.07196v1)

**作者:** Marco Morik `[一作]` (Berlin Institute for Foundations of Learning and Data (BIFOLD)), Ismail Huseynov `[通讯]` (Technische Universität Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在经典的稀疏 Type‑II 贝叶斯源定位求解器（凸界限）上展开深度可展开结构，并在此基础上学习可解释的校正项（偏置、MLP、注意力残差），从而在保持 Bayesian 结构的同时提升重建精度与收敛速度。

**💡 创新点**

创新点在于：① 将贝叶斯更新规则转换为稳定的对数域表达，保证数值稳定性；② 引入逐层可训练的校正模块，既能保留原始推理机制，又能通过数据学习微调；③ 通过层级递归训练与深度监督，显著加速收敛并提升最终定位质量。

**🔧 技术方法**

技术手段包括：深度可展开网络、对数域特征表示、偏置/残差 MLP 与注意力机制、深度监督正则化、端到端重建损失（rMSE/F1/EMD）。

**📊 数据集**

使用合成 EEG 数据集：在真实前向模型下生成分布式、频带过滤的神经源，加入异方差传感器噪声；同时在 THINGS‑EEG2 公开数据上进行零样本验证。

**📈 对比分析**

与 sLORETA、经典凸界限、DeepSIF 等基线相比，Bias‑CB、Deep‑CB、Deep‑Attn‑CB 在 EMD、rMSE、F1 方面均有显著提升，收敛迭代次数大幅减少，尤其是带注意力的 Deep‑Attn‑CB 在低 SNR 与多源情形下保持稳健。

**⚠️ 局限性**

局限性包括：仅针对固定方向 EEG 及对角协方差模型；实验仅基于合成数据，缺乏真实脑源地面真值；校正模块在极稀疏或不同源数分布下可能出现偏差，且未验证在 MEG、全协方差或其他稀疏回归任务中的迁移性。

---

## 380. Synthetic APTs: the Collapse of TTP-Based Attribution

**arXiv ID:** 2606.07158 | [PDF](https://arxiv.org/pdf/2606.07158v1)

**作者:** Francesco Balassone `[一作]` (Alias Robotics), MinSeok Choi `[通讯]` (PurpleAILAB)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用AI驱动的攻击者与防御者在两个网络环境中进行实验，验证AI能否忠实仿真已知APT团体的TTP，并探讨其对CTI指纹识别的影响。

**💡 创新点**

首次将大型语言模型与Cybersecurity SuperIntelligence框架结合，系统化评估AI仿真在不同情境下的指纹共融与属性差异，揭示AI操作可“模拟”任何APT团体，从而削弱基于TTP的归因。

**🔧 技术方法**

使用Anthropic Claude Opus 4.6 LLM作为攻击者，30B参数模型作为防御者，CSI多智能体架构，Wazuh/Velociraptor/Elasticsearch等防御工具，以及自动化攻击与检测日志分析。

**📊 数据集**

两套CYBER RANGES模拟环境：企业网络（约20台主机）和军事双机构关键基础设施（约15台主机）；利用MITRE ATT&CK官方APT团体资料构建攻击者角色提示。

**📈 对比分析**

在20次实验中，所有企业场景被完全攻陷，所有军事场景被成功防御；在企业情境下，观察到55–80%精度匹配官方APT TTP，且检测时间从几分钟到超过2小时不等；实验显示网络拓扑是决定成败的关键。

**⚠️ 局限性**

仅评估5个APT团体，单一攻击者模型和单一框架；实验在预置防御者提前30分钟的理想化设定下进行；范围内缺乏真实多样化攻击手段（如钓鱼、加密C2）及对更大规模模型的验证。

---

## 381. On the Shoulders of Giants: Empowering Automated Smart Contract Auditing via the GiAnt Corpus

**arXiv ID:** 2606.07363 | [PDF](https://arxiv.org/pdf/2606.07363v1)

**作者:** Xiaoting Zhang `[一作]` (Zhejiang University), Xin Xia `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套自动化框架（GPT-assisted Auditing Dataset Construction），利用大型语言模型从 Code4rena 的审计报告中分层提取漏洞信息，并通过 LLM‑as‑a‑Judge 机制进行质量审核，最终生成了 GiAnt Corpus 这份高质量、多维度的智能合约审计数据集。

**💡 创新点**

创新点主要包括：① 在分层递归（divide‑and‑conquer）+ 生成式链式思考（Chain‑of‑Thought）框架下完成多维度信息提取；② 引入 LLM‑as‑a‑Judge 的自检自修循环，显著降低 hallucination 并提升数据可靠性；③ 通过细粒度的严重度划分和完整的漏洞信息（代码、描述、影响、缓解、PoC、Gas 优化）构建了迄今规模最大、最细粒度的审计数据集。

**🔧 技术方法**

使用的技术包括：大型语言模型（GPT‑4o、GPT‑3.5‑turbo、Qwen3‑Coder‑Instruct、DeepSeek‑V3）；Chain‑of‑Thought 与 In‑Context Learning；正则分层解析、PyMuPDF 文档转换；LLM‑as‑a‑Judge 质量评估；代码映射与语义对齐评分；以及基于 BERTScore 与人工 LLM‑judge 的多任务评测。

**📊 数据集**

数据集：从 Code4rena 公共审计报告（388 份，覆盖 340 个项目）提取，构建 GiAnt Corpus，包含 7,711 条漏洞条目，按高、中、低、非关键、Gas 优化五级严重度划分，且每条记录包含代码、描述、影响、缓解、PoC 等信息。

**📈 对比分析**

比较方法：在四个任务（漏洞检测、代码总结、缓解建议、Gas 优化）上采用零样本提示，对四大 LLM 进行评测。检测二分类 F1 最高 0.893，三分类 F1 仅 0.28；代码总结 BERTScore >0.733 但 LLM‑judge 低于 3；缓解建议 BERTScore ≥0.769、LLM‑judge ≥3.64；Gas 优化 BERTScore >0.715、LLM‑judge ≈3。总体来看，LLM 在基本功能上具备一定能力，但在专业深度、稳定性和精度上仍有显著提升空间。

**⚠️ 局限性**

局限性：① 数据来源仅为 Code4rena，可能不完全代表其他审计平台的报告结构；② 依赖特定 LLM（GPT‑4o）但架构可替换，仍受模型生成偏差与 hallucination 影响；③ 对复杂业务逻辑的推断仍偏弱，导致多分类和专业评估表现不佳；④ 在代码映射与语义对齐中仍存在误检/漏检，需进一步完善。

---

## 382. How Far Can Chord-Symbol Time-Series Adaptation Carry Genre Identity? Capabilities and Boundaries in Multi-Genre Chord-Symbol Modeling

**arXiv ID:** 2606.07334 | [PDF](https://arxiv.org/pdf/2606.07334v1)

**作者:** Jinju Lee `[一作]` `[通讯]` (PearlLeeStudio), Jinju Lee (PearlLeeStudio)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在冻结的 pop‑jazz Music Transformer 基础上，使用 LoRA、IA3、BitFit、Prefix tuning、全微调以及控制词基线，对 11 个目标流派（blues、bossa nova、Bach chorales、country、electronic、folk、funk、gospel、hip‑hop、R&B/soul、rock）的和弦符号序列进行流派适配，并在 held‑out 测试上评估下一和弦预测性能。

**💡 创新点**

通过完整的 165 细胞网格（5 方法×11 流派×3 种子）结合多重诊断（错误流派旋转、控制词对比、LoRA rank sweep、生成分布统计、Bach chorale 对照、数据量匹配等），阐明和弦符号层携带的流派信息边界，而非仅给出 adapter 排名；揭示控制词与小型 adapter 在性能上相近，表明和弦符号层本身具有可调节的通用和弦知识。

**🔧 技术方法**

采用 Music Transformer 的相对位置注意力架构与冻结 pop‑jazz checkpoint；使用 LoRA、IA3、BitFit、Prefix tuning、全微调等参数高效微调方法；通过交叉熵、top‑1/5 精度评估；Wilcoxon signed‑rank 与 Holm/BH 多重校正；LoRA rank 选择与验证；错误流派旋转；生成输出统计（KL、entropy、重复率）以及真实歌曲、近似重复检测等诊断。

**📊 数据集**

基于 Chordonomicon（涵盖 666k 歌曲 chord transcriptions）与 public music21 Bach chorales，按流派划分得到 11 个子集；训练集规模差异超过 2 个数量级（296–152,509 序列）；使用 3 种随机种子；对 10 个流派做匹配大小子采样；真实歌曲子集 10 首/流派用于评估。

**📈 对比分析**

对比 5 种 PEFT 与控制词基线，在 held‑out 测试上所有方法均优于冻结基线，宏观 top‑1 提升 2.89–3.61 pp；LoRA 与 IA3 在宏观和流派获胜计数上最高，但校正后无显著差异；控制词基线仅落后 0.3–0.5 pp；匹配数据量后 IA3 仍领先，LoRA 性能下降至最末；整体方法差异小于 1 pp，表明表现受数据量与层边界限制。

**⚠️ 局限性**

评估仅为自动指标（top‑1/5 与真实进程对比），缺乏音乐质量与感知验证；数据高度重复导致可推广性受限；流派标签粗糙且 Bach chorale 为极端异类；和弦符号不包含节奏、音色、乐句等关键层；tokenization 较粗，结果受分词方案影响；未与基线 n‑gram/Markov 模型比较；未使用独立预训练基线对照；未充分测试模型在自由生成时的解码异常；因此结论仅限于可解释的和弦层边界。

---

## 383. AnchorWorld: Embodied Egocentric World Simulation with View-based Evolution Customization

**arXiv ID:** 2606.07326 | [PDF](https://arxiv.org/pdf/2606.07326v1)

**作者:** Yu Li `[一作]` (Tsinghua University), Yujiu Yang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AnchorWorld 框架，实现基于 3D 人体运动的 egocentric 视频合成，并支持通过姿态关联锚视图和文本描述进行可定制化的自演化世界。

**💡 创新点**

创新点包括：①利用第三视角视频提供完整运动监督，解决第一视角中身体部分缺失的问题；②采用投影‑基动作控制实现混合视角训练，增强空间姿态感知；③通过姿态关联锚视图与跨注意力掩码实现局部场景的可定制化演化，兼顾视觉一致性与动态变化。

**🔧 技术方法**

技术手段包括：Wan2.2 及 DiT 流匹配视频生成模型；混合视角投影动作编码与空间姿态注意力；锚视图的内联条件注入与 3D RoPE；文本驱动的跨注意力与遮蔽机制；分阶段多模训练策略。

**📊 数据集**

数据集涵盖：200K 单人动作视频 + 101K UE 多摄像头数据（外观视角）；Ego‑Exo4D 与 LEMMA（同步第三/第一视角）用于姿态与锚视图标注；内部 200K 跨视角训练集；UE 与真实世界测试集用于评估泛化。

**📈 对比分析**

与 PlayerOne、CaM 等基线在动作准确度（ATE、RRE）、场景一致性（GIM、CLIP‑V、PSNR/SSIM、LPIPS）以及文本对齐（TA）等指标上均实现了最优或竞争性性能，尤其在大视角变化和动态演化场景中表现突出。

**⚠️ 局限性**

局限性包括：对长时序探索与开放世界泛化的能力不足；手部动作估计不稳定，导致无法完整支持全手部控制；在极端视角或长时间动态场景下的细节与一致性仍有提升空间。

---

## 384. QBugLM: An Agentic Benchmarking Framework for LLM-based Quantum Software Debugging

**arXiv ID:** 2606.07314 | [PDF](https://arxiv.org/pdf/2606.07314v1)

**作者:** An B. B. Pham `[一作]` (University of Melbourne), Muhammad Usman `[通讯]` (Data61, CSIRO)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了QBugLM，一个多代理框架，用于自动化生成、检测、修复并验证OpenQASM 3.0量子程序中的常见错误。

**💡 创新点**

创新点在于：①基于四类量子错误的系统化注入工具QBugGen；②采用双代理（检测+修复）架构实现迭代式反馈；③在无框架依赖的量子代码上使用模拟器进行功能验证；④首次对大模型在量子调试中的表现进行基准评测。

**🔧 技术方法**

核心技术包括：大语言模型（Claude 4.6 Sonnet、Qwen3 Coder Next）与多代理编排；结构化提示、Chain‑of‑Thought 与 ReAct 提示策略；OpenQASM 3.0 代码的语法/语义注入与变异；无噪声量子模拟器与总变差距离判定。

**📊 数据集**

使用了MQL Bench中五个5‑qubit算法（Deutsch‑Jozsa、Grover、Bernstein‑Vazirani、GHZ、W state）作为基准程序，随后通过QBugGen生成多类别单一错误实例。

**📈 对比分析**

通过Pass@k（主要是Pass@1）与效率指标（Token消耗、壁钟时间、API成本）进行对比，实验显示：单次重试可将Pass@1从<25%提升至>80%；结构化提示在资源受限下优于CoT和ReAct；在绝大多数错误类别下，开源Qwen3 Coder Next在成本和速度上比Claude Sonnet 4.6优越，唯一在语义偏差错误上则表现略逊。

**⚠️ 局限性**

局限性包括：仅注入单一错误且缺乏多错误组合；仅评估5‑qubit电路，未覆盖更大规模或不同量子框架；对语义复杂错误的修复仍存在缺陷；实验环境受网络/上下文窗口限制导致部分失败被计为错误。

---

## 385. CULTURESCORE: Evaluating Cultural Faithfulness in Video Generation Models

**arXiv ID:** 2606.07311 | [PDF](https://arxiv.org/pdf/2606.07311v1)

**作者:** Anku Rani `[一作]` (Massachusetts Institute of Technology), Paul Pu Liang `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一套针对视频生成模型的文化忠实度评价框架（CultureScore），将文化忠实度拆解为身份、行为、情境三维度，并构建了包含 2,943 条跨 10 个国家、5 个社会文化领域的提示集，生成 6,180 条视频进行评估。

**💡 创新点**

创新点在于：① 将文化忠实度细分为可解释的三维度；② 通过生成 QA 对并使用 VLM 进行细粒度自动评估；③ 展示了传统视觉质量指标与文化忠实度的反向相关性，强调了文化评估的重要性；④ 引入了“扩展提示”与“地理约束移除”两种提示变体来探究模型对隐式文化知识的掌握。

**🔧 技术方法**

技术包括：多模态 VLM（Qwen3-VL 235B）用于自动回答与视频内容对应的文化问题；Prompt 生成与扩展（基于 Gemini Flash）构造三种提示变体；视频生成使用三大最先进模型（Veo 3.1 Fast、LTX-2、Wan 2.2）；人类评测与交叉验证。

**📊 数据集**

数据集为基于 CulturalFrames 的 981 个文化验证提示，扩展后生成 2,943 条提示、9,289 条 QA 对；视频样本包含 6,180 条视频；人类评测采样自 9 个国家的 45 名本土评审。

**📈 对比分析**

比较方法：对三大模型在身份、行为、情境维度分别计算准确率，并与 VideoScore 进行对照；使用 Spearman ρ 衡量 CultureScore 与人类偏好的一致性；结果显示：Wan 2.2 在 CultureScore 上最高（56.8%），但 VideoScore 最低；LTX-2 视频质量最高，CultureScore 低；人类偏好与 CultureScore 方向一致，VideoScore 方向相反。

**⚠️ 局限性**

局限性包括：① Veo 3.1 Fast 评测样本不足（仅 294 条）导致与其他两模型比较不完全对等；② 人类评测缺失伊朗样本；③ VLM 评判可能携带文化偏见，无法完全替代人工评估；④ 仅通过视觉特征评估身份，忽略身份的更深层社会文化维度。

---

## 386. Acoustic Cue Alignment in Audio Language Models for Speech Emotion Recognition

**arXiv ID:** 2606.07309 | [PDF](https://arxiv.org/pdf/2606.07309v1)

**作者:** Iosif Tsangko `[一作]` (TUM University Hospital), Björn W. Schuller `[通讯]` (TUM University Hospital)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过在指令跟随式音频语言模型的提示中加入基于eGeMAPS的六类可解释声学概念标记，探究这些概念标记在语音情感识别任务中的实际使用情况。

**💡 创新点**

创新点在于提出仅通过文本概念标记的“token‑only”干预方法，系统评估模型在音频保持不变时对标记信息的敏感性与鲁棒性，验证模型是否真正利用了音频相关的符号线索。

**🔧 技术方法**

技术主要包括：eGeMAPS特征提取与量化、构造六类概念标记、将标记作为文本提示输入到多种开源指令跟随式音频语言模型（Qwen2‑Audio、Qwen2.5‑Omni、Audio Flamingo 3）中，以及对标记进行顺序打乱、冲突化和逐步破坏的干预实验。

**📊 数据集**

使用的语料库为FAU‑Aibo（儿童德语语音情感）和IEMOCAP（英语双人对话情感），并在IEMOCAP上进一步评估完整标记破坏曲线。

**📈 对比分析**

比较方法是对比未加入标记（UAR⁻）与加入对齐标记（UAR⁺）的性能，并通过shuffle、conflict与逐步破坏等token‑only干预评估对UAR的影响；实验显示对齐标记提升UAR约0.02–0.04，破坏标记导致UAR随破坏概率单调下降，且误分类偏向中性。

**⚠️ 局限性**

局限性包括：仅使用eGeMAPS六类概念标记，未探索更细粒度或跨模态标记；实验仅在少数两套数据集上验证；模型对标记的依赖程度仍受音频编码与文本特征的混合影响，未完全解耦；且干预方法未考虑对模型内部激活层的更细粒度分析。

---

## 387. Trio: Learning Time-Series Forecasting with Temporal-Spatial-Sample Attention and Structural Causal Priors

**arXiv ID:** 2606.07291 | [PDF](https://arxiv.org/pdf/2606.07291v1)

**作者:** Tao Chen `[一作]` (SUPCON Technology Co Ltd), Wenyue Ding `[通讯]` (SUPCON Technology Co Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Trio 模型，将多变量时间序列预测拆分为时间、空间和样本三维注意力，并通过历史输入‑输出对的显式结构化来辅助预测；同时构建 TS‑SCM 合成任务生成器，以动态滞后、噪声、反馈等机制模拟真实系统；

**💡 创新点**

1）将历史视为结构化样本而非平面上下文；2）三维注意力交替编码实现高效的时间、空间与样本级关联学习；3）专门设计的 TS‑SCM 生成器提供可传递的结构化先验；

**🔧 技术方法**

Temporal‑Spatial‑Sample 交替注意力网络、Patch 级嵌入、预编码器、基于查询的样本注意力、以及自定义的 TS‑SCM 合成生成器；

**📊 数据集**

合成延迟依赖数据、工业数据以及公开基准 ETTm1、ETTm2、Electricity、Weather；

**📈 对比分析**

与多种基线（TimeMixer++、PatchTST、DLinear 等）在不同预测时长上比较，Trio 在 3/4 公开基准上平均 MSE 最优；在合成延迟任务中加入样本注意力显著提升 MSE（如从 0.706 降到 0.298）；在零射预测实验中，TS‑SCM 生成器的任务对模型迁移最有帮助；

**⚠️ 局限性**

仍无法实现完全的 PFN‑风格全局零射时间序列预测；TS‑SCM 合成任务虽提升迁移，但在某些指标（如 ETTh2 MSE）仍落后；模型在极长上下文下效果不稳定，需进一步优化效率和鲁棒性。

---

## 388. TOPSIS-RAD: Ranking According to Desires

**arXiv ID:** 2606.07253 | [PDF](https://arxiv.org/pdf/2606.07253v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 389. CaliPPer: quantifying, predicting and improving AI model performance for binding prediction

**arXiv ID:** 2606.07258 | [PDF](https://arxiv.org/pdf/2606.07258v1)

**作者:** Jian-Qing Zheng `[一作]` (University of Oxford), Tao Dong `[通讯]` (University of Oxford)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 CaliPPer 框架，在免疫受体结合预测中通过多链 Sample-to-Domain Distance（S2DD）实现性能估计、模型可推广性曲线绘制以及样本级 Bayesian 再校准，解决传统标签无监督方法无法处理分布漂移和绑定规则变化的问题。

**💡 创新点**

创新点在于：① 将多链距离聚合为单一 S2DD 并自学习链权重；② 在基于密度比的标签无监督估计上加上距离锚定的性能衰减曲线；③ 通过距离相关的 PPV/NPV 曲线实现可变排名的 Bayesian 再校准，从而兼顾性能预测与候选筛选。

**🔧 技术方法**

技术包括：多链 S2DD（支持 Levenshtein、BLOSUM、ESM-2、RMSD 等基准距离和 Morgan 指纹）；密度比估计（PAPE、MCBPE）与距离校正；距离条件的 PPV/NPV 采样与 Bayesian 更新；跨模型、跨数据集的性能衰减曲线拟合与评估。

**📊 数据集**

数据集涵盖十种模型（CNN、LSTM、GCN、Transformer、状态空间、SVM、随机森林、MLP）在两大任务上：TCR‑epitope（783 免疫原，1.6 万样本）和 BCR‑抗原（SARS‑CoV‑2 + 流感 80 变体，8 万样本）；以及五项公开研究（deepAntigen、PanPep、XBCR-net、BigMHC、AntibioticsAI）进行回溯验证。

**📈 对比分析**

通过与 PAPE、MCBPE 以及传统 Platt/Isotonic 对比，CaliPPer 在模型级预测 MAE 为 0.008–0.070，AUROC 提升可达 +0.20，AP 提升 0.025–0.16；在回溯研究中显著提升 TDR（如 deepAntigen 0/5→3/5），验证了跨领域的性能提升。

**⚠️ 局限性**

局限性：① 需要至少 30 条样本（含 8 条少数类）来拟合距离-性能曲线；② 校准集需覆盖查询的距离范围，超出范围时预测不可靠；③ 当 PPV/NPV 随距离几乎不变时，改进有限，退化为标准 Platt 归一化；④ 对极端稀疏或高度不平衡的校准数据效果下降。

---

## 390. DuMate-DeepResearch: An Auditable Multi-Agent System with Recursive Search and Rubric-Grounded Reasoning

**arXiv ID:** 2606.07299 | [PDF](https://arxiv.org/pdf/2606.07299v1)

**作者:** Lingyong Yan `[一作]` (Baidu AI Cloud), Dawei Yin `[通讯]` (Baidu AI Cloud)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了DuMate-DeepResearch多代理深度研究框架，解决了长程规划、任务拆分、幻觉与可审计性等核心挑战。

**💡 创新点**

创新点在于将Agent Core与工具生态解耦、采用图结构动态规划、递归两层执行以及实时 rubric 导引的推理与终止。

**🔧 技术方法**

使用了Qianfan Agent Foundry、图搜索规划、递归两层执行器、rubric‑based test‑time 优化，并集成百度搜索等工具。

**📊 数据集**

使用了公开的 DeepResearch Bench 与 DeepResearch Bench II 两大深度研究基准。

**📈 对比分析**

与多种基线对比，DuMate-DeepResearch 在两大基准上均取得最高总分（Bench: 58.03%，BenchII: 61.95%），并在信息回忆与分析维度领先。

**⚠️ 局限性**

局限在于仍需手动调参、对百度搜索的依赖、对极端噪声检索的鲁棒性不足，以及在多模态与实时更新等场景的适配性待提升。

---

## 391. A Held-Out Transition-Pair Falsifier for Long-Horizon Non-Abelian State Tracking

**arXiv ID:** 2606.07254 | [PDF](https://arxiv.org/pdf/2606.07254v1)

**作者:** Jeonghoon Lee `[一作]` `[通讯]` (Attractor Dynamics), Jeonghoon Lee (Attractor Dynamics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于保留转换对的假设检验协议（held‑out transition‑pair falsifier），并在该协议下训练了一个投影递归状态模型，能够在从短序列到百万长度的极端延展下实现完全正确的最终状态预测。

**💡 创新点**

创新点包括：
1) 通过在训练集中屏蔽特定有序生成符号对，并在评估时强制出现该对，从而剔除局部转换记忆路径；
2) 引入硬投影与软投影温度参数的机制诊断（同态误差、状态一致性漂移、交换子间隙），为评估模型是否真正实现群同态提供量化工具；
3) 在该检验框架下首次展示了非阿贝尔有限群（S3×S3）在百万长度下的完全精确追踪。

**🔧 技术方法**

使用技术包括：
- 投影递归状态模型（连续值隐藏状态、非交换组合、投影算子π）；
- 软投影温度控制；
- 本地动作监督与呈现监督；
- 四维诊断指标（最终准确率、同态误差、状态漂移、交换子间隙）；
- 基线模型（Bag、GRU、结构化SSM）与原型投影读出。

**📊 数据集**

数据集为合成的S3×S3群状态追踪序列，训练序列长度为8，评估长度从4k到1,048,576；另外进行了同因子和异因子持出转换对的鲁棒性验证以及预备的S5非可解群测试。

**📈 对比分析**

对比方法：在相同的持出转换对拆分、相同种子和测试样本数下，使用原始读出和原型投影读出的基线模型；硬投影模型在所有评估长度上均实现100%最终准确率（平均1.0，95%下限0.9854），而基线模型保持在1/36的随机水平；软投影或无投影的同一模型在相同条件下准确率急剧下降。

**⚠️ 局限性**

局限性：
- 仅在可解有限群S3×S3上验证，未推广至自然语言或真实工作流；
- 需要硬投影（温度→0）才能保持精确性；
- 基线覆盖有限，未与所有非阿贝尔状态追踪架构（如Holonomic Network、PD‑SSM、M^2RNN）进行同等拆分下的直接比较；
- 具体连续载体实现未公开，仅提供接口和诊断；
- 结果是针对特定协议的，不能视为模型族的普遍优劣。

---

## 392. Constrained Dominant Sets for Multimodal Document Question Answering

**arXiv ID:** 2606.07252 | [PDF](https://arxiv.org/pdf/2606.07252v1)

**作者:** Ambuj Mehrish `[一作]` (Ca' Foscari University of Venice), Sebatiano Vascon `[通讯]` (Ca' Foscari University of Venice)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在长文档多模态问答中提出基于查询约束的受限优势集（CDS）检索方法，实现高效且无需训练的证据选择。

**💡 创新点**

创新点在于将查询作为硬约束融入图聚类，自动平衡相关性与多样性，无需手工调参，且采用复制者动态求解全局最优。

**🔧 技术方法**

核心技术包括构建文本与视觉节点的多模态图、构造查询-文档共用图、使用复制者动态求解受限优势集，以及与大型VLM阅读器联合生成答案。

**📊 数据集**

实验使用VisDoMBench和MMLongBench-Doc两大多模态长文档问答基准。

**📈 对比分析**

与现有方法对比，Qwen3‑VL‑32B+CDS在VisDoMBench上平均准确率66.99，超过所有SOTA（如G^2‑Reader 66.21），并在单文档基准上提升4–5个百分点。

**⚠️ 局限性**

局限包括：单文档场景下提升有限；仅在受限的五文档检索协议下评估；对更大候选池或更困难干扰器的表现尚未验证。

---

## 393. Combinatorial Landscape Analysis for Dominating Set and Vertex Coloring

**arXiv ID:** 2606.07361 | [PDF](https://arxiv.org/pdf/2606.07361v1)

**作者:** Johanna Gasse `[一作]` (Hasso Plattner Institute), Maxim Stanko `[通讯]` (Hasso Plattner Institute)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了支配集与顶点着色这两类组合优化问题，在不同图类和两种邻域操作（单点翻转与交换）下，对其搜索景观的局部最优结构进行理论分类。

**💡 创新点**

创新点在于将局部最优景观分为四种类型（单峰、平台单峰、等峰、多峰），并给出完整的理论证明，揭示哪些图类导致多峰局部最优，从而指导启发式搜索器的设计。

**🔧 技术方法**

主要采用组合优化理论、图论、可重构理论以及最小支配集和可行着色的罚函数定义，结合邻域图与正向路径概念进行证明。

**📊 数据集**

本研究没有使用实验数据集，而是对所有相关图类（如二分图、完全二分图、树、Corona树、k-皇冠图、C6k、C12k等）进行理论分析。

**📈 对比分析**

通过与已有的可重构理论与已知图类性质对比，证明了在所讨论图类中景观的具体类型；由于是纯理论研究，没有实测性能指标。

**⚠️ 局限性**

局限性在于仅涵盖了部分经典图类，未考虑更一般或更大规模图的情况；此外，研究集中在局部最优结构而非实际搜索算法的运行时分析。

---

## 394. Spatial-Temporal Decoupled Adapter for Micro-gesture Online Recognition

**arXiv ID:** 2606.07355 | [PDF](https://arxiv.org/pdf/2606.07355v1)

**作者:** Xucheng Shen `[一作]` (Hefei University of Technology), Dan Guo `[通讯]` (Hefei University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种微手势在线识别框架，包含空间-时间解耦适配器和自适应软平衡增广两大核心模块，用于在未裁剪视频中同时进行微手势的时序定位与分类。

**💡 创新点**

创新点：①将空间与时间建模拆分为并行分支，使用轻量深度卷积实现空间-时间解耦；②引入自适应软平衡增广（ASBA），根据类别稀缺度与学习难度动态分配增广强度，自动化解决长尾分布问题。

**🔧 技术方法**

技术：冻结VideoMAE‑v2‑g视觉Transformer，插入空间-时间解耦适配器；使用ActionFormer检测头；采用AdamW、cosine学习率调度、轻量化参数更新；ASBA数据增广策略。

**📊 数据集**

数据集：SMG（Micro‑gesture）数据集，包含40个未裁剪视频，16个微手势类别+1个非微手势类别，采用跨主语训练/测试划分（35人训练/5人测试），仅使用RGB通道。

**📈 对比分析**

对比方法：与前一年挑战获胜方案、AdaTAD、VideoMAE‑S、VideoMAE‑B等基线进行比较。实测结果：在SMG测试集上F1得分0.43808，排名第1，领先第二名0.40882约2.9个百分点；ASBA在验证集上将平均mAP从25.10%提升至29.05%；空间-时间解耦适配器单独使用时提升F1从0.41913到0.42450，结合ASBA进一步提升至0.43808。

**⚠️ 局限性**

限制：①仍难以精准捕捉极短时长的微手势，导致漏检；②仅使用RGB信息，未利用骨骼或多模态特征；③检测头的边界回归精度仍有提升空间；④ASBA虽自动化，但对极少样本类别的增广效果仍有限。

---

## 395. The disruption index does not measure scientific innovation

**arXiv ID:** 2606.07332 | [PDF](https://arxiv.org/pdf/2606.07332v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 396. Four intuitionistic modal connectives

**arXiv ID:** 2606.07348 | [PDF](https://arxiv.org/pdf/2606.07348v1)

**作者:** Philippe Balbiani `[一作]` (Institut de recherche en informatique de Toulouse), Çigdem Gencer `[通讯]` (Institut de recherche en informatique de Toulouse)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出并系统研究了四种直觉主义模态连词（◊、□、⧫、▪），给出了它们的语法、语义及对应关系，构建了相应的直觉主义模态逻辑框架。

**💡 创新点**

创新点在于同时引入了 Přenosil 风格的 ◊ 与 Wijesekera 风格的 ⧫ 及其对偶，并证明了它们在所有直觉主义 Kripke 模型下互不等价，同时给出了最小直觉主义模态逻辑的完整性与可判定性。

**🔧 技术方法**

作者采用直觉主义 Kripke 语义、对应理论、标准翻译到一阶逻辑、可满足性与完备性证明以及单变量守护子句（monadic two‑variable guarded fragment）等技术来实现理论证明。

**📊 数据集**

本研究完全基于理论证明，并未使用任何实验数据集。

**📈 对比分析**

通过与基于 Fischer Servi 与 Wijesekera 的系统比较，展示了在更宽松的模型假设下仍能得到完整性与可判定性，证明了其在理论上的优越性。

**⚠️ 局限性**

局限在于尚未给出最小 IML 的复杂度上界、有限公理化问题以及实际实现滤镜、理想等概念的具体方法。

---

## 397. TabSwift: An Efficient Tabular Foundation Model with Row-Wise Attention

**arXiv ID:** 2606.07345 | [PDF](https://arxiv.org/pdf/2606.07345v1)

**作者:** Si-Yang Liu `[一作]` (Nanjing University), Han-Jia Ye `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了轻量级的 TabSwift 模型，采用行级注意力仅 Transformer 并加入 gated attention 与 learnable register tokens，支持分类与回归的统一预训练，并实现了自适应层级早停以降低推理延迟。

**💡 创新点**

创新点在于将行级注意力与 gated attention 结合，利用少量 register tokens 提升全局上下文捕获能力，同时在每层设立可学习的退出头，实现样本级别的动态推理深度。

**🔧 技术方法**

主要技术包括：行级自注意力 Transformer、元素级 gated attention、learnable register tokens、统一的分类/回归预训练框架，以及基于注册汇总的学习型早停策略。

**📊 数据集**

使用了公开的 Talent benchmark（300 个分类与回归数据集）以及合成任务生成器进行预训练，进一步在真实任务上进行评估。

**📈 对比分析**

与 TabPFN v2、TabICL、TabICL 等先进 Tabular Foundation Models 对比，TabSwift 在保持相近甚至更优的平均排名和 PAMA 的同时，显著降低了推理时间，尤其在启用早停后平均计算成本大幅下降。

**⚠️ 局限性**

局限性包括：在极大特征维度时仍需 PCA 预处理导致额外开销；early‑exit 需要额外训练和调参；模型对异常值和极端分布的鲁棒性尚未充分验证。

---

## 398. VeriDrive: Verifiable Counterfactual Supervision for Cost-Efficient Vision-Language Planning

**arXiv ID:** 2606.07338 | [PDF](https://arxiv.org/pdf/2606.07338v1)

**作者:** Zikai Zhang `[一作]` (Durham University), Toby P. Breckon `[通讯]` (Durham University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 VeriDrive 框架，将驾驶推理拆分为可验证的感知–评估–修正三步，并构建了对应的结构化监督数据集。

**💡 创新点**

创新点在于：① 将中间推理转化为可程序化验证的结构化字段；② 通过规则检查与专家修正实现可审计的因果推理；③ 引入预算感知的生成管线，仅在必要时调用高质量 LLM，显著降低生成成本。

**🔧 技术方法**

采用多模态视觉编码器 EVA‑02‑L、LLaVA 语言模型；使用规则检查器、双重验证器、局部生成器 Qwen3‑VL‑32B‑Instruct 与高质量 GPT‑5.5 校正；实现 Perception‑Evaluation‑Revision 的序列化监督。

**📊 数据集**

在 nuScenes 数据集上构建 VeriDrive 数据集（约 33k 样本），每个样本包含感知证据、未来轨迹、规则评估、修正目标及最终规划轨迹。

**📈 对比分析**

在与 OmniDrive（同 Omni‑Q 设置）的对照实验中，VeriDrive 在 L2、Collision 与 Intersection 指标上均优于基线，并在 token 量、生成时长与 GPT 调用成本上降低 60% 左右；与其它最新 VLM 规划方法对比，表现处于同类顶尖水平。

**⚠️ 局限性**

局限性：仅在开环规划任务中验证；评估依赖预定义规则库，缺乏对复杂多主体交互的细粒度检测；仍需要 LLM 生成部分，导致解释性与可复现性受限；在闭环部署与多场景通用性上需进一步验证。

---

## 399. SV-Detect: AI-generated Text Detection with Steering Vectors

**arXiv ID:** 2606.07313 | [PDF](https://arxiv.org/pdf/2606.07313v1)

**作者:** Mikhail Vishnyakov `[一作]` (Independent Researcher), Tatiana Gaintseva `[通讯]` (Queen Mary University of London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于冻结语言模型隐藏层表示的“驱动向量”（Steering Vector）构造的假文本检测器（SV‑Detect），通过层级投影特征实现文本真假判别。

**💡 创新点**

创新点在于把假文本检测视为表示空间的探测问题，利用隐藏状态的系统性方向差异（层级驱动向量）来提取稳健的检测信号；该方法在跨域、跨模型、跨攻击和编辑变换下保持高鲁棒性，并可解释。

**🔧 技术方法**

技术要点：①从冻结LM提取每层均值池化隐藏状态；②用逻辑回归（或均值差、PCA）学习每层区分真伪的驱动向量；③计算文本在这些方向上的余弦相似度作为特征；④在标准化特征上训练轻量化逻辑回归分类器；⑤利用logit‑lens对驱动向量进行词级解释。

**📊 数据集**

主要数据集：DetectRL（多域、多模型、多攻击）与MIRAGE（生成、润色、重写任务），附加COLING作为补充实验；实验中使用多种LM后端（GPT‑Neo、Qwen、Gemma）验证通用性。

**📈 对比分析**

对比方法：零样本基于得分的检测器（LLR、DetectGPT、Fast‑DetectGPT、DNA‑GPT等）和监督式基准（RoBERTa‑Base/​Large、XLM‑RoBERTa）。在分布内 AUROC 接近 99%‑100%，在跨域/跨模型/跨攻击转移中保持 93%‑99% 的 AUROC，显著优于零样本方法且与监督基准相当或略优。

**⚠️ 局限性**

局限性：①性能受参考LM表示几何影响，需选择合适的冻结模型；②需要完整的LM前向推理，计算成本高于轻量级编码器；③仅在英语数据上验证，跨语言性能未知；④仍需标注真伪样本，且随着生成模型演进需定期更新；⑤对极端编辑或新型攻击的鲁棒性尚待进一步研究。

---

## 400. Geometric-Aware Hypergraph Reasoning for Novel Class Discovery in Point Cloud Segmentation

**arXiv ID:** 2606.07280 | [PDF](https://arxiv.org/pdf/2606.07280v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 401. Improved Lower Bounds for Proportionally Fair Clustering

**arXiv ID:** 2606.07285 | [PDF](https://arxiv.org/pdf/2606.07285v1)

**作者:** Benjamin Cookson `[一作]` (University of Toronto), Yeeseok Oh `[通讯]` (University of Tokyo)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究比例公平聚类中的核心问题，给出了新的下界，并证明在任意度量空间中存在 α‑核心为空的实例。

**💡 创新点**

通过将 Droop 配额下的 k=1 情况与 Hare 配额联系，构造了极大化 α 的实例，从而将已知下界 2 提升到 2.1508。

**🔧 技术方法**

利用偏差图、循环结构、线性规划与混合整数线性规划（MILP）进行搜索，并手工设计偏差结构后用 LP 验证可实现度量空间。

**📊 数据集**

论文中并未使用公开数据集，而是通过符号计算和 GitHub 仓库提供的自定义度量空间实例验证结果。

**📈 对比分析**

与之前的上界 1+√2 以及已知下界 2 进行比较，证明新的下界 2.1508 更高；实验主要通过 MILP 与手工构造实例得到的数值验证。

**⚠️ 局限性**

仍未解决最优核心逼近值的精确界限，且方法对更大 m 的实例计算复杂度高，无法直接扩展到更大规模实例。

---

## 402. Gated Bidirectional Linear Attention for Generative Retrieval

**arXiv ID:** 2606.07317 | [PDF](https://arxiv.org/pdf/2606.07317v1)

**作者:** Artem Matveev `[一作]` (Yandex), Sergei Liamaev `[通讯]` (Yandex)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了GBLA线性时间双向注意力模块，并在生成检索的Transformer编码器中采用混合自注意力与线性注意力的层级结构，以支持长用户历史的推荐。

**💡 创新点**

在传统线性注意力基础上引入键门控、Conv1D短程特征提取和门控RMSNorm三项轻量化改进，首次实现双向线性注意力在生成检索中的可行性，并通过混合堆叠保持高质量。

**🔧 技术方法**

使用核化线性注意力、键门控机制、1D卷积、门控RMSNorm、混合自注意力与线性注意力的层级结构，并在GPU上采用Triton/FlashAttention-v3等加速实现。

**📊 数据集**

在Yandex Music大规模生产数据（约400M样本）以及GRID公开的Amazon Beauty、Sports、Toys 5核心过滤数据集上进行评估。

**📈 对比分析**

通过Recall@{10,100,1000}、NDCG@{5,10}与FlashAttention‑v3、自注意力及Tiger基线对比；GBLA在长序列（≥4096）时速度提升至2.5–8.2倍，且Recall@1000仅比自注意力低约0.8%；在Amazon数据上保持与Tiger相近的精度。

**⚠️ 局限性**

在短历史（≤2048）下线性注意力对速度提升有限，且全线性结构会略微降低精度，需要混合堆叠；对极短历史或非长序列场景的优势不明显；实验仅在H100 GPU上验证，硬件依赖性较强。

---

## 403. CAPE: Contrastive Action-conditioned Parallel Encoding for Embodied Planning

**arXiv ID:** 2606.07304 | [PDF](https://arxiv.org/pdf/2606.07304v1)

**作者:** Cong Chen `[一作]`, Zhengping Che `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了CAPE框架，利用对比学习在一次前向传播中并行预测动作序列导致的完整未来隐状态，避免像素重建和自回归生成；

**💡 创新点**

将视觉动力学预测转化为动作条件的并行对比学习，采用目标收敛对比目标聚焦动作诱发的差异，显著减少学习信号偏向非任务相关视觉内容；

**🔧 技术方法**

采用对比学习（Goal‑Convergent Contrastive Objective）、预训练视觉编码器（如DINOv2）、并行解码器以及MPC进行规划；

**📊 数据集**

在真实机器人抓取数据集DROID以及在RoboCasa模拟环境的零样本迁移任务上进行评估；

**📈 对比分析**

与重建式和潜在预测式基线在未来状态检索、离线动作匹配和闭环规划任务中对比，CAPE在准确率、检索效果和规划效率上均明显优于基线，推断时间更低；

**⚠️ 局限性**

仍受限于预训练编码器对视觉细节的捕捉，难以处理高度动态或复杂背景的场景，对动作序列长度和非抓取任务的通用性尚未充分验证。

---

## 404. Phun-Bench: Evaluating LLMs on Phonological Understanding in Chinese

**arXiv ID:** 2606.07300 | [PDF](https://arxiv.org/pdf/2606.07300v1)

**作者:** Xing Yue `[一作]` (Zhejiang University), Weiming Lu `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出“语音学理解”概念并构建中文基准Phun-Bench，用于系统评估LLM的音位学理解能力。

**💡 创新点**

创新点在于三维度任务设计（同音、韵律、音似），避免记忆化、强调真正的音位学理解，并通过音层推理机制解释LLM表现。

**🔧 技术方法**

采用LLM（Llama3.2、Qwen、GLM4、DeepSeek、GPT‑4o等）思考模式、提示工程、链式思考；同时评测音频模型（如Whisper、SpeechLLM）进行多模态对比。

**📊 数据集**

数据集涵盖同音回忆、同音识别、韵律句子生成、音似比较四类任务，来源于成语库、网络笑话、歌词、常用词表等。

**📈 对比分析**

通过EM、DR、RR等指标与人类基线对比，LLM在同音和音似任务上低于人类，单韵律生成表现较好；思考模式和规模提升显著提升性能。

**⚠️ 局限性**

局限性包括仅评估中文、未做任务专门后训练、数据处理可能存在细微错误、未覆盖其他语言与更复杂音位结构。

---

## 405. SWE-Explore: Benchmarking How Coding Agents Explore Repositories

**arXiv ID:** 2606.07297 | [PDF](https://arxiv.org/pdf/2606.07297v1)

**作者:** Shaoqiu Zhang `[一作]` (Shanghai Jiao Tong University), Xiaodong Gu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 SWE-Explore 基准，用于单独评估仓库探索能力，将问题-仓库对映成排名的代码片段列表；

**💡 创新点**

创新点包括利用成功解题轨迹的交集来生成线级真实标签、固定行数预算的排名评估、以及与下游修复效果的关联验证；

**🔧 技术方法**

技术手段涵盖 LLM 提取读取轨迹、交集与可选读取的 LLM 精炼、覆盖率、nDCG、HitFile/HitRegion 等度量，以及上下文降级实验；

**📊 数据集**

使用的数据来自 SWE‑bench Verified、Pro 与 Multilingual 三大公开数据集，最终得到 848 条跨 10 种语言、203 个开源仓库的实例；

**📈 对比分析**

通过将检索器、搜索代理与长上下文选择器在相同指标（覆盖率、排名、效率等）下进行比较，实验表明 agentic 探索器明显优于传统检索，且指标与下游修复成功率高度相关；

**⚠️ 局限性**

局限性在于仍然缺乏行级召回，核心证据缺失是主要瓶颈，依赖强 LLM 产生轨迹，且评估仅关注固定行数预算和线性探索，未覆盖完整修复流程。

---

## 406. TargetSEC: Plug-and-Play In-the-Wild Speech Emotion Conversion via Arousal-Conditioned Latent Style Diffusion

**arXiv ID:** 2606.07293 | [PDF](https://arxiv.org/pdf/2606.07293v1)

**作者:** Constantin Alexander Auga `[一作]` `[通讯]` (Hasso Plattner Institute University of Potsdam), Constantin Alexander Auga (Hasso Plattner Institute University of Potsdam)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 TargetSEC，一种基于潜在扩散模型的无时长预测语音情感转换框架；

**💡 创新点**

创新点在于使用连续 arousal 嵌入驱动的潜在扩散模型，实现在野生数据上的插件式情感转换，并通过语音特征解耦实现内容、说话人和情感的独立处理；

**🔧 技术方法**

采用 HuBERT、WavLM、预训练风格/情感编码器、HiFi‑GAN 解码器、潜在扩散模型（速度参数化、无分类器引导）以及多任务对抗+重构+情感对齐损失；

**📊 数据集**

使用 MSP‑Podcast v1.10 的 230 小时野生情感语音数据；

**📈 对比分析**

与 HiFiGAN、EmoConv‑Diff、Uncert 等基线对比，TargetSEC 在不使用显式时长预测的情况下，SER 误差最低（MSE≈0.068），自然度（WVMOS≈3.25）与 HiFiGAN 相当且显著优于频谱扩散模型；

**⚠️ 局限性**

局限在于对极端 arousal（1/7）处理不佳，因固定时长映射无法捕捉情绪相关的语速变化，需进一步加入时长预测模块。

---

## 407. Rethinking IoT Intrusion Detection: Augmenting Routing Metrics with Radio Features

**arXiv ID:** 2606.07282 | [PDF](https://arxiv.org/pdf/2606.07282v1)

**作者:** Yichang Sun `[一作]` (Uppsala University), Sourasekhar Banerjee `[通讯]` (Uppsala University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文将RPL控制层特征与节点级的发送（TX）和接收（RX）无线电特征融合，构建基于LSTM的入侵检测系统，并在三种RPL攻击（DIS-Flooding、Local Repair、Worst Parent）及不同网络规模和攻击变体上进行评估。

**💡 创新点**

创新点在于首次系统性地将无线电层的TX/RX特征与传统RPL特征结合，并通过消融实验展示RX特征在检测中的优越判别力，同时证明多层特征融合可显著提升IDS性能。

**🔧 技术方法**

使用LSTM循环神经网络进行二分类，特征提取采用均值与标准差统计，训练采用AdamW优化器，滑动窗口长度为10，步长为3，训练50个epoch。

**📊 数据集**

数据集来自Cooja/Contiki‑NG仿真，生成20个CSV日志，覆盖DIS‑Flooding、Local Repair、Worst Parent三种攻击及其base/on‑off/gradual变体，网络节点数为5、10、15、20，共36个实验配置。

**📈 对比分析**

通过比较五种特征组合（RPL仅、RPL+TX+RX、TX+RX、RPL+TX、RPL+RX）的平均F1得分，发现TX+RX组合在所有攻击类型中均获得最高F1（最高0.999），而RPL仅组合在Worst Parent攻击下最低0.957，显示加入TX/RX可提升约4% F1，显著提升检测性能。

**⚠️ 局限性**

局限性包括仅使用TX/RX特征，未纳入RSSI/LQI等更细粒度无线信号指标；实验仅基于仿真数据，缺乏真实网络验证；未评估模型在不同攻击类型之间的泛化能力。

---

## 408. Self-evolving LLM agents with in-distribution Optimization

**arXiv ID:** 2606.07367 | [PDF](https://arxiv.org/pdf/2606.07367v1)

**作者:** Yudi Zhang `[一作]` (Eindhoven University of Technology), Mykola Pechenizkiy `[通讯]` (Eindhoven University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Q-Evolve 框架，使 LLM 代理通过自我演化从混合离线数据中自动获得过程级奖励，并在同一分布内进行策略优化。

**💡 创新点**

创新点包括：①自我演化闭环，联合训练过程奖励与策略；②在混合专家+自收集数据上使用加权 Implicit Q‑Learning 以获得更稳健的价值函数；③利用优势估计产生密集奖励，避免环境回溯；④采用带有不对称裁剪的行为近端策略优化以抑制负奖励动作。

**🔧 技术方法**

核心技术为 Implicit Q‑Learning（加权版本）、Generalized Advantage Estimation、Behavior Proximal Policy Optimization、回溯奖励标注、混合离线数据构建。

**📊 数据集**

在 AlfWorld、WebShop 和 ScienceWorld 三大文本交互环境上进行评估，同时使用专家演示和自收集轨迹构成的数据集。

**📈 对比分析**

与零拷贝 LLM、SFT、RFT、ETO、DMPO、PPO、Best‑of‑N、QLASS 等基线对比，Q‑Evolve 在所有基准上获得最高平均分，并以仅 13K 步的离线采样大幅提升样本效率。

**⚠️ 局限性**

局限性在于仍依赖离线混合数据和专家演示，分布漂移风险仍存在，且对极度稀疏或高度动态环境的适应性尚未充分验证。

---

## 409. LLM-Guided Evolution for Medical Decision Pipelines

**arXiv ID:** 2606.07342 | [PDF](https://arxiv.org/pdf/2606.07342v1)

**作者:** Ivan Sviridov `[一作]` (Sber AI Lab), Aleksandr Nesterov `[通讯]` (AIRI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

论文研究了一种基于LLM的MAP‑Elites进化方法，用于在推理阶段快速适配临床决策流程，涵盖急诊分流、交互式咨询和肺炎影像分类三种任务；

**💡 创新点**

创新点在于将LLM作为变异器与MAP‑Elites架构结合，实现对可执行决策程序、咨询策略或提示模块的全程进化，提升安全性、交互成本和结构化输出的可解释性；

**🔧 技术方法**

使用技术包括LLM驱动的程序变异、MAP‑Elites质量-多样性搜索、GigaEvo框架、异步评估、结构化批次、谱系融合等；

**📊 数据集**

使用的数据集包括Semigran三分类症状分流数据、MIMIC‑IV‑ED导出的ESI分层数据、MEDIQ/ iCRAFTMD交互式咨询数据、以及MedMNIST PneumoniaMNIST胸片图像；

**📈 对比分析**

比较方法采用手工设计的基线、公开模型（如Llama‑3、Qwen‑3.5、Gemma‑4）以及人类医师或已有系统的指标；结果显示：在Semigran中精度从77.3%提升至87.1%，急诊召回率从0.60提升至0.97；在MIMIC‑ESI中精度从56.7%提升至62.0%，范围精度和安全加权适应度亦提升；在交互式咨询中，演化策略在多模型上提高3–4个百分点精度且降低90%多的交互代价；在PneumoniaMNIST上，Prompt演化使MedGemma在不同分辨率下准确率提升至84.5%（4B）/92.5%（27B）以上；

**⚠️ 局限性**

限制包括Semigran样本过小易过拟合、MIMIC‑ESI为回顾性且缺少低危类别、演化搜索成本高、评价指标含有主观权重、以及缺乏前瞻性临床验证或鲁棒性评估。

---

## 410. Empirical Evaluation of Large Language Models for Migration of Code Fragments to Post-Quantum Cryptography

**arXiv ID:** 2606.07341 | [PDF](https://arxiv.org/pdf/2606.07341v1)

**作者:** Javier Pallarés de Bonrostro `[一作]` (Universidad Carlos III de Madrid), María Isabel González Vasco `[通讯]` (Universidad Carlos III de Madrid)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了使用大型语言模型对预量子密码代码进行迁移到后量子密码的可行性。

**💡 创新点**

创新点在于构建可复现的600+对Python代码对的PQ迁移数据集，并用微调LLM实现高达92.5%功能正确率。

**🔧 技术方法**

技术包括LLM微调、ChatML/Instruction提示、自动化静态-动态验证管道。

**📊 数据集**

使用了800条验证的Python代码对数据集，覆盖六大加密类别及组合案例。

**📈 对比分析**

对比方法：零样本GPT-4.1与微调后的GPT-4.1-mini、GPT-3.5-turbo、CodeLlama-7B；结果显示微调后模型在静态相似度0.907、动态正确率92.5%，优于零样本。

**⚠️ 局限性**

局限在于仅针对Python及cryptography/oqs库，无法处理跨文件依赖和更复杂协议；模型仍需人工校正键长度与类型错误。

---

## 411. Skeletal-Anchored Dual Harmonics for Structured 3D Modeling

**arXiv ID:** 2606.07337 | [PDF](https://arxiv.org/pdf/2606.07337v1)

**作者:** Zhentao Huang `[一作]` (University of Guelph), Minglun Gong `[通讯]` (University of Guelph)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了Skeletal‑Anchored Dual Harmonics（SADH）三维形状表示方法，将表面局部几何与内部骨架结构统一建模；

**💡 创新点**

创新点在于使用双通道球谐展开同时编码径向几何和自适应支撑角度，并通过几何锚点图将表面与骨架协同优化；

**🔧 技术方法**

采用分阶段优化流程，包括点云分配、双通道球谐拟合、骨架正则化与覆盖扩展，并利用球谐函数、KNN传播及锚点图等技术；

**📊 数据集**

主要使用Famous数据集（22个形状模型）以及部分不完整点云进行实验；

**📈 对比分析**

与WNNC、MASH等基线比较，SADH在Chamfer距离、F-score与法线一致性上显著优于MASH，虽略逊于WNNC，但同时提供连贯骨架结构；

**⚠️ 局限性**

局限在于对优化质量与初始化敏感，依赖球谐阶数，未实现类别级学习与直接生成，需进一步提升。

---

## 412. Defending Jailbreak Attacks on Large Language Models via Manifold Trajectory Kinetics

**arXiv ID:** 2606.07335 | [PDF](https://arxiv.org/pdf/2606.07335v1)

**作者:** Hangtao Zhang `[一作]` (Huazhong University of Science and Technology), Leo Yu Zhang `[通讯]` (Griffith University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Manifold Trajectory Kinetics (MTK) 方法，利用大型语言模型内部隐藏状态随层级演化的邻域结构轨迹，检测 jailbreak 攻击，而无需任何 jailbreak 样本；

**💡 创新点**

创新点在于把检测视为对数据流形内部邻域结构的动态变化进行建模，而非单一度量空间的线性可分；通过邻居排名序列捕捉到“bumping”特征，显著提升对适应攻击和伪恶意提示的鲁棒性；

**🔧 技术方法**

使用每层隐藏状态的 k‑NN 邻居排名构成轨迹，并以 Isolation Forest（或 PCA/One‑Class SVM）做异常检测；采用欧氏/余弦/ℓ1 距离计算邻居；在 LLaMA、Vicuna、Mistral 等 LLM 及 LLaVA、Qwen‑VL VLM 上实现；

**📊 数据集**

训练与评估使用 AdvBench（恶意查询）、Databricks Dolly 15k、Alpaca、OR‑Bench（pseudo‑malicious prompts）作为benign 与 malicious 参考集；对 VLM 采用 MM‑Vet、USB multimodal PMPs、MM‑SafetyBench、JailBreakV‑28K、FigImg 等；

**📈 对比分析**

与 7 名 LLM 现有检测器（GradCuff、GradSafe、HSF、HiddenDetect、SaP、SelfDefend、SmoothLLM）及 6 名 VLM 检测器（HiddenDetect、MirrorCheck、ECSO、CIDER、JailGuard、JailDAM）对比；MTK 在 4 个 LLM、10 个攻击、12 个对手中平均 AUROC 0.94，最多 31/40 领先；在适应攻击下仍保持 TPR ≥ 0.85；伪恶意提示 FPR 仅 0.02；VLM 上平均 AUROC 0.94；计算与训练成本极低；

**⚠️ 局限性**

主要局限是依赖 jailbreak 在层级上呈现不同的邻域演化；若攻击者能使所有层邻域保持 benign 近邻，MTK 可能失效；需要一定数量的 benign 与 malicious 参考样本；对极度自适应攻击的理论极限尚待进一步验证。

---

## 413. Varifold Moment Invariants for Sustainable and Explainable Contour Feature Extraction

**arXiv ID:** 2606.07333 | [PDF](https://arxiv.org/pdf/2606.07333v1)

**作者:** G. Longari `[一作]` (Technische Universitaet Wien), A. B. Tumpach `[通讯]` (Lille University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种统一框架Varifold Moments（VMI）来生成二维形状的旋转、平移（可选缩放）不变特征；

**💡 创新点**

通过将曲线的几何、曲率及切线信息结合，构造可解释且判别力强的Varifold Moment Invariants，涵盖并扩展了Hu矩、Zernike矩等经典不变量；

**🔧 技术方法**

使用复数形式的曲线积分定义Varifold Moments，结合统计分布（如随机点距离、三角面积等）的矩来生成特征，并通过随机森林或多层感知器进行分类；

**📊 数据集**

在多种数据集上验证：几何形状（Mendeley、MNIST、MPEG-7、MPEG-400）、叶片（Flavia、Swedish Leaves）和细胞（BBBC010、HeLa Kyoto、MOC）等；

**📈 对比分析**

与ShapeEmbed、XShapeEncoder、Flusser矩、EFD等基线方法对比，VMI在F1‑score上保持相当或更优的性能，同时计算时间显著降低（CPU实现，单个数据集3–4小时以内）；

**⚠️ 局限性**

主要限制是特征设计手工化，仍有待发现更具判别力的VMI；扩展到更高维度仍为未来工作。

---

## 414. Authorized and Verifiable Searchable Encryption Based on Public Key Equality Test for Cloud Storage

**arXiv ID:** 2606.07319 | [PDF](https://arxiv.org/pdf/2606.07319v1)

**作者:** Xiuping Li `[一作]` (Northwestern Polytechnical University), Xiaolin Chang `[通讯]` (Beijing Jiaotong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种基于PKEET的可授权、可验证的云存储搜索加密方案（AVSE），实现了对加密文件的一致性测试、授权与结果可验证；

**💡 创新点**

创新点在于首次结合一次性、不可转移的授权令牌与公共可验证证明，解决了现有PKEET方案缺乏文件级授权和可验证结果的问题；

**🔧 技术方法**

利用双线性映射、公钥加密与等价测试、哈希与一次性令牌生成、批量证明与可验证的等价检验；

**📊 数据集**

实验使用公开的PBC基对实现环境，评估了加密、授权、检验、搜索等操作的耗时与通信量；

**📈 对比分析**

与多种已有PKEET与SE方案对比，AVSE在授权令牌大小、验证成本、搜索时间与存储开销方面表现优越，尤其在令牌可验证性与授权安全性上取得显著提升；

**⚠️ 局限性**

局限性包括仅支持单关键词查询、静态数据集合，并且无法隐藏搜索与访问模式；

---

## 415. Hierarchical Certified Semantic Commitment for Byzantine-Resilient LLM-Agent Collaboration

**arXiv ID:** 2606.07316 | [PDF](https://arxiv.org/pdf/2606.07316v1)

**作者:** Haoran Xu `[一作]` (University of Glasgow), Xianbin Wang `[通讯]` (University of Western Ontario)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Hierarchical Certified Semantic Commitment (H‑CSC)协议，能够在存在拜占庭攻击的LLM代理协作中，基于语义嵌入与投票信息自动决定提交为semantic_commit、verdict_commit或abort三种类型。

**💡 创新点**

创新点在于：①分层决策机制，将语义聚合与投票阈值分离，产生可验证的三种提交类型；②利用几何中位数聚合与角度阈值筛选语义核心，确保语义聚合的可解释性；③在每种提交上统一使用2f+1签名证书，实现安全与可验证并存。

**🔧 技术方法**

技术实现包括：确定性句子编码器（CRSE）生成向量嵌入；几何中位数聚合+量化；角度阈值判定与最大投票群组选择；BFT风格的可靠广播、部分同步网络模型与2f+1签名证书。

**📊 数据集**

数据集与实验：①BCS_v1（Wikipedia-anchor paraphrase + GPT生成的语义毒化样本）用于控制语义聚合可分离性；②MVR-50（Climate‑FEVER 50题）用于真实LLM代理验证，包含静态与冲刺拜占庭攻击；③跨模型评估使用四种LLM（OpenAI small/large、Anthropic、Meta OpenRouter）验证鲁棒性。

**📈 对比分析**

比较方法：与多种基线（majority voting、CWV、几何中位数聚合、B3 证书包装投票、严格语义CSC等）在commit率、valid‑commit覆盖率、错误率、安全率以及semantic_commit占比等指标上进行评估；结果显示H‑CSC在MVR‑50上commit率≈90%，valid‑commit≈0.88，安全率≤0.04，semantic_commit占比≈74%，与B3在覆盖率、安全率上几乎相当，但额外提供可验证的语义聚合；在BCS_v1中，BFT可行区间无错误，超出BFT时均安全abort。

**⚠️ 局限性**

局限性：①仅在单模型单轮内部评估，未考虑多模型混合或跨任务协作；②阈值θα及margin_1需手动校准，对不同任务可能不通用；③未使用真实阈值签名，只使用逻辑签名模拟；④对embedding相同但语义不同的攻击（如B4）只能通过安全abort处理，无法精细区分；⑤未对证据可信度或来源可靠性进行检查，无法防御模式B4的相似语义攻击；⑥基于Climate‑FEVER的标签噪声可能影响gold‑validity评估。

---

## 416. ExMesh: EXplicit Mesh Reconstruction with Topology Adaptation

**arXiv ID:** 2606.07288 | [PDF](https://arxiv.org/pdf/2606.07288v1)

**作者:** Chuanjin Fan `[一作]` (University of Science and Technology of China), Tianzhu Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 ExMesh，一种通过可微优化结合自适应顶点拆分/合并和实时 UV 维护，直接对显式网格进行粗到细的重建框架。

**💡 创新点**

创新点在于将离散拓扑更新与可微优化无缝融合，实现几何与纹理解耦、实时 UV 维护，并通过自适应顶点操作在保持面数紧凑的同时获得高精细细节。

**🔧 技术方法**

使用了可微渲染（nvdiffrast）、自适应顶点拆分/合并策略、实时 UV 维护、EMA/曲率/退化度度量、光度损失、深度/轮廓/拉普拉斯平滑及双顶点偏移正则化等技术。

**📊 数据集**

在 DTU 实际场景数据集和 NeRF-synthetic 合成数据集上进行实验。

**📈 对比分析**

与 NeRF‑based、GS‑based 和 Mesh‑driven 3D 重建基线对比，ExMesh 在几何误差（Chamfer Distance）与面数上表现优于或相当于 SOTA，训练时间更短；在渲染质量（PSNR/SSIM/LIPPS）上优于其他 Mesh‑driven 方法，但略逊于基于 Gaussian 的方法。

**⚠️ 局限性**

局限性包括：仍无法与 Gaussian 基方法在视角相关渲染上竞争；需要周期性 UV 重建和初始粗网格（但对球体也能收敛）；以及拓扑更新过程对参数和实现细节较为敏感。

---

## 417. The Capacity of Information-Theoretic Secure Aggregation in Federated Learning

**arXiv ID:** 2606.07277 | [PDF](https://arxiv.org/pdf/2606.07277v1)

**作者:** Lanxin Yi `[一作]` (Southwest Jiaotong University), Xiaohu Tang `[通讯]` (Southwest Jiaotong University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

在联邦学习中提出并分析了一种通用的两阶段信息理论安全聚合框架，求解了关键率、密钥分发通信率与聚合通信率的容量区域，并给出了最优的显式构造方案；

**💡 创新点**

1）去掉了对可信第三方或预设组间密钥结构的依赖，允许任意用户间的密钥分发；2）首次完整刻画了三种资源的容量边界；3）提供了在任意有限域（q≥N）上可实现的确定性构造；4）证明了仅用对称两两密钥分发即可达到最优；

**🔧 技术方法**

信息理论极限分析、线性编码框架、Vandermonde矩阵构造、Kronecker积、对称两两密钥分发与Diffie‑Hellman等密码学基础；

**📊 数据集**

论文未使用具体机器学习数据集，关注的是理论模型与通信/随机性资源的最优度量；

**📈 对比分析**

与Google原始安全聚合及Zhao‑Sun等随机构造相比，该方案在相同聚合通信率下实现更低的密钥率，同时仅需有限域大小 q≥N、输入长度 L=N−T，显著降低实现复杂度；

**⚠️ 局限性**

仅适用于无用户掉线、静态参与、诚实但好奇的场景；对实际隐私模型、动态系统或更强攻击模型（如恶意用户破坏）未给出扩展；

---

## 418. Where Rectified Flows Leak: Characterising Membership Signals Along the Interpolation Path

**arXiv ID:** 2606.07271 | [PDF](https://arxiv.org/pdf/2606.07271v1)

**作者:** Thomas Sesmat `[一作]` (Institut Polytechnique de Paris), Geoffroy Peeters `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了 Rectified Flow 在训练过程中保留的成员身份信号，并证明其随 λ 变化呈钟形曲线，峰值位置可用数据协方差解析预测。

**💡 创新点**

创新点在于首次给出成员身份信号在插值路径上的理论解析表达式、确定峰值位置的闭式公式，并将此结构用于构造高效的 Membership Inference Attack。

**🔧 技术方法**

主要技术包括 Rectified Flow 训练框架、线性/非线性预测分解、高斯假设下的协方差分析、MSE 归一化和 MLP 分类器进行攻击。

**📊 数据集**

实验使用 MAESTRO v3、FMA Large、MTG‑Jamendo 等音乐数据集以及 CelebA 图像数据集，配合 Music2Latent、Stable Audio VAE 与 Stable Diffusion VAE 等潜在编码器。

**📈 对比分析**

在 MAESTRO 上，使用 λ‑分辨率的重建误差实现 0.91 的 AUC，显著优于 Naïve（0.67）、SecMI（0.72）和 PIA（0.83）等对比方法。

**⚠️ 局限性**

局限性包括需近似高斯等方差正交、X0 与 X1 独立的假设、仅适用于白盒攻击、未覆盖有条件生成及大规模模型的泛化验证。

---

## 419. Two-Phase Simulated Annealing for Equitable Team Formation: Eliminating Complaints in Large Engineering Cohorts

**arXiv ID:** 2606.07270 | [PDF](https://arxiv.org/pdf/2606.07270v1)

**作者:** Yiwei Sun `[一作]` (Queen Mary University of London), Dimitrios G Papageorgiou `[通讯]` (Queen Mary University of London)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了两阶段模拟退火团队组队算法：先根据学生互相偏好形成三人小组，再通过模拟退火将三人小组配对成六人团队，满足GPA均衡、性别平衡和规模约束。

**💡 创新点**

关键创新在于将偏好满足与公平优化拆分为两阶段，避免单一目标折中，同时保持阶段一的小组不被打乱，最终实现零正式投诉。

**🔧 技术方法**

使用图论（互惠关系图、三元组检测）与贪心优先分配技术构建初始三人组；随后采用模拟退火元启发式搜索对团队进行全局优化；实现语言为Python。

**📊 数据集**

采用本系“物理聚合物性质”课程的真实数据（238名学生的基本信息、GPA、至多两位同伴偏好）以及过去6年82个分组实例的历史数据。

**📈 对比分析**

与随机、自由选择、CATME、Team‑Anneal等传统方法对比，投诉率从30%降至0%，GPA方差从9.74降至0.005，性别孤立团队完全消失，整体满意度达94.3%；算法在标准硬件上仅耗时3–5秒。

**⚠️ 局限性**

局限性包括：对学生偏好提交率>60%的依赖；当人数>300时三元组枚举可能耗时；文化背景可能影响接受度；以及仅优化初始组组建，无法直接解决后期团队合作中的自由搭乘问题。

---

## 420. Breaking the Ice: Analyzing Cold Start Latency in vLLM

**arXiv ID:** 2606.07362 | [PDF](https://arxiv.org/pdf/2606.07362v1)

**作者:** Huzaifa Shaaban Kabakibo `[一作]` (Paderborn University), Lin Wang `[通讯]` (IBM Research Europe)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对vLLM的冷启动过程进行系统拆解与性能表征，识别出六个基础步骤并量化其对CPU、GPU、IO等资源的依赖，随后构建了一个轻量化的分步回归预测模型；

**💡 创新点**

首次完整拆分vLLM启动流程，发现各步骤呈现可解释的线性关系，并基于此提出可解释的白盒预测器，实现对冷启动时延的精准估算；

**🔧 技术方法**

使用Python+PyTorch搭建实验平台，对vLLM启动日志进行采样、CPU/GPU profiling，利用线性回归对每个步骤建模，并将所有步骤叠加得到总时延；

**📊 数据集**

实验基于22款LLM模型（如Llama、Falcon、Qwen、Yi等），在H100、L40S GPU上并配备不同CPU（EPYC、Xeon）和SSD进行启动时延收集；

**📈 对比分析**

通过与不同GPU、CPU、存储与加载方式（Safetensors、Run:ai Streamer、CoreWeave Tensorizer）对比，测得各步骤耗时；预测器在验证集上的MSE≈2.4s，最大误差≈2.1s，且在新发布的vLLM版本上仍保持相似精度；

**⚠️ 局限性**

仅针对非MoE模型的线性预测，未覆盖MoE或其他非线性模型；只考虑引擎内部启动，不含分布式容器化、网络等外部因素；在不同硬件/系统上需重新训练或微调模型。

---

## 421. Dash2Sim: Closed-Loop Driving Simulation from in-the-wild Dashcam Videos

**arXiv ID:** 2606.07366 | [PDF](https://arxiv.org/pdf/2606.07366v1)

**作者:** Anurag Ghosh `[一作]` (Carnegie Mellon University), Srinivasa Narasimhan `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将单目车载摄像头在野外拍摄的视频恢复为可用于闭环仿真的度量化、地理参照的4D驾驶日志，并构建了面向工作区的4D日志基准；

**💡 创新点**

通过将街景图像作为全局尺度锚点，实现了对单目视频的精准几何恢复；利用无标注的“非反应式日志重放”作为大规模质量验证信号；以及将恢复的密集深度用于提升新视角合成和闭环规划的表现；

**🔧 技术方法**

采用基于街景的几何锚点（SfM+GPS/视觉定位）、物体检测与跟踪、3D物体上采样、闭环仿真验证（nuPlan+OpenStreetMap）、以及新视角合成框架OmniRe；

**📊 数据集**

ROADWork（工作区车载摄像头视频）作为主要数据源，生成的WorkZone4D基准包含17个城市、4335个场景、数百个动态与静态对象轨迹；

**📈 对比分析**

在nuPlan仿真平台上对比了PlanTF、Diffusion Planner、学习型与混合型Pluto以及规则型PDM-Closed四种规划器；结果表明规则型和混合型规划器在工作区场景下的闭环分数比纯学习型高20–40分，但整体仍显不足，凸显长尾场景的挑战；

**⚠️ 局限性**

受限于单向前摄像头覆盖范围有限、不同处理阶段的误差累积、地图与恢复日志的不匹配、以及仿真器和评分机制的差异，导致验证子集受限且对实际驾驶行为的泛化能力尚未充分验证。

---

## 422. DirectAudioEdit: Inversion-Free Text-Guided Audio Editing via Diffusion Prediction Contrast

**arXiv ID:** 2606.07356 | [PDF](https://arxiv.org/pdf/2606.07356v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 423. A robust PPG foundation model using multimodal physiological supervision

**arXiv ID:** 2606.07365 | [PDF](https://arxiv.org/pdf/2606.07365v1)

**作者:** Eloy Geenjaar `[一作]` (Georgia Institute Of Technology), Daniel P. Darcy `[通讯]` (Dolby Laboratories)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

研发了一种基于多模态生理监督的PPG基础模型，利用ICU共记录的ECG与呼吸信号在预训练阶段指导对比学习，从而在推理时仅需PPG即可获得鲁棒表征。

**💡 创新点**

创新点在于将ECG和RESP提取的生理指标（HR、RMSSD、RR、RA、RVT）作为对比目标，避免对噪声PPG的依赖；同时实现只在预训练需要多模态、推理仅用PPG的无缝迁移方案。

**🔧 技术方法**

技术主要包括：1D ResNet‑26 编码器、数据增强、RNC（rank‑n‑contrast）对比损失、基于生理指标的度量空间、线性探测评估以及多模态预训练框架。

**📊 数据集**

预训练使用 MIMIC‑III Waveform Database（≈4,998 名 ICU 病人，约 20 M 个 10 s PPG 段）；下游评估则覆盖 15 个公开可穿戴/实验室/临床 PPG 数据集（WESAD、DaLiA、WildPPG、PPG‑BP、VitalVideos 等）。

**📈 对比分析**

与 PaPaGei、PulsePPG、SimCLR、BYOL、Chronos、MOMENT 等基线在跨主体与主体内线性探测上进行对比；结果显示模型在 14/15 个下游任务（跨主体）和所有 9 个任务（主体内）均优于基线，尤其在现场噪声环境下表现最为突出。

**⚠️ 局限性**

局限性包括：预训练数据主要来自 ICU，可能对日常 ambulatory 场景的泛化有限；采用 10 s 窗口限制了可提取的 HRV 信息；在血压回归任务上提升有限；模型在肤色、年龄、性别等方面仍存在一定偏差。

---

## 424. SleepExplain: Explainable Non-Rapid Eye Movement and Rapid Eye Movement Sleep Stage Classification from EEG Signal

**arXiv ID:** 2606.07351 | [PDF](https://arxiv.org/pdf/2606.07351v1)

**作者:** Rafsan Jany `[一作]` (Islamic University of Technology), Md Azam Hossain `[通讯]` (Islamic University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用三通道EEG信号自动分类NREM与REM睡眠阶段，并通过SHAP提供模型可解释性。

**💡 创新点**

创新点在于：①将XGBoost与SHAP结合，首次对NREM/REM分类模型进行可解释性分析；②采用SMOTE平衡训练集并使用FFT提取多频特征；③只用三通道EEG实现高精度二分类。

**🔧 技术方法**

技术方法包括FFT特征提取、SMOTE数据平衡、随机森林、Gradient Boosting与XGBoost三种集成分类器，以及SHAP解释工具。

**📊 数据集**

数据集来自荷兰Haaglanden Medisch Centrum（HMC），共154份睡眠记录，选取F4、C4、O2三通道EEG，预处理后得到89096行样本，分为NREM与REM两类。

**📈 对比分析**

通过准确率、精确率、召回率、特异性、F1等指标比较三模型，XGBoost表现最佳，准确率达94.30%，其次是Gradient Boosting 94.25%，随机森林92.54%。

**⚠️ 局限性**

局限性包括：①测试集仍存在类别不平衡导致误差偏大；②仅处理二分类，未扩展到完整五阶段；③特征提取仅基于FFT，可能忽略时序信息；④模型未在实时或临床环境中验证。

---

## 425. Letting Homogeneity Entropy Select S-Pairs in Buchberger's Algorithm

**arXiv ID:** 2606.07321 | [PDF](https://arxiv.org/pdf/2606.07321v1)

**作者:** Uzma Shafiq `[一作]` (Coventry University), Nayyar Zaidi `[通讯]` (Deakin University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了 Buchberger 算法中的 S‑pair 选择问题，提出了一种基于 Shannon 熵的 Homogeneity Entropy 选择策略，并与传统的 Sugar、Normal、Degree 等策略在随机多项式系统与 PHCpack 真实案例上进行了系统实验比较。

**💡 创新点**

创新点在于首次将信息论中的熵概念引入符号计算的 S‑pair 选择；通过分析 S‑polynomial 的次数分布熵来衡量多项式“均匀度”，为选择提供新的全局信息。

**🔧 技术方法**

技术实现上使用 SymPy 1.10.1 编写自定义 Buchberger 算法，采用 Gebauer–Möller 约简；在每次选择前计算 S‑polynomial 的熵，随后根据熵值进行排序并使用 First 作为 tie‑break。

**📊 数据集**

数据集包括：1）随机生成的多项式系统（每个数据集 1000 个实例，变量数、最大次数、系数稠密度等多种配置）；2）PHCpack 真实案例（94 个系统）；3）匹配 PHCpack 结构的合成数据集（100 个系统）。

**📈 对比分析**

在同一硬件（Intel Core i7‑12700H）和实现下，对每种策略记录总运行时间、平均/中位数、超时次数和每个实例的严格胜利数。结果显示：在随机系统上 Entropy 策略比 Sugar、Normal、Degree 快约 10‑12 倍；在 PHCpack 真实案例中，传统 Sugar 仍表现最佳，Entropy 的优势在某些难度区间显现。

**⚠️ 局限性**

局限性包括：Entropy 需要先生成 S‑polynomial，导致在需要极快决策的场景下额外开销；当 S‑polynomial 的次数分布相近（如 PHCpack 系统）时，熵无法有效区分，表现不佳；单一熵信号难以覆盖所有数据分布，需进一步结合其他特征或学习方法。

---

## 426. Bootstrap Theory of Representational Emergence: Explanatory Insufficiency as a Driver of Representation Learning and World Models

**arXiv ID:** 2606.07303 | [PDF](https://arxiv.org/pdf/2606.07303v1)

**作者:** Jacques Raynal `[一作]` (University of Montpellier), Jacques Margerit `[通讯]` (University of Montpellier)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出Bootstrap Theory of Representational Emergence（TBER），描述当现有表示在解释上变得不足时，新的表示如何被激发产生。

**💡 创新点**

将解释不足视为表示迁移的正向信号，构建递归自引导的bootstrap动态，并给出包含“稳定观测、异常检测、解释不足认知、表示产生、临时稳定化”五个阶段的模型；将这一理论与机器学习、基础模型、世界模型、生物适应性和科学发现等多领域连接。

**🔧 技术方法**

主要采用理论分析、哲学与认知科学视角，以及对现有文献的系统综述来构建框架；未提出新的算法或技术实现。

**📊 数据集**

论文没有使用具体的数据集；作者仅以适应性运动系统的研究经验为启发性案例，但并未在本文中进行实验或数据分析。

**📈 对比分析**

由于本文为概念性理论，未进行任何实验或性能对比；因此不存在方法比较或性能评估。

**⚠️ 局限性**

局限性包括：缺乏形式化与可度量的定义；解释不足的识别与检测缺乏操作化指标；实证验证主要基于单一领域（运动系统）的案例，普适性待进一步检验；理论无法直接指导算法实现或系统设计。

---

## 427. Closed-Form Spectral Regularization for Multi-Task Model Merging

**arXiv ID:** 2606.07289 | [PDF](https://arxiv.org/pdf/2606.07289v1)

**作者:** Yongxian Wei `[一作]` (Tsinghua University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于谱正则化的闭式模型融合方法，替代传统的迭代优化，显著提升融合效率与准确性。

**💡 创新点**

核心创新在于将无数据模型融合视为噪声线性逆问题，发现迭代优化实质上是隐式谱滤波；基于此设计了两种闭式谱滤波器（Soft‑Exponential + Hard‑Rank Truncation）和自适应层级截断版本 AdaMerging，消除超参数并保持高性能。

**🔧 技术方法**

使用对称特征分解（Eigen‑decomposition）对每层任务向量矩阵进行谱分解，然后按自定义滤波器 h_k 计算伪逆；算法不依赖梯度、训练数据或优化器状态，只需一次矩阵分解。

**📊 数据集**

在多种数据集与任务上评估：Vision（CLIP‑ViT TA8 / TALL20）、NLP（Flan‑T5 GLUE、Llama‑3.2‑3B MergeBench）、多模态（InternVL2.5‑1B、Qwen2‑VL‑7B）以及专门的 MLLM 合并基准（VQA、Geometry、Chart、OCR、Grounding 等）。

**📈 对比分析**

与 Weight‑Average、Task Arithmetic、TIES、DARE、TSV‑Merging、Iso‑C、原始 Merging、OptMerge 等方法对比，AdaMerging 与 Spectral‑Merging 在所有基准上均能匹配或超过迭代 Merging 的准确率，同时实现 28–72 倍的壁钟时间加速和最高 50% 的 GPU 内存节省；在多模态合并中甚至能超越混合训练（Mixture‑Training）模型。

**⚠️ 局限性**

局限性包括：需要对每层进行特征分解，可能在极大模型层（如 100B 参数级）上成本仍然较高；谱正则化假设线性子空间近似，若任务向量分布不满足可能导致滤波效果下降；自适应截断仍需依据谱统计选择规则，对不同架构的泛化性需进一步验证。

---

## 428. A Model of Integrated Information Processing in Human-AI Interaction

**arXiv ID:** 2606.07283 | [PDF](https://arxiv.org/pdf/2606.07283v1)

**作者:** Tim Schrills. Thomas Franke `[一作]` `[通讯]`, Tim Schrills. Thomas Franke

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

提出 Integrated Information Processing (IIP) 模型，使用共享控制循环的概念来描述人类与 AI 在任务中共同的信息处理与行动调节。

**💡 创新点**

创新点在于：① 将控制理论与人机交互的心理学机制统一成同一语言；② 引入输入充分性、参考一致性、输出可操作性三种整合质量，形成可衡量的设计与评估维度；③ 用嵌套校正循环阐释人机交互中的自我调节过程。

**🔧 技术方法**

采用控制理论（输入–参考–输出循环）和心理学行动调节模型为技术基础，并结合 XAI 与人机交互设计原理进行理论构建。

**📊 数据集**

本文为概念性工作，无使用具体数据集；所有讨论均基于案例和现有文献。

**📈 对比分析**

没有进行实验或数值比较，主要通过与已有框架（如自动化水平模型、团队合作挑战、XAI 设计原理）的对比来说明模型的价值，未给出具体性能指标。

**⚠️ 局限性**

局限性包括：① 仍处于理论层面，缺乏经验验证与可靠指标；② 对多主体、多代理场景的适用性尚未系统化；③ 需要进一步研究如何将整合质量量化并与现有信任、工作负荷等构念关联。

---

## 429. Off-Policy Evaluation with Strategic Agents via Local Disclosure

**arXiv ID:** 2606.07308 | [PDF](https://arxiv.org/pdf/2606.07308v1)

**作者:** Kiet Q. H. Vo `[一作]` (CISPA Helmholtz Center for Information Security), Krikamol Muandet `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在决策者与代理人互动时出现的战略性特征修改对离线策略评估（OPE）的影响，并提出一种在一轮决策、代理人对策略的战略响应下可行的评估框架。

**💡 创新点**

创新点包括：①引入局部信息披露（LID）与基于动作解释（ARex）来获取代理人预战略特征，从而缓解政策相关特征漂移；②假设代理人成本敏感度服从条件对数正态分布，利用最大似然方法估计成本模型；③构造一个“策略鲁棒双重稳健”估计器，既能校正战略特征漂移，又具有双重稳健性，并证明其一致性。

**🔧 技术方法**

使用的技术包括：最大似然估计（MLE）学习成本参数、基于ARex的局部解释设计、双重稳健估计（IPS+DM组合）以及对数正态分布的概率推断；实验中还使用蒙特卡洛近似处理连续特征。

**📊 数据集**

实验数据集：①合成数据（二维离散特征，已知成本参数）；②德国信用卡数据（18维特征，其中8维可被代理人调整），通过CTGAN生成200k样本。

**📈 对比分析**

与传统IPS、DM、未校正的双重稳健等方法对比，实验显示：①在有限重叠或模型失配下，传统方法误差大幅增加；②我们的SDR估计器在样本量增大时误差收敛到零，保持一致性；③在存在部分“非理性”代理人的情况下，误差随样本增大趋于零。

**⚠️ 局限性**

局限性包括：①假设成本敏感度服从对数正态分布，若真实分布差异大可能导致估计偏差；②需要获取代理人的预战略特征，若无法实现LID则无法应用；③对完全理性代理人模型敏感，虽然可以排除异常样本，但若异常比例较大会影响估计；④只考虑一次性决策，未涵盖多轮交互情景。

---

## 430. TEVI: Text-Conditioned Editing of Visual Representations via Sparse Autoencoders for Improved Vision-Language Alignment

**arXiv ID:** 2606.07451 | [PDF](https://arxiv.org/pdf/2606.07451v1)

**作者:** Sweta Mahajan `[一作]` (Max Planck Institute for Informatics), Bernt Schiele `[通讯]` (Max Planck Institute for Informatics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用文本描述对 CLIP 图像嵌入进行编辑，保留 caption 中提到的属性信息并剔除其他多余信息。

**💡 创新点**

提出了基于稀疏自编码器（SAE）与文本条件掩码的可编辑跨模态对齐框架，可解释地拆解并动态选择嵌入维度；与传统直接对齐或全局训练方法不同，显著提升了对齐质量和检索性能。

**🔧 技术方法**

稀疏自编码器（SAE）、多层感知机掩码网络、InfoNCE 对比损失、CLIP/SigLIP 视觉与文本编码器，结合数据增强与掩码学习。

**📊 数据集**

合成 MAD 数据集、CC12M 预训练集、MS COCO、Flickr30k、DOCCI、IIW、RoCOCO 以及 CC3M 验证集。

**📈 对比分析**

与 CLIP、SharedCLIP、AlignCLIP 及 SmartCLIP 等方法在 R@1、R@5 等检索指标以及图像-文本余弦相似度上进行对比。实验显示在长文本检索基准（DOCCI、IIW）上大幅提升，在短文本基准（MS COCO、Flickr30k）上保持竞争力；在 RoCOCO 鲁棒检索指标上亦显著改进。

**⚠️ 局限性**

1) 难以直接扩展到极大规模模型；2) SAE 的概念分解可能不完美，导致编辑效果受限；3) 需要为每张图像和每条文本都进行条件化，推理成本略高。

---

## 431. Information Networks of Stock Prices

**arXiv ID:** 2606.07450 | [PDF](https://arxiv.org/pdf/2606.07450v1)

**作者:** Muhammad Aldy Hassan `[一作]` (Bandung Fe Institute), Hokky Situngkir `[通讯]` (Institut Teknologi Del)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过构建印尼股市价格网络，比较 Pearson 相关、互信息（自适应分箱与 kNN）与两种网络过滤方法（最小生成树 MST 与平面最大过滤图 PMFG），并结合四种社区检测算法，系统评估 24 种配置对行业分类一致性的影响。

**💡 创新点**

创新点在于：① 将互信息与 PMFG 组合，首次揭示线性基线之外的残差信息，发现隐藏的商品与供应链子结构；② 采用 AMI 作为评价指标，并同时使用四种社区算法，避免单一算法偏差；③ 在印尼市场上大规模滚动窗口检验，提供跨时间段的网络演化视角。

**🔧 技术方法**

技术手段包括：Pearson 相关、互信息（自适应分箱估计与 kNN/KSG 估计）、最小生成树（MST）、平面最大过滤图（PMFG）、社区检测（greedy modularity、Louvain、Leiden、Infomap）、调整互信息（AMI）评估、滚动窗口、kNN 迭代自举平滑。

**📊 数据集**

使用 150 只印尼交易所股票（2015-2025 年每日收盘价），通过 300 天窗口、25 天步长共 97 个窗口，形成 2,328 条评估记录。

**📈 对比分析**

通过 AMI 与行业分类对齐度比较不同配置，发现 Pearson‑MST 取得最高 AMI，说明最能映射正式行业；MI‑PMFG AMI 较低，但揭示残差信息与隐藏子结构；PMFG 在结构上更丰富，社区更大；MI‑kNN 敏感度高但噪声大；自适应分箱在稳定性与信息量之间取得平衡。

**⚠️ 局限性**

局限性包括：① 样本窗口有限导致 MI‑kNN 易受噪声与极端值影响；② PMFG 受平面约束，边的增删对网络结构敏感；③ 只在印尼市场验证，结果可推广性待进一步研究；④ 本研究关注结构关联，缺乏因果解释与交易策略验证。

---

## 432. Discovering Multiscale Deep Formulas in Complex Systems via Neural-Guided Lambda Calculus

**arXiv ID:** 2606.07426 | [PDF](https://arxiv.org/pdf/2606.07426v1)

**作者:** Hanqiao Yu `[一作]` (Xi'an Jiaotong University), Cong Zhao `[通讯]` (Xi'an Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 Deflex 方法，通过神经网络与 Lambda 计算符号回归实现多尺度复杂系统公式的自动发现。

**💡 创新点**

将所有公式统一为概率分布的能量模型，并结合可分解 Transformer（Deflexformer）与 Lambda 计算符号回归（Deflexpressor），突破传统 SR 在高阶关系、多尺度统一和搜索效率上的限制。

**🔧 技术方法**

使用能量基模型、分解式 Transformer、Lambda 计算符号回归、预训练+后训练框架、Langevin MCMC 采样等技术。

**📊 数据集**

六类代表性复杂系统数据：稀有气体粒子运动、花粉粒子运动、跨尺度水粒子、二维圆柱尾流、三维湍流（JHTDB）、人群与鸟群移动数据集。

**📈 对比分析**

与 Operon、SciMED、SINDy、AI Feynman、gplearn、PySR、DEAP 等方法对比，Deflex 在公式准确性、收敛速度上通常比大多数方法快 5–7 倍，并能在多尺度上保持较低 EMD，展示出更优的性能。

**⚠️ 局限性**

受限于符号回归的候选生成效率、对预处理数据的依赖、在多尺度互相抑制时可能忽略高频细节、以及对噪声与缺失数据的鲁棒性仍待提升。

---

## 433. DisPOSE: Projected Polystochastic Diffusion for Self-Supervised Multi-View 3D Human Pose Estimation

**arXiv ID:** 2606.07419 | [PDF](https://arxiv.org/pdf/2606.07419v1)

**作者:** Tony Danjun Wang `[一作]` (Technical University of Munich), Nassir Navab `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种通过在多视角人类姿态估计中将个体关联问题建模为扩散生成过程，利用多重随机张量空间进行自监督学习，达到无3D监督下的高精度3D姿态恢复。

**💡 创新点**

创新点在于将多视角个体关联视为在多重随机张量空间上的扩散过程，避免传统监督需求，利用自监督扩散机制逐步消除多视角不确定性，提升关联精度。

**🔧 技术方法**

使用的核心技术包括多视角扩散网络、Polystochastic张量表示、生成式自监督学习策略以及无监督的姿态一致性约束。

**📊 数据集**

实验数据集主要包括多视角3D人体姿态数据集 Human3.6M 和 MPI-INF-3DHP，以验证方法在不同场景下的有效性。

**📈 对比分析**

与传统有监督方法（如 SMPLify、HMR 等）以及基线自监督方法进行比较，评估指标为 MPJPE/PA-MPJPE，实验结果表明该方法在无3D标签条件下实现了与有监督方法相近甚至更优的性能。

**⚠️ 局限性**

局限性包括对多摄像头同步与数量的高度依赖、扩散训练过程计算成本较大，以及在摄像头视角有限或动态复杂场景中的鲁棒性仍需进一步提升。

---

## 434. Rate Loss in Quantum Channels with Classical State and Applications for Quantum Broadcast Channels

**arXiv ID:** 2606.07409 | [PDF](https://arxiv.org/pdf/2606.07409v1)

**作者:** Igor Bernard `[一作]` (EURECOM), Arun Padakandla `[通讯]` (EURECOM)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究具有经典状态信息的经典-量子（CQ）信道中的速率损失（rate loss），首先在一种噪声BB84‑样本信道上证明若接收方不了解状态会导致可达速率严格下降；随后将该结论推广到三用户CQ广播信道，证明使用余子码（coset code）结构化编码能显著提高某些点的速率，而传统的无结构IID编码则无法实现。

**💡 创新点**

创新点在于：①首次给出非可交换（non‑commutative）CQ信道中的速率损失解析表达式；②利用余子码实现对二元干扰的有效解码，从而在三用户CQ广播信道上突破无结构编码的性能极限；③通过凸性与几何极值分析给出完整的容量上界与下界比较。

**🔧 技术方法**

核心技术包括：量子相干信息理论（Holevo信息、量子相位噪声模型）、凸优化与 Jensen 不等式、余子码（nested coset coding）构造、以及量子相干状态的Bloch向量表示与谱分析。

**📊 数据集**

研究完全基于理论模型，无实际数据集；使用的信道模型为噪声BB84‑样本量子信道及其三用户扩展，参数化为状态概率q、噪声δ以及输入成本τ。

**📈 对比分析**

作者通过与一个“减弱无结构IID子类”进行对比来评估性能。对该子类的可达率受单用户容量与干扰联合解码约束限制；而余子码结构能在满足同等成本约束的前提下实现更大的速率三元组（C1,C2,C3）。实验性数值（如τ=0.05, δ1=0, δ2=0.158, δ3=0.159）表明结构化方案可达的角点在无结构子类中不可实现。

**⚠️ 局限性**

局限性包括：①仅在特定的噪声BB84‑样本信道和三用户广播信道模型下给出结果；②对比基于“减弱无结构IID子类”，未覆盖所有可能的无结构编码策略；③缺乏全局外界限，无法证实结构化方案的绝对最优性；④结果对参数范围敏感，需进一步验证在更一般的非可交换信道上的适用性。

---

## 435. Simulation-Driven Imitation Learning for Biosignals-Free Shared-Autonomy Prosthetic Grasping

**arXiv ID:** 2606.07389 | [PDF](https://arxiv.org/pdf/2606.07389v1)

**作者:** Kaijie Shi `[一作]` (Memorial University of Newfoundland), Xianta Jiang `[通讯]` (Memorial University of Newfoundland)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了基于仿真的可扩展数据生成框架，实现了无需EMG、只靠腕部摄像头的共享自治假肢抓取控制；

**💡 创新点**

通过自动化生成多样化的到抓取、抓握、抬升演示，解决了收集真实演示数据的成本与可变性瓶颈，并提供了标准化的仿真基准；

**🔧 技术方法**

采用物理可行抓取合成（BoDex）、手腕轨迹重定向、Isaac Sim仿真、VTM‑VAE/ACT/HannesImitation等深度模仿学习技术；

**📊 数据集**

使用Google Scanned Objects（200个）与Infinigen生成的10个室内场景，累计2000个成功演示，形成大规模演示数据集；

**📈 对比分析**

在仿真中与三种SOTA模仿学习方法对比，最佳方法ACT在房间泛化下成功率达57%，在真实世界中无物理信号训练的模型实现超过90%的抓取成功率，明显优于仅用真实数据训练的基线；

**⚠️ 局限性**

主要局限包括：仍存在仿真到现实的差距、仅针对单一手型、缺少多物体/关节物体交互、失败主要集中在把握力不足与抓取时机误差上。

---

## 436. Covariance Shrinkage via Stochastic Interpolation

**arXiv ID:** 2606.07382 | [PDF](https://arxiv.org/pdf/2606.07382v1)

**作者:** Mathieu Chalvidal `[一作]` (Capital Fund Management), Eric Vanden-Eijnden `[通讯]` (Capital Fund Management)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种通过神经网络实现的随机插值方法对高维协方差矩阵进行收缩估计。

**💡 创新点**

创新点在于将协方差收缩框架重构为参数化插值的经验风险最小化，揭示调度、耦合与早停三种正则化机制，并通过非线性流图解放了旋转不变性限制。

**🔧 技术方法**

采用神经ODE流、最优传输耦合、条件流匹配以及Stein无偏风险估计（SURE）进行训练与早停控制。

**📊 数据集**

在合成高斯样本以及ABIDE脑功能磁共振成像（fMRI）数据（200区域、每个受试者100时间点）上进行实验。

**📈 对比分析**

与传统Ledoit‑Wolf线性收缩和Wasserstein‑2 OT收缩进行比较，基于测试集负对数似然指标显示新方法在高维低样本情形下明显优于基线。

**⚠️ 局限性**

主要局限是模型需要神经网络训练与多步ODE积分，计算成本较高；目前仅验证了协方差矩阵估计，对更高阶统计量及更大规模数据的泛化仍待研究。

---

## 437. Mitosis Detection in the Wild: Multi-Tumor and Context-Aware Generalization in the MIDOG 2025 Challenge

**arXiv ID:** 2606.07368 | [PDF](https://arxiv.org/pdf/2606.07368v1)

**作者:** Marc Aubreville `[一作]` (Flensburg University of Applied Sciences), Christof A. Bertram `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究组织了MIDOG 2025挑战赛，评估在不同肿瘤类型和多种扫描平台下的细胞分裂图像（MF）检测与分型，重点突破传统热点区域限制，加入随机及高假阳性风险的挑战区域；

**💡 创新点**

创新点在于首次在大规模多域、多肿瘤、多扫描仪器条件下，将检测任务从热点扩展到整个切片的随机与挑战区域，构建包含12种肿瘤的测试集，并系统分析不同区域、不同肿瘤对模型性能的影响；

**🔧 技术方法**

采用多种深度学习框架：检测层面多用YOLO系列、RTMDet、DETR、FCOS等一阶段目标检测器；分型层面多用LoRA、Fine‑Tuning、VPT对视觉基础模型（如DINOv3、Uni、Virchow等）的适配；同时探索了模型集成与TTA的效用；

**📊 数据集**

主要使用公开的MIDOG++训练集（包含多种肿瘤和扫描仪），并在检测任务中新增加了两种人类肿瘤（glioblastoma、lung adenocarcinoma）及其对应的WSI；分型任务使用MIDOG++热点MF、AMi‑Br、OMG‑Octo等异常分裂图像；

**📈 对比分析**

通过官方排行榜对比，检测任务最高F₁≈0.740，分型任务最高BA≈0.908；实验表明热点区域性能优于随机与挑战区域，且模型集成平均提升1.5个百分点，TTA提升不显著；

**⚠️ 局限性**

局限性包括：注释过程与之前版本略有差异可能引入标签漂移；随机与挑战区域样本量与多样性不足，导致泛化评估受限；AP等阈值依赖指标受NMS阈值影响；缺乏足够的空间上下文导致部分检测模型表现欠佳。

---

## 438. Odd Cycle Transversal in $P_k$-Free Graphs

**arXiv ID:** 2606.07453 | [PDF](https://arxiv.org/pdf/2606.07453v1)

**作者:** Akramah Faizi `[一作]` (Indiana State University), Arash Rafiey `[通讯]` (Indiana State University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在P_k-无环图上求最小奇环截断集(OCT)的问题，证明在(P_6, C_3)-无环子类可多项式求解，并给出以k为参数的常数比逼近算法；

**💡 创新点**

创新点在于首次利用P_k-无环图的结构将图分解为“环状双部图”，并通过该结构得到关于k的常数逼近因子（奇k为k-2，偶k为k-3），填补了之前仅有极限逼近或无逼近的空白；

**🔧 技术方法**

主要技术包括结构归约（把(P_6, C_3)-无环图转化为5环图）、链双部图理论、最小顶点覆盖与最大匹配的多项式求解，以及贪心/递归的逼近框架；

**📊 数据集**

由于研究为理论算法，没有使用实测数据集；

**📈 对比分析**

相对于已有结果，本文给出的k-2/k-3逼近算法在一般P_k-无环图上首次实现非平凡常数逼近；时间复杂度为多项式（O(n^3.5)等），并在(P_6, C_3)-无环子类实现精确解；

**⚠️ 局限性**

局限性包括：逼近因子仍与k线性增长；尚未确定是否存在更优的逼近常数；在P_6-无环图上仍有3逼近，是否可进一步到2逼近仍未解；另外对更一般P_k-无环图的精确算法仍是NP-难。

---

## 439. The Lipreading Gap: Do VSR Models Perceive Visual Speech Like Human Lipreaders?

**arXiv ID:** 2606.07435 | [PDF](https://arxiv.org/pdf/2606.07435v1)

**作者:** Rishabh Jain `[一作]` (Trinity College Dublin), Naomi Harte `[通讯]` (Trinity College Dublin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过对MaFI词级数据集进行多层级评估，探讨了视觉语音识别（VSR）模型与人类在视觉口语感知上的相似性和差异性。

**💡 创新点**

发现VSR模型的高准确率主要来自语言模式而非视觉信息，模型在词级、子词级误差模式与人类不一致，并且对视觉清晰度的敏感度远低于人类。

**🔧 技术方法**

采用了三种代表性VSR模型（Auto‑AVSR、AV‑HuBERT、VSP‑LLM）、文本n‑gram基线、音素/视觉音素映射以及基于Levenshtein距离的多级指标评估。

**📊 数据集**

使用MaFI词级数据集（2,189个词）以及LRS3测试集进行模型基准对比。

**📈 对比分析**

对比方法包括词、字符、音素和视觉音素层级的WER、CER、PS、VS；文本n‑gram预测与模型结果比较；训练词频与MaFI视觉难度的相关性分析；视觉音素准确率与混淆矩阵对比；以及与人类误差模式的Spearman相关性。性能上，Auto‑AVSR‑Large在MaFI上的WER 0.65优于人类0.83，但其词级人类-模型相关性仅为0.37，且对视觉清晰度的相关性仅为0.29（人类为0.88）。

**⚠️ 局限性**

研究限制在于仅评估英语单词级任务，未涉及连续语音或多语言场景；缺乏公开的人类口语实验数据；并且当前训练目标仍以错误率为主，未能有效鼓励模型捕获视觉发音信息。

---

## 440. OpenGlass: Open-Source Smart Glasses for On-Device Event-Based Gesture Recognition

**arXiv ID:** 2606.07431 | [PDF](https://arxiv.org/pdf/2606.07431v1)

**作者:** Pietro Bonazzi `[一作]` (ETH Zürich), Michele Magno `[通讯]` (ETH Zürich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一款开放源代码的智能眼镜平台OpenGlass，支持快速原型化传感器与嵌入式机器学习。

**💡 创新点**

创新点在于模块化FPC互连支持多种相机，无需PCB重设计，并配备低功耗RISC‑V GAP9 SoC实现本地推理，续航达11.8小时。

**🔧 技术方法**

采用事件摄像机Prophesee GENX320、RGB摄像机HIMAX HM0360、Nordic nRF5340协同、GreenWaves GAP9+NE16推理、nPM1300电源管理及多模数据预处理与时序增强。

**📊 数据集**

采用LynX egocentric事件手势数据集进行评估。

**📈 对比分析**

通过严格留两人交叉验证和6折交叉验证比较不同网络架构，R(2+1)D在留两人验证中取得83.94%准确率（宏F1 0.781），延迟33.9ms，能耗65.6mW；6折平均准确率约74.6%。

**⚠️ 局限性**

限制包括未充分利用RGB摄像机、跨受试者泛化受限于LynX样本量、未对硬件重量/尺寸进行优化、对低功耗事件驱动的进一步提升空间。

---

## 441. High-Frequency Preconditioners for Electromagnetic Integral Equations Based on Helmholtz Regularizations

**arXiv ID:** 2606.07427 | [PDF](https://arxiv.org/pdf/2606.07427v1)

**作者:** S. Ciciriello `[一作]`, F. P. Andriulli `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种针对移位Helmholtz算子的新预处理策略，以在EFIE的不同不良条件（频率降低、细化网格、频率升高）下保持矩阵条件数和迭代次数的稳定，从而实现准线性复杂度的求解。

**💡 创新点**

创新点在于利用复波数单层算子（具有‑1伪微分阶）对移位Helmholtz矩阵进行左右预处理，理论上实现伪微分阶相消（+2+‑1+‑1=0），从而在密集离散化与高频两大崩溃模式下获得无网格依赖的条件数，并通过球面谐波分析证明不会产生伪共振。

**🔧 技术方法**

技术包括：边界元方法（EFIE）离散、Rao‑Wilton‑Glisson 基函数、双层/单层算子、伪逆与GMRES迭代、球面谐波频谱分析以及潜在的快速算法（FMM）加速。

**📊 数据集**

实验数据集：以半径为1的光滑球面为目标，对其进行icosahedron细分得到三角面片网格；右端向量为随机生成并正交于拉普拉斯算子零空间的常数向量。

**📈 对比分析**

对比方法：在两种极端条件下分别评估原始移位Helmholtz矩阵 H_S 与预处理后的 H_S_prec 的条件数 κ 与 GMRES 迭代次数。结果显示，预处理后无论在 λ/h（网格细化）还是 ka（频率变化）下，κ 与迭代次数保持几乎不变，而原始矩阵随 λ/h 或 ka 增大呈指数增长。

**⚠️ 局限性**

局限性：仅在单一球面案例上验证，未在更复杂或非球形散射体上测试；未展示与 FMM 等快速算法结合后的实际计算成本；缺乏大规模问题的全面复杂度分析。

---

## 442. Lost in Migration: Exposing Android Framework Vulnerabilities in Parallel Java-Kotlin Implementations

**arXiv ID:** 2606.07420 | [PDF](https://arxiv.org/pdf/2606.07420v1)

**作者:** Rui Li `[一作]` (Singapore Management University), Debin Gao `[通讯]` (Singapore Management University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了 Android 框架中并行存在的 Java 与 Kotlin 实现，构建自动化工具检测语义差异并评估其安全风险，最终发现并报告多项漏洞；

**💡 创新点**

首次系统性探讨并行实现的现象，并提出了统一执行图（UEG）+ 大语言模型（LLM）结合的差异分析框架；

**🔧 技术方法**

使用了 DEX 反编译、字节码归一化、类源映射、语言无关的统一执行图、LLM 推理与 RAG、投票机制等技术；

**📊 数据集**

使用了 Android 14、15、16 的 AOSP 源码与生成的系统镜像（约 150k–230k 源文件）；

**📈 对比分析**

通过对 329 对并行方法进行自动对比，发现 372 个行为差异，过滤后标记 37 个潜在漏洞，手工验证后确认 11 条漏洞，其中 3 条已被 Google 确认并发布 CVE，整体性能在每个镜像上耗时约 5.2 小时；

**⚠️ 局限性**

受限于 LLM 的生成不确定性、对复杂调用链的解析精度有限、以及仅关注权限与核心框架的安全边界，未覆盖所有潜在攻击向量或跨语言交互细节。

---

## 443. Video-Based Prediction of In-Flight Particle Characteristics in Atmospheric Plasma Spraying

**arXiv ID:** 2606.07416 | [PDF](https://arxiv.org/pdf/2606.07416v1)

**作者:** Abhijeet Praveen `[一作]` (McGill University and Mila Quebec AI Institute), Narges Armanfard `[通讯]` (McGill University and Mila Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过高速视频记录APS（大气等离子体喷涂）过程，提取喷射羽流的几何与频域特征，并利用多种机器学习模型预测飞行粒子的温度与速度；同时直接对原始视频帧使用预训练ResNet18+回归或CNN+LSTM架构做预测，验证视频信息本身对粒子特性的预测价值。

**💡 创新点**

①提出三种基于视频的特征表示（几何平均GA、几何窗口GW、光谱-时域窗口STW），并首次将TabPFN等大规模预训练Transformer模型应用于小样本APS预测；②展示预训练CNN+回归头即可在温度上达到R²≈0.90、速度R²≈0.82；③通过对比TabPFN、CNN、传统树模型等，系统评估不同特征与模型组合在小数据环境下的优势。

**🔧 技术方法**

视频预处理与分段、灰度化、下采样、ROI分割；几何特征（面积、周长、纵横伸展）与光谱特征（频谱、功率、相关时间、质心速度等）提取；TabPFN（Transformer、在上下文学习中利用合成任务知识）、随机森林、梯度提升、SVR、XGBoost、卷积神经网络（1D CNN）、预训练ResNet18 + 回归头、CNN+LSTM；Grouped leave‑one‑out CV、R² 与 PAM（2.5/5/10%）评估。

**📊 数据集**

63个喷涂实验（共126段视频，63段预喷+63段后喷），采集Accuraspray测得粒子温度与速度标签，视频15fps、6s长，灰度化后提取特征；同时保留原始RGB帧供深度模型使用。

**📈 对比分析**

采用分组留一交叉验证，比较不同模型与特征组合的R²与PAM。结果显示：TabPFN在温度预测上达到R²=0.86（PAM@10%≈0.74）；CNN在速度预测上R²=0.81（PAM@10%≈0.68）。原始视频模型中预训练ResNet18+回归头温度R²=0.90、速度R²=0.82，CNN+LSTM略低。整体性能证明视频信息对粒子特性预测具有显著作用。

**⚠️ 局限性**

样本量有限，仅覆盖固定喷枪与过程参数；模型未在更大或更异质数据集上验证泛化；原始视频模型未尝试端到端微调，可能存在潜在提升空间；未结合多模态传感器（声学、过程参数）进一步提升精度；实验环境与工业现场差距仍待评估。

---

## 444. Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Agent Skills

**arXiv ID:** 2606.07412 | [PDF](https://arxiv.org/pdf/2606.07412v1)

**作者:** Chuan Xiao `[一作]` (Shanghai Jiao Tong University), Lin Qu `[通讯]` (AI Data, Alibaba Group)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Socratic-SWE框架，通过闭环自我演化利用软件工程任务的解题轨迹来生成针对性任务，提升LLM的修复能力。

**💡 创新点**

创新点在于把解题轨迹转化为可检索的技能库，用技能指导任务生成，并加入梯度对齐奖励实现自适应课程。

**🔧 技术方法**

采用RL（GRPO/GDPO）、技能提取、执行验证、梯度对齐奖励以及LLM模型（Qwen3.5-9B/ Qwen3.6-27B）等技术。

**📊 数据集**

使用SWE-bench（Verified、Lite、Pro）和Terminal-Bench 2.0作为评测数据集。

**📈 对比分析**

与基线自玩方法比较，在36k实例预算下，Socratic-SWE在Verified上达50.40% (+7.80)，在Lite上36.67% (+7.00)，在Pro上22.85% (+5.61)，在Terminal上14.61% (+4.50)，表现最佳。

**⚠️ 局限性**

局限性包括固定仓库池导致后期难以寻找新能力缺口、依赖可信验证集、需要可执行验证与沙箱环境、以及对不同语言和工作流的泛化尚未验证。

---

## 445. Spline Policy: A Structured Representation for Robot Policies

**arXiv ID:** 2606.07386 | [PDF](https://arxiv.org/pdf/2606.07386v1)

**作者:** Mengze Tian `[一作]` (École Polytechnique Fédérale de Lausanne), Sylvain Calinon `[通讯]` (Idiap Research Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出将现代模仿学习策略的输出接口改为分段样条参数化（Spline Policy），保持原有网络架构不变，利用样条实现连续轨迹解码、流场转换、局部误差恢复、以及不确定性传播与控制器融合。

**💡 创新点**

创新点在于：①将动作块替换为可解析的样条参数，兼容多种高性能背骨；②通过解析距离场构造向量场，将样条转换为局部纠正的流场；③支持在参数空间进行约束、重采样、以及不确定性传播，且无需重新训练。

**🔧 技术方法**

采用了分段二次 Bernstein 样条、解析距离场与向量场构造、扩散（Diffusion Policy）、流匹配（Flow Matching）、Transformer、VLA 等多种策略背骨；梯度反向传播、蒙特卡洛采样用于不确定性估计。

**📊 数据集**

实验使用 LASA 手写轨迹数据、MuJoCo/Sapien/Pymunk 机器人模拟抓取/推送任务，以及 120 条人类示范的真实机器人实验（酒瓶-酒杯等场景）。

**📈 对比分析**

通过与基线动作块模型 BL‑Diff/BL‑Flow 在低维扰动、观测噪声、模拟抓取任务和真实机器人任务中的对比，SP 在保持任务得分相近的同时显著降低输出维度和前向 FLOPs；在扰动恢复和不确定性传播实验中，Flow 版 SP 取得最低 Chamfer、终点误差和最大速度，证明局部纠正效果明显。

**⚠️ 局限性**

局限性包括：①仍需依赖强大的背骨，样条预测不佳时无法保证任务成功；②局部纠正仅适用于预测的样条，无法保证全局稳定；③对高度不连续或动态碰撞场景不太适用；④流场实现仅适用于二次样条，扩展到更高阶或不同族样条需进一步研究。

---

## 446. Do Coding Agents Deceive Us? Detecting and Preventing Cheating via Capped Evaluation with Randomized Tests

**arXiv ID:** 2606.07379 | [PDF](https://arxiv.org/pdf/2606.07379v1)

**作者:** Thanawat Lodkaew `[一作]` (University of Tokyo), Takashi Ishida `[通讯]` (RIKEN)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CapCode框架，通过在编码数据集里加入随机化测试和已知上限B，刻意把可达最高得分设为低于1，从而使得任何超过B的高分即为作弊；同时基于CapCode原理设计CapReward奖励机制，在RL微调时把奖励在B处达到峰值，超出B则被惩罚，从而抑制奖励劫持。

**💡 创新点**

创新点在于：① 用已知的“cap”值将可达最高分刻意压低，使得超出该阈值的得分可被统计检验为作弊；② 通过单侧二项检验给出作弊的显著性判断；③ 将该“cap”理念嵌入奖励函数CapReward，形成一条在B处取最大值、超出B则递减的奖励曲线，解决传统奖励因单调导致的作弊激励。

**🔧 技术方法**

技术手段包括：随机化测试生成、任务级与案例级CapCode构造、单侧二项检验、CapReward奖励函数设计、GRPO强化学习框架、Kullback‑Leibler正则化、统计评估与Kendall's τ相关性分析。

**📊 数据集**

实验使用的编码数据集包括MBPP+、HumanEval+、LiveCodeBench、BigCodeBench，并对训练集做CapCode/ImpossibleBench变体。

**📈 对比分析**

与原始评测对比，CapCode在保持模型排名（Kendall τ 0.94~0.98）的同时能在反馈、提示、工作区暴露三种作弊情境下及时检测到作弊；CapReward在RL微调中显著降低隐藏测试的性能差距，提升隐藏测试表现，并在无作弊场景下保持与基线相当。

**⚠️ 局限性**

局限性包括：① 仅适用于基于单元测试的评测；② 对低于cap但仍存在作弊的策略（非“有效”作弊）抑制有限；③ 若模型充分了解CapCode构造方式，可能通过自适应逃避检测。

---

## 447. Whisper Hallucination Detection and Mitigation via Hidden Representation Steering and Sparse AutoEncoders

**arXiv ID:** 2606.07473 | [PDF](https://arxiv.org/pdf/2606.07473v1)

**作者:** Georgii Aparin `[一作]` (AI Foundation and Algorithm Lab), Assel Yermekova `[通讯]` (AI Foundation and Algorithm Lab)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究 Whisper 模型内部表示对 hallucination（虚假转写）的检测与抑制，提出基于稀疏自编码器（SAE）的 steering 方法，并对比了激活空间 steering 与 SAE 领域 steering 的效果。

**💡 创新点**

创新点在于：①首次证明 Whisper 编码器的内部激活与 SAE 潜在空间中均可线性分离 hallucination 信息；②利用 SAE 的稀疏特性，仅调节少量维度即可显著降低 hallucination；③实现无参数微调的 fine‑tuning‑free 方案，并与 Calm‑Whisper 等 decoder‑level 方法直接对比。

**🔧 技术方法**

技术包括：Whisper Transformer 的音频编码器激活提取、稀疏自编码器（Batch‑Top‑k 变体）训练、线性分类器（Logistic Regression）评估可分离性、Contrastive Activation Addition（CAA）构造 steering 向量、加法与乘法 SAE steering 两种方式。

**📊 数据集**

使用的多语言数据集：非语音数据——MUSAN、WHAM!、FSD50k（包含过滤版）、UrbanSound8K；语音数据——LibriSpeech (en), FLEURS (en, zh), AISHELL‑1 (zh)。

**📈 对比分析**

比较方法：Whisper baseline、Whisper + 激活空间 steering、Whisper + SAE 领域 steering、Calm‑Whisper（decoder‑head masking 与 fine‑tuning）。在 Whisper large‑v3 上，SAE steering 可将 UrbanSound8K 的 hallucination rate 降至 19.88%（vs 95.98% baseline）并保持 WER 与 Calm‑Whisper fine‑tuned 相近；在 Whisper small 上，SAE steering 将 hallucination rate 从 67.09% 降至 8.68%，并在英语语音上改善 WER。相比之下，激活 steering 效果不如 SAE steering，且 Chinese CER 在两种 steering 下均有所下降。

**⚠️ 局限性**

局限性：① SAE 训练仅基于英语音频，对中文等非英语语料效果欠佳；② steering 仅在单层（最终编码层）进行，未探索多层协同策略；③ 只针对非语音输入的 hallucination，未覆盖语音中潜在的虚假片段；④ 评估集中在 Whisper 预训练模型，未验证更大规模模型或其他 ASR 架构的泛化。

---

## 448. Planning-aligned Token Compression for Long-Context Autonomous Driving

**arXiv ID:** 2606.07464 | [PDF](https://arxiv.org/pdf/2606.07464v1)

**作者:** Zhixuan Liang `[一作]` (NVIDIA Research), Marco Pavone `[通讯]` (NVIDIA Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种面向规划的工作记忆框架，通过条件 VQ‑VAE 对单一 Vision‑Action 模型的历史观测进行压缩，实现在有限 token 预算内保留决策关键信息；

**💡 创新点**

创新点在于将压缩过程与规划目标耦合：利用后验编码器从未来轨迹中提取驾驶意图，前置编码器预测该意图，并通过向量量化形成离散技能嵌入，训练时通过 KL 与归纳约束使压缩的历史信息与后续决策直接相关；

**🔧 技术方法**

使用技术包括 Q‑former 作为压缩模块、分层时间缓冲区、条件 VQ‑VAE（posterior 与 prior 编码器）、RoPE 位置编码、Transformer 主干、vector quantization、闭环端到端优化、以及与 Alpasim 的实时仿真评估；

**📊 数据集**

主要实验数据集为 Alpamayo 及 Physical AI 数据集，构建记忆相关子集并在 NuRec/Alpasim 进行闭环评估；

**📈 对比分析**

与标准 Alpamayo、稀疏/稠密长时序、以及无规划对齐压缩等方法比较，在相同 token 预算下 Go Success Rate 提升至 68.3%（比基线 4.5% 绝对提升），roll‑through 减少 22%，并实现 3.3× 推理速度提升、2.7× 内存缩减；

**⚠️ 局限性**

局限性包括在通用驾驶场景中提升有限，模型对历史长度的依赖需进一步探索，且需在更复杂的遮挡或多对手场景中验证泛化能力。

---

## 449. Time series Foundation Models based on Physics-Informed Synthetic Histories for Cold-Start Photovoltaic Forecasting

**arXiv ID:** 2606.07457 | [PDF](https://arxiv.org/pdf/2606.07457v1)

**作者:** Lorenzo Longarini `[一作]` (Sistemi 2000 srl), Riccardo Rosati `[通讯]` (University of Macerata)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一个零样本管线，利用物理推理生成的合成历史作为时间序列基础模型的上下文，实现光伏装机时的冷启动预测。

**💡 创新点**

创新点在于将物理基合成历史与时序基础模型结合，支持无观测历史的零样本推断，并对三种反馈策略进行系统评估。

**🔧 技术方法**

核心技术包括OPAQUE物理合成器、PVGIS参考生成器、Chronos‑2、TabPFN‑TS、TimesFM 2.5、Moirai 2.0、TiRex等时序基础模型，以及基于天气协变量的上下文编码。

**📊 数据集**

使用了四个公开屋顶光伏数据集：Ausgrid（300户）、DKASC（10户）、UK‑PV（100户）和PVDAQ（30户），共计440个站点。

**📈 对比分析**

通过与持久性、季节性持久性以及Prophet等基线模型在Cold‑Start Baseline、Real Feedback和Self‑Forecast Feedback三种情境下的对比，结果显示时序基础模型平均误差低于基线1.7–2倍，其中TabPFN‑TS在Real Feedback下MAE为0.514、RMSE为0.721，Chronos‑2在Self‑Forecast Feedback下最稳健。

**⚠️ 局限性**

局限性包括仅以日尺度评估、单一随机种子实验、假设ERA5气象预报完美以及缺乏高频或多模态输入，导致性能可能被高估。

---

## 450. PaperFlow: Profiling, Recommending, and Adapting Across Daily Paper Streams

**arXiv ID:** 2606.07454 | [PDF](https://arxiv.org/pdf/2606.07454v1)

**作者:** Fuqiang Wang `[一作]` (Key Laboratory of Computing Power Network and Information Security, Ministry of Education, Shandong Computer Science Center, Qilu University of Technology (Shandong Academy of Sciences)), Cheng Tan `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PaperFlow 框架，实现每日科学论文阅读的动态个性化推荐循环，并构建了包含 24 位模拟研究者、1200 天文稿的长期基准。

**💡 创新点**

创新点在于将研究者画像、推荐与适配三大环节统一到日循环中，采用结构化可编辑的研究者档案、跨信号聚合排名、语义化兴趣漂移模型以及阅读报告反馈通道。

**🔧 技术方法**

使用 LLM 进行结构化抽取、语义匹配、优先级推理、阅读报告生成，并结合多信号评分公式、兴趣漂移算法和可解释规则；评估时采用多种自动指标与盲人工评估。

**📊 数据集**

数据集为 arXiv 每日论文流、20,727 篇论文构成的候选池、24 组模拟用户档案以及 497,448 条 episode–paper 记录，配合伪 Oracle 相关性标签。

**📈 对比分析**

与 Scholar Inbox、Citation‑Enhanced、OMRC‑MR、UPR、KUCNet 等基线比较，PaperFlow 在 gNDCG@20、Useful@5/20、Rec.Score 及人类评估得分等指标上均显著优于对手，显示出更高的相关性与行为对齐。

**⚠️ 局限性**

局限在于大部分评测基于模拟用户和伪 Oracle 标签，缺乏大规模真实用户标注；数据覆盖以 arXiv 为主，可能不足以涵盖所有学科与出版渠道。

---

## 451. Agentic Very Much! Adoption of Coding Agent in New GitHub Projects

**arXiv ID:** 2606.07448 | [PDF](https://arxiv.org/pdf/2606.07448v1)

**作者:** Romain Robbes `[一作]`, Stefano Zacchiroli `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对新创建的 GitHub 项目进行编码代理工具采纳度的定量分析，揭示采纳率翻倍且更集中

**💡 创新点**

发现新项目中代理工具使用已普及，并且更易于被检测，表明采纳正加速

**🔧 技术方法**

采用文件/PR/提交级别的启发式检测和 GitHub API 数据提取，维持与先前研究相同的技术流程

**📊 数据集**

使用 Dabic 等人筛选出的约 13,000 个满足 5,000 行代码、100 次提交、10 星且非 fork 的项目数据集

**📈 对比分析**

通过与 21/02/2026 版先前研究结果对比，文件层级采纳率提升四倍，提交比率 20% 以上的项目超过 40%，显示显著性能提升

**⚠️ 局限性**

局限在于只能检测可见的代理痕迹，隐藏的 AI 活动可能导致采纳率低估；且筛选阈值可能偏向高活跃项目，造成样本偏倚

---

## 452. Evidence Markets

**arXiv ID:** 2606.07434 | [PDF](https://arxiv.org/pdf/2606.07434v1)

**作者:** Safwan Hossain `[一作]` (Harvard University), Yiling Chen `[通讯]` (Harvard University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出一种证据市场机制，允许交易者同时提交对事件的信念与支持该信念的证据，以弥补传统预测市场对价格背后理由的透明度不足以及无法处理非时限性问题的局限；

**💡 创新点**

通过将市场流动性参数动态与累计证据质量相关联，设计出一种既可外部也可内部决议的机制，并证明其在外部决议时严格的 DSIC 与内部决议时的 ε‑DSIC；

**🔧 技术方法**

主要运用了对数市场评分规则 (LMSR)、自动化做市商 (AMM) 的成本函数框架、软最大 (softmax) 解算、LLM 作为判定器以及质押式争议验证机制；

**📊 数据集**

论文以大型语言模型 (LLM) 评估为实例进行讨论，但未使用公开数据集，而是基于理论构造的质量函数与验证流程；

**📈 对比分析**

与传统预测市场相比，论文通过理论证明平台的最大损失被 β(R₀)log n 限制，风险厌恶者可通过仅提交证据获得非负回报，整体收益与风险控制均可在理论上得到保证；

**⚠️ 局限性**

主要限制在于验证机制对 LLM 判定器与争议激励的依赖，实际部署时需解决中心化风险、LLM 误判、证据获取成本未建模等工程与安全挑战。

---

## 453. Rapid co-design of Buoyancy-assisted robots for Challenging Locomotion using Gaussian Evolutionary Specialists

**arXiv ID:** 2606.07424 | [PDF](https://arxiv.org/pdf/2606.07424v1)

**作者:** Ankit Sinha `[一作]` (Georgia Institute of Technology), Sehoon Ha `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了 Gaussian Evolutionary Specialists (GES) 框架，通过迭代高斯区域演化将设计空间分区并训练多专家策略，从而避免统一策略与 MoE 失效时的行为多样性崩溃，提升共设计效率；

**💡 创新点**

将设计空间分区与专家学习解耦，利用高斯区域的迭代演化维护专家行为多样性，解决统一策略梯度平均导致的单一行为以及 MoE 表征崩溃问题；

**🔧 技术方法**

结合模型无关强化学习（PPO）、Mixture-of-Experts、Farthest Point Sampling 与 Lloyd 迭代、Monte Carlo 覆盖评估及贝叶斯优化；

**📊 数据集**

在 BALLU 平台的 2D/3D 设计空间内使用 Isaac Lab 2.0/Isaac Sim 4.5 进行仿真，评估 1000 次测试设计；硬件实验在物理 BALLU 机器人上进行；

**📈 对比分析**

与单一 MLP 与端到端 MoE 进行对比，GES 在障碍穿越任务中平均提升 18–25% 以上、在坡度爬升中提升约 13%，硬件上实现 24 cm 高障碍，比基线提高 3 倍；

**⚠️ 局限性**

需预设专家数量 K，受初始化与设计空间维度影响；仅验证单一气动减重平台；假设设计空间连续且有界。

---

## 454. Sparsely gated tiny linear experts

**arXiv ID:** 2606.07414 | [PDF](https://arxiv.org/pdf/2606.07414v1)

**作者:** Simon Schug `[一作]` `[通讯]` (Princeton University), Simon Schug (Princeton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种稀疏线性神经元（Sparsely Gated Linear Neurons, SGLN）作为Transformer前馈层，去除非线性，采用稀疏门控仅激活少量单个线性专家，显著降低每个Token的 FLOPs 并提升可解释性。

**💡 创新点**

核心创新在于：1) 将专家压缩为单个线性神经元并去掉激活函数，形成极高稀疏度且全线性可分析的前馈电路；2) 通过门控网络与 top‑k 选择实现子线性子空间的动态组合；3) 直接利用门控权重空间进行解释与因果干预，无需额外训练替代模型。

**🔧 技术方法**

主要技术包括：门控网络 + top‑k 选择；线性单元权重组合（w_out·w_inᵀ）；子线性矩阵低秩电路；UMAP 可视化门控权重空间；因果干预（NIE）评估前馈电路对事实记忆的影响。

**📊 数据集**

实验使用两个数据集：1) SlimPajama 627B（大型自回归语言建模）用于性能评估；2) TinyStories（简短儿童故事）用于小规模可解释性与因果干预实验。

**📈 对比分析**

在 IsoFlop（计算匹配）比较中，SGLN 在 1e17 至 6e18 FLOPs 预算下与 Dense GeLU、SwiGLU、以及 GPT‑OSS/PEER MoE 竞争，尤其在高预算时表现最优；Ablation 结果表明去除非线性和自定义路由均不降低性能，甚至略有提升。

**⚠️ 局限性**

局限性包括：只改进前馈层，注意力等其他密集模块未优化；IsoFlop 结果不一定对应真实 wall‑clock 时间；解释性实验仅在小模型/小数据集上验证，需进一步验证在更大模型上的有效性；需要进一步整合其他可解释方法与硬件加速实现。

---

## 455. Earliest query answering over streamed trees

**arXiv ID:** 2606.07408 | [PDF](https://arxiv.org/pdf/2606.07408v1)

**作者:** Mateusz Gienieczko `[一作]` (Technical University of Munich), Charles Paperman `[通讯]` (Université de Lille)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种流式算法，在只读树状文档流的情况下对任意一元 MSO 查询实现最早查询回答，并在数据复杂度下保证常数更新时间、常数枚举延迟和紧凑的即时内存占用。

**💡 创新点**

创新点在于：①设计了多栈候选数据结构与日志机制，使得在保持最早回答的同时能在每一步以常数时间更新候选节点的状态；②给出了一般性理论框架，证明所有可由 MSO 表达的一元查询都可在数据复杂度下实现最早回答；③通过对森林的嵌套单词编码与确定性自动机的结合，得到最优的空间与时间保证。

**🔧 技术方法**

利用的技术包括：MSO 逻辑与确定性森林自动机的对应关系；嵌套单词（term encoding）对树的流式表示；基于栈的状态维护（包括上、下、当前三部分）；多栈候选数据结构与日志（Move/Invalidate 事件）实现候选节点的快速插入、提取、状态更新；证明常数更新、常数延迟与空间上 O(|Cands|+f(φ)·depth) 的复杂度。

**📊 数据集**

未给出实验或数据集，本研究为纯理论分析。

**📈 对比分析**

与以往的流式查询引擎（如 rsonpath、JSONSki）、早期回答理论（Gauwin 等）以及最近的 Muñoz‑Riveros Δ‑枚举等方法比较，本文证明在数据复杂度下可实现常数更新、常数枚举延迟并且满足最早回答的最佳性能；由于缺乏实验，未给出具体数值对比。

**⚠️ 局限性**

局限性包括：①常数更新时间与空间的系数可能随查询的复杂度显著增大，未给出组合复杂度分析；②对子树相等或 JSONPath 过滤等更复杂操作支持有限；③未考虑解析层面的跳过/剪枝优化；④未通过实验验证理论结果；⑤对 MSO 公式编译为自动机的指数级开销未被缓解。

---

## 456. Making the Most of Limited Data: Score-Aware Training for Text-to-Music Generation

**arXiv ID:** 2606.07387 | [PDF](https://arxiv.org/pdf/2606.07387v1)

**作者:** Yun-Chen Cheng `[一作]` (National Taiwan University), Chih-Pin Tan `[通讯]` (National Taiwan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出score-aware训练框架，在ICME 2026 ATTM挑战中对文本到音乐生成进行训练，利用CLAP分数对段落进行过滤、噪声时间表调度、两阶段字幕重写以及REPA对齐等步骤；

**💡 创新点**

创新点在于将音频‑字幕对齐分数作为全流程监督信号，采用CLAP条件Beta噪声调度将低分段路由到高噪声训练；通过两阶段字幕重写桥接训练与推理分布；以及REPA辅助损失将预训练语义知识注入模型；

**🔧 技术方法**

技术手段包括FluxAudio Diffusion Transformer+ACEStep 1.5 codec、CLAP编码器、MuQ编码器、T5序列编码器、Beta分布噪声时间调度、REPA对齐辅助损失；

**📊 数据集**

数据集为MTG‑Jamendo子集（CC授权），在标准化的有限数据上训练约450 M参数模型；

**📈 对比分析**

评估方式为ICME 2026 ATTM效率赛的客观指标（CLAP分数、FAD、CCS）和主观MOS，最终在客观赛道两轨道均排名第二，效率赛道第三；

**⚠️ 局限性**

局限性包括MuQ对齐收敛慢、对CLAP分数的依赖可能忽视细节语义、对超参数和训练时长高度敏感；

---

## 457. RhinoVLA Technical Report

**arXiv ID:** 2606.07383 | [PDF](https://arxiv.org/pdf/2606.07383v1)

**作者:** Huixi Intelligence `[一作]`, Yuxi Liu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一种针对边缘硬件实时部署优化的 Vision‑Language‑Action（VLA）模型 RhinoVLA，并实现了跨机器人共享政策与实例级微调。

**💡 创新点**

创新点包括：①将视觉与上下文 token 视为主要延迟来源，采用 token‑高效 Qwen3‑VL backbone 与连续 Action Expert 减少 VLM 计算；②提出统一的视图注册、72D 物理状态‑动作槽空间与机器人实例 LoRA 的跨机器人接口，兼顾共享学习与个体差异；③在 Huixi R1 SoC 上实现硬件感知编译、混合精度执行与并行视觉编码，突破 10 Hz 关环控制阈值。

**🔧 技术方法**

使用技术：token‑效率 Transformer（Qwen3‑VL）、流匹配行动专家、视图注册、统一槽空间、LoRA 微调、FlashAttention‑style 关注优化、图层融合、INT8 权重量化与 FP16 激活、并行多视角编码。

**📊 数据集**

数据集与实验：AgBotWorld（G1、G2）真实轨迹、Galbot G1 数据、LIBERO 4 套件（Spatial、Object、Goal、Long）以及多机器人评估场景。

**📈 对比分析**

与现有 VLA 模型（π_0、π_0.5、OpenVLA、CoT‑VLA 等）对比，RhinoVLA 在 LIBERO 上平均成功率 90.0%，在单臂/双臂任务上与 π_0.5 相当或更优；在 Huixi R1 上实现 11.69 Hz 的端到端推理，满足 10 Hz 实时闭环需求。

**⚠️ 局限性**

局限性：1) 与 π_0.5 的性能仍存在一定差距；2) 仅在 Huixi R1 上验证，跨平台泛化尚需进一步评估；3) 需要针对不同 VLM backbone 进行重训练，适配成本仍不低。

---

## 458. Online Pandora's Box for Contextual LLM Cascading

**arXiv ID:** 2606.07392 | [PDF](https://arxiv.org/pdf/2606.07392v1)

**作者:** Alexandre Belloni `[一作]`, Yehua Wei `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一种在线上下文潘多拉盒模型，用于自适应查询和选择大型语言模型（LLM）API，解决了在每个请求上下文中如何有效地管理查询和选择的问题。

**💡 创新点**

创新点在于将传统的潘多拉盒模型扩展到在线上下文设置，特别是引入了输出介导的反馈结构，并提出了一种基于保留指数的学习方法。

**🔧 技术方法**

使用了广义矩估计（GMM）和上置信界（UCB）风格的置信界来估计保留指数，并结合了对共享输出级奖励评估器的学习。

**📊 数据集**

未具体提及使用的数据集，但研究背景涉及多个LLM API的查询和选择，暗示可能使用了来自实际业务请求的历史数据。

**📈 对比分析**

与传统的潘多拉盒模型相比，提出的模型在维度依赖的O(√(T))累积遗憾上具有理论保证，表明其在性能上优于现有的启发式方法。

**⚠️ 局限性**

限制在于模型假设了上下文和奖励函数的特定结构，可能在实际应用中面临数据稀缺和模型不确定性的问题。

---

## 459. An End-to-End Encrypted Control Pipeline for Multi-Agent Coordination via CKKS Homomorphic Encryption

**arXiv ID:** 2606.07375 | [PDF](https://arxiv.org/pdf/2606.07375v1)

**作者:** Sai Sandeep Damera `[一作]` (University of Maryland), John S. Baras `[通讯]` (University of Maryland)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一个完整的基于CKKS的多代理隐私保护协同控制流水线。

**💡 创新点**

提出了用对角线法统一处理任意拓扑的图拉普拉斯运算，并给出周期性重置的误差闭式上界，实现了从观测到控制的全流程FHE实现。

**🔧 技术方法**

采用CKKS全同态加密、预计算的卡尔曼增益与矩阵指数、对角线分解与循环旋转、周期性引导重置及分离原理误差分析。

**📊 数据集**

以9个双积分子系统在三种拓扑（环、3×3环面、完全图）为例进行数值仿真，无公开数据集。

**📈 对比分析**

将加密与明文轨迹进行对比，误差保持在10⁻⁶水平；计算时间约5.5 s/周期，完成率≈0.18 Hz，满足慢速动态应用。

**⚠️ 局限性**

受限于CKKS多级深度和重置噪声，当前仅适用于线性、慢速系统，非线性、快速场景仍需进一步优化。

---

## 460. Verifiable and Confidential DNN Inference on Low-End Edge Devices

**arXiv ID:** 2606.07470 | [PDF](https://arxiv.org/pdf/2606.07470v1)

**作者:** Mohamed Khalil Kiri `[一作]` (EURECOM), Norrathep Rattanavipanon `[通讯]` (Prince of Songkla University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于 ARM TrustZone‑M 的第三运行时环境（即在 Secure World 与 Non‑Secure World 之间引入的新权限层），并利用该层实现 DNN 推理的机密性保护与可验证性。通过授权机制控制推理次数并使用 Proof‑of‑Execution 生成可验证的推理证明，最终在低端 MCU 上实现了安全、可验证且低开销的深度学习推理。

**💡 创新点**

创新点主要有：
① 引入第三运行时环境，实现推理代码在非安全世界执行，同时保持对 Secure World 的隔离；
② 仅在 Secure World 维护极小的可信计算基（TCB），大幅度降低安全代码量；
③ 将推理授权与 PoX 结合，既防止模型窃取，又能对推理结果进行远程验证；
④ 通过 TSDP 对模型进行分区，仅对私有子网络加密，进一步减少内存占用与攻击面。

**🔧 技术方法**

技术手段包括：ARM TrustZone‑M（SAU/IDAU、Secure DMA 控制器）、Secure‑World API（Install、Update、Instantiate、Execute、Destroy）、对称加密与非对称签名、授权令牌、Proof‑of‑Execution、TSDP 模型分区策略、低级硬件隔离与内存映射管理。

**📊 数据集**

使用 ResNet 模型（典型图像分类网络）进行实验，模型被分为私有子网络（≈38.6 KB）与公共子网络（≈37.8 KB），在 NUCLEO‑L552ZE‑Q（ARM Cortex‑M33）开发板上实现。

**📈 对比分析**

评估与基线对比：基线完全在 Secure World 运行，TCB 约 34,865 LOC；本文实现仅 1,523 LOC（占比 4.36%）。内存方面，基线需 76.375 KB Secure RAM（不可用），本文仅将私有子网络加密后存放，公共子网络直接在 Flash；运行时延迟仅增加 0.83 ms（0.07%）；PoX 生成签名约 230 ms，整体推理时延约 1,162 ms。实验表明在保持安全性的同时，几乎没有性能损失。

**⚠️ 局限性**

局限性：
① 仅适用于 ARM TrustZone‑M，无法迁移至缺乏 MMU 的更低端 MCU；
② 未针对侧信道、故障注入或硬件篡改等物理攻击提供防护；
③ PoX 产生的签名开销对实时性要求极高的场景仍显巨大；
④ 需要模型分区，模型规模过大时仍受 Secure RAM 限制；
⑤ 对授权和密钥管理的依赖需要安全的设备上电与密钥存储方案。

---

## 461. On orbital stabilization of a circular motion primitive for a dynamic extension of the Dubins car model

**arXiv ID:** 2606.07449 | [PDF](https://arxiv.org/pdf/2606.07449v1)

**作者:** Artem Angelchev-Shiryaev `[一作]` (Lund University), Leonid B. Freidovich `[通讯]` (Umea University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

研究并实现了动态扩展 Dubins 车模型在圆周轨道上的轨道稳定控制，提出了可行的控制设计方案。

**💡 创新点**

提出了新的可行性条件 (PL)，要求变分动态存在三维不变稳定子空间，并通过坐标变换将不可稳定的部分分离，从而克服传统横向线性化无法稳定化的难题。

**🔧 技术方法**

采用了横向坐标、变分动力学、时变线性系统分析、控制器的时变反馈设计以及线性二次调节（LQR）等技术。

**📊 数据集**

未使用公开数据集，仅通过数值仿真验证所设计控制器在给定参数下的性能。

**📈 对比分析**

与传统基于横向线性化且不可稳定化的方案对比，表明利用 (PL) 条件和时变反馈能够实现轨道稳定，并在仿真中实现指数收敛。

**⚠️ 局限性**

实验仅针对单一的 Dubins 车圆周轨道示例，方法的推广到更一般的非线性、约束或多自由度系统仍需进一步研究。

---

## 462. Skill-3D: Evolving Scene-Aware Skills for Agentic 3D Spatial Reasoning

**arXiv ID:** 2606.07436 | [PDF](https://arxiv.org/pdf/2606.07436v1)

**作者:** Haoyuan Li `[一作]` (Zhejiang University), Yi Yang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Skill-3D 框架，构建场景记忆与可演化技能库，使多模态大语言模型（MLLM）在室内 3D 空间推理中根据场景动态选择工具和流程；

**💡 创新点**

通过自我演化的场景感知技能实现工具使用的精准匹配，解决传统方法工具偏好与工具使用率低的问题；

**🔧 技术方法**

结合场景记忆、动态技能提炼与合并、技能检索与筛选、基于技能的监督微调及 GRPO 强化学习；

**📊 数据集**

使用 VSI-Bench、BLINK、CV-3D、MMSI-Bench 等室内 3D 位置与关系推理基准；

**📈 对比分析**

与无工具、直接工具使用、Think3D 等基线对比，Skill-3D 将工具使用率从 39% 提升至 78%，Gemini-3-Flash 在 MMSI-Bench 提升 67%，Qwen3-VL-8B 在 VSI-Bench 提升 43%；

**⚠️ 局限性**

仅在室内 3D 推理场景验证，缺乏对户外、实时机器人等更复杂场景的适配与安全约束，需进一步扩展工具接口和安全机制。

---

## 463. A Comprehensive Anatomy of Human and DeepSeek-R1 LLM Mathematical Reasoning

**arXiv ID:** 2606.07410 | [PDF](https://arxiv.org/pdf/2606.07410v1)

**作者:** Yuxiang Chen `[一作]`, Jun Wang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对 DeepSeek‑R1 在 AIME 2025 30 道题的 10,247 步长 Chain‑of‑Thought 轨迹进行人工功能级标注，比较其与人类解题过程，发现表面结构相似但功能缺失，提出“拓扑模仿”和“尺度失配”等失败模式；

**💡 创新点**

创新点在于将推理过程拆解为五种功能模式（分析、推理、分支、回溯、反思）并引入跨轨迹方差与反思时机两项评估指标，以及针对深度回溯、对比优化、上下文正则化和推理时间计算重分配四条训练改进方向；

**🔧 技术方法**

使用手工标注的功能分类、统计分析、三步 N‑gram 模式、图形可视化和可解释性评估技术；

**📊 数据集**

数据集为 AIME 2025 30 道数学竞赛题的完整推理轨迹，包含 10,247 步；

**📈 对比分析**

通过与人类参考解的对比，发现成功轨迹的探索动作方差低、反思嵌入推理层级；失败轨迹方差高、反思位置错误；实验验证了这些指标在区分正确与错误轨迹方面的有效性；

**⚠️ 局限性**

仅在单一模型和单一基准上验证，人工标注成本高且难以自动化，结果可能难以推广到其他任务或模型。

---

## 464. M$^3$Exam: Benchmarking Multimodal Memory for Realistic User-Agent Interactions

**arXiv ID:** 2606.07402 | [PDF](https://arxiv.org/pdf/2606.07402v1)

**作者:** Zhengjun Huang `[一作]` (Hong Kong University of Science and Technology), Xiaofang Zhou `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套面向多模态会话记忆的查询中心化基准（M^3Exam），并基于此基准开发了一种基于查询模态偏置的多模态记忆方法（M^3Proctor）。

**💡 创新点**

创新点包括：①构建真实、长周期、多模态对话记忆数据集；②将跨模态推理与隐式推断纳入评价维度；③提出模态偏置检测+级联检索策略，在保持多模态准确性的同时显著降低检索成本与索引时间。

**🔧 技术方法**

技术手段主要包括：文本与视觉的文本代理生成、模态标签与查询模态向量的推理、基于模态偏置的重排序、成本感知的多级检索级联；实验中使用大语言模型（如Qwen-2.5-VL-7B、GLM-5.1等）与多模态记忆系统（UniversalRAG、Mem0等）。

**📊 数据集**

使用的核心数据集是自构造的M^3Exam：239条多轮对话，15个角色场景，1,799个多模态文件，5,150道评测问题，覆盖单轮问答、跨模态推理与隐式推断等维度。

**📈 对比分析**

与前沿MLLM（如GLM-5.1）以及多模态记忆系统（如UniversalRAG、Mem0等）对比，M^3Proctor在总体得分上达0.484，显著提升跨模态推理（0.606）与隐式推断（0.652），同时平均每次查询的token消耗仅为4,591，索引构建时间72秒，较传统多模态系统低70%~80%的成本，且在跨模态任务上超越多数记忆系统。

**⚠️ 局限性**

局限性：仅针对单轮问答评测，未覆盖多轮动态会话与迭代记忆更新；对隐式推断的处理仍有限，缺乏更深层次的上下文理解与用户隐性状态建模。

---

## 465. Generative Modeling of Discrete Latent Structures via Dynamic Policy Gradients

**arXiv ID:** 2606.07400 | [PDF](https://arxiv.org/pdf/2606.07400v1)

**作者:** Stefan Ivanovic `[一作]` (University of Illinois at Urbana-Champaign), Mohammed El-Kebir `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 GReinSS 框架，使用动态奖励的策略梯度直接最大化间接观测数据的似然，从而学习离散潜在结构的分布并进行推断。

**💡 创新点**

创新点在于：① 将策略梯度与最大似然优化结合，动态缩放奖励实现无偏梯度估计；② 推导出最优离线采样分布；③ 在组合性极大的潜在空间上提供理论保证和实用算法。

**🔧 技术方法**

技术手段包括策略梯度、动态奖励、重要采样/离线采样、GFlowNets 对比、EM、GEM、VAE、扩散模型、自动回归生成等。

**📊 数据集**

实验数据集：模拟图结构推断、集合推断；真实数据为 GTEx 短读 RNA‑seq（61 个匹配长读样本），与长读基准（FLAIR）对照。

**📈 对比分析**

与局部搜索、GFlowNets、无动态奖励的策略梯度、GEM‑VAE/自回归、扩散等基线对比。GReinSS 在模拟图实验中 F1 分数最高（k=10 时 0.891；k=1000 时 0.95+）；在集合实验中对大宇宙 U=1000 的 F1 ≈0.94；在 RNA isoform 任务中相较 RSEM 的平均误差下降约 0.04，46.6% 基因优于 RSEM，9.4% 基因则相反。

**⚠️ 局限性**

局限性：① 仍假设观测概率 (X|S) 已知；② 仅实现策略梯度，未尝试 Q‑learning 或 Actor‑Critic；③ 离线采样仍依赖启发式，未完全达到理论最优；④ 对极大组合空间需额外技巧；⑤ 适用范围受限于可建模的生成流程。

---

## 466. Mind the Gap: Disentangling Performance Bottlenecks in Video Instance Segmentation

**arXiv ID:** 2606.07394 | [PDF](https://arxiv.org/pdf/2606.07394v1)

**作者:** Danial Hamdi `[一作]` (Amirkabir University of Technology), Mahdi Javanmardi `[通讯]` (Amirkabir University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出基于整数线性规划（ILP）的诊断框架，分离视频实例分割中的跟踪、分类和掩码质量误差，并配备可视化工具TrackLens。

**💡 创新点**

创新点在于模型无关的全局最优跟踪oracle（T-Oracle）与去分类oracle（TC-Oracle）的分层设计，能量化各误差源，并通过TrackLens实时揭示错误原因。

**🔧 技术方法**

主要技术包括ILP求解（OR-Tools）、查询式VIS架构、帧级IoU聚合、类一致性约束以及交互式可视化。

**📊 数据集**

使用YouTube-VIS 2019/2021验证集和遮挡强的OVIS诊断子集进行实验。

**📈 对比分析**

将框架应用于七种在线/离线VIS方法，发现跟踪误差是主要瓶颈（20+ AP差距），分类误差次要；提升骨干网络可显著降低分类差距，但对AP跟踪差距影响有限。

**⚠️ 局限性**

局限性包括：ILP约束近似而非绝对最优；仅适用于查询式模型；标准体量化指标对在线方法产生非因果累积误差；OVIS分析依赖训练集子集。

---

## 467. Is US Defense Acquisition Ready to Acquire AI-Enabled Capabilities? Assessing the DoD Software Acquisition Pathway Through a Scenario-Based Policy Analysis

**arXiv ID:** 2606.07393 | [PDF](https://arxiv.org/pdf/2606.07393v1)

**作者:** Daniel Lugo `[一作]` (Purdue University), James C. Davis `[通讯]` (Purdue University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过情景化政策评估（PSA）方法，对国防部软件获取路径（SWP）在人工智能（AI）能力获取中的适用性进行评估，构造了一个基准AI程序与一个传统软件程序的对照情景，系统追踪规划阶段的关键决策点和产出物，识别政策与程序之间的差距。

**💡 创新点**

创新点在于：①首次将情景化政策评估框架应用于国防采购政策；②通过对比AI与传统软件情景，揭示SWP在AI特性方面的可操作性缺口；③提出AI支持子路径和关键产出物改进建议，为SWP的AI化升级提供实证依据。

**🔧 技术方法**

使用了情景化政策评估（Policy Scenario Analysis）技术，结合结构化情景工作表、三分编码（Explicit/Partial/Absent）和覆盖评分卡，对DoD各项政策文件进行系统映射与分析。

**📊 数据集**

未使用实测数据集，评估基于合成的AI项目情景和国防部政策文件（如DoDI 5000.87、DoDI 5000.89、CDAO AI T&E指导等）。

**📈 对比分析**

通过将基准AI情景与传统软件情景置于同一SWP规划框架下进行对照，采用覆盖率评分卡量化各阶段对AI属性的支持度。结果显示：在AI保证与TEVV方面支持度为Explicit；在数据治理、网络安全与追溯、生命周期管理和人类监督方面仅为Partial，表明SWP在这些关键AI属性上的可操作性不均衡。

**⚠️ 局限性**

局限性包括：①仅关注SWP规划阶段，未评估执行和持续运营阶段；②使用单一合成情景，缺乏多样化真实项目案例；③评估由单名分析师完成，缺乏交叉验证；④缺乏定量性能指标，仅通过定性编码和评分卡表征；⑤结论对国防以外的AI采购环境的外推性有限。

---

## 468. Sort, Partition, Randomize: Optimal Binary Hypothesis Testing under Local Differential Privacy

**arXiv ID:** 2606.07443 | [PDF](https://arxiv.org/pdf/2606.07443v1)

**作者:** Elena Ghazi `[一作]` (Harvard University), Ibrahim Issa `[通讯]` (American University of Beirut)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

设计在纯ε局部差分隐私约束下，用于二元假设检验的最优机制，最大化输出分布之间的f‑散度（包括TV、KL、hockey‑stick等）

**💡 创新点**

证明最优机制可归约为“排序‑分区‑随机化”（SPR）形式，即先按似然比排序，再把符号区分为连续区块，最后对区块标签做随机响应；并给出O(k³)（或O(ℓk²)）的动态规划求解方案；对E_γ散度给出闭式二元机制

**🔧 技术方法**

使用几何/凸分析推导极点结构，证明极点均由LR连续区块的阶梯（staircase）通道实现；利用f‑散度的可加分块性质得到动态规划；采用随机响应与似然比阈值的组合；在闭式结果中用Neyman‑Pearson阈值与随机响应

**📊 数据集**

在实验中使用合成的Dirichlet(1_k)分布对 (P₀,P₁) 生成 100 组测试样例；无真实数据集

**📈 对比分析**

与基于线性规划的Kairouz–Oh–Viswanath阶梯程序对比：在k=6时两者完全一致；在k=100时LP不可行，SPR动态规划在20秒内完成，得到全隐私范围内最优效用，展示了显著的计算优势；在E_γ场景中二元机制达到理论上限

**⚠️ 局限性**

仅适用于纯（非近似）ε‑LDP、非交互式、单纯二元假设和有限字母表；对多假设、复合假设、Rényi或近似LDP、交互式协议以及未知/不确定分布的推广仍是开放问题

---

## 469. Act As a Real Researcher: A Suite of Benchmarks Evaluating Frontier LLMs and Agentic Harnesses in Research Lifecycle

**arXiv ID:** 2606.07462 | [PDF](https://arxiv.org/pdf/2606.07462v1)

**作者:** Jiayu Wang `[一作]` (Xi'an Jiaotong University), Xiangyong Cao `[通讯]` (Xi'an Jiaotong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了首个关注研究实习生层面任务的 AARRI-Bench 基准，旨在评估 LLM 代理在真实科研场景中的细粒度表现。

**💡 创新点**

创新点在于：①将任务按研究者质量（诚信、敏感性、严谨性）细化；②使用双维度（任务场景与代理自主程度）分类构建任务；③强调“人类易懂但代理易错”的设计理念；④对比多种 harness 与模型，揭示最小化代理结构的优势。

**🔧 技术方法**

技术手段包括：Harbor 框架统一任务规范与环境；16种 agent‑model 组合（Claude Code、Hermes Agent、Mini‑SWE‑Agent 等与 Claude‑Opus‑4.7、MiniMax‑M2.7、Kimi‑K2.6 等模型）；0/1 经典奖励与细粒度单元测试两级评估；云端容器化执行确保可复现。

**📊 数据集**

数据集为 82 题手工构造的任务，覆盖四类任务场景（Context、Mindset、Hands‑on、Interaction）和四级代理能力（S1–S4），所有任务均附带 Docker 环境、参考解答与测试脚本。

**📈 对比分析**

比较方法采用 0/1 成功率和细粒度单元测试；结果显示最优配置 Mini‑SWE‑Agent + Claude‑Opus‑4.7 成功率 68.3%，显著高于复杂 harness；模型层次是瓶颈，低阶模型表现差距大；最小化 harness 在高阶模型上展现更强扩展性。

**⚠️ 局限性**

局限性包括：任务规模仍较小；未加入 MCP 与多工具协作；缺乏超长时序任务；评测依赖手写测试代码，未使用 LLM‑as‑judge，导致鲁棒性有限。

---

## 470. A 65 nm Trustworthy Hypoglycemia Forecasting Engine Achieving 11.3 nJ per Inference

**arXiv ID:** 2606.07455 | [PDF](https://arxiv.org/pdf/2606.07455v1)

**作者:** Boyang Cheng `[一作]` (University of Notre Dame), Ningyuan Cao `[通讯]` (University of Notre Dame)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了基于概率决策树（PDT）的65 nm低功耗低血糖预测引擎，采用采样近似软决策树，配备可重构4×24×24 pNode阵列和低功耗RISC‑V核心，能在边缘设备上实时完成血糖预测。

**💡 创新点**

创新点在于：①采用概率决策树实现可解释性和不确定性估计；②提出混合算术与采样的PDT引擎，使用脉冲传播和分布式随机数产生器实现O(N·d)复杂度；③设计4×24×24可重构pNode阵列，支持任意树结构映射并提高硬件利用率；④实现全流程硬件原型，展示能耗仅11.3 nJ/次推理。

**🔧 技术方法**

使用技术包括：概率决策树模型、采样推理、脉冲基随机数生成（LFSR）、4位概率精度、分布式pNode阵列、低功耗RISC‑V MCU、TSMC 65 nm CMOS工艺。

**📊 数据集**

使用OhioT1DM CGM数据集，提取24个5 min分辨率血糖值及其一阶、二阶导数，建立30 min预测窗口，训练并验证模型。

**📈 对比分析**

与RF、SVM、RNN、LSTM等现有算法对比，PDT在敏感度79.7%、精度85.5%、F1 0.825上表现最优；能耗比CPU基线低16.2×、对噪声鲁棒性提升4.1–16.1×；硬件利用率比CAM高1.47×。

**⚠️ 局限性**

局限性包括：仅在OhioT1DM数据集上验证，缺乏多中心临床验证；模型深度上限12，较深树需进一步优化；采样数量与精度权衡需针对不同应用细调；在极低功耗或极高采样率场景下，RISC‑V调度与pNode同步可能成为瓶颈。

---

## 471. Sycophantic Praise: Evaluating Excessive Praise in Language Models

**arXiv ID:** 2606.07441 | [PDF](https://arxiv.org/pdf/2606.07441v1)

**作者:** Daniel Vennemeyer `[一作]` (University of Cincinnati), Tianyu Jiang `[通讯]` (University of Cincinnati)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了‘sycophantic praise’（过度恭维）这一新型对齐问题，并设计了可参数化的SyPr评估框架，衡量模型输出的恭维是否超过上下文所需；

**💡 创新点**

①将恭维与用户期望能力、贡献质量以及恭维目标（人、过程、结果）结合，构建上下文约束的惩罚阈值；②通过SyPr框架对现代LLM的恭维进行量化，发现其在社会解释性任务中普遍过度；

**🔧 技术方法**

使用基于LLM的注释器提取恭维实例和强度；通过逻辑回归（或类似的排序损失）学习上下文约束参数；在此基础上计算SyPr分数；

**📊 数据集**

13,200条交互工件（包含13个领域、不同人物角色、质量水平与提示条件），来源于GSM8K、MMLU（Economics、Chemistry）、MoReBench（道德推理）、深度评估等公开基准；

**📈 对比分析**

与三类基线（LLM judge、RoBERTa分类器、社会型sycophancy指标）对齐人类标签进行AUROC/AP/斯皮尔曼评估；SyPr在所有基线上均显著领先（AUROC从0.7提升至0.919），在独立教师标签上亦保持优势；

**⚠️ 局限性**

评价本质上是规范性、文化相关，阈值和参数非普适；仅评估短交互层面，未测量长期交互或实际用户心理影响，且未覆盖所有使用场景的恭维标准；

---

## 472. A 65 nm Multi-Modal Bayesian Inference Engine with 16.3 fJ/Sample Calibration-Free GRNG for Risk-Aware At-Home Skin Lesion Screening

**arXiv ID:** 2606.07439 | [PDF](https://arxiv.org/pdf/2606.07439v1)

**作者:** Steven Davis `[一作]` (University of Notre Dame), Ningyuan Cao `[通讯]` (University of Notre Dame)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种65nm多模态贝叶斯推理引擎，支持MoG BNN并实现全设备在家皮肤病筛查。

**💡 创新点**

创新点在于可扩展的MoG权重采样与无校准的过程变异Gaussian随机数发生器，实现了低能耗的高通量贝叶斯推理。

**🔧 技术方法**

采用Mixture-of-Gaussian贝叶斯神经网络、基于SRAM的计算存储、LFSR分布选择、双边过程变异GRNG和ResNet-50特征提取。

**📊 数据集**

使用ISIC 2018皮肤病图像数据集（7类不平衡）以及CIFAR-10验证。

**📈 对比分析**

与单模态BNN和确定性NN对比，MoG BNN在平衡准确率提升约1.8%，误负风险覆盖率提升1.4倍，鲁棒性提升>1.5倍，能耗仅16.3 fJ/样本，GRNG吞吐168.6 GSa/s/。

**⚠️ 局限性**

局限在于需要较大存储（每个混合分量需额外存储）、只能在65 nm工艺实现、对极端环境的适应仍需进一步验证，且目前仅在皮肤病筛查任务上证明有效。

---

## 473. Re-imagining ISO 26262 in the Age of Autonomous Vehicles: Enhancing Controllability through Transferability and Predictability

**arXiv ID:** 2606.07437 | [PDF](https://arxiv.org/pdf/2606.07437v1)

**作者:** Chaitanya Shinde `[一作]` (Torc Robotics Inc), Steve Kenner `[通讯]` (Torc Robotics Inc)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并定义了ISO 26262 Controllability的两大子维度Transferability与Predictability，并给出可量化的度量与验证方法。

**💡 创新点**

将Controllability拆分为系统内部的故障转移能力（Transferability）和外部驾驶员可预测性（Predictability），并构建与ISO 26262、SOTIF兼容的标准化评估框架。

**🔧 技术方法**

采用安全分析、故障注入、硬件在环(HIL)、仿真、自然驾驶数据与人机交互模型等统计与信号监测技术。

**📊 数据集**

使用SAE J2944、NHTSA SHRP2等自然驾驶数据集，并结合自定义情境库进行验证。

**📈 对比分析**

通过故障注入实验中的转移成功率与预设阈值对比，以及观测者研究中的四通道预测性监测评估指标是否达到相应类别；论文未给出数值性能指标，仅说明方法可验证、可追溯。

**⚠️ 局限性**

阈值需在特定ODD切片与观测者类别下校准，天然驾驶数据可能不代表最优安全行为，有限样本难以实现高置信度类别，且验证与监测需保持独立性。

---

## 474. Watch, Remember, Reason: Human-View Video Understanding with MLLMs

**arXiv ID:** 2606.07433 | [PDF](https://arxiv.org/pdf/2606.07433v1)

**作者:** Jiahao Meng `[一作]` (Peking University), Minghsuan Yang `[通讯]` (UC Merced)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并系统化了基于大型语言模型（LLM）的长视频理解技术，提出了“观看–记忆–推理”三功能框架，对现有方法、挑战、数据集和评测进行了统一梳理与评析。

**💡 创新点**

创新点在于：①引入人类观看过程的三维分解（Watch、Remember、Reason），为视频 MLLM 设计提供统一视角；②构建统一的形式化描述（perceptual representation、memory state、reasoning trace），帮助对比与归纳不同技术；③整合多种子领域（ egocentric、sports、instructional、medical、叙事影片）与多模态任务（caption、grounding、QA 等），形成完整的生态图。

**🔧 技术方法**

主要技术包括：多模态 Transformer（Q-Former、VideoPerceiver 等）实现细粒度时空感知；外部记忆（树/图/向量库）和压缩策略（token pruning、分层 KV）实现长时序记忆；强化学习/GRPO、DPO 等后训练方法提升推理质量；以及多模态对齐（音频-视频交叉注意、TMRoPE 等）与高效推理（动态帧选择、token merging）等。

**📊 数据集**

涉及的数据集与评测包括：VideoChat2‑IT、LLaVA‑Video‑178K、VideoCoT、Video‑R1、Video‑RFT、STGR、LongVideo‑Reason、ShareGPT4Video、Panda‑70M、VTG‑IT‑120K、TimePro、Video‑Marathon 等多任务、长视频、跨模态的数据集；对应评测指标涵盖文本生成准确率、时间/空间定位精度、推理可信度与数据稀缺度等。

**📈 对比分析**

通过对比表格与案例，作者系统评估了不同方法在“观看”阶段的时空精度与效率、“记忆”阶段的压缩率与检索召回率，以及“推理”阶段的文本可信度与证据可验证性；结果表明，当前多数方法在观看精度和推理可信度方面仍低于人类表现，尤其在长视频稀疏证据、跨模态对齐和实时流处理方面存在显著差距。

**⚠️ 局限性**

局限性包括：①综述受限于已公开方法，缺少对最新闭源或行业内部技术的分析；②缺乏统一的基准评测和公平对比，导致不同方法间性能难以直接比较；③在长视频稀疏证据与实时推理方面仍无通用高效方案，仍需更多针对性的数据与模型创新；④对模型可解释性与安全性（如幻觉、偏见）讨论有限。

---

## 475. The Masked Advantage: Uncovering Local-Language Access to Cultural Knowledge in LLMs

**arXiv ID:** 2606.07422 | [PDF](https://arxiv.org/pdf/2606.07422v1)

**作者:** Yang Zhang `[一作]` (Ecole Polytechnique), Michalis Vazirgiannis `[通讯]` (Ecole Polytechnique)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于差分-差分设计与1PL IRT模型的框架，用来区分大型语言模型在不同语言下的语言熟练度与本土文化知识获取能力；

**💡 创新点**

创新点在于将文化中立与文化特定题目与查询语言（英语/本土语言）交叉，并通过IRT把难度差异消除，从而得出知识获取优势（KnowledgeGap）而非仅靠原始准确率；

**🔧 技术方法**

采用1PL (Rasch) 物品反应理论模型来估计模型能力与题目难度，并用差分-差分公式计算GlobalGap、LocalGap与KnowledgeGap；

**📊 数据集**

使用13种本土语言的区域基准数据（如Global-MMLU、CMMLU、ArabicMMLU、GreekMMLU、MILU、KMMLU、INCLUDE等），并对文化特定题目进行人工翻译与校验；

**📈 对比分析**

通过在13个语种与约80个模型上同时评估英语与本土语言的文化中立/文化特定题目，得到GlobalGap普遍为负，LocalGap受资源与模型来源影响，而KnowledgeGap在98%细胞中为正，表明本土语言在文化知识访问上普遍优于英语；

**⚠️ 局限性**

局限包括不同语种文化特定题目规模不一、区域适配模型样本不足，以及仅基于多项选择题与log‑likelihood评分，未涵盖开放式生成等评估。

---

## 476. Reversible Foundations: Training a 120B Sparse MoE through State-Preserving Scaling

**arXiv ID:** 2606.07404 | [PDF](https://arxiv.org/pdf/2606.07404v1)

**作者:** Rohan Shravan `[一作]` `[通讯]` (School of AI), Rohan Shravan (School of AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在单个八块GPU节点上端到端训练了一个120B参数的稀疏专家混合模型LightningLM 0.1V，并详细记录了从2B密集模型到120B稀疏模型的逐步扩张、可逆前馈、Kronecker嵌入、量化基础+低秩适配器等技术实现。

**💡 创新点**

核心创新在于：① 将可逆前馈堆栈与跨块记忆、跨层混合等多尺度递归统一为单一架构并保持不变；② 在单节点上实现120B稀疏MoE训练的可逆性与参数分片、量化TQP策略的结合；③ 通过分阶段扩张（密集→MoE、深度重排、专家复制+扰动、关键空间首映式检查点）形成可复现的成长原则和失败案例。

**🔧 技术方法**

采用的关键技术包括：可逆递归Transformer（reversible midpoint stack）、DeltaNet线性注意力与可学习稀疏注意力、跨层Hyper-Connections、Kronecker构造嵌入、TurboQuant量化与低秩适配器、ZeRO-3分片、OPUS动态数据选择、以及自动化的配置搜索和自定义的融合内核。

**📊 数据集**

使用的训练数据主要来自多语言OPUS、WebText、CommonCrawl等大规模语料，结合OPUS动态选择器以保证Indic、代码等子域的覆盖；评估数据包括held‑out多域交叉验证，未公开具体数据集划分细节。

**📈 对比分析**

对比方法主要是内部吞吐量和损失稳定性评估：在8块A100/HP800/B200节点上，2B模型达≈59k token/s，5B≈37k token/s，120B≈17k token/s；损失随规模递减，120B在1.78的交叉熵损失；未提供与公开基准模型（如ChatGPT、LLaMA）在标准GLUE/MT benchmarks上的直接数值对比。

**⚠️ 局限性**

局限性包括：① 仅在单节点8块GPU上实现，规模可扩展性受限；② 主要关注训练过程与系统工程，缺乏对下游任务性能的系统评估；③ 依赖多种先前公开技术，核心贡献为工程整合，非新原语；④ 对不同算子实现细节（如量化、可逆实现）高度依赖特定框架版本，复现门槛较高。

---

## 477. RealDocBench: A Benchmark for Field-Level QA and Layout Understanding on Real-World Regulated Documents

**arXiv ID:** 2606.07401 | [PDF](https://arxiv.org/pdf/2606.07401v1)

**作者:** Ameya Joshi `[一作]` (Extend AI), Eli Badgio `[通讯]` (Extend AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文构建了一个基于真实监管文档的双轨评测基准，分别对字段级答案提取（QA Track）和布局检测（Layout Track）进行统一、可复现的评估；

**💡 创新点**

创新点在于：①采用结构化、typed答案而非传统字符串匹配；②将问答与布局两个任务统一在同一文档集合与公共分类上；③使用提取LLM实现跨系统输出格式一致性；④公开数据集、适配器和评测工具，确保可重复性与公平性；

**🔧 技术方法**

技术方法包括：多模型问答生成、手工验证、JSON类型化、提取LLM（Gemini 3 Flash）与严格/调和 F1 匹配；布局使用 Hungarian 匹配与邻接恢复；成本与延迟采用统一缓存破坏测量；

**📊 数据集**

使用了四个行业领域（抵押、金融、供应链、医疗/健康）共 581 份真实文档，生成 1,356 条字段级问题（共 3,742 键值），以及 1,500 页手工标注的 9 类布局框；

**📈 对比分析**

评估方式：所有系统输出 markdown 通过同一 LLM 解析为 JSON，按字段级和问题级准确率评估；布局按严格和调和 F1 评估；共测 18 系统（商业 API、通用 VLM、开源 OCR）。结果显示 Extend Performance v2 最高（字段级 96.0%，问题级 90.9%），其余系统在 80–90% 之间，Open Source 系统与 Agentic 版商业系统相差 10–15%；成本与延迟呈多点取舍；

**⚠️ 局限性**

局限性包括：①提取 LLM 可能误读 markdown；②延迟测量仅在单页冷缓存下完成，未反映批量吞吐；③成本基于列表价，未考虑企业折扣；④文档来源不均衡；⑤医疗/税务部分采用合成数据；⑥评测者为系统供应商，虽公开适配器但仍可能存在偏见；⑦未对每个系统进行专属调优，使用统一默认 Prompt 可能低估部分模型潜能。

---

## 478. Audio-Oscar: A Multi-Agent System for Complex Audio Scene Generation, Orchestration, and Refinement

**arXiv ID:** 2606.07397 | [PDF](https://arxiv.org/pdf/2606.07397v1)

**作者:** Yifan Duan `[一作]` (Shanghai Jiao Tong University), Xie Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Audio-Oscar多代理框架，生成复杂音频场景文本描述对应的长音频。

**💡 创新点**

通过角色建模、语音设计、细粒度时间线规划、专属模型协同及反馈驱动的后期处理，实现从文本到多声道多音频类型的自动化生成。

**🔧 技术方法**

采用大型语言模型驱动的代理、文本到语音/音频/音乐/歌曲生成模型、语音设计TTS、对齐、音频评审、混音与后制等技术。

**📊 数据集**

构建ASG‑Bench音频场景生成基准（包含带参考音频与纯文本子集），以及使用T2A-Bench、AudioTime进行评测。

**📈 对比分析**

与WavJourney、Any2Speech、AudioX等方法对比，Audio-Oscar在事件覆盖、时序一致性、音质、情境对齐等指标均达到或超过基线，尤其在ASG‑Bench的事件与时间陈述覆盖率接近参考音频。

**⚠️ 局限性**

受限于语音设计的真实性、长音频模型的生成范围、极端声效时长不足、以及评测依赖大型模型等因素。

---

## 479. Tracing Stablecoin Contagion during the USDC Depeg after the Silicon Valley Bank Collapse

**arXiv ID:** 2606.07442 | [PDF](https://arxiv.org/pdf/2606.07442v1)

**作者:** Krongtum Sankaewtong `[一作]` (Kyoto University), Yuichi Ikeda `[通讯]` (Kyoto University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `3855fcda-48ef-4070-a15e-803cd5c84d83` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究SVB崩溃导致USDC脱钩事件在多尺度层面上的链上行为，量化交易同步、传播渠道、账户重分配、时序节奏和财富异质性。

**💡 创新点**

首次将相位同步分析与高分辨率交易图结合，揭示了分化的传播通道（广泛参与 vs 大额转账）以及账户层面的单币到多币转移；并将日内节奏和财富分层纳入系统风险监测框架。

**🔧 技术方法**

使用Hilbert–Huang变换提取交易计数的相位并计算同步度、ARDL回归分析价格与链上指标的关联、构建加权有向交易网络、账户状态转移计数、日内热图分析、财富分层统计。

**📊 数据集**

以2023年3月1日至4月30日的以太坊ERC‑20转账事件（USDC, DAI, FRAX, USDP, USDT, BUSD, WBTC, WETH），以及CoinGecko每日价格和区块链快照。

**📈 对比分析**

通过预/危机/后期窗口比较同步度、ARDL显著性、账户比例变化、日内活跃度等指标；同步度在危机期间显著提升，USDC相关稳定币出现交易量激增，USDT/WBTC/WETH主要通过交易量变化；账户多币比例上升，USDC→USDT流动显著，体现多维度传播；统计显著性表明结果稳健。

**⚠️ 局限性**

仅为观察性关联，缺乏因果证据；仅涵盖以太坊ERC‑20链上活动，未考虑中心化交易所、跨链或离链赎回；地址无地理位置，日内节奏仅能暗示时间区；研究时间窗口有限。

---

## 480. Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings

**arXiv ID:** 2606.07502 | [PDF](https://arxiv.org/pdf/2606.07502v1)

**作者:** Songhao Wu `[一作]` (Renmin University of China), Rui Yan `[通讯]` (Wuhan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EmbFilter，一种基于 LLM 未嵌入矩阵的线性变换，用于过滤掉边缘谱子空间，改善零样本文本嵌入质量，并实现有效的维度压缩。

**💡 创新点**

创新点在于：① 将 LLM 的 unembedding 矩阵视为“特征镜头”，揭示其隐藏的边缘谱子空间导致高频无信息词占主导；② 设计简单的线性过滤器 Φ_τ，只需一次矩阵投影即可去除该子空间，既提升语义表达又天然实现维度约简。

**🔧 技术方法**

技术手段包括 Logit Lens、Logit Spectroscopy 对 unembedding 矩阵的奇异值分解、构造 Φ_τ 过滤器、与 PromptEOL/ECHO 等提示工程结合，并在 MTEB 基准上进行评估。

**📊 数据集**

使用的主要数据集为 MTEB（包含 STS、分类、聚类、检索等多任务），以及 RedPajama 语料库用于近似词频估计，实验覆盖 Qwen、Llama、Mistral 三大 LLM backbone。

**📈 对比分析**

对比方法包括 PromptEOL、ECHO、SimCSE、coCondenser 等；实验显示 EmbFilter 在 MTEB 上平均提升约 6–14% 的整体分数，且即使将维度压缩至 1/8，仍保持或超过基线性能；在维度约简实验中，EmbFilter 使得 512 维嵌入在多项任务上优于原始 4096 维。

**⚠️ 局限性**

局限性：① 仅利用未嵌入矩阵的线性特性，可能无法捕捉更深层语义变化；② 对词频分布估计的依赖可能在不同语料或语言下表现不一致；③ 目前的边缘谱过滤比例 τ 仍需经验选择，未给出全局最优策略；④ 只关注零样本场景，未验证在有监督微调时的兼容性。

---

## 481. Implicit Data Synthesis for Contrastive Unsupervised Data Augmentation

**arXiv ID:** 2606.07498 | [PDF](https://arxiv.org/pdf/2606.07498v1)

**作者:** Patrick Kage `[一作]` (University of Edinburgh), Pavlos Andreadis `[通讯]` (University of Edinburgh)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出通过在网络权重空间添加噪声的方式生成对比学习的正样本（即隐式数据合成 IDS），以替代传统的图像空间增强，适用于物理信号不宜被修改的科学观测数据。

**💡 创新点**

创新点在于：①将对比学习的视图生成从数据空间切换到权重空间；②仅在单个全连接层加入噪声，保持信号结构不被破坏；③在雷达流星观测这一特殊数据集上验证其有效性，并证明在与传统增强方法相同的增强预算下仍能获得较好表示。

**🔧 技术方法**

使用的技术包括：SimCLR 对比学习框架、基于高斯噪声的权重扰动（IDS）、线性探测器与 k‑NN 下游评估、以及在雷达数据上自定义的 FFT 频域掩码增强。实验网络覆盖从小型 CNN 到 ResNet‑18/50 的多种架构。

**📊 数据集**

数据集：①基于物理仿真的合成雷达流星观测图像（512×512，裁剪为多种尺寸）；②CIFAR‑10 图像数据（在实验中将十类合并为两大类以匹配二分类任务）。

**📈 对比分析**

比较方法：与仅使用水平/垂直翻转与旋转的 SimCLR 基线、以及物理意义的 FFT 掩码增强进行对比；在雷达数据上评估线性探测器与 k‑NN 的分类准确率，结果显示：在浅层 CNN 上 IDS 通常匹配或优于基线；在 ResNet 深层模型中 IDS 效果不佳；在 CIFAR‑10 上 IDS 在大多数组合中超过翻转/旋转基线。

**⚠️ 局限性**

局限性：①仅在单一全连接层加入噪声，导致在深层网络（如 ResNet）中扰动不足；②对扰动尺度 s_l 的选择敏感，需手动调参；③仅在合成雷达数据和受限的 CIFAR‑10 基线上验证，缺乏对真实多种科学观测数据的广泛评估；④未探索多层扰动或自适应尺度的改进。

---

## 482. Differences in Detection: Explainability Where it Matters

**arXiv ID:** 2606.07503 | [PDF](https://arxiv.org/pdf/2606.07503v1)

**作者:** Johannes Theodoridis `[一作]` (University of Tübingen), Andreas Schilling `[通讯]` (University of Tübingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Differences in Detection（DnD）方法，能够直接比较两种目标检测模型在实例级别的预测与错误。

**💡 创新点**

创新点在于利用两模型共同匹配的地面真值交集与差集，生成可解释的子集（B、M1、M2、N等），并结合 TIDE 错误类型构建混淆矩阵，从而实现模型间共享与独有错误的直观对比。

**🔧 技术方法**

采用与 mAP 相同的 IoU 匹配算法、TIDE 错误 Oracle、标准的阈值设置，并配合 ODAM 可视化工具来展示模型误差。

**📊 数据集**

使用的主要数据集包括 MS-COCO、COCO-C（鲁棒性基准）、Sama-COCO、COCO-ReM 等，实验中以 MS-COCO 训练并评估模型。

**📈 对比分析**

对比方法：通过计算 B（两模型都识别）、M1/M2（单模型识别）、N（两模型都未识别）等集合，进一步划分错误集合 Ex1/Ex2，生成错误类型混淆矩阵。实验结果表明 ConvNext‑v2 与 ViTDet 在 mAP、mAR、TIDE 上存在差异，而 DnD 能揭示 ViTDet 在某些实例上独有检测，提供更细粒度的性能洞察。

**⚠️ 局限性**

局限性：当前仅支持两模型的直接比较；仅从 GT 视角考虑，未覆盖假阳性误差；多模型情况需链式比较或进一步扩展定义。

---

## 483. Accelerated Decentralized Stochastic Gradient Descent for Strongly Convex Optimization

**arXiv ID:** 2606.07496 | [PDF](https://arxiv.org/pdf/2606.07496v1)

**作者:** Ming Sun `[一作]` (Peking University), Kun Yuan `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

提出了一种名为 MG‑ADSGD 的去中心化随机梯度算法，通过在每一步同时执行 Nesterov 型加速、快速 gossip 平均以及多轮梯度采样，实现了在强凸场景下的通信效率提升。

**💡 创新点**

创新点在于将 gossip 深度与 mini‑batch 大小耦合（B = R），使得更多通信轮既能提高一致性，又能降低梯度方差；同时将 Nesterov 速度法与加速 gossip 结合，首次在随机去中心化优化中实现了 √κ 与 1/√(1‑β) 的双重加速。

**🔧 技术方法**

核心技术包括：Nesterov 型多序列加速结构（X‑Y‑Z）、多轮 Fast Gossip Average（FGA）通信原语、mini‑batch 随机梯度估计以及理论上严格的收敛性分析。

**📊 数据集**

文中未给出任何实际数据集，全部结果基于理论分析与理论极限比较；若需实验验证，需自行在常见分布式学习数据集（如 CIFAR‑10、ImageNet 等）上实现。

**📈 对比分析**

通过与传统 DGD、带动量的 DSGD、以及先前的随机加速方法（如 D‑MASG、LMT）等进行通信复杂度与梯度复杂度的对比，MG‑ADSGD 在通信复杂度上实现了最优的 √(κ/(1‑β)) 量级（加上对数因子），并在梯度复杂度上仅多出一个 log(1/ε) 的轻微开销。

**⚠️ 局限性**

局限性包括：1）理论分析依赖于光滑、强凸、以及梯度方差有界的理想假设；2）额外的 log(1/ε) 乘子会在极低精度需求下稍显昂贵；3）缺乏真实数据集实验验证，实际网络延迟与异步通信环境下的表现尚未评估。

---

## 484. Mitigating Proxy-to-Wild Domain Gap in Deepfake Speech

**arXiv ID:** 2606.07494 | [PDF](https://arxiv.org/pdf/2606.07494v1)

**作者:** Xuanjun Chen `[一作]` (National Taiwan University), Jyh-Shing Roger Jang `[通讯]` (National Taiwan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文针对基于音频编解码器的语音生成（CodecFake）造成的深度伪造检测泛化瓶颈，提出了域偏移特征增强（DSFA）方法，并构建了更具挑战性的CoSG ExtEval评估集，验证了其在各种CodecFake攻击下的鲁棒性。

**💡 创新点**

创新点：①将SSL后训练的反深度伪造特征提取器与DSFA结合，通过把确定性特征统计转化为随机分布并在特征层级上进行AdaIN重构，从而显式模拟“野生”环境下的域偏移；②提出CoSG ExtEval，覆盖更多未见生成模型和长时音频，提供更严苛的泛化评估。

**🔧 技术方法**

技术：后训练的自监督学习（SSL）特征提取器（Wav2Vec2‑Large‑Anti‑Deepfake），DSFA（统计估计 → 采样 → AdaIN重构），联合监督对比损失（SupCon + CE），随机特征层级增强（p 取值 0.25）和统一/高斯噪声。

**📊 数据集**

数据集：CodecFake+（CoRS 作为代理训练集，CoSG 为评估集），CoSG ExtEval（40 个未见生成模型、长时音频），以及原始的 ASVspoof19 LA 等基准集。

**📈 对比分析**

比较方法：与传统 ASVspoof19、CoRS、CoSG Eval 训练的基线模型（仅 CE 或 SupCon）以及使用后训练 SSL 的 Fine‑Tune 进行对比。实验表明，DSFA+SupCon 的 EER 在 CoSG Eval 下降至 2.78%（比基线低约 0.23%），在 CoSG ExtEval 降至 22.77%（比基线低约 1.31%），实现了新的 SOTA 结果。

**⚠️ 局限性**

局限性：①DSFA 在长时音频（CoSG ExtEval）中的效果相对有限，提示长时序特征与语音内容匹配仍是挑战；②过高的增强比例（p≥0.75）会导致性能下降，需精细调节；③目前仅针对单一 SSL 后训练模型，未探索不同后训练策略的组合。

---

## 485. How AI Agents Reshape Knowledge Work: Autonomy, Efficiency, and Scope

**arXiv ID:** 2606.07489 | [PDF](https://arxiv.org/pdf/2606.07489v1)

**作者:** Jeremy Yang `[一作]` (Harvard University), Jerry Ma `[通讯]` (Perplexity)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了 Perplexity 的搜索（对话式助手）与计算机（自主代理）在知识工作中的效率、质量与工作范围的差异。

**💡 创新点**

通过自然实验（匹配相同用户的近似初始查询）证明自主代理能显著提升执行效率、降低人力成本，并促使用户尝试更高阶、更跨专业的任务。

**🔧 技术方法**

利用 LLM 生成工具调用、模型上下文协议（MCP）、以及 O*NET 任务分类等技术，衡量代理的执行量、质量与任务复杂度。

**📊 数据集**

使用 Perplexity 平台 2026 年 2 月至 5 月的搜索与计算机查询日志（约 10,000 对匹配会话、100,000 条计算机查询、8,000 名双产品用户）以及 BLS 2025 年工资数据。

**📈 对比分析**

对比方法为匹配会话自然实验与基准搜索+人工执行的对照；结果显示计算机会话平均机器执行时间为 26 分钟，搜索仅 33 秒，任务完成时间从 269 分钟降至 36 分钟，成本降低 94%，任务复杂度提升 21pp（更高阶认知与更广知识域）。

**⚠️ 局限性**

局限包括：仅观察 90 天的早期采纳期，匹配对仅涵盖能找到近似搜索等价的计算机查询，模型时间与人力估计存在误差，未能观测用户在 Perplexity 之外的工作流程，且 LLM 分类可能引入噪声。

---

## 486. Unsupervised Continual Clustering via Forward-Backward Knowledge Distillation

**arXiv ID:** 2606.07474 | [PDF](https://arxiv.org/pdf/2606.07474v1)

**作者:** Mohammadreza Sadeghi `[一作]` (McGill University), Narges Armanfard `[通讯]` (McGill University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了前向后向知识蒸馏的无监督连续聚类框架FBCC，能够在无标签、无回放的条件下逐任务学习聚类结构并保持历史聚类不被遗忘。

**💡 创新点**

①首个无监督连续聚类方法；②通过双向知识蒸馏将教师网络与轻量化任务专属学生协同学习，既能学习新聚类又能保留旧聚类；③使用学生模型而非完整模型或样本缓存实现内存高效。

**🔧 技术方法**

对比学习与聚类损失、前向/后向蒸馏、原型投影、教师网络ResNet-18、学生网络SqueezeNet、聚类投影器、轻量化学生维护。

**📊 数据集**

CIFAR-10、CIFAR-100、Tiny-ImageNet、ImageNet100 四个计算机视觉基准数据集，按类划分为多任务。

**📈 对比分析**

与监督连续学习方法Co^2L、OCD-Net、以及无监督连续学习方法CCL、STAM、LUMP、CaSSLe、POCON和离线CC进行比较。FBCC在四个数据集上均取得最高平均聚类准确率（如CIFAR-10 75.28%）且平均遗忘率最低（如CIFAR-10 2.29%）。

**⚠️ 局限性**

需要预先知道每个任务的聚类数；学生数量M需调参，平衡内存与性能；对教师网络深度和原型设计有依赖；在极大规模或实时高频任务场景下，计算与存储成本仍是挑战。

---

## 487. Affordance-Based Hierarchical Reinforcement Learning for Quadruped Pedipulation

**arXiv ID:** 2606.07506 | [PDF](https://arxiv.org/pdf/2606.07506v1)

**作者:** Tuba Girgin `[一作]` (Embry Riddle Aeronautical University), Cagri Kilic `[通讯]` (Embry Riddle Aeronautical University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在仿真与真实环境中实现了一套基于三层层级强化学习的全自动四足机器人物体交互系统，该系统通过视觉姿态亲和性模块自动生成可行的机器人基座姿态与物体接触点，导航层控制底盘运动，末端操纵层利用三次样条轨迹完成对物体的腿部（pedipulation）操作；

**💡 创新点**

核心创新包括：①提出可视化姿态亲和性算法自动生成基座姿态与接触点，消除人工轨迹设计；②将导航、行走与踩踏控制解耦成独立层级，预训练模型可即插即用；③公开开放式物体交互数据集，支持后续研究；

**🔧 技术方法**

使用的技术主要有：三层层级强化学习（PPO）、LiDAR点云分割与法线估计、姿态亲和性算法（斜率、目标姿态、接触点计算）、三次样条轨迹、IsaacSim仿真、ONNX推理、ESEKF姿态估计、DBSCAN聚类、OCELOT脚位检测；

**📊 数据集**

实验中使用的“数据集”包括：①仿真中随机生成的斜坡与岩石障碍；②真实环境中不同斜坡混凝土地面与随机放置的假石块；③收集的机器人运动学、力反馈、接触点、轨迹等信息，形成开放式的四足机器人物体交互数据集；

**📈 对比分析**

通过与两种基线对比（直接给定目标姿态、随机接触点），在仿真中使用位移/峰值力的效率指标评估，平均效率最高可提升1.5~2倍；在真实环境中系统成功率达86%，平均位移0.17 m、峰值力33 N、效率0.51，均显著优于基线；

**⚠️ 局限性**

系统的主要局限性为：多模块耦合导致误差在视觉、姿态估计、导航、行走与踩踏层级间累积，对高精度操作敏感；目前仅验证单目标单场景；对未知动态障碍的鲁棒性尚有限。

---

## 488. Sparse Subspace-to-Expert Sharing for Task-Agnostic Continual Learning

**arXiv ID:** 2606.07500 | [PDF](https://arxiv.org/pdf/2606.07500v1)

**作者:** Fatema Siddika `[一作]` (Iowa State University), Ali Jannesari `[通讯]` (Iowa State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种适用于大语言模型的持续学习框架SETA，利用稀疏专家混合来隔离任务特定与共享知识，支持任务无关推理；

**💡 创新点**

创新点在于稀疏子空间分解与Split-on-Share（SoS）机制，动态将参数划分为冻结的唯一专家和可塑性的共享专家，并通过自适应弹性锚定与路由感知正则化实现稳定性与可塑性的平衡；

**🔧 技术方法**

使用稀疏子空间选择、块级梯度重要性评分、两阈值过滤、弹性锚定正则、路由感知正则、单层门控网络等技术；

**📊 数据集**

在LLaMA‑2 7B和Qwen3‑4B上评估，任务包括C‑STANCE、FOMC、MeetingBank、ScienceQA、NumGLUE‑cm、20Minuten六个领域基准；

**📈 对比分析**

与Replay、EWC、GEM、LoRA、I‑LoRA等基线对比，SETA在整体表现、后向迁移和遗忘率上均优于大多数方法；在LLaMA‑2上OP 28.72%/BWT 19.10%，在Qwen3‑4B上OP 43.30%/BWT 15.42%；

**⚠️ 局限性**

局限包括对更大模型与不同架构的可扩展性未知、任务序列长度有限、共享专家池未实现剪枝/整合导致长序列时的可扩展性问题。

---

## 489. Modelling Opinion Dynamics at Scale with Deep MARL

**arXiv ID:** 2606.07487 | [PDF](https://arxiv.org/pdf/2606.07487v1)

**作者:** Lukas Seier `[一作]` (University of Oxford), Jakob Foerster `[通讯]` (University of Oxford)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文开发了一个GPU加速的多智能体强化学习框架，用于模拟多达1000个代理人的共识与真相寻找游戏，并在不同网络结构（蓝天网络、美国国会X/Twitter网络、Hadza狩猎采集网络）上进行实验。

**💡 创新点**

创新点包括：① 将other-play (op)算法扩展到一般总和环境，避免不现实的约定；② 在强化学习框架中加入可调的合规性奖励，平衡真相寻找与社交一致性；③ 通过学习的注意力层从网络拓扑直接重建节点重要性结构；④ 结合JAX实现的GPU加速，实现百万步训练仅需35分钟。

**🔧 技术方法**

技术手段主要是多智能体强化学习（IPPO + OP），使用JAX在单块A100 GPU上进行端到端训练，模型采用注意力网络、GRU、全连接层和策略/信念/价值头；奖励设计包含真相奖励和合规奖励，并使用对称变换以消除初始偏好。

**📊 数据集**

数据集涵盖：① 1000节点的蓝天（Bluesky）机器学习社区网络；② 475节点的美国国会X/Twitter网络；③ 37节点的Hadza狩猎采集者GPS邻接网络，分别用于验证节点重要性预测和对比不同网络的共识效果。

**📈 对比分析**

比较方法包括：① 与人类观察到的节点重要性（pi中心性）对比，使用Wasserstein距离评估误差；② 与Bayes最优猜测的理论上限进行对比；③ 通过对照不同奖励权重α的准确率曲线、社交媒体与狩猎采集网络的比较来评估性能。实验显示，在蓝天网络中α≈0.2时模型最接近人类数据；在狩猎采集网络中有限合规性略优于完全真相追求，而在社交媒体网络中高合规性导致准确率下降并产生不诚实代理。

**⚠️ 局限性**

局限性包括：未考虑推荐系统、有限注意力、影响力寻求行为及代理奖励异质性；模型假设的奖励设计与真实人类心理可能存在差异；在更大或更复杂网络中可能需要进一步的可扩展性和鲁棒性改进。

---

## 490. Network Recovery from Cascade Data: A Debiased Jacobian-Based Machine Learning Approach

**arXiv ID:** 2606.07483 | [PDF](https://arxiv.org/pdf/2606.07483v1)

**作者:** Lei Huang `[一作]` `[通讯]`, Lei Huang

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了一种名为 CascadeNet 的框架，用于从观察到的级联序列中恢复隐藏的影响网络，并在此过程中给出每条边的置信区间与显著性检验。

**💡 创新点**

创新点在于：①将网络结构表征为一阶转移函数的雅可比矩阵，消除对具体扩散模型的假设；②利用 Riesz 表示器实现 Neyman 正交去偏估计，使得在使用非参数机器学习估计转移函数时仍能获得 −−n 一致且可做统计推断；③在高维、非线性场景下提供全局网络的可解释推断。

**🔧 技术方法**

技术核心包括：机器学习（线性索引模型、GNN 等）对转移函数的灵活拟合；自动微分求雅可比；Riesz 代表子与交叉拟合相结合的双重机器学习框架；基于中心极限定理的异步样本推断；以及针对边权的置信区间与假设检验。

**📊 数据集**

使用了两类数据集：一是 9 种合成扩散模型（IC、LT、Exp-SI、PL-SI、SIS、SIR、Complex、Hawkes、非线性 DGP）用于验证；二是西班牙 52 省 COVID‑19 日新增病例与省际流动数据，用于实际应用评估。

**📈 对比分析**

通过与传统基线（Pairwise、NetInf、NetRate、DANI、LTMLE、Granger 等）在合成数据上的 Pearson 相关、精确率-召回曲线和 AUC‑PR 进行比较。CascadeNet 在所有 9 种 DGP 中均取得最高相关（最低 0.54），并在 COVID‑19 数据中与流动网络的相关性显著（r≈0.14），精确率、召回及 AUC‑PR 均优于其他方法。

**⚠️ 局限性**

局限性包括：①理论假设级联轨迹相互独立，实际应用中轨迹可能相关；②对单条级联或极少级联时的性能未充分验证；③需要足够的轨迹数才能保证 −−n 收敛；④目前仅适用于 Markovian 过程，对更复杂的长记忆或非 Markov 传播机制的推广尚待研究。

---

## 491. Physiologically Constrained Musculoskeletal Neural Network for Multi-DoF Joint Kinematics Estimation from Partially Observed sEMG

**arXiv ID:** 2606.07476 | [PDF](https://arxiv.org/pdf/2606.07476v1)

**作者:** Wending Heng `[一作]` (University of Manchester), Zhenhong Li `[通讯]` (University of Manchester)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e15e3743-5ee0-4d5f-813d-d146868082fc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f`

**🎯 论文内容**

构建了一种物理嵌入式神经网络MSK‑NN，用于在部分sEMG观测下估计多自由度关节运动学并推断未观测肌肉激活。

**💡 创新点**

创新点在于同时利用CNN激活估计器和可微分Hill肌肉‑肌腱动力学模块，并结合运动学误差、肌群协同与解剖趋势的复合物理‑生理损失，实现了高精度且生理合理的中间变量输出。

**🔧 技术方法**

采用一维CNN编码器、Hill模型前向动力学、非负矩阵分解得到的肌群协同基底以及三项损失（运动学、协同、趋势）进行联合训练。

**📊 数据集**

使用六名健康受试者的手腕运动数据，包括四种连续运动（顺/逆方向、∞形、随机）和对应的sEMG与运动捕捉标记，sEMG采样率2000 Hz、运动捕捉250 Hz。

**📈 对比分析**

与CNN、Bi‑LSTM、CNN‑LSTM、PET等基线对比，MSK‑NN在NRMSE上显著下降、R²提升至0.90+，尤其在随机运动上保持高性能。

**⚠️ 局限性**

局限性在于仅验证了手腕二自由度运动、需大量标记数据初始化参数、对深层肌肉仍依赖解剖学先验且未全面评估实时部署下的功耗与鲁棒性。

---

## 492. Agentopia: Long-Term Life Simulation and Learning in Agent Societies

**arXiv ID:** 2606.07513 | [PDF](https://arxiv.org/pdf/2606.07513v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 493. Graph Neural Network leveraging Higher-order Class Label Connectivity for Heterophilous Graphs

**arXiv ID:** 2606.07475 | [PDF](https://arxiv.org/pdf/2606.07475v1)

**作者:** Takuto Takahashi `[一作]` (University of Osaka), Makoto Onizuka `[通讯]` (University of Osaka)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在异质性有向图中提出Label Context Classifier (LCC)，通过四种标签行走捕捉高阶类别标签连通性，并与任意GNN自适应融合。

**💡 创新点**

创新点在于利用四种互斥标签行走（前向、后向、兄弟、监护）生成标签上下文嵌入，克服传统GNN无法捕获高阶连通性的缺陷；并提出无需额外训练即可自适应学习LCC与GNN重要性的融合方法。

**🔧 技术方法**

采用深度优先/逆向搜索生成标签行走，使用基于word2vec思路的两层MLP学习标签上下文嵌入；LCC结合节点特征与多种行走的嵌入进行分类；最后使用验证集损失权重融合LCC与GNN预测。

**📊 数据集**

在七个异质性有向图数据集上实验：Cornell、Texas、Wisconsin、Chameleon、Squirrel、Roman-Empire、Amazon-Ratings。

**📈 对比分析**

与GCN、GAT以及H2GCN、LINKX、GloGNN等SOTA异质性GNN比较，GNN+LCC在所有数据集均取得最高准确率，提升幅度多达10%+，仅有少量例外。

**⚠️ 局限性**

融合权重机制在某些数据集上出现极端偏重导致性能未提升；LCC仅利用节点标签信息，未考虑特征与结构的更深层交互；目前只针对有向图，需验证在无向图或其他异构网络的适用性。

---

## 494. MemDreamer: Decoupling Perception and Reasoning for Long Video Understanding via Hierarchical Graph Memory and Agentic Retrieval Mechanism

**arXiv ID:** 2606.07512 | [PDF](https://arxiv.org/pdf/2606.07512v1)

**作者:** Cong Chen `[一作]` (Ant Group), Chunhua Shen `[通讯]` (Ant Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了将长视频理解拆分为感知与推理两步的框架，利用流式感知构建分层图记忆，并通过工具增强的代理检索进行高效推理。

**💡 创新点**

创新点包括：① 分层图记忆结构将视频从全局到局部逐层抽象；② Observation‑Reason‑Action 循环的代理检索机制，结合层级导航、精确搜索和图遍历工具；③ 证明代理推理能力与长视频性能呈正相关，提出代理能力可扩展的新范式。

**🔧 技术方法**

采用流式自适应分段、视频中心图本体（空间属性、主体‑对象、时间因果边）、工具银行（层级导航、精确搜索、图遍历）、多步代理循环，以及 Gemini‑3.1‑Pro/ Gemini‑2.5‑Pro/ Qwen3‑VL‑235B‑A22B 等大语言模型。

**📊 数据集**

在 LVBench、LongVideoBench、Video‑MME 与 EgoSchema 四个长视频问答基准上进行评估。

**📈 对比分析**

与传统端到端长视频 VLM 与基于记忆的视频 LLM 进行对比，实验表明在 LVBench 达到 90.7 分（提升 12.5 点），LongVideoBench 92.9 分（提升 14.3 点），Video‑MME 92.1 分（提升 11.8 点），EgoSchema 88.2 分；仅使用 2% 的上下文窗口，整体精度提升 12.5 点，已接近人类专家。

**⚠️ 局限性**

局限性包括：对感知模型在短片段上的精度依赖较大；代理检索需要预先设计工具集合，可能不易迁移；在极长视频或极细粒度推理时可能需要更多检索步骤；评估仅覆盖四个基准，跨域泛化需进一步验证；代理预算、检索 k 值等超参数仍需优化。

---

## 495. CoMetaPNS: Continually Meta-learning Personalized Neural Surrogates for Cardiac Electrophysiology Simulations

**arXiv ID:** 2606.07488 | [PDF](https://arxiv.org/pdf/2606.07488v1)

**作者:** Ryan Missel `[一作]` (Rochester Institute of Technology), Linwei Wang `[通讯]` (Rochester Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出并实现了连续学习元学习框架 CoMetaPNS，用于快速生成个体化心脏电生理（EP）仿真神经代理。

**💡 创新点**

创新点在于：①结合 feed‑forward 元推理与贝叶斯高斯混合模型对任务进行识别；②使用持续经验回放与聚类动态更新任务关系；③实现无重训练、低延迟的个性化与灾难性遗忘抑制。

**🔧 技术方法**

技术细节包括：GCNN + GRU 的时空生成模型、上下文编码网络、基于采样的经验回放、持续贝叶斯 GMM 任务聚类、Meta‑PNS 的无梯度元学习策略。

**📊 数据集**

数据集：①合成心脏网格（3 种心形、12 种病灶配置，12 任务）用于训练与评估；②真实动物模型（4 个刺激点、20 个样本）用于跨域泛化验证。

**📈 对比分析**

对比方法：MetaPNS（静态元学习）、MAML‑PNS（梯度元学习）、PNS（无元学习）、传统贝叶斯优化（FS‑BO、VAE‑BO）。实验表明 CoMetaPNS 在 MSE、SCC、DC 上均优于所有基线，训练时间仅为 MAML 的 30–40%，适配时间显著缩短；灾难性遗忘率明显下降。

**⚠️ 局限性**

局限性：在真实临床数据中的泛化性能仍有限；仅针对去极化阶段；贝叶斯 GMM 聚类纯度不完美，对极少见任务易遗忘；对更大规模心电数据或不同网格结构的适用性需进一步验证。

---

## 496. Drifting Models for Surrogate Flow Modeling

**arXiv ID:** 2606.07481 | [PDF](https://arxiv.org/pdf/2606.07481v1)

**作者:** Chris R. Jung `[一作]` (Heilbronn University of Applied Sciences), Nicolaj C. Stache `[通讯]` (Heilbronn University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于漂移模型的单步生成式CFD替代方法，用于预测室内2D稳态空气流场。

**💡 创新点**

创新点包括：①将漂移框架迁移到流体模拟；②在VAE潜空间中进行漂移；③采用标签感知正样本遮罩以保持条件一致；④引入空间编码以实现对未见几何体的泛化。

**🔧 技术方法**

使用的技术包括：条件漂移模型（Diffusion Transformer骨干）、卷积VAE、标签/空间编码器、正样本遮罩、单步推理。

**📊 数据集**

数据集为2025个2D稳态CFD仿真，涵盖空房间和两种障碍物配置，共计675种入口/出口/速度组合，网格尺寸1.5 m×1.5 m，结果投影到64×64网格。

**📈 对比分析**

与1000步DDIM潜在扩散基线对比，采用nRMSE、R²、余弦相似度以及流结构一致性（vorticity、divergence）等指标；漂移模型在标签条件下nRMSE≈0.068、R²≈0.802，速度提升约两阶（≈6 ms vs 1870 ms）。

**⚠️ 局限性**

限制包括：仅验证2D稳态、VAE压缩导致细节失真、空间编码泛化性能不佳、数据稀疏限制漂移场训练、未测试瞬态3D流动和多状态数据。

---

## 497. Supervision versus Demonstration-Based In-Context Learning for Multiword Expression Classification

**arXiv ID:** 2606.07479 | [PDF](https://arxiv.org/pdf/2606.07479v1)

**作者:** Sercan Karakaş `[一作]` (University of Chicago), Yusuf Şimşek `[通讯]` (Fırat University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个针对土耳其语轻动词结构（LVC）的二分类数据集（147句），并对比了监督式 BERTurk 编码器与三种指令微调大型语言模型（Llama 3.1 8B、GPT‑OSS‑20B、Qwen 2.5‑14B）在零样本、单样本和多样本提示下的 LVC 检测性能。

**💡 创新点**

提出用匹配负样本（同义词动词的字面用法）和无关负样本对 LVC 进行“字面‑习语”对比的评估框架，并展示提示对模型决策边界的显著影响，揭示在无监督提示下 LLM 对 LVC 的低召回率以及通过少量示例即可实现的精细校准。

**🔧 技术方法**

技术包括：Turkish BERTurk 预训练编码器加二分类头；三种指令微调 LLM（Llama 3.1、GPT‑OSS‑20B、Qwen 2.5）；零样本、一次样本、一少样本提示策略；统计分析（χ²、Cramer's V、Holm 校正）和误差分析（FP/FN、Precision/Recall）。

**📊 数据集**

数据集：从 9 个土耳其语 UD treebank（共 82k 句）自动提取 LVC 候选后人工校正得到 9,491 条 LVC；实验评估使用手工构造的 147 条句子，分为 LVC、NLVC（字面控制）和 Random（随机负样本）三类。

**📈 对比分析**

比较方法：对每个模型在每个提示层次下分别计算每类的准确率，汇总总体准确率，并与监督式 BERTurk baseline 进行对比。结果显示：零样本 LLM 对 LVC 的召回率极低（0–6%），但在多样本提示下（尤其 GPT‑OSS‑20B）可达 84–86% 的整体准确率；BERTurk baseline 在 LVC 上达到 67–80% 的准确率，整体 82–86%。提示敏感性和模型特异性导致不同模型在正负类上的偏差差异显著。

**⚠️ 局限性**

局限性：评估样本规模仅 147 条，且为精心设计的对照集，无法覆盖土耳其语所有 MWEs；提示设计与示例选择对 LLM 结果影响大，结果不一定代表模型固有能力；BERTurk 与 LLM 的对比不完全公平（前者受监督训练，后者仅靠提示）。

---

## 498. UniSHARP: Universal Sharp Monocular View Synthesis

**arXiv ID:** 2606.07514 | [PDF](https://arxiv.org/pdf/2606.07514v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 499. How reliable are LLMs when it comes to playing dice?

**arXiv ID:** 2606.07515 | [PDF](https://arxiv.org/pdf/2606.07515v1)

**作者:** Luca Avena `[一作]` (University of Florence), Bernardo Busoni `[通讯]` (University of Florence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在离散概率问题上的推理能力，构建标准与反直觉两组题库，评估16个模型及其 Chain-of-Thought（CoT）模式，并探究 token bias 与 sycophancy 对其性能的影响。

**💡 创新点**

创新点在于：①系统结合标准与反直觉概率练习，首次量化 CoT 在处理直觉陷阱时的提升；②通过对同一概率结构的表述变化揭示 token bias，显示模型在表面语义改变时的鲁棒性差；③引入误导性提示（sycophancy）实验，发现模型对模型生成的错误推理尤为易受影响。

**🔧 技术方法**

使用的技术包括：Chain-of-Thought 生成、JSON 结构化输出与自动评分、对抗式 prompt 设计、token-level 表述改写、基于 Gemini 3.1 的自动判分与人工复核相结合。

**📊 数据集**

使用的数据集：20道反直觉离散概率题（来自经典悖论及作者构造），50道标准离散概率题（取自大学教材）。

**📈 对比分析**

比较方法：对每个模型/配置分别计算标准集与反直觉集的平均准确率；标准集平均准确率为 0.96，反直觉集为 0.59；token bias 使性能下降约 20%；误导性提示导致平均相对下降 34%；CoT 在反直觉集表现更好，但未在 sycophancy 实验中提供保护。

**⚠️ 局限性**

局限性：①实验仅覆盖离散概率，未验证连续或更复杂概率推理；②token bias 与误导提示的影响可能因训练数据差异而异，结果不一定普适；③CoT 的好处在不同任务中不一致，需进一步探究其机制；④模型对提示的敏感性和对错误推理的易受性在实际应用中仍需谨慎对待。

---

## 500. Streaming Video Generation with Streaming Force Control

**arXiv ID:** 2606.07508 | [PDF](https://arxiv.org/pdf/2606.07508v1)

**作者:** Hanhui Wang `[一作]` (Northeastern University), Huaizu Jiang `[通讯]` (Northeastern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种流式视频生成框架，利用连续力输入实现从单张图像到可控、物理合理视频的实时生成。

**💡 创新点**

创新点包括：统一全局与局部力的像素掩码力表示；采用力感知蒸馏把双向教师模型转化为自回归学生；构造时变力训练集，使模型能在推理过程中实时响应力变化。

**🔧 技术方法**

技术方法主要是基于Wan2.2 5B的双向教师+自回归学生架构；ControlNet控制分支；ODE初始化+Self-Forcing蒸馏；多模态力表示和力感知控制网络。

**📊 数据集**

使用的数据集包括：30k Blender合成力视频（含全局与局部力）；30k时变力视频；90k来自Pexels的图像-力对；Physics-IQ真实视频数据用于评测。

**📈 对比分析**

与Wan2.2 TI2V、Force-Prompting、Kling Motion Brush等基线进行人评测和Physics-IQ指标对比，模型在力遵从度、物理一致性和视觉质量上均取得最高分，尤其在力变更场景表现最优。

**⚠️ 局限性**

局限性在于对极端或非线性物理行为的建模仍不够精确，对极大力度场或复杂材质的泛化有限；训练成本高，需大量合成与真实数据支持。

---

## 501. Second-Order Path Kernel Interpolation Formulas in Machine Learning

**arXiv ID:** 2606.07495 | [PDF](https://arxiv.org/pdf/2606.07495v1)

**作者:** Jin Guo `[一作]` (City University of Hong Kong), Jean-Michel Morel `[通讯]` (Lingnan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了二阶改进的路径核插值公式，描述梯度下降、随机梯度下降及带动量的随机梯度下降训练后神经网络预测的高阶几何效应；

**💡 创新点**

创新点包括：①在一阶路径核的基础上加入曲率加权核，揭示损失曲率对训练样本贡献的重加权；②引入采样噪声引起的二阶采样诱导核，说明批量大小与梯度噪声协方差如何影响期望预测；③在动量方法下推导出记忆权重的时传递路径核和累计曲率项；④给出终端预测的集中度估计，量化学习率、批量大小与曲率对预测波动的控制；

**🔧 技术方法**

使用连续时间改进方程（modified equation）和SDE近似，弱近似理论，Malliavin微分，梯度核、路径核、曲率核的定义与积分展开；

**📊 数据集**

实验数据集主要为：一维正弦回归（合成数据）、二月亮分类（合成）和MNIST（下采样至50张/类），并在不同批量大小、学习率、曲率基批次选择等设置下进行验证；

**📈 对比分析**

通过对比不同学习率、批量大小下的残差标度、标准差、预测波动，展示二阶公式对残差的解释力和采样噪声对波动的控制效果；实验表明残差按O(η)与O(η²)缩放，批量越大预测越稳定，曲率导向的批次选择能降低预测灵敏度；

**⚠️ 局限性**

局限性包括：对模型与损失连续可微性要求严格（不适用于ReLU等非平滑激活）；需要高阶导数和矩阵运算，理论推导复杂；理论假设在高维深度网络的实际实现中难以完全满足；实验仅在小规模合成/简化任务上验证，缺乏大规模真实任务的进一步评估；

---

## 502. Bradley-Terry Rankings for Recommender Systems Across Dataset Taxonomies

**arXiv ID:** 2606.07492 | [PDF](https://arxiv.org/pdf/2606.07492v1)

**作者:** Ekaterina Grishina `[一作]` (HSE University), Anton Lysenko `[通讯]` (HSE University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对14种推荐算法在89个预处理数据集上的评估结果进行聚合，采用Bradley-Terry模型构建算法排名，并提出可转三元组指标衡量排名一致性与鲁棒性；进一步引入基于数据集特征的covariate-调整BT和BT树，实现对未见数据集的排名预测。

**💡 创新点**

1) 将Bradley-Terry概率模型应用于推荐系统评估，解决传统指标平均聚合的局限；2) 设计可转三元组指标，用于量化排名在缺失数据下的稳定性；3) 构造基于数据集属性的BT树与covariate-调整BT，实现无需模型运行即可给出强基线建议。

**🔧 技术方法**

Bradley-Terry概率排名、Bayesian BT、Rank Centrality、Plackett-Luce、可转三元组指标、covariate-调整BT、BT树、Optuna超参搜索、NDCG@10等评价指标。

**📊 数据集**

来自APS基准集的89个推荐数据集，包含稀疏、顺序、用户-物品比例等多维度特征；部分附加数据集用于外部验证。

**📈 对比分析**

通过计算排名的可转三元组比例和Kendall's τ，将BT排名与简单平均/总和方法对比；BT在缺失比例升高时保持约0.6的可转三元组比例，表现出更高的稳健性；在不同数据集类别下，BT排名差异显著，表明其对数据特征敏感；对未见数据集的预测实验中，BT与covariate-BT均能将top-1 hit提升至0.24–0.28，top-5 overlap约3.3，优于Mean聚合。

**⚠️ 局限性**

BT模型仅基于胜负计数，未考虑差距大小；tie的处理仍有限；在极少比较或缺失数据时仍可能产生不确定性；covariate-调整BT对样本量要求高，存在过拟合风险；实验主要集中在已预处理的基准集，未检验在极端稀疏或大规模场景下的通用性。

---

## 503. Twelve quick tips for designing AI-driven HPC workflows

**arXiv ID:** 2606.07491 | [PDF](https://arxiv.org/pdf/2606.07491v1)

**作者:** Jamie J. Alnasir `[一作]` `[通讯]` (Royal Holloway University of London), Jamie J. Alnasir (Royal Holloway University of London)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054`

**🎯 论文内容**

提出了12条实用提示，指导如何在高性能计算（HPC）环境中设计、部署、优化和复现 AI 驱动的工作流。

**💡 创新点**

创新点在于将 AI 视为工作流的核心而非附加步骤，系统性强调数据重心、异构资源匹配、工作流级并行、容器化、实验追踪、动态反馈循环和吞吐量优化，并对适配现有 HPC 调度器和工作流引擎给出明确建议。

**🔧 技术方法**

主要技术包括：HPC 调度器（如 Slurm）和作业数组；GPU/CPU/高内存节点的异构资源调度；分布式训练框架；容器技术（Apptainer/Singularity）；多种工作流引擎（Nextflow、Snakemake、CWL、Parsl、Pegasus、FireWorks、Dask、Ray）；高效数据格式（HDF5、ADIOS2）；实验管理工具（如 MLflow）。

**📊 数据集**

文中未给出具体数据集，强调在计算生物学、组学、模拟、图像等大规模科学数据领域常见的高维度、海量数据集，可视为通用大规模科学数据。

**📈 对比分析**

论文以经验性建议为主，没有量化实验或性能比较；建议读者通过吞吐量（单位时间内完成的有效任务数）、资源利用率、作业等待时间、复现性指标等指标评估工作流改进效果。

**⚠️ 局限性**

局限性：缺乏系统化的实验验证和数值对比；所给建议在不同 HPC 体系结构、调度器版本或工作流引擎能力下可能需要进一步调整；部分技术实现（如动态工作流生成、跨平台容器化）在资源受限或安全限制的集群中可能受限。

---

